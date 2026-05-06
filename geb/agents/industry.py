"""This module contains the Industry agent class for simulating industrial water demand in the GEB model."""

import calendar
import datetime
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import numpy as np
import xarray as xr
from scipy.ndimage import value_indices

from geb.geb_types import ArrayFloat32, ArrayInt32, ArrayInt64
from geb.hydrology.HRUs import load_water_demand_xr
from geb.store import Bucket

from .general import AgentBaseClass

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel


class IndustryVariables(Bucket):
    """Variables for the Industry agent."""

    current_water_demand: ArrayFloat32
    current_return_flow: ArrayFloat32
    last_water_demand_update: datetime.datetime


class Industry(AgentBaseClass):
    """This class is used to simulate industry.

    Note:
        Currently, this module is not actually agent-based but rather
        uses aggregated pre-defined water demand data.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    var: IndustryVariables

    def __init__(self, model: GEBModel, agents: Agents) -> None:
        """Initialize the Industry agent module.

        Args:
            model: The GEB model.
            agents: The class that includes all agent types (allowing easier communication between agents).
        """
        super().__init__(model)

        if self.model.simulate_hydrology:
            self.HRU = model.hydrology.HRU
            self.grid = model.hydrology.grid

        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["industry"]
            if "industry" in self.model.config["agent_settings"]
            else {}
        )

        self.industry_water_consumption_ds: xr.Dataset = load_water_demand_xr(
            self.model.files["other"]["water_demand/industry_water_consumption"]
        )
        self.industry_water_demand_ds: xr.Dataset = load_water_demand_xr(
            self.model.files["other"]["water_demand/industry_water_demand"]
        )

        if self.model.simulate_hydrology:
            self.abstraction_areas, self.abstraction_river_ids = (
                self.create_abstraction_areas()
            )
        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        """The name of the module.

        Used to save data to disk.

        Returns:
            The name of the module.
        """
        return "agents.industry"

    def spinup(self) -> None:
        """Set initial water demand and return flow during spinup."""
        water_demand, water_return_flow = self.update_water_demand()
        self.var.current_water_demand = water_demand
        self.var.current_return_flow = water_return_flow

    def create_abstraction_areas(
        self, minimum_shreve_stream_order: int = 4
    ) -> tuple[dict[int, ArrayInt32], dict[int, ArrayInt32]]:
        """Create abstraction areas for industry based on the river network.

        Abstraction from industry is assumed to be from larger rivers. If we
        let industry abstract from each grid cell that has any industry, the
        industrial users abstract water from very small rivers, which also
        leads to very high groundwater abstraction in those cells because
        the demand is not satisfiable from the river. This is highly unrealistic.

        Therefore, we define abstraction areas based on the river network. Each abstraction
        area is associated with a river of shreve stream order above a set threshold.

        All water demands from industry are essentially transferred downstream
        to the river of the abstraction area, and abstraction is assumed to occur from that river.

        Note:
            When the model is run in an area without a river of the minimum shreve stream order,
            no abstraction is assumed from industry within the study area.

        Args:
            minimum_shreve_stream_order: The minimum shreve stream order of rivers that can be abstraction rivers.

        Returns:
            A tuple containing:
            - A dictionary mapping abstraction area IDs to linear indices of grid cells in those areas.
            - A dictionary mapping abstraction area IDs to linear indices of river cells in the
                associated abstraction river.
        """
        basin_ids: ArrayInt32 = self.model.hydrology.grid.load2d(
            self.model.files["grid"]["routing/basin_ids"]
        )
        linear_idx_per_basin_id: dict[int, tuple[ArrayInt64]] = value_indices(
            basin_ids, ignore_value=-1
        )
        linear_idx_per_abstraction_area: dict[int, list[tuple[ArrayInt64]]] = {}

        rivers: gpd.GeoDataFrame = self.model.hydrology.routing.active_rivers.copy()
        for river_idx, river in rivers.iterrows():
            assert isinstance(river_idx, int)
            abstraction_river = river
            while (
                abstraction_river["shreve_stream_order"] < minimum_shreve_stream_order
                or not abstraction_river["represented_in_grid"]
            ):
                downstream_idx = abstraction_river["downstream_ID"]
                try:
                    abstraction_river = rivers.loc[downstream_idx]
                except KeyError:
                    abstraction_river = None
                    break

            if abstraction_river is not None:
                abstraction_river_id: int = abstraction_river.name  # ty:ignore[invalid-assignment]
                if abstraction_river.name not in linear_idx_per_abstraction_area:
                    linear_idx_per_abstraction_area[abstraction_river_id] = []
                linear_idx_per_abstraction_area[abstraction_river_id].append(
                    linear_idx_per_basin_id[river_idx]
                )

        linear_idx_per_abstraction_area: dict[int, ArrayInt32] = {
            abstraction_river_id: np.concatenate([xy[0] for xy in xy_list]).astype(
                np.int32
            )
            for abstraction_river_id, xy_list in linear_idx_per_abstraction_area.items()
        }
        linear_idx_per_abstraction_river_id: dict[int, tuple[ArrayInt64]] = (
            value_indices(self.model.hydrology.routing.river_ids, ignore_value=-1)
        )
        linear_idx_per_abstraction_river_id: dict[int, ArrayInt32] = {
            river_id: linear_idx_per_abstraction_river_id[river_id][0].astype(np.int32)
            for river_id in linear_idx_per_abstraction_area.keys()
        }

        return linear_idx_per_abstraction_area, linear_idx_per_abstraction_river_id

    def update_water_demand(self) -> tuple[ArrayFloat32, ArrayFloat32]:
        """Update the water demand for industry at the grid level.

        Returns:
            A tuple containing:
            - The updated water demand (m3/day).
            - The updated return flow (m3/day).
        """
        days_in_year: Literal[366, 365] = (
            366 if calendar.isleap(self.model.current_time.year) else 365
        )

        # Read water demand from source grid and convert from million m3/year to m3/day
        water_demand: xr.DataArray = (
            self.industry_water_demand_ds.sel(
                time=self.model.current_time,
                method="ffill",
                tolerance="366D",  # ty:ignore[invalid-argument-type]
            ).industry_water_demand
            * 1_000_000
            / days_in_year
        )
        # Reproject to model grid and correct for change in cell size
        water_demand: xr.DataArray = (
            water_demand.rio.write_crs(4326).rio.reproject(
                4326,
                shape=self.grid.shape,
                transform=self.grid.transform,
            )
            / (water_demand.rio.transform().a / self.grid.transform.a) ** 2
        )  # correct for change in cell size
        # Convert to linear and compressed model grid
        water_demand: ArrayFloat32 = self.grid.compress(water_demand.values)

        # Read water consumption from source grid and convert from million m3/year to m3/day
        water_consumption: xr.DataArray = (
            self.industry_water_consumption_ds.sel(
                time=self.model.current_time, method="ffill"
            ).industry_water_consumption
            * 1_000_000
            / days_in_year
        )
        # Reproject to model grid and correct for change in cell size
        water_consumption: xr.DataArray = (
            water_consumption.rio.write_crs(4326).rio.reproject(
                4326,
                shape=self.grid.shape,
                transform=self.grid.transform,
            )
            / (water_consumption.rio.transform().a / self.grid.transform.a) ** 2
        )
        # Convert to linear and compressed model grid
        water_consumption: ArrayFloat32 = self.grid.compress(water_consumption.values)

        # Initialize arrays to hold water demand and return flow assigned to rivers
        water_demand_assigned_to_rivers: ArrayFloat32 = self.grid.full_compressed(
            fill_value=0.0, dtype=np.float32
        )
        return_flow_assigned_to_rivers: ArrayFloat32 = self.grid.full_compressed(
            fill_value=0.0, dtype=np.float32
        )

        # Loop through abstraction areas and assign water demand and return flow to associated abstraction rivers
        for (
            abstraction_area_id,
            abstraction_area_linear_indices,
        ) in self.abstraction_areas.items():
            water_demand_in_abstraction_area = water_demand[
                abstraction_area_linear_indices
            ].sum()
            water_consumption_in_abstraction_area = water_consumption[
                abstraction_area_linear_indices
            ].sum()
            return_flow_in_abstraction_area = (
                water_demand_in_abstraction_area - water_consumption_in_abstraction_area
            )

            water_demand_assigned_to_rivers[
                self.abstraction_river_ids[abstraction_area_id]
            ] = (
                water_demand_in_abstraction_area
                / self.abstraction_river_ids[abstraction_area_id].size
            )

            return_flow_assigned_to_rivers[
                self.abstraction_river_ids[abstraction_area_id]
            ] = (
                return_flow_in_abstraction_area
                / self.abstraction_river_ids[abstraction_area_id].size
            )

        self.var.last_water_demand_update = self.model.current_time
        return water_demand_assigned_to_rivers, return_flow_assigned_to_rivers

    def water_demand(self) -> tuple[ArrayFloat32, ArrayFloat32]:
        """Get the current water demand for industry at the HRU level.

        Updates the water demand only if data for this timestep is available.
        Otherwise, assumes the last known water demand.

        Returns:
            A tuple containing:
            - The current water demand (m/day).
            - The current return flow (m/day).
        """
        if (
            np.datetime64(self.model.current_time, "ns")
            in self.industry_water_consumption_ds.time
        ):
            water_demand, current_return_flow = self.update_water_demand()
            self.var.current_water_demand = water_demand
            self.var.current_return_flow = current_return_flow

        assert (
            self.model.current_time - self.var.last_water_demand_update
        ).days < 366, (
            "Water demand has not been updated for over a year. "
            "Please check the industry water demand datasets."
        )
        return self.var.current_water_demand, self.var.current_return_flow

    def step(self) -> None:
        """This function is run each timestep."""
        self.report(locals())
