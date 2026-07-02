"""This module contains the Industry agent class for simulating industrial water demand in the GEB model."""

import calendar
import datetime
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import numpy as np
import xarray as xr
from scipy.ndimage import value_indices

from geb.geb_types import (
    ArrayFloat32,
    ArrayInt32,
    ArrayInt64,
    TwoDArrayInt32,
)
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
    abstraction_area_indices: TwoDArrayInt32
    abstraction_river_indices: TwoDArrayInt32


def create_abstraction_areas(
    basin_ids: ArrayInt32,
    river_ids: ArrayInt32,
    rivers: gpd.GeoDataFrame,
    minimum_shreve_stream_order: int = 6,
) -> tuple[TwoDArrayInt32, TwoDArrayInt32]:
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
        basin_ids: The basin IDs for each grid cell.
        river_ids: The river IDs for each grid cell.
        rivers: The active rivers in the model.
        minimum_shreve_stream_order: The minimum shreve stream order of rivers that can be abstraction rivers.

    Returns:
        A tuple containing:
        - A 2D array mapping abstraction area IDs to linear indices of grid cells in those areas.
        - A 2D array mapping abstraction area IDs to linear indices of river cells in the
            associated abstraction river.
    """
    linear_idx_per_basin_id: dict[int, tuple[ArrayInt64]] = value_indices(
        basin_ids, ignore_value=-1
    )
    abstraction_area_ids_per_river_id: dict[int, int] = {}
    abstraction_river_ids_by_area_id: list[int] = []
    linear_idx_per_abstraction_area_id: list[list[tuple[ArrayInt64]]] = []

    for river_idx, river in rivers.iterrows():
        if not river["represented_in_grid"]:
            continue
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

        if abstraction_river is None or river_idx not in linear_idx_per_basin_id:
            continue

        abstraction_river_id: int = abstraction_river.name  # ty:ignore[invalid-assignment]
        if abstraction_river_id not in abstraction_area_ids_per_river_id:
            abstraction_area_ids_per_river_id[abstraction_river_id] = len(
                abstraction_river_ids_by_area_id
            )
            abstraction_river_ids_by_area_id.append(abstraction_river_id)
            linear_idx_per_abstraction_area_id.append([])

        abstraction_area_id = abstraction_area_ids_per_river_id[abstraction_river_id]
        linear_idx_per_abstraction_area_id[abstraction_area_id].append(
            linear_idx_per_basin_id[river_idx]
        )

    # Convert to 2D arrays with -1 padding
    if not abstraction_river_ids_by_area_id:
        return (
            np.zeros((0, 0), dtype=np.int32),
            np.zeros((0, 0), dtype=np.int32),
        )

    # Abstraction areas (source cells)
    area_indices_list = [
        np.concatenate([xy[0] for xy in xy_list]).astype(np.int32)
        for xy_list in linear_idx_per_abstraction_area_id
    ]
    max_area_size = max(len(indices) for indices in area_indices_list)
    abstraction_area_indices = np.full(
        (len(area_indices_list), max_area_size), -1, dtype=np.int32
    )
    for i, indices in enumerate(area_indices_list):
        abstraction_area_indices[i, : len(indices)] = indices

    # Abstraction river cells (target cells)
    linear_idx_per_river_id: dict[int, tuple[ArrayInt64]] = value_indices(
        river_ids, ignore_value=-1
    )
    river_indices_list = [
        linear_idx_per_river_id[river_id][0].astype(np.int32)
        for river_id in abstraction_river_ids_by_area_id
    ]
    max_river_size = max(len(indices) for indices in river_indices_list)
    abstraction_river_indices = np.full(
        (len(river_indices_list), max_river_size), -1, dtype=np.int32
    )
    for i, indices in enumerate(river_indices_list):
        abstraction_river_indices[i, : len(indices)] = indices

    return abstraction_area_indices, abstraction_river_indices


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
        if self.model.simulate_hydrology:
            basin_ids: ArrayInt32 = self.model.hydrology.grid.load2d(
                self.model.files["grid"]["routing/basin_ids"]
            )
            river_ids: ArrayInt32 = self.model.hydrology.routing.var.river_ids
            rivers: gpd.GeoDataFrame = (
                self.model.hydrology.routing.get_active_rivers().copy()
            )
            self.var.abstraction_area_indices, self.var.abstraction_river_indices = (
                create_abstraction_areas(basin_ids, river_ids, rivers)
            )

        water_demand, water_return_flow = self.update_water_demand()
        self.var.current_water_demand = water_demand
        self.var.current_return_flow = water_return_flow

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
        for i in range(self.var.abstraction_area_indices.shape[0]):
            area_indices = self.var.abstraction_area_indices[i]
            area_indices = area_indices[area_indices != -1]
            river_indices = self.var.abstraction_river_indices[i]
            river_indices = river_indices[river_indices != -1]

            water_demand_in_abstraction_area = water_demand[area_indices].sum()
            water_consumption_in_abstraction_area = water_consumption[
                area_indices
            ].sum()
            return_flow_in_abstraction_area = (
                water_demand_in_abstraction_area - water_consumption_in_abstraction_area
            )

            demand_per_cell = water_demand_in_abstraction_area / river_indices.size
            return_flow_per_cell = return_flow_in_abstraction_area / river_indices.size
            if return_flow_per_cell < 0:
                return_flow_per_cell = 0.0

            water_demand_assigned_to_rivers[river_indices] = demand_per_cell
            return_flow_assigned_to_rivers[river_indices] = return_flow_per_cell

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
