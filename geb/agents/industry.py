"""This module contains the Industry agent class for simulating industrial water demand in the GEB model."""

import calendar
import datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from geb.geb_types import (
    ArrayFloat32,
)
from geb.hydrology.HRUs import load_water_demand_xr
from geb.hydrology.water_demand import (
    assign_demand_and_return_flow_to_abstraction_rivers,
)
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

        # Assign water demand and return flow to associated abstraction rivers
        water_demand_assigned_to_rivers, return_flow_assigned_to_rivers = (
            assign_demand_and_return_flow_to_abstraction_rivers(
                water_demand=water_demand,
                water_consumption=water_consumption,
                abstraction_area_indices=self.model.hydrology.water_demand.var.abstraction_area_indices,
                abstraction_river_indices=self.model.hydrology.water_demand.var.abstraction_river_indices,
            )
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
