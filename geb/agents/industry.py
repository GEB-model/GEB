"""This module contains the Industry agent class for simulating industrial water demand in the GEB model."""

from __future__ import annotations

import calendar
from typing import TYPE_CHECKING

import numpy as np

from geb.types import ArrayFloat32

from ..hydrology.landcovers import SEALED
from .general import AgentBaseClass, downscale_volume

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel


class Industry(AgentBaseClass):
    """This class is used to simulate industry.

    Note:
        Currently, this module is not actually agent-based but rather
        uses aggregated pre-defined water demand data.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

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
        """Set initial water demand and efficiency during spinup."""
        water_demand, efficiency = self.update_water_demand()
        self.var.current_water_demand = water_demand
        self.var.current_efficiency = efficiency

    def update_water_demand(self) -> tuple[ArrayFloat32, ArrayFloat32]:
        """Update the water demand for industry at the HRU level.

        Returns:
            A tuple containing:
            - The updated water demand (in m3/day).
            - The updated efficiency [0-1].
        """
        if self.config.get("disable_water_demand", False):
            self.model.logger.info(
                "[Industry] Water demand and efficiency set to 0 due to config setting."
            )
            zero_array = np.zeros(self.HRU.var.land_use_ratio.shape, dtype=float)
            efficiency_grid = self.model.hydrology.to_grid(
                HRU_data=zero_array, fn="max"
            )
            self.var.last_water_demand_update = self.model.current_time
            return zero_array, efficiency_grid

        else:
            downscale_mask = self.model.hydrology.HRU.var.land_use_type != SEALED
            days_in_year = 366 if calendar.isleap(self.model.current_time.year) else 365

            water_demand = (
                self.model.industry_water_demand_ds.sel(
                    time=self.model.current_time, method="ffill", tolerance="366D"
                ).industry_water_demand
                * 1_000_000
                / days_in_year
            )
            water_demand = (
                water_demand.rio.write_crs(4326).rio.reproject(
                    4326,
                    shape=self.grid.shape,
                    transform=self.grid.transform,
                )
                / (water_demand.rio.transform().a / self.grid.transform.a) ** 2
            )  # correct for change in cell size
            water_demand = (
                downscale_volume(
                    water_demand.rio.transform().to_gdal(),
                    self.grid.gt,
                    water_demand.values,
                    self.grid.mask,
                    self.model.hydrology.mapping_grid_to_HRU_uncompressed,
                    downscale_mask,
                    self.HRU.var.land_use_ratio,
                )
                / self.HRU.var.cell_area
            )  # convert to m/day

            water_demand = (
                water_demand
                * self.model.config["parameters"]["water_demand_multiplier_industry"]
            )

            water_consumption = (
                self.model.industry_water_consumption_ds.sel(
                    time=self.model.current_time, method="ffill"
                ).industry_water_consumption
                * 1_000_000
                / days_in_year
            )
            water_consumption = (
                water_consumption.rio.write_crs(4326).rio.reproject(
                    4326,
                    shape=self.grid.shape,
                    transform=self.grid.transform,
                )
                / (water_consumption.rio.transform().a / self.grid.transform.a) ** 2
            )
            water_consumption = (
                downscale_volume(
                    water_consumption.rio.transform().to_gdal(),
                    self.grid.gt,
                    water_consumption.values,
                    self.grid.mask,
                    self.model.hydrology.mapping_grid_to_HRU_uncompressed,
                    downscale_mask,
                    self.HRU.var.land_use_ratio,
                )
                / self.HRU.var.cell_area
            )  # convert to m/day

            efficiency = np.divide(
                water_consumption,
                water_demand,
                out=np.zeros_like(water_consumption, dtype=float),
                where=water_demand != 0,
            )

            efficiency = self.model.hydrology.to_grid(HRU_data=efficiency, fn="max")

            assert (efficiency <= 1).all()
            assert (efficiency >= 0).all()
            self.var.last_water_demand_update = self.model.current_time
            return water_demand, efficiency

    def water_demand(self) -> tuple[ArrayFloat32, ArrayFloat32]:
        """Get the current water demand for industry at the HRU level.

        Updates the water demand only if data for this timestep is available.
        Otherwise, assumes the last known water demand.

        Returns:
            A tuple containing:
            - The current water demand (in m3/day).
            - The current efficiency [0-1].
        """
        if (
            np.datetime64(self.model.current_time, "ns")
            in self.model.industry_water_consumption_ds.time
        ):
            water_demand, efficiency = self.update_water_demand()
            self.var.current_water_demand = water_demand
            self.var.current_efficiency = efficiency

        assert (
            self.model.current_time - self.var.last_water_demand_update
        ).days < 366, (
            "Water demand has not been updated for over a year. "
            "Please check the industry water demand datasets."
        )
        return self.var.current_water_demand, self.var.current_efficiency

    def step(self) -> None:
        """This function is run each timestep."""
        self.report(locals())
