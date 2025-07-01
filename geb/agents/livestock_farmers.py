# -*- coding: utf-8 -*-
import calendar

import numpy as np
import numpy.typing as npt

from ..hydrology.landcover import GRASSLAND_LIKE
from .general import AgentBaseClass, downscale_volume


class LiveStockFarmers(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model, agents, reduncancy):
        super().__init__(model)

        if self.model.simulate_hydrology:
            self.HRU = model.hydrology.HRU
            self.grid = model.hydrology.grid

        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["town_managers"]
            if "town_managers" in self.model.config["agent_settings"]
            else {}
        )

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self):
        return "agents.livestock_farmers"

    def spinup(self) -> None:
        water_demand, efficiency = self.update_water_demand()
        self.var.current_water_demand = water_demand
        self.var.current_efficiency = efficiency

    def update_water_demand(self):
        days_in_year = 366 if calendar.isleap(self.model.current_time.year) else 365

        # grassland/non-irrigated land that is not owned by a crop farmer
        land_use_type = self.HRU.var.land_use_type
        downscale_mask = (land_use_type != GRASSLAND_LIKE) | (
            self.HRU.var.land_owners != -1
        )

        # transform from mio m3 per year to m3/day
        water_consumption = (
            self.model.livestock_water_consumption_ds.sel(
                time=self.model.current_time, method="ffill", tolerance="366D"
            ).livestock_water_consumption
            * 1_000_000
            / days_in_year
        )
        water_consumption = (
            water_consumption.rio.write_crs(4326).rio.reproject(
                4326,
                shape=self.model.hydrology.grid.shape,
                transform=self.model.hydrology.grid.transform,
            )
            / (
                water_consumption.rio.transform().a
                / self.model.hydrology.grid.transform.a
            )
            ** 2
        )
        water_consumption: npt.NDArray[np.float32] = (
            downscale_volume(
                water_consumption.rio.transform().to_gdal(),
                self.model.hydrology.grid.gt,
                water_consumption.values,
                self.model.hydrology.grid.mask,
                self.model.hydrology.grid_to_HRU_uncompressed,
                downscale_mask,
                self.HRU.var.land_use_ratio,
            )
            / self.HRU.var.cell_area
        )  # convert to m/day

        efficiency = 1.0
        water_demand = water_consumption / efficiency
        self.var.last_water_demand_update = self.model.current_time
        return water_demand, efficiency

    def water_demand(self):
        if (
            np.datetime64(self.model.current_time, "ns")
            in self.model.livestock_water_consumption_ds.time
        ):
            water_demand, efficiency = self.update_water_demand()
            self.var.current_water_demand = water_demand
            self.var.current_efficiency = efficiency

        assert (
            self.model.current_time - self.var.last_water_demand_update
        ).days < 366, (
            "Water demand has not been updated for over a year. "
            "Please check the livestock water demand datasets."
        )
        return self.var.current_water_demand, self.var.current_efficiency

    def step(self) -> None:
        """This function is run each timestep."""
        self.report(self, locals())
