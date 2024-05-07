# -*- coding: utf-8 -*-
from . import AgentBaseClass
import calendar
import numpy as np
from .general import downscale_volume


class LiveStockFarmers(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model, agents, reduncancy):
        self.model = model
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["town_managers"]
            if "town_managers" in self.model.config["agent_settings"]
            else {}
        )

        AgentBaseClass.__init__(self)

    def initiate(self) -> None:
        return

    def water_demand(self):
        """
        Dynamic part of the water demand module - livestock
        read monthly (or yearly) water demand from netcdf and transform (if necessary) to [m/day]

        """
        days_in_month = calendar.monthrange(
            self.model.current_time.year, self.model.current_time.month
        )[1]

        # grassland/non-irrigated land that is not owned by a crop farmer
        if self.model.use_gpu:
            land_use_type = self.model.data.HRU.land_use_type.get()
        else:
            land_use_type = self.model.data.HRU.land_use_type
        downscale_mask = (land_use_type != 1) | (self.model.data.HRU.land_owners != -1)

        # transform from mio m3 per year (or month) to m/day
        water_consumption = (
            self.model.livestock_water_consumption_ds.sel(
                time=self.model.current_time, method="ffill"
            ).livestock_water_consumption
            * 1_000_000
            / days_in_month
        )
        water_consumption = downscale_volume(
            self.model.livestock_water_consumption_ds.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_consumption.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.data.HRU.land_use_ratio,
        )
        if self.model.use_gpu:
            water_consumption = cp.array(water_consumption)
        water_consumption = self.model.data.HRU.M3toM(water_consumption)

        efficiency = 1.0
        water_demand = water_consumption / efficiency
        return water_demand, efficiency

    def step(self) -> None:
        """This function is run each timestep."""
        pass
