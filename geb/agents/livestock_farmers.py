# -*- coding: utf-8 -*-
import calendar
import numpy as np
from .general import downscale_volume, AgentBaseClass

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass
from ..hydrology.landcover import GRASSLAND_LIKE


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

        water_demand, efficiency = self.update_water_demand()
        self.current_water_demand = water_demand
        self.current_efficiency = efficiency

    def initiate(self) -> None:
        pass

    def update_water_demand(self):
        """
        Dynamic part of the water demand module - livestock
        read monthly (or yearly) water demand from netcdf and transform (if necessary) to [m/day]

        """
        days_in_year = 366 if calendar.isleap(self.model.current_time.year) else 365

        # grassland/non-irrigated land that is not owned by a crop farmer
        if self.model.use_gpu:
            land_use_type = self.model.data.HRU.land_use_type.get()
        else:
            land_use_type = self.model.data.HRU.land_use_type
        downscale_mask = (land_use_type != GRASSLAND_LIKE) | (
            self.model.data.HRU.land_owners != -1
        )

        # transform from mio m3 per year to m3/day
        water_consumption = (
            self.model.livestock_water_consumption_ds.sel(
                time=self.model.current_time, method="ffill", tolerance="366D"
            ).livestock_water_consumption
            * 1_000_000
            / days_in_year
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
        self.last_water_demand_update = self.model.current_time
        return water_demand, efficiency

    def water_demand(self):
        if (
            np.datetime64(self.model.current_time, "ns")
            in self.model.livestock_water_consumption_ds.time
        ):
            water_demand, efficiency = self.update_water_demand()
            self.current_water_demand = water_demand
            self.current_efficiency = efficiency

        assert (self.model.current_time - self.last_water_demand_update).days < 366, (
            "Water demand has not been updated for over a year. "
            "Please check the livestock water demand datasets."
        )
        return self.current_water_demand, self.current_efficiency

    def step(self) -> None:
        """This function is run each timestep."""
        pass
