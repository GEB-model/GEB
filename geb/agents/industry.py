# -*- coding: utf-8 -*-
from . import AgentBaseClass
import calendar
import numpy as np

from .general import downscale_volume


class Industry(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model, agents):
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
        downscale_mask = self.model.data.HRU.land_use_type != 4
        if self.model.use_gpu:
            downscale_mask = downscale_mask.get()

        days_in_year = 366 if calendar.isleap(self.model.current_time.year) else 365

        water_demand = (
            self.model.industry_water_demand_ds.sel(
                time=self.model.current_time, method="ffill"
            ).industry_water_demand
            * 1_000_000
            / days_in_year
        )
        water_demand = downscale_volume(
            self.model.industry_water_demand_ds.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_demand.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.data.HRU.land_use_ratio,
        )
        if self.model.use_gpu:
            water_demand = cp.array(water_demand)
        water_demand = self.model.data.HRU.M3toM(water_demand)

        water_consumption = (
            self.model.industry_water_consumption_ds.sel(
                time=self.model.current_time, method="ffill"
            ).industry_water_consumption
            * 1_000_000
            / days_in_year
        )
        water_consumption = downscale_volume(
            self.model.industry_water_consumption_ds.rio.transform().to_gdal(),
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

        efficiency = np.divide(
            water_consumption,
            water_demand,
            out=np.zeros_like(water_consumption, dtype=float),
            where=water_demand != 0,
        )

        efficiency = self.model.data.to_grid(HRU_data=efficiency, fn="max")

        assert (efficiency <= 1).all()
        assert (efficiency >= 0).all()
        return water_demand, efficiency

    def step(self) -> None:
        """This function is run each timestep."""
        pass
