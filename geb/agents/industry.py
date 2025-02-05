# -*- coding: utf-8 -*-
import calendar
import numpy as np
from .general import downscale_volume, AgentBaseClass
from ..hydrology.landcover import SEALED


class Industry(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model, agents):
        self.model = model
        self.HRU = model.data.HRU
        self.grid = model.data.grid
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["town_managers"]
            if "town_managers" in self.model.config["agent_settings"]
            else {}
        )

        AgentBaseClass.__init__(self)

        if self.model.spinup:
            self.spinup()

    def spinup(self) -> None:
        self.var = self.model.store.create_bucket("agents.industry.var")
        water_demand, efficiency = self.update_water_demand()
        self.var.current_water_demand = water_demand
        self.var.current_efficiency = efficiency

    def update_water_demand(self):
        downscale_mask = self.HRU.var.land_use_type != SEALED
        days_in_year = 366 if calendar.isleap(self.model.current_time.year) else 365

        water_demand = (
            self.model.industry_water_demand_ds.sel(
                time=self.model.current_time, method="ffill", tolerance="366D"
            ).industry_water_demand
            * 1_000_000
            / days_in_year
        )
        water_demand = (
            water_demand.rio.set_crs(4326).rio.reproject(
                4326,
                shape=self.model.data.grid.shape,
                transform=self.model.data.grid.transform,
            )
            / (water_demand.rio.transform().a / self.model.data.grid.transform.a) ** 2
        )  # correct for change in cell size
        water_demand = downscale_volume(
            water_demand.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_demand.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.HRU.var.land_use_ratio,
        )
        water_demand = self.HRU.M3toM(water_demand)

        water_consumption = (
            self.model.industry_water_consumption_ds.sel(
                time=self.model.current_time, method="ffill"
            ).industry_water_consumption
            * 1_000_000
            / days_in_year
        )
        water_consumption = (
            water_consumption.rio.set_crs(4326).rio.reproject(
                4326,
                shape=self.model.data.grid.shape,
                transform=self.model.data.grid.transform,
            )
            / (water_consumption.rio.transform().a / self.model.data.grid.transform.a)
            ** 2
        )
        water_consumption = downscale_volume(
            water_consumption.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_consumption.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.HRU.var.land_use_ratio,
        )

        water_consumption = self.HRU.M3toM(water_consumption)

        efficiency = np.divide(
            water_consumption,
            water_demand,
            out=np.zeros_like(water_consumption, dtype=float),
            where=water_demand != 0,
        )

        efficiency = self.model.data.to_grid(HRU_data=efficiency, fn="max")

        assert (efficiency <= 1).all()
        assert (efficiency >= 0).all()
        self.var.last_water_demand_update = self.model.current_time
        return water_demand, efficiency

    def water_demand(self):
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
        pass
