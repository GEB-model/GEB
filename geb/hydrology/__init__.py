# --------------------------------------------------------------------------------
# This file contains code that has been adapted from an original source available
# in a public repository under the GNU General Public License. The original code
# has been modified to fit the specific needs of this project.
#
# Original source repository: https://github.com/iiasa/CWatM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------------

import numpy as np

from geb.HRUs import Data
from geb.module import Module
from geb.workflows import TimingModule, balance_check

from .erosion.hillslope import HillSlopeErosion
from .evaporation import Evaporation
from .groundwater import GroundWater
from .interception import Interception
from .lakes_res_small import SmallLakesReservoirs
from .lakes_reservoirs import LakesReservoirs
from .landcover import LandCover
from .potential_evapotranspiration import PotentialEvapotranspiration
from .routing import Routing
from .runoff_concentration import RunoffConcentration
from .sealed_water import SealedWater
from .snow_frost import SnowFrost
from .soil import Soil
from .water_demand import WaterDemand


class Hydrology(Data, Module):
    def __init__(self, model):
        """
        Init part of the initial part
        defines the mask map and the outlet points
        initialization of the hydrological modules
        """
        Data.__init__(self, model)
        Module.__init__(self, model)

        self.dynamic_water_bodies = False
        self.crop_factor_calibration_factor = 1

        self.potential_evapotranspiration = PotentialEvapotranspiration(
            self.model, self
        )
        self.snowfrost = SnowFrost(self.model, self)
        self.landcover = LandCover(self.model, self)
        self.soil = Soil(self.model, self)
        self.evaporation = Evaporation(self.model, self)
        self.groundwater = GroundWater(self.model, self)
        self.interception = Interception(self.model, self)
        self.sealed_water = SealedWater(self.model, self)
        self.runoff_concentration = RunoffConcentration(self.model, self)
        self.lakes_res_small = SmallLakesReservoirs(self.model, self)
        self.routing = Routing(self.model, self)
        self.lakes_reservoirs = LakesReservoirs(self.model, self)
        self.water_demand = WaterDemand(self.model, self)
        self.hillslope_erosion = HillSlopeErosion(self.model, self)

    def step(self):
        timer = TimingModule("Hydrology")

        self.potential_evapotranspiration.step()
        timer.new_split("PET")

        self.lakes_reservoirs.step()
        timer.new_split("Waterbodies")

        self.snowfrost.step()
        timer.new_split("Snow and frost")

        (
            interflow,
            runoff,
            groundwater_recharge,
            groundwater_abstraction,
            channel_abstraction_m3,
            return_flow,
        ) = self.landcover.step()
        timer.new_split("Landcover")

        baseflow = self.groundwater.step(groundwater_recharge, groundwater_abstraction)
        timer.new_split("GW")

        total_runoff = self.runoff_concentration.step(interflow, baseflow, runoff)
        timer.new_split("Runoff concentration")

        self.lakes_res_small.step()
        timer.new_split("Small waterbodies")

        self.routing.step(total_runoff, channel_abstraction_m3, return_flow)
        timer.new_split("Routing")

        self.hillslope_erosion.step()
        timer.new_split("Hill slope erosion")

        if self.model.timing:
            print(timer)

        if __debug__:
            self.water_balance()

        self.report(self, locals())

    def finalize(self) -> None:
        """
        Finalize the model
        """
        # finalize modflow model
        if hasattr(self, "groundwater") and hasattr(self.groundwater, "modflow"):
            self.groundwater.modflow.finalize()

        # if self.config["general"]["simulate_forest"] and self.soil.model.spinup is False:
        if self.config["general"]["simulate_forest"]:
            for plantFATE_model in self.plantFATE:
                if plantFATE_model is not None:
                    plantFATE_model.finalize()

    @property
    def n_individuals_per_m2(self):
        n_invidiuals_per_m2_per_HRU = np.array(
            [model.n_individuals for model in self.plantFATE if model is not None]
        )
        land_use_ratios = self.data.HRU.land_use_ratio[self.soil.plantFATE_forest_RUs]
        return np.array(
            (n_invidiuals_per_m2_per_HRU * land_use_ratios).sum()
            / land_use_ratios.sum()
        )

    @property
    def biomass_per_m2(self):
        biomass_per_m2_per_HRU = np.array(
            [model.biomass for model in self.plantFATE if model is not None]
        )
        land_use_ratios = self.data.HRU.land_use_ratio[self.soil.plantFATE_forest_RUs]
        return np.array(
            (biomass_per_m2_per_HRU * land_use_ratios).sum() / land_use_ratios.sum()
        )

    def water_balance(self):
        current_storage = (
            np.sum(self.HRU.var.SnowCoverS * self.HRU.var.cell_area)
            / self.snowfrost.var.numberSnowLayers
            + (self.HRU.var.interception_storage * self.HRU.var.cell_area).sum()
            + (np.nansum(self.HRU.var.w, axis=0) * self.HRU.var.cell_area).sum()
            + (self.HRU.var.topwater * self.HRU.var.cell_area).sum()
            + self.routing.router.get_available_storage().sum()
            + self.lakes_reservoirs.var.storage.sum()
            + self.groundwater.groundwater_content_m3.sum()
        )

        # in the first timestep of the spinup, we don't have the storage of the
        # previous timestep, so we can't check the balance
        if not self.model.current_timestep == 0 and self.model.in_spinup:
            influx = (self.HRU.var.precipitation_m_day * self.HRU.var.cell_area).sum()
            outflux = (
                self.HRU.var.actual_evapotranspiration * self.HRU.var.cell_area
            ).sum() + self.model.hydrology.routing.routing_loss

            balance_check(
                name="total water balance",
                how="sum",
                influxes=[
                    influx,
                ],
                outfluxes=[
                    outflux,
                ],
                prestorages=[self.var.system_storage],
                poststorages=[current_storage],
                tollerance=100,
            )

        # update the storage for the next timestep
        self.var.system_storage = current_storage
        self.var.pre_groundwater_content_m3 = (
            self.groundwater.groundwater_content_m3.copy()
        )

    @property
    def name(self):
        return "hydrology"
