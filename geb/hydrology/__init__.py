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

from geb.hydrology.HRUs import Data
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
    """The hydrological module of the GEB model.

    This module handles all hydrological processes, including potential evapotranspiration,
    snow and frost dynamics, land cover interactions, soil processes, evaporation, groundwater
    management, interception, sealed water bodies, runoff concentration, routing of water,
    lakes and reservoirs management, water demand, and hillslope erosion.

    Args:
        model: The GEB model instance.
    """

    def __init__(self, model):
        """Create the hydrology module."""
        Data.__init__(self, model)
        Module.__init__(self, model)

        if not self.model.simulate_hydrology:
            return

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

    def get_current_storage(self) -> np.float64:
        """Get the current water storage in the hydrological system.

        Uses float64 to ensure that the storage is calculated accurately. If float32
        is used, the storage can be under- or overestimated due to rounding errors.

        Returns:
            The total water storage in the hydrological system in cubic meters.
        """
        return (
            np.sum(self.HRU.var.SnowCoverS.astype(np.float64) * self.HRU.var.cell_area)
            / self.snowfrost.var.numberSnowLayers
            + (
                self.HRU.var.interception_storage.astype(np.float64)
                * self.HRU.var.cell_area
            ).sum()
            + (
                np.nansum(self.HRU.var.w.astype(np.float64), axis=0)
                * self.HRU.var.cell_area
            ).sum()
            + (self.HRU.var.topwater.astype(np.float64) * self.HRU.var.cell_area).sum()
            + self.routing.router.get_total_storage().astype(np.float64).sum()
            + self.lakes_reservoirs.var.storage.astype(np.float64).sum()
            + self.groundwater.groundwater_content_m3.astype(np.float64).sum()
        )

    def step(self):
        timer: TimingModule = TimingModule("Hydrology")

        if __debug__:
            prev_storage: np.float64 = self.get_current_storage()
            influx: np.float64 = (
                self.HRU.pr.astype(np.float64)
                * 0.001
                * 86400.0
                * self.HRU.var.cell_area
            ).sum()  # m3
            influx += (
                self.grid.var.capillar.astype(np.float64) * self.grid.var.cell_area
            ).sum()

        self.potential_evapotranspiration.step()
        timer.new_split("PET")

        self.lakes_reservoirs.step()
        timer.new_split("Waterbodies")

        snow, rain, snow_melt = self.snowfrost.step()
        timer.new_split("Snow and frost")

        (
            interflow,
            runoff,
            groundwater_recharge,
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow,
            capillary_m,
            total_water_demand_loss_m3,
        ) = self.landcover.step(snow, rain, snow_melt)

        if __debug__:
            outflux: np.float64 = (
                self.HRU.var.actual_evapotranspiration.astype(np.float64)
                * self.HRU.var.cell_area
            ).sum() + total_water_demand_loss_m3

            invented_water: np.float64 = (
                (
                    channel_abstraction_m3.sum()  # already applied but not yet removed from river
                    + groundwater_abstraction_m3.sum()  # already applied but not yet removed from GW
                    + self.model.agents.reservoir_operators.command_area_release_m3.sum()  # already applied but not yet removed from reservoir
                )
                - (
                    (interflow + runoff + return_flow + groundwater_recharge)
                    * self.grid.var.cell_area
                ).sum()  # already removed from sources but not yet added to sinks
            )

            balance_check(
                name="total water balance 2",
                how="sum",
                influxes=[influx, invented_water],
                outfluxes=[
                    outflux,
                ],
                prestorages=[prev_storage],
                poststorages=[self.get_current_storage()],
                tollerance=self.grid.compressed_size
                / 3,  # increase tollerance for large models
            )

        timer.new_split("Landcover")

        baseflow = self.groundwater.step(
            groundwater_recharge, groundwater_abstraction_m3
        )

        if __debug__:
            invented_water += (
                -(
                    baseflow * self.grid.var.cell_area
                ).sum()  # created but still needs to be added to river
                + (
                    groundwater_recharge * self.grid.var.cell_area
                ).sum()  # now accounted for
                - groundwater_abstraction_m3.sum()  # now accounted for
            )

            capillar_next_step: np.float64 = (
                self.grid.var.capillar.astype(np.float64) * self.grid.var.cell_area
            ).sum()

            outflux += capillar_next_step.sum()  # capillary rise is added to sinks

            balance_check(
                name="total water balance 2",
                how="sum",
                influxes=[influx, invented_water],
                outfluxes=[
                    outflux,
                ],
                prestorages=[prev_storage],
                poststorages=[self.get_current_storage()],
                tollerance=self.grid.compressed_size
                / 3,  # increase tollerance for large models
            )

        timer.new_split("GW")

        total_runoff = self.runoff_concentration.step(interflow, baseflow, runoff)
        timer.new_split("Runoff concentration")

        self.lakes_res_small.step()
        timer.new_split("Small waterbodies")

        routing_loss, over_abstraction_m3 = self.routing.step(
            total_runoff, channel_abstraction_m3, return_flow
        )

        if __debug__:
            influx += over_abstraction_m3

            outflux += routing_loss
            invented_water += (
                (interflow + runoff + return_flow) * self.grid.var.cell_area
            ).sum()  # added to sinks, so remove from invented water

            invented_water += (
                baseflow * self.grid.var.cell_area
            ).sum()  # now added to river

            invented_water -= (
                channel_abstraction_m3.sum()  # now removed from river
                + self.model.agents.reservoir_operators.command_area_release_m3.sum()  # now removed from reservoir
            )

            balance_check(
                name="total water balance 3",
                how="sum",
                influxes=[influx, invented_water],
                outfluxes=[
                    outflux,
                ],
                prestorages=[prev_storage],
                poststorages=[self.get_current_storage()],
                tollerance=self.grid.compressed_size
                / 3,  # increase tollerance for large models
            )

        timer.new_split("Routing")

        self.hillslope_erosion.step()
        timer.new_split("Hill slope erosion")

        if self.model.timing:
            print(timer)

        self.report(self, locals())

    def finalize(self) -> None:
        """Finalize the model."""
        # finalize modflow model
        if hasattr(self, "groundwater") and hasattr(self.groundwater, "modflow"):
            self.groundwater.modflow.finalize()

        # if self.config["general"]["simulate_forest"] and self.soil.model.spinup is False:
        if self.model.config["general"]["simulate_forest"]:
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

    @property
    def name(self):
        return "hydrology"
