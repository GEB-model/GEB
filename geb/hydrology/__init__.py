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

"""Hydrology submodule for the GEB model. Holds all hydrology related submodules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geb.hydrology.HRUs import Data
from geb.module import Module
from geb.workflows import TimingModule, balance_check

from .erosion.hillslope import HillSlopeErosion
from .groundwater import GroundWater
from .lakes_reservoirs import LakesReservoirs
from .landsurface import LandSurface
from .routing import Routing
from .runoff_concentration import concentrate_runoff
from .water_demand import WaterDemand

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


class Hydrology(Data, Module):
    """The hydrological module of the GEB model.

    This module handles all hydrological processes, including potential evapotranspiration,
    snow and frost dynamics, land cover interactions, soil processes, groundwater
    management, interception, sealed water bodies, runoff concentration, routing of water,
    lakes and reservoirs management, water demand, and hillslope erosion.

    Args:
        model: The GEB model instance.
    """

    def __init__(self, model: GEBModel) -> None:
        """Create the hydrology module."""
        Data.__init__(self, model)
        Module.__init__(self, model)

        if not self.model.simulate_hydrology:
            return

        self.dynamic_water_bodies = False

        self.landsurface = LandSurface(self.model, self)
        self.groundwater = GroundWater(self.model, self)
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
            (
                (
                    self.HRU.var.snow_water_equivalent_m.astype(np.float64)
                    + self.HRU.var.liquid_water_in_snow_m.astype(np.float64)
                    + self.HRU.var.interception_storage_m.astype(np.float64)
                    + np.nansum(self.HRU.var.w.astype(np.float64), axis=0)
                    + self.HRU.var.topwater_m.astype(np.float64)
                )
                * self.HRU.var.cell_area
            ).sum()
            + (self.HRU.var.topwater.astype(np.float64) * self.HRU.var.cell_area).sum()
            + self.routing.router.get_total_storage(
                self.grid.var.discharge_in_rivers_m3_s_substep
            )
            .astype(np.float64)
            .sum()
            + self.lakes_reservoirs.var.storage.astype(np.float64).sum()
            + self.groundwater.groundwater_content_m3.astype(np.float64).sum()
        )

    def step(self) -> None:
        """Perform a single time step of the hydrological model.

        Calculates the water balance and updates all hydrological components.
        """
        timer: TimingModule = TimingModule("Hydrology")

        if __debug__:
            prev_storage: np.float64 = self.get_current_storage()
            influx = (
                self.grid.var.capillar.astype(np.float64) * self.grid.var.cell_area
            ).sum()
        else:
            prev_storage: np.float64 = np.float64(np.nan)

        self.lakes_reservoirs.step()
        timer.finish_split("Waterbodies")

        (
            reference_evapotranspiration_water_m,
            interflow_m,
            runoff_m,
            groundwater_recharge_m,
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow_m,
            total_water_demand_loss_m3,
            actual_evapotranspiration_m,
            sublimation_or_deposition_m,
            pr_total_m3,
        ) = self.landsurface.step()

        if __debug__:
            influx += pr_total_m3

        interflow_m = self.to_grid(HRU_data=interflow_m, fn="weightedmean")
        runoff_m = self.to_grid(HRU_data=runoff_m, fn="weightedmean")
        groundwater_recharge_m = self.to_grid(
            HRU_data=groundwater_recharge_m, fn="weightedmean"
        )

        if __debug__:
            outflux_m3: np.float64 = (
                actual_evapotranspiration_m.astype(np.float64) * self.HRU.var.cell_area
            ).sum() + total_water_demand_loss_m3

            outflux_m3 -= (sublimation_or_deposition_m * self.HRU.var.cell_area).sum()

            invented_water: np.float64 = (
                (
                    channel_abstraction_m3.sum()  # already applied but not yet removed from river
                    + groundwater_abstraction_m3.sum()  # already applied but not yet removed from GW
                    + self.model.agents.reservoir_operators.command_area_release_m3.sum()  # already applied but not yet removed from reservoir
                )
                - (
                    (
                        interflow_m.sum(axis=0)
                        + runoff_m.sum(axis=0)
                        + return_flow_m
                        + groundwater_recharge_m
                    )
                    * self.grid.var.cell_area
                ).sum()  # already removed from sources but not yet added to sinks
            )

            balance_check(
                name="total water balance 1",
                how="sum",
                influxes=[influx, invented_water],
                outfluxes=[
                    outflux_m3,
                ],
                prestorages=[prev_storage],
                poststorages=[self.get_current_storage()],
                tolerance=self.grid.compressed_size
                / 3,  # increase tolerance for large models
            )

        timer.finish_split("Land surface")

        baseflow_m = self.groundwater.step(
            groundwater_recharge_m, groundwater_abstraction_m3
        )

        if __debug__:
            invented_water += (
                -(
                    baseflow_m * self.grid.var.cell_area
                ).sum()  # created but still needs to be added to river
                + (
                    groundwater_recharge_m * self.grid.var.cell_area
                ).sum()  # now accounted for
                - groundwater_abstraction_m3.sum()  # now accounted for
            )

            capillar_next_step: np.float64 = (
                self.grid.var.capillar.astype(np.float64) * self.grid.var.cell_area
            ).sum()

            outflux_m3 += capillar_next_step.sum()  # capillary rise is added to sinks

            balance_check(
                name="total water balance 2",
                how="sum",
                influxes=[influx, invented_water],
                outfluxes=[
                    outflux_m3,
                ],
                prestorages=[prev_storage],
                poststorages=[self.get_current_storage()],
                tolerance=self.grid.compressed_size
                / 3,  # increase tolerance for large models
            )

        timer.finish_split("GW")

        self.grid.var.total_runoff_m = concentrate_runoff(
            interflow_m, baseflow_m, runoff_m
        )
        timer.finish_split("Runoff concentration")

        routing_loss_m3, over_abstraction_m3 = self.routing.step(
            self.grid.var.total_runoff_m,
            channel_abstraction_m3,
            return_flow_m,
            reference_evapotranspiration_water_m,
        )

        current_storage: np.float64 = self.get_current_storage()

        if __debug__:
            influx += over_abstraction_m3

            outflux_m3 += routing_loss_m3
            invented_water += (
                (interflow_m.sum(axis=0) + runoff_m.sum(axis=0) + return_flow_m)
                * self.grid.var.cell_area
            ).sum()  # added to sinks, so remove from invented water

            invented_water += (
                baseflow_m * self.grid.var.cell_area
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
                    outflux_m3,
                ],
                prestorages=[prev_storage],
                poststorages=[current_storage],
                tolerance=self.grid.compressed_size
                / 3,  # increase tolerance for large models
            )

        timer.finish_split("Routing")

        self.hillslope_erosion.step()
        timer.finish_split("Hill slope erosion")

        if self.model.timing:
            print(timer)

        self.report(locals())

    def finalize(self) -> None:
        """Finalize the model."""
        # finalize modflow model
        if hasattr(self, "groundwater") and hasattr(self.groundwater, "modflow"):
            self.groundwater.modflow.finalize()

        if self.model.config["general"]["simulate_forest"]:
            for plantFATE_model in self.plantFATE:
                if plantFATE_model is not None:
                    plantFATE_model.finalize()

    @property
    def name(self) -> str:
        """Name of the module."""
        return "hydrology"
