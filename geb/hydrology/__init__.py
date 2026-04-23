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

from typing import TYPE_CHECKING

import numba
import numpy as np

from geb.geb_types import TwoDArrayFloat32 as TwoDArrayFloat32
from geb.hydrology.HRUs import Data
from geb.module import Module
from geb.workflows import TimingModule, balance_check

from .erosion.hillslope import HillSlopeErosion
from .groundwater import GroundWater
from .landsurface.landsurface_model import LandSurface
from .routing import Routing
from .runoff_concentration import RunoffConcentrator
from .water_demand import WaterDemand
from .waterbodies import WaterBodies

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


@numba.njit(cache=True, parallel=True)
def _sum_landsurface_storage(
    snow_water_equivalent_m: np.ndarray,
    liquid_water_in_snow_m: np.ndarray,
    interception_storage_m: np.ndarray,
    water_content_m: np.ndarray,
    topwater_m: np.ndarray,
    cell_area: np.ndarray,
) -> np.float64:
    """Fused parallel kernel summing all land-surface water stores.

    Performs a single pass over HRUs, accumulating directly in float64,
    which avoids the multiple temporary float64 arrays that numpy would
    allocate with chained arithmetic.

    Args:
        snow_water_equivalent_m: Snow water equivalent per HRU (m).
        liquid_water_in_snow_m: Liquid water held in the snowpack per HRU (m).
        interception_storage_m: Interception storage per HRU (m).
        water_content_m: Soil water content per layer and HRU, shape
            ``(n_layers, n_hru)`` (m). NaN values are ignored.
        topwater_m: Ponded topwater per HRU (m).
        cell_area: Area of each HRU (m2).

    Returns:
        Total land-surface water storage (m3).
    """
    n_hru = snow_water_equivalent_m.shape[0]
    n_layers = water_content_m.shape[0]
    total = np.float64(0.0)
    for i in numba.prange(n_hru):  # ty:ignore[not-iterable]
        soil_water = np.float64(0.0)
        for layer in range(n_layers):
            wc = water_content_m[layer, i]
            if not np.isnan(wc):
                soil_water += np.float64(wc)
        hru_storage = (
            np.float64(snow_water_equivalent_m[i])
            + np.float64(liquid_water_in_snow_m[i])
            + np.float64(interception_storage_m[i])
            + soil_water
            + np.float64(topwater_m[i])
        ) * np.float64(cell_area[i])
        total += hru_storage
    return total


@numba.njit(cache=True, parallel=True)
def _sum_overland_flow_buffer_storage(
    overland_flow_buffer: np.ndarray,
    cell_area: np.ndarray,
) -> np.float64:
    """Fused parallel kernel summing the overland flow buffer storage.

    Sums over substep layers and multiplies by cell area in a single pass,
    avoiding the temporary float64 copy that numpy would allocate.

    Args:
        overland_flow_buffer: Overland flow buffer, shape ``(n_substeps, n_grid)`` (m).
        cell_area: Area of each grid cell (m2).

    Returns:
        Total overland flow buffer storage (m3).
    """
    n_substeps = overland_flow_buffer.shape[0]
    n_grid = overland_flow_buffer.shape[1]
    total = np.float64(0.0)
    for i in numba.prange(n_grid):  # ty:ignore[not-iterable]
        cell_sum = np.float64(0.0)
        for s in range(n_substeps):
            cell_sum += np.float64(overland_flow_buffer[s, i])
        total += cell_sum * np.float64(cell_area[i])
    return total


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

        self.dynamic_waterbodies = False

        self.landsurface = LandSurface(self.model, self)
        self.groundwater = GroundWater(self.model, self)
        self.routing = Routing(self.model, self)
        self.waterbodies = WaterBodies(self.model, self)
        self.water_demand = WaterDemand(self.model, self)
        self.hillslope_erosion = HillSlopeErosion(self.model, self)
        self.runoff_concentrator = RunoffConcentrator(self.model, self)

    def get_landsurface_storage_m3(self) -> np.float64:
        """Get the current water storage in the land surface components.

        Includes snow, liquid water in snow, interception, soil water content,
        topwater, and the overland flow buffer. Uses float64 for accuracy.

        Returns:
            Total land surface water storage (m3).
        """
        return _sum_landsurface_storage(
            self.HRU.var.snow_water_equivalent_m,
            self.HRU.var.liquid_water_in_snow_m,
            self.HRU.var.interception_storage_m,
            self.HRU.var.water_content_m,
            self.HRU.var.topwater_m,
            self.HRU.var.cell_area,
        )

    def get_overland_flow_buffer_storage_m3(self) -> np.float64:
        """Get the current water storage in the overland flow buffer.

        Uses float64 for accuracy.

        Returns:
            Total overland flow buffer storage (m3).
        """
        return _sum_overland_flow_buffer_storage(
            self.grid.var.overland_flow_buffer,
            self.grid.var.cell_area,
        )

    def get_routing_storage_m3(self) -> np.float64:
        """Get the current water storage in the river routing network.

        Uses float64 for accuracy.

        Returns:
            Total river routing water storage (m3).
        """
        return (
            self.routing.router.get_total_storage(
                self.grid.var.discharge_in_rivers_m3_s_substep,
                self.grid.var.river_storage_alpha,
                self.grid.var.river_storage_beta,
            )
            .astype(np.float64)
            .sum()
        )

    def get_waterbodies_storage_m3(self) -> np.float64:
        """Get the current water storage in lakes and reservoirs.

        Uses float64 for accuracy.

        Returns:
            Total water body storage (m3).
        """
        return self.waterbodies.var.storage.astype(np.float64).sum()

    def get_groundwater_storage_m3(self) -> np.float64:
        """Get the current water storage in the groundwater system.

        Uses float64 for accuracy.

        Returns:
            Total groundwater storage (m3).
        """
        return self.groundwater.groundwater_content_m3.astype(np.float64).sum()

    def get_current_storage(self) -> np.float64:
        """Get the current water storage in the hydrological system.

        Uses float64 to ensure that the storage is calculated accurately. If float32
        is used, the storage can be under- or overestimated due to rounding errors.

        Returns:
            The total water storage in the hydrological system (m3).
        """
        return (
            self.get_landsurface_storage_m3()
            + self.get_overland_flow_buffer_storage_m3()
            + self.get_routing_storage_m3()
            + self.get_waterbodies_storage_m3()
            + self.get_groundwater_storage_m3()
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

        self.waterbodies.step()
        timer.finish_split("Waterbodies")

        (
            reference_evapotranspiration_water_m,
            interflow_m,
            overland_runoff_m,
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

        interflow_m = self.to_grid(HRU_data=interflow_m)
        overland_runoff_m = self.to_grid(HRU_data=overland_runoff_m)
        groundwater_recharge_m = self.to_grid(HRU_data=groundwater_recharge_m)

        if self.model.config["hazards"]["floods"]["simulate"]:
            self.model.hazard_driver.floods.save_runoff_m(overland_runoff_m)

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
                        + overland_runoff_m.sum(axis=0)
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

        total_runoff_m = self.runoff_concentrator.step(
            interflow_m=interflow_m, baseflow_m=baseflow_m, runoff_m=overland_runoff_m
        ).astype(np.float32)

        if __debug__:
            invented_water += (
                (interflow_m.sum(axis=0) + overland_runoff_m.sum(axis=0) + baseflow_m)
                * self.grid.var.cell_area
            ).sum()  # added to sinks, so remove from invented water
            invented_water -= (total_runoff_m * self.grid.var.cell_area).sum()

            balance_check(
                name="total water balance 3",
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

        timer.finish_split("Runoff concentration")

        total_inflow_m3, routing_loss_m3, over_abstraction_m3 = self.routing.step(
            total_runoff_m,
            channel_abstraction_m3,
            return_flow_m,
            reference_evapotranspiration_water_m,
        )

        if self.model.config["hazards"]["floods"]["simulate"]:
            self.model.hazard_driver.floods.save_discharge(
                discharge_m3_s_per_substep=self.grid.var.discharge_m3_s_per_substep
            )

        current_storage: np.float64 = self.get_current_storage()

        if __debug__:
            influx += over_abstraction_m3
            influx += total_inflow_m3

            outflux_m3 += routing_loss_m3
            invented_water += (
                return_flow_m * self.grid.var.cell_area
            ).sum()  # added to sinks, so remove from invented water

            invented_water += (total_runoff_m * self.grid.var.cell_area).sum()

            invented_water -= (
                channel_abstraction_m3.sum()  # now removed from river
                + self.model.agents.reservoir_operators.command_area_release_m3.sum()  # now removed from reservoir
            )

            balance_check(
                name="total water balance 4",
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

        self.hillslope_erosion.step(overland_runoff_m.sum(axis=0))
        timer.finish_split("Hill slope erosion")

        if self.model.timing:
            self.model.logger.debug(timer)

        self.report(locals())

    def finalize(self) -> None:
        """Finalize the model."""
        # finalize modflow model
        if hasattr(self, "groundwater") and hasattr(self.groundwater, "modflow"):
            self.groundwater.modflow.finalize()

        if self.model.config["general"]["simulate_forest"]:
            assert hasattr(self.model, "plantFATE")
            for plantFATE_model in self.model.plantFATE:
                if plantFATE_model is not None:
                    plantFATE_model.finalize()

    @property
    def name(self) -> str:
        """Name of the module."""
        return "hydrology"
