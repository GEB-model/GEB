"""Water demand module for the hydrological model."""

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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from geb.module import Module
from geb.types import ArrayFloat32
from geb.workflows import TimingModule, balance_check
from geb.workflows.io import read_grid
from geb.workflows.raster import write_to_array

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


def weighted_sum_per_reservoir(
    farmer_command_area: npt.NDArray[np.int32],
    weights: npt.NDArray[np.float32],
    min_length: int,
) -> npt.NDArray[np.float32]:
    """Calculate weighted sum of values per reservoir.

    Args:
        farmer_command_area: Array mapping each farmer to a reservoir command area.
        weights: Values to be summed, weighted by the command area.
        min_length: Minimum length of the output array, typically the number of reservoirs.

    Returns:
        Weighted sum of values per reservoir.
    """
    mask: npt.NDArray[np.bool] = farmer_command_area != -1
    farmer_command_area = farmer_command_area[mask]
    weights = weights[mask]
    return np.bincount(
        farmer_command_area, weights=weights, minlength=min_length
    ).astype(weights.dtype)


class WaterDemand(Module):
    """Water demand module for the hydrological model.

    Args:
        model: The GEB model instance.
        hydrology: The hydrology submodel instance.
    """

    def __init__(self, model: GEBModel, hydrology: Hydrology) -> None:
        """Initialize the water demand module.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology submodel instance.
        """
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        """Name of the module.

        Should be identical to the path of the module in the model.

        Returns:
            The name of the module.
        """
        return "hydrology.water_demand"

    def spinup(self) -> None:
        """Perform any necessary spinup for the water demand module.

        This method initializes the reservoir command areas for the HRU grid.
        """
        subgrid_command_areas = read_grid(
            self.model.files["subgrid"]["waterbodies/subcommand_areas"]
        )
        reservoir_command_areas = self.HRU.convert_subgrid_to_HRU(
            subgrid_command_areas,
            method="last",
        )

        water_body_mapping = self.hydrology.lakes_reservoirs.var.waterbody_mapping
        self.HRU.var.reservoir_command_areas = np.take(
            water_body_mapping, reservoir_command_areas, mode="clip"
        )

    def get_available_water(
        self, gross_irrigation_demand_m3_per_command_area: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get available water from reservoirs, channels, and groundwater.

        Args:
        gross_irrigation_demand_m3_per_command_area: Gross irrigation demand in m3 per command area.

        Returns:
            Available water in m3 from channels
            Available water in m3 reservoirs
            Available water in groundwater.
        """
        available_reservoir_storage_m3: np.ndarray = np.zeros(
            self.hydrology.lakes_reservoirs.n, dtype=np.float32
        )

        available_reservoir_storage_m3[self.hydrology.lakes_reservoirs.is_reservoir] = (
            self.model.agents.reservoir_operators.get_command_area_release(
                gross_irrigation_demand_m3_per_command_area
            )
        )

        available_channel_storage_m3: np.ndarray = (
            self.hydrology.routing.router.get_available_storage(
                Q=self.grid.var.discharge_in_rivers_m3_s_substep,
                maximum_abstraction_ratio=0.1,
            )
        )

        available_channel_storage_m3 = np.maximum(available_channel_storage_m3 - 100, 0)

        assert (
            available_channel_storage_m3[self.grid.var.waterBodyID != -1] == 0.0
        ).all()

        available_groundwater_m3: np.ndarray = (
            self.hydrology.groundwater.modflow.available_groundwater_m3.copy()
        )

        return (
            available_channel_storage_m3,
            available_reservoir_storage_m3,
            available_groundwater_m3,
        )

    def withdraw(
        self, source: npt.NDArray[np.floating], demand: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Withdraw water from a source to meet a demand.

        When the source is less than the demand, all available water is withdrawn.
        When the source is more than the demand, only the demanded amount is withdrawn.

        Source and demand are expected to be in the same units (e.g., m3).

        Source and demand are updated in place.

        Args:
            source: Available water from a source (e.g., channel, reservoir, groundwater).
            demand: Water demand.

        Returns:
            Water withdrawn from the source to meet the demand.
        """
        withdrawal = np.minimum(source, demand)
        source -= withdrawal  # update in place
        demand -= withdrawal  # update in place
        return withdrawal

    def step(
        self, root_depth_m: ArrayFloat32
    ) -> tuple[
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        float,
        ArrayFloat32,
    ]:
        """Perform a single time step of the water demand module.

        Water is abstracted in the following order:
            1. Domestic water demand (surface water first, then groundwater)
            2. Industry water demand (surface water first, then groundwater)
            3. Livestock water demand (surface water only)
            4. Irrigation water demand (surface water first, then reservoir water, then groundwater)

        For the domestic and irrigation water demand, the agent-based model is used,
        while for the industry and livestock water demand, a gridded approach is used.

        Args:
            root_depth_m: Root depth in meters for each HRU.

        Returns:
            Groundwater abstraction per grid cell [m3].
            Channel abstraction per grid cell [m3].
            Return flow from all sources per grid cell [m].
                This is added to the channel flow in the routing module.
            Irrigation loss to evaporation per HRU [m].
            Total water demand loss [m3].
            The actual irrigation consumption [m].
        """
        timer: TimingModule = TimingModule("Water demand")

        total_water_demand_loss_m3 = 0.0

        (
            domestic_water_demand_per_household,
            domestic_water_efficiency_per_household,
            household_locations,
        ) = self.model.agents.households.water_demand()
        timer.finish_split("Domestic")
        industry_water_demand, industry_water_efficiency = (
            self.model.agents.industry.water_demand()
        )
        timer.finish_split("Industry")
        livestock_water_demand, livestock_water_efficiency = (
            self.model.agents.livestock_farmers.water_demand()
        )
        timer.finish_split("Livestock")

        (
            gross_irrigation_demand_m3_per_field,
            gross_irrigation_demand_m3_per_field_limit_adjusted_reservoir,
            gross_irrigation_demand_m3_per_field_limit_adjusted_channel,
            gross_irrigation_demand_m3_per_field_limit_adjusted_groundwater,
        ) = self.model.agents.crop_farmers.get_gross_irrigation_demand_m3(root_depth_m)

        gross_irrigation_demand_m3_per_farmer_reservoir: npt.NDArray[np.float32] = (
            self.model.agents.crop_farmers.field_to_farmer(
                gross_irrigation_demand_m3_per_field_limit_adjusted_reservoir
            )
        )

        gross_irrigation_demand_m3_per_water_body: npt.NDArray[np.float32] = (
            weighted_sum_per_reservoir(
                self.model.agents.crop_farmers.command_area,
                gross_irrigation_demand_m3_per_farmer_reservoir,
                min_length=self.hydrology.lakes_reservoirs.n,
            )
        )

        gross_irrigation_demand_m3_per_reservoir: npt.NDArray[np.float32] = (
            gross_irrigation_demand_m3_per_water_body[
                self.hydrology.lakes_reservoirs.is_reservoir
            ]
        )

        assert (domestic_water_demand_per_household >= 0).all()
        assert (industry_water_demand >= 0).all()
        assert (livestock_water_demand >= 0).all()

        (
            available_channel_storage_m3,
            available_reservoir_storage_m3,
            available_groundwater_m3,
        ) = self.get_available_water(gross_irrigation_demand_m3_per_reservoir)

        available_channel_storage_m3_pre = available_channel_storage_m3.copy()
        available_reservoir_storage_m3_pre = available_reservoir_storage_m3.copy()
        available_groundwater_m3_pre = available_groundwater_m3.copy()

        domestic_water_demand_m3 = np.zeros(self.model.hydrology.grid.shape, np.float32)

        domestic_water_demand_m3 = write_to_array(
            domestic_water_demand_m3,
            domestic_water_demand_per_household,
            household_locations,
            self.model.hydrology.grid.gt,
        )
        domestic_water_demand_m3 = self.model.hydrology.grid.compress(
            domestic_water_demand_m3
        )

        assert (domestic_water_efficiency_per_household == 1).all()
        domestic_water_efficiency = 1

        # 1. domestic (surface + ground)
        self.hydrology.grid.domestic_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, domestic_water_demand_m3
        )  # withdraw from surface water
        self.hydrology.grid.domestic_withdrawal_m3 += self.withdraw(
            available_groundwater_m3, domestic_water_demand_m3
        )  # withdraw from groundwater
        domestic_return_flow_m3 = self.hydrology.grid.domestic_withdrawal_m3 * (
            1 - domestic_water_efficiency
        )
        domestic_return_flow_m = domestic_return_flow_m3 / self.grid.var.cell_area

        domestic_water_loss_m3 = (
            self.hydrology.grid.domestic_withdrawal_m3 - domestic_return_flow_m3
        ).sum()
        total_water_demand_loss_m3 += domestic_water_loss_m3

        # 2. industry (surface + ground)
        industry_water_demand = self.hydrology.to_grid(
            HRU_data=industry_water_demand, fn="weightedmean"
        )
        industry_water_demand_m3 = (
            industry_water_demand * self.hydrology.grid.var.cell_area
        )
        del industry_water_demand

        self.hydrology.grid.industry_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, industry_water_demand_m3
        )  # withdraw from surface water
        self.hydrology.grid.industry_withdrawal_m3 += self.withdraw(
            available_groundwater_m3, industry_water_demand_m3
        )  # withdraw from groundwater
        industry_return_flow_m3 = self.hydrology.grid.industry_withdrawal_m3 * (
            1 - industry_water_efficiency
        )
        industry_return_flow_m = industry_return_flow_m3 / self.grid.var.cell_area

        industry_water_loss_m3 = (
            self.hydrology.grid.industry_withdrawal_m3 - industry_return_flow_m3
        ).sum()
        total_water_demand_loss_m3 += industry_water_loss_m3

        # 3. livestock (surface)
        livestock_water_demand = self.hydrology.to_grid(
            HRU_data=livestock_water_demand, fn="weightedmean"
        )
        livestock_water_demand_m3 = (
            livestock_water_demand * self.hydrology.grid.var.cell_area
        )
        del livestock_water_demand

        self.hydrology.grid.livestock_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, livestock_water_demand_m3
        )  # withdraw from surface water
        livestock_return_flow_m3 = self.hydrology.grid.livestock_withdrawal_m3 * (
            1 - livestock_water_efficiency
        )
        livestock_return_flow_m = livestock_return_flow_m3 / self.grid.var.cell_area

        livestock_water_loss_m3 = (
            self.hydrology.grid.livestock_withdrawal_m3 - livestock_return_flow_m3
        ).sum()
        total_water_demand_loss_m3 += livestock_water_loss_m3

        timer.finish_split("Water withdrawal")

        # 4. irrigation (surface + reservoir + ground)
        (
            irrigation_water_withdrawal_m,
            irrigation_water_consumption_m,
            return_flow_irrigation_m,
            irrigation_loss_to_evaporation_m,
            reservoir_abstraction_m3_farmers,
            groundwater_abstraction_m3_farmers,
        ) = self.model.agents.crop_farmers.abstract_water(
            gross_irrigation_demand_m3_per_field=gross_irrigation_demand_m3_per_field,
            gross_irrigation_demand_m3_per_field_limit_adjusted_reservoir=gross_irrigation_demand_m3_per_field_limit_adjusted_reservoir,
            gross_irrigation_demand_m3_per_field_limit_adjusted_channel=gross_irrigation_demand_m3_per_field_limit_adjusted_channel,
            gross_irrigation_demand_m3_per_field_limit_adjusted_groundwater=gross_irrigation_demand_m3_per_field_limit_adjusted_groundwater,
            available_channel_storage_m3=available_channel_storage_m3,
            available_groundwater_m3=available_groundwater_m3,
            groundwater_depth=self.hydrology.groundwater.modflow.groundwater_depth,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
        )

        self.withdraw(available_reservoir_storage_m3, reservoir_abstraction_m3_farmers)
        self.withdraw(available_groundwater_m3, groundwater_abstraction_m3_farmers)

        assert (available_reservoir_storage_m3 < 1000).all(), (
            "Reservoir storage should be empty after abstraction. "
            f"Offending values: {available_reservoir_storage_m3[available_reservoir_storage_m3 >= 50]}"
        )

        timer.finish_split("Irrigation")

        if __debug__:
            assert balance_check(
                name="water_demand_1",
                how="cellwise",
                influxes=[irrigation_water_withdrawal_m],
                outfluxes=[
                    irrigation_water_consumption_m,
                    irrigation_loss_to_evaporation_m,
                    return_flow_irrigation_m,
                ],
                tolerance=1e-5,
            )

        self.HRU.var.actual_irrigation_consumption = irrigation_water_consumption_m

        assert (self.HRU.var.actual_irrigation_consumption + 1e-5 >= 0).all()

        actual_irrigation_consumption_m3 = (
            self.HRU.var.actual_irrigation_consumption * self.HRU.var.cell_area
        )

        self.hydrology.grid.irrigation_consumption_m3 = self.hydrology.to_grid(
            HRU_data=actual_irrigation_consumption_m3,
            fn="sum",
        )

        groundwater_abstraction_m3 = (
            available_groundwater_m3_pre - available_groundwater_m3
        )
        available_groundwater_modflow = (
            self.hydrology.groundwater.modflow.available_groundwater_m3
        )
        assert (groundwater_abstraction_m3 <= available_groundwater_modflow + 1e9).all()
        groundwater_abstraction_m3 = np.minimum(
            available_groundwater_modflow, groundwater_abstraction_m3
        )
        channel_abstraction_m3 = (
            available_channel_storage_m3_pre - available_channel_storage_m3
        )

        return_flow = (
            self.hydrology.to_grid(HRU_data=return_flow_irrigation_m, fn="weightedmean")
            + domestic_return_flow_m
            + industry_return_flow_m
            + livestock_return_flow_m
        )

        if __debug__:
            assert balance_check(
                name="water_demand_1",
                how="cellwise",
                influxes=[irrigation_water_withdrawal_m],
                outfluxes=[
                    irrigation_water_consumption_m,
                    irrigation_loss_to_evaporation_m,
                    return_flow_irrigation_m,
                ],
                tolerance=1e-6,
            )
            balance_check(
                name="water_demand_2",
                how="sum",
                influxes=[],
                outfluxes=[
                    self.hydrology.grid.domestic_withdrawal_m3,
                    self.hydrology.grid.industry_withdrawal_m3,
                    self.hydrology.grid.livestock_withdrawal_m3,
                    (irrigation_water_withdrawal_m * self.HRU.var.cell_area).sum(),
                ],
                prestorages=[
                    available_channel_storage_m3_pre,
                    available_reservoir_storage_m3_pre,
                    available_groundwater_m3_pre,
                ],
                poststorages=[
                    available_channel_storage_m3,
                    available_reservoir_storage_m3,
                    available_groundwater_m3,
                ],
                tolerance=10000,
            )
        if self.model.timing:
            print(timer)

        self.report(locals())

        return (
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow,  # from all sources, re-added in routing
            irrigation_loss_to_evaporation_m,
            total_water_demand_loss_m3,
            self.HRU.var.actual_irrigation_consumption,
        )
