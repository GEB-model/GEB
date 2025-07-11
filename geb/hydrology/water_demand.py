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
import numpy.typing as npt
from honeybees.library.raster import write_to_array

from geb.hydrology.HRUs import load_grid
from geb.module import Module
from geb.workflows import TimingModule, balance_check


def weighted_sum_per_reservoir(
    farmer_command_area: npt.NDArray[np.int32],
    weights: npt.NDArray[np.float32],
    min_length: int,
) -> npt.NDArray[np.float32]:
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

    def __init__(self, model, hydrology):
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self):
        return "hydrology.water_demand"

    def spinup(self):
        reservoir_command_areas = self.HRU.compress(
            load_grid(self.model.files["subgrid"]["waterbodies/subcommand_areas"]),
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

        Parameters
        ----------
        gross_irrigation_demand_m3_per_command_area : np.ndarray
            Gross irrigation demand in m3 per command area.

        Returns:
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Available water in m3 from channels, reservoirs, and groundwater.
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
                maximum_abstraction_ratio=0.1
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

    def withdraw(self, source, demand):
        withdrawal = np.minimum(source, demand)
        source -= withdrawal  # update in place
        demand -= withdrawal  # update in place
        return withdrawal

    def step(self, potential_evapotranspiration):
        timer: TimingModule = TimingModule("Water demand")

        total_water_demand_loss_m3 = 0

        (
            domestic_water_demand_per_household,
            domestic_water_efficiency_per_household,
            household_locations,
        ) = self.model.agents.households.water_demand()
        timer.new_split("Domestic")
        industry_water_demand, industry_water_efficiency = (
            self.model.agents.industry.water_demand()
        )
        timer.new_split("Industry")
        livestock_water_demand, livestock_water_efficiency = (
            self.model.agents.livestock_farmers.water_demand()
        )
        timer.new_split("Livestock")

        gross_irrigation_demand_m3_per_field: npt.NDArray[np.float32] = (
            self.model.agents.crop_farmers.get_gross_irrigation_demand_m3(
                potential_evapotranspiration=potential_evapotranspiration,
                available_infiltration=self.HRU.var.natural_available_water_infiltration,
            )
        )

        gross_irrigation_demand_m3_per_farmer: npt.NDArray[np.float32] = (
            self.model.agents.crop_farmers.field_to_farmer(
                gross_irrigation_demand_m3_per_field
            )
        )

        gross_irrigation_demand_m3_per_water_body: npt.NDArray[np.float32] = (
            weighted_sum_per_reservoir(
                self.model.agents.crop_farmers.farmer_command_area,
                gross_irrigation_demand_m3_per_farmer,
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

        # water withdrawal
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

        total_water_demand_loss_m3 += (
            self.hydrology.grid.domestic_withdrawal_m3 - domestic_return_flow_m3
        ).sum()

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

        total_water_demand_loss_m3 += (
            self.hydrology.grid.industry_withdrawal_m3 - industry_return_flow_m3
        ).sum()

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

        total_water_demand_loss_m3 += (
            self.hydrology.grid.livestock_withdrawal_m3 - livestock_return_flow_m3
        ).sum()

        timer.new_split("Water withdrawal")

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
            available_channel_storage_m3=available_channel_storage_m3,
            available_groundwater_m3=available_groundwater_m3,
            groundwater_depth=self.hydrology.groundwater.modflow.groundwater_depth,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
        )

        self.withdraw(available_reservoir_storage_m3, reservoir_abstraction_m3_farmers)
        self.withdraw(available_groundwater_m3, groundwater_abstraction_m3_farmers)

        assert (available_reservoir_storage_m3 < 50).all(), (
            "Reservoir storage should be empty after abstraction"
        )

        timer.new_split("Irrigation")

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
                tollerance=1e-5,
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
                tollerance=1e-6,
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
                tollerance=10000,
            )
        if self.model.timing:
            print(timer)

        self.report(self, locals())

        return (
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow,  # from all sources, re-added in routing
            irrigation_loss_to_evaporation_m,
            total_water_demand_loss_m3,
        )
