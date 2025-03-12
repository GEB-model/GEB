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
from .soil import (
    get_root_ratios,
    get_maximum_water_content,
    get_critical_water_level,
    get_available_water,
    get_fraction_easily_available_soil_water,
    get_crop_group_number,
)
from .landcover import PADDY_IRRIGATED, NON_PADDY_IRRIGATED
from .lakes_reservoirs import RESERVOIR
from geb.HRUs import load_grid

from geb.workflows import TimingModule, balance_check


class WaterDemand:
    def __init__(self, model, hydrology):
        self.model = model
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    def spinup(self):
        reservoir_command_areas = self.HRU.compress(
            load_grid(
                self.model.files["subgrid"]["routing/lakesreservoirs/subcommand_areas"]
            ),
            method="last",
        )

        water_body_mapping = self.hydrology.lakes_reservoirs.var.waterbody_mapping
        self.HRU.var.reservoir_command_areas = np.take(
            water_body_mapping, reservoir_command_areas, mode="clip"
        )

    def get_potential_irrigation_consumption(self, potential_evapotranspiration):
        """Calculate the potential irrigation consumption. Not that consumption
        is not the same as withdrawal. Consumption is the amount of water that
        is actually used by the farmers, while withdrawal is the amount of water
        that is taken from the source. The difference is the return flow."""
        # a function of cropKC (evaporation and transpiration) and available water see Wada et al. 2014 p. 19
        paddy_irrigated_land = np.where(self.HRU.var.land_use_type == PADDY_IRRIGATED)

        paddy_level = self.HRU.full_compressed(np.nan, dtype=np.float32)
        paddy_level[paddy_irrigated_land] = (
            self.HRU.var.topwater[paddy_irrigated_land]
            + self.HRU.var.natural_available_water_infiltration[paddy_irrigated_land]
        )

        nonpaddy_irrigated_land = np.where(
            self.HRU.var.land_use_type == NON_PADDY_IRRIGATED
        )[0]

        # load crop group number
        crop_group_number = get_crop_group_number(
            self.HRU.var.crop_map,
            self.model.agents.crop_farmers.var.crop_data["crop_group_number"].values,
            self.HRU.var.land_use_type,
            self.HRU.var.natural_crop_groups,
        )

        # p is between 0 and 1 => if p =1 wcrit = wwp, if p= 0 wcrit = wfc
        p = get_fraction_easily_available_soil_water(
            crop_group_number[nonpaddy_irrigated_land],
            potential_evapotranspiration[nonpaddy_irrigated_land],
        )

        root_ratios = get_root_ratios(
            self.HRU.var.root_depth[nonpaddy_irrigated_land],
            self.HRU.var.soil_layer_height[:, nonpaddy_irrigated_land],
        )

        max_water_content = self.HRU.full_compressed(np.nan, dtype=np.float32)
        max_water_content[nonpaddy_irrigated_land] = (
            get_maximum_water_content(
                self.HRU.var.wfc[:, nonpaddy_irrigated_land],
                self.HRU.var.wwp[:, nonpaddy_irrigated_land],
            )
            * root_ratios
        ).sum(axis=0)

        critical_water_level = self.HRU.full_compressed(np.nan, dtype=np.float32)
        critical_water_level[nonpaddy_irrigated_land] = (
            get_critical_water_level(
                p,
                self.HRU.var.wfc[:, nonpaddy_irrigated_land],
                self.HRU.var.wwp[:, nonpaddy_irrigated_land],
            )
            * root_ratios
        ).sum(axis=0)

        readily_available_water = self.HRU.full_compressed(np.nan, dtype=np.float32)
        readily_available_water[nonpaddy_irrigated_land] = (
            get_available_water(
                self.HRU.var.w[:, nonpaddy_irrigated_land],
                self.HRU.var.wwp[:, nonpaddy_irrigated_land],
            )
            * root_ratios
        ).sum(axis=0)

        # first 2 soil layers to estimate distribution between runoff and infiltration
        topsoil_w_nonpaddy_irrigated_land = self.HRU.var.w[:2, nonpaddy_irrigated_land]
        topsoil_ws_nonpaddy_irrigated_land = self.HRU.var.ws[
            :2, nonpaddy_irrigated_land
        ]

        assert (
            topsoil_w_nonpaddy_irrigated_land <= topsoil_ws_nonpaddy_irrigated_land
        ).all()

        soil_water_storage = topsoil_w_nonpaddy_irrigated_land.sum(axis=0)
        soil_water_storage_cap = topsoil_ws_nonpaddy_irrigated_land.sum(axis=0)

        relative_saturation = soil_water_storage / soil_water_storage_cap
        assert (relative_saturation <= 1 + 1e-7).all(), (
            "Relative saturation should always be <= 1"
        )
        relative_saturation[relative_saturation > 1] = 1

        relative_saturation[relative_saturation > 1] = 1

        satAreaFrac = (
            1
            - (1 - relative_saturation)
            ** self.HRU.var.arnoBeta[nonpaddy_irrigated_land]
        )
        satAreaFrac = np.maximum(np.minimum(satAreaFrac, 1.0), 0.0)

        store = soil_water_storage_cap / (
            self.HRU.var.arnoBeta[nonpaddy_irrigated_land] + 1
        )
        potBeta = (
            self.HRU.var.arnoBeta[nonpaddy_irrigated_land] + 1
        ) / self.HRU.var.arnoBeta[nonpaddy_irrigated_land]
        potential_infiltration_capacity = self.HRU.full_compressed(
            np.nan, dtype=np.float32
        )
        potential_infiltration_capacity[nonpaddy_irrigated_land] = store - store * (
            1 - (1 - satAreaFrac) ** potBeta
        )

        assert not (
            np.any(np.isnan(potential_infiltration_capacity[nonpaddy_irrigated_land]))
            and not np.all(
                np.isnan(potential_infiltration_capacity[nonpaddy_irrigated_land])
            )
        ), "Error: Some values in readily_available_water are NaN, but not all."

        return (
            paddy_level,
            readily_available_water,
            critical_water_level,
            max_water_content,
            potential_infiltration_capacity,
        )

    def get_available_water(self):
        assert (
            self.hydrology.lakes_reservoirs.var.waterBodyIDC.size
            == self.hydrology.lakes_reservoirs.var.storage.size
        )
        assert (
            self.hydrology.lakes_reservoirs.var.waterBodyIDC.size
            == self.hydrology.lakes_reservoirs.var.water_body_type.size
        )
        available_reservoir_storage_m3 = np.zeros_like(
            self.hydrology.lakes_reservoirs.var.storage
        )
        available_reservoir_storage_m3[
            self.hydrology.lakes_reservoirs.var.water_body_type == RESERVOIR
        ] = self.model.agents.reservoir_operators.get_available_water_reservoir_command_areas()
        return (
            self.grid.var.river_storage_m3.copy(),
            available_reservoir_storage_m3,
            self.hydrology.groundwater.modflow.available_groundwater_m3.copy(),
        )

    def withdraw(self, source, demand):
        withdrawal = np.minimum(source, demand)
        source -= withdrawal  # update in place
        demand -= withdrawal  # update in place
        return withdrawal

    def step(self, potential_evapotranspiration):
        timer = TimingModule("Water demand")

        domestic_water_demand, domestic_water_efficiency = (
            self.model.agents.households.water_demand()
        )
        timer.new_split("Domestic")
        industry_water_demand, industry_water_efficiency = (
            self.model.agents.industry.water_demand()
        )
        timer.new_split("Industry")
        livestock_water_demand, livestock_water_efficiency = (
            self.model.agents.livestock_farmers.water_demand()
        )
        timer.new_split("Livestock")
        (
            paddy_level,
            readily_available_water,
            critical_water_level,
            max_water_content,
            potential_infiltration_capacity,
        ) = self.get_potential_irrigation_consumption(potential_evapotranspiration)

        assert (domestic_water_demand >= 0).all()
        assert (industry_water_demand >= 0).all()
        assert (livestock_water_demand >= 0).all()

        (
            available_channel_storage_m3,
            available_reservoir_storage_m3,
            available_groundwater_m3,
        ) = self.get_available_water()

        available_channel_storage_m3_pre = available_channel_storage_m3.copy()
        available_reservoir_storage_m3_pre = available_reservoir_storage_m3.copy()
        available_groundwater_m3_pre = available_groundwater_m3.copy()

        # water withdrawal
        # 1. domestic (surface + ground)
        domestic_water_demand = self.hydrology.to_grid(
            HRU_data=domestic_water_demand, fn="weightedmean"
        )
        domestic_water_demand_m3 = self.hydrology.grid.MtoM3(domestic_water_demand)
        del domestic_water_demand

        self.hydrology.grid.domestic_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, domestic_water_demand_m3
        )  # withdraw from surface water
        self.hydrology.grid.domestic_withdrawal_m3 += self.withdraw(
            available_groundwater_m3, domestic_water_demand_m3
        )  # withdraw from groundwater
        domestic_return_flow_m = self.hydrology.grid.M3toM(
            self.hydrology.grid.domestic_withdrawal_m3 * (1 - domestic_water_efficiency)
        )

        # 2. industry (surface + ground)
        industry_water_demand = self.hydrology.to_grid(
            HRU_data=industry_water_demand, fn="weightedmean"
        )
        industry_water_demand_m3 = self.hydrology.grid.MtoM3(industry_water_demand)
        del industry_water_demand

        self.hydrology.grid.industry_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, industry_water_demand_m3
        )  # withdraw from surface water
        self.hydrology.grid.industry_withdrawal_m3 += self.withdraw(
            available_groundwater_m3, industry_water_demand_m3
        )  # withdraw from groundwater
        industry_return_flow_m = self.hydrology.grid.M3toM(
            self.hydrology.grid.industry_withdrawal_m3 * (1 - industry_water_efficiency)
        )

        # 3. livestock (surface)
        livestock_water_demand = self.hydrology.to_grid(
            HRU_data=livestock_water_demand, fn="weightedmean"
        )
        livestock_water_demand_m3 = self.hydrology.grid.MtoM3(livestock_water_demand)
        del livestock_water_demand

        self.hydrology.grid.livestock_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, livestock_water_demand_m3
        )  # withdraw from surface water
        livestock_return_flow_m = self.hydrology.grid.M3toM(
            self.hydrology.grid.livestock_withdrawal_m3
            * (1 - livestock_water_efficiency)
        )
        timer.new_split("Water withdrawal")

        # 4. irrigation (surface + reservoir + ground)
        (
            irrigation_water_withdrawal_m,
            irrigation_water_consumption_m,
            return_flow_irrigation_m,
            irrigation_loss_to_evaporation_m,
        ) = self.model.agents.crop_farmers.abstract_water(
            cell_area=self.HRU.var.cell_area,
            paddy_level=paddy_level,
            readily_available_water=readily_available_water,
            critical_water_level=critical_water_level,
            max_water_content=max_water_content,
            potential_infiltration_capacity=potential_infiltration_capacity,
            available_channel_storage_m3=available_channel_storage_m3,
            available_groundwater_m3=available_groundwater_m3,
            groundwater_depth=self.hydrology.groundwater.modflow.groundwater_depth,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
            command_areas=self.HRU.var.reservoir_command_areas,
        )
        timer.new_split("Irrigation")

        if __debug__:
            balance_check(
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

        self.hydrology.grid.irrigation_consumption_m3 = self.hydrology.to_grid(
            HRU_data=self.HRU.MtoM3(self.HRU.var.actual_irrigation_consumption),
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

        reservoir_abstraction_m3 = (
            available_reservoir_storage_m3_pre - available_reservoir_storage_m3
        )
        assert (
            self.hydrology.lakes_reservoirs.var.water_body_type[
                np.where(reservoir_abstraction_m3 > 0)
            ]
            == RESERVOIR
        ).all(), "Reservoir abstraction should only be from reservoirs"

        # Abstract water from reservoir
        self.hydrology.lakes_reservoirs.var.storage -= reservoir_abstraction_m3

        return_flow = (
            self.hydrology.to_grid(HRU_data=return_flow_irrigation_m, fn="weightedmean")
            + domestic_return_flow_m
            + industry_return_flow_m
            + livestock_return_flow_m
        )

        if __debug__:
            balance_check(
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
                    self.HRU.var.cell_area,
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

        return (
            groundwater_abstraction_m3,
            channel_abstraction_m3 / self.hydrology.grid.var.cell_area,
            return_flow,  # from all sources, re-added in routing
            irrigation_loss_to_evaporation_m,
        )
