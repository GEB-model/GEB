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

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from geb.geb_types import ArrayFloat32, ArrayFloat64, TwoDArrayFloat32
from geb.module import Module
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


def reconcile_hourly_groundwater_abstraction(
    available_at_start_m3: ArrayFloat64,
    available_at_end_m3: ArrayFloat64,
    accumulated_abstraction_m3: ArrayFloat64,
    modflow_available_m3: ArrayFloat64,
) -> ArrayFloat64:
    """Return a float64 daily take that does not exceed MODFLOW's source.

    Args:
        available_at_start_m3: Groundwater available before hourly withdrawals.
        available_at_end_m3: Groundwater left after hourly withdrawals.
        accumulated_abstraction_m3: Sum of the hourly withdrawal ledger.
        modflow_available_m3: Groundwater currently available in MODFLOW.

    Returns:
        Daily abstraction per cell in m3, retaining float64 precision.

    Raises:
        AssertionError: If hourly ledgers disagree or real over-abstraction occurs.
    """
    abstraction_m3 = available_at_start_m3 - available_at_end_m3
    if not np.isfinite(abstraction_m3).all() or (abstraction_m3 < 0).any():
        raise AssertionError("Invalid hourly groundwater availability ledger")
    if not np.allclose(
        accumulated_abstraction_m3, abstraction_m3, rtol=1e-5, atol=1e-3
    ):
        raise AssertionError("Daily groundwater ledger does not match availability")
    if not np.isfinite(modflow_available_m3).all():
        raise AssertionError("MODFLOW groundwater availability is not finite")
    excess_m3 = abstraction_m3 - modflow_available_m3
    tolerance_m3 = np.maximum(1e-6, np.abs(modflow_available_m3) * 1e-12)
    excessive = excess_m3 > tolerance_m3
    if excessive.any():
        raise AssertionError(
            "Hourly groundwater abstraction exceeds MODFLOW availability: "
            f"cells={np.count_nonzero(excessive)}, "
            f"max_excess_m3={excess_m3[excessive].max():.9g}"
        )
    # Clip only sub-tolerance differences between the two float64 snapshots.
    return np.minimum(abstraction_m3, modflow_available_m3)


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
            self.model.files["subgrid"]["waterbodies/subcommand_areas"], ndim=2
        )
        reservoir_command_areas = self.HRU.convert_subgrid_to_HRU(
            subgrid_command_areas,
            method="last",
        )

        waterbody_mapping = self.hydrology.waterbodies.var.waterbody_mapping
        self.HRU.var.reservoir_command_areas = np.take(
            waterbody_mapping, reservoir_command_areas, mode="clip"
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
            self.hydrology.waterbodies.n, dtype=np.float32
        )

        available_reservoir_storage_m3[self.hydrology.waterbodies.is_reservoir] = (
            self.model.agents.reservoir_operators.get_command_area_release(
                gross_irrigation_demand_m3_per_command_area
            )
        )

        available_channel_storage_m3: np.ndarray = (
            self.hydrology.routing.router.get_available_storage(
                Q=self.grid.var.discharge_in_rivers_m3_s_substep,
                river_storage_alpha=self.grid.var.river_storage_alpha,
                river_storage_beta=self.grid.var.river_storage_beta,
                maximum_abstraction_ratio=0.1,
            )
        )

        available_channel_storage_m3 = np.maximum(available_channel_storage_m3 - 100, 0)

        assert (
            available_channel_storage_m3[self.grid.var.waterbody_ids != -1] == 0.0
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

    def channel_budget_from_Q(
        self, discharge_in_rivers_m3_s_substep: ArrayFloat32
    ) -> npt.NDArray[np.float64]:
        """Calculate the shared channel abstraction budget from current discharge.

        Args:
            discharge_in_rivers_m3_s_substep: Discharge before this hour's routing
                step (m3/s).

        Returns:
            Channel water available to all water users in this hour (m3).
        """
        available_channel_storage_m3 = (
            self.hydrology.routing.router.get_available_storage(
                Q=discharge_in_rivers_m3_s_substep,
                river_storage_alpha=self.grid.var.river_storage_alpha,
                river_storage_beta=self.grid.var.river_storage_beta,
                maximum_abstraction_ratio=0.1,
            )
        )
        available_channel_storage_m3 = np.maximum(
            available_channel_storage_m3.astype(np.float64) - 100, 0
        )
        assert (
            available_channel_storage_m3[self.grid.var.waterbody_ids != -1] == 0.0
        ).all()
        return available_channel_storage_m3

    def prepare_day(self, root_depth_m: ArrayFloat32) -> None:
        """Calculate daily demands and source limits once for hourly coupling.

        Args:
            root_depth_m: Root depth per HRU (m).

        Raises:
            ValueError: If the agricultural groundwater fraction is outside [0, 1].
        """

        def household_demand_to_grid(
            water_demand_per_household_m3: ArrayFloat32,
            household_locations: TwoDArrayFloat32,
        ) -> ArrayFloat32:
            """Convert household demand to compressed grid volumes.

            Args:
                water_demand_per_household_m3: Demand per household (m3/day).
                household_locations: Household coordinates in model grid space.

            Returns:
                Household demand aggregated to compressed grid cells (m3/day).
            """
            assert (water_demand_per_household_m3 >= 0).all()
            demand_map = np.zeros(self.grid.shape, np.float32)
            demand_map = write_to_array(
                demand_map,
                water_demand_per_household_m3,
                household_locations,
                self.grid.gt,
            )
            return self.grid.compress(demand_map)

        (
            self._hourly_domestic_demand_m3,
            domestic_water_efficiency_per_household,
        ) = self.model.agents.households.water_demand(household_demand_to_grid)
        assert (domestic_water_efficiency_per_household == 1).all()
        self._hourly_domestic_efficiency = 1.0
        industry_demand_m, self._hourly_industry_efficiency = (
            self.model.agents.industry.water_demand()
        )
        livestock_demand_m, self._hourly_livestock_efficiency = (
            self.model.agents.livestock_farmers.water_demand()
        )
        self._hourly_industry_demand_m3 = (
            self.hydrology.to_grid(HRU_data=industry_demand_m, fn="weightedmean")
            * self.grid.var.cell_area
        )
        self._hourly_livestock_demand_m3 = (
            self.hydrology.to_grid(HRU_data=livestock_demand_m, fn="weightedmean")
            * self.grid.var.cell_area
        )

        (
            self._hourly_gross_demand_m3_per_field,
            reservoir_adjusted_demand,
            channel_adjusted_demand,
            groundwater_adjusted_demand,
        ) = self.model.agents.crop_farmers.get_gross_irrigation_demand_m3(root_depth_m)
        demand_by_farmer = self.model.agents.crop_farmers.field_to_farmer(
            reservoir_adjusted_demand
        )
        demand_by_waterbody = weighted_sum_per_reservoir(
            self.model.agents.crop_farmers.command_area,
            demand_by_farmer,
            min_length=self.hydrology.waterbodies.n,
        )
        demand_by_reservoir = demand_by_waterbody[
            self.hydrology.waterbodies.is_reservoir
        ]
        available_reservoir_storage_m3 = np.zeros(
            self.hydrology.waterbodies.n, dtype=np.float64
        )
        available_reservoir_storage_m3[self.hydrology.waterbodies.is_reservoir] = (
            self.model.agents.reservoir_operators.get_command_area_release(
                demand_by_reservoir
            )
        )
        # Equal-abstraction source caps depend on today's command-area release.
        self.model.agents.crop_farmers.prepare_irrigation_day(
            self._hourly_gross_demand_m3_per_field,
            reservoir_adjusted_demand,
            channel_adjusted_demand,
            groundwater_adjusted_demand,
        )
        self._hourly_available_reservoir_m3 = available_reservoir_storage_m3
        self._hourly_available_groundwater_m3 = (
            self.hydrology.groundwater.modflow.available_groundwater_m3.copy()
        )
        self._hourly_available_reservoir_m3_pre = available_reservoir_storage_m3.copy()
        self._hourly_available_groundwater_m3_pre = (
            self._hourly_available_groundwater_m3.copy()
        )
        gw_cfg = self.model.config["agent_settings"]["farmers"]["expected_utility"][
            "adaptation_well"
        ]
        gw_fraction = gw_cfg.get("groundwater_abstraction_fraction", 1.0)
        if not 0 <= gw_fraction <= 1:
            raise ValueError("groundwater_abstraction_fraction must be in [0, 1]")
        # Apply the irrigation groundwater fraction once per day, not once per hour.
        self._hourly_irrigation_groundwater_budget_m3 = (
            self._hourly_available_groundwater_m3 * gw_fraction
        )
        self._hourly_irrigation_groundwater_budget_pre_m3 = (
            self._hourly_irrigation_groundwater_budget_m3.copy()
        )

        self._hourly_channel_abstraction_m3 = np.zeros(
            self.grid.compressed_size, dtype=np.float64
        )
        self._hourly_groundwater_abstraction_m3 = np.zeros(
            self.grid.compressed_size, dtype=np.float64
        )
        self._hourly_return_flow_m = np.zeros(
            self.grid.compressed_size, dtype=np.float64
        )
        self._hourly_irrigation_withdrawal_m = np.zeros(
            self.HRU.var.cell_area.size, dtype=np.float64
        )
        self._hourly_irrigation_consumption_m = np.zeros(
            self.HRU.var.cell_area.size, dtype=np.float64
        )
        self._hourly_irrigation_return_m = np.zeros(
            self.HRU.var.cell_area.size, dtype=np.float64
        )
        self._hourly_irrigation_loss_m = np.zeros(
            self.HRU.var.cell_area.size, dtype=np.float64
        )
        self._hourly_return_flow_m3_agents = np.zeros(
            self.model.agents.crop_farmers.var.n, dtype=np.float64
        )
        self._hourly_nonfarm_loss_m3 = 0.0
        self._hourly_domestic_loss_m3 = 0.0
        self._hourly_industry_loss_m3 = 0.0
        self._hourly_livestock_loss_m3 = 0.0
        self._hourly_channel_pre_report = None
        self._hourly_last_channel_budget_post_m3 = None

    def allocate_hour(
        self, hour: int, discharge_in_rivers_m3_s_substep: ArrayFloat32
    ) -> tuple[ArrayFloat32, npt.NDArray[np.float64], ArrayFloat32]:
        """Allocate all water users for one hour using current river discharge.

        Args:
            hour: Zero-based hour of the current day.
            discharge_in_rivers_m3_s_substep: Current river discharge (m3/s).

        Returns:
            Actual net irrigation per HRU (m), channel abstraction per grid cell
            (m3), and total return flow per grid cell (m).

        Raises:
            ValueError: If ``hour`` is outside the range 0..23.
            AssertionError: If a source is over-abstracted or a ledger is unbalanced.
        """
        if not 0 <= hour < 24:
            raise ValueError("hour must be in the range 0..23")
        available_channel_storage_m3 = self.channel_budget_from_Q(
            discharge_in_rivers_m3_s_substep
        )
        available_channel_storage_m3_pre = available_channel_storage_m3.copy()
        if hour == 0:
            self._hourly_channel_pre_report = available_channel_storage_m3_pre.copy()

        domestic_demand_m3 = self._hourly_domestic_demand_m3 / 24.0
        domestic_channel_m3 = self.withdraw(
            available_channel_storage_m3, domestic_demand_m3
        )
        domestic_groundwater_m3 = self.withdraw(
            self._hourly_available_groundwater_m3, domestic_demand_m3
        )
        domestic_withdrawal_m3 = domestic_channel_m3 + domestic_groundwater_m3
        domestic_return_flow_m3 = domestic_withdrawal_m3 * (
            1 - self._hourly_domestic_efficiency
        )

        industry_demand_m3 = self._hourly_industry_demand_m3 / 24.0
        industry_channel_m3 = self.withdraw(
            available_channel_storage_m3, industry_demand_m3
        )
        industry_groundwater_m3 = self.withdraw(
            self._hourly_available_groundwater_m3, industry_demand_m3
        )
        industry_withdrawal_m3 = industry_channel_m3 + industry_groundwater_m3
        industry_return_flow_m3 = industry_withdrawal_m3 * (
            1 - self._hourly_industry_efficiency
        )

        livestock_demand_m3 = self._hourly_livestock_demand_m3 / 24.0
        livestock_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, livestock_demand_m3
        )
        livestock_return_flow_m3 = livestock_withdrawal_m3 * (
            1 - self._hourly_livestock_efficiency
        )

        irrigation_groundwater_m3 = np.minimum(
            self._hourly_available_groundwater_m3,
            self._hourly_irrigation_groundwater_budget_m3,
        ).copy()
        (
            irrigation_water_withdrawal_m,
            irrigation_water_consumption_m,
            return_flow_irrigation_m,
            irrigation_loss_to_evaporation_m,
            reservoir_abstraction_m3,
            groundwater_abstraction_m3_farmers,
            channel_abstraction_m3_farmers,
        ) = self.model.agents.crop_farmers.abstract_water_hour(
            available_channel_storage_m3=available_channel_storage_m3,
            available_groundwater_m3=irrigation_groundwater_m3,
            groundwater_depth=self.hydrology.groundwater.modflow.groundwater_depth,
            available_reservoir_storage_m3=self._hourly_available_reservoir_m3,
        )
        self.withdraw(self._hourly_available_reservoir_m3, reservoir_abstraction_m3)
        groundwater_withdrawn_m3 = self.withdraw(
            self._hourly_available_groundwater_m3,
            groundwater_abstraction_m3_farmers,
        )
        if (reservoir_abstraction_m3 > 1e-3).any() or (
            groundwater_abstraction_m3_farmers > 1e-3
        ).any():
            raise AssertionError("Hourly source request exceeded actual availability")
        self._hourly_irrigation_groundwater_budget_m3 -= groundwater_withdrawn_m3
        if (self._hourly_irrigation_groundwater_budget_m3 < -1e-3).any():
            raise AssertionError("Agricultural groundwater exceeded its daily fraction")
        self._hourly_irrigation_groundwater_budget_m3 = np.maximum(
            self._hourly_irrigation_groundwater_budget_m3, 0.0
        )

        channel_abstraction_m3 = (
            available_channel_storage_m3_pre - available_channel_storage_m3
        )
        if __debug__:
            residual_m3 = channel_abstraction_m3 - (
                domestic_channel_m3
                + industry_channel_m3
                + livestock_withdrawal_m3
                + channel_abstraction_m3_farmers
            )
            if not np.allclose(residual_m3, 0.0, rtol=1e-5, atol=1e-3):
                raise AssertionError(
                    "Hourly shared channel budget is not conserved: "
                    f"max={np.abs(residual_m3).max()} m3, "
                    f"sum={residual_m3.sum(dtype=np.float64)} m3"
                )
            if (channel_abstraction_m3 < -1e-4).any():
                raise AssertionError("Hourly channel abstraction exceeded its budget")

        groundwater_abstraction_m3 = (
            domestic_groundwater_m3 + industry_groundwater_m3 + groundwater_withdrawn_m3
        )

        return_flow_irrigation_grid_m = self.hydrology.to_grid(
            HRU_data=return_flow_irrigation_m, fn="weightedmean"
        )
        return_flow_m = (
            return_flow_irrigation_grid_m
            + (
                domestic_return_flow_m3
                + industry_return_flow_m3
                + livestock_return_flow_m3
            )
            / self.grid.var.cell_area
        )
        self._hourly_channel_abstraction_m3 += channel_abstraction_m3
        self._hourly_last_channel_budget_post_m3 = available_channel_storage_m3.copy()
        self._hourly_groundwater_abstraction_m3 += groundwater_abstraction_m3
        self._hourly_return_flow_m += return_flow_m
        self._hourly_irrigation_withdrawal_m += irrigation_water_withdrawal_m
        self._hourly_irrigation_consumption_m += irrigation_water_consumption_m
        self._hourly_irrigation_return_m += return_flow_irrigation_m
        self._hourly_irrigation_loss_m += irrigation_loss_to_evaporation_m
        domestic_loss_h = float(
            (domestic_withdrawal_m3 - domestic_return_flow_m3).sum(dtype=np.float64)
        )
        industry_loss_h = float(
            (industry_withdrawal_m3 - industry_return_flow_m3).sum(dtype=np.float64)
        )
        livestock_loss_h = float(
            (livestock_withdrawal_m3 - livestock_return_flow_m3).sum(dtype=np.float64)
        )
        self._hourly_domestic_loss_m3 += domestic_loss_h
        self._hourly_industry_loss_m3 += industry_loss_h
        self._hourly_livestock_loss_m3 += livestock_loss_h
        self._hourly_nonfarm_loss_m3 += (
            domestic_loss_h + industry_loss_h + livestock_loss_h
        )

        owner_mask = self.HRU.var.land_owners != -1
        self._hourly_return_flow_m3_agents += np.bincount(
            self.HRU.var.land_owners[owner_mask],
            weights=(
                return_flow_irrigation_m[owner_mask]
                * self.HRU.var.cell_area[owner_mask]
            ),
            minlength=self.model.agents.crop_farmers.var.n,
        )
        return (
            irrigation_water_consumption_m.astype(np.float32),
            channel_abstraction_m3,
            return_flow_m.astype(np.float32),
        )

    def finish_day(
        self,
    ) -> tuple[
        ArrayFloat64,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        float,
        ArrayFloat32,
    ]:
        """Finalize daily water-demand reports from the 24 hourly ledgers.

        Returns:
            Groundwater abstraction (m3), channel abstraction (m3), return flow
            (m), irrigation evaporation loss (m), non-farm consumption (m3), and
            net irrigation per HRU (m).

        Raises:
            ValueError: If unused command-area reservoir release exceeds tolerance.
            AssertionError: If a daily source or farmer ledger is unbalanced.
        """
        self.model.agents.crop_farmers.finish_irrigation_day()
        groundwater_abstraction_m3 = reconcile_hourly_groundwater_abstraction(
            self._hourly_available_groundwater_m3_pre,
            self._hourly_available_groundwater_m3,
            self._hourly_groundwater_abstraction_m3,
            self.hydrology.groundwater.modflow.available_groundwater_m3,
        )
        channel_abstraction_m3 = self._hourly_channel_abstraction_m3.astype(np.float32)
        return_flow = self._hourly_return_flow_m.astype(np.float32)
        irrigation_loss_to_evaporation_m = self._hourly_irrigation_loss_m.astype(
            np.float32
        )
        actual_irrigation_consumption = self._hourly_irrigation_consumption_m.astype(
            np.float32
        )
        total_water_demand_loss_m3 = self._hourly_nonfarm_loss_m3
        domestic_water_loss_m3 = self._hourly_domestic_loss_m3
        industry_water_loss_m3 = self._hourly_industry_loss_m3
        livestock_water_loss_m3 = self._hourly_livestock_loss_m3
        irrigation_water_withdrawal_m = self._hourly_irrigation_withdrawal_m.astype(
            np.float32
        )
        return_flow_irrigation_m = self._hourly_irrigation_return_m.astype(np.float32)
        gross_irrigation_demand_m3_per_field = (
            self._hourly_gross_demand_m3_per_field.astype(np.float32)
        )
        gross_irrigation_demand_m3_per_farmer = (
            self.model.agents.crop_farmers.field_to_farmer(
                gross_irrigation_demand_m3_per_field
            )
        )
        available_channel_storage_m3_pre = self._hourly_channel_pre_report.astype(
            np.float32
        )
        available_groundwater_m3_pre = self._hourly_available_groundwater_m3_pre
        available_reservoir_storage_m3_pre = (
            self._hourly_available_reservoir_m3_pre.astype(np.float32)
        )
        # There is no single end-of-day channel budget; report the last hour's remainder.
        available_channel_storage_m3 = (
            self._hourly_last_channel_budget_post_m3.astype(np.float32)
        )
        available_groundwater_m3 = self._hourly_available_groundwater_m3
        available_reservoir_storage_m3 = self._hourly_available_reservoir_m3.astype(
            np.float32
        )
        self.var.return_flow_m3_agents = self._hourly_return_flow_m3_agents.astype(
            np.float32
        )
        reservoir_storage_tolerance_m3 = 10000
        offending_mask = (
            available_reservoir_storage_m3 >= reservoir_storage_tolerance_m3
        )
        if offending_mask.any():
            raise ValueError(
                "Reservoir storage should be empty after abstraction. "
                f"Found remaining storage >= {reservoir_storage_tolerance_m3} m3: "
                f"{available_reservoir_storage_m3[offending_mask]}"
            )
        if __debug__:
            balance_check(
                name="hourly water demand daily partition",
                how="cellwise",
                influxes=[irrigation_water_withdrawal_m],
                outfluxes=[
                    actual_irrigation_consumption,
                    irrigation_loss_to_evaporation_m,
                    return_flow_irrigation_m,
                ],
                tolerance=1e-5,
            )
            expected_reservoir_m3 = (
                self._hourly_available_reservoir_m3_pre
                - self._hourly_available_reservoir_m3
            )
            farmer_reservoir_total_m3 = (
                self.model.agents.crop_farmers._hourly_reservoir_by_farmer.sum(
                    dtype=np.float64
                )
            )
            if not np.isclose(
                expected_reservoir_m3.sum(dtype=np.float64),
                farmer_reservoir_total_m3,
                rtol=1e-5,
                atol=10.0,
            ):
                raise AssertionError(
                    "Daily reservoir ledger does not match farmer reservoir withdrawals"
                )
            farmer_channel_total_m3 = (
                self.model.agents.crop_farmers._hourly_channel_by_farmer.sum(
                    dtype=np.float64
                )
            )
            if (
                farmer_channel_total_m3
                > self._hourly_channel_abstraction_m3.sum(dtype=np.float64) + 1e-2
            ):
                raise AssertionError(
                    "Farmer daily channel ledger exceeds total grid channel abstraction"
                )
            farmer_groundwater_m3 = (
                self._hourly_irrigation_groundwater_budget_pre_m3
                - self._hourly_irrigation_groundwater_budget_m3
            )
            if (farmer_groundwater_m3 < -1e-3).any() or not np.all(
                farmer_groundwater_m3
                <= self._hourly_irrigation_groundwater_budget_pre_m3 + 1e-3
            ):
                raise AssertionError("Agricultural groundwater daily cap was exceeded")
            farmer_total_m3 = (
                self.model.agents.crop_farmers._hourly_groundwater_by_farmer.sum(
                    dtype=np.float64
                )
            )
            if not np.isclose(
                farmer_groundwater_m3.sum(dtype=np.float64),
                farmer_total_m3,
                rtol=1e-5,
                atol=10.0,
            ):
                raise AssertionError("Farmer groundwater ledger differs from grid take")
        self.report(locals())
        return (
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow,
            irrigation_loss_to_evaporation_m,
            total_water_demand_loss_m3,
            actual_irrigation_consumption,
        )

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

        Raises:
            ValueError: If reservoir abstraction doesn't fully deplete calculated volume.
        """
        timer: TimingModule = TimingModule("Water demand")

        total_water_demand_loss_m3 = 0.0

        def household_demand_to_grid(
            water_demand_per_household_m3: ArrayFloat32,
            household_locations: TwoDArrayFloat32,
        ) -> ArrayFloat32:
            """Convert household water demand to grid-level water demand.

            This function is passed to the household agent's water demand function, which calculates
            the water demand per household and then uses this function to convert it to a grid-level
            water demand that can be used in the hydrological model.

            Args:
                water_demand_per_household_m3: Water demand per household in m3 per day.
                household_locations: Locations of households in the same order as the water demand array.

            Returns:
                Grid-level water demand in m3 per day.
            """
            assert (water_demand_per_household_m3 >= 0).all()

            domestic_water_demand_m3_map = np.zeros(
                self.model.hydrology.grid.shape, np.float32
            )

            domestic_water_demand_m3 = write_to_array(
                domestic_water_demand_m3_map,
                water_demand_per_household_m3,
                household_locations,
                self.model.hydrology.grid.gt,
            )
            domestic_water_demand_m3 = self.model.hydrology.grid.compress(
                domestic_water_demand_m3
            )
            return domestic_water_demand_m3

        (
            domestic_water_demand_m3,
            domestic_water_efficiency_per_household,
        ) = self.model.agents.households.water_demand(household_demand_to_grid)

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

        gross_irrigation_demand_m3_per_farmer = (
            self.model.agents.crop_farmers.field_to_farmer(
                gross_irrigation_demand_m3_per_field
            )
        )

        gross_irrigation_demand_m3_per_farmer_reservoir: npt.NDArray[np.float32] = (
            self.model.agents.crop_farmers.field_to_farmer(
                gross_irrigation_demand_m3_per_field_limit_adjusted_reservoir
            )
        )

        gross_irrigation_demand_m3_per_waterbody: npt.NDArray[np.float32] = (
            weighted_sum_per_reservoir(
                self.model.agents.crop_farmers.command_area,
                gross_irrigation_demand_m3_per_farmer_reservoir,
                min_length=self.hydrology.waterbodies.n,
            )
        )

        gross_irrigation_demand_m3_per_reservoir: npt.NDArray[np.float32] = (
            gross_irrigation_demand_m3_per_waterbody[
                self.hydrology.waterbodies.is_reservoir
            ]
        )

        assert (industry_water_demand >= 0).all()
        assert (livestock_water_demand >= 0).all()

        (
            available_channel_storage_m3,
            available_reservoir_storage_m3,
            available_groundwater_m3,
        ) = self.get_available_water(gross_irrigation_demand_m3_per_reservoir)

        # Optional diagnostic: uncomment the next line to disable all groundwater withdrawals.
        # available_groundwater_m3[:] = 0

        available_channel_storage_m3_pre = available_channel_storage_m3.copy()
        available_reservoir_storage_m3_pre = available_reservoir_storage_m3.copy()
        available_groundwater_m3_pre = available_groundwater_m3.copy()

        assert (domestic_water_efficiency_per_household == 1).all()
        domestic_water_efficiency = 1

        # 1. domestic (surface + ground)
        domestic_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, domestic_water_demand_m3
        )  # withdraw from surface water
        domestic_withdrawal_m3 += self.withdraw(
            available_groundwater_m3, domestic_water_demand_m3
        )  # withdraw from groundwater
        domestic_return_flow_m3 = domestic_withdrawal_m3 * (
            1 - domestic_water_efficiency
        )
        domestic_return_flow_m = domestic_return_flow_m3 / self.grid.var.cell_area

        domestic_water_loss_m3 = (
            domestic_withdrawal_m3 - domestic_return_flow_m3
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

        industry_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, industry_water_demand_m3
        )  # withdraw from surface water
        industry_withdrawal_m3 += self.withdraw(
            available_groundwater_m3, industry_water_demand_m3
        )  # withdraw from groundwater
        industry_return_flow_m3 = industry_withdrawal_m3 * (
            1 - industry_water_efficiency
        )
        industry_return_flow_m = industry_return_flow_m3 / self.grid.var.cell_area

        industry_water_loss_m3 = (
            industry_withdrawal_m3 - industry_return_flow_m3
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

        livestock_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, livestock_water_demand_m3
        )  # withdraw from surface water
        livestock_return_flow_m3 = livestock_withdrawal_m3 * (
            1 - livestock_water_efficiency
        )
        livestock_return_flow_m = livestock_return_flow_m3 / self.grid.var.cell_area

        livestock_water_loss_m3 = (
            livestock_withdrawal_m3 - livestock_return_flow_m3
        ).sum()
        total_water_demand_loss_m3 += livestock_water_loss_m3

        timer.finish_split("Water withdrawal")

        gw_cfg = self.model.config["agent_settings"]["farmers"]["expected_utility"][
            "adaptation_well"
        ]
        gw_fraction = gw_cfg.get("groundwater_abstraction_fraction", 1.0)
        # Restrict groundwater availability for agricultural irrigation only.
        # Domestic and industrial withdrawals above use the unscaled groundwater array.
        available_groundwater_m3_irrigation = available_groundwater_m3.copy()
        available_groundwater_m3_irrigation *= gw_fraction

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
            available_groundwater_m3=available_groundwater_m3_irrigation,
            groundwater_depth=self.hydrology.groundwater.modflow.groundwater_depth,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
        )

        self.withdraw(available_reservoir_storage_m3, reservoir_abstraction_m3_farmers)
        self.withdraw(available_groundwater_m3, groundwater_abstraction_m3_farmers)

        reservoir_storage_tolerance_m3 = 10000
        offending_mask = (
            available_reservoir_storage_m3 >= reservoir_storage_tolerance_m3
        )
        if offending_mask.any():
            raise ValueError(
                "Reservoir storage should be empty after abstraction. "
                f"Found remaining storage >= {reservoir_storage_tolerance_m3} m3: "
                f"{available_reservoir_storage_m3[offending_mask]}"
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

        actual_irrigation_consumption = irrigation_water_consumption_m

        assert (actual_irrigation_consumption + 1e-5 >= 0).all()

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
                    domestic_withdrawal_m3,
                    industry_withdrawal_m3,
                    livestock_withdrawal_m3,
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
            self.model.logger.debug(timer)

        self.var.return_flow_m3_agents = np.bincount(
            self.HRU.var.land_owners[self.HRU.var.land_owners != -1],
            weights=return_flow_irrigation_m[self.HRU.var.land_owners != -1]
            * self.HRU.var.cell_area[self.HRU.var.land_owners != -1],
        )

        self.report(locals())

        return (
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow,  # from all sources, re-added in routing
            irrigation_loss_to_evaporation_m,
            total_water_demand_loss_m3,
            actual_irrigation_consumption,
        )
