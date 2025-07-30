import calendar

import numpy as np
import numpy.typing as npt

from ..hydrology.lakes_reservoirs import RESERVOIR
from ..store import DynamicArray
from .general import AgentBaseClass

IRRIGATION_RESERVOIR: int = 1
FLOOD_RESERVOIR: int = 2
RESERVOIR_MEMORY_YEARS: int = 20


class ReservoirOperators(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model, agents):
        super().__init__(model)
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["reservoir_operators"]
            if "reservoir_operators" in self.model.config["agent_settings"]
            else {}
        )
        self.environmental_flow_requirement = 0.0

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        return "agents.reservoir_operators"

    def spinup(self):
        self.reservoirs = self.model.hydrology.lakes_reservoirs.var.water_body_data[
            self.model.hydrology.lakes_reservoirs.var.water_body_data["waterbody_type"]
            == 2
        ].copy()

        assert (self.reservoirs["volume_total"] > 0).all()
        self.var.active_reservoirs = self.reservoirs[
            self.reservoirs["waterbody_type"] == RESERVOIR
        ]

        self.var.reservoir_release_factors = DynamicArray(
            np.full(
                len(self.var.active_reservoirs),
                self.model.config["agent_settings"]["reservoir_operators"][
                    "max_reservoir_release_factor"
                ],
            )
        )

        self.var.dis_avg = DynamicArray(
            self.var.active_reservoirs["average_discharge"].values
        )

        # set the storage at the beginning of the year
        self.var.storage_year_start = self.storage.copy()

        # set all reservoirs to a capacity reduction factor of 0.85 (Hanasaki et al., 2006)
        # https://doi.org/10.1016/j.jhydrol.2005.11.011
        self.var.alpha = np.full_like(
            self.var.storage_year_start, 0.85, dtype=np.float32
        )

        total_monthly_inflow = (
            self.var.active_reservoirs["average_discharge"].values * 30 * 24 * 3600
        )

        # Make all reservoirs irrigation reservoirs. This could be changed in the future
        self.var.reservoir_purpose = np.full_like(
            self.storage, IRRIGATION_RESERVOIR, dtype=np.int8
        )

        # Create arrays for n year moving averages of mean total inflow and mean irrigation demand.
        self.var.multi_year_monthly_total_inflow = np.full(
            (self.storage.size, 12, RESERVOIR_MEMORY_YEARS), np.nan, dtype=np.float32
        )

        # set this hydrological year to 0 so that we can start counting
        self.var.multi_year_monthly_total_inflow[..., 0] = 0
        self.var.multi_year_monthly_total_inflow[..., 1] = total_monthly_inflow[
            :, np.newaxis
        ]

        self.var.multi_year_monthly_total_irrigation = (
            self.var.multi_year_monthly_total_inflow
        ) * 0.25

        self.var.hydrological_year_counter = (
            0  # Number of hydrological years for each reservoir
        )

    def get_command_area_release(self, gross_irrigation_demand_m3):
        assert gross_irrigation_demand_m3.size == self.storage.size

        # add the irrigation demand to the multi_year_monthly_total_irrigation, use the current month
        self.var.multi_year_monthly_total_irrigation[
            :, self.current_month_index, 0
        ] += gross_irrigation_demand_m3

        self.remaining_command_area_release = np.zeros_like(gross_irrigation_demand_m3)
        self.command_area_release_m3 = np.zeros_like(gross_irrigation_demand_m3)

        # environmental release is not used for irrigation
        usable_release_m3, environmental_release_m3 = self._get_release(
            irrigation_demand_m3=gross_irrigation_demand_m3,
            daily_substeps=1,
            enforce_minimum_usable_release_m3=False,
        )

        # limit command area release to the irrigation demand
        self.command_area_release_m3 = np.minimum(
            usable_release_m3, gross_irrigation_demand_m3
        )
        self.remaining_command_area_release = self.command_area_release_m3.copy()
        self.gross_irrigation_demand_m3 = gross_irrigation_demand_m3

        # print(
        #     "fullfillment",
        #     np.round(
        #         (self.command_area_release_m3 / self.gross_irrigation_demand_m3) * 100,
        #         decimals=1,
        #     ),
        # )

        return self.command_area_release_m3

    def track_inflow(self, inflow_m3):
        if inflow_m3.size == 0:
            return np.zeros_like(inflow_m3), np.zeros_like(inflow_m3)
        # add the inflow to the multi_year_monthly_total_inflow, use the current month
        self.var.multi_year_monthly_total_inflow[:, self.current_month_index, 0] += (
            inflow_m3
        )
        return None

    def release(
        self, daily_substeps: int, current_substep: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        command_area_release_substep: npt.NDArray[np.float64] = (
            self.command_area_release_m3 / daily_substeps
        )

        # subtract the evaporation in this timestep from the remaining evaporation
        self.remaining_command_area_release -= command_area_release_substep

        usable_release_m3, environmental_release_m3 = self._get_release(
            irrigation_demand_m3=self.gross_irrigation_demand_m3,
            daily_substeps=daily_substeps,
            enforce_minimum_usable_release_m3=True,
        )

        # main channel release is the usable release, the environmental release
        # minus the command area release. The command area release
        # is divided by the daily timesteps to get the release per timestep.
        main_channel_release: npt.NDArray[np.float64] = (
            usable_release_m3 + environmental_release_m3 - command_area_release_substep
        )
        assert (main_channel_release >= 0).all()

        if (
            current_substep + 1 == daily_substeps and __debug__
        ):  # current_substep starts counting at 0
            np.testing.assert_allclose(
                self.remaining_command_area_release, 0, atol=1e-7
            )

        np.testing.assert_allclose(
            main_channel_release + command_area_release_substep,
            usable_release_m3 + environmental_release_m3,
            atol=1e-7,
        )

        return main_channel_release, command_area_release_substep

    def _get_release(
        self, irrigation_demand_m3, daily_substeps, enforce_minimum_usable_release_m3
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if irrigation_demand_m3.size == 0:
            return np.zeros_like(irrigation_demand_m3), np.zeros_like(
                irrigation_demand_m3
            )
        days_in_month: int = calendar.monthrange(
            self.model.current_time.year, self.model.current_time.month
        )[1]

        n_monthly_substeps: int = daily_substeps * days_in_month

        month_weights: npt.NDArray[np.float32] = np.array(
            [31, 28.2425, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float32
        )
        month_weights: npt.NDArray[np.float32] = np.broadcast_to(
            month_weights, (irrigation_demand_m3.shape[0], 12)
        )[..., np.newaxis].repeat(self.var.history_fill_index - 1, axis=2)

        # get the long term inflow across all time. Do not consider the current month as
        # it is not yet complete.
        long_term_monthly_inflow_m3 = np.average(
            self.var.multi_year_monthly_total_inflow[
                ..., 1 : self.var.history_fill_index
            ],
            axis=(1, 2),
            weights=month_weights,
        )

        # get the long term inflow for this month. Do not consider the current month as
        # it is not yet complete.
        long_term_monthly_inflow_this_month_m3 = np.average(
            self.var.multi_year_monthly_total_inflow[
                :, self.current_month_index, 1 : self.var.history_fill_index
            ],
            axis=1,
        )

        long_term_monthly_irrigation_demand_m3 = np.average(
            self.var.multi_year_monthly_total_irrigation[
                ..., 1 : self.var.history_fill_index
            ],
            axis=(1, 2),
            weights=month_weights,
        )

        provisional_reservoir_release_m3 = np.full_like(
            self.storage, np.nan, dtype=np.float32
        )
        # Calculate reservoir release for each reservoir type in m3/s. Only execute if there are reservoirs of that type.
        irrigation_reservoirs = self.var.reservoir_purpose == IRRIGATION_RESERVOIR
        if np.any(irrigation_reservoirs):
            provisional_reservoir_release_m3[irrigation_reservoirs] = (
                self.get_irrigation_reservoir_release(
                    self.capacity[irrigation_reservoirs],
                    self.var.storage_year_start[irrigation_reservoirs],
                    long_term_monthly_inflow_m3[irrigation_reservoirs],
                    long_term_monthly_inflow_this_month_m3[irrigation_reservoirs],
                    irrigation_demand_m3[irrigation_reservoirs],
                    long_term_monthly_irrigation_demand_m3[irrigation_reservoirs],
                    self.var.alpha[irrigation_reservoirs],
                    n_monthly_substeps,
                )
            )

        if np.any(self.var.reservoir_purpose == FLOOD_RESERVOIR):
            raise NotImplementedError

        environmental_flow_requirement_m3 = (
            long_term_monthly_inflow_m3
            * self.environmental_flow_requirement
            / n_monthly_substeps
        )

        assert (provisional_reservoir_release_m3 >= 0).all()
        usable_release_m3, environmental_release_m3 = self._release_corrections(
            provisional_reservoir_release_m3=provisional_reservoir_release_m3,
            storage_m3=self.storage,
            capacity_m3=self.capacity,
            minimum_usable_release_m3=self.command_area_release_m3 / daily_substeps
            if enforce_minimum_usable_release_m3
            else None,
            environmental_flow_requirement_m3=environmental_flow_requirement_m3,
            alpha=self.var.alpha,
            daily_substeps=daily_substeps,
        )
        assert (usable_release_m3 >= 0).all()
        assert (environmental_release_m3 >= 0).all()
        assert (
            usable_release_m3 >= self.command_area_release_m3 / daily_substeps
        ).all()

        return usable_release_m3, environmental_release_m3

    def _release_corrections(
        self,
        provisional_reservoir_release_m3,
        storage_m3,
        capacity_m3,
        minimum_usable_release_m3,
        environmental_flow_requirement_m3,
        alpha,
        daily_substeps,
    ):
        """Adjusts the provisional reservoir release to ensure it meets environmental flow requirements, does not exceed the reservoir capacity, and maintains a minimum usable release."""
        # release is at least 10% of the mean monthly inflow (environmental flow)
        reservoir_release_m3 = np.maximum(
            provisional_reservoir_release_m3, environmental_flow_requirement_m3
        )

        assert (reservoir_release_m3 >= 0).all()

        # get provisional storage given inflow and release
        provisional_storage_m3 = storage_m3 - reservoir_release_m3

        # release any over capacity
        normal_capacity = alpha * capacity_m3
        over_capacity = provisional_storage_m3 - normal_capacity
        reservoir_release_m3 = np.where(
            over_capacity > 0,
            reservoir_release_m3 + over_capacity,
            reservoir_release_m3,
        )

        assert (reservoir_release_m3 >= 0).all()

        # storage can never drop below 10% of the capacity
        # also make sure the remaining storage is enough to provide the command area release
        # and remaining evaporative demand
        minimum_storage_m3 = 0.1 * capacity_m3 + self.remaining_command_area_release
        reservoir_release_m3_ = np.where(
            provisional_storage_m3 < minimum_storage_m3,
            np.maximum(storage_m3 - minimum_storage_m3, 0),
            reservoir_release_m3,
        )

        assert (reservoir_release_m3_ >= 0).all()

        environmental_flow_release_m3 = np.minimum(
            reservoir_release_m3_, environmental_flow_requirement_m3
        )
        usable_release_m3 = reservoir_release_m3_ - environmental_flow_release_m3
        if minimum_usable_release_m3 is not None:
            # make sure the usable release is at least the command area release
            # divided by the number of daily substeps
            usable_release_m3 = np.maximum(usable_release_m3, minimum_usable_release_m3)
        assert (
            usable_release_m3 >= self.command_area_release_m3 / daily_substeps
        ).all()
        return usable_release_m3, environmental_flow_release_m3

    def get_irrigation_reservoir_release(
        self,
        capacity,
        storage_year_start,
        long_term_monthly_inflow_m3,
        long_term_monthly_inflow_this_month_m3,
        current_irrigation_demand_m3,
        long_term_monthly_irrigation_demand_m3,
        alpha,
        n_monthly_substeps,
    ):
        """https://github.com/gutabeshu/xanthos-wm/blob/updatev1/xanthos-wm/xanthos/reservoirs/WaterManagement.py."""
        # Based on Shin et al. (2019)
        # https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018WR023025
        M = 0.1

        ratio_long_term_demand_to_inflow = (
            long_term_monthly_irrigation_demand_m3 / long_term_monthly_inflow_m3
        )  # here the units don't matter as long as they are the same
        irrigation_dominant_reservoirs = ratio_long_term_demand_to_inflow > 1.0 - M

        provisional_release = np.full_like(capacity, np.nan, dtype=np.float32)

        # equation 2a in Shin et al. (2019)
        # if the irrigation demand is not dominant, this assumes there is sufficient
        # water in the reservoir to meet the demand
        long_term_inflow_irrigation_delta_m3 = (
            long_term_monthly_inflow_m3[~irrigation_dominant_reservoirs]
            - long_term_monthly_irrigation_demand_m3[~irrigation_dominant_reservoirs]
        )
        provisional_release[~irrigation_dominant_reservoirs] = (
            long_term_inflow_irrigation_delta_m3 / n_monthly_substeps
            + current_irrigation_demand_m3[~irrigation_dominant_reservoirs]
        )

        provisional_release[irrigation_dominant_reservoirs] = (
            long_term_monthly_inflow_m3[irrigation_dominant_reservoirs]
            / n_monthly_substeps
            * (
                M
                + (1 - M)
                * current_irrigation_demand_m3[irrigation_dominant_reservoirs]
                / (
                    long_term_monthly_irrigation_demand_m3[
                        irrigation_dominant_reservoirs
                    ]
                    / n_monthly_substeps
                )
            )
        )  # equation 2b in Shin et al. (2019)

        # Targeted monthly release (Shin et al., 2019)
        ratio_capacity_to_total_inflow = capacity / (long_term_monthly_inflow_m3 * 12)
        # R in Shin et al. (2019). "As R varies from 0 to 1, the reservoir release changes
        # from run-of-the-river flow to demand-controlled release.""
        demand_controlled_release_ratio = np.minimum(
            alpha * ratio_capacity_to_total_inflow, 1.0
        )

        # kᵣₗₛ [−], "the ratio between the initial storage at the
        # beginning of the operational year (S0 [L3])
        # and the long-term target storage (αC [L3])" (Shin et al., 2019)
        release_coefficient = storage_year_start / (alpha * capacity)

        final_release = (
            demand_controlled_release_ratio * release_coefficient * provisional_release
            + (1 - demand_controlled_release_ratio)
            * long_term_monthly_inflow_this_month_m3
            / n_monthly_substeps
        )
        return final_release

    def get_flood_control_reservoir_release(
        self, cpa, cond_ppose, qin, S_begin_yr, mtifl, alpha
    ):
        """Computes release from flood control reservoirs.

        cpa = reservoir capacity                                    (m^3)
        cond_ppose = array containing irrigation reservoir cells
        based on selection mask
        qin = inflow                                                (m^3/s)
        Sini = initial storage                                      (m^3)
        mtifl = annual mean total annual inflow                     (m^3/s)
        alpha = reservoir capacity reduction factor                 (dimensionless).
        """
        # flood Reservoirs
        # initialization
        Nx = len(cond_ppose)
        Rprovisional = np.zeros(
            [
                Nx,
            ]
        )  # Provisional Release
        Rflood_final = np.zeros(
            [
                Nx,
            ]
        )  # Final Release
        # water management
        mtifl_flood = mtifl[cond_ppose]  # mean flow:  m^3/s
        cpa_flood = cpa[cond_ppose]  # capacity:   m^3
        qin_flood = qin[cond_ppose]  # mean flow:  m^3/s
        Sbeginning_ofyear = S_begin_yr[cond_ppose]  # capacity:   m^3
        # Provisional Release
        Rprovisional = mtifl_flood.copy()
        # Final Release
        # capacity & annual total infow
        c = np.divide(cpa_flood, (mtifl_flood * 365 * 24 * 3600))
        cond1 = np.where(c >= 0.5)[0]  # c = capacity/imean >= 0.5
        cond2 = np.where(c < 0.5)[0]  # c = capacity/imean < 0.5
        # c = capacity/imean >= 0.5
        Krls = np.divide(Sbeginning_ofyear, (alpha * cpa_flood))
        Rflood_final[cond1] = np.multiply(Krls[cond1], mtifl_flood[cond1])
        # c = capacity/imean < 0.5
        temp1 = (c[cond2] / 0.5) ** 2
        temp2 = np.multiply(temp1, Krls[cond2])
        temp3 = np.multiply(temp2, Rprovisional[cond2])
        temp4 = np.multiply((1 - temp1), qin_flood[cond2])
        Rflood_final[cond2] = temp3 + temp4
        return Rflood_final

    def step(self) -> None:
        # operational year should start after the end of the rainy season
        # this could also maybe be determined based on the start of the irrigation season
        # thus crop calendars
        if self.model.current_time.day == 1 and self.model.current_time.month == 10:
            # in the second year, we want to discard the default data that was estimated
            # from external sources
            if self.var.hydrological_year_counter == 1:
                self.var.multi_year_monthly_total_inflow[..., 1] = np.nan
                self.var.multi_year_monthly_total_irrigation[..., 1] = np.nan

            # in the first year, we don't want to save the data, because we don't have a full year yet
            # so no shifting should be done
            if self.var.hydrological_year_counter > 0:
                self.var.multi_year_monthly_total_inflow[..., 1:] = (
                    self.var.multi_year_monthly_total_inflow[..., 0:-1]
                )

                self.var.multi_year_monthly_total_irrigation[..., 1:] = (
                    self.var.multi_year_monthly_total_irrigation[..., 0:-1]
                )

            # always reset the counters for the next year
            self.var.multi_year_monthly_total_inflow[..., 0] = 0
            self.var.multi_year_monthly_total_irrigation[..., 0] = 0

            self.var.hydrological_year_counter += 1
            self.var.storage_year_start = self.storage.copy()

        self.var.history_fill_index = max(
            2, min(RESERVOIR_MEMORY_YEARS, self.var.hydrological_year_counter)
        )
        self.report(self, locals())

    @property
    def storage(self):
        return self.model.hydrology.lakes_reservoirs.reservoir_storage

    @storage.setter
    def storage(self, value):
        self.model.hydrology.lakes_reservoirs.reservoir_storage = value

    # @property
    # def evaporation_m3(self):
    #     return self.model.hydrology.lakes_reservoirs.potential_evaporation_per_water_body_m3_reservoir

    @property
    def capacity(self):
        return self.model.hydrology.lakes_reservoirs.reservoir_capacity

    @capacity.setter
    def capacity(self, value):
        self.model.hydrology.lakes_reservoirs.reservoir_capacity = value

    @property
    def fill_ratio(self):
        return self.storage / self.capacity

    @property
    def current_month_index(self):
        return self.model.current_time.month - 1
