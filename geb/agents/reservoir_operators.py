import numpy as np
from .general import AgentBaseClass
from ..store import DynamicArray

from numba import njit


IRRIGATION_RESERVOIR = 1
FLOOD_RESERVOIR = 2

RESERVOIR_MEMORY_YEARS = 20


@njit(cache=True)
def regulate_reservoir_outflow(
    current_storage,
    volume,
    inflow,
    minQ,
    normQ,
    nondmgQ,
    conservative_limit_ratio,
    normal_limit_ratio,
    flood_limit_ratio,
):
    day_to_sec = 1 / (24 * 60 * 60)
    outflow = np.zeros_like(current_storage)
    for i in range(current_storage.size):
        fill = current_storage[i] / volume[i]
        if fill <= conservative_limit_ratio[i] * 2:
            outflow[i] = min(minQ[i], current_storage[i] * day_to_sec)
        elif fill <= normal_limit_ratio[i]:
            outflow[i] = minQ[i] + (normQ[i] - minQ[i]) * (
                fill - 2 * conservative_limit_ratio[i]
            ) / (normal_limit_ratio[i] - 2 * conservative_limit_ratio[i])
        elif fill <= flood_limit_ratio[i]:
            outflow[i] = normQ[i] + (
                (fill - normal_limit_ratio[i])
                / (flood_limit_ratio[i] - normal_limit_ratio[i])
            ) * (nondmgQ[i] - normQ[i])
        else:
            outflow[i] = max(
                max(
                    (fill - flood_limit_ratio[i] - 0.01) * volume[i] * day_to_sec,
                    min(nondmgQ[i], np.maximum(inflow[i], normQ[i])),
                ),
                inflow[i],
            )
    return outflow


class ReservoirOperators(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["reservoir_operators"]
            if "reservoir_operators" in self.model.config["agent_settings"]
            else {}
        )

        if self.model.in_spinup:
            self.spinup()

        AgentBaseClass.__init__(self)
        super().__init__()

    def spinup(self):
        self.var = self.model.store.create_bucket("agents.reservoir_operators.var")

        self.reservoirs = self.model.hydrology.lakes_reservoirs.var.water_body_data[
            self.model.hydrology.lakes_reservoirs.var.water_body_data["waterbody_type"]
            == 2
        ].copy()

        assert (self.reservoirs["volume_total"] > 0).all()
        self.var.active_reservoirs = self.reservoirs[
            self.reservoirs["waterbody_type"] == 2
        ]

        self.var.reservoir_release_factors = DynamicArray(
            np.full(
                len(self.var.active_reservoirs),
                self.model.config["agent_settings"]["reservoir_operators"][
                    "max_reservoir_release_factor"
                ],
            )
        )

        self.var.reservoir_capacity = DynamicArray(
            self.var.active_reservoirs["volume_total"].values
        )
        self.var.flood_volume = DynamicArray(
            self.var.active_reservoirs["volume_flood"].values
        )
        self.var.dis_avg = DynamicArray(
            self.var.active_reservoirs["average_discharge"].values
        )
        self.var.norm_limit_ratio = DynamicArray(
            self.var.flood_volume / self.var.reservoir_capacity
        )
        self.var.cons_limit_ratio = DynamicArray(
            np.full(len(self.var.active_reservoirs), 0.02, dtype=np.float32)
        )
        self.var.flood_limit_ratio = DynamicArray(
            np.full(len(self.var.active_reservoirs), 1.0, dtype=np.float32)
        )

        self.var.minQC = DynamicArray(
            self.model.config["agent_settings"]["reservoir_operators"]["MinOutflowQ"]
            * self.var.dis_avg
        )
        self.var.normQC = DynamicArray(
            self.model.config["agent_settings"]["reservoir_operators"]["NormalOutflowQ"]
            * self.var.dis_avg
        )
        self.var.nondmgQC = DynamicArray(
            self.model.config["agent_settings"]["reservoir_operators"][
                "NonDamagingOutflowQ"
            ]
            * self.var.dis_avg
        )

        # set the storage at the beginning of the year
        self.var.storage_year_start = self.storage.copy()
        self.var.storage_day_start = self.storage.copy()

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
        self.var.multi_year_monthly_total_inflow = np.tile(
            total_monthly_inflow[:, np.newaxis, np.newaxis],
            (1, 12, RESERVOIR_MEMORY_YEARS),
        )
        # set this hydrological year to 0 so that we can start counting
        self.var.multi_year_monthly_total_inflow[:, :, 0] = 0

        self.var.multi_year_monthly_total_irrigation = (
            self.var.multi_year_monthly_total_inflow
        ) * 0.25

        self.var.months_since_start_hydrological_year = np.zeros_like(
            self.storage, dtype=np.float32
        )  # Number of months in new hydrological year for each reservoir
        self.var.hydrological_year_counter = np.zeros_like(
            self.storage, dtype=np.int32
        )  # Number of hydrological years for each reservoir

        self.var.irr_demand_area_yesterday = np.zeros_like(
            self.storage, dtype=np.float32
        )  # Total irrigation demand from yesterday.
        self.var.irr_demand_area = np.zeros_like(self.storage, dtype=np.float32)
        # Set variables for abstraction
        self.var.release_for_abstraction = np.zeros_like(self.storage, dtype=np.float32)

    def yearly_reservoir_calculations(self):
        """
        Function stores yearly parameters and updates at the end of a hydrological year (1st month where monthly inflow < mtifl).
        Updates are only performed for those reservoirs that enter a new hydrological year.
        """
        # Declare storage for begin of new year.
        self.S_begin_yr[self.new_yr_condition] = self.Sini_resv[
            self.new_yr_condition
        ].copy()
        # Calculate mean total inflow and irrigation demand from last year.
        temp_mtifl = (
            self.new_mtifl[self.new_yr_condition] / self.nt_y[self.new_yr_condition]
        )
        temp_irrmean = (
            self.new_irrmean[self.new_yr_condition] / self.nt_y[self.new_yr_condition]
        )
        # Calculate mean total inflow and mean irrigation demand based on 20 year moving average.
        # 1. Shift matrices 1 year to the right, and fill the first column with zeroes (new year)
        self._shift_and_reset_matrix_mask(self.mtifl_20yr, self.new_yr_condition)
        self._shift_and_reset_matrix_mask(self.irrmean_20yr, self.new_yr_condition)
        # 2. Insert the mean inflow and irrigation demand from last year.
        self.mtifl_20yr[self.new_yr_condition, 0] = temp_mtifl
        self.irrmean_20yr[self.new_yr_condition, 0] = temp_irrmean
        # 3. Calculate the new 20 year mean inflow and irrigation demand. All zeroes are ignored, so that the mean is only calculated over years that have run.
        self.mtifl = np.nanmean(self.mtifl_20yr, axis=1)
        self.irrmean = np.nanmean(self.irrmean_20yr, axis=1)
        # Reset variables for next year.
        self.new_mtifl[self.new_yr_condition] = 0
        self.new_irrmean[self.new_yr_condition] = 0
        self.n_months_in_new_year[self.new_yr_condition] = 0
        self.nt_y[self.new_yr_condition] = 0
        return

    def check_for_new_operational_year(self, month_number: int):
        # First calculate the monthly mean inflow.
        self.n_months_in_new_year += 1
        self.monthly_infl = self.monthly_infl / self.nt_m
        # Insert the monthly mean inflow into a matrix, that for each month stores the mean inflow for the last 20 years.
        # 0. Shift the 20 yrs mean inflow month matrix and  the mean monthly inflow matrix for the selected month.
        # 1. Add the monthly mean inflow to the n-1th row of the 20 year monthly mean inflow matrix, where n is the number of the month.
        # 2. Calculate the 20 year monthly mean inflow.
        # 3. Insert the new 20 year monthly mean inflow into the mean monthly inflow matrix.
        month_nr_idx = (
            month_number - 2
        )  # month number is first day of new month but, needs to be 0 indexed and the previous month.
        self._shift_and_reset_matrix_mask(self.monthly_infl_20yrs[:, month_nr_idx, :])
        self.monthly_infl_20yrs[:, month_nr_idx, 0] = self.monthly_infl
        self.mtifl_month_20yrs[:, month_nr_idx] = np.nanmean(
            self.monthly_infl_20yrs[:, month_nr_idx, :], axis=1
        )
        # # PRINT important variables.
        # print(
        #     "Long-term mean total inflow in this month is: ",
        #     self.mtifl_month_20yrs[:, month_nr_idx],
        # )
        # print(
        #     "Average total irrigation demand this year: ",
        #     np.sum(self.new_irrmean / self.nt_y),
        # )
        # Then check if this leads to a new operational year. If so, calculate yearly averages and
        # storage at begining of the year. Only check for new operational year if more than 8 months have passed. If 14th month has passed it is always a new year.
        # self.new_yr_condition = (
        #     self.mtifl_month_20yrs[:, month_nr_idx] < self.mtifl
        # ) & (self.n_months_in_new_year > 8) | (self.n_months_in_new_year == 15)
        self.new_yr_condition = self.n_months_in_new_year == 15
        if np.any(self.new_yr_condition):
            self.yearly_reservoir_calculations()
        # Reset the monthly mean inflow to zero for the next month.
        self.monthly_infl = np.zeros((self.N,), dtype="f8")
        self.nt_m = 0
        return

    def regulate_reservoir_outflow_hanasaki(
        self, inflow_m3, substep, irrigation_demand_m3, water_body_id
    ):
        """
        Management module, that manages reservoir outflow per day, based on Hanasaki protocol (Hanasaki et al., 2006).
        Input variables:                                                               Units
        inflow:         channel inflow value                                           (m^3/s)
        pot_irrConsumption_m_per_cell: irrigation consumption per cell in meters       (m/day)
        irr_demand:     irrigation demand per command area                             (m^3/s)
        All other necessary variables defined in initiate_agents().
        """
        current_month_index = self.model.current_time.month - 1

        # add the inflow to the multi_year_monthly_total_inflow, use the current month
        self.var.multi_year_monthly_total_inflow[:, current_month_index, 0] += inflow_m3
        # add the irrigation demand to the multi_year_monthly_total_irrigation, use the current month
        self.var.multi_year_monthly_total_irrigation[:, current_month_index, 0] += (
            irrigation_demand_m3
        )

        # get the long term inflow across all time. Do not consider the current month as
        # it is not yet complete.
        long_term_monthly_inflow_m3 = self.var.multi_year_monthly_total_inflow[
            ..., 1:
        ].mean(axis=(1, 2))

        # get the long term inflow for this month. Do not consider the current month as
        # it is not yet complete.
        long_term_monthly_inflow_this_month_m3 = (
            self.var.multi_year_monthly_total_inflow[:, current_month_index, 1:].mean(
                axis=1
            )
        )

        long_term_monthly_irrigation_demand_m3 = (
            self.var.multi_year_monthly_total_irrigation[..., 1:].mean(axis=(1, 2))
        )

        reservoir_release_m3 = np.full_like(self.storage, np.nan, dtype=np.float32)
        # Calculate reservoir release for each reservoir type in m3/s. Only execute if there are reservoirs of that type.
        irrigation_reservoirs = self.var.reservoir_purpose == IRRIGATION_RESERVOIR
        if np.any(irrigation_reservoirs):
            reservoir_release_m3[irrigation_reservoirs] = (
                self.get_irrigation_reservoir_release(
                    self.capacity[irrigation_reservoirs],
                    self.var.storage_year_start[irrigation_reservoirs],
                    long_term_monthly_inflow_m3[irrigation_reservoirs],
                    long_term_monthly_inflow_this_month_m3[irrigation_reservoirs],
                    irrigation_demand_m3[irrigation_reservoirs],
                    long_term_monthly_irrigation_demand_m3[irrigation_reservoirs],
                    self.var.alpha[irrigation_reservoirs],
                )
            )
        if np.any(self.var.reservoir_purpose == FLOOD_RESERVOIR):
            raise NotImplementedError

        assert (reservoir_release_m3 >= 0).all()
        reservoir_release_m3 = self.release_corrections(
            inflow_m3,
            reservoir_release_m3,
            self.storage,
            self.capacity,
            long_term_monthly_inflow_m3,
            self.var.alpha,
        )
        assert (reservoir_release_m3 >= 0).all()

        self.storage = self.storage - reservoir_release_m3 + inflow_m3
        assert (self.storage >= 0).all()

        return reservoir_release_m3

    def release_corrections(
        self,
        inflow_m3,
        reservoir_release_m3,
        storage_m3,
        capacity_m3,
        long_term_monthly_inflow_m3,
        alpha,
    ):
        # release is at least 10% of the mean monthly inflow (environmental flow)
        reservoir_release_m3 = np.minimum(
            reservoir_release_m3, long_term_monthly_inflow_m3 * 0.1
        )

        assert (reservoir_release_m3 >= 0).all()

        # get provisional storage given inflow and release
        provisional_storage_m3 = storage_m3 + (inflow_m3 - reservoir_release_m3)

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
        minimum_storage_m3 = 0.1 * capacity_m3
        reservoir_release_m3 = np.where(
            provisional_storage_m3 < minimum_storage_m3,
            np.maximum(storage_m3 + inflow_m3 - minimum_storage_m3, 0),
            reservoir_release_m3,
        )

        assert (reservoir_release_m3 >= 0).all()
        assert (storage_m3 + inflow_m3 - reservoir_release_m3 >= 0).all()

        return reservoir_release_m3

    def get_irrigation_reservoir_release(
        self,
        capacity,
        storage_year_start,
        long_term_monthly_inflow_m3,
        long_term_monthly_inflow_this_month_m3,
        current_irrigation_demand_m3,
        long_term_monthly_irrigation_demand_m3,
        alpha,
    ):
        """
        https://github.com/gutabeshu/xanthos-wm/blob/updatev1/xanthos-wm/xanthos/reservoirs/WaterManagement.py
        """
        # Based on Shin et al. (2019)
        # https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018WR023025
        M = 0.5

        n_monthly_substeps = 24 * 30  # 24 routing steps in 30 days

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
            long_term_monthly_inflow_this_month_m3[irrigation_dominant_reservoirs]
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
        """
        Computes release from flood control reservoirs
        cpa = reservoir capacity                                    (m^3)
        cond_ppose = array containing irrigation reservoir cells
        based on selection mask
        qin = inflow                                                (m^3/s)
        Sini = initial storage                                      (m^3)
        mtifl = annual mean total annual inflow                     (m^3/s)
        alpha = reservoir capacity reduction factor                 (dimensionless)
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

    def reservoir_water_balance(self, Qin, Qout, Sin, cpa, mtifl, alpha, cond_all):
        """Re-adjusts release to ensure minimal environmental flow, and prevent overflow and negative storage;
        and computes the storage level after release for all reservoir types.
        Qin = inflow into reservoir (m3/s)
        Qout = release from reservoir (m3/s)
        Sin = initial storage (m^3)
        cpa = reservoir capacity (m^3)
        mtifl = annual mean total annual inflow (m^3/s)
        alpha = reservoir capacity reduction factor
        dt = time step (hour)
        cond_all = mask selecting all reservoir types.
        """
        # inputs
        Qin_ = Qin[cond_all]
        Qout_ = Qout[cond_all]
        Sin_ = Sin[cond_all]
        cpa_ = cpa[cond_all]
        mtifl_month = mtifl[cond_all]
        dt = 3600  # convert from seconds to hour
        # final storage and release initialization
        Nx = len(cond_all)
        Rfinal = np.zeros(
            [
                Nx,
            ]
        )  # final release
        Sfinal = np.zeros(
            [
                Nx,
            ]
        )  # final storage
        ###### WATER BALANCE CALCULATIONS ######
        # 1. Environmental flow: Qout must be at least 10% of the mean monthly inflow.
        diff_rt = Qout_ - (mtifl_month * 0.1)
        indx_rt = np.where(diff_rt < 0)[0]
        Qout_[indx_rt] = 0.1 * mtifl_month[indx_rt]
        # Check for change in storage.
        dsdt_resv = (Qin_ - Qout_) * dt
        Stemp = Sin_ + dsdt_resv
        # 2. condition a : storage > capacity | Storage can not be larger than reservoir capacity, so remove overflow.
        Sa = Stemp > (alpha * cpa_)
        if Sa.any():
            Sfinal[Sa] = alpha[Sa] * cpa_[Sa]  # Storage is set at full level.
            Rspill = (
                (Stemp[Sa] - (alpha[Sa] * cpa_[Sa])) / dt
            )  # Spilling is determined as earlier storage level minus maximum capacity.
            Rfinal[Sa] = (
                Qout_[Sa] + Rspill
            )  # Calculate new release rate: Earlier determined outflow + spilling.
        # 3. condition b : storage <= 0 | StSorage can not be smaller than 10% of reservoir capacity, so remove the outflow that results in too low capacity.
        Sb = Stemp < (0.1 * self.cpa * alpha)
        if Sb.any():
            Sfinal[Sb] = (
                0.1 * self.cpa[Sb] * alpha[Sb]
            )  # Final storage is then 10% of effective capacity.
            Rfinal[Sb] = (
                ((Stemp[Sb] - Sfinal[Sb]) / dt) + Qin_[Sb]
            )  # Add negative storage to inflow, to lower final outflow and prevent negative storage.
            print("Storage before water balance correction:", Stemp)
            print("Storage after water balance correction:", Sfinal[Sb])
        # 4. condition c : Storage > 0 & Storage < total capacity | the reverse of condition a and b.
        Sc = (Stemp > 0) & (Stemp <= alpha * cpa_)
        if Sc.any():
            # Assign temporary storage and outflow as final storage and release, for all reservoirs that do have proper storage levels.
            Sfinal[Sc] = Stemp[Sc]
            Rfinal[Sc] = Qout_[Sc]
        return Rfinal, Sfinal

    def get_available_water_reservoir_command_areas(self, reservoir_storage_m3):
        self.date = self.model.current_time
        if "ruleset" in self.config and self.config["ruleset"] == "new_module":
            self.available_water_for_abstraction = self.release_for_abstraction.copy()
        elif "ruleset" in self.config and self.config["ruleset"] == "no_dams":
            self.available_water_for_abstraction = 0
        else:  # Use the old module
            self.available_water_for_abstraction = (
                self.reservoir_release_factors * reservoir_storage_m3
            )
        return self.available_water_for_abstraction * 3600

    def regulate_reservoir_outflow_staged(self, reservoirStorageM3, inflow):
        """Regulate the outflow of the reservoirs.

        Args:
            reservoirStorageM3C: The current storage of the reservoirs in m3.
            inflowC: The inflow of the reservoirs (m/s).
            waterBodyIDs: The IDs of the water bodies.
            delta_t: The time step in seconds.

        Returns:
            The outflow of the reservoirs (m/s).

        """
        # make outflow same as inflow for a setting without a reservoir
        if "ruleset" in self.config and self.config["ruleset"] == "no-human-influence":
            return inflow.copy()

        reservoir_outflow = regulate_reservoir_outflow(
            reservoirStorageM3,
            self.var.reservoir_capacity.data,
            inflow,
            self.var.minQC.data,
            self.var.normQC.data,
            self.var.nondmgQC.data,
            self.var.cons_limit_ratio.data,
            self.var.norm_limit_ratio.data,
            self.var.flood_limit_ratio.data,
        )
        assert (reservoir_outflow >= 0).all()

        return reservoir_outflow

    def get_available_water_reservoir_command_areas(self):
        return self.var.reservoir_release_factors * self.storage

    def step(self) -> None:
        if self.model.current_time.day == 1 and self.model.current_time.month == 6:
            self.var.new_hydrological_year = np.ones_like(self.storage, dtype=bool)
            self.var.hydrological_year_counter += 1
        else:
            self.var.new_hydrological_year = np.zeros_like(self.storage, dtype=bool)

    @property
    def storage(self):
        return self.model.hydrology.lakes_reservoirs.reservoir_storage

    @storage.setter
    def storage(self, value):
        self.model.hydrology.lakes_reservoirs.reservoir_storage = value

    @property
    def capacity(self):
        return self.model.hydrology.lakes_reservoirs.reservoir_capacity

    @capacity.setter
    def capacity(self, value):
        self.model.hydrology.lakes_reservoirs.reservoir_capacity = value
