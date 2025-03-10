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

        mean_annual_inflow = self.var.active_reservoirs["average_discharge"].values

        # Make all reservoirs irrigation reservoirs. This could be changed in the future
        self.var.reservoir_purpose = np.full_like(
            5, IRRIGATION_RESERVOIR, dtype=np.int8
        )

        # Create arrays for n year moving averages of mean total inflow and mean irrigation demand.
        self.var.multi_year_monthly_total_inflow = np.tile(
            mean_annual_inflow[:, np.newaxis, np.newaxis],
            (1, 12, RESERVOIR_MEMORY_YEARS),
        )
        self.var.multi_year_monthly_total_irrigation = (
            self.var.multi_year_monthly_total_inflow
        ) * 0.25

        self.var.months_since_start_hydrological_year = np.zeros_like(
            self.storage, dtype=np.float32
        )  # Number of months in new hydrological year for each reservoir
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
        self, storage_m3, inflow_m3_s, substep, irrigation_demand_m3, water_body_id
    ):
        """
        Management module, that manages reservoir outflow per day, based on Hanasaki protocol (Hanasaki et al., 2006).
        Input variables:                                                               Units
        inflow:         channel inflow value                                           (m^3/s)
        pot_irrConsumption_m_per_cell: irrigation consumption per cell in meters       (m/day)
        irr_demand:     irrigation demand per command area                             (m^3/s)
        All other necessary variables defined in initiate_agents().
        """

        # get the historic inflow for this month, and calculate the mean for each reservoir
        mean_historic_inflow_this_month = self.var.multi_year_monthly_total_inflow[
            :, self.model.current_time.month - 1
        ].mean(axis=1)

        mean_historic_annual_inflow = self.var.multi_year_monthly_total_inflow.mean(
            axis=(1, 2)
        )

        mean_historic_annual_irrigation_demand = (
            self.var.multi_year_monthly_total_irrigation.mean(axis=(1, 2))
        )

        reservoir_release = np.full_like(storage_m3, np.nan, dtype=np.float32)
        # Calculate reservoir release for each reservoir type in m3/s. Only execute if there are reservoirs of that type.
        irrigation_reservoirs = self.var.reservoir_purpose == IRRIGATION_RESERVOIR
        if np.any(irrigation_reservoirs):
            reservoir_release[irrigation_reservoirs] = (
                self.get_irrigation_reservoir_release(
                    self.capacity[irrigation_reservoirs],
                    inflow_m3_s[irrigation_reservoirs],
                    self.var.storage_year_start[irrigation_reservoirs],
                    mean_historic_annual_inflow[irrigation_reservoirs],
                    mean_historic_inflow_this_month[irrigation_reservoirs],
                    irrigation_demand_m3[irrigation_reservoirs],
                    mean_historic_annual_irrigation_demand[irrigation_reservoirs],
                    self.var.alpha[irrigation_reservoirs],
                )
            )
        if np.any(self.var.reservoir_purpose == FLOOD_RESERVOIR):
            raise NotImplementedError()
            reservoir_release[self.var.reservoir_purpose == FLOOD_RESERVOIR] = (
                self.get_flood_control_reservoir_release(
                    self.cpa,
                    self.condF,
                    Qin_resv,
                    self.S_begin_yr,
                    self.mtifl,
                    self.alpha,
                )
            )

        # Ensure environmental flow requirements, no reservoir overflow and no negative storage.
        S_ending = Sini_resv.copy()  # create temporary final storage variable. If nothing changes, initial storage will be final storage.

        Qout_resv, S_ending = self.reservoir_water_balance(
            Qin_resv, Rres, Sini_resv, self.cpa, mtifl_month, self.alpha, self.cond_all
        )
        ##### AVERAGES CALCULATIONS #####
        # Add results of calculations to instances, to calculate yearly averages at end of year.
        self.monthly_infl += Qin_resv
        self.new_mtifl += Qin_resv
        self.new_irrmean += self.irr_demand
        self.nt_y += 1
        self.nt_m += 1

        # Set new reservoir storage for next timestep.
        self.Sini_resv = S_ending.copy()
        # if it is the last day of the month, check if it also means the start of a new operational year.
        if self.date.day == 1 and NoRoutingExecuted == 0:
            self.check_for_new_operational_year(self.date.month)

        ###### ABSTRACTION calculation + release from remaining abstraction previous day ########
        # Get release from previous day that is not abstracted and add it to release today.

        if NoRoutingExecuted == 0:
            # 1. Get the abstracted release and the remaining release from previous day.
            # 2. Convert the abstraction from the previous day from m3/day to m3/s
            # 3. Set the release for abstraction to zero, so it can be filled up in the new day.
            actual_abstraction_abm = (
                self.model.data.HRU.reservoir_abstraction_m3.copy()
            )  ####
            self.unused_release = (
                self.potential_abstraction_for_release_yesterday
                - actual_abstraction_abm
            ) * self.model.InvDtSec

            if (self.unused_release < 0).any():
                pass
                # print(
                #     "abstraction larger than potential. Total difference %:",
                #     (
                #         (
                #             self.potential_abstraction_for_release_yesterday
                #             - actual_abstraction_abm
                #         )
                #         / self.cpa
                #     )
                #     * 100,
                # )

            self.unused_release = np.maximum(self.unused_release, 0)

            self.potential_abstraction_for_release_yesterday = 0
            self.release_for_abstraction.fill(0)
        # Get potential abstraction for the farmer agents next day.
        self.release_for_abstraction += Qout_resv - mtifl_month * 0.1
        # Sum to determine the total made available for release
        self.potential_abstraction_for_release_yesterday += self.release_for_abstraction

        # Determine outflow as the base flow plus what has not abstracted
        # Determine outflow as the base flow plus what has not been abstracted
        Qout_resv_river = mtifl_month * 0.1 + (self.unused_release)

        # Ensure that this cannot be negative
        self.release_for_abstraction = np.maximum(self.release_for_abstraction, 0)
        return Qout_resv_river, self.release_for_abstraction

    def get_irrigation_reservoir_release(
        self,
        capacity,
        inflow_m3_s,
        storage_year_start,
        mean_annual_inflow,
        mean_inflow_historic_period_month,
        irrigation_demand_m3,
        mean_annual_irrigation_demand,
        alpha,
    ):
        """
        Computes release from irrigation reservoirs
        cpa = reservoir capacity                                    (m^3)
        based on selection mask
        qin = inflow                                                (m^3/s)
        Sini = initial storage                                      (m^3)
        mtifl = annual mean total annual inflow                     (m^3/s)
        irr_demand = downstream irrigation demand                   (...)
        irrmean = mean irrigation demand                            (...)
        alpha = reservoir capacity reduction factor                 (dimensionless)

        https://github.com/gutabeshu/xanthos-wm/blob/updatev1/xanthos-wm/xanthos/reservoirs/WaterManagement.py
        """
        # irrigation Reservoirs
        # initialization
        provisional_release = np.zeros_like(capacity, dtype=np.float32)

        release_coefficient = storage_year_start / (alpha * capacity)

        ratio_annual_demand_to_inflow = (
            mean_annual_irrigation_demand / mean_annual_inflow
        )

        # mean demand & annual mean inflow
        M = 0.1
        m = mean_historic_annual_irrigation_demand - ((1 - M) * mtifl_month_irr)

        cond1 = np.where(m >= 0)[0]  # m >=0 ==> dmean > = 0.5*annual mean inflow
        cond2 = np.where(m < 0)[0]  # m < 0 ==> dmean <  0.5*annual mean inflow

        demand_ratio = current_demand[cond1] / mean_demand[cond1]
        # Provisional Release where DPI > 1-M, or m > 0
        # Old code from Abeshu et al. (2023):
        # Rprovisional[cond1] = np.multiply(0.5 * mtifl_irr[cond1],
        #                                   (1 + demand_ratio))
        # Update from Shin et al. (2019) https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018WR023025
        provisional_release[cond1] = np.multiply(
            mtifl_month_irr[cond1], (M + (1 - M) * demand_ratio)
        )
        # Provisional release where DPI < 1-M or m < 0
        provisional_release[cond2] = (
            mtifl_month_irr[cond2] + current_demand[cond2] - mean_demand[cond2]
        )
        # ******  Final Release ******
        ### NEW CODE based on Shin et al. (2019) ###
        c = np.divide(cpa_irr, (mtifl_irr * 365 * 24 * 3600))
        temp_alpha = np.full(
            (self.N,), 0.85
        )  # Can be set to 0.85 like in Shin et al. or to 4, as in Biemans et al and Hanasaki.
        R_factor = np.minimum(1, (temp_alpha * c))

        temp1 = np.multiply(R_factor, Krls)
        temp2 = np.multiply(temp1, provisional_release)
        temp3 = np.multiply((1 - R_factor), qin_irr)
        Rirrg_final = temp2 + temp3
        return Rirrg_final

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

    def regulate_reservoir_outflow_stages(self, reservoirStorageM3, inflow):
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

    def get_available_water_reservoir_command_areas(self, reservoir_storage_m3):
        return self.var.reservoir_release_factors * reservoir_storage_m3

    def step(self) -> None:
        return None

    @property
    def storage(self):
        return self.model.hydrology.lakes_reservoirs.var.storage

    @property
    def capacity(self):
        return self.model.hydrology.lakes_reservoirs.var.capacity
