import calendar

import numpy as np
from .general import AgentBaseClass
from ..store import DynamicArray

from numba import njit


from ..hydrology.lakes_reservoirs import RESERVOIR

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

    def regulate_reservoir_outflow_hanasaki(
        self, inflow_m3, irrigation_demand_m3, n_routing_steps
    ):
        if inflow_m3.size == 0:
            return np.zeros_like(inflow_m3)

        current_month_index = self.model.current_time.month - 1

        # add the inflow to the multi_year_monthly_total_inflow, use the current month
        self.var.multi_year_monthly_total_inflow[:, current_month_index, 0] += inflow_m3
        # add the irrigation demand to the multi_year_monthly_total_irrigation, use the current month
        self.var.multi_year_monthly_total_irrigation[:, current_month_index, 0] += (
            irrigation_demand_m3
        )

        days_in_month = calendar.monthrange(
            self.model.current_time.year, self.model.current_time.month
        )[1]

        n_monthly_substeps = n_routing_steps * days_in_month

        month_weights = np.array(
            [31, 28.2425, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float32
        )
        month_weights = np.broadcast_to(month_weights, (5, 12))[..., np.newaxis].repeat(
            self.var.history_fill_index - 1, axis=2
        )

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
                :, current_month_index, 1 : self.var.history_fill_index
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
                    n_monthly_substeps,
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
        n_monthly_substeps,
    ):
        """
        https://github.com/gutabeshu/xanthos-wm/blob/updatev1/xanthos-wm/xanthos/reservoirs/WaterManagement.py
        """
        # Based on Shin et al. (2019)
        # https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018WR023025
        M = 0.5

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

    def get_available_water_reservoir_command_areas(self):
        return 0

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

    def step(self) -> None:
        # operational year should start after the end of the rainy season
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

        self.var.history_fill_index = max(self.var.hydrological_year_counter, 2)

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
