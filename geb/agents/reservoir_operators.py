# -*- coding: utf-8 -*-
from honeybees.agents import AgentBaseClass
import os
import numpy as np
import pandas as pd
from .farmers import Farmers
from pathlib import Path
from datetime import datetime, timedelta



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
        AgentBaseClass.__init__(self)
        df = pd.read_csv(
            self.model.model_structure["table"][
                "routing/lakesreservoirs/basin_lakes_data"
            ]
        ).set_index("waterbody_id")

        self.reservoirs = df[df["waterbody_type"] == 2].copy()



    def initiate_agents(self, waterBodyIDs):
        assert (self.reservoirs["volume_total"] > 0).all()
        self.active_reservoirs = self.reservoirs.loc[waterBodyIDs]

        np.save(
            self.model.report_folder / "active_reservoirs_waterBodyIDs.npy",
            waterBodyIDs,
        )

        self.reservoir_release_factors = np.full(
            len(self.active_reservoirs),
            self.model.config["agent_settings"]["reservoir_operators"][
                "max_reservoir_release_factor"
            ],
        )

        self.reservoir_volume = self.active_reservoirs["volume_total"].values
        self.flood_volume = self.active_reservoirs["volume_flood"].values
        self.dis_avg = self.active_reservoirs["average_discharge"].values # average 1971-2000

        self.cons_limit_ratio = 0.02
        self.norm_limit_ratio = self.flood_volume / self.reservoir_volume
        self.flood_limit_ratio = 1
        self.norm_flood_limit_ratio = self.norm_limit_ratio + 0.5 * (
            self.flood_limit_ratio - self.norm_limit_ratio
        )

        self.minQC = (
            self.model.config["agent_settings"]["reservoir_operators"]["MinOutflowQ"]
            * self.dis_avg
        )
        self.normQC = (
            self.model.config["agent_settings"]["reservoir_operators"]["NormalOutflowQ"]
            * self.dis_avg
        )
        self.nondmgQC = (
            self.model.config["agent_settings"]["reservoir_operators"][
                "NonDamagingOutflowQ"
            ]
            * self.dis_avg
        )

        """
        Initiate variables for new reservoir module

        cpa:            reservoir capacity                              (m^3)
        S_begin_yr:     storage at beginning of the year                (m^3). Set at 0.5*cpa at t == 0.
        Sini_resv:      reservoir storage at start of day               (m^3). Set at 0.5*cpa at t == 0.
        alpha:          reservoir capacity reduction factor.            (dimensionless)
        irrmean:        mean irrigation demand                          (m^3/s)
        new_irrmean:    new mean irrigation demand after every year     (m3/s)
        mtifl:          annual mean total annual inflow into reservoir  (m^3/s)
        new_mtifl:      new mean total inflow after every year          (m3/s)
        res_ppose:      array with each reservoir purpose               (1 == irrigation; 2 == flood control)
        """
        
        #### Variables taken from reservoir csv sheet, or inferred from there. ###
        self.cpa = self.active_reservoirs["volume_total"].values
        self.N = self.cpa.shape[0]                                                  # Number of reservoirs. All arrays will have this dimension: (N,)
        self.S_begin_yr = 0.5 * self.cpa.copy()
        self.Sini_resv = self.S_begin_yr.copy()
        self.alpha = np.full((self.N,), 0.85)                                       # set all reservoirs to a capacity reduction factor of 0.85 (Biemans et al., 2011).
        self.mtifl = np.full((self.N,), self.active_reservoirs["average_discharge"].values)
        self.irrmean = np.full((self.N,), 0.1 * self.mtifl)                         # start with estimated mean irrigation demand of 0.5 of mtifl.        

        ### FUTURE CODE: WHEN RESERVOIR PURPOSE DATA IS AVAILABLE. ###
        # self.res_ppose = self.active_reservoirs["purpose"].values 
        
        ### TEMPORARY CODE: Make all reservoirs irrigation reservoirs ###
        self.res_ppose = np.full((self.N,), 1)                                      # temporarlily set all reservoirs to irrigation reservoirs, because data is still lacking.
        
        ### Create selection masks for irrigation and flood reservoirs. ###
        self.condI = (self.res_ppose == 1)                                          # irrigation reservoirs
        self.condF = (self.res_ppose == 2)                                          # flood reservoirs
        self.cond_all = (self.res_ppose > 0)

        """Calls functions to initialize all agent attributes, including their locations. Then, crops are initially planted."""
        # If initial conditions based on spinup period need to be loaded, load them. Otherwise, generate them.
        if self.model.load_initial_data:
            folder = os.path.join(self.model.initial_conditions_folder, 'reservoir_operators')
            for fn in os.listdir(folder):
                if not fn.startswith("reservoir_operators."):
                    continue
                attribute = fn.split(".")[1]
                fp = os.path.join(folder, fn)
                values = np.load(fp)["data"]
                setattr(self, attribute, values)

        else:
            # Create arrays for 20 year moving averages of mean total inflow and mean irrigation demand.
            self.mtifl_20yr = np.full((self.N, 20), np.nan, dtype='f8')
            self.irrmean_20yr = np.full((self.N, 20), np.nan, dtype='f8')
            self.mtifl_month_20yrs = np.full((self.N, 12), 1, dtype='f8')               # Stores 20 year moving average of mean monthly inflow for every month for every reservoir.              
            self.monthly_infl_20yrs = np.full((self.N, 12, 20), np.nan, dtype='f8')     # Stores mean inflow for every month for every reservoir for the last 20 years.

            # Set starting values for moving averages
            self.mtifl_20yr[:, 0] = self.mtifl*0.1
            self.irrmean_20yr[:, 0] = self.irrmean
            self.mtifl_month_20yrs *= self.mtifl.reshape(-1,1)*0.25                     # Fill mean monthly inflow with 25% average inflow of each reservoir, taken from database.
        
        ### Initiate variables for monthly and yearly calculations. ###
        self.new_mtifl = np.zeros((self.N,), dtype='f8')                                # Temporary new mean total annual inflow
        self.new_irrmean = np.zeros((self.N,), dtype='f8')                              # Temporary new mean irrigation demand
        
        self.monthly_infl = np.zeros((self.N,), dtype='f8')                             # Total inflow in that month, used to calculate mean inflow for that month.
        self.n_months_in_new_year = np.zeros((self.N,), dtype='f8')                     # Number of months in new hydrological year for each reservoir.
        
        self.nt_y = np.zeros((self.N,), dtype = 'f8')                                   # number of routing timesteps in hydrological year for each reservoir
        self.nt_m = 0                                                                   # number of routing timesteps in a month
        
        self.total_irr_demand_area_yesterday = np.zeros((self.N,), dtype='f8')          # Total irrigation demand from yesterday.

        # Set variables for abstraction
        self.release_for_abstraction = np.zeros((self.N,), dtype='f8')
        

        return self

    def yearly_reservoir_calculations(self):
        """
        Function stores yearly parameters and updates at the end of a hydrological year (1st month where monthly inflow < mtifl).
        Updates are only performed for those reservoirs that enter a new hydrological year.
        """
        # Declare storage for begin of new year.
        self.S_begin_yr[self.new_yr_condition] = self.Sini_resv[self.new_yr_condition].copy()

        # Calculate mean total inflow and irrigation demand from last year.
        temp_mtifl = self.new_mtifl[self.new_yr_condition] / self.nt_y[self.new_yr_condition]
        temp_irrmean = self.new_irrmean[self.new_yr_condition] / self.nt_y[self.new_yr_condition]

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
        month_nr_idx = month_number - 2 # month number is first day of new month but, needs to be 0 indexed and the previous month.
        self._shift_and_reset_matrix_mask(self.monthly_infl_20yrs[:, month_nr_idx,:])
        self.monthly_infl_20yrs[:, month_nr_idx, 0] = self.monthly_infl
        self.mtifl_month_20yrs[:, month_nr_idx] = np.nanmean(self.monthly_infl_20yrs[:, month_nr_idx, :], axis=1)
        
        print("Average total irrigation demand this year: ", np.sum(self.new_irrmean/self.nt_y))
        print("Long-term mean total inflow in this month is: ", self.mtifl_month_20yrs[:, month_nr_idx])
        # Then check if this leads to a new operational year. If so, calculate yearly averages and
        # storage at begining of the year. Only check for new operational year if more than 8 months have passed. If 14th month has passed it is always a new year.
        self.new_yr_condition = ((self.mtifl_month_20yrs[:, month_nr_idx] < self.mtifl) 
                                 & (self.n_months_in_new_year > 8) 
                                 | (self.n_months_in_new_year == 15))
        
        if np.any(self.new_yr_condition):
            self.yearly_reservoir_calculations()

        print("Current mean monthly inflow is:", self.monthly_infl)
        # Reset the monthly mean inflow to zero for the next month.
        self.monthly_infl = np.zeros((self.N,), dtype='f8')
        self.nt_m = 0

        

        return
    
    def new_reservoir_management(self, inflow, NoRoutingExecuted, get_irr_demand = False, pot_irrConsumption_m_per_cell=None):
        """
        Management module, that manages reservoir outflow per day, based on Hanasaki protocol (Hanasaki et al., 2006).

        Input variables:                                                               Units
        
        inflow:         channel inflow value                                           (m^3/s)
        pot_irrConsumption_m_per_cell: irrigation consumption per cell in meters       (m/day)
        irr_demand:     irrigation demand per command area                             (m^3/s)

        All other necessary variables defined in initiate_agents().
        """
        # initialize daily variables
        self.inflow = inflow / 3600 #* self.model.InvDtSec    # convert inflow from m3/hr to m3/s
        Rres = self.inflow.copy()       # if there are no reservoirs, inflow equals outflow.
        Qin_resv = self.inflow.copy()   # make local copy of inflow, to calculate reservoir release.

        # Call total irrigation demand per command area module, and make this into a demand ratio.
        # self.total_irr_demand_area = self.get_irrigation_per_command_area(pot_irrConsumption_m_per_cell)
        # self.irr_demand_ratio = self.total_irr_demand_area / self.total_irr_demand_area_yesterday
        # self.total_irr_demand_area_yesterday = self.total_irr_demand_area

        # Set irrigation demand as the abstraction from today. Abstraction is in m3/day, convert to m3/s.
        # If the model is starting up, set abstraction to zero, as there is no abstraction possible yet.
        if (self.date.date() == (self.model.config["general"]["start_time"]+ timedelta(days=1)) 
            or (self.date.date() == (self.model.config["general"]["spinup_time"]+ timedelta(days=1)))
            and NoRoutingExecuted == 1):
            self.irr_demand = np.zeros((self.N,), dtype='f8')
        else:
            # Convert irrigation demand from m3/day to m3/s.
            self.irr_demand = self.model.data.HRU.reservoir_abstraction_m3.copy() * self.model.InvDtSec
            # Make sure that the abstraction is not larger than the available water for abstraction.
            # if get_irr_demand == False:
            #     assert np.all(self.available_water_for_abstraction >= self.model.data.HRU.reservoir_abstraction_m3), "Abstracted water was more than available water."

        # Make sure all variables are in same size and shape:
        assert self.inflow.size == self.irr_demand.size == self.cpa.size, "Variables for reservoir management module are not same size"
        assert self.inflow.shape == self.irr_demand.shape == self.cpa.shape , "Variables for reservoir management module are not same shape"

        # Calculate reservoir release for each reservoir type in m3/s. Only execute if there are reservoirs of that type.
        if np.any(self.condI):
            Rres[self.condI]= self.irrigation_reservoir(self.cpa,
                                                    self.condI,
                                                    Qin_resv,
                                                    self.S_begin_yr,
                                                    self.mtifl,
                                                    self.mtifl_month_20yrs[:, self.date.month-1],
                                                    self.irr_demand,
                                                    self.irrmean,
                                                    self.alpha)
        
        if np.any(self.condF):
            Rres[self.condF] = self.flood_control_reservoir(self.cpa,
                                                        self.condF,
                                                        Qin_resv,
                                                        self.S_begin_yr,
                                                        self.mtifl,
                                                        self.alpha)

        # Ensure environmental flow requirements, no reservoir overflow and no negative storage.
        Sending = self.Sini_resv.copy() # create temporary final storage variable. If nothing changes, initial storage will be final storage.
        
        Qout_resv, Sending = self.reservoir_water_balance(Qin_resv,
                                                            Rres,
                                                            self.Sini_resv,
                                                            self.cpa,
                                                            self.mtifl_month_20yrs[:,self.date.month-1],
                                                            self.alpha,
                                                            self.cond_all,
                                                            NoRoutingExecuted)
        
        # If the function is used to calculate the irrigation demand, 
        # return the provisional final release and irrigation demand and break out.
        # if get_irr_demand == True:
        #     return Qout_resv, self.irr_demand
        
                ###### ABSTRACTION calculation + release from remaining abstraction previous day ########
        # Get release from previous day that is not abstracted and add it to release today.
        if NoRoutingExecuted == 0:
            """
            Remaining problems:
            - abstraction_prev_day does not yet get values from water_demand.py
            - self.abstraction_prev_day_dt is not universally available yet (needs to be available for all 24 timesteps in a day)
            - self.release_for_abstraction is not universally available yet (needs to be available in water_demand.py and in all 24 timesteps).
            """
            # 1. Get the abstracted release and the remaining release from previous day.
            # 2. Convert the abstraction from the previous day from m3/day to m3/s
            # 3. Set the release for abstraction to zero, so it can be filled up in the new day.
            abstraction_prev_day = self.model.data.HRU.reservoir_abstraction_m3.copy() ####
            self.abstraction_prev_day_dt = abstraction_prev_day * self.model.InvDtSec
            self.release_for_abstraction.fill(0)
        
        mean_inflow_month = self.mtifl_month_20yrs[:, self.date.month-1]
        mean_inflow_month_min = mean_inflow_month * 0.1
        # Get water available for abstraction for the next day. 
        self.release_for_abstraction += Qout_resv - self.mtifl_month_20yrs[:,self.date.month-1]*0.1
        # Subtract the abstraction from the previous day from today's final release.
        
        # Subtract abstraction previous day from outflow today (to prevent double counting in storage subtraction)
        Qout_resv -= self.abstraction_prev_day_dt

        if np.any(Qout_resv < mean_inflow_month_min):
            #Qout_original = Qout_
            Qout_resv[Qout_resv < mean_inflow_month_min] = mean_inflow_month_min[Qout_resv < mean_inflow_month_min]
            # Qout_deficit = abs(Qout_original - Qout_)
            # Qout_ = Qout_ + Qout_deficit

        # # Make reservoir release available once final release is calculated.
        # # Water available for abstraction must always be positive or zero.
        # self.release_for_abstraction[self.condI] += np.maximum(0, 
        #                                                        (Qout_resv[self.condI] - self.mtifl_month_20yrs[self.condI, date.month-1] * 0.2))

        # """
        # If the abstraction from previous day is larger than the release for today, the release is set to 10% of 
        # monthly mtifl. The unreleased abstraction is added to the remaining abstraction for the next day.
        # """
        # if Qout_resv[self.condI] < self.abstraction_prev_day_dt:
        #     Qout_resv = 0
        #     self.remaining_abstraction += self.abstraction_prev_day_dt[self.condI] - Qout_resv[self.condI]
        # else:
        #     Qout_resv[self.condI] -= self.abstraction_prev_day_dt[self.condI]


        ##### AVERAGES CALCULATIONS #####
        # Add results of calculations to instances, to calculate yearly averages at end of year.
        self.monthly_infl += Qin_resv
        self.new_mtifl += Qin_resv
        self.new_irrmean += self.irr_demand
        self.nt_y += 1
        self.nt_m += 1

        # Set new reservoir storage for next timestep.
        self.Sini_resv = Sending.copy()
        #self.irr_demand = np.zeros((self.N,), dtype='f8')

        # if it is the last day of the month, check if it also means the start of a new operational year.
        if self.date.day == 1 and NoRoutingExecuted == 0:
            self.check_for_new_operational_year(self.date.month)

        return Qout_resv, self.irr_demand

    def irrigation_reservoir(self, cpa, cond_ppose, qin, S_begin_yr, mtifl, mtifl_month, irr_demand, irrmean, alpha):
        """
        Computes release from irrigation reservoirs
        
        cpa = reservoir capacity                                    (m^3)
        cond_ppose = array containing irrigation reservoir cells 
        based on selection mask
        qin = inflow                                                (m^3/s)
        Sini = initial storage                                      (m^3)
        mtifl = annual mean total annual inflow                     (m^3/s)
        irr_demand = downstream irrigation demand                   (...)
        irrmean = mean irrigation demand                            (...)
        alpha = reservoir capacity reduction factor                 (dimensionless)        
        """
        # irrigation Reservoirs
        # condI = np.where((cpa != 0) & (ppose ==2))[0] irrigation reservoir cells

        # initialization
        Nx = len(cond_ppose)
        Rprovisional = np.zeros([Nx, ])  # Provisional Release
        Rirrg_final = np.zeros([Nx, ])  # Final Release

        # water management
        current_demand = irr_demand[cond_ppose]  # downstream  demand: m^3/s
        mean_demand = irrmean[cond_ppose]   # downstream mean demand:  m^3/s
        mtifl_irr = mtifl[cond_ppose]       # mean flow:  m^3/s
        mtifl_month_irr = mtifl_month[cond_ppose]
        cpa_irr = cpa[cond_ppose]           # capacity: 1e6 m^3
        qin_irr = qin[cond_ppose]
        Sbeginning_of_year = S_begin_yr[cond_ppose]

        # ****** Provisional Release *******
        # mean demand & annual mean inflow
        M = 0.1

        m = mean_demand - ((1 - M) * mtifl_month_irr)
        cond1 = np.where(m >= 0)[0]  # m >=0 ==> dmean > = 0.5*annual mean inflow
        cond2 = np.where(m < 0)[0]   # m < 0 ==> dmean <  0.5*annual mean inflow

        demand_ratio = np.divide(current_demand[cond1], mean_demand[cond1])
        
        # Provisional Release where DPI > 1-M, or m > 0
        # Old code from Abeshu et al. (2023):
        # Rprovisional[cond1] = np.multiply(0.5 * mtifl_irr[cond1],  
        #                                   (1 + demand_ratio))

        # Update from Shin et al. (2019)
        Rprovisional[cond1] = np.multiply(mtifl_month_irr[cond1],  
                                          (M + (1 - M) * demand_ratio))
        
        # Provisional release where DPI < 1-M or m < 0
        Rprovisional[cond2] = (mtifl_month_irr[cond2] + current_demand[cond2] - mean_demand[cond2])

        # ******  Final Release ******

        ### NEW CODE based on Shin et al. (2019) ###
        c = np.divide(cpa_irr, (mtifl_irr * 365 * 24 * 3600))
        temp_alpha = np.full((self.N,), 0.85) # Can be set to 0.85 like in Shin et al. or to 4, as in Biemans et al and Hanasaki.
        R_factor = np.minimum(1, (temp_alpha * c))
        Krls = np.divide(Sbeginning_of_year, (alpha * cpa_irr))
        
        temp1 = np.multiply(R_factor, Krls)
        temp2 = np.multiply(temp1, Rprovisional)
        temp3 = np.multiply((1 - R_factor), qin_irr)
        Rirrg_final = temp2 + temp3


        return Rirrg_final
    
    def flood_control_reservoir(self, cpa, cond_ppose, qin, S_begin_yr, mtifl, alpha):
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
        Rprovisional = np.zeros([Nx, ])      # Provisional Release
        Rflood_final = np.zeros([Nx, ])      # Final Release

        # water management
        mtifl_flood = mtifl[cond_ppose]      # mean flow:  m^3/s
        cpa_flood = cpa[cond_ppose]          # capacity:   m^3
        qin_flood = qin[cond_ppose]          # mean flow:  m^3/s
        Sbeginning_ofyear = S_begin_yr[cond_ppose] # capacity:   m^3

        # Provisional Release
        Rprovisional = mtifl_flood.copy()

        # Final Release
        # capacity & annual total infow
        c = np.divide(cpa_flood, (mtifl_flood * 365 * 24 * 3600))
        cond1 = np.where(c >= 0.5)[0]       # c = capacity/imean >= 0.5
        cond2 = np.where(c < 0.5)[0]        # c = capacity/imean < 0.5

        # c = capacity/imean >= 0.5
        Krls = np.divide(Sbeginning_ofyear, (alpha * cpa_flood))
        Rflood_final[cond1] = np.multiply(Krls[cond1], mtifl_flood[cond1])
        # c = capacity/imean < 0.5
        temp1 = (c[cond2] / 0.5)**2
        temp2 = np.multiply(temp1, Krls[cond2])
        temp3 = np.multiply(temp2, Rprovisional[cond2])
        temp4 = np.multiply((1 - temp1), qin_flood[cond2])
        Rflood_final[cond2] = temp3 + temp4

        return Rflood_final
    
    def reservoir_water_balance(self, Qin, Qout, Sin, cpa, mtifl, alpha, cond_all, NoRoutingExecuted):

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
        Rfinal = np.zeros([Nx, ])    # final release
        Sfinal = np.zeros([Nx, ])    # final storage

        ###### WATER BALANCE CALCULATIONS ######
        # 1. Environmental flow: Qout must be at least 10% of the mean monthly inflow.
        diff_rt = Qout_ - (mtifl_month * 0.1)
        indx_rt = np.where(diff_rt < 0)[0]
        Qout_[indx_rt] = 0.1 * mtifl_month[indx_rt]

        # Check for change in storage.
        dsdt_resv = (Qin_ - Qout_) * dt
        Stemp = Sin_ + dsdt_resv

        # 2. condition a : storage > capacity | Storage can not be larger than reservoir capacity, so remove overflow. 
        Sa = (Stemp > (alpha * cpa_))
        if Sa.any():
            Sfinal[Sa] = alpha[Sa] * cpa_[Sa] # Storage is set at full level.
            Rspill = (Stemp[Sa] - (alpha[Sa] * cpa_[Sa])) / dt # Spilling is determined as earlier storage level minus maximum capacity. 
            Rfinal[Sa] = Qout_[Sa] + Rspill # Calculate new release rate: Earlier determined outflow + spilling.

        # 3. condition b : storage <= 0 | StSorage can not be smaller than 10% of reservoir capacity, so remove the outflow that results in too low capacity.
        Sb = (Stemp < 0.1 * self.cpa * alpha)
        if Sb.any():
            
            Sfinal[Sb] = 0.1 * self.cpa[Sb] * alpha[Sb] # Final storage is then 10% of effective capacity.
            Rfinal[Sb] = ((Stemp[Sb] - Sfinal[Sb])/dt) + Qin_[Sb] # Add negative storage to inflow, to lower final outflow and prevent negative storage.
            print("Storage before water balance correction:", Stemp)
            print("Storage after water balance correction:", Sfinal[Sb])

        # 4. condition c : Storage > 0 & Storage < total capacity | the reverse of condition a and b.
        Sc = ((Stemp > 0) & (Stemp <= alpha*cpa_))
        if Sc.any():
            # Assign temporary storage and outflow as final storage and release, for all reservoirs that do have proper storage levels.
            Sfinal[Sc] = Stemp[Sc]
            Rfinal[Sc] = Qout_[Sc]


        return Rfinal, Sfinal     

    def get_available_water_reservoir_command_areas(self, reservoir_storage_m3):
        self.date = self.model.current_time
        if "ruleset" in self.config and self.config["ruleset"] == "new_module":
            # Get the the minimum possible outflow for today. Equation 3a. This is the water available for abstraction.
            # self.minimum_outflow_today, _ = self.new_reservoir_management(inflow = np.zeros(self.N), 
            #                                                          NoRoutingExecuted=1,
            #                                                          get_irr_demand = True)
            # # Subtract 10% and 5% mean inflow from the available water for abstraction. 
            # # Second reservoir has lower minimum required mean inflow as this is a pickup reservoir.
            # self.available_water_for_abstraction = np.maximum(0,((self.minimum_outflow_today - self.mtifl_month_20yrs[:,self.date.month-1]*[0.1, 0.05]) * 3600*24))
            
            self.available_water_for_abstraction = self.release_for_abstraction.copy()
        else: # Use the old module
            self.available_water_for_abstraction = self.reservoir_release_factors * reservoir_storage_m3

        return self.available_water_for_abstraction*3600
    
    def determine_outflow_module(self, reservoirStorageM3C, inflow, 
                                 waterBodyIDs, pot_irrConsumption_m_per_cell, NoRoutingExecuted):
        
        # Check if to use new module.
        if "ruleset" in self.config and self.config["ruleset"] == "new_module":
            outflow, irr_demand = self.new_reservoir_management(inflow, 
                                                                NoRoutingExecuted,
                                                                get_irr_demand=False,
                                                                pot_irrConsumption_m_per_cell = pot_irrConsumption_m_per_cell)
            
            if self.date == 1 and NoRoutingExecuted == 0:
                outflow2 = self.regulate_reservoir_outflow(reservoirStorageM3C, inflow, waterBodyIDs)

            return outflow, irr_demand
        
        # Check if to use natural state, where inflow is outflow.
        elif "ruleset" in self.config and self.config["ruleset"] == "no-human-influence":
            outflow = inflow.copy() / 3600
            return outflow, None

        # Otherwise, use original LISFLOOD reservoir module.
        else:
            outflow = self.regulate_reservoir_outflow(reservoirStorageM3C, inflow, waterBodyIDs)
            return outflow, None
        

########################OLD CODE########################
    def get_irrigation_per_reservoir_command_area_original(
            self, reservoir_storage_m3, potential_irrigation_consumption_m):
        """
        Jens' original code
        """
        potential_irrigation_consumption_m3 = (
            potential_irrigation_consumption_m * self.model.data.HRU.cellArea
        )

        # Only keep the irrigation consumption for cells that are in a command area.
        potential_irrigation_consumption_m3[self.model.data.HRU.land_owners != -1]

        # Calculate the irrigation per farmer that is in the command area.
        potential_irrigation_consumption_m3 = np.bincount(
            self.model.data.HRU.land_owners[self.model.data.HRU.land_owners != -1],
            weights= potential_irrigation_consumption_m3[self.model.data.HRU.land_owners != -1],
            minlength=self.model.agents.farmers.n,
        )

        return self.reservoir_release_factors * reservoir_storage_m3
    
    def get_irrigation_per_command_area(self, potential_irrigation_consumption_m):
        """
        Function that retrieves the irrigation demand per command area, which can be used
        calculating reservoir release with Hanasaki protocol.

        Input variables:                    Function:                                               Units
        potential_irrigation_consumption_m: potential irrigation consumption in meters per cell.    (m/day)    

        """
        # Create irrigation demand array, where irrigation demand per command area is stored.
        irr_demand = np.zeros((self.N,), dtype='f8')

        # Call arrays with land owners per HRU, command area per HRU and command area per farmer.
        land_owner_per_cell = self.model.data.HRU.land_owners
        command_area_per_cell = self.model.agents.farmers.var.reservoir_command_areas
        command_area_per_farmer = self.model.agents.farmers.farmer_command_area

        # Transform irrigation consumption from meters to m3. 
        pot_irr_consumption_m3_per_cell = (
            potential_irrigation_consumption_m * self.model.data.HRU.cellArea
        )
        # Only keep the irrigation consumption for cells that are in a command area and have a farmer.
        pot_irr_consumption_per_cell_in_ca = pot_irr_consumption_m3_per_cell[
            (command_area_per_cell != -1) & (land_owner_per_cell != -1)]
        
        # Only keep cell with a farmer in the command area (this makes that both arrays line up again)
        cells_with_farmer_in_ca = land_owner_per_cell[
            (land_owner_per_cell != -1) & (command_area_per_cell != -1)]
    

        # Calculate the irrigation per farmer that is in the command area.
        # Bincount counts the number of occurrences cells of a farmer in the command area and returns an array, where the index of the array matches the land owner ID and the value at that index is the total irrigation consumption of that farmer. 
        potential_irrigation_consumption_per_farmer_in_ca = np.bincount(
            cells_with_farmer_in_ca,                        # land owner ID for each cell with a farmer and in the command area.
            weights= pot_irr_consumption_per_cell_in_ca,    # irrigation consumption per cell in the command area.
            minlength=self.model.agents.farmers.n,          # minimum length of the array is the number of farmers.
        )

        # Find out which command areas are actually used by farmers, and which are kept void.
        # Make an array of those command area numbers that are actually used.
        command_area_count = np.bincount(command_area_per_farmer[command_area_per_farmer!=-1])
        command_area_numbers_in_use = np.nonzero(command_area_count)[0]
        
        # Loop over all the command areas, and create an array of irrigation consumption per farmer
        # situated in that command area. Then sum the entire array, to get the total irrigation 
        # consumption for that command area and add it to the irrigation demand array.
        for idx, i in enumerate(command_area_numbers_in_use):
            total__irr_demand_in_ca = potential_irrigation_consumption_per_farmer_in_ca[
                command_area_per_farmer == i].sum()
            
            irr_demand[idx] = total__irr_demand_in_ca
        
        # Transform irrigation demand from m3/day to m3/s, as routing module works on hourly timestep
        irr_demand_per_sec = irr_demand * self.model.InvDtSec
        irr_demand_yearly = irr_demand * 365

        # Code for trying out the diff in timestep between routing and rest of model.
        # Seems that hour remains at 0, but it does do 24 calculations in that timestep.
        # if irr_demand[0] > 0:
        #     print("irr_demand", irr_demand, " date:", date.hour, date.day)

        return irr_demand_per_sec

    def _shift_and_reset_matrix_mask(self, matrix: np.ndarray, condition: np.ndarray = None) -> None:
        """
        Shifts columns to the right in the matrix and sets the first column to zero.
        Optionally requires the use of a boolean mask for selecting which rows to shift.
        """
        if condition is None:
            matrix[:, 1:] = matrix[:, 0:-1]  # Shift columns to the right
            matrix[:, 0] = 0 

        else:
            matrix[condition, 1:] = matrix[condition, 0:-1]  # Shift columns to the right
            matrix[condition, 0] = 0  # Reset the first column to 0    
    
    @property
    def save_state_path(self):
        folder = Path(self.model.initial_conditions_folder, "reservoir_operators")
        folder.mkdir(parents=True, exist_ok=True)
        return folder
    
    def save_state(self) -> None:
        """Saves the initial conditions of the model state."""
        attributes_to_save = ["mtifl_20yr", 
                              "irrmean_20yr",
                              "mtifl_month_20yrs",
                              "monthly_infl_20yrs"
                              ]  
        for attribute in attributes_to_save:
            values = getattr(self, attribute)
            fn = f"reservoir_operators.{attribute}.npz"
            fp = self.save_state_path / fn
            np.savez_compressed(fp, data=values)

    def regulate_reservoir_outflow(self, reservoirStorageM3C, inflowC, waterBodyIDs):
        #print(reservoirStorageM3C.sum())
        #if self.config['ruleset'] == 'inflow_is_outflow':
            # return np.zeros_like(inflowC)
        assert reservoirStorageM3C.size == inflowC.size == waterBodyIDs.size

        # Four types of outflow
        reservoir_fill = reservoirStorageM3C / self.reservoir_volume
        reservoir_outflow1 = np.minimum(
            self.minQC, reservoirStorageM3C * self.model.InvDtSec
        )
        reservoir_outflow2 = self.minQC + (self.normQC - self.minQC) * (
            reservoir_fill - self.cons_limit_ratio
        ) / (self.norm_limit_ratio - self.cons_limit_ratio)
        reservoir_outflow3 = self.normQC
        temp = np.minimum(self.nondmgQC, np.maximum(inflowC * 1.2, self.normQC))
        reservoir_outflow4 = np.maximum(
            (reservoir_fill - self.flood_limit_ratio - 0.01)
            * self.reservoir_volume
            * self.model.InvDtSec,
            temp,
        )
        reservoir_outflow = reservoir_outflow1.copy()

        reservoir_outflow = np.where(
            reservoir_fill > self.cons_limit_ratio,
            reservoir_outflow2,
            reservoir_outflow,
        )
        reservoir_outflow = np.where(
            reservoir_fill > self.norm_limit_ratio, self.normQC, reservoir_outflow
        )
        reservoir_outflow = np.where(
            reservoir_fill > self.norm_flood_limit_ratio,
            reservoir_outflow3,
            reservoir_outflow,
        )
        reservoir_outflow = np.where(
            reservoir_fill > self.flood_limit_ratio,
            reservoir_outflow4,
            reservoir_outflow,
        )

        temp = np.minimum(reservoir_outflow, np.maximum(inflowC, self.normQC))

        reservoir_outflow = np.where(
            (reservoir_outflow > 1.2 * inflowC)
            & (reservoir_outflow > self.normQC)
            & (reservoir_fill < self.flood_limit_ratio),
            temp,
            reservoir_outflow,
        )

        return reservoir_outflow
    
        
    def step(self) -> None:
        return None
