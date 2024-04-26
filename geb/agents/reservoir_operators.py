# -*- coding: utf-8 -*-
from honeybees.agents import AgentBaseClass
import os
import numpy as np
import pandas as pd
from cwatm.hydrological_modules.water_demand.irrigation import waterdemand_irrigation


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

        # call irrigation water demand module
        self.irrigation = waterdemand_irrigation(model)


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
        self.N = self.cpa.shape[0] # Number of reservoirs. All arrays will have this dimension: (N,)
        self.S_begin_yr = 0.5 * self.cpa.copy()
        self.Sini_resv = self.S_begin_yr.copy()
        self.alpha = np.full((self.N,), 0.8) # set all reservoirs to a capacity reduction factor of 0.8 for now.
        self.mtifl = self.active_reservoirs["average_discharge"].values
        self.irrmean = 0.5 * self.mtifl # start with estimated mean irrigation demand of 0.5 of mtifl.
        


        ### TEMPORARY CODE: Create selection mask, which selects reservoirs based on purpose. ###
        self.res_ppose = np.full((self.N,), 1) # temporarlily set all reservoirs to irrigation reservoirs, because data is still lacking.
        self.condI = (self.res_ppose == 1)  # irrigation reservoirs
        self.condF = (self.res_ppose == 2)  # flood reservoirs
        self.cond_all = (self.res_ppose > 0)

        ### FUTURE CODE: WHEN RESERVOIR PURPOSE DATA IS AVAILABLE. ###
        # self.res_ppose = self.active_reservoirs["purpose"].values 
        # self.condI = (self.res_ppose == 1) # irrigation reservoirs
        # self.condF = (self.res_ppose == 2)  # flood reservoirs
        # self.cond_all = (self.res_ppose > 0)

        ### Initiate variables for yearly calculations. ###
        self.new_mtifl = np.zeros((self.N,), dtype='f8')
        self.new_irrmean = np.zeros((self.N,), dtype='f8')
        self.nt = 0

        return self

    def yearly_reservoir_calculations(self):
        """
        Function stores yearly parameters and updates after every year.
        """
        # Calculate averages and declare storage begin of year.
        self.S_begin_yr = self.Sini_resv.copy()
        self.mtifl = self.new_mtifl / self.nt
        self.irrmean = self.new_irrmean / self.nt

        # Reset variables for next year.
        self.new_mtifl = np.zeros((self.N,), dtype = 'f8')
        self.irrmean = np.zeros((self.N,), dtype = 'f8')
        self.nt = 0

        return
    
    def new_reservoir_management(self, inflow, pot_irrConsumption_m_per_cell):
        """
        Management module, that manages reservoir outflow per day, based on Hanasaki protocol (Hanasaki et al., 2006).

        Input variables:                                                               Units
        
        inflow:         channel inflow value                                           (m^3/s)
        pot_irrConsumption_m_per_cell: irrigation consumption per cell in meters       (m)
        irr_demand:     irrigation demand per command area                             (...)

        All other necessary variables defined in initiate_agents().
        """
        # initialize daily variables
        self.inflow = inflow
        Rres = self.inflow.copy() # if there are no reservoirs, inflow equals outflow.
        Qin_resv = self.inflow.copy()
        date = self.model.current_time

        # Call irrigation per command area module.
        self.irr_demand = self.get_irrigation_per_command_area(pot_irrConsumption_m_per_cell)

        # Make sure all variables are in same shape and unit of m3, m3/s or s:
        assert self.inflow.size == self.irr_demand.size == self.cpa.size == self.mtifl.size, "Variables for reservoir management module are not same size"
        assert self.inflow.shape == self.irr_demand.shape == self.cpa.shape == self.mtifl.shape, "Variables for reservoir management module are not same shape"

        # Calculate reservoir release for each reservoir type in m3/s
        Rres[self.condI] = self.irrigation_reservoir(self.cpa,
                                                self.condI,
                                                Qin_resv,
                                                self.S_begin_yr,
                                                self.mtifl,
                                                self.irr_demand,
                                                self.irrmean,
                                                self.alpha)
        Rres[self.condF] = self.flood_control_reservoir(self.cpa,
                                                    self.condF,
                                                    Qin_resv,
                                                    self.S_begin_yr,
                                                    self.mtifl,
                                                    self.alpha)
        
        # Ensure environmental flow requirements, no reservoir overflow and no negative storage.
        Qout_resv = Rres.copy()
        Sending = self.Sini_resv.copy() # create temporary final storage variable. If nothing changes, initial storage will be final storage.
        Qout_resv[self.cond_all], Sending[self.cond_all] = self.reservoir_water_balance(Qin_resv,
                                                                              Rres,
                                                                              self.Sini_resv,
                                                                              self.cpa,
                                                                              self.mtifl,
                                                                              self.alpha,
                                                                              self.cond_all)

        # Set new reservoir storage for next timestep.
        self.Sini_resv = Sending.copy()
        self.irr_demand = np.zeros((self.N,), dtype='f8')
        self.irr_demand2 = np.zeros((self.N,), dtype='f8')

        # Add results of calculations to instances, to calculate yearly averages at end of year.
        self.new_mtifl += Qin_resv
        self.new_irrmean += self.irr_demand
        self.nt += 1

        if date.hour == 12 and date.month > 3:
            print("12 o'clock")
            breakpoint()
        # if it is the last day of the year, calculate yearly averages.
        if date.month == 12 and date.day == 31:
            self.yearly_reservoir_calculations()
    
        return Qout_resv

    def irrigation_reservoir(self, cpa, cond_ppose, qin, S_begin_yr, mtifl, irr_demand, irrmean, alpha):
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
        monthly_demand = irr_demand[cond_ppose]  # downstream  demand: m^3/s
        mean_demand = irrmean[cond_ppose]   # downstream mean demand:  m^3/s
        mtifl_irr = mtifl[cond_ppose]       # mean flow:  m^3/s
        cpa_irr = cpa[cond_ppose]           # capacity: 1e6 m^3
        qin_irr = qin[cond_ppose]
        Sbeginning_of_year = S_begin_yr[cond_ppose]

        # ****** Provisional Release *******
        # mean demand & annual mean inflow
        m = mean_demand - (0.5 * mtifl_irr)
        cond1 = np.where(m >= 0)[0]  # m >=0 ==> dmean > = 0.5*annual mean inflow
        cond2 = np.where(m < 0)[0]   # m < 0 ==> dmean <  0.5*annual mean inflow

        # Provisional Release
        demand_ratio = np.divide(monthly_demand[cond1], mean_demand[cond1])
        # Irrigation dmean >= 0.5*imean
        Rprovisional[cond1] = np.multiply(0.5 * mtifl_irr[cond1],  
                                          (1 + demand_ratio))
        # Irrigation dmean < 0.5*imean
        Rprovisional[cond2] = (mtifl_irr[cond2] + 
                               monthly_demand[cond2] - 
                               mean_demand[cond2])

        # ******  Final Release ******
        # capacity & annual total infow
        c = np.divide(cpa_irr, (mtifl_irr * 365 * 24 * 3600))
        cond3 = np.where(c >= 0.5)[0]  # c = capacity/imean >= 0.5
        cond4 = np.where(c < 0.5)[0]   # c = capacity/imean < 0.5

        # c = capacity/imean >= 0.5
        Krls = np.divide(Sbeginning_of_year, (alpha * cpa_irr))
        Rirrg_final[cond3] = np.multiply(Krls[cond3], mtifl_irr[cond3])

        # c = capacity/imean < 0.5
        temp1 = (c[cond4]/0.5)**2
        temp2 = np.multiply(temp1, Krls[cond4])
        temp3 = np.multiply(temp2, Rprovisional[cond4])
        temp4 = np.multiply((1 - temp1), qin_irr[cond4])
        Rirrg_final[cond4] = temp3 + temp4

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
    
    def reservoir_water_balance(self, Qin, Qout, Sin, cpa, mtifl, alpha, cond_all):

        """Re-adjusts release to ensure minimal environmental flow, and prevent overflow and negative storage;
        and computes the storage level after release for all reservoir types.
        
        Qin = inflow into reservoir (m3/s)
        Qout = release from reservoir (m3/s) 
        Sin = initial storage (m^3)
        cpa = reservoir capacity (m^3)
        mtifl = annual mean total annual inflow (m^3/s)
        alpha = reservoir capacity reduction factor
        dt = time step (day)
        cond_all = mask selecting all reservoir types.
        
        """
        # inputs
        Qin_ = Qin[cond_all]
        Qout_ = Qout[cond_all]
        Sin_ = Sin[cond_all]
        cpa_ = cpa[cond_all]
        mtifl_ = mtifl[cond_all]
        dt = 24 * 3600  # time step in seconds

        # final storage and release initialization
        Nx = len(cond_all)

        # environmental flow: Qout must be at least 10% of the inflow.
        diff_rt = Qout_ - (mtifl_ * 0.1)
        indx_rt = np.where(diff_rt < 0)[0]
        Qout_[indx_rt] = 0.1 * mtifl_[indx_rt]

        # storage
        dsdt_resv = (Qin_ - Qout_) * dt
        Stemp = Sin_ + dsdt_resv

        Rfinal = np.zeros([Nx, ])    # final release
        Sfinal = np.zeros([Nx, ])    # final storage

        # condition a : storage > capacity | Storage can not be larger than reservoir capacity, so remove overflow. 
        Sa = (Stemp > (alpha * cpa_))
        if Sa.any():
            Sfinal[Sa] = alpha * cpa_[Sa] # Storage is set at full level.
            Rspill = (Stemp[Sa] - (alpha * cpa_[Sa])) / dt # Spilling is determined as earlier storage level minus maximum capacity. 
            Rfinal[Sa] = Qout_[Sa] + Rspill # Calculate new release rate: Earlier determined outflow + spilling.

        # condition b : storage <= 0 | Storage can not be smaller than zero, so remove negative capacity from outflow.
        Sb = (Stemp < 0)
        if Sb.any():
            Sfinal[Sb] = 0
            Rfinal[Sb] = (Sin_[Sb]/dt) + Qin_[Sb] # Add negative storage to inflow, to lower final outflow and prevent negative storage.

        # condition c : Storage > 0 & Storage < total capacity | the reverse of condition a and b.
        Sc = ((Stemp > 0) & (Stemp <= alpha*cpa_))
        if Sc.any():
            # Assign temporary storage and outflow as final storage and release.
            Sfinal[Sc] = Stemp[Sc]
            Rfinal[Sc] = Qout_[Sc]

        return Rfinal,  Sfinal
    
    
    def regulate_reservoir_outflow(self, reservoirStorageM3C, inflowC, waterBodyIDs):
        #print(reservoirStorageM3C.sum())
        date = self.model.current_time
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

        # make outflow same as inflow for a setting without a reservoir
        if "ruleset" in self.config and self.config["ruleset"] == "no-human-influence":
            reservoir_outflow = inflowC

        return reservoir_outflow

    def get_available_water_reservoir_command_areas(self, reservoir_storage_m3):
        return self.reservoir_release_factors * reservoir_storage_m3
    
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
        # Bincount counts the number of occurrences cells of a farmer in the command area and returns an array, where the index of the array matches the land owner ID the value at that index is the total irrigation consumption of that farmer. 
        potential_irrigation_consumption_per_farmer_in_ca = np.bincount(
            cells_with_farmer_in_ca, #land_owner_ID_per_farmer_in_ca,
            weights= pot_irr_consumption_per_cell_in_ca,
            minlength=self.model.agents.farmers.n,
        )
        
        # Loop over all the command areas, and create an array of irrigation consumption per farmer
        # situated in that command area. Then sum the entire array, to get the total irrigation 
        # consumption for that command area and add it to the irrigation demand array.
        for i in range(self.N):
            total__irr_demand_in_ca = potential_irrigation_consumption_per_farmer_in_ca[
                command_area_per_farmer == i].sum()
            
            irr_demand[i] = total__irr_demand_in_ca
        
        # Transform irrigation demand from m3/day to m3/s, as routing module works on hourly timestep
        irr_demand_hourly = irr_demand / 24
        irr_demand_yearly = irr_demand * 365

        # Code for trying out the diff in timestep between routing and rest of model.
        # Seems that hour remains at 0, but it does do 24 calculations in that timestep.
        # if irr_demand[0] > 0:
        #     print("irr_demand", irr_demand, " date:", date.hour, date.day)

        return irr_demand_hourly
    


    def step(self) -> None:
        return None
