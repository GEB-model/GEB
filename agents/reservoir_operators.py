# -*- coding: utf-8 -*-
from hyve.agents import AgentBaseClass
import os
import numpy as np
import rasterio
import pandas as pd
from cwatm.management_modules.data_handling import cbinding, loadmap

class ReservoirOperators(AgentBaseClass):
    """This class is used to simulate the government.
    
    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """
    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        AgentBaseClass.__init__(self)
        self.initiate_agents()

    def initiate_agents(self):
        fn = os.path.join(self.model.config['general']['input_folder'], 'routing', 'lakesreservoirs', 'basin_lakes_data.xlsx')
        df = pd.read_excel(fn).set_index('Hylak_id')
        
        self.reservoirs = df[df['Lake_type'] == 2].copy()
        self.reservoirs['agent_index'] = np.arange(len(self.reservoirs))
        assert (self.reservoirs['reservoir_volume'] > 0).all()

        self.set_reservoir_release_factors()
        self.waterBodyID_to_agent_id_dict = self.reservoirs['agent_index'].to_dict()

        self.reservoirs['cons_limit_ratio'] = 0.02
        self.reservoirs['norm_limit_ratio'] = self.reservoirs['flood_volume'] / self.reservoirs['reservoir_volume']
        self.reservoirs['flood_limit_ratio'] = 1
        self.reservoirs['norm_flood_limit_ratio'] = self.reservoirs['norm_limit_ratio'] + 0.5 * (self.reservoirs['flood_limit_ratio'] - self.reservoirs['norm_limit_ratio'])

        self.reservoirs['minQC'] = self.model.config['agent_settings']['reservoir_operators']['MinOutflowQ'] * self.reservoirs['Dis_avg']
        self.reservoirs['normQC'] = self.model.config['agent_settings']['reservoir_operators']['NormalOutflowQ'] * self.reservoirs['Dis_avg']
        self.reservoirs['nondmgQC'] = self.model.config['agent_settings']['reservoir_operators']['NonDamagingOutflowQ'] * self.reservoirs['Dis_avg']

    def waterBodyID_to_agent_id(self, waterBodyID):
        return self.waterBodyID_to_agent_id_dict[waterBodyID]

    def get_reservoir_data(self, key, waterBodyIDs):
        out = []
        for waterbody_ID in waterBodyIDs:
            out.append(self.reservoirs.at[waterbody_ID, key])
        return np.array(out)
    
    def set_reservoir_release_factors(self) -> None:
        self.reservoir_release_factors = np.full(len(self.reservoirs), self.model.config['agent_settings']['reservoir_operators']['max_reservoir_release_factor'])

    def regulate_reservoir_outflow(
        self,
        reservoirStorageM3C,
        inflowC,
        waterBodyIDs
    ):
        reservoirs = self.reservoirs[self.reservoirs.index.isin(waterBodyIDs)]
        reservoir_fill = reservoirStorageM3C / reservoirs['reservoir_volume']
        reservoir_outflow1 = np.minimum(reservoirs['minQC'], reservoirStorageM3C * self.model.InvDtSec)
        reservoir_outflow2 = reservoirs['minQC'] + (reservoirs['normQC'] - reservoirs['minQC']) * (reservoir_fill - reservoirs['cons_limit_ratio']) / (reservoirs['norm_limit_ratio'] - reservoirs['cons_limit_ratio'])
        reservoir_outflow3 = reservoirs['normQC']
        temp = np.minimum(reservoirs['nondmgQC'], np.maximum(inflowC * 1.2, reservoirs['normQC']))
        reservoir_outflow4 = np.maximum((reservoir_fill - reservoirs['flood_limit_ratio'] - 0.01) * reservoirs['reservoir_volume'] * self.model.InvDtSec, temp)
        reservoir_outflow = reservoir_outflow1.copy()

        reservoir_outflow = np.where(reservoir_fill > reservoirs['cons_limit_ratio'], reservoir_outflow2, reservoir_outflow)
        reservoir_outflow = np.where(reservoir_fill > reservoirs['norm_limit_ratio'], reservoirs['normQC'], reservoir_outflow)
        reservoir_outflow = np.where(reservoir_fill > reservoirs['norm_flood_limit_ratio'], reservoir_outflow3, reservoir_outflow)
        reservoir_outflow = np.where(reservoir_fill > reservoirs['flood_limit_ratio'], reservoir_outflow4, reservoir_outflow)

        temp = np.minimum(reservoir_outflow, np.maximum(inflowC, reservoirs['normQC']))

        reservoir_outflow = np.where((reservoir_outflow > 1.2 * inflowC) &
                                    (reservoir_outflow > reservoirs['normQC']) &
                                    (reservoir_fill < reservoirs['flood_limit_ratio']), temp, reservoir_outflow)
        return reservoir_outflow

    def get_available_water_reservoir_command_areas(self, reservoir_ID, reservoir_storage_m3):
        return self.reservoir_release_factors[self.waterBodyID_to_agent_id(reservoir_ID)] * reservoir_storage_m3

    def step(self) -> None:
        """This function is run each timestep. However, only in on the first day of each year and if the scenario is `government_subsidies` subsidies is actually provided to farmers."""
        self.set_reservoir_release_factors()
