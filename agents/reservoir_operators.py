# -*- coding: utf-8 -*-
from hyve.agents import AgentBaseClass
import os
import numpy as np
import pandas as pd

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
        df = pd.read_excel(fn)
        self.reservoirs = df[df['Lake_type'] == 2].copy()
        self.reservoirs['agent_index'] = np.arange(len(self.reservoirs))

        self.set_reservoir_release_factors()
        self.waterBodyID_to_agent_id_dict = self.reservoirs.set_index('Hylak_id')['agent_index'].to_dict()

    def waterBodyID_to_agent_id(self, waterBodyID):
        return self.waterBodyID_to_agent_id_dict[waterBodyID]
    
    def set_reservoir_release_factors(self) -> None:
        self.reservoir_release_factors = np.full(len(self.reservoirs), self.model.config['agent_settings']['reservoir_operators']['max_reservoir_release_factor'])

    def get_available_water_reservoir_command_areas(self, reservoir_ID, reservoir_storage_m3):
        return self.reservoir_release_factors[self.waterBodyID_to_agent_id(reservoir_ID)] * reservoir_storage_m3

    def step(self) -> None:
        """This function is run each timestep. However, only in on the first day of each year and if the scenario is `government_subsidies` subsidies is actually provided to farmers."""
        self.set_reservoir_release_factors()
