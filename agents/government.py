# -*- coding: utf-8 -*-
import numpy as np
from hyve.agents import AgentBaseClass

class Government(AgentBaseClass):
    """This class is used to simulate the government.
    
    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """
    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        self.ratio_farmers_to_provide_subsidies_per_year = 0.05
        AgentBaseClass.__init__(self)

    def provide_subsidies(self) -> None:
        """Provides subsidies to a number of farmers that are not yet efficient in their water usage."""
        total_water_use = self.agents.farmers.reservoir_abstraction_m3_by_farmer + self.agents.farmers.groundwater_abstraction_m3_by_farmer + self.agents.farmers.channel_abstraction_m3_by_farmer
        total_water_use_m = total_water_use / self.agents.farmers.field_size_per_farmer
        n_farmers_to_upgrade = int(self.agents.farmers.n * self.ratio_farmers_to_provide_subsidies_per_year / 365)

        farmer_indices = np.arange(0, self.agents.farmers.n)
        indices_not_yet_water_efficient = farmer_indices[~self.agents.farmers.is_water_efficient]

        if indices_not_yet_water_efficient.size <= n_farmers_to_upgrade:
            self.agents.farmers.is_water_efficient[:] = True
        else:
            total_water_use_m_not_yet_water_efficient = total_water_use_m[~self.agents.farmers.is_water_efficient]
            self.agents.farmers.is_water_efficient[indices_not_yet_water_efficient[np.argpartition(-total_water_use_m_not_yet_water_efficient, n_farmers_to_upgrade)[:n_farmers_to_upgrade]]] = True

    def step(self) -> None:
        """This function is run each timestep. However, only in on the first day of each year and if the scenario is `government_subsidies` subsidies is actually provided to farmers."""
        if self.model.args.scenario == 'government_subsidies' and self.model.current_timestep != 0:
            self.provide_subsidies()
