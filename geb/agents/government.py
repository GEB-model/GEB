# -*- coding: utf-8 -*-
import numpy as np
from honeybees.agents import AgentBaseClass


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
        if self.model.current_timestep == 1:
            for region in self.agents.farmers.borewell_cost_1[1].keys():
                self.agents.farmers.borewell_cost_1[1][region] = [
                    0.5 * x for x in self.agents.farmers.borewell_cost_1[1][region]
                ]
                self.agents.farmers.borewell_cost_2[1][region] = [
                    0.5 * x for x in self.agents.farmers.borewell_cost_2[1][region]
                ]

        return

    def request_flood_cushions(self, reservoirIDs):
        pass

    def step(self) -> None:
        """This function is run each timestep. However, only in on the first day of each year and if the scenario is `government_subsidies` subsidies is actually provided to farmers."""
        if self.model.scenario == "government_subsidies":
            self.provide_subsidies()
