# -*- coding: utf-8 -*-
import numpy as np
from hyve.agents import AgentBaseClass

class NGO(AgentBaseClass):
    """This class is used to simulate NGOs.
    
    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """
    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        self.n_trainings = 100_000  # Number of trainings to provide each year.
        AgentBaseClass.__init__(self)

    def provide_training(self) -> None:
        """Provides training to a set number of farmers that are not yet efficient in their water usage."""
        rng = np.random.default_rng()
        trained_farmers = rng.choice(self.agents.farmers.is_water_efficient.size, size=self.n_trainings, replace=False)
        self.agents.farmers.is_water_efficient[trained_farmers] = True

    def step(self):
        """This function is run each timestep. However, only in the first timestep and if the scenario is `ngo_training` training is actually provided to farmers."""
        if self.model.args.scenario == 'ngo_training' and self.model.current_timestep == 0:
            self.provide_training()
