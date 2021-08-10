import numpy as np
from hyve.agents import AgentBaseClass
from agents.farmers import Farmers

class NGO(AgentBaseClass):
    def __init__(self, model, agents):
        AgentBaseClass.__init__(self, model, agents)

    def initiate_agents(self): pass

    def provide_training(self):
        rng = np.random.default_rng()
        trained_farmers = rng.choice(self.agents.farmers.is_water_efficient.size, size=100_000, replace=False)
        self.agents.farmers.is_water_efficient[trained_farmers] = True

    def step(self):
        if self.model.current_timestep == 0 and self.model.args.scenario == 'ngo_training':
            self.provide_training()

class Government(AgentBaseClass):
    def __init__(self, model, agents):
        AgentBaseClass.__init__(self, model, agents)

    def initiate_agents(self): pass

    def provide_subsidies(self):
        total_water_use = self.agents.farmers.reservoir_abstraction_m3_by_farmer + self.agents.farmers.groundwater_abstraction_m3_by_farmer + self.agents.farmers.channel_abstraction_m3_by_farmer
        total_water_use_m = total_water_use / self.agents.farmers.field_size_per_farmer
        n_farmers_to_upgrade = self.agents.farmers.n // 20

        farmer_indices = np.arange(0, self.agents.farmers.n)
        indices_not_yet_water_efficient = farmer_indices[~self.agents.farmers.is_water_efficient]

        if indices_not_yet_water_efficient.size <= n_farmers_to_upgrade:
            self.agents.farmers.is_water_efficient[:] = True
        else:
            total_water_use_m_not_yet_water_efficient = total_water_use_m[~self.agents.farmers.is_water_efficient]
            self.agents.farmers.is_water_efficient[indices_not_yet_water_efficient[np.argpartition(-total_water_use_m_not_yet_water_efficient, n_farmers_to_upgrade)[:n_farmers_to_upgrade]]] = True

    def step(self):
        if self.model.current_time.day == 1 and self.model.current_time.month == 1 and self.model.current_timestep != 0 and self.model.args.scenario == 'government_subsidies':
            self.provide_subsidies()

class Agents:
    def __init__(self, model):
        self.model = model
        self.farmers = Farmers(model, self, 0.1)
        self.ngo = NGO(model, self)
        self.government = Government(model, self)

    def step(self):
        self.ngo.step()
        self.government.step()
        self.farmers.step()
