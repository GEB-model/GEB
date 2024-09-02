# -*- coding: utf-8 -*-
from .households import Households
from .crop_farmers import CropFarmers
from .livestock_farmers import LiveStockFarmers
from .industry import Industry
from .reservoir_operators import ReservoirOperators
from .town_managers import TownManagers
from .government import Government
from .market import Market


class Agents:
    """This class initalizes all agent classes, and is used to activate the agents each timestep.

    Args:
        model: The GEB model
    """

    def __init__(self, model) -> None:
        self.model = model
        self.households = Households(model, self, 0.1)
        self.crop_farmers = CropFarmers(model, self, 0.1)
        self.livestock_farmers = LiveStockFarmers(model, self, 0.1)
        self.industry = Industry(model, self)
        self.reservoir_operators = ReservoirOperators(model, self)
        self.town_managers = TownManagers(model, self)
        self.government = Government(model, self)
        self.market = Market(model, self)

        self.agents = [
            self.households,
            self.crop_farmers,
            self.livestock_farmers,
            self.industry,
            self.reservoir_operators,
            self.town_managers,
            self.government,
            self.market,
        ]

        if not self.model.load_initial_data:
            self.initiate()
        else:
            self.restore_state()
            self.restart()

    def initiate(self) -> None:
        """Initiate all agents."""
        for agent_type in self.agents:
            agent_type.initiate()

    def step(self) -> None:
        """This function is called every timestep and activates the agents in order of NGO, government and then farmers."""
        for agent_type in self.agents:
            agent_type.step()

    def save_state(self) -> None:
        """Save the state of all agents."""
        for agent_type in self.agents:
            agent_type.save_state(folder=agent_type.__class__.__name__)

    def restore_state(self) -> None:
        """Load the state of all agents."""
        for agent_type in self.agents:
            agent_type.restore_state(folder=agent_type.__class__.__name__)

    def restart(self):
        """Restart all agents."""
        for agent_type in self.agents:
            if hasattr(agent_type, "restart"):
                agent_type.restart()
