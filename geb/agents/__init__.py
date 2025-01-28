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
            self.government,
            self.crop_farmers,
            self.households,
            self.livestock_farmers,
            self.industry,
            self.reservoir_operators,
            self.town_managers,
            self.market,
        ]

    def step(self) -> None:
        """This function is called every timestep and activates the agents in order of NGO, government and then farmers."""
        for agent_type in self.agents:
            agent_type.step()
