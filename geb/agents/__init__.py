# -*- coding: utf-8 -*-
from geb.module import Module
from geb.workflows import TimingModule

from .crop_farmers import CropFarmers
from .government import Government
from .households import Households
from .industry import Industry
from .livestock_farmers import LiveStockFarmers
from .market import Market
from .reservoir_operators import ReservoirOperators
from .town_managers import TownManagers


class Agents(Module):
    """This class initalizes all agent classes, and is used to activate the agents each timestep.

    Args:
        model: The GEB model
    """

    def __init__(self, model) -> None:
        Module.__init__(self, model)

        self.households = Households(model, self, 0.1)
        self.crop_farmers = CropFarmers(model, self, 0.1)
        self.livestock_farmers = LiveStockFarmers(model, self, 0.1)
        self.industry = Industry(model, self)
        self.reservoir_operators = ReservoirOperators(model, self)
        self.town_managers = TownManagers(model, self)
        self.government = Government(model, self)
        self.market = Market(model, self)

        self.agents = [
            self.crop_farmers,
            self.households,
            self.livestock_farmers,
            self.industry,
            self.reservoir_operators,
            self.town_managers,
            self.government,
            self.market,
        ]

    @property
    def name(self) -> str:
        return "agents"

    def spinup(self) -> None:
        pass

    def step(self) -> None:
        """This function is called every timestep and activates the agents in order of NGO, government and then farmers."""
        timer = TimingModule("Agents")

        for agent_type in self.agents:
            agent_type.step()
            timer.new_split(agent_type.name)

        if self.model.timing:
            print(timer)

        self.report(self, locals())
