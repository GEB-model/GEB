"""
Agent classes for the GEB model.

This package exposes agent implementations used to simulate actors in the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from geb.module import Module
from geb.workflows import TimingModule

from .crop_farmers import CropFarmers
from .general import AgentBaseClass
from .government import Government
from .households import Households
from .industry import Industry
from .livestock_farmers import LiveStockFarmers
from .market import Market
from .reservoir_operators import ReservoirOperators

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology as Hydrology


class Agents(Module):
    """This class initalizes all agent classes, and is used to activate the agents each timestep."""

    def __init__(self, model: GEBModel) -> None:
        """Initialize the Agents module.

        Initalizes all agent classes and stores them in a list to be activated each timestep.

        Args:
            model: The GEB model instance.
        """
        super().__init__(model)

        self.households = Households(model, self, 0.1)
        self.crop_farmers = CropFarmers(model, self, 0.1)
        self.livestock_farmers = LiveStockFarmers(model, self, 0.1)
        self.industry = Industry(model, self)
        self.reservoir_operators = ReservoirOperators(model, self)
        self.government = Government(model, self)
        self.market = Market(model, self)

        self.agents: list[AgentBaseClass] = [
            self.crop_farmers,
            self.households,
            self.livestock_farmers,
            self.industry,
            self.reservoir_operators,
            self.government,
            self.market,
        ]

    @property
    def name(self) -> str:
        """Return the name of the module."""
        return "agents"

    def spinup(self) -> None:
        """This function is called during model spinup.

        Does not do anything as no spinup is needed.
        """
        pass

    def step(self) -> None:
        """This function is called every timestep and activates the agents in order of NGO, government and then farmers."""
        timer = TimingModule("Agents")

        for agent_type in self.agents:
            agent_type.step()
            timer.finish_split(agent_type.name)

        if self.model.timing:
            print(timer)

        self.report(locals())
