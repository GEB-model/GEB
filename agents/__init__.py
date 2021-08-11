from agents.farmers import Farmers
from agents.ngo import NGO
from agents.government import Government

class Agents:
    """This class initalizes all agent classes, and is used to activate the agents each timestep.

    Args:
        model: The GEB model
    """
    def __init__(self, model) -> None:
        self.model = model
        self.farmers = Farmers(model, self, 0.1)
        self.ngo = NGO(model, self)
        self.government = Government(model, self)

    def step(self) -> None:
        """This function is called every timestep and activates the agents in order."""
        self.ngo.step()
        self.government.step()
        self.farmers.step()
