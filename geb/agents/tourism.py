# -*- coding: utf-8 -*-
from . import AgentBaseClass


class Tourism(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["tourism"]
            if "tourism" in self.model.config["agent_settings"]
            else {}
        )

        AgentBaseClass.__init__(self)

    def initiate(self) -> None:
        return

    def step(self) -> None:
        """This function is run each timestep."""
        pass
