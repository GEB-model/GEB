from honeybees.agents import AgentBaseClass

class Households(AgentBaseClass):
    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.agents = agents

    def step(self) -> None:
        return None