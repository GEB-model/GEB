import numpy as np

from .general import AgentArray
from honeybees.agents import AgentBaseClass

class Households(AgentBaseClass):
    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.agents = agents

        locations = np.load(self.model.model_structure['binary']['agents/households/locations'])['data']
        self.max_size = int(locations.shape[0] * (1 + reduncancy) + 1)
        
        self.locations = AgentArray(locations, max_size=self.max_size)
        
        sizes = np.load(self.model.model_structure['binary']['agents/households/sizes'])['data']
        self.sizes = AgentArray(sizes, max_size=self.max_size)

        return None

    def step(self) -> None:
        return None