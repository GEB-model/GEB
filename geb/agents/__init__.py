# -*- coding: utf-8 -*-
import math
import numpy as np
from pathlib import Path
from honeybees.agents import AgentBaseClass as HoneybeesAgentBaseClass
from .general import AgentArray


class AgentBaseClass(HoneybeesAgentBaseClass):
    def __init__(self):
        if not hasattr(self, "redundancy"):
            self.redundancy = None  # default redundancy is None
        super().__init__()

    def get_max_n(self, n):
        if self.redundancy is None:
            return n
        else:
            max_n = math.ceil(n * (1 + self.redundancy))
            assert (
                max_n < 4294967295
            )  # max value of uint32, consider replacing with uint64
            return max_n

    def get_save_state_path(self, folder):
        folder = Path(self.model.initial_conditions_folder, folder)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_state(self, folder: str):
        save_state_path = self.get_save_state_path(folder)
        with open(save_state_path / "state.txt", "w") as f:
            for attribute, value in self.agent_arrays.items():
                f.write(f"{attribute}\n")
                fp = save_state_path / f"{attribute}.npz"
                np.savez_compressed(fp, data=value.data)

    def restore_state(self, folder: str):
        save_state_path = self.get_save_state_path(folder)
        with open(save_state_path / "state.txt", "r") as f:
            for line in f:
                attribute = line.strip()
                fp = save_state_path / f"{attribute}.npz"
                values = np.load(fp)["data"]
                if not hasattr(self, "max_n"):
                    self.max_n = self.get_max_n(values.shape[0])
                values = AgentArray(values, max_n=self.max_n)

                setattr(self, attribute, values)

    @property
    def agent_arrays(self):
        agent_arrays = {
            name: value
            for name, value in vars(self).items()
            if isinstance(value, AgentArray)
        }
        ids = [id(v) for v in agent_arrays.values()]
        if len(set(ids)) != len(ids):
            duplicate_arrays = [
                name for name, value in agent_arrays.items() if ids.count(id(value)) > 1
            ]
            raise AssertionError(
                f"Duplicate agent array names: {', '.join(duplicate_arrays)}."
            )
        return agent_arrays


from .households import Households
from .farmers import Farmers
from .government import Government
from .reservoir_operators import ReservoirOperators
from .town_managers import TownManagers
from .tourism import Tourism


class Agents:
    """This class initalizes all agent classes, and is used to activate the agents each timestep.

    Args:
        model: The GEB model
    """

    def __init__(self, model) -> None:
        self.model = model
        self.households = Households(model, self, 0.1)
        self.farmers = Farmers(model, self, 0.1)
        self.reservoir_operators = ReservoirOperators(model, self)
        self.town_managers = TownManagers(model, self)
        self.tourism = Tourism(model, self)
        self.government = Government(model, self)

        self.agents = [
            self.households,
            self.farmers,
            self.reservoir_operators,
            self.town_managers,
            self.tourism,
            self.government,
        ]

        if not self.model.load_initial_data:
            self.initiate()
        else:
            self.restore_state()

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
