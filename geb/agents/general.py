"""Module containing general agent functions and the base class for all agents."""

from typing import TYPE_CHECKING

from geb.module import Module
from geb.store import DynamicArray

if TYPE_CHECKING:
    from geb.model import GEBModel


class AgentBaseClass(Module):
    """Base class for all agent classes."""

    def __init__(self, model: GEBModel) -> None:
        """Initialize the agent base class.

        Args:
            model: The GEB model instance.
        """
        Module.__init__(self, model)

    @property
    def agent_arrays(self) -> dict[str, DynamicArray]:
        """Return a dictionary of all DynamicArray attributes of the agent.

        Raises:
            AssertionError: If there are duplicate DynamicArray attributes.
        """
        agent_arrays = {
            name: value
            for name, value in vars(self.var).items()
            if isinstance(value, DynamicArray)
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
