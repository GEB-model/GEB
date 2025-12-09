"""Base class for all modules in GEB."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

from geb.store import Bucket

if TYPE_CHECKING:
    from geb.model import GEBModel


class Module(ABC):
    """Base class for all modules."""

    def __init__(
        self,
        model: GEBModel,
        create_var: bool = True,
        var_validator: None | Callable = None,
    ) -> None:
        """Initialize the module.

        Args:
            model: The GEB model instance.
            create_var: Whether to create a variable bucket for this module.
            var_validator: Optional validator function for the variable bucket.
                Must return True if the variable is valid, False otherwise.
        """
        self.model: GEBModel = model
        if create_var:
            self.var: Bucket = self.model.store.create_bucket(
                f"{self.name}.var", validator=var_validator
            )

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the module. This method should be overridden by subclasses."""
        pass

    @abstractmethod
    def spinup(self) -> None:
        """Perform any necessary spinup for the module. This method should be overridden by subclasses."""
        pass

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Perform a single time step of the module. This method should be overridden by subclasses."""
        pass

    def report(self, local_variables: dict[str, Any]) -> None:
        """
        Used to report data from the module.

        Adds the name of the module to the report and calls the reporter in the model.

        Args:
            local_variables: A dictionary of local variables to report, typically the result of locals().
        """
        self.model.reporter.report(self, local_variables, self.name)
