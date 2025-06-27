from abc import ABC, abstractmethod
from typing import Any


class Module(ABC):
    """Base class for all modules."""

    def __init__(self, model, create_var: bool = True, var_validator=None):
        self.model = model
        if create_var:
            self.var = self.model.store.create_bucket(
                f"{self.name}.var", validator=var_validator
            )

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def spinup(self):
        pass

    @abstractmethod
    def step(self) -> Any | tuple[Any, ...]:
        pass

    def report(self, module, local_variables):
        self.model.reporter.report(module, local_variables, self.name)
