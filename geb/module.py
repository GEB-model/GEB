from abc import ABC, abstractmethod


class Module(ABC):
    """Base class for all modules."""

    def __init__(self, model, create_var: bool = True, var_validator=None) -> None:
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
    def spinup(self) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    def report(self, module, local_variables) -> None:
        self.model.reporter.report(module, local_variables, self.name)
