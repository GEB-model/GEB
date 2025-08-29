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
        """Return the name of the module. This method should be overridden by subclasses."""
        pass

    @abstractmethod
    def spinup(self) -> None:
        """Perform any necessary spinup for the module. This method should be overridden by subclasses."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Perform a single time step of the module. This method should be overridden by subclasses."""
        pass

    def report(self, local_variables) -> None:
        """Used to report data from the module.

        Adds the name of the module to the report and calls the reporter in the model.

        Args:
            module: The
            local_variables: _description_
        """
        self.model.reporter.report(self, local_variables, self.name)
