"""
Evaluation utilities for the GEB model.

Contains the Evaluate class which contains evaluation routines for model runs.
"""

from __future__ import annotations

from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .energy import Energy
from .hydrology import Hydrology
from .meteorological_forecasts import MeteorologicalForecasts

if TYPE_CHECKING:
    from geb.model import GEBModel


class Evaluate:
    """The main class that implements all evaluation procedures for the GEB model.

    Args:
        model: The GEB model instance.
    """

    def __init__(self, model: GEBModel) -> None:
        """Initialize the Evaluate class."""
        self.model: GEBModel = model
        self.hydrology = Hydrology(model, self)
        self.energy = Energy(model, self)
        self.meteorological_forecasts = MeteorologicalForecasts(model, self)

    @property
    def sub_evaluators(self) -> list[str]:
        """Returns a list of available sub-evaluators."""
        return [
            attr
            for attr, value in self.__dict__.items()
            if not attr.startswith("_") and attr != "model"
        ]

    def run(
        self,
        method: str,
        spinup_name: str = "spinup",
        run_name: str = "default",
        **kwargs: Any,
    ) -> Any:
        """Run a single evaluation method.

        Args:
            method: Fully-qualified method name to run, for example
                `hydrology.evaluate_discharge`.
            spinup_name: Name of the spinup run. Defaults to "spinup".
            run_name: Name of the run to evaluate. Defaults to "default".
            **kwargs: Additional keyword arguments to pass to the evaluation method.

        Returns:
            The result of the evaluation method.

        Raises:
            AttributeError: If the specified method is not implemented in the Evaluate class.
            TypeError: If method is not a string.
        """
        if not isinstance(method, str):
            raise TypeError("Method should be a string.")

        try:
            attr = attrgetter(method)(self)
        except AttributeError as exc:
            raise AttributeError(
                f"Method {method} is not implemented in Evaluate class."
            ) from exc

        # Merge spinup_name and run_name into kwargs to pass them all as keyword arguments
        all_kwargs = {
            "spinup_name": spinup_name,
            "run_name": run_name,
            **kwargs,
        }

        # Run the method and return the result
        return attr(**all_kwargs)

    @property
    def output_folder_evaluate(self) -> Path:
        """Returns the output folder for evaluation results."""
        folder: Path = self.model.output_folder / "evaluate"
        folder.mkdir(parents=True, exist_ok=True)
        return folder
