"""
Evaluation utilities for the GEB model.

Contains the Evaluate class which contains evaluation routines for model runs.
"""

from __future__ import annotations

from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING

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
        self.meteorological_forecasts = MeteorologicalForecasts(model, self)

    def run(
        self,
        method: str = "hydrology.evaluate_discharge",
        spinup_name: str = "spinup",
        run_name: str = "default",
        include_spinup: bool = False,
        include_yearly_plots: bool = True,
        correct_discharge_observations: bool = False,
    ) -> None:
        """Run a single evaluation method.

        Args:
            method: Fully-qualified method name to run, for example
                `hydrology.evaluate_discharge`.
            spinup_name: Name of the spinup run. Defaults to "spinup".
            run_name: Name of the run to evaluate. Defaults to "default".
            include_spinup: If True, includes the spinup run in the evaluation.
            include_yearly_plots: If True, creates plots for every year showing the evaluation
            correct_discharge_observations: If True, corrects the observed discharge values.

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

        attr(
            spinup_name=spinup_name,
            run_name=run_name,
            include_spinup=include_spinup,
            include_yearly_plots=include_yearly_plots,
            correct_discharge_observations=correct_discharge_observations,
        )

    @property
    def output_folder_evaluate(self) -> Path:
        """Returns the output folder for evaluation results."""
        folder: Path = self.model.output_folder / "evaluate"
        folder.mkdir(parents=True, exist_ok=True)
        return folder
