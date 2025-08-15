from pathlib import Path

from .hydrology import Hydrology


class Evaluate(Hydrology):
    """The main class that implements all evaluation procedures for the GEB model.

    Args:
        model: The GEB model instance.
    """

    def __init__(self, model):
        self.model = model

    def run(
        self,
        methods: list | None = None,
        spinup_name: str = "spinup",
        run_name: str = "default",
        include_spinup: bool = False,
        include_yearly_plots: bool = False,
        correct_Q_obs: bool = False,
    ) -> None:
        """Run the evaluation methods.

        Args:
            methods: List of method names to run. If None, defaults to
                ["plot_discharge", "evaluate_discharge"].
            spinup_name: Name of the spinup run. Defaults to "spinup".
            run_name: Name of the run to evaluate. Defaults to "default".
            include_spinup: If True, includes the spinup run in the evaluation.
            include_yearly_plots: If True, creates plots for every year showing the evaluation
            correct_Q_obs: If True, corrects the observed discharge values.

        Raises:
            AssertionError: If methods is not a list or tuple, or if any method is not a string.
            ValueError: If a specified method is not implemented in the Evaluate class.

        Returns:
            None
        """
        if methods is None:
            methods: list = [
                "plot_discharge",
                "evaluate_discharge",
                # "water_circle",
                "water_balance",
                "evaluate_hydrodynamics",
                "plot_discharge_floods",
            ]
        else:
            assert isinstance(methods, (list, tuple)), (
                "Methods should be a list or tuple."
            )
            assert all(isinstance(method, str) for method in methods), (
                "All methods should be strings."
            )

        for method in methods:
            assert hasattr(self, method), (
                f"Method {method} is not implemented in Evaluate class."
            )

        for method in methods:
            if hasattr(self, method):
                attr = getattr(self, method)
            else:
                raise ValueError(
                    f"Method {method} is not implemented in Evaluate class."
                )
            attr(
                spinup_name=spinup_name,
                run_name=run_name,
                include_spinup=include_spinup,
                include_yearly_plots=include_yearly_plots,
                correct_Q_obs=correct_Q_obs,
            )  # this calls the method and executes them

    @property
    def output_folder_evaluate(self) -> Path:
        """Returns the output folder for evaluation results."""
        folder: Path = self.model.output_folder / "evaluate"
        folder.mkdir(parents=True, exist_ok=True)
        return folder
