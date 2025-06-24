from pathlib import Path

from .hydrology import Hydrology


class Evaluate(Hydrology):
    """The main class that implements all evaluation procedures for the GEB model.

    Args:
        model: The GEB model instance.
    """

    def __init__(self, model):
        self.model = model

    def run(self, methods: list | None = None) -> None:
        """Run the evaluation methods.

        Args:
            methods (list, optional): List of method names to run. If None, defaults to
                ["plot_discharge", "evaluate_discharge"].

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
            attr()

    @property
    def output_folder_evaluate(self) -> Path:
        """Returns the output folder for evaluation results."""
        folder: Path = self.model.output_folder / "evaluate"
        folder.mkdir(parents=True, exist_ok=True)
        return folder
