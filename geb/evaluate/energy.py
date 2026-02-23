"""Module implementing energy evaluation functions for the GEB model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from geb.evaluate import Evaluate
    from geb.model import GEBModel


class Energy:
    """Implements several functions to evaluate the energy module of GEB."""

    def __init__(self, model: GEBModel, evaluator: Evaluate) -> None:
        """Initialize the Energy evaluation module.

        Args:
            model: The GEB model instance.
            evaluator: The main evaluator instance.
        """
        self.model = model
        self.evaluator = evaluator

    def plot_soil_temperature(
        self, run_name: str = "default", *args: Any, **kwargs: Any
    ) -> None:
        """Plot the soil temperature of the different layers as a time series for the entire basin.

        This function reads the reported soil temperature CSV files for each layer and creates
         a consolidated time series plot showing the mean temperature (weighted by area)
         across the entire basin. The generated plot is saved as a PNG image in the
         evaluation output directory.

        Notes:
            The model run must have been completed with the `_energy_balance` reporting module
            activated for these files to exist. The files are expected in the
            `hydrology.landsurface` subfolder of the model report.

        Args:
            run_name: Name of the simulation run to plot. This corresponds to the name
                assigned to the run in the model configuration. Defaults to "default".
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Raises:
            FileNotFoundError: If the report directory or the soil temperature CSV files
                cannot be found.
        """
        # Define the folder where reports are stored
        report_dir = (
            self.model.output_folder / "report" / run_name / "hydrology.landsurface"
        )

        if not report_dir.exists():
            raise FileNotFoundError(
                f"Report directory '{report_dir}' does not exist. Did you run the model with energy balance reporting enabled?"
            )

        fig, ax = plt.subplots(figsize=(12, 6))

        energy_balance_soil_layer_files: list[Path] = []
        for layer in range(6):
            energy_balance_soil_layer_file = (
                report_dir / f"_energy_balance_soil_temperature_layer_{layer}_C.csv"
            )
            energy_balance_soil_layer_files.append(energy_balance_soil_layer_file)
            if not energy_balance_soil_layer_file.exists():
                raise FileNotFoundError(
                    f"Expected soil temperature file '{energy_balance_soil_layer_file.name}' not found."
                )

            df = pd.read_csv(energy_balance_soil_layer_file, index_col=0)
            df[f"_energy_balance_soil_temperature_layer_{layer}_C"].plot(
                ax=ax, label=f"Layer {layer + 1}"
            )  # This will plot the time series of the soil temperature for this layer

        ax.set_title(f"Soil Temperature Profile - Run: {run_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature (°C)")
        ax.legend(title="Soil Layers")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.tight_layout()

        # Save the plot to the evaluation output folder
        energy_plot_folder = self.evaluator.output_folder_evaluate / "energy"
        energy_plot_folder.mkdir(parents=True, exist_ok=True)

        output_file = energy_plot_folder / f"soil_temperature_{run_name}.svg"
        plt.savefig(output_file)
        plt.close()

        print(f"Soil temperature plot saved to {output_file}")
