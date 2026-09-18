"""Coordinates the hydrology evaluation workflow modules.

Workflow modules do the data loading, calculations, dashboards, and plots.
"""

from functools import partialmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from geb.evaluate.workflows import (
    dashboard,
    discharge_evaluation,
    discharge_helpers,
    discharge_plots,
    discharge_publication,
    external_skill_scores,
    water_balance_plots,
)
from geb.evaluate.workflows.water_balance_helpers import (
    _get_datetime_index_step_label as _get_datetime_index_step_label,
)

if TYPE_CHECKING:
    from geb.evaluate import Evaluate
    from geb.model import GEBModel


class Hydrology:
    """Expose discharge, dashboard, water-balance, and storage evaluation commands."""

    def __init__(self, model: GEBModel, evaluator: Evaluate) -> None:
        """Initialize the Hydrology evaluation module."""
        self.model = model
        self.evaluator = evaluator

    # Discharge evaluation and data access
    evaluate_discharge = discharge_evaluation.evaluate_discharge
    get_discharge_per_river = discharge_helpers.get_discharge_per_river

    # Discharge plots (load and match external scores automatically)
    plot_discharge = discharge_plots.plot_discharge
    plot_discharge_skill_scores = discharge_plots.plot_discharge_skill_scores
    plot_skill_score_maps = partialmethod(plot_discharge_skill_scores, plots=("maps",))
    plot_skill_score_boxplots = partialmethod(
        plot_discharge_skill_scores, plots=("boxplots",)
    )
    plot_skill_scores_vs_upstream_area = partialmethod(
        plot_discharge_skill_scores, plots=("upstream_area",)
    )
    plot_discharge_characteristics = partialmethod(
        plot_discharge_skill_scores, plots=("characteristics",)
    )

    def create_discharge_dashboard(
        self,
        run_name: str = "default",
        correct_discharge_observations: bool = False,
        output_filename: str = "discharge_evaluation_map.html",
        include_return_period_plots: bool = True,
    ) -> dict[str, str]:
        """Create a dashboard from saved discharge evaluation metrics.

        Args:
            run_name: Simulation run to display.
            correct_discharge_observations: Apply the station-to-model area ratio.
            output_filename: Filename relative to the evaluation folder, or an absolute path.
            include_return_period_plots: Include return-period curves; defaults to True.

        Returns:
            Path to the created dashboard, keyed by ``dashboard``.
        """
        return dashboard.create_discharge_dashboard(
            evaluation_folder=self.evaluate_discharge_output_folder,
            run_output_folder=self.evaluator.output_folder_evaluate.parent,
            geometry_files=self.model.files["geom"],
            table_files=self.model.files["table"],
            minimum_upstream_area_km2=self.model.config["hydrology"]["evaluation"][
                "discharge"
            ]["minimum_upstream_area_km2"],
            logger=self.model.logger,
            correct_discharge_observations=correct_discharge_observations,
            output_filename=output_filename,
            include_return_period_plots=include_return_period_plots,
        )

    # Optional data exports; these are not prerequisites for plotting
    export_discharge_publication_data = (
        discharge_publication.export_discharge_publication_data
    )

    def export_external_skill_scores(self, **kwargs: Any) -> dict[str, pd.DataFrame]:
        """Export external skill scores matched to this model's stations.

        Args:
            **kwargs: Evaluation CLI arguments; no export settings are required.

        Returns:
            Matched station scores keyed by external model name.
        """
        return external_skill_scores.export_external_skill_scores(
            input_folder=self.model.input_folder,
            output_folder=self.evaluate_discharge_output_folder,
            snapped_locations_path=self.model.files["geom"][
                "discharge/discharge_snapped_locations"
            ],
            logger=self.model.logger,
        )

    prepare_external_evaluation = export_external_skill_scores

    # Water-circle, water-balance, and water-storage plots
    plot_water_circle = water_balance_plots.plot_water_circle
    plot_water_balance = water_balance_plots.plot_water_balance
    plot_water_storage = water_balance_plots.plot_water_storage

    # Output folders
    @property
    def discharge_output_folder(self) -> Path:
        """Path to the folder where discharge map outputs are stored."""
        folder = self.evaluator.output_folder_evaluate / "hydrology" / "discharge"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def evaluate_discharge_output_folder(self) -> Path:
        """Path to the folder where discharge evaluation outputs are stored."""
        folder = (
            self.evaluator.output_folder_evaluate / "hydrology" / "evaluate_discharge"
        )
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def water_circle_output_folder(self) -> Path:
        """Path to the folder where water circle outputs are stored."""
        folder = self.evaluator.output_folder_evaluate / "hydrology" / "water_circle"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def water_balance_output_folder(self) -> Path:
        """Path to the folder where water balance outputs are stored."""
        folder = self.evaluator.output_folder_evaluate / "hydrology" / "water_balance"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def water_storage_output_folder(self) -> Path:
        """Path to the folder where water storage outputs are stored."""
        folder = self.evaluator.output_folder_evaluate / "hydrology" / "water_storage"
        folder.mkdir(parents=True, exist_ok=True)
        return folder
