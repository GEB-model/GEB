"""Coordinates the hydrology evaluation workflow modules.

Workflow modules do the data loading, calculations, dashboards, and plots. Those modules use the hydrology class so
they can be run using the existing ``geb evaluate hydrology.*`` commands (e.g. ``geb evaluate hydrology.create_discharge_dashboard``).
"""

from functools import partialmethod
from pathlib import Path
from typing import TYPE_CHECKING

from geb.evaluate.workflows import (
    dashboard,
    discharge_evaluation,
    discharge_helpers,
    discharge_plots,
    discharge_publication,
    external_skill_scores,
    water_balance_plots,
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

    # Discharge dashboard
    create_discharge_dashboard = dashboard.create_discharge_dashboard

    # Optional data exports; these are not prerequisites for plotting
    export_discharge_publication_data = (
        discharge_publication.export_discharge_publication_data
    )
    export_external_skill_scores = external_skill_scores.export_external_skill_scores
    prepare_external_evaluation = export_external_skill_scores  # Legacy CLI name

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
