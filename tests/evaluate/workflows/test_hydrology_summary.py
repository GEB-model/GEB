"""Tests for hydrology summary orchestration."""

from pathlib import Path
from typing import Any

from geb.evaluate.workflows.hydrology_summary import (
    create_discharge_skill_score_summary,
)


class SummaryRecorder:
    """Record calls made by the hydrology summary workflow."""

    def __init__(self) -> None:
        """Initialize an empty call list."""
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def plot_skill_score_maps(self, **kwargs: Any) -> None:
        """Record a map request."""
        self.calls.append(("maps", kwargs))

    def plot_skill_score_boxplots(self, **kwargs: Any) -> None:
        """Record a boxplot request."""
        self.calls.append(("boxplots", kwargs))

    def plot_skill_scores_vs_upstream_area(self, **kwargs: Any) -> None:
        """Record an upstream-area plot request."""
        self.calls.append(("upstream_area", kwargs))


def test_create_discharge_skill_score_summary_coordinates_outputs() -> None:
    """Test that summary orchestration forwards shared and specific options."""
    recorder = SummaryRecorder()
    external_folder = Path("external_scores")

    create_discharge_skill_score_summary(
        recorder,
        export=False,
        minimum_upstream_area_km2=400.0,
        external_evaluation_folder=external_folder,
        include_geb=False,
        matched_only=True,
        include_external=True,
        start_year=2014,
        end_year=2021,
    )

    assert [name for name, _ in recorder.calls] == [
        "maps",
        "boxplots",
        "upstream_area",
    ]
    assert recorder.calls[0][1]["external_evaluation_folder"] == external_folder
    assert recorder.calls[1][1]["matched_only"] is True
    assert recorder.calls[2][1]["minimum_upstream_area_km2"] == 400.0
