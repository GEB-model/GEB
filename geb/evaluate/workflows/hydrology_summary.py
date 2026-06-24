"""Orchestration for hydrology summary plots and supporting score tables."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geb.evaluate.hydrology import Hydrology


def create_discharge_skill_score_summary(
    hydrology: Hydrology,
    *,
    export: bool = True,
    minimum_upstream_area_km2: float | None = None,
    external_evaluation_folder: str | Path | None = None,
    include_geb: bool = True,
    matched_only: bool = False,
    include_external: bool = False,
    start_year: int | None = None,
    end_year: int | None = None,
) -> None:
    """Create the complete discharge skill-score summary.

    The workflow coordinates summary maps, external-model table preparation and
    boxplots, and upstream-area plots while leaving data processing and drawing
    to their dedicated modules.

    Args:
        hydrology: Hydrology evaluator exposing the required summary methods.
        export: Whether to save generated figures.
        minimum_upstream_area_km2: Minimum modeled upstream area included (km2).
        external_evaluation_folder: Optional external skill-score data folder.
        include_geb: Whether boxplots include GEB scores.
        matched_only: Whether boxplots use only externally matched stations.
        include_external: Whether non-matched boxplots include external models.
        start_year: First calendar year included in the summary.
        end_year: Last calendar year included in the summary.
    """
    shared_options: dict[str, object] = {
        "export": export,
        "minimum_upstream_area_km2": minimum_upstream_area_km2,
        "start_year": start_year,
        "end_year": end_year,
    }
    hydrology.plot_skill_score_maps(
        **shared_options,
        external_evaluation_folder=external_evaluation_folder,
    )
    hydrology.plot_skill_score_boxplots(
        **shared_options,
        external_evaluation_folder=external_evaluation_folder,
        include_geb=include_geb,
        matched_only=matched_only,
        include_external=include_external,
    )
    hydrology.plot_skill_scores_vs_upstream_area(**shared_options)
