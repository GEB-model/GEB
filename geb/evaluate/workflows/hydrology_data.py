"""Data-processing helpers for hydrology evaluation workflows."""

from typing import Any

import numpy as np
import pandas as pd


def get_discharge_evaluation_minimum_upstream_area_km2(
    config: dict[str, Any],
    minimum_upstream_area_km2: float | None = None,
) -> float:
    """Resolve the minimum upstream area threshold for discharge evaluation.

    Args:
        config: Model configuration dictionary.
        minimum_upstream_area_km2: Optional explicit threshold override (km2).

    Returns:
        Minimum modeled upstream area required for discharge stations (km2).

    Raises:
        ValueError: If the threshold is negative or non-finite.
    """
    if minimum_upstream_area_km2 is None:
        minimum_upstream_area_km2 = config["hydrology"]["evaluation"]["discharge"][
            "minimum_upstream_area_km2"
        ]

    threshold_km2: float = float(minimum_upstream_area_km2)
    if not np.isfinite(threshold_km2) or threshold_km2 < 0.0:
        raise ValueError(
            "minimum_upstream_area_km2 must be a finite value greater than or equal to 0."
        )
    return threshold_km2


def has_minimum_upstream_area(
    upstream_area_m2: float,
    minimum_upstream_area_km2: float,
) -> bool:
    """Check whether modeled upstream area meets a threshold.

    Args:
        upstream_area_m2: Modeled upstream area (m2).
        minimum_upstream_area_km2: Minimum modeled upstream area threshold (km2).

    Returns:
        True when the modeled upstream area is finite and at least the threshold.
    """
    return bool(
        np.isfinite(upstream_area_m2)
        and upstream_area_m2 >= minimum_upstream_area_km2 * 1_000_000.0
    )


def filter_evaluation_by_minimum_upstream_area(
    evaluation_df: pd.DataFrame,
    minimum_upstream_area_km2: float,
) -> pd.DataFrame:
    """Filter discharge evaluation metrics by modeled upstream area.

    Args:
        evaluation_df: Evaluation metrics with `upstream_area_GEB` (m2).
        minimum_upstream_area_km2: Minimum modeled upstream area threshold (km2).

    Returns:
        Filtered evaluation metrics.

    Raises:
        ValueError: If `upstream_area_GEB` is missing.
    """
    if "upstream_area_GEB" not in evaluation_df.columns:
        raise ValueError("`upstream_area_GEB` is missing from evaluation metrics.")

    threshold_m2: float = minimum_upstream_area_km2 * 1_000_000.0
    upstream_area_filter: pd.Series = evaluation_df["upstream_area_GEB"] >= threshold_m2
    return evaluation_df.loc[upstream_area_filter].copy()
