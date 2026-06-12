"""Tests for hydrology evaluation plotting helpers."""

import logging
from pathlib import Path

import numpy as np

from geb.evaluate.workflows.hydrology_plot_engine import (
    _get_robust_error_metric_ylim,
    plot_kge_external_model_comparison,
)


def test_get_robust_error_metric_ylim_uses_percentile_range() -> None:
    """Extreme RMSE/RRMSE outliers do not set the visible y-axis range."""
    metric_values: np.ndarray = np.array(
        [
            0.45,
            0.55,
            0.60,
            0.70,
            0.80,
            0.95,
            1.10,
            1.20,
            1.30,
            1.40,
            1.50,
            48.0,
            162.0,
        ],
        dtype=float,
    )

    lower_limit, upper_limit = _get_robust_error_metric_ylim(metric_values)

    assert lower_limit == 0.0
    assert upper_limit < metric_values.max()


def test_get_robust_error_metric_ylim_handles_missing_values() -> None:
    """Missing error metrics get a neutral fallback y-axis range."""
    metric_values: np.ndarray = np.array([np.nan, np.inf, -np.inf], dtype=float)

    assert _get_robust_error_metric_ylim(metric_values) == (0.0, 1.0)


def test_plot_kge_external_model_comparison_writes_outputs(tmp_path: Path) -> None:
    """KGE-only external comparison plot is exported as SVG and PNG."""
    model_kge_values: dict[str, tuple[np.ndarray, np.ndarray, int, float | None]] = {
        "Google Streamflow": (
            np.array([0.2, 0.4, 0.5], dtype=float),
            np.array([0.7, 0.8, 0.9], dtype=float),
            3,
            0.0,
        ),
        "utrecht_1km": (
            np.array([0.3, 0.4, 0.6], dtype=float),
            np.array([0.5, 0.6, 0.7], dtype=float),
            3,
            400.0,
        ),
        "GloFAS": (
            np.array([0.1, 0.2, 0.3], dtype=float),
            np.array([0.2, 0.3, 0.4], dtype=float),
            3,
            0.0,
        ),
    }

    plot_kge_external_model_comparison(
        model_kge_values=model_kge_values,
        output_folder=tmp_path,
        logger=logging.getLogger(__name__),
        export=True,
    )

    assert (tmp_path / "evaluation_skill_scores_kge_external_comparison.svg").exists()
    assert (tmp_path / "evaluation_skill_scores_kge_external_comparison.png").exists()
