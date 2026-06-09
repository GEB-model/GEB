"""Tests for hydrology evaluation plotting helpers."""

import numpy as np

from geb.evaluate.workflows.hydrology_plot_engine import _get_robust_error_metric_ylim


def test_get_robust_error_metric_ylim_uses_boxplot_whisker_range() -> None:
    """Extreme RMSE/RRMSE outliers do not set the visible y-axis range."""
    metric_values: np.ndarray = np.array(
        [0.45, 0.55, 0.60, 0.70, 0.80, 0.95, 1.10, 1.20, 48.0, 162.0],
        dtype=float,
    )

    lower_limit, upper_limit = _get_robust_error_metric_ylim(metric_values)

    assert lower_limit == 0.0
    assert upper_limit < 10.0


def test_get_robust_error_metric_ylim_handles_missing_values() -> None:
    """Missing error metrics get a neutral fallback y-axis range."""
    metric_values: np.ndarray = np.array([np.nan, np.inf, -np.inf], dtype=float)

    assert _get_robust_error_metric_ylim(metric_values) == (0.0, 1.0)
