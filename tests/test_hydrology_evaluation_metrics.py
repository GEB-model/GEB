"""Tests for hydrology evaluation metric calculations."""

import math

import pandas as pd
import pytest

from geb.evaluate.hydrology import (
    DischargeMetrics,
    _calculate_discharge_validation_metrics,
)


def test_calculate_discharge_validation_metrics_known_values() -> None:
    """Test discharge metrics against hand-calculated values."""
    validation_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": [1.0, 2.0, 3.0, 4.0],
            "discharge_simulations": [1.1, 1.9, 3.2, 3.8],
        }
    )

    metrics: DischargeMetrics = _calculate_discharge_validation_metrics(validation_df)

    assert metrics.KGE == pytest.approx(0.947873410771797)
    assert metrics.KGE_modified == pytest.approx(0.9478734107717969)
    assert metrics.KGE_correlation == pytest.approx(0.9908470001860921)
    assert metrics.KGE_bias_ratio == pytest.approx(1.0)
    assert metrics.KGE_variability_ratio == pytest.approx(0.9486832980505137)
    assert metrics.NSE == pytest.approx(0.98)
    assert metrics.R2 == pytest.approx(0.9817777777777775)
    assert metrics.RMSE == pytest.approx(0.1581138830084191)
    assert metrics.RRMSE == pytest.approx(0.14142135623730961)


def test_calculate_discharge_validation_metrics_filters_missing_pairs() -> None:
    """Test that metrics use only paired observed and simulated discharge values."""
    validation_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": [1.0, math.nan, 3.0, 4.0],
            "discharge_simulations": [1.0, 2.0, math.nan, 4.0],
        }
    )

    metrics: DischargeMetrics = _calculate_discharge_validation_metrics(validation_df)

    assert metrics.KGE == pytest.approx(1.0)
    assert metrics.KGE_modified == pytest.approx(1.0)
    assert metrics.KGE_correlation == pytest.approx(1.0)
    assert metrics.NSE == pytest.approx(1.0)
    assert metrics.RMSE == pytest.approx(0.0)


def test_calculate_discharge_validation_metrics_requires_two_pairs() -> None:
    """Test that stations with too few discharge pairs receive missing metrics."""
    validation_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": [1.0, math.nan],
            "discharge_simulations": [1.0, 2.0],
        }
    )

    metrics: DischargeMetrics = _calculate_discharge_validation_metrics(validation_df)

    assert all(math.isnan(metric_value) for metric_value in metrics)


def test_calculate_discharge_validation_metrics_constant_observations() -> None:
    """Test undefined variance-based metrics with constant observed discharge."""
    validation_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": [2.0, 2.0, 2.0],
            "discharge_simulations": [1.0, 2.0, 3.0],
        }
    )

    metrics: DischargeMetrics = _calculate_discharge_validation_metrics(validation_df)

    assert math.isnan(metrics.KGE)
    assert math.isnan(metrics.KGE_modified)
    assert math.isnan(metrics.KGE_correlation)
    assert metrics.KGE_bias_ratio == pytest.approx(1.0)
    assert math.isnan(metrics.KGE_variability_ratio)
    assert math.isnan(metrics.NSE)
    assert math.isnan(metrics.R2)
    assert metrics.RMSE == pytest.approx(math.sqrt(2.0 / 3.0))
    assert math.isnan(metrics.RRMSE)
