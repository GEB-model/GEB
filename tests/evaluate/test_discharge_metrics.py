"""Unit tests for discharge metrics calculation and rounding."""

import numpy as np
import pandas as pd
import pytest

from geb.evaluate.workflows.discharge_metrics import (
    DISCHARGE_SCORE_COLUMNS,
    DischargeMetrics,
    calculate_discharge_metrics,
    calculate_seasonal_discharge_metrics,
)


def _count_decimal_places(value: float) -> int:
    """Count the number of decimal places of a float up to floating precision limits.

    Args:
        value: Numeric value to inspect.

    Returns:
        Number of decimal digits after the period.
    """
    text: str = f"{value:.8f}".rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".")[1])


def test_calculate_discharge_metrics_happy_path() -> None:
    """Test that calculated discharge metrics are properly rounded to 4 decimals."""
    # Generate data that would naturally produce many decimal digits.
    observed: np.ndarray = np.array(
        [10.12345, 12.67891, 15.23456, 11.78912, 14.34567, 13.90123]
    )
    simulated: np.ndarray = np.array(
        [9.87654, 13.01234, 14.87654, 12.12345, 13.98765, 14.12345]
    )

    df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": observed,
            "discharge_simulations": simulated,
        }
    )

    metrics: DischargeMetrics = calculate_discharge_metrics(df)

    # Verify that all metrics are finite floats
    for metric_name in DischargeMetrics._fields:
        val: float = getattr(metrics, metric_name)
        assert np.isfinite(val), f"Metric {metric_name} should be finite."
        assert round(val, 4) == val, (
            f"Metric {metric_name}={val} is not rounded to 4 decimal places."
        )
        assert _count_decimal_places(val) <= 4


def test_calculate_discharge_metrics_boundary_conditions() -> None:
    """Test boundary cases such as fewer than 2 pairs and constant flows."""
    # Case 1: Fewer than 2 rows
    empty_df: pd.DataFrame = pd.DataFrame(
        {"discharge_observations": [1.0], "discharge_simulations": [1.0]}
    )
    empty_metrics: DischargeMetrics = calculate_discharge_metrics(empty_df)
    for metric_name in DischargeMetrics._fields:
        val: float = getattr(empty_metrics, metric_name)
        assert np.isnan(val), f"Metric {metric_name} should be NaN for < 2 pairs."

    # Case 2: Constant observed flow (std == 0)
    constant_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": [5.0, 5.0, 5.0],
            "discharge_simulations": [4.0, 5.0, 6.0],
        }
    )
    constant_metrics: DischargeMetrics = calculate_discharge_metrics(constant_df)
    assert np.isnan(constant_metrics.KGE)
    assert np.isnan(constant_metrics.KGE_correlation)
    assert np.isnan(constant_metrics.NSE)
    assert np.isfinite(constant_metrics.RMSE)
    assert round(constant_metrics.RMSE, 4) == constant_metrics.RMSE


def test_calculate_discharge_metrics_with_infinities_and_nans() -> None:
    """Test that infinite and missing values are ignored when computing metrics."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": [10.0, np.inf, 15.0, np.nan, 14.0],
            "discharge_simulations": [9.0, 12.0, 14.0, 11.0, -np.inf],
        }
    )
    # Only rows 0 and 2 are valid pairs
    metrics: DischargeMetrics = calculate_discharge_metrics(df)
    assert np.isfinite(metrics.KGE)
    assert round(metrics.KGE, 4) == metrics.KGE


def test_calculate_seasonal_discharge_metrics() -> None:
    """Test seasonal discharge metric calculation and rounding across seasons."""
    dates: pd.DatetimeIndex = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    # Create non-trivial synthetic flows
    rng: np.random.Generator = np.random.default_rng(42)
    obs: np.ndarray = rng.uniform(5.0, 25.0, size=len(dates))
    sim: np.ndarray = obs * 0.9 + rng.normal(0.0, 1.0, size=len(dates))

    daily_df: pd.DataFrame = pd.DataFrame(
        {"discharge_observations": obs, "discharge_simulations": sim},
        index=dates,
    )

    seasonal: dict[str, DischargeMetrics] = calculate_seasonal_discharge_metrics(
        daily_df
    )
    assert set(seasonal.keys()) == {"winter", "spring", "summer", "autumn"}

    for season_name, season_metrics in seasonal.items():
        for field in DischargeMetrics._fields:
            val: float = getattr(season_metrics, field)
            assert np.isfinite(val)
            assert round(val, 4) == val


def test_calculate_seasonal_discharge_metrics_invalid_index() -> None:
    """Test TypeError is raised when input does not have a DatetimeIndex."""
    df_no_datetime: pd.DataFrame = pd.DataFrame(
        {"discharge_observations": [1.0, 2.0], "discharge_simulations": [1.0, 2.0]}
    )
    with pytest.raises(
        TypeError, match="Seasonal discharge evaluation requires a DatetimeIndex"
    ):
        calculate_seasonal_discharge_metrics(df_no_datetime)


def test_station_scores_rounding_with_discharge_score_columns() -> None:
    """Test that station score DataFrame columns are rounded to 4 decimals."""
    station_scores: pd.DataFrame = pd.DataFrame(
        {
            "station_ID": [1, 2],
            "KGE_daily": [0.8123456, 0.7654321],
            "NSE_hourly": [0.9876543, 0.5432109],
        }
    ).set_index("station_ID")

    for score_column in DISCHARGE_SCORE_COLUMNS:
        if score_column in station_scores.columns:
            station_scores[score_column] = station_scores[score_column].round(4)

    assert station_scores.loc[1, "KGE_daily"] == 0.8123
    assert station_scores.loc[2, "KGE_daily"] == 0.7654
    assert station_scores.loc[1, "NSE_hourly"] == 0.9877
    assert station_scores.loc[2, "NSE_hourly"] == 0.5432
