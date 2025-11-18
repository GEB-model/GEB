"""Tests for return period assignment functionality.

This module tests the assign_return_periods function which calculates
flood return period discharge values using the GPD-POT (Generalized
Pareto Distribution - Peaks Over Threshold) method.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from geb.hazards.floods.workflows.utils import assign_return_periods


def test_assign_return_periods_basic() -> None:
    """Test basic functionality of assign_return_periods.

    Tests that the function correctly assigns return period discharge values
    to rivers using GPD-POT method with synthetic data.
    """
    # Create test rivers GeoDataFrame
    rivers = gpd.GeoDataFrame(
        {
            "geometry": [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        },
        index=[1, 2],
    )

    # Create test discharge data with datetime index
    dates = pd.date_range("2000-01-01", periods=1000, freq="D")
    np.random.seed(42)

    # Generate synthetic discharge data with some variability
    discharge_data = {
        1: np.random.lognormal(mean=2, sigma=0.5, size=1000),
        2: np.random.lognormal(mean=1.5, sigma=0.7, size=1000),
    }

    discharge_df = pd.DataFrame(discharge_data, index=dates)

    return_periods = [2, 10, 100]

    result = assign_return_periods(
        rivers=rivers,
        discharge_dataframe=discharge_df,
        return_periods=return_periods,
        min_exceed=20,
        nboot=100,  # Reduced for faster testing
    )

    # Check that all return period columns were added
    for T in return_periods:
        assert f"Q_{T}" in result.columns

    # Check that values are positive and reasonable
    for river_id in rivers.index:
        for T in return_periods:
            value = result.loc[river_id, f"Q_{T}"]
            assert not pd.isna(value)
            assert value > 0

    # Check that higher return periods have higher discharge values
    for river_id in rivers.index:
        q2 = result.loc[river_id, "Q_2"]
        q10 = result.loc[river_id, "Q_10"]
        q100 = result.loc[river_id, "Q_100"]
        assert q2 <= q10 <= q100


def test_assign_return_periods_zero_discharge() -> None:
    """Test assign_return_periods with zero discharge values.

    Verifies that rivers with all-zero discharge get assigned zero values
    for all return periods.
    """
    rivers = gpd.GeoDataFrame(
        {
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        index=[1],
    )

    dates = pd.date_range("2000-01-01", periods=365, freq="D")
    discharge_df = pd.DataFrame({1: np.zeros(365)}, index=dates)

    return_periods = [2, 10]

    result = assign_return_periods(
        rivers=rivers,
        discharge_dataframe=discharge_df,
        return_periods=return_periods,
        nboot=50,
    )

    for T in return_periods:
        assert result.loc[1, f"Q_{T}"] == 0.0


def test_assign_return_periods_custom_prefix() -> None:
    """Test assign_return_periods with custom column prefix."""
    rivers = gpd.GeoDataFrame(
        {
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        index=[1],
    )

    dates = pd.date_range("2000-01-01", periods=500, freq="D")
    np.random.seed(123)
    discharge_df = pd.DataFrame(
        {1: np.random.exponential(scale=50, size=500)}, index=dates
    )

    return_periods = [5, 25]
    custom_prefix = "DISCHARGE"

    result = assign_return_periods(
        rivers=rivers,
        discharge_dataframe=discharge_df,
        return_periods=return_periods,
        prefix=custom_prefix,
        nboot=50,
    )

    for T in return_periods:
        assert f"{custom_prefix}_{T}" in result.columns
        assert result.loc[1, f"{custom_prefix}_{T}"] > 0


def test_assign_return_periods_extreme_values() -> None:
    """Test assign_return_periods with extreme discharge values.

    Verifies that the warning threshold of 400,000 m³/s is applied
    and values are capped at 2,000 m³/s.
    """
    rivers = gpd.GeoDataFrame(
        {
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        index=[1],
    )

    dates = pd.date_range("2000-01-01", periods=1000, freq="D")
    # Create data that will likely produce extreme return levels
    discharge_data = np.concatenate(
        [
            np.random.lognormal(mean=5, sigma=2, size=950),
            np.random.lognormal(mean=15, sigma=1, size=50),  # Extreme values
        ]
    )

    discharge_df = pd.DataFrame({1: discharge_data}, index=dates)

    return_periods = [1000]  # High return period likely to trigger warning

    result = assign_return_periods(
        rivers=rivers,
        discharge_dataframe=discharge_df,
        return_periods=return_periods,
        nboot=50,
    )

    # The value should be capped if it exceeds 400,000
    q_value = result.loc[1, "Q_1000"]
    if q_value == 2000.0:
        # Warning was triggered and value was capped
        assert True
    else:
        # Value was reasonable and not capped
        assert q_value <= 400_000


def test_assign_return_periods_insufficient_data() -> None:
    """Test assign_return_periods with insufficient exceedances.

    Verifies handling when there aren't enough exceedances for reliable fitting.
    """
    rivers = gpd.GeoDataFrame(
        {
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        index=[1],
    )

    # Very short time series with little variation
    dates = pd.date_range("2000-01-01", periods=50, freq="D")
    discharge_df = pd.DataFrame(
        {1: np.random.normal(loc=10, scale=1, size=50)}, index=dates
    )

    return_periods = [2, 10]

    result = assign_return_periods(
        rivers=rivers,
        discharge_dataframe=discharge_df,
        return_periods=return_periods,
        min_exceed=40,  # Require more exceedances than available
        nboot=50,
    )

    # Should handle gracefully - either assign NaN or reasonable values
    for T in return_periods:
        value = result.loc[1, f"Q_{T}"]
        assert pd.isna(value) or (value >= 0)


def test_assign_return_periods_invalid_datetime_index() -> None:
    """Test assign_return_periods with non-datetime index.

    Verifies that TypeError is raised when discharge data doesn't have
    a DatetimeIndex.
    """
    rivers = gpd.GeoDataFrame(
        {
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        index=[1],
    )

    # Create DataFrame with integer index instead of datetime
    discharge_df = pd.DataFrame(
        {1: np.random.exponential(scale=20, size=365)}, index=range(365)
    )

    return_periods = [2, 10]

    with pytest.raises(TypeError, match="must have a DatetimeIndex"):
        assign_return_periods(
            rivers=rivers,
            discharge_dataframe=discharge_df,
            return_periods=return_periods,
            nboot=50,
        )


def test_assign_return_periods_multiple_rivers() -> None:
    """Test assign_return_periods with multiple rivers.

    Ensures the function correctly processes multiple rivers with
    different discharge characteristics.
    """
    rivers = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
                LineString([(2, 2), (3, 3)]),
            ],
        },
        index=[10, 20, 30],
    )

    dates = pd.date_range("2000-01-01", periods=800, freq="D")
    np.random.seed(456)

    discharge_df = pd.DataFrame(
        {
            10: np.random.gamma(shape=2, scale=30, size=800),
            20: np.random.lognormal(mean=3, sigma=0.8, size=800),
            30: np.random.weibull(a=1.5, size=800) * 100,
        },
        index=dates,
    )

    return_periods = [5, 20, 50]

    result = assign_return_periods(
        rivers=rivers,
        discharge_dataframe=discharge_df,
        return_periods=return_periods,
        nboot=100,
    )

    # Check all rivers have values assigned
    for river_id in rivers.index:
        for T in return_periods:
            value = result.loc[river_id, f"Q_{T}"]
            assert not pd.isna(value)
            assert value >= 0

    # Check monotonic increase within each river
    for river_id in rivers.index:
        q5 = result.loc[river_id, "Q_5"]
        q20 = result.loc[river_id, "Q_20"]
        q50 = result.loc[river_id, "Q_50"]
        assert q5 <= q20 <= q50
