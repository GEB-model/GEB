"""Tests for return period assignment functionality.

This module tests the assign_return_periods function which calculates
flood return period discharge values using the GPD-POT (Generalized
Pareto Distribution - Peaks Over Threshold) method.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto
from shapely.geometry import LineString

from geb.hazards.floods.workflows.return_periods import (
    assign_return_periods,
    fit_gpd_lmom,
    fit_gpd_mle,
)


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

    Verifies that the function handles extreme discharge values appropriately
    and produces reasonable return period estimates.
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

    with pytest.raises(ValueError):
        result = assign_return_periods(
            rivers=rivers,
            discharge_dataframe=discharge_df,
            return_periods=return_periods,
            nboot=50,
        )


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

    with pytest.raises(ValueError):
        result = assign_return_periods(
            rivers=rivers,
            discharge_dataframe=discharge_df,
            return_periods=return_periods,
            min_exceed=40,  # Require more exceedances than available
            nboot=50,
        )


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


def test_fit_gpd_mle_fixes() -> None:
    """Test fit_gpd_mle with fixed parameters."""
    # Generate data from GPD(xi=0.5, scale=2.0)
    # y = x - u. We simulate y directly.
    np.random.seed(42)
    xi_true = 0.5
    sigma_true = 2.0
    # genpareto args: c (shape), loc=0, scale
    y = genpareto.rvs(c=xi_true, scale=sigma_true, size=100)

    # 1. Test default fitting (no fixes)
    sigma_hat, xi_hat = fit_gpd_mle(y)
    # Tolerances can be loose due to sample size
    assert abs(xi_hat - xi_true) < 0.2
    assert abs(sigma_hat - sigma_true) < 0.5

    # 2. Test fixed shape = 0 (Exponential / Gumbel-like)
    sigma_hat_0, xi_hat_0 = fit_gpd_mle(y, fixed_shape=0.0)
    assert xi_hat_0 == 0.0

    # 3. Test fixed shape = 0.5 (True value)
    sigma_hat_fixed, xi_hat_fixed = fit_gpd_mle(y, fixed_shape=0.5)
    assert xi_hat_fixed == 0.5
    assert abs(sigma_hat_fixed - sigma_true) < 0.3

    # 4. Test fixed scale = 2.0
    sigma_hat_scale, xi_hat_scale = fit_gpd_mle(y, fixed_scale=2.0)
    assert sigma_hat_scale == 2.0
    assert abs(xi_hat_scale - xi_true) < 0.2


def test_fit_gpd_lmom_fixes() -> None:
    """Test fit_gpd_lmom with fixed parameters."""
    # Generate data from GPD(xi=0.2, scale=10.0)
    # L-moments are generally robust but let's use a decent sample size
    np.random.seed(42)
    xi_true = 0.2
    sigma_true = 10.0
    n = 1000
    y = genpareto.rvs(c=xi_true, scale=sigma_true, size=n)

    # 1. Test standard fitting (no fixes)
    sigma_lmom, xi_lmom = fit_gpd_lmom(y)
    # L-moments should be quite accurate for this sample size and shape > -0.5
    assert abs(xi_lmom - xi_true) < 0.1
    assert abs(sigma_lmom - sigma_true) < 1.0

    # 2. Test fixed shape = 0.0 (Gumbel-like)
    # We fit to the same data (which is NOT Gumbel), so fit will be biased but
    # parameter should be fixed.
    sigma_0, xi_0 = fit_gpd_lmom(y, fixed_shape=0.0)
    assert xi_0 == 0.0
    # For xi=0, sigma estimate via L-moments is l1 * (1 - xi) = l1 * 1 = mean
    assert abs(sigma_0 - np.mean(y)) < 1e-9

    # 3. Test fixed shape = 0.2 (True value)
    sigma_fixed, xi_fixed = fit_gpd_lmom(y, fixed_shape=0.2)
    assert xi_fixed == 0.2
    assert abs(sigma_fixed - sigma_true) < 0.5

    # 4. Test fixed scale = 10.0
    sigma_scale, xi_scale = fit_gpd_lmom(y, fixed_scale=10.0)
    assert sigma_scale == 10.0
    assert abs(xi_scale - xi_true) < 0.1

    # 5. Test with exponential data (xi=0) to verify shape convergence
    y_exp = np.random.exponential(scale=5.0, size=1000)
    sigma_exp, xi_exp = fit_gpd_lmom(y_exp)
    assert abs(xi_exp - 0.0) < 0.1
    assert abs(sigma_exp - 5.0) < 0.5


def test_assign_return_periods_lmom() -> None:
    """Test assign_return_periods using L-moments method."""
    # Create simple river data
    idx = ["river1"]
    geom = [LineString([(0, 0), (1, 1)])]
    rivers = gpd.GeoDataFrame({"geometry": geom}, index=idx, crs="EPSG:4326")

    # Generate discharge with GPD tail
    np.random.seed(123)
    n_days = 365 * 10
    dates = pd.date_range("2000-01-01", periods=n_days, freq="D")

    # Base flow + random exceeding term
    # Most days low, occasional peaks
    base = np.random.uniform(10, 20, size=n_days)
    peaks = genpareto.rvs(c=0.1, loc=0, scale=50, size=n_days)
    # Only keep peaks occasionally
    # Make a series where most are just base, some are base+peak
    mask = np.random.random(n_days) > 0.95
    flow = base.copy()
    flow[mask] += peaks[mask]

    discharge_df = pd.DataFrame({"river1": flow}, index=dates)

    return_periods = [10, 100]

    # Run assignment with L-moments
    result = assign_return_periods(
        rivers=rivers.copy(),
        discharge_dataframe=discharge_df,
        return_periods=return_periods,
        fit_method="lmom",
        nboot=10,  # fast test
    )

    assert "Q_10" in result.columns
    assert "Q_100" in result.columns
    assert result.loc["river1", "Q_10"] > 0
    assert result.loc["river1", "Q_100"] > result.loc["river1", "Q_10"]
