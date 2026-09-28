"""Tests for extreme value analysis workflow and ReturnPeriodModel."""

import numpy as np
import pandas as pd
import pytest

from geb.workflows.extreme_value_analysis import (
    ReturnPeriodModel,
    bootstrap_pvalue_for_ad,
    fit_gpd_lmoments,
)


@pytest.fixture
def synthetic_daily_discharge() -> pd.Series:
    """Create synthetic daily discharge series for testing.

    Returns:
        pd.Series: Daily discharge series over 365 days.
    """
    rng: np.random.Generator = np.random.default_rng(42)
    dates: pd.DatetimeIndex = pd.date_range("2020-01-01", periods=365, freq="D")
    # Base seasonal flow + random noise + occasional extreme runoff peaks
    base_flow: np.ndarray = 10.0 + 5.0 * np.sin(np.linspace(0, 2 * np.pi, 365))
    noise: np.ndarray = rng.exponential(scale=3.0, size=365)
    flow: np.ndarray = base_flow + noise
    # Add distinct flood peaks separated by more than 7 days
    peak_days: list[int] = [30, 75, 120, 180, 220, 280, 330]
    for day in peak_days:
        flow[day] += 50.0

    return pd.Series(flow, index=dates, name="discharge")


def test_return_period_model_fixed_quantile(
    synthetic_daily_discharge: pd.Series,
) -> None:
    """Test ReturnPeriodModel with a fixed quantile threshold.

    Verifies that specifying fixed_quantile sets threshold u directly,
    bypasses bootstrap calculation (p_ad is NaN), and produces valid return levels.

    Args:
        synthetic_daily_discharge: Synthetic discharge series fixture.
    """
    quantile_value: float = 0.90
    expected_threshold: float = float(
        np.quantile(synthetic_daily_discharge, quantile_value)
    )

    model: ReturnPeriodModel = ReturnPeriodModel(
        series=synthetic_daily_discharge,
        return_periods=[2, 5, 10],
        fixed_quantile=quantile_value,
        min_exceed=2,
        fixed_shape=0.0,
    )

    # Threshold must match the specified quantile
    assert model.u == pytest.approx(expected_threshold, rel=1e-5)
    # Bootstrap should be automatically bypassed
    assert np.isnan(model.p_ad)
    # Return levels should be positive and monotonically increasing
    rls: dict[int, float] = model.rl_table.set_index("T_years")["GPD_POT_RL"].to_dict()
    assert rls[2] > 0
    assert rls[2] < rls[5] < rls[10]


def test_return_period_model_fixed_threshold(
    synthetic_daily_discharge: pd.Series,
) -> None:
    """Test ReturnPeriodModel with an explicit fixed absolute threshold.

    Verifies that specifying fixed_threshold sets threshold u to the exact value
    and bypasses bootstrap calculation.

    Args:
        synthetic_daily_discharge: Synthetic discharge series fixture.
    """
    threshold_value: float = 25.0
    model: ReturnPeriodModel = ReturnPeriodModel(
        series=synthetic_daily_discharge,
        return_periods=[2],
        fixed_threshold=threshold_value,
        min_exceed=2,
        fixed_shape=0.0,
    )

    assert model.u == pytest.approx(threshold_value)
    assert np.isnan(model.p_ad)
    rl_2: float = float(model.rl_table.set_index("T_years").loc[2, "GPD_POT_RL"])
    assert rl_2 > threshold_value


def test_return_period_model_validation(synthetic_daily_discharge: pd.Series) -> None:
    """Test validation for fixed_quantile and fixed_threshold parameters.

    Verifies that specifying both parameters or invalid quantiles raises ValueError.

    Args:
        synthetic_daily_discharge: Synthetic discharge series fixture.
    """
    # Cannot specify both
    with pytest.raises(ValueError, match="Cannot specify both"):
        ReturnPeriodModel(
            series=synthetic_daily_discharge,
            fixed_quantile=0.9,
            fixed_threshold=20.0,
        )

    # Quantile must be between 0 and 1
    with pytest.raises(ValueError, match="fixed_quantile must be between 0.0 and 1.0"):
        ReturnPeriodModel(
            series=synthetic_daily_discharge,
            fixed_quantile=-0.1,
        )

    with pytest.raises(ValueError, match="fixed_quantile must be between 0.0 and 1.0"):
        ReturnPeriodModel(
            series=synthetic_daily_discharge,
            fixed_quantile=1.5,
        )


def test_return_period_model_insufficient_exceedances(
    synthetic_daily_discharge: pd.Series,
) -> None:
    """Test that an excessively high fixed threshold raises ValueError for insufficient peaks.

    Args:
        synthetic_daily_discharge: Synthetic discharge series fixture.
    """
    # A threshold higher than the maximum value in the series will produce 0 exceedances
    extreme_threshold: float = float(synthetic_daily_discharge.max()) + 100.0
    with pytest.raises(ValueError, match="fewer than min_exceed"):
        ReturnPeriodModel(
            series=synthetic_daily_discharge,
            fixed_threshold=extreme_threshold,
            min_exceed=2,
        )


def test_fit_gpd_lmoments_minimum_exceedances() -> None:
    """Test minimum exceedance requirements for fit_gpd_lmoments.

    Verifies that when fixed_shape is provided, fewer exceedances (e.g. 2) are accepted,
    while unconstrained fit still requires at least 6.
    """
    few_exceedances: np.ndarray = np.array([2.5, 4.0])

    # With fixed_shape=0.0, only the 1st L-moment is estimated, so 2 exceedances succeed
    sigma, xi = fit_gpd_lmoments(few_exceedances, fixed_shape=0.0)
    assert xi == 0.0
    assert sigma > 0.0

    # Without fixed_shape, unrestricted fit requires at least 6 exceedances
    with pytest.raises(ValueError, match="Too few exceedances for reliable fit"):
        fit_gpd_lmoments(few_exceedances, fixed_shape=None)


def test_bootstrap_pvalue_for_ad_unrestricted() -> None:
    """Test vectorized bootstrap p-value computation for unrestricted GPD fit."""
    n: int = 40
    sigma_hat: float = 10.0
    xi_hat: float = 0.1
    observed_stat: float = 0.5
    nboot: int = 500

    p_value: float = bootstrap_pvalue_for_ad(
        observed_stat=observed_stat,
        n=n,
        sigma_hat=sigma_hat,
        xi_hat=xi_hat,
        nboot=nboot,
        random_seed=42,
    )
    assert 0.0 <= p_value <= 1.0


def test_bootstrap_pvalue_for_ad_fixed_constraints() -> None:
    """Test vectorized bootstrap p-value computation with fixed shape and scale."""
    n: int = 35
    nboot: int = 200

    # Fixed shape = 0.0 (exponential tail)
    p_fixed_shape: float = bootstrap_pvalue_for_ad(
        observed_stat=0.3,
        n=n,
        sigma_hat=8.0,
        xi_hat=0.0,
        nboot=nboot,
        fixed_shape=0.0,
        random_seed=42,
    )
    assert 0.0 <= p_fixed_shape <= 1.0

    # Fixed scale
    p_fixed_scale: float = bootstrap_pvalue_for_ad(
        observed_stat=0.3,
        n=n,
        sigma_hat=5.0,
        xi_hat=0.1,
        nboot=nboot,
        fixed_scale=5.0,
        random_seed=42,
    )
    assert 0.0 <= p_fixed_scale <= 1.0

    # nboot = 0 returns NaN
    p_zero: float = bootstrap_pvalue_for_ad(
        observed_stat=0.3,
        n=n,
        sigma_hat=5.0,
        xi_hat=0.1,
        nboot=0,
    )
    assert np.isnan(p_zero)


def test_return_period_model_automated_search_mode(
    synthetic_daily_discharge: pd.Series,
) -> None:
    """Test ReturnPeriodModel automated candidate threshold search with bootstrap p-values.

    Verifies that automated threshold search runs vectorized bootstrap simulations, selects a valid
    threshold with p_ad > p_value_threshold, and calculates valid return levels.

    Args:
        synthetic_daily_discharge: Synthetic discharge series fixture.
    """
    model: ReturnPeriodModel = ReturnPeriodModel(
        series=synthetic_daily_discharge,
        return_periods=[2, 5, 10],
        min_exceed=2,
        nboot=200,
        quantile_start=0.85,
        quantile_end=0.95,
        quantile_step=0.05,
        fixed_shape=0.0,
        p_value_threshold=0.05,
    )

    assert not np.isnan(model.p_ad)
    assert model.p_ad > 0.0
    assert model.u > 0.0
    assert len(model.candidates_df) > 0


def test_negative_nboot_raises_error(synthetic_daily_discharge: pd.Series) -> None:
    """Test that negative nboot raises ValueError in bootstrap and ReturnPeriodModel."""
    with pytest.raises(ValueError, match="nboot must be non-negative"):
        bootstrap_pvalue_for_ad(
            observed_stat=0.5,
            n=30,
            sigma_hat=10.0,
            xi_hat=0.1,
            nboot=-1,
        )

    with pytest.raises(ValueError, match="nboot must be non-negative"):
        ReturnPeriodModel(
            series=synthetic_daily_discharge,
            min_exceed=2,
            nboot=-5,
        )
