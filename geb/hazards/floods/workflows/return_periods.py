"""Functions for fitting GPD-POT models and assigning return periods to rivers."""

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import genpareto, kstest
from tqdm import tqdm


def fit_gpd_mle(
    exceedances: np.ndarray,
    fixed_shape: float | None = None,
    fixed_scale: float | None = None,
) -> tuple[float, float]:
    """Fit GPD to positive exceedances using Maximum Likelihood Estimation.

    Fits a Generalized Pareto Distribution to exceedances (y = x - u) using
    scipy's genpareto.fit.

    The Generalized Pareto Distribution is defined by three parameters:
        1. **Location** (u or loc): The threshold parameter. For exceedances (y = x - u),
           this parameter is always fixed to 0. This is because we model the *excess*
           over the threshold u. The threshold u itself is determined before fitting
           (e.g., via mean residual life plot or automated threshold selection).
        2. **Scale** (sigma): The scale parameter, describing the spread of the
           distribution.
        3. **Shape** (xi): The shape parameter, describing the tail behavior.
           - xi > 0: Heavy tail (Fréchet-like).
           - xi = 0: Light tail (Gumbel-like). In the GPD context, standard Gumbel
             behaviour corresponds to an Exponential distribution.
           - xi < 0: Bounded tail (Weibull-like).

    Options are provided to fix shape and scale. The location parameter is always fixed
    to 0, consistent with fitting to exceedances.

    To fix the fit to a Gumbel distribution (i.e. Exponential exceedances),
    set `fixed_shape=0`.

    Args:
        exceedances: Positive exceedance values above threshold (dimensionless).
        fixed_shape: Value to fix the shape parameter (xi) (dimensionless). If None, it is fitted.
            Set to 0 to force a Gumbel (Exponential) distribution.
        fixed_scale: Value to fix the scale parameter (sigma) (dimensionless). If None, it is fitted.

    Returns:
        Tuple of (sigma, xi) where sigma is the scale parameter and xi is
        the shape parameter of the fitted GPD.

    Raises:
        ValueError: If fewer than 6 exceedances provided for reliable fit.
    """
    if fixed_shape is not None and fixed_scale is not None:
        raise ValueError("Cannot fix both shape and scale parameters simultaneously.")

    y = np.asarray(exceedances, dtype=float)
    if len(y) < 6:
        raise ValueError("Too few exceedances for reliable fit")

    kwargs = {}
    if fixed_shape is not None:
        kwargs["fc"] = fixed_shape
    if fixed_scale is not None:
        kwargs["fscale"] = fixed_scale

    c_hat, loc_hat, scale_hat = genpareto.fit(y, floc=0.0, **kwargs)
    return float(scale_hat), float(c_hat)  # sigma, xi


def gpd_cdf(y: np.ndarray | float, sigma: float, xi: float) -> np.ndarray | float:
    """GPD CDF for exceedance y>=0 with loc=0.

    Args:
        y: Exceedance values (y = x - u), must be >= 0 (dimensionless)
        sigma: Scale parameter (>0) (dimensionless)
        xi: Shape parameter (can be any real number) (dimensionless)

    Returns:
        CDF values at y (dimensionless)
    """
    y = np.asarray(y, float)
    if abs(xi) < 1e-12:
        return 1 - np.exp(-y / sigma)
    return 1 - (1 + xi * y / sigma) ** (-1.0 / xi)


def right_tail_ad_from_uniforms(u_sorted: np.ndarray) -> float:
    """Right-tail weighted Anderson-Darling statistic for uniform data.

    Computes the right-tail weighted AD statistic:
    A_R^2 = -n - (1/n) * sum_{i=1}^n (2i-1) * ln(1 - u_{n+1-i})
    where u_sorted are uniforms in ascending order.
    This emphasizes misfit in the right tail (large exceedances).

    Args:
        u_sorted: Uniform random variables sorted in ascending order (dimensionless).

    Returns:
        Right-tail Anderson-Darling statistic (dimensionless).
    """
    u = np.asarray(u_sorted, dtype=float)
    n = u.size
    if n < 1:
        return np.nan
    i = np.arange(1, n + 1)
    s_tail = np.sum((2 * i - 1) * np.log(1.0 - u[::-1]))
    a_r2 = -n - s_tail / n
    return float(a_r2)


def bootstrap_pvalue_for_ad(
    observed_stat: float,
    n: int,
    sigma_hat: float,
    xi_hat: float,
    nboot: int = 2000,
    random_seed: int = 123,
) -> float:
    """Parametric bootstrap p-value for right-tail Anderson-Darling statistic.

    Simulates n exceedances from GPD(sigma_hat, xi_hat) nboot times,
    computes AD statistic on transformed uniforms for each simulation,
    and returns the proportion of simulated statistics >= observed statistic.

    Args:
        observed_stat: Observed Anderson-Darling statistic (dimensionless).
        n: Number of exceedances to simulate (dimensionless).
        sigma_hat: Fitted GPD scale parameter (dimensionless).
        xi_hat: Fitted GPD shape parameter (dimensionless).
        nboot: Number of bootstrap samples (dimensionless).
        random_seed: Random seed for reproducibility (dimensionless).

    Returns:
        Bootstrap p-value as proportion of simulated stats >= observed (dimensionless).
    """
    rng = np.random.default_rng(random_seed)
    sim_stats = np.empty(nboot, dtype=float)
    for k in range(nboot):
        ysim = genpareto.rvs(c=xi_hat, scale=sigma_hat, size=n, random_state=rng)
        u_sim = gpd_cdf(ysim, sigma_hat, xi_hat)
        sim_stats[k] = right_tail_ad_from_uniforms(np.sort(u_sim))
    pval = np.mean(sim_stats >= observed_stat)
    return float(pval)


def mean_residual_life(data: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """Calculate mean residual life for threshold values.

    Computes the mean excess over threshold u for each threshold in u_grid.
    This is used for mean residual life plots to assess threshold selection.

    Args:
        data: Array of observed values (dimensionless).
        u_grid: Array of threshold values to evaluate (dimensionless).

    Returns:
        Array of mean residual life values, one for each threshold (dimensionless).
        Returns NaN for thresholds with no exceedances.
    """
    x = np.asarray(data, dtype=float)
    return np.array(
        [(x[x > u] - u).mean() if np.sum(x > u) > 0 else np.nan for u in u_grid]
    )


def gpd_return_level(
    u: float, sigma: float, xi: float, lambda_per_year: float, T: np.ndarray | float
) -> np.ndarray | float:
    """Calculate return levels for given return periods using GPD parameters.

    Computes return levels using the Generalized Pareto Distribution fitted above
    threshold u. For shape parameter xi ≈ 0, uses exponential limit formula.

    Args:
        u: Threshold value (dimensionless).
        sigma: GPD scale parameter (>0) (dimensionless).
        xi: GPD shape parameter (dimensionless).
        lambda_per_year: Average number of exceedances per year (1/year).
        T: Return period(s) in years (years).

    Returns:
        Return level(s) corresponding to the given return period(s) (dimensionless).
    """
    T = np.asarray(T, float)
    LT = lambda_per_year * T
    if abs(xi) < 1e-8:
        return u + sigma * np.log(LT)
    return u + (sigma / xi) * (LT**xi - 1.0)


def gpd_pot_ad_auto(
    series: pd.Series,
    quantile_start: float = 0.80,
    quantile_end: float = 0.99,
    quantile_step: float = 0.01,
    min_exceed: int = 30,
    nboot: int = 2000,
    return_periods: np.ndarray | None = None,
    mrl_grid_q: np.ndarray | None = None,
    mrl_top_fraction: float = 0.75,
    random_seed: int = 123,
    fixed_shape: float | None = None,
    fixed_scale: float | None = None,
) -> dict:
    """Automated GPD-POT analysis with threshold selection using Anderson-Darling test.

    Performs automated Generalized Pareto Distribution (GPD) Peaks-Over-Threshold (POT) 
    analysis. The function iterates through candidate thresholds $u$, calculates the 
    excesses $y = x - u$, and fits the GPD to these excesses. 

    The location parameter is implicitly fixed at 0.0 because the distribution is 
    fitted to the magnitude of the exceedance above the threshold, not the absolute 
    river stage/discharge value.

    Args:
        series: Time series data with DatetimeIndex.
        quantile_start: Starting quantile for threshold scan.
        quantile_end: Ending quantile for threshold scan.
        quantile_step: Step size between quantiles.
        min_exceed: Minimum number of exceedances required for fitting.
        nboot: Number of bootstrap samples for AD p-value calculation.
        return_periods: Array of return periods in years for RL calculation.
        mrl_grid_q: Quantiles for mean residual life plot baseline.
        mrl_top_fraction: Fraction of top quantiles for MRL linear fit.
        random_seed: Random seed for reproducibility.
        fixed_shape: Value to fix the shape parameter ($\xi$). If 0, forces Exponential tail.
        fixed_scale: Value to fix the scale parameter ($\sigma$).

    Returns:
        Dictionary containing daily maxima, diagnostics, chosen parameters, 
        and return level table.

    Raises:
        TypeError: If series does not have a DatetimeIndex.
        ValueError: If no valid thresholds found for GPD-POT fitting.
    """
    if return_periods is None:
        return_periods = np.array(
            [2, 5, 10, 25, 50, 100, 200, 250, 500, 1000, 10000], float
        )

    if mrl_grid_q is None:
        mrl_grid_q = np.linspace(0.70, 0.995, 80)

    s = series.dropna().sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("Series must have a DatetimeIndex.")

    # Resample to daily maxima to ensure independence of observations (de-clustering)
    daily_max = s.resample("D").max().dropna()
    total_days = (daily_max.index.max() - daily_max.index.min()).days + 1
    years = total_days / 365.25

    # Create candidate thresholds $u$ based on quantiles of daily maxima
    q_grid = np.arange(quantile_start, quantile_end + 1e-9, quantile_step)
    u_candidates = np.quantile(daily_max.values, q_grid)

    # ---- MRL baseline ----
    mrl_grid_u = np.quantile(daily_max.values, mrl_grid_q)
    mrl_vals = mean_residual_life(daily_max.values, mrl_grid_u)

    top_idx = int(len(mrl_vals) * mrl_top_fraction)
    if np.sum(~np.isnan(mrl_vals[top_idx:])) >= 3:
        a_lin, b_lin = np.polyfit(mrl_grid_u[top_idx:], mrl_vals[top_idx:], 1)
    else:
        a_lin, b_lin = 0.0, 0.0

    diagnostics = []

    for u in u_candidates:
        exceed = daily_max[daily_max > u]
        n_exc = exceed.size

        if n_exc < min_exceed:
            diagnostics.append(
                (u, np.nan, np.nan, n_exc, np.nan, np.nan, np.nan, np.nan)
            )
            continue

        # Calculate excesses $y$. This shifts the data origin to 0.
        # Consequently, the GPD location parameter $\mu$ is effectively 0.
        y = exceed.values - u

        # Fit GPD to the excesses. Implementation assumes loc=0.
        sigma, xi = fit_gpd_mle(
            y,
            fixed_shape=fixed_shape,
            fixed_scale=fixed_scale,
        )

        u_vals = gpd_cdf(y, sigma, xi)
        A_R2 = right_tail_ad_from_uniforms(np.sort(u_vals))

        p_ad = bootstrap_pvalue_for_ad(A_R2, n_exc, sigma, xi, nboot, random_seed)

        ks_p = np.nan
        if n_exc >= 20:
            # Here, 0.0 is passed as the location parameter for the KS test
            _, ks_p = kstest(y, "genpareto", args=(xi, 0.0, sigma))

        idx = np.argmin(np.abs(mrl_grid_u - u))
        mrl_err = (
            np.nan
            if np.isnan(mrl_vals[idx])
            else abs(mrl_vals[idx] - (a_lin * mrl_grid_u[idx] + b_lin))
        )

        diagnostics.append((u, sigma, xi, n_exc, p_ad, ks_p, mrl_err, A_R2))

    diag_df = (
        pd.DataFrame(
            diagnostics,
            columns=np.array(
                ["u", "sigma", "xi", "n_exc", "p_ad", "ks_p", "mrl_err", "A_R2"]
            ),
        )
        .sort_values("u")
        .reset_index(drop=True)
    )

    diag_df["xi_step"] = diag_df["xi"].diff().abs().bfill()
    valid = diag_df.dropna(subset=["p_ad"])

    if valid.empty:
        raise ValueError("No valid thresholds found for GPD-POT fitting.")

    # Selection based on the best Anderson-Darling fit
    best = valid.loc[valid["p_ad"].idxmax()]

    u_star = float(best["u"])
    sigma_star = float(best["sigma"])
    xi_star = float(best["xi"])
    p_star = float(best["p_ad"])
    n_star = int(best["n_exc"])

    lambda_per_year = n_star / years
    pct = (daily_max < u_star).mean() * 100

    # Calculate return levels. Note that u_star is passed to re-add the 
    # threshold offset to the estimated excesses.
    water_level_for_return_periods = gpd_return_level(
        u_star, sigma_star, xi_star, lambda_per_year, return_periods
    )
    
    water_level_for_return_periods_table = pd.DataFrame(
        {
            "T_years": return_periods.astype(int),
            "GPD_POT_RL": water_level_for_return_periods,
        }
    )

    return {
        "daily_max": daily_max,
        "years": years,
        "diag_df": diag_df,
        "chosen": {
            "u": u_star,
            "sigma": sigma_star,
            "xi": xi_star,
            "p_ad": p_star,
            "n_exc": n_star,
            "lambda_per_year": lambda_per_year,
            "pct": pct,
        },
        "rl_table": water_level_for_return_periods_table,
    }


def assign_return_periods(
    rivers: gpd.GeoDataFrame,
    discharge_dataframe: pd.DataFrame,
    return_periods: list[int | float],
    prefix: str = "Q",
    min_exceed: int = 30,
    nboot: int = 2000,
    fixed_shape: float | None = None,
    fixed_scale: float | None = None,
) -> gpd.GeoDataFrame:
    """Assign return periods to rivers using GPD-POT analysis.

    Based on https://doi.org/10.1002/2016WR019426

    Uses Generalized Pareto Distribution Peaks-Over-Threshold method with:
        - Daily maxima resampling from input time series
        - GPD-POT exceedance model fitting above threshold
        - Anderson-Darling bootstrap p-value threshold selection
        - Return level estimation for specified return periods

    Args:
        rivers: GeoDataFrame with river IDs as index that must match columns in discharge_dataframe.
        discharge_dataframe: Time series DataFrame with datetime index containing discharge data for all rivers (m³/s).
        return_periods: List of return periods in years to compute return levels for.
        prefix: Column prefix for output return level columns. Defaults to "Q".
        min_exceed: Minimum number of exceedances required for reliable GPD fit. Defaults to 30.
        nboot: Number of bootstrap samples for Anderson-Darling threshold selection. Defaults to 2000.
        fixed_shape: Value to fix the shape parameter (xi) (dimensionless). If None, it is fitted.
            Set to 0 to force a Gumbel (Exponential) distribution.
        fixed_scale: Value to fix the scale parameter (sigma) (dimensionless). If None, it is fitted.

    Returns:
        Updated rivers GeoDataFrame with return level columns added (m³/s).

    Raises:
        TypeError: If discharge series does not have a DatetimeIndex.
        ValueError: If return periods contain non-positive values.
    """
    assert isinstance(return_periods, list)
    if not all((isinstance(T, (int, float)) and T > 0) for T in return_periods):
        raise ValueError("All return periods must be positive numbers (years > 0).")
    return_periods_arr = np.asarray(return_periods, float)

    for idx in tqdm(rivers.index, total=len(rivers), desc="GPD-POT Return Periods"):
        discharge = discharge_dataframe[idx].dropna()

        # If all values are zero, assign zeros
        if (discharge < 1e-10).all():
            print(f"Discharge all zero for river {idx}, assigning zeros.")
            for T in return_periods:
                rivers.loc[idx, f"{prefix}_{T}"] = 0.0
            continue

        if not isinstance(discharge.index, pd.DatetimeIndex):
            raise TypeError(
                f"Discharge series for river {idx} must have a DatetimeIndex."
            )

        result = gpd_pot_ad_auto(
            series=discharge,
            return_periods=return_periods_arr,
            min_exceed=min_exceed,
            nboot=nboot,
            fixed_shape=fixed_shape,
            fixed_scale=fixed_scale,
        )
        print(f"GPD-POT analysis completed for river {idx}.")

        # Assign and check return levels for each return period, with error handling for non-finite or extreme values
        MAX_Q = 400_000
        for return_period, return_water_level in (
            result["rl_table"].set_index("T_years")["GPD_POT_RL"].items()
        ):
            # If infinite or NaN, raise an error to flag potential issues with fit or threshold selection
            if not np.isfinite(return_water_level) or np.isnan(return_water_level):
                raise ValueError(
                    f"Computed return level for T={return_period} years for river {idx} is non-finite or nan ({return_water_level}). This likely indicates an issue with the GPD fit or threshold selection. Consider reviewing the diagnostics for this river."
                )

            # Return levels above MAX_Q are likely unreliable and can cause issues in downstream analysis, so raise an error
            if return_water_level > MAX_Q:
                raise ValueError(
                    f"Computed return level for T={return_period} years for river {idx} ({return_water_level:.3g}) exceeds maximum cap {MAX_Q}. This likely indicates an unreliable fit or extreme extrapolation. Consider reviewing the GPD fit diagnostics for this river."
                )

            rivers.loc[idx, f"{prefix}_{return_period}"] = return_water_level

    return rivers
