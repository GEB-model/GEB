"""Functions for fitting GPD-POT models and assigning return periods to rivers."""

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.signal import find_peaks
from scipy.stats import genpareto

from geb.geb_types import ArrayFloat64


def fit_gpd_lmoments(
    exceedances: np.ndarray,
    fixed_shape: float | None = None,
    fixed_scale: float | None = None,
) -> tuple[float, float]:
    r"""Fit GPD to positive exceedances using L-Moments.

    Estimates GPD parameters using the method of L-Moments (Hosking & Wallis, 1997).
    The location parameter is fixed at 0 (for exceedances).

    The L-moments ($l_1, l_2$) are robust summary statistics based on linear
    combinations of order statistics. For a GPD with shape $\xi$ and scale $\sigma$:
    $$ l_1 = \frac{\sigma}{1 - \xi} $$
    $$ l_2 = \frac{\sigma}{(1 - \xi)(2 - \xi)} $$

    From these relations:
    - Standard fit uses both $l_1$ and $l_2$ to solve for $\xi$ and $\sigma$.
    - Fixed shape uses only $l_1$ (mean) to estimate $\sigma$.
    - Fixed scale uses only $l_1$ (mean) to estimate $\xi$.

    Args:
        exceedances: Positive exceedance values above threshold (dimensionless).
        fixed_shape: Value to fix the shape parameter (xi) (dimensionless). If None, it is fitted.
            Set to 0 to force a Gumbel (Exponential) distribution.
        fixed_scale: Value to fix the scale parameter (sigma) (dimensionless). If None, it is fitted.

    Returns:
        Tuple of (sigma, xi) where sigma is the scale parameter and xi is
        the shape parameter of the fitted GPD.

    Raises:
        ValueError: If fewer than required exceedances provided (6 for unrestricted fit, 1 when shape or scale is fixed) or parameters invalid.
    """
    if fixed_shape is not None and fixed_scale is not None:
        raise ValueError("Cannot fix both shape and scale parameters simultaneously.")

    y = np.asarray(exceedances, dtype=float)
    n: int = len(y)
    # When shape or scale is fixed, only the 1st L-moment (mean) is estimated, requiring at least 1 exceedance.
    min_required: int = 1 if (fixed_shape is not None or fixed_scale is not None) else 6
    if n < min_required:
        raise ValueError(
            f"Too few exceedances for reliable fit (found {n}, requires at least {min_required})."
        )

    l1: float = float(np.mean(y))
    xi: float
    sigma: float
    if fixed_shape is not None:
        xi = fixed_shape
        # Use l1 to estimate sigma: sigma = l1 * (1 - xi)
        sigma = l1 * (1.0 - xi)
    elif fixed_scale is not None:
        sigma = fixed_scale
        # Use l1 to estimate xi: l1 = sigma / (1 - xi) -> 1 - xi = sigma / l1 -> xi = 1 - sigma / l1
        if abs(l1) < 1e-12:
            raise ValueError("Mean exceedance is 0, cannot fit with fixed scale.")
        xi = 1.0 - sigma / l1
    else:
        y_sorted: np.ndarray = np.sort(y)
        j: np.ndarray = np.arange(1, n + 1, dtype=float)
        b1: float = float(np.sum((j - 1.0) / (n - 1.0) * y_sorted) / n)
        l2: float = 2.0 * b1 - l1
        if abs(l2) < 1e-12:
            # If l2 is extremely small, the data may be nearly constant.
            raise ValueError("L2 moment is (almost) 0, cannot fit GPD.")

        xi = 2.0 - l1 / l2
        sigma = l1 * (1.0 - xi)

    return float(sigma), float(xi)


def gpd_cdf(y: np.ndarray | float, sigma: float, xi: float) -> ArrayFloat64:
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
        return np.maximum(0.0, 1.0 - np.exp(-np.maximum(y, 0.0) / sigma))

    # For xi < 0, the distribution has an upper bound at -sigma/xi.
    # We use np.maximum(..., 0) to handle values above the upper bound (result 1.0).
    val = 1.0 + xi * y / sigma
    cdf = 1.0 - np.power(np.maximum(val, 0.0), -1.0 / xi)
    return np.clip(cdf, 0.0, 1.0)


def right_tail_ad_from_uniforms(u: ArrayFloat64) -> float:
    """Right-tail weighted Anderson-Darling statistic for uniform data.

    Computes the right-tail weighted AD statistic:
    A_R^2 = -n - (1/n) * sum_{i=1}^n (2i-1) * ln(1 - u_{n+1-i})
    where u_sorted are uniforms in ascending order.
    This emphasizes misfit in the right tail (large exceedances).

    Notes:
        The uniforms are clipped to (0, 1) to avoid numerical issues with log.

    Args:
        u: Uniform random variables (dimensionless).

    Returns:
        Right-tail Anderson-Darling statistic (dimensionless).
    """
    # np sort, sorts the array in ascending order
    u_sorted = np.sort(u)
    n = u_sorted.size
    if n < 1:
        return np.nan
    # Avoid log(0)
    u_safe = np.clip(u_sorted, 1e-15, 1.0 - 1e-15)
    i = np.arange(1, n + 1)
    s_tail = np.sum((2 * i - 1) * np.log(1.0 - u_safe[::-1]))
    a_r2 = -n - s_tail / n
    return float(a_r2)


def bootstrap_pvalue_for_ad(
    observed_stat: float,
    n: int,
    sigma_hat: float,
    xi_hat: float,
    nboot: int = 1000,
    random_seed: int = 42,
    fixed_shape: float | None = None,
    fixed_scale: float | None = None,
) -> float:
    """Parametric bootstrap p-value for right-tail Anderson-Darling statistic.

    Simulates n exceedances from GPD(sigma_hat, xi_hat) nboot times in a fully
    vectorized computation. To correctly test the composite hypothesis (where
    parameters are estimated), each bootstrap sample is re-fitted using L-moments
    before calculating the right-tail AD statistic.

    Args:
        observed_stat: Observed Anderson-Darling statistic (dimensionless).
        n: Number of exceedances to simulate (dimensionless).
        sigma_hat: Fitted GPD scale parameter used as ground truth for simulation (dimensionless).
        xi_hat: Fitted GPD shape parameter used as ground truth for simulation (dimensionless).
        nboot: Number of bootstrap samples (dimensionless).
        random_seed: Random seed for reproducibility (dimensionless).
        fixed_shape: Fixed shape parameter if used during original fit (dimensionless).
        fixed_scale: Fixed scale parameter if used during original fit (dimensionless).

    Returns:
        Bootstrap p-value as proportion of simulated stats >= observed (dimensionless).

    Raises:
        ValueError: If nboot is negative.
    """
    if nboot < 0:
        raise ValueError(f"nboot must be non-negative, got {nboot}.")
    if nboot == 0:
        return float(np.nan)

    # Simulate n exceedances nboot times from the fitted GPD
    y: np.ndarray = genpareto.rvs(
        c=xi_hat, scale=sigma_hat, size=(nboot, n), random_state=random_seed
    )
    # Sort order statistics in-place along axis 1 for each bootstrap realization
    y.sort(axis=1)

    # Refit GPD parameters across all bootstrap realizations using vectorized L-moments
    l1: np.ndarray = np.mean(y, axis=1)
    xi: np.ndarray
    sigma: np.ndarray

    if fixed_shape is not None and fixed_scale is not None:
        raise ValueError("Cannot fix both shape and scale parameters simultaneously.")
    if fixed_shape is not None:
        xi = np.full(nboot, fixed_shape, dtype=float)
        sigma = l1 * (1.0 - fixed_shape)
    elif fixed_scale is not None:
        sigma = np.full(nboot, fixed_scale, dtype=float)
        xi = 1.0 - fixed_scale / l1
    else:
        # Probability-weighted moment b1 = (1/n) * sum_{j=1}^n ((j - 1) / (n - 1)) * y_(j)
        j: np.ndarray = np.arange(1, n + 1, dtype=float)
        weights_b1: np.ndarray = (j - 1.0) / ((n - 1.0) * n)
        b1: np.ndarray = np.dot(y, weights_b1)
        l2: np.ndarray = 2.0 * b1 - l1
        xi = 2.0 - l1 / l2
        sigma = l1 * (1.0 - xi)

    # Vectorized GPD CDF computation
    # Guard against division by zero in -1.0 / xi when xi is near zero (exponential tail)
    exp_mask: np.ndarray = np.abs(xi) < 1e-12
    xi_safe: np.ndarray = np.where(exp_mask, 1.0, xi)
    val: np.ndarray = 1.0 + (xi[:, None] * y) / sigma[:, None]
    cdf: np.ndarray = 1.0 - np.power(np.maximum(val, 0.0), -1.0 / xi_safe[:, None])

    if np.any(exp_mask):
        cdf_exp: np.ndarray = 1.0 - np.exp(-np.maximum(y, 0.0) / sigma[:, None])
        cdf = np.where(exp_mask[:, None], cdf_exp, cdf)

    u_sim: np.ndarray = np.clip(cdf, 0.0, 1.0)
    u_safe: np.ndarray = np.clip(u_sim, 1e-15, 1.0 - 1e-15)

    # Right-tail weighted Anderson-Darling statistic
    # Because y is sorted and GPD CDF is monotonic, u_safe is already sorted.
    # Reversing columns gives u_{n+1-i} directly for the right-tail weighting.
    weights_ad: np.ndarray = 2.0 * np.arange(1, n + 1, dtype=float) - 1.0
    log_terms: np.ndarray = np.log(1.0 - u_safe[:, ::-1])
    s_tail: np.ndarray = np.dot(log_terms, weights_ad)
    ad_stats: np.ndarray = -n - s_tail / n

    return float(np.mean(ad_stats >= observed_stat))


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


class ReturnPeriodModel:
    """Class for Generalized Pareto Distribution Peaks-Over-Threshold (GPD-POT) analysis.

    Fits a GPD model to extremes and provides methods for return level estimation
    and diagnostic plotting.
    """

    def __init__(
        self,
        series: pd.Series,
        return_periods: np.ndarray | list[int | float] | None = None,
        quantile_start: float = 0.80,
        quantile_end: float = 0.99,
        quantile_step: float = 0.01,
        min_exceed: int = 20,
        nboot: int = 2000,
        random_seed: int = 42,
        fixed_shape: float | None = None,
        fixed_scale: float | None = None,
        fixed_quantile: float | None = None,
        fixed_threshold: float | None = None,
        p_value_threshold: float = 0.10,
        selection_strategy: str = "first_significant",
    ) -> None:
        """Initialize and fit the Generalized Pareto Distribution Peaks-Over-Threshold (GPD-POT) model.

        Fits a GPD to declustered peak excesses above a threshold and calculates return levels
        for the requested return periods using L-moments. The model supports two operational modes
        for threshold selection:

        Fixed threshold mode:
            If `fixed_quantile` (e.g. 0.90 for the 90th percentile) or `fixed_threshold` (absolute
            value) is provided, that threshold is used directly without scanning. Peaks above this
            threshold are declustered (minimum 7-day separation), and the GPD parameters are fitted
            analytically via L-moments. All threshold scanning parameters (`quantile_start`,
            `quantile_end`, `quantile_step`) and bootstrap goodness-of-fit testing (`nboot`,
            `p_value_threshold`, `selection_strategy`) are bypassed. The bootstrap p-value (`p_ad`)
            is set to NaN.

        Automated threshold search mode:
            If neither `fixed_quantile` nor `fixed_threshold` is provided, candidate thresholds are
            evaluated across a quantile grid from `quantile_end` down to `quantile_start` at intervals
            of `quantile_step`. For each candidate with at least `min_exceed` peaks, GPD parameters are
            fitted and a right-tail weighted Anderson-Darling statistic is evaluated using a parametric
            bootstrap of size `nboot` to obtain a p-value. Under `selection_strategy='first_significant'`,
            scanning stops at the first (highest) threshold where the p-value exceeds `p_value_threshold`,
            falling back to the highest p-value if none qualify. Under `selection_strategy='best_fit'`,
            all candidates are evaluated and the one with the maximum p-value is chosen. Setting `nboot=0`
            bypasses bootstrap simulations in search mode.

        Distribution constraints:
            The GPD shape parameter can be fixed via `fixed_shape`. Setting `fixed_shape=0.0` forces an
            exponential tail (equivalent to a Poisson-Gumbel extreme value model), reducing estimation
            to only the scale parameter (mean excess). This provides robust return level estimates for
            short records (e.g. 1-2 years of simulation) or moderate return periods. Alternatively,
            `fixed_scale` fixes the scale parameter. Both constraints are mutually exclusive.

        Args:
            series: Time series data with regular DatetimeIndex (typically daily discharge in m3/s).
            return_periods: Return periods in years for return level calculation.
                Defaults to [2, 5, 10, 25, 50, 100, 200, 250, 500, 1000, 10000].
            quantile_start: Lower bound quantile for threshold scan in automated mode (dimensionless, 0.0 to 1.0).
            quantile_end: Upper bound quantile for threshold scan in automated mode (dimensionless, 0.0 to 1.0).
            quantile_step: Step size between candidate quantiles in automated mode (dimensionless).
            min_exceed: Minimum number of declustered peaks required above threshold (dimensionless).
            nboot: Number of bootstrap iterations for Anderson-Darling p-value calculation in automated mode.
                Set to 0 to bypass bootstrap testing. Ignored if a fixed threshold or quantile is used.
            random_seed: Random seed for bootstrap reproducibility.
            fixed_shape: Value to fix the shape parameter xi (dimensionless). Setting 0.0 forces an
                Exponential tail. Mutually exclusive with fixed_scale.
            fixed_scale: Value to fix the scale parameter sigma (series units, e.g. m3/s).
                Mutually exclusive with fixed_shape.
            fixed_quantile: Fixed quantile threshold (0.0 to 1.0) to use directly (dimensionless).
                Mutually exclusive with fixed_threshold.
            fixed_threshold: Fixed absolute threshold to use directly (series units, e.g. m3/s).
                Mutually exclusive with fixed_quantile.
            p_value_threshold: Anderson-Darling p-value threshold for early stopping in automated mode.
            selection_strategy: Strategy for selecting the best threshold in automated mode,
                either 'first_significant' or 'best_fit'.

        Raises:
            TypeError: If series index is not a DatetimeIndex.
            ValueError: If nboot is negative, series is non-monotonic, irregular,
                has fewer non-NaN values than min_exceed, both fixed_quantile and fixed_threshold
                are specified, fixed_quantile is outside [0, 1], both fixed_shape and fixed_scale
                are specified, or no valid thresholds satisfy min_exceed.
        """
        if return_periods is None:
            return_periods = np.array(
                [2, 5, 10, 25, 50, 100, 200, 250, 500, 1000, 10000], np.float32
            )
        self.return_periods = np.asarray(return_periods, np.float32)

        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("Series must have a DatetimeIndex.")

        if not series.index.is_monotonic_increasing:
            raise ValueError(
                "Series must have regular time steps and a monotonic increasing DateTimeIndex."
            )

        if series.index.freq is None:
            raise ValueError(
                "Series index must have a regular frequency (e.g. hourly, daily)."
            )

        if fixed_quantile is not None and fixed_threshold is not None:
            raise ValueError(
                "Cannot specify both fixed_quantile and fixed_threshold simultaneously."
            )
        if fixed_quantile is not None and not (0.0 <= fixed_quantile <= 1.0):
            raise ValueError(
                f"fixed_quantile must be between 0.0 and 1.0, got {fixed_quantile}."
            )
        if nboot < 0:
            raise ValueError(f"nboot must be non-negative, got {nboot}.")

        self.nanmask = series.isnull()
        self.n_nan = self.nanmask.sum()
        self.n_non_nan = (~self.nanmask).sum()
        if self.n_non_nan < min_exceed:
            raise ValueError(
                f"Series must have at least {min_exceed} non-NaN values for fitting. Found only {self.n_non_nan}."
            )

        series: pd.Series = series.fillna(0)

        n_data_points_per_week: int = math.ceil(
            pd.Timedelta("7D").value / series.index.freq.nanos  # ty:ignore[unresolved-attribute]
        )

        self.series = series
        # Resample to daily maxima to ensure independence of observations (de-clustering)
        self.years_non_nan = (
            (self.n_non_nan * self.series.index.freq.nanos) / pd.Timedelta(days=1).value  # ty:ignore[unresolved-attribute]
        ) / 365.2425

        # Handle fixed threshold vs candidate threshold search
        if fixed_threshold is not None or fixed_quantile is not None:
            if fixed_threshold is not None:
                u: float = float(fixed_threshold)
            else:
                assert fixed_quantile is not None
                u = float(np.quantile(self.series[~self.nanmask], fixed_quantile))

            # Find independent peaks above the fixed threshold
            _, properties_fixed = find_peaks(
                self.series.values, height=u, distance=n_data_points_per_week
            )
            self.all_peaks = properties_fixed["peak_heights"]
            n_exc: int = self.all_peaks.size

            if n_exc < min_exceed:
                raise ValueError(
                    f"Fixed threshold {u:.2f} has only {n_exc} exceedances, "
                    f"fewer than min_exceed={min_exceed}."
                )

            y: np.ndarray = self.all_peaks - u
            sigma, xi = fit_gpd_lmoments(
                y, fixed_shape=fixed_shape, fixed_scale=fixed_scale
            )
            u_vals = gpd_cdf(y, sigma, xi)
            A_R2: float = right_tail_ad_from_uniforms(u_vals)

            best_candidate: dict[str, Any] = {
                "u": u,
                "sigma": sigma,
                "xi": xi,
                "n_exc": n_exc,
                "p_ad": float(np.nan),
                "A_R2": A_R2,
            }
            self.candidates_df = pd.DataFrame([best_candidate])
        else:
            # Candidate threshold search
            # Start from upper quantile, so that we start evaluation with the most extreme thresholds
            q_grid: np.ndarray = np.arange(
                quantile_end, quantile_start - 1e-9, -quantile_step
            )
            u_candidates: np.ndarray = np.quantile(self.series[~self.nanmask], q_grid)

            # Find all independent peaks above the lowest candidate threshold once
            u_min: float = float(u_candidates.min())
            _, properties_all = find_peaks(
                self.series.values, height=u_min, distance=n_data_points_per_week
            )
            self.all_peaks = properties_all["peak_heights"]

            best_candidate_search: dict[str, Any] | None = None
            candidates_list: list[dict[str, Any]] = []

            for u_cand in u_candidates:
                peaks_u = self.all_peaks[self.all_peaks > u_cand]
                n_exc_cand: int = peaks_u.size

                if n_exc_cand < min_exceed:
                    if u_cand == u_candidates[-1]:
                        raise ValueError(
                            f"No valid thresholds found with at least {min_exceed} exceedances. "
                            f"Lowest candidate threshold {u_cand:.2f} has only {n_exc_cand} exceedances."
                        )
                    continue

                y_cand: np.ndarray = peaks_u - u_cand
                sigma_cand, xi_cand = fit_gpd_lmoments(
                    y_cand, fixed_shape=fixed_shape, fixed_scale=fixed_scale
                )
                u_vals_cand = gpd_cdf(y_cand, sigma_cand, xi_cand)
                A_R2_cand = right_tail_ad_from_uniforms(u_vals_cand)

                if nboot == 0:
                    p_ad_cand: float = float(np.nan)
                else:
                    p_ad_cand = bootstrap_pvalue_for_ad(
                        A_R2_cand,
                        n_exc_cand,
                        sigma_cand,
                        xi_cand,
                        nboot,
                        random_seed,
                        fixed_shape=fixed_shape,
                        fixed_scale=fixed_scale,
                    )

                current_candidate: dict[str, Any] = {
                    "u": u_cand,
                    "sigma": sigma_cand,
                    "xi": xi_cand,
                    "n_exc": n_exc_cand,
                    "p_ad": p_ad_cand,
                    "A_R2": A_R2_cand,
                }
                candidates_list.append(current_candidate)

                if (
                    selection_strategy == "first_significant"
                    and p_ad_cand > p_value_threshold
                ):
                    best_candidate_search = current_candidate
                    break

            self.candidates_df = pd.DataFrame(candidates_list)

            if best_candidate_search is None:
                if not candidates_list:
                    raise ValueError("No valid thresholds found for GPD-POT fitting.")
                best_candidate_search = max(
                    candidates_list,
                    key=lambda x: x["p_ad"] if np.isfinite(x["p_ad"]) else 0.0,
                )

            best_candidate = best_candidate_search

        self.u = best_candidate["u"]
        self.sigma = best_candidate["sigma"]
        self.xi = best_candidate["xi"]
        self.n_exc = best_candidate["n_exc"]
        self.p_ad = best_candidate["p_ad"]
        self.lambda_per_year = self.n_exc / self.years_non_nan

        self.water_level_for_return_periods = gpd_return_level(
            self.u, self.sigma, self.xi, self.lambda_per_year, self.return_periods
        )

        self.rl_table = pd.DataFrame(
            {
                "T_years": self.return_periods.astype(int),
                "GPD_POT_RL": self.water_level_for_return_periods,
            }
        )

        # Check return levels for each return period, with error handling for non-finite or extreme values
        MAX_Q = 400_000
        for _, row in self.rl_table.iterrows():
            T = row["T_years"]
            RL = row["GPD_POT_RL"]
            # If infinite or NaN, raise an error to flag potential issues with fit or threshold selection
            if not np.isfinite(RL) or np.isnan(RL):
                raise ValueError(
                    f"Computed return level for T={T} years is non-finite or nan ({RL}). "
                    "This likely indicates an issue with the GPD fit or threshold selection. "
                    "Consider reviewing the diagnostics for this series."
                )

            # Return levels above MAX_Q are likely unreliable and can cause issues in downstream analysis, so raise an error
            if RL > MAX_Q:
                raise ValueError(
                    f"Computed return level for T={T} years ({RL:.3g}) exceeds maximum cap {MAX_Q}. "
                    "This likely indicates an unreliable fit or extreme extrapolation. "
                    "Consider reviewing the GPD fit diagnostics for this series."
                )

    def plot_fit(
        self,
        ax: plt.Axes | None = None,
        label_prefix: str = "Q",
        color: str = "C0",
    ) -> plt.Axes:
        """Plot GPD-POT return-period curve for this series.

        Includes the fitted curve, POT points, annual maxima, and threshold u.

        Args:
            ax: Matplotlib axes to plot on. If None, creates a new figure.
            label_prefix: Prefix for labels in legend.
            color: Color for plotting.

        Returns:
            The matplotlib axes with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot fitted curve
        ax.plot(
            self.rl_table["T_years"],
            self.rl_table["GPD_POT_RL"],
            label=f"{label_prefix} (fit)",
            linewidth=1.5,
            marker="o",
            color=color,
        )

        # Plot de-clustered peaks
        # Using the same set of independent peaks found in __init__ for consistency
        all_peaks_sorted = np.sort(self.all_peaks)[::-1]
        n_all_peaks = len(all_peaks_sorted)
        ranks_all = np.arange(1, n_all_peaks + 1)
        T_all = (self.years_non_nan + 0.12) / (ranks_all - 0.44)

        mask_pot = all_peaks_sorted > self.u
        mask_ignored = all_peaks_sorted <= self.u

        # Plot POT points
        if np.any(mask_pot):
            ax.scatter(
                T_all[mask_pot],
                all_peaks_sorted[mask_pot],
                label=f"{label_prefix} (POT)",
                color=color,
                alpha=0.6,
                s=20,
                marker="x",
            )

        # Plot Ignored points
        if np.any(mask_ignored):
            ax.scatter(
                T_all[mask_ignored],
                all_peaks_sorted[mask_ignored],
                label=f"{label_prefix} (ignored < u)",
                color=color,
                alpha=0.5,
                s=15,
                marker=".",
            )

        # Plot AM points
        am_series = self.series.resample("YE").max().dropna()
        am_sorted = am_series.sort_values(ascending=False)
        n_am = len(am_sorted)
        if n_am > 0:
            ranks_am = np.arange(1, n_am + 1)
            p_am = (ranks_am - 0.44) / (n_am + 0.12)
            T_am = 1.0 / p_am
            ax.scatter(
                T_am,
                am_sorted.values,
                label=f"{label_prefix} (Annual Maxima)",
                color=color,
                alpha=0.6,
                s=20,
                marker="^",
            )

        # Plot threshold u
        ax.axhline(
            self.u,
            linestyle="--",
            color=color,
            alpha=0.5,
            linewidth=1,
            label=f"{label_prefix} (u)",
        )

        # Add text box
        text_str = (
            f"{label_prefix} Params:\n"
            f"u={self.u:.2f}\n"
            f"σ={self.sigma:.2f}\n"
            f"ξ={self.xi:.2f}\n"
            f"λ={self.lambda_per_year:.2f}"
        )
        # Position boxes side-by-side in bottom right to avoid overlap
        x_pos = 0.55 if label_prefix == "discharge_observations" else 0.78
        ax.text(
            x_pos,
            0.03,
            text_str,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="black", alpha=1.0, edgecolor=color),
            color="white",
        )

        ax.set_xscale("log")
        ax.set_xlabel("Return period [years]")
        ax.set_ylabel("Discharge [m3/s]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left", fontsize="small")

        return ax

    def plot_gof(
        self,
        axes: np.ndarray[tuple[int], Any] | list[Axes] | None = None,
        figsize: tuple[int, int] = (15, 5),
    ) -> np.ndarray:
        """Plot QQ, PP, and Density diagnostic plots for the chosen threshold.

        Args:
            axes: Array of 3 matplotlib axes to plot on. If None, creates a new figure.
            figsize: Figure size as (width, height) in inches.

        Returns:
            The array of 3 matplotlib axes.
        """
        # Get de-clustered exceedances at chosen threshold
        peaks_u = self.all_peaks[self.all_peaks > self.u]
        y = peaks_u - self.u
        y_sorted = np.sort(y)
        n = len(y)

        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            fig.suptitle(
                f"Goodness-of-Fit Diagnostics (u={self.u:.2f}, σ={self.sigma:.2f}, ξ={self.xi:.3f})",
                fontsize=14,
                fontweight="bold",
            )
        else:
            axes = np.asarray(axes).flatten()

        # QQ Plot (Quantile-Quantile)
        ax = axes[0]
        # Empirical quantiles
        empirical_probs = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
        empirical_quantiles = y_sorted

        # Theoretical quantiles from fitted GPD
        theoretical_quantiles = genpareto.ppf(
            empirical_probs, c=self.xi, scale=self.sigma
        )

        ax.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.6, s=20)

        # 45-degree reference line
        min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
        max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
        ax.plot(
            [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="1:1 line"
        )

        ax.set_xlabel("Theoretical Quantiles (GPD)")
        ax.set_ylabel("Empirical Quantiles")
        ax.set_title("Quantile-Quantile (QQ) Plot")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # PP Plot (Probability-Probability)
        ax = axes[1]
        # Empirical CDF
        empirical_cdf = empirical_probs

        # Theoretical CDF from fitted GPD
        theoretical_cdf = gpd_cdf(y_sorted, self.sigma, self.xi)

        ax.scatter(theoretical_cdf, empirical_cdf, alpha=0.6, s=20, color="green")
        ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="1:1 line")

        ax.set_xlabel("Theoretical CDF (GPD)")
        ax.set_ylabel("Empirical CDF")
        ax.set_title("Probability-Probability (PP) Plot")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Density Plot (Histogram vs fitted density)
        ax = axes[2]
        # Histogram
        ax.hist(
            y,
            bins=30,
            density=True,
            alpha=0.6,
            color="skyblue",
            edgecolor="black",
            label="Empirical",
        )

        # Fitted GPD density
        y_range = np.linspace(0, y.max(), 200)
        fitted_density = genpareto.pdf(y_range, c=self.xi, scale=self.sigma)
        ax.plot(
            y_range,
            fitted_density,
            "r-",
            linewidth=2,
            label=f"GPD(σ={self.sigma:.2f}, ξ={self.xi:.3f})",
        )

        ax.set_xlabel("Exceedance (y = x - u)")
        ax.set_ylabel("Density")
        ax.set_title("Density Plot")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        return axes

    def plot_selection_diagnostics(
        self,
        axes: np.ndarray[tuple[int], Any] | list[Axes] | None = None,
        figsize: tuple[int, int] = (12, 10),
    ) -> np.ndarray:
        """Plot diagnostics across all candidate thresholds.

        Shows how parameters and goodness-of-fit vary with the choice of threshold u.

        Args:
            axes: Array of 4 matplotlib axes to plot on. If None, creates a new figure.
            figsize: Figure size as (width, height) in inches.

        Returns:
            The array of 4 matplotlib axes.

        Raises:
            ValueError: If no candidates found to plot diagnostics.
        """
        if self.candidates_df.empty:
            raise ValueError("No candidates found to plot diagnostics.")

        diag = self.candidates_df.sort_values("u")

        if axes is None:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle(
                "GPD-POT Threshold Selection Diagnostics",
                fontsize=14,
                fontweight="bold",
            )
        else:
            axes = np.asarray(axes).flatten()

        # AD p-value vs threshold
        ax = axes[0]
        ax.plot(diag["u"], diag["p_ad"], "b.-", linewidth=1.5, markersize=3)
        ax.axvline(
            self.u,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Chosen u={self.u:.2f}",
        )
        ax.axhline(
            0.05, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="α=0.05"
        )
        ax.set_xlabel("Threshold (u)")
        ax.set_ylabel("AD p-value")
        p_val_label: str = (
            f"chosen p={self.p_ad:.3f}"
            if np.isfinite(self.p_ad)
            else "bootstrap skipped"
        )
        ax.set_title(f"Anderson-Darling p-value ({p_val_label})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Shape parameter (xi) vs threshold
        ax = axes[1]
        ax.plot(diag["u"], diag["xi"], "g.-", linewidth=1.5, markersize=3)
        ax.axvline(self.u, color="red", linestyle="--", linewidth=2)
        ax.axhline(self.xi, color="red", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Threshold (u)")
        ax.set_ylabel("Shape parameter (ξ)")
        ax.set_title(f"Shape Parameter stability (chosen ξ={self.xi:.3f})")
        ax.grid(True, alpha=0.3)

        # Scale parameter (sigma) vs threshold
        ax = axes[2]
        ax.plot(diag["u"], diag["sigma"], "m.-", linewidth=1.5, markersize=3)
        ax.axvline(self.u, color="red", linestyle="--", linewidth=2)
        ax.axhline(self.sigma, color="red", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Threshold (u)")
        ax.set_ylabel("Scale parameter (σ)")
        ax.set_title(f"Scale Parameter stability (chosen σ={self.sigma:.2f})")
        ax.grid(True, alpha=0.3)

        # Number of exceedances vs threshold
        ax = axes[3]
        ax.plot(diag["u"], diag["n_exc"], "r.-", linewidth=1.5, markersize=3)
        ax.axvline(self.u, color="red", linestyle="--", linewidth=2)
        ax.axhline(self.n_exc, color="red", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Threshold (u)")
        ax.set_ylabel("Number of Exceedances")
        ax.set_title(f"Number of exceedances (chosen n={self.n_exc})")
        ax.grid(True, alpha=0.3)

        return axes

    def plot_threshold_stability(
        self,
        axes: np.ndarray[tuple[int], Any] | list[Axes] | None = None,
        return_periods: list[int] | None = None,
        figsize: tuple[int, int] = (12, 8),
    ) -> np.ndarray:
        """Plot return level stability across different thresholds.

        Args:
            axes: Array of 4 matplotlib axes to plot on. If None, creates a new figure.
            return_periods: List of return periods to plot. If None, uses defaults [10, 50, 100, 500].
            figsize: Figure size as (width, height) in inches.

        Returns:
            The array of 4 matplotlib axes.

        Raises:
            ValueError: If no candidates found to plot diagnostics.
        """
        if self.candidates_df.empty:
            raise ValueError("No candidates found to plot diagnostics.")

        if return_periods is None:
            return_periods = [10, 50, 100, 500]

        diag = self.candidates_df.sort_values("u").copy()

        # Calculate return levels for each threshold in the candidates
        for T in return_periods:
            diag[f"RL_{T}"] = [
                gpd_return_level(
                    row["u"],
                    row["sigma"],
                    row["xi"],
                    row["n_exc"] / self.years_non_nan,
                    float(T),
                )
                for _, row in diag.iterrows()
            ]

        if axes is None:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle(
                "Threshold Stability: Return Levels vs Threshold",
                fontsize=14,
                fontweight="bold",
            )
        else:
            axes = np.asarray(axes).flatten()

        colors = ["blue", "green", "red", "purple"]

        for idx, (T, color) in enumerate(zip(return_periods, colors)):
            ax = axes[idx]
            ax.plot(
                diag["u"],
                diag[f"RL_{T}"],
                ".-",
                color=color,
                linewidth=1.5,
                markersize=4,
            )
            ax.axvline(
                self.u,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Chosen u={self.u:.2f}",
            )

            # Add chosen return level for this T as horizontal line if it's in our rl_table
            if T in self.rl_table["T_years"].values:
                chosen_rl = self.rl_table.loc[
                    self.rl_table["T_years"] == T, "GPD_POT_RL"
                ].values[0]
                ax.axhline(
                    chosen_rl, color="red", linestyle=":", linewidth=1, alpha=0.5
                )

            ax.set_xlabel("Threshold (u)")
            ax.set_ylabel(f"{T}-year RL")
            ax.set_title(f"Return Period: {T} years")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        return axes

    def plot_diagnostics(self, figsize: tuple[int, int] = (20, 15)) -> plt.Figure:
        """Create a main diagnostic figure combining all fit, GOF, and stability plots.

        Returns:
            Matplotlib Figure with 13 panels.
        """
        fig = plt.figure(figsize=figsize)
        # 4 rows, 4 columns grid
        gs = fig.add_gridspec(4, 4)

        # 1. Main Fit (top-left, 2x2 spans)
        ax_fit = fig.add_subplot(gs[0:2, 0:2])
        self.plot_fit(ax=ax_fit)
        ax_fit.set_title("Return Level Fit", fontweight="bold")

        # 2-4. GOF Plots (top-right area)
        ax_gof = [
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[0, 3]),
            fig.add_subplot(gs[1, 2]),
        ]
        self.plot_gof(axes=ax_gof)

        # 5-8. Selection Diagnostics (middle rows)
        ax_sel = [
            fig.add_subplot(gs[1, 3]),
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
            fig.add_subplot(gs[2, 2]),
        ]
        self.plot_selection_diagnostics(axes=ax_sel)

        # 9-12. Threshold Stability (bottom row)
        ax_stab = [
            fig.add_subplot(gs[2, 3]),
            fig.add_subplot(gs[3, 0]),
            fig.add_subplot(gs[3, 1]),
            fig.add_subplot(gs[3, 2]),
        ]
        self.plot_threshold_stability(axes=ax_stab)

        fig.suptitle(
            f"Main Diagnostic Plot: {self.n_exc} exceedances above u={self.u:.2f}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig
