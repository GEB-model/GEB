"""Calculate discharge skill scores from paired observations and simulations.

This module contains the metric definitions, seasonal grouping, and output columns.
It does not load model files or create plots.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

METEOROLOGICAL_SEASONS: dict[str, tuple[int, ...]] = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}


class DischargeMetrics(NamedTuple):
    """Discharge validation skill scores for a single station and time period."""

    KGE: float = float("nan")
    KGE_modified: float = float("nan")
    KGE_correlation: float = float("nan")
    KGE_bias_ratio: float = float("nan")
    KGE_variability_ratio: float = float("nan")
    NSE: float = float("nan")
    R2: float = float("nan")
    RMSE: float = float("nan")
    RRMSE: float = float("nan")


SEASONAL_KGE_METRICS: tuple[str, ...] = (
    "KGE",
    "KGE_correlation",
    "KGE_bias_ratio",
    "KGE_variability_ratio",
)


DISCHARGE_SCORE_COLUMNS: tuple[str, ...] = tuple(
    f"{metric}_{frequency}"
    for frequency in ("hourly", "daily", "monthly")
    for metric in DischargeMetrics._fields
) + tuple(
    f"{metric}_daily_{season}"
    for season in METEOROLOGICAL_SEASONS
    for metric in SEASONAL_KGE_METRICS
)


# Shared display metadata keeps score labels and limits consistent across figures.
SKILL_SCORE_PLOT_CONFIGS: tuple[dict[str, object], ...] = (
    {
        "col": "KGE",
        "label": "KGE",
        "ylim": (-1.0, 1.0),
        "cmap": "RdYlGn",
        "vmin": -1.0,
        "vmax": 1.0,
    },
    {
        "col": "KGE_correlation",
        "label": "KGE correlation (r)",
        "ylim": (-1.0, 1.0),
        "cmap": "RdYlGn",
        "vmin": -1.0,
        "vmax": 1.0,
    },
    {
        "col": "KGE_bias_ratio",
        "label": "KGE bias ratio (β)",
        "ylim": (0.0, 2.0),
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 2.0,
    },
    {
        "col": "KGE_variability_ratio",
        "label": "KGE variability ratio (α)",
        "ylim": (0.0, 2.0),
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 2.0,
    },
    {
        "col": "NSE",
        "label": "NSE",
        "ylim": (-1.0, 1.0),
        "cmap": "RdYlGn",
        "vmin": -1.0,
        "vmax": 1.0,
    },
    {
        "col": "R2",
        "label": "Pearson r²",
        "ylim": (0.0, 1.0),
        "cmap": "YlGn",
        "vmin": 0.0,
        "vmax": 1.0,
    },
    {
        "col": "RRMSE",
        "label": "RRMSE",
        "ylim": None,
        "cmap": "YlOrRd",
        "vmin": 0.0,
        "vmax": None,
    },
)

SKILL_SCORE_PLOT_CONFIG_BY_COLUMN: dict[str, dict[str, object]] = {
    str(config["col"]): config for config in SKILL_SCORE_PLOT_CONFIGS
}


def calculate_discharge_metrics(
    discharge_comparison: pd.DataFrame,
) -> DischargeMetrics:
    """Calculate station-level discharge validation metrics.

    Args:
        discharge_comparison: Validation dataframe with observed and simulated discharge
            columns named `discharge_observations` and `discharge_simulations` (m3/s).

    Returns:
        DischargeMetrics with KGE, modified KGE, KGE correlation/bias/variability
        components, NSE, squared Pearson correlation r² (stored as `R2`), RMSE,
        and RRMSE; all NaN when there are fewer than 2 valid pairs. RMSE is in
        m3/s; all other metrics are dimensionless.

    """
    discharge_columns: list[str] = [
        "discharge_observations",
        "discharge_simulations",
    ]
    valid_pairs_df: pd.DataFrame = (
        discharge_comparison[discharge_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if valid_pairs_df.shape[0] < 2:
        return DischargeMetrics()

    observed: np.ndarray = valid_pairs_df["discharge_observations"].to_numpy(
        dtype=float
    )
    simulated: np.ndarray = valid_pairs_df["discharge_simulations"].to_numpy(
        dtype=float
    )
    observed_mean: float = float(observed.mean())
    simulated_mean: float = float(simulated.mean())
    observed_std: float = float(observed.std())
    simulated_std: float = float(simulated.std())

    # Original KGE uses correlation, mean-flow ratio, and standard-deviation ratio.
    correlation: float = (
        float(np.corrcoef(observed, simulated)[0, 1])
        if observed_std > 0 and simulated_std > 0
        else float("nan")
    )
    bias_ratio: float = (
        simulated_mean / observed_mean if observed_mean != 0 else float("nan")
    )
    variability_ratio: float = (
        simulated_std / observed_std if observed_std > 0 else float("nan")
    )
    kge: float = 1 - float(
        np.sqrt(
            (correlation - 1) ** 2
            + (bias_ratio - 1) ** 2
            + (variability_ratio - 1) ** 2
        )
    )
    # Modified KGE replaces the standard-deviation ratio with the ratio of
    # coefficients of variation, which equals variability_ratio / bias_ratio.
    variation_ratio: float = (
        variability_ratio / bias_ratio if bias_ratio != 0 else float("nan")
    )
    modified_kge: float = 1 - float(
        np.sqrt(
            (correlation - 1) ** 2 + (bias_ratio - 1) ** 2 + (variation_ratio - 1) ** 2
        )
    )
    mean_squared_error: float = float(np.mean((simulated - observed) ** 2))
    rmse: float = float(np.sqrt(mean_squared_error))
    nse: float = (
        1 - mean_squared_error / observed_std**2 if observed_std > 0 else float("nan")
    )
    rrmse: float = rmse / observed_std if observed_std > 0 else float("nan")

    return DischargeMetrics(
        KGE=kge,
        KGE_modified=modified_kge,
        KGE_correlation=correlation,
        KGE_bias_ratio=bias_ratio,
        KGE_variability_ratio=variability_ratio,
        NSE=nse,
        # R2 stores Pearson r², not the coefficient of determination.
        R2=correlation**2,
        RMSE=rmse,
        RRMSE=rrmse,
    )


def calculate_seasonal_discharge_metrics(
    daily_discharge_comparison: pd.DataFrame,
) -> dict[str, DischargeMetrics]:
    """Calculate daily discharge metrics for each meteorological season.

    All available daily values from the same season are pooled across years.
    This preserves the existing full-period evaluation while exposing seasonal
    differences in overall KGE and its correlation, bias, and variability
    components.

    Args:
        daily_discharge_comparison: Daily observed and simulated discharge (m3/s) with
            a DatetimeIndex.

    Returns:
        Discharge metrics keyed by lowercase season name.

    Raises:
        TypeError: If the dataframe does not use a DatetimeIndex.
    """
    if not isinstance(daily_discharge_comparison.index, pd.DatetimeIndex):
        raise TypeError("Seasonal discharge evaluation requires a DatetimeIndex.")

    month_numbers: np.ndarray = (
        daily_discharge_comparison.index.to_series().dt.month.to_numpy()
    )
    seasonal_metrics: dict[str, DischargeMetrics] = {}
    for season_name, season_months in METEOROLOGICAL_SEASONS.items():
        season_mask: np.ndarray = np.isin(month_numbers, season_months)
        metrics: DischargeMetrics = calculate_discharge_metrics(
            daily_discharge_comparison.loc[season_mask]
        )
        seasonal_metrics[season_name] = metrics
    return seasonal_metrics


def use_daily_discharge_scores(station_scores: pd.DataFrame) -> None:
    """Copy daily scores to the unsuffixed columns used by plots.

    Args:
        station_scores: Discharge evaluation table modified in place.
    """
    for metric_name in DischargeMetrics._fields:
        daily_column: str = f"{metric_name}_daily"
        if daily_column in station_scores.columns:
            station_scores[metric_name] = station_scores[daily_column]
