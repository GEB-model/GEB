"""Module implementing hydrology evaluation functions for the GEB model."""

import base64
import datetime
import os
from datetime import datetime
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rioxarray as rxr
import xarray as xr
from matplotlib.cm import get_cmap
from matplotlib.colors import LightSource, ListedColormap
from matplotlib import colormaps as mcolormaps
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from permetrics.regression import RegressionMetric
from tqdm import tqdm

from geb.evaluate.workflows.dashboard import create_discharge_folium_map
from geb.evaluate.workflows.hydrology_plot_engine import (
    plot_skill_score_boxplots as _plot_skill_score_boxplots,
    plot_skill_score_maps as _plot_skill_score_maps,
    plot_skill_scores_vs_upstream_area as _plot_skill_scores_vs_upstream_area,
)
from geb.hydrology.routing import read_discharge_per_river
from geb.reporter import WATER_STORAGE_REPORT_CONFIG
from geb.workflows.visualise import plot_sunburst

if TYPE_CHECKING:
    from geb.evaluate import Evaluate
    from geb.model import GEBModel

from geb.workflows.extreme_value_analysis import (
    ReturnPeriodModel,
)
from geb.workflows.io import read_geom, read_table
from geb.workflows.timeseries import regularize_discharge_timeseries

OBSERVATIONS_COLOR = "#FED65D"
SIMULATIONS_DEFAULT_COLOR = "#278DD9"

# Configure global dark style for all plots in this module
mpl.rcParams["figure.facecolor"] = "#000000"
mpl.rcParams["axes.facecolor"] = "#000000"
mpl.rcParams["axes.edgecolor"] = "white"
mpl.rcParams["axes.labelcolor"] = "white"
mpl.rcParams["xtick.color"] = "white"
mpl.rcParams["ytick.color"] = "white"
mpl.rcParams["text.color"] = "white"
mpl.rcParams["figure.edgecolor"] = "white"
mpl.rcParams["grid.color"] = "white"
mpl.rcParams["legend.labelcolor"] = "white"
mpl.rcParams["savefig.facecolor"] = "#000000"
mpl.rcParams["savefig.edgecolor"] = "#000000"


class DischargeMetrics(NamedTuple):
    """Discharge validation skill scores for a single station and time period."""

    KGE: float = float("nan")
    NSE: float = float("nan")
    R: float = float("nan")
    R2: float = float("nan")
    RMSE: float = float("nan")
    RRMSE: float = float("nan")


def _calculate_discharge_validation_metrics(
    validation_df: pd.DataFrame,
) -> DischargeMetrics:
    """Calculate station-level discharge validation metrics.

    Args:
        validation_df: Validation dataframe with observed and simulated discharge
            columns named `discharge_observations` and `discharge_simulations` (m3/s).

    Returns:
        DischargeMetrics with KGE, NSE, R, R2, RMSE, RRMSE; all NaN when there are
        fewer than 2 valid pairs.
    """
    valid_pairs_df: pd.DataFrame = validation_df[
        ["discharge_observations", "discharge_simulations"]
    ].dropna()
    if valid_pairs_df.shape[0] < 2:
        return DischargeMetrics()

    observed_discharge_values: np.ndarray = valid_pairs_df[
        "discharge_observations"
    ].values
    simulated_discharge_values: np.ndarray = valid_pairs_df[
        "discharge_simulations"
    ].values
    evaluator: RegressionMetric = RegressionMetric(
        observed_discharge_values, simulated_discharge_values
    )

    kge: float = float(evaluator.kling_gupta_efficiency())
    nse: float = float(evaluator.nash_sutcliffe_efficiency())
    r_value: float = float(evaluator.pearson_correlation_coefficient())
    r2: float = float(evaluator.pearson_correlation_coefficient_square())
    rmse: float = float(evaluator.root_mean_squared_error())
    # RRMSE = RMSE / mean(observed); protected against zero mean
    mean_observed_discharge: float = float(np.mean(observed_discharge_values))
    rrmse: float = (
        rmse / mean_observed_discharge
        if mean_observed_discharge > 0.0
        else float("nan")
    )

    return DischargeMetrics(KGE=kge, NSE=nse, R=r_value, R2=r2, RMSE=rmse, RRMSE=rrmse)


def _plot_validation_return_periods(
    validation_df: pd.DataFrame,
    station_id: Any,
    station_name: str,
    eval_plot_folder: Path,
    frequency: str,
) -> None:
    """Plot overlaid GPD-POT return-period curves and save a simplified version for popups.

    Args:
        validation_df: Validation dataframe containing `discharge_observations` and `discharge_simulations` (m3/s).
        station_id: Station identifier used in output file names.
        station_name: Human-readable station name.
        eval_plot_folder: Output directory for generated plots.
        frequency: Data frequency string for plot titles (e.g., "daily", "hourly").
    """
    return_periods_years: list[int | float] = [2, 5, 10, 25, 50, 100]

    # Use first_significant strategy for consistent evaluation
    strategy = "first_significant"
    fixed_shape = 0.0  # 0.0 is Gumbel distribution for better stability in validation

    obs_model = ReturnPeriodModel(
        series=validation_df["discharge_observations"],
        return_periods=return_periods_years,
        fixed_shape=fixed_shape,
        selection_strategy=strategy,
    )

    # For the simulated series, we want to ensure that we only
    # include values where there are corresponding observed values
    simulated_series: pd.Series = validation_df["discharge_simulations"].copy()
    simulated_series[validation_df["discharge_observations"].isna()] = np.nan

    sim_model = ReturnPeriodModel(
        series=simulated_series,
        return_periods=return_periods_years,
        fixed_shape=fixed_shape,
        selection_strategy=strategy,
    )

    # 1. Simplified Fit for Popups
    fig_simple, ax_fit_simple = plt.subplots(figsize=(14, 4))
    obs_model.plot_fit(
        ax=ax_fit_simple, label_prefix="Observed", color=OBSERVATIONS_COLOR
    )
    sim_model.plot_fit(
        ax=ax_fit_simple, label_prefix="Simulated", color=SIMULATIONS_DEFAULT_COLOR
    )
    ax_fit_simple.set_title(
        f"GPD-POT Return Periods ({frequency}): {station_name}",
        fontsize=14,
        fontweight="bold",
    )
    return_periods_folder: Path = eval_plot_folder / "return_periods"
    return_periods_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        return_periods_folder / f"return_period_fit_{station_id}.png",
        bbox_inches="tight",
        dpi=72,
    )
    plt.close(fig_simple)

    # 2. Large composite figure for detailed reports
    # Top row: Combined return level fit (wide)
    # Below: Two columns of diagnostics (Obs on left, Sim on right)
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(5, 2)

    # Combined Fit (Top)
    ax_fit = fig.add_subplot(gs[0, :])
    obs_model.plot_fit(ax=ax_fit, label_prefix="Observed", color=OBSERVATIONS_COLOR)
    sim_model.plot_fit(
        ax=ax_fit, label_prefix="Simulated", color=SIMULATIONS_DEFAULT_COLOR
    )
    ax_fit.set_title(
        f"GPD-POT Return Periods ({frequency}): {station_name} (ID: {station_id})",
        fontsize=16,
        fontweight="bold",
    )

    # Obs Diagnostics (Column 1)
    gs_obs = gs[1:, 0].subgridspec(4, 2)
    obs_axes_gof = [
        fig.add_subplot(gs_obs[0, 0]),
        fig.add_subplot(gs_obs[0, 1]),
        fig.add_subplot(gs_obs[1, 0]),
    ]
    obs_model.plot_gof(axes=obs_axes_gof)
    for ax in obs_axes_gof:
        ax.set_title(f"Obs: {ax.get_title()}", fontsize=10)

    obs_axes_sel = [
        fig.add_subplot(gs_obs[1, 1]),
        fig.add_subplot(gs_obs[2, 0]),
        fig.add_subplot(gs_obs[2, 1]),
        fig.add_subplot(gs_obs[3, 0]),
    ]
    obs_model.plot_selection_diagnostics(axes=obs_axes_sel)

    # Sim Diagnostics (Column 2)
    gs_sim = gs[1:, 1].subgridspec(4, 2)
    sim_axes_gof = [
        fig.add_subplot(gs_sim[0, 0]),
        fig.add_subplot(gs_sim[0, 1]),
        fig.add_subplot(gs_sim[1, 0]),
    ]
    sim_model.plot_gof(axes=sim_axes_gof)
    for ax in sim_axes_gof:
        ax.set_title(f"Sim: {ax.get_title()}", fontsize=10)

    sim_axes_sel = [
        fig.add_subplot(gs_sim[1, 1]),
        fig.add_subplot(gs_sim[2, 0]),
        fig.add_subplot(gs_sim[2, 1]),
        fig.add_subplot(gs_sim[3, 0]),
    ]
    sim_model.plot_selection_diagnostics(axes=sim_axes_sel)

    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.savefig(
        return_periods_folder / f"return_period_validation_{station_id}.svg",
        bbox_inches="tight",
    )
    plt.close()


def _plot_outflow_return_period(
    outflow_series_m3_per_s: pd.Series,
    outlet_id: str,
    outflow_plot_folder: Path,
    outflow_file_stem: str,
    frequency: str,
) -> None:
    """Plot complete GPD-POT diagnostics for one outflow time series.

    Args:
        outflow_series_m3_per_s: Outflow discharge time series (m3/s).
        outlet_id: Outflow outlet identifier.
        outflow_plot_folder: Output directory for outflow plots.
        outflow_file_stem: Base filename stem used to save the figure.
        frequency: Data frequency string for plot titles (e.g., "daily", "hourly").
    """
    return_periods_years: list[int | float] = [2, 5, 10, 25, 50, 100]
    model = ReturnPeriodModel(
        series=outflow_series_m3_per_s,
        return_periods=return_periods_years,
        fixed_shape=0.0,
        selection_strategy="best_fit",
    )

    fig = model.plot_diagnostics(figsize=(18, 14))
    fig.suptitle(
        f"Outflow Diagnostics ({frequency}): {outlet_id}",
        fontsize=16,
        fontweight="bold",
    )

    plt.savefig(
        outflow_plot_folder / f"{outflow_file_stem}_return_period.svg",
        bbox_inches="tight",
    )
    plt.close()


def _format_outflow_volume_caption(
    outflow_series_m3_per_s: pd.Series,
    year: int,
    total_area_m2: float,
) -> str:
    """Format the total annual outflow volume caption for one yearly subplot.

    Args:
        outflow_series_m3_per_s: Discharge series for one outlet (m3/s).
        year: Calendar year represented by the subplot.
        total_area_m2: Total basin area used for the depth conversion (m2).

    Returns:
        Caption text containing the annual total outflow volume (m3) and depth (mm).
    """
    time_index: pd.DatetimeIndex = pd.DatetimeIndex(outflow_series_m3_per_s.index)
    timestep_seconds: float = float(
        pd.Timedelta(
            pd.tseries.frequencies.to_offset(str(time_index.inferred_freq))
        ).total_seconds()
    )
    yearly_mask: np.ndarray = pd.Series(time_index).dt.year.to_numpy(dtype=int) == year
    total_outflow_m3: float = float(
        outflow_series_m3_per_s.loc[yearly_mask].sum() * timestep_seconds
    )
    total_outflow_mm: float = total_outflow_m3 * 1000.0 / total_area_m2
    return (
        f"total river outflow at point: {total_outflow_m3:,.0f} m3 "
        f"({total_outflow_mm:.2f} mm basin-equivalent)"
    )


def _align_context_series_to_outflow_index(
    context_series: pd.Series,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Align a lower-frequency context series to outflow timestamps.

    Args:
        context_series: Context series to align.
        target_index: Target outflow timestamps.

    Returns:
        Context series reindexed to `target_index`.
    """
    aligned_series: pd.Series = context_series.reindex(target_index, method="ffill")
    aligned_series = aligned_series.bfill()
    aligned_series.name = context_series.name
    return aligned_series


def _bucket_outflow_context_percent(
    context_percent: np.ndarray,
    bucket_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize outflow context values into a fixed number of buckets.

    Args:
        context_percent: Context values expressed in percent.
        bucket_count: Number of discrete color buckets.

    Returns:
        Tuple containing:
            - Bucket index per value.
            - Bucket-center percent per value.
    """
    clipped_context_percent = np.clip(context_percent, 0.0, 100.0)
    bucket_edges_percent = np.linspace(0.0, 100.0, bucket_count + 1)
    bucket_indices = np.digitize(
        clipped_context_percent,
        bucket_edges_percent[1:-1],
        right=False,
    )
    bucket_centers_percent = (
        bucket_edges_percent[:-1] + bucket_edges_percent[1:]
    ) / 2.0
    return bucket_indices, bucket_centers_percent[bucket_indices]


def _plot_outflow_line_with_context(
    axis: plt.Axes,
    time_index: pd.DatetimeIndex,
    outflow_series_m3_per_s: pd.Series,
    frozen_fraction_percent: pd.Series,
    frozen_fraction_cmap: mcolors.Colormap,
    linewidth: float,
    bucket_count: int = 10,
) -> LineCollection | None:
    """Plot an outflow line colored by the top-soil frozen fraction.

    Args:
        axis: Axis receiving the colored line.
        time_index: Timestamps shown on the x-axis.
        outflow_series_m3_per_s: Outflow series aligned to `time_index` (m3/s).
        frozen_fraction_percent: Basin-mean top-soil frozen fraction (%).
        frozen_fraction_cmap: Colormap where blue maps to 0% and white to 100%.
        linewidth: Line width for the colored outflow path.
        bucket_count: Number of discrete color buckets used for the line.

    Returns:
        The matplotlib line collection, or `None` if there are too few points.
    """
    if len(time_index) < 2:
        axis.plot(
            time_index,
            outflow_series_m3_per_s.to_numpy(dtype=float),
            color="#1f77b4",
            linewidth=linewidth,
            zorder=2,
        )
        return None

    time_values = mdates.date2num(time_index.to_numpy())
    outflow_values = outflow_series_m3_per_s.to_numpy(dtype=float)
    line_points = np.column_stack([time_values, outflow_values])
    frozen_values_percent = frozen_fraction_percent.to_numpy(dtype=float)
    segment_context_percent = (
        frozen_values_percent[:-1] + frozen_values_percent[1:]
    ) / 2.0
    bucket_indices, bucket_centers_percent = _bucket_outflow_context_percent(
        segment_context_percent,
        bucket_count=bucket_count,
    )
    discrete_cmap = mcolors.ListedColormap(
        frozen_fraction_cmap(np.linspace(0.0, 1.0, bucket_count))
    )
    bucket_edges_percent = np.linspace(0.0, 100.0, bucket_count + 1)
    discrete_norm = mcolors.BoundaryNorm(bucket_edges_percent, discrete_cmap.N)

    line_segments: list[np.ndarray[Any, Any]] = []
    merged_bucket_values_percent: list[float] = []
    run_start_idx = 0
    for segment_idx in range(1, len(bucket_indices)):
        if bucket_indices[segment_idx] != bucket_indices[run_start_idx]:
            line_segments.append(line_points[run_start_idx : segment_idx + 1])
            merged_bucket_values_percent.append(bucket_centers_percent[run_start_idx])
            run_start_idx = segment_idx
    line_segments.append(line_points[run_start_idx:])
    merged_bucket_values_percent.append(bucket_centers_percent[run_start_idx])

    line_collection = LineCollection(
        line_segments,
        cmap=discrete_cmap,
        norm=discrete_norm,
        linewidth=linewidth,
        zorder=2,
    )
    line_collection.set_array(np.asarray(merged_bucket_values_percent, dtype=float))
    axis.add_collection(line_collection)
    axis.update_datalim(line_points)
    axis.autoscale_view()
    axis.set_xlim(time_index[0], time_index[-1])
    return line_collection


def _style_dark_timeseries_axis(axis: plt.Axes) -> None:
    """Apply the shared dark styling used for hydrology timeseries plots.

    Args:
        axis: Axis to style.
    """
    # Styling is now mostly handled via global mpl.rcParams
    pass


def _style_outflow_axis(axis: plt.Axes) -> None:
    """Apply the dark outflow-plot styling used for frozen-soil context plots.

    Args:
        axis: Axis to style.
    """
    _style_dark_timeseries_axis(axis)


def _style_water_balance_axis(axis: plt.Axes) -> None:
    """Apply the dark styling used for water-balance line plots.

    Args:
        axis: Axis to style.
    """
    _style_dark_timeseries_axis(axis)


def _format_full_timeseries_axis(
    axis: plt.Axes,
    time_index: pd.DatetimeIndex,
    title: str,
    y_label: str,
    draw_zero_line: bool = False,
) -> None:
    """Format a full-run timeseries axis with consistent dark-theme behavior.

    Args:
        axis: Axis to format.
        time_index: Full time index shown on the axis.
        title: Axis title.
        y_label: Y-axis label.
        draw_zero_line: Whether to add a horizontal zero reference line.
    """
    if draw_zero_line:
        axis.axhline(0, color="white", linewidth=0.8, linestyle="--")
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.set_xlabel("Time")
    axis.set_xlim(time_index.min(), time_index.max())
    axis.margins(x=0)
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axis.xaxis.get_major_locator())
    )
    axis.grid(True, alpha=0.2, color="white")


def _format_yearly_timeseries_axis(
    axis: plt.Axes,
    year: int,
    title: str,
    y_label: str,
    draw_zero_line: bool = False,
) -> None:
    """Format a single-year timeseries axis with consistent dark-theme behavior.

    Args:
        axis: Axis to format.
        year: Calendar year shown on the axis.
        title: Axis title.
        y_label: Y-axis label.
        draw_zero_line: Whether to add a horizontal zero reference line.
    """
    year_start: pd.Timestamp = pd.Timestamp(year=year, month=1, day=1)  # ty:ignore[invalid-assignment]
    year_end: pd.Timestamp = pd.Timestamp(year=year, month=12, day=31, hour=23)  # ty:ignore[invalid-assignment]
    if draw_zero_line:
        axis.axhline(0, color="white", linewidth=0.8, linestyle="--")
    axis.set_xlim(mdates.date2num(year_start), mdates.date2num(year_end))
    axis.margins(x=0)
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.2, color="white")


def _add_dark_legend(
    axis: plt.Axes,
    loc: str,
    ncol: int,
    fontsize: float,
    bbox_to_anchor: tuple[float, float] | None = None,
) -> None:
    """Add a legend with consistent dark-theme styling.

    Args:
        axis: Axis receiving the legend.
        loc: Matplotlib legend location.
        ncol: Number of legend columns.
        fontsize: Legend font size.
        bbox_to_anchor: Optional anchor tuple for legends outside the axis.
    """
    axis.legend(
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        fontsize=fontsize,
        frameon=False,
        labelcolor="white",
    )


def _set_outflow_axis_limits(
    axis: plt.Axes,
    outflow_series_m3_per_s: pd.Series,
) -> None:
    """Set a non-clipping y-limit for an outflow discharge axis.

    Args:
        axis: Axis receiving the y-limit.
        outflow_series_m3_per_s: Discharge series used to derive the upper limit (m3/s).
    """
    finite_values = outflow_series_m3_per_s.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        axis.set_ylim(0.0, 1.0)
        return

    peak_discharge_m3_per_s: float = float(np.max(finite_values))
    upper_limit_m3_per_s: float = max(peak_discharge_m3_per_s * 1.05, 1.0)
    axis.set_ylim(0.0, upper_limit_m3_per_s)


def _plot_outflow_discharge_timeseries(
    model: Any,
    output_folder: Path,
    run_name: str,
    eval_plot_folder: Path,
) -> int:
    """Plot modeled outflow discharge time series without validation overlays.

    This helper reads exported outflow time series from the reporter output
    (`river_outflow_hourly_m3_per_s_*.csv`) and creates one line plot per outflow
    location using simulated discharge only.

    Args:
        model: Model-like object used to derive the total basin area.
        output_folder: Path to the model output folder.
        run_name: Name of the run to evaluate.
        eval_plot_folder: Evaluation plot output directory.

    Returns:
        Number of outflow plots created (dimensionless).
    """
    routing_dir: Path = output_folder / "report" / run_name / "hydrology.routing"
    if not routing_dir.exists():
        model.logger.info(
            f"No hydrology routing directory found at {routing_dir}. Skipping outflow plots."
        )
        return 0

    outflow_files: list[Path] = sorted(
        routing_dir.glob("river_outflow_hourly_m3_per_s_*.parquet")
    )
    if not outflow_files:
        model.logger.info(
            "No exported outflow time series found. Skipping outflow plots."
        )
        return 0

    outflow_plot_folder: Path = eval_plot_folder / "outflow"
    outflow_plot_folder.mkdir(parents=True, exist_ok=True)
    total_area_m2: float = _get_total_model_area_m2(model)
    report_folder: Path = output_folder / "report"
    frozen_fraction_series_name: str = "_top_soil_frozen_fraction"
    frozen_fraction_series: pd.Series | None = None
    run_folder: Path = report_folder / run_name
    frozen_fraction_path: Path = (
        run_folder / "hydrology.landsurface" / frozen_fraction_series_name
    ).with_suffix(".parquet")
    if frozen_fraction_path.exists():
        frozen_fraction_series = _read_evaluation_series_with_date_index(
            run_folder,
            "hydrology.landsurface",
            frozen_fraction_series_name,
        )
        frozen_fraction_series = frozen_fraction_series.sort_index()
        frozen_fraction_series = frozen_fraction_series.loc[
            ~frozen_fraction_series.index.duplicated(keep="last")
        ]

    frozen_fraction_cmap: mcolors.Colormap = mcolors.LinearSegmentedColormap.from_list(
        "top_soil_frozen_fraction",
        ["#1f77b4", "#ffffff"],
    )

    plots_created: int = 0
    for outflow_file in outflow_files:
        outflow_series: pd.Series = pd.read_parquet(outflow_file).squeeze()

        if np.isnan(outflow_series.values).all():
            model.logger.info(
                f"Outflow file {outflow_file.name} contains only NaN values."
            )
            continue

        outlet_id: str = outflow_file.stem.replace(
            "river_outflow_hourly_m3_per_s_",
            "",
        )
        aligned_frozen_fraction_percent: pd.Series | None = None
        if frozen_fraction_series is not None:
            aligned_frozen_fraction_percent = _align_context_series_to_outflow_index(
                frozen_fraction_series,
                pd.DatetimeIndex(outflow_series.index),
            )
            aligned_frozen_fraction_percent = aligned_frozen_fraction_percent * 100.0

        fig, ax = plt.subplots(figsize=(7, 4))
        _style_outflow_axis(ax)
        if aligned_frozen_fraction_percent is not None:
            _plot_outflow_line_with_context(
                axis=ax,
                time_index=pd.DatetimeIndex(outflow_series.index),
                outflow_series_m3_per_s=outflow_series,
                frozen_fraction_percent=aligned_frozen_fraction_percent,
                frozen_fraction_cmap=frozen_fraction_cmap,
                linewidth=1.1,
            )
        else:
            ax.plot(
                outflow_series.index,
                outflow_series.values,
                linewidth=0.9,
                color=SIMULATIONS_DEFAULT_COLOR,
                zorder=2,
            )
        ax.set_ylabel("Discharge [m3/s]")
        ax.set_xlabel("Time")
        _set_outflow_axis_limits(ax, outflow_series)
        ax.legend(
            handles=[Line2D([0], [0], color=SIMULATIONS_DEFAULT_COLOR, linewidth=1.1)],
            labels=["GEB outflow simulation (blue = unfrozen, white = fully frozen)"],
            facecolor="#000000",
            edgecolor="white",
            labelcolor="white",
        )
        ax.set_title(
            f"GEB river outflow for outlet {outlet_id}, mean: {outflow_series.mean():.2f} m3/s"
        )

        plt.savefig(
            outflow_plot_folder / f"{outflow_file.stem}.svg",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.show()
        plt.close(fig)

        outflow_time_index: pd.DatetimeIndex = pd.DatetimeIndex(outflow_series.index)
        outflow_year_values: np.ndarray = pd.Series(
            outflow_time_index
        ).dt.year.to_numpy(dtype=int)
        outflow_years: list[int] = sorted(np.unique(outflow_year_values).tolist())
        yearly_figure, yearly_axes = plt.subplots(
            len(outflow_years),
            1,
            figsize=(10, max(3.2 * len(outflow_years), 4.5)),
            sharey=True,
            facecolor="#000000",
        )
        if len(outflow_years) == 1:
            yearly_axes = [yearly_axes]

        for axis, year in zip(yearly_axes, outflow_years, strict=True):
            _style_outflow_axis(axis)
            yearly_mask: np.ndarray = outflow_year_values == year
            yearly_outflow_series: pd.Series = outflow_series.loc[yearly_mask]
            yearly_frozen_fraction_percent: pd.Series | None = None
            if aligned_frozen_fraction_percent is not None:
                yearly_frozen_fraction_percent = aligned_frozen_fraction_percent.loc[
                    yearly_mask
                ]
            if yearly_frozen_fraction_percent is not None:
                _plot_outflow_line_with_context(
                    axis=axis,
                    time_index=pd.DatetimeIndex(yearly_outflow_series.index),
                    outflow_series_m3_per_s=yearly_outflow_series,
                    frozen_fraction_percent=yearly_frozen_fraction_percent,
                    frozen_fraction_cmap=frozen_fraction_cmap,
                    linewidth=1.0,
                )
            else:
                axis.plot(
                    yearly_outflow_series.index,
                    yearly_outflow_series.values,
                    color="#1f77b4",
                    linewidth=0.9,
                    zorder=2,
                )
            axis.set_title(
                f"GEB river outflow for outlet {outlet_id} - {year}. Mean: {yearly_outflow_series.mean():.2f} m3/s"
            )
            axis.set_ylabel("Discharge [m3/s]")
            _set_outflow_axis_limits(axis, yearly_outflow_series)
            axis.grid(True, alpha=0.2, color="white")
            axis.margins(x=0)
            axis.xaxis.set_major_locator(mdates.MonthLocator())
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            axis.set_xlim(
                pd.Timestamp(year=year, month=1, day=1),
                pd.Timestamp(year=year, month=12, day=31, hour=23),
            )
            axis.text(
                0.01,
                -0.22,
                _format_outflow_volume_caption(outflow_series, year, total_area_m2),
                transform=axis.transAxes,
                fontsize=7,
                va="top",
                ha="left",
                color="white",
                clip_on=False,
            )

        yearly_axes[-1].set_xlabel("Time")
        yearly_figure.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.95,
            bottom=0.1,
            hspace=0.55,
        )
        plt.savefig(
            outflow_plot_folder / f"{outflow_file.stem}_yearly.svg",
            bbox_inches="tight",
            facecolor=yearly_figure.get_facecolor(),
            edgecolor="none",
        )
        plt.show()
        plt.close(yearly_figure)

        outflow_series.index.freq = outflow_series.index.inferred_freq  # ty:ignore[unresolved-attribute]

        _plot_outflow_return_period(
            outflow_series_m3_per_s=outflow_series,
            outlet_id=outlet_id,
            outflow_plot_folder=outflow_plot_folder,
            outflow_file_stem=outflow_file.stem,
            frequency="hourly",
        )

        plots_created += 1

    return plots_created


def create_validation_df(
    output_folder: Path,
    run_name: str,
    station_id: str | int,
    observed_discharge: pd.Series,
    correct_discharge_observations: bool,
    discharge_observations_to_GEB_upstream_area_ratio: float,
) -> pd.DataFrame:
    """Create a validation dataframe with the discharge observations and the GEB discharge simulation for the selected station.

    Args:
        output_folder: Path to the model output folder.
        run_name: Name of the simulation run to evaluate. Must correspond to an existing run directory
            in the model output folder.
        station_id: Station identifier to create the validation dataframe for.
        observed_discharge: Series with the discharge observations for the selected station.
        correct_discharge_observations: Whether to correct the discharge_observations discharge timeseries for the difference in upstream
            area between the discharge_observations station and the discharge from GEB.
        discharge_observations_to_GEB_upstream_area_ratio: The ratio of the upstream area of the discharge_observations station to the upstream area of the GEB discharge grid cell. This is used to correct

    Returns:
        DataFrame with the discharge observations and the GEB discharge simulation for the selected station.

    Raises:
        FileNotFoundError: If the hydrology routing directory does not exist.
        ValueError: If NaN values are found in the GEB discharge data after loading.
    """
    # Check if the hydrology.routing directory exists
    routing_dir = output_folder / "report" / run_name / "hydrology.routing"
    if not routing_dir.exists():
        raise FileNotFoundError(
            f"Hydrology routing directory does not exist: {routing_dir}"
        )

    # Construct the path to the individual station discharge file
    station_file_name = f"discharge_hourly_m3_per_s_{station_id}.parquet"
    station_file_path = routing_dir / station_file_name

    # Load the individual station discharge timeseries
    simulated_discharge = pd.read_parquet(station_file_path)[
        f"discharge_hourly_m3_per_s_{station_id}"
    ]

    if np.isnan(simulated_discharge.values).any():
        raise ValueError(
            f"NaN values found in GEB discharge data for station {station_id}. Please check the station file {station_file_path}."
        )

    simulated_discharge = simulated_discharge.asfreq(
        pd.infer_freq(simulated_discharge.index)
    )

    if correct_discharge_observations:
        """Correct observed discharge by upstream-area ratio when requested."""
        simulated_discharge = (
            simulated_discharge * discharge_observations_to_GEB_upstream_area_ratio
        )

    if not observed_discharge.index.is_monotonic_increasing:
        raise ValueError(
            "Observed discharge index must be a regular time series with a monotonic increasing DateTimeIndex."
        )

    assert observed_discharge.index.freq is not None, (  # ty:ignore[unresolved-attribute]
        "Observed discharge index must have a defined frequency."
    )
    # check if simulated discharge is at least as frequent as observed discharge, and if multiple of observed discharge frequency
    if simulated_discharge.index.freq > observed_discharge.index.freq:  # ty:ignore[unresolved-attribute]
        raise ValueError(
            "Simulated discharge frequency is lower than observed discharge frequency. Please ensure the simulated discharge is at least as frequent as the observed discharge."
        )
    if (
        observed_discharge.index.freq.nanos % simulated_discharge.index.freq.nanos  # ty:ignore[unresolved-attribute]
    ) != 0:
        raise ValueError(
            "Observed discharge frequency is not a multiple of simulated discharge frequency. Please ensure the observed discharge frequency is a multiple of the simulated discharge frequency."
        )

    # resample simulated discharge to match the frequency of observed discharge if needed
    simulated_discharge = simulated_discharge.resample(
        observed_discharge.index.freq  # ty:ignore[unresolved-attribute]
    ).mean()

    # cut both observed and simulated discharge to the same time range
    start_time = max(observed_discharge.index.min(), simulated_discharge.index.min())
    end_time = min(observed_discharge.index.max(), simulated_discharge.index.max())
    observed_discharge = observed_discharge.loc[start_time:end_time]
    simulated_discharge = simulated_discharge.loc[start_time:end_time]

    # Create a combined dataframe with the union of all timestamps.
    # Values will be NaN where data is missing in either series.
    validation_df = pd.DataFrame(
        {
            "discharge_observations": observed_discharge,
            "discharge_simulations": simulated_discharge,
        }
    )

    return validation_df


def _plot_discharge_validation_graphs(
    model: GEBModel,
    station_id: Any,
    validation_df: pd.DataFrame,
    station_name: str,
    upstream_area_ratio: float,
    kge: float,
    nse: float,
    r_value: float,
    eval_plot_folder: Path,
    include_yearly_plots: bool,
    frequency: str,
) -> None:
    """Plot station-level full timeseries, and optional yearly timeseries.

    Args:
        model: The GEB model
        station_id: Station identifier used in output file names.
        validation_df: Validation dataframe containing `discharge_observations` and `discharge_simulations` (m3/s).
        station_name: Human-readable station name.
        upstream_area_ratio: Ratio between observed and modeled upstream area
            (dimensionless).
        kge: Kling-Gupta efficiency (dimensionless).
        nse: Nash-Sutcliffe efficiency (dimensionless).
        r_value: Pearson correlation coefficient (dimensionless).
        eval_plot_folder: Output directory for generated plots.
        include_yearly_plots: Whether to generate per-year timeseries plots.
        frequency: Data frequency string for plot titles (e.g., "daily", "hourly").
    """
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(
        validation_df.index,
        validation_df["discharge_simulations"],
        label="Simulated",
        linewidth=0.5,
        color=SIMULATIONS_DEFAULT_COLOR,
    )
    ax.plot(
        validation_df.index,
        validation_df["discharge_observations"],
        label="Observed",
        linewidth=0.5,
        color=OBSERVATIONS_COLOR,
    )
    ax.set_ylabel("Discharge [m3/s]")
    ax.set_xlabel("Time")
    ax.set_ylim(0, None)
    ax.set_xlim(validation_df.index.min(), validation_df.index.max())
    ax.legend(loc="upper right", fontsize=10)

    if np.isfinite(r_value):
        ax.text(0.02, 0.9, f"$R^2$={r_value:.2f}", transform=ax.transAxes, fontsize=12)
        ax.text(0.02, 0.85, f"KGE={kge:.2f}", transform=ax.transAxes, fontsize=12)
        ax.text(0.02, 0.8, f"NSE={nse:.2f}", transform=ax.transAxes, fontsize=12)
    else:
        ax.text(
            0.02,
            0.9,
            "No overlapping values for score metrics",
            transform=ax.transAxes,
            fontsize=12,
        )
    ax.text(
        0.02,
        0.75,
        f"Mean={validation_df['discharge_simulations'].dropna().mean():.2f}",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.text(
        0.02,
        0.70,
        f"upstream area ratio: {upstream_area_ratio:.2f}",
        transform=ax.transAxes,
        fontsize=12,
    )
    plt.title(f"Discharge vs observations for station {station_name}")
    timeseries_folder: Path = eval_plot_folder / "timeseries"
    timeseries_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        timeseries_folder / f"timeseries_plot_{station_id}.svg",
        bbox_inches="tight",
    )
    plt.savefig(
        timeseries_folder / f"timeseries_plot_{station_id}.png",
        bbox_inches="tight",
        dpi=72,
    )
    plt.show()
    plt.close()

    if include_yearly_plots:
        years_to_plot: list[int] = sorted(validation_df.index.year.unique())  # ty:ignore[unresolved-attribute]
        for year in years_to_plot:
            one_year_df: pd.DataFrame = validation_df[validation_df.index.year == year]  # ty:ignore[unresolved-attribute]
            if one_year_df.empty:
                model.logger.info(f"No data available for year {year}, skipping.")
                continue

            fig, ax = plt.subplots(figsize=(13, 4))
            ax.plot(
                one_year_df.index,
                one_year_df["discharge_simulations"],
                label="Simulated",
                color=SIMULATIONS_DEFAULT_COLOR,
            )
            ax.plot(
                one_year_df.index,
                one_year_df["discharge_observations"],
                label="Observed",
                color=OBSERVATIONS_COLOR,
            )
            ax.set_xlim(one_year_df.index[0], one_year_df.index[-1])
            ax.set_ylabel("Discharge [m3/s]")
            ax.legend()

            ax.text(
                0.02, 0.9, f"$R^2$={r_value:.2f}", transform=ax.transAxes, fontsize=12
            )
            ax.text(0.02, 0.85, f"KGE={kge:.2f}", transform=ax.transAxes, fontsize=12)
            ax.text(0.02, 0.8, f"NSE={nse:.2f}", transform=ax.transAxes, fontsize=12)
            ax.text(
                0.02,
                0.75,
                f"upstream area ratio: {upstream_area_ratio:.2f}",
                transform=ax.transAxes,
                fontsize=12,
            )

            plt.title(
                f"GEB discharge vs observations for {year} at station {station_name}"
            )
            plt.savefig(
                timeseries_folder / f"timeseries_plot_{station_id}_{year}.svg",
                bbox_inches="tight",
            )
            plt.show()
            plt.close()

    _plot_validation_return_periods(
        validation_df=validation_df,
        station_id=station_id,
        station_name=station_name,
        eval_plot_folder=eval_plot_folder,
        frequency=frequency,
    )


def _read_evaluation_series_with_date_index(
    folder: Path,
    module: str,
    name: str,
) -> pd.Series:
    """Read an evaluation time series from a parquet file.

    Args:
        folder: Path to the report folder for one model run.
        module: Name of the module subfolder containing the parquet file.
        name: Name of the parquet file without the `.parquet` suffix.

    Returns:
        Time-indexed series read from the parquet file.
    """
    series: pd.Series = pd.read_parquet(
        (folder / module / name).with_suffix(".parquet"),
        engine="pyarrow",
    )[name]
    return series


def _load_named_evaluation_series(
    folder: Path,
    series_specs: dict[str, tuple[str, str]],
) -> dict[str, pd.Series]:
    """Load a named collection of evaluation time series from parquet files.

    Args:
        folder: Path to the report folder for one model run.
        series_specs: Mapping from output name used by the caller to a tuple of
            `(module, reported_name)` describing where the parquet series lives.

    Returns:
        Mapping of caller-defined series names to time-indexed pandas series.
    """
    return {
        series_name: _read_evaluation_series_with_date_index(
            folder,
            module_name,
            reported_name,
        )
        for series_name, (module_name, reported_name) in series_specs.items()
    }


def _load_evaluation_dataframe(
    folder: Path,
    series_specs: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Load a collection of evaluation series into one time-indexed dataframe.

    Args:
        folder: Path to the report folder for one model run.
        series_specs: Mapping from dataframe column names to `(module, reported_name)`
            tuples describing where each CSV series lives.

    Returns:
        Dataframe with one column per requested series.
    """
    return pd.DataFrame(
        _load_named_evaluation_series(folder, series_specs)
    ).sort_index()


def _flatten_water_balance_hierarchy(
    prefix: str,
    hierarchy: dict[str, Any],
    flattened_series: dict[str, pd.Series],
) -> None:
    """Flatten a nested water balance hierarchy into a flat column mapping.

    Args:
        prefix: Current prefix for nested names.
        hierarchy: Nested mapping with dict nodes and `pd.Series` leaves.
        flattened_series: Output mapping populated in place.
    """
    for key, value in hierarchy.items():
        column_name: str = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_water_balance_hierarchy(column_name, value, flattened_series)
        elif isinstance(value, pd.Series):
            flattened_series[column_name] = value


def _load_water_balance_dataframe(folder: Path) -> pd.DataFrame:
    """Load water balance component time series for one run.

    Notes:
        Output components remain positive in the returned dataframe. Callers that
        want a signed plotting convention should negate the `out_` columns.

    Args:
        folder: Path to the report folder for one model run.

    Returns:
        Dataframe with one column per water balance component (m3 per timestep).
    """
    balance_series: dict[str, pd.Series] = _load_named_evaluation_series(
        folder,
        {
            "storage": ("hydrology", "_current_storage"),
            "rain": ("hydrology.landsurface", "_rain_m"),
            "snow": ("hydrology.landsurface", "_snow_m"),
            "domestic_water_loss": (
                "hydrology.water_demand",
                "_domestic_water_loss_m3",
            ),
            "industry_water_loss": (
                "hydrology.water_demand",
                "_industry_water_loss_m3",
            ),
            "livestock_water_loss": (
                "hydrology.water_demand",
                "_livestock_water_loss_m3",
            ),
            "river_outflow": ("hydrology.routing", "_total_outflow_at_pits_m3"),
            "transpiration": (
                "hydrology.landsurface",
                "_transpiration_m",
            ),
            "bare_soil_evaporation": (
                "hydrology.landsurface",
                "_bare_soil_evaporation_m",
            ),
            "open_water_evaporation": (
                "hydrology.landsurface",
                "_open_water_evaporation_m",
            ),
            "interception_evaporation": (
                "hydrology.landsurface",
                "_interception_evaporation_m",
            ),
            "sublimation_or_deposition": (
                "hydrology.landsurface",
                "_sublimation_or_deposition_m",
            ),
            "river_evaporation": (
                "hydrology.routing",
                "_total_evaporation_in_rivers_m3",
            ),
            "waterbody_evaporation": (
                "hydrology.routing",
                "_total_waterbody_evaporation_m3",
            ),
        },
    )

    storage_m3: pd.Series = balance_series["storage"]
    rain_m3: pd.Series = balance_series["rain"]
    snow_m3: pd.Series = balance_series["snow"]
    domestic_water_loss_m3: pd.Series = balance_series["domestic_water_loss"]
    industry_water_loss_m3: pd.Series = balance_series["industry_water_loss"]
    livestock_water_loss_m3: pd.Series = balance_series["livestock_water_loss"]
    river_outflow_m3: pd.Series = balance_series["river_outflow"]
    transpiration_m3: pd.Series = balance_series["transpiration"]
    bare_soil_evaporation_m3: pd.Series = balance_series["bare_soil_evaporation"]
    open_water_evaporation_m3: pd.Series = balance_series["open_water_evaporation"]
    interception_evaporation_m3: pd.Series = balance_series["interception_evaporation"]
    sublimation_or_deposition_m3: pd.Series = balance_series[
        "sublimation_or_deposition"
    ]
    river_evaporation_m3: pd.Series = balance_series["river_evaporation"]
    waterbody_evaporation_m3: pd.Series = balance_series["waterbody_evaporation"]

    storage_change_m3: pd.Series = storage_m3.diff().fillna(0)
    hierarchy: dict[str, Any] = {
        "in": {
            "rain": rain_m3,
            "snow": snow_m3,
        },
        "out": {
            "evapotranspiration": {
                "transpiration": transpiration_m3,
                "bare_soil_evaporation": bare_soil_evaporation_m3,
                "open_water_evaporation": open_water_evaporation_m3,
                "interception_evaporation": interception_evaporation_m3,
                "river_evaporation": river_evaporation_m3,
                "waterbody_evaporation": waterbody_evaporation_m3,
            },
            "water_demand": {
                "domestic_water_loss": domestic_water_loss_m3,
                "industry_water_loss": industry_water_loss_m3,
                "livestock_water_loss": livestock_water_loss_m3,
            },
            "river_outflow": river_outflow_m3,
        },
        "storage_change": storage_change_m3,
    }

    if sublimation_or_deposition_m3.sum() > 0:
        hierarchy["in"]["deposition"] = sublimation_or_deposition_m3
    else:
        hierarchy["out"]["evapotranspiration"]["sublimation"] = abs(
            sublimation_or_deposition_m3
        )

    flattened_series: dict[str, pd.Series] = {}
    _flatten_water_balance_hierarchy("", hierarchy, flattened_series)
    return pd.DataFrame(flattened_series).sort_index()


def _load_contextual_water_balance_series(folder: Path) -> dict[str, pd.Series]:
    """Load optional context series that support water-balance interpretation.

    Notes:
        These series are not part of the actual water balance and therefore must
        not be included in the balance dataframe, signed output conversion, or
        annual balance summaries.

    Args:
        folder: Path to the report folder for one model run.

    Returns:
        Mapping of context series names to their time series.
    """
    return _load_named_evaluation_series(
        folder,
        {
            "potential_evapotranspiration": (
                "hydrology.landsurface",
                "_potential_evapotranspiration_m",
            )
        },
    )


def _format_water_balance_component_label(column_name: str) -> str:
    """Format a water balance column name for plot legends.

    Args:
        column_name: Raw dataframe column name.

    Returns:
        Human-readable legend label.
    """
    simplified_column_name: str = column_name
    if simplified_column_name.startswith("in_"):
        simplified_column_name = simplified_column_name.removeprefix("in_")
    elif simplified_column_name.startswith("out_"):
        simplified_column_name = simplified_column_name.removeprefix("out_")

    simplified_column_name = simplified_column_name.removeprefix("evapotranspiration_")
    simplified_column_name = simplified_column_name.removeprefix("water_demand_")
    return simplified_column_name.replace("_", " ")


def _format_water_balance_context_label(series_name: str) -> str:
    """Format an optional water-balance context series label for plot legends.

    Args:
        series_name: Raw context series name.

    Returns:
        Human-readable legend label.
    """
    if series_name == "potential_evapotranspiration":
        return "potential ET"
    return series_name.replace("_", " ")


def _format_yearly_totals_caption_lines(
    prefix: str,
    column_names: list[str],
    values_mm: pd.Series,
    labels: dict[str, str],
    items_per_line: int,
) -> list[str]:
    """Format grouped yearly totals caption lines for one component direction.

    Args:
        prefix: Direction label such as `inputs` or `outputs`.
        column_names: Ordered component columns to render.
        values_mm: Annual component totals for one year (mm/year).
        labels: Human-readable labels for each component column.
        items_per_line: Maximum number of caption items per rendered line.

    Returns:
        Caption lines with the direction prefix shown only once per line.
    """
    if not column_names:
        return []

    lines: list[str] = []
    for start_index in range(0, len(column_names), items_per_line):
        chunk: list[str] = column_names[start_index : start_index + items_per_line]
        chunk_text: str = " | ".join(
            f"{labels[column_name]}: {values_mm[column_name]:.1f}"
            for column_name in chunk
        )
        lines.append(f"{prefix}: {chunk_text}")
    return lines


def _get_total_model_area_m2(model: Any) -> float:
    """Derive the total model area used for converting volumes to depths.

    Args:
        model: Model-like object expected to expose the basin mask geometry.

    Returns:
        Total model area represented by the evaluation outputs (m2).

    Raises:
        ValueError: If no positive total area can be derived from the model mask.
    """
    files: Any = getattr(model, "files", None)
    if files is not None:
        geom_files: Any = files.get("geom") if hasattr(files, "get") else None
        if geom_files is not None and "mask" in geom_files:
            total_area_m2: float = float(
                read_geom(geom_files["mask"]).to_crs("ESRI:54009").area.sum()
            )
            if total_area_m2 > 0:
                return total_area_m2

    raise ValueError("No positive area could be derived from the model mask geometry.")


def _create_yearly_totals_summary_mm(
    water_balance_df_m3_per_timestep: pd.DataFrame,
    total_area_m2: float,
) -> pd.DataFrame:
    """Summarize annual water balance totals per component as depths.

    Notes:
        Output components are converted to negative depths and storage change
        retains its sign. This mirrors the signed plotting convention used in
        the time-series figures.

    Args:
        water_balance_df_m3_per_timestep: Water balance components (m3 per timestep).
        total_area_m2: Total model area represented by the reported fluxes (m2).

    Returns:
        Dataframe indexed by calendar year with one column per component in mm/year.

    """
    annual_totals_m3: pd.DataFrame = water_balance_df_m3_per_timestep.resample(
        "YE"
    ).sum()
    conversion_factor_mm_per_m3: float = 1000.0 / total_area_m2

    summary_mm: pd.DataFrame = annual_totals_m3 * conversion_factor_mm_per_m3
    output_columns: list[str] = [
        column_name
        for column_name in summary_mm.columns
        if column_name.startswith("out_")
    ]
    summary_mm.loc[:, output_columns] = -summary_mm.loc[:, output_columns]
    summary_mm.index = summary_mm.index.year  # ty:ignore[unresolved-attribute]
    return summary_mm


def _get_datetime_index_step_label(time_index: pd.DatetimeIndex) -> str:
    """Infer a compact timestep label from a datetime index.

    Args:
        time_index: Datetime index for the plotted series.

    Returns:
        Compact timestep label such as `H`, `D`, or `MS`.

    Raises:
        ValueError: If the datetime frequency cannot be determined from the index.
    """
    frequency_label_map: dict[str, str] = {
        "D": "day",
    }

    if time_index.freq is not None and time_index.freq.freqstr is not None:
        frequency_label: str = str(time_index.freq.freqstr).upper()
        return frequency_label_map.get(frequency_label, frequency_label)

    inferred_frequency: str | None = pd.infer_freq(time_index)
    if inferred_frequency is not None:
        normalized_frequency: str = inferred_frequency.upper()
        return frequency_label_map.get(normalized_frequency, normalized_frequency)

    raise ValueError(
        "Could not determine the timestep frequency from the datetime index."
    )


def _add_yearly_totals_caption(
    axis: plt.Axes,
    year: int,
    yearly_totals_mm: pd.DataFrame,
    component_labels: dict[str, str],
    yearly_context_totals_mm: pd.DataFrame | None = None,
    context_labels: dict[str, str] | None = None,
) -> None:
    """Add a compact annual totals caption to a yearly water-balance subplot.

    Args:
        axis: Parent axis that receives the caption.
        year: Calendar year represented by the subplot.
        yearly_totals_mm: Annual totals indexed by year and expressed in mm/year.
        component_labels: Human-readable labels for each component column.
        yearly_context_totals_mm: Optional annual context totals indexed by year and
            expressed in mm/year.
        context_labels: Human-readable labels for each context series column.
    """
    if yearly_totals_mm.empty or year not in yearly_totals_mm.index:
        return

    yearly_values_mm: pd.Series = yearly_totals_mm.loc[year]
    ordered_columns: list[str] = list(yearly_totals_mm.columns)
    input_columns: list[str] = [
        column_name for column_name in ordered_columns if column_name.startswith("in_")
    ]
    output_columns: list[str] = [
        column_name for column_name in ordered_columns if column_name.startswith("out_")
    ]
    storage_columns: list[str] = [
        column_name
        for column_name in ordered_columns
        if column_name not in input_columns and column_name not in output_columns
    ]
    caption_lines: list[str] = []
    components_per_line: int = 4
    input_total_mm: float = float(yearly_values_mm[input_columns].sum())
    output_total_mm: float = float(-yearly_values_mm[output_columns].sum())
    caption_lines.extend(
        _format_yearly_totals_caption_lines(
            prefix="inputs",
            column_names=input_columns,
            values_mm=yearly_values_mm,
            labels=component_labels,
            items_per_line=components_per_line,
        )
    )
    caption_lines.extend(
        _format_yearly_totals_caption_lines(
            prefix="outputs",
            column_names=output_columns,
            values_mm=yearly_values_mm,
            labels=component_labels,
            items_per_line=components_per_line,
        )
    )
    if storage_columns:
        caption_lines.extend(
            _format_yearly_totals_caption_lines(
                prefix="storage",
                column_names=storage_columns,
                values_mm=yearly_values_mm,
                labels=component_labels,
                items_per_line=components_per_line,
            )
        )

    caption_text_lines: list[str] = ["mm/year\n" + "\n".join(caption_lines)]
    caption_text_lines.append(
        f"sum input: {input_total_mm:.1f} | sum output: {output_total_mm:.1f}"
    )

    if (
        yearly_context_totals_mm is not None
        and context_labels is not None
        and not yearly_context_totals_mm.empty
        and year in yearly_context_totals_mm.index
    ):
        context_values_mm: pd.Series = yearly_context_totals_mm.loc[year]
        ordered_context_columns: list[str] = list(yearly_context_totals_mm.columns)
        context_caption_parts: list[str] = [
            f"{context_labels[column_name]}: {context_values_mm[column_name]:.1f}"
            for column_name in ordered_context_columns
        ]
        caption_text_lines.append("context: " + " | ".join(context_caption_parts))

    caption_text: str = "\n".join(caption_text_lines)
    axis.text(
        0.01,
        -0.24,
        caption_text,
        transform=axis.transAxes,
        fontsize=6,
        va="top",
        ha="left",
        linespacing=1.15,
        color="white",
        clip_on=False,
    )


def _load_top_soil_water_balance_dataframe(folder: Path) -> pd.DataFrame:
    """Load top-soil-layer water balance diagnostics for one run.

    Notes:
        This dataframe is limited to terms that contribute directly to the
        reported top-soil storage balance. Additional land-surface terms that
        help interpret the plot, such as precipitation and runoff before
        infiltration enters the control volume, are loaded separately as
        context series.

    Args:
        folder: Path to the report folder for one model run.

    Returns:
        Dataframe with one column per top-soil water balance component (m3 per timestep).
    """
    top_soil_series: dict[str, pd.Series] = _load_named_evaluation_series(
        folder,
        {
            "storage": ("hydrology.landsurface", "_top_soil_water_content_m"),
            "infiltration": (
                "hydrology.landsurface",
                "_top_soil_infiltration_m",
            ),
            "rise_from_layer_2": (
                "hydrology.landsurface",
                "_top_soil_rise_from_layer_2_m",
            ),
            "evaporation": (
                "hydrology.landsurface",
                "_top_soil_evaporation_m",
            ),
            "transpiration": (
                "hydrology.landsurface",
                "_top_soil_transpiration_m",
            ),
            "percolation_to_layer_2": (
                "hydrology.landsurface",
                "_top_soil_percolation_to_layer_2_m",
            ),
        },
    )

    top_soil_storage_m3: pd.Series = top_soil_series["storage"]
    top_soil_infiltration_m3: pd.Series = top_soil_series["infiltration"]
    top_soil_rise_from_layer_2_m3: pd.Series = top_soil_series["rise_from_layer_2"]
    top_soil_evaporation_m3: pd.Series = top_soil_series["evaporation"]
    top_soil_transpiration_m3: pd.Series = top_soil_series["transpiration"]
    top_soil_percolation_to_layer_2_m3: pd.Series = top_soil_series[
        "percolation_to_layer_2"
    ]

    top_soil_storage_change_m3: pd.Series = top_soil_storage_m3.diff().fillna(0)
    hierarchy: dict[str, Any] = {
        "in": {
            "infiltration": top_soil_infiltration_m3,
            "rise_from_layer_2": top_soil_rise_from_layer_2_m3,
        },
        "out": {
            "evaporation": top_soil_evaporation_m3,
            "transpiration": top_soil_transpiration_m3,
            "percolation_to_layer_2": top_soil_percolation_to_layer_2_m3,
        },
        "storage_change": top_soil_storage_change_m3,
    }

    flattened_series: dict[str, pd.Series] = {}
    _flatten_water_balance_hierarchy("", hierarchy, flattened_series)
    return pd.DataFrame(flattened_series).sort_index()


def _load_contextual_top_soil_water_balance_series(
    folder: Path,
) -> dict[str, pd.Series]:
    """Load land-surface context series for the top-soil water balance plots.

    Notes:
        These series help explain how precipitation is partitioned before water
        enters or leaves the top-soil control volume, and how atmospheric
        demand linked to that store varies over time. They stay outside the
        strict top-soil storage balance and the balance totals.

    Args:
        folder: Path to the report folder for one model run.

    Returns:
        Mapping of context series names to their time series.
    """
    return _load_named_evaluation_series(
        folder,
        {
            # _rain_m is identical to the former top_soil_precipitation (same varname)
            "precipitation": (
                "hydrology.landsurface",
                "_rain_m",
            ),
            "runoff": (
                "hydrology.landsurface",
                "_runoff_m_daily",
            ),
            # _snow_m is identical to the former top_soil_snow (same varname)
            "snow": (
                "hydrology.landsurface",
                "_snow_m",
            ),
            "potential_evapotranspiration": (
                "hydrology.landsurface",
                "_potential_evapotranspiration_m",
            ),
        },
    )


def _load_water_storage_dataframe(folder: Path) -> pd.DataFrame:
    """Load reported water storage component time series for one run.

    Args:
        folder: Path to the report folder for one model run.

    Returns:
        Dataframe with one column per reported water storage component (m).

    Raises:
        ValueError: If no water storage reporter outputs are available. This
            usually means `report._water_storage` was disabled during the run.
    """
    module_name: str = "hydrology.landsurface"
    reported_storage_names: list[str] = list(
        WATER_STORAGE_REPORT_CONFIG[module_name].keys()
    )
    storage_specs: dict[str, tuple[str, str]] = {
        reported_name.removeprefix("_").removesuffix("_m"): (
            module_name,
            reported_name,
        )
        for reported_name in reported_storage_names
    }

    try:
        return _load_evaluation_dataframe(folder, storage_specs)
    except FileNotFoundError as exc:
        raise ValueError(
            "Water storage outputs are missing. Enable report._water_storage during the run and rerun before calling hydrology.plot_water_storage."
        ) from exc


def _format_water_storage_component_label(column_name: str) -> str:
    """Format a water storage column name for plot legends.

    Args:
        column_name: Raw dataframe column name.

    Returns:
        Human-readable legend label.
    """
    return column_name.replace("_", " ")


def _get_top_soil_water_balance_label(column_name: str) -> str:
    """Format top-soil water balance labels for plot legends.

    Args:
        column_name: Raw dataframe column name.

    Returns:
        Human-readable legend label with an explicit storage-change description.
    """
    if column_name == "storage_change":
        return "storage change (from top-soil storage)"
    return _format_water_balance_component_label(column_name)


class Hydrology:
    """Implements several functions to evaluate the hydrological module of GEB."""

    def __init__(self, model: GEBModel, evaluator: Evaluate) -> None:
        """Initialize the Hydrology evaluation module."""
        self.model = model
        self.evaluator = evaluator

    def get_discharge_per_river(
        self, run_name: str
    ) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """Get the discharge per river from the report directory.

        Args:
            run_name: Name of the simulation run to evaluate. Must correspond to an existing
                run directory in the model output folder.

        Raises:
            FileNotFoundError: If the discharge file for the specified run does not exist
                in the report directory.

        Returns:
            A GeoDataFrame containing the river geometries and a DataFrame containing the discharge data for each river.
        """
        # check if discharge files exists
        discharge_folder = (
            self.model.output_folder / "report" / run_name / "hydrology.routing"
        )
        if not discharge_folder.exists():
            raise FileNotFoundError(
                f"Discharge files for run '{run_name}' does not exist in the report directory. Did you run the model?"
            )

        # load rivers
        all_rivers: gpd.GeoDataFrame = read_geom(
            self.model.files["geom"]["routing/rivers"]
        )
        rivers_of_interest: gpd.GeoDataFrame = all_rivers[
            ~(
                all_rivers["is_downstream_outflow"]
                | all_rivers["is_upstream_of_downstream_basin"]
                | all_rivers["is_further_downstream_outflow"]
            )
        ].copy()

        # In merged multi-cluster runs some rivers may not have output files, mostly caused by the outflow reporter to be false in the model.yml. Filter out those rivers here.
        rivers_of_interest = rivers_of_interest[
            rivers_of_interest.index.map(
                lambda rid: (
                    discharge_folder / f"river_outflow_hourly_m3_per_s_{rid}.parquet"
                ).exists()
            )
        ].copy()

        discharge: pd.DataFrame = read_discharge_per_river(
            folder=discharge_folder,
            rivers=rivers_of_interest,
            all_rivers=all_rivers,
        )
        return rivers_of_interest, discharge

    def plot_discharge(
        self,
        run_name: str = "default",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Plot the mean discharge map and all exported outflow time series.

        Creates a spatial visualization of mean discharge values over time from the GEB model
        simulation results. The mean discharge field is saved as both a zarr file and PNG image.
        If outflow-point reporter CSV files are available, the method also creates one time-series
        plot per outflow point in the hydrology evaluation output folder.

        Notes:
            The discharge data must exist in the report directory structure. If the discharge
            file is not found, a FileNotFoundError will be raised. The mean is calculated
            across the entire simulation time period.

        Args:
            run_name: Name of the simulation run to plot. Must correspond to an existing
                run directory in the model output folder.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        if self.discharge_output_folder.exists():
            shutil.rmtree(self.discharge_output_folder)
        self.discharge_output_folder.mkdir(parents=True, exist_ok=True)

        rivers_of_interest, discharge = self.get_discharge_per_river(run_name)
        for river_id in discharge.columns:
            rivers_of_interest.loc[river_id, "discharge_m3_per_s"] = discharge[
                river_id
            ].mean()

        ax = rivers_of_interest.plot(
            column="discharge_m3_per_s",
            cmap="Blues",
            legend=True,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Mean discharge (m3/s)")

        plt.savefig(
            self.discharge_output_folder / "mean_discharge_m3_per_s_map.svg",
        )
        plt.close()

        outflow_plot_count: int = _plot_outflow_discharge_timeseries(
            model=self.model,
            output_folder=self.model.output_folder,
            run_name=run_name,
            eval_plot_folder=self.discharge_output_folder,
        )
        if outflow_plot_count > 0:
            self.model.logger.info(
                f"Created {outflow_plot_count} outflow discharge plots."
            )

    def evaluate_discharge(
        self,
        run_name: str = "default",
        include_yearly_plots: bool = True,
        correct_discharge_observations: bool = False,
        create_plots: bool = True,
        minimum_upstream_area_km2: float | None = None,
        minimum_timeseries_length_years: float | None = None,
    ) -> dict[str, float | None]:
        """Evaluate the discharge grid from GEB against observations from the discharge observations database.

        Compares simulated discharge from the GEB model with observed discharge data from
        gauging stations. Calculates performance metrics (KGE, NSE, R) and creates
        evaluation plots and interactive maps for analysis.

        Notes:
            The discharge simulation files must exist in the report directory structure.
            If no discharge stations are found in the basin, empty evaluation datasets
            are created. The evaluation can be skipped if results already exist.

        Args:
            run_name: Name of the simulation run to evaluate. Must correspond to an
                existing run directory in the model output folder.
            include_yearly_plots: Whether to create plots for every year showing the evaluation.
            correct_discharge_observations: Whether to correct the discharge observations discharge timeseries for the difference
                in upstream area between the discharge observations station and the discharge from GEB.
            create_plots: Whether to create evaluation plots. Set to False to only calculate the evaluation metrics and save the results without plotting.
            minimum_upstream_area_km2: Optional minimum modeled upstream area threshold for station evaluation (km2).
                If omitted, `hydrology.evaluation.discharge.minimum_upstream_area_km2` is used.
            minimum_timeseries_length_years: Optional minimum paired observation-simulation timeseries length for station evaluation (years).
                If omitted, `hydrology.evaluation.discharge.minimum_timeseries_length_years` is used.

        Returns:
            Dictionary containing mean metrics (KGE, NSE, R). In addition, the returned dictionary contains
            frequency-specific metrics (e.g., KGE_hourly, KGE_daily).
            Stations with hourly data are also evaluated on the daily resampled data, and those metrics are included in
            the returned dictionary. Stations with only daily data are not evaluated on the hourly data.

        Raises:
            FileNotFoundError: If the run folder does not exist in the report directory.
            ValueError: If a non-existing frequency label is encountered in the discharge observations data.
        """
        if self.evaluate_discharge_output_folder.exists():
            shutil.rmtree(self.evaluate_discharge_output_folder)
        self.evaluate_discharge_output_folder.mkdir(parents=True, exist_ok=True)

        if minimum_upstream_area_km2 is None:
            minimum_upstream_area_km2 = self.model.config["hydrology"]["evaluation"][
                "discharge"
            ]["minimum_upstream_area_km2"]
        self.model.logger.info(
            "Using %.1f km2 as the minimum upstream area threshold for discharge evaluation.",
            minimum_upstream_area_km2,
        )
        if minimum_timeseries_length_years is None:
            minimum_timeseries_length_years = self.model.config["hydrology"][
                "evaluation"
            ]["discharge"]["minimum_timeseries_length_years"]
        self.model.logger.info(
            "Using %.2f years as the minimum paired timeseries length for discharge evaluation.",
            minimum_timeseries_length_years,
        )

        # load input data files
        discharge_observations_hourly: pd.DataFrame = read_table(
            self.model.files["table"]["discharge/discharge_observations_hourly"]
        )
        discharge_observations_daily: pd.DataFrame = read_table(
            self.model.files["table"]["discharge/discharge_observations_daily"]
        )
        waterbodies = read_geom(self.model.files["geom"]["waterbodies/waterbody_data"])

        if not discharge_observations_hourly.empty:
            discharge_observations_hourly = regularize_discharge_timeseries(
                discharge_observations_hourly
            )
        if not discharge_observations_daily.empty:
            discharge_observations_daily = regularize_discharge_timeseries(
                discharge_observations_daily
            )

        snapped_locations = read_geom(
            self.model.files["geom"]["discharge/discharge_snapped_locations"]
        )

        self.model.logger.info(f"Loaded discharge simulation from {run_name} run.")

        # check if run file exists, if not, raise an error
        if not (self.model.output_folder / "report" / run_name).exists():
            raise FileNotFoundError(
                f"Run folder '{run_name}' does not exist in the report directory. Did you run the model?"
            )

        evaluation_per_station: list = []

        self.model.logger.info("Starting discharge evaluation...")
        for frequency_label, discharge_observations_df in zip(
            ["hourly", "daily"],
            [
                discharge_observations_hourly,
                discharge_observations_daily,
            ],
            strict=True,
        ):
            if discharge_observations_df.empty:
                continue
            for station_id in tqdm(discharge_observations_df.columns):
                # create a discharge timeseries dataframe
                observed_discharge_series = discharge_observations_df[station_id]
                if isinstance(observed_discharge_series, pd.DataFrame):
                    observed_discharge_series.columns = ["Q"]
                observed_discharge_series.name = "Q"

                # extract the properties from the snapping dataframe
                discharge_observations_station_name = snapped_locations.loc[
                    station_id
                ].discharge_observations_station_name
                discharge_observations_station_coords = snapped_locations.loc[
                    station_id
                ].discharge_observations_station_coords
                discharge_observations_to_GEB_upstream_area_ratio = (
                    snapped_locations.loc[
                        station_id
                    ].discharge_observations_to_GEB_upstream_area_ratio
                )
                geb_upstream_area_m2: float = float(
                    snapped_locations.at[station_id, "GEB_upstream_area_from_grid"]
                )
                if geb_upstream_area_m2 < minimum_upstream_area_km2 * 1_000_000.0:
                    # Smaller catchments tend to be dominated by local timing and snapping
                    # errors, so the default benchmark excludes them from summary scores.
                    continue
                try:
                    validation_df = create_validation_df(
                        self.model.output_folder,
                        run_name,
                        station_id,
                        observed_discharge_series,
                        correct_discharge_observations,
                        discharge_observations_to_GEB_upstream_area_ratio,
                    )
                except FileNotFoundError:
                    self.model.logger.warning(
                        f"Simulation discharge data for station {station_id} not found. Skipping this station."
                    )
                    continue

                minimum_valid_steps = (
                    minimum_timeseries_length_years
                    * 365.25
                    * (24 if frequency_label == "hourly" else 1)
                )
                if validation_df.dropna().shape[0] < minimum_valid_steps:
                    continue

                discharge_metrics = _calculate_discharge_validation_metrics(
                    validation_df
                )

                if create_plots:
                    _plot_discharge_validation_graphs(
                        model=self.model,
                        station_id=station_id,
                        validation_df=validation_df,
                        station_name=discharge_observations_station_name,
                        upstream_area_ratio=discharge_observations_to_GEB_upstream_area_ratio,
                        kge=discharge_metrics.KGE,
                        nse=discharge_metrics.NSE,
                        r_value=discharge_metrics.R,
                        eval_plot_folder=self.evaluate_discharge_output_folder,
                        include_yearly_plots=include_yearly_plots,
                        frequency=frequency_label,
                    )

                station_evaluation: dict[str, Any] = {
                    "station_ID": station_id,
                    "station_name": discharge_observations_station_name,
                    "x": discharge_observations_station_coords[0],
                    "y": discharge_observations_station_coords[1],
                    "discharge_observations_to_GEB_upstream_area_ratio": discharge_observations_to_GEB_upstream_area_ratio,
                    "upstream_area_GEB": geb_upstream_area_m2,
                    **discharge_metrics._asdict(),
                    **{
                        f"{metric_name}_{frequency_label}": metric_value
                        for metric_name, metric_value in discharge_metrics._asdict().items()
                    },
                }

                # if the frequency is hourly, also calculate the metrics on the daily resampled data
                if frequency_label == "hourly":
                    # Resample to daily, keeping only days with 24 valid hourly observations.
                    valid_hourly_counts_per_day = validation_df.resample("D").count()
                    validation_df_daily = (
                        validation_df.resample("D")
                        .mean()[valid_hourly_counts_per_day == 24]
                        .dropna()
                    )
                    daily_discharge_metrics = _calculate_discharge_validation_metrics(
                        validation_df_daily
                    )
                    station_evaluation.update(
                        {
                            f"{metric_name}_daily": metric_value
                            for metric_name, metric_value in daily_discharge_metrics._asdict().items()
                        }
                    )
                # Daily-frequency stations have no hourly data; fill with NaN so the
                # DataFrame columns remain consistent across all stations.
                elif frequency_label == "daily":
                    station_evaluation.update(
                        {
                            f"{metric_name}_hourly": float("nan")
                            for metric_name in DischargeMetrics._fields
                        }
                    )
                else:
                    raise ValueError(
                        f"Unexpected frequency label '{frequency_label}' in evaluation loop."
                    )

                # Monthly metrics
                validation_df_monthly = validation_df.resample("ME").mean().dropna()
                monthly_discharge_metrics = _calculate_discharge_validation_metrics(
                    validation_df_monthly
                )
                station_evaluation.update(
                    {
                        f"{metric_name}_monthly": metric_value
                        for metric_name, metric_value in monthly_discharge_metrics._asdict().items()
                    }
                )

                # attach to the evaluation dataframe
                evaluation_per_station.append(station_evaluation)

        if len(evaluation_per_station) == 0:
            # Create empty evaluation dataframe with proper structure
            # Column names are derived from DischargeMetrics so they stay in sync.
            freq_cols: list[str] = [
                f"{metric_name}_{frequency}"
                for frequency in ("monthly", "daily", "hourly")
                for metric_name in DischargeMetrics._fields
            ]
            evaluation_df = pd.DataFrame(
                columns=[  # ty:ignore[invalid-argument-type]
                    "station_name",
                    "x",
                    "y",
                    "upstream_area_GEB",
                    "discharge_observations_to_GEB_upstream_area_ratio",
                    *freq_cols,
                    *DischargeMetrics._fields,
                ],
                index=pd.Index([], name="station_ID"),
            )
        else:
            evaluation_df = pd.DataFrame(evaluation_per_station).set_index("station_ID")

        evaluation_df.to_excel(
            self.evaluate_discharge_output_folder / "evaluation_metrics.xlsx",
            index=True,
        )

        # Save evaluation metrics as as excel and parquet file
        evaluation_gdf = gpd.GeoDataFrame(
            evaluation_df,
            geometry=gpd.points_from_xy(evaluation_df.x, evaluation_df.y),
            crs="EPSG:4326",
        )  # create a geodataframe from the evaluation dataframe
        evaluation_gdf.to_parquet(
            self.evaluate_discharge_output_folder / "evaluation_metrics.geoparquet",
        )

        # Return median metrics if available
        if not evaluation_df.empty:
            if create_plots:
                region_geom = read_geom(
                    self.model.files["geom"]["mask"]
                )  # load the region shapefile

                rivers, discharge = self.get_discharge_per_river(
                    run_name
                )  # load in the rivers
                for river_id in discharge.columns:
                    rivers.loc[river_id, "discharge_m3_per_s"] = discharge[
                        river_id
                    ].mean()  # calculate mean per river used to scale the river line widths on the folium map

                create_discharge_folium_map(
                    evaluation_gdf=evaluation_gdf,
                    output_path=self.evaluate_discharge_output_folder
                    / "discharge_evaluation_map.html",
                    timeseries_plot_folder=self.evaluate_discharge_output_folder
                    / "timeseries",
                    return_period_plot_folder=self.evaluate_discharge_output_folder
                    / "return_periods",
                    region_geom=region_geom,
                    rivers=rivers,
                    waterbodies=waterbodies,
                )

                self.model.logger.info("Discharge evaluation dashboard created.")

                # GEB standalone plot — all GEB stations.
                self.plot_skill_score_boxplots(
                    export=True, include_geb=True, matched_only=False
                )
                self.plot_skill_score_maps(export=True)

                # When external evaluation data are present, also produce a
                # combined plot restricted to stations present in both datasets
                # so the comparison is fair.
                if self._read_external_evaluation_raw():
                    self.plot_skill_score_boxplots(
                        export=True, include_geb=True, matched_only=True
                    )

            scores: dict[str, float | None] = {
                **{
                    f"{metric_name}_{frequency}": float(
                        evaluation_df[f"{metric_name}_{frequency}"].median()
                    )
                    for frequency in ("hourly", "daily", "monthly")
                    for metric_name in DischargeMetrics._fields
                },
                **{
                    metric_name: float(evaluation_df[metric_name].median())
                    for metric_name in DischargeMetrics._fields
                },
            }
        else:
            self.model.logger.warning(
                "No discharge stations found for evaluation. Returning None for all metrics."
            )

            scores: dict[str, float | None] = {
                **{
                    f"{metric_name}_{frequency}": None
                    for frequency in ("hourly", "daily", "monthly")
                    for metric_name in DischargeMetrics._fields
                },
                **{metric_name: None for metric_name in DischargeMetrics._fields},
            }

        self.model.logger.info(f"Discharge evaluation completed. Scores: {scores}")

        return scores

    def _read_external_evaluation_raw(
        self, external_evaluation_folder: str | Path | None = None
    ) -> dict[str, pd.DataFrame]:
        """Read all external model evaluation CSVs without filtering to GEB stations.

        Args:
            external_evaluation_folder: Directory with one CSV per external model.
                Defaults to the configured folder, resolved from the model folder
                when relative.

        Returns:
            Mapping from model label to full (unfiltered) DataFrame
            (index = station name, columns = metrics).
        """
        configured_folder = self.model.config["hydrology"]["evaluation"][
            "discharge"
        ].get("external_evaluation_folder")
        folder = Path(
            external_evaluation_folder
            if external_evaluation_folder is not None
            else configured_folder
        )
        if not folder.is_absolute():
            folder = self.model.input_folder.parent / folder
        if not folder.exists():
            self.model.logger.info(
                "No external evaluation data folder found at %s, skipping.", folder
            )
            return {}
        if not folder.is_dir():
            self.model.logger.warning(
                "External evaluation path is not a folder: %s. Skipping.", folder
            )
            return {}

        csv_paths: list[Path] = sorted(folder.glob("*.csv"))
        if not csv_paths:
            self.model.logger.info(
                "No external evaluation CSV files found at %s, skipping.", folder
            )
            return {}

        self.model.logger.info(
            "Reading external evaluation data from %s.",
            folder,
        )

        external_models: dict[str, pd.DataFrame] = {}
        for csv_path in csv_paths:
            external_evaluation_df: pd.DataFrame = pd.read_csv(csv_path, index_col=0)
            external_evaluation_df.index = external_evaluation_df.index.str.upper()
            external_models[csv_path.stem] = external_evaluation_df
        return external_models

    def prepare_external_evaluation(
        self,
        external_evaluation_folder: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        """Filter external model evaluation CSVs to stations present in this model.

        Reads each ``.csv`` in ``external_evaluation_folder`` (one file per
        external model), keeps only rows whose station name matches GEB stations,
        saves filtered results as ``external_evaluation_filtered_<name>.xlsx``,
        and returns them.

        Notes:
            Station names are matched case-insensitively. Falls back to
            ``discharge_snapped_locations.geoparquet`` when
            ``evaluation_metrics.xlsx`` does not yet exist.

        Args:
            external_evaluation_folder: Directory with one CSV per external model
                (station names as index, metrics as columns). Defaults to
                ``external_evaluation_data/`` in the models-root directory.
            **kwargs: Ignored (CLI compatibility).

        Returns:
            Mapping from model label to matched-stations DataFrame.
        """
        external_models = self._read_external_evaluation_raw(external_evaluation_folder)
        if not external_models:
            self.model.logger.info("No external evaluation data found, skipping.")
            return {}

        # Use already-evaluated station list when available; fall back to geom file.
        geb_xlsx = self.evaluate_discharge_output_folder / "evaluation_metrics.xlsx"
        if geb_xlsx.exists():
            our_stations: set[str] = set(
                pd.read_excel(geb_xlsx)["station_name"].dropna().str.upper()
            )
        else:
            snapped = read_geom(
                self.model.files["geom"]["discharge/discharge_snapped_locations"]
            )
            our_stations = set(
                snapped["discharge_observations_station_name"].dropna().str.upper()
            )

        matched_external_models: dict[str, pd.DataFrame] = {}
        for model_name, all_stations_df in external_models.items():
            matched_stations_df: pd.DataFrame = all_stations_df[
                all_stations_df.index.isin(our_stations)
            ].copy()
            self.model.logger.info(
                "External model '%s': %d/%d stations matched.",
                model_name,
                len(matched_stations_df),
                len(all_stations_df),
            )
            matched_stations_df.to_excel(
                self.evaluate_discharge_output_folder
                / f"external_evaluation_filtered_{model_name}.xlsx"
            )
            if not matched_stations_df.empty:
                matched_external_models[model_name] = matched_stations_df

        return matched_external_models

    def plot_skill_score_maps(
        self,
        export: bool = True,
        minimum_upstream_area_km2: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot per-station skill scores on a satellite basemap, one map per metric.

        Notes:
            Requires ``evaluate_discharge`` to have been run first so that
            ``evaluation_metrics.xlsx`` or ``evaluation_metrics.geoparquet`` exist.

        Args:
            export: Whether to save the figures to disk.
            minimum_upstream_area_km2: Optional minimum modeled upstream area threshold for plotted stations (km2).
                If omitted, `hydrology.evaluation.discharge.minimum_upstream_area_km2` is used.
            **kwargs: Ignored (CLI compatibility).
        """
        if minimum_upstream_area_km2 is None:
            minimum_upstream_area_km2 = self.model.config["hydrology"]["evaluation"][
                "discharge"
            ]["minimum_upstream_area_km2"]
        geoparquet_path = (
            self.evaluate_discharge_output_folder / "evaluation_metrics.geoparquet"
        )
        xlsx_path = self.evaluate_discharge_output_folder / "evaluation_metrics.xlsx"

        if geoparquet_path.exists():
            evaluation_gdf: gpd.GeoDataFrame = gpd.read_parquet(geoparquet_path)
        elif xlsx_path.exists():
            eval_df: pd.DataFrame = pd.read_excel(xlsx_path)
            evaluation_gdf = gpd.GeoDataFrame(
                eval_df,
                geometry=gpd.points_from_xy(eval_df["x"], eval_df["y"]),
                crs="EPSG:4326",
            )
        else:
            self.model.logger.warning(
                "No evaluation_metrics file found. Run evaluate_discharge first."
            )
            return

        before_filter_count: int = len(evaluation_gdf)
        evaluation_gdf = gpd.GeoDataFrame(
            evaluation_gdf[
                evaluation_gdf["upstream_area_GEB"]
                >= minimum_upstream_area_km2 * 1_000_000.0
            ].copy(),
            geometry="geometry",
            crs=evaluation_gdf.crs,
        )
        self.model.logger.info(
            "Upstream-area plot filter retained %d/%d stations at %.1f km2 or larger.",
            len(evaluation_gdf),
            before_filter_count,
            minimum_upstream_area_km2,
        )
        if evaluation_gdf.empty:
            self.model.logger.warning(
                "No discharge evaluation stations remain after upstream-area filtering. "
                "Skipping skill score maps."
            )
            return

        if export:
            region_geom: gpd.GeoDataFrame = read_geom(self.model.files["geom"]["mask"])
            _plot_skill_score_maps(
                evaluation_gdf=evaluation_gdf,
                region_geom=region_geom,
                output_folder=self.evaluate_discharge_output_folder,
                logger=self.model.logger,
            )

    def plot_skill_score_boxplots(
        self,
        export: bool = True,
        include_geb: bool = True,
        matched_only: bool = False,
        minimum_upstream_area_km2: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Create skill score violin+boxplot graphs for each evaluation metric.

        Produces a 2×3 grid of violin/box plots across gauging stations.
        When ``matched_only=False`` (default), each model uses its full station
        set. When ``matched_only=True``, both GEB and external data are restricted
        to their overlapping stations for a fair comparison.

        Args:
            export: Save the figure to disk.
            include_geb: Include GEB in the plot.
            matched_only: Restrict all models to overlapping stations only.
            minimum_upstream_area_km2: Optional minimum modeled upstream area threshold for plotted GEB stations (km2).
                If omitted, `hydrology.evaluation.discharge.minimum_upstream_area_km2` is used.
            **kwargs: Ignored (CLI compatibility).
        """
        if minimum_upstream_area_km2 is None:
            minimum_upstream_area_km2 = self.model.config["hydrology"]["evaluation"][
                "discharge"
            ]["minimum_upstream_area_km2"]
        geb_evaluation_xlsx = (
            self.evaluate_discharge_output_folder / "evaluation_metrics.xlsx"
        )
        evaluation_df: pd.DataFrame = (
            pd.read_excel(geb_evaluation_xlsx)
            if geb_evaluation_xlsx.exists()
            else pd.DataFrame()
        )

        if not evaluation_df.empty:
            before_filter_count: int = len(evaluation_df)
            evaluation_df = evaluation_df[
                evaluation_df["upstream_area_GEB"]
                >= minimum_upstream_area_km2 * 1_000_000.0
            ].copy()
            self.model.logger.info(
                "Upstream-area plot filter retained %d/%d GEB stations at %.1f km2 or larger.",
                len(evaluation_df),
                before_filter_count,
                minimum_upstream_area_km2,
            )

        if include_geb and evaluation_df.empty:
            self.model.logger.info(
                "No discharge stations found for evaluation. Skipping skill score graphs."
            )
            return

        if matched_only:
            external_models: dict[str, pd.DataFrame] = (
                self.prepare_external_evaluation()
            )
        else:
            external_models = self._read_external_evaluation_raw()

        if external_models:
            self.model.logger.info("External models in plot: %s", list(external_models))
        else:
            self.model.logger.info("No external models found; showing GEB only.")

        if matched_only and include_geb and external_models and not evaluation_df.empty:
            matched_station_names: set[str] = set()
            for ext_df in external_models.values():
                matched_station_names.update(ext_df.index.str.upper())
            before_n: int = len(evaluation_df)
            evaluation_df = evaluation_df[
                evaluation_df["station_name"].str.upper().isin(matched_station_names)
            ].copy()
            self.model.logger.info(
                "matched_only=True: GEB restricted from %d to %d stations.",
                before_n,
                len(evaluation_df),
            )
            geb_station_names: set[str] = set(
                evaluation_df["station_name"].dropna().str.upper()
            )
            external_models = {
                model_name: model_df[model_df.index.isin(geb_station_names)].copy()
                for model_name, model_df in external_models.items()
            }

        _plot_skill_score_boxplots(
            evaluation_df=evaluation_df,
            external_models=external_models,
            output_folder=self.evaluate_discharge_output_folder,
            logger=self.model.logger,
            export=export,
            include_geb=include_geb,
            matched_only=matched_only,
        )

    def plot_skill_scores_vs_upstream_area(
        self,
        export: bool = True,
        minimum_upstream_area_km2: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot upstream area against discharge skill scores.

        Notes:
            Requires ``evaluate_discharge`` to have been run first so that
            ``evaluation_metrics.xlsx`` exists.

        Args:
            export: Whether to save the figure to disk.
            minimum_upstream_area_km2: Optional minimum modeled upstream area threshold for plotted stations (km2).
                If omitted, `hydrology.evaluation.discharge.minimum_upstream_area_km2` is used.
            **kwargs: Ignored (CLI compatibility).
        """
        if minimum_upstream_area_km2 is None:
            minimum_upstream_area_km2 = self.model.config["hydrology"]["evaluation"][
                "discharge"
            ]["minimum_upstream_area_km2"]
        evaluation_path: Path = (
            self.evaluate_discharge_output_folder / "evaluation_metrics.xlsx"
        )
        if not evaluation_path.exists():
            self.model.logger.warning(
                "No evaluation_metrics.xlsx file found. Run evaluate_discharge first."
            )
            return

        evaluation_df: pd.DataFrame = pd.read_excel(evaluation_path)
        before_filter_count: int = len(evaluation_df)
        evaluation_df = evaluation_df[
            evaluation_df["upstream_area_GEB"]
            >= minimum_upstream_area_km2 * 1_000_000.0
        ].copy()
        self.model.logger.info(
            "Upstream-area plot filter retained %d/%d GEB stations at %.1f km2 or larger.",
            len(evaluation_df),
            before_filter_count,
            minimum_upstream_area_km2,
        )
        if evaluation_df.empty:
            self.model.logger.warning(
                "No discharge evaluation stations remain after upstream-area filtering. "
                "Skipping skill-score upstream-area scatterplot."
            )
            return

        if export:
            _plot_skill_scores_vs_upstream_area(
                evaluation_df=evaluation_df,
                output_folder=self.evaluate_discharge_output_folder,
                logger=self.model.logger,
            )

    def plot_water_circle(
        self,
        run_name: str,
        *args: Any,
        export: bool = True,
        **kwargs: Any,
    ) -> plt.Figure:
        """Create a water circle plot for the GEB model.

        Adapted from: https://github.com/mikhailsmilovic/flowplot
        Also see the paper: https://doi.org/10.1088/1748-9326/ad18de

        Args:
            run_name: Name of the run to evaluate.
            export: Whether to export the water circle plot to a file.
            *args: ignored.
            **kwargs: ignored.

        Returns:
            A matplotlib Figure object representing the water circle.
        """
        folder = self.model.output_folder / "report" / run_name

        def read_parquet_with_date_index(
            folder: Path, module: str, name: str, skip_first_day: bool = True
        ) -> pd.Series:
            """Read a PARQUET file with a date index.

            Args:
                folder: Path to the folder containing the PARQUET file.
                module: Name of the module (subfolder) containing the PARQUET file.
                name: Name of the PARQUET file (without extension).
                skip_first_day: Whether to skip the first day of the time series.

            Returns:
                A pandas Series with the date index and the values from the PARQUET file.

            """
            time_series = pd.read_parquet(
                (folder / module / name).with_suffix(".parquet"),
            )[name]

            if skip_first_day:
                time_series = time_series.iloc[1:]

            return time_series

        # because storage is the storage at the end of the timestep, we need to calculate the change
        # across the entire simulation period. For all other variables we do skip the first day.
        storage = read_parquet_with_date_index(
            folder, "hydrology", "_current_storage", skip_first_day=False
        )
        storage_change = storage.iloc[-1] - storage.iloc[0]

        rain = read_parquet_with_date_index(
            folder, "hydrology.landsurface", "_rain_m"
        ).sum()
        snow = read_parquet_with_date_index(
            folder, "hydrology.landsurface", "_snow_m"
        ).sum()

        domestic_water_loss = read_parquet_with_date_index(
            folder, "hydrology.water_demand", "_domestic_water_loss_m3"
        ).sum()
        industry_water_loss = read_parquet_with_date_index(
            folder, "hydrology.water_demand", "_industry_water_loss_m3"
        ).sum()
        livestock_water_loss = read_parquet_with_date_index(
            folder, "hydrology.water_demand", "_livestock_water_loss_m3"
        ).sum()

        river_outflow = read_parquet_with_date_index(
            folder, "hydrology.routing", "_total_outflow_at_pits_m3"
        ).sum()

        transpiration = read_parquet_with_date_index(
            folder, "hydrology.landsurface", "_transpiration_m"
        ).sum()
        bare_soil_evaporation = read_parquet_with_date_index(
            folder, "hydrology.landsurface", "_bare_soil_evaporation_m"
        ).sum()
        open_water_evaporation = read_parquet_with_date_index(
            folder, "hydrology.landsurface", "_open_water_evaporation_m"
        ).sum()
        interception_evaporation = read_parquet_with_date_index(
            folder, "hydrology.landsurface", "_interception_evaporation_m"
        ).sum()
        sublimation_or_deposition = read_parquet_with_date_index(
            folder, "hydrology.landsurface", "_sublimation_or_deposition_m"
        ).sum()
        river_evaporation = read_parquet_with_date_index(
            folder, "hydrology.routing", "_total_evaporation_in_rivers_m3"
        ).sum()
        waterbody_evaporation = read_parquet_with_date_index(
            folder, "hydrology.routing", "_total_waterbody_evaporation_m3"
        ).sum()

        hierarchy: dict[str, Any] = {
            "in": {
                "rain": rain,
                "snow": snow,
            },
            "out": {
                "evapotranspiration": {
                    "transpiration": transpiration,
                    "bare soil evaporation": bare_soil_evaporation,
                    "open water evaporation": open_water_evaporation,
                    "interception evaporation": interception_evaporation,
                    "river evaporation": river_evaporation,
                    "waterbody evaporation": waterbody_evaporation,
                },
                "river outflow": river_outflow,
                "water demand": {
                    "domestic water loss": domestic_water_loss,
                    "industry water loss": industry_water_loss,
                    "livestock water loss": livestock_water_loss,
                },
            },
            "storage change": abs(storage_change),
        }

        if sublimation_or_deposition > 0:
            hierarchy["in"]["deposition"] = sublimation_or_deposition
        else:
            hierarchy["out"]["evapotranspiration"]["sublimation"] = abs(
                sublimation_or_deposition
            )

        if storage_change > 0:
            order: list[str] = ["in", "out", "storage change"]
        else:
            order: list[str] = ["storage change", "in", "out"]

        hierarchy = {key: hierarchy[key] for key in order}

        water_circle = plot_sunburst(hierarchy, title="water circle")

        if export:
            water_circle.savefig(
                self.water_circle_output_folder / "water_circle.svg",
            )

        return water_circle

<<<<<<< HEAD
    def evaluate_hydrodynamics2(
        self,
        run_name: str = "default",
        forecast_range: tuple[str, str] | None = (
            "20240426T000000",
            "20240502T000000",
        ),
        probability_maps: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Evaluate hydrodynamic model performance against flood observations.

        This method loads modelled flood maps and corresponding observations,
        computes spatial performance metrics (e.g., hit rate, false alarm ratio,
        critical success index), and generates diagnostic visualisations and
        summary outputs for the specified simulation run.
        Notes:
            For probability maps, when multiple zarr files are available, range1 files are
            prioritized for evaluation. Probability maps use exceedance probability thresholds
            (default: 50% probability threshold) to determine flood extent classification.
            Deterministic maps use a 15cm depth threshold for binary flood classification.

        Args:
            run_name: Name of the simulation run to evaluate.
            forecast_range: Optional tuple of (start_date, end_date) strings in format
                'YYYYMMDDTHHMMSS' to limit evaluation to specific forecast initialization
                range. If None, all available forecasts are evaluated.
            probability_maps: Whether to evaluate probability maps (from flood_prob_exceedance_maps
                folder) instead of deterministic maps (from flood_maps folder). When True,
                uses probability threshold-based classification instead of depth thresholds.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Raises:
            FileNotFoundError: If the flood map folder does not exist in the output directory.
            ValueError: If the flood observation file is not in .zarr format.
        """

        def parse_flood_forecast_initialisation(
            filename: str,
        ) -> tuple[str | None, str | None, str, str, str]:
            """Parse flood map filename to extract components.

            Expected format: YYYYMMDDTHHMMSS - MEMBER - EVENT_START - EVENT_END.zarr

            Args:
                filename: Name of the flood map file.

            Returns:
                Tuple containing (forecast_init, member, event_start, event_end, event_name)

            Raises:
                ValueError: If the filename does not match the expected format.

            """
            # Remove .zarr extension
            name_without_ext = filename.replace(".zarr", "")

            # Split by ' - ' to get components
            parts = name_without_ext.split(" - ")

            if len(parts) >= 4:
                # Handle case with forecasts included
                forecast_init = parts[0]  # First 17 characters: YYYYMMDDTHHMMSS
                member = parts[1]  # Member number
                event_start = parts[2]  # Event start time
                event_end = parts[3]  # Event end time

                # Create event name from start and end times
                event_name = f"{event_start} - {event_end}"

            elif len(parts) == 2:
                # Handle case with no forecasts included
                forecast_init = None
                member = None
                event_start = parts[0]  # Event start time
                event_end = parts[1]  # Event end time

                # Create event name from start and end times
                event_name = f"{event_start} - {event_end}"

            else:
                raise ValueError(
                    f"Filename '{filename}' does not match expected flood map format."
                )

            return forecast_init, member, event_start, event_end, event_name

        # Main function for the performance metrics
        def calculate_performance_metrics(
            observation: Path | str,
            flood_map_path: Path | str,
            output_folder: Path,
            visualization_type: str = "Hillshade",
            probability_maps: bool = False,
        ) -> dict[str, float | int] | dict[str, float]:
            """Calculate performance metrics for flood maps against observations.

            Args:
                observation: Path to the observed flood extent data (.zarr format).
                flood_map_path: Path to the model-generated flood map data (.zarr format).
                visualization_type: Type of visualization for plotting (default is "Hillshade").
                output_folder: Path to the folder where results will be saved.

            Returns:
                Dictionary containing performance metrics:
                    - hit_rate: Percentage of correctly predicted flooded areas.
                    - false_alarm_ratio: Percentage of falsely predicted flooded areas.
                    - critical_success_index: Overall accuracy of flood predictions.
                    - flooded_area_km2: Total flooded area in square kilometers.
                or None if an error occurs.

            Raises:
                ValueError: If the observation file is not in .zarr format.
            """
            # Step 1: Open needed datasets
            flood_map = read_zarr(flood_map_path)
            obs_map = read_zarr(observation)
            # Ensure both datasets have same CRS
            if obs_map.rio.crs != flood_map.rio.crs:
                obs_map = obs_map.rio.reproject(flood_map.rio.crs)
            print("obs crs:", obs_map.rio.crs)
            print("flood crs:", flood_map.rio.crs)
            print(
                f"DEBUG obs_map value range: {float(obs_map.min().compute()):.2f} to {float(obs_map.max().compute()):.2f}"
            )
            # Reproject observations to match flood map grid
            obs_map = obs_map.rio.write_nodata(0)
            obs = obs_map.rio.reproject_match(flood_map)
            sim = flood_map
            print(f"DEBUG sim.dims: {sim.dims}")
            print("DEBUG obs.dims:", obs.dims)
            print(f"DEBUG obs (after align).shape: {obs.shape}")
            print(f"DEBUG sim.shape: {sim.shape}")
            print(
                f"DEBUG obs value range: {float(obs.min().compute()):.2f} to {float(obs.max().compute()):.2f}"
            )
            rivers = gpd.read_parquet(
                Path("simulation_root") / run_name / "group_0" / "rivers.geoparquet"
            ).to_crs(obs.rio.crs)
            region_path = Path(self.model.input_folder / "geom" / "mask.geoparquet")
            region = gpd.read_parquet(region_path).to_crs(obs.rio.crs)

            # Step 2: Conditionally clip out rivers from observations and simulations
            # Skip river masking for probability maps as rivers are main flood source
            crs_wgs84 = CRS.from_epsg(4326)
            crs_mercator = CRS.from_epsg(3857)
            gdf_mercator = rivers.to_crs(crs_mercator)
            # Separate rivers with width values from those with NaN width
            rivers_with_width = gdf_mercator[gdf_mercator["width"].notna()].copy()
            rivers_no_width = gdf_mercator[gdf_mercator["width"].isna()].copy()

            print(f"Rivers with width data: {len(rivers_with_width)}")
            print(
                f"Rivers without width data (will mask grid cells only): {len(rivers_no_width)}"
            )

            # For rivers with width: create buffers based on width
            if len(rivers_with_width) > 0:
                rivers_with_width["geometry"] = rivers_with_width.buffer(
                    rivers_with_width["width"] / 2
                )

            # For rivers without width: keep original line geometry (will mask intersected grid cells)
            # No buffering needed - geometry_mask will handle line intersection with grid cells

            # Combine both geometries for final masking
            all_river_geometries = []
            if len(rivers_with_width) > 0:
                all_river_geometries.extend(rivers_with_width.geometry.tolist())
            if len(rivers_no_width) > 0:
                all_river_geometries.extend(rivers_no_width.geometry.tolist())

            # Create a single GeoDataFrame with all river geometries
            gdf_mercator = gpd.GeoDataFrame(
                {"geometry": all_river_geometries}, crs=crs_mercator
            )
            gdf_buffered = gdf_mercator.to_crs(sim.rio.crs)
            river_mask_array = geometry_mask(
                gdf_buffered.geometry,
                out_shape=sim.rio.shape,
                transform=sim.rio.transform(),
                all_touched=True,
                invert=True,  # True = inside rivers (to be masked)
            )
            river_mask_da = xr.DataArray(
                data=river_mask_array,
                coords={"y": sim.y.values, "x": sim.x.values},
                dims=("y", "x"),
            )
            obs_no_rivers = obs.where(~river_mask_da, 0)
            if not probability_maps:
                # Create river mask for simulation data
                sim_no_rivers = sim.where(~river_mask_da, 0)

                print("DEBUG: Applied river masking for flood depth evaluation")
            else:
                # For probability maps, don't mask rivers as they are the main flood source
                sim_no_rivers = sim

                print("DEBUG: Skipped river masking for probability map evaluation")

            # Clip out region from observations and simulations
            obs_region = obs_no_rivers.rio.clip(region.geometry.values, region.crs)

            # Optionally clip using extra validation region from config yml
            extra_validation_path = self.config["floods"].get(
                "extra_validation_region", None
            )

            if extra_validation_path and Path(extra_validation_path).exists():
                extra_clip_region = gpd.read_file(extra_validation_path).set_crs(28992)
                extra_clip_region = extra_clip_region.to_crs(region.crs)
                extra_clip_region_buffer = extra_clip_region.buffer(160)

                sim_extra_clipped = sim_no_rivers.rio.clip(
                    extra_clip_region_buffer.geometry.values,
                    extra_clip_region_buffer.crs,
                )
                clipped_out = (sim_no_rivers > 0.15) & (sim_extra_clipped.isnull())
                clipped_out_raster = sim_no_rivers.where(clipped_out)
            else:
                # If no extra validation region, skip clipping
                sim_extra_clipped = sim_no_rivers
                clipped_out_raster = xr.full_like(sim_no_rivers, np.nan)
            # Mask water depth values and handle probability maps
            hmin: float = self.config["floods"]["minimum_flood_depth"]

            probability_threshold = 0.3  # Default probability threshold
            sim_extra_clipped = sim_extra_clipped.raster.reproject_like(obs_region)

            if probability_maps:
                # For probability maps: use probability threshold to create binary flood map
                # Values >= threshold become 1 (flooded), values < threshold become 0 (not flooded)
                simulation_final = (sim_extra_clipped >= probability_threshold).astype(
                    int
                )

                print(
                    f"Using probability threshold: {probability_threshold} ({probability_threshold * 100:.0f}% chance)"
                )
                print(
                    f"Created binary flood map: 1 where probability >= {probability_threshold}, 0 otherwise"
                )
            else:
                # For deterministic maps: use water depth threshold
                simulation_final = sim_extra_clipped > hmin
                print(f"Using water depth threshold: {hmin} m")

            observation_final = obs_region > 0

            # DEBUG: Create probability map debug plot if probability_maps is True
            if probability_maps and output_folder:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Plot 1: Original probability values
                sim_extra_clipped.plot(
                    ax=axes[0],
                    cmap="Blues",
                    add_colorbar=True,
                    cbar_kwargs={"label": "Flood Probability (0-1)"},
                )
                region.boundary.plot(ax=axes[0], color="red", linewidth=2)
                axes[0].set_title(
                    f"Original Probability Map\n(Before {probability_threshold} threshold)"
                )
                axes[0].set_xlabel("Longitude")
                axes[0].set_ylabel("Latitude")

                # Plot 2: Binary simulation_final after threshold
                simulation_final.plot(
                    ax=axes[1],
                    cmap="Reds",
                    add_colorbar=True,
                    cbar_kwargs={"label": "Binary Flood (0=No, 1=Yes)"},
                )
                region.boundary.plot(ax=axes[1], color="red", linewidth=2)
                axes[1].set_title(
                    f"Binary Flood Map\n(>= {probability_threshold} threshold)"
                )
                axes[1].set_xlabel("Longitude")
                axes[1].set_ylabel("Latitude")

                # Plot 3: Observation for comparison
                observation_final.plot(
                    ax=axes[2],
                    cmap="Greens",
                    add_colorbar=True,
                    cbar_kwargs={"label": "Observed Flood (0=No, 1=Yes)"},
                )
                region.boundary.plot(ax=axes[2], color="red", linewidth=2)
                axes[2].set_title("Observed Flood Extent\n(Reference)")
                axes[2].set_xlabel("Longitude")
                axes[2].set_ylabel("Latitude")

                plt.tight_layout()
                debug_plot_path = (
                    output_folder / "debug_probability_to_binary_conversion.png"
                )
                plt.savefig(debug_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(
                    f"DEBUG: Probability to binary conversion plot saved to: {debug_plot_path}"
                )

            if visualization_type == "OSM":
                observation_final = (
                    observation_final.astype("uint8")
                    .rio.write_nodata(0)
                    .rio.reproject("EPSG:3857")
                )
                region = region.to_crs("EPSG:3857")
                simulation_final = (
                    simulation_final.astype("uint8")
                    .rio.write_nodata(0)
                    .rio.reproject("EPSG:3857")
                )
            xmin, ymin, xmax, ymax = region.total_bounds
            catchment_extent = [xmin, xmax, ymin, ymax]

            xmin, ymin, xmax, ymax = observation_final.rio.bounds()
            flood_extent: tuple[float, float, float, float] = (xmin, xmax, ymin, ymax)

            # Calculate performance metrics
            # Compute the arrays first to get concrete values
            sim_final_computed = simulation_final.compute()
            obs_final_computed = observation_final.compute()

            hit_rate = calculate_hit_rate(sim_final_computed, obs_final_computed) * 100
            false_rate = (
                calculate_false_alarm_ratio(sim_final_computed, obs_final_computed)
                * 100
            )
            csi = (
                calculate_critical_success_index(sim_final_computed, obs_final_computed)
                * 100
            )

            flooded_pixels = float(sim_final_computed.sum().item())

            # Calculate resolution in meters from coordinate spacing
            x_res = float(np.abs(simulation_final.x[1] - simulation_final.x[0]))
            pixel_size = x_res  # meters
            flooded_area_km2 = flooded_pixels * (pixel_size * pixel_size) / 1_000_000

            # Step 7: Save results to file and plot the results
            elevation_data = read_zarr(self.model.files["other"]["DEM/fabdem"])
            elevation_data = elevation_data.rio.reproject_match(obs)

            elevation_array = (
                elevation_data.squeeze().astype("float32").compute().values
            )

            ls = LightSource(azdeg=315, altdeg=45)
            hillshade = ls.hillshade(elevation_array, vert_exag=1, dx=1, dy=1)

            # Catchment borders
            target_crs = region.crs
            catchment_borders = region.boundary

            if simulation_final.sum() > 0:
                misses = (observation_final == 1) & (simulation_final == 0)
                simulation_masked = simulation_final.where(simulation_final == 1)
                hits = simulation_masked.where(observation_final)
                misses_masked = misses.where(misses == 1)

                if visualization_type == "OSM":
                    # Ensure all arrays have the same shape by squeezing extra dimensions
                    simulation_final = simulation_final.squeeze()
                    observation_final = observation_final.squeeze()
                    misses = misses.squeeze()
                    hits = hits.squeeze()
                    simulation_masked = simulation_masked.squeeze()
                    misses_masked = misses_masked.squeeze()
                    margin = 3000
                    fig, ax = plt.subplots(figsize=(10, 10))

                    # clipped_out_raster.plot(
                    #     ax=ax, cmap=blue_cmap, add_colorbar=False, add_labels=False
                    # )

                    ax.imshow(
                        simulation_masked,  # False alarms
                        extent=catchment_extent,
                        # origin="lower",
                        cmap="Wistia",
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        zorder=1,
                    )
                    ax.imshow(
                        misses_masked,  # Misses
                        extent=catchment_extent,
                        # origin="lower",
                        cmap="autumn_r",
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        zorder=2,
                    )

                    ax.imshow(
                        hits,
                        extent=catchment_extent,
                        # origin="lower",
                        cmap="brg",
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        zorder=3,
                    )

                    catchment_borders.plot(
                        ax=ax,
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        zorder=4,
                        alpha=0.5,
                    )
                    ctx.add_basemap(
                        ax,
                        crs=target_crs,
                        source="OpenStreetMap.Mapnik",
                        zoom=12,
                        zorder=0,
                        alpha=0.9,
                    )

                    # Set title based on map type
                    if probability_maps:
                        ax.set_title(
                            "Validation of the Predicted Flood Probabilities",
                            fontsize=14,
                        )
                    else:
                        ax.set_title(
                            "Validation of the Predicted Flood Areas", fontsize=14
                        )
                    ax.set_xlabel("x [m]")
                    ax.set_ylabel("y [m]")

                    # Set the extent based on the raster bounds
                    ax.set_xlim(
                        catchment_extent[0] - margin, catchment_extent[1] + margin
                    )
                    ax.set_ylim(
                        catchment_extent[2] - margin, catchment_extent[3] + margin
                    )
                    ax.set_aspect("equal", adjustable="box")

                    green_patch = mpatches.Patch(color="#94f944", label="Hits")
                    orange_patch = mpatches.Patch(color="orange", label="False Alarms")
                    red_patch = mpatches.Patch(color="red", label="Misses")

                    catchment_patch = Line2D(
                        [0],
                        [0],
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        label="Catchment Border",
                    )
                    legend = ax.legend(
                        handles=[
                            green_patch,
                            orange_patch,
                            red_patch,
                            catchment_patch,
                        ],
                        fontsize=12,
                        loc="upper right",
                    )

                    # Add a comment about the metrics in the plot
                    legend_bbox = legend.get_window_extent(
                        renderer=fig.canvas.get_renderer()  # ty:ignore[unresolved-attribute]
                    )
                    legend_bbox_ax = legend_bbox.transformed(ax.transAxes.inverted())

                    # Add text below legend using axes coordinates
                    ax.annotate(
                        f"Validation Metrics:\n"
                        f"HR    = {hit_rate:.2f} %\n"
                        f"FAR   = {false_rate:.2f} %\n"
                        f"CSI   = {csi:.2f} %",
                        xy=(legend_bbox_ax.x0 + 0.065, legend_bbox_ax.y0 + 0.005),
                        xycoords="axes fraction",
                        fontsize=12,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="grey",
                            boxstyle="round,pad=0.2",
                            alpha=0.8,
                        ),
                        verticalalignment="top",
                        horizontalalignment="left",
                        zorder=5,
                    )

                    crs_text = f"CRS: {target_crs.to_string()}"
                    ax.annotate(
                        crs_text,
                        xy=(0.99, 0.02),  # Bottom right corner in axes coordinates
                        xycoords="axes fraction",
                        fontsize=11,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="grey",
                            boxstyle="round,pad=0.2",
                            alpha=0.8,
                        ),
                        verticalalignment="bottom",
                        horizontalalignment="right",
                        zorder=6,
                    )

                    simulation_filename = os.path.splitext(
                        os.path.basename(flood_map_path)
                    )[0]
                    fig.savefig(
                        output_folder
                        / f"{simulation_filename}_validation_floodextent_plot.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    print(
                        f"Figure with {visualization_type} saved as: {output_folder / f'{simulation_filename}_validation_floodextent_plot.png'}"
                    )

                elif visualization_type == "Hillshade":
                    green_cmap = mcolors.ListedColormap(["green"])  # Hits
                    orange_cmap = mcolors.ListedColormap(["orange"])  # False alarms
                    red_cmap = mcolors.ListedColormap(["red"])  # Misses
                    blue_cmap = mcolors.ListedColormap(
                        ["#72c1db"]
                    )  # No observation data

                    fig, ax = plt.subplots(figsize=(10, 10))

                    ax.imshow(
                        hillshade,
                        cmap="gray",
                        extent=(
                            elevation_data.x.min(),
                            elevation_data.x.max(),
                            elevation_data.y.min(),
                            elevation_data.y.max(),
                        ),
                    )

                    region.boundary.plot(
                        ax=ax, edgecolor="black", linewidth=2, label="Region Boundary"
                    )

                    clipped_out_raster.plot(
                        ax=ax, cmap=blue_cmap, add_colorbar=False, add_labels=False
                    )
                    simulation_masked.plot(
                        ax=ax, cmap=orange_cmap, add_colorbar=False, add_labels=False
                    )
                    hits.plot(
                        ax=ax, cmap=green_cmap, add_colorbar=False, add_labels=False
                    )
                    misses_masked.plot(
                        ax=ax, cmap=red_cmap, add_colorbar=False, add_labels=False
                    )

                    ax.set_aspect("equal")
                    ax.axis("off")
                    handles = [
                        mpatches.Patch(color=green_cmap(0.5)),
                        mpatches.Patch(color=red_cmap(0.5)),
                        mpatches.Patch(color=orange_cmap(0.5)),
                        mpatches.Patch(color=blue_cmap(0.5)),
                        mlines.Line2D([], [], color="black", linewidth=2),
                    ]

                    labels = [
                        "Hits",
                        "Misses",
                        "False alarms",
                        "No observation data",
                        "Region",
                    ]

                    plt.legend(
                        handles=handles, labels=labels, loc="upper right", fontsize=16
                    )

                    simulation_filename = os.path.splitext(
                        os.path.basename(flood_map_path)
                    )[0]
                    plt.savefig(
                        output_folder
                        / f"{simulation_filename}_validation_floodextent_plot.png"
                    )
                    print(
                        f"Figure with {visualization_type} saved as: {output_folder / f'{simulation_filename}_validation_floodextent_plot.png'}"
                    )

                else:
                    raise ValueError(
                        f"Unknown visualization type: {visualization_type}, choose either 'OSM' or 'Hillshade'."
                    )

                performance_numbers = (
                    output_folder / f"{simulation_filename}_performance_metrics.txt"
                )

                with open(performance_numbers, "w") as f:
                    f.write(f"Hit rate (H): {hit_rate}\n")
                    f.write(f"False alarm rate (F): {false_rate}\n")
                    f.write(f"Critical Success Index (CSI) (C): {csi}\n")
                    f.write(f"Number of flooded pixels: {flooded_pixels}\n")
                    f.write(f"Flooded area (km2): {flooded_area_km2}")

                # Return metrics for further analysis
                return {
                    "hit_rate": hit_rate,
                    "false_alarm_rate": false_rate,
                    "csi": csi,
                    "flooded_area_km2": flooded_area_km2,
                }

        def create_forecast_performance_plots(
            performance_df: pd.DataFrame, event_name: str, output_folder: Path
        ) -> None:
            """Create performance metric plots showing spread and mean across forecast initializations.

            Generates line plots showing how performance metrics vary across different
            forecast initialization times for a given flood event. Shows both the spread
            of the ensemble (min-max range) with transparent fill and the ensemble mean as a black line.

            Notes:
                The spread is calculated using the minimum and maximum values across ensemble
                members for each forecast initialization. If only one member exists for a
                forecast initialization, no spread is shown for that time point.

            Args:
                performance_df: DataFrame containing performance metrics with forecast
                    initialization times and ensemble members. Must include columns:
                    'forecast_init', 'hit_rate', 'false_alarm_rate', 'csi', 'flooded_area_km2'.
                event_name: Name of the flood event being analyzed.
                output_folder: Directory to save the performance plots.

            Raises:
                ValueError: If required columns are missing from performance_df.
            """
            # Validate required columns
            required_columns = [
                "forecast_init",
                "hit_rate",
                "false_alarm_rate",
                "csi",
                "flooded_area_km2",
            ]
            missing_columns = [
                col for col in required_columns if col not in performance_df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in performance_df: {missing_columns}"
                )

            # Group by forecast initialization and calculate statistics
            grouped_stats = (
                performance_df.groupby("forecast_init")[
                    ["hit_rate", "false_alarm_rate", "csi", "flooded_area_km2"]
                ]
                .agg(["mean", "min", "max", "std", "count"])
                .reset_index()
            )

            # Convert forecast_init to datetime for better plotting
            grouped_stats["forecast_init_dt"] = pd.to_datetime(
                grouped_stats["forecast_init"], format="%Y%m%dT%H%M%S"
            )

            # Sort by datetime for proper line plotting
            grouped_stats = grouped_stats.sort_values("forecast_init_dt")

            # Create subplots for different metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                f"Forecast Performance Metrics with Ensemble Spread - {event_name}",
                fontsize=16,
                fontweight="bold",
            )

            # Define colors for spread (transparent) and metric-specific colors
            spread_colors = {
                "hit_rate": "#2E86AB",
                "false_alarm_rate": "#A23B72",
                "csi": "#F18F01",
                "flooded_area_km2": "#636EFA",
            }

            def plot_ensemble_density(
                ax: plt.Axes,
                metric: str,
                forecast_times: pd.Series,
                performance_df: pd.DataFrame,
            ) -> None:
                """Plot ensemble density for a given metric across forecast times."""
                import numpy as np
                from scipy.stats import gaussian_kde

                for i, forecast_time in enumerate(forecast_times):
                    # Get ensemble data for this forecast time
                    ensemble_data = performance_df[
                        performance_df["forecast_init"] == forecast_time
                    ][metric].values

                    if len(ensemble_data) > 1:  # Need at least 2 points for KDE
                        try:
                            # Create KDE
                            kde = gaussian_kde(ensemble_data)

                            # Create evaluation points
                            y_min, y_max = ensemble_data.min(), ensemble_data.max()
                            y_range = y_max - y_min
                            if y_range > 0:
                                y_eval = np.linspace(
                                    y_min - 0.1 * y_range, y_max + 0.1 * y_range, 50
                                )
                                density = kde(y_eval)

                                # Normalize density to reasonable width
                                density_normalized = density / density.max() * 0.8

                                # Convert forecast time to x-position
                                x_pos = pd.to_datetime(
                                    forecast_time, format="%Y%m%dT%H%M%S"
                                )

                                # Plot density as filled curve
                                ax.fill_betweenx(
                                    y_eval,
                                    x_pos - pd.Timedelta(hours=6),
                                    x_pos + pd.Timedelta(hours=6) * density_normalized,
                                    alpha=0.4,
                                    color=spread_colors[metric],
                                    label="Ensemble Density" if i == 0 else "",
                                )

                        except Exception:
                            # Fallback to simple scatter if KDE fails
                            x_pos = pd.to_datetime(
                                forecast_time, format="%Y%m%dT%H%M%S"
                            )
                            ax.scatter(
                                [x_pos] * len(ensemble_data),
                                ensemble_data,
                                alpha=0.3,
                                color=spread_colors[metric],
                                s=20,
                            )

            # Hit Rate
            ax = axes[0, 0]
            metric = "hit_rate"

            # Plot ensemble density distribution
            plot_ensemble_density(
                ax, metric, grouped_stats["forecast_init"], performance_df
            )

            # Plot mean as black line
            ax.plot(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "mean")],
                color="black",
                linewidth=2,
                marker="o",
                markersize=6,
                label="Ensemble Mean",
            )

            # Plot control forecast (member 0) with dotted line
            member_0_data = performance_df[performance_df["member"] == 0]
            if not member_0_data.empty:
                member_0_grouped = (
                    member_0_data.groupby("forecast_init")[metric].mean().reset_index()
                )
                member_0_grouped["forecast_init_dt"] = pd.to_datetime(
                    member_0_grouped["forecast_init"], format="%Y%m%dT%H%M%S"
                )
                member_0_grouped = member_0_grouped.sort_values("forecast_init_dt")
                ax.plot(
                    member_0_grouped["forecast_init_dt"],
                    member_0_grouped[metric],
                    color="grey",
                    linewidth=2.5,
                    linestyle="--",
                    marker="o",
                    markersize=7,
                    label="Control Forecast",
                )

            ax.set_title("Hit Rate (%)", fontweight="bold")
            ax.set_ylabel("Hit Rate (%)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.legend()

            # False Alarm Rate
            ax = axes[0, 1]
            metric = "false_alarm_rate"

            # Plot ensemble density distribution
            plot_ensemble_density(
                ax, metric, grouped_stats["forecast_init"], performance_df
            )

            ax.plot(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "mean")],
                color="black",
                linewidth=2,
                marker="s",
                markersize=6,
                label="Ensemble Mean",
            )

            # Plot control forecast (member 0) with dotted line
            member_0_data = performance_df[performance_df["member"] == 0]
            if not member_0_data.empty:
                member_0_grouped = (
                    member_0_data.groupby("forecast_init")[metric].mean().reset_index()
                )
                member_0_grouped["forecast_init_dt"] = pd.to_datetime(
                    member_0_grouped["forecast_init"], format="%Y%m%dT%H%M%S"
                )
                member_0_grouped = member_0_grouped.sort_values("forecast_init_dt")
                ax.plot(
                    member_0_grouped["forecast_init_dt"],
                    member_0_grouped[metric],
                    color="grey",
                    linewidth=2.5,
                    linestyle="--",
                    marker="s",
                    markersize=7,
                    label="Control Forecast",
                )

            ax.set_title("False Alarm Rate (%)", fontweight="bold")
            ax.set_ylabel("False Alarm Rate (%)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.legend()

            # Critical Success Index
            ax = axes[1, 0]
            metric = "csi"

            # Plot ensemble density distribution
            plot_ensemble_density(
                ax, metric, grouped_stats["forecast_init"], performance_df
            )

            ax.plot(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "mean")],
                color="black",
                linewidth=2,
                marker="^",
                markersize=6,
                label="Ensemble Mean",
            )

            # Plot control forecast (member 0) with dotted line
            member_0_data = performance_df[performance_df["member"] == 0]
            if not member_0_data.empty:
                member_0_grouped = (
                    member_0_data.groupby("forecast_init")[metric].mean().reset_index()
                )
                member_0_grouped["forecast_init_dt"] = pd.to_datetime(
                    member_0_grouped["forecast_init"], format="%Y%m%dT%H%M%S"
                )
                member_0_grouped = member_0_grouped.sort_values("forecast_init_dt")
                ax.plot(
                    member_0_grouped["forecast_init_dt"],
                    member_0_grouped[metric],
                    color="grey",
                    linewidth=2.5,
                    linestyle="--",
                    marker="^",
                    markersize=7,
                    label="Control Forecast",
                )

            ax.set_title("Critical Success Index (%)", fontweight="bold")
            ax.set_ylabel("CSI (%)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.legend()

            # Flooded Area
            ax = axes[1, 1]
            metric = "flooded_area_km2"

            # Plot ensemble density distribution
            plot_ensemble_density(
                ax, metric, grouped_stats["forecast_init"], performance_df
            )

            ax.plot(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "mean")],
                color="black",
                linewidth=2,
                marker="d",
                markersize=6,
                label="Ensemble Mean",
            )

            # Plot control forecast (member 0) with dotted line
            member_0_data = performance_df[performance_df["member"] == 0]
            if not member_0_data.empty:
                member_0_grouped = (
                    member_0_data.groupby("forecast_init")[metric].mean().reset_index()
                )
                member_0_grouped["forecast_init_dt"] = pd.to_datetime(
                    member_0_grouped["forecast_init"], format="%Y%m%dT%H%M%S"
                )
                member_0_grouped = member_0_grouped.sort_values("forecast_init_dt")
                ax.plot(
                    member_0_grouped["forecast_init_dt"],
                    member_0_grouped[metric],
                    color="grey",
                    linewidth=2.5,
                    linestyle="--",
                    marker="d",
                    markersize=7,
                    label="Control Forecast",
                )

            ax.set_title("Flooded Area (km²)", fontweight="bold")
            ax.set_ylabel("Area (km²)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Format x-axis for all subplots
            for ax in axes.flat:
                ax.tick_params(axis="x", rotation=45)
                ax.set_xlabel("Forecast Initialization Time")

            plt.tight_layout()

            # Save the plot
            plot_filename = (
                f"{event_name.replace(':', '_')}_forecast_performance_spread.png"
            )
            fig.savefig(output_folder / plot_filename, dpi=300, bbox_inches="tight")
            print(
                f"Forecast performance spread plot saved as: {output_folder / plot_filename}"
            )

        def find_exact_observation_file(
            event_name: str, files: list[Path]
        ) -> Path | None:
            """Find the matching observation file for a flood event.

            The observation files must be named exactly using the event's
            start and end times (e.g., `20210712T090000 - 20210720T090000.zarr`).
            Matching is done by comparing the filename stem (without extension)
            to the event_name.

            Args:
                event_name: The event identifier in the format
                    "YYYYMMDDTHHMMSS - YYYYMMDDTHHMMSS".
                files: List of file paths to available observation files.

            Returns:
                Path | None: The matching observation file if found, otherwise None.
            """
            for f in files:
                if f.stem == event_name:
                    return f
            return None

        self.config = self.model.config["hazards"]

        eval_hydrodynamics_folders = (
            Path(self.evaluator.output_folder_evaluate) / "hydrodynamics"
        )

        eval_hydrodynamics_folders.mkdir(parents=True, exist_ok=True)

        # Calculate performance metrics for every event in the config file
        for event in self.config["floods"]["events"]:
            event_name = f"{event['start_time'].strftime('%Y%m%dT%H%M%S')} - {event['end_time'].strftime('%Y%m%dT%H%M%S')}"
            print(f"event: {event_name}")

            # Determine which type of flood maps to use based on probability_maps parameter
            if probability_maps:
                flood_maps_folder = (
                    self.model.output_folder / "flood_prob_exceedance_maps"
                )
                map_type_description = "probability flood maps"
            else:
                flood_maps_folder = self.model.output_folder / "flood_maps"
                map_type_description = "deterministic flood maps"

            # Create event-specific folder (default). If forecasts are used,
            # forecast-specific subfolders will be created later. Ensure
            # `event_folder` is always defined to avoid UnboundLocalError.
            event_folder = eval_hydrodynamics_folders / event_name
            event_folder.mkdir(parents=True, exist_ok=True)

            # check if flood map folder exists
            if not flood_maps_folder.exists():
                if probability_maps:
                    raise FileNotFoundError(
                        f"Probability flood map folder does not exist in the output directory: {flood_maps_folder}. "
                        "Make sure probability maps were generated during model run."
                    )
                else:
                    raise FileNotFoundError(
                        f"Deterministic flood map folder does not exist in the output directory: {flood_maps_folder}. "
                        "Did you run the hydrodynamic model?"
                    )

            # Extract the observation files, find the match with the flood event
            obs_raw = self.config["floods"]["observation_files"]
            if isinstance(obs_raw, str):
                observation_files = [Path(obs_raw)]
            else:
                observation_files = [Path(p) for p in obs_raw]
            obs_file = find_exact_observation_file(event_name, observation_files)

            # check if observation file exists
            if obs_file is None:
                print(
                    f"No observation file for this event: '{event_name}'. Skipping event."
                )
                continue
            if not obs_file.exists():
                raise FileNotFoundError(
                    "Flood observation file is not found in the given path in the model.yml Please check the path in the config file."
                )
            if obs_file.suffix != ".zarr":
                raise ValueError(
                    "Flood observation file is not in the correct format. Please provide a .zarr file."
                    "Flood observation file is not found in the given path in the model.yml. "
                    "Please check the path in the config file."
                )

            if not self.model.config["general"]["forecasts"]["use"]:
                print(
                    "Forecasts use is set to false in the config, so no forecasts are included in the evaluation."
                )
                # Find all flood maps corresponding to the event
                all_flood_map_files = list(flood_maps_folder.glob("*.zarr"))

                # Filter flood_map_files for the current event only
                flood_map_files = []
                for flood_map_path in all_flood_map_files:
                    parsed = parse_flood_forecast_initialisation(flood_map_path.name)

                    # Skip files that do not match the expected format
                    if parsed is None:
                        continue

                    file_forecast_init, _, _, _, parsed_event_name = parsed
                    # Check if file matches current event
                    if parsed_event_name == event_name:
                        flood_map_files.append(flood_map_path)

                print(
                    f"Found {len(flood_map_files)} flood map files for event {event_name}"
                )

                if len(flood_map_files) == 1:
                    print(
                        "Only one flood map found, assuming no forecasts were included in the simulation."
                    )
                    flood_map_name = event_name + ".zarr"
                    flood_map_path = flood_maps_folder / flood_map_name

                    metrics = calculate_performance_metrics(
                        observation=str(obs_file),
                        flood_map_path=flood_map_path,
                        output_folder=event_folder,
                        probability_maps=False,
                        visualization_type="OSM",
                    )
                    print(f"Successfully evaluated: {flood_map_path.name}")

            else:
                print(f"Evaluating flood forecasts using {map_type_description}...")

                forecast_folders = sorted(flood_maps_folder.glob("forecast_*"))
                initialization_dates = [
                    f.name.split("_", 1)[1] for f in forecast_folders
                ]

                # Filter initialization dates by forecast_range if provided
                if forecast_range is not None:
                    start_date, end_date = forecast_range
                    # Validate forecast_range format
                    try:
                        datetime.strptime(start_date, "%Y%m%dT%H%M%S")
                        datetime.strptime(end_date, "%Y%m%dT%H%M%S")
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid forecast_range format. Expected 'YYYYMMDDTHHMMSS', got: {e}"
                        ) from e

                    # Filter dates within range
                    filtered_dates = [
                        date
                        for date in initialization_dates
                        if start_date <= date <= end_date
                    ]

                    if not filtered_dates:
                        print(f"No forecasts found in range {start_date} to {end_date}")
                        print(f"Available forecast dates: {initialization_dates}")
                        continue

                    initialization_dates = filtered_dates
                    print(
                        f"Filtered to {len(initialization_dates)} forecasts in range {start_date} to {end_date}"
                    )

                # get number of members from the first forecast folder (limit to member_1 to member_51)
                first_forecast_folder = (
                    flood_maps_folder / f"forecast_{initialization_dates[0]}"
                )
                if not first_forecast_folder.exists():
                    raise FileNotFoundError(
                        f"First forecast folder does not exist: {first_forecast_folder}"
                    )

                if probability_maps:
                    # For probability maps: no member subfolders, zarr files directly in forecast folder
                    n_ensemble_members = 0  # Only one probability map per forecast
                    print(
                        "Probability maps: no ensemble members, single probability map per forecast"
                    )
                else:
                    # For deterministic maps: count member subfolders (limit to member_1 to member_51)
                    n_ensemble_members = min(
                        50, sum(1 for _ in first_forecast_folder.glob("member_*"))
                    )
                    print(
                        f"Deterministic maps: found {n_ensemble_members + 1} ensemble members (including control)"
                    )

                forecast_metrics_list = []
                performance_metrics_list = []
                for init_date_str in initialization_dates:
                    print(f" Processing forecast initialization: {init_date_str}")

                    if probability_maps:
                        # For probability maps: single zarr file directly in forecast folder
                        forecast_folder = (
                            flood_maps_folder / f"forecast_{init_date_str}"
                        )
                        zarr_files = list(forecast_folder.glob("*.zarr"))

                        if not zarr_files:
                            raise FileNotFoundError(
                                f"No zarr files found in {forecast_folder}"
                            )
                        elif len(zarr_files) > 1:
                            # Prioritize range1 zarr files if available
                            range1_files = [f for f in zarr_files if "range1" in f.name]
                            if range1_files:
                                flood_map_path = range1_files[0]
                            else:
                                raise FileNotFoundError(
                                    f"No range1 zarr file found in {forecast_folder}"
                                )
                        else:
                            if "range1" in zarr_files[0].name:
                                flood_map_path = zarr_files[0]
                            else:
                                raise FileNotFoundError(
                                    f"No range1 zarr file found in {forecast_folder}"
                                )
                            flood_map_path = zarr_files[0]

                        forecast_event_name = (
                            flood_map_path.stem
                        )  # removes .zarr extension

                        event_folder = (
                            eval_hydrodynamics_folders
                            / "forecasts"
                            / f"forecast_{init_date_str}"
                            / "probability_map"
                        )
                        event_folder.mkdir(parents=True, exist_ok=True)

                        metrics = calculate_performance_metrics(
                            observation=str(obs_file),
                            flood_map_path=flood_map_path,
                            visualization_type="OSM",
                            output_folder=event_folder,
                            probability_maps=probability_maps,
                        )
                        print("   Probability flood map evaluation complete.")

                        metrics_with_metadata = {
                            "forecast_init": init_date_str,
                            "member": "probability",  # Indicate this is a probability map
                            "filename": flood_map_path.name,
                            **metrics,
                        }

                        performance_metrics_list.append(metrics_with_metadata)
                        forecast_metrics_list.append(metrics)
                        print(f"   Successfully evaluated: {flood_map_path.name}")

                    else:
                        # For deterministic maps: loop through member subfolders
                        for member in range(0, n_ensemble_members + 1):
                            member_folder = (
                                flood_maps_folder
                                / f"forecast_{init_date_str}"
                                / f"member_{member}"
                            )

                            # Dynamically find the zarr file in the member folder
                            zarr_files = list(member_folder.glob("*.zarr"))
                            if not zarr_files:
                                raise FileNotFoundError(
                                    f"No zarr files found in {member_folder}"
                                )
                            elif len(zarr_files) > 1:
                                raise FileExistsError(
                                    f"Multiple zarr files found in {member_folder}: {[f.name for f in zarr_files]}"
                                )

                            flood_map_path = zarr_files[0]
                            # Create event name from the zarr filename for consistency
                            forecast_event_name = (
                                flood_map_path.stem
                            )  # removes .zarr extension

                            event_folder = (
                                eval_hydrodynamics_folders
                                / "forecasts"
                                / f"forecast_{init_date_str}"
                                / f"member_{member}"
                            )
                            event_folder.mkdir(parents=True, exist_ok=True)

                            metrics = calculate_performance_metrics(
                                observation=str(obs_file),
                                flood_map_path=flood_map_path,
                                visualization_type="OSM",
                                output_folder=event_folder,
                                probability_maps=probability_maps,
                            )
                            print("   Flood map evaluation complete.")

                            metrics_with_metadata = {
                                "forecast_init": init_date_str,
                                "member": member,
                                "filename": flood_map_path.name,
                                **metrics,
                            }

                            performance_metrics_list.append(metrics_with_metadata)
                            forecast_metrics_list.append(metrics)
                            print(f"   Successfully evaluated: {flood_map_path.name}")

                if performance_metrics_list:
                    performance_df = pd.DataFrame(performance_metrics_list)

                    # Create forecast performance plots
                    create_forecast_performance_plots(
                        performance_df, forecast_event_name, event_folder
                    )

                    # Save detailed performance metrics
                    detailed_filename = f"{forecast_event_name.replace(' - ', '_')}_detailed_performance_metrics.csv"
                    performance_df.to_csv(event_folder / detailed_filename, index=False)
                    print(
                        f"Detailed performance metrics saved as: {event_folder / detailed_filename}"
                    )

            print(f"Completed processing event: {event_name}\n")

        print("Flood map performance metrics calculated for all events.")

    def water_balance(
        self,
        run_name: str,
        include_spinup: bool,
        spinup_name: str,
        *args: Any,
        export: bool = True,
        **kwargs: Any,
=======
    def plot_water_balance(
        self,
        run_name: str,
        export: bool = True,
>>>>>>> cf97c6b046a395cd4914d7d5fbd13329e71ca3ac
    ) -> None:
        """Create a csv file and plot showing the water balance components.

        Args:
            run_name: Name of the run to evaluate.
<<<<<<< HEAD
            include_spinup: Whether to include the spinup run in the evaluation.
            spinup_name: Name of the spinup run to include in the evaluation.
            export: Whether to export the water balance plot to a file.
            *args: ignored.
            **kwargs: ignored.
        """
        folder = self.model.output_folder / "report" / run_name

        def read_csv_with_date_index(
            folder: Path,
            module: str,
            name: str,
        ) -> pd.Series:
            """Read a CSV file with a date index.

            Args:
                folder: Path to the folder containing the CSV file.
                module: Name of the module (subfolder) containing the CSV file.
                name: Name of the CSV file (without extension).

            Returns:
                A pandas Series with the date index and the values from the CSV file.

            """
            df = pd.read_csv(
                (folder / module / name).with_suffix(".csv"),
                index_col=0,
                parse_dates=True,
            )[name]

            return df

        # because storage is the storage at the end of the timestep, we need to calculate the change
        # across the entire simulation period.
        storage = read_csv_with_date_index(
            folder, "hydrology", "_water_balance_storage"
        )
        storage_change = storage.iloc[-1] - storage.iloc[0]

        rain = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_balance_rain"
        )
        snow = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_balance_snow"
        )

        domestic_water_loss = read_csv_with_date_index(
            folder, "hydrology.water_demand", "_water_balance_domestic_water_loss"
        )
        industry_water_loss = read_csv_with_date_index(
            folder, "hydrology.water_demand", "_water_balance_industry_water_loss"
        )
        livestock_water_loss = read_csv_with_date_index(
            folder, "hydrology.water_demand", "_water_balance_livestock_water_loss"
        )

        river_outflow = read_csv_with_date_index(
            folder, "hydrology.routing", "_water_balance_river_outflow"
        )

        transpiration = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_balance_transpiration"
        )
        bare_soil_evaporation = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_balance_bare_soil_evaporation"
        )
        open_water_evaporation = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_balance_open_water_evaporation"
        )
        interception_evaporation = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_balance_interception_evaporation"
        )
        sublimation_or_deposition = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_balance_sublimation_or_deposition"
        )
        river_evaporation = read_csv_with_date_index(
            folder, "hydrology.routing", "_water_balance_river_evaporation"
        )
        waterbody_evaporation = read_csv_with_date_index(
            folder, "hydrology.routing", "_water_balance_waterbody_evaporation"
        )

        hierarchy: dict[str, Any] = {
            "in": {
                "rain": rain,
                "snow": snow,
            },
            "out": {
                "evapotranspiration": {
                    "transpiration": transpiration,
                    "bare soil evaporation": bare_soil_evaporation,
                    "open water evaporation": open_water_evaporation,
                    "interception evaporation": interception_evaporation,
                    "river evaporation": river_evaporation,
                    "waterbody evaporation": waterbody_evaporation,
                },
                "water demand": {
                    "domestic water loss": domestic_water_loss,
                    "industry water loss": industry_water_loss,
                    "livestock water loss": livestock_water_loss,
                },
                "river outflow": river_outflow,
            },
            "storage change": abs(storage_change),
        }

        if sublimation_or_deposition.sum() > 0:
            hierarchy["in"]["deposition"] = sublimation_or_deposition
        else:
            hierarchy["out"]["evapotranspiration"]["sublimation"] = abs(
                sublimation_or_deposition
            )

        storage_delta = storage.diff().fillna(
            0
        )  # Convert storage change into a Series so it appears in yearly results

        # Replace scalar in hierarchy
        hierarchy["storage change"] = storage_delta

        flat: dict[str, pd.Series] = {}

        def flatten(prefix: str, obj: dict[str, Any]) -> None:
            for k, v in obj.items():
                name = f"{prefix}_{k}" if prefix else k
                if isinstance(v, dict):
                    flatten(name, v)
                elif isinstance(v, pd.Series):
                    flat[name] = v
                else:
                    pass

        flatten("", hierarchy)

        df = pd.DataFrame(flat)
        df_yearly = df.resample("Y").sum()
        df_yearly.to_csv(folder / "water_balance_yearly.csv")
        print("Water balance yearly values saved.")

        years = df_yearly.index.year
        n_years = len(years)
=======
            export: Whether to export the water balance plot to a file.

        Notes:
            Potential evapotranspiration is shown as an optional context bar when
            the corresponding report output is available. It is not included in
            the actual water balance totals.

        Raises:
            ValueError: If the water balance dataframe does not contain any rows.
        """
        folder = self.model.output_folder / "report" / run_name
        df_m3_per_timestep: pd.DataFrame = _load_water_balance_dataframe(folder)
        context_series: dict[str, pd.Series] = _load_contextual_water_balance_series(
            folder
        )
        df_yearly: pd.DataFrame = df_m3_per_timestep.resample("YE").sum()
        df_yearly.to_csv(folder / "water_balance_yearly.csv")
        self.model.logger.info("Water balance yearly values saved.")

        years: pd.Index = df_yearly.index.year  # ty:ignore[unresolved-attribute]
        n_years: int = len(years)
>>>>>>> cf97c6b046a395cd4914d7d5fbd13329e71ca3ac

        fig, axes = plt.subplots(n_years, 1, figsize=(16, 4 * n_years), sharex=True)
        if n_years == 1:
            axes = [axes]

        inputs_cols = [c for c in df_yearly.columns if c.startswith("in_")]
        outputs_cols = [c for c in df_yearly.columns if c.startswith("out_")]
        storage_cols = [c for c in df_yearly.columns if "storage" in c.lower()]
<<<<<<< HEAD
=======
        yearly_context_series: dict[str, pd.Series] = {
            series_name: series.resample("YE").sum()
            for series_name, series in context_series.items()
        }
>>>>>>> cf97c6b046a395cd4914d7d5fbd13329e71ca3ac

        # legend building
        legend_handles = []
        legend_labels = []

        # Colormaps
<<<<<<< HEAD
        input_cmap = get_cmap("Blues")
        output_cmap = get_cmap("Set3")
        storage_cmap = get_cmap("Greens")
=======
        input_cmap = mcolormaps["Blues"]
        output_cmap = mcolormaps["Set3"]
        storage_cmap = mcolormaps["Greens"]
>>>>>>> cf97c6b046a395cd4914d7d5fbd13329e71ca3ac

        # Assign distinct colors per column
        input_colors = {
            col: input_cmap(0.4 + 0.5 * i / max(1, len(inputs_cols) - 1))
            for i, col in enumerate(inputs_cols)
        }

        output_colors = {
            col: output_cmap(i % output_cmap.N) for i, col in enumerate(outputs_cols)
        }

        storage_colors = {
            col: storage_cmap(0.5 + 0.4 * i / max(1, len(storage_cols) - 1))
            for i, col in enumerate(storage_cols)
        }

        def add_legend_entry(handle: Any, label: str) -> None:
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)

        for ax, year in zip(axes, years):
<<<<<<< HEAD
            row = df_yearly.loc[df_yearly.index.year == year].iloc[0]
=======
            row = df_yearly.loc[df_yearly.index.year == year].iloc[0]  # ty:ignore[unresolved-attribute]
>>>>>>> cf97c6b046a395cd4914d7d5fbd13329e71ca3ac

            bottom = 0
            for col in inputs_cols:
                label = col.replace("in_", "").replace("_", " ")
<<<<<<< HEAD
                h = ax.bar(
=======
                bar_container = ax.bar(
>>>>>>> cf97c6b046a395cd4914d7d5fbd13329e71ca3ac
                    "inputs",
                    row[col],
                    bottom=bottom,
                    color=input_colors[col],
                )
<<<<<<< HEAD
                add_legend_entry(h[0], f"input • {label}")
=======
                add_legend_entry(bar_container[0], f"input • {label}")
>>>>>>> cf97c6b046a395cd4914d7d5fbd13329e71ca3ac
                bottom += row[col]

            bottom = 0
            for col in outputs_cols:
                label = col.replace("out_", "").replace("_", " ")
<<<<<<< HEAD
                h = ax.bar(
=======
                bar_container = ax.bar(
>>>>>>> cf97c6b046a395cd4914d7d5fbd13329e71ca3ac
                    "outputs",
                    row[col],
                    bottom=bottom,
                    color=output_colors[col],
                )
<<<<<<< HEAD
                add_legend_entry(h[0], f"output • {label}")
=======
                add_legend_entry(bar_container[0], f"output • {label}")
>>>>>>> cf97c6b046a395cd4914d7d5fbd13329e71ca3ac
                bottom += row[col]

            for col in storage_cols:
                label = col.replace("_", " ")
<<<<<<< HEAD
                h = ax.bar(
=======
                bar_container = ax.bar(
>>>>>>> cf97c6b046a395cd4914d7d5fbd13329e71ca3ac
                    "storage",
                    row[col],
                    color=storage_colors[col],
                )
<<<<<<< HEAD
                add_legend_entry(h[0], label)
=======
                add_legend_entry(bar_container[0], label)

            for series_name, yearly_series in yearly_context_series.items():
                label = _format_water_balance_context_label(series_name)
                yearly_context_positions: list[int] = [
                    position
                    for position, timestamp in enumerate(yearly_series.index)
                    if pd.Timestamp(timestamp).year == year
                ]
                context_value_m3_per_year: float = float(
                    yearly_series.iloc[yearly_context_positions[0]]
                )
                bar_container = ax.bar(
                    "context",
                    context_value_m3_per_year,
                    color="none",
                    edgecolor="black",
                    linewidth=1.5,
                    hatch="//",
                )
                add_legend_entry(bar_container[0], label)
>>>>>>> cf97c6b046a395cd4914d7d5fbd13329e71ca3ac

            ax.set_title(f"Water Balance – {year}")
            ax.set_ylabel("m3/year")

        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=4,
        )

        if export:
<<<<<<< HEAD
            fig_path = folder / "water_balance_yearly_subplots.png"
            plt.savefig(fig_path, dpi=300)
            print(f"Water balance yearly plot saved as: {fig_path}")

        plt.show()

    def evaluate_warnings(self, *args: Any, **kwargs: Any) -> None:
        """Calculate performance metrics for warnings with respect to households.

        This method evaluates the number of warned households issued by the model with the households that should/should not have been warned.
        It calculates various performance metrics (hit rate, false alarms ratio, critical success index) and generates visualizations
        to illustrate the comparison. household_points: Path to the household points dataframe that contains the column with warning levels (geoparquet format).
        """
        # Load necessary geodataframes
        obs_flood_extent = gpd.read_parquet(
            self.model.config["hazards"]["floods"]["event_observation_files"]["vector"]
        )

        buildings_w_postal_codes = gpd.read_parquet(
            self.model.output_folder / "buildings_w_postal_codes.geoparquet"
        )

        # Folder where to get the households gdf from
        households_folder: Path = self.model.output_folder / "action_maps"

        def create_warning_performance_plots(
            hits: gpd.GeoDataFrame,
            misses: gpd.GeoDataFrame,
            false_alarms: gpd.GeoDataFrame,
            no_info: gpd.GeoDataFrame,
            HR: float,
            FAR: float,
            CSI: float,
            postal_codes: gpd.GeoDataFrame,
            criteria: str,
            output_folder: Path,
            initialization_date: str,
            obs_flood_extent: gpd.GeoDataFrame,
            catchment_borders: gpd.GeoDataFrame,
            target_crs: str | None = None,
            visualization_type: str = "OSM",
        ) -> None:
            """
            Creates a map showing the performance of the warnings for households.

            Args:
                hits: GeoDataFrame with the postal codes hits
                misses: GeoDataFrame with the postal codes misses
                false_alarms: GeoDataFrame with the postal code false alarms
                no_info: GeoDataFrame with the postal code no info
                HR: warnings hit rate for every forecast initialization date
                FAR: warnings false alarm ratio for every forecast initialization date
                CSI: warnings critical success index for every forecast initialization date
                postal_codes: GeoDataFrame with all postal codes
                output_folder: folder to save the figure
                initialization_date: string used in the title and filename
                obs_flood_extent: GeoDataFrame with the observed flood extent
                admin_units: GeoDataFrame with administrative units
                criteria: string indicating the criteria used for evaluation
                catchment_borders: GeoDataFrame with the catchment borders
                target_crs: target CRS (e.g., "EPSG:28992"). If None, uses the CRS of `flooded`.
                visualization_type: currently only "OSM" is used (to add basemap).
            """
            # Define the target CRS
            if target_crs is None:
                target_crs = obs_flood_extent.crs

            # Define extent based on observed flood extent
            minx, miny, maxx, maxy = catchment_borders.total_bounds
            margin = 500

            fig, ax = plt.subplots(figsize=(10, 10))

            # Get lead time
            start_time = pd.to_datetime(
                self.model.config["hazards"]["floods"]["flood_start"]
            )
            ini_dt = pd.to_datetime(initialization_date)
            leadtime_hours = (start_time - ini_dt).total_seconds() / 3600

            # Plot observed flood extent
            obs_flood_extent.plot(
                ax=ax,
                color="dodgerblue",
                edgecolor="dodgerblue",
                linestyle="-",
                linewidth=1.2,
                zorder=2,
                alpha=0.5,
            )

            # Plot hits misses, false alarms, and no info for postal codes
            postal_codes.plot(
                ax=ax,
                color="white",
                edgecolor="lightgrey",
                linestyle="-",
                linewidth=0.5,
                zorder=1,
                alpha=0.8,
            )

            no_info.plot(
                ax=ax,
                color="lightgrey",
                edgecolor="grey",
                linestyle="-",
                linewidth=0.5,
                zorder=3,
                alpha=0.8,
            )

            false_alarms.plot(
                ax=ax,
                color="orange",
                edgecolor="grey",
                linestyle="-",
                linewidth=0.5,
                zorder=4,
                alpha=0.8,
            )

            misses.plot(
                ax=ax,
                color="red",
                edgecolor="grey",
                linestyle="-",
                linewidth=0.5,
                alpha=0.8,
                zorder=5,
            )

            hits.plot(
                ax=ax,
                color="#94f944",
                edgecolor="grey",
                linestyle="-",
                linewidth=0.5,
                alpha=0.8,
                zorder=6,
            )

            catchment_borders.plot(
                ax=ax,
                color="black",
                linestyle="--",
                linewidth=1.5,
                zorder=7,
                alpha=0.5,
            )

            # Basemap
            if visualization_type == "OSM":
                # Extent
                ax.set_xlim(minx - margin, maxx + margin)
                ax.set_ylim(miny - margin, maxy + margin)
                ax.set_aspect("equal", adjustable="datalim")

                ctx.add_basemap(
                    ax,
                    crs=target_crs,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    zoom=12,
                    alpha=0.5,
                )

            ax.set_title(
                f"Lead Time: {leadtime_hours:.0f} hours",
                fontsize=16,
            )
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

            # Legend
            green_patch = mpatches.Patch(
                facecolor="#94f944", edgecolor="grey", label="Hits"
            )
            orange_patch = mpatches.Patch(
                facecolor="orange", edgecolor="grey", label="False Alarms"
            )
            red_patch = mpatches.Patch(
                facecolor="red", edgecolor="grey", label="Misses"
            )
            white_patch = mpatches.Patch(
                facecolor="white", edgecolor="lightgrey", label="Correct Negatives"
            )
            grey_patch = mpatches.Patch(
                facecolor="lightgrey", edgecolor="grey", label="No flood observation"
            )

            flood_extent_patch = mpatches.Patch(
                facecolor="dodgerblue",
                edgecolor="dodgerblue",
                alpha=0.5,
                label="Observed Flood Extent",
            )

            catchment_patch = Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Catchment Border (NL)",
            )

            legend_handles = [
                green_patch,
                orange_patch,
                red_patch,
                white_patch,
                grey_patch,
                flood_extent_patch,
                catchment_patch,
            ]

            legend = ax.legend(handles=legend_handles, loc="upper right", fontsize=14)

            # Box with metrics
            ax.text(
                0.02,
                0.02,
                (
                    "Metrics:\n"
                    f"HR   = {HR * 100:.1f} %\n"
                    f"FAR  = {FAR * 100:.1f} %\n"
                    f"CSI  = {CSI * 100:.1f} %"
                ),
                transform=ax.transAxes,
                fontsize=14,
                bbox=dict(
                    facecolor="white",
                    edgecolor="grey",
                    boxstyle="round,pad=0.3",
                    alpha=0.8,
                ),
                verticalalignment="bottom",
                horizontalalignment="left",
                zorder=6,
            )

            # CRS box
            label_crs = target_crs.to_epsg()
            crs_text = f"CRS: {label_crs}"
            ax.text(
                0.98,
                0.02,
                crs_text,
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(
                    facecolor="white",
                    edgecolor="grey",
                    boxstyle="round,pad=0.2",
                    alpha=0.8,
                ),
                verticalalignment="bottom",
                horizontalalignment="right",
                zorder=6,
            )

            # Save figure
            fig_name = f"warnings_performance_{criteria}_{initialization_date}.png"
            fig_path = output_folder / fig_name
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Warnings performance figure saved in: {fig_path}\n")

        def calculate_warning_performance_metrics(
            obs_flood_extent: gpd.GeoDataFrame,
            households_folder: Path,
            criteria: str = "fraction_buildings_flooded",
            buildings_w_postal_codes: Path | None = None,
            buildings_hit_threshold_flooded: float = 0.1,
            buildings_covered_threshold: float = 0.1,
        ) -> None:
            """Calculate performance metrics for warnings with respect to households.

            Args:
                obs_flood_extent: GeoDataFrame containing the observed flood extent.
                households_folder: Path to the folder containing household points GeoParquet files with the warning parameters. Must contain column of warning_level.
                criteria: Criteria to define flooded admin units. Options are 'any_building_flooded' or 'fraction_buildings_flooded'.
                buildings_w_postal_codes: GeoDataFrame of the buildings containing a column with their postal codes. Required if use_admin_units is True.
                buildings_hit_threshold_flooded: Fraction of buildings that must be flooded in an admin unit to consider it flooded.
                buildings_covered_threshold: Fraction of buildings that must be covered by orthophotos to consider it as a valid area.
            """
            files = sorted(
                households_folder.glob(
                    "households_with_warning_parameters_*.geoparquet"
                )
            )

            # Save the performance log to a csv file
            performance_folder = self.output_folder_evaluate / "warnings"
            performance_folder.mkdir(parents=True, exist_ok=True)

            # Prepare validation area based on flooded buildings per postal code
            postal_codes = gpd.read_parquet(
                self.model.files["geom"]["postal_codes"]
            ).to_crs(obs_flood_extent.crs)
            flooded_postal_codes = postal_codes.copy()
            valid_postal_codes = postal_codes.copy()
            catchment_borders = postal_codes.dissolve().boundary

            buildings = buildings_w_postal_codes.to_crs(obs_flood_extent.crs).copy()

            flood_union = obs_flood_extent.geometry.union_all()
            buildings["flooded"] = buildings.geometry.intersects(flood_union)

            # Save the buildings with their flooded status
            buildings.to_parquet(
                performance_folder / "buildings_flooded_status.parquet"
            )

            if criteria == "any_building_flooded":
                # Get postal codes with at least one flooded building
                stats = buildings.groupby("postcode")["flooded"].any()

                flooded_postal_codes["flooded"] = (
                    flooded_postal_codes["postcode"].map(stats).fillna(False)
                )

            elif criteria == "fraction_buildings_flooded":
                # Get the fraction of flooded buildings per postcode (mean of boolean is the fraction of True)
                stats = buildings.groupby("postcode")["flooded"].mean()

                # Mark postal codes as flooded if fraction of flooded buildings exceeds threshold
                flooded_postal_codes["flooded"] = (
                    flooded_postal_codes["postcode"].map(stats).fillna(0)
                    >= buildings_hit_threshold_flooded
                )
                print(
                    f"Threshold of fraction of buildings hit per postal code: {buildings_hit_threshold_flooded}"
                )

            # Define validation area: postal codes that are flooded (depending on the criteria used)
            flooded_area = flooded_postal_codes[flooded_postal_codes["flooded"] == True]
            # Save it for reference
            flooded_area.to_parquet(
                performance_folder
                / "postal_codes_considered_flooded_by_fraction.geoparquet"
            )
            print(
                "Number of flooded postal codes based on criteria:",
                flooded_area.shape[0],
            )

            # TODO: Check if we need to filter valid postal codes based on buildings within orthophoto boundary
            # TODO: remove hardcoded path
            # Read orthophoto + flooded area boundary
            ortho_boundary = gpd.read_parquet(
                "/scistor/ivm/adq582/GEB/models/geul_new_warning/base/input/geom/geul_ortho_plus_flood.parquet"
            )

            # Define postal codes with flood observations
            postal_codes_w_flood_obs = gpd.sjoin(
                valid_postal_codes, ortho_boundary, how="inner", predicate="intersects"
            ).drop(columns="index_right")

            # Get the fraction of buildings within the [orthophoto + flood extent] layer per postal code
            ortho_union = ortho_boundary.geometry.union_all()
            buildings["in_ortho"] = buildings.geometry.intersects(ortho_union)
            stats = buildings.groupby("postcode")["in_ortho"].mean()

            # Define the valid area as postal codes which the fraction of buildings within the [orthophoto + flood extent] layer exceeds buildings hit threshold
            # (otherwise the model would never be able to consider them as flooded)
            valid_postal_codes["in_ortho"] = (
                valid_postal_codes["postcode"].map(stats).fillna(0)
                >= buildings_covered_threshold
            )

            valid_postal_codes = valid_postal_codes[
                valid_postal_codes["in_ortho"] == True
            ]

            # Save valid postal codes
            valid_postal_codes.to_parquet(
                performance_folder / "valid_postal_codes.geoparquet"
            )

            # Evaluate each household points file at a postal code scale
            performance_log = []
            for file in files:
                initialization_date_str = file.stem.split("_")[-1]
                ini_date_folder = performance_folder / initialization_date_str
                ini_date_folder.mkdir(parents=True, exist_ok=True)

                household_points = gpd.read_parquet(file)

                forecast_init_dt = datetime.datetime.strptime(
                    initialization_date_str, "%Y%m%dT%H%M%S"
                )
                print(
                    f"Evaluating warnings for forecast initialization date: {forecast_init_dt.isoformat()}"
                )

                # Evaluate the warning performance at a postal code scale
                # For each postal code, check if any household got a warning (that means the warning reached that postal code)
                warned_status = (
                    household_points.groupby("postcode")["warning_reached"]
                    .any()
                    .rename("warned")
                    .reset_index()
                )

                # Merge the warned_status information in the postal codes geodataframe
                postal_codes_warned_status = postal_codes.merge(
                    warned_status,
                    on="postcode",
                    how="left",
                )

                # Postal codes with no household points (NaN) become not warned (False)
                postal_codes_warned_status["warned"] = postal_codes_warned_status[
                    "warned"
                ].fillna(False)

                # Save the postal codes with their warned status for this forecast initialization date
                postal_codes_warned_status.to_parquet(
                    ini_date_folder
                    / f"warned_postal_codes_{initialization_date_str}.parquet"
                )

                # Calculate hits, misses, false alarms and no info at postal code level
                hits_mask = (postal_codes_warned_status["warned"]) & (
                    postal_codes_warned_status["postcode"].isin(
                        flooded_area["postcode"]
                    )
                )

                n_hits = hits_mask.sum()
                print(f"Number of postal codes correctly warned: {n_hits}")

                misses_mask = (~postal_codes_warned_status["warned"]) & (
                    postal_codes_warned_status["postcode"].isin(
                        flooded_area["postcode"]
                    )
                )

                n_misses = misses_mask.sum()
                print(f"Number of postal codes missed: {n_misses}")

                false_alarms_mask = (
                    (postal_codes_warned_status["warned"])
                    & (
                        ~postal_codes_warned_status["postcode"].isin(
                            flooded_area["postcode"]
                        )
                    )
                    & (
                        postal_codes_warned_status["postcode"].isin(
                            valid_postal_codes["postcode"]
                        )
                    )
                )

                n_false_alarms = false_alarms_mask.sum()
                print(f"Number of postal codes false alarms: {n_false_alarms}")

                no_info_mask = ~postal_codes_warned_status["postcode"].isin(
                    postal_codes_w_flood_obs["postcode"]
                )
                # ~postal_codes_warned_status["postcode"].isin(
                #     valid_postal_codes["postcode"]
                #

                n_no_info = no_info_mask.sum()
                print(f"Number of postal codes with no info: {n_no_info}")

                # Get the geodataframes for hits, misses, false alarms and no info
                hits = postal_codes_warned_status[hits_mask]
                misses = postal_codes_warned_status[misses_mask]
                false_alarms = postal_codes_warned_status[false_alarms_mask]
                no_info = postal_codes_warned_status[no_info_mask]

                # Save the geodataframes for hits, misses, false alarms and no info for this forecast initialization date
                hits.to_parquet(
                    ini_date_folder
                    / f"postal_codes_hits_{initialization_date_str}.parquet"
                )
                misses.to_parquet(
                    ini_date_folder
                    / f"postal_codes_misses_{initialization_date_str}.parquet"
                )
                false_alarms.to_parquet(
                    ini_date_folder
                    / f"postal_codes_false_alarms_{initialization_date_str}.parquet"
                )
                no_info.to_parquet(
                    ini_date_folder
                    / f"postal_codes_no_info_{initialization_date_str}.parquet"
                )

                HR = n_hits / (n_hits + n_misses)
                print(f"Overall Hit Rate: {HR:.0%}")

                FAR = n_false_alarms / (n_hits + n_false_alarms)
                print(f"False Alarm Rate: {FAR:.0%}")

                CSI = n_hits / (n_hits + n_misses + n_false_alarms)
                print(f"Critical Success Index: {CSI:.0%}")

                performance_log.append(
                    {
                        "date_time": initialization_date_str,
                        "n_flooded_buildings": buildings["flooded"].sum(),
                        "n_warned_households": household_points[
                            household_points["warning_level"] >= 1
                        ].shape[0],
                        "n_postal_codes_hits": n_hits,
                        "n_postal_codes_misses": n_misses,
                        "n_postal_codes_false_alarms": n_false_alarms,
                        "HR": f"{HR:.2f}",
                        "FAR": f"{FAR:.2f}",
                        "CSI": f"{CSI:.2f}",
                    }
                )

                create_warning_performance_plots(
                    hits=hits,
                    misses=misses,
                    false_alarms=false_alarms,
                    no_info=no_info,
                    catchment_borders=catchment_borders,
                    criteria=criteria,
                    HR=HR,
                    FAR=FAR,
                    CSI=CSI,
                    postal_codes=postal_codes,
                    output_folder=ini_date_folder,
                    initialization_date=initialization_date_str,
                    obs_flood_extent=obs_flood_extent,
                    target_crs=obs_flood_extent.crs,
                    visualization_type="OSM",
                )

            path = performance_folder / "warnings_performance.csv"
            pd.DataFrame(performance_log).to_csv(path, index=False)

        calculate_warning_performance_metrics(
            obs_flood_extent=obs_flood_extent,
            households_folder=households_folder,
            buildings_w_postal_codes=buildings_w_postal_codes,
        )

    # roy improved version of evaluate_hydrodynamics
    def evaluate_hydrodynamics(
        self, run_name: str = "default", *args: Any, **kwargs: Any
    ) -> None:
        """Evaluate hydrodynamic model performance against flood observations.

        Calculates performance metrics (hit rate, false alarm ratio, critical success index)
        for flood maps generated by the hydrodynamic model by comparing them against
        observed flood extent data.

        Args:
            run_name: Name of the simulation run to evaluate.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Raises:
            FileNotFoundError: If the flood map folder does not exist in the output directory.
            ValueError: If the flood observation file is not in .zarr format.
        """

        # Main function for the performance metrics
        def calculate_performance_metrics(
            observation: Path | str,
            flood_map_path: Path | str,
            visualization_type: str = "Hillshade",
            output_folder: Path | str = None,
            flood_probability_map: bool = False,
            ini_dt: str | None = None,
            use_ortho_boundary: bool = True,
        ) -> None:
            """Calculate performance metrics for flood maps against observations.

            Args:
                observation: Path to the observed flood extent data (.zarr format).
                flood_map_path: Path to the model-generated flood map data (.zarr format).
                visualization_type: Type of visualization for plotting (default is "Hillshade").
                output_folder: Path to the folder where results will be saved.

            Raises:
                ValueError: If the observation file is not in .zarr format.
            """
            # Step 1: Open needed datasets
            flood_map = read_zarr(flood_map_path)
            obs = rxr.open_rasterio(observation)
            # # resolution of obs should match the sfincs resolution
            obs = obs.rio.write_crs(flood_map.rio.crs)
            sim = flood_map.raster.reproject_like(obs)
            if flood_probability_map:
                sim = sim >= 0.6
                # get lead time in hours between the forecast initialization and the flood start time
                ini_dt = pd.to_datetime(ini_dt)
                start_time = pd.to_datetime(self.config["floods"]["flood_start"])
                leadtime_hours = (start_time - ini_dt).total_seconds() / 3600
                print(f"Lead time of the forecast: {leadtime_hours:.0f} hours")
            rivers = gpd.read_parquet(
                Path("simulation_root")
                / run_name
                / "SFINCS"
                / "entire_region"
                / "rivers.geoparquet"
            )
            # region = gpd.read_file(
            #     Path("simulation_root")
            #     / run_name
            #     / "SFINCS"
            #     / "entire_region"
            #     / "gis"
            #     / "region.geojson"
            # ).to_crs(obs.rio.crs)
            # region = gpd.read_parquet(
            #     "/scistor/ivm/adq582/GEB/models/geul_new_warning/base/input/geom/geul_catchment.parquet"
            # ).to_crs(obs.rio.crs)
            region = gpd.read_parquet(
                "/scistor/ivm/adq582/GEB/models/geul_new_warning/base/input/geom/postal_codes6_no_bunde.parquet"
            ).to_crs(obs.rio.crs)
            region = (
                region.dissolve()
            )  # dissolve to get one geometry for the entire region

            # Step 2: Clip out rivers from observations and simulations
            crs_wgs84 = CRS.from_epsg(4326)
            crs_mercator = CRS.from_epsg(3857)
            rivers.set_crs(crs_wgs84, inplace=True)
            gdf_mercator = rivers.to_crs(crs_mercator)
            gdf_mercator["geometry"] = gdf_mercator.buffer(gdf_mercator["width"] / 2)

            # Create river mask for simulation data
            gdf_buffered_sim = gdf_mercator.to_crs(sim.rio.crs)
            rivers_mask_sim = ~geometry_mask(
                gdf_buffered_sim.geometry,
                out_shape=sim.rio.shape,
                transform=sim.rio.transform(),
                all_touched=True,
                invert=False,
            )
            sim_no_rivers = sim.where(~rivers_mask_sim).fillna(0)

            # Create river mask for observation data
            gdf_buffered_obs = gdf_mercator.to_crs(obs.rio.crs)
            rivers_mask_obs = ~geometry_mask(
                gdf_buffered_obs.geometry,
                out_shape=obs.rio.shape,
                transform=obs.rio.transform(),
                all_touched=True,
                invert=False,
            )
            obs_no_rivers = obs.where(~rivers_mask_obs).fillna(0)

            # Step 3: Clip out region from observations
            # obs_region = obs_no_rivers.rio.clip(geoms, region.crs, drop=True)
            obs_region = obs_no_rivers.rio.clip(
                region.geometry.values, region.crs, drop=True
            )
            sim_region = sim_no_rivers.rio.clip(
                region.geometry.values, region.crs, drop=True
            )

            # Step 4: Optionally clip using extra validation region from config yml
            extra_validation_path = self.config["floods"].get(
                "extra_validation_region", None
            )

            if extra_validation_path and Path(extra_validation_path).exists():
                extra_clip_region = gpd.read_file(extra_validation_path).set_crs(28992)
                extra_clip_region = extra_clip_region.to_crs(region.crs)
                extra_clip_region_buffer = extra_clip_region.buffer(160)

                sim_extra_clipped = sim_no_rivers.rio.clip(
                    extra_clip_region_buffer.geometry.values,
                    extra_clip_region_buffer.crs,
                )
                clipped_out = (sim_no_rivers > 0.15) & (sim_extra_clipped.isnull())
                clipped_out_raster = sim_no_rivers.where(clipped_out)
            else:
                sim_extra_clipped = sim_no_rivers
                clipped_out_raster = xr.full_like(sim_no_rivers, np.nan)

            # Step 5: Mask water depth values
            hmin = 0.15  # TODO: check if this makes sense
            sim_extra_clipped = sim_region.raster.reproject_like(obs_region)
            # sim_extra_clipped = sim_extra_clipped.raster.reproject_like(obs_region)
            simulation_final = sim_extra_clipped > hmin
            observation_final = obs_region > 0

            # Further clip to orthophoto boundary if provided
            if use_ortho_boundary:
                ortho_boundary = gpd.read_parquet(
                    "/scistor/ivm/adq582/GEB/models/geul_new_warning/base/input/geom/geul_ortho_plus_flood.parquet"
                )
                ortho_boundary = ortho_boundary.to_crs(flood_map.rio.crs)

                obs_valid = xr.DataArray(
                    geometry_mask(
                        ortho_boundary.geometry,
                        out_shape=obs_region.rio.shape,
                        transform=obs_region.rio.transform(),
                        all_touched=True,
                        invert=True,
                    ),
                    dims=obs_region.dims[-2:],  # normalmente ("y", "x")
                    coords={"y": obs_region.y, "x": obs_region.x},
                )

            xmin, ymin, xmax, ymax = region.total_bounds
            catchment_extent = [xmin, xmax, ymin, ymax]

            xmin, ymin, xmax, ymax = observation_final.rio.bounds()
            flood_extent = [xmin, xmax, ymin, ymax]

            # Need to adjust the extent, and remove the sim area that does not intersect the orthophoto boundary before going into the hit,misses, calculations

            # Step 6: Calculate performance metrics
            # Compute the arrays first to get concrete values
            simulation_valid = simulation_final.where(obs_valid)
            # observation_valid = observation_final.where(obs_valid)

            sim_final_computed = (
                simulation_valid.compute()
            )  # antes era simulation_final.compute()
            obs_final_computed = (
                observation_final.compute()
            )  # antes era observation_final.compute()

            hit_rate = calculate_hit_rate(sim_final_computed, obs_final_computed) * 100
            false_rate = (
                calculate_false_alarm_ratio(sim_final_computed, obs_final_computed)
                * 100
            )
            csi = (
                calculate_critical_success_index(sim_final_computed, obs_final_computed)
                * 100
            )

            flooded_pixels = float(sim_final_computed.sum().item())

            # Calculate resolution in meters from coordinate spacing
            x_res = float(np.abs(flood_map.x[1] - flood_map.x[0]))
            pixel_size = x_res  # meters
            flooded_area_km2 = flooded_pixels * (pixel_size * pixel_size) / 1_000_000

            # Step 7: Save results to file and plot the results
            elevation_data = read_zarr(self.model.files["other"]["DEM/fabdem"])
            elevation_data = elevation_data.rio.reproject_match(obs)

            elevation_array = (
                elevation_data.squeeze().astype("float32").compute().values
            )

            ls = LightSource(azdeg=315, altdeg=45)
            hillshade = ls.hillshade(elevation_array, vert_exag=1, dx=1, dy=1)

            # Catchment borders
            target_crs = obs_region.rio.crs
            catchment_borders = region.boundary

            if simulation_final.sum() > 0:
                # misses = (observation_final == 1) & (simulation_final == 0)
                # simulation_masked = simulation_final.where(simulation_final == 1)
                # hits = simulation_masked.where(observation_final)
                # misses_masked = misses.where(misses == 1)

                # no_info = (simulation_final == 1) & (~obs_valid)
                # no_info_masked = no_info.where(no_info == 1)
                # --- separar área válida vs não válida ---
                simulation_valid = (
                    simulation_final.where(obs_valid).fillna(0).astype(bool)
                )
                observation_valid = (
                    observation_final.where(obs_valid).fillna(0).astype(bool)
                )

                # --- classes dentro da área válida ---
                hits = simulation_valid & observation_valid
                misses = (~simulation_valid) & observation_valid
                false_alarms = simulation_valid & (~observation_valid)

                # --- classe fora da área válida (no_info) ---
                no_info = simulation_final & (~obs_valid)

                # mascarinhas pra plot (deixar só 1s)
                hits_masked = hits.where(hits)
                misses_masked = misses.where(misses)
                false_alarms_masked = false_alarms.where(false_alarms)
                no_info_masked = no_info.where(no_info)

                if visualization_type == "OSM":
                    # Ensure all arrays have the same shape by squeezing extra dimensions
                    simulation_final = simulation_final.squeeze()
                    observation_final = observation_final.squeeze()
                    misses = misses.squeeze()
                    hits = hits.squeeze()
                    # simulation_masked = simulation_masked.squeeze()
                    misses_masked = misses_masked.squeeze()
                    no_info = no_info.squeeze()
                    margin = 500
                    fig, ax = plt.subplots(figsize=(10, 10))

                    # clipped_out_raster.plot(
                    #     ax=ax, cmap=blue_cmap, add_colorbar=False, add_labels=False
                    # )
                    cmap_noinfo = ListedColormap(["grey"])

                    ax.imshow(
                        false_alarms_masked,  # False alarms, antes era simulation_masked
                        extent=flood_extent,
                        # origin="lower",
                        cmap="Wistia",
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        zorder=2,
                    )
                    ax.imshow(
                        misses_masked,  # Misses
                        extent=flood_extent,
                        # origin="lower",
                        cmap="autumn_r",
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        zorder=3,
                    )

                    ax.imshow(
                        hits_masked,
                        extent=flood_extent,
                        # origin="lower",
                        cmap="brg",
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        zorder=4,
                    )
                    if use_ortho_boundary:
                        ax.imshow(
                            no_info_masked,
                            extent=flood_extent,
                            # origin="lower",
                            cmap=cmap_noinfo,
                            vmin=0,
                            vmax=1,
                            interpolation="none",
                            zorder=1,
                        )

                    catchment_borders.plot(
                        ax=ax,
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        zorder=5,
                        alpha=0.5,
                    )
                    # Set the extent based on the raster bounds
                    ax.set_xlim(
                        catchment_extent[0] - margin, catchment_extent[1] + margin
                    )
                    ax.set_ylim(
                        catchment_extent[2] - margin, catchment_extent[3] + margin
                    )
                    ax.set_aspect("equal", adjustable="datalim")

                    if visualization_type == "OSM":
                        # Add a base map
                        ctx.add_basemap(
                            ax,
                            crs=target_crs,
                            source="OpenStreetMap.Mapnik",
                            zoom=12,
                            zorder=0,
                            alpha=0.5,
                        )
                    elif visualization_type == "Hillshade":
                        ax.imshow(
                            hillshade,
                            cmap="gray",
                            extent=(
                                elevation_data.x.min(),
                                elevation_data.x.max(),
                                elevation_data.y.min(),
                                elevation_data.y.max(),
                            ),
                        )

                    if flood_probability_map:
                        ax.set_title(
                            f"Lead Time: {leadtime_hours:.0f} hours\n",
                            fontsize=20,
                        )
                    else:
                        ax.set_title(
                            "ERA5 reanalysis\n",
                            fontsize=20,
                        )

                    ax.set_ylabel("y (m)", fontsize=14)
                    ax.tick_params(axis="y", labelsize=14)
                    ax.locator_params(axis="y", nbins=5)

                    ax.set_xlabel("x (m)", fontsize=14)
                    ax.tick_params(axis="x", labelsize=14)
                    ax.locator_params(axis="x", nbins=5)

                    green_patch = mpatches.Patch(color="#94f944", label="Hits")
                    orange_patch = mpatches.Patch(color="orange", label="False Alarms")
                    red_patch = mpatches.Patch(color="red", label="Misses")
                    grey_patch = mpatches.Patch(
                        color="grey", label="No flood observations"
                    )

                    catchment_patch = Line2D(
                        [0],
                        [0],
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        label="Catchment Border (NL)",
                    )
                    legend = ax.legend(
                        handles=[
                            green_patch,
                            orange_patch,
                            red_patch,
                            grey_patch,
                            catchment_patch,
                        ],
                        fontsize=18,
                    )

                    # Add a comment about the metrics in the plot
                    legend_bbox = legend.get_window_extent(
                        renderer=fig.canvas.get_renderer()
                    )
                    legend_bbox_ax = legend_bbox.transformed(ax.transAxes.inverted())

                    # Add text below legend using axes coordinates
                    # ax.annotate(
                    #     f"Validation Metrics:\n"
                    #     f"HR    = {hit_rate:.2f} %\n"
                    #     f"FAR   = {false_rate:.2f} %\n"
                    #     f"CSI   = {csi:.2f} %",
                    #     xy=(legend_bbox_ax.x0 + 0.055, legend_bbox_ax.y0 + 0.002),
                    #     xycoords="axes fraction",
                    #     fontsize=10,
                    #     bbox=dict(
                    #         facecolor="white",
                    #         edgecolor="grey",
                    #         boxstyle="round,pad=0.2",
                    #         alpha=0.8,
                    #     ),
                    #     verticalalignment="top",
                    #     horizontalalignment="right",
                    #     zorder=5,
                    # )
                    ax.annotate(
                        f"Validation Metrics:\n"
                        f"HR    = {hit_rate:.2f} %\n"
                        f"FAR   = {false_rate:.2f} %\n"
                        f"CSI   = {csi:.2f} %",
                        xy=(0.02, 0.02),  # bottom-left corner
                        xycoords="axes fraction",
                        fontsize=18,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="grey",
                            boxstyle="round,pad=0.25",
                            alpha=0.8,
                        ),
                        verticalalignment="bottom",
                        horizontalalignment="left",
                        zorder=5,
                    )

                    crs_text = f"CRS: {target_crs.to_string()}"
                    ax.annotate(
                        crs_text,
                        xy=(0.99, 0.02),  # Bottom right corner in axes coordinates
                        xycoords="axes fraction",
                        fontsize=14,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="grey",
                            boxstyle="round,pad=0.2",
                            alpha=0.8,
                        ),
                        verticalalignment="bottom",
                        horizontalalignment="right",
                        zorder=6,
                    )

                    simulation_filename = os.path.splitext(
                        os.path.basename(flood_map_path)
                    )[0]
                    if flood_probability_map:
                        fig.savefig(
                            output_folder
                            / f"forecast_{ini_dt.strftime('%Y%m%dT%H%M%S')}"
                            / f"{simulation_filename}_validation_floodextent_plot.png",
                            dpi=300,
                            bbox_inches="tight",
                        )
                        print(
                            f"Figure with {visualization_type} saved as: {output_folder / f'forecast_{ini_dt.strftime("%Y%m%dT%H%M%S")}' / f'{simulation_filename}_validation_extent_plot.png'}"
                        )
                    else:
                        fig.savefig(
                            output_folder
                            / f"{simulation_filename}_validation_floodextent_plot.png",
                            dpi=300,
                            bbox_inches="tight",
                        )
                        print(
                            f"Figure with {visualization_type} saved as: {output_folder / f'{simulation_filename}_validation_extent_plot.png'}"
                        )

                elif visualization_type == "Hillshade":
                    green_cmap = mcolors.ListedColormap(["green"])  # Hits
                    orange_cmap = mcolors.ListedColormap(["orange"])  # False alarms
                    red_cmap = mcolors.ListedColormap(["red"])  # Misses
                    blue_cmap = mcolors.ListedColormap(
                        ["#72c1db"]
                    )  # No observation data

                    fig, ax = plt.subplots(figsize=(10, 10))

                    ax.imshow(
                        hillshade,
                        cmap="gray",
                        extent=(
                            elevation_data.x.min(),
                            elevation_data.x.max(),
                            elevation_data.y.min(),
                            elevation_data.y.max(),
                        ),
                    )

                    region.boundary.plot(
                        ax=ax, edgecolor="black", linewidth=2, label="Region Boundary"
                    )

                    clipped_out_raster.plot(
                        ax=ax, cmap=blue_cmap, add_colorbar=False, add_labels=False
                    )
                    simulation_masked.plot(
                        ax=ax, cmap=orange_cmap, add_colorbar=False, add_labels=False
                    )
                    hits.plot(
                        ax=ax, cmap=green_cmap, add_colorbar=False, add_labels=False
                    )
                    misses_masked.plot(
                        ax=ax, cmap=red_cmap, add_colorbar=False, add_labels=False
                    )

                    ax.set_aspect("equal")
                    ax.axis("off")
                    handles = [
                        mpatches.Patch(color=green_cmap(0.5)),
                        mpatches.Patch(color=red_cmap(0.5)),
                        mpatches.Patch(color=orange_cmap(0.5)),
                        mpatches.Patch(color=blue_cmap(0.5)),
                        mlines.Line2D([], [], color="black", linewidth=2),
                    ]

                    labels = [
                        "Hits",
                        "Misses",
                        "False alarms",
                        "No observation data",
                        "Region",
                    ]

                    plt.legend(
                        handles=handles, labels=labels, loc="upper right", fontsize=16
                    )

                    simulation_filename = os.path.splitext(
                        os.path.basename(flood_map_path)
                    )[0]
                    plt.savefig(
                        output_folder
                        / f"{simulation_filename}_validation_floodextent_plot.png"
                    )
                    print(
                        f"Figure with {visualization_type} saved as: {output_folder / f'{simulation_filename}_validation_floodextent_plot.png'}"
                    )

                else:
                    raise ValueError(
                        f"Unknown visualization type: {visualization_type}, choose either 'OSM' or 'Hillshade'."
                    )

                performance_numbers = (
                    output_folder / f"{simulation_filename}_performance_metrics.txt"
                )

                with open(performance_numbers, "w") as f:
                    f.write(f"Hit rate (H): {hit_rate}\n")
                    f.write(f"False alarm rate (F): {false_rate}\n")
                    f.write(f"Critical Success Index (CSI) (C): {csi}\n")
                    f.write(f"Number of flooded pixels: {flooded_pixels}\n")
                    f.write(f"Flooded area (km2): {flooded_area_km2}")

                # Return metrics for further analysis
                return {
                    "hit_rate": hit_rate,
                    "false_alarm_rate": false_rate,
                    "csi": csi,
                    "flooded_area_km2": flooded_area_km2,
                }

        def create_forecast_performance_plots(
            performance_df: pd.DataFrame, event_name: str, output_folder: Path
        ) -> None:
            """Create performance metric plots showing spread and mean across forecast initializations.

            Generates line plots showing how performance metrics vary across different
            forecast initialization times for a given flood event. Shows both the spread
            of the ensemble (min-max range) with transparent fill and the ensemble mean as a black line.

            Notes:
                The spread is calculated using the minimum and maximum values across ensemble
                members for each forecast initialization. If only one member exists for a
                forecast initialization, no spread is shown for that time point.

            Args:
                performance_df: DataFrame containing performance metrics with forecast
                    initialization times and ensemble members. Must include columns:
                    'forecast_init', 'hit_rate', 'false_alarm_rate', 'csi', 'flooded_area_km2'.
                event_name: Name of the flood event being analyzed.
                output_folder: Directory to save the performance plots.

            Raises:
                ValueError: If required columns are missing from performance_df.
            """
            # Validate required columns
            required_columns = [
                "forecast_init",
                "hit_rate",
                "false_alarm_rate",
                "csi",
                "flooded_area_km2",
            ]
            missing_columns = [
                col for col in required_columns if col not in performance_df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in performance_df: {missing_columns}"
                )

            # Group by forecast initialization and calculate statistics
            grouped_stats = (
                performance_df.groupby("forecast_init")[
                    ["hit_rate", "false_alarm_rate", "csi", "flooded_area_km2"]
                ]
                .agg(["mean", "min", "max", "std", "count"])
                .reset_index()
            )

            # Convert forecast_init to datetime for better plotting
            grouped_stats["forecast_init_dt"] = pd.to_datetime(
                grouped_stats["forecast_init"], format="%Y%m%dT%H%M%S"
            )

            # Sort by datetime for proper line plotting
            grouped_stats = grouped_stats.sort_values("forecast_init_dt")

            # Create subplots for different metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                f"Forecast Performance Metrics with Ensemble Spread - {event_name}",
                fontsize=16,
                fontweight="bold",
            )

            # Define colors for spread (transparent) and metric-specific colors
            spread_colors = {
                "hit_rate": "#2E86AB",
                "false_alarm_rate": "#A23B72",
                "csi": "#F18F01",
                "flooded_area_km2": "#636EFA",
            }

            # Hit Rate
            ax = axes[0, 0]
            metric = "hit_rate"

            # Plot spread (min-max range) with transparent fill
            ax.fill_between(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "min")],
                grouped_stats[(metric, "max")],
                color=spread_colors[metric],
                alpha=0.3,
                label="Ensemble member range (Min-Max)",
            )

            # Plot mean as black line
            ax.plot(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "mean")],
                color="black",
                linewidth=2,
                marker="o",
                markersize=6,
                label="Ensemble Mean",
            )

            ax.set_title("Hit Rate (%)", fontweight="bold")
            ax.set_ylabel("Hit Rate (%)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.legend()

            # False Alarm Rate
            ax = axes[0, 1]
            metric = "false_alarm_rate"

            ax.fill_between(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "min")],
                grouped_stats[(metric, "max")],
                color=spread_colors[metric],
                alpha=0.3,
                label="Ensemble member range (Min-Max)",
            )

            ax.plot(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "mean")],
                color="black",
                linewidth=2,
                marker="s",
                markersize=6,
                label="Ensemble Mean",
            )

            ax.set_title("False Alarm Rate (%)", fontweight="bold")
            ax.set_ylabel("False Alarm Rate (%)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.legend()

            # Critical Success Index
            ax = axes[1, 0]
            metric = "csi"

            ax.fill_between(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "min")],
                grouped_stats[(metric, "max")],
                color=spread_colors[metric],
                alpha=0.3,
                label="Ensemble member range (Min-Max)",
            )

            ax.plot(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "mean")],
                color="black",
                linewidth=2,
                marker="^",
                markersize=6,
                label="Ensemble Mean",
            )

            ax.set_title("Critical Success Index (%)", fontweight="bold")
            ax.set_ylabel("CSI (%)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.legend()

            # Flooded Area
            ax = axes[1, 1]
            metric = "flooded_area_km2"

            ax.fill_between(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "min")],
                grouped_stats[(metric, "max")],
                color=spread_colors[metric],
                alpha=0.3,
                label="Ensemble member range (Min-Max)",
            )

            ax.plot(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "mean")],
                color="black",
                linewidth=2,
                marker="d",
                markersize=6,
                label="Ensemble Mean",
            )

            ax.set_title("Flooded Area (km²)", fontweight="bold")
            ax.set_ylabel("Area (km²)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Format x-axis for all subplots
            for ax in axes.flat:
                ax.tick_params(axis="x", rotation=45)
                ax.set_xlabel("Forecast Initialization Time")

            plt.tight_layout()

            # Save the plot
            plot_filename = (
                f"{event_name.replace(':', '_')}_forecast_performance_spread.png"
            )
            fig.savefig(output_folder / plot_filename, dpi=300, bbox_inches="tight")
            print(
                f"Forecast performance spread plot saved as: {output_folder / plot_filename}"
            )

        self.config = self.model.config["hazards"]

        eval_hydrodynamics_folders = Path(self.output_folder_evaluate) / "hydrodynamics"

        eval_hydrodynamics_folders.mkdir(parents=True, exist_ok=True)

        # Calculate performance metrics for every event in the config file
        for event in self.config["floods"]["events"]:
            event_name = f"{event['start_time'].strftime('%Y%m%dT%H%M%S')} - {event['end_time'].strftime('%Y%m%dT%H%M%S')}"
            print(f"event: {event_name}")

            flood_maps_folder = self.model.output_folder / "flood_maps"

            # check if flood map folder exists
            if not flood_maps_folder.exists():
                raise FileNotFoundError(
                    "Flood map folder does not exist in the output directory. Did you run the hydrodynamic model?"
                )

            # check if observation file exists
            if not Path(
                self.config["floods"]["event_observation_files"]["raster"]
            ).exists():
                raise FileNotFoundError(
                    "Flood observation file is not found in the given path in the model.yml. "
                    "Please check the path in the config file."
                )

            if self.model.config["evaluation"]["probability_map"]:
                print("Evaluating probabilistic flood extent (>hmin)...")
                flood_map_name = event_name + ".zarr"
                flood_map_path = flood_maps_folder / flood_map_name

                # TODO: need to adjust this later
                datetimes = pd.to_datetime(
                    ["2021-07-11", "2021-07-12", "2021-07-13", "2021-07-14"]
                )

                prob_exceedance_maps_folder = (
                    self.model.output_folder / "flood_prob_exceedance_maps"
                )
                save_folder = eval_hydrodynamics_folders / "prob_exceedance_maps"
                save_folder.mkdir(parents=True, exist_ok=True)

                # TODO: change the prob exceedance map path
                for dt in datetimes:
                    flood_prob_map_path = (
                        prob_exceedance_maps_folder
                        / f"forecast_{dt.strftime('%Y%m%dT%H%M%S')}"
                        / "prob_exceedance_map_range1_strategy1.zarr"
                    )

                    metrics = calculate_performance_metrics(
                        observation=self.config["floods"]["event_observation_files"][
                            "raster"
                        ],
                        flood_map_path=flood_prob_map_path,
                        visualization_type="OSM",
                        output_folder=save_folder,
                        flood_probability_map=True,
                        ini_dt=dt,
                    )
                    print(f"Successfully evaluated: {flood_prob_map_path.name}")

            # elif self.model.config["evaluation"]["ERA5"]:
            if self.model.config["evaluation"]["ERA5"]:
                print("Evaluating flood extent from ERA5 reanalysis...")
                event_folder = eval_hydrodynamics_folders / event_name
                event_folder.mkdir(parents=True, exist_ok=True)
                metrics = calculate_performance_metrics(
                    observation=self.config["floods"]["event_observation_files"][
                        "raster"
                    ],
                    flood_map_path=flood_map_path,
                    visualization_type="OSM",
                    output_folder=event_folder,
                    flood_probability_map=False,
                    ini_dt=dt,
                )
                print(f"Successfully evaluated: {flood_map_path.name}")

            if self.model.config["evaluation"]["forecasts"]:
                print("Evaluating flood extent of all ensemble flood maps...")

                forecast_folders = sorted(flood_maps_folder.glob("forecast_*"))
                initialization_dates = [
                    f.name.split("_", 1)[1] for f in forecast_folders
                ]

                # get number of members from the first forecast folder
                first_forecast_folder = forecast_folders[0]
                n_ensemble_members = sum(
                    1 for _ in first_forecast_folder.glob("member_*")
                )

                forecast_metrics_list = []
                performance_metrics_list = []
                for init_date_str in initialization_dates:
                    for member in range(1, n_ensemble_members + 1):
                        flood_map_path = (
                            flood_maps_folder
                            / f"forecast_{init_date_str}"
                            / f"member_{member}"
                            / f"{event_name}.zarr"
                        )

                        event_folder = (
                            eval_hydrodynamics_folders
                            / "forecasts"
                            / f"forecast_{init_date_str}"
                            / f"member_{member}"
                        )
                        event_folder.mkdir(parents=True, exist_ok=True)

                        metrics = calculate_performance_metrics(
                            observation=self.config["floods"][
                                "event_observation_files"
                            ]["raster"],
                            flood_map_path=flood_map_path,
                            visualization_type="OSM",
                            output_folder=event_folder,
                        )
                        print("   Flood map evaluation complete.")

                        metrics_with_metadata = {
                            "forecast_init": init_date_str,
                            "member": member,
                            "filename": f"{init_date_str}_member{member}.zarr",
                            **metrics,
                        }

                        performance_metrics_list.append(metrics_with_metadata)
                        forecast_metrics_list.append(metrics)
                        print(f"   Successfully evaluated: {flood_map_path.name}")

                if performance_metrics_list:
                    performance_df = pd.DataFrame(performance_metrics_list)

                    # Create forecast performance plots
                    create_forecast_performance_plots(
                        performance_df, event_name, event_folder
                    )

                    # Save detailed performance metrics
                    detailed_filename = f"{event_name.replace(' - ', '_')}_detailed_performance_metrics.csv"
                    performance_df.to_csv(event_folder / detailed_filename, index=False)
                    print(
                        f"Detailed performance metrics saved as: {event_folder / detailed_filename}"
                    )

            print(f"Completed processing event: {event_name}\n")

        print("Flood map performance metrics calculated for all events.")
=======
            fig_path = (
                self.water_balance_output_folder / "water_balance_yearly_subplots.svg"
            )
            plt.savefig(fig_path)
            self.model.logger.info(f"Water balance yearly plot saved as: {fig_path}")

        plt.show()
        plt.close(fig)

        folder: Path = self.model.output_folder / "report" / run_name
        water_balance_df_m3_per_timestep: pd.DataFrame = _load_water_balance_dataframe(
            folder
        )
        context_series: dict[str, pd.Series] = _load_contextual_water_balance_series(
            folder
        )

        if water_balance_df_m3_per_timestep.empty:
            raise ValueError("No water balance data available for plotting.")

        signed_water_balance_df_m3_per_timestep: pd.DataFrame = (
            water_balance_df_m3_per_timestep.copy()
        )
        output_columns: list[str] = [
            column_name
            for column_name in signed_water_balance_df_m3_per_timestep.columns
            if column_name.startswith("out_")
        ]
        # Plot outputs below zero so the full balance can be read on a single axis.
        signed_water_balance_df_m3_per_timestep.loc[
            :, output_columns
        ] = -signed_water_balance_df_m3_per_timestep.loc[:, output_columns]

        component_columns: list[str] = list(
            signed_water_balance_df_m3_per_timestep.columns
        )
        component_colors: dict[str, Any] = {
            column_name: mcolormaps["tab20"](
                color_index / max(1, len(component_columns) - 1)
            )
            for color_index, column_name in enumerate(component_columns)
        }
        component_labels: dict[str, str] = {
            column_name: _format_water_balance_component_label(column_name)
            for column_name in component_columns
        }
        total_area_m2: float = _get_total_model_area_m2(self.model)
        conversion_factor_mm_per_m3: float = 1000.0 / total_area_m2
        yearly_context_totals_mm: pd.DataFrame = pd.DataFrame(
            {
                series_name: series.resample("YE").sum() * 1000.0 / total_area_m2
                for series_name, series in context_series.items()
            }
        )
        yearly_context_totals_mm.index = pd.Index(
            [
                pd.Timestamp(timestamp).year
                for timestamp in yearly_context_totals_mm.index
            ]
        )
        context_colors: dict[str, str] = {
            "potential_evapotranspiration": "white",
        }
        context_linestyles: dict[str, str] = {
            "potential_evapotranspiration": ":",
        }
        context_linewidths: dict[str, float] = {
            "potential_evapotranspiration": 1.1,
        }
        context_labels: dict[str, str] = {
            series_name: _format_water_balance_context_label(series_name)
            for series_name in context_series
        }
        yearly_totals_mm: pd.DataFrame | None = None

        yearly_totals_mm = _create_yearly_totals_summary_mm(
            water_balance_df_m3_per_timestep,
            total_area_m2,
        )

        time_index: pd.DatetimeIndex = pd.DatetimeIndex(
            signed_water_balance_df_m3_per_timestep.index
        )
        timestep_label: str = _get_datetime_index_step_label(time_index)
        signed_water_balance_df_mm_per_timestep: pd.DataFrame = (
            signed_water_balance_df_m3_per_timestep * conversion_factor_mm_per_m3
        )
        context_series_mm_per_timestep: dict[str, pd.Series] = {
            series_name: series * conversion_factor_mm_per_m3
            for series_name, series in context_series.items()
        }
        full_figure, full_axis = plt.subplots(figsize=(15, 14), facecolor="#000000")
        _style_water_balance_axis(full_axis)
        for column_name in component_columns:
            full_axis.plot(
                signed_water_balance_df_mm_per_timestep.index,
                signed_water_balance_df_mm_per_timestep[column_name],
                label=component_labels[column_name],
                color=component_colors[column_name],
                linewidth=0.7,
            )
        for series_name, series in context_series_mm_per_timestep.items():
            full_axis.plot(
                series.index,
                series,
                label=context_labels[series_name],
                color=context_colors.get(series_name, "black"),
                linewidth=context_linewidths.get(series_name, 1.0),
                linestyle=context_linestyles.get(series_name, ":"),
                alpha=0.9,
            )

        _format_full_timeseries_axis(
            full_axis,
            time_index,
            f"Water Balance Over Time - {run_name}",
            f"mm/{timestep_label}",
            draw_zero_line=True,
        )
        _add_dark_legend(
            full_axis,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=min(3, len(component_columns)),
            fontsize=9,
        )
        full_figure.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.26)

        year_values: np.ndarray = pd.Series(time_index).dt.year.to_numpy(dtype=int)
        years: list[int] = sorted(np.unique(year_values).tolist())
        yearly_figure, yearly_axes = plt.subplots(
            len(years),
            1,
            figsize=(15, max(7.2 * len(years), 11.0)),
            sharey=True,
            facecolor="#000000",
        )
        if len(years) == 1:
            yearly_axes = [yearly_axes]

        for axis, year in zip(yearly_axes, years, strict=True):
            _style_water_balance_axis(axis)
            year_mask: np.ndarray = year_values == year
            yearly_df_mm_per_timestep: pd.DataFrame = (
                signed_water_balance_df_mm_per_timestep.loc[year_mask]
            )
            for column_name in component_columns:
                axis.plot(
                    yearly_df_mm_per_timestep.index,
                    yearly_df_mm_per_timestep[column_name],
                    color=component_colors[column_name],
                    linewidth=0.55,
                )
            for series_name, series in context_series_mm_per_timestep.items():
                yearly_context_series: pd.Series = series.loc[year_mask]
                axis.plot(
                    yearly_context_series.index,
                    yearly_context_series,
                    color=context_colors.get(series_name, "black"),
                    linewidth=context_linewidths.get(series_name, 1.0),
                    linestyle=context_linestyles.get(series_name, ":"),
                    alpha=0.9,
                )

            _format_yearly_timeseries_axis(
                axis,
                year,
                f"Water Balance Over Time - {year}",
                f"mm/{timestep_label}",
                draw_zero_line=True,
            )
            if yearly_totals_mm is not None:
                _add_yearly_totals_caption(
                    axis,
                    year,
                    yearly_totals_mm,
                    component_labels,
                    yearly_context_totals_mm,
                    context_labels,
                )

        yearly_axes[-1].set_xlabel("Time")
        yearly_handles: list[Line2D] = [
            Line2D([0], [0], color=component_colors[column_name], linewidth=0.9)
            for column_name in component_columns
        ]
        yearly_labels: list[str] = [
            component_labels[column_name] for column_name in component_columns
        ]
        yearly_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=context_colors.get(series_name, "black"),
                    linewidth=context_linewidths.get(series_name, 1.0),
                    linestyle=context_linestyles.get(series_name, ":"),
                )
                for series_name in context_series
            ]
        )
        yearly_labels.extend(
            [context_labels[series_name] for series_name in context_series]
        )
        yearly_figure.legend(
            yearly_handles,
            yearly_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(3, len(yearly_labels)),
            frameon=False,
            labelcolor="white",
        )
        yearly_figure.subplots_adjust(
            left=0.05,
            right=0.98,
            top=0.99,
            bottom=0.12 + 0.03 * max(0, (len(yearly_labels) - 1) // 3),
            hspace=0.7,
        )

        if export:
            full_path: Path = (
                self.water_balance_output_folder / "water_balance_timeseries.svg"
            )
            yearly_path: Path = (
                self.water_balance_output_folder / "water_balance_timeseries_yearly.svg"
            )
            full_figure.savefig(full_path)
            yearly_figure.savefig(yearly_path)
            self.model.logger.info(
                f"Water balance time-series plot saved as: {full_path}"
            )
            self.model.logger.info(
                f"Water balance yearly time-series plot saved as: {yearly_path}"
            )

        plt.show()
        plt.close(full_figure)
        plt.close(yearly_figure)

        top_soil_water_balance_df_m3_per_timestep: pd.DataFrame = (
            _load_top_soil_water_balance_dataframe(folder)
        )
        top_soil_context_series: dict[str, pd.Series] = (
            _load_contextual_top_soil_water_balance_series(folder)
        )

        signed_top_soil_water_balance_df_m3_per_timestep: pd.DataFrame = (
            top_soil_water_balance_df_m3_per_timestep.copy()
        )
        top_soil_output_columns: list[str] = [
            column_name
            for column_name in signed_top_soil_water_balance_df_m3_per_timestep.columns
            if column_name.startswith("out_")
        ]
        signed_top_soil_water_balance_df_m3_per_timestep.loc[
            :, top_soil_output_columns
        ] = -signed_top_soil_water_balance_df_m3_per_timestep.loc[
            :, top_soil_output_columns
        ]

        top_soil_component_columns: list[str] = list(
            signed_top_soil_water_balance_df_m3_per_timestep.columns
        )
        top_soil_component_colors: dict[str, Any] = {
            column_name: mcolormaps["Dark2"](
                color_index / max(1, len(top_soil_component_columns) - 1)
            )
            for color_index, column_name in enumerate(top_soil_component_columns)
        }
        if "storage_change" in top_soil_component_colors:
            top_soil_component_colors["storage_change"] = "black"
        top_soil_component_labels: dict[str, str] = {
            column_name: _get_top_soil_water_balance_label(column_name)
            for column_name in top_soil_component_columns
        }
        top_soil_yearly_context_totals_mm: pd.DataFrame = pd.DataFrame(
            {
                series_name: series.resample("YE").sum() * 1000.0 / total_area_m2
                for series_name, series in top_soil_context_series.items()
            }
        )
        top_soil_yearly_context_totals_mm.index = pd.Index(
            [
                pd.Timestamp(timestamp).year
                for timestamp in top_soil_yearly_context_totals_mm.index
            ]
        )
        top_soil_context_colors: dict[str, str] = {
            "precipitation": "#72b7b2",
            "runoff": "#f58518",
            "snow": "#4c78a8",
            "potential_evapotranspiration": "white",
            "transpiration": "#54a24b",
        }
        top_soil_context_linestyles: dict[str, str] = {
            "precipitation": ":",
            "runoff": "--",
            "snow": "-.",
            "potential_evapotranspiration": ":",
            "transpiration": "-",
        }
        top_soil_context_linewidths: dict[str, float] = {
            "precipitation": 1.0,
            "runoff": 1.0,
            "snow": 1.0,
            "potential_evapotranspiration": 1.1,
            "transpiration": 1.0,
        }
        top_soil_context_labels: dict[str, str] = {
            series_name: _format_water_balance_context_label(series_name)
            for series_name in top_soil_context_series
        }
        top_soil_yearly_totals_mm: pd.DataFrame | None = None
        if total_area_m2 is not None:
            top_soil_yearly_totals_mm = _create_yearly_totals_summary_mm(
                top_soil_water_balance_df_m3_per_timestep,
                total_area_m2,
            )
        top_soil_component_linewidths: dict[str, float] = {
            column_name: 1.0 if column_name == "storage_change" else 0.7
            for column_name in top_soil_component_columns
        }
        top_soil_component_linestyles: dict[str, str] = {
            column_name: "--" if column_name == "storage_change" else "-"
            for column_name in top_soil_component_columns
        }
        top_soil_component_zorders: dict[str, int] = {
            column_name: 4 if column_name == "storage_change" else 2
            for column_name in top_soil_component_columns
        }
        if "storage_change" in top_soil_component_colors:
            top_soil_component_colors["storage_change"] = "white"

        top_soil_time_index: pd.DatetimeIndex = pd.DatetimeIndex(
            signed_top_soil_water_balance_df_m3_per_timestep.index
        )
        top_soil_timestep_label: str = _get_datetime_index_step_label(
            top_soil_time_index
        )
        signed_top_soil_water_balance_df_mm_per_timestep: pd.DataFrame = (
            signed_top_soil_water_balance_df_m3_per_timestep
            * conversion_factor_mm_per_m3
        )
        top_soil_context_series_mm_per_timestep: dict[str, pd.Series] = {
            series_name: series * conversion_factor_mm_per_m3
            for series_name, series in top_soil_context_series.items()
        }
        top_soil_full_figure, top_soil_full_axis = plt.subplots(
            figsize=(15, 13.0), facecolor="#000000"
        )
        _style_water_balance_axis(top_soil_full_axis)
        for column_name in top_soil_component_columns:
            top_soil_full_axis.plot(
                signed_top_soil_water_balance_df_mm_per_timestep.index,
                signed_top_soil_water_balance_df_mm_per_timestep[column_name],
                label=top_soil_component_labels[column_name],
                color=top_soil_component_colors[column_name],
                linewidth=top_soil_component_linewidths[column_name],
                linestyle=top_soil_component_linestyles[column_name],
                zorder=top_soil_component_zorders[column_name],
            )
        for series_name, series in top_soil_context_series_mm_per_timestep.items():
            top_soil_full_axis.plot(
                series.index,
                series,
                label=top_soil_context_labels[series_name],
                color=top_soil_context_colors.get(series_name, "black"),
                linewidth=top_soil_context_linewidths.get(series_name, 1.0),
                linestyle=top_soil_context_linestyles.get(series_name, ":"),
                alpha=0.9,
                zorder=3,
            )

        _format_full_timeseries_axis(
            top_soil_full_axis,
            top_soil_time_index,
            f"Top-Soil Water Balance Over Time - {run_name}",
            f"mm/{top_soil_timestep_label}",
            draw_zero_line=True,
        )
        _add_dark_legend(
            top_soil_full_axis,
            loc="upper right",
            ncol=min(
                3,
                len(top_soil_component_columns) + len(top_soil_context_series),
            ),
            fontsize=9,
        )
        top_soil_full_figure.subplots_adjust(
            left=0.08, right=0.98, top=0.91, bottom=0.14
        )

        top_soil_year_values: np.ndarray = pd.Series(
            top_soil_time_index
        ).dt.year.to_numpy(dtype=int)
        top_soil_years: list[int] = sorted(np.unique(top_soil_year_values).tolist())
        top_soil_yearly_figure, top_soil_yearly_axes = plt.subplots(
            len(top_soil_years),
            1,
            figsize=(15, max(6.4 * len(top_soil_years), 10.0)),
            sharey=True,
            facecolor="#000000",
        )
        if len(top_soil_years) == 1:
            top_soil_yearly_axes = [top_soil_yearly_axes]

        for axis_index, (axis, year) in enumerate(
            zip(top_soil_yearly_axes, top_soil_years, strict=True)
        ):
            _style_water_balance_axis(axis)
            year_mask: np.ndarray = top_soil_year_values == year
            yearly_df_mm: pd.DataFrame = (
                signed_top_soil_water_balance_df_mm_per_timestep.loc[year_mask]
            )
            for column_name in top_soil_component_columns:
                axis.plot(
                    yearly_df_mm.index,
                    yearly_df_mm[column_name],
                    color=top_soil_component_colors[column_name],
                    linewidth=(0.8 if column_name == "storage_change" else 0.55),
                    linestyle=top_soil_component_linestyles[column_name],
                    zorder=top_soil_component_zorders[column_name],
                    label=(
                        top_soil_component_labels[column_name]
                        if axis_index == 0
                        else None
                    ),
                )
            for series_name, series in top_soil_context_series_mm_per_timestep.items():
                yearly_context_series: pd.Series = series.loc[year_mask]
                axis.plot(
                    yearly_context_series.index,
                    yearly_context_series,
                    color=top_soil_context_colors.get(series_name, "black"),
                    linewidth=top_soil_context_linewidths.get(series_name, 1.0),
                    linestyle=top_soil_context_linestyles.get(series_name, ":"),
                    alpha=0.9,
                    zorder=3,
                    label=(
                        top_soil_context_labels[series_name]
                        if axis_index == 0
                        else None
                    ),
                )

            _format_yearly_timeseries_axis(
                axis,
                year,
                f"Top-Soil Water Balance Over Time - {year}",
                f"mm/{top_soil_timestep_label}",
                draw_zero_line=True,
            )
            if top_soil_yearly_totals_mm is not None:
                _add_yearly_totals_caption(
                    axis,
                    year,
                    top_soil_yearly_totals_mm,
                    top_soil_component_labels,
                    top_soil_yearly_context_totals_mm,
                    top_soil_context_labels,
                )

            if axis_index == 0:
                _add_dark_legend(
                    axis,
                    loc="upper right",
                    ncol=min(
                        3,
                        len(top_soil_component_columns) + len(top_soil_context_series),
                    ),
                    fontsize=8.5,
                )

        top_soil_yearly_axes[-1].set_xlabel("Time")
        top_soil_yearly_figure.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.96,
            bottom=0.08,
            hspace=0.68,
        )

        if export:
            top_soil_full_path: Path = (
                self.water_balance_output_folder
                / "water_balance_top_soil_timeseries.svg"
            )
            top_soil_yearly_path: Path = (
                self.water_balance_output_folder
                / "water_balance_top_soil_timeseries_yearly.svg"
            )
            top_soil_full_figure.savefig(top_soil_full_path)
            top_soil_yearly_figure.savefig(top_soil_yearly_path)
            self.model.logger.info(
                f"Top-soil water balance time-series plot saved as: {top_soil_full_path}"
            )
            self.model.logger.info(
                f"Top-soil water balance yearly time-series plot saved as: {top_soil_yearly_path}"
            )

        plt.show()
        plt.close(top_soil_full_figure)
        plt.close(top_soil_yearly_figure)

    def plot_water_storage(
        self,
        run_name: str,
        export: bool = True,
    ) -> None:
        """Plot reported water storage component time series for the full run and per year.

        Notes:
            The currently available storage components come directly from
            `WATER_STORAGE_REPORT_CONFIG` in `geb.reporter`. At present these are the
            reported soil water content layers.

        Args:
            run_name: Name of the run to evaluate.
            export: Whether to export the water storage plots to files.

        Raises:
            ValueError: If the water storage dataframe does not contain any rows.
        """
        folder: Path = self.model.output_folder / "report" / run_name
        water_storage_df_m: pd.DataFrame = _load_water_storage_dataframe(folder)

        if water_storage_df_m.empty:
            raise ValueError("No water storage data available for plotting.")

        component_columns: list[str] = list(water_storage_df_m.columns)
        component_colors: dict[str, Any] = {
            column_name: mcolormaps["viridis"](
                0.15 + 0.75 * color_index / max(1, len(component_columns) - 1)
            )
            for color_index, column_name in enumerate(component_columns)
        }
        component_labels: dict[str, str] = {
            column_name: _format_water_storage_component_label(column_name)
            for column_name in component_columns
        }

        time_index: pd.DatetimeIndex = pd.DatetimeIndex(water_storage_df_m.index)
        full_figure, full_axis = plt.subplots(figsize=(14, 6.5), facecolor="#000000")
        _style_water_balance_axis(full_axis)
        for column_name in component_columns:
            full_axis.plot(
                water_storage_df_m.index,
                water_storage_df_m[column_name],
                label=component_labels[column_name],
                color=component_colors[column_name],
                linewidth=1.6,
            )

        _format_full_timeseries_axis(
            full_axis,
            time_index,
            f"Water Storage Over Time - {run_name}",
            "m",
        )
        _add_dark_legend(
            full_axis,
            loc="upper right",
            ncol=min(2, len(component_columns)),
            fontsize=9,
        )
        full_figure.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.12)

        year_values: np.ndarray = pd.Series(time_index).dt.year.to_numpy(dtype=int)
        years: list[int] = sorted(np.unique(year_values).tolist())
        yearly_figure, yearly_axes = plt.subplots(
            len(years),
            1,
            figsize=(14, max(3.2 * len(years), 5.0)),
            sharey=True,
            facecolor="#000000",
        )
        if len(years) == 1:
            yearly_axes = [yearly_axes]

        for axis_index, (axis, year) in enumerate(zip(yearly_axes, years, strict=True)):
            _style_water_balance_axis(axis)
            year_mask: np.ndarray = year_values == year
            yearly_df_m: pd.DataFrame = water_storage_df_m.loc[year_mask]
            for column_name in component_columns:
                axis.plot(
                    yearly_df_m.index,
                    yearly_df_m[column_name],
                    color=component_colors[column_name],
                    linewidth=1.3,
                    label=component_labels[column_name] if axis_index == 0 else None,
                )

            _format_yearly_timeseries_axis(
                axis,
                year,
                f"Water Storage Over Time - {year}",
                "m",
            )

            if axis_index == 0:
                _add_dark_legend(
                    axis,
                    loc="upper right",
                    ncol=min(2, len(component_columns)),
                    fontsize=8.5,
                )

        yearly_axes[-1].set_xlabel("Time")
        yearly_figure.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.96,
            bottom=0.08,
            hspace=0.26,
        )

        if export:
            full_path: Path = (
                self.water_storage_output_folder / "water_storage_timeseries.svg"
            )
            yearly_path: Path = (
                self.water_storage_output_folder / "water_storage_timeseries_yearly.svg"
            )
            full_figure.savefig(full_path)
            yearly_figure.savefig(yearly_path)
            self.model.logger.info(
                f"Water storage time-series plot saved as: {full_path}"
            )
            self.model.logger.info(
                f"Water storage yearly time-series plot saved as: {yearly_path}"
            )

        plt.show()
        plt.close(full_figure)
        plt.close(yearly_figure)

    @property
    def discharge_output_folder(self) -> Path:
        """Path to the folder where discharge map outputs are stored."""
        folder = self.evaluator.output_folder_evaluate / "hydrology" / "discharge"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def evaluate_discharge_output_folder(self) -> Path:
        """Path to the folder where discharge evaluation outputs are stored."""
        folder = (
            self.evaluator.output_folder_evaluate / "hydrology" / "evaluate_discharge"
        )
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def water_balance_output_folder(self) -> Path:
        """Path to the folder where water balance outputs are stored."""
        folder = self.evaluator.output_folder_evaluate / "hydrology" / "water_balance"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def water_storage_output_folder(self) -> Path:
        """Path to the folder where water storage outputs are stored."""
        folder = self.evaluator.output_folder_evaluate / "hydrology" / "water_storage"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def water_circle_output_folder(self) -> Path:
        """Path to the folder where water circle outputs are stored."""
        folder = self.evaluator.output_folder_evaluate / "hydrology" / "water_circle"
        folder.mkdir(parents=True, exist_ok=True)
        return folder
>>>>>>> cf97c6b046a395cd4914d7d5fbd13329e71ca3ac
