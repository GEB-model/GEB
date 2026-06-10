"""Module implementing hydrology evaluation functions for the GEB model."""

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
import xarray as xr
from matplotlib import colormaps as mcolormaps
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scores.continuous import (
    kge as calculate_kge,
    nse as calculate_nse,
    rmse as calculate_rmse,
)
from scores.continuous.correlation import pearsonr as calculate_pearsonr
from tqdm import tqdm

from geb.evaluate.workflows.dashboard import create_discharge_folium_map
from geb.evaluate.workflows.external_skill_scores import (
    get_geb_station_keys as _get_geb_station_keys,
    prepare_external_evaluation as _prepare_external_evaluation,
    prepare_skill_score_boxplot_inputs as _prepare_skill_score_boxplot_inputs,
    read_external_evaluation_raw as _read_external_evaluation_raw,
)
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
    KGE_modified: float = float("nan")
    KGE_correlation: float = float("nan")
    KGE_bias_ratio: float = float("nan")
    KGE_variability_ratio: float = float("nan")
    NSE: float = float("nan")
    R: float = float("nan")
    R2: float = float("nan")
    RMSE: float = float("nan")
    RRMSE: float = float("nan")


def _recreate_output_folder(output_folder: Path) -> None:
    """Remove an existing output path and recreate it as a directory.

    Args:
        output_folder: Output folder path to recreate.

    """
    if output_folder.is_symlink() or output_folder.is_file():
        # `shutil.rmtree` intentionally refuses symlinks; unlinking preserves
        # the target directory and only removes the compatibility link.
        output_folder.unlink()
    elif output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)


def _resolve_report_folder(output_folder: Path, run_name: str) -> Path:
    """Resolve the report folder for standard and merged-model output layouts.

    Standard model runs store reports under ``output/<run_name>/report``. Older
    merged model outputs store the same run under ``output/report/<run_name>``.

    Args:
        output_folder: Run-specific output folder from the model configuration.
        run_name: Name of the run to evaluate.

    Returns:
        Existing report folder path, or the standard report folder path when no
        compatible report folder exists.

    """
    standard_report_folder: Path = output_folder / "report"
    if standard_report_folder.exists():
        return standard_report_folder

    merged_report_folder: Path = output_folder.parent / "report" / run_name
    if merged_report_folder.exists():
        return merged_report_folder

    return standard_report_folder


def _calculate_discharge_validation_metrics(
    validation_df: pd.DataFrame,
) -> DischargeMetrics:
    """Calculate station-level discharge validation metrics.

    Args:
        validation_df: Validation dataframe with observed and simulated discharge
            columns named `discharge_observations` and `discharge_simulations` (m3/s).

    Returns:
        DischargeMetrics with KGE, modified KGE, its correlation/bias/variability
        components, NSE, R, coefficient of determination R2, RMSE, and RRMSE;
        all NaN when there are fewer than 2 valid pairs.

    Raises:
        TypeError: If the `scores` KGE result does not include component values.
    """
    discharge_columns: list[str] = [
        "discharge_observations",
        "discharge_simulations",
    ]
    valid_pairs_df: pd.DataFrame = validation_df[discharge_columns].dropna()
    if valid_pairs_df.shape[0] < 2:
        return DischargeMetrics()

    # Keep observed and simulated values aligned after dropping incomplete pairs.
    observed_discharge_values: np.ndarray = valid_pairs_df[
        "discharge_observations"
    ].to_numpy(dtype=float)
    simulated_discharge_values: np.ndarray = valid_pairs_df[
        "discharge_simulations"
    ].to_numpy(dtype=float)
    observed_discharge_array: xr.DataArray = xr.DataArray(
        observed_discharge_values, dims=["time"]
    )
    simulated_discharge_array: xr.DataArray = xr.DataArray(
        simulated_discharge_values, dims=["time"]
    )

    observed_discharge_mean: float = float(np.mean(observed_discharge_values))
    simulated_discharge_mean: float = float(np.mean(simulated_discharge_values))
    observed_discharge_std: float = float(np.std(observed_discharge_values))
    simulated_discharge_std: float = float(np.std(simulated_discharge_values))

    # KGE and its original Gupta et al. components are available from `scores`.
    kge_result: xr.DataArray | xr.Dataset = calculate_kge(
        simulated_discharge_array,
        observed_discharge_array,
        include_components=True,
    )
    if not isinstance(kge_result, xr.Dataset):
        raise TypeError("Expected KGE components as an xarray Dataset.")
    kge_components: xr.Dataset = kge_result
    kge: float = float(kge_components["kge"].item())
    kge_correlation: float = float(kge_components["rho"].item())
    kge_bias_ratio: float = float(kge_components["beta"].item())
    kge_variability_ratio: float = float(kge_components["alpha"].item())

    # Modified KGE uses the coefficient-of-variation ratio gamma, which is not
    # exposed by `scores`, so calculate it directly from the original formula.
    observed_discharge_variation: float = (
        float("nan")
        if observed_discharge_mean == 0.0
        else observed_discharge_std / observed_discharge_mean
    )
    simulated_discharge_variation: float = (
        float("nan")
        if simulated_discharge_mean == 0.0
        else simulated_discharge_std / simulated_discharge_mean
    )
    modified_kge_variability_ratio: float = (
        float("nan")
        if observed_discharge_variation == 0.0
        else simulated_discharge_variation / observed_discharge_variation
    )
    kge_modified: float = 1.0 - float(
        np.sqrt(
            (kge_correlation - 1.0) ** 2
            + (kge_bias_ratio - 1.0) ** 2
            + (modified_kge_variability_ratio - 1.0) ** 2
        )
    )

    # Remaining skill scores and error metrics use the same filtered time steps.
    nse: float = float(
        calculate_nse(simulated_discharge_array, observed_discharge_array).item()
    )
    r_value: float = float(
        calculate_pearsonr(
            simulated_discharge_array,
            observed_discharge_array,
        ).item()
    )
    rmse: float = float(
        calculate_rmse(simulated_discharge_array, observed_discharge_array).item()
    )
    rrmse: float = (
        float("nan") if observed_discharge_std == 0.0 else rmse / observed_discharge_std
    )

    # R2 is the coefficient of determination, not squared Pearson correlation.
    residual_sum_of_squares: float = float(
        np.sum((observed_discharge_values - simulated_discharge_values) ** 2)
    )
    total_sum_of_squares: float = float(
        np.sum((observed_discharge_values - np.mean(observed_discharge_values)) ** 2)
    )
    coefficient_of_determination: float = (
        float("nan")
        if total_sum_of_squares == 0.0
        else 1.0 - residual_sum_of_squares / total_sum_of_squares
    )

    return DischargeMetrics(
        KGE=kge,
        KGE_modified=kge_modified,
        KGE_correlation=kge_correlation,
        KGE_bias_ratio=kge_bias_ratio,
        KGE_variability_ratio=kge_variability_ratio,
        NSE=nse,
        R=r_value,
        R2=coefficient_of_determination,
        RMSE=rmse,
        RRMSE=rrmse,
    )


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
    report_folder: Path = _resolve_report_folder(output_folder, run_name)
    routing_dir: Path = report_folder / "hydrology.routing"
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
    report_folder = _resolve_report_folder(output_folder, run_name)
    frozen_fraction_series_name: str = "_top_soil_frozen_fraction"
    frozen_fraction_series: pd.Series | None = None
    frozen_fraction_path: Path = (
        report_folder / "hydrology.landsurface" / frozen_fraction_series_name
    ).with_suffix(".parquet")
    if frozen_fraction_path.exists():
        frozen_fraction_series = _read_evaluation_series_with_date_index(
            report_folder,
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
    report_folder: Path = _resolve_report_folder(output_folder, run_name)
    routing_dir = report_folder / "hydrology.routing"
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
        discharge_folder: Path = (
            self.evaluator.output_folder_evaluate.parent
            / "report"
            / "hydrology.routing"
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
        _recreate_output_folder(self.evaluate_discharge_output_folder)

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

        report_folder: Path = _resolve_report_folder(self.model.output_folder, run_name)
        if not report_folder.exists():
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

                validation_df: pd.DataFrame = create_validation_df(
                    self.model.output_folder,
                    run_name,
                    station_id,
                    observed_discharge_series,
                    correct_discharge_observations,
                    discharge_observations_to_GEB_upstream_area_ratio,
                )

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
        """Read external model evaluation CSVs without matching to GEB stations.

        Source-specific filters, such as the Utrecht GRDC station criteria, are
        applied before the external scores are returned.

        Args:
            external_evaluation_folder: Directory with one CSV per external model.
                Defaults to the configured folder, resolved from the model folder
                when relative.

        Returns:
            Mapping from model label to prepared external DataFrame (index =
            station name, columns = metrics and optional metadata).
        """
        configured_folder: str | Path | None = self.model.config["hydrology"][
            "evaluation"
        ]["discharge"].get("external_evaluation_folder")
        external_models: dict[str, pd.DataFrame] = _read_external_evaluation_raw(
            external_evaluation_folder=external_evaluation_folder,
            configured_external_evaluation_folder=configured_folder,
            model_folder=self.model.input_folder.parent,
            logger=self.model.logger,
        )
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
        external_models: dict[str, pd.DataFrame] = self._read_external_evaluation_raw(
            external_evaluation_folder
        )
        if not external_models:
            self.model.logger.info("No external evaluation data found, skipping.")
            return {}

        geb_xlsx = self.evaluate_discharge_output_folder / "evaluation_metrics.xlsx"
        station_keys: set[str] = _get_geb_station_keys(
            evaluation_metrics_path=geb_xlsx,
            snapped_locations_path=self.model.files["geom"][
                "discharge/discharge_snapped_locations"
            ],
        )

        return _prepare_external_evaluation(
            external_models=external_models,
            station_keys=station_keys,
            output_folder=self.evaluate_discharge_output_folder,
            logger=self.model.logger,
        )

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
        external_evaluation_folder: str | Path | None = None,
        include_external: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create skill score violin+boxplot graphs for each evaluation metric.

        Produces a 2×3 grid of violin/box plots across gauging stations.
        When ``matched_only=False`` (default), the plot is GEB-only unless
        ``include_external=True`` is passed. If external evaluation data are
        available, a matched-stations plot is generated automatically as an
        additional output. When ``matched_only=True``, only the matched-stations
        plot is generated for backwards compatibility.

        Args:
            export: Save the figure to disk.
            include_geb: Include GEB in the plot.
            matched_only: Restrict all models to overlapping stations only.
            minimum_upstream_area_km2: Optional minimum modeled upstream area threshold for plotted GEB stations (km2).
                If omitted, `hydrology.evaluation.discharge.minimum_upstream_area_km2` is used.
            external_evaluation_folder: Optional folder containing external
                model skill scores or Google streamflow `metrics.tgz`.
            include_external: Include external model scores in non-matched plots.
                External scores are always included for `matched_only=True` and
                when `include_geb=False`.
            **kwargs: Ignored (CLI compatibility).
        """
        if minimum_upstream_area_km2 is None:
            minimum_upstream_area_km2 = self.model.config["hydrology"]["evaluation"][
                "discharge"
            ]["minimum_upstream_area_km2"]
        configured_folder: str | Path | None = self.model.config["hydrology"][
            "evaluation"
        ]["discharge"].get("external_evaluation_folder")
        should_include_external: bool = (
            include_external or matched_only or not include_geb
        )
        evaluation_metrics_path: Path = (
            self.evaluate_discharge_output_folder / "evaluation_metrics.xlsx"
        )
        snapped_locations_path: Path = self.model.files["geom"][
            "discharge/discharge_snapped_locations"
        ]
        evaluation_df, external_models = _prepare_skill_score_boxplot_inputs(
            evaluation_metrics_path=evaluation_metrics_path,
            snapped_locations_path=snapped_locations_path,
            external_evaluation_folder=external_evaluation_folder,
            configured_external_evaluation_folder=configured_folder,
            model_folder=self.model.input_folder.parent,
            output_folder=self.evaluate_discharge_output_folder,
            logger=self.model.logger,
            minimum_upstream_area_km2=minimum_upstream_area_km2,
            include_geb=include_geb,
            matched_only=matched_only,
            include_external=should_include_external,
        )

        if include_geb and evaluation_df.empty:
            self.model.logger.info(
                "No discharge stations found for evaluation. Skipping skill score graphs."
            )
            return

        _plot_skill_score_boxplots(
            evaluation_df=evaluation_df,
            external_models=external_models,
            output_folder=self.evaluate_discharge_output_folder,
            logger=self.model.logger,
            export=export,
            include_geb=include_geb,
            matched_only=matched_only,
        )

        if not matched_only and include_geb:
            matched_evaluation_df, matched_external_models = (
                _prepare_skill_score_boxplot_inputs(
                    evaluation_metrics_path=evaluation_metrics_path,
                    snapped_locations_path=snapped_locations_path,
                    external_evaluation_folder=external_evaluation_folder,
                    configured_external_evaluation_folder=configured_folder,
                    model_folder=self.model.input_folder.parent,
                    output_folder=self.evaluate_discharge_output_folder,
                    logger=self.model.logger,
                    minimum_upstream_area_km2=minimum_upstream_area_km2,
                    include_geb=include_geb,
                    matched_only=True,
                    include_external=True,
                )
            )
            if matched_external_models and not matched_evaluation_df.empty:
                _plot_skill_score_boxplots(
                    evaluation_df=matched_evaluation_df,
                    external_models=matched_external_models,
                    output_folder=self.evaluate_discharge_output_folder,
                    logger=self.model.logger,
                    export=export,
                    include_geb=include_geb,
                    matched_only=True,
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
        folder = self.model.output_folder / "report"

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

    def plot_water_balance(
        self,
        run_name: str,
        export: bool = True,
    ) -> None:
        """Create a csv file and plot showing the water balance components.

        Args:
            run_name: Name of the run to evaluate.
            export: Whether to export the water balance plot to a file.

        Notes:
            Potential evapotranspiration is shown as an optional context bar when
            the corresponding report output is available. It is not included in
            the actual water balance totals.

        Raises:
            ValueError: If the water balance dataframe does not contain any rows.
        """
        folder = self.model.output_folder / "report"
        df_m3_per_timestep: pd.DataFrame = _load_water_balance_dataframe(folder)
        context_series: dict[str, pd.Series] = _load_contextual_water_balance_series(
            folder
        )
        df_yearly: pd.DataFrame = df_m3_per_timestep.resample("YE").sum()
        df_yearly.to_csv(folder / "water_balance_yearly.csv")
        self.model.logger.info("Water balance yearly values saved.")

        years: pd.Index = df_yearly.index.year  # ty:ignore[unresolved-attribute]
        n_years: int = len(years)

        fig, axes = plt.subplots(n_years, 1, figsize=(16, 4 * n_years), sharex=True)
        if n_years == 1:
            axes = [axes]

        inputs_cols = [c for c in df_yearly.columns if c.startswith("in_")]
        outputs_cols = [c for c in df_yearly.columns if c.startswith("out_")]
        storage_cols = [c for c in df_yearly.columns if "storage" in c.lower()]
        yearly_context_series: dict[str, pd.Series] = {
            series_name: series.resample("YE").sum()
            for series_name, series in context_series.items()
        }

        # legend building
        legend_handles = []
        legend_labels = []

        # Colormaps
        input_cmap = mcolormaps["Blues"]
        output_cmap = mcolormaps["Set3"]
        storage_cmap = mcolormaps["Greens"]

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
            row = df_yearly.loc[df_yearly.index.year == year].iloc[0]  # ty:ignore[unresolved-attribute]

            bottom = 0
            for col in inputs_cols:
                label = col.replace("in_", "").replace("_", " ")
                bar_container = ax.bar(
                    "inputs",
                    row[col],
                    bottom=bottom,
                    color=input_colors[col],
                )
                add_legend_entry(bar_container[0], f"input • {label}")
                bottom += row[col]

            bottom = 0
            for col in outputs_cols:
                label = col.replace("out_", "").replace("_", " ")
                bar_container = ax.bar(
                    "outputs",
                    row[col],
                    bottom=bottom,
                    color=output_colors[col],
                )
                add_legend_entry(bar_container[0], f"output • {label}")
                bottom += row[col]

            for col in storage_cols:
                label = col.replace("_", " ")
                bar_container = ax.bar(
                    "storage",
                    row[col],
                    color=storage_colors[col],
                )
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

            ax.set_title(f"Water Balance – {year}")
            ax.set_ylabel("m3/year")

        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=4,
        )

        if export:
            fig_path = (
                self.water_balance_output_folder / "water_balance_yearly_subplots.svg"
            )
            plt.savefig(fig_path)
            self.model.logger.info(f"Water balance yearly plot saved as: {fig_path}")

        plt.show()
        plt.close(fig)

        folder: Path = self.model.output_folder / "report"
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
        folder: Path = self.model.output_folder / "report"
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
