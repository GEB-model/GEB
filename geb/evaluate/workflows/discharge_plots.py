"""Draw discharge time series, return periods, maps, and score boxplots. The plots for GRDC-Caravan discharge characteristics are made in a seperate script (discharge_characteristics.py)."""

import logging
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import contextily as ctx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from geb.evaluate.workflows import discharge_characteristics, external_skill_scores
from geb.evaluate.workflows.discharge_helpers import (
    DischargeEvaluationPaths,
    _get_discharge_evaluation_paths,
)
from geb.evaluate.workflows.discharge_metrics import (
    SKILL_SCORE_PLOT_CONFIG_BY_COLUMN,
    SKILL_SCORE_PLOT_CONFIGS,
    use_daily_discharge_scores,
)
from geb.evaluate.workflows.external_skill_scores import (
    GLOFAS_MODEL_NAME,
    GOOGLE_MODEL_NAME,
    PCRGLOBWB_MODEL_NAME,
)
from geb.workflows.extreme_value_analysis import ReturnPeriodModel
from geb.workflows.io import read_geom

if TYPE_CHECKING:
    from geb.evaluate.hydrology import Hydrology

OBSERVATIONS_COLOR: str = "#E6900A"

SIMULATIONS_COLOR: str = "#278DD9"

LINE_WIDTH: float = 1.3

LINE_COLOR: str = "#111111"

_EXTERNAL_MODEL_PLOT_ORDER: dict[str, int] = {
    PCRGLOBWB_MODEL_NAME: 0,
    GOOGLE_MODEL_NAME: 1,
    GLOFAS_MODEL_NAME: 2,
}

_EXTERNAL_MODEL_DISPLAY_NAMES: dict[str, str] = {
    PCRGLOBWB_MODEL_NAME: "PCR-GLOBWB",
    GOOGLE_MODEL_NAME: "Google LSTM",
    GLOFAS_MODEL_NAME: "GloFAS v4.0",
}


def plot_discharge(
    self: Hydrology,
    run_name: str = "default",
    include_outflow_plots: bool = False,
) -> None:
    """Plot the mean discharge map, optionally including outflow diagnostics.

    Creates a spatial visualization of mean discharge values over time from
    the GEB model simulation results. Outflow diagnostics include full-period,
    yearly, and return-period plots for each exported outflow location.

    Notes:
        The discharge data must exist in the report directory structure. If the discharge
        file is not found, a FileNotFoundError will be raised. The mean is calculated
        across the entire simulation time period.

    Args:
        self: Hydrology evaluator providing model settings and output paths.
        run_name: Name of the simulation run to plot. Must correspond to an existing
            run directory in the model output folder.
        include_outflow_plots: Whether to write per-outflow diagnostics.
            Defaults to False to avoid creating many files for large regions.
    """
    if self.discharge_output_folder.exists():
        shutil.rmtree(self.discharge_output_folder)
    self.discharge_output_folder.mkdir(parents=True, exist_ok=True)

    rivers_of_interest: gpd.GeoDataFrame
    discharge: pd.DataFrame
    rivers_of_interest, discharge = self.get_discharge_per_river(run_name)
    river_id: int
    for river_id in discharge.columns:
        rivers_of_interest.loc[river_id, "discharge_m3_per_s"] = discharge[
            river_id
        ].mean()

    ax: plt.Axes = rivers_of_interest.plot(
        column="discharge_m3_per_s",
        cmap="Blues",
        legend=True,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Mean discharge (m3/s)")

    plt.savefig(
        self.discharge_output_folder / "mean_discharge_m3_per_s_map.svg",
    )
    plt.close()

    if include_outflow_plots:
        run_output_folder: Path = (
            Path(self.model.config["general"]["output_folder"]) / run_name
        )
        report_folder: Path = run_output_folder / "report"
        routing_folder: Path = report_folder / "hydrology.routing"
        if not routing_folder.exists():
            self.model.logger.info(
                "No hydrology routing directory found at %s. Skipping outflow plots.",
                routing_folder,
            )
            return
        consolidated_outflow_file: Path = (
            routing_folder / "river_outflow_hourly_m3_per_s.parquet"
        )
        outflow_files: list[Path] = sorted(
            routing_folder.glob("river_outflow_hourly_m3_per_s_*.parquet")
        )
        if not outflow_files and not consolidated_outflow_file.exists():
            self.model.logger.info(
                "No exported outflow time series found. Skipping outflow plots."
            )
            return
        frozen_fraction_path: Path = (
            report_folder
            / "hydrology.landsurface"
            / "_top_soil_frozen_fraction.parquet"
        )
        frozen_fraction: pd.Series | None = (
            pd.read_parquet(frozen_fraction_path)["_top_soil_frozen_fraction"]
            if frozen_fraction_path.exists()
            else None
        )
        outflow_plot_count: int = save_outflow_discharge_plots(
            outflow_files=outflow_files,
            total_area_m2=self.model.total_area_m2,
            outflow_plot_folder=self.discharge_output_folder / "outflow",
            logger=self.model.logger,
            frozen_fraction_series=frozen_fraction,
            consolidated_outflow_file=consolidated_outflow_file,
        )
        if outflow_plot_count > 0:
            self.model.logger.info(
                "Created %d outflow discharge plots.", outflow_plot_count
            )


def plot_discharge_skill_scores(
    self: Hydrology,
    export: bool = True,
    minimum_upstream_area_km2: float | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    *,
    plots: tuple[str, ...] = (
        "maps",
        "boxplots",
        "upstream_area",
        "characteristics",
    ),
    **kwargs: Any,
) -> None:
    """Load scores once and create the requested discharge evaluation figures.

    All plot commands use this workflow so station selection, daily score
    columns, and external model matching are shared. Figure calculations and
    filenames are the same for individual commands and the complete set.

    Args:
        self: Hydrology evaluator providing model settings and output paths.
        export: Whether to save figures. Boxplots still export matched tables.
        minimum_upstream_area_km2: Minimum modeled upstream area (km²), or None
            to use the discharge-evaluation configuration.
        start_year: First calendar year of the saved evaluation, or None.
        end_year: Last calendar year of the saved evaluation, or None.
        plots: Plot groups: maps, boxplots, upstream_area, characteristics.
            The former name "distributions" is accepted as an alias for "boxplots".
        **kwargs: Ignored CLI compatibility options.

    Raises:
        ValueError: If an unknown plot group is requested.
    """
    plots = tuple(
        "boxplots" if plot_group == "distributions" else plot_group
        for plot_group in plots
    )
    if set(plots) - {"maps", "boxplots", "upstream_area", "characteristics"}:
        raise ValueError(f"Unknown discharge plot groups: {plots}")
    if minimum_upstream_area_km2 is None:
        minimum_upstream_area_km2 = self.model.config["hydrology"]["evaluation"][
            "discharge"
        ]["minimum_upstream_area_km2"]
    evaluation_paths: DischargeEvaluationPaths = _get_discharge_evaluation_paths(
        self.evaluate_discharge_output_folder, start_year, end_year
    )
    if evaluation_paths.metrics_geoparquet.exists():
        station_scores: gpd.GeoDataFrame = gpd.read_parquet(
            evaluation_paths.metrics_geoparquet
        )
    elif evaluation_paths.metrics_excel.exists():
        saved_station_scores: pd.DataFrame = pd.read_excel(
            evaluation_paths.metrics_excel
        ).set_index("station_ID")
        station_scores = gpd.GeoDataFrame(
            saved_station_scores,
            geometry=gpd.points_from_xy(
                saved_station_scores["station_longitude"],
                saved_station_scores["station_latitude"],
            ),
            crs="EPSG:4326",
        )
    else:
        self.model.logger.warning(
            "No discharge scores found for %s. Run evaluate_discharge first.",
            evaluation_paths.label,
        )
        return
    station_count_before_filter: int = len(station_scores)
    station_scores = station_scores.loc[
        station_scores["upstream_area_GEB"] >= minimum_upstream_area_km2 * 1e6
    ].copy()
    self.model.logger.info(
        "Retained %d/%d stations with upstream area of at least %g km².",
        len(station_scores),
        station_count_before_filter,
        minimum_upstream_area_km2,
    )
    if station_scores.empty:
        return

    # Attribute analysis consumes the original score columns, before daily
    # metrics replace the generic plotting columns on a separate copy.
    daily_station_scores: gpd.GeoDataFrame = station_scores.copy()
    use_daily_discharge_scores(daily_station_scores)
    matched_scores_by_model: dict[str, external_skill_scores.MatchedSkillScores] = {}
    if ("maps" in plots and export) or "boxplots" in plots:
        matched_scores_by_model = external_skill_scores.match_external_skill_scores(
            station_scores=daily_station_scores,
            external_models=external_skill_scores.load_external_skill_scores(
                input_folder=self.model.input_folder,
                logger=self.model.logger,
            ),
            output_folder=evaluation_paths.plot_folder,
            logger=self.model.logger,
        )
    if "maps" in plots and export:
        plot_skill_score_maps(
            mapped_station_scores=daily_station_scores,
            region_geom=read_geom(self.model.files["geom"]["mask"]),
            output_folder=evaluation_paths.plot_folder,
            logger=self.model.logger,
            score_differences_by_model={
                model_name: matched_scores.geb_scores
                for model_name, matched_scores in matched_scores_by_model.items()
            },
        )
    if "boxplots" in plots:
        boxplot_folder: Path = evaluation_paths.plot_folder / "skill_score_boxplots"
        upstream_area_filename_suffix: str = (
            f"_upstream_area_{minimum_upstream_area_km2:g}km2".replace(".", "p")
            if minimum_upstream_area_km2 > 0
            else ""
        )
        plot_skill_score_boxplots(
            panels=_prepare_boxplot_panels({"GEB": daily_station_scores}),
            output_path=boxplot_folder
            / f"evaluation_skill_scores{upstream_area_filename_suffix}_n{len(daily_station_scores)}",
            export=export,
        )
        seasons: tuple[str, ...] = ("winter", "spring", "summer", "autumn")
        scores_by_season: dict[str, pd.DataFrame] = {
            season.title(): daily_station_scores.filter(
                regex=f"^KGE.*_daily_{season}$"
            ).rename(columns=lambda column: column.removesuffix(f"_daily_{season}"))
            for season in seasons
        }
        plot_skill_score_boxplots(
            panels=_prepare_boxplot_panels(scores_by_season),
            output_path=boxplot_folder / "evaluation_skill_scores_kge_seasonal",
            export=export,
        )
        external_kge_panels: list[tuple[str, str, pd.DataFrame]] = []
        model_name: str
        matched_scores: external_skill_scores.MatchedSkillScores
        for model_name in sorted(
            matched_scores_by_model,
            key=lambda name: (_EXTERNAL_MODEL_PLOT_ORDER.get(name, 99), name.lower()),
        ):
            matched_scores = matched_scores_by_model[model_name]
            model_filename_suffix: str = re.sub(
                r"[^a-z0-9]+", "_", model_name.lower()
            ).strip("_")
            matched_boxplot_panels: list[tuple[str, str, pd.DataFrame]] = (
                _prepare_boxplot_panels(
                    {
                        "GEB": matched_scores.geb_scores,
                        model_name: matched_scores.external_scores,
                    },
                    require_matching_stations=True,
                )
            )
            plot_skill_score_boxplots(
                panels=matched_boxplot_panels,
                output_path=boxplot_folder
                / f"evaluation_skill_scores_matched_{model_filename_suffix}{upstream_area_filename_suffix}_n{len(matched_scores.geb_scores)}",
                export=export,
            )
            metric: str
            values: pd.DataFrame
            for metric, _, values in matched_boxplot_panels:
                if metric == "KGE":
                    title: str = (
                        f"{_EXTERNAL_MODEL_DISPLAY_NAMES.get(model_name, model_name)}"
                        f"\nn={len(values)}"
                    )
                    if minimum_upstream_area_km2 > 0:
                        title += f"\nUpstream area ≥ {minimum_upstream_area_km2:g} km²"
                    external_kge_panels.append((metric, title, values))
        plot_skill_score_boxplots(
            panels=external_kge_panels,
            output_path=boxplot_folder
            / "evaluation_skill_scores_kge_external_comparison",
            export=export,
            extensions=("svg", "png"),
        )
    if "upstream_area" in plots and export:
        discharge_characteristics.plot_skill_scores_vs_upstream_area(
            station_scores=daily_station_scores,
            output_folder=evaluation_paths.plot_folder,
            logger=self.model.logger,
        )
    if "characteristics" in plots:
        discharge_characteristics.analyze_discharge_characteristics(
            station_scores=station_scores,
            output_folder=evaluation_paths.plot_folder / "skill_score_explanations",
            logger=self.model.logger,
            output_name_suffix=evaluation_paths.suffix,
            export=export,
        )


def save_discharge_timeseries_plots(
    station_id: Any,
    discharge_comparison: pd.DataFrame,
    upstream_area_ratio: float,
    metrics: Mapping[str, float],
    plot_folder: Path,
    export_yearly_timeseries_plots: bool,
) -> None:
    """Save full-period and optional yearly station discharge plots.

    Args:
        station_id: Station identifier used in output filenames.
        discharge_comparison: Observed and simulated discharge time series (m3/s).
        upstream_area_ratio: Observed-to-modeled upstream-area ratio
            (dimensionless).
        metrics: Discharge validation metrics keyed by metric name.
        plot_folder: Evaluation plot output folder.
        export_yearly_timeseries_plots: Whether to save one PNG for each calendar year.
    """
    timeseries_folder: Path = plot_folder / "timeseries"
    timeseries_folder.mkdir(parents=True, exist_ok=True)
    figure: plt.Figure = _create_discharge_timeseries_figure(
        discharge_comparison=discharge_comparison,
        upstream_area_ratio=upstream_area_ratio,
        metrics=metrics,
        include_mean=True,
    )
    figure.savefig(timeseries_folder / f"timeseries_plot_{station_id}.png", dpi=300)
    plt.close(figure)

    if export_yearly_timeseries_plots:
        yearly_groups: Any = discharge_comparison.groupby(
            discharge_comparison.index.to_series().dt.year.to_numpy()
        )
        for year, yearly_discharge_comparison in yearly_groups:
            year_value: int = int(year)
            yearly_figure: plt.Figure = _create_discharge_timeseries_figure(
                discharge_comparison=yearly_discharge_comparison,
                upstream_area_ratio=upstream_area_ratio,
                metrics=metrics,
                include_mean=False,
            )
            yearly_figure.savefig(
                timeseries_folder / f"timeseries_plot_{station_id}_{year_value}.png",
                dpi=300,
            )
            plt.close(yearly_figure)


def _create_discharge_timeseries_figure(
    discharge_comparison: pd.DataFrame,
    upstream_area_ratio: float,
    metrics: Mapping[str, float],
    include_mean: bool,
) -> plt.Figure:
    """Create a discharge comparison figure.

    Args:
        discharge_comparison: Observed and simulated discharge time series (m3/s).
        upstream_area_ratio: Observed-to-modeled upstream-area ratio
            (dimensionless).
        metrics: Discharge validation metrics keyed by metric name.
        include_mean: Whether to show mean simulated discharge (m3/s).

    Returns:
        Matplotlib figure containing the discharge comparison.
    """
    figure, axis = plt.subplots(figsize=(13, 4))
    for column_name, label, color in (
        ("discharge_simulations", "Simulated", SIMULATIONS_COLOR),
        ("discharge_observations", "Observed", OBSERVATIONS_COLOR),
    ):
        axis.plot(
            discharge_comparison.index,
            discharge_comparison[column_name],
            label=label,
            linewidth=0.5,
            color=color,
        )
    axis.set(
        xlabel="Time",
        ylabel="Discharge [m3/s]",
        xlim=(discharge_comparison.index.min(), discharge_comparison.index.max()),
        ylim=(0, None),
    )
    axis.legend(loc="upper right", fontsize=10)

    r2_value: float = metrics["R2"]
    metric_labels: list[str] = (
        [
            f"$r^2$={r2_value:.2f}",
            f"KGE={metrics['KGE']:.2f}",
            f"NSE={metrics['NSE']:.2f}",
        ]
        if np.isfinite(r2_value)
        else ["No overlapping values for score metrics"]
    )
    if include_mean:
        metric_labels.append(
            f"Mean={discharge_comparison['discharge_simulations'].mean():.2f}"
        )
    metric_labels.append(f"upstream area ratio: {upstream_area_ratio:.2f}")
    for row, label in enumerate(metric_labels):
        axis.text(
            0.02,
            0.9 - row * 0.05,
            label,
            transform=axis.transAxes,
            fontsize=12,
        )
    return figure


def save_station_return_period_plots(
    discharge_comparison: pd.DataFrame,
    station_id: str | int,
    eval_plot_folder: Path,
) -> None:
    """Save station return-level curves and observed/simulated fit diagnostics.

    Args:
        discharge_comparison: Observed and simulated discharge columns (m³/s).
        station_id: Station identifier for filenames.
        eval_plot_folder: Root output directory; files go in return_periods.
    """
    # Compare extremes only over observed intervals. Fixing shape at zero
    # stabilizes the fits for short evaluation records.
    simulated: pd.Series = discharge_comparison["discharge_simulations"].where(
        discharge_comparison["discharge_observations"].notna()
    )
    models: list[tuple[ReturnPeriodModel, str, str]] = [
        (
            ReturnPeriodModel(
                series=series,
                return_periods=[2, 5, 10, 25, 50, 100],
                fixed_shape=0.0,
                selection_strategy="first_significant",
            ),
            label,
            color,
        )
        for series, label, color in (
            (
                discharge_comparison["discharge_observations"],
                "Observed",
                OBSERVATIONS_COLOR,
            ),
            (simulated, "Simulated", SIMULATIONS_COLOR),
        )
    ]
    return_periods_folder: Path = eval_plot_folder / "return_periods"
    return_periods_folder.mkdir(parents=True, exist_ok=True)
    simple_figure: plt.Figure
    fit_axis: plt.Axes
    simple_figure, fit_axis = plt.subplots(figsize=(14, 4))
    for model, label, color in models:
        model.plot_fit(ax=fit_axis, label_prefix=label, color=color)
    simple_figure.savefig(
        return_periods_folder / f"return_period_fit_{station_id}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(simple_figure)

    figure: plt.Figure = plt.figure(figsize=(24, 20))
    grid: GridSpec = figure.add_gridspec(5, 2)
    fit_axis = figure.add_subplot(grid[0, :])
    for model, label, color in models:
        model.plot_fit(ax=fit_axis, label_prefix=label, color=color)
    for column, (model, label, _) in enumerate(models):
        diagnostics_grid: GridSpecFromSubplotSpec = grid[1:, column].subgridspec(4, 2)
        goodness_axes: list[plt.Axes] = [
            figure.add_subplot(diagnostics_grid[row, col])
            for row, col in ((0, 0), (0, 1), (1, 0))
        ]
        model.plot_gof(axes=goodness_axes)
        for axis in goodness_axes:
            axis.set_title(f"{label[:3]}: {axis.get_title()}", fontsize=10)
        selection_axes: list[plt.Axes] = [
            figure.add_subplot(diagnostics_grid[row, col])
            for row, col in ((1, 1), (2, 0), (2, 1), (3, 0))
        ]
        model.plot_selection_diagnostics(axes=selection_axes)
    figure.tight_layout()
    figure.savefig(
        return_periods_folder / f"return_period_validation_{station_id}.svg",
        bbox_inches="tight",
    )
    plt.close(figure)


def save_outflow_discharge_plots(
    outflow_files: list[Path],
    total_area_m2: float,
    outflow_plot_folder: Path,
    logger: logging.Logger,
    frozen_fraction_series: pd.Series | None = None,
    consolidated_outflow_file: Path | None = None,
) -> int:
    """Save full-period, yearly, and return-period plots for river outlets.

    Args:
        outflow_files: Hourly discharge Parquet files (m³/s).
        total_area_m2: Basin area used for equivalent outflow depth (m²).
        outflow_plot_folder: Directory receiving outlet figures.
        logger: Logger for skipped empty reports.
        frozen_fraction_series: Optional time-indexed basin frozen fraction (0–1).
        consolidated_outflow_file: Optional consolidated hourly discharge Parquet file (m³/s).

    Returns:
        Number of outlets plotted; all-NaN discharge reports are skipped.

    Raises:
        ValueError: If basin area is not positive and finite.
    """
    if not np.isfinite(total_area_m2) or total_area_m2 <= 0:
        raise ValueError("Basin area must be finite and positive.")
    outflow_plot_folder.mkdir(parents=True, exist_ok=True)
    if frozen_fraction_series is not None:
        frozen_fraction_series = frozen_fraction_series.sort_index()
        frozen_fraction_series = frozen_fraction_series.loc[
            ~frozen_fraction_series.index.duplicated(keep="last")
        ]
    frozen_fraction_cmap: mcolors.Colormap = mcolors.LinearSegmentedColormap.from_list(
        "top_soil_frozen_fraction",
        ["#1f77b4", "#ffffff"],
    )

    outflow_items: list[tuple[str, str, pd.Series]] = []
    if consolidated_outflow_file is not None and consolidated_outflow_file.exists():
        consolidated_df: pd.DataFrame = pd.read_parquet(consolidated_outflow_file)
        for col in consolidated_df.columns:
            outflow_items.append(
                (
                    f"river_outflow_hourly_m3_per_s_{col}",
                    str(col),
                    consolidated_df[col],
                )
            )
    for outflow_file in outflow_files:
        stem: str = outflow_file.stem
        outlet_id: str = stem.replace("river_outflow_hourly_m3_per_s_", "")
        outflow_items.append(
            (stem, outlet_id, pd.read_parquet(outflow_file).iloc[:, 0])
        )

    plots_created: int = 0
    for file_stem, outlet_id, outflow_series in outflow_items:
        if np.isnan(outflow_series.values).all():
            logger.info(f"Outflow for outlet {outlet_id} contains only NaN values.")
            continue
        aligned_frozen_fraction_percent: pd.Series | None = None
        if frozen_fraction_series is not None:
            # Repeat the latest daily context value across the hourly outflow data.
            aligned_frozen_fraction_percent = frozen_fraction_series.reindex(
                pd.DatetimeIndex(outflow_series.index), method="ffill"
            )
            aligned_frozen_fraction_percent = (
                aligned_frozen_fraction_percent.bfill() * 100.0
            )

        fig, ax = plt.subplots(figsize=(7, 4))
        _draw_outflow_series(
            ax,
            outflow_series,
            aligned_frozen_fraction_percent,
            frozen_fraction_cmap,
            linewidth=1.1,
            color=SIMULATIONS_COLOR,
        )
        ax.set_ylabel("Discharge [m3/s]")
        ax.set_xlabel("Time")
        ax.legend(
            handles=[Line2D([0], [0], color=SIMULATIONS_COLOR, linewidth=1.1)],
            labels=["GEB outflow simulation (blue = unfrozen, grey = fully frozen)"],
        )
        ax.set_title(
            f"GEB river outflow for outlet {outlet_id}, mean: {outflow_series.mean():.2f} m3/s"
        )

        plt.savefig(
            outflow_plot_folder / f"{file_stem}.svg",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.show()
        plt.close(fig)

        outflow_time_index: pd.DatetimeIndex = pd.DatetimeIndex(outflow_series.index)
        timestep_seconds: float = float(
            pd.Timedelta(
                pd.tseries.frequencies.to_offset(str(outflow_time_index.inferred_freq))
            ).total_seconds()
        )
        outflow_year_values: np.ndarray = pd.Series(
            outflow_time_index
        ).dt.year.to_numpy(dtype=int)
        outflow_years: list[int] = sorted(np.unique(outflow_year_values).tolist())
        yearly_figure, yearly_axes = plt.subplots(
            len(outflow_years),
            1,
            figsize=(10, max(3.2 * len(outflow_years), 4.5)),
            sharey=True,
        )
        if len(outflow_years) == 1:
            yearly_axes = [yearly_axes]

        for axis, year in zip(yearly_axes, outflow_years, strict=True):
            yearly_mask: np.ndarray = outflow_year_values == year
            yearly_outflow_series: pd.Series = outflow_series.loc[yearly_mask]
            yearly_frozen_fraction_percent: pd.Series | None = None
            if aligned_frozen_fraction_percent is not None:
                yearly_frozen_fraction_percent = aligned_frozen_fraction_percent.loc[
                    yearly_mask
                ]
            _draw_outflow_series(
                axis,
                yearly_outflow_series,
                yearly_frozen_fraction_percent,
                frozen_fraction_cmap,
                linewidth=1.0,
                color="#1f77b4",
            )
            axis.set_title(
                f"GEB river outflow for outlet {outlet_id} - {year}. Mean: {yearly_outflow_series.mean():.2f} m3/s"
            )
            axis.set_ylabel("Discharge [m3/s]")
            axis.grid(True, alpha=0.5, color="0.8")
            axis.margins(x=0)
            axis.xaxis.set_major_locator(mdates.MonthLocator())
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            axis.set_xlim(
                pd.Timestamp(year=year, month=1, day=1),
                pd.Timestamp(year=year, month=12, day=31, hour=23),
            )
            total_outflow_m3: float = float(
                yearly_outflow_series.sum() * timestep_seconds
            )
            total_outflow_mm: float = total_outflow_m3 * 1000.0 / total_area_m2
            axis.text(
                0.01,
                -0.22,
                f"total river outflow at point: {total_outflow_m3:,.0f} m3 "
                f"({total_outflow_mm:.2f} mm basin-equivalent)",
                transform=axis.transAxes,
                fontsize=7,
                va="top",
                ha="left",
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

        return_period_model: ReturnPeriodModel = ReturnPeriodModel(
            series=outflow_series,
            return_periods=[2, 5, 10, 25, 50, 100],
            fixed_shape=0.0,
            selection_strategy="best_fit",
        )
        diagnostics: plt.Figure = return_period_model.plot_diagnostics(figsize=(18, 14))
        diagnostics.suptitle(
            f"Outflow Diagnostics (hourly): {outlet_id}", fontsize=16, fontweight="bold"
        )
        diagnostics.savefig(
            outflow_plot_folder / f"{outflow_file.stem}_return_period.svg",
            bbox_inches="tight",
        )
        plt.close(diagnostics)

        plots_created += 1

    return plots_created


def _draw_outflow_series(
    axis: plt.Axes,
    outflow_series_m3_per_s: pd.Series,
    frozen_fraction_percent: pd.Series | None,
    frozen_fraction_cmap: mcolors.Colormap,
    linewidth: float,
    color: str,
    bucket_count: int = 10,
) -> LineCollection | None:
    """Draw outflow, optionally colored by frozen-soil fraction, with safe limits.

    Args:
        axis: Axis receiving the line.
        outflow_series_m3_per_s: Time-indexed discharge (m³/s).
        frozen_fraction_percent: Aligned frozen-soil fraction (%), or None.
        frozen_fraction_cmap: Colormap from unfrozen to fully frozen.
        linewidth: Width of a line colored by frozen fraction (points).
        color: Color for plain discharge lines.
        bucket_count: Number of frozen-fraction color buckets.

    Returns:
        Colored line collection, or None for a plain or single-point line.

    Raises:
        ValueError: If bucket_count is not positive.
    """
    if bucket_count < 1:
        raise ValueError("bucket_count must be positive.")
    time_index: pd.DatetimeIndex = pd.DatetimeIndex(outflow_series_m3_per_s.index)
    finite_values: np.ndarray = outflow_series_m3_per_s.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    upper_limit: float = (
        max(float(finite_values.max()) * 1.05, 1.0) if finite_values.size else 1.0
    )
    axis.set_ylim(0.0, upper_limit)
    if frozen_fraction_percent is None:
        axis.plot(
            time_index,
            outflow_series_m3_per_s.values,
            color=color,
            linewidth=0.9,
            zorder=2,
        )
        return None
    if len(time_index) < 2:
        axis.plot(
            time_index,
            outflow_series_m3_per_s.to_numpy(dtype=float),
            color="#1f77b4",
            linewidth=linewidth,
            zorder=2,
        )
        return None

    time_values: np.ndarray = mdates.date2num(time_index.to_numpy())
    outflow_values: np.ndarray = outflow_series_m3_per_s.to_numpy(dtype=float)
    line_points: np.ndarray = np.column_stack([time_values, outflow_values])
    frozen_values_percent: np.ndarray = frozen_fraction_percent.to_numpy(dtype=float)
    segment_context_percent: np.ndarray = (
        frozen_values_percent[:-1] + frozen_values_percent[1:]
    ) / 2.0
    # A small number of color buckets prevents thousands of tiny line segments.
    clipped_context_percent: np.ndarray = np.clip(segment_context_percent, 0.0, 100.0)
    bucket_edges_percent: np.ndarray = np.linspace(0.0, 100.0, bucket_count + 1)
    bucket_indices: np.ndarray = np.digitize(
        clipped_context_percent,
        bucket_edges_percent[1:-1],
        right=False,
    )
    bucket_centers_percent: np.ndarray = (
        bucket_edges_percent[:-1] + bucket_edges_percent[1:]
    ) / 2.0
    context_bucket_values_percent: np.ndarray = bucket_centers_percent[bucket_indices]
    discrete_cmap: mcolors.ListedColormap = mcolors.ListedColormap(
        frozen_fraction_cmap(np.linspace(0.0, 1.0, bucket_count))
    )
    discrete_norm: mcolors.BoundaryNorm = mcolors.BoundaryNorm(
        bucket_edges_percent, discrete_cmap.N
    )

    line_segments: list[np.ndarray[Any, Any]] = []
    merged_bucket_values_percent: list[float] = []
    run_start_idx: int = 0
    for segment_idx in range(1, len(bucket_indices)):
        if bucket_indices[segment_idx] != bucket_indices[run_start_idx]:
            line_segments.append(line_points[run_start_idx : segment_idx + 1])
            merged_bucket_values_percent.append(
                context_bucket_values_percent[run_start_idx]
            )
            run_start_idx = segment_idx
    line_segments.append(line_points[run_start_idx:])
    merged_bucket_values_percent.append(context_bucket_values_percent[run_start_idx])

    line_collection: LineCollection = LineCollection(
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


def plot_skill_score_maps(
    mapped_station_scores: gpd.GeoDataFrame,
    region_geom: gpd.GeoDataFrame,
    output_folder: Path,
    logger: logging.Logger,
    score_differences_by_model: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Plot per-station skill scores on a satellite basemap, one map per metric.

    Saves SVG and PNG files under ``output_folder/skill_score_maps/``.

    Args:
        mapped_station_scores: Per-station metrics with point geometry in any CRS.
        region_geom: Basin/region boundary overlaid on each map.
        output_folder: Root folder under which ``skill_score_maps/`` is created.
        logger: Logger to use for progress messages.
        score_differences_by_model: Optional matched GEB-vs-external station tables with
            ``KGE_difference`` values (dimensionless).
    """
    maps_folder: Path = output_folder / "skill_score_maps"
    maps_folder.mkdir(parents=True, exist_ok=True)

    kge_metric_columns: tuple[str, ...] = (
        "KGE",
        "KGE_correlation",
        "KGE_bias_ratio",
        "KGE_variability_ratio",
    )
    if all(
        column_name in mapped_station_scores.columns
        for column_name in kge_metric_columns
    ):
        kge_metric_configs: tuple[dict[str, object], ...] = tuple(
            SKILL_SCORE_PLOT_CONFIG_BY_COLUMN[column_name]
            for column_name in kge_metric_columns
        )
        _draw_score_maps(
            mapped_station_scores=mapped_station_scores,
            metric_configs=kge_metric_configs,
            output_path=maps_folder / "skill_score_map_kge_components",
            region_geom=region_geom,
        )
        logger.info("Saved KGE component skill score map.")

    for metric_config in SKILL_SCORE_PLOT_CONFIGS:
        metric_column: str = str(metric_config["col"])
        if metric_column not in mapped_station_scores.columns:
            logger.info("Metric '%s' not in evaluation data, skipping.", metric_column)
            continue

        metric_values: np.ndarray = pd.to_numeric(
            mapped_station_scores[metric_column], errors="coerce"
        ).to_numpy(dtype=float)
        valid_values: np.ndarray = metric_values[np.isfinite(metric_values)]
        if valid_values.size == 0:
            logger.info("No valid values for metric '%s', skipping.", metric_column)
            continue

        vmax: float = (
            float(np.nanpercentile(valid_values, 95))
            if metric_config["vmax"] is None
            else float(cast(float, metric_config["vmax"]))
        )
        _draw_score_maps(
            mapped_station_scores=mapped_station_scores,
            metric_configs=({**metric_config, "vmax": vmax},),
            output_path=maps_folder / f"skill_score_map_{metric_column.lower()}",
            region_geom=region_geom,
        )
        logger.info("Saved skill score map for %s.", metric_column)

    for model_name, model_score_differences in (
        score_differences_by_model or {}
    ).items():
        if "KGE_difference" not in model_score_differences:
            continue
        has_geometry: bool = "geometry" in model_score_differences
        coordinate_columns: set[str] = {"station_longitude", "station_latitude"}
        if not has_geometry and not coordinate_columns.issubset(
            model_score_differences
        ):
            logger.info("No station geometry found for %s difference map.", model_name)
            continue
        if not has_geometry:
            model_score_differences = gpd.GeoDataFrame(
                model_score_differences,
                geometry=gpd.points_from_xy(
                    model_score_differences["station_longitude"],
                    model_score_differences["station_latitude"],
                ),
                crs="EPSG:4326",
            )
        unmatched_station_scores: gpd.GeoDataFrame = gpd.GeoDataFrame(
            mapped_station_scores.loc[
                ~mapped_station_scores.index.isin(model_score_differences.index)
            ].copy(),
            geometry="geometry",
            crs=mapped_station_scores.crs,
        )
        unmatched_station_scores["KGE_difference"] = np.nan
        matched_station_count: int = len(model_score_differences)
        if not unmatched_station_scores.empty:
            # Difference maps otherwise hide whole regions where the external
            # source has no station match, which can look like a plotting error.
            model_score_differences = pd.concat(
                [model_score_differences, unmatched_station_scores],
                axis=0,
                copy=False,
            )
            logger.info(
                "%s difference map shows %d matched stations and %d unmatched "
                "eligible GEB stations.",
                model_name,
                matched_station_count,
                len(unmatched_station_scores),
            )
        mapped_score_differences: gpd.GeoDataFrame = gpd.GeoDataFrame(
            model_score_differences,
            geometry="geometry",
            crs=getattr(model_score_differences, "crs", None),
        )
        difference_values: np.ndarray = pd.to_numeric(
            mapped_score_differences["KGE_difference"], errors="coerce"
        ).to_numpy(dtype=float)
        valid_values: np.ndarray = difference_values[np.isfinite(difference_values)]
        if valid_values.size == 0:
            continue
        visible_limit: float = max(float(np.nanpercentile(abs(valid_values), 95)), 0.05)
        output_suffix: str = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        _draw_score_maps(
            mapped_station_scores=mapped_score_differences,
            metric_configs=(
                {
                    "col": "KGE_difference",
                    "label": "KGE difference (-)",
                    "cmap": "RdBu",
                    "vmin": -visible_limit,
                    "vmax": visible_limit,
                },
            ),
            output_path=maps_folder / f"skill_score_difference_map_{output_suffix}",
            region_geom=region_geom,
        )
        logger.info("Saved skill score difference map for %s.", model_name)

    logger.info("All skill score maps saved to: %s", maps_folder)


def _draw_score_maps(
    mapped_station_scores: gpd.GeoDataFrame,
    metric_configs: tuple[dict[str, object], ...],
    output_path: Path,
    region_geom: gpd.GeoDataFrame,
) -> None:
    """Draw station scores on satellite maps using one or four panels.

    Each panel reserves separate space for its map and colorbar so fixed map
    aspect ratios cannot collapse the four-panel layout.

    Args:
        mapped_station_scores: Dimensionless station scores with point geometry.
        metric_configs: One or four column, label, colormap, and color-limit settings.
        output_path: Output filename without extension; saves SVG and PNG.
        region_geom: Region boundary in any projected or geographic CRS.

    Raises:
        ValueError: If the number of metrics is not one or four.
    """
    if len(metric_configs) not in (1, 4):
        raise ValueError("Score maps require one or four metrics.")
    stations: gpd.GeoDataFrame = mapped_station_scores.to_crs("EPSG:3857")
    region: gpd.GeoDataFrame = region_geom.to_crs("EPSG:3857")
    multiple: bool = len(metric_configs) == 4
    figure: plt.Figure = plt.figure(figsize=(15, 12) if multiple else (10, 9))
    grid: GridSpec = figure.add_gridspec(
        2 if multiple else 1,
        2 if multiple else 1,
        left=0.03,
        right=0.94,
        bottom=0.03,
        top=0.95,
        wspace=0.25,
        hspace=0.12,
    )
    axis: plt.Axes
    config: dict[str, object]
    panel_index: int
    try:
        for panel_index, config in enumerate(metric_configs):
            panel_grid: GridSpecFromSubplotSpec = grid[panel_index].subgridspec(
                1, 2, width_ratios=(1, 0.035), wspace=0.04
            )
            axis = figure.add_subplot(panel_grid[0, 0])
            colorbar_axis: plt.Axes = figure.add_subplot(panel_grid[0, 1])
            values: pd.Series = pd.to_numeric(
                stations[str(config["col"])], errors="coerce"
            )
            valid: pd.Series = pd.Series(np.isfinite(values), index=values.index)
            norm: mcolors.Normalize = mcolors.Normalize(
                vmin=float(cast(float, config["vmin"])),
                vmax=float(cast(float, config["vmax"])),
            )
            cmap: str = str(config["cmap"])
            region.plot(
                ax=axis,
                color="none",
                edgecolor="white",
                linewidth=0.6,
                alpha=0.7,
                zorder=2,
            )
            if valid.any():
                axis.scatter(
                    stations.loc[valid].geometry.x,
                    stations.loc[valid].geometry.y,
                    c=values.loc[valid],
                    cmap=cmap,
                    norm=norm,
                    s=12,
                    zorder=4,
                    linewidths=0.2,
                    edgecolors="white",
                )
            if (~valid).any():
                axis.scatter(
                    stations.loc[~valid].geometry.x,
                    stations.loc[~valid].geometry.y,
                    c="grey",
                    marker="x",
                    s=10,
                    zorder=3,
                    linewidths=0.5,
                    label="No data",
                )
                axis.legend(fontsize=8, loc="lower right", framealpha=0.8)
            ctx.add_basemap(
                axis,
                crs="EPSG:3857",
                source=ctx.providers.Esri.WorldImagery,  # ty:ignore[unresolved-attribute]
                attribution=False,
                zoom="auto",
            )
            colorbar: Colorbar = figure.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=colorbar_axis,
            )
            colorbar.set_label(str(config["label"]), fontsize=10)
            axis.tick_params(
                labelbottom=False, labelleft=False, bottom=False, left=False
            )
            if multiple:
                # Automatic title positioning can become infinite on basemap axes,
                # which also makes tight export cropping omit entire map panels.
                axis.set_title(str(config["label"]), fontweight="bold", y=1.02)
                axis.text(
                    0.01,
                    0.99,
                    f"{chr(ord('a') + panel_index)})",
                    transform=axis.transAxes,
                    va="top",
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                    zorder=6,
                )
            # Match the colorbar height to the map after enforcing its aspect ratio.
            figure.canvas.draw()
            colorbar_axis.set_position(
                (
                    colorbar_axis.get_position().x0,
                    axis.get_position().y0,
                    colorbar_axis.get_position().width,
                    axis.get_position().height,
                )
            )
        # All panels share an extent, so one scale bar is sufficient.
        _add_map_scale_bar(axis)
        extension: str
        for extension in ("svg", "png"):
            figure.savefig(f"{output_path}.{extension}", bbox_inches="tight", dpi=300)
    finally:
        plt.close(figure)


def _add_map_scale_bar(axis: plt.Axes) -> None:
    """Add a scale bar to a projected discharge map.

    Args:
        axis: Map axis with coordinates in meters (EPSG:3857).
    """
    # Scale bar: round ~15% of the map width to a nice number (e.g. 152 km → 200 km)
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    map_width_m: float = x_max - x_min
    map_height_m: float = y_max - y_min
    bar_m: float = round(
        map_width_m * 0.15 / 10 ** np.floor(np.log10(map_width_m * 0.15))
    ) * 10 ** np.floor(np.log10(map_width_m * 0.15))
    bar_label: str = f"{int(bar_m / 1_000)} km" if bar_m >= 1_000 else f"{int(bar_m)} m"
    bar_x0: float = x_min + map_width_m * 0.03
    bar_y: float = y_min + map_height_m * 0.03
    axis.plot(
        [bar_x0, bar_x0 + bar_m],
        [bar_y, bar_y],
        color="white",
        linewidth=3,
        solid_capstyle="butt",
        zorder=5,
    )
    axis.text(
        bar_x0 + bar_m / 2,
        bar_y + map_height_m * 0.012,
        bar_label,
        color="white",
        fontsize=14,
        ha="center",
        va="bottom",
        zorder=5,
    )


def _prepare_boxplot_panels(
    tables: dict[str, pd.DataFrame],
    require_matching_stations: bool = False,
) -> list[tuple[str, str, pd.DataFrame]]:
    """Arrange dimensionless scores into one table per boxplot panel.

    Args:
        tables: Station-indexed score tables keyed by model or season.
        require_matching_stations: Keep only metrics and finite station pairs shared by all tables.

    Returns:
        Metric name, panel title, and numeric values for each available metric.
    """
    panels: list[tuple[str, str, pd.DataFrame]] = []
    metric: str
    config: dict[str, object]
    for metric, config in SKILL_SCORE_PLOT_CONFIG_BY_COLUMN.items():
        if require_matching_stations and any(
            metric not in table for table in tables.values()
        ):
            continue
        columns: dict[str, pd.Series] = {
            name: pd.to_numeric(table[metric], errors="coerce")
            for name, table in tables.items()
            if metric in table
        }
        if not columns:
            continue
        values: pd.DataFrame = pd.concat(columns, axis=1).replace(
            [np.inf, -np.inf], np.nan
        )
        values = values.dropna(how="any" if require_matching_stations else "all")
        if not values.empty:
            panels.append((metric, str(config["label"]), values))
    return panels


def plot_skill_score_boxplots(
    panels: list[tuple[str, str, pd.DataFrame]],
    output_path: Path,
    export: bool = True,
    extensions: tuple[str, ...] = ("pdf", "svg", "png"),
) -> None:
    """Plot score, seasonal, or external-model violin/boxplots with one layout.

    Box statistics use all finite scores; violin densities use only the visible
    score range. GEB-only panels omit group labels; comparison panels retain them.
    A shared legend identifies the best-possible score reference lines.
    Empty panels are skipped, and figures are closed after use.

    Args:
        panels: Metric name, title, and station-by-group table per panel.
            Scores are dimensionless; paired comparisons must already be aligned.
        output_path: Output filename without extension.
        export: Whether to save the figure.
        extensions: Figure formats to write.
    """
    if not panels:
        return
    column_count: int = min(4, len(panels))
    row_count: int = int(np.ceil(len(panels) / column_count))
    figure: plt.Figure = plt.figure(figsize=(3.2 * column_count, 3.5 * row_count))
    grid: GridSpec = figure.add_gridspec(row_count, 2 * column_count)
    colors: tuple[str, ...] = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#E69F00",
        "#CC79A7",
        "#56B4E9",
    )
    panel_index: int
    metric: str
    title: str
    group_scores: pd.DataFrame
    try:
        for panel_index, (metric, title, group_scores) in enumerate(panels):
            row: int = panel_index // column_count
            # Center the last row when it has fewer panels.
            offset: int = column_count - min(
                column_count, len(panels) - row * column_count
            )
            column: int = offset + 2 * (panel_index % column_count)
            axis: plt.Axes = figure.add_subplot(grid[row, column : column + 2])
            limits: tuple[float, float] = cast(
                tuple[float, float],
                SKILL_SCORE_PLOT_CONFIG_BY_COLUMN[metric]["ylim"] or (0.0, 2.0),
            )
            positions: list[int] = []
            labels: list[str] = []
            position: int
            name: str
            for position, name in enumerate(group_scores.columns):
                finite_scores: np.ndarray = group_scores[name].to_numpy(dtype=float)
                finite_scores = finite_scores[np.isfinite(finite_scores)]
                if not finite_scores.size:
                    continue
                color: str = colors[position % len(colors)]
                visible_scores: np.ndarray = finite_scores[
                    (finite_scores >= limits[0]) & (finite_scores <= limits[1])
                ]
                if visible_scores.size >= 3 and np.ptp(visible_scores) > 0:
                    violin: dict[str, Any] = axis.violinplot(
                        visible_scores,
                        positions=[position],
                        widths=0.65,
                        showextrema=False,
                        bw_method=0.15,
                    )
                    body: Any
                    for body in violin["bodies"]:
                        body.set(
                            facecolor=color, edgecolor=color, alpha=0.35, linewidth=0.8
                        )
                axis.boxplot(
                    finite_scores,
                    positions=[position],
                    widths=0.22,
                    patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.8},
                    boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.82},
                    whiskerprops={"color": color},
                    capprops={"color": color},
                    flierprops={
                        "marker": "o",
                        "markerfacecolor": color,
                        "markeredgecolor": "none",
                        "markersize": 2.5,
                        "alpha": 0.25,
                    },
                )
                positions.append(position)
                labels.append(_EXTERNAL_MODEL_DISPLAY_NAMES.get(name, name))
                axis.text(
                    position,
                    0.04,
                    f"med={np.median(finite_scores):.2f}",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    fontsize=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84},
                )
            if not positions:
                axis.set_visible(False)
                continue
            axis.axhline(
                0.0 if metric == "RRMSE" else 1.0,
                color=LINE_COLOR,
                linewidth=LINE_WIDTH,
                linestyle="--",
            )
            axis.set(
                title=title,
                xticks=positions,
                xticklabels=labels,
                xlim=(-0.6, len(group_scores.columns) - 0.4),
                ylim=(
                    limits[0] - (0.04 if metric == "RRMSE" else 0),
                    limits[1] + (0.04 if limits[1] == 1 else 0),
                ),
                yticks=np.linspace(*limits, 5),
            )
            axis.tick_params(labelsize=8)
            if list(group_scores.columns) == ["GEB"]:
                axis.set_xticks([])
            axis.spines[["top", "right", "bottom"]].set_visible(False)
            axis.grid(axis="y", color="0.88", linewidth=0.6)
            axis.set_axisbelow(True)
        figure.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=LINE_COLOR,
                    linewidth=LINE_WIDTH,
                    linestyle="--",
                    label="Best-possible value (1; RRMSE: 0)",
                )
            ],
            loc="lower center",
            fontsize=9,
            frameon=True,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 1))
        if export:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            extension: str
            for extension in extensions:
                figure.savefig(
                    f"{output_path}.{extension}", bbox_inches="tight", dpi=300
                )
    finally:
        plt.close(figure)
