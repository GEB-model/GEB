"""Analyze discharge skill against GRDC-Caravan catchment attributes.

This module contains catchment attribute definitions, data preparation,
statistics, and plotting, including skill scores versus upstream area.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colorbar import Colorbar
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import Bbox
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

from geb.build.data_catalog import DataCatalog
from geb.evaluate.workflows.discharge_metrics import SKILL_SCORE_PLOT_CONFIG_BY_COLUMN
from geb.evaluate.workflows.external_skill_scores import format_grdc_station_id


@dataclass(frozen=True)
class CatchmentCharacteristic:
    """Catchment attribute column, axis label, and conversion to display units.

    Args:
        column: Column in the enriched discharge-evaluation table.
        label: Publication label including units.
        scale: Multiplier applied once to raw source values to obtain display
            units. The data-catalog adapter preserves the original CSV units.
        logarithmic_x: Whether relationship plots use a base-10 x-axis.
    """

    column: str
    label: str
    scale: float = 1.0
    logarithmic_x: bool = False


@dataclass(frozen=True)
class KGEMetric:
    """Daily Kling–Gupta efficiency (KGE) or one of its three components.

    This metadata identifies a score to correlate with catchment attributes.
    It stores neither score values nor a desired calibration target.

    Args:
        score_column: Station-score column containing overall KGE, correlation,
            bias ratio, or variability ratio (all dimensionless).
        heatmap_label: Heatmap label for the Spearman correlation between this metric
            and a catchment attribute. The symbol ρₛ denotes that association,
            rather than the value of the KGE component itself.
    """

    score_column: str
    heatmap_label: str


# HydroATLAS stores temperature, terrain slope, and human footprint at 10×
# their display values (BasinATLAS v1 catalogue, C03/P02/A06). Caravan climate
# event frequencies are fractions (0–1), so multiply by 100 to display percent.
# Land-cover percentages already use 0–100 and need no conversion.
# These 32 attributes cover climate, topography, land cover, soils, hydrology,
# and human influence without pre-selecting variables by model performance.
SCREENING_CATCHMENT_CHARACTERISTICS: tuple[CatchmentCharacteristic, ...] = (
    CatchmentCharacteristic(
        "upstream_area_GEB",
        "Upstream catchment area (km²)",
        scale=1e-6,
        logarithmic_x=True,
    ),
    CatchmentCharacteristic("aridity_FAO_PM", "Aridity index, PET/P (–)"),
    CatchmentCharacteristic("ele_mt_sav", "Elevation (m)"),
    CatchmentCharacteristic("gwt_cm_sav", "Groundwater-table depth (cm)"),
    CatchmentCharacteristic("lka_pc_sse", "Lake-area extent (%)"),
    CatchmentCharacteristic("rev_mc_usu", "Upstream reservoir volume (million m³)"),
    CatchmentCharacteristic(
        "frac_snow", "Fraction of precipitation falling as snow (–)"
    ),
    CatchmentCharacteristic("p_mean", "Daily precipitation (mm/day)"),
    CatchmentCharacteristic("inu_pc_slt", "Long-term maximum inundation extent (%)"),
    CatchmentCharacteristic("tmp_dc_syr", "Annual air temperature (°C)", scale=0.1),
    CatchmentCharacteristic("seasonality_FAO_PM", "Moisture-index seasonality (–)"),
    CatchmentCharacteristic(
        "high_prec_freq", "High-precipitation-day frequency (%)", scale=100.0
    ),
    CatchmentCharacteristic(
        "high_prec_dur", "High-precipitation-event duration (days)"
    ),
    CatchmentCharacteristic(
        "low_prec_freq", "Low-precipitation-day frequency (%)", scale=100.0
    ),
    CatchmentCharacteristic("low_prec_dur", "Low-precipitation-event duration (days)"),
    CatchmentCharacteristic("slp_dg_sav", "Terrain slope (degrees)", scale=0.1),
    CatchmentCharacteristic("sgr_dk_sav", "Stream gradient (dm/km)"),
    CatchmentCharacteristic("wet_pc_sg1", "Wetland extent (%)"),
    CatchmentCharacteristic("dor_pc_pva", "Degree of regulation (%)"),
    CatchmentCharacteristic("for_pc_sse", "Forest-cover extent (%)"),
    CatchmentCharacteristic("crp_pc_sse", "Cropland extent (%)"),
    CatchmentCharacteristic("pst_pc_sse", "Pasture extent (%)"),
    CatchmentCharacteristic("ire_pc_sse", "Irrigated-area extent (%)"),
    CatchmentCharacteristic("urb_pc_sse", "Urban extent (%)"),
    CatchmentCharacteristic("gla_pc_sse", "Glacier extent (%)"),
    CatchmentCharacteristic("kar_pc_sse", "Karst-area extent (%)"),
    CatchmentCharacteristic("cly_pc_sav", "Soil clay fraction (%)"),
    CatchmentCharacteristic("snd_pc_sav", "Soil sand fraction (%)"),
    CatchmentCharacteristic("swc_pc_syr", "Annual soil-water content (%)"),
    CatchmentCharacteristic("hft_ix_s09", "Human-footprint index, 2009 (–)", scale=0.1),
    CatchmentCharacteristic("ppd_pk_sav", "Population density (people/km²)"),
    CatchmentCharacteristic("rdd_mk_sav", "Road density (m/km²)"),
)

_CATCHMENT_CHARACTERISTICS_BY_COLUMN: dict[str, CatchmentCharacteristic] = {
    catchment_characteristic.column: catchment_characteristic
    for catchment_characteristic in SCREENING_CATCHMENT_CHARACTERISTICS
}
_CATCHMENT_CHARACTERISTICS_BY_COLUMN["area"] = CatchmentCharacteristic(
    "area", "GRDC-Caravan catchment area (km²)", logarithmic_x=True
)


# characteristics to show on dashboard (keep this selection short to avoid slow dashboard)
DASHBOARD_CATCHMENT_COLUMNS: tuple[str, ...] = (
    "sgr_dk_sav",
    "ele_mt_sav",
    "area",
    "gwt_cm_sav",
    "low_prec_freq",
    "frac_snow",
    "aridity_FAO_PM",
    "dor_pc_pva",
    "lka_pc_sse",
)
DASHBOARD_CATCHMENT_CHARACTERISTICS: tuple[CatchmentCharacteristic, ...] = tuple(
    _CATCHMENT_CHARACTERISTICS_BY_COLUMN[column]
    for column in DASHBOARD_CATCHMENT_COLUMNS
)

KGE_METRICS: tuple[KGEMetric, ...] = (
    KGEMetric("KGE_correlation_daily", "ρₛ (correlation)"),
    KGEMetric("KGE_bias_ratio_daily", "ρₛ (bias ratio)"),
    KGEMetric("KGE_variability_ratio_daily", "ρₛ (variability ratio)"),
    KGEMetric("KGE_daily", "ρₛ (KGE)"),
)

KGE_RELATIONSHIP_PANELS: dict[str, str] = {
    "ele_mt_sav": "b) Elevation",
    "gwt_cm_sav": "c) Groundwater-table depth",
    "sgr_dk_sav": "d) Stream gradient",
    "low_prec_freq": "e) Low-precipitation-day frequency",
}

STATION_COLOR: str = "#35978F"
LOWESS_COLOR: str = "#01665E"
LOWESS_INTERVAL_COLOR: str = "#80CDC1"


def analyze_discharge_characteristics(
    station_scores: pd.DataFrame,
    output_folder: Path,
    logger: logging.Logger,
    output_name_suffix: str = "",
    export: bool = True,
    catchment_attributes: pd.DataFrame | None = None,
) -> None:
    """Load attributes, calculate associations, and draw the three figures.

    Analysis and plotting require more than 100 evaluated stations matched to
    GRDC-Caravan attributes.

    Args:
        station_scores: Filtered station scores; upstream area is in m².
        output_folder: Directory for the association CSV and figures.
        logger: Logger for data loading, matching, and exports.
        output_name_suffix: Evaluation-period suffix for filenames.
        export: Whether to save files. Figures are closed in either mode.
        catchment_attributes: Attributes keyed by gauge_id, or None to load
            GRDC-Caravan from the cached GEB data catalog.
    """
    if catchment_attributes is None:
        catchment_attributes = DataCatalog(logger=logger).fetch("GRDC_Caravan").read()
    enriched_scores: pd.DataFrame = enrich_discharge_evaluation(
        station_scores, catchment_attributes
    )
    matched_station_count: int = int(enriched_scores["grdc_caravan_matched"].sum())
    logger.info(
        "Matched %d/%d evaluated stations to GRDC-Caravan attributes.",
        matched_station_count,
        len(enriched_scores),
    )
    if matched_station_count <= 100:
        logger.info(
            "Skipping discharge characteristic analysis and plots: more than "
            "100 matched GRDC-Caravan stations are required (found %d).",
            matched_station_count,
        )
        return

    if export:
        output_folder.mkdir(parents=True, exist_ok=True)

    station_analysis_table: pd.DataFrame = prepare_kge_characteristic_analysis(
        enriched_scores
    )
    catchment_kge_correlations: pd.DataFrame = calculate_kge_component_associations(
        station_analysis_table
    )
    if export:
        association_path: Path = output_folder / (
            f"discharge_kge_component_associations{output_name_suffix}.csv"
        )
        catchment_kge_correlations.to_csv(association_path, index=False)
        logger.info(
            "Saved discharge characteristic associations to %s.", association_path
        )

    # Draw, save, and close one figure at a time to limit memory use.
    filename_stem: str
    draw_figure: Callable[[], plt.Figure]
    file_extensions: tuple[str, ...]
    file_extension: str
    for filename_stem, draw_figure, file_extensions in (
        (
            "discharge_characteristic_correlation_matrix",
            partial(create_characteristic_correlation_matrix, station_analysis_table),
            ("png",),
        ),
        (
            "discharge_kge_characteristic_heatmaps",
            partial(
                create_kge_characteristic_summary,
                station_analysis_table,
                catchment_kge_correlations,
            ),
            ("svg", "pdf", "png"),
        ),
        (
            "discharge_kge_all_characteristic_scatterplots",
            partial(
                create_kge_characteristic_scatterplots,
                station_analysis_table,
                catchment_kge_correlations,
            ),
            ("svg", "pdf", "png"),
        ),
    ):
        figure: plt.Figure = draw_figure()
        try:
            if export:
                for file_extension in file_extensions:
                    output_path: Path = (
                        output_folder
                        / f"{filename_stem}{output_name_suffix}.{file_extension}"
                    )
                    figure.savefig(
                        output_path,
                        dpi=300 if file_extension == "png" else None,
                        bbox_inches="tight",
                    )
                    logger.info(
                        "Saved discharge characteristic figure to %s.", output_path
                    )
        finally:
            plt.close(figure)


def load_dashboard_catchment_characteristics(
    mapped_station_scores: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame | None:
    """Load and prepare GRDC-Caravan attributes for the discharge dashboard.

    Args:
        mapped_station_scores: Evaluated station metrics and geometries.
        logger: Logger used by the GEB data catalog.

    Returns:
        Stations with nine attributes in display units. Unmatched stations have
        missing values; returns None when the optional data is unavailable.

    Raises:
        ValueError: If required columns are missing or station IDs are duplicated.
    """
    catchment_characteristic: CatchmentCharacteristic
    try:
        catchment_attributes: pd.DataFrame = (
            DataCatalog(logger=logger).fetch("GRDC_Caravan").read()
        )
    except RuntimeError as error:
        # Catchment attributes enrich the dashboard but must not make the core
        # discharge evaluation depend on network access.
        logger.warning(
            "GRDC-Caravan attributes are unavailable; creating the discharge "
            "dashboard without catchment-characteristic layers: %s",
            error,
        )
        return None
    dashboard_table: pd.DataFrame = enrich_discharge_evaluation(
        mapped_station_scores, catchment_attributes
    )
    required_columns: set[str] = {
        "station_ID",
        "grdc_caravan_matched",
        *(
            catchment_characteristic.column
            for catchment_characteristic in DASHBOARD_CATCHMENT_CHARACTERISTICS
        ),
    }
    missing_columns: set[str] = required_columns - set(dashboard_table.columns)
    if missing_columns:
        raise ValueError(
            "Enriched discharge metrics are missing dashboard columns: "
            f"{sorted(missing_columns)}"
        )
    if dashboard_table["station_ID"].duplicated().any():
        raise ValueError("Dashboard discharge metrics contain duplicate station IDs.")

    matched_stations: pd.Series = (
        dashboard_table["grdc_caravan_matched"].fillna(False).astype(bool)
    )
    for catchment_characteristic in DASHBOARD_CATCHMENT_CHARACTERISTICS:
        numeric_values: pd.Series = pd.to_numeric(
            dashboard_table[catchment_characteristic.column], errors="coerce"
        )
        # Mask every selected field explicitly so unmatched rows can never be
        # mistaken for valid zero-valued GRDC-Caravan observations.
        dashboard_table[catchment_characteristic.column] = (
            numeric_values.where(matched_stations) * catchment_characteristic.scale
        )
    return dashboard_table


def enrich_discharge_evaluation(
    station_scores: pd.DataFrame,
    catchment_attributes: pd.DataFrame,
) -> pd.DataFrame:
    """Join attributes by station ID, while keeping unmatched stations.

    Args:
        station_scores: Station-level GEB discharge metrics.
        catchment_attributes: GRDC-Caravan attributes keyed by ``gauge_id``.

    Returns:
        Evaluation table with catchment attributes and a match indicator.

    Raises:
        ValueError: If identifier columns are missing or gauge IDs are duplicated.
    """
    evaluation_table: pd.DataFrame = station_scores.copy()
    if "station_ID" not in evaluation_table.columns:
        if evaluation_table.index.name == "station_ID":
            evaluation_table = evaluation_table.reset_index()
        else:
            raise ValueError("Evaluation metrics have no station_ID column.")
    if "gauge_id" not in catchment_attributes.columns:
        raise ValueError("GRDC-Caravan attributes have no gauge_id column.")
    if catchment_attributes["gauge_id"].duplicated().any():
        raise ValueError("GRDC-Caravan attributes contain duplicate gauge_id values.")

    evaluation_table["gauge_id"] = evaluation_table["station_ID"].map(
        format_grdc_station_id
    )
    enriched_table: pd.DataFrame = evaluation_table.merge(
        catchment_attributes,
        on="gauge_id",
        how="left",
        validate="many_to_one",
        indicator="_grdc_caravan_merge",
    )
    enriched_table["grdc_caravan_matched"] = (
        enriched_table.pop("_grdc_caravan_merge") == "both"
    )
    return enriched_table


def prepare_kge_characteristic_analysis(
    station_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Select matched stations and convert attributes to display units.

    KGE and its components keep their original values and direction.

    Args:
        station_scores: Discharge metrics enriched with GRDC-Caravan attributes.

    Returns:
        Matched station table in display units.

    Raises:
        ValueError: If required columns or matched stations are unavailable.
    """
    catchment_characteristic: CatchmentCharacteristic
    column: str
    required_columns: set[str] = {
        "grdc_caravan_matched",
        *(kge_metric.score_column for kge_metric in KGE_METRICS),
        *(
            catchment_characteristic.column
            for catchment_characteristic in SCREENING_CATCHMENT_CHARACTERISTICS
        ),
    }
    missing_columns: set[str] = required_columns - set(station_scores.columns)
    if missing_columns:
        raise ValueError(
            f"Enriched discharge metrics are missing columns: {sorted(missing_columns)}"
        )

    analysis_table: pd.DataFrame = station_scores.loc[
        station_scores["grdc_caravan_matched"]
    ].copy()
    if analysis_table.empty:
        raise ValueError("No evaluated stations match GRDC-Caravan attributes.")

    numeric_columns: tuple[str, ...] = (
        *(kge_metric.score_column for kge_metric in KGE_METRICS),
        *(
            catchment_characteristic.column
            for catchment_characteristic in SCREENING_CATCHMENT_CHARACTERISTICS
        ),
    )
    for column in numeric_columns:
        analysis_table[column] = pd.to_numeric(analysis_table[column], errors="coerce")
    for catchment_characteristic in SCREENING_CATCHMENT_CHARACTERISTICS:
        analysis_table[catchment_characteristic.column] *= (
            catchment_characteristic.scale
        )
    return analysis_table


def calculate_kge_component_associations(
    station_analysis_table: pd.DataFrame,
) -> pd.DataFrame:
    """Correlate each catchment attribute with overall KGE and its components.

    Args:
        station_analysis_table: Output from :func:`prepare_kge_characteristic_analysis`.

    Returns:
        One row per catchment attribute and KGE metric. Existing CSV names
        are retained: ``variable`` identifies the catchment column, ``target``
        identifies the score column, and ``n`` counts paired stations.
        Correlations and p-values are dimensionless and are NaN when fewer
        than three pairs or fewer than two distinct values are available.
    """
    catchment_characteristic: CatchmentCharacteristic
    kge_metric: KGEMetric
    correlation_records: list[dict[str, float | int | str]] = []
    for catchment_characteristic in SCREENING_CATCHMENT_CHARACTERISTICS:
        for kge_metric in KGE_METRICS:
            paired_station_values: pd.DataFrame = station_analysis_table[
                [catchment_characteristic.column, kge_metric.score_column]
            ].dropna()
            spearman_correlation: float = np.nan
            p_value: float = np.nan
            if (
                len(paired_station_values) >= 3
                and paired_station_values[catchment_characteristic.column].nunique()
                >= 2
                and paired_station_values[kge_metric.score_column].nunique() >= 2
            ):
                correlation_result: Any = spearmanr(
                    paired_station_values[catchment_characteristic.column],
                    paired_station_values[kge_metric.score_column],
                )
                spearman_correlation = float(correlation_result.statistic)
                p_value = float(correlation_result.pvalue)
            correlation_records.append(
                {
                    "variable": catchment_characteristic.column,
                    "characteristic": catchment_characteristic.label,
                    "target": kge_metric.score_column,
                    "target_label": kge_metric.heatmap_label,
                    "n": len(paired_station_values),
                    "spearman_rho": spearman_correlation,
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(correlation_records)


def plot_skill_scores_vs_upstream_area(
    station_scores: pd.DataFrame,
    output_folder: Path,
    logger: logging.Logger,
) -> None:
    """Plot upstream area against discharge skill scores.

    Args:
        station_scores: Per-station evaluation metrics with `upstream_area_GEB` (m2).
        output_folder: Root folder where the scatterplot is saved.
        logger: Logger to use for progress messages.

    Raises:
        ValueError: If `upstream_area_GEB` is missing from the evaluation metrics.
    """  # noqa: DOC202
    if "upstream_area_GEB" not in station_scores.columns:
        raise ValueError("`upstream_area_GEB` is missing from evaluation metrics.")

    upstream_area_km2: pd.Series = (
        pd.to_numeric(station_scores["upstream_area_GEB"], errors="coerce")
        / 1_000_000.0
    )

    metric_order: tuple[str, ...] = (
        "KGE",
        "KGE_correlation",
        "KGE_bias_ratio",
        "KGE_variability_ratio",
        "NSE",
        "RRMSE",
    )
    fig: plt.Figure = plt.figure(figsize=(13.5, 7.2), constrained_layout=True)
    # 6-column grid: KGE spans 3 cols (half width), each KGE component 1 col.
    # NSE and RRMSE each span 3 cols in the bottom row.
    grid: GridSpec = fig.add_gridspec(2, 6)
    axes_by_metric: dict[str, plt.Axes] = {
        "KGE": fig.add_subplot(grid[0, 0:3]),
        "KGE_correlation": fig.add_subplot(grid[0, 3]),
        "KGE_bias_ratio": fig.add_subplot(grid[0, 4]),
        "KGE_variability_ratio": fig.add_subplot(grid[0, 5]),
        "NSE": fig.add_subplot(grid[1, 0:3]),
        "RRMSE": fig.add_subplot(grid[1, 3:6]),
    }
    has_values: bool = False
    plot_color: str = "#1f77b4"
    regression_color: str = "#D55E00"
    for metric_col in metric_order:
        metric_config: dict[str, object] = SKILL_SCORE_PLOT_CONFIG_BY_COLUMN[metric_col]
        axis: plt.Axes = axes_by_metric[str(metric_config["col"])]
        if metric_col not in station_scores.columns:
            logger.info("Metric '%s' not in evaluation data, skipping.", metric_col)
            axis.set_visible(False)
            continue

        metric_values: pd.Series = pd.to_numeric(
            station_scores[metric_col], errors="coerce"
        )
        valid_mask: pd.Series = (
            upstream_area_km2.gt(0)
            & np.isfinite(upstream_area_km2)
            & np.isfinite(metric_values)
        )
        if not valid_mask.any():
            logger.info("No valid values for metric '%s', skipping.", metric_col)
            axis.set_visible(False)
            continue

        has_values = True
        y_limits: tuple[float, float] = (
            cast(tuple[float, float], metric_config["ylim"])
            if metric_config["ylim"] is not None
            else (
                0.0,
                max(float(np.percentile(metric_values[valid_mask], 98)) * 1.1, 1.0),
            )
        )
        axis.scatter(
            upstream_area_km2[valid_mask],
            metric_values[valid_mask],
            s=14,
            alpha=0.55,
            color=plot_color,
            edgecolors="none",
        )
        trend_mask: pd.Series = valid_mask & metric_values.between(
            y_limits[0], y_limits[1]
        )
        if trend_mask.sum() >= 10 and upstream_area_km2[trend_mask].nunique() > 1:
            log_area: np.ndarray = np.log10(
                upstream_area_km2[trend_mask].to_numpy(dtype=float)
            )
            trend_values: np.ndarray = metric_values[trend_mask].to_numpy(dtype=float)
            fitted_values: np.ndarray = np.asarray(
                lowess(
                    trend_values,
                    log_area,
                    frac=0.35,
                    it=2,
                    return_sorted=True,
                ),
                dtype=float,
            )
            fitted_log_area: np.ndarray
            unique_indices: np.ndarray
            fitted_log_area, unique_indices = np.unique(
                fitted_values[:, 0], return_index=True
            )
            trend_x: np.ndarray = np.logspace(
                fitted_log_area.min(), fitted_log_area.max(), 180
            )
            trend_y: np.ndarray = np.interp(
                np.log10(trend_x),
                fitted_log_area,
                fitted_values[unique_indices, 1],
            )
            axis.plot(
                trend_x,
                trend_y,
                color=regression_color,
                linewidth=2.1,
                zorder=3,
            )
            association: Any = spearmanr(log_area, trend_values)
            rho: float = float(association.statistic)
            p_value: float = float(association.pvalue)
            axis.text(
                0.04,
                0.92,
                f"Spearman $\\rho_s$ = {rho:+.2f}\n"
                f"{'p<0.001' if p_value < 0.001 else f'p={p_value:.3f}'}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="black",
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "0.9",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )
        axis.set_xscale("log")
        axis.set_ylim(*y_limits)
        axis.grid(True, color="0.85", linewidth=0.5)
        axis.set_title(str(metric_config["label"]), color=plot_color, fontweight="bold")
        for spine in axis.spines.values():
            spine.set_edgecolor("0.7")

    if not has_values:
        logger.warning("No valid skill scores found for upstream-area scatterplot.")
        plt.close(fig)
        return

    for axis in axes_by_metric.values():
        axis.set_xlabel("Upstream area (km²)")

    lowess_legend_handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=plot_color,
            markeredgecolor="none",
            markersize=5.5,
            alpha=0.55,
            label="Station",
        ),
        Line2D(
            [0],
            [0],
            color=regression_color,
            linewidth=2.1,
            label="Local regression (LOWESS)",
        ),
    ]
    fig.legend(
        handles=lowess_legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncols=2,
        frameon=False,
        fontsize=9,
        columnspacing=1.8,
        handlelength=2.4,
    )

    explanations_folder: Path = output_folder / "skill_score_explanations"
    explanations_folder.mkdir(parents=True, exist_ok=True)
    output_path: Path = explanations_folder / "skill_scores_vs_upstream_area"
    ext: str
    for ext in ("svg", "png"):
        fig.savefig(f"{output_path}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved skill score upstream-area scatterplot to: %s", output_path)


def create_characteristic_correlation_matrix(
    station_analysis_table: pd.DataFrame,
) -> plt.Figure:
    """Plot correlations among GRDC-Caravan catchment characteristics.

    Args:
        station_analysis_table: Prepared matched-station analysis table.

    Returns:
        Lower-triangular heatmap of dimensionless Spearman correlations.
    """
    characteristic_columns: list[str] = [
        catchment_characteristic.column
        for catchment_characteristic in SCREENING_CATCHMENT_CHARACTERISTICS
    ]
    correlation_matrix: pd.DataFrame = station_analysis_table[
        characteristic_columns
    ].corr(method="spearman", min_periods=3)
    characteristic_labels: list[str] = [
        catchment_characteristic.label
        for catchment_characteristic in SCREENING_CATCHMENT_CHARACTERISTICS
    ]
    correlation_matrix.index = characteristic_labels
    correlation_matrix.columns = characteristic_labels

    figure: plt.Figure
    axis: plt.Axes
    figure, axis = plt.subplots(figsize=(15.8, 14.2))
    sns.heatmap(
        correlation_matrix,
        mask=np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1),
        ax=axis,
        cmap="BrBG",
        vmin=-1.0,
        vmax=1.0,
        annot=True,
        fmt="+.2f",
        annot_kws={"fontsize": 6.1},
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Spearman rank correlation, ρₛ", "shrink": 0.74},
    )
    axis.tick_params(axis="both", labelsize=7.3, length=0)
    axis.set_xticklabels(axis.get_xticklabels(), rotation=52, ha="right")
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
    axis.set_title(
        "Spearman correlations among GRDC–Caravan catchment characteristics",
        loc="left",
        fontsize=13.0,
        fontweight="bold",
        pad=16,
    )
    figure.subplots_adjust(left=0.31, right=0.91, top=0.93, bottom=0.285)
    return figure


def create_kge_characteristic_summary(
    station_analysis_table: pd.DataFrame,
    catchment_kge_correlations: pd.DataFrame,
) -> plt.Figure:
    """Plot the KGE-component heatmap and four linked relationships.

    Args:
        station_analysis_table: Prepared matched-station analysis table.
        catchment_kge_correlations: KGE-component association table.

    Returns:
        Figure containing the heatmap and four continuous relationship panels.

    Raises:
        ValueError: If selected relationship-panel inputs are incomplete.
    """
    column: str
    column_index: int
    relationship_axis: plt.Axes
    row_index: int
    spearman_correlation_matrix: pd.DataFrame = catchment_kge_correlations.pivot(
        index="variable", columns="target", values="spearman_rho"
    )
    p_value_matrix: pd.DataFrame = catchment_kge_correlations.pivot(
        index="variable", columns="target", values="p_value"
    )
    catchment_labels_by_column: dict[str, str] = {
        catchment_characteristic.column: catchment_characteristic.label
        for catchment_characteristic in SCREENING_CATCHMENT_CHARACTERISTICS
    }
    kge_metric_columns: list[str] = [
        kge_metric.score_column for kge_metric in KGE_METRICS
    ]
    required_catchment_columns: list[str] = [
        characteristic.column for characteristic in SCREENING_CATCHMENT_CHARACTERISTICS
    ]
    panel_catchment_columns: set[str] = set(KGE_RELATIONSHIP_PANELS)
    if not {"KGE_daily", *panel_catchment_columns}.issubset(
        station_analysis_table.columns
    ):
        raise ValueError("KGE relationship-panel inputs are incomplete.")

    spearman_correlation_matrix = spearman_correlation_matrix.reindex(
        index=required_catchment_columns, columns=kge_metric_columns
    )
    p_value_matrix = p_value_matrix.reindex(
        index=required_catchment_columns, columns=kge_metric_columns
    )
    ranked_catchment_columns: list[str] = (
        spearman_correlation_matrix["KGE_daily"]
        .abs()
        .sort_values(ascending=False, kind="stable", na_position="last")
        .index.tolist()
    )
    spearman_correlation_matrix = spearman_correlation_matrix.loc[
        ranked_catchment_columns
    ]
    p_value_matrix = p_value_matrix.loc[ranked_catchment_columns]

    figure: plt.Figure = plt.figure(figsize=(14.8, 11.8))
    outer_grid: GridSpec = figure.add_gridspec(
        1, 2, width_ratios=(1.12, 1.10), wspace=0.30
    )
    association_axis: plt.Axes = figure.add_subplot(outer_grid[0, 0])
    relationship_panel_grid: GridSpecFromSubplotSpec = outer_grid[0, 1].subgridspec(
        5,
        1,
        height_ratios=(1.0, 1.0, 1.0, 1.0, 0.005),
        hspace=0.52,
    )
    relationship_axes: list[plt.Axes] = [
        figure.add_subplot(relationship_panel_grid[row_index, 0])
        for row_index in range(len(KGE_RELATIONSHIP_PANELS))
    ]

    association_image: AxesImage = association_axis.imshow(
        spearman_correlation_matrix.to_numpy(dtype=float),
        cmap="BrBG",
        norm=TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5),
        aspect="auto",
    )
    association_axis.set_xticks(
        np.arange(len(kge_metric_columns)),
        [kge_metric.heatmap_label for kge_metric in KGE_METRICS],
    )
    association_axis.set_yticks(
        np.arange(len(ranked_catchment_columns)),
        [
            catchment_labels_by_column[catchment_column]
            for catchment_column in ranked_catchment_columns
        ],
    )
    for row_index in range(len(ranked_catchment_columns)):
        for column_index in range(len(kge_metric_columns)):
            spearman_correlation: float = float(
                spearman_correlation_matrix.iloc[row_index, column_index]
            )
            p_value: float = float(p_value_matrix.iloc[row_index, column_index])
            if np.isnan(spearman_correlation):
                continue
            significance_marker: str = "*" if p_value < 0.05 else ""
            correlation_text: str = (
                "0.00"
                if abs(spearman_correlation) < 0.005
                else f"{spearman_correlation:+.2f}"
            )
            text_color: str = (
                "white" if abs(spearman_correlation) >= 0.34 else "#222222"
            )
            association_axis.text(
                column_index,
                row_index,
                f"{correlation_text}{significance_marker}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_color,
            )
    association_axis.set_title(
        "a) Spearman correlations of catchment characteristics\n"
        "with KGE and its components",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        pad=30,
    )
    association_axis.tick_params(
        axis="x",
        labelsize=8.2,
        length=0,
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        pad=4,
    )
    association_axis.get_xticklabels()[-1].set_fontweight("bold")
    association_axis.tick_params(axis="y", labelsize=7.3, length=0)
    association_axis.set_xticks(np.arange(-0.5, 4.0, 1.0), minor=True)
    association_axis.set_yticks(
        np.arange(-0.5, len(ranked_catchment_columns), 1.0), minor=True
    )
    association_axis.grid(which="minor", color="white", linewidth=0.8)
    association_axis.tick_params(which="minor", bottom=False, left=False)
    association_colorbar: Colorbar = figure.colorbar(
        association_image,
        ax=association_axis,
        orientation="horizontal",
        pad=0.035,
        fraction=0.032,
    )
    association_colorbar.set_ticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    association_colorbar.ax.tick_params(labelsize=7.4, length=3.0)
    association_colorbar.set_label(
        "Spearman rank correlation, ρₛ  (* p-value < 0.05)", fontsize=8.5
    )

    random_generator: np.random.Generator = np.random.default_rng(42)
    kge_lower_limit: float
    kge_upper_limit: float
    kge_lower_limit, kge_upper_limit = station_analysis_table["KGE_daily"].quantile(
        [0.10, 0.90]
    )
    kge_axis_limits: tuple[float, float] = (
        float(kge_lower_limit),
        float(kge_upper_limit),
    )
    for relationship_axis, column in zip(
        relationship_axes, KGE_RELATIONSHIP_PANELS, strict=True
    ):
        catchment_characteristic: CatchmentCharacteristic = (
            _CATCHMENT_CHARACTERISTICS_BY_COLUMN[column]
        )
        _plot_relationship_panel(
            axis=relationship_axis,
            station_analysis_table=station_analysis_table,
            catchment_characteristic=catchment_characteristic,
            spearman_correlation=float(
                spearman_correlation_matrix.loc[column, "KGE_daily"]
            ),
            random_generator=random_generator,
            kge_axis_limits=kge_axis_limits,
        )
        if relationship_axis.axison:
            relationship_axis.set_title(
                KGE_RELATIONSHIP_PANELS[column],
                loc="left",
                fontsize=10.0,
                fontweight="bold",
                pad=7,
            )
        if pd.notna(spearman_correlation_matrix.loc[column, "KGE_daily"]):
            characteristic_row: int = ranked_catchment_columns.index(column)
            association_axis.add_patch(
                Rectangle(
                    (
                        kge_metric_columns.index("KGE_daily") - 0.5,
                        characteristic_row - 0.5,
                    ),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="#111111",
                    linewidth=2.2,
                    zorder=7,
                    clip_on=False,
                )
            )
            association_axis.get_yticklabels()[characteristic_row].set_fontweight(
                "bold"
            )

    figure.subplots_adjust(left=0.255, right=0.985, top=0.96, bottom=0.09)
    figure.canvas.draw()
    for relationship_axis, column in zip(
        relationship_axes, KGE_RELATIONSHIP_PANELS, strict=True
    ):
        characteristic_row = ranked_catchment_columns.index(column)
        connector_start_pixels: np.ndarray = association_axis.transData.transform(
            (len(kge_metric_columns) - 0.48, characteristic_row)
        )
        connector_start_figure: np.ndarray = figure.transFigure.inverted().transform(
            connector_start_pixels
        )
        connector_end_figure: tuple[float, float] = (
            relationship_axis.get_position().x0 - 0.008,
            relationship_axis.get_position().y1,
        )
        overall_kge_correlation: float = float(
            spearman_correlation_matrix.loc[column, "KGE_daily"]
        )
        if np.isnan(overall_kge_correlation):
            continue
        correlation_color_fraction: float = float(
            np.asarray(association_image.norm(np.asarray(overall_kge_correlation)))
        )
        connector: Line2D = Line2D(
            [float(connector_start_figure[0]), connector_end_figure[0]],
            [float(connector_start_figure[1]), connector_end_figure[1]],
            transform=figure.transFigure,
            color=association_image.cmap(correlation_color_fraction),
            linewidth=1.5,
            alpha=0.9,
            solid_capstyle="round",
            zorder=6,
        )
        figure.add_artist(connector)

    colorbar_position: Bbox = association_colorbar.ax.get_position()
    colorbar_center_y: float = float(
        colorbar_position.y0 + colorbar_position.height / 2.0
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=STATION_COLOR,
                markeredgecolor="none",
                markersize=6.0,
                alpha=0.5,
                label="Station",
            ),
            Line2D(
                [0],
                [0],
                color=LOWESS_COLOR,
                linewidth=2.8,
                label="Local regression (LOWESS)",
            ),
            Patch(
                facecolor=LOWESS_INTERVAL_COLOR,
                edgecolor="none",
                alpha=0.18,
                label="95% bootstrap CI",
            ),
        ],
        loc="center right",
        bbox_to_anchor=(0.985, colorbar_center_y),
        bbox_transform=figure.transFigure,
        ncols=3,
        frameon=False,
        fontsize=9.0,
        columnspacing=1.2,
        handlelength=2.6,
    )
    return figure


def create_kge_characteristic_scatterplots(
    station_analysis_table: pd.DataFrame,
    catchment_kge_correlations: pd.DataFrame,
) -> plt.Figure:
    """Draw 32 KGE relationships, ranked by absolute Spearman correlation.

    Catchment area uses a log x-axis; other attributes use display units.

    Args:
        station_analysis_table: Prepared matched-station analysis table.
        catchment_kge_correlations: KGE-component association table.

    Returns:
        Figure with four columns showing dimensionless KGE against all 32
        catchment attributes in their documented display units.
    """
    axis: plt.Axes
    catchment_characteristic: CatchmentCharacteristic
    panel_index: int
    overall_kge_correlations: pd.DataFrame = catchment_kge_correlations.loc[
        catchment_kge_correlations["target"] == "KGE_daily"
    ].set_index("variable")
    required_catchment_columns: list[str] = [
        catchment_characteristic.column
        for catchment_characteristic in SCREENING_CATCHMENT_CHARACTERISTICS
    ]
    overall_kge_correlations = overall_kge_correlations.reindex(
        required_catchment_columns
    )
    ranked_catchment_columns: list[str] = (
        overall_kge_correlations["spearman_rho"]
        .abs()
        .sort_values(ascending=False, kind="stable", na_position="last")
        .index.tolist()
    )
    ordered_characteristics: list[CatchmentCharacteristic] = [
        _CATCHMENT_CHARACTERISTICS_BY_COLUMN[column]
        for column in ranked_catchment_columns
    ]

    column_count: int = 4
    row_count: int = int(np.ceil(len(ordered_characteristics) / column_count))
    kge_lower_limit: float
    kge_upper_limit: float
    kge_lower_limit, kge_upper_limit = station_analysis_table["KGE_daily"].quantile(
        [0.10, 0.90]
    )
    kge_axis_limits: tuple[float, float] = (
        float(kge_lower_limit),
        float(kge_upper_limit),
    )
    figure: plt.Figure
    axes: np.ndarray
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(13.6, 1.68 * row_count),
        sharey=True,
    )
    for panel_index, (axis, catchment_characteristic) in enumerate(
        zip(axes.flat, ordered_characteristics, strict=True)
    ):
        spearman_correlation: float = float(
            overall_kge_correlations.loc[
                catchment_characteristic.column, "spearman_rho"
            ]
        )
        if not _plot_relationship_panel(
            axis,
            station_analysis_table,
            catchment_characteristic,
            spearman_correlation,
            kge_axis_limits,
        ):
            continue
        # Spreadsheet-style labels continue with aa after z.
        panel_number: int = panel_index + 1
        panel_label: str = ""
        remainder: int
        while panel_number:
            panel_number, remainder = divmod(panel_number - 1, 26)
            panel_label = chr(ord("a") + remainder) + panel_label
        axis.set_title(
            f"{panel_label}) {catchment_characteristic.label}\nSpearman ρₛ = {spearman_correlation:+.2f}",
            loc="left",
            fontsize=8.0,
            fontweight="bold",
            pad=4,
        )

    for axis in axes[:, 0]:
        axis.set_ylabel("KGE (–)", fontsize=7.6)
    scatterplot_legend_handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#315F70",
            markeredgecolor="none",
            markersize=5.5,
            alpha=0.5,
            label="Station",
        ),
        Line2D(
            [0],
            [0],
            color="#0C526C",
            linewidth=2.2,
            label="Local regression (LOWESS)",
        ),
    ]
    figure.legend(
        handles=scatterplot_legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncols=2,
        frameon=False,
        fontsize=9.2,
        columnspacing=1.8,
        handlelength=2.5,
    )
    figure.subplots_adjust(
        left=0.06,
        right=0.99,
        top=0.985,
        bottom=0.06,
        wspace=0.22,
        hspace=0.48,
    )
    return figure


def _plot_relationship_panel(
    axis: plt.Axes,
    station_analysis_table: pd.DataFrame,
    catchment_characteristic: CatchmentCharacteristic,
    spearman_correlation: float,
    kge_axis_limits: tuple[float, float],
    random_generator: np.random.Generator | None = None,
) -> bool:
    """Plot KGE against a catchment attribute with a locally smoothed curve.

    LOWESS is locally weighted regression. The curve spans the 1st–99th
    percentiles of the catchment attribute (log-transformed where configured). A supplied generator enables
    150 bootstrap samples for the summary figure; the 32-panel figure uses one curve.

    Args:
        axis: Axis receiving the relationship.
        station_analysis_table: Matched station values in display units.
        catchment_characteristic: Column metadata, display units, and x transformation.
        spearman_correlation: Dimensionless Spearman correlation, or NaN for unavailable pairs.
        kge_axis_limits: Shared dimensionless KGE limits.
        random_generator: Reproducible resampling generator for summary panels.

    Returns:
        Whether enough valid station pairs were available to draw the panel.
    """
    bootstrap_index: int
    paired_station_values: pd.DataFrame = station_analysis_table[
        [catchment_characteristic.column, "KGE_daily"]
    ].dropna()
    if catchment_characteristic.logarithmic_x:
        paired_station_values = paired_station_values.loc[
            paired_station_values[catchment_characteristic.column] > 0.0
        ]
    if np.isnan(spearman_correlation) or len(paired_station_values) < 10:
        axis.set_axis_off()
        return False

    catchment_values: np.ndarray = paired_station_values[
        catchment_characteristic.column
    ].to_numpy(dtype=float)
    kge_values: np.ndarray = paired_station_values["KGE_daily"].to_numpy(dtype=float)
    regression_x_values: np.ndarray = (
        np.log10(catchment_values)
        if catchment_characteristic.logarithmic_x
        else catchment_values
    )
    regression_x_lower_limit: float
    regression_x_upper_limit: float
    regression_x_lower_limit, regression_x_upper_limit = np.quantile(
        regression_x_values, [0.01, 0.99]
    )
    if not regression_x_upper_limit > regression_x_lower_limit:
        axis.set_axis_off()
        return False
    regression_x_grid: np.ndarray = np.linspace(
        regression_x_lower_limit, regression_x_upper_limit, 180
    )
    fitted_kge_curve: np.ndarray = _fit_lowess_to_grid(
        regression_x_values, kge_values, regression_x_grid, robust_iterations=2
    )
    catchment_value_grid: np.ndarray = (
        np.power(10.0, regression_x_grid)
        if catchment_characteristic.logarithmic_x
        else regression_x_grid
    )
    show_bootstrap_interval: bool = random_generator is not None
    if random_generator is not None:
        bootstrap_curves: np.ndarray = np.empty(
            (150, len(regression_x_grid)), dtype=float
        )
        for bootstrap_index in range(150):
            bootstrap_station_indices: np.ndarray = random_generator.integers(
                0, len(regression_x_values), size=len(regression_x_values)
            )
            if np.unique(regression_x_values[bootstrap_station_indices]).size < 2:
                # Sparse attributes can yield a bootstrap sample with one value.
                bootstrap_curves[bootstrap_index] = kge_values[
                    bootstrap_station_indices
                ].mean()
            else:
                bootstrap_curves[bootstrap_index] = _fit_lowess_to_grid(
                    regression_x_values[bootstrap_station_indices],
                    kge_values[bootstrap_station_indices],
                    regression_x_grid,
                    robust_iterations=1,
                )
        confidence_lower_bound: np.ndarray
        confidence_upper_bound: np.ndarray
        confidence_lower_bound, confidence_upper_bound = np.quantile(
            bootstrap_curves, [0.025, 0.975], axis=0
        )
        axis.fill_between(
            catchment_value_grid,
            confidence_lower_bound,
            confidence_upper_bound,
            color=LOWESS_INTERVAL_COLOR,
            alpha=0.20,
            linewidth=0.0,
            zorder=1,
        )
        axis.text(
            0.025,
            0.94,
            f"Spearman ρₛ = {spearman_correlation:+.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            color="#202020",
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#E8E8E8",
                "edgecolor": "none",
                "alpha": 0.60,
            },
        )
        axis.set_xlabel(catchment_characteristic.label, fontsize=8.5)
        axis.set_ylabel("KGE (–)", fontsize=8.5)

    axis.scatter(
        catchment_values,
        kge_values,
        s=9 if show_bootstrap_interval else 7,
        color=STATION_COLOR if show_bootstrap_interval else "#315F70",
        alpha=0.27 if show_bootstrap_interval else 0.20,
        linewidths=0.0,
        rasterized=True,
        zorder=2,
    )
    axis.plot(
        catchment_value_grid,
        fitted_kge_curve,
        color=LOWESS_COLOR if show_bootstrap_interval else "#0C526C",
        linewidth=2.3 if show_bootstrap_interval else 1.8,
        zorder=4,
    )
    if catchment_characteristic.logarithmic_x:
        axis.set_xscale("log")
    axis.set_xlim(catchment_value_grid[0], catchment_value_grid[-1])
    axis.set_ylim(kge_axis_limits)
    axis.grid(
        axis="y",
        color="#D9D9D9" if show_bootstrap_interval else "#DDDDDD",
        linewidth=0.6 if show_bootstrap_interval else 0.55,
    )
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=7.4 if show_bootstrap_interval else 6.7)
    if show_bootstrap_interval:
        axis.tick_params(axis="x", pad=3)
    axis.spines[["top", "right"]].set_visible(False)
    return True


def _fit_lowess_to_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    regression_x_grid: np.ndarray,
    robust_iterations: int,
) -> np.ndarray:
    """Fit LOWESS and interpolate fitted values to a common grid.

    Args:
        x_values: Finite characteristic values, optionally log-transformed.
        y_values: Corresponding dimensionless KGE values.
        regression_x_grid: Grid in the same units as ``x_values``.
        robust_iterations: Number of LOWESS residual-reweighting iterations.

    Returns:
        Fitted dimensionless KGE values on ``regression_x_grid``.

    Raises:
        ValueError: If fewer than two distinct x-values are available.
    """
    tied_value: float
    unique_x_values: np.ndarray = np.unique(x_values)
    if len(unique_x_values) < 2:
        raise ValueError("A LOWESS curve needs two distinct characteristic values.")

    tie_adjusted_x_values: np.ndarray = x_values.copy()
    if len(unique_x_values) < len(x_values):
        minimum_x_spacing: float = float(np.min(np.diff(unique_x_values)))
        tie_separation_width: float = minimum_x_spacing * 1e-6
        for tied_value in unique_x_values:
            tied_indices: np.ndarray = np.flatnonzero(x_values == tied_value)
            if len(tied_indices) > 1:
                # This tiny separation of tied values prevents division by zero while
                # retaining the full statistical weight of zero-heavy data.
                tie_adjusted_x_values[tied_indices] += np.linspace(
                    -0.5 * tie_separation_width,
                    0.5 * tie_separation_width,
                    len(tied_indices),
                )

    fitted_values: np.ndarray = np.asarray(
        lowess(
            y_values,
            tie_adjusted_x_values,
            frac=0.35,
            it=robust_iterations,
            return_sorted=True,
        ),
        dtype=float,
    )
    fitted_x_values: np.ndarray
    unique_fitted_indices: np.ndarray
    fitted_x_values, unique_fitted_indices = np.unique(
        fitted_values[:, 0], return_index=True
    )
    return np.interp(
        regression_x_grid, fitted_x_values, fitted_values[unique_fitted_indices, 1]
    )
