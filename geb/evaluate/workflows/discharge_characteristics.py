"""Relate discharge skill scores to GRDC-Caravan catchment attributes."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from geb.evaluate.workflows.external_skill_scores import format_grdc_station_key


@dataclass(frozen=True)
class Characteristic:
    """Metadata needed to analyse and label one catchment characteristic.

    Args:
        column: GRDC-Caravan or GEB table column.
        label: Concise plot label including units.
        decimals: Decimal places shown at category boundaries.
        scale: Factor converting stored values to the displayed unit.
    """

    column: str
    label: str
    decimals: int
    scale: float = 1.0


class MetricStyle(NamedTuple):
    """Plot settings for one dimensionless daily discharge metric."""

    label: str
    short_name: str
    limits: tuple[float, float]
    ticks: tuple[float, ...]
    benchmark: float | None
    benchmark_label: str | None


METRIC_STYLES: dict[str, MetricStyle] = {
    "KGE_daily": MetricStyle(
        "Kling–Gupta efficiency, KGE (–)",
        "kge",
        (-1.0, 1.0),
        (-1.0, -0.5, 0.0, 0.5, 1.0),
        -0.41,
        "Mean-flow benchmark (KGE = −0.41)",
    ),
    "NSE_daily": MetricStyle(
        "Nash–Sutcliffe efficiency, NSE (–)",
        "nse",
        (-2.0, 1.0),
        (-2.0, -1.0, 0.0, 1.0),
        0.0,
        "Mean-flow benchmark (NSE = 0)",
    ),
    "RRMSE_daily": MetricStyle(
        "Relative root-mean-square error, RRMSE (–)",
        "rrmse",
        (0.0, 2.0),
        (0.0, 0.5, 1.0, 1.5, 2.0),
        None,
        None,
    ),
}

# The paper panels span natural and managed process families; the larger set is
# for transparent exploratory screening, not automatic variable selection.
MAIN_CHARACTERISTICS: tuple[Characteristic, ...] = (
    Characteristic("upstream_area_GEB", "Catchment area (km²)", 0, 1e-6),
    Characteristic("aridity_FAO_PM", "Aridity, PET/P (–)", 2),
    Characteristic("ele_mt_sav", "Mean elevation (m)", 0),
    Characteristic("gwt_cm_sav", "Depth to groundwater table (cm)", 0),
    Characteristic("lka_pc_sse", "Lake-area coverage (%)", 2),
    Characteristic("rev_mc_usu", "Total reservoir volume (million m³)", 0),
)
SCREENING_CHARACTERISTICS: tuple[Characteristic, ...] = (
    *MAIN_CHARACTERISTICS,
    Characteristic("frac_snow", "Snowfall fraction (–)", 2),
    Characteristic("p_mean", "Mean precipitation (mm/day)", 1),
    Characteristic("moisture_index_FAO_PM", "Moisture index (–)", 2),
    Characteristic("tmp_dc_syr", "Mean annual temperature (°C)", 1, 0.1),
    Characteristic("seasonality_FAO_PM", "Precipitation seasonality (–)", 2),
    Characteristic("high_prec_freq", "High-precipitation days (%)", 1, 100.0),
    Characteristic("high_prec_dur", "High-precipitation duration (days)", 1),
    Characteristic("low_prec_freq", "Low-precipitation days (%)", 0, 100.0),
    Characteristic("low_prec_dur", "Low-precipitation duration (days)", 1),
    Characteristic("slp_dg_sav", "Mean slope (degrees)", 1, 0.1),
    Characteristic("sgr_dk_sav", "Stream gradient (dm/km)", 0),
    Characteristic("wet_pc_sg1", "Wetland-area coverage (%)", 2),
    Characteristic("dor_pc_pva", "Degree of regulation (–)", 0),
    Characteristic("for_pc_sse", "Forest cover (%)", 0),
    Characteristic("crp_pc_sse", "Cropland (%)", 0),
    Characteristic("pst_pc_sse", "Pasture (%)", 0),
    Characteristic("ire_pc_sse", "Irrigated area (%)", 2),
    Characteristic("urb_pc_sse", "Urban area (%)", 2),
    Characteristic("gla_pc_sse", "Glacier area (%)", 2),
    Characteristic("kar_pc_sse", "Karst area (%)", 1),
    Characteristic("cly_pc_sav", "Clay content (%)", 0),
    Characteristic("snd_pc_sav", "Sand content (%)", 0),
    Characteristic("swc_pc_syr", "Soil-water content (%)", 0),
    Characteristic("hft_ix_s09", "Human-footprint index (–)", 2, 0.01),
    Characteristic("ppd_pk_sav", "Population density (people/km²)", 0),
    Characteristic("rdd_mk_sav", "Road density (m/km²)", 0),
)


def enrich_discharge_evaluation(
    evaluation_df: pd.DataFrame,
    attribute_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add GRDC-Caravan catchment attributes to GEB discharge scores.

    The left join retains stations absent from the openly licensed
    GRDC-Caravan subset.

    Args:
        evaluation_df: Per-station GEB discharge evaluation metrics.
        attribute_df: GRDC-Caravan attributes keyed by ``gauge_id``.

    Returns:
        Evaluation table with catchment attributes and a match indicator.

    Raises:
        ValueError: If either required station identifier is missing or duplicated.
    """
    evaluation_table: pd.DataFrame = evaluation_df.copy()
    if "station_ID" not in evaluation_table.columns:
        if evaluation_table.index.name == "station_ID":
            evaluation_table = evaluation_table.reset_index()
        else:
            raise ValueError("Evaluation metrics have no station_ID column.")
    if "gauge_id" not in attribute_df.columns:
        raise ValueError("GRDC-Caravan attributes have no gauge_id column.")
    if attribute_df["gauge_id"].duplicated().any():
        raise ValueError("GRDC-Caravan attributes contain duplicate gauge_id values.")

    evaluation_table["gauge_id"] = evaluation_table["station_ID"].map(
        format_grdc_station_key
    )
    enriched_table: pd.DataFrame = evaluation_table.merge(
        attribute_df,
        on="gauge_id",
        how="left",
        validate="many_to_one",
        indicator="_grdc_caravan_merge",
    )
    enriched_table["grdc_caravan_matched"] = (
        enriched_table.pop("_grdc_caravan_merge") == "both"
    )
    return enriched_table


def calculate_characteristic_associations(
    evaluation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Screen catchment characteristics against KGE, NSE, and RRMSE.

    Spearman correlations are complemented by correlations between
    within-country percentile ranks. Benjamini-Hochberg correction is applied
    separately to each metric and correlation type.

    Args:
        evaluation_df: Enriched per-station discharge evaluation table.

    Returns:
        Long table of sample sizes, correlations, p-values, and FDR q-values.

    Raises:
        ValueError: If matched stations, metrics, country, or attributes are missing.
    """
    required_columns: set[str] = {
        "grdc_caravan_matched",
        "country",
        *METRIC_STYLES,
        *(item.column for item in SCREENING_CHARACTERISTICS),
    }
    missing_columns: set[str] = required_columns - set(evaluation_df.columns)
    if missing_columns:
        raise ValueError(
            f"Enriched discharge metrics are missing columns: {sorted(missing_columns)}"
        )
    matched_stations: pd.DataFrame = evaluation_df[
        evaluation_df["grdc_caravan_matched"]
    ].copy()
    if matched_stations.empty:
        raise ValueError("No evaluated stations match GRDC-Caravan attributes.")

    rows: list[dict[str, float | int | str]] = []
    for metric_column in METRIC_STYLES:
        for characteristic in SCREENING_CHARACTERISTICS:
            analysis_table: pd.DataFrame = matched_stations[
                ["country", characteristic.column, metric_column]
            ].copy()
            analysis_table[characteristic.column] = pd.to_numeric(
                analysis_table[characteristic.column], errors="coerce"
            )
            analysis_table[metric_column] = pd.to_numeric(
                analysis_table[metric_column], errors="coerce"
            )
            analysis_table = analysis_table.dropna()
            if (
                len(analysis_table) < 3
                or analysis_table[characteristic.column].nunique() < 2
            ):
                continue

            raw_result = spearmanr(
                analysis_table[characteristic.column], analysis_table[metric_column]
            )
            # Country ranks reduce broad geographic confounding while retaining
            # monotonic relationships and tied zero values.
            country_ranks: pd.DataFrame = cast(
                pd.DataFrame,
                analysis_table.groupby("country")[
                    [characteristic.column, metric_column]
                ].rank(pct=True),
            )
            country_result = spearmanr(
                country_ranks[characteristic.column], country_ranks[metric_column]
            )
            rows.append(
                {
                    "metric": metric_column,
                    "variable": characteristic.column,
                    "characteristic": characteristic.label,
                    "n": len(analysis_table),
                    "spearman_rho": float(raw_result.statistic),
                    "p_value": float(raw_result.pvalue),
                    "within_country_rho": float(country_result.statistic),
                    "within_country_p_value": float(country_result.pvalue),
                }
            )

    association_df: pd.DataFrame = pd.DataFrame(rows)
    for p_column, q_column in (
        ("p_value", "fdr_q_value"),
        ("within_country_p_value", "within_country_fdr_q_value"),
    ):
        association_df[q_column] = association_df.groupby("metric")[p_column].transform(
            lambda values: multipletests(values, method="fdr_bh")[1]
        )
    return association_df


def _categorize_characteristic(values: pd.Series, decimals: int) -> pd.Series:
    """Divide a numeric characteristic into four interpretable groups.

    Zero is separated when at least one third of stations are zero and the
    positive values are split into three groups. Otherwise sample quartiles are
    used. This makes high values visible while avoiding empty
    quantile groups for sparse land cover and regulation attributes.

    Args:
        values: Continuous catchment characteristic.
        decimals: Decimal places used in category labels.

    Returns:
        Ordered categorical series with four classes.

    Raises:
        ValueError: If four distinct groups cannot be formed.
    """
    numeric_values: pd.Series = pd.to_numeric(values, errors="coerce")
    valid_values: pd.Series = numeric_values.dropna()
    separate_zero: bool = bool((valid_values == 0.0).mean() >= 1.0 / 3.0)
    if separate_zero:
        positive_values: pd.Series = valid_values[valid_values > 0.0]
        if positive_values.empty:
            raise ValueError("A characteristic has no positive values.")
        breaks: list[float] = [0.0, *positive_values.quantile([1.0 / 3.0, 2.0 / 3.0])]
    else:
        breaks = [*valid_values.quantile([0.25, 0.50, 0.75])]
    if len(set(breaks)) != len(breaks):
        raise ValueError("A characteristic needs four distinct category ranges.")

    formatted_breaks: list[str] = [f"{value:.{decimals}f}" for value in breaks]
    if separate_zero:
        upper_labels: list[str] = [
            "None\n0",
            f"Low\n>0–{formatted_breaks[1]}",
            f"Medium\n{formatted_breaks[1]}–{formatted_breaks[2]}",
            f"High\n>{formatted_breaks[2]}",
        ]
    else:
        upper_labels = [
            f"Low\n≤{formatted_breaks[0]}",
            f"Medium\n{formatted_breaks[0]}–{formatted_breaks[1]}",
            f"High\n{formatted_breaks[1]}–{formatted_breaks[2]}",
            f"Very high\n>{formatted_breaks[2]}",
        ]

    return pd.cut(
        numeric_values,
        bins=[-np.inf, *breaks, np.inf],
        labels=upper_labels,
        include_lowest=True,
        ordered=True,
    )


def _finish_characteristic_axis(
    axis: plt.Axes,
    style: MetricStyle,
    positions: np.ndarray,
    category_values: pd.Index,
    title: str,
    title_size: float,
    title_pad: float,
    x_label_size: float,
    y_label_size: float,
) -> None:
    """Apply shared styling to one characteristic panel.

    Args:
        axis: Matplotlib axis to style.
        style: Metric-specific axis limits, ticks, and benchmark.
        positions: X positions of the characteristic groups.
        category_values: Ordered category labels.
        title: Panel title.
        title_size: Panel-title font size.
        title_pad: Padding below the panel title.
        x_label_size: X-axis tick-label font size.
        y_label_size: Y-axis tick-label font size.
    """
    if style.benchmark is not None:
        axis.axhline(
            style.benchmark, color="#B33A3A", linestyle="--", linewidth=0.9, zorder=0
        )
    axis.set_title(
        title, loc="left", fontsize=title_size, fontweight="bold", pad=title_pad
    )
    axis.set_xticks(positions, [str(label) for label in category_values])
    axis.set_ylim(style.limits)
    axis.set_yticks(style.ticks)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.tick_params(axis="x", labelsize=x_label_size, length=0)
    axis.tick_params(axis="y", labelsize=y_label_size)
    axis.spines[["top", "right"]].set_visible(False)


def _draw_panel(
    axis: plt.Axes,
    metric_values: pd.Series,
    categories: pd.Series,
    title: str,
    style: MetricStyle,
    random_generator: np.random.Generator,
    compact: bool = False,
) -> None:
    """Draw one categorical skill-score panel.

    Args:
        axis: Matplotlib axis receiving the plot.
        metric_values: Dimensionless daily skill-score values.
        categories: Ordered catchment-characteristic categories.
        title: Panel title.
        style: Metric-specific axis and benchmark settings.
        random_generator: Generator used for reproducible point jitter.
        compact: Whether to use the smaller exploratory-panel typography.
    """
    plot_table: pd.DataFrame = pd.DataFrame(
        {
            "metric": pd.to_numeric(metric_values, errors="coerce"),
            "category": categories,
        }
    ).dropna()
    category_values: pd.Index = plot_table["category"].cat.categories
    grouped_values: list[np.ndarray] = [
        plot_table.loc[plot_table["category"] == label, "metric"].to_numpy(dtype=float)
        for label in category_values
    ]
    positions: np.ndarray = np.arange(1, len(category_values) + 1, dtype=float)
    colors: tuple[str, ...] = ("#DCEAF1", "#91C1D3", "#3E8CA8", "#175D75")

    boxplot: dict[str, list[object]] = axis.boxplot(
        grouped_values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#404040", "linewidth": 0.8},
        capprops={"color": "#404040", "linewidth": 0.8},
        boxprops={"edgecolor": "#404040", "linewidth": 0.8},
    )
    boxes: list[Patch] = cast(list[Patch], boxplot["boxes"])
    for box, color in zip(boxes, colors, strict=False):
        box.set_facecolor(color)

    for position, values in zip(positions, grouped_values, strict=True):
        jitter: np.ndarray = random_generator.uniform(-0.20, 0.20, size=len(values))
        axis.scatter(
            position + jitter,
            values,
            s=5 if compact else 7,
            color="#163A4A",
            alpha=0.22 if compact else 0.28,
            linewidths=0.0,
            rasterized=True,
            zorder=1,
        )
        axis.text(
            position,
            0.97,
            f"n={len(values)}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6 if compact else 7.5,
            color="#303030",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.4},
            zorder=5,
        )

    _finish_characteristic_axis(
        axis,
        style,
        positions,
        category_values,
        title,
        title_size=8.5 if compact else 10.5,
        title_pad=6 if compact else 8,
        x_label_size=5.8 if compact else 7.5,
        y_label_size=7 if compact else 8,
    )


def _draw_paired_panel(
    axis: plt.Axes,
    geb_metric_values: pd.Series,
    external_metric_values: pd.Series,
    categories: pd.Series,
    title: str,
    style: MetricStyle,
    random_generator: np.random.Generator,
) -> None:
    """Draw one paired GEB-versus-external characteristic panel.

    Args:
        axis: Matplotlib axis receiving the plot.
        geb_metric_values: GEB skill-score values.
        external_metric_values: External-model skill-score values.
        categories: Ordered catchment-characteristic categories.
        title: Panel title.
        style: Metric-specific axis and benchmark settings.
        random_generator: Generator used for reproducible point jitter.

    """
    plot_table: pd.DataFrame = pd.DataFrame(
        {
            "GEB": pd.to_numeric(geb_metric_values, errors="coerce"),
            "External model": pd.to_numeric(external_metric_values, errors="coerce"),
            "category": categories,
        }
    ).dropna()
    category_values: pd.Index = plot_table["category"].cat.categories
    positions: np.ndarray = np.arange(1, len(category_values) + 1, dtype=float)
    offsets: dict[str, float] = {"GEB": -0.17, "External model": 0.17}
    colors: dict[str, str] = {"GEB": "#1F77B4", "External model": "#D65F00"}

    for model_label, offset in offsets.items():
        grouped_values: list[np.ndarray] = [
            plot_table.loc[
                plot_table["category"] == category_label, model_label
            ].to_numpy(dtype=float)
            for category_label in category_values
        ]
        boxplot: dict[str, list[object]] = axis.boxplot(
            grouped_values,
            positions=positions + offset,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"color": "#404040", "linewidth": 0.7},
            capprops={"color": "#404040", "linewidth": 0.7},
            boxprops={"edgecolor": "#404040", "linewidth": 0.7},
        )
        boxes: list[Patch] = cast(list[Patch], boxplot["boxes"])
        for box in boxes:
            box.set_facecolor(colors[model_label])
            box.set_alpha(0.45)

        for position, values in zip(positions + offset, grouped_values, strict=True):
            jitter: np.ndarray = random_generator.uniform(-0.06, 0.06, size=len(values))
            axis.scatter(
                position + jitter,
                values,
                s=7,
                color=colors[model_label],
                alpha=0.22,
                linewidths=0.0,
                rasterized=True,
                zorder=1,
            )

    for position, category_label in zip(positions, category_values, strict=True):
        station_count: int = int((plot_table["category"] == category_label).sum())
        axis.text(
            position,
            0.97,
            f"n={station_count}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.2,
            color="#303030",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.4},
            zorder=5,
        )

    _finish_characteristic_axis(
        axis,
        style,
        positions,
        category_values,
        title,
        title_size=10.5,
        title_pad=8,
        x_label_size=7.2,
        y_label_size=8,
    )


def _save_figure(
    figure: plt.Figure,
    output_folder: Path,
    output_stem: str,
    logger: logging.Logger,
) -> None:
    """Save one figure as editable vector files and a 300-dpi PNG.

    Args:
        figure: Matplotlib figure to save.
        output_folder: Figure output directory.
        output_stem: Filename without extension.
        logger: Logger used for export messages.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        output_path: Path = output_folder / f"{output_stem}.{extension}"
        figure.savefig(
            output_path,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
        )
        logger.info("Saved discharge characteristic figure to %s.", output_path)


def plot_skill_by_characteristics(
    evaluation_df: pd.DataFrame,
    metric_column: str,
    output_folder: Path,
    logger: logging.Logger,
    output_name_suffix: str = "",
    export: bool = True,
) -> plt.Figure:
    """Plot one daily skill metric across six selected characteristics.

    Catchment area uses every evaluated station. External characteristic panels
    use only the stations matched to GRDC-Caravan.

    Args:
        evaluation_df: Enriched per-station evaluation table.
        metric_column: One of ``KGE_daily``, ``NSE_daily``, or ``RRMSE_daily``.
        output_folder: Folder receiving SVG, PDF, and PNG outputs.
        logger: Logger used for export messages.
        output_name_suffix: Optional evaluation-period suffix for filenames.
        export: Whether to save the figure.

    Returns:
        Publication-sized Matplotlib figure.

    Raises:
        ValueError: If the metric, matched stations, or characteristics are missing.
    """
    if metric_column not in METRIC_STYLES:
        raise ValueError(f"Unsupported discharge metric: {metric_column}")
    required_columns: set[str] = {
        metric_column,
        "grdc_caravan_matched",
        *(item.column for item in MAIN_CHARACTERISTICS),
    }
    missing_columns: set[str] = required_columns - set(evaluation_df.columns)
    if missing_columns:
        raise ValueError(
            f"Enriched discharge metrics are missing columns: {sorted(missing_columns)}"
        )

    all_stations: pd.DataFrame = evaluation_df[
        pd.to_numeric(evaluation_df[metric_column], errors="coerce").notna()
    ].copy()
    matched_stations: pd.DataFrame = all_stations[
        all_stations["grdc_caravan_matched"]
    ].copy()
    if matched_stations.empty:
        raise ValueError("No evaluated stations match GRDC-Caravan attributes.")

    style: MetricStyle = METRIC_STYLES[metric_column]
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 7.2), sharey=True)
    random_generator: np.random.Generator = np.random.default_rng(42)
    for panel_index, (axis, item) in enumerate(
        zip(axes.flat, MAIN_CHARACTERISTICS, strict=True)
    ):
        stations: pd.DataFrame = (
            all_stations if item.column == "upstream_area_GEB" else matched_stations
        )
        title: str = (
            "Catchment area (km²) — all stations"
            if item.column == "upstream_area_GEB"
            else item.label
        )
        panel_letter: str = chr(ord("a") + panel_index)
        _draw_panel(
            axis,
            stations[metric_column],
            _categorize_characteristic(
                stations[item.column] * item.scale, decimals=item.decimals
            ),
            f"{panel_letter}  {title}",
            style,
            random_generator,
        )

    for axis in axes[:, 0]:
        axis.set_ylabel(style.label, fontsize=9)
    if style.benchmark is not None and style.benchmark_label is not None:
        benchmark_handle: Line2D = Line2D(
            [0],
            [0],
            color="#B33A3A",
            linestyle="--",
            linewidth=1.0,
            label=style.benchmark_label,
        )
        figure.legend(
            handles=[benchmark_handle],
            loc="lower right",
            bbox_to_anchor=(0.985, 0.015),
            frameon=False,
            fontsize=8.5,
        )
    figure.subplots_adjust(
        left=0.07, right=0.985, top=0.95, bottom=0.11, wspace=0.18, hspace=0.34
    )

    if export:
        _save_figure(
            figure,
            output_folder,
            f"discharge_{style.short_name}_by_catchment_characteristics{output_name_suffix}",
            logger,
        )
    return figure


def plot_paired_skill_by_characteristics(
    evaluation_df: pd.DataFrame,
    external_metric_df: pd.DataFrame,
    metric_column: str,
    external_model_name: str,
    output_folder: Path,
    logger: logging.Logger,
    output_name_suffix: str = "",
    export: bool = True,
) -> plt.Figure:
    """Plot paired GEB and external-model skill by catchment characteristics.

    The figure uses only stations where GEB, the external model, and the needed
    catchment characteristic are all available. This keeps every boxplot pair a
    direct matched-station comparison.

    Args:
        evaluation_df: Enriched paired-station GEB evaluation table.
        external_metric_df: External-model metrics aligned to ``evaluation_df``.
        metric_column: One of ``KGE_daily``, ``NSE_daily``, or ``RRMSE_daily``.
        external_model_name: Label of the external model.
        output_folder: Folder receiving SVG, PDF, and PNG outputs.
        logger: Logger used for export messages.
        output_name_suffix: Optional suffix for filenames.
        export: Whether to save the figure.

    Returns:
        Publication-sized paired-comparison Matplotlib figure.

    Raises:
        ValueError: If the metric or matched GRDC-Caravan stations are missing.
    """
    if metric_column not in METRIC_STYLES:
        raise ValueError(f"Unsupported discharge metric: {metric_column}")
    external_column: str = metric_column.removesuffix("_daily")
    if external_column not in external_metric_df.columns:
        raise ValueError(
            f"External model '{external_model_name}' has no {external_column} column."
        )

    plot_df: pd.DataFrame = evaluation_df.copy()
    plot_df["external_skill_score"] = pd.to_numeric(
        external_metric_df[external_column], errors="coerce"
    ).to_numpy(dtype=float)
    required_columns: set[str] = {
        metric_column,
        "external_skill_score",
        "grdc_caravan_matched",
        *(item.column for item in MAIN_CHARACTERISTICS),
    }
    missing_columns: set[str] = required_columns - set(plot_df.columns)
    if missing_columns:
        raise ValueError(
            f"Paired discharge metrics are missing columns: {sorted(missing_columns)}"
        )

    style: MetricStyle = METRIC_STYLES[metric_column]
    figure, axes = plt.subplots(2, 3, figsize=(11.4, 7.4), sharey=True)
    random_generator: np.random.Generator = np.random.default_rng(42)
    for panel_index, (axis, item) in enumerate(
        zip(axes.flat, MAIN_CHARACTERISTICS, strict=True)
    ):
        stations: pd.DataFrame = (
            plot_df
            if item.column == "upstream_area_GEB"
            else plot_df[plot_df["grdc_caravan_matched"]]
        )
        if stations.empty:
            raise ValueError("No paired stations match GRDC-Caravan attributes.")
        title: str = (
            "Catchment area (km²) — paired stations"
            if item.column == "upstream_area_GEB"
            else item.label
        )
        panel_letter: str = chr(ord("a") + panel_index)
        _draw_paired_panel(
            axis,
            stations[metric_column],
            stations["external_skill_score"],
            _categorize_characteristic(
                stations[item.column] * item.scale, decimals=item.decimals
            ),
            f"{panel_letter}  {title}",
            style,
            random_generator,
        )

    for axis in axes[:, 0]:
        axis.set_ylabel(style.label, fontsize=9)
    legend_handles: list[Patch | Line2D] = [
        Patch(facecolor="#1F77B4", edgecolor="#404040", alpha=0.45, label="GEB"),
        Patch(
            facecolor="#D65F00",
            edgecolor="#404040",
            alpha=0.45,
            label=external_model_name,
        ),
    ]
    if style.benchmark is not None and style.benchmark_label is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#B33A3A",
                linestyle="--",
                linewidth=1.0,
                label=style.benchmark_label,
            )
        )
    figure.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.015),
        ncols=3 if len(legend_handles) == 3 else 2,
        frameon=False,
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.07, right=0.985, top=0.95, bottom=0.11, wspace=0.18, hspace=0.34
    )

    if export:
        _save_figure(
            figure,
            output_folder,
            (
                f"discharge_{style.short_name}_by_catchment_characteristics"
                f"_paired{output_name_suffix}"
            ),
            logger,
        )
    return figure


def plot_characteristic_screening(
    evaluation_df: pd.DataFrame,
    association_df: pd.DataFrame,
    output_folder: Path,
    logger: logging.Logger,
    output_name_suffix: str = "",
    export: bool = True,
) -> plt.Figure:
    """Plot the broad exploratory KGE screen across 32 characteristics.

    Args:
        evaluation_df: Enriched per-station evaluation table.
        association_df: Output from :func:`calculate_characteristic_associations`.
        output_folder: Folder receiving SVG, PDF, and PNG outputs.
        logger: Logger used for export messages.
        output_name_suffix: Optional evaluation-period suffix for filenames.
        export: Whether to save the figure.

    Returns:
        Large multi-panel exploratory Matplotlib figure.

    Raises:
        ValueError: If no stations match GRDC-Caravan.
    """
    matched_stations: pd.DataFrame = evaluation_df[
        evaluation_df["grdc_caravan_matched"]
    ].copy()
    if matched_stations.empty:
        raise ValueError("No evaluated stations match GRDC-Caravan attributes.")
    kge_associations: pd.DataFrame = association_df[
        association_df["metric"] == "KGE_daily"
    ].set_index("variable")

    column_count: int = 4
    row_count: int = int(np.ceil(len(SCREENING_CHARACTERISTICS) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(13.0, 2.25 * row_count),
        sharey=True,
    )
    random_generator: np.random.Generator = np.random.default_rng(42)
    style: MetricStyle = METRIC_STYLES["KGE_daily"]
    for panel_index, (axis, item) in enumerate(
        zip(axes.flat, SCREENING_CHARACTERISTICS, strict=False)
    ):
        association: pd.Series = kge_associations.loc[item.column]
        title: str = (
            f"{panel_index + 1}  {item.label}\n"
            f"rₛ={association['spearman_rho']:+.2f}; q={association['fdr_q_value']:.2g}"
        )
        _draw_panel(
            axis,
            matched_stations["KGE_daily"],
            _categorize_characteristic(
                matched_stations[item.column] * item.scale, decimals=item.decimals
            ),
            title,
            style,
            random_generator,
            compact=True,
        )
    for axis in axes.flat[len(SCREENING_CHARACTERISTICS) :]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel("KGE (–)", fontsize=8)
    figure.subplots_adjust(
        left=0.055, right=0.99, top=0.985, bottom=0.035, wspace=0.18, hspace=0.60
    )

    if export:
        _save_figure(
            figure,
            output_folder,
            f"discharge_kge_characteristic_screening{output_name_suffix}",
            logger,
        )
    return figure
