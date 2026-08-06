"""Relate discharge KGE and its components to GRDC-Caravan attributes."""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

from geb.evaluate.workflows.external_skill_scores import format_grdc_station_key


@dataclass(frozen=True)
class Characteristic:
    """Metadata for one GRDC-Caravan catchment characteristic.

    Args:
        column: Column in the enriched discharge-evaluation table.
        label: Publication label including units.
        scale: Factor converting stored values to displayed units.
        logarithmic_x: Whether the 32-panel atlas uses a base-10 x-axis.
    """

    column: str
    label: str
    scale: float = 1.0
    logarithmic_x: bool = False


@dataclass(frozen=True)
class KGEComponentTarget:
    """One KGE quantity displayed in the association heatmap.

    Args:
        column: Column in the discharge-evaluation table.
        label: Concise heatmap-column label.
    """

    column: str
    label: str


@dataclass(frozen=True)
class KGERelationshipPanel:
    """Metadata for one relationship panel linked to the heatmap.

    Args:
        column: Prepared characteristic column.
        title: Panel title including its alphabetic label.
        x_label: Horizontal-axis label including units.
    """

    column: str
    title: str
    x_label: str


# These 32 attributes cover climate, topography, land cover, soils, hydrology,
# and human influence without pre-selecting variables by model performance.
SCREENING_CHARACTERISTICS: tuple[Characteristic, ...] = (
    Characteristic(
        "upstream_area_GEB",
        "Upstream catchment area (km²)",
        scale=1e-6,
        logarithmic_x=True,
    ),
    Characteristic("aridity_FAO_PM", "Aridity, PET/P (–)"),
    Characteristic("ele_mt_sav", "Mean elevation (m)"),
    Characteristic("gwt_cm_sav", "Depth to groundwater table (cm)"),
    Characteristic("lka_pc_sse", "Lake-area coverage (%)"),
    Characteristic("rev_mc_usu", "Total reservoir volume (million m³)"),
    Characteristic("frac_snow", "Snowfall fraction (–)"),
    Characteristic("p_mean", "Mean precipitation (mm/day)"),
    Characteristic("moisture_index_FAO_PM", "Moisture index (–)"),
    Characteristic("tmp_dc_syr", "Mean annual temperature (°C)", scale=0.1),
    Characteristic("seasonality_FAO_PM", "Precipitation seasonality (–)"),
    Characteristic("high_prec_freq", "High-precipitation days (%)", scale=100.0),
    Characteristic("high_prec_dur", "High-precipitation duration (days)"),
    Characteristic("low_prec_freq", "Low-precipitation days (%)", scale=100.0),
    Characteristic("low_prec_dur", "Low-precipitation duration (days)"),
    Characteristic("slp_dg_sav", "Mean slope (degrees)", scale=0.1),
    Characteristic("sgr_dk_sav", "Stream gradient (dm/km)"),
    Characteristic("wet_pc_sg1", "Wetland-area coverage (%)"),
    Characteristic("dor_pc_pva", "Degree of regulation (%)"),
    Characteristic("for_pc_sse", "Forest cover (%)"),
    Characteristic("crp_pc_sse", "Cropland (%)"),
    Characteristic("pst_pc_sse", "Pasture (%)"),
    Characteristic("ire_pc_sse", "Irrigated area (%)"),
    Characteristic("urb_pc_sse", "Urban area (%)"),
    Characteristic("gla_pc_sse", "Glacier area (%)"),
    Characteristic("kar_pc_sse", "Karst area (%)"),
    Characteristic("cly_pc_sav", "Clay content (%)"),
    Characteristic("snd_pc_sav", "Sand content (%)"),
    Characteristic("swc_pc_syr", "Soil-water content (%)"),
    Characteristic("hft_ix_s09", "Human-footprint index (–)", scale=0.01),
    Characteristic("ppd_pk_sav", "Population density (people/km²)"),
    Characteristic("rdd_mk_sav", "Road density (m/km²)"),
)

KGE_COMPONENT_TARGETS: tuple[KGEComponentTarget, ...] = (
    KGEComponentTarget("KGE_correlation_daily", "Correlation r"),
    KGEComponentTarget("KGE_bias_ratio_daily", "Mean-flow ratio β"),
    KGEComponentTarget("KGE_variability_ratio_daily", "Variability ratio α"),
    KGEComponentTarget("KGE_daily", "Overall KGE"),
)

# The selected panels represent distinct topographic, subsurface, channel, and
# climate relationships while avoiding redundant variants of the same signal.
KGE_RELATIONSHIP_PANELS: tuple[KGERelationshipPanel, ...] = (
    KGERelationshipPanel("ele_mt_sav", "b) Mean elevation", "Elevation (m)"),
    KGERelationshipPanel(
        "gwt_cm_sav",
        "c) Depth to groundwater table",
        "Depth to groundwater table (cm)",
    ),
    KGERelationshipPanel("sgr_dk_sav", "d) Stream gradient", "Stream gradient (dm/km)"),
    KGERelationshipPanel(
        "low_prec_freq",
        "e) Low-precipitation days",
        "Low-precipitation days (%)",
    ),
)

# These tones are sampled from the positive half of ColorBrewer's BrBG scale
# so the relationship panels and signed-correlation heatmap read as one design.
STATION_COLOR: str = "#35978F"
LOWESS_COLOR: str = "#01665E"
LOWESS_INTERVAL_COLOR: str = "#80CDC1"


def enrich_discharge_evaluation(
    evaluation_df: pd.DataFrame,
    attribute_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add GRDC-Caravan catchment attributes to GEB discharge scores.

    The left join retains evaluated stations absent from the openly licensed
    GRDC-Caravan subset.

    Args:
        evaluation_df: Per-station GEB discharge metrics.
        attribute_df: GRDC-Caravan attributes keyed by ``gauge_id``.

    Returns:
        Evaluation table with catchment attributes and a match indicator.

    Raises:
        ValueError: If either station identifier is missing or duplicated.
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


def prepare_kge_characteristic_analysis(
    evaluation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare matched stations for KGE-characteristic analysis.

    Characteristics are converted once to their displayed units. KGE and its
    three original components remain untransformed so that bias and variability
    direction are preserved.

    Args:
        evaluation_df: Discharge metrics enriched with GRDC-Caravan attributes.

    Returns:
        Matched station table in display units.

    Raises:
        ValueError: If required columns or matched stations are unavailable.
    """
    required_columns: set[str] = {
        "grdc_caravan_matched",
        *(target.column for target in KGE_COMPONENT_TARGETS),
        *(item.column for item in SCREENING_CHARACTERISTICS),
    }
    missing_columns: set[str] = required_columns - set(evaluation_df.columns)
    if missing_columns:
        raise ValueError(
            f"Enriched discharge metrics are missing columns: {sorted(missing_columns)}"
        )

    analysis_table: pd.DataFrame = evaluation_df.loc[
        evaluation_df["grdc_caravan_matched"]
    ].copy()
    if analysis_table.empty:
        raise ValueError("No evaluated stations match GRDC-Caravan attributes.")

    numeric_columns: tuple[str, ...] = (
        *(target.column for target in KGE_COMPONENT_TARGETS),
        *(item.column for item in SCREENING_CHARACTERISTICS),
    )
    for column in numeric_columns:
        analysis_table[column] = pd.to_numeric(analysis_table[column], errors="coerce")
    for characteristic in SCREENING_CHARACTERISTICS:
        analysis_table[characteristic.column] *= characteristic.scale
    return analysis_table


def calculate_kge_component_associations(
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate Spearman associations with KGE and its three components.

    Args:
        analysis_df: Output from :func:`prepare_kge_characteristic_analysis`.

    Returns:
        Long table containing sample sizes, Spearman correlations, and p-values.

    Raises:
        ValueError: If no valid characteristic-target pair is available.
    """
    association_rows: list[dict[str, float | int | str]] = []
    for characteristic in SCREENING_CHARACTERISTICS:
        for target in KGE_COMPONENT_TARGETS:
            pair_table: pd.DataFrame = analysis_df[
                [characteristic.column, target.column]
            ].dropna()
            if (
                len(pair_table) < 3
                or pair_table[characteristic.column].nunique() < 2
                or pair_table[target.column].nunique() < 2
            ):
                continue
            correlation_result = spearmanr(
                pair_table[characteristic.column], pair_table[target.column]
            )
            association_rows.append(
                {
                    "variable": characteristic.column,
                    "characteristic": characteristic.label,
                    "target": target.column,
                    "target_label": target.label,
                    "n": len(pair_table),
                    "spearman_rho": float(correlation_result.statistic),
                    "p_value": float(correlation_result.pvalue),
                }
            )
    if not association_rows:
        raise ValueError("No valid KGE-characteristic associations are available.")

    return pd.DataFrame(association_rows)


def _save_figure(
    figure: plt.Figure,
    output_folder: Path,
    output_stem: str,
    logger: logging.Logger,
) -> None:
    """Save a figure as editable vector files and a 300-dpi PNG.

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


def _fit_lowess_to_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    evaluation_x: np.ndarray,
    robust_iterations: int,
) -> np.ndarray:
    """Fit LOWESS and interpolate fitted values to a common grid.

    Args:
        x_values: Finite characteristic values, optionally log-transformed.
        y_values: Corresponding dimensionless KGE values.
        evaluation_x: Grid in the same units as ``x_values``.
        robust_iterations: Number of LOWESS residual-reweighting iterations.

    Returns:
        Interpolated LOWESS values on ``evaluation_x``.

    Raises:
        ValueError: If fewer than two distinct x-values are available.
    """
    unique_x: np.ndarray = np.unique(x_values)
    if len(unique_x) < 2:
        raise ValueError("A LOWESS curve needs two distinct characteristic values.")

    stable_x_values: np.ndarray = x_values.copy()
    if len(unique_x) < len(x_values):
        minimum_gap: float = float(np.min(np.diff(unique_x)))
        jitter_width: float = minimum_gap * 1e-6
        for tied_value in unique_x:
            tied_indices: np.ndarray = np.flatnonzero(x_values == tied_value)
            if len(tied_indices) > 1:
                # This sub-pixel separation prevents division by zero while
                # retaining the full statistical weight of zero-heavy data.
                stable_x_values[tied_indices] += np.linspace(
                    -0.5 * jitter_width,
                    0.5 * jitter_width,
                    len(tied_indices),
                )

    fitted_values: np.ndarray = np.asarray(
        lowess(
            y_values,
            stable_x_values,
            frac=0.35,
            it=robust_iterations,
            return_sorted=True,
        ),
        dtype=float,
    )
    fitted_x, unique_indices = np.unique(fitted_values[:, 0], return_index=True)
    return np.interp(evaluation_x, fitted_x, fitted_values[unique_indices, 1])


def _calculate_lowess_interval(
    x_values: np.ndarray,
    y_values: np.ndarray,
    evaluation_x: np.ndarray,
    random_generator: np.random.Generator,
    bootstrap_repetitions: int = 150,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate robust LOWESS and a station-bootstrap confidence interval.

    Args:
        x_values: Finite characteristic values, optionally log-transformed.
        y_values: Corresponding dimensionless KGE values.
        evaluation_x: Grid receiving the smooth and interval.
        random_generator: Generator for reproducible station resampling.
        bootstrap_repetitions: Number of station bootstrap samples.

    Returns:
        LOWESS estimate and lower and upper 95% confidence limits.

    Raises:
        ValueError: If fewer than ten paired stations are available.
    """
    station_count: int = len(x_values)
    if station_count < 10:
        raise ValueError("A LOWESS interval needs at least ten paired stations.")
    central_curve: np.ndarray = _fit_lowess_to_grid(
        x_values, y_values, evaluation_x, robust_iterations=2
    )
    bootstrap_curves: np.ndarray = np.empty(
        (bootstrap_repetitions, len(evaluation_x)), dtype=float
    )
    for repetition in range(bootstrap_repetitions):
        sample_indices: np.ndarray = random_generator.integers(
            0, station_count, size=station_count
        )
        bootstrap_curves[repetition] = _fit_lowess_to_grid(
            x_values[sample_indices],
            y_values[sample_indices],
            evaluation_x,
            robust_iterations=1,
        )
    lower_limit, upper_limit = np.quantile(bootstrap_curves, [0.025, 0.975], axis=0)
    return central_curve, lower_limit, upper_limit


def _relationship_data(
    analysis_df: pd.DataFrame,
    characteristic: Characteristic,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare station values and the central LOWESS evaluation grid.

    The plotted range is limited to the 1st–99th characteristic percentiles so
    isolated extremes do not compress the scientifically relevant pattern.

    Args:
        analysis_df: Prepared station-level analysis table.
        characteristic: Characteristic metadata and x-axis transformation.

    Returns:
        Displayed x-values, KGE values, model-space x-values, and evaluation grid.

    Raises:
        ValueError: If fewer than ten valid pairs or two distinct x-values remain.
    """
    pair_table: pd.DataFrame = analysis_df[
        [characteristic.column, "KGE_daily"]
    ].dropna()
    if characteristic.logarithmic_x:
        pair_table = pair_table.loc[pair_table[characteristic.column] > 0.0]
    if len(pair_table) < 10:
        raise ValueError(
            f"Characteristic {characteristic.column} has fewer than ten KGE pairs."
        )

    displayed_x: np.ndarray = pair_table[characteristic.column].to_numpy(dtype=float)
    kge_values: np.ndarray = pair_table["KGE_daily"].to_numpy(dtype=float)
    model_x: np.ndarray = (
        np.log10(displayed_x) if characteristic.logarithmic_x else displayed_x
    )
    lower_x, upper_x = np.quantile(model_x, [0.01, 0.99])
    if not upper_x > lower_x:
        raise ValueError(
            f"Characteristic {characteristic.column} needs two distinct x-values."
        )
    evaluation_x: np.ndarray = np.linspace(lower_x, upper_x, 180)
    return displayed_x, kge_values, model_x, evaluation_x


def _plot_relationship_panel(
    axis: plt.Axes,
    analysis_df: pd.DataFrame,
    association_df: pd.DataFrame,
    characteristic: Characteristic,
    title: str,
    x_label: str,
    random_generator: np.random.Generator,
    kge_axis_limits: tuple[float, float],
) -> None:
    """Plot one station-level KGE relationship with uncertainty.

    Args:
        axis: Matplotlib axis receiving the panel.
        analysis_df: Prepared matched-station analysis table.
        association_df: KGE-component Spearman association table.
        characteristic: Characteristic metadata.
        title: Panel title.
        x_label: Horizontal-axis label including units.
        random_generator: Generator for reproducible bootstrap resampling.
        kge_axis_limits: Shared lower and upper dimensionless KGE limits.

    Raises:
        ValueError: If the KGE association is missing.
    """
    displayed_x, kge_values, model_x, evaluation_x = _relationship_data(
        analysis_df, characteristic
    )
    central_curve, lower_limit, upper_limit = _calculate_lowess_interval(
        model_x, kge_values, evaluation_x, random_generator
    )
    displayed_evaluation_x: np.ndarray = (
        np.power(10.0, evaluation_x) if characteristic.logarithmic_x else evaluation_x
    )

    axis.fill_between(
        displayed_evaluation_x,
        lower_limit,
        upper_limit,
        color=LOWESS_INTERVAL_COLOR,
        alpha=0.20,
        linewidth=0.0,
        zorder=1,
    )
    axis.scatter(
        displayed_x,
        kge_values,
        s=9,
        color=STATION_COLOR,
        alpha=0.27,
        linewidths=0.0,
        rasterized=True,
        zorder=2,
    )
    axis.plot(
        displayed_evaluation_x,
        central_curve,
        color=LOWESS_COLOR,
        linewidth=2.3,
        zorder=4,
    )
    association_rows: pd.DataFrame = association_df.loc[
        (association_df["variable"] == characteristic.column)
        & (association_df["target"] == "KGE_daily")
    ]
    if len(association_rows) != 1:
        raise ValueError(f"Expected one KGE association for {characteristic.column}.")
    rho: float = float(association_rows.iloc[0]["spearman_rho"])
    axis.text(
        0.025,
        0.94,
        f"Spearman ρ = {rho:+.2f}",
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
            "alpha": 0.82,
        },
    )

    if characteristic.logarithmic_x:
        axis.set_xscale("log")
    axis.set_xlim(displayed_evaluation_x[0], displayed_evaluation_x[-1])
    axis.set_ylim(kge_axis_limits)
    axis.set_xlabel(x_label, fontsize=8.5)
    axis.set_ylabel("KGE (–)", fontsize=8.5)
    axis.set_title(title, loc="left", fontsize=10.0, fontweight="bold", pad=7)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelsize=7.4, pad=3)
    axis.tick_params(axis="y", labelsize=7.4)
    axis.spines[["top", "right"]].set_visible(False)


def _relationship_legend_handles() -> list[Line2D | Patch]:
    """Create the shared station, LOWESS, and confidence-interval legend.

    Returns:
        Matplotlib legend handles shared by both retained figures.
    """
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=STATION_COLOR,
            markeredgecolor="none",
            markersize=4.5,
            alpha=0.5,
            label="Station",
        ),
        Line2D(
            [0],
            [0],
            color=LOWESS_COLOR,
            linewidth=2.3,
            label="Robust LOWESS",
        ),
        Patch(
            facecolor=LOWESS_INTERVAL_COLOR,
            edgecolor="none",
            alpha=0.18,
            label="95% bootstrap CI",
        ),
    ]


def _ranked_characteristic_order(
    association_matrix: pd.DataFrame,
) -> list[str]:
    """Rank characteristics by overall-KGE association strength.

    Args:
        association_matrix: Spearman correlations indexed by characteristic.

    Returns:
        Characteristic columns sorted by decreasing absolute overall-KGE
        correlation.

    Raises:
        ValueError: If an expected characteristic is unavailable.
    """
    expected_columns: list[str] = [
        characteristic.column for characteristic in SCREENING_CHARACTERISTICS
    ]
    missing_columns: set[str] = set(expected_columns) - set(association_matrix.index)
    if missing_columns:
        raise ValueError(
            f"KGE association table is missing characteristics: {missing_columns}"
        )
    return (
        association_matrix.loc[expected_columns, "KGE_daily"]
        .abs()
        .sort_values(ascending=False, kind="stable")
        .index.tolist()
    )


def plot_kge_characteristic_heatmaps(
    analysis_df: pd.DataFrame,
    association_df: pd.DataFrame,
    output_folder: Path,
    logger: logging.Logger,
    output_name_suffix: str = "",
    export: bool = True,
) -> plt.Figure:
    """Plot the KGE-component heatmap and four linked relationships.

    Args:
        analysis_df: Prepared matched-station analysis table.
        association_df: KGE-component association table.
        output_folder: Folder receiving SVG, PDF, and PNG outputs.
        logger: Logger used for export messages.
        output_name_suffix: Optional evaluation-period filename suffix.
        export: Whether to save the figure.

    Returns:
        Figure containing the heatmap and four continuous relationship panels.

    Raises:
        ValueError: If associations or selected variables are incomplete.
    """
    association_matrix: pd.DataFrame = association_df.pivot(
        index="variable", columns="target", values="spearman_rho"
    )
    significance_matrix: pd.DataFrame = association_df.pivot(
        index="variable", columns="target", values="p_value"
    )
    labels_by_column: dict[str, str] = {
        item.column: item.label for item in SCREENING_CHARACTERISTICS
    }
    characteristics_by_column: dict[str, Characteristic] = {
        item.column: item for item in SCREENING_CHARACTERISTICS
    }
    target_columns: list[str] = [target.column for target in KGE_COMPONENT_TARGETS]
    required_variables: list[str] = list(characteristics_by_column)
    selected_variables: set[str] = {panel.column for panel in KGE_RELATIONSHIP_PANELS}
    if not set(required_variables).issubset(association_matrix.index):
        raise ValueError("KGE association table does not contain every characteristic.")
    if not {"KGE_daily", *selected_variables}.issubset(analysis_df.columns):
        raise ValueError("KGE relationship-panel inputs are incomplete.")

    variable_order: list[str] = _ranked_characteristic_order(association_matrix)
    association_matrix = association_matrix.loc[variable_order, target_columns]
    significance_matrix = significance_matrix.loc[variable_order, target_columns]

    figure: plt.Figure = plt.figure(figsize=(13.6, 10.9))
    outer_grid = figure.add_gridspec(1, 2, width_ratios=(1.18, 1.0), wspace=0.36)
    association_axis: plt.Axes = figure.add_subplot(outer_grid[0, 0])
    correlation_grid = outer_grid[0, 1].subgridspec(
        5,
        1,
        height_ratios=(1.0, 1.0, 1.0, 1.0, 0.005),
        hspace=0.52,
    )
    correlation_axes: list[plt.Axes] = [
        figure.add_subplot(correlation_grid[row_index, 0])
        for row_index in range(len(KGE_RELATIONSHIP_PANELS))
    ]

    association_image = association_axis.imshow(
        association_matrix.to_numpy(dtype=float),
        cmap="BrBG",
        norm=TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5),
        aspect="auto",
    )
    association_axis.set_xticks(
        np.arange(len(target_columns)),
        [target.label for target in KGE_COMPONENT_TARGETS],
    )
    association_axis.set_yticks(
        np.arange(len(variable_order)),
        [labels_by_column[variable] for variable in variable_order],
    )
    for row_index in range(len(variable_order)):
        for column_index in range(len(target_columns)):
            rho: float = float(association_matrix.iloc[row_index, column_index])
            p_value: float = float(significance_matrix.iloc[row_index, column_index])
            significance_marker: str = "*" if p_value < 0.05 else ""
            rho_text: str = "0.00" if abs(rho) < 0.005 else f"{rho:+.2f}"
            text_color: str = "white" if abs(rho) >= 0.34 else "#222222"
            association_axis.text(
                column_index,
                row_index,
                f"{rho_text}{significance_marker}",
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
        labelsize=7.4,
        length=0,
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        pad=4,
    )
    association_axis.tick_params(axis="y", labelsize=7.3, length=0)
    association_axis.set_xticks(np.arange(-0.5, 4.0, 1.0), minor=True)
    association_axis.set_yticks(np.arange(-0.5, len(variable_order), 1.0), minor=True)
    association_axis.grid(which="minor", color="white", linewidth=0.8)
    association_axis.tick_params(which="minor", bottom=False, left=False)
    association_colorbar = figure.colorbar(
        association_image,
        ax=association_axis,
        orientation="horizontal",
        pad=0.035,
        fraction=0.032,
    )
    association_colorbar.set_ticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    association_colorbar.ax.tick_params(labelsize=7.4, length=3.0)
    association_colorbar.set_label(
        "Spearman rank correlation, ρ  (* nominal p < 0.05)", fontsize=8.5
    )

    random_generator: np.random.Generator = np.random.default_rng(42)
    lower_kge, upper_kge = analysis_df["KGE_daily"].quantile([0.10, 0.90])
    kge_axis_limits: tuple[float, float] = (float(lower_kge), float(upper_kge))
    for correlation_axis, panel in zip(
        correlation_axes, KGE_RELATIONSHIP_PANELS, strict=True
    ):
        characteristic: Characteristic = characteristics_by_column[panel.column]
        _plot_relationship_panel(
            axis=correlation_axis,
            analysis_df=analysis_df,
            association_df=association_df,
            characteristic=characteristic,
            title=panel.title,
            x_label=panel.x_label,
            random_generator=random_generator,
            kge_axis_limits=kge_axis_limits,
        )
        characteristic_row: int = variable_order.index(panel.column)
        association_axis.add_patch(
            Rectangle(
                (target_columns.index("KGE_daily") - 0.5, characteristic_row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#111111",
                linewidth=2.2,
                zorder=7,
                clip_on=False,
            )
        )
        association_axis.get_yticklabels()[characteristic_row].set_fontweight("bold")

    figure.subplots_adjust(left=0.255, right=0.985, top=0.96, bottom=0.09)
    figure.canvas.draw()
    for correlation_axis, panel in zip(
        correlation_axes, KGE_RELATIONSHIP_PANELS, strict=True
    ):
        characteristic_row = variable_order.index(panel.column)
        start_display: np.ndarray = association_axis.transData.transform(
            (len(target_columns) - 0.48, characteristic_row)
        )
        start_figure: np.ndarray = figure.transFigure.inverted().transform(
            start_display
        )
        end_figure: tuple[float, float] = (
            correlation_axis.get_position().x0 - 0.008,
            correlation_axis.get_position().y1,
        )
        overall_kge_rho: float = float(
            association_matrix.loc[panel.column, "KGE_daily"]
        )
        normalized_rho: np.ndarray = np.asarray(
            association_image.norm(np.asarray([overall_kge_rho], dtype=float)),
            dtype=float,
        )
        connector_rgba: np.ndarray = np.asarray(
            association_image.cmap(normalized_rho[0]), dtype=float
        )
        connector_color: tuple[float, float, float, float] = (
            float(connector_rgba[0]),
            float(connector_rgba[1]),
            float(connector_rgba[2]),
            float(connector_rgba[3]),
        )
        connector = Line2D(
            [float(start_figure[0]), end_figure[0]],
            [float(start_figure[1]), end_figure[1]],
            transform=figure.transFigure,
            color=connector_color,
            linewidth=1.5,
            alpha=0.9,
            solid_capstyle="round",
            zorder=6,
        )
        figure.add_artist(connector)

    colorbar_label_bounds = association_colorbar.ax.xaxis.label.get_window_extent()
    colorbar_label_center_display: tuple[float, float] = (
        float(colorbar_label_bounds.x0 + colorbar_label_bounds.x1) / 2.0,
        float(colorbar_label_bounds.y0 + colorbar_label_bounds.y1) / 2.0,
    )
    colorbar_label_center_figure: np.ndarray = figure.transFigure.inverted().transform(
        colorbar_label_center_display
    )
    figure.legend(
        handles=_relationship_legend_handles(),
        loc="center right",
        bbox_to_anchor=(0.985, float(colorbar_label_center_figure[1])),
        bbox_transform=figure.transFigure,
        ncols=3,
        frameon=False,
        fontsize=6.7,
        columnspacing=0.8,
        handlelength=1.9,
    )
    if export:
        _save_figure(
            figure,
            output_folder,
            f"discharge_kge_characteristic_heatmaps{output_name_suffix}",
            logger,
        )
    return figure


def _alphabetic_panel_label(panel_index: int) -> str:
    """Convert a zero-based panel index to an alphabetic label.

    Args:
        panel_index: Zero-based non-negative panel index.

    Returns:
        Label such as ``a)`` or ``aa)``.

    Raises:
        ValueError: If ``panel_index`` is negative.
    """
    if panel_index < 0:
        raise ValueError("Panel index cannot be negative.")
    label: str = ""
    remaining_index: int = panel_index
    while True:
        remaining_index, remainder = divmod(remaining_index, 26)
        label = chr(ord("a") + remainder) + label
        if remaining_index == 0:
            break
        remaining_index -= 1
    return f"{label})"


def plot_all_kge_characteristic_scatterplots(
    analysis_df: pd.DataFrame,
    association_df: pd.DataFrame,
    output_folder: Path,
    logger: logging.Logger,
    output_name_suffix: str = "",
    export: bool = True,
) -> plt.Figure:
    """Plot station-level KGE relationships for all 32 characteristics.

    Panels are ranked by absolute Spearman association with overall KGE. The
    x-axis is logarithmic only for catchment area; every other characteristic
    remains in its original displayed units.

    Args:
        analysis_df: Prepared matched-station analysis table.
        association_df: KGE-component association table.
        output_folder: Folder receiving SVG, PDF, and PNG outputs.
        logger: Logger used for export messages.
        output_name_suffix: Optional evaluation-period filename suffix.
        export: Whether to save the figure.

    Returns:
        Four-column atlas containing all 32 continuous KGE relationships.

    Raises:
        ValueError: If an overall-KGE association is missing.
    """
    overall_associations: pd.DataFrame = association_df.loc[
        association_df["target"] == "KGE_daily"
    ].set_index("variable")
    required_variables: set[str] = {
        characteristic.column for characteristic in SCREENING_CHARACTERISTICS
    }
    if not required_variables.issubset(overall_associations.index):
        raise ValueError("Overall-KGE associations are incomplete.")
    ordered_characteristics: list[Characteristic] = sorted(
        SCREENING_CHARACTERISTICS,
        key=lambda item: abs(
            float(overall_associations.loc[item.column, "spearman_rho"])
        ),
        reverse=True,
    )

    column_count: int = 4
    row_count: int = int(np.ceil(len(ordered_characteristics) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(13.6, 2.25 * row_count),
        sharey=True,
    )
    for panel_index, (axis, characteristic) in enumerate(
        zip(axes.flat, ordered_characteristics, strict=True)
    ):
        displayed_x, kge_values, model_x, evaluation_x = _relationship_data(
            analysis_df, characteristic
        )
        central_curve: np.ndarray = _fit_lowess_to_grid(
            model_x, kge_values, evaluation_x, robust_iterations=2
        )
        displayed_evaluation_x: np.ndarray = (
            np.power(10.0, evaluation_x)
            if characteristic.logarithmic_x
            else evaluation_x
        )
        rho: float = float(
            overall_associations.loc[characteristic.column, "spearman_rho"]
        )
        axis.scatter(
            displayed_x,
            kge_values,
            s=7,
            color="#315F70",
            alpha=0.20,
            linewidths=0.0,
            rasterized=True,
            zorder=2,
        )
        axis.plot(
            displayed_evaluation_x,
            central_curve,
            color="#0C526C",
            linewidth=1.8,
            zorder=4,
        )
        axis.axhline(
            -0.41,
            color="#C44E52",
            linewidth=0.9,
            linestyle=(0, (3.0, 2.2)),
            zorder=1,
        )
        if characteristic.logarithmic_x:
            axis.set_xscale("log")
        axis.set_xlim(displayed_evaluation_x[0], displayed_evaluation_x[-1])
        axis.set_ylim(-1.0, 1.0)
        axis.set_title(
            f"{_alphabetic_panel_label(panel_index)} {characteristic.label}\n"
            f"Spearman ρ = {rho:+.2f}",
            loc="left",
            fontsize=8.2,
            fontweight="bold",
            pad=4,
        )
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", labelsize=6.8)
        axis.spines[["top", "right"]].set_visible(False)

    for axis in axes[:, 0]:
        axis.set_ylabel("KGE (–)", fontsize=7.6)
    atlas_legend_handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#315F70",
            markeredgecolor="none",
            markersize=4.0,
            alpha=0.5,
            label="Station",
        ),
        Line2D(
            [0],
            [0],
            color="#0C526C",
            linewidth=1.8,
            label="Robust LOWESS",
        ),
        Line2D(
            [0],
            [0],
            color="#C44E52",
            linewidth=0.9,
            linestyle=(0, (3.0, 2.2)),
            label="Mean-flow benchmark (KGE = −0.41)",
        ),
    ]
    figure.legend(
        handles=atlas_legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncols=3,
        frameon=False,
        fontsize=7.2,
    )
    figure.subplots_adjust(
        left=0.06,
        right=0.99,
        top=0.985,
        bottom=0.045,
        wspace=0.22,
        hspace=0.62,
    )
    if export:
        _save_figure(
            figure,
            output_folder,
            f"discharge_kge_all_characteristic_scatterplots{output_name_suffix}",
            logger,
        )
    return figure
