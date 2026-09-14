"""Relate discharge KGE and its components to GRDC-Caravan attributes.

``plot_discharge_characteristics`` runs the complete analysis: join attributes,
convert units, calculate associations, and draw/export the three figures.
Both relationship figures use the same panel renderer and LOWESS fit; only the
four summary panels add bootstrap intervals. Dashboard preparation retains
unmatched stations and is deliberately separate from this matched-only analysis.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


# These 32 attributes cover climate, topography, land cover, soils, hydrology,
# and human influence without pre-selecting variables by model performance.
SCREENING_CHARACTERISTICS: tuple[Characteristic, ...] = (
    Characteristic(
        "upstream_area_GEB",
        "Upstream catchment area (km²)",
        scale=1e-6,
        logarithmic_x=True,
    ),
    Characteristic("aridity_FAO_PM", "Aridity index, PET/P (–)"),
    Characteristic("ele_mt_sav", "Elevation (m)"),
    Characteristic("gwt_cm_sav", "Groundwater-table depth (cm)"),
    Characteristic("lka_pc_sse", "Lake-area extent (%)"),
    Characteristic("rev_mc_usu", "Upstream reservoir volume (million m³)"),
    Characteristic("frac_snow", "Fraction of precipitation falling as snow (–)"),
    Characteristic("p_mean", "Daily precipitation (mm/day)"),
    Characteristic("inu_pc_slt", "Long-term maximum inundation extent (%)"),
    Characteristic("tmp_dc_syr", "Annual air temperature (°C)", scale=0.1),
    Characteristic("seasonality_FAO_PM", "Moisture-index seasonality (–)"),
    Characteristic(
        "high_prec_freq", "High-precipitation-day frequency (%)", scale=100.0
    ),
    Characteristic("high_prec_dur", "High-precipitation-event duration (days)"),
    Characteristic("low_prec_freq", "Low-precipitation-day frequency (%)", scale=100.0),
    Characteristic("low_prec_dur", "Low-precipitation-event duration (days)"),
    Characteristic("slp_dg_sav", "Terrain slope (degrees)", scale=0.1),
    Characteristic("sgr_dk_sav", "Stream gradient (dm/km)"),
    Characteristic("wet_pc_sg1", "Wetland extent (%)"),
    Characteristic("dor_pc_pva", "Degree of regulation (%)"),
    Characteristic("for_pc_sse", "Forest-cover extent (%)"),
    Characteristic("crp_pc_sse", "Cropland extent (%)"),
    Characteristic("pst_pc_sse", "Pasture extent (%)"),
    Characteristic("ire_pc_sse", "Irrigated-area extent (%)"),
    Characteristic("urb_pc_sse", "Urban extent (%)"),
    Characteristic("gla_pc_sse", "Glacier extent (%)"),
    Characteristic("kar_pc_sse", "Karst-area extent (%)"),
    Characteristic("cly_pc_sav", "Soil clay fraction (%)"),
    Characteristic("snd_pc_sav", "Soil sand fraction (%)"),
    Characteristic("swc_pc_syr", "Annual soil-water content (%)"),
    Characteristic("hft_ix_s09", "Human-footprint index, 2009 (–)", scale=0.01),
    Characteristic("ppd_pk_sav", "Population density (people/km²)"),
    Characteristic("rdd_mk_sav", "Road density (m/km²)"),
)

_SCREENING_CHARACTERISTICS_BY_COLUMN: dict[str, Characteristic] = {
    characteristic.column: characteristic
    for characteristic in SCREENING_CHARACTERISTICS
}


# The dashboard subset prioritizes distinct, actionable hydrological mechanisms.
# Keeping this list short makes spatial comparison substantially easier than a
# layer menu containing every variable in the exploratory 32-panel atlas.
DASHBOARD_CHARACTERISTICS: tuple[Characteristic, ...] = (
    tuple(
        _SCREENING_CHARACTERISTICS_BY_COLUMN[column]
        for column in ("sgr_dk_sav", "ele_mt_sav")
    )
    + (
        Characteristic(
            "area",
            "GRDC-Caravan catchment area (km²)",
            logarithmic_x=True,
        ),
    )
    + tuple(
        _SCREENING_CHARACTERISTICS_BY_COLUMN[column]
        for column in (
            "gwt_cm_sav",
            "low_prec_freq",
            "frac_snow",
            "aridity_FAO_PM",
            "dor_pc_pva",
            "lka_pc_sse",
        )
    )
)

KGE_COMPONENT_TARGETS: tuple[KGEComponentTarget, ...] = (
    KGEComponentTarget("KGE_correlation_daily", "ρ (correlation)"),
    KGEComponentTarget("KGE_bias_ratio_daily", "ρ (bias ratio)"),
    KGEComponentTarget("KGE_variability_ratio_daily", "ρ (variability ratio)"),
    KGEComponentTarget("KGE_daily", "ρ (KGE)"),
)

# The selected panels represent distinct topographic, subsurface, channel, and
# climate relationships while avoiding redundant variants of the same signal.
KGE_RELATIONSHIP_PANELS: dict[str, str] = {
    "ele_mt_sav": "b) Elevation",
    "gwt_cm_sav": "c) Groundwater-table depth",
    "sgr_dk_sav": "d) Stream gradient",
    "low_prec_freq": "e) Low-precipitation-day frequency",
}

# These tones are sampled from the positive half of ColorBrewer's BrBG scale
# so the relationship panels and signed-correlation heatmap read as one design.
STATION_COLOR: str = "#35978F"
LOWESS_COLOR: str = "#01665E"
LOWESS_INTERVAL_COLOR: str = "#80CDC1"


def plot_discharge_characteristics(
    station_scores: pd.DataFrame,
    catchment_attributes: pd.DataFrame,
    output_folder: Path,
    logger: logging.Logger,
    output_name_suffix: str = "",
    export: bool = True,
) -> None:
    """Join, analyze, and plot discharge scores against catchment attributes.

    Args:
        station_scores: Filtered station scores, with upstream area in m².
        catchment_attributes: GRDC-Caravan attributes keyed by gauge_id.
        output_folder: Directory receiving the association CSV and three figures.
        logger: Logger for matching and export messages.
        output_name_suffix: Evaluation-period suffix for output filenames.
        export: Whether to write files; figures are always closed after drawing.

    Returns:
        None. Stations without attribute matches are logged and skipped.

    Raises:
        ValueError: If required score or attribute columns are missing, or
            catchment gauge identifiers are duplicated.
    """  # noqa: DOC202, DOC502
    scores_with_characteristics: pd.DataFrame = enrich_discharge_evaluation(
        station_scores=station_scores,
        catchment_attributes=catchment_attributes,
    )
    if export:
        output_folder.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Matched %d/%d evaluated stations to GRDC-Caravan attributes.",
        int(scores_with_characteristics["grdc_caravan_matched"].sum()),
        len(scores_with_characteristics),
    )

    if not scores_with_characteristics["grdc_caravan_matched"].any():
        logger.warning(
            "No discharge evaluation stations match GRDC-Caravan after "
            "upstream-area filtering. Skipping discharge characteristic plots."
        )
        return

    characteristic_analysis: pd.DataFrame = prepare_kge_characteristic_analysis(
        scores_with_characteristics
    )
    characteristic_associations: pd.DataFrame = calculate_kge_component_associations(
        characteristic_analysis
    )
    if export:
        association_path: Path = output_folder / (
            f"discharge_kge_component_associations{output_name_suffix}.csv"
        )
        characteristic_associations.to_csv(association_path, index=False)
        logger.info(
            "Saved discharge characteristic associations to %s.", association_path
        )

    correlation_figure: plt.Figure = create_characteristic_correlation_matrix(
        characteristic_analysis=characteristic_analysis,
        output_folder=output_folder,
        logger=logger,
        output_name_suffix=output_name_suffix,
        export=export,
    )
    plt.close(correlation_figure)

    for create_figure in (
        create_kge_characteristic_summary,
        create_kge_characteristic_scatterplots,
    ):
        figure: plt.Figure = create_figure(
            characteristic_analysis,
            characteristic_associations,
            output_folder,
            logger,
            output_name_suffix,
            export,
        )
        plt.close(figure)


def enrich_discharge_evaluation(
    station_scores: pd.DataFrame,
    catchment_attributes: pd.DataFrame,
) -> pd.DataFrame:
    """Add GRDC-Caravan catchment attributes to GEB discharge scores.

    The left join retains evaluated stations absent from the openly licensed
    GRDC-Caravan subset.

    Args:
        station_scores: Per-station GEB discharge metrics.
        catchment_attributes: GRDC-Caravan attributes keyed by ``gauge_id``.

    Returns:
        Evaluation table with catchment attributes and a match indicator.

    Raises:
        ValueError: If either station identifier is missing or duplicated.
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
        format_grdc_station_key
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
    """Prepare matched stations for KGE-characteristic analysis.

    Characteristics are converted once to their displayed units. KGE and its
    three original components remain untransformed so that bias and variability
    direction are preserved.

    Args:
        station_scores: Discharge metrics enriched with GRDC-Caravan attributes.

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
        *(target.column for target in KGE_COMPONENT_TARGETS),
        *(item.column for item in SCREENING_CHARACTERISTICS),
    )
    for column in numeric_columns:
        analysis_table[column] = pd.to_numeric(analysis_table[column], errors="coerce")
    for characteristic in SCREENING_CHARACTERISTICS:
        analysis_table[characteristic.column] *= characteristic.scale
    return analysis_table


def prepare_dashboard_characteristics(
    scores_with_characteristics: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the curated GRDC-Caravan attributes for the spatial dashboard.

    Unlike the publication-figure preparation, this function retains stations
    without a GRDC-Caravan match so they can be shown as missing on the map.
    Values are converted to the human-readable units defined by
    :data:`DASHBOARD_CHARACTERISTICS`.

    Args:
        scores_with_characteristics: Discharge evaluation enriched with
            GRDC-Caravan attributes and a ``grdc_caravan_matched`` indicator.

    Returns:
        Evaluation table with dashboard characteristics in display units.

    Raises:
        ValueError: If station identifiers, match status, or selected
            characteristic columns are unavailable.
    """
    required_columns: set[str] = {
        "station_ID",
        "grdc_caravan_matched",
        *(item.column for item in DASHBOARD_CHARACTERISTICS),
    }
    missing_columns: set[str] = required_columns - set(
        scores_with_characteristics.columns
    )
    if missing_columns:
        raise ValueError(
            "Enriched discharge metrics are missing dashboard columns: "
            f"{sorted(missing_columns)}"
        )
    if scores_with_characteristics["station_ID"].duplicated().any():
        raise ValueError("Dashboard discharge metrics contain duplicate station IDs.")

    dashboard_table: pd.DataFrame = scores_with_characteristics.copy()
    matched_stations: pd.Series = (
        dashboard_table["grdc_caravan_matched"].fillna(False).astype(bool)
    )
    for characteristic in DASHBOARD_CHARACTERISTICS:
        numeric_values: pd.Series = pd.to_numeric(
            dashboard_table[characteristic.column], errors="coerce"
        )
        # Mask every selected field explicitly so unmatched rows can never be
        # mistaken for valid zero-valued GRDC-Caravan observations.
        dashboard_table[characteristic.column] = (
            numeric_values.where(matched_stations) * characteristic.scale
        )
    return dashboard_table


def calculate_kge_component_associations(
    characteristic_analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate Spearman associations with KGE and its three components.

    Args:
        characteristic_analysis: Output from :func:`prepare_kge_characteristic_analysis`.

    Returns:
        Long table containing sample sizes, Spearman correlations, and p-values.
    """
    association_rows: list[dict[str, float | int | str]] = []
    for characteristic in SCREENING_CHARACTERISTICS:
        for target in KGE_COMPONENT_TARGETS:
            pair_table: pd.DataFrame = characteristic_analysis[
                [characteristic.column, target.column]
            ].dropna()
            rho: float = np.nan
            p_value: float = np.nan
            if (
                len(pair_table) >= 3
                and pair_table[characteristic.column].nunique() >= 2
                and pair_table[target.column].nunique() >= 2
            ):
                correlation_result: Any = spearmanr(
                    pair_table[characteristic.column], pair_table[target.column]
                )
                rho = float(correlation_result.statistic)
                p_value = float(correlation_result.pvalue)
            association_rows.append(
                {
                    "variable": characteristic.column,
                    "characteristic": characteristic.label,
                    "target": target.column,
                    "target_label": target.label,
                    "n": len(pair_table),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )

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


def create_characteristic_correlation_matrix(
    characteristic_analysis: pd.DataFrame,
    output_folder: Path,
    logger: logging.Logger,
    output_name_suffix: str = "",
    export: bool = True,
) -> plt.Figure:
    """Plot correlations among GRDC-Caravan catchment characteristics.

    Args:
        characteristic_analysis: Prepared matched-station analysis table.
        output_folder: Folder receiving the PNG output.
        logger: Logger used for export messages.
        output_name_suffix: Optional evaluation-period filename suffix.
        export: Whether to save the figure.

    Returns:
        Lower-triangular characteristic correlation heatmap.
    """
    characteristic_columns: list[str] = [
        characteristic.column for characteristic in SCREENING_CHARACTERISTICS
    ]
    correlation_matrix: pd.DataFrame = characteristic_analysis[
        characteristic_columns
    ].corr(method="spearman", min_periods=3)
    characteristic_labels: list[str] = [
        characteristic.label for characteristic in SCREENING_CHARACTERISTICS
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
        cbar_kws={"label": "Spearman rank correlation, ρ", "shrink": 0.74},
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
    if export:
        output_path: Path = output_folder / (
            f"discharge_characteristic_correlation_matrix{output_name_suffix}.png"
        )
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved characteristic correlation matrix to %s.", output_path)
    return figure


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


def _plot_relationship_panel(
    axis: plt.Axes,
    characteristic_analysis: pd.DataFrame,
    characteristic: Characteristic,
    rho: float,
    kge_axis_limits: tuple[float, float],
    random_generator: np.random.Generator | None = None,
) -> bool:
    """Draw station values and LOWESS, optionally with a bootstrap interval.

    The curve spans the 1st–99th x percentiles. A supplied generator enables
    the summary's 150 station-bootstrap samples; the atlas fits only one curve.

    Args:
        axis: Axis receiving the relationship.
        characteristic_analysis: Matched station values in display units.
        characteristic: Column metadata, display units, and x transformation.
        rho: Dimensionless Spearman correlation, or NaN for unavailable pairs.
        kge_axis_limits: Shared dimensionless KGE limits.
        random_generator: Reproducible resampling generator for summary panels.

    Returns:
        Whether enough valid station pairs were available to draw the panel.

    Raises:
        KeyError: If the characteristic or daily KGE column is missing.
    """  # noqa: DOC502
    pair_table: pd.DataFrame = characteristic_analysis[
        [characteristic.column, "KGE_daily"]
    ].dropna()
    if characteristic.logarithmic_x:
        pair_table = pair_table.loc[pair_table[characteristic.column] > 0.0]
    if np.isnan(rho) or len(pair_table) < 10:
        axis.set_axis_off()
        return False

    displayed_x: np.ndarray = pair_table[characteristic.column].to_numpy(dtype=float)
    kge_values: np.ndarray = pair_table["KGE_daily"].to_numpy(dtype=float)
    model_x: np.ndarray = (
        np.log10(displayed_x) if characteristic.logarithmic_x else displayed_x
    )
    lower_x: float
    upper_x: float
    lower_x, upper_x = np.quantile(model_x, [0.01, 0.99])
    if not upper_x > lower_x:
        axis.set_axis_off()
        return False
    evaluation_x: np.ndarray = np.linspace(lower_x, upper_x, 180)
    central_curve: np.ndarray = _fit_lowess_to_grid(
        model_x, kge_values, evaluation_x, robust_iterations=2
    )
    displayed_evaluation_x: np.ndarray = (
        np.power(10.0, evaluation_x) if characteristic.logarithmic_x else evaluation_x
    )
    summary: bool = random_generator is not None
    if random_generator is not None:
        bootstrap_curves: np.ndarray = np.empty((150, len(evaluation_x)), dtype=float)
        for repetition in range(150):
            sample_indices: np.ndarray = random_generator.integers(
                0, len(model_x), size=len(model_x)
            )
            if np.unique(model_x[sample_indices]).size < 2:
                # Sparse attributes can yield a bootstrap sample with one value.
                bootstrap_curves[repetition] = kge_values[sample_indices].mean()
            else:
                bootstrap_curves[repetition] = _fit_lowess_to_grid(
                    model_x[sample_indices],
                    kge_values[sample_indices],
                    evaluation_x,
                    robust_iterations=1,
                )
        lower_limit: np.ndarray
        upper_limit: np.ndarray
        lower_limit, upper_limit = np.quantile(bootstrap_curves, [0.025, 0.975], axis=0)
        axis.fill_between(
            displayed_evaluation_x,
            lower_limit,
            upper_limit,
            color=LOWESS_INTERVAL_COLOR,
            alpha=0.20,
            linewidth=0.0,
            zorder=1,
        )
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
                "alpha": 0.60,
            },
        )
        axis.set_xlabel(characteristic.label, fontsize=8.5)
        axis.set_ylabel("KGE (–)", fontsize=8.5)

    axis.scatter(
        displayed_x,
        kge_values,
        s=9 if summary else 7,
        color=STATION_COLOR if summary else "#315F70",
        alpha=0.27 if summary else 0.20,
        linewidths=0.0,
        rasterized=True,
        zorder=2,
    )
    axis.plot(
        displayed_evaluation_x,
        central_curve,
        color=LOWESS_COLOR if summary else "#0C526C",
        linewidth=2.3 if summary else 1.8,
        zorder=4,
    )
    if characteristic.logarithmic_x:
        axis.set_xscale("log")
    axis.set_xlim(displayed_evaluation_x[0], displayed_evaluation_x[-1])
    axis.set_ylim(kge_axis_limits)
    axis.grid(
        axis="y",
        color="#D9D9D9" if summary else "#DDDDDD",
        linewidth=0.6 if summary else 0.55,
    )
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=7.4 if summary else 6.7)
    if summary:
        axis.tick_params(axis="x", pad=3)
    axis.spines[["top", "right"]].set_visible(False)
    return True


def create_kge_characteristic_summary(
    characteristic_analysis: pd.DataFrame,
    characteristic_associations: pd.DataFrame,
    output_folder: Path,
    logger: logging.Logger,
    output_name_suffix: str = "",
    export: bool = True,
) -> plt.Figure:
    """Plot the KGE-component heatmap and four linked relationships.

    Args:
        characteristic_analysis: Prepared matched-station analysis table.
        characteristic_associations: KGE-component association table.
        output_folder: Folder receiving SVG, PDF, and PNG outputs.
        logger: Logger used for export messages.
        output_name_suffix: Optional evaluation-period filename suffix.
        export: Whether to save the figure.

    Returns:
        Figure containing the heatmap and four continuous relationship panels.

    Raises:
        ValueError: If selected relationship-panel inputs are incomplete.
    """
    association_matrix: pd.DataFrame = characteristic_associations.pivot(
        index="variable", columns="target", values="spearman_rho"
    )
    significance_matrix: pd.DataFrame = characteristic_associations.pivot(
        index="variable", columns="target", values="p_value"
    )
    labels_by_column: dict[str, str] = {
        item.column: item.label for item in SCREENING_CHARACTERISTICS
    }
    target_columns: list[str] = [target.column for target in KGE_COMPONENT_TARGETS]
    required_variables: list[str] = list(_SCREENING_CHARACTERISTICS_BY_COLUMN)
    selected_variables: set[str] = set(KGE_RELATIONSHIP_PANELS)
    if not {"KGE_daily", *selected_variables}.issubset(characteristic_analysis.columns):
        raise ValueError("KGE relationship-panel inputs are incomplete.")

    association_matrix = association_matrix.reindex(
        index=required_variables, columns=target_columns
    )
    significance_matrix = significance_matrix.reindex(
        index=required_variables, columns=target_columns
    )
    variable_order: list[str] = (
        association_matrix["KGE_daily"]
        .abs()
        .sort_values(ascending=False, kind="stable", na_position="last")
        .index.tolist()
    )
    association_matrix = association_matrix.loc[variable_order]
    significance_matrix = significance_matrix.loc[variable_order]

    figure: plt.Figure = plt.figure(figsize=(14.8, 11.8))
    outer_grid: GridSpec = figure.add_gridspec(
        1, 2, width_ratios=(1.12, 1.10), wspace=0.30
    )
    association_axis: plt.Axes = figure.add_subplot(outer_grid[0, 0])
    correlation_grid: GridSpecFromSubplotSpec = outer_grid[0, 1].subgridspec(
        5,
        1,
        height_ratios=(1.0, 1.0, 1.0, 1.0, 0.005),
        hspace=0.52,
    )
    correlation_axes: list[plt.Axes] = [
        figure.add_subplot(correlation_grid[row_index, 0])
        for row_index in range(len(KGE_RELATIONSHIP_PANELS))
    ]

    association_image: AxesImage = association_axis.imshow(
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
            if np.isnan(rho):
                continue
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
    association_axis.set_yticks(np.arange(-0.5, len(variable_order), 1.0), minor=True)
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
        "Spearman rank correlation, ρ  (* p-value < 0.05)", fontsize=8.5
    )

    random_generator: np.random.Generator = np.random.default_rng(42)
    lower_kge, upper_kge = characteristic_analysis["KGE_daily"].quantile([0.10, 0.90])
    kge_axis_limits: tuple[float, float] = (float(lower_kge), float(upper_kge))
    for correlation_axis, column in zip(
        correlation_axes, KGE_RELATIONSHIP_PANELS, strict=True
    ):
        characteristic: Characteristic = _SCREENING_CHARACTERISTICS_BY_COLUMN[column]
        _plot_relationship_panel(
            axis=correlation_axis,
            characteristic_analysis=characteristic_analysis,
            characteristic=characteristic,
            rho=float(association_matrix.loc[column, "KGE_daily"]),
            random_generator=random_generator,
            kge_axis_limits=kge_axis_limits,
        )
        if correlation_axis.axison:
            correlation_axis.set_title(
                KGE_RELATIONSHIP_PANELS[column],
                loc="left",
                fontsize=10.0,
                fontweight="bold",
                pad=7,
            )
        if pd.notna(association_matrix.loc[column, "KGE_daily"]):
            characteristic_row: int = variable_order.index(column)
            association_axis.add_patch(
                Rectangle(
                    (
                        target_columns.index("KGE_daily") - 0.5,
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
    for correlation_axis, column in zip(
        correlation_axes, KGE_RELATIONSHIP_PANELS, strict=True
    ):
        characteristic_row = variable_order.index(column)
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
        overall_kge_rho: float = float(association_matrix.loc[column, "KGE_daily"])
        if np.isnan(overall_kge_rho):
            continue
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
        connector: Line2D = Line2D(
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
    if export:
        _save_figure(
            figure,
            output_folder,
            f"discharge_kge_characteristic_heatmaps{output_name_suffix}",
            logger,
        )
    return figure


def create_kge_characteristic_scatterplots(
    characteristic_analysis: pd.DataFrame,
    characteristic_associations: pd.DataFrame,
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
        characteristic_analysis: Prepared matched-station analysis table.
        characteristic_associations: KGE-component association table.
        output_folder: Folder receiving SVG, PDF, and PNG outputs.
        logger: Logger used for export messages.
        output_name_suffix: Optional evaluation-period filename suffix.
        export: Whether to save the figure.

    Returns:
        Four-column atlas containing all 32 continuous KGE relationships.
    """
    overall_associations: pd.DataFrame = characteristic_associations.loc[
        characteristic_associations["target"] == "KGE_daily"
    ].set_index("variable")
    required_variables: list[str] = [
        characteristic.column for characteristic in SCREENING_CHARACTERISTICS
    ]
    overall_associations = overall_associations.reindex(required_variables)
    variable_order: list[str] = (
        overall_associations["spearman_rho"]
        .abs()
        .sort_values(ascending=False, kind="stable", na_position="last")
        .index.tolist()
    )
    ordered_characteristics: list[Characteristic] = [
        _SCREENING_CHARACTERISTICS_BY_COLUMN[column] for column in variable_order
    ]

    column_count: int = 4
    row_count: int = int(np.ceil(len(ordered_characteristics) / column_count))
    kge_lower_limit, kge_upper_limit = characteristic_analysis["KGE_daily"].quantile(
        [0.10, 0.90]
    )
    kge_axis_limits: tuple[float, float] = (
        float(kge_lower_limit),
        float(kge_upper_limit),
    )
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(13.6, 1.68 * row_count),
        sharey=True,
    )
    for panel_index, (axis, characteristic) in enumerate(
        zip(axes.flat, ordered_characteristics, strict=True)
    ):
        rho: float = float(
            overall_associations.loc[characteristic.column, "spearman_rho"]
        )
        if not _plot_relationship_panel(
            axis, characteristic_analysis, characteristic, rho, kge_axis_limits
        ):
            continue
        # Spreadsheet-style labels continue with aa after z.
        panel_number: int = panel_index + 1
        panel_label: str = ""
        while panel_number:
            panel_number, remainder = divmod(panel_number - 1, 26)
            panel_label = chr(ord("a") + remainder) + panel_label
        axis.set_title(
            f"{panel_label}) {characteristic.label}\nSpearman ρ = {rho:+.2f}",
            loc="left",
            fontsize=8.0,
            fontweight="bold",
            pad=4,
        )

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
        handles=atlas_legend_handles,
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
    if export:
        _save_figure(
            figure,
            output_folder,
            f"discharge_kge_all_characteristic_scatterplots{output_name_suffix}",
            logger,
        )
    return figure
