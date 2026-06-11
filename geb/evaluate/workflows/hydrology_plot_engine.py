"""Reusable plot helpers for hydrology evaluation outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import contextily as ctx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_DISPLAYED_SKILL_SCORE_CONFIGS: tuple[dict[str, object], ...] = (
    {
        "col": "KGE",
        "label": "KGE",
        "title": "Kling-Gupta Efficiency (KGE)",
        "unit": "(−)",
        "ylim": (-1.0, 1.0),
        "reference": 1.0,
        "cmap": "RdYlGn",
        "vmin": -1.0,
        "vmax": 1.0,
        "color": "#1f77b4",
    },
    {
        "col": "KGE_correlation",
        "label": "KGE r",
        "title": "KGE correlation component",
        "compact_title": "Correlation",
        "unit": "(−)",
        "ylim": (-1.0, 1.0),
        "reference": 1.0,
        "cmap": "RdYlGn",
        "vmin": -1.0,
        "vmax": 1.0,
        "color": "#17becf",
    },
    {
        "col": "KGE_bias_ratio",
        "label": "β",
        "title": "KGE bias-ratio component",
        "compact_title": "Bias ratio",
        "unit": "(−)",
        "ylim": (0.0, 2.0),
        "reference": 1.0,
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 2.0,
        "color": "#ff7f0e",
    },
    {
        "col": "KGE_variability_ratio",
        "label": "α",
        "title": "KGE variability-ratio component",
        "compact_title": "Variability ratio",
        "unit": "(−)",
        "ylim": (0.0, 2.0),
        "reference": 1.0,
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 2.0,
        "color": "#8B8C00",
    },
    {
        "col": "NSE",
        "label": "NSE",
        "title": "Nash-Sutcliffe Efficiency (NSE)",
        "unit": "(−)",
        "ylim": (-1.0, 1.0),
        "reference": 1.0,
        "cmap": "RdYlGn",
        "vmin": -1.0,
        "vmax": 1.0,
        "color": "#2ca02c",
    },
    {
        "col": "R2",
        "label": "R²",
        "title": "Coefficient of Determination (R²)",
        "unit": "(−)",
        "ylim": (-1.0, 1.0),
        "reference": 1.0,
        "cmap": "YlGn",
        "vmin": 0.0,
        "vmax": 1.0,
        "color": "#9467bd",
    },
    {
        "col": "RRMSE",
        "label": "RRMSE",
        "title": "Relative Root Mean Squared Error (RRMSE)",
        "unit": "(−)",
        "ylim": None,
        "robust_error_ylim": True,
        "reference": 0.0,
        "cmap": "YlOrRd_r",
        "vmin": 0.0,
        "vmax": None,
        "color": "#d62728",
    },
)


def _plot_skill_score_map_single(
    evaluation_gdf: gpd.GeoDataFrame,
    metric_col: str,
    metric_label: str,
    metric_title: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
    output_path: Path,
    region_geom: gpd.GeoDataFrame,
) -> None:
    """Plot gauging stations coloured by a single skill score on a satellite basemap.

    Args:
        evaluation_gdf: Per-station metrics with point geometry in any CRS.
        metric_col: Column name of the metric to plot (e.g. ``"KGE"``).
        metric_label: Short colorbar label (e.g. ``"KGE"``).
        metric_title: Figure title.
        cmap_name: Matplotlib colormap name.
        vmin: Colorbar minimum value.
        vmax: Colorbar maximum value.
        output_path: Output path stem (no extension); ``.svg`` and ``.png`` are saved.
        region_geom: Basin/region boundary overlaid on the map.
    """
    # Reproject to Web Mercator for the contextily basemap
    gdf_3857: gpd.GeoDataFrame = evaluation_gdf.to_crs("EPSG:3857")
    region_3857: gpd.GeoDataFrame = region_geom.to_crs("EPSG:3857")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 9))

    region_3857.plot(
        ax=ax,
        color="none",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.7,
        zorder=2,
    )

    # Valid stations coloured by metric value; NaN stations shown as grey crosses
    valid_mask: pd.Series = gdf_3857[metric_col].notna()
    if valid_mask.any():
        ax.scatter(
            gdf_3857.loc[valid_mask, "geometry"].x,
            gdf_3857.loc[valid_mask, "geometry"].y,
            c=gdf_3857.loc[valid_mask, metric_col],
            cmap=cmap_name,
            norm=norm,
            s=12,
            zorder=4,
            linewidths=0.2,
            edgecolors="white",
        )
    if (~valid_mask).any():
        ax.scatter(
            gdf_3857.loc[~valid_mask, "geometry"].x,
            gdf_3857.loc[~valid_mask, "geometry"].y,
            c="grey",
            marker="x",
            s=10,
            zorder=3,
            linewidths=0.5,
            label="No data",
        )
        ax.legend(
            fontsize=8,
            loc="lower right",
            framealpha=0.8,
        )

    # Satellite basemap
    ctx.add_basemap(
        ax,
        crs="EPSG:3857",
        source=ctx.providers.Esri.WorldImagery,  # ty:ignore[unresolved-attribute]
        attribution=False,
        zoom="auto",
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, aspect=30)
    cbar.set_label(metric_label, fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="black", labelcolor="black")
    cbar.outline.set_edgecolor("0.3")  # ty:ignore[call-non-callable]

    # Scale bar: round ~15% of the map width to a nice number (e.g. 152 km → 200 km)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    map_width_m: float = x_max - x_min
    map_height_m: float = y_max - y_min
    bar_m: float = round(
        map_width_m * 0.15 / 10 ** np.floor(np.log10(map_width_m * 0.15))
    ) * 10 ** np.floor(np.log10(map_width_m * 0.15))
    bar_label: str = f"{int(bar_m / 1_000)} km" if bar_m >= 1_000 else f"{int(bar_m)} m"
    bar_x0: float = x_min + map_width_m * 0.03
    bar_y: float = y_min + map_height_m * 0.03
    ax.plot(
        [bar_x0, bar_x0 + bar_m],
        [bar_y, bar_y],
        color="white",
        linewidth=3,
        solid_capstyle="butt",
        zorder=5,
    )
    ax.text(
        bar_x0 + bar_m / 2,
        bar_y + map_height_m * 0.012,
        bar_label,
        color="white",
        fontsize=8,
        ha="center",
        va="bottom",
        zorder=5,
    )

    # North arrow
    ax.annotate(
        "N",
        xy=(0.96, 0.12),
        xytext=(0.96, 0.06),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=10,
        color="white",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5),
    )

    ax.set_title(metric_title, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("0.3")

    plt.tight_layout()
    for ext in ("svg", "png"):
        plt.savefig(f"{output_path}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_skill_score_maps(
    evaluation_gdf: gpd.GeoDataFrame,
    region_geom: gpd.GeoDataFrame,
    output_folder: Path,
    logger: logging.Logger,
) -> None:
    """Plot per-station skill scores on a satellite basemap, one map per metric.

    Saves SVG and PNG files under ``output_folder/skill_score_maps/``.

    Args:
        evaluation_gdf: Per-station metrics with point geometry in any CRS.
        region_geom: Basin/region boundary overlaid on each map.
        output_folder: Root folder under which ``skill_score_maps/`` is created.
        logger: Logger to use for progress messages.
    """
    maps_folder = output_folder / "skill_score_maps"
    maps_folder.mkdir(parents=True, exist_ok=True)

    for cfg in _DISPLAYED_SKILL_SCORE_CONFIGS:
        col: str = str(cfg["col"])
        if col not in evaluation_gdf.columns:
            logger.info("Metric '%s' not in evaluation data, skipping.", col)
            continue

        valid_values: np.ndarray = evaluation_gdf[col].dropna().to_numpy(dtype=float)
        if valid_values.size == 0:
            logger.info("No valid values for metric '%s', skipping.", col)
            continue

        vmax: float = (
            float(np.nanpercentile(valid_values, 95))
            if cfg["vmax"] is None
            else float(cfg["vmax"])
        )

        _plot_skill_score_map_single(
            evaluation_gdf=evaluation_gdf,
            metric_col=col,
            metric_label=f"{cfg['label']} {cfg['unit']}",
            metric_title=str(cfg["title"]),
            cmap_name=str(cfg["cmap"]),
            vmin=float(cfg["vmin"]),
            vmax=vmax,
            output_path=maps_folder / f"skill_score_map_{col.lower()}",
            region_geom=region_geom,
        )
        logger.info("Saved skill score map for %s.", col)

    logger.info("All skill score maps saved to: %s", maps_folder)


def _draw_violin_box(
    axis: plt.Axes,
    values: np.ndarray,
    position: float,
    bar_color: str,
    violin_width: float = 0.35,
) -> None:
    """Draw a violin and boxplot for one model metric distribution.

    Args:
        axis: Axis receiving the distribution.
        values: Metric values for one model.
        position: X-axis position of the distribution.
        bar_color: Color used for the violin and box.
        violin_width: Width of the violin plot.
    """
    if len(values) >= 3:
        parts = axis.violinplot(
            values,
            positions=[position],
            showmedians=False,
            showextrema=False,
            widths=violin_width,
        )
        for body in cast(list, parts["bodies"]):
            body.set_facecolor(bar_color)
            body.set_edgecolor(bar_color)
            body.set_alpha(0.45)

    axis.boxplot(
        values,
        positions=[position],
        widths=violin_width * 0.35,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2.0, "zorder": 5},
        boxprops={
            "facecolor": bar_color,
            "alpha": 0.9,
            "linewidth": 0.8,
            "edgecolor": bar_color,
        },
        whiskerprops={"color": bar_color, "linewidth": 1.0},
        capprops={"color": bar_color, "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": bar_color,
            "markeredgecolor": bar_color,
            "markersize": 3,
            "alpha": 0.5,
            "linewidth": 0,
        },
    )


def _get_robust_error_metric_ylim(metric_values: np.ndarray) -> tuple[float, float]:
    """Get readable y-axis limits for unbounded error-metric distributions.

    Extreme RMSE and RRMSE outliers can make the violin and box unreadable. The
    95th percentile keeps the main distribution visible while medians still use
    all values.

    Args:
        metric_values: Error metric values across all plotted models.

    Returns:
        Lower and upper y-axis limits for the error metric.
    """
    finite_values: np.ndarray = metric_values[np.isfinite(metric_values)]
    if finite_values.size == 0:
        return (0.0, 1.0)

    nonnegative_values: np.ndarray = finite_values[finite_values >= 0.0]
    if nonnegative_values.size == 0:
        return (0.0, 1.0)

    visible_upper_limit: float = float(np.nanpercentile(nonnegative_values, 95))
    return (0.0, max(visible_upper_limit * 1.1, 1.0))


def _get_skill_score_config(metric_col: str) -> dict[str, object]:
    """Get the display configuration for a skill-score metric.

    Args:
        metric_col: Name of the metric column in the evaluation data.

    Returns:
        Plot display configuration for the requested metric.

    Raises:
        ValueError: If the metric column is not configured for plotting.
    """
    for metric_config in _DISPLAYED_SKILL_SCORE_CONFIGS:
        if metric_config["col"] == metric_col:
            return metric_config
    raise ValueError(f"No skill-score plot configuration found for '{metric_col}'.")


def _get_boxplot_axis_layout(
    fig: plt.Figure,
) -> dict[str, plt.Axes]:
    """Create the grouped layout for discharge skill-score boxplots.

    The top row contains the headline skill scores at equal size. The lower-left
    row contains the three smaller KGE components so their relationship to KGE
    is visually explicit without competing with the primary metrics.

    Args:
        fig: Matplotlib figure that receives the axes.

    Returns:
        Metric-column-to-axis mapping.
    """
    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=4,
        height_ratios=[3.2, 1.6],
        hspace=0.72,
        wspace=0.30,
    )
    axes_by_metric: dict[str, plt.Axes] = {
        "KGE": fig.add_subplot(outer_grid[0, 0]),
        "NSE": fig.add_subplot(outer_grid[0, 1]),
        "R2": fig.add_subplot(outer_grid[0, 2]),
        "RRMSE": fig.add_subplot(outer_grid[0, 3]),
    }
    component_grid = outer_grid[1, 0].subgridspec(
        nrows=1,
        ncols=3,
        wspace=0.42,
    )
    axes_by_metric["KGE_correlation"] = fig.add_subplot(component_grid[0, 0])
    axes_by_metric["KGE_bias_ratio"] = fig.add_subplot(component_grid[0, 1])
    axes_by_metric["KGE_variability_ratio"] = fig.add_subplot(component_grid[0, 2])

    spacer_axes: list[plt.Axes] = [
        fig.add_subplot(outer_grid[1, column_index]) for column_index in range(1, 4)
    ]
    for spacer_axis in spacer_axes:
        spacer_axis.set_visible(False)

    return axes_by_metric


def _annotate_metric_sample_sizes(
    axis: plt.Axes,
    models_with_data: list[tuple[str, np.ndarray]],
    model_colors: dict[str, str],
    compact: bool,
) -> None:
    """Add median and station-count labels below a metric axis.

    Args:
        axis: Axis receiving the labels.
        models_with_data: Model names and finite metric values to summarize.
        model_colors: Mapping from model name to plot color.
        compact: Whether the labels are drawn in a smaller component axis.
    """
    label_blocks: list[str] = [
        f"{model_name}:\nmed={float(np.median(metric_values)):.2f}\nn={len(metric_values)}"
        for model_name, metric_values in models_with_data
    ]
    label_color: str = model_colors.get("GEB", "#1f77b4")
    axis.text(
        0.5,
        -0.20 if compact else -0.11,
        "\n\n".join(label_blocks),
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=6,
        linespacing=0.9,
        color=label_color,
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "0.95",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
        clip_on=False,
    )


def _add_filter_summary(
    fig: plt.Figure,
    filter_summary: str,
) -> None:
    """Add the applied filtering criteria to the lower-right plot area.

    Args:
        fig: Figure receiving the annotation.
        filter_summary: Multi-line description of station filters.
    """
    if not filter_summary:
        return
    fig.text(
        0.985,
        0.18,
        filter_summary,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "0.95",
            "edgecolor": "0.7",
            "alpha": 0.9,
        },
    )


def plot_skill_score_boxplots(
    evaluation_df: pd.DataFrame,
    external_models: dict[str, pd.DataFrame],
    output_folder: Path,
    logger: logging.Logger,
    export: bool = True,
    include_geb: bool = True,
    matched_only: bool = False,
    filter_summary: str = "",
    output_name_suffix: str = "",
) -> None:
    """Create skill score violin+boxplot graphs for each evaluation metric.

    Args:
        evaluation_df: GEB evaluation metrics.
        external_models: External model metrics keyed by model name.
        output_folder: Folder where output figures are saved.
        logger: Logger to use for progress messages.
        export: Save the figure to disk.
        include_geb: Include GEB in the plot.
        matched_only: Whether the plotted data were restricted to matched stations.
        filter_summary: Description of filters to annotate in the plot.
        output_name_suffix: Optional suffix appended to the output file stem.
    """
    if include_geb and evaluation_df.empty:
        logger.info(
            "No discharge stations found for evaluation. Skipping skill score graphs."
        )
        return
    if not include_geb and not external_models:
        logger.warning(
            "include_geb=False but no external model data found. "
            "Run prepare_external_evaluation first."
        )
        return

    metric_order: tuple[str, ...] = (
        "KGE",
        "NSE",
        "R2",
        "RRMSE",
        "KGE_correlation",
        "KGE_bias_ratio",
        "KGE_variability_ratio",
    )
    metric_configs: tuple[dict[str, object], ...] = tuple(
        _get_skill_score_config(metric_col) for metric_col in metric_order
    )

    geb_color: str = "#1f77b4"
    external_colors: dict[str, str] = dict(
        zip(
            external_models,
            ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"],
            strict=False,
        )
    )
    model_colors: dict[str, str] = (
        {"GEB": geb_color} if include_geb else {}
    ) | external_colors

    logger.info("Creating evaluation metrics skill score plots...")

    fig = plt.figure(figsize=(15.5, 6.8), constrained_layout=False)
    axes_by_metric = _get_boxplot_axis_layout(fig)
    plot_subtitle: str = (
        " (matched stations only)" if matched_only and external_models else ""
    )
    fig.suptitle(
        f"Discharge Evaluation — Skill Score Distributions{plot_subtitle}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for config in metric_configs:
        axis = axes_by_metric[str(config["col"])]
        metric_col: str = str(config["col"])
        geb_metric_values: np.ndarray = (
            evaluation_df[metric_col].dropna().to_numpy(dtype=float)
            if include_geb and metric_col in evaluation_df.columns
            else np.array([], dtype=float)
        )
        external_metric_values: dict[str, np.ndarray] = {
            model_name: values
            for model_name, model_df in external_models.items()
            if metric_col in model_df.columns
            for values in (model_df[metric_col].dropna().to_numpy(dtype=float),)
            if values.size > 0
        }

        if geb_metric_values.size == 0 and not external_metric_values:
            axis.set_visible(False)
            continue

        models_with_data: list[tuple[str, np.ndarray]] = []
        if geb_metric_values.size > 0:
            models_with_data.append(("GEB", geb_metric_values))
        models_with_data.extend(external_metric_values.items())

        x_positions = (
            np.linspace(-0.4, 0.4, len(models_with_data))
            if len(models_with_data) > 1
            else np.array([0.0])
        )

        for (model_name, metric_values), x_position in zip(
            models_with_data, x_positions, strict=True
        ):
            bar_color: str = model_colors[model_name]
            _draw_violin_box(axis, metric_values, x_position, bar_color)
        is_component_axis: bool = metric_col.startswith("KGE_")
        _annotate_metric_sample_sizes(
            axis,
            models_with_data,
            model_colors,
            compact=is_component_axis,
        )

        axis.axhline(
            float(config["reference"]),
            color="0.5",
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )
        title_text: str = str(
            config.get("compact_title", config["title"])
            if is_component_axis
            else config["title"]
        )
        axis.set_title(
            f"{config['label']}\n{title_text} {config['unit']}",
            fontsize=8 if is_component_axis else 9,
            fontweight="bold",
            pad=8,
        )
        if config["ylim"] is not None:
            axis.set_ylim(*cast(tuple[float, float], config["ylim"]))
        elif config.get("robust_error_ylim", False):
            combined_metric_values: np.ndarray = np.concatenate(
                [metric_values for _, metric_values in models_with_data]
            )
            axis.set_ylim(*_get_robust_error_metric_ylim(combined_metric_values))
        axis.set_xticks([])
        axis.tick_params(
            axis="y", labelsize=7 if is_component_axis else 8,
        )
        axis.set_xlabel("")
        for spine in axis.spines.values():
            spine.set_edgecolor("0.7")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.grid(axis="y", color="0.85", linewidth=0.5)

    legend_handles: list[Line2D] = [
        Line2D([0], [0], color=color, linewidth=5, label=name)
        for name, color in model_colors.items()
        if name == "GEB" or name in external_colors
    ]
    if legend_handles:
        legend = fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=9,
            frameon=True,
            framealpha=1.0,
            edgecolor="0.7",
            bbox_to_anchor=(0.5, 0.03),
        )

    _add_filter_summary(fig, filter_summary)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.85, bottom=0.22)

    if export:
        suffix: str = output_name_suffix or (
            "_external_only"
            if not include_geb
            else "_matched"
            if matched_only and external_models
            else ""
        )
        for extension in ("svg", "png"):
            output_path: Path = (
                output_folder / f"evaluation_skill_scores{suffix}.{extension}"
            )
            plt.savefig(output_path, bbox_inches="tight", dpi=150)
            logger.info("Skill score plot saved to: %s", output_path)

    plt.show()
    plt.close()

    logger.info("Skill score plots created.")


def _plot_metric_trendline(
    axis: plt.Axes,
    upstream_area_km2: pd.Series,
    metric_values: pd.Series,
    color: str,
    y_limits: tuple[float, float],
) -> None:
    """Plot a linear trendline for one metric against log-scaled upstream area.

    Args:
        axis: Axis receiving the trendline.
        upstream_area_km2: Upstream area values (km2).
        metric_values: Metric values to fit.
        color: Trendline color.
        y_limits: Visible y-axis limits used to exclude extreme outliers from the fit.
    """
    valid_mask: pd.Series = (
        upstream_area_km2.gt(0)
        & metric_values.notna()
        & metric_values.between(y_limits[0], y_limits[1])
    )
    if valid_mask.sum() < 2:
        return

    log_area: np.ndarray = np.log10(upstream_area_km2[valid_mask].to_numpy(dtype=float))
    values: np.ndarray = metric_values[valid_mask].to_numpy(dtype=float)
    slope, intercept = np.polyfit(log_area, values, deg=1)
    trend_x: np.ndarray = np.logspace(log_area.min(), log_area.max(), 100)
    trend_y: np.ndarray = slope * np.log10(trend_x) + intercept
    axis.plot(trend_x, trend_y, color=color, linewidth=1.8, alpha=0.95)


def _get_robust_metric_ylim(
    metric_values: pd.Series,
    use_error_limits: bool,
) -> tuple[float, float]:
    """Get y-limits that keep scatterplots readable when metric outliers exist.

    Args:
        metric_values: Metric values plotted on one axis.
        use_error_limits: Whether to use nonnegative error-metric limits.

    Returns:
        Lower and upper y-axis limits.
    """
    finite_values: np.ndarray = metric_values.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return (0.0, 1.0)

    if not use_error_limits:
        lower_limit: float = max(float(np.nanpercentile(finite_values, 2)), -2.0)
        upper_limit: float = min(float(np.nanpercentile(finite_values, 98)), 1.05)
        return (lower_limit - 0.05, upper_limit + 0.05)

    upper_limit = float(np.nanpercentile(finite_values, 98))
    return (0.0, max(upper_limit * 1.1, 1.0))


def plot_skill_scores_vs_upstream_area(
    evaluation_df: pd.DataFrame,
    output_folder: Path,
    logger: logging.Logger,
) -> None:
    """Plot upstream area against discharge skill scores.

    Args:
        evaluation_df: Per-station evaluation metrics with `upstream_area_GEB` (m2).
        output_folder: Root folder where the scatterplot is saved.
        logger: Logger to use for progress messages.

    Raises:
        ValueError: If `upstream_area_GEB` is missing from the evaluation metrics.
    """
    if "upstream_area_GEB" not in evaluation_df.columns:
        raise ValueError("`upstream_area_GEB` is missing from evaluation metrics.")

    upstream_area_km2: pd.Series = (
        pd.to_numeric(evaluation_df["upstream_area_GEB"], errors="coerce") / 1_000_000.0
    )

    fig, axes = plt.subplots(
        1,
        len(_DISPLAYED_SKILL_SCORE_CONFIGS),
        figsize=(2.6 * len(_DISPLAYED_SKILL_SCORE_CONFIGS), 4.2),
        sharex=True,
    )
    has_values: bool = False
    for axis, cfg in zip(
        np.atleast_1d(axes).ravel(),
        _DISPLAYED_SKILL_SCORE_CONFIGS,
        strict=True,
    ):
        metric_col: str = str(cfg["col"])
        if metric_col not in evaluation_df.columns:
            logger.info("Metric '%s' not in evaluation data, skipping.", metric_col)
            axis.set_visible(False)
            continue

        metric_values: pd.Series = pd.to_numeric(
            evaluation_df[metric_col], errors="coerce"
        )
        valid_mask: pd.Series = upstream_area_km2.gt(0) & metric_values.notna()
        if not valid_mask.any():
            logger.info("No valid values for metric '%s', skipping.", metric_col)
            axis.set_visible(False)
            continue

        has_values = True
        use_error_limits: bool = bool(cfg.get("robust_error_ylim", False))
        y_limits: tuple[float, float] = _get_robust_metric_ylim(
            metric_values[valid_mask], use_error_limits=use_error_limits
        )
        axis.scatter(
            upstream_area_km2[valid_mask],
            metric_values[valid_mask],
            s=14,
            alpha=0.55,
            color=str(cfg["color"]),
            edgecolors="none",
        )
        _plot_metric_trendline(
            axis=axis,
            upstream_area_km2=upstream_area_km2,
            metric_values=metric_values,
            color=str(cfg["color"]),
            y_limits=y_limits,
        )
        axis.set_xscale("log")
        axis.set_ylim(*y_limits)
        axis.grid(True, color="0.85", linewidth=0.5)
        axis.set_title(str(cfg["label"]), color=str(cfg["color"]), fontweight="bold")
        axis.set_ylabel("Error" if use_error_limits else "Score (-)")
        axis.text(
            0.98,
            0.05,
            f"n={int(valid_mask.sum())}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="0.45",
            fontsize=8,
        )
        for spine in axis.spines.values():
            spine.set_edgecolor("0.7")

    if not has_values:
        logger.warning("No valid skill scores found for upstream-area scatterplot.")
        plt.close(fig)
        return

    fig.suptitle(
        "Discharge Skill Scores vs Upstream Area",
        fontweight="bold",
        fontsize=14,
    )
    for axis in np.atleast_1d(axes).ravel():
        axis.set_xlabel("Upstream area (km2)")

    fig.text(
        0.995,
        0.01,
        "Each panel uses robust y-limits; trendlines are linear fits vs log10(upstream area) after excluding clipped outliers.",
        ha="right",
        va="bottom",
        color="0.45",
        fontsize=8,
    )

    plt.tight_layout()
    output_path: Path = output_folder / "skill_scores_vs_upstream_area"
    for ext in ("svg", "png"):
        plt.savefig(f"{output_path}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info("Saved skill score upstream-area scatterplot to: %s", output_path)
