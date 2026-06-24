"""Reusable plot helpers for hydrology evaluation outputs."""

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
from matplotlib.transforms import blended_transform_factory

_SKILL_SCORE_MAP_METRIC_CONFIGS: list[dict] = [
    {
        "col": "KGE",
        "label": "KGE (−)",
        "title": "Kling-Gupta Efficiency (KGE)",
        "cmap": "RdYlGn",
        "vmin": -1.0,
        "vmax": 1.0,
    },
    {
        "col": "NSE",
        "label": "NSE (−)",
        "title": "Nash-Sutcliffe Efficiency (NSE)",
        "cmap": "RdYlGn",
        "vmin": -1.0,
        "vmax": 1.0,
    },
    {
        "col": "R2",
        "label": "R² (−)",
        "title": "Coefficient of Determination (R²)",
        "cmap": "YlGn",
        "vmin": 0.0,
        "vmax": 1.0,
    },
    {
        "col": "RMSE",
        "label": "RMSE (m³/s)",
        "title": "Root Mean Squared Error (RMSE)",
        "cmap": "YlOrRd_r",
        "vmin": 0.0,
        # Upper bound set to 95th percentile to avoid outlier compression
        "vmax": None,
    },
    {
        "col": "RRMSE",
        "label": "RRMSE (−)",
        "title": "Relative Root Mean Squared Error (RRMSE)",
        "cmap": "YlOrRd_r",
        "vmin": 0.0,
        "vmax": None,
    },
]

_SKILL_SCORE_SCATTER_CONFIGS: list[dict[str, str]] = [
    {"col": "KGE", "label": "KGE", "color": "#1f77b4", "panel": "skill"},
    {"col": "NSE", "label": "NSE", "color": "#2ca02c", "panel": "skill"},
    {"col": "R", "label": "R", "color": "#17becf", "panel": "skill"},
    {"col": "R2", "label": "R2", "color": "#9467bd", "panel": "skill"},
    {"col": "RMSE", "label": "RMSE", "color": "#d62728", "panel": "error"},
    {"col": "RRMSE", "label": "RRMSE", "color": "#ff7f0e", "panel": "error"},
]


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
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

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
            framealpha=0.6,
            facecolor="black",
            edgecolor="white",
            labelcolor="white",
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
    cbar.set_label(metric_label, fontsize=11, color="white")
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cbar.outline.set_edgecolor("white")  # ty:ignore[call-non-callable]

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

    ax.set_title(metric_title, fontsize=13, fontweight="bold", color="white", pad=10)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

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

    for cfg in _SKILL_SCORE_MAP_METRIC_CONFIGS:
        col: str = cfg["col"]
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
            metric_label=cfg["label"],
            metric_title=cfg["title"],
            cmap_name=cfg["cmap"],
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
        medianprops={"color": "white", "linewidth": 2.0, "zorder": 5},
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


def plot_skill_score_boxplots(
    evaluation_df: pd.DataFrame,
    external_models: dict[str, pd.DataFrame],
    output_folder: Path,
    logger: logging.Logger,
    export: bool = True,
    include_geb: bool = True,
    matched_only: bool = False,
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

    metric_configs: list[dict] = [
        {
            "col": "KGE",
            "label": "KGE",
            "title": "Kling-Gupta Efficiency",
            "ylim": (-1.0, 1.0),
            "reference": 1.0,
            "unit": "(−)",
        },
        {
            "col": "NSE",
            "label": "NSE",
            "title": "Nash-Sutcliffe Efficiency",
            "ylim": (-1.0, 1.0),
            "reference": 1.0,
            "unit": "(−)",
        },
        {
            "col": "R2",
            "label": "R²",
            "title": "Coefficient of Determination",
            "ylim": (0.0, 1.0),
            "reference": 1.0,
            "unit": "(−)",
        },
        {
            "col": "RMSE",
            "label": "RMSE",
            "title": "Root Mean Squared Error",
            "ylim": None,
            "reference": 0.0,
            "unit": "(m³/s)",
        },
        {
            "col": "RRMSE",
            "label": "RRMSE",
            "title": "Relative RMSE",
            "ylim": None,
            "reference": 0.0,
            "unit": "(−)",
        },
    ]

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

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.patch.set_facecolor("black")
        plot_subtitle: str = (
            " (matched stations only)" if matched_only and external_models else ""
        )
        fig.suptitle(
            f"Discharge Evaluation — Skill Score Distributions{plot_subtitle}",
            fontsize=14,
            fontweight="bold",
            color="white",
            y=1.01,
        )

        for axis, config in zip(axes.flat, metric_configs, strict=False):
            metric_col: str = config["col"]
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
                transform = blended_transform_factory(axis.transData, axis.transAxes)
                axis.text(
                    x_position,
                    1.03,
                    f"med={float(np.median(metric_values)):.2f}  n={len(metric_values)}",
                    transform=transform,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="white",
                )

            axis.axhline(
                config["reference"],
                color="0.5",
                linewidth=0.8,
                linestyle="--",
                zorder=0,
            )
            axis.set_facecolor("black")
            axis.set_title(
                f"{config['label']} — {config['title']} {config['unit']}",
                fontsize=9,
                fontweight="bold",
                color="white",
                pad=18,
            )
            if config["ylim"] is not None:
                axis.set_ylim(*config["ylim"])
            axis.set_xticks([])
            axis.tick_params(axis="y", labelsize=8, colors="white")
            axis.set_xlabel("")
            for spine in axis.spines.values():
                spine.set_edgecolor("0.4")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["bottom"].set_visible(False)
            axis.yaxis.label.set_color("white")
            axis.grid(axis="y", color="0.25", linewidth=0.5)

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
                edgecolor="0.4",
                bbox_to_anchor=(0.5, -0.04),
            )
            legend.get_frame().set_facecolor("black")
            for text in legend.get_texts():
                text.set_color("white")

        plt.tight_layout()

        if export:
            suffix: str = (
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
    panel: str,
) -> tuple[float, float]:
    """Get y-limits that keep scatterplots readable when metric outliers exist.

    Args:
        metric_values: Metric values plotted on one axis.
        panel: Panel name, either `skill` or `error`.

    Returns:
        Lower and upper y-axis limits.
    """
    finite_values: np.ndarray = metric_values.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return (0.0, 1.0)

    if panel == "skill":
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

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("black")

    has_values: bool = False
    for axis, cfg in zip(axes.flat, _SKILL_SCORE_SCATTER_CONFIGS, strict=True):
        metric_col: str = cfg["col"]
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

        panel: str = cfg["panel"]
        has_values = True
        y_limits: tuple[float, float] = _get_robust_metric_ylim(
            metric_values[valid_mask], panel=panel
        )
        axis.scatter(
            upstream_area_km2[valid_mask],
            metric_values[valid_mask],
            s=14,
            alpha=0.45,
            color=cfg["color"],
            edgecolors="white",
            linewidths=0.15,
        )
        _plot_metric_trendline(
            axis=axis,
            upstream_area_km2=upstream_area_km2,
            metric_values=metric_values,
            color=cfg["color"],
            y_limits=y_limits,
        )
        axis.set_xscale("log")
        axis.set_ylim(*y_limits)
        axis.set_facecolor("black")
        axis.grid(True, color="0.25", linewidth=0.5)
        axis.tick_params(colors="white")
        axis.set_title(cfg["label"], color=cfg["color"], fontweight="bold")
        axis.set_ylabel("Score (-)" if panel == "skill" else "Error")
        axis.text(
            0.98,
            0.05,
            f"n={int(valid_mask.sum())}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="0.75",
            fontsize=8,
        )
        for spine in axis.spines.values():
            spine.set_edgecolor("white")

    if not has_values:
        logger.warning("No valid skill scores found for upstream-area scatterplot.")
        plt.close(fig)
        return

    fig.suptitle(
        "Discharge Skill Scores vs Upstream Area",
        color="white",
        fontweight="bold",
        fontsize=14,
    )
    for axis in axes[-1, :]:
        axis.set_xlabel("Upstream area (km2)")

    fig.text(
        0.995,
        0.01,
        "Each panel uses robust y-limits; trendlines are linear fits vs log10(upstream area) after excluding clipped outliers.",
        ha="right",
        va="bottom",
        color="0.7",
        fontsize=8,
    )

    plt.tight_layout()
    output_path: Path = output_folder / "skill_scores_vs_upstream_area"
    for ext in ("svg", "png"):
        plt.savefig(f"{output_path}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info("Saved skill score upstream-area scatterplot to: %s", output_path)
