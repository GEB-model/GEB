"""Reusable plot helpers for hydrology evaluation outputs."""

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import contextily as ctx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colorbar import Colorbar
from matplotlib.lines import Line2D
from scipy.stats import linregress

OBSERVATIONS_COLOR: str = "#E6900A"
SIMULATIONS_DEFAULT_COLOR: str = "#278DD9"


def _create_discharge_timeseries_figure(
    validation_df: pd.DataFrame,
    title: str,
    upstream_area_ratio: float,
    metrics: Mapping[str, float],
    include_mean: bool,
) -> plt.Figure:
    """Create a discharge comparison figure.

    Args:
        validation_df: Observed and simulated discharge time series (m3/s).
        title: Figure title.
        upstream_area_ratio: Observed-to-modeled upstream-area ratio
            (dimensionless).
        metrics: Discharge validation metrics keyed by metric name.
        include_mean: Whether to show mean simulated discharge (m3/s).

    Returns:
        Matplotlib figure containing the discharge comparison.
    """
    figure, axis = plt.subplots(figsize=(13, 4))
    for column_name, label, color in (
        ("discharge_simulations", "Simulated", SIMULATIONS_DEFAULT_COLOR),
        ("discharge_observations", "Observed", OBSERVATIONS_COLOR),
    ):
        axis.plot(
            validation_df.index,
            validation_df[column_name],
            label=label,
            linewidth=0.5,
            color=color,
        )
    axis.set(
        title=title,
        xlabel="Time",
        ylabel="Discharge [m3/s]",
        xlim=(validation_df.index.min(), validation_df.index.max()),
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
            f"Mean={validation_df['discharge_simulations'].mean():.2f}"
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


def save_discharge_timeseries_plots(
    station_id: Any,
    validation_df: pd.DataFrame,
    station_name: str,
    upstream_area_ratio: float,
    metrics: Mapping[str, float],
    plot_folder: Path,
    include_yearly_plots: bool,
) -> None:
    """Save full-period and optional yearly station discharge plots.

    Args:
        station_id: Station identifier used in output filenames.
        validation_df: Observed and simulated discharge time series (m3/s).
        station_name: Human-readable station name.
        upstream_area_ratio: Observed-to-modeled upstream-area ratio
            (dimensionless).
        metrics: Discharge validation metrics keyed by metric name.
        plot_folder: Evaluation plot output folder.
        include_yearly_plots: Whether to save one PNG for each calendar year.
    """
    timeseries_folder: Path = plot_folder / "timeseries"
    timeseries_folder.mkdir(parents=True, exist_ok=True)
    figure: plt.Figure = _create_discharge_timeseries_figure(
        validation_df=validation_df,
        title=f"Discharge vs observations for station {station_name}",
        upstream_area_ratio=upstream_area_ratio,
        metrics=metrics,
        include_mean=True,
    )
    figure.savefig(timeseries_folder / f"timeseries_plot_{station_id}.png", dpi=72)
    plt.close(figure)

    if include_yearly_plots:
        yearly_groups: Any = validation_df.groupby(validation_df.index.year)  # ty:ignore[unresolved-attribute]
        for year, yearly_df in yearly_groups:
            year_value: int = int(year)
            yearly_figure: plt.Figure = _create_discharge_timeseries_figure(
                validation_df=yearly_df,
                title=f"GEB discharge vs observations for {year_value} at station {station_name}",
                upstream_area_ratio=upstream_area_ratio,
                metrics=metrics,
                include_mean=False,
            )
            yearly_figure.savefig(
                timeseries_folder / f"timeseries_plot_{station_id}_{year_value}.png",
                dpi=72,
            )
            plt.close(yearly_figure)


_DISPLAYED_SKILL_SCORE_CONFIGS: tuple[dict[str, object], ...] = (
    {
        "col": "KGE",
        "label": "KGE",
        "title": "KGE",
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
        "label": "KGE correlation (r)",
        "title": "KGE correlation (r)",
        "compact_title": "r",
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
        "label": "KGE bias ratio (β)",
        "title": "KGE bias ratio (β)",
        "compact_title": "β",
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
        "label": "KGE variability ratio (α)",
        "title": "KGE variability ratio (α)",
        "compact_title": "α",
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
        "title": "NSE",
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
        "label": "Pearson r²",
        "title": "Pearson r²",
        "unit": "(−)",
        "ylim": (0.0, 1.0),
        "reference": 1.0,
        "cmap": "YlGn",
        "vmin": 0.0,
        "vmax": 1.0,
        "color": "#9467bd",
    },
    {
        "col": "RRMSE",
        "label": "RRMSE",
        "title": "RRMSE",
        "unit": "(−)",
        "ylim": None,
        "robust_error_ylim": True,
        "reference": 0.0,
        "cmap": "YlOrRd",
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
        fontsize=14,
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
        fontsize=18,
        color="white",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5),
    )

    ax.set_title(metric_title, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("0.3")

    plt.tight_layout()
    for ext in ("svg", "png"):
        plt.savefig(f"{output_path}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _plot_kge_component_maps(
    evaluation_gdf: gpd.GeoDataFrame,
    metric_configs: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ],
    output_path: Path,
    region_geom: gpd.GeoDataFrame,
) -> None:
    """Plot KGE and its three components in a two-by-two map grid.

    Each panel uses its metric-specific color scale because KGE and correlation
    range from -1 to 1, while the bias and variability ratios use 0 to 2.

    Args:
        evaluation_gdf: Per-station metrics with point geometry in any CRS.
        metric_configs: Four metric configuration dictionaries in display order:
            KGE, KGE correlation, KGE bias ratio, and KGE variability ratio.
        output_path: Output path stem (no extension); ``.svg`` and ``.png`` are saved.
        region_geom: Basin/region boundary overlaid on each map.

    Raises:
        KeyError: If a configured metric is absent from ``evaluation_gdf``.
        ValueError: If there are not exactly four metric configurations or a
            metric configuration has no finite color scale.
    """
    if len(metric_configs) != 4:
        raise ValueError("Exactly four KGE metric configurations are required.")

    for metric_config in metric_configs:
        configured_column: str = str(metric_config["col"])
        if configured_column not in evaluation_gdf.columns:
            raise KeyError(
                f"Metric '{configured_column}' is missing from evaluation data."
            )
        if metric_config["vmin"] is None or metric_config["vmax"] is None:
            raise ValueError(
                f"Metric '{configured_column}' requires finite vmin and vmax values."
            )

    gdf_3857: gpd.GeoDataFrame = evaluation_gdf.to_crs("EPSG:3857")
    region_3857: gpd.GeoDataFrame = region_geom.to_crs("EPSG:3857")

    fig: plt.Figure
    axes: np.ndarray
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 12),
        constrained_layout=True,
    )

    for ax, cfg in zip(axes.flat, metric_configs, strict=True):
        metric_col: str = str(cfg["col"])
        vmin_value: object = cfg["vmin"]
        vmax_value: object = cfg["vmax"]
        assert vmin_value is not None
        assert vmax_value is not None
        cmap_name: str = str(cfg["cmap"])
        norm: mcolors.Normalize = mcolors.Normalize(
            vmin=float(cast(float, vmin_value)),
            vmax=float(cast(float, vmax_value)),
        )

        region_3857.plot(
            ax=ax,
            color="none",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.7,
            zorder=2,
        )

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
            ax.legend(fontsize=8, loc="lower right", framealpha=0.8)

        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=ctx.providers.Esri.WorldImagery,  # ty:ignore[unresolved-attribute]
            attribution=False,
            zoom="auto",
        )

        scalar_mappable: plt.cm.ScalarMappable = plt.cm.ScalarMappable(
            cmap=cmap_name,
            norm=norm,
        )
        scalar_mappable.set_array([])
        colorbar: Colorbar = fig.colorbar(
            scalar_mappable,
            ax=ax,
            fraction=0.035,
            pad=0.02,
            aspect=28,
        )
        colorbar.set_label(str(cfg["label"]), fontsize=10)
        colorbar.ax.yaxis.set_tick_params(color="black", labelcolor="black")
        colorbar.outline.set_edgecolor("0.3")  # ty:ignore[call-non-callable]

        ax.set_title(str(cfg["label"]), fontsize=12, fontweight="bold", pad=8)
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("0.3")

    # One orientation aid is enough because all panels have identical extents.
    lower_right_axis: plt.Axes = axes[1, 1]
    x_min, x_max = lower_right_axis.get_xlim()
    y_min, y_max = lower_right_axis.get_ylim()
    map_width_m: float = x_max - x_min
    map_height_m: float = y_max - y_min
    bar_m: float = round(
        map_width_m * 0.15 / 10 ** np.floor(np.log10(map_width_m * 0.15))
    ) * 10 ** np.floor(np.log10(map_width_m * 0.15))
    bar_label: str = f"{int(bar_m / 1_000)} km" if bar_m >= 1_000 else f"{int(bar_m)} m"
    bar_x0: float = x_min + map_width_m * 0.03
    bar_y: float = y_min + map_height_m * 0.03
    lower_right_axis.plot(
        [bar_x0, bar_x0 + bar_m],
        [bar_y, bar_y],
        color="white",
        linewidth=3,
        solid_capstyle="butt",
        zorder=5,
    )
    lower_right_axis.text(
        bar_x0 + bar_m / 2,
        bar_y + map_height_m * 0.012,
        bar_label,
        color="white",
        fontsize=14,
        ha="center",
        va="bottom",
        zorder=5,
    )
    lower_right_axis.annotate(
        "N",
        xy=(0.96, 0.12),
        xytext=(0.96, 0.06),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=18,
        color="white",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5),
    )

    fig.suptitle(
        "KGE and its components",
        fontsize=14,
        fontweight="bold",
    )

    for extension in ("svg", "png"):
        fig.savefig(f"{output_path}.{extension}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_skill_score_maps(
    evaluation_gdf: gpd.GeoDataFrame,
    region_geom: gpd.GeoDataFrame,
    output_folder: Path,
    logger: logging.Logger,
    difference_gdfs: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Plot per-station skill scores on a satellite basemap, one map per metric.

    Saves SVG and PNG files under ``output_folder/skill_score_maps/``.

    Args:
        evaluation_gdf: Per-station metrics with point geometry in any CRS.
        region_geom: Basin/region boundary overlaid on each map.
        output_folder: Root folder under which ``skill_score_maps/`` is created.
        logger: Logger to use for progress messages.
        difference_gdfs: Optional matched GEB-vs-external station tables with
            ``KGE_difference`` values (dimensionless).
    """
    maps_folder = output_folder / "skill_score_maps"
    maps_folder.mkdir(parents=True, exist_ok=True)

    kge_metric_columns: tuple[str, str, str, str] = (
        "KGE",
        "KGE_correlation",
        "KGE_bias_ratio",
        "KGE_variability_ratio",
    )
    if all(column_name in evaluation_gdf.columns for column_name in kge_metric_columns):
        kge_metric_configs: tuple[
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, object],
        ] = (
            _get_skill_score_config(kge_metric_columns[0]),
            _get_skill_score_config(kge_metric_columns[1]),
            _get_skill_score_config(kge_metric_columns[2]),
            _get_skill_score_config(kge_metric_columns[3]),
        )
        _plot_kge_component_maps(
            evaluation_gdf=evaluation_gdf,
            metric_configs=kge_metric_configs,
            output_path=maps_folder / "skill_score_map_kge_components",
            region_geom=region_geom,
        )
        logger.info("Saved KGE component skill score map.")

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
            else float(cast(float, cfg["vmax"]))
        )
        _plot_skill_score_map_single(
            evaluation_gdf=evaluation_gdf,
            metric_col=col,
            metric_label=str(cfg["label"]),
            metric_title=str(cfg["title"]),
            cmap_name=str(cfg["cmap"]),
            vmin=float(cast(float, cfg["vmin"])),
            vmax=vmax,
            output_path=maps_folder / f"skill_score_map_{col.lower()}",
            region_geom=region_geom,
        )
        logger.info("Saved skill score map for %s.", col)

    for model_name, difference_df in (difference_gdfs or {}).items():
        if "KGE_difference" not in difference_df:
            continue
        has_geometry: bool = "geometry" in difference_df
        if not has_geometry and not {"x", "y"}.issubset(difference_df):
            logger.info("No station geometry found for %s difference map.", model_name)
            continue
        if not has_geometry:
            difference_df = gpd.GeoDataFrame(
                difference_df,
                geometry=gpd.points_from_xy(difference_df["x"], difference_df["y"]),
                crs="EPSG:4326",
            )
        unmatched_difference_df: gpd.GeoDataFrame = gpd.GeoDataFrame(
            evaluation_gdf.loc[~evaluation_gdf.index.isin(difference_df.index)].copy(),
            geometry="geometry",
            crs=evaluation_gdf.crs,
        )
        unmatched_difference_df["KGE_difference"] = np.nan
        matched_station_count: int = len(difference_df)
        if not unmatched_difference_df.empty:
            # Difference maps otherwise hide whole regions where the external
            # source has no station match, which can look like a plotting error.
            difference_df = pd.concat(
                [difference_df, unmatched_difference_df],
                axis=0,
                copy=False,
            )
            logger.info(
                "%s difference map shows %d matched stations and %d unmatched "
                "eligible GEB stations.",
                model_name,
                matched_station_count,
                len(unmatched_difference_df),
            )
        difference_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
            difference_df,
            geometry="geometry",
            crs=getattr(difference_df, "crs", None),
        )
        valid_values: np.ndarray = (
            difference_gdf["KGE_difference"].dropna().to_numpy(dtype=float)
        )
        if valid_values.size == 0:
            continue
        visible_limit: float = max(float(np.nanpercentile(abs(valid_values), 95)), 0.05)
        output_suffix: str = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        _plot_skill_score_map_single(
            evaluation_gdf=difference_gdf,
            metric_col="KGE_difference",
            metric_label="KGE difference (-)",
            metric_title=f"KGE Difference: GEB - {model_name}",
            cmap_name="RdBu",
            vmin=-visible_limit,
            vmax=visible_limit,
            output_path=maps_folder / f"skill_score_difference_map_{output_suffix}",
            region_geom=region_geom,
        )
        logger.info("Saved skill score difference map for %s.", model_name)

    logger.info("All skill score maps saved to: %s", maps_folder)


def _draw_violin_box(
    axis: plt.Axes,
    values: np.ndarray,
    position: float,
    bar_color: str,
    violin_width: float = 0.35,
    violin_limits: tuple[float, float] | None = None,
) -> None:
    """Draw a violin and boxplot for one model metric distribution.

    Args:
        axis: Axis receiving the distribution.
        values: Metric values for one model.
        position: X-axis position of the distribution.
        bar_color: Color used for the violin and box.
        violin_width: Width of the violin plot.
        violin_limits: Optional visible y-axis limits. Values outside these
            limits are excluded from the violin density but retained in the
            boxplot and median.
    """
    density_values: np.ndarray = values[np.isfinite(values)]
    if violin_limits is not None:
        lower_limit, upper_limit = violin_limits
        density_values = density_values[
            (density_values >= lower_limit) & (density_values <= upper_limit)
        ]
    if len(density_values) >= 3:
        parts = axis.violinplot(
            density_values,
            positions=[position],
            showmedians=False,
            showextrema=False,
            widths=violin_width,
            bw_method=0.15,
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
    single_model: bool,
) -> dict[str, plt.Axes]:
    """Create the grouped layout for discharge skill-score boxplots.

    The top row keeps KGE next to its three component scores. The bottom row
    contains the remaining headline skill scores across the full figure width.

    Args:
        fig: Matplotlib figure that receives the axes.
        single_model: Whether only one model distribution is plotted.

    Returns:
        Metric-column-to-axis mapping.
    """
    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1.0, 1.0],
        hspace=0.80,
    )
    kge_grid = outer_grid[0, 0].subgridspec(
        nrows=1,
        ncols=4,
        wspace=0.20,
        width_ratios=[2.4 if single_model else 4.0, 1, 1, 1],
    )
    other_metric_grid = outer_grid[1, 0].subgridspec(
        nrows=1,
        ncols=3,
        wspace=0.28,
    )
    axes_by_metric: dict[str, plt.Axes] = {
        "KGE": fig.add_subplot(kge_grid[0, 0]),
        "KGE_correlation": fig.add_subplot(kge_grid[0, 1]),
        "KGE_bias_ratio": fig.add_subplot(kge_grid[0, 2]),
        "KGE_variability_ratio": fig.add_subplot(kge_grid[0, 3]),
        "NSE": fig.add_subplot(other_metric_grid[0, 0]),
        "R2": fig.add_subplot(other_metric_grid[0, 1]),
        "RRMSE": fig.add_subplot(other_metric_grid[0, 2]),
    }

    return axes_by_metric


def _annotate_metric_medians(
    axis: plt.Axes,
    models_with_data: list[tuple[str, np.ndarray]],
    x_positions: np.ndarray,
    compact: bool,
) -> None:
    """Add median labels below each metric violin.

    Args:
        axis: Axis receiving the labels.
        models_with_data: Model names and finite metric values to summarize.
        x_positions: Violin x positions in axis data coordinates.
        compact: Whether the labels are drawn in a smaller component axis.
    """
    base_label_y: float = -0.11
    for label_index, ((_, metric_values), x_position) in enumerate(
        zip(
            models_with_data,
            x_positions,
            strict=True,
        )
    ):
        label_y: float = base_label_y
        label_fontsize: float = 5.5 if compact else 7.0
        axis.text(
            float(x_position),
            label_y,
            f"med={float(np.median(metric_values)):.2f}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=label_fontsize,
            color="0.25",
            clip_on=False,
        )


def _has_positive_upstream_area_threshold(
    minimum_upstream_area_km2: float | None,
) -> bool:
    """Check whether an upstream-area threshold should be shown.

    Args:
        minimum_upstream_area_km2: Minimum upstream-area threshold (km2).

    Returns:
        True when the threshold is positive and should be displayed.
    """
    return minimum_upstream_area_km2 is not None and minimum_upstream_area_km2 > 0.0


def _format_boxplot_upstream_area_label(
    minimum_upstream_area_km2: float | None,
) -> str:
    """Format an upstream-area threshold for titles.

    Args:
        minimum_upstream_area_km2: Minimum upstream-area threshold (km2).

    Returns:
        Human-readable upstream-area threshold.
    """
    if not _has_positive_upstream_area_threshold(minimum_upstream_area_km2):
        return ""
    return f"upstream area >= {minimum_upstream_area_km2:g} km2"


def _format_boxplot_output_context(
    minimum_upstream_area_km2: float | None,
    station_count: int | None,
) -> str:
    """Format plot metadata for output filenames.

    Args:
        minimum_upstream_area_km2: Minimum upstream-area threshold (km2).
        station_count: Number of plotted GEB stations.

    Returns:
        Filename-safe suffix containing the threshold and station count.
    """
    context_parts: list[str] = []
    if _has_positive_upstream_area_threshold(minimum_upstream_area_km2):
        area_label: str = f"{minimum_upstream_area_km2:g}".replace(".", "p")
        context_parts.append(f"upstream_area_{area_label}km2")
    if station_count is not None:
        context_parts.append(f"n{station_count}")
    return f"_{'_'.join(context_parts)}" if context_parts else ""


def _format_boxplot_title_context(
    minimum_upstream_area_km2: float | None,
    station_count: int | None,
    matched_only: bool,
    has_external_models: bool,
) -> str:
    """Format plot metadata for the boxplot title.

    Args:
        minimum_upstream_area_km2: Minimum upstream-area threshold (km2).
        station_count: Number of plotted GEB stations.
        matched_only: Whether GEB and external stations are restricted to overlap.
        has_external_models: Whether external model scores are plotted.

    Returns:
        Parenthesized title context, or an empty string when no metadata is known.
    """
    context_parts: list[str] = []
    if matched_only and has_external_models:
        context_parts.append("matched stations only")
    if upstream_area_label := _format_boxplot_upstream_area_label(
        minimum_upstream_area_km2
    ):
        context_parts.append(upstream_area_label)
    if station_count is not None:
        context_parts.append(f"n={station_count}")
    return f" ({'; '.join(context_parts)})" if context_parts else ""


def _get_external_model_plot_order(model_name: str) -> tuple[int, str]:
    """Get a stable display order for final external-model comparisons.

    Args:
        model_name: External model label.

    Returns:
        Sort key with priority and lower-case model label.
    """
    model_name_lower: str = model_name.lower()
    model_groups: tuple[tuple[str, ...], ...] = (
        ("pcr-globwb", "utrecht"),
        ("google",),
        ("glofas",),
    )
    for priority, model_keys in enumerate(model_groups):
        if any(model_key in model_name_lower for model_key in model_keys):
            return priority, model_name_lower
    return 99, model_name_lower


def _format_external_model_short_name(model_name: str) -> str:
    """Format an external model name for compact axis labels.

    Args:
        model_name: External model label.

    Returns:
        Short display label.
    """
    model_name_lower: str = model_name.lower()
    if "pcr-globwb" in model_name_lower or "utrecht" in model_name_lower:
        return "PCR-GLOBWB"
    if "google" in model_name_lower:
        return "Google"
    if "glofas" in model_name_lower:
        if "non-calibrated" in model_name_lower or "non_calibrated" in model_name_lower:
            return "GloFAS non-cal gauges"
        if "all" in model_name_lower:
            return "GloFAS all"
        return "GloFAS"
    return model_name


def plot_skill_score_boxplots(
    evaluation_df: pd.DataFrame,
    external_models: dict[str, pd.DataFrame],
    output_folder: Path,
    logger: logging.Logger,
    export: bool = True,
    include_geb: bool = True,
    matched_only: bool = False,
    output_name_suffix: str = "",
    minimum_upstream_area_km2: float | None = None,
    station_count: int | None = None,
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
        output_name_suffix: Optional suffix appended to the output file stem.
        minimum_upstream_area_km2: Minimum modeled upstream area threshold for
            plotted GEB stations (km2).
        station_count: Number of plotted GEB stations.

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
    single_model: bool = len(model_colors) == 1

    logger.info("Creating evaluation metrics skill score plots...")

    fig = plt.figure(
        figsize=(8.2, 5.4) if single_model else (13.0, 6.5),
        constrained_layout=False,
    )
    axes_by_metric = _get_boxplot_axis_layout(fig, single_model=single_model)
    displayed_station_count: int | None = (
        station_count if station_count is not None else len(evaluation_df)
    )
    plot_context: str = _format_boxplot_title_context(
        minimum_upstream_area_km2=minimum_upstream_area_km2,
        station_count=displayed_station_count,
        matched_only=matched_only,
        has_external_models=bool(external_models),
    )
    fig.suptitle(
        f"Discharge Evaluation — Skill Score Distributions{plot_context}",
        fontsize=14,
        fontweight="bold",
        y=0.97,
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

        y_limits: tuple[float, float] = (
            cast(tuple[float, float], config["ylim"])
            if config["ylim"] is not None
            else _get_robust_error_metric_ylim(
                np.concatenate([metric_values for _, metric_values in models_with_data])
            )
        )

        for (model_name, metric_values), x_position in zip(
            models_with_data, x_positions, strict=True
        ):
            _draw_violin_box(
                axis,
                metric_values,
                x_position,
                model_colors[model_name],
                violin_width=0.28 if single_model else 0.35,
                violin_limits=y_limits,
            )
        is_component_axis: bool = metric_col.startswith("KGE_")
        _annotate_metric_medians(
            axis,
            models_with_data,
            x_positions,
            compact=is_component_axis,
        )

        if is_component_axis:
            title_str: str = str(config["title"])
            last_space_paren: int = title_str.rfind(" (")
            if last_space_paren != -1:
                axis_title: str = (
                    title_str[:last_space_paren]
                    + "\n"
                    + title_str[last_space_paren + 1 :]
                )
            else:
                axis_title = title_str
        else:
            axis_title = str(config["title"])
        axis.set_title(
            axis_title,
            fontsize=8 if is_component_axis else 9,
            fontweight="bold",
            pad=8,
        )
        axis.set_ylim(*y_limits)
        axis.set_xticks([])
        axis.tick_params(
            axis="y",
            labelsize=7 if is_component_axis else 8,
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
    ]
    if len(legend_handles) > 1:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=9,
            frameon=True,
            framealpha=1.0,
            edgecolor="0.7",
            bbox_to_anchor=(0.5, 0.01),
        )

    fig.subplots_adjust(
        left=0.07 if single_model else 0.055,
        right=0.985,
        top=0.82 if single_model else 0.88,
        bottom=0.10 if single_model else 0.14,
    )

    if export:
        context_suffix: str = _format_boxplot_output_context(
            minimum_upstream_area_km2=minimum_upstream_area_km2,
            station_count=displayed_station_count,
        )
        suffix: str = output_name_suffix or (
            "_external_only"
            if not include_geb
            else "_matched"
            if matched_only and external_models
            else ""
        )
        boxplots_folder: Path = output_folder / "skill_score_boxplots"
        boxplots_folder.mkdir(parents=True, exist_ok=True)
        for extension in ("svg", "png"):
            output_path: Path = (
                boxplots_folder
                / f"evaluation_skill_scores{suffix}{context_suffix}.{extension}"
            )
            plt.savefig(output_path, bbox_inches="tight", dpi=150)
            logger.info("Skill score plot saved to: %s", output_path)

    plt.show()
    plt.close()

    logger.info("Skill score plots created.")


def plot_kge_external_model_comparison(
    model_kge_values: dict[str, tuple[np.ndarray, np.ndarray, int, float | None]],
    output_folder: Path,
    logger: logging.Logger,
    export: bool = True,
) -> None:
    """Create one KGE-only comparison plot for all external models.

    Args:
        model_kge_values: Mapping from external model name to a tuple with GEB
            KGE values, external-model KGE values, matched station count, and
            effective GEB upstream-area threshold (km2).
        output_folder: Folder where output figures are saved.
        logger: Logger to use for progress messages.
        export: Save the figure to disk.
    """
    ordered_model_items: list[
        tuple[str, tuple[np.ndarray, np.ndarray, int, float | None]]
    ] = sorted(
        model_kge_values.items(),
        key=lambda item: _get_external_model_plot_order(item[0]),
    )
    if not ordered_model_items:
        logger.info("No matched external KGE data found. Skipping KGE comparison plot.")
        return

    geb_color: str = "#1f77b4"
    external_color: str = "#d62728"
    fig, axis = plt.subplots(figsize=(8.8, 4.9), constrained_layout=False)

    x_tick_positions: list[float] = []
    x_tick_labels: list[str] = []
    visible_y_min: float = -1.0
    visible_y_max: float = 1.0
    median_label_y: float = -0.94
    for group_index, (model_name, model_values) in enumerate(ordered_model_items):
        geb_values, external_values, station_count, minimum_upstream_area_km2 = (
            model_values
        )
        finite_geb_values: np.ndarray = geb_values[np.isfinite(geb_values)]
        finite_external_values: np.ndarray = external_values[
            np.isfinite(external_values)
        ]
        if finite_geb_values.size == 0 or finite_external_values.size == 0:
            continue

        group_position: float = float(group_index)
        x_positions: np.ndarray = np.array(
            [group_position - 0.18, group_position + 0.18]
        )
        _draw_violin_box(
            axis,
            finite_geb_values,
            x_positions[0],
            geb_color,
            violin_width=0.28,
            violin_limits=(visible_y_min, visible_y_max),
        )
        _draw_violin_box(
            axis,
            finite_external_values,
            x_positions[1],
            external_color,
            violin_width=0.28,
            violin_limits=(visible_y_min, visible_y_max),
        )
        for x_position, metric_values in zip(
            x_positions,
            (finite_geb_values, finite_external_values),
            strict=True,
        ):
            axis.text(
                float(x_position),
                median_label_y,
                f"med={float(np.median(metric_values)):.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="0.25",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )

        threshold_label: str = (
            f"\n>= {minimum_upstream_area_km2:g} km2"
            if _has_positive_upstream_area_threshold(minimum_upstream_area_km2)
            else ""
        )
        x_tick_positions.append(group_position)
        x_tick_labels.append(
            f"{_format_external_model_short_name(model_name)}\nn={station_count}{threshold_label}"
        )

    if not x_tick_positions:
        logger.info("No finite matched KGE data found. Skipping KGE comparison plot.")
        plt.close(fig)
        return

    axis.axhline(1.0, color="0.5", linewidth=0.8, linestyle="--", zorder=0)
    axis.set_ylim(visible_y_min, visible_y_max)
    axis.set_ylabel("KGE")
    axis.set_xticks(x_tick_positions)
    axis.set_xticklabels(x_tick_labels, fontsize=8)
    axis.set_title(
        "Discharge Evaluation — KGE Comparison Across Matched External Models",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axis.grid(axis="y", color="0.85", linewidth=0.5)
    for spine in axis.spines.values():
        spine.set_edgecolor("0.7")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)

    axis.legend(
        handles=[
            Line2D([0], [0], color=geb_color, linewidth=5, label="GEB"),
            Line2D([0], [0], color=external_color, linewidth=5, label="External model"),
        ],
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=True,
        framealpha=1.0,
        edgecolor="0.7",
        bbox_to_anchor=(0.5, -0.34),
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.33)

    if export:
        boxplots_folder: Path = output_folder / "skill_score_boxplots"
        boxplots_folder.mkdir(parents=True, exist_ok=True)
        for extension in ("svg", "png"):
            output_path: Path = (
                boxplots_folder
                / f"evaluation_skill_scores_kge_external_comparison.{extension}"
            )
            plt.savefig(output_path, bbox_inches="tight", dpi=150)
            logger.info("KGE external comparison plot saved to: %s", output_path)

    plt.show()
    plt.close(fig)


def _plot_metric_trendline(
    axis: plt.Axes,
    upstream_area_km2: pd.Series,
    metric_values: pd.Series,
    color: str,
    y_limits: tuple[float, float],
) -> tuple[float, float] | None:
    """Plot a linear trendline for one metric against log-scaled upstream area.

    Args:
        axis: Axis receiving the trendline.
        upstream_area_km2: Upstream area values (km2).
        metric_values: Metric values to fit.
        color: Trendline color.
        y_limits: Visible y-axis limits used to exclude extreme outliers from the fit.

    Returns:
        Tuple with slope per log10(km2) and p-value, or `None` when too few
        values are available.
    """
    valid_mask: pd.Series = (
        upstream_area_km2.gt(0)
        & metric_values.notna()
        & metric_values.between(y_limits[0], y_limits[1])
    )
    if valid_mask.sum() < 2:
        return None

    log_area: np.ndarray = np.log10(upstream_area_km2[valid_mask].to_numpy(dtype=float))
    values: np.ndarray = metric_values[valid_mask].to_numpy(dtype=float)
    trend_result = linregress(log_area, values)
    trend_x: np.ndarray = np.logspace(log_area.min(), log_area.max(), 100)
    trend_y: np.ndarray = (
        trend_result.slope * np.log10(trend_x) + trend_result.intercept
    )
    axis.plot(trend_x, trend_y, color=color, linewidth=1.8, alpha=0.95)
    return (
        float(trend_result.slope),
        float(trend_result.pvalue),
    )


def _format_p_value(p_value: float) -> str:
    """Format a regression p-value compactly for plot annotations.

    Args:
        p_value: Trendline p-value (dimensionless).

    Returns:
        Compact p-value label.
    """
    return "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"


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

    metric_order: tuple[str, ...] = (
        "KGE",
        "KGE_correlation",
        "KGE_bias_ratio",
        "KGE_variability_ratio",
        "NSE",
        "RRMSE",
    )
    metric_configs: tuple[dict[str, object], ...] = tuple(
        _get_skill_score_config(metric_col) for metric_col in metric_order
    )
    fig = plt.figure(figsize=(13.5, 7.2), constrained_layout=True)
    # 6-column grid: KGE spans 3 cols (half width), each KGE component 1 col.
    # NSE and RRMSE each span 3 cols in the bottom row.
    grid = fig.add_gridspec(2, 6)
    axes_by_metric: dict[str, plt.Axes] = {
        "KGE": fig.add_subplot(grid[0, 0:3]),
        "KGE_correlation": fig.add_subplot(grid[0, 3]),
        "KGE_bias_ratio": fig.add_subplot(grid[0, 4]),
        "KGE_variability_ratio": fig.add_subplot(grid[0, 5]),
        "NSE": fig.add_subplot(grid[1, 0:3]),
        "RRMSE": fig.add_subplot(grid[1, 3:6]),
    }
    has_values: bool = False
    total_valid_station_count: int = int(upstream_area_km2.gt(0).sum())
    plot_color: str = "#1f77b4"
    for cfg in metric_configs:
        axis: plt.Axes = axes_by_metric[str(cfg["col"])]
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
        if cfg["ylim"] is not None:
            y_limits: tuple[float, float] = cast(tuple[float, float], cfg["ylim"])
            use_error_limits: bool = False
        else:
            use_error_limits = bool(cfg.get("robust_error_ylim", False))
            y_limits = _get_robust_metric_ylim(
                metric_values[valid_mask], use_error_limits=use_error_limits
            )
        axis.scatter(
            upstream_area_km2[valid_mask],
            metric_values[valid_mask],
            s=14,
            alpha=0.55,
            color=plot_color,
            edgecolors="none",
        )
        trend_stats: tuple[float, float] | None = _plot_metric_trendline(
            axis=axis,
            upstream_area_km2=upstream_area_km2,
            metric_values=metric_values,
            color=plot_color,
            y_limits=y_limits,
        )
        if trend_stats is not None:
            slope, p_value = trend_stats
            axis.text(
                0.04,
                0.92,
                f"slope={slope:.2f}\n{_format_p_value(p_value)}",
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
        axis.set_title(str(cfg["title"]), color=plot_color, fontweight="bold")
        axis.set_ylabel("Error" if use_error_limits else "Score")
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
    for axis in axes_by_metric.values():
        axis.set_xlabel("Upstream area (km2)")

    # Robust y-limits keep the main scatter visible; trendlines are fitted vs
    # log10(upstream area) after excluding values outside the visible limits.
    fig.text(
        0.985,
        0.02,
        f"n={total_valid_station_count}",
        ha="right",
        va="bottom",
        color="0.45",
        fontsize=8,
    )

    scatterplots_folder: Path = output_folder / "skill_score_scatterplots"
    scatterplots_folder.mkdir(parents=True, exist_ok=True)
    output_path: Path = scatterplots_folder / "skill_scores_vs_upstream_area"
    for ext in ("svg", "png"):
        plt.savefig(f"{output_path}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info("Saved skill score upstream-area scatterplot to: %s", output_path)
