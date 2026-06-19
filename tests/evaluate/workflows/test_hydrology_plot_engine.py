"""Tests for hydrology evaluation plotting helpers."""

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from geb.evaluate.workflows.hydrology_plot_engine import (
    _draw_violin_box,
    _get_robust_error_metric_ylim,
    _get_skill_score_config,
    _plot_kge_component_maps,
    plot_kge_external_model_comparison,
    plot_skill_score_boxplots,
)


def test_get_robust_error_metric_ylim_uses_percentile_range() -> None:
    """Extreme RMSE/RRMSE outliers do not set the visible y-axis range."""
    metric_values: np.ndarray = np.array(
        [
            0.45,
            0.55,
            0.60,
            0.70,
            0.80,
            0.95,
            1.10,
            1.20,
            1.30,
            1.40,
            1.50,
            48.0,
            162.0,
        ],
        dtype=float,
    )

    lower_limit, upper_limit = _get_robust_error_metric_ylim(metric_values)

    assert lower_limit == 0.0
    assert upper_limit < metric_values.max()


def test_get_robust_error_metric_ylim_handles_missing_values() -> None:
    """Missing error metrics get a neutral fallback y-axis range."""
    metric_values: np.ndarray = np.array([np.nan, np.inf, -np.inf], dtype=float)

    assert _get_robust_error_metric_ylim(metric_values) == (0.0, 1.0)


def test_rrmse_map_uses_red_for_higher_errors() -> None:
    """Higher RRMSE values use the red end of the error-map color scale."""
    rrmse_config: dict[str, object] = _get_skill_score_config("RRMSE")

    assert rrmse_config["cmap"] == "YlOrRd"


def test_violin_density_excludes_values_outside_visible_limits() -> None:
    """Hidden outliers do not stretch the visible violin density."""
    figure, axis = plt.subplots()
    _draw_violin_box(
        axis,
        np.array([-200.0, -0.5, 0.2, 0.5, 0.8]),
        position=0.0,
        bar_color="#1f77b4",
        violin_limits=(-1.0, 1.0),
    )

    violin_vertices: np.ndarray = axis.collections[0].get_paths()[0].vertices
    assert violin_vertices[:, 1].min() >= -1.0
    plt.close(figure)


def test_single_model_skill_score_plot_is_compact_without_legend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GEB-only evaluation uses the compact layout and omits its legend."""
    evaluation_df: pd.DataFrame = pd.DataFrame(
        {
            "KGE": [-200.0, -0.4, 0.2, 0.5, 0.8],
            "NSE": [-3.0, -0.3, 0.1, 0.4, 0.7],
            "R2": [0.1, 0.2, 0.4, 0.6, 0.8],
            "RRMSE": [0.4, 0.6, 0.8, 1.0, 20.0],
            "KGE_correlation": [0.2, 0.4, 0.6, 0.7, 0.9],
            "KGE_bias_ratio": [0.5, 0.8, 1.0, 1.2, 3.0],
            "KGE_variability_ratio": [0.4, 0.7, 0.9, 1.1, 4.0],
        }
    )
    original_close = plt.close
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)

    plot_skill_score_boxplots(
        evaluation_df=evaluation_df,
        external_models={},
        output_folder=Path("."),
        logger=logging.getLogger(__name__),
        export=False,
    )

    figure: plt.Figure = plt.gcf()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    kge_axis, correlation_axis = figure.axes[:2]
    pearson_r2_axis = next(
        axis for axis in figure.axes if axis.get_title() == "Pearson r²"
    )

    np.testing.assert_allclose(figure.get_size_inches(), [8.2, 5.4])
    assert kge_axis.get_position().width > 2 * correlation_axis.get_position().width
    assert pearson_r2_axis.get_title() == "Pearson r²"
    assert figure._suptitle is not None
    assert (
        figure._suptitle.get_window_extent(renderer).y0
        > correlation_axis.title.get_window_extent(renderer).y1
    )
    assert figure.legends == []
    original_close(figure)


def test_plot_kge_component_maps_creates_two_by_two_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The combined spatial plot contains KGE and all three KGE components."""
    evaluation_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "KGE": [0.2, 0.8],
            "KGE_correlation": [0.4, 0.9],
            "KGE_bias_ratio": [0.8, 1.2],
            "KGE_variability_ratio": [0.7, 1.1],
        },
        geometry=[Point(4.0, 51.0), Point(5.0, 52.0)],
        crs="EPSG:4326",
    )
    region_geom: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=[LineString([(3.5, 50.5), (5.5, 52.5)])],
        crs="EPSG:4326",
    )
    metric_columns: tuple[str, str, str, str] = (
        "KGE",
        "KGE_correlation",
        "KGE_bias_ratio",
        "KGE_variability_ratio",
    )
    metric_configs: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ] = (
        _get_skill_score_config(metric_columns[0]),
        _get_skill_score_config(metric_columns[1]),
        _get_skill_score_config(metric_columns[2]),
        _get_skill_score_config(metric_columns[3]),
    )
    original_close = plt.close
    monkeypatch.setattr(
        "geb.evaluate.workflows.hydrology_plot_engine.ctx.add_basemap",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)

    output_path: Path = tmp_path / "skill_score_map_kge_components"
    _plot_kge_component_maps(
        evaluation_gdf=evaluation_gdf,
        metric_configs=metric_configs,
        output_path=output_path,
        region_geom=region_geom,
    )

    figure: plt.Figure = plt.gcf()
    map_titles: list[str] = [axis.get_title() for axis in figure.axes[:4]]
    assert map_titles == [
        "KGE",
        "KGE correlation (r)",
        "KGE bias ratio (β)",
        "KGE variability ratio (α)",
    ]
    assert len(figure.axes) == 8
    assert output_path.with_suffix(".svg").exists()
    assert output_path.with_suffix(".png").exists()
    original_close(figure)


def test_plot_kge_component_maps_rejects_missing_metric(tmp_path: Path) -> None:
    """A clear error is raised when a required KGE component is unavailable."""
    evaluation_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"KGE": [0.5]},
        geometry=[Point(4.0, 51.0)],
        crs="EPSG:4326",
    )
    region_geom: gpd.GeoDataFrame = evaluation_gdf[["geometry"]].copy()
    metric_configs: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ] = (
        _get_skill_score_config("KGE"),
        _get_skill_score_config("KGE_correlation"),
        _get_skill_score_config("KGE_bias_ratio"),
        _get_skill_score_config("KGE_variability_ratio"),
    )

    with pytest.raises(KeyError, match="KGE_correlation"):
        _plot_kge_component_maps(
            evaluation_gdf=evaluation_gdf,
            metric_configs=metric_configs,
            output_path=tmp_path / "unused",
            region_geom=region_geom,
        )


def test_plot_kge_external_model_comparison_writes_outputs(tmp_path: Path) -> None:
    """KGE-only external comparison plot is exported as SVG and PNG."""
    model_kge_values: dict[str, tuple[np.ndarray, np.ndarray, int, float | None]] = {
        "Google Streamflow": (
            np.array([0.2, 0.4, 0.5], dtype=float),
            np.array([0.7, 0.8, 0.9], dtype=float),
            3,
            0.0,
        ),
        "utrecht_1km": (
            np.array([0.3, 0.4, 0.6], dtype=float),
            np.array([0.5, 0.6, 0.7], dtype=float),
            3,
            400.0,
        ),
        "GloFAS": (
            np.array([0.1, 0.2, 0.3], dtype=float),
            np.array([0.2, 0.3, 0.4], dtype=float),
            3,
            0.0,
        ),
    }

    plot_kge_external_model_comparison(
        model_kge_values=model_kge_values,
        output_folder=tmp_path,
        logger=logging.getLogger(__name__),
        export=True,
    )

    boxplots_folder: Path = tmp_path / "skill_score_boxplots"
    assert (
        boxplots_folder / "evaluation_skill_scores_kge_external_comparison.svg"
    ).exists()
    assert (
        boxplots_folder / "evaluation_skill_scores_kge_external_comparison.png"
    ).exists()
