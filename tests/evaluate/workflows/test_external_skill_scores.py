"""Tests for external discharge skill-score processing."""

import logging
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from geb.evaluate.workflows.external_skill_scores import (
    GOOGLE_STREAMFLOW_MODEL_NAME,
    prepare_skill_score_boxplot_inputs,
    read_external_evaluation_raw,
    read_google_streamflow_skill_scores,
    read_google_streamflow_skill_scores_from_archive,
)


def test_read_external_evaluation_raw_reads_csv_scores(
    tmp_path: Path,
) -> None:
    """External CSV reading keeps score-only tables unchanged."""
    utrecht_df: pd.DataFrame = pd.DataFrame(
        {"KGE": [0.8, 0.7]},
        index=pd.Index(["station_a", "station_b"]),
    )
    google_df: pd.DataFrame = pd.DataFrame(
        {"KGE": [0.6, 0.5]},
        index=pd.Index(["station_a", "station_b"]),
    )
    utrecht_df.to_csv(tmp_path / "utrecht.csv")
    google_df.to_csv(tmp_path / "google_streamflow.csv")

    external_models: dict[str, pd.DataFrame] = read_external_evaluation_raw(
        external_evaluation_folder=None,
        configured_external_evaluation_folder=tmp_path,
        model_folder=tmp_path,
        logger=logging.getLogger(__name__),
    )

    assert external_models["utrecht"].index.to_list() == ["STATION_A", "STATION_B"]
    assert external_models["google_streamflow"].index.to_list() == [
        "STATION_A",
        "STATION_B",
    ]


def test_prepare_skill_score_boxplot_inputs_matches_after_geb_filter(
    tmp_path: Path,
) -> None:
    """Matched boxplot inputs keep external stations retained after GEB filtering."""
    evaluation_metrics_path: Path = tmp_path / "evaluation_metrics.xlsx"
    external_folder: Path = tmp_path / "external"
    external_folder.mkdir()
    output_folder: Path = tmp_path / "output"
    output_folder.mkdir()
    evaluation_df: pd.DataFrame = pd.DataFrame(
        {
            "station_name": ["station_a", "station_b", "station_c"],
            "upstream_area_GEB": [500_000_000.0, 200_000_000.0, 600_000_000.0],
            "KGE": [0.8, 0.7, float("nan")],
        }
    )
    external_df: pd.DataFrame = pd.DataFrame(
        {"KGE": [0.6, 0.5, 0.4]},
        index=pd.Index(["station_a", "station_b", "station_c"]),
    )
    evaluation_df.to_excel(evaluation_metrics_path, index=False)
    external_df.to_csv(external_folder / "reference.csv")

    prepared_geb_df, external_models = prepare_skill_score_boxplot_inputs(
        evaluation_metrics_path=evaluation_metrics_path,
        snapped_locations_path=tmp_path / "unused.geoparquet",
        external_evaluation_folder=None,
        configured_external_evaluation_folder=external_folder,
        model_folder=tmp_path,
        output_folder=output_folder,
        logger=logging.getLogger(__name__),
        minimum_upstream_area_km2=400.0,
        include_geb=True,
        matched_only=True,
    )

    assert prepared_geb_df["station_name"].to_list() == ["station_a"]
    assert external_models["reference"].index.to_list() == ["STATION_A"]


def test_prepare_skill_score_boxplot_inputs_can_skip_external_models(
    tmp_path: Path,
) -> None:
    """Plain skill-score plots can keep `evaluation_skill_scores.png` GEB-only."""
    evaluation_metrics_path: Path = tmp_path / "evaluation_metrics.xlsx"
    external_folder: Path = tmp_path / "external"
    external_folder.mkdir()
    output_folder: Path = tmp_path / "output"
    output_folder.mkdir()
    evaluation_df: pd.DataFrame = pd.DataFrame(
        {
            "station_name": ["station_a"],
            "upstream_area_GEB": [500_000_000.0],
            "KGE": [0.8],
        }
    )
    external_df: pd.DataFrame = pd.DataFrame(
        {"KGE": [0.6]},
        index=pd.Index(["station_a"]),
    )
    evaluation_df.to_excel(evaluation_metrics_path, index=False)
    external_df.to_csv(external_folder / "reference.csv")

    prepared_geb_df, external_models = prepare_skill_score_boxplot_inputs(
        evaluation_metrics_path=evaluation_metrics_path,
        snapped_locations_path=tmp_path / "unused.geoparquet",
        external_evaluation_folder=None,
        configured_external_evaluation_folder=external_folder,
        model_folder=tmp_path,
        output_folder=output_folder,
        logger=logging.getLogger(__name__),
        minimum_upstream_area_km2=400.0,
        include_geb=True,
        matched_only=False,
        include_external=False,
    )

    assert prepared_geb_df["station_name"].to_list() == ["station_a"]
    assert external_models == {}


def _write_google_metric_file(
    metric_folder: Path,
    filename: str,
    values: list[float],
) -> None:
    """Write a small Google-style per-metric CSV fixture."""
    metric_df: pd.DataFrame = pd.DataFrame(
        {"0": values, "1": [value + 0.1 for value in values]},
        index=pd.Index(["GRDC_1001", "GRDC_1002"]),
    )
    metric_df.to_csv(metric_folder / filename)


def _write_google_metrics_tree(root_folder: Path) -> Path:
    """Write a small Google-style extracted metrics tree.

    Args:
        root_folder: Folder receiving the `metrics/` tree.

    Returns:
        Folder containing the Google per-metric CSV files.
    """
    metric_folder: Path = (
        root_folder
        / "metrics"
        / "hydrograph_metrics"
        / "per_metric"
        / "google"
        / "2014"
        / "dual_lstm"
        / "hydrologically_separated"
    )
    metric_folder.mkdir(parents=True)
    _write_google_metric_file(metric_folder, "KGE.csv", [0.8, 0.7])
    _write_google_metric_file(metric_folder, "NSE.csv", [0.6, 0.5])
    _write_google_metric_file(metric_folder, "Pearson-r.csv", [0.9, 0.8])
    _write_google_metric_file(metric_folder, "RMSE.csv", [1.2, 1.3])
    return metric_folder


def test_read_google_streamflow_skill_scores_from_extracted_metrics(
    tmp_path: Path,
) -> None:
    """Google streamflow metrics are read from the extracted Zenodo metrics tree."""
    _write_google_metrics_tree(tmp_path)

    google_df: pd.DataFrame = read_google_streamflow_skill_scores(
        tmp_path,
        logging.getLogger(__name__),
    )

    assert google_df.index.to_list() == ["GRDC_1001", "GRDC_1002"]
    assert google_df.loc["GRDC_1001", "KGE"] == 0.8
    assert google_df.loc["GRDC_1001", "R"] == 0.9
    assert google_df.loc["GRDC_1001", "R2"] == pytest.approx(0.81)


def test_read_external_evaluation_raw_finds_nested_google_metrics(
    tmp_path: Path,
) -> None:
    """Google streamflow metrics are discovered inside nested external folders."""
    _write_google_metrics_tree(tmp_path / "global_streamflow_model_paper")

    external_models: dict[str, pd.DataFrame] = read_external_evaluation_raw(
        external_evaluation_folder=None,
        configured_external_evaluation_folder=tmp_path,
        model_folder=tmp_path,
        logger=logging.getLogger(__name__),
    )

    google_df: pd.DataFrame = external_models[GOOGLE_STREAMFLOW_MODEL_NAME]
    assert google_df.index.to_list() == ["GRDC_1001", "GRDC_1002"]
    assert google_df.loc["GRDC_1001", "KGE"] == 0.8


def test_read_google_streamflow_skill_scores_from_metrics_archive(
    tmp_path: Path,
) -> None:
    """Google streamflow metrics are read from the Zenodo metrics archive layout."""
    metrics_source_folder: Path = tmp_path / "source"
    _write_google_metrics_tree(metrics_source_folder)
    metrics_archive_path: Path = tmp_path / "metrics.tgz"
    with tarfile.open(metrics_archive_path, "w:gz") as archive:
        archive.add(metrics_source_folder / "metrics", arcname="metrics")

    google_df: pd.DataFrame = read_google_streamflow_skill_scores_from_archive(
        metrics_archive_path,
        logging.getLogger(__name__),
    )

    assert google_df.index.to_list() == ["GRDC_1001", "GRDC_1002"]
    assert google_df.loc["GRDC_1002", "NSE"] == 0.5


def test_prepare_skill_score_boxplot_inputs_matches_google_by_grdc_station_id(
    tmp_path: Path,
) -> None:
    """Google GRDC station IDs are matched to GEB station_ID values."""
    evaluation_metrics_path: Path = tmp_path / "evaluation_metrics.xlsx"
    _write_google_metrics_tree(tmp_path / "external")
    evaluation_df: pd.DataFrame = pd.DataFrame(
        {
            "station_ID": [1001, 9999],
            "station_name": ["Local Name", "Other Name"],
            "upstream_area_GEB": [500_000_000.0, 500_000_000.0],
            "KGE": [0.3, 0.2],
        }
    )
    evaluation_df.to_excel(evaluation_metrics_path, index=False)

    prepared_geb_df, external_models = prepare_skill_score_boxplot_inputs(
        evaluation_metrics_path=evaluation_metrics_path,
        snapped_locations_path=tmp_path / "unused.geoparquet",
        external_evaluation_folder=None,
        configured_external_evaluation_folder=tmp_path / "external",
        model_folder=tmp_path,
        output_folder=tmp_path,
        logger=logging.getLogger(__name__),
        minimum_upstream_area_km2=400.0,
        include_geb=True,
        matched_only=True,
    )

    assert prepared_geb_df["station_ID"].to_list() == [1001]
    assert external_models[GOOGLE_STREAMFLOW_MODEL_NAME].index.to_list() == [
        "GRDC_1001"
    ]


def test_prepare_skill_score_boxplot_inputs_matches_google_with_snapped_index(
    tmp_path: Path,
) -> None:
    """Google station IDs match when snapped locations store IDs in the index."""
    gpd = pytest.importorskip("geopandas")
    shapely_geometry = pytest.importorskip("shapely.geometry")
    _write_google_metrics_tree(tmp_path / "external")
    snapped_locations_path: Path = tmp_path / "snapped_locations.geoparquet"
    snapped_locations_gdf = gpd.GeoDataFrame(
        {
            "discharge_observations_station_name": ["Local Name"],
        },
        index=pd.Index([1001], name="discharge_observations_station_ID"),
        geometry=[shapely_geometry.Point(0.0, 0.0)],
        crs="EPSG:4326",
    )
    snapped_locations_gdf.to_parquet(snapped_locations_path)

    prepared_geb_df, external_models = prepare_skill_score_boxplot_inputs(
        evaluation_metrics_path=tmp_path / "missing_evaluation_metrics.xlsx",
        snapped_locations_path=snapped_locations_path,
        external_evaluation_folder=None,
        configured_external_evaluation_folder=tmp_path / "external",
        model_folder=tmp_path,
        output_folder=tmp_path,
        logger=logging.getLogger(__name__),
        minimum_upstream_area_km2=400.0,
        include_geb=False,
        matched_only=True,
    )

    assert prepared_geb_df.empty
    assert external_models[GOOGLE_STREAMFLOW_MODEL_NAME].index.to_list() == [
        "GRDC_1001"
    ]
