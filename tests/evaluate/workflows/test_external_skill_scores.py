"""Tests for external discharge skill-score processing."""

import logging
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from geb.evaluate.workflows.external_skill_scores import (
    GOOGLE_STREAMFLOW_MODEL_NAME,
    filter_utrecht_skill_scores,
    prepare_google_streamflow_skill_scores,
    prepare_skill_score_boxplot_inputs,
    read_external_evaluation_raw,
    read_google_streamflow_skill_scores,
    read_google_streamflow_skill_scores_from_archive,
)


def test_filter_utrecht_skill_scores_applies_grdc_station_criteria() -> None:
    """Utrecht skill scores are filtered to record length, end year, and area criteria."""
    external_evaluation_df: pd.DataFrame = pd.DataFrame(
        {
            "KGE": [0.8, 0.7, 0.6, 0.5],
            "end_year": [1991, 1990, 1995, 2000],
            "data_years": [10.0, 10.0, 9.9, 12.0],
            "catchment_area_km2": [400.0, 500.0, 800.0, 399.9],
        },
        index=pd.Index(["passes", "too_early", "too_short", "too_small"]),
    )

    filtered_evaluation_df: pd.DataFrame = filter_utrecht_skill_scores(
        external_evaluation_df,
        logging.getLogger(__name__),
    )

    assert filtered_evaluation_df.index.to_list() == ["passes"]


def test_filter_utrecht_skill_scores_converts_m2_area_column() -> None:
    """Utrecht catchment-area metadata in square meters is converted to km2."""
    external_evaluation_df: pd.DataFrame = pd.DataFrame(
        {
            "KGE": [0.8, 0.7],
            "end_date": ["1991-12-31", "1991-12-31"],
            "n_years": [10.0, 10.0],
            "catchment_area_m2": [400_000_000.0, 399_000_000.0],
        },
        index=pd.Index(["passes", "too_small"]),
    )

    filtered_evaluation_df: pd.DataFrame = filter_utrecht_skill_scores(
        external_evaluation_df,
        logging.getLogger(__name__),
    )

    assert filtered_evaluation_df.index.to_list() == ["passes"]


def test_filter_utrecht_skill_scores_keeps_score_only_tables(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Utrecht score-only tables are kept because metadata filters need metadata."""
    external_evaluation_df: pd.DataFrame = pd.DataFrame(
        {"KGE": [0.8]},
        index=pd.Index(["station"]),
    )

    with caplog.at_level(logging.WARNING):
        filtered_evaluation_df: pd.DataFrame = filter_utrecht_skill_scores(
            external_evaluation_df,
            logging.getLogger(__name__),
        )

    pd.testing.assert_frame_equal(filtered_evaluation_df, external_evaluation_df)
    assert filtered_evaluation_df is not external_evaluation_df
    assert "Skipping Utrecht GRDC station-criteria filter" in caplog.text


def test_read_external_evaluation_raw_filters_only_utrecht_sources(
    tmp_path: Path,
) -> None:
    """External CSV reading applies Utrecht filtering without affecting Google data."""
    utrecht_df: pd.DataFrame = pd.DataFrame(
        {
            "KGE": [0.8, 0.7],
            "end_year": [1991, 1990],
            "data_years": [10.0, 10.0],
            "catchment_area_km2": [400.0, 500.0],
        },
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

    assert external_models["utrecht"].index.to_list() == ["STATION_A"]
    assert external_models["google_streamflow"].index.to_list() == [
        "STATION_A",
        "STATION_B",
    ]


def test_prepare_google_streamflow_skill_scores_is_pass_through() -> None:
    """Google streamflow preparation currently keeps the external table unchanged."""
    external_evaluation_df: pd.DataFrame = pd.DataFrame(
        {"KGE": [0.6]},
        index=pd.Index(["station"]),
    )

    prepared_evaluation_df: pd.DataFrame = prepare_google_streamflow_skill_scores(
        external_evaluation_df
    )

    pd.testing.assert_frame_equal(prepared_evaluation_df, external_evaluation_df)
    assert prepared_evaluation_df is not external_evaluation_df


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
            "station_name": ["station_a", "station_b"],
            "upstream_area_GEB": [500_000_000.0, 200_000_000.0],
            "KGE": [0.8, 0.7],
        }
    )
    external_df: pd.DataFrame = pd.DataFrame(
        {"KGE": [0.6, 0.5]},
        index=pd.Index(["station_a", "station_b"]),
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
