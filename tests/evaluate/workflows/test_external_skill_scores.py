"""Tests for external discharge skill-score processing."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from geb.evaluate.workflows.external_skill_scores import (
    filter_utrecht_skill_scores,
    prepare_google_streamflow_skill_scores,
    prepare_skill_score_boxplot_inputs,
    read_external_evaluation_raw,
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


def test_filter_utrecht_skill_scores_requires_metadata_columns() -> None:
    """Utrecht filtering fails clearly when the published criteria cannot be applied."""
    external_evaluation_df: pd.DataFrame = pd.DataFrame(
        {"KGE": [0.8]},
        index=pd.Index(["station"]),
    )

    with pytest.raises(ValueError, match="Utrecht external skill-score filtering"):
        filter_utrecht_skill_scores(
            external_evaluation_df,
            logging.getLogger(__name__),
        )


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
