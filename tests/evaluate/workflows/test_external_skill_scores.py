"""Tests for external discharge skill-score processing."""

import logging
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from geb.evaluate.workflows.external_skill_scores import (
    GLOFAS_CSV_NAME,
    GLOFAS_MODEL_NAME,
    GOOGLE_STREAMFLOW_ARCHIVE_NAME,
    GOOGLE_STREAMFLOW_CSV_NAME,
    GOOGLE_STREAMFLOW_METRICS_URL,
    GOOGLE_STREAMFLOW_MODEL_NAME,
    get_external_model_output_suffix,
    prepare_pairwise_skill_score_boxplot_inputs,
    prepare_skill_score_boxplot_inputs,
    read_external_evaluation_raw,
    read_google_streamflow_skill_scores,
    read_google_streamflow_skill_scores_from_archive,
)


def test_google_streamflow_metrics_url_points_to_zenodo_archive() -> None:
    """Google metrics URL points to the Zenodo metrics archive."""
    assert GOOGLE_STREAMFLOW_METRICS_URL.startswith(
        "https://zenodo.org/records/10397664/files/metrics.tgz"
    )


def test_read_external_evaluation_raw_reads_csv_scores(
    tmp_path: Path,
) -> None:
    """External CSV reading normalizes station IDs and metric columns."""
    utrecht_df: pd.DataFrame = pd.DataFrame(
        {"KGE": [0.8, 0.7], "R": [0.9, 0.8], "R2": [0.4, 0.3]},
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
        auto_fetch_google_streamflow=False,
    )

    assert external_models["utrecht"].index.to_list() == ["STATION_A", "STATION_B"]
    assert "R" not in external_models["utrecht"].columns
    assert external_models["utrecht"].loc["STATION_A", "KGE_correlation"] == 0.9
    assert external_models["utrecht"].loc["STATION_A", "NSE"] == pytest.approx(0.4)
    assert external_models["utrecht"].loc["STATION_A", "R2"] == pytest.approx(0.81)
    assert external_models[GOOGLE_STREAMFLOW_MODEL_NAME].index.to_list() == [
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

    plot_inputs = prepare_skill_score_boxplot_inputs(
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
        auto_fetch_google_streamflow=False,
    )

    assert plot_inputs.evaluation_df["station_name"].to_list() == ["station_a"]
    assert plot_inputs.external_models["reference"].index.to_list() == ["STATION_A"]
    assert "GEB upstream area >= 400 km2" in plot_inputs.filter_summary


def test_prepare_skill_score_boxplot_inputs_applies_utrecht_paper_threshold(
    tmp_path: Path,
) -> None:
    """Utrecht comparisons apply the configured 400 km2 paper threshold."""
    evaluation_metrics_path: Path = tmp_path / "evaluation_metrics.xlsx"
    external_folder: Path = tmp_path / "external"
    external_folder.mkdir()
    output_folder: Path = tmp_path / "output"
    output_folder.mkdir()
    evaluation_df: pd.DataFrame = pd.DataFrame(
        {
            "station_name": ["station_a", "station_b"],
            "upstream_area_GEB": [500_000_000.0, 300_000_000.0],
            "KGE": [0.8, 0.7],
        }
    )
    utrecht_df: pd.DataFrame = pd.DataFrame(
        {"KGE": [0.6, 0.5]},
        index=pd.Index(["station_a", "station_b"]),
    )
    evaluation_df.to_excel(evaluation_metrics_path, index=False)
    utrecht_df.to_csv(external_folder / "utrecht_1km.csv")

    plot_inputs = prepare_skill_score_boxplot_inputs(
        evaluation_metrics_path=evaluation_metrics_path,
        snapped_locations_path=tmp_path / "unused.geoparquet",
        external_evaluation_folder=None,
        configured_external_evaluation_folder=external_folder,
        model_folder=tmp_path,
        output_folder=output_folder,
        logger=logging.getLogger(__name__),
        minimum_upstream_area_km2=0.0,
        include_geb=True,
        matched_only=True,
        auto_fetch_google_streamflow=False,
    )

    assert plot_inputs.evaluation_df["station_name"].to_list() == ["station_a"]
    assert plot_inputs.external_models["utrecht_1km"].index.to_list() == ["STATION_A"]
    assert "GEB upstream area >= 0 km2" not in plot_inputs.filter_summary
    assert "utrecht_1km: upstream area >= 400 km2" in plot_inputs.filter_summary


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

    plot_inputs = prepare_skill_score_boxplot_inputs(
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
        auto_fetch_google_streamflow=False,
    )

    assert plot_inputs.evaluation_df["station_name"].to_list() == ["station_a"]
    assert plot_inputs.external_models == {}


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
    _write_google_metric_file(metric_folder, "Beta-KGE.csv", [1.1, 1.2])
    _write_google_metric_file(metric_folder, "RMSE.csv", [1.2, 1.3])
    return metric_folder


def _write_glofas_metrics_tree(root_folder: Path) -> Path:
    """Write a small GloFAS-style extracted metrics tree.

    Args:
        root_folder: Folder receiving the `metrics/` tree.

    Returns:
        Folder containing the GloFAS per-metric CSV files.
    """
    metric_folder: Path = (
        root_folder
        / "metrics"
        / "hydrograph_metrics"
        / "per_metric"
        / "glofas"
        / "2014"
        / "glofas_prediction"
    )
    metric_folder.mkdir(parents=True)
    _write_google_metric_file(metric_folder, "KGE.csv", [0.4, 0.3])
    _write_google_metric_file(metric_folder, "NSE.csv", [0.2, 0.1])
    _write_google_metric_file(metric_folder, "Pearson-r.csv", [0.7, 0.6])
    _write_google_metric_file(metric_folder, "Beta-KGE.csv", [0.9, 0.8])
    _write_google_metric_file(metric_folder, "RMSE.csv", [2.2, 2.3])
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
    assert "R" not in google_df.columns
    assert google_df.loc["GRDC_1001", "KGE_correlation"] == 0.9
    assert google_df.loc["GRDC_1001", "R2"] == pytest.approx(0.81)
    assert google_df.loc["GRDC_1001", "KGE_bias_ratio"] == pytest.approx(1.1)


def test_read_external_evaluation_raw_finds_nested_google_metrics(
    tmp_path: Path,
) -> None:
    """Google streamflow metrics are discovered inside nested external folders."""
    _write_google_metrics_tree(tmp_path / "global_streamflow_model_paper")
    _write_glofas_metrics_tree(tmp_path / "global_streamflow_model_paper")

    external_models: dict[str, pd.DataFrame] = read_external_evaluation_raw(
        external_evaluation_folder=None,
        configured_external_evaluation_folder=tmp_path,
        model_folder=tmp_path,
        logger=logging.getLogger(__name__),
    )

    google_df: pd.DataFrame = external_models[GOOGLE_STREAMFLOW_MODEL_NAME]
    assert google_df.index.to_list() == ["GRDC_1001", "GRDC_1002"]
    assert google_df.loc["GRDC_1001", "KGE"] == 0.8
    glofas_df: pd.DataFrame = external_models[GLOFAS_MODEL_NAME]
    assert glofas_df.index.to_list() == ["GRDC_1001", "GRDC_1002"]
    assert glofas_df.loc["GRDC_1001", "KGE"] == 0.4


def test_read_google_streamflow_skill_scores_from_metrics_archive(
    tmp_path: Path,
) -> None:
    """Google streamflow metrics are read from the Zenodo metrics archive layout."""
    metrics_source_folder: Path = tmp_path / "source"
    _write_google_metrics_tree(metrics_source_folder)
    _write_glofas_metrics_tree(metrics_source_folder)
    metrics_archive_path: Path = tmp_path / "metrics.tgz"
    with tarfile.open(metrics_archive_path, "w:gz") as archive:
        archive.add(metrics_source_folder / "metrics", arcname="metrics")

    google_df: pd.DataFrame = read_google_streamflow_skill_scores_from_archive(
        metrics_archive_path,
        logging.getLogger(__name__),
    )

    assert google_df.index.to_list() == ["GRDC_1001", "GRDC_1002"]
    assert google_df.loc["GRDC_1002", "NSE"] == 0.5


def test_read_external_evaluation_raw_fetches_google_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Google metrics are fetched, archived, and normalized into external data."""
    metrics_source_folder: Path = tmp_path / "source"
    _write_google_metrics_tree(metrics_source_folder)
    _write_glofas_metrics_tree(metrics_source_folder)
    source_archive_path: Path = tmp_path / "source_metrics.tgz"
    with tarfile.open(source_archive_path, "w:gz") as archive:
        archive.add(metrics_source_folder / "metrics", arcname="metrics")

    def fake_fetch_google_streamflow_metrics_archive(
        output_path: Path,
        logger: logging.Logger,
    ) -> Path:
        """Copy a fixture archive instead of downloading from Zenodo.

        Args:
            output_path: Archive path to create.
            logger: Unused logger matching the production fetch signature.

        Returns:
            Created archive path.
        """
        output_path.write_bytes(source_archive_path.read_bytes())
        return output_path

    monkeypatch.setattr(
        "geb.evaluate.workflows.external_skill_scores.fetch_google_streamflow_metrics_archive",
        fake_fetch_google_streamflow_metrics_archive,
    )
    external_folder: Path = tmp_path / "external"

    external_models: dict[str, pd.DataFrame] = read_external_evaluation_raw(
        external_evaluation_folder=None,
        configured_external_evaluation_folder=external_folder,
        model_folder=tmp_path,
        logger=logging.getLogger(__name__),
    )

    google_df: pd.DataFrame = external_models[GOOGLE_STREAMFLOW_MODEL_NAME]
    assert len(google_df) == 2
    glofas_df: pd.DataFrame = external_models[GLOFAS_MODEL_NAME]
    assert len(glofas_df) == 2
    assert (external_folder / GOOGLE_STREAMFLOW_ARCHIVE_NAME).exists()
    assert (external_folder / GOOGLE_STREAMFLOW_CSV_NAME).exists()
    assert (external_folder / GLOFAS_CSV_NAME).exists()


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

    plot_inputs = prepare_skill_score_boxplot_inputs(
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
        auto_fetch_google_streamflow=False,
    )

    assert plot_inputs.evaluation_df["station_ID"].to_list() == [1001]
    assert plot_inputs.external_models[
        GOOGLE_STREAMFLOW_MODEL_NAME
    ].index.to_list() == ["GRDC_1001"]
    assert "GEB upstream area >= 400 km2" in plot_inputs.filter_summary
    assert "Google Streamflow: upstream area >=" not in plot_inputs.filter_summary


def test_prepare_pairwise_skill_score_boxplot_inputs_keeps_model_specific_matches(
    tmp_path: Path,
) -> None:
    """Pairwise matched plots use each external model's own station overlap."""
    evaluation_metrics_path: Path = tmp_path / "evaluation_metrics.xlsx"
    external_folder: Path = tmp_path / "external"
    external_folder.mkdir()
    evaluation_df: pd.DataFrame = pd.DataFrame(
        {
            "station_ID": [1001, 1002, 1003],
            "station_name": ["utrecht_a", "google_b", "shared_c"],
            "upstream_area_GEB": [450_000_000.0, 550_000_000.0, 700_000_000.0],
            "KGE": [0.8, 0.7, 0.6],
        }
    )
    utrecht_df: pd.DataFrame = pd.DataFrame(
        {"KGE": [0.2, 0.3]},
        index=pd.Index(["utrecht_a", "shared_c"]),
    )
    google_df: pd.DataFrame = pd.DataFrame(
        {"KGE": [0.4, 0.5]},
        index=pd.Index(["GRDC_1002", "GRDC_1003"]),
    )
    evaluation_df.to_excel(evaluation_metrics_path, index=False)
    utrecht_df.to_csv(external_folder / "utrecht_1km.csv")
    google_df.to_csv(external_folder / GOOGLE_STREAMFLOW_CSV_NAME)

    pairwise_inputs = prepare_pairwise_skill_score_boxplot_inputs(
        evaluation_metrics_path=evaluation_metrics_path,
        external_evaluation_folder=None,
        configured_external_evaluation_folder=external_folder,
        model_folder=tmp_path,
        output_folder=tmp_path,
        logger=logging.getLogger(__name__),
        minimum_upstream_area_km2=0.0,
        auto_fetch_google_streamflow=False,
    )

    assert pairwise_inputs["utrecht_1km"].evaluation_df["station_ID"].to_list() == [
        1001,
        1003,
    ]
    assert pairwise_inputs[GOOGLE_STREAMFLOW_MODEL_NAME].evaluation_df[
        "station_ID"
    ].to_list() == [1002, 1003]
    assert pairwise_inputs["utrecht_1km"].external_models[
        "utrecht_1km"
    ].index.to_list() == ["UTRECHT_A", "SHARED_C"]
    assert pairwise_inputs["utrecht_1km"].evaluation_df["KGE_difference"].to_list() == [
        pytest.approx(0.6),
        pytest.approx(0.3),
    ]
    assert pairwise_inputs[GOOGLE_STREAMFLOW_MODEL_NAME].external_models[
        GOOGLE_STREAMFLOW_MODEL_NAME
    ].index.to_list() == ["GRDC_1002", "GRDC_1003"]
    assert pairwise_inputs[GOOGLE_STREAMFLOW_MODEL_NAME].evaluation_df[
        "KGE_difference"
    ].to_list() == [pytest.approx(0.3), pytest.approx(0.1)]
    assert (
        pairwise_inputs[GOOGLE_STREAMFLOW_MODEL_NAME].minimum_upstream_area_km2 == 0.0
    )
    assert (
        "upstream area >="
        not in pairwise_inputs[GOOGLE_STREAMFLOW_MODEL_NAME].filter_summary
    )
    assert get_external_model_output_suffix("Google Streamflow") == (
        "_matched_google_streamflow"
    )


def test_prepare_pairwise_skill_score_boxplot_inputs_applies_glofas_threshold(
    tmp_path: Path,
) -> None:
    """GloFAS pairwise plots apply the 500 km2 paper threshold."""
    evaluation_metrics_path: Path = tmp_path / "evaluation_metrics.xlsx"
    external_folder: Path = tmp_path / "external"
    external_folder.mkdir()
    evaluation_df: pd.DataFrame = pd.DataFrame(
        {
            "station_ID": [1001, 1002],
            "station_name": ["small_basin", "large_basin"],
            "upstream_area_GEB": [450_000_000.0, 550_000_000.0],
            "KGE": [0.8, 0.7],
        }
    )
    glofas_df: pd.DataFrame = pd.DataFrame(
        {"KGE": [0.2, 0.3]},
        index=pd.Index(["GRDC_1001", "GRDC_1002"]),
    )
    evaluation_df.to_excel(evaluation_metrics_path, index=False)
    glofas_df.to_csv(external_folder / GLOFAS_CSV_NAME)

    pairwise_inputs = prepare_pairwise_skill_score_boxplot_inputs(
        evaluation_metrics_path=evaluation_metrics_path,
        external_evaluation_folder=None,
        configured_external_evaluation_folder=external_folder,
        model_folder=tmp_path,
        output_folder=tmp_path,
        logger=logging.getLogger(__name__),
        minimum_upstream_area_km2=0.0,
        auto_fetch_google_streamflow=False,
    )

    glofas_inputs = pairwise_inputs[GLOFAS_MODEL_NAME]
    assert glofas_inputs.minimum_upstream_area_km2 == 500.0
    assert glofas_inputs.evaluation_df["station_ID"].to_list() == [1002]
    assert glofas_inputs.external_models[GLOFAS_MODEL_NAME].index.to_list() == [
        "GRDC_1002"
    ]
    assert "GEB upstream area >= 500 km2" in glofas_inputs.filter_summary


def test_prepare_pairwise_inputs_skip_missing_archive_metrics(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing archive metrics skip Google while retaining other comparisons."""
    evaluation_metrics_path: Path = tmp_path / "evaluation_metrics.xlsx"
    external_folder: Path = tmp_path / "external"
    external_folder.mkdir()
    evaluation_df: pd.DataFrame = pd.DataFrame(
        {
            "station_ID": [1001],
            "station_name": ["utrecht_a"],
            "upstream_area_GEB": [500_000_000.0],
            "KGE": [0.8],
        }
    )
    evaluation_df.to_excel(evaluation_metrics_path, index=False)
    pd.DataFrame(
        {"KGE": [0.2]},
        index=pd.Index(["utrecht_a"]),
    ).to_csv(external_folder / "utrecht_1km.csv")
    pd.DataFrame(
        {"KGE": [0.4]},
        index=pd.Index(["GRDC_1001"]),
    ).to_csv(external_folder / GOOGLE_STREAMFLOW_CSV_NAME)

    with caplog.at_level(logging.WARNING):
        pairwise_inputs = prepare_pairwise_skill_score_boxplot_inputs(
            evaluation_metrics_path=evaluation_metrics_path,
            external_evaluation_folder=None,
            configured_external_evaluation_folder=external_folder,
            model_folder=tmp_path,
            output_folder=tmp_path,
            logger=logging.getLogger(__name__),
            minimum_upstream_area_km2=0.0,
            auto_fetch_google_streamflow=False,
            archive_evaluation_metrics_path=tmp_path
            / "evaluation_metrics_2014_2021.xlsx",
        )

    assert list(pairwise_inputs) == ["utrecht_1km"]
    assert "Skipping archive comparisons" in caplog.text


def test_prepare_pairwise_inputs_return_empty_when_only_archive_metrics_missing(
    tmp_path: Path,
) -> None:
    """Missing archive metrics yield no pairwise inputs for archive-only data."""
    external_folder: Path = tmp_path / "external"
    external_folder.mkdir()
    pd.DataFrame(
        {"KGE": [0.4]},
        index=pd.Index(["GRDC_1001"]),
    ).to_csv(external_folder / GOOGLE_STREAMFLOW_CSV_NAME)

    pairwise_inputs = prepare_pairwise_skill_score_boxplot_inputs(
        evaluation_metrics_path=tmp_path / "evaluation_metrics.xlsx",
        external_evaluation_folder=None,
        configured_external_evaluation_folder=external_folder,
        model_folder=tmp_path,
        output_folder=tmp_path,
        logger=logging.getLogger(__name__),
        minimum_upstream_area_km2=0.0,
        auto_fetch_google_streamflow=False,
        archive_evaluation_metrics_path=tmp_path / "evaluation_metrics_2014_2021.xlsx",
    )

    assert pairwise_inputs == {}


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

    plot_inputs = prepare_skill_score_boxplot_inputs(
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
        auto_fetch_google_streamflow=False,
    )

    assert plot_inputs.evaluation_df.empty
    assert plot_inputs.external_models[
        GOOGLE_STREAMFLOW_MODEL_NAME
    ].index.to_list() == ["GRDC_1001"]


def test_prepare_skill_score_boxplot_inputs_uses_snapped_index_when_metrics_empty(
    tmp_path: Path,
) -> None:
    """Empty evaluation metrics fall back to snapped station IDs for matching."""
    gpd = pytest.importorskip("geopandas")
    shapely_geometry = pytest.importorskip("shapely.geometry")
    _write_google_metrics_tree(tmp_path / "external")
    evaluation_metrics_path: Path = tmp_path / "evaluation_metrics.xlsx"
    empty_evaluation_df: pd.DataFrame = pd.DataFrame(
        {
            "station_ID": pd.Series(dtype="int64"),
            "station_name": pd.Series(dtype="str"),
        }
    )
    empty_evaluation_df.to_excel(evaluation_metrics_path, index=False)
    snapped_locations_path: Path = tmp_path / "snapped_locations.geoparquet"
    snapped_locations_gdf = gpd.GeoDataFrame(
        {"discharge_observations_station_name": ["Local Name"]},
        index=pd.Index([1001], name="discharge_observations_station_ID"),
        geometry=[shapely_geometry.Point(0.0, 0.0)],
        crs="EPSG:4326",
    )
    snapped_locations_gdf.to_parquet(snapped_locations_path)

    plot_inputs = prepare_skill_score_boxplot_inputs(
        evaluation_metrics_path=evaluation_metrics_path,
        snapped_locations_path=snapped_locations_path,
        external_evaluation_folder=None,
        configured_external_evaluation_folder=tmp_path / "external",
        model_folder=tmp_path,
        output_folder=tmp_path,
        logger=logging.getLogger(__name__),
        minimum_upstream_area_km2=400.0,
        include_geb=False,
        matched_only=True,
        auto_fetch_google_streamflow=False,
    )

    assert plot_inputs.external_models[
        GOOGLE_STREAMFLOW_MODEL_NAME
    ].index.to_list() == ["GRDC_1001"]
