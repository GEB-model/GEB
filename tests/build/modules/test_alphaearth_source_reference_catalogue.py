"""Offline tests for alphaearth_source_reference_catalogue.py."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np


WORKFLOW_SCRIPT_FILENAME = "alphaearth_source_reference_catalogue.py"
WORKFLOW_SCRIPT_ENV = "GEB_ALPHAEARTH_SOURCE_REFERENCE_SCRIPT"
GEB_SUPPORT_ROOT_ENV = "GEB_SUPPORT_ROOT"


def _find_repository_root(start: Path) -> Path | None:
    """Return the nearest parent containing the GEB package and pyproject.toml."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "geb").is_dir():
            return candidate
    return None


def _find_workflow_script_path() -> Path:
    """Locate the standalone workflow in local packages or GEB_support/main.

    The production test lives under ``GEB/tests/build/modules`` while the
    standalone workflow lives in the sibling ``GEB_support/main`` checkout.
    Keeping the lookup here avoids a machine-specific absolute path and still
    lets developers override the location explicitly.
    """
    test_file = Path(__file__).resolve()
    repository_root = _find_repository_root(test_file.parent)

    candidates: list[Path] = []
    explicit_script = os.environ.get(WORKFLOW_SCRIPT_ENV)
    if explicit_script:
        candidates.append(Path(explicit_script).expanduser())

    support_root = os.environ.get(GEB_SUPPORT_ROOT_ENV)
    if support_root:
        candidates.append(
            Path(support_root).expanduser() / "main" / WORKFLOW_SCRIPT_FILENAME
        )

    # This supports running the test directly from an extracted development ZIP.
    candidates.append(test_file.with_name(WORKFLOW_SCRIPT_FILENAME))

    if repository_root is not None:
        candidates.extend(
            [
                repository_root.parent
                / "GEB_support"
                / "main"
                / WORKFLOW_SCRIPT_FILENAME,
                repository_root / "support" / "main" / WORKFLOW_SCRIPT_FILENAME,
            ]
        )

    checked: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if resolved.is_file():
            return resolved

    checked_text = "\n  - ".join(str(candidate) for candidate in checked)
    raise FileNotFoundError(
        f"Could not locate {WORKFLOW_SCRIPT_FILENAME}. Checked:\n  - "
        f"{checked_text}\nSet {WORKFLOW_SCRIPT_ENV} to the full script path "
        f"or {GEB_SUPPORT_ROOT_ENV} to the GEB_support directory."
    )


def _install_geb_stubs() -> None:
    geb = types.ModuleType("geb")
    build = types.ModuleType("geb.build")
    data_catalog = types.ModuleType("geb.build.data_catalog")
    alphaearth = types.ModuleType("geb.build.data_catalog.alphaearth")
    custom_models = types.ModuleType("geb.build.custom_models")
    europe = types.ModuleType("geb.build.custom_models.europe")
    workflows = types.ModuleType("geb.build.workflows")
    farmers = types.ModuleType("geb.build.workflows.farmers")
    runner = types.ModuleType("geb.runner")
    geb_workflows = types.ModuleType("geb.workflows")
    io = types.ModuleType("geb.workflows.io")

    class DataCatalog:
        def __init__(self, *args, **kwargs):
            pass

        def fetch(self, *_args, **_kwargs):
            raise AssertionError(
                "DataCatalog.fetch should not be called by offline tests"
            )

    data_catalog.DataCatalog = DataCatalog
    alphaearth.AVAILABLE_YEARS = tuple(range(2017, 2026))
    europe._active_subgrid_mask_geometry_for_hrl = lambda *args, **kwargs: None
    farmers.ALPHAEARTH_EMBEDDING_BANDS = tuple(f"A{i:02d}" for i in range(64))
    farmers.parse_europe_model_ids = lambda values: tuple(
        int(value) for value in values
    )
    farmers.sample_alphaearth_embeddings = lambda *args, **kwargs: None
    runner.create_logger = lambda *args, **kwargs: None
    runner.get_builder = lambda *args, **kwargs: None
    runner.parse_config = lambda *args, **kwargs: {}

    class WorkingDirectory:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    io.WorkingDirectory = WorkingDirectory

    sys.modules.update(
        {
            "geb": geb,
            "geb.build": build,
            "geb.build.data_catalog": data_catalog,
            "geb.build.data_catalog.alphaearth": alphaearth,
            "geb.build.custom_models": custom_models,
            "geb.build.custom_models.europe": europe,
            "geb.build.workflows": workflows,
            "geb.build.workflows.farmers": farmers,
            "geb.runner": runner,
            "geb.workflows": geb_workflows,
            "geb.workflows.io": io,
        }
    )


def load_workflow():
    _install_geb_stubs()
    specification = importlib.util.spec_from_file_location(
        "alphaearth_source_reference_catalogue_test_module",
        _find_workflow_script_path(),
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_exact_hrl_training_legend() -> None:
    workflow = load_workflow()
    expected = {
        1110: "Wheat",
        1120: "Barley",
        1130: "Maize",
        1140: "Rice",
        1150: "Other cereals",
        1210: "Fresh Vegetables",
        1220: "Dry pulses",
        1310: "Potatoes",
        1320: "Sugar Beet",
        1410: "Sunflower",
        1420: "Soybeans",
        1430: "Rapeseed",
        1440: "Flax, cotton and hemp",
        2100: "Grapes",
        2200: "Olives",
        2310: "Fruits",
        2320: "Nuts",
    }
    observed = {
        item.code: item.name
        for item in workflow.HRL_CTY_CLASSES
        if item.is_source_training_class
    }
    assert observed == expected


def test_worldcereal_ewoc_crosswalk() -> None:
    workflow = load_workflow()
    examples = {
        1101010011: 1110,
        1101020002: 1120,
        1101060001: 1130,
        1101080000: 1140,
        1101030011: 1150,
        1103010010: 1210,
        1105010040: 1220,
        1107000012: 1310,
        1107000031: 1320,
        1106000010: 1410,
        1106000020: 1420,
        1106000031: 1430,
        1108020020: 1440,
        1201000010: 2100,
        1203000010: 2200,
        1201010020: 2310,
        1201040040: 2320,
    }
    for source_code, target_code in examples.items():
        mapped = workflow.map_source_label_to_hrl(
            source_label_system="ewoc",
            source_label_code=source_code,
            source_label_name="",
        )
        assert mapped["hrl_cty_code"] == target_code
        assert mapped["training_eligible"] is True


def test_hcat_name_crosswalk_and_ambiguous_exclusion() -> None:
    workflow = load_workflow()
    examples = {
        "winter_common_soft_wheat": 1110,
        "spring_barley": 1120,
        "maize_corn_popcorn": 1130,
        "rice": 1140,
        "winter_triticale": 1150,
        "fresh_vegetables": 1210,
        "lentils": 1220,
        "potatoes": 1310,
        "sugar_beet": 1320,
        "sunflower": 1410,
        "soy_soybeans": 1420,
        "winter_rapeseed_rape": 1430,
        "hemp_cannabis": 1440,
        "vineyards_wine_vine_rebland_grapes": 2100,
        "olive": 2200,
        "apples": 2310,
        "hazelnuts_hazel": 2320,
    }
    for source_name, target_code in examples.items():
        mapped = workflow.map_source_label_to_hrl(
            source_label_system="hcat4",
            source_label_code="",
            source_label_name=source_name,
        )
        assert mapped["hrl_cty_code"] == target_code

    excluded = workflow.map_source_label_to_hrl(
        source_label_system="hcat4",
        source_label_code="3310000000",
        source_label_name="arable_crops",
    )
    assert excluded["mapping_status"] == "excluded"
    assert excluded["training_eligible"] is False


def test_split_is_block_consistent_and_uses_lucas_2022_as_test() -> None:
    workflow = load_workflow()
    data = gpd.GeoDataFrame(
        {
            "reference_source": [
                "worldcereal_rdm",
                "eurocrops_v2",
                "eurocrops_v2",
                "eurocrops_v2",
                "eurocrops_v2",
                "eurocrops_v2",
            ],
            "source_dataset_id": [
                "2022_EU_LUCAS_POINT_110",
                "nl_2021.parquet",
                "nl_2021.parquet",
                "de4_2021.parquet",
                "de4_2021.parquet",
                "fr_2021.parquet",
            ],
            "source_collection_id": [
                "2022_EU_LUCAS_POINT_110",
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
            ],
            "observation_year": [2022, 2021, 2021, 2021, 2021, 2021],
            "training_eligible": [True] * 6,
            "mapping_status": ["mapped"] * 6,
            "hrl_cty_code": [1110, 1120, 1130, 1140, 1150, 1210],
            "hrl_cty_name": [
                "Wheat",
                "Barley",
                "Maize",
                "Rice",
                "Other cereals",
                "Fresh Vegetables",
            ],
        },
        geometry=[
            Point(10_000, 10_000),
            Point(20_000, 20_000),  # Same 50 km block as LUCAS -> test.
            Point(60_000, 10_000),
            Point(110_000, 10_000),
            Point(160_000, 10_000),
            Point(210_000, 10_000),
        ],
        crs=3035,
    )
    split = workflow._assign_spatial_blocks(data)
    candidates = workflow._independent_test_candidate(split)
    test_blocks = set(split.loc[candidates, "spatial_block_id"])
    assert len(test_blocks) == 1
    test_mask = split["spatial_block_id"].isin(test_blocks)
    assert test_mask.iloc[0]
    assert test_mask.iloc[1]

    remaining = split.loc[~test_mask, "spatial_block_id"].unique()
    validation = workflow._validation_blocks(remaining)
    assert validation
    assert validation.isdisjoint(test_blocks)


def _minimal_training_catalogue() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "reference_id": ["euro::1", "wc::1"],
            "reference_source": ["eurocrops_v2", "worldcereal_rdm"],
            "source_dataset_id": ["cz_2021.parquet", "2022_eu_lucas_poly_111"],
            "source_collection_id": [pd.NA, "2022_eu_lucas_poly_111"],
            "source_feature_id": ["1", "2"],
            "observation_year": [2021, 2022],
            "hrl_cty_code": [1110, 1120],
            "hrl_cty_name": ["Wheat", "Barley"],
            "training_eligible": [True, True],
            "split": ["train", "test"],
            "split_group_id": ["0_0", "1_0"],
            "country_iso3": ["CZE", "CZE"],
            "local_region_id": [1, 1],
            "europe_model_id": [3, 3],
            "europe_model_name": ["Europe_003", "Europe_003"],
        },
        geometry=[
            Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
            Polygon([(200, 0), (300, 0), (300, 100), (200, 100)]),
        ],
        crs=3035,
    )


def test_reference_sampling_points_are_source_specific_and_deterministic() -> None:
    workflow = load_workflow()
    catalogue = _minimal_training_catalogue()
    first = workflow.build_reference_sampling_points(catalogue)
    second = workflow.build_reference_sampling_points(catalogue)

    counts = first.groupby("reference_id").size().to_dict()
    assert counts == {
        "euro::1": workflow.EUROCROPS_POINTS_PER_PARCEL,
        "wc::1": workflow.WORLDCEREAL_POLYGON_POINTS_PER_FEATURE,
    }
    assert first["predictor_point_id"].tolist() == second["predictor_point_id"].tolist()
    assert np.allclose(first["longitude"], second["longitude"])
    assert np.allclose(first["latitude"], second["latitude"])
    assert set(
        first.loc[first["reference_id"].eq("euro::1"), "predictor_aggregation"]
    ) == {"per_point"}
    assert set(
        first.loc[first["reference_id"].eq("wc::1"), "predictor_aggregation"]
    ) == {"median"}


def test_reference_predictor_aggregation_preserves_feature_weight() -> None:
    workflow = load_workflow()
    embedding_columns = [
        f"alphaearth_{band}" for band in workflow.ALPHAEARTH_EMBEDDING_BANDS
    ]
    rows = []
    for reference_id, source, aggregation, count, base in (
        ("euro::1", "eurocrops_v2", "per_point", 2, 1.0),
        ("wc::1", "worldcereal_rdm", "median", 3, 10.0),
    ):
        for point_index in range(count):
            row = {
                "reference_id": reference_id,
                "reference_source": source,
                "observation_year": 2022,
                "split": "train" if source == "eurocrops_v2" else "test",
                "hrl_cty_code": 1110,
                "predictor_point_id": f"{reference_id}::p{point_index:02d}",
                "point_index": point_index,
                "planned_point_count": count,
                "predictor_aggregation": aggregation,
                "longitude": 5.0 + point_index,
                "latitude": 52.0,
                "embedding_valid": True,
                "embedding_l2_norm": 1.0,
                "geometry": Point(5.0 + point_index, 52.0),
            }
            row.update({column: base + point_index for column in embedding_columns})
            rows.append(row)
    points = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    samples, coverage = workflow._aggregate_reference_predictors(points)

    euro = samples.loc[samples["reference_id"].eq("euro::1")]
    worldcereal = samples.loc[samples["reference_id"].eq("wc::1")]
    assert len(euro) == 2
    assert np.isclose(euro["sample_weight"].sum(), 1.0)
    assert len(worldcereal) == 1
    assert np.isclose(worldcereal["sample_weight"].iloc[0], 1.0)
    assert np.isclose(worldcereal[embedding_columns[0]].iloc[0], 11.0)
    assert coverage["reference_has_predictors"].all()


def test_all_stage_expansion_includes_predictor_stages() -> None:
    workflow = load_workflow()
    assert workflow.resolve_workflow_stages(["all"]) == (
        "catalogue",
        "harmonize",
        "split",
        "tile_manifest",
        "extract_predictors",
    )


def test_tile_manifest_and_predictor_stages_with_fake_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = load_workflow()
    output_dir = tmp_path / workflow.OUTPUT_DIRECTORY
    output_dir.mkdir(parents=True)
    catalogue = _minimal_training_catalogue()
    memory_tables: dict[Path, object] = {}

    def store_geoparquet(data, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        memory_tables[path] = data.copy()

    def store_parquet(data, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        memory_tables[path] = data.copy()

    training_path = output_dir / workflow.TRAINING_CATALOGUE_FILENAME
    store_geoparquet(catalogue, training_path)
    monkeypatch.setattr(workflow, "atomic_write_geoparquet", store_geoparquet)
    monkeypatch.setattr(workflow, "atomic_write_parquet", store_parquet)
    monkeypatch.setattr(
        workflow.gpd,
        "read_parquet",
        lambda path: memory_tables[Path(path)].copy(),
    )
    monkeypatch.setattr(workflow, "file_sha256", lambda _path: "test-sha256")
    monkeypatch.setattr(workflow, "write_table", lambda *_args, **_kwargs: None)

    fake_cog_path = tmp_path / "alphaearth" / "fake.tif"

    class FakeAdapter:
        max_parallel_downloads = 1

        def read_geometry(self, *, years, geometry, dry_run, **_kwargs):
            year = int(tuple(years)[0])
            if not dry_run:
                fake_cog_path.parent.mkdir(parents=True, exist_ok=True)
                fake_cog_path.touch()
            return gpd.GeoDataFrame(
                {
                    "year": [year],
                    "remote_url": [f"https://example.test/{year}/fake.tif"],
                    "local_path": [str(fake_cog_path)],
                },
                geometry=gpd.GeoSeries(
                    [Polygon([(-40, 0), (50, 0), (50, 85), (-40, 85)])],
                    name="geom",
                    crs=4326,
                ),
                crs=4326,
            )

    monkeypatch.setattr(
        workflow,
        "_configure_standalone_alphaearth_adapter",
        lambda: FakeAdapter(),
    )
    monkeypatch.setattr(
        workflow,
        "sample_alphaearth_embeddings",
        lambda _tiles, longitude, latitude: np.tile(
            np.arange(64, dtype=np.float32), (len(longitude), 1)
        ),
    )

    tile_path = workflow.setup_build_alphaearth_tile_manifest(
        tmp_path,
        overwrite=True,
    )
    assert tile_path.exists()
    points = memory_tables[output_dir / workflow.REFERENCE_SAMPLING_POINTS_FILENAME]
    assert len(points) == (
        workflow.EUROCROPS_POINTS_PER_PARCEL
        + workflow.WORLDCEREAL_POLYGON_POINTS_PER_FEATURE
    )

    sample_path = workflow.setup_extract_alphaearth_predictors(
        tmp_path,
        overwrite=True,
    )
    samples = memory_tables[sample_path]
    euro = samples.loc[samples["reference_id"].eq("euro::1")]
    worldcereal = samples.loc[samples["reference_id"].eq("wc::1")]
    assert len(euro) == workflow.EUROCROPS_POINTS_PER_PARCEL
    assert np.isclose(euro["sample_weight"].sum(), 1.0)
    assert len(worldcereal) == 1
    assert (
        len([column for column in samples if column.startswith("alphaearth_A")]) == 64
    )
