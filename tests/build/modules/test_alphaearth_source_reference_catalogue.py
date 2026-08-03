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
import pytest


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
    farmers.alphaearth_crop_feature_importance = lambda *_args, **_kwargs: (
        pd.DataFrame()
    )
    farmers.alphaearth_embedding_diagnostics = lambda *_args, **_kwargs: {}
    farmers.calibrate_alphaearth_class_thresholds = lambda *_args, **_kwargs: {}
    farmers.evaluate_alphaearth_crop_models = lambda *_args, **_kwargs: (
        pd.DataFrame(),
        pd.DataFrame(),
    )
    farmers.fit_alphaearth_crop_models = lambda *_args, **_kwargs: None
    farmers.format_alphaearth_accuracy_report = lambda *_args, **_kwargs: ""
    farmers.parse_europe_model_ids = lambda values: tuple(
        int(value) for value in values
    )
    farmers.sample_alphaearth_embeddings = lambda *args, **kwargs: None
    farmers.save_alphaearth_crop_models = lambda _model, path: Path(path)
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


def load_comparison():
    workflow = load_workflow()
    sys.modules["alphaearth_source_reference_catalogue"] = workflow
    comparison_path = _find_workflow_script_path().with_name(
        "compare_alphaearth_source_reference_models.py"
    )
    specification = importlib.util.spec_from_file_location(
        "compare_alphaearth_source_reference_models_test_module",
        comparison_path,
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


def test_geb_crop_parameter_taxonomy_and_crosswalk() -> None:
    workflow = load_workflow()
    assert {item.crop_id: item.crop_name for item in workflow.GEB_CROP_CLASSES} == {
        0: "wheat",
        1: "maize",
        2: "rice",
        3: "barley",
        4: "rye",
        5: "millet",
        6: "sorghum",
        7: "soybeans",
        8: "sunflower",
        9: "potatoes",
        10: "cassava",
        11: "sugar cane",
        12: "sugar beets",
        13: "oil palm",
        14: "rapeseed",
        15: "groundnuts",
        16: "pulses",
        17: "citrus",
        18: "date palm",
        19: "grapes",
        20: "cotton",
        21: "cocoa",
        22: "coffee",
        23: "others perennial",
        24: "fodder grasses",
        25: "others annual",
    }
    examples = {
        "winter_common_soft_wheat": 0,
        "maize": 1,
        "rice": 2,
        "spring_barley": 3,
        "winter_rye": 4,
        "millet": 5,
        "sorghum": 6,
        "soybeans": 7,
        "sunflower": 8,
        "potatoes": 9,
        "cassava": 10,
        "sugar_cane": 11,
        "sugar_beet": 12,
        "oil_palm": 13,
        "rapeseed": 14,
        "groundnuts": 15,
        "lentils": 16,
        "oranges": 17,
        "date_palm": 18,
        "vineyard_grapes": 19,
        "cotton": 20,
        "cocoa": 21,
        "coffee": 22,
        "olives": 23,
        "permanent_grassland": 24,
        "fresh_vegetables": 25,
    }
    for source_name, target_id in examples.items():
        mapped = workflow.map_source_label_to_geb_crop(
            source_label_system="hcat4",
            source_label_code="",
            source_label_name=source_name,
        )
        assert mapped["geb_crop_id"] == target_id
        assert mapped["training_eligible"] is True

    excluded = workflow.map_source_label_to_geb_crop(
        source_label_system="hcat4",
        source_label_code="",
        source_label_name="arable_crops",
    )
    assert excluded["mapping_status"] == "excluded"
    assert excluded["training_eligible"] is False

    combined_cereal = workflow.map_source_label_to_geb_crop(
        source_label_system="ewoc",
        source_label_code="110107",
        source_label_name="millet and sorghum",
    )
    assert combined_cereal["geb_crop_id"] == 25

    combined_fibre = workflow.map_source_label_to_geb_crop(
        source_label_system="hcat4",
        source_label_code="",
        source_label_name="flax cotton and hemp",
    )
    assert combined_fibre["mapping_status"] == "excluded"


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
            "geb_crop_id": [0, 3],
            "geb_crop_name": ["wheat", "barley"],
            "geb_crop_group_code": [11, 11],
            "geb_crop_group_name": ["Cereals", "Cereals"],
            "target_granularity": ["specific", "specific"],
            "mapping_status": ["mapped", "mapped"],
            "mapping_reason": [pd.NA, pd.NA],
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


def test_empty_model_can_harmonize_and_split_without_predictors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = load_workflow()
    model_dir = workflow.model_output_directory(tmp_path, 10)
    memory_tables: dict[Path, gpd.GeoDataFrame] = {}

    raw = workflow.empty_raw_catalogue()
    raw = workflow._assign_global_region_ids(raw)
    raw_path = model_dir / workflow.RAW_CATALOGUE_FILENAME
    raw_path.touch()
    memory_tables[raw_path] = raw

    def fake_read(path):
        return memory_tables[Path(path)].copy()

    def fake_write(data, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        memory_tables[path] = data.copy()

    monkeypatch.setattr(workflow.gpd, "read_parquet", fake_read)
    monkeypatch.setattr(workflow, "atomic_write_geoparquet", fake_write)
    monkeypatch.setattr(workflow, "write_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "atomic_write_json", lambda *_args, **_kwargs: None)

    with workflow.model_output_scope(10):
        harmonized_path = workflow.setup_harmonize_reference_catalogue(
            tmp_path, overwrite=True
        )
        split_path = workflow.setup_define_reference_split(tmp_path, overwrite=True)

    assert memory_tables[harmonized_path].empty
    assert memory_tables[split_path].empty
    training_path = model_dir / workflow.TRAINING_CATALOGUE_FILENAME
    assert memory_tables[training_path].empty


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
                "europe_model_id": 3,
                "europe_model_name": "Europe_003",
                "observation_year": 2022,
                "split": "train" if source == "eurocrops_v2" else "test",
                "hrl_cty_code": 1110,
                "geb_crop_id": 0,
                "geb_crop_name": "wheat",
                "geb_crop_group_code": 11,
                "geb_crop_group_name": "Cereals",
                "target_granularity": "specific",
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
        "train",
    )


def test_tile_manifest_and_predictor_stages_with_fake_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = load_workflow()
    output_dir = workflow.model_output_directory(tmp_path, 3)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    with workflow.model_output_scope(3):
        tile_path = workflow.setup_build_alphaearth_tile_manifest(
            tmp_path,
            (3,),
            overwrite=True,
        )
    assert tile_path.exists()
    points = memory_tables[output_dir / workflow.REFERENCE_SAMPLING_POINTS_FILENAME]
    assert len(points) == (
        workflow.EUROCROPS_POINTS_PER_PARCEL
        + workflow.WORLDCEREAL_POLYGON_POINTS_PER_FEATURE
    )

    with workflow.model_output_scope(3):
        sample_path = workflow.setup_extract_alphaearth_predictors(
            tmp_path,
            (3,),
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


def test_cached_predictor_model_set_must_match_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = load_workflow()
    cached = tmp_path / "cached_predictors.parquet"
    cached.touch()
    monkeypatch.setattr(
        workflow.pd,
        "read_parquet",
        lambda path, columns=None: pd.DataFrame({"europe_model_id": [3, 3]}),
    )

    assert workflow._parquet_model_ids_match(cached, (3,))
    assert not workflow._parquet_model_ids_match(cached, (0, 1, 2))
    assert not workflow._parquet_model_ids_match(cached, (3, 10))


def test_global_region_ids_are_independent_of_model_selection() -> None:
    workflow = load_workflow()
    model_3 = gpd.GeoDataFrame(
        {
            "europe_model_id": [3, 3],
            "local_region_id": [7, 9],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs=3035,
    )
    model_10 = gpd.GeoDataFrame(
        {
            "europe_model_id": [10],
            "local_region_id": [2],
        },
        geometry=[Point(2, 2)],
        crs=3035,
    )
    separate_3 = workflow._assign_global_region_ids(model_3)
    separate_10 = workflow._assign_global_region_ids(model_10)
    pooled = workflow._assign_global_region_ids(
        gpd.GeoDataFrame(
            pd.concat([model_3, model_10], ignore_index=True),
            geometry="geometry",
            crs=3035,
        )
    )
    expected = {
        (int(row.europe_model_id), int(row.local_region_id)): int(row.region_id)
        for row in pd.concat([separate_3, separate_10]).itertuples(index=False)
    }
    observed = {
        (int(row.europe_model_id), int(row.local_region_id)): int(row.region_id)
        for row in pooled.itertuples(index=False)
    }
    assert observed == expected


def test_model_output_directories_are_isolated(tmp_path: Path) -> None:
    workflow = load_workflow()
    model_3 = workflow.model_output_directory(tmp_path, 3)
    model_10 = workflow.model_output_directory(tmp_path, 10)
    assert model_3 != model_10
    assert model_3 == (
        tmp_path / "Europe_003" / "base" / "machine_learning" / "crop_classification"
    )
    assert model_10 == (
        tmp_path / "Europe_010" / "base" / "machine_learning" / "crop_classification"
    )

    with workflow.model_output_scope(3):
        assert workflow.output_directory(tmp_path) == model_3
    with workflow.model_output_scope(10):
        assert workflow.output_directory(tmp_path) == model_10
    assert workflow.output_directory(tmp_path) == workflow.shared_output_directory(
        tmp_path
    )


def _minimal_predictor_training_rows() -> pd.DataFrame:
    rows = []
    for model_id, split, reference_id, source, crop_id, crop_name, hrl_code in (
        (3, "train", "euro::1", "eurocrops_v2", 0, "wheat", 1110),
        (3, "validation", "wc::1", "worldcereal_rdm", 3, "barley", 1120),
        (10, "train", "euro::2", "eurocrops_v2", 1, "maize", 1130),
    ):
        point_count = 3 if source == "eurocrops_v2" else 1
        for point_index in range(point_count):
            row = {
                "reference_id": reference_id,
                "reference_source": source,
                "europe_model_id": model_id,
                "europe_model_name": f"Europe_{model_id:03d}",
                "local_region_id": model_id,
                "region_id": model_id,
                "country_iso3": "NLD",
                "observation_year": 2022,
                "hrl_cty_code": hrl_code,
                "geb_crop_id": crop_id,
                "geb_crop_name": crop_name,
                "geb_crop_group_code": 11,
                "geb_crop_group_name": "Cereals",
                "target_granularity": "specific",
                "split": split,
                "spatial_block_id": f"block-{model_id}-{split}",
                "predictor_sample_id": f"{reference_id}::p{point_index:02d}",
                "sample_weight": 1.0 / point_count,
            }
            row.update(
                {
                    f"alphaearth_A{band:02d}": float(point_index + band)
                    for band in range(64)
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_training_sample_normalization_filters_selected_models() -> None:
    workflow = load_workflow()
    normalized = workflow.normalize_source_reference_predictor_samples(
        _minimal_predictor_training_rows(),
        europe_model_ids=(3,),
    )
    assert set(normalized["europe_model_id"]) == {3}
    assert set(normalized["cty_label"]) == {0, 3}
    assert set(normalized["year"]) == {2022}


def test_reference_mean_representation_returns_one_row_per_feature() -> None:
    workflow = load_workflow()
    normalized = workflow.normalize_source_reference_predictor_samples(
        _minimal_predictor_training_rows(),
        europe_model_ids=(3, 10),
    )
    represented = workflow.apply_source_reference_sample_representation(
        normalized,
        "reference_mean",
    )
    assert len(represented) == normalized["reference_id"].nunique()
    euro = represented.loc[represented["reference_id"].eq("euro::1")].iloc[0]
    assert np.isclose(euro["alphaearth_A00"], 1.0)
    assert np.isclose(euro["sample_weight"], 1.0)


def test_pooling_arbitrary_model_subset_uses_only_requested_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = load_workflow()
    samples = _minimal_predictor_training_rows()
    extra = samples.loc[samples["europe_model_id"].eq(3)].copy()
    extra["europe_model_id"] = 1
    extra["europe_model_name"] = "Europe_001"
    extra["reference_id"] = extra["reference_id"].astype(str) + "::model1"
    extra["predictor_sample_id"] = extra["predictor_sample_id"].astype(str) + "::model1"
    combined = pd.concat([samples, extra], ignore_index=True)

    memory_tables: dict[Path, pd.DataFrame] = {}
    for model_id in (1, 3, 10):
        model_dir = workflow.model_output_directory(tmp_path, model_id)
        sample_path = model_dir / workflow.ALPHAEARTH_TRAINING_SAMPLES_FILENAME
        sample_path.touch()
        memory_tables[sample_path] = combined.loc[
            combined["europe_model_id"].eq(model_id)
        ].copy()
    monkeypatch.setattr(
        workflow.pd,
        "read_parquet",
        lambda path, columns=None: memory_tables[Path(path)].copy(),
    )

    pooled, paths = workflow.pool_model_predictor_samples(tmp_path, (1, 10))
    assert set(pooled["europe_model_id"]) == {1, 10}
    assert len(paths) == 2
    assert workflow.model_selection_key((10, 1)) == "models_1_10"


def test_comparison_internal_calibration_keeps_blocks_together() -> None:
    comparison = load_comparison()
    rows = []
    for block in range(10):
        for index in range(3):
            rows.append(
                {
                    "spatial_block_id": f"block-{block}",
                    "reference_id": f"ref-{block}-{index}",
                    "cty_label": 1110,
                    "region_id": 1,
                    "year": 2022,
                }
            )
    training = pd.DataFrame(rows)
    fit, calibration = comparison.split_internal_spatial_calibration(
        training,
        fraction=0.2,
        random_seed=42,
    )
    assert not fit.empty
    assert not calibration.empty
    assert set(fit["spatial_block_id"]).isdisjoint(set(calibration["spatial_block_id"]))


def test_comparison_sample_configuration_cross_product() -> None:
    comparison = load_comparison()
    arguments = types.SimpleNamespace(
        source_configurations=("all", "eurocrops"),
        sample_representations=("reference_mean", "as_extracted"),
    )
    configurations = comparison.build_sample_configurations(arguments)
    assert [item.name for item in configurations] == [
        "all__reference_mean",
        "all__as_extracted",
        "eurocrops__reference_mean",
        "eurocrops__as_extracted",
    ]


def test_train_stage_uses_preassigned_spatial_splits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workflow = load_workflow()
    samples = _minimal_predictor_training_rows()
    memory_tables: dict[Path, pd.DataFrame] = {}
    for model_id in (3, 10):
        model_dir = workflow.model_output_directory(tmp_path, model_id)
        sample_path = model_dir / workflow.ALPHAEARTH_TRAINING_SAMPLES_FILENAME
        sample_path.touch()
        memory_tables[sample_path] = samples.loc[
            samples["europe_model_id"].eq(model_id)
        ].copy()

    def fake_read_parquet(path, columns=None):
        table = memory_tables[Path(path)].copy()
        return table if columns is None else table.loc[:, list(columns)]

    def fake_atomic_write_parquet(data, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        memory_tables[path] = data.copy()

    monkeypatch.setattr(workflow.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(workflow, "atomic_write_parquet", fake_atomic_write_parquet)
    monkeypatch.setattr(workflow, "write_table", lambda *_args, **_kwargs: None)

    fit_splits: list[set[str]] = []

    class FakeModel:
        model_type = "hist_gradient_boosting"
        model_parameters = {"max_iter": 10}
        class_probability_thresholds = {}

    def fake_fit(data, **_kwargs):
        fit_splits.append(set(data["split"].astype(str)))
        return FakeModel()

    def fake_evaluate(_model, data, *, split_name):
        metrics = pd.DataFrame(
            [
                {
                    "split": split_name,
                    "target": "CTY",
                    "metric_scope": "summary",
                    "accuracy": 1.0,
                    "macro_f_score_observed": 1.0,
                    "balanced_accuracy": 1.0,
                }
            ]
        )
        confusion = pd.DataFrame(
            [
                {
                    "split": split_name,
                    "target": "CTY",
                    "reference_class": 1120,
                    "product_class": 1120,
                    "count": len(data),
                }
            ]
        )
        return metrics, confusion

    def fake_save(_model, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
        return path

    monkeypatch.setattr(workflow, "fit_alphaearth_crop_models", fake_fit)
    monkeypatch.setattr(workflow, "evaluate_alphaearth_crop_models", fake_evaluate)
    monkeypatch.setattr(workflow, "save_alphaearth_crop_models", fake_save)
    monkeypatch.setattr(
        workflow,
        "format_alphaearth_accuracy_report",
        lambda *_args, **_kwargs: "accuracy",
    )
    monkeypatch.setattr(
        workflow,
        "alphaearth_crop_feature_importance",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(workflow, "alphaearth_embedding_diagnostics", lambda *_args: {})

    model_path = workflow.setup_train_alphaearth_source_reference_model(
        tmp_path,
        (3, 10),
        overwrite=True,
        model_parameters_json='{"max_iter": 10}',
    )
    assert model_path.is_file()
    assert fit_splits[0] == {"train"}
    assert fit_splits[-1] == {"train", "validation"}
    assert "test" not in fit_splits[-1]
