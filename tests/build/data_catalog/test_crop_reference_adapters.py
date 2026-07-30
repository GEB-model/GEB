"""Fast isolated tests for the crop-reference data adapters."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types

import geopandas as gpd
from shapely.geometry import Point, box


TEST_PACKAGE = "crop_reference_adapter_test_package"


def _repository_root() -> Path:
    """Locate the repository root regardless of the tests subdirectory depth."""
    for candidate in Path(__file__).resolve().parents:
        data_catalog = candidate / "geb" / "build" / "data_catalog"
        if data_catalog.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate geb/build/data_catalog above {Path(__file__).resolve()}."
    )


PACKAGE_ROOT = _repository_root() / "geb" / "build" / "data_catalog"


def _install_test_stubs() -> None:
    """Install minimal Adapter and aiohttp-retry interfaces in an isolated package."""
    retry = types.ModuleType("aiohttp_retry")

    class ExponentialRetry:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class RetryClient:
        pass

    retry.ExponentialRetry = ExponentialRetry
    retry.RetryClient = RetryClient
    sys.modules.setdefault("aiohttp_retry", retry)

    package = types.ModuleType(TEST_PACKAGE)
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules.setdefault(TEST_PACKAGE, package)

    base = types.ModuleType(f"{TEST_PACKAGE}.base")

    class Adapter:
        """Test double that follows GEB_DATA_ROOT through self.root."""

        def __init__(
            self,
            *args,
            folder="adapter",
            local_version=1,
            filename="data",
            cache="global",
            **kwargs,
        ):
            del args, filename, cache, kwargs
            configured_root = os.environ.get("GEB_DATA_ROOT")
            data_root = (
                Path(configured_root).expanduser()
                if configured_root
                else Path(tempfile.mkdtemp())
            )
            self.root = data_root / folder / f"v{int(local_version)}"
            self.root.mkdir(parents=True, exist_ok=True)

    base.Adapter = Adapter
    sys.modules[f"{TEST_PACKAGE}.base"] = base


def _load_module(name: str, filename: str):
    """Load one adapter without importing the complete GEB data catalog."""
    module_name = f"{TEST_PACKAGE}.{name}"
    specification = importlib.util.spec_from_file_location(
        module_name,
        PACKAGE_ROOT / filename,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def load_adapters():
    """Return isolated EuroCrops, WorldCereal and AlphaEarth modules."""
    _install_test_stubs()
    return (
        _load_module("eurocrops_v2", "eurocrops_v2.py"),
        _load_module(
            "worldcereal_reference_data",
            "worldcereal_reference_data.py",
        ),
        _load_module("alphaearth", "alphaearth.py"),
    )


def test_repository_root_resolution() -> None:
    assert (PACKAGE_ROOT / "eurocrops_v2.py").is_file()
    assert (PACKAGE_ROOT / "worldcereal_reference_data.py").is_file()


def test_adapters_derive_storage_from_geb_data_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    eurocrops, worldcereal, alphaearth = load_adapters()

    adapters = (
        eurocrops.EuroCropsV2(
            folder="eurocrops_v2",
            local_version=1,
            filename="parcels",
            cache="global",
        ),
        worldcereal.WorldCerealReferenceData(
            folder="worldcereal_reference_data",
            local_version=1,
            filename="reference_data",
            cache="global",
        ),
        alphaearth.AlphaEarth(
            folder="alphaearth",
            local_version=1,
            filename="tiles",
            cache="global",
        ),
    )

    resolved_data_root = tmp_path.resolve()
    for adapter in adapters:
        assert Path(adapter.root).resolve().is_relative_to(resolved_data_root)


def test_eurocrops_manifest_country_mapping_and_latest_version() -> None:
    eurocrops, _, _ = load_adapters()
    manifest = (
        "path\ngpqt/fr_2018.parquet\ngpqt/de4_2020.parquet\ngpqt/fr_stack.parquet\n"
    )
    inventory = eurocrops.EuroCropsV2._parse_manifest_text(manifest)
    assert set(inventory["filename"]) == {
        "fr_2018.parquet",
        "de4_2020.parquet",
        "fr_stack.parquet",
    }
    assert eurocrops.EuroCropsV2.regions_for_countries(["FRA", "DEU", "PRT"]) == (
        "fr",
        "de4",
        "dea",
        "pt",
    )
    assert eurocrops.DEFAULT_DATA_SUBDIRECTORIES[0] == "gpqtv202"


def test_worldcereal_prefers_public_collection_id_and_resolves_uuid(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    _, worldcereal, _ = load_adapters()
    adapter = worldcereal.WorldCerealReferenceData(
        folder="worldcereal_reference_data",
        local_version=1,
        filename="reference_data",
        cache="global",
    )

    database_uuid = "3a1a04e7-d8d8-d640-3748-de6e02b136ce"
    public_collection_id = "2018_eu_lucascopernicus_poly_110"
    payload = {
        "totalCount": 1,
        "items": [
            {
                "collectionId": public_collection_id,
                "title": "LUCAS Copernicus 2018",
                "id": database_uuid,
            }
        ],
    }
    adapter.collections_cache_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    collections = adapter.list_collections()
    assert collections["collection_id"].tolist() == [public_collection_id]
    assert (
        asyncio.run(adapter._resolve_public_collection_id_async(database_uuid))
        == public_collection_id
    )
    assert adapter.feature_collections_url == (
        "https://rdm.esa-worldcereal.org/collections"
    )


def test_worldcereal_metadata_and_signed_url_normalization() -> None:
    _, worldcereal, _ = load_adapters()
    payload = {
        "collections": [
            {
                "id": "2018_EU_LUCAS_POLY_110",
                "title": "LUCAS crop type 2018",
            }
        ]
    }
    records = worldcereal.WorldCerealReferenceData._collection_records(payload)
    assert records[0]["id"].startswith("2018")

    descriptor = {
        "data": {
            "download": {"signedUrl": "https://example.test/reference.parquet?token=x"}
        }
    }
    assert (
        worldcereal.WorldCerealReferenceData._download_url_from_payload(descriptor)
        == "https://example.test/reference.parquet?token=x"
    )
    assert worldcereal.WorldCerealReferenceData._looks_like_parquet(
        b"PAR1payload",
        "application/octet-stream",
    )


def test_worldcereal_semicolon_legend_parser(tmp_path: Path) -> None:
    _, worldcereal, _ = load_adapters()
    legend_path = tmp_path / "WorldCereal_LC_CT_legend_latest.csv"
    legend_path.write_text(
        "ewoc_code;label_full;level_1;definition\n"
        "1111020036;Common wheat;Cropland;Cereal crop, including winter wheat\n"
        "1111020040;Barley;Cropland;Cereal crop\n",
        encoding="utf-8",
    )

    legend = worldcereal.WorldCerealReferenceData._read_legend_csv(legend_path)

    assert legend.columns.tolist() == [
        "ewoc_code",
        "label_full",
        "level_1",
        "definition",
    ]
    assert legend["label_full"].tolist() == ["Common wheat", "Barley"]
    assert "including winter wheat" in legend.iloc[0]["definition"]


def test_worldcereal_reference_normalization_and_filtering() -> None:
    _, worldcereal, _ = load_adapters()
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [5.0, 52.0]},
            "properties": {
                "sample_id": "a",
                "ewoc_code": 1111020036,
                "valid_time": "2018-06-01",
                "extract": 1,
                "quality_score_ct": 90,
            },
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [15.0, 45.0]},
            "properties": {
                "sample_id": "b",
                "ewoc_code": 1111020036,
                "valid_time": "2022-06-01",
                "extract": 0,
                "quality_score_ct": 40,
            },
        },
    ]
    table = gpd.GeoDataFrame.from_features(features, crs=4326)
    normalized = worldcereal.WorldCerealReferenceData._normalize_reference_data(
        table,
        collection_id="lucas",
    )
    filtered = worldcereal.WorldCerealReferenceData._filter_reference_data(
        normalized,
        years=(2018,),
        use_extract_only=True,
        min_quality_score_ct=80,
        query_geometry=Point(5.0, 52.0).buffer(0.1),
        bounds=None,
    )
    assert filtered["sample_id"].tolist() == ["a"]
    assert filtered["source_collection_id"].tolist() == ["lucas"]


def test_alphaearth_exact_geometry_excludes_boundary_only_tile() -> None:
    _, _, alphaearth = load_adapters()
    index = gpd.GeoDataFrame(
        {
            "year": [2018, 2018],
            "path": ["a.tif", "b.tif"],
            "utm_zone": ["31N", "31N"],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs=4326,
    )
    adapter = alphaearth.AlphaEarth(
        folder="alphaearth",
        local_version=1,
        filename="tiles",
        cache="global",
    )
    selected = adapter.select_files_for_geometry(
        index,
        2018,
        box(0.1, 0.1, 1.0, 0.9),
    )
    assert len(selected) == 1
    assert selected.iloc[0]["path"] == "a.tif"
