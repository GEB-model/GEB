"""Live integration tests for EuroCrops and WorldCereal adapters.

These tests perform real network downloads and use the catalog-managed cache
below ``GEB_DATA_ROOT``. They are skipped only in GitHub Actions.
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import pytest

from geb.build.data_catalog import data_catalog

from ...testconfig import IN_GITHUB_ACTIONS


EUROCROPS_REGION = os.environ.get("GEB_LIVE_EUROCROPS_REGION", "be2")
EUROCROPS_YEAR = int(os.environ.get("GEB_LIVE_EUROCROPS_YEAR", "2008"))
WORLDCEREAL_COLLECTION_ID = os.environ.get(
    "GEB_LIVE_WORLDCEREAL_COLLECTION_ID",
    "2021_PT_EUROCROP_POLY_110",
)
WORLDCEREAL_BOUNDS = (-8.60, 40.50, -8.45, 40.65)


def _catalog_adapter(name: str):
    entry = data_catalog[name]
    return entry["adapter"].fetch(entry.get("url"))


def _assert_uses_geb_data_root(adapter) -> None:
    configured = os.environ.get("GEB_DATA_ROOT")
    assert configured, "GEB_DATA_ROOT must be configured for live adapter tests."
    root = Path(adapter.root).expanduser().resolve()
    data_root = Path(configured).expanduser().resolve()
    assert root.is_relative_to(data_root), (root, data_root)


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Downloads and opens a real EuroCrops GeoParquet parcel dataset.",
)
def test_live_eurocrops_parcel_download() -> None:
    """Download one real annual EuroCrops parcel file and read its schema."""
    adapter = _catalog_adapter("eurocrops_v2")
    _assert_uses_geb_data_root(adapter)

    parcels = adapter.read(
        years=EUROCROPS_YEAR,
        regions=EUROCROPS_REGION,
        include_mapping=True,
        drop_unmapped=False,
        overwrite=False,
        refresh_manifest=False,
        refresh_mapping=False,
        max_files=1,
    )

    assert isinstance(parcels, gpd.GeoDataFrame)
    assert not parcels.empty
    assert parcels.crs is not None and parcels.crs.to_epsg() == 3035
    assert {
        "cropfield",
        "original_code",
        "area_ha",
        "source_feature_id",
        "source_file",
        "source_url",
    }.issubset(parcels.columns)

    source_files = {Path(value).resolve() for value in parcels["source_file"].unique()}
    assert len(source_files) == 1
    source_file = next(iter(source_files))
    assert source_file.is_file() and source_file.stat().st_size > 0
    assert source_file.is_relative_to(Path(adapter.root).resolve())


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Downloads and opens a real WorldCereal reference-data collection.",
)
def test_live_worldcereal_collection_download() -> None:
    """Download one real public WorldCereal GeoParquet collection."""
    adapter = _catalog_adapter("worldcereal_reference_data")
    _assert_uses_geb_data_root(adapter)

    observations = adapter.read_collection(
        WORLDCEREAL_COLLECTION_ID,
        bounds=WORLDCEREAL_BOUNDS,
        use_extract_only=False,
        min_quality_score_ct=None,
        refresh=False,
        max_features=150_000,
    )

    assert isinstance(observations, gpd.GeoDataFrame)
    assert not observations.empty
    assert observations.crs is not None and observations.crs.to_epsg() == 4326
    assert {
        "sample_id",
        "ewoc_code",
        "valid_time",
        "source_collection_id",
        "reference_source",
    }.issubset(observations.columns)
    assert observations["source_collection_id"].eq(WORLDCEREAL_COLLECTION_ID).all()

    full_cache = adapter._collection_download_cache_path(
        WORLDCEREAL_COLLECTION_ID,
        use_extract_only=False,
    )
    assert full_cache.is_file() and full_cache.stat().st_size > 0
    assert full_cache.resolve().is_relative_to(Path(adapter.root).resolve())
