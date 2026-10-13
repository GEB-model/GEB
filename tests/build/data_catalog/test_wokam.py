"""Tests for WOKAM archive reading and cache reuse."""

import logging
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from geb.build.data_catalog.wokam import WOKAM


def test_wokam_archive_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repair downloaded polygons and reuse their cached GeoParquet.

    Args:
        tmp_path: Temporary directory for the synthetic WOKAM archive.
        monkeypatch: Fixture replacing the network downloader.
    """
    source: Path = tmp_path / "source"
    source.mkdir()
    polygons: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"rock_type": [1]},
        geometry=[Polygon([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)])],
        crs=4326,
    )
    polygons.to_file(source / "whymap_karst__v1_poly.shp")
    archive: Path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for shapefile_part in source.iterdir():
            zipped.write(shapefile_part, f"WHYMAP_WOKAM/shp/{shapefile_part.name}")
    downloads: list[str] = []

    def download(url: str, target: Path, logger: logging.Logger) -> None:
        """Copy the test archive instead of requesting a remote dataset.

        Args:
            url: Requested source URL.
            target: Destination archive path.
            logger: Adapter logger.
        """
        downloads.append(url)
        shutil.copyfile(archive, target)

    monkeypatch.setattr("geb.build.data_catalog.wokam.fetch_and_save", download)
    adapter: WOKAM = WOKAM(
        folder=tmp_path / "cache",
        filename="wokam.parquet",
        local_version=1,
        cache="global",
    )
    adapter.logger = logging.getLogger(__name__)
    adapter.fetch("test-url")
    result: gpd.GeoDataFrame = adapter.read()
    assert result.is_valid.all()
    assert result["rock_type"].tolist() == [1]
    adapter.fetch("test-url")
    assert downloads == ["test-url"]
