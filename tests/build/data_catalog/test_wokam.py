"""Test WOKAM normalization and caching without downloading external data."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from geb.build.data_catalog.wokam import WOKAM
from geb.workflows.io import write_geom


@pytest.fixture
def wokam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WOKAM:
    """Create a WOKAM adapter using an isolated cache.

    Args:
        tmp_path: Temporary cache root.
        monkeypatch: Fixture for setting the cache environment variable.

    Returns:
        An adapter that reads and writes only within the temporary cache.
    """
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    adapter: WOKAM = WOKAM(
        folder="wokam", filename="wokam.parquet", local_version=1, cache="global"
    )
    adapter.logger = logging.getLogger("test_wokam")
    return adapter


def test_wokam_read(wokam: WOKAM) -> None:
    """Normalize all rock classes and attach source attribution on reading.

    Args:
        wokam: Adapter with an isolated cache.
    """
    polygons: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"rock_type": [5, 3, 1, 4, 2]},
        geometry=[box(index, 0, index + 1, 1) for index in range(5)],
        index=[40, 10, 30, 20, 0],
        crs="EPSG:4326",
    )
    write_geom(polygons, wokam.path)
    actual: gpd.GeoDataFrame = wokam.read()
    np.testing.assert_allclose(actual["karst_fraction"], [0.4, 0.825, 0.825, 0.65, 0.4])
    assert actual.index.equals(polygons.index)
    assert actual.crs == polygons.crs
    assert actual.geometry.equals(polygons.geometry)
    assert actual.attrs["source"] == "WHYMAP WOKAM, BGR, IAH, KIT, UNESCO, 2017"


@pytest.mark.parametrize("invalid_input", ["missing_crs", "unknown_class"])
def test_wokam_read_invalid_data(wokam: WOKAM, invalid_input: str) -> None:
    """Reject invalid source data before it reaches the model build.

    Args:
        wokam: Adapter with an isolated cache.
        invalid_input: Missing spatial reference or an unknown rock-class code.
    """
    polygons: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"rock_type": [99 if invalid_input == "unknown_class" else 1]},
        geometry=[box(0, 0, 1, 1)],
        crs=None if invalid_input == "missing_crs" else "EPSG:4326",
    )
    write_geom(polygons, wokam.path)
    message: str = (
        "must have a CRS" if invalid_input == "missing_crs" else "unknown rock classes"
    )
    with pytest.raises(ValueError, match=message):
        wokam.read()


def test_wokam_fetch_repairs_and_caches(
    wokam: WOKAM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repair invalid polygons, use the geometry writer and reuse the cache.

    Args:
        wokam: Adapter with an isolated cache.
        monkeypatch: Fixture replacing network and archive reads.
    """
    polygons: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"rock_type": [1]},
        geometry=[Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])],
        crs="EPSG:4326",
    )
    assert not polygons.geometry.is_valid.all()
    download: MagicMock = MagicMock()
    read_file: MagicMock = MagicMock(return_value=polygons)
    writer: MagicMock = MagicMock(wraps=write_geom)
    monkeypatch.setattr("geb.build.data_catalog.wokam.fetch_and_save", download)
    monkeypatch.setattr("geb.build.data_catalog.wokam.gpd.read_file", read_file)
    monkeypatch.setattr("geb.build.data_catalog.wokam.write_geom", writer)

    assert wokam.fetch("https://example.org/wokam.zip") is wokam
    writer.assert_called_once()
    assert writer.call_args.args[1] == wokam.path
    assert wokam.read().geometry.is_valid.all()
    assert wokam.fetch("https://example.org/wokam.zip") is wokam
    download.assert_called_once()
    read_file.assert_called_once()
    writer.assert_called_once()
