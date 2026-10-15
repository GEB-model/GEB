"""Test reading AMBER atlas fields."""

import logging
import shutil
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import pytest

from geb.build.data_catalog import DataCatalog
from geb.build.data_catalog.amber import AMBER, prepare_amber
from geb.build.data_catalog.base import Adapter


def test_prepare_amber() -> None:
    """Keep valid locations, remove duplicate IDs and clean heights.

    Returns:
        None.

    Raises:
        AssertionError: If coordinates, IDs, normalized types or heights are wrong.
    """  # noqa: DOC202, DOC502
    records: pd.DataFrame = pd.DataFrame(
        {
            "GUID": ["a", "a", "b", "c", "d"],
            "Longitude_WGS84": [4, 4, 5, 6, 200],
            "Latitude_WGS84": [52, 52, 53, 54, 55],
            "LabelAtlas": ["DAM", "DAM", "SLUICE", "WEIR", "OTHER"],
            "Height": [2.5, 2.5, -99, "unknown", 1],
        }
    )
    barriers: gpd.GeoDataFrame = prepare_amber(records)
    assert barriers.crs.to_epsg() == 4326
    assert barriers.amber_id.tolist() == ["a", "b", "c"]
    assert barriers.dam_type.tolist() == ["Dam", "Sluice", "Weir"]
    assert barriers.dam_hgt_m.iloc[0] == 2.5
    assert barriers.dam_hgt_m.iloc[1:].isna().all()
    assert barriers.geometry.iloc[0].x == 4


@pytest.mark.parametrize("missing_id", [True, False])
def test_invalid_amber(missing_id: bool) -> None:
    """Reject missing fields and missing barrier IDs.

    Args:
        missing_id: Whether to supply all fields but omit the ID value.

    Returns:
        None.

    Raises:
        AssertionError: If incomplete source records are accepted.
    """  # noqa: DOC202, DOC502
    records: pd.DataFrame = pd.DataFrame()
    if missing_id:
        records = pd.DataFrame(
            {
                "GUID": [None],
                "Longitude_WGS84": [4],
                "Latitude_WGS84": [52],
                "LabelAtlas": ["DAM"],
                "Height": [1],
            }
        )
    with pytest.raises(ValueError, match="AMBER"):
        prepare_amber(records)


@pytest.mark.parametrize("missing_height", [False, True])
def test_fetch_amber_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_height: bool
) -> None:
    """Fetch a local atlas archive, validate it, and read the cached points.

    Args:
        tmp_path: Directory for the archive and isolated dataset cache.
        monkeypatch: Fixture replacing the network download and cache location.
        missing_height: Whether the archive omits a required field.

    Returns:
        None.

    Raises:
        AssertionError: If invalid data is cached, downloads repeat, or reading fails.
    """  # noqa: DOC202, DOC502
    records: pd.DataFrame = pd.DataFrame(
        {
            "GUID": ["near", "far"],
            "Longitude_WGS84": [4.0, 10.0],
            "Latitude_WGS84": [52.0, 55.0],
            "LabelAtlas": ["DAM", "WEIR"],
            "Height": [2.5, 1.0],
        }
    )
    if missing_height:
        records = records.drop(columns="Height")
    archive: Path = tmp_path / "atlas.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("AMBER_BARRIER_ATLAS_V1.csv", records.to_csv(index=False))
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path / "cache"))

    def copy_archive(url: str, file_path: Path, logger: logging.Logger) -> None:
        """Copy the fixture archive instead of accessing the network.

        Args:
            url: Download URL from the catalog.
            file_path: Destination of the downloaded archive.
            logger: Catalog logger supplied to the download helper.

        Returns:
            None.

        Raises:
            OSError: If the fixture archive cannot be copied.
        """  # noqa: DOC202, DOC502
        shutil.copyfile(archive, file_path)

    download: Mock = Mock(side_effect=copy_archive)
    monkeypatch.setattr("geb.build.data_catalog.amber.fetch_and_save", download)
    catalog: DataCatalog = DataCatalog(logger=logging.getLogger(__name__))
    if missing_height:
        with pytest.raises(ValueError, match="required V1 fields"):
            catalog.fetch("amber_barriers")
        assert not catalog.catalog["amber_barriers"]["adapter"].path.exists()
        return
    adapter: Adapter = catalog.fetch("amber_barriers")
    assert isinstance(adapter, AMBER)
    barriers: gpd.GeoDataFrame = adapter.read(bbox=(3.0, 51.0, 5.0, 53.0))
    assert barriers.amber_id.tolist() == ["near"]
    assert barriers.crs.to_epsg() == 4326
    assert barriers.dam_hgt_m.tolist() == [2.5]
    assert catalog.fetch("amber_barriers") is adapter
    assert len(adapter.read()) == 2
    download.assert_called_once()
