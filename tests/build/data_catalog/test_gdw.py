"""Tests for the Global Dam Watch adapter."""

from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from geb.build.data_catalog.gdw import GlobalDamWatch, prepare_gdw


@pytest.fixture
def raw_gdw() -> gpd.GeoDataFrame:
    """Return two records using the original GDW field names and units.

    Returns:
        GDW data including documented missing-value codes.
    """
    return gpd.GeoDataFrame(
        {
            "GDW_ID": [1, 2],
            "HYLAK_ID": [10, 0],
            "DAM_TYPE": ["Dam", "Dam"],
            "LAKE_CTRL": ["", "Maybe"],
            "CAP_MCM": [2.0, -99.0],
            "AREA_SKM": [3.0, -99.0],
            "DIS_AVG_LS": [4000, -99],
            "YEAR_DAM": [2001, -99],
            "LONG_DAM": [-99.0, -98.0],
            "ELEV_MASL": [-99, -9999],
        },
        geometry=[Point(-99, 30), Point(-98, 30)],
        crs=4326,
    )


def test_gdw_units_and_missing_values(raw_gdw: gpd.GeoDataFrame) -> None:
    """Convert units without treating real coordinates as missing.

    Args:
        raw_gdw: Raw GDW test records.
    """
    data: gpd.GeoDataFrame = prepare_gdw(raw_gdw)
    assert data.loc[0, "capacity_m3"] == 2_000_000
    assert data.loc[0, "area_m2"] == 3_000_000
    assert data.loc[0, "average_discharge_m3_per_s"] == 4
    assert (
        data.loc[
            1,
            [
                "capacity_m3",
                "area_m2",
                "average_discharge_m3_per_s",
                "construction_year",
                "hydrolakes_id",
            ],
        ]
        .isna()
        .all()
    )
    assert pd.isna(data.loc[0, "lake_control"])
    assert data.loc[0, "long_dam"] == -99
    assert data.loc[0, "elev_masl"] == -99
    assert pd.isna(data.loc[1, "elev_masl"])


def test_gdw_duplicate_ids(raw_gdw: gpd.GeoDataFrame) -> None:
    """Reject duplicate GDW records.

    Args:
        raw_gdw: Raw GDW test records.
    """
    raw_gdw["GDW_ID"] = 1
    with pytest.raises(ValueError, match="GDW IDs"):
        prepare_gdw(raw_gdw)


def test_gdw_caches_both_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_gdw: gpd.GeoDataFrame,
) -> None:
    """Cache both layers once and support reading by bounding box.

    Args:
        tmp_path: Test cache directory.
        monkeypatch: Pytest patch helper.
        raw_gdw: Raw GDW test records.
    """
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))

    def download(url: str, file_path: Path) -> None:
        """Create a test ZIP file without downloading data.

        Args:
            url: Unused download URL.
            file_path: Destination ZIP path.
        """
        with ZipFile(file_path, "w") as archive:
            archive.writestr("GDW_v1_0.gdb/", "")

    fetch: Mock = Mock(side_effect=download)
    read: Mock = Mock(return_value=raw_gdw)
    monkeypatch.setattr("geb.build.data_catalog.gdw.fetch_and_save", fetch)
    monkeypatch.setattr("geb.build.data_catalog.gdw.gpd.read_file", read)
    adapter: GlobalDamWatch = GlobalDamWatch(
        folder="gdw", filename="barriers.parquet", local_version=1, cache="global"
    )
    adapter.fetch("test-url")
    assert (adapter.root / "reservoirs.parquet").exists()
    assert len(adapter.read(bbox=(-99.5, 29, -98.5, 31))) == 1
    adapter.fetch("test-url")
    assert fetch.call_count == 1
    assert [call.kwargs["layer"] for call in read.call_args_list] == [
        "GDW_barriers_v1_0",
        "GDW_reservoirs_v1_0",
    ]
