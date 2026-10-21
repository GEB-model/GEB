"""Test GDW field names, units, and missing-data values."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from geb.build.data_catalog.gdw import prepare_gdw


@pytest.fixture
def gdw_records() -> gpd.GeoDataFrame:
    """Create GDW records with valid values and missing-data codes.

    Returns:
        Dam points in EPSG:4326 using the original GDW field names and units.
    """
    return gpd.GeoDataFrame(
        {
            "GDW_ID": [1, 2],
            "HYLAK_ID": [42, -99],
            "DAM_TYPE": [" Dam ", ""],
            "LAKE_CTRL": [" No ", ""],
            "CAP_MCM": [2.5, -99.0],
            "AREA_SKM": [0.5, -99.0],
            "DIS_AVG_LS": [1500.0, -99.0],
            "YEAR_DAM": [2000, -99],
            "ELEV_MASL": [-99.0, -9999.0],
            "LONG_DAM": [-99.0, 0.0],
        },
        geometry=[Point(-99, 45), Point(0, 45)],
        crs=4326,
    )


def test_prepare_gdw(gdw_records: gpd.GeoDataFrame) -> None:
    """Convert units and missing values without changing valid negative locations.

    Args:
        gdw_records: Original GDW records.

    Returns:
        None.

    Raises:
        AssertionError: If converted data or the original records are incorrect.
    """  # noqa: DOC202, DOC502
    original: gpd.GeoDataFrame = gdw_records.copy()
    result: gpd.GeoDataFrame = prepare_gdw(gdw_records)
    assert result.crs.to_epsg() == 4326
    assert result.gdw_id.tolist() == [1, 2]
    assert result.hydrolakes_id.iloc[0] == 42
    assert pd.isna(result.hydrolakes_id.iloc[1])
    assert result.dam_type.iloc[0] == "Dam"
    assert pd.isna(result.dam_type.iloc[1])
    assert result.capacity_m3.iloc[0] == 2.5e6
    assert result.area_m2.iloc[0] == 0.5e6
    assert result.average_discharge_m3_per_s.iloc[0] == 1.5
    assert result.construction_year.iloc[0] == 2000
    assert (
        result.iloc[1][
            [
                "capacity_m3",
                "area_m2",
                "average_discharge_m3_per_s",
                "construction_year",
            ]
        ]
        .isna()
        .all()
    )
    assert result.elev_masl.iloc[0] == -99.0
    assert pd.isna(result.elev_masl.iloc[1])
    assert result.long_dam.iloc[0] == -99.0
    pd.testing.assert_frame_equal(gdw_records, original)


@pytest.mark.parametrize(
    "invalid",
    ["missing_field", "crs", "duplicate", "zero", "fraction", "infinite", "text"],
)
def test_invalid_gdw(gdw_records: gpd.GeoDataFrame, invalid: str) -> None:
    """Reject incomplete records and IDs that cannot identify individual dams.

    Args:
        gdw_records: Original GDW records.
        invalid: Field, coordinate system, or ID error to check.

    Returns:
        None.

    Raises:
        AssertionError: If invalid source data is accepted.
    """  # noqa: DOC202, DOC502
    if invalid == "missing_field":
        gdw_records.drop(columns="CAP_MCM", inplace=True)
    elif invalid == "crs":
        gdw_records.set_crs(None, allow_override=True, inplace=True)
    else:
        gdw_records["GDW_ID"] = {
            "duplicate": [1, 1],
            "zero": [0, 2],
            "fraction": [1.5, 2.0],
            "infinite": [np.inf, 2.0],
            "text": ["a", "b"],
        }[invalid]
    with pytest.raises(ValueError, match="GDW"):
        prepare_gdw(gdw_records)
