"""Test reading AMBER atlas fields."""

import geopandas as gpd
import pandas as pd
import pytest

from geb.build.data_catalog.amber import prepare_amber


def test_prepare_amber() -> None:
    """Keep valid locations, remove duplicate IDs and clean heights."""
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
    """
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
