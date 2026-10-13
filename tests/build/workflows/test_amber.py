"""Test the fixed AMBER distance limit."""

from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod
from shapely.geometry import LineString, Point

from geb.build.workflows.weirs import nearest_amber_river


@pytest.mark.parametrize("latitude", [0.0, 52.0, 70.0])
@pytest.mark.parametrize("distance_m", [0.0, 249.999, 250.001, 1000.0])
def test_amber_distance_limit(latitude: float, distance_m: float) -> None:
    """Reject distances above 250 m at different latitudes.

    Args:
        latitude: Barrier latitude (degrees).
        distance_m: Distance to the river endpoint (m).
    """
    longitude: float
    river_latitude: float
    longitude, river_latitude, _ = Geod(ellps="WGS84").fwd(0, latitude, 90, distance_m)
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=[
            LineString(
                [(longitude, river_latitude), (longitude + 0.01, river_latitude)]
            )
        ],
        crs=4326,
    )
    result: gpd.GeoDataFrame
    distance_to_river_m: float
    result, distance_to_river_m = nearest_amber_river(Point(0, latitude), rivers)
    assert (len(result) == 1) == (distance_m <= 250)
    assert distance_to_river_m == pytest.approx(distance_m, abs=1e-6)


def test_amber_nearest_in_meters() -> None:
    """Choose the nearest river in meters at high latitude."""
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 70.0018), (0.01, 70.0018)]),
            LineString([(0.003, 70), (0.013, 70)]),
        ],
        index=pd.Index([1, 2]),
        crs=4326,
    )
    result: gpd.GeoDataFrame
    distance_to_river_m: float
    result, distance_to_river_m = nearest_amber_river(Point(0, 70), rivers)
    assert result.index.tolist() == [2]


def test_amber_empty_rivers() -> None:
    """Allow regions without represented rivers."""
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(geometry=[], crs=4326)
    result: gpd.GeoDataFrame
    distance_m: float
    result, distance_m = nearest_amber_river(Point(0, 52), rivers)
    assert result.empty
    assert pd.isna(distance_m)


def test_amber_invalid_crs() -> None:
    """Reject coordinates that are not WGS84 degrees."""
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(geometry=[], crs=3857)
    with pytest.raises(ValueError, match="EPSG:4326"):
        nearest_amber_river(Point(0, 52), rivers)


def test_amber_exact_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a distance of exactly 250 m without adding a tolerance.

    Args:
        monkeypatch: Fixture for an exact projected distance.
    """
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=[LineString([(0.002, 0), (0.003, 0)])], crs=4326
    )
    monkeypatch.setattr(
        "geb.build.workflows.weirs.shapely.distance",
        Mock(return_value=np.array([250.0])),
    )
    result: gpd.GeoDataFrame
    distance_m: float
    result, distance_m = nearest_amber_river(Point(0, 0), rivers)
    assert len(result) == 1
    assert distance_m == 250.0


@pytest.mark.parametrize("latitude", [-60.0, 0.0, 52.0, 70.0, 89.9])
def test_direct_projection_matches_geopandas(latitude: float) -> None:
    """Keep the same distances and river selection as the original CRS conversion.

    Args:
        latitude: Latitude of the barrier (degrees).

    Returns:
        None.

    Raises:
        AssertionError: If direct projection changes the distance or selected river.
    """  # noqa: DOC202, DOC502
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=[
            LineString([(0.0005, latitude - 0.0001), (0.001, latitude + 0.0002)]),
            LineString([(0.002, latitude - 0.0002), (0.003, latitude + 0.0001)]),
        ],
        crs=4326,
    )
    reference_distances: pd.Series = rivers.to_crs(
        f"+proj=aeqd +lat_0={latitude} +lon_0=0 +datum=WGS84 +units=m"
    ).distance(Point(0, 0))
    selected: gpd.GeoDataFrame
    distance_m: float
    selected, distance_m = nearest_amber_river(Point(0, latitude), rivers)
    assert selected.index.tolist() == [reference_distances.idxmin()]
    assert distance_m == float(reference_distances.min())
