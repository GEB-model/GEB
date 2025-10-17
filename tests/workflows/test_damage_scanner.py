"""Tests for damage scanner workflow."""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray as rxr  # noqa: F401
import xarray as xr
from pytest import fixture
from shapely.geometry import Polygon

from geb.workflows.damage_scanner import VectorScanner


@fixture
def flood_raster() -> xr.DataArray:
    """Flood hazard raster for testing the VectorScanner.

    Returns:
        A DataArray representing flood hazard severities.
    """
    x = np.zeros((10, 10), dtype=np.float32)
    x[:5, :5] = 0.5
    x[:5, 5:] = 3
    x[5:, :5] = 0
    x[5:, 5:] = np.nan
    flood_raster = xr.DataArray(
        x,
        coords={
            "x": np.arange(10, dtype=np.int32) + 0.5,  # 0.5 for center of the pixel
            "y": np.arange(10, dtype=np.int32) + 0.5,  # 0.5 for center of the pixel
        },
        dims=["y", "x"],
    )
    flood_raster.attrs["crs"] = "EPSG:32631"
    flood_raster.attrs["_FillValue"] = np.nan
    return flood_raster


@fixture
def buildings() -> gpd.GeoDataFrame:
    """Building features for testing the VectorScanner.

    Returns:
        A GeoDataFrame containing building geometries and attributes.
    """
    data = {
        "object_type": [
            "residential",
            "residential",
            "commercial",
            "residential",
            "residential",
            "residential",
            "residential",
            "residential",
        ],
        "maximum_damage": [100, 200, 300, 100, 100, 100, 100, 100],
    }
    polygons = [
        Polygon([(1, 1), (1, 2), (2, 2), (2, 1)]),
        Polygon([(1, 1), (1, 3), (3, 3), (3, 1)]),
        Polygon([(1, 1), (1, 3), (3, 3), (3, 1)]),
        Polygon([(8, 1), (8, 2), (9, 2), (9, 1)]),
        Polygon([(4.5, 1), (4.5, 2), (5.5, 2), (5.5, 1)]),
        Polygon([(1, 8), (1, 9), (2, 9), (2, 8)]),
        Polygon([(8, 8), (8, 9), (9, 9), (9, 8)]),
        Polygon([(4.5, 4.5), (4.5, 5.5), (5.5, 5.5), (5.5, 4.5)]),
    ]
    data["geometry"] = polygons

    gdf = gpd.GeoDataFrame(data, crs="EPSG:32631", geometry="geometry")
    return gdf


@fixture
def vulnerability_curves() -> pd.DataFrame:
    """Vulnerability curves for testing the VectorScanner.

    Returns:
        A DataFrame containing vulnerability curves for residential and commercial buildings.
    """
    return pd.DataFrame(
        {
            "residential": [0.0, 0.2, 0.3],
            "commercial": [0.0, 0.4, 0.6],
        },
        index=[0, 1, 2],
    )


@pytest.mark.parametrize("clip", [False, True])
def test_vector_scanner(
    flood_raster: xr.DataArray,
    buildings: gpd.GeoDataFrame,
    vulnerability_curves: pd.DataFrame,
    clip: bool,
) -> None:
    """Test the VectorScanner for calculating damages to buildings based on flood hazard.

    This test verifies that the VectorScanner correctly computes damage values for
    various building features under different flood hazard severities and building
    types. It checks damage calculations for residential and commercial buildings
    with different areas, hazard intensities, and clipping scenarios.

    The test includes assertions for expected damage values based on predefined
    vulnerability curves and hazard data. It covers cases with uniform hazard,
    mixed hazards, no hazard, and NaN hazard values.

    Args:
        flood_raster: The flood hazard raster data.
        buildings: GeoDataFrame containing building geometries and attributes.
        vulnerability_curves: DataFrame with vulnerability curves for damage calculation.
        clip: Boolean flag to clip the flood raster to a specific polygon.
    """
    if clip:
        flood_raster = flood_raster.rio.clip(
            [Polygon([(1, 1), (1, 9), (9, 9), (9, 1)])]
        )

    damage = VectorScanner(
        features=buildings,
        hazard=flood_raster,
        vulnerability_curves=vulnerability_curves,
    )

    assert math.isclose(
        damage.iloc[0], 10.0
    )  # 1m2, .5 hazard severity, residential, max_damage 100 > 0.1 damage ratio > damage of 10
    assert math.isclose(
        damage.iloc[1], 80.0
    )  # 4m2, .5 hazard severity, residential, max_damage 200 > 0.1 damage ratio > damage of 80
    assert math.isclose(
        damage.iloc[2], 240.0
    )  # 4m2, .5 hazard severity, commercial, max_damage 300 > 0.2 damage ratio > damage of 240
    assert math.isclose(
        damage.iloc[3], 30.0
    )  # 1m2, 3 hazard severity, residential, max_damage 100 > 0.3 damage ratio (max of curve) > damage of 30

    # 1m2, half 0.5 hazard severity + half 3 hazard severity, residential, max_damage 100
    # > 0.1 + 0.3 damage ratio (max of curve) > damage of 30
    assert math.isclose(damage.iloc[4], 20.0)

    assert math.isclose(
        damage.iloc[5], 0.0
    )  # 1m2, no hazard severity, residential, max_damage 100
    assert math.isclose(
        damage.iloc[6], 0.0
    )  # 1m2, hazard is nan, residential, max_damage 100

    assert math.isclose(
        damage.iloc[7], 10.0
    )  # 1m2, 0.25 x 0, 0.25 x nan, 0.25 x 0.5, 0.25 x 3 hazard severity, residential, max_damage 100 > damage of 10


def test_vector_scanner_missing_data(
    flood_raster: xr.DataArray,
    buildings: gpd.GeoDataFrame,
    vulnerability_curves: pd.DataFrame,
) -> None:
    """Test the VectorScanner's handling of missing or incorrect data.

    This test checks that the VectorScanner raises appropriate errors when
    required data is missing or incorrect. It verifies the following scenarios:

    - Missing 'maximum_damage' column in the buildings GeoDataFrame.
    - Missing 'object_type' column in the buildings GeoDataFrame.
    - Object types in the buildings GeoDataFrame that are not present in the
      vulnerability curves DataFrame.

    Args:
        flood_raster: The flood hazard raster data.
        buildings: GeoDataFrame containing building geometries and attributes.
        vulnerability_curves: DataFrame with vulnerability curves for damage calculation.
    """
    # Test missing 'maximum_damage' column
    buildings_missing_damage = buildings.drop(columns=["maximum_damage"])
    with pytest.raises(AssertionError):
        VectorScanner(
            features=buildings_missing_damage,
            hazard=flood_raster,
            vulnerability_curves=vulnerability_curves,
        )

    # Test missing 'object_type' column
    buildings_missing_type = buildings.drop(columns=["object_type"])
    with pytest.raises(AssertionError):
        VectorScanner(
            features=buildings_missing_type,
            hazard=flood_raster,
            vulnerability_curves=vulnerability_curves,
        )

    # Test object types not in vulnerability curves
    buildings_wrong_type = buildings.copy()
    buildings_wrong_type["object_type"] = "non_existent"
    with pytest.raises(AssertionError):
        VectorScanner(
            features=buildings_wrong_type,
            hazard=flood_raster,
            vulnerability_curves=vulnerability_curves,
        )
