"""Test GDW weirs outside reservoir outlines."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point

from geb.build.workflows.weirs import create_weir_height_grid


@pytest.mark.parametrize("height_m", [0.0, -1.0, float("inf")])
def test_invalid_default_height(height_m: float) -> None:
    """Reject a default height that cannot be used.

    Args:
        height_m: Invalid default height (m).
    """
    grid: xr.DataArray = xr.DataArray([[-1]])
    points: gpd.GeoDataFrame = gpd.GeoDataFrame()
    with pytest.raises(ValueError, match="crest_height_m"):
        create_weir_height_grid(points, height_m, grid, points, grid, grid, grid)


@pytest.mark.parametrize(
    "case",
    [
        "weir",
        "dam",
        "lock",
        "low_dam",
        "linked",
        "offstream",
        "boundary",
        "occupied",
        "distant",
        "empty",
        "lake_control",
        "inside",
        "height",
        "zero_height",
        "negative_height",
        "infinite_height",
    ],
)
def test_build_weir(case: str) -> None:
    """Use all points outside polygons, whatever their name or type.

    Args:
        case: Selection, overlap, or height scenario.
    """
    barriers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "gdw_id": [42],
            "dam_type": ["Dam"],
            "dam_name": ["Example Weir"],
            "waterbody_id": [np.nan],
            "instream": ["Instream"],
            "lake_control": [None],
            "inside_gdw_polygon": [False],
            "dam_hgt_m": [np.nan],
        },
        geometry=[Point(1 if case == "distant" else 0.01, 0.01)],
        crs=4326,
    )
    if case == "inside":
        barriers["inside_gdw_polygon"] = True
    if case == "height":
        barriers["dam_hgt_m"] = 2.5
    if case == "zero_height":
        barriers["dam_hgt_m"] = 0.0
    if case == "negative_height":
        barriers["dam_hgt_m"] = -99.0
    if case == "infinite_height":
        barriers["dam_hgt_m"] = np.inf
    if case == "linked":
        barriers["waterbody_id"] = 1
    if case == "dam":
        barriers["dam_name"] = None
    if case == "lock":
        barriers["dam_type"] = "Lock"
    if case == "low_dam":
        barriers["dam_type"] = "Low Permeable Dam"
    if case == "offstream":
        barriers["instream"] = "Offstream"
    if case == "lake_control":
        barriers["lake_control"] = "Yes"
    if case == "distant":
        barriers.geometry = [Point(1, 0.01)]
    if case == "empty":
        barriers = gpd.GeoDataFrame()
    upstream_area: xr.DataArray = xr.DataArray(
        np.full((2, 2), 1000.0),
        coords={"y": [0.0, -0.1], "x": [0.0, 0.1]},
        dims=("y", "x"),
    )
    waterbody_ids: xr.DataArray = xr.full_like(upstream_area, -1, dtype=np.int32)
    river_cells: xr.DataArray = xr.full_like(upstream_area, True, dtype=bool)
    if case == "boundary":
        river_cells.values[0, 0] = False
    if case == "occupied":
        waterbody_ids.values[0, 0] = 1
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "represented_in_grid": [True],
            "uparea_m2": [1000.0],
            "hydrography_xy": [[(0, 0), (0, 1)]],
            "shreve_stream_order": [1],
        },
        index=pd.Index([1], dtype="int64"),
        geometry=[LineString([(0, 0), (0, -0.1)])],
        crs=4326,
    )
    result: xr.DataArray = create_weir_height_grid(
        barriers,
        1.0,
        waterbody_ids,
        rivers,
        upstream_area,
        upstream_area,
        river_cells,
    )
    assert result.dtype == np.float32
    if case == "height":
        assert result.values.sum() == 2.5
    elif case in (
        "weir",
        "dam",
        "lock",
        "low_dam",
        "offstream",
        "lake_control",
        "zero_height",
        "negative_height",
        "infinite_height",
    ):
        assert result.values.sum() == 1.0
        assert result.values[0, 0] == 1.0
    else:
        assert result.values.sum() == 0.0
