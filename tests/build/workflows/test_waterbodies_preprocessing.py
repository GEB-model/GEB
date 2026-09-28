"""Tests for GDW reservoirs without outlines."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point, box

from geb.build.workflows.waterbodies_preprocessing import (
    GDW_ID_OFFSET,
    _add_missing_gdw_reservoirs,
    snap_waterbody_points,
)
from geb.hydrology.waterbodies import RESERVOIR


@pytest.mark.parametrize("capacity", [1000.0, 0.0])
@pytest.mark.parametrize("has_polygon", [True, False])
def test_add_reservoir(capacity: float, has_polygon: bool) -> None:
    """Add valid reservoirs with or without an outline; skip missing capacity."""
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"waterbody_id": [1], "waterbody_type": [1]},
        geometry=[box(5, 5, 6, 6)],
        crs=4326,
    )
    dams: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "gdw_id": [42],
            "waterbody_id": [np.nan],
            "dam_type": ["Dam"],
            "lake_control": [None],
            "capacity_m3": [capacity],
            "area_m2": [100.0],
            "average_discharge_m3_per_s": [2.0],
        },
        geometry=[Point(0.01, 0)],
        crs=4326,
    )
    outlines: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"gdw_id": [42] if has_polygon else []},
        geometry=[box(-0.01, -0.01, 0.01, 0.01)] if has_polygon else [],
        crs=4326,
    )
    result: gpd.GeoDataFrame
    checks: gpd.GeoDataFrame
    result, checks = _add_missing_gdw_reservoirs(waterbodies, dams, outlines)
    if capacity == 0:
        assert len(result) == 1
        assert checks.iloc[0].addition_reason == "missing_model_values"
    else:
        reservoir: pd.Series = result.iloc[-1]
        assert reservoir.waterbody_id == GDW_ID_OFFSET + 42
        assert reservoir.waterbody_type == RESERVOIR
        assert reservoir.volume_total == capacity
        assert reservoir.average_area == 100.0
        assert reservoir.average_discharge == 2.0
        assert checks.iloc[0].match_method == (
            "gdw_polygon" if has_polygon else "gdw_point"
        )


@pytest.mark.parametrize("case", ["valid", "occupied", "distant"])
def test_snap_reservoir(case: str) -> None:
    """Use the real river snapper and reject occupied or distant cells."""
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"waterbody_id": [GDW_ID_OFFSET + 42], "gdw_id": [42]},
        geometry=[Point(1 if case == "distant" else 0.01, 0.01)],
        crs=4326,
    )
    upstream_area: xr.DataArray = xr.DataArray(
        np.full((2, 2), 1000.0),
        coords={"y": [0.0, -0.1], "x": [0.0, 0.1]},
        dims=("y", "x"),
    )
    waterbody_id: xr.DataArray = xr.full_like(upstream_area, -1, dtype=np.int32)
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
    if case == "occupied":
        waterbody_id.values[0, 0] = 1
    if case != "valid":
        with pytest.raises(
            ValueError, match="occupied" if case == "occupied" else "Could not snap"
        ):
            snap_waterbody_points(
                waterbodies, waterbody_id, rivers, upstream_area, upstream_area
            )
    else:
        snap_waterbody_points(
            waterbodies, waterbody_id, rivers, upstream_area, upstream_area
        )
        assert waterbody_id.values[0, 0] == GDW_ID_OFFSET + 42
        assert np.count_nonzero(waterbody_id.values != -1) == 1
        assert waterbodies.geometry.iloc[0] == Point(0, 0)
