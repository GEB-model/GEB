"""Tests for GDW reservoir matching and classification."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point, box

from geb.build.workflows.waterbodies_preprocessing import (
    GDW_ID_OFFSET,
    _add_missing_gdw_reservoirs,
    enrich_waterbodies,
    snap_waterbody_points,
)
from geb.hydrology.waterbodies import LAKE, RESERVOIR


@pytest.mark.parametrize(
    "capacity,has_polygon,dam_type",
    [
        (1000.0, True, "Dam"),
        (1000.0, False, "Dam"),
        (1000.0, True, "Lake Control Dam"),
        (0.0, True, "Dam"),
        (1000.0, True, "Lock"),
    ],
)
def test_add_reservoir(capacity: float, has_polygon: bool, dam_type: str) -> None:
    """Add supported GDW reservoirs when geometry and model parameters are available.

    Args:
        capacity: GDW capacity (m3).
        has_polygon: Whether a GDW outline is available.
        dam_type: GDW structure type.
    """
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"waterbody_id": [1], "waterbody_type": [1]},
        geometry=[box(5, 5, 6, 6)],
        crs=4326,
    )
    dams: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "gdw_id": [42],
            "waterbody_id": [np.nan],
            "dam_type": [dam_type],
            "lake_control": ["Yes"],
            "capacity_m3": [capacity],
            "area_m2": [100.0],
            "average_discharge_m3_per_s": [2.0],
        },
        index=[42],
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
    if dam_type == "Lock":
        assert len(result) == 1
        assert checks.iloc[0].addition_reason == "review_barrier_type"
    elif capacity == 0:
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
        assert reservoir.geometry.geom_type == ("Polygon" if has_polygon else "Point")


@pytest.mark.parametrize("case", ["valid", "occupied", "distant"])
@pytest.mark.parametrize("dask_backed", [False, True])
def test_snap_reservoir(case: str, dask_backed: bool) -> None:
    """Skip occupied or distant cells with eager and Dask-backed drainage areas.

    Args:
        case: Whether the first reservoir has a valid, occupied, or distant cell.
        dask_backed: Whether drainage areas are lazily loaded, as in regional builds.
    """
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"waterbody_id": [GDW_ID_OFFSET + 42, GDW_ID_OFFSET + 43], "gdw_id": [42, 43]},
        geometry=[Point(1 if case == "distant" else 0.01, 0.01), Point(0, -0.1)],
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
    if dask_backed:
        upstream_area = upstream_area.chunk({"y": 1, "x": 1})
    skipped: dict[int, str] = snap_waterbody_points(
        waterbodies, waterbody_id, rivers, upstream_area, upstream_area
    )
    assert waterbody_id.values[1, 0] == GDW_ID_OFFSET + 43
    assert waterbodies.loc[1, "geometry"] == Point(0, -0.1)
    if case == "valid":
        assert skipped == {}
        assert len(waterbodies) == 2
        assert waterbody_id.values[0, 0] == GDW_ID_OFFSET + 42
        assert waterbodies.loc[0, "geometry"] == Point(0, 0)
    else:
        assert skipped == {
            42: "occupied_waterbody_cell" if case == "occupied" else "could_not_snap"
        }
        assert waterbodies.waterbody_id.tolist() == [GDW_ID_OFFSET + 43]
        assert waterbody_id.values[0, 0] == (1 if case == "occupied" else -1)


@pytest.mark.parametrize(
    "source_type,dam_type,case,expected_type,capacity_updated",
    [
        (1, "Dam", "id", RESERVOIR, True),
        (1, "Lake Control Dam", "id", RESERVOIR, True),
        (1, "Dam", "unknown_id", RESERVOIR, True),
        (1, "Dam", "conflicting_id", LAKE, False),
        (1, "Dam", "outside", LAKE, False),
        (1, "Dam", "boundary", RESERVOIR, True),
        (1, "Dam", "multiple", RESERVOIR, False),
        (1, "Dam", "missing_attributes", RESERVOIR, False),
        (1, "Dam", "no_control", RESERVOIR, True),
        (1, "Dam", "zero_capacity", RESERVOIR, False),
        (1, "Dam", "nan_capacity", RESERVOIR, False),
        (1, "Lock", "id", LAKE, False),
        (1, "Low Permeable Dam", "id", LAKE, False),
        (2, "Dam", "id", RESERVOIR, False),
        (2, "Lock", "conflicting_id", RESERVOIR, False),
        (3, "Lock", "conflicting_id", RESERVOIR, False),
        (3, "Dam", "outside", RESERVOIR, False),
        (3, "Dam", "multiple", RESERVOIR, False),
        (3, "Dam", "missing_attributes", RESERVOIR, False),
        (3, "Dam", "id", RESERVOIR, True),
    ],
)
def test_waterbody_classification(
    source_type: int,
    dam_type: str,
    case: str,
    expected_type: int,
    capacity_updated: bool,
) -> None:
    """Classify regulated lakes and retain HydroLAKES values when GDW is incomplete.

    Args:
        source_type: Original HydroLAKES type code.
        dam_type: GDW structure type.
        case: Dam match or missing-attribute scenario.
        expected_type: Expected GEB lake or reservoir type.
        capacity_updated: Whether GDW should replace the HydroLAKES volume.
    """
    capacity_m3: float = {"zero_capacity": 0.0, "nan_capacity": np.nan}.get(case, 5.9e9)
    dam_count: int = 2 if case == "multiple" else 1
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "waterbody_id": [1],
            "waterbody_type": [source_type],
            "volume_total": np.array([1000.0], dtype="float32"),
        },
        geometry=[box(0, 0, 1, 1)],
        crs=4326,
    )
    barriers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "gdw_id": list(range(1, dam_count + 1)),
            "hydrolakes_id": [
                None if case == "unknown_id" else 99 if case == "conflicting_id" else 1
            ]
            * dam_count,
            "dam_type": [dam_type] * dam_count,
        },
        geometry=[
            Point(2.0 if case == "outside" else 1.0 if case == "boundary" else 0.5, 0.5)
        ]
        * dam_count,
        crs=4326,
    )
    if case == "multiple":
        barriers.loc[0, "dam_type"] = "Lock"
    if case != "missing_attributes":
        barriers["capacity_m3"] = capacity_m3
        barriers["lake_control"] = "No" if case == "no_control" else "Maybe"
    result: gpd.GeoDataFrame
    checks: gpd.GeoDataFrame
    result, checks = enrich_waterbodies(waterbodies, barriers)
    assert result.waterbody_type.tolist() == [expected_type]
    assert result.volume_total.tolist() == [capacity_m3 if capacity_updated else 1000.0]
    assert result.volume_total.dtype == np.dtype(
        "float64" if capacity_updated else "float32"
    )
    assert result.hydrolakes_type.tolist() == [source_type]
    assert result.hydrolakes_volume_total.tolist() == [1000.0]
    assert checks.changed_to_reservoir.eq(
        source_type != RESERVOIR and expected_type == RESERVOIR
    ).all()


@pytest.mark.parametrize("source_type", [1, 2, 3])
def test_waterbody_types_without_dams(source_type: int) -> None:
    """Resolve source types without dams during the build.

    Args:
        source_type: HydroLAKES type code (unitless).
    """
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "waterbody_id": [1],
            "waterbody_type": [source_type],
            "volume_total": [1000.0],
        },
        geometry=[box(0, 0, 1, 1)],
        crs=4326,
    )
    barriers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "hydrolakes_id": pd.Series(dtype="Int64"),
            "dam_type": pd.Series(dtype=str),
            "lake_control": pd.Series(dtype=str),
            "capacity_m3": pd.Series(dtype=float),
        },
        geometry=[],
        crs=4326,
    )
    result: gpd.GeoDataFrame
    checks: gpd.GeoDataFrame
    result, checks = enrich_waterbodies(waterbodies, barriers)
    assert result.waterbody_type.tolist() == [LAKE if source_type == 1 else RESERVOIR]
    assert result.waterbody_type.dtype == np.int32
    assert result.hydrolakes_type.tolist() == [source_type]
    assert waterbodies.waterbody_type.tolist() == [source_type]
    assert checks.empty


def test_ambiguous_dam_match() -> None:
    """Leave a dam unmatched when its unknown lake ID and shared boundary are ambiguous."""
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "waterbody_id": [1, 2],
            "waterbody_type": [LAKE, LAKE],
            "volume_total": [1000.0, 1000.0],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs=4326,
    )
    dams: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"gdw_id": [42], "hydrolakes_id": [np.nan], "dam_type": ["Dam"]},
        geometry=[Point(1, 0.5)],
        crs=4326,
    )
    result: gpd.GeoDataFrame
    checks: gpd.GeoDataFrame
    result, checks = enrich_waterbodies(waterbodies, dams)
    assert result.waterbody_type.tolist() == [LAKE, LAKE]
    assert checks.match_method.tolist() == ["multiple_lakes"]
    assert checks.waterbody_id.isna().all()
