"""Tests for GDW reservoir outlines and weir classification."""

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


@pytest.mark.parametrize("longitude, covered", [(0.5, True), (1.0, True), (2.0, False)])
def test_gdw_polygon_coverage(longitude: float, covered: bool) -> None:
    """Use actual coverage, including edges, regardless of polygon ID.

    Args:
        longitude: Point longitude (degrees).
        covered: Whether the point should count as inside a polygon.
    """
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"waterbody_id": [1], "waterbody_type": [1]},
        geometry=[box(0, 0, 1, 1)],
        crs=4326,
    )
    dams: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"gdw_id": [42], "waterbody_id": [1]},
        geometry=[Point(longitude, 0.5)],
        crs=4326,
    )
    outlines: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"gdw_id": [99]},
        geometry=[box(0, 0, 1, 1)],
        crs=4326,
    )
    _, checks = _add_missing_gdw_reservoirs(waterbodies, dams, outlines)
    assert bool(checks.iloc[0].inside_gdw_polygon) == covered


@pytest.mark.parametrize("capacity", [1000.0, 0.0])
@pytest.mark.parametrize("has_polygon", [True, False])
def test_add_reservoir(capacity: float, has_polygon: bool) -> None:
    """Only add reservoirs covered by GDW polygons with valid capacity."""
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
    if not has_polygon:
        assert len(result) == 1
        assert checks.iloc[0].addition_reason == "weir"
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


@pytest.mark.parametrize(
    "dam_type,lake_control",
    [
        (dam_type, flag)
        for dam_type in ("Dam", "Lake Control Dam")
        for flag in (None, "", "Yes", "Enlarged", "Maybe")
    ],
)
@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "natural",
        "zero_capacity",
        "nan_capacity",
        "outside",
        "id_conflict",
        "multiple",
        "unknown_flag",
        "no_control",
        "lock",
    ],
)
@pytest.mark.parametrize("volume_dtype", ["float32", "float64"])
def test_controlled_lake_conversion(
    dam_type: str, lake_control: str | None, case: str, volume_dtype: str
) -> None:
    """Convert recognized controlled lakes while retaining matching safeguards.

    Args:
        dam_type: GDW barrier classification.
        lake_control: GDW lake-control flag, including uncertain classifications.
        case: Matching or capacity scenario to verify.
        volume_dtype: Input storage dtype.

    Returns:
        None.

    Raises:
        AssertionError: If conversion or capacity precision is incorrect.
    """  # noqa: DOC202, DOC502
    original_type: int = 1 if case == "natural" else 3
    capacity_m3: float = {"zero_capacity": 0.0, "nan_capacity": np.nan}.get(case, 5.9e9)
    dam_count: int = 2 if case == "multiple" else 1
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "waterbody_id": [1],
            "waterbody_type": [original_type],
            "volume_total": np.array([1000.0], dtype=volume_dtype),
        },
        geometry=[box(0, 0, 1, 1)],
        crs=4326,
    )
    barriers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "gdw_id": list(range(dam_count)),
            "hydrolakes_id": [99 if case == "id_conflict" else 1] * dam_count,
            "dam_type": ["Lock" if case == "lock" else dam_type] * dam_count,
            "lake_control": [
                "Unknown"
                if case == "unknown_flag"
                else "No"
                if case == "no_control"
                else lake_control
            ]
            * dam_count,
            "capacity_m3": [capacity_m3] * dam_count,
        },
        geometry=[Point(2 if case == "outside" else 0.5, 0.5)] * dam_count,
        crs=4326,
    )
    result: gpd.GeoDataFrame
    checks: gpd.GeoDataFrame
    result, checks = enrich_waterbodies(waterbodies, barriers)
    converted: bool = case == "valid"
    assert result.iloc[0].waterbody_type == (RESERVOIR if converted else LAKE)
    assert result.iloc[0].volume_total == (capacity_m3 if converted else 1000.0)
    assert result.volume_total.dtype == (
        np.dtype("float64") if converted else np.dtype(volume_dtype)
    )
    assert result.iloc[0].hydrolakes_type == original_type
    assert result.iloc[0].hydrolakes_volume_total == 1000.0
    assert checks.changed_to_reservoir.eq(converted).all()
    if lake_control == "Maybe" and case not in {"unknown_flag", "no_control"}:
        assert checks.type_check.eq("uncertain_lake_control").all()


@pytest.mark.parametrize("source_type", [1, 2, 3])
def test_waterbody_types_without_dams(source_type: int) -> None:
    """Resolve source types without dams during the build.

    Args:
        source_type: HydroLAKES type code (unitless).

    Returns:
        None.

    Raises:
        AssertionError: If final types or source records are incorrect.
    """  # noqa: DOC202, DOC502
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
    assert result.waterbody_type.tolist() == [RESERVOIR if source_type == 2 else LAKE]
    assert result.waterbody_type.dtype == np.int32
    assert result.hydrolakes_type.tolist() == [source_type]
    assert waterbodies.waterbody_type.tolist() == [source_type]
    assert checks.empty


@pytest.mark.parametrize("source_type", [0, 4, np.nan])
def test_invalid_source_waterbody_type(source_type: float) -> None:
    """Reject invalid source types even when there are no matching dams.

    Args:
        source_type: Invalid HydroLAKES type code (unitless).

    Returns:
        None.

    Raises:
        AssertionError: If invalid source types are accepted.
    """  # noqa: DOC202, DOC502
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"waterbody_id": [1], "waterbody_type": [source_type]},
        geometry=[box(0, 0, 1, 1)],
        crs=4326,
    )
    with pytest.raises(ValueError, match="HydroLAKES"):
        enrich_waterbodies(waterbodies, gpd.GeoDataFrame())


@pytest.mark.parametrize(
    "case", ["id", "outside", "point", "boundary", "ambiguous", "unmatched"]
)
def test_dam_matching(case: str) -> None:
    """Check ID precedence, lake edges, and ambiguous spatial matches.

    Args:
        case: Dam location and recorded-ID scenario.

    Returns:
        None.

    Raises:
        AssertionError: If matching or reservoir classification is incorrect.
    """  # noqa: DOC202, DOC502
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "waterbody_id": [1, 2],
            "waterbody_type": [3, 3],
            "volume_total": [1000.0, 1000.0],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs=4326,
    )
    point_x: float = {
        "outside": 2.5,
        "boundary": 0.0,
        "ambiguous": 1.0,
        "unmatched": 2.5,
    }.get(case, 0.5)
    barriers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "gdw_id": [42],
            "hydrolakes_id": [1 if case in {"id", "outside"} else np.nan],
            "dam_type": ["Dam"],
            "lake_control": [None],
            "capacity_m3": [2000.0],
        },
        geometry=[Point(point_x, 0.5)],
        crs=4326,
    )
    original_waterbodies: gpd.GeoDataFrame = waterbodies.copy(deep=True)
    original_barriers: gpd.GeoDataFrame = barriers.copy(deep=True)
    result: gpd.GeoDataFrame
    checks: gpd.GeoDataFrame
    result, checks = enrich_waterbodies(waterbodies, barriers)
    expected_method: str = {
        "id": "id",
        "outside": "id",
        "point": "point_in_polygon",
        "boundary": "point_in_polygon",
        "ambiguous": "multiple_lakes",
        "unmatched": "unmatched",
    }[case]
    assert checks.match_method.tolist() == [expected_method]
    assert checks.point_outside_lake.tolist() == [case == "outside"]
    assert result.waterbody_type.tolist() == [
        RESERVOIR if case in {"id", "point", "boundary"} else LAKE,
        LAKE,
    ]
    if case in {"ambiguous", "unmatched"}:
        assert checks.waterbody_id.isna().all()
    else:
        assert checks.waterbody_id.tolist() == [1]
    pd.testing.assert_frame_equal(waterbodies, original_waterbodies)
    pd.testing.assert_frame_equal(barriers, original_barriers)


@pytest.mark.parametrize("lake_ids", [[1, 1], [1, None]])
def test_invalid_lake_ids(lake_ids: list[int | None]) -> None:
    """Reject missing or duplicate lake IDs before matching dams.

    Args:
        lake_ids: Invalid source IDs (unitless).

    Returns:
        None.

    Raises:
        AssertionError: If invalid lake IDs are accepted.
    """  # noqa: DOC202, DOC502
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"waterbody_id": lake_ids, "waterbody_type": [1, 1]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs=4326,
    )
    with pytest.raises(ValueError, match="IDs must be present and unique"):
        enrich_waterbodies(waterbodies, gpd.GeoDataFrame())
