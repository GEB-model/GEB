"""Tests for GDW matching, type checks, and missing reservoir outlines."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from shapely.geometry import Point, box

from geb.build.workflows.waterbodies_preprocessing import (
    GDW_ID_OFFSET,
    enrich_waterbodies,
)


@pytest.mark.parametrize("control", [None, "Yes", "Maybe", "Enlarged"])
def test_controlled_lake_as_reservoir(
    lakes: gpd.GeoDataFrame,
    barriers: gpd.GeoDataFrame,
    control: str | None,
) -> None:
    """Use GDW capacity for controlled lakes with a clear dam match.

    Args:
        lakes: HydroLAKES test polygons.
        barriers: GDW test points.
        control: GDW lake-control flag to check.
    """
    barriers = barriers.iloc[[2]].copy()
    barriers["dam_type"] = "Dam"
    barriers["lake_control"] = control
    barriers.geometry = [Point(2.5, 0.5)]
    result: gpd.GeoDataFrame
    dam_checks: gpd.GeoDataFrame
    result, dam_checks = enrich_waterbodies(lakes, barriers)
    assert result.loc[2, "hydrolakes_type"] == 3
    assert result.loc[2, "hydrolakes_volume_total"] == 300
    assert result.loc[2, "waterbody_type"] == (2 if control is None else 3)
    assert result.loc[2, "volume_total"] == (900 if control is None else 300)
    assert bool(dam_checks.loc[0, "changed_to_reservoir"]) == (control is None)


@pytest.mark.parametrize(
    "problem", ["outside", "wrong_id", "missing_capacity", "multiple_dams"]
)
def test_uncertain_controlled_lake_keeps_type(
    lakes: gpd.GeoDataFrame,
    barriers: gpd.GeoDataFrame,
    problem: str,
) -> None:
    """Keep controlled lakes unchanged when the dam match or capacity is uncertain.

    Args:
        lakes: HydroLAKES test polygons.
        barriers: GDW test points.
        problem: Reason the type should not change.
    """
    barriers = barriers.iloc[[2]].copy()
    barriers["dam_type"] = "Dam"
    barriers["lake_control"] = None
    barriers.geometry = [Point(2.5, 0.5)]
    if problem == "outside":
        barriers.geometry = [Point(9, 9)]
    elif problem == "wrong_id":
        barriers["hydrolakes_id"] = 999
    elif problem == "missing_capacity":
        barriers["capacity_m3"] = np.nan
    else:
        barriers = pd.concat([barriers, barriers], ignore_index=True)
        barriers["gdw_id"] = [3, 7]
    result: gpd.GeoDataFrame
    dam_checks: gpd.GeoDataFrame
    result, dam_checks = enrich_waterbodies(lakes, barriers)
    assert result.loc[2, "waterbody_type"] == 3
    assert result.loc[2, "volume_total"] == 300
    assert not dam_checks["changed_to_reservoir"].any()


@pytest.fixture
def lakes() -> gpd.GeoDataFrame:
    """Return three adjacent HydroLAKES polygons, one for each type.

    Returns:
        Lake records with model values in m2, m3, and m3/s.
    """
    return gpd.GeoDataFrame(
        {
            "waterbody_id": [10, 20, 30],
            "waterbody_type": [1, 2, 3],
            "volume_total": [100.0, 200.0, 300.0],
            "average_area": [10.0, 20.0, 30.0],
            "average_discharge": [1.0, 2.0, 3.0],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs=4326,
    )


@pytest.fixture
def barriers() -> gpd.GeoDataFrame:
    """Return barriers covering ID, location, boundary, and missing matches.

    Returns:
        GDW points with capacity in m3, area in m2, and discharge in m3/s.
    """
    return gpd.GeoDataFrame(
        {
            "gdw_id": [1, 2, 3, 4, 5, 6],
            "hydrolakes_id": pd.array([10, None, 30, None, None, None], dtype="Int64"),
            "dam_type": ["Dam", "Dam", "Lake Control Dam", "Dam", "Dam", "Dam"],
            "lake_control": [None, None, "Yes", None, None, None],
            "capacity_m3": [900.0] * 6,
            "area_m2": [50.0] * 6,
            "average_discharge_m3_per_s": [5.0] * 6,
        },
        geometry=[
            Point(0.5, 0.5),
            Point(1.5, 0.5),
            Point(9, 9),
            Point(1, 0.5),
            Point(5, 5),
            Point(0, 0.5),
        ],
        crs=4326,
    )


def test_enrichment_keeps_lake_values(
    lakes: gpd.GeoDataFrame, barriers: gpd.GeoDataFrame
) -> None:
    """Check lake values, repeated matches, and dam locations.

    Args:
        lakes: HydroLAKES test polygons.
        barriers: GDW test points.
    """
    result: gpd.GeoDataFrame
    dam_checks: gpd.GeoDataFrame
    result, dam_checks = enrich_waterbodies(lakes, barriers)
    assert_frame_equal(result[lakes.columns], lakes)
    assert result["gdw_count"].tolist() == [2, 1, 1]
    assert pd.isna(result.loc[0, "gdw_capacity_m3"])
    assert result.loc[1, "gdw_capacity_m3"] == 900
    assert result["gdw_type_conflict"].tolist() == [False, False, False]
    assert dam_checks["match_method"].tolist() == [
        "id",
        "point_in_polygon",
        "id",
        "multiple_lakes",
        "unmatched",
        "point_in_polygon",
    ]
    assert dam_checks.loc[2, "point_outside_lake"]
    assert dam_checks.loc[0, "type_check"] == "type_1_with_barrier"
    assert dam_checks.loc[2, "type_check"] == "agree"


@pytest.mark.parametrize(
    ("lake_type", "dam_type", "control", "expected"),
    [
        (1, "Dam", "", "type_1_with_barrier"),
        (1, "Lock", "", "type_1_with_barrier"),
        (1, "Low Permeable Dam", "", "type_1_with_barrier"),
        (1, "Weir", "", "type_1_with_barrier"),
        (1, "Dam", "Maybe", "uncertain_lake_control"),
        (1, "Dam", "Yes", "lake_control_differs"),
        (1, "Dam", "Enlarged", "lake_control_differs"),
        (1, "Lake Control Dam", "", "lake_control_differs"),
        (2, "Dam", "Maybe", "uncertain_lake_control"),
        (2, "Lake Control Dam", "", "lake_control_differs"),
        (3, "Lake Control Dam", "", "agree"),
        (2, "Dam", "", "agree"),
        (2, "Dam", "Yes", "lake_control_differs"),
        (3, "Lake Control Dam", "Yes", "agree"),
        (3, "Dam", "Enlarged", "agree"),
        (3, "Dam", "Maybe", "uncertain_lake_control"),
        (3, "Dam", "", "controlled_lake_with_dam"),
        (3, "Lock", "", "review_barrier_type"),
        (3, "Dam", "Unknown", "review_barrier_type"),
        (2, "Lock", "", "review_barrier_type"),
    ],
)
def test_type_checks(
    lakes: gpd.GeoDataFrame,
    barriers: gpd.GeoDataFrame,
    lake_type: int,
    dam_type: str,
    control: str,
    expected: str,
) -> None:
    """Check whether lake and dam types agree.

    Args:
        lakes: HydroLAKES test polygons.
        barriers: GDW test points.
        lake_type: HydroLAKES type.
        dam_type: GDW barrier label.
        control: GDW lake-control flag.
        expected: Expected QC label.
    """
    lakes = lakes.iloc[[0]].copy()
    lakes["waterbody_type"] = lake_type
    barriers = barriers.iloc[[0]].copy()
    barriers["dam_type"] = dam_type
    barriers["lake_control"] = control
    waterbodies, dam_checks = enrich_waterbodies(lakes, barriers)
    assert dam_checks.loc[0, "type_check"] == expected
    assert bool(waterbodies.loc[0, "gdw_type_conflict"]) == (
        expected == "lake_control_differs"
    )


def test_invalid_type(lakes: gpd.GeoDataFrame, barriers: gpd.GeoDataFrame) -> None:
    """Reject an unknown HydroLAKES type during matching.

    Args:
        lakes: HydroLAKES test polygons.
        barriers: GDW test points.
    """
    lakes = lakes.iloc[[0]].copy()
    lakes["waterbody_type"] = 4
    with pytest.raises(ValueError, match="Unknown HydroLAKES type"):
        enrich_waterbodies(lakes, barriers.iloc[[0]])


def test_empty_inputs(lakes: gpd.GeoDataFrame, barriers: gpd.GeoDataFrame) -> None:
    """Handle regions without either lakes or barriers.

    Args:
        lakes: HydroLAKES test polygons.
        barriers: GDW test points.
    """
    result: gpd.GeoDataFrame
    dam_checks: gpd.GeoDataFrame
    result, dam_checks = enrich_waterbodies(lakes, barriers.iloc[:0])
    assert result["gdw_count"].eq(0).all()
    assert dam_checks.empty
    result, dam_checks = enrich_waterbodies(lakes.iloc[:0], barriers)
    assert result.empty
    assert dam_checks["match_method"].eq("unmatched").all()


def test_duplicate_ids(lakes: gpd.GeoDataFrame, barriers: gpd.GeoDataFrame) -> None:
    """Reject duplicate lake IDs before joining.

    Args:
        lakes: HydroLAKES test polygons.
        barriers: GDW test points.
    """
    lakes.loc[1, "waterbody_id"] = 10
    with pytest.raises(ValueError, match="Waterbody IDs"):
        enrich_waterbodies(lakes, barriers)


def test_add_missing_gdw_reservoir(
    lakes: gpd.GeoDataFrame, barriers: gpd.GeoDataFrame
) -> None:
    """Add a missing reservoir while rejecting an overlapping polygon.

    Args:
        lakes: HydroLAKES test polygons.
        barriers: GDW test points.
    """
    result: gpd.GeoDataFrame
    dam_checks: gpd.GeoDataFrame
    reservoir_shapes: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"gdw_id": [4, 5]},
        geometry=[box(0.5, 0, 1.5, 1), box(4, 4, 6, 6)],
        crs=4326,
    )
    result, dam_checks = enrich_waterbodies(lakes, barriers, reservoir_shapes)
    assert len(result) == 4
    assert result.iloc[-1]["waterbody_id"] == GDW_ID_OFFSET + 5
    assert result.iloc[-1]["volume_total"] == 900
    assert result.iloc[-1]["waterbody_source"] == "gdw"
    assert result["waterbody_id"].dtype == np.int32
    assert dam_checks.loc[3, "addition_reason"] == "overlaps_waterbody"
    assert dam_checks.loc[4, "addition_reason"] == "added"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("capacity_m3", np.nan, "missing_model_values"),
        ("average_discharge_m3_per_s", 0, "missing_model_values"),
        ("dam_type", "Lock", "review_barrier_type"),
        ("lake_control", "Maybe", "review_barrier_type"),
        ("lake_control", "Yes", "review_barrier_type"),
    ],
)
def test_new_reservoir_needs_model_values(
    lakes: gpd.GeoDataFrame,
    barriers: gpd.GeoDataFrame,
    field: str,
    value: float | str,
    reason: str,
) -> None:
    """Record why GDW reservoirs were skipped when HydroLAKES has no lakes.

    Args:
        lakes: HydroLAKES test polygons.
        barriers: GDW test points.
        field: Field to change.
        value: Invalid or uncertain value.
        reason: Expected exclusion reason.
    """
    barriers = barriers.iloc[[4]].copy()
    barriers[field] = value
    result: gpd.GeoDataFrame
    dam_checks: gpd.GeoDataFrame
    reservoir_shapes: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"gdw_id": [5]}, geometry=[box(4, 4, 6, 6)], crs=4326
    )
    result, dam_checks = enrich_waterbodies(lakes.iloc[:0], barriers, reservoir_shapes)
    assert result.empty
    assert dam_checks.loc[0, "addition_reason"] == reason


def test_gdw_without_hydrolakes(
    lakes: gpd.GeoDataFrame, barriers: gpd.GeoDataFrame
) -> None:
    """Use a GDW polygon in a region with no HydroLAKES polygon.

    Args:
        lakes: HydroLAKES test polygons.
        barriers: GDW test points.
    """
    result: gpd.GeoDataFrame
    dam_checks: gpd.GeoDataFrame
    reservoir_shapes: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"gdw_id": [5]}, geometry=[box(4, 4, 6, 6)], crs=4326
    )
    result, dam_checks = enrich_waterbodies(
        lakes.iloc[:0], barriers.iloc[[4]], reservoir_shapes
    )
    assert result["waterbody_id"].tolist() == [GDW_ID_OFFSET + 5]
    assert dam_checks.loc[0, "addition_reason"] == "added"
