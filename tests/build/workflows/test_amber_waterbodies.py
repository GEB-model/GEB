"""Test AMBER dam classification of HydroLAKES waterbodies."""

from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from geb.build.workflows.waterbodies_preprocessing import (
    enrich_waterbodies,
    load_and_enrich_waterbodies,
    promote_lakes_with_amber_dams,
)
from geb.hydrology.waterbodies import RESERVOIR


@pytest.mark.parametrize("source_type", [1, 2, 3])
@pytest.mark.parametrize(
    "case",
    [
        "inside",
        "edge",
        "outside",
        "weir",
        "empty",
        "multiple",
        "ambiguous",
        "zero",
        "nan",
        "infinite",
        "gdw",
    ],
)
def test_amber_lake_promotion(source_type: int, case: str) -> None:
    """Promote lakes with unambiguous dams and usable volume.

    Args:
        source_type: Original HydroLAKES type code (unitless).
        case: Geometry, barrier type, storage, or GDW-priority scenario.

    Returns:
        None.

    Raises:
        AssertionError: If classification, capacity, or provenance changes incorrectly.
    """  # noqa: DOC202, DOC502
    volume_m3: float = {"zero": 0.0, "nan": np.nan, "infinite": np.inf}.get(
        case, 1000.0
    )
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "waterbody_id": [1],
            "waterbody_type": [source_type],
            "volume_total": [volume_m3],
        },
        geometry=[box(0, 0, 1, 1)],
        crs=4326,
    )
    gdw_points: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "hydrolakes_id": [1],
            "dam_type": ["Dam"],
            "lake_control": ["Yes"],
            "capacity_m3": [500.0],
        },
        geometry=[Point(0.5, 0.5)],
        crs=4326,
    )
    if case != "gdw":
        gdw_points = gdw_points.iloc[:0]
    enriched: gpd.GeoDataFrame
    enriched, _ = enrich_waterbodies(waterbodies, gdw_points)
    if case == "ambiguous":
        second_lake: gpd.GeoDataFrame = enriched.copy()
        second_lake["waterbody_id"] = 2
        enriched = gpd.GeoDataFrame(
            pd.concat([enriched, second_lake], ignore_index=True), crs=enriched.crs
        )
    longitude: float = {"edge": 1.0, "outside": 2.0}.get(case, 0.5)
    amber_points: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"dam_type": ["Weir" if case == "weir" else "DAM", "Dam"]},
        geometry=[Point(longitude, 0.5), Point(0.6, 0.5)],
        crs=4326,
    ).iloc[: 2 if case == "multiple" else 0 if case == "empty" else 1]
    original: gpd.GeoDataFrame = enriched.copy()
    result: gpd.GeoDataFrame = promote_lakes_with_amber_dams(
        enriched, amber_points.to_crs(3857)
    )
    should_promote: bool = (
        source_type in (1, 3)
        and case in ("inside", "edge", "multiple", "gdw")
        and not (case == "gdw" and source_type == 3)
    )
    assert bool(result.iloc[0].amber_changed_to_reservoir) == should_promote
    assert result.iloc[0].waterbody_type == (
        RESERVOIR if should_promote else original.iloc[0].waterbody_type
    )
    assert result.iloc[0].amber_dam_count == (
        2
        if case == "multiple"
        else 0
        if case in ("outside", "weir", "empty", "ambiguous")
        else 1
    )
    pd.testing.assert_series_equal(result.volume_total, original.volume_total)
    pd.testing.assert_series_equal(result.hydrolakes_type, original.hydrolakes_type)
    pd.testing.assert_series_equal(
        result.hydrolakes_volume_total, original.hydrolakes_volume_total
    )
    pd.testing.assert_frame_equal(enriched, original)
    assert result.waterbody_type.dtype == np.int32


@pytest.mark.parametrize(
    "invalid", ["waterbody_crs", "amber_crs", "duplicate_id", "missing_id"]
)
def test_invalid_amber_waterbodies(invalid: str) -> None:
    """Reject missing coordinate systems and ambiguous waterbody IDs.

    Args:
        invalid: Invalid coordinate system or ID scenario.

    Returns:
        None.

    Raises:
        AssertionError: If invalid input is accepted.
    """  # noqa: DOC202, DOC502
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"waterbody_id": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs=4326,
    )
    amber_points: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"dam_type": ["Dam"]},
        geometry=[Point(0.5, 0.5)],
        crs=4326,
    )
    if invalid == "waterbody_crs":
        waterbodies.set_crs(None, allow_override=True, inplace=True)
    elif invalid == "amber_crs":
        amber_points.set_crs(None, allow_override=True, inplace=True)
    elif invalid == "duplicate_id":
        waterbodies["waterbody_id"] = 1
    else:
        waterbodies["waterbody_id"] = np.nan
    with pytest.raises(ValueError, match="CRS|IDs"):
        promote_lakes_with_amber_dams(waterbodies, amber_points)


@pytest.mark.parametrize("mode", ["on", "lakes_only", "reservoirs_only", "off"])
def test_load_amber_waterbodies(mode: str) -> None:
    """Load dams across the full lake extent before mode filtering.

    Args:
        mode: Requested waterbody build mode.

    Returns:
        None.

    Raises:
        AssertionError: If AMBER loading or classification is incorrect.
    """  # noqa: DOC202, DOC502
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"waterbody_id": [1], "waterbody_type": [3], "volume_total": [1000.0]},
        geometry=[box(0, 0, 1, 1)],
        crs=4326,
    )
    region: gpd.GeoDataFrame = gpd.GeoDataFrame(geometry=[box(0, 0, 0.25, 1)], crs=4326)
    gdw_points: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "hydrolakes_id": [],
            "dam_type": [],
            "lake_control": [],
            "capacity_m3": [],
            "gdw_id": [],
        },
        geometry=[],
        crs=4326,
    )
    reservoir_shapes: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"gdw_id": []}, geometry=[], crs=4326
    )
    amber_points: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"dam_type": ["Dam"]},
        geometry=[Point(0.75, 0.5)],
        crs=4326,
    )
    adapters: dict[str, Mock] = {
        name: Mock(read=Mock(return_value=data))
        for name, data in {
            "hydrolakes": waterbodies,
            "gdw_barriers": gdw_points,
            "gdw_reservoirs": reservoir_shapes,
            "amber_barriers": amber_points,
        }.items()
    }
    catalog: Mock = Mock(fetch=Mock(side_effect=adapters.__getitem__))
    result: gpd.GeoDataFrame
    result, _ = load_and_enrich_waterbodies(catalog, region, mode)
    if mode == "off":
        assert result.empty
        adapters["amber_barriers"].read.assert_not_called()
    else:
        assert result.iloc[0].waterbody_type == RESERVOIR
        assert result.iloc[0].amber_changed_to_reservoir
        adapters["amber_barriers"].read.assert_called_once_with(
            bbox=(0.0, 0.0, 1.0, 1.0)
        )
