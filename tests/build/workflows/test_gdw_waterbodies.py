"""Test loading HydroLAKES and GDW waterbody inputs."""

from unittest.mock import Mock

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from geb.build.workflows.waterbodies_preprocessing import load_and_enrich_waterbodies
from geb.hydrology.waterbodies import RESERVOIR


@pytest.mark.parametrize("mode", ["on", "lakes_only", "reservoirs_only", "off"])
def test_load_gdw_waterbodies(mode: str) -> None:
    """Load GDW data and classify reservoirs before mode filtering.

    Args:
        mode: Requested waterbody build mode.

    Returns:
        None.

    Raises:
        AssertionError: If GDW loading or classification is incorrect.
    """  # noqa: DOC202, DOC502
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"waterbody_id": [1], "waterbody_type": [3], "volume_total": [1000.0]},
        geometry=[box(0, 0, 1, 1)],
        crs=4326,
    )
    region: gpd.GeoDataFrame = gpd.GeoDataFrame(geometry=[box(0, 0, 0.25, 1)], crs=4326)
    gdw_points: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "hydrolakes_id": [1],
            "dam_type": ["Dam"],
            "lake_control": ["Yes"],
            "capacity_m3": [2000.0],
            "gdw_id": [42],
        },
        geometry=[Point(0.75, 0.5)],
        crs=4326,
    )
    reservoir_shapes: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"gdw_id": []}, geometry=[], crs=4326
    )
    adapters: dict[str, Mock] = {
        name: Mock(read=Mock(return_value=data))
        for name, data in {
            "hydrolakes": waterbodies,
            "gdw_barriers": gdw_points,
            "gdw_reservoirs": reservoir_shapes,
        }.items()
    }
    catalog: Mock = Mock(fetch=Mock(side_effect=adapters.__getitem__))
    result: gpd.GeoDataFrame
    result, _ = load_and_enrich_waterbodies(catalog, region, mode)
    if mode == "off":
        assert result.empty
        adapters["gdw_barriers"].read.assert_not_called()
        adapters["gdw_reservoirs"].read.assert_not_called()
    else:
        assert result.iloc[0].waterbody_type == RESERVOIR
        assert result.iloc[0].volume_total == 2000.0
        adapters["gdw_barriers"].read.assert_called_once_with()
        adapters["gdw_reservoirs"].read.assert_called_once_with(
            bbox=(0.0, 0.0, 0.25, 1.0)
        )
        assert not any("amber" in column for column in result.columns)


@pytest.mark.parametrize("invalid", ["mode", "crs"])
def test_invalid_waterbody_loading(invalid: str) -> None:
    """Reject invalid build settings before downloading datasets.

    Args:
        invalid: Invalid mode or missing region coordinate system.

    Returns:
        None.

    Raises:
        AssertionError: If validation downloads data or accepts invalid settings.
    """  # noqa: DOC202, DOC502
    region: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1)], crs=4326 if invalid == "mode" else None
    )
    catalog: Mock = Mock()
    with pytest.raises(ValueError, match="mode|CRS"):
        load_and_enrich_waterbodies(
            catalog, region, "invalid" if invalid == "mode" else "on"
        )
    catalog.fetch.assert_not_called()
