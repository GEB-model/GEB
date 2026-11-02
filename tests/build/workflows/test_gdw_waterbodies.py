"""Test loading HydroLAKES and GDW waterbody inputs."""

from unittest.mock import Mock

import geopandas as gpd
from shapely.geometry import Point, box

from geb.build.workflows.waterbodies_preprocessing import load_and_enrich_waterbodies
from geb.hydrology.waterbodies import RESERVOIR


def test_load_gdw_waterbodies() -> None:
    """Match dams outside the model region to lakes that cross its boundary."""
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
    result, _ = load_and_enrich_waterbodies(catalog, region, "on")
    assert result.iloc[0].waterbody_type == RESERVOIR
    assert result.iloc[0].volume_total == 2000.0
    adapters["gdw_barriers"].read.assert_called_once_with()
    adapters["gdw_reservoirs"].read.assert_called_once_with(bbox=(0.0, 0.0, 0.25, 1.0))
