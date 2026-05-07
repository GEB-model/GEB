"""Test the GROW data adapter."""

import logging

import geopandas as gpd
import pytest

from geb.build.data_catalog import DataCatalog

from ...testconfig import IN_GITHUB_ACTIONS


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Too heavy for GitHub Actions and needs GEB_TEST_ALL=yes.",
)
def test_grow_data_adapter() -> None:
    """Test the GROW data adapter."""
    logger: logging.Logger = logging.getLogger("test_grow")
    data_adapter = DataCatalog(logger=logger).fetch("grow")

    # large density of wells
    benelux_geom: gpd.GeoDataFrame = gpd.GeoDataFrame.from_features(
        [
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [2.5, 49.5],
                            [6.5, 49.5],
                            [6.5, 51.5],
                            [2.5, 51.5],
                            [2.5, 49.5],
                        ]
                    ],
                },
                "properties": {"name": "Benelux Region"},
            }
        ],
        crs="EPSG:4326",
    )

    attributes, timeseries = data_adapter.read(geom=benelux_geom)
    assert not timeseries["groundwater_depth_from_ground_elevation_m"].isnull().any()

    # West costa-rica geom. This area has wells with the well top elevation attribute, so we can check that the groundwater depth from well top elevation is not null.
    costa_rica_geom: gpd.GeoDataFrame = gpd.GeoDataFrame.from_features(
        [
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-86.0, 8.0],
                            [-83.0, 8.0],
                            [-83.0, 11.0],
                            [-86.0, 11.0],
                            [-86.0, 8.0],
                        ]
                    ],
                },
                "properties": {"name": "West Costa Rica"},
            }
        ],
        crs="EPSG:4326",
    )
    attributes, timeseries = data_adapter.read(geom=costa_rica_geom)
    assert not timeseries["groundwater_depth_from_ground_elevation_m"].isnull().any()

    # Geom in the ocean so that no wells are return
    ocean_geom: gpd.GeoDataFrame = gpd.GeoDataFrame.from_features(
        [
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-30.0, 0.0],
                            [-25.0, 0.0],
                            [-25.0, 5.0],
                            [-30.0, 5.0],
                            [-30.0, 0.0],
                        ]
                    ],
                },
                "properties": {"name": "Ocean"},
            }
        ],
        crs="EPSG:4326",
    )
    attributes, timeseries = data_adapter.read(geom=ocean_geom)
    assert attributes.empty
    assert timeseries.empty
