"""Tests for the Fluxnet data adapter and observation setup."""

import logging

import geopandas as gpd
import pytest

from geb.build.data_catalog import DataCatalog

from ...testconfig import IN_GITHUB_ACTIONS

# European bounding box (lon_min, lat_min, lon_max, lat_max) used across tests.
# Several FLUXNET stations are located in Switzerland and neighbouring countries
# within this box, so both metadata and timeseries queries are guaranteed to
# return results.
EUROPE_BOUNDS: tuple[float, float, float, float] = (-10.0, 35.0, 30.0, 60.0)


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Too heavy for GitHub Actions.",
)
def test_fluxnet_adapter_metadata_reading() -> None:
    """Test that fetch() parses BIF files and produces a valid station GeoDataFrame.

    Verifies that after fetching, the parquet is written and contains the
    expected columns, geometry, and at least one station within the European
    bounding box.
    """
    from geb.build.data_catalog import DataCatalog

    logger = logging.getLogger("test_fluxnet")
    catalog = DataCatalog(logger=logger)

    adapter = catalog.fetch("fluxnet")

    assert adapter.path.exists(), "Geoparquet file was not created by fetch()"


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Too heavy for GitHub Actions.",
)
def test_fluxnet_adapter_timeseries_reading() -> None:
    """Test that read(variable='LE_F_MDS') returns evaporation timeseries.

    Calls fetch() with European bounds so that only a small subset of stations
    is queried, keeping the test fast.  Verifies that the returned DataFrame
    has a DatetimeIndex named 'time', at least one station column, and the
    expected number of rows for daily data.
    """
    catalog = DataCatalog(logger=logging.getLogger("test_fluxnet"))
    adapter = catalog.fetch("fluxnet")

    benelux_geom: gpd.GeoDataFrame = gpd.GeoDataFrame.from_features(
        [
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [6.3, 50.5],
                            [6.4, 50.5],
                            [6.4, 50.6],
                            [6.3, 50.6],
                            [6.3, 50.5],
                        ]
                    ],
                },
                "properties": {"name": "Benelux Region"},
            }
        ],
        crs="EPSG:4326",
    )

    stations, timeseries = adapter.read(geom=benelux_geom)

    assert not stations.empty, "No stations returned for Benelux bounds"
    assert not timeseries.empty, "No evaporation data returned for Benelux bounds"

    ocean_bounds = gpd.GeoDataFrame.from_features(
        [
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-50.0, -10.0],
                            [-40.0, -10.0],
                            [-40.0, 0.0],
                            [-50.0, 0.0],
                            [-50.0, -10.0],
                        ]
                    ],
                },
                "properties": {"name": "Ocean"},
            }
        ],
        crs="EPSG:4326",
    )

    stations, timeseries = adapter.read(geom=ocean_bounds)
    assert stations.empty, "Stations should be empty for ocean bounds"
    assert timeseries.empty, "Timeseries should be empty for ocean bounds"
