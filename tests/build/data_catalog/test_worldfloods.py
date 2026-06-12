"""Test the WorldFloodsV2 data adapter."""

import logging

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr

from geb.build.data_catalog.worldfloods import WorldFloodsV2

from ...testconfig import IN_GITHUB_ACTIONS


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Too heavy for GitHub Actions.",
)
def test_worldfloods_data_adapter() -> None:
    """Test the WorldFloodsV2 data adapter with real data."""
    logger = logging.getLogger("test_worldfloods")
    data_adapter = WorldFloodsV2(
        folder="worldfloodsv2",
        local_version=1,
        filename="metadata.parquet",
        cache="global",
    ).fetch(url="isp-uv-es/WorldFloodsv2")

    region: gpd.GeoDataFrame = gpd.GeoDataFrame.from_features(
        [
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-180.0, -90.0],
                            [180.0, -90.0],
                            [180.0, 90.0],
                            [-180.0, 90.0],
                            [-180.0, -90.0],
                        ]
                    ],
                },
                "properties": {"name": "Global"},
            }
        ],
        crs="EPSG:4326",
    )

    floods, metadata, flood_maps = data_adapter.read(region=region)

    assert "observation_date" in metadata.columns
    assert pd.api.types.is_datetime64_any_dtype(metadata["observation_date"])
    assert not metadata["observation_date"].isnull().any()

    # Check if we have at least one map read
    sample_event = metadata["event id"].iloc[0]
    assert sample_event in flood_maps
    assert isinstance(flood_maps[sample_event], xr.DataArray)
