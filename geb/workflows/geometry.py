"""Workflow utilities for handling geometry data."""

from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon


def read_parquet_with_geom(
    path: Path, geom: Polygon | MultiPolygon | None = None, **kwargs: Any
) -> gpd.GeoDataFrame:
    """Read a parquet file with optional spatial filtering.

    Args:
        path: Path to the parquet file.
        geom: Optional shapely Polygon to filter the geometries.
        **kwargs: Additional keyword arguments to pass to gpd.read_parquet.

    Returns:
        A GeoDataFrame with the data from the parquet file, optionally filtered by the geometry.
    """
    if geom is not None:
        assert "bbox" not in kwargs, "Cannot use both geom and bbox"
        assert isinstance(geom, (Polygon, MultiPolygon)), (
            "geom must be a Polygon or MultiPolygon"
        )
        gdf: gpd.GeoDataFrame = gpd.read_parquet(path, **kwargs, bbox=geom.bounds)
        gdf: gpd.GeoDataFrame = gdf[gdf.intersects(geom)]
        return gdf
    else:
        return gpd.read_parquet(path, **kwargs)
