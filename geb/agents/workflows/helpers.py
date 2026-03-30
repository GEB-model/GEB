"""This module contains helper functions for the agents' workflows."""

import geopandas as gpd
import numpy as np
from affine import Affine
from pyproj import CRS
from rasterio.features import shapes
from shapely.geometry import shape

from geb.geb_types import TwoDArrayBool, TwoDArrayInt


def from_landuse_raster_to_polygon(
    mask: TwoDArrayBool | TwoDArrayInt, transform: Affine, crs: str | int | CRS
) -> gpd.GeoDataFrame:
    """Convert raster data into separate GeoDataFrames for specified land use values.

    Args:
        mask: A 2D numpy array representing the land use raster, where each unique value corresponds to a different land use type.
        transform: A rasterio Affine transform object that defines the spatial reference of the raster.
        crs: The coordinate reference system (CRS) to use for the resulting GeoDataFrame.

    Returns:
        A GeoDataFrame containing polygons for the specified land use values.
    """
    shapes_gen = shapes(mask.astype(np.uint8), mask=mask, transform=transform)

    polygons = []
    for geom, _ in shapes_gen:
        polygons.append(shape(geom))

    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=crs)

    return gdf
