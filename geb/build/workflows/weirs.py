"""Place GDW points without polygons on the river network as weirs."""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from geb.build.workflows.river_snapping import (
    SnappingResults,
    snap_point_to_river_network,
)


def create_weir_height_grid(
    gdw_points: gpd.GeoDataFrame,
    crest_height_m: float,
    waterbody_id: xr.DataArray,
    rivers: gpd.GeoDataFrame,
    upstream_area_grid: xr.DataArray,
    upstream_area_subgrid: xr.DataArray,
    valid_river_cells: xr.DataArray,
) -> xr.DataArray:
    """Set the height of each weir on the river grid.

    Args:
        gdw_points: GDW points and their linked lake or reservoir IDs.
        crest_height_m: Height to use when the GDW height is missing or invalid (m).
        waterbody_id: Existing waterbody grid; -1 means no lake or reservoir.
        rivers: Rivers used to place the points on the grid.
        upstream_area_grid: Model-grid drainage area (m2).
        upstream_area_subgrid: Original drainage area (m2).
        valid_river_cells: True where a weir can be placed.

    Returns:
        Weir heights (m), with zero where there is no weir.

    Raises:
        ValueError: If the default height is not positive and finite.
    """
    if not np.isfinite(crest_height_m) or crest_height_m <= 0:
        raise ValueError("crest_height_m must be positive and finite.")
    weir_height_grid: xr.DataArray = xr.zeros_like(waterbody_id, dtype=np.float32)
    if gdw_points.empty:
        return weir_height_grid
    weirs: gpd.GeoDataFrame = gdw_points.loc[~gdw_points["inside_gdw_polygon"]]
    logger: logging.Logger = logging.getLogger(__name__)
    gdw_point: pd.Series
    for _, gdw_point in weirs.iterrows():
        if pd.notna(gdw_point.waterbody_id):
            logger.warning(
                "Skipping GDW weir %s: already part of a lake or reservoir.",
                gdw_point.gdw_id,
            )
            continue
        river_location: SnappingResults | None = snap_point_to_river_network(
            point=gdw_point.geometry,
            rivers=rivers,
            upstream_area_grid=upstream_area_grid,
            upstream_area_subgrid=upstream_area_subgrid,
        )
        if river_location is None:
            logger.warning("Skipping GDW weir %s: no river found.", gdw_point.gdw_id)
            continue
        column: int
        row: int
        column, row = river_location.snapped_grid_pixel_xy
        if (
            not valid_river_cells.values[row, column]
            or waterbody_id.values[row, column] != -1
            or weir_height_grid.values[row, column] > 0
        ):
            logger.warning(
                "Skipping GDW weir %s: cell already used, or no river cell on both sides.",
                gdw_point.gdw_id,
            )
            continue
        height_m: float = crest_height_m
        gdw_height_m: float = gdw_point.dam_hgt_m
        if pd.notna(gdw_height_m) and np.isfinite(gdw_height_m) and gdw_height_m > 0:
            height_m = float(gdw_height_m)
        weir_height_grid.values[row, column] = height_m
    return weir_height_grid
