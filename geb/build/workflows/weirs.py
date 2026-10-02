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
from geb.workflows.raster import full_like


def create_weir_grids(
    gdw_points: gpd.GeoDataFrame,
    crest_height_m: float | None,
    waterbody_id: xr.DataArray,
    rivers: gpd.GeoDataFrame,
    upstream_area_grid: xr.DataArray,
    upstream_area_subgrid: xr.DataArray,
    valid_river_cells: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Place weir heights and gates on the river grid.

    Args:
        gdw_points: GDW points and their linked lake or reservoir IDs.
        crest_height_m: Missing-height override (m). None uses type defaults.
        waterbody_id: Existing waterbody grid; -1 means no lake or reservoir.
        rivers: Rivers used to place the points on the grid.
        upstream_area_grid: Model-grid drainage area (m2).
        upstream_area_subgrid: Original drainage area (m2).
        valid_river_cells: True where a weir can be placed.

    Returns:
        Height grid and gate flags. Positive heights are in meters, zero means
        no structure, -1 means bankfull depth + 1 m, and -2 means half
        bankfull depth.
        Missing heights are resolved from bankfull depth at runtime. Gate
        thresholds use the configured fractions of the resolved crest height.

    Raises:
        ValueError: If the missing-height override is invalid.
    """
    if crest_height_m is not None and (
        not np.isfinite(crest_height_m) or crest_height_m <= 0
    ):
        raise ValueError("crest_height_m must be positive and finite.")
    # Load the small grids once so cell reads and writes use arrays in memory.
    waterbody_id = waterbody_id.compute()
    valid_river_cells = valid_river_cells.compute()
    upstream_area_grid = upstream_area_grid.compute()
    weir_height_grid: xr.DataArray = full_like(
        waterbody_id, fill_value=0.0, nodata=np.nan, dtype=np.float32
    )
    gate_grid: xr.DataArray = full_like(waterbody_id, False, nodata=None, dtype=bool)
    if gdw_points.empty:
        return weir_height_grid, gate_grid
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
            or weir_height_grid.values[row, column] != 0
        ):
            logger.warning(
                "Skipping GDW weir %s: cell already used, or no river cell on both sides.",
                gdw_point.gdw_id,
            )
            continue
        dam_type: str = str(gdw_point.get("dam_type", "")).strip()
        gdw_height_m: float = gdw_point.dam_hgt_m
        height_m: float = 0.0
        if pd.notna(gdw_height_m) and np.isfinite(gdw_height_m) and gdw_height_m > 0:
            height_m = float(gdw_height_m)
        elif crest_height_m is not None:
            height_m = crest_height_m
        elif dam_type in {"Dam", "Lake Control Dam"}:
            height_m = -1.0  # Resolve to bankfull depth + 1 m when routing starts.
        else:
            height_m = -2.0  # Resolve to half bankfull depth when routing starts.
        # Gate operation is a simple assumption based on the barrier type.
        is_controlled: bool = dam_type in {"Dam", "Sluice", "Lake Control Dam"}
        weir_height_grid.values[row, column] = height_m
        gate_grid.values[row, column] = is_controlled
    return weir_height_grid, gate_grid
