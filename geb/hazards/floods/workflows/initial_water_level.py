"""Calculate initial SFINCS water volumes for river channels."""

import math
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

from geb.workflows.raster import calculate_cell_area_m2


def calculate_initial_water_volume(
    initial_level: xr.DataArray,
    cell_area_coarse: xr.DataArray,
    bed_elevation: xr.DataArray,
    subgrid_dem: Path | None,
    is_geographic: bool,
) -> float:
    """Calculate the total volume of initial water levels across active river cells.

    For subgrid models, the initial water level map is expanded to subgrid resolution
    and water depth is computed against the subgrid DEM channel bathymetry.
    For non-subgrid models, depth is computed directly on the coarse grid as the initial
    water level minus bed elevation.

    Args:
        initial_level: Initial water-level map (meters), with NaN where water level is not set.
        cell_area_coarse: Coarse cell area grid (m²).
        bed_elevation: Bed elevation grid (meters). In subgrid mode, this is z_zmin;
            in non-subgrid mode, this is the burned dep grid.
        subgrid_dem: Path to ``dep_subgrid.tif`` for subgrid models, or None for
            non-subgrid models.
        is_geographic: Whether the model CRS is in geographic coordinates (lat/lon).

    Returns:
        Total initial water volume in cubic meters (m³).
    """
    if subgrid_dem is not None:
        with rasterio.open(subgrid_dem) as src:
            dep_subgrid: np.ndarray = src.read(1).astype(np.float32)
            height: int = src.height
            width: int = src.width
            transform = src.transform
            res = src.res

        pixels_per_cell_y: int = height // initial_level.shape[0]
        pixels_per_cell_x: int = width // initial_level.shape[1]

        zs_subgrid: np.ndarray = np.repeat(
            np.repeat(initial_level.values, pixels_per_cell_y, axis=0),
            pixels_per_cell_x,
            axis=1,
        )

        if is_geographic:
            cell_area: np.ndarray = calculate_cell_area_m2(transform, height, width)
        else:
            cell_area = np.full((height, width), abs(math.prod(res)), dtype=np.float32)

        valid_water: np.ndarray = np.isfinite(zs_subgrid)
        water_depth: np.ndarray = np.maximum(zs_subgrid - dep_subgrid, 0.0)
        return float(np.sum(np.where(valid_water, water_depth * cell_area, 0.0)))

    initial_level_values: np.ndarray = initial_level.values
    bed_elevation_values: np.ndarray = bed_elevation.values
    cell_area_values: np.ndarray = cell_area_coarse.values

    valid_cells: np.ndarray = np.isfinite(initial_level_values)
    depth_coarse: np.ndarray = np.maximum(
        initial_level_values - bed_elevation_values, 0.0
    )
    return float(np.sum(np.where(valid_cells, depth_coarse * cell_area_values, 0.0)))
