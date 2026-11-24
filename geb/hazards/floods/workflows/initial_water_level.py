"""Calculate initial SFINCS water volumes for river channels."""

import numpy as np
import xarray as xr


def calculate_initial_water_volume(
    initial_level: xr.DataArray,
    cell_area_coarse: xr.DataArray,
    bed_elevation: xr.DataArray,
) -> float:
    """Calculate the total volume of initial water levels in a model without subgrid.

    The depth is the initial water level minus the bed elevation of each cell. For
    subgrid models, see `geb.hazards.floods.workflows.subgrid.subgrid_water_volume`.

    Args:
        initial_level: Initial water-level map (meters), with NaN where water level is not set.
        cell_area_coarse: Cell area grid (m²).
        bed_elevation: Bed elevation grid (meters), the burned dep grid.

    Returns:
        Total initial water volume in cubic meters (m³).
    """
    initial_level_values: np.ndarray = initial_level.values
    bed_elevation_values: np.ndarray = bed_elevation.values
    cell_area_values: np.ndarray = cell_area_coarse.values

    valid_cells: np.ndarray = np.isfinite(initial_level_values)
    depth_coarse: np.ndarray = np.maximum(
        initial_level_values - bed_elevation_values, 0.0
    )
    return float(np.sum(np.where(valid_cells, depth_coarse * cell_area_values, 0.0)))
