"""Workflows for DEM processing and river burning."""

from typing import Callable

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.features import rasterize
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]
from shapely.ops import transform as shapely_transform


def _validate_inputs(
    elevation_grid: xr.DataArray, manning_grid: xr.DataArray, rivers: gpd.GeoDataFrame
) -> None:
    """Validates structural and spatial constraints of inputs.

    Args:
        elevation_grid: Elevation DataArray with CRS and transform (meters).
        manning_grid: Manning's n DataArray with CRS and transform (s/m^(1/3)).
        rivers: GeoDataFrame containing river geometries and attributes.

    Raises:
        ValueError: If any validation check fails.
    """
    if elevation_grid.shape != manning_grid.shape:
        raise ValueError(
            "Input elevation and Manning's n DataArrays must have the same shape."
        )
    if rivers.empty:
        raise ValueError("Input 'rivers' GeoDataFrame is empty.")
    for col in [
        "width",
        "depth",
        "downstream_ID",
        "manning",
    ]:
        if col not in rivers.columns:
            raise ValueError(f"Required column '{col}' not found in GeoDataFrame.")
    if elevation_grid.rio.crs is None:
        raise ValueError("Input elevation DataArray must have a valid CRS.")
    if manning_grid.rio.crs is None:
        raise ValueError("Input Manning's n DataArray must have a valid CRS.")
    if rivers.crs != elevation_grid.rio.crs:
        raise ValueError(
            f"Spatial CRS mismatch! Rivers layer uses ({rivers.crs}) "
            f"but DEM layer uses ({elevation_grid.rio.crs}). Please reproject."
        )


def _burn(
    river_network: list[dict],
    shape: tuple,
    transform: tuple,
    to_m: Callable[
        [np.ndarray | float, np.ndarray | float], tuple[np.ndarray, np.ndarray]
    ],
    to_deg: Callable[
        [np.ndarray | float, np.ndarray | float], tuple[np.ndarray, np.ndarray]
    ],
    elevation_grid: xr.DataArray,
    manning_grid: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Burns the river network into the DEM and Manning's n grid.

    Args:
        river_network: List of river segment dictionaries.
        shape: Shape of the DEM grid.
        transform: Affine transform of the DEM grid.
        to_m: Function to convert geographic coordinates to meters.
        to_deg: Function to convert meters back to geographic coordinates.
        elevation_grid: Elevation DataArray with CRS and transform (meters).
        manning_grid: Manning's n DataArray with CRS and transform (s/m^(1/3)).

    Returns:
        Tuple containing:
            - Modified elevation DataArray with rivers burned in (meters).
            - Modified Manning's n DataArray with river roughness values (s/m^(1/3)).
            - Initial water level DataArray on burned river cells (meters), with NaN elsewhere.
    """
    grid_shape = (shape[-2], shape[-1])
    global_burn_elevation = np.full(grid_shape, fill_value=np.inf, dtype=np.float32)
    global_burn_manning = np.full(grid_shape, fill_value=-np.inf, dtype=np.float32)
    global_burn_depth = np.full(grid_shape, fill_value=-np.inf, dtype=np.float32)

    # Extract the elevation values for path-of-least-resistance lookups
    dem_base = elevation_grid.values

    for river in river_network:
        xs_m, ys_m = to_m(river["xs"], river["ys"])
        tree = cKDTree(np.c_[xs_m, ys_m])

        effective_width_m = river["width"]
        poly = shapely_transform(
            to_deg,
            shapely_transform(to_m, river["geom"]).buffer(
                effective_width_m / 2.0, cap_style=1, join_style=1
            ),
        )

        mask = rasterize(
            [(poly, 1)],
            out_shape=grid_shape,
            transform=transform,
            fill=0,
            all_touched=False,
            dtype=np.uint8,
        )

        line_mask = rasterize(
            [(river["geom"], 1)],
            out_shape=grid_shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )

        mask = np.maximum(mask, line_mask)

        # Identify diagonal connectivity gaps in the rasterized river footprint.
        # A gap occurs when two diagonal cells are set and the orthogonal
        # neighbors that would connect them are unset.
        top_left_bottom_right_gap = (
            (mask[:-1, :-1] == 1)
            & (mask[1:, 1:] == 1)
            & (mask[:-1, 1:] == 0)
            & (mask[1:, :-1] == 0)
        )
        top_right_bottom_left_gap = (
            (mask[:-1, 1:] == 1)
            & (mask[1:, :-1] == 1)
            & (mask[:-1, :-1] == 0)
            & (mask[1:, 1:] == 0)
        )

        # For each diagonal gap, choose the orthogonal neighbor with the lower
        # terrain elevation and fill it to preserve 4-connected river continuity.
        if np.any(top_left_bottom_right_gap):
            row_indices, col_indices = np.where(top_left_bottom_right_gap)
            elev_top_right = dem_base[row_indices, col_indices + 1]
            elev_bottom_left = dem_base[row_indices + 1, col_indices]
            use_top_right = elev_top_right <= elev_bottom_left

            mask[row_indices[use_top_right], col_indices[use_top_right] + 1] = 1
            mask[row_indices[~use_top_right] + 1, col_indices[~use_top_right]] = 1

        if np.any(top_right_bottom_left_gap):
            row_indices, col_indices = np.where(top_right_bottom_left_gap)
            elev_top_left = dem_base[row_indices, col_indices]
            elev_bottom_right = dem_base[row_indices + 1, col_indices + 1]
            use_top_left = elev_top_left <= elev_bottom_right

            mask[row_indices[use_top_left], col_indices[use_top_left]] = 1
            mask[row_indices[~use_top_left] + 1, col_indices[~use_top_left] + 1] = 1

        p_rows, p_cols = np.where(mask == 1)

        p_xs_m, p_ys_m = to_m(
            transform[2] + (p_cols + 0.5) * transform[0],
            transform[5] + (p_rows + 0.5) * transform[4],
        )
        _, nearest = tree.query(np.c_[p_xs_m, p_ys_m])

        global_burn_elevation[p_rows, p_cols] = np.fmin(
            global_burn_elevation[p_rows, p_cols], river["target_z"][nearest]
        )
        global_burn_manning[p_rows, p_cols] = np.fmax(
            global_burn_manning[p_rows, p_cols], river["manning"]
        )
        global_burn_depth[p_rows, p_cols] = np.fmax(
            global_burn_depth[p_rows, p_cols], river["depth"]
        )

    update_mask = np.isfinite(global_burn_elevation)

    elevation_grid.values[update_mask] = global_burn_elevation[update_mask]

    out_mannings = manning_grid.copy()
    out_mannings.values[update_mask] = global_burn_manning[update_mask]

    initial_water_level: xr.DataArray = xr.full_like(
        elevation_grid, np.nan, dtype=np.float32
    )
    initial_water_level.attrs["_FillValue"] = np.nan
    initial_water_level.values[update_mask] = (
        global_burn_elevation[update_mask] + global_burn_depth[update_mask]
    )

    return elevation_grid, out_mannings, initial_water_level


def burn_rivers(
    elevation_grid: xr.DataArray,
    manning_grid: xr.DataArray,
    rivers: gpd.GeoDataFrame,
    river_bed_points: gpd.GeoDataFrame,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Burns river networks into a Digital Elevation Model (DEM).

    The riverbed comes from the river profiles of the model build (water surface from
    FathomDEM minus the river depth, see ``geb.workflows.river_profiles``), so the
    flood model and the hydrological routing use the same rivers. Every DEM cell
    within half the river width of a river line gets the bed elevation of the nearest
    profile point and the river roughness.

    Args:
        elevation_grid: Elevation DataArray with CRS and transform (meters).
        manning_grid: Manning's n DataArray with CRS and transform (s/m^(1/3)).
        rivers: GeoDataFrame containing river geometries and attributes, indexed by
            river ID.
        river_bed_points: Riverbed points of these rivers (``river_bed_profiles``) in
            the CRS of the elevation grid, ordered from upstream to downstream per
            river.

    Returns:
        Tuple containing:
            - Modified elevation DataArray with rivers burned in (meters).
            - Modified Manning's n DataArray with river roughness values (s/m^(1/3)).
            - Initial water level DataArray on burned river cells (meters), with NaN elsewhere.

    Raises:
        ValueError: If a river has no riverbed points.
    """
    _validate_inputs(elevation_grid, manning_grid, rivers)

    out_elevation = elevation_grid.copy()
    transform = out_elevation.rio.transform()
    shape = out_elevation.shape
    is_geo = out_elevation.rio.crs.is_geographic

    x_orig, y_orig = (
        float(out_elevation.coords["x"].mean()),
        float(out_elevation.coords["y"].mean()),
    )

    # Conversion factors for geographic coordinates to meters.
    m_x = 111320.0 * np.cos(np.radians(y_orig)) if is_geo else 1.0
    m_y = 111320.0 if is_geo else 1.0

    def to_m(
        x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        return (np.asanyarray(x) - (x_orig if is_geo else 0)) * m_x, (
            np.asanyarray(y) - (y_orig if is_geo else 0)
        ) * m_y

    def to_deg(
        x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.asanyarray(x) / m_x + (x_orig if is_geo else 0), np.asanyarray(
            y
        ) / m_y + (y_orig if is_geo else 0)

    points_by_river = river_bed_points.groupby("river_id")
    river_network: list[dict] = []
    for river_id, river in rivers.iterrows():
        if river_id not in points_by_river.groups:
            raise ValueError(f"River {river_id} has no riverbed points.")
        points = points_by_river.get_group(river_id)
        river_network.append(
            {
                "xs": points.geometry.x.to_numpy(),
                "ys": points.geometry.y.to_numpy(),
                "target_z": points["bed_elevation_m"].to_numpy(dtype=np.float32),
                "width": float(river["width"]),
                "depth": float(river["depth"]),
                "manning": float(river["manning"]),
                "geom": river.geometry,
            }
        )

    # Actually modify the elevation and roughness grids.
    out_elevation, out_mannings, initial_water_level = _burn(
        river_network, shape, transform, to_m, to_deg, out_elevation, manning_grid
    )

    return (
        out_elevation,
        out_mannings,
        initial_water_level,
    )
