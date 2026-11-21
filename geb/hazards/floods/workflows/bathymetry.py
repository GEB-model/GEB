"""Workflows for DEM processing and river burning."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Callable

import geopandas as gpd
import hydromt_sfincs.workflows.bathymetry as hydromt_sfincs_bathymetry
import numpy as np
import xarray as xr
from rasterio.features import rasterize
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]
from shapely.ops import transform as shapely_transform


@contextmanager
def skip_subgrid_tiles_without_river_bed_levels() -> Generator[None, None, None]:
    """Makes hydromt-sfincs subgrid river burning robust for tiles without river bed levels.

    When hydromt-sfincs burns rivers into a subgrid tile, the river bed level is
    estimated from the elevation of the river banks. If a tile only contains a
    sliver of a river (for example at the tile edge), no valid river bed level can
    be derived and hydromt-sfincs crashes with
    `ValueError: Lengths of inputs do not match`. Within this context, such tiles
    are left unchanged (no river is burned) instead.

    Notes:
        This temporarily replaces `interp_along_line_to_grid` in the
        hydromt-sfincs bathymetry workflow and restores it on exit.

    Yields:
        None.
    """
    original_interpolation: Callable[..., Any] = (
        hydromt_sfincs_bathymetry.interp_along_line_to_grid
    )

    def interpolation_with_empty_fallback(
        da_mask: xr.DataArray,
        gdf_lines: gpd.GeoDataFrame,
        gdf_zb: gpd.GeoDataFrame,
        column_names: list[str] = ["z"],
        **kwargs: Any,
    ) -> Any:
        """Interpolates along lines, returning NaN everywhere if there are no valid points.

        Args:
            da_mask: Boolean mask of the cells to interpolate.
            gdf_lines: River center lines.
            gdf_zb: Points with the values to interpolate.
            column_names: Names of the columns in `gdf_zb` to interpolate.
            **kwargs: Additional arguments passed to the original function.

        Returns:
            Dataset with one NaN-filled variable per column name if `gdf_zb` or
            `gdf_lines` is empty, otherwise the interpolated values.
        """
        if gdf_zb.empty or gdf_lines.empty:
            return xr.Dataset(
                {
                    name: xr.full_like(da_mask, np.nan, dtype=np.float32)
                    for name in column_names
                }
            )
        return original_interpolation(
            da_mask=da_mask,
            gdf_lines=gdf_lines,
            gdf_zb=gdf_zb,
            column_names=column_names,
            **kwargs,
        )

    setattr(
        hydromt_sfincs_bathymetry,
        "interp_along_line_to_grid",
        interpolation_with_empty_fallback,
    )
    try:
        yield
    finally:
        setattr(
            hydromt_sfincs_bathymetry,
            "interp_along_line_to_grid",
            original_interpolation,
        )


@contextmanager
def capture_subgrid_initial_water_level(
    grid_mask: xr.DataArray,
    nr_subgrid_pixels: int,
) -> Generator[xr.DataArray, None, None]:
    """Captures initial water levels on the coarse grid during subgrid river burning.

    Wraps `hydromt_sfincs.workflows.bathymetry.burn_river_rect` so that as each
    subgrid tile is burned during `SfincsSubgridTable.create`, any subgrid pixel
    whose elevation is lowered (`burned < unburned`) is detected directly in memory.
    For each active coarse cell (`grid_mask == 1`) containing burned subgrid pixels,
    the initial water level is set to the minimum unburned elevation of the burned
    subgrid pixels in that cell.

    Notes:
        This temporarily replaces `burn_river_rect` in the hydromt-sfincs
        bathymetry workflow and restores it on exit.

    Args:
        grid_mask: Active cell mask for the coarse SFINCS grid (1 = active cell).
        nr_subgrid_pixels: Number of subgrid pixels per coarse cell along each axis.

    Yields:
        Coarse-grid DataArray of initial water levels (meters), populated as
        subgrid tiles are burned, with NaN in cells without burned river pixels.
    """
    initial_water_level: xr.DataArray = xr.full_like(
        grid_mask, np.nan, dtype=np.float32
    )
    initial_water_level.attrs["_FillValue"] = np.nan

    grid_transform = grid_mask.rio.transform(recalc=True)
    original_burn_river_rect: Callable[..., Any] = (
        hydromt_sfincs_bathymetry.burn_river_rect
    )

    def burn_river_rect_with_capture(
        da_elv: xr.DataArray,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Burns rivers into a subgrid tile and records coarse initial water levels.

        Args:
            da_elv: Unburned subgrid elevation tile (meters).
            *args: Positional arguments forwarded to `burn_river_rect`.
            **kwargs: Keyword arguments forwarded to `burn_river_rect`.

        Returns:
            Tuple of burned elevation DataArray (meters) and burned Manning DataArray.
        """
        unburned_subgrid: np.ndarray = da_elv.values[
            :-nr_subgrid_pixels, :-nr_subgrid_pixels
        ].copy()
        da_elv_burned, da_man_burned = original_burn_river_rect(
            da_elv,
            *args,
            **kwargs,
        )
        burned_subgrid: np.ndarray = da_elv_burned.values[
            :-nr_subgrid_pixels, :-nr_subgrid_pixels
        ]
        river_pixels: np.ndarray = burned_subgrid < unburned_subgrid
        if river_pixels.any():
            tile_transform = da_elv.rio.transform(recalc=True)
            block_column_start: int = round(
                (tile_transform.c - grid_transform.c) / grid_transform.a
            )
            block_row_start: int = round(
                (tile_transform.f - grid_transform.f) / grid_transform.e
            )
            n_rows_block: int = unburned_subgrid.shape[0] // nr_subgrid_pixels
            n_cols_block: int = unburned_subgrid.shape[1] // nr_subgrid_pixels
            block_row_end: int = block_row_start + n_rows_block
            block_column_end: int = block_column_start + n_cols_block

            cell_shape: tuple[int, int, int, int] = (
                n_rows_block,
                nr_subgrid_pixels,
                n_cols_block,
                nr_subgrid_pixels,
            )
            has_river_pixels: np.ndarray = river_pixels.reshape(cell_shape).any(
                axis=(1, 3)
            )
            cell_river_surface: np.ndarray = (
                np.where(river_pixels, unburned_subgrid, np.inf)
                .reshape(cell_shape)
                .min(axis=(1, 3))
            )
            active_river_cells: np.ndarray = has_river_pixels & (
                grid_mask.values[
                    block_row_start:block_row_end, block_column_start:block_column_end
                ]
                == 1
            )
            block_view: np.ndarray = initial_water_level.values[
                block_row_start:block_row_end, block_column_start:block_column_end
            ]
            block_view[active_river_cells] = np.fmin(
                block_view[active_river_cells],
                cell_river_surface[active_river_cells],
            )

        return da_elv_burned, da_man_burned

    setattr(
        hydromt_sfincs_bathymetry,
        "burn_river_rect",
        burn_river_rect_with_capture,
    )
    try:
        yield initial_water_level
    finally:
        setattr(
            hydromt_sfincs_bathymetry,
            "burn_river_rect",
            original_burn_river_rect,
        )


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
