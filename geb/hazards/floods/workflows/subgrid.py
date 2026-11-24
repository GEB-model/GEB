"""Fast, parallel replacements for the hydromt-sfincs subgrid functions.

hydromt-sfincs builds the subgrid tables of a regular SFINCS grid block by block in
one thread, keeps every table as a full-grid array, burns rivers per block with a
point geometry for every river pixel, and finds the index of every active cell and
u/v point with a Python loop over all grid cells. That is fine for small models, but
for a basin-scale grid with a fine subgrid (e.g. a 30 m grid with 1 m subgrid pixels)
it takes days and around a hundred GB of memory.

This module computes the same tables with the same method:

- `build_subgrid` replaces `SfincsSubgridTable.create` (plus its netcdf writer). The
  blocks are processed in parallel threads; the elevation and roughness of each block
  are resampled and merged as hydromt-sfincs' `merge_multi_dataarrays` does (with
  GDAL, which releases the GIL), and the tables are computed with a numba kernel
  that gives the same result as hydromt-sfincs' `process_tile_regular`, compiled
  without the GIL and cached, with a faster conveyance table. The tables are written directly into arrays of active cells and u/v points, as in the
  netcdf file, instead of full-grid arrays.
- `RiverBurner` replaces the subgrid use of `burn_river_rect`: the river polygons,
  merged center lines and bed-level profiles along them are prepared once for the
  whole model instead of per block, so that the burned bed level no longer depends
  on where the block boundaries fall.
- `sfincs_uv_indices` replaces `find_uv_indices` (vectorized, identical result).
- `read_subgrid_bed_elevation` reads only the minimum bed elevation (z_zmin) of a
  subgrid file, instead of every table on the full grid.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.fill
import rasterio.warp
import shapely
import xarray as xr
from affine import Affine
from hydromt_sfincs.workflows.subgrid import subgrid_v_table
from numba import njit
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import Window
from scipy.interpolate import griddata

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

# minimum depth for a u/v point to be wet in the conveyance table (m), as in SFINCS
HUTHRESH: float = 0.01


@njit(nogil=True, cache=True)
def _conveyance_table(
    elevation: np.ndarray, manning: np.ndarray, nr_levels: int, huthresh: float
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Conveyance table of one u/v point (hydromt-sfincs' `subgrid_q_table`).

    Same method and summation order as hydromt-sfincs with Manning's n,
    q_table_option 2 and weight_option "min": the u/v control volume is split into the
    halves of the two adjacent cells (a: first half of the pixels, b: second half); at
    each level the conveyance depth and representative roughness are those of the
    half with the smallest unit discharge. hydromt-sfincs also computes the discharge
    of the whole control volume at every level, which this option does not use, and
    takes the power of dry pixels (whose discharge is zero); both are skipped here.

    Args:
        elevation: Pixel elevations of the control volume (m), flattened.
        manning: Pixel Manning's n (s/m^(1/3)), flattened.
        nr_levels: Number of levels.
        huthresh: Minimum depth for a wet point (m).

    Returns:
        zmin, zmax, havg, nrep, pwet, ffit and navg, as in hydromt-sfincs.
    """
    n: int = elevation.size
    n_half: int = n // 2
    havg = np.zeros(nr_levels)
    nrep = np.zeros(nr_levels)
    pwet = np.zeros(nr_levels)

    zmin_a = elevation[0]
    zmax_a = elevation[0]
    for i in range(1, n_half):
        zmin_a = min(zmin_a, elevation[i])
        zmax_a = max(zmax_a, elevation[i])
    zmin_b = elevation[n_half]
    zmax_b = elevation[n_half]
    for i in range(n_half + 1, n):
        zmin_b = min(zmin_b, elevation[i])
        zmax_b = max(zmax_b, elevation[i])
    zmin = max(zmin_a, zmin_b) + huthresh
    zmax = max(float(max(zmax_a, zmax_b)), zmin + 0.01)
    dlevel = (zmax - zmin) / (nr_levels - 1)
    exponent: float = 5.0 / 3.0

    q_a = 0.0
    q_b = 0.0
    for level in range(nr_levels):
        z = zmin + level * dlevel
        sum_h_a = 0.0
        sum_q_a = 0.0
        wet_a = 0
        for i in range(n_half):
            depth = z - elevation[i]
            if depth > 0.0:
                sum_h_a += depth
                sum_q_a += depth**exponent / manning[i]
                wet_a += 1
        sum_h_b = 0.0
        sum_q_b = 0.0
        wet_b = 0
        for i in range(n_half, n):
            depth = z - elevation[i]
            if depth > 0.0:
                sum_h_b += depth
                sum_q_b += depth**exponent / manning[i]
                wet_b += 1
        h_a = sum_h_a / n_half
        q_a = sum_q_a / n_half
        h_b = sum_h_b / (n - n_half)
        q_b = sum_q_b / (n - n_half)
        pwet_a = wet_a / n_half
        pwet_b = wet_b / n_half
        if level == 0:
            if pwet_a < pwet_b:
                pwet_a = 0.0
            else:
                pwet_b = 0.0
        elif level == nr_levels - 1:
            pwet_a = 1.0
            pwet_b = 1.0
        if q_a < q_b:
            hmean = h_a
            q = q_a
        else:
            hmean = h_b
            q = q_b
        pwet[level] = 0.5 * (pwet_a + pwet_b)
        havg[level] = hmean
        nrep[level] = hmean**exponent / q

    # roughness above the top level, fitted to the conveyance at twice the range
    nrep_top = nrep[-1]
    havg_top = havg[-1]
    zfit = float(zmax + zmax - zmin)
    hfit = havg_top + zmax - zmin
    start, end = (0, n_half) if q_a < q_b else (n_half, n)
    sum_q = 0.0
    sum_n = 0.0
    for i in range(start, end):
        depth = zfit - elevation[i]
        if depth > 0.0:
            sum_q += depth**exponent / manning[i]
        sum_n += manning[i]
    q = sum_q / (end - start)
    navg = sum_n / (end - start)
    nfit = hfit**exponent / q
    gnavg2 = float(9.81 * navg**2)
    gnavg_top2 = float(9.81 * nrep_top**2)
    if gnavg2 / gnavg_top2 > 0.99 and gnavg2 / gnavg_top2 < 1.01:
        ffit = 0.0
    else:
        if navg > nrep_top:
            if nfit > navg:
                nfit = nrep_top + 0.9 * (navg - nrep_top)
            if nfit < nrep_top:
                nfit = nrep_top + 0.1 * (navg - nrep_top)
        else:
            if nfit < navg:
                nfit = nrep_top + 0.9 * (navg - nrep_top)
            if nfit > nrep_top:
                nfit = nrep_top + 0.1 * (navg - nrep_top)
        gnfit2 = float(9.81 * nfit**2)
        zfit = max(zfit, zmax + 1.0e-6)
        gnavg2 = max(gnavg2, gnfit2 + 1.0e-8)
        ffit = (((gnavg2 - gnavg_top2) / (gnavg2 - gnfit2)) - 1) / (zfit - zmax)
    return zmin, zmax, havg, nrep, pwet, ffit, navg


@njit(nogil=True, cache=True)
def _process_block_tables(
    mask: np.ndarray,
    elevation: np.ndarray,
    manning: np.ndarray,
    dxp: float,
    dyp: float,
    refi: int,
    nr_levels: int,
    y: np.ndarray,
    is_geographic: bool,
) -> tuple[np.ndarray, ...]:
    """Subgrid tables of the active cells of one block.

    Same as hydromt-sfincs' `process_tile_regular` with GEB's options (Manning's n,
    q_table_option 2, weight_option "min", no maximum gradient), with the fast
    conveyance table.

    Args:
        mask: Mask of the block's cells (> 0: active).
        elevation: Subgrid elevation (m), with one extra cell in both directions.
        manning: Subgrid Manning's n (s/m^(1/3)), same shape as elevation.
        dxp: Subgrid pixel size in x (grid units).
        dyp: Subgrid pixel size in y (grid units).
        refi: Subgrid pixels per cell along each axis.
        nr_levels: Number of levels.
        y: y coordinate of each subgrid pixel row.
        is_geographic: Whether the grid is in a geographic CRS (pixel size in degrees).

    Returns:
        The tables in the order of `_TABLE_NAMES`, each of shape (rows, cols) or
        (levels, rows, cols).
    """
    shape = mask.shape
    z_zmin = np.full(shape, np.nan, dtype=np.float32)
    z_zmax = np.full(shape, np.nan, dtype=np.float32)
    z_volmax = np.full(shape, np.nan, dtype=np.float32)
    z_level = np.full((nr_levels, shape[0], shape[1]), np.nan, dtype=np.float32)
    u_zmin = np.full(shape, np.nan, dtype=np.float32)
    u_zmax = np.full(shape, np.nan, dtype=np.float32)
    u_havg = np.full((nr_levels, shape[0], shape[1]), np.nan, dtype=np.float32)
    u_nrep = np.full((nr_levels, shape[0], shape[1]), np.nan, dtype=np.float32)
    u_pwet = np.full((nr_levels, shape[0], shape[1]), np.nan, dtype=np.float32)
    u_ffit = np.full(shape, np.nan, dtype=np.float32)
    u_navg = np.full(shape, np.nan, dtype=np.float32)
    v_zmin = np.full(shape, np.nan, dtype=np.float32)
    v_zmax = np.full(shape, np.nan, dtype=np.float32)
    v_havg = np.full((nr_levels, shape[0], shape[1]), np.nan, dtype=np.float32)
    v_nrep = np.full((nr_levels, shape[0], shape[1]), np.nan, dtype=np.float32)
    v_pwet = np.full((nr_levels, shape[0], shape[1]), np.nan, dtype=np.float32)
    v_ffit = np.full(shape, np.nan, dtype=np.float32)
    v_navg = np.full(shape, np.nan, dtype=np.float32)
    half: int = refi // 2

    for row in range(shape[0]):
        r0 = row * refi
        if is_geographic:
            # mean latitude of the cell's pixels, summed as hydromt-sfincs does
            # (over all refi x refi pixels), so that the result is identical
            sum_lat = 0.0
            for i in range(refi):
                for _ in range(refi):
                    sum_lat += y[r0 + i]
            mean_lat = abs(sum_lat / (refi * refi))
            dxpm = float(dxp * 111111.0 * np.cos(np.pi * mean_lat / 180.0))
            dypm = float(dyp * 111111.0)
        else:
            dxpm = float(dxp)
            dypm = float(dyp)
        for col in range(shape[1]):
            if mask[row, col] < 1:
                continue
            c0 = col * refi

            # volumes in the cell
            levels, volumes, zmin, zmax = subgrid_v_table(
                elevation[r0 : r0 + refi, c0 : c0 + refi].flatten(),
                dxpm,
                dypm,
                nr_levels,
                -20.0,
                99999.0,
            )
            z_zmin[row, col] = zmin
            z_zmax[row, col] = zmax
            z_volmax[row, col] = volumes[-1]
            z_level[:, row, col] = levels

            # u point: from the cell centre to the centre of the next column
            u = _conveyance_table(
                elevation[r0 : r0 + refi, c0 + half : c0 + half + refi].T.flatten(),
                manning[r0 : r0 + refi, c0 + half : c0 + half + refi].T.flatten(),
                nr_levels,
                HUTHRESH,
            )
            u_zmin[row, col] = u[0]
            u_zmax[row, col] = u[1]
            u_havg[:, row, col] = u[2]
            u_nrep[:, row, col] = u[3]
            u_pwet[:, row, col] = u[4]
            u_ffit[row, col] = u[5]
            u_navg[row, col] = u[6]

            # v point: from the cell centre to the centre of the next row
            v = _conveyance_table(
                elevation[r0 + half : r0 + half + refi, c0 : c0 + refi].flatten(),
                manning[r0 + half : r0 + half + refi, c0 : c0 + refi].flatten(),
                nr_levels,
                HUTHRESH,
            )
            v_zmin[row, col] = v[0]
            v_zmax[row, col] = v[1]
            v_havg[:, row, col] = v[2]
            v_nrep[:, row, col] = v[3]
            v_pwet[:, row, col] = v[4]
            v_ffit[row, col] = v[5]
            v_navg[row, col] = v[6]

    return (
        z_zmin,
        z_zmax,
        z_volmax,
        z_level,
        u_zmin,
        u_zmax,
        u_havg,
        u_nrep,
        u_pwet,
        u_ffit,
        u_navg,
        v_zmin,
        v_zmax,
        v_havg,
        v_nrep,
        v_pwet,
        v_ffit,
        v_navg,
    )


# order of the tables returned by the kernel
_TABLE_NAMES: tuple[str, ...] = (
    "z_zmin",
    "z_zmax",
    "z_volmax",
    "z_level",
    "u_zmin",
    "u_zmax",
    "u_havg",
    "u_nrep",
    "u_pwet",
    "u_ffit",
    "u_navg",
    "v_zmin",
    "v_zmax",
    "v_havg",
    "v_nrep",
    "v_pwet",
    "v_ffit",
    "v_navg",
)

# tile size of the subgrid elevation GeoTIFF; blocks are aligned to it
_TIFF_TILE_SIZE: int = 256

# elevation and roughness datasets up to this size are loaded into memory before the
# blocks are processed; larger ones are read per block
_MAX_LOADED_DATASET_BYTES: int = 4 * 1024**3


def sfincs_uv_indices(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index of every active cell and u/v point in the SFINCS netcdf subgrid file.

    Vectorized equivalent of `hydromt_sfincs.utils.find_uv_indices`. Cells are
    numbered in SFINCS (column-major) order, ``ip = col * n_rows + row``. Active cells
    are numbered consecutively; u/v points are numbered consecutively over the active
    cells, first the u point (to the next column) and then the v point (to the next
    row), each only if that neighbour is active.

    Args:
        mask: SFINCS mask (rows, cols); cells with mask > 0 are active.

    Returns:
        Tuple of index_nm, index_mu1 and index_nu1, each of length n_rows * n_cols in
        column-major order, with -1 for cells (or u/v points) that are not stored.
    """
    n_rows, n_cols = mask.shape
    dtype = np.int32 if mask.size < np.iinfo(np.int32).max else np.int64
    active: np.ndarray = (mask > 0).T.reshape(-1)  # column-major flatten

    active_2d: np.ndarray = active.reshape(n_cols, n_rows)
    has_v: np.ndarray = np.zeros_like(active_2d)
    has_v[:, :-1] = active_2d[:, :-1] & active_2d[:, 1:]
    has_u: np.ndarray = np.zeros_like(active_2d)
    has_u[:-1, :] = active_2d[:-1, :] & active_2d[1:, :]
    has_u = has_u.reshape(-1)
    has_v = has_v.reshape(-1)

    index_nm: np.ndarray = np.full(active.size, -1, dtype=dtype)
    index_nm[active] = np.arange(int(active.sum()), dtype=dtype)

    points_per_cell: np.ndarray = has_u.astype(dtype) + has_v.astype(dtype)
    first_point: np.ndarray = np.cumsum(points_per_cell, dtype=dtype) - points_per_cell
    index_mu1: np.ndarray = np.where(has_u, first_point, -1).astype(dtype)
    index_nu1: np.ndarray = np.where(
        has_v, first_point + has_u.astype(dtype), -1
    ).astype(dtype)
    return index_nm, index_mu1, index_nu1


def _column_major_to_grid(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Reshapes a column-major flattened array back to a (rows, cols) grid.

    Args:
        values: Array of length rows * cols in column-major order.
        shape: Grid shape (rows, cols).

    Returns:
        View of the values with shape (rows, cols).
    """
    return values.reshape(shape[1], shape[0]).T


def read_subgrid_bed_elevation(path: Path, mask: xr.DataArray) -> xr.DataArray:
    """Reads the minimum bed elevation of each cell (z_zmin) from a SFINCS subgrid file.

    Only z_zmin is needed (outflow boundary levels, initial water volume). hydromt's
    reader (`SfincsModel.subgrid.read`) loads every subgrid table on the full grid,
    which for a fine subgrid over a large basin takes around a hundred GB.

    Args:
        path: Path to the netcdf subgrid file (sfincs_subgrid.nc).
        mask: SFINCS grid mask, with the grid's coordinates.

    Returns:
        z_zmin on the grid in meters, NaN in inactive cells.
    """
    # SFINCS stores a value only for the active cells, in column-major (Fortran)
    # order: the row index runs fastest. Flatten the mask in the same order, so
    # that the n-th active cell gets the n-th value of the file.
    is_active: np.ndarray = (mask.values > 0).T.reshape(-1)
    with xr.open_dataset(path) as ds:
        z_zmin_active: np.ndarray = ds["z_zmin"].values
    z_zmin: np.ndarray = np.full(is_active.size, np.nan, dtype=np.float32)
    z_zmin[is_active] = z_zmin_active
    return xr.DataArray(
        _column_major_to_grid(z_zmin, mask.shape),
        coords=mask.coords,
        dims=mask.dims,
        name="z_zmin",
    )


# spacing (in pixels) of the lattice of points that are projected exactly to UTM;
# pixels in between are interpolated bilinearly (the projection is smooth enough that
# the error over a few tens of meters is far below a millimeter)
_PROJECTION_LATTICE_PIXELS: int = 32

# cell size of the grid index of river line segments (m)
_SEGMENT_INDEX_CELL_SIZE: float = 200.0


@njit(nogil=True, cache=True)
def _build_segment_index(
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell_size: float,
    n_cells_x: int,
    n_cells_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid index of line segments: the segments overlapping each grid cell.

    Args:
        x0: x of the segment start points (m).
        y0: y of the segment start points (m).
        x1: x of the segment end points (m).
        y1: y of the segment end points (m).
        origin_x: x of the lower left corner of the index grid (m).
        origin_y: y of the lower left corner of the index grid (m).
        cell_size: Size of the index cells (m).
        n_cells_x: Number of index cells in x.
        n_cells_y: Number of index cells in y.

    Returns:
        Start of each cell's segments in the segment list (length n_cells + 1, cells
        numbered row by row from the origin) and the segment list.
    """
    n_cells: int = n_cells_x * n_cells_y
    counts = np.zeros(n_cells + 1, dtype=np.int64)
    for pass_number in range(2):
        if pass_number == 1:
            starts = np.zeros(n_cells + 1, dtype=np.int64)
            for cell in range(n_cells):
                starts[cell + 1] = starts[cell] + counts[cell]
            segments = np.empty(starts[n_cells], dtype=np.int64)
            fill = starts[:-1].copy()
        for segment in range(x0.size):
            cx0 = int((min(x0[segment], x1[segment]) - origin_x) // cell_size)
            cx1 = int((max(x0[segment], x1[segment]) - origin_x) // cell_size)
            cy0 = int((min(y0[segment], y1[segment]) - origin_y) // cell_size)
            cy1 = int((max(y0[segment], y1[segment]) - origin_y) // cell_size)
            for cy in range(cy0, cy1 + 1):
                for cx in range(cx0, cx1 + 1):
                    cell = cy * n_cells_x + cx
                    if pass_number == 0:
                        counts[cell] += 1
                    else:
                        segments[fill[cell]] = segment
                        fill[cell] += 1
    return starts, segments


@njit(nogil=True, cache=True)
def _locate_on_nearest_segment(
    px: np.ndarray,
    py: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    length_squared: np.ndarray,
    cell_starts: np.ndarray,
    cell_segments: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell_size: float,
    n_cells_x: int,
    n_cells_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest segment of each point and the position of the point projected on it.

    Exact nearest-segment search: the index cells are searched in rings of growing
    size around the point's cell, until the nearest segment found is closer than any
    segment in the cells beyond the searched rings can be. The search starts from the
    nearest segment of the previous point, which for neighbouring points (in raster
    order) is usually the answer, so that the search mostly ends in the first cell.

    Args:
        px: x of the points (m), inside the index grid.
        py: y of the points (m), inside the index grid.
        x0: x of the segment start points (m).
        y0: y of the segment start points (m).
        dx: x extent of the segments (m).
        dy: y extent of the segments (m).
        length_squared: Squared segment lengths (m2).
        cell_starts: Start of each cell's segments in `cell_segments`.
        cell_segments: Segments per index cell.
        origin_x: x of the lower left corner of the index grid (m).
        origin_y: y of the lower left corner of the index grid (m).
        cell_size: Size of the index cells (m).
        n_cells_x: Number of index cells in x.
        n_cells_y: Number of index cells in y.

    Returns:
        Index of the nearest segment of each point, the fraction (0-1) along that
        segment of the point's projection, and the distance to the segment (m).
    """
    nearest = np.empty(px.size, dtype=np.int64)
    fraction = np.empty(px.size, dtype=np.float64)
    distance = np.empty(px.size, dtype=np.float64)
    max_ring: int = max(n_cells_x, n_cells_y)
    previous_segment: int = -1
    for i in range(px.size):
        cx = int((px[i] - origin_x) // cell_size)
        cy = int((py[i] - origin_y) // cell_size)
        best = np.inf
        best_segment = -1
        best_fraction = 0.0
        if previous_segment >= 0:
            s = previous_segment
            if length_squared[s] > 0.0:
                t = (
                    (px[i] - x0[s]) * dx[s] + (py[i] - y0[s]) * dy[s]
                ) / length_squared[s]
                t = min(max(t, 0.0), 1.0)
            else:
                t = 0.0
            ex = px[i] - (x0[s] + t * dx[s])
            ey = py[i] - (y0[s] + t * dy[s])
            best = ex * ex + ey * ey
            best_segment = s
            best_fraction = t
        for ring in range(max_ring + 1):
            for cell_y in range(cy - ring, cy + ring + 1):
                if cell_y < 0 or cell_y >= n_cells_y:
                    continue
                on_edge_row = cell_y == cy - ring or cell_y == cy + ring
                step = 1 if on_edge_row else 2 * ring
                cell_x = cx - ring
                while cell_x <= cx + ring:
                    # skip cells that cannot hold a closer segment
                    gap_x = max(
                        origin_x + cell_x * cell_size - px[i],
                        px[i] - (origin_x + (cell_x + 1) * cell_size),
                        0.0,
                    )
                    gap_y = max(
                        origin_y + cell_y * cell_size - py[i],
                        py[i] - (origin_y + (cell_y + 1) * cell_size),
                        0.0,
                    )
                    if 0 <= cell_x < n_cells_x and gap_x * gap_x + gap_y * gap_y < best:
                        cell = cell_y * n_cells_x + cell_x
                        for k in range(cell_starts[cell], cell_starts[cell + 1]):
                            s = cell_segments[k]
                            if length_squared[s] > 0.0:
                                t = (
                                    (px[i] - x0[s]) * dx[s] + (py[i] - y0[s]) * dy[s]
                                ) / length_squared[s]
                                t = min(max(t, 0.0), 1.0)
                            else:
                                t = 0.0
                            ex = px[i] - (x0[s] + t * dx[s])
                            ey = py[i] - (y0[s] + t * dy[s])
                            distance_squared = ex * ex + ey * ey
                            if distance_squared < best:
                                best = distance_squared
                                best_segment = s
                                best_fraction = t
                    if step == 0:
                        break
                    cell_x += step
            # cells beyond this ring are at least as far as the edge of the square
            # of searched cells
            edge_distance = min(
                px[i] - (origin_x + (cx - ring) * cell_size),
                origin_x + (cx + ring + 1) * cell_size - px[i],
                py[i] - (origin_y + (cy - ring) * cell_size),
                origin_y + (cy + ring + 1) * cell_size - py[i],
            )
            if best_segment >= 0 and best <= edge_distance * edge_distance:
                break
        nearest[i] = best_segment
        fraction[i] = best_fraction
        distance[i] = np.sqrt(best)
        previous_segment = best_segment
    return nearest, fraction, distance


def subgrid_water_volume(
    subgrid_file: Path,
    mask: xr.DataArray,
    water_level: xr.DataArray,
    cell_area: xr.DataArray,
) -> float:
    """Water volume in the cells with a water level, from the subgrid volume tables.

    Uses the volume as SFINCS represents it: zero below the lowest subgrid pixel of
    a cell, interpolated linearly between the levels of the volume table (each level
    holds an equal share of the maximum volume), and the maximum volume plus the depth
    above the highest level times the cell area above it.

    Args:
        subgrid_file: Path to the netcdf subgrid file (sfincs_subgrid.nc).
        mask: SFINCS grid mask.
        water_level: Water level per cell (m), NaN where not set.
        cell_area: Area of each cell (m2).

    Returns:
        Total water volume (m3).
    """
    level: np.ndarray = water_level.values
    has_level: np.ndarray = np.isfinite(level) & (mask.values > 0)
    if not has_level.any():
        return 0.0
    rows, cols = np.nonzero(has_level)
    index_nm, _, _ = sfincs_uv_indices(mask.values)
    table_index: np.ndarray = _column_major_to_grid(index_nm, mask.shape)[rows, cols]
    with xr.open_dataset(subgrid_file) as ds:
        z_level: np.ndarray = (
            ds["z_level"].transpose("np", "levels").values[table_index]
        )
        volume_max: np.ndarray = ds["z_volmax"].values[table_index]
    level = level[rows, cols]
    nr_levels: int = z_level.shape[1]

    # level band of each cell: z_level[band] <= level < z_level[band + 1]
    band: np.ndarray = np.clip(
        (z_level <= level[:, None]).sum(axis=1) - 1, 0, nr_levels - 2
    )
    cells: np.ndarray = np.arange(level.size)
    z_low: np.ndarray = z_level[cells, band]
    z_high: np.ndarray = z_level[cells, band + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction: np.ndarray = np.where(
            z_high > z_low, (level - z_low) / (z_high - z_low), 1.0
        )
    volume: np.ndarray = (
        (band + np.clip(fraction, 0.0, 1.0)) / (nr_levels - 1) * (volume_max)
    )
    volume = np.where(level <= z_level[:, 0], 0.0, volume)
    above: np.ndarray = level > z_level[:, -1]
    volume = np.where(
        above,
        volume_max + (level - z_level[:, -1]) * cell_area.values[rows, cols],
        volume,
    )
    return float(volume.sum(dtype=np.float64))


class _LineProfile:
    """Bed elevation (and Manning's n) as a function of distance along one line."""

    def __init__(
        self, distance: np.ndarray, values: dict[str, np.ndarray], offset: np.ndarray
    ) -> None:
        """Stores the profile, with points at equal distance merged.

        Follows hydromt-sfincs: if all points project onto the same position, only
        the points closest to the line are used.

        Args:
            distance: Position of each point along the line (m).
            values: Value per point for each quantity (e.g. bed elevation).
            offset: Distance of each point to the line (m).
        """
        if distance.size > 1 and np.unique(distance).size == 1:
            closest: np.ndarray = offset == offset.min()
            distance = distance[closest]
            values = {name: v[closest] for name, v in values.items()}
        order: np.ndarray = np.argsort(distance, kind="stable")
        distance = distance[order]
        unique_distance, inverse = np.unique(distance, return_inverse=True)
        counts: np.ndarray = np.bincount(inverse)
        self.distance: np.ndarray = unique_distance
        self.values: dict[str, np.ndarray] = {
            name: np.bincount(inverse, weights=v[order]) / counts
            for name, v in values.items()
        }

    def interpolate(self, name: str, distance: np.ndarray) -> np.ndarray:
        """Interpolates linearly along the line, extrapolating beyond the end points.

        Args:
            name: Quantity to interpolate.
            distance: Positions along the line (m).

        Returns:
            Interpolated values at the positions.
        """
        x0: np.ndarray = self.distance
        z0: np.ndarray = self.values[name]
        if x0.size == 1:
            return np.full(distance.shape, z0[0])
        z: np.ndarray = np.interp(distance, x0, z0)
        before: np.ndarray = distance < x0[0]
        z[before] = z0[0] + (distance[before] - x0[0]) * (z0[1] - z0[0]) / (
            x0[1] - x0[0]
        )
        after: np.ndarray = distance > x0[-1]
        z[after] = z0[-1] + (distance[after] - x0[-1]) * (z0[-1] - z0[-2]) / (
            x0[-1] - x0[-2]
        )
        return z


# bank-referenced river bed (as hydromt-sfincs' burn_river_rect with river depths): the
# bank elevation is the median DEM elevation along both banks per river segment
_BANK_SEGMENT_LENGTH_M: float = 500.0
# spacing of the bank samples along the river (m)
_BANK_SAMPLE_SPACING_M: float = 10.0
# distance beyond the river edge at which the bank is sampled (m): the centre of the
# first pixel of a 30 m DEM (FathomDEM) outside the channel; hydromt-sfincs samples the
# ring of subgrid pixels just outside the river, which is the same on a 30 m subgrid
_BANK_SAMPLE_OFFSET_M: float = 15.0


def _sample_elevation(
    elevation_list: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    crs: pyproj.CRS,
) -> np.ndarray:
    """Samples the merged elevation of a list of DEMs at points.

    The DEMs are merged as in hydromt-sfincs' merge_multi_dataarrays: the first DEM is
    the base, a later DEM fills its gaps (merge_method "first", the default) or
    replaces it where it has data (merge_method "last"); offset, zmin and zmax apply.

    Args:
        elevation_list: Elevation datasets with key 'da' (DataArray) and optionally
            'offset', 'zmin', 'zmax' and 'merge_method'.
        x: x of the points.
        y: y of the points.
        crs: CRS of the points.

    Returns:
        Elevation at the points (m), NaN where no DEM has data.
    """
    values: np.ndarray = np.full(x.size, np.nan)
    for i, dataset in enumerate(elevation_list):
        da: xr.DataArray = dataset["da"]
        x_dem, y_dem = pyproj.Transformer.from_crs(
            crs, da.rio.crs, always_xy=True
        ).transform(x, y)
        col, row = ~da.rio.transform(recalc=True) * (
            np.asarray(x_dem),
            np.asarray(y_dem),
        )
        row = np.floor(row).astype(np.int64)
        col = np.floor(col).astype(np.int64)
        inside: np.ndarray = (
            (row >= 0) & (row < da.sizes["y"]) & (col >= 0) & (col < da.sizes["x"])
        )
        sampled: np.ndarray = np.full(x.size, np.nan)
        if inside.any():
            sampled_inside: np.ndarray = (
                da.isel(
                    y=xr.DataArray(row[inside], dims="points"),
                    x=xr.DataArray(col[inside], dims="points"),
                )
                .values.astype(np.float64)
                .ravel()
            )
            nodata = da.rio.nodata
            if nodata is not None and not np.isnan(nodata):
                sampled_inside[sampled_inside == nodata] = np.nan
            sampled[inside] = sampled_inside
        with np.errstate(invalid="ignore"):
            if dataset.get("zmin") is not None:
                sampled[sampled < dataset["zmin"]] = np.nan
            if dataset.get("zmax") is not None:
                sampled[sampled > dataset["zmax"]] = np.nan
        if dataset.get("offset") is not None:
            sampled += dataset["offset"]
        if i > 0 and dataset.get("merge_method", "first") == "last":
            values = np.where(np.isfinite(sampled), sampled, values)
        else:
            values = np.where(np.isnan(values), sampled, values)
    return values


def bank_referenced_river_bed_points(
    rivers: gpd.GeoDataFrame,
    elevation_list: list[dict[str, Any]],
    utm_crs: pyproj.CRS,
) -> gpd.GeoDataFrame:
    """River bed points at the bank elevation minus the river depth.

    As hydromt-sfincs' burn_river_rect when river depths are given: each river is cut
    into segments of about 500 m; the bank elevation of a segment is the median of the
    DEM along both banks (sampled every 10 m, just outside the river); the bed is the
    bank elevation minus the bankfull depth, at the middle of the segment.

    Args:
        rivers: River center lines with columns 'width' and 'depth' (m).
        elevation_list: Elevation datasets, merged as for the subgrid (see
            `_sample_elevation`).
        utm_crs: Metric CRS for distances along and across the rivers.

    Returns:
        Points (in the UTM CRS) with columns 'river_id', 'bank_elevation_m',
        'depth_m' and 'bed_elevation_m', for segments with bank elevation data.
    """
    rivers_utm: gpd.GeoDataFrame = rivers.to_crs(utm_crs)
    lines: np.ndarray = rivers_utm.geometry.values
    length: np.ndarray = shapely.length(lines)
    half_width: np.ndarray = rivers_utm["width"].fillna(2.0).to_numpy() / 2
    depth: np.ndarray = rivers_utm["depth"].to_numpy(dtype=np.float64)
    n_segments: np.ndarray = np.maximum(
        1, np.round(length / _BANK_SEGMENT_LENGTH_M)
    ).astype(np.int64)
    segment_length: np.ndarray = length / n_segments

    # sample positions along each river, in the middle of equal intervals
    n_samples: np.ndarray = np.maximum(
        1, np.ceil(length / _BANK_SAMPLE_SPACING_M)
    ).astype(np.int64)
    river: np.ndarray = np.repeat(np.arange(len(lines)), n_samples)
    sample_index: np.ndarray = np.arange(river.size) - np.repeat(
        np.cumsum(n_samples) - n_samples, n_samples
    )
    position: np.ndarray = (sample_index + 0.5) / n_samples[river] * length[river]
    centre = shapely.get_coordinates(
        shapely.line_interpolate_point(lines[river], position)
    )
    ahead = shapely.get_coordinates(
        shapely.line_interpolate_point(
            lines[river], np.minimum(position + 1.0, length[river])
        )
    )
    behind = shapely.get_coordinates(
        shapely.line_interpolate_point(lines[river], np.maximum(position - 1.0, 0.0))
    )
    tangent: np.ndarray = ahead - behind
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal: np.ndarray = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    distance: np.ndarray = (half_width[river] + _BANK_SAMPLE_OFFSET_M)[:, None]
    banks: np.ndarray = np.concatenate(
        [centre + normal * distance, centre - normal * distance]
    )
    bank_elevation: np.ndarray = _sample_elevation(
        elevation_list, banks[:, 0], banks[:, 1], pyproj.CRS(utm_crs)
    )

    # median bank elevation per segment (both banks)
    segment: np.ndarray = np.minimum(
        (position // segment_length[river]).astype(np.int64), n_segments[river] - 1
    )
    segment_id: np.ndarray = (np.cumsum(n_segments) - n_segments)[river] + segment
    samples = pd.DataFrame(
        {"segment": np.tile(segment_id, 2), "elevation": bank_elevation}
    ).dropna()
    median_bank: pd.Series = samples.groupby("segment")["elevation"].median()

    segment_river: np.ndarray = np.repeat(np.arange(len(lines)), n_segments)
    segment_number: np.ndarray = np.arange(segment_river.size) - np.repeat(
        np.cumsum(n_segments) - n_segments, n_segments
    )
    midpoints = shapely.line_interpolate_point(
        lines[segment_river], (segment_number + 0.5) * segment_length[segment_river]
    )
    bank: np.ndarray = median_bank.reindex(np.arange(segment_river.size)).to_numpy()
    points = gpd.GeoDataFrame(
        {
            "river_id": rivers_utm.index.to_numpy()[segment_river],
            "bank_elevation_m": bank,
            "depth_m": depth[segment_river],
            "bed_elevation_m": bank - depth[segment_river],
        },
        geometry=midpoints,
        crs=utm_crs,
    )
    return points[np.isfinite(points["bed_elevation_m"])].reset_index(drop=True)


class RiverBurner:
    """Burns rivers into subgrid blocks, prepared once for the whole model.

    Same method as hydromt-sfincs' `burn_river_rect` with river bed points: the river
    area is the center lines buffered by half the river width; in that area, the bed
    level (and Manning's n) is interpolated linearly along the merged center lines from
    the bed points, and the elevation is lowered to the bed level where it is higher.
    The center lines, polygons and profiles are built once for all rivers, so the bed
    level does not depend on the block boundaries.
    """

    def __init__(
        self,
        rivers: gpd.GeoDataFrame,
        river_bed_points: gpd.GeoDataFrame,
        grid_crs: pyproj.CRS,
        utm_crs: pyproj.CRS,
        min_width_m: float,
    ) -> None:
        """Prepares the river polygons, merged center lines and bed profiles.

        Args:
            rivers: River center lines with columns 'width' (m), 'shape_exponent'
                (-), the power-law exponent r of the channel shape of the river routing
                (0.5: parabolic), and optionally 'manning' (s/m^(1/3)). The bed lies at
                the bank level minus depth * (1 - (2y / W) ** (1 / r)) at distance y
                from the center line, so that the cross-section area is
                W * depth / (r + 1).
            river_bed_points: Points with columns 'bed_elevation_m' (m), the bed on
                the center line, and 'depth_m' (m), the bankfull depth.
            grid_crs: CRS of the SFINCS grid.
            utm_crs: Metric CRS in which distances and buffers are computed.
            min_width_m: Minimum river width (m), the subgrid pixel size, so that each
                river is at least one pixel wide.
        """
        self.grid_crs: pyproj.CRS = pyproj.CRS(grid_crs)
        self.utm_crs: pyproj.CRS = pyproj.CRS(utm_crs)
        self._local: threading.local = threading.local()

        rivers_utm: gpd.GeoDataFrame = rivers.to_crs(self.utm_crs)
        half_width: np.ndarray = (
            np.maximum(rivers_utm["width"].fillna(2.0).to_numpy(), min_width_m) / 2
        )
        polygons: gpd.GeoSeries = gpd.GeoSeries(
            rivers_utm.geometry.buffer(half_width), crs=self.utm_crs
        ).to_crs(self.grid_crs)
        self.polygons: np.ndarray = polygons.values
        self.polygon_tree: shapely.STRtree = shapely.STRtree(self.polygons)

        # river bed points, with Manning's n of the nearest river
        points: gpd.GeoDataFrame = river_bed_points.to_crs(self.utm_crs)
        point_geometries: np.ndarray = points.geometry.values
        values: dict[str, np.ndarray] = {
            "bed": points["bed_elevation_m"].to_numpy(dtype=np.float64),
            "depth": points["depth_m"].to_numpy(dtype=np.float64),
        }
        if "manning" in rivers_utm.columns:
            nearest_river: np.ndarray = shapely.STRtree(
                rivers_utm.geometry.values
            ).query_nearest(point_geometries, all_matches=False)[1]
            values["manning"] = rivers_utm["manning"].to_numpy(dtype=np.float64)[
                nearest_river
            ]
        valid: np.ndarray = np.logical_and.reduce(
            [np.isfinite(v) for v in values.values()]
        )
        point_geometries = point_geometries[valid]
        values = {name: v[valid] for name, v in values.items()}

        # merged center lines, as in hydromt-sfincs (merged at the junctions)
        merged = shapely.line_merge(shapely.union_all(rivers_utm.geometry.values))
        lines: np.ndarray = shapely.get_parts(merged)
        point_line: np.ndarray = shapely.STRtree(lines).query_nearest(
            point_geometries, all_matches=False
        )[1]
        point_distance: np.ndarray = shapely.line_locate_point(
            lines[point_line], point_geometries
        )
        point_offset: np.ndarray = shapely.distance(lines[point_line], point_geometries)

        # only lines with bed points are used for interpolation
        line_ids: np.ndarray = np.unique(point_line)
        self.lines: np.ndarray = lines[line_ids]

        # segments of these lines, with a grid index for the nearest-segment search
        coordinates, vertex_line = shapely.get_coordinates(
            self.lines, return_index=True
        )
        is_segment: np.ndarray = vertex_line[:-1] == vertex_line[1:]
        start: np.ndarray = coordinates[:-1][is_segment]
        end: np.ndarray = coordinates[1:][is_segment]
        self.segment_line: np.ndarray = vertex_line[:-1][is_segment]
        self.segment_x0: np.ndarray = np.ascontiguousarray(start[:, 0])
        self.segment_y0: np.ndarray = np.ascontiguousarray(start[:, 1])
        self.segment_dx: np.ndarray = end[:, 0] - start[:, 0]
        self.segment_dy: np.ndarray = end[:, 1] - start[:, 1]
        self.segment_length_squared: np.ndarray = (
            self.segment_dx**2 + self.segment_dy**2
        )
        segment_length: np.ndarray = np.sqrt(self.segment_length_squared)
        # distance along the line at the start of each segment
        cumulative: np.ndarray = np.cumsum(segment_length)
        line_start: np.ndarray = np.r_[
            True, self.segment_line[1:] != self.segment_line[:-1]
        ]
        offset_of_line: np.ndarray = (cumulative - segment_length)[line_start]
        self.segment_start_distance: np.ndarray = (
            cumulative
            - segment_length
            - np.repeat(
                offset_of_line,
                np.diff(np.r_[np.flatnonzero(line_start), cumulative.size]),
            )
        )
        self.segment_length: np.ndarray = segment_length
        # half width and channel shape of each segment: those of the nearest river
        # at its middle
        segment_middle: np.ndarray = shapely.points(
            self.segment_x0 + self.segment_dx / 2, self.segment_y0 + self.segment_dy / 2
        )
        segment_river: np.ndarray = shapely.STRtree(
            rivers_utm.geometry.values
        ).query_nearest(segment_middle, all_matches=False)[1]
        self.segment_half_width: np.ndarray = half_width[segment_river]
        self.segment_shape_exponent: np.ndarray = rivers_utm["shape_exponent"].to_numpy(
            dtype=np.float64
        )[segment_river]

        # index grid covering the rivers with a margin wider than any river polygon,
        # so that all river pixels fall inside it
        margin: float = float(half_width.max()) + 2 * _SEGMENT_INDEX_CELL_SIZE
        all_x: np.ndarray = np.r_[self.segment_x0, self.segment_x0 + self.segment_dx]
        all_y: np.ndarray = np.r_[self.segment_y0, self.segment_y0 + self.segment_dy]
        self.index_origin_x: float = float(all_x.min()) - margin
        self.index_origin_y: float = float(all_y.min()) - margin
        self.index_cells_x: int = (
            int(
                (all_x.max() + margin - self.index_origin_x) // _SEGMENT_INDEX_CELL_SIZE
            )
            + 1
        )
        self.index_cells_y: int = (
            int(
                (all_y.max() + margin - self.index_origin_y) // _SEGMENT_INDEX_CELL_SIZE
            )
            + 1
        )
        self.index_cell_starts, self.index_cell_segments = _build_segment_index(
            self.segment_x0,
            self.segment_y0,
            self.segment_x0 + self.segment_dx,
            self.segment_y0 + self.segment_dy,
            self.index_origin_x,
            self.index_origin_y,
            _SEGMENT_INDEX_CELL_SIZE,
            self.index_cells_x,
            self.index_cells_y,
        )
        self.profiles: list[_LineProfile] = []
        for line_id in line_ids:
            on_line: np.ndarray = point_line == line_id
            self.profiles.append(
                _LineProfile(
                    distance=point_distance[on_line],
                    values={name: v[on_line] for name, v in values.items()},
                    offset=point_offset[on_line],
                )
            )
        self.has_manning: bool = "manning" in values

    def _pixels_to_utm(
        self, rows: np.ndarray, cols: np.ndarray, transform: Affine
    ) -> tuple[np.ndarray, np.ndarray]:
        """UTM coordinates of pixel centres.

        A lattice of points every `_PROJECTION_LATTICE_PIXELS` pixels is projected
        exactly; the pixel centres are interpolated bilinearly between them.

        Args:
            rows: Pixel rows.
            cols: Pixel columns.
            transform: Affine transform of the raster.

        Returns:
            x and y of the pixel centres in UTM (m).
        """
        spacing: int = _PROJECTION_LATTICE_PIXELS
        n_lattice_rows: int = int(rows.max()) // spacing + 2
        n_lattice_cols: int = int(cols.max()) // spacing + 2
        lattice_cols, lattice_rows = np.meshgrid(
            np.arange(n_lattice_cols) * spacing, np.arange(n_lattice_rows) * spacing
        )
        lattice_x, lattice_y = transform * (lattice_cols, lattice_rows)
        lattice_x_utm, lattice_y_utm = self._to_utm.transform(lattice_x, lattice_y)

        row_position: np.ndarray = (rows + 0.5) / spacing
        col_position: np.ndarray = (cols + 0.5) / spacing
        row0: np.ndarray = row_position.astype(np.intp)
        col0: np.ndarray = col_position.astype(np.intp)
        row_weight: np.ndarray = row_position - row0
        col_weight: np.ndarray = col_position - col0

        def interpolate(values: np.ndarray) -> np.ndarray:
            """Bilinear interpolation of lattice values at the pixel centres.

            Args:
                values: Values at the lattice points.

            Returns:
                Interpolated values at the pixel centres.
            """
            return (1 - row_weight) * (
                (1 - col_weight) * values[row0, col0]
                + col_weight * values[row0, col0 + 1]
            ) + row_weight * (
                (1 - col_weight) * values[row0 + 1, col0]
                + col_weight * values[row0 + 1, col0 + 1]
            )

        return interpolate(np.asarray(lattice_x_utm)), interpolate(
            np.asarray(lattice_y_utm)
        )

    @property
    def _to_utm(self) -> pyproj.Transformer:
        """Transformer from the grid CRS to UTM, one per thread (not thread-safe)."""
        transformer: pyproj.Transformer | None = getattr(
            self._local, "transformer", None
        )
        if transformer is None:
            transformer = pyproj.Transformer.from_crs(
                self.grid_crs, self.utm_crs, always_xy=True
            )
            self._local.transformer = transformer
        return transformer

    def burn(
        self, elevation: np.ndarray, manning: np.ndarray, transform: Affine
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Burns the rivers into one block, in place.

        Args:
            elevation: Subgrid elevation of the block (m), lowered in place.
            manning: Subgrid Manning's n of the block (s/m^(1/3)), set in place in the
                river area if the rivers have Manning's n.
            transform: Affine transform of the block.

        Returns:
            Rows, columns and the original (unburned) elevation of the pixels whose
            elevation was lowered.
        """
        empty: tuple[np.ndarray, np.ndarray, np.ndarray] = (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=elevation.dtype),
        )
        if not self.profiles:
            return empty
        n_rows, n_cols = elevation.shape
        west, north = transform * (0, 0)
        east, south = transform * (n_cols, n_rows)
        block_box = shapely.box(
            min(west, east), min(south, north), max(west, east), max(south, north)
        )
        polygons: np.ndarray = self.polygons[self.polygon_tree.query(block_box)]
        if polygons.size == 0:
            return empty
        in_river: np.ndarray = geometry_mask(
            polygons,
            out_shape=elevation.shape,
            transform=transform,
            all_touched=False,
            invert=True,
        )
        rows, cols = np.nonzero(in_river)
        if rows.size == 0:
            return empty

        x_utm, y_utm = self._pixels_to_utm(rows, cols, transform)
        x_utm = np.clip(
            np.asarray(x_utm, dtype=np.float64),
            self.index_origin_x,
            self.index_origin_x + self.index_cells_x * _SEGMENT_INDEX_CELL_SIZE - 1e-6,
        )
        y_utm = np.clip(
            np.asarray(y_utm, dtype=np.float64),
            self.index_origin_y,
            self.index_origin_y + self.index_cells_y * _SEGMENT_INDEX_CELL_SIZE - 1e-6,
        )
        segment, fraction, pixel_offset = _locate_on_nearest_segment(
            x_utm,
            y_utm,
            self.segment_x0,
            self.segment_y0,
            self.segment_dx,
            self.segment_dy,
            self.segment_length_squared,
            self.index_cell_starts,
            self.index_cell_segments,
            self.index_origin_x,
            self.index_origin_y,
            _SEGMENT_INDEX_CELL_SIZE,
            self.index_cells_x,
            self.index_cells_y,
        )
        pixel_line: np.ndarray = self.segment_line[segment]
        pixel_distance: np.ndarray = (
            self.segment_start_distance[segment]
            + fraction * self.segment_length[segment]
        )

        bed: np.ndarray = np.empty(rows.size, dtype=np.float64)
        river_depth: np.ndarray = np.empty(rows.size, dtype=np.float64)
        river_manning: np.ndarray = np.empty(rows.size, dtype=np.float64)
        order: np.ndarray = np.argsort(pixel_line, kind="stable")
        line_ids, starts = np.unique(pixel_line[order], return_index=True)
        ends: np.ndarray = np.append(starts[1:], order.size)
        for line_id, start, end in zip(line_ids, starts, ends, strict=True):
            on_line: np.ndarray = order[start:end]
            profile: _LineProfile = self.profiles[line_id]
            bed[on_line] = profile.interpolate("bed", pixel_distance[on_line])
            river_depth[on_line] = profile.interpolate("depth", pixel_distance[on_line])
            if self.has_manning:
                river_manning[on_line] = profile.interpolate(
                    "manning", pixel_distance[on_line]
                )

        # power-law channel: full depth on the center line, none at the banks
        relative_offset: np.ndarray = np.clip(
            pixel_offset / self.segment_half_width[segment], 0.0, 1.0
        )
        bed += river_depth * relative_offset ** (
            1.0 / self.segment_shape_exponent[segment]
        )

        if self.has_manning:
            manning[rows, cols] = river_manning
        unburned: np.ndarray = elevation[rows, cols]
        lowered: np.ndarray = bed < unburned
        rows, cols = rows[lowered], cols[lowered]
        unburned = unburned[lowered]
        elevation[rows, cols] = bed[lowered]
        return rows, cols, unburned


@dataclass
class _Block:
    """A block of grid cells processed together."""

    row_start: int
    row_end: int
    col_start: int
    col_end: int


@dataclass
class _MergeRaster:
    """An elevation or roughness dataset to merge into the subgrid blocks.

    Prepared once in the main thread, so that the blocks are merged with numpy and
    rasterio only. hydromt-sfincs' `merge_multi_dataarrays` creates xarray objects
    with cached accessors per block, which form reference cycles: their arrays are
    only freed by the cycle collector, so that memory grew by a block's worth of
    arrays with every block (240 GB for the Meuse at 30 m with a 1 m subgrid).
    """

    data: np.ndarray | xr.DataArray
    """Values (y, x), or a lazily read DataArray for very large datasets."""
    transform: Affine
    crs: pyproj.CRS
    nodata: float | None
    resolution_m: float
    """Pixel size (m), for the default resampling method."""
    reproject_method: str | None
    merge_method: str
    offset: float | None
    min_valid: float | None
    max_valid: float | None


def _prepare_merge_rasters(datasets: list[dict[str, Any]]) -> list[_MergeRaster]:
    """Prepares parsed hydromt-sfincs elevation or roughness datasets for merging.

    Datasets up to `_MAX_LOADED_DATASET_BYTES` are loaded into memory; larger ones
    (e.g. a 1 m DEM of a long river reach) are read per block.

    Args:
        datasets: Datasets as returned by `_parse_elevation_list` or
            `_parse_roughness_list` (keys 'da' and optionally 'reproj_method',
            'merge_method', 'offset', 'zmin', 'zmax').

    Returns:
        The prepared datasets, in merge order.

    Raises:
        NotImplementedError: For valid-area masks or spatially varying offsets,
            which GEB does not use.
    """
    rasters: list[_MergeRaster] = []
    for dataset in datasets:
        if "gdf_valid" in dataset:
            raise NotImplementedError("Valid-area masks are not supported.")
        offset = dataset.get("offset")
        if offset is not None and not isinstance(offset, (int, float)):
            raise NotImplementedError("Only uniform offsets are supported.")
        da: xr.DataArray = dataset["da"]
        da = da.transpose(da.rio.y_dim, da.rio.x_dim)
        crs: pyproj.CRS = pyproj.CRS.from_user_input(da.rio.crs)
        transform: Affine = da.rio.transform(recalc=True)
        rasters.append(
            _MergeRaster(
                data=da.values if da.nbytes <= _MAX_LOADED_DATASET_BYTES else da,
                transform=transform,
                crs=crs,
                nodata=da.rio.nodata,
                resolution_m=abs(transform.a)
                * (111111.0 if crs.is_geographic else 1.0),
                reproject_method=dataset.get("reproj_method"),
                merge_method=dataset.get("merge_method", "first"),
                offset=None if offset is None else float(offset),
                min_valid=dataset.get("zmin"),
                max_valid=dataset.get("zmax"),
            )
        )
    return rasters


def _read_raster_window(
    raster: _MergeRaster,
    dst_transform: Affine,
    dst_shape: tuple[int, int],
    dst_crs: pyproj.CRS,
) -> tuple[np.ndarray, Affine] | None:
    """Reads the part of a dataset that covers a destination grid.

    As hydromt's `clip_bbox` with a buffer of two pixels around the bounds of the
    destination grid.

    Args:
        raster: The dataset.
        dst_transform: Transform of the destination grid.
        dst_shape: Shape (rows, columns) of the destination grid.
        dst_crs: CRS of the destination grid.

    Returns:
        The values and their transform, or None if at most two pixels lie in the
        window along an axis.
    """
    xs: list[float] = [
        dst_transform.c,
        dst_transform.c + dst_shape[1] * dst_transform.a,
    ]
    ys: list[float] = [
        dst_transform.f,
        dst_transform.f + dst_shape[0] * dst_transform.e,
    ]
    bounds: tuple[float, float, float, float] = (min(xs), min(ys), max(xs), max(ys))
    if raster.crs != dst_crs:
        bounds = rasterio.warp.transform_bounds(dst_crs, raster.crs, *bounds)
    west, south, east, north = bounds
    cols, rows = raster.transform.__invert__() * (
        np.array([west, east]),
        np.array([south, north]),
    )
    buffer: int = 2
    col_start: int = max(int(np.round(cols.min() - buffer)), 0)
    row_start: int = max(int(np.round(rows.min() - buffer)), 0)
    col_end: int = max(int(np.round(cols.max() + buffer)), 0)
    row_end: int = max(int(np.round(rows.max() + buffer)), 0)
    n_rows, n_cols = raster.data.shape
    col_end, row_end = min(col_end, n_cols), min(row_end, n_rows)
    if col_end - col_start <= 2 or row_end - row_start <= 2:
        return None
    window = raster.data[row_start:row_end, col_start:col_end]
    values: np.ndarray = np.asarray(
        window.values if isinstance(window, xr.DataArray) else window
    )
    return values, raster.transform * Affine.translation(col_start, row_start)


def _resample_raster(
    raster: _MergeRaster,
    method: str,
    dst_transform: Affine,
    dst_shape: tuple[int, int],
    dst_crs: pyproj.CRS,
) -> np.ndarray | None:
    """Resamples a dataset to a destination grid, as hydromt's `reproject_like`.

    Args:
        raster: The dataset.
        method: rasterio resampling method (e.g. 'bilinear').
        dst_transform: Transform of the destination grid.
        dst_shape: Shape (rows, columns) of the destination grid.
        dst_crs: CRS of the destination grid.

    Returns:
        The values on the destination grid with no data as NaN (offset and valid
        range applied), or None if the dataset does not cover the grid.
    """
    window = _read_raster_window(raster, dst_transform, dst_shape, dst_crs)
    if window is None:
        return None
    source, source_transform = window
    dst_nodata = raster.nodata if raster.nodata is not None else np.nan
    values: np.ndarray = np.full(dst_shape, dst_nodata, dtype=source.dtype)
    rasterio.warp.reproject(
        source=source,
        destination=values,
        src_transform=source_transform,
        src_crs=raster.crs,
        src_nodata=raster.nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=dst_nodata,
        resampling=getattr(Resampling, method),
    )
    if raster.nodata is not None and not np.isnan(raster.nodata):
        values = np.where(values == raster.nodata, np.nan, values)
    with np.errstate(invalid="ignore"):
        if raster.offset is not None:
            values = np.where(np.isnan(values), values, values + raster.offset)
        if raster.min_valid is not None:
            values = np.where(values >= raster.min_valid, values, np.nan)
        if raster.max_valid is not None:
            values = np.where(values <= raster.max_valid, values, np.nan)
    return values


def _merge_rasters(
    rasters: list[_MergeRaster],
    dst_transform: Affine,
    dst_shape: tuple[int, int],
    dst_crs: pyproj.CRS,
) -> np.ndarray:
    """Merges datasets on a grid, as hydromt-sfincs' `merge_multi_dataarrays`.

    The first dataset is resampled to the grid (bilinear when its pixels are at
    least as large as the grid's, otherwise average, unless a method is given);
    each next dataset is resampled likewise (compared with the first dataset's
    pixel size) and merged by its merge method: 'first' (fill no data), 'last'
    (overwrite where valid), 'mean', 'max' or 'min', with hydromt-sfincs'
    handling of no data.

    Args:
        rasters: Prepared datasets, in merge order.
        dst_transform: Transform of the destination grid.
        dst_shape: Shape (rows, columns) of the destination grid.
        dst_crs: CRS of the destination grid.

    Returns:
        The merged values, NaN where no dataset has data.

    Raises:
        ValueError: For an unknown merge method.
    """
    dst_resolution_m: float = abs(dst_transform.a) * (
        111111.0 if dst_crs.is_geographic else 1.0
    )
    first: _MergeRaster = rasters[0]
    method: str = first.reproject_method or (
        "bilinear" if first.resolution_m >= dst_resolution_m else "average"
    )
    merged = _resample_raster(first, method, dst_transform, dst_shape, dst_crs)
    if merged is None:
        merged = np.full(dst_shape, np.nan, dtype=np.float32)
    dtype = merged.dtype
    for raster in rasters[1:]:
        if raster.merge_method == "first" and not np.isnan(merged).any():
            continue
        method = raster.reproject_method or (
            "bilinear" if raster.resolution_m >= first.resolution_m else "average"
        )
        values = _resample_raster(raster, method, dst_transform, dst_shape, dst_crs)
        if values is None:
            continue
        with np.errstate(invalid="ignore"):
            if raster.merge_method == "first":
                keep = np.logical_not(np.isnan(merged))
            elif raster.merge_method == "last":
                keep = np.isnan(values)
            elif raster.merge_method == "mean":
                keep = np.isnan(merged)
                values = (merged + values) / 2
            elif raster.merge_method == "max":
                keep = merged >= values
            elif raster.merge_method == "min":
                keep = merged <= values
            else:
                raise ValueError(f"Unknown merge method: {raster.merge_method}")
        merged = np.where(keep, merged, values).astype(dtype)
    return merged


def _fill_nodata(values: np.ndarray, transform: Affine) -> np.ndarray:
    """Fills no data, as hydromt's `interpolate_na(method="rio_idw", extrapolate=True)`.

    Inverse distance weighting (rasterio's `fillnodata`), then the nearest value
    for pixels that are still empty.

    Args:
        values: Values with no data as NaN.
        transform: Transform of the values.

    Returns:
        The filled values.
    """
    valid: np.ndarray = np.logical_not(np.isnan(values))
    if not valid.any() or valid.all():
        return values
    filled: np.ndarray = rasterio.fill.fillnodata(values.copy(), valid)
    empty: np.ndarray = np.isnan(filled)
    if empty.any():
        rows, cols = np.indices(values.shape)
        xs: np.ndarray = transform.c + (cols + 0.5) * transform.a
        ys: np.ndarray = transform.f + (rows + 0.5) * transform.e
        known: np.ndarray = np.logical_not(empty)
        filled[empty] = griddata(
            points=(xs[known], ys[known]),
            values=filled[known],
            xi=(xs[empty], ys[empty]),
            method="nearest",
            fill_value=np.nan,
        )
    return filled


@dataclass
class _BlockResult:
    """Subgrid tables and by-products of one block."""

    block: _Block
    tables: tuple[np.ndarray, ...]
    elevation: np.ndarray | None
    river_cells: np.ndarray
    river_water_level: np.ndarray
    step_seconds: dict[str, float]
    """Thread time per step of the block (s), for the timing summary."""


@dataclass
class SubgridResult:
    """Grids derived during the subgrid build that GEB uses afterwards."""

    bed_elevation: xr.DataArray
    """Minimum subgrid elevation in each cell (z_zmin), in meters."""

    river_initial_water_level: xr.DataArray
    """Per active cell with burned river pixels, the lowest unburned elevation of those
    pixels (m), NaN elsewhere."""


def _block_size_in_cells(nr_subgrid_pixels: int, block_size_pixels: int) -> int:
    """Number of cells per block side, so that blocks align with the GeoTIFF tiles.

    Args:
        nr_subgrid_pixels: Subgrid pixels per cell along each axis.
        block_size_pixels: Desired block size in subgrid pixels.

    Returns:
        Number of cells along each side of a block: a multiple of the number of
        cells that spans a whole number of GeoTIFF tiles, close to the desired size.
    """
    cells_per_tile_multiple: int = _TIFF_TILE_SIZE // gcd(
        _TIFF_TILE_SIZE, nr_subgrid_pixels
    )
    multiples: int = max(
        1, round(block_size_pixels / nr_subgrid_pixels / cells_per_tile_multiple)
    )
    return multiples * cells_per_tile_multiple


def build_subgrid(
    sfincs_model: SfincsModel,
    elevation_list: list[dict[str, Any]],
    roughness_list: list[dict[str, Any]],
    river_burner: RiverBurner | None,
    nr_subgrid_pixels: int,
    nr_levels: int,
    logger: logging.Logger,
    write_dep_tif: bool = True,
    block_size_pixels: int = 4000,
    n_workers: int | None = None,
) -> SubgridResult:
    """Builds and writes the subgrid tables of a regular SFINCS grid in parallel.

    Replaces `SfincsSubgridTable.create` followed by its netcdf writer, with the
    defaults GEB uses (hydromt-sfincs' defaults for the table options). The tables are
    written to `sfincs_subgrid.nc` and the subgrid elevation, for downscaling flood
    maps, to `subgrid/dep_subgrid.tif` in the model root. The model's subgrid component
    is left empty, so that `SfincsModel.write` does not write the tables again.

    Args:
        sfincs_model: The hydromt-sfincs model, with the grid and mask set.
        elevation_list: Elevation datasets, as for `SfincsSubgridTable.create`.
        roughness_list: Manning's n datasets, as for `SfincsSubgridTable.create`.
        river_burner: The rivers to burn, or None for no rivers.
        nr_subgrid_pixels: Subgrid pixels per cell along each axis (even).
        nr_levels: Number of levels of the tables.
        logger: Logger for progress messages.
        write_dep_tif: Whether to write the subgrid elevation GeoTIFF.
        block_size_pixels: Approximate block size in subgrid pixels along each axis;
            bounds the memory per thread.
        n_workers: Number of threads; default: the CPUs available to this process.

    Returns:
        The bed elevation (z_zmin) and the initial water level in river cells.

    Raises:
        ValueError: If the number of subgrid pixels is odd.
    """
    if nr_subgrid_pixels % 2 != 0:
        raise ValueError("nr_subgrid_pixels must be even for the subgrid tables")
    if n_workers is None:
        n_workers = len(os.sched_getaffinity(0))
    start_time: float = time.perf_counter()

    mask_da: xr.DataArray = sfincs_model.grid.mask
    mask: np.ndarray = mask_da.values
    n_rows, n_cols = mask.shape
    crs: pyproj.CRS = pyproj.CRS.from_user_input(mask_da.rio.crs)
    is_geographic: bool = bool(crs.is_geographic)
    grid_transform: Affine = mask_da.rio.transform(recalc=True)
    dx, dy = mask_da.rio.resolution(recalc=True)
    refi: int = nr_subgrid_pixels
    dxp, dyp = dx / refi, dy / refi
    subgrid_transform: Affine = grid_transform * Affine.scale(1 / refi)

    # elevation and roughness datasets, loaded once (read by all threads)
    pixel_size: float = abs(dx) / refi * (111111.0 if is_geographic else 1.0)
    elevation_list = sfincs_model._parse_elevation_list(elevation_list, res=pixel_size)
    if roughness_list:
        roughness_list = sfincs_model._parse_roughness_list(roughness_list)
    # prepared once (read by all threads); very large datasets are read per block
    elevation_rasters: list[_MergeRaster] = _prepare_merge_rasters(elevation_list)
    roughness_rasters: list[_MergeRaster] = (
        _prepare_merge_rasters(roughness_list) if roughness_list else []
    )
    grid_crs: pyproj.CRS = pyproj.CRS.from_user_input(crs)

    # compact tables of active cells and u/v points, as in the netcdf file
    index_nm, index_mu1, index_nu1 = sfincs_uv_indices(mask)
    n_cells: int = int((index_nm >= 0).sum())
    n_uv: int = int(max(index_mu1.max(), index_nu1.max()) + 1)
    index_nm_grid: np.ndarray = _column_major_to_grid(index_nm, mask.shape)
    index_mu1_grid: np.ndarray = _column_major_to_grid(index_mu1, mask.shape)
    index_nu1_grid: np.ndarray = _column_major_to_grid(index_nu1, mask.shape)
    z_tables: dict[str, np.ndarray] = {
        "z_zmin": np.full(n_cells, np.nan, dtype=np.float32),
        "z_zmax": np.full(n_cells, np.nan, dtype=np.float32),
        "z_volmax": np.full(n_cells, np.nan, dtype=np.float32),
        "z_level": np.full((nr_levels, n_cells), np.nan, dtype=np.float32),
    }
    uv_tables: dict[str, np.ndarray] = {
        name: np.zeros(n_uv, dtype=np.float32)
        for name in ("zmin", "zmax", "ffit", "navg")
    } | {
        name: np.zeros((nr_levels, n_uv), dtype=np.float32)
        for name in ("havg", "nrep", "pwet")
    }
    river_initial_water_level: np.ndarray = np.full(
        mask.shape, np.nan, dtype=np.float32
    )

    # blocks with active cells, aligned with the GeoTIFF tiles
    cells_per_block: int = _block_size_in_cells(refi, block_size_pixels)
    blocks: list[_Block] = []
    for row_start in range(0, n_rows, cells_per_block):
        for col_start in range(0, n_cols, cells_per_block):
            block = _Block(
                row_start,
                min(row_start + cells_per_block, n_rows),
                col_start,
                min(col_start + cells_per_block, n_cols),
            )
            if (
                mask[block.row_start : block.row_end, block.col_start : block.col_end]
                > 0
            ).any():
                blocks.append(block)
    logger.info(
        f"Building subgrid tables ({time.perf_counter() - start_time:.0f} s of "
        f"preparation): {n_rows} x {n_cols} cells ({n_cells} active), "
        f"{refi} x {refi} subgrid pixels per cell, {len(blocks)} blocks of "
        f"{cells_per_block} x {cells_per_block} cells, {n_workers} threads"
    )

    def process_block(block: _Block) -> _BlockResult:
        """Computes the subgrid tables of one block, with NaN comparisons allowed.

        hydromt-sfincs' resampling and kernel compare NaN values (pixels without
        data); numpy's error state is per thread, so it is set in each worker.

        Args:
            block: The block of cells.

        Returns:
            The tables and by-products of the block.
        """
        with np.errstate(invalid="ignore"):
            return compute_block(block)

    def compute_block(block: _Block) -> _BlockResult:
        """Computes the subgrid tables of one block.

        Args:
            block: The block of cells.

        Returns:
            The tables and by-products of the block.

        Raises:
            ValueError: If the block has pixels without elevation or roughness.
        """
        step_start: float = time.perf_counter()
        step_seconds: dict[str, float] = {}

        def end_step(name: str) -> None:
            """Records the thread time of a step.

            Args:
                name: Name of the step.
            """
            nonlocal step_start
            now: float = time.perf_counter()
            step_seconds[name] = now - step_start
            step_start = now

        block_rows: int = block.row_end - block.row_start
        block_cols: int = block.col_end - block.col_start
        # one extra cell in both directions for the u/v points on the block edge
        shape: tuple[int, int] = ((block_rows + 1) * refi, (block_cols + 1) * refi)
        transform: Affine = subgrid_transform * Affine.translation(
            block.col_start * refi, block.row_start * refi
        )
        elevation: np.ndarray = _merge_rasters(
            elevation_rasters, transform, shape, grid_crs
        )
        if np.isnan(elevation).any():
            elevation = _fill_nodata(elevation, transform)
        elevation = np.ascontiguousarray(elevation, dtype=np.float32)

        # Manning's n in float64, as hydromt-sfincs passes it to the kernel; pixels
        # without data get the default for land (elevation >= 0) or sea
        default_manning: np.ndarray = np.where(elevation >= 0.0, 0.04, 0.02)
        if roughness_rasters:
            manning: np.ndarray = _merge_rasters(
                roughness_rasters, transform, shape, grid_crs
            )
            manning = np.where(np.isnan(manning), default_manning, manning)
        else:
            manning = default_manning

        end_step("resampling")

        river_cells: np.ndarray = np.empty((0, 2), dtype=np.intp)
        river_water_level: np.ndarray = np.empty(0, dtype=np.float32)
        if river_burner is not None:
            rows, cols, unburned = river_burner.burn(elevation, manning, transform)
            # initial water level: lowest unburned elevation of the burned pixels of
            # each cell (outside the overlap)
            inside: np.ndarray = (rows < block_rows * refi) & (cols < block_cols * refi)
            if inside.any():
                cell_index: np.ndarray = (rows[inside] // refi) * block_cols + (
                    cols[inside] // refi
                )
                level: np.ndarray = np.full(block_rows * block_cols, np.inf)
                np.minimum.at(level, cell_index, unburned[inside])
                has_river: np.ndarray = np.isfinite(level)
                cells: np.ndarray = np.flatnonzero(has_river)
                river_cells = np.stack(
                    [cells // block_cols, cells % block_cols], axis=1
                )
                river_water_level = level[has_river].astype(np.float32)

        end_step("rivers")

        if not (np.isfinite(elevation).all() and np.isfinite(manning).all()):
            raise ValueError(
                "Subgrid block without elevation or roughness data at rows "
                f"{block.row_start}-{block.row_end}, columns "
                f"{block.col_start}-{block.col_end}"
            )
        block_mask: np.ndarray = np.ascontiguousarray(
            mask[block.row_start : block.row_end, block.col_start : block.col_end]
        )
        y: np.ndarray = transform.f + (np.arange(shape[0]) + 0.5) * transform.e
        tables = _process_block_tables(
            block_mask,
            elevation,
            manning,
            dxp,
            dyp,
            refi,
            nr_levels,
            y,
            is_geographic,
        )
        end_step("tables")

        elevation_out: np.ndarray | None = None
        if write_dep_tif:
            elevation_out = elevation[: block_rows * refi, : block_cols * refi]
            pixel_mask: np.ndarray = np.repeat(
                np.repeat(block_mask > 0, refi, axis=0), refi, axis=1
            )
            elevation_out = np.where(pixel_mask, elevation_out, np.float32(np.nan))
        end_step("other")
        return _BlockResult(
            block=block,
            tables=tables,
            elevation=elevation_out,
            river_cells=river_cells,
            river_water_level=river_water_level,
            step_seconds=step_seconds,
        )

    def store(result: _BlockResult) -> None:
        """Copies the tables of one block into the compact tables.

        Args:
            result: The tables and by-products of the block.
        """
        block: _Block = result.block
        rows = slice(block.row_start, block.row_end)
        cols = slice(block.col_start, block.col_end)
        tables: dict[str, np.ndarray] = dict(
            zip(_TABLE_NAMES, result.tables, strict=True)
        )

        cell_index: np.ndarray = index_nm_grid[rows, cols]
        is_cell: np.ndarray = cell_index >= 0
        for name in ("z_zmin", "z_zmax", "z_volmax"):
            z_tables[name][cell_index[is_cell]] = tables[name][is_cell]
        z_tables["z_level"][:, cell_index[is_cell]] = tables["z_level"][:, is_cell]

        for direction, uv_index_grid in (("u", index_mu1_grid), ("v", index_nu1_grid)):
            uv_index: np.ndarray = uv_index_grid[rows, cols]
            is_uv: np.ndarray = uv_index >= 0
            for name in ("zmin", "zmax", "ffit", "navg"):
                uv_tables[name][uv_index[is_uv]] = tables[f"{direction}_{name}"][is_uv]
            for name in ("havg", "nrep", "pwet"):
                uv_tables[name][:, uv_index[is_uv]] = tables[f"{direction}_{name}"][
                    :, is_uv
                ]

        if result.river_water_level.size:
            river_rows: np.ndarray = result.river_cells[:, 0] + block.row_start
            river_cols: np.ndarray = result.river_cells[:, 1] + block.col_start
            is_active: np.ndarray = mask[river_rows, river_cols] == 1
            river_initial_water_level[river_rows[is_active], river_cols[is_active]] = (
                result.river_water_level[is_active]
            )

    root: Path = Path(sfincs_model.root.path)
    dep_tif = None
    if write_dep_tif:
        (root / "subgrid").mkdir(parents=True, exist_ok=True)
        dep_tif = rasterio.open(
            root / "subgrid" / "dep_subgrid.tif",
            "w",
            driver="GTiff",
            width=n_cols * refi,
            height=n_rows * refi,
            count=1,
            dtype=np.float32,
            crs=crs,
            transform=subgrid_transform,
            tiled=True,
            blockxsize=_TIFF_TILE_SIZE,
            blockysize=_TIFF_TILE_SIZE,
            compress="deflate",
            predictor=2,
            nodata=np.nan,
            BIGTIFF="YES",
            NUM_THREADS="ALL_CPUS",
        )

    thread_seconds: dict[str, float] = {}
    main_seconds: dict[str, float] = {"storing tables": 0.0, "writing GeoTIFF": 0.0}
    blocks_start: float = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            pending: set[Future[_BlockResult]] = set()
            remaining = iter(blocks)
            n_done: int = 0
            next_report: int = 1
            while True:
                # keep a bounded number of blocks in flight, to bound memory
                while len(pending) < 2 * n_workers:
                    block = next(remaining, None)
                    if block is None:
                        break
                    pending.add(executor.submit(process_block, block))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result: _BlockResult = future.result()
                    for name, seconds in result.step_seconds.items():
                        thread_seconds[name] = thread_seconds.get(name, 0.0) + seconds
                    main_start: float = time.perf_counter()
                    store(result)
                    main_seconds["storing tables"] += time.perf_counter() - main_start
                    main_start = time.perf_counter()
                    if dep_tif is not None and result.elevation is not None:
                        dep_tif.write(
                            result.elevation,
                            1,
                            window=Window(
                                result.block.col_start * refi,
                                result.block.row_start * refi,
                                result.elevation.shape[1],
                                result.elevation.shape[0],
                            ),
                        )
                    main_seconds["writing GeoTIFF"] += time.perf_counter() - main_start
                    n_done += 1
                    if n_done >= next_report or n_done == len(blocks):
                        logger.info(
                            f"Subgrid blocks done: {n_done}/{len(blocks)} "
                            f"({time.perf_counter() - start_time:.0f} s)"
                        )
                        next_report = n_done + max(1, len(blocks) // 20)
    finally:
        if dep_tif is not None:
            dep_tif.close()
    blocks_seconds: float = time.perf_counter() - blocks_start
    logger.info(
        f"Subgrid blocks took {blocks_seconds:.0f} s on {n_workers} threads; thread "
        "time per step: "
        + ", ".join(
            f"{name} {seconds:.0f} s" for name, seconds in thread_seconds.items()
        )
        + f" (parallel efficiency {sum(thread_seconds.values()) / max(blocks_seconds * n_workers, 1e-9):.0%}); "
        "main thread: "
        + ", ".join(f"{name} {seconds:.0f} s" for name, seconds in main_seconds.items())
    )

    # write the tables in the netcdf format of SFINCS (as hydromt-sfincs does)
    tables_ds: xr.Dataset = xr.Dataset(attrs={"_FillValue": np.nan})
    for name in ("z_zmin", "z_zmax", "z_volmax"):
        tables_ds[name] = xr.DataArray(z_tables[name], dims=("np",))
    tables_ds["z_level"] = xr.DataArray(z_tables["z_level"], dims=("levels", "np"))
    for name in ("zmin", "zmax", "ffit", "navg"):
        tables_ds[f"uv_{name}"] = xr.DataArray(uv_tables[name], dims=("npuv",))
    for name in ("havg", "nrep", "pwet"):
        tables_ds[f"uv_{name}"] = xr.DataArray(uv_tables[name], dims=("levels", "npuv"))
    tables_ds = tables_ds.transpose("npuv", "np", "levels")
    tables_ds.to_netcdf(root / "sfincs_subgrid.nc")
    sfincs_model.config.update(sbgfile="sfincs_subgrid.nc")
    logger.info(
        f"Subgrid tables written ({time.perf_counter() - start_time:.0f} s in total)"
    )

    bed_elevation: np.ndarray = np.full(index_nm.size, np.nan, dtype=np.float32)
    is_active: np.ndarray = index_nm >= 0
    bed_elevation[is_active] = z_tables["z_zmin"][index_nm[is_active]]
    return SubgridResult(
        bed_elevation=xr.DataArray(
            _column_major_to_grid(bed_elevation, mask.shape),
            coords=mask_da.coords,
            dims=mask_da.dims,
            name="z_zmin",
        ),
        river_initial_water_level=xr.DataArray(
            river_initial_water_level, coords=mask_da.coords, dims=mask_da.dims
        ),
    )
