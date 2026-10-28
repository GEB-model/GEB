"""Workflow to fill the burned river channels of a SFINCS model with water before the run."""

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from hydromt_sfincs.utils import (
    read_binary_map,
    read_binary_map_index,
    write_binary_map,
)
from pyproj import Transformer
from shapely import STRtree, line_locate_point, points

from geb.workflows.methods import get_utm_zone

# A subgrid pixel is burned river when the burn lowered the DEM by more than this (m)
BURN_MIN_M: float = 0.1


def read_grid(root: Path) -> dict[str, Any]:
    """Read the SFINCS grid of the root model.

    Args:
        root: Path of the SFINCS root model (holds sfincs.inp, sfincs.ind and sfincs.msk).

    Returns:
        Dictionary with mmax, nmax, y0, dx, the active-cell index ind and the mask msk (rows south to north).
    """
    with open(root / "sfincs.inp") as f:
        inp = {
            k.strip(): v.strip()
            for k, v in (line.split("=", 1) for line in f if "=" in line)
        }
    grid: dict[str, Any] = dict(
        mmax=int(inp["mmax"]),
        nmax=int(inp["nmax"]),
        y0=float(inp["y0"]),
        dx=float(inp["dx"]),
    )
    grid["ind"] = read_binary_map_index(root / "sfincs.ind")
    grid["msk"] = read_binary_map(
        root / "sfincs.msk",
        grid["ind"],
        (grid["nmax"], grid["mmax"]),
        mv=0,
        dtype="u1",
    )
    return grid


def burned_river_cells(
    root: Path,
    dem: xr.DataArray,
    mmax: int,
    nmax: int,
    msk: np.ndarray,
) -> pd.DataFrame:
    """Find the active SFINCS cells that hold burned river pixels.

    Args:
        root: Path of the SFINCS root model (holds subgrid/dep_subgrid.tif).
        dem: The DEM the rivers were burned into.
        mmax: Number of SFINCS columns.
        nmax: Number of SFINCS rows.
        msk: SFINCS mask (rows south to north).

    Returns:
        Table with per cell: n, m (indices), bed (lowest burned pixel), bank (lowest DEM value over the burned
        pixels), npx (number of burned pixels), lon and lat (centroid of the burned pixels).
    """
    rec: dict[str, list[np.ndarray]] = {
        k: [] for k in ("n", "m", "bed", "bank", "npx", "lon", "lat")
    }
    with rasterio.open(root / "subgrid" / "dep_subgrid.tif") as src:
        assert src.transform.e > 0, "expected a south-up subgrid tif"
        assert src.width % mmax == 0 and src.width // mmax == src.height // nmax
        k: int = src.width // mmax  # subgrid pixels per cell side
        px: float = src.transform.a
        for n0 in range(0, nmax, 1000):
            n1: int = min(n0 + 1000, nmax)
            win = rasterio.windows.Window(0, k * n0, src.width, k * (n1 - n0))
            bed = src.read(1, window=win).astype(np.float32)
            ys = src.transform.f + (k * n0 + np.arange(bed.shape[0]) + 0.5) * px
            xs = src.transform.c + (np.arange(bed.shape[1]) + 0.5) * px
            fd = dem.sel(y=ys, x=xs, method="nearest", tolerance=px / 4).values.astype(
                np.float32
            )
            burned = (fd - bed) > BURN_MIN_M
            shape4 = (n1 - n0, k, mmax, k)
            npx = burned.reshape(shape4).sum(axis=(1, 3))
            cell_bed = np.where(burned, bed, np.inf).reshape(shape4).min(axis=(1, 3))
            cell_bank = np.where(burned, fd, np.inf).reshape(shape4).min(axis=(1, 3))
            lon = np.where(burned, xs[None, :], 0.0).reshape(shape4).sum(
                axis=(1, 3)
            ) / np.maximum(npx, 1)
            lat = np.where(burned, ys[:, None], 0.0).reshape(shape4).sum(
                axis=(1, 3)
            ) / np.maximum(npx, 1)
            nn, mm = np.where((npx > 0) & (msk[n0:n1] > 0))
            for key, value in (
                ("n", nn + n0),
                ("m", mm),
                ("bed", cell_bed[nn, mm]),
                ("bank", cell_bank[nn, mm]),
                ("npx", npx[nn, mm]),
                ("lon", lon[nn, mm]),
                ("lat", lat[nn, mm]),
            ):
                rec[key].append(value)
    return pd.DataFrame({key: np.concatenate(value) for key, value in rec.items()})


def subgrid_volume_m3(
    root: Path,
    grid: dict[str, Any],
    zs: np.ndarray,
    n: np.ndarray,
    m: np.ndarray,
) -> np.ndarray:
    """Water volume per cell at a given level, from the subgrid tables as SFINCS computes it at t0.

    Args:
        root: Path of the SFINCS root model (holds sfincs_subgrid.nc).
        grid: Grid dictionary from read_grid.
        zs: Water level per cell.
        n: Row index per cell.
        m: Column index per cell.

    Returns:
        Water volume (m3) per cell.
    """
    np_index = np.full((grid["nmax"], grid["mmax"]), -1, dtype=np.int64)
    np_index.T.flat[grid["ind"]] = np.arange(grid["ind"].size)
    cells = np_index[n, m]
    with xr.open_dataset(root / "sfincs_subgrid.nc") as sbg:
        zmin = sbg["z_zmin"].values[cells].astype(np.float64)
        zmax = sbg["z_zmax"].values[cells].astype(np.float64)
        volmax = sbg["z_volmax"].values[cells].astype(np.float64)
        zlevel = sbg["z_level"].values
        zlevel = (zlevel.T if zlevel.shape[0] != grid["ind"].size else zlevel)[
            cells
        ]  # (cells, levels)
    nlev: int = zlevel.shape[1]
    dx: float = grid["dx"]
    with rasterio.open(root / "subgrid" / "dep_subgrid.tif") as src:
        is_geographic: bool = src.crs.is_geographic
    area: np.ndarray
    if is_geographic:
        lat: np.ndarray = grid["y0"] + (n + 0.5) * dx
        area = (dx * 111320.0 * np.cos(np.deg2rad(lat))) * (dx * 110574.0)
    else:
        area = np.full(zs.shape, dx * dx)
    vol = np.zeros(zs.shape)
    full = zs >= zmax
    vol[full] = volmax[full] + area[full] * (zs[full] - np.maximum(zmax[full], -20.0))
    part = (~full) & (zs > zmin)
    if part.any():
        levels, z = zlevel[part], zs[part]
        ivol = np.clip((levels <= z[:, None]).sum(axis=1), 1, nlev - 1)
        z_lo = levels[np.arange(z.size), ivol - 1]
        z_hi = levels[np.arange(z.size), ivol]
        dzvol = volmax[part] / (nlev - 1)
        vol[part] = (ivol - 1) * dzvol + np.clip(
            (z - z_lo) / np.maximum(z_hi - z_lo, 0.001), 0, 1
        ) * dzvol
    return vol


def channel_depth_level(
    cells: pd.DataFrame,
    burn: gpd.GeoDataFrame,
    rivers_utm: gpd.GeoDataFrame,
    ux: np.ndarray,
    uy: np.ndarray,
) -> np.ndarray:
    """Initial level from stored channel depth, with downstream backwater and a bank cap.

    The stored depth is the channel geometry used when burning the river and already reflects the routing
    model's bankfull discharge calculation. Reusing it keeps the initial fill consistent with that geometry.

    1. Set each target level to the burned bed plus the depth of its nearest burned river segment.
    2. Backwater: segments from the outlets upwards, cells from the downstream end upwards. A level is at least
       the level just downstream (for the last cell of a segment: the nearest filled cell of its downstream
       segment).
    3. Cap at the DEM river surface.

    Adds the columns depth and zs_normal to cells.

    Args:
        cells: Burned river cells with bed, bank, depth and segment.
        burn: Rivers that were burned (rivers_to_burn), in UTM.
        rivers_utm: Full, unclipped river geometries, in UTM.
        ux: Cell x coordinate (UTM).
        uy: Cell y coordinate (UTM).

    Returns:
        Initial water level per cell (-9999 where the cell stays dry).
    """
    groups = cells.groupby("segment").indices
    normal = cells.bed.values + cells.depth.values
    cells["zs_normal"] = normal

    segments = set(groups)
    downstream: dict[int, int | None] = {}
    for segment in segments:
        ds = int(burn.loc[segment, "downstream_ID"])
        downstream[segment] = ds if ds in segments else None
    upstream: dict[int | None, list[int]] = {}
    for segment, ds in downstream.items():
        upstream.setdefault(ds, []).append(segment)
    order: list[int] = []
    todo: list[int] = list(upstream.get(None, []))
    while todo:
        segment = todo.pop()
        order.append(segment)
        todo.extend(upstream.get(segment, []))
    assert len(order) == len(segments), (
        "river network of the burned segments is not a tree"
    )

    bank = cells.bank.values
    zs = np.full(len(cells), -9999.0)
    for segment in order:
        # Use the full reach geometry: clipping can turn a reach into a MultiLineString.
        line = rivers_utm.loc[segment].geometry  # digitised upstream -> downstream
        distance_m = line_locate_point(
            line, points(ux[groups[segment]], uy[groups[segment]])
        )
        ii = groups[segment][np.argsort(-distance_m)]  # downstream end first
        running = -np.inf
        if downstream[segment] is not None:
            x_end, y_end = line.coords[-1][:2]
            jj = groups[downstream[segment]]
            jj = jj[zs[jj] > -9999.0]
            if jj.size:
                running = zs[jj[np.argmin(np.hypot(ux[jj] - x_end, uy[jj] - y_end))]]
        for i in ii:
            if np.isfinite(normal[i]):
                zs[i] = min(max(normal[i], running), bank[i])
                running = zs[i]
    return zs


def q2_initial_water_level(
    root: Path,
    dem: xr.DataArray,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Initial water level per SFINCS cell from the burned rivers' stored channel depth.

    Args:
        root: Path of the SFINCS root model (holds the grid, subgrid, rivers_to_burn and rivers files).
        dem: The DEM the rivers were burned into.

    Returns:
        The level per cell (rows south to north, -9999 = dry), the per cell table and the grid dictionary.
    """
    grid = read_grid(root)
    cells = burned_river_cells(root, dem, grid["mmax"], grid["nmax"], grid["msk"])

    burn = gpd.read_parquet(root / "rivers_to_burn.geoparquet")
    rivers = gpd.read_parquet(root / "rivers.geoparquet")
    centre = burn.to_crs(4326).union_all().centroid
    utm = get_utm_zone(centre.y, centre.x)
    burn = burn.to_crs(utm)
    to_utm = Transformer.from_crs(dem.rio.crs, utm, always_xy=True)
    ux, uy = to_utm.transform(cells.lon.values, cells.lat.values)

    # each burned cell belongs to the nearest burned river segment
    pos = STRtree(burn.geometry.values).query_nearest(points(ux, uy), all_matches=False)
    pos = np.asarray(pos)[1]
    cells["segment"] = burn.index.values[pos]
    cells["depth"] = burn["depth"].values[pos]

    zs = channel_depth_level(cells, burn, rivers.to_crs(utm), ux, uy)
    cells["capped"] = (
        cells.zs_normal.values > cells.bank.values
    )  # Stored channel depth reaches above the DEM river surface.
    cells["zs"] = zs

    level = np.full((grid["nmax"], grid["mmax"]), -9999.0, dtype=np.float32)
    level[cells.n.values, cells.m.values] = zs
    return level, cells, grid


def set_q2_initial_water_level(
    simulation: Any,
    dem: xr.DataArray,
) -> str:
    """Write the Q2 initial water level of a SFINCS simulation and tell SFINCS to start from it.

    Args:
        simulation: The SFINCSSimulation whose forcing has been set.
        dem: The DEM the rivers were burned into.

    Returns:
        A one-line summary for the log.
    """
    root: Path = simulation.root_path
    level, cells, grid = q2_initial_water_level(root, dem)

    write_binary_map(simulation.path / "sfincs.ini", level, grid["msk"], dtype="f4")
    simulation.sfincs_model.config.update(inifile="sfincs.ini")
    simulation.sfincs_model.config.write()

    filled = cells.zs.values > -9999.0
    volume = subgrid_volume_m3(
        root, grid, cells.zs.values.astype(np.float64), cells.n.values, cells.m.values
    )
    return (
        f"Channel-depth initial water level: {int(filled.sum())} of {len(cells)} burned river cells filled, "
        f"{int(cells.capped.sum())} capped at the river surface, {volume.sum() / 1e6:.0f} Mm3 at t0"
    )
