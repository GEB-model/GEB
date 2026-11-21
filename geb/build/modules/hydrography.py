"""Build methods for the hydrography for GEB."""

import os
import warnings
from datetime import timedelta
from pathlib import Path
from types import CodeType
from typing import Any, Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyflwdir
import rasterio.features
import shapely
import xarray as xr
from networkx.classes.digraph import DiGraph
from numba import njit
from pyflwdir import FlwdirRaster
from rasterio.enums import MergeAlg
from scipy import interpolate
from scipy.ndimage import value_indices
from scipy.spatial import KDTree
from shapely.geometry import LineString, shape

from geb.build.data_catalog import DataCatalog
from geb.build.data_catalog.gtsm import gtsm_filters
from geb.build.methods import build_method
from geb.build.workflows.command_areas import derive_command_areas_from_routing
from geb.build.workflows.hydrography import (
    add_stream_orders,
    get_river_graph,
)
from geb.build.workflows.river_snapping import (
    SnappingResults,
    snap_point_to_river_network,
)
from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayInt32,
    ArrayInt64,
    TwoDArrayBool,
    TwoDArrayFloat32,
    TwoDArrayFloat64,
    TwoDArrayInt32,
    TwoDArrayInt64,
    TwoDArrayUint8,
)
from geb.hydrology.routing.geometry import (
    FLOODPLAIN_PROFILE_DEPTHS,
    NUM_FLOODPLAIN_LEVELS,
)
from geb.hydrology.waterbodies import LAKE, LAKE_CONTROL, RESERVOIR
from geb.workflows.raster import (
    calculate_height_m,
    calculate_width_m,
    full_like,
    rasterize_like,
    snap_to_grid,
)
from geb.workflows.river_network import propagate_downstream
from geb.workflows.river_profiles import (
    estimate_river_surface_profiles,
    line_vertex_distances_m,
)

from .base import BuildModelBase

D8_NORTHWEST = 32
D8_NORTH = 64
D8_NORTHEAST = 128
D8_EAST = 1
D8_SOUTHEAST = 2
D8_SOUTH = 4
D8_SOUTHWEST = 8
D8_WEST = 16
D8_RIVER_MOUTH = 0
D8_PIT = 255
D8_NODATA = 247


def calculate_stream_length(
    d8_ldd: xr.DataArray,
    upstream_area_m2: xr.DataArray,
    threshold_m2: float = 1_000_000,
) -> xr.DataArray:
    """Calculate the stream length for a grid.

    Args:
        d8_ldd: The high resolution flow direction raster (D8).
        upstream_area_m2: The high resolution upstream area raster (m2).
        threshold_m2: The threshold for defining a stream based on upstream area (m2).

    Returns:
        A DataArray containing the stream length for the high resolution grid (meters).
    """
    is_stream: xr.DataArray = (
        upstream_area_m2 > threshold_m2
    )  # threshold for streams is 1 km^2 upstream area

    cell_width_m = calculate_width_m(
        is_stream.rio.transform(),
        height=is_stream.shape[0],
        width=is_stream.shape[1],
    )
    cell_height_m = calculate_height_m(
        is_stream.rio.transform(),
        height=is_stream.shape[0],
        width=is_stream.shape[1],
    )

    stream_length = xr.DataArray(
        np.full_like(is_stream, np.nan, dtype=np.float32),
        coords=is_stream.coords,
        dims=is_stream.dims,
    )

    # pit or river mouth -> no channel, so set stream length to 0
    stream_length = xr.where(
        (d8_ldd != D8_RIVER_MOUTH) & (d8_ldd != D8_PIT), stream_length, 0
    )
    # vertical -> set stream length to cell height
    stream_length = xr.where(
        ~((d8_ldd == D8_NORTH) | (d8_ldd == D8_SOUTH)),
        stream_length,
        cell_height_m,
    )
    # horizontal -> set stream length to cell width
    stream_length = xr.where(
        ~((d8_ldd == D8_WEST) | (d8_ldd == D8_EAST)),
        stream_length,
        cell_width_m,
    )
    # diagonal -> set stream length to cell diagonal
    stream_length = xr.where(
        ~(
            (d8_ldd == D8_NORTHEAST)
            | (d8_ldd == D8_SOUTHEAST)
            | (d8_ldd == D8_SOUTHWEST)
            | (d8_ldd == D8_NORTHWEST)
        ),
        stream_length,
        np.sqrt(cell_width_m**2 + cell_height_m**2),
    )
    stream_length = xr.where(is_stream | np.isnan(stream_length), stream_length, 0)

    return stream_length


def _extract_1sec_subgrid(
    elevation_1sec: xr.DataArray,
    elevation_high_res: xr.DataArray,
) -> npt.NDArray[np.float32]:
    """Extract 1 arc-second subgrid aligned 3:1 to the 3 arc-second elevation grid.

    Args:
        elevation_1sec: High-resolution DEM at ~1 arc-second resolution.
        elevation_high_res: Hydrography elevation grid at ~3 arc-second resolution.

    Returns:
        2D float32 array of shape (3 * H_hd, 3 * W_hd) containing 1 arc-second elevations.

    Raises:
        ValueError: If elevation_1sec or elevation_high_res is missing coordinates,
            has degenerate coordinate dimensions, or does not overlap spatially.
    """
    if "x" not in elevation_1sec.coords or "y" not in elevation_1sec.coords:
        raise ValueError(
            "elevation_1sec is missing 'x' or 'y' coordinates required for subgrid alignment."
        )
    if "x" not in elevation_high_res.coords or "y" not in elevation_high_res.coords:
        raise ValueError(
            "elevation_high_res is missing 'x' or 'y' coordinates required for subgrid alignment."
        )

    x3: npt.NDArray[np.floating] = elevation_high_res.x.values
    y3: npt.NDArray[np.floating] = elevation_high_res.y.values
    x1: npt.NDArray[np.floating] = elevation_1sec.x.values
    y1: npt.NDArray[np.floating] = elevation_1sec.y.values

    if x1.size <= 1 or y1.size <= 1:
        raise ValueError(
            f"elevation_1sec coordinate arrays must have length > 1, got x={x1.size}, y={y1.size}."
        )

    dx1: float = float(x1[1] - x1[0])
    dy1: float = float(y1[1] - y1[0])

    if dx1 == 0.0 or dy1 == 0.0:
        raise ValueError("elevation_1sec coordinate spacing must be non-zero.")

    H_hd: int = int(elevation_high_res.shape[0])
    W_hd: int = int(elevation_high_res.shape[1])
    target_shape: tuple[int, int] = (3 * H_hd, 3 * W_hd)

    c_center: npt.NDArray[np.int_] = np.round((x3 - x1[0]) / dx1).astype(int)
    r_center: npt.NDArray[np.int_] = np.round((y3 - y1[0]) / dy1).astype(int)

    c_start: int = int(c_center[0] - 1)
    c_end: int = int(c_center[-1] + 2)
    r_start: int = int(r_center[0] - 1)
    r_end: int = int(r_center[-1] + 2)

    src_r0: int = max(0, r_start)
    src_r1: int = min(int(elevation_1sec.shape[0]), r_end)
    src_c0: int = max(0, c_start)
    src_c1: int = min(int(elevation_1sec.shape[1]), c_end)

    if src_r1 <= src_r0 or src_c1 <= src_c0:
        raise ValueError(
            "elevation_1sec does not overlap with elevation_high_res. "
            f"elevation_1sec extent: x=[{float(x1[0])}, {float(x1[-1])}], y=[{float(y1[0])}, {float(y1[-1])}]; "
            f"elevation_high_res extent: x=[{float(x3[0])}, {float(x3[-1])}], y=[{float(y3[0])}, {float(y3[-1])}]."
        )

    dst_r0: int = src_r0 - r_start
    dst_r1: int = src_r1 - r_start
    dst_c0: int = src_c0 - c_start
    dst_c1: int = src_c1 - c_start

    fathomdem_sub: npt.NDArray[np.float32] = np.full(
        target_shape, np.nan, dtype=np.float32
    )
    fathomdem_sub[dst_r0:dst_r1, dst_c0:dst_c1] = elevation_1sec.values[
        src_r0:src_r1, src_c0:src_c1
    ]
    return fathomdem_sub


# Upper bound of a floodplain profile width (m), against division by tiny lengths.
MAX_FLOODPLAIN_WIDTH_M: float = 100_000.0


def _downstream_of(
    flow_raster_high_res: FlwdirRaster, is_start: npt.NDArray[np.float32]
) -> npt.NDArray[np.bool_]:
    """Pixels on or downstream of any start pixel.

    Accumulating an indicator of the start pixels downstream gives a value > 0 exactly
    on the flow paths that leave them.

    Args:
        flow_raster_high_res: High-resolution flow direction raster.
        is_start: 1 at the start pixels, 0 elsewhere.

    Returns:
        True on the flow paths downstream of the start pixels (inclusive).
    """
    return flow_raster_high_res.accuflux(is_start, nodata=np.float32(0.0)) > 0


def _flooded_area_per_outflow(
    flow_raster_high_res: FlwdirRaster,
    drain_high_res: npt.NDArray[np.bool_],
    elev_3sec: npt.NDArray[np.floating],
    elevation_1sec_subgrid: npt.NDArray[np.float32],
    outflows_hd: npt.NDArray[np.int64],
    subpixel_area_m2: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Area below each floodplain profile depth in the land draining to each outflow pixel.

    1. HAND at 3 arc-seconds above the drainage pixels gives the elevation of the
       drainage that each 3 arc-second pixel drains to.
    2. Each 1 arc-second subpixel gets its height above that drainage elevation.
    3. Subpixels are assigned to the outflow pixel they drain to first (its subbasin);
       subpixels that drain to none of the outflow pixels are skipped.
    4. For each depth in FLOODPLAIN_PROFILE_DEPTHS, the subpixels at or below that
       height are counted per outflow pixel and converted to area.

    Args:
        flow_raster_high_res: High-resolution (3 arc-second) flow direction raster.
        drain_high_res: Drainage pixels on the high-resolution grid.
        elev_3sec: Elevation of each 3 arc-second pixel (m).
        elevation_1sec_subgrid: 1 arc-second elevation, aligned 3:1 with elev_3sec (m).
        outflows_hd: Flat index of the outflow pixels on the high-resolution grid.
        subpixel_area_m2: Area of one 1 arc-second subpixel for each outflow pixel (m2).

    Returns:
        Array of shape (number of depths, number of outflow pixels) with the flooded
        area (m2).
    """
    hand_3sec = np.maximum(
        flow_raster_high_res.hand(drain=drain_high_res, elevtn=elev_3sec),
        np.float32(0.0),
    )
    drain_elev_1sec = np.repeat(np.repeat(elev_3sec - hand_3sec, 3, axis=0), 3, axis=1)
    hand_1sec = np.maximum(
        elevation_1sec_subgrid - drain_elev_1sec, np.float32(0.0)
    ).ravel()

    n_outflows: int = outflows_hd.size
    subbasins_hd = flow_raster_high_res.basins(
        idxs=outflows_hd, ids=np.arange(1, n_outflows + 1, dtype=np.uint32)
    )
    subbasins_1sec = np.repeat(np.repeat(subbasins_hd, 3, axis=0), 3, axis=1).ravel()
    valid = (subbasins_1sec > 0) & ~np.isnan(hand_1sec)

    areas = np.zeros((NUM_FLOODPLAIN_LEVELS, n_outflows), dtype=np.float32)
    for k, depth in enumerate(FLOODPLAIN_PROFILE_DEPTHS):
        counts = np.bincount(
            subbasins_1sec[valid & (hand_1sec <= depth)], minlength=n_outflows + 1
        )
        areas[k] = counts[1 : n_outflows + 1].astype(np.float32) * subpixel_area_m2
    return areas


def _side_slope_above_profile(
    widths: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Valley side slope above the deepest profile level.

    The slope of the last profile interval is extended upward: z is half the increase
    in width (one valley side) per metre of depth. Where the width does not increase
    in the last interval, z = 0 (vertical walls above the top width).

    Args:
        widths: Profile widths (m), with the depth levels on the first axis.

    Returns:
        Side slope z (m horizontal per m vertical), one per profile.
    """
    depth_step = np.float32(
        FLOODPLAIN_PROFILE_DEPTHS[-1] - FLOODPLAIN_PROFILE_DEPTHS[-2]
    )
    side_slope = (
        np.maximum((widths[-1] - widths[-2]) * np.float32(0.5), np.float32(0.0))
        / depth_step
    )
    return np.where(np.isfinite(side_slope), side_slope, np.float32(0.0)).astype(
        np.float32
    )


def calculate_dem_floodplain_geometry(
    flow_raster_high_res: FlwdirRaster,
    elevation_high_res: xr.DataArray,
    elevation_1sec: xr.DataArray,
    upstream_area_high_res_m2: xr.DataArray,
    idxs_outflow_low_res: xr.DataArray,
    river_raster_low_res: npt.NDArray[np.int32],
    cell_area_low_res: xr.DataArray,
    river_length_low_res: xr.DataArray,
    ldd_scale_factor: int,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate 10-point floodplain width profile and side-slope ratio for each river reach.

    Uses Height Above Nearest Drainage (HAND) (Nobre et al. 2016) calculated from a 1 arc-second DEM
    (e.g., FathomDEM) mapped to the 3 arc-second hydrologically conditioned flow paths from MERIT Hydro.
    The exact 3:1 integer subgrid alignment samples flooded width across 10 fixed depth levels above bankfull
    ([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0] meters). Flooded subpixels are mapped to river reaches
    via subbasins and aggregated by river segment to produce uniform cross-sectional width profiles.

    Two kinds of cells get a profile, each measured above its own drainage line:

    - River cells (``river_raster_low_res != -1``): HAND is taken above the routed river
      only, i.e. the high-resolution flow path downstream of the outflow pixels of the
      river cells. Side-stream valleys are not drainage here: land along a side stream
      lies higher above the main river and floods only once the river reaches that
      level, so it must not count as floodplain at the side stream's own level. Widths
      are aggregated per river segment.
    - Other cells with a stream (small streams that are not routed rivers): HAND is
      taken above the flow path through the cell's own outflow pixel, over the land
      that drains to that outflow pixel. Steep streams in narrow valleys thus get a
      narrow profile (or none), streams on flat valley floors a wide one. Widths are
      per cell.

    Args:
        flow_raster_high_res: High-resolution flow direction raster (typically 3 arc-seconds).
        elevation_high_res: Hydrography digital elevation model (DEM) (meters).
        elevation_1sec: High-resolution DEM at ~1 arc-second resolution (meters).
        upstream_area_high_res_m2: Upstream contributing area on high-resolution grid (m²).
        idxs_outflow_low_res: Low-resolution array mapping each model grid cell to its high-resolution outflow pixel index.
        river_raster_low_res: Low-resolution 2D array of river IDs (-1 indicates non-river cells).
        cell_area_low_res: Grid cell area at model resolution (m²).
        river_length_low_res: River (or, for cells without a routed river, stream)
            length in each model grid cell (meters).
        ldd_scale_factor: Coarsening scale factor from high-resolution (3 arc-seconds) to model resolution.

    Returns:
        A tuple of (floodplain_side_slope_m_per_m, floodplain_profile_widths_m) where:
            - floodplain_side_slope_m_per_m: Valley side-slope ratio z (m/m horizontal:vertical).
            - floodplain_profile_widths_m: 3D array of shape (10, y, x) containing top widths at the 10 depth levels (meters).
    """
    n_levels: int = NUM_FLOODPLAIN_LEVELS
    profile_coords: dict[str, Any] = {
        "floodplain_depth": np.array(FLOODPLAIN_PROFILE_DEPTHS, dtype=np.float32),
        "y": river_length_low_res.y,
        "x": river_length_low_res.x,
    }
    floodplain_profile_widths_m = xr.DataArray(
        np.full(
            (n_levels, river_length_low_res.shape[0], river_length_low_res.shape[1]),
            np.nan,
            dtype=np.float32,
        ),
        coords=profile_coords,
        dims=("floodplain_depth", "y", "x"),
    )
    floodplain_profile_widths_m.attrs["_FillValue"] = np.nan
    floodplain_side_slope_m_per_m = xr.full_like(
        river_length_low_res, fill_value=np.nan, dtype=np.float32
    )

    river_mask_flat = river_raster_low_res.ravel() != -1
    river_cell_indices = np.where(river_mask_flat)[0]

    # Cells with a stream but no routed river (see docstring).
    stream_length_flat = river_length_low_res.values.ravel().astype(np.float32)
    all_outflows_hd = idxs_outflow_low_res.values.ravel()
    is_stream_cell = (
        ~river_mask_flat
        & np.isfinite(stream_length_flat)
        & (stream_length_flat > 0)
        & (all_outflows_hd >= 0)
    )
    stream_cell_indices = np.where(is_stream_cell)[0]

    if river_cell_indices.size == 0 and stream_cell_indices.size == 0:
        return (
            floodplain_side_slope_m_per_m,
            floodplain_profile_widths_m,
        )

    H_hd: int = int(elevation_high_res.shape[0])
    W_hd: int = int(elevation_high_res.shape[1])
    fathomdem_sub = _extract_1sec_subgrid(elevation_1sec, elevation_high_res)

    # Elevation of each 3 arc-second pixel for the HAND calculation: the lowest of its
    # 3x3 1 arc-second subpixels (the hydrography DEM where all subpixels are missing).
    blocks = (
        fathomdem_sub.reshape(H_hd, 3, W_hd, 3)
        .transpose(0, 2, 1, 3)
        .reshape(H_hd, W_hd, 9)
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="All-NaN slice"
        )
        min_subpixels = np.nanmin(blocks, axis=2)
    elev_3sec = np.where(
        np.isnan(min_subpixels), elevation_high_res.values, min_subpixels
    )

    # Area of one 1 arc-second subpixel, per model grid cell (m2).
    subpixel_area_flat = (
        cell_area_low_res.values.ravel().astype(np.float32)
        / np.float32(ldd_scale_factor**2)
        / np.float32(9.0)
    )

    profile_widths_data = np.full(
        (n_levels, river_raster_low_res.size), np.nan, dtype=np.float32
    )
    floodplain_side_slope_m_per_m_data = np.full(
        river_raster_low_res.size, np.nan, dtype=np.float32
    )

    # Rivers: the drainage is the routed river at high resolution, i.e. every pixel on
    # or downstream of the outflow pixel of a river cell. Widths are aggregated per
    # river segment, so all cells of a segment share one profile.
    river_outflows_hd = all_outflows_hd[river_cell_indices].astype(np.int64)
    is_river_outflow = np.zeros(upstream_area_high_res_m2.shape, dtype=np.float32)
    is_river_outflow.ravel()[river_outflows_hd] = np.float32(1.0)
    if river_cell_indices.size > 0:
        reach_profile_areas = _flooded_area_per_outflow(
            flow_raster_high_res=flow_raster_high_res,
            drain_high_res=_downstream_of(flow_raster_high_res, is_river_outflow),
            elev_3sec=elev_3sec,
            elevation_1sec_subgrid=fathomdem_sub,
            outflows_hd=river_outflows_hd,
            subpixel_area_m2=subpixel_area_flat[river_cell_indices],
        )

        river_raster_flat = river_raster_low_res.ravel()
        river_length_flat = river_length_low_res.values.ravel().astype(np.float32)
        for segment_id in np.unique(river_raster_flat[river_mask_flat]):
            segment_mask = river_raster_flat == segment_id
            total_seg_len = np.sum(river_length_flat[segment_mask])
            seg_reach_mask = river_raster_flat[river_cell_indices] == segment_id
            if total_seg_len > 0:
                seg_widths = np.clip(
                    reach_profile_areas[:, seg_reach_mask].sum(axis=1) / total_seg_len,
                    np.float32(0.0),
                    np.float32(MAX_FLOODPLAIN_WIDTH_M),
                )
            else:
                seg_widths = np.zeros(n_levels, dtype=np.float32)
            # inundated width must not decrease with depth
            seg_widths = np.maximum.accumulate(seg_widths)
            profile_widths_data[:, segment_mask] = seg_widths[:, None]
            floodplain_side_slope_m_per_m_data[segment_mask] = (
                _side_slope_above_profile(seg_widths)
            )

    # Small streams: the drainage also includes the paths downstream of the outflow
    # pixels of the stream cells, so each stream is its own drainage line. Each stream
    # cell gets only the land between its own outflow pixel and the next outflow pixel
    # upstream; land that drains straight into a river cell is left to the river.
    # Widths are per cell.
    if stream_cell_indices.size > 0:
        stream_outflows_hd = all_outflows_hd[stream_cell_indices].astype(np.int64)
        is_cell_outflow = is_river_outflow.copy()
        is_cell_outflow.ravel()[stream_outflows_hd] = np.float32(1.0)
        stream_profile_areas = _flooded_area_per_outflow(
            flow_raster_high_res=flow_raster_high_res,
            drain_high_res=_downstream_of(flow_raster_high_res, is_cell_outflow),
            elev_3sec=elev_3sec,
            elevation_1sec_subgrid=fathomdem_sub,
            outflows_hd=stream_outflows_hd,
            subpixel_area_m2=subpixel_area_flat[stream_cell_indices],
        )
        stream_widths = np.clip(
            stream_profile_areas / stream_length_flat[stream_cell_indices],
            np.float32(0.0),
            np.float32(MAX_FLOODPLAIN_WIDTH_M),
        )
        # inundated width must not decrease with depth
        stream_widths = np.maximum.accumulate(stream_widths, axis=0)
        profile_widths_data[:, stream_cell_indices] = stream_widths
        floodplain_side_slope_m_per_m_data[stream_cell_indices] = (
            _side_slope_above_profile(stream_widths)
        )

    floodplain_profile_widths_m.data = profile_widths_data.reshape(
        (n_levels, river_raster_low_res.shape[0], river_raster_low_res.shape[1])
    )
    floodplain_side_slope_m_per_m.data = floodplain_side_slope_m_per_m_data.reshape(
        river_raster_low_res.shape
    )

    return (
        floodplain_side_slope_m_per_m,
        floodplain_profile_widths_m,
    )


def get_all_upstream_subbasin_ids(
    river_graph: nx.DiGraph, subbasin_ids: list[int]
) -> set[int]:
    """Get all upstream subbasin IDs for the given subbasin IDs.

    Args:
        river_graph: The river graph to use for determining upstream subbasins.
        subbasin_ids: The subbasin IDs to get the upstream subbasins for.

    Returns:
        A set of all upstream subbasin IDs.
    """
    ancenstors = set()
    for subbasin_id in subbasin_ids:
        ancenstors |= nx.ancestors(river_graph, subbasin_id)
    return ancenstors


def get_all_downstream_subbasin_ids(
    river_graph: nx.DiGraph, subbasin_ids: list[int]
) -> set[int]:
    """Get all downstream subbasin IDs for the given subbasin IDs.

    Args:
        river_graph: The river graph to use for determining downstream subbasins.
        subbasin_ids: The subbasin IDs to get the downstream subbasins for.

    Returns:
        A set of all downstream subbasin IDs.
    """
    descendants = set()
    for subbasin_id in subbasin_ids:
        descendants |= nx.descendants(river_graph, subbasin_id)
    return descendants


def get_immediate_downstream_subbasins(
    river_graph: nx.DiGraph, sink_subbasin_ids: list[int]
) -> dict[int, list[int]]:
    """Get the immediately downstream subbasins for the given sink subbasin IDs.

    The downstream subbasins are returned as a dictionary mapping each downstream subbasin ID
    to a list of its upstream sink subbasin IDs. Some downstream subbasins may have multiple
    upstream sink subbasins.

    Args:
        river_graph: The river graph to use for determining downstream subbasins.
        sink_subbasin_ids: The subbasin IDs of the sink subbasins (i.e., the outflow subbasins).

    Returns:
        A dictionary mapping each downstream subbasin ID to a list of its upstream sink subbasin IDs.
    """
    downstream_subbasins: dict[int, list[int]] = {}
    for subbasin_id in sink_subbasin_ids:
        downstream_subbasin = list(river_graph.neighbors(subbasin_id))
        if len(downstream_subbasin) == 0:
            pass
        else:
            assert len(downstream_subbasin) == 1, (
                "A subbasin has more than one downstream subbasin"
            )
            downstream_subbasin = downstream_subbasin[0]
            if downstream_subbasin not in downstream_subbasins:
                downstream_subbasins[downstream_subbasin] = []
            downstream_subbasins[downstream_subbasin].append(subbasin_id)

    return downstream_subbasins


def get_rivers_geometry(
    data_catalog: DataCatalog, subbasin_ids: list[int]
) -> gpd.GeoDataFrame:
    """Get the subbasins geometry and other attributes for the given subbasin IDs.

    Args:
        data_catalog: The data catalog to use for accessing the MERIT dataset.
        subbasin_ids: The subbasin IDs to get the geometry for.

    Returns:
        A GeoDataFrame containing the subbasins geometry for the given subbasin IDs.
            The index of the GeoDataFrame is the subbasin ID (COMID).
    """
    rivers = data_catalog.fetch("merit_basins_rivers").read(
        filters=[
            ("COMID", "in", subbasin_ids),
        ],
        columns=[
            "COMID",
            "lengthkm",
            "slope",
            "uparea",
            "maxup",
            "NextDownID",
            "geometry",
        ],
    )
    assert isinstance(rivers, gpd.GeoDataFrame)
    assert len(rivers) == len(subbasin_ids), "Some rivers were not found"

    rivers: gpd.GeoDataFrame = rivers.rename(
        columns={
            "NextDownID": "downstream_ID",
        }
    ).set_index("COMID")
    rivers["uparea_m2"] = rivers["uparea"] * 1e6  # convert from km^2 to m^2
    rivers["is_headwater_catchment"] = rivers["maxup"] == 0
    rivers: gpd.GeoDataFrame = rivers.drop(columns=["uparea"])  # ty:ignore[invalid-assignment]
    rivers.loc[rivers["downstream_ID"] == 0, "downstream_ID"] = -1

    # reverse the river lines to have the downstream direction
    rivers["geometry"] = rivers["geometry"].apply(
        lambda x: LineString(list(x.coords)[::-1])
    )

    return rivers


def extend_rivers_into_pits_and_set_pit_type(
    rivers: gpd.GeoDataFrame,
    flow_raster: FlwdirRaster,
    ldd: TwoDArrayUint8,
) -> gpd.GeoDataFrame:
    """Extend rivers that end just before the ocean into the ocean.

    For some rivers that end up in oceans, the end point is not in the ocean
    but rather stops just before the ocean. Here we extend those rivers to the downstream point
    in the ocean, based on the flow raster downstream index.

    Args:
        rivers: A GeoDataFrame containing the river lines.
        flow_raster: The flow raster to use for determining the downstream point.
        ldd: The flow direction raster to use for determining the downstream point.

    Returns:
        A GeoDataFrame containing the extended river lines.

    Raises:
        ValueError: If the distance between the river end point and the downstream point is too large.
    """
    rivers["is_coastal"] = False

    resolution = abs(flow_raster.transform.a)
    for river_id, river in rivers.iterrows():
        # only select rivers that have no downstream river (i.e., end in ocean)
        if river["downstream_ID"] == -1 and not river["is_further_downstream_outflow"]:
            river_end_point = river.geometry.coords[-1]
            lon, lat = river_end_point

            # retrieve the linear index of the river end point
            cell_index = flow_raster.index(lon, lat)

            # find the linear index of the downstream point
            downstream_index = flow_raster.idxs_ds[cell_index]

            downstream_pit_type = ldd.ravel()[downstream_index]
            if downstream_pit_type == D8_PIT:
                rivers.at[river_id, "is_coastal"] = (
                    False  # is already set to False, but just to be explicit
                )
            elif downstream_pit_type == D8_RIVER_MOUTH:
                rivers.at[river_id, "is_coastal"] = True
            else:
                raise ValueError(
                    f"Did not find a pit downstream of the river end point, but found: {downstream_pit_type}"
                )

            # get the lonlat of the downstream point
            downstream_lon, downstream_lat = flow_raster.xy(downstream_index)

            # check that distance between river end point and downstream point is within one cell
            distance = np.sqrt(
                (downstream_lon - lon) ** 2 + (downstream_lat - lat) ** 2
            )
            if distance > resolution * 1.5:
                raise ValueError(
                    f"River {river_id} end point is too far from downstream point: {distance} m"
                )

            # extend the river line to the downstream point
            rivers.at[river_id, "geometry"] = LineString(
                list(river.geometry.coords) + [(downstream_lon, downstream_lat)]
            )

    return rivers


def create_river_raster_from_river_lines(
    rivers: gpd.GeoDataFrame,
    target: xr.DataArray,
    column: str | None = None,
    index: bool | None = None,
) -> TwoDArrayInt32:
    """Create a river raster from river lines.

    Args:
        rivers: A GeoDataFrame containing the river lines.
        target: The target raster to match the spatial properties.
        column: The column in the GeoDataFrame to use for rasterization values. If None, the index will be used.
        index: If True, the index will be used for rasterization values. If None or False, the column will be used.
               If both column and index are None, the index will be used.

    Returns:
        A 2D numpy array containing the rasterized river lines.

    Raises:
        ValueError: If both column and index are provided.

    """
    # The rivers touch as the confluence (typically 2 upstream, 1 downstream). Here,
    # we want to have the upstream area of the downstream river. Because we use the "replace"
    # merge algorithm, we need to sort the rivers from upstream to downstream,
    # such that the upstream rivers are rasterized first.
    rivers_sorted_from_upstream_to_downstream: gpd.GeoDataFrame = rivers.sort_values(
        by="shreve_stream_order", ascending=True
    )  # ty:ignore[invalid-assignment]

    if column is None and (index is None or index is True):
        values = rivers_sorted_from_upstream_to_downstream.index
    elif column is not None:
        values = rivers_sorted_from_upstream_to_downstream[column]
    else:
        raise ValueError(
            "Either column or index must be provided, or both must be None"
        )

    river_raster = rasterio.features.rasterize(
        zip(rivers_sorted_from_upstream_to_downstream.geometry, values),
        out_shape=target.shape,
        fill=-1,
        dtype=np.int32,
        transform=target.rio.transform(recalc=True),
        all_touched=False,  # because this is a line, Bresenham's line algorithm is used, which is perfect here :-)
        merge_alg=MergeAlg.replace,
    )
    return river_raster


def get_SWORD_translation_IDs_and_lengths(
    data_catalog: DataCatalog, rivers: gpd.GeoDataFrame
) -> tuple[TwoDArrayInt64, TwoDArrayFloat64]:
    """Get the SWORD reach IDs and lengths for each river based on the MERIT basin ID.

    Each river can have multiple SWORD reach IDs, so the output is a 2D array of shape (N, M)
    where N is the number of SWORD reaches per river and M is the number of rivers.

    Args:
        data_catalog: The data catalog to use for accessing the MERIT and SWORD datasets.
        rivers: A GeoDataFrame containing the rivers with their MERIT basin IDs as index.

    Returns:
        A tuple of two 2D numpy arrays:
        - SWORD_reach_IDs: A 2D numpy array of shape (N, M) where N is the number of SWORD reaches per river
          and M is the number of rivers. Each element is the SWORD reach ID for that river.
        - SWORD_reach_lengths: A 2D numpy array of shape (N, M) where N is the number of SWORD reaches per river
            and M is the number of rivers. Each element is the length of the SWORD reach for that river.
    """
    MERIT_Basins_to_SWORD = (
        data_catalog.fetch("merit_sword").read().sel(mb=rivers.index.tolist())
    )
    assert isinstance(MERIT_Basins_to_SWORD, xr.Dataset)

    SWORD_reach_IDs: TwoDArrayInt64 = np.full(
        (40, len(rivers)), dtype=np.int64, fill_value=-1
    )
    SWORD_reach_lengths: TwoDArrayFloat64 = np.full(
        (40, len(rivers)), dtype=np.float64, fill_value=np.nan
    )
    for i in range(1, 41):
        SWORD_reach_IDs[i - 1] = np.where(
            MERIT_Basins_to_SWORD[f"sword_{i}"] != 0,
            MERIT_Basins_to_SWORD[f"sword_{i}"],
            -1,
        )
        SWORD_reach_lengths[i - 1] = MERIT_Basins_to_SWORD[f"part_len_{i}"]

    return SWORD_reach_IDs, SWORD_reach_lengths


def get_SWORD_river_widths(
    data_catalog: DataCatalog, SWORD_reach_IDs: TwoDArrayInt64
) -> TwoDArrayFloat64:
    """Get the river widths from the SWORD dataset based on the SWORD reach IDs.

    Each river can have multiple SWORD reach IDs, so the input is a 2D array of shape (N, M).

    Args:
        data_catalog: The data catalog to use for accessing the SWORD dataset.
        SWORD_reach_IDs: A 2D numpy array of shape (N, M) where N is the number of SWORD reaches per river
                         and M is the number of rivers. Each element is the SWORD reach ID for that river.

    Returns:
        A 2D numpy array of shape (N, M) where N is the number of SWORD reaches per river
        and M is the number of rivers. Each element is the river width for that SWORD reach ID.
        If a SWORD reach ID is -1, the corresponding river width will be NaN.
    """
    unique_SWORD_reach_ids = np.unique(SWORD_reach_IDs[SWORD_reach_IDs != -1])

    SWORD = data_catalog.fetch("sword").read(
        sql=f"""SELECT * FROM sword WHERE reach_id IN ({",".join([str(ID) for ID in unique_SWORD_reach_ids])})"""
    )
    assert isinstance(SWORD, pd.DataFrame)
    SWORD = SWORD.set_index("reach_id")

    assert len(SWORD) == len(unique_SWORD_reach_ids), (
        "Some SWORD reaches were not found, possibly the SWORD and MERIT data version are not correct"
    )

    def lookup_river_width(reach_id: int) -> float:
        """Lookup the river width in SWORD dataset for a given SWORD reach ID.

        Args:
            reach_id: The SWORD reach ID.

        Returns:
            The river width for the given SWORD reach ID. If the reach ID is -1, returns NaN.
        """
        if reach_id == -1:  # when no SWORD reach is found
            return np.nan  # return NaN
        width: np.float64 = SWORD.loc[reach_id, "width"]
        if width <= 0.0:
            width = np.float64(np.nan)

        assert pd.isna(width) or width > 0, (
            f"Invalid river width {width} for SWORD reach ID {reach_id}"
        )

        return width

    SWORD_river_width = np.vectorize(lookup_river_width)(SWORD_reach_IDs)
    return SWORD_river_width


def identify_waterbody_outflows(
    waterbody_id: TwoDArrayInt32,
    upstream_area_n_cells: TwoDArrayInt32,
    elevation_min_m: TwoDArrayFloat32,
) -> TwoDArrayInt32:
    """Identifies the outflow points for each water body.

    Finds the cell with the highest upstream area in each water body as the outflow point.
    If there are multiple cells with the same upstream area, the one with the lowest elevation is chosen.

    Args:
        waterbody_id: 2D array of water body IDs (-1 for non-waterbody).
        upstream_area_n_cells: 2D array of upstream area in number of cells.
        elevation_min_m: 2D array of minimum elevation in meters.

    Returns:
        2D array containing the original waterbody ID at each outflow point and -1 elsewhere.
    """
    valid_wb: TwoDArrayBool = waterbody_id != -1
    if not np.any(valid_wb):
        return np.full_like(waterbody_id, -1, dtype=np.int32)

    max_wb_id: int = int(np.max(waterbody_id))
    upstream_area_within_waterbodies: ArrayInt32 = np.zeros(
        max_wb_id + 2, dtype=np.int32
    )
    upstream_area_within_waterbodies[-1] = -1

    np.maximum.at(
        upstream_area_within_waterbodies,
        waterbody_id[valid_wb],
        upstream_area_n_cells[valid_wb],
    )
    upstream_area_at_cells: TwoDArrayInt32 = np.take(
        upstream_area_within_waterbodies, waterbody_id
    )

    waterbody_outflow_points: TwoDArrayInt32 = np.where(
        valid_wb & (upstream_area_n_cells == upstream_area_at_cells),
        waterbody_id,
        np.int32(-1),
    )

    number_of_outflow_points_per_waterbody = np.unique(
        waterbody_outflow_points, return_counts=True
    )
    duplicate_outflow_points = number_of_outflow_points_per_waterbody[0][
        number_of_outflow_points_per_waterbody[1] > 1
    ]
    duplicate_outflow_points = duplicate_outflow_points[duplicate_outflow_points != -1]

    if duplicate_outflow_points.size > 0:
        for duplicate_outflow_point in duplicate_outflow_points:
            duplicate_coords = np.where(
                waterbody_outflow_points == duplicate_outflow_point
            )
            elevations: ArrayFloat32 = elevation_min_m[duplicate_coords]
            minimum_elevation_idx: int = int(np.argmin(elevations))
            for i in range(len(duplicate_coords[0])):
                if i != minimum_elevation_idx:
                    waterbody_outflow_points[
                        duplicate_coords[0][i], duplicate_coords[1][i]
                    ] = -1

    return waterbody_outflow_points


def river_cell_surface_elevation(
    rivers: gpd.GeoDataFrame,
    profiles: gpd.GeoDataFrame,
    idxs_outflow: npt.NDArray[np.int64],
    high_res_x: npt.NDArray[np.float64],
    high_res_y: npt.NDArray[np.float64],
    grid_shape: tuple[int, int],
) -> npt.NDArray[np.float32]:
    """Water-surface elevation of the routing cells of each river.

    Each routing cell drains through one outflow pixel on the high-resolution river
    path. The cell gets the river profile at that pixel: we find the river-line vertex
    closest to the outflow pixel, take its distance along the line, and interpolate the
    profile at that distance. Using the outflow pixel (rather than, say, the mean over
    the cell) keeps the cells in downstream order along the profile, also where the
    river meanders through a cell more than once. The outflow pixel normally lies on
    the river line; where it does not (rare, where the coarse flow path leaves the
    high-resolution path) the closest vertex is used.

    Args:
        rivers: Rivers represented in the grid, indexed by river ID, with the line
            geometry (EPSG:4326, upstream to downstream) and ``hydrography_xy`` (the
            (x, y) indices of the river's routing cells).
        profiles: River surface profiles (``estimate_river_surface_profiles``).
        idxs_outflow: Flat index of the outflow pixel of each routing cell in the
            high-resolution grid (flattened routing grid).
        high_res_x: x coordinates of the high-resolution grid.
        high_res_y: y coordinates of the high-resolution grid.
        grid_shape: Shape of the routing grid.

    Returns:
        Water-surface elevation per routing cell (m), NaN for cells without river.
    """
    n_cols_high_res: int = high_res_x.size
    surface = np.full(grid_shape[0] * grid_shape[1], np.nan, dtype=np.float32)
    point_groups = profiles.groupby("river_id")
    for river_id, river in rivers.iterrows():
        coords = np.asarray(river.geometry.coords)
        vertex_distance = line_vertex_distances_m(river.geometry, is_geographic=True)
        points = point_groups.get_group(river_id)
        for x, y in river["hydrography_xy"]:
            cell = int(y) * grid_shape[1] + int(x)
            # closest line vertex to the outflow pixel (degrees, x scaled by cos(lat))
            outflow_row, outflow_col = divmod(int(idxs_outflow[cell]), n_cols_high_res)
            offset = np.hypot(
                (coords[:, 0] - high_res_x[outflow_col])
                * np.cos(np.deg2rad(high_res_y[outflow_row])),
                coords[:, 1] - high_res_y[outflow_row],
            )
            surface[cell] = np.interp(
                vertex_distance[int(np.argmin(offset))],
                points["distance_m"].to_numpy(),
                points["water_surface_elevation_m"].to_numpy(),
            )
    return surface.reshape(grid_shape)


@njit(cache=True)
def raise_to_downstream(
    elevation: npt.NDArray[np.float32],
    idxs_ds: npt.NDArray[np.int64],
    idxs_seq: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Raise every cell that lies below its downstream cell to that cell's elevation.

    The result is non-increasing downstream. Cells are visited in idxs_seq order,
    which runs from downstream to upstream (pyflwdir), so a downstream cell is final
    before the cells that drain into it are compared with it.

    Args:
        elevation: Elevation per cell (flat array, m).
        idxs_ds: Flat index of the downstream cell of each cell (itself for pits).
        idxs_seq: Cells ordered from downstream to upstream.

    Returns:
        Raised elevation (a copy).
    """
    raised = elevation.copy()
    for cell in idxs_seq:
        downstream = idxs_ds[cell]
        if downstream != cell and raised[cell] < raised[downstream]:
            raised[cell] = raised[downstream]
    return raised


# A waterbody whose water is replaced within a day does not attenuate floods more than
# the river reach itself; it is represented as river.
MIN_WATERBODY_RESIDENCE_TIME_DAYS: float = 1.0


def is_flow_through_waterbody(
    volume_total_m3: npt.NDArray[np.floating],
    average_discharge_m3_per_s: npt.NDArray[np.floating],
    min_residence_time_days: float = MIN_WATERBODY_RESIDENCE_TIME_DAYS,
) -> npt.NDArray[np.bool_]:
    """Find waterbodies whose residence time is too short to act as a lake or reservoir.

    Residence time = total volume / long-term average discharge. A waterbody that is
    flushed within a day hardly attenuates a flood wave, so it is better represented
    as river. This applies to lakes and reservoirs alike:

    - Lakes: HydroLAKES links every lake to the HydroSHEDS river network and gives it
      the discharge of the river it touches. Gravel pits and side lakes along a large
      river (e.g. the Maasplassen along the Meuse) then get the full river discharge
      and a residence time of minutes to hours, whether the river really flows through
      them or passes next to them.
    - Reservoirs: run-of-river weirs and navigation barrages hold back less than a
      day of flow. They only raise the water level and have no storage to operate,
      so they are intentionally also represented as river.

    Waterbodies without discharge (average_discharge 0) have an infinite (or, with
    zero volume, undefined) residence time and are kept.

    Args:
        volume_total_m3: Total volume of each waterbody (m3).
        average_discharge_m3_per_s: Long-term average discharge through each
            waterbody (m3/s).
        min_residence_time_days: Shortest residence time of a waterbody that is kept.

    Returns:
        True for waterbodies with a residence time below the minimum.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        residence_time_days = volume_total_m3 / (average_discharge_m3_per_s * 86400.0)
    return residence_time_days < min_residence_time_days


# A cell can belong to a lake only if the lake drains its upstream area. Allow a factor
# for the difference between the HydroLAKES catchment (HydroSHEDS 15") and the
# upstream area of the coarse model grid.
MAX_UPSTREAM_TO_LAKE_CATCHMENT_AREA_RATIO: float = 2.0


def remove_waterbody_cells_outside_catchment(
    waterbody_id: TwoDArrayInt32,
    upstream_area_m2: TwoDArrayFloat32,
    waterbody_ids: ArrayInt32,
    catchment_area_m2: ArrayFloat32,
    max_ratio: float = MAX_UPSTREAM_TO_LAKE_CATCHMENT_AREA_RATIO,
) -> TwoDArrayInt32:
    """Remove lake cells whose upstream area is larger than the lake can drain.

    Lakes are rasterized on all grid cells they touch. A small lake next to a river
    (e.g. a gravel pit) then takes over the river cell, and the whole river would flow
    through the lake. A cell whose upstream area is much larger than the catchment area
    of the lake (HydroLAKES Wshd_area) carries water that does not pass through the
    lake, so it stays river. Lakes the river flows through have a catchment as large
    as the river's and keep their cells.

    Args:
        waterbody_id: 2D array of waterbody IDs (-1 for no waterbody).
        upstream_area_m2: 2D array of the upstream area of each grid cell (m2).
        waterbody_ids: IDs of the waterbodies.
        catchment_area_m2: Catchment area of each waterbody (m2), same order as
            waterbody_ids.
        max_ratio: Largest allowed ratio of cell upstream area to lake catchment area.

    Returns:
        Updated 2D array of waterbody IDs.
    """
    updated_waterbody_id: TwoDArrayInt32 = waterbody_id.copy()
    is_waterbody: TwoDArrayBool = waterbody_id != -1
    if not is_waterbody.any():
        return updated_waterbody_id
    order: ArrayInt64 = np.argsort(waterbody_ids)
    positions: ArrayInt64 = np.searchsorted(
        waterbody_ids[order], waterbody_id[is_waterbody]
    )
    assert (waterbody_ids[order][positions] == waterbody_id[is_waterbody]).all(), (
        "every rasterized waterbody needs a catchment area"
    )
    cell_catchment_area_m2 = catchment_area_m2[order][positions]
    outside_catchment = upstream_area_m2[is_waterbody] > (
        max_ratio * cell_catchment_area_m2
    )
    flat_index = np.flatnonzero(is_waterbody.ravel())[outside_catchment]
    updated_waterbody_id.ravel()[flat_index] = -1
    return updated_waterbody_id


def absorb_trapped_river_cells(
    waterbody_id: TwoDArrayInt32,
    waterbody_outflows: TwoDArrayInt32,
    river_ids: TwoDArrayInt32,
    flow_raster: FlwdirRaster,
) -> TwoDArrayInt32:
    """Finds trapped river cells between waterbodies and assigns them the upstream waterbody ID.

    A trapped river cell is a cell with an active river ID that is downstream of a
    non-outflow waterbody cell and reconnects downstream into a waterbody.

    Args:
        waterbody_id: 2D array of water body IDs (-1 for non-waterbody).
        waterbody_outflows: 2D array containing waterbody IDs at outflow points and -1 elsewhere.
        river_ids: 2D array of river reach IDs (-1 for non-river).
        flow_raster: FlwdirRaster flow direction object.

    Returns:
        Updated 2D array of waterbody IDs with trapped cells absorbed.
    """
    updated_waterbody_id: TwoDArrayInt32 = waterbody_id.copy()

    # Identify all waterbody cells that are NOT outflows
    non_outflow_mask: TwoDArrayBool = (waterbody_id != -1) & (waterbody_outflows == -1)
    non_outflow_idxs: ArrayInt64 = np.where(non_outflow_mask.ravel())[0]
    if non_outflow_idxs.size == 0:
        return updated_waterbody_id

    downstream_idxs: ArrayInt64 = flow_raster.idxs_ds[non_outflow_idxs]

    # Filter for downstream cells that exit the waterbody into an active river cell
    valid_candidates: ArrayBool = (
        (downstream_idxs != -1)
        & (downstream_idxs != non_outflow_idxs)
        & (river_ids.ravel()[downstream_idxs] != -1)
        & (waterbody_id.ravel()[downstream_idxs] == -1)
    )
    if not np.any(valid_candidates):
        return updated_waterbody_id

    start_idxs: ArrayInt64 = downstream_idxs[valid_candidates]
    source_waterbody_ids: ArrayInt32 = waterbody_id.ravel()[
        non_outflow_idxs[valid_candidates]
    ]

    # Trace downstream paths until hitting a waterbody or leaving the river network
    termination_mask: TwoDArrayBool = (waterbody_id != -1) | (river_ids == -1)
    paths: list[ArrayInt64]
    paths, _ = flow_raster.path(idxs=start_idxs, mask=termination_mask)

    flat_updated_waterbody_id: ArrayInt32 = updated_waterbody_id.ravel()
    trapped_cells_to_assign: dict[int, int] = {}

    for path, source_waterbody_id in zip(paths, source_waterbody_ids):
        # path[-1] is the cell where tracing stopped. If it belongs to the same waterbody,
        # all preceding cells along the path are river cells trapped inside this waterbody.
        if flat_updated_waterbody_id[path[-1]] == source_waterbody_id:
            for cell_idx in path[:-1]:
                cell_idx_int: int = int(cell_idx)
                if cell_idx_int not in trapped_cells_to_assign:
                    trapped_cells_to_assign[cell_idx_int] = int(source_waterbody_id)

    for cell_idx_int, assigned_waterbody_id in trapped_cells_to_assign.items():
        flat_updated_waterbody_id[cell_idx_int] = assigned_waterbody_id

    return updated_waterbody_id


# Longest hillslope (m), for cells with very few streams.
MAX_HILLSLOPE_LENGTH_M: float = 1000.0


def hillslope_length_from_streams_length(
    streams_length_m: xr.DataArray, cell_area_m2: xr.DataArray
) -> xr.DataArray:
    """Hillslope length as half the mean stream spacing.

    With drainage density D = stream length / area, streams lie on average 1 / D apart,
    so the distance from a divide to the stream is 1 / (2 D). Capped at
    MAX_HILLSLOPE_LENGTH_M.

    Args:
        streams_length_m: Stream length per grid cell (m).
        cell_area_m2: Area of each grid cell (m2).

    Returns:
        Hillslope length per grid cell (m).
    """
    drainage_density = streams_length_m / cell_area_m2
    hillslope_length = 1 / (2 * drainage_density)
    hillslope_length = xr.where(
        hillslope_length < MAX_HILLSLOPE_LENGTH_M,
        hillslope_length,
        MAX_HILLSLOPE_LENGTH_M,
    )
    hillslope_length.attrs["_FillValue"] = np.nan
    return hillslope_length


class Hydrography(BuildModelBase):
    """Contains all build methods for the hydrography for GEB."""

    def __init__(self) -> None:
        """Initializes the Hydrography class."""

    @build_method(
        depends_on=["setup_hydrography", "setup_cell_area", "setup_elevation"],
        required=True,
    )
    def setup_geomorphology(self, channel_initiation_area_km2: float = 0.2) -> None:
        """Sets up the subgrid slope, surface area ratio, and hillslope length.

        The slope of the subgrid elevation (Horn 3x3 stencil, latitude-dependent cell
        size in metres) is saved as ``landsurface/slope_m_per_m`` on the subgrid. The
        grid slope is computed from elevation averaged over a whole grid cell, which
        smooths out valley sides, so local processes such as soil thinning use this one.
        Pixels outside the mask are nodata: a missing neighbour takes the elevation of
        the centre pixel, so there is no artificial cliff at the region edge.

        The hillslope length (``drainage/hillslope_length_m``) is half the stream
        spacing, 1 / (2 * drainage density), of the streams that start at
        channel_initiation_area_km2 upstream area on the original D8 grid. The routing
        streams (``drainage/streams_length_m``) start at 1 km2, which is too sparse a
        network for hillslopes: channel heads mapped in the field in soil-mantled
        terrain in Oregon and California drain about 0.0001-0.1 km2, less on steeper
        slopes and in wetter climates (Montgomery, D. R., & Dietrich, W. E. (1988).
        Where do channels begin? Nature, 336, 232-234, doi:10.1038/336232a0, Fig. 1).
        The default of 0.2 km2 is a calibrated value, not a measured one: it lies above
        the field values, and it gives drainage densities and hillslope lengths that
        match mapped stream networks (e.g. the Meuse uplands: 1.6 km/km2 with 0.2 km2,
        against 0.8 km/km2 and a median hillslope length of about 730 m with 1 km2).

        Args:
            channel_initiation_area_km2: Upstream area at which the channels that
                bound the hillslopes start (km2).
        """
        outside: xr.DataArray = self.subgrid["mask"]
        subgrid_elevation: xr.DataArray = self.subgrid["landsurface/elevation"]
        # pyflwdir tests nodata with !=, which never matches NaN, so use a number
        elevation_nodata: np.float32 = np.float32(-9999.0)
        elevation_values: npt.NDArray[np.float32] = np.where(
            outside.values, elevation_nodata, subgrid_elevation.values
        ).astype(np.float32)
        slope_values: npt.NDArray[np.float32] = pyflwdir.dem.slope(
            elevation_values,
            nodata=float(elevation_nodata),
            latlon=True,
            transform=np.array(subgrid_elevation.rio.transform(recalc=True)),
        )
        del elevation_values

        slope_high_res: xr.DataArray = xr.DataArray(
            slope_values.astype(np.float32),
            coords=subgrid_elevation.coords,
            dims=subgrid_elevation.dims,
        ).where(~outside, np.float32(0.0))
        slope_high_res.attrs["_FillValue"] = np.nan
        slope_high_res = slope_high_res.rio.write_crs(subgrid_elevation.rio.crs)
        slope_high_res = self.set_subgrid(
            slope_high_res, name="landsurface/slope_m_per_m"
        )

        # pixels outside the mask are left out of the grid-cell mean
        surface_area_ratio_high_res: xr.DataArray = (
            (1 + slope_high_res**2) ** 0.5
        ).where(~outside)

        surface_area_ratio: xr.DataArray = surface_area_ratio_high_res.coarsen(
            x=self.subgrid_factor,
            y=self.subgrid_factor,
            boundary="exact",
            coord_func="mean",
        ).mean()
        surface_area_ratio.attrs["_FillValue"] = np.nan

        surface_area_ratio: xr.DataArray = snap_to_grid(
            surface_area_ratio, self.grid["mask"], relative_tolerance=0.03
        )
        self.set_grid(surface_area_ratio, name="landsurface/surface_area_ratio")

        streams_length: xr.DataArray = self._grid_streams_length(
            self.other["drainage/original_d8_flow_directions"].compute(),
            self.other["drainage/original_d8_upstream_area_m2"].compute(),
            channel_initiation_area_m2=channel_initiation_area_km2 * 1e6,
        )
        hillslope_length = hillslope_length_from_streams_length(
            streams_length, self.grid["cell_area"]
        )
        self.set_grid(hillslope_length, name="drainage/hillslope_length_m")

    def _grid_streams_length(
        self,
        original_d8_ldd: xr.DataArray,
        original_d8_upstream_area_m2: xr.DataArray,
        channel_initiation_area_m2: float,
    ) -> xr.DataArray:
        """Total stream length in each model grid cell.

        Streams are the original D8 pixels with an upstream area of at least
        channel_initiation_area_m2. Their length is summed per grid cell, with at least
        half a cell length per cell (every cell drains through some channel).

        Args:
            original_d8_ldd: Original high-resolution D8 flow directions.
            original_d8_upstream_area_m2: Upstream area on the original D8 grid (m2).
            channel_initiation_area_m2: Upstream area at which streams start (m2).

        Returns:
            Stream length per grid cell (m).
        """
        streams_length: xr.DataArray = (
            calculate_stream_length(
                original_d8_ldd,
                original_d8_upstream_area_m2,
                threshold_m2=channel_initiation_area_m2,
            )
            .coarsen(
                x=self.ldd_scale_factor,
                y=self.ldd_scale_factor,
                boundary="exact",
                coord_func="mean",
            )
            .sum()
        )
        streams_length.attrs["_FillValue"] = np.nan
        streams_length = snap_to_grid(streams_length, self.grid["mask"])
        return np.maximum(streams_length, np.sqrt(self.grid["cell_area"]) * 0.5)

    @build_method(required=False)
    def setup_mannings(self) -> None:
        """Removed method, use setup_geomorphology instead."""
        raise NotImplementedError(
            "The setup_mannings method is removed, use setup_geomorphology instead"
        )

    def get_rivers(
        self, river_graph: nx.DiGraph, sink_subbasin_ids: list[int]
    ) -> gpd.GeoDataFrame:
        """Sets the routing subbasins for the model.

        For each sink subbasin, all upstream subbasins and the immediately downstream subbasins are included.
        The downstream subbasins have a flag set to indicate that they are downstream outflow subbasins.

        If a sink subbasin has no downstream subbasin, it is assumed to be a coastal basin. This is
        also saved in the subbasins attribute.

        Args:
            river_graph: The river graph to use for determining upstream and downstream subbasins.
            sink_subbasin_ids: The subbasin IDs of the sink subbasins (i.e., the outflow subbasins).

        Returns:
            A GeoDataFrame containing the subbasins geometry for the model.
        """
        # always make a list of the subbasin ids, such that the function always gets the same type of input
        if not isinstance(sink_subbasin_ids, (list, set)):
            sink_subbasin_ids = [sink_subbasin_ids]

        subbasin_ids: set[int] = get_all_upstream_subbasin_ids(
            river_graph=river_graph, subbasin_ids=sink_subbasin_ids
        )
        subbasin_ids.update(sink_subbasin_ids)

        downstream_subbasins: dict[int, list[int]] = get_immediate_downstream_subbasins(
            river_graph, sink_subbasin_ids
        )
        # remove downstream subbasins that are also upstream subbasins, because those are already included in the river network
        downstream_subbasins: dict[int, list[int]] = {
            k: v for k, v in downstream_subbasins.items() if k not in subbasin_ids
        }
        subbasin_ids.update(downstream_subbasins)

        is_further_downstream_outflow: set[int] = set()

        is_further_downstream_outflow.update(
            get_all_downstream_subbasin_ids(
                river_graph, list(downstream_subbasins.keys())
            )
        )

        # remove further downstream subbasins that are also upstream subbasins, because those are already included in the river network
        is_further_downstream_outflow = {
            subbasin_id
            for subbasin_id in is_further_downstream_outflow
            if subbasin_id not in subbasin_ids
        }
        subbasin_ids.update(is_further_downstream_outflow)

        # later we want to include the downstream outflow basins. However, we don't want to include
        # other branches that are upstream of those downstream basins, but are not part
        # of the area that we are interested in. Therefore, we also include
        # the immediately upstream basins of the downstream basins, so that we can stop the subbasin
        # construction there.
        upstream_basins_of_downstream_basins = set()
        for downstream_subbasin in downstream_subbasins.keys():
            for upstream_basin in river_graph.predecessors(downstream_subbasin):
                if upstream_basin not in subbasin_ids:
                    upstream_basins_of_downstream_basins.add(upstream_basin)

        subbasin_ids.update(upstream_basins_of_downstream_basins)

        rivers: gpd.GeoDataFrame = get_rivers_geometry(
            self.data_catalog, list(subbasin_ids)
        )

        rivers["is_downstream_outflow"] = pd.Series(
            True, index=downstream_subbasins
        ).reindex(rivers.index, fill_value=False)

        rivers["is_upstream_of_downstream_basin"] = pd.Series(
            True, index=upstream_basins_of_downstream_basins
        ).reindex(rivers.index, fill_value=False)

        rivers["is_further_downstream_outflow"] = pd.Series(
            True, index=is_further_downstream_outflow
        ).reindex(rivers.index, fill_value=False)

        return rivers

    @build_method(required=True, depends_on=["setup_cell_area"])
    def setup_hydrography(
        self,
        custom_rivers: str | None = None,
        custom_rivers_width_m_column: str | None = None,
        custom_rivers_depth_m_column: str | None = None,
    ) -> None:
        """Sets up the hydrography for the model.

        Steps:
            1. Resample the original D8 elevation and flow direction to the model resolution.
            2. Calculate the flow direction (LDD) and upstream area based on the resampled elevation and flow direction.
            3. Calculate the slope based on the resampled elevation.
            4. Rasterize the river network, and setting up variables to link
                the low-resolution and high resolution grids.
            5. Calculate river attributes such as river length, river slope, and river width.

        Args:
            custom_rivers: Optional path to a custom river shapefile. The MERIT rivers will still be
                used to create the river raster, but the burning of rivers in the DEM will be done
                based on the custom river shapefile. Must be readable by geopandas.read_file() or geopandas.read_parquet().
            custom_rivers_width_m_column: The column name in the custom rivers file that contains the river width in meters.
                must be provided if custom_rivers is provided.
            custom_rivers_depth_m_column: The column name in the custom rivers file that contains the river depth in meters.
                Must be provided if custom_rivers is provided.

        Raises:
            FileNotFoundError: If the custom rivers file is not found.
            ValueError: If custom_rivers_width_m_column or custom_rivers_depth_m_column is not provided when using custom_rivers.
            KeyError: If custom_rivers_width_m_column or custom_rivers_depth_m_column is not found in the custom rivers file.
            AssertionError: If some large rivers are not represented in the grid.
        """
        if custom_rivers is not None:
            custom_rivers: Path = Path(custom_rivers)
            self.logger.info(f"Using custom rivers from {custom_rivers}")
            if not custom_rivers.exists():
                raise FileNotFoundError(f"Custom rivers file {custom_rivers} not found")
            if custom_rivers.suffix in [".parquet", ".pq", ".geoparquet"]:
                custom_rivers_gdf = gpd.read_parquet(custom_rivers)
            else:
                custom_rivers_gdf = gpd.read_file(custom_rivers)
            custom_rivers_gdf = custom_rivers_gdf.to_crs("EPSG:4326")
            if (
                custom_rivers_width_m_column is None
                or custom_rivers_depth_m_column is None
            ):
                raise ValueError(
                    "custom_rivers_width_m_column and custom_rivers_depth_m_column must be provided when using custom_rivers"
                )
            if custom_rivers_width_m_column not in custom_rivers_gdf.columns:
                raise KeyError(
                    f"custom_rivers_width_m_column '{custom_rivers_width_m_column}' not found in custom rivers file"
                )
            if custom_rivers_depth_m_column not in custom_rivers_gdf.columns:
                raise KeyError(
                    f"custom_rivers_depth_m_column '{custom_rivers_depth_m_column}' not found in custom rivers file"
                )
            custom_rivers_gdf = custom_rivers_gdf.rename(
                columns={
                    custom_rivers_width_m_column: "width",
                    custom_rivers_depth_m_column: "depth",
                }
            )
            self.set_geom(custom_rivers_gdf, name="routing/custom_rivers")

        original_d8_elevation = self.other["drainage/original_d8_elevation"]
        original_d8_ldd = self.other["drainage/original_d8_flow_directions"].compute()
        original_d8_ldd_data = original_d8_ldd.values

        flow_raster_original = pyflwdir.from_array(
            original_d8_ldd_data,
            ftype="d8",
            transform=np.array(original_d8_ldd.rio.transform(recalc=True)),
            latlon=True,  # hydrography is specified in latlon
            mask=original_d8_ldd_data
            != original_d8_ldd.attrs[
                "_FillValue"
            ],  # this mask is True within study area
        )

        original_d8_upstream_area_m2 = self.full_like(
            original_d8_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        original_d8_upstream_area_m2_data = flow_raster_original.upstream_area(
            unit="m2"
        ).astype(np.float32)
        original_d8_upstream_area_m2_data[
            original_d8_upstream_area_m2_data == -9999.0
        ] = np.nan
        original_d8_upstream_area_m2.data = original_d8_upstream_area_m2_data
        self.set_other(
            original_d8_upstream_area_m2, name="drainage/original_d8_upstream_area_m2"
        )

        routing_streams_length = self._grid_streams_length(
            original_d8_ldd,
            original_d8_upstream_area_m2,
            channel_initiation_area_m2=1e6,
        )
        self.set_grid(routing_streams_length, name="drainage/streams_length_m")

        elevation_coarsened = original_d8_elevation.coarsen(
            x=self.ldd_scale_factor,
            y=self.ldd_scale_factor,
            boundary="exact",
            coord_func="mean",
        )

        # Mean and minimum of the hydrography DEM (MERIT) per grid cell. These are only
        # used here, for the grid slope and routing/height_above_nearest_drainage_m,
        # because those must match the flow directions derived from this DEM. They
        # are deliberately not saved as the model elevation: landsurface/elevation_m
        # and landsurface/elevation_min_m come from the subgrid DEM
        # (setup_grid_elevation), so that all elevations in the model share one DEM.
        elevation = elevation_coarsened.mean()
        elevation = snap_to_grid(elevation, self.grid["mask"])

        elevation_min = elevation_coarsened.min()
        elevation_min = snap_to_grid(elevation_min, self.grid["mask"])

        slope = self.full_like(
            elevation_min, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        slope_data = pyflwdir.dem.slope(
            elevation.values,
            nodata=np.nan,
            latlon=True,
            transform=np.array(elevation.rio.transform(recalc=True)),
        )
        # set slope to zero on the mask boundary
        slope_data[np.isnan(slope_data) & (~self.grid["mask"].data)] = 0
        slope.data = slope_data
        self.set_grid(slope, name="landsurface/slope_m_per_m")

        flow_raster_idxs_ds = self.grid["flow_raster_idxs_ds"].compute()
        flow_raster = FlwdirRaster(
            flow_raster_idxs_ds.values.ravel(),
            shape=flow_raster_idxs_ds.shape,
            transform=np.array(flow_raster_idxs_ds.rio.transform(recalc=True)),
            ftype="d8",
            latlon=True,
        )

        # flow direction
        ldd: xr.DataArray = self.full_like(
            elevation_min, fill_value=255, nodata=255, dtype=np.uint8
        )
        ldd.data = flow_raster.to_array(ftype="ldd")
        self.set_grid(ldd, name="routing/ldd")

        # upstream area
        upstream_area: xr.DataArray = self.full_like(
            elevation_min, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        upstream_area_data: npt.NDArray[np.float32] = flow_raster.upstream_area(
            unit="m2"
        ).astype(np.float32)
        upstream_area_data[upstream_area_data == -9999.0] = np.nan
        upstream_area.data = upstream_area_data
        self.set_grid(upstream_area, name="routing/upstream_area_m2")

        upstream_area_n_cells: xr.DataArray = self.full_like(
            elevation_min, fill_value=-1, nodata=-1, dtype=np.int32
        )
        upstream_area_n_cells_data: npt.NDArray[np.int32] = flow_raster.upstream_area(
            unit="cell"
        ).astype(np.int32)
        upstream_area_n_cells.data = upstream_area_n_cells_data
        self.set_grid(upstream_area_n_cells, name="routing/upstream_area_n_cells")

        # river length
        river_length: xr.DataArray = self.full_like(
            elevation_min, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_length_data: npt.NDArray[np.float32] = (
            flow_raster_original.subgrid_rivlen(
                self.grid["idxs_outflow"].values, unit="m", direction="down"
            )
        )
        minimum_river_length = (
            np.sqrt(self.grid["cell_area"].mean().compute().item()) / 2
        ).astype(river_length_data.dtype)

        # we smooth the river length, so that the local inertial
        # calculations do not need to consider very small domains.
        self.logger.info("Smoothening river length")
        river_length_data = flow_raster.smooth_rivlen(
            river_length_data,
            minimum_river_length,
            max_window=100,
            nodata=-9999.0,
        )
        river_length_data[river_length_data == -9999.0] = np.nan
        river_length_data[river_length_data < minimum_river_length] = (
            minimum_river_length
        )
        assert np.nanmin(river_length_data.ravel()) >= minimum_river_length
        river_length.data = river_length_data
        self.set_grid(river_length, name="routing/river_length_m")

        self.logger.info("Retrieving river data")
        rivers: gpd.GeoDataFrame = self.geom["routing/rivers"]
        rivers = add_stream_orders(rivers)

        self.logger.info("Processing river data")

        original_river_raster: npt.NDArray[np.int32] = (
            create_river_raster_from_river_lines(rivers, original_d8_elevation)
        )
        # Discharge stations are matched directly to these original river pixels.
        original_river_ids: xr.DataArray = self.full_like(
            original_d8_elevation, fill_value=-1, nodata=-1, dtype=np.int32
        )
        original_river_ids.data = original_river_raster
        self.set_other(original_river_ids, name="drainage/original_river_ids")
        routing_river_raster: npt.NDArray[np.int32] = original_river_raster.ravel()[
            self.grid["idxs_outflow"].values.ravel()
        ].reshape(self.grid["idxs_outflow"].shape)

        for COMID in rivers.sort_values(
            by="shreve_stream_order",
            ascending=False,
        ).index:
            river_cells = np.where(routing_river_raster.ravel() == COMID)[0]
            if river_cells.size == 0:
                continue
            upstream_area_river_cells = upstream_area_data.ravel()[river_cells]
            most_upstream_cell = np.argmin(upstream_area_river_cells)
            most_upstream_cell_index = river_cells[most_upstream_cell]
            upstream_river_cells = (flow_raster.idxs_ds == most_upstream_cell_index) & (
                routing_river_raster.ravel() != -1
            )
            if upstream_river_cells.sum() == 1:
                routing_river_raster[
                    upstream_river_cells.reshape(routing_river_raster.shape)
                ] = COMID

        # Propagate river IDs downstream to fill gaps in the routing river network
        routing_river_raster = propagate_downstream(
            data=routing_river_raster,
            idxs_ds=flow_raster.idxs_ds,
            upstream_area=upstream_area_data,
            missing_val=-1,
        )

        missing_rivers: set[int] = set(rivers.index) - set(
            np.unique(routing_river_raster[routing_river_raster != -1]).tolist()
        )

        rivers["represented_in_grid"] = True
        rivers.iloc[
            rivers.index.isin(missing_rivers),
            rivers.columns.get_loc("represented_in_grid"),
        ] = False

        assert (
            rivers[
                (~rivers["represented_in_grid"])
                & (~rivers["is_downstream_outflow"])
                & (~rivers["is_further_downstream_outflow"])
            ]["lengthkm"]
            < 5
        ).all(), (
            "Some large rivers are not represented in the grid, please check the "
            "rasterization of the river lines"
        )

        rivers["hydrography_xy"] = [[] for _ in range(len(rivers))]
        rivers["hydrography_upstream_area_m2"] = [[] for _ in range(len(rivers))]
        xy_per_river_segment = value_indices(routing_river_raster, ignore_value=-1)
        for COMID, (ys, xs) in xy_per_river_segment.items():
            upstream_area = upstream_area_data[ys, xs]
            up_to_downstream_ids = np.argsort(upstream_area)
            upstream_area_sorted = upstream_area[up_to_downstream_ids]

            assert (routing_river_raster[ys, xs] == COMID).all(), (
                f"River segment {COMID} has inconsistent raster values"
            )

            ys: npt.NDArray[np.int64] = ys[up_to_downstream_ids]
            xs: npt.NDArray[np.int64] = xs[up_to_downstream_ids]
            assert ys.size > 0, "No xy coordinates found for river segment"
            rivers.at[COMID, "hydrography_xy"] = list(zip(xs, ys, strict=True))
            rivers.at[COMID, "hydrography_upstream_area_m2"] = (
                upstream_area_sorted.tolist()
            )

        rivers["hydrography_high_res_lons_lats"] = [[] for _ in range(len(rivers))]
        rivers["hydrography_high_res_upstream_area_m2"] = [
            [] for _ in range(len(rivers))
        ]
        xy_per_river_segment = value_indices(original_river_raster, ignore_value=-1)

        for river_ID, river in rivers.iterrows():
            if river_ID not in xy_per_river_segment:
                if river["is_downstream_outflow"] or not river["represented_in_grid"]:
                    continue
                else:
                    raise AssertionError("River xy not found, but should be found.")

            (ys, xs) = xy_per_river_segment[river_ID]
            upstream_area: ArrayFloat32 = original_d8_upstream_area_m2_data[ys, xs]
            nan_mask: ArrayBool = np.isnan(upstream_area)

            if nan_mask.all():
                if river["is_downstream_outflow"] or not river["represented_in_grid"]:
                    continue
                else:
                    raise AssertionError(
                        "River xy all NaN upstream area, but should have valid values."
                    )

            non_nan_mask: ArrayBool = ~nan_mask
            upstream_area = upstream_area[non_nan_mask]
            ys: ArrayInt64 = ys[non_nan_mask]
            xs: ArrayInt64 = xs[non_nan_mask]

            assert not np.isnan(upstream_area).any()

            up_to_downstream_ids = np.argsort(upstream_area)
            upstream_area_sorted = upstream_area[up_to_downstream_ids]

            assert (original_river_raster[ys, xs] == river_ID).all(), (
                f"River segment {river_ID} has inconsistent raster values"
            )

            ys: ArrayInt64 = ys[up_to_downstream_ids]
            xs: ArrayInt64 = xs[up_to_downstream_ids]

            lats: ArrayFloat32 = original_d8_upstream_area_m2.y.values[ys]
            lons: ArrayFloat32 = original_d8_upstream_area_m2.x.values[xs]

            assert ys.size > 0, "No xy coordinates found for river segment"
            rivers.at[river_ID, "hydrography_high_res_lons_lats"] = list(
                zip(lons.tolist(), lats.tolist(), strict=True)
            )
            rivers.at[river_ID, "hydrography_high_res_upstream_area_m2"] = (
                upstream_area_sorted.tolist()
            )

        for river_ID, river in rivers.iterrows():
            if river["represented_in_grid"]:
                assert len(river["hydrography_xy"]) > 0, (
                    f"River {river_ID} has no xy coordinates, please check the "
                    "rasterization of the river lines"
                )

        COMID_IDs_raster: xr.DataArray = self.full_like(
            elevation_min, fill_value=-1, nodata=-1, dtype=np.int32
        )
        COMID_IDs_raster.data = routing_river_raster
        self.set_grid(COMID_IDs_raster, name="routing/river_ids")

        height_above_nearest_drainage_m = self.full_like(
            elevation_min, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        height_above_nearest_drainage_m.values = flow_raster.hand(
            COMID_IDs_raster != -1, elevation_min
        ).astype(np.float32)
        height_above_nearest_drainage_m.values[
            height_above_nearest_drainage_m.values == -9999.0
        ] = np.nan
        assert not np.isnan(
            height_above_nearest_drainage_m.values[~self.grid["mask"].values]
        ).all()
        self.set_grid(
            height_above_nearest_drainage_m,
            name="routing/height_above_nearest_drainage_m",
        )

        basin_ids = self.full_like(
            elevation_min, fill_value=-1, nodata=-1, dtype=np.int32
        )

        river_linear_indices = np.where(COMID_IDs_raster.values.ravel() != -1)[0]
        ids = COMID_IDs_raster.values.ravel()[river_linear_indices]
        assert 0 not in ids, (
            "COMID ID of 0 found in river raster, which would lead to errors in pyflwdir, which uses 0 as nodata value."
        )
        basin_ids.values = flow_raster.basins(
            idxs=river_linear_indices,
            ids=COMID_IDs_raster.values.ravel()[river_linear_indices],
        )
        basin_ids = xr.where(basin_ids != 0, basin_ids, -1)
        assert (
            np.unique(basin_ids.values[basin_ids.values != -1])
            == np.unique(COMID_IDs_raster.values[COMID_IDs_raster.values != -1])
        ).all()
        self.set_grid(basin_ids, name="routing/basin_ids")

        SWORD_reach_IDs, SWORD_reach_lengths = get_SWORD_translation_IDs_and_lengths(
            self.data_catalog, rivers
        )

        SWORD_river_widths: npt.NDArray[np.float64] = get_SWORD_river_widths(
            self.data_catalog, SWORD_reach_IDs
        )

        valid_SWORD_widths = ~np.isnan(SWORD_river_widths)
        valid_reach_lengths_m = np.where(valid_SWORD_widths, SWORD_reach_lengths, 0)
        total_valid_reach_length_m = np.sum(valid_reach_lengths_m, axis=0)
        rivers_with_data = valid_SWORD_widths.any(axis=0) & (
            total_valid_reach_length_m > 0
        )

        rivers["width"] = np.nan
        rivers.loc[rivers_with_data, "width"] = (
            np.nansum(SWORD_river_widths * valid_reach_lengths_m, axis=0)[
                rivers_with_data
            ]
            / total_valid_reach_length_m[rivers_with_data]
        )

        self.set_geom(rivers, name="routing/rivers")

        river_with_mapper: dict[int, float] = rivers["width"].to_dict()
        river_width_data: npt.NDArray[np.float32] = np.vectorize(
            lambda ID: river_with_mapper.get(ID, np.nan)
        )(COMID_IDs_raster.values).astype(np.float32)

        # Propagate missing river width data downstream
        river_width_data = propagate_downstream(
            data=river_width_data,
            idxs_ds=flow_raster.idxs_ds,
            upstream_area=upstream_area_data,
            missing_val=np.nan,
            mask=COMID_IDs_raster.values != -1,
        )

        # check river width is positive where river is present
        assert ((river_width_data > 0) | np.isnan(river_width_data)).all(), (
            "River width should be positive or nan"
        )

        river_width: xr.DataArray = self.full_like(
            elevation_min, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_width.data = river_width_data
        self.set_grid(river_width, name="routing/river_width_m")

    @build_method(
        depends_on=[
            "setup_hydrography",
            "setup_elevation",
            "setup_grid_elevation",
            "setup_waterbodies",
        ],
        required=True,
    )
    def setup_river_surface_profiles(
        self,
        dem_name: str = "fathomdem",
        sampling_distance_m: float = 30.0,
        window_radius_pixels: int = 1,
    ) -> None:
        """Sets up river water-surface profiles and the riverbed of the routing grid.

        The water surface along every MERIT-Hydro river line is taken from a
        high-resolution DEM (default FathomDEM, 1 arc-second) with
        ``estimate_river_surface_profiles``: the lowest DEM pixel next to the line,
        made non-increasing downstream (holes such as gravel pits filled, bumps such
        as bridges cut) and with the 0.5 m terraces of the DEM water surfaces smoothed;
        lakes stay flat. The profiles are stored as points for the flood model.

        From these profiles the riverbed of the routing grid
        (``routing/bankfull_river_elevation_m``) is built in three steps:

        1. All cells start from the lowest subgrid elevation in the cell
           (``landsurface/elevation_min_m``, taken as the cell's stream), made
           non-increasing downstream with ``dem_adjust``.
        2. Cells with a represented river are overwritten with the profile water
           surface at their outflow pixel (``river_cell_surface_elevation``).
        3. Any cell that now lies below its downstream cell is raised to it, so the
           bed is non-increasing downstream again.

        Because the elevation levels inside a cell come from the same subgrid
        elevation, the land of a cell without a river lies at or above its stream.

        Depends on the waterbodies (lakes stay flat), so rebuild this after the
        waterbodies.

        Args:
            dem_name: Name of the DEM in the model 'other' store.
            sampling_distance_m: Distance between profile points (m).
            window_radius_pixels: Half-width of the DEM window searched for the water
                surface around each point (DEM pixels).

        Raises:
            KeyError: If the DEM is not in the model 'other' store.
            ValueError: If a river cell gets no elevation.
        """
        dem_key: str = f"DEM/{dem_name}"
        if dem_key not in self.other:
            raise KeyError(f"DEM '{dem_name}' not found at '{dem_key}'.")
        elevation: xr.DataArray = self.other[dem_key]

        rivers: gpd.GeoDataFrame = self.geom["routing/rivers"]
        profiles: gpd.GeoDataFrame = estimate_river_surface_profiles(
            rivers=rivers,
            elevation=elevation,
            lakes=self.geom["waterbodies/waterbody_data"],
            sampling_distance_m=sampling_distance_m,
            window_radius_pixels=window_radius_pixels,
        )
        self.set_geom(profiles, name="routing/river_surface_profiles")

        flow_raster_idxs_ds = self.grid["flow_raster_idxs_ds"].compute()
        flow_raster = FlwdirRaster(
            flow_raster_idxs_ds.values.ravel(),
            shape=flow_raster_idxs_ds.shape,
            transform=np.array(flow_raster_idxs_ds.rio.transform(recalc=True)),
            ftype="d8",
            latlon=True,
        )
        bed: npt.NDArray[np.float32] = flow_raster.dem_adjust(
            self.grid["landsurface/elevation_min_m"].values
        ).astype(np.float32)

        high_res = self.other["drainage/original_d8_flow_directions"]
        river_surface = river_cell_surface_elevation(
            rivers=rivers[rivers["represented_in_grid"]],
            profiles=profiles,
            idxs_outflow=self.grid["idxs_outflow"].values.ravel().astype(np.int64),
            high_res_x=high_res.x.values,
            high_res_y=high_res.y.values,
            grid_shape=bed.shape,
        )
        is_river: npt.NDArray[np.bool_] = self.grid["routing/river_ids"].values != -1
        if np.isnan(river_surface[is_river]).any():
            raise ValueError("Some river cells got no water-surface elevation.")
        bed[is_river] = river_surface[is_river]

        # Make the bed non-increasing downstream again (step 3 in the docstring). This
        # raises mostly cells where a small stream (subgrid minimum) joins a river (DEM
        # water surface), and rarely cells along meanders.
        before = bed.ravel().copy()
        flat_bed = raise_to_downstream(
            before,
            idxs_ds=flow_raster.idxs_ds.astype(np.int64),
            idxs_seq=flow_raster.idxs_seq.astype(np.int64),
        )
        raised = flat_bed - before
        self.logger.info(
            f"River surface profiles: {len(profiles)} points on {len(rivers)} rivers; "
            f"routing cells raised to their downstream cell: "
            f"{int((raised[is_river.ravel()] > 0).sum())} river cells "
            f"(max {float(raised[is_river.ravel()].max(initial=0.0)):.2f} m), "
            f"{int((raised[~is_river.ravel()] > 0).sum())} other cells "
            f"(max {float(np.nanmax(raised[~is_river.ravel()], initial=0.0)):.2f} m)"
        )
        bankfull_river_elevation_m = self.full_like(
            self.grid["landsurface/elevation_min_m"],
            fill_value=np.nan,
            nodata=np.nan,
            dtype=np.float32,
        )
        bankfull_river_elevation_m.data = flat_bed.reshape(bed.shape)
        self.set_grid(
            bankfull_river_elevation_m, name="routing/bankfull_river_elevation_m"
        )

    @build_method(
        depends_on=[
            "setup_hydrography",
            "setup_elevation",
        ],
        required=True,
    )
    def setup_floodplains(
        self,
        dem_name: str = "fathomdem",
    ) -> None:
        """Sets up 10-point floodplain width profile and side-slope ratio per reach using 1 arc-second DEM.

        Combines the hydrologically conditioned drainage network and subbasins from MERIT Hydro with
        subgrid elevations from a high-resolution DEM (default: FathomDEM at 1 arc-second resolution)
        to compute Height Above Nearest Drainage (HAND) and reach-level inundated width profiles.

        Args:
            dem_name: Name of the DEM raster in model 'other' store to use for subgrid elevations. Default is 'fathomdem'.

        Raises:
            KeyError: If the specified DEM is not found in the model 'other' store.
        """
        dem_key: str = f"DEM/{dem_name}"
        if dem_key not in self.other:
            raise KeyError(
                f"DEM '{dem_name}' not found at '{dem_key}' in model 'other' store. "
                f"Ensure setup_elevation has completed and configured '{dem_name}'."
            )

        elevation_1sec: xr.DataArray = self.other[dem_key]
        flow_raster_original_ldd: xr.DataArray = self.other[
            "drainage/original_d8_flow_directions"
        ].compute()
        original_d8_ldd_data = flow_raster_original_ldd.values
        flow_raster_original = pyflwdir.from_array(
            original_d8_ldd_data,
            ftype="d8",
            transform=np.array(flow_raster_original_ldd.rio.transform(recalc=True)),
            latlon=True,
            mask=original_d8_ldd_data != flow_raster_original_ldd.attrs["_FillValue"],
        )
        original_d8_elevation: xr.DataArray = self.other[
            "drainage/original_d8_elevation"
        ]
        original_d8_upstream_area_m2: xr.DataArray = self.other[
            "drainage/original_d8_upstream_area_m2"
        ]

        self.logger.info(
            f"Computing basin-attributed floodplain geometry per river reach using {dem_name}"
        )
        (
            floodplain_side_slope_m_per_m,
            floodplain_profile_widths_m,
        ) = calculate_dem_floodplain_geometry(
            flow_raster_high_res=flow_raster_original,
            elevation_high_res=original_d8_elevation,
            elevation_1sec=elevation_1sec,
            upstream_area_high_res_m2=original_d8_upstream_area_m2,
            idxs_outflow_low_res=self.grid["idxs_outflow"],
            river_raster_low_res=self.grid["routing/river_ids"].values,
            cell_area_low_res=self.grid["cell_area"],
            river_length_low_res=self.grid["routing/river_length_m"],
            ldd_scale_factor=self.ldd_scale_factor,
        )
        self.set_grid(
            floodplain_side_slope_m_per_m,
            name="routing/floodplain_side_slope_m_per_m",
        )
        self.set_grid(
            floodplain_profile_widths_m,
            name="routing/floodplain_profile_widths_m",
        )

    @build_method(required=True)
    def setup_global_ocean_mean_dynamic_topography(self) -> None:
        """Sets up the global ocean mean dynamic topography for the model."""
        if not self.geom["routing/subbasins"]["is_coastal"].any():
            self.logger.info(
                "No coastal basins found, skipping setup_global_ocean_mean_dynamic_topography"
            )
            return

        # get the model bounds and buffer by ~10km
        model_bounds = self.bounds
        model_bounds = (
            model_bounds[0] - 0.083,  # min_lon
            model_bounds[1] - 0.083,  # min_lat
            model_bounds[2] + 0.083,  # max_lon
            model_bounds[3] + 0.083,  # max_lat
        )

        global_ocean_mdt = self.data_catalog.fetch(
            "global_ocean_mean_dynamic_topography"
        ).read(model_bounds)

        # write to model
        self.set_other(
            global_ocean_mdt,
            name="coastal/global_ocean_mean_dynamic_topography",
        )

    def create_low_elevation_coastal_zone_mask(self) -> gpd.GeoDataFrame:
        """creates the low elevation coastal zone (LECZ) mask for sfincs models.

        Returns:
            A GeoDataFrame containing the low elevation coastal zone mask.
        """
        # load low elevation coastal zone mask
        low_elevation_coastal_zone = self.other[
            "landsurface/low_elevation_coastal_zone"
        ]
        mask_data = low_elevation_coastal_zone.values.astype(np.uint8)

        # Get transform from raster metadata
        transform = low_elevation_coastal_zone.rio.transform()

        # Use rasterio.features.shapes() to get polygons for each contiguous region with same value
        shapes = rasterio.features.shapes(mask_data, mask=None, transform=transform)

        # Build GeoDataFrame from the shapes generator
        records = [{"geometry": shape(geom), "value": value} for geom, value in shapes]

        gdf = gpd.GeoDataFrame.from_records(records)
        gdf.set_geometry("geometry", inplace=True)
        gdf.crs = low_elevation_coastal_zone.rio.crs
        # Keep only mask == 1
        gdf = gdf[gdf["value"] == 1]

        # load mask to select coastal areas in model region
        # intersect the mask with the low elevation coastal zone mask
        low_elevation_coastal_zone_mask = gpd.overlay(
            gdf, self.geom["mask"], how="intersection"
        )
        # merge all polygons into a single polygon
        low_elevation_coastal_zone_mask = gpd.GeoDataFrame(
            geometry=[low_elevation_coastal_zone_mask.union_all()],
            crs=gdf.crs,
        )
        return low_elevation_coastal_zone_mask

    @build_method(required=True)
    def setup_coastlines(self) -> None:
        """Sets up the coastlines for the model."""
        if self.geom["routing/subbasins"]["is_coastal"].any():
            # load the coastline from the data catalog
            coastlines = self.data_catalog.fetch("open_street_map_coastlines").read()

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                )
                mask = gpd.GeoDataFrame.from_features(
                    self.geom["mask"].buffer(0.04).__geo_interface__,
                    crs=self.geom["mask"].crs,
                )

            # clip the coastline to overlapping with mask
            coastlines = gpd.overlay(coastlines, mask, how="intersection")

            # merge all coastlines into a single linestring
            coastlines = gpd.GeoDataFrame(
                geometry=[coastlines.union_all()], crs=coastlines.crs
            )

            # write to model files
            self.set_geom(coastlines, name="coastal/coastlines")

            # create rectangular box around coastlines
            if not coastlines.empty:
                bbox = coastlines.minimum_rotated_rectangle().iloc[0]  # get the Polygon
                bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=coastlines.crs)

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                    )
                    bbox_gdf.geometry = bbox_gdf.geometry.buffer(
                        0.04, join_style=2
                    )  # buffer by 0.04 degree
                self.set_geom(bbox_gdf, name="coastal/coastline_bbox")
        else:
            self.logger.info("No coastal basins found, setting empty coastlines")
            empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            self.set_geom(empty_gdf, name="coastal/coastlines")
            self.set_geom(empty_gdf, name="coastal/coastline_bbox")

    @build_method(required=True)
    def setup_osm_land_polygons(
        self,
    ) -> None:
        """Sets up the OSM land polygons for the model."""
        if not self.geom["routing/subbasins"]["is_coastal"].any():
            self.logger.info(
                "No coastal basins found, skipping setup_osm_land_polygons"
            )
            return

        # load the land polygon from the data catalog
        land_polygons = self.data_catalog.fetch("open_street_map_land_polygons").read()
        # select only the land polygons that intersect with the region
        land_polygons = land_polygons[land_polygons.intersects(self.region.union_all())]
        # merge all land polygons into a single polygon
        land_polygons = gpd.GeoDataFrame(
            geometry=[land_polygons.union_all()], crs=land_polygons.crs
        )

        # clip and write to model files
        self.set_geom(land_polygons.clip(self.bounds), name="coastal/land_polygons")

    @build_method(
        depends_on=["setup_coastlines", "setup_elevation"],
        required=True,
    )
    def setup_coastal_sfincs_model_regions(self) -> None:
        """Sets up the coastal sfincs model regions."""
        if self.geom["routing/subbasins"]["is_coastal"].any():
            # load elevation data
            elevation = self.other["DEM/fathomdem"]
            # load the lecz mask
            low_elevation_coastal_zone_mask = (
                self.create_low_elevation_coastal_zone_mask()
            )

            # add small buffer to ensure connection of 'islands' with coastlines
            low_elevation_coastal_zone_mask.geometry = (
                low_elevation_coastal_zone_mask.geometry.buffer(0.001)
            )

            # sample the minimum elevation present in the lecz mask
            mask = elevation.rio.clip(
                low_elevation_coastal_zone_mask.geometry,
                low_elevation_coastal_zone_mask.crs,
                all_touched=True,
                drop=False,
            )

            initial_water_levels = float(np.nanmin(mask.values))

            low_elevation_coastal_zone_mask["initial_water_level"] = (
                initial_water_levels
            )
        else:
            self.logger.info(
                "No coastal basins found, skipping setup_coastal_sfincs_model_regions"
            )
            low_elevation_coastal_zone_mask = gpd.GeoDataFrame(
                geometry=[], crs=self.geom["mask"].crs
            )
            low_elevation_coastal_zone_mask["initial_water_level"] = pd.Series(
                dtype=np.float32
            )

        self.set_geom(
            low_elevation_coastal_zone_mask,
            name="coastal/low_elevation_coastal_zone_mask",
        )

    @build_method(
        required=True,
        depends_on=["setup_hydrography", "setup_elevation", "setup_grid_elevation"],
    )
    def setup_waterbodies(
        self,
        mode: Literal["on", "off", "lakes_only", "reservoirs_only"] = "on",
        command_areas: None | str = None,
        calculate_command_areas: bool = False,
        custom_reservoir_capacity: None | str = None,
    ) -> None:
        """Set up waterbodies, reservoirs, and command areas.

        Configure waterbodies and their associated command areas for the model grid.
        This includes rasterizing lake and reservoir identifiers, optionally
        assigning command areas from preset data or by deriving them from the river
        routing network, and updating reservoir storage capacities.

        Args:
            mode: Whether to enable waterbodies ("on"), disable them ("off"),
                or include only lakes ("lakes_only") or only reservoirs ("reservoirs_only").
            command_areas: Identifier of the preset command area data in the
                data catalog. If None, command areas can be calculated from the
                routing network instead.
            calculate_command_areas: Flag or identifier indicating that command areas
                should be calculated based on the routing network. If false, command
                areas are not calculated and default to a missing value.
            custom_reservoir_capacity: Identifier of the custom reservoir capacity
                data in the data catalog. If None, the default reservoir capacities
                from the waterbody dataset are used. The table must have
                ``waterbody_id`` as the index and a ``volume_total`` column defining
                the reservoir capacity.

        Raises:
            ValueError: If the custom_reservoir_capacity file is not a .csv or .xlsx file.
            ValueError: If command_areas and calculate_command_areas are both provided.
            ValueError: If mode is not "on", "off", "lakes_only", or "reservoirs_only".
            ValueError: If command areas are requested but mode is "off" or "lakes_only".
        """
        if mode not in ["on", "off", "lakes_only", "reservoirs_only"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'on', 'off', 'lakes_only', or 'reservoirs_only'."
            )

        if command_areas is not None and calculate_command_areas:
            raise ValueError(
                "Cannot provide both command_areas and calculate_command_areas. Please choose one."
            )

        if (command_areas is not None or calculate_command_areas) and mode in (
            "off",
            "lakes_only",
        ):
            raise ValueError(
                "Command areas cannot be used or calculated when waterbodies are disabled or set to lakes only."
            )

        waterbodies: gpd.GeoDataFrame = self.data_catalog.fetch("hydrolakes").read(
            bbox=self.bounds,
            columns=[
                "waterbody_id",
                "waterbody_type",
                "volume_total",
                "average_discharge",
                "average_area",
                "Wshd_area",
                "geometry",
            ],
        )
        # HydroLAKES catchment area of the lake (Wshd_area, km2). Only used below to
        # keep lakes off river cells they do not drain; dropped before saving.
        waterbodies["catchment_area_m2"] = waterbodies.pop("Wshd_area") * 1e6
        hydrolakes_to_geb: dict[int, np.int32] = {
            1: np.int32(LAKE),
            2: np.int32(RESERVOIR),
            3: np.int32(LAKE_CONTROL),
        }
        assert set(waterbodies["waterbody_type"]).issubset(hydrolakes_to_geb.keys())
        waterbodies["waterbody_type"] = waterbodies["waterbody_type"].map(
            hydrolakes_to_geb
        )
        assert waterbodies["waterbody_type"].dtype == np.int32

        if (
            mode == "off"
        ):  # to keep the dtypes consistent we still load the waterbodies, but we will remove all rows to disable them
            # remove all rows
            waterbodies = waterbodies.iloc[0:0]
        elif mode == "lakes_only":
            waterbodies = waterbodies[waterbodies["waterbody_type"] == LAKE]
        elif mode == "reservoirs_only":
            waterbodies = waterbodies[waterbodies["waterbody_type"] == RESERVOIR]

        # only select waterbodies that intersect with the region
        waterbodies = waterbodies[waterbodies.intersects(self.region.union_all())]

        # drop lakes and reservoirs (incl. run-of-river weirs) flushed within a day;
        # they are represented as river
        waterbodies = waterbodies[
            ~is_flow_through_waterbody(
                volume_total_m3=waterbodies["volume_total"].to_numpy(),
                average_discharge_m3_per_s=waterbodies["average_discharge"].to_numpy(),
            )
        ]

        # Sample lake surface elevation directly from original_d8_elevation at representative points
        if not waterbodies.empty:
            original_d8_elevation: xr.DataArray = self.other[
                "drainage/original_d8_elevation"
            ]
            rep_points = waterbodies.geometry.representative_point()
            xs = xr.DataArray(rep_points.x.values, dims="points")
            ys = xr.DataArray(rep_points.y.values, dims="points")
            waterbodies["elevation"] = original_d8_elevation.sel(
                x=xs, y=ys, method="nearest"
            ).values.astype(np.float32)

        waterbodies["volume_flood"] = waterbodies["volume_total"]

        waterbody_id: xr.DataArray = rasterize_like(
            gdf=waterbodies,
            column="waterbody_id",
            raster=self.grid["mask"],
            nodata=-1,
            dtype=np.int32,
            all_touched=True,
        )

        waterbody_id_array: TwoDArrayInt32 = waterbody_id.values.copy()
        if np.any(waterbody_id_array != -1):
            # a lake keeps only the cells whose upstream area it can drain, so a small
            # lake next to a river does not take over the river cell
            waterbody_id_array = remove_waterbody_cells_outside_catchment(
                waterbody_id=waterbody_id_array,
                upstream_area_m2=self.grid["routing/upstream_area_m2"].values,
                waterbody_ids=waterbodies["waterbody_id"].to_numpy(dtype=np.int32),
                catchment_area_m2=waterbodies["catchment_area_m2"].to_numpy(
                    dtype=np.float32
                ),
            )
            # lakes left without cells (lakes next to a river smaller than a cell) are
            # not represented
            waterbodies = waterbodies[
                waterbodies["waterbody_id"].isin(np.unique(waterbody_id_array))
            ]
            river_ids: TwoDArrayInt32 = self.grid["routing/river_ids"].values
            flow_raster_idxs_ds = self.grid["flow_raster_idxs_ds"].compute()
            flow_raster = FlwdirRaster(
                flow_raster_idxs_ds.values.ravel(),
                shape=flow_raster_idxs_ds.shape,
                transform=np.array(flow_raster_idxs_ds.rio.transform(recalc=True)),
                ftype="d8",
                latlon=True,
            )
            upstream_area_n_cells: TwoDArrayInt32 = self.grid[
                "routing/upstream_area_n_cells"
            ].values
            elevation_min_m: TwoDArrayFloat32 = self.grid[
                "landsurface/elevation_min_m"
            ].values

            waterbody_outflows: TwoDArrayInt32 = identify_waterbody_outflows(
                waterbody_id=waterbody_id_array,
                upstream_area_n_cells=upstream_area_n_cells,
                elevation_min_m=elevation_min_m,
            )
            waterbody_id_array = absorb_trapped_river_cells(
                waterbody_id=waterbody_id_array,
                waterbody_outflows=waterbody_outflows,
                river_ids=river_ids,
                flow_raster=flow_raster,
            )
            waterbody_outflows = identify_waterbody_outflows(
                waterbody_id=waterbody_id_array,
                upstream_area_n_cells=upstream_area_n_cells,
                elevation_min_m=elevation_min_m,
            )
        else:
            waterbody_outflows = np.full_like(waterbody_id_array, -1, dtype=np.int32)

        waterbody_id.data = waterbody_id_array
        self.set_grid(waterbody_id, name="waterbodies/waterbody_id")

        waterbody_outflow_points: xr.DataArray = self.full_like(
            self.grid["mask"],
            fill_value=-1,
            nodata=-1,
            dtype=np.int32,
        )
        waterbody_outflow_points.data = waterbody_outflows
        self.set_grid(
            waterbody_outflow_points, name="waterbodies/waterbody_outflow_points"
        )

        if mode in ("on", "reservoirs_only") and command_areas:
            command_areas_gdf: gpd.GeoDataFrame = gpd.read_file(
                command_areas, mask=self.region
            )
            assert isinstance(command_areas_gdf, gpd.GeoDataFrame)
            command_areas_gdf = command_areas_gdf[
                ~command_areas_gdf["waterbody_id"].isnull()
            ].reset_index(drop=True)
            command_areas_gdf["waterbody_id"] = command_areas_gdf[
                "waterbody_id"
            ].astype(np.int32)

            # Dissolve command areas with same reservoir
            command_areas_gdf = command_areas_gdf.dissolve(
                by="waterbody_id", as_index=False
            )

            # Set lakes with command area to reservoirs and reservoirs without command area to lakes
            ids_with_command: set[int] = set(command_areas_gdf["waterbody_id"])
            waterbodies.loc[
                waterbodies["waterbody_id"].isin(ids_with_command),
                "waterbody_type",
            ] = RESERVOIR

            # Lastly remove command areas that have no associated water body
            reservoir_ids: set[int] = set(
                waterbodies.loc[
                    waterbodies["waterbody_type"] == RESERVOIR, "waterbody_id"
                ]
            )
            command_areas_dissolved = command_areas_gdf[
                command_areas_gdf["waterbody_id"].isin(reservoir_ids)
            ].reset_index(drop=True)

            self.set_geom(command_areas_dissolved, name="waterbodies/command_areas")

            assert command_areas_dissolved["waterbody_id"].isin(reservoir_ids).all()

            command_area_raster = rasterize_like(
                gdf=command_areas_dissolved,
                column="waterbody_id",
                raster=self.grid["mask"],
                nodata=-1,
                dtype=np.int32,
                all_touched=True,
            )

            subcommand_area_raster = rasterize_like(
                gdf=command_areas_dissolved,
                column="waterbody_id",
                raster=self.subgrid["mask"],
                nodata=-1,
                dtype=np.int32,
                all_touched=True,
            )

        elif mode in ("on", "reservoirs_only") and calculate_command_areas:
            river_graph: DiGraph = get_river_graph(self.data_catalog)

            mapped_reservoirs_grid, mapped_reservoirs_subgrid = (
                derive_command_areas_from_routing(
                    waterbodies=waterbodies,
                    rivers=self.geom["routing/rivers"],
                    basin_ids=self.grid["routing/basin_ids"],
                    grid_mask=self.grid["mask"],
                    subgrid_mask=self.subgrid["mask"],
                    subgrid_factor=self.subgrid_factor,
                    river_graph=river_graph,
                )
            )
            command_area_raster = self.full_like(
                self.grid["mask"],
                fill_value=-1,
                nodata=-1,
                dtype=np.int32,
            )
            command_area_raster.data = mapped_reservoirs_grid

            subcommand_area_raster = self.full_like(
                self.subgrid["mask"],
                fill_value=-1,
                nodata=-1,
                dtype=np.int32,
            )
            subcommand_area_raster.data = mapped_reservoirs_subgrid
        else:
            command_area_raster: xr.DataArray = self.full_like(
                self.grid["mask"],
                fill_value=-1,
                nodata=-1,
                dtype=np.int32,
            )
            subcommand_area_raster = self.full_like(
                self.subgrid["mask"],
                fill_value=-1,
                nodata=-1,
                dtype=np.int32,
            )

        self.set_grid(command_area_raster, name="waterbodies/command_area")
        self.set_subgrid(subcommand_area_raster, name="waterbodies/subcommand_areas")

        if mode in ("on", "reservoirs_only") and custom_reservoir_capacity:
            if custom_reservoir_capacity.endswith(".xlsx"):
                custom_reservoir_capacity_df: pd.DataFrame = pd.read_excel(
                    custom_reservoir_capacity
                )
            elif custom_reservoir_capacity.endswith(".csv"):
                custom_reservoir_capacity_df: pd.DataFrame = pd.read_csv(
                    custom_reservoir_capacity
                )
            else:
                raise ValueError(
                    "custom_reservoir_capacity must be a .csv or .xlsx file"
                )

            custom_reservoir_capacity_df = custom_reservoir_capacity_df[
                custom_reservoir_capacity_df.index != -1
            ]

            waterbodies = waterbodies.set_index("waterbody_id")
            waterbodies.update(custom_reservoir_capacity_df)
            waterbodies = waterbodies.reset_index()

        assert "waterbody_id" in waterbodies.columns, "waterbody_id is required"
        assert "waterbody_type" in waterbodies.columns, "waterbody_type is required"
        assert "volume_total" in waterbodies.columns, "volume_total is required"
        assert "average_discharge" in waterbodies.columns, (
            "average_discharge is required"
        )
        assert "average_area" in waterbodies.columns, "average_area is required"
        waterbodies = waterbodies.drop(columns="catchment_area_m2")
        self.set_geom(waterbodies, name="waterbodies/waterbody_data")

    def setup_total_water_levels(self) -> None:
        """Sets up the coastal water levels for within the model bounds."""
        water_level_data_region, stations = self.data_catalog.fetch(
            "gtsm_timeseries", variable="total_water_level"
        ).read(bounds=self.bounds, variable="total_water_level")
        self.set_other(
            water_level_data_region,
            name="water_levels/total_water_levels",
            filters=gtsm_filters,
            shards={"stations": 100},
        )
        self.set_geom(stations, name="water_levels/stations")
        self.logger.info("Water level stations and total water levels set")

    def setup_surge_levels(self) -> None:
        """Sets up the storm surge residual hydrographs for station within the model bounds."""
        surge_data_region, _ = self.data_catalog.fetch(
            "gtsm_timeseries", variable="storm_surge_residual"
        ).read(bounds=self.bounds, variable="storm_surge_residual")
        self.set_other(
            surge_data_region,
            name="water_levels/surge",
            filters=gtsm_filters,
            shards={"stations": 100},
        )
        self.logger.info("Water level surge levels set")

    def setup_coast_rp(self) -> None:
        """Sets up the coastal return period data for the model."""
        self.logger.info("Setting up coastal return period data")

        coast_rp = self.data_catalog.fetch("coast_rp").read()

        stations = gpd.read_parquet(
            os.path.join("input", self.files["geom"]["water_levels/stations"])
        )
        # remove stations that are not in coast_rp index
        stations = stations[stations.index.astype(int).isin(coast_rp.index)]

        coast_rp = coast_rp.loc[stations.index]
        self.set_table(coast_rp, name="coast_rp")

        # also set stations (only those that are in coast_rp)
        self.set_geom(stations, "water_levels/stations_coast_rp")

    def setup_sea_level_projections(self, projection_quantile: float) -> None:
        """Sets up the AR6 sea level projections for the model.

        For each water level station the nearest sea level projection location (great-circle distance) is
        selected. Its projected sea level change is expressed relative to 2020 and
        interpolated (and linearly extrapolated) to annual values for 2000-2100.

        Args:
            projection_quantile: AR6 quantile to select (fraction between 0 and 1). Default is 0.5 (median projection).
        """
        # get the scenario from the model configuration
        scenario = self.ssp_rcp
        self.logger.info(
            f"Setting up sea level rise data for scenario {scenario}, using {projection_quantile} quantile of AR6 projections"
        )
        sea_level_change_m: xr.Dataset = self.data_catalog.fetch("ar6").read(
            scenario, projection_quantile
        )

        stations: gpd.GeoDataFrame = gpd.read_parquet(
            os.path.join("input", self.files["geom"]["water_levels/stations"])
        )

        lon_deg: npt.NDArray[np.floating] = sea_level_change_m["lon"].values
        lat_deg: npt.NDArray[np.floating] = sea_level_change_m["lat"].values
        years: npt.NDArray[np.int64] = sea_level_change_m["years"].values

        def to_unit_vectors(
            lon_deg: npt.NDArray[np.floating], lat_deg: npt.NDArray[np.floating]
        ) -> npt.NDArray[np.floating]:
            """Convert longitude and latitude in degrees to 3D unit vectors.

            Args:
                lon_deg: Longitude in degrees.
                lat_deg: Latitude in degrees.
            Returns:
                3D unit vectors corresponding to the input coordinates.
            """
            lon_rad: npt.NDArray[np.floating] = np.radians(lon_deg)
            lat_rad: npt.NDArray[np.floating] = np.radians(lat_deg)
            return np.column_stack(
                (
                    np.cos(lat_rad) * np.cos(lon_rad),
                    np.cos(lat_rad) * np.sin(lon_rad),
                    np.sin(lat_rad),
                )
            )

        # Euclidean distance between points on the unit sphere increases monotonically
        # with great-circle distance, so a KD-tree on 3D unit vectors gives the same
        # nearest neighbour as the Haversine distance, without an
        # (n_stations x n_locations) distance matrix.
        tree: KDTree = KDTree(
            to_unit_vectors(
                lon_deg,
                lat_deg,
            )
        )
        _, nearest_valid_index = tree.query(
            to_unit_vectors(
                stations.geometry.x.to_numpy(),
                stations.geometry.y.to_numpy(),
            )
        )
        nearest_location_indices: npt.NDArray[np.intp] = nearest_valid_index

        # Load only the nearest locations (stations may share a location)
        unique_location_indices: npt.NDArray[np.intp]
        inverse_indices: npt.NDArray[np.intp]
        unique_location_indices, inverse_indices = np.unique(
            nearest_location_indices, return_inverse=True
        )

        sea_level_change_m_transposed: npt.NDArray[np.floating] = (
            sea_level_change_m["sea_level_change"]
            .transpose("locations", "years")
            .isel(locations=unique_location_indices)
            .values[inverse_indices]
        )

        unique_location_indices: npt.NDArray[np.intp]
        inverse_indices: npt.NDArray[np.intp]
        unique_location_indices, inverse_indices = np.unique(
            nearest_location_indices, return_inverse=True
        )

        target_years: npt.NDArray[np.int64] = np.arange(2000, 2101)
        interpolated_values: npt.NDArray[np.floating] = interpolate.interp1d(
            x=years,
            y=sea_level_change_m_transposed,
            axis=1,
            fill_value="extrapolate",
            bounds_error=False,
        )(target_years)

        sea_level_rise_stations: pd.DataFrame = pd.DataFrame(
            data=interpolated_values.T,
            index=pd.to_datetime([f"{year}-01-01" for year in target_years]),
            columns=stations.index,
        )
        self.set_table(sea_level_rise_stations, name="water_levels/sea_level_rise")

    @build_method(depends_on=["set_ssp"], required=True)
    def setup_coastal_water_levels(self, projection_quantile: float = 0.5) -> None:
        """Set up water levels and station data for the model.

        Args:
            projection_quantile: Sea level rise projection quantile as a fraction between 0 and 1. Default is 0.5 (median projection).
        """
        if not self.geom["routing/subbasins"]["is_coastal"].any():
            self.logger.info(
                "No coastal basins found, skipping coastal water levels setup"
            )
            return
        # Continue with coastal water levels setup
        self.setup_total_water_levels()
        self.setup_surge_levels()
        self.setup_coast_rp()
        self.setup_sea_level_projections(projection_quantile)

    @build_method(required=False)
    def setup_inflow(
        self,
        locations: str,
        inflow_m3_per_s: str,
        interpolate: bool = False,
        extrapolate: bool = False,
    ) -> None:
        """Sets up a inflow hydrograph for the model.

        Args:
            locations: A vector file that can be read by geopandas containing the inflow location,
                with column ID (str) and a point geometry.
            inflow_m3_per_s: A CSV file containing the inflow hydrograph in m3/s with a datetime index (date),
                and a column for each inflow location ID matching the IDs (str) in the locations file.
            interpolate: Whether to interpolate missing values in the inflow hydrograph.
            extrapolate: Whether to extrapolate missing values in the inflow hydrograph.

        Raises:
            ValueError: If the inflow location is outside of the model grid or study area.
        """
        mask = self.grid["mask"].compute()
        transform = mask.rio.transform()

        inflow_locations = gpd.read_file(locations)

        if inflow_locations.crs is None:
            raise ValueError("Locations file must have a defined CRS.")
        inflow_locations = inflow_locations.to_crs(mask.rio.crs)

        if "ID" not in inflow_locations.columns:
            raise ValueError("Locations file must have an 'ID' column.")

        for ID in inflow_locations["ID"]:
            if not isinstance(ID, str):
                raise ValueError("Locations 'ID' column must be of type string.")

        y, x = rasterio.transform.rowcol(
            transform, inflow_locations.geometry.x, inflow_locations.geometry.y
        )

        inflow_locations["y"] = y
        inflow_locations["x"] = x

        self.set_geom(inflow_locations, name="routing/inflow_locations")

        if (
            (inflow_locations["y"] < 0)
            | (inflow_locations["y"] >= mask.shape[0])
            | (inflow_locations["x"] < 0)
            | (inflow_locations["x"] >= mask.shape[1])
        ).any():
            raise ValueError("Inflow location is outside of the model grid.")

        # check if any of the locations is outside of the study area (mask is True outside the study area)
        if (mask.values[inflow_locations["y"], inflow_locations["x"]]).any():
            raise ValueError("Inflow location is outside of the study area.")

        inflow_df_m3_per_s = pd.read_csv(inflow_m3_per_s, index_col=0, parse_dates=True)
        # ensure index is datetime
        if not np.issubdtype(inflow_df_m3_per_s.index.dtype, np.datetime64):  # ty:ignore[invalid-argument-type]
            raise ValueError("Inflow hydrograph index must be datetime.")

        # ensure columns are strings
        if not all(isinstance(col, str) for col in inflow_df_m3_per_s.columns):
            raise ValueError("Inflow hydrograph columns must be of type string.")

        # check if all IDs in locations are in the inflow file
        missing_ids: set[str] = set(inflow_locations["ID"]) - set(
            inflow_df_m3_per_s.columns
        )
        if missing_ids:
            raise ValueError(
                f"Missing inflow data for location IDs: {missing_ids}. "
                "Ensure column names in CSV match location IDs."
            )

        # subset to only include the locations that are in the locations file
        inflow_df_m3_per_s = inflow_df_m3_per_s[inflow_locations["ID"]]

        # align to model time step
        model_time_step = pd.date_range(
            self.start_date,
            end=self.end_date + timedelta(hours=23),
            freq="h",
            inclusive="both",
        )

        inflow_df_m3_per_s: pd.DataFrame = inflow_df_m3_per_s.reindex(model_time_step)

        if interpolate:
            # interpolate missing values by first interpolating in time
            inflow_df_m3_per_s: pd.DataFrame = inflow_df_m3_per_s.interpolate(
                method="time"
            )
        if extrapolate:
            # extrapolate missing values by forward and backward filling
            inflow_df_m3_per_s: pd.DataFrame = inflow_df_m3_per_s.ffill().bfill()

        if inflow_df_m3_per_s.isnull().any():
            raise ValueError(
                "Inflow hydrograph contains missing values. "
                "Set interpolate=True to interpolate missing values. "
                "or fill missing values in the inflow hydrograph. "
                "model start and end date: "
                f"{self.start_date} to {self.end_date} "
                "with hourly frequency."
            )

        self.set_table(inflow_df_m3_per_s, name="routing/inflow_m3_per_s")

    @build_method(required=True, depends_on=["setup_hydrography", "setup_waterbodies"])
    def setup_retention_basins(
        self,
        retention_basins: Path | None = None,
        max_storage_m3_to_min_shreve_stream_order_function: str | None = None,
        create_plots: bool = False,
        dominant_river_search_radius_m: float = 2000.0,
    ) -> None:
        """Setup retention basins.

        There can only be a maximum of one retention basin per river pixel. When a retention
        basin is already occupied or falls inside a lake/reservoir, the retention basin being set up
        will be assigned to the next downstream river pixel that is not occupied and not in a waterbody.
        Within dominant_river_search_radius_m, basins prioritize the dominant river (largest upstream area).

        Args:
            retention_basins: A vector file that can be read by geopandas containing the retention basins,
                with a point geometry representing the location of the retention basin. If none (default),
                no retention basins will be set up in the model.
            max_storage_m3_to_min_shreve_stream_order_function: A string representing a function that takes the maximum storage of a retention basin in m3 and
                returns the minimum Shreve stream order of the river pixel to which the retention basin can be snapped. If none, all retention basins will be snapped
                to the nearest river pixel regardless of its Shreve stream order.
            create_plots: Whether to create plots of the retention basins and their snapped locations on the river network.
            dominant_river_search_radius_m: Radius in meters within which to search for the dominant river (defaults to 2000m).

        Raises:
            ValueError: If a retention basin cannot be snapped to the river network.
            ValueError: If multiple retention basins snap to the same river pixel, which is not allowed.

        Sets:
            routing/retention_basin_ids: A grid with the same dimensions as the model grid, where each pixel that is
            part of a retention basin has the ID of the basin it is associated with, and pixels that are not part of a
            retention basin have a value of -1.
            routing/retention_basin_data: A table containing the ID of each retention basin and other data (e.g., is_controlled).
        """
        # create a grid with the same dimensions as the model grid to store the retention basin IDs
        retention_basin_ids = full_like(
            self.grid["mask"], fill_value=-1, nodata=-1, dtype=np.int32
        ).compute()

        if retention_basins is None:
            self.set_grid(retention_basin_ids, name="routing/retention_basin_ids")
            retention_basin_gdf = gpd.GeoDataFrame(
                columns=np.array(
                    ["ID", "retention_max_storage_m3", "controlled_retention", "active"]
                ),
                geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            ).astype(
                {
                    "ID": np.int32,
                    "retention_max_storage_m3": np.float32,
                    "controlled_retention": bool,
                    "active": bool,
                }
            )
            self.set_geom(retention_basin_gdf, name="routing/retention_basins")
        else:
            if max_storage_m3_to_min_shreve_stream_order_function is not None:
                max_storage_m3_to_min_shreve_stream_order_function: CodeType = compile(
                    max_storage_m3_to_min_shreve_stream_order_function,
                    "<string>",
                    "eval",
                )

            # read the retention basins from the provided file
            retention_basins = gpd.read_file(retention_basins).to_crs(
                self.grid["mask"].rio.crs
            )

            # Replace empty or missing values in 'controlled_retention' with "uncontrolled"
            retention_basins["controlled_retention"] = (
                retention_basins["controlled_retention"]
                .fillna("uncontrolled")
                .replace("", "uncontrolled")
            )

            rivers = self.geom["routing/rivers"]

            upstream_area_grid = self.grid["routing/upstream_area_m2"].compute()
            upstream_area_subgrid = self.other[
                "drainage/original_d8_upstream_area_m2"
            ].compute()

            waterbody_id_grid: xr.DataArray = self.grid[
                "waterbodies/waterbody_id"
            ].compute()

            wb_values: np.ndarray = waterbody_id_grid.values

            has_river_ids = "routing/river_ids" in self.grid
            if has_river_ids:
                river_ids_grid = self.grid["routing/river_ids"].compute().values
            else:
                river_ids_grid = None

            retention_basin_data = []

            ldd = self.grid["routing/ldd"]
            flow_direction_raster = pyflwdir.from_array(
                ldd.values,
                ftype="ldd",
                transform=np.array(ldd.rio.transform(recalc=True)),
                latlon=True,
            )
            downstream_indices = flow_direction_raster.idxs_ds.reshape(ldd.shape)

            for _, retention_basin in retention_basins.iterrows():
                centroid = retention_basin.geometry.centroid

                if np.isnan(retention_basin["retention_max_storage_m3"]):
                    warnings.warn(
                        f"Retention basin ID {retention_basin['ID']} has missing retention_max_storage_m3 value. Setting it to 0.0."
                    )
                    retention_basin["retention_max_storage_m3"] = 0.0

                # the centroid of the retention basin may not fall exactly on the river network,
                # so we snap it to the river network, prioritizing the dominant river within 2 km
                if max_storage_m3_to_min_shreve_stream_order_function is not None:
                    rivers_with_min_shreve_order = rivers[
                        rivers["shreve_stream_order"]
                        >= eval(
                            max_storage_m3_to_min_shreve_stream_order_function,
                            {"x": retention_basin["retention_max_storage_m3"]},
                        )
                    ]
                else:
                    rivers_with_min_shreve_order = rivers

                snapped_data: SnappingResults | None = snap_point_to_river_network(
                    point=centroid,
                    rivers=rivers_with_min_shreve_order,
                    upstream_area_grid=upstream_area_grid,
                    upstream_area_subgrid=upstream_area_subgrid,
                    waterbody_id_grid=waterbody_id_grid,
                    max_spatial_difference_degrees=180,
                    dominant_river_buffer_m=dominant_river_search_radius_m,
                )
                if snapped_data is None:
                    raise ValueError(
                        f"Could not snap retention basin for basin ID {retention_basin['ID']} to river network. "
                    )
                snapped_grid_pixel_xy = snapped_data.snapped_grid_pixel_xy

                def is_invalid_cell(col: int, row: int) -> bool:
                    if retention_basin_ids.values[row, col] != -1:
                        return True
                    wb_val = wb_values[row, col]
                    if wb_val != -1 and not np.isnan(wb_val):
                        return True
                    if river_ids_grid is not None:
                        if river_ids_grid[row, col] == -1:
                            return True
                    return False

                # assign the ID of the retention basin to the corresponding pixel in the retention_basin_ids grid
                search_steps = 0
                while is_invalid_cell(
                    snapped_grid_pixel_xy[0], snapped_grid_pixel_xy[1]
                ):
                    assigned_retention_basin_id = retention_basin_ids.values[
                        snapped_grid_pixel_xy[1], snapped_grid_pixel_xy[0]
                    ]
                    downstream_index = downstream_indices[
                        snapped_grid_pixel_xy[1], snapped_grid_pixel_xy[0]
                    ]
                    if downstream_index == -1:
                        raise ValueError(
                            f"Retention basin ID {retention_basin['ID']} at location {centroid} snaps to an occupied or invalid (waterbody/non-river) pixel. No valid unassigned downstream river pixel was found because downstream index is -1."
                        )

                    # Follow the downstream river path until we find an unassigned pixel.
                    downstream_row, downstream_col = np.unravel_index(
                        downstream_index, retention_basin_ids.shape
                    )
                    self.logger.warning(
                        f"Retention basin ID {retention_basin['ID']} snapped to an occupied or waterbody pixel (assigned RB: {assigned_retention_basin_id}); searching downstream at step {search_steps + 1} from ({snapped_grid_pixel_xy[0]}, {snapped_grid_pixel_xy[1]}) to ({downstream_col}, {downstream_row})."
                    )
                    snapped_grid_pixel_xy = np.array([downstream_col, downstream_row])
                    search_steps += 1

                retention_basin_ids.values[
                    snapped_grid_pixel_xy[1], snapped_grid_pixel_xy[0]
                ] = retention_basin["ID"]

                final_lon = upstream_area_grid.x.values[snapped_grid_pixel_xy[0]].item()
                final_lat = upstream_area_grid.y.values[snapped_grid_pixel_xy[1]].item()
                snapped_geom = shapely.geometry.Point(final_lon, final_lat)

                # store the retention basin data in a list to create a table later
                controlled_flag = (
                    str(retention_basin["controlled_retention"]).lower() == "controlled"
                )
                retention_basin_data.append(
                    {
                        "ID": retention_basin["ID"],
                        "retention_max_storage_m3": retention_basin[
                            "retention_max_storage_m3"
                        ],
                        "controlled_retention": controlled_flag,
                        "active": retention_basin["active"],
                        "river_segment_id": snapped_data.closest_river_segment.ID,
                        "river_segment_shreve_order": snapped_data.closest_river_segment.shreve_stream_order,
                        "geometry": snapped_geom,
                    }
                )

            retention_basin_gdf = gpd.GeoDataFrame(
                retention_basin_data, crs="EPSG:4326"
            )

        self.set_geom(retention_basin_gdf, name="routing/retention_basins")
        self.set_grid(retention_basin_ids, name="routing/retention_basin_ids")

        if create_plots:
            import matplotlib.pyplot as plt

            # plot maximum storage volume vs shreve order
            plt.figure(figsize=(10, 6))
            plt.scatter(
                retention_basin_gdf["river_segment_shreve_order"],
                retention_basin_gdf["retention_max_storage_m3"],
                alpha=0.7,
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Shreve Order of River Segment")
            plt.ylabel("Maximum Storage Volume (m³)")
            plt.title("Retention Basin Maximum Storage Volume vs Shreve Order")
            plt.grid(True, which="both", ls="--", lw=0.5)
            plt.savefig(
                self.report_dir / "retention_basins_shreve_order_vs_storage.svg"
            )
            plt.close()
