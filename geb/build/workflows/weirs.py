"""Place GDW and AMBER barriers on the river network."""

import logging
import sys
from collections.abc import Hashable
from time import monotonic

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
import xarray as xr
from pyproj import Geod, Proj
from shapely.geometry import Point, Polygon, box
from tqdm import tqdm

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
    amber_points: gpd.GeoDataFrame | None = None,
    logger: logging.Logger | None = None,
) -> tuple[xr.DataArray, gpd.GeoDataFrame]:
    """Place fixed barrier heights on the river grid.

    Show a progress bar in terminals and periodic progress in batch logs.
    Log inclusion counts and exclusion reasons once after placement; retain
    individual barrier diagnostics in the returned records.
    If the closest cell already has a weir, try only the second-closest
    distinct cell on the same river segment before excluding the barrier.

    Args:
        gdw_points: GDW points and their linked lake or reservoir IDs.
        crest_height_m: Missing-height override (m). None uses type defaults: bankfull for AMBER weirs/sluices/locks,
            half bankfull for fords/ramps and unknown types, and GDW defaults.
        waterbody_id: Existing waterbody grid; -1 means no lake or reservoir.
        rivers: Rivers used to place the points on the grid.
        upstream_area_grid: Model-grid drainage area (m2).
        upstream_area_subgrid: Original drainage area (m2), retained for caller
            compatibility. Weir placement does not read this diagnostic raster.
        valid_river_cells: True where a weir can be placed.
        amber_points: AMBER barriers, kept only within 250 m of a river.
        logger: Build logger for progress and summaries. None uses the module logger.

    Returns:
        Height grid and barrier records at their original locations.
        Records show source heights (m), river distances (m) and exclusion reasons.
        Positive grid heights are in meters, zero means
        no structure, -1 means bankfull depth + 1 m, -2 means full
        bankfull depth, and -3 means half bankfull depth.
        AMBER weirs and unknown types retain known heights. Sluices/locks use
        bankfull depth; fords/ramps use half bankfull depth. AMBER dams and
        culverts are excluded from river routing. All river structures are fixed.
        Bankfull markers are resolved when routing initializes; crests remain fixed.

    Raises:
        ValueError: If the height override is invalid.
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
    barriers: gpd.GeoDataFrame = gdw_points.copy()
    if not barriers.empty:
        barriers = barriers.to_crs(4326)
        barriers["barrier_id"] = barriers["gdw_id"].astype(str)
        barriers["source"] = "GDW"
    if amber_points is not None and not amber_points.empty:
        amber_points = amber_points.to_crs(4326).copy()
        amber_points["barrier_id"] = amber_points["amber_id"]
        amber_points["source"] = "AMBER"
        amber_points["waterbody_id"] = np.nan
        # Process GDW first so AMBER cannot replace an existing structure.
        barriers = gpd.GeoDataFrame(
            pd.concat([barriers, amber_points], ignore_index=True), crs=4326
        )
    barriers.reset_index(drop=True, inplace=True)
    barrier_records: gpd.GeoDataFrame = gpd.GeoDataFrame(
        barriers.reindex(
            columns=["source", "barrier_id", "dam_type", "dam_hgt_m", "geometry"]
        ).rename(columns={"dam_type": "barrier_type", "dam_hgt_m": "height_m"}),
        geometry="geometry",
        crs=4326,
    )
    barrier_records["source"] = barrier_records["source"].astype(str)
    barrier_records["barrier_id"] = barrier_records["barrier_id"].astype(str)
    barrier_records["barrier_type"] = barrier_records["barrier_type"].astype("string")
    barrier_records["height_m"] = pd.to_numeric(
        barrier_records["height_m"], errors="coerce"
    )
    barrier_records["height_m"] = barrier_records["height_m"].where(
        np.isfinite(barrier_records["height_m"]) & (barrier_records["height_m"] > 0)
    )
    barrier_records["distance_to_river_m"] = np.nan
    barrier_records["included"] = False
    barrier_records["exclusion_reason"] = ""
    if barriers.empty:
        return weir_height_grid, barrier_records
    represented_rivers: gpd.GeoDataFrame = rivers.loc[
        rivers["represented_in_grid"]
    ].to_crs(4326)
    # Batch the existing conservative search boxes; exact meter checks still
    # decide inclusion, and rejected barriers remain in the diagnostic output.
    amber_search_candidates: npt.NDArray[np.bool_] = np.ones(len(barriers), dtype=bool)
    amber_positions: npt.NDArray[np.int64] = np.flatnonzero(
        barriers["source"].eq("AMBER")
    )
    if amber_positions.size:
        points: gpd.GeoSeries = barriers.geometry.iloc[amber_positions]
        longitude_buffers: npt.NDArray[np.float64] = 0.01 / np.maximum(
            np.cos(np.deg2rad(points.y.to_numpy())), 0.01
        )
        search_boxes: npt.NDArray = shapely.box(
            points.x.to_numpy() - longitude_buffers,
            points.y.to_numpy() - 0.01,
            points.x.to_numpy() + longitude_buffers,
            points.y.to_numpy() + 0.01,
        )
        matches: npt.NDArray[np.int64] = represented_rivers.sindex.query(search_boxes)
        amber_search_candidates[amber_positions] = False
        amber_search_candidates[amber_positions[np.unique(matches[0])]] = True
    river_coordinate_cache: dict[int, npt.NDArray[np.float64]] = {}
    geod: Geod = Geod(ellps="WGS84")
    if logger is None:
        logger = logging.getLogger(__name__)
    interactive_progress: bool = sys.stderr.isatty()
    last_progress_time: float = monotonic()
    logger.info("Snapping %s barriers to the river network.", len(barriers))
    barrier: pd.Series
    barrier_index: int
    for barrier_index, (_, barrier) in enumerate(
        tqdm(
            barriers.iterrows(),
            total=len(barriers),
            desc="Snapping barriers",
            unit="barrier",
            mininterval=1.0,
            disable=not interactive_progress,
        )
    ):
        progress_time: float = monotonic()
        # Batch logs have no terminal to redraw a bar; limit updates to one
        # readable line every 30 seconds instead of accumulating escape sequences.
        if not interactive_progress and progress_time - last_progress_time >= 30.0:
            logger.info(
                "Snapping barriers: %s/%s (%.0f%%).",
                barrier_index,
                len(barriers),
                100.0 * barrier_index / len(barriers),
            )
            last_progress_time = progress_time
        if barrier.source == "GDW" and barrier.inside_gdw_polygon:
            barrier_records.at[barrier_index, "exclusion_reason"] = (
                "inside reservoir outline"
            )
            continue
        if pd.notna(barrier.waterbody_id):
            barrier_records.at[barrier_index, "exclusion_reason"] = (
                "part of lake or reservoir"
            )
            continue
        dam_type: str = str(barrier.get("dam_type", "")).strip()
        if barrier.source == "AMBER" and dam_type in {
            "Dam",
            "Lake Control Dam",
            "Culvert",
        }:
            # Dams only inform waterbody classification; culverts need an opening
            # model rather than a solid river crest.
            barrier_records.at[barrier_index, "exclusion_reason"] = (
                "AMBER dam classification only"
                if dam_type != "Culvert"
                else "AMBER culvert excluded"
            )
            continue
        nearby_rivers: gpd.GeoDataFrame = rivers
        if barrier.source == "AMBER":
            if not amber_search_candidates[barrier_index]:
                barrier_records.at[barrier_index, "exclusion_reason"] = (
                    "no river within 250 m"
                )
                continue
            distance_m: float
            nearby_rivers, distance_m = nearest_amber_river(
                barrier.geometry, represented_rivers
            )
            barrier_records.at[barrier_index, "distance_to_river_m"] = distance_m
            if nearby_rivers.empty:
                barrier_records.at[barrier_index, "exclusion_reason"] = (
                    "farther than 250 m"
                    if np.isfinite(distance_m)
                    else "no river within 250 m"
                )
                continue
        river_location: SnappingResults | None = snap_point_to_river_network(
            point=barrier.geometry,
            rivers=nearby_rivers,
            upstream_area_grid=upstream_area_grid,
            upstream_area_subgrid=None,
            river_coordinate_cache=river_coordinate_cache,
        )
        if river_location is None:
            barrier_records.at[barrier_index, "exclusion_reason"] = "no river found"
            continue
        # Use the returned river point to check the final snapping distance.
        distance_m = geod.inv(
            barrier.geometry.x,
            barrier.geometry.y,
            *river_location.closest_point_coords,
        )[2]
        barrier_records.at[barrier_index, "distance_to_river_m"] = distance_m
        if barrier.source == "AMBER" and distance_m > 250.0:
            barrier_records.at[barrier_index, "exclusion_reason"] = "farther than 250 m"
            continue
        column: int
        row: int
        column, row = river_location.snapped_grid_pixel_xy
        if weir_height_grid.values[row, column] != 0:
            river_cells: npt.NDArray[np.int64] = np.stack(
                river_location.closest_river_segment.hydrography_xy
            ).astype(np.int64)
            alternative_positions: npt.NDArray[np.int64] = np.flatnonzero(
                np.any(river_cells != (column, row), axis=1)
            )
            if alternative_positions.size:
                # Keep the same river and distance ranking as the initial snap;
                # limiting the search avoids moving barriers far along it.
                cell_coordinates: npt.NDArray[np.float64] = river_coordinate_cache[
                    river_location.closest_river_segment.ID
                ][alternative_positions]
                offsets: npt.NDArray[np.float64] = (
                    cell_coordinates - river_location.closest_point_coords
                )
                second_position: int = int(
                    alternative_positions[
                        np.argmin(np.hypot(offsets[:, 0], offsets[:, 1]))
                    ]
                )
                column, row = map(int, river_cells[second_position])

        if (
            not valid_river_cells.values[row, column]
            or waterbody_id.values[row, column] != -1
            or weir_height_grid.values[row, column] != 0
        ):
            if waterbody_id.values[row, column] != -1:
                barrier_records.at[barrier_index, "exclusion_reason"] = (
                    "part of lake or reservoir"
                )
            elif weir_height_grid.values[row, column] != 0:
                barrier_records.at[barrier_index, "exclusion_reason"] = (
                    "cell already used"
                )
            else:
                barrier_records.at[barrier_index, "exclusion_reason"] = (
                    "unsuitable river cell"
                )
            continue
        barrier_height_m: float = barrier.dam_hgt_m
        height_m: float = 0.0
        if barrier.source == "AMBER" and dam_type in {"Sluice", "Lock", "Ford", "Ramp"}:
            # These types use the requested common river-depth proxies, not
            # atlas heights that may describe the whole structure.
            height_m = (
                crest_height_m
                if crest_height_m is not None
                else -2.0
                if dam_type in {"Sluice", "Lock"}
                else -3.0
            )
        elif (
            pd.notna(barrier_height_m)
            and np.isfinite(barrier_height_m)
            and barrier_height_m > 0
        ):
            height_m = float(barrier_height_m)
        elif crest_height_m is not None:
            height_m = crest_height_m
        elif barrier.source == "AMBER" and dam_type != "Weir":
            height_m = -3.0  # Unknown types use a lower fallback crest.
        elif barrier.source == "GDW" and dam_type in {"Dam", "Lake Control Dam"}:
            height_m = -1.0  # Resolve to bankfull depth + 1 m when routing starts.
        else:
            height_m = -2.0  # Resolve to full bankfull depth when routing starts.
        weir_height_grid.values[row, column] = height_m
        barrier_records.at[barrier_index, "included"] = True
    source: Hashable
    source_records: pd.DataFrame
    for source, source_records in barrier_records.groupby("source", sort=True):
        included_count: int = int(source_records["included"].sum())
        logger.info(
            "%s barriers: %s included, %s excluded (%s total).",
            source,
            included_count,
            len(source_records) - included_count,
            len(source_records),
        )
        exclusion_counts: pd.Series = source_records.loc[
            ~source_records["included"], "exclusion_reason"
        ].value_counts()
        exclusion_reason: Hashable
        excluded_count: int
        for exclusion_reason, excluded_count in exclusion_counts.items():
            logger.info(
                "%s excluded: %s — %s.", source, excluded_count, exclusion_reason
            )
    return weir_height_grid, barrier_records


def nearest_amber_river(
    point: Point, rivers: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, float]:
    """Find the nearest represented river within 250 m of an AMBER point.

    Args:
        point: Barrier location in WGS84 degrees.
        rivers: Represented river lines in EPSG:4326.

    Returns:
        One river, or an empty frame beyond 250 m, plus its distance (m).
        Distance is NaN when the search finds no nearby river.

    Raises:
        ValueError: If the rivers are not in EPSG:4326.
    """
    if rivers.crs is None or rivers.crs.to_epsg() != 4326:
        raise ValueError("AMBER snapping requires rivers in EPSG:4326.")
    # A wide search box avoids projecting the whole network for each barrier.
    latitude_buffer: float = 0.01
    longitude_buffer: float = latitude_buffer / max(np.cos(np.deg2rad(point.y)), 0.01)
    search_box: Polygon = box(
        point.x - longitude_buffer,
        point.y - latitude_buffer,
        point.x + longitude_buffer,
        point.y + latitude_buffer,
    )
    candidates: gpd.GeoDataFrame = rivers.iloc[rivers.sindex.query(search_box)]
    if candidates.empty:
        return candidates, float("nan")
    # A projection centered on the barrier measures distance in meters at any latitude.
    local_crs: str = (
        f"+proj=aeqd +lat_0={point.y} +lon_0={point.x} +datum=WGS84 +units=m"
    )
    # Both CRSs use WGS84, so project directly without repeatedly constructing
    # GeoDataFrames and asking PROJ to discover a datum transformation.
    projection: Proj = Proj(local_crs)
    projected_lines: npt.NDArray = shapely.transform(
        candidates.geometry.to_numpy(), projection, interleaved=False
    )
    distances_m: npt.NDArray[np.float64] = shapely.distance(
        projected_lines, Point(0, 0)
    )
    nearest_position: int = int(np.argmin(distances_m))
    distance_m: float = float(distances_m[nearest_position])
    if distance_m > 250.0:
        return candidates.iloc[:0], distance_m
    return candidates.iloc[[nearest_position]], distance_m
