"""Prepare AMBER weir locations and height fractions for river routing."""

from collections.abc import Hashable

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
import xarray as xr
from pyproj import Proj
from shapely.geometry import Point, box
from shapely.ops import nearest_points

from geb.workflows.raster import full_like

MAX_WEIR_DISTANCE_M: float = 250.0


def snap_amber_weirs(
    barriers: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    valid_river_cells: xr.DataArray,
) -> tuple[xr.DataArray, gpd.GeoDataFrame]:
    """Snap AMBER weirs and assign half-bankfull-depth crest heights.

    Notes:
        Match the river line within 250 m before selecting its nearest model cell.
        Distances use a WGS84 azimuthal equidistant projection centered on each
        source point. The distance limit applies to the line, not the grid center.
        One weir controls each cell's downstream interface. If its closest cell
        is occupied, try only the second-closest distinct cell on the same river
        segment. A wider search could move the structure far from its source.
        Process by AMBER ID so assignments do not depend on input row order.

    Args:
        barriers: AMBER points with amber_id and dam_type. Recorded heights
            are retained as metadata when present but never used for placement.
        rivers: River lines with represented_in_grid and hydrography_xy columns.
            hydrography_xy contains model column/row pairs.
        valid_river_cells: Boolean y/x grid in EPSG:4326; True where both sides
            of the downstream connection are active river cells outside waterbodies.

    Returns:
        Bankfull-depth fraction grid (0.5 at weirs, zero elsewhere; dimensionless)
        and source-point records
        with distances (meters), snapped coordinates (degrees), grid indices,
        snap_rank (1 for closest, 2 for second-closest, 0 for exclusions) and
        exclusion reasons. Excluded records have grid indices of -1.

    Raises:
        ValueError: If required fields, coordinate systems, geometries or grid
            indices are invalid.
    """
    if not {"amber_id", "dam_type"}.issubset(barriers.columns):
        raise ValueError("AMBER snapping requires IDs and barrier types.")
    if not {"represented_in_grid", "hydrography_xy"}.issubset(rivers.columns):
        raise ValueError("River snapping requires represented rivers and grid cells.")
    if barriers.crs is None or rivers.crs is None:
        raise ValueError(
            "AMBER points and rivers require coordinate reference systems."
        )
    if (
        valid_river_cells.dims != ("y", "x")
        or valid_river_cells.dtype != bool
        or valid_river_cells.rio.crs is None
        or valid_river_cells.rio.crs.to_epsg() != 4326
    ):
        raise ValueError("Valid river cells must be a boolean y/x grid in EPSG:4326.")
    records: gpd.GeoDataFrame = (
        barriers.loc[barriers["dam_type"].str.strip().str.casefold().eq("weir")]
        .to_crs(4326)
        .reset_index(drop=True)
    )
    if records.geometry.isna().any() or not records.geom_type.eq("Point").all():
        raise ValueError("AMBER weir geometries must be nonempty points.")
    if (
        records.geometry.is_empty.any()
        or not (
            records.geometry.x.between(-180, 180) & records.geometry.y.between(-90, 90)
        ).all()
    ):
        raise ValueError("AMBER weir coordinates must be finite WGS84 locations.")
    represented: gpd.GeoDataFrame = rivers.loc[rivers.represented_in_grid].to_crs(4326)
    if not represented.geom_type.isin(["LineString", "MultiLineString"]).all():
        raise ValueError("Represented rivers must have line geometries.")
    # Materialize the output: assigning through .values on Dask returns a copy.
    fractions: xr.DataArray = full_like(
        valid_river_cells, fill_value=0.0, nodata=np.nan, dtype=np.float32
    ).compute()
    records["bankfull_depth_fraction"] = np.nan
    records["distance_to_river_m"] = np.nan
    records["distance_to_cell_m"] = np.nan
    records["snap_rank"] = np.int64(0)
    records["grid_row"] = np.int64(-1)
    records["grid_column"] = np.int64(-1)
    records["river_id"] = pd.Series(pd.NA, index=records.index, dtype="Int64")
    records["snapped_longitude"] = np.nan
    records["snapped_latitude"] = np.nan
    records["included"] = False
    records["exclusion_reason"] = ""
    valid_cells: npt.NDArray[np.bool_] = valid_river_cells.values
    row_count: int
    column_count: int
    row_count, column_count = valid_cells.shape
    record_index: Hashable
    barrier: pd.Series
    processing_order: pd.Index = (
        records["amber_id"].astype(str).sort_values(kind="stable").index
    )
    for record_index, barrier in records.loc[processing_order].iterrows():
        point: Point = barrier.geometry
        # This generous search box avoids projecting the full river network per point.
        longitude_buffer: float = 0.01 / max(np.cos(np.deg2rad(point.y)), 0.01)
        candidate_positions: npt.NDArray[np.int64] = represented.sindex.query(
            box(
                point.x - longitude_buffer,
                point.y - 0.01,
                point.x + longitude_buffer,
                point.y + 0.01,
            )
        )
        if not candidate_positions.size:
            records.at[record_index, "exclusion_reason"] = "no river within 250 m"
            continue
        candidates: gpd.GeoDataFrame = represented.iloc[np.sort(candidate_positions)]
        projection: Proj = Proj(
            f"+proj=aeqd +lat_0={point.y} +lon_0={point.x} +datum=WGS84 +units=m"
        )
        projected_lines: npt.NDArray = shapely.transform(
            candidates.geometry.to_numpy(), projection, interleaved=False
        )
        distances_m: npt.NDArray[np.float64] = shapely.distance(
            projected_lines, Point(0, 0)
        )
        nearest_position: int = int(np.argmin(distances_m))
        distance_m: float = float(distances_m[nearest_position])
        records.at[record_index, "distance_to_river_m"] = distance_m
        # Allow micrometer-scale projection roundoff at the inclusive boundary.
        if not np.isfinite(distance_m) or distance_m > MAX_WEIR_DISTANCE_M + 1e-6:
            records.at[record_index, "exclusion_reason"] = "farther than 250 m"
            continue
        river: pd.Series = candidates.iloc[nearest_position]
        if not len(river.hydrography_xy):
            records.at[record_index, "exclusion_reason"] = "river has no model cells"
            continue
        cells: npt.NDArray[np.int64] = np.asarray(
            list(river.hydrography_xy), dtype=np.int64
        )
        if (
            cells.ndim != 2
            or cells.shape[1] != 2
            or np.any(cells < 0)
            or np.any(cells[:, 0] >= column_count)
            or np.any(cells[:, 1] >= row_count)
        ):
            raise ValueError("River grid indices are outside the model grid.")
        projected_x: npt.NDArray[np.float64]
        projected_y: npt.NDArray[np.float64]
        projected_x, projected_y = projection(
            valid_river_cells.x.values[cells[:, 0]],
            valid_river_cells.y.values[cells[:, 1]],
        )
        closest_point: Point = nearest_points(
            Point(0, 0), projected_lines[nearest_position]
        )[1]
        cell_distances_m: npt.NDArray[np.float64] = np.hypot(
            projected_x - closest_point.x, projected_y - closest_point.y
        )
        nearest_cell: int = int(np.argmin(cell_distances_m))
        column: int = int(cells[nearest_cell, 0])
        row: int = int(cells[nearest_cell, 1])
        snap_rank: int = 1
        if fractions.values[row, column] != 0:
            # Match the old waterworks method: retain the first weir and try
            # one neighboring placement, without jumping to another river.
            alternatives: npt.NDArray[np.int64] = np.flatnonzero(
                np.any(cells != (column, row), axis=1)
            )
            if not alternatives.size:
                records.at[record_index, "exclusion_reason"] = "cell already used"
                continue
            nearest_cell = int(alternatives[np.argmin(cell_distances_m[alternatives])])
            column, row = map(int, cells[nearest_cell])
            snap_rank = 2
        if fractions.values[row, column] != 0:
            records.at[record_index, "exclusion_reason"] = "cell already used"
            continue
        if not valid_cells[row, column]:
            records.at[record_index, "exclusion_reason"] = "unsuitable river connection"
            continue
        fractions.values[row, column] = 0.5
        records.at[record_index, "bankfull_depth_fraction"] = 0.5
        records.at[record_index, "snap_rank"] = snap_rank
        records.at[record_index, "distance_to_cell_m"] = float(
            np.hypot(projected_x[nearest_cell], projected_y[nearest_cell])
        )
        records.at[record_index, "grid_row"] = row
        records.at[record_index, "grid_column"] = column
        records.at[record_index, "river_id"] = candidates.index[nearest_position]
        records.at[record_index, "snapped_longitude"] = float(
            valid_river_cells.x[column]
        )
        records.at[record_index, "snapped_latitude"] = float(valid_river_cells.y[row])
        records.at[record_index, "included"] = True
    return fractions, records
