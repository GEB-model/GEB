"""River water-surface profiles along a vector river network from a high-resolution DEM.

The profiles are computed once in the build (on FathomDEM along the MERIT-Hydro river
lines) and used both for the riverbed of the hydrological routing and for burning
rivers into the SFINCS flood model, so both models see the same rivers.
"""

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pyflwdir.dem import _adjust_elevation
from shapely.geometry import LineString

EARTH_RADIUS_M: float = 6371000.0


def line_vertex_distances_m(
    line: LineString, is_geographic: bool
) -> npt.NDArray[np.float64]:
    """Distance from the start of a line to each of its vertices.

    Args:
        line: The line.
        is_geographic: Whether the coordinates are longitude and latitude (degrees);
            otherwise they are in metres.

    Returns:
        Cumulative distance at each vertex (m).
    """
    coords = np.asarray(line.coords)
    dx, dy = np.diff(coords[:, 0]), np.diff(coords[:, 1])
    if is_geographic:
        mean_lat = np.deg2rad(0.5 * (coords[1:, 1] + coords[:-1, 1]))
        dx = np.deg2rad(dx) * np.cos(mean_lat) * EARTH_RADIUS_M
        dy = np.deg2rad(dy) * EARTH_RADIUS_M
    return np.r_[0.0, np.cumsum(np.hypot(dx, dy))]


def _sample_lines(rivers: gpd.GeoDataFrame, sampling_distance_m: float) -> pd.DataFrame:
    """Place sample points at regular distances along each river line.

    Args:
        rivers: River lines oriented from upstream to downstream, indexed by river ID.
        sampling_distance_m: Distance between sample points (m).

    Returns:
        One row per sample with river ID, position along the line (0 = upstream end),
        distance from the upstream end (m) and coordinates. The first and last sample
        of each line are its end points.
    """
    is_geographic: bool = bool(rivers.crs.is_geographic)
    rows: list[pd.DataFrame] = []
    for river_id, geometry in rivers.geometry.items():
        assert isinstance(geometry, LineString), (
            f"river {river_id} must be a LineString"
        )
        coords = np.asarray(geometry.coords)
        vertex_distance = line_vertex_distances_m(geometry, is_geographic)
        length = vertex_distance[-1]
        number_of_samples = max(2, int(np.ceil(length / sampling_distance_m)) + 1)
        distance = np.linspace(0.0, length, number_of_samples)
        rows.append(
            pd.DataFrame(
                {
                    "river_id": river_id,
                    "position": np.arange(number_of_samples),
                    "distance_m": distance,
                    "x": np.interp(distance, vertex_distance, coords[:, 0]),
                    "y": np.interp(distance, vertex_distance, coords[:, 1]),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _window_minimum(
    elevation: xr.DataArray,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    radius_pixels: int,
    band_rows: int = 2000,
) -> npt.NDArray[np.float32]:
    """Minimum DEM value in a square window around each point.

    The DEM is read in bands of rows so that large DEMs need not fit in memory. NaN
    pixels are ignored; points with only NaN pixels (or outside the DEM) get NaN.

    Args:
        elevation: DEM on a regular grid with x and y coordinates (m).
        x: Point x coordinates.
        y: Point y coordinates.
        radius_pixels: Half-width of the window (pixels).
        band_rows: Number of DEM rows read at once.

    Returns:
        Window minimum for each point (m).
    """
    x0, y0 = float(elevation.x[0]), float(elevation.y[0])
    dx = float(elevation.x[1] - elevation.x[0])
    dy = float(elevation.y[1] - elevation.y[0])
    column_index = np.round((x - x0) / dx).astype(np.int64)
    row_index = np.round((y - y0) / dy).astype(np.int64)
    n_rows, n_cols = elevation.shape
    result = np.full(x.size, np.inf, dtype=np.float32)
    for band_start in range(0, n_rows, band_rows):
        in_band = np.where(
            (row_index >= band_start) & (row_index < band_start + band_rows)
        )[0]
        if in_band.size == 0:
            continue
        read_start = max(0, band_start - radius_pixels)
        read_end = min(n_rows, band_start + band_rows + radius_pixels)
        block = np.asarray(
            elevation.isel(y=slice(read_start, read_end)).values, dtype=np.float32
        )
        block = np.where(np.isnan(block), np.inf, block)
        for d_row in range(-radius_pixels, radius_pixels + 1):
            window_row = row_index[in_band] + d_row
            row_is_valid = (window_row >= 0) & (window_row < n_rows)
            for d_col in range(-radius_pixels, radius_pixels + 1):
                window_column = column_index[in_band] + d_col
                valid = row_is_valid & (window_column >= 0) & (window_column < n_cols)
                values = np.full(in_band.size, np.inf, dtype=np.float32)
                values[valid] = block[
                    window_row[valid] - read_start, window_column[valid]
                ]
                result[in_band] = np.minimum(result[in_band], values)
    result[~np.isfinite(result)] = np.nan
    return result


def _interpolate_terraces(
    elevation: npt.NDArray[np.float64],
    downstream: npt.NDArray[np.int64],
    distance_to_outlet_m: npt.NDArray[np.float64],
    main_upstream: npt.NDArray[np.int64],
    order: npt.NDArray[np.int64],
    keep_flat: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Replace flat terraces by straight lines between terrace midpoints.

    A DEM water surface that is stored in steps (FathomDEM: 0.5 terrace_midpoint_m) is flat over long
    distances and drops at the end of each terrace. The true surface falls steadily and
    crosses the terrace level near the middle of the terrace, so the elevation is
    interpolated between the midpoints of consecutive terraces along the network (the
    upstream terrace is the one on the main upstream river). Points flagged
    ``keep_flat`` (lakes, sea) keep their level, and a terrace that drains into them
    stays flat down to them.

    Args:
        elevation: Elevation of each point, non-increasing downstream (m).
        downstream: Index of the downstream point (itself at the outlet).
        distance_to_outlet_m: Distance along the network to the outlet (m).
        main_upstream: Index of the upstream point with the largest upstream area (-1 if
            none).
        order: Point indices ordered from downstream to upstream.
        keep_flat: Points that keep their elevation (lakes, sea).

    Returns:
        Interpolated elevation (m), non-increasing in the downstream direction.
    """
    number_of_points = elevation.size
    terrace = np.arange(number_of_points)
    for i in order:
        downstream_point = downstream[i]
        if (
            downstream_point != i
            and elevation[downstream_point] == elevation[i]
            and keep_flat[downstream_point] == keep_flat[i]
        ):
            terrace[i] = terrace[downstream_point]
    grouped = pd.Series(distance_to_outlet_m[order], index=order).groupby(
        terrace[order]
    )
    start_distance = grouped.max()
    entry = grouped.idxmax()
    terrace_ids = start_distance.index.to_numpy()
    midpoint = np.full(number_of_points, np.nan)
    midpoint[terrace_ids] = 0.5 * (
        start_distance.to_numpy() + distance_to_outlet_m[terrace_ids]
    )
    downstream_terrace = np.full(number_of_points, -1)
    upstream_terrace = np.full(number_of_points, -1)
    for terrace_id in terrace_ids:
        if downstream[terrace_id] != terrace_id:
            downstream_terrace[terrace_id] = terrace[downstream[terrace_id]]
        upstream_point = main_upstream[entry[terrace_id]]
        if upstream_point >= 0:
            upstream_terrace[terrace_id] = terrace[upstream_point]

    result = elevation.copy()
    for i in order:
        terrace_id = terrace[i]
        if keep_flat[terrace_id]:
            continue
        terrace_midpoint_m, terrace_elevation_m = (
            midpoint[terrace_id],
            elevation[terrace_id],
        )
        if distance_to_outlet_m[i] <= terrace_midpoint_m:
            downstream_terrace_id = downstream_terrace[terrace_id]
            if (
                downstream_terrace_id >= 0
                and not keep_flat[downstream_terrace_id]
                and midpoint[downstream_terrace_id] < terrace_midpoint_m
            ):
                result[i] = terrace_elevation_m + (
                    elevation[downstream_terrace_id] - terrace_elevation_m
                ) * (terrace_midpoint_m - distance_to_outlet_m[i]) / (
                    terrace_midpoint_m - midpoint[downstream_terrace_id]
                )
        else:
            upstream_terrace_id = upstream_terrace[terrace_id]
            if (
                upstream_terrace_id >= 0
                and not keep_flat[upstream_terrace_id]
                and midpoint[upstream_terrace_id] > terrace_midpoint_m
            ):
                result[i] = terrace_elevation_m + (
                    elevation[upstream_terrace_id] - terrace_elevation_m
                ) * (distance_to_outlet_m[i] - terrace_midpoint_m) / (
                    midpoint[upstream_terrace_id] - terrace_midpoint_m
                )
    # a tributary can join an interpolated main river below its own level: raise it
    for i in order:
        downstream_point = downstream[i]
        if downstream_point != i and result[i] < result[downstream_point]:
            result[i] = result[downstream_point]
    return result


def _order_from_outlets(downstream_id: pd.Series) -> pd.Index:
    """Order rivers so that every river comes after the river it drains into.

    Args:
        downstream_id: Downstream river ID per river (index); -1 or an ID that is not
            in the index marks an outlet.

    Returns:
        River IDs sorted by the number of rivers between them and their outlet.

    Raises:
        ValueError: If the network contains a loop.
    """
    hops: dict[int, int] = {}
    for river_id in downstream_id.index:
        path: list[int] = []
        current = river_id
        while current not in hops:
            path.append(current)
            downstream_river = downstream_id[current]
            if downstream_river == -1 or downstream_river not in downstream_id.index:
                hops[current] = 0
                path.pop()
                break
            current = downstream_river
            if len(path) > len(downstream_id):
                raise ValueError("The river network contains a loop.")
        for depth_from_known, node in enumerate(reversed(path), start=1):
            hops[node] = hops[current] + depth_from_known
    return pd.Index(sorted(downstream_id.index, key=lambda r: hops[r]))


def estimate_river_surface_profiles(
    rivers: gpd.GeoDataFrame,
    elevation: xr.DataArray,
    lakes: gpd.GeoDataFrame | None = None,
    sampling_distance_m: float = 30.0,
    window_radius_pixels: int = 1,
    sea_level_m: float = 0.0,
) -> gpd.GeoDataFrame:
    """Estimate the water-surface elevation along every river line.

    1. Sample the DEM along each line every ``sampling_distance_m``; each sample is the
       minimum DEM value in a small window, which finds the water surface next to a
       line that does not follow the river exactly. Samples without DEM data are at
       sea (FathomDEM has no data over the sea) and get the sea level.
    2. Make each river non-increasing downstream with the minimum-modification
       adjustment of Yamazaki et al. (2012) (pyflwdir), from the outlets upstream. The
       adjustment fills a hole (gravel pit, mine) or cuts through a bump, whichever
       changes less; the start of the downstream river is the lowest allowed level at
       the end of each river.
    3. Interpolate flat terraces between their midpoints, except in lakes and at sea.

    Args:
        rivers: River lines from upstream to downstream, indexed by river ID, with
            columns ``downstream_ID`` (-1 or a river not in the table at outlets) and
            ``uparea_m2``. Geographic or metric CRS.
        elevation: DEM (m) in the CRS of the rivers.
        lakes: Lake polygons; points inside stay flat. None if there are no lakes.
        sampling_distance_m: Distance between samples (m).
        window_radius_pixels: Half-width of the sampling window (DEM pixels).
        sea_level_m: Water level of samples without DEM data (m).

    Returns:
        Points (CRS of the rivers) with columns ``river_id``, ``position`` (0 at the
        upstream end), ``distance_m`` (from the upstream end of the river),
        ``dem_minimum_m`` (raw sample), ``is_lake`` and ``water_surface_elevation_m``.

    Raises:
        ValueError: If the result is not monotone.
    """
    samples = _sample_lines(rivers, sampling_distance_m)
    samples["dem_minimum_m"] = _window_minimum(
        elevation,
        samples["x"].to_numpy(),
        samples["y"].to_numpy(),
        window_radius_pixels,
    )
    samples["is_lake"] = False
    if lakes is not None and len(lakes) > 0:
        points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(samples["x"], samples["y"]), crs=rivers.crs
        )
        inside = gpd.sjoin(
            points,
            lakes[["geometry"]].to_crs(rivers.crs),
            predicate="within",
            how="inner",
        )
        samples.loc[np.unique(inside.index.to_numpy()), "is_lake"] = True

    starts = samples.index[samples["position"] == 0].to_numpy()
    start_of = pd.Series(starts, index=samples.loc[starts, "river_id"].to_numpy())
    adjusted = samples["dem_minimum_m"].to_numpy(dtype=np.float64).copy()

    # 2. monotone per river, outlets first; the downstream river's start is the floor
    river_order = _order_from_outlets(rivers["downstream_ID"])
    groups = samples.groupby("river_id").indices
    for river_id in river_order:
        sample_indices = groups[river_id]
        elevation_m = np.where(
            np.isfinite(adjusted[sample_indices]), adjusted[sample_indices], sea_level_m
        )
        downstream_id = rivers.at[river_id, "downstream_ID"]
        if downstream_id != -1 and downstream_id in start_of.index:
            floor = adjusted[start_of[downstream_id]]
            elevation_m = _adjust_elevation(
                np.r_[elevation_m, floor].astype(np.float64)
            )[:-1]
        else:
            elevation_m = _adjust_elevation(elevation_m.astype(np.float64))
        adjusted[sample_indices] = elevation_m

    # network of points: each point drains to the next point of its river, the last
    # point of a river to the first point of the downstream river
    number_of_samples = len(samples)
    downstream = np.arange(number_of_samples) + 1
    last = np.r_[starts[1:] - 1, number_of_samples - 1]
    downstream_ids = rivers.loc[
        samples.loc[last, "river_id"].to_numpy(), "downstream_ID"
    ].to_numpy()
    has_downstream = (downstream_ids != -1) & np.isin(downstream_ids, start_of.index)
    downstream[last] = last
    downstream[last[has_downstream]] = start_of.loc[
        downstream_ids[has_downstream]
    ].to_numpy()
    step = np.r_[np.diff(samples["distance_m"].to_numpy()), 0.0]
    step[last] = 0.0
    upstream_area = rivers.loc[samples["river_id"], "uparea_m2"].to_numpy()

    order = np.concatenate([groups[river_id][::-1] for river_id in river_order])
    distance_to_outlet = np.zeros(number_of_samples)
    for i in order:
        downstream_point = downstream[i]
        if downstream_point != i:
            distance_to_outlet[i] = distance_to_outlet[downstream_point] + step[i]
    main_upstream = np.full(number_of_samples, -1, dtype=np.int64)
    best_area = np.full(number_of_samples, -np.inf)
    for i in range(number_of_samples):
        downstream_point = downstream[i]
        if downstream_point != i and upstream_area[i] > best_area[downstream_point]:
            best_area[downstream_point] = upstream_area[i]
            main_upstream[downstream_point] = i

    at_sea = ~np.isfinite(samples["dem_minimum_m"].to_numpy())
    surface = _interpolate_terraces(
        adjusted,
        downstream,
        distance_to_outlet,
        main_upstream,
        order,
        samples["is_lake"].to_numpy() | at_sea,
    )
    if (surface - surface[downstream] < -1e-6).any():
        raise ValueError("River surface profile is not monotone.")
    samples["water_surface_elevation_m"] = surface.astype(np.float32)
    return gpd.GeoDataFrame(
        samples.drop(columns=["x", "y"]),
        geometry=gpd.points_from_xy(samples["x"], samples["y"]),
        crs=rivers.crs,
    )


def river_bed_profiles(
    rivers: gpd.GeoDataFrame, profiles: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Riverbed elevation along the rivers: water surface minus river depth.

    The depth at each junction is the largest depth of the rivers that meet there, and
    is interpolated linearly along each river between its two junctions. The bed_elevation_per_sample_m is
    made non-increasing downstream.

    Args:
        rivers: Rivers indexed by river ID with ``depth`` (m) and ``downstream_ID``.
        profiles: Output of ``estimate_river_surface_profiles`` containing at least
            these rivers.

    Returns:
        The profile points of the given rivers (in profile order) with an extra column
        ``bed_elevation_m``.
    """
    profiles = profiles[profiles["river_id"].isin(rivers.index)].copy()
    depth = rivers["depth"].astype(np.float64)
    joining_max = depth.groupby(rivers["downstream_ID"]).max()
    start_node_depth = np.maximum(depth, joining_max.reindex(depth.index).fillna(0.0))
    end_node_depth = pd.Series(
        [
            max(
                depth[river_id],
                joining_max.get(downstream_id, 0.0),
                depth.get(downstream_id, 0.0),
            )
            for river_id, downstream_id in rivers["downstream_ID"].items()
        ],
        index=depth.index,
    )

    bed_elevation_per_sample_m = np.empty(len(profiles), dtype=np.float64)
    for river_id, sample_indices in profiles.groupby(
        "river_id", sort=False
    ).indices.items():
        distance_along_river_m = profiles["distance_m"].to_numpy()[sample_indices]
        fraction = (
            distance_along_river_m / distance_along_river_m[-1]
            if distance_along_river_m[-1] > 0
            else np.zeros_like(distance_along_river_m)
        )
        depth_m = (
            start_node_depth[river_id]
            + (end_node_depth[river_id] - start_node_depth[river_id]) * fraction
        )
        bed_elevation_m = (
            profiles["water_surface_elevation_m"].to_numpy(dtype=np.float64)[
                sample_indices
            ]
            - depth_m
        )
        bed_elevation_per_sample_m[sample_indices] = np.minimum.accumulate(
            bed_elevation_m
        )
    profiles["bed_elevation_m"] = bed_elevation_per_sample_m.astype(np.float32)
    return profiles
