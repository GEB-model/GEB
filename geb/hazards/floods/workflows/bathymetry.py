"""Workflows for DEM processing and river burning."""

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.features import rasterize
from scipy.spatial import cKDTree  # ty:ignore[unresolved-import]
from shapely.ops import transform as shapely_transform
from skimage.morphology import reconstruction


def fill_depressions(da_elv: xr.DataArray) -> xr.DataArray:
    """Fills topological sinks in a DEM using morphological reconstruction.

    Args:
        da_elv: Elevation grid with CRS and transform.

    Returns:
        Elevation grid with depressions filled.
    """
    dem_arr = da_elv.values.copy()
    nan_mask = np.isnan(dem_arr)
    if nan_mask.any():
        dem_arr[nan_mask] = np.nanmax(dem_arr) + 10.0

    seed = np.full(dem_arr.shape, np.nanmax(dem_arr) + 10.0)
    seed[0, :] = dem_arr[0, :]
    seed[-1, :] = dem_arr[-1, :]
    seed[:, 0] = dem_arr[:, 0]
    seed[:, -1] = dem_arr[:, -1]

    filled_arr = reconstruction(seed, dem_arr, method="erosion")
    if nan_mask.any():
        filled_arr[nan_mask] = np.nan

    return da_elv.copy(data=filled_arr)


def burn_rivers(
    da_elevation: xr.DataArray,
    da_manning: xr.DataArray,
    rivers: gpd.GeoDataFrame,
    fill_first: bool = True,
) -> tuple[xr.DataArray, xr.DataArray, gpd.GeoDataFrame]:
    """Burns river networks into a Digital Elevation Model (DEM).

    This function sanitizes vector river inputs, structures them into a
    topologically sorted tree, enforces matching profiles across confluences,
    interpolates longitudinal beds, ensures downhill flow, and stamps them
    back into a 2D raster grid.

    Args:
        da_elevation: Elevation grid with CRS and transform.
        da_manning: Manning's n grid with CRS and transform.
        rivers: GeoDataFrame containing river centerlines and attributes.
            Must have "downstream_ID", "manning", "width", and "depth" columns.
        fill_first: Whether to fill depressions in the DEM before burning.

    Returns:
        A tuple containing:
            - The updated elevation grid with rivers burned in.
            - A GeoDataFrame with updated river attributes.

    Raises:
        ValueError: For missing attribute columns, invalid geometries, or empty datasets.
        ValueError: If the DEM and River layer CRS projections do not match.
        ValueError: If network topology is broken (dead downstream IDs or self-loops).
        ValueError: If any node falls outside the valid data boundaries of the DEM.
        ValueError: If a river profile structurally or internally flows uphill.
    """
    # Input validation
    if rivers.empty:
        raise ValueError("Input 'rivers' GeoDataFrame is empty.")
    if "width" not in rivers.columns:
        raise ValueError(f"Width column 'width' not found in GeoDataFrame.")
    if "depth" not in rivers.columns:
        raise ValueError(f"Depth column 'depth' not found in GeoDataFrame.")
    if "downstream_ID" not in rivers.columns:
        raise ValueError(
            "Required network topology column 'downstream_ID' not found in GeoDataFrame."
        )
    if "manning" not in rivers.columns:
        raise ValueError(f"Manning column 'manning' not found in GeoDataFrame.")
    if da_elevation.rio.crs is None:
        raise ValueError("Input elevation DataArray must have a valid CRS.")

    if rivers.crs != da_elevation.rio.crs:
        raise ValueError(
            f"Spatial CRS mismatch! Rivers layer uses ({rivers.crs}) "
            f"but DEM layer uses ({da_elevation.rio.crs}). Please reproject."
        )

    out_elevation = (
        fill_depressions(da_elevation.copy()) if fill_first else da_elevation.copy()
    )
    transform = out_elevation.rio.transform()
    shape = out_elevation.shape
    is_geo = out_elevation.rio.crs.is_geographic

    # Hydrological slope and distance calculations require physical meters.
    # If the CRS is geographic (degrees), 1° Lat != 1° Lon. We compute a localized
    # scale factor at the DEM center: m_y is fixed, m_x scales via cos(latitude).
    x_orig, y_orig = (
        float(out_elevation.coords["x"].mean()),
        float(out_elevation.coords["y"].mean()),
    )
    m_x = 111320.0 * np.cos(np.radians(y_orig)) if is_geo else 1.0
    m_y = 111320.0 if is_geo else 1.0

    # Match sampling distance to the grid resolution (meters)
    sampling_dist = min(abs(transform[0]) * m_x, abs(transform[4]) * m_y)

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

    # High-density points are sampled along sparse vector lines every `sampling_dist`.
    # Using `downstream_ID`, the structural orientation of the LineString is validated.
    # If the geometry was digitized backwards (start point closer to the downstream
    # link than the end point), a ValueError is raised to enforce strict network data health.
    # Junction nodes are registered, sampling a local 3x3 pixel window to find
    # the local terrain minimum and subtracting the channel depth.
    river_network, node_registry = [], {}

    # explode multilinestrings
    rivers: gpd.GeoDataFrame = rivers.explode(index_parts=True)  # ty:ignore[invalid-assignment]
    # create composite index for multi-part geometries
    rivers.index = [f"{idx[0]}_{idx[1]}" for idx in rivers.index]

    for idx, row in rivers.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            raise ValueError(
                f"River feature at index {idx} has an empty or null geometry."
            )
        if geom.geom_type != "LineString":
            raise ValueError(
                f"River geometry at index {idx} is not a LineString: {geom.geom_type}"
            )

        width, depth, downstream_id = (
            float(row["width"]),
            float(row["depth"]),
            row["downstream_ID"],
        )

        # if the downstream_id is not in the rivers index, treat it as a sink (-1)
        if downstream_id not in rivers.index:
            downstream_id = -1

        if is_geo:
            len_m = shapely_transform(to_m, geom).length
            distances_norm = np.linspace(
                0, 1, max(2, int(np.ceil(len_m / sampling_dist)))
            )
            points = [geom.interpolate(d, normalized=True) for d in distances_norm]
        else:
            distances = np.arange(0, geom.length, sampling_dist)
            if not len(distances) or distances[-1] != geom.length:
                distances = np.append(distances, geom.length)
            points = [geom.interpolate(d) for d in distances]

        # Enforce strict line digitization direction matching network flow direction
        if downstream_id != -1 and downstream_id in rivers.index:
            ds_geom = rivers.loc[downstream_id].geometry
            if points[0].distance(ds_geom) < points[-1].distance(ds_geom):
                raise ValueError(
                    f"Geometry direction error: River segment {idx} is digitized backwards. "
                    "The geometry's starting node is closer to its downstream receiver than its ending node."
                )

        xs, ys = np.array([p.x for p in points]), np.array([p.y for p in points])
        xs_m, ys_m = to_m(xs, ys)
        cumulative_distances_m = np.insert(
            np.cumsum(np.sqrt(np.diff(xs_m) ** 2 + np.diff(ys_m) ** 2)), 0, 0.0
        ).astype(np.float32)

        start_key, end_key = (
            idx,
            downstream_id if downstream_id != -1 else f"sink_{idx}",
        )

        river_network.append(
            {
                "idx": idx,
                "distances": cumulative_distances_m,
                "xs": xs,
                "ys": ys,
                "start_key": start_key,
                "end_key": end_key,
                "width": width,
                "depth": depth,
                "manning": float(row["manning"]),
                "geom": geom,
                "target_z": None,
            }
        )

        for key, px, py in [(start_key, xs[0], ys[0]), (end_key, xs[-1], ys[-1])]:
            if key not in node_registry:
                node_registry[key] = {"x": px, "y": py, "max_depth": 0.0}
            node_registry[key]["max_depth"] = max(
                node_registry[key]["max_depth"], depth
            )

    for node in node_registry.values():
        c = int(np.floor((node["x"] - transform[2]) / transform[0]))
        assert 0 <= c < shape[1], (
            f"Node at ({node['x']}, {node['y']}) falls outside the DEM bounds. "
            "Ensure your river geometries are within the DEM extent."
        )
        r = int(np.floor((node["y"] - transform[5]) / transform[4]))
        assert 0 <= r < shape[0], (
            f"Node at ({node['x']}, {node['y']}) falls outside the DEM bounds. "
            "Ensure your river geometries are within the DEM extent."
        )
        window = out_elevation.values[
            max(0, r - 1) : min(shape[0], r + 2), max(0, c - 1) : min(shape[1], c + 2)
        ]

        if not np.all(np.isfinite(window)):
            raise ValueError(
                f"Node at ({node['x']}, {node['y']}) falls into a NoData region of the DEM."
            )

        node["z_bed"] = np.nanmin(window) - node["max_depth"]

        # Verify that the initial water surface elevation is not below sea level.
        bankfull_water_z_m = node["z_bed"] + node["max_depth"]
        if bankfull_water_z_m < -3:
            raise ValueError(
                f"Bankfull water surface at node ({node['x']:.5f}, {node['y']:.5f}) was computed well below sea level ({bankfull_water_z_m:.3f}m). "
                "GEB requires all river water levels to be non-negative (>= 0m)."
            )

    # Identifies parent segments and tracks "in-degrees" (incoming connections).
    # Headwater segments (0 incoming links) enter a queue and drive the sequence.
    # Pruning links cascades down the basin, ordering indices so that any upstream
    # segment is fully evaluated before its downstream confluence is reached.
    for i, river in enumerate(river_network):
        river["parents"] = {
            j
            for j, other in enumerate(river_network)
            if i != j and other["end_key"] == river["start_key"]
        }

    in_degrees = {i: len(river["parents"]) for i, river in enumerate(river_network)}
    queue = [i for i, degree in in_degrees.items() if degree == 0]
    ordered_indices = []

    while queue:
        current_idx = queue.pop(0)
        ordered_indices.append(current_idx)
        for i, river in enumerate(river_network):
            if current_idx in river["parents"]:
                in_degrees[i] -= 1
                if in_degrees[i] == 0:
                    queue.append(i)

    if len(ordered_indices) < len(river_network):
        raise ValueError(
            "Cyclic loop dependency found in your river network! "
            "Ensure downstream tracking paths do not circle back up into previous tributaries."
        )

    # Loops through the topologically sorted network to check boundary nodes.
    # Checks the structural water elevations of the start and end nodes directly.
    # If a downstream node is discovered to be higher than its corresponding
    # upstream source, it halts execution and reports the uphill tracking failure.
    for idx in ordered_indices:
        river = river_network[idx]
        node_start = node_registry[river["start_key"]]
        node_end = node_registry[river["end_key"]]

        # Water surface elevation (bed + depth)
        water_z_start_m = node_start["z_bed"] + node_start["max_depth"]
        water_z_end_m = node_end["z_bed"] + node_end["max_depth"]

        if water_z_end_m > water_z_start_m:
            # Small uphill gradients (<= 3m) are likely DEM artifacts or noise.
            # We flatten these to ensure downstream monotonicity without failing.
            if water_z_end_m - water_z_start_m <= 3.0:
                # Adjust z_bed such that w_end == w_start
                node_end["z_bed"] = min(
                    node_end["z_bed"], water_z_start_m - node_end["max_depth"]
                )
            else:
                raise ValueError(
                    f"River segment {river['idx']} flows uphill (water surface) by more than 3m! "
                    f"Start node w: {water_z_start_m:.3f}, End node w: {water_z_end_m:.3f}"
                )

    # Maps point locations to the DEM grid, sampling raw underlying heights.
    # Applies a linear correction blend to keep the unique terrain signature
    # while eliminating boundary steps at confluences.
    # If the local topography creates a localized hill or bump along the channel,
    # a forward running minimum filter is applied to shave down the peaks, ensuring
    # that the river profile monotonically trends downward toward the outlet.
    for river in river_network:
        cols = np.floor((river["xs"] - transform[2]) / transform[0]).astype(int)
        assert cols.min() >= 0 and cols.max() < shape[1], (
            f"River segment {river['idx']} has x-coordinates outside the DEM bounds. "
            "Ensure your river geometries are within the DEM extent."
        )

        rows = np.floor((river["ys"] - transform[5]) / transform[4]).astype(int)
        assert rows.min() >= 0 and rows.max() < shape[0], (
            f"River segment {river['idx']} has y-coordinates outside the DEM bounds. "
            "Ensure your river geometries are within the DEM extent."
        )

        sampled_z_m = out_elevation.values[rows, cols]
        if (sampled_z_m < -3).any():
            raise ValueError(
                f"River segment {river['idx']} has sampled elevations below -3m. "
                "Ensure your DEM is valid and does not contain negative artifacts."
            )

        valid = np.isfinite(sampled_z_m)
        if not valid.all():
            sampled_z_m = np.interp(
                river["distances"], river["distances"][valid], sampled_z_m[valid]
            )

        start_node = node_registry[river["start_key"]]
        end_node = node_registry[river["end_key"]]

        bankfull_start_z_m = start_node["z_bed"] + start_node["max_depth"]
        bankfull_end_z_m = end_node["z_bed"] + end_node["max_depth"]

        if bankfull_start_z_m < -3:
            raise ValueError(
                f"River segment {river['idx']} has a start node water surface below -3m. "
                "Ensure your DEM and depth attributes are consistent and non-negative."
            )
        if bankfull_end_z_m < -3:
            raise ValueError(
                f"River segment {river['idx']} has an end node water surface below -3m. "
                "Ensure your DEM and depth attributes are consistent and non-negative."
            )

        segment_length_m = river["distances"][-1]
        profile_z_bed_m = sampled_z_m.copy()

        assert segment_length_m > 0
        assert len(profile_z_bed_m) > 1

        # Shift intermediate terrain smoothly to tie into the raw endpoint anchors
        profile_z_bed_m += np.linspace(
            bankfull_start_z_m - sampled_z_m[0],
            bankfull_end_z_m - sampled_z_m[-1],
            len(profile_z_bed_m),
        )

        # Now carve the river channel depth uniformly out of the corrected surface profile
        profile_z_bed_m -= river["depth"]

        # Fetch final bed bounds for clipping
        target_z_bed_start_m = node_registry[river["start_key"]]["z_bed"]
        target_z_bed_end_m = node_registry[river["end_key"]]["z_bed"]
        profile_z_bed_m = np.clip(
            profile_z_bed_m, target_z_bed_end_m, target_z_bed_start_m
        )

        if len(profile_z_bed_m) > 4:
            filter_width = min(9, (len(profile_z_bed_m) // 4) * 2 + 1)
            if filter_width > 1:
                profile_z_bed_m = np.convolve(
                    np.pad(profile_z_bed_m, filter_width // 2, mode="edge"),
                    np.ones(filter_width) / filter_width,
                    mode="valid",
                )
                profile_z_bed_m = np.pad(
                    profile_z_bed_m,
                    (0, max(0, len(river["distances"]) - len(profile_z_bed_m))),
                    mode="edge",
                )[: len(river["distances"])]

        if (profile_z_bed_m + river["depth"] < -3).any():
            raise ValueError(
                f"River segment {river['idx']} has a smoothed water surface below -3m. "
                "Ensure your DEM and depth attributes are consistent and non-negative."
            )

        # Dynamically flatten internal topographical ridges/bumps
        # Every point along the profile is forced to be less than or equal to its upstream neighbor
        for i in range(1, len(profile_z_bed_m)):
            profile_z_bed_m[i] = min(profile_z_bed_m[i], profile_z_bed_m[i - 1])

        # Snap boundary conditions explicitly to node entries
        profile_z_bed_m[0], profile_z_bed_m[-1] = (
            target_z_bed_start_m,
            target_z_bed_end_m,
        )

        # Verify that the profile calculations did not push the water surface below sea level.
        water_surface_m = profile_z_bed_m + river["depth"]
        if (water_surface_m < -3).any():
            min_water_z_m = np.min(water_surface_m)
            raise ValueError(
                f"The profile for river {river['idx']} has a bankfull water surface below sea level (min: {min_water_z_m:.3f}m). "
                "Ensure your DEM and depth attributes are consistent and non-negative."
            )

        river["target_z"] = profile_z_bed_m

    # Projects the 1D line profiles into a 2D channel width on the raster grid.
    # Generates a polygon buffer in meter space (using flat caps to prevent blobs
    # at confluences) and rasterizes it into a local binary pixel mask.
    # A cKDTree links mask pixels to the closest profile node to extract depth.
    # Merges parallel channels using `np.fmin` to let the lowest bed win.
    global_burn_elevation = np.full(shape, fill_value=np.inf, dtype=np.float32)
    global_burn_manning = np.full(shape, fill_value=-np.inf, dtype=np.float32)
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
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )
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

    update_mask = np.isfinite(global_burn_elevation)
    out_elevation.values[update_mask] = global_burn_elevation[update_mask]

    out_mannings = da_manning.copy()
    out_mannings.values[update_mask] = global_burn_manning[update_mask]

    # Pulls all original user attributes (ignoring geometry duplicates) and
    # updates them with specific longitudinal metrics like minimum bed levels,
    # exact physical length in meters, and calculated average channel slopes.
    burned_river_data = []
    for river in river_network:
        attributes = {
            k: v
            for k, v in rivers.loc[river["idx"]].to_dict().items()
            if k != "geometry"
        }
        attributes.update(
            {
                "z_bed_start": float(river["target_z"][0]),
                "z_bed_end": float(river["target_z"][-1]),
                "z_bed_min": float(np.min(river["target_z"])),
                "z_bed_max": float(np.max(river["target_z"])),
                "z_bed_avg": float(np.mean(river["target_z"])),
                "length_m": float(river["distances"][-1]),
                "slope": float(
                    (river["target_z"][0] - river["target_z"][-1])
                    / river["distances"][-1]
                )
                if river["distances"][-1] > 0
                else 0.0,
                "geometry": river["geom"],
            }
        )
        burned_river_data.append(attributes)

    return (
        out_elevation,
        out_mannings,
        gpd.GeoDataFrame(burned_river_data, crs=rivers.crs),
    )
