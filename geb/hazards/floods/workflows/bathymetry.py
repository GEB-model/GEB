"""Workflows for DEM processing and river burning."""

from typing import Callable

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.features import rasterize
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]
from shapely.ops import transform as shapely_transform
from skimage.morphology import reconstruction


def fill_depressions(elevation_grid: xr.DataArray) -> xr.DataArray:
    """Fills topological sinks in a DEM using morphological reconstruction.

    Args:
        elevation_grid: Elevation grid with CRS and transform (meters).

    Returns:
        Elevation grid with depressions filled (meters).
    """
    dem_arr = elevation_grid.values.copy()
    nan_mask = np.isnan(dem_arr)
    # If there are NaNs, we temporarily fill them with a very high value
    # so the reconstruction algorithm treats them as peaks rather than sinks.
    if nan_mask.any():
        dem_arr[nan_mask] = np.nanmax(dem_arr) + 10.0

    seed = np.full(dem_arr.shape, np.nanmax(dem_arr) + 10.0)
    seed[0, :] = dem_arr[0, :]
    seed[-1, :] = dem_arr[-1, :]
    seed[:, 0] = dem_arr[:, 0]
    seed[:, -1] = dem_arr[:, -1]

    filled_arr = reconstruction(seed, dem_arr, method="erosion")
    # Restore the original NaNs after filling the sinks.
    if nan_mask.any():
        filled_arr[nan_mask] = np.nan

    return elevation_grid.copy(data=filled_arr)


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
        "topological_stream_order",
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


def _initialize_network_and_nodes(
    rivers: gpd.GeoDataFrame,
    elevation_grid: xr.DataArray,
    transform: tuple,
    shape: tuple,
    is_geo: bool,
    sampling_dist: float,
    to_m: Callable[
        [np.ndarray | float, np.ndarray | float], tuple[np.ndarray, np.ndarray]
    ],
) -> tuple[list[dict], dict]:
    """Initialize the river network and node registry.

    This step breaks down the river geometries into points, samples the terrain
    elevation at each junction, and calculates the initial bed elevation.

    Args:
        rivers: GeoDataFrame containing river geometries and attributes.
        elevation_grid: Elevation DataArray with CRS and transform (meters).
        transform: Affine transform of the grid.
        shape: Shape of the grid.
        is_geo: Whether the CRS is geographic.
        sampling_dist: Distance between sampled points along the river (meters).
        to_m: Function to convert coordinates to meters.

    Returns:
        A tuple containing the list of river segments and the node registry.

    Raises:
        ValueError: If a river geometry is invalid or a node falls outside the grid.
    """
    river_network = []
    node_registry = {}

    # Track original index to map topological downstream relationships accurately
    rivers_working = rivers.copy()
    rivers_working["_orig_id"] = rivers_working.index

    rivers_exploded = rivers_working.explode(index_parts=True)
    rivers_exploded["_new_id"] = [f"{idx[0]}_{idx[1]}" for idx in rivers_exploded.index]
    rivers_exploded.index = rivers_exploded["_new_id"]

    # Map original IDs to their corresponding first exploded segment ID
    orig_to_new_map = rivers_exploded.groupby("_orig_id")["_new_id"].first().to_dict()

    for idx, row in rivers_exploded.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != "LineString":
            raise ValueError(
                f"River feature at index {idx} must be a valid, non-empty LineString."
            )

        width, depth = float(row["width"]), float(row["depth"])
        raw_downstream_id = row["downstream_ID"]
        downstream_id = orig_to_new_map.get(raw_downstream_id, -1)

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

        xs, ys = np.array([p.x for p in points]), np.array([p.y for p in points])
        xs_m, ys_m = to_m(xs, ys)
        cumulative_distances_m = np.insert(
            np.cumsum(np.sqrt(np.diff(xs_m) ** 2 + np.diff(ys_m) ** 2)), 0, 0.0
        ).astype(np.float32)

        start_key, end_key = (
            idx,
            downstream_id if downstream_id != -1 else f"sink_{idx}",
        )

        # Cache original properties safely away from index transformations
        clean_attributes = {
            k: v
            for k, v in row.to_dict().items()
            if k not in ["geometry", "_orig_id", "_new_id"]
        }

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
                "orig_attributes": clean_attributes,
            }
        )

        for key, px, py in [(start_key, xs[0], ys[0]), (end_key, xs[-1], ys[-1])]:
            if key not in node_registry:
                node_registry[key] = {"x": px, "y": py, "max_depth": 0.0}
            node_registry[key]["max_depth"] = max(
                node_registry[key]["max_depth"], depth
            )

    # Sample a 3x3 window around each node to find the local minimum elevation.
    # This ensures the river starts at the lowest point in its immediate vicinity.
    dem_grid = (
        elevation_grid.values[0] if elevation_grid.ndim == 3 else elevation_grid.values
    )
    for node in node_registry.values():
        c = int(np.floor((node["x"] - transform[2]) / transform[0]))
        r = int(np.floor((node["y"] - transform[5]) / transform[4]))

        if not (0 <= c < shape[-1] and 0 <= r < shape[-2]):
            raise ValueError(
                f"Node at ({node['x']}, {node['y']}) falls outside the DEM bounds."
            )

        window = dem_grid[
            max(0, r - 1) : min(shape[-2], r + 2), max(0, c - 1) : min(shape[-1], c + 2)
        ]

        if not np.all(np.isfinite(window)):
            raise ValueError(
                f"Node at ({node['x']}, {node['y']}) falls into a NoData region."
            )

        # The water surface is assumed to be the local minimum terrain elevation.
        node["z_surf"] = np.nanmin(window)
        # The river bed is then carved down by the nominal depth.
        node["z_bed"] = node["z_surf"] - node["max_depth"]

        if node["z_surf"] < -3:
            raise ValueError(
                f"Water surface at node ({node['x']:.5f}, {node['y']:.5f}) is well below sea level."
            )

    return river_network, node_registry


def _enforce_upstream_propagation(
    river_network: list[dict], ordered_indices: list[int], node_registry: dict
) -> None:
    """Enforce downhill continuity across the network.

    Iterates backward from river mouths to headwaters to ensure that every
    upstream node is at least slightly higher than its downstream neighbor.

    Args:
        river_network: List of river segment dictionaries.
        ordered_indices: Topologically sorted indices of river segments.
        node_registry: Registry of node elevations.
    """
    epsilon = 0.001  # Minute elevation buffer (meters)

    for idx in reversed(ordered_indices):
        river = river_network[idx]
        node_down = node_registry[river["end_key"]]
        node_up = node_registry[river["start_key"]]

        # If the upstream bed is lower than the downstream bed, we lift it up.
        node_up["z_bed"] = max(node_up["z_bed"], node_down["z_bed"] + epsilon)


def _interpolate_longitudinal_profiles(
    river_network: list[dict],
    ordered_indices: list[int],
    node_registry: dict,
    transform: tuple,
    shape: tuple,
    elevation_grid: xr.DataArray,
) -> None:
    """Phase 3: Interpolate river bed elevations along each segment.

    Calculates the bed elevation for every point along a river segment by
    interpolating between the start and end nodes, while ensuring a monotonic
    downhill slope.

    Args:
        river_network: List of river segment dictionaries.
        ordered_indices: Topologically sorted indices of river segments.
        node_registry: Registry of node elevations.
        transform: Affine transform of the grid.
        shape: Shape of the grid.
        elevation_grid: Elevation DataArray with CRS and transform (meters).

    Raises:
        ValueError: If the generated profile goes below sea level.
    """
    dem_grid = (
        elevation_grid.values[0] if elevation_grid.ndim == 3 else elevation_grid.values
    )

    for idx in ordered_indices:
        river = river_network[idx]
        node_start = node_registry[river["start_key"]]
        node_end = node_registry[river["end_key"]]

        # Calculate the depth at the start and end of the segment.
        d_start = node_start["z_surf"] - node_start["z_bed"]
        d_end = node_end["z_surf"] - node_end["z_bed"]

        cols = np.floor((river["xs"] - transform[2]) / transform[0]).astype(int)
        rows = np.floor((river["ys"] - transform[5]) / transform[4]).astype(int)
        sampled_z_m = dem_grid[rows, cols].copy()

        valid = np.isfinite(sampled_z_m)
        if not valid.all():
            sampled_z_m = np.interp(
                river["distances"], river["distances"][valid], sampled_z_m[valid]
            )

        # Linearly interpolate the depth along the river segment.
        d_max = river["distances"][-1] if river["distances"][-1] > 0 else 1e-6
        depth_array = d_start + (d_end - d_start) * (river["distances"] / d_max)

        # The initial bed profile is the terrain elevation minus the interpolated depth.
        profile_z_bed_m = sampled_z_m - depth_array

        # Smooth out small bumps in the terrain to prevent artificial "dams" in the river.
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

        # Snap the upstream boundary first to guide the accumulation cascade.
        profile_z_bed_m[0] = node_start["z_bed"]

        # Enforce downhill flow: each point must be no higher than the previous one,
        # and no lower than the final downstream node.
        for i in range(1, len(profile_z_bed_m)):
            profile_z_bed_m[i] = max(
                node_end["z_bed"], min(profile_z_bed_m[i], profile_z_bed_m[i - 1])
            )

        # Explicitly set the exact node boundary condition at the mouth.
        profile_z_bed_m[-1] = node_end["z_bed"]

        if (profile_z_bed_m + depth_array < -3).any():
            raise ValueError(
                f"The profile for river {river['idx']} generated elevations below sea level."
            )

        river["target_z"] = profile_z_bed_m


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
) -> tuple[xr.DataArray, xr.DataArray]:
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
    """
    # Accommodate both 2D and 3D shapes gracefully
    grid_shape = (shape[-2], shape[-1])
    global_burn_elevation = np.full(grid_shape, fill_value=np.inf, dtype=np.float32)
    global_burn_manning = np.full(grid_shape, fill_value=-np.inf, dtype=np.float32)

    for river in river_network:
        xs_m, ys_m = to_m(river["xs"], river["ys"])
        tree = cKDTree(np.c_[xs_m, ys_m])

        effective_width_m = river["width"]
        # Create a buffer around the river line to represent its width.
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
            all_touched=True,
            dtype=np.uint8,
        )
        p_rows, p_cols = np.where(mask == 1)

        p_xs_m, p_ys_m = to_m(
            transform[2] + (p_cols + 0.5) * transform[0],
            transform[5] + (p_rows + 0.5) * transform[4],
        )
        _, nearest = tree.query(np.c_[p_xs_m, p_ys_m])

        # We use np.fmin to ensure that if multiple rivers overlap, we take the lowest bed.
        global_burn_elevation[p_rows, p_cols] = np.fmin(
            global_burn_elevation[p_rows, p_cols], river["target_z"][nearest]
        )
        # For roughness, we take the maximum value if they overlap.
        global_burn_manning[p_rows, p_cols] = np.fmax(
            global_burn_manning[p_rows, p_cols], river["manning"]
        )

    update_mask = np.isfinite(global_burn_elevation)

    if elevation_grid.ndim == 3:
        elevation_grid.values[0, update_mask] = global_burn_elevation[update_mask]
    else:
        elevation_grid.values[update_mask] = global_burn_elevation[update_mask]

    out_mannings = manning_grid.copy()
    if out_mannings.ndim == 3:
        out_mannings.values[0, update_mask] = global_burn_manning[update_mask]
    else:
        out_mannings.values[update_mask] = global_burn_manning[update_mask]

    return elevation_grid, out_mannings


def burn_rivers(
    elevation_grid: xr.DataArray,
    manning_grid: xr.DataArray,
    rivers: gpd.GeoDataFrame,
    fill_first: bool = True,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Burns river networks into a Digital Elevation Model (DEM).

    This workflow modifies the terrain elevation and roughness grids to
    explicitly include river channels, ensuring they are hydrologically
    consistent (i.e., they always flow downhill).

    Args:
        elevation_grid: Elevation DataArray with CRS and transform (meters).
        manning_grid: Manning's n DataArray with CRS and transform (s/m^(1/3)).
        rivers: GeoDataFrame containing river geometries and attributes.
        fill_first: If True, fills depressions in the DEM before burning rivers.

    Returns:
        Tuple containing:
            - Modified elevation DataArray with rivers burned in (meters).
            - Modified Manning's n DataArray with river roughness values (s/m^(1/3)).
    """
    _validate_inputs(elevation_grid, manning_grid, rivers)

    # Sort rivers by topological stream order to ensure headwaters are processed before outlets.
    rivers: gpd.GeoDataFrame = rivers.sort_values(
        "topological_stream_order", ascending=True
    )  # ty:ignore[invalid-assignment]

    out_elevation = (
        fill_depressions(elevation_grid.copy()) if fill_first else elevation_grid.copy()
    )
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

    # Create river network and node registry
    river_network, node_registry = _initialize_network_and_nodes(
        rivers, out_elevation, transform, shape, is_geo, sampling_dist, to_m
    )

    # Use the sorted order from the GeoDataFrame throughout the workflow.
    ordered_indices = list(range(len(river_network)))

    # Ensure that the bed elevation always decreases in the downstream direction.
    _enforce_upstream_propagation(river_network, ordered_indices, node_registry)

    # Interpolate the bed elevation for every point along the river segments.
    _interpolate_longitudinal_profiles(
        river_network, ordered_indices, node_registry, transform, shape, out_elevation
    )

    # Actually modify the elevation and roughness grids.
    out_elevation, out_mannings = _burn(
        river_network, shape, transform, to_m, to_deg, out_elevation, manning_grid
    )

    return (
        out_elevation,
        out_mannings,
    )
