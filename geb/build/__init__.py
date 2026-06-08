"""This module contains the main setup for the GEB model.

Notes:
- All prices are in nominal USD (face value) for their respective years. That means that the prices are not adjusted for inflation.
"""

import inspect
import logging
import os
import time
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, cast

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyflwdir
import rasterio.features
import xarray as xr
import zarr
from affine import Affine
from rasterio.env import defenv
from scipy.ndimage import binary_dilation
from shapely.geometry import Point, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from geb import GEB_PACKAGE_DIR, __version__
from geb.build.data_catalog import DataCatalog
from geb.build.methods import build_method
from geb.build.version_updates import get_and_maybe_do_version_updates
from geb.workflows.io import (
    read_params,
    write_array,
    write_geom,
    write_params,
    write_table,
)
from geb.workflows.raster import (
    clip_region,
    clip_with_grid,
    create_temp_zarr,
    full_like,
    interpolate_na_along_dim as interpolate_na_along_dim,
    repeat_grid,
    snap_to_grid as snap_to_grid,
)

from ..workflows.io import (
    read_zarr,
    write_zarr,
)
from .modules import (
    Agents,
    Crops,
    Forcing,
    GroundWater,
    Hydrography,
    LandSurface,
    Observations,
)
from .modules.hydrography import (
    create_river_raster_from_river_lines,
    extend_rivers_into_pits_and_set_pit_type,
)
from .workflows.hydrography import (
    get_river_graph,
)

INIT_MULTIPLE_EXCLUDED_OUTLET_ISO3_CODES: frozenset[str] = frozenset(
    {
        "DZA",  # Algeria
        "EGY",  # Egypt
        "ESH",  # Western Sahara
        "LBY",  # Libya
        "MAR",  # Morocco
        "MLT",  # Malta
        "SDN",  # Sudan
        "TUN",  # Tunisia
    }
)  # these are areas that are included in the squared BBOX but that should be excluded

# Set environment options for robustness
GDAL_HTTP_ENV_OPTS = {
    "GDAL_HTTP_MAX_RETRY": "10",  # Number of retry attempts
    "GDAL_HTTP_RETRY_DELAY": "2",  # Delay (seconds) between retries
    "GDAL_HTTP_TIMEOUT": "30",  # Timeout in seconds
    "GDAL_CACHEMAX": 1 * 1024**3,  # 1 GB cache size
    "GDAL_MAX_BAND_COUNT": "200000",  # Increase max band count
}
defenv(**GDAL_HTTP_ENV_OPTS)

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"


@contextmanager
def suppress_logging_warning(logger: logging.Logger) -> Iterator[None]:
    """A context manager to suppress logging warning messages temporarily.

    Args:
        logger: The logger to suppress warnings for.

    Yields:
        None

    """
    current_level: int = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)  # Set level to ERROR to suppress WARNING messages
    try:
        yield
    finally:
        logger.setLevel(current_level)  # Restore the original logging level


def boolean_mask_to_graph(
    mask: npt.NDArray[np.bool_], connectivity: int = 4, **kwargs: npt.NDArray[Any]
) -> networkx.Graph:
    """Convert a boolean mask to an undirected NetworkX graph.

    Additional attributes can be passed as keyword arguments, which
    will be added as attributes to the nodes of the graph.

    Args:
        mask (xarray.DataArray or numpy.ndarray):
            Boolean mask where True values are nodes in the graph
        connectivity:
            4 for von Neumann neighborhood (up, down, left, right)
            8 for Moore neighborhood (includes diagonals)
        **kwargs:
            Additional attributes to be added to the nodes of the graph.
            Must be 2D arrays with the same shape as the mask.

    Returns:
        networkx.Graph:
            Undirected graph where nodes are (y,x) coordinates of True cells

    Raises:
        ValueError: If connectivity is not 4 or 8.
    """
    # check dtypes
    assert isinstance(mask, (np.ndarray))
    for item in kwargs.values():
        assert isinstance(item, (np.ndarray))

    # Create an empty undirected graph
    G = networkx.Graph()

    # Get indices of True cells
    y_indices, x_indices = np.where(mask)

    # Add all True cells as nodes
    for y, x in zip(y_indices, x_indices, strict=True):
        node_attrs = {}
        for key, array in kwargs.items():
            node_attrs[key] = array[y, x].item()
        node_attrs["yx"] = (y, x)
        G.add_node((y, x), **node_attrs)

    # Define neighbor directions for 4-connectivity
    if connectivity == 4:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    elif connectivity == 8:  # 8-connectivity
        directions = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),  # right, down, left, up
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]  # diagonals
    else:
        raise ValueError("Connectivity must be either 4 or 8.")

    # Connect neighboring cells
    for y, x in zip(y_indices, x_indices, strict=True):
        # Check each direction for a neighbor
        for dy, dx in directions:
            ny, nx = y + dy, x + dx

            # Check if neighbor is within bounds and is True
            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx]:
                # Add an edge between current cell and neighbor
                G.add_edge((y, x), (ny, nx))

    return G


def get_subbasin_id_from_coordinate(
    data_catalog: DataCatalog, lon: float, lat: float
) -> int:
    """Find the subbasin ID for a given coordinate.

    Args:
        data_catalog: Data catalog containing the MERIT basins.
        lon: Longitude of the point.
        lat: Latitude of the point.

    Returns:
        The COMID value for the subbasin containing the point.

    Raises:
        ValueError: If the point is not in a basin or in multiple basins.
    """
    # we select the basin that contains the point. To do so
    # we use a bounding box with the point coordinates, thus
    # xmin == xmax and ymin == ymax
    # geoparquet uses < and >, not <= and >=, so we need to add
    # a small value to the coordinates to avoid missing the point
    COMID = (
        data_catalog.fetch("merit_basins_catchments")
        .read(
            bbox=(lon - 10e-6, lat - 10e-6, lon + 10e-6, lat + 10e-6),
        )
        .set_index("COMID")
    )
    assert isinstance(COMID, gpd.GeoDataFrame)

    # get only the points where the point is inside the basin
    COMID = COMID[COMID.geometry.contains(Point(lon, lat))]

    if len(COMID) == 0:
        raise ValueError(
            f"The point is not in a basin. Note, that there are some holes in the MERIT basins dataset ({data_catalog.fetch('merit_basins_catchments').path}), ensure that the point is in a basin."
        )
    assert len(COMID) == 1, "The point is not in a single basin"
    # get the COMID value from the GeoDataFrame
    return COMID.index.values[0]


def get_sink_subbasin_id_for_geom(
    data_catalog: DataCatalog, geom: gpd.GeoDataFrame, river_graph: networkx.DiGraph
) -> list[int]:
    """Find all sink subbasins that intersect with the given geometry.

    This can be coastal basins, but also inland basins where the river
    flows out of the geometry. For example, if the geometry is a country
    boundary, all basins that intersect with the country boundary and
    have no downstream basin within the country are returned.

    Args:
        data_catalog: Data catalog containing the MERIT basins.
        geom: GeoDataFrame containing the geometry to find the sink subbasins for.
        river_graph: The river graph containing all subbasins and their connections.

    Returns:
        A list of COMID values for the sink subbasins.
    """
    subbasins = gpd.read_parquet(
        data_catalog.fetch("merit_basins_catchments").path,
        bbox=tuple([float(c) for c in geom.total_bounds]),
    ).set_index("COMID")

    subbasins = subbasins.iloc[
        subbasins.sindex.query(geom.union_all(), predicate="intersects")
    ]

    assert len(subbasins) > 0, "The geometry does not intersect with any subbasin."

    # create a subgraph containing only the selected subbasins
    region_river_graph = river_graph.subgraph(subbasins.index.tolist())

    # get all subbasins with no downstream subbasin (out degree is 0)
    # in the subgraph. These are the sink subbasins
    sink_nodes: list[int] = [
        COMID_ID
        for COMID_ID, out_degree in region_river_graph.out_degree(
            region_river_graph.nodes
        )
        if out_degree == 0
    ]

    return sink_nodes


def get_all_downstream_subbasins_in_geom(
    data_catalog: DataCatalog,
    geom: gpd.GeoDataFrame,
    ocean_outlets_only: bool,
    logger: logging.Logger,
) -> list[int]:
    """Find all downstream subbasins (with NextDownID = 0) that intersect with the given geometry.

    Args:
        data_catalog: Data catalog containing the MERIT basins.
        geom: GeoDataFrame containing the geometry to find the downstream subbasins for.
        ocean_outlets_only: If True, only include subbasins that flow to the ocean.
        logger: Logger for progress tracking.

    Returns:
        A list of COMID values for all downstream subbasins in the geometry.
    """
    logger.info("Finding subbasins that intersect with geometry...")

    # Get all subbasins that intersect with the geometry
    subbasins = gpd.read_parquet(
        data_catalog.fetch("merit_basins_catchments").path,
        bbox=tuple([float(c) for c in geom.total_bounds]),
    ).set_index("COMID")

    subbasins = subbasins.iloc[
        subbasins.sindex.query(geom.union_all(), predicate="intersects")
    ]

    assert len(subbasins) > 0, "The geometry does not intersect with any subbasin."

    logger.info(f"Found {len(subbasins)} subbasins intersecting with geometry")
    logger.info("Loading river network data...")

    # Get river network data to find downstream basins (NextDownID = 0)
    river_network = (
        data_catalog.fetch("merit_basins_rivers")
        .read(columns=["COMID", "NextDownID", "uparea"])
        .set_index("COMID")
    )
    assert isinstance(river_network, pd.DataFrame)

    logger.info("Filtering for downstream subbasins (NextDownID = 0)...")

    # Filter for subbasins that are within the geometry and are downstream (NextDownID = 0)
    intersecting_subbasins = subbasins.index.intersection(river_network.index)
    downstream_subbasins = river_network.loc[intersecting_subbasins]
    downstream_subbasins = downstream_subbasins[downstream_subbasins["NextDownID"] == 0]

    if ocean_outlets_only:
        logger.info("Filtering for ocean outlets only (NextDownID = 0)...")
        coastlines = data_catalog.fetch("open_street_map_coastlines").read()
        coastlines = coastlines.cx[
            subbasins.total_bounds[0] : subbasins.total_bounds[2],
            subbasins.total_bounds[1] : subbasins.total_bounds[3],
        ]
        candidates = subbasins.loc[downstream_subbasins.index]
        # Buffer distance expressed in degrees (assumes geographic CRS); used to
        # capture subbasins that are close to, but not exactly on, the coastline.
        buffer_distance_deg = 0.1
        buffered = candidates.geometry.buffer(buffer_distance_deg)
        mask = buffered.intersects(coastlines.union_all())
        # Get verified IDs
        downstream_subbasins = candidates.index[mask].tolist()
        logger.info(f"Found {len(downstream_subbasins)} downstream subbasins (outlets)")
        return downstream_subbasins
    else:
        logger.info(f"Found {len(downstream_subbasins)} downstream subbasins (outlets)")
        return downstream_subbasins.index.tolist()


def get_subbasin_upstream_areas(
    data_catalog: DataCatalog, subbasin_ids: list[int]
) -> dict[int, float]:
    """Get upstream areas for a list of subbasins with optimized batch loading.

    Args:
        data_catalog: Data catalog containing the MERIT basins.
        subbasin_ids: List of COMID values to get upstream areas for.

    Returns:
        Dictionary mapping COMID to upstream area in km2.
    """
    # Use filters to only read the rows we need - much faster than reading all data
    river_network = (
        data_catalog.fetch("merit_basins_rivers")
        .read(columns=["COMID", "uparea"], filters=[("COMID", "in", subbasin_ids)])
        .set_index("COMID")
    )
    assert isinstance(river_network, pd.DataFrame)

    # Convert to dict for faster lookup and handle missing values
    upstream_areas = river_network["uparea"].to_dict()

    # Fill in any missing subbasins with 0.0 area
    for subbasin_id in subbasin_ids:
        if subbasin_id not in upstream_areas:
            upstream_areas[subbasin_id] = 0.0

    return upstream_areas


def _load_merit_catchments(
    data_catalog: DataCatalog, subbasin_ids: Iterable[int], columns: list[str]
) -> gpd.GeoDataFrame:
    """Load selected MERIT Basins catchments from the data catalog.

    Args:
        data_catalog: Data catalog containing the MERIT Basins catchments.
        subbasin_ids: COMID values to load.
        columns: Additional column names to read.

    Returns:
        GeoDataFrame indexed by COMID.

    Raises:
        ValueError: If no COMID values are provided or requested COMIDs are missing.
    """
    unique_subbasin_ids: list[int] = sorted(
        {int(subbasin_id) for subbasin_id in subbasin_ids}
    )
    if not unique_subbasin_ids:
        raise ValueError("At least one subbasin ID is required.")

    requested_columns: list[str] = ["COMID"]
    for column_name in columns:
        if column_name != "COMID" and column_name not in requested_columns:
            requested_columns.append(column_name)

    catchments: gpd.GeoDataFrame = gpd.read_parquet(
        data_catalog.fetch("merit_basins_catchments").path,
        filters=[("COMID", "in", unique_subbasin_ids)],
        columns=requested_columns,
    ).set_index("COMID")

    missing_subbasin_ids: set[int] = set(unique_subbasin_ids) - set(catchments.index)
    if missing_subbasin_ids:
        preview_missing_ids: list[int] = sorted(missing_subbasin_ids)[:10]
        raise ValueError(
            "MERIT Basins catchments are missing "
            f"{len(missing_subbasin_ids)} COMID values, including {preview_missing_ids}."
        )

    return catchments


def get_init_multiple_region_geometry(
    geometry_bounds: str, region_shapefile: str | None, working_directory: Path
) -> gpd.GeoDataFrame:
    """Create the region geometry used to find init-multiple outlet basins.

    Args:
        geometry_bounds: Bounding box as ``xmin,ymin,xmax,ymax`` in degrees.
        region_shapefile: Optional path to a region shapefile.
        working_directory: Working directory used to resolve relative shapefile paths.

    Returns:
        Region geometry in EPSG:4326.

    Raises:
        FileNotFoundError: If the region shapefile does not exist.
        ValueError: If geometry_bounds format is invalid.
    """
    if region_shapefile:
        region_shapefile_path: Path = Path(region_shapefile).expanduser()
        if not region_shapefile_path.is_absolute():
            region_shapefile_path = working_directory / region_shapefile_path
        if not region_shapefile_path.exists():
            raise FileNotFoundError(
                f"Region shapefile not found at: {region_shapefile_path}"
            )
        region_geometry: gpd.GeoDataFrame = gpd.read_file(region_shapefile_path)
    else:
        bounds: list[float] = [
            float(value.strip()) for value in geometry_bounds.split(",")
        ]
        if len(bounds) != 4:
            raise ValueError(
                "Invalid geometry_bounds format. Expected 'xmin,ymin,xmax,ymax'."
            )
        region_geometry = gpd.GeoDataFrame(geometry=[box(*bounds)], crs="EPSG:4326")

    if region_geometry.crs != "EPSG:4326":
        region_geometry = region_geometry.to_crs("EPSG:4326")
    return region_geometry


def remove_init_multiple_excluded_outlets(
    data_catalog: DataCatalog,
    clusters: list[list[int]],
    excluded_iso3_codes: Iterable[str] = INIT_MULTIPLE_EXCLUDED_OUTLET_ISO3_CODES,
) -> tuple[list[list[int]], list[int]]:
    """Remove init-multiple outlet COMIDs that intersect excluded countries.

    Args:
        data_catalog: Data catalog containing MERIT Basins catchments and GADM level 0 countries.
        clusters: Outlet COMID values grouped by cluster.
        excluded_iso3_codes: ISO3 country codes whose outlet COMIDs should be removed.

    Returns:
        Filtered clusters and sorted removed outlet COMID values.

    Raises:
        ValueError: If no clusters are provided.
    """
    if not clusters:
        raise ValueError("At least one cluster is required.")

    outlet_ids: list[int] = sorted(
        {int(subbasin_id) for cluster in clusters for subbasin_id in cluster}
    )
    excluded_iso3_code_set: set[str] = {
        str(iso3_code).upper() for iso3_code in excluded_iso3_codes
    }
    if not excluded_iso3_code_set:
        return clusters, []

    countries: gpd.GeoDataFrame = data_catalog.fetch("GADM_level0").read(
        columns=["GID_0", "geometry"]
    )
    excluded_countries: gpd.GeoDataFrame = countries[
        countries["GID_0"].isin(excluded_iso3_code_set)
    ]
    if excluded_countries.empty:
        return clusters, []

    outlet_catchments: gpd.GeoDataFrame = _load_merit_catchments(
        data_catalog, outlet_ids, columns=["geometry"]
    )
    if excluded_countries.crs != outlet_catchments.crs:
        excluded_countries = excluded_countries.to_crs(outlet_catchments.crs)

    excluded_geometry: BaseGeometry = excluded_countries.union_all()
    removed_ids: set[int] = set(
        int(subbasin_id)
        for subbasin_id in outlet_catchments.index[
            outlet_catchments.geometry.intersects(excluded_geometry)
        ]
    )
    if not removed_ids:
        return clusters, []

    filtered_clusters: list[list[int]] = []
    for cluster in clusters:
        filtered_cluster: list[int] = [
            int(subbasin_id)
            for subbasin_id in cluster
            if int(subbasin_id) not in removed_ids
        ]
        if filtered_cluster:
            filtered_clusters.append(filtered_cluster)

    return filtered_clusters, sorted(removed_ids)


def _get_upstream_subbasin_ids(
    river_graph: networkx.DiGraph, outlet_id: int
) -> set[int]:
    """Get all subbasins upstream of an outlet, including the outlet.

    Args:
        river_graph: Directed river graph with edges from upstream to downstream.
        outlet_id: Outlet COMID value.

    Returns:
        COMID values for the outlet watershed.
    """
    upstream_subbasin_ids: set[int] = set(networkx.ancestors(river_graph, outlet_id))
    upstream_subbasin_ids.add(int(outlet_id))
    return upstream_subbasin_ids


def _get_cluster_upstream_subbasins(
    clusters: list[list[int]], river_graph: networkx.DiGraph
) -> list[list[int]]:
    """Get full upstream watershed COMIDs for each outlet cluster.

    Args:
        clusters: Outlet COMID values grouped by cluster.
        river_graph: Directed river graph with edges from upstream to downstream.

    Returns:
        Upstream COMID values for each cluster, sorted for reproducible output.
    """
    upstream_cache: dict[int, set[int]] = {}
    cluster_upstream_subbasins: list[list[int]] = []

    for cluster in clusters:
        watershed_subbasins: set[int] = set()
        for outlet_id in cluster:
            if outlet_id not in upstream_cache:
                upstream_cache[outlet_id] = _get_upstream_subbasin_ids(
                    river_graph, outlet_id
                )
            watershed_subbasins.update(upstream_cache[outlet_id])
        cluster_upstream_subbasins.append(sorted(watershed_subbasins))

    return cluster_upstream_subbasins


def _nearest_outlet_to_cluster(
    cluster_outlet_ids: list[int],
    candidate_outlet_ids: set[int],
    outlet_to_index: dict[int, int],
    outlet_coordinates: npt.NDArray[np.float64],
    distance_matrix: npt.NDArray[np.float64] | None,
) -> tuple[int, float]:
    """Find the nearest candidate outlet to any outlet in a cluster.

    Args:
        cluster_outlet_ids: COMID values already in the cluster.
        candidate_outlet_ids: Candidate COMID values not yet clustered.
        outlet_to_index: Mapping from outlet COMID to coordinate array row.
        outlet_coordinates: Coordinate array with longitude and latitude in degrees.
        distance_matrix: Optional pairwise distance matrix in degrees.

    Returns:
        The nearest candidate COMID and its distance in degrees.

    Raises:
        ValueError: If no candidate outlets are provided.
    """
    if not candidate_outlet_ids:
        raise ValueError("At least one candidate outlet is required.")

    cluster_indices: npt.NDArray[np.int64] = np.array(
        [outlet_to_index[outlet_id] for outlet_id in cluster_outlet_ids],
        dtype=np.int64,
    )
    candidate_ids: list[int] = sorted(candidate_outlet_ids)
    candidate_indices: npt.NDArray[np.int64] = np.array(
        [outlet_to_index[outlet_id] for outlet_id in candidate_ids],
        dtype=np.int64,
    )

    if distance_matrix is not None:
        candidate_distances: npt.NDArray[np.float64] = distance_matrix[
            np.ix_(cluster_indices, candidate_indices)
        ].min(axis=0)
    else:
        cluster_coordinates: npt.NDArray[np.float64] = outlet_coordinates[
            cluster_indices
        ]
        candidate_coordinates: npt.NDArray[np.float64] = outlet_coordinates[
            candidate_indices
        ]
        coordinate_differences: npt.NDArray[np.float64] = (
            cluster_coordinates[:, np.newaxis, :]
            - candidate_coordinates[np.newaxis, :, :]
        )
        distances: npt.NDArray[np.float64] = np.sqrt(
            np.sum(coordinate_differences * coordinate_differences, axis=2)
        )
        candidate_distances = distances.min(axis=0)

    nearest_position: int = int(candidate_distances.argmin())
    return candidate_ids[nearest_position], float(candidate_distances[nearest_position])


def calculate_bbox_efficiency(
    outlet_ids: list[int],
    upstream_areas: dict[int, float],
    river_graph: networkx.DiGraph,
    subbasin_centroids: dict[int, tuple[float, float]],
    subbasin_areas: dict[int, float] | None = None,
) -> float:
    """Calculate how compact a cluster watershed is for rectangular downloads.

    Args:
        outlet_ids: Outlet COMID values in the cluster.
        upstream_areas: Upstream area by outlet COMID (km2).
        river_graph: Directed river graph with edges from upstream to downstream.
        subbasin_centroids: Centroid coordinates by COMID in longitude/latitude degrees.
        subbasin_areas: Optional subbasin areas by COMID (km2).

    Returns:
        Efficiency ratio from 0.0 to 1.0.
    """
    watershed_subbasins: set[int] = set()
    for outlet_id in outlet_ids:
        watershed_subbasins.update(_get_upstream_subbasin_ids(river_graph, outlet_id))

    watershed_coordinates: list[tuple[float, float]] = [
        subbasin_centroids[subbasin_id]
        for subbasin_id in watershed_subbasins
        if subbasin_id in subbasin_centroids
    ]
    if not watershed_coordinates:
        return 1.0

    total_upstream_area_km2: float = sum(
        upstream_areas.get(outlet_id, 0.0) for outlet_id in outlet_ids
    )

    if subbasin_areas is not None:
        watershed_area_km2: float = sum(
            subbasin_areas.get(subbasin_id, 0.0) for subbasin_id in watershed_subbasins
        )
        if watershed_area_km2 < 1.0:
            return 1.0
        return min(1.0, total_upstream_area_km2 / watershed_area_km2)

    longitudes: list[float] = [
        longitude for longitude, _latitude in watershed_coordinates
    ]
    latitudes: list[float] = [
        latitude for _longitude, latitude in watershed_coordinates
    ]
    latitude_span_km: float = (max(latitudes) - min(latitudes)) * 111.0
    mean_latitude_deg: float = (min(latitudes) + max(latitudes)) / 2.0
    longitude_span_km: float = (max(longitudes) - min(longitudes)) * (
        111.320 * abs(np.cos(np.deg2rad(mean_latitude_deg)))
    )
    bbox_area_km2: float = latitude_span_km * longitude_span_km
    if bbox_area_km2 < 1.0:
        return 1.0

    return min(1.0, total_upstream_area_km2 / bbox_area_km2)


def cluster_subbasins_following_coastline(
    data_catalog: DataCatalog,
    subbasin_ids: list[int],
    target_area_km2: float,
    logger: logging.Logger,
    river_graph: networkx.DiGraph,
    max_distance_km: float = 100.0,  # Maximum distance in km for cluster growth
    min_bbox_efficiency: float = 0.99,  # Minimum bounding box efficiency (0-1) for ERA5 download optimization
) -> list[list[int]]:
    """Cluster outlet basins by adding entire upstream watersheds of nearby outlets.

    This function creates clusters by:
    1. Starting from an outlet basin (NextDownID=0) - adds its entire upstream watershed
    2. Finding the nearest remaining outlet basin to the current cluster
    3. Adding that outlet's entire upstream watershed to the cluster
    4. Repeating until target area is reached, then starting a new cluster

    The clustering uses bounding box efficiency to prevent thin/diagonal clusters that would
    waste bandwidth when downloading ERA5 climate data (which uses rectangular bounding boxes).

    Args:
        data_catalog: Data catalog containing the MERIT basins.
        subbasin_ids: List of outlet basin COMID values (NextDownID=0) to cluster.
        target_area_km2: Target cumulative upstream area per cluster (e.g., 800,000 km²).
        logger: Logger for progress tracking.
        river_graph: River graph for finding upstream subbasins to calculate true bbox efficiency.
        max_distance_km: Maximum distance in km for adding outlets to a cluster (default: 100km).
        min_bbox_efficiency: Minimum ratio of cluster_area/bbox_area (default: 0.25 = 25%).
            Higher values create more compact clusters. Lower values allow more elongated shapes.

    Returns:
        List of clusters, where each cluster is a list of outlet basin COMID values.

    Raises:
        ValueError: If the outlet list or numeric clustering options are invalid.
    """
    if not subbasin_ids:
        raise ValueError("At least one downstream subbasin is required.")
    if target_area_km2 <= 0:
        raise ValueError("target_area_km2 must be greater than zero.")
    if max_distance_km <= 0:
        raise ValueError("max_distance_km must be greater than zero.")
    if not 0 <= min_bbox_efficiency <= 1:
        raise ValueError("min_bbox_efficiency must be between 0 and 1.")

    outlet_ids: list[int] = [int(subbasin_id) for subbasin_id in subbasin_ids]
    logger.info(f"Clustering {len(outlet_ids)} outlet basins by upstream watersheds...")
    logger.info(f"Target area per cluster: {target_area_km2:,.0f} km²")
    logger.info(f"Maximum cluster growth distance: {max_distance_km:.0f} km")
    logger.info(
        f"Minimum bounding box efficiency: {min_bbox_efficiency:.1%} "
        "(for ERA5 download optimization)"
    )

    outlet_catchments: gpd.GeoDataFrame = _load_merit_catchments(
        data_catalog, outlet_ids, columns=["geometry"]
    )
    upstream_areas: dict[int, float] = get_subbasin_upstream_areas(
        data_catalog, outlet_ids
    )

    logger.info("Computing outlet basin centroids...")
    outlet_coordinates: npt.NDArray[np.float64] = np.zeros((len(outlet_ids), 2))
    outlet_to_index: dict[int, int] = {}
    for outlet_index, outlet_id in enumerate(outlet_ids):
        outlet_centroid: Point = outlet_catchments.loc[outlet_id].geometry.centroid
        outlet_coordinates[outlet_index] = [outlet_centroid.x, outlet_centroid.y]
        outlet_to_index[outlet_id] = outlet_index

    all_upstream_subbasins: set[int] = set()
    for outlet_id in outlet_ids:
        all_upstream_subbasins.update(
            _get_upstream_subbasin_ids(river_graph, outlet_id)
        )

    logger.info(
        f"Loading centroids for {len(all_upstream_subbasins)} total subbasins "
        "(outlets + upstream)..."
    )
    all_catchments: gpd.GeoDataFrame = _load_merit_catchments(
        data_catalog, all_upstream_subbasins, columns=["geometry", "unitarea"]
    )
    all_subbasin_centroids: dict[int, tuple[float, float]] = {}
    all_subbasin_areas: dict[int, float] = {}
    for subbasin_id, row in all_catchments.iterrows():
        subbasin_id_value: Any = subbasin_id
        subbasin_id_int: int = int(subbasin_id_value)
        centroid: Point = row.geometry.centroid
        all_subbasin_centroids[subbasin_id_int] = (
            float(centroid.x),
            float(centroid.y),
        )
        all_subbasin_areas[subbasin_id_int] = float(row.unitarea)

    if len(outlet_ids) < 1000:
        coordinate_differences: npt.NDArray[np.float64] = (
            outlet_coordinates[:, np.newaxis, :] - outlet_coordinates[np.newaxis, :, :]
        )
        distance_matrix: npt.NDArray[np.float64] | None = np.sqrt(
            np.sum(coordinate_differences * coordinate_differences, axis=2)
        )
        logger.info(f"Pre-computed distance matrix for {len(outlet_ids)} outlets.")
    else:
        distance_matrix = None
        logger.info("Computing outlet distances on demand.")

    max_distance_degrees: float = max_distance_km / 111.0
    logger.info(
        f"Distance threshold: {max_distance_degrees:.2f}° (~{max_distance_km:.0f} km)"
    )

    clusters: list[list[int]] = []
    remaining_outlet_ids: set[int] = set(outlet_ids)
    previous_cluster: list[int] | None = None
    cluster_number: int = 1
    start_time_s: float = time.time()

    while remaining_outlet_ids:
        processed_outlets: int = len(outlet_ids) - len(remaining_outlet_ids)
        progress_percent: float = (processed_outlets / len(outlet_ids)) * 100
        logger.info(
            f"\nCluster {cluster_number}: {len(remaining_outlet_ids)} outlets "
            f"remaining ({progress_percent:.1f}% complete)"
        )

        if previous_cluster is None:
            start_outlet: int = max(
                remaining_outlet_ids,
                key=lambda outlet_id: (
                    outlet_coordinates[outlet_to_index[outlet_id], 1],
                    outlet_coordinates[outlet_to_index[outlet_id], 0],
                ),
            )
            logger.info(f"  Starting outlet: {start_outlet} (top-right corner)")
        else:
            start_outlet, start_distance_degrees = _nearest_outlet_to_cluster(
                previous_cluster,
                remaining_outlet_ids,
                outlet_to_index,
                outlet_coordinates,
                distance_matrix,
            )
            logger.info(
                f"  Starting outlet: {start_outlet} "
                f"(nearest to previous cluster, dist={start_distance_degrees:.3f}°)"
            )

        current_cluster: list[int] = [start_outlet]
        current_area_km2: float = upstream_areas.get(start_outlet, 0.0)
        remaining_outlet_ids.remove(start_outlet)
        logger.info(f"  Initial area: {current_area_km2:,.0f} km²")

        while current_area_km2 < target_area_km2 and remaining_outlet_ids:
            current_efficiency: float = calculate_bbox_efficiency(
                current_cluster,
                upstream_areas,
                river_graph,
                all_subbasin_centroids,
                all_subbasin_areas,
            )
            if current_efficiency < min_bbox_efficiency:
                logger.info(
                    "  Current cluster bbox efficiency too low "
                    f"({current_efficiency:.1%} < {min_bbox_efficiency:.1%}); "
                    "stopping cluster."
                )
                break

            nearest_outlet, nearest_distance_degrees = _nearest_outlet_to_cluster(
                current_cluster,
                remaining_outlet_ids,
                outlet_to_index,
                outlet_coordinates,
                distance_matrix,
            )
            if nearest_distance_degrees > max_distance_degrees:
                logger.info(
                    f"  Nearest outlet is {nearest_distance_degrees:.3f}° away "
                    f"(>{max_distance_degrees:.3f}°); stopping cluster."
                )
                break

            candidate_cluster: list[int] = current_cluster + [nearest_outlet]
            candidate_efficiency: float = calculate_bbox_efficiency(
                candidate_cluster,
                upstream_areas,
                river_graph,
                all_subbasin_centroids,
                all_subbasin_areas,
            )
            if candidate_efficiency < min_bbox_efficiency:
                logger.info(
                    f"  Adding outlet {nearest_outlet} would reduce bbox efficiency "
                    f"to {candidate_efficiency:.1%} (< {min_bbox_efficiency:.1%}); "
                    "stopping cluster."
                )
                break

            outlet_area_km2: float = upstream_areas.get(nearest_outlet, 0.0)
            current_cluster.append(nearest_outlet)
            current_area_km2 += outlet_area_km2
            remaining_outlet_ids.remove(nearest_outlet)
            if len(current_cluster) <= 10 or len(current_cluster) % 5 == 0:
                logger.info(
                    f"  Added outlet {nearest_outlet} "
                    f"(area: {outlet_area_km2:,.0f} km², "
                    f"dist: {nearest_distance_degrees:.3f}°) - "
                    f"Total: {current_area_km2:,.0f} km² "
                    f"[{len(current_cluster)} outlets]"
                )

        clusters.append(current_cluster)
        previous_cluster = current_cluster
        logger.info(
            f"Completed cluster {cluster_number}: "
            f"{len(current_cluster)} outlets, {current_area_km2:,.0f} km²"
        )
        cluster_number += 1

    total_time_s: float = time.time() - start_time_s
    logger.info(
        f"\nClustering complete! Created {len(clusters)} clusters "
        f"in {total_time_s:.1f}s ({total_time_s / 60:.1f} min)"
    )

    logger.info("\nPost-processing clusters...")
    small_cluster_threshold_km2: float = 0.10 * target_area_km2
    cluster_areas_km2: list[float] = [
        float(sum(upstream_areas.get(outlet_id, 0.0) for outlet_id in cluster))
        for cluster in clusters
    ]
    clusters_to_remove: set[int] = set()
    for cluster_index, cluster_area_km2 in enumerate(cluster_areas_km2):
        if (
            len(clusters) <= 1
            or cluster_index in clusters_to_remove
            or cluster_area_km2 >= small_cluster_threshold_km2
        ):
            continue

        nearest_cluster_index: int | None = None
        nearest_distance_degrees: float = float("inf")
        for candidate_index, candidate_cluster in enumerate(clusters):
            if (
                candidate_index == cluster_index
                or candidate_index in clusters_to_remove
            ):
                continue
            _unused_outlet_id, distance_degrees = _nearest_outlet_to_cluster(
                clusters[cluster_index],
                set(candidate_cluster),
                outlet_to_index,
                outlet_coordinates,
                distance_matrix,
            )
            if distance_degrees < nearest_distance_degrees:
                nearest_distance_degrees = distance_degrees
                nearest_cluster_index = candidate_index

        if nearest_cluster_index is None:
            logger.warning(f"Could not merge small cluster {cluster_index}.")
            continue

        clusters[nearest_cluster_index].extend(clusters[cluster_index])
        clusters_to_remove.add(cluster_index)
        logger.info(
            f"Merged small cluster {cluster_index} "
            f"({cluster_area_km2:,.0f} km2) into cluster {nearest_cluster_index} "
            f"at {nearest_distance_degrees * 111.0:.1f} km"
        )

    if clusters_to_remove:
        clusters = [
            cluster
            for cluster_index, cluster in enumerate(clusters)
            if cluster_index not in clusters_to_remove
        ]

    assigned_outlet_ids: set[int] = {
        outlet_id for cluster in clusters for outlet_id in cluster
    }
    missing_outlet_ids: set[int] = set(outlet_ids) - assigned_outlet_ids
    if not missing_outlet_ids:
        logger.info("No unclustered outlets found.")
    elif not clusters:
        clusters = [[outlet_id] for outlet_id in sorted(missing_outlet_ids)]
    else:
        logger.info(f"Assigning {len(missing_outlet_ids)} unclustered outlets.")
        for outlet_id in sorted(missing_outlet_ids):
            nearest_cluster_index = min(
                range(len(clusters)),
                key=lambda cluster_index: _nearest_outlet_to_cluster(
                    [outlet_id],
                    set(clusters[cluster_index]),
                    outlet_to_index,
                    outlet_coordinates,
                    distance_matrix,
                )[1],
            )
            clusters[nearest_cluster_index].append(outlet_id)
            logger.info(
                f"Assigned outlet {outlet_id} to cluster {nearest_cluster_index} "
                f"(area: {upstream_areas.get(outlet_id, 0.0):,.0f} km2)"
            )
    logger.info(f"\n=== FINAL RESULT: {len(clusters)} clusters ===")

    return clusters


def save_clusters_to_geoparquet(
    clusters: list[list[int]],
    data_catalog: DataCatalog,
    output_path: Path,
    cluster_prefix: str = "cluster",
) -> None:
    """Save clusters to a geoparquet file with cluster IDs.

    Args:
        clusters: List of clusters, where each cluster is a list of COMID values.
        data_catalog: Data catalog containing the MERIT basins.
        output_path: Path where to save the geoparquet file.
        cluster_prefix: Prefix for cluster names.

    Raises:
        ValueError: If no clusters are provided.
    """
    print(f"Saving clusters to geoparquet: {output_path}")

    all_subbasin_ids: list[int] = [
        int(subbasin_id) for cluster in clusters for subbasin_id in cluster
    ]
    if not all_subbasin_ids:
        raise ValueError("At least one clustered subbasin is required.")

    print(f"Loading geometries for {len(all_subbasin_ids)} subbasins...")
    subbasins: gpd.GeoDataFrame = _load_merit_catchments(
        data_catalog, all_subbasin_ids, columns=["geometry"]
    )

    print("Loading upstream areas...")
    upstream_areas: dict[int, float] = get_subbasin_upstream_areas(
        data_catalog, all_subbasin_ids
    )

    print("Pre-calculating cluster areas...")
    cluster_areas: dict[int, float] = {}
    for cluster_idx, cluster_subbasins in enumerate(clusters):
        cluster_areas[cluster_idx] = float(
            sum(
                upstream_areas.get(subbasin_id, 0.0)
                for subbasin_id in cluster_subbasins
            )
        )

    print("Creating cluster assignments...")
    cluster_data: list[dict[str, Any]] = []
    for cluster_idx, cluster_subbasins in enumerate(clusters):
        cluster_id: str = f"{cluster_prefix}_{cluster_idx:03d}"
        cluster_area_km2: float = cluster_areas[cluster_idx]
        for subbasin_id in cluster_subbasins:
            cluster_data.append(
                {
                    "COMID": int(subbasin_id),
                    "cluster_id": cluster_id,
                    "cluster_number": cluster_idx,
                    "cluster_area_km2": cluster_area_km2,
                    "subbasin_area_km2": upstream_areas.get(subbasin_id, 0),
                    "geometry": subbasins.loc[subbasin_id, "geometry"],
                }
            )

    print("Creating GeoDataFrame...")
    cluster_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(cluster_data, crs=subbasins.crs)

    print("Saving to geoparquet...")
    write_geom(cluster_gdf, output_path)
    print(
        f"Saved {len(cluster_data)} subbasins in {len(clusters)} clusters to {output_path}"
    )


def _union_geometries(
    geometries: Iterable[BaseGeometry], chunk_size: int = 5000
) -> BaseGeometry:
    """Merge geometries into one exact outline geometry.

    Args:
        geometries: Polygon or multipolygon geometries to dissolve.
        chunk_size: Number of geometries per union chunk.

    Returns:
        Dissolved outline geometry.

    Raises:
        ValueError: If no geometries are provided.
    """
    geometry_list: list[BaseGeometry] = list(geometries)
    if not geometry_list:
        raise ValueError("At least one geometry is required to create an outline.")

    if len(geometry_list) <= chunk_size:
        merged_geometry: BaseGeometry = unary_union(geometry_list)
    else:
        chunk_unions: list[BaseGeometry] = [
            unary_union(geometry_list[start_index : start_index + chunk_size])
            for start_index in range(0, len(geometry_list), chunk_size)
        ]
        merged_geometry = unary_union(chunk_unions)

    if not merged_geometry.is_valid:
        # Repair only invalid topology; this is not a simplification step.
        merged_geometry = merged_geometry.buffer(0)

    return merged_geometry


def _geometry_area_km2(geometry: BaseGeometry, crs: Any) -> float:
    """Calculate geometry area in an equal-area projection.

    Args:
        geometry: Geometry to measure.
        crs: Coordinate reference system of the geometry.

    Returns:
        Geometry area (km2).
    """
    projected_geometry: gpd.GeoSeries = gpd.GeoSeries([geometry], crs=crs).to_crs(
        "EPSG:6933"
    )
    area_m2: float = float(projected_geometry.area.iloc[0])
    return area_m2 / 1e6


def create_cluster_outline_geodataframe(
    clusters: list[list[int]],
    data_catalog: DataCatalog,
    river_graph: networkx.DiGraph,
    cluster_prefix: str = "cluster",
) -> gpd.GeoDataFrame:
    """Create one dissolved basin outline for each outlet cluster.

    Each output geometry is the exact union of all upstream MERIT Basins
    catchments that drain to the outlets in the cluster. The result may be a
    Polygon or MultiPolygon, but each cluster is represented by exactly one row.

    Args:
        clusters: Outlet COMID values grouped by cluster.
        data_catalog: Data catalog containing the MERIT Basins catchments.
        river_graph: Directed river graph with edges from upstream to downstream.
        cluster_prefix: Prefix for cluster names.

    Returns:
        GeoDataFrame with one exact outline geometry per cluster.

    Raises:
        ValueError: If no clusters are provided.
    """
    if not clusters:
        raise ValueError("At least one cluster is required.")

    cluster_upstream_subbasins: list[list[int]] = _get_cluster_upstream_subbasins(
        clusters, river_graph
    )
    all_upstream_subbasin_ids: list[int] = sorted(
        {
            subbasin_id
            for upstream_subbasins in cluster_upstream_subbasins
            for subbasin_id in upstream_subbasins
        }
    )

    print(
        "Loading geometries for "
        f"{len(all_upstream_subbasin_ids)} unique upstream subbasins..."
    )
    catchments: gpd.GeoDataFrame = _load_merit_catchments(
        data_catalog, all_upstream_subbasin_ids, columns=["geometry"]
    )

    print(f"Creating exact outlines for {len(clusters)} clusters...")
    outline_records: list[dict[str, Any]] = []
    for cluster_index, (outlet_ids, upstream_subbasin_ids) in enumerate(
        zip(clusters, cluster_upstream_subbasins, strict=True)
    ):
        cluster_id: str = f"{cluster_prefix}_{cluster_index:03d}"
        cluster_geometries: gpd.GeoDataFrame = catchments.loc[upstream_subbasin_ids]
        outline_geometry: BaseGeometry = _union_geometries(
            cluster_geometries.geometry.values
        )
        total_basin_area_km2: float = _geometry_area_km2(
            outline_geometry, catchments.crs
        )
        outline_records.append(
            {
                "cluster_id": cluster_id,
                "cluster_number": cluster_index,
                "num_outlet_subbasins": len(outlet_ids),
                "num_total_subbasins": len(upstream_subbasin_ids),
                "total_basin_area_km2": total_basin_area_km2,
                "geometry_type": outline_geometry.geom_type,
                "geometry": outline_geometry,
            }
        )
        print(
            f"  {cluster_id}: {len(upstream_subbasin_ids)} subbasins, "
            f"{total_basin_area_km2:,.0f} km2"
        )

    return gpd.GeoDataFrame(outline_records, crs=catchments.crs)


def create_cluster_visualization_map(
    cluster_outlines: gpd.GeoDataFrame,
    output_path: str | Path,
    figsize: tuple[int, int] = (20, 16),
) -> None:
    """Create a visualization map from precomputed cluster outlines.

    Args:
        cluster_outlines: GeoDataFrame with one outline geometry per cluster.
        output_path: Path where to save the PNG file.
        figsize: Figure size in inches.

    Raises:
        ValueError: If no cluster outlines are provided.
    """
    from matplotlib.patches import Patch

    if cluster_outlines.empty:
        raise ValueError("At least one cluster outline is required.")

    print(f"Creating cluster visualization map: {output_path}")
    cluster_count: int = len(cluster_outlines)
    colors = plt.cm.tab20(np.linspace(0, 1, min(cluster_count, 20)))  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
    if cluster_count > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, cluster_count))  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]

    cluster_outlines_mercator: gpd.GeoDataFrame = cluster_outlines.to_crs(epsg=3857)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    print("Plotting clusters...")
    for _row_index, row in cluster_outlines_mercator.iterrows():
        cluster_number: int = int(row.cluster_number)
        gpd.GeoSeries([row.geometry], crs=cluster_outlines_mercator.crs).plot(
            ax=ax,
            color=colors[cluster_number],
            alpha=0.6,
            edgecolor="black",
            linewidth=2.5,
        )

        centroid: Point = row.geometry.centroid
        ax.annotate(
            f"{cluster_number}",
            (centroid.x, centroid.y),
            fontsize=20,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="black",
                alpha=0.8,
                edgecolor="white",
                linewidth=2,
            ),
        )

    print("Adding background map...")
    ctx.add_basemap(
        ax,
        crs=cluster_outlines_mercator.crs,
        source=ctx.providers.CartoDB.Positron,  # ty:ignore[unresolved-attribute]
        alpha=0.5,
        zoom="auto",
    )
    ax.set_title(
        f"Basin Clusters - {cluster_count} Clusters",
        fontsize=22,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Longitude", fontsize=16)
    ax.set_ylabel("Latitude", fontsize=16)

    legend_elements: list[Patch] = [
        Patch(
            facecolor=colors[cluster_number],
            alpha=0.6,
            edgecolor="black",
            linewidth=2,
            label=f"Cluster {cluster_number}",
        )
        for cluster_number in range(cluster_count)
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=12,
        framealpha=0.9,
        title="Clusters",
        title_fontsize=14,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    print(f"Saving map to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved cluster visualization map to {output_path}")


def create_multi_basin_configs(
    clusters: list[list[int]],
    working_directory: Path,
    config: Path = Path("model.yml"),
    build_config: Path = Path("build.yml"),
    update_config: Path = Path("update.yml"),
    cluster_prefix: str = "cluster",
    from_example: str = "geul",
    cluster_basin_areas_km2: dict[int, float] | None = None,
) -> list[Path]:
    """Create separate config files and directories for each cluster of subbasins.

    Args:
        clusters: List of clusters, where each cluster is a list of COMID values.
        working_directory: Working directory for the models (init_multiple_dir).
        config: Path to the model configuration file to create.
        build_config: Path to the model build configuration file to create.
        update_config: Path to the model update configuration file to create.
        cluster_prefix: Prefix for cluster directory names.
        from_example: Example model name to inherit build and model settings from.
        cluster_basin_areas_km2: Optional basin area by cluster number (km2).

    Returns:
        List of paths to created cluster directories.

    Raises:
        FileNotFoundError: If the requested example model files do not exist.
    """
    print(f"Creating configuration files for {len(clusters)} clusters...")
    working_directory.mkdir(parents=True, exist_ok=True)
    example_directory: Path = GEB_PACKAGE_DIR / "examples" / from_example
    example_model_config_path: Path = example_directory / "model.yml"
    example_build_config_path: Path = example_directory / "build.yml"
    example_update_config_path: Path = example_directory / "update.yml"
    if not example_model_config_path.exists():
        raise FileNotFoundError(
            f"Example model.yml not found: {example_model_config_path}"
        )
    if not example_build_config_path.exists():
        raise FileNotFoundError(
            f"Example build.yml not found: {example_build_config_path}"
        )
    if not example_update_config_path.exists():
        raise FileNotFoundError(
            f"Example update.yml not found: {example_update_config_path}"
        )

    config_path: Path = working_directory / config
    build_config_path: Path = working_directory / build_config
    update_config_path: Path = working_directory / update_config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    build_config_path.parent.mkdir(parents=True, exist_ok=True)
    update_config_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating {build_config_path.name} in main init multiple directory...")
    build_config_path.write_text(
        f'inherits: "{{GEB_PACKAGE_DIR}}/examples/{from_example}/build.yml"\n'
    )

    print(f"Created {build_config_path.name} in {working_directory}")
    print(f"Creating {config_path.name} in init_multiple_dir directory...")
    print(f"Reading model configuration from: {example_model_config_path}")
    config_path.write_text(example_model_config_path.read_text())

    print(f"Created {config_path.name} in {working_directory}")
    print(f"Creating {update_config_path.name} in init_multiple_dir directory...")
    update_config_path.write_text(
        f'inherits: "{{GEB_PACKAGE_DIR}}/examples/{from_example}/update.yml"\n'
    )

    print(f"Created {update_config_path.name} in {working_directory}")

    print("Creating cluster directories and configuration files...")
    cluster_directories: list[Path] = []
    for cluster_index, cluster in enumerate(clusters):
        print(
            f"Creating cluster {cluster_index + 1}/{len(clusters)}: "
            f"{cluster_prefix}_{cluster_index:03d} ({len(cluster)} subbasins)"
        )

        cluster_dir: Path = working_directory / f"{cluster_prefix}_{cluster_index:03d}"
        base_dir: Path = cluster_dir / "base"
        base_dir.mkdir(parents=True, exist_ok=True)
        cluster_directories.append(cluster_dir)

        relative_build_config: str = os.path.relpath(build_config_path, base_dir)
        base_build_config_path: Path = base_dir / "build.yml"
        base_build_config: dict[str, str] = {"inherits": relative_build_config}
        write_params(base_build_config, base_build_config_path)

        relative_update_config: str = os.path.relpath(update_config_path, base_dir)
        base_update_config_path: Path = base_dir / "update.yml"
        base_update_config: dict[str, str] = {"inherits": relative_update_config}
        write_params(base_update_config, base_update_config_path)

        relative_model_config: str = os.path.relpath(config_path, base_dir)
        base_model_config_path: Path = base_dir / "model.yml"
        cluster_subbasin_ids: list[int] = [int(subbasin_id) for subbasin_id in cluster]
        total_basin_area_km2: float | None = (
            cluster_basin_areas_km2.get(cluster_index)
            if cluster_basin_areas_km2 is not None
            else None
        )
        if total_basin_area_km2 is not None:
            print(f"  Basin area: {total_basin_area_km2:,.0f} km²")

        model_config: dict[str, Any] = {
            "inherits": relative_model_config,
            "general": {"region": {"subbasin": cluster_subbasin_ids}},
        }
        if total_basin_area_km2 is not None:
            model_config["basin"] = {"total_area_km2": round(total_basin_area_km2, 2)}

        write_params(model_config, base_model_config_path)

        print(
            f"  Created configuration files in {base_dir.relative_to(working_directory)}"
        )

    print(
        f"Successfully created {len(clusters)} cluster configurations in {working_directory}"
    )

    return cluster_directories


def save_clusters_as_merged_geometries(
    cluster_outlines: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Save precomputed cluster outlines to a GeoParquet file.

    Args:
        cluster_outlines: GeoDataFrame with one outline geometry per cluster.
        output_path: Path where to save the geoparquet file.

    Raises:
        ValueError: If no cluster outlines are provided.
    """
    if cluster_outlines.empty:
        raise ValueError("At least one cluster outline is required.")

    print(f"Saving to {output_path}...")
    write_geom(cluster_outlines, output_path)


def get_coastline_nodes(
    coastline_graph: networkx.Graph,
    riverine_mask: xr.DataArray,
    STUDY_AREA_OUTFLOW: int,
    NEARBY_OUTFLOW: int,
) -> set:
    """Get all coastline nodes that are part of the coastline for the study area.

    Args:
        coastline_graph: The graph containing all coastline nodes.
        riverine_mask: A DataArray containing the riverine mask.
        STUDY_AREA_OUTFLOW: The outflow type value for outflows within the study area.
        NEARBY_OUTFLOW: The outflow type value for outflows outside the study area, but close enough to influence the coastline.

    Returns:
        A set of all coastline nodes that are part of the coastline for the study area.

    Raises:
        AssertionError: If a coastal segment has both a study area outflow and a nearby outflow, but not exactly one of each.
    """
    coastline_nodes = set()

    for island in networkx.connected_components(coastline_graph):
        island = coastline_graph.subgraph(island)

        # extract the outflow nodes from the island, these should be included for sure
        island_outflow_nodes_study_area = set(
            [
                node
                for node, attrs in island.nodes(data=True)
                if attrs["outflow_type"] == STUDY_AREA_OUTFLOW
            ]
        )
        island_outflow_nodes_nearby = set(
            [
                node
                for node, attrs in island.nodes(data=True)
                if attrs["outflow_type"] == NEARBY_OUTFLOW
            ]
        )

        # no outflows found, we can skip this island
        if len(island_outflow_nodes_study_area) == 0:
            continue

        # if there is only one outflow node in the study area and no nearby
        # outflow nodes, it is an island, with only this outflow node
        # so we can add all nodes to the coastline
        if (
            len(island_outflow_nodes_study_area) == 1
            and len(island_outflow_nodes_nearby) == 0
        ):
            coastline_nodes.update(island.nodes)
            continue

        for node in island.nodes:
            island.nodes[node]["neighbor_of_study_area_outflow"] = False
            island.nodes[node]["neighbor_of_nearby_outflow"] = False

        island_with_outflow_cuts = island.copy()
        for node in island_outflow_nodes_study_area:
            neighbors = island.neighbors(node)
            for neighbor in neighbors:
                island_with_outflow_cuts.nodes[neighbor][
                    "neighbor_of_study_area_outflow"
                ] = True
                island_with_outflow_cuts.nodes[neighbor]["study_area_outflow_node"] = (
                    node
                )

        for node in island_outflow_nodes_nearby:
            neighbors = island.neighbors(node)
            for neighbor in neighbors:
                island_with_outflow_cuts.nodes[neighbor][
                    "neighbor_of_nearby_outflow"
                ] = True
                island_with_outflow_cuts.nodes[neighbor]["nearby_outflow_node"] = node

        island_with_outflow_cuts.remove_nodes_from(island_outflow_nodes_study_area)
        island_with_outflow_cuts.remove_nodes_from(island_outflow_nodes_nearby)

        for coastal_segment_nodes in networkx.connected_components(
            island_with_outflow_cuts
        ):
            coastal_segment = island_with_outflow_cuts.subgraph(
                coastal_segment_nodes
            ).copy()
            assert len(coastal_segment) == coastal_segment.number_of_nodes()

            study_area_nodes = [
                (node, attr)
                for node, attr in coastal_segment.nodes(data=True)
                if attr["neighbor_of_study_area_outflow"] is True
            ]

            nearby_nodes = [
                (node, attr)
                for node, attr in coastal_segment.nodes(data=True)
                if attr["neighbor_of_nearby_outflow"] is True
            ]

            outflow_neighbor_types = [
                attrs["neighbor_type"]
                for _, attrs in coastal_segment.nodes(data=True)
                if "neighbor_type" in attrs
            ]
            assert len(outflow_neighbor_types) <= 2

            # in case the segment has both a study area outflow and a nearby outflow
            # we divide the segment in a part that is closer to the study area outflow
            # and a part that is not
            if study_area_nodes and nearby_nodes:
                if len(study_area_nodes) != 1 or len(nearby_nodes) != 1:
                    # create a diagnostic visualization grid that marks, for this coastal
                    # segment, which cells neighbor the study area outflow, which neighbor
                    # the nearby outflow, and which belong to neither. This is written to
                    # disk to help debug cases where there is not exactly one study area
                    # outflow and one nearby outflow per coastal segment.
                    nodes_grid: xr.DataArray = riverine_mask.copy().astype(np.int32)
                    nodes_grid.attrs["_FillValue"] = 0
                    nodes_grid.values[:] = 0
                    for node, node_attributes in coastal_segment.nodes(data=True):
                        node_y, node_x = node_attributes["yx"]
                        if node_attributes["neighbor_of_nearby_outflow"] is True:
                            nodes_grid.values[node_y, node_x] = NEARBY_OUTFLOW
                        elif node_attributes["neighbor_of_study_area_outflow"] is True:
                            nodes_grid.values[node_y, node_x] = STUDY_AREA_OUTFLOW
                        else:
                            nodes_grid.values[node_y, node_x] = -1

                    # clip to area of interest for smaller output
                    nodes_grid, _ = clip_with_grid(nodes_grid, mask=nodes_grid != 0)

                    debug_file = Path("debug_coastal_segment.zarr")
                    write_zarr(
                        nodes_grid,
                        debug_file,
                        crs=nodes_grid.rio.crs,
                    )
                    raise AssertionError(
                        f"There should only be one study area outflow and one nearby outflow per coastal segment, "
                        f"found {len(study_area_nodes)} study area outflows and {len(nearby_nodes)} nearby outflows. "
                        f"Debug output written to {debug_file}."
                    )

                study_area_node = study_area_nodes[0]
                nearby_node = nearby_nodes[0]

                for node in coastal_segment.nodes:
                    # find distance to the nearest outflow node. If the path
                    # to the study area node is shorter, we add the node to the
                    # coastline nodes. If the path to the nearby node is shorter,
                    # we don't add the node to the coastline nodes.
                    path_length_to_study_area_node = len(
                        networkx.shortest_path(
                            coastline_graph, source=node, target=study_area_node[0]
                        )
                    )
                    path_length_to_nearby_node = len(
                        networkx.shortest_path(
                            coastline_graph, source=node, target=nearby_node[0]
                        )
                    )

                    # In case of break even, we need to take some special steps. We
                    # only want to add the node in one of the two datasets, if we
                    # also build another model for the nearby outflow. To make this
                    # decision, we need to use parameters that behave identically
                    # for both models. Therefore we use the relative xy location
                    # of the outflow nodes.
                    if path_length_to_study_area_node == path_length_to_nearby_node:
                        study_area_outflow_node = study_area_node[1][
                            "study_area_outflow_node"
                        ]
                        nearby_outflow_node = nearby_node[1]["nearby_outflow_node"]
                        if study_area_outflow_node[0] >= nearby_outflow_node[0]:
                            if study_area_outflow_node[0] == nearby_outflow_node[0]:
                                if study_area_outflow_node[0] > nearby_outflow_node[0]:
                                    coastline_nodes.add(node)
                                else:
                                    pass
                            else:
                                coastline_nodes.add(node)
                        else:
                            pass

                    elif path_length_to_study_area_node < path_length_to_nearby_node:
                        coastline_nodes.add(node)
                    else:
                        pass

            # if the segment has only a study area outflow, we add all nodes to
            # the coastline nodes
            elif study_area_nodes:
                coastline_nodes.update(coastal_segment.nodes)
            # if the segment has only a nearby outflows, we don't add any nodes to
            # the coastline nodes
            else:
                pass

    return coastline_nodes


def create_riverine_mask(
    ldd: xr.DataArray, ldd_network: pyflwdir.FlwdirRaster, geom: gpd.GeoDataFrame
) -> xr.DataArray:
    """Create a riverine mask from the ldd and the river network.

    Args:
        ldd : The local drainage direction (ldd) data array.
        ldd_network : The flow direction raster created from the ldd data.
        geom : The geometry of the riverine basin, which is used to clip the mask.

    Returns:
        A boolean mask where True indicates riverine cells and False indicates non-riverine cells.

    """
    riverine_mask = full_like(
        ldd,
        fill_value=True,
        nodata=False,
        dtype=bool,
    ).compute()

    riverine_mask = riverine_mask.rio.clip([geom.union_all()], drop=False)

    # because the basin mask from the source is not perfect and has some holes
    # we need to extend the riverine mask to include all cells upstream of any
    # riverine cell. This is done by creating a flow raster from the masked
    # ldd, find all the pits within the riverine mask, and then get all upstream
    # cells from these pits.
    ldd_network_masked = pyflwdir.from_array(
        ldd.values,
        ftype="d8",
        mask=riverine_mask.values,
        transform=ldd.rio.transform(recalc=True),
        latlon=True,  # hydrography is specified in latlon
    )

    # set all cells that are upstream of a pit to True
    riverine_mask.values[ldd_network.basins(idxs=ldd_network_masked.idxs_pit) > 0] = (
        True
    )

    # MERIT-Basins rivers don't always extend all the way into coastline pits.
    # To extend these rivers, we first find all downstream cells of the area
    # within the initial riverine mask. Then we find all pits within these
    # downstream cells, and set these pits to be part of the riverine mask.
    downstream_indices_of_area_in_mask = ldd_network.idxs_ds[
        riverine_mask.values.ravel()
    ]
    all_pits_in_area = ldd_network.idxs_pit
    downstream_indices_that_are_pits = np.intersect1d(
        downstream_indices_of_area_in_mask, all_pits_in_area
    )
    riverine_mask.values.ravel()[downstream_indices_that_are_pits] = True

    return riverine_mask


class DelayedReader(dict):
    """A dictionary that reads data from files only when accessed.

    This is useful because some datasets are very large and reading them
    all at once would require a lot of memory. Furthermore, when updating the model
    it is usually not required to have all data in memory. This class allows to
    read data only when it is actually needed.

    When setting an item, we should not set the actual data, but the file path.
    """

    def __init__(self, reader: Any) -> None:
        """Initialize the DelayedReader with a reader function.

        Args:
            reader: A function that takes a file path and returns the data.
        """
        self.reader: Any = reader

    def __getitem__(self, key: str) -> Any:
        """Get item from the dictionary using reader.

        Args:
            key: dictionary key

        Returns:
            The data read from the file.
        """
        fp: str | Path = super().__getitem__(key)
        return self.reader(fp)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item in the dictionary using a file path that refers to the actual data.

        The file path is stored in the dictionary, not the actual data.

        Args:
            key: dictionary key
            value: file path to the data

        Raises:
            ValueError: If the value is not a string or Path.
        """
        if isinstance(value, (str, Path)):
            super().__setitem__(key, value)
        else:
            raise ValueError("Value must be a file path (str or Path).")


class GEBModel(
    Hydrography, Forcing, Crops, LandSurface, Agents, GroundWater, Observations
):
    """Main GEB model build class.

    This class contains:
    - methods to setup the model region and grid
    - all general methods for example for saving and loading data, calling methods etc.
    - subclasses all build modules, which contain methods for building specific parts of the model.
    """

    def __init__(
        self,
        logger: logging.Logger,
        root: Path,
        epsg: int = 4326,
    ) -> None:
        """Initialize the GEB build model.

        Args:
            logger: Logger to use for logging.
            root: Root directory for the model build. If None, the current working directory is used.
            epsg: EPSG code for the model grid. Default is 4326 (WGS84).
        """
        self._logger = logger
        build_method.logger = logger

        Hydrography.__init__(self)
        Forcing.__init__(self)
        Crops.__init__(self)
        LandSurface.__init__(self)
        Agents.__init__(self)
        GroundWater.__init__(self)
        Observations.__init__(self)

        self.root = root
        self.epsg = epsg
        self._data_catalog = DataCatalog(logger=logger)

        # the grid, subgrid, and region subgrids are all datasets, which should
        # have exactly matching coordinates
        self.grid = xr.Dataset()
        self.subgrid = xr.Dataset()

        # all other data types are dictionaries because these entries don't
        # necessarily match the grid coordinates, shapes etc.
        self.geom = DelayedReader(reader=gpd.read_parquet)
        self.table = DelayedReader(reader=pd.read_parquet)
        self.array = DelayedReader(zarr.load)
        self.params = DelayedReader(reader=read_params)
        self.other = DelayedReader(reader=read_zarr)
        self.files = {}

    def set_version(self, version: str) -> None:
        """Set the version in the version file.

        Args:
            version: The version to set in the version file.
        """
        self.logger.info(f"Setting version in version file to: {version}")
        self.version_path.write_text(version)

    def set_current_version(self) -> None:
        """Set the current version in the version file."""
        self.logger.info(
            f"Setting version in version file to current version: {__version__}"
        )
        self.version_path.write_text(__version__)

    def version_is_current(self) -> bool:
        """Check if the version in the version file is the same as the current version.

        Returns:
            True if the version is current, False otherwise.
        """
        if not self.version_path.exists():
            return False
        version_info = self.version_path.read_text()
        return version_info == __version__

    @property
    def logger(self) -> logging.Logger:
        """Get the logger."""
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        self._logger = value
        build_method.logger = value

    @property
    def data_catalog(self) -> DataCatalog:
        """Get the new data catalog."""
        return self._data_catalog

    @data_catalog.setter
    def data_catalog(self, value: DataCatalog) -> None:
        self._data_catalog = value

    @property
    def grid(self) -> xr.Dataset:
        """Get the grid."""
        return self._grid

    @grid.setter
    def grid(self, value: xr.Dataset) -> None:
        self._grid = value

    @property
    def subgrid(self) -> xr.Dataset:
        """Get the subgrid."""
        return self._subgrid

    @subgrid.setter
    def subgrid(self, value: xr.Dataset) -> None:
        self._subgrid = value

    @property
    def geom(self) -> DelayedReader:
        """Get the geometry reader."""
        return self._geom

    @geom.setter
    def geom(self, value: DelayedReader) -> None:
        self._geom = value

    @property
    def table(self) -> DelayedReader:
        """Get the table reader."""
        return self._table

    @table.setter
    def table(self, value: DelayedReader) -> None:
        self._table = value

    @property
    def array(self) -> DelayedReader:
        """Get the array reader."""
        return self._array

    @array.setter
    def array(self, value: DelayedReader) -> None:
        self._array = value

    @property
    def params(self) -> DelayedReader:
        """Get the params reader."""
        return self._params

    @params.setter
    def params(self, value: DelayedReader) -> None:
        self._params = value

    @property
    def other(self) -> DelayedReader:
        """Get the other reader."""
        return self._other

    @other.setter
    def other(self, value: DelayedReader) -> None:
        self._other = value

    @property
    def files(self) -> dict:
        """Get the files dictionary."""
        return self._files

    @files.setter
    def files(self, value: dict) -> None:
        self._files = value

    @build_method(required=True)
    def setup_region(
        self,
        region: dict,
        subgrid_factor: int,
        resolution_arcsec: int = 30,
        include_coastal_area: bool = True,
    ) -> None:
        """Creates a 2D regular grid or reads an existing grid.

        An 2D regular grid will be created from a geometry (geom_fn) or bbox. If an existing
        grid is given, then no new grid will be generated.

        Adds/Updates model layers:
        * **grid** grid mask: add grid mask to grid object

        Args:
            region: Dictionary describing region of interest, e.g.:
                * {'basin': [x, y]}

                Region must be of kind [basin, subbasin].
            subgrid_factor: GEB implements a subgrid. This parameter determines the factor by which the subgrid is smaller than the original grid.
            resolution_arcsec: Resolution of the grid in arcseconds. Must be a multiple of 3 to align with MERIT.
            include_coastal_area: Because the subbasins are delinated using a minimum upstream area small basins near the coast are not included.
                If this parameter is set to True, the coastal area will be included in the riverine mask by automatically extending the riverine mask to the coastal area,
                by finding all coastal basins between the outlets within the study area and half the distance to the nearest outlet outside the study area.
                All cells upstream of these coastal basins will be included in the riverine mask.

        Raises:
            ValueError: If region is not understood.
        """
        assert resolution_arcsec % 3 == 0, (
            "resolution_arcsec must be a multiple of 3 to align with MERIT"
        )
        assert subgrid_factor >= 2

        self.logger.info("Loading river network.")
        river_graph = get_river_graph(self.data_catalog)

        self.logger.info("Finding sinks in river network of requested region.")
        if "subbasin" in region:
            if isinstance(region["subbasin"], list):
                sink_subbasin_ids: list[int] = region["subbasin"]
            else:
                sink_subbasin_ids: list[int] = [region["subbasin"]]
        elif "outflow" in region:
            lat, lon = region["outflow"]["lat"], region["outflow"]["lon"]
            sink_subbasin_ids: list[int] = [
                get_subbasin_id_from_coordinate(self.data_catalog, lon, lat)
            ]
        elif "geom" in region:
            regions = self.data_catalog.fetch(region["geom"]["source"]).read()
            assert isinstance(regions, gpd.GeoDataFrame)
            regions = regions[
                regions[region["geom"]["column"]] == region["geom"]["key"]
            ]
            sink_subbasin_ids: list[int] = get_sink_subbasin_id_for_geom(
                self.data_catalog, regions, river_graph
            )
        else:
            raise ValueError(f"Region {region} not understood.")

        self.logger.info(
            f"Found {len(sink_subbasin_ids)} sink subbasins in region {region}."
        )
        rivers: gpd.GeoDataFrame = self.get_rivers(river_graph, sink_subbasin_ids)

        buffer = 1.0  # buffer in degrees
        xmin, ymin, xmax, ymax = rivers[~rivers["is_downstream_outflow"]].total_bounds
        xmin -= buffer
        ymin -= buffer
        xmax += buffer
        ymax += buffer

        ldd = (
            self.data_catalog.fetch(
                "merit_hydro_dir",
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
            )
            .read()
            .compute()
        )
        assert isinstance(ldd, xr.DataArray), "Expected ldd to be an xarray DataArray."

        # We remove all pits that are not directly adjacent to valid flow directions
        # Identify cells with flow directions
        cells_with_flow_directions = (ldd.values > 0) & (ldd.values != 247)
        # grow valid values mask by one cell
        valid_values_and_coastline = binary_dilation(
            cells_with_flow_directions, structure=np.ones((3, 3))
        )
        # and use said mask which includes valid values and its neighbors to set ldd to no data
        ldd.values[~valid_values_and_coastline] = 247

        ldd_network = pyflwdir.from_array(
            ldd.values,
            ftype="d8",
            transform=ldd.rio.transform(recalc=True),
            latlon=True,
        )

        rivers: gpd.GeoDataFrame = extend_rivers_into_pits_and_set_pit_type(
            rivers, ldd_network, ldd.values
        )

        self.logger.info("Preparing 2D grid.")
        if "outflow" in region:
            # get basin geometry
            riverine_mask = full_like(
                ldd,
                fill_value=False,
                nodata=False,
                dtype=bool,
            )
            riverine_mask.values[ldd_network.basins(xy=(lon, lat)) > 0] = True
            rivers = rivers[~rivers["is_upstream_of_downstream_basin"]]
        elif "subbasin" in region or "geom" in region:
            rivers_outlets_for_basins = rivers[~rivers["is_further_downstream_outflow"]]
            outlet_lonlats = rivers_outlets_for_basins.geometry.apply(
                lambda geom: geom.coords[-2]
            ).tolist()
            subbasins_grid = ldd_network.basins(
                xy=(
                    [lon for lon, lat in outlet_lonlats],
                    [lat for lon, lat in outlet_lonlats],
                ),
                ids=rivers_outlets_for_basins.index,
            ).astype(np.int32)

            # we want to remove the areas upstream of the downstream outflow basins
            # that are not part of the study area.
            is_upstream_of_downstream_basin = rivers["is_upstream_of_downstream_basin"]

            # by setting these basins to 0 in the subbasins grid, they will be removed
            subbasins_grid[
                np.isin(subbasins_grid, rivers[is_upstream_of_downstream_basin].index)
            ] = 0

            # and we can now remove them from the rivers geodataframe as well. They have had
            # their purpose now.
            rivers = rivers[~is_upstream_of_downstream_basin]

            subbasins: list[tuple[dict[str, Any], float]] = list(
                rasterio.features.shapes(
                    subbasins_grid,
                    mask=subbasins_grid != 0,
                    transform=ldd_network.transform,
                    connectivity=8,
                )
            )
            subbasins: gpd.GeoDataFrame = (
                gpd.GeoDataFrame.from_records(
                    [
                        {
                            "COMID": int(value),
                            "geometry": shape(geom),
                        }
                        for geom, value in subbasins
                    ],
                )
                .set_index("COMID")
                .set_crs(4326)
            )
            subbasins["is_downstream_outflow"] = subbasins.index.isin(
                rivers[rivers["is_downstream_outflow"]].index
            )
            subbasins["is_coastal"] = subbasins.apply(
                lambda subbasin: rivers.loc[subbasin.name, "is_coastal"],
                axis=1,
            )

            geom = gpd.GeoDataFrame(
                geometry=[subbasins[~subbasins["is_downstream_outflow"]].union_all()],
                crs=subbasins.crs,
            )
            # ESPG 6933 (WGS 84 / NSIDC EASE-Grid 2.0 Global) is an equal area projection
            # while the shape of the polygons becomes vastly different, the area is preserved mostly.
            # usable between 86°S and 86°N.
            self.logger.info(
                f"Approximate riverine basin size: {round(geom.to_crs('ESRI:54009').area.sum() / 1e6, 2)} km2"
            )

            riverine_mask = create_riverine_mask(ldd, ldd_network, geom)
            assert not riverine_mask.attrs["_FillValue"]
        else:
            raise ValueError(f"Region {region} not understood.")

        self.set_geom(rivers, name="routing/rivers")
        self.set_geom(subbasins, name="routing/subbasins")

        if include_coastal_area and subbasins["is_coastal"].any():
            mask: xr.DataArray = self.extend_mask_to_coastal_area(
                ldd, riverine_mask, subbasins
            )
        else:
            mask: xr.DataArray = riverine_mask

        mask.attrs["_FillValue"] = None
        self.set_other(mask, name="drainage/mask")

        ldd: xr.DataArray = xr.where(
            mask,
            ldd,
            ldd.attrs["_FillValue"],
        )

        ldd_elevation = self.data_catalog.fetch(
            "merit_hydro_elv",
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        ).read()
        assert isinstance(ldd_elevation, xr.DataArray), (
            "Expected ldd_elevation to be an xarray DataArray."
        )

        assert ldd_elevation.shape == ldd.shape == mask.shape

        ldd_elevation: xr.DataArray = xr.where(
            mask,
            ldd_elevation,
            ldd_elevation.attrs["_FillValue"],
        )

        mask, ldd, ldd_elevation = clip_region(
            mask, ldd, ldd_elevation, align=30 / 60 / 60
        )

        self.set_other(ldd_elevation, name="drainage/original_d8_elevation")

        ldd: xr.DataArray = xr.where(
            mask,
            ldd,
            ldd.attrs["_FillValue"],
            keep_attrs=True,
        )
        ldd: xr.DataArray = self.set_other(
            ldd, name="drainage/original_d8_flow_directions"
        )

        self.derive_mask(ldd, ldd.rio.transform(recalc=True), resolution_arcsec)

        self.create_subgrid(subgrid_factor)

    def extend_mask_to_coastal_area(
        self,
        ldd: xr.DataArray,
        riverine_mask: xr.DataArray,
        subbasins: gpd.GeoDataFrame,
    ) -> xr.DataArray:
        """Extend the riverine mask to include coastal areas.

        This is done by finding all coastal basins between the outlets within the study area
        and half the distance to the nearest outlet outside the study area. All cells upstream of these
        coastal basins will be included in the riverine mask.

        Args:
            ldd: The local drainage direction (ldd) data array.
            riverine_mask: The initial riverine mask, which will be extended to the coastal area.
            subbasins: The subbasins used to create the initial riverine mask.

        Returns:
            The extended riverine mask.
        """
        flow_raster: pyflwdir.FlwdirRaster = pyflwdir.from_array(
            ldd.values,
            ftype="d8",
            transform=ldd.rio.transform(recalc=True),
            latlon=True,
        )

        STUDY_AREA_OUTFLOW: Literal[1] = 1
        NEARBY_OUTFLOW: Literal[2] = 2

        rivers = (
            self.data_catalog.fetch(
                "merit_basins_rivers",
            )
            .read(
                columns=["COMID", "lengthkm", "uparea", "maxup", "geometry"],
                bbox=ldd.rio.bounds(),
            )
            .set_index("COMID")
        )
        assert isinstance(rivers, gpd.GeoDataFrame), (
            "Expected rivers to be a GeoDataFrame."
        )

        rivers["outflow_type"] = rivers.apply(
            lambda row: (
                STUDY_AREA_OUTFLOW if row.name in subbasins.index else NEARBY_OUTFLOW
            ),
            axis=1,
        )

        river_raster_outflow_type = create_river_raster_from_river_lines(
            rivers, ldd, column="outflow_type"
        )
        river_raster_ID = create_river_raster_from_river_lines(rivers, ldd, index=True)
        river_raster_ID = river_raster_ID.ravel()

        downstream_indices = flow_raster.idxs_ds

        # MERIT-Basins rivers don't always extend all the way into coastline pits.
        # To extend these rivers, we first find the river (which have a river ID),
        # but have a no downstream cell.
        river_cells_with_downstream_pits = (downstream_indices != -1) & (
            river_raster_ID != -1
        )
        # For those cells, we extract the river ID, and set this river ID to the downstream cells.
        river_raster_ID[downstream_indices[river_cells_with_downstream_pits]] = (
            river_raster_ID[river_cells_with_downstream_pits]
        )
        river_raster_ID = river_raster_ID.reshape(ldd.shape)

        pits = ldd == 0
        # TODO: Filter non-coastline pits
        coastline = pits
        coastline.attrs["_FillValue"] = None

        # save coastline, for now mostly for debugging purposes
        self.set_other(coastline, name="drainage/full_coastline_in_bounding_box")

        pits = flow_raster.idxs_pit

        downstream_indices_study_area = flow_raster.idxs_ds[
            river_raster_outflow_type.ravel() == STUDY_AREA_OUTFLOW
        ]
        outflow_pits_study_area = np.intersect1d(pits, downstream_indices_study_area)

        downstream_indices_nearby = flow_raster.idxs_ds[
            river_raster_outflow_type.ravel() == NEARBY_OUTFLOW
        ]
        outflow_pits_nearby = np.intersect1d(pits, downstream_indices_nearby)

        outflows = np.full(ldd.size, -1, dtype=np.int8)
        outflows[outflow_pits_study_area] = STUDY_AREA_OUTFLOW
        outflows[outflow_pits_nearby] = NEARBY_OUTFLOW
        outflows = outflows.reshape(ldd.shape)

        outflow_da = self.full_like(
            ldd,
            fill_value=False,
            nodata=-1,
            dtype=bool,
        )
        outflow_da.values = outflows
        self.set_other(outflow_da, name="drainage/outflows")

        # first we create a graph with all coastline cells. Neighbouring cells
        # are connected. From this graph we want to select all coastline
        # cells in between river outlets. The riverine mask does not include
        # any of the coastal cells. So we first grow the riverine mask by one cell
        # and use this to set the outflow nodes in the graph.
        coastline_graph = boolean_mask_to_graph(
            coastline.values,
            connectivity=4,
            outflow_type=outflows,
            river_id=river_raster_ID,
        )
        coastline_nodes = get_coastline_nodes(
            coastline_graph,
            riverine_mask,
            STUDY_AREA_OUTFLOW=STUDY_AREA_OUTFLOW,
            NEARBY_OUTFLOW=NEARBY_OUTFLOW,
        )

        # here we go from the graph back to a mask. We do this by creating a new mask
        # and setting coastline cells to true
        simulated_coastline_da = self.full_like(
            ldd,
            fill_value=False,
            nodata=None,
            dtype=bool,
        )
        simulated_coastline = np.zeros(
            ldd.shape,
            dtype=bool,
        )

        for node in coastline_nodes:
            yx = coastline_graph.nodes[node]["yx"]
            simulated_coastline[yx] = True

        simulated_coastline_da.values = simulated_coastline

        # save the connected coastline, for now mostly for debugging purposes
        self.set_other(simulated_coastline_da, name="drainage/simulated_coastline")

        # get all upstream cells from the selected coastline
        flow_raster = pyflwdir.from_array(ldd.values, ftype="d8")
        coastal_mask = (
            flow_raster.basins(idxs=np.where(simulated_coastline_da.values.ravel())[0])
            > 0
        )

        # return the combination of the riverine mask and the coastal mask
        return riverine_mask | coastal_mask

    def derive_mask(
        self,
        d8_original: xr.DataArray,
        transform: Affine,
        target_resolution_arcsec: int,
    ) -> None:
        """Derives the model grid mask from the original D8 flow directions.

        The model grid mask is derived by upscaling the original D8 flow directions
        to the model resolution using the IHU method (Iterative Hydrography Upscaling).
        The model grid mask is True for all cells that are part of the upscaled flow
        directions, and False for all other cells.

        Args:
            d8_original: The original D8 flow directions.
            transform: The affine transform of the original D8 flow directions.
            target_resolution_arcsec: The resolution of the target low resolution model grid in arcseconds.
        """
        assert d8_original.dtype == np.uint8

        d8_original_data = d8_original.values
        flow_raster = pyflwdir.from_array(
            d8_original_data,
            ftype="d8",
            transform=transform,
            latlon=True,  # hydrography is specified in latlon
            mask=d8_original_data
            != d8_original.attrs["_FillValue"],  # this mask is True within study area
        )

        # the original resolution is 3 arcseconds, thus we divide by 3
        scale_factor: int = target_resolution_arcsec // 3

        self.logger.info(
            "Coarsening drainage network to resolution %d arcseconds",
            target_resolution_arcsec,
        )
        # IHU = Iterative hydrography upscaling method, see https://doi.org/10.5194/hess-25-5287-2021
        flow_raster_upscaled, idxs_out = flow_raster.upscale(
            scale_factor=scale_factor,
            method="ihu",
        )
        flow_raster_upscaled.repair_loops()

        mask = xr.DataArray(
            ~flow_raster_upscaled.mask.reshape(flow_raster_upscaled.shape),
            coords={
                "y": flow_raster_upscaled.transform.f
                + flow_raster_upscaled.transform.e
                * (np.arange(flow_raster_upscaled.shape[0]) + 0.5),
                "x": flow_raster_upscaled.transform.c
                + flow_raster_upscaled.transform.a
                * (np.arange(flow_raster_upscaled.shape[1]) + 0.5),
            },
            dims=("y", "x"),
            name="mask",
            attrs={
                "_FillValue": None,
            },
        )
        self.set_grid(mask, name="mask")

        mask_geom = list(
            rasterio.features.shapes(
                mask.astype(np.uint8),
                mask=~mask,
                connectivity=8,
                transform=mask.rio.transform(recalc=True),
            ),
        )
        mask_geom = gpd.GeoDataFrame.from_features(
            [{"geometry": geom[0], "properties": {}} for geom in mask_geom],
            crs=4326,
        )

        self.set_geom(mask_geom, name="mask")

        flow_raster_idxs_ds = self.full_like(
            self.grid["mask"],
            fill_value=-1,
            nodata=-1,
            dtype=np.int32,
        )
        flow_raster_idxs_ds.name = "flow_raster_idxs_ds"
        flow_raster_idxs_ds.data = flow_raster_upscaled.idxs_ds.reshape(
            flow_raster_upscaled.shape
        )
        self.set_grid(flow_raster_idxs_ds, name="flow_raster_idxs_ds")

        idxs_out_da = self.full_like(
            self.grid["mask"],
            fill_value=-1,
            nodata=-1,
            dtype=np.int32,
        )
        idxs_out_da.name = "idxs_outflow"
        idxs_out_da.data = idxs_out
        self.set_grid(idxs_out_da, name="idxs_outflow")

    def create_subgrid(self, subgrid_factor: int) -> None:
        """Creates the model subgrid.Affine.

        The model subgrid is a higher resolution grid that is used for representing
        subgrid processes, such as different land uses within a grid cell, and
        unique agents.

        Args:
            subgrid_factor: The factor by which the subgrid is smaller than the original grid.
        """
        mask: xr.DataArray = self.grid["mask"]
        dst_transform: Affine = mask.rio.transform(recalc=True) * Affine.scale(
            1 / subgrid_factor
        )

        submask: xr.DataArray = xr.DataArray(
            data=repeat_grid(mask.data, subgrid_factor),
            coords={
                "y": dst_transform.f
                + np.arange(mask.shape[0] * subgrid_factor) * dst_transform.e,
                "x": dst_transform.c
                + np.arange(mask.shape[1] * subgrid_factor) * dst_transform.a,
            },
            attrs={"_FillValue": None},
        ).chunk({"x": -1, "y": -1})
        self.set_subgrid(submask, name="mask")

    @build_method(required=True)
    def set_time_range(self, start_date: date, end_date: date) -> None:
        """Sets the time range for the build model.

        This time range is used to ensure that all datasets with a time dimension
        cover at least this time range.

        Start date must be on or after 1960, because of data availability. End date can be in the future.

        Args:
            start_date: The start date of the model.
            end_date: The end date of the model.

        Raises:
            ValueError: If the start date is not before the end date.
            ValueError: If the start date is before 1960, because of data availability.
        """
        if not start_date < end_date:
            raise ValueError("Start date must be before end date.")

        if start_date.year < 1960:
            raise ValueError(
                "Start date must be on or after 1960, because of data availability."
            )
        self.set_params(
            {"start_date": start_date, "end_date": end_date},
            name="model_time_range",
        )

    @property
    def start_date(self) -> datetime:
        """Gets the start date of the build model.

        All datasets with a time dimension should cover at least from this date.

        Returns:
            The start date of the model.
        """
        start_date = self.params["model_time_range"]["start_date"]

        # TODO: This can be removed in 2026
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        return start_date

    @property
    def end_date(self) -> datetime:
        """Gets the end date of the build model.

        All datasets with a time dimension should cover at least until this date.

        Returns:
            The end date of the model.
        """
        end_date = self.params["model_time_range"]["end_date"]

        # TODO: This can be removed in 2026
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        return end_date

    @build_method(required=True)
    def set_ssp(self, ssp: str) -> None:
        """Sets the SSP name for the model.

        Args:
            ssp: The SSP name. Supported SSPs are: ssp1, ssp3, ssp5.
        """
        assert ssp in ["ssp1", "ssp3", "ssp5"], (
            f"SSP {ssp} not supported. Supported SSPs are: ssp1, ssp3, ssp5."
        )
        self.set_params({"ssp": ssp}, name="ssp")

    @property
    def ssp(self) -> str:
        """Gets the SSP name that was set using set_ssp, or the default SSP of "ssp3".

        Returns:
            The SSP name. If no SSP was set, returns "ssp3" (middle of the road).
        """
        return self.params["ssp"]["ssp"] if "ssp" in self.params else "ssp3"

    @property
    def ISIMIP_ssp(self) -> str:
        """Returns the ISIMIP SSP name.

        Raises:
            ValueError: If the SSP is not supported.
        """
        if self.ssp == "ssp1":
            return "ssp126"
        elif self.ssp == "ssp3":
            return "ssp370"
        elif self.ssp == "ssp5":
            return "ssp585"
        else:
            raise ValueError(f"SSP {self.ssp} not supported.")

    @build_method(required=False)
    def setup_precipitation_scaling_factors_for_return_periods(
        self, risk_scaling_factors: list[tuple[float, float]]
    ) -> None:
        """Sets up precipitation scaling factors for different return periods.

        Args:
            risk_scaling_factors: A list of tuples containing the exceedance probability
                and the corresponding scaling factor.
        """
        risk_scaling_factors_df = pd.DataFrame(
            risk_scaling_factors,
            columns=np.array(["exceedance_probability", "scaling_factor"]),
        )
        self.set_table(risk_scaling_factors_df, name="precipitation_scaling_factors")

    def set_table(self, table: pd.DataFrame, name: str, write: bool = True) -> None:
        """Set a table and save it to disk.

        Args:
            table: The table to save.
            name: The name of the table.
            write: Whether to write the table to disk. If False, the table
                is only added to the file library, but not written to disk.
        """
        fp: Path = Path("table") / (name + ".parquet")
        fp_with_root: Path = Path(self.root, fp)
        if write:
            self.logger.info(f"Writing file {fp}")

            self.files["table"][name] = fp

            fp_with_root.parent.mkdir(parents=True, exist_ok=True)
            write_table(table, fp_with_root)

        self.table[name] = fp_with_root

    def set_array(self, data: np.ndarray, name: str, write: bool = True) -> None:
        """Set an array and save it to disk.

        Args:
            data: The array to save.
            name: The name of the array.
            write: Whether to write the array to disk. If False, the array
                is only added to the file library, but not written to disk.
        """
        assert isinstance(data, np.ndarray)
        fp: Path = Path("array") / (name + ".zarr")
        fp_with_root: Path = Path(self.root, fp)

        if write:
            self.logger.info(f"Writing file {fp}")
            self.files["array"][name] = fp
            fp_with_root.parent.mkdir(parents=True, exist_ok=True)
            write_array(cast(Any, data), fp_with_root, compression_level=18)

        self.array[name] = fp_with_root

    def set_params(self, data: dict, name: str, write: bool = True) -> None:
        """Set a dictionary and save it to disk.

        Args:
            data: The dictionary to save.
            name: The name of the dictionary.
            write: Whether to write the dictionary to disk. If False, the dictionary
                is only added to the file library, but not written to disk.
        """
        fp: Path = Path("dict") / (name + ".yml")
        fp_with_root: Path = Path(self.root) / fp
        fp_with_root.parent.mkdir(parents=True, exist_ok=True)
        if write:
            self.logger.info(f"Writing file {fp}")

            self.files["dict"][name] = fp

            write_params(data, fp_with_root)

        self.params[name] = fp_with_root

    def set_geom(self, geom: gpd.GeoDataFrame, name: str, write: bool = True) -> None:
        """Set a geometry and save it to disk.

        Args:
            geom: The geometry to save.
            name: The name of the geometry.
            write: Whether to write the geometry to disk. If False, the geometry
                is only added to the file library, but not written to disk.
        """
        assert isinstance(geom, gpd.GeoDataFrame)
        fp: Path = Path("geom") / (name + ".geoparquet")
        fp_with_root: Path = self.root / fp
        if write:
            self.logger.info(f"Writing file {fp}")
            self.files["geom"][name] = fp
            fp_with_root.parent.mkdir(parents=True, exist_ok=True)
            write_geom(geom, fp_with_root)

        self.geom[name] = fp_with_root

    @property
    def files_path(self) -> Path:
        """Path to the files.yml file that contains the file library."""
        return Path(self.root, "files.yml")

    @property
    def progress_path(self) -> Path:
        """Path to the progress file that contains the build progress."""
        return Path(self.root, "progress.txt")

    def write_build_complete(self) -> None:
        """Writes a file that indicates that the build is complete."""
        self.build_complete_path.write_text("Build complete.")

    @property
    def build_complete_path(self) -> Path:
        """Path to the file that indicates that the build is complete."""
        return Path(self.root, "build_complete.txt")

    @property
    def version_path(self) -> Path:
        """Path to the version file that contains the build version."""
        return Path(self.root, "version.txt")

    def write_file_library(self) -> None:
        """Writes the file library to disk.

        Rather than overwriting the file library, we read the existing
        file library from disk and merge it with the new files, prioritizing
        the new files. This way, if another process has added files to the
        file library, we don't overwrite them.
        """
        file_library: dict = self.read_or_create_file_library()
        # merge file library from disk with new files, prioritizing new files
        for type_name, type_files in self.files.items():
            if type_name not in file_library:
                file_library[type_name] = type_files
            else:
                file_library[type_name].update(type_files)

        write_params(file_library, self.files_path)

    def read_or_create_file_library(self) -> dict:
        """Reads the file library from disk.

        If the file library does not exist, an empty file library is returned, but
        with all expected categories.

        Returns:
            A dictionary with the file library.
        """
        fp: Path = Path(self.files_path)
        if not fp.exists():
            return {
                "geom": {},
                "array": {},
                "table": {},
                "dict": {},
                "grid": {},
                "subgrid": {},
                "other": {},
            }
        else:
            files = read_params(self.files_path)

            # geoms was renamed to geom in the file library. To upgrade old models,
            # we check if "geoms" is in the files and rename it to "geom"
            # this line can be removed in august 2026 (also in geb/model.py)
            if "geoms" in files:
                files["geom"] = files.pop("geoms", {})
        return files

    def read_geom(self) -> None:
        """Reads all geometries from disk based on the file library."""
        for name, fn in self.files["geom"].items():
            self.geom[name] = Path(self.root, fn)

    def read_array(self) -> None:
        """Reads all arrays from disk based on the file library."""
        for name, fn in self.files["array"].items():
            self.array[name] = Path(self.root, fn)

    def read_table(self) -> None:
        """Reads all tables from disk based on the file library."""
        for name, fn in self.files["table"].items():
            self.table[name] = Path(self.root, fn)

    def read_params(self) -> None:
        """Reads all dictionaries from disk based on the file library."""
        for name, fn in self.files["dict"].items():
            self.params[name] = Path(self.root, fn)

    def read_grid(self) -> None:
        """Reads all grid data arrays from disk based on the file library."""
        # first read and set the mask. This is required.
        grid_files: dict[str, Path] = self.files["grid"]
        if len(grid_files) == 0:
            return
        mask: xr.DataArray = read_zarr(Path(self.root) / grid_files["mask"])
        self.set_grid(mask, name="mask", write=False)

        for name, fn in self.files["grid"].items():
            if name == "mask":  # mask already read
                continue
            data: xr.DataArray = read_zarr(Path(self.root) / fn)
            self.set_grid(data, name=name, write=False)

    def read_subgrid(self) -> None:
        """Reads all subgrid data arrays from disk based on the file library."""
        # first read and set the mask. This is required.
        subgrid_files: dict[str, Path] = self.files["subgrid"]
        if len(subgrid_files) == 0:
            return
        mask: xr.DataArray = read_zarr(Path(self.root) / subgrid_files["mask"])
        self.set_subgrid(mask, name="mask", write=False)
        for name, fn in self.files["subgrid"].items():
            if name == "mask":  # mask already read
                continue
            data: xr.DataArray = read_zarr(Path(self.root) / fn)
            self.set_subgrid(data, name=name, write=False)

    def read_other(self) -> None:
        """Reads all "other" data arrays from disk based on the file library."""
        for name, fn in self.files["other"].items():
            self.other[name] = Path(self.root, fn)

    def read(self) -> None:
        """Reads all data from disk based on the file library.

        Useful when continuing a build from an existing folder.
        """
        self.files = self.read_or_create_file_library()
        with suppress_logging_warning(self.logger):
            self.read_geom()
            self.read_array()
            self.read_table()
            self.read_params()

            self.read_subgrid()
            self.read_grid()

            self.read_other()

    def set_other(
        self,
        da: xr.DataArray,
        name: str,
        write: bool = True,
        time_chunksize: int = 1,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Set a data array that does not fit into other categories.

        The data array is saved to disk as a zarr file. Unlike the other gridded
        methods, this category does not assume any specific coordinate system
        or dimension order, and does not combine multiple layers into a single dataset.

        Args:
            da: The data array to set.
            name: The name of the data array.
            write: If True, write the data array to disk. Defaults to True.
            time_chunksize: The chunk size in the time dimension for writing to zarr.
                Defaults to 1.
            **kwargs: Additional keyword arguments to pass to write_zarr.

        Returns:
            The data array that was set. If write=True, this is the data array read from
            disk, so it is not the same object as the input data array.
        """
        assert isinstance(da, xr.DataArray)

        if write:
            self.logger.info(f"Write {name}")

            fp: Path = Path("other") / (name + ".zarr")
            self.files["other"][name] = fp

            fp_with_root: Path = Path(self.root, fp)

            da: xr.DataArray = write_zarr(
                da,
                fp_with_root,
                crs=da.rio.crs,
                **kwargs,
            )
            self.other[name] = fp_with_root
        return self.other[name]

    def _set_grid(
        self,
        grid_name: str,
        grid: xr.Dataset,
        data: xr.DataArray,
        name: str,
        write: bool,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Add data to grid dataset.

        All layers of grid must have identical spatial coordinates.

        Args:
            grid_name: name of the grid, e.g. "grid", "subgrid"
            grid: the gridded dataset itself
            data: the data to add to the grid
            write: if True, write the data to disk
            name: the name of the layer that will be added to the grid.
            **kwargs: additional keyword arguments to pass to write_zarr

        Returns:
            grid: the updated grid with the new layer addedå
        """
        assert isinstance(data, xr.DataArray)

        if name in grid:
            grid = grid.drop_vars(name)

        if len(grid) == 0:
            assert name == "mask", "First grid layer must be mask"
            assert data.dtype == bool
        else:
            assert np.array_equal(data.x.values, grid.x.values)
            assert np.array_equal(data.y.values, grid.y.values)

            # when updating, it is possible that the mask already exists.
            if name != "mask":
                # if the mask exists, mask the data, saving some valuable space on disk
                if data.chunks is not None and (
                    len(data.chunksizes["x"]) != 1 or len(data.chunksizes["y"]) != 1
                ):
                    # if the data is chunked, we need to chunk the mask in the same way before applying it
                    mask: xr.DataArray = grid["mask"].chunk(
                        {"x": data.chunksizes["x"], "y": data.chunksizes["y"]}
                    )
                else:
                    mask: xr.DataArray = grid["mask"]
                data_ = xr.where(
                    ~mask,
                    data,
                    data.attrs["_FillValue"] if data.dtype != bool else False,
                    keep_attrs=True,
                )
                # depending on the order of the dimensions, xr.where may change the dimension order
                # so we change it back
                data = data_.transpose(*data.dims)

        if write:
            fn = Path(grid_name) / (name + ".zarr")
            self.logger.info(f"Writing file {fn}")
            if data.chunks is not None and (
                len(data.chunksizes["x"]) != 1 or len(data.chunksizes["y"]) != 1
            ):
                with create_temp_zarr(
                    data,
                    name=grid_name + "_" + "tmp",
                ) as tmp_zarr_path:
                    del data
                    data: xr.DataArray = write_zarr(
                        tmp_zarr_path.chunk({"x": -1, "y": -1}),
                        path=self.root / fn,
                        crs=4326,
                        **kwargs,
                    )
            else:
                data: xr.DataArray = write_zarr(
                    data,
                    path=self.root / fn,
                    crs=4326,
                    **kwargs,
                )
            self.files[grid_name][name] = Path(grid_name) / (name + ".zarr")

        grid[name] = data
        return grid

    def set_grid(
        self, data: xr.DataArray, name: str, write: bool = True, **kwargs: Any
    ) -> xr.DataArray:
        """Set a new grid layer.

        When the first layer is added to the grid, it must be the mask layer.
        This layer is used to define the spatial extent of the grid and set
        the active cells.

        Args:
            data: The data to add to the grid. Must have the same spatial coordinates
            name: The name of the layer to add to the grid.
            write: If True, write the data to disk. Defaults to True.
            **kwargs: Additional keyword arguments to pass to write_zarr.

        Returns:
            The added grid layer. The returned layer is read from disk if write=True, so
            it is not the same object as the input data.
        """
        self._set_grid("grid", self.grid, data, write=write, name=name)
        return self.grid[name]

    def set_subgrid(
        self,
        data: xr.DataArray,
        name: str,
        write: bool = True,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Set a new subgrid layer.

        When the first layer is added to the subgrid, it must be the mask layer.
        This layer is used to define the spatial extent of the subgrid and set
        the active cells.

        Args:
            data: The data to add to the subgrid. Must have the same spatial coordinates
                as the existing subgrid.
            name: The name of the layer to add to the subgrid.
            write: If True, write the data to disk. Defaults to True.
            **kwargs: Additional keyword arguments to pass to write_zarr.

        Returns:
            The added subgrid layer. The returned layer is read from disk if write=True, so
            it is not the same object as the input data.
        """
        self.subgrid = self._set_grid(
            "subgrid",
            self.subgrid,
            data,
            write=write,
            name=name,
            **kwargs,
        )
        return self.subgrid[name]

    @property
    def subgrid_factor(self) -> int:
        """The factor by which the subgrid is smaller than the original grid.

        Returns:
            The subgrid factor as an integer.
        """
        subgrid_factor: int = self.subgrid.sizes["x"] // self.grid.sizes["x"]
        assert subgrid_factor == self.subgrid.sizes["y"] // self.grid.sizes["y"]
        return subgrid_factor

    @property
    def ldd_scale_factor(self) -> int:
        """The factor by which the original D8 flow directions are higher resolution than the model grid.

        Returns:
            The scale factor as an integer.
        """
        scale_factor: int = (
            self.other["drainage/original_d8_flow_directions"].shape[0]
            // self.grid["mask"].shape[0]
        )
        assert scale_factor == (
            self.other["drainage/original_d8_flow_directions"].shape[1]
            // self.grid["mask"].shape[1]
        )
        return scale_factor

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Get the region geometry.

        Returns:
            The region geometry as a GeoDataFrame.
        """
        return self.geom["mask"]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Get the bounds of the model grid.

        Returns:
            The bounds of the model grid as a tuple (minx, miny, maxx, maxy).
        """
        return self.grid.rio.bounds(recalc=True)

    @property
    def root(self) -> Path:
        """Get the root directory for the model.

        Returns:
            The root directory path.
        """
        return self._root

    @root.setter
    def root(self, root: Path) -> None:
        """Set the root directory for the model.

        Makes the path absolute.

        Args:
            root: The root directory path.
        """
        self._root = Path(root).absolute()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def report_dir(self) -> Path:
        """Directory to save reports to for logging and checking.

        Returns:
            Directory path
        """
        path: Path = Path(self.root).parent / "output" / "build"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def full_like(
        self,
        data: xr.DataArray,
        fill_value: int | float | bool,
        nodata: int | float | bool | None,
        attrs: dict | None = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Create a DataArray full of a specified value, with the same shape and coordinates as another DataArray.

        Args:
            data: The DataArray to use as a template.
            fill_value: The value to fill the new DataArray with.
            nodata: The value to use for no data in the new DataArray.
            attrs: Optional dictionary of attributes to set on the new DataArray.
            *args: Additional positional arguments to pass to xr.full_like.
            **kwargs: Additional keyword arguments to pass to xr.full_like.

        Returns:
            A new DataArray with the same shape and coordinates as `data`, filled with `fill_value` and with `nodata` set.
        """
        return full_like(
            data=data,
            fill_value=fill_value,
            nodata=nodata,
            attrs=attrs,
            **kwargs,
        )

    def run_method(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Log method parameters before running a method.

        Args:
            method: The name of the method to run.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        """
        func: Callable = getattr(self, method)
        signature = inspect.signature(func)
        # combine user and default options
        params: dict[str, Any] = {}
        for i, (k, v) in enumerate(signature.parameters.items()):
            if k in ["args", "kwargs"]:
                if k == "args":
                    params[k] = args[i:]
                else:
                    params.update(**kwargs)
            else:
                v: Any = kwargs.get(k, v.default)
                if len(args) > i:
                    v: Any = args[i]

                params[k] = v

        # call the method
        func(*args, **kwargs)

    def run_methods(
        self,
        methods: dict[str, Any],
        validate_order: bool = True,
        record_progress: bool = False,
        continue_: bool = False,
    ) -> None:
        """Run methods in the order specified in the methods dictionary.

        Args:
            methods: A dictionary with method names as keys and their parameters as values.
            validate_order: If True, validate the order of methods using the build_method decorator.
            record_progress: If True, record progress after each method.
            continue_: Continue previous build if it was interrupted or failed.

        Raises:
            ValueError: If continuing a build and completed methods are not in the methods to run
                or if the order is incorrect.
        """
        # then loop over other methods
        # TODO: Allow validate order for custom models
        methods = build_method.validate_methods(methods, validate_order=validate_order)
        self.files = self.read_or_create_file_library()

        completed_methods: list[str] = (
            build_method.read_progress(self.progress_path) if continue_ else []
        )

        # check if all completed methods are in the methods to run and if order is correct
        if continue_:
            methods_to_run = list(methods.keys())
            for i, completed_method in enumerate(completed_methods):
                if completed_method not in methods_to_run:
                    raise ValueError(
                        f"Cannot continue build: completed method {completed_method} not in methods to run. Restore the method or start a new build."
                    )
                if completed_method != methods_to_run[i]:
                    raise ValueError(
                        f"Cannot continue build: completed method {completed_method} is out of order. Restore the method order or start a new build."
                    )

        build_run_started_at: datetime = datetime.now()

        # For cluster builds (large_scale6/<cluster>/<scenario>/<input_folder>),
        # detect the top-level model dir by looking for model.yml three levels up.
        root_abs: Path = Path(self.root).resolve()
        scenario_dir: Path = root_abs.parent
        cluster_dir: Path = scenario_dir.parent
        model_dir: Path = cluster_dir.parent
        stats_path: Path | None = None
        cluster_name_for_stats: str = ""
        if record_progress and (model_dir / "model.yml").exists():
            cluster_name_for_stats = cluster_dir.name
            # Each cluster writes its own CSV to avoid corrupt conditions when
            # multiple Snakemake jobs build clusters in parallel.
            stats_path = (
                model_dir / "build_memory_stats" / f"{cluster_name_for_stats}.csv"
            )

        for method in methods:
            if method in completed_methods:
                self.logger.info(f"Skipping already completed method: {method}")
                continue

            kwargs = {} if methods[method] is None else methods[method]
            try:
                self.run_method(method, **kwargs)
            finally:
                # Write memory stats after every method regardless of success or
                # failure so partial results survive job crashes.
                if stats_path is not None:
                    build_method.write_build_stats(
                        stats_path=stats_path,
                        cluster_name=cluster_name_for_stats,
                        run_timestamp=build_run_started_at,
                        cluster_dir=scenario_dir,  # measure subdirs of base/, not input/
                    )
            self.write_file_library()

            if record_progress:
                build_method.record_progress(
                    self.progress_path,
                    method,
                )

        self.logger.info("Finished!")

        build_method.log_statistics()

    def build(
        self,
        region: dict,
        methods: dict[str, dict[str, Any]],
        continue_: bool,
        check_required_methods: bool = True,
    ) -> None:
        """Build the model with the specified region and methods.

        Args:
            region: A dictionary defining the region to build the model for.
            methods: A dictionary with method names as keys and their parameters as values.
            continue_: Continue previous build if it was interrupted or failed.
            check_required_methods: If True, check if all required methods are present in methods.

        Raises:
            ValueError: If "setup_region" is not in methods when building a new model.
            ValueError: If continue is requested but the code version does not match the build version.
        """
        methods: dict[str, dict[str, Any]] = methods or {}
        if "setup_region" not in methods:
            raise ValueError(
                '"setup_region" must be present in methods when building a new model.'
            )
        methods["setup_region"].update(region=region)

        if check_required_methods:
            build_method.check_required_methods(methods.keys())

        # if not continuing, remove existing files path
        if (
            continue_ and self.progress_path.exists()
        ):  # check if continue and already some progress was made
            if not self.version_is_current():
                raise ValueError(
                    "Cannot continue build: version mismatch. The version of the existing build is different from the current version of the code. This likely means that the code was updated since the last build. To continue, either restore the old version of the code or start a new build."
                )

            self.read()
        else:
            # for new build, remove existing files path and progress file
            self.files_path.unlink(missing_ok=True)
            self.progress_path.unlink(missing_ok=True)

            # Fresh build so remove the version file and set
            # to current version.
            self.version_path.unlink(missing_ok=True)
            self.set_current_version()

        self.run_methods(
            methods,
            validate_order=True and type(self) is GEBModel,
            record_progress=True,
            continue_=continue_,
        )

        self.write_build_complete()

    def update_version(self, methods: dict[str, Any]) -> None:
        """Check if the version in the version file is the same as the current version.

        If the version is not current, print a warning with the updates that need to be made to update to the current version.
        """
        # No version file exists, so we create one with the current version
        if not self.version_path.exists():
            self.set_current_version()
            return
        version_info = self.version_path.read_text()
        if self.version_is_current():
            self.logger.info("Version is already current.")
        else:
            self.read()
            # Find and print all updates between the stored version and the current version
            get_and_maybe_do_version_updates(
                version_info,
                build_model=self,
                methods=methods,
                logger=self.logger,
            )

    def update(
        self,
        methods: dict,
    ) -> None:
        """This function updates an already existing model by running the specified methods.

        Args:
            methods: A dictionary with method names as keys and their parameters as values.

        Raises:
            ValueError: If "setup_region" is in methods, as this can only be called when
                building a new model.
        """
        self.read()

        methods = methods or {}

        if "setup_region" in methods:
            raise ValueError(
                '"setup_region" can only be called when starting a new model.'
            )

        self.run_methods(methods, validate_order=False, record_progress=False)
