"""This module contains the main setup for the GEB model.

Notes:
- All prices are in nominal USD (face value) for their respective years. That means that the prices are not adjusted for inflation.
"""

import inspect
import json
import logging
import math
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import geopandas as gpd
import networkx
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyflwdir
import rasterio
import xarray as xr
import zarr
from affine import Affine
from hydromt.data_catalog import DataCatalog
from rasterio.env import defenv
from shapely.geometry import Point

from geb.build.data_catalog import NewDataCatalog
from geb.build.methods import build_method
from geb.workflows.raster import full_like, repeat_grid

from ..workflows.io import open_zarr, to_zarr
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
)

# Set environment options for robustness
GDAL_HTTP_ENV_OPTS = {
    "GDAL_HTTP_MAX_RETRY": "10",  # Number of retry attempts
    "GDAL_HTTP_RETRY_DELAY": "2",  # Delay (seconds) between retries
    "GDAL_HTTP_TIMEOUT": "30",  # Timeout in seconds
    "GDAL_CACHEMAX": 1 * 1024**3,  # 1 GB cache size
}
defenv(**GDAL_HTTP_ENV_OPTS)

XY_CHUNKSIZE = 3000  # chunksize for xy coordinates

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


class PathEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Path objects.

    Paths are converted to their string representation in posix format.
    All files should be posix format to ensure compatibility across different operating systems.
    """

    def default(self, obj: object) -> Any:
        """Convert Path objects to strings for JSON serialization.

        Ottherwise, use the default serialization.

        Args:
            obj: Object to serialize.

        Returns:
            The serialized object. For Path objects, this is a string.
        """
        if isinstance(obj, Path):
            obj = obj.as_posix()
            return str(obj)
        return super().default(obj)


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


def clip_region(
    mask: xr.DataArray, *data_arrays: xr.DataArray, align: float | int
) -> tuple[xr.DataArray, ...]:
    """Use the given mask to clip the mask itself and the given data arrays.

    The clipping is done to the bounding box of the True values in the mask. The bounding box
    is aligned to the given align value. The align value is in the same units as the coordinates
    of the mask and data arrays.

    Args:
        mask: The mask to use for clipping. Must be a 2D boolean DataArray with x and y coordinates.
            True values indicate the area to keep.
        *data_arrays: The data arrays to clip. Must have the same x and y coordinates as the mask.
        align: Align the bounding box to a specific grid spacing. For example, when this is set to 1
            the bounding box will be aligned to whole numbers. If set to 0.5, the bounding box will
            be aligned to 0.5 intervals.

    Returns:
        A tuple containing the clipped mask and the clipped data arrays.

    Raises:
        ValueError: If the data arrays do not have the same shape or coordinates as the mask.
    """
    rows, cols = np.where(mask)
    mincol = cols.min()
    maxcol = cols.max()
    minrow = rows.min()
    maxrow = rows.max()

    minx = mask.x[mincol].item()
    maxx = mask.x[maxcol].item()
    miny = mask.y[minrow].item()
    maxy = mask.y[maxrow].item()

    xres, yres = mask.rio.resolution()

    mincol_aligned = mincol + round(((minx // align * align) - minx) / xres)
    maxcol_aligned = maxcol + round(((maxx // align * align) + align - maxx) / xres)
    minrow_aligned = minrow + round(((miny // align * align) + align - miny) / yres)
    maxrow_aligned = maxrow + round((((maxy // align) * align) - maxy) / yres)

    assert math.isclose(mask.x[mincol_aligned] // align % 1, 0)
    assert math.isclose(mask.x[maxcol_aligned] // align % 1, 0)
    assert math.isclose(mask.y[minrow_aligned] // align % 1, 0)
    assert math.isclose(mask.y[maxrow_aligned] // align % 1, 0)

    assert mincol_aligned <= mincol
    assert maxcol_aligned >= maxcol
    assert minrow_aligned <= minrow
    assert maxrow_aligned >= maxrow

    clipped_mask = mask.isel(
        y=slice(minrow_aligned, maxrow_aligned),
        x=slice(mincol_aligned, maxcol_aligned),
    )
    clipped_arrays = []
    for da in data_arrays:
        if da.shape != mask.shape:
            raise ValueError("All data arrays must have the same shape as the mask.")
        if not np.array_equal(da.x, mask.x) or not np.array_equal(da.y, mask.y):
            raise ValueError(
                "All data arrays must have the same coordinates as the mask."
            )
        clipped_arrays.append(
            da.isel(
                y=slice(minrow_aligned, maxrow_aligned),
                x=slice(mincol_aligned, maxcol_aligned),
            )
        )
    return clipped_mask, *clipped_arrays


def get_river_graph(data_catalog: DataCatalog) -> networkx.DiGraph:
    """Create a directed graph for the river network.

    Args:
        data_catalog: Data catalog containing the MERIT basins.

    Returns:
        A directed graph where nodes are COMID values and edges point downstream.
    """
    river_network: pd.DataFrame = (
        data_catalog.fetch("merit_basins_rivers")
        .read(columns=["COMID", "NextDownID"])
        .set_index("COMID")
    )
    assert river_network.index.name == "COMID", (
        "The index of the river network is not the COMID column"
    )

    # create a directed graph for the river network
    river_graph: networkx.DiGraph = networkx.DiGraph()

    # add rivers with downstream connection
    river_network_with_downstream_connection = river_network[
        river_network["NextDownID"] != 0
    ]
    river_network_with_downstream_connection = (
        river_network_with_downstream_connection.itertuples(index=True, name=None)
    )
    river_graph.add_edges_from(river_network_with_downstream_connection)

    river_network_without_downstream_connection = river_network[
        river_network["NextDownID"] == 0
    ]
    river_graph.add_nodes_from(river_network_without_downstream_connection.index)

    return river_graph


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
    COMID: gpd.GeoDataFrame = (
        data_catalog.fetch("merit_basins_catchments")
        .read(
            bbox=(lon - 10e-6, lat - 10e-6, lon + 10e-6, lat + 10e-6),
        )
        .set_index("COMID")
    )
    COMID = gpd.read_parquet(
        data_catalog.get_source("MERIT_Basins_cat").path,
        bbox=(lon - 10e-6, lat - 10e-6, lon + 10e-6, lat + 10e-6),
    ).set_index("COMID")

    # get only the points where the point is inside the basin
    COMID = COMID[COMID.geometry.contains(Point(lon, lat))]

    if len(COMID) == 0:
        raise ValueError(
            f"The point is not in a basin. Note, that there are some holes in the MERIT basins dataset ({data_catalog.get_source('MERIT_Basins_cat').path}), ensure that the point is in a basin."
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
        data_catalog.get_source("MERIT_Basins_cat").path,
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


def get_touching_subbasins(
    data_catalog: DataCatalog, subbasins: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Find all subbasins that touch the given subbasins.

    Args:
        data_catalog: Data catalog containing the MERIT basins.
        subbasins: GeoDataFrame containing the subbasins to find touching subbasins for.

    Returns:
        A GeoDataFrame containing all subbasins that touch the given subbasins.
    """
    bbox = subbasins.total_bounds
    buffer: float = 0.1
    buffered_bbox = (
        bbox[0] - buffer,
        bbox[1] - buffer,
        bbox[2] + buffer,
        bbox[3] + buffer,
    )
    potentially_touching_basins = gpd.read_parquet(
        data_catalog.get_source("MERIT_Basins_cat").path,
        bbox=buffered_bbox,
        filters=[
            ("COMID", "not in", subbasins.index.tolist()),
        ],
    )
    # get all touching subbasins
    touching_subbasins = potentially_touching_basins[
        potentially_touching_basins.geometry.touches(subbasins.union_all())
    ]

    return touching_subbasins.set_index("COMID")


def get_coastline_nodes(
    coastline_graph: networkx.Graph, STUDY_AREA_OUTFLOW: int, NEARBY_OUTFLOW: int
) -> set:
    """Get all coastline nodes that are part of the coastline for the study area.

    Args:
        coastline_graph: The graph containing all coastline nodes.
        STUDY_AREA_OUTFLOW: The outflow type value for outflows within the study area.
        NEARBY_OUTFLOW: The outflow type value for outflows outside the study area, but close enough to influence the coastline.

    Returns:
        A set of all coastline nodes that are part of the coastline for the study area.
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
                assert len(study_area_nodes) == 1
                study_area_node = study_area_nodes[0]

                assert len(nearby_nodes) == 1
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
        root: str | None = None,
        data_catalog: str | None = None,
        epsg: int = 4326,
        data_provider: str = "default",
    ) -> None:
        """Initialize the GEB build model.

        Args:
            logger: Logger to use for logging.
            root: Root directory for the model build. If None, the current working directory is used.
            data_catalog: List of data catalogs to use. If None, the default data catalogs are used.
            epsg: EPSG code for the model grid. Default is 4326 (WGS84).
            data_provider: Data provider to use for the data catalog. Default is "default".
        """
        self.logger = logger
        self.data_catalog = DataCatalog(
            data_libs=[data_catalog], logger=self.logger, fallback_lib=None
        )

        Hydrography.__init__(self)
        Forcing.__init__(self)
        Crops.__init__(self)
        LandSurface.__init__(self)
        Agents.__init__(self)
        GroundWater.__init__(self)
        Observations.__init__(self)

        self.root = root
        self.epsg = epsg
        self.data_provider = data_provider
        self.new_data_catalog = NewDataCatalog()

        # the grid, subgrid, and region subgrids are all datasets, which should
        # have exactly matching coordinates
        self.grid = xr.Dataset()
        self.subgrid = xr.Dataset()
        self.region_subgrid = xr.Dataset()

        # all other data types are dictionaries because these entries don't
        # necessarily match the grid coordinates, shapes etc.
        self.geom: DelayedReader = DelayedReader(reader=gpd.read_parquet)
        self.table: DelayedReader = DelayedReader(reader=pd.read_parquet)
        self.array: DelayedReader = DelayedReader(zarr.load)
        self.dict: DelayedReader = DelayedReader(
            reader=lambda x: json.load(open(x, "r"))
        )
        self.other: DelayedReader = DelayedReader(reader=open_zarr)

    @build_method
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
        river_graph = get_river_graph(self.new_data_catalog)

        self.logger.info("Finding sinks in river network of requested region.")
        if "subbasin" in region:
            if isinstance(region["subbasin"], list):
                sink_subbasin_ids = region["subbasin"]
            else:
                sink_subbasin_ids = [region["subbasin"]]
        elif "outflow" in region:
            lat, lon = region["outflow"]["lat"], region["outflow"]["lon"]
            sink_subbasin_ids = [
                get_subbasin_id_from_coordinate(self.new_data_catalog, lon, lat)
            ]
        elif "geom" in region:
            regions = self.data_catalog.get_geodataframe(region["geom"]["source"])
            regions = regions[
                regions[region["geom"]["column"]] == region["geom"]["key"]
            ]
            sink_subbasin_ids = get_sink_subbasin_id_for_geom(
                self.data_catalog, regions, river_graph
            )
        else:
            raise ValueError(f"Region {region} not understood.")

        self.logger.info(
            f"Found {len(sink_subbasin_ids)} sink subbasins in region {region}."
        )
        self.set_routing_subbasins(river_graph, sink_subbasin_ids)

        subbasins = self.geom["routing/subbasins"]
        subbasins_without_outflow_basin = subbasins[
            ~subbasins["is_downstream_outflow_subbasin"]
        ]

        buffer = 0.5  # buffer in degrees
        xmin, ymin, xmax, ymax = subbasins_without_outflow_basin.total_bounds
        xmin -= buffer
        ymin -= buffer
        xmax += buffer
        ymax += buffer

        ldd: xr.DataArray = self.new_data_catalog.fetch(
            "merit_hydro_dir",
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        ).read()

        ldd_network = pyflwdir.from_array(
            ldd.values,
            ftype="d8",
            transform=ldd.rio.transform(recalc=True),
            latlon=True,
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
        elif "subbasin" in region or "geom" in region:
            geom = gpd.GeoDataFrame(
                geometry=[subbasins_without_outflow_basin.union_all()],
                crs=subbasins_without_outflow_basin.crs,
            )
            # ESPG 6933 (WGS 84 / NSIDC EASE-Grid 2.0 Global) is an equal area projection
            # while thhe shape of the polygons becomes vastly different, the area is preserved mostly.
            # usable between 86°S and 86°N.
            self.logger.info(
                f"Approximate riverine basin size: {round(geom.to_crs(epsg=6933).area.sum() / 1e6, 2)} km2"
            )

            riverine_mask = create_riverine_mask(ldd, ldd_network, geom)
            assert not riverine_mask.attrs["_FillValue"]
        else:
            raise ValueError(f"Region {region} not understood.")

        if include_coastal_area and subbasins["is_coastal_basin"].any():
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

        ldd_elevation: xr.DataArray = self.new_data_catalog.fetch(
            "merit_hydro_elv",
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        ).read()

        assert ldd_elevation.shape == ldd.shape == mask.shape

        ldd_elevation = xr.where(
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

        self.derive_mask(ldd, ldd.rio.transform(), resolution_arcsec)

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

        STUDY_AREA_OUTFLOW: int = 1
        NEARBY_OUTFLOW: int = 2

        rivers: gpd.GeoDataFrame = (
            self.new_data_catalog.fetch(
                "merit_basins_rivers",
            )
            .read(
                columns=["COMID", "lengthkm", "uparea", "maxup", "geometry"],
                bbox=ldd.rio.bounds(),
            )
            .set_index("COMID")
        )

        rivers["outflow_type"] = rivers.apply(
            lambda row: STUDY_AREA_OUTFLOW
            if row.name in subbasins.index
            else NEARBY_OUTFLOW,
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

        self.logger.info("Coarsening hydrography")
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
        )

        self.set_subgrid(submask, name="mask")

    @build_method
    def set_time_range(self, start_date: datetime, end_date: datetime) -> None:
        """Sets the time range for the build model.

        This time range is used to ensure that all datasets with a time dimension
        cover at least this time range.

        Args:
            start_date: The start date of the model.
            end_date: The end date of the model.

        """
        assert start_date < end_date, "Start date must be before end date."
        self.set_dict(
            {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
            name="model_time_range",
        )

    @property
    def start_date(self) -> datetime:
        """Gets the start date of the build model.

        All datasets with a time dimension should cover at least from this date.

        Returns:
            The start date of the model.
        """
        return datetime.fromisoformat(self.dict["model_time_range"]["start_date"])

    @property
    def end_date(self) -> datetime:
        """Gets the end date of the build model.

        All datasets with a time dimension should cover at least until this date.

        Returns:
            The end date of the model.
        """
        return datetime.fromisoformat(self.dict["model_time_range"]["end_date"])

    @build_method
    def set_ssp(self, ssp: str) -> None:
        """Sets the SSP name for the model.

        Args:
            ssp: The SSP name. Supported SSPs are: ssp1, ssp3, ssp5.
        """
        assert ssp in ["ssp1", "ssp3", "ssp5"], (
            f"SSP {ssp} not supported. Supported SSPs are: ssp1, ssp3, ssp5."
        )
        self.set_dict({"ssp": ssp}, name="ssp")

    @property
    def ssp(self) -> str:
        """Gets the SSP name that was set using set_ssp, or the default SSP of "ssp3".

        Returns:
            The SSP name. If no SSP was set, returns "ssp3" (middle of the road).
        """
        return self.dict["ssp"]["ssp"] if "ssp" in self.dict else "ssp3"

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

    def snap_to_grid(
        self,
        ds: xr.DataArray | xr.Dataset,
        reference: xr.DataArray | xr.Dataset,
        relative_tollerance: float = 0.02,
        ydim: str = "y",
        xdim: str = "x",
    ) -> xr.Dataset | xr.DataArray:
        """Snaps the coordinates of a dataset to a reference dataset.

        Some datasets have a slightly different grid than the model grid, usually
        because of different rounding errors when creating the grid, and floating
        point precision issues. This method checks if the coordinates are more or
        less the same, and if so, snaps the coordinates of the dataset to the
        reference dataset.

        Args:
            ds: The dataset to snap.
            reference: The reference dataset.
            relative_tollerance: The relative tolerance for snapping.
            ydim: The name of the y dimension.
            xdim: The name of the x dimension.

        Returns:
            The snapped dataset.
        """
        # make sure all datasets have more or less the same coordinates
        assert np.isclose(
            ds.coords[ydim].values,
            reference[ydim].values,
            atol=abs(ds.rio.resolution()[1] * relative_tollerance),
            rtol=0,
        ).all()
        assert np.isclose(
            ds.coords[xdim].values,
            reference[xdim].values,
            atol=abs(ds.rio.resolution()[0] * relative_tollerance),
            rtol=0,
        ).all()
        return ds.assign_coords({ydim: reference[ydim], xdim: reference[xdim]})

    def setup_coastal_water_levels(
        self,
    ) -> None:
        """Sets up coastal water level data from the GTSM dataset.

        Filters the dataset to include only stations within the model bounds,
        and ensures that the time dimension is consistent.
        """
        water_levels = self.data_catalog.get_dataset("GTSM")
        assert (
            water_levels.time.diff("time").astype(np.int64)
            == (water_levels.time[1] - water_levels.time[0]).astype(np.int64)
        ).all()
        # convert to geodataframe
        stations = gpd.GeoDataFrame(
            water_levels.stations,
            geometry=gpd.points_from_xy(
                water_levels.station_x_coordinate, water_levels.station_y_coordinate
            ),
        )
        # filter all stations within the bounds, considering a buffer
        station_ids = stations.cx[
            self.bounds[0] - 0.1 : self.bounds[2] + 0.1,
            self.bounds[1] - 0.1 : self.bounds[3] + 0.1,
        ].index.values

        water_levels = water_levels.sel(stations=station_ids)

        assert len(water_levels.stations) > 0, (
            "No stations found in the region. If no stations should be set, set include_coastal_area=False"
        )

        self.set_other(
            water_levels,
            name="waterlevels",
            time_chunksize=24 * 6,  # 10 minute data
            byteshuffle=True,
        )

    @build_method
    def setup_damage_parameters(
        self,
        parameters: dict[
            str, dict[str, dict[str, dict[str, list[tuple[float, float]] | float]]]
        ],
    ) -> None:
        """Sets up damage parameters for different hazards and asset types.

        Args:
            parameters: A nested dictionary containing damage parameters. The structure is:
                {
                    "hazard_type": {
                        "asset_type": {
                            "component": {
                                "curve": [(severity, damage_ratio), ...],
                                "maximum_damage": float
                            },
                            ...
                        },
                        ...
                    },
                    ...
                }
        """
        for hazard, hazard_parameters in parameters.items():
            for asset_type, asset_parameters in hazard_parameters.items():
                for component, asset_compontents in asset_parameters.items():
                    curve = pd.DataFrame(
                        asset_compontents["curve"], columns=["severity", "damage_ratio"]
                    )

                    self.set_table(
                        curve,
                        name=f"damage_parameters/{hazard}/{asset_type}/{component}/curve",
                    )

                    maximum_damage = {
                        "maximum_damage": asset_compontents["maximum_damage"]
                    }

                    self.set_dict(
                        maximum_damage,
                        name=f"damage_parameters/{hazard}/{asset_type}/{component}/maximum_damage",
                    )

    @build_method
    def setup_precipitation_scaling_factors_for_return_periods(
        self, risk_scaling_factors: list[tuple[float, float]]
    ) -> None:
        """Sets up precipitation scaling factors for different return periods.

        Args:
            risk_scaling_factors: A list of tuples containing the exceedance probability
                and the corresponding scaling factor.
        """
        risk_scaling_factors_df = pd.DataFrame(
            risk_scaling_factors, columns=["exceedance_probability", "scaling_factor"]
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
            # brotli is a bit slower but gives better compression,
            # gzip is faster to read. Higher compression levels
            # generally don't make it slower to read, therefore
            # we use the highest compression level for gzip
            table.to_parquet(
                fp_with_root, engine="pyarrow", compression="gzip", compression_level=9
            )

        self.table[name] = fp_with_root

    def set_array(self, data: npt.NDArray[Any], name: str, write: bool = True) -> None:
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
            zarr.save_array(fp_with_root, data, overwrite=True)

        self.array[name] = fp_with_root

    def set_dict(self, data: dict, name: str, write: bool = True) -> None:
        """Set a dictionary and save it to disk.

        Args:
            data: The dictionary to save.
            name: The name of the dictionary.
            write: Whether to write the dictionary to disk. If False, the dictionary
                is only added to the file library, but not written to disk.
        """
        fp: Path = Path("dict") / (name + ".json")
        fp_with_root: Path = Path(self.root) / fp
        fp_with_root.parent.mkdir(parents=True, exist_ok=True)
        if write:
            self.logger.info(f"Writing file {fp}")

            self.files["dict"][name] = fp

            with open(fp_with_root, "w") as f:
                json.dump(data, f, default=lambda o: o.isoformat(), indent=4)

        self.dict[name] = fp_with_root

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
            # brotli is a bit slower but gives better compression,
            # gzip is faster to read. Higher compression levels
            # generally don't make it slower to read, therefore
            # we use the highest compression level for gzip
            geom.to_parquet(
                fp_with_root, engine="pyarrow", compression="gzip", compression_level=9
            )

        self.geom[name] = fp_with_root

    @property
    def files_path(self) -> Path:
        """Path to the files.json file that contains the file library."""
        return Path(self.root, "files.json")

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

        with open(self.files_path, "w") as f:
            json.dump(file_library, f, indent=4, cls=PathEncoder)

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
                "region_subgrid": {},
                "other": {},
            }
        else:
            with open(Path(self.files_path), "r") as f:
                files: dict[str, dict[str, str]] = json.load(f)

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

    def read_dict(self) -> None:
        """Reads all dictionaries from disk based on the file library."""
        for name, fn in self.files["dict"].items():
            self.dict[name] = Path(self.root, fn)

    def read_grid(self) -> None:
        """Reads all grid data arrays from disk based on the file library."""
        for name, fn in self.files["grid"].items():
            data: xr.DataArray = open_zarr(Path(self.root) / fn)
            self.set_grid(data, name=name, write=False)

    def read_subgrid(self) -> None:
        """Reads all subgrid data arrays from disk based on the file library."""
        for name, fn in self.files["subgrid"].items():
            data: xr.DataArray = open_zarr(Path(self.root) / fn)
            self.set_subgrid(data, name=name, write=False)

    def read_region_subgrid(self) -> None:
        """Reads all region subgrid data arrays from disk based on the file library."""
        for name, fn in self.files["region_subgrid"].items():
            data: xr.DataArray = open_zarr(Path(self.root) / fn)
            self.set_region_subgrid(data, name=name, write=False)

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
            self.read_dict()

            self.read_subgrid()
            self.read_grid()
            self.read_region_subgrid()

            self.read_other()

    def set_other(
        self,
        da: xr.DataArray,
        name: str,
        write: bool = True,
        x_chunksize: int = XY_CHUNKSIZE,
        y_chunksize: int = XY_CHUNKSIZE,
        time_chunksize: int = 1,
        *args: Any,
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
            x_chunksize: The chunk size in the x dimension for writing to zarr.
                Defaults to XY_CHUNKSIZE.
            y_chunksize: The chunk size in the y dimension for writing to zarr.
                Defaults to XY_CHUNKSIZE.
            time_chunksize: The chunk size in the time dimension for writing to zarr.
                Defaults to 1.
            *args: Additional arguments to pass to to_zarr.
            **kwargs: Additional keyword arguments to pass to to_zarr.

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
            da: xr.DataArray = to_zarr(
                da,
                fp_with_root,
                x_chunksize=x_chunksize,
                y_chunksize=y_chunksize,
                time_chunksize=time_chunksize,
                crs=da.rio.crs,
                *args,
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
        x_chunksize: int = XY_CHUNKSIZE,
        y_chunksize: int = XY_CHUNKSIZE,
    ) -> xr.Dataset:
        """Add data to grid dataset.

        All layers of grid must have identical spatial coordinates.

        Args:
            grid_name: name of the grid, e.g. "grid", "subgrid", "region_subgrid"
            grid: the gridded dataset itself
            data: the data to add to the grid
            write: if True, write the data to disk
            name: the name of the layer that will be added to the grid.
            x_chunksize: the chunk size in the x dimension for writing to zarr
            y_chunksize: the chunk size in the y dimension for writing to zarr

        Returns:
            grid: the updated grid with the new layer addedå
        """
        assert isinstance(data, xr.DataArray)

        if name in grid:
            self.logger.warning(f"Replacing grid map: {name}")
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
                data_ = xr.where(
                    ~grid["mask"],
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
            data = to_zarr(
                data,
                path=self.root / fn,
                x_chunksize=x_chunksize,
                y_chunksize=y_chunksize,
                crs=4326,
            )
            self.files[grid_name][name] = Path(grid_name) / (name + ".zarr")

        grid[name] = data
        return grid

    def set_grid(
        self, data: xr.DataArray, name: str, write: bool = True
    ) -> xr.DataArray:
        """Set a new grid layer.

        When the first layer is added to the grid, it must be the mask layer.
        This layer is used to define the spatial extent of the grid and set
        the active cells.

        Args:
            data: The data to add to the grid. Must have the same spatial coordinates
            name: The name of the layer to add to the grid.
            write: If True, write the data to disk. Defaults to True.

        Returns:
            The added grid layer. The returned layer is read from disk if write=True, so
            it is not the same object as the input data.
        """
        self._set_grid("grid", self.grid, data, write=write, name=name)
        return self.grid[name]

    def set_subgrid(
        self, data: xr.DataArray, name: str, write: bool = True
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

        Returns:
            The added subgrid layer. The returned layer is read from disk if write=True, so
            it is not the same object as the input data.
        """
        self.subgrid = self._set_grid(
            "subgrid", self.subgrid, data, write=write, name=name
        )
        return self.subgrid[name]

    def set_region_subgrid(
        self, data: xr.DataArray, name: str, write: bool = True
    ) -> xr.DataArray:
        """Set a new region subgrid layer.

        When the first layer is added to the region subgrid, it must be the mask layer.
        This layer is used to define the spatial extent of the region subgrid and set
        the active cells.

        Args:
            data: The data to add to the region subgrid. Must have the same spatial coordinates
                as the existing region subgrid.
            name: The name of the layer to add to the region subgrid.
            write: If True, write the data to disk. Defaults to True.

        Returns:
            The added region subgrid layer. The returned layer is read from disk if write=True, so
            it is not the same object as the input data.
        """
        self.region_subgrid = self._set_grid(
            "region_subgrid",
            self.region_subgrid,
            data,
            write=write,
            name=name,
        )
        return self.region_subgrid[name]

    @property
    def subgrid_factor(self) -> int:
        """The factor by which the subgrid is smaller than the original grid.

        Returns:
            The subgrid factor as an integer.
        """
        subgrid_factor: int = self.subgrid.dims["x"] // self.grid.dims["x"]
        assert subgrid_factor == self.subgrid.dims["y"] // self.grid.dims["y"]
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

    @property
    def preprocessing_dir(self) -> Path:
        """Directory where preprocessing data is stored.

        Used for caching data.

        Returns:
            Directory path
        """
        return Path(self.root).parent / "preprocessing"

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
        nodata: int | float | bool,
        attrs: dict | None = None,
        *args: Any,
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
            *args,
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

    def run_methods(self, methods: dict[str, Any], validate_order: bool = True) -> None:
        """Run methods in the order specified in the methods dictionary.

        Args:
            methods: A dictionary with method names as keys and their parameters as values.
            validate_order: If True, validate the order of methods using the build_method decorator.
        """
        # then loop over other methods
        # TODO: Allow validate order for custom models
        build_method.validate_methods(methods, validate_order=validate_order)
        self.files = self.read_or_create_file_library()
        for method in methods:
            kwargs = {} if methods[method] is None else methods[method]
            self.run_method(method, **kwargs)
            self.write_file_library()

        self.logger.info("Finished!")

    def build(self, region: dict, methods: dict) -> None:
        """Build the model with the specified region and methods."""
        methods: dict[str:Any] = methods or {}
        methods["setup_region"].update(region=region)
        self.files_path.unlink(missing_ok=True)

        self.run_methods(methods, validate_order=True and type(self) is GEBModel)

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
        methods = methods or {}

        if "setup_region" in methods:
            raise ValueError(
                '"setup_region" can only be called when starting a new model.'
            )

        self.run_methods(methods, validate_order=False and type(self) is GEBModel)

    def get_linear_indices(self, da: xr.DataArray) -> xr.DataArray:
        """Get linear indices for each cell in a 2D DataArray.

        A linear index is a single integer that represents the position of a cell in a flattened version of the array.
        For a 2D array with shape (ny, nx), the linear index of a cell at position (row, column) is calculated as:
        `linear_index = row * nx + column`.

        Args:
            da: A 2D xarray DataArray with dimensions 'y' and 'x'.

        Returns:
            A 2D xarray DataArray of the same shape as `da`, where each cell contains its linear index.
            The linear index is calculated as `row * number_of_columns + column`.

        """
        # Get the sizes of the spatial dimensions
        ny, nx = da.sizes["y"], da.sizes["x"]

        # Create an array of sequential integers from 0 to ny*nx - 1
        grid_ids = np.arange(ny * nx).reshape(ny, nx)

        # Create a DataArray with the same coordinates and dimensions as your spatial grid
        grid_id_da: xr.DataArray = xr.DataArray(
            grid_ids,
            coords={
                "y": da.coords["y"],
                "x": da.coords["x"],
            },
            dims=["y", "x"],
        )

        return grid_id_da

    def get_neighbor_cell_ids_for_linear_indices(
        self, cell_id: int, nx: int, ny: int, radius: int = 1
    ) -> list[int]:
        """Get the linear indices of the neighboring cells of a cell in a 2D grid.

        Linear indices are the indices of the cells in the flattened version of an array.
        For a 2D array with shape (ny, nx), the linear index of a cell at position (row, column)
        is calculated as:
            `linear_index = row * nx + column`.

        Args:
            cell_id: The linear index of the cell for which to find neighbors.
            nx: The number of columns in the grid.
            ny: The number of rows in the grid.
            radius: The radius around the cell to consider as neighbors. Default is 1.

        Returns:
            A list of linear indices of the neighboring cells.

        """
        row: int = cell_id // nx
        col: int = cell_id % nx

        neighbor_cell_ids: list[int] = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue  # Skip the cell itself
                r: int = row + dr
                c: int = col + dc
                if 0 <= r < ny and 0 <= c < nx:
                    neighbor_id: int = r * nx + c
                    neighbor_cell_ids.append(neighbor_id)
        return neighbor_cell_ids
