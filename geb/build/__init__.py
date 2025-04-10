"""This module contains the main setup for the GEB model.

Notes:
- All prices are in nominal USD (face value) for their respective years. That means that the prices are not adjusted for inflation.
"""

import inspect
import itertools
import json
import logging
import math
import os
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Union

import geopandas as gpd
import hydromt.workflows
import networkx
import numpy as np
import pandas as pd
import pyflwdir
import rasterio
import xarray as xr
from affine import Affine
from hydromt.data_catalog import DataCatalog
from rasterio.env import defenv
from scipy.ndimage import binary_dilation
from shapely.geometry import Point

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
from .workflows.general import (
    repeat_grid,
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

logger = logging.getLogger(__name__)


def convert_timestamp_to_string(timestamp):
    return timestamp.isoformat()


@contextmanager
def suppress_logging_warning(logger):
    """
    A context manager to suppress logging warning messages temporarily.
    """
    current_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)  # Set level to ERROR to suppress WARNING messages
    try:
        yield
    finally:
        logger.setLevel(current_level)  # Restore the original logging level


class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            obj = obj.as_posix()
            return str(obj)
        return super().default(obj)


def boolean_mask_to_graph(mask, connectivity=4, **kwargs):
    """
    Convert a boolean mask to an undirected NetworkX graph.
    Additional attributes can be passed as keyword arguments, which
    will be added as attributes to the nodes of the graph.

    Parameters:
    -----------
    mask : xarray.DataArray or numpy.ndarray
        Boolean mask where True values are nodes in the graph
    connectivity : int
        4 for von Neumann neighborhood (up, down, left, right)
        8 for Moore neighborhood (includes diagonals)
    kwargs : dict
        Additional attributes to be added to the nodes of the graph.
        Must be 2D arrays with the same shape as the mask.

    Returns:
    --------
    G : networkx.Graph
        Undirected graph where nodes are (y,x) coordinates of True cells
    """
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


def clip_region(hydrography, align):
    rows, cols = np.where(hydrography["mask"])
    mincol = cols.min()
    maxcol = cols.max()
    minrow = rows.min()
    maxrow = rows.max()

    minx = hydrography.x[mincol].item()
    maxx = hydrography.x[maxcol].item()
    miny = hydrography.y[minrow].item()
    maxy = hydrography.y[maxrow].item()

    xres, yres = hydrography.rio.resolution()

    mincol_aligned = mincol + round(((minx // align * align) - minx) / xres)
    maxcol_aligned = maxcol + round(((maxx // align * align) + align - maxx) / xres)
    minrow_aligned = minrow + round(((miny // align * align) + align - miny) / yres)
    maxrow_aligned = maxrow + round((((maxy // align) * align) - maxy) / yres)

    assert math.isclose(hydrography.x[mincol_aligned] // align % 1, 0)
    assert math.isclose(hydrography.x[maxcol_aligned] // align % 1, 0)
    assert math.isclose(hydrography.y[minrow_aligned] // align % 1, 0)
    assert math.isclose(hydrography.y[maxrow_aligned] // align % 1, 0)

    assert mincol_aligned <= mincol
    assert maxcol_aligned >= maxcol
    assert minrow_aligned <= minrow
    assert maxrow_aligned >= maxrow

    hydrography = hydrography.isel(
        y=slice(minrow_aligned, maxrow_aligned),
        x=slice(mincol_aligned, maxcol_aligned),
    )
    return hydrography


def get_river_graph(data_catalog):
    river_network = pd.read_parquet(
        data_catalog.get_source("MERIT_Basins_riv").path,
        columns=["COMID", "NextDownID"],
    ).set_index("COMID")
    assert river_network.index.name == "COMID", (
        "The index of the river network is not the COMID column"
    )

    # create a directed graph for the river network
    river_graph = networkx.DiGraph()

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


def get_subbasin_id_from_coordinate(data_catalog, lon, lat):
    # we select the basin that contains the point. To do so
    # we use a bounding box with the point coordinates, thus
    # xmin == xmax and ymin == ymax
    # geoparquet uses < and >, not <= and >=, so we need to add
    # a small value to the coordinates to avoid missing the point
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


def get_sink_subbasin_id_for_geom(data_catalog, geom, river_graph):
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
    sink_nodes = [
        COMID_ID
        for COMID_ID, out_degree in region_river_graph.out_degree(
            region_river_graph.nodes
        )
        if out_degree == 0
    ]

    return sink_nodes


class GEBModel(
    Hydrography, Forcing, Crops, LandSurface, Agents, GroundWater, Observations
):
    def __init__(
        self,
        root: str = None,
        data_catalogs: List[str] = None,
        logger=logger,
        epsg=4326,
        data_provider: str = "default",
    ):
        self.logger = logger
        self.data_catalog = DataCatalog(
            data_libs=data_catalogs, logger=self.logger, fallback_lib=None
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

        # the grid, subgrid, and region subgrids are all datasets, which should
        # have exactly matching coordinates
        self.grid = xr.Dataset()
        self.subgrid = xr.Dataset()
        self.region_subgrid = xr.Dataset()

        # all other data types are dictionaries because these entries don't
        # necessarily match the grid coordinates, shapes etc.
        self.geoms = {}
        self.table = {}
        self.array = {}
        self.dict = {}
        self.other = {}

        self.files = defaultdict(dict)

    def setup_region(
        self,
        region: dict,
        subgrid_factor: int,
        resolution_arcsec: int = 30,
        include_coastal: bool = True,
    ) -> None:
        """Creates a 2D regular grid or reads an existing grid.
        An 2D regular grid will be created from a geometry (geom_fn) or bbox. If an existing
        grid is given, then no new grid will be generated.

        Adds/Updates model layers:
        * **grid** grid mask: add grid mask to grid object

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g.:
            * {'basin': [x, y]}

            Region must be of kind [basin, subbasin].
        subgrid_factor : int
            GEB implements a subgrid. This parameter determines the factor by which the subgrid is smaller than the original grid.
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
                sink_subbasin_ids = region["subbasin"]
            else:
                sink_subbasin_ids = [region["subbasin"]]
        elif "outflow" in region:
            lon, lat = region["outflow"][0], region["outflow"][1]
            sink_subbasin_ids = [
                get_subbasin_id_from_coordinate(self.data_catalog, lon, lat)
            ]
        elif "admin" in region:
            admin_regions = self.data_catalog.get_geodataframe(
                region["admin"]["source"]
            )
            admin_regions = admin_regions[
                admin_regions[region["admin"]["column"]] == region["admin"]["key"]
            ]
            sink_subbasin_ids = get_sink_subbasin_id_for_geom(
                self.data_catalog, admin_regions, river_graph
            )
        else:
            raise ValueError(f"Region {region} not understood.")

        self.logger.info(
            f"Found {len(sink_subbasin_ids)} sink subbasins in region {region}."
        )
        self.set_routing_subbasins(river_graph, sink_subbasin_ids)

        subbasins = self.geoms["routing/subbasins"]
        subbasins_without_outflow_basin = subbasins[
            ~subbasins["is_downstream_outflow_subbasin"]
        ]

        xmin, ymin, xmax, ymax = subbasins_without_outflow_basin.total_bounds

        with rasterio.Env(
            GDAL_HTTP_USERPWD=f"{os.environ['MERIT_USERNAME']}:{os.environ['MERIT_PASSWORD']}"
        ):
            hydrography = self.data_catalog.get_rasterdataset(
                "merit_hydro",
                bbox=[
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ],
                variables=["dir", "elv"],
                buffer=10,
            )
            hydrography["dir"].attrs["_FillValue"] = 247
            hydrography["elv"].attrs["_FillValue"] = -9999.0

            self.logger.info("Preparing 2D grid.")
            if "outflow" in region:
                # get basin geometry
                geom, _ = hydromt.workflows.get_basin_geometry(
                    ds=hydrography,
                    flwdir_name="dir",
                    kind="subbasin",
                    logger=self.logger,
                    xy=(lon, lat),
                )
            elif "subbasin" in region or "admin" in region:
                geom = gpd.GeoDataFrame(
                    geometry=[subbasins_without_outflow_basin.union_all()],
                    crs=subbasins_without_outflow_basin.crs,
                )
            else:
                raise ValueError(f"Region {region} not understood.")

            # ESPG 6933 (WGS 84 / NSIDC EASE-Grid 2.0 Global) is an equal area projection
            # while thhe shape of the polygons becomes vastly different, the area is preserved mostly.
            # usable between 86°S and 86°N.
            self.logger.info(
                f"Approximate riverine basin size: {round(geom.to_crs(epsg=6933).area.sum() / 1e6, 2)} km2"
            )

            buffer = 1 / 120  # buffer in degrees
            hydrography = hydrography.rio.clip_box(
                minx=xmin - buffer,
                miny=ymin - buffer,
                maxx=xmax + buffer,
                maxy=ymax + buffer,
            ).compute()

            riverine_mask = self.full_like(
                hydrography["dir"],
                fill_value=True,
                nodata=False,
                dtype=bool,
            )

            assert riverine_mask.attrs["_FillValue"] is False
            riverine_mask = riverine_mask.rio.clip([geom.union_all()], drop=False)

            if (
                include_coastal
                and subbasins_without_outflow_basin["is_coastal_basin"].any()
            ):
                mask = self.get_coastal_area(hydrography["dir"], riverine_mask)
            else:
                mask = riverine_mask

            hydrography["mask"] = mask
            hydrography["dir"] = xr.where(
                mask,
                hydrography["dir"],
                hydrography["dir"].attrs["_FillValue"],
            )

            hydrography["elv"] = xr.where(
                mask,
                hydrography["elv"],
                hydrography["elv"].attrs["_FillValue"],
            )

            # Several datasets come with ...
            hydrography = clip_region(hydrography, align=30 / 60 / 60)

            d8_original = hydrography["dir"]
            d8_original.attrs["_FillValue"] = 247
            d8_original = xr.where(
                hydrography["mask"],
                d8_original,
                d8_original.attrs["_FillValue"],
                keep_attrs=True,
            )
            d8_original = self.set_other(
                d8_original, name="original_d8_flow_directions"
            )

            d8_elv_original = hydrography["elv"]
            d8_elv_original.attrs["_FillValue"] = -9999.0
            d8_elv_original = xr.where(
                hydrography["mask"],
                d8_elv_original,
                d8_elv_original.attrs["_FillValue"],
                keep_attrs=True,
            )
            d8_elv_original = d8_elv_original.raster.mask_nodata()

            self.set_other(d8_elv_original, name="original_d8_elevation")

            self.derive_mask(
                d8_original, hydrography.rio.transform(), resolution_arcsec
            )

        self.create_subgrid(subgrid_factor)

    def get_coastal_area(self, ldd, riverine_mask):
        pits = ldd == 0
        # TODO: Filter non-coastline pits
        coastline = pits
        coastline.attrs["_FillValue"] = None

        # save coastline, for now mostly for debugging purposes
        self.set_other(coastline, name="coastline")

        # first we create a graph with all coastline cells. Neighbouring cells
        # are connected. From this graph we want to select all coastline
        # cells in between river outlets. The riverine mask does not include
        # any of the coastal cells. So we first grow the riverine mask by one cell
        # and use this to set the outflow nodes in the graph.
        coastline_graph = boolean_mask_to_graph(
            coastline,
            connectivity=4,
            outflow=binary_dilation(riverine_mask.values, iterations=1),
        )

        # extract the outflow nodes from the graph, these should be included for sure
        outflow_nodes = set(
            [
                node
                for node, attrs in coastline_graph.nodes(data=True)
                if attrs["outflow"] is True
            ]
        )

        # then for each combination of outflow nodes, we get the shortest path between
        # them. This should be the coastline between the two nodes. Note that this is not
        # very efficient as the number of combinations grows quickly. So we may
        # be able to improve this in the future.
        coastline_nodes = set()
        for node0, node1 in itertools.combinations(outflow_nodes, 2):
            # get the shortest path between the two nodes
            try:
                coastline_nodes.update(
                    networkx.shortest_path(coastline_graph, source=node0, target=node1)
                )
            except networkx.NetworkXNoPath:
                # if there is no path, skip this combination
                continue

        # Even though we already have the shortest path, we need to find the entire coastline
        # which may include neighbours of the nodes in the shortest path. So we need to grow the
        # coastline. We do this iteratively, so until we found all coastline nodes.
        # We do not search for neighbours of the outflow nodes, as this will stop the
        # search from expanding outside the domain that we are interested in.
        coastline_size = -1
        new_coastline_size = len(coastline_nodes)
        while coastline_size < new_coastline_size:
            new_coastline_nodes = set()
            for node in coastline_nodes:
                if node in outflow_nodes:
                    continue
                # get the neighbors of the node
                neighbors = coastline_graph.neighbors(node)
                # add the neighbors to the included coastline
                new_coastline_nodes.update(neighbors)

            coastline_nodes.update(new_coastline_nodes)

            coastline_size, new_coastline_size = (
                new_coastline_size,
                len(coastline_nodes),
            )

        # here we go from the graph back to a mask. We do this by creating a new mask
        # and setting coastline cells to true
        connected_coastline = self.full_like(
            ldd,
            fill_value=False,
            nodata=None,
            dtype=bool,
        )

        for node in coastline_nodes:
            if node in outflow_nodes:
                continue
            yx = coastline_graph.nodes[node]["yx"]
            connected_coastline[yx] = True

        # save the connected coastline, for now mostly for debugging purposes
        self.set_other(connected_coastline, name="connected_coastline")

        # get all upstream cells from the selected coastline
        flow_raster = pyflwdir.from_array(ldd.values, ftype="d8")
        coastal_mask = (
            flow_raster.basins(idxs=np.where(connected_coastline.values.ravel())[0]) > 0
        )

        # return the combination of the riverine mask and the coastal mask
        return riverine_mask | coastal_mask

    def derive_mask(self, d8_original, transform, resolution_arcsec):
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

        scale_factor = resolution_arcsec // 3

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

        self.set_geoms(mask_geom, name="mask")

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

    def create_subgrid(self, subgrid_factor):
        mask = self.grid["mask"]
        dst_transform = mask.rio.transform(recalc=True) * Affine.scale(
            1 / subgrid_factor
        )

        submask = xr.DataArray(
            data=repeat_grid(mask.data, subgrid_factor),
            coords={
                "y": dst_transform.f
                + np.arange(mask.shape[0] * subgrid_factor) * dst_transform.e,
                "x": dst_transform.c
                + np.arange(mask.shape[1] * subgrid_factor) * dst_transform.a,
            },
            attrs={"_FillValue": None},
        )

        submask = self.set_subgrid(submask, name="mask")

    def set_time_range(self, start_date, end_date):
        assert start_date < end_date, "Start date must be before end date."
        self.set_dict(
            {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
            name="model_time_range",
        )

    @property
    def start_date(self):
        return datetime.fromisoformat(self.dict["model_time_range"]["start_date"])

    @property
    def end_date(self):
        return datetime.fromisoformat(self.dict["model_time_range"]["end_date"])

    def snap_to_grid(self, ds, reference, relative_tollerance=0.02, ydim="y", xdim="x"):
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

    def interpolate(self, ds, interpolation_method, ydim="y", xdim="x"):
        out_ds = ds.interp(
            method=interpolation_method,
            **{
                ydim: self.grid.y.rename({"y": ydim}),
                xdim: self.grid.x.rename({"x": xdim}),
            },
        )
        if "inplace" in out_ds.coords:
            out_ds = out_ds.drop_vars(["dparams", "inplace"])
        assert len(ds.dims) == len(out_ds.dims)
        return out_ds

    def setup_coastal_water_levels(
        self,
    ):
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
            "No stations found in the region. If no stations should be set, set include_coastal=False"
        )

        self.set_other(
            water_levels,
            name="waterlevels",
            time_chunksize=24 * 6,  # 10 minute data
            byteshuffle=True,
        )

    def setup_damage_parameters(self, parameters):
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

    def setup_precipitation_scaling_factors_for_return_periods(
        self, risk_scaling_factors
    ):
        risk_scaling_factors = pd.DataFrame(
            risk_scaling_factors, columns=["exceedance_probability", "scaling_factor"]
        )
        self.set_table(risk_scaling_factors, name="precipitation_scaling_factors")

    def set_table(self, table, name, write=True):
        self.table[name] = table
        if write:
            fn = Path("table") / (name + ".parquet")
            self.logger.info(f"Writing file {fn}")

            self.files["table"][name] = fn

            fp = Path(self.root, fn)
            fp.parent.mkdir(parents=True, exist_ok=True)
            # brotli is a bit slower but gives better compression,
            # gzip is faster to read. Higher compression levels
            # generally don't make it slower to read, therefore
            # we use the highest compression level for gzip
            table.to_parquet(
                fp, engine="pyarrow", compression="gzip", compression_level=9
            )

    def set_array(self, data, name, write=True):
        self.array[name] = data

        if write:
            fn = Path("array") / (name + ".npz")
            self.logger.info(f"Writing file {fn}")
            self.files["array"][name] = fn
            fp = Path(self.root, fn)
            fp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fp, data=data)

    def set_dict(self, data, name, write=True):
        self.dict[name] = data

        if write:
            fn = Path("dict") / (name + ".json")
            self.logger.info(f"Writing file {fn}")

            self.files["dict"][name] = fn

            fp = Path(self.root) / fn
            fp.parent.mkdir(parents=True, exist_ok=True)
            with open(fp, "w") as f:
                json.dump(data, f, default=convert_timestamp_to_string)

    def set_geoms(self, geoms, name, write=True):
        self.geoms[name] = geoms

        if write:
            fn = Path("geom") / (name + ".geoparquet")
            self.logger.info(f"Writing file {fn}")
            self.files["geoms"][name] = fn
            fp = self.root / fn
            fp.parent.mkdir(parents=True, exist_ok=True)
            # brotli is a bit slower but gives better compression,
            # gzip is faster to read. Higher compression levels
            # generally don't make it slower to read, therefore
            # we use the highest compression level for gzip
            geoms.to_parquet(
                fp, engine="pyarrow", compression="gzip", compression_level=9
            )

        return self.geoms[name]

    def write_file_library(self):
        file_library = self.read_file_library()

        # merge file library from disk with new files, prioritizing new files
        for type_name, type_files in self.files.items():
            if type_name not in file_library:
                file_library[type_name] = type_files
            else:
                file_library[type_name].update(type_files)

        with open(Path(self.root, "files.json"), "w") as f:
            json.dump(self.files, f, indent=4, cls=PathEncoder)

    def read_file_library(self):
        fp = Path(self.root, "files.json")
        if not fp.exists():
            return {}
        else:
            with open(Path(self.root, "files.json"), "r") as f:
                files = json.load(f)
        return defaultdict(dict, files)  # convert dict to defaultdict

    def read_geoms(self):
        for name, fn in self.files["geoms"].items():
            geom = gpd.read_parquet(Path(self.root, fn))
            self.set_geoms(geom, name=name, write=False)

    def read_array(self):
        for name, fn in self.files["array"].items():
            array = np.load(Path(self.root, fn))["data"]
            self.set_array(array, name=name, write=False)

    def read_table(self):
        for name, fn in self.files["table"].items():
            table = pd.read_parquet(Path(self.root, fn))
            self.set_table(table, name=name, write=False)

    def read_dict(self):
        for name, fn in self.files["dict"].items():
            with open(Path(self.root, fn), "r") as f:
                d = json.load(f)
            self.set_dict(d, name=name, write=False)

    def _read_grid(self, fn) -> xr.Dataset:
        da = open_zarr(Path(self.root) / fn)
        return da

    def read_grid(self) -> None:
        for name, fn in self.files["grid"].items():
            data = self._read_grid(fn)
            self.set_grid(data, name=name, write=False)

    def read_subgrid(self) -> None:
        for name, fn in self.files["subgrid"].items():
            data = self._read_grid(fn)
            self.set_subgrid(data, name=name, write=False)

    def read_region_subgrid(self) -> None:
        for name, fn in self.files["region_subgrid"].items():
            data = self._read_grid(fn)
            self.set_region_subgrid(data, name=name, write=False)

    def read_other(self) -> None:
        for name, fn in self.files["other"].items():
            da = open_zarr(Path(self.root) / fn)
            self.set_other(da, name=name, write=False)
        return None

    def read(self):
        with suppress_logging_warning(self.logger):
            self.files = self.read_file_library()

            self.read_geoms()
            self.read_array()
            self.read_table()
            self.read_dict()

            self.read_subgrid()
            self.read_grid()
            self.read_region_subgrid()

            self.read_other()

    def set_other(
        self,
        da,
        name: str,
        write=True,
        x_chunksize=XY_CHUNKSIZE,
        y_chunksize=XY_CHUNKSIZE,
        time_chunksize=1,
        *args,
        **kwargs,
    ):
        assert isinstance(da, xr.DataArray)

        if write:
            self.logger.info(f"Write {name}")

            fp = Path("other") / (name + ".zarr")
            self.files["other"][name] = fp

            dst_file = Path(self.root, fp)
            da = to_zarr(
                da,
                dst_file,
                x_chunksize=x_chunksize,
                y_chunksize=y_chunksize,
                time_chunksize=time_chunksize,
                crs=da.rio.crs,
                *args,
                **kwargs,
            )
        self.other[name] = da
        return self.other[name]

    def _set_grid(
        self,
        grid_name,
        grid,
        data: xr.DataArray,
        name: str,
        write,
        x_chunksize=XY_CHUNKSIZE,
        y_chunksize=XY_CHUNKSIZE,
    ):
        """Add data to grid.

        All layers of grid must have identical spatial coordinates.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to grid
        name: str
            Name of new map layer, this is used to overwrite the name of a DataArray
            and ignored if data is a Dataset
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
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, write=True
    ) -> None:
        self._set_grid("grid", self.grid, data, write=write, name=name)
        return self.grid[name]

    def set_subgrid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, write=True
    ) -> None:
        self.subgrid = self._set_grid(
            "subgrid", self.subgrid, data, write=write, name=name
        )
        return self.subgrid[name]

    def set_region_subgrid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, write=True
    ) -> None:
        self.region_subgrid = self._set_grid(
            "region_subgrid",
            self.region_subgrid,
            data,
            write=write,
            name=name,
        )
        return self.region_subgrid[name]

    def set_alternate_root(self, root, mode):
        relative_path = Path(os.path.relpath(Path(self.root), root.resolve()))
        for data in self.files.values():
            for name, fn in data.items():
                data[name] = relative_path / fn
        super().set_root(root, mode)

    @property
    def subgrid_factor(self):
        subgrid_factor = self.subgrid.dims["x"] // self.grid.dims["x"]
        assert subgrid_factor == self.subgrid.dims["y"] // self.grid.dims["y"]
        return subgrid_factor

    @property
    def ldd_scale_factor(self):
        scale_factor = (
            self.other["original_d8_flow_directions"].shape[0]
            // self.grid["mask"].shape[0]
        )
        assert scale_factor == (
            self.other["original_d8_flow_directions"].shape[1]
            // self.grid["mask"].shape[1]
        )
        return scale_factor

    @property
    def region(self):
        return self.geoms["mask"]

    @property
    def bounds(self):
        return self.grid.rio.bounds(recalc=True)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        self._root = Path(root).absolute()

    @property
    def preprocessing_dir(self):
        return Path(self.root).parent / "preprocessing"

    @property
    def report_dir(self):
        return Path(self.root).parent / "report"

    def full_like(self, data, fill_value, nodata, attrs=None, *args, **kwargs):
        ds = xr.full_like(data, fill_value, *args, **kwargs)
        ds.attrs = attrs or {}
        ds.attrs["_FillValue"] = nodata
        return ds

    def check_methods(self, opt):
        """Check all opt keys and raise sensible error messages if unknown."""
        for method in opt.keys():
            if not callable(getattr(self, method, None)):
                raise ValueError(f'Model {self._NAME} has no method "{method}"')
        return opt

    def run_method(self, method, *args, **kwargs):
        """Log method parameters before running a method."""
        func = getattr(self, method)
        signature = inspect.signature(func)
        # combine user and default options
        params = {}
        for i, (k, v) in enumerate(signature.parameters.items()):
            if k in ["args", "kwargs"]:
                if k == "args":
                    params[k] = args[i:]
                else:
                    params.update(**kwargs)
            else:
                v = kwargs.get(k, v.default)
                if len(args) > i:
                    v = args[i]
                params[k] = v
        # log options
        for k, v in params.items():
            if v is not inspect._empty:
                self.logger.info(f"{method}.{k}: {v}")
        return func(*args, **kwargs)

    def run_methods(self, methods):
        # then loop over other methods
        for method in methods:
            kwargs = {} if methods[method] is None else methods[method]
            self.run_method(method, **kwargs)
            self.write_file_library()

        self.logger.info("Finished!")

    def build(self, region: dict, methods: dict):
        methods = methods or {}
        methods = self.check_methods(methods)
        methods["setup_region"].update(region=region)

        self.run_methods(methods)

    def update(
        self,
        methods: dict,
    ):
        methods = methods or {}
        methods = self.check_methods(methods)

        if "setup_region" in methods:
            raise ValueError('"setup_region" can only be called when building a model.')

        self.run_methods(methods)

    def get_linear_indices(self, da):
        """Get linear indices for each cell in a 2D DataArray."""
        # Get the sizes of the spatial dimensions
        ny, nx = da.sizes["y"], da.sizes["x"]

        # Create an array of sequential integers from 0 to ny*nx - 1
        grid_ids = np.arange(ny * nx).reshape(ny, nx)

        # Create a DataArray with the same coordinates and dimensions as your spatial grid
        grid_id_da = xr.DataArray(
            grid_ids,
            coords={
                "y": da.coords["y"],
                "x": da.coords["x"],
            },
            dims=["y", "x"],
        )

        return grid_id_da

    def get_neighbor_cell_ids_for_linear_indices(self, cell_id, nx, ny, radius=1):
        """Get the linear indices of the neighboring cells of a cell in a 2D grid."""
        row = cell_id // nx
        col = cell_id % nx

        neighbor_cell_ids = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue  # Skip the cell itself
                r = row + dr
                c = col + dc
                if 0 <= r < ny and 0 <= c < nx:
                    neighbor_id = r * nx + c
                    neighbor_cell_ids.append(neighbor_id)
        return neighbor_cell_ids
