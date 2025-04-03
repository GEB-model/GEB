import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import rasterize
from shapely.geometry import LineString, Point


def get_upstream_subbasin_ids(river_graph, subbasin_ids):
    ancenstors = set()
    for subbasin_id in subbasin_ids:
        ancenstors |= nx.ancestors(river_graph, subbasin_id)
    return ancenstors


def get_downstream_subbasins(river_graph, sink_subbasin_ids):
    downstream_subbasins = {}
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


def get_river_graph(data_catalog):
    river_network = pd.read_parquet(
        data_catalog.get_source("MERIT_Basins_riv").path,
        columns=["COMID", "NextDownID"],
    ).set_index("COMID")
    assert river_network.index.name == "COMID", (
        "The index of the river network is not the COMID column"
    )

    # create a directed graph for the river network
    river_graph = nx.DiGraph()

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


def get_subbasins_geometry(data_catalog, subbasin_ids):
    subbasins = gpd.read_parquet(
        data_catalog.get_source("MERIT_Basins_cat").path,
        filters=[
            ("COMID", "in", subbasin_ids),
        ],
    )
    assert len(subbasins) == len(subbasin_ids), "Some subbasins were not found"
    return subbasins.set_index("COMID")


def get_rivers(data_catalog, subbasin_ids):
    rivers = gpd.read_parquet(
        data_catalog.get_source("MERIT_Basins_riv").path,
        columns=["COMID", "lengthkm", "uparea", "maxup", "geometry"],
        filters=[
            ("COMID", "in", subbasin_ids),
        ],
    )
    assert len(rivers) == len(subbasin_ids), "Some rivers were not found"
    # reverse the river lines to have the downstream direction
    rivers["geometry"] = rivers["geometry"].apply(
        lambda x: LineString(list(x.coords)[::-1])
    )
    return rivers.set_index("COMID")


def create_river_raster_from_river_lines(
    rivers, flwdir_idxs_out, original_d8_elevation
):
    river_raster = rasterize(
        zip(rivers.geometry, rivers.index),
        out_shape=original_d8_elevation.shape,
        fill=-1,
        dtype=np.int32,
        transform=original_d8_elevation.rio.transform(),
        all_touched=False,  # because this is a line, Bresenham's line algorithm is used, which is perfect here :-)
    )

    river_raster_coarsened = river_raster.ravel()[
        flwdir_idxs_out.values.ravel()
    ].reshape(flwdir_idxs_out.shape)
    return river_raster_coarsened


def get_SWORD_translation_IDs_and_lenghts(data_catalog, rivers):
    MERIT_Basins_to_SWORD = data_catalog.get_source("MERIT_Basins_to_SWORD").path
    SWORD_files = []
    for SWORD_region in (
        list(range(11, 19))
        + list(range(21, 30))
        + list(range(31, 37))
        + list(range(41, 50))
        + list(range(51, 58))
        + list(range(61, 68))
        + list(range(71, 79))
        + list(range(81, 87))
        + [91]
    ):
        SWORD_files.append(MERIT_Basins_to_SWORD.format(SWORD_Region=str(SWORD_region)))
    assert len(SWORD_files) == 61, "There are 61 SWORD regions"
    MERIT_Basins_to_SWORD = (
        xr.open_mfdataset(SWORD_files).sel(mb=rivers.index.tolist()).compute()
    )

    SWORD_reach_IDs = np.full((40, len(rivers)), dtype=np.int64, fill_value=-1)
    SWORD_reach_lengths = np.full(
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


def get_SWORD_river_widths(data_catalog, SWORD_reach_IDs):
    # NOTE: To open SWORD NetCDFs: xr.open_dataset("oc_sword_v16.nc", group="reaches/discharge_models/constrained/MOMMA")
    unique_SWORD_reach_ids = np.unique(SWORD_reach_IDs[SWORD_reach_IDs != -1])

    SWORD = gpd.read_file(
        data_catalog.get_source("SWORD").path,
        sql=f"""SELECT * FROM sword_reaches_v16 WHERE reach_id IN ({",".join([str(ID) for ID in unique_SWORD_reach_ids])})""",
    ).set_index("reach_id")

    assert len(SWORD) == len(unique_SWORD_reach_ids), (
        "Some SWORD reaches were not found, possibly the SWORD and MERIT data version are not correct"
    )

    def lookup_river_width(reach_id):
        if reach_id == -1:  # when no SWORD reach is found
            return np.nan  # return NaN
        return SWORD.loc[reach_id, "width"]

    SWORD_river_width = np.vectorize(lookup_river_width)(SWORD_reach_IDs)
    return SWORD_river_width
