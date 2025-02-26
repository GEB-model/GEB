import networkx as nx
import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio.features import rasterize


def get_upstream_subbasin_ids(river_graph, subbasin_id):
    return nx.ancestors(river_graph, subbasin_id)


def get_downstream_subbasin(river_graph, subbasin_id):
    downstream_node = list(river_graph.neighbors(subbasin_id))
    if len(downstream_node) == 0:
        return None
    else:
        assert len(downstream_node) == 1, (
            "A subbasin has more than one downstream subbasin"
        )
        return downstream_node[0]


def get_river_graph(data_catalog):  # , reverse=False):
    river_network = gpd.read_file(
        data_catalog.get_source("MERIT_Basins_riv").path,
        columns=["COMID", "NextDownID"],
        ignore_geometry=True,
    )
    # remove all rivers without downstream connection
    river_network = river_network[river_network["NextDownID"] != 0]

    river_network = river_network.itertuples(index=False, name=None)

    # create a directed graph from the river network
    river_graph = nx.DiGraph()
    river_graph.add_edges_from(river_network)

    return river_graph


def get_subbasin_id_from_coordinate(data_catalog, lon, lat):
    # we select the basin that contains the point. To do so
    # we use a bounding box with the point coordinates, thus
    # xmin == xmax and ymin == ymax
    COMID = gpd.read_file(
        data_catalog.get_source("MERIT_Basins_cat").path,
        bbox=(lon, lat, lon, lat),
    )
    if len(COMID) == 0:
        raise ValueError(
            f"The point is not in a basin. Note, that there are some holes in the MERIT basins dataset ({data_catalog.get_source('MERIT_Basins_cat').path}), ensure that the point is in a basin."
        )
    assert len(COMID) == 1, "The point is not in a single basin"
    # get the COMID value from the GeoDataFrame
    return COMID["COMID"].values[0]


def get_subbasins(data_catalog, subbasin_ids):
    subbasins = gpd.read_file(
        data_catalog.get_source("MERIT_Basins_cat").path,
        sql=f"""SELECT * FROM cat_pfaf_MERIT_Hydro_v07_Basins_v01 WHERE COMID IN ({
            ",".join([str(ID) for ID in subbasin_ids])
        })""",
    )
    assert len(subbasins) == len(subbasin_ids), "Some subbasins were not found"
    return subbasins


def get_rivers(data_catalog, subbasin_ids):
    rivers = gpd.read_file(
        data_catalog.get_source("MERIT_Basins_riv").path,
        sql=f"""SELECT * FROM riv_pfaf_MERIT_Hydro_v07_Basins_v01 WHERE COMID IN ({
            ",".join([str(ID) for ID in subbasin_ids])
        })""",
    )
    assert len(rivers) == len(subbasin_ids), "Some rivers were not found"
    return rivers


def create_river_raster_from_river_lines(rivers, flwdir_idxs_out, hydrography):
    river_raster = rasterize(
        zip(rivers.geometry, rivers.index),
        out_shape=hydrography["flwdir"].shape,
        fill=-1,
        dtype=np.int32,
        transform=hydrography.rio.transform(),
        all_touched=False,  # because this is a line, Bresenham's line algorithm is used, which is perfect here :-)
    )

    river_raster_coarsened = river_raster.ravel()[flwdir_idxs_out.ravel()].reshape(
        flwdir_idxs_out.shape
    )
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
