import networkx as nx
import geopandas as gpd


def get_upstream_subbasin_ids(data_catalog, subbasin_id):
    river_graph = build_river_network(data_catalog)
    return nx.ancestors(river_graph, subbasin_id)


def build_river_network(data_catalog):  # , reverse=False):
    river_network = gpd.read_file(
        data_catalog.get_source("MERIT_Basins_rivers").path,
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
        data_catalog.get_source("MERIT_Basins_basins").path,
        bbox=(lon, lat, lon, lat),
    )
    assert len(COMID) == 1, "The point is not in a single basin"
    # get the COMID value from the GeoDataFrame
    return COMID["COMID"].values[0]


def get_subbasins(data_catalog, subbasin_ids):
    subbasins = gpd.read_file(
        data_catalog.get_source("MERIT_Basins_basins").path,
        sql=f"""SELECT * FROM cat_pfaf_MERIT_Hydro_v07_Basins_v01_bugfix1 WHERE COMID IN ({
            ",".join([str(ID) for ID in subbasin_ids])
        })""",
    )
    assert len(subbasins) == len(subbasin_ids), "Some subbasins were not found"
    return subbasins
