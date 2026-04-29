import networkx
import pandas as pd

from geb.build.data_catalog import DataCatalog


def get_river_graph(data_catalog: DataCatalog) -> networkx.DiGraph:
    """Create a directed graph for the river network.

    Args:
        data_catalog: Data catalog containing the MERIT basins.

    Returns:
        A directed graph where nodes are COMID values and edges point downstream.
    """
    # load river network data
    river_network = (
        data_catalog.fetch("merit_basins_rivers")
        .read(columns=["COMID", "NextDownID"])
        .set_index("COMID")
    )
    assert isinstance(river_network, pd.DataFrame)
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
