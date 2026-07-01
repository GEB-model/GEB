"""Workflows for the hydrography build files."""

import networkx
import pandas as pd

from geb.build.data_catalog import DataCatalog


def calculate_shreve_stream_order(rivers: pd.DataFrame) -> pd.Series:
    """Calculate the Shreve stream order for each river segment.

    Args:
        rivers: A DataFrame containing river segments with at least downstream_ID.
            the river id is expected to be the index of the DataFrame.

    Returns:
        A Series indexed by river id with the Shreve stream order as values.

    Notes:
        Shreve stream order is calculated by assigning an order of 1 to all sources.
        For any other segment, the order is the sum of the orders of its upstream
        tributaries. This is calculated using a topological sort to ensure
        all upstream segments are processed before their downstream children.
    """
    # Create the river graph to find dependencies
    # We use COMID as nodes and downstream_ID to define edges
    # Shreve order requires knowing upstream contributors
    river_graph = networkx.DiGraph()

    # Filter out terminals (downstream_ID == -1) and add edges
    assert rivers.dtype == "int64", (
        "Expected rivers DataFrame to have int64 dtype for downstream_ID"
    )
    edges: list[tuple[int, int]] = [
        (idx, row["downstream_ID"])
        for idx, row in rivers.iterrows()
        if row["downstream_ID"] != -1
    ]  # ty:ignore[invalid-assignment]
    river_graph.add_edges_from(edges)

    # Add all river IDs as nodes to ensure isolated segments are included
    river_graph.add_nodes_from(rivers.index)

    # Initialize shreve order series
    shreve_orders: pd.Series = pd.Series(0, index=rivers.index, dtype=int)

    # Perform topological sort to process nodes from headwaters to outlets
    # topological_sort returns a flat list where u comes before v if (u, v) is an edge
    sorted_nodes = list(networkx.topological_sort(river_graph))

    for node in sorted_nodes:
        # Find upstream nodes that flow into this one
        # Because edges are (Upstream, Downstream), predecessors are upstream
        upstream_nodes: list = list(river_graph.predecessors(node))

        if not upstream_nodes:
            # Source node (headwater)
            shreve_orders[node] = 1
        else:
            # Sum the shreve orders of all upstream contributors
            shreve_orders[node] = shreve_orders[upstream_nodes].sum()

    return shreve_orders


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
