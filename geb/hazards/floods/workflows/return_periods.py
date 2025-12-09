"""Functions to determine calculation groups for river segments based on their topology."""

import geopandas as gpd
import numpy as np
import numpy.typing as npt

from .utils import get_end_point


def get_topological_stream_order(rivers: gpd.GeoDataFrame) -> npt.NDArray[np.int32]:
    """Calculate the topological stream order for each river segment.

    The topological stream order is calculated by following the river rivers upstream
    and each time finding the rivers that connect to the previous segment.

    The first coordinate of the geometry is the start point, the last coordinate is the end point.

    Args:
        rivers: GeoDataFrame with the river. Start and end points must be connected.

    Returns:
        Array with the topological stream order for each river segment.
    """
    # find endpoints of each geometry
    startpoint = rivers.geometry.apply(lambda geom: geom.coords[0])
    endpoint = rivers.geometry.apply(lambda geom: geom.coords[-1])

    topological_stream_order: npt.NDArray[np.int32] = np.full(
        len(rivers), -1, dtype=np.int32
    )

    prev_order_idx = ~endpoint.isin(startpoint)
    topological_stream_order_idx: int = 0
    topological_stream_order[prev_order_idx] = topological_stream_order_idx

    prev_order_start_points = startpoint[prev_order_idx]

    while len(prev_order_start_points) > 0:
        topological_stream_order_idx += 1

        next_order_idx = endpoint.isin(prev_order_start_points)
        topological_stream_order[next_order_idx] = topological_stream_order_idx

        next_order_start_points = startpoint[next_order_idx]
        prev_order_start_points = next_order_start_points

    assert (topological_stream_order != -1).all()

    return topological_stream_order


def assign_calculation_group(rivers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign calculation groups to river segments based on their topological stream order and endpoints.

    We want to run every river rivers independently of the other river rivers. Here, we make
    two observations:

    1. If two river rivers have the same endpoint, they must performed in another run.
    2. If a river shares its endpoint with the startpoint of another river, they must be performed in another run.

    Thus, we first split rivers by their topological stream order. If more there is at least one stream
    order between two rivers, they can be performed in the same run. We can simplify this to running every
    even stream order in one run and every odd stream order in another run.

    Furthermore, when two (or in rare cases three) rivers share the same endpoint, we must also run them
    in different runs. Therefore, we group the rivers by their endpoint and assign them to a calculation group.

    The outcome is calculation groups 1-6, where 3 and 6 are only used when there are three rivers
    sharing the same endpoint.

    Args:
        rivers: GeoDataFrame with the river rivers

    Returns:
        GeoDataFrame with an additional column 'calculation_group' that assigns the rivers to a calculation group
    """

    def assign(group: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        # There cannot be four rivers with the same endpoint (there must be an outflow point)
        assert len(group) <= 3, "Found more than 3 nodes with the same endpoint"
        # Stream order must be the same for rivers within a group
        assert group["topological_stream_order"].nunique() == 1, (
            f"Found nodes with different topological stream order: {group['topological_stream_order'].unique()}"
        )
        # Use modulo 2 to assign even and odd stream orders to different calculation groups,
        # then multiply by 3 to get 0 and 3 as calculation groups (leaving 1, 2 and 3)
        calculation_group = (group["topological_stream_order"] % 2 * 3).values
        # for one of the branches in a group, increment the calculation group by 1
        # using the previously open 1 and 3 group
        if calculation_group.size == 2:
            calculation_group[1] += 1
        elif calculation_group.size == 3:
            calculation_group[1] += 1
            calculation_group[2] += 2
        group["calculation_group"] = calculation_group

        return group

    return (
        rivers.groupby(rivers["geometry"].apply(get_end_point))
        .apply(assign)
        .reset_index(drop=True)  # ty: ignore[invalid-return-type]
    )
