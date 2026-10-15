"""Tests for the water demand module and abstraction area functions."""

import geopandas as gpd
import numpy as np

from geb.hydrology.water_demand import (
    assign_demand_and_return_flow_to_abstraction_rivers,
    assign_demand_to_abstraction_rivers,
    create_abstraction_areas,
)


def test_create_abstraction_areas() -> None:
    """Test the create_abstraction_areas function with stream order filtering."""
    # Setup mock data: 4 grid cells, 2 basins, 2 rivers
    basin_ids = np.array([10, 10, 11, 11], dtype=np.int32)
    river_ids = np.array([10, 10, 11, 11], dtype=np.int32)

    # Mock rivers GeoDataFrame: River 10 flows into River 11
    rivers_data = {
        "shreve_stream_order": [1, 6],
        "represented_in_grid": [True, True],
        "downstream_ID": [11, -1],
    }
    rivers = gpd.GeoDataFrame(rivers_data, index=[10, 11])

    # Test case 1: minimum_shreve_stream_order = 6
    # River 10 (order 1) drains into River 11 (order 6)
    area_indices, river_indices = create_abstraction_areas(
        basin_ids, river_ids, rivers, minimum_shreve_stream_order=6
    )

    assert area_indices.shape[0] == 1
    assert river_indices.shape[0] == 1
    assert set(area_indices[0][area_indices[0] != -1]) == {0, 1, 2, 3}
    assert set(river_indices[0][river_indices[0] != -1]) == {2, 3}

    # Test case 2: minimum_shreve_stream_order = 1
    # Both rivers are valid abstraction rivers
    area_indices, river_indices = create_abstraction_areas(
        basin_ids, river_ids, rivers, minimum_shreve_stream_order=1
    )

    assert area_indices.shape[0] == 2
    assert river_indices.shape[0] == 2

    row_10 = -1
    row_11 = -1
    for i in range(2):
        r_idx = set(river_indices[i][river_indices[i] != -1])
        if r_idx == {0, 1}:
            row_10 = i
        elif r_idx == {2, 3}:
            row_11 = i

    assert row_10 != -1
    assert row_11 != -1

    assert set(area_indices[row_10][area_indices[row_10] != -1]) == {0, 1}
    assert set(area_indices[row_11][area_indices[row_11] != -1]) == {2, 3}
    assert set(river_indices[row_10][river_indices[row_10] != -1]) == {0, 1}
    assert set(river_indices[row_11][river_indices[row_11] != -1]) == {2, 3}


def test_create_abstraction_areas_no_valid_rivers() -> None:
    """Test create_abstraction_areas when no rivers meet the threshold."""
    basin_ids = np.array([10, 11], dtype=np.int32)
    river_ids = np.array([10, 11], dtype=np.int32)
    rivers_data = {
        "shreve_stream_order": [1, 2],
        "represented_in_grid": [True, True],
        "downstream_ID": [-1, -1],
    }
    rivers = gpd.GeoDataFrame(rivers_data, index=[10, 11])

    area_indices, river_indices = create_abstraction_areas(
        basin_ids, river_ids, rivers, minimum_shreve_stream_order=6
    )

    assert area_indices.shape == (0, 0)
    assert river_indices.shape == (0, 0)


def test_create_abstraction_areas_uses_minus_one_padding() -> None:
    """Test that abstraction area outputs are padded with -1 where needed."""
    basin_ids = np.array([10, 11, 11], dtype=np.int32)
    river_ids = np.array([10, 11, 11], dtype=np.int32)
    rivers = gpd.GeoDataFrame(
        {
            "shreve_stream_order": [1, 6],
            "represented_in_grid": [True, True],
            "downstream_ID": [11, -1],
        },
        index=[10, 11],
    )

    area_indices, river_indices = create_abstraction_areas(
        basin_ids, river_ids, rivers, minimum_shreve_stream_order=1
    )

    assert area_indices.shape == (2, 2)
    assert river_indices.shape == (2, 2)
    assert np.count_nonzero(area_indices == -1) == 1
    assert np.count_nonzero(river_indices == -1) == 1


def test_assign_demand_to_abstraction_rivers() -> None:
    """Test assigning water demand across abstraction areas to rivers."""
    # 4 grid cells
    # Area 0 covers cells 0, 1, 2, 3; target river is cells 2, 3
    area_indices = np.array([[0, 1, 2, 3]], dtype=np.int32)
    river_indices = np.array([[2, 3, -1, -1]], dtype=np.int32)

    # Water demand: 10 in cell 0, 10 in cell 1, 0 in cells 2 and 3 -> total 20
    water_demand = np.array([10.0, 10.0, 0.0, 0.0], dtype=np.float32)

    assigned_demand = assign_demand_to_abstraction_rivers(
        water_demand=water_demand,
        abstraction_area_indices=area_indices,
        abstraction_river_indices=river_indices,
    )

    # Total demand of 20 should be distributed equally between river cells (2, 3) -> 10 each
    expected = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    np.testing.assert_allclose(assigned_demand, expected)


def test_assign_demand_and_return_flow_to_abstraction_rivers() -> None:
    """Test assigning demand and calculating return flow to abstraction rivers."""
    area_indices = np.array([[0, 1, 2, 3]], dtype=np.int32)
    river_indices = np.array([[2, 3]], dtype=np.int32)

    # Water demand: 30 total
    water_demand = np.array([15.0, 15.0, 0.0, 0.0], dtype=np.float32)
    # Water consumption: 10 total -> return flow = 30 - 10 = 20 total
    water_consumption = np.array([5.0, 5.0, 0.0, 0.0], dtype=np.float32)

    assigned_demand, assigned_return_flow = (
        assign_demand_and_return_flow_to_abstraction_rivers(
            water_demand=water_demand,
            water_consumption=water_consumption,
            abstraction_area_indices=area_indices,
            abstraction_river_indices=river_indices,
        )
    )

    # Demand: 30 / 2 = 15 per river cell
    # Return flow: 20 / 2 = 10 per river cell
    np.testing.assert_allclose(
        assigned_demand, np.array([0.0, 0.0, 15.0, 15.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        assigned_return_flow, np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    )


def test_assign_demand_no_abstraction_areas() -> None:
    """Test assigning demand when no abstraction areas exist."""
    area_indices = np.zeros((0, 0), dtype=np.int32)
    river_indices = np.zeros((0, 0), dtype=np.int32)
    water_demand = np.array([5.0, 10.0], dtype=np.float32)

    assigned_demand = assign_demand_to_abstraction_rivers(
        water_demand=water_demand,
        abstraction_area_indices=area_indices,
        abstraction_river_indices=river_indices,
    )
    np.testing.assert_allclose(assigned_demand, np.zeros(2, dtype=np.float32))
