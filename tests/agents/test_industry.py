"""Tests for the industry agent functions."""

import geopandas as gpd
import numpy as np

from geb.agents.industry import create_abstraction_areas


def test_create_abstraction_areas() -> None:
    """Test the create_abstraction_areas function."""
    # Setup mock data
    # 4 grid cells, 2 basins
    basin_ids = np.array([10, 10, 11, 11], dtype=np.int32)
    # 4 grid cells, 2 rivers
    river_ids = np.array([10, 10, 11, 11], dtype=np.int32)

    # Mock rivers GeoDataFrame
    # River 10 flows into River 11
    rivers_data = {
        "shreve_stream_order": [1, 6],
        "represented_in_grid": [True, True],
        "downstream_ID": [11, -1],
    }
    rivers = gpd.GeoDataFrame(rivers_data, index=[10, 11])

    # Test case 1: minimum_shreve_stream_order = 6
    # River 10 (order 1) should drain into River 11 (order 6)
    # Basin 10 (cells 0, 1) associated with River 10 -> should be mapped to River 11
    # Basin 11 (cells 2, 3) associated with River 11 -> should be mapped to River 11
    area_indices, river_indices = create_abstraction_areas(
        basin_ids, river_ids, rivers, minimum_shreve_stream_order=6
    )

    # Both basins should be mapped to River 11
    # So we should have 1 abstraction area (River 11)
    assert area_indices.shape[0] == 1
    assert river_indices.shape[0] == 1

    # Area indices should contain all cells (0, 1, 2, 3)
    expected_area = {0, 1, 2, 3}
    actual_area = set(area_indices[0][area_indices[0] != -1])
    assert actual_area == expected_area

    # River indices should contain cells for River 11 (2, 3)
    expected_river = {2, 3}
    actual_river = set(river_indices[0][river_indices[0] != -1])
    assert actual_river == expected_river

    # Test case 2: minimum_shreve_stream_order = 1
    # Both rivers are valid abstraction rivers
    area_indices, river_indices = create_abstraction_areas(
        basin_ids, river_ids, rivers, minimum_shreve_stream_order=1
    )

    # We should have 2 abstraction areas (River 10 and River 11)
    assert area_indices.shape[0] == 2
    assert river_indices.shape[0] == 2

    # Find which row is which by looking at river_indices
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

    # Basin 10 (0, 1) -> River 10 (0, 1)
    assert set(area_indices[row_10][area_indices[row_10] != -1]) == {0, 1}
    # Basin 11 (2, 3) -> River 11 (2, 3)
    assert set(area_indices[row_11][area_indices[row_11] != -1]) == {2, 3}

    # Check river indices
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
