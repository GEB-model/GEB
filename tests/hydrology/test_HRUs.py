"""Tests for the HRU functions.

The HRUs are hydrological response units, which are subdivisions of grid cells. There are
several utility functions to go from grid to HRU and back again.
"""

import numpy as np
import numpy.typing as npt
import pytest

from geb.hydrology.HRUs import (
    determine_nearest_river_cell,
    to_grid,
    to_HRU,
)


def test_determine_nearest_river_cell() -> None:
    """Test whether the nearest river cell determination works as expected."""
    upstream_area = np.array(
        [
            [np.nan, np.nan, 10, np.nan, np.nan],
            [np.nan, np.nan, 9, np.nan, np.nan],
            [3, 5, 8, 2, np.nan],
            [4, 2, 7, 1, 2],
            [2, 2, 6, 3, np.nan],
        ]
    )
    mask = np.array(
        [
            [True, True, False, True, True],
            [True, True, False, True, True],
            [False, False, False, False, True],
            [False, False, False, False, False],
            [False, False, False, False, True],
        ]
    )
    n_non_masked = (~mask).sum()

    # simulate that last grid cell has 2 HRUs
    HRU_to_grid = np.arange(n_non_masked + 1, dtype=np.int32)
    HRU_to_grid[-1] = n_non_masked - 1

    nearest_river = determine_nearest_river_cell(
        upstream_area,
        HRU_to_grid=HRU_to_grid,
        mask=mask,
        threshold_m2=5,
    )

    np.testing.assert_array_equal(
        nearest_river,
        np.array([0, 1, 4, 4, 4, 4, 8, 8, 8, 8, 8, 13, 13, 13, 13, 13], dtype=np.int32),
    )


@pytest.fixture
def grid_data() -> npt.NDArray[np.float32]:
    """Test data for data that could be on the grid used in GEB.

    Because it is compressed, it is a 1D array.

    Returns:
        An array with hypothetical grid data. Corresponds with other fixutes in this file.
    """
    return np.array([1, 2, 3], dtype=np.float32)


@pytest.fixture
def HRU_data() -> npt.NDArray[np.float32]:
    """Test data for data that could be on the HRUs used in GEB.

    Returns:
        An array with hypothetical HRU data. Corresponds with other fixutes in this file.
    """
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)


@pytest.fixture
def land_use_ratio() -> npt.NDArray[np.float32]:
    """The land use ratio for each HRU, as a fraction of the grid cell.

    The size of this array is the number of HRUs. The size of the land use in one grid cell
    is multiple HRUs that sum to 1.

    Returns:
        An array with hypothetical land use ratios for each HRU. Corresponds with other fixutes in this file.
    """
    return np.array([1, 0.2, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)


def test_to_grid(
    HRU_data: npt.NDArray[np.float32],
    grid_to_HRU: npt.NDArray[np.int32],
    land_use_ratio: npt.NDArray[np.float32],
) -> None:
    """Test the to_grid function for going from HRU to grid using a weighted mean.

    Tests both 1D (single timestep) and 2D (multiple timesteps) input shapes.

    Args:
        HRU_data: Hypothetical data on the HRUs.
        grid_to_HRU: Exclusive end index of each grid cell's HRUs.
        land_use_ratio: The land use ratio (of a grid cell) for each HRU.
    """
    expected_1d = np.array([1.0, 3.2, 7.0], dtype=np.float32)
    np.testing.assert_almost_equal(
        to_grid(HRU_data, grid_to_HRU, land_use_ratio),
        expected_1d,
        decimal=5,
    )

    # 2D case: two timesteps stacked as rows (n_timesteps × n_HRUs)
    HRU_data_2d = np.vstack([HRU_data, HRU_data * 2])
    expected_2d = np.vstack([expected_1d, expected_1d * 2])
    np.testing.assert_almost_equal(
        to_grid(HRU_data_2d, grid_to_HRU, land_use_ratio),
        expected_2d,
        decimal=5,
    )


@pytest.fixture
def grid_to_HRU() -> npt.NDArray[np.int32]:
    """Exclusive end index of each grid cell's HRUs in the sorted HRU array.

    grid_to_HRU[i] is the first HRU index that belongs to grid cell i+1,
    i.e. the exclusive end of cell i's HRUs.

    Returns:
        Array of size n_grid_cells. Corresponds with other fixtures in this file.
    """
    return np.array([1, 4, 9], dtype=np.int32)


@pytest.fixture
def HRU_to_grid() -> npt.NDArray[np.int32]:
    """The indexes that map HRUs to grid cells.

    Each value is the compressed grid cell index that the HRU belongs to.

    Returns:
        Array of size n_HRUs mapping each HRU to its parent grid cell.
            Corresponds with other fixtures in this file.
    """
    return np.array([0, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.int32)


def test_to_HRU(
    grid_data: npt.NDArray[np.float32],
    HRU_to_grid: npt.NDArray[np.int32],
) -> None:
    """Test the to_HRU function for going from grid to HRU.

    Args:
        grid_data: Hypothetical data on the grid.
        HRU_to_grid: The indexes that map HRUs to grid cells.
    """
    expected = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32)
    result = to_HRU(grid_data, HRU_to_grid)
    np.testing.assert_almost_equal(result, expected)


def test_create_HRUs_splits_by_elevation_zone() -> None:
    """Farm fields and land use types are split by the elevation zone of their subcells."""
    from geb.hydrology.HRUs import HRUs

    scaling = 4
    mask = np.array([[False, False], [True, False]])
    farms = np.full((8, 8), -1, dtype=np.int32)
    farms[0:2, 0:3] = 0
    farms[5:8, 5:8] = 1
    rng = np.random.default_rng(0)
    land_use = np.where(farms >= 0, 1, rng.choice([0, 1, 4, 5], size=(8, 8))).astype(
        np.int32
    )
    zones = np.zeros((8, 8), dtype=np.int32)
    zones[:, 2:] = 1
    zones[6:, :] = 2

    (
        land_use_type,
        land_use_ratio,
        land_owners,
        HRU_to_grid,
        _,
        linear_mapping,
        elevation_zone,
    ) = HRUs.create_HRUs_numba(farms, land_use, zones, mask, scaling, 4)

    # every subcell of a HRU has the farm, land use and zone of the HRU
    for HRU in range(land_use_type.size):
        subcells = linear_mapping == HRU
        assert (zones[subcells] == elevation_zone[HRU]).all()
        assert (land_use[subcells] == land_use_type[HRU]).all()
        assert (farms[subcells] == land_owners[HRU]).all()
    # the masked cell has no HRUs, the three other cells are fully covered
    assert (linear_mapping[4:8, 0:4] == -1).all()
    np.testing.assert_allclose(land_use_ratio.sum(), 3.0)
    # farm 0 lies in zones 0 and 1 of its cell: two HRUs with the same owner
    assert sorted(elevation_zone[land_owners == 0].tolist()) == [0, 1]

    without_zones = HRUs.create_HRUs_numba(
        farms, land_use, np.zeros_like(zones), mask, scaling, 4
    )
    assert without_zones[0].size < land_use_type.size
