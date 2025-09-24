"""Tests for the HRU functions.

The HRUs are hydrological response units, which are subdivisions of grid cells. There are
several utility functions to go from grid to HRU and back again.
"""

import numpy as np
import numpy.typing as npt
import pytest

from geb.hydrology.HRUs import determine_nearest_river_cell, to_grid, to_HRU


def test_determine_nearest_river_cell() -> None:
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
def HRU_data_with_nan() -> npt.NDArray[np.float32]:
    """Test data for data that could be on the HRUs used in GEB, with a NaN value.

    Returns:
        An array with hypothetical HRU data including a NaN. Corresponds with other fixutes in this file.
    """
    return np.array([1, np.nan, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)


@pytest.fixture
def grid_to_HRU() -> npt.NDArray[np.int32]:
    """The indexes that map grid cells to HRUs.

    Each value maps to the index of the first unit of the next cell.

    Returns:
        Array of size of the compressed grid cells. Each value maps to the index of the first unit of the next cell.
            Corresponds with other fixutes in this file.
    """
    return np.array([1, 4, 9], dtype=np.int32)


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
    HRU_data_with_nan: npt.NDArray[np.float32],
    grid_to_HRU: npt.NDArray[np.int32],
    land_use_ratio: npt.NDArray[np.float32],
) -> None:
    """Test the to_grid function with various aggregation methods for going from HRU to grid.

    Args:
        HRU_data: Hypothetical data on the HRUs.
        HRU_data_with_nan: Hypothetical data on the HRUs with a NaN value.
        grid_to_HRU: The indexes that map grid cells to HRUs.
        land_use_ratio: The land use ratio (of a grid cell) for each HRU.
    """
    np.testing.assert_almost_equal(
        to_grid(HRU_data, grid_to_HRU, land_use_ratio),
        np.array([1.0, 3.2, 7.0], dtype=np.float32),
        decimal=1,
    )
    np.testing.assert_almost_equal(
        to_grid(
            HRU_data,
            grid_to_HRU,
            land_use_ratio,
            fn="weightedmean",
        ),
        np.array([1.0, 3.2, 7.0], dtype=np.float32),
        decimal=1,
    )
    np.testing.assert_almost_equal(
        to_grid(
            HRU_data_with_nan,
            grid_to_HRU,
            land_use_ratio,
            fn="weightednanmean",
        ),
        np.array([1.0, 3.5, 7.0], dtype=np.float32),
        decimal=1,
    )
    np.testing.assert_almost_equal(
        to_grid(
            HRU_data,
            grid_to_HRU,
            land_use_ratio,
            fn="sum",
        ),
        np.array([1, 9, 35], dtype=np.float32),
        decimal=1,
    )
    np.testing.assert_almost_equal(
        to_grid(
            HRU_data_with_nan,
            grid_to_HRU,
            land_use_ratio,
            fn="nansum",
        ),
        np.array([1, 7, 35], dtype=np.float32),
        decimal=1,
    )
    np.testing.assert_almost_equal(
        to_grid(
            HRU_data,
            grid_to_HRU,
            land_use_ratio,
            fn="max",
        ),
        np.array([1, 4, 9], dtype=np.float32),
        decimal=1,
    )
    np.testing.assert_almost_equal(
        to_grid(
            HRU_data,
            grid_to_HRU,
            land_use_ratio,
            fn="min",
        ),
        np.array([1, 2, 5], dtype=np.float32),
        decimal=1,
    )


def test_to_HRU(
    grid_data: npt.NDArray[np.float32],
    grid_to_HRU: npt.NDArray[np.int32],
    land_use_ratio: npt.NDArray[np.float32],
) -> None:
    """Test the to_HRU function with various methods for going from grid to HRU.

    Args:
        grid_data: Hypothetical data on the grid.
        grid_to_HRU: The indexes that map grid cells to HRUs.
        land_use_ratio: The land use ratio (of a grid cell) for each HRU.
    """
    data = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32)

    out = np.zeros((*data.shape[:-1], land_use_ratio.size), dtype=data.dtype)
    np.testing.assert_almost_equal(
        to_HRU(grid_data, grid_to_HRU, land_use_ratio, out),
        data,
    )
    np.testing.assert_almost_equal(out, data)

    data = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32)
    np.testing.assert_almost_equal(
        to_HRU(grid_data, grid_to_HRU, land_use_ratio, out, fn=None), data
    )
    np.testing.assert_almost_equal(out, data)

    data = np.array([1.0, 0.4, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6], dtype=np.float32)
    np.testing.assert_almost_equal(
        to_HRU(grid_data, grid_to_HRU, land_use_ratio, out, fn="weightedsplit"),
        data,
    )
    np.testing.assert_almost_equal(out, data)
