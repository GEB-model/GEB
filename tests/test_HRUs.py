import pytest
import numpy as np

from geb.HRUs import to_grid, to_HRU


@pytest.fixture
def common_data():
    grid_data = np.array([1, 2, 3], dtype=np.float32)
    HRU_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    HRU_data_with_nan = np.array([1, np.nan, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    grid_to_HRU = np.array([1, 4, 9])
    land_use_ratio = np.array(
        [1, 0.2, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32
    )
    return grid_data, HRU_data, HRU_data_with_nan, grid_to_HRU, land_use_ratio


def test_to_grid(common_data):
    grid_data, HRU_data, HRU_data_with_nan, grid_to_HRU, land_use_ratio = common_data

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


def test_to_HRU(common_data):
    grid_data, HRU_data, HRU_data_with_nan, grid_to_HRU, land_use_ratio = common_data

    np.testing.assert_almost_equal(
        to_HRU(grid_data, grid_to_HRU, land_use_ratio),
        np.array([1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_almost_equal(
        to_HRU(grid_data, grid_to_HRU, land_use_ratio, fn=None),
        np.array([1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_almost_equal(
        to_HRU(grid_data, grid_to_HRU, land_use_ratio, fn="weightedsplit"),
        np.array([1.0, 0.4, 0.8, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6], dtype=np.float32),
    )
