"""Tests for farmer workflows (create_farms_numba)."""

import numpy as np

from geb.build.workflows.farmers import create_farms_numba


def test_create_farms_numba_no_farms() -> None:
    """When there are no farms and no cultivated land, the result is all -1."""
    # No cultivated land (all zeros), and no farmers
    cultivated_land = np.zeros((3, 3), dtype=np.int32)
    ids = np.array([], dtype=np.int32)
    farm_sizes = np.array([], dtype=np.int32)

    np.random.seed(0)
    farms = create_farms_numba(cultivated_land, ids, farm_sizes)

    assert farms.shape == cultivated_land.shape
    # All cells should be -1 (non-farm land)
    assert np.all(farms == -1)


def test_create_farms_numba_some_farmers() -> None:
    """Allocate a small set of farms across a contiguous cultivated block."""
    # 4x5 grid with a 2x3 cultivated block (6 cells)
    cultivated_land = np.zeros((4, 5), dtype=np.int32)
    cultivated_land[0:2, 0:3] = 1

    # Two farmers with sizes summing to cultivated cells
    ids = np.array([1, 2], dtype=np.int32)
    farm_sizes = np.array([2, 4], dtype=np.int32)

    np.random.seed(42)
    farms = create_farms_numba(cultivated_land, ids, farm_sizes)

    assert farms.shape == cultivated_land.shape

    # Non-cultivated cells should be -1
    assert np.all(farms[cultivated_land == 0] == -1)

    # Cultivated cells must be assigned to one of the provided IDs
    assigned_ids = np.unique(farms[cultivated_land == 1])
    assigned_ids = assigned_ids[assigned_ids != -1]
    assert set(assigned_ids.tolist()) == {1, 2}

    # Check each farmer received exactly their target number of cells
    assert int(np.count_nonzero(farms == 1)) == 2
    assert int(np.count_nonzero(farms == 2)) == 4


def test_create_farms_numba_single_farmer_single_cell() -> None:
    """Single farmer owning exactly one cultivated cell."""
    cultivated_land = np.zeros((2, 2), dtype=np.int32)
    cultivated_land[1, 1] = 1

    ids = np.array([99], dtype=np.int32)
    farm_sizes = np.array([1], dtype=np.int32)

    np.random.seed(123)
    farms = create_farms_numba(cultivated_land, ids, farm_sizes)

    # Only the cultivated cell should be assigned to the farmer ID
    assert farms[1, 1] == 99
    # All other cells remain non-farm land (-1)
    mask_other = np.ones_like(cultivated_land, dtype=bool)
    mask_other[1, 1] = False
    assert np.all(farms[mask_other] == -1)
