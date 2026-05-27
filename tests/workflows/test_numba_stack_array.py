"""Tests for numba_stack_array stack-allocation utilities."""

import numpy as np
import pytest
from numba import njit

from geb.workflows.numba_stack_array import stack_empty

# Array size used throughout (small, representative of typical hot-loop usage)
ARRAY_SIZE = 6
N_WARMUP = 1
N_BENCHMARK = 100_000_000


# Numba kernels must be module-level so they are compiled once and reused.
@njit(inline="never")
def _fill_and_sum_stack_1d() -> np.float64:
    """Fill a 1D stack-allocated array and return the element sum.

    Returns:
        The sum of the array elements.
    """
    arr = stack_empty(ARRAY_SIZE, np.float64)
    for i in range(ARRAY_SIZE):
        arr[i] = np.float64(i)
    total = np.float64(0.0)
    for i in range(ARRAY_SIZE):
        total += arr[i]
    return total


@njit(inline="never")
def _fill_and_sum_heap_1d() -> np.float64:
    """Fill a 1D heap-allocated array and return the element sum.

    Returns:
        The sum of the array elements.
    """
    arr = np.empty(ARRAY_SIZE, dtype=np.float64)
    for i in range(ARRAY_SIZE):
        arr[i] = np.float64(i)
    total = np.float64(0.0)
    for i in range(ARRAY_SIZE):
        total += arr[i]
    return total


EXPECTED_SUM = float(sum(range(ARRAY_SIZE)))  # 0+1+2+3+4+5 = 15.0


def test_1d_sum() -> None:
    """Stack-allocated 1D array gives correct element sum."""
    assert _fill_and_sum_stack_1d() == pytest.approx(EXPECTED_SUM)


def test_matches_heap_result() -> None:
    """Stack and heap allocations produce identical numeric results."""
    assert _fill_and_sum_stack_1d() == pytest.approx(_fill_and_sum_heap_1d())
