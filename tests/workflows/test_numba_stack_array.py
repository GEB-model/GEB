"""Tests for numba_stack_array stack-allocation utilities."""

import time

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


@njit(inline="never")
def _benchmark_stack(n: int) -> np.float64:
    """Repeat stack allocation + fill + sum ``n`` times; return final sum.

    Args:
        n: Number of iterations to run the benchmark loop.

    Returns:
        The final sum after all iterations.
    """
    total = np.float64(0.0)
    for _ in range(n):
        arr = stack_empty(ARRAY_SIZE, np.float64)
        for i in range(ARRAY_SIZE):
            arr[i] = np.float64(i)
        for i in range(ARRAY_SIZE):
            total += arr[i]
    return total


@njit(inline="never")
def _benchmark_heap(n: int) -> np.float64:
    """Repeat heap allocation + fill + sum ``n`` times; return final sum.

    Args:
        n: Number of iterations to run the benchmark loop.

    Returns:
        The final sum after all iterations.
    """
    total = np.float64(0.0)
    for _ in range(n):
        arr = np.empty(ARRAY_SIZE, dtype=np.float64)
        for i in range(ARRAY_SIZE):
            arr[i] = np.float64(i)
        for i in range(ARRAY_SIZE):
            total += arr[i]
    return total


# Trigger JIT compilation once at module import time to avoid compile overhead
# in individual tests.
_fill_and_sum_stack_1d()
_fill_and_sum_heap_1d()
_benchmark_stack(N_WARMUP)
_benchmark_heap(N_WARMUP)

EXPECTED_SUM = float(sum(range(ARRAY_SIZE)))  # 0+1+2+3+4+5 = 15.0


def test_1d_sum() -> None:
    """Stack-allocated 1D array gives correct element sum."""
    assert _fill_and_sum_stack_1d() == pytest.approx(EXPECTED_SUM)


def test_matches_heap_result() -> None:
    """Stack and heap allocations produce identical numeric results."""
    assert _fill_and_sum_stack_1d() == pytest.approx(_fill_and_sum_heap_1d())


def test_stack_not_slower_than_heap() -> None:
    """Stack allocation should not be slower than heap for size-6 arrays.

    Both kernels run inside a single JIT-compiled loop to isolate
    allocation cost from Python call overhead.
    """
    t0 = time.perf_counter()
    _benchmark_stack(N_BENCHMARK)
    stack_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _benchmark_heap(N_BENCHMARK)
    heap_time_s = time.perf_counter() - t0

    print(
        f"\nStack time ({N_BENCHMARK} iterations): {stack_time_s * 1e3:.2f} ms"
        f"\nHeap  time ({N_BENCHMARK} iterations): {heap_time_s * 1e3:.2f} ms"
        f"\nSpeedup (heap/stack): {heap_time_s / stack_time_s:.2f}×"
    )

    assert stack_time_s <= heap_time_s, (
        f"Stack allocation unexpectedly slow: "
        f"stack={stack_time_s:.4f}s heap={heap_time_s:.4f}s"
    )
