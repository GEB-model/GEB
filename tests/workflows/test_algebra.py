"""Unit tests for algebraic solvers in the GEB model."""

import numpy as np

from geb.workflows.algebra import tdma_solver


def test_tdma_solver_identity() -> None:
    """Test TDMA solver with an identity matrix."""
    n = 5
    a = np.zeros(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    d = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    x = tdma_solver(a, b, c, d)

    np.testing.assert_allclose(x, d, atol=1e-6)


def test_tdma_solver_simple_system() -> None:
    """Test TDMA solver with a known 3x3 system.

    Matrix:
    [ 2  -1   0 ] [ x1 ]   [ 1 ]
    [ -1  2  -1 ] [ x2 ] = [ 0 ]
    [ 0  -1   2 ] [ x3 ]   [ 1 ]

    Solution: x = [1, 1, 1]
    """
    n = 3
    a = np.array([0, -1, -1], dtype=np.float32)
    b = np.array([2, 2, 2], dtype=np.float32)
    c = np.array([-1, -1, 0], dtype=np.float32)
    d = np.array([1, 0, 1], dtype=np.float32)

    x = tdma_solver(a, b, c, d)

    expected_x = np.ones(3, dtype=np.float32)
    np.testing.assert_allclose(x, expected_x, atol=1e-6)


def test_tdma_solver_sum_of_rows() -> None:
    """Test TDMA solver using the sum of rows method.

    If x is a vector of ones, then Ax = d where d_i is the sum of row i.
    """
    n = 10
    # Create a diagonally dominant tridiagonal matrix to ensure stability
    # a_i, b_i, c_i
    a = np.random.uniform(-1, -0.5, n).astype(np.float32)
    c = np.random.uniform(-1, -0.5, n).astype(np.float32)
    # b_i should be > |a_i| + |c_i|
    b = (np.abs(a) + np.abs(c) + 1.0).astype(np.float32)

    # Correct bounds for TDMA (a[0] and c[n-1] are unused)
    a[0] = 0.0
    c[n - 1] = 0.0

    # Calculate d as the sum of rows (A * ones)
    d = np.zeros(n, dtype=np.float32)
    d[0] = b[0] + c[0]
    for i in range(1, n - 1):
        d[i] = a[i] + b[i] + c[i]
    d[n - 1] = a[n - 1] + b[n - 1]

    x = tdma_solver(a, b, c, d)

    expected_x = np.ones(n, dtype=np.float32)
    np.testing.assert_allclose(x, expected_x, atol=1e-5)
