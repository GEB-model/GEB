"""This module contains algebraic solvers and utilities for the GEB model."""

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def tdma_solver(
    lower_diagonal_a: np.ndarray,
    main_diagonal_b: np.ndarray,
    upper_diagonal_c: np.ndarray,
    rhs_vector_d: np.ndarray,
) -> np.ndarray:
    """Solves a tridiagonal system Ax = d using the Thomas algorithm (TDMA).

    Args:
        lower_diagonal_a: lower diagonal (length n).
        main_diagonal_b: main diagonal (length n).
        upper_diagonal_c: upper diagonal (length n).
        rhs_vector_d: right hand side (length n).

    Returns:
        solution_vector_x: The solution of the tridiagonal system.
    """
    n = len(rhs_vector_d)
    c_prime = np.zeros(n, dtype=np.float32)
    d_prime = np.zeros(n, dtype=np.float32)
    x = np.zeros(n, dtype=np.float32)

    c_prime[0] = upper_diagonal_c[0] / main_diagonal_b[0]
    d_prime[0] = rhs_vector_d[0] / main_diagonal_b[0]

    for i in range(1, n):
        denominator = main_diagonal_b[i] - lower_diagonal_a[i] * c_prime[i - 1]

        # Avoid division by zero
        if abs(denominator) < 1e-10:
            denominator = 1e-10

        if i < n - 1:
            c_prime[i] = upper_diagonal_c[i] / denominator
        d_prime[i] = (
            rhs_vector_d[i] - lower_diagonal_a[i] * d_prime[i - 1]
        ) / denominator

    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x
