"""This module contains algebraic solvers and utilities for the GEB model."""

import numpy as np
from numba import njit

from geb.geb_types import ArrayFloat


@njit(cache=True, inline="always")
def tdma_solver(
    lower_diagonal_a: ArrayFloat,
    main_diagonal_b: ArrayFloat,
    upper_diagonal_c: ArrayFloat,
    rhs_vector_d: ArrayFloat,
    solution_vector_x: ArrayFloat,
    c_prime: ArrayFloat,
    d_prime: ArrayFloat,
) -> None:
    """Solve a tridiagonal system Ax = d in-place using the Thomas algorithm.

    Notes:
        All arrays must have the same length and dtype.

    Args:
        lower_diagonal_a: Lower diagonal (length n).
        main_diagonal_b: Main diagonal (length n).
        upper_diagonal_c: Upper diagonal (length n).
        rhs_vector_d: Right hand side (length n).
        solution_vector_x: Output solution array (length n), modified in-place.
        c_prime: Work array (length n), modified in-place.
        d_prime: Work array (length n), modified in-place.
    """
    n = len(rhs_vector_d)

    c_prime[0] = upper_diagonal_c[0] / main_diagonal_b[0]
    d_prime[0] = rhs_vector_d[0] / main_diagonal_b[0]

    for i in range(1, n):
        denominator = main_diagonal_b[i] - lower_diagonal_a[i] * c_prime[i - 1]

        # Avoid division by zero
        eps = np.float32(1e-10)
        if abs(denominator) < eps:
            denominator = eps

        if i < n - 1:
            c_prime[i] = upper_diagonal_c[i] / denominator
        d_prime[i] = (
            rhs_vector_d[i] - lower_diagonal_a[i] * d_prime[i - 1]
        ) / denominator

    solution_vector_x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        solution_vector_x[i] = d_prime[i] - c_prime[i] * solution_vector_x[i + 1]
