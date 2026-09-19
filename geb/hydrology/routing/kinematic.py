"""Kinematic wave routing equation solver for river networks."""

import numpy as np
from numba import njit

MAX_ITERS_KINEMATIC: int = 10


@njit(cache=True)
def update_node_kinematic(
    inflow_m3_s: np.float32,
    previous_discharge_m3_s: np.float32,
    sideflow_m3_s: np.float32,
    evaporation_m3_s: np.float32,
    river_storage_alpha: np.float32,
    river_storage_beta: np.float32,
    timestep_s: np.float32,
    river_length_m: np.float32,
    epsilon: np.float32 = np.float32(0.00001),
) -> tuple[np.float32, np.float32]:
    """Calculate the new discharge and actual evaporation for a single cell using kinematic wave routing.

    Solves the non-linear kinematic wave equation using the Newton-Raphson method.

    Args:
        inflow_m3_s: Inflow discharge from upstream cells (m³/s).
        previous_discharge_m3_s: Discharge from the previous time step (m³/s).
        sideflow_m3_s: Lateral inflow rate into the river cell (m³/s).
        evaporation_m3_s: Potential evaporation rate from the river surface (m³/s).
        river_storage_alpha: Kinematic wave alpha parameter (s^beta / m^(3*beta - 1)).
        river_storage_beta: Kinematic wave beta parameter (-).
        timestep_s: Routing time step duration (seconds).
        river_length_m: Length of the river channel in the cell (meters).
        epsilon: Convergence threshold for Newton-Raphson solver.

    Returns:
        A tuple containing:
            - new_discharge_m3_s: Updated discharge at current timestep (m³/s).
            - actual_evaporation_m3_s: Actual evaporation rate constrained by available flow (m³/s).
    """
    inv_river_length: np.float32 = np.float32(1.0) / river_length_m
    evaporation_m3_s_per_m: np.float32 = evaporation_m3_s * inv_river_length
    lateral_inflow_m3_s_per_m: np.float32 = sideflow_m3_s * inv_river_length

    # Limit evaporation to available flow
    evaporation_m3_s_per_m = min(
        evaporation_m3_s_per_m,
        (inflow_m3_s + previous_discharge_m3_s) * np.float32(0.5)
        + max(lateral_inflow_m3_s_per_m, np.float32(0.0)),
    )

    lateral_inflow_m3_s_per_m -= evaporation_m3_s_per_m
    actual_evaporation_m3_s: np.float32 = evaporation_m3_s_per_m * river_length_m

    # Return tiny positive flow if reach is dry
    if (inflow_m3_s + previous_discharge_m3_s + lateral_inflow_m3_s_per_m) < 1e-30:
        return np.float32(1e-30), actual_evaporation_m3_s

    inflow_m3_s = max(inflow_m3_s, np.float32(1e-30))
    previous_discharge_m3_s = max(previous_discharge_m3_s, np.float32(1e-30))

    # Initial guess for discharge using a linearized approximation
    beta_minus_1: np.float32 = river_storage_beta - np.float32(1.0)
    derivative_storage: np.float32 = (
        river_storage_alpha
        * river_storage_beta
        * (
            max(
                (previous_discharge_m3_s + inflow_m3_s) * np.float32(0.5),
                np.float32(1e-30),
            )
            ** beta_minus_1
        )
    )
    dt_over_dx: np.float32 = timestep_s * inv_river_length
    prev_storage: np.float32 = previous_discharge_m3_s**river_storage_beta
    rhs_constant: np.float32 = (
        dt_over_dx * inflow_m3_s
        + river_storage_alpha * prev_storage
        + timestep_s * lateral_inflow_m3_s_per_m
    )

    new_discharge: np.float32 = (
        dt_over_dx * inflow_m3_s
        + previous_discharge_m3_s * derivative_storage
        + timestep_s * lateral_inflow_m3_s_per_m
    ) / (dt_over_dx + derivative_storage)
    new_discharge = max(new_discharge, np.float32(1e-30))

    # Newton-Raphson iteration
    alpha_times_beta: np.float32 = river_storage_alpha * river_storage_beta
    q_current: np.float32 = max(new_discharge, np.float32(1e-30))
    q_pow_beta: np.float32 = q_current**river_storage_beta
    f_q: np.float32 = (
        dt_over_dx * new_discharge + river_storage_alpha * q_pow_beta - rhs_constant
    )

    iteration: int = 0
    while np.abs(f_q) > epsilon and iteration < MAX_ITERS_KINEMATIC:
        df_q: np.float32 = dt_over_dx + alpha_times_beta * (q_pow_beta / q_current)
        new_discharge -= f_q / df_q
        new_discharge = max(new_discharge, np.float32(1e-30))

        q_current = new_discharge
        q_pow_beta = q_current**river_storage_beta
        f_q = (
            dt_over_dx * new_discharge + river_storage_alpha * q_pow_beta - rhs_constant
        )
        iteration += 1

    return new_discharge, actual_evaporation_m3_s
