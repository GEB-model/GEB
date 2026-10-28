"""Local inertial momentum solver for 1D shallow water channel routing.

Solves the 1D de Saint-Venant momentum equation simplified by neglecting the advective
acceleration term (Bates et al. 2010).
"""

import numpy as np
from numba import njit

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    TwoDArrayFloat32,
)
from geb.hydrology.routing.geometry import (
    GEOM_IN_BANKFULL_AREA,
    GEOM_IN_BANKFULL_DEPTH,
    GEOM_IN_BANKFULL_PERIMETER,
    GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE,
    GEOM_IN_MANNING_N_SQUARED,
    GEOM_IN_RIVER_WIDTH,
    GEOM_IN_SHAPE_EXPONENT,
    GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH,
    compute_cross_section_from_depth,
)

__all__ = [
    "evaluate_discharge_pow_0_6",
    "evaluate_hydraulic_radius_pow_4_3",
    "solve_inertial_momentum",
    "solve_steady_backwater_network",
]

R_LUT_SIZE: int = 1024
R_LUT_MAX: np.float32 = np.float32(30.0)
_U_MAX: np.float32 = np.float32(np.sqrt(30.0))
_INV_DU: np.float32 = np.float32((R_LUT_SIZE - 1) / _U_MAX)

# Precomputed global lookup table for Rh^(4/3) using u = sqrt(Rh).
# Table size is 1024 float32 entries = exactly 4 KB, fitting entirely in CPU L1 data cache.
_U_GRID: ArrayFloat32 = np.linspace(0.0, _U_MAX, R_LUT_SIZE, dtype=np.float32)
R_POW_4_3_TABLE: ArrayFloat32 = (_U_GRID ** np.float32(8.0 / 3.0)).astype(np.float32)

# Precomputed global lookup table for Q^(0.6) using fourth-root scaling u = sqrt(sqrt(Q)).
# Table size is 1024 float32 entries = exactly 4 KB, fitting entirely in CPU L1 data cache.
# Fourth-root scaling provides < 0.38% relative error across 0.001 to 100,000 m³/s.
Q_LUT_SIZE: int = 1024
Q_LUT_MAX: np.float32 = np.float32(100_000.0)
_U_Q_MAX: np.float32 = np.float32(100_000.0**0.25)
_INV_DU_Q: np.float32 = np.float32((Q_LUT_SIZE - 1) / _U_Q_MAX)
_U_GRID_Q: ArrayFloat32 = np.linspace(0.0, _U_Q_MAX, Q_LUT_SIZE, dtype=np.float32)
Q_POW_0_6_TABLE: ArrayFloat32 = (_U_GRID_Q ** np.float32(2.4)).astype(np.float32)


@njit(inline="always")
def evaluate_discharge_pow_0_6(discharge: np.float32) -> np.float32:
    """Evaluates Q^(0.6) via fourth-root scaled lookup table with linear interpolation.

    Uses substitution u = sqrt(sqrt(Q)) and Q^(0.6) = u^(2.4). The table spans [0, 100000.0] m³/s
    with 1024 entries (4 KB), fitting entirely inside the CPU L1 data cache.

    Args:
        discharge: Flow rate or discharge magnitude (m³/s).

    Returns:
        Approximation of discharge^(0.6) ((m³/s)^0.6).
    """
    if discharge <= np.float32(0.0):
        return np.float32(0.0)
    u: np.float32 = np.sqrt(np.sqrt(discharge))
    scaled: np.float32 = min(u * _INV_DU_Q, np.float32(Q_LUT_SIZE - 1))
    idx: int = min(int(scaled), Q_LUT_SIZE - 2)
    frac: np.float32 = scaled - np.float32(idx)
    return Q_POW_0_6_TABLE[idx] + frac * (
        Q_POW_0_6_TABLE[idx + 1] - Q_POW_0_6_TABLE[idx]
    )


@njit(inline="always")
def evaluate_hydraulic_radius_pow_4_3(hydraulic_radius: np.float32) -> np.float32:
    """Evaluates Rh^(4/3) via square-root scaled lookup table with linear interpolation.

    Uses substitution u = sqrt(Rh) and Rh^(4/3) = u^(8/3). The table spans [0, 30.0] meters
    with 1024 entries (4 KB), fitting entirely inside the CPU L1 data cache.

    Args:
        hydraulic_radius: Hydraulic radius of the river reach (meters).

    Returns:
        Approximation of hydraulic_radius^(4/3) (m^(4/3)).
    """
    u: np.float32 = np.sqrt(hydraulic_radius)
    scaled: np.float32 = min(u * _INV_DU, np.float32(R_LUT_SIZE - 1))
    idx: int = min(int(scaled), R_LUT_SIZE - 2)
    frac: np.float32 = scaled - np.float32(idx)
    return R_POW_4_3_TABLE[idx] + frac * (
        R_POW_4_3_TABLE[idx + 1] - R_POW_4_3_TABLE[idx]
    )


@njit(inline="always")
def solve_inertial_momentum(
    reach_idx: int,
    effective_depth: np.float32,
    water_slope: np.float32,
    curr_discharge: np.float32,
    min_wet_depth_m: np.float32,
    geom_inbank: TwoDArrayFloat32,
    g_dt_substep: np.float32,
    sqrt_gravity: np.float32,
    can_reverse: bool,
    river_storage_m3_inertial: ArrayFloat64,
    inv_dt_substep: np.float32,
) -> np.float32:
    """Evaluates the 1D local inertial momentum equation for a single reach interface.

    Solves the momentum equation neglecting advective acceleration:

        dQ/dt + g * A * dz/dx + (g * n^2 * |Q| * Q) / (R^(4/3) * A) = 0

    Discretized semi-implicitly in friction (Bates et al. 2010):

        Q^{t+dt} = (Q^t - g * dt * A * dz/dx) / (1 + g * dt * n^2 * |Q^t| / (A * R^(4/3)))

    Args:
        reach_idx: Reach index within local inertial domain.
        effective_depth: Flow depth at the interface between reaches (meters).
        water_slope: Longitudinal water surface slope dz/dx = (z_ds - z_us) / dx (dimensionless).
        curr_discharge: Instantaneous discharge at start of substep Q^t (m³/s).
        min_wet_depth_m: Minimum depth threshold below which reach is treated as dry (meters).
        geom_inbank: Compact 2D in-bank reach geometry table (meters, dimensionless).
        g_dt_substep: Gravitational acceleration times substep dt: g * dt (m/s).
        sqrt_gravity: Square root of gravitational acceleration sqrt(g) (m^(1/2)/s).
        can_reverse: Whether adverse-slope reverse flow is permitted.
        river_storage_m3_inertial: Upstream reach storage volume array (m³).
        inv_dt_substep: Reciprocal of substep timestep duration 1 / dt (1/s).

    Returns:
        Discharge Q^{t+dt} at end of substep (m³/s).
    """
    if effective_depth < min_wet_depth_m:
        return np.float32(0.0)

    bankfull_depth_m: np.float32 = geom_inbank[reach_idx, GEOM_IN_BANKFULL_DEPTH]
    inv_shape_exponent_plus_one: np.float32 = geom_inbank[
        reach_idx, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE
    ]
    shape_exponent: np.float32 = geom_inbank[reach_idx, GEOM_IN_SHAPE_EXPONENT]
    bankfull_area: np.float32 = geom_inbank[reach_idx, GEOM_IN_BANKFULL_AREA]
    bankfull_width_m: np.float32 = geom_inbank[reach_idx, GEOM_IN_RIVER_WIDTH]
    bankfull_perimeter_m: np.float32 = geom_inbank[
        reach_idx, GEOM_IN_BANKFULL_PERIMETER
    ]

    (
        cross_sectional_area,
        interface_top_width,
        interface_p_wetted,
    ) = compute_cross_section_from_depth(
        effective_depth_m=effective_depth,
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        inverse_shape_exponent_plus_one=inv_shape_exponent_plus_one,
        bankfull_area_m2=bankfull_area,
        bankfull_perimeter_m=bankfull_perimeter_m,
        width_over_sqrt_bankfull_depth=geom_inbank[
            reach_idx, GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH
        ],
    )

    # Semi-implicit Manning friction term: A * R^(4/3) = A * (A / P)^(4/3)
    hydraulic_radius: np.float32 = cross_sectional_area / max(
        interface_p_wetted, np.float32(1e-6)
    )
    rh_pow_4_3: np.float32 = evaluate_hydraulic_radius_pow_4_3(hydraulic_radius)
    friction_denom_term: np.float32 = max(
        cross_sectional_area * rh_pow_4_3,
        np.float32(1e-12),
    )
    gravity_dt_manning_n_sq: np.float32 = (
        g_dt_substep * geom_inbank[reach_idx, GEOM_IN_MANNING_N_SQUARED]
    )
    friction_factor: np.float32 = np.float32(1.0) + (
        gravity_dt_manning_n_sq * abs(curr_discharge) / friction_denom_term
    )
    computed_discharge: np.float32 = (
        curr_discharge - g_dt_substep * cross_sectional_area * water_slope
    ) / friction_factor

    # Critical flow limiter (Froude <= 1) prevents supercritical runaway over steep drops
    hydraulic_depth: np.float32 = cross_sectional_area / max(
        interface_top_width, np.float32(1e-3)
    )
    critical_discharge: np.float32 = (
        sqrt_gravity * cross_sectional_area * np.sqrt(hydraulic_depth)
    )

    # downward flow
    if computed_discharge >= np.float32(0.0):
        # Cap outflow to 95% of available upstream volume to prevent over-draining in one substep
        available_storage_m3: float = max(river_storage_m3_inertial[reach_idx], 0.0)
        max_drain_rate: np.float32 = (
            np.float32(0.95) * np.float32(available_storage_m3) * inv_dt_substep
        )
        return min(computed_discharge, critical_discharge, max_drain_rate)

    else:  # upward flow
        # if flow cannot reverse, flow is 0 when there is upward flow
        if not can_reverse:
            return np.float32(0.0)

        # otherwise limit by critical discharge, which is now negative (as flow is upward)
        return max(computed_discharge, -critical_discharge)


@njit(fastmath=True)
def solve_steady_backwater_network(
    compressed_seq: ArrayInt32,
    compressed_ds: ArrayInt32,
    discharge: ArrayFloat32,
    width: ArrayFloat32,
    bed_elevation: ArrayFloat32,
    reach_length: ArrayFloat32,
    manning_n: ArrayFloat32,
    shape_exponent: np.float32,
    min_depth_m: np.float32,
    is_kinematic: ArrayBool,
    min_kinematic_slope_m_per_m: np.float32,
) -> ArrayFloat32:
    """Calculates bankfull channel depths using the 1D Standard Step backwater method.

    Traverses the river network from downstream boundary controls (ocean or terminal sinks)
    upstream to headwaters in reverse topological sequence, integrating the steady local
    inertial momentum equation to determine water surface stage and channel depth. For reaches
    where downstream water levels submerge the channel bed, backwater profiles (M1 curves)
    are calculated by integrating friction slope losses upstream. For steep reaches and kinematic
    wave reaches, flow depth reverts to Manning normal depth:
        h_norm = (r + 1) * (n * Q / (W * sqrt(S)))^0.6

    Args:
        compressed_seq: 1D array of active cell indices ordered from downstream (pits)
            to upstream (headwaters).
        compressed_ds: 1D array of downstream active cell indices (-1 or self-reference if pit).
        discharge: Bankfull discharge per cell (m³/s).
        width: Channel top width per cell (meters).
        bed_elevation: Riverbed elevation above datum per cell (meters).
        reach_length: Longitudinal reach length per cell (meters).
        manning_n: Manning roughness coefficient per cell (s / m^(1/3)). Must be >= 0.01.
        shape_exponent: Cross-sectional power-law shape exponent (dimensionless, 0.5 for parabolic).
        min_depth_m: Minimum allowable bankfull depth floor (meters).
        is_kinematic: Boolean mask identifying kinematic wave reaches (True)
            that do not transmit or receive backwater effects.
        min_kinematic_slope_m_per_m: Minimum bed slope floor applied to kinematic wave reaches
            and boundary sinks to prevent division by zero (m/m).

    Returns:
        1D array of bankfull centerline channel depths per cell (meters).

    Raises:
        ValueError: If discharge contains negative values.
        ValueError: If width or reach_length are not strictly positive.
        ValueError: If manning_n has values smaller than 0.01.
        ValueError: If shape_exponent is negative.
        ValueError: If min_depth_m or min_kinematic_slope_m_per_m are not strictly positive.
    """
    if not np.all(discharge >= np.float32(0.0)):
        raise ValueError("discharge must be non-negative.")
    if not np.all(width > np.float32(0.0)):
        raise ValueError("width must be strictly positive.")
    if not np.all(reach_length > np.float32(0.0)):
        raise ValueError("reach_length must be strictly positive.")
    if not np.all(manning_n >= np.float32(0.01)):
        raise ValueError("manning_n must be at least 0.01 s / m^(1/3).")
    if shape_exponent < np.float32(0.0):
        raise ValueError("shape_exponent must be non-negative.")
    if min_depth_m <= np.float32(0.0):
        raise ValueError("min_depth_m must be strictly positive.")
    if min_kinematic_slope_m_per_m <= np.float32(0.0):
        raise ValueError("min_kinematic_slope_m_per_m must be strictly positive.")

    total_cells: int = compressed_ds.shape[0]
    n_seq_cells: int = compressed_seq.shape[0]
    bankfull_depth_m: ArrayFloat32 = np.empty(total_cells, dtype=np.float32)
    water_stage_m: ArrayFloat32 = np.empty(total_cells, dtype=np.float32)
    shape_factor: np.float32 = shape_exponent + np.float32(1.0)
    ten_thirds: np.float32 = np.float32(10.0 / 3.0)

    for seq_idx in range(n_seq_cells):
        reach_idx: int = int(compressed_seq[seq_idx])
        discharge_m3_s: np.float32 = discharge[reach_idx]
        bed_elevation_m: np.float32 = bed_elevation[reach_idx]

        if discharge_m3_s == np.float32(0.0):
            bankfull_depth_m[reach_idx] = min_depth_m
            water_stage_m[reach_idx] = bed_elevation_m + min_depth_m
            continue

        river_width_m: np.float32 = width[reach_idx]
        reach_length_m: np.float32 = reach_length[reach_idx]
        manning_roughness: np.float32 = manning_n[reach_idx]
        downstream_reach_idx: int = int(compressed_ds[reach_idx])
        has_downstream_reach: bool = (
            downstream_reach_idx != -1 and downstream_reach_idx != reach_idx
        )

        # Evaluate bed slope for normal depth
        bed_slope_m_per_m: np.float32 = min_kinematic_slope_m_per_m
        if has_downstream_reach:
            elevation_drop_m: np.float32 = (
                bed_elevation_m - bed_elevation[downstream_reach_idx]
            )
            bed_slope_m_per_m = max(
                elevation_drop_m / reach_length_m, min_kinematic_slope_m_per_m
            )

        # Normal depth from Manning uniform flow:
        # Q = (1/n) * (W * h_mean) * h_mean^(2/3) * sqrt(S0)
        # h_mean_norm = (n * Q / (W * sqrt(S0)))^0.6
        # Centerline bankfull depth: h_norm = shape_factor * h_mean_norm
        mean_normal_depth_m: np.float32 = (
            (manning_roughness * discharge_m3_s)
            / (river_width_m * np.sqrt(bed_slope_m_per_m))
        ) ** np.float32(0.6)
        normal_depth_m: np.float32 = max(
            shape_factor * mean_normal_depth_m, min_depth_m
        )

        final_reach_depth_m: np.float32 = normal_depth_m
        if not is_kinematic[reach_idx]:
            if not has_downstream_reach:
                # Terminal boundary for local inertial routing (ocean or inland sink)
                ocean_depth_m: np.float32 = max(-bed_elevation_m, np.float32(0.0))
                final_reach_depth_m = max(normal_depth_m, ocean_depth_m)
            else:
                # Use Heun's method to estimate backwater profile from downstream stage and friction slope
                downstream_stage_m: np.float32 = water_stage_m[downstream_reach_idx]
                downstream_mean_depth_m: np.float32 = max(
                    bankfull_depth_m[downstream_reach_idx] / shape_factor,
                    mean_normal_depth_m,
                )

                friction_factor: np.float32 = (
                    (manning_roughness * discharge_m3_s) / river_width_m
                ) ** np.float32(2.0)

                downstream_friction_slope: np.float32 = friction_factor / (
                    downstream_mean_depth_m**ten_thirds
                )

                # Predictor step: estimate upstream stage from downstream friction slope
                predicted_stage_m: np.float32 = max(
                    downstream_stage_m + downstream_friction_slope * reach_length_m,
                    bed_elevation_m + normal_depth_m,
                )
                predicted_mean_depth_m: np.float32 = (
                    predicted_stage_m - bed_elevation_m
                ) / shape_factor

                # Corrector step (see Heun's method)
                predicted_friction_slope: np.float32 = friction_factor / (
                    predicted_mean_depth_m**ten_thirds
                )
                average_friction_slope: np.float32 = np.float32(0.5) * (
                    downstream_friction_slope + predicted_friction_slope
                )
                final_stage_m: np.float32 = max(
                    downstream_stage_m + average_friction_slope * reach_length_m,
                    bed_elevation_m + normal_depth_m,
                )
                final_reach_depth_m = max(
                    normal_depth_m, final_stage_m - bed_elevation_m
                )

        bankfull_depth_m[reach_idx] = final_reach_depth_m
        water_stage_m[reach_idx] = bed_elevation_m + final_reach_depth_m

    return bankfull_depth_m
