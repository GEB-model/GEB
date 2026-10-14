"""Local inertial momentum solver for 1D shallow water channel routing.

Solves the 1D de Saint-Venant momentum equation simplified by neglecting the advective
acceleration term, providing an unconditionally stable semi-implicit formulation
(Bates et al. 2010; de Almeida et al. 2012).
"""

import numpy as np
from numba import njit

from geb.geb_types import ArrayFloat64, TwoDArrayFloat32
from geb.hydrology.routing.geometry import (
    GEOM_IN_BANKFULL_DEPTH,
    GEOM_IN_BANKFULL_VOLUME,
    GEOM_IN_INVERSE_LENGTH,
    GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE,
    GEOM_IN_MANNING_N_SQUARED,
    GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH,
    GEOM_OV_BANKFULL_PERIMETER,
    GEOM_OV_RIVER_WIDTH,
    compute_cross_section_from_depth,
)

__all__ = [
    "_solve_inertial_momentum",
    "solve_inertial_momentum",
]


@njit(inline="always")
def solve_inertial_momentum(
    reach_idx: int,
    effective_depth: np.float32,
    water_slope: np.float32,
    curr_discharge: np.float32,
    min_wet_depth_m: np.float32,
    geom_inbank: TwoDArrayFloat32,
    geom_overbank: TwoDArrayFloat32,
    g_dt_substep: np.float32,
    sqrt_gravity: np.float32,
    can_reverse: bool,
    river_storage_m3_inertial: ArrayFloat64,
    inv_dt_substep: np.float32,
) -> np.float32:
    """Evaluates the 1D local inertial momentum equation for a single reach interface.

    This function represents the mathematical core of the local inertial hydrodynamics
    formulation (Bates et al. 2010; de Almeida et al. 2012). It solves the 1D de Saint-Venant
    momentum equation simplified by neglecting the advective acceleration term d(Q^2/A)/dx,
    which is negligible in subcritical river and floodplain dynamics:

        dQ/dt + g * A * dz/dx + (g * n^2 * |Q| * Q) / (R^(4/3) * A) = 0

    Discretizing from time t to t + dt with a semi-implicit friction formulation yields
    the explicit, unconditionally friction-stable update:

        Q^{t+dt} = (Q^t - g * dt * A * dz/dx) / (1 + g * dt * n^2 * |Q^t| / (A * R^(4/3)))

    Where:
        1. Gravity acceleration predictor:
           Q* = Q^t - g * dt * A * dz/dx
           Accelerates (or decelerates) flow along the water surface slope dz/dx. When downstream
           water elevation exceeds upstream (backwater effects), dz/dx > 0 and flow decelerates
           or reverses.
        2. Semi-implicit Manning friction corrector:
           Dampens the flow by dividing Q* by (1 + friction_damping). Using |Q^t| in the denominator
           prevents numerical friction oscillation and ensures unconditional stability for any Manning n.
        3. Physical limiters:
           - Critical flow limiter (Fr <= 1): Caps discharge at critical depth celerity across steep drops.
           - Volume conservation limiter: Caps outgoing discharge at 95% of available reach volume per dt.

    Args:
        reach_idx: Reach index within local inertial domain.
        effective_depth: Flow depth at the interface between reaches (meters).
        water_slope: Longitudinal water surface slope dz/dx = (z_ds - z_us) / dx (dimensionless).
        curr_discharge: Instantaneous discharge at start of substep Q^t (m³/s).
        min_wet_depth_m: Minimum depth threshold below which reach is treated as dry (meters).
        geom_inbank: Compact 2D in-bank reach geometry table (meters, dimensionless).
        geom_overbank: Secondary 2D overbank floodplain geometry table (meters, dimensionless).
        g_dt_substep: Precalculated gravitational acceleration times substep dt: g * dt (m/s).
        sqrt_gravity: Square root of gravitational acceleration sqrt(g) (m^(1/2)/s).
        can_reverse: Boolean flag indicating whether reverse/backwater flow is permitted.
        river_storage_m3_inertial: Upstream reach storage volume array (m³).
        inv_dt_substep: Reciprocal of substep timestep duration 1 / dt (1/s).

    Returns:
        Computed channel discharge Q^{t+dt} at end of substep (m³/s).
    """
    # Dry-bed check: no momentum transfer across dry channel interfaces
    if effective_depth < min_wet_depth_m:
        return np.float32(0.0)

    # Reconstruct cross-sectional parameters from precomputed geometry tables
    bankfull_depth_m: np.float32 = geom_inbank[reach_idx, GEOM_IN_BANKFULL_DEPTH]
    inv_shape_exponent_plus_one: np.float32 = geom_inbank[
        reach_idx, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE
    ]
    shape_exponent: np.float32 = (
        np.float32(0.5)
        if inv_shape_exponent_plus_one == np.float32(1.0 / 1.5)
        else (np.float32(1.0) / inv_shape_exponent_plus_one) - np.float32(1.0)
    )
    bankfull_area: np.float32 = (
        geom_inbank[reach_idx, GEOM_IN_BANKFULL_VOLUME]
        * geom_inbank[reach_idx, GEOM_IN_INVERSE_LENGTH]
    )
    (
        cross_sectional_area,
        interface_top_width,
        interface_p_wetted,
    ) = compute_cross_section_from_depth(
        effective_depth_m=effective_depth,
        bankfull_width_m=geom_overbank[reach_idx, GEOM_OV_RIVER_WIDTH],
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        inverse_shape_exponent_plus_one=inv_shape_exponent_plus_one,
        bankfull_area_m2=bankfull_area,
        bankfull_perimeter_m=geom_overbank[reach_idx, GEOM_OV_BANKFULL_PERIMETER],
        width_over_sqrt_bankfull_depth=geom_inbank[
            reach_idx, GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH
        ],
    )

    # Evaluate Manning friction term:
    # Hydraulic radius R = A / P, giving denominator term A * R^(4/3) = A * (A / P)^(4/3)
    hydraulic_radius_term: np.float32 = (
        cross_sectional_area / max(interface_p_wetted, np.float32(1e-6))
    ) ** np.float32(4.0 / 3.0)
    friction_denom_term: np.float32 = max(
        cross_sectional_area * hydraulic_radius_term, np.float32(1e-12)
    )

    incoming_discharge: np.float32 = curr_discharge
    gravity_dt_manning_n_sq: np.float32 = (
        g_dt_substep * geom_inbank[reach_idx, GEOM_IN_MANNING_N_SQUARED]
    )

    # 1. PREDICTOR STEP (Gravity acceleration):
    # Accelerates flow along water surface slope dz/dx: Q* = Q^t - g * dt * A * (dz/dx).
    # Downhill slopes (dz/dx < 0) accelerate forward flow; adverse slopes (dz/dx > 0) decelerate flow.
    intermediate_discharge: np.float32 = (
        incoming_discharge - g_dt_substep * cross_sectional_area * water_slope
    )

    # 2. CORRECTOR STEP (Semi-implicit Manning bed friction):
    # Q^{t+dt} = Q* / (1 + g * dt * n^2 * |Q^t| / (A * R^(4/3))).
    # Evaluating friction semi-implicitly provides unconditional stability regardless of Manning n.
    effective_friction_q: np.float32 = abs(incoming_discharge)
    friction_denom: np.float32 = np.float32(1.0) + (
        gravity_dt_manning_n_sq * effective_friction_q / friction_denom_term
    )
    computed_discharge: np.float32 = intermediate_discharge / friction_denom

    # 3. PHYSICAL LIMITER: Critical flow ceiling (Froude number Fr <= 1):
    # Q_crit = A * sqrt(g * D_hydraulic). Prevents unphysical supercritical runaway over sharp drops.
    hydraulic_depth: np.float32 = cross_sectional_area / max(
        interface_top_width, np.float32(1e-3)
    )
    critical_discharge: np.float32 = (
        sqrt_gravity * cross_sectional_area * np.sqrt(hydraulic_depth)
    )

    # 4. PHYSICAL LIMITER: Upstream storage availability (Courant mass ceiling):
    # Ensures a reach cannot discharge more than 95% of its available water volume in a single substep.
    if computed_discharge >= np.float32(0.0):
        available_storage_m3: float = river_storage_m3_inertial[reach_idx]
        if available_storage_m3 > 0.0:
            max_drain_rate: np.float32 = (
                np.float32(0.95) * np.float32(available_storage_m3) * inv_dt_substep
            )
            return min(computed_discharge, critical_discharge, max_drain_rate)
        return np.float32(0.0)

    # Reverse flow (backwater wave propagation) is permitted only for internal reaches
    if not can_reverse:
        return np.float32(0.0)

    return -min(-computed_discharge, critical_discharge)


_solve_inertial_momentum = solve_inertial_momentum
