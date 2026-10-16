"""Local inertial momentum solver for 1D shallow water channel routing.

Solves the 1D de Saint-Venant momentum equation simplified by neglecting the advective
acceleration term (Bates et al. 2010).
"""

import numpy as np
from numba import njit

from geb.geb_types import ArrayFloat64, TwoDArrayFloat32
from geb.hydrology.routing.geometry import (
    GEOM_IN_BANKFULL_DEPTH,
    GEOM_IN_BANKFULL_VOLUME,
    GEOM_IN_BED_ELEVATION,
    GEOM_IN_INVERSE_LENGTH,
    GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE,
    GEOM_IN_MANNING_N_SQUARED,
    GEOM_IN_WEIR_HEIGHT,
    GEOM_IN_WEIR_WIDTH,
    GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH,
    GEOM_OV_BANKFULL_PERIMETER,
    GEOM_OV_RIVER_WIDTH,
    compute_cross_section_from_depth,
)

__all__ = [
    "solve_inertial_momentum",
    "rectangular_weir_discharge",
]


@njit(inline="always")
def rectangular_weir_discharge(
    upstream_stage_m: np.float32,
    downstream_stage_m: np.float32,
    crest_elevation_m: np.float32,
    width_m: np.float32,
) -> np.float32:
    """Calculate signed rectangular weir flow, including downstream submergence.

    Use Aigner's Table 2: Q = (2/3) * 0.577 * sqrt(2g) * b * H**1.5.
    Treat each structure as an idealized broad-crested overflow with negligible
    approach velocity head. Crest shape and width are prescribed by the model.
    The broad-crested reduction follows USACE CHETN-IV-94 (2013), Eq. 4,
    normalized for its rounded coefficients to give exactly zero at equal heads.

    References:
        https://rcswww.urz.tu-dresden.de/~daigner/pdf/ueberf.pdf
        https://usace.contentdm.oclc.org/digital/api/collection/p266001coll1/id/4443/download

    Args:
        upstream_stage_m: Water elevation on the upstream side (meters).
        downstream_stage_m: Water elevation on the downstream side (meters).
        crest_elevation_m: Crest elevation in the same vertical datum (meters).
        width_m: Effective rectangular crest width (meters).

    Returns:
        Signed flow (m³/s), positive downstream; zero below the crest or when
        either the head or water-level difference is at most 0.1 mm.
        Inputs are finite elevations and a positive width from validated geometry.
    """
    difference_m: np.float32 = upstream_stage_m - downstream_stage_m
    head_m: np.float32 = max(upstream_stage_m, downstream_stage_m) - crest_elevation_m
    if head_m <= np.float32(1e-4) or abs(difference_m) <= np.float32(1e-4):
        return np.float32(0.0)
    discharge: np.float32 = (
        np.float32(2.0 / 3.0 * 0.577)
        * np.sqrt(np.float32(2.0 * 9.80665))
        * width_m
        * head_m
        * np.sqrt(head_m)
    )
    # Tailwater affects flow once its head exceeds 67% of the higher head.
    # With d = (higher level - lower level) / (0.33 * head), the reduction
    # 1 - (1 - d)**3 equals d * (3 - d * (3 - d)). This form avoids losing
    # precision by subtracting nearly equal numbers at deeply submerged crests.
    relative_drop: np.float32 = abs(difference_m) / (np.float32(0.33) * head_m)
    if relative_drop < np.float32(1.0):
        discharge *= relative_drop * (
            np.float32(3.0) - relative_drop * (np.float32(3.0) - relative_drop)
        )
    return discharge if difference_m > 0 else -discharge


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
    upstream_stage_m: np.float32 = np.float32(0.0),
    downstream_stage_m: np.float32 = np.float32(0.0),
) -> np.float32:
    """Evaluates the 1D local inertial momentum equation for a single reach interface.

    Solves the momentum equation neglecting advective acceleration:

        dQ/dt + g * A * dz/dx + (g * n^2 * |Q| * Q) / (R^(4/3) * A) = 0

    At a weir, replace momentum flow with the rectangular overflow rating.
    River storage and the ordinary channel geometry are unchanged.

    Discretized semi-implicitly in friction (Bates et al. 2010):

        Q^{t+dt} = (Q^t - g * dt * A * dz/dx) / (1 + g * dt * n^2 * |Q^t| / (A * R^(4/3)))

    Args:
        reach_idx: Reach index within local inertial domain.
        effective_depth: Flow depth at the interface between reaches (meters).
        water_slope: Longitudinal water surface slope dz/dx = (z_ds - z_us) / dx (dimensionless).
        curr_discharge: Instantaneous discharge at start of substep Q^t (m³/s).
        min_wet_depth_m: Minimum depth threshold below which reach is treated as dry (meters).
        geom_inbank: Compact 2D in-bank reach geometry table (meters, dimensionless).
        geom_overbank: Secondary 2D overbank floodplain geometry table (meters, dimensionless).
        g_dt_substep: Gravitational acceleration times substep dt: g * dt (m/s).
        sqrt_gravity: Square root of gravitational acceleration sqrt(g) (m^(1/2)/s).
        can_reverse: Whether adverse-slope reverse flow is permitted.
        river_storage_m3_inertial: Upstream reach storage volume array (m³).
        inv_dt_substep: Reciprocal of substep timestep duration 1 / dt (1/s).
        upstream_stage_m: Upstream water elevation (meters); required at weirs.
        downstream_stage_m: Downstream water elevation (meters); required at weirs.

    Returns:
        Discharge Q^{t+dt} at end of substep (m³/s).
    """
    weir_height_m: np.float32 = geom_inbank[reach_idx, GEOM_IN_WEIR_HEIGHT]
    if weir_height_m > np.float32(0.0):
        crest_m: np.float32 = (
            geom_inbank[reach_idx, GEOM_IN_BED_ELEVATION] + weir_height_m
        )
        weir_discharge: np.float32 = rectangular_weir_discharge(
            upstream_stage_m,
            downstream_stage_m,
            crest_m,
            geom_inbank[reach_idx, GEOM_IN_WEIR_WIDTH],
        )
        if weir_discharge < 0:
            # The kernel limits the shared downstream donor volume for reverse flow.
            return weir_discharge if can_reverse else np.float32(0.0)
        # Only storage above the crest can spill. The volume relation is
        # V(h) = V_bankfull * (h / h_bankfull)**(shape + 1); geometry validation
        # keeps the crest within the banks where this relation applies.
        # The 95% cap matches ordinary routing's protection against over-draining.
        crest_volume_m3: np.float32 = geom_inbank[
            reach_idx, GEOM_IN_BANKFULL_VOLUME
        ] * (weir_height_m / geom_inbank[reach_idx, GEOM_IN_BANKFULL_DEPTH]) ** (
            np.float32(1.0)
            / geom_inbank[reach_idx, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE]
        )
        available_m3: np.float32 = np.float32(
            max(river_storage_m3_inertial[reach_idx] - crest_volume_m3, 0.0)
        )
        return min(weir_discharge, np.float32(0.95) * available_m3 * inv_dt_substep)

    if effective_depth < min_wet_depth_m:
        return np.float32(0.0)

    bankfull_depth_m: np.float32 = geom_inbank[reach_idx, GEOM_IN_BANKFULL_DEPTH]
    inv_shape_exponent_plus_one: np.float32 = geom_inbank[
        reach_idx, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE
    ]
    # Preserve exact 0.5 for standard parabolic channels to match geometry lookup branches
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

    # Semi-implicit Manning friction term: A * R^(4/3) = A * (A / P)^(4/3)
    hydraulic_radius: np.float32 = cross_sectional_area / max(
        interface_p_wetted, np.float32(1e-6)
    )
    friction_denom_term: np.float32 = max(
        cross_sectional_area * (hydraulic_radius ** np.float32(4.0 / 3.0)),
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
