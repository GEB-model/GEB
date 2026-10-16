"""Unit tests for the 1D local inertial momentum solver.

Tests the semi-implicit momentum equation solver (solve_inertial_momentum),
including gravity acceleration, Manning friction damping, critical flow limiting,
upstream storage volume limiting, and backwater/reversal dynamics.
"""

import numpy as np
import pytest

from geb.geb_types import ArrayFloat64, TwoDArrayFloat32
from geb.hydrology.routing.geometry import (
    GEOM_IN_BANKFULL_AREA,
    GEOM_IN_BANKFULL_DEPTH,
    GEOM_IN_BANKFULL_PERIMETER,
    GEOM_IN_BANKFULL_VOLUME,
    GEOM_IN_INVERSE_LENGTH,
    GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE,
    GEOM_IN_MANNING_N_SQUARED,
    GEOM_IN_NUM_COLS,
    GEOM_IN_RIVER_WIDTH,
    GEOM_IN_SHAPE_EXPONENT,
    GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH,
    GEOM_OV_BANKFULL_PERIMETER,
    GEOM_OV_NUM_COLS,
    GEOM_OV_RIVER_WIDTH,
)
from geb.hydrology.routing.solver import (
    evaluate_discharge_pow_0_6,
    evaluate_hydraulic_radius_pow_4_3,
    solve_inertial_momentum,
)


def _create_single_reach_geometry(
    bankfull_width_m: float = 10.0,
    bankfull_depth_m: float = 2.0,
    reach_length_m: float = 1000.0,
    manning_n: float = 0.035,
) -> tuple[TwoDArrayFloat32, TwoDArrayFloat32]:
    """Helper to create inbank and overbank geometry tables for a single test reach.

    Args:
        bankfull_width_m: Bankfull channel width (meters).
        bankfull_depth_m: Bankfull channel depth (meters).
        reach_length_m: Reach length (meters).
        manning_n: Manning roughness coefficient (s / m^(1/3)).

    Returns:
        Tuple of (geom_inbank, geom_overbank) 2D arrays.
    """
    geom_inbank: TwoDArrayFloat32 = np.zeros((1, GEOM_IN_NUM_COLS), dtype=np.float32)
    geom_overbank: TwoDArrayFloat32 = np.zeros((1, GEOM_OV_NUM_COLS), dtype=np.float32)

    # Parabolic shape exponent 0.5 -> inv_shape_exponent_plus_one = 1 / 1.5
    inv_shape_plus_one: float = 1.0 / 1.5
    bankfull_area_m2: float = bankfull_width_m * bankfull_depth_m * inv_shape_plus_one
    bankfull_volume_m3: float = bankfull_area_m2 * reach_length_m

    geom_inbank[0, GEOM_IN_BANKFULL_DEPTH] = np.float32(bankfull_depth_m)
    geom_inbank[0, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE] = np.float32(
        inv_shape_plus_one
    )
    geom_inbank[0, GEOM_IN_BANKFULL_VOLUME] = np.float32(bankfull_volume_m3)
    geom_inbank[0, GEOM_IN_INVERSE_LENGTH] = np.float32(1.0 / reach_length_m)
    geom_inbank[0, GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH] = np.float32(
        bankfull_width_m / np.sqrt(bankfull_depth_m)
    )
    geom_inbank[0, GEOM_IN_MANNING_N_SQUARED] = np.float32(manning_n**2)
    geom_inbank[0, GEOM_IN_BANKFULL_AREA] = np.float32(
        bankfull_volume_m3 * (1.0 / reach_length_m)
    )
    geom_inbank[0, GEOM_IN_SHAPE_EXPONENT] = np.float32(0.5)
    geom_inbank[0, GEOM_IN_RIVER_WIDTH] = np.float32(bankfull_width_m)
    geom_inbank[0, GEOM_IN_BANKFULL_PERIMETER] = np.float32(
        bankfull_width_m + 2.0 * bankfull_depth_m
    )

    geom_overbank[0, GEOM_OV_RIVER_WIDTH] = np.float32(bankfull_width_m)
    geom_overbank[0, GEOM_OV_BANKFULL_PERIMETER] = np.float32(
        bankfull_width_m + 2.0 * bankfull_depth_m
    )

    return geom_inbank, geom_overbank


def test_dry_bed_cutoff() -> None:
    """Tests that dry reach interfaces produce strictly zero discharge."""
    geom_inbank, geom_overbank = _create_single_reach_geometry()
    storage: ArrayFloat64 = np.array([10000.0], dtype=np.float64)

    # Effective depth below minimum wet depth
    q: np.float32 = solve_inertial_momentum(
        reach_idx=0,
        effective_depth=np.float32(0.005),
        water_slope=np.float32(-0.001),
        curr_discharge=np.float32(5.0),
        min_wet_depth_m=np.float32(0.01),
        geom_inbank=geom_inbank,
        geom_overbank=geom_overbank,
        g_dt_substep=np.float32(9.80665 * 10.0),
        sqrt_gravity=np.float32(np.sqrt(9.80665)),
        can_reverse=True,
        river_storage_m3_inertial=storage,
        inv_dt_substep=np.float32(1.0 / 10.0),
    )
    assert q == np.float32(0.0)


def test_downhill_gravity_acceleration() -> None:
    """Tests forward flow acceleration down a downhill water surface slope."""
    geom_inbank, geom_overbank = _create_single_reach_geometry(manning_n=0.03)
    storage: ArrayFloat64 = np.array([50000.0], dtype=np.float64)

    # Downhill slope: dz/dx = -0.001
    q_forward: np.float32 = solve_inertial_momentum(
        reach_idx=0,
        effective_depth=np.float32(1.0),
        water_slope=np.float32(-0.001),
        curr_discharge=np.float32(0.0),
        min_wet_depth_m=np.float32(0.01),
        geom_inbank=geom_inbank,
        geom_overbank=geom_overbank,
        g_dt_substep=np.float32(9.80665 * 5.0),
        sqrt_gravity=np.float32(np.sqrt(9.80665)),
        can_reverse=True,
        river_storage_m3_inertial=storage,
        inv_dt_substep=np.float32(1.0 / 5.0),
    )
    assert q_forward > 0.0


def test_adverse_slope_deceleration_and_reversal() -> None:
    """Tests backwater deceleration and reverse flow on an adverse uphill slope."""
    geom_inbank, geom_overbank = _create_single_reach_geometry(manning_n=0.03)
    storage: ArrayFloat64 = np.array([50000.0], dtype=np.float64)

    # Adverse slope: dz/dx = +0.002
    q_rev: np.float32 = solve_inertial_momentum(
        reach_idx=0,
        effective_depth=np.float32(1.0),
        water_slope=np.float32(0.002),
        curr_discharge=np.float32(0.0),
        min_wet_depth_m=np.float32(0.01),
        geom_inbank=geom_inbank,
        geom_overbank=geom_overbank,
        g_dt_substep=np.float32(9.80665 * 5.0),
        sqrt_gravity=np.float32(np.sqrt(9.80665)),
        can_reverse=True,
        river_storage_m3_inertial=storage,
        inv_dt_substep=np.float32(1.0 / 5.0),
    )
    assert q_rev < 0.0


def test_reverse_flow_suppressed_when_can_reverse_false() -> None:
    """Tests that reverse flow is capped at 0 when can_reverse is False."""
    geom_inbank, geom_overbank = _create_single_reach_geometry()
    storage: ArrayFloat64 = np.array([50000.0], dtype=np.float64)

    # Adverse slope would cause negative flow, but can_reverse=False
    q: np.float32 = solve_inertial_momentum(
        reach_idx=0,
        effective_depth=np.float32(1.0),
        water_slope=np.float32(0.005),
        curr_discharge=np.float32(0.0),
        min_wet_depth_m=np.float32(0.01),
        geom_inbank=geom_inbank,
        geom_overbank=geom_overbank,
        g_dt_substep=np.float32(9.80665 * 5.0),
        sqrt_gravity=np.float32(np.sqrt(9.80665)),
        can_reverse=False,
        river_storage_m3_inertial=storage,
        inv_dt_substep=np.float32(1.0 / 5.0),
    )
    assert q == np.float32(0.0)


def test_manning_friction_damping_monotonicity_and_stability() -> None:
    """Tests that higher Manning n increases friction damping monotonically without instability."""
    geom_low_n, geom_ov_low = _create_single_reach_geometry(manning_n=0.015)
    geom_high_n, geom_ov_high = _create_single_reach_geometry(manning_n=0.060)
    geom_extreme_n, geom_ov_extreme = _create_single_reach_geometry(manning_n=0.50)
    storage: ArrayFloat64 = np.array([100000.0], dtype=np.float64)

    q_low: np.float32 = solve_inertial_momentum(
        reach_idx=0,
        effective_depth=np.float32(1.5),
        water_slope=np.float32(-0.002),
        curr_discharge=np.float32(10.0),
        min_wet_depth_m=np.float32(0.01),
        geom_inbank=geom_low_n,
        geom_overbank=geom_ov_low,
        g_dt_substep=np.float32(9.80665 * 10.0),
        sqrt_gravity=np.float32(np.sqrt(9.80665)),
        can_reverse=True,
        river_storage_m3_inertial=storage,
        inv_dt_substep=np.float32(1.0 / 10.0),
    )
    q_high: np.float32 = solve_inertial_momentum(
        reach_idx=0,
        effective_depth=np.float32(1.5),
        water_slope=np.float32(-0.002),
        curr_discharge=np.float32(10.0),
        min_wet_depth_m=np.float32(0.01),
        geom_inbank=geom_high_n,
        geom_overbank=geom_ov_high,
        g_dt_substep=np.float32(9.80665 * 10.0),
        sqrt_gravity=np.float32(np.sqrt(9.80665)),
        can_reverse=True,
        river_storage_m3_inertial=storage,
        inv_dt_substep=np.float32(1.0 / 10.0),
    )
    q_extreme: np.float32 = solve_inertial_momentum(
        reach_idx=0,
        effective_depth=np.float32(1.5),
        water_slope=np.float32(-0.002),
        curr_discharge=np.float32(10.0),
        min_wet_depth_m=np.float32(0.01),
        geom_inbank=geom_extreme_n,
        geom_overbank=geom_ov_extreme,
        g_dt_substep=np.float32(9.80665 * 10.0),
        sqrt_gravity=np.float32(np.sqrt(9.80665)),
        can_reverse=True,
        river_storage_m3_inertial=storage,
        inv_dt_substep=np.float32(1.0 / 10.0),
    )

    # Monotonic damping with roughness
    assert q_low > q_high > q_extreme > 0.0


def test_critical_flow_froude_limiter() -> None:
    """Tests that supercritical runaway over a vertical drop is capped by Fr <= 1 celerity."""
    geom_inbank, geom_overbank = _create_single_reach_geometry(manning_n=0.01)
    storage: ArrayFloat64 = np.array([1000000.0], dtype=np.float64)

    # Extreme waterfall slope: dz/dx = -1.0
    q: np.float32 = solve_inertial_momentum(
        reach_idx=0,
        effective_depth=np.float32(1.0),
        water_slope=np.float32(-1.0),
        curr_discharge=np.float32(0.0),
        min_wet_depth_m=np.float32(0.01),
        geom_inbank=geom_inbank,
        geom_overbank=geom_overbank,
        g_dt_substep=np.float32(9.80665 * 60.0),
        sqrt_gravity=np.float32(np.sqrt(9.80665)),
        can_reverse=True,
        river_storage_m3_inertial=storage,
        inv_dt_substep=np.float32(1.0 / 60.0),
    )

    # Critical discharge Fr=1: Q_crit = A * sqrt(g * D_h)
    # Area for h=1.0: W = 10 / sqrt(2) * sqrt(1) = 7.071, A = W * h * (1/1.5) = 4.714
    # D_h = A / W = 2/3 = 0.667
    # Q_crit = 4.714 * sqrt(9.80665 * 0.667) ~ 12.0
    assert q > 0.0
    assert q < 15.0  # Must be strictly bounded by critical flow celerity


def test_upstream_storage_volume_limiter() -> None:
    """Tests that outflow cannot exceed 95% of available upstream reach volume per substep."""
    geom_inbank, geom_overbank = _create_single_reach_geometry()
    dt_s: float = 10.0
    small_storage: ArrayFloat64 = np.array([50.0], dtype=np.float64)

    q: np.float32 = solve_inertial_momentum(
        reach_idx=0,
        effective_depth=np.float32(1.0),
        water_slope=np.float32(-0.01),
        curr_discharge=np.float32(20.0),
        min_wet_depth_m=np.float32(0.01),
        geom_inbank=geom_inbank,
        geom_overbank=geom_overbank,
        g_dt_substep=np.float32(9.80665 * dt_s),
        sqrt_gravity=np.float32(np.sqrt(9.80665)),
        can_reverse=True,
        river_storage_m3_inertial=small_storage,
        inv_dt_substep=np.float32(1.0 / dt_s),
    )

    # Maximum drain rate: 0.95 * 50.0 / 10.0 = 4.75 m3/s
    assert q == pytest.approx(4.75, rel=1e-4)


def test_zero_upstream_storage_produces_zero_discharge() -> None:
    """Tests that a reach with zero available storage produces zero forward discharge."""
    geom_inbank, geom_overbank = _create_single_reach_geometry()
    zero_storage: ArrayFloat64 = np.array([0.0], dtype=np.float64)

    q: np.float32 = solve_inertial_momentum(
        reach_idx=0,
        effective_depth=np.float32(1.0),
        water_slope=np.float32(-0.01),
        curr_discharge=np.float32(10.0),
        min_wet_depth_m=np.float32(0.01),
        geom_inbank=geom_inbank,
        geom_overbank=geom_overbank,
        g_dt_substep=np.float32(9.80665 * 10.0),
        sqrt_gravity=np.float32(np.sqrt(9.80665)),
        can_reverse=True,
        river_storage_m3_inertial=zero_storage,
        inv_dt_substep=np.float32(1.0 / 10.0),
    )
    assert q == np.float32(0.0)


def test_evaluate_hydraulic_radius_pow_4_3_happy_path() -> None:
    """Tests that hydraulic radius exponentiation matches analytical values across normal depths."""
    test_radii: list[float] = [0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 28.0]
    for r in test_radii:
        r_f32: np.float32 = np.float32(r)
        approx_val: np.float32 = evaluate_hydraulic_radius_pow_4_3(r_f32)
        exact_val: float = r ** (4.0 / 3.0)
        rel_error: float = abs(float(approx_val) - exact_val) / exact_val
        # Across normal river depths >= 0.01 m, error must be well under 0.25%
        assert rel_error < 0.0025


def test_evaluate_hydraulic_radius_pow_4_3_boundary_conditions() -> None:
    """Tests boundary conditions for lookup table exponentiation."""
    # Zero hydraulic radius produces zero
    assert evaluate_hydraulic_radius_pow_4_3(np.float32(0.0)) == np.float32(0.0)

    # Minimum wet depth (0.1 mm) has bounded relative error (< 10%)
    min_wet: np.float32 = np.float32(1e-4)
    approx_min: np.float32 = evaluate_hydraulic_radius_pow_4_3(min_wet)
    exact_min: float = 1e-4 ** (4.0 / 3.0)
    assert abs(float(approx_min) - exact_min) / exact_min < 0.10

    # At table boundary (30.0 m) produces accurate (30.0)^(4/3)
    r_30: np.float32 = np.float32(30.0)
    approx_30: np.float32 = evaluate_hydraulic_radius_pow_4_3(r_30)
    exact_30: float = 30.0 ** (4.0 / 3.0)
    assert abs(float(approx_30) - exact_30) / exact_30 < 0.001


def test_evaluate_discharge_pow_0_6_happy_path() -> None:
    """Tests that discharge exponentiation Q^(0.6) matches analytical values across normal flows."""
    test_discharges: list[float] = [
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
        5000.0,
        25000.0,
        90000.0,
    ]
    for q in test_discharges:
        q_f32: np.float32 = np.float32(q)
        approx_val: np.float32 = evaluate_discharge_pow_0_6(q_f32)
        exact_val: float = q**0.6
        rel_error: float = abs(float(approx_val) - exact_val) / exact_val
        # Relative error must stay well below 0.4% across the entire operational range
        assert rel_error < 0.004


def test_evaluate_discharge_pow_0_6_boundary_conditions() -> None:
    """Tests boundary conditions for Q^(0.6) lookup table."""
    # Zero or negative discharge produces zero
    assert evaluate_discharge_pow_0_6(np.float32(0.0)) == np.float32(0.0)
    assert evaluate_discharge_pow_0_6(np.float32(-10.0)) == np.float32(0.0)

    # At table boundary (100,000 m³/s) matches exact power accurately
    q_max: np.float32 = np.float32(100_000.0)
    approx_max: np.float32 = evaluate_discharge_pow_0_6(q_max)
    exact_max: float = 100_000.0**0.6
    assert abs(float(approx_max) - exact_max) / exact_max < 0.001

    # Beyond table boundary clamps cleanly without crashing
    q_beyond: np.float32 = np.float32(200_000.0)
    approx_beyond: np.float32 = evaluate_discharge_pow_0_6(q_beyond)
    assert approx_beyond > 0.0
