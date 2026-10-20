"""Tests for get_green_ampt_params helper function."""

import numpy as np

from geb.hydrology.landsurface.water import (
    calculate_effective_wetting_front_suction,
    get_green_ampt_params,
)


def test_calculate_effective_wetting_front_suction_values() -> None:
    """Test effective wetting front suction against analytical solutions for standard soils."""
    # Sand: psi_b = 0.0726 m, lambda = 0.694 (Rawls et al., 1983)
    psi_b_sand: np.float32 = np.float32(0.0726)
    lambda_sand: np.float32 = np.float32(0.694)

    psi_f_sand_dry: np.float32 = calculate_effective_wetting_front_suction(
        psi_b_sand, lambda_sand, np.float32(0.0)
    )
    # Expected: (0.0726 / 2) * (2 + 3*0.694) / (1 + 3*0.694) ~ 0.0481 m
    assert 0.045 < psi_f_sand_dry < 0.052

    # Loam: psi_b = 0.1115 m, lambda = 0.252 (Rawls et al., 1983)
    psi_b_loam: np.float32 = np.float32(0.1115)
    lambda_loam: np.float32 = np.float32(0.252)

    psi_f_loam_dry: np.float32 = calculate_effective_wetting_front_suction(
        psi_b_loam, lambda_loam, np.float32(0.0)
    )
    # Expected: (0.1115 / 2) * (2 + 3*0.252) / (1 + 3*0.252) ~ 0.0875 m
    assert 0.080 < psi_f_loam_dry < 0.095

    # Clay: psi_b = 0.3730 m, lambda = 0.131 (Rawls et al., 1983)
    psi_b_clay: np.float32 = np.float32(0.3730)
    lambda_clay: np.float32 = np.float32(0.131)

    psi_f_clay_dry: np.float32 = calculate_effective_wetting_front_suction(
        psi_b_clay, lambda_clay, np.float32(0.0)
    )
    # Expected: (0.3730 / 2) * (2 + 3*0.131) / (1 + 3*0.131) ~ 0.3204 m
    assert 0.300 < psi_f_clay_dry < 0.350


def test_calculate_effective_wetting_front_suction_monotonicity() -> None:
    """Test that suction head decreases monotonically as effective saturation increases."""
    psi_b: np.float32 = np.float32(0.15)
    lambda_val: np.float32 = np.float32(0.25)

    saturations: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    suctions: list[float] = [
        float(
            calculate_effective_wetting_front_suction(psi_b, lambda_val, np.float32(s))
        )
        for s in saturations
    ]

    for i in range(len(suctions) - 1):
        assert suctions[i] >= suctions[i + 1], (
            f"Suction at S_e={saturations[i]} ({suctions[i]}) must be >= suction at S_e={saturations[i + 1]} ({suctions[i + 1]})"
        )

    # At full saturation, suction must equal psi_b / 2
    psi_f_saturated: np.float32 = calculate_effective_wetting_front_suction(
        psi_b, lambda_val, np.float32(1.0)
    )
    assert abs(psi_f_saturated - psi_b / np.float32(2.0)) < 1e-5


def test_calculate_effective_wetting_front_suction_bounds_and_safeguards() -> None:
    """Test boundary clipping and protection against degenerate inputs."""
    psi_b: np.float32 = np.float32(0.2)
    lambda_val: np.float32 = np.float32(0.3)

    # Out of bounds saturation should be clipped
    psi_subzero: np.float32 = calculate_effective_wetting_front_suction(
        psi_b, lambda_val, np.float32(-0.5)
    )
    psi_zero: np.float32 = calculate_effective_wetting_front_suction(
        psi_b, lambda_val, np.float32(0.0)
    )
    assert psi_subzero == psi_zero

    psi_overone: np.float32 = calculate_effective_wetting_front_suction(
        psi_b, lambda_val, np.float32(1.5)
    )
    psi_one: np.float32 = calculate_effective_wetting_front_suction(
        psi_b, lambda_val, np.float32(1.0)
    )
    assert psi_overone == psi_one

    # Degenerate/zero inputs should not crash or return NaN / <= 0
    psi_deg: np.float32 = calculate_effective_wetting_front_suction(
        np.float32(0.0), np.float32(0.0), np.float32(0.5)
    )
    assert not np.isnan(psi_deg)
    assert psi_deg >= np.float32(1e-4)


def test_get_green_ampt_params_start_of_infiltration() -> None:
    """Test parameters at the very start (wetting front depth = 0)."""
    # 2 layers, 1m each
    soil_layer_height_m: np.ndarray = np.array([1.0, 1.0], dtype=np.float32)
    # Saturation 50%, Current 20%
    ws: np.ndarray = np.array([0.5, 0.5], dtype=np.float32)
    w: np.ndarray = np.array([0.2, 0.2], dtype=np.float32)
    wres: np.ndarray = np.array([0.05, 0.05], dtype=np.float32)

    # Hydraulic props
    bubbling: np.ndarray = np.array([0.10, 0.10], dtype=np.float32)  # meters
    lamb: np.ndarray = np.array([0.5, 0.5], dtype=np.float32)

    wetting_front_depth_m: np.float32 = np.float32(0.0)

    idx, suction, dtheta = get_green_ampt_params(
        wetting_front_depth_m, soil_layer_height_m, w, ws, wres, bubbling, lamb
    )

    assert idx == 0
    # At depth 0, entire layer 1 is ahead.
    # theta_initial = 0.2 / 1.0 = 0.2
    # theta_sat = 0.5 / 1.0 = 0.5
    # dtheta = 0.5 - 0.2 = 0.3
    assert abs(dtheta - 0.3) < 1e-6
    # Suction should be calculated based on theta_initial=0.2 (w=0.2)
    assert suction > 0.0
    assert 0.02 <= suction <= 0.15


def test_get_green_ampt_params_advanced_front() -> None:
    """Test parameters when wetting front has advanced into the first layer."""
    soil_layer_height_m: np.ndarray = np.array([1.0, 1.0], dtype=np.float32)
    # Layer 1: ws=0.5. Current total w=0.25m.
    # We assume front is at 0.1m.
    # Behind front (0.1m): Saturated (theta=0.5) -> water = 0.05m.
    # Ahead of front (0.9m): Remaining water = 0.25 - 0.05 = 0.20m.
    # theta_ahead = 0.20 / 0.9 = 0.2222...

    ws: np.ndarray = np.array([0.5, 0.5], dtype=np.float32)
    w: np.ndarray = np.array([0.25, 0.2], dtype=np.float32)
    wres: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)
    bubbling: np.ndarray = np.array([1.0, 1.0], dtype=np.float32)  # Dummy
    lamb: np.ndarray = np.array([1.0, 1.0], dtype=np.float32)  # Dummy

    wetting_front_depth_m: np.float32 = np.float32(0.1)

    idx, _, dtheta = get_green_ampt_params(
        wetting_front_depth_m, soil_layer_height_m, w, ws, wres, bubbling, lamb
    )

    assert idx == 0
    theta_sat: np.float32 = np.float32(0.5)
    theta_initial_expected: np.float32 = np.float32(0.2 / 0.9)
    dtheta_expected: np.float32 = np.float32(theta_sat) - theta_initial_expected
    assert abs(dtheta - dtheta_expected) < 1e-5


def test_get_green_ampt_params_second_layer() -> None:
    """Test parameters when front is in second layer."""
    soil_layer_height_m: np.ndarray = np.array([0.5, 1.0], dtype=np.float32)
    # Front at 0.6m. (0.1m into second layer).

    # Layer 2: ws=0.4 (theta_sat=0.4/1.0=0.4). w=0.15.
    # Behind in Layer 2: 0.1m * 0.4 = 0.04m water.
    # Ahead in Layer 2: 0.15 - 0.04 = 0.11m water.
    # Height ahead: 0.9m.
    # theta_ahead = 0.11 / 0.9 = 0.1222...

    ws: np.ndarray = np.array([0.25, 0.4], dtype=np.float32)
    w: np.ndarray = np.array(
        [0.25, 0.15], dtype=np.float32
    )  # Layer 1 is full/irrelevant for params
    wres: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)
    bubbling: np.ndarray = np.array([1.0, 1.0], dtype=np.float32)
    lamb: np.ndarray = np.array([1.0, 1.0], dtype=np.float32)

    wetting_front_depth_m: np.float32 = np.float32(0.6)

    idx, _, dtheta = get_green_ampt_params(
        wetting_front_depth_m, soil_layer_height_m, w, ws, wres, bubbling, lamb
    )

    assert idx == 1
    theta_sat: np.float32 = np.float32(0.4)
    theta_initial_expected: np.float32 = np.float32(0.11 / 0.9)
    dtheta_expected: np.float32 = np.float32(theta_sat) - theta_initial_expected
    assert abs(dtheta - dtheta_expected) < 1e-5


def test_get_green_ampt_params_layered_no_interference() -> None:
    """Test a scenario where a dry coarse layer sits on top of a wet fine layer.

    Ensures that the wet lower layer does not artificially saturate the upper layer parameter calculation.

    Layer 0 (Top): Sand-like.
       - Height: 1.0m
       - Saturation (ws): 0.4
       - Current Content: 0.1 (Very dry) -> theta = 0.1

    Layer 1 (Bottom): Clay-like.
       - Height: 1.0m
       - Saturation (ws): 0.6
       - Current Content: 0.5 (Wet) -> theta = 0.5

    The heuristic  sets theta_floor = 0.5.
    This forces Layer 0 theta_initial to be max(0.1, 0.5) = 0.5.
    But Layer 0 capacity is only 0.4.
    So it gets clamped to 0.4 (saturated).
    Delta_theta becomes ~0.

    Physical expectation: Layer 0 is dry. Delta_theta should be 0.4 - 0.1 = 0.3.
    """
    soil_layer_height_m: np.ndarray = np.array([1.0, 1.0], dtype=np.float32)

    ws: np.ndarray = np.array([0.4, 0.6], dtype=np.float32)
    w: np.ndarray = np.array([0.1, 0.5], dtype=np.float32)
    wres: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)

    # Hydraulic props (dummies)
    bubbling: np.ndarray = np.array([5.0, 20.0], dtype=np.float32)
    lamb: np.ndarray = np.array([0.5, 0.2], dtype=np.float32)

    wetting_front_depth_m: np.float32 = np.float32(0.0)

    idx, _, dtheta = get_green_ampt_params(
        wetting_front_depth_m, soil_layer_height_m, w, ws, wres, bubbling, lamb
    )

    # We assert the CORRECT physical behavior
    assert abs(dtheta - 0.3) < 1e-4, (
        f"Lower layer moisture incorrectly affected upper layer suction! dtheta={dtheta}, expected 0.3"
    )
