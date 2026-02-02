"""Tests for get_green_ampt_params helper function."""

import numpy as np

from geb.hydrology.soil import get_green_ampt_params


def test_get_green_ampt_params_start_of_infiltration() -> None:
    """Test parameters at the very start (wetting front depth = 0)."""
    # 2 layers, 1m each
    soil_layer_height_m = np.array([1.0, 1.0], dtype=np.float32)
    # Saturation 50%, Current 20%
    ws = np.array([0.5, 0.5], dtype=np.float32)
    w = np.array([0.2, 0.2], dtype=np.float32)
    wres = np.array([0.05, 0.05], dtype=np.float32)

    # Hydraulic props
    ks = np.array([1e-5, 1e-5], dtype=np.float32)
    bubbling = np.array([10.0, 10.0], dtype=np.float32)  # cm
    lamb = np.array([0.5, 0.5], dtype=np.float32)

    wetting_front_depth_m = np.float32(0.0)

    idx, suction, dtheta = get_green_ampt_params.py_func(
        wetting_front_depth_m, soil_layer_height_m, w, ws, wres, ks, bubbling, lamb
    )

    assert idx == 0
    # At depth 0, entire layer 1 is ahead.
    # theta_initial = 0.2 / 1.0 = 0.2
    # theta_sat = 0.5 / 1.0 = 0.5
    # dtheta = 0.5 - 0.2 = 0.3
    assert abs(dtheta - 0.3) < 1e-6
    # Suction should be calculated based on theta_initial=0.2 (w=0.2)
    assert suction > 0.0


def test_get_green_ampt_params_advanced_front() -> None:
    """Test parameters when wetting front has advanced into the first layer."""
    soil_layer_height_m = np.array([1.0, 1.0], dtype=np.float32)
    # Layer 1: ws=0.5. Current total w=0.25m.
    # We assume front is at 0.1m.
    # Behind front (0.1m): Saturated (theta=0.5) -> water = 0.05m.
    # Ahead of front (0.9m): Remaining water = 0.25 - 0.05 = 0.20m.
    # theta_ahead = 0.20 / 0.9 = 0.2222...

    ws = np.array([0.5, 0.5], dtype=np.float32)
    w = np.array([0.25, 0.2], dtype=np.float32)
    wres = np.array([0.0, 0.0], dtype=np.float32)
    ks = np.array([1.0, 1.0], dtype=np.float32)  # Dummy
    bubbling = np.array([1.0, 1.0], dtype=np.float32)  # Dummy
    lamb = np.array([1.0, 1.0], dtype=np.float32)  # Dummy

    wetting_front_depth_m = np.float32(0.1)

    idx, _, dtheta = get_green_ampt_params.py_func(
        wetting_front_depth_m, soil_layer_height_m, w, ws, wres, ks, bubbling, lamb
    )

    assert idx == 0
    theta_sat = 0.5
    theta_initial_expected = np.float32(0.2 / 0.9)
    dtheta_expected = theta_sat - theta_initial_expected
    assert abs(dtheta - dtheta_expected) < 1e-5


def test_get_green_ampt_params_second_layer() -> None:
    """Test parameters when front is in second layer."""
    soil_layer_height_m = np.array([0.5, 1.0], dtype=np.float32)
    # Front at 0.6m. (0.1m into second layer).

    # Layer 2: ws=0.4 (theta_sat=0.4/1.0=0.4). w=0.15.
    # Behind in Layer 2: 0.1m * 0.4 = 0.04m water.
    # Ahead in Layer 2: 0.15 - 0.04 = 0.11m water.
    # Height ahead: 0.9m.
    # theta_ahead = 0.11 / 0.9 = 0.1222...

    ws = np.array([0.25, 0.4], dtype=np.float32)
    w = np.array(
        [0.25, 0.15], dtype=np.float32
    )  # Layer 1 is full/irrelevant for params
    wres = np.array([0.0, 0.0], dtype=np.float32)
    ks = np.array([1.0, 1.0], dtype=np.float32)
    bubbling = np.array([1.0, 1.0], dtype=np.float32)
    lamb = np.array([1.0, 1.0], dtype=np.float32)

    wetting_front_depth_m = np.float32(0.6)

    idx, _, dtheta = get_green_ampt_params.py_func(
        wetting_front_depth_m, soil_layer_height_m, w, ws, wres, ks, bubbling, lamb
    )

    assert idx == 1
    theta_sat = 0.4
    theta_initial_expected = np.float32(0.11 / 0.9)
    dtheta_expected = theta_sat - theta_initial_expected
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

    The heuristic `theta_floor = theta_next` sets theta_floor = 0.5.
    This forces Layer 0 theta_initial to be max(0.1, 0.5) = 0.5.
    But Layer 0 capacity is only 0.4.
    So it gets clamped to 0.4 (saturated).
    Delta_theta becomes ~0.

    Physical expectation: Layer 0 is dry. Delta_theta should be 0.4 - 0.1 = 0.3.
    """
    soil_layer_height_m = np.array([1.0, 1.0], dtype=np.float32)

    ws = np.array([0.4, 0.6], dtype=np.float32)
    w = np.array([0.1, 0.5], dtype=np.float32)
    wres = np.array([0.0, 0.0], dtype=np.float32)

    # Hydraulic props (dummies)
    ks = np.array([1e-5, 1e-6], dtype=np.float32)
    bubbling = np.array([5.0, 20.0], dtype=np.float32)
    lamb = np.array([0.5, 0.2], dtype=np.float32)

    wetting_front_depth_m = np.float32(0.0)

    idx, _, dtheta = get_green_ampt_params.py_func(
        wetting_front_depth_m, soil_layer_height_m, w, ws, wres, ks, bubbling, lamb
    )

    # We assert the CORRECT physical behavior
    assert abs(dtheta - 0.3) < 1e-4, (
        f"Lower layer moisture incorrectly affected upper layer suction! dtheta={dtheta}, expected 0.3"
    )
