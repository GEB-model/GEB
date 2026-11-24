"""Unit tests for the Green-Ampt infiltration increment.

The increment over a step is compared against a bisection solution of the implicit
Green-Ampt relation t(F) = (F - P ln(1 + F/P)) / K, with P = psi * dtheta.
"""

import numpy as np

from geb.hydrology.landsurface.water import (
    calculate_green_ampt_infiltration_increment,
)


def _solve_standard_green_ampt_cumulative_infiltration(
    time_steps: float,
    saturated_hydraulic_conductivity_m_per_step: float,
    suction_head_m: float,
    dtheta: float,
    *,
    max_iter: int = 200,
) -> float:
    """Solve the standard (implicit) Green-Ampt I(t) relation numerically.

    Notes:
        The standard Green-Ampt model can be written as an implicit equation in
        cumulative infiltration I:
            t(I) = (I - S_f ln(1 + I/S_f)) / K_s
        where S_f = psi_f Δθ.

        This helper uses bisection (robust monotonic root find) and is intended
        only for unit tests.

    Args:
        time_steps: Time since start of infiltration (timestep units).
        saturated_hydraulic_conductivity_m_per_step: Saturated hydraulic conductivity K_s (m/timestep).
        suction_head_m: Wetting front suction head psi_f (m).
        dtheta: Moisture deficit Δθ (-).
        max_iter: Maximum bisection iterations.

    Returns:
        Cumulative infiltration I(t) (m).
    """
    if time_steps <= float(0.0):
        return float(0.0)

    if saturated_hydraulic_conductivity_m_per_step <= float(0.0):
        return float(0.0)

    sf_m: float = suction_head_m * dtheta
    if sf_m <= float(0.0):
        return saturated_hydraulic_conductivity_m_per_step * time_steps

    def t_from_i(i_m: float) -> float:
        return (
            i_m - sf_m * np.log(float(1.0) + i_m / sf_m)
        ) / saturated_hydraulic_conductivity_m_per_step

    # Bracket: t_from_i(0)=0 <= time_steps. Find hi so that t_from_i(hi) >= time_steps.
    lo_m: float = float(0.0)
    hi_m: float = saturated_hydraulic_conductivity_m_per_step * time_steps + sf_m

    # Expand hi if needed (rare for extremely large suction effects).
    for _ in range(60):
        if t_from_i(hi_m) >= time_steps:
            break
        hi_m = hi_m * float(2.0)
    else:
        # If we failed to bracket, return the best-effort hi.
        return hi_m

    for _ in range(max_iter):
        mid_m: float = (lo_m + hi_m) / float(2.0)
        if t_from_i(mid_m) < time_steps:
            lo_m = mid_m
        else:
            hi_m = mid_m

    return (lo_m + hi_m) / float(2.0)


def _time_from_cumulative_infiltration(
    cumulative_m: float, conductivity: float, suction_head_m: float, dtheta: float
) -> float:
    """Exact Green-Ampt time needed to infiltrate a cumulative depth.

    Args:
        cumulative_m: Cumulative infiltration F (m).
        conductivity: Saturated hydraulic conductivity K (m per time unit).
        suction_head_m: Wetting front suction head psi (m).
        dtheta: Moisture deficit (-).

    Returns:
        Time since the start of ponding (time units).
    """
    suction_potential = suction_head_m * dtheta
    return (
        cumulative_m - suction_potential * np.log1p(cumulative_m / suction_potential)
    ) / conductivity


def test_increment_darcy_limit_zero_suction() -> None:
    """Without capillary suction the increment is K dt."""
    increment = calculate_green_ampt_infiltration_increment(
        np.float32(0.05),
        np.float32(1.2e-5),
        np.float32(0.0),
        np.float32(0.25),
        np.float32(600.0),
    )
    assert abs(increment - np.float32(1.2e-5 * 600.0)) < 1e-9


def test_increment_zero_conductivity_or_duration() -> None:
    """No conductivity or no time gives no infiltration."""
    assert (
        calculate_green_ampt_infiltration_increment(
            np.float32(0.05),
            np.float32(0.0),
            np.float32(0.2),
            np.float32(0.25),
            np.float32(600.0),
        )
        == 0.0
    )
    assert (
        calculate_green_ampt_infiltration_increment(
            np.float32(0.05),
            np.float32(1e-6),
            np.float32(0.2),
            np.float32(0.25),
            np.float32(0.0),
        )
        == 0.0
    )


def test_increment_matches_implicit_green_ampt() -> None:
    """The increment matches the implicit solution for sand to clay and deep fronts.

    These are the cases where the previous approach, the difference between an
    explicit approximation of the cumulative infiltration and the current
    cumulative infiltration, was off by tens of percent or returned zero.
    """
    soils = [  # K (m/s), psi (m), dtheta (-)
        (5.8e-5, 0.05, 0.4),  # sand
        (3.7e-6, 0.09, 0.3),  # loam
        (1.0e-6, 0.21, 0.2),  # clay loam
        (1.7e-7, 0.32, 0.1),  # clay
    ]
    for conductivity, suction_head_m, dtheta in soils:
        for front_depth_m in [0.0, 0.001, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
            for duration_s in [60.0, 600.0, 3600.0]:
                cumulative_m = front_depth_m * dtheta
                t_start = (
                    _time_from_cumulative_infiltration(
                        cumulative_m, conductivity, suction_head_m, dtheta
                    )
                    if cumulative_m > 0
                    else 0.0
                )
                expected = (
                    _solve_standard_green_ampt_cumulative_infiltration(
                        t_start + duration_s, conductivity, suction_head_m, dtheta
                    )
                    - cumulative_m
                )
                increment = calculate_green_ampt_infiltration_increment(
                    np.float32(cumulative_m),
                    np.float32(conductivity),
                    np.float32(suction_head_m),
                    np.float32(dtheta),
                    np.float32(duration_s),
                )
                assert increment >= np.float32(conductivity * duration_s) * (1 - 1e-6)
                assert abs(increment - expected) <= 1e-3 * expected + 1e-9, (
                    conductivity,
                    front_depth_m,
                    duration_s,
                    increment,
                    expected,
                )


def test_increments_add_up_to_one_long_step() -> None:
    """Six 10-minute steps give the same infiltration as one hour."""
    conductivity = np.float32(1.0e-6)
    suction_head_m = np.float32(0.21)
    dtheta = np.float32(0.2)
    cumulative = np.float32(0.02)
    one_step = calculate_green_ampt_infiltration_increment(
        cumulative, conductivity, suction_head_m, dtheta, np.float32(3600.0)
    )
    total = np.float32(0.0)
    for _ in range(6):
        increment = calculate_green_ampt_infiltration_increment(
            cumulative + total, conductivity, suction_head_m, dtheta, np.float32(600.0)
        )
        total += increment
    assert abs(total - one_step) < 1e-4 * one_step


def test_increment_decreases_as_front_deepens() -> None:
    """With suction, the increment over a fixed step falls as the soil wets up."""
    increments = [
        calculate_green_ampt_infiltration_increment(
            np.float32(cumulative),
            np.float32(1e-5),
            np.float32(0.5),
            np.float32(0.25),
            np.float32(600.0),
        )
        for cumulative in [0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5]
    ]
    assert np.all(np.diff(increments) < 0.0)


def test_green_ampt_verification_silty_clay_example() -> None:
    """Verify against a solved textbook example for silty clay.

    Source: IIT Guwahati, Lecture 13 "Infiltration".
    Url: https://www.iitg.ac.in/kartha/CE551/Lectures/Lecture13.pdf

    Parameters: K = 0.05 cm/hr, psi = 29.22 cm, delta_theta = 0.2961.
    Cumulative infiltration is built up from 15-minute increments.
    """
    ks_cm_hr = np.float32(0.05)
    psi_cm = np.float32(29.22)
    dtheta = np.float32(0.2961)

    expected_f_cm = [0.4735, 0.6745, 0.8307, 0.9638, 1.082]

    cumulative_cm = np.float32(0.0)
    for expected in expected_f_cm:
        cumulative_cm += calculate_green_ampt_infiltration_increment(
            cumulative_cm, ks_cm_hr, psi_cm, dtheta, np.float32(0.25)
        )
        assert abs(cumulative_cm - expected) / expected < 0.002, (
            cumulative_cm,
            expected,
        )
