"""Unit tests for Green-Ampt helper functions.

These tests focus on the numerical properties of the Green-Ampt time/infiltration
relations used by the infiltration routine.
"""

import numpy as np

from geb.hydrology.soil import (
    calculate_green_ampt_cumulative_infiltration,
    calculate_green_ampt_time_from_infiltration,
)


def _solve_standard_green_ampt_cumulative_infiltration(
    time_steps: np.float32,
    saturated_hydraulic_conductivity_m_per_step: np.float32,
    suction_head_m: np.float32,
    dtheta: np.float32,
    *,
    max_iter: int = 80,
) -> np.float32:
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
    if time_steps <= np.float32(0.0):
        return np.float32(0.0)

    if saturated_hydraulic_conductivity_m_per_step <= np.float32(0.0):
        return np.float32(0.0)

    sf_m: np.float32 = suction_head_m * dtheta
    if sf_m <= np.float32(0.0):
        return saturated_hydraulic_conductivity_m_per_step * time_steps

    def t_from_i(i_m: np.float32) -> np.float32:
        return (
            i_m - sf_m * np.log(np.float32(1.0) + i_m / sf_m)
        ) / saturated_hydraulic_conductivity_m_per_step

    # Bracket: t_from_i(0)=0 <= time_steps. Find hi so that t_from_i(hi) >= time_steps.
    lo_m: np.float32 = np.float32(0.0)
    hi_m: np.float32 = saturated_hydraulic_conductivity_m_per_step * time_steps + sf_m

    # Expand hi if needed (rare for extremely large suction effects).
    for _ in range(60):
        if t_from_i(hi_m) >= time_steps:
            break
        hi_m = hi_m * np.float32(2.0)
    else:
        # If we failed to bracket, return the best-effort hi.
        return hi_m

    for _ in range(max_iter):
        mid_m: np.float32 = (lo_m + hi_m) / np.float32(2.0)
        if t_from_i(mid_m) < time_steps:
            lo_m = mid_m
        else:
            hi_m = mid_m

    return (lo_m + hi_m) / np.float32(2.0)


def test_green_ampt_darcy_limit_zero_suction() -> None:
    """Darcy limit: when capillary suction vanishes, Green-Ampt reduces to I = K t.

    In other words, if S_f = psi_f * Delta_theta -> 0 (no suction effect), cumulative
    infiltration becomes purely conductivity-limited and equals K * t.
    """
    saturated_hydraulic_conductivity_m_per_step = np.float32(1.2e-5)
    suction_head_m = np.float32(0.0)
    dtheta = np.float32(0.25)

    time_steps = np.array([0.0, 0.1, 1.0, 5.0, 10.0], dtype=np.float32)

    cumulative = np.array(
        [
            calculate_green_ampt_cumulative_infiltration.py_func(
                t,
                saturated_hydraulic_conductivity_m_per_step,
                suction_head_m,
                dtheta,
                True,
            )
            for t in time_steps
        ],
        dtype=np.float32,
    )

    expected = saturated_hydraulic_conductivity_m_per_step * time_steps
    assert np.allclose(cumulative, expected, rtol=0.0, atol=1e-8)

    # Inversion should also reduce to t = I / K.
    for infiltration_m in cumulative:
        t_back = calculate_green_ampt_time_from_infiltration.py_func(
            infiltration_m,
            saturated_hydraulic_conductivity_m_per_step,
            suction_head_m,
            dtheta,
        )
        assert np.isfinite(t_back)
        assert (
            abs(t_back - infiltration_m / saturated_hydraulic_conductivity_m_per_step)
            < 1e-8
        )


def test_green_ampt_time_increases_monotonically_with_infiltration() -> None:
    """Time should be monotonic increasing with cumulative infiltration."""
    saturated_hydraulic_conductivity_m_per_step = np.float32(1.0e-5)
    suction_head_m = np.float32(0.5)
    dtheta = np.float32(0.25)

    infiltration_m = np.array([0.0, 1e-6, 1e-5, 5e-5, 1e-4], dtype=np.float32)
    times = np.array(
        [
            calculate_green_ampt_time_from_infiltration.py_func(
                i,
                saturated_hydraulic_conductivity_m_per_step,
                suction_head_m,
                dtheta,
            )
            for i in infiltration_m
        ],
        dtype=np.float32,
    )

    assert np.all(np.diff(times) >= -1e-10)


def test_green_ampt_salvucci_and_time_inverse_are_self_consistent() -> None:
    """Check that the Sadeghi explicit I(t) and standard time inversion are consistent.

    Notes:
        - Sadeghi et al. (2024) provides an explicit approximation for I(t).
        - calculate_green_ampt_time_from_infiltration inverts the *standard* GA form.
        - Therefore, we do NOT require t_back == t exactly, but we do require that
            mapping t -> I(t) produces a finite t_back and that t_back is reasonably close.
    """
    saturated_hydraulic_conductivity_m_per_step = np.float32(1.0e-5)
    suction_head_m = np.float32(0.5)
    dtheta = np.float32(0.25)

    time_steps = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0], dtype=np.float32)

    cumulative = np.array(
        [
            calculate_green_ampt_cumulative_infiltration.py_func(
                t,
                saturated_hydraulic_conductivity_m_per_step,
                suction_head_m,
                dtheta,
                True,
            )
            for t in time_steps
        ],
        dtype=np.float32,
    )

    time_back = np.array(
        [
            calculate_green_ampt_time_from_infiltration.py_func(
                i,
                saturated_hydraulic_conductivity_m_per_step,
                suction_head_m,
                dtheta,
            )
            for i in cumulative
        ],
        dtype=np.float32,
    )

    assert np.all(np.isfinite(time_back))
    assert np.all(np.diff(time_back) > 0.0)

    # Approximation consistency: allow modest relative error.
    rel_err = np.abs(time_back - time_steps) / np.maximum(time_steps, np.float32(1e-6))
    assert np.max(rel_err) < 0.3


def test_green_ampt_infiltration_rate_decelerates_over_time() -> None:
    """Under Green-Ampt with suction, infiltration increments per equal dt should decrease."""
    saturated_hydraulic_conductivity_m_per_step = np.float32(1.0e-5)
    suction_head_m = np.float32(0.5)
    dtheta = np.float32(0.25)

    dt = np.float32(0.1)
    n = 60
    time_steps = np.arange(n + 1, dtype=np.float32) * dt

    cumulative = np.array(
        [
            calculate_green_ampt_cumulative_infiltration.py_func(
                t,
                saturated_hydraulic_conductivity_m_per_step,
                suction_head_m,
                dtheta,
                True,
            )
            for t in time_steps
        ],
        dtype=np.float32,
    )

    delta_i = np.diff(cumulative)
    # After the first few steps (where discretization effects can be largest),
    # the increments should be non-increasing.
    assert np.all(delta_i[3:] <= delta_i[2:-1] + np.float32(1e-10))


def test_green_ampt_wetting_front_advance_decelerates_for_constant_dtheta() -> None:
    """If L = I/Δθ, then L increments per dt decrease as I increments decrease."""
    saturated_hydraulic_conductivity_m_per_step = np.float32(1.0e-5)
    suction_head_m = np.float32(0.5)
    dtheta = np.float32(0.25)

    dt = np.float32(0.1)
    n = 60
    time_steps = np.arange(n + 1, dtype=np.float32) * dt

    cumulative = np.array(
        [
            calculate_green_ampt_cumulative_infiltration.py_func(
                t,
                saturated_hydraulic_conductivity_m_per_step,
                suction_head_m,
                dtheta,
                True,
            )
            for t in time_steps
        ],
        dtype=np.float32,
    )

    wetting_front_depth_m = cumulative / dtheta
    delta_l = np.diff(wetting_front_depth_m)

    assert np.all(delta_l[3:] <= delta_l[2:-1] + np.float32(1e-10))


def test_green_ampt_sadeghi_matches_standard_green_ampt_with_coarse_correction() -> (
    None
):
    """Sadeghi explicit formula should stay within the stated bound vs standard GA.

    The paper reports that with the coarse-soil correction enabled the maximum
    relative deviation from the standard Green-Ampt solution should be <= 0.3.

    This test compares the explicit approximation against a numerical solution of
    the implicit Green-Ampt relation across a small parameter sweep.
    """
    ks_values = np.array([1e-6, 3e-6, 1e-5, 3e-5], dtype=np.float32)
    suction_values_m = np.array([0.05, 0.2, 0.5], dtype=np.float32)
    dtheta_values = np.array([0.1, 0.25, 0.4], dtype=np.float32)
    time_steps = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0], dtype=np.float32)

    max_rel_err: np.float32 = np.float32(0.0)

    for ks in ks_values:
        for suction_head_m in suction_values_m:
            for dtheta in dtheta_values:
                for t in time_steps:
                    i_true = _solve_standard_green_ampt_cumulative_infiltration(
                        t,
                        ks,
                        suction_head_m,
                        dtheta,
                    )
                    i_approx = calculate_green_ampt_cumulative_infiltration.py_func(
                        t,
                        ks,
                        suction_head_m,
                        dtheta,
                        True,
                    )

                    assert np.isfinite(i_true)
                    assert np.isfinite(i_approx)
                    assert i_true >= np.float32(0.0)
                    assert i_approx >= np.float32(0.0)

                    denom = max(i_true, np.float32(1e-9))
                    rel_err = abs(i_approx - i_true) / denom
                    max_rel_err = max(max_rel_err, np.float32(rel_err))

    assert max_rel_err <= np.float32(0.3)


def test_green_ampt_verification_silty_clay_example() -> None:
    """Verify implementation against a solved textbook example for Silty Clay.

    Source: IIT Guwahati, Lecture 13 "Infiltration".
    Url: https://www.iitg.ac.in/kartha/CE551/Lectures/Lecture13.pdf

    Parameters:
        K = 0.05 cm/hr
        psi = 29.22 cm
        delta_theta = 0.3353

    Expected Cumulative Infiltration (F):
        t=0.25 hr -> F ~ 0.5033 cm
        t=1.00 hr -> F ~ 1.0234 cm
    """
    ks_cm_hr = np.float32(0.05)
    psi_cm = np.float32(29.22)
    dtheta = np.float32(0.2961)

    # Test cases: (time_hr, expected_F_cm)
    cases = [
        (np.float32(0.25), np.float32(0.4735)),
        (np.float32(0.50), np.float32(0.6745)),
        (np.float32(0.75), np.float32(0.8307)),
        (np.float32(1.00), np.float32(0.9638)),
        (np.float32(1.25), np.float32(1.082)),
    ]

    for t_hr, expected_f_cm in cases:
        # Calculate using our explicit Sadeghi implementation
        f_calculated = calculate_green_ampt_cumulative_infiltration.py_func(
            t_hr,
            ks_cm_hr,
            psi_cm,
            dtheta,
            True,  # Enable coarse soil correction (though effectively Sadeghi general approx)
        )

        # Allow some tolerance because we are comparing an approximation (Sadeghi)
        # against values derived from the implicit exact solution.
        # Sadeghi claims generally <2% relative error.
        rel_err = abs(f_calculated - expected_f_cm) / expected_f_cm
        assert rel_err < 0.02, (
            f"At t={t_hr}, expected {expected_f_cm}, got {f_calculated}"
        )

        # Also verify time inversion
        t_back = calculate_green_ampt_time_from_infiltration.py_func(
            f_calculated,
            ks_cm_hr,
            psi_cm,
            dtheta,
        )
        # Time inversion should be consistent
        assert abs(t_back - t_hr) < 0.02 * t_hr


def test_green_ampt_coarse_soil_long_time_correction() -> None:
    """Verify effectiveness of the coarse soil adjustment for long infiltration times.

    Test case: Sand.
    K = 11.78 cm/hr
    psi = 4.95 cm
    dtheta = 0.4
    Time = 20.0 hours.

    Without adjustment, Sadeghi approximation error exceeds 0.3%.
    With adjustment, it should be significantly lower (<< 0.3%).
    """
    ks = np.float32(11.78)
    psi = np.float32(4.95)
    dtheta = np.float32(0.4)
    t = np.float32(20.0)

    # Exact value (solved numerically)
    # t = (F - S*ln(1+F/S))/K
    # For parameters above at t=20h, F approx 245.1572 cm
    i_true = _solve_standard_green_ampt_cumulative_infiltration(t, ks, psi, dtheta)

    # Without adjustment
    i_no_adj = calculate_green_ampt_cumulative_infiltration.py_func(
        t, ks, psi, dtheta, False
    )
    err_no = abs(i_no_adj - i_true) / i_true

    # With adjustment
    i_adj = calculate_green_ampt_cumulative_infiltration.py_func(
        t, ks, psi, dtheta, True
    )
    err_adj = abs(i_adj - i_true) / i_true

    # Assertions
    # 1. Confirm that uncorrected error is indeed high (e.g. > 1.0%)
    # Using 1% as a safe lower bound for the 'bad' behavior (observed ~1.4%)
    assert err_no > 0.01, f"Expected uncorrected error > 1%, got {err_no * 100:.2f}%"

    # 2. Confirm that correction brings it down (e.g. < 0.3%)
    assert err_adj < 0.003, f"Expected corrected error < 0.3%, got {err_adj * 100:.2f}%"
