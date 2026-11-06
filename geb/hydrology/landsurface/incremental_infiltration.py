"""Numerically consistent incremental Green–Ampt infiltration."""

import numpy as np
from numba import njit


@njit(cache=True, inline="never")
def calculate_green_ampt_increment_m(
    cumulative_infiltration_m: float | np.float32,
    conductivity_m_per_s: float | np.float32,
    suction_head_m: float | np.float32,
    moisture_deficit: float | np.float32,
    timestep_s: float | np.float32,
) -> np.float32:
    """Advance infiltration with one consistent integrated Green–Ampt equation.

    Notes:
        Solves dF - A log(1 + dF/(F+A)) = K dt in double precision, with
        A = suction head times moisture deficit. Safeguarded Newton iterations
        stay inside an analytical bracket. The returned depth is single precision
        to match the surrounding soil model; no water is added to storage here.

    Args:
        cumulative_infiltration_m: Nonnegative cumulative infiltration (meters).
        conductivity_m_per_s: Nonnegative saturated conductivity (meters/second).
        suction_head_m: Nonnegative capillary suction head (meters).
        moisture_deficit: Nonnegative volumetric moisture deficit (dimensionless).
        timestep_s: Nonnegative integration duration (seconds).

    Returns:
        Potential infiltration increment during the interval (meters).

    Raises:
        ValueError: If inputs are negative or nonfinite.
        RuntimeError: If the bounded solve cannot converge.
    """
    cumulative_m: np.float64 = np.float64(cumulative_infiltration_m)
    conductivity: np.float64 = np.float64(conductivity_m_per_s)
    suction_m: np.float64 = np.float64(suction_head_m)
    deficit: np.float64 = np.float64(moisture_deficit)
    duration_s: np.float64 = np.float64(timestep_s)
    if (
        not np.isfinite(cumulative_m)
        or cumulative_m < 0.0
        or not np.isfinite(conductivity)
        or conductivity < 0.0
        or not np.isfinite(suction_m)
        or suction_m < 0.0
        or not np.isfinite(deficit)
        or deficit < 0.0
        or not np.isfinite(duration_s)
        or duration_s < 0.0
    ):
        raise ValueError("Green–Ampt inputs must be finite and nonnegative.")
    gravity_m: np.float64 = conductivity * duration_s
    capillary_m: np.float64 = suction_m * deficit
    if gravity_m == 0.0 or capillary_m == 0.0:
        return np.float32(gravity_m)
    denominator_m: np.float64 = cumulative_m + capillary_m
    lower_m: np.float64 = gravity_m
    # At F=0, exp(t)>=1+t+t²/2 proves this upper bound; F>0 only lowers the root.
    upper_m: np.float64 = gravity_m + np.sqrt(2.0 * capillary_m * gravity_m)
    trial_m: np.float64 = upper_m
    tolerance_m: np.float64 = max(np.float64(1e-16), gravity_m * np.float64(1e-11))
    for iteration in range(80):
        ratio: np.float64 = trial_m / denominator_m
        log_remainder: np.float64
        if ratio < 1e-3:
            # Avoid cancellation of x-A*log1p(x/(F+A)) near a newly formed wetting front.
            log_remainder = (
                ratio
                * ratio
                * (
                    0.5
                    + ratio
                    * (-1.0 / 3.0 + ratio * (0.25 + ratio * (-0.2 + ratio / 6.0)))
                )
            )
        else:
            log_remainder = ratio - np.log1p(ratio)
        residual_m: np.float64 = (
            trial_m * cumulative_m / denominator_m
            + capillary_m * log_remainder
            - gravity_m
        )
        if abs(residual_m) <= tolerance_m:
            return np.float32(trial_m)
        if residual_m > 0.0:
            upper_m = trial_m
        else:
            lower_m = trial_m
        if upper_m - lower_m <= max(np.float64(1e-15), upper_m * np.float64(1e-12)):
            return np.float32((lower_m + upper_m) * 0.5)
        derivative: np.float64 = (cumulative_m + trial_m) / (denominator_m + trial_m)
        next_m: np.float64 = trial_m - residual_m / derivative
        if iteration >= 20 or not (lower_m < next_m < upper_m):
            next_m = (lower_m + upper_m) * 0.5
        trial_m = next_m
    raise RuntimeError("Consistent Green–Ampt increment failed to converge.")
