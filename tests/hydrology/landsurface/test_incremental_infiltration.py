# Explicit Returns/Raises sections follow the supplied project instructions.
# ruff: noqa: DOC202, DOC502
"""Independent regression checks for the incremental Green-Ampt solver."""

import math

import pytest
from scipy.optimize import brentq

from geb.hydrology.landsurface.incremental_infiltration import (
    calculate_green_ampt_increment_m,
)


def test_limits_and_invalid_inputs() -> None:
    """Check limiting cases and reject nonphysical inputs.

    Returns:
        None.

    Raises:
        AssertionError: If a physical limit or input validation is violated.
    """
    assert calculate_green_ampt_increment_m(0.5, 1e-6, 0.5, 0.2, 0) == 0
    assert calculate_green_ampt_increment_m(0.5, 0, 0.5, 0.2, 600) == 0
    assert calculate_green_ampt_increment_m(0.5, 1e-6, 0, 0.2, 600) == pytest.approx(
        0.0006
    )
    for position in range(5):
        for invalid in [-1.0, float("nan"), float("inf")]:
            values: list[float] = [0.5, 1e-6, 0.5, 0.2, 600]
            values[position] = invalid
            with pytest.raises(ValueError):
                calculate_green_ampt_increment_m(*values)


def test_independent_reference_and_timestep_subdivision() -> None:
    """Compare with a separate bracketed root solver and a subdivided interval.

    Returns:
        None.

    Raises:
        AssertionError: If the integrated equation or subdivision consistency fails.
    """
    for cumulative in [0.0, 1e-8, 0.001, 0.1, 0.5, 2.0]:
        for capillary in [1e-6, 0.01, 0.1, 1.0]:
            for conductivity in [1e-9, 1e-6, 1e-4]:
                for duration in [1.0, 600.0, 3600.0]:
                    gravity: float = conductivity * duration
                    upper: float = (
                        gravity + capillary + math.sqrt(2 * capillary * gravity)
                    )
                    reference: float = brentq(
                        lambda depth: (
                            depth
                            - capillary * math.log1p(depth / (cumulative + capillary))
                            - gravity
                        ),
                        0,
                        upper,
                        xtol=1e-15,
                    )
                    result: float = float(
                        calculate_green_ampt_increment_m(
                            cumulative, conductivity, capillary, 1.0, duration
                        )
                    )
                    assert result == pytest.approx(reference, rel=3e-6, abs=1e-12)
                    assert result >= gravity * (1 - 1e-6)
                    first: float = float(
                        calculate_green_ampt_increment_m(
                            cumulative, conductivity, capillary, 1.0, duration / 2
                        )
                    )
                    second: float = float(
                        calculate_green_ampt_increment_m(
                            cumulative + first,
                            conductivity,
                            capillary,
                            1.0,
                            duration / 2,
                        )
                    )
                    assert first + second == pytest.approx(result, rel=3e-6, abs=1e-12)
