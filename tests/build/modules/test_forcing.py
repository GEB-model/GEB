"""Tests for the forcing module functions."""

from math import isclose

import matplotlib.pyplot as plt
import numpy as np

from geb.forcing import get_pressure_correction_factor

from ...testconfig import output_folder


def test_get_pressure_correction_factor() -> None:
    """Test the pressure correction factor function.

    Pressure decreases with elevation, following the barometric formula.
    This test checks that the function returns reasonable values.
    """
    elevation = np.linspace(0, 8848.86, 1000)
    g = 9.80665
    Mo = 0.0289644
    lapse_rate = -0.0065

    pressure_at_sea_level = 101325  # Pa
    pressure = pressure_at_sea_level * get_pressure_correction_factor(
        elevation, g, Mo, lapse_rate
    )
    assert isclose(pressure[-1], 33700, abs_tol=3000)

    fig, ax = plt.subplots()

    ax.plot(pressure, elevation)
    ax.set_xlabel("Pressure (Pa)")
    ax.set_ylabel("Elevation (m)")
    ax.set_ylim(0, 9000)
    ax.set_xlim(0, 105000)

    plt.savefig(output_folder / "pressure_correction_factor.png")
