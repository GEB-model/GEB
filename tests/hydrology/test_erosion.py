"""Tests for the hillslope erosion module of GEB."""

import math

import numpy as np
from geb.hydrology.erosion.hillslope import get_particle_fall_velocity


def test_get_particle_fall_velocity() -> None:
    """Tests the get_particle_fall_velocity function from the hillslope erosion module of GEB."""
    particle_diameter = np.float32(2e-6)
    assert math.isclose(
        2.2526666666666666e-06,
        get_particle_fall_velocity(
            particle_diameter=particle_diameter,
            rho_s=np.float32(2650),
            rho=np.float32(1100),
            eta=np.float32(0.0015),
        ),
        rel_tol=1e-6,
    )
    particle_diameter = np.float32(6e-5)
    assert math.isclose(
        0.0020274,
        get_particle_fall_velocity(
            particle_diameter=particle_diameter,
            rho_s=np.float32(2650),
            rho=np.float32(1100),
            eta=np.float32(0.0015),
        ),
        rel_tol=1e-6,
    )
    particle_diameter = np.float32(2e-4)
    assert math.isclose(
        0.022526666666666667,
        get_particle_fall_velocity(
            particle_diameter=particle_diameter,
            rho_s=np.float32(2650),
            rho=np.float32(1100),
            eta=np.float32(0.0015),
        ),
        rel_tol=1e-6,
    )
