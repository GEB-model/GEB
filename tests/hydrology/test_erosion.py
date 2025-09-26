"""Tests for the hillslope erosion module of GEB."""

from geb.hydrology.erosion.hillslope import get_particle_fall_velocity


def test_get_particle_fall_velocity() -> None:
    """Tests the get_particle_fall_velocity function from the hillslope erosion module of GEB."""
    particle_diameter = 2e-6
    assert 2.2526666666666666e-06 == get_particle_fall_velocity(
        particle_diameter=particle_diameter, rho_s=2650, rho=1100, eta=0.0015
    )
    particle_diameter = 6e-5
    assert 0.0020274 == get_particle_fall_velocity(
        particle_diameter=particle_diameter, rho_s=2650, rho=1100, eta=0.0015
    )
    particle_diameter = 2e-4
    assert 0.022526666666666667 == get_particle_fall_velocity(
        particle_diameter=particle_diameter, rho_s=2650, rho=1100, eta=0.0015
    )
