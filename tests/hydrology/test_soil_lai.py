"""Tests for soil radiation logic with LAI."""

import numpy as np

from geb.hydrology.soil import calculate_net_radiation_flux


def test_lai_attenuation() -> None:
    """Test that LAI attenuates incoming radiation and includes canopy emission."""
    # Inputs (all scalar for helper check)
    sw_in = np.float32(500.0)
    lw_in = np.float32(300.0)
    t_soil_c = np.float32(20.0)
    t_air_k = np.float32(293.15)  # 20 C

    # Constants
    SOIL_ALBEDO = np.float32(0.23)
    SOIL_EMISSIVITY = np.float32(0.95)
    STEFAN_BOLTZMANN_CONSTANT = np.float32(5.670374419e-8)
    EXTINCTION = np.float32(0.5)

    # Helper function to compute expected manually
    def compute_expected(lai: float) -> np.float32:
        # Attenuation
        att = np.exp(-EXTINCTION * lai)

        # Shortwave
        abs_sw = (1.0 - SOIL_ALBEDO) * sw_in * att

        # Longwave
        # Transmitted
        lw_trans = lw_in * att
        # Canopy emitted
        lw_canopy = STEFAN_BOLTZMANN_CONSTANT * (t_air_k**4) * (1.0 - att)

        abs_lw = SOIL_EMISSIVITY * (lw_trans + lw_canopy)

        # Outgoing
        outgoing = (
            SOIL_EMISSIVITY * STEFAN_BOLTZMANN_CONSTANT * ((t_soil_c + 273.15) ** 4)
        )

        return abs_sw + abs_lw - outgoing

    # Case 1: LAI = 0
    net_0, d_net_0 = calculate_net_radiation_flux(
        shortwave_radiation_W_per_m2=sw_in,
        longwave_radiation_W_per_m2=lw_in,
        soil_temperature_C=t_soil_c,
        leaf_area_index=np.float32(0.0),
        air_temperature_K=t_air_k,
        soil_emissivity=SOIL_EMISSIVITY,
        soil_albedo=SOIL_ALBEDO,
    )
    expected_0 = compute_expected(0.0)
    np.testing.assert_allclose(net_0, expected_0, rtol=1e-5)

    # Case 2: LAI = 2.0
    net_2, d_net_2 = calculate_net_radiation_flux(
        shortwave_radiation_W_per_m2=sw_in,
        longwave_radiation_W_per_m2=lw_in,
        soil_temperature_C=t_soil_c,
        leaf_area_index=np.float32(2.0),
        air_temperature_K=t_air_k,
        soil_emissivity=SOIL_EMISSIVITY,
        soil_albedo=SOIL_ALBEDO,
    )
    expected_2 = compute_expected(2.0)
    np.testing.assert_allclose(net_2, expected_2, rtol=1e-5)

    # Derivative check
    assert d_net_0 == d_net_2


def test_canopy_warming_effect() -> None:
    """Test that warm canopy increases soil net radiation compared to cold sky."""
    sw_in = np.float32(0.0)
    lw_in_sky = np.float32(200.0)  # Cold sky
    t_soil_c = np.float32(20.0)
    t_air_k = np.float32(303.15)  # 30 C, Hot canopy (~470 W/m2 emitted)

    # No canopy (LAI=0): Soil sees cold sky (200 W/m2)
    net_0, _ = calculate_net_radiation_flux(
        sw_in,
        lw_in_sky,
        t_soil_c,
        np.float32(0.0),
        t_air_k,
        soil_emissivity=np.float32(0.95),
        soil_albedo=np.float32(0.23),
    )

    # Dense canopy (LAI=5): Soil sees hot canopy (~470 W/m2)
    net_5, _ = calculate_net_radiation_flux(
        sw_in,
        lw_in_sky,
        t_soil_c,
        np.float32(5.0),
        t_air_k,
        soil_emissivity=np.float32(0.95),
        soil_albedo=np.float32(0.23),
    )

    # Soil should receive more energy with canopy
    assert net_5 > net_0
