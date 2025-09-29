"""Tests for potential evapotranspiration functions in GEB."""

import math

import numpy as np

from geb.hydrology.potential_evapotranspiration import (
    W_m2_to_MJ_m2_day,
    adjust_wind_speed,
    get_actual_vapour_pressure,
    get_latent_heat_of_vaporization,
    get_net_solar_radiation,
    get_psychrometric_constant,
    get_reference_evapotranspiration,
    get_slope_of_saturation_vapour_pressure_curve,
    get_upwelling_long_wave_radiation,
    get_vapour_pressure,
    get_vapour_pressure_deficit,
)


def test_get_vapour_pressure() -> None:
    """See example 3: https://www.fao.org/4/X0490E/x0490e07.htm."""
    saturated_vapour_pressure = get_vapour_pressure(
        temperature_C=15,
    )
    assert math.isclose(saturated_vapour_pressure, 1.705, abs_tol=1e-2)
    saturated_vapour_pressure = get_vapour_pressure(temperature_C=24.5)
    assert math.isclose(saturated_vapour_pressure, 3.075, abs_tol=1e-2)


def test_get_actual_vapour_pressure() -> None:
    """See example 5: https://www.fao.org/4/X0490E/x0490e07.htm."""
    saturated_vapour_pressure_min = get_vapour_pressure(
        temperature_C=18,
    )
    saturated_vapour_pressure_max = get_vapour_pressure(
        temperature_C=25,
    )
    actual_vapour_pressure_deficit = get_actual_vapour_pressure(
        saturated_vapour_pressure_min=saturated_vapour_pressure_min,
        saturated_vapour_pressure_max=saturated_vapour_pressure_max,
        hurs=(82 + 54) / 2,  # Average relative humidity
    )
    assert math.isclose(actual_vapour_pressure_deficit, 1.78, abs_tol=1e-2)


def test_get_vapour_pressure_deficit() -> None:
    """See example 6: https://www.fao.org/4/X0490E/x0490e07.htm."""
    vapour_pressure_deficit = get_vapour_pressure_deficit(
        saturated_vapour_pressure_min=2.064,
        saturated_vapour_pressure_max=3.168,
        actual_vapour_pressure=1.70,
    )
    assert math.isclose(vapour_pressure_deficit, 0.91, abs_tol=0.1)


def test_W_m2_to_MJ_m2_day() -> None:
    """Test the conversion from W/m^2 to MJ/m^2/day."""
    W_m2 = np.float32(100.0)  # Example value in W/m^2
    MJ_m2_day = W_m2_to_MJ_m2_day(W_m2)

    expected_value = W_m2 * (86400 * 1e-6)  # Convert W/m^2 to MJ/m^2/day
    assert math.isclose(MJ_m2_day, expected_value, rel_tol=1e-6)


def test_get_upwelling_long_wave_radiation() -> None:
    """Test the upwelling long wave radiation calculation."""
    tasmin_C = 19.1  # Minimum temperature in Celsius
    tasmax_C = 25.1  # Maximum temperature in Celsius
    rlus_MJ_m2_day = get_upwelling_long_wave_radiation(tasmin_C, tasmax_C)

    expected_value = (38.8 + 35.8) / 2
    assert math.isclose(rlus_MJ_m2_day, expected_value, rel_tol=1e-2)


def test_get_psychrometric_constant() -> None:
    """Test the psychrometric constant calculation."""
    ps_pascal = np.float32(81800.0)  # Example surface pressure in Pascals
    psychrometric_constant = get_psychrometric_constant(ps_pascal)

    assert math.isclose(psychrometric_constant, 0.054, abs_tol=0.01)


def test_get_net_solar_radiation() -> None:
    """Test the net solar radiation calculation."""
    solar_radiation = np.float32(200.0)  # Example solar radiation in W/m^2
    albedo = np.float32(0.23)  # Example albedo for vegetation

    net_solar_radiation = get_net_solar_radiation(solar_radiation, albedo)

    assert math.isclose(net_solar_radiation, 200 - 46, rel_tol=1e-6)


def test_get_slope_of_saturation_vapour_pressure_curve() -> None:
    """Test the slope of the saturation vapour pressure curve calculation."""
    temperature_C = np.float32(25.0)  # Example temperature in Celsius

    slope = get_slope_of_saturation_vapour_pressure_curve(temperature_C)

    expected_value = (
        np.float32(4098.0)
        * get_vapour_pressure(temperature_C=temperature_C)
        / ((temperature_C + np.float32(237.3)) ** 2)
    )
    assert math.isclose(slope, expected_value, rel_tol=1e-6)


def test_get_latent_heat_of_vaporization() -> None:
    """Test the latent heat of vaporization calculation."""
    temperature_C = np.float32(25.0)  # Example temperature in Celsius

    latent_heat = get_latent_heat_of_vaporization(temperature_C)

    expected_value = 2.501 - 0.002361 * temperature_C
    assert math.isclose(latent_heat, expected_value, rel_tol=1e-6)


def test_adjust_wind_speed() -> None:
    """Test the wind speed adjustment function."""
    wind_speed = np.float32(100.0)  # Example wind speed in m/s
    adjusted_wind_speed = adjust_wind_speed(wind_speed)

    assert math.isclose(adjusted_wind_speed, 74.8, rel_tol=1e-6)


def test_get_reference_evapotranspiration() -> None:
    """Test the reference evapotranspiration calculation."""
    (
        reference_evapotranspiration_land_m_per_day,
        reference_evapotranspiration_water_m_per_day,
    ) = get_reference_evapotranspiration(
        net_radiation_land=13.28,
        net_radiation_water=13.28,
        slope_of_saturated_vapour_pressure_curve=0.122,
        psychrometric_constant=0.066,
        wind_2m=2.078,
        latent_heat_of_vaporarization=1 / 0.408,
        termperature_C=16.9,
        vapour_pressure_deficit=0.589,
    )
    assert math.isclose(reference_evapotranspiration_land_m_per_day, 3.9, rel_tol=1e-2)
