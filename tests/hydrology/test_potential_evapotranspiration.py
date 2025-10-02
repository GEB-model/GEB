"""Tests for potential evapotranspiration functions in GEB."""

import math

import numpy as np

from geb.hydrology.potential_evapotranspiration import (
    W_per_m2_to_MJ_per_m2_per_hour,
    adjust_wind_speed,
    get_net_solar_radiation,
    get_psychrometric_constant,
    get_slope_of_saturation_vapour_pressure_curve,
    get_upwelling_long_wave_radiation,
    get_vapour_pressure,
    get_vapour_pressure_deficit,
    penman_monteith,
)


def test_get_vapour_pressure() -> None:
    """See example 3: https://www.fao.org/4/X0490E/x0490e07.htm."""
    saturated_vapour_pressure = get_vapour_pressure(
        temperature_C=15,
    )
    assert math.isclose(saturated_vapour_pressure, 1.705, abs_tol=1e-2)
    saturated_vapour_pressure = get_vapour_pressure(temperature_C=24.5)
    assert math.isclose(saturated_vapour_pressure, 3.075, abs_tol=1e-2)


def test_get_vapour_pressure_deficit() -> None:
    """See example 18: https://www.fao.org/4/x0490e/x0490e08.htm."""
    vapour_pressure_deficit = get_vapour_pressure_deficit(
        saturated_vapour_pressure=1.997,
        actual_vapour_pressure=1.409,
    )
    assert math.isclose(a=vapour_pressure_deficit, b=0.589, abs_tol=0.1)


def test_W_per_m2_to_MJ_per_m2_per_hour() -> None:
    """Test the conversion from W/m^2 to MJ/m^2/hour."""
    W_per_m2 = np.float32(100.0)  # Example value in W/m^2
    MJ_per_m2_per_hour = W_per_m2_to_MJ_per_m2_per_hour(W_per_m2)

    expected_value = W_per_m2 * (3600 * 1e-6)  # Convert W/m^2 to MJ/m^2/hour
    assert math.isclose(MJ_per_m2_per_hour, expected_value, rel_tol=1e-6)


def test_get_upwelling_long_wave_radiation() -> None:
    """Test the upwelling long wave radiation calculation.

    See example 18: https://www.fao.org/4/x0490e/x0490e08.htm.
    """
    tas_C = 38  # Temperature in Celsius
    rlus_MJ_m2_per_hour = get_upwelling_long_wave_radiation(tas_C)

    expected_value = 1.915  # Expected value in MJ/m^2/hour
    assert math.isclose(rlus_MJ_m2_per_hour, expected_value, rel_tol=1e-2)


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


def test_adjust_wind_speed() -> None:
    """Test the wind speed adjustment function."""
    wind_speed = np.float32(100.0)  # Example wind speed in m/s
    adjusted_wind_speed = adjust_wind_speed(wind_speed)

    assert math.isclose(adjusted_wind_speed, 74.8, rel_tol=1e-6)


def test_penman_monteith_day() -> None:
    """Test the reference evapotranspiration calculation during the day.

    See example 19: https://www.fao.org/4/x0490e/x0490e08.htm
    """
    (
        reference_evapotranspiration_land_m_per_day,
        reference_evapotranspiration_water_m_per_day,
    ) = penman_monteith(
        net_radiation_land=1.749,
        net_radiation_water=1.749,
        soil_heat_flux=0.175,
        slope_of_saturated_vapour_pressure_curve=0.358,
        psychrometric_constant=0.0673,
        wind_2m=3.3,
        temperature_C=38,
        vapour_pressure_deficit=3.180,
    )
    assert math.isclose(
        a=reference_evapotranspiration_land_m_per_day, b=0.63, rel_tol=1e-2
    )


def test_penman_monteith_night() -> None:
    """Test the reference evapotranspiration calculation at night.

    See example 19: https://www.fao.org/4/x0490e/x0490e08.htm
    """
    (
        reference_evapotranspiration_land_m_per_day,
        reference_evapotranspiration_water_m_per_day,
    ) = penman_monteith(
        net_radiation_land=-0.100,
        net_radiation_water=-0.100,
        soil_heat_flux=-0.050,
        slope_of_saturated_vapour_pressure_curve=0.220,
        psychrometric_constant=0.0673,
        wind_2m=1.9,
        temperature_C=28,
        vapour_pressure_deficit=0.378,
    )
    assert math.isclose(
        a=reference_evapotranspiration_land_m_per_day, b=0.00, abs_tol=1e-2
    )
