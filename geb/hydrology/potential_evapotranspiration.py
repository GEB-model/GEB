"""Functions to calculate potential evapotranspiration based on the Penman-Monteith equation."""

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True, inline="always")
def get_vapour_pressure(
    temperature_C: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the vapour pressure based on temperature.

    Args:
        temperature_C: Temperature in Celsius.

    Returns:
        Saturated vapour pressure in kPa.
    """
    return np.float32(0.6108) * np.exp(
        (np.float32(17.27) * temperature_C) / (temperature_C + np.float32(237.3))
    )


@njit(cache=True, inline="always")
def get_vapour_pressure_deficit(
    saturated_vapour_pressure_kPa: npt.NDArray[np.float32],
    actual_vapour_pressure_kPa: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the vapour pressure deficit.

    Args:
        saturated_vapour_pressure_kPa: Saturated vapour pressure (kPa).
        actual_vapour_pressure_kPa: Actual vapour pressure (kPa).

    Returns:
        Vapour pressure deficit in kPa.
    """
    return np.maximum(
        saturated_vapour_pressure_kPa - actual_vapour_pressure_kPa,
        np.float32(0.0),
    )


@njit(cache=True, inline="always")
def get_psychrometric_constant(
    ps_pa: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the psychrometric constant.

    Args:
        ps_pa: Surface pressure in Pascals.

    Returns:
        Psychrometric constant in kPa/°C.
    """
    return np.float32(0.665e-3) * np.float32(0.001) * ps_pa  # Convert Pa to kPa


@njit(cache=True, inline="always")
def get_upwelling_long_wave_radiation(
    tas_C: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the upwelling long wave radiation based on temperature.

    Args:
        tas_C: Temperature in Celsius.

    Returns:
        Upwelling long wave radiation in MJ/m^2/hour.
    """
    return np.float32(2.043e-10) * ((tas_C + np.float32(273.16)) ** 4)


@njit(cache=True, inline="always")
def W_per_m2_to_MJ_per_m2_per_hour(
    solar_radiation_W_per_m2: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Convert solar radiation from W/m^2 to MJ/m^2/hour.

    Args:
        solar_radiation_W_per_m2: Solar radiation in W/m^2.

    Returns:
        Solar radiation in MJ/m^2/hour.
    """
    return solar_radiation_W_per_m2 * (np.float32(3600) * np.float32(1e-6))


@njit(cache=True, inline="always")
def get_net_solar_radiation(
    solar_radiation_MJ_per_m2_per_hour: npt.NDArray[np.float32], albedo: np.float32
) -> npt.NDArray[np.float32]:
    """Calculate net solar radiation based on incoming solar radiation and albedo.

    Args:
        solar_radiation_MJ_per_m2_per_hour: Incoming solar radiation (MJ/m^2/hour).
        albedo: Albedo of the surface (fraction of reflected solar radiation).

    Returns:
        Net solar radiation in MJ/m²/timestep.
    """
    return (np.float32(1) - albedo) * solar_radiation_MJ_per_m2_per_hour


@njit(cache=True, inline="always")
def get_slope_of_saturation_vapour_pressure_curve(
    temperature_C: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the slope of the saturation vapour pressure curve.

    Args:
        temperature_C: Temperature in Celsius.

    Returns:
        Slope of the saturation vapour pressure curve in kPa/°C.
    """
    return (
        np.float32(4098.0)
        * get_vapour_pressure(temperature_C=temperature_C)
        / ((temperature_C + np.float32(237.3)) ** 2)
    )


@njit(cache=True, inline="always")
def adjust_wind_speed(
    wind_speed_10m_m_per_s: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Adjust wind speed to surface level.

    Args:
        wind_speed_10m_m_per_s: Wind speed at 10 m height in m/s.

    Returns:
        Adjusted wind speed at 2 m height in m/s.
    """
    return wind_speed_10m_m_per_s * np.float32(0.748)


@njit(cache=True, inline="always")
def penman_monteith(
    net_radiation_land_MJ_per_m2_per_hour: npt.NDArray[np.float32],
    net_radiation_water_MJ_per_m2_per_hour: npt.NDArray[np.float32],
    soil_heat_flux_MJ_per_m2_per_hour: npt.NDArray[np.float32],
    slope_of_saturated_vapour_pressure_curve_kPa_per_C: npt.NDArray[np.float32],
    psychrometric_constant_kPa_per_C: npt.NDArray[np.float32],
    wind_2m_m_per_s: npt.NDArray[np.float32],
    temperature_C: npt.NDArray[np.float32],
    vapour_pressure_deficit_kPa: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Combine all terms of the Penman-Monteith equation to calculate reference evapotranspiration.

    Args:
        net_radiation_land_MJ_per_m2_per_hour: Net radiation for land in MJ/m^2/hour.
        net_radiation_water_MJ_per_m2_per_hour: Net radiation for water in MJ/m^2/hour.
        soil_heat_flux_MJ_per_m2_per_hour: Soil heat flux in MJ/m^2/hour.
        slope_of_saturated_vapour_pressure_curve_kPa_per_C: Slope of the saturation vapour pressure curve in kPa/°C.
        psychrometric_constant_kPa_per_C: Psychrometric constant in kPa/°C.
        wind_2m_m_per_s: Wind speed at 2 m height in m/s.
        temperature_C: Temperature in Celsius.
        vapour_pressure_deficit_kPa: Vapour pressure deficit in kPa.

    Returns:
        reference_evapotranspiration_land: Reference evapotranspiration for land in mm/hour.
        reference_evapotranspiration_water: Reference evapotranspiration for water in mm/hour.
    """
    denominator = (
        slope_of_saturated_vapour_pressure_curve_kPa_per_C
        + psychrometric_constant_kPa_per_C
        * (np.float32(1) + np.float32(0.34) * wind_2m_m_per_s)
    )

    common_energy_factor = (
        np.float32(0.408)
        * slope_of_saturated_vapour_pressure_curve_kPa_per_C
        / denominator
    )

    energy_term_land = (
        net_radiation_land_MJ_per_m2_per_hour - soil_heat_flux_MJ_per_m2_per_hour
    ) * common_energy_factor
    energy_term_water = (
        net_radiation_water_MJ_per_m2_per_hour - soil_heat_flux_MJ_per_m2_per_hour
    ) * common_energy_factor

    aerodynamic_term = (
        psychrometric_constant_kPa_per_C
        * np.float32(37)
        / (temperature_C + np.float32(273.16))
        * wind_2m_m_per_s
        * vapour_pressure_deficit_kPa
    ) / denominator
    return energy_term_land + aerodynamic_term, energy_term_water + aerodynamic_term


@njit(cache=False, parallel=True)
def potential_evapotranspiration(
    tas_K: npt.NDArray[np.float32],
    dewpoint_tas_K: npt.NDArray[np.float32],
    ps_pa: npt.NDArray[np.float32],
    rlds_W_per_m2: npt.NDArray[np.float32],
    rsds_W_per_m2: npt.NDArray[np.float32],
    wind_u10m_m_per_s: npt.NDArray[np.float32],
    wind_v10m_m_per_s: npt.NDArray[np.float32],
    albedo_canopy: np.float32 = np.float32(0.13),
    albedo_water: np.float32 = np.float32(0.05),
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Calculate potential evapotranspiration based on Penman-Monteith equation.

    Penman-Montheith equation:

        ET0 = (0.408 * (Rn - G) + γ * (37 / (T + 273)) * u2 * (es - ea)) / (Δ + γ * (1 + 0.34 * u2))

    where:

        ET0   reference evapotranspiration [mm dt-1],
        Rn    net radiation at the crop surface [MJ m-2 dt-1],
        G     soil heat flux density [MJ m-2 dt-1],
        T     mean daily air temperature at 2 m height [°C],
        es    saturation vapour pressure [kPa],
        ea    actual vapour pressure [kPa],
        es - ea saturation vapour pressure deficit [kPa],
        Δ     slope of the vapour pressure curve [kPa °C-1],
        γ     psychrometric constant [kPa °C-1],
        u2    wind speed at 2 m height [m s-1]

    Note:
        TODO: Add soil heat flux density (G) term. Currently assumed to be 0.

    Args:
        tas_K: average air temperature in Kelvin.
        dewpoint_tas_K: dew point temperature in Kelvin.
        ps_pa: surface pressure in Pascals.
        rlds_W_per_m2: long wave downward surface radiation fluxes in W/m^2.
        rsds_W_per_m2: short wave downward surface radiation fluxes in W/m^2.
        wind_u10m_m_per_s: wind speed at 10 m height in m/s (u component).
        wind_v10m_m_per_s: wind speed at 10 m height in m/s (v component).
        albedo_canopy: albedo of vegetation canopy (default = 0.13).
        albedo_water: albedo of water surface (default = 0.05).

    Returns:
        reference_evapotranspiration_land_m_per_dt: reference evapotranspiration for land in m/dt.
        reference_evapotranspiration_water_m_per_dt: reference evapotranspiration for water in m/dt.
        net_radiation_land_MJ_per_m2_per_dt: net radiation for land in MJ/m^2/dt.
    """
    tas_C: npt.NDArray[np.float32] = tas_K - np.float32(273.15)
    dewpoint_tas_C: npt.NDArray[np.float32] = dewpoint_tas_K - np.float32(273.15)

    actual_vapour_pressure_kPa: npt.NDArray[np.float32] = get_vapour_pressure(
        temperature_C=dewpoint_tas_C
    )
    saturated_vapour_pressure_kPa: npt.NDArray[np.float32] = get_vapour_pressure(
        temperature_C=tas_C
    )

    vapour_pressure_deficit_kPa: npt.NDArray[np.float32] = get_vapour_pressure_deficit(
        saturated_vapour_pressure_kPa=saturated_vapour_pressure_kPa,
        actual_vapour_pressure_kPa=actual_vapour_pressure_kPa,
    )

    psychrometric_constant_kPa_per_C: npt.NDArray[np.float32] = (
        get_psychrometric_constant(ps_pa=ps_pa)
    )

    rlus_MJ_per_m2_per_hour: npt.NDArray[np.float32] = (
        get_upwelling_long_wave_radiation(tas_C)
    )
    rlds_MJ_per_m2_per_hour: npt.NDArray[np.float32] = W_per_m2_to_MJ_per_m2_per_hour(
        rlds_W_per_m2
    )

    net_longwave_radation_MJ_per_m2_per_hour: npt.NDArray[np.float32] = (
        rlus_MJ_per_m2_per_hour - rlds_MJ_per_m2_per_hour
    )

    solar_radiation_MJ_per_m2_per_hour: npt.NDArray[np.float32] = (
        W_per_m2_to_MJ_per_m2_per_hour(rsds_W_per_m2)
    )

    net_solar_radiation_land_MJ_per_m2_per_hour: npt.NDArray[np.float32] = (
        get_net_solar_radiation(solar_radiation_MJ_per_m2_per_hour, albedo_canopy)
    )

    net_radiation_land_MJ_per_m2_per_hour: npt.NDArray[np.float32] = (
        net_solar_radiation_land_MJ_per_m2_per_hour
        - net_longwave_radation_MJ_per_m2_per_hour
    )

    net_solar_radiation_water_MJ_per_m2_per_hour: npt.NDArray[np.float32] = (
        get_net_solar_radiation(solar_radiation_MJ_per_m2_per_hour, albedo_water)
    )
    net_radiation_water_MJ_per_m2_per_hour: npt.NDArray[np.float32] = (
        net_solar_radiation_water_MJ_per_m2_per_hour
        - net_longwave_radation_MJ_per_m2_per_hour
    )

    slope_of_saturated_vapour_pressure_curve_kPa_per_C: npt.NDArray[np.float32] = (
        get_slope_of_saturation_vapour_pressure_curve(temperature_C=tas_C)
    )

    wind_10m_m_per_s: npt.NDArray[np.float32] = np.sqrt(
        wind_u10m_m_per_s**2 + wind_v10m_m_per_s**2
    )
    wind_2m_m_per_s: npt.NDArray[np.float32] = adjust_wind_speed(wind_10m_m_per_s)

    soil_heat_flux_MJ_per_m2_per_hour = np.zeros_like(
        net_radiation_land_MJ_per_m2_per_hour
    )

    (
        reference_evapotranspiration_land_mm_per_hour,
        reference_evapotranspiration_water_mm_per_hour,
    ) = penman_monteith(
        net_radiation_land_MJ_per_m2_per_hour=net_radiation_land_MJ_per_m2_per_hour,
        net_radiation_water_MJ_per_m2_per_hour=net_radiation_water_MJ_per_m2_per_hour,
        soil_heat_flux_MJ_per_m2_per_hour=soil_heat_flux_MJ_per_m2_per_hour,
        slope_of_saturated_vapour_pressure_curve_kPa_per_C=slope_of_saturated_vapour_pressure_curve_kPa_per_C,
        psychrometric_constant_kPa_per_C=psychrometric_constant_kPa_per_C,
        wind_2m_m_per_s=wind_2m_m_per_s,
        temperature_C=tas_C,
        vapour_pressure_deficit_kPa=vapour_pressure_deficit_kPa,
    )

    return (
        reference_evapotranspiration_land_mm_per_hour / np.float32(1000),
        reference_evapotranspiration_water_mm_per_hour / np.float32(1000),
        net_radiation_land_MJ_per_m2_per_hour,
    )
