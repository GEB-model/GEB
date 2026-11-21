"""Potential evapotranspiration: Penman-Monteith and the FAO-56 crop coefficients."""

import math
from typing import NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt
from numba import njit

from geb.geb_types import ArrayFloat32, TwoDArrayFloat32
from geb.numba_config import NUMBA_CACHE

from ..landcovers import (
    FOREST,
    GRASSLAND_LIKE,
    NON_PADDY_IRRIGATED,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
    SEALED_PAVED_FRACTION,
)

# Physical constants (FAO-56, Annex 3).
STEFAN_BOLTZMANN_W_PER_M2_K4: np.float32 = np.float32(5.670374e-8)
SPECIFIC_HEAT_AIR_J_PER_KG_K: np.float32 = np.float32(1013.0)
LATENT_HEAT_OF_VAPORISATION_J_PER_KG: np.float32 = np.float32(2.45e6)
SPECIFIC_GAS_CONSTANT_DRY_AIR_J_PER_KG_K: np.float32 = np.float32(287.0)
VON_KARMAN_CONSTANT: np.float32 = np.float32(0.41)

# Heights of the meteorological forcing above the ground (ERA5): wind at 10 m, air
# temperature and dew point at 2 m.
WIND_MEASUREMENT_HEIGHT_M: np.float32 = np.float32(10.0)
AIR_MEASUREMENT_HEIGHT_M: np.float32 = np.float32(2.0)
# Grass reference surface of FAO-56: a closed, well-watered grass of 0.12 m with a
# surface resistance of 70 s/m and an albedo of 0.23.
GRASS_HEIGHT_M: np.float32 = np.float32(0.12)
# Lowest wind speed used for the aerodynamic resistance. In calm air buoyancy (free
# convection) still carries heat and vapour away from the surface, which the
# logarithmic wind profile does not describe. FAO-56 (ch. 3) limits the 2 m wind over
# grass to u2 >= 0.5 m/s; the forcing gives the 10 m wind, so the limit is the 10 m
# wind of the logarithmic profile over the grass reference (d = 2/3 h, z0m = 0.123 h)
# with u2 = 0.5 m/s, about 0.67 m/s.
# Ratio of the 2 m to the 10 m wind over the grass reference, about 0.748 (FAO-56
# eq. 47, see adjust_wind_speed_log_profile).
WIND_2M_PER_WIND_10M_OVER_GRASS: np.float32 = np.float32(
    math.log(
        (2.0 - 2.0 / 3.0 * float(GRASS_HEIGHT_M)) / (0.123 * float(GRASS_HEIGHT_M))
    )
    / math.log(
        (float(WIND_MEASUREMENT_HEIGHT_M) - 2.0 / 3.0 * float(GRASS_HEIGHT_M))
        / (0.123 * float(GRASS_HEIGHT_M))
    )
)
MINIMUM_WIND_SPEED_2M_M_PER_S: float = 0.5
MINIMUM_WIND_SPEED_M_PER_S: np.float32 = np.float32(
    MINIMUM_WIND_SPEED_2M_M_PER_S / float(WIND_2M_PER_WIND_10M_OVER_GRASS)
)
GRASS_SURFACE_RESISTANCE_S_PER_M: np.float32 = np.float32(70.0)
GRASS_ALBEDO: np.float32 = np.float32(0.23)
# Longwave emissivity of vegetation and of water; both are close to 1 (0.95-0.99).
VEGETATION_EMISSIVITY: np.float32 = np.float32(0.98)
WATER_EMISSIVITY: np.float32 = np.float32(0.97)
# Open water: albedo at moderate solar elevations and the roughness length of a water
# surface with small waves, of the order of 0.1-1 mm, for momentum as well as for heat
# and vapour.
WATER_ALBEDO: np.float32 = np.float32(0.05)
WATER_ROUGHNESS_LENGTH_M: np.float32 = np.float32(0.001)
# Height and shortwave albedo of a closed temperate forest canopy.
FOREST_CANOPY_HEIGHT_M: np.float32 = np.float32(20.0)
FOREST_ALBEDO: np.float32 = np.float32(0.12)
# Height of field crops at full development, used in the climate adjustment of the
# crop coefficients (get_crop_coefficient_climate_adjustment).
# NOTE: a generic height for all crops, because the crop data have no height per crop.
# FAO-56 Table 12 gives about 1 m for cereals, 2 m for maize and 0.5-0.6 m for potato
# and sugar beet. The adjustment depends weakly on height ((h/3)^0.3: 0.76 for 0.5 m,
# 1.10 for 2 m), so the error is at most a few hundredths in Kcb and Kc_max. Replace
# with a height per crop (FAO-56 Table 12) when the crop data get one.
CROP_HEIGHT_M: np.float32 = np.float32(1.0)


# A scalar or an array of temperatures; the result has the same type.
TemperatureC = TypeVar("TemperatureC", np.float32, npt.NDArray[np.float32])


@njit(cache=NUMBA_CACHE, inline="always")
def get_vapour_pressure(
    temperature_C: TemperatureC,
) -> TemperatureC:
    """Calculate the saturation vapour pressure at a temperature (FAO-56 eq. 11).

    Args:
        temperature_C: Temperature in Celsius, a scalar or an array.

    Returns:
        Saturated vapour pressure in kPa, of the same shape.
    """
    return np.float32(0.6108) * np.exp(
        (np.float32(17.27) * temperature_C) / (temperature_C + np.float32(237.3))
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_vapour_pressure_deficit(
    saturated_vapour_pressure_kPa: np.float32,
    actual_vapour_pressure_kPa: np.float32,
) -> np.float32:
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


@njit(cache=NUMBA_CACHE, inline="always")
def get_psychrometric_constant(
    ps_pa: np.float32,
) -> np.float32:
    """Calculate the psychrometric constant (FAO-56 eq. 8).

    Args:
        ps_pa: Surface pressure in Pascals.

    Returns:
        Psychrometric constant in kPa/°C.
    """
    return np.float32(0.665e-3) * np.float32(0.001) * ps_pa  # Convert Pa to kPa


@njit(cache=NUMBA_CACHE, inline="always")
def get_slope_of_saturation_vapour_pressure_curve(
    temperature_C: np.float32,
    saturated_vapour_pressure_kPa: np.float32,
) -> np.float32:
    """Calculate the slope of the saturation vapour pressure curve (FAO-56 eq. 13).

    Args:
        temperature_C: Temperature in Celsius.
        saturated_vapour_pressure_kPa: Saturation vapour pressure at that
            temperature (kPa), see get_vapour_pressure.

    Returns:
        Slope of the saturation vapour pressure curve in kPa/°C.
    """
    return (
        np.float32(4098.0)
        * saturated_vapour_pressure_kPa
        / ((temperature_C + np.float32(237.3)) ** 2)
    )


@njit(cache=NUMBA_CACHE, inline="always")
def adjust_wind_speed_log_profile(
    wind_measurement_height_m_per_s: np.float32,
    measurement_height_m: np.float32,
    target_height_m: np.float32,
    canopy_height_m: np.float32,
) -> np.float32:
    """
    Translates wind speed from a measurement height to a target height.

    Uses a logarithmic wind profile.

    See: https://www.fao.org/4/x0490e/x0490e06.htm

    displacement height (d) is set to 2/3 of canopy height
    roughness length is set to 0.123 of canopy height based on FAO guidelines for short grass.

    Args:
        wind_measurement_height_m_per_s: Wind speed at the measurement height (m/s).
        measurement_height_m: Height above ground of the wind-speed measurement (m).
        target_height_m: Height above ground to which wind speed is adjusted (m).
        canopy_height_m: Vegetation or surface canopy height used for displacement and roughness (m).

    Returns:
        Adjusted wind speed at the target height.
    """
    # d: Zero-plane displacement height
    # The height above the ground where the wind speed is effectively zero
    # due to the density of the canopy.
    displacement_height_m = np.float32(2 / 3) * canopy_height_m

    # z0: Aerodynamic roughness length
    # A parameter representing the roughness/drag of the surface.
    roughness_length_m = np.float32(0.123) * canopy_height_m

    log_target: np.float32 = np.log(
        (target_height_m - displacement_height_m) / roughness_length_m
    )
    log_measured: np.float32 = np.log(
        (measurement_height_m - displacement_height_m) / roughness_length_m
    )

    return wind_measurement_height_m_per_s * (log_target / log_measured)


def get_aerodynamic_resistance_times_wind(
    wind_height_above_displacement_m: float,
    air_height_above_displacement_m: float,
    roughness_length_momentum_m: float,
    roughness_length_heat_m: float,
) -> np.float32:
    """Aerodynamic resistance times wind speed, ln((zm-d)/z0m) ln((zh-d)/z0h) / k^2 (m).

    For a surface with a fixed geometry the aerodynamic resistance (FAO-56 eq. 4) is
    this constant divided by the wind speed. Computed once per surface, at import.

    Args:
        wind_height_above_displacement_m: Height of the wind above d, zm - d (m).
        air_height_above_displacement_m: Height of the air temperature and humidity
            above d, zh - d (m).
        roughness_length_momentum_m: Roughness length for momentum z0m (m).
        roughness_length_heat_m: Roughness length for heat and vapour z0h (m).

    Returns:
        Aerodynamic resistance times wind speed (m).
    """
    return np.float32(
        math.log(wind_height_above_displacement_m / roughness_length_momentum_m)
        * math.log(air_height_above_displacement_m / roughness_length_heat_m)
        / (float(VON_KARMAN_CONSTANT) ** 2)
    )


def get_canopy_aerodynamic_resistance_times_wind(
    canopy_height_m: float,
    wind_height_above_displacement_m: float,
    air_height_above_displacement_m: float,
) -> np.float32:
    """Aerodynamic resistance times wind speed above a vegetation canopy (m).

    The zero-plane displacement is d = 2/3 h, the roughness length for momentum
    z0m = 0.123 h and for heat and vapour z0h = 0.1 z0m (FAO-56, ch. 2).

    Args:
        canopy_height_m: Canopy height h (m).
        wind_height_above_displacement_m: Height of the wind above d (m).
        air_height_above_displacement_m: Height of the air temperature and humidity
            above d (m).

    Returns:
        Aerodynamic resistance times wind speed (m).
    """
    roughness_length_momentum_m = 0.123 * canopy_height_m
    return get_aerodynamic_resistance_times_wind(
        wind_height_above_displacement_m,
        air_height_above_displacement_m,
        roughness_length_momentum_m,
        0.1 * roughness_length_momentum_m,
    )


# Aerodynamic resistance times wind speed (m) of the surfaces with a fixed geometry;
# r_a = constant / u10 (get_hourly_weather limits u10 to MINIMUM_WIND_SPEED_M_PER_S).
# Grass reference: wind 10 m and air 2 m above the ground, about 278 / u10 s/m, the
# same as the FAO-56 value 208 / u2 with the 2 m wind of the logarithmic profile.
GRASS_AERODYNAMIC_RESISTANCE_TIMES_WIND_M: np.float32 = (
    get_canopy_aerodynamic_resistance_times_wind(
        float(GRASS_HEIGHT_M),
        float(WIND_MEASUREMENT_HEIGHT_M) - 2.0 / 3.0 * float(GRASS_HEIGHT_M),
        float(AIR_MEASUREMENT_HEIGHT_M) - 2.0 / 3.0 * float(GRASS_HEIGHT_M),
    )
)
# Open water: wind 10 m and air 2 m above a smooth water surface.
WATER_AERODYNAMIC_RESISTANCE_TIMES_WIND_M: np.float32 = (
    get_aerodynamic_resistance_times_wind(
        float(WIND_MEASUREMENT_HEIGHT_M),
        float(AIR_MEASUREMENT_HEIGHT_M),
        float(WATER_ROUGHNESS_LENGTH_M),
        float(WATER_ROUGHNESS_LENGTH_M),
    )
)
# Forest: the 10 m wind, air temperature and humidity of the forcing over a forested
# grid cell are valid above the canopy and taken 10 m above the zero-plane
# displacement height. For h = 20 m and u = 3 m/s, r_a is about 10 s/m.
FOREST_AERODYNAMIC_RESISTANCE_TIMES_WIND_M: np.float32 = (
    get_canopy_aerodynamic_resistance_times_wind(
        float(FOREST_CANOPY_HEIGHT_M),
        float(WIND_MEASUREMENT_HEIGHT_M),
        float(WIND_MEASUREMENT_HEIGHT_M),
    )
)


@njit(cache=NUMBA_CACHE, inline="always")
def get_aerodynamic_resistance(
    wind_m_per_s: np.float32,
    wind_height_above_displacement_m: np.float32,
    air_height_above_displacement_m: np.float32,
    roughness_length_momentum_m: np.float32,
    roughness_length_heat_m: np.float32,
) -> np.float32:
    """Aerodynamic resistance for heat and water vapour (FAO-56 eq. 4).

    r_a = ln((zm - d) / z0m) ln((zh - d) / z0h) / (k^2 u_zm), for neutral stability,
    with zm the height of the wind measurement and zh the height of the air
    temperature and humidity measurement. For surfaces whose geometry changes in time
    (e.g. growing crops); fixed surfaces use a constant divided by the wind speed
    (GRASS_AERODYNAMIC_RESISTANCE_TIMES_WIND_M etc.).

    Args:
        wind_m_per_s: Wind speed at height zm (m/s), at least
            MINIMUM_WIND_SPEED_M_PER_S.
        wind_height_above_displacement_m: zm - d (m).
        air_height_above_displacement_m: zh - d (m).
        roughness_length_momentum_m: Roughness length for momentum z0m (m).
        roughness_length_heat_m: Roughness length for heat and vapour z0h (m).

    Returns:
        Aerodynamic resistance (s/m).
    """
    return (
        np.log(wind_height_above_displacement_m / roughness_length_momentum_m)
        * np.log(air_height_above_displacement_m / roughness_length_heat_m)
        / (VON_KARMAN_CONSTANT * VON_KARMAN_CONSTANT * wind_m_per_s)
    )


class HourlyWeather(NamedTuple):
    """Weather terms of one hour, shared by the evaporation of all surfaces.

    Computed once per HRU and hour (get_hourly_weather), so that every surface uses
    the same values and the exponentials and powers are evaluated only once.
    """

    tas_C: np.float32
    """Air temperature (°C)."""
    actual_vapour_pressure_kPa: np.float32
    """Actual vapour pressure, the saturation vapour pressure at the dew point (kPa)."""
    vapour_pressure_deficit_kPa: np.float32
    """Vapour pressure deficit (kPa)."""
    relative_humidity_percent: np.float32
    """Relative humidity, at most 100 (%)."""
    slope_kPa_per_C: np.float32
    """Slope of the saturation vapour pressure curve at air temperature (kPa/°C)."""
    psychrometric_constant_kPa_per_C: np.float32
    """Psychrometric constant (kPa/°C)."""
    volumetric_heat_capacity_air_J_per_m3_K: np.float32
    """Air density times specific heat of air, rho c_p (J/m3/K)."""
    black_body_emission_W_per_m2: np.float32
    """Longwave emission of a black body at air temperature, sigma T^4 (W/m2)."""
    black_body_radiative_conductance_W_per_m2_K: np.float32
    """Change of that emission with temperature, 4 sigma T^3 (W/m2/K)."""
    rsds_W_per_m2: np.float32
    """Downward shortwave radiation (W/m2)."""
    rlds_W_per_m2: np.float32
    """Downward longwave radiation (W/m2)."""
    wind_10m_m_per_s: np.float32
    """Wind speed at 10 m, at least MINIMUM_WIND_SPEED_M_PER_S (m/s), for the
    aerodynamic resistance."""


@njit(cache=NUMBA_CACHE, inline="always")
def get_hourly_weather(
    tas_C: np.float32,
    dewpoint_tas_C: np.float32,
    ps_pa: np.float32,
    rsds_W_per_m2: np.float32,
    rlds_W_per_m2: np.float32,
    wind_10m_m_per_s: np.float32,
) -> HourlyWeather:
    """Weather terms of one hour shared by the evaporation of all surfaces.

    Args:
        tas_C: Air temperature (°C).
        dewpoint_tas_C: Dew point temperature (°C).
        ps_pa: Surface pressure (Pa).
        rsds_W_per_m2: Downward shortwave radiation (W/m2).
        rlds_W_per_m2: Downward longwave radiation (W/m2).
        wind_10m_m_per_s: Wind speed at 10 m (m/s).

    Returns:
        The weather terms, see HourlyWeather.
    """
    saturated_vapour_pressure_kPa: np.float32 = get_vapour_pressure(tas_C)
    actual_vapour_pressure_kPa: np.float32 = get_vapour_pressure(dewpoint_tas_C)
    air_temperature_K: np.float32 = tas_C + np.float32(273.15)
    air_temperature_cubed_K3: np.float32 = (
        air_temperature_K * air_temperature_K * air_temperature_K
    )
    return HourlyWeather(
        tas_C=tas_C,
        actual_vapour_pressure_kPa=actual_vapour_pressure_kPa,
        vapour_pressure_deficit_kPa=get_vapour_pressure_deficit(
            saturated_vapour_pressure_kPa, actual_vapour_pressure_kPa
        ),
        relative_humidity_percent=min(
            np.float32(100.0)
            * actual_vapour_pressure_kPa
            / saturated_vapour_pressure_kPa,
            np.float32(100.0),
        ),
        slope_kPa_per_C=get_slope_of_saturation_vapour_pressure_curve(
            tas_C, saturated_vapour_pressure_kPa
        ),
        psychrometric_constant_kPa_per_C=get_psychrometric_constant(ps_pa),
        volumetric_heat_capacity_air_J_per_m3_K=get_air_density(ps_pa, tas_C)
        * SPECIFIC_HEAT_AIR_J_PER_KG_K,
        black_body_emission_W_per_m2=STEFAN_BOLTZMANN_W_PER_M2_K4
        * air_temperature_cubed_K3
        * air_temperature_K,
        black_body_radiative_conductance_W_per_m2_K=np.float32(4.0)
        * STEFAN_BOLTZMANN_W_PER_M2_K4
        * air_temperature_cubed_K3,
        rsds_W_per_m2=rsds_W_per_m2,
        rlds_W_per_m2=rlds_W_per_m2,
        wind_10m_m_per_s=max(wind_10m_m_per_s, MINIMUM_WIND_SPEED_M_PER_S),
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_air_density(ps_pa: np.float32, tas_C: np.float32) -> np.float32:
    """Density of moist air (FAO-56 Annex 3, eq. 3-5).

    The virtual temperature, which accounts for the lower density of water vapour, is
    approximated as 1.01 (T + 273).

    Args:
        ps_pa: Surface pressure (Pa).
        tas_C: Air temperature (°C).

    Returns:
        Air density (kg/m3).
    """
    return ps_pa / (
        SPECIFIC_GAS_CONSTANT_DRY_AIR_J_PER_KG_K
        * np.float32(1.01)
        * (tas_C + np.float32(273.15))
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_isothermal_net_radiation(
    weather: HourlyWeather,
    albedo: np.float32,
    emissivity: np.float32,
) -> np.float32:
    """Net radiation of a surface at air temperature ("isothermal" net radiation).

    Rn_iso = (1 - albedo) Rs + emissivity (Rl_down - sigma Ta^4). The surface
    absorbs the fraction `emissivity` of the downward longwave radiation and emits
    emissivity sigma T^4. Its real temperature differs from the air temperature; in
    Penman-Monteith that difference is handled by the radiative resistance (see
    penman_monteith), so the surface temperature does not need to be known.

    Args:
        weather: Weather of the hour, see get_hourly_weather.
        albedo: Shortwave albedo of the surface (-).
        emissivity: Longwave emissivity of the surface (-).

    Returns:
        Isothermal net radiation (W/m2), positive towards the surface.
    """
    return (np.float32(1.0) - albedo) * weather.rsds_W_per_m2 + emissivity * (
        weather.rlds_W_per_m2 - weather.black_body_emission_W_per_m2
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_radiative_resistance(
    weather: HourlyWeather, emissivity: np.float32
) -> np.float32:
    """Resistance to the exchange of heat by longwave radiation (s/m).

    A surface warmer than the air emits more longwave radiation than at air
    temperature: d(emissivity sigma T^4)/dT = 4 emissivity sigma T^3 (W/m2/K). Written
    as a resistance with the same units as the aerodynamic resistance, so that it can
    be put in parallel with it, r_R = rho c_p / (4 emissivity sigma Ta^3) (Monteith &
    Unsworth, 2013, ch. 13). About 200 s/m at 20 °C.

    Args:
        weather: Weather of the hour, see get_hourly_weather.
        emissivity: Longwave emissivity of the surface (-).

    Returns:
        Radiative resistance (s/m).
    """
    return weather.volumetric_heat_capacity_air_J_per_m3_K / (
        emissivity * weather.black_body_radiative_conductance_W_per_m2_K
    )


@njit(cache=NUMBA_CACHE, inline="always")
def penman_monteith(
    available_energy_W_per_m2: np.float32,
    vapour_pressure_deficit_kPa: np.float32,
    slope_kPa_per_C: np.float32,
    psychrometric_constant_kPa_per_C: np.float32,
    volumetric_heat_capacity_air_J_per_m3_K: np.float32,
    aerodynamic_resistance_s_per_m: np.float32,
    surface_resistance_s_per_m: np.float32,
    radiative_resistance_s_per_m: np.float32,
) -> np.float32:
    """Evaporation from the Penman-Monteith equation (m/hour, negative for dew).

    The surface loses heat to the air through two paths in parallel: turbulent
    transfer of sensible heat (aerodynamic resistance r_a) and longwave radiation
    (radiative resistance r_R). Water vapour leaves through the stomata or the
    water film (surface resistance r_s) and then the same turbulent path (r_a):

        lambda E = (Delta A + rho c_p D / r_HR) / (Delta + gamma (r_a + r_s) / r_HR)

    with 1 / r_HR = 1 / r_a + 1 / r_R, A the isothermal available energy
    (Rn_iso - G), D the vapour pressure deficit, Delta the slope of the saturation
    vapour pressure curve and gamma the psychrometric constant (Monteith & Unsworth,
    2013, ch. 13). Without the radiative path (r_R infinite) this is FAO-56 eq. 3.

    The resistances (s/m) are the time the air needs to carry heat or vapour over a
    unit of length; a larger value means a weaker exchange:
    - r_a, aerodynamic resistance: turbulent mixing between the surface and the
      measurement height. Small for tall, rough canopies and strong wind (forest,
      about 10 s/m), large for smooth surfaces (grass about 90 s/m at 3 m/s wind,
      water about 140 s/m).
    - r_s, surface resistance: diffusion of vapour out of the stomata of a dry
      canopy (70 s/m for the grass reference). Zero for a wet surface: water on
      leaves, on pavement or open water evaporates directly.
    - r_R, radiative resistance: a surface warmer than the air also loses heat by
      longwave radiation (about 200 s/m). Using it, the net radiation can be
      computed at air temperature, without knowing the surface temperature.

    The weather terms come from get_hourly_weather; get_surface_evaporation
    combines them with the properties of a surface.

    Args:
        available_energy_W_per_m2: Isothermal net radiation minus heat flux into
            the soil or water (W/m2).
        vapour_pressure_deficit_kPa: Vapour pressure deficit of the air (kPa).
        slope_kPa_per_C: Slope of the saturation vapour pressure curve (kPa/°C).
        psychrometric_constant_kPa_per_C: Psychrometric constant (kPa/°C).
        volumetric_heat_capacity_air_J_per_m3_K: Air density times specific heat of
            air (J/m3/K).
        aerodynamic_resistance_s_per_m: Aerodynamic resistance r_a (s/m).
        surface_resistance_s_per_m: Surface resistance r_s (s/m).
        radiative_resistance_s_per_m: Radiative resistance r_R (s/m), infinite to
            leave out the radiative path.

    Returns:
        Evaporation (m/hour). Negative when vapour condenses on the surface (dew).
    """
    heat_conductance_m_per_s: np.float32 = (
        np.float32(1.0) / aerodynamic_resistance_s_per_m
        + np.float32(1.0) / radiative_resistance_s_per_m
    )
    latent_heat_flux_W_per_m2: np.float32 = (
        slope_kPa_per_C * available_energy_W_per_m2
        + volumetric_heat_capacity_air_J_per_m3_K
        * vapour_pressure_deficit_kPa
        * heat_conductance_m_per_s
    ) / (
        slope_kPa_per_C
        + psychrometric_constant_kPa_per_C
        * (aerodynamic_resistance_s_per_m + surface_resistance_s_per_m)
        * heat_conductance_m_per_s
    )
    # W/m2 = J/s/m2 -> kg/m2/hour = mm/hour -> m/hour
    return (
        latent_heat_flux_W_per_m2
        * np.float32(3600.0)
        / (LATENT_HEAT_OF_VAPORISATION_J_PER_KG * np.float32(1000.0))
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_surface_evaporation(
    weather: HourlyWeather,
    albedo: np.float32,
    emissivity: np.float32,
    heat_flux_into_ground_W_per_m2: np.float32,
    aerodynamic_resistance_s_per_m: np.float32,
    surface_resistance_s_per_m: np.float32,
) -> np.float32:
    """Penman-Monteith evaporation of one surface in the weather of the hour (m/hour).

    The generic function for a surface: its albedo and emissivity set the net
    radiation, the heat flux into the ground the available energy, and its
    aerodynamic and surface resistance the transport of vapour.

    Args:
        weather: Weather of the hour, see get_hourly_weather.
        albedo: Shortwave albedo of the surface (-).
        emissivity: Longwave emissivity of the surface (-).
        heat_flux_into_ground_W_per_m2: Heat flux into the soil or water (W/m2).
        aerodynamic_resistance_s_per_m: Aerodynamic resistance r_a (s/m).
        surface_resistance_s_per_m: Surface resistance r_s (s/m).

    Returns:
        Evaporation (m/hour). Negative when vapour condenses on the surface (dew).
    """
    return penman_monteith(
        available_energy_W_per_m2=get_isothermal_net_radiation(
            weather, albedo, emissivity
        )
        - heat_flux_into_ground_W_per_m2,
        vapour_pressure_deficit_kPa=weather.vapour_pressure_deficit_kPa,
        slope_kPa_per_C=weather.slope_kPa_per_C,
        psychrometric_constant_kPa_per_C=weather.psychrometric_constant_kPa_per_C,
        volumetric_heat_capacity_air_J_per_m3_K=weather.volumetric_heat_capacity_air_J_per_m3_K,
        aerodynamic_resistance_s_per_m=aerodynamic_resistance_s_per_m,
        surface_resistance_s_per_m=surface_resistance_s_per_m,
        radiative_resistance_s_per_m=get_radiative_resistance(weather, emissivity),
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_reference_evapotranspiration(
    weather: HourlyWeather,
    soil_heat_flux_W_per_m2: np.float32,
) -> tuple[np.float32, np.float32, np.float32]:
    """Reference evapotranspiration and the evaporation of wet surfaces (m/hour).

    All three rates use penman_monteith with the measured downward longwave radiation
    and differ in their surface:

    - Grass reference (FAO-56): grass of 0.12 m, r_s = 70 s/m, albedo 0.23,
      r_a = 208 / u2. The crop coefficients are relative to this rate, so it uses the
      FAO-56 Penman-Monteith equation (eq. 3, hourly eq. 53): the isothermal net
      radiation without the radiative path (r_R infinite). Given the same net
      radiation and soil heat flux it is within 0.5 % of FAO-56 eq. 53. Different
      from FAO-56, the net longwave radiation uses the downward longwave radiation
      of the forcing (FAO-56 estimates it, eq. 39) and the soil heat flux comes from
      the soil temperature model (FAO-56: 0.1 Rn by day, 0.5 Rn at night). With the
      radiative path the rate would be about 2-12 % lower in daytime hours than the
      FAO-56 reference on which the crop coefficients were derived.
    - Wet short vegetation: the same grass with water on its leaves, r_s = 0. The
      evaporation of intercepted water from grass and crops, and dew. Not scaled
      with crop coefficients, so it keeps the radiative resistance.
    - Open water: a smooth water surface, r_s = 0, albedo 0.05, roughness length
      1 mm. Water bodies store heat, but there is no water temperature model, so
      the heat flux into the water is taken as zero: the hourly evaporation follows
      the radiation more closely than in reality, the daily total is little affected.

    Args:
        weather: Weather of the hour, see get_hourly_weather.
        soil_heat_flux_W_per_m2: Soil heat flux in W/m^2. Positive = flux INTO the soil.

    Returns:
        reference_evapotranspiration_m: grass reference evapotranspiration (m/hour),
            negative for dew.
        wet_vegetation_evaporation_m: evaporation from wet short vegetation (m/hour).
        open_water_evaporation_m: evaporation from open water (m/hour).
    """
    grass_aerodynamic_resistance_s_per_m: np.float32 = (
        GRASS_AERODYNAMIC_RESISTANCE_TIMES_WIND_M / weather.wind_10m_m_per_s
    )
    reference_evapotranspiration_m: np.float32 = penman_monteith(
        available_energy_W_per_m2=get_isothermal_net_radiation(
            weather, GRASS_ALBEDO, VEGETATION_EMISSIVITY
        )
        - soil_heat_flux_W_per_m2,
        vapour_pressure_deficit_kPa=weather.vapour_pressure_deficit_kPa,
        slope_kPa_per_C=weather.slope_kPa_per_C,
        psychrometric_constant_kPa_per_C=weather.psychrometric_constant_kPa_per_C,
        volumetric_heat_capacity_air_J_per_m3_K=weather.volumetric_heat_capacity_air_J_per_m3_K,
        aerodynamic_resistance_s_per_m=grass_aerodynamic_resistance_s_per_m,
        surface_resistance_s_per_m=GRASS_SURFACE_RESISTANCE_S_PER_M,
        radiative_resistance_s_per_m=np.float32(np.inf),  # FAO-56 eq. 53
    )
    wet_vegetation_evaporation_m: np.float32 = get_surface_evaporation(
        weather,
        albedo=GRASS_ALBEDO,
        emissivity=VEGETATION_EMISSIVITY,
        heat_flux_into_ground_W_per_m2=soil_heat_flux_W_per_m2,
        aerodynamic_resistance_s_per_m=grass_aerodynamic_resistance_s_per_m,
        surface_resistance_s_per_m=np.float32(0.0),
    )
    open_water_evaporation_m: np.float32 = get_surface_evaporation(
        weather,
        albedo=WATER_ALBEDO,
        emissivity=WATER_EMISSIVITY,
        heat_flux_into_ground_W_per_m2=np.float32(0.0),
        aerodynamic_resistance_s_per_m=WATER_AERODYNAMIC_RESISTANCE_TIMES_WIND_M
        / weather.wind_10m_m_per_s,
        surface_resistance_s_per_m=np.float32(0.0),
    )
    return (
        reference_evapotranspiration_m,
        wet_vegetation_evaporation_m,
        open_water_evaporation_m,
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_wet_forest_canopy_evaporation(weather: HourlyWeather) -> np.float32:
    """Evaporation rate of water intercepted on a wet forest canopy (m/hour).

    Penman-Monteith for a wet surface (r_s = 0, Rutter/Gash) with the aerodynamic
    resistance of a forest canopy (FOREST_AERODYNAMIC_RESISTANCE_TIMES_WIND_M, about
    10 s/m at 3 m/s). A tall, rough canopy mixes air far more efficiently than short
    grass, so a wet forest canopy evaporates much faster than the grass reference,
    especially during and after rain when radiation is low and evaporation is driven
    by the vapour pressure deficit.

    Args:
        weather: Weather of the hour, see get_hourly_weather.

    Returns:
        Potential evaporation from the wet canopy (m/hour), not negative.
    """
    return max(
        np.float32(0.0),
        get_surface_evaporation(
            weather,
            albedo=FOREST_ALBEDO,
            emissivity=VEGETATION_EMISSIVITY,
            heat_flux_into_ground_W_per_m2=np.float32(0.0),
            aerodynamic_resistance_s_per_m=FOREST_AERODYNAMIC_RESISTANCE_TIMES_WIND_M
            / weather.wind_10m_m_per_s,
            surface_resistance_s_per_m=np.float32(0.0),
        ),
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_canopy_radiation_attenuation(
    leaf_area_index: np.float32,
    extinction_coefficient: np.float32 = np.float32(0.7),
) -> np.float32:
    """Calculate the radiation attenuation factor through a canopy using Beer's Law.

    Args:
        leaf_area_index: Leaf Area Index [-].
        extinction_coefficient: Extinction coefficient (k) [-]. Default is 0.7 following FAO-56 eq. 97.

    Returns:
        Attenuation factor (0-1), representing the fraction of radiation transmitted through the canopy.
    """
    return np.exp(-extinction_coefficient * leaf_area_index)


@njit(cache=NUMBA_CACHE, inline="always")
def get_potential_evapotranspiration(
    reference_evapotranspiration_grass_m: np.float32,
    crop_factor: np.float32,
    CO2_induced_crop_factor_adustment: np.float32,
) -> np.float32:
    """Calculate potential evapotranspiration.

    Args:
        reference_evapotranspiration_grass_m: Reference evapotranspiration [m]
        crop_factor: Crop factor for each land use type [dimensionless]
        CO2_induced_crop_factor_adustment: Adjustment factor for CO2 effects [dimensionless]

    Returns:
        Potential evapotranspiration (m).
    """
    return (
        crop_factor * reference_evapotranspiration_grass_m
    ) * CO2_induced_crop_factor_adustment


@njit(cache=NUMBA_CACHE, inline="always")
def get_interception_capacity(leaf_area_index: np.float32) -> np.float32:
    """Water a canopy can hold on its leaves and stems (m).

    Von Hoyningen-Huene (1981), as in LISFLOOD (eq. 2-7):
    S_max = 0.935 + 0.498 LAI - 0.00575 LAI^2 (mm). The regression was fitted on
    closed crop canopies. Its constant term is water held independent of the leaf
    area (stems, ears, the film over the wetted canopy), but only where there is a
    canopy. LISFLOOD applies the constant from LAI > 0.1, so that the capacity jumps
    from 0 to about 1 mm. Here, deviating from LISFLOOD, the constant is scaled with
    the cover fraction of the canopy, 1 - exp(-0.7 LAI): the capacity grows
    smoothly from zero without leaves to the LISFLOOD value for a closed canopy
    (cover 0.94 at LAI = 4).

    Args:
        leaf_area_index: Leaf area index (-).

    Returns:
        Interception capacity (m).
    """
    cover_fraction: np.float32 = np.float32(1.0) - get_canopy_radiation_attenuation(
        leaf_area_index
    )
    return (
        np.float32(0.935) * cover_fraction
        + np.float32(0.498) * leaf_area_index
        - np.float32(0.00575) * leaf_area_index**2
    ) / np.float32(1000.0)


# FAO-56 dual crop coefficient (Allen et al., 1998, ch. 7; Allen & Pereira, 2009).
# Basal crop coefficient of bare soil (FAO-56 eq. 97, Kc_min).
BASAL_CROP_COEFFICIENT_MINIMUM: np.float32 = np.float32(0.15)
# Upper limit of evapotranspiration from a wet surface relative to ET0 in a
# sub-humid climate (RHmin = 45 %, u2 = 2 m/s), FAO-56 eq. 72.
CROP_COEFFICIENT_MAXIMUM_SUBHUMID: np.float32 = np.float32(1.2)
# The end-of-season basal crop coefficient is adjusted to the climate only when it
# is at least 0.45 (FAO-56 eq. 70): crops harvested dry transpire little at the end.
MINIMUM_END_SEASON_BASAL_CROP_COEFFICIENT_FOR_ADJUSTMENT: np.float32 = np.float32(0.45)
# Minimum soil evaporation coefficient after complete wetting of the soil surface,
# also under full cover (FAO-56 eq. 72: Kc_max >= Kcb + 0.05).
MINIMUM_WET_SOIL_EVAPORATION_COEFFICIENT: np.float32 = np.float32(0.05)
# Difference between the single crop coefficient (FAO-56 Table 12) and the basal
# crop coefficient in the mid-season (ground cover over 80 %) and at the end of the
# season (infrequently wetted), FAO-56 Table 18.
SINGLE_TO_BASAL_CROP_COEFFICIENT_DIFFERENCE: np.float32 = np.float32(0.05)
# Basal crop coefficient at full cover. Grass: the reference crop has Kc = 1 at full
# cover; FAO-56 Table 18 gives Kcb = Kc - 0.05 at over 80 % cover. Forest: FAO-56
# has two values. Conifer trees: Kcb = 0.95 (Table 17; Kc = 1.00 in Table 12),
# because their stomata control transpiration strongly (Table 17, footnote 24).
# Natural tall vegetation in general: Kcb_full = 1.0 + 0.1 h, at most 1.20 for
# h > 2 m (FAO-56 eq. 100-101), an upper limit set by the available energy, not a
# measured broadleaf value. The model uses the conifer value for all forest (it has
# no needleleaf map). The rough-canopy effect matters for evaporation of
# intercepted water, which is computed separately for forest
# (get_wet_forest_canopy_evaporation), not for transpiration.
BASAL_CROP_COEFFICIENT_FULL_COVER_GRASS: np.float32 = np.float32(0.95)
BASAL_CROP_COEFFICIENT_FULL_COVER_FOREST: np.float32 = np.float32(0.95)


# Seasonal activity of evergreen trees: the minimum-temperature index of the
# Growing Season Index (GSI) of Jolly, W. M., Nemani, R., & Running, S. W. (2005). A
# generalized, bioclimatic index to predict foliar phenology in response to climate.
# Global Change Biology, 11(4), 619-632, doi:10.1111/j.1365-2486.2005.00930.x.
# The daily index is 0 when the daily minimum air temperature is at or below -2 °C,
# 1 at or above 5 °C and linear in between (eq. 1); the GSI is its 21-day moving
# average (p. 622), which buffers single cold or warm days. The thresholds come from
# the literature (cold soil restricts water uptake by roots; below -2 °C water can
# freeze in the xylem of some trees) and are used unchanged for all biomes; with
# them the GSI followed the satellite greenness (NDVI) of nine sites across the
# globe (r = 0.57-0.99, Table 2).
# The GSI is an index of the seasonal state of the canopy. It is used here only where
# the model has no seasonal signal of its own: the evergreen part of the forest,
# which is held at its annual maximum leaf area because the satellite leaf area of
# evergreen canopies drops in winter for other reasons
# (get_forest_leaf_area_index_with_evergreen). Evergreen needles are present all
# year; in the cold season every needle transpires less (closed stomata, reduced
# photosynthetic capacity, restricted water uptake). The index therefore scales the
# conductance of the evergreen leaves, so the transpiration capacity of the canopy
# falls in proportion to the share of its leaf area that is active
# (get_active_leaf_fraction). This differs from a canopy with fewer leaves, where
# the remaining leaves receive more light and make up for part of the loss.
# Deciduous trees are not scaled: their leaf-out and leaf fall are in the satellite
# leaf area, and the index would count the same season twice. Interception and
# shading use the full leaf area. The use of the index for transpiration is a model
# assumption: Jolly et al. derived it for greenness.
# The other two GSI indices are not used: soil water stress is modelled directly (the
# paper suggests soil water in place of the vapour pressure deficit in a land-surface
# model), and day length is left out, so that only the minimum temperature limits the
# winter transpiration of evergreen trees.
GROWING_SEASON_AVERAGING_DAYS: int = 21
GROWING_SEASON_INACTIVE_MINIMUM_TEMPERATURE_C: np.float32 = np.float32(-2.0)
GROWING_SEASON_ACTIVE_MINIMUM_TEMPERATURE_C: np.float32 = np.float32(5.0)


def get_minimum_temperature_index(
    daily_minimum_air_temperature_C: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Daily minimum-temperature index of the Growing Season Index (0-1).

    Jolly et al. (2005), eq. 1: 0 at or below -2 °C, 1 at or above 5 °C of the daily
    minimum air temperature, linear in between.

    Args:
        daily_minimum_air_temperature_C: Minimum air temperature of the day (°C).

    Returns:
        Minimum-temperature index of the day (0-1).
    """
    return np.clip(
        (
            daily_minimum_air_temperature_C
            - GROWING_SEASON_INACTIVE_MINIMUM_TEMPERATURE_C
        )
        / (
            GROWING_SEASON_ACTIVE_MINIMUM_TEMPERATURE_C
            - GROWING_SEASON_INACTIVE_MINIMUM_TEMPERATURE_C
        ),
        np.float32(0.0),
        np.float32(1.0),
    ).astype(np.float32)


def update_canopy_temperature_acclimation(
    minimum_temperature_index_record: TwoDArrayFloat32,
    daily_minimum_air_temperature_C: ArrayFloat32,
) -> ArrayFloat32:
    """Seasonal activity of evergreen trees (0-1).

    The 21-day moving average of the daily minimum-temperature index (Jolly et al.,
    2005). In spring the activity recovers with rising temperature and in autumn it
    declines; it scales the transpiration of the evergreen leaves
    (get_active_leaf_fraction). The record holds the daily index of the last
    GROWING_SEASON_AVERAGING_DAYS days (oldest first), NaN for days before the start
    of the spinup, which are left out of the average. It is updated in place.

    Args:
        minimum_temperature_index_record: Daily minimum-temperature index of the
            last days (-), shape (GROWING_SEASON_AVERAGING_DAYS, cells). Updated in
            place.
        daily_minimum_air_temperature_C: Minimum air temperature of today (°C).

    Returns:
        Seasonal activity of evergreen trees (0-1).
    """
    minimum_temperature_index_record[:-1] = minimum_temperature_index_record[1:]
    minimum_temperature_index_record[-1] = get_minimum_temperature_index(
        daily_minimum_air_temperature_C
    )
    record_days = np.sum(~np.isnan(minimum_temperature_index_record), axis=0)
    return (np.nansum(minimum_temperature_index_record, axis=0) / record_days).astype(
        np.float32
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_active_leaf_fraction(
    leaf_area_index: np.float32,
    evergreen_leaf_area_index: np.float32,
    evergreen_activity: np.float32,
) -> np.float32:
    """Share of the leaf area of a forest that is active in the current season (0-1).

    Evergreen leaves count with the seasonal activity of evergreen trees; the rest of
    the leaf area is deciduous and seasonal already, so it counts in full:
    (deciduous LAI + evergreen LAI * activity) / LAI. All leaves share the light in
    the canopy, so the transpiration capacity of the canopy scales with this share.

    Args:
        leaf_area_index: Leaf area index of the forest (-), evergreen plus deciduous.
        evergreen_leaf_area_index: Evergreen part of that leaf area index (-).
        evergreen_activity: Seasonal activity of evergreen trees (0-1), see
            update_canopy_temperature_acclimation.

    Returns:
        Active share of the leaf area (0-1), 1 for a canopy without leaves.
    """
    if leaf_area_index <= np.float32(0.0):
        return np.float32(1.0)
    evergreen: np.float32 = min(evergreen_leaf_area_index, leaf_area_index)
    return (
        np.float32(1.0)
        - evergreen * (np.float32(1.0) - evergreen_activity) / leaf_area_index
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_crop_coefficient_climate_adjustment(
    wind_2m_m_per_s: np.float32,
    minimum_relative_humidity_percent: np.float32,
    vegetation_height_m: np.float32,
) -> np.float32:
    """Adjustment of the crop coefficients to the local climate (FAO-56 eq. 70, 72).

    [0.04 (u2 - 2) - 0.004 (RHmin - 45)] (h / 3)^0.3. The tabulated crop
    coefficients and the wet-surface limit 1.2 hold for a sub-humid climate with
    moderate wind (RHmin = 45 %, u2 = 2 m/s). Compared with the grass reference, tall
    vegetation is coupled more strongly to the air: in dry, windy air it evaporates
    relatively more, in humid, calm air relatively less, and more so the taller it
    is. FAO-56 applies it to the mid-season and end-of-season basal crop
    coefficients (eq. 70), the basal crop coefficient of natural vegetation at full
    cover (eq. 101) and the upper limit Kc_max (eq. 72). FAO-56 uses the mean wind
    and humidity of a growth stage; here those of the day, so the crop coefficients
    follow the weather. The inputs are limited to the range for which FAO-56 gives
    the adjustment: 1-6 m/s, 20-80 % and 0.1-10 m.

    Args:
        wind_2m_m_per_s: Mean wind speed at 2 m over grass (m/s).
        minimum_relative_humidity_percent: Daily minimum relative humidity (%).
        vegetation_height_m: Height of the vegetation (m).

    Returns:
        Adjustment added to the crop coefficients (-).
    """
    wind_m_per_s: np.float32 = min(
        max(wind_2m_m_per_s, np.float32(1.0)), np.float32(6.0)
    )
    humidity_percent: np.float32 = min(
        max(minimum_relative_humidity_percent, np.float32(20.0)), np.float32(80.0)
    )
    height_m: np.float32 = min(
        max(vegetation_height_m, np.float32(0.1)), np.float32(10.0)
    )
    return (
        np.float32(0.04) * (wind_m_per_s - np.float32(2.0))
        - np.float32(0.004) * (humidity_percent - np.float32(45.0))
    ) * (height_m / np.float32(3.0)) ** np.float32(0.3)


def get_daily_wind_and_minimum_relative_humidity(
    tas_2m_K: TwoDArrayFloat32,
    dewpoint_tas_2m_K: TwoDArrayFloat32,
    wind_u10m_m_per_s: TwoDArrayFloat32,
    wind_v10m_m_per_s: TwoDArrayFloat32,
) -> tuple[ArrayFloat32, ArrayFloat32]:
    """Daily mean 2 m wind over grass and daily minimum relative humidity.

    The climate inputs of the FAO-56 adjustment of the crop coefficients
    (get_crop_coefficient_climate_adjustment), from the hourly forcing of one day.

    Args:
        tas_2m_K: Air temperature (K), shape (cells, 24).
        dewpoint_tas_2m_K: Dew point temperature (K), shape (cells, 24).
        wind_u10m_m_per_s: Eastward wind at 10 m (m/s), shape (cells, 24).
        wind_v10m_m_per_s: Northward wind at 10 m (m/s), shape (cells, 24).

    Returns:
        Daily mean wind speed at 2 m (m/s) and daily minimum relative humidity (%)
        per cell.
    """
    wind_10m_m_per_s = np.sqrt(wind_u10m_m_per_s**2 + wind_v10m_m_per_s**2).mean(axis=1)
    relative_humidity_percent = np.float32(100.0) * np.minimum(
        get_vapour_pressure(dewpoint_tas_2m_K - np.float32(273.15))
        / get_vapour_pressure(tas_2m_K - np.float32(273.15)),
        np.float32(1.0),
    )
    return (
        (wind_10m_m_per_s * WIND_2M_PER_WIND_10M_OVER_GRASS).astype(np.float32),
        relative_humidity_percent.min(axis=1).astype(np.float32),
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_wet_soil_evaporation_coefficient(
    basal_crop_coefficient: np.float32,
    cover_fraction: np.float32,
    maximum_crop_coefficient: np.float32,
) -> np.float32:
    """Soil evaporation coefficient of a wet soil surface (FAO-56 eq. 71).

    Soil evaporation is Ke = min(Kr (Kc_max - Kcb), few Kc_max) (FAO-56 eq. 71), with
    the exposed and wetted fraction few = min(1 - fc, fw) = 1 - fc (FAO-56 eq. 75;
    rain wets the whole surface, fw = 1), where 1 - fc is the soil not covered or
    shaded by vegetation. For a wet surface (Kr = 1)
    Ke_max = min(Kc_max - Kcb, (1 - fc) Kc_max). The drying of the surface (Kr) is
    applied hourly to the energy-limited term, with the soil moisture of that hour
    (calculate_bare_soil_evaporation).

    Args:
        basal_crop_coefficient: Basal crop coefficient Kcb (-).
        cover_fraction: Fraction of the soil covered by vegetation fc (0-1).
        maximum_crop_coefficient: Upper limit of the crop coefficient Kc_max (-).

    Returns:
        Soil evaporation coefficient of a wet soil surface Ke_max (-).
    """
    return max(
        np.float32(0.0),
        min(
            maximum_crop_coefficient - basal_crop_coefficient,
            (np.float32(1.0) - cover_fraction) * maximum_crop_coefficient,
        ),
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_maximum_crop_coefficient(
    basal_crop_coefficient: np.float32,
    climate_adjustment: np.float32,
) -> np.float32:
    """Upper limit of the crop coefficient of a wet surface (FAO-56 eq. 72).

    Kc_max = max(1.2 + climate adjustment, Kcb + 0.05).

    Args:
        basal_crop_coefficient: Basal crop coefficient Kcb (-).
        climate_adjustment: See get_crop_coefficient_climate_adjustment (-).

    Returns:
        Kc_max (-).
    """
    return max(
        CROP_COEFFICIENT_MAXIMUM_SUBHUMID + climate_adjustment,
        basal_crop_coefficient + MINIMUM_WET_SOIL_EVAPORATION_COEFFICIENT,
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_dual_crop_coefficients(
    leaf_area_index: np.float32,
    active_leaf_fraction: np.float32,
    basal_crop_coefficient_full_cover: np.float32,
    climate_adjustment: np.float32,
) -> tuple[np.float32, np.float32, np.float32]:
    """Crop coefficients for natural vegetation with the FAO-56 dual approach.

    The basal crop coefficient (transpiration) follows FAO-56 eq. 97,
    Kcb = Kc_min + (Kcb_full - Kc_min)(1 - exp(-0.7 LAI)), times the active share of
    the leaf area (1, except for evergreen trees in the cold season, see
    get_active_leaf_fraction). Kcb_full is adjusted to
    the climate (FAO-56 eq. 101). The cover fraction fc
    is the fraction of radiation intercepted by the canopy, 1 - exp(-0.7 LAI)
    (get_canopy_radiation_attenuation), the effective (shaded) cover fraction of
    FAO-56 Box 18. Soil evaporation from a wet surface follows FAO-56 eq. 71-72
    (get_wet_soil_evaporation_coefficient). The total is the crop coefficient for a
    wet soil surface; the drying of the surface reduces soil evaporation later
    (FAO-56 Kr).

    Args:
        leaf_area_index: Leaf area index (-).
        active_leaf_fraction: Share of the leaf area that is active in the current
            season (0-1).
        basal_crop_coefficient_full_cover: Basal crop coefficient at full cover (-).
        climate_adjustment: Adjustment of the crop coefficients to the climate (-),
            see get_crop_coefficient_climate_adjustment.

    Returns:
        Crop coefficient for a wet soil surface (Kcb + Ke_max), the basal crop
        coefficient Kcb and the upper limit Kc_max (-).
    """
    cover_fraction: np.float32 = np.float32(1.0) - get_canopy_radiation_attenuation(
        leaf_area_index
    )
    basal: np.float32 = BASAL_CROP_COEFFICIENT_MINIMUM + max(
        basal_crop_coefficient_full_cover
        + climate_adjustment
        - BASAL_CROP_COEFFICIENT_MINIMUM,
        np.float32(0.0),
    ) * (cover_fraction * active_leaf_fraction)
    maximum: np.float32 = get_maximum_crop_coefficient(basal, climate_adjustment)
    soil_evaporation: np.float32 = get_wet_soil_evaporation_coefficient(
        basal, cover_fraction, maximum
    )
    return (
        np.float32(basal + soil_evaporation),
        np.float32(basal),
        np.float32(maximum),
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_expected_heat_units(
    mean_air_temperature_C: np.float32,
    air_temperature_standard_deviation_C: np.float32,
    base_temperature_C: np.float32,
) -> np.float32:
    """Expected daily heat units of a day with a normally distributed temperature.

    Heat units are the daily mean air temperature above the base temperature of a
    crop (zero on colder days). For a climatological mean and day-to-day spread
    the expectation of max(T - T_base, 0) is (mean - T_base) * Phi(z) + std * phi(z)
    with z = (mean - T_base) / std, so that warm days in a cold period count.

    Args:
        mean_air_temperature_C: Climatological daily mean air temperature (C).
        air_temperature_standard_deviation_C: Standard deviation of the daily mean
            air temperature between days (C).
        base_temperature_C: Base temperature for crop development (C).

    Returns:
        Expected heat units of one day (C day).
    """
    excess_C = mean_air_temperature_C - base_temperature_C
    if air_temperature_standard_deviation_C <= np.float32(0.0):
        return max(excess_C, np.float32(0.0))
    z = excess_C / air_temperature_standard_deviation_C
    cumulative_probability = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    probability_density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return np.float32(
        excess_C * cumulative_probability
        + air_temperature_standard_deviation_C * probability_density
    )


@njit(cache=NUMBA_CACHE)
def update_crop_heat_units(
    crop_map: npt.NDArray[np.int32],
    crop_age_days_map: npt.NDArray[np.int32],
    crop_harvest_age_days: npt.NDArray[np.int32],
    crop_heat_units_C_day: npt.NDArray[np.float32],
    crop_heat_units_at_harvest_C_day: npt.NDArray[np.float32],
    crop_base_temperature_C: npt.NDArray[np.float32],
    mean_daily_air_temperature_C: npt.NDArray[np.float32],
    HRU_to_grid: npt.NDArray[np.int32],
    air_temperature_dekadal_mean_C: npt.NDArray[np.float32],
    air_temperature_dekadal_standard_deviation_C: npt.NDArray[np.float32],
    dekad_per_day_of_year: npt.NDArray[np.int32],
    day_of_year: int,
) -> npt.NDArray[np.float32]:
    """Accumulate crop heat units and return the development progress of each crop.

    Crop development follows accumulated heat units (growing degree days): the daily
    mean air temperature above the base temperature of the crop. The heat units a
    crop needs until harvest are those expected from the temperature climatology
    between its sowing and harvest date, so the cultivar fits the local climate and
    calendar (as the potential heat units in SWAT and LPJmL). An autumn-sown crop
    gains few heat units in winter and stays in its early stages until spring; a
    warm season speeds the crop up, a cold one delays it.

    Fields that were sown today, or that are met for the first time in the middle
    of their season (start of a run), get their heat units until harvest and the
    expected heat units since sowing. The state arrays are updated in place.

    Args:
        crop_map: Crop per HRU (-1 without crop).
        crop_age_days_map: Days since sowing, 1 on the sowing day (-1 without crop).
        crop_harvest_age_days: Age at which the crop is harvested (days).
        crop_heat_units_C_day: Heat units accumulated since sowing (C day), NaN
            without crop. Updated in place.
        crop_heat_units_at_harvest_C_day: Heat units expected between sowing and
            harvest (C day), NaN without crop. Updated in place.
        crop_base_temperature_C: Base temperature for development per crop (C).
        mean_daily_air_temperature_C: Mean air temperature of today per HRU (C).
        HRU_to_grid: Grid cell of each HRU.
        air_temperature_dekadal_mean_C: Climatological daily mean air temperature
            per dekad and grid cell (C), shape (36, number of grid cells).
        air_temperature_dekadal_standard_deviation_C: Standard deviation of the
            daily mean air temperature per dekad and grid cell (C).
        dekad_per_day_of_year: Dekad (0-35) of each day of a leap year (366 values).
        day_of_year: Day of the year of today, starting at 0.

    Returns:
        Development progress per HRU, from 0 at sowing to 1 at maturity (NaN
        without crop).
    """
    crop_progress = np.full(crop_map.size, np.nan, dtype=np.float32)
    days_per_year: int = dekad_per_day_of_year.size

    for i in range(crop_map.size):
        crop = crop_map[i]
        if crop == -1:
            crop_heat_units_C_day[i] = np.nan
            crop_heat_units_at_harvest_C_day[i] = np.nan
            continue

        base_temperature_C = crop_base_temperature_C[crop]
        age_days = crop_age_days_map[i]

        if age_days == 1 or np.isnan(crop_heat_units_at_harvest_C_day[i]):
            grid_cell = HRU_to_grid[i]
            heat_units_since_sowing = np.float32(0.0)
            heat_units_at_harvest = np.float32(0.0)
            for age in range(1, crop_harvest_age_days[i] + 1):
                dekad = dekad_per_day_of_year[
                    (day_of_year + age - age_days) % days_per_year
                ]
                expected_heat_units = get_expected_heat_units(
                    air_temperature_dekadal_mean_C[dekad, grid_cell],
                    air_temperature_dekadal_standard_deviation_C[dekad, grid_cell],
                    base_temperature_C,
                )
                heat_units_at_harvest += expected_heat_units
                if age < age_days:
                    heat_units_since_sowing += expected_heat_units
            crop_heat_units_C_day[i] = heat_units_since_sowing
            crop_heat_units_at_harvest_C_day[i] = heat_units_at_harvest

        crop_heat_units_C_day[i] += max(
            mean_daily_air_temperature_C[i] - base_temperature_C, np.float32(0.0)
        )
        if crop_heat_units_at_harvest_C_day[i] > np.float32(0.0):
            crop_progress[i] = min(
                crop_heat_units_C_day[i] / crop_heat_units_at_harvest_C_day[i],
                np.float32(1.0),
            )
        else:
            # never warm enough for this crop to develop
            crop_progress[i] = np.float32(0.0)

    return crop_progress


@njit(cache=NUMBA_CACHE)
def get_crop_factors_and_root_depths_and_lai(
    land_use_map: npt.NDArray[np.int32],
    leaf_area_index_forest: npt.NDArray[np.float32],
    leaf_area_index_forest_evergreen: npt.NDArray[np.float32],
    leaf_area_index_grassland_like: npt.NDArray[np.float32],
    evergreen_activity: npt.NDArray[np.float32],
    wind_2m_m_per_s: npt.NDArray[np.float32],
    minimum_relative_humidity_percent: npt.NDArray[np.float32],
    crop_map: npt.NDArray[np.int32],
    crop_progress: npt.NDArray[np.float32],
    land_owners: npt.NDArray[np.int32],
    crop_stage_lengths: npt.NDArray[np.int32],
    crop_sub_stage_lengths: npt.NDArray[np.int32],
    crop_factor_per_crop_stage: npt.NDArray[np.float32],
    crop_root_depths: npt.NDArray[np.float32],
    grassland_root_depth_m: np.float32,
    crop_init_root_depth: np.float32 = np.float32(0.2),
    get_crop_sub_stage: bool = False,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int8],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Calculate crop factors, root depths, dynamic LAI, and interception capacity.

    The crop factor is the FAO-56 crop coefficient for a wet soil surface; the basal
    crop factor is its transpiration part. For crops in season the crop factor is
    the basal crop coefficient derived from the tabulated single crop coefficient
    (FAO-56 Table 18) for transpiration, plus evaporation of the exposed wet soil up
    to the FAO-56 wet-surface maximum Kc_max, adjusted to the daily wind and
    humidity (FAO-56 eq. 72). The crop stages follow the development progress from heat
    units (update_crop_heat_units). Grassland and forest use the dual crop
    coefficient from leaf area (get_dual_crop_coefficients). A farm field without a
    crop (after harvest, before sowing) is bare soil or stubble: no leaf area, the
    FAO-56 minimum basal crop coefficient and evaporation from the wet soil.

    Args:
        land_use_map: Map of land use types per HRU.
        leaf_area_index_forest: Leaf area index for forest land use type (-).
        leaf_area_index_grassland_like: Leaf area index for grassland-like land use type (-).
        leaf_area_index_forest_evergreen: Evergreen part of the forest leaf area
            index (-), see get_evergreen_leaf_area_index.
        evergreen_activity: Seasonal activity of evergreen trees (0-1), see
            update_canopy_temperature_acclimation.
        wind_2m_m_per_s: Daily mean wind speed at 2 m per HRU (m/s).
        minimum_relative_humidity_percent: Daily minimum relative humidity per
            HRU (%).
        crop_map: Map of crop types (-1 for non-crop land use).
        crop_progress: Development progress of the crop from 0 at sowing to 1 at
            maturity (NaN without crop), see update_crop_heat_units.
        land_owners: Farmer that owns the HRU (-1 for land that is not a farm field).
        crop_stage_lengths: Array of lengths (l1, l2, l3, l4) in % of the heat units until harvest for each crop type (summing to 100).
        crop_sub_stage_lengths: Array of sub-stage lengths (d1, d2a, d2b, d3a, d3b, d4) in % of the heat units until harvest for each crop type.
        crop_factor_per_crop_stage: Crop factors (kc_initial, kc_mid, kc_end) for each crop type (-).
        crop_root_depths: Maximum root depths for each crop type under rainfed and irrigated conditions (m).
        crop_init_root_depth: Initial crop seedling rooting depth (m).
        get_crop_sub_stage: Whether to compute detailed GAEZ crop sub-stages.
        grassland_root_depth_m: Rooting depth of grassland and other non-crop
            grassland-like land (m).

    Returns:
        crop_factor: Array of crop factors (wet soil surface) per HRU (-).
        basal_crop_factor: Array of basal crop factors (transpiration) per HRU (-).
        maximum_crop_factor: Array of upper limits of the crop factor Kc_max per
            HRU (-), see get_maximum_crop_coefficient.
        root_depth: Array of rooting depths per HRU (m).
        crop_sub_stage: Array of crop sub-stages (0-5, or -1 if not calculated) per HRU.
        leaf_area_index: Dynamic Leaf Area Index per HRU (-).
        interception_capacity_m: Dynamic canopy interception capacity in meters (m).

    Raises:
        ValueError: If an unhandled land use type is encountered.
    """
    crop_factor = np.full_like(crop_map, np.nan, dtype=np.float32)
    basal_crop_factor = np.full_like(crop_map, np.nan, dtype=np.float32)
    maximum_crop_factor = np.zeros_like(crop_map, dtype=np.float32)
    root_depth = np.full_like(crop_map, np.nan, dtype=np.float32)
    crop_sub_stage = np.full_like(crop_map, -1, dtype=np.int8)
    leaf_area_index = np.zeros_like(crop_map, dtype=np.float32)
    interception_capacity_m = np.zeros_like(crop_map, dtype=np.float32)

    for i in range(crop_map.size):
        land_use = land_use_map[i]
        crop = crop_map[i]

        curr_crop_sub_stage = np.int8(-1)

        # All cropland
        if crop != -1:
            progress = crop_progress[i]
            assert progress >= 0.0 and progress <= 1.0
            # rounded, so that single precision does not move a crop that is
            # exactly at a stage boundary
            progress_percentage = round(float(progress) * 100.0, 3)
            l1, l2, l3, l4 = crop_stage_lengths[crop]
            kc1, kc2, kc3 = crop_factor_per_crop_stage[crop]
            assert l1 + l2 + l3 + l4 == 100

            # FAO-56 dual crop coefficient. The tabulated crop coefficients are
            # single crop coefficients (FAO-56 Table 12), which include the soil
            # evaporation of a typical wetting frequency. The basal (transpiration)
            # part follows FAO-56 Table 18: Kc - 0.05 in the mid-season (ground cover
            # over 80 %) and at the end of the season (infrequently wetted), and
            # Kc_min = 0.15 at the start of an annual crop on (nearly) bare soil
            # (initial Kc below the mid-season Kc). Crops with the same Kc all season
            # (perennials) keep Kc - 0.05 from the start. The mid-season value, and
            # the end-of-season value if at least 0.45, are adjusted to the climate
            # (FAO-56 eq. 70); the initial value is not.
            climate_adjustment: np.float32 = get_crop_coefficient_climate_adjustment(
                wind_2m_m_per_s[i],
                minimum_relative_humidity_percent[i],
                CROP_HEIGHT_M,
            )
            kcb2 = max(
                kc2 - SINGLE_TO_BASAL_CROP_COEFFICIENT_DIFFERENCE + climate_adjustment,
                BASAL_CROP_COEFFICIENT_MINIMUM,
            )
            kcb3 = kc3 - SINGLE_TO_BASAL_CROP_COEFFICIENT_DIFFERENCE
            if kcb3 >= MINIMUM_END_SEASON_BASAL_CROP_COEFFICIENT_FOR_ADJUSTMENT:
                kcb3 += climate_adjustment
            kcb3 = max(kcb3, BASAL_CROP_COEFFICIENT_MINIMUM)
            if kc1 < kc2:
                kcb1 = BASAL_CROP_COEFFICIENT_MINIMUM
            else:
                kcb1 = max(
                    kc1 - SINGLE_TO_BASAL_CROP_COEFFICIENT_DIFFERENCE,
                    BASAL_CROP_COEFFICIENT_MINIMUM,
                )

            # Combined stage progression for piecewise linear variables (Kcb, LAI)
            s1_end = l1
            s2_end = l1 + l2
            s3_end = s2_end + l3

            if progress_percentage <= s1_end:
                # Stage 1: Initial
                curr_basal_crop_factor = kcb1
                frac = progress_percentage / l1 if l1 > 0 else 1.0
                curr_lai = 0.1 + frac * (0.5 - 0.1)

            elif progress_percentage <= s2_end:
                # Stage 2: Development
                frac = (progress_percentage - s1_end) / l2 if l2 > 0 else 1.0
                curr_basal_crop_factor = kcb1 + frac * (kcb2 - kcb1)
                curr_lai = 0.5 + frac * (3.5 - 0.5)

            elif progress_percentage <= s3_end:
                # Stage 3: Mid-season
                curr_basal_crop_factor = kcb2
                curr_lai = 3.5

            else:
                # Stage 4: Late-season
                frac = (progress_percentage - s3_end) / l4 if l4 > 0 else 1.0
                curr_basal_crop_factor = kcb2 + frac * (kcb3 - kcb2)
                curr_lai = 3.5 - frac * (3.5 - 1.5)

            assert not np.isnan(curr_basal_crop_factor)
            # The exposed soil of a wet field adds soil evaporation (FAO-56 eq. 71-72).
            crop_cover: np.float32 = np.float32(1.0) - get_canopy_radiation_attenuation(
                np.float32(curr_lai)
            )
            curr_maximum_crop_factor = get_maximum_crop_coefficient(
                np.float32(curr_basal_crop_factor), climate_adjustment
            )
            curr_crop_factor = curr_basal_crop_factor + (
                get_wet_soil_evaporation_coefficient(
                    np.float32(curr_basal_crop_factor),
                    crop_cover,
                    curr_maximum_crop_factor,
                )
            )

            # Assumption: roots grow linearly from the seedling depth to their
            # maximum depth during the initial and development stages and keep that
            # depth from the start of the mid-season, when the canopy is complete.
            # FAO-56 gives only the maximum rooting depth (Table 22), not its
            # development over the season.
            is_irrigated: int = int(land_use in (PADDY_IRRIGATED, NON_PADDY_IRRIGATED))
            root_growth_fraction = (
                min(progress_percentage / s2_end, 1.0) if s2_end > 0 else 1.0
            )
            curr_root_depth = crop_init_root_depth + root_growth_fraction * max(
                (crop_root_depths[crop, is_irrigated] - crop_init_root_depth), 0.0
            )

            # Sub-stages for yield response models (GAEZ)
            if get_crop_sub_stage:
                d1, d2a, d2b, d3a, d3b, d4 = crop_sub_stage_lengths[crop]
                assert d1 + d2a + d2b + d3a + d3b + d4 == 100

                if progress_percentage <= d1:
                    curr_crop_sub_stage = np.int8(0)
                elif progress_percentage <= d1 + d2a:
                    curr_crop_sub_stage = np.int8(1)
                elif progress_percentage <= d1 + d2a + d2b:
                    curr_crop_sub_stage = np.int8(2)
                elif progress_percentage <= d1 + d2a + d2b + d3a:
                    curr_crop_sub_stage = np.int8(3)
                elif progress_percentage <= d1 + d2a + d2b + d3a + d3b:
                    curr_crop_sub_stage = np.int8(4)
                else:
                    curr_crop_sub_stage = np.int8(5)

        # Grassland. While unirrigated crops are also classified as grassland-like
        # they are handled above.
        elif land_use == GRASSLAND_LIKE:
            curr_root_depth = grassland_root_depth_m
            if land_owners[i] != -1:
                # farm field without a crop: bare soil or stubble
                curr_lai = np.float32(0.0)
            else:
                curr_lai = leaf_area_index_grassland_like[i]
            (
                curr_crop_factor,
                curr_basal_crop_factor,
                curr_maximum_crop_factor,
            ) = get_dual_crop_coefficients(
                curr_lai,
                np.float32(1.0),  # grass has no evergreen part
                BASAL_CROP_COEFFICIENT_FULL_COVER_GRASS,
                climate_adjustment=get_crop_coefficient_climate_adjustment(
                    wind_2m_m_per_s[i],
                    minimum_relative_humidity_percent[i],
                    GRASS_HEIGHT_M,
                ),
            )

        elif land_use == FOREST:
            # Trees root into the weathered rock below the 2 m soil profile
            # (regolith layer, up to 4 m in total).
            curr_root_depth = 4.0
            curr_lai = leaf_area_index_forest[i]
            (
                curr_crop_factor,
                curr_basal_crop_factor,
                curr_maximum_crop_factor,
            ) = get_dual_crop_coefficients(
                curr_lai,
                get_active_leaf_fraction(
                    curr_lai,
                    leaf_area_index_forest_evergreen[i],
                    evergreen_activity[i],
                ),
                BASAL_CROP_COEFFICIENT_FULL_COVER_FOREST,
                climate_adjustment=get_crop_coefficient_climate_adjustment(
                    wind_2m_m_per_s[i],
                    minimum_relative_humidity_percent[i],
                    FOREST_CANOPY_HEIGHT_M,
                ),
            )

        elif land_use == OPEN_WATER:
            curr_root_depth = 0.0
            curr_crop_factor = 0.0
            curr_basal_crop_factor = 0.0
            curr_maximum_crop_factor = 0.0
            curr_lai = 0.0
            curr_interception_capacity_m = np.float32(0.0)

        elif land_use == SEALED:
            # The unpaved part of sealed land (gardens, verges) carries grass. Leaf
            # area, crop coefficients and interception capacity are means over the
            # whole unit, so they scale with the unpaved share; the paved part has
            # no plants.
            unpaved_fraction: np.float32 = np.float32(1.0 - SEALED_PAVED_FRACTION)
            curr_root_depth = grassland_root_depth_m
            (
                grass_crop_factor,
                grass_basal_crop_factor,
                grass_maximum_crop_factor,
            ) = get_dual_crop_coefficients(
                leaf_area_index_grassland_like[i],
                np.float32(1.0),  # grass has no evergreen part
                BASAL_CROP_COEFFICIENT_FULL_COVER_GRASS,
                climate_adjustment=get_crop_coefficient_climate_adjustment(
                    wind_2m_m_per_s[i],
                    minimum_relative_humidity_percent[i],
                    GRASS_HEIGHT_M,
                ),
            )
            curr_crop_factor = unpaved_fraction * grass_crop_factor
            curr_basal_crop_factor = unpaved_fraction * grass_basal_crop_factor
            curr_maximum_crop_factor = unpaved_fraction * grass_maximum_crop_factor
            curr_lai = unpaved_fraction * leaf_area_index_grassland_like[i]
            curr_interception_capacity_m = unpaved_fraction * get_interception_capacity(
                leaf_area_index_grassland_like[i]
            )

        else:
            raise ValueError("Unhandled land use type")

        if land_use not in (OPEN_WATER, SEALED):
            curr_interception_capacity_m = get_interception_capacity(
                np.float32(curr_lai)
            )

        # Final array assignments
        crop_factor[i] = np.float32(curr_crop_factor)
        basal_crop_factor[i] = np.float32(curr_basal_crop_factor)
        maximum_crop_factor[i] = np.float32(curr_maximum_crop_factor)
        root_depth[i] = np.float32(curr_root_depth)
        crop_sub_stage[i] = curr_crop_sub_stage
        leaf_area_index[i] = np.float32(curr_lai)
        interception_capacity_m[i] = curr_interception_capacity_m

    return (
        crop_factor,
        basal_crop_factor,
        maximum_crop_factor,
        root_depth,
        crop_sub_stage,
        leaf_area_index,
        interception_capacity_m,
    )


@njit(cache=NUMBA_CACHE, inline="always")
def get_CO2_induced_crop_factor_adustment(
    CO2_concentration_ppm: float | np.floating,
) -> float:
    """Calculate the CO2 induced crop factor adjustment.

    For reference see:
        Reference Manual, Chapter 3 – AquaCrop, Version 7.1
        Eq. 3.10e/2

    Args:
        CO2_concentration_ppm: The CO2 concentration in ppm.

    Returns:
        The CO2 induced crop factor adjustment [dimensionless]
    """
    base_co2_concentration_ppm: float = 369.41
    return float(
        1.0
        - 0.05
        * (CO2_concentration_ppm - base_co2_concentration_ppm)
        / (550.0 - base_co2_concentration_ppm)
    )


def get_evergreen_leaf_area_index(
    leaf_area_index_forest: TwoDArrayFloat32,
    evergreen_fraction: ArrayFloat32,
) -> ArrayFloat32:
    """Evergreen part of the forest leaf area index: constant at the annual maximum.

    Args:
        leaf_area_index_forest: Forest LAI from the satellite climatology per time
            step and grid cell (time, cells).
        evergreen_fraction: Evergreen fraction of tree cover per grid cell (0-1).

    Returns:
        Evergreen leaf area index per grid cell (-).
    """
    return (
        evergreen_fraction.astype(np.float32)
        * np.nanmax(leaf_area_index_forest, axis=0)
    ).astype(np.float32)


def get_forest_leaf_area_index_with_evergreen(
    leaf_area_index_forest: TwoDArrayFloat32,
    evergreen_fraction: ArrayFloat32,
) -> TwoDArrayFloat32:
    """Forest leaf area index with evergreen canopies approximated at their annual maximum.

    Satellite-derived LAI drops strongly in winter for all forest, also for
    evergreen canopies, because of snow, low sun angles and understory die-back.
    The true seasonal change of evergreen canopies is small (~10-30 %, needle
    shedding and spring flush), so the evergreen part of the forest is
    approximated as constant at the annual maximum LAI of the cell; the deciduous
    part follows the seasonal LAI. In mixed cells the annual maximum includes
    deciduous foliage and understory, so this slightly overestimates winter LAI.

    Args:
        leaf_area_index_forest: Forest LAI per time step and grid cell (time, cells).
        evergreen_fraction: Evergreen fraction of tree cover per grid cell (0-1).

    Returns:
        Adjusted forest LAI (time, cells).
    """
    evergreen_share: ArrayFloat32 = evergreen_fraction.astype(np.float32)
    return (
        get_evergreen_leaf_area_index(leaf_area_index_forest, evergreen_fraction)[
            np.newaxis, :
        ]
        + (np.float32(1.0) - evergreen_share[np.newaxis, :]) * leaf_area_index_forest
    ).astype(np.float32)
