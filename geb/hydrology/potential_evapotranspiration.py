"""Functions to calculate potential evapotranspiration based on the Penman-Monteith equation."""

import numpy as np
import numpy.typing as npt
from numba import njit

from .landcovers import FOREST, GRASSLAND_LIKE, NON_PADDY_IRRIGATED, PADDY_IRRIGATED


@njit(cache=True, inline="always")
def get_vapour_pressure(
    temperature_C: np.float32,
) -> np.float32:
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


@njit(cache=True, inline="always")
def get_psychrometric_constant(
    ps_pa: np.float32,
) -> np.float32:
    """Calculate the psychrometric constant.

    Args:
        ps_pa: Surface pressure in Pascals.

    Returns:
        Psychrometric constant in kPa/°C.
    """
    return np.float32(0.665e-3) * np.float32(0.001) * ps_pa  # Convert Pa to kPa


@njit(cache=True, inline="always")
def get_upwelling_long_wave_radiation(
    tas_C: np.float32,
) -> np.float32:
    """Calculate the upwelling long wave radiation based on temperature.

    Args:
        tas_C: Temperature in Celsius.

    Returns:
        Upwelling long wave radiation in MJ/m^2/hour.
    """
    return np.float32(2.043e-10) * ((tas_C + np.float32(273.16)) ** 4)


@njit(cache=True, inline="always")
def W_per_m2_to_MJ_per_m2_per_hour(
    solar_radiation_W_per_m2: np.float32,
) -> np.float32:
    """Convert solar radiation from W/m^2 to MJ/m^2/hour.

    Args:
        solar_radiation_W_per_m2: Solar radiation in W/m^2.

    Returns:
        Solar radiation in MJ/m^2/hour.
    """
    return solar_radiation_W_per_m2 * (np.float32(3600) * np.float32(1e-6))


@njit(cache=True, inline="always")
def get_net_solar_radiation(
    solar_radiation_MJ_per_m2_per_hour: np.float32, albedo: np.float32
) -> np.float32:
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
    temperature_C: np.float32,
) -> np.float32:
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
    wind_10m_m_per_s: np.float32,
) -> np.float32:
    """Adjust wind speed to surface level.

    Args:
        wind_10m_m_per_s: Wind speed at 10 m height in m/s.

    Returns:
        Adjusted wind speed at 2 m height in m/s.
    """
    return wind_10m_m_per_s * np.float32(0.748)


@njit(cache=True, inline="always")
def penman_monteith(
    net_radiation_land_MJ_per_m2_per_hour: np.float32,
    net_radiation_water_MJ_per_m2_per_hour: np.float32,
    soil_heat_flux_MJ_per_m2_per_hour: np.float32,
    slope_of_saturated_vapour_pressure_curve_kPa_per_C: np.float32,
    psychrometric_constant_kPa_per_C: np.float32,
    wind_2m_m_per_s: np.float32,
    temperature_C: np.float32,
    vapour_pressure_deficit_kPa: np.float32,
) -> tuple[np.float32, np.float32]:
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


@njit(cache=True, inline="always")
def get_reference_evapotranspiration(
    tas_C: np.float32,
    dewpoint_tas_C: np.float32,
    ps_pa: np.float32,
    rlds_W_per_m2: np.float32,
    rsds_W_per_m2: np.float32,
    wind_10m_m_per_s: np.float32,
    albedo_canopy: np.float32 = np.float32(0.13),
    albedo_water: np.float32 = np.float32(0.05),
) -> tuple[np.float32, np.float32, np.float32, np.float32]:
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
        tas_C: average air temperature in Celsius.
        dewpoint_tas_C: dew point temperature in Celsius.
        ps_pa: surface pressure in Pascals.
        rlds_W_per_m2: long wave downward surface radiation flux in W/m^2.
        rsds_W_per_m2: short wave downward surface radiation flux in W/m^2.
        wind_10m_m_per_s: wind speed at 10 m height in m/s.
        albedo_canopy: albedo of vegetation canopy (default = 0.13).
        albedo_water: albedo of water surface (default = 0.05).

    Returns:
        reference_evapotranspiration_land_m_per_dt: reference evapotranspiration for land in m/dt.
        reference_evapotranspiration_water_m_per_dt: reference evapotranspiration for water in m/dt.
        net_radiation_land_MJ_per_m2_per_dt: net radiation for land in MJ/m^2/dt.
        actual_vapour_pressure_Pa: actual vapour pressure in Pa.
    """
    actual_vapour_pressure_kPa: np.float32 = get_vapour_pressure(
        temperature_C=dewpoint_tas_C
    )
    saturated_vapour_pressure_kPa: np.float32 = get_vapour_pressure(temperature_C=tas_C)

    vapour_pressure_deficit_kPa: np.float32 = get_vapour_pressure_deficit(
        saturated_vapour_pressure_kPa=saturated_vapour_pressure_kPa,
        actual_vapour_pressure_kPa=actual_vapour_pressure_kPa,
    )

    psychrometric_constant_kPa_per_C: np.float32 = get_psychrometric_constant(
        ps_pa=ps_pa
    )

    rlus_MJ_per_m2_per_hour: np.float32 = get_upwelling_long_wave_radiation(tas_C)
    rlds_MJ_per_m2_per_hour: np.float32 = W_per_m2_to_MJ_per_m2_per_hour(rlds_W_per_m2)

    net_longwave_radation_MJ_per_m2_per_hour: np.float32 = (
        rlus_MJ_per_m2_per_hour - rlds_MJ_per_m2_per_hour
    )

    solar_radiation_MJ_per_m2_per_hour: np.float32 = W_per_m2_to_MJ_per_m2_per_hour(
        rsds_W_per_m2
    )

    net_solar_radiation_land_MJ_per_m2_per_hour: np.float32 = get_net_solar_radiation(
        solar_radiation_MJ_per_m2_per_hour, albedo_canopy
    )

    net_radiation_land_MJ_per_m2_per_hour: np.float32 = (
        net_solar_radiation_land_MJ_per_m2_per_hour
        - net_longwave_radation_MJ_per_m2_per_hour
    )

    net_solar_radiation_water_MJ_per_m2_per_hour: np.float32 = get_net_solar_radiation(
        solar_radiation_MJ_per_m2_per_hour, albedo_water
    )
    net_radiation_water_MJ_per_m2_per_hour: np.float32 = (
        net_solar_radiation_water_MJ_per_m2_per_hour
        - net_longwave_radation_MJ_per_m2_per_hour
    )

    slope_of_saturated_vapour_pressure_curve_kPa_per_C: np.float32 = (
        get_slope_of_saturation_vapour_pressure_curve(temperature_C=tas_C)
    )

    wind_2m_m_per_s: np.float32 = adjust_wind_speed(wind_10m_m_per_s)

    soil_heat_flux_MJ_per_m2_per_hour: np.float32 = np.float32(0.0)

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
        actual_vapour_pressure_kPa * np.float32(1000),
    )


@njit(cache=True, inline="always")
def get_potential_transpiration(
    potential_evapotranspiration_m: np.float32,
    potential_bare_soil_evaporation_m: np.float32,
) -> np.float32:
    """Calculate potential transpiration.

    Args:
        potential_evapotranspiration_m: Potential evapotranspiration [m]
        potential_bare_soil_evaporation_m: Potential bare soil evaporation [m]

    Returns:
        Potential transpiration [m]
    """
    return max(0.0, potential_evapotranspiration_m - potential_bare_soil_evaporation_m)


@njit(cache=True, inline="always")
def get_potential_bare_soil_evaporation(
    reference_evapotranspiration_grass_m_per_day: np.float32,
) -> np.float32:
    """Calculate potential bare soil evaporation.

    Removes sublimation from potential bare soil evaporation and ensures non-negative result.

    Args:
        reference_evapotranspiration_grass_m_per_day: Reference evapotranspiration [m]

    Returns:
        Potential bare soil evaporation [m]
    """
    return max(
        np.float32(0.2) * reference_evapotranspiration_grass_m_per_day,
        np.float32(0.0),
    )


@njit(cache=True, inline="always")
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
        Potential evapotranspiration [m]
    """
    return (
        crop_factor * reference_evapotranspiration_grass_m
    ) * CO2_induced_crop_factor_adustment


@njit(cache=True)
def get_crop_factors_and_root_depths(
    land_use_map: npt.NDArray[np.int32],
    crop_factor_forest_map: npt.NDArray[np.float32],
    crop_map: npt.NDArray[np.int32],
    crop_age_days_map: npt.NDArray[np.int32],
    crop_harvest_age_days: npt.NDArray[np.int32],
    crop_stage_lengths: npt.NDArray[np.int32],
    crop_sub_stage_lengths: npt.NDArray[np.int32],
    crop_factor_per_crop_stage: npt.NDArray[np.float32],
    crop_root_depths: npt.NDArray[np.float32],
    crop_init_root_depth: np.float32 = np.float32(0.2),
    get_crop_sub_stage: bool = False,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int8],
]:
    """Calculate crop factors and root depths based on land use and crop information.

    Args:
        land_use_map: Map of land use types.
        crop_factor_forest_map: Map of crop factors for forest land use if forested.
        crop_map: Map of crop types, -1 for non-crop land use. Indices refer to crop arrays.
        crop_age_days_map: Map of crop ages in days, -1 for non-crop land use.
        crop_harvest_age_days: Array of harvest ages in days for each crop type.
        crop_stage_lengths: Array of lengths of growth stages for each crop type.
        crop_sub_stage_lengths: Array of lengths of sub-stages for each crop type.
        crop_factor_per_crop_stage: Array of crop factors for each growth stage for each crop type.
        crop_root_depths: Array of root depths for each crop type and irrigation status.
        crop_init_root_depth: Initial root depth for crops.
        get_crop_sub_stage: Whether to calculate and return crop sub-stages.

    Returns:
        crop_factor: Array of crop factors for each grid cell.
        root_depth: Array of root depths for each grid cell.
        crop_sub_stage: Array of crop sub-stages for each grid cell, -1 if not calculated.

    """
    crop_factor = np.full_like(crop_map, np.nan, dtype=np.float32)
    root_depth = np.full_like(crop_map, np.nan, dtype=np.float32)
    crop_sub_stage = np.full_like(crop_map, -1, dtype=np.int8)

    for i in range(crop_map.size):
        land_use = land_use_map[i]
        crop = crop_map[i]
        if crop != -1:
            age_days = crop_age_days_map[i]
            harvest_day = crop_harvest_age_days[i]
            assert harvest_day > 0
            crop_progress = age_days * 100 // harvest_day  # for to be integer
            assert crop_progress <= 100
            l1, l2, l3, l4 = crop_stage_lengths[crop]
            kc1, kc2, kc3 = crop_factor_per_crop_stage[crop]
            assert l1 + l2 + l3 + l4 == 100
            if crop_progress <= l1:
                field_kc = kc1
            elif crop_progress <= l1 + l2:
                field_kc = kc1 + (crop_progress - l1) * (kc2 - kc1) / l2
            elif crop_progress <= l1 + l2 + l3:
                field_kc = kc2
            else:
                assert crop_progress <= l1 + l2 + l3 + l4
                field_kc = kc2 + (crop_progress - (l1 + l2 + l3)) * (kc3 - kc2) / l4
            assert not np.isnan(field_kc)
            crop_factor[i] = field_kc

            is_irrigated: int = int(land_use in (PADDY_IRRIGATED, NON_PADDY_IRRIGATED))

            root_depth[i] = (
                crop_init_root_depth
                + age_days
                * max(
                    (crop_root_depths[crop, is_irrigated] - crop_init_root_depth), 0.0
                )
                / harvest_day
            )

            if get_crop_sub_stage:
                d1, d2a, d2b, d3a, d3b, d4 = crop_sub_stage_lengths[crop]
                assert d1 + d2a + d2b + d3a + d3b + d4 == 100

                if crop_progress <= d1:
                    crop_sub_stage[i] = 0
                elif crop_progress <= d1 + d2a:
                    crop_sub_stage[i] = 1
                elif crop_progress <= d1 + d2a + d2b:
                    crop_sub_stage[i] = 2
                elif crop_progress <= d1 + d2a + d2b + d3a:
                    crop_sub_stage[i] = 3
                elif crop_progress <= d1 + d2a + d2b + d3a + d3b:
                    crop_sub_stage[i] = 4
                else:
                    assert crop_progress <= d1 + d2a + d2b + d3a + d3b + d4
                    crop_sub_stage[i] = 5

        elif land_use == FOREST:
            root_depth[i] = 2.0  # forest root depth is set to 2m
            # crop sub stage remains -1
            crop_factor[i] = crop_factor_forest_map[i]

        elif land_use == GRASSLAND_LIKE:
            root_depth[i] = 0.1  # grassland root depth is set to 0.1m
            # crop sub stage remains -1
            crop_factor[i] = 1.0

    return crop_factor, root_depth, crop_sub_stage


@njit(cache=True, inline="always")
def get_CO2_induced_crop_factor_adustment(
    CO2_concentration_ppm: float,
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
    return 1.0 - 0.05 * (CO2_concentration_ppm - base_co2_concentration_ppm) / (
        550 - base_co2_concentration_ppm
    )
