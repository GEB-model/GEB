"""Functions to calculate snow accumulation and melt."""

import numpy as np
import numpy.typing as npt
from numba import njit

STEFAN_BOLTZMANN_CONSTANT = np.float32(5.670374419e-8)  # W m⁻² K⁻⁴
SNOW_EMISSIVITY = np.float32(0.99)  # Emissivity of snow surface
KELVIN_OFFSET = np.float32(273.15)

# Precomputed SWE range for albedo caching (0 to 10m, step 0.01m)
SWE_RANGE_M = np.arange(0, 10.01, 0.01, dtype=np.float32)


@njit(cache=True, inline="always")
def discriminate_precipitation(
    precipitation_m_per_hour: npt.NDArray[np.float32],
    air_temperature_C: npt.NDArray[np.float32],
    snowfall_threshold_temperature_C: np.float32,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Discriminate between snowfall and rainfall based on temperature.

    Args:
        precipitation_m_per_hour: Precipitation rate (m/hour).
        air_temperature_C: Air temperature (°C).
        snowfall_threshold_temperature_C: Threshold temperature for snowfall (°C).

    Returns:
        A tuple of snowfall rate (m/hour) and rainfall rate (m/hour).
    """
    is_snow = air_temperature_C <= snowfall_threshold_temperature_C
    snowfall_m_per_hour = np.where(is_snow, precipitation_m_per_hour, np.float32(0.0))
    rainfall_m_per_hour = np.where(is_snow, np.float32(0.0), precipitation_m_per_hour)
    return snowfall_m_per_hour, rainfall_m_per_hour


@njit(cache=True, inline="always")
def update_snow_temperature(
    snow_water_equivalent_m: npt.NDArray[np.float32],
    snow_temperature_C: npt.NDArray[np.float32],
    snowfall_m_per_hour: npt.NDArray[np.float32],
    air_temperature_C: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Update the snow pack temperature based on new snowfall.

    Args:
        snow_water_equivalent_m: Current snow water equivalent (m).
        snow_temperature_C: Current snow temperature (°C).
        snowfall_m_per_hour: Snowfall rate (m/hour).
        air_temperature_C: Air temperature (°C).

    Returns:
        Updated snow temperature (°C).
    """
    new_snow_water_equivalent_m = snow_water_equivalent_m + snowfall_m_per_hour
    total_snow_thermal_content_C_m = snow_temperature_C * snow_water_equivalent_m
    new_snow_thermal_content_C_m = air_temperature_C * snowfall_m_per_hour
    new_snow_temperature_C = np.where(
        new_snow_water_equivalent_m > 0,
        (total_snow_thermal_content_C_m + new_snow_thermal_content_C_m)
        / new_snow_water_equivalent_m,
        air_temperature_C,
    )
    new_snow_temperature_C = np.minimum(new_snow_temperature_C, np.float32(0.0))
    return new_snow_temperature_C


@njit(cache=True, inline="always")
def calculate_albedo(
    snow_water_equivalent_m: npt.NDArray[np.float32],
    albedo_min: np.float32,
    albedo_max: np.float32,
    albedo_decay_coefficient: np.float32,
) -> npt.NDArray[np.float32]:
    """
    Calculate snow albedo based on snow water equivalent using a cached lookup table.

    This function uses nearest-neighbor lookup from a precomputed table to avoid repeated
    exponential calculations, improving performance for varying SWE values.

    Args:
        snow_water_equivalent_m: Current snow water equivalent (m).
        albedo_min: Minimum albedo (dimensionless).
        albedo_max: Maximum albedo (dimensionless).
        albedo_decay_coefficient: Albedo decay coefficient (per mm of SWE).

    Returns:
        Snow albedo (dimensionless).
    """
    # Compute the albedo table for the given parameters (cached implicitly via Numba)
    albedo_table = albedo_min + (albedo_max - albedo_min) * np.exp(
        -albedo_decay_coefficient * SWE_RANGE_M * np.float32(1000.0)
    ).astype(np.float32)

    # Find nearest index for input SWE values
    # Clamp SWE to the table range (0 to 10m)
    clamped_swe = np.clip(snow_water_equivalent_m, np.float32(0.0), np.float32(10.0))
    indices = np.round(clamped_swe * 100).astype(np.int32)
    return albedo_table[indices]


@njit(cache=True, inline="always")
def calculate_turbulent_fluxes(
    air_temperature_C: npt.NDArray[np.float32],
    snow_surface_temperature_C: npt.NDArray[np.float32],
    vapor_pressure_air_Pa: npt.NDArray[np.float32],
    air_pressure_Pa: npt.NDArray[np.float32],
    wind_speed_m_per_s: npt.NDArray[np.float32],
    bulk_transfer_coefficient: np.float32,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Calculate sensible and latent heat fluxes, and sublimation/deposition rate.

    Args:
        air_temperature_C: Air temperature (°C).
        snow_surface_temperature_C: Snow surface temperature (°C).
        vapor_pressure_air_Pa: Vapor pressure of the air (Pa).
        air_pressure_Pa: Air pressure (Pa).
        wind_speed_m_per_s: Wind speed (m/s).
        bulk_transfer_coefficient: Bulk transfer coefficient for heat and moisture.

    Returns:
        A tuple of sensible heat flux (W/m²), latent heat flux (W/m²),
        and sublimation/deposition rate (m/hour, negative for sublimation).
    """
    # Constants
    SPECIFIC_HEAT_AIR_J_PER_KG_K = np.float32(1005.0)  # Specific heat of air
    LATENT_HEAT_VAPORIZATION_J_PER_KG = np.float32(
        2.501e6  # Latent heat of vaporization (for liquid)
    )
    LATENT_HEAT_SUBLIMATION_J_PER_KG = np.float32(
        2.834e6
    )  # Latent heat of sublimation (for ice)
    GAS_CONSTANT_DRY_AIR_J_PER_KG_K = np.float32(287.058)  # Gas constant for dry air
    DENSITY_WATER_KG_PER_M3 = np.float32(1000.0)

    # Calculate air density (rho) using the ideal gas law: rho = P / (R * T)
    air_temperature_K = air_temperature_C + KELVIN_OFFSET
    air_density_kg_per_m3 = air_pressure_Pa / (
        GAS_CONSTANT_DRY_AIR_J_PER_KG_K * air_temperature_K
    )

    # Sensible Heat Flux (Q_S)
    # Based on bulk aerodynamic formula: Q_s = rho * c_p * C_h * U * (T_air - T_surf)
    sensible_heat_flux_W_per_m2 = (
        air_density_kg_per_m3
        * SPECIFIC_HEAT_AIR_J_PER_KG_K
        * bulk_transfer_coefficient
        * wind_speed_m_per_s
        * (air_temperature_C - snow_surface_temperature_C)
    )

    # Latent Heat Flux (Q_L)
    # Based on bulk aerodynamic formula: Q_l = rho * L * C_e * U * (q_air - q_surf)
    # We determine whether to use latent heat of vaporization or sublimation based on air temp.
    # This is a simplification; a more complex model might check snow surface temp.
    latent_heat_J_per_kg = np.where(
        air_temperature_C >= np.float32(0.0),
        LATENT_HEAT_VAPORIZATION_J_PER_KG,
        LATENT_HEAT_SUBLIMATION_J_PER_KG,
    )

    # Saturation vapor pressure at snow surface (e_surf)
    # Use Buck's equation for saturation vapor pressure over ice
    e_surf = np.float32(611.15) * np.exp(
        (np.float32(22.46) * snow_surface_temperature_C)
        / (np.float32(272.62) + snow_surface_temperature_C)
    )

    # Calculate specific humidity (q = 0.622 * e / (p - 0.378 * e))
    # 0.622 is the ratio of molar masses of water vapor and dry air.
    specific_humidity_air = (np.float32(0.622) * vapor_pressure_air_Pa) / (
        air_pressure_Pa - np.float32(0.378) * vapor_pressure_air_Pa
    )
    specific_humidity_surface = (np.float32(0.622) * e_surf) / (
        air_pressure_Pa - np.float32(0.378) * e_surf
    )

    latent_heat_flux_W_per_m2 = (
        air_density_kg_per_m3
        * latent_heat_J_per_kg
        * bulk_transfer_coefficient
        * wind_speed_m_per_s
        * (specific_humidity_air - specific_humidity_surface)
    )

    # Sublimation/Deposition Mass Flux
    # E = Q_l / L, where E is mass flux in kg/m²/s
    # Convert to m/hour: (kg/m²/s) / (kg/m³) * (s/hour) = m/hour
    sublimation_deposition_rate_m_per_hour = (
        latent_heat_flux_W_per_m2 / latent_heat_J_per_kg
    ) * (np.float32(3600.0) / DENSITY_WATER_KG_PER_M3)

    return (
        sensible_heat_flux_W_per_m2,
        latent_heat_flux_W_per_m2,
        sublimation_deposition_rate_m_per_hour,
    )


@njit(cache=True, inline="always")
def calculate_melt(
    air_temperature_C: npt.NDArray[np.float32],
    snow_surface_temperature_C: npt.NDArray[np.float32],
    snow_water_equivalent_m: npt.NDArray[np.float32],
    shortwave_radiation_W_per_m2: npt.NDArray[np.float32],
    downward_longwave_radiation_W_per_m2: npt.NDArray[np.float32],
    vapor_pressure_air_Pa: npt.NDArray[np.float32],
    air_pressure_Pa: npt.NDArray[np.float32],
    wind_speed_m_per_s: npt.NDArray[np.float32],
    albedo_min: np.float32 = np.float32(0.4),
    albedo_max: np.float32 = np.float32(0.9),
    albedo_decay_coefficient: np.float32 = np.float32(0.01),
    snow_radiation_coefficient: np.float32 = np.float32(1.0),
    bulk_transfer_coefficient: np.float32 = np.float32(0.0015),
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """
    Calculate snow melt based on a simplified energy balance model.

    Args:
        air_temperature_C: Air temperature (°C).
        snow_surface_temperature_C: Snow surface temperature (°C).
        snow_water_equivalent_m: Current snow water equivalent (m).
        shortwave_radiation_W_per_m2: Shortwave radiation (W/m²).
        downward_longwave_radiation_W_per_m2: Downward longwave radiation (W/m²).
        vapor_pressure_air_Pa: Vapor pressure of the air (Pa).
        air_pressure_Pa: Air pressure (Pa).
        wind_speed_m_per_s: Wind speed (m/s).
        albedo_min: Minimum albedo.
        albedo_max: Maximum albedo.
        albedo_decay_coefficient: Albedo decay coefficient.
        snow_radiation_coefficient: Snow radiation coefficient.
        bulk_transfer_coefficient: Bulk transfer coefficient for heat and moisture.

    Returns:
        A tuple of melt rate (m/hour), sublimation/deposition rate (m/hour),
        updated snow water equivalent (m), net shortwave radiation (W/m²),
        upward longwave radiation (W/m²), sensible heat flux (W/m²),
        and latent heat flux (W/m²).
    """
    # Calculate turbulent fluxes and sublimation/deposition
    (
        sensible_heat_flux_W_per_m2,
        latent_heat_flux_W_per_m2,
        sublimation_deposition_rate_m_per_hour,
    ) = calculate_turbulent_fluxes(
        air_temperature_C,
        snow_surface_temperature_C,
        vapor_pressure_air_Pa,
        air_pressure_Pa,
        wind_speed_m_per_s,
        bulk_transfer_coefficient,
    )

    # Update SWE with sublimation/deposition, ensuring it doesn't go below zero.
    # Sublimation is negative, deposition is positive.
    swe_after_sublimation_m = (
        snow_water_equivalent_m + sublimation_deposition_rate_m_per_hour
    )
    swe_after_sublimation_m = np.maximum(np.float32(0.0), swe_after_sublimation_m)

    # Radiation Balance
    # Net shortwave radiation
    albedo = calculate_albedo(
        swe_after_sublimation_m, albedo_min, albedo_max, albedo_decay_coefficient
    )
    net_shortwave_radiation_W_per_m2 = (
        (np.float32(1.0) - albedo)
        * snow_radiation_coefficient
        * shortwave_radiation_W_per_m2
    )

    # Net longwave radiation
    snow_surface_temp_K = snow_surface_temperature_C + KELVIN_OFFSET
    upward_longwave_radiation_W_per_m2 = (
        SNOW_EMISSIVITY * STEFAN_BOLTZMANN_CONSTANT * (snow_surface_temp_K**4)
    )
    net_longwave_radiation_W_per_m2 = (
        downward_longwave_radiation_W_per_m2 - upward_longwave_radiation_W_per_m2
    )

    net_radiation_W_per_m2 = (
        net_shortwave_radiation_W_per_m2 + net_longwave_radiation_W_per_m2
    )

    # Total energy available for melt (W/m²)
    # Latent heat flux is included here. If it's negative (sublimation), it reduces
    # the energy available for melt. If positive (deposition), it adds energy.
    total_energy_flux_W_per_m2 = (
        net_radiation_W_per_m2 + sensible_heat_flux_W_per_m2 + latent_heat_flux_W_per_m2
    )
    # Ensure melt only occurs when energy is positive
    total_energy_flux_W_per_m2 = np.maximum(np.float32(0.0), total_energy_flux_W_per_m2)

    # Convert energy flux from W/m² to m/hour of melt
    # 1 W/m² = 1 J/s/m²
    # Latent heat of fusion = 334,000 J/kg
    # 1 J/s/m² * 3600 s/hr / (334,000 J/kg * 1000 kg/m³) = m/hour
    conversion_factor = np.float32(3600.0) / (np.float32(334000.0) * np.float32(1000.0))
    potential_melt_m_per_hour = total_energy_flux_W_per_m2 * conversion_factor

    # Actual melt is limited by the available snow after sublimation
    snow_melt_rate_m_per_hour = np.minimum(
        potential_melt_m_per_hour, swe_after_sublimation_m
    )
    updated_snow_water_equivalent_m = (
        swe_after_sublimation_m - snow_melt_rate_m_per_hour
    )

    return (
        snow_melt_rate_m_per_hour,
        sublimation_deposition_rate_m_per_hour,
        updated_snow_water_equivalent_m,
        net_shortwave_radiation_W_per_m2,
        upward_longwave_radiation_W_per_m2,
        sensible_heat_flux_W_per_m2,
        latent_heat_flux_W_per_m2,
    )


@njit(cache=True, inline="always")
def handle_refreezing(
    snow_surface_temperature_C: npt.NDArray[np.float32],
    liquid_water_in_snow_m: npt.NDArray[np.float32],
    rainfall_m_per_hour: npt.NDArray[np.float32],
    snow_water_equivalent_m: npt.NDArray[np.float32],
    activate_layer_thickness_m: np.float32,
    max_refreezing_rate_m_per_hour: np.float32 = np.float32(0.01),
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Handle refreezing of liquid water in the snow pack based on energy balance.

    This function uses an active thermal layer to approximate refreezing dynamics in
    deep snowpacks, preventing the entire cold content of a deep glacier from
    unrealistically refreezing all surface melt.

    Args:
        snow_surface_temperature_C: Current snow surface temperature (°C).
        liquid_water_in_snow_m: Current liquid water in snow (m).
        rainfall_m_per_hour: Rainfall rate (m/hour).
        snow_water_equivalent_m: Current snow water equivalent (m).
        activate_layer_thickness_m: Thickness of the active thermal layer for refreezing (m).
        max_refreezing_rate_m_per_hour: Upper bound on refreezing per time step (m/hour) to
            avoid unrealistically freezing all liquid water when the active layer is very thick.

    Returns:
        A tuple of refreezing rate (m/hour), updated snow water equivalent (m), and updated liquid water (m).
    """
    # Constants
    SPECIFIC_HEAT_ICE_J_PER_KG_K = np.float32(2108.0)  # Specific heat of ice
    LATENT_HEAT_FUSION_J_PER_KG = np.float32(334000.0)  # Latent heat of fusion
    DENSITY_WATER_KG_PER_M3 = np.float32(1000.0)

    # Determine the depth of the snowpack to consider for refreezing.
    # This prevents the entire cold content of a very deep snowpack (glacier)
    # from refreezing all surface melt, which is more realistic.
    active_swe_for_refreezing_m = np.minimum(
        snow_water_equivalent_m, activate_layer_thickness_m
    )

    # Energy required to bring the active snow layer to 0°C (J/m²)
    cold_content_J_per_m2 = (
        -snow_surface_temperature_C
        * active_swe_for_refreezing_m
        * DENSITY_WATER_KG_PER_M3
        * SPECIFIC_HEAT_ICE_J_PER_KG_K
    )

    # Potential refreezing based on cold content (m/hour)
    potential_refreezing_m_per_hour = cold_content_J_per_m2 / (
        LATENT_HEAT_FUSION_J_PER_KG * DENSITY_WATER_KG_PER_M3
    )
    potential_refreezing_m_per_hour = np.where(
        snow_surface_temperature_C < np.float32(0.0),
        potential_refreezing_m_per_hour,
        np.float32(0.0),
    )

    # Liquid water available for refreezing within this time step consists of
    # the stored liquid water and incoming rainfall (both expressed here on an
    # hourly step). However, to avoid unrealistically freezing "all" water when
    # the active layer is very thick (large cold content), we cap the refreezing
    # rate per time step with max_refreezing_rate_m_per_hour.
    total_liquid_water_m = liquid_water_in_snow_m + rainfall_m_per_hour
    refreezing_capacity_m_per_hour = np.minimum(
        total_liquid_water_m, max_refreezing_rate_m_per_hour
    )
    actual_refreezing_m_per_hour = np.minimum(
        potential_refreezing_m_per_hour, refreezing_capacity_m_per_hour
    )
    updated_snow_water_equivalent_m = (
        snow_water_equivalent_m + actual_refreezing_m_per_hour
    )
    updated_liquid_water_m = total_liquid_water_m - actual_refreezing_m_per_hour
    return (
        actual_refreezing_m_per_hour,
        updated_snow_water_equivalent_m,
        updated_liquid_water_m,
    )


@njit(cache=True, inline="always")
def calculate_runoff(
    liquid_water_in_snow_m: npt.NDArray[np.float32],
    snow_water_equivalent_m: npt.NDArray[np.float32],
    water_holding_capacity_fraction: np.float32,
    activate_layer_thickness_m: np.float32,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Calculate runoff from the snow pack based on water holding capacity.

    Args:
        liquid_water_in_snow_m: Current liquid water in snow (m).
        snow_water_equivalent_m: Current snow water equivalent (m).
        water_holding_capacity_fraction: Water holding capacity fraction.
        activate_layer_thickness_m: Thickness of the active layer for water holding capacity (m).

    Returns:
        A tuple of runoff rate (m/hour) and updated liquid water (m).
    """
    max_water_content_m = (
        np.minimum(snow_water_equivalent_m, activate_layer_thickness_m)
        * water_holding_capacity_fraction
    )
    runoff_rate_m_per_hour = np.maximum(
        np.float32(0.0), liquid_water_in_snow_m - max_water_content_m
    )
    updated_liquid_water_m = liquid_water_in_snow_m - runoff_rate_m_per_hour
    return runoff_rate_m_per_hour, updated_liquid_water_m


@njit(cache=True, parallel=True)
def snow_model(
    precipitation_rate_kg_per_m2_per_s: npt.NDArray[np.float32],
    air_temperature_C: npt.NDArray[np.float32],
    snow_water_equivalent_m: npt.NDArray[np.float32],
    liquid_water_in_snow_m: npt.NDArray[np.float32],
    snow_temperature_C: npt.NDArray[np.float32],
    shortwave_radiation_W_per_m2: npt.NDArray[np.float32],
    downward_longwave_radiation_W_per_m2: npt.NDArray[np.float32],
    vapor_pressure_air_Pa: npt.NDArray[np.float32],
    air_pressure_Pa: npt.NDArray[np.float32],
    wind_speed_m_per_s: npt.NDArray[np.float32],
    snowfall_threshold_temperature_C: np.float32 = np.float32(0.0),
    water_holding_capacity_fraction: np.float32 = np.float32(0.1),
    albedo_min: np.float32 = np.float32(0.4),
    albedo_max: np.float32 = np.float32(0.9),
    albedo_decay_coefficient: np.float32 = np.float32(0.01),
    snow_radiation_coefficient: np.float32 = np.float32(1.0),
    bulk_transfer_coefficient: np.float32 = np.float32(0.0015),
    activate_layer_thickness_m: np.float32 = np.float32(0.2),
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """
    Calculate snow accumulation and melt based on a simple energy balance model.

    This model is a simplified Python implementation based on the Wflow.jl snow model.
    It tracks snow pack depth, water content, and temperature, and calculates melt,
    refreezing, and outflow.

    Notes:
        The model state (snow_water_equivalent_m, liquid_water_in_snow_m, snow_temperature_C) must be preserved
        between time steps.

    Args:
        precipitation_rate_kg_per_m2_per_s: Precipitation rate (kg/m²/s).
        air_temperature_C: Air temperature (°C).
        snow_water_equivalent_m: Snow water equivalent from the previous time step (m).
        liquid_water_in_snow_m: Liquid water content in the snow pack from the previous time step (m).
        snow_temperature_C: Temperature of the snow pack from the previous time step (°C).
        shortwave_radiation_W_per_m2: Shortwave radiation (W/m²).
        downward_longwave_radiation_W_per_m2: Downward longwave radiation (W/m²).
        vapor_pressure_air_Pa: Vapor pressure of the air (Pa).
        air_pressure_Pa: Air pressure (Pa).
        wind_speed_m_per_s: Wind speed (m/s).
        snowfall_threshold_temperature_C: Threshold temperature for snowfall/rainfall (°C).
        water_holding_capacity_fraction: Water holding capacity of the snow pack as a fraction of snow water equivalent.
        albedo_min: Minimum albedo.
        albedo_max: Maximum albedo.
        albedo_decay_coefficient: Albedo decay coefficient.
        snow_radiation_coefficient: Snow radiation coefficient.
        bulk_transfer_coefficient: Bulk transfer coefficient for heat and moisture.
        activate_layer_thickness_m: Thickness of the active layer for water holding capacity and refreezing (m).

    Returns:
        A tuple containing:
        - new_snow_water_equivalent_m: Updated snow water equivalent (m).
        - new_liquid_water_in_snow_m: Updated liquid water content in snow pack (m).
        - new_snow_temperature_C: Updated snow pack temperature (°C).
        - snow_melt_rate_m_per_hour: Rate of snow melt (m/hour).
        - sublimation_deposition_rate_m_per_hour: Rate of sublimation/deposition (m/hour, negative for sublimation).
        - runoff_rate_m_per_hour: Combined rainfall and snowmelt runoff (m/hour).
        - actual_refreezing_m_per_hour: Rate of refreezing (m/hour).
        - snow_surface_temperature_C: Snow surface temperature (°C).
        - net_shortwave_radiation_W_per_m2: Net shortwave radiation (W/m²).
        - upward_longwave_radiation_W_per_m2: Upward longwave radiation (W/m²).
        - sensible_heat_flux_W_per_m2: Sensible heat flux (W/m²).
        - latent_heat_flux_W_per_m2: Latent heat flux (W/m²).
    """
    # Convert precipitation to meters per hour
    precipitation_m_per_hour = precipitation_rate_kg_per_m2_per_s * np.float32(
        3.6
    )  # kg/m²/s to m/hour

    # Discriminate between rainfall and snowfall
    snowfall_m_per_hour, rainfall_m_per_hour = discriminate_precipitation(
        precipitation_m_per_hour, air_temperature_C, snowfall_threshold_temperature_C
    )

    # Update snow pack temperature
    new_snow_temperature_C = update_snow_temperature(
        snow_water_equivalent_m,
        snow_temperature_C,
        snowfall_m_per_hour,
        air_temperature_C,
    )

    # Calculate snow surface temperature
    snow_surface_temperature_C = calculate_snow_surface_temperature(
        air_temperature_C,
        new_snow_temperature_C,
        snow_water_equivalent_m + snowfall_m_per_hour,
    )

    # Calculate snow melt
    (
        snow_melt_rate_m_per_hour,
        sublimation_deposition_rate_m_per_hour,
        swe_after_melt_m,
        net_shortwave_radiation_W_per_m2,
        upward_longwave_radiation_W_per_m2,
        sensible_heat_flux_W_per_m2,
        latent_heat_flux_W_per_m2,
    ) = calculate_melt(
        air_temperature_C,
        snow_surface_temperature_C,
        snow_water_equivalent_m + snowfall_m_per_hour,
        shortwave_radiation_W_per_m2,
        downward_longwave_radiation_W_per_m2,
        vapor_pressure_air_Pa,
        air_pressure_Pa,
        wind_speed_m_per_s,
        albedo_min,
        albedo_max,
        albedo_decay_coefficient,
        snow_radiation_coefficient,
        bulk_transfer_coefficient,
    )

    # Update snow temperature to 0 if melt occurs
    final_snow_temperature_C = np.where(
        snow_melt_rate_m_per_hour > 0, np.float32(0.0), new_snow_temperature_C
    )

    # Add melt to liquid water content
    liquid_water_after_melt_m = liquid_water_in_snow_m + snow_melt_rate_m_per_hour

    # Handle refreezing of liquid water
    (
        actual_refreezing_m_per_hour,
        swe_after_refreezing_m,
        liquid_water_after_refreezing_m,
    ) = handle_refreezing(
        snow_surface_temperature_C,
        liquid_water_after_melt_m,
        rainfall_m_per_hour,
        swe_after_melt_m,
        activate_layer_thickness_m,
    )
    # Refreezing releases latent heat and should therefore warm the snowpack.
    # We approximate this by distributing the released heat over the active
    # thermal layer mass (same thickness used for refreezing), and update the
    # bulk snow temperature accordingly, while capping at 0°C.
    SPECIFIC_HEAT_ICE_J_PER_KG_K = np.float32(2108.0)
    LATENT_HEAT_FUSION_J_PER_KG = np.float32(334000.0)
    DENSITY_WATER_KG_PER_M3 = np.float32(1000.0)

    # Heat released by refreezing in this time step (J/m²)
    heat_released_J_per_m2 = (
        actual_refreezing_m_per_hour
        * DENSITY_WATER_KG_PER_M3
        * LATENT_HEAT_FUSION_J_PER_KG
    )
    # Active mass considered for warming (kg/m²)
    active_swe_for_refreezing_m = np.minimum(swe_after_melt_m, np.float32(2.0))
    active_mass_kg_per_m2 = active_swe_for_refreezing_m * DENSITY_WATER_KG_PER_M3

    # Temperature rise, avoid division by zero for vanishing active mass
    delta_T_C = np.where(
        active_mass_kg_per_m2 > 0,
        heat_released_J_per_m2 / (active_mass_kg_per_m2 * SPECIFIC_HEAT_ICE_J_PER_KG_K),
        np.float32(0.0),
    )
    final_snow_temperature_C = np.minimum(
        np.float32(0.0), final_snow_temperature_C + delta_T_C
    )

    # Calculate runoff from the snow pack
    runoff_rate_m_per_hour, new_liquid_water_in_snow_m = calculate_runoff(
        liquid_water_after_refreezing_m,
        swe_after_refreezing_m,
        water_holding_capacity_fraction,
        activate_layer_thickness_m=activate_layer_thickness_m,
    )

    return (
        swe_after_refreezing_m,
        new_liquid_water_in_snow_m,
        final_snow_temperature_C,
        snow_melt_rate_m_per_hour,
        sublimation_deposition_rate_m_per_hour,
        runoff_rate_m_per_hour,
        actual_refreezing_m_per_hour,
        snow_surface_temperature_C,
        net_shortwave_radiation_W_per_m2,
        upward_longwave_radiation_W_per_m2,
        sensible_heat_flux_W_per_m2,
        latent_heat_flux_W_per_m2,
    )


@njit(cache=True, inline="always")
def calculate_snow_surface_temperature(
    air_temperature_C: npt.NDArray[np.float32],
    snow_temperature_C: npt.NDArray[np.float32],
    snow_water_equivalent_m: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Estimate snow surface temperature based on snowpack insulation.

    This function approximates the surface temperature as a weighted average between
    the bulk snow temperature and the air temperature, considering the thermal
    resistance of the snowpack.

    Args:
        air_temperature_C: Air temperature (°C).
        snow_temperature_C: Bulk temperature of the snowpack (°C).
        snow_water_equivalent_m: Snow water equivalent (m).

    Returns:
        Estimated snow surface temperature (°C), capped at 0°C.
    """
    # Constants for thermal conductivity calculation from density
    # Based on Sturm et al. (1997), k = 0.021 + 2.5 * (rho/1000)^2
    K_C0 = np.float32(0.021)  # W/m/K
    K_C1 = np.float32(2.5)  # W/m/K

    # Estimate snow density (kg/m³) from SWE (m). This is a simplification.
    # Assumes density increases with SWE, from fresh snow (~150) to firn (~550).
    snow_density_kg_per_m3 = np.minimum(
        np.float32(550.0), 150.0 + 400.0 * snow_water_equivalent_m
    )

    # Calculate thermal conductivity (k) in W/m/K
    snow_thermal_conductivity = (
        K_C0 + K_C1 * (snow_density_kg_per_m3 / np.float32(1000.0)) ** 2
    )

    # Estimate snow depth (m) from SWE and density
    snow_depth_m = snow_water_equivalent_m / (
        snow_density_kg_per_m3 / np.float32(1000.0)
    )

    # Calculate thermal resistance (R = depth / k)
    # Add a small epsilon to avoid division by zero for zero depth
    thermal_resistance = snow_depth_m / (snow_thermal_conductivity + np.float32(1e-9))

    # Weighting factor (0 to 1). High resistance -> weight closer to 1 (air temp).
    # The formula is a simplification of the heat transfer solution.
    weight = np.tanh(thermal_resistance)

    # Weighted average of air and bulk snow temperature
    surface_temp_C = (
        weight * air_temperature_C + (np.float32(1.0) - weight) * snow_temperature_C
    )

    # Surface temperature cannot exceed melting point
    return np.minimum(np.float32(0.0), surface_temp_C)
