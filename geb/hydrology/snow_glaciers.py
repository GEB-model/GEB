"""Functions to calculate snow accumulation and melt."""

import numpy as np
from numba import njit

STEFAN_BOLTZMANN_CONSTANT = np.float32(5.670374419e-8)  # W m⁻² K⁻⁴
SNOW_EMISSIVITY = np.float32(0.99)  # Emissivity of snow surface
KELVIN_OFFSET = np.float32(273.15)


@njit(cache=True, inline="always")
def discriminate_precipitation(
    precipitation_m_per_hour: np.float32,
    air_temperature_C: np.float32,
    snowfall_threshold_temperature_C: np.float32,
) -> tuple[np.float32, np.float32]:
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
    snowfall_m_per_hour = (
        np.float32(precipitation_m_per_hour) if is_snow else np.float32(0.0)
    )
    rainfall_m_per_hour = (
        np.float32(0.0) if is_snow else np.float32(precipitation_m_per_hour)
    )
    return snowfall_m_per_hour, rainfall_m_per_hour


@njit(cache=True, inline="always")
def update_snow_temperature(
    snow_water_equivalent_m: np.float32,
    snow_temperature_C: np.float32,
    snowfall_m_per_hour: np.float32,
    air_temperature_C: np.float32,
) -> np.float32:
    """
    Update the snow pack temperature based on new snowfall and air temperature.

    This function first calculates the mixed temperature of the old snowpack and
    the new snowfall. Then, it applies a conductive heat transfer adjustment
    based on the air temperature using a tau-based relaxation approach.

    Args:
        snow_water_equivalent_m: Current snow water equivalent (m).
        snow_temperature_C: Current snow temperature (°C).
        snowfall_m_per_hour: Snowfall rate (m/hour).
        air_temperature_C: Air temperature (°C).

    Returns:
        Updated snow temperature (°C).
    """
    # Constants for thermal properties
    SPECIFIC_HEAT_ICE_J_PER_KG_K = np.float32(2108.0)
    PI = np.float32(np.pi)

    # Add new snowfall to SWE
    new_swe_m = snow_water_equivalent_m + snowfall_m_per_hour

    # Avoid division by zero if there's no snow
    mixed_temp_C = (
        (
            snow_temperature_C * snow_water_equivalent_m
            + air_temperature_C * snowfall_m_per_hour
        )
        / new_swe_m
        if new_swe_m > 0
        else air_temperature_C
    )

    # Conductive heat transfer adjustment from the atmosphere
    # Estimate snow density (kg/m³) from SWE (m).
    snow_density_kg_per_m3 = np.minimum(
        np.float32(550.0), np.float32(150.0) + np.float32(400.0) * new_swe_m
    )

    # Calculate thermal conductivity (k) in W/m/K
    snow_thermal_conductivity = (
        np.float32(0.021)
        + np.float32(2.5) * (snow_density_kg_per_m3 / np.float32(1000.0)) ** 2
    )

    # Estimate snow depth (m) from SWE and density
    snow_depth_m = new_swe_m / (snow_density_kg_per_m3 / np.float32(1000.0))
    snow_depth_m = np.maximum(snow_depth_m, np.float32(0.01))  # Min depth of 1cm

    # Calculate thermal diffusivity (alpha) in m²/s
    thermal_diffusivity_m2_per_s = snow_thermal_conductivity / (
        snow_density_kg_per_m3 * SPECIFIC_HEAT_ICE_J_PER_KG_K
    )

    # Calculate characteristic time for thermal adjustment (tau) in seconds
    # this is a simplified version where 1 / pi2 appoximates the lowest-order
    # Fourier term.
    tau_s = (snow_depth_m**2) / (
        PI**2 * thermal_diffusivity_m2_per_s + np.float32(1e-12)
    )

    # Fraction of adjustment towards air temperature over one hour (3600s)
    adjustment_fraction = np.float32(1.0) - np.exp(np.float32(-3600.0) / tau_s)

    # Apply adjustment to the mixed temperature
    final_temp_C = mixed_temp_C + adjustment_fraction * (
        air_temperature_C - mixed_temp_C
    )

    # Final temperature cannot exceed the melting point
    return np.minimum(final_temp_C, np.float32(0.0))


@njit(cache=True, inline="always")
def calculate_albedo(
    snow_water_equivalent_m: np.float32,
    albedo_min: np.float32,
    albedo_max: np.float32,
    albedo_decay_coefficient: np.float32,
) -> np.float32:
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
    # Clamp SWE to the table range (0 to 10m)
    clamped_swe = max(np.float32(0.0), min(snow_water_equivalent_m, np.float32(10.0)))
    return albedo_min + (albedo_max - albedo_min) * np.exp(
        -albedo_decay_coefficient * clamped_swe * np.float32(1000.0)
    )


@njit(cache=True, inline="always")
def calculate_turbulent_fluxes(
    air_temperature_C: np.float32,
    snow_surface_temperature_C: np.float32,
    vapor_pressure_air_Pa: np.float32,
    air_pressure_Pa: np.float32,
    wind_10m_m_per_s: np.float32,
    bulk_transfer_coefficient: np.float32,
) -> tuple[np.float32, np.float32, np.float32]:
    """
    Calculate sensible and latent heat fluxes, and sublimation/deposition rate.

    Args:
        air_temperature_C: Air temperature (°C).
        snow_surface_temperature_C: Snow surface temperature (°C).
        vapor_pressure_air_Pa: Vapor pressure of the air (Pa).
        air_pressure_Pa: Air pressure (Pa).
        wind_10m_m_per_s: Wind speed (m/s).
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
        * wind_10m_m_per_s
        * (air_temperature_C - snow_surface_temperature_C)
    )

    # Latent Heat Flux (Q_L)
    # Based on bulk aerodynamic formula: Q_l = rho * L * C_e * U * (q_air - q_surf)
    # Use latent heat of sublimation for snow (ice) surfaces, vaporization for liquid.
    # This is more physically realistic than using air temperature.
    latent_heat_J_per_kg = (
        LATENT_HEAT_VAPORIZATION_J_PER_KG
        if snow_surface_temperature_C >= np.float32(0.0)
        else LATENT_HEAT_SUBLIMATION_J_PER_KG
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
        * wind_10m_m_per_s
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
    air_temperature_C: np.float32,
    snow_surface_temperature_C: np.float32,
    snow_water_equivalent_m: np.float32,
    shortwave_radiation_W_per_m2: np.float32,
    downward_longwave_radiation_W_per_m2: np.float32,
    vapor_pressure_air_Pa: np.float32,
    air_pressure_Pa: np.float32,
    wind_10m_m_per_s: np.float32,
    albedo_min: np.float32 = np.float32(0.4),
    albedo_max: np.float32 = np.float32(0.9),
    albedo_decay_coefficient: np.float32 = np.float32(0.01),
    snow_radiation_coefficient: np.float32 = np.float32(1.0),
    bulk_transfer_coefficient: np.float32 = np.float32(0.0015),
) -> tuple[
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
]:
    """
    Calculate potential snow melt and sublimation based on energy balance.

    This function calculates fluxes but does not alter the state.

    Returns:
        A tuple of potential melt rate (m/hour), sublimation/deposition rate (m/hour),
        net shortwave radiation (W/m²), upward longwave radiation (W/m²),
        sensible heat flux (W/m²), and latent heat flux (W/m²).
    """
    # If there is no snow, there can be no sublimation or deposition from the snowpack.
    if snow_water_equivalent_m == np.float32(0.0):
        return (
            np.float32(0.0),  # potential_melt_m_per_hour
            np.float32(0.0),  # sublimation_deposition_rate_m_per_hour
            np.float32(0.0),  # net_shortwave_radiation_W_per_m2
            np.float32(0.0),  # upward_longwave_radiation_W_per_m2
            np.float32(0.0),  # sensible_heat_flux_W_per_m2
            np.float32(0.0),  # latent_heat_flux_W_per_m2
        )

    # Calculate turbulent fluxes (sensible, latent) and the resulting mass flux
    # from sublimation/deposition.
    (
        sensible_heat_flux_W_per_m2,
        latent_heat_flux_W_per_m2,
        sublimation_deposition_rate_m_per_hour,
    ) = calculate_turbulent_fluxes(
        air_temperature_C,
        snow_surface_temperature_C,
        vapor_pressure_air_Pa,
        air_pressure_Pa,
        wind_10m_m_per_s,
        bulk_transfer_coefficient,
    )

    # Temporarily update SWE with sublimation/deposition to get the correct albedo.
    swe_after_sublimation_m = (
        snow_water_equivalent_m + sublimation_deposition_rate_m_per_hour
    )
    swe_after_sublimation_m = np.maximum(np.float32(0.0), swe_after_sublimation_m)

    # Radiation Balance
    albedo = calculate_albedo(
        swe_after_sublimation_m, albedo_min, albedo_max, albedo_decay_coefficient
    )
    net_shortwave_radiation_W_per_m2 = (
        (np.float32(1.0) - albedo)
        * snow_radiation_coefficient
        * shortwave_radiation_W_per_m2
    )
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

    # Energy available for melt is from net radiation and sensible heat.
    # Latent heat is NOT included here because its energy is realized as a mass
    # change (sublimation/deposition), which is handled separately.
    total_energy_flux_W_per_m2 = net_radiation_W_per_m2 + sensible_heat_flux_W_per_m2
    total_energy_flux_W_per_m2 = np.maximum(np.float32(0.0), total_energy_flux_W_per_m2)

    # Convert energy flux to potential melt rate
    conversion_factor = np.float32(3600.0) / (np.float32(334000.0) * np.float32(1000.0))
    potential_melt_m_per_hour = total_energy_flux_W_per_m2 * conversion_factor

    return (
        potential_melt_m_per_hour,
        sublimation_deposition_rate_m_per_hour,
        net_shortwave_radiation_W_per_m2,
        upward_longwave_radiation_W_per_m2,
        sensible_heat_flux_W_per_m2,
        latent_heat_flux_W_per_m2,
    )


@njit(cache=True, inline="always")
def handle_refreezing(
    snow_surface_temperature_C: np.float32,
    liquid_water_in_snow_m: np.float32,
    snow_water_equivalent_m: np.float32,
    activate_layer_thickness_m: np.float32,
    max_refreezing_rate_m_per_hour: np.float32 = np.float32(0.01),
) -> tuple[np.float32, np.float32, np.float32]:
    """
    Handle refreezing of liquid water in the snow pack based on energy balance.

    This function uses an active thermal layer to approximate refreezing dynamics in
    deep snowpacks, preventing the entire cold content of a deep glacier from
    unrealistically refreezing all surface melt.

    Args:
        snow_surface_temperature_C: Current snow surface temperature (°C).
        liquid_water_in_snow_m: Current liquid water in snow (m).
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
    potential_refreezing_m_per_hour = (
        potential_refreezing_m_per_hour
        if snow_surface_temperature_C < np.float32(0.0)
        else np.float32(0.0)
    )

    # Liquid water available for refreezing is the stored liquid water.
    # To avoid unrealistically freezing "all" water when the active layer is
    # very thick (large cold content), we cap the refreezing rate per time step.
    refreezing_capacity_m_per_hour = np.minimum(
        liquid_water_in_snow_m, max_refreezing_rate_m_per_hour
    )
    actual_refreezing_m_per_hour = np.minimum(
        potential_refreezing_m_per_hour, refreezing_capacity_m_per_hour
    )
    updated_snow_water_equivalent_m = (
        snow_water_equivalent_m + actual_refreezing_m_per_hour
    )
    updated_liquid_water_m = liquid_water_in_snow_m - actual_refreezing_m_per_hour
    return (
        actual_refreezing_m_per_hour,
        updated_snow_water_equivalent_m,
        updated_liquid_water_m,
    )


@njit(cache=True, inline="always")
def calculate_runoff(
    liquid_water_in_snow_m: np.float32,
    snow_water_equivalent_m: np.float32,
    water_holding_capacity_fraction: np.float32,
    activate_layer_thickness_m: np.float32,
) -> tuple[np.float32, np.float32]:
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


@njit(cache=True, inline="always")
def warm_snowpack(
    snow_temperature_C: np.float32,
    snow_water_equivalent_m: np.float32,
    energy_J_per_m2_per_s: np.float32,
) -> np.float32:
    """
    Warm or cool the snowpack based on a given energy flux over one second.

    This function calculates the temperature change of the snowpack resulting from a
    net energy input or deficit. The temperature is capped at 0°C for warming.

    Args:
        snow_temperature_C: Current bulk temperature of the snowpack (°C).
        snow_water_equivalent_m: Snow water equivalent (m), used to determine thermal mass.
        energy_J_per_m2_per_s: Net energy flux into (+) or out of (-) the snowpack
            over one second (J/m²/s).

    Returns:
        The new bulk snow temperature (°C).
    """
    SPECIFIC_HEAT_ICE_J_PER_KG_K = np.float32(2108.0)
    DENSITY_WATER_KG_PER_M3 = np.float32(1000.0)

    # Thermal mass of the snowpack (kg/m²)
    snow_mass_kg_per_m2 = snow_water_equivalent_m * DENSITY_WATER_KG_PER_M3

    # Temperature change (delta T) = Energy / (mass * specific heat)
    # Avoid division by zero if there is no snow mass
    delta_T_C = (
        energy_J_per_m2_per_s / (snow_mass_kg_per_m2 * SPECIFIC_HEAT_ICE_J_PER_KG_K)
        if snow_mass_kg_per_m2 > 0
        else np.float32(0.0)
    )

    new_temp_C = snow_temperature_C + delta_T_C

    # The bulk temperature of the snowpack cannot exceed the melting point.
    return np.minimum(np.float32(0.0), new_temp_C)


@njit(cache=True, inline="always")
def snow_model(
    pr_kg_per_m2_per_s: np.float32,
    air_temperature_C: np.float32,
    snow_water_equivalent_m: np.float32,
    liquid_water_in_snow_m: np.float32,
    snow_temperature_C: np.float32,
    shortwave_radiation_W_per_m2: np.float32,
    downward_longwave_radiation_W_per_m2: np.float32,
    vapor_pressure_air_Pa: np.float32,
    air_pressure_Pa: np.float32,
    wind_10m_m_per_s: np.float32,
    snowfall_threshold_temperature_C: np.float32 = np.float32(0.0),
    water_holding_capacity_fraction: np.float32 = np.float32(0.1),
    albedo_min: np.float32 = np.float32(0.4),
    albedo_max: np.float32 = np.float32(0.9),
    albedo_decay_coefficient: np.float32 = np.float32(0.01),
    snow_radiation_coefficient: np.float32 = np.float32(1.0),
    bulk_transfer_coefficient: np.float32 = np.float32(0.0015),
    activate_layer_thickness_m: np.float32 = np.float32(0.2),
) -> tuple[
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
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
        pr_kg_per_m2_per_s: Precipitation rate (kg/m²/s).
        air_temperature_C: Air temperature (°C).
        snow_water_equivalent_m: Snow water equivalent from the previous time step (m).
        liquid_water_in_snow_m: Liquid water content in the snow pack from the previous time step (m).
        snow_temperature_C: Temperature of the snow pack from the previous time step (°C).
        shortwave_radiation_W_per_m2: Shortwave radiation (W/m²).
        downward_longwave_radiation_W_per_m2: Downward longwave radiation (W/m²).
        vapor_pressure_air_Pa: Vapor pressure of the air (Pa).
        air_pressure_Pa: Air pressure (Pa).
        wind_10m_m_per_s: Wind speed (m/s).
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
        - rainfall_m_per_hour: Rainfall rate (m).
        - snowfall_m_per_hour: Snowfall rate (m).
        - new_snow_water_equivalent_m: Updated snow water equivalent (m).
        - new_liquid_water_in_snow_m: Updated liquid water content in snow pack (m).
        - new_snow_temperature_C: Updated snow pack temperature (°C).
        - snow_melt_rate_m_per_hour: Rate of snow melt (m/hour).
        - melt_runoff_rate_m_per_hour: Runoff from snowmelt (m/hour).
        - direct_rainfall_m_per_hour: Direct rainfall runoff (m/hour).
        - sublimation_deposition_rate_m_per_hour: Rate of sublimation/deposition (m/hour, negative for sublimation).
        - actual_refreezing_m_per_hour: Rate of refreezing (m/hour).
        - snow_surface_temperature_C: Snow surface temperature (°C).
        - net_shortwave_radiation_W_per_m2: Net shortwave radiation (W/m²).
        - upward_longwave_radiation_W_per_m2: Upward longwave radiation (W/m²).
        - sensible_heat_flux_W_per_m2: Sensible heat flux (W/m²).
        - latent_heat_flux_W_per_m2: Latent heat flux (W/m²).
    """
    # Convert precipitation to meters per hour
    precipitation_m_per_hour = pr_kg_per_m2_per_s * np.float32(3.6)  # kg/m²/s to m/hour

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

    # The initial SWE for the melt calculation should be the sum of the previous
    # state's SWE and the new snowfall from this timestep.
    swe_for_melt_calculation = snow_water_equivalent_m + snowfall_m_per_hour

    # Calculate potential melt and sublimation fluxes based on the updated snowpack
    (
        potential_melt_m_per_hour,
        sublimation_deposition_rate_m_per_hour,
        net_shortwave_radiation_W_per_m2,
        upward_longwave_radiation_W_per_m2,
        sensible_heat_flux_W_per_m2,
        latent_heat_flux_W_per_m2,
    ) = calculate_melt(
        air_temperature_C,
        snow_surface_temperature_C,
        swe_for_melt_calculation,  # Use SWE after snowfall for flux calculation
        shortwave_radiation_W_per_m2,
        downward_longwave_radiation_W_per_m2,
        vapor_pressure_air_Pa,
        air_pressure_Pa,
        wind_10m_m_per_s,
        albedo_min,
        albedo_max,
        albedo_decay_coefficient,
        snow_radiation_coefficient,
        bulk_transfer_coefficient,
    )

    # Add new snowfall to the previous state's SWE
    swe_after_snowfall_m = snow_water_equivalent_m + snowfall_m_per_hour

    # Apply sublimation/deposition with water-balance limiter:
    # Do not allow sublimation to exceed available SWE in this timestep.
    # Positive values (deposition) are not limited here.
    applied_sublimation_deposition_rate_m_per_hour = np.maximum(
        -swe_after_snowfall_m, sublimation_deposition_rate_m_per_hour
    )

    swe_after_sublimation_m = (
        swe_after_snowfall_m + applied_sublimation_deposition_rate_m_per_hour
    )

    # Water balance check: sublimation/deposition
    water_after_sublimation_m = swe_after_sublimation_m + liquid_water_in_snow_m

    # Calculate actual melt, limited by available snow
    actual_melt_m_per_hour = np.minimum(
        potential_melt_m_per_hour, swe_after_sublimation_m
    )
    swe_after_melt_m = swe_after_sublimation_m - actual_melt_m_per_hour

    # Add meltwater to liquid water
    liquid_water_after_melt_m = liquid_water_in_snow_m + actual_melt_m_per_hour

    # Handle refreezing
    (
        actual_refreezing_m_per_hour,
        swe_after_refreezing_m,
        liquid_water_after_refreezing_m,
    ) = handle_refreezing(
        new_snow_temperature_C,
        liquid_water_after_melt_m,
        swe_after_melt_m,
        activate_layer_thickness_m,
    )

    # Update snow temperature from refreezing energy
    LATENT_HEAT_FUSION_J_PER_KG = np.float32(334000.0)
    DENSITY_WATER_KG_PER_M3 = np.float32(1000.0)
    refreezing_energy_J_per_m2_per_hour = (
        actual_refreezing_m_per_hour
        * DENSITY_WATER_KG_PER_M3
        * LATENT_HEAT_FUSION_J_PER_KG
    )
    # Convert to J/m²/s for the warm_snowpack function
    refreezing_energy_J_per_m2_per_s = refreezing_energy_J_per_m2_per_hour / np.float32(
        3600.0
    )
    final_snow_temperature_C = warm_snowpack(
        new_snow_temperature_C,
        swe_after_refreezing_m,
        refreezing_energy_J_per_m2_per_s,
    )

    # Calculate runoff from meltwater
    melt_runoff_m_per_hour, liquid_water_after_melt_runoff_m = calculate_runoff(
        liquid_water_after_refreezing_m,
        swe_after_refreezing_m,
        water_holding_capacity_fraction,
        activate_layer_thickness_m=activate_layer_thickness_m,
    )

    # Add rainfall
    liquid_water_with_rain_m = liquid_water_after_melt_runoff_m + rainfall_m_per_hour

    # Calculate runoff from rainfall
    rainfall_that_resulted_in_runoff_m_per_hour, new_liquid_water_in_snow_m = (
        calculate_runoff(
            liquid_water_with_rain_m,
            swe_after_refreezing_m,
            water_holding_capacity_fraction,
            activate_layer_thickness_m=activate_layer_thickness_m,
        )
    )

    return (
        rainfall_m_per_hour,
        snowfall_m_per_hour,
        swe_after_refreezing_m,
        new_liquid_water_in_snow_m,
        final_snow_temperature_C,
        actual_melt_m_per_hour,
        melt_runoff_m_per_hour,
        rainfall_that_resulted_in_runoff_m_per_hour,
        applied_sublimation_deposition_rate_m_per_hour,
        actual_refreezing_m_per_hour,
        snow_surface_temperature_C,
        net_shortwave_radiation_W_per_m2,
        upward_longwave_radiation_W_per_m2,
        sensible_heat_flux_W_per_m2,
        latent_heat_flux_W_per_m2,
    )


@njit(cache=True, inline="always")
def calculate_snow_surface_temperature(
    air_temperature_C: np.float32,
    snow_temperature_C: np.float32,
    snow_water_equivalent_m: np.float32,
) -> np.float32:
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
