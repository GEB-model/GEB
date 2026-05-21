"""Soil energy flow functions."""

import numpy as np
from numba import njit

from geb.geb_types import ArrayFloat32, Shape
from geb.workflows.numba_stack_array import stack_empty

from .constants import (
    C_MINERAL_VOLUMETRIC_J_PER_M3_K,
    KELVIN_OFFSET,
    LATENT_HEAT_FUSION_J_PER_KG,
    LATENT_HEAT_SUBLIMATION_J_PER_KG,
    LATENT_HEAT_VAPORIZATION_J_PER_KG,
    N_SOIL_LAYERS,
    RHO_MINERAL_KG_PER_M3,
    RHO_WATER_KG_PER_M3,
    SNOW_EMISSIVITY,
    SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K,
    STEFAN_BOLTZMANN_W_PER_M2_K4,
    THERMAL_CONDUCTIVITY_NON_QUARTZ_WATT_PER_MKELVIN,
    THERMAL_CONDUCTIVITY_QUARTZ_WATT_PER_MKELVIN,
    VOLUMETRIC_HEAT_CAPACITY_ICE_J_PER_M3_K,
    VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K,
)
from .potential_evapotranspiration import get_canopy_radiation_attenuation

N_SNOW_LAYERS: int = 2
_N_COUPLED_SURFACE_LAYERS: int = N_SOIL_LAYERS + N_SNOW_LAYERS
MIN_ACTIVE_SNOW_SWE_M: np.float64 = np.float64(1.0e-6)


@njit(cache=True, inline="always")
def calculate_snow_thermal_properties(
    snow_water_equivalent_m: np.float64,
) -> tuple[np.float32, np.float64, np.float32]:
    """Calculate snow density, depth, and thermal conductivity.

    This local helper mirrors the snow-module parameterization so the energy
    module can remain independent from snow module imports.

    Args:
        snow_water_equivalent_m: Snow water equivalent [m].

    Returns:
        Tuple of:
            - Snow density [kg/m3].
            - Snow depth [m].
            - Thermal conductivity [W/m/K].
    """
    snow_density_kg_per_m3: np.float32 = min(
        np.float32(550.0),
        np.float32(150.0)
        + np.float32(400.0)
        * np.float32(
            snow_water_equivalent_m,
        ),
    )

    # Based on Anderson (2019)
    # Full report: https://repository.library.noaa.gov/view/noaa/6392
    # Implementation in Community Firn Model:
    # https://github.com/UWGlaciology/CommunityFirnModel/blob/main/CFM_main/diffusion.py
    snow_thermal_conductivity: np.float32 = (
        np.float32(0.021)
        + np.float32(2.5) * (snow_density_kg_per_m3 / np.float32(1000.0)) ** 2
    )
    snow_depth_m: np.float64 = snow_water_equivalent_m / (
        snow_density_kg_per_m3 / np.float32(1000.0)
    )
    return snow_density_kg_per_m3, snow_depth_m, snow_thermal_conductivity


@njit(cache=True, inline="always")
def calculate_snow_thermal_conductivity_from_density(
    snow_density_kg_per_m3: np.float32,
) -> np.float32:
    """Calculate snow thermal conductivity from bulk density.

    Args:
        snow_density_kg_per_m3: Snow density (kg/m3).

    Returns:
        Snow thermal conductivity (W/m/K).
    """
    # See Sturm et al. 1997;  equation in abstract
    # https://doi.org/10.3189/S0022143000002781
    rho_g_per_cm3: np.float32 = snow_density_kg_per_m3 / np.float32(1000.0)
    if rho_g_per_cm3 < np.float32(0.156):
        return np.float32(0.023) + np.float32(0.234) * rho_g_per_cm3
    else:
        return (
            np.float32(0.138)
            - np.float32(1.01) * rho_g_per_cm3
            + np.float32(3.233) * (rho_g_per_cm3**2)
        )


@njit(cache=True, inline="always", fastmath=True)
def calculate_snow_net_radiation_flux(
    shortwave_radiation_W_per_m2: np.float32,
    longwave_radiation_W_per_m2: np.float32,
    snow_temperature_C: np.float32,
    total_snow_water_equivalent_m: np.float32,
    snow_albedo_min: np.float32 = np.float32(0.4),
    snow_albedo_max: np.float32 = np.float32(0.9),
    snow_albedo_decay_per_mm: np.float32 = np.float32(0.01),
    snow_radiation_coefficient: np.float32 = np.float32(1.0),
) -> tuple[np.float32, np.float32]:
    """Calculate net snow radiation flux and its derivative.

    Args:
        shortwave_radiation_W_per_m2: Incoming shortwave radiation (W/m2).
        longwave_radiation_W_per_m2: Incoming longwave radiation (W/m2).
        snow_temperature_C: Snow surface-layer temperature (C).
        total_snow_water_equivalent_m: Total snow water equivalent (m).
        snow_albedo_min: Minimum snow albedo (-).
        snow_albedo_max: Maximum snow albedo (-).
        snow_albedo_decay_per_mm: Snow albedo decay coefficient (per mm SWE).
        snow_radiation_coefficient: Multiplier applied to net shortwave radiation (-).

    Returns:
        Tuple of:
            - Net radiation flux into snow (W/m2).
            - Derivative of the net radiation with respect to snow temperature (W/m2/K).
    """
    # Albedo should approach snow_albedo_max as SWE increases,
    # and approach snow_albedo_min (ground influence) as SWE -> 0.
    albedo: np.float32 = snow_albedo_max - (snow_albedo_max - snow_albedo_min) * np.exp(
        -snow_albedo_decay_per_mm
        * min(total_snow_water_equivalent_m, np.float32(10.0))
        * np.float32(1000.0)
    )

    absorbed_shortwave_W_per_m2: np.float32 = (
        (np.float32(1.0) - albedo)
        * snow_radiation_coefficient
        * shortwave_radiation_W_per_m2
    )

    snow_temperature_K: np.float32 = snow_temperature_C + KELVIN_OFFSET
    outgoing_longwave_W_per_m2: np.float32 = (
        SNOW_EMISSIVITY * STEFAN_BOLTZMANN_W_PER_M2_K4 * (snow_temperature_K**4)
    )
    net_radiation_flux_W_per_m2: np.float32 = (
        absorbed_shortwave_W_per_m2
        + longwave_radiation_W_per_m2
        - outgoing_longwave_W_per_m2
    )
    conductance_W_per_m2_K: np.float32 = (
        np.float32(4.0)
        * SNOW_EMISSIVITY
        * STEFAN_BOLTZMANN_W_PER_M2_K4
        * (snow_temperature_K**3)
    )
    return net_radiation_flux_W_per_m2, conductance_W_per_m2_K


@njit(cache=True)
def get_heat_capacity_solid_fraction(
    bulk_density_kg_per_dm3: np.ndarray[Shape, np.dtype[np.float32]],
    layer_thickness_m: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray:
    """Calculate the heat capacity of the solid fraction of the soil layer [J/(m2·K)].

    This calculates the total heat capacity per unit area for the solid part of the soil layer.

    Args:
        bulk_density_kg_per_dm3: Soil bulk density [kg/dm3].
        layer_thickness_m: Thickness of the soil layer [m].

    Returns:
        The areal heat capacity of the solid fraction [J/(m2·K)].
    """
    # Calculate total volume fraction of solids from bulk density
    # Convert bulk density from kg/dm3 (= g/cm3) to kg/m3 (factor 1000)
    phi_s = (bulk_density_kg_per_dm3 * np.float32(1000.0)) / RHO_MINERAL_KG_PER_M3

    # Calculate volumetric heat capacity [J/(m3·K)]
    volumetric_heat_capacity_solid = phi_s * C_MINERAL_VOLUMETRIC_J_PER_M3_K

    # Calculate areal heat capacity [J/(m2·K)]
    # We allocate an explicitly-float32 output buffer so both Numba and static type
    # checkers can unambiguously see the return dtype.
    areal_heat_capacity = np.empty_like(layer_thickness_m, dtype=np.float32)
    areal_heat_capacity[:] = volumetric_heat_capacity_solid * layer_thickness_m
    return areal_heat_capacity


@njit(cache=True, inline="always")
def calculate_thermal_conductivity_solid_fraction_watt_per_meter_kelvin(
    sand_percentage: np.ndarray[Shape, np.dtype[np.float32]],
    silt_percentage: np.ndarray[Shape, np.dtype[np.float32]],
    clay_percentage: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    r"""Calculate the thermal conductivity of the solid fraction of soil [W/(m·K)].

    Based on: https://apps.dtic.mil/sti/tr/pdf/ADA044002.pdf
    See equation 2 on page 102.

    The thermal conductivity of the solid fraction ($\lambda_s$) is calculated as a
    geometric mean of the conductivity of quartz ($\lambda_q$) and other minerals ($\lambda_o$):

    $$ \lambda_s = \lambda_q^q \cdot \lambda_o^{1-q} $$

    where $q$ is the quartz content fraction.

    We assume characteristic quartz content is equal to the sand content fraction.

    Args:
        sand_percentage: Percentage of sand [0-100].
        silt_percentage: Percentage of silt [0-100].
        clay_percentage: Percentage of clay [0-100].

    Returns:
        Thermal conductivity of the solid fraction [W/(m·K)].
    """
    quartz_ratio = np.minimum(
        np.maximum(sand_percentage / np.float32(100.0), np.float32(0.0)),
        np.float32(1.0),
    )

    # Geometric mean calculation
    # lambda_s = (lambda_quartz^q) * (lambda_other^(1-q))
    thermal_conductivity_solid_fraction = (
        THERMAL_CONDUCTIVITY_QUARTZ_WATT_PER_MKELVIN**quartz_ratio
    ) * (
        THERMAL_CONDUCTIVITY_NON_QUARTZ_WATT_PER_MKELVIN
        ** (np.float32(1.0) - quartz_ratio)
    )

    return thermal_conductivity_solid_fraction


@njit(cache=True)
def calculate_thermal_conductivity_dry_soil_johansen_watt_per_meter_kelvin(
    bulk_density_kg_per_dm3: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Calculate the dry thermal conductivity of soil using Johansen (1975) [W/(m·K)].

    Formula: lambda_dry = (0.135 * rho_d + 64.7) / (2700 - 0.947 * rho_d)
    where rho_d is the dry bulk density in kg/m3.

    Args:
        bulk_density_kg_per_dm3: Soil bulk density [kg/dm3].

    Returns:
        Dry thermal conductivity [W/(m·K)].
    """
    bulk_density_kg_per_m3 = bulk_density_kg_per_dm3 * np.float32(1000.0)
    thermal_conductivity_dry = (
        np.float32(0.135) * bulk_density_kg_per_m3 + np.float32(64.7)
    ) / (np.float32(2700.0) - np.float32(0.947) * bulk_density_kg_per_m3)
    return thermal_conductivity_dry


@njit(cache=True)
def calculate_thermal_conductivity_saturated_soil_johansen_watt_per_meter_kelvin(
    thermal_conductivity_solid_fraction: np.ndarray[Shape, np.dtype[np.float32]],
    porosity: np.ndarray[Shape, np.dtype[np.float32]],
    thermal_conductivity_fluid: np.float32,
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Calculate the saturated thermal conductivity of soil using Johansen (1975).

    Formula: lambda = lambda_s^(1-n) * lambda_fluid^n
    where n is porosity and lambda_s is the solid-fraction conductivity.

    Args:
        thermal_conductivity_solid_fraction: Conductivity of the solid fraction [W/(m·K)].
        porosity: Soil porosity [-].
        thermal_conductivity_fluid: Conductivity of the fluid phase (water or ice) [W/(m·K)].

    Returns:
        Saturated thermal conductivity [W/(m·K)].
    """
    thermal_conductivity_saturated = (
        thermal_conductivity_solid_fraction ** (np.float32(1.0) - porosity)
    ) * (thermal_conductivity_fluid**porosity)
    return thermal_conductivity_saturated


@njit(cache=True, inline="always", fastmath=True)
def calculate_soil_thermal_conductivity(
    thermal_conductivity_saturated_unfrozen: np.float32,
    thermal_conductivity_saturated_frozen: np.float32,
    thermal_conductivity_dry_soil_W_per_m_K: np.float32,
    degree_of_saturation: np.float32,
    sand_percentage: np.float32,
    frozen_fraction: np.float32,
) -> np.float32:
    """Calculate the effective thermal conductivity of the soil layer [W/(m·K)].

    Uses the Côté-Konrad (2005) model:

        λ = Ke · (λ_sat - λ_dry) + λ_dry

    which interpolates linearly between λ_dry (at Sr = 0) and λ_sat (at Sr = 1)
    via the Kersten number Ke.

    Args:
        thermal_conductivity_saturated_unfrozen: Thermal conductivity of the soil when fully saturated and unfrozen [W/(m·K)].
        thermal_conductivity_saturated_frozen: Thermal conductivity of the soil when fully saturated and frozen [W/(m·K)].
        thermal_conductivity_dry_soil_W_per_m_K: Thermal conductivity of dry soil [W/(m·K)].
        degree_of_saturation: Degree of saturation (Sr) [0-1].
        sand_percentage: Percentage of sand in the soil [0-100].
        frozen_fraction: Fraction of the soil water that is frozen [0-1].

    Returns:
        Effective thermal conductivity of the soil layer [W/(m·K)].
    """
    # Calculate Côté-Konrad kappa (https://doi.org/10.1139/t04-106)
    sand_fraction: np.float32 = sand_percentage * np.float32(0.01)
    # Unfrozen: 1.9 (silt/clay) to 4.6 (sand)
    # 1.9 * (1 - sand_fraction) + 4.6 * sand_fraction = 1.9 + 2.7 * sand_fraction
    kappa_unfrozen: np.float32 = np.float32(1.9) + np.float32(2.7) * sand_fraction
    # Frozen: 0.85 (silt/clay) to 1.7 (sand)
    # 0.85 * (1 - sand_fraction) + 1.7 * sand_fraction = 0.85 + 0.85 * sand_fraction
    kappa_frozen: np.float32 = np.float32(0.85) + np.float32(0.85) * sand_fraction

    # Linearly combine kappa based on the frozen fraction
    kappa_effective: np.float32 = (
        frozen_fraction * kappa_frozen
        + (np.float32(1.0) - frozen_fraction) * kappa_unfrozen
    )

    # Ke = (kappa * Sr) / (1 + (kappa - 1) * Sr)
    # Equation 13 from Côté and Konrad (2005): https://doi.org/10.1139/t04-106
    Ke: np.float32 = (kappa_effective * degree_of_saturation) / (
        np.float32(1.0) + (kappa_effective - np.float32(1.0)) * degree_of_saturation
    )

    # Combine saturated conductivities linearly
    thermal_conductivity_saturated_soil: np.float32 = (
        frozen_fraction * thermal_conductivity_saturated_frozen
        + (np.float32(1.0) - frozen_fraction) * thermal_conductivity_saturated_unfrozen
    )

    return (
        Ke
        * (
            thermal_conductivity_saturated_soil
            - thermal_conductivity_dry_soil_W_per_m_K
        )
        + thermal_conductivity_dry_soil_W_per_m_K
    )


@njit(cache=True, inline="always", fastmath=True)
def calculate_net_radiation_flux(
    shortwave_radiation_W_per_m2: np.float32,
    longwave_radiation_W_per_m2: np.float32,
    soil_temperature_C: np.float32,
    leaf_area_index: np.float32,
    air_temperature_K: np.float32,
    soil_emissivity: np.float32,
    soil_albedo: np.float32,
) -> tuple[np.float32, np.float32]:
    """Calculate the net radiation energy flux and its derivative.

    Calculates absorbed incoming radiation and outgoing longwave radiation,
    and the derivative of the outgoing longwave radiation with respect to
    soil temperature (conductance equivalent).

    Args:
        shortwave_radiation_W_per_m2: Incoming shortwave [W/m2].
        longwave_radiation_W_per_m2: Incoming longwave [W/m2].
        soil_temperature_C: Current soil temperature [C].
        leaf_area_index: Leaf Area Index [-].
        air_temperature_K: Air temperature [K], used as proxy for canopy temperature.
        soil_emissivity: Soil emissivity [-].
        soil_albedo: Soil albedo [-].

    Returns:
        Tuple of:
            - Net radiation flux [W/m2]. Positive = warming (incoming > outgoing).
            - Derivative of outgoing radiation flux [W/m2/K] (Conductance equivalent).
    """
    temperature_K: np.float32 = soil_temperature_C + KELVIN_OFFSET

    # Beer's law attenuation factor, i.e., the fraction of radiation reaching the soil surface
    # after canopy attenuation
    attenuation_factor: np.float32 = get_canopy_radiation_attenuation(leaf_area_index)

    # Shortwave radiation absorbed by the soil
    absorbed_shortwave_W: np.float32 = (
        (np.float32(1.0) - soil_albedo)
        * shortwave_radiation_W_per_m2
        * attenuation_factor
    )

    # Longwave radiation reaching the soil
    # Atmospheric longwave transmitted through canopy
    transmitted_longwave_W: np.float32 = (
        longwave_radiation_W_per_m2 * attenuation_factor
    )

    # The canopy emits longwave radiation based on its temperature and emissivity,
    # and some of that reaches the soil surface. We assume canopy temperature is
    # equal to air temperature.
    canopy_longwave_W: np.float32 = (
        STEFAN_BOLTZMANN_W_PER_M2_K4
        * (air_temperature_K**4)
        * (np.float32(1.0) - attenuation_factor)
    )

    incoming_longwave_at_soil_surface_W: np.float32 = (
        transmitted_longwave_W + canopy_longwave_W
    )

    absorbed_longwave_W: np.float32 = (
        soil_emissivity * incoming_longwave_at_soil_surface_W
    )

    # Total incoming radiation is the sum of absorbed shortwave and absorbed longwave
    incoming_W: np.float32 = absorbed_shortwave_W + absorbed_longwave_W

    # Outing longwave radiation from the soil surface (Stefan-Boltzmann law)
    outgoing_W: np.float32 = (
        soil_emissivity * STEFAN_BOLTZMANN_W_PER_M2_K4 * (temperature_K**4)
    )

    net_flux_W: np.float32 = incoming_W - outgoing_W

    # Calculate Derivative of Outgoing Radiation with respect to T:
    # d(sigma * eps * T^4)/dT = 4 * sigma * eps * T^3
    # Note that the derivate below only includes the temperature-dependent outgoing flux
    # However, because these are assumed to be constant with respect to soil temperature,
    # we treat them as constants and thus their derivative is zero.
    conductance_W_per_m2_K: np.float32 = (
        np.float32(4.0)
        * soil_emissivity
        * STEFAN_BOLTZMANN_W_PER_M2_K4
        * (temperature_K**3)
    )

    return net_flux_W, conductance_W_per_m2_K


@njit(cache=True, inline="always")
def calculate_aerodynamic_conductance_W_per_m2_K(
    air_temperature_K: np.float32,
    wind_speed_10m_m_per_s: np.float32,
    surface_pressure_pa: np.float32,
) -> np.float32:
    """Calculate the aerodynamic conductance for sensible and latent heat.

    Args:
        air_temperature_K: Air temperature at reference height [K].
        wind_speed_10m_m_per_s: Wind speed at 10m height [m/s].
        surface_pressure_pa: Surface air pressure [Pa].

    Returns:
        Aerodynamic conductance [W/m2/K].
    """
    # Physics Constants
    SPECIFIC_HEAT_AIR_J_KG_K: np.float32 = np.float32(1005.0)
    GAS_CONSTANT_AIR_J_KG_K: np.float32 = np.float32(287.058)
    VON_KARMAN_CONSTANT: np.float32 = np.float32(0.41)

    # Assumptions for Aerodynamic Resistance over bare soil/snow
    WIND_MEASUREMENT_HEIGHT_M: np.float32 = np.float32(10.0)
    TEMP_MEASUREMENT_HEIGHT_M: np.float32 = np.float32(2.0)
    ROUGHNESS_LENGTH_M: np.float32 = np.float32(0.001)

    # Calculate Air Density [kg/m3]
    air_density_kg_per_m3: np.float32 = surface_pressure_pa / (
        GAS_CONSTANT_AIR_J_KG_K * air_temperature_K
    )

    # Use minimum wind speed to avoid extreme resistances and numerical instability
    wind_speed_10m_m_per_s = np.maximum(wind_speed_10m_m_per_s, np.float32(0.1))

    log_wind_height_over_roughness = np.log(
        WIND_MEASUREMENT_HEIGHT_M / ROUGHNESS_LENGTH_M
    )
    log_temp_height_over_roughness = np.log(
        TEMP_MEASUREMENT_HEIGHT_M / ROUGHNESS_LENGTH_M
    )

    aerodynamic_resistance_s_per_m = (
        log_wind_height_over_roughness * log_temp_height_over_roughness
    ) / (VON_KARMAN_CONSTANT**2 * wind_speed_10m_m_per_s)

    # Calculate Conductance [W/m2/K]
    return (
        air_density_kg_per_m3 * SPECIFIC_HEAT_AIR_J_KG_K
    ) / aerodynamic_resistance_s_per_m


@njit(cache=True, inline="always", fastmath=True)
def calculate_sensible_heat_flux(
    soil_temperature_C: np.float32,
    air_temperature_K: np.float32,
    wind_speed_10m_m_per_s: np.float32,
    surface_pressure_pa: np.float32,
) -> tuple[np.float32, np.float32]:
    """Calculate the sensible heat flux and aerodynamic conductance.

    Args:
        soil_temperature_C: Soil temperature in Celsius [C].
        air_temperature_K: Air temperature at 2m height [K].
        wind_speed_10m_m_per_s: Wind speed at 10m height [m/s].
        surface_pressure_pa: Surface air pressure [Pa].

    Returns:
        Tuple of:
            - Sensible heat flux [W/m2]. Positive = warming (Heat flow from Air to Soil).
            - Aerodynamic conductance [W/m2/K].
    """
    conductance_W_per_m2_K: np.float32 = calculate_aerodynamic_conductance_W_per_m2_K(
        air_temperature_K,
        wind_speed_10m_m_per_s,
        surface_pressure_pa,
    )

    # Calculate Explicit Sensible Heat Flux [W/m2]
    # H = Conductance * (Ta - Ts)
    air_temperature_C: np.float32 = air_temperature_K - KELVIN_OFFSET
    temperature_difference_C: np.float32 = air_temperature_C - soil_temperature_C

    sensible_heat_flux_W_per_m2: np.float32 = (
        conductance_W_per_m2_K * temperature_difference_C
    )

    return sensible_heat_flux_W_per_m2, conductance_W_per_m2_K


# used indirectly for lsm only
@njit(cache=True, inline="always")
def get_temperature_and_frozen_fraction_from_enthalpy(
    enthalpy_J_per_m2: np.ndarray[Shape, np.dtype[np.float32]],
    solid_heat_capacity_J_per_m2_K: np.ndarray[Shape, np.dtype[np.float32]],
    water_content_m: np.ndarray[Shape, np.dtype[np.float32]],
    topwater_m: np.float32 | np.ndarray[Shape, np.dtype[np.float32]] = np.float32(0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert enthalpy to temperature and frozen fraction.

    This applies the sharp-freezing enthalpy formulation (0°C plateau) and
    returns temperature and frozen fraction.

    Args:
        enthalpy_J_per_m2: Layer enthalpy relative to 0°C liquid water (J/m2).
        solid_heat_capacity_J_per_m2_K: Areal heat capacity of the solid soil fraction (J/m2/K).
        water_content_m: Total soil water storage (liquid + ice) (m).
        topwater_m: Standing water depth included in the layer thermal mass (m). May be a scalar
            (applied to all elements) or an array broadcastable to `water_content_m`.

    Returns:
        Tuple of:
            - Temperature (C).
            - Frozen fraction of water mass [0-1].
    """
    water_depth_m = water_content_m + topwater_m

    temperature_C: np.ndarray = np.empty_like(enthalpy_J_per_m2, dtype=np.float32)
    frozen_fraction: np.ndarray = np.empty_like(enthalpy_J_per_m2, dtype=np.float32)

    enthalpy_flat = enthalpy_J_per_m2.ravel()
    solid_heat_capacity_flat = solid_heat_capacity_J_per_m2_K.ravel()
    water_depth_flat = water_depth_m.ravel()
    temperature_flat = temperature_C.ravel()
    frozen_fraction_flat = frozen_fraction.ravel()

    for flat_idx in range(enthalpy_flat.size):
        water_depth_cell_m = water_depth_flat[flat_idx]

        latent_heat_areal_J_per_m2 = (
            water_depth_cell_m * RHO_WATER_KG_PER_M3 * LATENT_HEAT_FUSION_J_PER_KG
        )
        heat_capacity_liquid_J_per_m2_K = solid_heat_capacity_flat[flat_idx] + (
            water_depth_cell_m * VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K
        )
        heat_capacity_frozen_J_per_m2_K = solid_heat_capacity_flat[flat_idx] + (
            water_depth_cell_m * VOLUMETRIC_HEAT_CAPACITY_ICE_J_PER_M3_K
        )

        temperature_cell_C, frozen_fraction_cell, _, _ = get_phase_state(
            enthalpy_J_per_m2=enthalpy_flat[flat_idx],
            latent_heat_areal_J_per_m2=latent_heat_areal_J_per_m2,
            heat_capacity_liquid_J_per_m2_K=heat_capacity_liquid_J_per_m2_K,
            heat_capacity_frozen_J_per_m2_K=heat_capacity_frozen_J_per_m2_K,
        )
        temperature_flat[flat_idx] = temperature_cell_C
        frozen_fraction_flat[flat_idx] = frozen_fraction_cell

    return temperature_C, frozen_fraction


# used after lsm only
@njit(cache=True, inline="always")
def get_temperature_from_enthalpy(
    enthalpy_J_per_m2: np.ndarray[Shape, np.dtype[np.float32]],
    solid_heat_capacity_J_per_m2_K: np.ndarray[Shape, np.dtype[np.float32]],
    water_content_m: np.ndarray[Shape, np.dtype[np.float32]],
    topwater_m: np.float32 | np.ndarray[Shape, np.dtype[np.float32]] = np.float32(0.0),
) -> np.float32 | np.ndarray[Shape, np.dtype[np.float32]]:
    """Calculate temperature from enthalpy.

    Args:
        enthalpy_J_per_m2: Layer enthalpy relative to 0°C liquid water (J/m2).
        solid_heat_capacity_J_per_m2_K: Areal heat capacity of the solid soil fraction (J/m2/K).
        water_content_m: Total soil water storage (liquid + ice) (m).
        topwater_m: Standing water depth included in the layer thermal mass (m).

    Returns:
        Temperature (C).
    """
    temperature_C, _ = get_temperature_and_frozen_fraction_from_enthalpy(
        enthalpy_J_per_m2=enthalpy_J_per_m2,
        solid_heat_capacity_J_per_m2_K=solid_heat_capacity_J_per_m2_K,
        water_content_m=water_content_m,
        topwater_m=topwater_m,
    )
    return temperature_C


# used after lsm only
@njit(cache=True, inline="always")
def get_frozen_fraction_from_enthalpy(
    enthalpy_J_per_m2: np.ndarray[Shape, np.dtype[np.float32]],
    solid_heat_capacity_J_per_m2_K: np.ndarray[Shape, np.dtype[np.float32]],
    water_content_m: np.ndarray[Shape, np.dtype[np.float32]],
    topwater_m: np.float32 | np.ndarray[Shape, np.dtype[np.float32]] = np.float32(0.0),
) -> np.float32 | np.ndarray[Shape, np.dtype[np.float32]]:
    """Return frozen fraction [0-1] from enthalpy.

    Args:
        enthalpy_J_per_m2: Layer enthalpy relative to 0°C liquid water (J/m2).
        solid_heat_capacity_J_per_m2_K: Areal heat capacity of the solid soil fraction (J/m2/K).
        water_content_m: Total soil water storage (liquid + ice) (m).
        topwater_m: Standing water depth included in the layer thermal mass (m).

    Returns:
        Frozen fraction of water mass [0-1].
    """
    _, frozen_fraction = get_temperature_and_frozen_fraction_from_enthalpy(
        enthalpy_J_per_m2=enthalpy_J_per_m2,
        solid_heat_capacity_J_per_m2_K=solid_heat_capacity_J_per_m2_K,
        water_content_m=water_content_m,
        topwater_m=topwater_m,
    )
    return frozen_fraction


@njit(cache=True, inline="always")
def get_temperature_and_frozen_fraction_from_enthalpy_scalar(
    enthalpy_J_per_m2: np.float32,
    solid_heat_capacity_J_per_m2_K: np.float32,
    water_content_m: np.float32,
    topwater_m: np.float32 = np.float32(0.0),
) -> tuple[np.float32, np.float32]:
    """Diagnose temperature and frozen fraction from enthalpy for scalar inputs.

    Notes:
        This is a scalar-only helper intended for hot loops where both temperature
        and frozen fraction are needed (e.g., to limit liquid-water fluxes while
        advecting sensible heat).

        The enthalpy reference is 0°C liquid water, consistent with the enthalpy
        column solver.

    Args:
        enthalpy_J_per_m2: Layer enthalpy relative to 0°C liquid water (J/m2).
        solid_heat_capacity_J_per_m2_K: Areal heat capacity of the solid soil fraction (J/m2/K).
        water_content_m: Total soil water storage (liquid + ice) (m).
        topwater_m: Standing water depth included in the layer thermal mass (m).

    Returns:
        Tuple of:
            - temperature_C: Layer temperature (C).
            - frozen_fraction: Frozen fraction of water mass (0-1).
    """
    water_depth_m = water_content_m + topwater_m
    latent_heat_areal_J_per_m2 = (
        water_depth_m * RHO_WATER_KG_PER_M3 * LATENT_HEAT_FUSION_J_PER_KG
    )

    heat_capacity_liquid_J_per_m2_K = (
        solid_heat_capacity_J_per_m2_K
        + water_depth_m * VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K
    )
    heat_capacity_frozen_J_per_m2_K = (
        solid_heat_capacity_J_per_m2_K
        + water_depth_m * VOLUMETRIC_HEAT_CAPACITY_ICE_J_PER_M3_K
    )

    temperature_C, frozen_fraction, _, _ = get_phase_state(
        enthalpy_J_per_m2=enthalpy_J_per_m2,
        latent_heat_areal_J_per_m2=latent_heat_areal_J_per_m2,
        heat_capacity_liquid_J_per_m2_K=heat_capacity_liquid_J_per_m2_K,
        heat_capacity_frozen_J_per_m2_K=heat_capacity_frozen_J_per_m2_K,
    )
    return temperature_C, frozen_fraction


@njit(cache=True, inline="always", fastmath=True)
def get_phase_state(
    enthalpy_J_per_m2: np.float32,
    latent_heat_areal_J_per_m2: np.float32,
    heat_capacity_liquid_J_per_m2_K: np.float32,
    heat_capacity_frozen_J_per_m2_K: np.float32,
) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    """Diagnose temperature, frozen fraction, and linearization from enthalpy.

    This function implements a sharp-freezing (0°C plateau) enthalpy-temperature
    relation. It is used to diagnostically determine the thermal state of a soil layer
    and to provide gradients for implicit nonlinear solvers where $T \approx \alpha H + \beta$.

    Notes:
        The enthalpy reference point is 0°C liquid water.
        - H > 0: Fully liquid. Temperature T = H / C_liquid.
        - -L < H <= 0: Phase change (mushy zone). Temperature T = 0.
          Frozen fraction f_ice = -H / L.
        - H <= -L: Fully frozen. Temperature T = (H + L) / C_ice.

    Args:
        enthalpy_J_per_m2: Layer enthalpy relative to 0°C liquid water (J/m2).
        latent_heat_areal_J_per_m2: Areal latent heat of fusion for the layer water (J/m2).
        heat_capacity_liquid_J_per_m2_K: Areal heat capacity in fully liquid regime (J/m2/K).
        heat_capacity_frozen_J_per_m2_K: Areal heat capacity in fully frozen regime (J/m2/K).

    Returns:
        A tuple containing:
            - temperature_C: Layer temperature (C).
            - frozen_fraction: Frozen fraction of water mass (0-1).
            - dT_dH: The derivative $\partial T/\partial H$ (K per (J/m2)).
            - beta: Intercept for the linear approximation $T \approx (dT/dH) \cdot H + \beta$.
    """
    # Fully liquid
    if enthalpy_J_per_m2 >= np.float32(0.0):
        if heat_capacity_liquid_J_per_m2_K <= np.float32(0.0):
            return np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0)
        alpha = np.float32(1.0) / heat_capacity_liquid_J_per_m2_K
        temperature_C = enthalpy_J_per_m2 * alpha
        return temperature_C, np.float32(0.0), alpha, np.float32(0.0)

    # Mushy zone (0°C plateau)
    if enthalpy_J_per_m2 >= -latent_heat_areal_J_per_m2:
        frozen_fraction = -enthalpy_J_per_m2 / (
            latent_heat_areal_J_per_m2 + np.float32(1e-12)
        )
        frozen_fraction = np.minimum(
            np.maximum(frozen_fraction, np.float32(0.0)), np.float32(1.0)
        )
        return np.float32(0.0), frozen_fraction, np.float32(0.0), np.float32(0.0)

    # Fully frozen (below 0°C)
    if heat_capacity_frozen_J_per_m2_K <= np.float32(0.0):
        return np.float32(0.0), np.float32(1.0), np.float32(0.0), np.float32(0.0)

    alpha = np.float32(1.0) / heat_capacity_frozen_J_per_m2_K
    beta = latent_heat_areal_J_per_m2 * alpha
    temperature_C = (enthalpy_J_per_m2 + latent_heat_areal_J_per_m2) * alpha
    return temperature_C, np.float32(1.0), alpha, beta


@njit(cache=True, inline="always", fastmath=True)
def apply_evaporative_cooling(
    soil_enthalpy_top_layer_J_per_m2: np.float32,
    evaporation_m: np.float32,
    frozen_fraction_top_layer: np.float32,
) -> np.float32:
    """Apply evaporative cooling to the top soil layer enthalpy.

    Sublimation and vaporization can only occur if there is water available.
    However, we assume evaporation has already been checked against available water.

    Args:
        soil_enthalpy_top_layer_J_per_m2: Top-layer enthalpy (J/m2).
        evaporation_m: Amount of evaporation (m).
        frozen_fraction_top_layer: Frozen fraction of the top-layer water mass [0-1].

    Returns:
        Updated top-layer enthalpy (J/m2).
    """
    fraction_unfrozen = np.float32(1.0) - np.minimum(
        np.maximum(frozen_fraction_top_layer, np.float32(0.0)), np.float32(1.0)
    )

    # Weighted latent heat
    # If liquid: Vaporization energy
    # If ice: Sublimation energy (Fusion + Vaporization)
    latent_heat = (
        fraction_unfrozen * LATENT_HEAT_VAPORIZATION_J_PER_KG
        + (np.float32(1.0) - fraction_unfrozen) * LATENT_HEAT_SUBLIMATION_J_PER_KG
    )

    energy_loss_J_per_m2 = evaporation_m * RHO_WATER_KG_PER_M3 * latent_heat

    return soil_enthalpy_top_layer_J_per_m2 - energy_loss_J_per_m2


@njit(cache=True, inline="always", fastmath=True)
def apply_rain_heat_advection(
    soil_enthalpy_top_layer_J_per_m2: np.float32,
    liquid_water_input_m: np.float32,
    rain_temperature_C: np.float32,
) -> np.float32:
    """Apply advective heat transport from liquid water to top-layer enthalpy.

    Notes:
        The liquid water enters the top-layer thermal control volume at
        `rain_temperature_C`. This is used for rainfall reaching the soil surface,
        even when part of that water remains ponded instead of infiltrating.
        This updates enthalpy conservatively:
        $H_{new} = H_{old} + m c_w T_{rain}$.

    Args:
        soil_enthalpy_top_layer_J_per_m2: Top-layer enthalpy (J/m2).
        liquid_water_input_m: Liquid water added to the top-layer control volume (m).
        rain_temperature_C: Rain temperature (C).

    Returns:
        Updated top-layer enthalpy (J/m2).
    """
    enthalpy_added_J_per_m2: np.float32 = (
        liquid_water_input_m
        * VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K
        * rain_temperature_C
    )
    return soil_enthalpy_top_layer_J_per_m2 + enthalpy_added_J_per_m2


_N_SOIL_LAYERS_PLUS_ONE: int = N_SOIL_LAYERS + 1


@njit(cache=True)
def solve_soil_enthalpy_column(
    soil_enthalpies_J_per_m2: ArrayFloat32,
    layer_thicknesses_m: ArrayFloat32,
    thermal_conductivity_dry_soil_W_per_m_K: ArrayFloat32,
    solid_heat_capacities_J_per_m2_K: ArrayFloat32,
    thermal_conductivity_saturated_unfrozen_W_per_m_K: ArrayFloat32,
    thermal_conductivity_saturated_frozen_W_per_m_K: ArrayFloat32,
    water_content_saturated_m: ArrayFloat32,
    sand_percentage: ArrayFloat32,
    water_content_m: ArrayFloat32,
    shortwave_radiation_W_per_m2: np.float32,
    longwave_radiation_W_per_m2: np.float32,
    air_temperature_K: np.float32,
    wind_speed_10m_m_per_s: np.float32,
    surface_pressure_pa: np.float32,
    timestep_seconds: np.float32,
    deep_soil_temperature_C: np.float32,
    soil_emissivity: np.float32,
    soil_albedo: np.float32,
    leaf_area_index: np.float32,
    snow_water_equivalent_m: ArrayFloat32,
    snow_enthalpy_J_per_m2: ArrayFloat32,
    snow_density_kg_per_m3: ArrayFloat32,
    topwater_m: np.float32,
) -> tuple[np.float32, np.float32]:
    """Solve the coupled snow-soil enthalpy profile with an implicit scheme.

    Notes:
        The solver uses a two-step approach for snow thermal coupling:
        1.  Active snow layers (SWE > MIN_ACTIVE_SNOW_SWE_M) are fully coupled
            into the tridiagonal matrix as part of the thermal column.
        2.  Inactive snow layers (trace SWE below the threshold) are excluded
            from the matrix for numerical stability. However, they are thermally
            synchronized with the top soil layer (assigned the same temperature).
            This synchronization prevents "zombie snow" where trace mass is
            thermally isolated and stuck at 0°C, ensuring it eventually melts or
            sublimates when the soil warms up.

    Args:
        soil_enthalpies_J_per_m2: Current layer enthalpies (J/m2).
        layer_thicknesses_m: Layer thicknesses (m).
        thermal_conductivity_dry_soil_W_per_m_K: Thermal conductivity of dry soil per layer (W/m/K).
        solid_heat_capacities_J_per_m2_K: Areal heat capacity of the solid fraction (J/m2/K). Areal heat capacity of the solid fraction (J/m2/K).
        thermal_conductivity_saturated_unfrozen_W_per_m_K: Saturated conductivity in
            unfrozen state (W/m/K).
        thermal_conductivity_saturated_frozen_W_per_m_K: Saturated conductivity in
            frozen state (W/m/K).
        water_content_saturated_m: Saturated water storage (m).
        sand_percentage: Sand fraction (0-100).
        water_content_m: Layer water storage (m).
        shortwave_radiation_W_per_m2: Incoming shortwave radiation (W/m2).
        longwave_radiation_W_per_m2: Incoming longwave radiation (W/m2).
        air_temperature_K: Air temperature (K).
        wind_speed_10m_m_per_s: Wind speed at 10m (m/s).
        surface_pressure_pa: Surface pressure (Pa).
        timestep_seconds: Timestep length (s).
        deep_soil_temperature_C: Bottom boundary temperature (C).
        soil_emissivity: Soil emissivity (-).
        soil_albedo: Soil albedo (-).
        leaf_area_index: Leaf area index (-).
        snow_water_equivalent_m: Snow water equivalent per snow layer (m).
        snow_enthalpy_J_per_m2: Snow enthalpy per snow layer (J/m2).
        snow_density_kg_per_m3: Snow density per snow layer (kg/m3).
        topwater_m: Standing water depth (m).

    Returns:
        Tuple of:
            - Soil heat flux (W/m2).
            - Top-layer frozen fraction (0-1).
    """
    lower_diagonal_a = stack_empty(_N_COUPLED_SURFACE_LAYERS, np.float32)
    main_diagonal_b = stack_empty(_N_COUPLED_SURFACE_LAYERS, np.float32)
    upper_diagonal_c = stack_empty(_N_COUPLED_SURFACE_LAYERS, np.float32)
    rhs_vector_d = stack_empty(_N_COUPLED_SURFACE_LAYERS, np.float32)
    thermal_conductances_between_layer_centers_W_per_m2_K = stack_empty(
        _N_COUPLED_SURFACE_LAYERS, dtype=np.float32
    )
    dT_dH_linearized = stack_empty(_N_COUPLED_SURFACE_LAYERS, dtype=np.float32)
    beta_linearized = stack_empty(_N_COUPLED_SURFACE_LAYERS, dtype=np.float32)
    conductivities_W_per_m_K = stack_empty(_N_COUPLED_SURFACE_LAYERS, dtype=np.float32)
    half_layer_thicknesses_m = stack_empty(_N_COUPLED_SURFACE_LAYERS, dtype=np.float32)
    enthalpies_at_start_of_timestep = stack_empty(
        _N_COUPLED_SURFACE_LAYERS, dtype=np.float32
    )
    enthalpies_updated = stack_empty(_N_COUPLED_SURFACE_LAYERS, dtype=np.float32)

    n_active_snow_layers = 0
    for snow_layer_idx in range(N_SNOW_LAYERS):
        if snow_water_equivalent_m[
            snow_layer_idx
        ] > MIN_ACTIVE_SNOW_SWE_M and snow_density_kg_per_m3[
            snow_layer_idx
        ] > np.float32(0.0):
            n_active_snow_layers += 1

    n_total_layers: int = n_active_snow_layers + N_SOIL_LAYERS

    surface_temperature_guess_C: np.float32 = np.float32(0.0)
    top_layer_latent_heat_areal_J_per_m2: np.float32 = np.float32(0.0)
    top_layer_heat_capacity_liquid_J_per_m2_K: np.float32 = np.float32(0.0)
    top_layer_heat_capacity_frozen_J_per_m2_K: np.float32 = np.float32(0.0)

    for snow_layer_idx in range(n_active_snow_layers):
        swe_m: np.float64 = snow_water_equivalent_m[snow_layer_idx]
        enthalpy_J_per_m2: np.float32 = snow_enthalpy_J_per_m2[snow_layer_idx]
        density_kg_per_m3: np.float32 = snow_density_kg_per_m3[snow_layer_idx]
        heat_capacity_J_per_m2_K: np.float32 = (
            np.float32(swe_m)
            * RHO_WATER_KG_PER_M3
            * SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K
        )
        temperature_C: np.float32 = enthalpy_J_per_m2 / heat_capacity_J_per_m2_K

        enthalpies_at_start_of_timestep[snow_layer_idx] = enthalpy_J_per_m2
        dT_dH_linearized[snow_layer_idx] = np.float32(1.0) / heat_capacity_J_per_m2_K
        beta_linearized[snow_layer_idx] = np.float32(0.0)
        conductivities_W_per_m_K[snow_layer_idx] = (
            calculate_snow_thermal_conductivity_from_density(density_kg_per_m3)
        )
        snow_depth_m: np.float32 = np.float32(swe_m) / max(
            density_kg_per_m3 / np.float32(1000.0), np.float32(1e-6)
        )
        half_layer_thicknesses_m[snow_layer_idx] = np.float32(0.5) * max(
            snow_depth_m, np.float32(0.01)
        )

        if snow_layer_idx == 0:
            surface_temperature_guess_C: np.float32 = temperature_C

    for layer_idx in range(N_SOIL_LAYERS):
        combined_layer_idx = n_active_snow_layers + layer_idx
        topwater_layer_m = topwater_m if layer_idx == 0 else np.float32(0.0)
        water_depth_m = water_content_m[layer_idx] + topwater_layer_m
        latent_heat_areal_J_per_m2 = (
            water_depth_m * RHO_WATER_KG_PER_M3 * LATENT_HEAT_FUSION_J_PER_KG
        )
        heat_capacity_liquid_J_per_m2_K = (
            solid_heat_capacities_J_per_m2_K[layer_idx]
            + water_depth_m * VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K
        )
        heat_capacity_frozen_J_per_m2_K = (
            solid_heat_capacities_J_per_m2_K[layer_idx]
            + water_depth_m * VOLUMETRIC_HEAT_CAPACITY_ICE_J_PER_M3_K
        )

        enthalpy_J_per_m2 = soil_enthalpies_J_per_m2[layer_idx]
        temperature_C, frozen_fraction, dT_dH, beta = get_phase_state(
            enthalpy_J_per_m2=enthalpy_J_per_m2,
            latent_heat_areal_J_per_m2=latent_heat_areal_J_per_m2,
            heat_capacity_liquid_J_per_m2_K=heat_capacity_liquid_J_per_m2_K,
            heat_capacity_frozen_J_per_m2_K=heat_capacity_frozen_J_per_m2_K,
        )
        enthalpies_at_start_of_timestep[combined_layer_idx] = enthalpy_J_per_m2
        dT_dH_linearized[combined_layer_idx] = dT_dH
        beta_linearized[combined_layer_idx] = beta

        if combined_layer_idx == n_active_snow_layers:
            # The iterative flux balance must be linearized around the temperature
            # of the top layer (either snow surface or soil surface).
            if n_active_snow_layers == 0:
                surface_temperature_guess_C = temperature_C

            # These properties are specifically needed for the top soil layer's
            # phase state diagnostics returned by this function.
            top_layer_latent_heat_areal_J_per_m2 = latent_heat_areal_J_per_m2
            top_layer_heat_capacity_liquid_J_per_m2_K = heat_capacity_liquid_J_per_m2_K
            top_layer_heat_capacity_frozen_J_per_m2_K = heat_capacity_frozen_J_per_m2_K

        lambda_layer_W_per_m_K = calculate_soil_thermal_conductivity(
            thermal_conductivity_saturated_unfrozen=thermal_conductivity_saturated_unfrozen_W_per_m_K[
                layer_idx
            ],
            thermal_conductivity_saturated_frozen=thermal_conductivity_saturated_frozen_W_per_m_K[
                layer_idx
            ],
            thermal_conductivity_dry_soil_W_per_m_K=thermal_conductivity_dry_soil_W_per_m_K[
                layer_idx
            ],
            degree_of_saturation=water_content_m[layer_idx]
            / water_content_saturated_m[layer_idx],
            sand_percentage=sand_percentage[layer_idx],
            frozen_fraction=frozen_fraction,
        )

        conductivities_W_per_m_K[combined_layer_idx] = lambda_layer_W_per_m_K
        half_layer_thicknesses_m[combined_layer_idx] = (
            np.float32(0.5) * layer_thicknesses_m[layer_idx]
        )

    for combined_layer_idx in range(1, n_total_layers):
        lambda_upper = conductivities_W_per_m_K[combined_layer_idx - 1]
        lambda_lower = conductivities_W_per_m_K[combined_layer_idx]

        resistance_upper_half_layer = half_layer_thicknesses_m[
            combined_layer_idx - 1
        ] / max(lambda_upper, np.float32(1e-6))
        resistance_lower_half_layer = half_layer_thicknesses_m[
            combined_layer_idx
        ] / max(lambda_lower, np.float32(1e-6))
        thermal_conductances_between_layer_centers_W_per_m2_K[
            combined_layer_idx - 1
        ] = np.float32(1.0) / (
            resistance_upper_half_layer + resistance_lower_half_layer
        )

    lambda_bottom_layer_W_per_m_K: np.float32 = conductivities_W_per_m_K[
        n_total_layers - 1
    ]

    inv_dt: np.float32 = np.float32(1.0) / timestep_seconds

    if n_active_snow_layers > 0:
        total_snow_water_equivalent_m = np.float32(0.0)
        for snow_layer_idx in range(n_active_snow_layers):
            total_snow_water_equivalent_m += np.float32(
                snow_water_equivalent_m[snow_layer_idx]
            )

        net_radiation_flux_W_per_m2, derivative_net_radiation_W_per_m2_K = (
            calculate_snow_net_radiation_flux(
                shortwave_radiation_W_per_m2=shortwave_radiation_W_per_m2,
                longwave_radiation_W_per_m2=longwave_radiation_W_per_m2,
                snow_temperature_C=surface_temperature_guess_C,
                total_snow_water_equivalent_m=total_snow_water_equivalent_m,
            )
        )
        sensible_heat_flux_W_per_m2, derivative_sensible_heat_W_per_m2_K = (
            calculate_sensible_heat_flux(
                soil_temperature_C=surface_temperature_guess_C,
                air_temperature_K=air_temperature_K,
                wind_speed_10m_m_per_s=wind_speed_10m_m_per_s,
                surface_pressure_pa=surface_pressure_pa,
            )
        )
    else:
        net_radiation_flux_W_per_m2, derivative_net_radiation_W_per_m2_K = (
            calculate_net_radiation_flux(
                shortwave_radiation_W_per_m2=shortwave_radiation_W_per_m2,
                longwave_radiation_W_per_m2=longwave_radiation_W_per_m2,
                soil_temperature_C=surface_temperature_guess_C,
                leaf_area_index=leaf_area_index,
                air_temperature_K=air_temperature_K,
                soil_emissivity=soil_emissivity,
                soil_albedo=soil_albedo,
            )
        )
        sensible_heat_flux_W_per_m2, derivative_sensible_heat_W_per_m2_K = (
            calculate_sensible_heat_flux(
                soil_temperature_C=surface_temperature_guess_C,
                air_temperature_K=air_temperature_K,
                wind_speed_10m_m_per_s=wind_speed_10m_m_per_s,
                surface_pressure_pa=surface_pressure_pa,
            )
        )

    flux_star_W_per_m2 = (
        net_radiation_flux_W_per_m2
        + sensible_heat_flux_W_per_m2
        + (derivative_net_radiation_W_per_m2_K + derivative_sensible_heat_W_per_m2_K)
        * surface_temperature_guess_C
    )
    surface_thermal_conductance_W_per_m2_K = (
        derivative_net_radiation_W_per_m2_K + derivative_sensible_heat_W_per_m2_K
    )

    # Build the tridiagonal system in H
    # Top layer
    conductance_to_layer_below = thermal_conductances_between_layer_centers_W_per_m2_K[
        0
    ]
    alpha_0 = dT_dH_linearized[0]
    beta_0 = beta_linearized[0]
    alpha_1 = dT_dH_linearized[1]
    beta_1 = beta_linearized[1]

    lower_diagonal_a[0] = np.float32(0.0)
    main_diagonal_b[0] = (
        inv_dt
        + (surface_thermal_conductance_W_per_m2_K + conductance_to_layer_below)
        * alpha_0
    )
    upper_diagonal_c[0] = -conductance_to_layer_below * alpha_1
    rhs_vector_d[0] = (
        inv_dt * enthalpies_at_start_of_timestep[0]
        + flux_star_W_per_m2
        - (surface_thermal_conductance_W_per_m2_K + conductance_to_layer_below) * beta_0
        + conductance_to_layer_below * beta_1
    )

    # Intermediate layers
    for layer_idx in range(1, n_total_layers - 1):
        conductance_to_layer_above = (
            thermal_conductances_between_layer_centers_W_per_m2_K[layer_idx - 1]
        )
        conductance_to_layer_below = (
            thermal_conductances_between_layer_centers_W_per_m2_K[layer_idx]
        )

        alpha_im1 = dT_dH_linearized[layer_idx - 1]
        beta_im1 = beta_linearized[layer_idx - 1]
        alpha_i = dT_dH_linearized[layer_idx]
        beta_i = beta_linearized[layer_idx]
        alpha_ip1 = dT_dH_linearized[layer_idx + 1]
        beta_ip1 = beta_linearized[layer_idx + 1]

        lower_diagonal_a[layer_idx] = -conductance_to_layer_above * alpha_im1
        main_diagonal_b[layer_idx] = (
            inv_dt + (conductance_to_layer_above + conductance_to_layer_below) * alpha_i
        )
        upper_diagonal_c[layer_idx] = -conductance_to_layer_below * alpha_ip1
        rhs_vector_d[layer_idx] = (
            inv_dt * enthalpies_at_start_of_timestep[layer_idx]
            - (conductance_to_layer_above + conductance_to_layer_below) * beta_i
            + conductance_to_layer_above * beta_im1
            + conductance_to_layer_below * beta_ip1
        )

    # Bottom layer (Dirichlet)
    last_idx = n_total_layers - 1
    conductance_to_layer_above = thermal_conductances_between_layer_centers_W_per_m2_K[
        last_idx - 1
    ]
    conductance_to_deep_soil_boundary_W_per_m2_K = lambda_bottom_layer_W_per_m_K / (
        np.float32(0.5) * max(layer_thicknesses_m[N_SOIL_LAYERS - 1], np.float32(1e-6))
    )

    alpha_last = dT_dH_linearized[last_idx]
    beta_last = beta_linearized[last_idx]
    alpha_above = dT_dH_linearized[last_idx - 1]
    beta_above = beta_linearized[last_idx - 1]

    lower_diagonal_a[last_idx] = -conductance_to_layer_above * alpha_above
    main_diagonal_b[last_idx] = (
        inv_dt
        + (conductance_to_layer_above + conductance_to_deep_soil_boundary_W_per_m2_K)
        * alpha_last
    )
    upper_diagonal_c[last_idx] = np.float32(0.0)
    rhs_vector_d[last_idx] = (
        inv_dt * enthalpies_at_start_of_timestep[last_idx]
        + conductance_to_deep_soil_boundary_W_per_m2_K * deep_soil_temperature_C
        - (conductance_to_layer_above + conductance_to_deep_soil_boundary_W_per_m2_K)
        * beta_last
        + conductance_to_layer_above * beta_above
    )

    # Thomas forward/back substitution using compact loops.
    c_prime = stack_empty(_N_COUPLED_SURFACE_LAYERS, np.float32)
    d_prime = stack_empty(_N_COUPLED_SURFACE_LAYERS, np.float32)

    if main_diagonal_b[0] == np.float32(0.0):
        c_prime[0] = np.float32(0.0)
        d_prime[0] = np.float32(0.0)
    else:
        c_prime[0] = upper_diagonal_c[0] / main_diagonal_b[0]
        d_prime[0] = rhs_vector_d[0] / main_diagonal_b[0]

    for layer_idx in range(1, n_total_layers):
        denominator = (
            main_diagonal_b[layer_idx]
            - lower_diagonal_a[layer_idx] * c_prime[layer_idx - 1]
        )
        if layer_idx < n_total_layers - 1:
            c_prime[layer_idx] = upper_diagonal_c[layer_idx] / denominator
        d_prime[layer_idx] = (
            rhs_vector_d[layer_idx]
            - lower_diagonal_a[layer_idx] * d_prime[layer_idx - 1]
        ) / denominator

    enthalpies_updated[n_total_layers - 1] = d_prime[n_total_layers - 1]
    for layer_idx in range(n_total_layers - 2, -1, -1):
        enthalpies_updated[layer_idx] = (
            d_prime[layer_idx]
            - c_prime[layer_idx] * (enthalpies_updated[layer_idx + 1])
        )

    for snow_layer_idx in range(n_active_snow_layers):
        snow_enthalpy_J_per_m2[snow_layer_idx] = enthalpies_updated[snow_layer_idx]

    for soil_layer_idx in range(N_SOIL_LAYERS):
        soil_enthalpies_J_per_m2[soil_layer_idx] = enthalpies_updated[
            n_active_snow_layers + soil_layer_idx
        ]

    # Calculate final surface state (top soil layer or active snow if present).
    # This is needed for the surface flux and for synchronizing inactive snow layers.
    surface_temperature_final_C, frozen_fraction_top_layer_final, _, _ = (
        get_phase_state(
            enthalpy_J_per_m2=enthalpies_updated[n_active_snow_layers],
            latent_heat_areal_J_per_m2=top_layer_latent_heat_areal_J_per_m2,
            heat_capacity_liquid_J_per_m2_K=top_layer_heat_capacity_liquid_J_per_m2_K,
            heat_capacity_frozen_J_per_m2_K=top_layer_heat_capacity_frozen_J_per_m2_K,
        )
    )

    # Handle inactive snow layers (trace mass below MIN_ACTIVE_SNOW_SWE_M).
    # We synchronize their enthalpy (and thus temperature) with the top soil layer.
    # This prevents "zombie snow" where trace amounts of snow never melt because
    # they are excluded from the thermal column solver and were previously reset to 0°C.
    # By assigning the soil surface temperature, these trace layers will correctly
    # melt or sublimate in the subsequent mass-balance step.
    for snow_layer_idx in range(n_active_snow_layers, N_SNOW_LAYERS):
        swe_m = snow_water_equivalent_m[snow_layer_idx]
        if swe_m > 0:
            snow_enthalpy_J_per_m2[snow_layer_idx] = (
                np.float32(swe_m)
                * RHO_WATER_KG_PER_M3
                * SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K
                * surface_temperature_final_C
            )
        else:
            snow_enthalpy_J_per_m2[snow_layer_idx] = np.float32(0.0)

    surface_temperature_for_flux_C: np.float32 = (
        dT_dH_linearized[0] * enthalpies_updated[0] + beta_linearized[0]
    )
    soil_heat_flux_W_per_m2 = (
        flux_star_W_per_m2
        - surface_thermal_conductance_W_per_m2_K * surface_temperature_for_flux_C
    )

    return soil_heat_flux_W_per_m2, frozen_fraction_top_layer_final
