"""Soil energy flow functions."""

import numpy as np
from numba import njit

from geb.geb_types import Shape
from geb.workflows.algebra import tdma_solver

from .constants import (
    C_MINERAL_VOLUMETRIC_J_PER_M3_K,
    L_FUSION_J_PER_KG,
    L_SUBLIMATION_J_PER_KG,
    L_VAPORIZATION_J_PER_KG,
    LAMBDA_ICE,
    LAMBDA_OTHER_FINE,
    LAMBDA_QUARTZ,
    LAMBDA_WATER,
    RHO_MINERAL_KG_PER_M3,
    RHO_WATER_KG_PER_M3,
    STEFAN_BOLTZMANN_W_PER_M2_K4,
    VOLUMETRIC_HEAT_CAPACITY_ICE_J_PER_M3_K,
    VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K,
)
from .potential_evapotranspiration import get_canopy_radiation_attenuation
from .snow_glaciers import calculate_snow_thermal_properties


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
    # Estimate from sand if not provided (ensuring it is bounded between 0 and 1)
    # Ensure it's bounded [0, 1]
    quartz_ratio = np.minimum(
        np.maximum(sand_percentage / np.float32(100.0), np.float32(0.0)),
        np.float32(1.0),
    )

    # Geometric mean calculation
    # lambda_s = (lambda_quartz^q) * (lambda_other^(1-q))
    lambda_s = (LAMBDA_QUARTZ**quartz_ratio) * (
        LAMBDA_OTHER_FINE ** (np.float32(1.0) - quartz_ratio)
    )

    return lambda_s


@njit(cache=True, inline="always")
def calculate_soil_thermal_conductivity_from_frozen_fraction(
    thermal_conductivity_solid_W_per_m_K: np.ndarray[Shape, np.dtype[np.float32]],
    bulk_density_kg_per_dm3: np.ndarray[Shape, np.dtype[np.float32]],
    porosity: np.ndarray[Shape, np.dtype[np.float32]],
    degree_of_saturation: np.ndarray[Shape, np.dtype[np.float32]],
    sand_percentage: np.ndarray[Shape, np.dtype[np.float32]],
    frozen_fraction: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Calculate the total soil thermal conductivity [W/(m·K)].

    Based on the Johansen (1975) method as described in Farouki (1981).
    This method interpolates between dry and saturated thermal conductivities
    using the Kersten number, which depends on the degree of saturation and
    whether the soil is frozen.

    Args:
        thermal_conductivity_solid_W_per_m_K: Thermal conductivity of the solid fraction [W/(m·K)].
        bulk_density_kg_per_dm3: Soil bulk density [kg/dm3].
        porosity: Soil porosity [-].
        degree_of_saturation: Degree of saturation [0-1].
        sand_percentage: Percentage of sand [0-100].
        frozen_fraction: Frozen water mass fraction (0-1).

    Returns:
        Total soil thermal conductivity [W/(m·K)].
    """
    Sr = np.maximum(np.float32(1e-5), np.minimum(degree_of_saturation, np.float32(1.0)))

    # Convert bulk density from kg/dm3 to kg/m3 for calculation
    rho_bulk = bulk_density_kg_per_dm3 * np.float32(1000.0)

    # Johansen (1975) semi-empirical formula for dry mineral soils
    lambda_dry = (np.float32(0.135) * rho_bulk + np.float32(64.7)) / (
        RHO_MINERAL_KG_PER_M3 - np.float32(0.947) * rho_bulk
    )

    # Clip into [0, 1]. We avoid `out=` here because Numba does not support ufunc
    # keyword arguments in nopython mode.
    frozen_fraction_clipped = np.minimum(
        np.maximum(frozen_fraction, np.float32(0.0)), np.float32(1.0)
    ).astype(np.float32)

    # We use a geometric mean for the saturated state components
    lambda_sat_unfrozen = (
        thermal_conductivity_solid_W_per_m_K ** (np.float32(1.0) - porosity)
    ) * (LAMBDA_WATER**porosity)
    lambda_sat_frozen = (
        thermal_conductivity_solid_W_per_m_K ** (np.float32(1.0) - porosity)
    ) * (LAMBDA_ICE**porosity)

    # Kersten number (Ke) is an empirical weighting factor [0-1] that interpolates
    # between dry and saturated thermal conductivities based on moisture content.

    # Calculate Ke for unfrozen conditions
    log10_sr = np.log10(Sr)

    # Coarse soils (sand content > 40%) use a different moisture sensitivity
    Ke_unfrozen_coarse = np.float32(0.7) * log10_sr + np.float32(1.0)
    # Fine soils use a steeper moisture sensitivity
    Ke_unfrozen_fine = log10_sr + np.float32(1.0)

    is_coarse = sand_percentage > np.float32(40.0)
    Ke_unfrozen = np.where(is_coarse, Ke_unfrozen_coarse, Ke_unfrozen_fine)
    Ke_unfrozen = np.maximum(Ke_unfrozen, np.float32(0.0))

    # Calculate Ke for frozen conditions
    # For frozen soil, the relationship is typically assumed linear with saturation
    Ke_frozen = Sr

    # Determine the saturated thermal conductivity and Kersten number
    # We use a geometric mean for the saturated state components and linear
    # interpolation for the Kersten number based on the frozen fraction.
    lambda_sat = (lambda_sat_frozen**frozen_fraction_clipped) * (
        lambda_sat_unfrozen ** (np.float32(1.0) - frozen_fraction_clipped)
    )
    Ke = (frozen_fraction_clipped * Ke_frozen) + (
        (np.float32(1.0) - frozen_fraction_clipped) * Ke_unfrozen
    )

    # Combine into total thermal conductivity [W/(m·K)]
    # lambda = Ke * lambda_sat + (1 - Ke) * lambda_dry
    lambda_total = Ke * (lambda_sat - lambda_dry) + lambda_dry

    return lambda_total


@njit(cache=True, inline="always")
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

    Calculates absorbed incoming radiation - outgoing longwave radiation.
    Also returns the derivative of the outgoing longwave radiation with respect to temperature,
    which can be used for stability calculations in explicit schemes or damping in implicit schemes.

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
    # Calculate Fluxes
    temperature_K = soil_temperature_C + np.float32(273.15)

    # Beer's law attenuation factor
    attenuation_factor = get_canopy_radiation_attenuation(leaf_area_index)

    absorbed_shortwave_W = (
        (np.float32(1.0) - soil_albedo)
        * shortwave_radiation_W_per_m2
        * attenuation_factor
    )

    # Longwave radiation reaching the soil
    # Atmospheric longwave transmitted through canopy
    transmitted_longwave_W = longwave_radiation_W_per_m2 * attenuation_factor

    # Emissions from canopy
    # We assume canopy temperature ~= air temperature
    canopy_longwave_W = (
        STEFAN_BOLTZMANN_W_PER_M2_K4
        * (air_temperature_K**4)
        * (np.float32(1.0) - attenuation_factor)
    )

    incoming_longwave_at_soil_surface_W = transmitted_longwave_W + canopy_longwave_W

    absorbed_longwave_W = soil_emissivity * incoming_longwave_at_soil_surface_W

    incoming_W = absorbed_shortwave_W + absorbed_longwave_W
    outgoing_W = soil_emissivity * STEFAN_BOLTZMANN_W_PER_M2_K4 * (temperature_K**4)

    net_flux_W = incoming_W - outgoing_W

    # Calculate Derivative of Outgoing Radiation with respect to T:
    # d(sigma * eps * T^4)/dT = 4 * sigma * eps * T^3
    conductance_W_per_m2_K = (
        np.float32(4.0)
        * soil_emissivity
        * STEFAN_BOLTZMANN_W_PER_M2_K4
        * (temperature_K**3)
    )

    return net_flux_W, conductance_W_per_m2_K


@njit(cache=True, inline="always")
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
    # Physics Constants
    SPECIFIC_HEAT_AIR_J_KG_K: np.float32 = np.float32(1005.0)
    GAS_CONSTANT_AIR_J_KG_K: np.float32 = np.float32(287.058)
    VON_KARMAN_CONSTANT: np.float32 = np.float32(0.41)

    # Assumptions for Aerodynamic Resistance over bare soil
    WIND_MEASUREMENT_HEIGHT_M: np.float32 = np.float32(10.0)
    TEMP_MEASUREMENT_HEIGHT_M: np.float32 = np.float32(2.0)
    ROUGHNESS_LENGTH_M: np.float32 = np.float32(0.001)

    # Calculate Air Density [kg/m3]
    # Ideal Gas Law: rho = P / (R * T)
    # Using air temperature for density calculation
    air_density_kg_per_m3: np.float32 = surface_pressure_pa / (
        GAS_CONSTANT_AIR_J_KG_K * air_temperature_K
    )

    # Calculate Aerodynamic Resistance (ra) [s/m]
    # For neutral conditions: ra = (ln((zm-d)/z0m) * ln((zh-d)/z0h)) / (k^2 * u)
    # Assuming d=0, z0m=z0h=z0
    # where zm = wind measurement height, zh = temp measurement height, z0 = roughness length, u = wind speed

    # Ensure minimum wind speed to avoid division by zero
    wind_speed_safe = np.maximum(wind_speed_10m_m_per_s, np.float32(0.1))

    log_wind_height_over_roughness = np.log(
        WIND_MEASUREMENT_HEIGHT_M / ROUGHNESS_LENGTH_M
    )
    log_temp_height_over_roughness = np.log(
        TEMP_MEASUREMENT_HEIGHT_M / ROUGHNESS_LENGTH_M
    )

    aerodynamic_resistance_s_per_m = (
        log_wind_height_over_roughness * log_temp_height_over_roughness
    ) / (VON_KARMAN_CONSTANT**2 * wind_speed_safe)

    # Calculate Conductance [W/m2/K]
    # Conductance = rho * Cp / ra
    conductance_W_per_m2_K = (
        air_density_kg_per_m3 * SPECIFIC_HEAT_AIR_J_KG_K
    ) / aerodynamic_resistance_s_per_m

    # Calculate Explicit Sensible Heat Flux [W/m2]
    # H = Conductance * (Ta - Ts)
    air_temperature_C = air_temperature_K - np.float32(273.15)
    temperature_difference_C = air_temperature_C - soil_temperature_C

    sensible_heat_flux_W_per_m2 = conductance_W_per_m2_K * temperature_difference_C

    return sensible_heat_flux_W_per_m2, conductance_W_per_m2_K


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
            water_depth_cell_m * RHO_WATER_KG_PER_M3 * L_FUSION_J_PER_KG
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
    latent_heat_areal_J_per_m2 = water_depth_m * RHO_WATER_KG_PER_M3 * L_FUSION_J_PER_KG

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


@njit(cache=True, inline="always")
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
        - $H > 0$: Fully liquid, $T = H / C_{liq}$
        - $-L < H \leq 0$: Phase change (mushy zone), $T = 0$, $f_{ice} = -H / L$
        - $H \leq -L$: Fully frozen, $T = (H + L) / C_{ice}$

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


@njit(cache=True, inline="always")
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
        fraction_unfrozen * L_VAPORIZATION_J_PER_KG
        + (np.float32(1.0) - fraction_unfrozen) * L_SUBLIMATION_J_PER_KG
    )

    energy_loss_J_per_m2 = evaporation_m * RHO_WATER_KG_PER_M3 * latent_heat

    return soil_enthalpy_top_layer_J_per_m2 - energy_loss_J_per_m2


@njit(cache=True, inline="always")
def apply_rain_heat_advection(
    soil_enthalpy_top_layer_J_per_m2: np.float32,
    infiltration_amount_m: np.float32,
    rain_temperature_C: np.float32,
) -> np.float32:
    """Apply advective heat transport from infiltrating rain to top-layer enthalpy.

    Notes:
        The infiltrating water enters as liquid water at `rain_temperature_C`.
        This updates enthalpy conservatively:
        $H_{new} = H_{old} + m c_w T_{rain}$.

    Args:
        soil_enthalpy_top_layer_J_per_m2: Top-layer enthalpy (J/m2).
        infiltration_amount_m: Infiltration added to the top layer (m).
        rain_temperature_C: Rain temperature (C).

    Returns:
        Updated top-layer enthalpy (J/m2).
    """
    enthalpy_added_J_per_m2: np.float32 = (
        infiltration_amount_m
        * VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K
        * rain_temperature_C
    )
    return soil_enthalpy_top_layer_J_per_m2 + enthalpy_added_J_per_m2


@njit(cache=True, inline="always")
def solve_soil_enthalpy_column(
    soil_enthalpies_J_per_m2: np.ndarray,
    layer_thicknesses_m: np.ndarray,
    bulk_density_kg_per_dm3: np.ndarray,
    solid_heat_capacities_J_per_m2_K: np.ndarray,
    thermal_conductivity_solid_W_per_m_K: np.ndarray,
    water_content_saturated_m: np.ndarray,
    sand_percentage: np.ndarray,
    water_content_m: np.ndarray,
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
    snow_water_equivalent_m: np.float32 = np.float32(0.0),
    snow_temperature_C: np.float32 = np.float32(0.0),
    topwater_m: np.float32 = np.float32(0.0),
) -> tuple[np.ndarray, np.float32, np.ndarray]:
    """Solve the soil enthalpy profile with an implicit scheme.

    The prognostic state is enthalpy H (J/m2) per layer. Temperature and frozen fraction
    are derived diagnostically each nonlinear iteration using a sharp-freezing enthalpy
    formulation (0°C plateau).

    Notes:
        The diffusion term depends on temperature gradients, so we linearize temperature
        w.r.t. enthalpy each iteration using $T \approx \alpha H + \beta$ where
        $\alpha = \partial T/\partial H$.

        For performance and robustness, the soil thermal conductivity is held fixed
        over the timestep. It is computed once from the start-of-timestep frozen
        fraction and then reused for all nonlinear iterations.

    Args:
        soil_enthalpies_J_per_m2: Current layer enthalpies (J/m2).
        layer_thicknesses_m: Layer thicknesses (m).
        bulk_density_kg_per_dm3: Soil bulk density (kg/dm3).
        solid_heat_capacities_J_per_m2_K: Areal heat capacity of the solid fraction (J/m2/K).
        thermal_conductivity_solid_W_per_m_K: Conductivity of solid fraction (W/m/K).
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
        snow_water_equivalent_m: Snow water equivalent (m).
        snow_temperature_C: Snow temperature (C).
        topwater_m: Standing water depth (m).

    Returns:
        Tuple of:
            - Updated enthalpies (J/m2).
            - Soil heat flux (W/m2).
            - Frozen fractions (0-1).
    """
    n_soil_layers = len(soil_enthalpies_J_per_m2)

    # Keep the start-of-timestep state without copying; we never mutate the input array.
    enthalpies_at_start_of_timestep = soil_enthalpies_J_per_m2
    enthalpies_current_iteration = soil_enthalpies_J_per_m2.copy()

    dT_dH_current_iteration = np.zeros_like(soil_enthalpies_J_per_m2)
    beta_current_iteration = np.zeros_like(soil_enthalpies_J_per_m2)

    lower_diagonal_a = np.zeros(n_soil_layers, dtype=soil_enthalpies_J_per_m2.dtype)
    main_diagonal_b = np.zeros(n_soil_layers, dtype=soil_enthalpies_J_per_m2.dtype)
    upper_diagonal_c = np.zeros(n_soil_layers, dtype=soil_enthalpies_J_per_m2.dtype)
    rhs_vector_d = np.zeros(n_soil_layers, dtype=soil_enthalpies_J_per_m2.dtype)

    # Work arrays for allocation-free TDMA and iteration bookkeeping.
    tdma_c_prime = np.empty(n_soil_layers, dtype=soil_enthalpies_J_per_m2.dtype)
    tdma_d_prime = np.empty(n_soil_layers, dtype=soil_enthalpies_J_per_m2.dtype)
    enthalpies_new_iteration = np.empty_like(soil_enthalpies_J_per_m2)
    thermal_conductances_between_layer_centers_W_per_m2_K = np.empty(
        n_soil_layers - 1, dtype=soil_enthalpies_J_per_m2.dtype
    )

    # Freeze thermal conductivity during the timestep:
    # compute it once from the start-of-timestep frozen fraction and reuse.
    frozen_fraction_for_conductivity = np.empty_like(soil_enthalpies_J_per_m2)

    # Precompute per-layer thermodynamic constants used in the scalar enthalpy diagnostics.
    #
    # These depend only on (water content, layer thickness, topwater) which we treat as fixed
    # over this implicit solve. Precomputing once avoids recomputing them for every nonlinear
    # iteration and layer.
    latent_heat_areal_J_per_m2_per_layer = np.empty_like(soil_enthalpies_J_per_m2)
    heat_capacity_liquid_J_per_m2_K_per_layer = np.empty_like(soil_enthalpies_J_per_m2)
    heat_capacity_frozen_J_per_m2_K_per_layer = np.empty_like(soil_enthalpies_J_per_m2)

    for layer_idx in range(n_soil_layers):
        topwater_layer_m = topwater_m if layer_idx == 0 else np.float32(0.0)
        water_depth_m = water_content_m[layer_idx] + topwater_layer_m
        latent_heat_areal_J_per_m2 = (
            water_depth_m * RHO_WATER_KG_PER_M3 * L_FUSION_J_PER_KG
        )
        heat_capacity_liquid_J_per_m2_K = (
            solid_heat_capacities_J_per_m2_K[layer_idx]
            + water_depth_m * VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K
        )
        heat_capacity_frozen_J_per_m2_K = (
            solid_heat_capacities_J_per_m2_K[layer_idx]
            + water_depth_m * VOLUMETRIC_HEAT_CAPACITY_ICE_J_PER_M3_K
        )

        latent_heat_areal_J_per_m2_per_layer[layer_idx] = latent_heat_areal_J_per_m2
        heat_capacity_liquid_J_per_m2_K_per_layer[layer_idx] = (
            heat_capacity_liquid_J_per_m2_K
        )
        heat_capacity_frozen_J_per_m2_K_per_layer[layer_idx] = (
            heat_capacity_frozen_J_per_m2_K
        )

        _, frozen_fraction, _, _ = get_phase_state(
            enthalpy_J_per_m2=enthalpies_at_start_of_timestep[layer_idx],
            latent_heat_areal_J_per_m2=latent_heat_areal_J_per_m2,
            heat_capacity_liquid_J_per_m2_K=heat_capacity_liquid_J_per_m2_K,
            heat_capacity_frozen_J_per_m2_K=heat_capacity_frozen_J_per_m2_K,
        )
        frozen_fraction_for_conductivity[layer_idx] = frozen_fraction

    # Thermal conductivity needs porosity
    porosity_for_conductivity = water_content_saturated_m / layer_thicknesses_m

    # Compute diagnostic volumetric terms once for thermal conductivity
    degree_of_saturation = water_content_m / water_content_saturated_m

    thermal_conductivities_fixed_W_per_m_K = (
        calculate_soil_thermal_conductivity_from_frozen_fraction(
            thermal_conductivity_solid_W_per_m_K=thermal_conductivity_solid_W_per_m_K,
            bulk_density_kg_per_dm3=bulk_density_kg_per_dm3,
            porosity=porosity_for_conductivity,
            degree_of_saturation=degree_of_saturation,
            sand_percentage=sand_percentage,
            frozen_fraction=frozen_fraction_for_conductivity,
        )
    )

    for layer_idx in range(n_soil_layers - 1):
        resistance_upper_half_layer = (
            np.float32(0.5) * layer_thicknesses_m[layer_idx]
        ) / thermal_conductivities_fixed_W_per_m_K[layer_idx]
        resistance_lower_half_layer = (
            np.float32(0.5) * layer_thicknesses_m[layer_idx + 1]
        ) / thermal_conductivities_fixed_W_per_m_K[layer_idx + 1]
        thermal_conductances_between_layer_centers_W_per_m2_K[layer_idx] = np.float32(
            1.0
        ) / (resistance_upper_half_layer + resistance_lower_half_layer)

    MAX_ITERATIONS = 15
    # Convergence tolerance in enthalpy.
    # Rough scaling: an areal heat capacity of ~1e6 J/m2/K means that a 0.01 K
    # change in temperature corresponds to ~1e4 J/m2.
    # We use an enthalpy-based criterion because the sharp-freezing (0°C plateau)
    # formulation has dT/dH = 0 in the mushy zone.
    TOLERANCE_ENTHALPY_J_PER_M2 = np.float32(1.0e4)

    final_net_radiation_flux_W_per_m2 = np.float32(0.0)
    final_sensible_heat_flux_W_per_m2 = np.float32(0.0)

    inv_dt = np.float32(1.0) / timestep_seconds

    for iteration_index in range(MAX_ITERATIONS):
        # Derive temperature, frozen fraction, and linearization T ≈ alpha*H + beta
        surface_temperature_guess_C = np.float32(np.nan)
        for layer_idx in range(n_soil_layers):
            temperature_C, _, dT_dH, beta = get_phase_state(
                enthalpy_J_per_m2=enthalpies_current_iteration[layer_idx],
                latent_heat_areal_J_per_m2=latent_heat_areal_J_per_m2_per_layer[
                    layer_idx
                ],
                heat_capacity_liquid_J_per_m2_K=heat_capacity_liquid_J_per_m2_K_per_layer[
                    layer_idx
                ],
                heat_capacity_frozen_J_per_m2_K=heat_capacity_frozen_J_per_m2_K_per_layer[
                    layer_idx
                ],
            )
            dT_dH_current_iteration[layer_idx] = dT_dH
            beta_current_iteration[layer_idx] = beta

            if layer_idx == 0:
                surface_temperature_guess_C = temperature_C

        if snow_water_equivalent_m > np.float32(0.0):
            net_radiation_flux_W_per_m2 = np.float32(0.0)
            derivative_net_radiation_W_per_m2_K = np.float32(0.0)

            (
                _,
                snow_depth_m,
                snow_thermal_conductivity_W_per_m_K,
            ) = calculate_snow_thermal_properties(snow_water_equivalent_m)
            conductance_distance_m = snow_depth_m * np.float32(0.5)
            snow_conductance_W_per_m2_K = (
                snow_thermal_conductivity_W_per_m_K / conductance_distance_m
            )

            sensible_heat_flux_W_per_m2 = snow_conductance_W_per_m2_K * (
                snow_temperature_C - surface_temperature_guess_C
            )
            derivative_sensible_heat_W_per_m2_K = snow_conductance_W_per_m2_K
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

        final_net_radiation_flux_W_per_m2 = net_radiation_flux_W_per_m2
        final_sensible_heat_flux_W_per_m2 = sensible_heat_flux_W_per_m2

        flux_star_W_per_m2 = (
            net_radiation_flux_W_per_m2
            + sensible_heat_flux_W_per_m2
            + (
                derivative_net_radiation_W_per_m2_K
                + derivative_sensible_heat_W_per_m2_K
            )
            * surface_temperature_guess_C
        )
        surface_thermal_conductance_W_per_m2_K = (
            derivative_net_radiation_W_per_m2_K + derivative_sensible_heat_W_per_m2_K
        )

        # Build the tridiagonal system in H
        # Top layer
        conductance_to_layer_below = (
            thermal_conductances_between_layer_centers_W_per_m2_K[0]
        )
        alpha_0 = dT_dH_current_iteration[0]
        beta_0 = beta_current_iteration[0]
        alpha_1 = dT_dH_current_iteration[1]
        beta_1 = beta_current_iteration[1]

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
            - (surface_thermal_conductance_W_per_m2_K + conductance_to_layer_below)
            * beta_0
            + conductance_to_layer_below * beta_1
        )

        # Intermediate layers
        for layer_idx in range(1, n_soil_layers - 1):
            conductance_to_layer_above = (
                thermal_conductances_between_layer_centers_W_per_m2_K[layer_idx - 1]
            )
            conductance_to_layer_below = (
                thermal_conductances_between_layer_centers_W_per_m2_K[layer_idx]
            )

            alpha_im1 = dT_dH_current_iteration[layer_idx - 1]
            beta_im1 = beta_current_iteration[layer_idx - 1]
            alpha_i = dT_dH_current_iteration[layer_idx]
            beta_i = beta_current_iteration[layer_idx]
            alpha_ip1 = dT_dH_current_iteration[layer_idx + 1]
            beta_ip1 = beta_current_iteration[layer_idx + 1]

            lower_diagonal_a[layer_idx] = -conductance_to_layer_above * alpha_im1
            main_diagonal_b[layer_idx] = (
                inv_dt
                + (conductance_to_layer_above + conductance_to_layer_below) * alpha_i
            )
            upper_diagonal_c[layer_idx] = -conductance_to_layer_below * alpha_ip1
            rhs_vector_d[layer_idx] = (
                inv_dt * enthalpies_at_start_of_timestep[layer_idx]
                - (conductance_to_layer_above + conductance_to_layer_below) * beta_i
                + conductance_to_layer_above * beta_im1
                + conductance_to_layer_below * beta_ip1
            )

        # Bottom layer (Dirichlet)
        last_idx = n_soil_layers - 1
        conductance_to_layer_above = (
            thermal_conductances_between_layer_centers_W_per_m2_K[last_idx - 1]
        )
        conductance_to_deep_soil_boundary_W_per_m2_K = (
            thermal_conductivities_fixed_W_per_m_K[last_idx]
            / (np.float32(0.5) * layer_thicknesses_m[last_idx])
        )

        alpha_last = dT_dH_current_iteration[last_idx]
        beta_last = beta_current_iteration[last_idx]
        alpha_above = dT_dH_current_iteration[last_idx - 1]
        beta_above = beta_current_iteration[last_idx - 1]

        lower_diagonal_a[last_idx] = -conductance_to_layer_above * alpha_above
        main_diagonal_b[last_idx] = (
            inv_dt
            + (
                conductance_to_layer_above
                + conductance_to_deep_soil_boundary_W_per_m2_K
            )
            * alpha_last
        )
        upper_diagonal_c[last_idx] = np.float32(0.0)
        rhs_vector_d[last_idx] = (
            inv_dt * enthalpies_at_start_of_timestep[last_idx]
            + conductance_to_deep_soil_boundary_W_per_m2_K * deep_soil_temperature_C
            - (
                conductance_to_layer_above
                + conductance_to_deep_soil_boundary_W_per_m2_K
            )
            * beta_last
            + conductance_to_layer_above * beta_above
        )

        tdma_solver(
            lower_diagonal_a,
            main_diagonal_b,
            upper_diagonal_c,
            rhs_vector_d,
            enthalpies_new_iteration,
            tdma_c_prime,
            tdma_d_prime,
        )

        # Convergence check in enthalpy.
        max_enthalpy_correction_J_per_m2 = np.float32(0.0)
        for layer_idx in range(n_soil_layers):
            enthalpy_correction_J_per_m2 = abs(
                enthalpies_new_iteration[layer_idx]
                - enthalpies_current_iteration[layer_idx]
            )
            if enthalpy_correction_J_per_m2 > max_enthalpy_correction_J_per_m2:
                max_enthalpy_correction_J_per_m2 = enthalpy_correction_J_per_m2

        enthalpies_current_iteration[:] = enthalpies_new_iteration

        # With frozen conductivity, the remaining nonlinearity is only in T(H).
        # Converging in H is more robust than converging in T near 0 C.
        if max_enthalpy_correction_J_per_m2 < TOLERANCE_ENTHALPY_J_PER_M2:
            break

    soil_heat_flux_W_per_m2 = (
        final_net_radiation_flux_W_per_m2 + final_sensible_heat_flux_W_per_m2
    )

    frozen_fractions_final = np.empty_like(soil_enthalpies_J_per_m2)
    for layer_idx in range(n_soil_layers):
        _, frozen_fraction, _, _ = get_phase_state(
            enthalpy_J_per_m2=enthalpies_current_iteration[layer_idx],
            latent_heat_areal_J_per_m2=latent_heat_areal_J_per_m2_per_layer[layer_idx],
            heat_capacity_liquid_J_per_m2_K=heat_capacity_liquid_J_per_m2_K_per_layer[
                layer_idx
            ],
            heat_capacity_frozen_J_per_m2_K=heat_capacity_frozen_J_per_m2_K_per_layer[
                layer_idx
            ],
        )
        frozen_fractions_final[layer_idx] = frozen_fraction

    return (
        enthalpies_current_iteration,
        soil_heat_flux_W_per_m2,
        frozen_fractions_final,
    )
