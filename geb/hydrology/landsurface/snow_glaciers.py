"""Functions to calculate snow accumulation and melt."""

import numpy as np
from numba import njit

from .constants import (
    KELVIN_OFFSET,
    LATENT_HEAT_FUSION_J_PER_KG,
    LATENT_HEAT_SUBLIMATION_J_PER_KG,
    LATENT_HEAT_VAPORIZATION_J_PER_KG,
    RHO_WATER_KG_PER_M3,
    SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K,
)
from .energy import (
    calculate_aerodynamic_conductance_W_per_m2_K,
)

# Max top-layer SWE that can accumulate before excess flows to the bottom
MAX_TOP_LAYER_SWE_M: np.float64 = np.float64(0.05)

# Density of newly fallen snow (kg/m³).
FRESH_SNOW_DENSITY_KG_PER_M3: np.float32 = np.float32(100.0)

# Gravitational acceleration (m/s²).
GRAVITY_M_PER_S2: np.float32 = np.float32(9.81)

# Floating point epsilon for mass balance checks.
EPSILON_M: np.float64 = np.float64(1e-14)


@njit(cache=True, inline="always")
def get_snow_enthalpy_from_temperature(
    snow_water_equivalent_m: np.float64,
    snow_temperature_C: np.float32,
) -> np.float32:
    """Convert snow temperature to bulk snow enthalpy.

    Args:
        snow_water_equivalent_m: Snow water equivalent of the frozen snowpack (m).
        snow_temperature_C: Bulk snow temperature (°C).

    Returns:
        Bulk snow enthalpy (J/m²).
    """
    snow_mass_kg_per_m2: np.float64 = snow_water_equivalent_m * RHO_WATER_KG_PER_M3
    bounded_temperature_C: np.float32 = min(np.float32(0.0), snow_temperature_C)
    return np.float32(
        snow_mass_kg_per_m2
        * SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K
        * bounded_temperature_C
    )


@njit(cache=True, inline="always")
def get_snow_temperature_from_enthalpy(
    snow_water_equivalent_m: np.float64,
    snow_enthalpy_J_per_m2: np.float32,
) -> np.float32:
    """Diagnose bulk snow temperature from snow enthalpy.

    Args:
        snow_water_equivalent_m: Snow water equivalent of the frozen snowpack (m).
        snow_enthalpy_J_per_m2: Bulk snow enthalpy (J/m²).

    Returns:
        Bulk snow temperature (°C).
    """
    snow_heat_capacity_J_per_m2_K: np.float64 = (
        snow_water_equivalent_m
        * np.float64(RHO_WATER_KG_PER_M3)
        * np.float64(SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K)
    )
    if snow_heat_capacity_J_per_m2_K <= np.float64(1e-10):
        return np.float32(0.0)

    temperature_C: np.float32 = np.float32(
        np.float64(snow_enthalpy_J_per_m2) / snow_heat_capacity_J_per_m2_K
    )
    return min(np.float32(0.0), temperature_C)


@njit(cache=False, inline="always")
def calculate_snow_metamorphism_compaction_rate_per_s(
    density_kg_per_m3: np.float32,
    snow_temperature_C: np.float32,
) -> np.float32:
    """Calculate snow compaction rate due to thermal metamorphism.

    Args:
        density_kg_per_m3: Current snow density (kg/m³).
        snow_temperature_C: Bulk snow temperature (°C).

    Returns:
        Fractional compaction rate (1/s).
    """
    # Temperature difference from freezing point (K).
    delta_T_K: np.float32 = max(np.float32(0.0), -snow_temperature_C)

    # Thermal metamorphism term: equation 8.38 in
    # https://www.ecmwf.int/sites/default/files/elibrary/2016/17117-part-iv-physical-processes.pdf
    # a_ξ = 2.8 ×10−6 s−1
    # b_ξ = 4.2 ×10−2
    # cξ = 460 m3
    # ρξ = 150 kg/m3
    return np.float32(2.8e-6) * np.exp(
        -np.float32(0.042) * delta_T_K
        - np.float32(0.046)
        * max(
            np.float32(0.0),
            density_kg_per_m3 - np.float32(150.0),
        )
    )


@njit(cache=True, inline="always")
def calculate_snow_overburden_compaction_rate_per_s(
    density_kg_per_m3: np.float32,
    snow_temperature_C: np.float32,
    overburden_pressure_Pa: np.float32,
) -> np.float32:
    """Calculate snow compaction rate due to overburden pressure.

    Args:
        density_kg_per_m3: Current snow density (kg/m³).
        snow_temperature_C: Bulk snow temperature (°C).
        overburden_pressure_Pa: Pressure from overlying snow layers (Pa).

    Returns:
        Fractional compaction rate (1/s).
    """
    # Temperature difference from freezing point (K).
    delta_T_K: np.float32 = max(np.float32(0.0), -snow_temperature_C)

    # Viscosity term: equation 8.37 in
    # https://www.ecmwf.int/sites/default/files/elibrary/2016/17117-part-iv-physical-processes.pdf
    viscosity_Pa_s: np.float32 = np.float32(3.7e7) * np.exp(
        np.float32(0.081) * delta_T_K + np.float32(0.018) * density_kg_per_m3
    )
    return overburden_pressure_Pa / viscosity_Pa_s


@njit(cache=True, inline="always")
def compact_snow_density(
    density_kg_per_m3: np.float32,
    snow_temperature_C: np.float32,
    overburden_pressure_Pa: np.float32,
) -> np.float32:
    """Compact snow density due to metamorphism and overburden pressure.

    Args:
        density_kg_per_m3: Current snow density (kg/m³).
        snow_temperature_C: Bulk snow temperature (°C).
        overburden_pressure_Pa: Pressure from overlying snow layers (Pa).

    Returns:
        Compacted snow density after one hour (kg/m³).
    """
    metamorphism_rate_per_s: np.float32 = (
        calculate_snow_metamorphism_compaction_rate_per_s(
            density_kg_per_m3,
            snow_temperature_C,
        )
    )
    overburden_rate_per_s: np.float32 = calculate_snow_overburden_compaction_rate_per_s(
        density_kg_per_m3,
        snow_temperature_C,
        overburden_pressure_Pa,
    )

    # Total fractional compaction rate (1/s).
    total_rate_per_s: np.float32 = metamorphism_rate_per_s + overburden_rate_per_s

    # Update density for one hour (3600 seconds).
    return density_kg_per_m3 * (np.float32(1.0) + total_rate_per_s * np.float32(3600.0))


@njit(cache=True, inline="always")
def discriminate_precipitation(
    precipitation_m_per_hour: np.float32,
    air_temperature_C: np.float32,
) -> tuple[np.float32, np.float32]:
    """
    Discriminate between snowfall and rainfall based on temperature.

    Args:
        precipitation_m_per_hour: Precipitation rate (m/hour).
        air_temperature_C: Air temperature (°C).

    Returns:
        A tuple of snowfall rate (m/hour) and rainfall rate (m/hour).
    """
    is_snow: np.bool_ = air_temperature_C <= np.float32(0.0)
    if is_snow:
        # If it's snowing, all precipitation is snowfall and there is no rainfall.
        snowfall_m_per_hour: np.float32 = np.float32(precipitation_m_per_hour)
        rainfall_m_per_hour: np.float32 = np.float32(0.0)
    else:
        # If it's not snowing, all precipitation is rainfall and there is no snowfall.
        snowfall_m_per_hour: np.float32 = np.float32(0.0)
        rainfall_m_per_hour: np.float32 = np.float32(precipitation_m_per_hour)
    return snowfall_m_per_hour, rainfall_m_per_hour


@njit(cache=True, inline="always")
def add_snowfall_to_top_layer(
    snow_water_equivalent_m: np.float64,
    snow_enthalpy_J_per_m2: np.float32,
    snowfall_m_per_hour: np.float32,
    air_temperature_C: np.float32,
) -> tuple[np.float64, np.float32]:
    """Add snowfall mass and enthalpy to the frozen snow reservoir.

    Args:
        snow_water_equivalent_m: Existing frozen snow water equivalent (m).
        snow_enthalpy_J_per_m2: Existing frozen-snow enthalpy (J/m²).
        snowfall_m_per_hour: Snowfall added during the step (m/hour).
        air_temperature_C: Air temperature of the snowfall source (°C).

    Returns:
        Tuple of updated frozen snow water equivalent (m) and enthalpy (J/m²).
    """
    # In case of no snowfall, return the original
    if snowfall_m_per_hour == np.float32(0.0):
        return snow_water_equivalent_m, snow_enthalpy_J_per_m2

    # In principle, there is only snow when the air temperature is
    # at or below freezing, but in reality snow can fall at slightly above 0°C.
    # So the min here is mostly to ensure that future updates
    # don't break this.
    snowfall_temperature_C: np.float32 = min(np.float32(0.0), air_temperature_C)
    snowfall_enthalpy_J_per_m2: np.float32 = get_snow_enthalpy_from_temperature(
        np.float64(snowfall_m_per_hour),
        snowfall_temperature_C,
    )
    return (
        snow_water_equivalent_m + np.float64(snowfall_m_per_hour),
        snow_enthalpy_J_per_m2 + snowfall_enthalpy_J_per_m2,
    )


@njit(cache=True, inline="always")
def split_snow_enthalpy(
    snow_water_equivalent_m: np.float64,
    snow_enthalpy_J_per_m2: np.float32,
    transferred_snow_water_equivalent_m: np.float64,
) -> tuple[np.float32, np.float32]:
    """Split snow enthalpy proportionally with transferred frozen mass.

    Frequently, snow "jumps" from one layer to the other. Of course, in reality
    this wouldn't happen, but the layers are the model's way of tracking vertical structure.
    When the snow jumps, the enthalpy must be partitioned proportionally
    to the mass transfer to maintain energy balance.

    Args:
        snow_water_equivalent_m: Total frozen snow water equivalent before transfer (m).
        snow_enthalpy_J_per_m2: Total frozen-snow enthalpy before transfer (J/m²).
        transferred_snow_water_equivalent_m: Frozen snow water equivalent transferred out (m).

    Returns:
        Tuple of remaining and transferred enthalpy (J/m²).
    """
    if transferred_snow_water_equivalent_m <= np.float64(0.0):
        return snow_enthalpy_J_per_m2, np.float32(0.0)

    if (
        transferred_snow_water_equivalent_m >= snow_water_equivalent_m
        or snow_water_equivalent_m <= EPSILON_M
    ):
        return np.float32(0.0), snow_enthalpy_J_per_m2

    transfer_fraction: np.float64 = (
        transferred_snow_water_equivalent_m / snow_water_equivalent_m
    )
    transferred_enthalpy_J_per_m2: np.float32 = np.float32(
        np.float64(snow_enthalpy_J_per_m2) * transfer_fraction
    )
    return (
        np.float32(
            np.float64(snow_enthalpy_J_per_m2)
            - np.float64(transferred_enthalpy_J_per_m2)
        ),
        transferred_enthalpy_J_per_m2,
    )


@njit(cache=True, inline="always")
def mix_snow_properties(
    swe_1_m: np.float64,
    density_1_kg_per_m3: np.float32,
    liquid_1_m: np.float64,
    swe_2_m: np.float64,
    density_2_kg_per_m3: np.float32,
    liquid_2_m: np.float64,
) -> tuple[np.float64, np.float32, np.float64]:
    """Mix properties of two snow masses (mass-weighted average).

    Args:
        swe_1_m: Snow water equivalent of first mass (m).
        density_1_kg_per_m3: Density of first mass (kg/m³).
        liquid_1_m: Liquid water in first mass (m).
        swe_2_m: Snow water equivalent of second mass (m).
        density_2_kg_per_m3: Density of second mass (kg/m³).
        liquid_2_m: Liquid water in second mass (m).

    Returns:
        Combined SWE (m), mass-weighted density (kg/m³), and total liquid water (m).
    """
    total_swe_m: np.float64 = swe_1_m + swe_2_m
    total_liquid_m: np.float64 = liquid_1_m + liquid_2_m

    if total_swe_m > np.float64(0.0):
        # Weighted average of densities.
        mixed_density_kg_per_m3: np.float32 = np.float32(
            (
                np.float64(density_1_kg_per_m3) * swe_1_m
                + np.float64(density_2_kg_per_m3) * swe_2_m
            )
            / total_swe_m
        )
    else:
        mixed_density_kg_per_m3 = density_1_kg_per_m3

    return total_swe_m, mixed_density_kg_per_m3, total_liquid_m


@njit(cache=True, inline="always")
def promote_snow_to_top_layer(
    swe_top_m: np.float64,
    liquid_water_top_m: np.float64,
    enthalpy_top_J_per_m2: np.float32,
    density_top_kg_per_m3: np.float32,
    swe_bottom_m: np.float64,
    liquid_water_bottom_m: np.float64,
    enthalpy_bottom_J_per_m2: np.float32,
    density_bottom_kg_per_m3: np.float32,
) -> tuple[
    np.float64,
    np.float64,
    np.float32,
    np.float32,
    np.float64,
    np.float64,
    np.float32,
    np.float32,
]:
    """Move snow from bottom layer to the top layer when the top layer is thinner than its target thickness.

    Notes:
        The top layer should stay at its target thickness (MAX_TOP_LAYER_SWE_M).
        If it becomes thinner due to melt or sublimation, mass is 'pulled'
        up from the bottom layer to replenish it.

    Args:
        swe_top_m: Top-layer frozen snow water equivalent (m).
        liquid_water_top_m: Top-layer liquid water storage (m).
        enthalpy_top_J_per_m2: Top-layer snow enthalpy (J/m²).
        density_top_kg_per_m3: Top-layer snow density (kg/m³).
        swe_bottom_m: Bottom-layer frozen snow water equivalent (m).
        liquid_water_bottom_m: Bottom-layer liquid water storage (m).
        enthalpy_bottom_J_per_m2: Bottom-layer snow enthalpy (J/m²).
        density_bottom_kg_per_m3: Bottom-layer snow density (kg/m³).

    Returns:
        Snow state with mass replenished in the top layer if possible.
    """
    if swe_top_m < np.float64(MAX_TOP_LAYER_SWE_M) and swe_bottom_m > np.float64(0.0):
        # Calculate how much frozen mass we want to move to refill the top layer.
        transfer_m: np.float64 = min(
            np.float64(MAX_TOP_LAYER_SWE_M) - swe_top_m, swe_bottom_m
        )

        # Proportional enthalpy transfer.
        enthalpy_bottom_J_per_m2, trans_enthalpy_J_per_m2 = split_snow_enthalpy(
            swe_bottom_m, enthalpy_bottom_J_per_m2, transfer_m
        )
        enthalpy_top_J_per_m2 += trans_enthalpy_J_per_m2

        # Pull up liquid water proportionally.
        liquid_transfer_m: np.float64 = liquid_water_bottom_m * (
            transfer_m / swe_bottom_m
        )

        # Update top layer properties (mass-weighted).
        swe_top_m, density_top_kg_per_m3, liquid_water_top_m = mix_snow_properties(
            swe_top_m,
            density_top_kg_per_m3,
            liquid_water_top_m,
            transfer_m,
            density_bottom_kg_per_m3,
            liquid_transfer_m,
        )

        swe_bottom_m -= transfer_m
        liquid_water_bottom_m -= liquid_transfer_m

    return (
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
    )


@njit(cache=True, inline="always")
def apply_precipitation_compaction_and_top_layer_transfer(
    pr_kg_per_m2_per_s: np.float32,
    air_temperature_C: np.float32,
    swe_top_m: np.float64,
    liquid_water_top_m: np.float64,
    enthalpy_top_J_per_m2: np.float32,
    density_top_kg_per_m3: np.float32,
    swe_bottom_m: np.float64,
    liquid_water_bottom_m: np.float64,
    enthalpy_bottom_J_per_m2: np.float32,
    density_bottom_kg_per_m3: np.float32,
) -> tuple[
    np.float32,
    np.float32,
    np.float64,
    np.float64,
    np.float32,
    np.float32,
    np.float64,
    np.float64,
    np.float32,
    np.float32,
]:
    """Apply precipitation partitioning, compaction, and top-layer transfer.

    Notes:
        Partition precipitation into rain and snow, compact both snow layers,
        add snowfall to the top layer, and move
        any excess top-layer snow mass into the bottom layer.

    Args:
        pr_kg_per_m2_per_s: Precipitation rate (kg/m²/s).
        air_temperature_C: Air temperature (°C).
        swe_top_m: Top-layer frozen snow water equivalent (m).
        liquid_water_top_m: Top-layer liquid water storage (m).
        enthalpy_top_J_per_m2: Top-layer snow enthalpy (J/m²).
        density_top_kg_per_m3: Top-layer snow density (kg/m³).
        swe_bottom_m: Bottom-layer frozen snow water equivalent (m).
        liquid_water_bottom_m: Bottom-layer liquid water storage (m).
        enthalpy_bottom_J_per_m2: Bottom-layer snow enthalpy (J/m²).
        density_bottom_kg_per_m3: Bottom-layer snow density (kg/m³).

    Returns:
        Rainfall, snowfall, and updated snow state ready for the coupled energy solve.
    """
    (
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
    ) = promote_snow_to_top_layer(
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
    )

    # Get temperatures for compaction.
    temperature_top_C: np.float32 = get_snow_temperature_from_enthalpy(
        swe_top_m, enthalpy_top_J_per_m2
    )
    temperature_bottom_C: np.float32 = get_snow_temperature_from_enthalpy(
        swe_bottom_m, enthalpy_bottom_J_per_m2
    )

    # Calculate overburden pressure (Pa).
    # Top layer has 50% of its own weight as overburden (centered pressure).
    # Bottom layer has 100% of top layer + 50% of its own weight.
    overburden_top_Pa: np.float32 = (
        np.float32(0.5) * np.float32(swe_top_m) * RHO_WATER_KG_PER_M3 * GRAVITY_M_PER_S2
    )
    overburden_bottom_Pa: np.float32 = (
        (np.float32(swe_top_m) + np.float32(0.5) * np.float32(swe_bottom_m))
        * RHO_WATER_KG_PER_M3
        * GRAVITY_M_PER_S2
    )

    # Apply compaction
    density_top_kg_per_m3: np.float32 = compact_snow_density(
        density_top_kg_per_m3,
        temperature_top_C,
        overburden_top_Pa,
    )

    density_bottom_kg_per_m3: np.float32 = compact_snow_density(
        density_bottom_kg_per_m3,
        temperature_bottom_C,
        overburden_bottom_Pa,
    )

    # Identify precipitation types.
    precip_m_hr: np.float32 = pr_kg_per_m2_per_s * np.float32(3.6)
    snowfall_m_hr, rainfall_m_hr = discriminate_precipitation(
        precip_m_hr, air_temperature_C
    )

    # Add snowfall to top layer and update its density.
    if snowfall_m_hr > 0.0:
        swe_top_m, density_top_kg_per_m3, _ = mix_snow_properties(
            swe_top_m,
            density_top_kg_per_m3,
            np.float64(0.0),
            np.float64(snowfall_m_hr),
            np.float32(FRESH_SNOW_DENSITY_KG_PER_M3),
            np.float64(0.0),
        )
        swe_top_m, enthalpy_top_J_per_m2 = add_snowfall_to_top_layer(
            swe_top_m
            - np.float64(
                snowfall_m_hr
            ),  # Pass original SWE for the adder function logic
            enthalpy_top_J_per_m2,
            snowfall_m_hr,
            air_temperature_C,
        )

    # If the top layer exceeds its max SWE, transfer the excess to the bottom layer.
    if swe_top_m > np.float64(MAX_TOP_LAYER_SWE_M):
        excess_swe_m: np.float64 = swe_top_m - np.float64(MAX_TOP_LAYER_SWE_M)

        # Proportional enthalpy and liquid water transfer.
        enthalpy_top_J_per_m2, trans_enthalpy_J_per_m2 = split_snow_enthalpy(
            swe_top_m, enthalpy_top_J_per_m2, excess_swe_m
        )
        liquid_trans_m: np.float64 = liquid_water_top_m * (excess_swe_m / swe_top_m)

        # Update bottom layer properties.
        swe_bottom_m, density_bottom_kg_per_m3, liquid_water_bottom_m = (
            mix_snow_properties(
                swe_bottom_m,
                density_bottom_kg_per_m3,
                liquid_water_bottom_m,
                excess_swe_m,
                density_top_kg_per_m3,
                liquid_trans_m,
            )
        )

        swe_top_m = np.float64(MAX_TOP_LAYER_SWE_M)
        liquid_water_top_m -= liquid_trans_m
        enthalpy_bottom_J_per_m2 += trans_enthalpy_J_per_m2

    return (
        rainfall_m_hr,
        snowfall_m_hr,
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
    )


@njit(cache=True, inline="always")
def melt_snow_from_enthalpy(
    snow_water_equivalent_m: np.float64,
    snow_enthalpy_J_per_m2: np.float32,
) -> tuple[np.float32, np.float64, np.float32]:
    """Convert positive snow enthalpy into melt.

    0 enthalpy means the snow is at 0°C and thus any positive enthalpy
    represents energy available to melt the snow.

    Args:
        snow_water_equivalent_m: Frozen snow water equivalent (m).
        snow_enthalpy_J_per_m2: Frozen snow enthalpy (J/m²).

    Returns:
        Tuple of melt amount (m/hour equivalent for the timestep), updated SWE (m),
        and updated snow enthalpy (J/m²).
    """
    # If there is no snow, or the enthalpy is negative
    # no melt occurs and we just return
    if snow_water_equivalent_m <= np.float64(
        0.0
    ) or snow_enthalpy_J_per_m2 <= np.float32(0.0):
        # no melt, original SWE and enthalpy remain unchanged
        return (np.float32(0.0), snow_water_equivalent_m, snow_enthalpy_J_per_m2)

    # Calculate actual melt, limited by total available SWE.
    potential_melt_m: np.float64 = np.float64(snow_enthalpy_J_per_m2) / (
        np.float64(RHO_WATER_KG_PER_M3) * np.float64(LATENT_HEAT_FUSION_J_PER_KG)
    )
    actual_melt_m: np.float64 = min(potential_melt_m, snow_water_equivalent_m)
    actual_melt_m_float32: np.float32 = np.float32(actual_melt_m)
    updated_swe_m: np.float64 = snow_water_equivalent_m - actual_melt_m

    # After melting, the snowpack will have 0 enthalpy because it's at the melting point.
    # Any excess enthalpy beyond what was needed to melt the available snow is now in the
    # meltwater, but since we don't track this (yet), enthalpy is 0 now.
    updated_enthalpy_J_per_m2: np.float32 = np.float32(0.0)

    return (
        actual_melt_m_float32,
        updated_swe_m,
        updated_enthalpy_J_per_m2,
    )


@njit(cache=True, inline="always")
def update_snow_mass_and_phase(
    rainfall_m_per_hour: np.float32,
    swe_top_m: np.float64,
    liquid_water_top_m: np.float64,
    enthalpy_top_J_per_m2: np.float32,
    density_top_kg_per_m3: np.float32,
    swe_bottom_m: np.float64,
    liquid_water_bottom_m: np.float64,
    enthalpy_bottom_J_per_m2: np.float32,
    density_bottom_kg_per_m3: np.float32,
    air_temperature_C: np.float32,
    vapor_pressure_air_Pa: np.float32,
    air_pressure_Pa: np.float32,
    wind_10m_m_per_s: np.float32,
    activate_layer_thickness_m: np.float32,
) -> tuple[
    np.float64,
    np.float64,
    np.float32,
    np.float32,
    np.float64,
    np.float64,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
]:
    """Snow mass balance and phase change updates.

    This function updates the snowpack state by applying mass changes that
    result from the solved enthalpy state (melt/refreeze) and atmospheric
    exchange (sublimation/deposition).

    Args:
        rainfall_m_per_hour: Rainfall incident on the snowpack (m/hour).
        swe_top_m: Top-layer frozen snow water equivalent (m).
        liquid_water_top_m: Top-layer liquid water storage (m).
        enthalpy_top_J_per_m2: Top-layer snow enthalpy (J/m²).
        density_top_kg_per_m3: Top-layer snow density (kg/m³).
        swe_bottom_m: Bottom-layer frozen snow water equivalent (m).
        liquid_water_bottom_m: Bottom-layer liquid water storage (m).
        enthalpy_bottom_J_per_m2: Bottom-layer snow enthalpy (J/m²).
        density_bottom_kg_per_m3: Bottom-layer snow density (kg/m³).
        air_temperature_C: Air temperature (°C).
        vapor_pressure_air_Pa: Air vapour pressure (Pa).
        air_pressure_Pa: Air pressure (Pa).
        wind_10m_m_per_s: Wind speed at 10 m height (m/s).
        activate_layer_thickness_m: Active-layer thickness for refreezing/runoff (m).

    Returns:
        Updated snow state (layers 0/1) and diagnostic hourly fluxes (m/hour).
    """
    (
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
    ) = promote_snow_to_top_layer(
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
    )

    # Sublimation/deposition
    snow_surface_temperature_C: np.float32 = get_snow_temperature_from_enthalpy(
        swe_top_m, enthalpy_top_J_per_m2
    )
    latent_heat_flux_W_per_m2, sublimation_rate_m_per_hour = (
        calculate_latent_heat_flux_and_sublimation(
            air_temperature_C,
            snow_surface_temperature_C,
            vapor_pressure_air_Pa,
            air_pressure_Pa,
            wind_10m_m_per_s,
        )
    )

    # No deposition above freezing
    if air_temperature_C > np.float32(0.0):
        sublimation_rate_m_per_hour: np.float32 = np.float32(0.0)
        latent_heat_flux_W_per_m2: np.float32 = np.float32(0.0)

    applied_sublimation_m_per_hour: np.float32 = max(
        sublimation_rate_m_per_hour, np.float32(-swe_top_m)
    )

    # Cooling limit: prevent over-cooling thin layers below air temperature.
    potential_energy_added_J_per_m2: np.float32 = (
        latent_heat_flux_W_per_m2 * np.float32(3600.0)
    )
    if potential_energy_added_J_per_m2 < np.float32(0.0):
        min_enthalpy: np.float32 = get_snow_enthalpy_from_temperature(
            swe_top_m, air_temperature_C
        )
        max_cooling: np.float32 = max(
            np.float32(0.0), enthalpy_top_J_per_m2 - min_enthalpy
        )

        # If the potential cooling exceeds the maximum allowed cooling,
        # we need to scale back the sublimation rate accordingly.
        if -potential_energy_added_J_per_m2 > max_cooling:
            scaling_factor: np.float32 = (
                max_cooling / (-potential_energy_added_J_per_m2)
                if potential_energy_added_J_per_m2 < np.float32(-1e-9)
                else np.float32(0.0)
            )
            potential_energy_added_J_per_m2: np.float32 = -max_cooling
            applied_sublimation_m_per_hour *= scaling_factor

    # Apply sublimation/deposition mass and energy changes together.
    if applied_sublimation_m_per_hour < 0:
        # Sublimation: remove mass and its proportionate enthalpy content.
        enthalpy_top_J_per_m2, _ = split_snow_enthalpy(
            swe_top_m,
            enthalpy_top_J_per_m2,
            np.float64(-applied_sublimation_m_per_hour),
        )
    elif applied_sublimation_m_per_hour > 0:
        # Deposition: add mass and its sensible enthalpy at surface temperature.
        enthalpy_top_J_per_m2 += get_snow_enthalpy_from_temperature(
            np.float64(applied_sublimation_m_per_hour), snow_surface_temperature_C
        )

    swe_top_m += np.float64(applied_sublimation_m_per_hour)
    enthalpy_top_J_per_m2 += potential_energy_added_J_per_m2

    # Refreezing and Melt
    refreezing_m_per_hour, swe_top_m, liquid_water_top_m, enthalpy_top_J_per_m2 = (
        handle_refreezing(
            enthalpy_top_J_per_m2,
            liquid_water_top_m,
            swe_top_m,
            activate_layer_thickness_m,
        )
    )

    melt_top_m, swe_top_m, enthalpy_top_J_per_m2 = melt_snow_from_enthalpy(
        swe_top_m, enthalpy_top_J_per_m2
    )
    melt_bottom_m, swe_bottom_m, enthalpy_bottom_J_per_m2 = melt_snow_from_enthalpy(
        swe_bottom_m, enthalpy_bottom_J_per_m2
    )

    snow_melt_m_per_hour = melt_top_m + melt_bottom_m
    liquid_water_top_m += np.float64(snow_melt_m_per_hour) + np.float64(
        rainfall_m_per_hour
    )

    # Runoff
    melt_runoff_m_per_hour, liquid_water_top_m = calculate_runoff(
        liquid_water_top_m,
        swe_top_m,
        activate_layer_thickness_m,
    )

    # If the top layer is too thin, pull up snow from the bottom layer
    (
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
    ) = promote_snow_to_top_layer(
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
    )

    return (
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
        snow_melt_m_per_hour,
        melt_runoff_m_per_hour,
        rainfall_m_per_hour,
        applied_sublimation_m_per_hour,
        refreezing_m_per_hour,
    )


@njit(cache=True, inline="always")
def calculate_latent_heat_flux_and_sublimation(
    air_temperature_C: np.float32,
    snow_surface_temperature_C: np.float32,
    vapor_pressure_air_Pa: np.float32,
    air_pressure_Pa: np.float32,
    wind_10m_m_per_s: np.float32,
) -> tuple[np.float32, np.float32]:
    """
    Calculate latent heat flux and sublimation/deposition rate.

    Args:
        air_temperature_C: Air temperature (°C).
        snow_surface_temperature_C: Snow surface temperature (°C).
        vapor_pressure_air_Pa: Vapor pressure of the air (Pa).
        air_pressure_Pa: Air pressure (Pa).
        wind_10m_m_per_s: Wind speed (m/s).

    Returns:
        A tuple of latent heat flux (W/m²) and sublimation/deposition rate
        (m/hour, negative for sublimation).
    """
    air_temperature_K = air_temperature_C + KELVIN_OFFSET

    # Latent Heat Flux (Q_L)
    # Based on bulk aerodynamic formula: Q_l = rho * L * C_e * U * (q_air - q_surf)
    # Use latent heat of sublimation for snow (ice) surfaces, vaporization for liquid.
    latent_heat_J_per_kg = (
        LATENT_HEAT_VAPORIZATION_J_PER_KG
        if snow_surface_temperature_C >= np.float32(0.0)
        else LATENT_HEAT_SUBLIMATION_J_PER_KG
    )

    aerodynamic_conductance_W_per_m2_K = calculate_aerodynamic_conductance_W_per_m2_K(
        air_temperature_K, wind_10m_m_per_s, air_pressure_Pa
    )

    # Saturation vapor pressure at snow surface (e_surf)
    # Use Buck's equation for saturation vapor pressure over ice
    e_surf_denominator = np.float32(272.62) + snow_surface_temperature_C
    e_surf = np.float32(611.15) * np.exp(
        (np.float32(22.46) * snow_surface_temperature_C) / e_surf_denominator
    )

    # Specific humidity of air and surface
    # q = 0.622 * e / (P - 0.378 * e)
    specific_humidity_air = (np.float32(0.622) * vapor_pressure_air_Pa) / (
        air_pressure_Pa - np.float32(0.378) * vapor_pressure_air_Pa
    )
    specific_humidity_surface = (np.float32(0.622) * e_surf) / (
        air_pressure_Pa - np.float32(0.378) * e_surf
    )

    SPECIFIC_HEAT_AIR_J_PER_KG_K = np.float32(1005.0)
    latent_heat_flux_W_per_m2 = (
        latent_heat_J_per_kg
        * (aerodynamic_conductance_W_per_m2_K / SPECIFIC_HEAT_AIR_J_PER_KG_K)
        * (specific_humidity_air - specific_humidity_surface)
    )

    # Sublimation/deposition rate (m/hour)
    # Rate = Flux / (L * rho_water) * 3600
    sublimation_deposition_rate_m_per_hour = (
        latent_heat_flux_W_per_m2 / (latent_heat_J_per_kg * RHO_WATER_KG_PER_M3)
    ) * np.float32(3600.0)

    return latent_heat_flux_W_per_m2, sublimation_deposition_rate_m_per_hour


@njit(cache=True, inline="always")
def handle_refreezing(
    snow_enthalpy_J_per_m2: np.float32,
    liquid_water_in_snow_m: np.float64,
    snow_water_equivalent_m: np.float64,
    activate_layer_thickness_m: np.float32,
    max_refreezing_rate_m_per_hour: np.float32 = np.float32(0.01),
) -> tuple[np.float32, np.float64, np.float64, np.float32]:
    """
    Handle refreezing of liquid water in the snow pack based on energy balance.

    This function uses an active thermal layer to approximate refreezing dynamics in
    deep snowpacks, preventing the entire cold content of a deep glacier from
    unrealistically refreezing all surface melt.

    Args:
        snow_enthalpy_J_per_m2: Current frozen-snow enthalpy (J/m²).
        liquid_water_in_snow_m: Current liquid water in snow (m).
        snow_water_equivalent_m: Current snow water equivalent (m).
        activate_layer_thickness_m: Thickness of the active thermal layer for refreezing (m).
        max_refreezing_rate_m_per_hour: Upper bound on refreezing per time step (m/hour) to
            avoid unrealistically freezing all liquid water when the active layer is very thick.

    Returns:
        A tuple of refreezing rate (m/hour), updated snow water equivalent (m),
        updated liquid water (m), and updated snow enthalpy (J/m²).
    """
    snow_temperature_C: np.float32 = get_snow_temperature_from_enthalpy(
        snow_water_equivalent_m=snow_water_equivalent_m,
        snow_enthalpy_J_per_m2=snow_enthalpy_J_per_m2,
    )

    # Determine the depth of the snowpack to consider for refreezing.
    # This prevents the entire cold content of a very deep snowpack (glacier)
    # from refreezing all surface melt, which is more realistic.
    active_swe_for_refreezing_m: np.float32 = min(
        np.float32(snow_water_equivalent_m), activate_layer_thickness_m
    )

    if snow_water_equivalent_m > 0.0:
        active_fraction: np.float32 = np.float32(
            np.float64(active_swe_for_refreezing_m) / snow_water_equivalent_m
        )
    else:
        active_fraction = np.float32(0.0)

    # Cold content is stored directly in the negative enthalpy reservoir.
    cold_content_J_per_m2: np.float32 = max(
        np.float32(0.0), -snow_enthalpy_J_per_m2 * active_fraction
    )

    # Potential refreezing based on cold content (m/hour)
    # The latent heat is huge, so we should be very careful around zero logic here,
    # but the division has a constant denominator > 0.
    refreezing_denominator = LATENT_HEAT_FUSION_J_PER_KG * RHO_WATER_KG_PER_M3
    if refreezing_denominator > np.float32(0.0):
        potential_refreezing_m_per_hour = cold_content_J_per_m2 / refreezing_denominator
    else:
        potential_refreezing_m_per_hour = np.float32(0.0)

    potential_refreezing_m_per_hour = (
        potential_refreezing_m_per_hour
        if snow_temperature_C < np.float32(0.0)
        else np.float32(0.0)
    )

    # Liquid water available for refreezing is the stored liquid water.
    # To avoid unrealistically freezing "all" water when the active layer is
    # very thick (large cold content), we cap the refreezing rate per time step.
    refreezing_capacity_m_per_hour = min(
        np.float32(liquid_water_in_snow_m), max_refreezing_rate_m_per_hour
    )
    actual_refreezing_m_per_hour = min(
        potential_refreezing_m_per_hour, refreezing_capacity_m_per_hour
    )
    updated_snow_water_equivalent_m = (
        snow_water_equivalent_m + actual_refreezing_m_per_hour
    )

    updated_liquid_water_m: np.float64 = (
        liquid_water_in_snow_m - actual_refreezing_m_per_hour
    )
    updated_snow_enthalpy_J_per_m2: np.float32 = snow_enthalpy_J_per_m2 + (
        actual_refreezing_m_per_hour * RHO_WATER_KG_PER_M3 * LATENT_HEAT_FUSION_J_PER_KG
    )

    return (
        actual_refreezing_m_per_hour,
        updated_snow_water_equivalent_m,
        updated_liquid_water_m,
        updated_snow_enthalpy_J_per_m2,
    )


@njit(cache=True, inline="always")
def calculate_runoff(
    liquid_water_in_snow_m: np.float64,
    snow_water_equivalent_m: np.float64,
    activate_layer_thickness_m: np.float32,
) -> tuple[np.float32, np.float64]:
    """
    Calculate runoff from the snow pack based on water holding capacity.

    Args:
        liquid_water_in_snow_m: Current liquid water in snow (m).
        snow_water_equivalent_m: Current snow water equivalent (m).
        activate_layer_thickness_m: Thickness of the active layer for water holding capacity (m).

    Returns:
        A tuple of runoff rate (m/hour) and updated liquid water (m).
    """
    # water holding capacity of 13% (Koren et al. 1999)
    # source: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/1999JD900232
    max_water_content_m: np.float64 = min(
        snow_water_equivalent_m, np.float64(activate_layer_thickness_m)
    ) * np.float64(0.13)
    runoff_rate_m_per_hour: np.float64 = max(
        np.float64(0.0), liquid_water_in_snow_m - max_water_content_m
    )
    updated_liquid_water_m: np.float64 = liquid_water_in_snow_m - runoff_rate_m_per_hour
    runoff_rate_m_per_hour: np.float32 = np.float32(runoff_rate_m_per_hour)
    return runoff_rate_m_per_hour, updated_liquid_water_m
