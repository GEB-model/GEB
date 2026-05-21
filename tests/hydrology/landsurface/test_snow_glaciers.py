"""Tests for snow model functions in GEB."""

import math

import matplotlib.pyplot as plt
import numpy as np

from geb.hydrology.landsurface.constants import (
    KELVIN_OFFSET,
    LATENT_HEAT_FUSION_J_PER_KG,
    RHO_WATER_KG_PER_M3,
    SNOW_EMISSIVITY,
    STEFAN_BOLTZMANN_W_PER_M2_K4,
)
from geb.hydrology.landsurface.energy import (
    calculate_sensible_heat_flux,
    calculate_snow_net_radiation_flux,
)
from geb.hydrology.landsurface.snow_glaciers import (
    FRESH_SNOW_DENSITY_KG_PER_M3,
    MAX_TOP_LAYER_SWE_M,
    add_snowfall_to_top_layer,
    apply_precipitation_compaction_and_top_layer_transfer,
    calculate_latent_heat_flux_and_sublimation,
    calculate_runoff,
    calculate_snow_metamorphism_compaction_rate_per_s,
    calculate_snow_overburden_compaction_rate_per_s,
    compact_snow_density,
    discriminate_precipitation,
    get_snow_enthalpy_from_temperature,
    get_snow_temperature_from_enthalpy,
    handle_refreezing,
    melt_snow_from_enthalpy,
    promote_snow_to_top_layer,
    split_snow_enthalpy,
    update_snow_mass_and_phase,
)
from tests.testconfig import output_folder

output_folder_snow = output_folder / "snow_glaciers"
output_folder_snow.mkdir(exist_ok=True)


def _snow_enthalpy(
    snow_water_equivalent_m: np.float64,
    snow_temperature_C: np.float32,
) -> np.float32:
    """Convert a diagnostic snow temperature to snow enthalpy.

    Args:
        snow_water_equivalent_m: Snow water equivalent (m).
        snow_temperature_C: Snow temperature (°C).

    Returns:
        Snow enthalpy (J/m²).
    """
    return get_snow_enthalpy_from_temperature(
        snow_water_equivalent_m=snow_water_equivalent_m,
        snow_temperature_C=snow_temperature_C,
    )


def test_discriminate_precipitation_threshold() -> None:
    """Precipitation at or below freezing should be classified as snowfall."""
    snowfall, rainfall = discriminate_precipitation(np.float32(0.005), np.float32(0.0))

    assert math.isclose(snowfall, 0.005, abs_tol=1e-6)
    assert math.isclose(rainfall, 0.0, abs_tol=1e-6)

    snowfall_rain, rainfall_rain = discriminate_precipitation(
        np.float32(0.005), np.float32(2.0)
    )
    assert math.isclose(snowfall_rain, 0.0, abs_tol=1e-6)
    assert math.isclose(rainfall_rain, 0.005, abs_tol=1e-6)


def test_calculate_snow_metamorphism_compaction_rate_per_s() -> None:
    """Metamorphism rate should decrease with lower temperature and higher density."""
    # Base case: -5C, 100 kg/m3
    rate_cold = calculate_snow_metamorphism_compaction_rate_per_s(
        np.float32(100.0), np.float32(-5.0)
    )
    # Warmer case: -1C, 100 kg/m3 (should be higher)
    rate_warm = calculate_snow_metamorphism_compaction_rate_per_s(
        np.float32(100.0), np.float32(-1.0)
    )
    assert rate_warm > rate_cold

    # Denser case: -5C, 200 kg/m3 (should be lower)
    rate_dense = calculate_snow_metamorphism_compaction_rate_per_s(
        np.float32(200.0), np.float32(-5.0)
    )
    assert rate_dense < rate_cold


def test_calculate_snow_overburden_compaction_rate_per_s() -> None:
    """Overburden rate should increase with pressure and decrease with density/lower temp."""
    # Base case: -5C, 200 kg/m3, 1000 Pa
    rate_base = calculate_snow_overburden_compaction_rate_per_s(
        np.float32(200.0), np.float32(-5.0), np.float32(1000.0)
    )
    # Higher pressure: 2000 Pa (should be higher)
    rate_high_p = calculate_snow_overburden_compaction_rate_per_s(
        np.float32(200.0), np.float32(-5.0), np.float32(2000.0)
    )
    assert rate_high_p > rate_base

    # Denser case: 300 kg/m3 (should be lower due to higher viscosity)
    rate_dense = calculate_snow_overburden_compaction_rate_per_s(
        np.float32(300.0), np.float32(-5.0), np.float32(1000.0)
    )
    assert rate_dense < rate_base


def test_plot_snow_compaction_processes() -> None:
    """Generate plots of metamorphism and overburden compaction rates."""
    densities = np.linspace(50, 450, 50).astype(np.float32)
    temperatures = np.array([-10.0, -5.0, -1.0, 0.0], dtype=np.float32)
    pressures = np.array([500.0, 2000.0, 5000.0], dtype=np.float32)

    # Plot Metamorphism
    plt.figure(figsize=(10, 6))
    for t in temperatures:
        rates = [
            calculate_snow_metamorphism_compaction_rate_per_s(d, t) * 3600
            for d in densities
        ]
        plt.plot(densities, rates, label=f"T = {t}°C")
    plt.xlabel("Snow Density (kg/m³)")
    plt.ylabel("Compaction Rate (1/h)")
    plt.title("Snow Metamorphism Compaction Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_folder_snow / "snow_metamorphism_compaction.png")
    plt.close()

    # Plot Overburden
    plt.figure(figsize=(10, 6))
    for p in pressures:
        for t in [-5.0]:  # Fix temperature to show pressure effect
            rates = [
                calculate_snow_overburden_compaction_rate_per_s(d, np.float32(t), p)
                * 3600
                for d in densities
            ]
            plt.plot(densities, rates, label=f"P = {p} Pa, T = {t}°C")
    plt.xlabel("Snow Density (kg/m³)")
    plt.ylabel("Compaction Rate (1/h)")
    plt.title("Snow Overburden Compaction Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_folder_snow / "snow_overburden_compaction.png")
    plt.close()


def test_snow_enthalpy_temperature_roundtrip() -> None:
    """Snow enthalpy should diagnose back to the original subfreezing temperature."""
    swe_m = np.float64(0.12)
    temperature_C = np.float32(-4.0)

    enthalpy_J_per_m2 = get_snow_enthalpy_from_temperature(swe_m, temperature_C)
    diagnosed_temperature_C = get_snow_temperature_from_enthalpy(
        swe_m, enthalpy_J_per_m2
    )

    assert math.isclose(diagnosed_temperature_C, temperature_C, abs_tol=1e-6)
    assert get_snow_enthalpy_from_temperature(np.float64(0.0), temperature_C) == 0.0
    assert get_snow_temperature_from_enthalpy(np.float64(0.0), enthalpy_J_per_m2) == 0.0


def test_add_snowfall_to_top_layer_adds_mass_and_cold_content() -> None:
    """Snowfall should add frozen mass and matching cold content."""
    initial_swe_m = np.float64(0.04)
    initial_enthalpy_J_per_m2 = _snow_enthalpy(initial_swe_m, np.float32(-2.0))

    updated_swe_m, updated_enthalpy_J_per_m2 = add_snowfall_to_top_layer(
        snow_water_equivalent_m=initial_swe_m,
        snow_enthalpy_J_per_m2=initial_enthalpy_J_per_m2,
        snowfall_m_per_hour=np.float32(0.01),
        air_temperature_C=np.float32(-5.0),
    )

    assert math.isclose(updated_swe_m, 0.05, abs_tol=1e-9)
    expected_added_enthalpy = _snow_enthalpy(np.float64(0.01), np.float32(-5.0))
    assert math.isclose(
        updated_enthalpy_J_per_m2,
        initial_enthalpy_J_per_m2 + expected_added_enthalpy,
        rel_tol=0,
        abs_tol=1e-3,
    )


def test_split_snow_enthalpy_preserves_temperature() -> None:
    """Enthalpy splitting should preserve bulk snow temperature."""
    initial_swe_m = np.float64(0.10)
    initial_enthalpy_J_per_m2 = _snow_enthalpy(initial_swe_m, np.float32(-3.0))

    remaining_enthalpy_J_per_m2, transferred_enthalpy_J_per_m2 = split_snow_enthalpy(
        snow_water_equivalent_m=initial_swe_m,
        snow_enthalpy_J_per_m2=initial_enthalpy_J_per_m2,
        transferred_snow_water_equivalent_m=np.float64(0.025),
    )
    assert math.isclose(
        remaining_enthalpy_J_per_m2 + transferred_enthalpy_J_per_m2,
        initial_enthalpy_J_per_m2,
        rel_tol=1e-7,
    )
    assert math.isclose(
        get_snow_temperature_from_enthalpy(
            np.float64(0.075), remaining_enthalpy_J_per_m2
        ),
        -3.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        get_snow_temperature_from_enthalpy(
            np.float64(0.025), transferred_enthalpy_J_per_m2
        ),
        -3.0,
        abs_tol=1e-6,
    )


def test_promote_snow_to_top_layer_promotes_bottom_layer_upward() -> None:
    """If the top layer is empty, the bottom layer should be promoted upward."""
    promoted = promote_snow_to_top_layer(
        swe_top_m=np.float64(0.0),
        liquid_water_top_m=np.float64(0.001),
        enthalpy_top_J_per_m2=np.float32(0.0),
        density_top_kg_per_m3=FRESH_SNOW_DENSITY_KG_PER_M3,
        swe_bottom_m=np.float64(0.03),
        liquid_water_bottom_m=np.float64(0.002),
        enthalpy_bottom_J_per_m2=_snow_enthalpy(np.float64(0.03), np.float32(-1.0)),
        density_bottom_kg_per_m3=np.float32(250.0),
    )

    assert math.isclose(promoted[0], 0.03, abs_tol=1e-9)
    assert math.isclose(promoted[1], 0.003, abs_tol=1e-9)
    assert promoted[3] == np.float32(250.0)
    assert promoted[4] == np.float64(0.0)


def test_apply_precipitation_compaction_and_top_layer_transfer() -> None:
    """The pre-solver snow update should partition precipitation and move excess top snow down."""
    (
        rainfall_m_per_hour,
        snowfall_m_per_hour,
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
    ) = apply_precipitation_compaction_and_top_layer_transfer(
        pr_kg_per_m2_per_s=np.float32(0.002),
        air_temperature_C=np.float32(-3.0),
        swe_top_m=np.float64(0.049),
        liquid_water_top_m=np.float64(0.0),
        enthalpy_top_J_per_m2=_snow_enthalpy(np.float64(0.049), np.float32(-2.0)),
        density_top_kg_per_m3=FRESH_SNOW_DENSITY_KG_PER_M3,
        swe_bottom_m=np.float64(0.02),
        liquid_water_bottom_m=np.float64(0.0),
        enthalpy_bottom_J_per_m2=_snow_enthalpy(np.float64(0.02), np.float32(-4.0)),
        density_bottom_kg_per_m3=np.float32(300.0),
    )

    assert math.isclose(rainfall_m_per_hour, 0.0, abs_tol=1e-9)
    assert math.isclose(snowfall_m_per_hour, 0.0072, abs_tol=1e-9)
    assert math.isclose(swe_top_m, float(MAX_TOP_LAYER_SWE_M), abs_tol=1e-9)
    assert swe_bottom_m > np.float64(0.02)
    assert liquid_water_top_m == np.float64(0.0)
    assert liquid_water_bottom_m == np.float64(0.0)
    assert density_top_kg_per_m3 > FRESH_SNOW_DENSITY_KG_PER_M3
    assert np.float32(200.0) < density_bottom_kg_per_m3 < np.float32(300.0)
    assert enthalpy_top_J_per_m2 < np.float32(0.0)
    assert enthalpy_bottom_J_per_m2 < np.float32(0.0)


def test_melt_snow_from_enthalpy_is_limited_by_available_snow() -> None:
    """Positive enthalpy should melt snow but never exceed available SWE."""
    enthalpy_for_2_cm_melt = np.float32(
        0.02 * RHO_WATER_KG_PER_M3 * LATENT_HEAT_FUSION_J_PER_KG
    )
    melt_m, updated_swe_m, updated_enthalpy_J_per_m2 = melt_snow_from_enthalpy(
        snow_water_equivalent_m=np.float64(0.05),
        snow_enthalpy_J_per_m2=enthalpy_for_2_cm_melt,
    )

    assert math.isclose(melt_m, 0.02, abs_tol=1e-6)
    assert math.isclose(updated_swe_m, 0.03, abs_tol=1e-9)
    assert updated_enthalpy_J_per_m2 == np.float32(0.0)

    melt_all_m, updated_swe_all_m, _ = melt_snow_from_enthalpy(
        snow_water_equivalent_m=np.float64(0.01),
        snow_enthalpy_J_per_m2=np.float32(
            0.05 * RHO_WATER_KG_PER_M3 * LATENT_HEAT_FUSION_J_PER_KG
        ),
    )
    assert math.isclose(melt_all_m, 0.01, abs_tol=1e-6)
    assert math.isclose(updated_swe_all_m, 0.0, abs_tol=1e-9)


def test_handle_refreezing_uses_cold_content_and_liquid_limits() -> None:
    """Refreezing should be limited by both cold content and available liquid water."""
    snow_enthalpy_J_per_m2 = _snow_enthalpy(np.float64(0.1), np.float32(-5.0))
    (
        refreezing_m_per_hour,
        updated_swe_m,
        updated_liquid_m,
        updated_enthalpy_J_per_m2,
    ) = handle_refreezing(
        snow_enthalpy_J_per_m2=snow_enthalpy_J_per_m2,
        liquid_water_in_snow_m=np.float64(0.002),
        snow_water_equivalent_m=np.float64(0.1),
        activate_layer_thickness_m=np.float32(0.2),
    )

    assert refreezing_m_per_hour > np.float32(0.0)
    assert updated_swe_m > np.float64(0.1)
    assert updated_liquid_m < np.float64(0.002)
    assert updated_enthalpy_J_per_m2 > snow_enthalpy_J_per_m2


def test_calculate_runoff_uses_active_layer_capacity() -> None:
    """Runoff should only occur once liquid water exceeds the active-layer capacity."""
    # Water holding capacity is 0.13 * min(SWE, activate_layer_thickness)
    # Here min(0.1, 0.2) = 0.1, so capacity = 0.13 * 0.1 = 0.013
    # Liquid water = 0.02, so runoff = 0.02 - 0.013 = 0.007
    runoff_m_per_hour, updated_liquid_m = calculate_runoff(
        liquid_water_in_snow_m=np.float64(0.02),
        snow_water_equivalent_m=np.float64(0.1),
        activate_layer_thickness_m=np.float32(0.2),
    )

    assert math.isclose(runoff_m_per_hour, 0.007, abs_tol=1e-6)
    assert math.isclose(updated_liquid_m, 0.013, abs_tol=1e-6)


def test_calculate_latent_heat_flux_and_sublimation_signs() -> None:
    """Latent flux should change sign between dry sublimating and moist condensing air."""
    dry_latent_W_per_m2, dry_mass_flux_m_per_hour = (
        calculate_latent_heat_flux_and_sublimation(
            air_temperature_C=np.float32(-5.0),
            snow_surface_temperature_C=np.float32(-5.0),
            vapor_pressure_air_Pa=np.float32(200.0),
            air_pressure_Pa=np.float32(90000.0),
            wind_10m_m_per_s=np.float32(4.0),
        )
    )
    moist_latent_W_per_m2, moist_mass_flux_m_per_hour = (
        calculate_latent_heat_flux_and_sublimation(
            air_temperature_C=np.float32(-5.0),
            snow_surface_temperature_C=np.float32(-5.0),
            vapor_pressure_air_Pa=np.float32(450.0),
            air_pressure_Pa=np.float32(90000.0),
            wind_10m_m_per_s=np.float32(4.0),
        )
    )

    assert dry_latent_W_per_m2 < np.float32(0.0)
    assert dry_mass_flux_m_per_hour < np.float32(0.0)
    assert moist_latent_W_per_m2 > np.float32(0.0)
    assert moist_mass_flux_m_per_hour > np.float32(0.0)


def test_update_snow_mass_and_phase_conserves_mass_without_turbulence() -> None:
    """Finalization should conserve water mass when turbulent exchange is turned off."""
    initial_frozen_m = np.float64(0.05)
    rainfall_m_per_hour = np.float32(0.01)
    melt_enthalpy_J_per_m2 = np.float32(
        0.02 * RHO_WATER_KG_PER_M3 * LATENT_HEAT_FUSION_J_PER_KG
    )

    (
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        _density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        _density_bottom_kg_per_m3,
        snow_melt_m_per_hour,
        melt_runoff_m_per_hour,
        rainfall_runoff_m_per_hour,
        sublimation_m_per_hour,
        actual_refreezing_m_per_hour,
    ) = update_snow_mass_and_phase(
        rainfall_m_per_hour=rainfall_m_per_hour,
        swe_top_m=initial_frozen_m,
        liquid_water_top_m=np.float64(0.0),
        enthalpy_top_J_per_m2=melt_enthalpy_J_per_m2,
        density_top_kg_per_m3=compact_snow_density(
            FRESH_SNOW_DENSITY_KG_PER_M3,
            np.float32(0.0),
            np.float32(0.0),
        ),
        swe_bottom_m=np.float64(0.0),
        liquid_water_bottom_m=np.float64(0.0),
        enthalpy_bottom_J_per_m2=np.float32(0.0),
        density_bottom_kg_per_m3=FRESH_SNOW_DENSITY_KG_PER_M3,
        air_temperature_C=np.float32(0.0),
        vapor_pressure_air_Pa=np.float32(611.15),
        air_pressure_Pa=np.float32(101325.0),
        wind_10m_m_per_s=np.float32(0.0),
        activate_layer_thickness_m=np.float32(0.2),
    )

    final_total_water_m = (
        swe_top_m + liquid_water_top_m + swe_bottom_m + liquid_water_bottom_m
    )
    total_runoff_m = melt_runoff_m_per_hour
    initial_total_water_m = initial_frozen_m + np.float64(rainfall_m_per_hour)

    assert math.isclose(snow_melt_m_per_hour, 0.02, abs_tol=1e-6)
    assert actual_refreezing_m_per_hour == np.float32(0.0)
    assert sublimation_m_per_hour == np.float32(0.0)
    assert enthalpy_top_J_per_m2 == np.float32(0.0)
    assert enthalpy_bottom_J_per_m2 == np.float32(0.0)
    # The water balance: Initial SWE + Rain = Final (SWE + LW) + Runoff
    # In this case: 0.05 + 0.01 = 0.06
    # Final state: top_swe (0.05 - 0.02) = 0.03
    # LW content: rain (0.01) + melt (0.02) = 0.03
    # Max LW: 0.03 * 0.1 = 0.003
    # Runoff: 0.03 - 0.003 = 0.027
    # Final SWE + LW + Runoff = 0.03 + 0.003 + 0.027 = 0.06
    assert math.isclose(
        initial_total_water_m,
        final_total_water_m + np.float64(total_runoff_m),
        abs_tol=1e-6,
    )


def _initial_snow_state(
    swe_m: float,
    temperature_C: float,
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
    """Set up initial two-layer snow state for scenario tests.

    The top layer is filled to at most MAX_TOP_LAYER_SWE_M; any surplus goes
    to the bottom layer.  Both layers share the same initial temperature.

    Args:
        swe_m: Total initial snow water equivalent (m).
        temperature_C: Initial bulk snow temperature (°C).

    Returns:
        Tuple of (swe_top, lw_top, enthalpy_top, density_top,
                  swe_bottom, lw_bottom, enthalpy_bottom, density_bottom).
    """
    swe_top: np.float64 = np.float64(min(swe_m, float(MAX_TOP_LAYER_SWE_M)))
    swe_bottom: np.float64 = np.float64(max(0.0, swe_m - float(MAX_TOP_LAYER_SWE_M)))
    enthalpy_top: np.float32 = get_snow_enthalpy_from_temperature(
        swe_top, np.float32(temperature_C)
    )
    enthalpy_bottom: np.float32 = get_snow_enthalpy_from_temperature(
        swe_bottom, np.float32(temperature_C)
    )
    return (
        swe_top,
        np.float64(0.0),
        enthalpy_top,
        FRESH_SNOW_DENSITY_KG_PER_M3,
        swe_bottom,
        np.float64(0.0),
        enthalpy_bottom,
        FRESH_SNOW_DENSITY_KG_PER_M3,
    )


def _run_snow_scenario_step(
    pr_kg_per_m2_per_s: np.float32,
    air_temperature_C: np.float32,
    shortwave_radiation_W_per_m2: np.float32,
    longwave_radiation_W_per_m2: np.float32,
    vapor_pressure_air_Pa: np.float32,
    air_pressure_Pa: np.float32,
    wind_10m_m_per_s: np.float32,
    swe_top_m: np.float64,
    liquid_water_top_m: np.float64,
    enthalpy_top_J_per_m2: np.float32,
    density_top_kg_per_m3: np.float32,
    swe_bottom_m: np.float64,
    liquid_water_bottom_m: np.float64,
    enthalpy_bottom_J_per_m2: np.float32,
    density_bottom_kg_per_m3: np.float32,
    water_holding_capacity_fraction: np.float32 = np.float32(0.1),
    activate_layer_thickness_m: np.float32 = np.float32(0.2),
) -> tuple:
    """Run one 1-hour timestep of the two-layer snow model for scenario testing.

    Replicates the sequence used in the full land-surface model:
    (1) precipitation partitioning, compaction, and layer transfer;
    (2) explicit surface energy-balance update of the top-layer enthalpy;
    (3) mass/phase update (sublimation, melt, refreezing, runoff).

    The energy step uses the simplified explicit scheme
    (enthalpy += (net_radiation + sensible) * 3600 s), which is sufficient
    for multi-hour scenario diagnostics without the coupled soil solver.

    Args:
        pr_kg_per_m2_per_s: Precipitation rate (kg/m²/s).
        air_temperature_C: Air temperature (°C).
        shortwave_radiation_W_per_m2: Incoming shortwave radiation (W/m²).
        longwave_radiation_W_per_m2: Incoming longwave radiation (W/m²).
        vapor_pressure_air_Pa: Atmospheric vapour pressure (Pa).
        air_pressure_Pa: Air pressure (Pa).
        wind_10m_m_per_s: Wind speed at 10 m (m/s).
        swe_top_m: Top-layer SWE (m).
        liquid_water_top_m: Top-layer liquid water (m).
        enthalpy_top_J_per_m2: Top-layer enthalpy (J/m²).
        density_top_kg_per_m3: Top-layer snow density (kg/m³).
        swe_bottom_m: Bottom-layer SWE (m).
        liquid_water_bottom_m: Bottom-layer liquid water (m).
        enthalpy_bottom_J_per_m2: Bottom-layer enthalpy (J/m²).
        density_bottom_kg_per_m3: Bottom-layer snow density (kg/m³).
        water_holding_capacity_fraction: Liquid holding capacity as fraction of SWE (-).
        activate_layer_thickness_m: Active thermal layer thickness (m).

    Returns:
        Tuple of updated state and diagnostic fluxes:
        (swe_top, lw_top, enthalpy_top, density_top,
         swe_bottom, lw_bottom, enthalpy_bottom, density_bottom,
         snow_melt_m_per_hour, runoff_m_per_hour,
         sublimation_m_per_hour, refreezing_m_per_hour,
         net_radiation_W_per_m2, sensible_heat_W_per_m2,
         latent_heat_W_per_m2, absorbed_sw_W_per_m2, upward_lw_W_per_m2).
    """
    # Step 1: Precipitation, compaction, and layer transfer.
    (
        rainfall_m_per_hour,
        snowfall_m_per_hour,
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
    ) = apply_precipitation_compaction_and_top_layer_transfer(
        pr_kg_per_m2_per_s=pr_kg_per_m2_per_s,
        air_temperature_C=air_temperature_C,
        swe_top_m=swe_top_m,
        liquid_water_top_m=liquid_water_top_m,
        enthalpy_top_J_per_m2=enthalpy_top_J_per_m2,
        density_top_kg_per_m3=density_top_kg_per_m3,
        swe_bottom_m=swe_bottom_m,
        liquid_water_bottom_m=liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2=enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3=density_bottom_kg_per_m3,
    )

    # Step 2: Surface energy balance — update top-layer enthalpy.
    total_swe_m = float(swe_top_m + swe_bottom_m)
    if total_swe_m > 0.0:
        snow_surface_temp_C = get_snow_temperature_from_enthalpy(
            swe_top_m, enthalpy_top_J_per_m2
        )

        net_radiation_W_per_m2, _ = calculate_snow_net_radiation_flux(
            shortwave_radiation_W_per_m2=shortwave_radiation_W_per_m2,
            longwave_radiation_W_per_m2=longwave_radiation_W_per_m2,
            snow_temperature_C=snow_surface_temp_C,
            total_snow_water_equivalent_m=np.float32(total_swe_m),
        )
        sensible_heat_W_per_m2, _ = calculate_sensible_heat_flux(
            soil_temperature_C=snow_surface_temp_C,
            air_temperature_K=np.float32(air_temperature_C + float(KELVIN_OFFSET)),
            wind_speed_10m_m_per_s=wind_10m_m_per_s,
            surface_pressure_pa=air_pressure_Pa,
        )
        latent_heat_W_per_m2, _ = calculate_latent_heat_flux_and_sublimation(
            air_temperature_C=air_temperature_C,
            snow_surface_temperature_C=snow_surface_temp_C,
            vapor_pressure_air_Pa=vapor_pressure_air_Pa,
            air_pressure_Pa=air_pressure_Pa,
            wind_10m_m_per_s=wind_10m_m_per_s,
        )

        # Diagnostic decomposition of net radiation into SW and LW.
        snow_temp_K = np.float32(snow_surface_temp_C + float(KELVIN_OFFSET))
        upward_lw_W_per_m2 = (
            SNOW_EMISSIVITY * STEFAN_BOLTZMANN_W_PER_M2_K4 * snow_temp_K**4
        )
        # Absorbed SW = total net radiation minus the LW component.
        absorbed_sw_W_per_m2 = (
            net_radiation_W_per_m2 - longwave_radiation_W_per_m2 + upward_lw_W_per_m2
        )

        # Advance top-layer enthalpy with net radiative and sensible heat.
        # Latent heat (sublimation/deposition) is handled inside update_snow_mass_and_phase.
        enthalpy_top_J_per_m2 += (
            net_radiation_W_per_m2 + sensible_heat_W_per_m2
        ) * np.float32(3600.0)
    else:
        net_radiation_W_per_m2 = np.float32(0.0)
        sensible_heat_W_per_m2 = np.float32(0.0)
        latent_heat_W_per_m2 = np.float32(0.0)
        absorbed_sw_W_per_m2 = np.float32(0.0)
        upward_lw_W_per_m2 = np.float32(0.0)

    # Step 3: Mass and phase update (sublimation, melt, refreezing, runoff).
    (
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
        rainfall_runoff_m_per_hour,
        sublimation_m_per_hour,
        refreezing_m_per_hour,
    ) = update_snow_mass_and_phase(
        rainfall_m_per_hour=rainfall_m_per_hour,
        swe_top_m=swe_top_m,
        liquid_water_top_m=liquid_water_top_m,
        enthalpy_top_J_per_m2=enthalpy_top_J_per_m2,
        density_top_kg_per_m3=density_top_kg_per_m3,
        swe_bottom_m=swe_bottom_m,
        liquid_water_bottom_m=liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2=enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3=density_bottom_kg_per_m3,
        air_temperature_C=air_temperature_C,
        vapor_pressure_air_Pa=vapor_pressure_air_Pa,
        air_pressure_Pa=air_pressure_Pa,
        wind_10m_m_per_s=wind_10m_m_per_s,
        activate_layer_thickness_m=activate_layer_thickness_m,
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
        rainfall_runoff_m_per_hour,
        sublimation_m_per_hour,
        refreezing_m_per_hour,
        net_radiation_W_per_m2,
        sensible_heat_W_per_m2,
        latent_heat_W_per_m2,
        absorbed_sw_W_per_m2,
        upward_lw_W_per_m2,
    )


def _run_scenario(
    n_hours: int,
    precip_kg_per_m2_per_s_series: np.ndarray,
    air_temp_series_C: np.ndarray,
    sw_rad_series_W_per_m2: np.ndarray,
    lw_rad_series_W_per_m2: np.ndarray,
    vapor_pressure_series_Pa: np.ndarray,
    air_pressure_Pa: np.float32,
    wind_10m_m_per_s: np.float32,
    initial_swe_m: float,
    initial_snow_temperature_C: float,
    activate_layer_thickness_m: np.float32 = np.float32(0.2),
) -> dict[str, np.ndarray]:
    """Run a multi-hour snow model scenario and return logged diagnostics.

    Args:
        n_hours: Number of hours to simulate.
        precip_kg_per_m2_per_s_series: Precipitation rate time series (kg/m²/s).
        air_temp_series_C: Air temperature time series (°C).
        sw_rad_series_W_per_m2: Incoming shortwave radiation time series (W/m²).
        lw_rad_series_W_per_m2: Incoming longwave radiation time series (W/m²).
        vapor_pressure_series_Pa: Vapour pressure time series (Pa).
        air_pressure_Pa: Constant air pressure (Pa).
        wind_10m_m_per_s: Constant wind speed at 10 m (m/s).
        initial_swe_m: Initial total snow water equivalent (m).
        initial_snow_temperature_C: Initial snow temperature (°C).
        activate_layer_thickness_m: Active thermal layer thickness (m).

    Returns:
        Dictionary of logged arrays (length n_hours) for: swe_log, lw_log,
        snow_temp_log, melt_log, runoff_log, precipitation_log, rain_log, sublimation_log, refreezing_log,
        net_radiation_log, sensible_log, latent_log, absorbed_sw_log,
        upward_lw_log.
    """
    swe_log = np.zeros(n_hours, dtype=np.float64)
    lw_log = np.zeros(n_hours, dtype=np.float64)
    snow_temp_log = np.zeros(n_hours, dtype=np.float32)
    melt_log = np.zeros(n_hours, dtype=np.float32)
    runoff_log = np.zeros(n_hours, dtype=np.float32)
    precipitation_log = np.zeros(n_hours, dtype=np.float32)
    rain_log = np.zeros(n_hours, dtype=np.float32)
    sublimation_log = np.zeros(n_hours, dtype=np.float32)
    refreezing_log = np.zeros(n_hours, dtype=np.float32)
    net_radiation_log = np.zeros(n_hours, dtype=np.float32)
    sensible_log = np.zeros(n_hours, dtype=np.float32)
    latent_log = np.zeros(n_hours, dtype=np.float32)
    absorbed_sw_log = np.zeros(n_hours, dtype=np.float32)
    upward_lw_log = np.zeros(n_hours, dtype=np.float32)

    (
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
    ) = _initial_snow_state(initial_swe_m, initial_snow_temperature_C)

    for i in range(n_hours):
        (
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
            rainfall_runoff_m_per_hour,
            sublimation_m_per_hour,
            refreezing_m_per_hour,
            net_radiation_W_per_m2,
            sensible_heat_W_per_m2,
            latent_heat_W_per_m2,
            absorbed_sw_W_per_m2,
            upward_lw_W_per_m2,
        ) = _run_snow_scenario_step(
            pr_kg_per_m2_per_s=np.float32(precip_kg_per_m2_per_s_series[i]),
            air_temperature_C=np.float32(air_temp_series_C[i]),
            shortwave_radiation_W_per_m2=np.float32(sw_rad_series_W_per_m2[i]),
            longwave_radiation_W_per_m2=np.float32(lw_rad_series_W_per_m2[i]),
            vapor_pressure_air_Pa=np.float32(vapor_pressure_series_Pa[i]),
            air_pressure_Pa=air_pressure_Pa,
            wind_10m_m_per_s=wind_10m_m_per_s,
            swe_top_m=swe_top_m,
            liquid_water_top_m=liquid_water_top_m,
            enthalpy_top_J_per_m2=enthalpy_top_J_per_m2,
            density_top_kg_per_m3=density_top_kg_per_m3,
            swe_bottom_m=swe_bottom_m,
            liquid_water_bottom_m=liquid_water_bottom_m,
            enthalpy_bottom_J_per_m2=enthalpy_bottom_J_per_m2,
            density_bottom_kg_per_m3=density_bottom_kg_per_m3,
            activate_layer_thickness_m=activate_layer_thickness_m,
        )

        swe_log[i] = float(swe_top_m + swe_bottom_m)
        lw_log[i] = float(liquid_water_top_m + liquid_water_bottom_m)
        snow_temp_log[i] = get_snow_temperature_from_enthalpy(
            swe_top_m, enthalpy_top_J_per_m2
        )
        melt_log[i] = float(snow_melt_m_per_hour)
        runoff_log[i] = float(melt_runoff_m_per_hour)
        precipitation_log[i] = float(
            precip_kg_per_m2_per_s_series[i] * 3600.0 / RHO_WATER_KG_PER_M3
        )
        sublimation_log[i] = float(sublimation_m_per_hour)
        refreezing_log[i] = float(refreezing_m_per_hour)
        net_radiation_log[i] = float(net_radiation_W_per_m2)
        sensible_log[i] = float(sensible_heat_W_per_m2)
        latent_log[i] = float(latent_heat_W_per_m2)
        absorbed_sw_log[i] = float(absorbed_sw_W_per_m2)
        upward_lw_log[i] = float(upward_lw_W_per_m2)

    return {
        "swe_log": swe_log,
        "lw_log": lw_log,
        "snow_temp_log": snow_temp_log,
        "melt_log": melt_log,
        "runoff_log": runoff_log,
        "precipitation_log": precipitation_log,
        "sublimation_log": sublimation_log,
        "refreezing_log": refreezing_log,
        "net_radiation_log": net_radiation_log,
        "sensible_log": sensible_log,
        "latent_log": latent_log,
        "absorbed_sw_log": absorbed_sw_log,
        "upward_lw_log": upward_lw_log,
        "initial_total_water_m": initial_swe_m,
    }


def _plot_scenario_results(
    scenario_name: str,
    timesteps: np.ndarray,
    results: dict[str, np.ndarray],
    precip_mm_per_hr: np.ndarray,
    air_temp_C: np.ndarray,
    sw_rad_W_per_m2: np.ndarray,
    lw_rad_W_per_m2: np.ndarray,
) -> None:
    """Plot and save a five-panel diagnostic figure for a snow scenario.

    Args:
        scenario_name: Identifier used in the figure title and filename.
        timesteps: Array of integer hour indices.
        results: Dictionary returned by _run_scenario.
        precip_mm_per_hr: Precipitation time series (mm/hr, for the bar chart).
        air_temp_C: Air temperature time series (°C).
        sw_rad_W_per_m2: Incoming shortwave radiation time series (W/m²).
        lw_rad_W_per_m2: Incoming longwave radiation time series (W/m²).
    """
    fig, axs = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    fig.suptitle(f"Snow Model Scenario: {scenario_name}", fontsize=16)

    # Panel 1: Mass balance.
    axs[0].set_title("Snowpack Mass Balance")
    axs[0].plot(timesteps, results["swe_log"], label="SWE (m)", color="blue")
    axs[0].plot(
        timesteps,
        results["lw_log"],
        label="Liquid Water (m)",
        color="cyan",
        linestyle="--",
    )
    axs[0].set_ylabel("Water Equivalent (m)")
    axs[0].legend(loc="upper left")
    axs[0].grid(True)
    ax_precip = axs[0].twinx()
    ax_precip.bar(
        timesteps,
        precip_mm_per_hr,
        label="Precip (mm/hr)",
        color="gray",
        alpha=0.5,
        width=1.0,
    )
    ax_precip.set_ylabel("Precipitation (mm/hr)")
    ax_precip.legend(loc="upper right")

    # Panel 2: Water fluxes.
    axs[1].set_title("Water Fluxes")
    axs[1].plot(
        timesteps,
        results["runoff_log"] * 1000,
        label="Runoff (mm/hr)",
        color="green",
    )
    axs[1].plot(
        timesteps,
        results["sublimation_log"] * 1000,
        label="Sublimation/Deposition (mm/hr)",
        color="purple",
        linestyle=":",
    )
    axs[1].plot(
        timesteps,
        results["refreezing_log"] * 1000,
        label="Refreezing (mm/hr)",
        color="red",
        linestyle="-.",
    )
    axs[1].plot(
        timesteps,
        results["melt_log"] * 1000,
        label="Melt (mm/hr)",
        color="orange",
        linestyle="--",
    )
    axs[1].set_ylabel("Water Flux (mm/hr)")
    axs[1].legend()
    axs[1].grid(True)

    # Panel 3: Temperature.
    axs[2].set_title("Temperatures")
    axs[2].plot(timesteps, air_temp_C, label="Air Temp (°C)", color="red")
    axs[2].plot(
        timesteps,
        results["snow_temp_log"],
        label="Snow Temp (°C)",
        color="blue",
        linestyle="--",
    )
    axs[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axs[2].set_ylabel("Temperature (°C)")
    axs[2].legend()
    axs[2].grid(True)

    # Panel 4: Net radiation and sensible/latent heat.
    axs[3].set_title("Energy Fluxes")
    net_lw_log = lw_rad_W_per_m2 - results["upward_lw_log"]
    axs[3].plot(
        timesteps,
        results["absorbed_sw_log"],
        label="Absorbed SW (W/m²)",
        color="orange",
    )
    axs[3].plot(timesteps, net_lw_log, label="Net LW (W/m²)", color="magenta")
    axs[3].plot(
        timesteps,
        results["sensible_log"],
        label="Sensible Heat (W/m²)",
        color="red",
        linestyle="--",
    )
    axs[3].plot(
        timesteps,
        results["latent_log"],
        label="Latent Heat (W/m²)",
        color="cyan",
        linestyle=":",
    )
    net_energy = (
        results["absorbed_sw_log"]
        + net_lw_log
        + results["sensible_log"]
        + results["latent_log"]
    )
    axs[3].plot(
        timesteps,
        net_energy,
        label="Net Energy (W/m²)",
        color="black",
        linewidth=2,
    )
    axs[3].axhline(0, color="black", linestyle="-", linewidth=0.8)
    axs[3].set_ylabel("Energy Flux (W/m²)")
    axs[3].legend()
    axs[3].grid(True)

    # Panel 5: Radiation components.
    axs[4].set_title("Radiation Components")
    axs[4].plot(timesteps, sw_rad_W_per_m2, label="Incoming SW (W/m²)", color="orange")
    axs[4].plot(
        timesteps,
        lw_rad_W_per_m2,
        label="Incoming LW (W/m²)",
        color="magenta",
        linestyle="--",
    )
    axs[4].plot(
        timesteps,
        -results["upward_lw_log"],
        label="Outgoing LW (W/m²)",
        color="purple",
        linestyle=":",
    )
    axs[4].axhline(0, color="black", linestyle="-", linewidth=0.8)
    axs[4].set_ylabel("Radiation (W/m²)")
    axs[4].legend()
    axs[4].grid(True)

    axs[-1].set_xlabel("Time (hours)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = (
        output_folder_snow / f"scenario_{scenario_name.replace(' ', '_').lower()}.png"
    )
    plt.savefig(plot_path)
    plt.close(fig)
    assert plot_path.exists()


def _vapor_pressure_Pa(dewpoint_C: np.ndarray) -> np.ndarray:
    """Compute saturation vapour pressure from dewpoint (Buck's equation, Pa).

    Args:
        dewpoint_C: Dewpoint temperature time series or scalar (°C).

    Returns:
        Vapour pressure (Pa).
    """
    return 610.94 * np.exp(17.625 * dewpoint_C / (243.04 + dewpoint_C))


def _verify_scenario_mass_balance(results: dict[str, np.ndarray]) -> None:
    """Verify mass balance and check for NaN values in scenario results.

    The mass balance equation is:
    DSWE + DLW = Precipitation - Runoff + Sublimation/Deposition

    Args:
        results: Dictionary of scenario results.

    Raises:
        ValueError: If mass balance is not closed or NaNs are found.
    """
    # 1. Check for NaN values in all logged arrays
    for key, arr in results.items():
        if isinstance(arr, np.ndarray):
            if np.any(np.isnan(arr)):
                raise ValueError(f"NaN values found in {key}")

    # 2. Verify mass balance
    # Total water at start (meters)
    initial_total_water_m = results["initial_total_water_m"]

    # Total water at end (meters)
    final_total_water_m = results["swe_log"][-1] + results["lw_log"][-1]

    # Cumulative fluxes (meters)
    total_precipitation_m = np.sum(results["precipitation_log"])
    total_runoff_m = np.sum(results["runoff_log"])
    total_sublimation_m = np.sum(results["sublimation_log"])

    # Change in storage
    delta_storage_m = final_total_water_m - initial_total_water_m

    # Net flux (Precipitation - Runoff + Sublimation)
    # Note: sublimation_log is negative for loss, positive for deposition.
    net_flux_m = total_precipitation_m - total_runoff_m + total_sublimation_m

    # We use a strict tolerance for scenario tests.
    scenario_abs_tol = 1e-6

    # Check closure with a small absolute tolerance (meters)
    if not math.isclose(delta_storage_m, net_flux_m, abs_tol=scenario_abs_tol):
        raise ValueError(
            f"Mass balance not closed. Delta storage: {delta_storage_m}, "
            f"Net flux: {net_flux_m}, Difference: {delta_storage_m - net_flux_m}"
        )


def test_snowpack_development_scenario() -> None:
    """Snowpack should accumulate, partially melt, and generate runoff over 72 h."""
    n_hours = 72
    timesteps = np.arange(n_hours)

    # Diurnal temperature cycle that crosses zero — snow events at hours 10-14 and 40-44.
    air_temp_C = (5 * np.sin(2 * np.pi * (timesteps - 6) / 24) - 3).astype(np.float32)
    air_temp_C[40:45] += 3

    pr_series = np.zeros(n_hours, dtype=np.float32)
    pr_series[10:15] = np.float32(0.002 / 3.6)  # Convert mm/hr → kg/m²/s
    pr_series[40:45] = np.float32(0.003 / 3.6)

    sw_rad = np.maximum(0, 400 * np.sin(2 * np.pi * (timesteps - 6) / 24)).astype(
        np.float32
    )
    lw_rad = (280 + 40 * np.sin(2 * np.pi * (timesteps - 6) / 24 + np.pi / 2)).astype(
        np.float32
    )
    vapor_pressure = _vapor_pressure_Pa(air_temp_C - 2).astype(np.float32)

    results = _run_scenario(
        n_hours=n_hours,
        precip_kg_per_m2_per_s_series=pr_series,
        air_temp_series_C=air_temp_C,
        sw_rad_series_W_per_m2=sw_rad,
        lw_rad_series_W_per_m2=lw_rad,
        vapor_pressure_series_Pa=vapor_pressure,
        air_pressure_Pa=np.float32(95000.0),
        wind_10m_m_per_s=np.float32(2.0),
        initial_swe_m=0.02,
        initial_snow_temperature_C=-2.0,
    )

    assert np.any(results["melt_log"] > 0), "Melt should occur during warm periods."
    assert np.any(results["runoff_log"] > 0), "Runoff should be generated."
    assert results["swe_log"][-1] >= 0.0

    _verify_scenario_mass_balance(results)

    _plot_scenario_results(
        scenario_name="snowpack_development",
        timesteps=timesteps,
        results=results,
        precip_mm_per_hr=pr_series * 3600.0,
        air_temp_C=air_temp_C,
        sw_rad_W_per_m2=sw_rad,
        lw_rad_W_per_m2=lw_rad,
    )


def test_extreme_sublimation_cooling_regression() -> None:
    """Test case to reproduce extreme negative temperatures due to sublimation.

    Regression test for a bug where sublimation on very thin snow layers
    driven by high wind speeds caused unphysical cooling below the air temperature floor.
    """
    # Initial conditions from user case 1
    swe_top_m = np.float64(9.492729532212252e-06)
    liquid_water_top_m = np.float64(0.0)
    # Very cold temperature to start with
    enthalpy_top_J_per_m2 = get_snow_enthalpy_from_temperature(
        swe_top_m, np.float32(-30.0)
    )
    density_top_kg_per_m3 = np.float32(200.0)

    swe_bottom_m = np.float64(0.0)
    liquid_water_bottom_m = np.float64(0.0)
    enthalpy_bottom_J_per_m2 = np.float32(0.0)
    density_bottom_kg_per_m3 = np.float32(300.0)

    # Atmospheric conditions: Extremely cold air and high wind
    air_temperature_C = np.float32(-50.0)
    vapor_pressure_air_Pa = np.float32(0.1)
    air_pressure_Pa = np.float32(80000.0)
    wind_10m_m_per_s = np.float32(30.0)

    rainfall_m_per_hour = np.float32(0.0)
    activate_layer_thickness_m = np.float32(0.1)

    T_init = get_snow_temperature_from_enthalpy(swe_top_m, enthalpy_top_J_per_m2)

    result = update_snow_mass_and_phase(
        rainfall_m_per_hour,
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
        air_temperature_C,
        vapor_pressure_air_Pa,
        air_pressure_Pa,
        wind_10m_m_per_s,
        activate_layer_thickness_m,
    )

    updated_swe_top = result[0]
    updated_enthalpy_top = result[2]

    if updated_swe_top > 1e-14:
        final_temp = get_snow_temperature_from_enthalpy(
            updated_swe_top, updated_enthalpy_top
        )
        # The logic should prevent it from going much below air temp unless it was already colder
        assert final_temp >= min(T_init, air_temperature_C) - 0.1

    # Case 2: Extreme sublimation forcing on thin SWE
    swe_top_m = np.float64(9.817802947509335e-06)
    # Start at -10C
    enthalpy_top_J_per_m2 = get_snow_enthalpy_from_temperature(
        swe_top_m, np.float32(-10.0)
    )
    air_temperature_C = np.float32(-10.0)
    wind_10m_m_per_s = np.float32(50.0)

    result = update_snow_mass_and_phase(
        rainfall_m_per_hour,
        swe_top_m,
        liquid_water_top_m,
        enthalpy_top_J_per_m2,
        density_top_kg_per_m3,
        swe_bottom_m,
        liquid_water_bottom_m,
        enthalpy_bottom_J_per_m2,
        density_bottom_kg_per_m3,
        air_temperature_C,
        vapor_pressure_air_Pa,
        air_pressure_Pa,
        wind_10m_m_per_s,
        activate_layer_thickness_m,
    )

    updated_swe_top = result[0]
    updated_enthalpy_top = result[2]

    if updated_swe_top > 1e-14:
        final_temp = get_snow_temperature_from_enthalpy(
            updated_swe_top, updated_enthalpy_top
        )
        # Should stay at the air temperature floor
        assert final_temp >= air_temperature_C - 1e-5


def test_snowpack_arctic_scenario() -> None:
    """Arctic cold snowpack should not significantly melt over 72 h."""
    n_hours = 72
    timesteps = np.arange(n_hours)

    air_temp_C = (-15 + 5 * np.sin(2 * np.pi * (timesteps - 6) / 24)).astype(np.float32)
    pr_series = np.zeros(n_hours, dtype=np.float32)
    pr_series[20:25] = np.float32(0.001 / 3.6)
    pr_series[50:55] = np.float32(0.0005 / 3.6)

    sw_rad = np.maximum(0, 100 * np.sin(2 * np.pi * (timesteps - 6) / 24)).astype(
        np.float32
    )
    lw_rad = (200 + 20 * np.sin(2 * np.pi * (timesteps - 6) / 24 + np.pi / 2)).astype(
        np.float32
    )
    vapor_pressure = _vapor_pressure_Pa(air_temp_C - 5).astype(np.float32)

    results = _run_scenario(
        n_hours=n_hours,
        precip_kg_per_m2_per_s_series=pr_series,
        air_temp_series_C=air_temp_C,
        sw_rad_series_W_per_m2=sw_rad,
        lw_rad_series_W_per_m2=lw_rad,
        vapor_pressure_series_Pa=vapor_pressure,
        air_pressure_Pa=np.float32(101000.0),
        wind_10m_m_per_s=np.float32(3.0),
        initial_swe_m=0.1,
        initial_snow_temperature_C=-10.0,
    )

    assert np.sum(results["melt_log"]) < 0.001, "Negligible melt in Arctic conditions."
    assert np.all(results["runoff_log"] == 0.0), (
        "No runoff expected in Arctic conditions."
    )
    # SWE should stay roughly constant or grow slightly from snowfall.
    assert results["swe_log"][-1] >= results["swe_log"][0]

    _verify_scenario_mass_balance(results)

    _plot_scenario_results(
        scenario_name="snowpack_arctic",
        timesteps=timesteps,
        results=results,
        precip_mm_per_hr=pr_series * 3600.0,
        air_temp_C=air_temp_C,
        sw_rad_W_per_m2=sw_rad,
        lw_rad_W_per_m2=lw_rad,
    )


def test_snowpack_high_altitude_scenario() -> None:
    """High-altitude snowpack with strong radiation should lose mass over 72 h."""
    n_hours = 72
    timesteps = np.arange(n_hours)

    air_temp_C = (-2 + 8 * np.sin(2 * np.pi * (timesteps - 6) / 24)).astype(np.float32)
    pr_series = np.zeros(n_hours, dtype=np.float32)
    pr_series[15:20] = np.float32(0.002 / 3.6)
    pr_series[45:50] = np.float32(0.001 / 3.6)

    sw_rad = np.maximum(0, 800 * np.sin(2 * np.pi * (timesteps - 6) / 24)).astype(
        np.float32
    )
    lw_rad = (250 + 30 * np.sin(2 * np.pi * (timesteps - 6) / 24 + np.pi / 2)).astype(
        np.float32
    )
    vapor_pressure = _vapor_pressure_Pa(air_temp_C - 3).astype(np.float32)

    results = _run_scenario(
        n_hours=n_hours,
        precip_kg_per_m2_per_s_series=pr_series,
        air_temp_series_C=air_temp_C,
        sw_rad_series_W_per_m2=sw_rad,
        lw_rad_series_W_per_m2=lw_rad,
        vapor_pressure_series_Pa=vapor_pressure,
        air_pressure_Pa=np.float32(60000.0),
        wind_10m_m_per_s=np.float32(5.0),
        initial_swe_m=0.15,
        initial_snow_temperature_C=-5.0,
    )

    assert np.any(results["melt_log"] > 0), "Melt must occur with high radiation input."
    assert np.any(results["runoff_log"] > 0), "Runoff must be generated."
    assert results["swe_log"][-1] < results["swe_log"][0], (
        "Net SWE loss expected at high altitude."
    )

    _verify_scenario_mass_balance(results)

    _plot_scenario_results(
        scenario_name="snowpack_high_altitude",
        timesteps=timesteps,
        results=results,
        precip_mm_per_hr=pr_series * 3600.0,
        air_temp_C=air_temp_C,
        sw_rad_W_per_m2=sw_rad,
        lw_rad_W_per_m2=lw_rad,
    )


def test_complete_ablation_scenario() -> None:
    """Snowpack should fully ablate under steadily warming conditions over 48 h."""
    n_hours = 48
    timesteps = np.arange(n_hours)

    air_temp_C = np.linspace(2, 10, n_hours, dtype=np.float32)
    pr_series = np.zeros(n_hours, dtype=np.float32)

    sw_rad = np.maximum(0, 600 * np.sin(2 * np.pi * (timesteps - 6) / 24)).astype(
        np.float32
    )
    lw_rad = np.linspace(300, 350, n_hours, dtype=np.float32)
    vapor_pressure = _vapor_pressure_Pa(air_temp_C - 2).astype(np.float32)

    results = _run_scenario(
        n_hours=n_hours,
        precip_kg_per_m2_per_s_series=pr_series,
        air_temp_series_C=air_temp_C,
        sw_rad_series_W_per_m2=sw_rad,
        lw_rad_series_W_per_m2=lw_rad,
        vapor_pressure_series_Pa=vapor_pressure,
        air_pressure_Pa=np.float32(98000.0),
        wind_10m_m_per_s=np.float32(3.0),
        initial_swe_m=0.05,
        initial_snow_temperature_C=-1.0,
    )

    ablation_steps = np.where(results["swe_log"] <= 1e-6)[0]
    assert len(ablation_steps) > 0, "Snowpack must fully ablate."
    # Once ablated, SWE must remain zero.
    assert np.all(results["swe_log"][ablation_steps[0] :] <= 1e-6), (
        "SWE must remain zero after full ablation."
    )

    _verify_scenario_mass_balance(results)

    _plot_scenario_results(
        scenario_name="complete_ablation",
        timesteps=timesteps,
        results=results,
        precip_mm_per_hr=pr_series * 3600.0,
        air_temp_C=air_temp_C,
        sw_rad_W_per_m2=sw_rad,
        lw_rad_W_per_m2=lw_rad,
    )


def test_deposition_scenario() -> None:
    """Mass should increase from deposition under cold, moist, calm conditions."""
    n_hours = 48
    timesteps = np.arange(n_hours)

    air_temp_C = np.full(n_hours, -8.0, dtype=np.float32)
    pr_series = np.zeros(n_hours, dtype=np.float32)
    sw_rad = np.zeros(n_hours, dtype=np.float32)
    lw_rad = np.full(n_hours, 220.0, dtype=np.float32)
    # Dewpoint above air temperature → supersaturated air → deposition.
    vapor_pressure = _vapor_pressure_Pa(air_temp_C + 2).astype(np.float32)

    results = _run_scenario(
        n_hours=n_hours,
        precip_kg_per_m2_per_s_series=pr_series,
        air_temp_series_C=air_temp_C,
        sw_rad_series_W_per_m2=sw_rad,
        lw_rad_series_W_per_m2=lw_rad,
        vapor_pressure_series_Pa=vapor_pressure,
        air_pressure_Pa=np.float32(100000.0),
        wind_10m_m_per_s=np.float32(2.0),
        initial_swe_m=0.02,
        initial_snow_temperature_C=-10.0,
    )

    # Sublimation array: positive values indicate deposition (mass gain).
    assert np.any(results["sublimation_log"] > 0), (
        "Deposition expected under supersaturated conditions."
    )
    assert results["swe_log"][-1] > results["swe_log"][0], (
        "SWE must grow due to deposition."
    )

    _verify_scenario_mass_balance(results)

    _plot_scenario_results(
        scenario_name="deposition",
        timesteps=timesteps,
        results=results,
        precip_mm_per_hr=pr_series * 3600.0,
        air_temp_C=air_temp_C,
        sw_rad_W_per_m2=sw_rad,
        lw_rad_W_per_m2=lw_rad,
    )


def test_intermittent_snowfall_scenario() -> None:
    """Multiple snowfall events with intervening melt periods over 96 h.

    Snowfall events are placed during cold nighttime hours (t ≈ 2–6, 26–30,
    70–74) when the diurnal cycle is below 0 °C, so precipitation is classified
    as snow.  Melt occurs during the warmer daytime intervals.
    """
    n_hours = 96
    timesteps = np.arange(n_hours)

    # Diurnal cycle: minimum ≈ -7 °C at t=0, maximum ≈ +5 °C at t=12.
    air_temp_C = (6 * np.sin(2 * np.pi * (timesteps - 6) / 24) - 1).astype(np.float32)

    # Precipitation events placed in cold nighttime windows.
    pr_series = np.zeros(n_hours, dtype=np.float32)
    pr_series[2:7] = np.float32(0.001 / 3.6)  # First snowfall: hours 2-6 (~-6 °C)
    pr_series[26:31] = np.float32(0.0015 / 3.6)  # Second snowfall: hours 26-30 (~-7 °C)
    pr_series[70:75] = np.float32(0.002 / 3.6)  # Third snowfall: hours 70-74 (~-6 °C)

    sw_rad = np.maximum(0, 500 * np.sin(2 * np.pi * (timesteps - 6) / 24)).astype(
        np.float32
    )
    lw_rad = (290 + 50 * np.sin(2 * np.pi * (timesteps - 18) / 24)).astype(np.float32)
    vapor_pressure = _vapor_pressure_Pa(air_temp_C - 3).astype(np.float32)

    results = _run_scenario(
        n_hours=n_hours,
        precip_kg_per_m2_per_s_series=pr_series,
        air_temp_series_C=air_temp_C,
        sw_rad_series_W_per_m2=sw_rad,
        lw_rad_series_W_per_m2=lw_rad,
        vapor_pressure_series_Pa=vapor_pressure,
        air_pressure_Pa=np.float32(96000.0),
        wind_10m_m_per_s=np.float32(2.5),
        initial_swe_m=0.0,
        initial_snow_temperature_C=0.0,
    )

    assert results["swe_log"][7] > 0.0, "SWE must be positive after first snowfall."
    assert results["swe_log"][31] > results["swe_log"][25], (
        "Second snowfall event must increase SWE."
    )
    # At least one melt-driven reduction in SWE must occur.
    assert np.any(results["swe_log"] < np.maximum.accumulate(results["swe_log"])), (
        "Melt should cause SWE to decrease after peak."
    )
    assert results["swe_log"][-1] == 0.0, (
        "Snow should not persist at the end due to low albedo of shallow snow."
    )

    _verify_scenario_mass_balance(results)

    _plot_scenario_results(
        scenario_name="intermittent_snowfall",
        timesteps=timesteps,
        results=results,
        precip_mm_per_hr=pr_series * 3600.0,
        air_temp_C=air_temp_C,
        sw_rad_W_per_m2=sw_rad,
        lw_rad_W_per_m2=lw_rad,
    )


def test_summer_no_deposition() -> None:
    """Ensure no snow deposition occurs when temperatures are above freezing."""
    n_hours = 48
    timesteps = np.arange(n_hours)

    # Force summer conditions with very high humidity to promote condensation
    air_temp_C = np.full(n_hours, 15.0, dtype=np.float32)
    pr_series = np.zeros(n_hours, dtype=np.float32)
    sw_rad = np.full(n_hours, 300.0, dtype=np.float32)
    lw_rad = np.full(n_hours, 350.0, dtype=np.float32)

    # High vapor pressure (near saturation at 15 C is ~1700 Pa)
    vapor_pressure = np.full(n_hours, 1600.0, dtype=np.float32)

    results = _run_scenario(
        n_hours=n_hours,
        precip_kg_per_m2_per_s_series=pr_series,
        air_temp_series_C=air_temp_C,
        sw_rad_series_W_per_m2=sw_rad,
        lw_rad_series_W_per_m2=lw_rad,
        vapor_pressure_series_Pa=vapor_pressure,
        air_pressure_Pa=np.float32(101325.0),
        wind_10m_m_per_s=np.float32(2.0),
        initial_swe_m=0.0,
        initial_snow_temperature_C=0.0,
    )

    assert np.all(results["sublimation_log"] <= 0.0), (
        f"Deposition occurred during summer conditions! Max rate: {np.max(results['sublimation_log'])}"
    )


def test_glacier_ice_scenario() -> None:
    """Deep glacier-like snowpack should show daily melt cycles over 72 h."""
    n_hours = 72
    timesteps = np.arange(n_hours)

    pr_series = np.zeros(n_hours, dtype=np.float32)
    air_temp_C = np.zeros(n_hours, dtype=np.float32)
    sw_rad = np.zeros(n_hours, dtype=np.float32)
    lw_rad = np.zeros(n_hours, dtype=np.float32)
    vapor_pressure = np.zeros(n_hours, dtype=np.float32)

    for t in timesteps:
        hour = int(t) % 24
        if hour < 12:
            pr_series[t] = np.float32(0.0005 / 3.6)
            air_temp_C[t] = -10.0 + hour * 0.5
            sw_rad[t] = 0.0
            lw_rad[t] = 250.0
            dewpoint = -12.0
        else:
            pr_series[t] = 0.0
            air_temp_C[t] = 5.0 + (hour - 12) * 0.5
            sw_rad[t] = 200.0 + (hour - 12) * 50.0
            lw_rad[t] = 300.0
            dewpoint = 2.0
        vapor_pressure[t] = 610.94 * np.exp(17.625 * dewpoint / (243.04 + dewpoint))

    results = _run_scenario(
        n_hours=n_hours,
        precip_kg_per_m2_per_s_series=pr_series,
        air_temp_series_C=air_temp_C.astype(np.float32),
        sw_rad_series_W_per_m2=sw_rad.astype(np.float32),
        lw_rad_series_W_per_m2=lw_rad.astype(np.float32),
        vapor_pressure_series_Pa=vapor_pressure.astype(np.float32),
        air_pressure_Pa=np.float32(70000.0),
        wind_10m_m_per_s=np.float32(2.0),
        initial_swe_m=20.0,
        initial_snow_temperature_C=-15.0,
        activate_layer_thickness_m=np.float32(2.0),
    )

    assert np.sum(results["melt_log"]) > 0, "Melt must occur in glacier scenario."
    assert results["swe_log"][-1] > results["swe_log"][0], (
        "Net SWE gain expected for glacier under diurnal melt forcing due to high protective albedo."
    )

    _plot_scenario_results(
        scenario_name="glacier_ice",
        timesteps=timesteps,
        results=results,
        precip_mm_per_hr=pr_series * 3600.0,
        air_temp_C=air_temp_C.astype(np.float32),
        sw_rad_W_per_m2=sw_rad.astype(np.float32),
        lw_rad_W_per_m2=lw_rad.astype(np.float32),
    )
