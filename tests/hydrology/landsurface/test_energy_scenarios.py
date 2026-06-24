"""Tests for soil energy solver scenarios with visual exports."""

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from geb.hydrology.landsurface.constants import (
    KELVIN_OFFSET,
    N_SOIL_LAYERS,
    RHO_WATER_KG_PER_M3,
    SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K,
    STEFAN_BOLTZMANN_W_PER_M2_K4,
    THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN,
    THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN,
    VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K,
)
from geb.hydrology.landsurface.energy import (
    get_temperature_from_enthalpy,
    solve_soil_enthalpy_column,
)
from geb.hydrology.landsurface.snow_glaciers import (
    FRESH_SNOW_DENSITY_KG_PER_M3,
    get_snow_enthalpy_from_temperature,
)
from tests.testconfig import output_folder


class EnergySimulationResults(NamedTuple):
    """Container for energy simulation diagnostics."""

    time_hours: np.ndarray
    air_temp_C: np.ndarray
    soil_temps_C: np.ndarray  # (n_steps, n_layers)
    snow_temps_C: np.ndarray  # (n_steps, n_snow_layers)
    snow_depth_m: np.ndarray  # (n_steps, n_snow_layers)
    surface_flux_W_per_m2: np.ndarray
    top_frozen_fraction: np.ndarray


def run_energy_simulation(
    air_temp_series_C: np.ndarray,
    initial_soil_temp_C: float = 10.0,
    initial_swe_m: float = 0.0,
    initial_snow_temp_C: float = -2.0,
    timestep_seconds: float = 3600.0,
    shortwave_peak_W_m2: float = 0.0,
    leaf_area_index: float = 0.0,
    title: str = "Energy Simulation",
) -> EnergySimulationResults:
    """Run a time-series simulation of soil energy balance.

    Args:
        air_temp_series_C: Array of hourly air temperatures in C.
        initial_soil_temp_C: Initial uniform soil temperature.
        initial_swe_m: Initial snow water equivalent (m).
        initial_snow_temp_C: Initial snow temperature.
        timestep_seconds: Timestep in seconds.
        shortwave_peak_W_m2: Peak shortwave radiation for diurnal cycle.
        leaf_area_index: Leaf area index for shading.
        title: Title for plots.

    Returns:
        EnergySimulationResults: Aggregated diagnostics.
    """
    n_steps = len(air_temp_series_C)
    n_layers = N_SOIL_LAYERS
    layer_thickness_m = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2], dtype=np.float32)

    # Properties
    porosity = np.full(n_layers, 0.4, dtype=np.float32)
    sand_percentage = np.full(n_layers, 50.0, dtype=np.float32)
    volumetric_water_content = np.full(n_layers, 0.2, dtype=np.float32)
    thermal_conductivity_solid_W_per_m_K = np.full(n_layers, 2.0, dtype=np.float32)

    conductivity_solid_factor = thermal_conductivity_solid_W_per_m_K ** (
        np.float32(1.0) - porosity
    )
    thermal_conductivity_dry_soil_W_per_m_K = (
        np.float32(0.053) * thermal_conductivity_solid_W_per_m_K + np.float32(0.0051)
    ) / (np.float32(1.0) - np.float32(0.947) * porosity)

    solid_heat_capacity_J_per_m2_K = np.array(
        [2.0e6 * d for d in layer_thickness_m], dtype=np.float32
    )
    thermal_conductivity_saturated_unfrozen_W_per_m_K = conductivity_solid_factor * (
        THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN**porosity
    )
    thermal_conductivity_saturated_frozen_W_per_m_K = conductivity_solid_factor * (
        THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN**porosity
    )

    # Initial State
    # Calculate initial enthalpy (Liquid)
    water_heat_capacity_areal = (
        volumetric_water_content
        * layer_thickness_m
        * VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K
    )
    total_heat_capacity_areal = (
        solid_heat_capacity_J_per_m2_K + water_heat_capacity_areal
    )
    soil_enthalpies_J_per_m2 = total_heat_capacity_areal * np.float32(
        initial_soil_temp_C
    )

    # Snow State
    n_snow_layers = 2
    snow_water_equivalent_m = np.zeros(n_snow_layers, dtype=np.float64)
    snow_water_equivalent_m[0] = initial_swe_m / 2.0
    snow_water_equivalent_m[1] = initial_swe_m / 2.0

    snow_enthalpy_J_per_m2 = np.array(
        [
            get_snow_enthalpy_from_temperature(
                snow_water_equivalent_m[0], np.float32(initial_snow_temp_C)
            ),
            get_snow_enthalpy_from_temperature(
                snow_water_equivalent_m[1], np.float32(initial_snow_temp_C)
            ),
        ],
        dtype=np.float32,
    )

    snow_density_kg_per_m3 = np.full(
        n_snow_layers, FRESH_SNOW_DENSITY_KG_PER_M3, dtype=np.float32
    )

    # Logging
    soil_temps_history = np.zeros((n_steps, n_layers))
    snow_temps_history = np.zeros((n_steps, n_snow_layers))
    snow_depth_history = np.zeros((n_steps, n_snow_layers))
    flux_history = np.zeros(n_steps)
    frozen_fraction_history = np.zeros(n_steps)

    soil_emissivity = np.float32(0.95)
    soil_albedo = np.float32(0.2)

    for t in range(n_steps):
        air_temp_K = np.float32(air_temp_series_C[t] + KELVIN_OFFSET)

        # Diurnal shortwave
        hour = t % 24
        sw_in = np.float32(
            max(0.0, shortwave_peak_W_m2 * np.sin(np.pi * (hour - 6) / 12))
        )

        # LW in approx = sigma * Ta^4
        incoming_lw = soil_emissivity * STEFAN_BOLTZMANN_W_PER_M2_K4 * (air_temp_K**4)

        heat_flux, frozen_fraction_top_layer = solve_soil_enthalpy_column(
            soil_enthalpies_J_per_m2=soil_enthalpies_J_per_m2,
            layer_thicknesses_m=layer_thickness_m,
            thermal_conductivity_dry_soil_W_per_m_K=thermal_conductivity_dry_soil_W_per_m_K,
            solid_heat_capacities_J_per_m2_K=solid_heat_capacity_J_per_m2_K,
            thermal_conductivity_saturated_unfrozen_W_per_m_K=thermal_conductivity_saturated_unfrozen_W_per_m_K,
            thermal_conductivity_saturated_frozen_W_per_m_K=thermal_conductivity_saturated_frozen_W_per_m_K,
            water_content_saturated_m=porosity * layer_thickness_m,
            sand_percentage=sand_percentage,
            water_content_m=volumetric_water_content * layer_thickness_m,
            shortwave_radiation_W_per_m2=sw_in,
            longwave_radiation_W_per_m2=incoming_lw,
            air_temperature_K=air_temp_K,
            wind_speed_10m_m_per_s=np.float32(2.0),
            surface_pressure_pa=np.float32(101325.0),
            timestep_seconds=np.float32(timestep_seconds),
            deep_soil_temperature_C=np.float32(initial_soil_temp_C),
            soil_emissivity=soil_emissivity,
            soil_albedo=soil_albedo,
            leaf_area_index=np.float32(leaf_area_index),
            snow_water_equivalent_m=snow_water_equivalent_m,
            snow_enthalpy_J_per_m2=snow_enthalpy_J_per_m2,
            snow_density_kg_per_m3=snow_density_kg_per_m3,
            topwater_m=np.float32(0.0),
        )

        # Diagnose temperatures
        current_soil_temps = get_temperature_from_enthalpy(
            soil_enthalpies_J_per_m2,
            solid_heat_capacity_J_per_m2_K,
            volumetric_water_content * layer_thickness_m,
        )
        soil_temps_history[t, :] = current_soil_temps

        # Snow temp diag (Simplified for logging)
        for i in range(n_snow_layers):
            if snow_water_equivalent_m[i] > 1e-12:  # Include trace snow
                c_snow = (
                    np.float32(snow_water_equivalent_m[i])
                    * RHO_WATER_KG_PER_M3
                    * SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K
                )
                snow_temps_history[t, i] = snow_enthalpy_J_per_m2[i] / c_snow
                snow_depth_history[t, i] = (
                    snow_water_equivalent_m[i]
                    * RHO_WATER_KG_PER_M3
                    / snow_density_kg_per_m3[i]
                )
            else:
                snow_temps_history[t, i] = np.nan
                snow_depth_history[t, i] = 0.0

        flux_history[t] = heat_flux
        frozen_fraction_history[t] = frozen_fraction_top_layer

    return EnergySimulationResults(
        time_hours=np.arange(n_steps) * (timestep_seconds / 3600.0),
        air_temp_C=air_temp_series_C,
        soil_temps_C=soil_temps_history,
        snow_temps_C=snow_temps_history,
        snow_depth_m=snow_depth_history,
        surface_flux_W_per_m2=flux_history,
        top_frozen_fraction=frozen_fraction_history,
    )


def test_energy_snow_comparison() -> None:
    """Compare energy balance scenarios with and without snow."""
    # Define a cold snap: 10C -> -20C for 3 days -> 10C
    n_days = 20
    total_steps = n_days * 24
    air_temp = np.full(total_steps, 10.0)
    air_temp[48 : 48 + 24 * 5] = -20.0  # Cold snap starts day 2
    air_temp[48 + 24 * 5 :] = 2.0  # Slow stable cold

    # Scenario 1: No Snow
    res_no_snow = run_energy_simulation(air_temp, initial_swe_m=0.0, title="Bare Soil")

    # Scenario 2: With Snow (0.2m SWE = ~1m snow)
    res_snow = run_energy_simulation(air_temp, initial_swe_m=0.2, title="Snow Covered")

    # Plot results - Comprehensive Comparison
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), sharex=True, sharey="row")

    # Titles for columns
    axes[0, 0].set_title("SCENARIO A: BARE SOIL", fontsize=14, fontweight="bold")
    axes[0, 1].set_title("SCENARIO B: SNOW COVERED", fontsize=14, fontweight="bold")

    # 1. Temperature Profiles (Snow and Soil)
    for i, res in enumerate([res_no_snow, res_snow]):
        ax = axes[0, i]
        ax.plot(res.time_hours, res.air_temp_C, "k:", alpha=0.3, label="Air Temp")

        # Snow layers
        if i == 1:  # Only for snow scenario
            for j in range(res.snow_temps_C.shape[1]):
                ax.plot(
                    res.time_hours,
                    res.snow_temps_C[:, j],
                    label=f"Snow Layer {j + 1}",
                    linewidth=1.2,
                )

        # Soil layers (Top 3 for clarity)
        colors = ["#d7191c", "#fdae61", "#abdda4"]
        layers = [0.1, 0.2, 0.4]
        for j in range(3):
            ax.plot(
                res.time_hours,
                res.soil_temps_C[:, j],
                color=colors[j],
                label=f"Soil {layers[j]}m",
                linewidth=2,
            )

        ax.axhline(0, color="k", linestyle="-", alpha=0.2)
        ax.set_ylabel("Temperature (C)")
        ax.legend(fontsize="small", loc="lower left")

    # 2. Freezing State & Snow Thickness
    for i, res in enumerate([res_no_snow, res_snow]):
        ax = axes[1, i]
        ax.fill_between(
            res.time_hours, 0, res.top_frozen_fraction, color="blue", alpha=0.3
        )
        ax.plot(
            res.time_hours,
            res.top_frozen_fraction,
            color="blue",
            label="Top Soil Frozen Fraction",
        )
        # Add Snow Thickness if present
        if np.any(res.snow_depth_m > 0):
            ax_twin = ax.twinx()
            for j in range(res.snow_depth_m.shape[1]):
                ax_twin.plot(
                    res.time_hours,
                    res.snow_depth_m[:, j] * 100,
                    label=f"Snow Layer {j + 1} Depth",
                    linestyle=":",
                    alpha=0.7,
                )
            ax_twin.set_ylabel("Snow Depth (cm)")
            ax_twin.legend(loc="upper right", fontsize="x-small")

        ax.set_ylabel("Frozen Fraction (-)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize="small", loc="upper left")

    # 3. Energy Fluxes
    for i, res in enumerate([res_no_snow, res_snow]):
        ax = axes[2, i]
        ax.plot(
            res.time_hours,
            res.surface_flux_W_per_m2,
            color="green",
            label="Net Surface Flux",
        )
        ax.axhline(0, color="k", linestyle="-", alpha=0.2)
        ax.set_ylabel("Flux (W/m2)")
        ax.set_xlabel("Time (hours)")
        ax.legend(fontsize="small")

    # Save to output folder
    energy_output_dir = output_folder / "energy_scenarios"
    energy_output_dir.mkdir(exist_ok=True)
    plot_path = energy_output_dir / "snow_vs_bare_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Enhanced comparison plot saved to {plot_path}")

    # Assertions
    # Bare soil should get much colder than snow-covered soil
    assert np.min(res_no_snow.soil_temps_C[:, 0]) < np.min(res_snow.soil_temps_C[:, 0])
    # Bare soil should freeze more
    assert np.max(res_no_snow.top_frozen_fraction) >= np.max(
        res_snow.top_frozen_fraction
    )
    # Snow covered should stay much warmer
    assert (
        np.min(res_snow.soil_temps_C[:, 0]) > -5.0
    )  # Insulation should keep it near 0


def test_energy_shade_comparison() -> None:
    """Compare energy balance scenarios with and without vegetation shading."""
    # Define a hot week: 20C -> 35C hot days -> 20C
    n_days = 7
    total_steps = n_days * 24
    air_temp = np.full(total_steps, 20.0)
    for d in range(n_days):
        day_slice = slice(d * 24, (d + 1) * 24)
        # Diurnal temp cycle
        air_temp[day_slice] = 20.0 + 15.0 * np.sin(
            np.pi * (np.arange(24) - 6) / 12
        ).clip(0)

    # Scenario 1: Bare Soil (LAI=0, High SW)
    res_bare = run_energy_simulation(
        air_temp,
        initial_soil_temp_C=20.0,
        shortwave_peak_W_m2=800.0,
        leaf_area_index=0.0,
        title="Bare Soil Summer",
    )

    # Scenario 2: Shaded Soil (LAI=5, High SW)
    res_shaded = run_energy_simulation(
        air_temp,
        initial_soil_temp_C=20.0,
        shortwave_peak_W_m2=800.0,
        leaf_area_index=5.0,
        title="Shaded Soil Summer",
    )

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Air Temp & Soil Temps
    axes[0].plot(
        res_bare.time_hours, res_bare.air_temp_C, "k--", alpha=0.5, label="Air Temp"
    )
    axes[0].plot(
        res_bare.time_hours,
        res_bare.soil_temps_C[:, 0],
        "r-",
        label="Bare Soil (0-10cm)",
    )
    axes[0].plot(
        res_shaded.time_hours,
        res_shaded.soil_temps_C[:, 0],
        "g-",
        label="Shaded Soil (0-10cm)",
    )
    axes[0].set_ylabel("Temperature (C)")
    axes[0].legend()
    axes[0].set_title("Canopy Shading Effect (Summer)")

    # Surface Fluxes
    axes[1].plot(
        res_bare.time_hours,
        res_bare.surface_flux_W_per_m2,
        "r-",
        label="Bare Soil Flux",
    )
    axes[1].plot(
        res_shaded.time_hours,
        res_shaded.surface_flux_W_per_m2,
        "g-",
        label="Shaded Soil Flux",
    )
    axes[1].set_ylabel("Net Surface Flux (W/m2)")
    axes[1].set_xlabel("Time (hours)")
    axes[1].legend()
    axes[1].set_title("Energy Flux into Soil")

    # Save to output folder
    energy_output_dir = output_folder / "energy_scenarios"
    energy_output_dir.mkdir(exist_ok=True)
    plot_path = energy_output_dir / "summer_shade_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Summer shade scenario plot saved to {plot_path}")

    # Assertions
    # Bare soil should get much hotter than shaded soil
    assert np.max(res_bare.soil_temps_C[:, 0]) > np.max(res_shaded.soil_temps_C[:, 0])
    # Fluxes should be lower for shaded soil
    assert np.max(res_bare.surface_flux_W_per_m2) > np.max(
        res_shaded.surface_flux_W_per_m2
    )
