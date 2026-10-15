"""Test the soil enthalpy solver for energy balance and stability."""

import numpy as np

from geb.hydrology.landsurface.constants import (
    N_SOIL_LAYERS,
    STEFAN_BOLTZMANN_W_PER_M2_K4,
    THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN,
    THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN,
    VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K,
)
from geb.hydrology.landsurface.energy import solve_soil_enthalpy_column
from geb.hydrology.landsurface.snow_glaciers import (
    FRESH_SNOW_DENSITY_KG_PER_M3,
    get_snow_enthalpy_from_temperature,
)


def test_solve_soil_enthalpy_column_energy_balance() -> None:
    """Test that the implicit enthalpy solver conserves energy and behaves reasonably."""
    # The optimized solver is specialized for the configured soil discretization.
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
    # Johansen (1975) empirical dry-soil formula
    thermal_conductivity_dry_soil_W_per_m_K = (
        np.float32(0.053) * thermal_conductivity_solid_W_per_m_K + np.float32(0.0051)
    ) / (np.float32(1.0) - np.float32(0.947) * porosity)

    # Thermal properties
    # Approx mineral capacity J/m2/K scaling with thickness
    solid_heat_capacity_J_per_m2_K = np.array(
        [2.0e6 * d for d in layer_thickness_m], dtype=np.float32
    )
    thermal_conductivity_saturated_unfrozen_W_per_m_K = conductivity_solid_factor * (
        THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN**porosity
    )
    thermal_conductivity_saturated_frozen_W_per_m_K = conductivity_solid_factor * (
        THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN**porosity
    )

    # Initial state: 10 degrees C uniform
    initial_temp_C = np.float32(10.0)
    # Calculate initial enthalpy (Liquid)
    # H = (C_solid + theta * C_water) * T
    water_heat_capacity_areal = (
        volumetric_water_content
        * layer_thickness_m
        * VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K
    )
    total_heat_capacity_areal = (
        solid_heat_capacity_J_per_m2_K + water_heat_capacity_areal
    )
    initial_enthalpy_J_per_m2 = total_heat_capacity_areal * initial_temp_C

    soil_enthalpies_J_per_m2 = initial_enthalpy_J_per_m2.copy()

    # Forcing: Equilibrium condition first
    # If Air T = Soil T, and Longwave in = Longwave out, and Shortwave = 0, Net Flux should be 0.

    soil_emissivity = np.float32(0.95)
    soil_albedo = np.float32(0.2)

    air_temp_K = initial_temp_C + 273.15
    # Outgoing LW = eps * sigma * T^4
    outgoing_lw = soil_emissivity * STEFAN_BOLTZMANN_W_PER_M2_K4 * (air_temp_K**4)
    incoming_lw = (
        outgoing_lw / soil_emissivity
    )  # Simplified approximation for equilibrium test

    timestep_seconds = np.float32(3600.0)
    no_snow_swe_m = np.zeros(2, dtype=np.float64)
    no_snow_enthalpy_J_per_m2 = np.zeros(2, dtype=np.float32)
    no_snow_density_kg_per_m3 = np.full(
        2, FRESH_SNOW_DENSITY_KG_PER_M3, dtype=np.float32
    )

    # Run Solver
    enthalpies_equilibrium = soil_enthalpies_J_per_m2.copy()
    heat_flux, frozen_fraction_top_layer = solve_soil_enthalpy_column(
        soil_enthalpies_J_per_m2=enthalpies_equilibrium,
        layer_thicknesses_m=layer_thickness_m,
        thermal_conductivity_dry_soil_W_per_m_K=thermal_conductivity_dry_soil_W_per_m_K,
        solid_heat_capacities_J_per_m2_K=solid_heat_capacity_J_per_m2_K,
        thermal_conductivity_saturated_unfrozen_W_per_m_K=thermal_conductivity_saturated_unfrozen_W_per_m_K,
        thermal_conductivity_saturated_frozen_W_per_m_K=thermal_conductivity_saturated_frozen_W_per_m_K,
        water_content_saturated_m=porosity * layer_thickness_m,
        sand_percentage=sand_percentage,
        water_content_m=volumetric_water_content * layer_thickness_m,
        shortwave_radiation_W_per_m2=np.float32(0.0),
        longwave_radiation_W_per_m2=incoming_lw,
        air_temperature_K=air_temp_K,
        wind_speed_10m_m_per_s=np.float32(2.0),
        surface_pressure_pa=np.float32(101325.0),
        timestep_seconds=timestep_seconds,
        deep_soil_temperature_C=initial_temp_C,  # Bottom BC same as initial
        soil_emissivity=soil_emissivity,
        soil_albedo=soil_albedo,
        leaf_area_index=np.float32(0.0),
        snow_water_equivalent_m=no_snow_swe_m.copy(),
        snow_enthalpy_J_per_m2=no_snow_enthalpy_J_per_m2.copy(),
        snow_density_kg_per_m3=no_snow_density_kg_per_m3.copy(),
        topwater_m=np.float32(0.0),
    )

    # Check 1: Equilibrium should be maintained (approx)
    assert np.isclose(
        enthalpies_equilibrium[0], initial_enthalpy_J_per_m2[0], rtol=1e-4
    )
    assert np.isclose(heat_flux, 0.0, atol=1.0)  # W/m2
    assert np.float32(0.0) <= frozen_fraction_top_layer <= np.float32(1.0)

    # Test 2: Strong Heating
    # Add significant Shortwave radiation
    sw_in = np.float32(500.0)  # W/m2

    enthalpies_heating = soil_enthalpies_J_per_m2.copy()
    heat_flux_heating, _ = solve_soil_enthalpy_column(
        soil_enthalpies_J_per_m2=enthalpies_heating,
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
        timestep_seconds=timestep_seconds,
        deep_soil_temperature_C=initial_temp_C,
        soil_emissivity=soil_emissivity,
        soil_albedo=soil_albedo,
        leaf_area_index=np.float32(0.0),
        snow_water_equivalent_m=no_snow_swe_m.copy(),
        snow_enthalpy_J_per_m2=no_snow_enthalpy_J_per_m2.copy(),
        snow_density_kg_per_m3=no_snow_density_kg_per_m3.copy(),
        topwater_m=np.float32(0.0),
    )

    # Enthalpy should increase (total)
    total_new_enthalpy = np.sum(enthalpies_heating)
    total_initial_enthalpy = np.sum(initial_enthalpy_J_per_m2)
    assert total_new_enthalpy > total_initial_enthalpy

    # Check Energy Balance over the whole column: Delta H_total approx (Flux_in_surface - Flux_out_bottom) * dt
    delta_H_total = total_new_enthalpy - total_initial_enthalpy

    # Calculate bottom flux manually based on new T of the bottom layer
    last_idx = n_layers - 1
    final_temp_C_bottom = (
        enthalpies_heating[last_idx] / total_heat_capacity_areal[last_idx]
    )

    conductance_to_deep = thermal_conductivity_solid_W_per_m_K[last_idx] / (
        0.5 * layer_thickness_m[last_idx]
    )
    flux_bottom_loss = conductance_to_deep * (
        final_temp_C_bottom - initial_temp_C
    )  # Positive is loss downwards

    # Energy in = (Flux_surface - Flux_bottom) * dt
    # heat_flux_heating is the surface flux into the soil (positive downwards)
    expected_energy_change = (heat_flux_heating - flux_bottom_loss) * timestep_seconds

    # Compare with tolerance
    assert np.isclose(delta_H_total, expected_energy_change, rtol=0.05)

    # Test 3: Linearization stability check
    # Provide an massive heat pulse that would cause instability in explicit schemes
    # The implicit scheme should dampen it and find a valid solution

    enthalpies_extreme = initial_enthalpy_J_per_m2.copy()
    solve_soil_enthalpy_column(
        soil_enthalpies_J_per_m2=enthalpies_extreme,
        layer_thicknesses_m=layer_thickness_m,
        thermal_conductivity_dry_soil_W_per_m_K=thermal_conductivity_dry_soil_W_per_m_K,
        solid_heat_capacities_J_per_m2_K=solid_heat_capacity_J_per_m2_K,
        thermal_conductivity_saturated_unfrozen_W_per_m_K=thermal_conductivity_saturated_unfrozen_W_per_m_K,
        thermal_conductivity_saturated_frozen_W_per_m_K=thermal_conductivity_saturated_frozen_W_per_m_K,
        water_content_saturated_m=porosity * layer_thickness_m,
        sand_percentage=sand_percentage,
        water_content_m=volumetric_water_content * layer_thickness_m,
        shortwave_radiation_W_per_m2=np.float32(5000.0),  # EXTREME
        longwave_radiation_W_per_m2=incoming_lw,
        air_temperature_K=air_temp_K,
        wind_speed_10m_m_per_s=np.float32(2.0),
        surface_pressure_pa=np.float32(101325.0),
        timestep_seconds=timestep_seconds,
        deep_soil_temperature_C=initial_temp_C,
        soil_emissivity=soil_emissivity,
        soil_albedo=soil_albedo,
        leaf_area_index=np.float32(0.0),
        snow_water_equivalent_m=no_snow_swe_m.copy(),
        snow_enthalpy_J_per_m2=no_snow_enthalpy_J_per_m2.copy(),
        snow_density_kg_per_m3=no_snow_density_kg_per_m3.copy(),
        topwater_m=np.float32(0.0),
    )

    assert enthalpies_extreme[0] > initial_enthalpy_J_per_m2[0]
    # Check that it didn't explode to infinity or NaN
    assert np.isfinite(enthalpies_extreme[0])


def test_solve_soil_enthalpy_column_tiny_snow_stays_stable() -> None:
    """Test that trace snow cover does not over-couple the soil thermally."""
    n_layers = N_SOIL_LAYERS
    layer_thickness_m = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2], dtype=np.float32)
    porosity = np.full(n_layers, 0.4, dtype=np.float32)
    sand_percentage = np.full(n_layers, 50.0, dtype=np.float32)
    volumetric_water_content = np.full(n_layers, 0.2, dtype=np.float32)
    thermal_conductivity_solid_W_per_m_K = np.full(n_layers, 2.0, dtype=np.float32)
    conductivity_solid_factor = thermal_conductivity_solid_W_per_m_K ** (
        np.float32(1.0) - porosity
    )
    # Johansen (1975) empirical dry-soil formula
    thermal_conductivity_dry_soil_W_per_m_K = (
        np.float32(0.053) * thermal_conductivity_solid_W_per_m_K + np.float32(0.0051)
    ) / (np.float32(1.0) - np.float32(0.947) * porosity)
    solid_heat_capacity_J_per_m2_K = np.array(
        [2.0e6 * depth_m for depth_m in layer_thickness_m], dtype=np.float32
    )
    thermal_conductivity_saturated_unfrozen_W_per_m_K = conductivity_solid_factor * (
        THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN**porosity
    )
    thermal_conductivity_saturated_frozen_W_per_m_K = conductivity_solid_factor * (
        THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN**porosity
    )

    initial_temperature_C = np.float32(2.0)
    water_heat_capacity_areal_J_per_m2_K = (
        volumetric_water_content
        * layer_thickness_m
        * VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K
    )
    initial_enthalpy_J_per_m2 = (
        solid_heat_capacity_J_per_m2_K + water_heat_capacity_areal_J_per_m2_K
    ) * initial_temperature_C
    trace_snow_swe_m = np.array([1.0e-8, 0.0], dtype=np.float64)
    trace_snow_enthalpy_J_per_m2 = np.array(
        [
            get_snow_enthalpy_from_temperature(trace_snow_swe_m[0], np.float32(-2.0)),
            0.0,
        ],
        dtype=np.float32,
    )
    trace_snow_density_kg_per_m3 = np.full(
        2, FRESH_SNOW_DENSITY_KG_PER_M3, dtype=np.float32
    )

    new_enthalpies_J_per_m2 = initial_enthalpy_J_per_m2.copy()
    solve_soil_enthalpy_column(
        soil_enthalpies_J_per_m2=new_enthalpies_J_per_m2,
        layer_thicknesses_m=layer_thickness_m,
        thermal_conductivity_dry_soil_W_per_m_K=thermal_conductivity_dry_soil_W_per_m_K,
        solid_heat_capacities_J_per_m2_K=solid_heat_capacity_J_per_m2_K,
        thermal_conductivity_saturated_unfrozen_W_per_m_K=thermal_conductivity_saturated_unfrozen_W_per_m_K,
        thermal_conductivity_saturated_frozen_W_per_m_K=thermal_conductivity_saturated_frozen_W_per_m_K,
        water_content_saturated_m=porosity * layer_thickness_m,
        sand_percentage=sand_percentage,
        water_content_m=volumetric_water_content * layer_thickness_m,
        shortwave_radiation_W_per_m2=np.float32(0.0),
        longwave_radiation_W_per_m2=np.float32(0.0),
        air_temperature_K=np.float32(273.15),
        wind_speed_10m_m_per_s=np.float32(2.0),
        surface_pressure_pa=np.float32(101325.0),
        timestep_seconds=np.float32(3600.0),
        deep_soil_temperature_C=initial_temperature_C,
        soil_emissivity=np.float32(0.95),
        soil_albedo=np.float32(0.2),
        leaf_area_index=np.float32(0.0),
        snow_water_equivalent_m=trace_snow_swe_m,
        snow_enthalpy_J_per_m2=trace_snow_enthalpy_J_per_m2,
        snow_density_kg_per_m3=trace_snow_density_kg_per_m3,
        topwater_m=np.float32(0.0),
    )

    assert np.all(np.isfinite(new_enthalpies_J_per_m2))
    assert np.sum(new_enthalpies_J_per_m2) > np.float32(-1.0e7)
