import numpy as np

from geb.hydrology.landsurface.constants import (
    STEFAN_BOLTZMANN_W_PER_M2_K4,
    VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K,
)
from geb.hydrology.landsurface.energy import solve_soil_enthalpy_column


def test_solve_soil_enthalpy_column_energy_balance():
    """Test that the implicit enthalpy solver conserves energy and behaves reasonably."""
    # Setup a 3-layer soil column
    n_layers = 3
    layer_thickness_m = np.array([0.1, 0.2, 0.4], dtype=np.float32)

    # Properties
    porosity = np.full(n_layers, 0.4, dtype=np.float32)
    bulk_density_kg_per_dm3 = np.full(n_layers, 1.3, dtype=np.float32)
    sand_percentage = np.full(n_layers, 50.0, dtype=np.float32)
    volumetric_water_content = np.full(n_layers, 0.2, dtype=np.float32)
    degree_of_saturation = volumetric_water_content / porosity

    # Thermal properties
    # Approx mineral capacity J/m2/K scaling with thickness
    solid_heat_capacity_J_per_m2_K = np.array(
        [2.0e6 * d for d in layer_thickness_m], dtype=np.float32
    )
    thermal_conductivity_solid_W_per_m_K = np.full(n_layers, 2.0, dtype=np.float32)

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

    # Run Solver
    new_enthalpies, heat_flux, frozen_fraction = solve_soil_enthalpy_column(
        soil_enthalpies_J_per_m2=soil_enthalpies_J_per_m2,
        layer_thicknesses_m=layer_thickness_m,
        bulk_density_kg_per_dm3=bulk_density_kg_per_dm3,
        solid_heat_capacities_J_per_m2_K=solid_heat_capacity_J_per_m2_K,
        thermal_conductivity_solid_W_per_m_K=thermal_conductivity_solid_W_per_m_K,
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
        snow_water_equivalent_m=np.float32(0.0),
        snow_temperature_C=np.float32(0.0),
        topwater_m=np.float32(0.0),
    )

    # Check 1: Equilibrium should be maintained (approx)
    assert np.isclose(new_enthalpies[0], initial_enthalpy_J_per_m2[0], rtol=1e-4)
    assert np.isclose(heat_flux, 0.0, atol=1.0)  # W/m2

    # Test 2: Strong Heating
    # Add significant Shortwave radiation
    sw_in = np.float32(500.0)  # W/m2

    new_enthalpies_heating, heat_flux_heating, _ = solve_soil_enthalpy_column(
        soil_enthalpies_J_per_m2=soil_enthalpies_J_per_m2,
        layer_thicknesses_m=layer_thickness_m,
        bulk_density_kg_per_dm3=bulk_density_kg_per_dm3,
        solid_heat_capacities_J_per_m2_K=solid_heat_capacity_J_per_m2_K,
        thermal_conductivity_solid_W_per_m_K=thermal_conductivity_solid_W_per_m_K,
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
        snow_water_equivalent_m=np.float32(0.0),
        snow_temperature_C=np.float32(0.0),
        topwater_m=np.float32(0.0),
    )

    # Enthalpy should increase (total)
    total_new_enthalpy = np.sum(new_enthalpies_heating)
    total_initial_enthalpy = np.sum(initial_enthalpy_J_per_m2)
    assert total_new_enthalpy > total_initial_enthalpy

    # Check Energy Balance over the whole column: Delta H_total approx (Flux_in_surface - Flux_out_bottom) * dt
    delta_H_total = total_new_enthalpy - total_initial_enthalpy

    # Calculate bottom flux manually based on new T of the bottom layer
    last_idx = n_layers - 1
    final_temp_C_bottom = (
        new_enthalpies_heating[last_idx] / total_heat_capacity_areal[last_idx]
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

    new_enthalpies_extreme, _, _ = solve_soil_enthalpy_column(
        soil_enthalpies_J_per_m2=soil_enthalpies_J_per_m2,
        layer_thicknesses_m=layer_thickness_m,
        bulk_density_kg_per_dm3=bulk_density_kg_per_dm3,
        solid_heat_capacities_J_per_m2_K=solid_heat_capacity_J_per_m2_K,
        thermal_conductivity_solid_W_per_m_K=thermal_conductivity_solid_W_per_m_K,
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
        snow_water_equivalent_m=np.float32(0.0),
        snow_temperature_C=np.float32(0.0),
        topwater_m=np.float32(0.0),
    )

    assert new_enthalpies_extreme[0] > new_enthalpies_heating[0]
    # Check that it didn't explode to infinity or NaN
    assert np.isfinite(new_enthalpies_extreme[0])


if __name__ == "__main__":
    test_solve_soil_enthalpy_column_energy_balance()
