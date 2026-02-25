"""Tests for scalar soil energy functions in GEB."""

import numpy as np

from geb.hydrology.landsurface.energy import (
    calculate_sensible_heat_flux,
    calculate_thermal_conductivity_solid_fraction_watt_per_meter_kelvin,
    get_heat_capacity_solid_fraction,
    solve_energy_balance_implicit_iterative,
    solve_soil_temperature_column,
)


def test_get_heat_capacity() -> None:
    """Test get_heat_capacity_solid_fraction."""
    # Test valid input
    # Bulk density of 1.3 g/cm3 (~1300 kg/m3) should yield a porosity of roughly 0.5
    # (actually 1 - 1300/2650 = 0.49 void, 0.51 solid).
    # Solid fraction phi_s = 1300 / 2650 = 0.490566
    # Volumetric Heat capacity C_s = phi_s * C_mineral = 0.490566 * 2.13e6 = 1.045e6
    # Layer thickness = 1.0 m
    # Areal Heat Capacity = 1.045e6 * 1.0 = 1.045e6 J/(m2 K)

    bulk_density = np.array([1.3], dtype=np.float32)
    layer_thickness = np.array([1.0], dtype=np.float32)

    expected_phi_s = 1300.0 / 2650.0
    volumetric_hc = expected_phi_s * 2.13e6
    expected_areal_hc = volumetric_hc * 1.0

    result = get_heat_capacity_solid_fraction(bulk_density, layer_thickness)

    np.testing.assert_allclose(
        result, np.array([expected_areal_hc], dtype=np.float32), rtol=1e-5
    )

    # Test with bulk density equal to mineral density (solid rock)
    # Bulk density 2.65 g/cm3. phi_s should be 1.0. Heat capacity = C_mineral = 2.13e6.
    # Layer thickness = 2.0 m
    # Areal Heat Capacity = 2.13e6 * 2.0

    bulk_density_rock = np.array([2.65], dtype=np.float32)
    layer_thickness_rock = np.array([2.0], dtype=np.float32)

    expected_hc_rock = 2.13e6
    expected_areal_hc_rock = expected_hc_rock * 2.0

    result_rock = get_heat_capacity_solid_fraction(
        bulk_density_rock, layer_thickness_rock
    )
    np.testing.assert_allclose(
        result_rock, np.array([expected_areal_hc_rock], dtype=np.float32), rtol=1e-5
    )

    # Test with multiple layers summing to 1.0 m thickness
    # Bulk density 1.3 g/cm3 for all layers
    bulk_density_val = 1.3
    # Layer thicknesses: 0.1, 0.2, 0.3, 0.4 -> Sum = 1.0
    layer_thicknesses = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    bulk_densities = np.full_like(layer_thicknesses, bulk_density_val)

    # Expected volumetric heat capacity (same as first test case)
    expected_phi_s_multi = 1300.0 / 2650.0
    expected_volumetric_hc_multi = expected_phi_s_multi * 2.13e6

    # Calculate for multi-layer case
    result_multi = get_heat_capacity_solid_fraction(bulk_densities, layer_thicknesses)

    # Total heat capacity sum should match calculation for single 1.0m layer
    total_heat_capacity_sum = np.sum(result_multi)
    expected_total_1m = expected_volumetric_hc_multi * 1.0

    np.testing.assert_allclose(
        total_heat_capacity_sum,
        expected_total_1m,
        rtol=1e-5,
        err_msg="Sum of layer heat capacities should match single block of combined thickness",
    )

    # Individual layer check
    expected_result_multi = expected_volumetric_hc_multi * layer_thicknesses
    np.testing.assert_allclose(
        result_multi,
        expected_result_multi,
        rtol=1e-5,
        err_msg="Individual layer heat capacities mismatch",
    )


def test_calculate_thermal_conductivity_solid_fraction() -> None:
    """Test calculate_thermal_conductivity_solid_fraction.

    Validates against Johansen (1975) parameterization and expected ranges
    for different soil textures.
    """
    # Pure Sand (100% Sand)
    # Quartz ~ 100% -> q = 1.0
    # lambda_s = 7.7^1.0 * 2.0^0 = 7.7
    res_sand = calculate_thermal_conductivity_solid_fraction_watt_per_meter_kelvin(
        np.float32(100.0), np.float32(0.0), np.float32(0.0)
    )
    assert abs(res_sand - 7.7) < 1e-4

    # Pure Clay (0% Sand)
    # Quartz ~ 0% -> q = 0.0
    # lambda_s = 7.7^0 * 2.0^1 = 2.0
    res_clay = calculate_thermal_conductivity_solid_fraction_watt_per_meter_kelvin(
        np.float32(0.0), np.float32(0.0), np.float32(100.0)
    )
    assert abs(res_clay - 2.0) < 1e-4

    # Loam (40% Sand, 40% Silt, 20% Clay)
    # Quartz ~ 40% -> q = 0.4
    # lambda_s = 7.7^0.4 * 2.0^0.6
    expected_loam = (7.7**0.4) * (2.0**0.6)
    res_loam = calculate_thermal_conductivity_solid_fraction_watt_per_meter_kelvin(
        np.float32(40.0), np.float32(40.0), np.float32(20.0)
    )
    assert abs(res_loam - expected_loam) < 1e-4

    # Plausibility checks for typical soil textures (mostly to avoid unit errors)
    # Solid soil thermal conductivity typically ranges from 2.0 to 9.0 W/(m·K)
    # depending on quartz content.
    textures = [
        (10, 10, 80),  # Heavy clay
        (50, 40, 10),  # Sandy loam
        (90, 5, 5),  # Sand
    ]
    for sand, silt, clay in textures:
        res = calculate_thermal_conductivity_solid_fraction_watt_per_meter_kelvin(
            np.float32(sand), np.float32(silt), np.float32(clay)
        )
        # Literature range for solid particles
        assert 2.0 <= res <= 7.7, (
            f"Conductivity {res} outside common range for {sand}/{silt}/{clay}"
        )


def test_calculate_sensible_heat_flux() -> None:
    """Test the calculate_sensible_heat_flux function."""
    # Equilibrium (No transfer)
    flux, G = calculate_sensible_heat_flux(
        soil_temperature_C=np.float32(20.0),
        air_temperature_K=np.float32(293.15),  # 20C
        wind_speed_10m_m_per_s=np.float32(2.0),
        surface_pressure_pa=np.float32(101325.0),
    )
    assert abs(flux) < 1e-4

    # Air warmer than soil (Warming)
    flux_warming, _ = calculate_sensible_heat_flux(
        soil_temperature_C=np.float32(10.0),
        air_temperature_K=np.float32(293.15),  # 20C
        wind_speed_10m_m_per_s=np.float32(5.0),
        surface_pressure_pa=np.float32(101325.0),
    )
    assert flux_warming > 0.0

    # Soil warmer than air (Cooling)
    flux_cooling, _ = calculate_sensible_heat_flux(
        soil_temperature_C=np.float32(30.0),
        air_temperature_K=np.float32(293.15),  # 20C
        wind_speed_10m_m_per_s=np.float32(5.0),
        surface_pressure_pa=np.float32(101325.0),
    )
    assert flux_cooling < 0.0


def test_solve_energy_balance_implicit_iterative() -> None:
    """Test the implicit iterative energy balance solver."""
    # Common Parameters (Scalars)
    soil_temperature_old = np.float32(10.0)  # 10 C
    bulk_density = np.float32(1300.0)
    layer_thickness = np.float32(0.1)

    # Calculate heat capacity for scalar input
    # Note: get_heat_capacity_solid_fraction expects arrays usually, but let's see.
    # It seems to accept arrays. Let's compute it with array helper but use result as scalar.
    heat_capacity_arr = get_heat_capacity_solid_fraction(
        np.array([bulk_density / 1000.0], dtype=np.float32),
        np.array([layer_thickness], dtype=np.float32),
    )
    heat_capacity_areal = heat_capacity_arr[0]

    # Steady State / No Forcing
    sw_in = np.float32(0.0)
    lw_in = np.float32(363.0)  # Approx balance
    air_temp_k = np.float32(283.15)
    wind_speed = np.float32(2.0)
    pressure = np.float32(101325.0)
    dt_seconds = np.float32(3600.0)
    soil_emissivity = np.float32(0.95)
    soil_albedo = np.float32(0.23)

    t_new = solve_energy_balance_implicit_iterative(
        soil_temperature_C=soil_temperature_old,
        shortwave_radiation_W_per_m2=sw_in,
        longwave_radiation_W_per_m2=lw_in,
        air_temperature_K=air_temp_k,
        wind_speed_10m_m_per_s=wind_speed,
        surface_pressure_pa=pressure,
        solid_heat_capacity_J_per_m2_K=heat_capacity_areal,
        timestep_seconds=dt_seconds,
        soil_emissivity=soil_emissivity,
        soil_albedo=soil_albedo,
        leaf_area_index=np.float32(0.0),
    )

    # Should stay close to 10.0
    assert abs(t_new - 10.0) < 0.5, f"Steady state failed, got {t_new}"

    # Strong heating (daytime)
    sw_in = np.float32(800.0)
    lw_in = np.float32(350.0)
    air_temp_k = np.float32(303.15)

    t_new_hot = solve_energy_balance_implicit_iterative(
        soil_temperature_C=soil_temperature_old,
        shortwave_radiation_W_per_m2=sw_in,
        longwave_radiation_W_per_m2=lw_in,
        air_temperature_K=air_temp_k,
        wind_speed_10m_m_per_s=wind_speed,
        surface_pressure_pa=pressure,
        solid_heat_capacity_J_per_m2_K=heat_capacity_areal,
        timestep_seconds=dt_seconds,
        soil_emissivity=soil_emissivity,
        soil_albedo=soil_albedo,
        leaf_area_index=np.float32(0.0),
    )
    assert t_new_hot > 10.0, "Soil should warm up significantly"

    # Strong cooling (nighttime clear sky)
    sw_in = np.float32(0.0)
    lw_in = np.float32(200.0)
    air_temp_k = np.float32(263.15)

    t_new_cold = solve_energy_balance_implicit_iterative(
        soil_temperature_C=soil_temperature_old,
        shortwave_radiation_W_per_m2=sw_in,
        longwave_radiation_W_per_m2=lw_in,
        air_temperature_K=air_temp_k,
        wind_speed_10m_m_per_s=wind_speed,
        surface_pressure_pa=pressure,
        solid_heat_capacity_J_per_m2_K=heat_capacity_areal,
        timestep_seconds=dt_seconds,
        soil_emissivity=soil_emissivity,
        soil_albedo=soil_albedo,
        leaf_area_index=np.float32(0.0),
    )

    assert t_new_cold < 10.0, "Soil should cool down"


def test_solve_soil_temperature_column() -> None:
    """Test the 1D soil temperature column solver."""
    # Common Parameters
    n_soil_layers = 6
    soil_temperatures_old = np.full(n_soil_layers, 10.0, dtype=np.float32)
    layer_thicknesses = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.0], dtype=np.float32)

    # Simplified heat capacities and conductivities
    # (Using the same logic as test_solve_energy_balance_implicit_iterative but for multiple layers)
    bulk_density = np.float32(1300.0)
    heat_capacity_arr = get_heat_capacity_solid_fraction(
        np.full(n_soil_layers, bulk_density / 1000.0, dtype=np.float32),
        layer_thicknesses,
    )
    thermal_conductivities = np.full(n_soil_layers, 2.0, dtype=np.float32)

    # Steady State / No Forcing
    sw_in = np.float32(0.0)
    lw_in = np.float32(363.0)  # Approx balance
    air_temp_k = np.float32(283.15)
    wind_speed = np.float32(2.0)
    pressure = np.float32(101325.0)
    dt_seconds = np.float32(3600.0)
    deep_soil_temp = np.float32(10.0)
    soil_emissivity = np.float32(0.95)
    soil_albedo = np.float32(0.23)

    t_new, _ = solve_soil_temperature_column(
        soil_temperatures_C=soil_temperatures_old,
        layer_thicknesses_m=layer_thicknesses,
        solid_heat_capacities_J_per_m2_K=heat_capacity_arr,
        thermal_conductivities_W_per_m_K=thermal_conductivities,
        shortwave_radiation_W_per_m2=sw_in,
        longwave_radiation_W_per_m2=lw_in,
        air_temperature_K=air_temp_k,
        wind_speed_10m_m_per_s=wind_speed,
        surface_pressure_pa=pressure,
        timestep_seconds=dt_seconds,
        deep_soil_temperature_C=deep_soil_temp,
        soil_emissivity=soil_emissivity,
        soil_albedo=soil_albedo,
        leaf_area_index=np.float32(0.0),
    )

    # Should stay close to 10.0 across all layers
    np.testing.assert_allclose(t_new, 10.0, atol=0.5)

    # Strong heating (daytime) - Top layer should warm most
    sw_in = np.float32(800.0)
    lw_in = np.float32(350.0)
    air_temp_k = np.float32(303.15)

    t_new_hot, _ = solve_soil_temperature_column(
        soil_temperatures_C=soil_temperatures_old,
        layer_thicknesses_m=layer_thicknesses,
        solid_heat_capacities_J_per_m2_K=heat_capacity_arr,
        thermal_conductivities_W_per_m_K=thermal_conductivities,
        shortwave_radiation_W_per_m2=sw_in,
        longwave_radiation_W_per_m2=lw_in,
        air_temperature_K=air_temp_k,
        wind_speed_10m_m_per_s=wind_speed,
        surface_pressure_pa=pressure,
        timestep_seconds=dt_seconds,
        deep_soil_temperature_C=deep_soil_temp,
        soil_emissivity=soil_emissivity,
        soil_albedo=soil_albedo,
        leaf_area_index=np.float32(0.0),
    )
    assert t_new_hot[0] > 10.0, "Top layer should warm up"
    assert t_new_hot[0] > t_new_hot[1], "Top layer should be warmer than second layer"

    # Bottom boundary influence (Deep soil heating)
    deep_soil_temp_hot = np.float32(20.0)
    t_new_bottom, _ = solve_soil_temperature_column(
        soil_temperatures_C=soil_temperatures_old,
        layer_thicknesses_m=layer_thicknesses,
        solid_heat_capacities_J_per_m2_K=heat_capacity_arr,
        thermal_conductivities_W_per_m_K=thermal_conductivities,
        shortwave_radiation_W_per_m2=np.float32(0.0),
        longwave_radiation_W_per_m2=np.float32(363.0),
        air_temperature_K=np.float32(283.15),
        wind_speed_10m_m_per_s=np.float32(2.0),
        surface_pressure_pa=np.float32(101325.0),
        timestep_seconds=np.float32(3600.0 * 24.0 * 10),  # Long time to see diffusion
        deep_soil_temperature_C=deep_soil_temp_hot,
        soil_emissivity=soil_emissivity,
        soil_albedo=soil_albedo,
        leaf_area_index=np.float32(0.0),
    )
    assert t_new_bottom[-1] > 10.0, "Bottom layer should warm up from deep soil"
    assert t_new_bottom[-1] > t_new_bottom[-2], (
        "Bottom layer should be warmer than one above"
    )


def test_solve_soil_temperature_column_snow() -> None:
    """Test that snow cover effectively blocks surface fluxes."""
    # Setup similar to basic test
    n_soil_layers = 2
    soil_temperatures_old = np.full(n_soil_layers, 10.0, dtype=np.float32)
    layer_thicknesses = np.array([0.1, 0.2], dtype=np.float32)
    bulk_density = np.float32(1300.0)
    heat_capacity_arr = get_heat_capacity_solid_fraction(
        np.full(n_soil_layers, bulk_density / 1000.0, dtype=np.float32),
        layer_thicknesses,
    )
    thermal_conductivities = np.full(n_soil_layers, 2.0, dtype=np.float32)

    # Strong incoming radiation that WOULD heat the soil if no snow
    sw_in = np.float32(1000.0)
    lw_in = np.float32(400.0)
    air_temp_k = np.float32(303.15)  # 30C
    wind_speed = np.float32(5.0)
    pressure = np.float32(101325.0)
    dt_seconds = np.float32(3600.0)
    deep_soil_temp = np.float32(10.0)  # Same as initial
    soil_emissivity = np.float32(0.95)
    soil_albedo = np.float32(0.23)

    # WITH SNOW
    t_new_snow, fluxes_snow = solve_soil_temperature_column(
        soil_temperatures_C=soil_temperatures_old,
        layer_thicknesses_m=layer_thicknesses,
        solid_heat_capacities_J_per_m2_K=heat_capacity_arr,
        thermal_conductivities_W_per_m_K=thermal_conductivities,
        shortwave_radiation_W_per_m2=sw_in,
        longwave_radiation_W_per_m2=lw_in,
        air_temperature_K=air_temp_k,
        wind_speed_10m_m_per_s=wind_speed,
        surface_pressure_pa=pressure,
        timestep_seconds=dt_seconds,
        deep_soil_temperature_C=deep_soil_temp,
        soil_emissivity=soil_emissivity,
        soil_albedo=soil_albedo,
        leaf_area_index=np.float32(0.0),
        snow_water_equivalent_m=np.float32(0.1),  # Significant snow
    )

    # Fluxes should be zero (adiabatic surface boundary as configured)
    assert fluxes_snow == 0.0
    # Temperature should not change significantly (only internal redistribution if any gradient existed, but here uniform 10C)
    np.testing.assert_allclose(t_new_snow, 10.0, atol=1e-4)

    # WITHOUT SNOW (Control)
    t_new_control, fluxes_control = solve_soil_temperature_column(
        soil_temperatures_C=soil_temperatures_old,
        layer_thicknesses_m=layer_thicknesses,
        solid_heat_capacities_J_per_m2_K=heat_capacity_arr,
        thermal_conductivities_W_per_m_K=thermal_conductivities,
        shortwave_radiation_W_per_m2=sw_in,
        longwave_radiation_W_per_m2=lw_in,
        air_temperature_K=air_temp_k,
        wind_speed_10m_m_per_s=wind_speed,
        surface_pressure_pa=pressure,
        timestep_seconds=dt_seconds,
        deep_soil_temperature_C=deep_soil_temp,
        soil_emissivity=soil_emissivity,
        soil_albedo=soil_albedo,
        leaf_area_index=np.float32(0.0),
        snow_water_equivalent_m=np.float32(0.0),
    )

    # Control should have heated up
    assert t_new_control[0] > 10.1
    assert fluxes_control > 0.0  # Positive flux into soil
