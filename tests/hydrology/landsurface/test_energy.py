"""Tests for scalar soil energy functions in GEB."""

import numpy as np

from geb.hydrology.landsurface.energy import (
    calculate_sensible_heat_flux,
    calculate_soil_thermal_conductivity_from_frozen_fraction,
    calculate_thermal_conductivity_solid_fraction_watt_per_meter_kelvin,
    get_heat_capacity_solid_fraction,
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


def test_calculate_soil_thermal_conductivity() -> None:
    """Test the total soil thermal conductivity calculation."""
    # Input data
    lambda_s = np.array([2.5], dtype=np.float32)
    bd = np.array([1.3], dtype=np.float32)
    sand = np.array([50.0], dtype=np.float32)
    # theta = np.array([0.2], dtype=np.float32)
    temp_hot = np.array([20.0], dtype=np.float32)
    temp_cold = np.array([-10.0], dtype=np.float32)

    # Calculate porosity to get degree of saturation
    RHO_MINERAL = 2650.0
    rho_bulk = bd * 1000.0
    phi = 1.0 - (rho_bulk / RHO_MINERAL)
    Sr = np.array([0.4], dtype=np.float32)
    phi_val = phi.astype(np.float32)
    bd_val = bd.astype(np.float32)

    # Calculate for unfrozen (frozen_fraction = 0.0)
    lambda_total_hot = calculate_soil_thermal_conductivity_from_frozen_fraction(
        thermal_conductivity_solid_W_per_m_K=lambda_s,
        bulk_density_kg_per_dm3=bd_val,
        porosity=phi_val,
        degree_of_saturation=Sr,
        sand_percentage=sand,
        frozen_fraction=np.array([0.0], dtype=np.float32),
    )

    # Calculate for frozen (frozen_fraction = 1.0)
    lambda_total_cold = calculate_soil_thermal_conductivity_from_frozen_fraction(
        thermal_conductivity_solid_W_per_m_K=lambda_s,
        bulk_density_kg_per_dm3=bd_val,
        porosity=phi_val,
        degree_of_saturation=Sr,
        sand_percentage=sand,
        frozen_fraction=np.array([1.0], dtype=np.float32),
    )

    # Saturated frozen conductivity should be higher than unfrozen because lambda_ice > lambda_water
    assert lambda_total_cold[0] > lambda_total_hot[0]

    # Both should be between lambda_dry (~0.2-0.5) and lambda_sat (~2-4)
    assert lambda_total_hot[0] > 0.1
    assert lambda_total_hot[0] < 5.0


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
