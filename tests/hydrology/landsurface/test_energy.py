"""Tests for scalar soil energy functions in GEB."""

import numpy as np

from geb.hydrology.landsurface.constants import (
    THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN,
    THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN,
)
from geb.hydrology.landsurface.energy import (
    apply_rain_heat_advection,
    calculate_sensible_heat_flux,
    calculate_soil_thermal_conductivity,
    calculate_thermal_conductivity_dry_soil_johansen_watt_per_meter_kelvin,
    calculate_thermal_conductivity_saturated_soil_johansen_watt_per_meter_kelvin,
    calculate_thermal_conductivity_solid_fraction_watt_per_meter_kelvin,
    get_heat_capacity_solid_fraction,
    get_temperature_and_frozen_fraction_from_enthalpy_scalar,
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
    lambda_s = np.float32(2.5)
    sand = np.float32(50.0)

    # Porosity and degree of saturation
    phi_val = np.float32(0.4)
    Sr = np.float32(0.4)

    lambda_sat_unfrozen = (lambda_s ** (np.float32(1.0) - phi_val)) * (
        THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN**phi_val
    )
    lambda_sat_frozen = (lambda_s ** (np.float32(1.0) - phi_val)) * (
        THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN**phi_val
    )
    # Johansen (1975) empirical dry-soil formula
    thermal_conductivity_dry_soil = (
        np.float32(0.053) * lambda_s + np.float32(0.0051)
    ) / (np.float32(1.0) - np.float32(0.947) * phi_val)

    # Calculate for unfrozen (frozen_fraction = 0.0)
    lambda_total_hot = calculate_soil_thermal_conductivity(
        thermal_conductivity_saturated_unfrozen=lambda_sat_unfrozen,
        thermal_conductivity_saturated_frozen=lambda_sat_frozen,
        thermal_conductivity_dry_soil_W_per_m_K=thermal_conductivity_dry_soil,
        degree_of_saturation=Sr,
        sand_percentage=sand,
        frozen_fraction=np.float32(0.0),
    )

    # Calculate for frozen (frozen_fraction = 1.0)
    lambda_total_cold = calculate_soil_thermal_conductivity(
        thermal_conductivity_saturated_unfrozen=lambda_sat_unfrozen,
        thermal_conductivity_saturated_frozen=lambda_sat_frozen,
        thermal_conductivity_dry_soil_W_per_m_K=thermal_conductivity_dry_soil,
        degree_of_saturation=Sr,
        sand_percentage=sand,
        frozen_fraction=np.float32(1.0),
    )

    # Saturated frozen conductivity should be higher than unfrozen because lambda_ice > lambda_water
    assert lambda_total_cold > lambda_total_hot

    # Result should be between lambda_dry and lambda_sat
    assert lambda_total_hot > float(thermal_conductivity_dry_soil)
    assert lambda_total_hot < float(lambda_sat_unfrozen) * 1.1


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


def test_apply_rain_heat_advection_reduces_frozen_fraction() -> None:
    """Test that warm liquid input can thaw a partially frozen top layer."""
    solid_heat_capacity_J_per_m2_K = np.float32(1.0e6)
    water_content_m = np.float32(0.02)
    topwater_m = np.float32(0.01)

    latent_heat_areal_J_per_m2 = (
        (water_content_m + topwater_m) * np.float32(1000.0) * np.float32(334000.0)
    )
    initial_enthalpy_J_per_m2 = np.float32(-0.5) * latent_heat_areal_J_per_m2

    initial_temperature_C, initial_frozen_fraction = (
        get_temperature_and_frozen_fraction_from_enthalpy_scalar(
            enthalpy_J_per_m2=initial_enthalpy_J_per_m2,
            solid_heat_capacity_J_per_m2_K=solid_heat_capacity_J_per_m2_K,
            water_content_m=water_content_m,
            topwater_m=topwater_m,
        )
    )

    updated_enthalpy_J_per_m2 = apply_rain_heat_advection(
        soil_enthalpy_top_layer_J_per_m2=initial_enthalpy_J_per_m2,
        liquid_water_input_m=topwater_m,
        rain_temperature_C=np.float32(5.0),
    )
    updated_temperature_C, updated_frozen_fraction = (
        get_temperature_and_frozen_fraction_from_enthalpy_scalar(
            enthalpy_J_per_m2=updated_enthalpy_J_per_m2,
            solid_heat_capacity_J_per_m2_K=solid_heat_capacity_J_per_m2_K,
            water_content_m=water_content_m,
            topwater_m=topwater_m,
        )
    )

    assert initial_temperature_C == np.float32(0.0)
    assert updated_temperature_C == np.float32(0.0)
    assert initial_frozen_fraction > np.float32(0.0)
    assert updated_frozen_fraction < initial_frozen_fraction
    assert updated_frozen_fraction >= np.float32(0.0)


def test_johansen_thermal_conductivity_functions() -> None:
    """Test dry and saturated thermal conductivity functions from Johansen (1975)."""
    # 1. Test Dry Conductivity
    bd = np.array([1.3], dtype=np.float32)
    expected_dry = (0.135 * 1300.0 + 64.7) / (2700.0 - 0.947 * 1300.0)
    res_dry = calculate_thermal_conductivity_dry_soil_johansen_watt_per_meter_kelvin(bd)
    np.testing.assert_allclose(
        res_dry, np.array([expected_dry], dtype=np.float32), rtol=1e-5
    )

    # 2. Test Saturated Conductivity
    lambda_s = np.array([2.5], dtype=np.float32)
    porosity = np.array([0.4], dtype=np.float32)
    f_water = np.float32(
        0.57
    )  # np.float32(THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN)

    expected_sat = (2.5**0.6) * (0.57**0.4)
    res_sat = (
        calculate_thermal_conductivity_saturated_soil_johansen_watt_per_meter_kelvin(
            lambda_s, porosity, f_water
        )
    )
    np.testing.assert_allclose(
        res_sat, np.array([expected_sat], dtype=np.float32), rtol=1e-5
    )

    # Test with ice
    f_ice = np.float32(2.2)  # np.float32(THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN)
    expected_sat_ice = (2.5**0.6) * (2.2**0.4)
    res_sat_ice = (
        calculate_thermal_conductivity_saturated_soil_johansen_watt_per_meter_kelvin(
            lambda_s, porosity, f_ice
        )
    )
    np.testing.assert_allclose(
        res_sat_ice, np.array([expected_sat_ice], dtype=np.float32), rtol=1e-5
    )
