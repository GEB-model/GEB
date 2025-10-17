"""Tests for potential evapotranspiration functions in GEB."""

import math

import numpy as np

from geb.hydrology.landcovers import (
    FOREST,
    GRASSLAND_LIKE,
    NON_PADDY_IRRIGATED,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
)
from geb.hydrology.potential_evapotranspiration import (
    W_per_m2_to_MJ_per_m2_per_hour,
    adjust_wind_speed,
    get_CO2_induced_crop_factor_adustment,
    get_crop_factors_and_root_depths,
    get_net_solar_radiation,
    get_potential_bare_soil_evaporation,
    get_potential_evapotranspiration,
    get_potential_transpiration,
    get_psychrometric_constant,
    get_slope_of_saturation_vapour_pressure_curve,
    get_upwelling_long_wave_radiation,
    get_vapour_pressure,
    get_vapour_pressure_deficit,
    penman_monteith,
)


def test_get_vapour_pressure() -> None:
    """See example 3: https://www.fao.org/4/X0490E/x0490e07.htm."""
    saturated_vapour_pressure = get_vapour_pressure(
        temperature_C=np.float32(15.0),
    )
    assert math.isclose(saturated_vapour_pressure, 1.705, abs_tol=1e-2)
    saturated_vapour_pressure = get_vapour_pressure(temperature_C=np.float32(24.5))
    assert math.isclose(saturated_vapour_pressure, 3.075, abs_tol=1e-2)


def test_get_vapour_pressure_deficit() -> None:
    """See example 18: https://www.fao.org/4/x0490e/x0490e08.htm."""
    vapour_pressure_deficit_kPa = get_vapour_pressure_deficit(
        saturated_vapour_pressure_kPa=np.float32(1.997),
        actual_vapour_pressure_kPa=np.float32(1.409),
    )
    assert math.isclose(a=vapour_pressure_deficit_kPa, b=0.589, abs_tol=0.1)


def test_W_per_m2_to_MJ_per_m2_per_hour() -> None:
    """Test the conversion from W/m^2 to MJ/m^2/hour."""
    W_per_m2 = np.float32(100.0)  # Example value in W/m^2
    MJ_per_m2_per_hour = W_per_m2_to_MJ_per_m2_per_hour(
        solar_radiation_W_per_m2=W_per_m2
    )

    expected_value = W_per_m2 * (3600 * 1e-6)  # Convert W/m^2 to MJ/m^2/hour
    assert math.isclose(MJ_per_m2_per_hour, expected_value, rel_tol=1e-6)


def test_get_upwelling_long_wave_radiation() -> None:
    """Test the upwelling long wave radiation calculation.

    See example 18: https://www.fao.org/4/x0490e/x0490e08.htm.
    """
    tas_C = np.float32(38.0)  # Temperature in Celsius
    rlus_MJ_m2_per_hour = get_upwelling_long_wave_radiation(tas_C)

    expected_value = 1.915  # Expected value in MJ/m^2/hour
    assert math.isclose(rlus_MJ_m2_per_hour, expected_value, rel_tol=1e-2)


def test_get_psychrometric_constant() -> None:
    """Test the psychrometric constant calculation."""
    ps_pa = np.float32(81800.0)  # Example surface pressure in Pascals
    psychrometric_constant_kPa_per_C = get_psychrometric_constant(ps_pa=ps_pa)

    assert math.isclose(psychrometric_constant_kPa_per_C, 0.054, abs_tol=0.01)


def test_get_net_solar_radiation() -> None:
    """Test the net solar radiation calculation."""
    solar_radiation_MJ_per_m2_per_hour = np.float32(
        200.0
    )  # Example solar radiation in W/m^2
    albedo = np.float32(0.23)  # Example albedo for vegetation

    net_solar_radiation_MJ_per_m2_per_hour = get_net_solar_radiation(
        solar_radiation_MJ_per_m2_per_hour, albedo
    )

    assert math.isclose(net_solar_radiation_MJ_per_m2_per_hour, 200 - 46, rel_tol=1e-6)


def test_get_slope_of_saturation_vapour_pressure_curve() -> None:
    """Test the slope of the saturation vapour pressure curve calculation."""
    temperature_C = np.float32(25.0)  # Example temperature in Celsius

    slope_kPa_per_C = get_slope_of_saturation_vapour_pressure_curve(temperature_C)

    expected_value = (
        np.float32(4098.0)
        * get_vapour_pressure(temperature_C=temperature_C)
        / ((temperature_C + np.float32(237.3)) ** 2)
    )
    assert math.isclose(slope_kPa_per_C, expected_value, rel_tol=1e-6)


def test_adjust_wind_speed() -> None:
    """Test the wind speed adjustment function."""
    wind_10m_m_per_s = np.float32(100.0)  # Example wind speed in m/s
    adjusted_wind_speed_m_per_s = adjust_wind_speed(wind_10m_m_per_s=wind_10m_m_per_s)

    assert math.isclose(adjusted_wind_speed_m_per_s, 74.8, rel_tol=1e-6)


def test_penman_monteith_day() -> None:
    """Test the reference evapotranspiration calculation during the day.

    See example 19: https://www.fao.org/4/x0490e/x0490e08.htm
    """
    (
        reference_evapotranspiration_land_mm_per_hour,
        reference_evapotranspiration_water_mm_per_hour,
    ) = penman_monteith(
        net_radiation_land_MJ_per_m2_per_hour=np.float32(1.749),
        net_radiation_water_MJ_per_m2_per_hour=np.float32(1.749),
        soil_heat_flux_MJ_per_m2_per_hour=np.float32(0.175),
        slope_of_saturated_vapour_pressure_curve_kPa_per_C=np.float32(0.358),
        psychrometric_constant_kPa_per_C=np.float32(0.0673),
        wind_2m_m_per_s=np.float32(3.3),
        temperature_C=np.float32(38.0),
        vapour_pressure_deficit_kPa=np.float32(3.180),
    )
    assert math.isclose(
        a=reference_evapotranspiration_land_mm_per_hour, b=0.63, rel_tol=1e-2
    )


def test_penman_monteith_night() -> None:
    """Test the reference evapotranspiration calculation at night.

    See example 19: https://www.fao.org/4/x0490e/x0490e08.htm
    """
    (
        reference_evapotranspiration_land_mm_per_hour,
        reference_evapotranspiration_water_mm_per_hour,
    ) = penman_monteith(
        net_radiation_land_MJ_per_m2_per_hour=np.float32(-0.100),
        net_radiation_water_MJ_per_m2_per_hour=np.float32(-0.100),
        soil_heat_flux_MJ_per_m2_per_hour=np.float32(-0.050),
        slope_of_saturated_vapour_pressure_curve_kPa_per_C=np.float32(0.220),
        psychrometric_constant_kPa_per_C=np.float32(0.0673),
        wind_2m_m_per_s=np.float32(1.9),
        temperature_C=np.float32(28.0),
        vapour_pressure_deficit_kPa=np.float32(0.378),
    )
    assert math.isclose(
        a=reference_evapotranspiration_land_mm_per_hour, b=0.00, abs_tol=1e-2
    )


def test_get_potential_transpiration() -> None:
    """Test the calculation of potential transpiration."""
    potential_transpiration_m = get_potential_transpiration(
        potential_evapotranspiration_m=np.float32(5.0),
        potential_bare_soil_evaporation_m=np.float32(2.0),
    )
    expected_value = np.float32(3.0)
    assert np.isclose(potential_transpiration_m, expected_value, rtol=1e-6)

    potential_transpiration_m = get_potential_transpiration(
        potential_evapotranspiration_m=np.float32(5.0),
        potential_bare_soil_evaporation_m=np.float32(6.0),
    )
    expected_value = np.float32(0.0)
    assert np.isclose(potential_transpiration_m, expected_value, rtol=1e-6)


def test_get_potential_evapotranspiration() -> None:
    """Test the calculation of potential evapotranspiration."""
    potential_evapotranspiration_m = get_potential_evapotranspiration(
        reference_evapotranspiration_grass_m=np.float32(5.0),
        crop_factor=np.float32(1.0),
        CO2_induced_crop_factor_adustment=np.float32(1.0),
    )
    expected_value = np.float32(5.0)
    assert np.isclose(potential_evapotranspiration_m, expected_value, rtol=1e-6)

    elevated_CO2_induced_crop_factor_adustment = get_CO2_induced_crop_factor_adustment(
        2000.0
    )
    potential_evapotranspiration_m_elevated_CO2 = get_potential_evapotranspiration(
        reference_evapotranspiration_grass_m=np.float32(5.0),
        crop_factor=np.float32(1.0),
        CO2_induced_crop_factor_adustment=elevated_CO2_induced_crop_factor_adustment,
    )

    # elevated CO2 should make stomata more efficient, close more and therefore
    # reduce evapotranspiration
    assert potential_evapotranspiration_m_elevated_CO2 < potential_evapotranspiration_m

    potential_evapotranspiration_m = get_potential_evapotranspiration(
        reference_evapotranspiration_grass_m=np.float32(5.0),
        crop_factor=np.float32(0.5),
        CO2_induced_crop_factor_adustment=np.float32(1.0),
    )
    expected_value = np.float32(2.5)
    assert np.isclose(potential_evapotranspiration_m, expected_value, rtol=1e-6)

    potential_evapotranspiration_m = get_potential_evapotranspiration(
        reference_evapotranspiration_grass_m=np.float32(5.0),
        crop_factor=np.float32(1.0),
        CO2_induced_crop_factor_adustment=np.float32(0.5),
    )
    expected_value = np.float32(2.5)
    assert np.isclose(potential_evapotranspiration_m, expected_value, rtol=1e-6)


def test_get_potential_bare_soil_evaporation() -> None:
    """Test the calculation of potential bare soil evaporation."""
    reference_evapotranspiration_grass_m = np.float32(5.0)
    bare_soil_evaporation = get_potential_bare_soil_evaporation(
        reference_evapotranspiration_grass_m,
        sublimation_m=np.float32(0.0),
    )

    expected_value = np.float32(1.0)
    assert np.isclose(bare_soil_evaporation, expected_value, rtol=1e-6)

    bare_soil_evaporation = get_potential_bare_soil_evaporation(
        reference_evapotranspiration_grass_m,
        sublimation_m=np.float32(0.5),
    )
    expected_value = np.float32(0.5)
    assert np.isclose(bare_soil_evaporation, expected_value, rtol=1e-6)

    bare_soil_evaporation = get_potential_bare_soil_evaporation(
        reference_evapotranspiration_grass_m,
        sublimation_m=np.float32(5.0),
    )
    expected_value = np.float32(0.0)
    assert np.isclose(bare_soil_evaporation, expected_value, rtol=1e-6)


def test_get_CO2_induced_crop_factor_adustment() -> None:
    """Test the CO2 induced crop factor adjustment calculation."""
    assert get_CO2_induced_crop_factor_adustment(369.41) == 1.0
    assert get_CO2_induced_crop_factor_adustment(550.0) == 0.95
    assert get_CO2_induced_crop_factor_adustment(550.0 + (550.0 - 369.41) == 0.9)


def test_get_crop_factors_and_root_depths() -> None:
    """Test the calculation of crop factors and root depths."""
    land_use_map = np.array(
        [
            PADDY_IRRIGATED,
            NON_PADDY_IRRIGATED,
            GRASSLAND_LIKE,
            GRASSLAND_LIKE,
            FOREST,
            GRASSLAND_LIKE,
            SEALED,
            OPEN_WATER,
        ],
        dtype=np.int32,
    )
    crop_factor_forest_map = np.array(
        [0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        dtype=np.float32,
    )
    crop_map = np.array([1, 2, 0, 0, -1, -1, -1, -1], dtype=np.int32)
    crop_age_days_map = np.array([10, 30, 85, 95, -1, -1, -1, -1], dtype=np.int32)
    crop_harvest_age_days = np.array(
        [100, 100, 100, 100, -1, -1, -1, -1], dtype=np.int32
    )
    crop_stage_lengths = np.array(
        [[30, 40, 20, 10], [20, 30, 30, 20], [10, 40, 40, 10]],
        dtype=np.int32,
    )
    crop_sub_stage_lengths = np.array(
        [
            [10, 10, 20, 20, 30, 10],
            [10, 10, 10, 20, 40, 10],
            [5, 5, 20, 20, 40, 10],
        ],
        dtype=np.int32,
    )
    crop_factor_per_crop_stage = np.array(
        [[0.7, 1.2, 0.9], [0.6, 1.1, 0.8], [0.5, 1.0, 0.7]],
        dtype=np.float32,
    )
    crop_root_depths = np.array(
        [[1.0, 1.5], [0.5, 1.0], [0.2, 0.1]],
        dtype=np.float32,
    )
    crop_init_root_depth = np.float32(0.2)

    crop_factor, root_depth, crop_sub_stage = get_crop_factors_and_root_depths(
        land_use_map=land_use_map,
        crop_factor_forest_map=crop_factor_forest_map,
        crop_map=crop_map,
        crop_age_days_map=crop_age_days_map,
        crop_harvest_age_days=crop_harvest_age_days,
        crop_stage_lengths=crop_stage_lengths,
        crop_sub_stage_lengths=crop_sub_stage_lengths,
        crop_factor_per_crop_stage=crop_factor_per_crop_stage,
        crop_root_depths=crop_root_depths,
        crop_init_root_depth=crop_init_root_depth,
    )

    # first crop in initial stage, second crop halfway in second stage, third crop fully grown, fourth crop halfway in last stage
    # fifth is forest, sixth is grassland, seventh is sealed, eighth is open water
    expected_crop_factor = np.array(
        [0.6, 0.75, 1.2, 1.05, 1.3, 1.0, np.nan, np.nan], dtype=np.float32
    )
    expected_root_depth = np.array(
        [
            0.28,  # 0.2 + (1.0 - 0.2) * 10 / 100, irrigated, crop 1
            0.2,  # 0.2 + max((0.1 - 0.2), 0) * 30 / 100  irrigated, crop 2, 0.2 because final root depth is smaller than initial
            0.2 + (1.0 - 0.2) * 85 / 100,  # not irrigated, crop 0
            0.2 + (1.0 - 0.2) * 95 / 100,  # not irrigated, crop 0
            2.0,  # forest
            0.1,  # grassland
            np.nan,  # sealed
            np.nan,  # open water
        ],
        dtype=np.float32,
    )
    expected_crop_sub_stage_first_call = np.array(
        [-1, -1, -1, -1, -1, -1, -1, -1], dtype=np.int8
    )

    np.testing.assert_allclose(
        crop_factor,
        expected_crop_factor,
    )
    np.testing.assert_allclose(
        root_depth,
        expected_root_depth,
    )
    np.testing.assert_allclose(
        crop_sub_stage,
        expected_crop_sub_stage_first_call,
    )

    crop_factor, root_depth, crop_sub_stage = get_crop_factors_and_root_depths(
        land_use_map=land_use_map,
        crop_factor_forest_map=crop_factor_forest_map,
        crop_map=crop_map,
        crop_age_days_map=crop_age_days_map,
        crop_harvest_age_days=crop_harvest_age_days,
        crop_stage_lengths=crop_stage_lengths,
        crop_sub_stage_lengths=crop_sub_stage_lengths,
        crop_factor_per_crop_stage=crop_factor_per_crop_stage,
        crop_root_depths=crop_root_depths,
        crop_init_root_depth=crop_init_root_depth,
        get_crop_sub_stage=True,
    )

    expected_crop_sub_stage_second_call = np.array(
        [0, 2, 4, 5, -1, -1, -1, -1], dtype=np.int8
    )

    np.testing.assert_allclose(
        crop_sub_stage,
        expected_crop_sub_stage_second_call,
    )

    # Re-check that crop_factor and root_depth remain the same after second call
    np.testing.assert_allclose(
        crop_factor,
        expected_crop_factor,
    )
    np.testing.assert_allclose(
        root_depth,
        expected_root_depth,
    )
