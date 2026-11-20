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
from geb.hydrology.landsurface.potential_evapotranspiration import (
    FOREST_AERODYNAMIC_RESISTANCE_TIMES_WIND_M,
    GRASS_AERODYNAMIC_RESISTANCE_TIMES_WIND_M,
    MINIMUM_WIND_SPEED_M_PER_S,
    WATER_AERODYNAMIC_RESISTANCE_TIMES_WIND_M,
    HourlyWeather,
    adjust_wind_speed_log_profile,
    get_active_leaf_fraction,
    get_aerodynamic_resistance,
    get_CO2_induced_crop_factor_adustment,
    get_crop_coefficient_climate_adjustment,
    get_crop_factors_and_root_depths_and_lai,
    get_daily_wind_and_minimum_relative_humidity,
    get_dual_crop_coefficients,
    get_expected_heat_units,
    get_hourly_weather,
    get_interception_capacity,
    get_isothermal_net_radiation,
    get_maximum_crop_coefficient,
    get_minimum_temperature_index,
    get_potential_evapotranspiration,
    get_psychrometric_constant,
    get_radiative_resistance,
    get_reference_evapotranspiration,
    get_slope_of_saturation_vapour_pressure_curve,
    get_vapour_pressure,
    get_vapour_pressure_deficit,
    get_wet_forest_canopy_evaporation,
    get_wet_soil_evaporation_coefficient,
    penman_monteith,
    update_canopy_temperature_acclimation,
    update_crop_heat_units,
)


def weather_of(
    tas_C: float = 20.0,
    dewpoint_tas_C: float = 12.0,
    ps_pa: float = 100000.0,
    rsds_W_per_m2: float = 0.0,
    rlds_W_per_m2: float = 330.0,
    wind_10m_m_per_s: float = 3.0,
) -> HourlyWeather:
    """Weather record of one hour for the tests.

    Returns:
        The weather record.
    """
    return get_hourly_weather(
        np.float32(tas_C),
        np.float32(dewpoint_tas_C),
        np.float32(ps_pa),
        np.float32(rsds_W_per_m2),
        np.float32(rlds_W_per_m2),
        np.float32(wind_10m_m_per_s),
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


def test_get_psychrometric_constant() -> None:
    """Test the psychrometric constant calculation."""
    ps_pa = np.float32(81800.0)  # Example surface pressure in Pascals
    psychrometric_constant_kPa_per_C = get_psychrometric_constant(ps_pa=ps_pa)

    assert math.isclose(psychrometric_constant_kPa_per_C, 0.054, abs_tol=0.01)


def test_get_slope_of_saturation_vapour_pressure_curve() -> None:
    """FAO-56 eq. 13 from the saturation vapour pressure (Table 2.4: 0.189 at 25 °C)."""
    temperature_C = np.float32(25.0)
    slope_kPa_per_C = get_slope_of_saturation_vapour_pressure_curve(
        temperature_C, get_vapour_pressure(temperature_C)
    )
    assert math.isclose(slope_kPa_per_C, 0.189, abs_tol=1e-3)


def test_adjust_wind_speed_log_profile() -> None:
    """Test the wind speed adjustment function.

    See Example 14: https://www.fao.org/4/x0490e/x0490e07.htm#wind%20profile%20relationship
        u2 = uz * 4.87 / ln(67.8 * z - 5.42)

    where uz is the wind speed at height z, and u2 is the wind speed at 2 m, with example
    values of uz = 3.2 m/s at z = 10 m, which should yield u2 = 2.4 m/s.

    Derivation of FAO 56 Equation 47 from the general logarithmic profile used here:

    The code uses:
        u_2 = u_z * (ln((z_2 - d) / z_0) / ln((z_z - d) / z_0))

    FAO 56 assumes for a standard crop:
    - h (height) = 0.12 m
    - d (displacement height) = 0.67 * h = 0.08 m
    - z_0 (roughness length) = 0.123 * h = 0.01476 m

    Numerator term with z_2 = 2.0 m:
        ln((2.0 - 0.08) / 0.01476) approx 4.87

    Denominator term inner part:
        (z_z - 0.08) / 0.01476 = (1/0.01476) * z_z - (0.08/0.01476)
                           approx 67.8 * z_z - 5.42

    Putting it together yields Eq. 47:
        u_2 = u_z * 4.87 / ln(67.8 * z_z - 5.42)
    """
    wind_10m_m_per_s = np.float32(3.2)

    adjusted_wind_speed_m_per_s = adjust_wind_speed_log_profile(
        wind_measurement_height_m_per_s=wind_10m_m_per_s,
        measurement_height_m=np.float32(10.0),
        target_height_m=np.float32(2.0),
        canopy_height_m=np.float32(0.12),
    )

    assert math.isclose(adjusted_wind_speed_m_per_s, 2.4, abs_tol=0.05)


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
        CO2_induced_crop_factor_adustment=np.float32(
            elevated_CO2_induced_crop_factor_adustment
        ),
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


def test_get_CO2_induced_crop_factor_adustment() -> None:
    """Test the CO2 induced crop factor adjustment calculation."""
    assert get_CO2_induced_crop_factor_adustment(369.41) == 1.0
    assert get_CO2_induced_crop_factor_adustment(550.0) == 0.95
    assert get_CO2_induced_crop_factor_adustment(550.0 + (550.0 - 369.41) == 0.9)


def test_get_crop_factors_and_root_depths_and_lai() -> None:
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
    leaf_area_index_forest = np.array(
        [0.8, 1.0, 1.47, 1.785, 2.197, 3.77, 10.0, 10.0],
        dtype=np.float32,
    )
    leaf_area_index_grassland_like = np.array(
        [0.0, 0.0, 10.0, 2.7, 0.0, 2.3, 1.8, 0.0],
        dtype=np.float32,
    )
    crop_map = np.array([1, 2, 0, 0, -1, -1, -1, -1], dtype=np.int32)
    crop_progress = np.array(
        [0.10, 0.30, 0.85, 0.95, np.nan, np.nan, np.nan, np.nan], dtype=np.float32
    )
    land_owners = np.array([0, 1, 2, 3, -1, -1, -1, -1], dtype=np.int32)
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

    (
        crop_factor,
        basal_crop_factor,
        maximum_crop_factor,
        root_depth,
        crop_sub_stage,
        leaf_area_index,
        interception_capacity_m,
    ) = get_crop_factors_and_root_depths_and_lai(
        land_use_map=land_use_map,
        leaf_area_index_forest=leaf_area_index_forest,
        leaf_area_index_grassland_like=leaf_area_index_grassland_like,
        leaf_area_index_forest_evergreen=np.zeros_like(
            np.ones_like(leaf_area_index_forest)
        ),
        evergreen_activity=np.ones_like(leaf_area_index_forest),
        wind_2m_m_per_s=np.full_like(leaf_area_index_forest, 2.0),
        minimum_relative_humidity_percent=np.full_like(leaf_area_index_forest, 45.0),
        crop_map=crop_map,
        crop_progress=crop_progress,
        land_owners=land_owners,
        crop_stage_lengths=crop_stage_lengths,
        crop_sub_stage_lengths=crop_sub_stage_lengths,
        crop_factor_per_crop_stage=crop_factor_per_crop_stage,
        crop_root_depths=crop_root_depths,
        crop_init_root_depth=crop_init_root_depth,
        grassland_root_depth_m=np.float32(0.3),
    )

    # first crop in initial stage, second crop halfway in second stage, third crop fully grown, fourth crop halfway in last stage
    # fifth is forest, sixth is grassland, seventh is sealed, eighth is open water
    def dual(lai: float, basal_full: float) -> tuple[float, float]:
        # FAO-56 dual crop coefficient for a wet soil surface (Kcb + Ke) and Kcb
        cover = 1.0 - np.exp(-0.7 * lai)
        basal = 0.15 + (basal_full - 0.15) * cover
        maximum = max(1.2, basal + 0.05)
        return basal + min(maximum - basal, (1.0 - cover) * maximum), basal

    forest_kc, forest_kcb = dual(2.197, 0.95)
    grass_kc, grass_kcb = dual(2.3, 0.95)
    # sealed land: grass on the unpaved 65 %
    sealed_kc, sealed_kcb = (0.65 * value for value in dual(1.8, 0.95))
    crop_cover = 1.0 - np.exp(-0.7 * np.array([0.3, 2.0, 3.5, 2.5]))
    # FAO-56 Table 18: Kcb = 0.15 at the start of an annual crop, Kc - 0.05 later.
    # crop 1 initial; crop 2 halfway development 0.15 -> 1.0 - 0.05; crop 0 mid
    # season 1.2 - 0.05; crop 0 halfway late season 1.15 -> 0.9 - 0.05
    crop_basal = np.array([0.15, 0.55, 1.15, 1.0])
    crop_max = np.maximum(1.2, crop_basal + 0.05)
    crop_soil = np.minimum(crop_max - crop_basal, (1 - crop_cover) * crop_max)
    crop_wet = crop_basal + crop_soil
    np.testing.assert_allclose(
        maximum_crop_factor,
        [*crop_max, 1.2, 1.2, 0.65 * 1.2, 0.0],
        rtol=1e-5,
    )
    expected_crop_factor = np.array(
        [*crop_wet, forest_kc, grass_kc, sealed_kc, 0.0], dtype=np.float32
    )
    expected_basal_crop_factor = np.array(
        [*crop_basal, forest_kcb, grass_kcb, sealed_kcb, 0.0], dtype=np.float32
    )
    expected_root_depth = np.array(
        [
            # roots reach their maximum depth at the end of the development
            # stage (crop 1: 20 + 30 %) and keep it afterwards
            0.2 + (1.0 - 0.2) * 10 / 50,  # irrigated, crop 1
            0.2,  # irrigated, crop 2, maximum root depth below the initial depth
            1.0,  # not irrigated, crop 0, mid-season
            1.0,  # not irrigated, crop 0, late season
            4.0,  # forest, roots into the regolith layer
            0.3,  # grassland
            0.3,  # sealed: grass on the unpaved part
            0.0,  # open water
        ],
        dtype=np.float32,
    )
    expected_crop_sub_stage_first_call = np.array(
        [-1, -1, -1, -1, -1, -1, -1, -1], dtype=np.int8
    )
    # Dynamic LAI follows 4-stage canopy progression for crops, lookup for forest/grassland, 0 for sealed/water
    expected_leaf_area_index = np.array(
        [
            0.3,  # crop 1, initial stage (10%): 0.1 + (0.5 - 0.1) * 10 / 20
            2.0,  # crop 2, development stage (30%): 0.5 + (3.5 - 0.5) * 20 / 40
            3.5,  # crop 0, mid-season plateau (85%): 3.5
            2.5,  # crop 0, late season (95%): 3.5 - (3.5 - 1.5) * 5 / 10
            2.197,  # forest: leaf_area_index_forest[4]
            2.3,  # grassland: leaf_area_index_grassland_like[5]
            0.65 * 1.8,  # sealed: grass on the unpaved 65 %
            0.0,  # open water
        ],
        dtype=np.float32,
    )

    def calc_cap_m(lai: float) -> float:
        cover = 1.0 - np.exp(-0.7 * lai)
        return (0.935 * cover + 0.498 * lai - 0.00575 * (lai**2)) / 1000.0

    expected_interception_capacity_m = np.array(
        [
            calc_cap_m(0.3),
            calc_cap_m(2.0),
            calc_cap_m(3.5),
            calc_cap_m(2.5),
            calc_cap_m(2.197),
            calc_cap_m(2.3),
            0.65 * calc_cap_m(1.8),  # sealed: grass on the unpaved 65 %
            0.0,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        crop_factor,
        expected_crop_factor,
        equal_nan=True,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        basal_crop_factor, expected_basal_crop_factor, rtol=1e-5, atol=1e-6
    )
    # wet-surface potential is never below the transpiration part
    assert (crop_factor >= basal_crop_factor).all()
    np.testing.assert_allclose(
        root_depth,
        expected_root_depth,
        equal_nan=True,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        crop_sub_stage,
        expected_crop_sub_stage_first_call,
    )
    np.testing.assert_allclose(
        leaf_area_index,
        expected_leaf_area_index,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        interception_capacity_m,
        expected_interception_capacity_m,
        atol=1e-6,
    )

    (
        crop_factor,
        basal_crop_factor,
        maximum_crop_factor,
        root_depth,
        crop_sub_stage,
        leaf_area_index,
        interception_capacity_m,
    ) = get_crop_factors_and_root_depths_and_lai(
        land_use_map=land_use_map,
        leaf_area_index_forest=leaf_area_index_forest,
        leaf_area_index_grassland_like=leaf_area_index_grassland_like,
        leaf_area_index_forest_evergreen=np.zeros_like(
            np.ones_like(leaf_area_index_forest)
        ),
        evergreen_activity=np.ones_like(leaf_area_index_forest),
        wind_2m_m_per_s=np.full_like(leaf_area_index_forest, 2.0),
        minimum_relative_humidity_percent=np.full_like(leaf_area_index_forest, 45.0),
        crop_map=crop_map,
        crop_progress=crop_progress,
        land_owners=land_owners,
        crop_stage_lengths=crop_stage_lengths,
        crop_sub_stage_lengths=crop_sub_stage_lengths,
        crop_factor_per_crop_stage=crop_factor_per_crop_stage,
        crop_root_depths=crop_root_depths,
        crop_init_root_depth=crop_init_root_depth,
        get_crop_sub_stage=True,
        grassland_root_depth_m=np.float32(0.3),
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
        equal_nan=True,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        root_depth,
        expected_root_depth,
        equal_nan=True,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        leaf_area_index,
        expected_leaf_area_index,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        interception_capacity_m,
        expected_interception_capacity_m,
        atol=1e-6,
    )


def test_get_crop_factors_and_root_depths_and_lai_non_cropland() -> None:
    """Test non-cropland (fallow/unplanted paddy, grassland, forest) LAI and interception capacity."""
    land_use_map = np.array(
        [
            GRASSLAND_LIKE,
            FOREST,
        ],
        dtype=np.int32,
    )
    leaf_area_index_forest = np.array([0.0, 10.0], dtype=np.float32)
    leaf_area_index_grassland_like = np.array([1.8, 0.0], dtype=np.float32)
    crop_map = np.array([-1, -1], dtype=np.int32)
    crop_progress = np.array([np.nan, np.nan], dtype=np.float32)
    land_owners = np.array([-1, -1], dtype=np.int32)
    crop_stage_lengths = np.array([[25, 25, 25, 25]], dtype=np.int32)
    crop_sub_stage_lengths = np.array([[10, 15, 25, 25, 15, 10]], dtype=np.int32)
    crop_factor_per_crop_stage = np.array([[0.5, 1.0, 0.7]], dtype=np.float32)
    crop_root_depths = np.array([[0.8, 1.2]], dtype=np.float32)

    (
        crop_factor,
        basal_crop_factor,
        maximum_crop_factor,
        root_depth,
        crop_sub_stage,
        leaf_area_index,
        interception_capacity_m,
    ) = get_crop_factors_and_root_depths_and_lai(
        land_use_map=land_use_map,
        leaf_area_index_forest=leaf_area_index_forest,
        leaf_area_index_grassland_like=leaf_area_index_grassland_like,
        leaf_area_index_forest_evergreen=np.zeros_like(
            np.ones_like(leaf_area_index_forest)
        ),
        evergreen_activity=np.ones_like(leaf_area_index_forest),
        wind_2m_m_per_s=np.full_like(leaf_area_index_forest, 2.0),
        minimum_relative_humidity_percent=np.full_like(leaf_area_index_forest, 45.0),
        crop_map=crop_map,
        crop_progress=crop_progress,
        land_owners=land_owners,
        crop_stage_lengths=crop_stage_lengths,
        crop_sub_stage_lengths=crop_sub_stage_lengths,
        crop_factor_per_crop_stage=crop_factor_per_crop_stage,
        crop_root_depths=crop_root_depths,
        grassland_root_depth_m=np.float32(0.3),
    )

    # Grassland -> uses grassland LAI (1.8), capacity from Von Hoyningen-Huene formula
    expected_cap_grass = (
        0.935 * (1 - np.exp(-0.7 * 1.8)) + 0.498 * 1.8 - 0.00575 * (1.8**2)
    ) / 1000.0
    assert math.isclose(leaf_area_index[0], 1.8, abs_tol=1e-6)
    assert math.isclose(interception_capacity_m[0], expected_cap_grass, abs_tol=1e-6)

    # Forest with high LAI (10.0) -> capacity from Von Hoyningen-Huene formula
    expected_cap_forest = (
        0.935 * (1 - np.exp(-7.0)) + 0.498 * 10.0 - 0.00575 * (10.0**2)
    ) / 1000.0
    assert math.isclose(leaf_area_index[1], 10.0, abs_tol=1e-6)
    assert math.isclose(interception_capacity_m[1], expected_cap_forest, abs_tol=1e-6)


def test_get_crop_factors_farm_field_without_crop() -> None:
    """A farm field without a crop is bare soil; grassland keeps its leaf area."""
    land_use_map = np.array([GRASSLAND_LIKE, GRASSLAND_LIKE], dtype=np.int32)
    leaf_area_index = np.array([2.5, 2.5], dtype=np.float32)
    (
        crop_factor,
        basal_crop_factor,
        maximum_crop_factor,
        root_depth,
        _,
        leaf_area_index_out,
        interception_capacity_m,
    ) = get_crop_factors_and_root_depths_and_lai(
        land_use_map=land_use_map,
        leaf_area_index_forest=leaf_area_index,
        leaf_area_index_grassland_like=leaf_area_index,
        leaf_area_index_forest_evergreen=np.zeros_like(np.ones_like(leaf_area_index)),
        evergreen_activity=np.ones_like(leaf_area_index),
        wind_2m_m_per_s=np.full_like(leaf_area_index, 2.0),
        minimum_relative_humidity_percent=np.full_like(leaf_area_index, 45.0),
        crop_map=np.array([-1, -1], dtype=np.int32),
        crop_progress=np.array([np.nan, np.nan], dtype=np.float32),
        land_owners=np.array([7, -1], dtype=np.int32),
        crop_stage_lengths=np.array([[25, 25, 25, 25]], dtype=np.int32),
        crop_sub_stage_lengths=np.array([[10, 15, 25, 25, 15, 10]], dtype=np.int32),
        crop_factor_per_crop_stage=np.array([[0.5, 1.0, 0.7]], dtype=np.float32),
        crop_root_depths=np.array([[0.8, 1.2]], dtype=np.float32),
        grassland_root_depth_m=np.float32(0.3),
    )
    # field: no leaf area, minimum basal coefficient, wet soil up to the maximum
    assert leaf_area_index_out[0] == 0.0
    assert interception_capacity_m[0] == 0.0
    assert math.isclose(basal_crop_factor[0], 0.15, abs_tol=1e-6)
    assert math.isclose(crop_factor[0], 1.2, abs_tol=1e-6)
    # grassland: leaf area of the cell
    assert math.isclose(leaf_area_index_out[1], 2.5, abs_tol=1e-6)
    assert basal_crop_factor[1] > 0.7


def test_get_expected_heat_units() -> None:
    """Expected heat units: linear when warm, positive when the mean is below base."""
    assert math.isclose(
        get_expected_heat_units(np.float32(15.0), np.float32(3.0), np.float32(5.0)),
        10.0,
        abs_tol=1e-2,
    )
    # mean at the base temperature: std / sqrt(2 pi)
    assert math.isclose(
        get_expected_heat_units(np.float32(5.0), np.float32(3.0), np.float32(5.0)),
        3.0 / math.sqrt(2.0 * math.pi),
        abs_tol=1e-5,
    )
    assert (
        get_expected_heat_units(np.float32(-10.0), np.float32(3.0), np.float32(5.0))
        < 1e-3
    )
    assert (
        get_expected_heat_units(np.float32(2.0), np.float32(0.0), np.float32(5.0))
        == 0.0
    )


def test_update_crop_heat_units() -> None:
    """Winter crops wait for spring; progress reaches 1 at harvest in a normal year."""
    # cold first half of the year (0 C), warm second half (20 C), no spread
    dekad_per_day_of_year = (np.arange(366) * 36 // 366).astype(np.int32)
    dekadal_mean_C = np.where(np.arange(36) < 18, 0.0, 20.0).astype(np.float32)[
        :, np.newaxis
    ]
    dekadal_std_C = np.zeros((36, 1), dtype=np.float32)
    crop_map = np.array([0, -1], dtype=np.int32)
    crop_harvest_age_days = np.array([300, -1], dtype=np.int32)
    heat_units = np.full(2, np.nan, dtype=np.float32)
    heat_units_at_harvest = np.full(2, np.nan, dtype=np.float32)

    progress_per_day = []
    for age in range(1, 301):
        day_of_year = age - 1  # sown on 1 January
        temperature_C = dekadal_mean_C[dekad_per_day_of_year[day_of_year], 0]
        progress = update_crop_heat_units(
            crop_map=crop_map,
            crop_age_days_map=np.array([age, -1], dtype=np.int32),
            crop_harvest_age_days=crop_harvest_age_days,
            crop_heat_units_C_day=heat_units,
            crop_heat_units_at_harvest_C_day=heat_units_at_harvest,
            crop_base_temperature_C=np.array([5.0], dtype=np.float32),
            mean_daily_air_temperature_C=np.full(2, temperature_C, dtype=np.float32),
            HRU_to_grid=np.array([0, 0], dtype=np.int32),
            air_temperature_dekadal_mean_C=dekadal_mean_C,
            air_temperature_dekadal_standard_deviation_C=dekadal_std_C,
            dekad_per_day_of_year=dekad_per_day_of_year,
            day_of_year=day_of_year,
        )
        progress_per_day.append(progress[0])
        assert np.isnan(progress[1])

    # 183 cold days without progress, then 117 warm days of 15 C day each
    assert math.isclose(heat_units_at_harvest[0], 117 * 15.0, rel_tol=1e-5)
    assert progress_per_day[150] == 0.0
    assert math.isclose(progress_per_day[183 + 58], 59 / 117, abs_tol=1e-5)
    assert math.isclose(progress_per_day[-1], 1.0, abs_tol=1e-5)

    # a field first met in the middle of its season starts at the expected heat units
    heat_units[:] = np.nan
    heat_units_at_harvest[:] = np.nan
    progress = update_crop_heat_units(
        crop_map=crop_map,
        crop_age_days_map=np.array([242, -1], dtype=np.int32),
        crop_harvest_age_days=crop_harvest_age_days,
        crop_heat_units_C_day=heat_units,
        crop_heat_units_at_harvest_C_day=heat_units_at_harvest,
        crop_base_temperature_C=np.array([5.0], dtype=np.float32),
        mean_daily_air_temperature_C=np.full(2, 20.0, dtype=np.float32),
        HRU_to_grid=np.array([0, 0], dtype=np.int32),
        air_temperature_dekadal_mean_C=dekadal_mean_C,
        air_temperature_dekadal_standard_deviation_C=dekadal_std_C,
        dekad_per_day_of_year=dekad_per_day_of_year,
        day_of_year=241,
    )
    assert math.isclose(progress[0], 59 / 117, abs_tol=1e-5)

    # harvested field: state is cleared
    update_crop_heat_units(
        crop_map=np.array([-1, -1], dtype=np.int32),
        crop_age_days_map=np.array([-1, -1], dtype=np.int32),
        crop_harvest_age_days=np.array([-1, -1], dtype=np.int32),
        crop_heat_units_C_day=heat_units,
        crop_heat_units_at_harvest_C_day=heat_units_at_harvest,
        crop_base_temperature_C=np.array([5.0], dtype=np.float32),
        mean_daily_air_temperature_C=np.full(2, 20.0, dtype=np.float32),
        HRU_to_grid=np.array([0, 0], dtype=np.int32),
        air_temperature_dekadal_mean_C=dekadal_mean_C,
        air_temperature_dekadal_standard_deviation_C=dekadal_std_C,
        dekad_per_day_of_year=dekad_per_day_of_year,
        day_of_year=300,
    )
    assert np.isnan(heat_units).all() and np.isnan(heat_units_at_harvest).all()


def test_get_wet_forest_canopy_evaporation() -> None:
    """A wet forest canopy evaporates faster than the grass/water reference, mostly via VPD."""
    # Rainy, cloudy daytime hour: little radiation, small vapour pressure deficit.
    kwargs = dict(
        tas_C=10.0,
        dewpoint_tas_C=9.0,
        ps_pa=97000.0,
        rlds_W_per_m2=330.0,
        rsds_W_per_m2=50.0,
        wind_10m_m_per_s=4.0,
    )
    forest = get_wet_forest_canopy_evaporation(weather_of(**kwargs))
    _, _, water = get_reference_evapotranspiration(
        weather_of(**kwargs), np.float32(0.0)
    )
    assert forest > np.float32(0.0)
    assert forest > 2 * max(water, np.float32(0.0))
    # Typical wet-canopy rates during rain are a few tenths of a mm per hour.
    assert np.float32(0.05e-3) < forest < np.float32(0.6e-3)

    # Saturated air without radiation: no evaporation, never negative.
    still = dict(kwargs, dewpoint_tas_C=10.0, rsds_W_per_m2=0.0)
    assert get_wet_forest_canopy_evaporation(weather_of(**still)) >= np.float32(0.0)

    # More wind means lower aerodynamic resistance and faster evaporation.
    windy = dict(kwargs, wind_10m_m_per_s=8.0)
    assert get_wet_forest_canopy_evaporation(weather_of(**windy)) > forest


def test_minimum_temperature_index() -> None:
    """Jolly et al. (2005) eq. 1: 0 at or below -2 degC, 1 at or above 5 degC."""
    index = get_minimum_temperature_index(
        np.array([-10.0, -2.0, 1.5, 5.0, 25.0], dtype=np.float32)
    )
    np.testing.assert_allclose(index, [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-6)


def test_canopy_acclimation_is_21_day_mean_of_daily_index() -> None:
    """The daily index is averaged (not the temperature), over the last 21 days."""
    record = np.full((21, 2), np.nan, dtype=np.float32)
    # first day: only that day counts
    factor = update_canopy_temperature_acclimation(
        record, np.array([1.5, -10.0], dtype=np.float32)
    )
    np.testing.assert_allclose(factor, [0.5, 0.0], atol=1e-6)
    # a very cold and a very warm day: mean of the indices (0 and 1), while the
    # index of the mean temperature (5 degC) would be 1
    record[:] = np.nan
    update_canopy_temperature_acclimation(
        record, np.array([-20.0, 0.0], dtype=np.float32)
    )
    factor = update_canopy_temperature_acclimation(
        record, np.array([30.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(factor[0], 0.5, atol=1e-6)
    # after 30 warm days only the last 21 count
    for _ in range(30):
        factor = update_canopy_temperature_acclimation(
            record, np.array([10.0, 10.0], dtype=np.float32)
        )
    np.testing.assert_allclose(factor, [1.0, 1.0], atol=1e-6)


def test_crop_basal_coefficients_from_single_crop_coefficients() -> None:
    """FAO-56 Table 18: annual crops start at Kcb = 0.15; perennials keep Kc - 0.05."""
    land_use_map = np.array([NON_PADDY_IRRIGATED, NON_PADDY_IRRIGATED], dtype=np.int32)
    zeros = np.zeros(2, dtype=np.float32)
    _, basal_crop_factor, *_ = get_crop_factors_and_root_depths_and_lai(
        land_use_map=land_use_map,
        leaf_area_index_forest=zeros,
        leaf_area_index_grassland_like=zeros,
        leaf_area_index_forest_evergreen=np.zeros_like(np.ones(2, dtype=np.float32)),
        evergreen_activity=np.ones(2, dtype=np.float32),
        wind_2m_m_per_s=np.full(2, 2.0, dtype=np.float32),
        minimum_relative_humidity_percent=np.full(2, 45.0, dtype=np.float32),
        crop_map=np.array([0, 1], dtype=np.int32),
        crop_progress=np.array([0.05, 0.05], dtype=np.float32),
        land_owners=np.array([0, 1], dtype=np.int32),
        crop_stage_lengths=np.array(
            [[25, 25, 25, 25], [25, 25, 25, 25]], dtype=np.int32
        ),
        crop_sub_stage_lengths=np.array(
            [[10, 15, 25, 25, 15, 10], [10, 15, 25, 25, 15, 10]], dtype=np.int32
        ),
        # annual crop (wheat) and perennial (citrus, FAO-56 Table 12)
        crop_factor_per_crop_stage=np.array(
            [[0.4, 1.15, 0.3], [0.8, 0.8, 0.8]], dtype=np.float32
        ),
        crop_root_depths=np.array([[0.8, 1.2], [0.8, 1.2]], dtype=np.float32),
        grassland_root_depth_m=np.float32(0.3),
    )
    np.testing.assert_allclose(basal_crop_factor, [0.15, 0.75], rtol=1e-5)


def test_penman_monteith_fao_example_19() -> None:
    """Without the radiative path the equation is FAO-56 eq. 53 (hourly grass).

    See example 19: https://www.fao.org/4/x0490e/x0490e08.htm. FAO-56 gives 0.63 mm
    in the day hour and 0.0 mm in the night hour.
    """

    def fao(
        net_radiation_MJ: float,
        soil_heat_flux_MJ: float,
        slope: float,
        temperature_C: float,
        wind_2m: float,
        deficit_kPa: float,
    ) -> float:
        pressure_pa = 0.0673 / 0.665e-3 * 1000.0
        density = pressure_pa / (287.0 * 1.01 * (temperature_C + 273.15))
        return float(
            penman_monteith(
                available_energy_W_per_m2=np.float32(
                    (net_radiation_MJ - soil_heat_flux_MJ) / 0.0036
                ),
                vapour_pressure_deficit_kPa=np.float32(deficit_kPa),
                slope_kPa_per_C=np.float32(slope),
                psychrometric_constant_kPa_per_C=np.float32(0.0673),
                volumetric_heat_capacity_air_J_per_m3_K=np.float32(density * 1013.0),
                aerodynamic_resistance_s_per_m=np.float32(208.0 / wind_2m),
                surface_resistance_s_per_m=np.float32(70.0),
                radiative_resistance_s_per_m=np.float32(np.inf),
            )
        )

    assert math.isclose(
        fao(1.749, 0.175, 0.358, 38.0, 3.3, 3.180) * 1000, 0.63, rel_tol=1e-2
    )
    assert math.isclose(
        fao(-0.100, -0.050, 0.220, 28.0, 1.9, 0.378) * 1000, 0.0, abs_tol=1e-2
    )


def test_grass_aerodynamic_resistance_matches_fao() -> None:
    """With the 10 m wind and 2 m air, r_a equals the FAO-56 value 208 / u2."""
    for wind_10m in (1.0, 3.2, 8.0):
        wind_2m = adjust_wind_speed_log_profile(
            np.float32(wind_10m), np.float32(10.0), np.float32(2.0), np.float32(0.12)
        )
        assert math.isclose(
            GRASS_AERODYNAMIC_RESISTANCE_TIMES_WIND_M / wind_10m,
            208.0 / wind_2m,
            rel_tol=1e-2,
        )
    # the constants are those of the general formula at 1 m/s
    displacement_m = np.float32(2.0 / 3.0 * 0.12)
    assert math.isclose(
        GRASS_AERODYNAMIC_RESISTANCE_TIMES_WIND_M,
        get_aerodynamic_resistance(
            np.float32(1.0),
            np.float32(10.0) - displacement_m,
            np.float32(2.0) - displacement_m,
            np.float32(0.123 * 0.12),
            np.float32(0.1 * 0.123 * 0.12),
        ),
        rel_tol=1e-5,
    )
    # calm air: the wind is limited to u2 = 0.5 m/s over grass (FAO-56 ch. 3)
    assert weather_of(wind_10m_m_per_s=0.0).wind_10m_m_per_s == (
        MINIMUM_WIND_SPEED_M_PER_S
    )
    assert math.isclose(
        adjust_wind_speed_log_profile(
            MINIMUM_WIND_SPEED_M_PER_S,
            np.float32(10.0),
            np.float32(2.0),
            np.float32(0.12),
        ),
        0.5,
        rel_tol=1e-5,
    )


def test_water_is_smoother_than_grass() -> None:
    """Open water has a larger aerodynamic resistance than grass, a forest a smaller one."""
    wind = 3.0
    water = WATER_AERODYNAMIC_RESISTANCE_TIMES_WIND_M / wind
    assert math.isclose(
        water,
        get_aerodynamic_resistance(
            np.float32(wind),
            np.float32(10.0),
            np.float32(2.0),
            np.float32(0.001),
            np.float32(0.001),
        ),
        rel_tol=1e-5,
    )
    assert (
        FOREST_AERODYNAMIC_RESISTANCE_TIMES_WIND_M
        < GRASS_AERODYNAMIC_RESISTANCE_TIMES_WIND_M
        < WATER_AERODYNAMIC_RESISTANCE_TIMES_WIND_M
    )
    # Penman (1948) wind function 2.6 (1 + 0.54 u2) mm/day/kPa corresponds to about
    # 110 s/m at this wind speed
    assert 90.0 < water < 180.0
    # forest: about 10 s/m at 3 m/s
    assert 8.0 < FOREST_AERODYNAMIC_RESISTANCE_TIMES_WIND_M / wind < 12.0


def test_radiative_resistance() -> None:
    """About 200 s/m at 20 °C (Monteith & Unsworth, 2013)."""
    resistance = get_radiative_resistance(
        weather_of(tas_C=20.0, ps_pa=101325.0), np.float32(0.98)
    )
    assert 190.0 < resistance < 230.0


def test_hourly_weather() -> None:
    """Shared weather terms against their definitions."""
    weather = weather_of(tas_C=20.0, dewpoint_tas_C=10.0, ps_pa=101325.0)
    saturated = get_vapour_pressure(np.float32(20.0))
    actual = get_vapour_pressure(np.float32(10.0))
    assert math.isclose(weather.actual_vapour_pressure_kPa, actual, rel_tol=1e-6)
    assert math.isclose(
        weather.vapour_pressure_deficit_kPa, saturated - actual, rel_tol=1e-5
    )
    assert math.isclose(
        weather.relative_humidity_percent, 100 * actual / saturated, rel_tol=1e-5
    )
    # FAO-56 Table 2.4: 0.145 kPa/°C at 20 °C
    assert math.isclose(weather.slope_kPa_per_C, 0.145, abs_tol=1e-3)
    assert math.isclose(
        weather.psychrometric_constant_kPa_per_C,
        get_psychrometric_constant(np.float32(101325.0)),
        rel_tol=1e-6,
    )
    emission = 5.670374e-8 * 293.15**4
    assert math.isclose(weather.black_body_emission_W_per_m2, emission, rel_tol=1e-5)
    assert math.isclose(
        weather.black_body_radiative_conductance_W_per_m2_K,
        4 * emission / 293.15,
        rel_tol=1e-5,
    )
    # supersaturated air (dew point above air temperature) is limited to 100 %
    assert weather_of(tas_C=5.0, dewpoint_tas_C=6.0).relative_humidity_percent == 100.0


def test_isothermal_net_radiation() -> None:
    """A black surface at air temperature under air at the same temperature has Rn = 0."""
    emitted = 5.670374e-8 * (15.0 + 273.15) ** 4
    assert math.isclose(
        get_isothermal_net_radiation(
            weather_of(tas_C=15.0, rsds_W_per_m2=0.0, rlds_W_per_m2=emitted),
            np.float32(0.23),
            np.float32(1.0),
        ),
        0.0,
        abs_tol=1e-2,
    )
    net = get_isothermal_net_radiation(
        weather_of(tas_C=15.0, rsds_W_per_m2=500.0, rlds_W_per_m2=300.0),
        np.float32(0.23),
        np.float32(0.98),
    )
    assert math.isclose(net, 0.77 * 500.0 + 0.98 * (300.0 - emitted), rel_tol=1e-4)


def test_reference_evapotranspiration_is_fao_56() -> None:
    """The grass reference is FAO-56 eq. 53: no radiative path, r_a = 208 / u2."""
    weather = weather_of(
        tas_C=25.0, dewpoint_tas_C=12.0, rsds_W_per_m2=600.0, wind_10m_m_per_s=3.0
    )
    soil_heat_flux = np.float32(30.0)
    reference, _, _ = get_reference_evapotranspiration(weather, soil_heat_flux)
    wind_2m = adjust_wind_speed_log_profile(
        np.float32(3.0), np.float32(10.0), np.float32(2.0), np.float32(0.12)
    )
    available_energy = (
        get_isothermal_net_radiation(weather, np.float32(0.23), np.float32(0.98))
        - soil_heat_flux
    )
    expected = penman_monteith(
        available_energy_W_per_m2=available_energy,
        vapour_pressure_deficit_kPa=weather.vapour_pressure_deficit_kPa,
        slope_kPa_per_C=weather.slope_kPa_per_C,
        psychrometric_constant_kPa_per_C=weather.psychrometric_constant_kPa_per_C,
        volumetric_heat_capacity_air_J_per_m3_K=weather.volumetric_heat_capacity_air_J_per_m3_K,
        aerodynamic_resistance_s_per_m=np.float32(208.0 / wind_2m),
        surface_resistance_s_per_m=np.float32(70.0),
        radiative_resistance_s_per_m=np.float32(np.inf),
    )
    assert math.isclose(reference, expected, rel_tol=1e-2)


def test_radiative_path_reduces_reference_evapotranspiration() -> None:
    """The radiative path damps the surface temperature and the evaporation a little."""
    weather = weather_of(tas_C=25.0, dewpoint_tas_C=15.0)
    kwargs = dict(
        available_energy_W_per_m2=np.float32(400.0),
        vapour_pressure_deficit_kPa=np.float32(1.5),
        slope_kPa_per_C=weather.slope_kPa_per_C,
        psychrometric_constant_kPa_per_C=weather.psychrometric_constant_kPa_per_C,
        volumetric_heat_capacity_air_J_per_m3_K=weather.volumetric_heat_capacity_air_J_per_m3_K,
        aerodynamic_resistance_s_per_m=np.float32(90.0),
        surface_resistance_s_per_m=np.float32(70.0),
    )
    without = penman_monteith(**kwargs, radiative_resistance_s_per_m=np.float32(np.inf))
    with_radiation = penman_monteith(
        **kwargs, radiative_resistance_s_per_m=np.float32(210.0)
    )
    assert 0.85 * without < with_radiation < without


def test_wet_surfaces_evaporate_faster_than_reference() -> None:
    """Without surface resistance, wet grass and open water exceed the grass reference."""
    reference, wet_vegetation, open_water = get_reference_evapotranspiration(
        weather_of(
            tas_C=22.0,
            dewpoint_tas_C=12.0,
            rsds_W_per_m2=600.0,
            rlds_W_per_m2=330.0,
            wind_10m_m_per_s=3.0,
        ),
        np.float32(40.0),
    )
    assert reference > 0.0
    assert wet_vegetation > reference
    assert open_water > reference
    # a wet canopy evaporates within the FAO-56 wet-surface range of the crop
    # coefficient (Kc_max = 1.2 in a sub-humid climate)
    assert 1.05 * reference < wet_vegetation < 1.6 * reference


def test_crop_coefficient_climate_adjustment() -> None:
    """FAO-56 eq. 70/72: zero in a sub-humid climate, positive when dry and windy."""
    for height in (0.12, 1.0, 20.0):
        assert math.isclose(
            get_crop_coefficient_climate_adjustment(
                np.float32(2.0), np.float32(45.0), np.float32(height)
            ),
            0.0,
            abs_tol=1e-7,
        )
    humid_calm = get_crop_coefficient_climate_adjustment(
        np.float32(1.0), np.float32(80.0), np.float32(1.0)
    )
    dry_windy = get_crop_coefficient_climate_adjustment(
        np.float32(5.0), np.float32(25.0), np.float32(1.0)
    )
    assert math.isclose(
        dry_windy, (0.04 * 3.0 + 0.004 * 20.0) * (1.0 / 3.0) ** 0.3, rel_tol=1e-5
    )
    assert humid_calm < 0.0 < dry_windy
    # taller vegetation is more strongly coupled to the air
    assert dry_windy < get_crop_coefficient_climate_adjustment(
        np.float32(5.0), np.float32(25.0), np.float32(3.0)
    )
    # inputs outside the FAO-56 range are limited to it
    assert get_crop_coefficient_climate_adjustment(
        np.float32(10.0), np.float32(5.0), np.float32(30.0)
    ) == get_crop_coefficient_climate_adjustment(
        np.float32(6.0), np.float32(20.0), np.float32(10.0)
    )
    # Kc_max = 1.2 + adjustment, at least Kcb + 0.05
    assert math.isclose(
        get_maximum_crop_coefficient(np.float32(0.5), dry_windy),
        1.2 + dry_windy,
        rel_tol=1e-6,
    )
    assert math.isclose(
        get_maximum_crop_coefficient(np.float32(1.15), humid_calm), 1.2, rel_tol=1e-6
    )


def test_climate_adjustment_of_basal_crop_coefficients() -> None:
    """Mid-season Kcb and the end-of-season Kcb >= 0.45 follow the climate (eq. 70)."""
    land_use_map = np.array([NON_PADDY_IRRIGATED] * 3, dtype=np.int32)
    ones = np.ones(3, dtype=np.float32)

    def basal(progress: float, wind: float, humidity: float) -> np.ndarray:
        return get_crop_factors_and_root_depths_and_lai(
            land_use_map=land_use_map,
            leaf_area_index_forest=0 * ones,
            leaf_area_index_grassland_like=0 * ones,
            leaf_area_index_forest_evergreen=np.zeros_like(ones),
            evergreen_activity=ones,
            wind_2m_m_per_s=wind * ones,
            minimum_relative_humidity_percent=humidity * ones,
            crop_map=np.array([0, 1, 0], dtype=np.int32),
            crop_progress=np.array([progress, 1.0, 0.05], dtype=np.float32),
            land_owners=np.array([0, 1, 2], dtype=np.int32),
            crop_stage_lengths=np.array([[25, 25, 25, 25]] * 2, dtype=np.int32),
            crop_sub_stage_lengths=np.array(
                [[10, 15, 25, 25, 15, 10]] * 2, dtype=np.int32
            ),
            # crop 0 ends green (Kcb end 0.85), crop 1 is harvested dry (0.20)
            crop_factor_per_crop_stage=np.array(
                [[0.4, 1.15, 0.9], [0.4, 1.15, 0.25]], dtype=np.float32
            ),
            crop_root_depths=np.array([[0.8, 1.2]] * 2, dtype=np.float32),
            grassland_root_depth_m=np.float32(0.3),
        )[1]

    adjustment = float(
        get_crop_coefficient_climate_adjustment(
            np.float32(4.0), np.float32(30.0), np.float32(1.0)
        )
    )
    standard = basal(0.6, 2.0, 45.0)
    dry = basal(0.6, 4.0, 30.0)
    np.testing.assert_allclose(standard, [1.10, 0.20, 0.15], rtol=1e-5)
    # mid-season adjusted, end of a dry-harvested crop and initial stage not
    np.testing.assert_allclose(dry, [1.10 + adjustment, 0.20, 0.15], rtol=1e-5)
    # a green end of season is adjusted
    np.testing.assert_allclose(
        basal(1.0, 4.0, 30.0)[0] - basal(1.0, 2.0, 45.0)[0], adjustment, rtol=1e-4
    )


def test_daily_wind_and_minimum_relative_humidity() -> None:
    """Daily mean 2 m wind from the 10 m wind and the lowest hourly humidity."""
    tas_K = np.full((2, 24), 293.15, dtype=np.float32)
    dewpoint_K = np.full((2, 24), 283.15, dtype=np.float32)
    dewpoint_K[1, 5] = 293.15
    tas_K[0, 14] = 298.15
    wind_u = np.full((2, 24), 3.0, dtype=np.float32)
    wind_v = np.full((2, 24), 4.0, dtype=np.float32)
    wind_2m, humidity = get_daily_wind_and_minimum_relative_humidity(
        tas_K, dewpoint_K, wind_u, wind_v
    )
    np.testing.assert_allclose(wind_2m, 5.0 * 0.748, rtol=1e-2)
    expected = [
        100
        * get_vapour_pressure(np.float32(10.0))
        / get_vapour_pressure(np.float32(25.0)),
        100
        * get_vapour_pressure(np.float32(10.0))
        / get_vapour_pressure(np.float32(20.0)),
    ]
    np.testing.assert_allclose(humidity, expected, rtol=1e-5)


def test_interception_capacity_grows_smoothly_from_zero() -> None:
    """No jump at low leaf area; the LISFLOOD value for a closed canopy."""
    assert get_interception_capacity(np.float32(0.0)) == 0.0
    leaf_area = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    capacity = np.array([get_interception_capacity(lai) for lai in leaf_area])
    assert (np.diff(capacity) > 0).all()
    assert np.diff(capacity).max() < 0.02e-3  # at most 0.02 mm per 0.01 LAI
    for lai in (5.0, 8.0):
        lisflood_m = (0.935 + 0.498 * lai - 0.00575 * lai**2) / 1000.0
        assert math.isclose(
            get_interception_capacity(np.float32(lai)), lisflood_m, rel_tol=0.01
        )


def test_dual_crop_coefficients_with_active_leaf_fraction() -> None:
    """Inactive evergreen leaves do not transpire but still shade the soil."""
    lai = np.float32(5.0)
    full_kc, full_kcb, _ = get_dual_crop_coefficients(
        lai, np.float32(1.0), np.float32(0.95), np.float32(0.0)
    )
    half_kc, half_kcb, half_maximum = get_dual_crop_coefficients(
        lai, np.float32(0.5), np.float32(0.95), np.float32(0.0)
    )
    cover = 1.0 - np.exp(-0.7 * 5.0)
    np.testing.assert_allclose(full_kcb, 0.15 + 0.8 * cover, rtol=1e-5)
    np.testing.assert_allclose(half_kcb, 0.15 + 0.8 * cover * 0.5, rtol=1e-5)
    assert half_kcb < full_kcb
    # soil evaporation stays limited to the exposed soil under the full canopy
    np.testing.assert_allclose(half_kc - half_kcb, (1.0 - cover) * 1.2, rtol=1e-4)
    np.testing.assert_allclose(half_maximum, 1.2, rtol=1e-6)
    assert half_kc < full_kc


def test_active_leaf_fraction_scales_only_the_evergreen_part() -> None:
    """The seasonal activity applies to evergreen leaves; deciduous leaves count in full."""

    def fraction(lai: float, evergreen_lai: float, activity: float) -> float:
        return float(
            get_active_leaf_fraction(
                np.float32(lai), np.float32(evergreen_lai), np.float32(activity)
            )
        )

    # deciduous forest: the seasonal leaf area is used as it is
    np.testing.assert_allclose(fraction(2.5, 0.0, 0.5), 1.0)
    # evergreen forest: the activity itself
    np.testing.assert_allclose(fraction(5.0, 5.0, 0.5), 0.5)
    # mixed forest in a frost period: only the deciduous leaves transpire
    np.testing.assert_allclose(fraction(4.0, 3.0, 0.0), 0.25)
    # fully active, and a canopy without leaves
    np.testing.assert_allclose(fraction(4.0, 3.0, 1.0), 1.0)
    np.testing.assert_allclose(fraction(0.0, 0.0, 0.0), 1.0)


def test_wet_soil_evaporation_coefficient_full_cover() -> None:
    """FAO-56 eq. 72: Kc_max >= Kcb + 0.05, so wet soil adds at least 0.05."""

    def wet_soil(basal: float, cover: float) -> float:
        maximum = get_maximum_crop_coefficient(np.float32(basal), np.float32(0.0))
        return float(
            get_wet_soil_evaporation_coefficient(
                np.float32(basal), np.float32(cover), maximum
            )
        )

    np.testing.assert_allclose(wet_soil(1.18, 0.0), 0.05, rtol=1e-5)
    np.testing.assert_allclose(wet_soil(0.15, 0.0), 1.05, rtol=1e-5)
    # under full cover there is no exposed soil
    assert wet_soil(0.95, 1.0) == 0.0
