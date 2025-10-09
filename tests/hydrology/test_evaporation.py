"""Tests for evaporation functions."""

import numpy as np

from geb.hydrology.evaporation import (
    get_CO2_induced_crop_factor_adustment,
    get_crop_factors_and_root_depths,
    get_potential_bare_soil_evaporation,
    get_potential_evapotranspiration,
    get_potential_transpiration,
)
from geb.hydrology.landcover import (
    FOREST,
    GRASSLAND_LIKE,
    NON_PADDY_IRRIGATED,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
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
    assert get_CO2_induced_crop_factor_adustment(369.41) == 1.0
    assert get_CO2_induced_crop_factor_adustment(550.0) == 0.95
    assert get_CO2_induced_crop_factor_adustment(550.0 + (550.0 - 369.41) == 0.9)


def test_get_crop_factors_and_root_depths() -> None:
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
