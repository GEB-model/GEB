import numpy as np

from .modules.soil import (
    get_critical_soil_moisture_content,
    get_fraction_easily_available_soil_water,
    get_transpiration_reduction_factor,
    get_total_transpiration_reduction_factor,
)


def test_get_fraction_easily_available_soil_water():
    potential_evapotranspiration = (
        np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) / 100
    )  # cm/day to m/day
    crop_group_number = np.full_like(potential_evapotranspiration, 5)

    p = get_fraction_easily_available_soil_water(
        crop_group_number=crop_group_number,
        potential_evapotranspiration=potential_evapotranspiration,
    )

    np.testing.assert_almost_equal(
        p,
        [
            0.94339623,
            0.82644628,
            0.73529412,
            0.66225166,
            0.60240964,
            0.55248619,
            0.51020408,
            0.47393365,
            0.44247788,
        ],
    )

    crop_group_number = np.full_like(potential_evapotranspiration, 1)

    p = get_fraction_easily_available_soil_water(
        crop_group_number=crop_group_number,
        potential_evapotranspiration=potential_evapotranspiration,
    )

    np.testing.assert_almost_equal(
        p,
        [
            0.44339623,
            0.35144628,
            0.28529412,
            0.23725166,
            0.20240964,
            0.17748619,
            0.16020408,
            0.14893365,
            0.14247788,
        ],
    )


def test_get_critical_soil_moisture_content():
    p = np.array([0.3, 0.7, 1.0, 0.0])
    wfc = np.array([0.35, 0.35, 0.35, 0.35])  # field capacity
    wwp = np.array([0.15, 0.15, 0.15, 0.15])  # wilting point

    critical_soil_moisture_content = get_critical_soil_moisture_content(
        p=p, wfc=wfc, wwp=wwp
    )
    assert np.array_equal(critical_soil_moisture_content, [0.29, 0.21, 0.15, 0.35])


def test_get_transpiration_reduction_factor():
    critical_soil_moisture_content = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    wwp = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15])  # wilting point
    w = np.array([0.3, 0.2, 0.175, 0.15, 0.1, 0.0])

    transpiration_reduction_factor = get_transpiration_reduction_factor(
        w, wwp, critical_soil_moisture_content
    )

    np.testing.assert_almost_equal(
        transpiration_reduction_factor,
        np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]),
    )


def test_get_total_transpiration_reduction_factor():
    transpiration_reduction_factor_per_layer = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        ]
    )

    root_ratios = np.array(
        [
            [1, 1, 1, 1, 0.5],
            [1, 1, 0.5, 0.0, 0.0],
            [1, 0.5, 0, 0, 0],
        ]
    )

    soil_layer_height = np.array(
        [
            [0.05, 0.05, 0.05, 0.05, 0.05],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
        ]
    )

    # when transpiration_reduction_factor is equal among all layers, output should be equal to transpiration_reduction_factor
    total_transpiration_reduction_factor = get_total_transpiration_reduction_factor(
        transpiration_reduction_factor_per_layer, root_ratios, soil_layer_height
    )

    transpiration_reduction_factor_per_layer = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3, 0.3],
        ]
    )
    total_transpiration_reduction_factor = get_total_transpiration_reduction_factor(
        transpiration_reduction_factor_per_layer, root_ratios, soil_layer_height
    )
    # the first one is fully in all layers, so should be equal to the transpiration_reduction_factor
    # of all layers, consdering soil layer height
    # (0.05 * 0.1 * 1 + 1.0 * 0.2 * 1 + 2.0 * 0.3 * 1) / (0.05 + 1.0 + 2.0) = 0.2639344262295082
    # the second one is only half in the bottom layer
    # (0.05 * 0.1 * 1 + 1.0 * 0.2 * 1 + 2.0 * 0.3 * 0.5) / (0.05 * 1 + 1.0 * 1 + 2.0 * 0.5) = 0.24634146
    # the third one is only in the top layer, and half in the second layer
    # (0.05 * 0.1 * 1 + 1.0 * 0.2 * 0.5) / (0.05 * 1 + 1.0 * 0.5) = 0.19090909
    # last two are fully in the top layer, so should be equal to the transpiration_reduction_factor of the top layer
    np.testing.assert_almost_equal(
        total_transpiration_reduction_factor,
        np.array([0.2639344262295082, 0.24634146, 0.19090909, 0.1, 0.1]),
    )
