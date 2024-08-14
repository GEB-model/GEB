import numpy as np
import matplotlib.pyplot as plt

from ..setup import output_folder

from cwatm.modules.soil import (
    get_critical_soil_moisture_content,
    get_fraction_easily_available_soil_water,
    get_transpiration_reduction_factor_single,
    get_total_transpiration_reduction_factor,
    get_aeration_stress_threshold,
    get_aeration_stress_reduction_factor,
    get_unsaturated_hydraulic_conductivity,
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

    transpiration_reduction_factor = np.zeros_like(w)
    for i in range(len(w)):
        transpiration_reduction_factor[i] = get_transpiration_reduction_factor_single(
            w[i], wwp[i], critical_soil_moisture_content[i]
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


def test_get_aeration_stress_threshold():
    soil_layer_height = 0.10
    ws = 0.05
    aeration_stress_threshold = get_aeration_stress_threshold(
        ws=ws,
        soil_layer_height=soil_layer_height,
        crop_aeration_stress_threshold=0,
    )
    # if crop_aeration_stress_threshold is 0, then it should be equal to ws
    assert aeration_stress_threshold == ws

    aeration_stress_threshold = get_aeration_stress_threshold(
        ws=0.1,
        soil_layer_height=soil_layer_height,
        crop_aeration_stress_threshold=100,
    )

    # if crop_aeration_stress_threshold is 100, the crop is always in aeration stress
    assert aeration_stress_threshold == 0.0

    aeration_stress_threshold = get_aeration_stress_threshold(
        ws=0.1,
        soil_layer_height=soil_layer_height,
        crop_aeration_stress_threshold=50,
    )

    # if crop_aeration_stress_threshold is 50, the crop is in aeration stress at half of the ws
    assert aeration_stress_threshold == soil_layer_height / 2


def test_get_aeration_stress_reduction_factor():
    # default settings
    aeration_days_counter = 0
    crop_lag_aeration_days = 3
    ws = 0.1
    w = 0.09
    aeration_stress_threshold = 0.08

    aeration_stress_reduction_factor = get_aeration_stress_reduction_factor(
        aeration_days_counter=aeration_days_counter,
        crop_lag_aeration_days=crop_lag_aeration_days,
        ws=ws,
        w=w,
        aeration_stress_threshold=aeration_stress_threshold,
    )

    # at zero aeration_days_counter, the reduction factor should be 1 (no reduction)
    assert aeration_stress_reduction_factor == 1

    aeration_stress_reduction_factor = get_aeration_stress_reduction_factor(
        aeration_days_counter=1,
        crop_lag_aeration_days=crop_lag_aeration_days,
        ws=ws,
        w=w,
        aeration_stress_threshold=aeration_stress_threshold,
    )

    print(aeration_stress_reduction_factor)

    aeration_stress_reduction_factor = get_aeration_stress_reduction_factor(
        aeration_days_counter=4,
        crop_lag_aeration_days=crop_lag_aeration_days,
        ws=ws,
        w=w,
        aeration_stress_threshold=aeration_stress_threshold,
    )

    print(aeration_stress_reduction_factor)

    aeration_stress_reduction_factor = get_aeration_stress_reduction_factor(
        aeration_days_counter=aeration_days_counter,
        crop_lag_aeration_days=crop_lag_aeration_days,
        ws=ws,
        w=w,
        aeration_stress_threshold=aeration_stress_threshold,
    )


def test_get_unsaturated_hydraulic_conductivity():
    wres = np.full(1000, 0.1)
    ws = np.full_like(wres, 0.4)

    w = np.linspace(0, ws[-1], wres.size)

    lambdas_ = np.arange(0.1, 0.6, 0.1)
    # we take 1 so that we the outcome is the relative hydraulic conductivity
    saturated_hydraulic_conductivity = np.full_like(wres, 1.0)

    fig, ax = plt.subplots()

    for lambda_ in lambdas_:
        unsaturated_hydraulic_conductivity = np.zeros_like(w)
        for i in range(w.size):
            unsaturated_hydraulic_conductivity[i] = (
                get_unsaturated_hydraulic_conductivity(
                    w=w[i],
                    wres=wres[i],
                    ws=ws[i],
                    lambda_=lambda_,
                    saturated_hydraulic_conductivity=saturated_hydraulic_conductivity[
                        i
                    ],
                )
            )

        relative_water_content = w / ws
        log_unsaturated_hydraulic_conductivity = np.full_like(
            unsaturated_hydraulic_conductivity, np.nan
        )
        ax.plot(
            relative_water_content,
            np.log10(
                unsaturated_hydraulic_conductivity,
                out=log_unsaturated_hydraulic_conductivity,
                where=unsaturated_hydraulic_conductivity > 0,
            ),
            label=round(lambda_, 1),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-15, 0)
    ax.set_xlabel("Soil moisture content")
    ax.set_ylabel("Unsaturated hydraulic conductivity")

    ax.legend()

    plt.savefig(output_folder / "unsaturated_hydraulic_conductivity.png")
