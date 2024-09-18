import numpy as np
import pytest

import matplotlib.pyplot as plt

from ..testconfig import output_folder

import geb.hydrology.soil
from geb.hydrology.soil import (
    get_critical_soil_moisture_content,
    get_fraction_easily_available_soil_water,
    get_transpiration_factor_single,
    get_total_transpiration_factor,
    get_aeration_stress_threshold,
    get_aeration_stress_factor,
    get_unsaturated_hydraulic_conductivity,
    get_soil_moisture_at_pressure,
    capillary_rise_between_soil_layers,
    vertical_water_transport,
    get_soil_water_potential,
)

output_folder_soil = output_folder / "soil"
output_folder_soil.mkdir(exist_ok=True)


def test_get_soil_moisture_at_pressure():
    capillary_suction = np.linspace(-1, -20000, 10000)

    soils = ["sand", "silt", "clay"]
    bubbling_pressure_cms = np.array([20, 40, 150], dtype=float)
    thetass = np.array([0.4, 0.45, 0.50])
    thetars = np.array([0.075, 0.15, 0.25])
    lambda_s = np.array([2.5, 1.45, 1.2])

    fig, ax = plt.subplots()
    for i in range(len(soils)):
        bubbling_pressure_cm = bubbling_pressure_cms[i]
        thetas = thetass[i]
        thetar = thetars[i]
        lambda_ = lambda_s[i]

        soil_moisture_at_pressure = get_soil_moisture_at_pressure(
            capillary_suction, bubbling_pressure_cm, thetas, thetar, lambda_
        )
        ax.plot(-capillary_suction, soil_moisture_at_pressure, label=soils[i])

    ax.set_xlabel("|Capillary suction (cm)|")
    ax.set_ylabel("Soil moisture content")
    ax.set_xscale("log")
    ax.legend()

    plt.savefig(output_folder / "soil_moisture_at_pressure.png")


@pytest.mark.parametrize("pf_value", [2.0, 4.2])
def test_soil_moisture_potential_inverse(pf_value):
    # Convert pF value to capillary suction in cm (h)
    capillary_suction = -(10**pf_value)  # Negative value for suction

    # Define soil parameters for the test
    thetas = 0.45  # Saturated water content (volumetric)
    thetar = 0.05  # Residual water content (volumetric)
    lambda_ = 0.5  # Pore-size distribution index
    bubbling_pressure_cm = 10.0  # Bubbling pressure in cm

    # Step 1: Calculate theta from capillary suction
    theta = get_soil_moisture_at_pressure(
        capillary_suction, bubbling_pressure_cm, thetas, thetar, lambda_
    )

    # Step 2: Calculate capillary suction from theta
    capillary_suction_calculated = get_soil_water_potential(
        theta,
        thetar,
        thetas,
        lambda_,
        bubbling_pressure_cm,
        minimum_effective_saturation=0,
    )

    # Allow a small tolerance due to numerical approximations
    tolerance = 1e-2 * abs(capillary_suction)  # 1% of the suction value

    # Assert that the original and calculated capillary suctions are approximately equal
    assert np.isclose(
        capillary_suction, capillary_suction_calculated, atol=tolerance
    ), f"Capillary suction mismatch at pF {pf_value}: original {capillary_suction}, calculated {capillary_suction_calculated}"


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


def test_get_transpiration_factor():
    critical_soil_moisture_content = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    wwp = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15])  # wilting point
    w = np.array([0.3, 0.2, 0.175, 0.15, 0.1, 0.0])

    transpiration_factor = np.zeros_like(w)
    for i in range(len(w)):
        transpiration_factor[i] = get_transpiration_factor_single(
            w[i], wwp[i], critical_soil_moisture_content[i]
        )

    np.testing.assert_almost_equal(
        transpiration_factor,
        np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]),
    )


def test_get_total_transpiration_factor():
    transpiration_factor_per_layer = np.array(
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

    # when transpiration_factor is equal among all layers, output should be equal to transpiration_factor
    total_transpiration_factor = get_total_transpiration_factor(
        transpiration_factor_per_layer, root_ratios, soil_layer_height
    )

    transpiration_factor_per_layer = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3, 0.3],
        ]
    )
    total_transpiration_factor = get_total_transpiration_factor(
        transpiration_factor_per_layer, root_ratios, soil_layer_height
    )
    # the first one is fully in all layers, so should be equal to the transpiration_factor
    # of all layers, consdering soil layer height
    # (0.05 * 0.1 * 1 + 1.0 * 0.2 * 1 + 2.0 * 0.3 * 1) / (0.05 + 1.0 + 2.0) = 0.2639344262295082
    # the second one is only half in the bottom layer
    # (0.05 * 0.1 * 1 + 1.0 * 0.2 * 1 + 2.0 * 0.3 * 0.5) / (0.05 * 1 + 1.0 * 1 + 2.0 * 0.5) = 0.24634146
    # the third one is only in the top layer, and half in the second layer
    # (0.05 * 0.1 * 1 + 1.0 * 0.2 * 0.5) / (0.05 * 1 + 1.0 * 0.5) = 0.19090909
    # last two are fully in the top layer, so should be equal to the transpiration_factor of the top layer
    np.testing.assert_almost_equal(
        total_transpiration_factor,
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


def test_get_aeration_stress_factor():
    # default settings
    aeration_days_counter = 0
    crop_lag_aeration_days = 3
    ws = 0.1
    w = 0.09
    aeration_stress_threshold = 0.08

    aeration_stress_factor = get_aeration_stress_factor(
        aeration_days_counter=aeration_days_counter,
        crop_lag_aeration_days=crop_lag_aeration_days,
        ws=ws,
        w=w,
        aeration_stress_threshold=aeration_stress_threshold,
    )

    # at zero aeration_days_counter, the reduction factor should be 1 (no reduction)
    assert aeration_stress_factor == 1

    aeration_stress_factor = get_aeration_stress_factor(
        aeration_days_counter=1,
        crop_lag_aeration_days=crop_lag_aeration_days,
        ws=ws,
        w=w,
        aeration_stress_threshold=aeration_stress_threshold,
    )

    aeration_stress_factor = get_aeration_stress_factor(
        aeration_days_counter=4,
        crop_lag_aeration_days=crop_lag_aeration_days,
        ws=ws,
        w=w,
        aeration_stress_threshold=aeration_stress_threshold,
    )

    aeration_stress_factor = get_aeration_stress_factor(
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


def plot_soil_layers(ax, soil_thickness, w, wres, ws, fluxes=None):
    n_soil_columns = soil_thickness.shape[1]
    for column in range(n_soil_columns):
        current_depth = 0
        for layer in range(soil_thickness.shape[0]):
            cell_thickness = soil_thickness[layer, column]
            cell_center = current_depth + cell_thickness / 2

            alpha = (
                w[layer, column] / cell_thickness - wres[layer, column] / cell_thickness
            ) / (
                ws[layer, column] / cell_thickness
                - wres[layer, column] / cell_thickness
            )
            color = "blue"
            if alpha < 0:
                alpha = 1
                color = "red"
            if alpha > 1:
                alpha = 1
                color = "green"

            rect = plt.Rectangle(
                (column, current_depth),
                1,
                cell_thickness,
                color=color,
                alpha=alpha,
                linewidth=0,
                zorder=0,
            )
            ax.add_patch(rect)
            current_depth += cell_thickness

            if fluxes is not None:
                flux = fluxes[layer, column]
                if flux != 0:
                    ax.arrow(
                        column + 0.5,
                        cell_center,
                        0,  # vertical arrow
                        flux * 10,
                        head_width=0.1,
                        head_length=0.05,
                        fc="red",
                        ec="red",
                        zorder=1,
                    )

    ax.set_xlim(0, n_soil_columns)
    ax.set_ylim(0, current_depth)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Soil layer depth")
    ax.invert_yaxis()


def test_capillary_rise_between_soil_layers():
    soil_thickness = np.array([[0.001, 0.2, 0.4, 0.8, 0.3, 0.2]])
    soil_thickness = np.vstack([soil_thickness] * 11).T

    geb.hydrology.soil.N_SOIL_LAYERS = soil_thickness.shape[0]

    theta_fc = np.full_like(soil_thickness, 0.4)
    theta_s = np.full_like(soil_thickness, 0.5)
    theta_res = np.full_like(soil_thickness, 0.1)

    saturated_hydraulic_conductivity = np.full_like(soil_thickness, 100.0)
    lambda_ = np.full_like(soil_thickness, 0.9)

    wres = theta_res * soil_thickness
    ws = theta_s * soil_thickness
    wfc = theta_fc * soil_thickness

    theta = np.full_like(soil_thickness, 0)
    theta[:, 0] = theta_res[:, 0]
    theta[:, 1] = theta_s[:, 1]
    theta[:, 2] = theta_fc[:, 2]
    theta[:, 3] = 0.2
    theta[:, 4] = np.linspace(theta_res[0, 4], theta_s[0, 4], soil_thickness.shape[0])
    theta[:, 5] = np.linspace(theta_s[0, 5], theta_res[0, 5], soil_thickness.shape[0])
    theta[:, 6] = np.linspace(theta_fc[0, 6], theta_res[0, 6], soil_thickness.shape[0])
    theta[:, 7] = np.linspace(theta_fc[0, 7], theta_s[0, 7], soil_thickness.shape[0])
    theta[:, 8] = np.linspace(theta_res[0, 8], theta_fc[0, 8], soil_thickness.shape[0])
    theta[:, 9] = np.linspace(theta_s[0, 9], theta_fc[0, 9], soil_thickness.shape[0])
    theta[:, 10] = theta_res[:, 10]
    theta[-1, 10] = theta_s[-1, 10]

    w = theta * soil_thickness

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.tight_layout()

    plot_soil_layers(axes[0], soil_thickness, w, wres, ws)

    capillary_rise = capillary_rise_between_soil_layers(
        wfc=wfc,
        ws=ws,
        wres=wres,
        saturated_hydraulic_conductivity=saturated_hydraulic_conductivity,
        lambda_=lambda_,
        w=w,
    )

    plot_soil_layers(axes[1], soil_thickness, w, wres, ws, capillary_rise)

    for _ in range(1000):
        capillary_rise = capillary_rise_between_soil_layers(
            wfc=wfc,
            ws=ws,
            wres=wres,
            saturated_hydraulic_conductivity=saturated_hydraulic_conductivity,
            lambda_=lambda_,
            w=w,
        )

    plot_soil_layers(axes[2], soil_thickness, w, wres, ws, capillary_rise)

    plt.savefig(output_folder / "soil_layers.png")

    soil_thickness[0] = 0.001
    capillary_rise = capillary_rise_between_soil_layers(
        wfc=wfc,
        ws=ws,
        wres=wres,
        saturated_hydraulic_conductivity=saturated_hydraulic_conductivity,
        lambda_=lambda_,
        w=w,
    )


@pytest.mark.parametrize("capillary_rise_from_groundwater", [0.0, 0.01])
def test_vertical_water_transport(capillary_rise_from_groundwater):
    ncols = 11

    soil_thickness = np.array([[0.05, 0.10, 0.15, 0.30, 0.40, 1.00]], dtype=np.float32)
    # soil_thickness = np.array([[0.4, 0.4, 0.4, 0.4, 0.4, 0.4]])
    soil_thickness = np.vstack([soil_thickness] * ncols).T

    available_water_infiltration = np.full(ncols, 0.005, dtype=np.float32)
    land_use_type = np.full_like(available_water_infiltration, 0.1, dtype=np.int32)
    frost_index = np.full_like(
        available_water_infiltration, -9999, dtype=np.float32
    )  # no frost
    arno_beta = np.full_like(available_water_infiltration, 1.0, dtype=np.float32)
    preferential_flow_constant = 4.5
    topwater = np.zeros_like(available_water_infiltration)

    geb.hydrology.soil.N_SOIL_LAYERS = soil_thickness.shape[0]
    geb.hydrology.soil.FROST_INDEX_THRESHOLD = 0

    theta_fc = np.full_like(soil_thickness, 0.4)
    theta_s = np.full_like(soil_thickness, 0.5)
    theta_res = np.full_like(soil_thickness, 0.1)

    saturated_hydraulic_conductivity = np.full_like(soil_thickness, 0.1)
    lambda_ = np.full_like(soil_thickness, 0.9)

    wres = theta_res * soil_thickness
    ws = theta_s * soil_thickness

    theta = np.full_like(soil_thickness, 0)
    theta[:, 0] = theta_res[:, 0]
    theta[:, 1] = theta_s[:, 1]
    theta[:, 2] = theta_fc[:, 2]
    theta[:, 3] = 0.2
    theta[:, 4] = np.linspace(theta_res[0, 4], theta_s[0, 4], soil_thickness.shape[0])
    theta[:, 5] = np.linspace(theta_s[0, 5], theta_res[0, 5], soil_thickness.shape[0])
    theta[:, 6] = np.linspace(theta_fc[0, 6], theta_res[0, 6], soil_thickness.shape[0])
    theta[:, 7] = np.linspace(theta_fc[0, 7], theta_s[0, 7], soil_thickness.shape[0])
    theta[:, 8] = np.linspace(theta_res[0, 8], theta_fc[0, 8], soil_thickness.shape[0])
    theta[:, 9] = np.linspace(theta_s[0, 9], theta_fc[0, 9], soil_thickness.shape[0])
    theta[:, 10] = theta_res[:, 10]
    theta[-1, 10] = theta_s[-1, 10]

    w = theta * soil_thickness

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    fig.tight_layout()

    plot_soil_layers(axes[0], soil_thickness, w, wres, ws)

    preferential_flow, direct_runoff, groundwater_recharge, net_fluxes = (
        vertical_water_transport(
            available_water_infiltration,
            ws,
            wres,
            saturated_hydraulic_conductivity,
            lambda_,
            land_use_type,
            frost_index,
            np.full_like(available_water_infiltration, capillary_rise_from_groundwater),
            arno_beta,
            preferential_flow_constant,
            w,
            topwater,
            soil_thickness,
        )
    )

    plot_soil_layers(axes[1], soil_thickness, w, wres, ws, net_fluxes)

    # available_water_infiltration.fill(0)
    for _ in range(1000):
        preferential_flow, direct_runoff, groundwater_recharge, net_fluxes = (
            vertical_water_transport(
                available_water_infiltration,
                ws,
                wres,
                saturated_hydraulic_conductivity,
                lambda_,
                land_use_type,
                frost_index,
                np.full_like(
                    available_water_infiltration, capillary_rise_from_groundwater
                ),
                arno_beta,
                preferential_flow_constant,
                w,
                topwater,
                soil_thickness,
            )
        )

    plot_soil_layers(axes[2], soil_thickness, w, wres, ws, net_fluxes)

    plt.savefig(
        output_folder_soil
        / f"vertical_water_transport_caprise_{capillary_rise_from_groundwater}.png"
    )
