import math
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

import geb.hydrology.soil
from geb.hydrology.soil import (
    get_critical_soil_moisture_content,
    get_fraction_easily_available_soil_water,
    get_soil_moisture_at_pressure,
    get_soil_water_flow_parameters,
    get_transpiration_factor,
    vertical_water_transport,
)

from ..testconfig import output_folder

output_folder_soil = output_folder / "soil"
output_folder_soil.mkdir(exist_ok=True)


def test_get_transpiration_factor():
    critical_soil_moisture_content = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    wwp = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15])  # wilting point
    w = np.array([0.3, 0.2, 0.175, 0.15, 0.1, 0.0])

    transpiration_factor = np.zeros_like(w)
    for i in range(len(w)):
        transpiration_factor[i] = get_transpiration_factor(
            w[i], wwp[i], critical_soil_moisture_content[i]
        )

    np.testing.assert_almost_equal(
        transpiration_factor,
        np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]),
    )


def test_get_soil_moisture_at_pressure():
    capillary_suction = np.linspace(-1, -20000, 10000, dtype=np.float32).reshape(1, -1)

    soils = ["sand", "silt", "clay"]
    bubbling_pressure_cms = np.array([20, 40, 150], dtype=np.float32)
    thetass = np.array([0.4, 0.45, 0.50], dtype=np.float32)
    thetars = np.array([0.075, 0.15, 0.25], dtype=np.float32)
    lambda_s = np.array([2.5, 1.45, 1.2], dtype=np.float32)

    fig, ax = plt.subplots()
    for i in range(len(soils)):
        bubbling_pressure_cm = bubbling_pressure_cms[i]
        thetas = thetass[i]
        thetar = thetars[i]
        lambda_ = lambda_s[i]

        soil_moisture_at_pressure = get_soil_moisture_at_pressure(
            capillary_suction,
            np.full_like(capillary_suction, bubbling_pressure_cm),
            np.full_like(capillary_suction, thetas),
            np.full_like(capillary_suction, thetar),
            np.full_like(capillary_suction, lambda_),
        )
        ax.plot(-capillary_suction[0], soil_moisture_at_pressure[0], label=soils[i])

    ax.set_xlabel("|Capillary suction (cm)|")
    ax.set_ylabel("Soil moisture content")
    ax.set_xscale("log")
    ax.legend()

    plt.savefig(output_folder / "soil_moisture_at_pressure.png")


def test_get_soil_water_flow_parameters_potential():
    assert not np.isnan(
        get_soil_water_flow_parameters(
            w=np.array([0.068], dtype=np.float32),
            wres=np.array([0.016], dtype=np.float32),
            ws=np.array([0.067], dtype=np.float32),
            lambda_=np.array([0.202], dtype=np.float32),
            bubbling_pressure_cm=np.array([0.007], dtype=np.float32),
            saturated_hydraulic_conductivity=np.array([1.0], dtype=np.float32),
        )[1]
    )

    assert (
        get_soil_water_flow_parameters(
            w=np.array([0.015], dtype=np.float32),
            wres=np.array([0.016], dtype=np.float32),
            ws=np.array([0.067], dtype=np.float32),
            lambda_=np.array([0.202], dtype=np.float32),
            bubbling_pressure_cm=np.array([40], dtype=np.float32),
            saturated_hydraulic_conductivity=np.array([1.0], dtype=np.float32),
        )[1]
        != np.inf
    )


@pytest.mark.parametrize("pf_value", [2.0, 4.2])
def test_soil_moisture_potential_inverse(pf_value):
    # Convert pF value to capillary suction in cm (h)
    capillary_suction = np.array(
        [-(10**pf_value)], dtype=np.float32
    )  # Negative value for suction

    # Define soil parameters for the test
    thetas = np.array([0.45], dtype=np.float32)  # Saturated water content (volumetric)
    thetar = np.array([0.05], dtype=np.float32)  # Residual water content (volumetric)
    lambda_ = np.array([0.5], dtype=np.float32)  # Pore-size distribution index
    bubbling_pressure_cm = np.array([10.0], dtype=np.float32)  # Bubbling pressure in cm

    # Step 1: Calculate theta from capillary suction
    theta = get_soil_moisture_at_pressure(
        capillary_suction, bubbling_pressure_cm, thetas, thetar, lambda_
    )

    # Step 2: Calculate capillary suction from theta
    capillary_suction_calculated = get_soil_water_flow_parameters(
        w=theta,
        wres=thetar,
        ws=thetas,
        lambda_=lambda_,
        saturated_hydraulic_conductivity=np.array([1.0], dtype=np.float32),
        bubbling_pressure_cm=bubbling_pressure_cm,
    )[0]

    # Allow a small tolerance due to numerical approximations
    tolerance = 1e-2 * abs(capillary_suction)  # 1% of the suction value

    # Assert that the original and calculated capillary suctions are approximately equal
    assert np.isclose(
        capillary_suction, capillary_suction_calculated, atol=tolerance
    ), (
        f"Capillary suction mismatch at pF {pf_value}: original {capillary_suction}, calculated {capillary_suction_calculated}"
    )


def test_get_fraction_easily_available_soil_water():
    potential_evapotranspiratios = (
        np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) / 100
    )  # cm/day to m/day
    p5s_test = np.array(
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
        ]
    )
    p1s_test = np.array(
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
        ]
    )
    for potential_evapotranspiration, p5_test, p1_test in zip(
        potential_evapotranspiratios, p5s_test, p1s_test, strict=True
    ):
        p5 = get_fraction_easily_available_soil_water(
            crop_group_number=5,
            potential_evapotranspiration=potential_evapotranspiration,
        )

        assert math.isclose(p5, p5_test, rel_tol=1e-6)

        p1 = get_fraction_easily_available_soil_water(
            crop_group_number=1,
            potential_evapotranspiration=potential_evapotranspiration,
        )

        assert math.isclose(p1, p1_test, rel_tol=1e-6)


def test_get_critical_soil_moisture_content():
    p = np.array([0.3, 0.7, 1.0, 0.0])
    wfc = np.array([0.35, 0.35, 0.35, 0.35])  # field capacity
    wwp = np.array([0.15, 0.15, 0.15, 0.15])  # wilting point

    critical_soil_moisture_content = get_critical_soil_moisture_content(
        p=p, wfc=wfc, wwp=wwp
    )
    assert np.array_equal(critical_soil_moisture_content, [0.29, 0.21, 0.15, 0.35])


def test_get_unsaturated_hydraulic_conductivity():
    wres = np.full(1_000_000, 0.1, dtype=np.float32)
    ws = np.full_like(wres, 0.4)

    w = np.linspace(0, ws[-1], wres.size, dtype=np.float32)

    lambdas_ = np.arange(0.1, 0.6, 0.1, dtype=np.float32)
    # we take 1 so that we the outcome is the relative hydraulic conductivity
    saturated_hydraulic_conductivity = np.full_like(wres, 1.0)

    fig, (ax0, ax1) = plt.subplots(1, 2)

    for lambda_ in lambdas_:
        unsaturated_hydraulic_conductivity, soil_water_potential = (
            get_soil_water_flow_parameters(
                w=w,
                wres=wres,
                ws=ws,
                lambda_=np.full_like(wres, lambda_),
                saturated_hydraulic_conductivity=saturated_hydraulic_conductivity,
                bubbling_pressure_cm=np.full_like(wres, 40.0),
            )
        )

        relative_water_content = w / ws
        log_unsaturated_hydraulic_conductivity = np.full_like(
            unsaturated_hydraulic_conductivity, np.nan
        )
        plot_every_n = 100
        ax0.plot(
            relative_water_content[::plot_every_n],
            np.log10(
                unsaturated_hydraulic_conductivity,
                out=log_unsaturated_hydraulic_conductivity,
                where=unsaturated_hydraulic_conductivity > 0,
            )[::plot_every_n],
            label=round(lambda_, 1),
        )
        ax1.plot(
            relative_water_content[::plot_every_n],
            soil_water_potential[::plot_every_n],
            label=round(lambda_, 1),
        )

    with open(
        output_folder_soil / "get_soil_water_flow_parameters.txt",
        "w",
    ) as f:
        f.write(
            get_soil_water_flow_parameters.inspect_asm(
                get_soil_water_flow_parameters.signatures[0]
            )
        )

    start_time = time()
    for i in range(10):
        get_soil_water_flow_parameters(
            w=w,
            wres=wres,
            ws=ws,
            lambda_=np.full_like(wres, lambda_),
            saturated_hydraulic_conductivity=saturated_hydraulic_conductivity,
            bubbling_pressure_cm=np.full_like(wres, 40.0),
        )
    end_time = time()
    print(f"took {end_time - start_time:.6f} seconds")

    ax0.set_xlim(0, 1)
    ax0.set_ylim(-15, 0)
    ax0.set_xlabel("Soil moisture content")
    ax0.set_ylabel("Unsaturated hydraulic conductivity")

    ax0.legend()

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-15_000, 0)
    ax1.set_xlabel("Soil moisture content")
    ax1.set_ylabel("Soil water potential")

    ax1.legend()

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


@pytest.mark.parametrize("capillary_rise_from_groundwater", [0.0, 0.01])
def test_vertical_water_transport(capillary_rise_from_groundwater):
    ncols = 11

    soil_layer_height = np.array(
        [[0.05, 0.10, 0.15, 0.30, 0.40, 1.00]], dtype=np.float32
    )
    # soil_thickness = np.array([[0.4, 0.4, 0.4, 0.4, 0.4, 0.4]])
    soil_layer_height = np.vstack([soil_layer_height] * ncols).T

    available_water_infiltration = np.full(ncols, 0.005, dtype=np.float32)
    land_use_type = np.full_like(available_water_infiltration, 0.1, dtype=np.int32)
    frost_index = np.full_like(
        available_water_infiltration, -9999, dtype=np.float32
    )  # no frost
    arno_beta = np.full_like(available_water_infiltration, 1.0, dtype=np.float32)
    preferential_flow_constant = 4.5
    topwater = np.zeros_like(available_water_infiltration)

    geb.hydrology.soil.N_SOIL_LAYERS = soil_layer_height.shape[0]
    geb.hydrology.soil.FROST_INDEX_THRESHOLD = 0

    theta_fc = np.full_like(soil_layer_height, 0.4)
    theta_s = np.full_like(soil_layer_height, 0.5)
    theta_res = np.full_like(soil_layer_height, 0.1)

    saturated_hydraulic_conductivity = np.full_like(soil_layer_height, 0.1)
    lambda_ = np.full_like(soil_layer_height, 0.9)
    bubbling_pressure_cm = np.full_like(soil_layer_height, 40)

    wres = theta_res * soil_layer_height
    ws = theta_s * soil_layer_height

    theta = np.full_like(soil_layer_height, 0)
    theta[:, 0] = theta_res[:, 0]
    theta[:, 1] = theta_s[:, 1]
    theta[:, 2] = theta_fc[:, 2]
    theta[:, 3] = 0.2
    theta[:, 4] = np.linspace(
        theta_res[0, 4], theta_s[0, 4], soil_layer_height.shape[0]
    )
    theta[:, 5] = np.linspace(
        theta_s[0, 5], theta_res[0, 5], soil_layer_height.shape[0]
    )
    theta[:, 6] = np.linspace(
        theta_fc[0, 6], theta_res[0, 6], soil_layer_height.shape[0]
    )
    theta[:, 7] = np.linspace(theta_fc[0, 7], theta_s[0, 7], soil_layer_height.shape[0])
    theta[:, 8] = np.linspace(
        theta_res[0, 8], theta_fc[0, 8], soil_layer_height.shape[0]
    )
    theta[:, 9] = np.linspace(theta_s[0, 9], theta_fc[0, 9], soil_layer_height.shape[0])
    theta[:, 10] = theta_res[:, 10]
    theta[-1, 10] = theta_s[-1, 10]

    w = theta * soil_layer_height

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    fig.tight_layout()

    plot_soil_layers(axes[0], soil_layer_height, w, wres, ws)

    preferential_flow, direct_runoff, groundwater_recharge = vertical_water_transport(
        available_water_infiltration=available_water_infiltration,
        capillary_rise_from_groundwater=np.full_like(
            available_water_infiltration, capillary_rise_from_groundwater
        ),
        ws=ws,
        wres=wres,
        saturated_hydraulic_conductivity=saturated_hydraulic_conductivity,
        lambda_=lambda_,
        bubbling_pressure_cm=bubbling_pressure_cm,
        land_use_type=land_use_type,
        frost_index=frost_index,
        arno_beta=arno_beta,
        preferential_flow_constant=preferential_flow_constant,
        w=w,
        topwater=topwater,
        soil_layer_height=soil_layer_height,
    )

    # with open(output_folder_soil / "vertical_water_transport_compiled.txt", "w") as f:
    #     f.write(
    #         vertical_water_transport.inspect_asm(vertical_water_transport.signatures[0])
    #     )

    plot_soil_layers(axes[1], soil_layer_height, w, wres, ws)

    available_water_infiltration.fill(0)
    for _ in range(1000):
        preferential_flow, direct_runoff, groundwater_recharge = (
            vertical_water_transport(
                available_water_infiltration=available_water_infiltration,
                capillary_rise_from_groundwater=np.full_like(
                    available_water_infiltration, capillary_rise_from_groundwater
                ),
                ws=ws,
                wres=wres,
                saturated_hydraulic_conductivity=saturated_hydraulic_conductivity,
                lambda_=lambda_,
                bubbling_pressure_cm=bubbling_pressure_cm,
                land_use_type=land_use_type,
                frost_index=frost_index,
                arno_beta=arno_beta,
                preferential_flow_constant=preferential_flow_constant,
                w=w,
                topwater=topwater,
                soil_layer_height=soil_layer_height,
            )
        )

    plot_soil_layers(axes[2], soil_layer_height, w, wres, ws)

    plt.savefig(
        output_folder_soil
        / f"vertical_water_transport_caprise_{capillary_rise_from_groundwater}.png"
    )
