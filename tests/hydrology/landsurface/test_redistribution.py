"""Tests for the Ross soil water redistribution scheme in the GEB land surface hydrology module."""

import os

import numpy as np
import pytest

from geb.hydrology.landsurface.constants import N_SOIL_LAYERS
from geb.hydrology.landsurface.redistribution import distribute_soil_water_ross


def test_distribute_soil_water_ross_basic() -> None:
    """Test basic downward flow in a 6-layer soil column using Ross scheme."""
    n_layers = 6
    timestep_length_s = np.float32(3600.0)
    soil_layer_height_m = np.array([0.05, 0.1, 0.15, 0.3, 0.4, 1.0], dtype=np.float32)
    # Top wet, bottom drier
    w = (
        np.array([0.45, 0.4, 0.3, 0.2, 0.2, 0.2], dtype=np.float32)
        * soil_layer_height_m
    )
    wr = np.full(n_layers, 0.05, dtype=np.float32) * soil_layer_height_m
    ws = np.full(n_layers, 0.5, dtype=np.float32) * soil_layer_height_m

    # Sensible enthalpy/heat capacity
    enth = np.full(n_layers, 1e7, dtype=np.float32)
    hc = np.full(n_layers, 1e6, dtype=np.float32)

    # Hydraulic properties
    ksat = np.full(n_layers, 0.01 / np.float32(3600.0), dtype=np.float32)  # m/s
    bubbling_pressure_m_positive = np.full(
        n_layers, 0.6666, dtype=np.float32
    )  # -1 / 0.015 (cm to m)
    lambda_ = np.full(n_layers, 1.3 - 1.0, dtype=np.float32)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)

    # Infiltration (none)
    active_layer_idx = np.int32(0)
    topwater_m = np.float32(0.0)

    # Run redistribution
    interface_dist_m = (soil_layer_height_m[:-1] + soil_layer_height_m[1:]) / 2.0
    w_new = w.copy()
    e_new = enth.copy()
    (
        perc,
        rise,
        perc_gw,
        perc_gw_e,
        lateral_outflow,
        interflow_e,
    ) = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_new,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        soil_enthalpy_J_per_m2=e_new,
        solid_heat_capacity_J_per_m2_K=hc,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        interface_dist_m=interface_dist_m,
        soil_layer_height=soil_layer_height_m,
        bubbling_pressure_m_positive=bubbling_pressure_m_positive,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        slope_m_per_m=np.float32(0.01),
        hillslope_length_m=np.float32(100.0),
        interflow_multiplier=np.float32(1.0),
        green_ampt_active_layer_idx=active_layer_idx,
        topwater_m=topwater_m,
        gw_ksat_m_per_s=np.float32(0.0),
    )

    # Check that water moved downwards (layer 0 decreased, layer 1 increased)
    # The first layer should lose water due to gravity/gradient.
    assert w_new[0] < w[0]
    # Total mass balance check (precision 1e-7 due to float32 linear solve)
    assert np.float32(np.sum(w_new) + lateral_outflow) == pytest.approx(
        np.float32(np.sum(w)), abs=1e-7
    )
    # Check water content bounds
    assert np.all(w_new >= wr)
    assert np.all(w_new <= ws)


def test_distribute_soil_water_ross_active_layer_masking() -> None:
    """Test that layers above the green_ampt_active_layer_idx do not lose water."""
    n_layers = 6
    timestep_length_s = np.float32(3600.0)
    soil_layer_height_m = np.array([0.05, 0.1, 0.15, 0.3, 0.4, 1.0], dtype=np.float32)

    # Case: Infiltration front is at the 3rd layer (index 2).
    # Layers 0 and 1 should be "frozen" in terms of redistribution
    # to avoid interference with the explicit Green-Ampt front.
    active_idx = np.int32(2)

    # Wet soil
    w = (
        np.array([0.45, 0.45, 0.45, 0.45, 0.45, 0.45], dtype=np.float32)
        * soil_layer_height_m
    )
    wr = np.full(n_layers, 0.05, dtype=np.float32) * soil_layer_height_m
    ws = np.full(n_layers, 0.5, dtype=np.float32) * soil_layer_height_m
    enth = np.full(n_layers, 1e7, dtype=np.float32)
    hc = np.full(n_layers, 1e6, dtype=np.float32)
    ksat = np.full(n_layers, 0.1 / np.float32(3600.0), dtype=np.float32)
    bubbling_pressure_m_positive = np.full(n_layers, 0.5, dtype=np.float32)
    lambda_ = np.full(n_layers, 1.5 - 1.0, dtype=np.float32)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)

    interface_dist_m = (soil_layer_height_m[:-1] + soil_layer_height_m[1:]) / 2.0
    w_new = w.copy()
    e_new = enth.copy()
    (
        _,
        _,
        _,
        _,
        lateral_outflow,
        _,
    ) = distribute_soil_water_ross(
        timestep_length_s,
        w_new,
        wr,
        ws,
        e_new,
        hc,
        ksat,
        interface_dist_m,
        soil_layer_height_m,
        bubbling_pressure_m_positive,
        lambda_,
        pore_size_index,
        np.float32(0.01),
        np.float32(100.0),
        np.float32(1.0),
        active_idx,
        np.float32(0.0),
        np.float32(0.0),
    )

    # Layers above active_idx must NOT change
    assert w_new[0] == pytest.approx(w[0], abs=1e-5)
    assert w_new[1] == pytest.approx(w[1], abs=1e-5)

    # Check mass balance
    assert np.float32(np.sum(w_new) + lateral_outflow) == pytest.approx(
        np.float32(np.sum(w)), abs=1e-7
    )
    # Check water content bounds
    assert np.all(w_new >= wr)
    assert np.all(w_new <= ws)


def test_distribute_soil_water_ross_extreme_dry_wet() -> None:
    """Test Ross scheme with a mix of extremely dry and extremely wet layers."""
    n_layers = 6
    timestep_length_s = np.float32(3600.0)
    soil_layer_height_m = np.array([0.05, 0.1, 0.15, 0.3, 0.4, 1.0], dtype=np.float32)

    # Top and bottom are saturated, middle is residual
    wr_val, ws_val = 0.05, 0.5
    w = (
        np.array([ws_val, wr_val, wr_val, wr_val, wr_val, ws_val], dtype=np.float32)
        * soil_layer_height_m
    )
    wr = np.full(n_layers, wr_val, dtype=np.float32) * soil_layer_height_m
    ws = np.full(n_layers, ws_val, dtype=np.float32) * soil_layer_height_m

    enth = np.full(n_layers, 1e7, dtype=np.float32)
    hc = np.full(n_layers, 1e6, dtype=np.float32)
    ksat = np.full(n_layers, 0.1 / np.float32(3600.0), dtype=np.float32)
    bubbling_pressure_m_positive = np.full(n_layers, 0.5, dtype=np.float32)
    lambda_ = np.full(n_layers, 1.5 - 1.0, dtype=np.float32)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)

    interface_dist_m = (soil_layer_height_m[:-1] + soil_layer_height_m[1:]) / 2.0
    w_new = w.copy()
    e_tmp = enth.copy()
    (
        _,
        _,
        _,
        _,
        lateral_outflow,
        _,
    ) = distribute_soil_water_ross(
        timestep_length_s,
        w_new,
        wr,
        ws,
        e_tmp,
        hc,
        ksat,
        interface_dist_m,
        soil_layer_height_m,
        bubbling_pressure_m_positive,
        lambda_,
        pore_size_index,
        np.float32(0.01),
        np.float32(100.0),
        np.float32(1.0),
        np.int32(0),
        np.float32(0.0),
        np.float32(0.0),
    )

    # Mass balance
    assert np.float32(np.sum(w_new) + lateral_outflow) == pytest.approx(
        np.float32(np.sum(w)), abs=1e-7
    )
    # Check water content bounds
    assert np.all(w_new >= wr)
    assert np.all(w_new <= ws)
    # Water should have moved from saturated layers to dry layers
    assert w_new[0] < w[0]
    assert w_new[5] < w[5]
    assert np.any(w_new[1:5] > w[1:5])


def test_distribute_soil_water_ross_interflow_matrix() -> None:
    """Test that interflow is correctly handled within the Ross matrix.

    We compare a case with interflow vs a case without interflow.
    With interflow, layers above field capacity should lose more water
    and the total lateral outflow should be non-zero.
    """
    timestep_length_s = np.float32(3600.0)
    soil_layer_height_m = np.array([0.05, 0.1, 0.15, 0.3, 0.4, 1.0], dtype=np.float32)

    # All layers wet (above field capacity)
    w_initial = (
        np.array([0.45, 0.45, 0.45, 0.45, 0.45, 0.45], dtype=np.float32)
        * soil_layer_height_m
    )
    wr = np.full(N_SOIL_LAYERS, 0.05, dtype=np.float32) * soil_layer_height_m
    ws = np.full(N_SOIL_LAYERS, 0.5, dtype=np.float32) * soil_layer_height_m

    enth = np.full(N_SOIL_LAYERS, 1e7, dtype=np.float32)
    hc = np.full(N_SOIL_LAYERS, 1e6, dtype=np.float32)

    ksat = np.full(N_SOIL_LAYERS, 0.01 / np.float32(3600.0), dtype=np.float32)  # m/s
    bubbling_pressure_m_positive = np.full(N_SOIL_LAYERS, 0.6666, dtype=np.float32)
    lambda_ = np.full(N_SOIL_LAYERS, 1.3 - 1.0, dtype=np.float32)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)

    interface_dist_m = (soil_layer_height_m[:-1] + soil_layer_height_m[1:]) / 2.0

    # 1. Run WITHOUT interflow (interflow_multiplier = 0)
    w_no_interflow = w_initial.copy()
    e_no_interflow = enth.copy()
    results_no_interflow = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_no_interflow,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        soil_enthalpy_J_per_m2=e_no_interflow,
        solid_heat_capacity_J_per_m2_K=hc,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        interface_dist_m=interface_dist_m,
        soil_layer_height=soil_layer_height_m,
        bubbling_pressure_m_positive=bubbling_pressure_m_positive,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        slope_m_per_m=np.float32(0.1),
        hillslope_length_m=np.float32(100.0),
        interflow_multiplier=np.float32(0.0),  # OFF
        green_ampt_active_layer_idx=np.int32(0),
        topwater_m=np.float32(0.0),
        gw_ksat_m_per_s=np.float32(0.0),
    )
    lateral_outflow_no = results_no_interflow[4]

    # 2. Run WITH interflow (interflow_multiplier = 10.0 for visible effect)
    w_with_interflow = w_initial.copy()
    e_with_interflow = enth.copy()
    results_with_interflow = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_with_interflow,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        soil_enthalpy_J_per_m2=e_with_interflow,
        solid_heat_capacity_J_per_m2_K=hc,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        interface_dist_m=interface_dist_m,
        soil_layer_height=soil_layer_height_m,
        bubbling_pressure_m_positive=bubbling_pressure_m_positive,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        slope_m_per_m=np.float32(0.1),
        hillslope_length_m=np.float32(100.0),
        interflow_multiplier=np.float32(10.0),  # ON
        green_ampt_active_layer_idx=np.int32(0),
        topwater_m=np.float32(0.0),
        gw_ksat_m_per_s=np.float32(0.0),
    )
    lateral_outflow_with = results_with_interflow[4]

    # Assertions
    assert lateral_outflow_no == 0.0
    assert lateral_outflow_with > 0.0

    # Layers should be drier with interflow than without
    for i in range(N_SOIL_LAYERS):
        assert w_with_interflow[i] < w_no_interflow[i]

    # Mass balance check for both cases
    assert np.float32(
        np.sum(w_no_interflow) + results_no_interflow[2] + results_no_interflow[4]
    ) == pytest.approx(np.float32(np.sum(w_initial)), abs=1e-7)
    assert np.float32(
        np.sum(w_with_interflow) + results_with_interflow[2] + results_with_interflow[4]
    ) == pytest.approx(np.float32(np.sum(w_initial)), abs=1e-7)


def test_distribute_soil_water_ross_interflow_dry_soil() -> None:
    """Test that interflow is zero when soil is below field capacity."""
    timestep_length_s = np.float32(3600.0)
    soil_layer_height_m = np.array([0.05, 0.1, 0.15, 0.3, 0.4, 1.0], dtype=np.float32)

    # Soil below field capacity (0.2 < 0.3)
    w_initial = (
        np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32) * soil_layer_height_m
    )
    wr = np.full(N_SOIL_LAYERS, 0.05, dtype=np.float32) * soil_layer_height_m
    ws = np.full(N_SOIL_LAYERS, 0.5, dtype=np.float32) * soil_layer_height_m

    enth = np.full(N_SOIL_LAYERS, 1e7, dtype=np.float32)
    hc = np.full(N_SOIL_LAYERS, 1e6, dtype=np.float32)

    ksat = np.full(N_SOIL_LAYERS, 0.01 / np.float32(3600.0), dtype=np.float32)
    bubbling_pressure_m_positive = np.full(N_SOIL_LAYERS, 0.6666, dtype=np.float32)
    lambda_ = np.full(N_SOIL_LAYERS, 1.3 - 1.0, dtype=np.float32)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)

    interface_dist_m = (soil_layer_height_m[:-1] + soil_layer_height_m[1:]) / 2.0

    results = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_initial.copy(),
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        soil_enthalpy_J_per_m2=enth.copy(),
        solid_heat_capacity_J_per_m2_K=hc,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        interface_dist_m=interface_dist_m,
        soil_layer_height=soil_layer_height_m,
        bubbling_pressure_m_positive=bubbling_pressure_m_positive,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        slope_m_per_m=np.float32(0.1),
        hillslope_length_m=np.float32(100.0),
        interflow_multiplier=np.float32(10.0),
        green_ampt_active_layer_idx=np.int32(0),
        topwater_m=np.float32(0.0),
        gw_ksat_m_per_s=np.float32(0.0),
    )

    lateral_outflow = results[4]
    # Small interflow values can occur due to floating point precision in the
    # tridiagonal solve. We use a small epsilon check.
    assert lateral_outflow == pytest.approx(0.0, abs=1e-7)


def test_distribute_soil_water_ross_frozen_barrier() -> None:
    """Test Ross scheme with a frozen layer acting as a hydraulic barrier."""
    n_layers = 6
    timestep_length_s = np.float32(3600.0)
    soil_layer_height_m = np.array([0.05, 0.1, 0.15, 0.3, 0.4, 1.0], dtype=np.float32)

    # Wet top, dry bottom
    w = (
        np.array([0.45, 0.45, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        * soil_layer_height_m
    )
    wr = np.full(n_layers, 0.05, dtype=np.float32) * soil_layer_height_m
    ws = np.full(n_layers, 0.5, dtype=np.float32) * soil_layer_height_m

    # Layer 2 is frozen (enthalpy very negative)
    latent_heat_areal = 0.5 * 1000.0 * 3.34e5  # max possible water depth * rho * L
    enth = np.full(n_layers, 1e7, dtype=np.float32)
    enth[2] = -2.0 * latent_heat_areal

    hc = np.full(n_layers, 1e6, dtype=np.float32)
    ksat = np.full(n_layers, 0.1 / np.float32(3600.0), dtype=np.float32)
    bubbling_pressure_m_positive = np.full(n_layers, 0.5, dtype=np.float32)
    lambda_ = np.full(n_layers, 1.5 - 1.0, dtype=np.float32)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)

    interface_dist_m = (soil_layer_height_m[:-1] + soil_layer_height_m[1:]) / 2.0
    w_new = w.copy()
    e_new = enth.copy()
    (
        _,
        _,
        _,
        _,
        lateral_outflow,
        _,
    ) = distribute_soil_water_ross(
        timestep_length_s,
        w_new,
        wr,
        ws,
        e_new,
        hc,
        ksat,
        interface_dist_m,
        soil_layer_height_m,
        bubbling_pressure_m_positive,
        lambda_,
        pore_size_index,
        np.float32(0.01),
        np.float32(100.0),
        np.float32(1.0),
        np.int32(0),
        np.float32(0.0),
        np.float32(0.0),
    )

    # Mass balance
    assert np.float32(np.sum(w_new) + lateral_outflow) == pytest.approx(
        np.float32(np.sum(w)), abs=1e-7
    )
    # Check water content bounds
    assert np.all(w_new >= wr)
    assert np.all(w_new <= ws)

    # Layer 2 is frozen, it should act as a barrier.
    assert w_new[3] == pytest.approx(w[3], abs=1e-5)


@pytest.mark.parametrize("ksat", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("gw_ksat", [0.0, 0.01, 0.1])
def test_ross_redistribution_scenarios(ksat: float, gw_ksat: float) -> None:
    """Test Ross scheme with a variety of redistribution scenarios and optionally plot results.

    This includes scenarios with and without groundwater percolation.
    """
    import matplotlib.pyplot as plt

    n_layers = 6
    timestep_length_s = np.float32(3600.0)
    duration_hours = 48
    soil_layer_height_m = np.full(n_layers, 0.1, dtype=np.float32)
    z = np.cumsum(soil_layer_height_m) - 0.5 * soil_layer_height_m

    # Hydraulic properties
    ksat_arr = np.full(n_layers, ksat / np.float32(3600.0), dtype=np.float32)
    bubbling_pressure_m_positive_val = np.full(n_layers, 0.5, dtype=np.float32)
    lambda_ = np.full(n_layers, 1.5 - 1.0, dtype=np.float32)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)
    wr_vol, ws_vol = 0.05, 0.5
    wr = (np.full(n_layers, wr_vol) * soil_layer_height_m).astype(np.float32)
    ws = (np.full(n_layers, ws_vol) * soil_layer_height_m).astype(np.float32)

    enth = np.full(n_layers, 1e7, dtype=np.float32)
    hc = np.full(n_layers, 1e6, dtype=np.float32)

    scenarios = {
        "top_wet": np.array([0.49] * 1 + [0.1] * 5),
        "uniform": np.array([0.3] * 6),
        "interflow_test": np.array([0.45] * 6),
    }

    output_dir = "tests/output/ross_scenarios"
    os.makedirs(output_dir, exist_ok=True)

    for name, initial_theta in scenarios.items():
        w = (initial_theta * soil_layer_height_m).astype(np.float32)
        current_w = w.copy()
        current_enth = enth.copy()

        history_w = [current_w.copy() / soil_layer_height_m]
        history_e = [current_enth.copy() / 1e6]  # MJ/m2
        history_lateral = [0.0]  # Cumulative lateral outflow
        perc_gw_total = 0.0
        perc_gw_e_total = 0.0
        interflow_e_total = 0.0
        lateral_outflow_total = 0.0

        interface_dist_m = (soil_layer_height_m[:-1] + soil_layer_height_m[1:]) / 2.0
        for _ in range(duration_hours):
            (
                perc_step,
                rise_step,
                perc_gw_step,
                perc_gw_e_step,
                lateral_outflow_step,
                interflow_e_step,
            ) = distribute_soil_water_ross(
                timestep_length_s=timestep_length_s,
                water_content_m=current_w,
                water_content_residual_m=wr,
                water_content_saturated_m=ws,
                soil_enthalpy_J_per_m2=current_enth,
                solid_heat_capacity_J_per_m2_K=hc,
                saturated_hydraulic_conductivity_m_per_s=ksat_arr,
                interface_dist_m=interface_dist_m,
                soil_layer_height=soil_layer_height_m,
                bubbling_pressure_m_positive=bubbling_pressure_m_positive_val,
                lambda_=lambda_,
                pore_size_index=pore_size_index,
                slope_m_per_m=np.float32(1),
                hillslope_length_m=np.float32(100.0),
                interflow_multiplier=np.float32(1.0),
                green_ampt_active_layer_idx=np.int32(0),
                topwater_m=np.float32(0.0),
                gw_ksat_m_per_s=np.float32(gw_ksat / 3600.0),
            )
            history_w.append(current_w.copy() / soil_layer_height_m)
            history_e.append(current_enth.copy() / 1e6)
            perc_gw_total += perc_gw_step
            perc_gw_e_total += perc_gw_e_step
            lateral_outflow_total += lateral_outflow_step
            interflow_e_total += interflow_e_step
            history_lateral.append(lateral_outflow_total)

            # Mass balance per step
            assert np.float32(
                np.sum(current_w) + perc_gw_total + lateral_outflow_total
            ) == pytest.approx(np.float32(np.sum(w)), abs=1e-6)
            # Energy balance per step
            assert np.float32(
                np.sum(current_enth) + perc_gw_e_total + interflow_e_total
            ) == pytest.approx(np.float32(np.sum(enth)), abs=100.0)

        # Plotting side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8), sharey=False)

        for i, (hw, he) in enumerate(zip(history_w, history_e)):
            if i % 6 == 0 or i == len(history_w) - 1:
                ax1.plot(hw, z, label=f"t={i}h")
                ax2.plot(he, z, label=f"t={i}h")

        ax3.plot(range(len(history_lateral)), history_lateral, "k-")
        ax3.set_xlabel("Time (h)")
        ax3.set_ylabel("Cumulative Lateral Outflow (m)")
        ax3.set_title("Lateral Outflow over Time")
        ax3.grid(True)

        ax1.invert_yaxis()
        ax1.set_xlabel("Volumetric Water Content (-)")
        ax1.set_ylabel("Depth (m)")
        ax1.set_title(
            f"Water: {name}\nksat={ksat}, gw_ksat={gw_ksat}\nRecharge: {perc_gw_total:.4f}m, Lateral: {lateral_outflow_total:.4f}m"
        )
        ax1.grid(True)

        ax2.invert_yaxis()
        ax2.set_xlabel("Enthalpy (MJ/m2)")
        ax2.set_ylabel("Depth (m)")
        ax2.set_title(
            f"Energy: {name}\nTotal Heat Loss: {(perc_gw_e_total + interflow_e_total) / 1e6:.4f}MJ/m2"
        )
        ax2.grid(True)
        ax2.legend()

        filename = f"{name}_k{ksat}_gw{gw_ksat}_sidebyside.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


def test_distribute_soil_water_ross_percolation_to_groundwater() -> None:
    """Test Ross scheme with a gravity bottom boundary (groundwater percolation)."""
    n_layers = 6
    timestep_length_s = np.float32(3600.0)
    soil_layer_height_m = np.array([0.05, 0.1, 0.15, 0.3, 0.4, 1.0], dtype=np.float32)
    # Saturated column
    w = (
        np.array([0.45, 0.45, 0.45, 0.45, 0.45, 0.45], dtype=np.float32)
        * soil_layer_height_m
    )
    wr = np.full(n_layers, 0.05, dtype=np.float32) * soil_layer_height_m
    ws = np.full(n_layers, 0.5, dtype=np.float32) * soil_layer_height_m
    enth = np.full(n_layers, 1e7, dtype=np.float32)
    hc = np.full(n_layers, 1e6, dtype=np.float32)
    # High Ksat allows significant percolation
    ksat = np.full(n_layers, 0.1 / np.float32(3600.0), dtype=np.float32)
    bubbling_pressure_m_positive = np.full(n_layers, 0.5, dtype=np.float32)
    lambda_ = np.full(n_layers, 1.5 - 1.0, dtype=np.float32)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)

    # Enable groundwater percolation
    gw_ksat = np.float32(0.1 / 3600.0)

    interface_dist_m = (soil_layer_height_m[:-1] + soil_layer_height_m[1:]) / 2.0
    w_new = w.copy()
    e_new = enth.copy()
    (
        perc,
        rise,
        perc_gw,
        perc_gw_e,
        lateral_outflow,
        interflow_e,
    ) = distribute_soil_water_ross(
        timestep_length_s,
        w_new,
        wr,
        ws,
        e_new,
        hc,
        ksat,
        interface_dist_m,
        soil_layer_height_m,
        bubbling_pressure_m_positive,
        lambda_,
        pore_size_index,
        np.float32(0.01),
        np.float32(100.0),
        np.float32(1.0),
        np.int32(0),
        np.float32(0.0),
        gw_ksat,
    )

    # Column should have lost water to groundwater
    assert perc_gw > 0
    # Mass balance: Sum(new) + perc_gw + lateral_outflow == Sum(old)
    assert np.float32(np.sum(w_new) + perc_gw + lateral_outflow) == pytest.approx(
        np.float32(np.sum(w)), abs=1e-7
    )
    # Check water content bounds
    assert np.all(w_new >= wr)
    assert np.all(w_new <= ws)

    # Check that enthalpy was also advected out
    assert np.sum(e_new) < np.sum(enth)
    assert perc_gw_e > 0
    assert np.float32(np.sum(e_new) + perc_gw_e + interflow_e) == pytest.approx(
        np.float32(np.sum(enth)), abs=5.0
    )
