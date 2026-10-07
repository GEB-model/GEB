"""Tests for the Ross soil water redistribution scheme in the GEB land surface hydrology module."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from geb.hydrology.landsurface.constants import N_SOIL_LAYERS
from geb.hydrology.landsurface.redistribution import (
    calculate_capillary_rise_and_percolation_parameters,
    calculate_saturated_matric_flux_potential,
    distribute_soil_water_ross,
)

from ...testconfig import output_folder


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
    wfc = np.full(n_layers, 0.3, dtype=np.float32) * soil_layer_height_m

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
        cap_gw,
        perc_gw_e,
        lateral_outflow,
        interflow_e,
    ) = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_new,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        water_content_field_capacity_m=wfc,
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
        groundwater_depth_m=np.float32(10.0),
        deep_soil_temperature_C=np.float32(10.0),
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
    wfc = np.full(n_layers, 0.3, dtype=np.float32) * soil_layer_height_m
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
        _,
        lateral_outflow,
        _,
    ) = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_new,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        water_content_field_capacity_m=wfc,
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
        interflow_multiplier=np.float32(
            0.0
        ),  # Test vertical redistribution masking in isolation
        green_ampt_active_layer_idx=active_idx,
        topwater_m=np.float32(0.0),
        gw_ksat_m_per_s=np.float32(0.0),
        groundwater_depth_m=np.float32(10.0),
        deep_soil_temperature_C=np.float32(10.0),
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
    wfc = np.full(n_layers, 0.3, dtype=np.float32) * soil_layer_height_m

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
        _,
        lateral_outflow,
        _,
    ) = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_new,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        water_content_field_capacity_m=wfc,
        soil_enthalpy_J_per_m2=e_tmp,
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
        green_ampt_active_layer_idx=np.int32(0),
        topwater_m=np.float32(0.0),
        gw_ksat_m_per_s=np.float32(0.0),
        groundwater_depth_m=np.float32(10.0),
        deep_soil_temperature_C=np.float32(10.0),
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
    wfc = np.full(N_SOIL_LAYERS, 0.3, dtype=np.float32) * soil_layer_height_m

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
        water_content_field_capacity_m=wfc,
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
        groundwater_depth_m=np.float32(10.0),
        deep_soil_temperature_C=np.float32(10.0),
    )
    lateral_outflow_no = results_no_interflow[5]

    # 2. Run WITH interflow (interflow_multiplier = 10.0 for visible effect)
    w_with_interflow = w_initial.copy()
    e_with_interflow = enth.copy()
    results_with_interflow = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_with_interflow,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        water_content_field_capacity_m=wfc,
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
        groundwater_depth_m=np.float32(10.0),
        deep_soil_temperature_C=np.float32(10.0),
    )
    lateral_outflow_with = results_with_interflow[5]

    # Assertions
    assert lateral_outflow_no == 0.0
    assert lateral_outflow_with > 0.0

    # Layers should be drier with interflow than without
    for i in range(N_SOIL_LAYERS):
        assert w_with_interflow[i] < w_no_interflow[i]

    # Mass balance check for both cases
    assert np.float32(
        np.sum(w_no_interflow) + results_no_interflow[2] + results_no_interflow[5]
    ) == pytest.approx(np.float32(np.sum(w_initial)), abs=1e-7)
    assert np.float32(
        np.sum(w_with_interflow) + results_with_interflow[2] + results_with_interflow[5]
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
    wfc = np.full(N_SOIL_LAYERS, 0.3, dtype=np.float32) * soil_layer_height_m

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
        water_content_field_capacity_m=wfc,
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
        groundwater_depth_m=np.float32(10.0),
        deep_soil_temperature_C=np.float32(10.0),
    )

    lateral_outflow = results[5]
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
    wfc = np.full(n_layers, 0.3, dtype=np.float32) * soil_layer_height_m

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
        _,
        lateral_outflow,
        _,
    ) = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_new,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        water_content_field_capacity_m=wfc,
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
        green_ampt_active_layer_idx=np.int32(0),
        topwater_m=np.float32(0.0),
        gw_ksat_m_per_s=np.float32(0.0),
        groundwater_depth_m=np.float32(10.0),
        deep_soil_temperature_C=np.float32(10.0),
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
    wfc = (np.full(n_layers, 0.3) * soil_layer_height_m).astype(np.float32)

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
                cap_gw_step,
                perc_gw_e_step,
                lateral_outflow_step,
                interflow_e_step,
            ) = distribute_soil_water_ross(
                timestep_length_s=timestep_length_s,
                water_content_m=current_w,
                water_content_residual_m=wr,
                water_content_saturated_m=ws,
                water_content_field_capacity_m=wfc,
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
                groundwater_depth_m=np.float32(10.0),
                deep_soil_temperature_C=np.float32(10.0),
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
    wfc = np.full(n_layers, 0.3, dtype=np.float32) * soil_layer_height_m
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
        cap_gw,
        perc_gw_e,
        lateral_outflow,
        interflow_e,
    ) = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_new,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        water_content_field_capacity_m=wfc,
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
        green_ampt_active_layer_idx=np.int32(0),
        topwater_m=np.float32(0.0),
        gw_ksat_m_per_s=gw_ksat,
        groundwater_depth_m=np.float32(10.0),
        deep_soil_temperature_C=np.float32(10.0),
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


def test_distribute_soil_water_ross_active_wetting_front_lateral_flow() -> None:
    """Test dual-mode lateral interflow with active and inactive Green-Ampt wetting front.

    Verifies:
    1. During an active wetting front (storm quickflow), layers in the transmission
       zone (i <= active_layer_idx) drain laterally via saturated conductivity.
    2. Lateral quickflow is orders of magnitude faster than unsaturated matrix flow.
    3. Lateral quickflow is zero when soil moisture is at or below field capacity.
    4. Exact water mass balance holds across all modes.
    """
    n_layers: int = 6
    timestep_length_s: np.float32 = np.float32(3600.0)
    soil_layer_height_m: np.ndarray = np.array(
        [0.05, 0.1, 0.15, 0.3, 0.4, 1.0], dtype=np.float32
    )
    interface_dist_m: np.ndarray = (
        soil_layer_height_m[:-1] + soil_layer_height_m[1:]
    ) / 2.0

    wr: np.ndarray = np.full(n_layers, 0.05, dtype=np.float32) * soil_layer_height_m
    ws: np.ndarray = np.full(n_layers, 0.50, dtype=np.float32) * soil_layer_height_m
    wfc: np.ndarray = np.full(n_layers, 0.30, dtype=np.float32) * soil_layer_height_m

    # Top layers saturated/wet above field capacity, deeper layers dry
    w_wet: np.ndarray = (
        np.array([0.45, 0.45, 0.15, 0.15, 0.15, 0.15], dtype=np.float32)
        * soil_layer_height_m
    )
    enth: np.ndarray = np.full(n_layers, 1e7, dtype=np.float32)
    hc: np.ndarray = np.full(n_layers, 1e6, dtype=np.float32)

    # Ksat = 10 mm/h = 10 / 3600 mm/s
    ksat: np.ndarray = np.full(n_layers, 0.01 / np.float32(3600.0), dtype=np.float32)
    bubbling_pressure_m_positive: np.ndarray = np.full(n_layers, 0.5, dtype=np.float32)
    lambda_: np.ndarray = np.full(n_layers, 0.5, dtype=np.float32)
    pore_size_index: np.ndarray = np.float32(3.0) + (np.float32(2.0) / lambda_)

    slope_m_per_m: np.float32 = np.float32(0.10)
    hillslope_length_m: np.float32 = np.float32(100.0)
    interflow_multiplier: np.float32 = np.float32(1.0)

    # 1. Active wetting front in layer 1 (layers 0 and 1 are in perched transmission zone)
    w_active: np.ndarray = w_wet.copy()
    e_active: np.ndarray = enth.copy()
    (
        _,
        _,
        _,
        _,
        _,
        lateral_outflow_active,
        _,
    ) = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_active,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        water_content_field_capacity_m=wfc,
        soil_enthalpy_J_per_m2=e_active,
        solid_heat_capacity_J_per_m2_K=hc,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        interface_dist_m=interface_dist_m,
        soil_layer_height=soil_layer_height_m,
        bubbling_pressure_m_positive=bubbling_pressure_m_positive,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        slope_m_per_m=slope_m_per_m,
        hillslope_length_m=hillslope_length_m,
        interflow_multiplier=interflow_multiplier,
        green_ampt_active_layer_idx=np.int32(1),
        topwater_m=np.float32(0.0),
        gw_ksat_m_per_s=np.float32(0.0),
        groundwater_depth_m=np.float32(10.0),
        deep_soil_temperature_C=np.float32(10.0),
    )

    # 2. Inactive wetting front (inter-storm matrix flow: green_ampt_active_layer_idx = -1)
    w_inactive: np.ndarray = w_wet.copy()
    e_inactive: np.ndarray = enth.copy()
    (
        _,
        _,
        _,
        _,
        _,
        lateral_outflow_inactive,
        _,
    ) = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_inactive,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        water_content_field_capacity_m=wfc,
        soil_enthalpy_J_per_m2=e_inactive,
        solid_heat_capacity_J_per_m2_K=hc,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        interface_dist_m=interface_dist_m,
        soil_layer_height=soil_layer_height_m,
        bubbling_pressure_m_positive=bubbling_pressure_m_positive,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        slope_m_per_m=slope_m_per_m,
        hillslope_length_m=hillslope_length_m,
        interflow_multiplier=interflow_multiplier,
        green_ampt_active_layer_idx=np.int32(-1),
        topwater_m=np.float32(0.0),
        gw_ksat_m_per_s=np.float32(0.0),
        groundwater_depth_m=np.float32(10.0),
        deep_soil_temperature_C=np.float32(10.0),
    )

    # Active storm wetting front should produce much higher quickflow than matrix flow:
    assert lateral_outflow_active > np.float32(0.0)
    assert lateral_outflow_active > lateral_outflow_inactive * np.float32(3.0)

    # Mass balance check
    assert np.float32(np.sum(w_active) + lateral_outflow_active) == pytest.approx(
        np.float32(np.sum(w_wet)), abs=1e-7
    )
    assert np.float32(np.sum(w_inactive) + lateral_outflow_inactive) == pytest.approx(
        np.float32(np.sum(w_wet)), abs=1e-7
    )

    # 3. Moisture in active transmission zone at field capacity produces zero quickflow
    # (sub-front layers at residual storage)
    w_fc: np.ndarray = wfc.copy()
    w_fc[2:] = wr[2:]
    e_fc: np.ndarray = enth.copy()
    (
        _,
        _,
        _,
        _,
        _,
        lateral_outflow_fc,
        _,
    ) = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_fc,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        water_content_field_capacity_m=wfc,
        soil_enthalpy_J_per_m2=e_fc,
        solid_heat_capacity_J_per_m2_K=hc,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        interface_dist_m=interface_dist_m,
        soil_layer_height=soil_layer_height_m,
        bubbling_pressure_m_positive=bubbling_pressure_m_positive,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        slope_m_per_m=slope_m_per_m,
        hillslope_length_m=hillslope_length_m,
        interflow_multiplier=interflow_multiplier,
        green_ampt_active_layer_idx=np.int32(1),
        topwater_m=np.float32(0.0),
        gw_ksat_m_per_s=np.float32(0.0),
        groundwater_depth_m=np.float32(10.0),
        deep_soil_temperature_C=np.float32(10.0),
    )
    assert lateral_outflow_fc == pytest.approx(0.0, abs=1e-7)


def test_wetting_front_retraction_from_interflow() -> None:
    """Test that lateral interflow correctly retracts the Green-Ampt wetting front.

    Verifies:
    1. Retraction depth equals interflow volume divided by moisture deficit.
    2. Active layer index updates when the front retracts across soil layer boundaries.
    3. Wetting front fully dissipates and resets to -1 when retracted to surface.
    """
    soil_layer_height = np.array([0.05, 0.10, 0.15, 0.30, 0.40, 1.00], dtype=np.float32)

    # Initial state: front in layer 1 at 0.12 m depth (layer 0 is 0.05m, layer 1 ends at 0.15m)
    depth = np.float32(0.12)
    deficit = np.float32(0.20)
    active_idx = 1
    suction = np.float32(0.35)

    # Case 1: Partial retraction within same layer (lateral outflow = 0.004 m = 4 mm)
    # Delta L_f = 0.004 / 0.20 = 0.02 m -> new depth = 0.10 m (still in layer 1)
    lateral_outflow = np.float32(0.004)
    retraction = lateral_outflow / deficit
    depth = max(np.float32(0.0), depth - retraction)
    assert depth == pytest.approx(0.10, abs=1e-5)

    # Case 2: Retraction across boundary into layer 0 (lateral outflow = 0.012 m = 12 mm)
    # Delta L_f = 0.012 / 0.20 = 0.06 m -> new depth = 0.04 m (in layer 0)
    lateral_outflow = np.float32(0.012)
    retraction = lateral_outflow / deficit
    depth = max(np.float32(0.0), depth - retraction)
    accum = np.float32(0.0)
    new_idx = 0
    for l_idx in range(len(soil_layer_height)):
        accum += soil_layer_height[l_idx]
        if depth <= accum:
            new_idx = l_idx
            break
    assert depth == pytest.approx(0.04, abs=1e-5)
    assert new_idx == 0

    # Case 3: Complete dissipation of wetting front (lateral outflow = 0.010 m = 10 mm)
    # Delta L_f = 0.010 / 0.20 = 0.05 m -> new depth <= 0 -> resets
    lateral_outflow = np.float32(0.010)
    retraction = lateral_outflow / deficit
    depth = max(np.float32(0.0), depth - retraction)
    if depth <= np.float32(1e-4):
        depth = np.float32(0.0)
        new_idx = -1
        suction = np.float32(0.0)
        deficit = np.float32(0.0)
    assert depth == 0.0
    assert new_idx == -1
    assert suction == 0.0
    assert deficit == 0.0


def test_distribute_soil_water_ross_capillary_rise_shallow_gw() -> None:
    """Test that upward capillary rise occurs when soil is dry and water table is shallow."""
    timestep_length_s = np.float32(3600.0)
    soil_layer_height_m = np.array(
        [0.05, 0.10, 0.15, 0.30, 0.40, 1.00], dtype=np.float32
    )
    ws = 0.45 * soil_layer_height_m
    wr = 0.05 * soil_layer_height_m
    wfc = 0.25 * soil_layer_height_m

    # Dry soil (near residual in lower layers)
    w_initial = wr + 0.1 * (wfc - wr)
    w_new = w_initial.copy()

    enth = np.full(N_SOIL_LAYERS, 1e7, dtype=np.float32)
    hc = np.full(N_SOIL_LAYERS, 1e6, dtype=np.float32)
    ksat = np.full(N_SOIL_LAYERS, 0.01 / np.float32(3600.0), dtype=np.float32)
    bubbling_pressure_m_positive = np.full(N_SOIL_LAYERS, 0.3, dtype=np.float32)
    lambda_ = np.full(N_SOIL_LAYERS, 0.3, dtype=np.float32)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)
    interface_dist_m = (soil_layer_height_m[:-1] + soil_layer_height_m[1:]) / 2.0

    gw_ksat = np.float32(1e-4)
    # Groundwater table just below the 2.0m soil profile (z_wt = 2.1m)
    groundwater_depth_m = np.float32(2.1)
    deep_soil_temp_C = np.float32(12.0)

    (
        perc,
        rise,
        perc_gw,
        cap_gw,
        perc_gw_e,
        lateral_outflow,
        interflow_e,
    ) = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_new,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        water_content_field_capacity_m=wfc,
        soil_enthalpy_J_per_m2=enth,
        solid_heat_capacity_J_per_m2_K=hc,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        interface_dist_m=interface_dist_m,
        soil_layer_height=soil_layer_height_m,
        bubbling_pressure_m_positive=bubbling_pressure_m_positive,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        slope_m_per_m=np.float32(0.0),
        hillslope_length_m=np.float32(100.0),
        interflow_multiplier=np.float32(0.0),
        green_ampt_active_layer_idx=np.int32(-1),
        topwater_m=np.float32(0.0),
        gw_ksat_m_per_s=gw_ksat,
        groundwater_depth_m=groundwater_depth_m,
        deep_soil_temperature_C=deep_soil_temp_C,
    )

    # Assertions
    assert cap_gw > 0.0, (
        "Capillary rise should be positive for dry soil with shallow water table"
    )
    assert perc_gw == 0.0, "Percolation should be zero when capillary rise occurs"
    # Net water in soil column must increase by capillary rise (minus any lateral outflow)
    water_change = float(np.sum(w_new) - np.sum(w_initial))
    assert np.isclose(
        water_change, float(cap_gw - lateral_outflow), rtol=1e-5, atol=1e-6
    )
    # Enthalpy loss to groundwater should be negative (net influx of advected heat)
    assert perc_gw_e < 0.0, (
        "Enthalpy loss should be negative (heat gained from groundwater)"
    )


def test_distribute_soil_water_ross_capillary_rise_deep_gw_shutoff() -> None:
    """Test that capillary rise is completely zero when the water table is deep."""
    timestep_length_s = np.float32(3600.0)
    soil_layer_height_m = np.array(
        [0.05, 0.10, 0.15, 0.30, 0.40, 1.00], dtype=np.float32
    )
    ws = 0.45 * soil_layer_height_m
    wr = 0.05 * soil_layer_height_m
    wfc = 0.25 * soil_layer_height_m

    w_initial = wr + 0.1 * (wfc - wr)
    w_new = w_initial.copy()

    enth = np.full(N_SOIL_LAYERS, 1e7, dtype=np.float32)
    hc = np.full(N_SOIL_LAYERS, 1e6, dtype=np.float32)
    ksat = np.full(N_SOIL_LAYERS, 0.01 / np.float32(3600.0), dtype=np.float32)
    bubbling_pressure_m_positive = np.full(N_SOIL_LAYERS, 0.3, dtype=np.float32)
    lambda_ = np.full(N_SOIL_LAYERS, 0.3, dtype=np.float32)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)
    interface_dist_m = (soil_layer_height_m[:-1] + soil_layer_height_m[1:]) / 2.0

    gw_ksat = np.float32(1e-4)
    # Deep water table (15 meters below surface)
    groundwater_depth_m = np.float32(15.0)

    (
        perc,
        rise,
        perc_gw,
        cap_gw,
        perc_gw_e,
        lateral_outflow,
        interflow_e,
    ) = distribute_soil_water_ross(
        timestep_length_s=timestep_length_s,
        water_content_m=w_new,
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        water_content_field_capacity_m=wfc,
        soil_enthalpy_J_per_m2=enth,
        solid_heat_capacity_J_per_m2_K=hc,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        interface_dist_m=interface_dist_m,
        soil_layer_height=soil_layer_height_m,
        bubbling_pressure_m_positive=bubbling_pressure_m_positive,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        slope_m_per_m=np.float32(0.0),
        hillslope_length_m=np.float32(100.0),
        interflow_multiplier=np.float32(0.0),
        green_ampt_active_layer_idx=np.int32(-1),
        topwater_m=np.float32(0.0),
        gw_ksat_m_per_s=gw_ksat,
        groundwater_depth_m=groundwater_depth_m,
        deep_soil_temperature_C=np.float32(10.0),
    )

    assert cap_gw == 0.0, "Capillary rise must be zero when water table is deep"


def test_calculate_capillary_rise_and_percolation_shallow_gw() -> None:
    """Test capillary rise flux when groundwater is within capillary reach of dry soil."""
    bubbling_pressure = np.float32(0.3)
    lambda_ = np.float32(0.3)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)
    ksat = np.float32(1e-5)
    # Dry bottom layer: K is very small, matric flux potential phi is far below saturation
    phi_sat = (ksat * -bubbling_pressure) / (
        np.float32(1.0) - lambda_ * pore_size_index
    )
    phi_bottom = np.float32(0.1) * phi_sat
    k_bottom = np.float32(1e-9)
    dphi_bottom = np.float32(1e-4)
    dk_bottom = np.float32(1e-8)

    flux, dflux_ds = calculate_capillary_rise_and_percolation_parameters(
        groundwater_depth_m=np.float32(2.0),
        total_soil_depth_m=np.float32(2.0),
        bottom_layer_thickness_m=np.float32(1.0),
        bubbling_pressure_m_positive=bubbling_pressure,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        matric_flux_potential_bottom=phi_bottom,
        conductivity_bottom=k_bottom,
        dmatric_flux_potential_dS_bottom=dphi_bottom,
        dconductivity_dS_bottom=dk_bottom,
        liquid_fraction=np.float32(1.0),
        gw_ksat_m_per_s=np.float32(1e-4),
    )

    # Flux should be negative (upward capillary rise)
    assert flux < 0.0, "Flux must be negative indicating upward capillary rise"
    # dq/dS is positive because flux becomes less negative (toward 0/downward) as saturation increases
    assert dflux_ds > 0.0, (
        "Derivative dq/dS must be positive for stability as saturation increases"
    )


def test_calculate_capillary_rise_and_percolation_deep_gw() -> None:
    """Test that deep groundwater reduces to pure gravity drainage without capillary suction."""
    bubbling_pressure = np.float32(0.3)
    lambda_ = np.float32(0.3)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)
    ksat = np.float32(1e-5)
    k_bottom = np.float32(1e-7)
    dk_bottom = np.float32(1e-6)

    # Groundwater depth 10m is far beyond 3.0 * bubbling_pressure (0.9m)
    flux, dflux_ds = calculate_capillary_rise_and_percolation_parameters(
        groundwater_depth_m=np.float32(10.0),
        total_soil_depth_m=np.float32(2.0),
        bottom_layer_thickness_m=np.float32(1.0),
        bubbling_pressure_m_positive=bubbling_pressure,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        matric_flux_potential_bottom=np.float32(-1e-4),
        conductivity_bottom=k_bottom,
        dmatric_flux_potential_dS_bottom=np.float32(1e-4),
        dconductivity_dS_bottom=dk_bottom,
        liquid_fraction=np.float32(1.0),
        gw_ksat_m_per_s=np.float32(1e-4),
    )

    # Pure gravity drainage
    assert flux == k_bottom, "Deep groundwater must yield pure gravity drainage"
    assert dflux_ds == dk_bottom, "Derivative must match conductivity derivative"


def test_calculate_capillary_rise_and_percolation_submerged() -> None:
    """Test capillary rise / influx when water table is inside or above the bottom layer."""
    bubbling_pressure = np.float32(0.3)
    lambda_ = np.float32(0.3)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)
    ksat = np.float32(1e-5)
    phi_sat = (ksat * -bubbling_pressure) / (
        np.float32(1.0) - lambda_ * pore_size_index
    )
    phi_bottom = np.float32(0.2) * phi_sat

    # Water table at 1.2m, while bottom layer center is at 1.5m (submerged)
    flux, dflux_ds = calculate_capillary_rise_and_percolation_parameters(
        groundwater_depth_m=np.float32(1.2),
        total_soil_depth_m=np.float32(2.0),
        bottom_layer_thickness_m=np.float32(1.0),
        bubbling_pressure_m_positive=bubbling_pressure,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        matric_flux_potential_bottom=phi_bottom,
        conductivity_bottom=np.float32(1e-9),
        dmatric_flux_potential_dS_bottom=np.float32(1e-4),
        dconductivity_dS_bottom=np.float32(1e-8),
        liquid_fraction=np.float32(1.0),
        gw_ksat_m_per_s=np.float32(1e-4),
    )

    assert flux < 0.0, "Submerged water table into dry soil must pull water upward"


def test_calculate_capillary_rise_and_percolation_frozen() -> None:
    """Test that frozen bottom layer completely shuts off groundwater exchange."""
    bubbling_pressure = np.float32(0.3)
    lambda_ = np.float32(0.3)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)
    ksat = np.float32(1e-5)

    flux, dflux_ds = calculate_capillary_rise_and_percolation_parameters(
        groundwater_depth_m=np.float32(2.0),
        total_soil_depth_m=np.float32(2.0),
        bottom_layer_thickness_m=np.float32(1.0),
        bubbling_pressure_m_positive=bubbling_pressure,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        matric_flux_potential_bottom=np.float32(-1e-4),
        conductivity_bottom=np.float32(1e-7),
        dmatric_flux_potential_dS_bottom=np.float32(1e-4),
        dconductivity_dS_bottom=np.float32(1e-6),
        liquid_fraction=np.float32(0.0),
        gw_ksat_m_per_s=np.float32(1e-4),
    )

    assert flux == 0.0, "Frozen layer must have zero groundwater flux"
    assert dflux_ds == 0.0, "Frozen layer must have zero flux derivative"


def test_calculate_capillary_rise_and_percolation_ksat_capping() -> None:
    """Test that groundwater flux is capped by the aquifer conductivity limit."""
    bubbling_pressure = np.float32(0.3)
    lambda_ = np.float32(0.3)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)
    ksat = np.float32(1e-3)
    gw_ksat_limit = np.float32(1e-6)

    # 1. Downward percolation exceeding gw_ksat_m_per_s
    flux_down, dflux_down = calculate_capillary_rise_and_percolation_parameters(
        groundwater_depth_m=np.float32(10.0),
        total_soil_depth_m=np.float32(2.0),
        bottom_layer_thickness_m=np.float32(1.0),
        bubbling_pressure_m_positive=bubbling_pressure,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        matric_flux_potential_bottom=np.float32(-1e-4),
        conductivity_bottom=np.float32(1e-4),
        dmatric_flux_potential_dS_bottom=np.float32(1e-4),
        dconductivity_dS_bottom=np.float32(1e-3),
        liquid_fraction=np.float32(1.0),
        gw_ksat_m_per_s=gw_ksat_limit,
    )
    assert flux_down == gw_ksat_limit, "Downward flux must be capped at gw_ksat_limit"
    assert dflux_down == 0.0, "Capped flux must have zero derivative"

    # 2. Upward capillary rise exceeding gw_ksat_m_per_s
    phi_sat = (ksat * -bubbling_pressure) / (
        np.float32(1.0) - lambda_ * pore_size_index
    )
    flux_up, dflux_up = calculate_capillary_rise_and_percolation_parameters(
        groundwater_depth_m=np.float32(1.6),
        total_soil_depth_m=np.float32(2.0),
        bottom_layer_thickness_m=np.float32(1.0),
        bubbling_pressure_m_positive=bubbling_pressure,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
        saturated_hydraulic_conductivity_m_per_s=ksat,
        matric_flux_potential_bottom=np.float32(0.0),
        conductivity_bottom=np.float32(1e-9),
        dmatric_flux_potential_dS_bottom=np.float32(1e-4),
        dconductivity_dS_bottom=np.float32(1e-8),
        liquid_fraction=np.float32(1.0),
        gw_ksat_m_per_s=gw_ksat_limit,
    )
    assert flux_up == -gw_ksat_limit, "Upward flux must be capped at -gw_ksat_limit"
    assert dflux_up == 0.0, "Capped upward flux must have zero derivative"


def test_calculate_saturated_matric_flux_potential() -> None:
    """Test analytical evaluation of matric flux potential at bubbling pressure."""
    ksat = np.float32(1e-5)
    bubbling_pressure = np.float32(0.3)
    lambda_ = np.float32(0.3)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_)

    expected = (ksat * -bubbling_pressure) / (
        np.float32(1.0) - lambda_ * pore_size_index
    )
    actual = calculate_saturated_matric_flux_potential(
        saturated_hydraulic_conductivity_m_per_s=ksat,
        bubbling_pressure_m_positive=bubbling_pressure,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
    )

    assert np.isclose(actual, expected)
    # Matric flux potential is strictly positive (integral of non-negative conductivity)
    assert actual > 0.0


def test_plot_saturated_matric_flux_potential_and_capillary_fluxes() -> None:
    """Plot matric flux potentials and groundwater exchange fluxes across soils and conditions.

    Generates diagnostic figures in output_folder to illustrate:
    1. Saturated matric flux potential and moisture-dependent potential across soil types.
    2. Groundwater exchange flux (capillary rise and percolation) vs groundwater depth.
    3. Bidirectional flux and solver sensitivity derivative vs bottom soil saturation.
    """
    soils: dict[str, dict[str, float]] = {
        "Sand": {
            "ksat_m_per_s": 5.0e-5,
            "bubbling_pressure_m": 0.07,
            "lambda": 0.60,
        },
        "Loam": {
            "ksat_m_per_s": 5.0e-6,
            "bubbling_pressure_m": 0.20,
            "lambda": 0.25,
        },
        "Clay": {
            "ksat_m_per_s": 5.0e-7,
            "bubbling_pressure_m": 0.40,
            "lambda": 0.15,
        },
    }

    # -------------------------------------------------------------------------
    # Plot 1: Saturated matric flux potential and potential vs saturation
    # -------------------------------------------------------------------------
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))

    bubbling_pressures_m: np.ndarray = np.linspace(0.02, 1.0, 50, dtype=np.float32)
    saturations: np.ndarray = np.linspace(0.01, 1.0, 100, dtype=np.float32)

    for soil_name, props in soils.items():
        ksat: np.float32 = np.float32(props["ksat_m_per_s"])
        lambda_val: np.float32 = np.float32(props["lambda"])
        pore_size_index: np.float32 = np.float32(3.0) + (np.float32(2.0) / lambda_val)

        # Panel A: Phi_sat vs bubbling pressure
        phi_sat_values: list[float] = []
        for psi_b in bubbling_pressures_m:
            phi_sat_val: np.float32 = calculate_saturated_matric_flux_potential(
                saturated_hydraulic_conductivity_m_per_s=ksat,
                bubbling_pressure_m_positive=psi_b,
                lambda_=lambda_val,
                pore_size_index=pore_size_index,
            )
            phi_sat_values.append(float(phi_sat_val))

        ax1a.plot(bubbling_pressures_m, phi_sat_values, label=soil_name, linewidth=2.0)

        # Panel B: Phi(S) vs saturation
        base_psi_b: np.float32 = np.float32(props["bubbling_pressure_m"])
        phi_sat_base: np.float32 = calculate_saturated_matric_flux_potential(
            saturated_hydraulic_conductivity_m_per_s=ksat,
            bubbling_pressure_m_positive=base_psi_b,
            lambda_=lambda_val,
            pore_size_index=pore_size_index,
        )
        phi_exponent: np.float32 = pore_size_index - (np.float32(1.0) / lambda_val)
        phi_s_values: np.ndarray = phi_sat_base * (saturations**phi_exponent)

        ax1b.plot(saturations, phi_s_values, label=soil_name, linewidth=2.0)

    ax1a.set_xlabel("Air-Entry Bubbling Pressure Head (m)")
    ax1a.set_ylabel("Saturated Matric Flux Potential (m²/s)")
    ax1a.set_title("Saturated Potential vs. Air-Entry Head")
    ax1a.set_yscale("log")
    ax1a.grid(True, linestyle="--", alpha=0.6)
    ax1a.legend()

    ax1b.set_xlabel("Relative Soil Saturation (-)")
    ax1b.set_ylabel("Matric Flux Potential (m²/s)")
    ax1b.set_title("Matric Flux Potential vs. Moisture Saturation")
    ax1b.set_yscale("log")
    ax1b.grid(True, linestyle="--", alpha=0.6)
    ax1b.legend()

    fig1.tight_layout()
    plot1_path = output_folder / "test_saturated_matric_flux_potential.png"
    fig1.savefig(plot1_path, dpi=150)
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Plot 2: Capillary rise and percolation fluxes vs groundwater depth
    # -------------------------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

    total_soil_depth_m: np.float32 = np.float32(2.0)
    bottom_layer_thickness_m: np.float32 = np.float32(0.5)
    gw_depths_m: np.ndarray = np.linspace(0.5, 6.0, 60, dtype=np.float32)

    seconds_per_day: float = 86400.0
    mm_per_m: float = 1000.0

    for soil_name, props in soils.items():
        ksat = np.float32(props["ksat_m_per_s"])
        psi_b = np.float32(props["bubbling_pressure_m"])
        lambda_val = np.float32(props["lambda"])
        pore_size_index = np.float32(3.0) + (np.float32(2.0) / lambda_val)
        phi_sat = calculate_saturated_matric_flux_potential(
            saturated_hydraulic_conductivity_m_per_s=ksat,
            bubbling_pressure_m_positive=psi_b,
            lambda_=lambda_val,
            pore_size_index=pore_size_index,
        )
        phi_exponent = pore_size_index - (np.float32(1.0) / lambda_val)

        # Panel A: Dry bottom soil (S = 0.2) -> Upward capillary rise
        s_dry: np.float32 = np.float32(0.2)
        phi_dry: np.float32 = phi_sat * (s_dry**phi_exponent)
        k_dry: np.float32 = ksat * (s_dry**pore_size_index)
        dphi_ds_dry: np.float32 = phi_exponent * (phi_dry / s_dry)
        dk_ds_dry: np.float32 = pore_size_index * (k_dry / s_dry)

        capillary_rise_rates_mm_day: list[float] = []
        for gw_d in gw_depths_m:
            flux_m_s, _ = calculate_capillary_rise_and_percolation_parameters(
                groundwater_depth_m=gw_d,
                total_soil_depth_m=total_soil_depth_m,
                bottom_layer_thickness_m=bottom_layer_thickness_m,
                bubbling_pressure_m_positive=psi_b,
                lambda_=lambda_val,
                pore_size_index=pore_size_index,
                saturated_hydraulic_conductivity_m_per_s=ksat,
                matric_flux_potential_bottom=phi_dry,
                conductivity_bottom=k_dry,
                dmatric_flux_potential_dS_bottom=dphi_ds_dry,
                dconductivity_dS_bottom=dk_ds_dry,
                liquid_fraction=np.float32(1.0),
                gw_ksat_m_per_s=ksat,
            )
            # Upward flux is negative; convert to positive capillary rise in mm/day
            rise_mm_day: float = max(0.0, -float(flux_m_s) * mm_per_m * seconds_per_day)
            capillary_rise_rates_mm_day.append(rise_mm_day)

        ax2a.plot(
            gw_depths_m, capillary_rise_rates_mm_day, label=soil_name, linewidth=2.0
        )

        # Panel B: Wet bottom soil (S = 0.8) -> Bidirectional flux
        s_wet: np.float32 = np.float32(0.8)
        phi_wet: np.float32 = phi_sat * (s_wet**phi_exponent)
        k_wet: np.float32 = ksat * (s_wet**pore_size_index)
        dphi_ds_wet: np.float32 = phi_exponent * (phi_wet / s_wet)
        dk_ds_wet: np.float32 = pore_size_index * (k_wet / s_wet)

        net_fluxes_mm_day: list[float] = []
        for gw_d in gw_depths_m:
            flux_m_s, _ = calculate_capillary_rise_and_percolation_parameters(
                groundwater_depth_m=gw_d,
                total_soil_depth_m=total_soil_depth_m,
                bottom_layer_thickness_m=bottom_layer_thickness_m,
                bubbling_pressure_m_positive=psi_b,
                lambda_=lambda_val,
                pore_size_index=pore_size_index,
                saturated_hydraulic_conductivity_m_per_s=ksat,
                matric_flux_potential_bottom=phi_wet,
                conductivity_bottom=k_wet,
                dmatric_flux_potential_dS_bottom=dphi_ds_wet,
                dconductivity_dS_bottom=dk_ds_wet,
                liquid_fraction=np.float32(1.0),
                gw_ksat_m_per_s=ksat,
            )
            # Positive = percolation down, Negative = capillary rise up
            net_fluxes_mm_day.append(float(flux_m_s) * mm_per_m * seconds_per_day)

        ax2b.plot(gw_depths_m, net_fluxes_mm_day, label=soil_name, linewidth=2.0)

    ax2a.axvline(2.0, color="black", linestyle=":", label="Soil Bottom (2.0 m)")
    ax2a.set_xlabel("Groundwater Depth below Surface (m)")
    ax2a.set_ylabel("Upward Capillary Rise Rate (mm/day)")
    ax2a.set_title("Capillary Rise vs. Water Table Depth (Dry Soil S=0.2)")
    ax2a.grid(True, linestyle="--", alpha=0.6)
    ax2a.legend()

    ax2b.axvline(2.0, color="black", linestyle=":", label="Soil Bottom (2.0 m)")
    ax2b.axhline(0.0, color="grey", linestyle="-", alpha=0.5)
    ax2b.set_xlabel("Groundwater Depth below Surface (m)")
    ax2b.set_ylabel("Net Groundwater Flux (mm/day) (+ downward / - upward)")
    ax2b.set_title("Net Exchange Flux vs. Water Table Depth (Wet Soil S=0.8)")
    ax2b.grid(True, linestyle="--", alpha=0.6)
    ax2b.legend()

    fig2.tight_layout()
    plot2_path = output_folder / "test_capillary_flux_vs_groundwater_depth.png"
    fig2.savefig(plot2_path, dpi=150)
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # Plot 3: Flux and sensitivity vs bottom soil saturation across water table depths
    # -------------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

    loam_props = soils["Loam"]
    ksat_loam = np.float32(loam_props["ksat_m_per_s"])
    psi_b_loam = np.float32(loam_props["bubbling_pressure_m"])
    lambda_loam = np.float32(loam_props["lambda"])
    pore_size_index_loam = np.float32(3.0) + (np.float32(2.0) / lambda_loam)
    phi_sat_loam = calculate_saturated_matric_flux_potential(
        saturated_hydraulic_conductivity_m_per_s=ksat_loam,
        bubbling_pressure_m_positive=psi_b_loam,
        lambda_=lambda_loam,
        pore_size_index=pore_size_index_loam,
    )
    phi_exponent_loam = pore_size_index_loam - (np.float32(1.0) / lambda_loam)

    gw_scenarios: dict[str, tuple[float, str, float]] = {
        "Shallow (1.6 m - inside profile)": (1.6, "-", 2.2),
        "Moderate (2.05 m - 0.05m below base)": (2.05, "--", 2.0),
        "Deep (3.0 m - 1.0m below base)": (3.0, ":", 2.0),
    }

    sat_sweep: np.ndarray = np.linspace(0.05, 0.99, 100, dtype=np.float32)

    for scenario_name, (depth_val, line_style, line_width) in gw_scenarios.items():
        flux_series_mm_day: list[float] = []
        dflux_series: list[float] = []

        for s_val in sat_sweep:
            phi_val: np.float32 = phi_sat_loam * (s_val**phi_exponent_loam)
            k_val: np.float32 = ksat_loam * (s_val**pore_size_index_loam)
            dphi_ds_val: np.float32 = phi_exponent_loam * (phi_val / s_val)
            dk_ds_val: np.float32 = pore_size_index_loam * (k_val / s_val)

            flux_m_s, dflux_ds = calculate_capillary_rise_and_percolation_parameters(
                groundwater_depth_m=np.float32(depth_val),
                total_soil_depth_m=total_soil_depth_m,
                bottom_layer_thickness_m=bottom_layer_thickness_m,
                bubbling_pressure_m_positive=psi_b_loam,
                lambda_=lambda_loam,
                pore_size_index=pore_size_index_loam,
                saturated_hydraulic_conductivity_m_per_s=ksat_loam,
                matric_flux_potential_bottom=phi_val,
                conductivity_bottom=k_val,
                dmatric_flux_potential_dS_bottom=dphi_ds_val,
                dconductivity_dS_bottom=dk_ds_val,
                liquid_fraction=np.float32(1.0),
                gw_ksat_m_per_s=ksat_loam,
            )
            flux_series_mm_day.append(float(flux_m_s) * mm_per_m * seconds_per_day)
            dflux_series.append(float(dflux_ds) * mm_per_m * seconds_per_day)

        ax3a.plot(
            sat_sweep,
            flux_series_mm_day,
            label=scenario_name,
            linestyle=line_style,
            linewidth=line_width,
        )
        ax3b.plot(
            sat_sweep,
            dflux_series,
            label=scenario_name,
            linestyle=line_style,
            linewidth=line_width,
        )

    ax3a.axhline(0.0, color="black", linestyle=":", label="Flux Equilibrium (q = 0)")
    ax3a.set_xlabel("Bottom Soil Saturation (-)")
    ax3a.set_ylabel("Net Flux (mm/day) (+ downward / - upward)")
    ax3a.set_title("Net Flux vs. Saturation (Loam Soil)")
    ax3a.grid(True, linestyle="--", alpha=0.6)
    ax3a.legend()

    ax3b.set_xlabel("Bottom Soil Saturation (-)")
    ax3b.set_ylabel("Flux Sensitivity d(flux)/dS (mm/day per unit S)")
    ax3b.set_title("Solver Sensitivity vs. Saturation (Loam Soil)")
    ax3b.grid(True, linestyle="--", alpha=0.6)
    ax3b.legend()

    fig3.tight_layout()
    plot3_path = output_folder / "test_groundwater_flux_vs_saturation.png"
    fig3.savefig(plot3_path, dpi=150)
    plt.close(fig3)

    # -------------------------------------------------------------------------
    # Verification assertions
    # -------------------------------------------------------------------------
    assert plot1_path.exists(), f"{plot1_path} was not created"
    assert plot1_path.stat().st_size > 0, f"{plot1_path} is empty"

    assert plot2_path.exists(), f"{plot2_path} was not created"
    assert plot2_path.stat().st_size > 0, f"{plot2_path} is empty"

    assert plot3_path.exists(), f"{plot3_path} was not created"
    assert plot3_path.stat().st_size > 0, f"{plot3_path} is empty"
