"""Tests for Darcy flow redistribution in GEB."""

import numpy as np
import pytest

from geb.hydrology.landsurface.redistribution import (
    distribute_soil_water_ross,
)


def test_distribute_soil_water_ross_basic() -> None:
    """Test basic downward flow in a 6-layer soil column using Ross scheme."""
    n_layers = 6
    dt_s = np.float32(3600.0)
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
    ksat = np.full(n_layers, 0.01, dtype=np.float32)  # m/h
    alpha = np.full(n_layers, 0.015, dtype=np.float32)  # 1/cm
    n_vg = np.full(n_layers, 1.3, dtype=np.float32)

    # Infiltration (none)
    active_layer_idx = np.int32(0)
    topwater_m = np.float32(0.0)

    # Run redistribution
    w_new, e_new, perc, rise = distribute_soil_water_ross(
        dt_s=dt_s,
        water_content_m=w.copy(),
        water_content_residual_m=wr,
        water_content_saturated_m=ws,
        soil_enthalpy_J_per_m2=enth.copy(),
        solid_heat_capacity_J_per_m2_K=hc,
        saturated_hydraulic_conductivity_m_per_hour=ksat,
        soil_layer_height_m=soil_layer_height_m,
        alpha_vg=alpha,
        n_vg=n_vg,
        green_ampt_active_layer_idx=active_layer_idx,
        topwater_m=topwater_m,
    )

    # Check mass balance
    assert np.float32(np.sum(w_new)) == pytest.approx(np.float32(np.sum(w)), abs=1e-7)
    # Check that water moved (downwards)
    assert w_new[0] < w[0]
    assert w_new[-1] > w[-1]


def test_distribute_soil_water_ross_active_layer_masking() -> None:
    """Test that layers above the green_ampt_active_layer_idx do not lose water."""
    n_layers = 6
    dt_s = np.float32(3600.0)
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
    ksat = np.full(n_layers, 0.1, dtype=np.float32)
    alpha = np.full(n_layers, 0.02, dtype=np.float32)
    n_vg = np.full(n_layers, 1.5, dtype=np.float32)

    w_new, _, _, _ = distribute_soil_water_ross(
        dt_s,
        w.copy(),
        wr,
        ws,
        enth.copy(),
        hc,
        ksat,
        soil_layer_height_m,
        alpha,
        n_vg,
        active_idx,
        0.0,
    )

    # Layers above active_idx must NOT change
    assert w_new[0] == pytest.approx(w[0], abs=1e-12)
    assert w_new[1] == pytest.approx(w[1], abs=1e-12)

    # Check mass balance
    assert np.float32(np.sum(w_new)) == pytest.approx(np.float32(np.sum(w)), abs=1e-7)


def test_distribute_soil_water_ross_extreme_dry_wet() -> None:
    """Test Ross scheme with a mix of extremely dry and extremely wet layers."""
    n_layers = 6
    dt_s = np.float32(3600.0)
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
    ksat = np.full(n_layers, 0.1, dtype=np.float32)
    alpha = np.full(n_layers, 0.02, dtype=np.float32)
    n_vg = np.full(n_layers, 1.5, dtype=np.float32)

    w_new, _, _, _ = distribute_soil_water_ross(
        dt_s,
        w.copy(),
        wr,
        ws,
        enth.copy(),
        hc,
        ksat,
        soil_layer_height_m,
        alpha,
        n_vg,
        0,
        0.0,
    )

    # Mass balance
    assert np.float32(np.sum(w_new)) == pytest.approx(np.float32(np.sum(w)), abs=1e-7)
    # Water should have moved from saturated layers to dry layers
    assert w_new[0] < w[0]
    assert w_new[5] < w[5]
    assert np.any(w_new[1:5] > w[1:5])


def test_distribute_soil_water_ross_frozen_barrier() -> None:
    """Test Ross scheme with a frozen layer acting as a hydraulic barrier."""
    n_layers = 6
    dt_s = np.float32(3600.0)
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
    ksat = np.full(n_layers, 0.1, dtype=np.float32)
    alpha = np.full(n_layers, 0.02, dtype=np.float32)
    n_vg = np.full(n_layers, 1.5, dtype=np.float32)

    w_new, _, _, _ = distribute_soil_water_ross(
        dt_s,
        w.copy(),
        wr,
        ws,
        enth.copy(),
        hc,
        ksat,
        soil_layer_height_m,
        alpha,
        n_vg,
        0,
        0.0,
    )

    # Mass balance
    assert np.float32(np.sum(w_new)) == pytest.approx(np.float32(np.sum(w)), abs=1e-7)

    # Layer 2 is frozen, it should act as a barrier.
    # We allow flux into a frozen layer but NOT out of it (as it has 0 liquid fraction).
    # So water can move from 1 to 2, but 2 cannot pass to 3.
    # We use a reasonably tight tolerance for float32 implicit solver.
    assert w_new[3] == pytest.approx(w[3], abs=1e-6)
