"""Tests for scalar soil functions in GEB."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from geb.hydrology.landcovers import (
    NON_PADDY_IRRIGATED,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
)
from geb.hydrology.soil import (
    add_water_to_topwater_and_evaporate_open_water,
    calculate_sensible_heat_flux,
    calculate_spatial_infiltration_excess,
    get_bubbling_pressure,
    get_heat_capacity_solid_fraction,
    get_pore_size_index_brakensiek,
    get_pore_size_index_wosten,
    get_soil_moisture_at_pressure,
    get_soil_water_flow_parameters,
    infiltration,
    kv_brakensiek,
    kv_cosby,
    kv_wosten,
    rise_from_groundwater,
    solve_energy_balance_implicit_iterative,
    thetar_brakensiek,
    thetas_toth,
    thetas_wosten,
)

from ..testconfig import output_folder


def test_add_water_to_topwater_and_evaporate_open_water() -> None:
    """Test the scalar version of add_water_to_topwater_and_evaporate_open_water."""
    # Test different land use types
    test_cases = [
        (PADDY_IRRIGATED, 0.01, 0.0, 0.025, 0.05),  # paddy with evaporation
        (NON_PADDY_IRRIGATED, 0.01, 0.0, 0.025, 0.0),  # non-paddy, no evaporation
        (SEALED, 0.01, 0.0, 0.025, 0.0),  # sealed, no evaporation in this function
        (
            OPEN_WATER,
            0.01,
            0.0,
            0.025,
            0.0,
        ),  # open water, no evaporation in this function
    ]

    for land_use_type, infiltration, irrigation, et_ref, expected_evap in test_cases:
        topwater = np.float32(0.05)
        topwater_pre = topwater

        updated_topwater, open_water_evaporation = (
            add_water_to_topwater_and_evaporate_open_water(
                natural_available_water_infiltration_m=np.float32(infiltration),
                actual_irrigation_consumption_m=np.float32(irrigation),
                land_use_type=np.int32(land_use_type),
                reference_evapotranspiration_water_m=np.float32(et_ref),
                topwater_m=topwater,
            )
        )

        # Check water balance
        expected_topwater = (
            topwater_pre + infiltration + irrigation - open_water_evaporation
        )
        assert abs(updated_topwater - expected_topwater) < 1e-6, (
            f"Water balance failed for land_use_type {land_use_type}"
        )

        # Check evaporation
        assert open_water_evaporation >= 0.0, (
            f"Negative evaporation for land_use_type {land_use_type}"
        )
        assert updated_topwater >= 0.0, (
            f"Negative topwater for land_use_type {land_use_type}"
        )

    # Test boundary conditions
    # Zero inputs
    topwater = np.float32(0.0)
    updated_topwater, open_water_evaporation = (
        add_water_to_topwater_and_evaporate_open_water(
            natural_available_water_infiltration_m=np.float32(0.0),
            actual_irrigation_consumption_m=np.float32(0.0),
            land_use_type=np.int32(PADDY_IRRIGATED),
            reference_evapotranspiration_water_m=np.float32(0.0),
            topwater_m=topwater,
        )
    )
    assert updated_topwater == 0.0
    assert open_water_evaporation == 0.0

    # Large ET exceeding topwater
    topwater = np.float32(0.01)
    updated_topwater, open_water_evaporation = (
        add_water_to_topwater_and_evaporate_open_water(
            natural_available_water_infiltration_m=np.float32(0.0),
            actual_irrigation_consumption_m=np.float32(0.0),
            land_use_type=np.int32(PADDY_IRRIGATED),
            reference_evapotranspiration_water_m=np.float32(
                1.0
            ),  # Much larger than topwater
            topwater_m=topwater,
        )
    )
    assert abs(updated_topwater - 0.0) < 1e-6  # All water evaporated
    assert abs(open_water_evaporation - 0.01) < 1e-6


def test_rise_from_groundwater() -> None:
    """Test the scalar version of rise_from_groundwater."""
    # Test case 1: No overflow
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    ws = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    capillary_rise = np.float32(0.05)

    w_pre = w.copy()
    runoff = rise_from_groundwater(w, ws, capillary_rise)

    # Bottom layer should have increased by capillary_rise
    assert abs(w[5] - (w_pre[5] + capillary_rise)) < 1e-6
    # No runoff since there's capacity
    assert runoff == 0.0

    # Test case 2: Overflow to runoff
    w = np.array(
        [0.29, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32
    )  # Top layer almost full
    ws = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    capillary_rise = np.float32(0.05)

    w_pre = w.copy()
    runoff = rise_from_groundwater(w, ws, capillary_rise)

    # Should have runoff
    assert runoff > 0.0
    # Top layer should be at capacity
    assert abs(w[0] - ws[0]) < 1e-6

    # Test case 3: Zero capillary rise
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    ws = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    capillary_rise = np.float32(0.0)

    w_pre = w.copy()
    runoff = rise_from_groundwater(w, ws, capillary_rise)

    # No change in soil water
    assert np.allclose(w, w_pre)
    assert runoff == 0.0

    # Test case 4: Capillary rise exceeding total soil capacity
    w = np.array(
        [0.29, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32
    )  # All layers almost full
    ws = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    capillary_rise = np.float32(0.1)  # More than remaining capacity

    runoff = rise_from_groundwater(w, ws, capillary_rise)

    # Should have significant runoff
    assert runoff > 0.05  # At least the excess
    # All layers should be at capacity
    assert np.allclose(w, ws)

    # Test case 5: Soil already saturated
    w = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    ws = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    capillary_rise = np.float32(0.01)

    w_pre = w.copy()
    runoff = rise_from_groundwater(w, ws, capillary_rise)

    # All water should go to runoff
    assert abs(runoff - capillary_rise) < 1e-6
    # Soil water unchanged
    assert np.allclose(w, w_pre)


def test_infiltration() -> None:
    """Test the scalar infiltration function."""
    # Test case 1: Normal infiltration
    ws = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    # Define arrays for new signature
    bub_arr = np.full_like(ws, 100.0)
    h_arr = np.full_like(ws, 0.1)
    lam_arr = np.full_like(ws, 0.5)

    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    saturated_hydraulic_conductivity = np.full_like(w, np.float32(0.01))
    land_use_type = np.int32(NON_PADDY_IRRIGATED)
    soil_is_frozen = False
    topwater = np.float32(0.005)

    w_pre = w.copy()
    topwater_pre = topwater

    (
        updated_topwater,
        direct_runoff,
        groundwater_recharge,
        infiltration_amount,
        wetting_front,
        _,
        _,
        _,
    ) = infiltration(
        ws,
        np.zeros_like(ws),
        saturated_hydraulic_conductivity,
        np.float32(0.0),  # groundwater_toplayer_conductivity_m_per_timestep
        land_use_type,
        soil_is_frozen,
        w,
        topwater,
        np.float32(0.0),  # capillary_rise_from_groundwater_m
        np.float32(0.0),
        np.float32(0.1),  # wetting_front_suction_head_m
        np.float32(0.1),  # wetting_front_moisture_deficit
        np.int32(0),
        np.float32(0.1),
        bub_arr,
        h_arr,
        lam_arr,
    )

    # Check that some water was infiltrated
    assert infiltration_amount > 0.0
    # Check groundwater recharge is 0
    assert groundwater_recharge == 0.0
    # Check water balance
    total_water_added = infiltration_amount + direct_runoff
    assert abs(total_water_added - topwater_pre) < 1e-6
    # Check that soil water increased
    assert np.sum(w) > np.sum(w_pre)

    # Test case 2: Frozen soil - no infiltration
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    topwater = np.float32(0.005)
    soil_is_frozen = True

    (
        updated_topwater,
        direct_runoff,
        groundwater_recharge,
        infiltration_amount,
        wetting_front,
        _,
        _,
        _,
    ) = infiltration(
        ws,
        np.zeros_like(ws),
        saturated_hydraulic_conductivity,
        np.float32(0.0),  # groundwater_toplayer_conductivity_m_per_timestep
        land_use_type,
        soil_is_frozen,
        w,
        topwater,
        np.float32(0.0),  # capillary_rise_from_groundwater_m
        np.float32(0.0),
        np.float32(0.1),  # wetting_front_suction_head_m
        np.float32(0.1),  # wetting_front_moisture_deficit
        np.int32(0),
        np.float32(0.1),
        bub_arr,
        h_arr,
        lam_arr,
    )

    # No infiltration on frozen soil
    assert infiltration_amount == 0.0
    assert groundwater_recharge == 0.0
    # All water should go to direct runoff
    assert direct_runoff == topwater_pre

    # Test case 3: Sealed area - no infiltration
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    topwater = np.float32(0.005)
    soil_is_frozen = False
    land_use_type = np.int32(SEALED)

    (
        updated_topwater,
        direct_runoff,
        groundwater_recharge,
        infiltration_amount,
        wetting_front,
        _,
        _,
        _,
    ) = infiltration(
        ws,
        np.zeros_like(ws),
        saturated_hydraulic_conductivity,
        np.float32(0.0),  # groundwater_toplayer_conductivity_m_per_timestep
        land_use_type,
        soil_is_frozen,
        w,
        topwater,
        np.float32(0.0),  # capillary_rise_from_groundwater_m
        np.float32(0.0),
        np.float32(0.1),  # wetting_front_suction_head_m
        np.float32(0.1),  # wetting_front_moisture_deficit
        np.int32(0),
        np.float32(0.1),
        bub_arr,
        h_arr,
        lam_arr,
    )

    # No infiltration on sealed areas
    assert infiltration_amount == 0.0
    assert groundwater_recharge == 0.0
    # All water should go to direct runoff
    assert direct_runoff == topwater_pre

    # Test case 4: Zero topwater
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    topwater = np.float32(0.0)
    soil_is_frozen = False
    land_use_type = np.int32(NON_PADDY_IRRIGATED)

    (
        updated_topwater,
        direct_runoff,
        groundwater_recharge,
        infiltration_amount,
        wetting_front,
        _,
        _,
        _,
    ) = infiltration(
        ws,
        np.zeros_like(ws),
        saturated_hydraulic_conductivity,
        np.float32(0.0),  # groundwater_toplayer_conductivity_m_per_timestep
        land_use_type,
        soil_is_frozen,
        w,
        topwater,
        np.float32(0.0),  # capillary_rise_from_groundwater_m
        np.float32(0.0),
        np.float32(0.1),  # wetting_front_suction_head_m
        np.float32(0.1),  # wetting_front_moisture_deficit
        np.int32(0),
        np.float32(0.1),
        bub_arr,
        h_arr,
        lam_arr,
    )

    # No infiltration possible
    assert infiltration_amount == 0.0
    assert groundwater_recharge == 0.0
    assert direct_runoff == 0.0
    # Soil water unchanged
    assert np.allclose(w, np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32))

    # Test case 5: Open water - no infiltration
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    topwater = np.float32(0.005)
    soil_is_frozen = False
    land_use_type = np.int32(OPEN_WATER)

    (
        updated_topwater,
        direct_runoff,
        groundwater_recharge,
        infiltration_amount,
        wetting_front,
        _,
        _,
        _,
    ) = infiltration(
        ws,
        np.zeros_like(ws),
        saturated_hydraulic_conductivity,
        np.float32(0.0),  # groundwater_toplayer_conductivity_m_per_timestep
        land_use_type,
        soil_is_frozen,
        w,
        topwater,
        np.float32(0.0),  # capillary_rise_from_groundwater_m
        np.float32(0.0),
        np.float32(0.1),  # wetting_front_suction_head_m
        np.float32(0.1),  # wetting_front_moisture_deficit
        np.int32(0),
        np.float32(0.1),
        bub_arr,
        h_arr,
        lam_arr,
    )

    # No infiltration on open water
    assert infiltration_amount == 0.0
    assert groundwater_recharge == 0.0
    # All water should go to direct runoff
    assert direct_runoff == topwater_pre

    # Test case 6: Paddy irrigated with ponding allowance
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    topwater = np.float32(0.1)  # More than ponding allowance
    soil_is_frozen = False
    land_use_type = np.int32(PADDY_IRRIGATED)

    (
        updated_topwater,
        direct_runoff,
        groundwater_recharge,
        infiltration_amount,
        wetting_front,
        _,
        _,
        _,
    ) = infiltration(
        ws,
        np.zeros_like(ws),
        saturated_hydraulic_conductivity,
        np.float32(0.0),  # groundwater_toplayer_conductivity_m_per_timestep
        land_use_type,
        soil_is_frozen,
        w,
        topwater,
        np.float32(0.0),  # capillary_rise_from_groundwater_m
        np.float32(0.0),
        np.float32(0.1),  # wetting_front_suction_head_m
        np.float32(0.1),  # wetting_front_moisture_deficit
        np.int32(0),
        np.float32(0.1),
        bub_arr,
        h_arr,
        lam_arr,
    )

    # Should infiltrate up to capacity, then pond up to 0.05m before runoff
    assert infiltration_amount > 0.0
    assert groundwater_recharge == 0.0
    # Direct runoff should be topwater - 0.05 (ponding allowance)
    remaining_after_infil = topwater - infiltration_amount
    expected_runoff = max(0.0, remaining_after_infil - 0.05)
    assert abs(direct_runoff - expected_runoff) < 1e-6

    # Test case 7: Soil layers at different saturation levels
    ws = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    w = np.array(
        [0.3, 0.3, 0.1, 0.1, 0.1, 0.1], dtype=np.float32
    )  # Top layers saturated
    topwater = np.float32(0.005)
    soil_is_frozen = False
    land_use_type = np.int32(NON_PADDY_IRRIGATED)

    w_pre = w.copy()
    (
        updated_topwater,
        direct_runoff,
        groundwater_recharge,
        infiltration_amount,
        wetting_front,
        _,
        _,
        _,
    ) = infiltration(
        ws,
        np.zeros_like(ws),
        saturated_hydraulic_conductivity,
        np.float32(0.0),  # groundwater_toplayer_conductivity_m_per_timestep
        land_use_type,
        soil_is_frozen,
        w,
        topwater,
        np.float32(0.0),  # capillary_rise_from_groundwater_m
        np.float32(0.0),
        np.float32(0.1),  # wetting_front_suction_head_m
        np.float32(0.1),  # wetting_front_moisture_deficit
        np.int32(0),
        np.float32(0.1),
        bub_arr,
        h_arr,
        lam_arr,
    )

    # Infiltration should happen into lower unsaturated layers
    assert infiltration_amount > 0.0
    assert groundwater_recharge == 0.0


def test_infiltration_groundwater_recharge_is_capped_by_groundwater_conductivity() -> (
    None
):
    """Test that infiltration() computes groundwater recharge with a conductivity cap.

    Notes:
        This checks only the bottom-layer drainage component implemented in
        `geb.hydrology.soil.infiltration`.
    """
    ws = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    wres = np.zeros_like(ws)
    soil_layer_height_m = np.full_like(ws, 0.1)
    bubbling_pressure_cm = np.full_like(ws, 100.0)
    lambda_param = np.full_like(ws, 0.5)
    # Use very high soil conductivity so the Green-Ampt rate limit does not bind;
    # the groundwater conductivity cap should be the limiting factor.
    saturated_hydraulic_conductivity = np.full_like(ws, np.float32(10.0))

    # Fully saturated profile with wetting front at the bottom: no storage available,
    # so incoming topwater can only pass to groundwater (capped) or become runoff.
    w = ws.copy()
    topwater_m = np.float32(0.06)
    groundwater_toplayer_conductivity_m_per_timestep = np.float32(0.02)
    total_soil_depth_m = np.float32(np.sum(soil_layer_height_m))

    (
        _,
        direct_runoff,
        groundwater_recharge,
        infiltration_amount,
        _,
        _,
        _,
        _,
    ) = infiltration.py_func(
        ws,
        wres,
        saturated_hydraulic_conductivity,
        groundwater_toplayer_conductivity_m_per_timestep,
        np.int32(NON_PADDY_IRRIGATED),
        False,
        w,
        topwater_m,
        np.float32(0.0),
        total_soil_depth_m,
        np.float32(10.0),
        np.float32(0.1),
        np.int32(0),
        np.float32(0.1),
        bubbling_pressure_cm,
        soil_layer_height_m,
        lambda_param,
    )

    assert infiltration_amount == 0.0
    assert (
        abs(groundwater_recharge - groundwater_toplayer_conductivity_m_per_timestep)
        < 1e-6
    )
    assert abs(direct_runoff - (topwater_m - groundwater_recharge)) < 1e-6

    # If there is capillary rise, groundwater recharge should be suppressed.
    w = ws.copy()
    (
        _,
        direct_runoff,
        groundwater_recharge_with_rise,
        infiltration_amount,
        _,
        _,
        _,
        _,
    ) = infiltration.py_func(
        ws,
        wres,
        saturated_hydraulic_conductivity,
        groundwater_toplayer_conductivity_m_per_timestep,
        np.int32(NON_PADDY_IRRIGATED),
        False,
        w,
        topwater_m,
        np.float32(0.001),
        total_soil_depth_m,
        np.float32(10.0),
        np.float32(0.1),
        np.int32(0),
        np.float32(0.1),
        bubbling_pressure_cm,
        soil_layer_height_m,
        lambda_param,
    )
    assert infiltration_amount == 0.0
    assert groundwater_recharge_with_rise == 0.0
    assert abs(direct_runoff - topwater_m) < 1e-6


def test_get_soil_water_flow_parameters() -> None:
    """Test get_soil_water_flow_parameters function."""
    # Test case 1: Saturated soil
    w = np.float32(0.3)  # saturated
    wres = np.float32(0.05)  # residual
    ws = np.float32(0.3)  # saturated
    lambda_ = np.float32(0.5)  # van Genuchten parameter
    ksat = np.float32(0.01)  # saturated hydraulic conductivity
    bubbling_pressure = np.float32(10.0)  # cm

    psi, k_unsat = get_soil_water_flow_parameters(
        w, wres, ws, lambda_, ksat, bubbling_pressure
    )

    # At saturation, psi should be close to 0 (no suction)
    assert abs(psi) < 1e-3, f"Expected psi close to 0 at saturation, got {psi}"
    # Unsaturated conductivity should equal saturated at saturation
    assert abs(k_unsat - ksat) < 1e-6, (
        f"Expected k_unsat={ksat} at saturation, got {k_unsat}"
    )

    # Test case 2: Residual soil water content
    w = np.float32(0.05)  # at residual
    psi, k_unsat = get_soil_water_flow_parameters(
        w, wres, ws, lambda_, ksat, bubbling_pressure
    )

    # At residual, psi should be very negative (high suction)
    assert psi < -1000, f"Expected high suction at residual, got {psi}"
    # Unsaturated conductivity should be very low
    assert k_unsat < ksat * 1e-6, (
        f"Expected very low k_unsat at residual, got {k_unsat}"
    )

    # Test case 3: Field capacity (typical value)
    w = np.float32(0.15)  # field capacity
    psi, k_unsat = get_soil_water_flow_parameters(
        w, wres, ws, lambda_, ksat, bubbling_pressure
    )

    # At field capacity, psi should be around -1 to -10 m
    assert -10 < psi < -0.1, f"Expected moderate suction at field capacity, got {psi}"
    # Unsaturated conductivity should be reduced but not extremely low
    assert k_unsat < ksat, f"Expected k_unsat < ksat at field capacity, got {k_unsat}"
    assert k_unsat > ksat * 1e-4, (
        f"Expected reasonable k_unsat at field capacity, got {k_unsat}"
    )

    # Test case 4: Boundary conditions - w exactly at wres
    w = np.float32(0.05)  # exactly residual
    psi, k_unsat = get_soil_water_flow_parameters(
        w, wres, ws, lambda_, ksat, bubbling_pressure
    )

    assert psi < 0, "Psi should be negative (suction)"
    assert k_unsat >= 0, "Unsaturated conductivity should be non-negative"

    # Test case 5: Boundary conditions - w exactly at ws
    w = np.float32(0.3)  # exactly saturated
    psi, k_unsat = get_soil_water_flow_parameters(
        w, wres, ws, lambda_, ksat, bubbling_pressure
    )

    assert psi >= -1e-3, "Psi should be close to 0 at saturation"
    assert k_unsat <= ksat, "Unsaturated conductivity should not exceed saturated"

    # Test case 6: Different van Genuchten parameters
    lambda_values = [0.2, 0.8, 1.5]
    for lambda_val in lambda_values:
        w = np.float32(0.15)
        psi, k_unsat = get_soil_water_flow_parameters(
            w, wres, ws, np.float32(lambda_val), ksat, bubbling_pressure
        )
        assert psi < 0, f"Psi should be negative for lambda={lambda_val}"
        assert 0 <= k_unsat <= ksat, (
            f"k_unsat should be between 0 and ksat for lambda={lambda_val}"
        )

    # Test case 7: Different bubbling pressures
    bubbling_pressures = [5.0, 20.0, 50.0]  # cm
    for bp in bubbling_pressures:
        w = np.float32(0.15)
        psi, k_unsat = get_soil_water_flow_parameters(
            w, wres, ws, lambda_, ksat, np.float32(bp)
        )
        assert psi < 0, f"Psi should be negative for bubbling_pressure={bp}"
        assert 0 <= k_unsat <= ksat, (
            f"k_unsat should be between 0 and ksat for bubbling_pressure={bp}"
        )

    # Test case 8: Very dry soil (effective saturation approaches 0)
    w = np.float32(0.0501)  # just above residual
    psi, k_unsat = get_soil_water_flow_parameters(
        w, wres, ws, lambda_, ksat, bubbling_pressure
    )

    # Should have high suction and very low conductivity
    assert psi < -100, f"Expected high suction for very dry soil, got {psi}"
    assert k_unsat < ksat * 1e-5, (
        f"Expected very low conductivity for very dry soil, got {k_unsat}"
    )


def test_get_soil_moisture_at_pressure() -> None:
    """Test get_soil_moisture_at_pressure function."""
    # Test with different soil types
    soils_data = [
        ("sand", 20.0, 0.4, 0.075, 2.5),
        ("silt", 40.0, 0.45, 0.15, 1.45),
        ("clay", 150.0, 0.50, 0.25, 1.2),
    ]

    for soil_name, bp, thetas, thetar, lambda_val in soils_data:
        # Test at a few capillary suction values
        capillary_suctions = np.array([-1.0, -10.0, -100.0, -1000.0], dtype=np.float32)
        bubbling_pressure_cm = np.full_like(capillary_suctions, bp)
        thetas_arr = np.full_like(capillary_suctions, thetas)
        thetar_arr = np.full_like(capillary_suctions, thetar)
        lambda_arr = np.full_like(capillary_suctions, lambda_val)

        soil_moisture = get_soil_moisture_at_pressure(
            capillary_suctions,
            bubbling_pressure_cm,
            thetas_arr,
            thetar_arr,
            lambda_arr,
        )

        # Check that moisture is between residual and saturated
        assert np.all(soil_moisture >= thetar_arr), (
            f"{soil_name}: moisture should be >= residual"
        )
        assert np.all(soil_moisture <= thetas_arr), (
            f"{soil_name}: moisture should be <= saturated"
        )

        # Check that moisture decreases with increasing suction (more negative)
        for i in range(len(soil_moisture) - 1):
            assert soil_moisture[i] >= soil_moisture[i + 1], (
                f"{soil_name}: moisture should decrease with increasing suction"
            )

    # Test edge cases
    # Very low suction (close to saturation)
    capillary_suction = np.array([-0.1], dtype=np.float32)
    bubbling_pressure_cm = np.array([20.0], dtype=np.float32)
    thetas_arr = np.array([0.4], dtype=np.float32)
    thetar_arr = np.array([0.075], dtype=np.float32)
    lambda_arr = np.array([2.5], dtype=np.float32)

    soil_moisture = get_soil_moisture_at_pressure(
        capillary_suction, bubbling_pressure_cm, thetas_arr, thetar_arr, lambda_arr
    )
    assert soil_moisture[0] > 0.35, "Should be close to saturated at low suction"

    # Very high suction (close to residual)
    capillary_suction = np.array([-20000.0], dtype=np.float32)
    soil_moisture = get_soil_moisture_at_pressure(
        capillary_suction, bubbling_pressure_cm, thetas_arr, thetar_arr, lambda_arr
    )
    assert soil_moisture[0] < 0.1, "Should be close to residual at high suction"


def test_infiltration_variable_runoff_integration() -> None:
    """Test infiltration with Variable Infiltration (PDM) runoff enabled."""
    # Setup inputs
    ws = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32)
    w = np.array(
        [50.0, 50.0, 50.0, 50.0, 50.0, 50.0], dtype=np.float32
    )  # 50% saturation
    saturated_hydraulic_conductivity = np.full_like(w, np.float32(10.0))  # High Ksat
    land_use_type = np.int32(NON_PADDY_IRRIGATED)
    soil_is_frozen = False
    topwater = np.float32(10.0)  # 10mm rain

    # Run infiltration using .py_func to use the python implementation with the updated global
    # With variable runoff (b=0.4), even if not saturated, there should be some runoff.
    # In the standard model, with Ksat=10 and topwater=10, and capacity=50,
    # potential_infiltration = min(10, 50) = 10.
    # infiltration = min(10, 10) = 10.
    # So standard model would have 0 runoff.

    w_variable = w.copy()

    # Define arrays for new signature
    bub_arr = np.full_like(ws, 100.0)
    h_arr = np.full_like(ws, 0.1)  # Assume layer height
    lam_arr = np.full_like(ws, 0.5)

    _, runoff_variable, _, infil_variable, _, _, _, _ = infiltration.py_func(
        ws,
        np.zeros_like(ws),
        saturated_hydraulic_conductivity,
        np.float32(0.0),
        land_use_type,
        soil_is_frozen,
        w_variable,
        topwater,
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.1),  # wetting_front_suction_head_m
        np.float32(0.1),  # wetting_front_moisture_deficit
        np.int32(0),
        np.float32(0.4),
        bub_arr,
        h_arr,
        lam_arr,
    )

    # Variable runoff is enabled, so runoff expected given spatial variability
    assert runoff_variable > 0.0
    assert abs(runoff_variable + infil_variable - topwater) < 1e-5


def test_infiltration_variable_capacity_limit() -> None:
    """Test that Variable infiltration is limited by infiltration capacity (Ksat)."""
    # Setup inputs
    ws = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32)
    w = np.array(
        [10.0, 50.0, 50.0, 50.0, 50.0, 50.0], dtype=np.float32
    )  # Low saturation (10/100 = 0.1)

    # Ksat is low (2.0), Topwater is high (10.0)
    # Variable infiltration would likely infiltrate most of the 10.0 if not limited,
    # because relative saturation is low (0.1).
    # Pass as array because infiltration expects array (for layers)
    saturated_hydraulic_conductivity = np.full_like(w, np.float32(2.0))

    land_use_type = np.int32(NON_PADDY_IRRIGATED)
    soil_is_frozen = False
    topwater = np.float32(10.0)
    variable_runoff_shape_beta = np.float32(0.4)

    # Define arrays for new signature
    bub_arr = np.full_like(ws, 100.0)
    h_arr = np.full_like(ws, 0.1)  # Assume layer height
    lam_arr = np.full_like(ws, 0.5)

    # Run infiltration using .py_func
    # Use a copy of w to prevent inplace modification affecting subsequent tests
    _, runoff, _, infiltration_amount, _, _, _, _ = infiltration.py_func(
        ws,
        np.zeros_like(ws),
        saturated_hydraulic_conductivity,
        np.float32(0.0),
        land_use_type,
        soil_is_frozen,
        w.copy(),
        topwater,
        np.float32(0.0),
        np.float32(0.0),
        np.float32(100.0),  # wetting_front_suction_head_m
        np.float32(0.1),  # wetting_front_moisture_deficit
        np.int32(0),
        variable_runoff_shape_beta,
        bub_arr,
        h_arr,
        lam_arr,
    )

    # With suction adjustment:
    # Relative saturation S = 10.0 / 100.0 = 0.1
    # Capacity = Ksat / S = 2.0 / 0.1 = 20.0
    # Incoming water = 10.0
    # Since 10.0 < 20.0, it is NOT limited by capacity.
    # It is determined effectively by the variable infiltration curve (8.947...)

    # We assert that infiltration significantly exceeds the old Ksat limit of 2.0
    assert infiltration_amount > 2.0

    # We can also check a case where we EXCEED the new higher capacity
    # For this, we'll use a lower Ksat to lower the capacity threshold
    # Ksat = 0.5. Capacity at start = 0.5 / 0.1 = 5.0.
    # We provide topwater = 50.0.
    # With sub-stepping (N=6):
    # Step 1: P=8.33, Cap=0.833. Infil=0.833. w increases -> Cap decreases.
    # Step 2: Cap < 0.833.
    # ...
    # Total infiltration should be roughly integrated capacity, which is < initial capacity (5.0)
    # But significantly higher than Ksat (0.5).

    saturated_hydraulic_conductivity_low = np.full_like(w, np.float32(0.5))
    topwater_high = np.float32(50.0)

    _, runoff_high, _, infiltration_amount_high, _, _, _, _ = infiltration.py_func(
        ws,
        np.zeros_like(ws),
        saturated_hydraulic_conductivity_low,
        np.float32(0.0),
        land_use_type,
        soil_is_frozen,
        w.copy(),  # Use a fresh copy of w (10.0, 50.0, ...)
        topwater_high,
        np.float32(0.0),
        np.float32(0.0),
        np.float32(100.0),  # wetting_front_suction_head_m
        np.float32(0.1),  # wetting_front_moisture_deficit
        np.int32(0),
        variable_runoff_shape_beta,
        bub_arr,
        h_arr,
        lam_arr,
    )

    # Check that we get the benefit of suction ( > Ksat)
    assert infiltration_amount_high > 0.5, "Should exceed Ksat due to suction"

    # Check that dynamic capacity reduces infiltration compared to static initial estimate
    # Note: With Green-Ampt starting from wetting_front=0, the initial capacity is very high.
    # We just ensure it is limited by available water and physically reasonable.
    assert infiltration_amount_high < topwater_high, (
        "Cannot infiltrate more than available"
    )

    # Based on manual calculation/execution, result is substantial
    assert infiltration_amount_high > 3.5, "Should still be substantial"

    # Check runoff for the first case
    assert np.isclose(runoff, 10.0 - infiltration_amount)

    # Verify that if Ksat is high, infiltration is higher
    saturated_hydraulic_conductivity_high = np.full_like(w, np.float32(20.0))
    _, runoff_high, _, infiltration_amount_high, _, _, _, _ = infiltration.py_func(
        ws,
        np.zeros_like(ws),
        saturated_hydraulic_conductivity_high,
        np.float32(0.0),
        land_use_type,
        soil_is_frozen,
        w,
        topwater,
        np.float32(0.0),
        np.float32(0.0),
        np.float32(100.0),  # wetting_front_suction_head_m
        np.float32(0.1),  # wetting_front_moisture_deficit
        np.int32(0),
        variable_runoff_shape_beta,
        bub_arr,
        h_arr,
        lam_arr,
    )

    # With high Ksat, infiltration should be higher than 2.0
    # (It will be determined by variable infiltration curve)
    assert infiltration_amount_high > 2.0
    assert infiltration_amount_high <= 10.0


def test_pedotransfer_functions_consistency() -> None:
    """Test consistency between different pedotransfer functions for typical soils."""
    # Define typical soil types with their properties
    # Composition: sand, clay, silt (must sum to 100)
    # Other props: bulk_density (g/cm3), organic_carbon (%)
    soils = {
        "Sand": {
            "sand": 92.0,
            "clay": 3.0,
            "silt": 5.0,
            "bulk_density": 1.6,
            "organic_carbon": 0.5,
        },
        "Loamy Sand": {
            "sand": 82.0,
            "clay": 6.0,
            "silt": 12.0,
            "bulk_density": 1.55,
            "organic_carbon": 0.8,
        },
        "Sandy Loam": {
            "sand": 65.0,
            "clay": 10.0,
            "silt": 25.0,
            "bulk_density": 1.5,
            "organic_carbon": 1.2,
        },
        "Loam": {
            "sand": 40.0,
            "clay": 20.0,
            "silt": 40.0,
            "bulk_density": 1.4,
            "organic_carbon": 2.0,
        },
        "Silt Loam": {
            "sand": 20.0,
            "clay": 15.0,
            "silt": 65.0,
            "bulk_density": 1.35,
            "organic_carbon": 2.5,
        },
        "Silt": {
            "sand": 5.0,
            "clay": 5.0,
            "silt": 90.0,
            "bulk_density": 1.3,
            "organic_carbon": 3.0,
        },
        "Clay Loam": {
            "sand": 30.0,
            "clay": 35.0,
            "silt": 35.0,
            "bulk_density": 1.35,
            "organic_carbon": 2.0,
        },
        "Clay": {
            "sand": 20.0,
            "clay": 60.0,
            "silt": 20.0,
            "bulk_density": 1.25,
            "organic_carbon": 3.5,
        },
    }

    print()
    print(
        f"{'Soil Type':<15} | {'Kv Brakensiek':<15} | {'Kv Wosten':<15} | {'Kv Cosby':<15} | {'Thetas Toth':<15} | {'Thetas Wosten':<15} | {'Thetar':<15} | {'Bubbling P':<15} | {'Psi Index B':<15} | {'Psi Index W':<15}"
    )
    print("-" * 160)

    results = {}

    for name, props in soils.items():
        # Prepare inputs as numpy arrays (scalar-like)
        sand = np.array([props["sand"]], dtype=np.float32)
        clay = np.array([props["clay"]], dtype=np.float32)
        silt = np.array([props["silt"]], dtype=np.float32)
        bulk_density_kg_per_dm3 = np.array([props["bulk_density"]], dtype=np.float32)
        organic_carbon_percentage = np.array(
            [props["organic_carbon"]], dtype=np.float32
        )
        is_top_soil = np.array([True], dtype=bool)

        # Calculate Thetas (Saturated Water Content)
        # Using Toth as a baseline for Brakensiek input
        thetas_val_toth = thetas_toth(
            organic_carbon_percentage=organic_carbon_percentage,
            bulk_density_kg_per_dm3=bulk_density_kg_per_dm3,
            is_top_soil=is_top_soil,
            clay=clay,
            silt=silt,
        )

        thetas_val_wosten = thetas_wosten(
            clay=clay,
            bulk_density_kg_per_dm3=bulk_density_kg_per_dm3,
            silt=silt,
            organic_carbon_percentage=organic_carbon_percentage,
            is_topsoil=is_top_soil,
        )

        # Check if Thetas values are reasonable (0 to 1)
        assert 0.3 < thetas_val_toth[0] < 0.8, (
            f"Thetas Toth out of range for {name}: {thetas_val_toth[0]}"
        )
        assert 0.3 < thetas_val_wosten[0] < 0.8, (
            f"Thetas Wosten out of range for {name}: {thetas_val_wosten[0]}"
        )

        # Calculate Hydraulic Conductivity (Kv)
        kv_b = kv_brakensiek(thetas=thetas_val_toth, clay=clay, sand=sand)
        kv_w = kv_wosten(
            silt=silt,
            clay=clay,
            bulk_density_kg_per_dm3=bulk_density_kg_per_dm3,
            organic_carbon_percentage=organic_carbon_percentage,
            is_topsoil=is_top_soil,
        )
        kv_c = kv_cosby(sand=sand, clay=clay)

        val_b = kv_b[0]
        val_w = kv_w[0]
        val_c = kv_c[0]

        # Additional parameters
        thetar = thetar_brakensiek(sand=sand, clay=clay, thetas=thetas_val_toth)
        bubbling_pressure = get_bubbling_pressure(
            clay=clay, sand=sand, thetas=thetas_val_toth
        )
        psi_index_b = get_pore_size_index_brakensiek(
            sand=sand, thetas=thetas_val_toth, clay=clay
        )
        psi_index_w = get_pore_size_index_wosten(
            clay=clay,
            silt=silt,
            organic_carbon_percentage=organic_carbon_percentage,
            bulk_density_kg_per_dm3=bulk_density_kg_per_dm3,
            is_top_soil=is_top_soil,
        )

        print(
            f"{name:<15} | {val_b:.2e}        | {val_w:.2e}        | {val_c:.2e}        | {thetas_val_toth[0]:.4f}          | {thetas_val_wosten[0]:.4f}          | {thetar[0]:.4f}          | {bubbling_pressure[0]:.2f}           | {psi_index_b[0]:.4f}          | {psi_index_w[0]:.4f}"
        )

        # Store results for cross-soil comparison
        results[name] = {
            "kv_b": val_b,
            "kv_w": val_w,
            "kv_c": val_c,
            "thetar": thetar[0],
            "thetas_toth": thetas_val_toth[0],
            "thetas_wosten": thetas_val_wosten[0],
            "bubbling_pressure": bubbling_pressure[0],
            "psi_index_b": psi_index_b[0],
            "psi_index_w": psi_index_w[0],
        }

        # Check if Kv values are positive
        assert val_b > 0
        assert val_w > 0
        assert val_c > 0

        # Check order of magnitude consistency
        # We allow for some deviation, e.g., 1-2 orders of magnitude, as these are empirical functions
        # and can vary significantly. However, they shouldn't be wildly different (e.g. 1e-5 vs 1e-9).

        # Using log10 to compare orders of magnitude
        log_b = np.log10(val_b)
        log_w = np.log10(val_w)
        log_c = np.log10(val_c)

        # For clayey soils, allow larger tolerance due to known discrepancies between Brakensiek and others
        # Brakensiek is known to give lower Kv for clays
        if name == "Clay" or name == "Clay Loam":
            brakensiek_tollerance = 2.5
        else:
            brakensiek_tollerance = 1.0
        assert abs(log_b - log_w) < brakensiek_tollerance, (
            f"Kv mismatch Brakensiek vs Wosten for {name}"
        )
        assert abs(log_b - log_c) < brakensiek_tollerance, (
            f"Kv mismatch Brakensiek vs Cosby for {name}"
        )
        assert abs(log_w - log_c) < 1.0, f"Kv mismatch Wosten vs Cosby for {name}"

        # Additional checks for other parameters
        assert 0 < thetar[0] < thetas_val_toth[0], f"Thetar invalid for {name}"
        assert bubbling_pressure[0] > 0, f"Bubbling pressure invalid for {name}"
        assert psi_index_b[0] > 0, f"Pore size index Brakensiek invalid for {name}"
        assert psi_index_w[0] > 0, f"Pore size index Wosten invalid for {name}"

    # Cross-soil consistency checks
    # Sand should have higher conductivity than Clay
    assert results["Sand"]["kv_b"] > results["Clay"]["kv_b"], (
        "Sand should be more conductive than Clay (Brakensiek)"
    )
    assert results["Sand"]["kv_w"] > results["Clay"]["kv_w"], (
        "Sand should be more conductive than Clay (Wosten)"
    )
    assert results["Sand"]["kv_c"] > results["Clay"]["kv_c"], (
        "Sand should be more conductive than Clay (Cosby)"
    )

    # Clay should have higher bubbling pressure (suction) than Sand
    assert (
        results["Clay"]["bubbling_pressure"] > results["Sand"]["bubbling_pressure"]
    ), "Clay should have higher bubbling pressure than Sand"

    # Sand should have higher pore size index (lambda) than Clay
    assert results["Sand"]["psi_index_b"] > results["Clay"]["psi_index_b"], (
        "Sand should have higher lambda than Clay (Brakensiek)"
    )

    # Clay should have higher residual water content than Sand
    assert results["Clay"]["thetar"] > results["Sand"]["thetar"], (
        "Clay should have higher residual water content than Sand"
    )

    # Clay should have higher saturated water content (porosity) than Sand
    # (Due to lower bulk density and structure)
    assert results["Clay"]["thetas_toth"] > results["Sand"]["thetas_toth"], (
        "Clay should have higher Thetas than Sand (Toth)"
    )
    assert results["Clay"]["thetas_wosten"] > results["Sand"]["thetas_wosten"], (
        "Clay should have higher Thetas than Sand (Wosten)"
    )


def test_get_heat_capacity() -> None:
    """Test get_heat_capacity_solid_fraction."""
    # Test valid input
    # Bulk density of 1.3 g/cm3 (~1300 kg/m3) should yield a porosity of roughly 0.5
    # (actually 1 - 1300/2650 = 0.49 void, 0.51 solid).
    # Solid fraction phi_s = 1300 / 2650 = 0.490566
    # Volumetric Heat capacity C_s = phi_s * C_mineral = 0.490566 * 2.13e6 = 1.045e6
    # Layer thickness = 1.0 m
    # Areal Heat Capacity = 1.045e6 * 1.0 = 1.045e6 J/(m2 K)

    bulk_density = np.array([1.3], dtype=np.float32)
    layer_thickness = np.array([1.0], dtype=np.float32)

    expected_phi_s = 1300.0 / 2650.0
    volumetric_hc = expected_phi_s * 2.13e6
    expected_areal_hc = volumetric_hc * 1.0

    result = get_heat_capacity_solid_fraction(bulk_density, layer_thickness)

    np.testing.assert_allclose(
        result, np.array([expected_areal_hc], dtype=np.float32), rtol=1e-5
    )

    # Test with bulk density equal to mineral density (solid rock)
    # Bulk density 2.65 g/cm3. phi_s should be 1.0. Heat capacity = C_mineral = 2.13e6.
    # Layer thickness = 2.0 m
    # Areal Heat Capacity = 2.13e6 * 2.0

    bulk_density_rock = np.array([2.65], dtype=np.float32)
    layer_thickness_rock = np.array([2.0], dtype=np.float32)

    expected_hc_rock = 2.13e6
    expected_areal_hc_rock = expected_hc_rock * 2.0

    result_rock = get_heat_capacity_solid_fraction(
        bulk_density_rock, layer_thickness_rock
    )
    np.testing.assert_allclose(
        result_rock, np.array([expected_areal_hc_rock], dtype=np.float32), rtol=1e-5
    )

    # Test with multiple layers summing to 1.0 m thickness
    # Bulk density 1.3 g/cm3 for all layers
    bulk_density_val = 1.3
    # Layer thicknesses: 0.1, 0.2, 0.3, 0.4 -> Sum = 1.0
    layer_thicknesses = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    bulk_densities = np.full_like(layer_thicknesses, bulk_density_val)

    # Expected volumetric heat capacity (same as first test case)
    expected_phi_s_multi = 1300.0 / 2650.0
    expected_volumetric_hc_multi = expected_phi_s_multi * 2.13e6

    # Calculate for multi-layer case
    result_multi = get_heat_capacity_solid_fraction(bulk_densities, layer_thicknesses)

    # Total heat capacity sum should match calculation for single 1.0m layer
    total_heat_capacity_sum = np.sum(result_multi)
    expected_total_1m = expected_volumetric_hc_multi * 1.0

    np.testing.assert_allclose(
        total_heat_capacity_sum,
        expected_total_1m,
        rtol=1e-5,
        err_msg="Sum of layer heat capacities should match single block of combined thickness",
    )

    # Individual layer Check
    expected_result_multi = expected_volumetric_hc_multi * layer_thicknesses
    np.testing.assert_allclose(
        result_multi,
        expected_result_multi,
        rtol=1e-5,
        err_msg="Individual layer heat capacities mismatch",
    )


def test_calculate_sensible_heat_flux() -> None:
    """Test the calculate_sensible_heat_flux function."""
    # Equilibrium (No transfer)
    flux, G = calculate_sensible_heat_flux(
        soil_temperature_C=np.float32(20.0),
        air_temperature_K=np.float32(293.15),  # 20C
        wind_speed_10m_m_per_s=np.float32(2.0),
        surface_pressure_pa=np.float32(101325.0),
    )
    assert abs(flux) < 1e-4

    # Air warmer than soil (Warming)
    flux_warming, _ = calculate_sensible_heat_flux(
        soil_temperature_C=np.float32(10.0),
        air_temperature_K=np.float32(293.15),  # 20C
        wind_speed_10m_m_per_s=np.float32(5.0),
        surface_pressure_pa=np.float32(101325.0),
    )
    assert flux_warming > 0.0

    # Soil warmer than air (Cooling)
    flux_cooling, _ = calculate_sensible_heat_flux(
        soil_temperature_C=np.float32(30.0),
        air_temperature_K=np.float32(293.15),  # 20C
        wind_speed_10m_m_per_s=np.float32(5.0),
        surface_pressure_pa=np.float32(101325.0),
    )
    assert flux_cooling < 0.0


def test_solve_energy_balance_implicit_iterative() -> None:
    """Test the implicit iterative energy balance solver."""
    # Common Parameters (Scalars)
    soil_temperature_old = np.float32(10.0)  # 10 C
    bulk_density = np.float32(1300.0)
    layer_thickness = np.float32(0.1)

    # Calculate heat capacity for scalar input
    # Note: get_heat_capacity_solid_fraction expects arrays usually, but let's see.
    # It seems to accept arrays. Let's compute it with array helper but use result as scalar.
    heat_capacity_arr = get_heat_capacity_solid_fraction(
        np.array([bulk_density / 1000.0], dtype=np.float32),
        np.array([layer_thickness], dtype=np.float32),
    )
    heat_capacity_areal = heat_capacity_arr[0]

    # Steady State / No Forcing
    sw_in = np.float32(0.0)
    lw_in = np.float32(363.0)  # Approx balance
    air_temp_k = np.float32(283.15)
    wind_speed = np.float32(2.0)
    pressure = np.float32(101325.0)
    dt_seconds = np.float32(3600.0)

    t_new = solve_energy_balance_implicit_iterative(
        soil_temperature_C=soil_temperature_old,
        shortwave_radiation_W_per_m2=sw_in,
        longwave_radiation_W_per_m2=lw_in,
        air_temperature_K=air_temp_k,
        wind_speed_10m_m_per_s=wind_speed,
        surface_pressure_pa=pressure,
        solid_heat_capacity_J_per_m2_K=heat_capacity_areal,
        timestep_seconds=dt_seconds,
    )

    # Should stay close to 10.0
    assert abs(t_new - 10.0) < 0.5, f"Steady state failed, got {t_new}"

    # Strong heating (daytime)
    sw_in = np.float32(800.0)
    lw_in = np.float32(350.0)
    air_temp_k = np.float32(303.15)

    t_new_hot = solve_energy_balance_implicit_iterative(
        soil_temperature_C=soil_temperature_old,
        shortwave_radiation_W_per_m2=sw_in,
        longwave_radiation_W_per_m2=lw_in,
        air_temperature_K=air_temp_k,
        wind_speed_10m_m_per_s=wind_speed,
        surface_pressure_pa=pressure,
        solid_heat_capacity_J_per_m2_K=heat_capacity_areal,
        timestep_seconds=dt_seconds,
    )
    assert t_new_hot > 10.0, "Soil should warm up significantly"

    # Strong cooling (nighttime clear sky)
    sw_in = np.float32(0.0)
    lw_in = np.float32(200.0)
    air_temp_k = np.float32(263.15)

    t_new_cold = solve_energy_balance_implicit_iterative(
        soil_temperature_C=soil_temperature_old,
        shortwave_radiation_W_per_m2=sw_in,
        longwave_radiation_W_per_m2=lw_in,
        air_temperature_K=air_temp_k,
        wind_speed_10m_m_per_s=wind_speed,
        surface_pressure_pa=pressure,
        solid_heat_capacity_J_per_m2_K=heat_capacity_areal,
        timestep_seconds=dt_seconds,
    )

    assert t_new_cold < 10.0, "Soil should cool down"


def test_calculate_spatial_infiltration_excess() -> None:
    """Test the spatial variability of infiltration capacity function."""
    # Case 1: Low precipitation (P << f_max), should be mostly infiltrated
    f_GA = np.float32(10.0)
    beta = np.float32(0.2)
    f_max = f_GA * (beta + np.float32(1.0))  # 12.0

    # Very small P
    P = np.float32(0.01)
    infil, runoff = calculate_spatial_infiltration_excess(f_GA, P, beta)
    # For very small P, infiltration approaches P
    assert abs(infil - P) < 1e-4
    assert abs(runoff - 0.0) < 1e-4

    # Case 2: High precipitation (P >= f_max)
    P_high = np.float32(20.0)
    infil, runoff = calculate_spatial_infiltration_excess(f_GA, P_high, beta)
    assert abs(infil - f_GA) < 1e-4
    assert abs(runoff - (P_high - f_GA)) < 1e-4

    # Case 3: Intermediate P
    P_mid = f_max / np.float32(2.0)  # 6.0
    # ratio = 0.5
    # power = 1.2
    # term = 1 - (0.5)^1.2
    # infil = f_GA * term
    infil, runoff = calculate_spatial_infiltration_excess(f_GA, P_mid, beta)
    expected_term = np.float32(1.0) - (np.float32(0.5) ** np.float32(1.2))
    expected_infil = f_GA * expected_term
    assert abs(infil - expected_infil) < 1e-4
    assert abs(runoff - (P_mid - infil)) < 1e-4

    # Case 4: Zero capacity
    infil, runoff = calculate_spatial_infiltration_excess(np.float32(0.0), P_mid, beta)
    assert infil == 0.0
    assert runoff == P_mid

    # Case 5: Zero precip
    infil, runoff = calculate_spatial_infiltration_excess(f_GA, np.float32(0.0), beta)
    assert infil == 0.0
    assert runoff == 0.0


def test_plot_spatial_infiltration_curve() -> None:
    """Plot the spatial infiltration curve for verification."""
    # Mean Green-Ampt Capacity
    f_GA = np.float32(10.0)  # arbitrary units

    betas = [0.01, 0.2, 0.5, 1.0, 5.0]
    precip_values = np.linspace(0, 30, 100).astype(np.float32)

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(6, 6))

    for beta in betas:
        infils = []
        for P in precip_values:
            infil, _ = calculate_spatial_infiltration_excess(f_GA, P, np.float32(beta))
            infils.append(infil)

        plt.plot(precip_values, infils, label=f"beta={beta}")

    # Plot f=P line (100% infiltration)
    plt.plot(
        precip_values,
        precip_values,
        "k--",
        alpha=0.3,
        label="1:1 Line (100% Infiltration)",
    )

    # Plot f=f_GA line (Mean capacity)
    plt.axhline(y=f_GA, color="r", linestyle=":", label="Mean Capacity (F)")

    plt.gca().set_aspect(
        "equal", adjustable="box"
    )  # Ensure the axes box is exactly square
    plt.xlim(0, 30)
    plt.ylim(0, 30)

    plt.xlabel("Rainfall Intensity / Available Water [L/T]", fontweight="bold")
    plt.ylabel("Effective Infiltration Rate [L/T]", fontweight="bold")
    plt.title("Spatial Variability of Infiltration")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_folder / "spatial_infiltration_curve.svg")
    plt.close()
