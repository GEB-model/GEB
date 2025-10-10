"""Tests for scalar soil functions in GEB."""

import numpy as np

from geb.hydrology.landcover import (
    NON_PADDY_IRRIGATED,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
)
from geb.hydrology.soil_scalar import (
    add_water_to_topwater_and_evaporate_open_water,
    get_infiltration_capacity,
    infiltration_scalar,
    rise_from_groundwater,
)


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
                reference_evapotranspiration_water_m_per_day=np.float32(et_ref),
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
            reference_evapotranspiration_water_m_per_day=np.float32(0.0),
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
            reference_evapotranspiration_water_m_per_day=np.float32(
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
    assert runoff == capillary_rise
    # Soil water unchanged
    assert np.allclose(w, w_pre)


def test_get_infiltration_capacity() -> None:
    """Test get_infiltration_capacity function."""
    k_sat = np.float32(0.1)
    capacity = get_infiltration_capacity(k_sat)
    assert capacity == k_sat, (
        "Infiltration capacity should equal saturated hydraulic conductivity"
    )

    k_sat = np.float32(0.05)
    capacity = get_infiltration_capacity(k_sat)
    assert capacity == k_sat, (
        "Infiltration capacity should equal saturated hydraulic conductivity"
    )

    # Boundary conditions
    # Zero conductivity
    k_sat = np.float32(0.0)
    capacity = get_infiltration_capacity(k_sat)
    assert capacity == 0.0

    # Very small conductivity
    k_sat = np.float32(1e-6)
    capacity = get_infiltration_capacity(k_sat)
    assert capacity == 1e-6

    # Very large conductivity
    k_sat = np.float32(1e6)
    capacity = get_infiltration_capacity(k_sat)
    assert capacity == 1e6


def test_infiltration() -> None:
    """Test the scalar infiltration function."""
    # Test case 1: Normal infiltration
    ws = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    saturated_hydraulic_conductivity = np.float32(0.01)
    land_use_type = np.int32(NON_PADDY_IRRIGATED)
    frost_index = np.float32(0.0)
    topwater = np.float32(0.005)

    w_pre = w.copy()
    topwater_pre = topwater

    direct_runoff, groundwater_recharge, infiltration_amount = infiltration_scalar(
        ws, saturated_hydraulic_conductivity, land_use_type, frost_index, w, topwater
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
    frost_index = np.float32(100.0)  # Above threshold

    direct_runoff, groundwater_recharge, infiltration_amount = infiltration_scalar(
        ws, saturated_hydraulic_conductivity, land_use_type, frost_index, w, topwater
    )

    # No infiltration on frozen soil
    assert infiltration_amount == 0.0
    assert groundwater_recharge == 0.0
    # All water should go to direct runoff
    assert direct_runoff == topwater_pre

    # Test case 3: Sealed area - no infiltration
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    topwater = np.float32(0.005)
    frost_index = np.float32(0.0)
    land_use_type = np.int32(SEALED)

    direct_runoff, groundwater_recharge, infiltration_amount = infiltration_scalar(
        ws, saturated_hydraulic_conductivity, land_use_type, frost_index, w, topwater
    )

    # No infiltration on sealed areas
    assert infiltration_amount == 0.0
    assert groundwater_recharge == 0.0
    # All water should go to direct runoff
    assert direct_runoff == topwater_pre

    # Test case 4: Zero topwater
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    topwater = np.float32(0.0)
    frost_index = np.float32(0.0)
    land_use_type = np.int32(NON_PADDY_IRRIGATED)

    direct_runoff, groundwater_recharge, infiltration_amount = infiltration_scalar(
        ws, saturated_hydraulic_conductivity, land_use_type, frost_index, w, topwater
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
    frost_index = np.float32(0.0)
    land_use_type = np.int32(OPEN_WATER)

    direct_runoff, groundwater_recharge, infiltration_amount = infiltration_scalar(
        ws, saturated_hydraulic_conductivity, land_use_type, frost_index, w, topwater
    )

    # No infiltration on open water
    assert infiltration_amount == 0.0
    assert groundwater_recharge == 0.0
    # All water should go to direct runoff
    assert direct_runoff == topwater_pre

    # Test case 6: Paddy irrigated with ponding allowance
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    topwater = np.float32(0.1)  # More than ponding allowance
    frost_index = np.float32(0.0)
    land_use_type = np.int32(PADDY_IRRIGATED)

    direct_runoff, groundwater_recharge, infiltration_amount = infiltration_scalar(
        ws, saturated_hydraulic_conductivity, land_use_type, frost_index, w, topwater
    )

    # Should infiltrate up to capacity, then pond up to 0.05m before runoff
    assert infiltration_amount > 0.0
    assert groundwater_recharge == 0.0
    # Direct runoff should be topwater - 0.05 (ponding allowance)
    expected_runoff = topwater - 0.05
    assert abs(direct_runoff - expected_runoff) < 1e-6

    # Test case 7: Frost index exactly at threshold
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    topwater = np.float32(0.005)
    frost_index = np.float32(85.0)  # Exactly at threshold
    land_use_type = np.int32(NON_PADDY_IRRIGATED)

    direct_runoff, groundwater_recharge, infiltration_amount = infiltration_scalar(
        ws, saturated_hydraulic_conductivity, land_use_type, frost_index, w, topwater
    )

    # Should still infiltrate (frost_index > threshold means frozen)
    assert infiltration_amount == 0.0
    assert groundwater_recharge == 0.0
    assert direct_runoff == topwater_pre

    # Test case 8: Soil layers at different saturation levels
    ws = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    w = np.array(
        [0.3, 0.3, 0.1, 0.1, 0.1, 0.1], dtype=np.float32
    )  # Top layers saturated
    topwater = np.float32(0.005)
    frost_index = np.float32(0.0)
    land_use_type = np.int32(NON_PADDY_IRRIGATED)

    w_pre = w.copy()
    direct_runoff, groundwater_recharge, infiltration_amount = infiltration_scalar(
        ws, saturated_hydraulic_conductivity, land_use_type, frost_index, w, topwater
    )

    # Should infiltrate into unsaturated layers
    assert infiltration_amount > 0.0
    assert groundwater_recharge == 0.0
    # Check that unsaturated layers received water
    assert w[2] > w_pre[2] or w[3] > w_pre[3] or w[4] > w_pre[4] or w[5] > w_pre[5]
