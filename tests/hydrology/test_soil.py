"""Tests for scalar soil functions in GEB."""

import numpy as np

from geb.hydrology.landcovers import (
    NON_PADDY_IRRIGATED,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
)
from geb.hydrology.soil import (
    add_water_to_topwater_and_evaporate_open_water,
    get_infiltration_capacity,
    get_soil_moisture_at_pressure,
    get_soil_water_flow_parameters,
    infiltration,
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
    assert abs(runoff - capillary_rise) < 1e-6
    # Soil water unchanged
    assert np.allclose(w, w_pre)


def test_get_infiltration_capacity() -> None:
    """Test get_infiltration_capacity function."""
    k_sat = np.full(6, np.float32(0.1))
    capacity = get_infiltration_capacity(k_sat)
    assert capacity == k_sat[0], (
        "Infiltration capacity should equal saturated hydraulic conductivity"
    )

    k_sat = np.full(6, np.float32(0.05))
    capacity = get_infiltration_capacity(k_sat)
    assert capacity == k_sat[0], (
        "Infiltration capacity should equal saturated hydraulic conductivity"
    )

    # Boundary conditions
    # Zero conductivity
    k_sat = np.full(6, np.float32(0.0))
    capacity = get_infiltration_capacity(k_sat)
    assert capacity == 0.0

    # Very small conductivity
    k_sat = np.full(6, np.float32(1e-6))
    capacity = get_infiltration_capacity(k_sat)
    assert abs(capacity - 1e-6) < 1e-9

    # Very large conductivity
    k_sat = np.full(6, np.float32(1e6))
    capacity = get_infiltration_capacity(k_sat)
    assert capacity == 1e6


def test_infiltration() -> None:
    """Test the scalar infiltration function."""
    # Test case 1: Normal infiltration
    ws = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    saturated_hydraulic_conductivity = np.full_like(w, np.float32(0.01))
    land_use_type = np.int32(NON_PADDY_IRRIGATED)
    soil_is_frozen = False
    topwater = np.float32(0.005)

    w_pre = w.copy()
    topwater_pre = topwater

    updated_topwater, direct_runoff, groundwater_recharge, infiltration_amount = (
        infiltration(
            ws,
            saturated_hydraulic_conductivity,
            land_use_type,
            soil_is_frozen,
            w,
            topwater,
        )
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

    updated_topwater, direct_runoff, groundwater_recharge, infiltration_amount = (
        infiltration(
            ws,
            saturated_hydraulic_conductivity,
            land_use_type,
            soil_is_frozen,
            w,
            topwater,
        )
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

    updated_topwater, direct_runoff, groundwater_recharge, infiltration_amount = (
        infiltration(
            ws,
            saturated_hydraulic_conductivity,
            land_use_type,
            soil_is_frozen,
            w,
            topwater,
        )
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

    updated_topwater, direct_runoff, groundwater_recharge, infiltration_amount = (
        infiltration(
            ws,
            saturated_hydraulic_conductivity,
            land_use_type,
            soil_is_frozen,
            w,
            topwater,
        )
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

    updated_topwater, direct_runoff, groundwater_recharge, infiltration_amount = (
        infiltration(
            ws,
            saturated_hydraulic_conductivity,
            land_use_type,
            soil_is_frozen,
            w,
            topwater,
        )
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

    updated_topwater, direct_runoff, groundwater_recharge, infiltration_amount = (
        infiltration(
            ws,
            saturated_hydraulic_conductivity,
            land_use_type,
            soil_is_frozen,
            w,
            topwater,
        )
    )

    # Should infiltrate up to capacity, then pond up to 0.05m before runoff
    assert infiltration_amount > 0.0
    assert groundwater_recharge == 0.0
    # Direct runoff should be topwater - 0.05 (ponding allowance)
    expected_runoff = topwater - 0.05 - infiltration_amount
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
    updated_topwater, direct_runoff, groundwater_recharge, infiltration_amount = (
        infiltration(
            ws,
            saturated_hydraulic_conductivity,
            land_use_type,
            soil_is_frozen,
            w,
            topwater,
        )
    )

    # Should infiltrate into unsaturated layers
    assert infiltration_amount > 0.0
    assert groundwater_recharge == 0.0
    # Check that unsaturated layers received water
    assert w[2] > w_pre[2] or w[3] > w_pre[3] or w[4] > w_pre[4] or w[5] > w_pre[5]


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
