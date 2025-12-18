"""Tests for scalar soil functions in GEB."""

import matplotlib.pyplot as plt
import numpy as np

from geb.hydrology.landcovers import (
    NON_PADDY_IRRIGATED,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
)
from geb.hydrology.soil import (
    add_water_to_topwater_and_evaporate_open_water,
    calculate_arno_runoff,
    get_bubbling_pressure,
    get_infiltration_capacity,
    get_pore_size_index_brakensiek,
    get_pore_size_index_wosten,
    get_soil_moisture_at_pressure,
    get_soil_water_flow_parameters,
    infiltration,
    kv_brakensiek,
    kv_cosby,
    kv_wosten,
    rise_from_groundwater,
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
            np.float32(0.1),
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
            np.float32(0.1),
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
            np.float32(0.1),
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
            np.float32(0.1),
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
            np.float32(0.1),
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
            np.float32(0.1),
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
            np.float32(0.1),
        )
    )

    # No infiltration into already saturated layers
    assert infiltration_amount == 0.0
    assert groundwater_recharge == 0.0


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


def test_plot_arno_runoff_response() -> None:
    """Visualize the runoff response of the Arno/Xinanjiang model."""
    ws = np.float32(100.0)

    # Plot 1: Runoff vs Precipitation for different initial soil moisture (fixed b)
    b = np.float32(0.4)  # Typical value
    precip_values = np.linspace(0, 50, 100, dtype=np.float32)
    w_ratios = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]

    plt.figure(figsize=(10, 6))
    for ratio in w_ratios:
        w_init = np.float32(ratio * ws)
        runoff_values = []
        for p in precip_values:
            r, _ = calculate_arno_runoff(
                w_init, ws, b, p, infiltration_capacity_m=np.float32(1e9)
            )
            runoff_values.append(r)

        plt.plot(precip_values, runoff_values, label=f"Initial W/Ws = {ratio}")

    plt.plot(
        precip_values, precip_values, "k--", alpha=0.3, label="1:1 Line (All Runoff)"
    )
    plt.xlabel("Precipitation (mm)")
    plt.ylabel("Runoff (mm)")
    plt.title(f"Runoff vs Precipitation (Ws={ws}, b={b})")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_folder / "arno_runoff_vs_precip.png")
    plt.close()

    # Plot 2: Runoff vs Soil Moisture for different b values (fixed P)
    p = np.float32(10.0)
    b_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    w_ratios = np.linspace(0, 1, 100, dtype=np.float32)

    plt.figure(figsize=(10, 6))
    for b_val in b_values:
        b = np.float32(b_val)
        runoff_values = []
        for ratio in w_ratios:
            w_init = np.float32(ratio * ws)
            r, _ = calculate_arno_runoff(
                w_init, ws, b, p, infiltration_capacity_m=np.float32(1e9)
            )
            runoff_values.append(r)

        plt.plot(w_ratios, runoff_values, label=f"b = {b_val}")

    plt.axhline(y=p, color="k", linestyle="--", alpha=0.3, label="Precipitation")
    plt.xlabel("Relative Soil Moisture (W/Ws)")
    plt.ylabel("Runoff (mm)")
    plt.title(f"Runoff vs Soil Moisture (P={p}mm, Ws={ws}mm)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_folder / "arno_runoff_vs_moisture.png")
    plt.close()


def test_infiltration_arno_integration() -> None:
    """Test infiltration with Arno runoff enabled."""
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
    # With Arno (b=0.4), even if not saturated, there should be some runoff.
    # In the standard model, with Ksat=10 and topwater=10, and capacity=50,
    # potential_infiltration = min(10, 50) = 10.
    # infiltration = min(10, 10) = 10.
    # So standard model would have 0 runoff.

    w_arno = w.copy()
    _, runoff_arno, _, infil_arno = infiltration.py_func(
        ws,
        saturated_hydraulic_conductivity,
        land_use_type,
        soil_is_frozen,
        w_arno,
        topwater,
        np.float32(0.4),
    )

    # Arno should produce some runoff because of the curve
    assert runoff_arno > 0.0
    assert infil_arno < 10.0
    assert abs(runoff_arno + infil_arno - topwater) < 1e-5


def test_infiltration_arno_capacity_limit() -> None:
    """Test that Arno infiltration is limited by infiltration capacity (Ksat)."""
    # Setup inputs
    ws = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32)
    w = np.array(
        [10.0, 50.0, 50.0, 50.0, 50.0, 50.0], dtype=np.float32
    )  # Low saturation (10/100 = 0.1)

    # Ksat is low (2.0), Topwater is high (10.0)
    # Arno would likely infiltrate most of the 10.0 if not limited,
    # because relative saturation is low (0.1).
    # Pass as array because infiltration expects array (for layers)
    saturated_hydraulic_conductivity = np.full_like(w, np.float32(2.0))

    land_use_type = np.int32(NON_PADDY_IRRIGATED)
    soil_is_frozen = False
    topwater = np.float32(10.0)
    arno_shape_parameter = np.float32(0.4)

    # Run infiltration using .py_func
    _, runoff, _, infiltration_amount = infiltration.py_func(
        ws,
        saturated_hydraulic_conductivity,
        land_use_type,
        soil_is_frozen,
        w,
        topwater,
        arno_shape_parameter,
    )

    # Infiltration should be limited to Ksat (2.0)
    assert infiltration_amount == 2.0

    # Runoff should be the rest (10.0 - 2.0 = 8.0)
    assert runoff == 8.0

    # Verify that if Ksat is high, infiltration is higher
    saturated_hydraulic_conductivity_high = np.full_like(w, np.float32(20.0))
    _, runoff_high, _, infiltration_amount_high = infiltration.py_func(
        ws,
        saturated_hydraulic_conductivity_high,
        land_use_type,
        soil_is_frozen,
        w,
        topwater,
        arno_shape_parameter,
    )

    # With high Ksat, infiltration should be higher than 2.0
    # (It will be determined by Arno curve)
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
        bulk_density = np.array([props["bulk_density"]], dtype=np.float32)
        organic_carbon_percentage = np.array(
            [props["organic_carbon"]], dtype=np.float32
        )
        is_top_soil = np.array([True], dtype=bool)

        # Calculate Thetas (Saturated Water Content)
        # Using Toth as a baseline for Brakensiek input
        thetas_val_toth = thetas_toth(
            organic_carbon_percentage=organic_carbon_percentage,
            bulk_density=bulk_density,
            is_top_soil=is_top_soil,
            clay=clay,
            silt=silt,
        )

        thetas_val_wosten = thetas_wosten(
            clay=clay,
            bulk_density=bulk_density,
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
            bulk_density=bulk_density,
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
            bulk_density=bulk_density,
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

        # Check if they are within 1.5 orders of magnitude of each other
        assert abs(log_b - log_w) < 1.5, f"Kv mismatch Brakensiek vs Wosten for {name}"
        assert abs(log_b - log_c) < 1.5, f"Kv mismatch Brakensiek vs Cosby for {name}"
        assert abs(log_w - log_c) < 1.5, f"Kv mismatch Wosten vs Cosby for {name}"

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
