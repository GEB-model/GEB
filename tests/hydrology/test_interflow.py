"""Unit tests for interflow calculation in hydrology module."""

import matplotlib.pyplot as plt
import numpy as np

from geb.hydrology.soil import get_interflow

from ..testconfig import output_folder


def test_get_interflow_normal_conditions() -> None:
    """Test get_interflow under normal conditions."""
    # Soil layer properties (physically consistent)
    soil_layer_height_m = np.float32(0.1)

    # Water contents in meters (must be <= soil_layer_height_m)
    # porosity ~ 0.4 -> ws = 0.04
    # field capacity ~ 0.25 -> wfc = 0.025
    # current moisture ~ 0.3 -> w = 0.03
    ws = np.float32(0.04)
    wfc = np.float32(0.025)
    w = np.float32(0.03)

    # Hydraulic properties
    unsaturated_hydraulic_conductivity_m_per_hour = np.float32(0.01)
    slope_m_per_m = np.float32(0.05)
    hillslope_length_m = np.float32(100.0)
    interflow_multiplier = np.float32(1.0)

    # Expected behavior according to physical model (Darcy/Storage coefficient)
    # Drainable porosity = (0.04 - 0.025) / 0.1 = 0.15 (dimensionless)
    # Lateral K = 0.01 * 10 = 0.1 m/h (Note: factor 10 is hardcoded in implementation)
    # Storage coefficient = (0.1 * 0.05) / (0.15 * 100) = 0.005 / 15 = 3.333e-4 h^-1
    # Free water = 0.03 - 0.025 = 0.005 m
    # Expected Interflow = 0.005 * 3.333e-4 = 1.666e-6 m/h

    interflow = get_interflow(
        w,
        wfc,
        ws,
        soil_layer_height_m,
        unsaturated_hydraulic_conductivity_m_per_hour,
        slope_m_per_m,
        hillslope_length_m,
        interflow_multiplier,
    )

    # Check that interflow is positive
    assert interflow > 0.0

    # Check that interflow removes water (but not more than available)
    free_water = w - wfc
    assert interflow <= free_water


def test_get_interflow_boundary_conditions() -> None:
    """Test get_interflow boundary conditions."""
    soil_layer_height_m = np.float32(0.1)
    ws = np.float32(0.04)
    wfc = np.float32(0.025)
    hillslope_length_m = np.float32(100.0)
    interflow_multiplier = np.float32(1.0)

    # Case 1: No free water (w <= wfc)
    w = np.float32(0.02)  # below field capacity
    unsaturated_hydraulic_conductivity_m_per_hour = np.float32(0.01)
    slope_m_per_m = np.float32(0.05)

    interflow = get_interflow(
        w,
        wfc,
        ws,
        soil_layer_height_m,
        unsaturated_hydraulic_conductivity_m_per_hour,
        slope_m_per_m,
        hillslope_length_m,
        interflow_multiplier,
    )
    assert interflow == 0.0

    # Case 2: Zero slope
    slope_m_per_m = np.float32(0.0)
    w = np.float32(0.03)  # above field capacity

    interflow = get_interflow(
        w,
        wfc,
        ws,
        soil_layer_height_m,
        unsaturated_hydraulic_conductivity_m_per_hour,
        slope_m_per_m,
        hillslope_length_m,
        interflow_multiplier,
    )

    assert interflow == 0.0

    # Case 3: Zero conductivity
    slope_m_per_m = np.float32(0.05)
    unsaturated_hydraulic_conductivity_m_per_hour = np.float32(0.0)

    interflow = get_interflow(
        w,
        wfc,
        ws,
        soil_layer_height_m,
        unsaturated_hydraulic_conductivity_m_per_hour,
        slope_m_per_m,
        hillslope_length_m,
        interflow_multiplier,
    )

    assert interflow == 0.0


def test_interflow_conforms_to_literature_theory() -> None:
    """Test if the interflow calculation matches the theoretical formulation.

    Based on the code structure (calculating storage_coefficient), the expected formula is:
    Q = free_water * storage_coefficient
    where storage_coefficient = (K_lat * slope) / (drainable_porosity * L) * multiplier
    and K_lat = K_unsat * 10 (hardcoded anisotropy factor)
    """
    soil_layer_height_m = np.float32(0.1)
    ws = np.float32(0.04)  # 40%
    wfc = np.float32(0.02)  # 20%
    w = np.float32(0.03)  # 30%

    unsaturated_hydraulic_conductivity_m_per_hour = np.float32(0.01)  # m/h
    slope_m_per_m = np.float32(0.05)
    hillslope_length_m = np.float32(100.0)  # m
    interflow_multiplier = np.float32(1.0)

    # Calculate theoretical value
    free_water = w - wfc  # 0.01 m
    drainable_porosity = (ws - wfc) / soil_layer_height_m  # (0.02) / 0.1 = 0.2

    # Lateral conductivity is assumed 10x vertical
    lateral_conductivity = unsaturated_hydraulic_conductivity_m_per_hour * 10.0

    # storage_coefficient [1/h] = (K * S) / (phi * L)
    storage_coefficient = (lateral_conductivity * slope_m_per_m) / (
        drainable_porosity * hillslope_length_m
    )
    # (0.1 * 0.05) / (0.2 * 100) = 0.005 / 20 = 2.5e-4 h^-1

    expected_interflow = free_water * storage_coefficient  # 0.01 * 2.5e-4 = 2.5e-6 m/h

    actual_interflow = get_interflow(
        w,
        wfc,
        ws,
        soil_layer_height_m,
        unsaturated_hydraulic_conductivity_m_per_hour,
        slope_m_per_m,
        hillslope_length_m,
        interflow_multiplier,
    )

    # Allow small floating point differences
    assert abs(actual_interflow - expected_interflow) < 1e-9, (
        f"Interflow {actual_interflow} does not match theoretical value {expected_interflow}"
    )


def test_interflow_multiplier() -> None:
    """Test that the interflow multiplier correctly scales the result."""
    soil_layer_height_m = np.float32(0.1)
    ws = np.float32(0.04)
    wfc = np.float32(0.02)
    w = np.float32(0.03)

    unsaturated_hydraulic_conductivity_m_per_hour = np.float32(0.01)
    slope_m_per_m = np.float32(0.05)
    hillslope_length_m = np.float32(100.0)

    # Base calculation with multiplier 1.0
    val_1 = get_interflow(
        w,
        wfc,
        ws,
        soil_layer_height_m,
        unsaturated_hydraulic_conductivity_m_per_hour,
        slope_m_per_m,
        hillslope_length_m,
        np.float32(1.0),
    )

    # Calculation with multiplier 2.0
    val_2 = get_interflow(
        w,
        wfc,
        ws,
        soil_layer_height_m,
        unsaturated_hydraulic_conductivity_m_per_hour,
        slope_m_per_m,
        hillslope_length_m,
        np.float32(2.0),
    )

    assert abs(val_2 - (val_1 * 2.0)) < 1e-9

    # Calculation with multiplier 0.5
    val_05 = get_interflow(
        w,
        wfc,
        ws,
        soil_layer_height_m,
        unsaturated_hydraulic_conductivity_m_per_hour,
        slope_m_per_m,
        hillslope_length_m,
        np.float32(0.5),
    )

    assert abs(val_05 - (val_1 * 0.5)) < 1e-9


def test_plot_interflow_response() -> None:
    """Plot interflow response to varying slope (boundary condition check)."""
    soil_layer_height_m = np.float32(0.1)
    w = np.float32(0.03)
    wfc = np.float32(0.02)
    ws = np.float32(0.04)

    unsaturated_hydraulic_conductivity_m_per_hour = np.float32(0.01)
    hillslope_length_m = np.float32(100.0)
    interflow_multiplier = np.float32(1.0)

    slopes = np.linspace(0, 0.5, 50).astype(np.float32)
    interflows = []

    for slope in slopes:
        val = get_interflow(
            w,
            wfc,
            ws,
            soil_layer_height_m,
            unsaturated_hydraulic_conductivity_m_per_hour,
            slope,
            hillslope_length_m,
            interflow_multiplier,
        )
        interflows.append(val)

    plt.figure(figsize=(10, 6))
    plt.plot(slopes, interflows, "b-", label="Calculated Interflow")
    plt.xlabel("Slope (m/m)")
    plt.ylabel("Interflow (m/h)")
    plt.title("Interflow Response to Slope")
    plt.grid(True)
    plt.legend()

    plot_file = output_folder / "interflow_slope_response.png"
    plt.savefig(plot_file)
    plt.close()


def test_plot_interflow_moisture_response() -> None:
    """Plot interflow response to varying soil moisture."""
    soil_layer_height_m = np.float32(0.1)
    wfc = np.float32(0.025)
    ws = np.float32(0.04)

    # Range from somewhat dry to saturated
    w_range = np.linspace(0.02, 0.045, 100).astype(np.float32)

    unsaturated_hydraulic_conductivity_m_per_hour = np.float32(0.01)
    slope_m_per_m = np.float32(0.1)
    hillslope_length_m = np.float32(100.0)
    interflow_multiplier = np.float32(1.0)

    interflows = []
    for w in w_range:
        val = get_interflow(
            w,
            wfc,
            ws,
            soil_layer_height_m,
            unsaturated_hydraulic_conductivity_m_per_hour,
            slope_m_per_m,
            hillslope_length_m,
            interflow_multiplier,
        )
        interflows.append(val)

    plt.figure(figsize=(10, 6))
    plt.plot(w_range, interflows, "g-", label="Interflow")
    plt.axvline(x=wfc, color="r", linestyle="--", label="Field Capacity")
    plt.axvline(x=ws, color="k", linestyle=":", label="Saturation")
    plt.xlabel("Soil Moisture (m)")
    plt.ylabel("Interflow (m/h)")
    plt.title("Interflow Response to Soil Moisture")
    plt.grid(True)
    plt.legend()

    plot_file = output_folder / "interflow_moisture_response.png"
    plt.savefig(plot_file)
    plt.close()
