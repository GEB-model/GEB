"""Tests for the RunoffConcentrator module."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geb.hydrology.runoff_concentration import (
    advance_buffer,
    apply_triangular,
    compute_slope_dependent_triangular_weights,
    triangular_weights,
)

from ...testconfig import output_folder

output_folder_runoff_concentrator: Path = output_folder / "runoff_concentrator"
output_folder_runoff_concentrator.mkdir(exist_ok=True, parents=True)


def test_triangular_weights_normalization() -> None:
    """Test that triangular weights always sum to approximately 1.0."""
    lag_time_hours = 48
    peak_hour = 12.0
    weights = triangular_weights(peak_hour, lag_time_hours)
    assert len(weights) == lag_time_hours
    assert np.isclose(weights.sum(), 1.0, atol=1e-10)


def test_triangular_weights_zero_peak() -> None:
    """Test that peak_hour=0 results in immediate runoff."""
    lag_time_hours = 24
    peak_hour = 0.0
    weights = triangular_weights(peak_hour, lag_time_hours)
    assert weights[0] == 1.0
    assert np.all(weights[1:] == 0.0)
    assert weights.sum() == 1.0


def test_triangular_weights_shape() -> None:
    """Test that the peak occurs at the expected location."""
    lag_time_hours = 48
    peak_hour = 10.0
    weights = triangular_weights(peak_hour, lag_time_hours)

    # The weight increases up to 'peak_hour' then decreases
    # Due to zero-indexing, lag 'k' represents time k+1
    # peak_hour=10.0 means peak should be around index 9
    assert np.argmax(weights) in [8, 9, 10]

    # Check monotonicity before peak
    for i in range(1, 9):
        assert weights[i] >= weights[i - 1]

    # Check monotonicity after peak (before it hits 0)
    # 2*peak_hour - hour_index. If this <= 0, weight is 0.
    # 2*10 - (i+1) <= 0 => i+1 >= 20 => i >= 19
    for i in range(11, 19):
        assert weights[i] <= weights[i - 1]


def test_advance_buffer_logic() -> None:
    """Test the buffer advancement logic."""
    lag_time_hours = 10
    n_cells = 2
    buffer = np.arange(lag_time_hours * n_cells, dtype=np.float32).reshape(
        lag_time_hours, n_cells
    )

    hours_to_advance = 3
    expected_buffer = np.zeros_like(buffer)
    expected_buffer[:-hours_to_advance] = buffer[hours_to_advance:]

    advance_buffer(buffer, hours_to_advance)
    np.testing.assert_array_equal(buffer, expected_buffer)


def test_advance_buffer_overflow() -> None:
    """Test buffer advancement when steps exceed lag_time_hours."""
    lag_time_hours = 10
    n_cells = 2
    buffer = np.ones((lag_time_hours, n_cells), dtype=np.float32)

    advance_buffer(buffer, 15)
    assert np.all(buffer == 0.0)


def test_apply_triangular_mass_balance() -> None:
    """Test that applying triangular weights preserves mass."""
    lag_time_hours = 48
    peak_hour = 3.0
    n_cells = 5
    n_steps = 24

    weights = np.repeat(
        triangular_weights(peak_hour, lag_time_hours)[:, None], n_cells, axis=1
    )
    flow = np.random.rand(n_steps, n_cells).astype(np.float32)
    buffer = np.zeros((lag_time_hours, n_cells), dtype=np.float64)

    apply_triangular(flow, weights, buffer)

    expected_mass = flow.astype(np.float64).sum(axis=0)

    # Since weights sum to 1.0, and idx = t + k < lag is always true for peak=3, lag=48, t=24
    # t_max = 23, k_max = 2*peak = 6. idx_max = 29 < 48.
    # So no mass should be lost.
    actual_mass = buffer.sum(axis=0)

    np.testing.assert_allclose(actual_mass, expected_mass, atol=1e-12)


def test_apply_triangular_delay() -> None:
    """Test that flow is correctly delayed in the buffer."""
    lag_time_hours = 10
    peak_hour = 2.0
    weights_1d = triangular_weights(peak_hour, lag_time_hours)
    weights = weights_1d[:, None]
    # weights: hour_index=1, 2, 3, 4...
    # hour_index <= peak_hour (2): weights[0] = 1^2 / (2*2^2) = 1/8 = 0.125
    # hour_index <= peak_hour (2): areaFractionSum = 2^2 / 8 = 0.5. weights[1] = 0.5 - 0.125 = 0.375
    # hour_index > peak_hour (2): falling_limb_index = 2*2 - 3 = 1. areaAlt = 1 - 1^2 / 8 = 0.875. weights[2] = 0.875 - 0.5 = 0.375
    # hour_index > peak_hour (2): falling_limb_index = 2*2 - 4 = 0. areaAlt = 1. weights[3] = 1.0 - 0.875 = 0.125
    # weights = [0.125, 0.375, 0.375, 0.125, 0, 0, ...]

    n_cells = 1
    n_steps = 2
    flow = np.zeros((n_steps, n_cells), dtype=np.float32)
    flow[0, 0] = 1.0  # pulse at t=0

    buffer = np.zeros((lag_time_hours, n_cells), dtype=np.float64)
    apply_triangular(flow, weights, buffer)

    # Pulse at t=0 should contribute to buffer indices 0, 1, 2, 3
    assert buffer[0, 0] == weights_1d[0]
    assert buffer[1, 0] == weights_1d[1]
    assert buffer[2, 0] == weights_1d[2]
    assert buffer[3, 0] == weights_1d[3]

    # Create pulse at t=1
    buffer[:] = 0.0
    flow[:] = 0.0
    flow[1, 0] = 1.0
    apply_triangular(flow, weights, buffer)

    # Pulse at t=1 should contribute to buffer indices 1, 2, 3, 4
    assert buffer[1, 0] == weights_1d[0]
    assert buffer[2, 0] == weights_1d[1]
    assert buffer[3, 0] == weights_1d[2]
    assert buffer[4, 0] == weights_1d[3]


def test_triangular_weights_peak_sum() -> None:
    """Test that the weights peak at the expected lag and sum to 1."""
    lag_time_hours = 24
    peak_hour = 6.0
    weights = triangular_weights(peak_hour, lag_time_hours)

    # Peak should be around index 5 or 6
    assert np.argmax(weights) in [5, 6]
    assert np.isclose(weights.sum(), 1.0, atol=1e-10)

    # Check that weights increase up to peak and then decrease
    for i in range(1, 6):
        assert weights[i] > weights[i - 1]
    for i in range(7, 16):
        assert weights[i] < weights[i - 1]

    # Beyond Tb = (8/3)*peak = 16 hours, weights should be 0
    assert np.all(weights[16:] == 0.0)

    # Plot the weights for multiple peak scenarios
    plt.figure(figsize=(10, 6))
    peaks_to_test = [3.0, 6.0, 9.0, 12.0]
    for p in peaks_to_test:
        w = triangular_weights(p, lag_time_hours)
        plt.plot(range(1, lag_time_hours + 1), w, "o-", label=f"Peak={p}h")

    plt.xlabel("Lag Time [hours]")
    plt.ylabel("Weight [-]")
    plt.title(f"Triangular Weight Distributions (Lag={lag_time_hours}h)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_folder_runoff_concentrator / "triangular_weights_scenarios.svg")
    plt.close()


def test_apply_triangular_visualization() -> None:
    """Visualize how a pulse of inflow is distributed over time in the buffer."""
    lag_time_hours = 48
    peak_hour = 6.0
    n_cells = 1
    n_steps = 24

    weights = triangular_weights(peak_hour, lag_time_hours)[:, None]

    # 1. Test a single pulse at t=0
    flow_pulse = np.zeros((n_steps, n_cells), dtype=np.float32)
    flow_pulse[0, 0] = 1.0
    buffer_pulse = np.zeros((lag_time_hours, n_cells), dtype=np.float64)
    apply_triangular(flow_pulse, weights, buffer_pulse)

    # 2. Test a constant inflow over the first 6 hours
    flow_const = np.zeros((n_steps, n_cells), dtype=np.float32)
    flow_const[:6, 0] = 1.0
    buffer_const = np.zeros((lag_time_hours, n_cells), dtype=np.float64)
    apply_triangular(flow_const, weights, buffer_const)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(
        range(lag_time_hours),
        buffer_pulse[:, 0],
        "b-",
        label="Response to Pulse at t=0",
    )
    plt.title("Runoff Concentration: Response to Unit Input")
    plt.ylabel("Outflow Depth [m]")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(
        range(lag_time_hours),
        buffer_const[:, 0],
        "g-",
        label="Response to Constant Inflow (0-6h)",
    )
    plt.xlabel("Hours from start of day")
    plt.ylabel("Outflow Depth [m]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_folder_runoff_concentrator / "runoff_concentration_response.svg")
    plt.close()


def test_apply_triangular_varying_buffer_lengths_visualization() -> None:
    """Visualize how a rainfall peak is distributed with varying peak hours."""
    lag_time_hours = 96
    n_cells = 1
    n_steps = 24

    # We will simulate exactly 1 unit of runoff at t=0
    flow_pulse = np.zeros((n_steps, n_cells), dtype=np.float32)
    flow_pulse[0, 0] = 1.0

    plt.figure(figsize=(10, 6))

    # 'buffer length' usually implies the total duration of the runoff spread, which is 2 * peak_hour
    peaks_to_test = [3.0, 6.0, 12.0, 24.0]
    for p in peaks_to_test:
        buffer = np.zeros((lag_time_hours, n_cells), dtype=np.float64)
        weights = triangular_weights(p, lag_time_hours)[:, None]
        apply_triangular(flow_pulse, weights, buffer)

        plt.plot(
            range(lag_time_hours),
            buffer[:, 0],
            label=f"Peak Hour = {p}h (Duration = {p * 2}h)",
        )

    plt.xlabel("Time [hours]")
    plt.ylabel("Outflow Fraction [-] / h")
    plt.title("Runoff Concentration Response to 1h Pulse\n(Varying Runoff Peak Hours)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 72)
    plt.tight_layout()
    plt.savefig(output_folder_runoff_concentrator / "runoff_varying_peaks_pulse.svg")
    plt.close()


def test_apply_triangular_varying_peaks_event_visualization() -> None:
    """Visualize how a multi-hour rainfall event is distributed with varying peak hours."""
    lag_time_hours = 96
    n_cells = 1
    n_steps = 24

    # Simulate 6-hour rainfall event, 1 unit of water total spread evenly
    flow_event = np.zeros((n_steps, n_cells), dtype=np.float32)
    flow_event[2:8, 0] = 1.0 / 6.0

    plt.figure(figsize=(10, 6))

    peaks_to_test = [3.0, 6.0, 12.0, 24.0]
    for p in peaks_to_test:
        buffer = np.zeros((lag_time_hours, n_cells), dtype=np.float64)
        weights = triangular_weights(p, lag_time_hours)[:, None]
        apply_triangular(flow_event, weights, buffer)

        plt.plot(
            range(lag_time_hours),
            buffer[:, 0],
            label=f"Peak Hour = {p}h (Duration = {p * 2}h)",
        )

    # Plot the rainfall itself
    rainfall_plot = np.zeros(lag_time_hours)
    rainfall_plot[2:8] = flow_event[2:8, 0]

    # We use a secondary axis or just plot it directly since values are compatible
    plt.bar(
        range(lag_time_hours),
        rainfall_plot,
        alpha=0.3,
        color="gray",
        label="Rainfall Event (t=2 to t=8)",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Outflow Depth [m] / h")
    plt.title(
        "Runoff Concentration Response to 6h Rainfall Event\n(Varying Runoff Peak Hours)"
    )
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 72)
    plt.tight_layout()
    plt.savefig(output_folder_runoff_concentrator / "runoff_varying_peaks_event.svg")
    plt.close()


def test_apply_triangular_successive_events_visualization() -> None:
    """Visualize how successive rainfall events combine in the runoff buffer."""
    lag_time_hours = 120
    n_cells = 1
    n_steps = 72

    # Simulate two rainfall events
    # Event 1: 4 hours of rain (1 unit total)
    # Event 2: 4 hours of heavier rain, 12 hours after the first event ended (2 units total)
    flow_events = np.zeros((n_steps, n_cells), dtype=np.float32)
    flow_events[4:8, 0] = 1.0 / 4.0
    flow_events[20:24, 0] = 2.0 / 4.0

    plt.figure(figsize=(10, 6))

    peaks_to_test = [3.0, 6.0, 12.0, 24.0]
    for p in peaks_to_test:
        buffer = np.zeros((lag_time_hours, n_cells), dtype=np.float64)
        weights = triangular_weights(p, lag_time_hours)[:, None]
        apply_triangular(flow_events, weights, buffer)

        plt.plot(
            range(lag_time_hours),
            buffer[:, 0],
            label=f"Peak Hour = {p}h (Duration = {p * 2}h)",
        )

    # Plot the rainfall itself
    rainfall_plot = np.zeros(lag_time_hours)
    rainfall_plot[:n_steps] = flow_events[:, 0]

    plt.bar(
        range(lag_time_hours),
        rainfall_plot,
        alpha=0.3,
        color="gray",
        label="Rainfall Events (t=4-8, t=20-24)",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Depth [m] / h")
    plt.title("Runoff Concentration Response to Successive Rainfall Events")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 96)
    plt.tight_layout()
    plt.savefig(output_folder_runoff_concentrator / "runoff_successive_events.svg")
    plt.close()


def test_compute_slope_dependent_triangular_weights_physics() -> None:
    """Verifies that slope-dependent triangular weights follow hillslope physics.

    1. All columns sum to exactly 1.0 (mass conservation).
    2. Steep slopes produce earlier/sharper peaks than gentle slopes.
    3. Longer hillslopes produce later peaks than shorter hillslopes.
    4. Kerby (1959) overland flow formula naturally operates with factor ~ 1.0.
    """
    lag_time_hours: int = 72
    # 3 cells: steep, moderate, flat
    hillslope_lengths: np.ndarray = np.array([300.0, 300.0, 300.0], dtype=np.float32)
    slopes: np.ndarray = np.array([0.10, 0.02, 0.001], dtype=np.float32)

    retardance: np.ndarray = np.full_like(slopes, 0.4, dtype=np.float32)
    # Standard Kerby formulation with natural concentration_factor=1.0
    weights: np.ndarray = compute_slope_dependent_triangular_weights(
        hillslope_length_m=hillslope_lengths,
        slope_m_per_m=slopes,
        retardance_roughness=retardance,
        lag_time_hours=lag_time_hours,
        concentration_factor=1.0,
    )

    assert weights.shape == (lag_time_hours, 3)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-6)

    # Steeper slopes retain higher proportion in early hours than flat slopes
    assert weights[0, 0] > weights[0, 1] > weights[0, 2]
    # Flatter slopes have later response spread into subsequent hours
    assert weights[1, 2] > weights[1, 0]

    # Test hillslope length dependence: short vs long at constant slope
    lengths_vary: np.ndarray = np.array([100.0, 1000.0], dtype=np.float32)
    slopes_const: np.ndarray = np.array([0.02, 0.02], dtype=np.float32)
    retardance_const: np.ndarray = np.full_like(slopes_const, 0.4, dtype=np.float32)
    weights_length: np.ndarray = compute_slope_dependent_triangular_weights(
        hillslope_length_m=lengths_vary,
        slope_m_per_m=slopes_const,
        retardance_roughness=retardance_const,
        lag_time_hours=lag_time_hours,
        concentration_factor=1.0,
    )
    # Shorter length discharges more immediately than longer length
    assert weights_length[0, 0] > weights_length[0, 1]
    assert weights_length[1, 1] > weights_length[1, 0]


def test_kerby_retardance_coefficients_ordering() -> None:
    """Verifies that TxDOT Table 4-5 retardance coefficients control hydrograph attenuation.

    Pavement (0.02) < Cropland (0.20) < Pasture/Grass (0.40) < Forest (0.60).
    Smoother surfaces must discharge more immediately; rougher forest surfaces delay peak runoff.
    """
    lag_time_hours: int = 48
    n_surfaces: int = 4
    # Identical slope and hillslope length:
    hillslope_lengths: np.ndarray = np.array([400.0] * n_surfaces, dtype=np.float32)
    slopes: np.ndarray = np.array([0.02] * n_surfaces, dtype=np.float32)
    retardance_values: np.ndarray = np.array([0.02, 0.20, 0.40, 0.60], dtype=np.float32)

    weights: np.ndarray = compute_slope_dependent_triangular_weights(
        hillslope_length_m=hillslope_lengths,
        slope_m_per_m=slopes,
        lag_time_hours=lag_time_hours,
        retardance_roughness=retardance_values,
        concentration_factor=1.0,
    )

    # Mass conservation across all surfaces
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-6)

    # Hour 0 discharge fraction must strictly decrease as retardance increases
    # (Pavement > Cropland > Grass > Forest)
    assert weights[0, 0] > weights[0, 1] > weights[0, 2] > weights[0, 3]

    # Later hour outflow must be highest for forest
    assert weights[1, 3] > weights[1, 0]


def test_apply_triangular_2d_weights_mass_balance() -> None:
    """Verifies that 2D cell-specific weights preserve mass exactly in apply_triangular."""
    lag_time_hours: int = 72
    n_cells: int = 10
    n_steps: int = 24

    hillslope_lengths = np.linspace(100.0, 800.0, n_cells, dtype=np.float32)
    slopes = np.linspace(0.002, 0.15, n_cells, dtype=np.float32)
    retardance_cells = np.full_like(slopes, 0.4, dtype=np.float32)

    weights = compute_slope_dependent_triangular_weights(
        hillslope_length_m=hillslope_lengths,
        slope_m_per_m=slopes,
        retardance_roughness=retardance_cells,
        lag_time_hours=lag_time_hours,
        concentration_factor=1.0,
    )

    inflow = np.random.rand(n_steps, n_cells).astype(np.float32)
    buffer = np.zeros((lag_time_hours, n_cells), dtype=np.float32)

    apply_triangular(inflow, weights, buffer)

    expected_mass = inflow.sum(axis=0)
    actual_mass = buffer.sum(axis=0)
    np.testing.assert_allclose(actual_mass, expected_mass, rtol=1e-5)


def test_slope_dependent_weights_visualization() -> None:
    """Visualizes slope and hillslope length controls on the triangular unit hydrograph."""
    lag_time_hours: int = 48
    factor: float = 1.0

    # Panel 1: Varying slope (fixed L = 400m)
    slopes_to_test = [0.002, 0.01, 0.04, 0.15]
    l_fixed = np.array([400.0] * len(slopes_to_test), dtype=np.float32)
    s_vary = np.array(slopes_to_test, dtype=np.float32)
    retardance_slope = np.full_like(s_vary, 0.4, dtype=np.float32)
    weights_slope = compute_slope_dependent_triangular_weights(
        hillslope_length_m=l_fixed,
        slope_m_per_m=s_vary,
        retardance_roughness=retardance_slope,
        lag_time_hours=lag_time_hours,
        concentration_factor=factor,
    )

    # Panel 2: Varying hillslope length (fixed S = 0.03)
    lengths_to_test = [100.0, 250.0, 500.0, 1000.0]
    s_fixed = np.array([0.03] * len(lengths_to_test), dtype=np.float32)
    l_vary = np.array(lengths_to_test, dtype=np.float32)
    retardance_len = np.full_like(s_fixed, 0.4, dtype=np.float32)
    weights_length = compute_slope_dependent_triangular_weights(
        hillslope_length_m=l_vary,
        slope_m_per_m=s_fixed,
        retardance_roughness=retardance_len,
        lag_time_hours=lag_time_hours,
        concentration_factor=factor,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for idx, s in enumerate(slopes_to_test):
        ax1.plot(
            range(1, lag_time_hours + 1),
            weights_slope[:, idx],
            "o-",
            label=f"Slope S = {s:.3f} m/m",
        )
    ax1.set_xlabel("Lag Time [hours]")
    ax1.set_ylabel("Triangular Weight [-]")
    ax1.set_title("Kerby Overland Flow: Slope Sensitivity (L = 400 m)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, 10)

    for idx, length in enumerate(lengths_to_test):
        ax2.plot(
            range(1, lag_time_hours + 1),
            weights_length[:, idx],
            "s-",
            label=f"Hillslope L = {int(length)} m",
        )
    ax2.set_xlabel("Lag Time [hours]")
    ax2.set_ylabel("Triangular Weight [-]")
    ax2.set_title("Kerby Overland Flow: Length Sensitivity (S = 0.03 m/m)")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig(
        output_folder_runoff_concentrator / "slope_dependent_weights_scenarios.svg"
    )
    plt.close()


def test_kerby_tc_and_pulse_routing_visualization() -> None:
    """Visualizes Kerby time of concentration and resulting hydrograph routing response."""
    # 1. Kerby Tc curves in minutes vs length for multiple slopes
    lengths_m = np.linspace(50.0, 1000.0, 50, dtype=np.float32)
    slopes = [0.001, 0.005, 0.02, 0.10]
    n_retardance = 0.4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for s in slopes:
        tc_minutes = 1.44 * ((n_retardance * lengths_m) ** 0.467) * (s**-0.235)
        ax1.plot(lengths_m, tc_minutes, label=f"Slope S = {s:.3f}")

    ax1.set_xlabel("Hillslope Length L [m]")
    ax1.set_ylabel("Kerby Tc [minutes]")
    ax1.set_title("Kerby (1959) Overland Flow Time of Concentration")
    ax1.legend()
    ax1.grid(True)

    # 2. Inflow pulse routing comparison (10 mm unit pulse at t=0)
    lag_time_hours = 24
    n_steps = 24
    # Three landscape archetypes: steep Ardennes hillslope, rolling upland, flat lowland
    test_lengths = np.array([200.0, 400.0, 800.0], dtype=np.float32)
    test_slopes = np.array([0.12, 0.03, 0.002], dtype=np.float32)
    labels = [
        "Steep tributary (L=200m, S=12%)",
        "Rolling upland (L=400m, S=3%)",
        "Flat lowland (L=800m, S=0.2%)",
    ]

    retardance_test = np.full_like(test_slopes, 0.4, dtype=np.float32)
    weights = compute_slope_dependent_triangular_weights(
        hillslope_length_m=test_lengths,
        slope_m_per_m=test_slopes,
        retardance_roughness=retardance_test,
        lag_time_hours=lag_time_hours,
        concentration_factor=1.0,
    )

    pulse_inflow = np.zeros((n_steps, 3), dtype=np.float32)
    pulse_inflow[0, :] = 0.010  # 10 mm = 0.010 m
    buffer = np.zeros((lag_time_hours, 3), dtype=np.float32)
    apply_triangular(pulse_inflow, weights, buffer)

    for i in range(3):
        ax2.plot(range(lag_time_hours), buffer[:, i] * 1000.0, "o-", label=labels[i])

    ax2.set_xlabel("Hours from Rainfall Pulse")
    ax2.set_ylabel("Channel Inflow [mm/h]")
    ax2.set_title("Overland Inflow into River Channels (10 mm Pulse at t=0)")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig(
        output_folder_runoff_concentrator / "kerby_tc_and_hydrograph_comparison.svg"
    )
    plt.close()
