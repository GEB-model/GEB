"""Tests for the RunoffConcentrator module."""

import matplotlib.pyplot as plt
import numpy as np

from geb.hydrology.runoff_concentration import (
    advance_buffer,
    apply_triangular,
    triangular_weights,
)

from ...testconfig import output_folder


def test_triangular_weights_normalization() -> None:
    """Test that triangular weights always sum to approximately 1.0."""
    lag_time_hours = 48
    peak_hour = 12.0
    weights = triangular_weights(peak_hour, lag_time_hours)
    assert len(weights) == lag_time_hours
    assert np.isclose(weights.sum(), 1.0, atol=1e-10)


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

    weights = triangular_weights(peak_hour, lag_time_hours)
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
    weights = triangular_weights(peak_hour, lag_time_hours)
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
    assert buffer[0, 0] == weights[0]
    assert buffer[1, 0] == weights[1]
    assert buffer[2, 0] == weights[2]
    assert buffer[3, 0] == weights[3]

    # Create pulse at t=1
    buffer[:] = 0.0
    flow[:] = 0.0
    flow[1, 0] = 1.0
    apply_triangular(flow, weights, buffer)

    # Pulse at t=1 should contribute to buffer indices 1, 2, 3, 4
    assert buffer[1, 0] == weights[0]
    assert buffer[2, 0] == weights[1]
    assert buffer[3, 0] == weights[2]
    assert buffer[4, 0] == weights[3]


def test_triangular_weights_peak_sum() -> None:
    """Test that the weights peak at the expected lag and sum to 1."""
    lag_time_hours = 24
    peak_hour = 6.0
    weights = triangular_weights(peak_hour, lag_time_hours)

    # Peak should be at index peak_hour-1 = 5
    assert np.argmax(weights) == 5
    assert np.isclose(weights.sum(), 1.0, atol=1e-10)

    # Check that weights increase up to peak and then decrease
    for i in range(1, 6):
        assert weights[i] > weights[i - 1]
    for i in range(7, 11):
        assert weights[i] < weights[i - 1]

    # Beyond 2*peak, weights should be 0
    # hour_index > 2*peak => hour_index > 12 => index > 11
    assert np.all(weights[12:] == 0.0)

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
    plt.savefig(output_folder / "triangular_weights_scenarios.svg")
    plt.close()


def test_apply_triangular_visualization() -> None:
    """Visualize how a pulse of inflow is distributed over time in the buffer."""
    lag_time_hours = 48
    peak_hour = 6.0
    n_cells = 1
    n_steps = 24

    weights = triangular_weights(peak_hour, lag_time_hours)

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
    plt.savefig(output_folder / "runoff_concentration_response.svg")
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
        weights = triangular_weights(p, lag_time_hours)
        apply_triangular(flow_pulse, weights, buffer)
        
        plt.plot(range(lag_time_hours), buffer[:, 0], label=f"Peak Hour = {p}h (Duration = {p*2}h)")

    plt.xlabel("Time [hours]")
    plt.ylabel("Outflow Fraction [-] / h")
    plt.title("Runoff Concentration Response to 1h Pulse\n(Varying Runoff Peak Hours)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 72)
    plt.tight_layout()
    plt.savefig(output_folder / "runoff_varying_peaks_pulse.svg")
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
        weights = triangular_weights(p, lag_time_hours)
        apply_triangular(flow_event, weights, buffer)
        
        plt.plot(range(lag_time_hours), buffer[:, 0], label=f"Peak Hour = {p}h (Duration = {p*2}h)")

    # Plot the rainfall itself
    rainfall_plot = np.zeros(lag_time_hours)
    rainfall_plot[2:8] = flow_event[2:8, 0]
    
    # We use a secondary axis or just plot it directly since values are compatible
    plt.bar(range(lag_time_hours), rainfall_plot, alpha=0.3, color='gray', label='Rainfall Event (t=2 to t=8)')

    plt.xlabel("Time [hours]")
    plt.ylabel("Outflow Depth [m] / h")
    plt.title("Runoff Concentration Response to 6h Rainfall Event\n(Varying Runoff Peak Hours)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 72)
    plt.tight_layout()
    plt.savefig(output_folder / "runoff_varying_peaks_event.svg")
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
        weights = triangular_weights(p, lag_time_hours)
        apply_triangular(flow_events, weights, buffer)
        
        plt.plot(range(lag_time_hours), buffer[:, 0], label=f"Peak Hour = {p}h (Duration = {p*2}h)")

    # Plot the rainfall itself
    rainfall_plot = np.zeros(lag_time_hours)
    rainfall_plot[:n_steps] = flow_events[:, 0]
    
    plt.bar(range(lag_time_hours), rainfall_plot, alpha=0.3, color='gray', label='Rainfall Events (t=4-8, t=20-24)')

    plt.xlabel("Time [hours]")
    plt.ylabel("Depth [m] / h")
    plt.title("Runoff Concentration Response to Successive Rainfall Events")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 96)
    plt.tight_layout()
    plt.savefig(output_folder / "runoff_successive_events.svg")
    plt.close()
