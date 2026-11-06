# Explicit Returns/Raises sections follow project instructions.
# ruff: noqa: DOC202, DOC502
"""Check optional interflow travel, complete pulse recovery and restart state."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from geb.hydrology.runoff_concentration import RunoffConcentrator, triangular_weights


def make_concentrator(peak_hours: float) -> RunoffConcentrator:
    """Create a two-cell concentrator with instantaneous surface runoff.

    Args:
        peak_hours: Interflow response peak (hours).

    Returns:
        Concentrator with empty water buffers and no external model data.

    Raises:
        ValueError: If the peak is invalid.
    """
    model: MagicMock = MagicMock()
    model.in_spinup = False
    model.config = {
        "parameters": {"runoff_concentration_factor": 1.0},
        "hydrology": {"routing": {"interflow_peak_hours": peak_hours}},
    }
    hydrology: MagicMock = MagicMock()
    hydrology.grid.var.cell_area = np.ones(2, dtype=np.float32)
    result: RunoffConcentrator = RunoffConcentrator(model, hydrology)
    hydrology.grid.var.overland_flow_buffer = np.zeros(
        (result.lag_time_hours, 2), dtype=np.float32
    )
    hydrology.grid.var.overland_flow_buffer_weights = np.repeat(
        triangular_weights(0.0, result.lag_time_hours)[:, None], 2, axis=1
    )
    return result


def test_zero_delay_preserves_immediate_delivery() -> None:
    """Preserve the legacy hourly sum when the optional peak is zero.

    Returns:
        None.

    Raises:
        AssertionError: If zero delay changes outflow or retains water.
    """
    model: RunoffConcentrator = make_concentrator(0.0)
    runoff: np.ndarray = np.full((24, 2), 0.001, dtype=np.float32)
    interflow: np.ndarray = np.full((24, 2), 0.002, dtype=np.float32)
    baseflow: np.ndarray = np.array([0.0024, 0.0048], dtype=np.float32)
    expected: np.ndarray = runoff + baseflow / 24 + interflow
    np.testing.assert_array_equal(model.step(interflow, baseflow, runoff), expected)
    assert model.lag_time_hours == 72
    assert not model.grid.var.overland_flow_buffer.any()


def test_last_hour_interflow_preserves_volume_and_restart() -> None:
    """Recover a late-day pulse across days with unchanged baseflow and restart.

    Returns:
        None.

    Raises:
        AssertionError: If pulse water is lost, counted twice or changes on restart.
    """
    model: RunoffConcentrator = make_concentrator(48.0)
    assert model.lag_time_hours == 168
    empty: np.ndarray = np.zeros((24, 2), dtype=np.float32)
    pulse: np.ndarray = empty.copy()
    pulse[-1, :] = [0.01, 0.02]
    baseflow: np.ndarray = np.array([0.0024, 0.0048], dtype=np.float32)
    outflow: np.ndarray = model.step(pulse, baseflow, empty)
    np.testing.assert_allclose(
        outflow[:-1], np.repeat((baseflow / 24)[None, :], 23, axis=0)
    )
    recovered: np.ndarray = outflow.sum(axis=0).astype(float)
    np.testing.assert_allclose(
        recovered + model.grid.var.overland_flow_buffer.sum(axis=0),
        pulse.sum(axis=0) + baseflow,
        atol=1e-8,
    )
    restarted: RunoffConcentrator = make_concentrator(48.0)
    restarted.grid.var.overland_flow_buffer[:] = model.grid.var.overland_flow_buffer
    for _ in range(7):
        continued: np.ndarray = model.step(empty, baseflow, empty)
        np.testing.assert_array_equal(restarted.step(empty, baseflow, empty), continued)
        recovered += continued.sum(axis=0)
    np.testing.assert_allclose(recovered, pulse.sum(axis=0) + baseflow * 8, atol=1e-8)
    assert not model.grid.var.overland_flow_buffer.any()


@pytest.mark.parametrize("peak", [-1.0, float("nan"), float("inf")])
def test_invalid_peak_rejected(peak: float) -> None:
    """Reject invalid interflow peaks before allocating response buffers.

    Args:
        peak: Invalid interflow peak (hours).

    Returns:
        None.

    Raises:
        AssertionError: If an invalid peak is accepted.
    """
    with pytest.raises(ValueError, match="finite and nonnegative"):
        make_concentrator(peak)


def test_old_checkpoint_requires_fresh_spinup() -> None:
    """Reject a short legacy checkpoint instead of truncating delayed interflow.

    Returns:
        None.

    Raises:
        AssertionError: If an incompatible checkpoint is accepted.
    """
    model: RunoffConcentrator = make_concentrator(48.0)
    model.grid.var.overland_flow_buffer = np.zeros((72, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="Rerun spinup"):
        model.step(
            np.zeros((24, 2), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros((24, 2), dtype=np.float32),
        )
