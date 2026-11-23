"""Regression tests for the channel controlling lake outlet release."""

# Explicit Returns sections follow AGENTS.md, including tests returning None.
# ruff: noqa: DOC202

from typing import Any

import numpy as np
import pytest

from geb.hydrology.routing.geometry import get_lake_outlet_width


@pytest.fixture
def outlet_inputs() -> dict[str, Any]:
    """Create river, other-waterbody, pit and boundary outlet cases.

    Returns:
        Width-selector arguments, with discharge in m³/s and widths in meters.
    """
    return {
        "outflow_cell": np.array([0, 2, 4, 5], dtype=np.int32),
        "downstream_cell": np.array([1, 2, 3, 3, 4, -1], dtype=np.int32),
        "is_river_cell": np.array([False, True, False, False, False, False]),
        "river_top_width_m": np.array([0.5, 25, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "average_discharge_m3_s": np.array([10, 16, 4, 9], dtype=np.float32),
        "width_alpha": 7.2,
        "width_beta": 0.5,
    }


def test_lake_width_uses_receiving_river(outlet_inputs: dict[str, Any]) -> None:
    """Use the receiving channel instead of the lake cell's 0.5 m fallback.

    Args:
        outlet_inputs: Valid selector arguments.

    Returns:
        None.
    """
    widths_m: np.ndarray = get_lake_outlet_width(**outlet_inputs)
    np.testing.assert_allclose(widths_m, [25, 28.8, 14.4, 21.6], rtol=1e-6)
    assert widths_m.dtype == np.float32


def test_boundary_does_not_select_last_river(outlet_inputs: dict[str, Any]) -> None:
    """Keep a boundary outlet independent of the final grid cell.

    Args:
        outlet_inputs: Valid selector arguments.

    Returns:
        None.
    """
    outlet_inputs["is_river_cell"][-1] = True
    outlet_inputs["river_top_width_m"][-1] = 1000
    widths_m: np.ndarray = get_lake_outlet_width(**outlet_inputs)
    np.testing.assert_allclose(widths_m[-1], 21.6, rtol=1e-6)


def test_no_lake_outlets(outlet_inputs: dict[str, Any]) -> None:
    """Allow an empty waterbody selection.

    Args:
        outlet_inputs: Valid selector arguments.

    Returns:
        None.
    """
    outlet_inputs["outflow_cell"] = np.empty(0, dtype=np.int32)
    outlet_inputs["average_discharge_m3_s"] = np.empty(0, dtype=np.float32)
    assert get_lake_outlet_width(**outlet_inputs).size == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("outflow_cell", np.array([-1, 2, 4, 5], dtype=np.int32)),
        ("outflow_cell", np.array([0.0, 2.0, 4.0, 5.0])),
        ("downstream_cell", np.array([6, 2, 3, 3, 4, -1], dtype=np.int32)),
        ("downstream_cell", np.array([-2, 2, 3, 3, 4, -1], dtype=np.int32)),
        ("is_river_cell", np.array([True])),
        ("average_discharge_m3_s", np.array([10, np.nan, 4, 9], dtype=np.float32)),
        ("average_discharge_m3_s", np.array([10, 0, 4, 9], dtype=np.float32)),
        ("width_alpha", 0.0),
        ("width_beta", -0.5),
        ("width_beta", np.nan),
        (
            "river_top_width_m",
            np.array([0.5, np.nan, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        ),
    ],
)
def test_invalid_lake_width_inputs(
    outlet_inputs: dict[str, Any], field: str, value: Any
) -> None:
    """Reject malformed dimensions, indices or physical values explicitly.

    Args:
        outlet_inputs: Valid selector arguments before replacement.
        field: Argument to replace.
        value: Invalid replacement value.

    Returns:
        None.
    """
    outlet_inputs[field] = value
    with pytest.raises(ValueError):
        get_lake_outlet_width(**outlet_inputs)
