"""Tests for parabolic lake weir factor and outflow estimation in routing."""

import math

import numpy as np

from geb.hydrology.routing.geometry import (
    estimate_outflow_height,
    get_lake_factor,
    get_lake_outlet_width,
)
from geb.hydrology.waterbodies import (
    get_lake_height_above_outflow,
)


def test_get_lake_factor_dimensions_and_calculation() -> None:
    """Test parabolic lake factor calculation using Dresden table formula."""
    river_width = np.array([20.0], dtype=np.float32)
    # lake_factor = a * mu * (b / 2) * sqrt(g / h_ref)
    # with a = 1.0, mu = 0.612, b = 20 m, g = 9.80665 m/s2, h_ref = 1 m:
    # lake_factor = 1.0 * 0.612 * 10 * 3.131557 = 19.165 m/s
    factor = get_lake_factor(river_width=river_width)
    expected = np.float32(0.612 * 10.0 * np.sqrt(9.80665))
    np.testing.assert_allclose(factor, expected, rtol=1e-5)


def test_estimate_outflow_height_and_outflow() -> None:
    """Test estimation of lake outflow height and parabolic weir discharge."""
    lake_area = np.array([3_480_000.0], dtype=np.float32)
    lake_capacity = np.array([7_630_000.0], dtype=np.float64)
    avg_outflow = np.array([2.494], dtype=np.float32)
    river_width = np.array([12.0], dtype=np.float32)

    lake_factor = get_lake_factor(
        river_width=river_width,
        lake_a_factor=np.float32(1.0),
    )

    outflow_height = estimate_outflow_height(
        lake_capacity=lake_capacity,
        lake_factor=lake_factor,
        lake_area=lake_area,
        avg_outflow=avg_outflow,
    )

    # Lake storage when full
    lake_storage = lake_capacity.copy()
    height_above_outflow = get_lake_height_above_outflow(
        lake_storage=lake_storage, lake_area=lake_area, outflow_height=outflow_height
    )

    # Height when full equals height above outflow plus sill height
    assert math.isclose(
        float(((height_above_outflow + outflow_height) * lake_area)[0]),
        float(lake_storage[0]),
        rel_tol=1e-5,
    )

    # Outflow at that water level must match average outflow
    outflow = lake_factor * height_above_outflow**2
    np.testing.assert_allclose(outflow, avg_outflow, rtol=1e-5)


def test_get_lake_outlet_width_uses_downstream_river_or_discharge() -> None:
    """Outlet width is the downstream river width, else alpha * Q^beta."""
    # Cells: 0 lake outflow -> 1 river; 2 lake outflow -> 3 other lake; 4 pit outflow.
    downstream = np.array([1, 2, 3, 3, 4], dtype=np.int32)
    is_river = np.array([False, True, False, False, False])
    top_width = np.array([0.5, 25.0, 0.5, 0.5, 0.5], dtype=np.float32)
    outflow_cell = np.array([0, 2, 4], dtype=np.int32)
    average_q = np.array([10.0, 16.0, 4.0], dtype=np.float32)

    width = get_lake_outlet_width(
        outflow_cell=outflow_cell,
        downstream_cell=downstream,
        is_river_cell=is_river,
        river_top_width_m=top_width,
        average_discharge_m3_s=average_q,
        width_alpha=7.2,
        width_beta=0.5,
    )
    np.testing.assert_allclose(width, [25.0, 7.2 * 4.0, 7.2 * 2.0], rtol=1e-6)
