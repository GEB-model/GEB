"""Tests for parabolic lake weir factor and outflow estimation in routing."""

import math

import numpy as np

from geb.hydrology.routing.geometry import (
    estimate_lake_outflow,
    estimate_outflow_height,
    get_lake_factor,
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
    outflow = estimate_lake_outflow(
        lake_factor=lake_factor, height_above_outflow=height_above_outflow
    )
    np.testing.assert_allclose(outflow, avg_outflow, rtol=1e-5)


def test_estimate_lake_outflow_zero_below_sill() -> None:
    """Test that outflow is zero when stage is at or below sill."""
    lake_factor = np.array([15.0], dtype=np.float32)
    height_zero = np.array([0.0], dtype=np.float32)
    outflow = estimate_lake_outflow(lake_factor, height_zero)
    assert outflow[0] == 0.0
