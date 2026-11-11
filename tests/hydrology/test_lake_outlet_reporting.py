"""Compare reported natural-lake releases with the routed water budget."""

# Project instructions require explicit None returns and assertion documentation.
# ruff: noqa: DOC202, DOC502

from unittest.mock import Mock

import numpy as np
import pyflwdir
import pytest
from affine import Affine

from geb.hydrology.routing import Routing, create_river_network
from geb.hydrology.routing.local_inertial import LocalInertial
from geb.hydrology.waterbodies import WaterBodies
from tests.hydrology.routing.test_local_inertial import _make_local_inertial
from tests.hydrology.test_reservoir_routing import _make_reservoir_routing


@pytest.mark.parametrize("head_above_sill_m", [0.0, 1.0])
def test_reported_lake_release_matches_storage_budget(head_above_sill_m: float) -> None:
    """Report real releases, including zero at the sill, instead of incoming flow.

    Args:
        head_above_sill_m: Initial lake depth above its outlet sill (meters).

    Returns:
        None.

    Raises:
        AssertionError: If reported releases differ from lake or domain water budgets.
    """
    routing: Routing
    waterbodies: WaterBodies
    model: Mock
    routing, waterbodies, model = _make_reservoir_routing(existing_lake=True)
    waterbodies.substep = Mock(
        return_value=(
            np.array([0.0, np.nan], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )
    )
    # With no upstream water, lake releases can be checked independently of inflow.
    routing.var.river_storage_m3.fill(0.0)
    waterbodies.var.storage[1] = float(
        (waterbodies.var.outflow_height[1] + head_above_sill_m)
        * waterbodies.var.lake_area[1]
    )
    initial_lake_storage_m3: float = float(waterbodies.var.storage[1])
    zeros: np.ndarray = np.zeros((24, 4), dtype=np.float32)
    boundary_outflow_m3: np.float64
    _, boundary_outflow_m3, _ = routing.step(zeros, zeros[0], zeros[0], zeros)
    released_m3: float = initial_lake_storage_m3 - float(waterbodies.var.storage[1])
    reported_m3: float = float(
        routing.var.discharge_m3_s_per_substep[:, 2].astype(np.float64).sum() * 3600
    )
    np.testing.assert_allclose(reported_m3, released_m3, rtol=1e-6, atol=0.01)
    np.testing.assert_allclose(
        routing.var.discharge_m3_s[2], released_m3 / 86400, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        released_m3,
        routing.var.river_storage_m3.sum() + boundary_outflow_m3,
        rtol=1e-6,
        atol=0.1,
    )
    if head_above_sill_m > 0:
        assert released_m3 > 1000
    else:
        assert reported_m3 == 0


@pytest.mark.parametrize(
    ("destination", "prescribed"),
    [
        ("river", False),
        ("cascade", False),
        ("terminal", False),
        ("river", True),
        ("cascade", True),
        ("terminal", True),
        ("kinematic", True),
    ],
)
def test_release_accumulator_covers_outlet_paths(
    destination: str, prescribed: bool
) -> None:
    """Account for actual transfers and reset reported volumes every routing hour.

    Args:
        destination: Outlet destination: river, cascade, terminal or kinematic reach.
        prescribed: Use a controlled 500 m³ release instead of a natural-lake rating.

    Returns:
        None.

    Raises:
        AssertionError: If actual releases disagree with waterbody budgets or persist
            into a subsequent step without releases.
    """
    network: pyflwdir.FlwdirRaster = create_river_network(
        np.array([[2], [2], [5]], dtype=np.uint8),
        np.ones((3, 1), dtype=bool),
        Affine.identity(),
    )
    waterbody_ids: np.ndarray = np.array(
        [-1, 0, 1]
        if destination == "cascade"
        else [-1, -1, 0]
        if destination == "terminal"
        else [-1, 0, -1],
        dtype=np.int32,
    )
    count: int = 2 if destination == "cascade" else 1
    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=network,
        river_length=np.full(3, 1000.0, dtype=np.float32),
        river_width=np.full(3, 10.0, dtype=np.float32),
        waterbody_ids=waterbody_ids,
        river_ids=np.array([0, 1, 2], dtype=np.int32),
        is_waterbody_outflow=waterbody_ids >= 0,
        use_kinematic=np.array([False, False, destination == "kinematic"]),
        bankfull_river_elevation_m=np.array([10.0, 5.0, 0.0], dtype=np.float32),
        waterbody_lake_area=np.full(count, 10000.0, dtype=np.float32),
        waterbody_lake_factor=np.full(count, 5.0, dtype=np.float32),
        waterbody_outflow_height=np.ones(count, dtype=np.float32),
        waterbody_outflow_bed_elev=np.arange(count, 0, -1, dtype=np.float32) * 5,
    )
    initial_storage_m3: np.ndarray = np.zeros(count, dtype=np.float64)
    initial_storage_m3[0] = 50000.0
    controls: np.ndarray = np.full(
        count, 500.0 if prescribed else np.nan, dtype=np.float32
    )
    zeros: np.ndarray = np.zeros(3, dtype=np.float32)
    result: tuple = router.step(
        zeros,
        zeros.astype(np.float64),
        zeros,
        zeros,
        zeros,
        initial_storage_m3.copy(),
        controls,
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
    )
    expected_release_m3: np.ndarray = initial_storage_m3 + result[6] - result[5]
    np.testing.assert_allclose(
        router.waterbody_outflow_m3, expected_release_m3, rtol=1e-6, atol=0.01
    )
    assert router.waterbody_outflow_m3[0] > 0
    if prescribed:
        np.testing.assert_allclose(router.waterbody_outflow_m3[0], 500.0, atol=0.01)
    if destination == "terminal":
        np.testing.assert_allclose(
            router.waterbody_outflow_m3.sum(), result[7], rtol=1e-6
        )
    # An empty following hour must not retain the previous hour's releases.
    router.step(
        zeros,
        zeros.astype(np.float64),
        zeros,
        zeros,
        zeros,
        np.zeros(count, dtype=np.float64),
        np.zeros(count, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
    )
    np.testing.assert_array_equal(router.waterbody_outflow_m3, 0.0)
