"""Tests for hydrological routing functions in GEB.

Tests for accuflux routing are quite nice and complete. Tests for kinematic wave routing are
more limited, as this is a more complex function. More tests should be added in the future.

"""

import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyflwdir
import pytest

from geb.geb_types import ArrayFloat32, ArrayInt32
from geb.hydrology.routing import (
    Accuflux,
    KinematicWave,
    create_river_network,
    fill_discharge_gaps,
    get_channel_ratio,
    update_node_kinematic,
)


def test_fill_discharge_gaps() -> None:
    """Test the fill_discharge_gaps function to ensure it fills NaN values correctly."""
    Q: ArrayFloat32 = np.array(
        [
            [np.nan, 1, np.nan, 4],
            [7, np.nan, 0, np.nan],
            [5, 2, 0, 0],
            [np.nan, np.nan, 3, np.nan],
            [6, 6, 6, 6],
            [np.nan, 4, 4, 5],
            [np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    ).ravel()

    # rivers are defined up to downstream
    rivers: pd.DataFrame = pd.DataFrame(
        data={
            "river_id": [0, 1, 2, 3, 4, 5, 6],
            "hydrography_linear": [
                [0, 1, 5, 9, 14, 15],  # river with nans at start, middle and end
                [2, 3, 7],  # river with nans at start and end
                [8, 12, 13],  # river with nan at end
                [19, 18, 17, 16],  # river with no nans
                [20, 21, 22, 23],  # river with nan at start
                [24, 25, 26, 27],  # river with all nans
                [4],  # single cell river with no nans
            ],
        },
    )
    Q_filled: ArrayFloat32 = fill_discharge_gaps(
        Q,
        rivers,
        waterbody_ids=np.full(Q.shape, -1, dtype=np.int32),
        outflow_per_waterbody_m3_s=np.array([], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        Q_filled.reshape((7, 4)),
        np.array(
            [
                [1, 1, 4, 4],
                [7, 1, 0, 4],
                [5, 2, 0, 0],
                [5, 5, 3, 3],
                [6, 6, 6, 6],
                [4, 4, 4, 5],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        ),
    )


def test_fill_discharge_gaps_with_waterbodies() -> None:
    """Test the fill_discharge_gaps function to ensure it fills NaN values correctly."""
    Q: ArrayFloat32 = np.array(
        [
            [np.nan, 1, np.nan, 4],
            [7, np.nan, 0, np.nan],
            [5, 2, 0, 0],
            [np.nan, np.nan, 3, np.nan],
            [6, 6, 6, 6],
            [np.nan, 4, 4, 5],
            [np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    ).ravel()

    # rivers are defined up to downstream
    rivers: pd.DataFrame = pd.DataFrame(
        data={
            "river_id": [0, 1, 2, 3, 4, 5, 6],
            "hydrography_linear": [
                [0, 1, 5, 9, 14, 15],  # river with nans at start, middle and end
                [2, 3, 7],  # river with nans at start and end
                [8, 12, 13],  # river with nan at end
                [19, 18, 17, 16],  # river with no nans
                [20, 21, 22, 23],  # river with nan at start
                [24, 25, 26, 27],  # river with all nans
                [4],  # single cell river with no nans
            ],
        },
    )
    waterbody_ids: ArrayInt32 = np.array(
        [
            [0, -1, 2, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, 4, -1, -1],
            [-1, -1, -1, -1],
            [3, -1, -1, -1],
            [-1, 1, -1, -1],
        ],
        dtype=np.int32,
    ).ravel()

    outflow_per_waterbody_m3_s: ArrayFloat32 = np.array(
        [8, 9, 10, 11, 12], dtype=np.float32
    )
    Q_filled: ArrayFloat32 = fill_discharge_gaps(
        Q,
        rivers,
        waterbody_ids=waterbody_ids,
        outflow_per_waterbody_m3_s=outflow_per_waterbody_m3_s,
    )

    np.testing.assert_array_equal(
        Q_filled.reshape((7, 4)),
        np.array(
            [
                [8, 1, 10, 4],
                [7, 1, 0, 4],
                [5, 2, 0, 0],
                [5, 12, 3, 3],
                [6, 6, 6, 6],
                [11, 4, 4, 5],
                [9, 9, 9, 9],
            ],
            dtype=np.float32,
        ),
    )


def test_update_node_kinematic_1() -> None:
    """Test the update_node_kinematic function with known inputs and outputs.

    Test adopted from PCRaster implementation.
    """
    deltaX: int = 10
    Q_new, evaporation_m3_s = update_node_kinematic(
        Qin=0.000201343,
        Qold=0.000115866,
        Qside=-0.000290263 * deltaX,
        evaporation_m3_s=0.0,
        alpha=1.73684,
        beta=0.6,
        deltaT=15,
        deltaX=deltaX,
        epsilon=np.float32(1e-12),
    )
    Q_check = 0.000031450866300937
    assert math.isclose(Q_new, Q_check, abs_tol=1e-12)


def test_update_node_kinematic_2() -> None:
    """Test the update_node_kinematic function with negative sideflow.

    In this function, the sideflow is so strongly negative that the discharge
    should be set to the minimum value by update_node_kinematic (1e-30).
    The 1e-30 is to avoid numerical issues.

    Test adopted from PCRaster implementation.
    """
    deltaX: int = 10
    Q_new, evaporation_m3_s = update_node_kinematic(
        Qin=0,
        Qold=1.11659e-07,
        Qside=-1.32678e-05 * deltaX,
        evaporation_m3_s=0.0,
        alpha=1.6808,
        beta=0.6,
        deltaT=15,
        deltaX=deltaX,
        epsilon=np.float32(1e-12),
    )
    assert math.isclose(Q_new, 1e-30, abs_tol=1e-12)


def test_update_node_kinematic_no_flow() -> None:
    """Test kinematic wave update with zero flow conditions.

    Verifies that when all inflows are zero, the discharge
    is set to the minimum value (1e-30) to avoid numerical issues.
    """
    Q_new, evaporation_m3_s = update_node_kinematic(
        Qin=0,
        Qold=0,
        Qside=0,
        evaporation_m3_s=0.0,
        alpha=1.6808,
        beta=0.6,
        deltaT=15,
        deltaX=10,
        epsilon=np.float32(1e-12),
    )
    assert math.isclose(Q_new, 1e-30, abs_tol=1e-12)


def test_get_channel_ratio() -> None:
    """Test calculation of channel ratio for routing.

    Verifies that the channel ratio is correctly computed
    based on channel width and length parameters.
    """
    river_width = np.array([1, 2, 3, 4, 5])
    river_length = np.array([1000, 2000, 3000, 4000, 5000])
    cell_area = np.full_like(river_width, 10000)

    channel_ratio = get_channel_ratio(
        river_width=river_width, river_length=river_length, cell_area=cell_area
    )

    assert np.allclose(channel_ratio, np.array([0.1, 0.4, 0.9, 1.0, 1.0]))


@pytest.fixture
def ldd() -> npt.NDArray[np.uint8]:
    """Fixture providing a local drainage direction (ldd) array for routing tests.

    Returns:
        np.ndarray: A 4x4 array with ldd values in PCRaster format.

    """
    return np.array(
        [
            [6, 5, 255, 2],
            [6, 8, 7, 2],
            [6, 8, 6, 5],
            [9, 8, 4, 4],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def mask() -> npt.NDArray[np.bool_]:
    """Fixture providing a mask array for routing tests.

    Returns:
        np.ndarray: A 4x4 boolean array indicating valid cells.
    """
    return np.array(
        [
            [True, True, False, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ],
        dtype=bool,
    )


@pytest.fixture
def Q_initial() -> npt.NDArray[np.float32]:
    """Fixture providing a sample discharge array for testing.

    Returns:
        np.ndarray: A 4x4 array with discharge values.
    """
    return np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )


def test_accuflux(
    ldd: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test accumulation flux calculation for routing.

    Verifies that the accuflux function correctly accumulates
    water fluxes through the routing network over time.
    """
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)

    router: Accuflux = Accuflux(
        dt=1,
        river_network=river_network,
        waterbody_id=np.full_like(mask, -1, dtype=np.int32)[mask],
        is_waterbody_outflow=np.zeros_like(mask, dtype=bool)[mask],
    )

    sideflow = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )[mask]

    (
        Q_new,
        actual_evaporation_m3,
        over_abstraction_m3,
        waterbody_storage_m3,
        waterbody_inflow_m3,
        outflow_at_pits_m3,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float64),
    )

    assert (
        Q_new
        == np.array(
            [
                [0, 3, 0, 0],
                [0, 2, 0, 1],
                [0, 3, 0, 2],
                [0, 1, 1, 0],
            ]
        )[mask]
    ).all()
    assert (waterbody_inflow_m3 == 0).all()
    assert outflow_at_pits_m3 == 2
    assert waterbody_storage_m3.size == 0


def test_accuflux_with_longer_dt(
    ldd: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test accumulation flux with longer time step.

    Verifies that accuflux correctly handles longer time steps
    in the routing calculations.
    """
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)
    router: Accuflux = Accuflux(
        dt=15,
        river_network=river_network,
        waterbody_id=np.full_like(mask, -1, dtype=np.int32)[mask],
        is_waterbody_outflow=np.zeros_like(mask, dtype=bool)[mask],
    )

    sideflow = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )[mask]

    (
        Q_new,
        actual_evaporation_m3,
        over_abstraction_m3,
        waterbody_storage_m3,
        waterbody_inflow_m3,
        outflow_at_pits_m3,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float64),
    )

    assert (
        Q_new
        == np.array(
            [
                [0, 3, 0, 0],
                [0, 2, 0, 1],
                [0, 3, 0, 2],
                [0, 1, 1, 0],
            ]
        )[mask]
    ).all()
    assert (waterbody_inflow_m3 == 0).all()
    assert outflow_at_pits_m3 == 30
    assert waterbody_storage_m3.size == 0


def test_accuflux_with_sideflow(
    mask: npt.NDArray[np.bool_],
    ldd: npt.NDArray[np.uint8],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test accumulation flux with side flow inputs.

    Verifies that accuflux correctly incorporates side flow
    contributions into the routing calculations.
    """
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)
    router = Accuflux(
        dt=1,
        river_network=river_network,
        waterbody_id=np.full_like(mask, -1, dtype=np.int32)[mask],
        is_waterbody_outflow=np.zeros_like(mask, dtype=bool)[mask],
    )

    sideflow = np.array(
        [
            [0, 0, 0, 3],
            [0, 0, 0, 2],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]
    )[mask]

    (
        Q_new,
        actual_evaporation_m3,
        over_abstraction_m3,
        waterbody_storage_m3,
        waterbody_inflow_m3,
        outflow_at_pits_m3,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float64),
    )

    assert (
        Q_new
        == np.array(
            [
                [0, 3, 0, 0],
                [0, 2, 0, 4],
                [0, 3, 0, 4],
                [0, 1, 1, 0],
            ]
        )[mask]
    ).all()
    assert outflow_at_pits_m3 == 3  # 2 + 1 from the sideflow
    assert waterbody_storage_m3.size == 0
    assert (waterbody_inflow_m3 == 0).all()

    assert (
        Q_initial[mask].sum() + sideflow.sum() - outflow_at_pits_m3
        == Q_new.sum() + waterbody_storage_m3.sum()
    )


def test_accuflux_with_waterbodies(
    mask: npt.NDArray[np.bool_],
    ldd: npt.NDArray[np.uint8],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test accumulation flux with water bodies.

    Verifies that accuflux correctly handles routing through
    water bodies like lakes and reservoirs.
    """
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)

    waterbody_id = np.array(
        [
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, 0, -1, -1],
            [1, -1, -1, -1],
        ]
    )
    Q_initial[waterbody_id != -1] = np.nan

    router: Accuflux = Accuflux(
        dt=1,
        river_network=river_network,
        is_waterbody_outflow=np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, True, False, False],
                [True, False, False, False],
            ]
        )[mask],
        waterbody_id=waterbody_id[mask],
    )

    sideflow = np.array(
        [
            [0, 0, 0, 3],
            [0, 0, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )[mask]

    waterbody_storage_m3 = np.array([10, 5])
    outflow_per_waterbody_m3 = np.array([7, 2])

    waterbody_storage_m3_pre = waterbody_storage_m3.copy()

    (
        Q_new,
        actual_evaporation_m3,
        over_abstraction_m3,
        waterbody_storage_m3,
        waterbody_inflow_m3,
        outflow_at_pits_m3,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow, dtype=np.float32),
        waterbody_storage_m3=waterbody_storage_m3,
        outflow_per_waterbody_m3=outflow_per_waterbody_m3,
    )

    np.testing.assert_array_equal(
        Q_new,
        np.array(
            [
                [0, 3, 0, 0],
                [0, 8, 0, 4],
                [0, np.nan, 0, 4],
                [np.nan, 1, 1, 0],
            ]
        )[mask],
    )
    assert outflow_at_pits_m3 == 2
    assert (
        waterbody_inflow_m3[0] == 4
    )  # 2 from upstream reservoir, 1 from both other river inflows
    assert waterbody_inflow_m3[1] == 0  # 2 no upstream cells

    assert waterbody_storage_m3[0] == 7  # 10 - 7 + 2 + 1 + 1
    assert waterbody_storage_m3[1] == 3  # 5 - 2
    assert (
        np.nansum(Q_initial[mask])
        + waterbody_storage_m3_pre.sum()
        + sideflow.sum()
        - outflow_at_pits_m3
        == np.nansum(Q_new) + waterbody_storage_m3.sum()
    )


def test_kinematic(
    mask: npt.NDArray[np.bool_],
    ldd: npt.NDArray[np.uint8],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test kinematic wave routing implementation.

    Verifies that the kinematic wave routing correctly
    simulates water flow through river networks.
    """
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)
    router: KinematicWave = KinematicWave(
        river_network=river_network,
        river_width=np.full_like(mask, np.float32(2.0), dtype=np.float32)[mask],
        river_length=np.full_like(mask, np.float32(5.0), dtype=np.float32)[mask],
        river_alpha=np.full_like(mask, np.float32(1.0), dtype=np.float32)[mask],
        river_beta=np.float32(0.6),
        dt=15,
        waterbody_id=np.full_like(mask, -1, dtype=np.int32)[mask],
        is_waterbody_outflow=np.zeros_like(mask, dtype=bool)[mask],
    )

    sideflow = np.array(
        [
            [0, 0, 0, 3],
            [0, 0, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )[mask]

    router.step(
        Q_prev_m3_s=Q_initial[mask],
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float64),
    )
