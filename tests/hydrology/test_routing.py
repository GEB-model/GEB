import math

import numpy as np
import numpy.typing as npt
import pyflwdir
import pytest

from geb.hydrology.routing import (
    Accuflux,
    KinematicWave,
    create_river_network,
    get_channel_ratio,
    update_node_kinematic,
)


def test_update_node_kinematic_1() -> None:
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


# def test_storage_discharge_conversions():
#     # one should be the inverse of the other
#     river_alpha = np.array([100, 200, 300, 300])
#     river_beta = np.array([0.1, 0.2, 0.3, 0.3])
#     river_storage = np.array([1000, 2000, 3000, 3000])
#     river_length = np.array([2, 4, 6, 6])
#     waterbody_id = np.array([-1, -1, -1, 0])

#     discharge = calculate_discharge_from_storage(
#         river_storage=river_storage,
#         river_length=river_length,
#         river_alpha=river_alpha,
#         river_beta=river_beta,
#     )

#     storage_check = calculate_river_storage_from_discharge(
#         discharge=discharge,
#         river_length=river_length,
#         river_alpha=river_alpha,
#         river_beta=river_beta,
#         waterbody_id=waterbody_id,
#     )

#     assert np.allclose(river_storage[:-1], storage_check[:-1], rtol=1e-9)
#     # storage should be 0 for the waterbody
#     assert storage_check[-1] == 0


def test_get_channel_ratio() -> None:
    river_width = np.array([1, 2, 3, 4, 5])
    river_length = np.array([1000, 2000, 3000, 4000, 5000])
    cell_area = 10000

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


def test_accuflux(ldd, mask, Q_initial) -> None:
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)

    router: Accuflux = Accuflux(
        dt=1,
        river_network=river_network,
        Q_initial=Q_initial[mask],
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
        sideflow,
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


def test_accuflux_with_longer_dt(ldd, mask, Q_initial) -> None:
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)
    router: Accuflux = Accuflux(
        dt=15,
        river_network=river_network,
        Q_initial=Q_initial[mask],
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
        sideflow,
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


def test_accuflux_with_sideflow(mask, ldd, Q_initial) -> None:
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)
    router = Accuflux(
        dt=1,
        river_network=river_network,
        Q_initial=Q_initial[mask],
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
        sideflow,
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


def test_accuflux_with_water_bodies(mask, ldd, Q_initial) -> None:
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
        Q_initial=Q_initial[mask],
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
        sideflow,
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


def test_kinematic(mask, ldd, Q_initial) -> None:
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)
    router: KinematicWave = KinematicWave(
        river_network=river_network,
        Q_initial=Q_initial[mask],
        river_width=np.full_like(mask, 2.0)[mask],
        river_length=np.full_like(mask, 5.0)[mask],
        river_alpha=np.full_like(mask, 1.0)[mask],
        river_beta=0.6,
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
        sideflow,
        evaporation_m3=np.zeros_like(sideflow, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float64),
    )
