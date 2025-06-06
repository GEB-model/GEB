import math

import numpy as np

from geb.hydrology.routing import (
    calculate_discharge_from_storage,
    calculate_river_storage_from_discharge,
    get_channel_ratio,
    update_discharge,
)


def test_update_discharge_1():
    Q_new = update_discharge(
        Qin=0.000201343,
        Qold=0.000115866,
        q=-0.000290263,
        alpha=1.73684,
        beta=0.6,
        deltaT=15,
        deltaX=10,
        epsilon=1e-12,
    )
    Q_check = 0.000031450866300937
    assert math.isclose(Q_new, Q_check, abs_tol=1e-12)


def test_update_discharge_2():
    Q_new = update_discharge(
        Qin=0,
        Qold=1.11659e-07,
        q=-1.32678e-05,
        alpha=1.6808,
        beta=0.6,
        deltaT=15,
        deltaX=10,
        epsilon=1e-12,
    )
    assert Q_new == 1e-30


def test_update_discharge_no_flow():
    Q_new = update_discharge(
        Qin=0,
        Qold=0,
        q=0,
        alpha=1.6808,
        beta=0.6,
        deltaT=15,
        deltaX=10,
        epsilon=1e-12,
    )
    assert Q_new == 1e-30


def test_storage_discharge_conversions():
    # one should be the inverse of the other
    river_alpha = np.array([100, 200, 300, 300])
    river_beta = np.array([0.1, 0.2, 0.3, 0.3])
    river_storage = np.array([1000, 2000, 3000, 3000])
    river_length = np.array([2, 4, 6, 6])
    waterbody_id = np.array([-1, -1, -1, 0])

    discharge = calculate_discharge_from_storage(
        river_storage=river_storage,
        river_length=river_length,
        river_alpha=river_alpha,
        river_beta=river_beta,
    )

    storage_check = calculate_river_storage_from_discharge(
        discharge=discharge,
        river_length=river_length,
        river_alpha=river_alpha,
        river_beta=river_beta,
        waterbody_id=waterbody_id,
    )

    assert np.allclose(river_storage[:-1], storage_check[:-1], rtol=1e-9)
    # storage should be 0 for the waterbody
    assert storage_check[-1] == 0


def test_get_channel_ratio():
    river_width = np.array([1, 2, 3, 4, 5])
    river_length = np.array([1000, 2000, 3000, 4000, 5000])
    cell_area = 10000

    channel_ratio = get_channel_ratio(
        river_width=river_width, river_length=river_length, cell_area=cell_area
    )

    assert np.allclose(channel_ratio, np.array([0.1, 0.4, 0.9, 1.0, 1.0]))


@pytest.fixture
def ldd():
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
def mask():
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
def Q_initial():
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


def test_accuflux(ldd, mask, Q_initial):
    router = Accuflux(
        dt=1,
        ldd=ldd[mask],
        mask=mask,
        Q_initial=Q_initial[mask],
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

    Q_new, over_abstraction_m3, waterbody_storage_m3, outflow_at_pits_m3 = router.step(
        sideflow,
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
    assert outflow_at_pits_m3 == 2
    assert waterbody_storage_m3.size == 0


def test_accuflux_with_longer_dt(ldd, mask, Q_initial):
    router = Accuflux(
        dt=15,
        ldd=ldd[mask],
        mask=mask,
        Q_initial=Q_initial[mask],
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

    Q_new, over_abstraction_m3, waterbody_storage_m3, outflow_at_pits_m3 = router.step(
        sideflow,
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
    assert outflow_at_pits_m3 == 30
    assert waterbody_storage_m3.size == 0


def test_accuflux_with_sideflow(mask, ldd, Q_initial):
    router = Accuflux(
        dt=1,
        ldd=ldd[mask],
        mask=mask,
        Q_initial=Q_initial[mask],
    )

    sideflow = np.array(
        [
            [0, 0, 0, 3],
            [0, 0, 0, 2],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]
    )

    Q_new, over_abstraction_m3, waterbody_storage_m3, outflow_at_pits_m3 = router.step(
        sideflow[mask],
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

    assert (
        Q_initial[mask].sum() + sideflow[mask].sum() - outflow_at_pits_m3
        == Q_new.sum() + waterbody_storage_m3.sum()
    )


def test_accuflux_with_water_bodies(mask, ldd, Q_initial):
    waterbody_id = np.array(
        [
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, 0, -1, -1],
            [1, -1, -1, -1],
        ]
    )
    Q_initial[waterbody_id != -1] = 0

    router = Accuflux(
        dt=1,
        ldd=ldd[mask],
        mask=mask,
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
    )

    waterbody_storage_m3 = np.array([10, 5])
    outflow_per_waterbody_m3 = np.array([7, 2])

    waterbody_storage_m3_pre = waterbody_storage_m3.copy()

    Q_new, over_abstraction_m3, waterbody_storage_m3, outflow_at_pits_m3 = router.step(
        sideflow[mask],
        waterbody_storage_m3,
        outflow_per_waterbody_m3,
    )

    assert (
        Q_new
        == np.array(
            [
                [0, 3, 0, 0],
                [0, 8, 0, 4],
                [0, 0, 0, 4],
                [0, 1, 1, 0],
            ]
        )[mask]
    ).all()
    assert outflow_at_pits_m3 == 2

    assert waterbody_storage_m3[0] == 7  # 10 - 7 + 2 + 1
    assert waterbody_storage_m3[1] == 3  # 5 - 2
    assert (
        Q_initial[mask].sum()
        + waterbody_storage_m3_pre.sum()
        + sideflow.sum()
        - outflow_at_pits_m3
        == Q_new.sum() + waterbody_storage_m3.sum()
    )


def test_kinematic(mask, ldd, Q_initial):
    router = KinematicWave(
        ldd=ldd[mask],
        mask=mask,
        Q_initial=Q_initial[mask],
        river_width=np.full_like(mask, 2.0)[mask],
        river_length=np.full_like(mask, 5.0)[mask],
        river_alpha=np.full_like(mask, 1.0)[mask],
        river_beta=0.6,
        dt=15,
    )

    sideflow = np.array(
        [
            [0, 0, 0, 3],
            [0, 0, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    router.step(
        sideflow[mask],
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float64),
    )
