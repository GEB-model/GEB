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
