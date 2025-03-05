import math
import numpy as np

from geb.hydrology.routing import (
    calculate_river_storage_from_discharge,
    calculate_discharge_from_storage,
    get_channel_ratio,
)


def test_storage_discharge_conversions():
    # one should be the inverse of the other
    river_alpha = 100
    river_beta = 0.1
    river_storage = 1000
    river_length = 2

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
    )

    assert math.isclose(river_storage, storage_check, rel_tol=1e-9)


def test_get_channel_ratio():
    river_width = np.array([1, 2, 3, 4, 5])
    river_length = np.array([1000, 2000, 3000, 4000, 5000])
    cell_area = 10000

    channel_ratio = get_channel_ratio(
        river_width=river_width, river_length=river_length, cell_area=cell_area
    )

    assert np.allclose(channel_ratio, np.array([0.1, 0.4, 0.9, 1.0, 1.0]))
