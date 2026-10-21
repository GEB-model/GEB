"""Check reservoir activation preserves water and stable waterbody IDs."""

from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest

from geb.hydrology.waterbodies import WaterBodies, WaterBodyVariables


@pytest.mark.parametrize("year", [1999, 2000])
def test_reservoir_activation_preserves_storage(year: int) -> None:
    """Keep future reservoirs as rivers and transfer water at construction.

    Args:
        year: Simulation year relative to construction in 2000.
    """
    waterbodies: WaterBodies = WaterBodies.__new__(WaterBodies)
    waterbodies.var = WaterBodyVariables()
    waterbodies.var.construction_year = np.array([0, 2000], dtype=np.int32)
    waterbodies.var.active = np.array([True, False])
    waterbodies.var.all_waterbody_ids = np.array([0, 1, 1, -1], dtype=np.int32)
    waterbodies.var.waterbody_outflow_linear_mapping = np.array([0, 2], dtype=np.int32)
    waterbodies.var.capacity = np.array([100.0, 200.0], dtype=np.float64)
    waterbodies.var.storage = np.array([10.0, 0.0], dtype=np.float64)
    waterbodies.model = Mock(current_time=datetime(year, 1, 1))
    waterbodies.grid = Mock(
        compressed_size=4,
        var=Mock(waterbody_ids=np.array([0, -1, -1, -1], dtype=np.int32)),
    )
    routing: Mock = Mock()
    routing.var.river_storage_m3 = np.array([0.0, 3.0, 4.0, 5.0])
    routing.var.discharge_in_rivers_m3_s_substep = np.zeros(4, dtype=np.float32)
    waterbodies.hydrology = Mock(routing=routing)
    waterbodies.report = Mock()

    waterbodies.step()
    assert waterbodies.var.storage.sum() + routing.var.river_storage_m3.sum() == 22.0
    if year < 2000:
        np.testing.assert_array_equal(
            waterbodies.grid.var.waterbody_ids, [0, -1, -1, -1]
        )
        routing.set_router.assert_not_called()
    else:
        np.testing.assert_array_equal(waterbodies.grid.var.waterbody_ids, [0, 1, 1, -1])
        np.testing.assert_array_equal(waterbodies.var.storage, [10.0, 7.0])
        np.testing.assert_array_equal(
            routing.var.river_storage_m3, [0.0, 0.0, 0.0, 5.0]
        )
        routing.set_router.assert_called_once_with(initialize_storage=False)
        waterbodies.step()
        routing.set_router.assert_called_once()
        assert waterbodies.var.storage[1] == 7.0
