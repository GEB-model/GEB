"""Check construction years and water storage."""

from datetime import datetime
from functools import partial
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from geb.hydrology.routing import (
    Accuflux,
    KinematicWave,
    Routing,
    RoutingVariables,
    create_river_network,
)
from geb.hydrology.waterbodies import LAKE, RESERVOIR, WaterBodies, WaterBodyVariables


@pytest.fixture(params=[Accuflux, KinematicWave])
def waterbodies(request: pytest.FixtureRequest) -> WaterBodies:
    """Build a four-cell river with two future reservoirs and one existing lake.

    Args:
        request: Router class selected by pytest.

    Returns:
        Waterbodies with a real router and a small test grid.
    """
    bodies: WaterBodies = WaterBodies.__new__(WaterBodies)
    bodies.var = WaterBodyVariables()
    bodies.var.waterbody_type = np.array([RESERVOIR, LAKE, RESERVOIR], dtype=np.int32)
    bodies.var.construction_year = np.array([2000, 0, 2001], dtype=np.int32)
    bodies.var.active = np.array([False, True, False])
    bodies.var.all_waterbody_ids = np.array([0, 1, 2, -1], dtype=np.int32)
    bodies.var.waterbody_outflow_linear_mapping = np.array([0, 1, 2], dtype=np.int32)
    bodies.var.capacity = np.array([1000, 2000, 3000], dtype=np.float64)
    bodies.var.storage = np.array([0, 200, 0], dtype=np.float64)
    bodies.model = Mock(current_time=datetime(1999, 12, 31))
    bodies.grid = Mock(
        compressed_size=4,
        var=SimpleNamespace(
            discharge_in_rivers_m3_s_substep=np.array(
                [2, np.nan, 3, 1], dtype=np.float32
            ),
            river_storage_alpha=np.ones(4, dtype=np.float32),
            river_storage_beta=np.full(4, 0.6, dtype=np.float32),
        ),
    )
    bodies.set_active_waterbody_maps()
    router: Accuflux | KinematicWave = request.param(
        dt=3600,
        river_network=create_river_network(
            np.array([[6, 6, 6, 5]], dtype=np.uint8),
            np.ones((1, 4), dtype=bool),
        ),
        river_length=np.ones(4, dtype=np.float32),
        waterbody_id=bodies.grid.var.waterbody_ids,
        is_waterbody_outflow=bodies.grid.var.waterbody_outflow_points != -1,
        retention_max_storage_m3=np.zeros(0, dtype=np.float32),
        retention_node_id=np.full(4, -1, dtype=np.int32),
        controlled_retention=np.zeros(0, dtype=bool),
        retention_activation_threshold_controlled_m3_s=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_uncontrolled_m3_s=np.zeros(0, dtype=np.float32),
    )
    bodies.hydrology = Mock(routing=Mock(router=router))
    return bodies


def test_future_waterbodies_remain_river_cells(waterbodies: WaterBodies) -> None:
    """Keep future reservoirs empty and allow river flow.

    Args:
        waterbodies: Small river and waterbody model.
    """
    waterbodies.step()
    np.testing.assert_array_equal(waterbodies.grid.var.waterbody_ids, [-1, 1, -1, -1])
    np.testing.assert_array_equal(
        cast(Mock, waterbodies.grid).var.capacity, [0, 2000, 0, 0]
    )
    np.testing.assert_array_equal(waterbodies.var.storage, [0, 200, 0])
    discharge: np.ndarray = np.array([2, 9, 3, 1], dtype=np.float32)
    waterbodies.map_to_grid_outflow(
        np.array([0, 4, 0], dtype=np.float32), out=discharge
    )
    np.testing.assert_array_equal(discharge, [2, 4, 3, 1])


def test_construction_transfers_river_storage_once(waterbodies: WaterBodies) -> None:
    """Start on January 1 using the water already in the river.

    Args:
        waterbodies: Small river and waterbody model.
    """
    router: Accuflux | KinematicWave = waterbodies.hydrology.routing.router
    river_storage: np.ndarray = router.get_total_storage(
        waterbodies.grid.var.discharge_in_rivers_m3_s_substep,
        waterbodies.grid.var.river_storage_alpha,
        waterbodies.grid.var.river_storage_beta,
    )
    total_before: float = float(river_storage.sum() + waterbodies.var.storage.sum())
    cast(Mock, waterbodies.model).current_time = datetime(2000, 1, 1)
    waterbodies.step()
    np.testing.assert_array_equal(waterbodies.grid.var.waterbody_ids, [0, 1, -1, -1])
    assert waterbodies.var.storage[0] == river_storage[0]
    assert waterbodies.var.storage[2] == 0
    assert np.isnan(waterbodies.grid.var.discharge_in_rivers_m3_s_substep[0])
    total_after: float = float(
        waterbodies.var.storage.sum()
        + router.get_total_storage(
            waterbodies.grid.var.discharge_in_rivers_m3_s_substep,
            waterbodies.grid.var.river_storage_alpha,
            waterbodies.grid.var.river_storage_beta,
        ).sum()
    )
    assert total_after == pytest.approx(total_before)
    waterbodies.step()
    assert waterbodies.var.storage[0] == river_storage[0]
    cast(Mock, waterbodies.model).current_time = datetime(2001, 1, 1)
    waterbodies.step()
    assert waterbodies.var.storage[2] == river_storage[2]
    np.testing.assert_array_equal(waterbodies.is_active, [True, True, True])


def test_older_saved_state_stays_active(waterbodies: WaterBodies) -> None:
    """Keep old saved states working without construction years.

    Args:
        waterbodies: Small river and waterbody model.
    """
    del waterbodies.var.active
    del waterbodies.var.construction_year
    waterbodies.set_active_waterbody_maps()
    waterbodies.step()
    np.testing.assert_array_equal(waterbodies.is_active, [True, True, True])


def test_no_active_waterbodies(waterbodies: WaterBodies) -> None:
    """Allow a grid with no active lakes or reservoirs.

    Args:
        waterbodies: Small river and waterbody model.
    """
    waterbodies.var.active[:] = False
    waterbodies.var.construction_year[:] = 2000
    waterbodies.var.storage[:] = 0
    waterbodies.set_active_waterbody_maps()
    waterbodies.step()
    np.testing.assert_array_equal(waterbodies.grid.var.waterbody_ids, [-1, -1, -1, -1])
    np.testing.assert_array_equal(
        waterbodies.grid.var.waterbody_outflow_points, [-1, -1, -1, -1]
    )


def test_routing_skips_future_storage(waterbodies: WaterBodies) -> None:
    """Route future reservoir cells as rivers.

    Args:
        waterbodies: Small river and waterbody model.
    """
    result: tuple = waterbodies.hydrology.routing.router.step(
        Q_prev_m3_s=waterbodies.grid.var.discharge_in_rivers_m3_s_substep,
        sideflow_m3=np.ones(4, dtype=np.float32),
        evaporation_m3=np.zeros(4, dtype=np.float32),
        waterbody_storage_m3=waterbodies.var.storage,
        outflow_per_waterbody_m3=np.array([0, 4, 0], dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        river_storage_alpha=waterbodies.grid.var.river_storage_alpha,
        river_storage_beta=waterbodies.grid.var.river_storage_beta,
    )
    assert np.isfinite(result[0][[0, 2]]).all()
    np.testing.assert_array_equal(waterbodies.var.storage[[0, 2]], [0, 0])
    np.testing.assert_array_equal(result[4][[0, 2]], [0, 0])


@pytest.mark.parametrize("has_years", [True, False])
def test_spinup_uses_construction_year(
    waterbodies: WaterBodies, has_years: bool
) -> None:
    """Start future reservoirs empty during spinup.

    Args:
        waterbodies: Small river and waterbody model.
        has_years: Whether input data include GDW construction years.
    """
    del waterbodies.var.active
    waterbodies.model.config = {"parameters": {"lake_outflow_multiplier": 1.0}}
    waterbodies.model.files = {"grid": {"waterbodies/waterbody_id": "unused"}}
    original_ids: np.ndarray = np.array([10, 20, 30, -1], dtype=np.int32)
    waterbodies.grid.load2d = Mock(return_value=original_ids)
    waterbodies.get_outflows = Mock(return_value=original_ids)
    data: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "waterbody_type": [RESERVOIR, LAKE, RESERVOIR],
            "volume_total": [1000.0, 2000.0, 3000.0],
            "average_area": [10.0, 20.0, 30.0],
            "average_discharge": [1.0, 2.0, 3.0],
        }
    )
    if has_years:
        data["gdw_construction_year"] = [2000, 2050, 2001]
    waterbodies.load_waterbody_data = Mock(return_value=data)
    waterbodies.spinup()
    np.testing.assert_array_equal(
        waterbodies.is_active, [not has_years, True, not has_years]
    )
    if has_years:
        np.testing.assert_array_equal(waterbodies.var.storage[[0, 2]], [0, 0])
        assert np.isfinite(
            waterbodies.grid.var.discharge_in_rivers_m3_s_substep[[0, 2]]
        ).all()
    else:
        assert waterbodies.var.storage[0] == 500


@pytest.mark.parametrize("all_future", [True, False])
def test_daily_routing_before_construction(
    waterbodies: WaterBodies, all_future: bool
) -> None:
    """Check storage and evaporation before construction for a full day.

    Args:
        waterbodies: Small river and waterbody model.
        all_future: Whether all waterbodies are future reservoirs.
    """
    grid: Mock = cast(Mock, waterbodies.grid)
    model: Mock = cast(Mock, waterbodies.model)
    hydrology: Mock = cast(Mock, waterbodies.hydrology)
    routing: Routing = Routing.__new__(Routing)
    routing.router = hydrology.routing.router
    routing.model = model
    routing.hydrology = hydrology
    routing.grid = grid
    routing.var = RoutingVariables()
    routing.var.river_ids = np.arange(4, dtype=np.int32)
    routing.var.sum_of_all_discharge_steps = np.zeros(4, dtype=np.float64)
    routing.var.discharge_step_count = 0
    routing.retention_basin_data = pd.DataFrame()
    routing.retention_basin_ids = np.full(4, -1, dtype=np.int32)
    routing.inflow = {}
    routing.inflow_idx = -1
    hydrology.routing = routing
    hydrology.grid = grid
    hydrology.waterbodies = waterbodies
    model.hydrology = hydrology
    model.in_spinup = False
    model.current_timestep = 1
    model.current_day_of_year = 365
    grid.full_compressed = Mock(side_effect=partial(np.full, 4))
    grid.var.cell_area = np.ones(4, dtype=np.float32)
    grid.var.river_length = np.ones(4, dtype=np.float32)
    grid.var.river_width_alpha = np.ones(4, dtype=np.float32)
    grid.var.river_width_beta = np.full(4, 0.5, dtype=np.float32)
    grid.var.retention_basin_storage_m3 = np.zeros(0, dtype=np.float32)
    grid.var.discharge_m3_s_per_substep = np.zeros((24, 4), dtype=np.float32)
    grid.var.discharge_in_rivers_m3_s_substep[:] = 0
    waterbodies.var.lake_area = np.full(3, 10, dtype=np.float32)
    waterbodies.var.lake_factor = np.ones(3, dtype=np.float32)
    waterbodies.var.outflow_height = np.zeros(3, dtype=np.float32)
    if all_future:
        waterbodies.var.waterbody_type[:] = RESERVOIR
        waterbodies.var.active[:] = False
        waterbodies.var.storage[:] = 0
    waterbodies.set_active_waterbody_maps()
    grid.var.discharge_in_rivers_m3_s_substep[grid.var.waterbody_ids != -1] = np.nan
    routing.router.waterbody_id = grid.var.waterbody_ids
    routing.router.is_waterbody_outflow = grid.var.waterbody_outflow_points != -1
    model.agents.reservoir_operators.release.return_value = (
        np.zeros(waterbodies.is_reservoir.sum(), dtype=np.float32),
        np.zeros(waterbodies.is_reservoir.sum(), dtype=np.float32),
    )
    _, _, waterbody_evaporation = routing.step(
        total_runoff_m=np.full((24, 4), 0.1, dtype=np.float32),
        channel_abstraction_m3=np.zeros(4, dtype=np.float32),
        return_flow=np.zeros(4, dtype=np.float32),
        reference_evapotranspiration_water_m=np.full((24, 4), 1e-6, dtype=np.float32),
    )
    assert np.isfinite(grid.var.discharge_m3_s_per_substep).all()
    assert (waterbodies.var.storage[~waterbodies.is_active] == 0).all()
    if all_future:
        assert waterbody_evaporation == 0
