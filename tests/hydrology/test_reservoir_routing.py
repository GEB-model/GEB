"""Check construction-year activation with the incoming routing solver."""

from datetime import datetime
from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine

from geb.hydrology.routing import Routing, RoutingVariables, create_river_network
from geb.hydrology.waterbodies import LAKE, RESERVOIR, WaterBodies, WaterBodyVariables


@pytest.mark.parametrize("existing_lake", [False, True])
def test_construction_with_incoming_solver(existing_lake: bool) -> None:
    """Keep water and stable model IDs when a future reservoir starts operating.

    Args:
        existing_lake: Start with lake ID 1 active, or with both reservoirs unbuilt.
    """
    routing: Routing = Routing.__new__(Routing)
    waterbodies: WaterBodies = WaterBodies.__new__(WaterBodies)
    model: Mock = Mock(
        in_spinup=False,
        current_time=datetime(1999, 1, 1),
        current_day_of_year=1,
        current_timestep=0,
    )
    model.config = {"parameters": {"lake_outflow_multiplier": 1.0}}
    model.files = {
        "grid": {
            "routing/floodplain_side_slope_m_per_m": "slope",
            "routing/floodplain_profile_widths_m": "profile",
            "routing/bankfull_river_elevation_m": "bed",
        }
    }
    routing.model = waterbodies.model = model
    routing.config = {
        "retention_basins": {"release_threshold_factor": 1.0},
        "river_depth": {"parameters": {"shape_exponent": 0.5}},
        "max_threads": 1,
    }
    grid: Mock = Mock(compressed_size=4)
    grid.var.river_length = np.full(4, 1000.0, dtype=np.float32)
    grid.var.river_mannings = np.full(4, 0.03, dtype=np.float32)
    grid.var.cell_area = np.full(4, 1e6, dtype=np.float32)
    grid.load2d.side_effect = lambda path: (
        np.array([30, 20, 10, 0], dtype=np.float32)
        if path == "bed"
        else np.full(4, 2.0, dtype=np.float32)
    )
    grid.load3d.return_value = np.zeros((10, 4), dtype=np.float32)
    routing.grid = waterbodies.grid = grid
    hydrology: Mock = Mock(routing=routing, waterbodies=waterbodies)
    routing.hydrology = waterbodies.hydrology = hydrology
    model.hydrology = hydrology
    waterbodies.var = WaterBodyVariables()
    waterbodies.var.all_waterbody_ids = np.array([0, -1, 1, -1], dtype=np.int32)
    waterbodies.var.waterbody_outflow_linear_mapping = np.array([0, 2], dtype=np.int32)
    waterbodies.var.waterbody_type = np.array(
        [RESERVOIR, LAKE if existing_lake else RESERVOIR], dtype=np.int32
    )
    waterbodies.var.construction_year = np.array(
        [2000, 0 if existing_lake else 2001], dtype=np.int32
    )
    waterbodies.var.active = np.array([False, existing_lake])
    waterbodies.var.lake_area = np.full(2, 1e6, dtype=np.float32)
    waterbodies.var.capacity = np.full(2, 1e7, dtype=np.float64)
    waterbodies.var.average_discharge_m3_per_s = np.full(2, 1.0, dtype=np.float32)
    waterbodies.var.storage = np.array([0.0, 1000.0 if existing_lake else 0.0])
    waterbodies.set_active_waterbody_maps()
    waterbodies.report = Mock()
    waterbodies.substep = Mock(
        return_value=(np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    )

    routing.var = RoutingVariables()
    routing.var.rivers = gpd.GeoDataFrame(
        {
            "represented_in_grid": [True],
            "return_period_2_years_daily_m3_per_s": [np.nan],
            "downstream_ID": [-1],
            "slope": [0.01],
        },
        index=[10],
    )
    routing.var.river_ids = np.where(grid.var.waterbody_ids == -1, 10, -1).astype(
        np.int32
    )
    routing.var.river_storage_m3 = np.where(grid.var.waterbody_ids == -1, 100.0, 0.0)
    routing.var.water_stage_m = np.array([30, 20, 10, 0], dtype=np.float32)
    routing.var.discharge_in_rivers_m3_s_substep = np.where(
        grid.var.waterbody_ids == -1, 0.0, np.nan
    ).astype(np.float32)
    routing.var.retention_basin_storage_m3 = np.zeros(0, dtype=np.float32)
    routing.var.discharge_m3_s_per_substep = np.zeros((24, 4), dtype=np.float32)
    routing.var.retention_basin_storage_m3_per_substep = np.zeros(
        (24, 4), dtype=np.float32
    )
    routing.var.discharge_m3_s = np.zeros(4, dtype=np.float32)
    routing.var.river_width_alpha = np.ones(4, dtype=np.float32)
    routing.var.river_width_beta = np.full(4, 0.5, dtype=np.float32)
    routing.var.river_storage_alpha = np.ones(4, dtype=np.float32)
    routing.var.river_storage_beta = np.full(4, 0.6, dtype=np.float32)
    routing.retention_max_storage_m3 = np.zeros(0, dtype=np.float32)
    routing.retention_basin_ids = np.full(4, -1, dtype=np.int32)
    routing.controlled_retention = np.zeros(0, dtype=bool)
    routing.retention_activation_threshold_m3_s = np.zeros(0, dtype=np.float32)
    routing.retention_basin_data = gpd.GeoDataFrame()
    routing._pot_evap_ret_buf = routing._retention_act_evap_buf = np.zeros(
        0, dtype=np.float32
    )
    for name in [
        "_pot_evap_buf",
        "_side_flow_buf",
        "_over_abstraction_daily",
        "_evaporation_in_rivers_m3",
    ]:
        setattr(routing, name, np.zeros(4, dtype=np.float32))
    routing._inflow_list = []
    routing.inflow_idx = -1
    routing.calculate_bankfull_width = Mock(
        return_value=np.full(4, 10.0, dtype=np.float32)
    )
    routing.calculate_bankfull_depth = Mock(
        return_value=np.full(4, 2.0, dtype=np.float32)
    )
    routing._update_rivers_geometry = Mock()
    routing.report = routing.update_return_periods = Mock()
    # This straight river keeps the same geometry; only the reservoir mask changes.
    routing.update_river_cells = Mock(
        side_effect=lambda: setattr(
            routing.var,
            "river_ids",
            np.where(grid.var.waterbody_ids == -1, 10, -1).astype(np.int32),
        )
    )
    routing.river_network = create_river_network(
        np.array([[2], [2], [2], [5]], dtype=np.uint8),
        np.ones((4, 1), dtype=bool),
        Affine.identity(),
    )
    routing.set_router(initialize_storage=False)
    assert routing.router.n_wb == int(existing_lake)
    np.testing.assert_array_equal(
        routing._routing_waterbody_indices, [1] if existing_lake else []
    )
    zeros: np.ndarray = np.zeros((24, 4), dtype=np.float32)
    initial_storage: float = float(
        waterbodies.var.storage.sum() + routing.var.river_storage_m3.sum()
    )
    _, loss, _ = routing.step(zeros, zeros[0], zeros[0], zeros)
    assert waterbodies.var.storage[0] == 0
    assert np.isclose(
        waterbodies.var.storage.sum() + routing.var.river_storage_m3.sum() + loss,
        initial_storage,
        atol=0.01,
    )
    before_construction: float = float(
        waterbodies.var.storage.sum() + routing.var.river_storage_m3.sum()
    )
    model.current_time = datetime(2000, 1, 1)
    waterbodies.step()
    assert np.isclose(
        waterbodies.var.storage.sum() + routing.var.river_storage_m3.sum(),
        before_construction,
        atol=0.01,
    )
    assert routing.router.n_wb == 1 + int(existing_lake)
    _, loss, _ = routing.step(zeros, zeros[0], zeros[0], zeros)
    assert np.isclose(
        waterbodies.var.storage.sum() + routing.var.river_storage_m3.sum() + loss,
        before_construction,
        atol=0.01,
    )
