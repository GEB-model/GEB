"""Tests for hydrological routing."""

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import numpy as np

from geb.hydrology.routing import (
    select_active_rivers,
)
from geb.hydrology.routing.topology import build_routing_topology


def test_select_active_rivers() -> None:
    """Test selecting active rivers inside the model domain.

    Validates that:
    1. Outflow segments marked as downstream are excluded.
    2. Segments represented in grid are retained.
    3. Segments not represented in grid are kept only if they connect to represented upstream reaches.
    """
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "is_downstream_outflow": [False, False, True, False, False],
            "is_further_downstream_outflow": [False, False, False, True, False],
            "represented_in_grid": [True, False, True, True, False],
            "downstream_ID": [1, -1, -1, -1, -1],
        },
        index=[0, 1, 2, 3, 4],
    )

    # Reach 0 is active and represented
    # Reach 1 is not represented, but reach 0 flows into it (downstream_ID of 0 is 1), so reach 1 is kept
    # Reach 2 is downstream outflow, excluded
    # Reach 3 is further downstream outflow, excluded
    # Reach 4 is not represented, and has no upstream represented reach, excluded
    active: gpd.GeoDataFrame = select_active_rivers(rivers)
    assert set(active.index) == {0, 1}

    active_all_unrep: gpd.GeoDataFrame = select_active_rivers(
        rivers, include_rivers_not_represented_in_grid=True
    )
    assert set(active_all_unrep.index) == {0, 1, 4}


def test_build_routing_topology_kinematic_and_inertial_split() -> None:
    """Tests build_routing_topology constructs kinematic and inertial topologies correctly."""
    n_kinematic = 1
    n_inertial = 1
    n_kin_headwater = 1
    upstream_matrix = np.full((10, 2), -1, dtype=np.int32)
    upstream_matrix[0, 1] = 0
    ds_node = np.array([1, -1], dtype=np.int32)
    bed_elevation_perm = np.array([10.0, 5.0], dtype=np.float32)
    river_length_perm = np.array([100.0, 100.0], dtype=np.float32)
    is_pit_perm = np.array([False, True], dtype=bool)
    is_ocean_pit_perm = np.array([False, False], dtype=bool)
    pit_slope_perm = np.array([0.001, 0.001], dtype=np.float32)
    waterbody_ids_perm = np.array([-1, -1], dtype=np.int32)
    use_kinematic_perm = np.array([True, False], dtype=bool)
    is_waterbody_outflow_perm = np.array([False, False], dtype=bool)
    retention_node_id_perm = np.array([-1, -1], dtype=np.int32)
    wb_outflow_indices = np.zeros(0, dtype=np.int32)
    river_storage_alpha_kin = np.array([1.0], dtype=np.float32)
    river_storage_beta_kin = np.array([0.6], dtype=np.float32)

    topo = build_routing_topology(
        n_kinematic=n_kinematic,
        n_inertial=n_inertial,
        n_kin_headwater=n_kin_headwater,
        upstream_matrix=upstream_matrix,
        ds_node=ds_node,
        bed_elevation_perm=bed_elevation_perm,
        river_length_perm=river_length_perm,
        is_pit_perm=is_pit_perm,
        is_ocean_pit_perm=is_ocean_pit_perm,
        pit_slope_perm=pit_slope_perm,
        waterbody_ids_perm=waterbody_ids_perm,
        use_kinematic_perm=use_kinematic_perm,
        is_waterbody_outflow_perm=is_waterbody_outflow_perm,
        retention_node_id_perm=retention_node_id_perm,
        wb_outflow_indices=wb_outflow_indices,
        river_storage_alpha_kin=river_storage_alpha_kin,
        river_storage_beta_kin=river_storage_beta_kin,
    )

    assert topo.kinematic_topology.n_kinematic == 1
    assert len(topo.topology.inertial_up_offsets) == 2
    assert topo.kinematic_topology.n_kin_headwater == 1
    assert isinstance(topo.retention_links.inertial_retention_k, np.ndarray)
    assert isinstance(topo.waterbody_links.inertial_wb_ds_k, np.ndarray)
