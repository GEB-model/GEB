"""Tests for hydrological routing topology and boundary configuration."""

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from geb.hydrology.routing import (
    select_active_rivers,
)
from geb.hydrology.routing.topology import (
    BOUNDARY_INTERNAL,
    BOUNDARY_LAKE,
    BOUNDARY_OCEAN,
    BOUNDARY_PIT,
    build_routing_topology,
    compute_retention_routing,
)


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
    n_kinematic: int = 1
    n_inertial: int = 1
    n_kin_headwater: int = 1
    upstream_matrix: np.ndarray = np.full((10, 2), -1, dtype=np.int32)
    upstream_matrix[0, 1] = 0
    ds_node: np.ndarray = np.array([1, -1], dtype=np.int32)
    bed_elevation_perm: np.ndarray = np.array([10.0, 5.0], dtype=np.float32)
    river_length_perm: np.ndarray = np.array([100.0, 100.0], dtype=np.float32)
    is_pit_perm: np.ndarray = np.array([False, True], dtype=bool)
    is_ocean_pit_perm: np.ndarray = np.array([False, False], dtype=bool)
    pit_slope_perm: np.ndarray = np.array([0.001, 0.001], dtype=np.float32)
    waterbody_ids_perm: np.ndarray = np.array([-1, -1], dtype=np.int32)
    use_kinematic_perm: np.ndarray = np.array([True, False], dtype=bool)
    is_waterbody_outflow_perm: np.ndarray = np.array([False, False], dtype=bool)
    retention_node_id_perm: np.ndarray = np.array([-1, -1], dtype=np.int32)
    wb_outflow_indices: np.ndarray = np.zeros(0, dtype=np.int32)
    river_storage_alpha_kin: np.ndarray = np.array([1.0], dtype=np.float32)
    river_storage_beta_kin: np.ndarray = np.array([0.6], dtype=np.float32)

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
    assert topo.topology.ds_boundary_type[0] == BOUNDARY_PIT


def test_build_routing_topology_validation_and_boundaries() -> None:
    """Tests input validation and boundary condition assignments in build_routing_topology."""
    upstream_matrix: np.ndarray = np.full((5, 3), -1, dtype=np.int32)
    ds_node: np.ndarray = np.array([1, 2, 3, -1, -1], dtype=np.int32)
    bed_elev: np.ndarray = np.array([20.0, 18.0, 15.0, 10.0, 0.0], dtype=np.float32)
    river_len: np.ndarray = np.array(
        [200.0, 200.0, 200.0, 200.0, 200.0], dtype=np.float32
    )
    is_pit: np.ndarray = np.array([False, False, False, True, True], dtype=bool)
    is_ocean: np.ndarray = np.array([False, False, False, False, True], dtype=bool)
    pit_slope: np.ndarray = np.full(5, 0.001, dtype=np.float32)
    wb_ids: np.ndarray = np.array([-1, -1, 5, -1, -1], dtype=np.int32)
    use_kin: np.ndarray = np.array([True, False, False, False, False], dtype=bool)
    is_wb_outflow: np.ndarray = np.zeros(5, dtype=bool)
    ret_id: np.ndarray = np.full(5, -1, dtype=np.int32)
    wb_outflow_idx: np.ndarray = np.zeros(0, dtype=np.int32)
    alpha: np.ndarray = np.array([1.0], dtype=np.float32)
    beta: np.ndarray = np.array([0.6], dtype=np.float32)

    # 1. Invalid river length <= 0
    bad_len: np.ndarray = river_len.copy()
    bad_len[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        build_routing_topology(
            1,
            4,
            1,
            upstream_matrix,
            ds_node,
            bed_elev,
            bad_len,
            is_pit,
            is_ocean,
            pit_slope,
            wb_ids,
            use_kin,
            is_wb_outflow,
            ret_id,
            wb_outflow_idx,
            alpha,
            beta,
        )

    # 2. Inertial reach discharging into a kinematic reach must raise ValueError
    # Reach 1 is inertial and flows into node 3. Setting node 3 to kinematic wave triggers ValueError.
    bad_use_kin_reach: np.ndarray = np.array(
        [False, False, False, True, False], dtype=bool
    )
    with pytest.raises(ValueError, match="cannot discharge into a kinematic reach"):
        build_routing_topology(
            1,
            4,
            1,
            upstream_matrix,
            ds_node,
            bed_elev,
            river_len,
            is_pit,
            is_ocean,
            pit_slope,
            wb_ids,
            bad_use_kin_reach,
            is_wb_outflow,
            ret_id,
            wb_outflow_idx,
            alpha,
            beta,
        )

    # 3. Successful boundary classification
    topo = build_routing_topology(
        1,
        4,
        1,
        upstream_matrix,
        ds_node,
        bed_elev,
        river_len,
        is_pit,
        is_ocean,
        pit_slope,
        wb_ids,
        use_kin,
        is_wb_outflow,
        ret_id,
        wb_outflow_idx,
        alpha,
        beta,
    )
    # Reach 1 (inertial 0) flows into node 2 (wb_id=5) -> boundary lake
    # Reach 2 (inertial 1) flows into node 3 (inertial 2) -> internal
    # Reach 3 (inertial 2) is inland pit -> boundary pit
    # Reach 4 (inertial 3) is ocean pit -> boundary ocean
    assert topo.topology.ds_boundary_type[0] == BOUNDARY_LAKE
    assert topo.topology.ds_boundary_type[1] == BOUNDARY_INTERNAL
    assert topo.topology.ds_boundary_type[2] == BOUNDARY_PIT
    assert topo.topology.ds_boundary_type[3] == BOUNDARY_OCEAN


def test_compute_retention_routing_invariants() -> None:
    """Tests input validation and physical constraints of compute_retention_routing."""
    dt: np.float32 = np.float32(3600.0)

    # Invalid dt
    with pytest.raises(ValueError, match="Time step dt must be positive"):
        compute_retention_routing(
            np.float32(0.0),
            np.float32(100.0),
            np.float32(50.0),
            True,
            np.float32(0.0),
            np.float32(1000.0),
            False,
            np.float32(20.0),
            np.float32(0.5),
            np.float32(100.0),
            np.float32(100.0),
        )

    # Invalid max storage
    with pytest.raises(
        ValueError, match="Maximum retention storage must be non-negative"
    ):
        compute_retention_routing(
            dt,
            np.float32(100.0),
            np.float32(50.0),
            True,
            np.float32(0.0),
            np.float32(-10.0),
            False,
            np.float32(20.0),
            np.float32(0.5),
            np.float32(100.0),
            np.float32(100.0),
        )

    # Diversion non-negative even if available river volume is zero
    div, out, ret_s, riv_v = compute_retention_routing(
        dt,
        np.float32(0.0),
        np.float32(50.0),
        True,
        np.float32(0.0),
        np.float32(1000.0),
        False,
        np.float32(20.0),
        np.float32(0.5),
        np.float32(100.0),
        np.float32(100.0),
    )
    assert div == 0.0
    assert out == 0.0
    assert ret_s == 0.0
    assert riv_v == 0.0
