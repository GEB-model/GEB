"""Network topology graphs, CSR matrices, and boundary classification for river routing."""

from typing import NamedTuple

import numpy as np

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayInt32,
    TwoDArrayInt32,
)

__all__: list[str] = [
    "InertialTopology",
    "KinematicTopology",
    "RoutingTopology",
    "build_routing_topology",
]


class KinematicTopology(NamedTuple):
    """Encapsulates static topology and routing parameters for kinematic reaches.

    Attributes:
        n_kinematic: Number of reaches routed with kinematic wave.
        n_kin_headwater: Number of kinematic headwater reaches (solvable in parallel).
        kin_ds_reach: Downstream kinematic reach index per kinematic reach (-1 if none).
        kin_inflow_m3_s: Buffer accumulating upstream inflow discharges into kinematic reaches (m³/s).
        kin_ds_waterbody_id: Downstream waterbody ID per kinematic reach (-1 if none).
        wb_to_kin_target_reach: Target kinematic reaches receiving waterbody outflow.
        wb_to_kin_wb_id: Waterbody IDs discharging into kinematic reaches.
        kin_wb_inflow_m3: Buffer storing waterbody release volumes per kinematic reach (m³).
        inertial_up_kin_offsets: CSR offsets of upstream kinematic reaches per inertial reach.
        inertial_up_kin_indices: CSR indices of upstream kinematic reaches per inertial reach.
        inertial_up_kin_reach_idx: Precomputed direct upstream kinematic index array.
        river_storage_alpha: Storage equation alpha parameter array.
        river_storage_beta: Storage equation beta parameter array.
        river_length: Channel length array (meters).
        retention_node_id: Retention basin index mapping per node (-1 if none).
    """

    n_kinematic: int
    n_kin_headwater: int
    kin_ds_reach: ArrayInt32
    kin_inflow_m3_s: ArrayFloat32
    kin_ds_waterbody_id: ArrayInt32
    wb_to_kin_target_reach: ArrayInt32
    wb_to_kin_wb_id: ArrayInt32
    kin_wb_inflow_m3: ArrayFloat32
    inertial_up_kin_offsets: ArrayInt32
    inertial_up_kin_indices: ArrayInt32
    inertial_up_kin_reach_idx: ArrayInt32
    river_storage_alpha: ArrayFloat32
    river_storage_beta: ArrayFloat32
    river_length: ArrayFloat32
    retention_node_id: ArrayInt32


class InertialTopology(NamedTuple):
    """Encapsulates network connectivity, CSR graphs, and boundary conditions for inertial reaches.

    Attributes:
        inertial_up_offsets: CSR row offsets for upstream inertial tributary connections.
        inertial_up_indices: CSR column reach indices for upstream inertial tributaries.
        inertial_up_reach_idx: Optimized direct reach index array for single-tributary junctions.
        ds_boundary_type: Boundary condition classification code per reach (0=internal, 2=lake, 3=pit, 4=ocean).
        ds_inertial_k: Downstream reach index within inertial domain (-1 if boundary).
        ds_stage_idx: Unified index into stage buffer for downstream water level lookup.
        ds_bed_elevation: Downstream boundary bed elevation (meters).
        inertial_topo_order: Topologically ordered sequence of reach indices.
    """

    inertial_up_offsets: ArrayInt32
    inertial_up_indices: ArrayInt32
    inertial_up_reach_idx: ArrayInt32
    ds_boundary_type: ArrayInt32
    ds_inertial_k: ArrayInt32
    ds_stage_idx: ArrayInt32
    ds_bed_elevation: ArrayFloat32
    inertial_topo_order: ArrayInt32


class RoutingTopology(NamedTuple):
    """Container grouping all topology graphs, CSR matrices, and cross-domain links.

    Attributes:
        topology: Connectivity and boundary structure for the inertial domain.
        kinematic_topology: Topology and reach parameter tables for the kinematic domain.
        interface_bed_elev_max: Max interface bed elevation between reach and its downstream node (meters).
        inv_dx_interface: Reciprocal of spatial interface length (1/meters).
        inertial_up_count: Tributary junction in-degree count for CFL scaling.
        inertial_retention_k: Inertial reach indices with coupled retention basins.
        inertial_retention_id: Retention basin identifiers coupled to inertial reaches.
        inertial_boundary_outflow_k: Reach indices exiting the hydrodynamic domain for mass balance tracking.
        ds_boundary_type: Boundary condition code array.
        ds_inertial_k: Downstream reach indices within inertial subdomain.
        ds_waterbody_id: Downstream waterbody indices for lake-terminating reaches.
        ds_stage_idx: Unified index into stage buffer for downstream water level lookup.
        ds_bed_elevation: Bed elevation array at downstream boundaries (meters).
        inertial_wb_ds_k: Inertial reaches discharging into downstream waterbodies.
        inertial_wb_ds_id: Waterbody IDs receiving discharges from inertial reaches.
        lake_outflow_target_k: Inertial reach indices receiving lake releases.
        lake_outflow_wb_id: Waterbody IDs supplying releases to inertial reaches.
        wb_to_wb_src_wb: Source waterbody IDs for inter-waterbody transfer links.
        wb_to_wb_tgt_wb: Destination waterbody IDs for inter-waterbody transfer links.
        wb_terminal_wb_ids: Waterbody IDs discharging out of domain (pits).
        inertial_up_offsets: CSR offsets for upstream inertial reaches.
        inertial_up_indices: CSR indices for upstream inertial reaches.
        inertial_up_reach_idx: Direct single-junction upstream inertial reach indices.
    """

    topology: InertialTopology
    kinematic_topology: KinematicTopology
    interface_bed_elev_max: ArrayFloat32
    inv_dx_interface: ArrayFloat32
    inertial_up_count: ArrayInt32
    inertial_retention_k: ArrayInt32
    inertial_retention_id: ArrayInt32
    inertial_boundary_outflow_k: ArrayInt32
    ds_boundary_type: ArrayInt32
    ds_inertial_k: ArrayInt32
    ds_waterbody_id: ArrayInt32
    ds_stage_idx: ArrayInt32
    ds_bed_elevation: ArrayFloat32
    inertial_wb_ds_k: ArrayInt32
    inertial_wb_ds_id: ArrayInt32
    lake_outflow_target_k: ArrayInt32
    lake_outflow_wb_id: ArrayInt32
    wb_to_wb_src_wb: ArrayInt32
    wb_to_wb_tgt_wb: ArrayInt32
    wb_terminal_wb_ids: ArrayInt32
    inertial_up_offsets: ArrayInt32
    inertial_up_indices: ArrayInt32
    inertial_up_reach_idx: ArrayInt32


def build_routing_topology(
    n_kinematic: int,
    n_inertial: int,
    n_kin_headwater: int,
    upstream_matrix: TwoDArrayInt32,
    ds_node: ArrayInt32,
    bed_elevation_perm: ArrayFloat32,
    river_length_perm: ArrayFloat32,
    is_pit_perm: ArrayBool,
    is_ocean_pit_perm: ArrayBool,
    pit_slope_perm: ArrayFloat32,
    waterbody_ids_perm: ArrayInt32,
    use_kinematic_perm: ArrayBool,
    is_waterbody_outflow_perm: ArrayBool,
    retention_node_id_perm: ArrayInt32,
    wb_outflow_indices: ArrayInt32,
    river_storage_alpha_kin: ArrayFloat32,
    river_storage_beta_kin: ArrayFloat32,
    inertial_topo_order: ArrayInt32,
) -> RoutingTopology:
    """Constructs CSR connectivity graphs, boundary classifications, and cross-domain links.

    Args:
        n_kinematic: Number of kinematic reaches.
        n_inertial: Number of local inertial hydrodynamic reaches.
        n_kin_headwater: Number of independent headwater reaches in the kinematic subdomain.
        upstream_matrix: Upstream connectivity matrix in permuted order.
        ds_node: Downstream node index array in permuted order.
        bed_elevation_perm: River reach bed elevations in permuted order (meters).
        river_length_perm: River reach lengths in permuted order (meters).
        is_pit_perm: Boolean array indicating terminal sink / pit reaches.
        is_ocean_pit_perm: Boolean array indicating ocean boundary pit reaches.
        pit_slope_perm: Default slope assigned to terminal pit reaches (m/m).
        waterbody_ids_perm: Waterbody ID map per permuted reach (-1 if absent).
        use_kinematic_perm: Boolean mask indicating kinematic reaches.
        is_waterbody_outflow_perm: Boolean mask locating waterbody outlet reaches.
        retention_node_id_perm: Retention basin ID per reach (-1 if none).
        wb_outflow_indices: Permuted indices of waterbody outlet reaches.
        river_storage_alpha_kin: Storage equation alpha parameter array for kinematic reaches.
        river_storage_beta_kin: Storage equation beta parameter array for kinematic reaches.
        inertial_topo_order: Topologically sorted indices within the inertial domain.

    Returns:
        Structured RoutingTopology containing all precomputed topology objects.

    Raises:
        ValueError: If a reach is neither a pit nor connected to a valid downstream reach.
    """
    ds_boundary_type: ArrayInt32 = np.empty(n_inertial, dtype=np.int32)
    ds_inertial_k: ArrayInt32 = np.full(n_inertial, -1, dtype=np.int32)
    ds_waterbody_id: ArrayInt32 = np.full(n_inertial, -1, dtype=np.int32)
    ds_bed_elevation: ArrayFloat32 = np.empty(n_inertial, dtype=np.float32)

    for k in range(n_inertial):
        i: int = n_kinematic + k
        ds: int = int(ds_node[i])
        bed_node: float = float(bed_elevation_perm[i])
        r_len: float = max(float(river_length_perm[i]), 1.0)

        if is_pit_perm[i]:
            if is_ocean_pit_perm[i]:
                ds_boundary_type[k] = 4
                ds_bed_elevation[k] = np.float32(bed_node)
            else:
                ds_boundary_type[k] = 3
                eff_slope: float = float(pit_slope_perm[i])
                ds_bed_elevation[k] = np.float32(bed_node - eff_slope * r_len)
        elif ds != -1:
            bed_ds: float = float(bed_elevation_perm[ds])
            ds_bed_elevation[k] = np.float32(bed_ds)
            wb_ds: int = int(waterbody_ids_perm[ds])
            if wb_ds != -1:
                ds_boundary_type[k] = 2
                ds_waterbody_id[k] = wb_ds
            else:
                assert not use_kinematic_perm[ds], (
                    f"Inertial reach at sorted index {i} cannot discharge into a kinematic reach."
                )
                ds_boundary_type[k] = 0
                ds_k: int = ds - n_kinematic
                ds_inertial_k[k] = ds_k
        else:
            raise ValueError(
                f"Invalid river topology at sorted index {i}: reach is neither a pit nor connected to a downstream node."
            )

    bed_elevation_inertial: ArrayFloat32 = bed_elevation_perm[
        n_kinematic : n_kinematic + n_inertial
    ]
    river_length_inertial: ArrayFloat32 = river_length_perm[
        n_kinematic : n_kinematic + n_inertial
    ]
    interface_bed_elev_max: ArrayFloat32 = np.maximum(
        bed_elevation_inertial, ds_bed_elevation
    )

    inv_dx_interface: ArrayFloat32 = np.empty(n_inertial, dtype=np.float32)
    for k in range(n_inertial):
        if ds_boundary_type[k] == 0:
            ds_k = ds_inertial_k[k]
            dx: np.float32 = np.float32(0.5) * (
                river_length_inertial[k] + river_length_inertial[ds_k]
            )
        else:
            dx = river_length_inertial[k]
        inv_dx_interface[k] = np.float32(1.0) / max(dx, np.float32(1.0))

    ds_stage_idx: ArrayInt32 = np.full(n_inertial, -1, dtype=np.int32)
    for k in range(n_inertial):
        if ds_boundary_type[k] == 0:
            ds_stage_idx[k] = ds_inertial_k[k]
        elif ds_boundary_type[k] == 2:
            ds_stage_idx[k] = n_inertial + ds_waterbody_id[k]

    max_up: int = upstream_matrix.shape[1]
    inertial_up_offsets: list[int] = [0]
    inertial_up_indices: list[int] = []
    inertial_up_reach_idx: ArrayInt32 = np.full(n_inertial, -1, dtype=np.int32)
    for k in range(n_inertial):
        i = n_kinematic + k
        start_count: int = len(inertial_up_indices)
        for j in range(max_up):
            up_node: int = int(upstream_matrix[i, j])
            if up_node == -1:
                break
            if waterbody_ids_perm[up_node] == -1 and not use_kinematic_perm[up_node]:
                inertial_up_indices.append(up_node - n_kinematic)
        cnt: int = len(inertial_up_indices) - start_count
        if cnt == 1:
            inertial_up_reach_idx[k] = inertial_up_indices[start_count]
        elif cnt > 1:
            inertial_up_reach_idx[k] = -(start_count + 2)
        inertial_up_offsets.append(len(inertial_up_indices))

    inertial_up_offsets_arr: ArrayInt32 = np.array(inertial_up_offsets, dtype=np.int32)
    inertial_up_indices_arr: ArrayInt32 = np.array(inertial_up_indices, dtype=np.int32)

    ret_inertial_k: list[int] = []
    ret_inertial_id: list[int] = []
    for k in range(n_inertial):
        ret_id: int = int(retention_node_id_perm[n_kinematic + k])
        if ret_id != -1:
            ret_inertial_k.append(k)
            ret_inertial_id.append(ret_id)
    inertial_retention_k: ArrayInt32 = np.array(ret_inertial_k, dtype=np.int32)
    inertial_retention_id: ArrayInt32 = np.array(ret_inertial_id, dtype=np.int32)

    wb_ds_k: list[int] = []
    wb_ds_id: list[int] = []
    for k in range(n_inertial):
        if ds_boundary_type[k] == 2:
            wb_ds_k.append(k)
            wb_ds_id.append(int(ds_waterbody_id[k]))
    inertial_wb_ds_k: ArrayInt32 = np.array(wb_ds_k, dtype=np.int32)
    inertial_wb_ds_id: ArrayInt32 = np.array(wb_ds_id, dtype=np.int32)

    lake_outflow_targets: list[int] = []
    lake_outflow_wbs: list[int] = []
    wb_to_wb_src: list[int] = []
    wb_to_wb_tgt: list[int] = []
    wb_to_kin_targets: list[int] = []
    wb_to_kin_wbs: list[int] = []
    wb_term_ids: list[int] = []

    for idx in range(len(wb_outflow_indices)):
        wb_node: int = int(wb_outflow_indices[idx])
        wb_id: int = int(waterbody_ids_perm[wb_node])
        if wb_id == -1:
            continue
        ds = int(ds_node[wb_node])

        if ds == -1 or is_pit_perm[wb_node]:
            wb_term_ids.append(wb_id)
        elif waterbody_ids_perm[ds] != -1:
            target_wb: int = int(waterbody_ids_perm[ds])
            if target_wb != wb_id:
                wb_to_wb_src.append(wb_id)
                wb_to_wb_tgt.append(target_wb)
        elif 0 <= ds < n_kinematic:
            wb_to_kin_targets.append(ds)
            wb_to_kin_wbs.append(wb_id)
        elif n_kinematic <= ds < n_kinematic + n_inertial:
            lake_outflow_targets.append(ds - n_kinematic)
            lake_outflow_wbs.append(wb_id)

    lake_outflow_target_k: ArrayInt32 = np.array(lake_outflow_targets, dtype=np.int32)
    lake_outflow_wb_id: ArrayInt32 = np.array(lake_outflow_wbs, dtype=np.int32)
    wb_to_wb_src_wb: ArrayInt32 = np.array(wb_to_wb_src, dtype=np.int32)
    wb_to_wb_tgt_wb: ArrayInt32 = np.array(wb_to_wb_tgt, dtype=np.int32)
    wb_to_kin_target_reach: ArrayInt32 = np.array(wb_to_kin_targets, dtype=np.int32)
    wb_to_kin_wb_id: ArrayInt32 = np.array(wb_to_kin_wbs, dtype=np.int32)
    kin_wb_inflow_m3: ArrayFloat32 = np.zeros(n_kinematic, dtype=np.float32)
    wb_terminal_wb_ids: ArrayInt32 = np.array(wb_term_ids, dtype=np.int32)

    kin_ds_wb_id: ArrayInt32 = np.full(n_kinematic, -1, dtype=np.int32)
    kin_ds_reach: ArrayInt32 = np.full(n_kinematic, -1, dtype=np.int32)
    for i in range(n_kinematic):
        ds = int(ds_node[i])
        if ds != -1:
            if waterbody_ids_perm[ds] != -1:
                kin_ds_wb_id[i] = waterbody_ids_perm[ds]
            elif 0 <= ds < n_kinematic:
                kin_ds_reach[i] = ds
    kin_inflow_m3_s: ArrayFloat32 = np.zeros(n_kinematic, dtype=np.float32)

    inertial_kin_offsets: list[int] = [0]
    inertial_kin_indices: list[int] = []
    inertial_up_kin_reach_idx: ArrayInt32 = np.full(n_inertial, -1, dtype=np.int32)
    for k in range(n_inertial):
        i = n_kinematic + k
        start_count = len(inertial_kin_indices)
        for j in range(max_up):
            up_node = int(upstream_matrix[i, j])
            if up_node == -1:
                break
            if (
                not is_waterbody_outflow_perm[up_node]
                and waterbody_ids_perm[up_node] == -1
                and use_kinematic_perm[up_node]
            ):
                inertial_kin_indices.append(up_node)
        cnt = len(inertial_kin_indices) - start_count
        if cnt == 1:
            inertial_up_kin_reach_idx[k] = inertial_kin_indices[start_count]
        elif cnt > 1:
            inertial_up_kin_reach_idx[k] = -(start_count + 2)
        inertial_kin_offsets.append(len(inertial_kin_indices))
    inertial_up_kin_offsets: ArrayInt32 = np.array(inertial_kin_offsets, dtype=np.int32)
    inertial_up_kin_indices: ArrayInt32 = np.array(inertial_kin_indices, dtype=np.int32)

    inertial_up_count: ArrayInt32 = np.zeros(n_inertial, dtype=np.int32)
    for k in range(n_inertial):
        n_in: int = (
            1
            if inertial_up_reach_idx[k] >= 0
            else (
                inertial_up_offsets_arr[k + 1] - (-inertial_up_reach_idx[k] - 2)
                if inertial_up_reach_idx[k] < -1
                else 0
            )
        )
        n_kin: int = (
            1
            if inertial_up_kin_reach_idx[k] >= 0
            else (
                inertial_up_kin_offsets[k + 1] - (-inertial_up_kin_reach_idx[k] - 2)
                if inertial_up_kin_reach_idx[k] < -1
                else 0
            )
        )
        inertial_up_count[k] = n_in + n_kin

    inertial_boundary_outflow_k_list: list[int] = [
        k for k in range(n_inertial) if ds_boundary_type[k] != 0
    ]
    inertial_boundary_outflow_k: ArrayInt32 = np.array(
        inertial_boundary_outflow_k_list, dtype=np.int32
    )

    topology: InertialTopology = InertialTopology(
        inertial_up_offsets=inertial_up_offsets_arr,
        inertial_up_indices=inertial_up_indices_arr,
        inertial_up_reach_idx=inertial_up_reach_idx,
        ds_boundary_type=ds_boundary_type,
        ds_inertial_k=ds_inertial_k,
        ds_stage_idx=ds_stage_idx,
        ds_bed_elevation=ds_bed_elevation,
        inertial_topo_order=inertial_topo_order,
    )

    kinematic_topology: KinematicTopology = KinematicTopology(
        n_kinematic=n_kinematic,
        n_kin_headwater=n_kin_headwater,
        kin_ds_reach=kin_ds_reach,
        kin_inflow_m3_s=kin_inflow_m3_s,
        kin_ds_waterbody_id=kin_ds_wb_id,
        wb_to_kin_target_reach=wb_to_kin_target_reach,
        wb_to_kin_wb_id=wb_to_kin_wb_id,
        kin_wb_inflow_m3=kin_wb_inflow_m3,
        inertial_up_kin_offsets=inertial_up_kin_offsets,
        inertial_up_kin_indices=inertial_up_kin_indices,
        inertial_up_kin_reach_idx=inertial_up_kin_reach_idx,
        river_storage_alpha=river_storage_alpha_kin,
        river_storage_beta=river_storage_beta_kin,
        river_length=river_length_perm[:n_kinematic],
        retention_node_id=retention_node_id_perm[:n_kinematic],
    )

    return RoutingTopology(
        topology=topology,
        kinematic_topology=kinematic_topology,
        interface_bed_elev_max=interface_bed_elev_max,
        inv_dx_interface=inv_dx_interface,
        inertial_up_count=inertial_up_count,
        inertial_retention_k=inertial_retention_k,
        inertial_retention_id=inertial_retention_id,
        inertial_boundary_outflow_k=inertial_boundary_outflow_k,
        ds_boundary_type=ds_boundary_type,
        ds_inertial_k=ds_inertial_k,
        ds_waterbody_id=ds_waterbody_id,
        ds_stage_idx=ds_stage_idx,
        ds_bed_elevation=ds_bed_elevation,
        inertial_wb_ds_k=inertial_wb_ds_k,
        inertial_wb_ds_id=inertial_wb_ds_id,
        lake_outflow_target_k=lake_outflow_target_k,
        lake_outflow_wb_id=lake_outflow_wb_id,
        wb_to_wb_src_wb=wb_to_wb_src_wb,
        wb_to_wb_tgt_wb=wb_to_wb_tgt_wb,
        wb_terminal_wb_ids=wb_terminal_wb_ids,
        inertial_up_offsets=inertial_up_offsets_arr,
        inertial_up_indices=inertial_up_indices_arr,
        inertial_up_reach_idx=inertial_up_reach_idx,
    )
