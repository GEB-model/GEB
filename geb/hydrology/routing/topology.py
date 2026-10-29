"""Network topology graphs, CSR matrices, and boundary classification for river routing."""

from typing import NamedTuple

import numpy as np
from numba import njit

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    TwoDArrayInt32,
)

__all__: list[str] = [
    "BOUNDARY_INTERNAL",
    "BOUNDARY_LAKE",
    "BOUNDARY_OCEAN",
    "BOUNDARY_PIT",
    "InertialTopology",
    "KinematicTopology",
    "RetentionCoupling",
    "RetentionLinks",
    "RoutingTopology",
    "WaterbodyCoupling",
    "WaterbodyLinks",
    "build_routing_topology",
    "compute_retention_routing",
]

# Downstream boundary classification codes for hydrodynamic inertial reaches.
BOUNDARY_INTERNAL: int = 0
"""Internal hydrodynamic reach discharging into another inertial reach."""

BOUNDARY_LAKE: int = 1
"""Reach discharging into a downstream lake or reservoir waterbody."""

BOUNDARY_PIT: int = 2
"""Inland terminal sink or basin boundary edge draining via free slope drawdown."""

BOUNDARY_OCEAN: int = 3
"""Coastal terminal sink discharging into the ocean at sea level (0.0 m datum)."""


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
        ds_boundary_type: Boundary condition classification code per reach (0=internal, 1=lake, 2=pit, 3=ocean).
        ds_inertial_k: Downstream reach index within inertial domain (-1 if boundary).
        ds_stage_idx: Unified index into stage buffer for downstream water level lookup.
        ds_bed_elevation: Downstream boundary bed elevation (meters).
        confluence_reach_k: 1D array of reach indices with multiple upstream or kinematic tributaries.
    """

    inertial_up_offsets: ArrayInt32
    inertial_up_indices: ArrayInt32
    inertial_up_reach_idx: ArrayInt32
    ds_boundary_type: ArrayInt32
    ds_inertial_k: ArrayInt32
    ds_stage_idx: ArrayInt32
    ds_bed_elevation: ArrayFloat32
    confluence_reach_k: ArrayInt32


class RetentionLinks(NamedTuple):
    """Linkage mapping between inertial reaches and retention basins.

    Attributes:
        inertial_retention_k: Inertial reach indices with coupled retention basins.
        inertial_retention_id: Retention basin identifiers coupled to inertial reaches.
    """

    inertial_retention_k: ArrayInt32
    inertial_retention_id: ArrayInt32


class WaterbodyLinks(NamedTuple):
    """Static linkage mapping between river reaches and waterbodies.

    Attributes:
        inertial_wb_ds_k: Inertial reaches discharging into downstream waterbodies.
        inertial_wb_ds_id: Waterbody IDs receiving discharges from inertial reaches.
        lake_outflow_target_k: Inertial reach indices receiving lake releases.
        lake_outflow_wb_id: Waterbody IDs supplying releases to inertial reaches.
        wb_to_wb_src_wb: Source waterbody IDs for inter-waterbody transfer links.
        wb_to_wb_tgt_wb: Destination waterbody IDs for inter-waterbody transfer links.
        wb_terminal_wb_ids: Waterbody IDs discharging out of domain (terminal pits).
    """

    inertial_wb_ds_k: ArrayInt32
    inertial_wb_ds_id: ArrayInt32
    lake_outflow_target_k: ArrayInt32
    lake_outflow_wb_id: ArrayInt32
    wb_to_wb_src_wb: ArrayInt32
    wb_to_wb_tgt_wb: ArrayInt32
    wb_terminal_wb_ids: ArrayInt32


class RoutingTopology(NamedTuple):
    """Container grouping all topology graphs, CSR matrices, and cross-domain links.

    Attributes:
        topology: Connectivity and boundary structure for the inertial domain.
        kinematic_topology: Topology and reach parameter tables for the kinematic domain.
        interface_bed_elev_max: Max interface bed elevation between reach and its downstream node (meters).
        inv_dx_interface: Reciprocal of spatial interface length (1/meters).
        inertial_up_count: Tributary junction in-degree count for CFL scaling.
        inertial_boundary_outflow_k: Reach indices exiting the hydrodynamic domain for mass balance tracking.
        retention_links: Linkage mapping between inertial reaches and retention basins.
        waterbody_links: Linkage mapping between river reaches and waterbodies.
    """

    topology: InertialTopology
    kinematic_topology: KinematicTopology
    interface_bed_elev_max: ArrayFloat32
    inv_dx_interface: ArrayFloat32
    inertial_up_count: ArrayInt32
    inertial_boundary_outflow_k: ArrayInt32
    retention_links: RetentionLinks
    waterbody_links: WaterbodyLinks


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
) -> RoutingTopology:
    """Constructs CSR connectivity graphs, boundary classifications, and cross-domain links.

    Notes:
        River reaches are topologically ordered such that kinematic wave reaches occupy indices
        `0` to `n_kinematic - 1`, and hydrodynamic local inertial reaches occupy indices
        `n_kinematic` to `n_kinematic + n_inertial - 1`. Reaches are mapped to boundary types:
        internal reaches (0), lakes (1), inland pits (2), and ocean pits (3).

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

    Returns:
        Structured RoutingTopology containing all precomputed topology objects.

    Raises:
        ValueError: If reaches have invalid configuration, non-positive lengths, or illegal connections.
    """
    if n_kinematic < 0:
        raise ValueError("n_kinematic must be non-negative.")
    if n_inertial < 0:
        raise ValueError("n_inertial must be non-negative.")
    if n_kin_headwater < 0 or n_kin_headwater > n_kinematic:
        raise ValueError(
            f"n_kin_headwater ({n_kin_headwater}) must be between 0 and n_kinematic ({n_kinematic})."
        )
    if np.any(river_length_perm <= 0.0):
        raise ValueError("River channel lengths must be strictly positive.")

    # Step 1: Classify downstream boundaries for all hydrodynamic inertial reaches.
    # We inspect the downstream connection for each inertial reach to determine whether
    # it discharges into another inertial reach (internal), a lake/reservoir, an inland
    # drainage pit, or an open ocean boundary.
    ds_boundary_type: ArrayInt32 = np.empty(n_inertial, dtype=np.int32)
    ds_inertial_k: ArrayInt32 = np.full(n_inertial, -1, dtype=np.int32)
    ds_waterbody_id: ArrayInt32 = np.full(n_inertial, -1, dtype=np.int32)
    ds_bed_elevation: ArrayFloat32 = np.empty(n_inertial, dtype=np.float32)

    for k in range(n_inertial):
        i: int = n_kinematic + k
        ds: int = int(ds_node[i])
        bed_node: float = float(bed_elevation_perm[i])
        reach_length_m: float = float(river_length_perm[i])

        if is_pit_perm[i]:
            if is_ocean_pit_perm[i]:
                # Ocean outlet: water drops to sea level (0.0 m datum).
                ds_boundary_type[k] = BOUNDARY_OCEAN
                ds_bed_elevation[k] = np.float32(bed_node)
            else:
                # Inland pit: extrapolate downstream bed elevation along the reach slope.
                ds_boundary_type[k] = BOUNDARY_PIT
                eff_slope: float = float(pit_slope_perm[i])
                ds_bed_elevation[k] = np.float32(bed_node - eff_slope * reach_length_m)
        elif ds != -1:
            bed_ds: float = float(bed_elevation_perm[ds])
            ds_bed_elevation[k] = np.float32(bed_ds)
            wb_ds: int = int(waterbody_ids_perm[ds])
            if wb_ds != -1:
                # Reach discharges directly into a downstream lake or reservoir waterbody.
                ds_boundary_type[k] = BOUNDARY_LAKE
                ds_waterbody_id[k] = wb_ds
            else:
                # Inertial reach flows into another river reach.
                # Discharging into a kinematic reach is strictly forbidden because kinematic wave
                # routing cannot propagate backwater waves across the domain interface.
                if use_kinematic_perm[ds]:
                    raise ValueError(
                        f"Inertial reach at sorted index {i} cannot discharge into a kinematic reach (downstream index {ds}). "
                        "Kinematic wave routing does not support downstream backwater boundary conditions."
                    )
                ds_boundary_type[k] = BOUNDARY_INTERNAL
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

    # Step 2: Compute maximum interface bed elevation (hydraulic sill).
    # In the local inertial approximation, the flow depth at the cell interface is governed by
    # max(water_stage_node, water_stage_ds) - max(bed_node, bed_ds). If the water stage does
    # not exceed this highest bed elevation, the interface is dry and flow is blocked.
    interface_bed_elev_max: ArrayFloat32 = np.maximum(
        bed_elevation_inertial, ds_bed_elevation
    )

    # Step 3: Compute reciprocal of the interface distance (inv_dx_interface).
    # For internal reaches, the cell interface lies midway between the centers of two reaches:
    # dx_internal = 0.5 * (length_node + length_ds).
    # For boundary reaches (lake, pit, ocean), the interface is at the cell boundary: dx = length_node.
    # Precomputing 1 / dx replaces repeated divisions with fast multiplications in the momentum kernel.
    is_internal: ArrayBool = ds_boundary_type == BOUNDARY_INTERNAL
    ds_k_internal: ArrayInt32 = ds_inertial_k[is_internal]
    inv_dx_interface: ArrayFloat32 = np.empty(n_inertial, dtype=np.float32)
    inv_dx_interface[~is_internal] = (
        np.float32(1.0) / river_length_inertial[~is_internal]
    )
    dx_internal: ArrayFloat32 = np.float32(0.5) * (
        river_length_inertial[is_internal] + river_length_inertial[ds_k_internal]
    )
    inv_dx_interface[is_internal] = np.float32(1.0) / dx_internal

    # Step 4: Map downstream stage lookup index (ds_stage_idx).
    # The solver workspace `stage_buf` stores river reach stages in indices [0 : n_inertial)
    # and lake water levels in indices [n_inertial : n_inertial + n_waterbodies).
    # Mapping lake boundaries to offset `n_inertial + ds_waterbody_id` allows the momentum solver
    # to retrieve downstream water stage for both river reaches and lakes via a single branchless lookup.
    ds_stage_idx: ArrayInt32 = np.full(n_inertial, -1, dtype=np.int32)
    ds_stage_idx[is_internal] = ds_inertial_k[is_internal]
    is_lake: ArrayBool = ds_boundary_type == BOUNDARY_LAKE
    ds_stage_idx[is_lake] = n_inertial + ds_waterbody_id[is_lake]

    # Step 5: Build CSR graph for upstream inertial tributaries.
    # A reach can have 0 (headwater), 1 (linear channel), or >= 2 (confluence) upstream tributaries.
    # To optimize memory access and CPU cache utilization during momentum substepping:
    # - If cnt == 1: store the direct upstream reach index (>= 0) in `inertial_up_reach_idx`.
    #   This bypasses CSR indirection entirely for ~90% of river reaches.
    # - If cnt > 1: encode the CSR starting offset as a negative sentinel `-(start_count + 2) < -1`.
    #   This differentiates multi-tributary confluences from headwaters (-1) and single reaches (>= 0).
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
            # Exclude waterbodies (handled by lake outflow links) and kinematic reaches (handled separately)
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

    # Step 6: Identify inertial reaches coupled to flood retention basins.
    # Only a small fraction of cells contain retention basins; filtering active pairs avoids
    # scanning the entire reach array during retention diversion and release calculations.
    inertial_retention_k: ArrayInt32 = np.flatnonzero(
        retention_node_id_perm[n_kinematic : n_kinematic + n_inertial] != -1
    ).astype(np.int32)
    inertial_retention_id: ArrayInt32 = retention_node_id_perm[
        n_kinematic + inertial_retention_k
    ]
    retention_links: RetentionLinks = RetentionLinks(
        inertial_retention_k=inertial_retention_k,
        inertial_retention_id=inertial_retention_id,
    )

    # Step 7: Classify waterbody linkages across model domains.
    # River reaches entering lakes are gathered for inflow accumulation.
    # Waterbody outlets are routed according to what lies downstream:
    # (a) Terminal pit: waterbody discharges outside the model domain.
    # (b) Another waterbody: water transfers to a cascading downstream lake/reservoir.
    # (c) Kinematic reach: water discharges into a kinematic wave river cell.
    # (d) Inertial reach: water discharges into a hydrodynamic local inertial river cell.
    inertial_wb_ds_k: ArrayInt32 = np.flatnonzero(is_lake).astype(np.int32)
    inertial_wb_ds_id: ArrayInt32 = ds_waterbody_id[inertial_wb_ds_k]

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
            # If target_wb == wb_id, it is internal to the same waterbody level pool (no transfer link needed)
            if target_wb != wb_id:
                wb_to_wb_src.append(wb_id)
                wb_to_wb_tgt.append(target_wb)
        elif 0 <= ds < n_kinematic:
            wb_to_kin_targets.append(ds)
            wb_to_kin_wbs.append(wb_id)
        elif n_kinematic <= ds < n_kinematic + n_inertial:
            lake_outflow_targets.append(ds - n_kinematic)
            lake_outflow_wbs.append(wb_id)
        else:
            raise ValueError(
                f"Waterbody outlet at node {wb_node} (waterbody ID {wb_id}) has invalid downstream node index {ds}."
            )

    lake_outflow_target_k: ArrayInt32 = np.array(lake_outflow_targets, dtype=np.int32)
    lake_outflow_wb_id: ArrayInt32 = np.array(lake_outflow_wbs, dtype=np.int32)
    wb_to_wb_src_wb: ArrayInt32 = np.array(wb_to_wb_src, dtype=np.int32)
    wb_to_wb_tgt_wb: ArrayInt32 = np.array(wb_to_wb_tgt, dtype=np.int32)
    wb_to_kin_target_reach: ArrayInt32 = np.array(wb_to_kin_targets, dtype=np.int32)
    wb_to_kin_wb_id: ArrayInt32 = np.array(wb_to_kin_wbs, dtype=np.int32)
    kin_wb_inflow_m3: ArrayFloat32 = np.zeros(n_kinematic, dtype=np.float32)
    wb_terminal_wb_ids: ArrayInt32 = np.array(wb_term_ids, dtype=np.int32)

    # Step 8: Build downstream connectivity for kinematic wave reaches.
    # Kinematic reaches route downstream sequentially. If a kinematic reach drains into
    # another kinematic reach, outflow is deposited directly into `kin_inflow_m3_s`.
    # If it drains into an inertial reach, `kin_ds_reach` remains -1; the receiving inertial
    # reach pulls this discharge as boundary inflow via `inertial_up_kin_*` during substepping.
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

    # Step 9: Build CSR graph for upstream kinematic reaches feeding inertial reaches.
    # At the kinematic-inertial domain boundary, upstream kinematic reaches supply lateral
    # boundary inflow to the receiving hydrodynamic reaches.
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

    # Step 10: Compute total upstream in-degree count per inertial reach.
    # The total number of incoming tributaries (both inertial and kinematic) is determined directly
    # from CSR row lengths (offsets[k + 1] - offsets[k]). This in-degree count is used for
    # confluence stability and inflow tracking.
    inertial_up_count: ArrayInt32 = (
        np.diff(inertial_up_offsets_arr) + np.diff(inertial_up_kin_offsets)
    ).astype(np.int32)

    waterbody_links: WaterbodyLinks = WaterbodyLinks(
        inertial_wb_ds_k=inertial_wb_ds_k,
        inertial_wb_ds_id=inertial_wb_ds_id,
        lake_outflow_target_k=lake_outflow_target_k,
        lake_outflow_wb_id=lake_outflow_wb_id,
        wb_to_wb_src_wb=wb_to_wb_src_wb,
        wb_to_wb_tgt_wb=wb_to_wb_tgt_wb,
        wb_terminal_wb_ids=wb_terminal_wb_ids,
    )

    # Step 11: Identify reaches discharging across domain boundaries.
    # Reaches with ds_boundary_type != 0 discharge out of the hydrodynamic network (lake, pit, ocean).
    # Tracking these indices allows exact mass balance summation without scanning internal reaches.
    inertial_boundary_outflow_k: ArrayInt32 = np.flatnonzero(
        ds_boundary_type != BOUNDARY_INTERNAL
    ).astype(np.int32)

    has_mult_inertial: np.ndarray = inertial_up_reach_idx < -1
    has_kin_tribs: np.ndarray = (
        inertial_up_kin_offsets[1 : n_inertial + 1]
        > inertial_up_kin_offsets[:n_inertial]
    )
    confluence_reach_k: ArrayInt32 = np.flatnonzero(
        has_mult_inertial | has_kin_tribs
    ).astype(np.int32)

    topology: InertialTopology = InertialTopology(
        inertial_up_offsets=inertial_up_offsets_arr,
        inertial_up_indices=inertial_up_indices_arr,
        inertial_up_reach_idx=inertial_up_reach_idx,
        ds_boundary_type=ds_boundary_type,
        ds_inertial_k=ds_inertial_k,
        ds_stage_idx=ds_stage_idx,
        ds_bed_elevation=ds_bed_elevation,
        confluence_reach_k=confluence_reach_k,
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
        inertial_boundary_outflow_k=inertial_boundary_outflow_k,
        retention_links=retention_links,
        waterbody_links=waterbody_links,
    )


class WaterbodyCoupling(NamedTuple):
    """Encapsulates coupled lake and reservoir state arrays, rating parameters, and linkage maps.

    Attributes:
        storage_m3: Current water storage volume per waterbody (m³).
        prescribed_outflow_m3: Outflow volume prescribed by human operations or dams (m³).
        inflow_m3: Cumulative inflow volume accumulator per waterbody (m³).
        lake_area: Surface area per waterbody (m²).
        lake_factor: Rating curve weir discharge multiplier per waterbody.
        outflow_height: Weir sill elevation height above bed per waterbody (meters).
        outflow_bed_elev: Bed elevation at waterbody outlet (meters).
        inertial_wb_ds_k: Inertial reach indices that terminate into downstream waterbodies.
        inertial_wb_ds_id: Waterbody identifier receiving inflow from each terminal reach.
        lake_outflow_target_k: Inertial reach index receiving waterbody outflow releases.
        lake_outflow_wb_id: Source waterbody identifier discharging into each target reach.
        wb_to_wb_src_wb: Source waterbody IDs discharging into downstream waterbodies.
        wb_to_wb_tgt_wb: Target waterbody IDs receiving cascading waterbody releases.
        wb_terminal_wb_ids: Waterbody IDs discharging directly across domain boundaries.
    """

    storage_m3: ArrayFloat64
    prescribed_outflow_m3: ArrayFloat32
    inflow_m3: ArrayFloat32
    lake_area: ArrayFloat32
    lake_factor: ArrayFloat32
    outflow_height: ArrayFloat32
    outflow_bed_elev: ArrayFloat32
    inertial_wb_ds_k: ArrayInt32
    inertial_wb_ds_id: ArrayInt32
    lake_outflow_target_k: ArrayInt32
    lake_outflow_wb_id: ArrayInt32
    wb_to_wb_src_wb: ArrayInt32
    wb_to_wb_tgt_wb: ArrayInt32
    wb_terminal_wb_ids: ArrayInt32


class RetentionCoupling(NamedTuple):
    """Encapsulates flood retention basin state arrays, capacities, and control parameters.

    Attributes:
        storage_m3: Current retention basin storage volume array (m³).
        inflow_m3: Cumulative diverted inflow volume accumulator per retention basin (m³).
        outflow_m3: Cumulative released outflow volume accumulator per retention basin (m³).
        max_storage_m3: Maximum storage capacity per retention basin (m³).
        inertial_retention_k: Inertial reach indices connected to retention basins.
        inertial_retention_id: Retention basin identifier linked to each connected reach.
        controlled_retention: Flag defining controlled vs uncontrolled intake dynamics.
        activation_threshold_m3_s: Discharge threshold triggering retention basin diversion (m³/s).
        release_threshold_factor: Capacity fraction triggering controlled basin release.
    """

    storage_m3: ArrayFloat32
    inflow_m3: ArrayFloat32
    outflow_m3: ArrayFloat32
    max_storage_m3: ArrayFloat32
    inertial_retention_k: ArrayInt32
    inertial_retention_id: ArrayInt32
    controlled_retention: ArrayBool
    activation_threshold_m3_s: ArrayFloat32
    release_threshold_factor: np.float32


@njit(cache=True)
def compute_retention_routing(
    dt: np.float32,
    river_volume_m3: np.float32,
    discharge_before_diversion_m3_s: np.float32,
    is_rising_limb: bool,
    retention_storage_m3: np.float32,
    retention_max_storage_m3: np.float32,
    controlled_retention: bool,
    activation_threshold_m3_s: np.float32,
    release_threshold_factor: np.float32,
    inflow_limit_m3: np.float32,
    max_outflow_limit_m3: np.float32,
) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    """Calculate the water diversion and release for a single retention basin node.

    This function computes how much water is diverted from the river into a
    retention basin (during high flows) or released from the basin back into
    the river (during low flows) based on predefined thresholds.

    Notes:
        Diversions only occur when the river's discharge rate exceeds the activation threshold.
        For controlled retention basins, diversion can only occur on a rising limb of the hydrograph.
        Releases only occur on the falling limb when the storage is non-empty and the initial
        discharge rate is below the release threshold. High flows and low flows are handled as mutually
        exclusive states.

    Args:
        dt: The length of the routing time step (seconds).
        river_volume_m3: The current available volume of water in the river cell (m³).
        discharge_before_diversion_m3_s: The initial discharge rate in the river cell before diversion (m³/s).
        is_rising_limb: Flag indicating if the discharge is currently rising compared to the previous time step.
        retention_storage_m3: The current amount of water stored in the retention basin (m³).
        retention_max_storage_m3: The maximum storage capacity of the retention basin (m³).
        controlled_retention: Flag indicating whether the retention basin is controlled.
        activation_threshold_m3_s: The discharge threshold above which water is diverted into the basin (m³/s).
        release_threshold_factor: The multiplier applied to the activation threshold to determine the release threshold.
        inflow_limit_m3: The physical limit of water volume that can enter the basin during this step (m³).
        max_outflow_limit_m3: The maximum volume of water that the basin can release during this step (m³).

    Returns:
        A tuple containing:
            diverted_volume_m3: The volume of water diverted into the basin (m³).
            outflow_volume_m3: The volume of water released from the basin (m³).
            updated_retention_storage_m3: The updated retention storage (m³).
            final_river_volume_m3: The updated available water volume in the river cell (m³).

    Raises:
        ValueError: If `dt` is <= 0 or if `retention_max_storage_m3` is negative.
    """
    if dt <= np.float32(0.0):
        raise ValueError("Time step dt must be positive.")
    if retention_max_storage_m3 < np.float32(0.0):
        raise ValueError("Maximum retention storage must be non-negative.")

    # Available capacity in the retention basin. Cannot be negative.
    available_storage_m3: np.float32 = max(
        np.float32(0.0),
        retention_max_storage_m3 - retention_storage_m3,
    )

    diverted_volume_m3: np.float32 = np.float32(0.0)
    outflow_volume_m3: np.float32 = np.float32(0.0)

    # Only release water on the falling limb
    effective_release_factor: np.float32 = (
        release_threshold_factor if not is_rising_limb else np.float32(0.0)
    )
    release_threshold_m3_s: np.float32 = (
        activation_threshold_m3_s * effective_release_factor
    )

    # High flow regime: divert peak discharge above activation threshold into the basin
    if discharge_before_diversion_m3_s > activation_threshold_m3_s:
        # Controlled basins only divert on the rising limb of the hydrograph to shave the peak
        if not controlled_retention or is_rising_limb:
            discharge_above_activation_threshold_m3_s: np.float32 = (
                discharge_before_diversion_m3_s - activation_threshold_m3_s
            )
            # Divert at most available water in river cell, available basin capacity,
            # physical intake limit, or the discharge volume above activation threshold.
            diverted_volume_m3 = max(
                np.float32(0.0),
                min(
                    river_volume_m3,
                    available_storage_m3,
                    inflow_limit_m3,
                    discharge_above_activation_threshold_m3_s * dt,
                ),
            )

    # Low/recession flow regime: release stored water back to river without exceeding release threshold
    elif (
        not is_rising_limb and discharge_before_diversion_m3_s <= release_threshold_m3_s
    ):
        if retention_storage_m3 > np.float32(0.0):
            # Limit release so resulting river discharge does not exceed the release threshold
            allowed_extra_outflow_m3: np.float32 = (
                release_threshold_m3_s - discharge_before_diversion_m3_s
            ) * dt
            outflow_volume_m3 = min(
                allowed_extra_outflow_m3,
                max_outflow_limit_m3,
                retention_storage_m3,
            )
            outflow_volume_m3 = max(np.float32(0.0), outflow_volume_m3)

    # Update retention storage and remaining river volume conserving total mass.
    updated_retention_storage_m3: np.float32 = (
        retention_storage_m3 + diverted_volume_m3 - outflow_volume_m3
    )
    final_river_volume_m3: np.float32 = (
        river_volume_m3 - diverted_volume_m3 + outflow_volume_m3
    )

    return (
        diverted_volume_m3,
        outflow_volume_m3,
        updated_retention_storage_m3,
        final_river_volume_m3,
    )
