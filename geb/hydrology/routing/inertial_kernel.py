"""Substepping kernel for 1D local inertial dynamic river routing.

Solves the de Saint-Venant momentum and continuity equations using an adaptive
substepping scheme with analytical cross-sectional channel and floodplain geometry.
"""

from typing import NamedTuple

import numpy as np
from numba import njit, prange

from geb.geb_types import (
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    TwoDArrayFloat32,
)

from .geometry import (
    GEOM_CFL_CONSTANT,
    GEOM_CFL_MANNING_COEFFICIENT,
    GEOM_IN_BANKFULL_DEPTH,
    GEOM_IN_BANKFULL_VOLUME,
    GEOM_IN_BED_ELEVATION,
    GEOM_IN_INTERFACE_BED_ELEVATION_MAX,
    GEOM_IN_INVERSE_INTERFACE_LENGTH,
    GEOM_IN_INVERSE_LENGTH,
    GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE,
    GEOM_IN_STAGE_VOLUME_COEFFICIENT,
    GEOM_IN_WEIR_HEIGHT,
    GEOM_IN_WEIR_WIDTH,
    GEOM_OV_FLOODPLAIN_WIDTH,
    GEOM_OV_RIVER_WIDTH,
    InertialGeometry,
    compute_cfl_area_and_top_width,
    compute_overbank_depth,
)
from .solver import rectangular_weir_discharge, solve_inertial_momentum
from .topology import (
    InertialTopology,
    KinematicTopology,
    RetentionCoupling,
    WaterbodyCoupling,
    compute_retention_routing,
)

__all__ = [
    "ALLOW_REVERSE_FLOW",
    "INERTIAL_PARALLEL_THRESHOLD",
    "RoutingStepDiagnostics",
    "SubstepBuffers",
    "_compute_reach_stage_from_vol",
    "_compute_stage_from_storage_inertial",
    "_evaluate_reach_cfl_dt",
    "_gather_inputs",
    "_run_inertial_substeps",
    "_run_inertial_substeps_parallel",
    "_run_inertial_substeps_serial",
    "_scatter_outputs",
    "compute_inertial_substeps_cfl",
]

# Global option to toggle reverse flow across river reaches
ALLOW_REVERSE_FLOW: bool = False

# Threshold of reach count above which parallel multi-threading outperforms serial execution
INERTIAL_PARALLEL_THRESHOLD: int = 5000


class RoutingStepDiagnostics(NamedTuple):
    """Encapsulates numerical stability and CFL limiting diagnostics from a routing step.

    Attributes:
        num_inertial_substeps: Total number of sub-timesteps executed.
        limiting_reach_idx: Reach index requiring the smallest stable timestep (-1 if none).
        min_stable_timestep_s: Minimum stable timestep duration across all reaches (seconds).
        h_vol: Hydraulic depth from stored channel volume in limiting reach (meters).
        h_flow: Hydraulic depth estimated from oncoming flow in limiting reach (meters).
        h_gov: Governing hydraulic depth in limiting reach (meters).
        celerity: Shallow-water wave celerity in limiting reach (m/s).
        limiting_flow: Characteristic peak flow rate through limiting reach (m³/s).
        limiting_storage: Stored volume in limiting reach (m³).
    """

    num_inertial_substeps: int
    limiting_reach_idx: int
    min_stable_timestep_s: np.float32
    h_vol: np.float32
    h_flow: np.float32
    h_gov: np.float32
    celerity: np.float32
    limiting_flow: np.float32
    limiting_storage: np.float32


class SubstepBuffers(NamedTuple):
    """Encapsulates pre-allocated scratch workspace arrays for adaptive inertial substepping.

    Attributes:
        stage_buf: Unified water surface elevation buffer for reaches and lakes (meters).
        min_dt_buf: Scratch buffer for adaptive CFL timestep duration per reach (seconds).
        rev_demand_buf: Workspace buffer accumulating reverse flow demands per reach (m³/s).
        wb_release_volume_substep: Buffer for substep waterbody release volume per reach (m³).
        total_flow_rate_buf: Workspace buffer storing peak flow rate magnitude per reach (m³/s).
        discharge_vol_sum_inertial: Accumulator buffer integrating reach volumetric outflow (m³).
        retention_inflow_limit_sub_buf: Substep maximum intake diversion capacity buffer (m³).
        retention_max_outflow_limit_sub_buf: Substep maximum release capacity buffer (m³).
        wb_outflow_avail_buf: Buffer holding available outflow volume per waterbody (m³).
        wb_extra_lateral_accum_m3: Accumulator tracking total waterbody release volume transferred (m³).
        terminal_wb_outflow_accum_m3: Single-element accumulator for domain boundary lake outflows (m³).
        substep_discharge_inertial: Dynamic substep discharge state vector (m³/s).
        kinematic_inflow_rate_inertial: Workspace storing upstream kinematic inflow rates (m³/s).
    """

    stage_buf: ArrayFloat32
    min_dt_buf: ArrayFloat32
    rev_demand_buf: ArrayFloat32
    wb_release_volume_substep: ArrayFloat32
    total_flow_rate_buf: ArrayFloat32
    discharge_vol_sum_inertial: ArrayFloat64
    retention_inflow_limit_sub_buf: ArrayFloat64
    retention_max_outflow_limit_sub_buf: ArrayFloat64
    wb_outflow_avail_buf: ArrayFloat64
    wb_extra_lateral_accum_m3: ArrayFloat64
    terminal_wb_outflow_accum_m3: ArrayFloat32
    substep_discharge_inertial: ArrayFloat32 = np.empty(0, dtype=np.float32)
    kinematic_inflow_rate_inertial: ArrayFloat32 = np.empty(0, dtype=np.float32)


@njit(fastmath=True, parallel=True)
def _gather_inputs(
    sorted_idxs: ArrayInt32,
    q_in: ArrayFloat32,
    storage_in: ArrayFloat64,
    sideflow_in: ArrayFloat32,
    evap_in: ArrayFloat32,
    q_out: ArrayFloat32,
    storage_out: ArrayFloat64,
    sideflow_out: ArrayFloat32,
    evap_out: ArrayFloat32,
) -> None:
    """Gathers and permutes multiple routing input arrays into topological order.

    Notes:
        Inverse of `_scatter_outputs`.

    Args:
        sorted_idxs: Precomputed 1D permutation index array.
        q_in: Input discharge array (m³/s).
        storage_in: Input river storage volume array (m³).
        sideflow_in: Input lateral sideflow volume array (m³).
        evap_in: Input potential evaporation volume array (m³).
        q_out: Target pre-allocated discharge array (m³/s).
        storage_out: Target pre-allocated river storage array (m³).
        sideflow_out: Target pre-allocated sideflow array (m³).
        evap_out: Target pre-allocated evaporation array (m³).
    """
    n: int = len(sorted_idxs)
    for i in prange(n):
        idx: int = sorted_idxs[i]
        q_out[i] = q_in[idx]
        storage_out[i] = storage_in[idx]
        sideflow_out[i] = sideflow_in[idx]
        evap_out[i] = evap_in[idx]


@njit(fastmath=True, parallel=True)
def _scatter_outputs(
    sorted_idxs: ArrayInt32,
    q_perm: ArrayFloat32,
    evap_perm: ArrayFloat32,
    over_abs_perm: ArrayFloat32,
    storage_perm: ArrayFloat64,
    q_out: ArrayFloat32,
    evap_out: ArrayFloat32,
    over_abs_out: ArrayFloat32,
    storage_out: ArrayFloat64,
) -> None:
    """Scatters permuted routing output arrays back to original geographic order.

    Notes:
        Inverse of `_gather_inputs`.

    Args:
        sorted_idxs: Precomputed 1D permutation index array.
        q_perm: Permuted channel discharge array (m³/s).
        evap_perm: Permuted actual evaporation volume array (m³).
        over_abs_perm: Permuted over-abstraction deficit array (m³).
        storage_perm: Permuted river storage volume array (m³).
        q_out: Target original-order discharge array (m³/s).
        evap_out: Target original-order actual evaporation array (m³).
        over_abs_out: Target original-order over-abstraction array (m³).
        storage_out: Target original-order river storage array (m³).
    """
    n: int = len(sorted_idxs)
    for i in prange(n):
        idx: int = sorted_idxs[i]
        q_out[idx] = q_perm[i]
        evap_out[idx] = evap_perm[i]
        over_abs_out[idx] = over_abs_perm[i]
        storage_out[idx] = storage_perm[i]


@njit(inline="always")
def _compute_reach_stage_from_vol(
    reach_idx: int,
    volume_m3: np.float32,
    geom_inbank: TwoDArrayFloat32,
    geom_overbank: TwoDArrayFloat32,
) -> np.float32:
    """Computes reach water surface elevation from storage volume.

    Assumes a two-tier system where we have a channel that is filled up to
    bankfull elevation before water starts flowing over the floodplain.

    Args:
        reach_idx: Reach index within local inertial domain.
        volume_m3: Reach storage volume (m³).
        geom_inbank: Compact 2D in-bank reach geometry array (meters, dimensionless).
        geom_overbank: Secondary 2D overbank floodplain geometry array (meters, dimensionless).

    Returns:
        Water surface elevation (meters).
    """
    bankfull_volume_m3: np.float32 = geom_inbank[reach_idx, GEOM_IN_BANKFULL_VOLUME]
    bankfull_depth_m: np.float32 = geom_inbank[reach_idx, GEOM_IN_BANKFULL_DEPTH]

    if volume_m3 <= bankfull_volume_m3 or bankfull_depth_m <= np.float32(0.0):
        if volume_m3 <= np.float32(0.0):
            channel_depth_m: np.float32 = np.float32(0.0)
        else:
            channel_depth_m = geom_inbank[
                reach_idx, GEOM_IN_STAGE_VOLUME_COEFFICIENT
            ] * (
                volume_m3
                ** geom_inbank[reach_idx, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE]
            )
        return geom_inbank[reach_idx, GEOM_IN_BED_ELEVATION] + channel_depth_m

    overbank_volume: np.float32 = volume_m3 - bankfull_volume_m3
    floodplain_depth: np.float32 = compute_overbank_depth(
        overbank_volume_m3=overbank_volume,
        inverse_reach_length=geom_inbank[reach_idx, GEOM_IN_INVERSE_LENGTH],
        floodplain_width_m=geom_overbank[reach_idx, GEOM_OV_FLOODPLAIN_WIDTH],
    )
    return (
        geom_inbank[reach_idx, GEOM_IN_BED_ELEVATION]
        + bankfull_depth_m
        + floodplain_depth
    )


@njit(fastmath=True)
def _compute_stage_from_storage_inertial(
    n_inertial: int,
    river_storage_m3_inertial: ArrayFloat64,
    geom_inbank: TwoDArrayFloat32,
    geom_overbank: TwoDArrayFloat32,
    stage_buf: ArrayFloat32,
) -> None:
    """Computes reach water surface elevation directly from storage volumes.

    Args:
        n_inertial: Number of local inertial reaches.
        river_storage_m3_inertial: Storage volume array for inertial reaches (m³).
        geom_inbank: Precomputed in-bank geometry lookup table.
        geom_overbank: Precomputed overbank geometry lookup table.
        stage_buf: Target stage buffer storing reach water surface elevations (meters).
    """
    for reach_idx in prange(n_inertial):
        reach_volume_m3: np.float32 = max(
            np.float32(river_storage_m3_inertial[reach_idx]), np.float32(0.0)
        )
        stage_buf[reach_idx] = _compute_reach_stage_from_vol(
            reach_idx, reach_volume_m3, geom_inbank, geom_overbank
        )


@njit(inline="always")
def _compute_reach_cfl_depths(
    reach_idx: int,
    total_flow_rate: np.float32,
    current_volume_m3: float,
    geom_inbank: TwoDArrayFloat32,
    geom_overbank: TwoDArrayFloat32,
    geom_cfl: TwoDArrayFloat32,
    gravity_acceleration: np.float32,
) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    """Calculates hydraulic depth components and wave celerity for CFL stability evaluation.

    Args:
        reach_idx: Local inertial reach index.
        total_flow_rate: Peak flow rate through reach (m³/s).
        current_volume_m3: Current river storage volume (m³).
        geom_inbank: In-bank geometry lookup array.
        geom_overbank: Overbank floodplain geometry array.
        geom_cfl: CFL parameter matrix.
        gravity_acceleration: Gravitational acceleration (m/s²).

    Returns:
        A tuple of (h_volume, h_flow, h_governing, wave_celerity) in meters and m/s.
    """
    h_vol: np.float32 = np.float32(0.0)
    if current_volume_m3 > 0.0:
        inv_shape_exponent_plus_one: np.float32 = geom_inbank[
            reach_idx, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE
        ]
        shape_exponent: np.float32 = (
            np.float32(0.5)
            if inv_shape_exponent_plus_one == np.float32(1.0 / 1.5)
            else (np.float32(1.0) / inv_shape_exponent_plus_one) - np.float32(1.0)
        )
        bankfull_area_m2: np.float32 = (
            geom_inbank[reach_idx, GEOM_IN_BANKFULL_VOLUME]
            * geom_inbank[reach_idx, GEOM_IN_INVERSE_LENGTH]
        )
        wetted_cross_sectional_area_m2, water_top_width_m = (
            compute_cfl_area_and_top_width(
                volume_m3=np.float32(current_volume_m3),
                bankfull_width_m=geom_overbank[reach_idx, GEOM_OV_RIVER_WIDTH],
                shape_exponent=shape_exponent,
                bankfull_depth_m=geom_inbank[reach_idx, GEOM_IN_BANKFULL_DEPTH],
                floodplain_width_m=geom_overbank[reach_idx, GEOM_OV_FLOODPLAIN_WIDTH],
                inverse_reach_length=geom_inbank[reach_idx, GEOM_IN_INVERSE_LENGTH],
                inverse_shape_exponent_plus_one=inv_shape_exponent_plus_one,
                bankfull_area_m2=bankfull_area_m2,
                bankfull_volume_m3=geom_inbank[reach_idx, GEOM_IN_BANKFULL_VOLUME],
            )
        )
        h_vol = wetted_cross_sectional_area_m2 / max(
            water_top_width_m, np.float32(1e-3)
        )

    h_flow: np.float32 = np.float32(0.0)
    if total_flow_rate > np.float32(0.0):
        raw_depth_from_flow: np.float32 = geom_cfl[
            reach_idx, GEOM_CFL_MANNING_COEFFICIENT
        ] * (total_flow_rate ** np.float32(0.6))
        bankfull_depth_m: np.float32 = geom_inbank[reach_idx, GEOM_IN_BANKFULL_DEPTH]
        if raw_depth_from_flow > bankfull_depth_m and bankfull_depth_m > np.float32(
            0.0
        ):
            extra_depth: np.float32 = raw_depth_from_flow - bankfull_depth_m
            bf_width: np.float32 = geom_overbank[reach_idx, GEOM_OV_RIVER_WIDTH]
            fp_width: np.float32 = geom_overbank[reach_idx, GEOM_OV_FLOODPLAIN_WIDTH]
            width_ratio: np.float32 = bf_width / max(bf_width + fp_width, bf_width)
            h_flow = bankfull_depth_m + (extra_depth * width_ratio)
        else:
            h_flow = raw_depth_from_flow

    h_gov: np.float32 = max(h_vol, h_flow)
    celerity: np.float32 = np.sqrt(gravity_acceleration * h_gov)
    return h_vol, h_flow, h_gov, celerity


@njit(inline="always")
def _evaluate_reach_cfl_dt(
    reach_idx: int,
    total_flow_rate: np.float32,
    current_volume_m3: float,
    geom_inbank: TwoDArrayFloat32,
    geom_overbank: TwoDArrayFloat32,
    geom_cfl: TwoDArrayFloat32,
    gravity_acceleration: np.float32,
    dt_f32: np.float32,
) -> np.float32:
    """Calculates Courant-Friedrichs-Lewy (CFL) stable timestep for a single reach.

    Numerical stability is dictated by shallow-water wave celerity (Bates et al. 2010):
        dt_stable <= CFL * dx / wave_celerity
    where:
        - wave_celerity c = sqrt(g * hydraulic_depth)

    Args:
        reach_idx: Local inertial reach index.
        total_flow_rate: Characteristic incoming or conveyance flow rate (m³/s).
        current_volume_m3: Current river reach storage volume (m³).
        geom_inbank: Compact in-bank geometry table.
        geom_overbank: Overbank floodplain geometry table.
        geom_cfl: Precomputed CFL stability parameters table.
        gravity_acceleration: Gravitational acceleration (m/s²).
        dt_f32: Routing macro-timestep duration (seconds).

    Returns:
        Maximum CFL-stable timestep duration for this reach (seconds).
    """
    if current_volume_m3 <= 0.0 and total_flow_rate <= np.float32(0.0):
        return dt_f32

    _, _, h_gov, wave_celerity = _compute_reach_cfl_depths(
        reach_idx=reach_idx,
        total_flow_rate=total_flow_rate,
        current_volume_m3=current_volume_m3,
        geom_inbank=geom_inbank,
        geom_overbank=geom_overbank,
        geom_cfl=geom_cfl,
        gravity_acceleration=gravity_acceleration,
    )

    if h_gov <= np.float32(0.0):
        return dt_f32

    dt_cfl: np.float32 = (
        geom_cfl[reach_idx, GEOM_CFL_CONSTANT]
        * np.sqrt(gravity_acceleration)
        / max(wave_celerity, np.float32(1e-3))
    )
    return min(dt_cfl, dt_f32)


@njit(cache=True)
def compute_inertial_substeps_cfl(
    dt_f32: np.float32,
    inv_dt_f32: np.float32,
    n_inertial: int,
    kin: KinematicTopology,
    previous_discharge_m3_s_inertial: ArrayFloat32,
    updated_discharge_m3_s: ArrayFloat32,
    river_storage_m3_inertial: ArrayFloat64,
    sideflow_m3_inertial: ArrayFloat32,
    kinematic_inflow_rate_inertial: ArrayFloat32,
    geom: InertialGeometry,
    topo: InertialTopology,
    wb: WaterbodyCoupling,
    buffers: SubstepBuffers,
) -> RoutingStepDiagnostics:
    """Computes upstream kinematic inflow accumulations and adaptive CFL substepping stability constraint.

    Operates in parallel across available CPU threads without lock contention.

    Args:
        dt_f32: Macro routing timestep duration (seconds).
        inv_dt_f32: Reciprocal of macro routing timestep (1/seconds).
        n_inertial: Number of reaches routed with local inertial equations.
        kin: Kinematic topology parameters and upstream index maps.
        previous_discharge_m3_s_inertial: Inertial reach discharge from previous timestep (m³/s).
        updated_discharge_m3_s: Discharge array updated from kinematic reaches (m³/s).
        river_storage_m3_inertial: Inertial reach water storage volume array (m³).
        sideflow_m3_inertial: Inertial reach lateral sideflow volume array (m³).
        kinematic_inflow_rate_inertial: Output array populated with upstream kinematic inflow rates (m³/s).
        geom: Static channel and floodplain geometry lookup tables.
        topo: Network connectivity and topological ordering structure.
        wb: Waterbody coupling parameters and storage state arrays.
        buffers: Pre-allocated workspace scratch buffers.

    Returns:
        Encapsulated routing step stability diagnostics.
    """
    inertial_up_kin_offsets: ArrayInt32 = kin.inertial_up_kin_offsets
    inertial_up_kin_indices: ArrayInt32 = kin.inertial_up_kin_indices
    inertial_up_kin_reach_idx: ArrayInt32 = kin.inertial_up_kin_reach_idx

    kinematic_inflow_rate_inertial.fill(np.float32(0.0))
    buffers.total_flow_rate_buf.fill(np.float32(0.0))

    wb_inflow_rate: ArrayFloat32 = buffers.wb_release_volume_substep
    wb_inflow_rate.fill(np.float32(0.0))
    n_lake_links: int = len(wb.lake_outflow_target_k)
    if n_lake_links > 0:
        for link_idx in range(n_lake_links):
            wb_id: int = wb.lake_outflow_wb_id[link_idx]
            target_reach: int = wb.lake_outflow_target_k[link_idx]
            target_outflow: float = float(wb.prescribed_outflow_m3[wb_id])

            # when waterbody outflow is determined outside the routing,
            # such as by reservoir operations
            if not np.isnan(target_outflow):
                wb_inflow_rate[target_reach] += np.float32(target_outflow) * inv_dt_f32
            else:
                lake_area: np.float32 = wb.lake_area[wb_id]
                depth_from_bottom: np.float32 = (
                    np.float32(wb.storage_m3[wb_id]) / lake_area
                )
                head_above_sill: np.float32 = max(
                    depth_from_bottom - wb.outflow_height[wb_id],
                    np.float32(0.0),
                )
                if head_above_sill > np.float32(1e-5):
                    lake_factor: np.float32 = wb.lake_factor[wb_id]
                    wb_inflow_rate[target_reach] += lake_factor * (
                        head_above_sill * head_above_sill
                    )

    gravity_acceleration: np.float32 = np.float32(9.80665)

    for reach_idx in range(n_inertial):
        # Accumulate inflow from upstream kinematic reaches.
        upstream_kinematic_inflow: np.float32 = np.float32(0.0)
        upstream_kinematic_reach: int = inertial_up_kin_reach_idx[reach_idx]
        if upstream_kinematic_reach >= 0:
            upstream_kinematic_inflow = updated_discharge_m3_s[upstream_kinematic_reach]
        elif upstream_kinematic_reach < -1:
            start_kinematic_idx: int = -upstream_kinematic_reach - 2
            end_kinematic_idx: int = inertial_up_kin_offsets[reach_idx + 1]
            for idx in range(start_kinematic_idx, end_kinematic_idx):
                upstream_kinematic_node: int = inertial_up_kin_indices[idx]
                upstream_kinematic_inflow += updated_discharge_m3_s[
                    upstream_kinematic_node
                ]

        upstream_inertial_inflow: np.float32 = np.float32(0.0)
        rev_outflow_upstream: np.float32 = np.float32(0.0)
        upstream_inertial_reach: int = topo.inertial_up_reach_idx[reach_idx]
        if upstream_inertial_reach >= 0:
            q_up_prev: np.float32 = previous_discharge_m3_s_inertial[
                upstream_inertial_reach
            ]
            if q_up_prev >= np.float32(0.0):
                upstream_inertial_inflow += q_up_prev
            else:
                rev_outflow_upstream += -q_up_prev
        elif upstream_inertial_reach < -1:
            start_inertial_idx: int = -upstream_inertial_reach - 2
            end_inertial_idx: int = topo.inertial_up_offsets[reach_idx + 1]
            for idx in range(start_inertial_idx, end_inertial_idx):
                upstream_junction_reach: int = topo.inertial_up_indices[idx]
                q_up_prev = previous_discharge_m3_s_inertial[upstream_junction_reach]
                if q_up_prev >= np.float32(0.0):
                    upstream_inertial_inflow += q_up_prev
                else:
                    rev_outflow_upstream += -q_up_prev

        kinematic_inflow_rate_inertial[reach_idx] = upstream_kinematic_inflow

        # Lateral inflow rate from land-surface runoff / interflow (or extraction if negative)
        sideflow_sub: np.float32 = sideflow_m3_inertial[reach_idx] * inv_dt_f32
        if sideflow_sub >= np.float32(0.0):
            sideflow_in: np.float32 = sideflow_sub
            sideflow_out: np.float32 = np.float32(0.0)
        else:
            sideflow_in: np.float32 = np.float32(0.0)
            sideflow_out: np.float32 = -sideflow_sub

        # Downstream reach interface flux: positive is normal forward outflow,
        # negative indicates reverse backwater flow entering from downstream.
        q_ds_prev: np.float32 = previous_discharge_m3_s_inertial[reach_idx]
        if q_ds_prev >= np.float32(0.0):
            outflow_downstream: np.float32 = q_ds_prev
            incoming_downstream_rev: np.float32 = np.float32(0.0)
        else:
            outflow_downstream: np.float32 = np.float32(0.0)
            incoming_downstream_rev: np.float32 = -q_ds_prev

        # Sum all water entering and leaving this reach control volume
        total_incoming_rate: np.float32 = (
            upstream_inertial_inflow
            + upstream_kinematic_inflow
            + incoming_downstream_rev
            + sideflow_in
            + wb_inflow_rate[reach_idx]
        )
        total_outgoing_rate: np.float32 = (
            outflow_downstream + rev_outflow_upstream + sideflow_out
        )
        # Peak flow rate through the reach, used as characteristic Q for wave speed / normal depth
        total_flow_rate: np.float32 = max(total_incoming_rate, total_outgoing_rate)
        buffers.total_flow_rate_buf[reach_idx] = total_flow_rate
        current_volume_m3: float = river_storage_m3_inertial[reach_idx]

        # Calculate the maximum CFL-stable timestep duration for this specific reach
        buffers.min_dt_buf[reach_idx] = _evaluate_reach_cfl_dt(
            reach_idx=reach_idx,
            total_flow_rate=total_flow_rate,
            current_volume_m3=current_volume_m3,
            geom_inbank=geom.geom_inbank,
            geom_overbank=geom.geom_overbank,
            geom_cfl=geom.geom_cfl,
            gravity_acceleration=gravity_acceleration,
            dt_f32=dt_f32,
        )

    # Find the most restrictive reach in the entire domain and compute the required sub-steps
    limiting_reach_idx: int = int(np.argmin(buffers.min_dt_buf))
    min_stable_timestep_s: np.float32 = buffers.min_dt_buf[limiting_reach_idx]
    raw_substeps: int = int(np.ceil(dt_f32 / min_stable_timestep_s))
    num_inertial_substeps: int = max(1, raw_substeps)

    h_vol, h_flow, h_gov, celerity = _compute_reach_cfl_depths(
        reach_idx=limiting_reach_idx,
        total_flow_rate=buffers.total_flow_rate_buf[limiting_reach_idx],
        current_volume_m3=river_storage_m3_inertial[limiting_reach_idx],
        geom_inbank=geom.geom_inbank,
        geom_overbank=geom.geom_overbank,
        geom_cfl=geom.geom_cfl,
        gravity_acceleration=gravity_acceleration,
    )
    limiting_flow: np.float32 = buffers.total_flow_rate_buf[limiting_reach_idx]
    limiting_storage: np.float32 = np.float32(
        river_storage_m3_inertial[limiting_reach_idx]
    )

    return RoutingStepDiagnostics(
        num_inertial_substeps=num_inertial_substeps,
        limiting_reach_idx=limiting_reach_idx,
        min_stable_timestep_s=min_stable_timestep_s,
        h_vol=h_vol,
        h_flow=h_flow,
        h_gov=h_gov,
        celerity=celerity,
        limiting_flow=limiting_flow,
        limiting_storage=limiting_storage,
    )


@njit(inline="always")
def _update_waterbodies_substep(
    n_inertial: int,
    substep_fraction_f64: float,
    dt_substep: np.float32,
    wb: WaterbodyCoupling,
    buffers: SubstepBuffers,
) -> None:
    """Updates waterbody water levels, rating curves, and transfers boundary releases.

    Args:
        n_inertial: Number of reaches in local inertial domain.
        substep_fraction_f64: Fraction of macro timestep represented by current substep.
        dt_substep: Duration of current adaptive sub-timestep (seconds).
        wb: Waterbody coupling structure containing geometries, links, and storages.
        buffers: Pre-allocated workspace scratch buffers.
    """
    n_wb: int = wb.prescribed_outflow_m3.size
    n_lake_links: int = wb.lake_outflow_target_k.size

    if n_wb > 0:
        for wb_id in range(n_wb):
            lake_area: np.float32 = wb.lake_area[wb_id]
            depth_from_bottom: np.float32 = np.float32(wb.storage_m3[wb_id]) / lake_area
            bottom_elevation: np.float32 = (
                wb.outflow_bed_elev[wb_id] - wb.outflow_height[wb_id]
            )
            wb_stage: np.float32 = bottom_elevation + depth_from_bottom
            buffers.stage_buf[n_inertial + wb_id] = wb_stage

            target_outflow: float = (
                float(wb.prescribed_outflow_m3[wb_id]) * substep_fraction_f64
            )
            # For waterbodies without pre-defined outflow, evaluate weir rating curve
            if np.isnan(target_outflow):
                head_above_sill: np.float32 = max(
                    depth_from_bottom - wb.outflow_height[wb_id],
                    np.float32(0.0),
                )
                if head_above_sill > np.float32(1e-5):
                    lake_factor: np.float32 = wb.lake_factor[wb_id]
                    outflow_rate_m3_s: np.float32 = lake_factor * (
                        head_above_sill * head_above_sill
                    )
                    lake_outflow_vol: float = min(
                        float(outflow_rate_m3_s * dt_substep),
                        float(head_above_sill * lake_area),
                    )
                    buffers.wb_outflow_avail_buf[wb_id] = lake_outflow_vol
                else:
                    buffers.wb_outflow_avail_buf[wb_id] = 0.0
            else:
                # For waterbodies with predefined outflow (e.g. controlled reservoirs)
                buffers.wb_outflow_avail_buf[wb_id] = (
                    min(target_outflow, float(wb.storage_m3[wb_id]))
                    if wb.storage_m3[wb_id] > 0.0
                    else 0.0
                )

        # Buffer waterbody releases into reaches as continuous boundary flux
        if n_lake_links > 0:
            for link_idx in range(n_lake_links):
                wb_id_node: int = wb.lake_outflow_wb_id[link_idx]
                release: float = buffers.wb_outflow_avail_buf[wb_id_node]
                if release > 0.0:
                    target_reach: int = wb.lake_outflow_target_k[link_idx]
                    buffers.wb_release_volume_substep[target_reach] += np.float32(
                        release
                    )
                    buffers.wb_extra_lateral_accum_m3[target_reach] += release
                    buffers.wb_outflow_avail_buf[wb_id_node] = 0.0
                    wb.storage_m3[wb_id_node] -= release

        # Transfer waterbody releases to connected downstream cascading waterbodies
        n_wb_to_wb: int = len(wb.wb_to_wb_src_wb)
        if n_wb_to_wb > 0:
            for link_idx in range(n_wb_to_wb):
                src_wb: int = wb.wb_to_wb_src_wb[link_idx]
                release = buffers.wb_outflow_avail_buf[src_wb]
                if release > 0.0:
                    tgt_wb: int = wb.wb_to_wb_tgt_wb[link_idx]
                    wb.storage_m3[src_wb] -= release
                    wb.storage_m3[tgt_wb] += release
                    wb.inflow_m3[tgt_wb] += np.float32(release)
                    buffers.wb_outflow_avail_buf[src_wb] = 0.0

        # Transfer releases of terminal waterbodies discharging outside the model domain
        n_wb_term: int = len(wb.wb_terminal_wb_ids)
        if n_wb_term > 0:
            for link_idx in range(n_wb_term):
                term_wb: int = wb.wb_terminal_wb_ids[link_idx]
                release = buffers.wb_outflow_avail_buf[term_wb]
                if release > 0.0:
                    wb.storage_m3[term_wb] -= release
                    buffers.terminal_wb_outflow_accum_m3[0] += np.float32(release)
                    buffers.wb_outflow_avail_buf[term_wb] = 0.0


@njit(inline="always")
def _solve_momentum_loop(
    n_inertial: int,
    substep_discharge_m3_s: ArrayFloat32,
    river_storage_m3_inertial: ArrayFloat64,
    updated_discharge_m3_s_inertial: ArrayFloat32,
    geom: InertialGeometry,
    topo: InertialTopology,
    wb: WaterbodyCoupling,
    buffers: SubstepBuffers,
    g_dt_substep: np.float32,
    inv_dt_substep: np.float32,
    sqrt_gravity: np.float32,
    min_wet_depth_m: np.float32,
) -> None:
    """Evaluates the 1D local inertial momentum equation across all reaches.

    Args:
        n_inertial: Number of reaches in local inertial domain.
        substep_discharge_m3_s: Primary reach discharge array from previous substep (m³/s).
        river_storage_m3_inertial: Primary reach water storage volume array (m³).
        updated_discharge_m3_s_inertial: Destination buffer storing updated substep discharge (m³/s).
        geom: Static channel and floodplain geometry lookup tables.
        topo: Network connectivity and topological ordering structure.
        wb: Waterbody coupling parameters and storage state arrays.
        buffers: Pre-allocated workspace scratch buffers.
        g_dt_substep: Product of gravitational acceleration and substep duration (m/s).
        inv_dt_substep: Reciprocal of substep duration (1/s).
        sqrt_gravity: Square root of gravitational acceleration (m^(1/2)/s).
        min_wet_depth_m: Minimum depth threshold below which flow is truncated to zero (meters).
    """
    ds_boundary_type = topo.ds_boundary_type
    ds_stage_idx = topo.ds_stage_idx
    inertial_up_reach_idx = topo.inertial_up_reach_idx
    inertial_up_offsets = topo.inertial_up_offsets
    inertial_up_indices = topo.inertial_up_indices

    geom_inbank = geom.geom_inbank
    geom_overbank = geom.geom_overbank
    pit_slope_inertial = geom.pit_slope_inertial

    stage_buf = buffers.stage_buf
    lake_outflow_target_k = wb.lake_outflow_target_k
    lake_outflow_wb_id = wb.lake_outflow_wb_id

    n_lake_links: int = len(lake_outflow_target_k)

    for reach_idx in prange(n_inertial):
        boundary_type: int = ds_boundary_type[reach_idx]
        water_stage_node: np.float32 = stage_buf[reach_idx]
        inv_interface_len: np.float32 = geom_inbank[
            reach_idx, GEOM_IN_INVERSE_INTERFACE_LENGTH
        ]

        effective_depth: np.float32 = np.float32(0.0)
        water_slope: np.float32 = np.float32(0.0)

        water_stage_ds: np.float32 = water_stage_node
        reach_can_reverse: bool = False
        if boundary_type == 0 or boundary_type == 2:
            # Internal reach or lake boundary
            ds_idx: int = ds_stage_idx[reach_idx]
            water_stage_ds: np.float32 = stage_buf[ds_idx]
            max_bed: np.float32 = geom_inbank[
                reach_idx, GEOM_IN_INTERFACE_BED_ELEVATION_MAX
            ]
            bed_elev_node: np.float32 = geom_inbank[reach_idx, GEOM_IN_BED_ELEVATION]
            max_stage: np.float32 = max(water_stage_node, water_stage_ds)
            effective_depth = max(max_stage - max_bed, np.float32(0.0))
            water_slope = (water_stage_ds - water_stage_node) * inv_interface_len
            reach_can_reverse = bool(
                ALLOW_REVERSE_FLOW
                and boundary_type == 0
                and water_stage_ds > bed_elev_node
            )
        elif boundary_type == 3 or boundary_type == 4:
            # Open boundary reaches (no downstream water level available):
            #   - boundary_type 3: inland outlet / domain edge (pit)
            #   - boundary_type 4: coastal outlet to the sea (ocean pit)
            bed_elev_node = geom_inbank[reach_idx, GEOM_IN_BED_ELEVATION]
            effective_depth = max(water_stage_node - bed_elev_node, np.float32(0.0))

            # (a) Incoming surface slope from upstream reaches
            max_upstream_slope: np.float32 = np.float32(0.0)
            upstream_reach_k: int = inertial_up_reach_idx[reach_idx]
            if upstream_reach_k >= 0:
                upstream_stage: np.float32 = stage_buf[upstream_reach_k]
                inv_up_dx: np.float32 = geom_inbank[
                    upstream_reach_k, GEOM_IN_INVERSE_INTERFACE_LENGTH
                ]
                s_up: np.float32 = (upstream_stage - water_stage_node) * inv_up_dx
                if s_up > np.float32(0.0):
                    max_upstream_slope = s_up
            elif upstream_reach_k < -1:
                start_up_idx: int = -upstream_reach_k - 2
                end_up_idx: int = inertial_up_offsets[reach_idx + 1]
                for up_i in range(start_up_idx, end_up_idx):
                    u_k: int = inertial_up_indices[up_i]
                    upstream_stage = stage_buf[u_k]
                    inv_up_dx = geom_inbank[u_k, GEOM_IN_INVERSE_INTERFACE_LENGTH]
                    s_up = (upstream_stage - water_stage_node) * inv_up_dx
                    if s_up > max_upstream_slope:
                        max_upstream_slope = s_up

            # Also check incoming surface slope from upstream lakes/reservoirs:
            if n_lake_links > 0:
                for link_i in range(n_lake_links):
                    if lake_outflow_target_k[link_i] == reach_idx:
                        wb_id: int = lake_outflow_wb_id[link_i]
                        wb_stage: np.float32 = stage_buf[n_inertial + wb_id]
                        s_wb: np.float32 = (
                            wb_stage - water_stage_node
                        ) * inv_interface_len
                        if s_wb > max_upstream_slope:
                            max_upstream_slope = s_wb

            # (b) Free drawdown slope
            drawdown_slope: np.float32 = effective_depth * inv_interface_len

            # (c) Pit outlet bed slope as minimum baseline slope
            base_slope: np.float32 = pit_slope_inertial[reach_idx]
            outflow_slope: np.float32 = max(
                max_upstream_slope, drawdown_slope, base_slope
            )
            if boundary_type == 4:
                # (d) Ocean outlet: drop to sea level
                sea_stage: np.float32 = np.float32(0.0)
                if water_stage_node > sea_stage:
                    sea_drop_slope: np.float32 = (
                        water_stage_node - sea_stage
                    ) * inv_interface_len
                    outflow_slope = max(outflow_slope, sea_drop_slope)
            water_slope = -outflow_slope

        discharge: np.float32 = solve_inertial_momentum(
            reach_idx=reach_idx,
            effective_depth=effective_depth,
            water_slope=water_slope,
            curr_discharge=substep_discharge_m3_s[reach_idx],
            min_wet_depth_m=min_wet_depth_m,
            geom_inbank=geom_inbank,
            geom_overbank=geom_overbank,
            g_dt_substep=g_dt_substep,
            sqrt_gravity=sqrt_gravity,
            can_reverse=reach_can_reverse,
            river_storage_m3_inertial=river_storage_m3_inertial,
            inv_dt_substep=inv_dt_substep,
            upstream_stage_m=water_stage_node,
            downstream_stage_m=water_stage_ds,
        )
        updated_discharge_m3_s_inertial[reach_idx] = discharge


@njit(inline="always")
def _apply_reverse_flow_limits(
    n_inertial: int,
    updated_discharge_m3_s_inertial: ArrayFloat32,
    river_storage_m3_inertial: ArrayFloat64,
    topo: InertialTopology,
    buffers: SubstepBuffers,
    dt_substep_f64: float,
    inv_dt_substep: np.float32,
) -> None:
    """Scales back reverse flows proportionally if confluence demands exceed storage.

    Args:
        n_inertial: Number of reaches in local inertial domain.
        updated_discharge_m3_s_inertial: Updated substep discharge array (m³/s).
        river_storage_m3_inertial: Reach water storage volume array (m³).
        topo: Network connectivity and topological ordering structure.
        buffers: Pre-allocated workspace scratch buffers.
        dt_substep_f64: Duration of current adaptive sub-timestep as float64 (seconds).
        inv_dt_substep: Reciprocal of substep duration (1/s).
    """
    has_reverse_flow: bool = False
    buffers.rev_demand_buf.fill(0.0)
    for reach_idx in range(n_inertial):
        q_cand: np.float32 = updated_discharge_m3_s_inertial[reach_idx]
        if q_cand < np.float32(0.0):
            ds_reach: int = topo.ds_inertial_k[reach_idx]
            if ds_reach >= 0:
                buffers.rev_demand_buf[ds_reach] += -q_cand
                has_reverse_flow = True

    if has_reverse_flow:
        for reach_idx in range(n_inertial):
            q_cand = updated_discharge_m3_s_inertial[reach_idx]
            if q_cand < np.float32(0.0):
                ds_reach = topo.ds_inertial_k[reach_idx]
                if ds_reach >= 0:
                    tot_rev_demand: np.float32 = buffers.rev_demand_buf[ds_reach]
                    ds_storage: float = river_storage_m3_inertial[ds_reach]
                    q_ds: np.float32 = updated_discharge_m3_s_inertial[ds_reach]
                    ds_fwd_vol: float = (
                        float(q_ds) * dt_substep_f64 if q_ds > np.float32(0.0) else 0.0
                    )
                    ds_avail_vol: float = max(ds_storage - ds_fwd_vol, 0.0)
                    if ds_avail_vol > 0.0 and tot_rev_demand > np.float32(0.0):
                        max_tot_rev_rate: np.float32 = (
                            np.float32(0.95) * np.float32(ds_avail_vol) * inv_dt_substep
                        )
                        if tot_rev_demand > max_tot_rev_rate:
                            scale: np.float32 = max_tot_rev_rate / tot_rev_demand
                            updated_discharge_m3_s_inertial[reach_idx] = q_cand * scale
                    else:
                        updated_discharge_m3_s_inertial[reach_idx] = np.float32(0.0)


@njit(inline="always")
def _apply_confluence_inflow_limits(
    n_inertial: int,
    updated_discharge_m3_s_inertial: ArrayFloat32,
    geom_inbank: TwoDArrayFloat32,
    geom_overbank: TwoDArrayFloat32,
    topo: InertialTopology,
    buffers: SubstepBuffers,
    dt_substep_f32: np.float32,
) -> None:
    """Scales back incoming forward tributary discharges at confluences to prevent stage overshoots.

    At river junctions where multiple reaches converge, simultaneously evaluating
    unconstrained momentum updates from all incoming tributaries can dump more volume
    than the confluence reach can accommodate in a single substep, causing non-physical
    stage overshoots and high-frequency numerical oscillations.

    This function calculates the maximum physical absorption capacity of each confluence
    reach—bounded by approaching upstream water stage and a maximum depth rise cap—and
    proportionally scales down incoming forward tributary discharges if total inflow exceeds
    that capacity. Unrouted water remains in upstream reaches, ensuring 100% mass conservation.

    Args:
        n_inertial: Number of reaches in local inertial domain.
        updated_discharge_m3_s_inertial: Updated substep discharge array (m³/s).
        geom_inbank: Compact 2D in-bank reach geometry table.
        geom_overbank: Secondary 2D overbank floodplain geometry table.
        topo: Network connectivity and topological ordering structure.
        buffers: Pre-allocated workspace scratch buffers.
        dt_substep_f32: Current adaptive sub-timestep duration (seconds).
    """
    for reach_idx in range(n_inertial):
        upstream_reach_k: int = topo.inertial_up_reach_idx[reach_idx]
        has_multiple_inertial: bool = upstream_reach_k < -1
        has_inertial_and_kinematic: bool = (
            upstream_reach_k >= 0
            and buffers.kinematic_inflow_rate_inertial[reach_idx] > np.float32(0.0)
        )

        if not (has_multiple_inertial or has_inertial_and_kinematic):
            continue

        tot_inertial_inflow: np.float32 = np.float32(0.0)
        max_up_stage: np.float32 = np.float32(-1e9)

        if has_multiple_inertial:
            start_up_idx: int = -upstream_reach_k - 2
            end_up_idx: int = topo.inertial_up_offsets[reach_idx + 1]
            for up_i in range(start_up_idx, end_up_idx):
                u_k: int = topo.inertial_up_indices[up_i]
                q_up: np.float32 = updated_discharge_m3_s_inertial[u_k]
                if q_up > np.float32(0.0):
                    tot_inertial_inflow += q_up
                    u_stage: np.float32 = buffers.stage_buf[u_k]
                    if u_stage > max_up_stage:
                        max_up_stage = u_stage
        else:
            u_k_single: int = upstream_reach_k
            q_up_single: np.float32 = updated_discharge_m3_s_inertial[u_k_single]
            if q_up_single > np.float32(0.0):
                tot_inertial_inflow = q_up_single
                max_up_stage = buffers.stage_buf[u_k_single]

        if tot_inertial_inflow <= np.float32(0.0):
            continue

        stage_c: np.float32 = buffers.stage_buf[reach_idx]
        bed_c: np.float32 = geom_inbank[reach_idx, GEOM_IN_BED_ELEVATION]
        depth_c: np.float32 = max(stage_c - bed_c, np.float32(0.0))
        bankfull_depth_c: np.float32 = geom_inbank[reach_idx, GEOM_IN_BANKFULL_DEPTH]

        depth_cap: np.float32 = max(
            np.float32(0.5) * depth_c,
            np.float32(0.2) * bankfull_depth_c,
            np.float32(0.05),
        )
        head_diff: np.float32 = max(max_up_stage - stage_c, np.float32(0.0))
        delta_z_max: np.float32 = (
            min(head_diff, depth_cap)
            if head_diff > np.float32(0.0)
            else np.float32(0.05)
        )

        inv_length_c: np.float32 = geom_inbank[reach_idx, GEOM_IN_INVERSE_LENGTH]
        reach_length_c: np.float32 = np.float32(1.0) / inv_length_c
        bf_width_c: np.float32 = geom_overbank[reach_idx, GEOM_OV_RIVER_WIDTH]
        fp_width_c: np.float32 = geom_overbank[reach_idx, GEOM_OV_FLOODPLAIN_WIDTH]

        if depth_c > bankfull_depth_c:
            top_width_c: np.float32 = bf_width_c + fp_width_c
        else:
            depth_ratio: np.float32 = depth_c / max(bankfull_depth_c, np.float32(1e-3))
            inv_shape_exp_plus_one: np.float32 = geom_inbank[
                reach_idx, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE
            ]
            shape_exponent: np.float32 = (
                np.float32(0.5)
                if inv_shape_exp_plus_one == np.float32(1.0 / 1.5)
                else (np.float32(1.0) / inv_shape_exp_plus_one) - np.float32(1.0)
            )
            top_width_c = max(
                bf_width_c * (depth_ratio**shape_exponent),
                bf_width_c * np.float32(0.1),
            )

        surface_area_c: np.float32 = top_width_c * reach_length_c
        delta_v_cap: np.float32 = surface_area_c * delta_z_max

        q_out_c: np.float32 = updated_discharge_m3_s_inertial[reach_idx]
        outflow_vol_c: np.float32 = (
            q_out_c * dt_substep_f32 if q_out_c > np.float32(0.0) else np.float32(0.0)
        )
        kin_inflow_vol: np.float32 = (
            buffers.kinematic_inflow_rate_inertial[reach_idx] * dt_substep_f32
        )
        v_in_avail: np.float32 = max(
            delta_v_cap + outflow_vol_c - kin_inflow_vol, np.float32(0.0)
        )
        v_in_proposed: np.float32 = tot_inertial_inflow * dt_substep_f32

        if v_in_proposed > v_in_avail:
            scale: np.float32 = v_in_avail / v_in_proposed
            if has_multiple_inertial:
                start_up_idx_s = -upstream_reach_k - 2
                end_up_idx_s = topo.inertial_up_offsets[reach_idx + 1]
                for up_i in range(start_up_idx_s, end_up_idx_s):
                    u_k_idx: int = topo.inertial_up_indices[up_i]
                    q_cand: np.float32 = updated_discharge_m3_s_inertial[u_k_idx]
                    if q_cand > np.float32(0.0):
                        updated_discharge_m3_s_inertial[u_k_idx] = q_cand * scale
            else:
                u_k_single_idx: int = upstream_reach_k
                q_cand_s: np.float32 = updated_discharge_m3_s_inertial[u_k_single_idx]
                if q_cand_s > np.float32(0.0):
                    updated_discharge_m3_s_inertial[u_k_single_idx] = q_cand_s * scale


@njit(inline="always")
def _route_inertial_retention(
    substep_discharge_m3_s: ArrayFloat32,
    sideflow_m3_inertial: ArrayFloat32,
    river_storage_m3_inertial: ArrayFloat64,
    ret: RetentionCoupling,
    buffers: SubstepBuffers,
    substep_fraction_f32: np.float32,
    inv_dt_substep: np.float32,
    dt_substep: np.float32,
) -> None:
    """Routes flood basin intake diversions and releases for coupled retention basins.

    Args:
        substep_discharge_m3_s: Primary reach discharge array (m³/s).
        sideflow_m3_inertial: Macro-timestep lateral sideflow volume (m³).
        river_storage_m3_inertial: Reach water storage volume array (m³).
        ret: Flood retention basin state arrays and parameters.
        buffers: Pre-allocated workspace scratch buffers.
        substep_fraction_f32: Fraction of macro timestep represented by current substep.
        inv_dt_substep: Reciprocal of substep duration (1/s).
        dt_substep: Duration of current adaptive sub-timestep (seconds).
    """
    n_ret_inertial: int = len(ret.inertial_retention_k)
    if n_ret_inertial > 0:
        for idx in range(n_ret_inertial):
            reach_idx: int = ret.inertial_retention_k[idx]
            retention_id: int = ret.inertial_retention_id[idx]
            discharge_positive: np.float32 = max(
                substep_discharge_m3_s[reach_idx], np.float32(0.0)
            )
            sideflow_sub: np.float32 = (
                sideflow_m3_inertial[reach_idx] * substep_fraction_f32
            )
            avail_flow_rate: np.float32 = (
                sideflow_sub * inv_dt_substep
            ) + discharge_positive
            inflow_limit: float = buffers.retention_inflow_limit_sub_buf[retention_id]
            max_outflow_limit: float = buffers.retention_max_outflow_limit_sub_buf[
                retention_id
            ]
            river_volume_substep: np.float32 = avail_flow_rate * dt_substep
            is_rising_limb: bool = bool(avail_flow_rate > discharge_positive)

            (
                diverted_vol,
                released_vol,
                ret.storage_m3[retention_id],
                river_volume_substep,
            ) = compute_retention_routing(
                dt=dt_substep,
                river_volume_m3=river_volume_substep,
                discharge_before_diversion_m3_s=avail_flow_rate,
                is_rising_limb=is_rising_limb,
                retention_storage_m3=ret.storage_m3[retention_id],
                retention_max_storage_m3=ret.max_storage_m3[retention_id],
                controlled_retention=ret.controlled_retention[retention_id],
                activation_threshold_m3_s=ret.activation_threshold_m3_s[retention_id],
                release_threshold_factor=ret.release_threshold_factor,
                inflow_limit_m3=np.float32(inflow_limit),
                max_outflow_limit_m3=np.float32(max_outflow_limit),
            )

            ret.inflow_m3[retention_id] += diverted_vol
            ret.outflow_m3[retention_id] += released_vol
            net_retention_vol: float = float(diverted_vol - released_vol)
            river_storage_m3_inertial[reach_idx] = max(
                river_storage_m3_inertial[reach_idx] - net_retention_vol, 0.0
            )


@njit(inline="always")
def _update_continuity_and_stages(
    n_inertial: int,
    substep_discharge_m3_s: ArrayFloat32,
    kinematic_inflow_rate: ArrayFloat32,
    sideflow_m3_inertial: ArrayFloat32,
    evaporation_m3_inertial: ArrayFloat32,
    river_storage_m3_inertial: ArrayFloat64,
    over_abstraction_m3_inertial: ArrayFloat32,
    actual_evaporation_m3_inertial: ArrayFloat32,
    geom: InertialGeometry,
    topo: InertialTopology,
    buffers: SubstepBuffers,
    dt_substep: np.float32,
    substep_fraction_f32: np.float32,
) -> None:
    """Updates river storage mass balance, evaporation, abstraction, and water stage.

    Args:
        n_inertial: Number of reaches in local inertial domain.
        substep_discharge_m3_s: Primary reach discharge array (m³/s).
        kinematic_inflow_rate: Upstream inflow rates from kinematic domain (m³/s).
        sideflow_m3_inertial: Macro-timestep lateral sideflow volume (m³).
        evaporation_m3_inertial: Macro-timestep potential channel evaporation volume (m³).
        river_storage_m3_inertial: Primary reach water storage volume array (m³).
        over_abstraction_m3_inertial: Output over-abstraction deficit volume accumulator (m³).
        actual_evaporation_m3_inertial: Output actual evaporation volume accumulator (m³).
        geom: Static channel and floodplain geometry lookup tables.
        topo: Network connectivity and topological ordering structure.
        buffers: Pre-allocated workspace scratch buffers.
        dt_substep: Duration of current adaptive sub-timestep (seconds).
        substep_fraction_f32: Fraction of macro timestep represented by current substep.
    """
    inertial_up_reach_idx = topo.inertial_up_reach_idx
    inertial_up_offsets = topo.inertial_up_offsets
    inertial_up_indices = topo.inertial_up_indices

    geom_inbank = geom.geom_inbank
    geom_overbank = geom.geom_overbank

    wb_release_volume_substep = buffers.wb_release_volume_substep
    discharge_vol_sum_inertial = buffers.discharge_vol_sum_inertial
    stage_buf = buffers.stage_buf

    for reach_idx in prange(n_inertial):
        # Sum upstream inflows
        inflow_rate: np.float32 = kinematic_inflow_rate[reach_idx]

        upstream_reach_idx: int = inertial_up_reach_idx[reach_idx]
        if upstream_reach_idx >= 0:
            inflow_rate += substep_discharge_m3_s[upstream_reach_idx]
        elif upstream_reach_idx < -1:
            start_up: int = -upstream_reach_idx - 2
            end_up: int = inertial_up_offsets[reach_idx + 1]
            for idx in range(start_up, end_up):
                inflow_rate += substep_discharge_m3_s[inertial_up_indices[idx]]

        outflow_rate: np.float32 = substep_discharge_m3_s[reach_idx]

        # Match the float64 storage precision so the same interface flux
        # cancels between neighboring reaches over many short weir substeps.
        net_flow_volume: np.float64 = (
            np.float64(inflow_rate) - np.float64(outflow_rate)
        ) * np.float64(dt_substep) + np.float64(wb_release_volume_substep[reach_idx])
        sideflow_volume: np.float32 = (
            sideflow_m3_inertial[reach_idx] * substep_fraction_f32
        )
        current_storage: float = river_storage_m3_inertial[reach_idx]

        if sideflow_volume < np.float32(0.0):
            abstraction_demand: np.float32 = -sideflow_volume
            available_for_abstraction: np.float32 = np.float32(
                max(current_storage + net_flow_volume, 0.0)
            )
            if abstraction_demand > available_for_abstraction:
                over_abstraction_m3_inertial[reach_idx] += (
                    abstraction_demand - available_for_abstraction
                )
                sideflow_volume = -available_for_abstraction

        new_storage: float = max(
            current_storage + float(net_flow_volume + sideflow_volume), 0.0
        )

        evaporation_volume: np.float32 = (
            evaporation_m3_inertial[reach_idx] * substep_fraction_f32
        )
        actual_evaporation: np.float32 = min(
            evaporation_volume, np.float32(new_storage)
        )
        actual_evaporation_m3_inertial[reach_idx] += actual_evaporation
        new_storage -= float(actual_evaporation)

        river_storage_m3_inertial[reach_idx] = new_storage
        discharge_vol_sum_inertial[reach_idx] += np.float64(outflow_rate) * np.float64(
            dt_substep
        )

        # Calculate water stage from updated storage volume
        reach_volume: np.float32 = np.float32(new_storage)
        stage_buf[reach_idx] = _compute_reach_stage_from_vol(
            reach_idx, reach_volume, geom_inbank, geom_overbank
        )


@njit(inline="always")
def _evaluate_cfl_dt_all(
    n_inertial: int,
    substep_discharge_m3_s: ArrayFloat32,
    kinematic_inflow_rate: ArrayFloat32,
    sideflow_m3_inertial: ArrayFloat32,
    river_storage_m3_inertial: ArrayFloat64,
    geom: InertialGeometry,
    topo: InertialTopology,
    buffers: SubstepBuffers,
    dt_f32: np.float32,
    inv_dt_substep: np.float32,
    substep_fraction_f32: np.float32,
    gravity_acceleration: np.float32,
) -> None:
    """Evaluates adaptive CFL stability timestep constraints across all inertial reaches.

    Args:
        n_inertial: Number of reaches in local inertial domain.
        substep_discharge_m3_s: Primary reach discharge array (m³/s).
        kinematic_inflow_rate: Upstream inflow rates from kinematic domain (m³/s).
        sideflow_m3_inertial: Macro-timestep lateral sideflow volume (m³).
        river_storage_m3_inertial: Primary reach water storage volume array (m³).
        geom: Static channel and floodplain geometry lookup tables.
        topo: Network connectivity and topological ordering structure.
        buffers: Pre-allocated workspace scratch buffers.
        dt_f32: Macro routing timestep duration (seconds).
        inv_dt_substep: Reciprocal of substep duration (1/s).
        substep_fraction_f32: Fraction of macro timestep represented by current substep.
        gravity_acceleration: Gravitational acceleration constant (m/s²).
    """
    inertial_up_reach_idx = topo.inertial_up_reach_idx
    inertial_up_offsets = topo.inertial_up_offsets
    inertial_up_indices = topo.inertial_up_indices

    geom_inbank = geom.geom_inbank
    geom_overbank = geom.geom_overbank
    geom_cfl = geom.geom_cfl

    wb_release_volume_substep = buffers.wb_release_volume_substep
    total_flow_rate_buf = buffers.total_flow_rate_buf
    min_dt_buf = buffers.min_dt_buf

    for reach_idx in range(n_inertial):
        upstream_reach_idx: int = inertial_up_reach_idx[reach_idx]
        incoming_upstream_inertial: np.float32 = np.float32(0.0)
        rev_outflow_upstream: np.float32 = np.float32(0.0)

        if upstream_reach_idx >= 0:
            q_up: np.float32 = substep_discharge_m3_s[upstream_reach_idx]
            if q_up >= np.float32(0.0):
                incoming_upstream_inertial += q_up
            else:
                rev_outflow_upstream += -q_up
        elif upstream_reach_idx < -1:
            start_up: int = -upstream_reach_idx - 2
            end_up: int = inertial_up_offsets[reach_idx + 1]
            for idx in range(start_up, end_up):
                u_reach: int = inertial_up_indices[idx]
                q_up = substep_discharge_m3_s[u_reach]
                if q_up >= np.float32(0.0):
                    incoming_upstream_inertial += q_up
                else:
                    rev_outflow_upstream += -q_up

        outflow_rate: np.float32 = substep_discharge_m3_s[reach_idx]
        if outflow_rate >= np.float32(0.0):
            outflow_downstream: np.float32 = outflow_rate
            incoming_downstream_rev: np.float32 = np.float32(0.0)
        else:
            outflow_downstream = np.float32(0.0)
            incoming_downstream_rev = -outflow_rate

        incoming_kinematic: np.float32 = kinematic_inflow_rate[reach_idx]
        wb_inflow_rate: np.float32 = (
            wb_release_volume_substep[reach_idx] * inv_dt_substep
        )

        sideflow_rate: np.float32 = (
            sideflow_m3_inertial[reach_idx] * substep_fraction_f32
        ) * inv_dt_substep
        if sideflow_rate >= np.float32(0.0):
            sideflow_in: np.float32 = sideflow_rate
            sideflow_out: np.float32 = np.float32(0.0)
        else:
            sideflow_in = np.float32(0.0)
            sideflow_out = -sideflow_rate

        total_incoming_rate: np.float32 = (
            incoming_kinematic
            + incoming_upstream_inertial
            + incoming_downstream_rev
            + wb_inflow_rate
            + sideflow_in
        )
        total_outgoing_rate: np.float32 = (
            outflow_downstream + rev_outflow_upstream + sideflow_out
        )

        flow_mag: np.float32 = max(total_incoming_rate, total_outgoing_rate)
        total_flow_rate_buf[reach_idx] = flow_mag

        min_dt_buf[reach_idx] = _evaluate_reach_cfl_dt(
            reach_idx=reach_idx,
            total_flow_rate=flow_mag,
            current_volume_m3=river_storage_m3_inertial[reach_idx],
            geom_inbank=geom_inbank,
            geom_overbank=geom_overbank,
            geom_cfl=geom_cfl,
            gravity_acceleration=gravity_acceleration,
            dt_f32=dt_f32,
        )


@njit(inline="always")
def _accumulate_downstream_waterbody_inflows(
    substep_discharge_m3_s: ArrayFloat32,
    wb: WaterbodyCoupling,
    dt_substep: np.float32,
) -> None:
    """Accumulates terminal reach outflow into receiving downstream waterbodies.

    Args:
        substep_discharge_m3_s: Primary reach discharge array (m³/s).
        wb: Waterbody coupling structure containing geometries, links, and storages.
        dt_substep: Duration of current adaptive sub-timestep (seconds).
    """
    n_wb_ds: int = len(wb.inertial_wb_ds_k)
    if n_wb_ds > 0:
        for idx in range(n_wb_ds):
            reach_idx: int = wb.inertial_wb_ds_k[idx]
            waterbody_id: int = wb.inertial_wb_ds_id[idx]
            q_wb: np.float32 = substep_discharge_m3_s[reach_idx]
            if q_wb > np.float32(0.0):
                inflow_vol: float = float(q_wb * dt_substep)
                wb.inflow_m3[waterbody_id] += np.float32(inflow_vol)
                wb.storage_m3[waterbody_id] += inflow_vol


@njit(inline="always")
def _limit_weir_timestep(
    dt_s: np.float32,
    geom: InertialGeometry,
    topo: InertialTopology,
    buffers: SubstepBuffers,
) -> np.float32:
    """Limit explicit weir exchange to a small fraction of the level difference.

    Exchange Q changes the level difference at a rate Q * (1/A_up + 1/A_down),
    where A = dV/d(level) is each reach's surface storage area. Limit this
    estimated change to 20% of the current level difference to reduce overshoot
    at submerged crests. This is a numerical safeguard, not a hydraulic loss;
    ordinary channel CFL limits and conservative volume limits still apply.

    Args:
        dt_s: Candidate routing timestep (seconds).
        geom: Static reach geometry, including crest heights (meters).
        topo: River connectivity.
        buffers: Current water elevations (meters) and scratch arrays.

    Returns:
        Candidate timestep shortened for weir exchange (seconds).
    """
    reach: np.int32
    cell: int | np.int32
    for reach in geom.weir_indices:
        height_m: np.float32 = geom.geom_inbank[reach, GEOM_IN_WEIR_HEIGHT]
        downstream: int = topo.ds_inertial_k[reach]
        upstream_stage: np.float32 = buffers.stage_buf[reach]
        downstream_stage: np.float32 = buffers.stage_buf[downstream]
        crest_m: np.float32 = geom.geom_inbank[reach, GEOM_IN_BED_ELEVATION] + height_m
        flow: np.float32 = rectangular_weir_discharge(
            upstream_stage,
            downstream_stage,
            crest_m,
            geom.geom_inbank[reach, GEOM_IN_WEIR_WIDTH],
        )
        if flow == 0 or (flow < 0 and not ALLOW_REVERSE_FLOW):
            continue
        inverse_storage_area: np.float32 = np.float32(0.0)
        for cell in (reach, downstream):
            depth_m: np.float32 = max(
                buffers.stage_buf[cell] - geom.geom_inbank[cell, GEOM_IN_BED_ELEVATION],
                np.float32(0.0),
            )
            bankfull_m: np.float32 = geom.geom_inbank[cell, GEOM_IN_BANKFULL_DEPTH]
            width_m: np.float32 = geom.geom_overbank[cell, GEOM_OV_RIVER_WIDTH]
            if depth_m <= bankfull_m:
                exponent: np.float32 = np.float32(1.0) / geom.geom_inbank[
                    cell, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE
                ] - np.float32(1.0)
                width_m *= (max(depth_m, np.float32(1e-4)) / bankfull_m) ** exponent
            else:
                # Match _compute_reach_stage_from_vol: above bankfull, storage
                # increments are L * (W_floodplain * h + 2*h**2). The bankfull
                # channel volume is fixed, so its width is not added again.
                width_m = geom.geom_overbank[
                    cell, GEOM_OV_FLOODPLAIN_WIDTH
                ] + np.float32(4.0) * (depth_m - bankfull_m)
            # Dry or newly flooded cells can have vanishing storage width.
            # Keep a small floor for a finite exchange timestep.
            inverse_storage_area += geom.geom_inbank[
                cell, GEOM_IN_INVERSE_LENGTH
            ] / max(width_m, np.float32(1e-3))
        exchange_dt: np.float32 = (
            np.float32(0.2)
            * abs(upstream_stage - downstream_stage)
            / (abs(flow) * inverse_storage_area)
        )
        dt_s = min(dt_s, exchange_dt)
    return dt_s


def _run_inertial_substeps(
    num_inertial_substeps: int,
    dt_f32: np.float32,
    n_inertial: int,
    substep_discharge_m3_s: ArrayFloat32,
    river_storage_m3_inertial: ArrayFloat64,
    sideflow_m3_inertial: ArrayFloat32,
    evaporation_m3_inertial: ArrayFloat32,
    over_abstraction_m3_inertial: ArrayFloat32,
    actual_evaporation_m3_inertial: ArrayFloat32,
    updated_discharge_m3_s_inertial: ArrayFloat32,
    kinematic_inflow_rate: ArrayFloat32,
    geom: InertialGeometry,
    topo: InertialTopology,
    wb: WaterbodyCoupling,
    ret: RetentionCoupling,
    buffers: SubstepBuffers,
) -> int:
    """Advances dynamic local inertial routing across adaptive substeps.

    Operates using a modular architecture per substep:
    - Stage 1: Waterbody Levels & Outflow Releases
    - Stage 2: Momentum Equation (evaluates instantaneous flux Q across all reaches)
    - Stage 3: Reverse Flow Multi-Outflow Limiter at Confluences
    - Stage 4: Retention Basin Inflow Diversion and Controlled Release
    - Stage 5: Continuity, Storage & Stage Update
    - Stage 6: Dynamic CFL Stability Evaluation across Reaches
    - Stage 7: Downstream Waterbody Inflow Accumulation

    Args:
        num_inertial_substeps: Initial number of sub-timesteps predicted by macro CFL.
        dt_f32: Macro routing timestep duration (seconds).
        n_inertial: Number of reaches routed with local inertial equations.
        substep_discharge_m3_s: Primary discharge array (m³/s).
        river_storage_m3_inertial: Primary reach water storage volume array (m³).
        sideflow_m3_inertial: Macro-timestep lateral sideflow volume (m³).
        evaporation_m3_inertial: Macro-timestep potential channel evaporation volume (m³).
        over_abstraction_m3_inertial: Accumulator tracking unfulfilled abstraction deficit volume (m³).
        actual_evaporation_m3_inertial: Accumulator tracking actual volume evaporated from channel (m³).
        updated_discharge_m3_s_inertial: Output array storing timestep-averaged reach discharge (m³/s).
        kinematic_inflow_rate: Upstream inflow rates from kinematic domain (m³/s).
        geom: Static channel and floodplain geometry lookup tables.
        topo: Network connectivity and topological ordering structure.
        wb: Waterbody coupling parameters, links, and storage state arrays.
        ret: Flood retention basin state arrays, capacities, and control parameters.
        buffers: Pre-allocated workspace scratch buffers.

    Returns:
        Total number of internal adaptive substeps executed.
    """
    gravity_acceleration: np.float32 = np.float32(9.80665)
    sqrt_gravity: np.float32 = np.sqrt(gravity_acceleration)
    min_wet_depth_m: np.float32 = np.float32(1e-4)

    buffers.discharge_vol_sum_inertial.fill(0.0)
    buffers.wb_extra_lateral_accum_m3.fill(0.0)
    buffers.terminal_wb_outflow_accum_m3.fill(0.0)
    buffers.wb_release_volume_substep.fill(0.0)

    # Initialize reach water stages from current storage to ensure physical consistency
    _compute_stage_from_storage_inertial(
        n_inertial,
        river_storage_m3_inertial,
        geom.geom_inbank,
        geom.geom_overbank,
        buffers.stage_buf,
    )

    dt_substep: np.float32 = dt_f32 / np.float32(num_inertial_substeps)

    # Many short weir substeps otherwise accumulate float32 timing error and
    # integrate sideflow over slightly more or less than the routing interval.
    t_elapsed: float = 0.0
    substep_count: int = 0
    n_ret: int = ret.max_storage_m3.size

    while t_elapsed < dt_f32:
        time_remaining: np.float32 = np.float32(float(dt_f32) - t_elapsed)
        if time_remaining <= np.float32(1e-4):
            break

        if dt_substep >= time_remaining or (
            time_remaining - dt_substep <= np.float32(1e-3)
        ):
            dt_substep = time_remaining

        dt_substep = _limit_weir_timestep(dt_substep, geom, topo, buffers)
        inv_dt_substep: np.float32 = np.float32(1.0) / dt_substep
        g_dt_substep: np.float32 = gravity_acceleration * dt_substep
        dt_substep_f64: float = float(dt_substep)
        substep_fraction_f32: np.float32 = dt_substep / dt_f32
        substep_fraction_f64: float = float(substep_fraction_f32)

        for i in range(n_ret):
            ret_in_lim: np.float32 = np.float32(0.20) * ret.max_storage_m3[i]
            ret_out_lim: np.float32 = np.float32(0.05) * ret.max_storage_m3[i]
            buffers.retention_inflow_limit_sub_buf[i] = (
                np.float64(ret_in_lim) * substep_fraction_f64
            )
            buffers.retention_max_outflow_limit_sub_buf[i] = (
                np.float64(ret_out_lim) * substep_fraction_f64
            )

        buffers.wb_release_volume_substep.fill(0.0)

        # 1. Update waterbody water levels and calculate continuous releases
        _update_waterbodies_substep(
            n_inertial=n_inertial,
            substep_fraction_f64=substep_fraction_f64,
            dt_substep=dt_substep,
            wb=wb,
            buffers=buffers,
        )

        # 2. Solve 1D inertial momentum equation
        _solve_momentum_loop(
            n_inertial=n_inertial,
            substep_discharge_m3_s=substep_discharge_m3_s,
            river_storage_m3_inertial=river_storage_m3_inertial,
            updated_discharge_m3_s_inertial=updated_discharge_m3_s_inertial,
            geom=geom,
            topo=topo,
            wb=wb,
            buffers=buffers,
            g_dt_substep=g_dt_substep,
            inv_dt_substep=inv_dt_substep,
            sqrt_gravity=sqrt_gravity,
            min_wet_depth_m=min_wet_depth_m,
        )

        # 3. Scale reverse flows proportionally if total demand exceeds available storage
        _apply_reverse_flow_limits(
            n_inertial=n_inertial,
            updated_discharge_m3_s_inertial=updated_discharge_m3_s_inertial,
            river_storage_m3_inertial=river_storage_m3_inertial,
            topo=topo,
            buffers=buffers,
            dt_substep_f64=dt_substep_f64,
            inv_dt_substep=inv_dt_substep,
        )

        # 4. Limit forward inflows at confluences to prevent stage overshooting
        _apply_confluence_inflow_limits(
            n_inertial=n_inertial,
            updated_discharge_m3_s_inertial=updated_discharge_m3_s_inertial,
            geom_inbank=geom.geom_inbank,
            geom_overbank=geom.geom_overbank,
            topo=topo,
            buffers=buffers,
            dt_substep_f32=dt_substep,
        )

        substep_discharge_m3_s[:] = updated_discharge_m3_s_inertial[:]

        # 4. Route diversions and releases for retention basins
        _route_inertial_retention(
            substep_discharge_m3_s=substep_discharge_m3_s,
            sideflow_m3_inertial=sideflow_m3_inertial,
            river_storage_m3_inertial=river_storage_m3_inertial,
            ret=ret,
            buffers=buffers,
            substep_fraction_f32=substep_fraction_f32,
            inv_dt_substep=inv_dt_substep,
            dt_substep=dt_substep,
        )

        # 5. Continuity, Storage & Water Stage Update
        _update_continuity_and_stages(
            n_inertial=n_inertial,
            substep_discharge_m3_s=substep_discharge_m3_s,
            kinematic_inflow_rate=kinematic_inflow_rate,
            sideflow_m3_inertial=sideflow_m3_inertial,
            evaporation_m3_inertial=evaporation_m3_inertial,
            river_storage_m3_inertial=river_storage_m3_inertial,
            over_abstraction_m3_inertial=over_abstraction_m3_inertial,
            actual_evaporation_m3_inertial=actual_evaporation_m3_inertial,
            geom=geom,
            topo=topo,
            buffers=buffers,
            dt_substep=dt_substep,
            substep_fraction_f32=substep_fraction_f32,
        )

        # 6. Dynamic Stability & CFL Evaluation
        _evaluate_cfl_dt_all(
            n_inertial=n_inertial,
            substep_discharge_m3_s=substep_discharge_m3_s,
            kinematic_inflow_rate=kinematic_inflow_rate,
            sideflow_m3_inertial=sideflow_m3_inertial,
            river_storage_m3_inertial=river_storage_m3_inertial,
            geom=geom,
            topo=topo,
            buffers=buffers,
            dt_f32=dt_f32,
            inv_dt_substep=inv_dt_substep,
            substep_fraction_f32=substep_fraction_f32,
            gravity_acceleration=gravity_acceleration,
        )

        # 7. Accumulate inflows into downstream waterbodies
        _accumulate_downstream_waterbody_inflows(
            substep_discharge_m3_s=substep_discharge_m3_s,
            wb=wb,
            dt_substep=dt_substep,
        )

        t_elapsed += dt_substep_f64
        substep_count += 1

        if t_elapsed < dt_f32:
            min_dt_next: np.float32 = dt_f32
            for r_idx in range(n_inertial):
                if buffers.min_dt_buf[r_idx] < min_dt_next:
                    min_dt_next = buffers.min_dt_buf[r_idx]
            dt_substep = min_dt_next

    # Calculate average discharge over the time step
    inv_macro_dt: np.float32 = np.float32(1.0) / dt_f32
    updated_discharge_m3_s_inertial[:] = (
        buffers.discharge_vol_sum_inertial * inv_macro_dt
    )
    actual_evaporation_m3_inertial[:] = np.minimum(
        actual_evaporation_m3_inertial, evaporation_m3_inertial
    )

    return substep_count


_run_inertial_substeps_serial = njit(parallel=False)(_run_inertial_substeps)
_run_inertial_substeps_parallel = njit(parallel=True)(_run_inertial_substeps)
