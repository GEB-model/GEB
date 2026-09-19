"""1D local inertial river routing algorithm with kinematic routing for upstream reaches."""

import geopandas as gpd
import numpy as np
import pyflwdir
import pyflwdir.core
from numba import njit, prange

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    ArrayInt64,
    TwoDArrayFloat32,
    TwoDArrayInt32,
)

from .geometry import (
    compute_static_geometry,
)
from .inertial_substeps import (
    GEOM_CFL_CONSTANT,
    GEOM_CFL_MANNING_COEFFICIENT,
    GEOM_CFL_NUM_COLS,
    GEOM_IN_BANKFULL_DEPTH,
    GEOM_IN_BANKFULL_VOLUME,
    GEOM_IN_BED_ELEVATION,
    GEOM_IN_INTERFACE_BED_ELEVATION_MAX,
    GEOM_IN_INVERSE_INTERFACE_LENGTH,
    GEOM_IN_INVERSE_LENGTH,
    GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE,
    GEOM_IN_MANNING_N_SQUARED,
    GEOM_IN_NUM_COLS,
    GEOM_IN_STAGE_VOLUME_COEFFICIENT,
    GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH,
    GEOM_OV_BANKFULL_PERIMETER,
    GEOM_OV_FLOODPLAIN_AREA_THRESHOLD,
    GEOM_OV_FLOODPLAIN_DEPTH_THRESHOLD,
    GEOM_OV_FLOODPLAIN_SIDE_SLOPE,
    GEOM_OV_FLOODPLAIN_WIDTH,
    GEOM_OV_NUM_COLS,
    GEOM_OV_RIVER_WIDTH,
    GEOM_OV_SQRT_ONE_PLUS_FLOODPLAIN_SLOPE_SQUARED,
    INERTIAL_PARALLEL_THRESHOLD,
    _compute_stage_from_storage_inertial,
    _gather_inputs,
    _run_inertial_substeps_parallel,
    _run_inertial_substeps_serial,
    _scatter_outputs,
    compute_inertial_substeps_cfl,
    compute_retention_routing,
)
from .kinematic import update_node_kinematic

__all__: list[str] = [
    "LocalInertial",
    "KINEMATIC_PARALLEL_THRESHOLD",
    "_run_kinematic_step",
    "_run_kinematic_step_parallel",
    "_run_kinematic_step_serial",
]

# Threshold of headwater reach count above which parallel multi-threading outperforms serial execution
KINEMATIC_PARALLEL_THRESHOLD: int = 5000


@njit(cache=True)
def _transfer_waterbody_outflows(
    wb_to_wb_src_wb: ArrayInt32,
    wb_to_wb_tgt_wb: ArrayInt32,
    inertial_outflow_per_waterbody: ArrayFloat32,
    waterbody_storage_m3: ArrayFloat64,
    waterbody_inflow_m3: ArrayFloat32,
    wb_to_kin_target_reach: ArrayInt32,
    wb_to_kin_wb_id: ArrayInt32,
    kin_wb_inflow_m3: ArrayFloat32,
) -> None:
    """Transfers waterbody outflows to connected downstream waterbodies and kinematic reaches.

    Args:
        wb_to_wb_src_wb: Source waterbody IDs discharging into another waterbody.
        wb_to_wb_tgt_wb: Destination waterbody IDs receiving waterbody outflow.
        inertial_outflow_per_waterbody: Outflow volume buffered per waterbody (m³).
        waterbody_storage_m3: Storage volume array for all waterbodies (m³).
        waterbody_inflow_m3: Aggregated waterbody inflow volume (m³).
        wb_to_kin_target_reach: Target kinematic reach indices receiving waterbody outflow.
        wb_to_kin_wb_id: Source waterbody IDs discharging into kinematic reaches.
        kin_wb_inflow_m3: Waterbody inflow volume buffered per kinematic reach (m³).
    """
    # Waterbody-to-waterbody transfers
    for idx in range(len(wb_to_wb_src_wb)):
        src_wb: int = wb_to_wb_src_wb[idx]
        tgt_wb: int = wb_to_wb_tgt_wb[idx]
        outflow_vol: float = np.float64(inertial_outflow_per_waterbody[src_wb])
        if outflow_vol > 0.0:
            waterbody_storage_m3[src_wb] -= outflow_vol
            waterbody_storage_m3[tgt_wb] += outflow_vol
            waterbody_inflow_m3[tgt_wb] += np.float32(outflow_vol)
            inertial_outflow_per_waterbody[src_wb] = np.float32(0.0)

    # Waterbody outflows into receiving kinematic reaches
    for idx in range(len(wb_to_kin_target_reach)):
        tgt_reach: int = wb_to_kin_target_reach[idx]
        wb_id_up: int = wb_to_kin_wb_id[idx]
        wb_of: np.float32 = inertial_outflow_per_waterbody[wb_id_up]
        if wb_of > np.float32(0.0):
            rel_vol: float = np.float64(wb_of)
            waterbody_storage_m3[wb_id_up] -= rel_vol
            kin_wb_inflow_m3[tgt_reach] += wb_of
            inertial_outflow_per_waterbody[wb_id_up] = np.float32(0.0)


def _run_kinematic_step(
    dt_f32: np.float32,
    inv_dt_f32: np.float32,
    n_kinematic: int,
    n_kin_headwater: int,
    kin_ds_reach: ArrayInt32,
    kin_inflow_m3_s: ArrayFloat32,
    sideflow_m3: ArrayFloat32,
    evaporation_m3: ArrayFloat32,
    previous_discharge_m3_s: ArrayFloat32,
    river_storage_alpha: ArrayFloat32,
    river_storage_beta: ArrayFloat32,
    river_length: ArrayFloat32,
    river_storage_m3: ArrayFloat64,
    updated_discharge_m3_s: ArrayFloat32,
    actual_evaporation_m3: ArrayFloat32,
    over_abstraction_m3: ArrayFloat32,
    kin_ds_waterbody_id: ArrayInt32,
    waterbody_storage_m3: ArrayFloat64,
    waterbody_inflow_m3: ArrayFloat32,
    wb_to_kin_target_reach: ArrayInt32,
    kin_wb_inflow_m3: ArrayFloat32,
    retention_node_id: ArrayInt32,
    retention_storage_m3: ArrayFloat32,
    retention_max_storage_m3: ArrayFloat32,
    controlled_retention: ArrayBool,
    retention_activation_threshold_m3_s: ArrayFloat32,
    retention_basin_release_threshold_factor: np.float32,
    retention_inflow_m3: ArrayFloat32,
    retention_outflow_m3: ArrayFloat32,
) -> None:
    """Executes kinematic wave routing across headwater and connected reaches with in-place updates.

    Phase 1 solves all independent headwater reaches (in-degree 0) in parallel using prange.
    Phase 2 solves all downstream connected reaches in topological order with O(1) push-based inflow accumulation.

    Args:
        dt_f32: Routing time step duration (seconds).
        inv_dt_f32: Inverse of routing time step duration (1/s).
        n_kinematic: Total count of kinematic reaches.
        n_kin_headwater: Count of independent headwater kinematic reaches.
        kin_ds_reach: Target downstream kinematic reach index (-1 if exiting kinematic domain).
        kin_inflow_m3_s: Workspace buffer accumulating upstream inflow discharge rates (m³/s).
        sideflow_m3: Lateral inflow volume per reach (m³).
        evaporation_m3: Potential evaporation volume per reach (m³).
        previous_discharge_m3_s: Discharge from previous time step (m³/s).
        river_storage_alpha: Kinematic wave alpha parameter.
        river_storage_beta: Kinematic wave beta parameter.
        river_length: Channel length per reach (meters).
        river_storage_m3: River reach storage volume (m³).
        updated_discharge_m3_s: Output discharge array to populate (m³/s).
        actual_evaporation_m3: Output actual evaporation volume array to populate (m³).
        over_abstraction_m3: Cumulative volume of over-abstraction (m³).
        kin_ds_waterbody_id: Target downstream waterbody ID per reach (-1 if none).
        waterbody_storage_m3: Storage volume array for all waterbodies (m³).
        waterbody_inflow_m3: Aggregated waterbody inflow volume (m³).
        wb_to_kin_target_reach: Target kinematic reach indices receiving waterbody outflow.
        kin_wb_inflow_m3: Waterbody inflow volume buffered per kinematic reach (m³).
        retention_node_id: Retention basin mapping ID per reach (-1 if none).
        retention_storage_m3: Retention basin current storage volume (m³).
        retention_max_storage_m3: Retention basin capacity (m³).
        controlled_retention: Controlled retention operational flag.
        retention_activation_threshold_m3_s: Discharge threshold triggering diversion (m³/s).
        retention_basin_release_threshold_factor: Threshold factor governing retention release.
        retention_inflow_m3: Total volume diverted into retention basin (m³).
        retention_outflow_m3: Total volume released from retention basin (m³).

    Raises:
        ValueError: If non-finite discharge is computed in any kinematic wave reach.
    """
    kin_inflow_m3_s.fill(np.float32(0.0))

    # Route headwater reaches in parallel across threads.
    # Headwaters have no upstream river reaches feeding them, making them mutually
    # independent. They can be computed concurrently without race conditions.
    for i in prange(n_kin_headwater):  # ty: ignore[not-iterable]
        # Collect local lateral runoff and waterbody releases (upstream river inflow is 0).
        node_sideflow: np.float32 = sideflow_m3[i]
        if len(wb_to_kin_target_reach) > 0:
            node_sideflow += kin_wb_inflow_m3[i]
            kin_wb_inflow_m3[i] = np.float32(0.0)

        upstream_inflow_m3_s: np.float32 = np.float32(0.0)

        # Divert peak floodwater to (or release water from) retention basins if present.
        ret_id: int = retention_node_id[i]
        if ret_id != -1:
            discharge_before_diversion: np.float32 = max(
                upstream_inflow_m3_s, np.float32(0.0)
            ) + (node_sideflow * inv_dt_f32)

            discharge_at_basin_vol: np.float32 = (
                upstream_inflow_m3_s * dt_f32 + node_sideflow
            )
            inflow_limit: np.float32 = (
                np.float32(0.20) * retention_max_storage_m3[ret_id]
            )
            max_outflow_limit: np.float32 = (
                np.float32(0.05) * retention_max_storage_m3[ret_id]
            )
            is_rising_limb: bool = (
                discharge_before_diversion > previous_discharge_m3_s[i]
            )

            (
                diverted_volume,
                outflow_volume,
                retention_storage_m3[ret_id],
                discharge_at_basin_vol,
            ) = compute_retention_routing(
                dt=dt_f32,
                river_volume_m3=discharge_at_basin_vol,
                discharge_before_diversion_m3_s=discharge_before_diversion,
                is_rising_limb=bool(is_rising_limb),
                retention_storage_m3=retention_storage_m3[ret_id],
                retention_max_storage_m3=retention_max_storage_m3[ret_id],
                controlled_retention=controlled_retention[ret_id],
                activation_threshold_m3_s=retention_activation_threshold_m3_s[ret_id],
                release_threshold_factor=retention_basin_release_threshold_factor,
                inflow_limit_m3=np.float32(inflow_limit),
                max_outflow_limit_m3=np.float32(max_outflow_limit),
            )

            retention_inflow_m3[ret_id] += diverted_volume
            retention_outflow_m3[ret_id] += outflow_volume
            node_sideflow = discharge_at_basin_vol - upstream_inflow_m3_s * dt_f32

        # Solve kinematic wave routing for outflow discharge and channel evaporation.
        kinematic_discharge, act_evap_rate = update_node_kinematic(
            inflow_m3_s=upstream_inflow_m3_s,
            previous_discharge_m3_s=previous_discharge_m3_s[i],
            sideflow_m3_s=node_sideflow * inv_dt_f32,
            evaporation_m3_s=evaporation_m3[i] * inv_dt_f32,
            river_storage_alpha=river_storage_alpha[i],
            river_storage_beta=river_storage_beta[i],
            timestep_s=dt_f32,
            river_length_m=river_length[i],
        )

        kinematic_discharge = max(kinematic_discharge, np.float32(0.0))
        updated_discharge_m3_s[i] = kinematic_discharge

        evap_vol: np.float32 = act_evap_rate * dt_f32
        actual_evaporation_m3[i] = evap_vol

        # Update reach storage (mass balance) and track over-abstraction deficit if any.
        inflow_vol: np.float32 = upstream_inflow_m3_s * dt_f32 + node_sideflow
        outflow_vol: np.float32 = kinematic_discharge * dt_f32
        river_storage_m3[i] += (
            np.float64(inflow_vol) - np.float64(outflow_vol) - np.float64(evap_vol)
        )

        if river_storage_m3[i] < np.float64(0.0):
            over_abstraction_m3[i] += np.float32(-river_storage_m3[i])
            river_storage_m3[i] = np.float64(0.0)

    # Pass headwater outflows to downstream reaches and waterbodies.
    # This must be done in serial to avoid race conditions.
    for i in range(n_kin_headwater):
        q_head: np.float32 = updated_discharge_m3_s[i]
        if __debug__:
            if not np.isfinite(q_head):
                raise ValueError(
                    "Non-finite discharge computed in kinematic wave reach."
                )
        ds_reach: int = kin_ds_reach[i]
        if ds_reach != -1:
            kin_inflow_m3_s[ds_reach] += q_head

        wb_ds_id: int = kin_ds_waterbody_id[i]
        if wb_ds_id != -1:
            outflow_vol_hw: np.float32 = q_head * dt_f32
            waterbody_storage_m3[wb_ds_id] += np.float64(outflow_vol_hw)
            if outflow_vol_hw > np.float32(0.0):
                waterbody_inflow_m3[wb_ds_id] += outflow_vol_hw

    # Route connected reaches in downstream topological order.
    # Because reaches are sorted topologically, all upstream reaches feeding reach `i`
    # have already finished and deposited their outflow into kin_inflow_m3_s[i].
    for i in range(n_kin_headwater, n_kinematic):
        node_sideflow = sideflow_m3[i]
        if len(wb_to_kin_target_reach) > 0:
            node_sideflow += kin_wb_inflow_m3[i]
            kin_wb_inflow_m3[i] = np.float32(0.0)

        upstream_inflow_m3_s = kin_inflow_m3_s[i]

        # Manage flood retention diversions/releases.
        ret_id = retention_node_id[i]
        if ret_id != -1:
            discharge_before_diversion = max(upstream_inflow_m3_s, np.float32(0.0)) + (
                node_sideflow * inv_dt_f32
            )

            discharge_at_basin_vol = upstream_inflow_m3_s * dt_f32 + node_sideflow
            inflow_limit = np.float32(0.20) * retention_max_storage_m3[ret_id]
            max_outflow_limit = np.float32(0.05) * retention_max_storage_m3[ret_id]
            is_rising_limb = discharge_before_diversion > previous_discharge_m3_s[i]

            (
                diverted_volume,
                outflow_volume,
                retention_storage_m3[ret_id],
                discharge_at_basin_vol,
            ) = compute_retention_routing(
                dt=dt_f32,
                river_volume_m3=discharge_at_basin_vol,
                discharge_before_diversion_m3_s=discharge_before_diversion,
                is_rising_limb=bool(is_rising_limb),
                retention_storage_m3=retention_storage_m3[ret_id],
                retention_max_storage_m3=retention_max_storage_m3[ret_id],
                controlled_retention=controlled_retention[ret_id],
                activation_threshold_m3_s=retention_activation_threshold_m3_s[ret_id],
                release_threshold_factor=retention_basin_release_threshold_factor,
                inflow_limit_m3=np.float32(inflow_limit),
                max_outflow_limit_m3=np.float32(max_outflow_limit),
            )

            retention_inflow_m3[ret_id] += diverted_volume
            retention_outflow_m3[ret_id] += outflow_volume
            node_sideflow = discharge_at_basin_vol - upstream_inflow_m3_s * dt_f32

        # Solve kinematic wave routing for this reach.
        kinematic_discharge, act_evap_rate = update_node_kinematic(
            inflow_m3_s=upstream_inflow_m3_s,
            previous_discharge_m3_s=previous_discharge_m3_s[i],
            sideflow_m3_s=node_sideflow * inv_dt_f32,
            evaporation_m3_s=evaporation_m3[i] * inv_dt_f32,
            river_storage_alpha=river_storage_alpha[i],
            river_storage_beta=river_storage_beta[i],
            timestep_s=dt_f32,
            river_length_m=river_length[i],
        )

        kinematic_discharge = max(kinematic_discharge, np.float32(0.0))
        updated_discharge_m3_s[i] = kinematic_discharge

        evap_vol = act_evap_rate * dt_f32
        actual_evaporation_m3[i] = evap_vol

        # Update reach storage (mass balance) and waterbody storage.
        inflow_vol = upstream_inflow_m3_s * dt_f32 + node_sideflow
        outflow_vol = kinematic_discharge * dt_f32
        river_storage_m3[i] += (
            np.float64(inflow_vol) - np.float64(outflow_vol) - np.float64(evap_vol)
        )

        wb_ds_id = kin_ds_waterbody_id[i]
        if wb_ds_id != -1:
            waterbody_storage_m3[wb_ds_id] += np.float64(outflow_vol)
            if outflow_vol > np.float32(0.0):
                waterbody_inflow_m3[wb_ds_id] += outflow_vol

        if river_storage_m3[i] < np.float64(0.0):
            over_abstraction_m3[i] += np.float32(-river_storage_m3[i])
            river_storage_m3[i] = np.float64(0.0)

        if __debug__:
            if not np.isfinite(updated_discharge_m3_s[i]):
                raise ValueError(
                    "Non-finite discharge computed in kinematic wave reach."
                )

        # Push outflow into downstream reach for subsequent steps in the topological sweep.
        ds_reach = kin_ds_reach[i]
        if ds_reach != -1:
            kin_inflow_m3_s[ds_reach] += kinematic_discharge


# 2 versions of _run_kinematic_step: one serial, one parallel
_run_kinematic_step_serial = njit(parallel=False, cache=True)(_run_kinematic_step)
_run_kinematic_step_parallel = njit(parallel=True, cache=True)(_run_kinematic_step)


@njit(cache=True)
def _run_inertial_routing_step(
    dt_f32: np.float32,
    inv_dt_f32: np.float32,
    n_inertial: int,
    inertial_up_kin_offsets: ArrayInt32,
    inertial_up_kin_indices: ArrayInt32,
    inertial_up_kin_reach_idx: ArrayInt32,
    inertial_up_offsets: ArrayInt32,
    inertial_up_indices: ArrayInt32,
    inertial_up_reach_idx: ArrayInt32,
    previous_discharge_m3_s_inertial: ArrayFloat32,
    updated_discharge_m3_s: ArrayFloat32,
    river_storage_m3_inertial: ArrayFloat64,
    sideflow_m3_inertial: ArrayFloat32,
    geom_inbank: TwoDArrayFloat32,
    geom_overbank: TwoDArrayFloat32,
    geom_cfl: TwoDArrayFloat32,
    ds_boundary_type: ArrayInt32,
    ds_inertial_k: ArrayInt32,
    ds_stage_idx: ArrayInt32,
    ds_bed_elevation: ArrayFloat32,
    kin_ds_slope: ArrayFloat32,
    pit_slope_inertial: ArrayFloat32,
    min_dt_buf: ArrayFloat32,
    evaporation_m3_inertial: ArrayFloat32,
    over_abstraction_m3_inertial: ArrayFloat32,
    actual_evaporation_m3_inertial: ArrayFloat32,
    updated_discharge_m3_s_inertial: ArrayFloat32,
    kinematic_inflow_rate_inertial: ArrayFloat32,
    discharge_vol_sum_inertial: ArrayFloat32,
    wb_outflow_substep_m3_buf: ArrayFloat64,
    retention_inflow_limit_sub_buf: ArrayFloat64,
    retention_max_outflow_limit_sub_buf: ArrayFloat64,
    wb_outflow_avail_buf: ArrayFloat64,
    wb_extra_lateral_accum_m3: ArrayFloat64,
    substep_discharge_inertial: ArrayFloat32,
    wb_lake_area: ArrayFloat32,
    wb_lake_factor: ArrayFloat32,
    wb_outflow_height: ArrayFloat32,
    wb_outflow_bed_elev: ArrayFloat32,
    stage_buf: ArrayFloat32,
    inertial_retention_k: ArrayInt32,
    inertial_retention_id: ArrayInt32,
    retention_storage_m3: ArrayFloat32,
    retention_max_storage_m3: ArrayFloat32,
    controlled_retention: ArrayBool,
    retention_activation_threshold_m3_s: ArrayFloat32,
    retention_basin_release_threshold_factor: np.float32,
    waterbody_inflow_m3: ArrayFloat32,
    retention_inflow_m3: ArrayFloat32,
    retention_outflow_m3: ArrayFloat32,
    waterbody_storage_m3: ArrayFloat64,
    outflow_per_waterbody_m3: ArrayFloat32,
    inertial_wb_ds_k: ArrayInt32,
    inertial_wb_ds_id: ArrayInt32,
    lake_outflow_target_k: ArrayInt32,
    lake_outflow_wb_id: ArrayInt32,
    wb_terminal_wb_ids: ArrayInt32,
    rev_demand_buf: ArrayFloat32,
    wb_release_volume_substep: ArrayFloat32,
    inertial_topo_order: ArrayInt32,
    total_flow_rate_buf: ArrayFloat32,
) -> tuple[int, np.float32]:
    """Routes 1D local inertial substeps and resolves lake and retention couplings.

    Args:
        dt_f32: Macro-timestep duration (seconds).
        inv_dt_f32: Inverse macro-timestep (1/s).
        n_inertial: Number of local inertial reaches.
        inertial_up_kin_offsets: CSR offsets of kinematic waves feeding inertial reaches.
        inertial_up_kin_indices: CSR reach indices of upstream kinematic nodes.
        inertial_up_kin_reach_idx: Optimized upstream kinematic reach map.
        inertial_up_offsets: CSR offsets of inertial reaches upstream.
        inertial_up_indices: CSR indices of inertial reaches upstream.
        inertial_up_reach_idx: Direct upstream index for single-parent reaches (-1 if none, negative if multi).
        previous_discharge_m3_s_inertial: Previous substep discharge for inertial reaches (m³/s).
        updated_discharge_m3_s: Discharge array updated by kinematic routing (m³/s).
        river_storage_m3_inertial: Reach storage for inertial reaches (m³).
        sideflow_m3_inertial: Sideflow volume per reach for macro-timestep (m³).
        geom_inbank: Static in-bank channel geometry lookup table.
        geom_overbank: Static compound overbank geometry lookup table.
        geom_cfl: Static CFL stability parameters table.
        ds_boundary_type: Downstream boundary condition type flags.
        ds_inertial_k: Downstream inertial reach indices.
        ds_stage_idx: Downstream stage buffer lookup index.
        ds_bed_elevation: Downstream boundary bed elevation (meters).
        kin_ds_slope: Downstream bed slope for kinematic wave connections.
        pit_slope_inertial: Terminal pit bed slope.
        min_dt_buf: Scratch buffer for adaptive CFL timestep calculation.
        evaporation_m3_inertial: Potential evaporation volume for macro-timestep (m³).
        over_abstraction_m3_inertial: Output over-abstraction deficit volume (m³).
        actual_evaporation_m3_inertial: Output actual evaporation volume (m³).
        updated_discharge_m3_s_inertial: Output time-averaged discharge (m³/s).
        kinematic_inflow_rate_inertial: Inflow rate from upstream kinematic reaches (m³/s).
        discharge_vol_sum_inertial: Accumulator for volume integrated across substeps (m³).
        wb_outflow_substep_m3_buf: Substep waterbody outflow volume buffer (m³).
        retention_inflow_limit_sub_buf: Substep retention inflow volume limit buffer (m³).
        retention_max_outflow_limit_sub_buf: Substep retention outflow volume limit buffer (m³).
        wb_outflow_avail_buf: Available waterbody outflow volume buffer (m³).
        wb_extra_lateral_accum_m3: Accumulator for extra lateral lake volume (m³).
        substep_discharge_inertial: Substep discharge state vector (m³/s).
        wb_lake_area: Waterbody surface area (m²).
        wb_lake_factor: Lake weir/rating multiplier.
        wb_outflow_height: Weir crest height above lake bed (meters).
        wb_outflow_bed_elev: Bed elevation at waterbody outflow outlet (meters).
        stage_buf: Unified stage buffer for reaches and connected waterbodies (meters).
        inertial_retention_k: Inertial reach indices connected to retention basins.
        inertial_retention_id: Retention basin IDs connected to inertial reaches.
        retention_storage_m3: Retention basin current storage volume (m³).
        retention_max_storage_m3: Retention basin capacity (m³).
        controlled_retention: Boolean flag per retention basin for controlled diversion.
        retention_activation_threshold_m3_s: Discharge threshold triggering diversion (m³/s).
        retention_basin_release_threshold_factor: Capacity fraction triggering controlled release.
        waterbody_inflow_m3: Waterbody cumulative inflow volume (m³).
        retention_inflow_m3: Retention basin cumulative inflow volume (m³).
        retention_outflow_m3: Retention basin cumulative outflow volume (m³).
        waterbody_storage_m3: Waterbody storage volume (m³).
        outflow_per_waterbody_m3: Prescribed outflow volume per waterbody (m³).
        inertial_wb_ds_k: Inertial reach indices draining into a waterbody.
        inertial_wb_ds_id: Waterbody IDs receiving inflow from inertial reaches.
        lake_outflow_target_k: Inertial reach indices receiving waterbody outflow.
        lake_outflow_wb_id: Waterbody IDs discharging into inertial reaches.
        wb_terminal_wb_ids: Waterbody IDs that discharge directly out of model domain.
        rev_demand_buf: Scratch buffer for reverse flow demand accumulation (m³/s).
        wb_release_volume_substep: Substep waterbody release volume buffer per reach (m³).
        inertial_topo_order: Topological traversal order of inertial reaches.
        total_flow_rate_buf: Pre-allocated workspace buffer storing peak flow rate per reach (m³/s).

    Returns:
        A tuple of (num_inertial_substeps, terminal_wb_outflow_m3):
            - num_inertial_substeps: Number of adaptive substeps executed (-1 if n_inertial == 0).
            - terminal_wb_outflow_m3: Total outflow volume exiting domain via terminal waterbodies (m³).
    """
    num_inertial_substeps: int = -1  # for return if no inertial reaches are present

    if n_inertial > 0:
        num_inertial_substeps = compute_inertial_substeps_cfl(
            dt_f32=dt_f32,
            inv_dt_f32=inv_dt_f32,
            n_inertial=n_inertial,
            inertial_up_kin_offsets=inertial_up_kin_offsets,
            inertial_up_kin_indices=inertial_up_kin_indices,
            inertial_up_kin_reach_idx=inertial_up_kin_reach_idx,
            inertial_up_offsets=inertial_up_offsets,
            inertial_up_indices=inertial_up_indices,
            inertial_up_reach_idx=inertial_up_reach_idx,
            previous_discharge_m3_s_inertial=previous_discharge_m3_s_inertial,
            updated_discharge_m3_s=updated_discharge_m3_s,
            river_storage_m3_inertial=river_storage_m3_inertial,
            sideflow_m3_inertial=sideflow_m3_inertial,
            geom_inbank=geom_inbank,
            geom_overbank=geom_overbank,
            geom_cfl=geom_cfl,
            min_dt_buf=min_dt_buf,
            kinematic_inflow_rate_inertial=kinematic_inflow_rate_inertial,
            lake_outflow_target_k=lake_outflow_target_k,
            lake_outflow_wb_id=lake_outflow_wb_id,
            outflow_per_waterbody_m3=outflow_per_waterbody_m3,
            waterbody_storage_m3=waterbody_storage_m3,
            wb_lake_area=wb_lake_area,
            wb_lake_factor=wb_lake_factor,
            wb_outflow_height=wb_outflow_height,
            wb_outflow_bed_elev=wb_outflow_bed_elev,
            inertial_topo_order=inertial_topo_order,
            total_flow_rate_buf=total_flow_rate_buf,
        )

        substep_discharge_inertial[:] = previous_discharge_m3_s_inertial[:]

        if n_inertial >= INERTIAL_PARALLEL_THRESHOLD:
            num_inertial_substeps = _run_inertial_substeps_parallel(
                num_inertial_substeps=num_inertial_substeps,
                dt_f32=dt_f32,
                n_inertial=n_inertial,
                substep_discharge_m3_s=substep_discharge_inertial,
                river_storage_m3_inertial=river_storage_m3_inertial,
                sideflow_m3_inertial=sideflow_m3_inertial,
                evaporation_m3_inertial=evaporation_m3_inertial,
                waterbody_storage_m3=waterbody_storage_m3,
                outflow_per_waterbody_m3=outflow_per_waterbody_m3,
                geom_inbank=geom_inbank,
                geom_overbank=geom_overbank,
                retention_storage_m3=retention_storage_m3,
                retention_max_storage_m3=retention_max_storage_m3,
                inertial_retention_k=inertial_retention_k,
                inertial_retention_id=inertial_retention_id,
                controlled_retention=controlled_retention,
                retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
                retention_basin_release_threshold_factor=retention_basin_release_threshold_factor,
                pit_slope_inertial=pit_slope_inertial,
                over_abstraction_m3_inertial=over_abstraction_m3_inertial,
                actual_evaporation_m3_inertial=actual_evaporation_m3_inertial,
                waterbody_inflow_m3=waterbody_inflow_m3,
                retention_inflow_m3=retention_inflow_m3,
                retention_outflow_m3=retention_outflow_m3,
                updated_discharge_m3_s_inertial=updated_discharge_m3_s_inertial,
                kinematic_inflow_rate=kinematic_inflow_rate_inertial,
                discharge_vol_sum_m3_inertial=discharge_vol_sum_inertial,
                wb_outflow_substep_m3_buf=wb_outflow_substep_m3_buf,
                retention_inflow_limit_sub_buf=retention_inflow_limit_sub_buf,
                retention_max_outflow_limit_sub_buf=retention_max_outflow_limit_sub_buf,
                wb_outflow_avail_buf=wb_outflow_avail_buf,
                wb_extra_lateral_accum_m3=wb_extra_lateral_accum_m3,
                wb_lake_area=wb_lake_area,
                wb_lake_factor=wb_lake_factor,
                wb_outflow_height=wb_outflow_height,
                wb_outflow_bed_elev=wb_outflow_bed_elev,
                stage_buf=stage_buf,
                ds_boundary_type=ds_boundary_type,
                ds_inertial_k=ds_inertial_k,
                ds_stage_idx=ds_stage_idx,
                ds_bed_elevation=ds_bed_elevation,
                kin_ds_slope=kin_ds_slope,
                inertial_up_offsets=inertial_up_offsets,
                inertial_up_indices=inertial_up_indices,
                inertial_up_reach_idx=inertial_up_reach_idx,
                inertial_wb_ds_k=inertial_wb_ds_k,
                inertial_wb_ds_id=inertial_wb_ds_id,
                lake_outflow_target_k=lake_outflow_target_k,
                lake_outflow_wb_id=lake_outflow_wb_id,
                geom_cfl=geom_cfl,
                min_dt_buf=min_dt_buf,
                rev_demand_buf=rev_demand_buf,
                wb_release_volume_substep=wb_release_volume_substep,
                inertial_topo_order=inertial_topo_order,
                total_flow_rate_buf=total_flow_rate_buf,
            )
        else:
            num_inertial_substeps = _run_inertial_substeps_serial(
                num_inertial_substeps=num_inertial_substeps,
                dt_f32=dt_f32,
                n_inertial=n_inertial,
                substep_discharge_m3_s=substep_discharge_inertial,
                river_storage_m3_inertial=river_storage_m3_inertial,
                sideflow_m3_inertial=sideflow_m3_inertial,
                evaporation_m3_inertial=evaporation_m3_inertial,
                waterbody_storage_m3=waterbody_storage_m3,
                outflow_per_waterbody_m3=outflow_per_waterbody_m3,
                geom_inbank=geom_inbank,
                geom_overbank=geom_overbank,
                retention_storage_m3=retention_storage_m3,
                retention_max_storage_m3=retention_max_storage_m3,
                inertial_retention_k=inertial_retention_k,
                inertial_retention_id=inertial_retention_id,
                controlled_retention=controlled_retention,
                retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
                retention_basin_release_threshold_factor=retention_basin_release_threshold_factor,
                pit_slope_inertial=pit_slope_inertial,
                over_abstraction_m3_inertial=over_abstraction_m3_inertial,
                actual_evaporation_m3_inertial=actual_evaporation_m3_inertial,
                waterbody_inflow_m3=waterbody_inflow_m3,
                retention_inflow_m3=retention_inflow_m3,
                retention_outflow_m3=retention_outflow_m3,
                updated_discharge_m3_s_inertial=updated_discharge_m3_s_inertial,
                kinematic_inflow_rate=kinematic_inflow_rate_inertial,
                discharge_vol_sum_m3_inertial=discharge_vol_sum_inertial,
                wb_outflow_substep_m3_buf=wb_outflow_substep_m3_buf,
                retention_inflow_limit_sub_buf=retention_inflow_limit_sub_buf,
                retention_max_outflow_limit_sub_buf=retention_max_outflow_limit_sub_buf,
                wb_outflow_avail_buf=wb_outflow_avail_buf,
                wb_extra_lateral_accum_m3=wb_extra_lateral_accum_m3,
                wb_lake_area=wb_lake_area,
                wb_lake_factor=wb_lake_factor,
                wb_outflow_height=wb_outflow_height,
                wb_outflow_bed_elev=wb_outflow_bed_elev,
                stage_buf=stage_buf,
                ds_boundary_type=ds_boundary_type,
                ds_inertial_k=ds_inertial_k,
                ds_stage_idx=ds_stage_idx,
                ds_bed_elevation=ds_bed_elevation,
                kin_ds_slope=kin_ds_slope,
                inertial_up_offsets=inertial_up_offsets,
                inertial_up_indices=inertial_up_indices,
                inertial_up_reach_idx=inertial_up_reach_idx,
                inertial_wb_ds_k=inertial_wb_ds_k,
                inertial_wb_ds_id=inertial_wb_ds_id,
                lake_outflow_target_k=lake_outflow_target_k,
                lake_outflow_wb_id=lake_outflow_wb_id,
                geom_cfl=geom_cfl,
                min_dt_buf=min_dt_buf,
                rev_demand_buf=rev_demand_buf,
                wb_release_volume_substep=wb_release_volume_substep,
                inertial_topo_order=inertial_topo_order,
                total_flow_rate_buf=total_flow_rate_buf,
            )

    # Waterbodies discharging directly out of the domain
    terminal_wb_outflow_m3: np.float32 = np.float32(0.0)
    for idx in range(len(wb_terminal_wb_ids)):
        wb_id_term: int = wb_terminal_wb_ids[idx]
        terminal_rel: np.float32 = outflow_per_waterbody_m3[wb_id_term]
        if terminal_rel > np.float32(0.0):
            rel_vol_term: float = np.float64(terminal_rel)
            waterbody_storage_m3[wb_id_term] -= rel_vol_term
            terminal_wb_outflow_m3 += terminal_rel
            outflow_per_waterbody_m3[wb_id_term] = np.float32(0.0)

    return (
        num_inertial_substeps,
        terminal_wb_outflow_m3,
    )


class LocalInertial:
    """Local inertial river network router with non-rectangular & compound floodplain geometry.

    Combines non-linear kinematic wave routing for overland / non-river cells with a
    1D local inertial dynamic routing formulation for river network reaches,
    supporting trapezoidal and compound floodplain cross-sections.
    """

    def __init__(
        self,
        dt: float | int,
        river_network: pyflwdir.FlwdirRaster,
        river_length: ArrayFloat32,
        river_width: ArrayFloat32,
        waterbody_ids: ArrayInt32,
        river_ids: ArrayInt32,
        is_waterbody_outflow: ArrayBool,
        retention_max_storage_m3: ArrayFloat32,
        retention_node_id: ArrayInt32,
        controlled_retention: ArrayBool,
        retention_basin_release_threshold_factor: float,
        bankfull_river_elevation_m: ArrayFloat32,
        manning_n: ArrayFloat32,
        use_kinematic: ArrayBool,
        rivers_gdf: gpd.GeoDataFrame,
        shape_exponent: ArrayFloat32,
        bankfull_depth_m: ArrayFloat32,
        floodplain_width_m: ArrayFloat32,
        waterbody_lake_area: ArrayFloat32,
        waterbody_lake_factor: ArrayFloat32,
        waterbody_outflow_height: ArrayFloat32,
        waterbody_outflow_bed_elev: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        in_spinup: bool,
        min_slope: float = 1e-4,
    ) -> None:
        """Initializes the LocalInertial router object.

        Args:
            dt: Model time step duration (seconds).
            river_network: pyflwdir river network raster detailing topological connections.
            river_length: Channel length per cell (meters).
            river_width: Channel width per cell (meters).
            waterbody_ids: Map of waterbody IDs per cell (-1 if absent).
            river_ids: Map of river reach IDs corresponding to the vector network.
            is_waterbody_outflow: Boolean mask locating waterbody outlets.
            retention_max_storage_m3: Maximum storage capacity per retention basin (m³).
            retention_node_id: Identifier map linking cells to retention basins.
            controlled_retention: Flag defining controlled vs uncontrolled retention dynamics.
            retention_basin_release_threshold_factor: Release threshold factor for retention basins.
            bankfull_river_elevation_m: Channel bed elevation above datum (meters).
            manning_n: Manning roughness coefficient per reach (s/m^(1/3)).
            use_kinematic: Boolean flag selecting kinematic (True) vs inertial (False) routing per cell.
            rivers_gdf: Vector GeoDataFrame containing river geometry attributes.
            shape_exponent: Power-law cross-sectional shape exponent per reach (dimensionless).
            bankfull_depth_m: Bankfull channel depth before floodplain spilling (meters).
            floodplain_width_m: Floodplain width beyond channel (meters).
            waterbody_lake_area: Surface area of each waterbody (m²).
            waterbody_lake_factor: Rating curve multiplier factor for lake outflow calculation.
            waterbody_outflow_height: Sill/outflow height above channel bed for each waterbody (meters).
            waterbody_outflow_bed_elev: Channel bed elevation at each waterbody outlet (meters).
            river_storage_alpha: Kinematic wave alpha parameter per reach (s^beta / m^(3*beta - 1)).
            river_storage_beta: Kinematic wave beta parameter per reach (dimensionless).
            min_slope: Minimum allowable slope for ocean/pit boundary boundaries (dimensionless).
            in_spinup: Whether the model is in spinup mode.

        Raises:
            KeyError: If a local inertial pit reach has a river ID not found in rivers_gdf.
        """
        assert dt > 0, "dt must be greater than 0"
        self.dt = dt
        self.in_spinup = in_spinup
        self.retention_basin_release_threshold_factor = np.float32(
            retention_basin_release_threshold_factor
        )

        mapper: ArrayInt32 = np.full(river_network.size + 1, -1, dtype=np.int32)
        indices_init: ArrayInt64 = np.arange(river_network.size, dtype=np.int32)[
            river_network.mask
        ]
        mapper[indices_init] = np.arange(indices_init.size, dtype=np.int32)

        river_network.order_cells(method="walk")
        upstream_matrix: ArrayInt32 = pyflwdir.core.upstream_matrix(
            river_network.idxs_ds,
        )

        self.idxs_up_to_downstream: ArrayInt32 = river_network.idxs_seq[::-1]
        self.upstream_matrix_from_up_to_downstream: TwoDArrayInt32 = mapper[
            upstream_matrix[self.idxs_up_to_downstream]
        ]
        self.idxs_up_to_downstream = mapper[self.idxs_up_to_downstream]

        is_pit: ArrayBool = np.zeros_like(self.idxs_up_to_downstream, dtype=bool)
        is_pit[mapper[river_network.idxs_pit]] = True

        assert is_waterbody_outflow is not None, (
            "is_waterbody_outflow must be provided if waterbody_id is provided"
        )
        assert waterbody_ids.shape == self.idxs_up_to_downstream.shape
        self.waterbody_ids: ArrayInt32 = waterbody_ids

        assert is_waterbody_outflow.shape == self.idxs_up_to_downstream.shape
        assert (
            np.bincount(
                self.waterbody_ids[self.waterbody_ids != -1],
                weights=is_waterbody_outflow[self.waterbody_ids != -1],
            )
            == 1
        ).all()

        self.river_length = river_length
        self.river_width = river_width
        self.bed_elevation = bankfull_river_elevation_m
        self.retention_max_storage_m3 = retention_max_storage_m3
        self.controlled_retention = controlled_retention
        self.shape_exponent = shape_exponent
        self.bankfull_depth = bankfull_depth_m
        self.floodplain_width = floodplain_width_m

        n_nodes_total = is_pit.size
        is_ocean_pit_orig = np.zeros(n_nodes_total, dtype=np.bool_)
        pit_slope_orig = np.zeros(n_nodes_total, dtype=np.float32)

        for node_idx in range(n_nodes_total):
            if (
                is_pit[node_idx]
                and not use_kinematic[node_idx]
                and self.waterbody_ids[node_idx] == -1
            ):
                river_id: int = int(river_ids[node_idx])
                if river_id not in rivers_gdf.index:
                    raise KeyError(
                        f"Value with index {node_idx} has river ID {river_id}, which was not found in rivers_gdf."
                    )
                row = rivers_gdf.loc[river_id]
                ds_id: int = row["downstream_ID"]
                is_ocean_pit_orig[node_idx] = bool(ds_id == -1)
                pit_slope_orig[node_idx] = np.float32(max(row["slope"], min_slope))

        unmasked_ds: ArrayInt32 = river_network.idxs_ds[indices_init]
        ds_node_orig: ArrayInt32 = mapper[unmasked_ds]
        ds_node_orig[is_pit] = -1

        has_ds: ArrayBool = ds_node_orig != -1
        is_routed_river_reach: ArrayBool = (
            has_ds
            & (self.waterbody_ids == -1)
            & (self.waterbody_ids[ds_node_orig] == -1)
        )
        assert np.all(
            self.bed_elevation[is_routed_river_reach]
            >= self.bed_elevation[ds_node_orig[is_routed_river_reach]]
        ), "Bed elevation must decrease or be flat downstream along the river network."

        n_up_nodes: int = self.upstream_matrix_from_up_to_downstream.shape[0]
        sorted_idxs: ArrayInt32 = np.empty(n_up_nodes, dtype=np.int32)

        # For computational efficiency, it is best if kinematic wave headwaters are processed first,
        # then connected kinematic reaches in downstream order. This is because kinematic wave
        # routing can be performed in parallel for disconnected networks of headwaters,
        # but must be performed sequentially for connected reaches.
        n_kin_headwater: int = 0
        n_kin_connected: int = 0
        for i in range(n_up_nodes):
            node = self.idxs_up_to_downstream[i]
            if use_kinematic[node] and waterbody_ids[node] == -1:
                if self.upstream_matrix_from_up_to_downstream[i, 0] == -1:
                    n_kin_headwater += 1
                else:
                    n_kin_connected += 1

        n_kinematic: int = n_kin_headwater + n_kin_connected

        # Order kinematic reaches: headwaters first, then connected reaches in downstream order
        hw_idx: int = 0
        conn_idx: int = n_kin_headwater
        for i in range(n_up_nodes):
            node = self.idxs_up_to_downstream[i]
            if use_kinematic[node] and waterbody_ids[node] == -1:
                if self.upstream_matrix_from_up_to_downstream[i, 0] == -1:
                    sorted_idxs[hw_idx] = node
                    hw_idx += 1
                else:
                    sorted_idxs[conn_idx] = node
                    conn_idx += 1

        # Collect all inertial reaches
        inertial_nodes_orig: list[int] = []
        for i in range(n_up_nodes):
            node = int(self.idxs_up_to_downstream[i])
            if not use_kinematic[node] and waterbody_ids[node] == -1:
                inertial_nodes_orig.append(node)

        # Order inertial reaches along continuous downstream paths for sequential memory access
        inertial_set: set[int] = set(inertial_nodes_orig)
        in_degree: dict[int, int] = {node: 0 for node in inertial_nodes_orig}
        for node in inertial_nodes_orig:
            ds = int(ds_node_orig[node])
            if ds in in_degree:
                in_degree[ds] += 1

        visited: set[int] = set()
        chained_inertial_nodes: list[int] = []
        for node in inertial_nodes_orig:
            if in_degree[node] == 0:
                curr: int = node
                while curr in inertial_set and curr not in visited:
                    visited.add(curr)
                    chained_inertial_nodes.append(curr)
                    curr = int(ds_node_orig[curr])

        for node in inertial_nodes_orig:
            if node not in visited:
                curr = node
                while curr in inertial_set and curr not in visited:
                    visited.add(curr)
                    chained_inertial_nodes.append(curr)
                    curr = int(ds_node_orig[curr])

        for k, node in enumerate(chained_inertial_nodes):
            idx: int = n_kinematic + k
            sorted_idxs[idx] = node

        n_inertial: int = len(chained_inertial_nodes)

        n_other_start: int = n_kinematic + n_inertial
        n_other: int = 0
        for i in range(n_up_nodes):
            node = self.idxs_up_to_downstream[i]
            if waterbody_ids[node] != -1:
                idx = n_other_start + n_other
                sorted_idxs[idx] = node
                n_other += 1

        node_to_orig_i: ArrayInt32 = np.empty(n_up_nodes, dtype=np.int32)
        node_to_orig_i[self.idxs_up_to_downstream] = np.arange(
            n_up_nodes, dtype=np.int32
        )
        sorted_orig_i: ArrayInt32 = node_to_orig_i[sorted_idxs]

        inv_idxs: ArrayInt32 = np.full(n_up_nodes, -1, dtype=np.int32)
        inv_idxs[sorted_idxs] = np.arange(n_up_nodes, dtype=np.int32)

        if n_inertial > 0:
            self._inertial_topo_order: ArrayInt32 = np.array(
                [inv_idxs[node] - n_kinematic for node in inertial_nodes_orig],
                dtype=np.int32,
            )
        else:
            self._inertial_topo_order: ArrayInt32 = np.empty(0, dtype=np.int32)

        self.sorted_idxs: ArrayInt32 = sorted_idxs
        self.inv_idxs: ArrayInt32 = inv_idxs
        self.n_kin_headwater: int = n_kin_headwater
        self.n_kinematic: int = n_kinematic
        self.n_inertial: int = n_inertial

        self._river_length: ArrayFloat32 = self.river_length[sorted_idxs]
        self._river_width: ArrayFloat32 = self.river_width[sorted_idxs]
        self._shape_exponent: ArrayFloat32 = self.shape_exponent[sorted_idxs]
        self._bankfull_depth: ArrayFloat32 = self.bankfull_depth[sorted_idxs]
        self._floodplain_width: ArrayFloat32 = self.floodplain_width[sorted_idxs]
        self._bed_elevation: ArrayFloat32 = self.bed_elevation[sorted_idxs]
        self._use_kinematic: ArrayBool = use_kinematic[sorted_idxs]
        self._is_pit: ArrayBool = is_pit[sorted_idxs]
        self._is_ocean_pit: ArrayBool = is_ocean_pit_orig[sorted_idxs]
        self._pit_slope: ArrayFloat32 = pit_slope_orig[sorted_idxs]
        self._waterbody_ids: ArrayInt32 = self.waterbody_ids[sorted_idxs]
        self._river_ids: ArrayInt32 = river_ids[sorted_idxs]
        self._is_waterbody_outflow: ArrayBool = is_waterbody_outflow[sorted_idxs]
        self._retention_node_id: ArrayInt32 = retention_node_id[sorted_idxs]

        manning_n_sorted: ArrayFloat32 = manning_n[sorted_idxs].astype(np.float32)
        self._manning_n_sq: ArrayFloat32 = manning_n_sorted * manning_n_sorted

        inertial_start: int = self.n_kinematic
        inertial_end: int = self.n_kinematic + self.n_inertial
        self._river_length_inertial: ArrayFloat32 = self._river_length[
            inertial_start:inertial_end
        ]
        self._river_width_inertial: ArrayFloat32 = self._river_width[
            inertial_start:inertial_end
        ]
        self._shape_exponent_inertial: ArrayFloat32 = self._shape_exponent[
            inertial_start:inertial_end
        ]
        self._bankfull_depth_inertial: ArrayFloat32 = self._bankfull_depth[
            inertial_start:inertial_end
        ]
        self._floodplain_width_inertial: ArrayFloat32 = self._floodplain_width[
            inertial_start:inertial_end
        ]
        self._bed_elevation_inertial: ArrayFloat32 = self._bed_elevation[
            inertial_start:inertial_end
        ]
        self._manning_n_sq_inertial: ArrayFloat32 = self._manning_n_sq[
            inertial_start:inertial_end
        ]

        ds_perm: ArrayInt32 = ds_node_orig[sorted_idxs]
        self._ds_node: ArrayInt32 = np.where(
            ds_perm != -1, inv_idxs[np.maximum(ds_perm, 0)], -1
        )

        up_perm: TwoDArrayInt32 = self.upstream_matrix_from_up_to_downstream[
            sorted_orig_i, :
        ]
        self._upstream_matrix: TwoDArrayInt32 = np.where(
            up_perm != -1, inv_idxs[np.maximum(up_perm, 0)], -1
        )

        self._pit_indices: ArrayInt32 = np.where(
            self._is_pit[: self.n_kinematic + self.n_inertial]
        )[0].astype(np.int32)
        self._wb_outflow_indices: ArrayInt32 = np.where(self._is_waterbody_outflow)[
            0
        ].astype(np.int32)

        n_wb: int = int(is_waterbody_outflow.sum())
        self.n_wb: int = n_wb
        n_ret: int = retention_max_storage_m3.size
        assert controlled_retention.size == n_ret, (
            f"controlled_retention size ({controlled_retention.size}) must match "
            f"retention_max_storage_m3 size ({n_ret})."
        )

        self._setup_inertial_boundary_arrays()
        self._compute_static_geometry()

        self._f64_global_workspace: np.ndarray = np.empty(
            (1, n_up_nodes), dtype=np.float64
        )
        self._f32_global_workspace: np.ndarray = np.empty(
            (6, n_up_nodes), dtype=np.float32
        )

        self._river_storage_perm: ArrayFloat64 = self._f64_global_workspace[0]

        self._discharge_prev_perm: ArrayFloat32 = self._f32_global_workspace[0]
        self._sideflow_perm: ArrayFloat32 = self._f32_global_workspace[1]
        self._evaporation_perm: ArrayFloat32 = self._f32_global_workspace[2]
        self._alpha_perm: ArrayFloat32 = river_storage_alpha[self.sorted_idxs].astype(
            np.float32
        )
        self._beta_perm: ArrayFloat32 = river_storage_beta[self.sorted_idxs].astype(
            np.float32
        )
        self._over_abstraction_perm: ArrayFloat32 = self._f32_global_workspace[3]
        self._actual_evaporation_perm: ArrayFloat32 = self._f32_global_workspace[4]
        self._updated_discharge_perm: ArrayFloat32 = self._f32_global_workspace[5]

        # Slices of the global arrays for inertial reaches
        inertial_start: int = self.n_kinematic
        inertial_end: int = self.n_kinematic + self.n_inertial
        self._river_storage_perm_inertial: ArrayFloat64 = self._river_storage_perm[
            inertial_start:inertial_end
        ]
        self._discharge_prev_perm_inertial: ArrayFloat32 = self._discharge_prev_perm[
            inertial_start:inertial_end
        ]
        self._sideflow_perm_inertial: ArrayFloat32 = self._sideflow_perm[
            inertial_start:inertial_end
        ]
        self._evaporation_perm_inertial: ArrayFloat32 = self._evaporation_perm[
            inertial_start:inertial_end
        ]
        self._over_abstraction_perm_inertial: ArrayFloat32 = (
            self._over_abstraction_perm[inertial_start:inertial_end]
        )
        self._actual_evaporation_perm_inertial: ArrayFloat32 = (
            self._actual_evaporation_perm[inertial_start:inertial_end]
        )
        self._updated_discharge_perm_inertial: ArrayFloat32 = (
            self._updated_discharge_perm[inertial_start:inertial_end]
        )

        self._f32_inertial_workspace: np.ndarray = np.empty(
            (7, n_inertial), dtype=np.float32
        )
        self._wb_extra_lateral_accum_m3: ArrayFloat64 = np.zeros(
            n_inertial, dtype=np.float64
        )

        self._substep_discharge_inertial: ArrayFloat32 = self._f32_inertial_workspace[0]
        self._kinematic_inflow_rate_inertial: ArrayFloat32 = (
            self._f32_inertial_workspace[1]
        )
        self._min_dt_buf: ArrayFloat32 = self._f32_inertial_workspace[2]
        self._discharge_vol_sum_inertial: ArrayFloat32 = self._f32_inertial_workspace[3]
        self._rev_demand_buf: ArrayFloat32 = self._f32_inertial_workspace[4]
        self._wb_release_volume_substep: ArrayFloat32 = self._f32_inertial_workspace[5]
        self._total_flow_rate_buf: ArrayFloat32 = self._f32_inertial_workspace[6]
        # Buffer holding both reach stages and lake stages
        self._stage_buf: ArrayFloat32 = np.empty(n_inertial + n_wb, dtype=np.float32)

        self._waterbody_inflow_perm: ArrayFloat32 = np.empty(n_wb, dtype=np.float32)
        self._retention_inflow_perm: ArrayFloat32 = np.empty(n_ret, dtype=np.float32)
        self._retention_outflow_perm: ArrayFloat32 = np.empty(n_ret, dtype=np.float32)

        self._wb_outflow_substep_m3_buf: ArrayFloat64 = np.empty(n_wb, dtype=np.float64)
        self._retention_inflow_limit_sub_buf: ArrayFloat64 = np.empty(
            n_ret, dtype=np.float64
        )
        self._retention_max_outflow_limit_sub_buf: ArrayFloat64 = np.empty(
            n_ret, dtype=np.float64
        )
        self._wb_outflow_avail_buf: ArrayFloat64 = np.empty(n_wb, dtype=np.float64)

        assert waterbody_lake_area.size == n_wb, (
            f"waterbody_lake_area size ({waterbody_lake_area.size}) must match number of waterbodies ({n_wb})."
        )
        assert waterbody_lake_factor.size == n_wb, (
            f"waterbody_lake_factor size ({waterbody_lake_factor.size}) must match number of waterbodies ({n_wb})."
        )
        assert waterbody_outflow_height.size == n_wb, (
            f"waterbody_outflow_height size ({waterbody_outflow_height.size}) must match number of waterbodies ({n_wb})."
        )
        assert waterbody_outflow_bed_elev.size == n_wb, (
            f"waterbody_outflow_bed_elev size ({waterbody_outflow_bed_elev.size}) must match number of waterbodies ({n_wb})."
        )

        self._wb_lake_area: ArrayFloat32 = waterbody_lake_area.astype(np.float32)
        self._wb_lake_factor: ArrayFloat32 = waterbody_lake_factor.astype(np.float32)
        self._wb_outflow_height: ArrayFloat32 = waterbody_outflow_height.astype(
            np.float32
        )
        self._wb_outflow_bed_elev: ArrayFloat32 = waterbody_outflow_bed_elev.astype(
            np.float32
        )

        self._discharge_out: ArrayFloat32 = np.empty(n_up_nodes, dtype=np.float32)
        self._actual_evap_out: ArrayFloat32 = np.empty(n_up_nodes, dtype=np.float32)
        self._over_abs_out: ArrayFloat32 = np.empty(n_up_nodes, dtype=np.float32)

    def initialize_stage(
        self,
        water_stage_m: ArrayFloat32 | None = None,
        river_storage_m3: ArrayFloat64 | None = None,
        waterbody_storage_m3: ArrayFloat64 | None = None,
        already_permuted: bool = False,
        in_spinup: bool | None = None,
    ) -> None:
        """Initialize the persistent stage buffer from water stage or storage.

        Args:
            water_stage_m: Optional water surface elevation array in model grid ordering (meters).
            river_storage_m3: Optional river reach storage array (m³).
            waterbody_storage_m3: Optional waterbody storage array (m³).
            already_permuted: Whether input arrays are already in topological ordering.
            in_spinup: Optional flag indicating spinup mode; if None, defaults to self.in_spinup.
        """
        inertial_start: int = self.n_kinematic

        if river_storage_m3 is not None:
            if already_permuted:
                self._river_storage_perm[:] = river_storage_m3
            else:
                self._river_storage_perm[:] = river_storage_m3[self.sorted_idxs]
        else:
            self._river_storage_perm[:] = 0.0

        if water_stage_m is not None:
            if already_permuted:
                for k in range(self.n_inertial):
                    self._stage_buf[k] = water_stage_m[inertial_start + k]
            else:
                for k in range(self.n_inertial):
                    node: int = int(self.sorted_idxs[inertial_start + k])
                    self._stage_buf[k] = water_stage_m[node]
        else:
            # Compute stage directly from storage to ensure strict numerical consistency with geometry
            _compute_stage_from_storage_inertial(
                self.n_inertial,
                self._river_storage_perm_inertial,
                self._geom_inbank,
                self._geom_overbank,
                self._stage_buf,
            )

        if self.n_wb > 0:
            assert waterbody_storage_m3 is not None, (
                "waterbody_storage_m3 must be provided when n_wb > 0"
            )
            for wb_id in range(self.n_wb):
                lake_area: np.float32 = self._wb_lake_area[wb_id]
                depth_from_bottom: np.float32 = (
                    np.float32(waterbody_storage_m3[wb_id]) / lake_area
                )
                bottom_elevation: np.float32 = (
                    self._wb_outflow_bed_elev[wb_id] - self._wb_outflow_height[wb_id]
                )
                self._stage_buf[self.n_inertial + wb_id] = (
                    bottom_elevation + depth_from_bottom
                )

        check_spinup: bool = self.in_spinup if in_spinup is None else in_spinup
        if check_spinup and self.n_inertial > 0:
            internal_inertial_mask: ArrayBool = self._ds_boundary_type == 0
            if np.any(internal_inertial_mask):
                k_up: ArrayInt32 = np.where(internal_inertial_mask)[0].astype(np.int32)
                k_ds: ArrayInt32 = self._ds_inertial_k[k_up]

                stage_diff: ArrayFloat32 = self._stage_buf[k_ds] - self._stage_buf[k_up]
                adverse_mask: ArrayBool = stage_diff > np.float32(0.0)

                assert not np.any(adverse_mask), (
                    f"Found {np.sum(adverse_mask)} internal reaches where initial downstream stage exceeds upstream stage. "
                    f"Max adverse difference: {np.max(stage_diff):.2f} m. "
                    f"Worst node index: {self.sorted_idxs[inertial_start + k_up[np.argmax(stage_diff)]]}"
                )

    def get_water_stage(
        self,
        out: ArrayFloat32 | None = None,
    ) -> ArrayFloat32:
        """Returns the current water surface elevation for all grid reaches.

        Args:
            out: Optional pre-allocated output array in grid order (meters).

        Returns:
            Water surface elevation array in model grid ordering (meters).
        """
        n_cells: int = len(self.river_length)
        if out is None:
            stage_out: ArrayFloat32 = np.empty(n_cells, dtype=np.float32)
        else:
            stage_out = out

        inertial_start: int = self.n_kinematic
        for k in range(self.n_inertial):
            node: int = int(self.sorted_idxs[inertial_start + k])
            stage_out[node] = self._stage_buf[k]

        for i in range(self.n_kinematic):
            node = int(self.sorted_idxs[i])
            stage_out[node] = self.bed_elevation[node]

        for i in range(self.n_kinematic + self.n_inertial, len(self.sorted_idxs)):
            node = int(self.sorted_idxs[i])
            waterbody_id: int = int(self.waterbody_ids[node])
            if waterbody_id != -1 and waterbody_id < self.n_wb:
                stage_out[node] = self._stage_buf[self.n_inertial + waterbody_id]
            else:
                stage_out[node] = self.bed_elevation[node]

        return stage_out

    def _setup_inertial_boundary_arrays(self) -> None:
        """Precomputes downstream boundary classifications and lookups for all inertial reaches.

        Raises:
            ValueError: If a reach is neither a pit nor connected to a downstream reach.
        """
        n_inertial: int = self.n_inertial
        n_kinematic: int = self.n_kinematic
        ds_boundary_type: ArrayInt32 = np.empty(n_inertial, dtype=np.int32)
        ds_inertial_k: ArrayInt32 = np.full(n_inertial, -1, dtype=np.int32)
        ds_waterbody_id: ArrayInt32 = np.full(n_inertial, -1, dtype=np.int32)
        ds_bed_elevation: ArrayFloat32 = np.empty(n_inertial, dtype=np.float32)
        kin_ds_slope: ArrayFloat32 = np.zeros(n_inertial, dtype=np.float32)

        for k in range(n_inertial):
            i: int = n_kinematic + k
            ds: int = int(self._ds_node[i])
            bed_node: float = float(self._bed_elevation[i])
            r_len: float = max(float(self._river_length[i]), 1.0)

            if self._is_pit[i]:
                if self._is_ocean_pit[i]:
                    ds_boundary_type[k] = 4
                    ds_bed_elevation[k] = np.float32(bed_node)
                else:
                    ds_boundary_type[k] = 3
                    eff_slope: float = float(self._pit_slope[i])
                    ds_bed_elevation[k] = np.float32(bed_node - eff_slope * r_len)
            elif ds != -1:
                bed_ds: float = float(self._bed_elevation[ds])
                ds_bed_elevation[k] = np.float32(bed_ds)
                wb_ds: int = int(self._waterbody_ids[ds])
                if wb_ds != -1:
                    ds_boundary_type[k] = 2
                    ds_waterbody_id[k] = wb_ds
                elif self._use_kinematic[ds]:
                    ds_boundary_type[k] = 1
                    slope_val: float = max((bed_node - bed_ds) / r_len, 1e-4)
                    kin_ds_slope[k] = np.float32(slope_val)
                else:
                    ds_boundary_type[k] = 0
                    ds_k: int = ds - n_kinematic
                    ds_inertial_k[k] = ds_k
            else:
                raise ValueError(
                    f"Invalid river topology at sorted index {i}: reach is neither a pit nor connected to a downstream node."
                )

        self._ds_boundary_type: ArrayInt32 = ds_boundary_type
        self._ds_inertial_k: ArrayInt32 = ds_inertial_k
        self._ds_waterbody_id: ArrayInt32 = ds_waterbody_id
        self._ds_bed_elevation: ArrayFloat32 = ds_bed_elevation
        self._interface_bed_elev_max: ArrayFloat32 = np.maximum(
            self._bed_elevation_inertial, self._ds_bed_elevation
        )

        inv_dx_interface: ArrayFloat32 = np.empty(n_inertial, dtype=np.float32)
        for k in range(n_inertial):
            if ds_boundary_type[k] == 0:
                ds_k: int = ds_inertial_k[k]
                dx: np.float32 = np.float32(0.5) * (
                    self._river_length_inertial[k] + self._river_length_inertial[ds_k]
                )
            else:
                dx = self._river_length_inertial[k]
            inv_dx_interface[k] = np.float32(1.0) / max(dx, np.float32(1.0))
        self._inv_dx_interface: ArrayFloat32 = inv_dx_interface

        # Downstream stage buffer index for each reach
        ds_stage_idx: ArrayInt32 = np.full(n_inertial, -1, dtype=np.int32)
        for k in range(n_inertial):
            if ds_boundary_type[k] == 0:
                ds_stage_idx[k] = ds_inertial_k[k]
            elif ds_boundary_type[k] == 2:
                ds_stage_idx[k] = n_inertial + ds_waterbody_id[k]
        self._ds_stage_idx: ArrayInt32 = ds_stage_idx
        self._kin_ds_slope: ArrayFloat32 = kin_ds_slope

        # Upstream reach connectivity for inertial reaches
        max_up: int = self._upstream_matrix.shape[1]
        inertial_up_offsets: list[int] = [0]
        inertial_up_indices: list[int] = []
        inertial_up_reach_idx: ArrayInt32 = np.full(n_inertial, -1, dtype=np.int32)
        for k in range(n_inertial):
            i = n_kinematic + k
            start_count: int = len(inertial_up_indices)
            for j in range(max_up):
                up_node: int = int(self._upstream_matrix[i, j])
                if up_node == -1:
                    break
                if (
                    self._waterbody_ids[up_node] == -1
                    and not self._use_kinematic[up_node]
                ):
                    inertial_up_indices.append(up_node - n_kinematic)
            cnt: int = len(inertial_up_indices) - start_count
            if cnt == 1:
                inertial_up_reach_idx[k] = inertial_up_indices[start_count]
            elif cnt > 1:
                inertial_up_reach_idx[k] = -(start_count + 2)
            inertial_up_offsets.append(len(inertial_up_indices))

        self._inertial_up_offsets: ArrayInt32 = np.array(
            inertial_up_offsets, dtype=np.int32
        )
        self._inertial_up_indices: ArrayInt32 = np.array(
            inertial_up_indices, dtype=np.int32
        )
        self._inertial_up_reach_idx: ArrayInt32 = inertial_up_reach_idx

        # Retention basin connections for inertial reaches
        ret_inertial_k: list[int] = []
        ret_inertial_id: list[int] = []
        for k in range(n_inertial):
            ret_id: int = int(self._retention_node_id[n_kinematic + k])
            if ret_id != -1:
                ret_inertial_k.append(k)
                ret_inertial_id.append(ret_id)
        self._inertial_retention_k: ArrayInt32 = np.array(
            ret_inertial_k, dtype=np.int32
        )
        self._inertial_retention_id: ArrayInt32 = np.array(
            ret_inertial_id, dtype=np.int32
        )

        # Downstream waterbody links
        wb_ds_k: list[int] = []
        wb_ds_id: list[int] = []
        for k in range(n_inertial):
            if ds_boundary_type[k] == 2:
                wb_ds_k.append(k)
                wb_ds_id.append(int(ds_waterbody_id[k]))
        self._inertial_wb_ds_k: ArrayInt32 = np.array(wb_ds_k, dtype=np.int32)
        self._inertial_wb_ds_id: ArrayInt32 = np.array(wb_ds_id, dtype=np.int32)

        # Waterbody outlets to receiving inertial reaches
        lake_outflow_targets: list[int] = []
        lake_outflow_wbs: list[int] = []
        for k in range(n_inertial):
            i = n_kinematic + k
            for j in range(max_up):
                up_node = int(self._upstream_matrix[i, j])
                if up_node == -1:
                    break
                if self._is_waterbody_outflow[up_node]:
                    wb_id_val: int = int(self._waterbody_ids[up_node])
                    if wb_id_val != -1:
                        lake_outflow_targets.append(k)
                        lake_outflow_wbs.append(wb_id_val)

        self._lake_outflow_target_k: ArrayInt32 = np.array(
            lake_outflow_targets, dtype=np.int32
        )
        self._lake_outflow_wb_id: ArrayInt32 = np.array(
            lake_outflow_wbs, dtype=np.int32
        )

        # Waterbody-to-waterbody routing links
        wb_to_wb_src: list[int] = []
        wb_to_wb_tgt: list[int] = []
        for idx in range(len(self._wb_outflow_indices)):
            wb_node: int = int(self._wb_outflow_indices[idx])
            wb_id: int = int(self._waterbody_ids[wb_node])
            ds: int = int(self._ds_node[wb_node])
            if ds != -1:
                target_wb: int = int(self._waterbody_ids[ds])
                if target_wb != -1 and target_wb != wb_id:
                    wb_to_wb_src.append(wb_id)
                    wb_to_wb_tgt.append(target_wb)
        self._wb_to_wb_src_wb: ArrayInt32 = np.array(wb_to_wb_src, dtype=np.int32)
        self._wb_to_wb_tgt_wb: ArrayInt32 = np.array(wb_to_wb_tgt, dtype=np.int32)

        # Waterbody outflows into receiving kinematic reaches
        wb_to_kin_targets: list[int] = []
        wb_to_kin_wbs: list[int] = []
        for idx in range(len(self._wb_outflow_indices)):
            wb_node = int(self._wb_outflow_indices[idx])
            wb_id = int(self._waterbody_ids[wb_node])
            ds = int(self._ds_node[wb_node])
            if 0 <= ds < n_kinematic:
                wb_to_kin_targets.append(ds)
                wb_to_kin_wbs.append(wb_id)
        self._wb_to_kin_target_reach: ArrayInt32 = np.array(
            wb_to_kin_targets, dtype=np.int32
        )
        self._wb_to_kin_wb_id: ArrayInt32 = np.array(wb_to_kin_wbs, dtype=np.int32)
        self._kin_wb_inflow_m3: ArrayFloat32 = np.zeros(n_kinematic, dtype=np.float32)

        # Waterbodies discharging directly out of the domain
        wb_term_ids: list[int] = []
        for idx in range(len(self._wb_outflow_indices)):
            wb_node = int(self._wb_outflow_indices[idx])
            if self._ds_node[wb_node] == -1 or self._is_pit[wb_node]:
                wb_id_term: int = int(self._waterbody_ids[wb_node])
                if wb_id_term != -1:
                    wb_term_ids.append(wb_id_term)
        self._wb_terminal_wb_ids: ArrayInt32 = np.array(wb_term_ids, dtype=np.int32)

        # Downstream waterbodies for kinematic reaches
        kin_ds_wb_id: ArrayInt32 = np.full(n_kinematic, -1, dtype=np.int32)
        for i in range(n_kinematic):
            ds = int(self._ds_node[i])
            if ds != -1 and self._waterbody_ids[ds] != -1:
                kin_ds_wb_id[i] = self._waterbody_ids[ds]
        self._kin_ds_waterbody_id: ArrayInt32 = kin_ds_wb_id

        # Downstream targets for kinematic reaches
        kin_ds_reach: ArrayInt32 = np.full(n_kinematic, -1, dtype=np.int32)
        for i in range(n_kinematic):
            ds = int(self._ds_node[i])
            if 0 <= ds < n_kinematic:
                kin_ds_reach[i] = ds
        self._kin_ds_reach: ArrayInt32 = kin_ds_reach
        self._kin_inflow_m3_s: ArrayFloat32 = np.zeros(n_kinematic, dtype=np.float32)

        # Upstream kinematic inflows into inertial reaches
        inertial_kin_offsets: list[int] = [0]
        inertial_kin_indices: list[int] = []
        inertial_up_kin_reach_idx: ArrayInt32 = np.full(n_inertial, -1, dtype=np.int32)
        for k in range(n_inertial):
            i = n_kinematic + k
            start_count = len(inertial_kin_indices)
            for j in range(max_up):
                up_node = int(self._upstream_matrix[i, j])
                if up_node == -1:
                    break
                if (
                    not self._is_waterbody_outflow[up_node]
                    and self._waterbody_ids[up_node] == -1
                    and self._use_kinematic[up_node]
                ):
                    inertial_kin_indices.append(up_node)
            cnt = len(inertial_kin_indices) - start_count
            if cnt == 1:
                inertial_up_kin_reach_idx[k] = inertial_kin_indices[start_count]
            elif cnt > 1:
                inertial_up_kin_reach_idx[k] = -(start_count + 2)
            inertial_kin_offsets.append(len(inertial_kin_indices))
        self._inertial_up_kin_offsets: ArrayInt32 = np.array(
            inertial_kin_offsets, dtype=np.int32
        )
        self._inertial_up_kin_indices: ArrayInt32 = np.array(
            inertial_kin_indices, dtype=np.int32
        )
        self._inertial_up_kin_reach_idx: ArrayInt32 = inertial_up_kin_reach_idx

        inertial_up_count: ArrayInt32 = np.zeros(n_inertial, dtype=np.int32)
        for k in range(n_inertial):
            n_in: int = (
                1
                if self._inertial_up_reach_idx[k] >= 0
                else (
                    self._inertial_up_offsets[k + 1]
                    - (-self._inertial_up_reach_idx[k] - 2)
                    if self._inertial_up_reach_idx[k] < -1
                    else 0
                )
            )
            n_kin: int = (
                1
                if self._inertial_up_kin_reach_idx[k] >= 0
                else (
                    self._inertial_up_kin_offsets[k + 1]
                    - (-self._inertial_up_kin_reach_idx[k] - 2)
                    if self._inertial_up_kin_reach_idx[k] < -1
                    else 0
                )
            )
            inertial_up_count[k] = n_in + n_kin
        self._inertial_up_count: ArrayInt32 = inertial_up_count

        # Boundary outflow reaches for mass balance checks
        inertial_boundary_outflow_k: list[int] = [
            k for k in range(n_inertial) if self._ds_boundary_type[k] != 0
        ]
        self._inertial_boundary_outflow_k: ArrayInt32 = np.array(
            inertial_boundary_outflow_k, dtype=np.int32
        )

    def _compute_static_geometry(self) -> None:
        """Computes and stores reach-static cross-sectional geometric arrays."""
        inertial_start: int = self.n_kinematic
        inertial_end: int = self.n_kinematic + self.n_inertial
        self._river_length_inertial: ArrayFloat32 = self._river_length[
            inertial_start:inertial_end
        ]
        self._river_width_inertial: ArrayFloat32 = self._river_width[
            inertial_start:inertial_end
        ]
        self._shape_exponent_inertial: ArrayFloat32 = self._shape_exponent[
            inertial_start:inertial_end
        ]
        self._bankfull_depth_inertial: ArrayFloat32 = self._bankfull_depth[
            inertial_start:inertial_end
        ]
        self._floodplain_width_inertial: ArrayFloat32 = self._floodplain_width[
            inertial_start:inertial_end
        ]
        self._bed_elevation_inertial: ArrayFloat32 = self._bed_elevation[
            inertial_start:inertial_end
        ]
        self._manning_n_sq_inertial: ArrayFloat32 = self._manning_n_sq[
            inertial_start:inertial_end
        ]

        (
            self._inverse_reach_length_inertial,
            self._inverse_shape_exponent_plus_one_inertial,
            self._bankfull_area_inertial,
            self._bankfull_volume_inertial,
            self._bankfull_perimeter_inertial,
            self._floodplain_side_slope_inertial,
            self._floodplain_depth_threshold_inertial,
            self._floodplain_area_threshold_inertial,
            self._sqrt_one_plus_floodplain_slope_squared_inertial,
            self._width_over_sqrt_bankfull_depth_inertial,
        ) = compute_static_geometry(
            river_length=self._river_length_inertial,
            river_width=self._river_width_inertial,
            shape_exponent=self._shape_exponent_inertial,
            bankfull_depth=self._bankfull_depth_inertial,
            floodplain_width=self._floodplain_width_inertial,
        )

        n_nodes: int = len(self._river_length)
        self._inverse_reach_length: ArrayFloat32 = np.zeros(n_nodes, dtype=np.float32)
        self._inverse_shape_exponent_plus_one: ArrayFloat32 = np.zeros(
            n_nodes, dtype=np.float32
        )
        self._bankfull_area: ArrayFloat32 = np.zeros(n_nodes, dtype=np.float32)
        self._bankfull_volume: ArrayFloat32 = np.zeros(n_nodes, dtype=np.float32)
        self._bankfull_perimeter: ArrayFloat32 = np.zeros(n_nodes, dtype=np.float32)
        self._floodplain_side_slope: ArrayFloat32 = np.zeros(n_nodes, dtype=np.float32)
        self._floodplain_depth_threshold: ArrayFloat32 = np.zeros(
            n_nodes, dtype=np.float32
        )
        self._floodplain_area_threshold: ArrayFloat32 = np.zeros(
            n_nodes, dtype=np.float32
        )
        self._sqrt_one_plus_floodplain_slope_squared: ArrayFloat32 = np.zeros(
            n_nodes, dtype=np.float32
        )
        self._width_over_sqrt_bankfull_depth: ArrayFloat32 = np.zeros(
            n_nodes, dtype=np.float32
        )

        self._inverse_reach_length[inertial_start:inertial_end] = (
            self._inverse_reach_length_inertial
        )
        self._inverse_shape_exponent_plus_one[inertial_start:inertial_end] = (
            self._inverse_shape_exponent_plus_one_inertial
        )
        self._bankfull_area[inertial_start:inertial_end] = self._bankfull_area_inertial
        self._bankfull_volume[inertial_start:inertial_end] = (
            self._bankfull_volume_inertial
        )
        self._bankfull_perimeter[inertial_start:inertial_end] = (
            self._bankfull_perimeter_inertial
        )
        self._floodplain_side_slope[inertial_start:inertial_end] = (
            self._floodplain_side_slope_inertial
        )
        self._floodplain_depth_threshold[inertial_start:inertial_end] = (
            self._floodplain_depth_threshold_inertial
        )
        self._floodplain_area_threshold[inertial_start:inertial_end] = (
            self._floodplain_area_threshold_inertial
        )
        self._sqrt_one_plus_floodplain_slope_squared[inertial_start:inertial_end] = (
            self._sqrt_one_plus_floodplain_slope_squared_inertial
        )
        self._width_over_sqrt_bankfull_depth[inertial_start:inertial_end] = (
            self._width_over_sqrt_bankfull_depth_inertial
        )

        self._stage_vol_coeff_inertial: ArrayFloat32 = (
            self._bankfull_depth_inertial
            / (
                self._bankfull_volume_inertial
                ** self._inverse_shape_exponent_plus_one_inertial
            )
        ).astype(np.float32)
        self._pit_slope_inertial: ArrayFloat32 = self._pit_slope[
            inertial_start:inertial_end
        ]
        self._retention_node_id_inertial: ArrayInt32 = self._retention_node_id[
            inertial_start:inertial_end
        ]

        # Precompute in-bank channel geometry
        self._geom_inbank: TwoDArrayFloat32 = np.empty(
            (self.n_inertial, GEOM_IN_NUM_COLS), dtype=np.float32
        )
        self._geom_inbank[:, GEOM_IN_INVERSE_LENGTH] = (
            self._inverse_reach_length_inertial
        )
        self._geom_inbank[:, GEOM_IN_BED_ELEVATION] = self._bed_elevation_inertial
        self._geom_inbank[:, GEOM_IN_BANKFULL_DEPTH] = self._bankfull_depth_inertial
        self._geom_inbank[:, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE] = (
            self._inverse_shape_exponent_plus_one_inertial
        )
        self._geom_inbank[:, GEOM_IN_BANKFULL_VOLUME] = self._bankfull_volume_inertial
        self._geom_inbank[:, GEOM_IN_STAGE_VOLUME_COEFFICIENT] = (
            self._stage_vol_coeff_inertial
        )
        self._geom_inbank[:, GEOM_IN_INTERFACE_BED_ELEVATION_MAX] = (
            self._interface_bed_elev_max
        )
        self._geom_inbank[:, GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH] = (
            self._width_over_sqrt_bankfull_depth_inertial
        )
        self._geom_inbank[:, GEOM_IN_MANNING_N_SQUARED] = self._manning_n_sq_inertial
        self._geom_inbank[:, GEOM_IN_INVERSE_INTERFACE_LENGTH] = self._inv_dx_interface

        # Precompute overbank floodplain geometry
        self._geom_overbank: TwoDArrayFloat32 = np.empty(
            (self.n_inertial, GEOM_OV_NUM_COLS), dtype=np.float32
        )
        self._geom_overbank[:, GEOM_OV_FLOODPLAIN_WIDTH] = (
            self._floodplain_width_inertial
        )
        self._geom_overbank[:, GEOM_OV_RIVER_WIDTH] = self._river_width_inertial
        self._geom_overbank[:, GEOM_OV_BANKFULL_PERIMETER] = (
            self._bankfull_perimeter_inertial
        )
        self._geom_overbank[:, GEOM_OV_FLOODPLAIN_SIDE_SLOPE] = (
            self._floodplain_side_slope_inertial
        )
        self._geom_overbank[:, GEOM_OV_FLOODPLAIN_DEPTH_THRESHOLD] = (
            self._floodplain_depth_threshold_inertial
        )
        self._geom_overbank[:, GEOM_OV_FLOODPLAIN_AREA_THRESHOLD] = (
            self._floodplain_area_threshold_inertial
        )
        self._geom_overbank[:, GEOM_OV_SQRT_ONE_PLUS_FLOODPLAIN_SLOPE_SQUARED] = (
            self._sqrt_one_plus_floodplain_slope_squared_inertial
        )

        # Precompute CFL stability parameters
        self._geom_cfl: TwoDArrayFloat32 = np.empty(
            (self.n_inertial, GEOM_CFL_NUM_COLS), dtype=np.float32
        )
        bed_drop: ArrayFloat32 = np.maximum(
            self._bed_elevation_inertial - self._ds_bed_elevation,
            np.float32(0.0),
        )
        eff_slope: ArrayFloat32 = np.maximum(
            bed_drop * self._inverse_reach_length_inertial, np.float32(1e-4)
        )
        manning_n: ArrayFloat32 = np.sqrt(self._manning_n_sq_inertial)
        self._geom_cfl[:, GEOM_CFL_MANNING_COEFFICIENT] = (
            (manning_n / (self._river_width_inertial * np.sqrt(eff_slope)))
            ** np.float32(0.6)
        ).astype(np.float32)
        cfl_safety_factor: np.float32 = np.float32(0.7)
        sqrt_g: np.float32 = np.sqrt(np.float32(9.80665))
        cfl_dx: ArrayFloat32 = np.minimum(
            self._river_length_inertial, np.float32(1.0) / self._inv_dx_interface
        )
        confluence_factor: ArrayFloat32 = np.where(
            self._inertial_up_count > 1,
            np.float32(1.0)
            / np.sqrt(np.maximum(self._inertial_up_count, 1).astype(np.float32)),
            np.float32(1.0),
        ).astype(np.float32)
        self._geom_cfl[:, GEOM_CFL_CONSTANT] = (
            confluence_factor * cfl_safety_factor * cfl_dx / sqrt_g
        ).astype(np.float32)

    def update_channel_width(
        self,
        river_width: ArrayFloat32,
        bankfull_depth_m: ArrayFloat32 | None = None,
    ) -> None:
        """Updates channel bankfull top width and depth dynamically during simulation.

        Args:
            river_width: Updated channel bankfull top width (meters).
            bankfull_depth_m: Optional updated bankfull channel depth (meters).
        """
        self.river_width = river_width.astype(np.float32)
        self._river_width = self.river_width[self.sorted_idxs]
        if bankfull_depth_m is not None:
            self.bankfull_depth = bankfull_depth_m.astype(np.float32)
            self._bankfull_depth = self.bankfull_depth[self.sorted_idxs]
        self._compute_static_geometry()

    def calculate_river_storage_from_discharge(
        self,
        discharge: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_length: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        waterbody_id: ArrayInt32,
    ) -> ArrayFloat64:
        """Calculates storage from discharge for all river reaches.

        Args:
            discharge: River discharge array (m³/s).
            river_storage_alpha: Storage equation alpha parameter.
            river_length: Channel length (meters).
            river_storage_beta: Storage equation beta parameter.
            waterbody_id: Waterbody ID map (-1 indicates regular channel).

        Returns:
            River storage volume array (m³).
        """
        cross_sectional_area: ArrayFloat32 = (
            river_storage_alpha
            * np.maximum(discharge, np.float32(0.0)) ** river_storage_beta
        )
        river_storage: ArrayFloat64 = (cross_sectional_area * river_length).astype(
            np.float64
        )
        river_storage[waterbody_id != -1] = 0.0
        return river_storage

    def calculate_discharge_from_river_storage(
        self,
        river_storage: ArrayFloat64,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        river_length: ArrayFloat32,
        waterbody_id: ArrayInt32,
    ) -> ArrayFloat32:
        """Calculates discharge from storage for all river reaches.

        Args:
            river_storage: River storage volume array (m³).
            river_storage_alpha: Storage equation alpha parameter.
            river_storage_beta: Storage equation beta parameter.
            river_length: Channel length (meters).
            waterbody_id: Waterbody ID map (-1 indicates regular channel).

        Returns:
            River discharge array (m³/s).
        """
        discharge: ArrayFloat32 = (
            np.maximum(river_storage, 0.0) / (river_storage_alpha * river_length)
        ) ** (1 / river_storage_beta)
        discharge[waterbody_id != -1] = np.nan
        return discharge

    def get_available_storage(
        self,
        discharge: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        maximum_abstraction_ratio: float = 0.9,
    ) -> ArrayFloat64:
        """Calculates allowable volume for abstraction while ensuring environmental minimums.

        Args:
            discharge: River discharge array (m³/s).
            river_storage_alpha: Storage equation alpha parameter.
            river_storage_beta: Storage equation beta parameter.
            maximum_abstraction_ratio: Fraction of river storage accessible for abstraction.

        Returns:
            Available river storage volume array (m³).
        """
        return (
            self.get_total_storage(discharge, river_storage_alpha, river_storage_beta)
            * maximum_abstraction_ratio
        )

    def get_total_storage(
        self,
        discharge: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
    ) -> ArrayFloat64:
        """Calculates total reach storage volume.

        Args:
            discharge: River discharge array (m³/s).
            river_storage_alpha: Storage equation alpha parameter.
            river_storage_beta: Storage equation beta parameter.

        Returns:
            Total river storage volume array (m³).
        """
        total_storage: ArrayFloat64 = self.calculate_river_storage_from_discharge(
            discharge,
            river_storage_alpha,
            self.river_length,
            river_storage_beta,
            self.waterbody_ids,
        )

        assert not np.isnan(total_storage).any()
        return total_storage

    @staticmethod
    @njit(cache=True)
    def _step(
        routing_timestep_s: float | int,
        previous_discharge_m3_s: ArrayFloat32,
        river_storage_m3: ArrayFloat64,
        sideflow_m3: ArrayFloat32,
        evaporation_m3: ArrayFloat32,
        waterbody_storage_m3: ArrayFloat64,
        outflow_per_waterbody_m3: ArrayFloat32,
        kin_ds_reach: ArrayInt32,
        kin_inflow_m3_s: ArrayFloat32,
        n_kin_headwater: int,
        inertial_up_kin_offsets: ArrayInt32,
        inertial_up_kin_indices: ArrayInt32,
        inertial_up_kin_reach_idx: ArrayInt32,
        wb_to_wb_src_wb: ArrayInt32,
        wb_to_wb_tgt_wb: ArrayInt32,
        wb_to_kin_target_reach: ArrayInt32,
        wb_to_kin_wb_id: ArrayInt32,
        kin_wb_inflow_m3: ArrayFloat32,
        wb_terminal_wb_ids: ArrayInt32,
        kin_ds_waterbody_id: ArrayInt32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        river_length: ArrayFloat32,
        retention_storage_m3: ArrayFloat32,
        retention_max_storage_m3: ArrayFloat32,
        retention_node_id: ArrayInt32,
        controlled_retention: ArrayBool,
        retention_activation_threshold_m3_s: ArrayFloat32,
        retention_basin_release_threshold_factor: np.float32,
        n_kinematic: int,
        n_inertial: int,
        over_abstraction_m3: ArrayFloat32,
        actual_evaporation_m3: ArrayFloat32,
        waterbody_inflow_m3: ArrayFloat32,
        retention_inflow_m3: ArrayFloat32,
        retention_outflow_m3: ArrayFloat32,
        updated_discharge_m3_s: ArrayFloat32,
        river_storage_m3_inertial: ArrayFloat64,
        previous_discharge_m3_s_inertial: ArrayFloat32,
        sideflow_m3_inertial: ArrayFloat32,
        evaporation_m3_inertial: ArrayFloat32,
        over_abstraction_m3_inertial: ArrayFloat32,
        actual_evaporation_m3_inertial: ArrayFloat32,
        updated_discharge_m3_s_inertial: ArrayFloat32,
        kinematic_inflow_rate_inertial: ArrayFloat32,
        discharge_vol_sum_inertial: ArrayFloat32,
        wb_outflow_substep_m3_buf: ArrayFloat64,
        retention_inflow_limit_sub_buf: ArrayFloat64,
        retention_max_outflow_limit_sub_buf: ArrayFloat64,
        wb_outflow_avail_buf: ArrayFloat64,
        wb_extra_lateral_accum_m3: ArrayFloat64,
        substep_discharge_inertial: ArrayFloat32,
        wb_lake_area: ArrayFloat32,
        wb_lake_factor: ArrayFloat32,
        wb_outflow_height: ArrayFloat32,
        wb_outflow_bed_elev: ArrayFloat32,
        stage_buf: ArrayFloat32,
        ds_boundary_type: ArrayInt32,
        ds_inertial_k: ArrayInt32,
        ds_stage_idx: ArrayInt32,
        ds_bed_elevation: ArrayFloat32,
        kin_ds_slope: ArrayFloat32,
        min_dt_buf: ArrayFloat32,
        inertial_up_offsets: ArrayInt32,
        inertial_up_indices: ArrayInt32,
        inertial_up_reach_idx: ArrayInt32,
        inertial_retention_k: ArrayInt32,
        inertial_retention_id: ArrayInt32,
        inertial_wb_ds_k: ArrayInt32,
        inertial_wb_ds_id: ArrayInt32,
        lake_outflow_target_k: ArrayInt32,
        lake_outflow_wb_id: ArrayInt32,
        geom_inbank: TwoDArrayFloat32,
        geom_overbank: TwoDArrayFloat32,
        geom_cfl: TwoDArrayFloat32,
        pit_slope_inertial: ArrayFloat32,
        rev_demand_buf: ArrayFloat32,
        wb_release_volume_substep: ArrayFloat32,
        inertial_topo_order: ArrayInt32,
        total_flow_rate_buf: ArrayFloat32,
    ) -> tuple[
        ArrayFloat32,
        ArrayFloat64,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        np.float32,
        int,
    ]:
        dt_f32: np.float32 = np.float32(routing_timestep_s)
        inv_dt_f32: np.float32 = np.float32(1.0) / dt_f32
        n_cells: int = len(river_storage_m3)

        over_abstraction_m3.fill(0.0)
        actual_evaporation_m3.fill(0.0)
        waterbody_inflow_m3.fill(0.0)
        retention_inflow_m3.fill(0.0)
        retention_outflow_m3.fill(0.0)
        updated_discharge_m3_s.fill(0.0)
        kinematic_inflow_rate_inertial.fill(0.0)

        n_other_start: int = n_kinematic + n_inertial
        for i in range(n_other_start, n_cells):
            updated_discharge_m3_s[i] = np.float32(np.nan)

        _transfer_waterbody_outflows(
            wb_to_wb_src_wb=wb_to_wb_src_wb,
            wb_to_wb_tgt_wb=wb_to_wb_tgt_wb,
            inertial_outflow_per_waterbody=outflow_per_waterbody_m3,
            waterbody_storage_m3=waterbody_storage_m3,
            waterbody_inflow_m3=waterbody_inflow_m3,
            wb_to_kin_target_reach=wb_to_kin_target_reach,
            wb_to_kin_wb_id=wb_to_kin_wb_id,
            kin_wb_inflow_m3=kin_wb_inflow_m3,
        )

        if n_kinematic > 0:
            if n_kinematic >= KINEMATIC_PARALLEL_THRESHOLD:
                _run_kinematic_step_parallel(
                    n_kinematic=n_kinematic,
                    n_kin_headwater=n_kin_headwater,
                    dt_f32=dt_f32,
                    inv_dt_f32=inv_dt_f32,
                    previous_discharge_m3_s=previous_discharge_m3_s,
                    river_storage_m3=river_storage_m3,
                    sideflow_m3=sideflow_m3,
                    evaporation_m3=evaporation_m3,
                    waterbody_storage_m3=waterbody_storage_m3,
                    kin_ds_reach=kin_ds_reach,
                    kin_inflow_m3_s=kin_inflow_m3_s,
                    wb_to_kin_target_reach=wb_to_kin_target_reach,
                    kin_wb_inflow_m3=kin_wb_inflow_m3,
                    kin_ds_waterbody_id=kin_ds_waterbody_id,
                    river_storage_alpha=river_storage_alpha,
                    river_storage_beta=river_storage_beta,
                    river_length=river_length,
                    retention_storage_m3=retention_storage_m3,
                    retention_max_storage_m3=retention_max_storage_m3,
                    retention_node_id=retention_node_id,
                    controlled_retention=controlled_retention,
                    retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
                    retention_basin_release_threshold_factor=retention_basin_release_threshold_factor,
                    over_abstraction_m3=over_abstraction_m3,
                    actual_evaporation_m3=actual_evaporation_m3,
                    waterbody_inflow_m3=waterbody_inflow_m3,
                    retention_inflow_m3=retention_inflow_m3,
                    retention_outflow_m3=retention_outflow_m3,
                    updated_discharge_m3_s=updated_discharge_m3_s,
                )
            else:
                _run_kinematic_step_serial(
                    n_kinematic=n_kinematic,
                    n_kin_headwater=n_kin_headwater,
                    dt_f32=dt_f32,
                    inv_dt_f32=inv_dt_f32,
                    previous_discharge_m3_s=previous_discharge_m3_s,
                    river_storage_m3=river_storage_m3,
                    sideflow_m3=sideflow_m3,
                    evaporation_m3=evaporation_m3,
                    waterbody_storage_m3=waterbody_storage_m3,
                    kin_ds_reach=kin_ds_reach,
                    kin_inflow_m3_s=kin_inflow_m3_s,
                    wb_to_kin_target_reach=wb_to_kin_target_reach,
                    kin_wb_inflow_m3=kin_wb_inflow_m3,
                    kin_ds_waterbody_id=kin_ds_waterbody_id,
                    river_storage_alpha=river_storage_alpha,
                    river_storage_beta=river_storage_beta,
                    river_length=river_length,
                    retention_storage_m3=retention_storage_m3,
                    retention_max_storage_m3=retention_max_storage_m3,
                    retention_node_id=retention_node_id,
                    controlled_retention=controlled_retention,
                    retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
                    retention_basin_release_threshold_factor=retention_basin_release_threshold_factor,
                    over_abstraction_m3=over_abstraction_m3,
                    actual_evaporation_m3=actual_evaporation_m3,
                    waterbody_inflow_m3=waterbody_inflow_m3,
                    retention_inflow_m3=retention_inflow_m3,
                    retention_outflow_m3=retention_outflow_m3,
                    updated_discharge_m3_s=updated_discharge_m3_s,
                )

        (
            num_inertial_substeps,
            terminal_wb_outflow_m3,
        ) = _run_inertial_routing_step(
            dt_f32=dt_f32,
            inv_dt_f32=inv_dt_f32,
            n_inertial=n_inertial,
            inertial_up_kin_offsets=inertial_up_kin_offsets,
            inertial_up_kin_indices=inertial_up_kin_indices,
            inertial_up_kin_reach_idx=inertial_up_kin_reach_idx,
            inertial_up_offsets=inertial_up_offsets,
            inertial_up_indices=inertial_up_indices,
            inertial_up_reach_idx=inertial_up_reach_idx,
            previous_discharge_m3_s_inertial=previous_discharge_m3_s_inertial,
            updated_discharge_m3_s=updated_discharge_m3_s,
            river_storage_m3_inertial=river_storage_m3_inertial,
            sideflow_m3_inertial=sideflow_m3_inertial,
            geom_inbank=geom_inbank,
            geom_overbank=geom_overbank,
            geom_cfl=geom_cfl,
            ds_boundary_type=ds_boundary_type,
            ds_inertial_k=ds_inertial_k,
            ds_stage_idx=ds_stage_idx,
            ds_bed_elevation=ds_bed_elevation,
            kin_ds_slope=kin_ds_slope,
            pit_slope_inertial=pit_slope_inertial,
            min_dt_buf=min_dt_buf,
            evaporation_m3_inertial=evaporation_m3_inertial,
            over_abstraction_m3_inertial=over_abstraction_m3_inertial,
            actual_evaporation_m3_inertial=actual_evaporation_m3_inertial,
            updated_discharge_m3_s_inertial=updated_discharge_m3_s_inertial,
            kinematic_inflow_rate_inertial=kinematic_inflow_rate_inertial,
            discharge_vol_sum_inertial=discharge_vol_sum_inertial,
            wb_outflow_substep_m3_buf=wb_outflow_substep_m3_buf,
            retention_inflow_limit_sub_buf=retention_inflow_limit_sub_buf,
            retention_max_outflow_limit_sub_buf=retention_max_outflow_limit_sub_buf,
            wb_outflow_avail_buf=wb_outflow_avail_buf,
            wb_extra_lateral_accum_m3=wb_extra_lateral_accum_m3,
            substep_discharge_inertial=substep_discharge_inertial,
            wb_lake_area=wb_lake_area,
            wb_lake_factor=wb_lake_factor,
            wb_outflow_height=wb_outflow_height,
            wb_outflow_bed_elev=wb_outflow_bed_elev,
            stage_buf=stage_buf,
            inertial_retention_k=inertial_retention_k,
            inertial_retention_id=inertial_retention_id,
            retention_storage_m3=retention_storage_m3,
            retention_max_storage_m3=retention_max_storage_m3,
            controlled_retention=controlled_retention,
            retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
            retention_basin_release_threshold_factor=retention_basin_release_threshold_factor,
            waterbody_inflow_m3=waterbody_inflow_m3,
            retention_inflow_m3=retention_inflow_m3,
            retention_outflow_m3=retention_outflow_m3,
            waterbody_storage_m3=waterbody_storage_m3,
            outflow_per_waterbody_m3=outflow_per_waterbody_m3,
            inertial_wb_ds_k=inertial_wb_ds_k,
            inertial_wb_ds_id=inertial_wb_ds_id,
            lake_outflow_target_k=lake_outflow_target_k,
            lake_outflow_wb_id=lake_outflow_wb_id,
            wb_terminal_wb_ids=wb_terminal_wb_ids,
            rev_demand_buf=rev_demand_buf,
            wb_release_volume_substep=wb_release_volume_substep,
            inertial_topo_order=inertial_topo_order,
            total_flow_rate_buf=total_flow_rate_buf,
        )

        return (
            updated_discharge_m3_s,
            river_storage_m3,
            actual_evaporation_m3,
            over_abstraction_m3,
            waterbody_inflow_m3,
            retention_inflow_m3,
            retention_outflow_m3,
            terminal_wb_outflow_m3,
            num_inertial_substeps,
        )

    def _check_inertial_water_balance(
        self,
        storage_before: float,
        storage_after: float,
        sideflow_m3: float,
        kinematic_inflow_m3: float,
        evaporation_m3: float,
        over_abstraction_m3: float,
        wb_extra_lateral_m3: float,
        net_retention_m3: float,
        dt_s: float,
    ) -> float:
        """Verifies exact conservation of volumetric water mass across the inertial river network.

        Args:
            storage_before: Total inertial reach storage before timestep (m³).
            storage_after: Total inertial reach storage after timestep (m³).
            sideflow_m3: Sum of lateral catchment sideflows (m³).
            kinematic_inflow_m3: Total volumetric inflow across kinematic boundary reaches (m³).
            evaporation_m3: Total actual evaporation volume (m³).
            over_abstraction_m3: Total volume of unfulfilled abstraction deficit (m³).
            wb_extra_lateral_m3: Cumulative extra lateral volume transferred from waterbodies (m³).
            net_retention_m3: Net volumetric diversion into retention basins (m³).
            dt_s: Timestep duration (seconds).

        Returns:
            Absolute volumetric mass balance error (m³).

        Raises:
            ValueError: If mass balance error exceeds numerical tolerance.
        """
        if len(self._inertial_boundary_outflow_k) > 0:
            net_inertial_outflow_m3: float = float(
                np.sum(
                    self._updated_discharge_perm_inertial[
                        self._inertial_boundary_outflow_k
                    ]
                )
                * dt_s
            )
        else:
            net_inertial_outflow_m3 = 0.0

        expected_delta: float = (
            sideflow_m3
            + kinematic_inflow_m3
            + wb_extra_lateral_m3
            + over_abstraction_m3
            - evaporation_m3
            - net_retention_m3
            - net_inertial_outflow_m3
        )
        actual_delta: float = storage_after - storage_before
        balance_err: float = actual_delta - expected_delta

        tolerance: float = max(
            1.0,
            1e-3
            * max(
                storage_before,
                storage_after,
                net_inertial_outflow_m3,
                kinematic_inflow_m3,
                1.0,
            ),
        )
        if abs(balance_err) > tolerance:
            raise ValueError(
                f"Inertial network mass balance violation: error={balance_err:.4e} m³, "
                f"storage_before={storage_before:.2f}, storage_after={storage_after:.2f}, "
                f"sideflow={sideflow_m3:.2f}, kin_inflow={kinematic_inflow_m3:.2f}, "
                f"wb_extra_lat={wb_extra_lateral_m3:.2f}, evap={evaporation_m3:.2f}, "
                f"net_retention={net_retention_m3:.2f}, "
                f"over_abs={over_abstraction_m3:.2f}, net_outflow={net_inertial_outflow_m3:.2f}"
            )
        return balance_err

    def step(
        self,
        Q_prev_m3_s: ArrayFloat32,
        river_storage_m3: ArrayFloat64,
        sideflow_m3: ArrayFloat32,
        evaporation_m3: ArrayFloat32,
        waterbody_storage_m3: ArrayFloat64,
        outflow_per_waterbody_m3: ArrayFloat32,
        retention_storage_m3: ArrayFloat32,
        retention_activation_threshold_m3_s: ArrayFloat32,
        already_permuted: bool = False,
    ) -> tuple[
        ArrayFloat32,
        ArrayFloat64,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat64,
        ArrayFloat32,
        np.float32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
    ]:
        """Executes a full macro-timestep routing update.

        Args:
            Q_prev_m3_s: Channel discharge at start of timestep (m³/s).
            river_storage_m3: Total reach storage volume (m³).
            sideflow_m3: Lateral sideflow forcing volume per reach (m³).
            evaporation_m3: Potential evaporation volume per reach (m³).
            waterbody_storage_m3: Current waterbody storage volume (m³).
            outflow_per_waterbody_m3: Prescribed outflow volume per waterbody (m³).
            retention_storage_m3: Current storage volume per retention basin (m³).
            retention_activation_threshold_m3_s: Activation discharge threshold per basin (m³/s).
            already_permuted: If True, indicates input arrays are already permuted into topological order,
                bypassing gather and scatter overhead.

        Returns:
            A tuple containing:
                - Q_out_m3_s: Calculated channel discharge at end of step (m³/s).
                - river_storage_m3: Updated river reach storage (m³).
                - actual_evap_m3: Actual evaporation volume per reach (m³).
                - over_abs_m3: Deficit from unfulfilled abstractions (m³).
                - waterbody_storage_m3: Updated waterbody storage (m³).
                - waterbody_inflow_m3: Inflow volume per waterbody (m³).
                - outflow_at_pits_m3: Total outflow volume exiting through domain boundaries (m³).
                - retention_storage_m3: Updated storage volume per retention basin (m³).
                - retention_inflow_m3: Diverted inflow volume per retention basin (m³).
                - retention_outflow_m3: Released outflow volume per retention basin (m³).

        Raises:
            ValueError: If non-finite values are encountered in input forcing or state arrays.
        """
        if __debug__:
            if not np.all(np.isfinite(river_storage_m3)):
                raise ValueError("Non-finite river_storage encountered in inputs.")
            if not np.all(np.isfinite(sideflow_m3)):
                raise ValueError("Non-finite sideflow encountered in inputs.")
            if not np.all(np.isfinite(evaporation_m3)):
                raise ValueError("Non-finite evaporation encountered in inputs.")

        if not already_permuted:
            _gather_inputs(
                self.sorted_idxs,
                Q_prev_m3_s,
                river_storage_m3,
                sideflow_m3,
                evaporation_m3,
                self._discharge_prev_perm,
                self._river_storage_perm,
                self._sideflow_perm,
                self._evaporation_perm,
            )
        else:
            self._discharge_prev_perm[:] = Q_prev_m3_s
            self._river_storage_perm[:] = river_storage_m3
            self._sideflow_perm[:] = sideflow_m3
            self._evaporation_perm[:] = evaporation_m3

        self._wb_extra_lateral_accum_m3.fill(0.0)

        storage_inertial_before: float = float(
            np.sum(self._river_storage_perm_inertial)
        )
        sideflow_inertial: float = float(np.sum(self._sideflow_perm_inertial))

        (
            discharge_perm,
            river_storage_perm,
            actual_evaporation_perm,
            over_abstraction_perm,
            waterbody_inflow_m3,
            retention_inflow_m3,
            retention_outflow_m3,
            terminal_wb_outflow_m3,
            num_inertial_substeps,
        ) = self._step(
            routing_timestep_s=self.dt,
            previous_discharge_m3_s=self._discharge_prev_perm,
            river_storage_m3=self._river_storage_perm,
            sideflow_m3=self._sideflow_perm,
            evaporation_m3=self._evaporation_perm,
            waterbody_storage_m3=waterbody_storage_m3,
            outflow_per_waterbody_m3=outflow_per_waterbody_m3,
            kin_ds_reach=self._kin_ds_reach,
            kin_inflow_m3_s=self._kin_inflow_m3_s,
            n_kin_headwater=self.n_kin_headwater,
            inertial_up_kin_offsets=self._inertial_up_kin_offsets,
            inertial_up_kin_indices=self._inertial_up_kin_indices,
            inertial_up_kin_reach_idx=self._inertial_up_kin_reach_idx,
            wb_to_wb_src_wb=self._wb_to_wb_src_wb,
            wb_to_wb_tgt_wb=self._wb_to_wb_tgt_wb,
            wb_to_kin_target_reach=self._wb_to_kin_target_reach,
            wb_to_kin_wb_id=self._wb_to_kin_wb_id,
            kin_wb_inflow_m3=self._kin_wb_inflow_m3,
            wb_terminal_wb_ids=self._wb_terminal_wb_ids,
            kin_ds_waterbody_id=self._kin_ds_waterbody_id,
            river_storage_alpha=self._alpha_perm,
            river_storage_beta=self._beta_perm,
            river_length=self._river_length,
            retention_storage_m3=retention_storage_m3,
            retention_max_storage_m3=self.retention_max_storage_m3,
            retention_node_id=self._retention_node_id,
            controlled_retention=self.controlled_retention,
            retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
            retention_basin_release_threshold_factor=self.retention_basin_release_threshold_factor,
            n_kinematic=self.n_kinematic,
            n_inertial=self.n_inertial,
            over_abstraction_m3=self._over_abstraction_perm,
            actual_evaporation_m3=self._actual_evaporation_perm,
            waterbody_inflow_m3=self._waterbody_inflow_perm,
            retention_inflow_m3=self._retention_inflow_perm,
            retention_outflow_m3=self._retention_outflow_perm,
            updated_discharge_m3_s=self._updated_discharge_perm,
            river_storage_m3_inertial=self._river_storage_perm_inertial,
            previous_discharge_m3_s_inertial=self._discharge_prev_perm_inertial,
            sideflow_m3_inertial=self._sideflow_perm_inertial,
            evaporation_m3_inertial=self._evaporation_perm_inertial,
            over_abstraction_m3_inertial=self._over_abstraction_perm_inertial,
            actual_evaporation_m3_inertial=self._actual_evaporation_perm_inertial,
            updated_discharge_m3_s_inertial=self._updated_discharge_perm_inertial,
            kinematic_inflow_rate_inertial=self._kinematic_inflow_rate_inertial,
            discharge_vol_sum_inertial=self._discharge_vol_sum_inertial,
            wb_outflow_substep_m3_buf=self._wb_outflow_substep_m3_buf,
            retention_inflow_limit_sub_buf=self._retention_inflow_limit_sub_buf,
            retention_max_outflow_limit_sub_buf=self._retention_max_outflow_limit_sub_buf,
            wb_outflow_avail_buf=self._wb_outflow_avail_buf,
            wb_extra_lateral_accum_m3=self._wb_extra_lateral_accum_m3,
            substep_discharge_inertial=self._substep_discharge_inertial,
            wb_lake_area=self._wb_lake_area,
            wb_lake_factor=self._wb_lake_factor,
            wb_outflow_height=self._wb_outflow_height,
            wb_outflow_bed_elev=self._wb_outflow_bed_elev,
            stage_buf=self._stage_buf,
            ds_boundary_type=self._ds_boundary_type,
            ds_inertial_k=self._ds_inertial_k,
            ds_stage_idx=self._ds_stage_idx,
            ds_bed_elevation=self._ds_bed_elevation,
            kin_ds_slope=self._kin_ds_slope,
            min_dt_buf=self._min_dt_buf,
            inertial_up_offsets=self._inertial_up_offsets,
            inertial_up_indices=self._inertial_up_indices,
            inertial_up_reach_idx=self._inertial_up_reach_idx,
            inertial_retention_k=self._inertial_retention_k,
            inertial_retention_id=self._inertial_retention_id,
            inertial_wb_ds_k=self._inertial_wb_ds_k,
            inertial_wb_ds_id=self._inertial_wb_ds_id,
            lake_outflow_target_k=self._lake_outflow_target_k,
            lake_outflow_wb_id=self._lake_outflow_wb_id,
            geom_inbank=self._geom_inbank,
            geom_overbank=self._geom_overbank,
            geom_cfl=self._geom_cfl,
            pit_slope_inertial=self._pit_slope_inertial,
            rev_demand_buf=self._rev_demand_buf,
            wb_release_volume_substep=self._wb_release_volume_substep,
            inertial_topo_order=self._inertial_topo_order,
            total_flow_rate_buf=self._total_flow_rate_buf,
        )

        storage_inertial_after: float = float(np.sum(self._river_storage_perm_inertial))
        kinematic_inflow_total: float = float(
            np.sum(self._kinematic_inflow_rate_inertial) * self.dt
        )
        evap_inertial: float = float(np.sum(self._actual_evaporation_perm_inertial))
        over_abs_inertial: float = float(np.sum(self._over_abstraction_perm_inertial))
        wb_extra_lateral_total: float = float(np.sum(self._wb_extra_lateral_accum_m3))

        net_retention_inertial: float = (
            float(
                np.sum(
                    self._retention_inflow_perm[self._inertial_retention_id]
                    - self._retention_outflow_perm[self._inertial_retention_id]
                )
            )
            if len(self._inertial_retention_id) > 0
            else 0.0
        )

        self._check_inertial_water_balance(
            storage_before=storage_inertial_before,
            storage_after=storage_inertial_after,
            sideflow_m3=sideflow_inertial,
            kinematic_inflow_m3=kinematic_inflow_total,
            evaporation_m3=evap_inertial,
            over_abstraction_m3=over_abs_inertial,
            wb_extra_lateral_m3=wb_extra_lateral_total,
            net_retention_m3=net_retention_inertial,
            dt_s=float(self.dt),
        )

        outflow_at_pits_m3: np.float32 = (
            np.float32(np.sum(discharge_perm[self._pit_indices]) * self.dt)
            if len(self._pit_indices) > 0
            else np.float32(0.0)
        ) + terminal_wb_outflow_m3

        if not already_permuted:
            _scatter_outputs(
                self.sorted_idxs,
                discharge_perm,
                actual_evaporation_perm,
                over_abstraction_perm,
                river_storage_perm,
                self._discharge_out,
                self._actual_evap_out,
                self._over_abs_out,
                river_storage_m3,
            )

            return (
                self._discharge_out,
                river_storage_m3,
                self._actual_evap_out,
                self._over_abs_out,
                waterbody_storage_m3,
                waterbody_inflow_m3,
                outflow_at_pits_m3,
                retention_storage_m3,
                retention_inflow_m3,
                retention_outflow_m3,
            )
        else:
            return (
                discharge_perm,
                river_storage_perm,
                actual_evaporation_perm,
                over_abstraction_perm,
                waterbody_storage_m3,
                waterbody_inflow_m3,
                outflow_at_pits_m3,
                retention_storage_m3,
                retention_inflow_m3,
                retention_outflow_m3,
            )
