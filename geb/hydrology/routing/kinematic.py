"""Kinematic wave routing equation solver and network step execution for river networks."""

import numpy as np
from numba import njit, prange

from geb.geb_types import (
    ArrayFloat32,
    ArrayFloat64,
)

from .inertial_kernel import (
    RetentionCoupling,
    WaterbodyCoupling,
    compute_retention_routing,
)
from .topology import KinematicTopology

__all__: list[str] = [
    "KINEMATIC_PARALLEL_THRESHOLD",
    "MAX_ITERS_KINEMATIC",
    "update_node_kinematic",
    "_run_kinematic_step",
    "_run_kinematic_step_parallel",
    "_run_kinematic_step_serial",
    "_solve_kinematic_reach",
    "_transfer_waterbody_outflows",
]

MAX_ITERS_KINEMATIC: int = 10

# Threshold of headwater reach count above which parallel multi-threading outperforms serial execution
KINEMATIC_PARALLEL_THRESHOLD: int = 5000


@njit(cache=True)
def update_node_kinematic(
    inflow_m3_s: np.float32,
    previous_discharge_m3_s: np.float32,
    sideflow_m3_s: np.float32,
    evaporation_m3_s: np.float32,
    river_storage_alpha: np.float32,
    river_storage_beta: np.float32,
    timestep_s: np.float32,
    river_length_m: np.float32,
    epsilon: np.float32 = np.float32(0.00001),
) -> tuple[np.float32, np.float32]:
    """Calculate the new discharge and actual evaporation for a single cell using kinematic wave routing.

    Solves the non-linear kinematic wave equation using the Newton-Raphson method.

    Args:
        inflow_m3_s: Inflow discharge from upstream cells (m³/s).
        previous_discharge_m3_s: Discharge from the previous time step (m³/s).
        sideflow_m3_s: Lateral inflow rate into the river cell (m³/s).
        evaporation_m3_s: Potential evaporation rate from the river surface (m³/s).
        river_storage_alpha: Kinematic wave alpha parameter (s^beta / m^(3*beta - 1)).
        river_storage_beta: Kinematic wave beta parameter (-).
        timestep_s: Routing time step duration (seconds).
        river_length_m: Length of the river channel in the cell (meters).
        epsilon: Convergence threshold for Newton-Raphson solver.

    Returns:
        A tuple containing:
            - new_discharge_m3_s: Updated discharge at current timestep (m³/s).
            - actual_evaporation_m3_s: Actual evaporation rate constrained by available flow (m³/s).
    """
    inv_river_length: np.float32 = np.float32(1.0) / river_length_m
    # Limit evaporation to available flow rate (m³/s)
    available_flow_m3_s: np.float32 = (
        inflow_m3_s + previous_discharge_m3_s
    ) * np.float32(0.5) + max(sideflow_m3_s, np.float32(0.0))
    actual_evaporation_m3_s: np.float32 = min(
        evaporation_m3_s,
        max(available_flow_m3_s, np.float32(0.0)),
    )

    evaporation_m3_s_per_m: np.float32 = actual_evaporation_m3_s * inv_river_length
    lateral_inflow_m3_s_per_m: np.float32 = (
        sideflow_m3_s * inv_river_length - evaporation_m3_s_per_m
    )

    # Return tiny positive flow if reach is dry
    if (inflow_m3_s + previous_discharge_m3_s + lateral_inflow_m3_s_per_m) < 1e-30:
        return np.float32(1e-30), actual_evaporation_m3_s

    inflow_m3_s = max(inflow_m3_s, np.float32(1e-30))
    previous_discharge_m3_s = max(previous_discharge_m3_s, np.float32(1e-30))

    # Initial guess for discharge using a linearized approximation
    beta_minus_1: np.float32 = river_storage_beta - np.float32(1.0)
    derivative_storage: np.float32 = (
        river_storage_alpha
        * river_storage_beta
        * (
            max(
                (previous_discharge_m3_s + inflow_m3_s) * np.float32(0.5),
                np.float32(1e-30),
            )
            ** beta_minus_1
        )
    )
    dt_over_dx: np.float32 = timestep_s * inv_river_length
    prev_storage: np.float32 = previous_discharge_m3_s**river_storage_beta
    rhs_constant: np.float32 = (
        dt_over_dx * inflow_m3_s
        + river_storage_alpha * prev_storage
        + timestep_s * lateral_inflow_m3_s_per_m
    )

    new_discharge: np.float32 = (
        dt_over_dx * inflow_m3_s
        + previous_discharge_m3_s * derivative_storage
        + timestep_s * lateral_inflow_m3_s_per_m
    ) / (dt_over_dx + derivative_storage)
    new_discharge = max(new_discharge, np.float32(1e-30))

    # Newton-Raphson iteration
    alpha_times_beta: np.float32 = river_storage_alpha * river_storage_beta
    q_current: np.float32 = max(new_discharge, np.float32(1e-30))
    q_pow_beta: np.float32 = q_current**river_storage_beta
    f_q: np.float32 = (
        dt_over_dx * new_discharge + river_storage_alpha * q_pow_beta - rhs_constant
    )

    iteration: int = 0
    while np.abs(f_q) > epsilon and iteration < MAX_ITERS_KINEMATIC:
        df_q: np.float32 = dt_over_dx + alpha_times_beta * (q_pow_beta / q_current)
        new_discharge -= f_q / df_q
        new_discharge = max(new_discharge, np.float32(1e-30))

        q_current = new_discharge
        q_pow_beta = q_current**river_storage_beta
        f_q = (
            dt_over_dx * new_discharge + river_storage_alpha * q_pow_beta - rhs_constant
        )
        iteration += 1

    return new_discharge, actual_evaporation_m3_s


@njit(cache=True)
def _transfer_waterbody_outflows(
    wb: WaterbodyCoupling,
    kin: KinematicTopology,
) -> None:
    """Transfers waterbody outflows to connected downstream waterbodies and kinematic reaches.

    Args:
        wb: Waterbody coupling parameters, links, and storage state tables.
        kin: Kinematic wave network topology and reach tables.
    """
    # Waterbody-to-waterbody transfers for prescribed releases (e.g., reservoirs).
    # Natural lakes return NaN from routing_lakes() to request dynamic rating-curve
    # evaluation at each inertial substep (in _run_inertial_substeps), so NaN > 0
    # evaluates to False and they are intentionally skipped here.
    for idx in range(len(wb.wb_to_wb_src_wb)):
        src_wb: int = wb.wb_to_wb_src_wb[idx]
        tgt_wb: int = wb.wb_to_wb_tgt_wb[idx]
        outflow_vol: float = np.float64(wb.prescribed_outflow_m3[src_wb])
        if outflow_vol > 0.0:
            wb.storage_m3[src_wb] -= outflow_vol
            wb.storage_m3[tgt_wb] += outflow_vol
            wb.inflow_m3[tgt_wb] += np.float32(outflow_vol)
            wb.prescribed_outflow_m3[src_wb] = np.float32(0.0)

    # Waterbody outflows into receiving kinematic reaches for prescribed releases.
    # As with wb-to-wb transfers, dynamic natural lake outflows drain into inertial
    # reaches and are calculated within the substepping kernel, not here.
    for idx in range(len(kin.wb_to_kin_target_reach)):
        tgt_reach: int = kin.wb_to_kin_target_reach[idx]
        wb_id_up: int = kin.wb_to_kin_wb_id[idx]
        wb_of: np.float32 = wb.prescribed_outflow_m3[wb_id_up]
        if wb_of > np.float32(0.0):
            rel_vol: float = np.float64(wb_of)
            wb.storage_m3[wb_id_up] -= rel_vol
            kin.kin_wb_inflow_m3[tgt_reach] += wb_of
            wb.prescribed_outflow_m3[wb_id_up] = np.float32(0.0)


@njit(inline="always")
def _solve_kinematic_reach(
    reach_idx: int,
    upstream_inflow_m3_s: np.float32,
    node_sideflow: np.float32,
    dt_f32: np.float32,
    inv_dt_f32: np.float32,
    previous_discharge: np.float32,
    evaporation_vol_m3: np.float32,
    river_storage_alpha: np.float32,
    river_storage_beta: np.float32,
    river_length: np.float32,
    river_storage_m3: ArrayFloat64,
    updated_discharge_m3_s: ArrayFloat32,
    actual_evaporation_m3: ArrayFloat32,
    over_abstraction_m3: ArrayFloat32,
    ret_id: int,
    ret: RetentionCoupling,
) -> np.float32:
    """Solves flood retention, kinematic wave routing, evaporation, and mass balance for a single reach.

    Args:
        reach_idx: Index of reach to update.
        upstream_inflow_m3_s: Aggregated upstream inflow discharge rate (m³/s).
        node_sideflow: Net lateral inflow volume (including waterbody release) for reach (m³).
        dt_f32: Routing timestep duration (seconds).
        inv_dt_f32: Reciprocal of routing timestep (1/s).
        previous_discharge: Discharge at start of timestep (m³/s).
        evaporation_vol_m3: Potential evaporation volume for reach (m³).
        river_storage_alpha: Kinematic wave alpha parameter.
        river_storage_beta: Kinematic wave beta parameter.
        river_length: Channel reach length (meters).
        river_storage_m3: Storage volume array for all reaches (m³).
        updated_discharge_m3_s: Output array to store computed discharge (m³/s).
        actual_evaporation_m3: Output array to store realized evaporation volume (m³).
        over_abstraction_m3: Output array accumulating unmet extraction volume (m³).
        ret_id: Reach-to-retention basin mapping index (-1 if none).
        ret: Flood retention basin state tables and control parameters.

    Returns:
        Computed downstream discharge rate for reach (m³/s).
    """
    if ret_id != -1:
        discharge_before_diversion: np.float32 = max(
            upstream_inflow_m3_s, np.float32(0.0)
        ) + (node_sideflow * inv_dt_f32)

        discharge_at_basin_vol: np.float32 = (
            upstream_inflow_m3_s * dt_f32 + node_sideflow
        )
        inflow_limit: np.float32 = np.float32(0.20) * ret.max_storage_m3[ret_id]
        max_outflow_limit: np.float32 = np.float32(0.05) * ret.max_storage_m3[ret_id]
        is_rising_limb: bool = bool(discharge_before_diversion > previous_discharge)

        (
            diverted_volume,
            outflow_volume,
            new_ret_storage,
            discharge_at_basin_vol,
        ) = compute_retention_routing(
            dt=dt_f32,
            river_volume_m3=discharge_at_basin_vol,
            discharge_before_diversion_m3_s=discharge_before_diversion,
            is_rising_limb=bool(is_rising_limb),
            retention_storage_m3=ret.storage_m3[ret_id],
            retention_max_storage_m3=ret.max_storage_m3[ret_id],
            controlled_retention=ret.controlled_retention[ret_id],
            activation_threshold_m3_s=ret.activation_threshold_m3_s[ret_id],
            release_threshold_factor=ret.release_threshold_factor,
            inflow_limit_m3=inflow_limit,
            max_outflow_limit_m3=max_outflow_limit,
        )

        ret.storage_m3[ret_id] = new_ret_storage
        ret.inflow_m3[ret_id] += diverted_volume
        ret.outflow_m3[ret_id] += outflow_volume
        node_sideflow = discharge_at_basin_vol - upstream_inflow_m3_s * dt_f32

    # Solve kinematic wave routing for outflow discharge and channel evaporation.
    kinematic_discharge, act_evap_rate = update_node_kinematic(
        inflow_m3_s=upstream_inflow_m3_s,
        previous_discharge_m3_s=previous_discharge,
        sideflow_m3_s=node_sideflow * inv_dt_f32,
        evaporation_m3_s=evaporation_vol_m3 * inv_dt_f32,
        river_storage_alpha=river_storage_alpha,
        river_storage_beta=river_storage_beta,
        timestep_s=dt_f32,
        river_length_m=river_length,
    )

    kinematic_discharge = max(kinematic_discharge, np.float32(0.0))
    updated_discharge_m3_s[reach_idx] = kinematic_discharge

    evap_vol: np.float32 = act_evap_rate * dt_f32
    actual_evaporation_m3[reach_idx] = evap_vol

    # Update reach storage (mass balance) and track over-abstraction deficit if any.
    inflow_vol: np.float32 = upstream_inflow_m3_s * dt_f32 + node_sideflow
    outflow_vol: np.float32 = kinematic_discharge * dt_f32
    river_storage_m3[reach_idx] += (
        np.float64(inflow_vol) - np.float64(outflow_vol) - np.float64(evap_vol)
    )

    if river_storage_m3[reach_idx] < np.float64(0.0):
        over_abstraction_m3[reach_idx] += np.float32(-river_storage_m3[reach_idx])
        river_storage_m3[reach_idx] = np.float64(0.0)

    return kinematic_discharge


def _run_kinematic_step(
    dt_f32: np.float32,
    inv_dt_f32: np.float32,
    previous_discharge_m3_s: ArrayFloat32,
    river_storage_m3: ArrayFloat64,
    sideflow_m3: ArrayFloat32,
    evaporation_m3: ArrayFloat32,
    over_abstraction_m3: ArrayFloat32,
    actual_evaporation_m3: ArrayFloat32,
    updated_discharge_m3_s: ArrayFloat32,
    kin: KinematicTopology,
    wb: WaterbodyCoupling,
    ret: RetentionCoupling,
) -> None:
    """Executes kinematic wave routing across headwater and connected reaches with in-place updates.

    Phase 1 solves all independent headwater reaches (in-degree 0) in parallel using prange.
    Phase 2 solves all downstream connected reaches in topological order with O(1) push-based inflow accumulation.

    Args:
        dt_f32: Routing time step duration (seconds).
        inv_dt_f32: Inverse of routing time step duration (1/s).
        previous_discharge_m3_s: Discharge from previous time step (m³/s).
        river_storage_m3: River reach storage volume (m³).
        sideflow_m3: Lateral inflow volume per reach (m³).
        evaporation_m3: Potential evaporation volume per reach (m³).
        over_abstraction_m3: Cumulative volume of over-abstraction (m³).
        actual_evaporation_m3: Output actual evaporation volume array to populate (m³).
        updated_discharge_m3_s: Output discharge array to populate (m³/s).
        kin: Kinematic wave network topology and reach parameter tables.
        wb: Waterbody coupling parameters, links, and storage state tables.
        ret: Flood retention basin state tables and control parameters.

    Raises:
        ValueError: If non-finite discharge is computed in any kinematic wave reach.
    """
    kin.kin_inflow_m3_s.fill(np.float32(0.0))

    # Route headwater reaches in parallel across threads.
    # Headwaters have no upstream river reaches feeding them, making them mutually
    # independent. They can be computed concurrently without race conditions.
    for i in prange(kin.n_kin_headwater):
        # Collect local lateral runoff and waterbody releases (upstream river inflow is 0).
        node_sideflow: np.float32 = sideflow_m3[i]
        if len(kin.wb_to_kin_target_reach) > 0:
            node_sideflow += kin.kin_wb_inflow_m3[i]
            kin.kin_wb_inflow_m3[i] = np.float32(0.0)

        _solve_kinematic_reach(
            reach_idx=i,
            upstream_inflow_m3_s=np.float32(0.0),
            node_sideflow=node_sideflow,
            dt_f32=dt_f32,
            inv_dt_f32=inv_dt_f32,
            previous_discharge=previous_discharge_m3_s[i],
            evaporation_vol_m3=evaporation_m3[i],
            river_storage_alpha=kin.river_storage_alpha[i],
            river_storage_beta=kin.river_storage_beta[i],
            river_length=kin.river_length[i],
            river_storage_m3=river_storage_m3,
            updated_discharge_m3_s=updated_discharge_m3_s,
            actual_evaporation_m3=actual_evaporation_m3,
            over_abstraction_m3=over_abstraction_m3,
            ret_id=int(kin.retention_node_id[i]),
            ret=ret,
        )

    # Pass headwater outflows to downstream reaches and waterbodies.
    # This must be done in serial to avoid race conditions.
    for i in range(kin.n_kin_headwater):
        q_head: np.float32 = updated_discharge_m3_s[i]
        if __debug__:
            if not np.isfinite(q_head):
                raise ValueError(
                    "Non-finite discharge computed in kinematic wave reach."
                )
        ds_reach: int = kin.kin_ds_reach[i]
        if ds_reach != -1:
            kin.kin_inflow_m3_s[ds_reach] += q_head

        wb_ds_id: int = kin.kin_ds_waterbody_id[i]
        if wb_ds_id != -1:
            outflow_vol_hw: np.float32 = q_head * dt_f32
            wb.storage_m3[wb_ds_id] += np.float64(outflow_vol_hw)
            if outflow_vol_hw > np.float32(0.0):
                wb.inflow_m3[wb_ds_id] += outflow_vol_hw

    # Route connected reaches in downstream topological order.
    # Because reaches are sorted topologically, all upstream reaches feeding reach `i`
    # have already finished and deposited their outflow into kin_inflow_m3_s[i].
    for i in range(kin.n_kin_headwater, kin.n_kinematic):
        node_sideflow: np.float32 = sideflow_m3[i]
        if len(kin.wb_to_kin_target_reach) > 0:
            node_sideflow += kin.kin_wb_inflow_m3[i]
            kin.kin_wb_inflow_m3[i] = np.float32(0.0)

        kinematic_discharge: np.float32 = _solve_kinematic_reach(
            reach_idx=i,
            upstream_inflow_m3_s=kin.kin_inflow_m3_s[i],
            node_sideflow=node_sideflow,
            dt_f32=dt_f32,
            inv_dt_f32=inv_dt_f32,
            previous_discharge=previous_discharge_m3_s[i],
            evaporation_vol_m3=evaporation_m3[i],
            river_storage_alpha=kin.river_storage_alpha[i],
            river_storage_beta=kin.river_storage_beta[i],
            river_length=kin.river_length[i],
            river_storage_m3=river_storage_m3,
            updated_discharge_m3_s=updated_discharge_m3_s,
            actual_evaporation_m3=actual_evaporation_m3,
            over_abstraction_m3=over_abstraction_m3,
            ret_id=int(kin.retention_node_id[i]),
            ret=ret,
        )

        wb_ds_id: int = kin.kin_ds_waterbody_id[i]
        if wb_ds_id != -1:
            outflow_vol: np.float32 = kinematic_discharge * dt_f32
            wb.storage_m3[wb_ds_id] += np.float64(outflow_vol)
            if outflow_vol > np.float32(0.0):
                wb.inflow_m3[wb_ds_id] += outflow_vol

        if __debug__:
            if not np.isfinite(kinematic_discharge):
                raise ValueError(
                    "Non-finite discharge computed in kinematic wave reach."
                )

        ds_reach: int = kin.kin_ds_reach[i]
        if ds_reach != -1:
            kin.kin_inflow_m3_s[ds_reach] += kinematic_discharge


_run_kinematic_step_serial = njit(parallel=False, cache=True)(_run_kinematic_step)
_run_kinematic_step_parallel = njit(parallel=True, cache=True)(_run_kinematic_step)
