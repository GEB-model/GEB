"""Kinematic wave routing equation solver and network step execution for river networks."""

import numpy as np
from numba import njit, prange

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64,
)
from geb.numba_config import NUMBA_CACHE

from .topology import (
    KinematicTopology,
    RetentionCoupling,
    WaterbodyCoupling,
    compute_retention_routing,
)

__all__: list[str] = [
    "KINEMATIC_PARALLEL_THRESHOLD",
    "MAX_ITERS_KINEMATIC",
    "update_node_kinematic",
    "_run_kinematic_step",
    "_run_kinematic_step_parallel",
    "_run_kinematic_step_serial",
    "_solve_kinematic_reach",
    "_transfer_waterbody_outflows",
    "_waterbody_release_over_timestep",
]

MAX_ITERS_KINEMATIC: int = 10

# Threshold of headwater reach count above which parallel multi-threading outperforms serial execution
KINEMATIC_PARALLEL_THRESHOLD: int = 5000


@njit(cache=NUMBA_CACHE)
def update_node_kinematic(
    inflow_m3_s: np.float32,
    previous_discharge_m3_s: np.float32,
    sideflow_m3_s: np.float32,
    evaporation_m3_s: np.float32,
    transmission_loss_m3_s: np.float32,
    river_storage_alpha: np.float32,
    river_storage_beta: np.float32,
    timestep_s: np.float32,
    river_length_m: np.float32,
    stream_width_m: np.float32,
    bankfull_depth_m: np.float32,
    floodplain_width_m: np.float32,
    floodplain_conveyance: np.float32,
    epsilon: np.float32 = np.float32(0.00001),
) -> tuple[np.float32, np.float32, np.float32]:
    """Calculate the new discharge, actual evaporation, and transmission loss for a single cell using kinematic wave routing.

    Solves the non-linear kinematic wave equation using the Newton-Raphson method.

    In the channel the flow area follows A = alpha * Q^beta. Where the stream has a
    valley floor next to it (floodplain_conveyance > 0), water above the banks also
    flows over that floor. With the depth y above the bed, the bankfull depth h, the
    channel width W and the floodplain width Wf next to the channel:

        Q(y) = (W * y / alpha)^(1 / beta) + floodplain_conveyance * (y - h)^(5/3)
        A(y) = W * y + Wf * (y - h)

    The first term is the channel and the column of water above it, the second the
    flow over the valley floor (wide and shallow, Manning). Without a valley floor this
    is the channel relation at every depth.

    Args:
        inflow_m3_s: Inflow discharge from upstream cells (m³/s).
        previous_discharge_m3_s: Discharge from the previous time step (m³/s).
        sideflow_m3_s: Lateral inflow rate into the river cell (m³/s).
        evaporation_m3_s: Potential evaporation rate from the river surface (m³/s).
        transmission_loss_m3_s: Potential riverbed transmission loss rate (m³/s).
        river_storage_alpha: Kinematic wave alpha parameter (s^beta / m^(3*beta - 1)).
        river_storage_beta: Kinematic wave beta parameter (-).
        timestep_s: Routing time step duration (seconds).
        river_length_m: Length of the river channel in the cell (meters).
        epsilon: Convergence threshold for Newton-Raphson solver.
        stream_width_m: Bankfull width of the channel (meters).
        bankfull_depth_m: Depth of the channel when full to the banks (meters); infinite
            for a channel without a valley floor.
        floodplain_width_m: Width of the valley floor next to the channel (meters).
        floodplain_conveyance: sqrt(slope) * floodplain width / Manning n of the floodplain
            (m^(4/3)/s); zero for a channel without a valley floor.

    Returns:
        A tuple containing:
            - new_discharge_m3_s: Updated discharge at current timestep (m³/s).
            - actual_evaporation_m3_s: Actual evaporation rate constrained by available flow (m³/s).
            - actual_transmission_loss_m3_s: Actual transmission loss rate constrained by available flow (m³/s).
    """
    inv_river_length: np.float32 = np.float32(1.0) / river_length_m
    # Limit evaporation and transmission loss to available flow rate (m³/s)
    available_flow_m3_s: np.float32 = (
        inflow_m3_s + previous_discharge_m3_s
    ) * np.float32(0.5) + max(sideflow_m3_s, np.float32(0.0))
    actual_evaporation_m3_s: np.float32 = min(
        evaporation_m3_s,
        max(available_flow_m3_s, np.float32(0.0)),
    )
    actual_transmission_loss_m3_s: np.float32 = min(
        transmission_loss_m3_s,
        max(available_flow_m3_s - actual_evaporation_m3_s, np.float32(0.0)),
    )

    total_loss_m3_s_per_m: np.float32 = (
        actual_evaporation_m3_s + actual_transmission_loss_m3_s
    ) * inv_river_length
    lateral_inflow_m3_s_per_m: np.float32 = (
        sideflow_m3_s * inv_river_length - total_loss_m3_s_per_m
    )

    # Return tiny positive flow if reach is dry
    if (inflow_m3_s + previous_discharge_m3_s + lateral_inflow_m3_s_per_m) < 1e-30:
        return np.float32(1e-30), actual_evaporation_m3_s, actual_transmission_loss_m3_s

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
    prev_flow_area_m2: np.float32 = (
        river_storage_alpha * previous_discharge_m3_s**river_storage_beta
    )

    # Stream with a valley floor: flow area of the previous discharge when it is above the banks
    has_floodplain: bool = bool(
        floodplain_conveyance > np.float32(0.0)
        and stream_width_m > np.float32(0.0)
        and bankfull_depth_m < np.float32(np.inf)
    )
    inv_beta: np.float32 = np.float32(1.0) / river_storage_beta
    five_thirds: np.float32 = np.float32(5.0 / 3.0)
    two_thirds: np.float32 = np.float32(2.0 / 3.0)
    bankfull_discharge_m3_s: np.float32 = np.float32(np.inf)
    if has_floodplain:
        bankfull_discharge_m3_s = (
            stream_width_m * bankfull_depth_m / river_storage_alpha
        ) ** inv_beta
        if previous_discharge_m3_s > bankfull_discharge_m3_s:
            # depth for the previous discharge: Newton from the depth the channel alone
            # would need, which is above the solution (the function is convex)
            depth_m: np.float32 = prev_flow_area_m2 / stream_width_m
            for _ in range(MAX_ITERS_KINEMATIC):
                channel_discharge: np.float32 = (
                    stream_width_m * depth_m / river_storage_alpha
                ) ** inv_beta
                over_bank_m: np.float32 = depth_m - bankfull_depth_m
                residual: np.float32 = (
                    channel_discharge
                    + floodplain_conveyance * over_bank_m**five_thirds
                    - previous_discharge_m3_s
                )
                if abs(residual) <= epsilon * max(
                    previous_discharge_m3_s, np.float32(1.0)
                ):
                    break
                derivative: np.float32 = (
                    inv_beta * channel_discharge / depth_m
                    + five_thirds * floodplain_conveyance * over_bank_m**two_thirds
                )
                depth_m = max(depth_m - residual / derivative, bankfull_depth_m)
            prev_flow_area_m2 = stream_width_m * depth_m + floodplain_width_m * (
                depth_m - bankfull_depth_m
            )

    rhs_constant: np.float32 = (
        dt_over_dx * inflow_m3_s
        + prev_flow_area_m2
        + timestep_s * lateral_inflow_m3_s_per_m
    )

    if has_floodplain and (
        dt_over_dx * bankfull_discharge_m3_s + stream_width_m * bankfull_depth_m
        < rhs_constant
    ):
        # The new discharge is above the banks. Solve for the depth:
        # dt/dx * Q(y) + A(y) = rhs. The function rises and is convex, and at
        # y = rhs / W the flow area alone exceeds rhs, so Newton from there
        # converges from above without overshooting.
        depth_m = rhs_constant / stream_width_m
        for _ in range(2 * MAX_ITERS_KINEMATIC):
            channel_discharge = (
                stream_width_m * depth_m / river_storage_alpha
            ) ** inv_beta
            over_bank_m = depth_m - bankfull_depth_m
            residual = (
                dt_over_dx
                * (channel_discharge + floodplain_conveyance * over_bank_m**five_thirds)
                + stream_width_m * depth_m
                + floodplain_width_m * over_bank_m
                - rhs_constant
            )
            if abs(residual) <= epsilon:
                break
            derivative = (
                dt_over_dx
                * (
                    inv_beta * channel_discharge / depth_m
                    + five_thirds * floodplain_conveyance * over_bank_m**two_thirds
                )
                + stream_width_m
                + floodplain_width_m
            )
            depth_m = max(depth_m - residual / derivative, bankfull_depth_m)
        over_bank_m = depth_m - bankfull_depth_m
        return (
            (stream_width_m * depth_m / river_storage_alpha) ** inv_beta
            + floodplain_conveyance * over_bank_m**five_thirds,
            actual_evaporation_m3_s,
            actual_transmission_loss_m3_s,
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

    return new_discharge, actual_evaporation_m3_s, actual_transmission_loss_m3_s


@njit(cache=NUMBA_CACHE, inline="always")
def _waterbody_release_over_timestep(
    wb: WaterbodyCoupling, wb_id: int, dt_f32: np.float32
) -> float:
    """Volume a waterbody releases over one routing timestep.

    Reservoirs (and other waterbodies with a prescribed outflow) release the
    prescribed volume, limited by their storage. Natural lakes (prescribed outflow
    NaN) release through their parabolic weir, Q = lake_factor * h^2, with h the
    head above the sill. Without inflow during the timestep, the lake level follows
    dh/dt = -(lake_factor / A) * h^2, so h(dt) = h0 / (1 + lake_factor * h0 * dt / A).
    This exact solution is used, so the release never exceeds the water above the
    sill, whatever the timestep.

    Args:
        wb: Waterbody coupling parameters and storage state tables.
        wb_id: Waterbody index.
        dt_f32: Routing timestep duration (seconds).

    Returns:
        Released volume (m³).
    """
    storage: float = max(wb.storage_m3[wb_id], 0.0)
    prescribed: np.float32 = wb.prescribed_outflow_m3[wb_id]
    if not np.isnan(prescribed):
        return min(max(float(prescribed), 0.0), storage)

    lake_area: float = float(wb.lake_area[wb_id])
    head_above_sill: float = storage / lake_area - float(wb.outflow_height[wb_id])
    if head_above_sill <= 0.0:
        return 0.0
    drawdown: float = (
        float(wb.lake_factor[wb_id]) * head_above_sill * float(dt_f32) / lake_area
    )
    return lake_area * head_above_sill * drawdown / (1.0 + drawdown)


@njit(cache=NUMBA_CACHE)
def _transfer_waterbody_outflows(
    wb: WaterbodyCoupling,
    kin: KinematicTopology,
    dt_f32: np.float32,
    terminal_wb_outflow_accum_m3: ArrayFloat32,
) -> None:
    """Releases waterbodies that do not drain into a local inertial reach.

    Waterbodies draining into another waterbody, into a kinematic reach or out of
    the domain release their water here, once per routing timestep and before the
    kinematic step, so that the release reaches the kinematic reaches in the same
    timestep. Waterbodies draining into an inertial reach are coupled to that reach
    within the inertial substeps instead. The prescribed outflow of the waterbodies
    handled here is set to 0 so they are not released again in the substeps.

    Args:
        wb: Waterbody coupling parameters, links, and storage state tables.
        kin: Kinematic wave network topology and reach tables.
        dt_f32: Routing timestep duration (seconds).
        terminal_wb_outflow_accum_m3: Single-element accumulator for releases of
            waterbodies draining out of the domain (m³).
    """
    # Waterbody-to-waterbody transfers. Releases are applied in link order, so an
    # upstream waterbody's release is part of the downstream waterbody's storage
    # when the latter is released.
    for idx in range(len(wb.wb_to_wb_src_wb)):
        src_wb: int = wb.wb_to_wb_src_wb[idx]
        tgt_wb: int = wb.wb_to_wb_tgt_wb[idx]
        release: float = _waterbody_release_over_timestep(wb, src_wb, dt_f32)
        wb.prescribed_outflow_m3[src_wb] = np.float32(0.0)
        if release > 0.0:
            wb.storage_m3[src_wb] -= release
            wb.outflow_m3[src_wb] += np.float32(release)
            wb.storage_m3[tgt_wb] += release
            wb.inflow_m3[tgt_wb] += np.float32(release)

    # Waterbody releases into receiving kinematic reaches.
    for idx in range(len(kin.wb_to_kin_target_reach)):
        tgt_reach: int = kin.wb_to_kin_target_reach[idx]
        wb_id_up: int = kin.wb_to_kin_wb_id[idx]
        release = _waterbody_release_over_timestep(wb, wb_id_up, dt_f32)
        wb.prescribed_outflow_m3[wb_id_up] = np.float32(0.0)
        if release > 0.0:
            wb.storage_m3[wb_id_up] -= release
            wb.outflow_m3[wb_id_up] += np.float32(release)
            kin.kin_wb_inflow_m3[tgt_reach] += np.float32(release)

    # Releases of waterbodies draining out of the domain.
    for idx in range(len(wb.wb_terminal_wb_ids)):
        terminal_waterbody_id: int = wb.wb_terminal_wb_ids[idx]
        release = _waterbody_release_over_timestep(wb, terminal_waterbody_id, dt_f32)
        wb.prescribed_outflow_m3[terminal_waterbody_id] = np.float32(0.0)
        if release > 0.0:
            wb.storage_m3[terminal_waterbody_id] -= release
            wb.outflow_m3[terminal_waterbody_id] += np.float32(release)
            terminal_wb_outflow_accum_m3[0] += np.float32(release)


# The retention arrays are passed one by one, not as the RetentionCoupling NamedTuple:
# in a numba prange body, a write through an array attribute of a NamedTuple argument
# with a loop-dependent index is silently lost (numba 0.68), so the parallel kernel
# released retention water to the river without emptying the basin.
@njit(inline="always")
def _solve_kinematic_reach(
    reach_idx: int,
    upstream_inflow_m3_s: np.float32,
    node_sideflow: np.float32,
    dt_f32: np.float32,
    inv_dt_f32: np.float32,
    previous_discharge: np.float32,
    evaporation_vol_m3: np.float32,
    transmission_loss_vol_m3: np.float32,
    river_storage_alpha: np.float32,
    river_storage_beta: np.float32,
    river_length: np.float32,
    stream_width_m: np.float32,
    bankfull_depth_m: np.float32,
    floodplain_width_m: np.float32,
    floodplain_conveyance: np.float32,
    river_storage_m3: ArrayFloat64,
    updated_discharge_m3_s: ArrayFloat32,
    actual_evaporation_m3: ArrayFloat32,
    actual_transmission_loss_m3: ArrayFloat32,
    over_abstraction_m3: ArrayFloat32,
    ret_id: int,
    ret_storage_m3: ArrayFloat32,
    ret_inflow_m3: ArrayFloat32,
    ret_outflow_m3: ArrayFloat32,
    ret_max_storage_m3: ArrayFloat32,
    ret_controlled: ArrayBool,
    ret_activation_threshold_m3_s: ArrayFloat32,
    ret_release_threshold_factor: np.float32,
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
        transmission_loss_vol_m3: Potential riverbed transmission loss volume for reach (m³).
        river_storage_alpha: Kinematic wave alpha parameter.
        river_storage_beta: Kinematic wave beta parameter.
        river_length: Channel reach length (meters).
        stream_width_m: Bankfull width of the channel (meters).
        bankfull_depth_m: Depth of the channel when full to the banks (meters).
        floodplain_width_m: Width of the valley floor next to the channel (meters).
        floodplain_conveyance: Conveyance coefficient of the valley floor (m^(4/3)/s).
        river_storage_m3: Storage volume array for all reaches (m³).
        updated_discharge_m3_s: Output array to store computed discharge (m³/s).
        actual_evaporation_m3: Output array to store realized evaporation volume (m³).
        actual_transmission_loss_m3: Output array to store realized transmission loss volume (m³).
        over_abstraction_m3: Output array accumulating abstraction volume that exceeded
            the water in the reach (m³).
        ret_id: Reach-to-retention basin mapping index (-1 if none).
        ret_storage_m3: Retention basin storage per basin (m³), updated in place.
        ret_inflow_m3: Diverted inflow accumulator per basin (m³), updated in place.
        ret_outflow_m3: Released outflow accumulator per basin (m³), updated in place.
        ret_max_storage_m3: Retention basin capacity per basin (m³).
        ret_controlled: Whether each basin is controlled (diverts on the rising limb only).
        ret_activation_threshold_m3_s: Diversion threshold discharge per basin (m³/s).
        ret_release_threshold_factor: Release threshold as a fraction of the activation threshold.

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
        inflow_limit: np.float32 = np.float32(0.20) * ret_max_storage_m3[ret_id]
        max_outflow_limit: np.float32 = np.float32(0.05) * ret_max_storage_m3[ret_id]
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
            retention_storage_m3=ret_storage_m3[ret_id],
            retention_max_storage_m3=ret_max_storage_m3[ret_id],
            controlled_retention=ret_controlled[ret_id],
            activation_threshold_m3_s=ret_activation_threshold_m3_s[ret_id],
            release_threshold_factor=ret_release_threshold_factor,
            inflow_limit_m3=inflow_limit,
            max_outflow_limit_m3=max_outflow_limit,
        )

        ret_storage_m3[ret_id] = new_ret_storage
        ret_inflow_m3[ret_id] += diverted_volume
        ret_outflow_m3[ret_id] += outflow_volume
        node_sideflow = discharge_at_basin_vol - upstream_inflow_m3_s * dt_f32

    # Solve kinematic wave routing for outflow discharge, channel evaporation, and transmission loss.
    kinematic_discharge, act_evap_rate, act_trans_rate = update_node_kinematic(
        inflow_m3_s=upstream_inflow_m3_s,
        previous_discharge_m3_s=previous_discharge,
        sideflow_m3_s=node_sideflow * inv_dt_f32,
        evaporation_m3_s=evaporation_vol_m3 * inv_dt_f32,
        transmission_loss_m3_s=transmission_loss_vol_m3 * inv_dt_f32,
        river_storage_alpha=river_storage_alpha,
        river_storage_beta=river_storage_beta,
        timestep_s=dt_f32,
        river_length_m=river_length,
        stream_width_m=stream_width_m,
        bankfull_depth_m=bankfull_depth_m,
        floodplain_width_m=floodplain_width_m,
        floodplain_conveyance=floodplain_conveyance,
    )

    kinematic_discharge = max(kinematic_discharge, np.float32(0.0))

    evap_vol: np.float32 = act_evap_rate * dt_f32
    trans_vol: np.float32 = act_trans_rate * dt_f32
    actual_evaporation_m3[reach_idx] = evap_vol
    actual_transmission_loss_m3[reach_idx] = trans_vol

    # Update reach storage (mass balance). The water that can leave the reach is
    # the stored water plus inflow (lateral inflow minus abstraction) minus losses.
    inflow_vol: np.float32 = upstream_inflow_m3_s * dt_f32 + node_sideflow
    available_volume_m3: np.float64 = (
        river_storage_m3[reach_idx]
        + np.float64(inflow_vol)
        - np.float64(evap_vol)
        - np.float64(trans_vol)
    )
    if available_volume_m3 < np.float64(0.0):
        # Abstraction exceeded the water in the reach: genuine over-abstraction.
        over_abstraction_m3[reach_idx] += np.float32(-available_volume_m3)
        river_storage_m3[reach_idx] = np.float64(0.0)
        kinematic_discharge = np.float32(0.0)
    else:
        # The kinematic outflow cannot exceed the water actually in the reach (the
        # kinematic storage relation can deviate from the tracked storage), so it
        # is limited instead of creating water.
        outflow_volume_m3: np.float64 = min(
            np.float64(kinematic_discharge * dt_f32), available_volume_m3
        )
        kinematic_discharge = np.float32(outflow_volume_m3) * inv_dt_f32
        river_storage_m3[reach_idx] = available_volume_m3 - outflow_volume_m3

    updated_discharge_m3_s[reach_idx] = kinematic_discharge
    return kinematic_discharge


def _run_kinematic_step(
    dt_f32: np.float32,
    inv_dt_f32: np.float32,
    previous_discharge_m3_s: ArrayFloat32,
    river_storage_m3: ArrayFloat64,
    sideflow_m3: ArrayFloat32,
    evaporation_m3: ArrayFloat32,
    transmission_loss_m3: ArrayFloat32,
    over_abstraction_m3: ArrayFloat32,
    actual_evaporation_m3: ArrayFloat32,
    actual_transmission_loss_m3: ArrayFloat32,
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
        transmission_loss_m3: Potential river transmission loss volume per reach (m³).
        over_abstraction_m3: Cumulative volume of over-abstraction (m³).
        actual_evaporation_m3: Output actual evaporation volume array to populate (m³).
        actual_transmission_loss_m3: Output actual transmission loss volume array to populate (m³).
        updated_discharge_m3_s: Output discharge array to populate (m³/s).
        kin: Kinematic wave network topology and reach parameter tables.
        wb: Waterbody coupling parameters, links, and storage state tables.
        ret: Flood retention basin state tables and control parameters.

    Raises:
        ValueError: If non-finite discharge is computed in any kinematic wave reach.
    """
    kin.kin_inflow_m3_s.fill(np.float32(0.0))
    # Arrays written inside the prange loop are taken out of the NamedTuples first:
    # numba loses prange writes through NamedTuple attributes (see _solve_kinematic_reach).
    kin_wb_inflow_m3: ArrayFloat32 = kin.kin_wb_inflow_m3
    ret_storage_m3: ArrayFloat32 = ret.storage_m3
    ret_inflow_m3: ArrayFloat32 = ret.inflow_m3
    ret_outflow_m3: ArrayFloat32 = ret.outflow_m3
    ret_max_storage_m3: ArrayFloat32 = ret.max_storage_m3
    ret_controlled: ArrayBool = ret.controlled_retention
    ret_activation_threshold_m3_s: ArrayFloat32 = ret.activation_threshold_m3_s
    has_wb_to_kin: bool = len(kin.wb_to_kin_target_reach) > 0
    n_headwater: int = kin.n_kin_headwater

    # Route headwater reaches in parallel across threads.
    # Headwaters have no upstream river reaches feeding them, making them mutually
    # independent. They can be computed concurrently without race conditions.
    for i in prange(n_headwater):
        # Collect local lateral runoff and waterbody releases (upstream river inflow is 0).
        node_sideflow: np.float32 = sideflow_m3[i]
        if has_wb_to_kin:
            node_sideflow += kin_wb_inflow_m3[i]
            kin_wb_inflow_m3[i] = np.float32(0.0)

        _solve_kinematic_reach(
            reach_idx=i,
            upstream_inflow_m3_s=np.float32(0.0),
            node_sideflow=node_sideflow,
            dt_f32=dt_f32,
            inv_dt_f32=inv_dt_f32,
            previous_discharge=previous_discharge_m3_s[i],
            evaporation_vol_m3=evaporation_m3[i],
            transmission_loss_vol_m3=transmission_loss_m3[i],
            river_storage_alpha=kin.river_storage_alpha[i],
            river_storage_beta=kin.river_storage_beta[i],
            river_length=kin.river_length[i],
            stream_width_m=kin.stream_width[i],
            bankfull_depth_m=kin.bankfull_depth[i],
            floodplain_width_m=kin.floodplain_width[i],
            floodplain_conveyance=kin.floodplain_conveyance[i],
            river_storage_m3=river_storage_m3,
            updated_discharge_m3_s=updated_discharge_m3_s,
            actual_evaporation_m3=actual_evaporation_m3,
            actual_transmission_loss_m3=actual_transmission_loss_m3,
            over_abstraction_m3=over_abstraction_m3,
            ret_id=int(kin.retention_node_id[i]),
            ret_storage_m3=ret_storage_m3,
            ret_inflow_m3=ret_inflow_m3,
            ret_outflow_m3=ret_outflow_m3,
            ret_max_storage_m3=ret_max_storage_m3,
            ret_controlled=ret_controlled,
            ret_activation_threshold_m3_s=ret_activation_threshold_m3_s,
            ret_release_threshold_factor=ret.release_threshold_factor,
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
    for i in range(n_headwater, kin.n_kinematic):
        node_sideflow: np.float32 = sideflow_m3[i]
        if has_wb_to_kin:
            node_sideflow += kin_wb_inflow_m3[i]
            kin_wb_inflow_m3[i] = np.float32(0.0)

        kinematic_discharge: np.float32 = _solve_kinematic_reach(
            reach_idx=i,
            upstream_inflow_m3_s=kin.kin_inflow_m3_s[i],
            node_sideflow=node_sideflow,
            dt_f32=dt_f32,
            inv_dt_f32=inv_dt_f32,
            previous_discharge=previous_discharge_m3_s[i],
            evaporation_vol_m3=evaporation_m3[i],
            transmission_loss_vol_m3=transmission_loss_m3[i],
            river_storage_alpha=kin.river_storage_alpha[i],
            river_storage_beta=kin.river_storage_beta[i],
            river_length=kin.river_length[i],
            stream_width_m=kin.stream_width[i],
            bankfull_depth_m=kin.bankfull_depth[i],
            floodplain_width_m=kin.floodplain_width[i],
            floodplain_conveyance=kin.floodplain_conveyance[i],
            river_storage_m3=river_storage_m3,
            updated_discharge_m3_s=updated_discharge_m3_s,
            actual_evaporation_m3=actual_evaporation_m3,
            actual_transmission_loss_m3=actual_transmission_loss_m3,
            over_abstraction_m3=over_abstraction_m3,
            ret_id=int(kin.retention_node_id[i]),
            ret_storage_m3=ret_storage_m3,
            ret_inflow_m3=ret_inflow_m3,
            ret_outflow_m3=ret_outflow_m3,
            ret_max_storage_m3=ret_max_storage_m3,
            ret_controlled=ret_controlled,
            ret_activation_threshold_m3_s=ret_activation_threshold_m3_s,
            ret_release_threshold_factor=ret.release_threshold_factor,
        )

        wb_ds_id: int = kin.kin_ds_waterbody_id[i]
        if wb_ds_id != -1:
            outflow_volume_m3: np.float32 = kinematic_discharge * dt_f32
            wb.storage_m3[wb_ds_id] += np.float64(outflow_volume_m3)
            if outflow_volume_m3 > np.float32(0.0):
                wb.inflow_m3[wb_ds_id] += outflow_volume_m3

        if __debug__:
            if not np.isfinite(kinematic_discharge):
                raise ValueError(
                    "Non-finite discharge computed in kinematic wave reach."
                )

        ds_reach: int = kin.kin_ds_reach[i]
        if ds_reach != -1:
            kin.kin_inflow_m3_s[ds_reach] += kinematic_discharge


_run_kinematic_step_serial = njit(parallel=False, cache=NUMBA_CACHE)(
    _run_kinematic_step
)
_run_kinematic_step_parallel = njit(parallel=True, cache=NUMBA_CACHE)(
    _run_kinematic_step
)
