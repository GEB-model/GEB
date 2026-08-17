"""1D local inertial river routing algorithm for channel networks.

Implements a hydrodynamically simplified 1D local inertial routing formulation
(neglecting convective acceleration) integrated with kinematic wave sections,
retention basins, and waterbody (lakes/reservoirs) dynamics.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
from numba import njit

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    TwoDArrayInt32,
)

MAX_ITERS: int = 10


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
    evaporation_m3_s_per_m: np.float32 = evaporation_m3_s / river_length_m

    lateral_inflow_m3_s_per_m: np.float32 = sideflow_m3_s / river_length_m

    # If evaporation is larger than inflow and sideflow, limit it to sum of inflow and sideflow
    evaporation_m3_s_per_m = min(
        evaporation_m3_s_per_m,
        (inflow_m3_s + previous_discharge_m3_s) / 2 + max(lateral_inflow_m3_s_per_m, 0),
    )

    lateral_inflow_m3_s_per_m -= evaporation_m3_s_per_m

    actual_evaporation_m3_s: np.float32 = evaporation_m3_s_per_m * river_length_m

    # If there is no inflow, no previous flow, and no lateral inflow,
    # then the discharge at the new time step will be zero.
    if (inflow_m3_s + previous_discharge_m3_s + lateral_inflow_m3_s_per_m) < 1e-30:
        return np.float32(1e-30), actual_evaporation_m3_s

    inflow_m3_s = max(inflow_m3_s, np.float32(1e-30))

    # Common terms
    ab_pQ: np.float32 = (
        river_storage_alpha
        * river_storage_beta
        * ((previous_discharge_m3_s + inflow_m3_s) / 2) ** (river_storage_beta - 1)
    )
    delta_t_over_delta_x: np.float32 = timestep_s / river_length_m
    C: np.float32 = (
        delta_t_over_delta_x * inflow_m3_s
        + river_storage_alpha * previous_discharge_m3_s**river_storage_beta
        + timestep_s * lateral_inflow_m3_s_per_m
    )

    # Initial guess for new discharge and iterative process
    new_discharge: np.float32 = (
        delta_t_over_delta_x * inflow_m3_s
        + previous_discharge_m3_s * ab_pQ
        + timestep_s * lateral_inflow_m3_s_per_m
    ) / (delta_t_over_delta_x + ab_pQ)
    new_discharge = max(new_discharge, np.float32(1e-30))

    # Newton-Raphson method
    count: int = 0
    f_q: np.float32 = (
        delta_t_over_delta_x * new_discharge
        + river_storage_alpha * new_discharge**river_storage_beta
        - C
    )

    while np.abs(f_q) > epsilon and count < MAX_ITERS:
        df_q: np.float32 = (
            delta_t_over_delta_x
            + river_storage_alpha
            * river_storage_beta
            * new_discharge ** (river_storage_beta - 1)
        )
        new_discharge -= f_q / df_q
        new_discharge = max(new_discharge, np.float32(1e-30))

        # Update f_q for next iteration check
        f_q = (
            delta_t_over_delta_x * new_discharge
            + river_storage_alpha * new_discharge**river_storage_beta
            - C
        )
        count += 1

    return new_discharge, actual_evaporation_m3_s


from .routerbase import Router, compute_retention_routing


@njit(parallel=False, cache=True, fastmath=True, inline="always")
def _run_single_inertial_substep(
    dt_substep: np.float32,
    n_kinematic: int,
    n_inertial: int,
    max_up_connections: int,
    substep_discharge_in: ArrayFloat32,
    river_storage_in: ArrayFloat64,
    waterbody_storage_in: ArrayFloat64,
    retention_storage_in: ArrayFloat32,
    over_abstraction_in: ArrayFloat32,
    actual_evaporation_in: ArrayFloat32,
    waterbody_inflow_in: ArrayFloat32,
    retention_inflow_in: ArrayFloat32,
    retention_outflow_in: ArrayFloat32,
    discharge_vol_sum_in: ArrayFloat64,
    wb_processed_in: ArrayBool,
    substep_discharge_out: ArrayFloat32,
    river_storage_out: ArrayFloat64,
    waterbody_storage_out: ArrayFloat64,
    retention_storage_out: ArrayFloat32,
    over_abstraction_out: ArrayFloat32,
    actual_evaporation_out: ArrayFloat32,
    waterbody_inflow_out: ArrayFloat32,
    retention_inflow_out: ArrayFloat32,
    retention_outflow_out: ArrayFloat32,
    discharge_vol_sum_out: ArrayFloat64,
    wb_processed_out: ArrayBool,
    sideflow_substep_m3: ArrayFloat64,
    evap_substep_m3: ArrayFloat64,
    wb_outflow_substep_m3: ArrayFloat64,
    retention_inflow_limit_sub: ArrayFloat64,
    retention_max_outflow_limit_sub: ArrayFloat64,
    wb_outflow_avail_buf: ArrayFloat64,
    wb_extra_lateral_m3_buf: ArrayFloat64,
    net_vol_m3: ArrayFloat64,
    upstream_matrix: TwoDArrayInt32,
    is_waterbody_outflow: ArrayBool,
    waterbody_id: ArrayInt32,
    river_length: ArrayFloat32,
    inv_reach_length: ArrayFloat32,
    inv_cell_area: ArrayFloat32,
    river_width: ArrayFloat32,
    bed_elevation: ArrayFloat32,
    g_manning_n_sq: ArrayFloat32,
    retention_max_storage_m3: ArrayFloat32,
    retention_node_id: ArrayInt32,
    controlled_retention: ArrayBool,
    retention_activation_threshold_m3_s: ArrayFloat32,
    retention_basin_release_threshold_factor: np.float32,
    ds_node: ArrayInt32,
    is_pit: ArrayBool,
    is_ocean_pit: ArrayBool,
    pit_slope: ArrayFloat32,
    use_kinematic: ArrayBool,
    lateral_flux_m3: ArrayFloat32,
    next_substep_discharge_m3_s: ArrayFloat32,
    kinematic_inflow_rate: ArrayFloat32,
    previous_discharge_m3_s: ArrayFloat32,
    reset_counters: ArrayInt32,
) -> None:
    """Executes a single numerical substep for the local inertial routing domain.

    Args:
        dt_substep: Time duration of the current substep [s].
        n_kinematic: Number of nodes designated for kinematic wave routing.
        n_inertial: Number of nodes designated for local inertial wave routing.
        max_up_connections: Maximum number of upstream node connections per cell.
        substep_discharge_in: Input array of channel discharges at the start of the substep [m³/s].
        river_storage_in: Input array of channel volumes at the start of the substep [m³].
        waterbody_storage_in: Input array of waterbody storage volumes [m³].
        retention_storage_in: Input array of retention basin storage volumes [m³].
        over_abstraction_in: Accumulated mass deficit from unfulfilled channel abstractions [m³].
        actual_evaporation_in: Accumulated actual evaporation volume [m³].
        waterbody_inflow_in: Accumulated volumetric inflow to waterbodies [m³].
        retention_inflow_in: Accumulated volumetric diversion into retention basins [m³].
        retention_outflow_in: Accumulated volumetric discharge released from retention basins [m³].
        discharge_vol_sum_in: Accumulated discharge volume sum over substeps [m³].
        wb_processed_in: Boolean array indicating whether waterbodies were processed.
        substep_discharge_out: Output array for calculated channel discharges [m³/s].
        river_storage_out: Output array for updated channel storage volumes [m³].
        waterbody_storage_out: Output array for updated waterbody volumes [m³].
        retention_storage_out: Output array for updated retention basin volumes [m³].
        over_abstraction_out: Output array for updated over-abstraction volumes [m³].
        actual_evaporation_out: Output array for updated actual evaporation volumes [m³].
        waterbody_inflow_out: Output array for updated waterbody inflow volumes [m³].
        retention_inflow_out: Output array for updated retention inflow volumes [m³].
        retention_outflow_out: Output array for updated retention outflow volumes [m³].
        discharge_vol_sum_out: Output array for updated discharge volume sums [m³].
        wb_processed_out: Output array for updated waterbody processed flags.
        sideflow_substep_m3: Substep lateral sideflow volume for inertial nodes [m³].
        evap_substep_m3: Substep potential evaporation volume for inertial nodes [m³].
        wb_outflow_substep_m3: Substep available waterbody outflow volume [m³].
        retention_inflow_limit_sub: Maximum allowable substep inflow to retention basins [m³].
        retention_max_outflow_limit_sub: Maximum allowable substep outflow from retention basins [m³].
        wb_outflow_avail_buf: Buffer array tracking available waterbody outflow [m³].
        wb_extra_lateral_m3_buf: Buffer tracking extra lateral volume from waterbodies [m³].
        net_vol_m3: Buffer array storing net volume changes per reach [m³].
        upstream_matrix: Matrix specifying 0-indexed topological upstream dependencies.
        is_waterbody_outflow: Boolean mask indicating waterbody outlet nodes.
        waterbody_id: Array of waterbody identifiers (-1 for non-waterbody nodes).
        river_length: Channel reach length [m].
        inv_reach_length: Precomputed inverse of reach length [1/m].
        inv_cell_area: Precomputed inverse of cell surface area [1/m²].
        river_width: Channel width [m].
        bed_elevation: Channel bed elevation above datum [m].
        g_manning_n_sq: Precalculated term $g \\cdot n^2$ where $g$ is gravity and $n$ is Manning's coefficient.
        retention_max_storage_m3: Maximum storage capacity of retention basins [m³].
        retention_node_id: Identifier map for retention basin nodes.
        controlled_retention: Boolean flag indicating if retention basin release is active.
        retention_activation_threshold_m3_s: Discharge threshold triggering basin diversion [m³/s].
        retention_basin_release_threshold_factor: Fractional factor determining release behavior.
        ds_node: Index map for downstream target nodes.
        is_pit: Boolean mask indicating domain outlets/pits.
        is_ocean_pit: Boolean mask specifying terminal sea/ocean outlet nodes.
        pit_slope: Bed slope applied at boundary terminal nodes [-].
        use_kinematic: Boolean flag specifying kinematic wave formulation per node.
        lateral_flux_m3: Buffer array for resolved substep lateral fluxes [m³].
        next_substep_discharge_m3_s: Buffer array storing solved forward substep discharges [m³/s].
        kinematic_inflow_rate: Upstream inflow rate sourced from kinematic reaches [m³/s].
        previous_discharge_m3_s: Discharge array from the preceding global timestep [m³/s].
        reset_counters: Diagnostic counter array tracking low storage or drying resets.

    Raises:
        ValueError: If non-finite values are encountered in flow depth, cross-sectional area, or computed discharge during the substep.
    """
    min_wet_depth_m = np.float32(1e-4)
    gravity_acceleration = np.float32(9.80665)
    sqrt_gravity = np.float32(3.1315574)

    inv_dt_substep = np.float32(1.0) / dt_substep
    g_dt_substep = gravity_acceleration * dt_substep

    # Copy state variables to target double buffers
    river_storage_out[:] = river_storage_in[:]
    over_abstraction_out[:] = over_abstraction_in[:]
    actual_evaporation_out[:] = actual_evaporation_in[:]
    substep_discharge_out[:] = substep_discharge_in[:]
    discharge_vol_sum_out[:] = discharge_vol_sum_in[:]

    waterbody_storage_out[:] = waterbody_storage_in[:]
    retention_storage_out[:] = retention_storage_in[:]
    waterbody_inflow_out[:] = waterbody_inflow_in[:]
    retention_inflow_out[:] = retention_inflow_in[:]
    retention_outflow_out[:] = retention_outflow_in[:]
    wb_processed_out[:] = wb_processed_in[:]

    lateral_flux_m3.fill(np.float32(0.0))
    next_substep_discharge_m3_s.fill(np.float32(0.0))
    net_vol_m3.fill(0.0)

    wb_extra_lateral_m3_buf.fill(0.0)
    wb_outflow_avail_buf[:] = wb_outflow_substep_m3[:]

    # 1. Upstream waterbody release loop
    for k in range(n_inertial):
        i = n_kinematic + k
        for j in range(max_up_connections):
            up_node = upstream_matrix[i, j]
            if up_node == -1:
                break
            if is_waterbody_outflow[up_node]:
                wb_id = waterbody_id[up_node]
                if wb_id != -1 and wb_outflow_avail_buf[wb_id] > 0.0:
                    waterbody_storage = max(waterbody_storage_out[wb_id], 0.0)
                    actual_outflow = min(wb_outflow_avail_buf[wb_id], waterbody_storage)
                    if actual_outflow > 0.0:
                        waterbody_storage_out[wb_id] -= actual_outflow
                        wb_extra_lateral_m3_buf[k] += actual_outflow
                    wb_outflow_avail_buf[wb_id] = 0.0
                    wb_processed_out[wb_id] = True

    # 2. Hydraulic discharge routing & momentum solver
    for k in range(n_inertial):
        i = n_kinematic + k
        sideflow_substep = sideflow_substep_m3[k]
        if not np.isfinite(sideflow_substep):
            sideflow_substep = 0.0

        node_lateral_inflow = np.float32(sideflow_substep + wb_extra_lateral_m3_buf[k])

        ret_id = retention_node_id[i]
        if ret_id != -1:
            discharge_in_positive = max(substep_discharge_in[k], np.float32(0.0))
            avail_flow_rate = (
                node_lateral_inflow * inv_dt_substep
            ) + discharge_in_positive
            inflow_limit = retention_inflow_limit_sub[ret_id]
            max_outflow_limit = retention_max_outflow_limit_sub[ret_id]
            river_volume_substep = avail_flow_rate * dt_substep

            prev_discharge_i = previous_discharge_m3_s[i]
            if not np.isfinite(prev_discharge_i):
                prev_discharge_i = np.float32(0.0)
            is_rising_limb = avail_flow_rate > prev_discharge_i

            (
                diverted_vol,
                outflow_vol,
                retention_storage_out[ret_id],
                river_volume_substep,
            ) = compute_retention_routing(
                dt=dt_substep,
                river_volume_m3=river_volume_substep,
                discharge_before_diversion_m3_s=avail_flow_rate,
                is_rising_limb=bool(is_rising_limb),
                retention_storage_m3=retention_storage_in[ret_id],
                retention_max_storage_m3=retention_max_storage_m3[ret_id],
                controlled_retention=controlled_retention[ret_id],
                activation_threshold_m3_s=retention_activation_threshold_m3_s[ret_id],
                release_threshold_factor=retention_basin_release_threshold_factor,
                inflow_limit_m3=inflow_limit,
                max_outflow_limit_m3=max_outflow_limit,
            )

            retention_inflow_out[ret_id] = retention_inflow_in[ret_id] + diverted_vol
            retention_outflow_out[ret_id] = retention_outflow_in[ret_id] + outflow_vol
            node_lateral_inflow = river_volume_substep - (
                discharge_in_positive * dt_substep
            )

        lateral_flux_m3[k] = node_lateral_inflow

        bed_elev_node = bed_elevation[i]
        width_m = river_width[i]
        inv_length_m = inv_reach_length[i]
        inv_area_m2 = inv_cell_area[i]

        curr_storage = river_storage_in[k]
        if not np.isfinite(curr_storage):
            curr_storage = 0.0

        available_vol_f32 = max(
            np.float32(curr_storage) + node_lateral_inflow, np.float32(0.0)
        )
        flow_depth = max(available_vol_f32 * inv_area_m2, np.float32(0.0))

        if not np.isfinite(flow_depth):
            raise ValueError(
                "Non-finite flow_depth encountered in local inertial substep."
            )

        water_stage_node = bed_elev_node + flow_depth
        ds = ds_node[i]

        if is_pit[i]:
            if is_ocean_pit[i]:
                water_stage_ds = np.float32(0.0)
                bed_elev_ds = bed_elev_node
            else:
                reach_len = river_length[i]
                effective_slope = pit_slope[i]
                bed_elev_ds = bed_elev_node - effective_slope * reach_len
                water_stage_ds = bed_elev_ds + flow_depth
        elif ds != -1:
            bed_elev_ds = bed_elevation[ds]
            wb_ds = waterbody_id[ds]
            if wb_ds == -1:
                ds_k = ds - n_kinematic
                ds_storage = river_storage_in[ds_k]
                if not np.isfinite(ds_storage):
                    ds_storage = 0.0
                flow_depth_ds = max(
                    np.float32(ds_storage) * inv_cell_area[ds],
                    np.float32(0.0),
                )
                water_stage_ds = bed_elev_ds + flow_depth_ds
            else:
                water_stage_ds = bed_elev_ds
        else:
            bed_elev_ds = bed_elev_node
            water_stage_ds = bed_elev_node

        max_stage = max(water_stage_node, water_stage_ds)
        max_bed = max(bed_elev_node, bed_elev_ds)
        effective_depth = max(max_stage - max_bed, np.float32(0.0))

        if (effective_depth < min_wet_depth_m) or (
            available_vol_f32 <= np.float32(0.0)
        ):
            next_substep_discharge_m3_s[k] = np.float32(0.0)
            reset_counters[1] += 1
            continue

        cross_sectional_area = effective_depth * width_m

        if (not np.isfinite(cross_sectional_area)) or (
            cross_sectional_area >= np.float32(1e6)
        ):
            raise ValueError(
                "Invalid cross_sectional_area (non-finite or >= 1e6 m2) in local inertial substep."
            )

        hydraulic_radius = max(effective_depth, np.float32(1e-6))
        water_slope = (water_stage_ds - water_stage_node) * inv_length_m

        r_4_3 = hydraulic_radius * np.cbrt(hydraulic_radius)
        friction_denom_term = max(r_4_3 * cross_sectional_area, np.float32(1e-6))

        discharge_in = substep_discharge_in[k]
        if not np.isfinite(discharge_in):
            discharge_in = np.float32(0.0)

        friction_factor_val = dt_substep * g_manning_n_sq[i]
        friction_denom = np.float32(1.0) + (
            friction_factor_val * abs(discharge_in) / friction_denom_term
        )

        computed_discharge = (
            discharge_in - g_dt_substep * cross_sectional_area * water_slope
        ) / friction_denom

        is_valid_ds = (ds != -1) and (not is_pit[i])
        is_boundary = (
            (ds == -1) or is_pit[i] or (is_valid_ds and waterbody_id[ds] != -1)
        )
        if is_boundary:
            computed_discharge = max(computed_discharge, np.float32(0.0))

        critical_discharge = (
            cross_sectional_area * sqrt_gravity * np.sqrt(effective_depth)
        )
        computed_discharge = min(
            max(computed_discharge, -critical_discharge), critical_discharge
        )

        if computed_discharge > np.float32(0.0):
            max_discharge_avail = available_vol_f32 * inv_dt_substep
            computed_discharge = min(computed_discharge, max_discharge_avail)
        elif computed_discharge < np.float32(0.0):
            if is_valid_ds and not is_pit[ds]:
                ds_k = ds - n_kinematic
                ds_storage = river_storage_in[ds_k]
                if not np.isfinite(ds_storage):
                    ds_storage = 0.0
                max_discharge_avail = (
                    max(np.float32(ds_storage), np.float32(0.0)) * inv_dt_substep
                )
                computed_discharge = -min(abs(computed_discharge), max_discharge_avail)
            else:
                computed_discharge = np.float32(0.0)

        if not np.isfinite(computed_discharge):
            raise ValueError(
                "Non-finite computed_discharge encountered in local inertial substep."
            )

        next_substep_discharge_m3_s[k] = computed_discharge

    # 3. Continuity equation & storage updates
    for k in range(n_inertial):
        i = n_kinematic + k
        ds = ds_node[i]
        if ds != -1 and not is_pit[i]:
            wb_ds_id = waterbody_id[ds]
            if wb_ds_id != -1:
                vol_out = next_substep_discharge_m3_s[k] * dt_substep
                waterbody_storage_out[wb_ds_id] += np.float64(vol_out)
                if vol_out > np.float32(0.0):
                    waterbody_inflow_out[wb_ds_id] += vol_out

        vol_in = kinematic_inflow_rate[k] * dt_substep + lateral_flux_m3[k]

        for j in range(max_up_connections):
            up_node = upstream_matrix[i, j]
            if up_node == -1:
                break

            if waterbody_id[up_node] == -1 and not use_kinematic[up_node]:
                up_k = up_node - n_kinematic
                vol_in += next_substep_discharge_m3_s[up_k] * dt_substep

        vol_out = next_substep_discharge_m3_s[k] * dt_substep
        net_vol = np.float64(vol_in) - np.float64(vol_out)
        net_vol_m3[k] = net_vol
        discharge_vol_sum_out[k] = discharge_vol_sum_in[k] + np.float64(vol_out)

        storage = river_storage_in[k] + net_vol

        if not np.isfinite(storage):
            raise ValueError(
                "Non-finite river storage encountered in local inertial substep."
            )

        over_abs = over_abstraction_in[k]
        if storage < 0.0:
            over_abs += np.float32(-storage)
            storage = 0.0
            reset_counters[0] += 1
        over_abstraction_out[k] = over_abs

        evap_substep = evap_substep_m3[k]
        if np.isfinite(evap_substep) and evap_substep > 0.0 and storage > 0.0:
            evap_actual = np.float32(min(evap_substep, storage))
        else:
            evap_actual = np.float32(0.0)

        actual_evaporation_out[k] = actual_evaporation_in[k] + evap_actual
        storage -= np.float64(evap_actual)

        river_storage_out[k] = storage
        substep_discharge_out[k] = next_substep_discharge_m3_s[k]


@njit(cache=True)
def _run_inertial_substeps(
    num_inertial_substeps: int,
    dt_f32: np.float32,
    n_kinematic: int,
    n_inertial: int,
    max_up_connections: int,
    previous_discharge_m3_s: ArrayFloat32,
    river_storage_m3_inertial: ArrayFloat64,
    sideflow_m3_inertial: ArrayFloat32,
    evaporation_m3_inertial: ArrayFloat32,
    waterbody_storage_m3: ArrayFloat64,
    outflow_per_waterbody_m3: ArrayFloat32,
    upstream_matrix: TwoDArrayInt32,
    is_waterbody_outflow: ArrayBool,
    waterbody_id: ArrayInt32,
    river_length: ArrayFloat32,
    inv_reach_length: ArrayFloat32,
    inv_cell_area: ArrayFloat32,
    river_width: ArrayFloat32,
    bed_elevation: ArrayFloat32,
    manning_n_sq: ArrayFloat32,
    retention_storage_m3: ArrayFloat32,
    retention_max_storage_m3: ArrayFloat32,
    retention_node_id: ArrayInt32,
    controlled_retention: ArrayBool,
    retention_activation_threshold_m3_s: ArrayFloat32,
    retention_basin_release_threshold_factor: np.float32,
    ds_node: ArrayInt32,
    is_pit: ArrayBool,
    is_ocean_pit: ArrayBool,
    pit_slope: ArrayFloat32,
    use_kinematic: ArrayBool,
    over_abstraction_m3_inertial: ArrayFloat32,
    actual_evaporation_m3_inertial: ArrayFloat32,
    waterbody_inflow_m3: ArrayFloat32,
    retention_inflow_m3: ArrayFloat32,
    retention_outflow_m3: ArrayFloat32,
    updated_discharge_m3_s_inertial: ArrayFloat32,
    lateral_flux_m3: ArrayFloat32,
    next_substep_discharge_m3_s: ArrayFloat32,
    kinematic_inflow_rate: ArrayFloat32,
    discharge_vol_sum_m3_inertial: ArrayFloat64,
    sideflow_substep_m3_buf: ArrayFloat64,
    evap_substep_m3_buf: ArrayFloat64,
    wb_outflow_substep_m3_buf: ArrayFloat64,
    retention_inflow_limit_sub_buf: ArrayFloat64,
    retention_max_outflow_limit_sub_buf: ArrayFloat64,
    wb_outflow_avail_buf: ArrayFloat64,
    wb_extra_lateral_m3_buf: ArrayFloat64,
    net_vol_m3_buf: ArrayFloat64,
    river_storage_inertial_A: ArrayFloat64,
    waterbody_storage_A: ArrayFloat64,
    retention_storage_A: ArrayFloat32,
    over_abstraction_A: ArrayFloat32,
    actual_evaporation_A: ArrayFloat32,
    waterbody_inflow_A: ArrayFloat32,
    retention_inflow_A: ArrayFloat32,
    retention_outflow_A: ArrayFloat32,
    substep_discharge_A: ArrayFloat32,
    discharge_vol_sum_A: ArrayFloat64,
    wb_processed_A: ArrayBool,
    river_storage_inertial_B: ArrayFloat64,
    waterbody_storage_B: ArrayFloat64,
    retention_storage_B: ArrayFloat32,
    over_abstraction_B: ArrayFloat32,
    actual_evaporation_B: ArrayFloat32,
    waterbody_inflow_B: ArrayFloat32,
    retention_inflow_B: ArrayFloat32,
    retention_outflow_B: ArrayFloat32,
    substep_discharge_B: ArrayFloat32,
    discharge_vol_sum_B: ArrayFloat64,
    wb_processed_B: ArrayBool,
) -> None:
    """Manages adaptive substepping loops for inertial wave routing.

    Args:
        num_inertial_substeps: Total number of sub-iterations to divide the main step into.
        dt_f32: Outer global timestep length [s].
        n_kinematic: Number of nodes designated for kinematic wave routing.
        n_inertial: Number of nodes designated for local inertial wave routing.
        max_up_connections: Maximum number of upstream node connections per cell.
        previous_discharge_m3_s: Initial channel discharges at the start of the timestep [m³/s].
        river_storage_m3_inertial: Channel storage array for inertial reaches [m³].
        sideflow_m3_inertial: Total timestep lateral inflow for inertial reaches [m³].
        evaporation_m3_inertial: Total timestep potential evaporation for inertial reaches [m³].
        waterbody_storage_m3: Waterbody storage array [m³].
        outflow_per_waterbody_m3: Prescribed or calculated waterbody outflows [m³].
        upstream_matrix: Matrix specifying 0-indexed topological upstream dependencies.
        is_waterbody_outflow: Boolean mask indicating waterbody outlet nodes.
        waterbody_id: Array of waterbody identifiers (-1 for non-waterbody nodes).
        river_length: Channel reach length [m].
        inv_reach_length: Precomputed inverse reach length [1/m].
        inv_cell_area: Precomputed inverse cell surface area [1/m²].
        river_width: Channel width [m].
        bed_elevation: Channel bed elevation above datum [m].
        manning_n_sq: Squared Manning's roughness coefficient ($n^2$).
        retention_storage_m3: Retention basin storage array [m³].
        retention_max_storage_m3: Maximum storage capacity of retention basins [m³].
        retention_node_id: Identifier map for retention basin nodes.
        controlled_retention: Boolean flag indicating if retention basin release is active.
        retention_activation_threshold_m3_s: Discharge threshold triggering basin diversion [m³/s].
        retention_basin_release_threshold_factor: Fractional factor determining release behavior.
        ds_node: Index map for downstream target nodes.
        is_pit: Boolean mask indicating domain outlets/pits.
        is_ocean_pit: Boolean mask specifying terminal sea/ocean outlet nodes.
        pit_slope: Bed slope applied at boundary terminal nodes [-].
        use_kinematic: Boolean flag specifying kinematic wave formulation per node.
        over_abstraction_m3_inertial: Array tracking unfulfilled abstractions in inertial reaches [m³].
        actual_evaporation_m3_inertial: Array tracking actual evaporation in inertial reaches [m³].
        waterbody_inflow_m3: Accumulated volumetric inflow to waterbodies [m³].
        retention_inflow_m3: Accumulated volumetric diversion into retention basins [m³].
        retention_outflow_m3: Accumulated volumetric discharge released from retention basins [m³].
        updated_discharge_m3_s_inertial: Output array storing calculated average inertial discharges [m³/s].
        lateral_flux_m3: Buffer array for resolved substep lateral fluxes [m³].
        next_substep_discharge_m3_s: Buffer array storing solved forward substep discharges [m³/s].
        kinematic_inflow_rate: Upstream inflow rate sourced from kinematic reaches [m³/s].
        discharge_vol_sum_m3_inertial: Buffer tracking cumulative discharge volume per reach [m³].
        sideflow_substep_m3_buf: Allocated buffer for substep sideflow volume [m³].
        evap_substep_m3_buf: Allocated buffer for substep evaporation volume [m³].
        wb_outflow_substep_m3_buf: Allocated buffer for substep waterbody outflow volume [m³].
        retention_inflow_limit_sub_buf: Allocated buffer for substep retention inflow limit [m³].
        retention_max_outflow_limit_sub_buf: Allocated buffer for substep retention outflow limit [m³].
        wb_outflow_avail_buf: Allocated workspace tracking available waterbody outflow [m³].
        wb_extra_lateral_m3_buf: Workspace tracking extra lateral volume from waterbodies [m³].
        net_vol_m3_buf: Workspace array for net reach volume changes [m³].
        river_storage_inertial_A: Double-buffering workspace A for river storage [m³].
        waterbody_storage_A: Double-buffering workspace A for waterbody storage [m³].
        retention_storage_A: Double-buffering workspace A for retention storage [m³].
        over_abstraction_A: Double-buffering workspace A for over-abstraction [m³].
        actual_evaporation_A: Double-buffering workspace A for actual evaporation [m³].
        waterbody_inflow_A: Double-buffering workspace A for waterbody inflow [m³].
        retention_inflow_A: Double-buffering workspace A for retention inflow [m³].
        retention_outflow_A: Double-buffering workspace A for retention outflow [m³].
        substep_discharge_A: Double-buffering workspace A for substep discharge [m³/s].
        discharge_vol_sum_A: Double-buffering workspace A for cumulative discharge volume [m³].
        wb_processed_A: Double-buffering workspace A for waterbody processing flags.
        river_storage_inertial_B: Double-buffering workspace B for river storage [m³].
        waterbody_storage_B: Double-buffering workspace B for waterbody storage [m³].
        retention_storage_B: Double-buffering workspace B for retention storage [m³].
        over_abstraction_B: Double-buffering workspace B for over-abstraction [m³].
        actual_evaporation_B: Double-buffering workspace B for actual evaporation [m³].
        waterbody_inflow_B: Double-buffering workspace B for waterbody inflow [m³].
        retention_inflow_B: Double-buffering workspace B for retention inflow [m³].
        retention_outflow_B: Double-buffering workspace B for retention outflow [m³].
        substep_discharge_B: Double-buffering workspace B for substep discharge [m³/s].
        discharge_vol_sum_B: Double-buffering workspace B for cumulative discharge volume [m³].
        wb_processed_B: Double-buffering workspace B for waterbody processing flags.
    """
    dt_f64 = np.float64(dt_f32)
    substep_dt_f64 = dt_f64 / np.float64(num_inertial_substeps)
    substep_dt_f32 = np.float32(substep_dt_f64)
    fraction_f64 = substep_dt_f64 / dt_f64

    g_manning_n_sq = np.float32(9.80665) * manning_n_sq

    reset_counters = np.zeros(3, dtype=np.int32)

    # Initialize double-buffer workspace arrays
    river_storage_inertial_A[:] = river_storage_m3_inertial[:]
    river_storage_inertial_B[:] = river_storage_m3_inertial[:]

    waterbody_storage_A[:] = waterbody_storage_m3[:]
    waterbody_storage_B[:] = waterbody_storage_m3[:]

    retention_storage_A[:] = retention_storage_m3[:]
    retention_storage_B[:] = retention_storage_m3[:]

    over_abstraction_A[:] = over_abstraction_m3_inertial[:]
    over_abstraction_B[:] = over_abstraction_m3_inertial[:]

    actual_evaporation_A[:] = actual_evaporation_m3_inertial[:]
    actual_evaporation_B[:] = actual_evaporation_m3_inertial[:]

    waterbody_inflow_A[:] = waterbody_inflow_m3[:]
    waterbody_inflow_B[:] = waterbody_inflow_m3[:]

    retention_inflow_A[:] = retention_inflow_m3[:]
    retention_inflow_B[:] = retention_inflow_m3[:]

    retention_outflow_A[:] = retention_outflow_m3[:]
    retention_outflow_B[:] = retention_outflow_m3[:]

    substep_discharge_A[:] = previous_discharge_m3_s[
        n_kinematic : n_kinematic + n_inertial
    ]
    substep_discharge_B[:] = previous_discharge_m3_s[
        n_kinematic : n_kinematic + n_inertial
    ]

    discharge_vol_sum_A.fill(0.0)
    discharge_vol_sum_B.fill(0.0)

    wb_processed_A.fill(False)
    wb_processed_B.fill(False)

    curr_river_storage = river_storage_inertial_A
    curr_waterbody_storage = waterbody_storage_A
    curr_retention_storage = retention_storage_A
    curr_over_abstraction = over_abstraction_A
    curr_actual_evaporation = actual_evaporation_A
    curr_waterbody_inflow = waterbody_inflow_A
    curr_retention_inflow = retention_inflow_A
    curr_retention_outflow = retention_outflow_A
    curr_substep_discharge = substep_discharge_A
    curr_discharge_vol_sum = discharge_vol_sum_A
    curr_wb_processed = wb_processed_A

    next_river_storage = river_storage_inertial_B
    next_waterbody_storage = waterbody_storage_B
    next_retention_storage = retention_storage_B
    next_over_abstraction = over_abstraction_B
    next_actual_evaporation = actual_evaporation_B
    next_waterbody_inflow = waterbody_inflow_B
    next_retention_inflow = retention_inflow_B
    next_retention_outflow = retention_outflow_B
    next_substep_discharge = substep_discharge_B
    next_discharge_vol_sum = discharge_vol_sum_B
    next_wb_processed = wb_processed_B

    n_wb = outflow_per_waterbody_m3.size
    n_ret = retention_max_storage_m3.size

    for k in range(n_inertial):
        sf_val = (
            sideflow_m3_inertial[k]
            if np.isfinite(sideflow_m3_inertial[k])
            else np.float32(0.0)
        )
        ev_val = (
            evaporation_m3_inertial[k]
            if np.isfinite(evaporation_m3_inertial[k])
            else np.float32(0.0)
        )
        sideflow_substep_m3_buf[k] = np.float64(sf_val) * fraction_f64
        evap_substep_m3_buf[k] = np.float64(ev_val) * fraction_f64

    for i in range(n_wb):
        wb_of = (
            outflow_per_waterbody_m3[i]
            if np.isfinite(outflow_per_waterbody_m3[i])
            else np.float32(0.0)
        )
        wb_outflow_substep_m3_buf[i] = np.float64(wb_of) * fraction_f64

    for i in range(n_ret):
        ret_max = np.float64(retention_max_storage_m3[i]) * fraction_f64
        retention_inflow_limit_sub_buf[i] = np.float64(0.20) * ret_max
        retention_max_outflow_limit_sub_buf[i] = np.float64(0.05) * ret_max

    wb_extra_lateral_accum = 0.0

    # Substep iteration loop
    for _step_idx in range(num_inertial_substeps):
        _run_single_inertial_substep(
            dt_substep=substep_dt_f32,
            n_kinematic=n_kinematic,
            n_inertial=n_inertial,
            max_up_connections=max_up_connections,
            substep_discharge_in=curr_substep_discharge,
            river_storage_in=curr_river_storage,
            waterbody_storage_in=curr_waterbody_storage,
            retention_storage_in=curr_retention_storage,
            over_abstraction_in=curr_over_abstraction,
            actual_evaporation_in=curr_actual_evaporation,
            waterbody_inflow_in=curr_waterbody_inflow,
            retention_inflow_in=curr_retention_inflow,
            retention_outflow_in=curr_retention_outflow,
            discharge_vol_sum_in=curr_discharge_vol_sum,
            wb_processed_in=curr_wb_processed,
            substep_discharge_out=next_substep_discharge,
            river_storage_out=next_river_storage,
            waterbody_storage_out=next_waterbody_storage,
            retention_storage_out=next_retention_storage,
            over_abstraction_out=next_over_abstraction,
            actual_evaporation_out=next_actual_evaporation,
            waterbody_inflow_out=next_waterbody_inflow,
            retention_inflow_out=next_retention_inflow,
            retention_outflow_out=next_retention_outflow,
            discharge_vol_sum_out=next_discharge_vol_sum,
            wb_processed_out=next_wb_processed,
            sideflow_substep_m3=sideflow_substep_m3_buf,
            evap_substep_m3=evap_substep_m3_buf,
            wb_outflow_substep_m3=wb_outflow_substep_m3_buf,
            retention_inflow_limit_sub=retention_inflow_limit_sub_buf,
            retention_max_outflow_limit_sub=retention_max_outflow_limit_sub_buf,
            wb_outflow_avail_buf=wb_outflow_avail_buf,
            wb_extra_lateral_m3_buf=wb_extra_lateral_m3_buf,
            net_vol_m3=net_vol_m3_buf,
            upstream_matrix=upstream_matrix,
            is_waterbody_outflow=is_waterbody_outflow,
            waterbody_id=waterbody_id,
            river_length=river_length,
            inv_reach_length=inv_reach_length,
            inv_cell_area=inv_cell_area,
            river_width=river_width,
            bed_elevation=bed_elevation,
            g_manning_n_sq=g_manning_n_sq,
            retention_max_storage_m3=retention_max_storage_m3,
            retention_node_id=retention_node_id,
            controlled_retention=controlled_retention,
            retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
            retention_basin_release_threshold_factor=retention_basin_release_threshold_factor,
            ds_node=ds_node,
            is_pit=is_pit,
            is_ocean_pit=is_ocean_pit,
            pit_slope=pit_slope,
            use_kinematic=use_kinematic,
            lateral_flux_m3=lateral_flux_m3,
            next_substep_discharge_m3_s=next_substep_discharge_m3_s,
            kinematic_inflow_rate=kinematic_inflow_rate,
            previous_discharge_m3_s=previous_discharge_m3_s,
            reset_counters=reset_counters,
        )

        wb_extra_lateral_accum += np.sum(wb_extra_lateral_m3_buf)

        # Swap double buffers
        curr_river_storage, next_river_storage = next_river_storage, curr_river_storage
        curr_waterbody_storage, next_waterbody_storage = (
            next_waterbody_storage,
            curr_waterbody_storage,
        )
        curr_retention_storage, next_retention_storage = (
            next_retention_storage,
            curr_retention_storage,
        )
        curr_over_abstraction, next_over_abstraction = (
            next_over_abstraction,
            curr_over_abstraction,
        )
        curr_actual_evaporation, next_actual_evaporation = (
            next_actual_evaporation,
            curr_actual_evaporation,
        )
        curr_waterbody_inflow, next_waterbody_inflow = (
            next_waterbody_inflow,
            curr_waterbody_inflow,
        )
        curr_retention_inflow, next_retention_inflow = (
            next_retention_inflow,
            curr_retention_inflow,
        )
        curr_retention_outflow, next_retention_outflow = (
            next_retention_outflow,
            curr_retention_outflow,
        )
        curr_substep_discharge, next_substep_discharge = (
            next_substep_discharge,
            curr_substep_discharge,
        )
        curr_discharge_vol_sum, next_discharge_vol_sum = (
            next_discharge_vol_sum,
            curr_discharge_vol_sum,
        )
        curr_wb_processed, next_wb_processed = next_wb_processed, curr_wb_processed

    # Store accumulated lateral volume into buffer for balance validation
    wb_extra_lateral_m3_buf.fill(0.0)
    wb_extra_lateral_m3_buf[0] = wb_extra_lateral_accum

    # Copy final accumulated state variables back to output arrays
    river_storage_m3_inertial[:] = curr_river_storage[:]
    waterbody_storage_m3[:] = curr_waterbody_storage[:]
    retention_storage_m3[:] = curr_retention_storage[:]
    over_abstraction_m3_inertial[:] = curr_over_abstraction[:]
    actual_evaporation_m3_inertial[:] = curr_actual_evaporation[:]
    waterbody_inflow_m3[:] = curr_waterbody_inflow[:]
    retention_inflow_m3[:] = curr_retention_inflow[:]
    retention_outflow_m3[:] = curr_retention_outflow[:]
    discharge_vol_sum_m3_inertial[:] = curr_discharge_vol_sum[:]

    inv_dt_f32 = np.float32(1.0) / dt_f32
    for k in range(n_inertial):
        updated_discharge_m3_s_inertial[k] = np.float32(
            curr_discharge_vol_sum[k] * np.float64(inv_dt_f32)
        )

    # Zero out waterbody outflow array for waterbodies processed by the inertial solver
    for wb_id in range(waterbody_storage_m3.size):
        if curr_wb_processed[wb_id]:
            outflow_per_waterbody_m3[wb_id] = np.float32(0.0)


class LocalInertial(Router):
    """Local inertial river network router.

    Combines non-linear kinematic wave routing for upstream steep reaches with a
    1D local inertial dynamic routing formulation for mild-sloped mainstem reaches.
    Internally reorders node topology to optimize contiguous memory access for Numba.
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
        min_slope: float = 1e-4,
    ) -> None:
        """Initializes the LocalInertial router object.

        Args:
            dt: Model time step duration [s].
            river_network: pyflwdir river network raster detailing topological connections.
            river_length: Channel length per cell [m].
            river_width: Channel width per cell [m].
            waterbody_ids: Map of waterbody IDs per cell (-1 if absent).
            river_ids: Map of river reach IDs corresponding to the vector network.
            is_waterbody_outflow: Boolean mask locating waterbody outlets.
            retention_max_storage_m3: Maximum storage capacity per retention basin [m³].
            retention_node_id: Identifier map linking cells to retention basins.
            controlled_retention: Flag defining controlled vs uncontrolled retention dynamics.
            retention_basin_release_threshold_factor: Release threshold factor for retention basins.
            bankfull_river_elevation_m: Channel bed elevation above datum [m].
            manning_n: Manning's roughness coefficient per reach [s/m^(1/3)].
            use_kinematic: Boolean flag selecting kinematic (True) vs inertial (False) routing per cell.
            rivers_gdf: Vector GeoDataFrame containing river geometry attributes.
            min_slope: Minimum allowable slope for ocean/pit boundary boundaries [-].
        """
        super().__init__(
            dt,
            river_network,
            waterbody_ids,
            is_waterbody_outflow,
            retention_basin_release_threshold_factor,
        )
        self.river_length = np.maximum(river_length, np.float32(1.0))
        self.river_width = river_width.astype(np.float32)
        self.river_ids = river_ids
        self.bed_elevation = bankfull_river_elevation_m.astype(np.float32)
        self.manning_n = manning_n.astype(np.float32)
        self.use_kinematic = use_kinematic
        self.retention_node_id = retention_node_id
        self.retention_max_storage_m3 = retention_max_storage_m3
        self.controlled_retention = controlled_retention

        n_nodes_total = self.is_pit.size
        is_ocean_pit_orig = np.zeros(n_nodes_total, dtype=np.bool_)
        pit_slope_orig = np.zeros(n_nodes_total, dtype=np.float32)

        for node_idx in range(n_nodes_total):
            if self.is_pit[node_idx]:
                rid = self.river_ids[node_idx]
                row = rivers_gdf.loc[rid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                ds_id = row["downstream_ID"]
                if ds_id == -1:
                    is_ocean_pit_orig[node_idx] = True
                else:
                    raw_slope = float(row["slope"])
                    pit_slope_orig[node_idx] = np.float32(max(raw_slope, min_slope))

        mapper = np.full(river_network.size + 1, -1, dtype=np.int32)
        indices = np.arange(river_network.size, dtype=np.int32)[river_network.mask]
        mapper[indices] = np.arange(indices.size, dtype=np.int32)

        unmasked_ds = river_network.idxs_ds[indices]
        ds_node_orig = mapper[unmasked_ds]
        ds_node_orig[self.is_pit] = -1

        has_ds = ds_node_orig != -1
        assert np.all(
            self.bed_elevation[has_ds] >= self.bed_elevation[ds_node_orig[has_ds]]
        ), "Bed elevation must strictly decrease downstream along the river network."

        for node_compressed, node in enumerate(indices):
            if (
                use_kinematic[node_compressed]
                or waterbody_ids[node_compressed] != -1
                or self.is_pit[node_compressed]
            ):
                continue

            ds_node = river_network.idxs_ds[node]
            ds_node_compressed = mapper[ds_node]
            assert ds_node_compressed != -1, (
                f"Downstream node for {node} is not in the river network."
            )

            if waterbody_ids[ds_node_compressed] == -1:
                assert not use_kinematic[ds_node_compressed], (
                    f"Invalid topology: inertial node ({node}, river_id {river_ids[node_compressed]}) "
                    f"has downstream kinematic node ({ds_node}, river_id {river_ids[ds_node_compressed]}). "
                    "Downstream nodes of inertial reaches cannot be kinematic."
                )

        assert np.all(
            np.logical_or(np.logical_not(self.is_pit), waterbody_ids == -1)
        ), "Pits cannot be located inside waterbodies."

        n_up_nodes = self.upstream_matrix_from_up_to_downstream.shape[0]
        sorted_idxs = np.empty(n_up_nodes, dtype=np.int32)
        sorted_orig_i = np.empty(n_up_nodes, dtype=np.int32)

        n_kinematic = 0
        n_inertial = 0

        for i in range(n_up_nodes):
            node = self.idxs_up_to_downstream[i]
            if use_kinematic[node] and waterbody_ids[node] == -1:
                sorted_idxs[n_kinematic] = node
                sorted_orig_i[n_kinematic] = i
                n_kinematic += 1

        for i in range(n_up_nodes):
            node = self.idxs_up_to_downstream[i]
            if not use_kinematic[node] and waterbody_ids[node] == -1:
                idx = n_kinematic + n_inertial
                sorted_idxs[idx] = node
                sorted_orig_i[idx] = i
                n_inertial += 1

        n_other_start = n_kinematic + n_inertial
        n_other = 0
        for i in range(n_up_nodes):
            node = self.idxs_up_to_downstream[i]
            if waterbody_ids[node] != -1:
                idx = n_other_start + n_other
                sorted_idxs[idx] = node
                sorted_orig_i[idx] = i
                n_other += 1

        inv_idxs = np.full(n_up_nodes, -1, dtype=np.int32)
        inv_idxs[sorted_idxs] = np.arange(n_up_nodes, dtype=np.int32)

        self.sorted_idxs = sorted_idxs
        self.inv_idxs = inv_idxs
        self.n_kinematic = n_kinematic
        self.n_inertial = n_inertial

        self._river_length = self.river_length[sorted_idxs]
        self._river_width = self.river_width[sorted_idxs]
        self._bed_elevation = self.bed_elevation[sorted_idxs]
        self._manning_n = self.manning_n[sorted_idxs]
        self._use_kinematic = self.use_kinematic[sorted_idxs]
        self._is_pit = self.is_pit[sorted_idxs]
        self._is_ocean_pit = is_ocean_pit_orig[sorted_idxs]
        self._pit_slope = pit_slope_orig[sorted_idxs]
        self._waterbody_ids = self.waterbody_ids[sorted_idxs]
        self._river_ids = self.river_ids[sorted_idxs]
        self._is_waterbody_outflow = self.is_waterbody_outflow[sorted_idxs]
        self._retention_node_id = self.retention_node_id[sorted_idxs]

        self._inv_reach_length = np.float32(1.0) / self._river_length
        self._cell_area = self._river_width * self._river_length
        self._inv_cell_area = np.float32(1.0) / self._cell_area
        self._manning_n_sq = self._manning_n * self._manning_n

        ds_perm = ds_node_orig[sorted_idxs]
        self._ds_node = np.where(ds_perm != -1, inv_idxs[np.maximum(ds_perm, 0)], -1)

        up_perm = self.upstream_matrix_from_up_to_downstream[sorted_orig_i, :]
        self._upstream_matrix = np.where(
            up_perm != -1, inv_idxs[np.maximum(up_perm, 0)], -1
        )

        n_wb = is_waterbody_outflow.sum()
        n_ret = retention_max_storage_m3.size
        assert controlled_retention.size == n_ret, (
            f"controlled_retention size ({controlled_retention.size}) must match "
            f"retention_max_storage_m3 size ({n_ret})."
        )

        # Global permutation workspace
        self._f64_global_workspace = np.empty((1, n_up_nodes), dtype=np.float64)
        self._f32_global_workspace = np.empty((8, n_up_nodes), dtype=np.float32)

        self._river_storage_perm = self._f64_global_workspace[0]

        self._discharge_prev_perm = self._f32_global_workspace[0]
        self._sideflow_perm = self._f32_global_workspace[1]
        self._evaporation_perm = self._f32_global_workspace[2]
        self._alpha_perm = self._f32_global_workspace[3]
        self._beta_perm = self._f32_global_workspace[4]
        self._over_abstraction_perm = self._f32_global_workspace[5]
        self._actual_evaporation_perm = self._f32_global_workspace[6]
        self._updated_discharge_perm = self._f32_global_workspace[7]

        # Inertial-only workspace & double buffers
        self._f64_inertial_workspace = np.empty((9, n_inertial), dtype=np.float64)
        self._f32_inertial_workspace = np.empty((9, n_inertial), dtype=np.float32)

        self._river_storage_inertial_A = self._f64_inertial_workspace[0]
        self._river_storage_inertial_B = self._f64_inertial_workspace[1]
        self._sideflow_substep_m3_buf = self._f64_inertial_workspace[2]
        self._evap_substep_m3_buf = self._f64_inertial_workspace[3]
        self._net_vol_m3_buf = self._f64_inertial_workspace[4]
        self._wb_extra_lateral_m3_buf = self._f64_inertial_workspace[5]
        self._discharge_vol_sum_A = self._f64_inertial_workspace[6]
        self._discharge_vol_sum_B = self._f64_inertial_workspace[7]
        self._discharge_vol_sum_inertial = self._f64_inertial_workspace[8]

        self._substep_discharge_A = self._f32_inertial_workspace[0]
        self._substep_discharge_B = self._f32_inertial_workspace[1]
        self._over_abstraction_A = self._f32_inertial_workspace[2]
        self._over_abstraction_B = self._f32_inertial_workspace[3]
        self._actual_evaporation_A = self._f32_inertial_workspace[4]
        self._actual_evaporation_B = self._f32_inertial_workspace[5]
        self._lateral_flux_inertial = self._f32_inertial_workspace[6]
        self._next_substep_discharge_inertial = self._f32_inertial_workspace[7]
        self._kinematic_inflow_rate_inertial = self._f32_inertial_workspace[8]

        # Waterbody and retention buffers
        self._waterbody_inflow_perm = np.empty(n_wb, dtype=np.float32)
        self._retention_inflow_perm = np.empty(n_ret, dtype=np.float32)
        self._retention_outflow_perm = np.empty(n_ret, dtype=np.float32)

        self._wb_outflow_substep_m3_buf = np.empty(n_wb, dtype=np.float64)
        self._retention_inflow_limit_sub_buf = np.empty(n_ret, dtype=np.float64)
        self._retention_max_outflow_limit_sub_buf = np.empty(n_ret, dtype=np.float64)
        self._wb_outflow_avail_buf = np.empty(n_wb, dtype=np.float64)

        self._waterbody_storage_A = np.empty(n_wb, dtype=np.float64)
        self._waterbody_storage_B = np.empty(n_wb, dtype=np.float64)
        self._retention_storage_A = np.empty(n_ret, dtype=np.float32)
        self._retention_storage_B = np.empty(n_ret, dtype=np.float32)
        self._waterbody_inflow_A = np.empty(n_wb, dtype=np.float32)
        self._waterbody_inflow_B = np.empty(n_wb, dtype=np.float32)
        self._retention_inflow_A = np.empty(n_ret, dtype=np.float32)
        self._retention_inflow_B = np.empty(n_ret, dtype=np.float32)
        self._retention_outflow_A = np.empty(n_ret, dtype=np.float32)
        self._retention_outflow_B = np.empty(n_ret, dtype=np.float32)
        self._wb_processed_A = np.empty(n_wb, dtype=np.bool_)
        self._wb_processed_B = np.empty(n_wb, dtype=np.bool_)

        # Output buffers
        self._discharge_out = np.empty(n_up_nodes, dtype=np.float32)
        self._actual_evap_out = np.empty(n_up_nodes, dtype=np.float32)
        self._over_abs_out = np.empty(n_up_nodes, dtype=np.float32)

    def calculate_river_storage_from_discharge(
        self,
        discharge: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_length: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        waterbody_id: ArrayInt32,
    ) -> ArrayFloat64:
        """Calculate the river storage from the discharge.

        Args:
            discharge: The discharge in each cell, in m3/s.
            river_storage_alpha: The alpha parameter for the kinematic wave equation.
            river_length: The length of the river in each cell, in meters.
            river_storage_beta: The beta parameter for the kinematic wave equation.
            waterbody_id: A 1D array with the same shape as the grid.

        Returns:
            A 1D array with the calculated river storage for each cell, in m3.
        """
        cross_sectional_area: ArrayFloat64 = (
            river_storage_alpha
            * np.abs(discharge).astype(np.float64) ** river_storage_beta
        )
        river_storage: ArrayFloat64 = cross_sectional_area * river_length
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
        """Calculate the discharge from the river storage.

        Args:
            river_storage: The storage in each cell, in m3.
            river_storage_alpha: The alpha parameter for the kinematic wave equation.
            river_storage_beta: The beta parameter for the kinematic wave equation.
            river_length: The length of the river in each cell, in meters.
            waterbody_id: A 1D array with the same shape as the grid.

        Returns:
            A 1D array with the calculated discharge for each cell, in m3/s.
        """
        cross_sectional_area: ArrayFloat32 = (river_storage / river_length).astype(
            np.float32
        )
        discharge: ArrayFloat32 = (cross_sectional_area / river_storage_alpha) ** (
            1 / river_storage_beta
        )
        discharge[waterbody_id != -1] = np.nan
        return discharge

    def get_available_storage(
        self,
        discharge: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        maximum_abstraction_ratio: float = 0.9,
    ) -> ArrayFloat64:
        """Get the available storage of the river network.

        Args:
            discharge: The discharge in each cell, in m3/s.
            river_storage_alpha: The alpha parameter for the kinematic wave equation.
            river_storage_beta: The beta parameter for the kinematic wave equation.
            maximum_abstraction_ratio: The maximum abstraction ratio.

        Returns:
            The available storage of the river network [m3].
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
        """Get the total storage of the river network.

        Args:
            discharge: The discharge in each cell, in m3/s.
            river_storage_alpha: The alpha parameter for the kinematic wave equation.
            river_storage_beta: The beta parameter for the kinematic wave equation.

        Returns:
            The total storage of the river network [m3].
        """
        total_storage: ArrayFloat64 = self.calculate_river_storage_from_discharge(
            discharge=discharge,
            river_storage_alpha=river_storage_alpha,
            river_length=self.river_length,
            river_storage_beta=river_storage_beta,
            waterbody_id=self.waterbody_ids,
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
        upstream_matrix: TwoDArrayInt32,
        is_waterbody_outflow: ArrayBool,
        waterbody_id: ArrayInt32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        river_length: ArrayFloat32,
        inv_reach_length: ArrayFloat32,
        inv_cell_area: ArrayFloat32,
        river_width: ArrayFloat32,
        bed_elevation: ArrayFloat32,
        manning_n_sq: ArrayFloat32,
        retention_storage_m3: ArrayFloat32,
        retention_max_storage_m3: ArrayFloat32,
        retention_node_id: ArrayInt32,
        controlled_retention: ArrayBool,
        retention_activation_threshold_m3_s: ArrayFloat32,
        retention_basin_release_threshold_factor: np.float32,
        ds_node: ArrayInt32,
        is_pit: ArrayBool,
        is_ocean_pit: ArrayBool,
        pit_slope: ArrayFloat32,
        use_kinematic: ArrayBool,
        n_kinematic: int,
        n_inertial: int,
        over_abstraction_m3: ArrayFloat32,
        actual_evaporation_m3: ArrayFloat32,
        waterbody_inflow_m3: ArrayFloat32,
        retention_inflow_m3: ArrayFloat32,
        retention_outflow_m3: ArrayFloat32,
        updated_discharge_m3_s: ArrayFloat32,
        lateral_flux_inertial: ArrayFloat32,
        next_substep_discharge_inertial: ArrayFloat32,
        kinematic_inflow_rate_inertial: ArrayFloat32,
        discharge_vol_sum_inertial: ArrayFloat64,
        sideflow_substep_m3_buf: ArrayFloat64,
        evap_substep_m3_buf: ArrayFloat64,
        wb_outflow_substep_m3_buf: ArrayFloat64,
        retention_inflow_limit_sub_buf: ArrayFloat64,
        retention_max_outflow_limit_sub_buf: ArrayFloat64,
        wb_outflow_avail_buf: ArrayFloat64,
        wb_extra_lateral_m3_buf: ArrayFloat64,
        net_vol_m3_buf: ArrayFloat64,
        river_storage_inertial_A: ArrayFloat64,
        waterbody_storage_A: ArrayFloat64,
        retention_storage_A: ArrayFloat32,
        over_abstraction_A: ArrayFloat32,
        actual_evaporation_A: ArrayFloat32,
        waterbody_inflow_A: ArrayFloat32,
        retention_inflow_A: ArrayFloat32,
        retention_outflow_A: ArrayFloat32,
        substep_discharge_A: ArrayFloat32,
        discharge_vol_sum_A: ArrayFloat64,
        wb_processed_A: ArrayBool,
        river_storage_inertial_B: ArrayFloat64,
        waterbody_storage_B: ArrayFloat64,
        retention_storage_B: ArrayFloat32,
        over_abstraction_B: ArrayFloat32,
        actual_evaporation_B: ArrayFloat32,
        waterbody_inflow_B: ArrayFloat32,
        retention_inflow_B: ArrayFloat32,
        retention_outflow_B: ArrayFloat32,
        substep_discharge_B: ArrayFloat32,
        discharge_vol_sum_B: ArrayFloat64,
        wb_processed_B: ArrayBool,
    ) -> tuple[
        ArrayFloat32,
        ArrayFloat64,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        np.float32,
    ]:
        """Runs one combined routing step across kinematic and local inertial domains.

        Args:
            routing_timestep_s: Model timestep length [s].
            previous_discharge_m3_s: Initial channel discharges at start of step [m³/s].
            river_storage_m3: Initial channel storage volume [m³].
            sideflow_m3: Lateral inflow volume per reach [m³].
            evaporation_m3: Potential evaporation volume per reach [m³].
            waterbody_storage_m3: Waterbody storage volume [m³].
            outflow_per_waterbody_m3: Prescribed/calculated waterbody outflow [m³].
            upstream_matrix: Topological upstream connection indices.
            is_waterbody_outflow: Boolean mask indicating waterbody outlets.
            waterbody_id: Map of waterbody IDs per cell (-1 if absent).
            river_storage_alpha: Kinematic wave coefficient $\alpha$.
            river_storage_beta: Kinematic wave exponent $\beta$.
            river_length: Reach length [m].
            inv_reach_length: Precomputed inverse reach length [1/m].
            inv_cell_area: Precomputed inverse cell surface area [1/m²].
            river_width: Reach channel width [m].
            bed_elevation: Elevation of riverbed above datum [m].
            manning_n_sq: Squared Manning's $n$ coefficient.
            retention_storage_m3: Retention basin storage volume [m³].
            retention_max_storage_m3: Capacity limit per retention basin [m³].
            retention_node_id: Identifier map linking cells to retention basins.
            controlled_retention: Flag defining controlled vs uncontrolled retention dynamics.
            retention_activation_threshold_m3_s: Threshold triggering basin diversion [m³/s].
            retention_basin_release_threshold_factor: Release threshold factor for retention basins.
            ds_node: Index map for downstream target nodes.
            is_pit: Boolean mask for terminal nodes.
            is_ocean_pit: Boolean mask for ocean outlet boundary nodes.
            pit_slope: Bed slope applied at boundary terminal nodes [-].
            use_kinematic: Boolean flag specifying kinematic routing per node.
            n_kinematic: Count of kinematic reaches.
            n_inertial: Count of local inertial reaches.
            over_abstraction_m3: Output tracking unfulfilled abstractions [m³].
            actual_evaporation_m3: Output tracking actual channel evaporation [m³].
            waterbody_inflow_m3: Output tracking total inflows to waterbodies [m³].
            retention_inflow_m3: Output tracking total diversions to retention basins [m³].
            retention_outflow_m3: Output tracking releases from retention basins [m³].
            updated_discharge_m3_s: Output array for calculated timestep discharges [m³/s].
            lateral_flux_inertial: Buffer array for inertial lateral fluxes [m³].
            next_substep_discharge_inertial: Buffer array storing solved inertial substep discharges [m³/s].
            kinematic_inflow_rate_inertial: Upstream inflow rate sourced from kinematic reaches [m³/s].
            discharge_vol_sum_inertial: Buffer tracking cumulative discharge volume in inertial reaches [m³].
            sideflow_substep_m3_buf: Allocated buffer for substep sideflow volume [m³].
            evap_substep_m3_buf: Allocated buffer for substep evaporation volume [m³].
            wb_outflow_substep_m3_buf: Allocated buffer for substep waterbody outflow volume [m³].
            retention_inflow_limit_sub_buf: Allocated buffer for substep retention inflow limit [m³].
            retention_max_outflow_limit_sub_buf: Allocated buffer for substep retention outflow limit [m³].
            wb_outflow_avail_buf: Allocated workspace tracking available waterbody outflow [m³].
            wb_extra_lateral_m3_buf: Workspace tracking extra lateral volume from waterbodies [m³].
            net_vol_m3_buf: Workspace array for net reach volume changes [m³].
            river_storage_inertial_A: Double-buffering workspace A for river storage [m³].
            waterbody_storage_A: Double-buffering workspace A for waterbody storage [m³].
            retention_storage_A: Double-buffering workspace A for retention storage [m³].
            over_abstraction_A: Double-buffering workspace A for over-abstraction [m³].
            actual_evaporation_A: Double-buffering workspace A for actual evaporation [m³].
            waterbody_inflow_A: Double-buffering workspace A for waterbody inflow [m³].
            retention_inflow_A: Double-buffering workspace A for retention inflow [m³].
            retention_outflow_A: Double-buffering workspace A for retention outflow [m³].
            substep_discharge_A: Double-buffering workspace A for substep discharge [m³/s].
            discharge_vol_sum_A: Double-buffering workspace A for cumulative discharge volume [m³].
            wb_processed_A: Double-buffering workspace A for waterbody processing flags.
            river_storage_inertial_B: Double-buffering workspace B for river storage [m³].
            waterbody_storage_B: Double-buffering workspace B for waterbody storage [m³].
            retention_storage_B: Double-buffering workspace B for retention storage [m³].
            over_abstraction_B: Double-buffering workspace B for over-abstraction [m³].
            actual_evaporation_B: Double-buffering workspace B for actual evaporation [m³].
            waterbody_inflow_B: Double-buffering workspace B for waterbody inflow [m³].
            retention_inflow_B: Double-buffering workspace B for retention inflow [m³].
            retention_outflow_B: Double-buffering workspace B for retention outflow [m³].
            substep_discharge_B: Double-buffering workspace B for substep discharge [m³/s].
            discharge_vol_sum_B: Double-buffering workspace B for cumulative discharge volume [m³].
            wb_processed_B: Double-buffering workspace B for waterbody processing flags.

        Returns:
            A tuple containing:
                - updated_discharge_m3_s: Calculated reach discharge at end of step [m³/s].
                - river_storage_m3: Updated river storage volume [m³].
                - actual_evaporation_m3: Evaporation loss from channel [m³].
                - over_abstraction_m3: Deficit from over-abstraction [m³].
                - waterbody_inflow_m3: Total volumetric inflow to waterbodies [m³].
                - retention_inflow_m3: Total diversion to retention basins [m³].
                - retention_outflow_m3: Total releases from retention basins [m³].
                - terminal_waterbody_outflow_m3: Volume discharged out of boundary waterbody outlets [m³].

        Raises:
            ValueError: If the routing timestep is non-positive or if the number of kinematic/inertial reaches exceeds the total number of cells.
        """
        n_cells: int = previous_discharge_m3_s.size
        max_up_connections: int = upstream_matrix.shape[1]

        over_abstraction_m3.fill(np.float32(0.0))
        actual_evaporation_m3.fill(np.float32(0.0))
        waterbody_inflow_m3.fill(np.float32(0.0))
        retention_inflow_m3.fill(np.float32(0.0))
        retention_outflow_m3.fill(np.float32(0.0))
        updated_discharge_m3_s.fill(np.float32(0.0))

        lateral_flux_inertial.fill(np.float32(0.0))
        next_substep_discharge_inertial.fill(np.float32(0.0))
        kinematic_inflow_rate_inertial.fill(np.float32(0.0))
        discharge_vol_sum_inertial.fill(0.0)

        n_other_start = n_kinematic + n_inertial
        for i in range(n_other_start, n_cells):
            updated_discharge_m3_s[i] = np.float32(np.nan)

        for i in range(n_other_start, n_cells):
            wb_id = waterbody_id[i]
            if wb_id != -1:
                waterbody_storage_m3[wb_id] += np.float64(sideflow_m3[i])

                evap_substep = np.float32(
                    min(np.float64(evaporation_m3[i]), waterbody_storage_m3[wb_id])
                )
                evap_substep = max(evap_substep, np.float32(0.0))
                actual_evaporation_m3[i] = evap_substep
                waterbody_storage_m3[wb_id] -= np.float64(evap_substep)

        dt_f32: np.float32 = np.float32(routing_timestep_s)
        inv_dt_f32: np.float32 = np.float32(1.0) / dt_f32

        inertial_outflow_per_waterbody = outflow_per_waterbody_m3.copy()

        # Direct Lake-to-Lake Topological Transfers
        for i in range(n_cells):
            if is_waterbody_outflow[i]:
                wb_id = waterbody_id[i]
                if wb_id != -1 and inertial_outflow_per_waterbody[wb_id] > 0.0:
                    ds = ds_node[i]
                    if ds != -1 and waterbody_id[ds] != -1:
                        ds_wb_id = waterbody_id[ds]
                        if ds_wb_id != wb_id:
                            actual_outflow = min(
                                np.float64(inertial_outflow_per_waterbody[wb_id]),
                                waterbody_storage_m3[wb_id],
                            )
                            if actual_outflow > 0.0:
                                waterbody_storage_m3[wb_id] -= actual_outflow
                                waterbody_storage_m3[ds_wb_id] += actual_outflow
                                waterbody_inflow_m3[ds_wb_id] += np.float32(
                                    actual_outflow
                                )
                            inertial_outflow_per_waterbody[wb_id] = np.float32(0.0)

        # Kinematic wave routing loop
        for i in range(n_kinematic):
            upstream_inflow_m3_s: np.float32 = np.float32(0.0)
            node_sideflow: np.float32 = sideflow_m3[i]

            for j in range(max_up_connections):
                up_node = upstream_matrix[i, j]
                if up_node == -1:
                    break

                if is_waterbody_outflow[up_node]:
                    wb_id = waterbody_id[up_node]
                    if wb_id != -1 and inertial_outflow_per_waterbody[wb_id] > 0.0:
                        wb_storage = max(waterbody_storage_m3[wb_id], 0.0)
                        wb_outflow = np.float32(
                            min(
                                np.float64(inertial_outflow_per_waterbody[wb_id]),
                                wb_storage,
                            )
                        )
                        if wb_outflow > 0.0:
                            waterbody_storage_m3[wb_id] -= np.float64(wb_outflow)
                            node_sideflow += wb_outflow
                        inertial_outflow_per_waterbody[wb_id] = np.float32(0.0)
                elif waterbody_id[up_node] == -1:
                    upstream_inflow_m3_s += max(
                        updated_discharge_m3_s[up_node], np.float32(0.0)
                    )

            ret_id = retention_node_id[i]
            if ret_id != -1:
                discharge_before_diversion = (
                    (upstream_inflow_m3_s + previous_discharge_m3_s[i])
                    * np.float32(0.5)
                ) + (node_sideflow * inv_dt_f32)

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
                    activation_threshold_m3_s=retention_activation_threshold_m3_s[
                        ret_id
                    ],
                    release_threshold_factor=retention_basin_release_threshold_factor,
                    inflow_limit_m3=inflow_limit,
                    max_outflow_limit_m3=max_outflow_limit,
                )

                retention_inflow_m3[ret_id] += diverted_volume
                retention_outflow_m3[ret_id] += outflow_volume
                node_sideflow = discharge_at_basin_vol - upstream_inflow_m3_s * dt_f32

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

            inflow_vol = upstream_inflow_m3_s * dt_f32 + node_sideflow
            outflow_vol = kinematic_discharge * dt_f32
            river_storage_m3[i] += (
                np.float64(inflow_vol) - np.float64(outflow_vol) - np.float64(evap_vol)
            )

            ds = ds_node[i]
            if ds != -1 and waterbody_id[ds] != -1:
                wb_ds_id = waterbody_id[ds]
                waterbody_storage_m3[wb_ds_id] += np.float64(outflow_vol)
                if outflow_vol > np.float32(0.0):
                    waterbody_inflow_m3[wb_ds_id] += outflow_vol

            if river_storage_m3[i] < np.float64(0.0):
                over_abstraction_m3[i] += np.float32(-river_storage_m3[i])
                river_storage_m3[i] = np.float64(0.0)

            if not np.isfinite(updated_discharge_m3_s[i]):
                raise ValueError(
                    "Non-finite discharge computed in kinematic wave reach."
                )

        # Local inertial wave routing loop
        if n_inertial > 0:
            inertial_end: int = n_kinematic + n_inertial

            for k in range(n_inertial):
                i = n_kinematic + k
                for j in range(max_up_connections):
                    up_node = upstream_matrix[i, j]
                    if up_node == -1:
                        break
                    if is_waterbody_outflow[up_node]:
                        continue
                    if use_kinematic[up_node] and waterbody_id[up_node] == -1:
                        kinematic_inflow_rate_inertial[k] += max(
                            updated_discharge_m3_s[up_node], np.float32(0.0)
                        )

            gravity_acceleration: np.float32 = np.float32(9.80665)
            cfl_safety_factor: np.float32 = np.float32(0.7)
            min_stable_dt: np.float32 = np.float32(1e9)

            for k in range(n_inertial):
                i = n_kinematic + k
                sideflow_rate = sideflow_m3[i] * inv_dt_f32
                est_inflow_vol = (
                    kinematic_inflow_rate_inertial[k] * dt_f32 + sideflow_m3[i]
                )
                pred_storage = max(river_storage_m3[i] + est_inflow_vol, 0.0)

                pred_depth = max(
                    np.float32(pred_storage) * inv_cell_area[i], np.float32(0.1)
                )
                wave_celerity: np.float32 = np.sqrt(gravity_acceleration * pred_depth)

                pred_discharge = (
                    np.abs(previous_discharge_m3_s[i])
                    + kinematic_inflow_rate_inertial[k]
                    + sideflow_rate
                )
                flow_vel: np.float32 = pred_discharge / (pred_depth * river_width[i])

                max_dt: np.float32 = (cfl_safety_factor * river_length[i]) / (
                    wave_celerity + flow_vel + np.float32(1e-9)
                )
                if max_dt < min_stable_dt:
                    min_stable_dt = max_dt

            min_stable_dt = max(min_stable_dt, np.float32(0.5))
            raw_substeps = int(np.ceil(dt_f32 / min_stable_dt))
            num_inertial_substeps: int = max(1, min(raw_substeps, 1000))

            _run_inertial_substeps(
                num_inertial_substeps=num_inertial_substeps,
                dt_f32=dt_f32,
                n_kinematic=n_kinematic,
                n_inertial=n_inertial,
                max_up_connections=max_up_connections,
                previous_discharge_m3_s=previous_discharge_m3_s,
                river_storage_m3_inertial=river_storage_m3[n_kinematic:inertial_end],
                sideflow_m3_inertial=sideflow_m3[n_kinematic:inertial_end],
                evaporation_m3_inertial=evaporation_m3[n_kinematic:inertial_end],
                waterbody_storage_m3=waterbody_storage_m3,
                outflow_per_waterbody_m3=inertial_outflow_per_waterbody,
                upstream_matrix=upstream_matrix,
                is_waterbody_outflow=is_waterbody_outflow,
                waterbody_id=waterbody_id,
                river_length=river_length,
                inv_reach_length=inv_reach_length,
                inv_cell_area=inv_cell_area,
                river_width=river_width,
                bed_elevation=bed_elevation,
                manning_n_sq=manning_n_sq,
                retention_storage_m3=retention_storage_m3,
                retention_max_storage_m3=retention_max_storage_m3,
                retention_node_id=retention_node_id,
                controlled_retention=controlled_retention,
                retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
                retention_basin_release_threshold_factor=retention_basin_release_threshold_factor,
                ds_node=ds_node,
                is_pit=is_pit,
                is_ocean_pit=is_ocean_pit,
                pit_slope=pit_slope,
                use_kinematic=use_kinematic,
                over_abstraction_m3_inertial=over_abstraction_m3[
                    n_kinematic:inertial_end
                ],
                actual_evaporation_m3_inertial=actual_evaporation_m3[
                    n_kinematic:inertial_end
                ],
                waterbody_inflow_m3=waterbody_inflow_m3,
                retention_inflow_m3=retention_inflow_m3,
                retention_outflow_m3=retention_outflow_m3,
                updated_discharge_m3_s_inertial=updated_discharge_m3_s[
                    n_kinematic:inertial_end
                ],
                lateral_flux_m3=lateral_flux_inertial,
                next_substep_discharge_m3_s=next_substep_discharge_inertial,
                kinematic_inflow_rate=kinematic_inflow_rate_inertial,
                discharge_vol_sum_m3_inertial=discharge_vol_sum_inertial,
                sideflow_substep_m3_buf=sideflow_substep_m3_buf,
                evap_substep_m3_buf=evap_substep_m3_buf,
                wb_outflow_substep_m3_buf=wb_outflow_substep_m3_buf,
                retention_inflow_limit_sub_buf=retention_inflow_limit_sub_buf,
                retention_max_outflow_limit_sub_buf=retention_max_outflow_limit_sub_buf,
                wb_outflow_avail_buf=wb_outflow_avail_buf,
                wb_extra_lateral_m3_buf=wb_extra_lateral_m3_buf,
                net_vol_m3_buf=net_vol_m3_buf,
                river_storage_inertial_A=river_storage_inertial_A,
                waterbody_storage_A=waterbody_storage_A,
                retention_storage_A=retention_storage_A,
                over_abstraction_A=over_abstraction_A,
                actual_evaporation_A=actual_evaporation_A,
                waterbody_inflow_A=waterbody_inflow_A,
                retention_inflow_A=retention_inflow_A,
                retention_outflow_A=retention_outflow_A,
                substep_discharge_A=substep_discharge_A,
                discharge_vol_sum_A=discharge_vol_sum_A,
                wb_processed_A=wb_processed_A,
                river_storage_inertial_B=river_storage_inertial_B,
                waterbody_storage_B=waterbody_storage_B,
                retention_storage_B=retention_storage_B,
                over_abstraction_B=over_abstraction_B,
                actual_evaporation_B=actual_evaporation_B,
                waterbody_inflow_B=waterbody_inflow_B,
                retention_inflow_B=retention_inflow_B,
                retention_outflow_B=retention_outflow_B,
                substep_discharge_B=substep_discharge_B,
                discharge_vol_sum_B=discharge_vol_sum_B,
                wb_processed_B=wb_processed_B,
            )

        terminal_waterbody_outflow_m3 = np.float32(0.0)
        for wb_id in range(waterbody_storage_m3.size):
            remaining_outflow = inertial_outflow_per_waterbody[wb_id]
            if (
                remaining_outflow > np.float32(0.0)
                and waterbody_storage_m3[wb_id] > 0.0
            ):
                actual_term_outflow = min(
                    np.float64(remaining_outflow),
                    waterbody_storage_m3[wb_id],
                )
                waterbody_storage_m3[wb_id] -= actual_term_outflow
                terminal_waterbody_outflow_m3 += np.float32(actual_term_outflow)

        return (
            updated_discharge_m3_s,
            river_storage_m3,
            actual_evaporation_m3,
            over_abstraction_m3,
            waterbody_inflow_m3,
            retention_inflow_m3,
            retention_outflow_m3,
            terminal_waterbody_outflow_m3,
        )

    def _check_inertial_water_balance(
        self,
        storage_before: float,
        storage_after: float,
        sideflow_m3: float,
        kinematic_inflow_m3: float,
        actual_evap_m3: float,
        over_abs_m3: float,
        discharge_perm: ArrayFloat32,
        retention_inflow_m3: ArrayFloat32,
        retention_outflow_m3: ArrayFloat32,
    ) -> float:
        """Validates mass conservation strictly for the inertial channel subdomain.

        Args:
            storage_before: Total inertial storage volume before timestep [m³].
            storage_after: Total inertial storage volume after timestep [m³].
            sideflow_m3: Total lateral volume added to inertial reaches [m³].
            kinematic_inflow_m3: Total inflow volume from upstream kinematic reaches [m³].
            actual_evap_m3: Total actual evaporation volume from inertial reaches [m³].
            over_abs_m3: Total unfulfilled abstraction deficit volume [m³].
            discharge_perm: Output discharge array in topological permuted order [m³/s].
            retention_inflow_m3: Array of total inflow volume diverted to retention basins [m³].
            retention_outflow_m3: Array of total outflow volume released from retention basins [m³].

        Returns:
            Computed water balance volume discrepancy [m³].

        Raises:
            ValueError: If the computed discrepancy exceeds tolerance_m3.
        """
        if self.n_inertial == 0:
            return 0.0

        inertial_start = self.n_kinematic
        inertial_end = self.n_kinematic + self.n_inertial

        total_wb_inflow_to_inertial = (
            float(self._wb_extra_lateral_m3_buf[0])
            if self._wb_extra_lateral_m3_buf.size > 0
            else 0.0
        )

        ds_flux_m3 = 0.0
        total_inertial_outflow_to_wb = 0.0

        for i in range(inertial_start, inertial_end):
            vol = float(discharge_perm[i]) * self.dt
            if self._is_pit[i]:
                ds_flux_m3 += vol
            else:
                ds = self._ds_node[i]
                if ds != -1 and self._waterbody_ids[ds] != -1:
                    total_inertial_outflow_to_wb += vol

        total_retention_diverted_m3 = 0.0
        total_retention_released_m3 = 0.0
        for i in range(inertial_start, inertial_end):
            ret_id = self._retention_node_id[i]
            if ret_id != -1:
                total_retention_diverted_m3 += float(retention_inflow_m3[ret_id])
                total_retention_released_m3 += float(retention_outflow_m3[ret_id])

        delta_storage = storage_after - storage_before
        expected_delta = (
            sideflow_m3
            + kinematic_inflow_m3
            + total_wb_inflow_to_inertial
            + total_retention_released_m3
            + over_abs_m3
            - actual_evap_m3
            - ds_flux_m3
            - total_inertial_outflow_to_wb
            - total_retention_diverted_m3
        )

        balance_err = delta_storage - expected_delta

        tolerance_m3 = max(
            float(discharge_perm.size) * 0.001,
            max(abs(storage_before), abs(storage_after)) * 1e-4,
            abs(expected_delta) * 1e-4,
        )
        if abs(balance_err) > tolerance_m3:
            raise ValueError(
                f"Inertial water balance error exceeded tolerance: {balance_err:+.6f} m³ "
                f"(tolerance: {tolerance_m3} m³)."
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
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        retention_activation_threshold_m3_s: ArrayFloat32,
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
        """Advances the river routing state forward by one timestep.

        Args:
            Q_prev_m3_s: Channel discharge at start of timestep [m³/s].
            river_storage_m3: Storage volume per reach [m³].
            sideflow_m3: Lateral runoff volume per reach [m³].
            evaporation_m3: Potential evaporation volume per reach [m³].
            waterbody_storage_m3: Storage volume per waterbody [m³].
            outflow_per_waterbody_m3: Target outflow volume per waterbody [m³].
            retention_storage_m3: Storage volume per retention basin [m³].
            river_storage_alpha: Kinematic wave parameter $\alpha$.
            river_storage_beta: Kinematic wave parameter $\beta$.
            retention_activation_threshold_m3_s: Discharge threshold triggering retention basin diversion [m³/s].

        Returns:
            A tuple containing:
                - Q_out_m3_s: Calculated channel discharge at end of step [m³/s].
                - river_storage_m3: Updated river reach storage [m³].
                - actual_evap_m3: Actual evaporation volume per reach [m³].
                - over_abs_m3: Deficit from unfulfilled abstractions [m³].
                - waterbody_storage_m3: Updated waterbody storage [m³].
                - waterbody_inflow_m3: Inflow volume per waterbody [m³].
                - outflow_at_pits_m3: Total outflow volume exiting through domain boundaries [m³].
                - retention_storage_m3: Updated storage volume per retention basin [m³].
                - retention_inflow_m3: Diverted inflow volume per retention basin [m³].
                - retention_outflow_m3: Released outflow volume per retention basin [m³].
        """
        np.take(Q_prev_m3_s, self.sorted_idxs, out=self._discharge_prev_perm)
        np.take(river_storage_m3, self.sorted_idxs, out=self._river_storage_perm)
        np.take(sideflow_m3, self.sorted_idxs, out=self._sideflow_perm)
        np.take(evaporation_m3, self.sorted_idxs, out=self._evaporation_perm)
        np.take(river_storage_alpha, self.sorted_idxs, out=self._alpha_perm)
        np.take(river_storage_beta, self.sorted_idxs, out=self._beta_perm)

        inertial_start = self.n_kinematic
        inertial_end = self.n_kinematic + self.n_inertial

        self._wb_extra_lateral_m3_buf.fill(0.0)

        storage_inertial_before = float(
            np.sum(self._river_storage_perm[inertial_start:inertial_end])
        )
        sideflow_inertial = float(
            np.sum(self._sideflow_perm[inertial_start:inertial_end])
        )

        (
            discharge_perm,
            river_storage_perm,
            actual_evaporation_perm,
            over_abstraction_perm,
            waterbody_inflow_m3,
            retention_inflow_m3,
            retention_outflow_m3,
            terminal_wb_outflow_m3,
        ) = self._step(
            routing_timestep_s=self.dt,
            previous_discharge_m3_s=self._discharge_prev_perm,
            river_storage_m3=self._river_storage_perm,
            sideflow_m3=self._sideflow_perm,
            evaporation_m3=self._evaporation_perm,
            waterbody_storage_m3=waterbody_storage_m3,
            outflow_per_waterbody_m3=outflow_per_waterbody_m3,
            upstream_matrix=self._upstream_matrix,
            is_waterbody_outflow=self._is_waterbody_outflow,
            waterbody_id=self._waterbody_ids,
            river_storage_alpha=self._alpha_perm,
            river_storage_beta=self._beta_perm,
            river_length=self._river_length,
            inv_reach_length=self._inv_reach_length,
            inv_cell_area=self._inv_cell_area,
            river_width=self._river_width,
            bed_elevation=self._bed_elevation,
            manning_n_sq=self._manning_n_sq,
            retention_storage_m3=retention_storage_m3,
            retention_max_storage_m3=self.retention_max_storage_m3,
            retention_node_id=self._retention_node_id,
            controlled_retention=self.controlled_retention,
            retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
            retention_basin_release_threshold_factor=self.retention_basin_release_threshold_factor,
            ds_node=self._ds_node,
            is_pit=self._is_pit,
            is_ocean_pit=self._is_ocean_pit,
            pit_slope=self._pit_slope,
            use_kinematic=self._use_kinematic,
            n_kinematic=self.n_kinematic,
            n_inertial=self.n_inertial,
            over_abstraction_m3=self._over_abstraction_perm,
            actual_evaporation_m3=self._actual_evaporation_perm,
            waterbody_inflow_m3=self._waterbody_inflow_perm,
            retention_inflow_m3=self._retention_inflow_perm,
            retention_outflow_m3=self._retention_outflow_perm,
            updated_discharge_m3_s=self._updated_discharge_perm,
            lateral_flux_inertial=self._lateral_flux_inertial,
            next_substep_discharge_inertial=self._next_substep_discharge_inertial,
            kinematic_inflow_rate_inertial=self._kinematic_inflow_rate_inertial,
            discharge_vol_sum_inertial=self._discharge_vol_sum_inertial,
            sideflow_substep_m3_buf=self._sideflow_substep_m3_buf,
            evap_substep_m3_buf=self._evap_substep_m3_buf,
            wb_outflow_substep_m3_buf=self._wb_outflow_substep_m3_buf,
            retention_inflow_limit_sub_buf=self._retention_inflow_limit_sub_buf,
            retention_max_outflow_limit_sub_buf=self._retention_max_outflow_limit_sub_buf,
            wb_outflow_avail_buf=self._wb_outflow_avail_buf,
            wb_extra_lateral_m3_buf=self._wb_extra_lateral_m3_buf,
            net_vol_m3_buf=self._net_vol_m3_buf,
            river_storage_inertial_A=self._river_storage_inertial_A,
            waterbody_storage_A=self._waterbody_storage_A,
            retention_storage_A=self._retention_storage_A,
            over_abstraction_A=self._over_abstraction_A,
            actual_evaporation_A=self._actual_evaporation_A,
            waterbody_inflow_A=self._waterbody_inflow_A,
            retention_inflow_A=self._retention_inflow_A,
            retention_outflow_A=self._retention_outflow_A,
            substep_discharge_A=self._substep_discharge_A,
            discharge_vol_sum_A=self._discharge_vol_sum_A,
            wb_processed_A=self._wb_processed_A,
            river_storage_inertial_B=self._river_storage_inertial_B,
            waterbody_storage_B=self._waterbody_storage_B,
            retention_storage_B=self._retention_storage_B,
            over_abstraction_B=self._over_abstraction_B,
            actual_evaporation_B=self._actual_evaporation_B,
            waterbody_inflow_B=self._waterbody_inflow_B,
            retention_inflow_B=self._retention_inflow_B,
            retention_outflow_B=self._retention_outflow_B,
            substep_discharge_B=self._substep_discharge_B,
            discharge_vol_sum_B=self._discharge_vol_sum_B,
            wb_processed_B=self._wb_processed_B,
        )

        storage_inertial_after = float(
            np.sum(river_storage_perm[inertial_start:inertial_end])
        )
        actual_evap_inertial = float(
            np.sum(actual_evaporation_perm[inertial_start:inertial_end])
        )
        over_abs_inertial = float(
            np.sum(over_abstraction_perm[inertial_start:inertial_end])
        )
        kinematic_inflow_vol_m3 = (
            float(np.sum(self._kinematic_inflow_rate_inertial)) * self.dt
        )

        if __debug__:
            self._check_inertial_water_balance(
                storage_before=storage_inertial_before,
                storage_after=storage_inertial_after,
                sideflow_m3=sideflow_inertial,
                kinematic_inflow_m3=kinematic_inflow_vol_m3,
                actual_evap_m3=actual_evap_inertial,
                over_abs_m3=over_abs_inertial,
                discharge_perm=discharge_perm,
                retention_inflow_m3=retention_inflow_m3,
                retention_outflow_m3=retention_outflow_m3,
            )

        outflow_at_pits_m3 = np.float32(0.0)
        for i in range(self.n_kinematic + self.n_inertial):
            if self._is_pit[i]:
                outflow_at_pits_m3 += discharge_perm[i] * self.dt
        outflow_at_pits_m3 += terminal_wb_outflow_m3

        np.take(discharge_perm, self.inv_idxs, out=self._discharge_out)
        np.take(actual_evaporation_perm, self.inv_idxs, out=self._actual_evap_out)
        np.take(over_abstraction_perm, self.inv_idxs, out=self._over_abs_out)
        np.take(river_storage_perm, self.inv_idxs, out=river_storage_m3)

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
