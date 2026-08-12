"""1D local inertial river routing algorithm for channel networks.

Implements a hydrodynamically simplified 1D local inertial routing formulation
(neglecting convective acceleration) integrated with kinematic wave sections,
retention basins, and waterbody (lakes/reservoirs) dynamics.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
from numba import njit, prange

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    TwoDArrayInt32,
)

from .kinematic_wave import update_node_kinematic
from .routerbase import Router, compute_retention_routing


@njit(parallel=False, cache=True, fastmath=True, inline="always")
def _run_single_inertial_substep(
    dt_substep: np.float32,
    n_kinematic: int,
    inertial_end: int,
    max_up_connections: int,
    # Read-only input state
    substep_discharge_in: ArrayFloat32,
    river_storage_in: ArrayFloat64,
    waterbody_storage_in: ArrayFloat64,
    retention_storage_in: ArrayFloat32,
    over_abstraction_in: ArrayFloat32,
    actual_evaporation_in: ArrayFloat32,
    waterbody_inflow_in: ArrayFloat32,
    retention_inflow_in: ArrayFloat32,
    retention_outflow_in: ArrayFloat32,
    discharge_vol_sum_in: ArrayFloat32,
    wb_processed_in: ArrayBool,
    # Target output state
    substep_discharge_out: ArrayFloat32,
    river_storage_out: ArrayFloat64,
    waterbody_storage_out: ArrayFloat64,
    retention_storage_out: ArrayFloat32,
    over_abstraction_out: ArrayFloat32,
    actual_evaporation_out: ArrayFloat32,
    waterbody_inflow_out: ArrayFloat32,
    retention_inflow_out: ArrayFloat32,
    retention_outflow_out: ArrayFloat32,
    discharge_vol_sum_out: ArrayFloat32,
    wb_processed_out: ArrayBool,
    # Pre-allocated buffers
    sideflow_substep_m3: ArrayFloat64,
    evap_substep_m3: ArrayFloat64,
    wb_outflow_substep_m3: ArrayFloat64,
    retention_inflow_limit_sub: ArrayFloat64,
    retention_max_outflow_limit_sub: ArrayFloat64,
    wb_outflow_avail_buf: ArrayFloat64,
    wb_extra_lateral_m3_buf: ArrayFloat32,
    net_vol_m3: ArrayFloat64,
    # Static properties & Pre-computed terms
    upstream_matrix: TwoDArrayInt32,
    is_waterbody_outflow: ArrayBool,
    waterbody_id: ArrayInt32,
    cfl_river_length: ArrayFloat32,  # Pre-computed: 0.85 * river_length
    river_length: ArrayFloat32,
    inv_reach_length: ArrayFloat32,
    inv_cell_area: ArrayFloat32,
    river_width: ArrayFloat32,
    bed_elevation: ArrayFloat32,
    g_manning_n_sq: ArrayFloat32,  # Pre-computed: GRAVITY * manning_n_sq
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
) -> bool:
    # Scalar constants
    MIN_WET_DEPTH_M = np.float32(0.01)
    GRAVITY_ACCELERATION = np.float32(9.80665)
    SQRT_GRAVITY = np.float32(3.1315574)
    MAX_ALLOWED_DEPTH_M = np.float32(100.0)

    inv_dt_substep = np.float32(1.0) / dt_substep
    g_dt_substep = GRAVITY_ACCELERATION * dt_substep

    # 1. State Buffer Initialization
    river_storage_out[:] = river_storage_in
    waterbody_storage_out[:] = waterbody_storage_in
    retention_storage_out[:] = retention_storage_in
    over_abstraction_out[:] = over_abstraction_in
    actual_evaporation_out[:] = actual_evaporation_in
    waterbody_inflow_out[:] = waterbody_inflow_in
    retention_inflow_out[:] = retention_inflow_in
    retention_outflow_out[:] = retention_outflow_in
    substep_discharge_out[:] = substep_discharge_in
    discharge_vol_sum_out[:] = discharge_vol_sum_in
    wb_processed_out[:] = wb_processed_in

    lateral_flux_m3.fill(np.float32(0.0))
    next_substep_discharge_m3_s.fill(np.float32(0.0))
    net_vol_m3.fill(0.0)
    wb_extra_lateral_m3_buf.fill(np.float32(0.0))
    wb_outflow_avail_buf[:] = wb_outflow_substep_m3[:]

    wb_extra_lateral_m3_buf[n_kinematic:inertial_end].fill(np.float32(0.0))
    wb_outflow_avail_buf[:] = wb_outflow_substep_m3[:]

    # 2. Waterbody Outflows Pass
    for i in range(n_kinematic, inertial_end):
        for j in range(max_up_connections):
            up_node = upstream_matrix[i, j]
            if up_node == -1:
                break
            if is_waterbody_outflow[up_node]:
                wb_id = waterbody_id[up_node]
                if wb_id != -1 and wb_outflow_avail_buf[wb_id] > 0.0:
                    actual_outflow = min(
                        wb_outflow_avail_buf[wb_id], waterbody_storage_out[wb_id]
                    )
                    if actual_outflow > 0.0:
                        waterbody_storage_out[wb_id] -= actual_outflow
                        wb_extra_lateral_m3_buf[i] += np.float32(actual_outflow)
                    wb_outflow_avail_buf[wb_id] = 0.0
                    wb_processed_out[wb_id] = True

    # --- Pass 1 & 2: Local Inertial Momentum Solver ---
    error_flag = 0
    for i in prange(n_kinematic, inertial_end):  # ty:ignore[not-iterable]
        node_lateral_inflow = (
            np.float32(sideflow_substep_m3[i]) + wb_extra_lateral_m3_buf[i]
        )

        # Retention node calculation
        ret_id = retention_node_id[i]
        if ret_id != -1:
            q_in_pos = max(substep_discharge_in[i], np.float32(0.0))
            avail_flow_rate = (node_lateral_inflow * inv_dt_substep) + q_in_pos
            inflow_limit = retention_inflow_limit_sub[ret_id]
            max_outflow_limit = retention_max_outflow_limit_sub[ret_id]
            river_volume_sub = avail_flow_rate * dt_substep
            is_rising_limb = avail_flow_rate > previous_discharge_m3_s[i]

            (
                diverted_vol,
                outflow_vol,
                retention_storage_out[ret_id],
                river_volume_sub,
            ) = compute_retention_routing(
                dt=dt_substep,
                river_volume_m3=river_volume_sub,
                discharge_before_diversion_m3_s=avail_flow_rate,
                is_rising_limb=is_rising_limb,
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
            node_lateral_inflow = river_volume_sub - (q_in_pos * dt_substep)

        lateral_flux_m3[i] = node_lateral_inflow

        # Pre-fetch properties
        bed_elev_node = bed_elevation[i]
        width_m = river_width[i]
        inv_length_m = inv_reach_length[i]
        inv_area_m2 = inv_cell_area[i]
        avail_vol_f32 = max(np.float32(river_storage_in[i]), np.float32(0.0))

        flow_depth = max(avail_vol_f32 * inv_area_m2, np.float32(0.0))

        if not np.isfinite(flow_depth):
            next_substep_discharge_m3_s[i] = np.float32(0.0)
            error_flag += 1
            continue

        water_stage_node = bed_elev_node + flow_depth
        ds = ds_node[i]

        # Stage calculation with Boundary Conditions
        if is_pit[i]:
            if is_ocean_pit[i]:
                # 1. Ocean Outlets: Fixed 0m MSL Boundary
                water_stage_ds = np.float32(0.0)
                bed_elev_ds = bed_elev_node
            else:
                # 2. Inland River Pits: Slope-floored normal depth
                dx = river_length[i]
                eff_slope = pit_slope[i]
                bed_elev_ds = bed_elev_node - eff_slope * dx
                water_stage_ds = bed_elev_ds + flow_depth
        elif ds != -1:
            bed_elev_ds = bed_elevation[ds]
            wb_ds = waterbody_id[ds]
            if wb_ds == -1:
                flow_depth_ds = max(
                    np.float32(river_storage_in[ds]) * inv_cell_area[ds],
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

        if (effective_depth < MIN_WET_DEPTH_M) or (avail_vol_f32 <= np.float32(0.0)):
            next_substep_discharge_m3_s[i] = np.float32(0.0)
            continue

        cross_sectional_area = effective_depth * width_m

        if (not np.isfinite(cross_sectional_area)) or (
            cross_sectional_area >= np.float32(1e6)
        ):
            next_substep_discharge_m3_s[i] = np.float32(0.0)
            error_flag += 1
            continue

        hydraulic_radius = max(effective_depth, np.float32(1e-6))
        water_slope = (water_stage_ds - water_stage_node) * inv_length_m

        r_4_3 = hydraulic_radius * np.cbrt(hydraulic_radius)
        friction_denom_term = max(r_4_3 * cross_sectional_area, np.float32(1e-6))

        q_in = substep_discharge_in[i]
        friction_factor_val = dt_substep * g_manning_n_sq[i]
        friction_denom = np.float32(1.0) + (
            friction_factor_val * abs(q_in) / friction_denom_term
        )

        computed_discharge = (
            q_in - g_dt_substep * cross_sectional_area * water_slope
        ) / friction_denom

        # Boundary condition masking
        is_valid_ds = (ds != -1) and (not is_pit[i])
        is_boundary = (
            (ds == -1) or is_pit[i] or (is_valid_ds and waterbody_id[ds] != -1)
        )
        if is_boundary:
            computed_discharge = max(computed_discharge, np.float32(0.0))

        critical_discharge = (
            cross_sectional_area * SQRT_GRAVITY * np.sqrt(effective_depth)
        )
        computed_discharge = min(
            max(computed_discharge, -critical_discharge), critical_discharge
        )

        if computed_discharge > np.float32(0.0):
            max_q_avail = avail_vol_f32 * inv_dt_substep
            computed_discharge = min(computed_discharge, max_q_avail)
        elif computed_discharge < np.float32(0.0):
            if is_valid_ds and not is_pit[ds]:
                max_q_avail = np.float32(river_storage_in[ds]) * inv_dt_substep
                computed_discharge = -min(abs(computed_discharge), max_q_avail)
            else:
                computed_discharge = np.float32(0.0)

        if not np.isfinite(computed_discharge):
            next_substep_discharge_m3_s[i] = np.float32(0.0)
            error_flag += 1
            continue

        next_substep_discharge_m3_s[i] = computed_discharge

    if error_flag > 0:
        return False

    # --- Post-Pass: Waterbody Inflows ---
    for i in range(n_kinematic, inertial_end):
        ds = ds_node[i]
        if ds != -1 and not is_pit[i]:
            wb_ds_id = waterbody_id[ds]
            if wb_ds_id != -1:
                vol_out = next_substep_discharge_m3_s[i] * dt_substep
                waterbody_storage_out[wb_ds_id] += np.float64(vol_out)
                if vol_out > np.float32(0.0):
                    waterbody_inflow_out[wb_ds_id] += vol_out

    # --- Pass 3: Mass Balance & CFL Check ---
    error_flag = 0
    for i in prange(n_kinematic, inertial_end):  # ty:ignore[not-iterable]
        vol_in = kinematic_inflow_rate[i] * dt_substep + lateral_flux_m3[i]

        for j in range(max_up_connections):
            up_node = upstream_matrix[i, j]
            if up_node == -1:
                break
            if not is_waterbody_outflow[up_node] and not (
                use_kinematic[up_node] and waterbody_id[up_node] == -1
            ):
                vol_in += next_substep_discharge_m3_s[up_node] * dt_substep

        vol_out = next_substep_discharge_m3_s[i] * dt_substep
        net_vol = np.float64(vol_in - vol_out)
        net_vol_m3[i] = net_vol
        discharge_vol_sum_out[i] = discharge_vol_sum_in[i] + vol_out

        st = river_storage_in[i] + net_vol

        over_abs = over_abstraction_in[i]
        if st < 0.0:
            over_abs += np.float32(-st)
            st = 0.0
        over_abstraction_out[i] = over_abs

        evap_actual = np.float32(min(evap_substep_m3[i], st))
        evap_actual = max(evap_actual, np.float32(0.0))
        actual_evaporation_out[i] = actual_evaporation_in[i] + evap_actual
        st -= np.float64(evap_actual)

        river_storage_out[i] = st

        depth_new = np.float32(st) * inv_cell_area[i]

        if depth_new >= MAX_ALLOWED_DEPTH_M or not np.isfinite(st):
            substep_discharge_out[i] = next_substep_discharge_m3_s[i]
            error_flag += 1
            continue

        effective_depth_new = max(depth_new, MIN_WET_DEPTH_M)
        celerity = np.sqrt(GRAVITY_ACCELERATION * effective_depth_new)
        wet_mask = np.float32(depth_new > MIN_WET_DEPTH_M)

        flow_velocity = wet_mask * (
            abs(next_substep_discharge_m3_s[i]) / (effective_depth_new * river_width[i])
        )

        max_dt_cfl = cfl_river_length[i] / (celerity + flow_velocity + np.float32(1e-9))

        if dt_substep > max_dt_cfl:
            substep_discharge_out[i] = next_substep_discharge_m3_s[i]
            error_flag += 1
            continue

        substep_discharge_out[i] = next_substep_discharge_m3_s[i]

    return error_flag == 0


@njit(cache=True)
def _run_inertial_substeps(
    num_inertial_substeps: int,
    dt_f32: np.float32,
    n_kinematic: int,
    inertial_end: int,
    max_up_connections: int,
    previous_discharge_m3_s: ArrayFloat32,
    river_storage_m3: ArrayFloat64,
    sideflow_m3: ArrayFloat32,
    evaporation_m3: ArrayFloat32,
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
    over_abstraction_m3: ArrayFloat32,
    actual_evaporation_m3: ArrayFloat32,
    waterbody_inflow_m3: ArrayFloat32,
    retention_inflow_m3: ArrayFloat32,
    retention_outflow_m3: ArrayFloat32,
    updated_discharge_m3_s: ArrayFloat32,
    lateral_flux_m3: ArrayFloat32,
    next_substep_discharge_m3_s: ArrayFloat32,
    kinematic_inflow_rate: ArrayFloat32,
    discharge_vol_sum_m3: ArrayFloat32,
) -> None:
    dt_f64 = np.float64(dt_f32)
    original_substep_dt = dt_f64 / np.float64(num_inertial_substeps)
    current_substep_dt = original_substep_dt
    t_simulated = np.float64(0.0)
    MIN_DT_THRESHOLD = np.float64(1e-6)

    g_manning_n_sq = np.float32(9.80665) * manning_n_sq
    cfl_river_length = np.float32(0.85) * river_length

    sideflow_m3_f64 = sideflow_m3.astype(np.float64)
    evaporation_m3_f64 = evaporation_m3.astype(np.float64)
    outflow_per_waterbody_m3_f64 = outflow_per_waterbody_m3.astype(np.float64)
    retention_max_storage_m3_f64 = retention_max_storage_m3.astype(np.float64)

    sideflow_substep_m3_buf = np.empty(sideflow_m3.size, dtype=np.float64)
    evap_substep_m3_buf = np.empty(evaporation_m3.size, dtype=np.float64)
    wb_outflow_substep_m3_buf = np.empty(
        outflow_per_waterbody_m3.size, dtype=np.float64
    )
    retention_inflow_limit_sub_buf = np.empty(
        retention_max_storage_m3.size, dtype=np.float64
    )
    retention_max_outflow_limit_sub_buf = np.empty(
        retention_max_storage_m3.size, dtype=np.float64
    )

    wb_outflow_avail_buf = np.empty(waterbody_storage_m3.size, dtype=np.float64)
    wb_extra_lateral_m3_buf = np.empty(inertial_end, dtype=np.float32)
    net_vol_m3_buf = np.empty(river_storage_m3.size, dtype=np.float64)

    river_storage_A = river_storage_m3.copy()
    waterbody_storage_A = waterbody_storage_m3.copy()
    retention_storage_A = retention_storage_m3.copy()
    over_abstraction_A = over_abstraction_m3.copy()
    actual_evaporation_A = actual_evaporation_m3.copy()
    waterbody_inflow_A = waterbody_inflow_m3.copy()
    retention_inflow_A = retention_inflow_m3.copy()
    retention_outflow_A = retention_outflow_m3.copy()
    substep_discharge_A = previous_discharge_m3_s.copy()
    discharge_vol_sum_A = discharge_vol_sum_m3.copy()
    discharge_vol_sum_A.fill(np.float32(0.0))
    wb_processed_A = np.zeros(waterbody_storage_m3.size, dtype=np.bool_)

    river_storage_B = np.empty_like(river_storage_A)
    waterbody_storage_B = np.empty_like(waterbody_storage_A)
    retention_storage_B = np.empty_like(retention_storage_A)
    over_abstraction_B = np.empty_like(over_abstraction_A)
    actual_evaporation_B = np.empty_like(actual_evaporation_A)
    waterbody_inflow_B = np.empty_like(waterbody_inflow_A)
    retention_inflow_B = np.empty_like(retention_inflow_A)
    retention_outflow_B = np.empty_like(retention_outflow_A)
    substep_discharge_B = np.empty_like(substep_discharge_A)
    discharge_vol_sum_B = np.empty_like(discharge_vol_sum_A)
    wb_processed_B = np.empty_like(wb_processed_A)

    while t_simulated < dt_f64 - 1e-9:
        if t_simulated + current_substep_dt > dt_f64:
            current_substep_dt = dt_f64 - t_simulated

        fraction_f64 = current_substep_dt / dt_f64
        substep_dt_f32 = np.float32(current_substep_dt)

        n_nodes = sideflow_m3.size
        for i in range(n_nodes):
            sideflow_substep_m3_buf[i] = sideflow_m3_f64[i] * fraction_f64
            evap_substep_m3_buf[i] = evaporation_m3_f64[i] * fraction_f64

        n_wb = outflow_per_waterbody_m3.size
        for i in range(n_wb):
            wb_outflow_substep_m3_buf[i] = (
                outflow_per_waterbody_m3_f64[i] * fraction_f64
            )

        n_ret = retention_max_storage_m3.size
        for i in range(n_ret):
            ret_max = retention_max_storage_m3_f64[i] * fraction_f64
            retention_inflow_limit_sub_buf[i] = np.float64(0.20) * ret_max
            retention_max_outflow_limit_sub_buf[i] = np.float64(0.05) * ret_max

        success = _run_single_inertial_substep(
            dt_substep=substep_dt_f32,
            n_kinematic=n_kinematic,
            inertial_end=inertial_end,
            max_up_connections=max_up_connections,
            substep_discharge_in=substep_discharge_A,
            river_storage_in=river_storage_A,
            waterbody_storage_in=waterbody_storage_A,
            retention_storage_in=retention_storage_A,
            over_abstraction_in=over_abstraction_A,
            actual_evaporation_in=actual_evaporation_A,
            waterbody_inflow_in=waterbody_inflow_A,
            retention_inflow_in=retention_inflow_A,
            retention_outflow_in=retention_outflow_A,
            discharge_vol_sum_in=discharge_vol_sum_A,
            wb_processed_in=wb_processed_A,
            substep_discharge_out=substep_discharge_B,
            river_storage_out=river_storage_B,
            waterbody_storage_out=waterbody_storage_B,
            retention_storage_out=retention_storage_B,
            over_abstraction_out=over_abstraction_B,
            actual_evaporation_out=actual_evaporation_B,
            waterbody_inflow_out=waterbody_inflow_B,
            retention_inflow_out=retention_inflow_B,
            retention_outflow_out=retention_outflow_B,
            discharge_vol_sum_out=discharge_vol_sum_B,
            wb_processed_out=wb_processed_B,
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
            cfl_river_length=cfl_river_length,
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
        )

        if success:
            river_storage_A, river_storage_B = river_storage_B, river_storage_A
            waterbody_storage_A, waterbody_storage_B = (
                waterbody_storage_B,
                waterbody_storage_A,
            )
            retention_storage_A, retention_storage_B = (
                retention_storage_B,
                retention_storage_A,
            )
            over_abstraction_A, over_abstraction_B = (
                over_abstraction_B,
                over_abstraction_A,
            )
            actual_evaporation_A, actual_evaporation_B = (
                actual_evaporation_B,
                actual_evaporation_A,
            )
            waterbody_inflow_A, waterbody_inflow_B = (
                waterbody_inflow_B,
                waterbody_inflow_A,
            )
            retention_inflow_A, retention_inflow_B = (
                retention_inflow_B,
                retention_inflow_A,
            )
            retention_outflow_A, retention_outflow_B = (
                retention_outflow_B,
                retention_outflow_A,
            )
            substep_discharge_A, substep_discharge_B = (
                substep_discharge_B,
                substep_discharge_A,
            )
            discharge_vol_sum_A, discharge_vol_sum_B = (
                discharge_vol_sum_B,
                discharge_vol_sum_A,
            )
            wb_processed_A, wb_processed_B = wb_processed_B, wb_processed_A

            t_simulated += current_substep_dt
            current_substep_dt = min(current_substep_dt * 1.25, original_substep_dt)
        else:
            current_substep_dt *= 0.5

            if current_substep_dt < MIN_DT_THRESHOLD:
                raise RuntimeError(
                    "Substep dt reduced below minimum threshold. Integration Unstable."
                )

    river_storage_m3[:] = river_storage_A[:]
    waterbody_storage_m3[:] = waterbody_storage_A[:]
    retention_storage_m3[:] = retention_storage_A[:]
    over_abstraction_m3[:] = over_abstraction_A[:]
    actual_evaporation_m3[:] = actual_evaporation_A[:]
    waterbody_inflow_m3[:] = waterbody_inflow_A[:]
    retention_inflow_m3[:] = retention_inflow_A[:]
    retention_outflow_m3[:] = retention_outflow_A[:]
    discharge_vol_sum_m3[:] = discharge_vol_sum_A[:]

    inv_dt_f32 = np.float32(1.0) / dt_f32
    for i in range(n_kinematic, inertial_end):
        updated_discharge_m3_s[i] = discharge_vol_sum_A[i] * inv_dt_f32

    for wb_id in range(waterbody_storage_m3.size):
        if wb_processed_A[wb_id]:
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
        """Initializes the LocalInertial routing instance.

        Args:
            dt: Routing model main timestep duration in seconds.
            river_network: PyFLWDIR flow direction raster object.
            river_length: Channel length per node.
            river_width: Channel width per node.
            waterbody_ids: Waterbody ID per node (-1 if standard river node).
            river_ids: Reach identifier per node.
            is_waterbody_outflow: Mask identifying waterbody outlet nodes.
            retention_max_storage_m3: Retention basin capacities.
            retention_node_id: Node index mapping for retention basins.
            controlled_retention: Flag array for controlled retention operation.
            retention_basin_release_threshold_factor: Factor controlling release rate.
            bankfull_river_elevation_m: Bed elevation above datum per node.
            manning_n: Manning's roughness coefficient per node.
            use_kinematic: Mask indicating nodes that use kinematic wave formulation.
            rivers_gdf: GeoDataFrame containing COMID, downstream_ID, and slope attributes.
            min_slope: Minimum slope floor threshold for non-ocean pit boundaries.
        """
        super().__init__(
            dt,
            river_network,
            waterbody_ids,
            is_waterbody_outflow,
            retention_basin_release_threshold_factor,
        )

        assert np.all(river_length > 0), "River length must be strictly greater than 0."

        self.river_length = np.maximum(river_length, np.float32(1.0))
        self.river_width = river_width.astype(np.float32)
        self.river_ids = river_ids
        self.bed_elevation = bankfull_river_elevation_m.astype(np.float32)
        self.manning_n = manning_n.astype(np.float32)
        self.use_kinematic = use_kinematic
        self.retention_node_id = retention_node_id
        self.retention_max_storage_m3 = retention_max_storage_m3
        self.controlled_retention = controlled_retention

        # Classify pits and calculate effective boundary slope
        n_nodes_total = self.is_pit.size
        is_ocean_pit_orig = np.zeros(n_nodes_total, dtype=np.bool_)
        pit_slope_orig = np.zeros(n_nodes_total, dtype=np.float32)

        for node_idx in range(n_nodes_total):
            if self.is_pit[node_idx]:
                rid = self.river_ids[node_idx]
                if rid in rivers_gdf.index:
                    row = rivers_gdf.loc[rid]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]

                    ds_id = row["downstream_ID"]
                    if ds_id == -1:
                        is_ocean_pit_orig[node_idx] = True
                    else:
                        raw_slope = float(row["slope"])
                        pit_slope_orig[node_idx] = np.float32(max(raw_slope, min_slope))
                else:
                    # Fallback if COMID not in GeoDataFrame
                    pit_slope_orig[node_idx] = np.float32(min_slope)

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

        self._Q_prev_perm = np.empty(n_up_nodes, dtype=np.float32)
        self._river_storage_perm = np.empty(n_up_nodes, dtype=np.float64)
        self._sideflow_perm = np.empty(n_up_nodes, dtype=np.float32)
        self._evaporation_perm = np.empty(n_up_nodes, dtype=np.float32)
        self._alpha_perm = np.empty(n_up_nodes, dtype=np.float32)
        self._beta_perm = np.empty(n_up_nodes, dtype=np.float32)

        self._over_abstraction_perm = np.empty(n_up_nodes, dtype=np.float32)
        self._actual_evaporation_perm = np.empty(n_up_nodes, dtype=np.float32)
        self._updated_discharge_perm = np.empty(n_up_nodes, dtype=np.float32)
        self._lateral_flux_perm = np.empty(n_up_nodes, dtype=np.float32)
        self._next_substep_discharge_perm = np.empty(n_up_nodes, dtype=np.float32)
        self._kinematic_inflow_rate_perm = np.empty(n_up_nodes, dtype=np.float32)
        self._discharge_vol_sum_perm = np.empty(n_up_nodes, dtype=np.float32)

        self._waterbody_inflow_perm = np.empty(
            is_waterbody_outflow.sum(), dtype=np.float32
        )
        n_retention_basins = (
            int(np.max(retention_node_id)) + 1
            if retention_node_id.size > 0 and np.max(retention_node_id) >= 0
            else 0
        )
        self._retention_inflow_perm = np.empty(
            max(n_retention_basins, 1), dtype=np.float32
        )
        self._retention_outflow_perm = np.empty(
            max(n_retention_basins, 1), dtype=np.float32
        )

    def calculate_river_storage_from_discharge(
        self,
        discharge: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_length: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        waterbody_id: ArrayInt32,
    ) -> ArrayFloat64:
        """Calculates channel volume storage from discharge using kinematic wave relationship.

        Args:
            discharge: Discharge rate per node.
            river_storage_alpha: Kinematic wave alpha parameter per node.
            river_length: Reach length per node.
            river_storage_beta: Kinematic wave beta parameter per node.
            waterbody_id: Waterbody ID per node (-1 if standard river node).

        Returns:
            Calculated total river storage volume array.
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
        """Calculates discharge from channel volume storage using kinematic wave relationship.

        Args:
            river_storage: Channel volume storage per node.
            river_storage_alpha: Kinematic wave alpha parameter per node.
            river_storage_beta: Kinematic wave beta parameter per node.
            river_length: Reach length per node.
            waterbody_id: Waterbody ID per node (-1 if standard river node).

        Returns:
            Calculated discharge rate array (NaN for waterbodies).
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
        Q: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        maximum_abstraction_ratio: float = 0.9,
    ) -> ArrayFloat64:
        """Gets available abstraction volume based on maximum allowed fraction of total storage.

        Args:
            Q: Current discharge per node.
            river_storage_alpha: Kinematic wave alpha parameter per node.
            river_storage_beta: Kinematic wave beta parameter per node.
            maximum_abstraction_ratio: Maximum safe fraction of total channel volume allowed.

        Returns:
            Available abstraction volume array per node.
        """
        return (
            self.get_total_storage(Q, river_storage_alpha, river_storage_beta)
            * maximum_abstraction_ratio
        )

    def get_total_storage(
        self,
        Q: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
    ) -> ArrayFloat64:
        """Gets total channel storage using original grid-ordered arrays.

        Args:
            Q: Current discharge per node.
            river_storage_alpha: Kinematic wave alpha parameter per node.
            river_storage_beta: Kinematic wave beta parameter per node.

        Returns:
            Total storage volume array.
        """
        total_storage: ArrayFloat64 = self.calculate_river_storage_from_discharge(
            discharge=Q,
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
        lateral_flux_m3: ArrayFloat32,
        next_substep_discharge_m3_s: ArrayFloat32,
        kinematic_inflow_rate: ArrayFloat32,
        discharge_vol_sum_m3: ArrayFloat32,
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
        """Core routing step executed on internally re-ordered topology arrays.

        Performs sequential execution of kinematic wave routing for upstream segments,
        sub-sampled adaptive local inertial routing for mainstem reaches, direct lake
        evaporation/inflow accounting, and terminal outflow summation.

        Args:
            routing_timestep_s: Timestep length in seconds.
            previous_discharge_m3_s: Initial discharge state array.
            river_storage_m3: Initial river volume storage state array.
            sideflow_m3: Total lateral inflow volume over timestep.
            evaporation_m3: Potential channel evaporation volume over timestep.
            waterbody_storage_m3: Reservoir and lake storage state array.
            outflow_per_waterbody_m3: Target outflow volume per waterbody.
            upstream_matrix: Downstream-to-upstream adjacency array.
            is_waterbody_outflow: Lake outlet node identifier mask.
            waterbody_id: Waterbody index per node.
            river_storage_alpha: Kinematic wave alpha parameter.
            river_storage_beta: Kinematic wave beta parameter.
            river_length: Channel length per node.
            inv_reach_length: Reciprocal reach length.
            inv_cell_area: Reciprocal channel area.
            river_width: Channel width per node.
            bed_elevation: Bed elevation per node.
            manning_n_sq: Squared Manning's roughness coefficient.
            retention_storage_m3: Retention basin storage state.
            retention_max_storage_m3: Maximum retention storage capacities.
            retention_node_id: Node-to-retention basin mapping vector.
            controlled_retention: Retention basin control mode flag vector.
            retention_activation_threshold_m3_s: Discharge threshold triggering retention inflow.
            retention_basin_release_threshold_factor: Release threshold factor for retention.
            ds_node: Downstream node index vector.
            is_pit: Sink/pour point mask vector.
            is_ocean_pit: Ocean pour point mask vector.
            pit_slope: Bed slope boundary array for non-ocean pits.
            use_kinematic: Routing mode flag vector.
            n_kinematic: Count of kinematic nodes.
            n_inertial: Count of local inertial nodes.
            over_abstraction_m3: Output storage deficit tracker.
            actual_evaporation_m3: Output actual evaporation volume tracker.
            waterbody_inflow_m3: Output total waterbody inflow volume array.
            retention_inflow_m3: Output total retention inflow volume array.
            retention_outflow_m3: Output total retention outflow volume array.
            updated_discharge_m3_s: Output average discharge array.
            lateral_flux_m3: Scratch workspace array for lateral fluxes.
            next_substep_discharge_m3_s: Scratch workspace array for target discharges.
            kinematic_inflow_rate: Scratch workspace array for upstream kinematic rates.
            discharge_vol_sum_m3: Scratch workspace array for discharge volume sums.

        Returns:
            Tuple containing:
                - updated_discharge_m3_s: New discharge array.
                - river_storage_m3: New river storage array.
                - actual_evaporation_m3: Actual evaporation volume array.
                - over_abstraction_m3: Over-abstraction volume array.
                - waterbody_inflow_m3: Inflow volume into waterbodies array.
                - retention_inflow_m3: Diversion inflow into retention basins array.
                - retention_outflow_m3: Release outflow from retention basins array.
                - terminal_waterbody_outflow_m3: Outflow volume from unconnected lakes.
        """
        n_cells: int = previous_discharge_m3_s.size
        max_up_connections: int = upstream_matrix.shape[1]

        over_abstraction_m3.fill(np.float32(0.0))
        actual_evaporation_m3.fill(np.float32(0.0))
        waterbody_inflow_m3.fill(np.float32(0.0))
        retention_inflow_m3.fill(np.float32(0.0))
        retention_outflow_m3.fill(np.float32(0.0))
        updated_discharge_m3_s.fill(np.float32(0.0))
        lateral_flux_m3.fill(np.float32(0.0))
        next_substep_discharge_m3_s.fill(np.float32(0.0))
        kinematic_inflow_rate.fill(np.float32(0.0))
        discharge_vol_sum_m3.fill(np.float32(0.0))

        n_other_start = n_kinematic + n_inertial
        for i in range(n_other_start, n_cells):
            updated_discharge_m3_s[i] = np.float32(np.nan)

        # Direct lateral inflow and evaporation on waterbody nodes
        for i in range(n_other_start, n_cells):
            wb_id = waterbody_id[i]
            if wb_id != -1:
                waterbody_storage_m3[wb_id] += np.float64(sideflow_m3[i])

                evap_sub = np.float32(
                    min(np.float64(evaporation_m3[i]), waterbody_storage_m3[wb_id])
                )
                evap_sub = max(evap_sub, np.float32(0.0))
                actual_evaporation_m3[i] = evap_sub
                waterbody_storage_m3[wb_id] -= np.float64(evap_sub)

        H_MIN_WET: np.float32 = np.float32(0.01)
        GRAVITY_ACCELERATION: np.float32 = np.float32(9.80665)
        CFL_SAFETY_FACTOR: np.float32 = np.float32(0.7)
        dt_f32: np.float32 = np.float32(routing_timestep_s)
        inv_dt_f32: np.float32 = np.float32(1.0) / dt_f32

        inertial_outflow_per_waterbody = outflow_per_waterbody_m3.copy()

        # Kinematic wave routing
        for i in range(n_kinematic):
            upstream_inflow_m3_s: np.float32 = np.float32(0.0)
            node_sideflow: np.float32 = sideflow_m3[i]

            for j in range(max_up_connections):
                up_node = upstream_matrix[i, j]
                if up_node == -1:
                    break

                if is_waterbody_outflow[up_node]:
                    wb_id = waterbody_id[up_node]
                    wb_outflow: np.float32 = min(
                        inertial_outflow_per_waterbody[wb_id],
                        np.float32(waterbody_storage_m3[wb_id]),
                    )
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
                    is_rising_limb=is_rising_limb,
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
                Qin=upstream_inflow_m3_s,
                Qold=previous_discharge_m3_s[i],
                Qside=node_sideflow * inv_dt_f32,
                evaporation_m3_s=evaporation_m3[i] * inv_dt_f32,
                alpha=river_storage_alpha[i],
                beta=river_storage_beta[i],
                deltaT=dt_f32,
                deltaX=river_length[i],
            )

            kinematic_discharge = max(kinematic_discharge, np.float32(0.0))
            updated_discharge_m3_s[i] = kinematic_discharge

            evap_vol = act_evap_rate * dt_f32
            actual_evaporation_m3[i] = evap_vol

            inflow_vol = upstream_inflow_m3_s * dt_f32 + node_sideflow
            outflow_vol = kinematic_discharge * dt_f32
            river_storage_m3[i] += np.float64(inflow_vol - outflow_vol - evap_vol)

            ds = ds_node[i]
            if ds != -1 and waterbody_id[ds] != -1:
                wb_ds_id = waterbody_id[ds]
                waterbody_storage_m3[wb_ds_id] += np.float64(outflow_vol)
                if outflow_vol > np.float32(0.0):
                    waterbody_inflow_m3[wb_ds_id] += outflow_vol

            if river_storage_m3[i] < np.float64(0.0):
                over_abstraction_m3[i] += np.float32(-river_storage_m3[i])
                river_storage_m3[i] = np.float64(0.0)

            assert np.isfinite(updated_discharge_m3_s[i])

        # Local inertial wave routing
        if n_inertial > 0:
            inertial_end: int = n_kinematic + n_inertial

            for i in range(n_kinematic, inertial_end):
                for j in range(max_up_connections):
                    up_node = upstream_matrix[i, j]
                    if up_node == -1:
                        break
                    if is_waterbody_outflow[up_node]:
                        continue
                    if use_kinematic[up_node] and waterbody_id[up_node] == -1:
                        kinematic_inflow_rate[i] += max(
                            updated_discharge_m3_s[up_node], np.float32(0.0)
                        )

            min_stable_dt: np.float32 = np.float32(1e9)

            for i in range(n_kinematic, inertial_end):
                est_inflow_vol = (
                    kinematic_inflow_rate[i] + sideflow_m3[i] * inv_dt_f32
                ) * dt_f32
                pred_storage = max(river_storage_m3[i] + est_inflow_vol, 0.0)
                pred_depth = max(np.float32(pred_storage) * inv_cell_area[i], H_MIN_WET)

                wave_celerity: np.float32 = np.sqrt(GRAVITY_ACCELERATION * pred_depth)

                pred_discharge = (
                    np.abs(previous_discharge_m3_s[i])
                    + kinematic_inflow_rate[i]
                    + sideflow_m3[i] * inv_dt_f32
                )
                flow_vel: np.float32 = pred_discharge / (pred_depth * river_width[i])

                max_dt: np.float32 = (CFL_SAFETY_FACTOR * river_length[i]) / (
                    wave_celerity + flow_vel + np.float32(1e-9)
                )
                if max_dt < min_stable_dt:
                    min_stable_dt = max_dt

            min_stable_dt = max(min_stable_dt, np.float32(1e-4))
            raw_substeps = int(np.ceil(dt_f32 / min_stable_dt))

            num_inertial_substeps: int = max(1, min(raw_substeps, 3600))
            _run_inertial_substeps(
                num_inertial_substeps=num_inertial_substeps,
                dt_f32=dt_f32,
                n_kinematic=n_kinematic,
                inertial_end=inertial_end,
                max_up_connections=max_up_connections,
                previous_discharge_m3_s=previous_discharge_m3_s,
                river_storage_m3=river_storage_m3,
                sideflow_m3=sideflow_m3,
                evaporation_m3=evaporation_m3,
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
                over_abstraction_m3=over_abstraction_m3,
                actual_evaporation_m3=actual_evaporation_m3,
                waterbody_inflow_m3=waterbody_inflow_m3,
                retention_inflow_m3=retention_inflow_m3,
                retention_outflow_m3=retention_outflow_m3,
                updated_discharge_m3_s=updated_discharge_m3_s,
                lateral_flux_m3=lateral_flux_m3,
                next_substep_discharge_m3_s=next_substep_discharge_m3_s,
                kinematic_inflow_rate=kinematic_inflow_rate,
                discharge_vol_sum_m3=discharge_vol_sum_m3,
            )

        terminal_waterbody_outflow_m3 = np.float32(0.0)
        for wb_id in range(waterbody_storage_m3.size):
            rem_outflow = inertial_outflow_per_waterbody[wb_id]
            if rem_outflow > np.float32(0.0) and waterbody_storage_m3[wb_id] > 0.0:
                actual_term_outflow = min(
                    np.float64(rem_outflow),
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
        """Executes a single river routing timestep over the network.

        Handles permutation of input arrays into optimized contiguous memory layouts,
        triggers computational routing kernel `_step`, and un-permutes outputs back to
        the standard grid node indexing scheme.

        Args:
            Q_prev_m3_s: Previous timestep discharge per node.
            river_storage_m3: Previous timestep river storage volume per node.
            sideflow_m3: Lateral surface/subsurface inflow volume during timestep.
            evaporation_m3: Potential channel evaporation volume during timestep.
            waterbody_storage_m3: Storage volume in lakes/reservoirs.
            outflow_per_waterbody_m3: Prescribed or calculated outflow volume per waterbody.
            retention_storage_m3: Retention basin volume storage.
            river_storage_alpha: Kinematic wave alpha parameter per node.
            river_storage_beta: Kinematic wave beta parameter per node.
            retention_activation_threshold_m3_s: Retention activation threshold per node.

        Returns:
            Tuple containing:
                - Q: New discharge per node.
                - river_storage_m3: Updated river storage volume per node.
                - actual_evaporation_m3: Actual channel evaporation volume per node.
                - over_abstraction_m3: Unmet abstraction volume deficit per node.
                - waterbody_storage_m3: Updated waterbody storage array.
                - waterbody_inflow_m3: Inflow volume into waterbodies array.
                - outflow_at_pits_m3: Total outflow volume exiting through domain sinks/pits.
                - retention_storage_m3: Updated retention storage array.
                - retention_inflow_m3: Inflow volume into retention basins array.
                - retention_outflow_m3: Outflow volume released from retention basins array.
        """
        np.take(Q_prev_m3_s, self.sorted_idxs, out=self._Q_prev_perm)
        np.take(river_storage_m3, self.sorted_idxs, out=self._river_storage_perm)
        np.take(sideflow_m3, self.sorted_idxs, out=self._sideflow_perm)
        np.take(evaporation_m3, self.sorted_idxs, out=self._evaporation_perm)
        np.take(river_storage_alpha, self.sorted_idxs, out=self._alpha_perm)
        np.take(river_storage_beta, self.sorted_idxs, out=self._beta_perm)

        (
            Q_perm,
            river_storage_perm,
            actual_evaporation_perm,
            over_abstraction_perm,
            waterbody_inflow_m3,
            retention_inflow_m3,
            retention_outflow_m3,
            terminal_wb_outflow_m3,
        ) = self._step(
            routing_timestep_s=self.dt,
            previous_discharge_m3_s=self._Q_prev_perm,
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
            lateral_flux_m3=self._lateral_flux_perm,
            next_substep_discharge_m3_s=self._next_substep_discharge_perm,
            kinematic_inflow_rate=self._kinematic_inflow_rate_perm,
            discharge_vol_sum_m3=self._discharge_vol_sum_perm,
        )

        outflow_at_pits_m3 = np.float32(0.0)
        for i in range(self.n_kinematic + self.n_inertial):
            if self._is_pit[i]:
                outflow_at_pits_m3 += Q_perm[i] * self.dt
        outflow_at_pits_m3 += terminal_wb_outflow_m3

        Q = Q_perm[self.inv_idxs]
        river_storage_m3[:] = river_storage_perm[self.inv_idxs]
        actual_evaporation_m3 = actual_evaporation_perm[self.inv_idxs]
        over_abstraction_m3 = over_abstraction_perm[self.inv_idxs]

        return (
            Q,
            river_storage_m3,
            actual_evaporation_m3,
            over_abstraction_m3,
            waterbody_storage_m3,
            waterbody_inflow_m3,
            outflow_at_pits_m3,
            retention_storage_m3,
            retention_inflow_m3,
            retention_outflow_m3,
        )
