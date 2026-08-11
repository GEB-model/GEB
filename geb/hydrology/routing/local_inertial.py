"""Local inertial routing algorithm for river networks."""

import numpy as np
import pyflwdir
from numba import njit

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    TwoDArrayInt32,
)

from .kinematic_wave import update_node_kinematic
from .routerbase import Router, compute_retention_routing


class LocalInertial(Router):
    """Local inertial routing algorithm.

    This class implements the 1D local inertial routing algorithm for river networks.
    """

    def __init__(
        self,
        dt: float | int,
        river_network: pyflwdir.FlwdirRaster,
        river_length: ArrayFloat32,
        waterbody_id: ArrayInt32,
        is_waterbody_outflow: ArrayBool,
        retention_max_storage_m3: ArrayFloat32,
        retention_node_id: ArrayInt32,
        controlled_retention: ArrayBool,
        retention_basin_release_threshold_factor: float,
        bed_elevation: ArrayFloat32,
        manning_n: ArrayFloat32,
    ) -> None:
        """Initializes the LocalInertial class."""
        super().__init__(
            dt,
            river_network,
            waterbody_id,
            is_waterbody_outflow,
            retention_basin_release_threshold_factor,
        )

        self.river_length = np.maximum(river_length, np.float32(1.0))

        # Recreate the mapper to construct downstream nodes (ds_node)
        mapper = np.full(river_network.size + 1, -1, dtype=np.int32)
        indices = np.arange(river_network.size, dtype=np.int32)[river_network.mask]
        mapper[indices] = np.arange(indices.size, dtype=np.int32)

        # Map flat unmasked downstream indices to masked indices
        unmasked_ds = river_network.idxs_ds[indices]
        self.ds_node = mapper[unmasked_ds]
        # Pits have no downstream node
        self.ds_node[self.is_pit] = -1

        self.bed_elevation = bed_elevation.astype(np.float32)
        self.manning_n = manning_n.astype(np.float32)

        # Retention basin parameters
        self.retention_max_storage_m3 = retention_max_storage_m3
        self.retention_node_id = retention_node_id
        self.controlled_retention = controlled_retention

    def calculate_river_storage_from_discharge(
        self,
        discharge: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_length: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        waterbody_id: ArrayInt32,
    ) -> ArrayFloat64:
        """Calculate the river storage from the discharge using the kinematic wave equation relation.

        Args:
            discharge: Discharge in m3/s for each cell.
            river_storage_alpha: Kinematic wave alpha parameter for each cell.
            river_length: Length of the river reach in m for each cell.
            river_storage_beta: Kinematic wave beta parameter for each cell.
            waterbody_id: Array indicating the waterbody ID for each cell (-1 if not a waterbody).

        Returns:
            A 1D array with the calculated river storage for each cell, in m3.
        """
        cross_sectional_area_of_flow: ArrayFloat64 = (
            river_storage_alpha
            * np.abs(discharge).astype(np.float64) ** river_storage_beta
        )
        river_storage: ArrayFloat64 = cross_sectional_area_of_flow * river_length
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
        """Calculate the discharge from the river storage using the kinematic wave equation relation.

        Args:
            river_storage: River storage in m3 for each cell.
            river_storage_alpha: Kinematic wave alpha parameter for each cell.
            river_storage_beta: Kinematic wave beta parameter for each cell.
            river_length: Length of the river reach in m for each cell.
            waterbody_id: Array indicating the waterbody ID for each cell (-1 if not a waterbody).

        Returns:
            A 1D array with the calculated discharge for each cell, in m3/s.

        """
        cross_sectional_area_of_flow: ArrayFloat32 = (
            river_storage / river_length
        ).astype(np.float32)
        discharge: ArrayFloat32 = (
            cross_sectional_area_of_flow / river_storage_alpha
        ) ** (1 / river_storage_beta)
        discharge[waterbody_id != -1] = np.nan
        return discharge

    def get_available_storage(
        self,
        Q: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        maximum_abstraction_ratio: float = 0.9,
    ) -> ArrayFloat64:
        """Get the available storage of the river network.

        Args:
            Q: Discharge in m3/s for each cell.
            river_storage_alpha: Kinematic wave alpha parameter for each cell.
            river_storage_beta: Kinematic wave beta parameter for each cell.
            maximum_abstraction_ratio: Maximum fraction of storage that can be abstracted.

        Returns:
            A 1D array with the available storage for each cell, in m3.
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
        """Get the total storage of the river network.

        Args:
            Q: Discharge in m3/s for each cell.
            river_storage_alpha: Kinematic wave alpha parameter for each cell.
            river_storage_beta: Kinematic wave beta parameter for each cell.

        Returns:
            A 1D array with the total storage for each cell, in m3.
        """
        total_storage: ArrayFloat64 = self.calculate_river_storage_from_discharge(
            discharge=Q,
            river_storage_alpha=river_storage_alpha,
            river_length=self.river_length,
            river_storage_beta=river_storage_beta,
            waterbody_id=self.waterbody_id,
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
        upstream_matrix_from_up_to_downstream: TwoDArrayInt32,
        idxs_up_to_downstream: ArrayInt32,
        is_waterbody_outflow: ArrayBool,
        waterbody_id: ArrayInt32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        river_length: ArrayFloat32,
        river_width: ArrayFloat32,
        bed_elevation: ArrayFloat32,
        manning_n: ArrayFloat32,
        retention_storage_m3: ArrayFloat32,
        retention_max_storage_m3: ArrayFloat32,
        retention_node_id: ArrayInt32,
        controlled_retention: ArrayBool,
        retention_activation_threshold_m3_s: ArrayFloat32,
        retention_basin_release_threshold_factor: np.float32,
        ds_node: ArrayInt32,
        is_pit: ArrayBool,
        use_kinematic: ArrayBool,
    ) -> tuple[
        ArrayFloat32,
        ArrayFloat64,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
    ]:
        n_cells: int = previous_discharge_m3_s.size
        n_up_nodes: int = upstream_matrix_from_up_to_downstream.shape[0]
        max_up_connections: int = upstream_matrix_from_up_to_downstream.shape[1]

        sorted_idxs: ArrayInt32 = np.empty(n_up_nodes, dtype=np.int32)
        sorted_orig_i: ArrayInt32 = np.empty(n_up_nodes, dtype=np.int32)

        n_kinematic: int = 0
        n_inertial: int = 0

        # Pass 1: Extract kinematic nodes into contiguous range [0 : n_kinematic]
        for i in range(n_up_nodes):
            node = idxs_up_to_downstream[i]
            if use_kinematic[node] and waterbody_id[node] == -1:
                sorted_idxs[n_kinematic] = node
                sorted_orig_i[n_kinematic] = i
                n_kinematic += 1

        # Pass 2: Extract inertial nodes into contiguous range [n_kinematic : n_kinematic + n_inertial]
        for i in range(n_up_nodes):
            node = idxs_up_to_downstream[i]
            if not use_kinematic[node] and waterbody_id[node] == -1:
                idx = n_kinematic + n_inertial
                sorted_idxs[idx] = node
                sorted_orig_i[idx] = i
                n_inertial += 1

        # Tracking arrays
        over_abstraction_m3: ArrayFloat32 = np.zeros(n_cells, dtype=np.float32)
        actual_evaporation_m3: ArrayFloat32 = np.zeros(n_cells, dtype=np.float32)
        waterbody_inflow_m3: ArrayFloat32 = np.zeros(
            waterbody_storage_m3.size, dtype=np.float32
        )
        retention_inflow_m3: ArrayFloat32 = np.zeros(
            retention_storage_m3.size, dtype=np.float32
        )
        retention_outflow_m3: ArrayFloat32 = np.zeros(
            retention_storage_m3.size, dtype=np.float32
        )
        updated_discharge_m3_s: ArrayFloat32 = np.zeros(n_cells, dtype=np.float32)

        # Pass 3: Waterbody nodes partitioning + immediate NaN assignment
        n_other_start = n_kinematic + n_inertial
        n_other = 0
        for i in range(n_up_nodes):
            node = idxs_up_to_downstream[i]
            if waterbody_id[node] != -1:
                idx = n_other_start + n_other
                sorted_idxs[idx] = node
                sorted_orig_i[idx] = i
                updated_discharge_m3_s[node] = np.float32(np.nan)
                n_other += 1

        # Buffer arrays
        lateral_flux_m3: ArrayFloat32 = np.zeros(n_cells, dtype=np.float32)
        next_substep_discharge_m3_s: ArrayFloat32 = np.zeros(n_cells, dtype=np.float32)
        kinematic_inflow_rate: ArrayFloat32 = np.zeros(n_cells, dtype=np.float32)
        pit_discharge_sum: ArrayFloat32 = np.zeros(n_cells, dtype=np.float32)

        H_MIN_WET: np.float32 = np.float32(1e-3)
        GRAVITY_M_PER_S2: np.float32 = np.float32(9.80665)
        SQRT_G: np.float32 = np.float32(3.1315574)
        CFL_SAFETY_FACTOR: np.float32 = np.float32(0.7)
        dt_f32: np.float32 = np.float32(routing_timestep_s)
        inv_dt_f32: np.float32 = np.float32(1.0) / dt_f32

        # -------------------------------------------------------------------------
        # Kinematic wave routing (Slices 0 : n_kinematic)
        # -------------------------------------------------------------------------
        for i in range(n_kinematic):
            node = sorted_idxs[i]
            orig_i = sorted_orig_i[i]

            upstream_inflow_m3_s: np.float32 = np.float32(0.0)
            node_sideflow_m3: np.float32 = sideflow_m3[node]

            for j in range(max_up_connections):
                up_node = upstream_matrix_from_up_to_downstream[orig_i, j]
                if up_node == -1:
                    break

                if is_waterbody_outflow[up_node]:
                    wb_id = waterbody_id[up_node]
                    wb_outflow_m3: np.float32 = min(
                        outflow_per_waterbody_m3[wb_id],
                        np.float32(waterbody_storage_m3[wb_id]),
                    )
                    waterbody_storage_m3[wb_id] -= np.float64(wb_outflow_m3)
                    node_sideflow_m3 += wb_outflow_m3
                elif waterbody_id[up_node] == -1:
                    up_q: np.float32 = (
                        updated_discharge_m3_s[up_node]
                        if use_kinematic[up_node]
                        else previous_discharge_m3_s[up_node]
                    )
                    upstream_inflow_m3_s += max(up_q, np.float32(0.0))

            node_retention_id = retention_node_id[node]
            if node_retention_id != -1:
                discharge_before_diversion_m3_s = (
                    (upstream_inflow_m3_s + previous_discharge_m3_s[node])
                    * np.float32(0.5)
                ) + (node_sideflow_m3 * inv_dt_f32)

                discharge_at_retention_basin_m3_per_dt = (
                    upstream_inflow_m3_s * dt_f32 + node_sideflow_m3
                )
                inflow_limit_m3 = (
                    np.float32(0.20) * retention_max_storage_m3[node_retention_id]
                )
                max_outflow_limit_m3 = (
                    np.float32(0.05) * retention_max_storage_m3[node_retention_id]
                )
                is_rising_limb = (
                    discharge_before_diversion_m3_s > previous_discharge_m3_s[node]
                )

                (
                    diverted_volume_m3,
                    outflow_volume_m3,
                    retention_storage_m3[node_retention_id],
                    discharge_at_retention_basin_m3_per_dt,
                ) = compute_retention_routing(
                    dt=dt_f32,
                    river_volume_m3=discharge_at_retention_basin_m3_per_dt,
                    discharge_before_diversion_m3_s=discharge_before_diversion_m3_s,
                    is_rising_limb=is_rising_limb,
                    retention_storage_m3=retention_storage_m3[node_retention_id],
                    retention_max_storage_m3=retention_max_storage_m3[
                        node_retention_id
                    ],
                    controlled_retention=controlled_retention[node_retention_id],
                    activation_threshold_m3_s=retention_activation_threshold_m3_s[
                        node_retention_id
                    ],
                    release_threshold_factor=retention_basin_release_threshold_factor,
                    inflow_limit_m3=inflow_limit_m3,
                    max_outflow_limit_m3=max_outflow_limit_m3,
                )

                retention_inflow_m3[node_retention_id] += diverted_volume_m3
                retention_outflow_m3[node_retention_id] += outflow_volume_m3
                node_sideflow_m3 = (
                    discharge_at_retention_basin_m3_per_dt
                    - upstream_inflow_m3_s * dt_f32
                )

            kinematic_discharge_m3_s, act_evap_rate = update_node_kinematic(
                Qin=upstream_inflow_m3_s,
                Qold=previous_discharge_m3_s[node],
                Qside=node_sideflow_m3 * inv_dt_f32,
                evaporation_m3_s=evaporation_m3[node] * inv_dt_f32,
                alpha=river_storage_alpha[node],
                beta=river_storage_beta[node],
                deltaT=dt_f32,
                deltaX=river_length[node],
            )

            kinematic_discharge_m3_s = max(kinematic_discharge_m3_s, np.float32(0.0))
            updated_discharge_m3_s[node] = kinematic_discharge_m3_s

            evap_vol = act_evap_rate * dt_f32
            actual_evaporation_m3[node] = evap_vol

            inflow_vol = upstream_inflow_m3_s * dt_f32 + node_sideflow_m3
            outflow_vol = kinematic_discharge_m3_s * dt_f32
            river_storage_m3[node] += np.float64(inflow_vol - outflow_vol - evap_vol)

            if river_storage_m3[node] < np.float64(0.0):
                over_abstraction_m3[node] += np.float32(-river_storage_m3[node])
                river_storage_m3[node] = np.float64(0.0)

        # -------------------------------------------------------------------------
        # Inertial wave routing (Slices n_kinematic : n_kinematic + n_inertial)
        # -------------------------------------------------------------------------
        if n_inertial > 0:
            inertial_end: int = n_kinematic + n_inertial

            for i in range(n_kinematic, inertial_end):
                node = sorted_idxs[i]
                orig_i = sorted_orig_i[i]
                for j in range(max_up_connections):
                    up_node = upstream_matrix_from_up_to_downstream[orig_i, j]
                    if up_node == -1:
                        break
                    if use_kinematic[up_node]:
                        kinematic_inflow_rate[node] += max(
                            updated_discharge_m3_s[up_node], np.float32(0.0)
                        )

            min_stable_dt_s: np.float32 = np.float32(1e9)

            for i in range(n_kinematic, inertial_end):
                node = sorted_idxs[i]

                river_area_m2: np.float32 = river_width[node] * river_length[node]
                water_depth_m: np.float32 = (
                    np.float32(river_storage_m3[node]) / river_area_m2
                )
                effective_depth_m: np.float32 = max(water_depth_m, H_MIN_WET)
                wave_celerity: np.float32 = np.sqrt(
                    GRAVITY_M_PER_S2 * effective_depth_m
                )

                depth_mask: np.float32 = np.float32(water_depth_m > H_MIN_WET)
                flow_vel: np.float32 = depth_mask * (
                    np.abs(previous_discharge_m3_s[node])
                    / (effective_depth_m * river_width[node])
                )

                max_dt: np.float32 = (CFL_SAFETY_FACTOR * river_length[node]) / (
                    wave_celerity + flow_vel + np.float32(1e-9)
                )
                if max_dt < min_stable_dt_s:
                    min_stable_dt_s = max_dt

            min_stable_dt_s = max(min_stable_dt_s, np.float32(1e-4))
            num_inertial_substeps: int = int(np.ceil(dt_f32 / min_stable_dt_s))
            num_inertial_substeps = max(1, min(num_inertial_substeps, 3600))

            num_substeps_f32: np.float32 = np.float32(num_inertial_substeps)
            inv_num_substeps_f32: np.float32 = np.float32(1.0) / num_substeps_f32
            num_substeps_f64: np.float64 = np.float64(num_inertial_substeps)
            substep_dt_s_f32: np.float32 = dt_f32 * inv_num_substeps_f32
            inv_substep_dt_s_f32: np.float32 = np.float32(1.0) / substep_dt_s_f32

            gravity_substep_dt_f32: np.float32 = GRAVITY_M_PER_S2 * substep_dt_s_f32

            sideflow_substep_m3: ArrayFloat64 = sideflow_m3 * np.float64(
                inv_num_substeps_f32
            )
            evap_substep_m3: ArrayFloat64 = evaporation_m3 * np.float64(
                inv_num_substeps_f32
            )
            outflow_wb_substep_m64: ArrayFloat64 = (
                outflow_per_waterbody_m3.astype(np.float64) / num_substeps_f64
            )

            retention_inflow_limit_sub: ArrayFloat64 = (
                np.float32(0.20) * retention_max_storage_m3
            ) * np.float64(inv_num_substeps_f32)
            retention_max_outflow_limit_sub: ArrayFloat64 = (
                np.float32(0.05) * retention_max_storage_m3
            ) * np.float64(inv_num_substeps_f32)

            fric_coeff_f32: ArrayFloat32 = gravity_substep_dt_f32 * (
                manning_n * manning_n
            )

            # Pre-compute spatial invariants outside the substep loop
            inv_cell_area_m2_arr: ArrayFloat32 = np.empty(n_cells, dtype=np.float32)
            inv_reach_length_m_arr: ArrayFloat32 = np.empty(n_cells, dtype=np.float32)

            for i in range(n_kinematic, inertial_end):
                node = sorted_idxs[i]
                w_m = river_width[node]
                l_m = river_length[node]
                cell_area = w_m * l_m
                inv_cell_area_m2_arr[node] = np.float32(1.0) / cell_area
                inv_reach_length_m_arr[node] = np.float32(1.0) / l_m

            substep_discharge_m3_s: ArrayFloat32 = previous_discharge_m3_s.copy()

            for substep in range(num_inertial_substeps):
                lateral_flux_m3.fill(np.float32(0.0))
                next_substep_discharge_m3_s.fill(np.float32(0.0))

                # Fused Pass 1 & 2: Lateral Fluxes + Momentum Solver
                for i in range(n_kinematic, inertial_end):
                    node = sorted_idxs[i]
                    orig_i = sorted_orig_i[i]

                    # --- Part 1: Lateral Fluxes & Retention Basin Routing ---
                    node_lateral_inflow_sub: np.float32 = sideflow_substep_m3[node]

                    for j in range(max_up_connections):
                        up_node = upstream_matrix_from_up_to_downstream[orig_i, j]
                        if up_node == -1:
                            break
                        if is_waterbody_outflow[up_node]:
                            wb_id = waterbody_id[up_node]
                            if wb_id != -1:
                                wb_outflow_sub: np.float32 = min(
                                    outflow_wb_substep_m64[wb_id],
                                    waterbody_storage_m3[wb_id],
                                )
                                waterbody_storage_m3[wb_id] -= wb_outflow_sub
                                node_lateral_inflow_sub += np.float32(wb_outflow_sub)

                    node_retention_id = retention_node_id[node]
                    if node_retention_id != -1:
                        avail_flow_rate = (
                            node_lateral_inflow_sub * inv_substep_dt_s_f32
                        ) + max(substep_discharge_m3_s[node], np.float32(0.0))
                        inflow_limit_sub = retention_inflow_limit_sub[node_retention_id]
                        max_outflow_limit_sub = retention_max_outflow_limit_sub[
                            node_retention_id
                        ]
                        river_volume_sub = avail_flow_rate * substep_dt_s_f32
                        is_rising_limb = avail_flow_rate > previous_discharge_m3_s[node]

                        (
                            diverted_vol,
                            outflow_vol,
                            retention_storage_m3[node_retention_id],
                            river_volume_sub,
                        ) = compute_retention_routing(
                            dt=substep_dt_s_f32,
                            river_volume_m3=river_volume_sub,
                            discharge_before_diversion_m3_s=avail_flow_rate,
                            is_rising_limb=is_rising_limb,
                            retention_storage_m3=retention_storage_m3[
                                node_retention_id
                            ],
                            retention_max_storage_m3=retention_max_storage_m3[
                                node_retention_id
                            ],
                            controlled_retention=controlled_retention[
                                node_retention_id
                            ],
                            activation_threshold_m3_s=retention_activation_threshold_m3_s[
                                node_retention_id
                            ],
                            release_threshold_factor=retention_basin_release_threshold_factor,
                            inflow_limit_m3=inflow_limit_sub,
                            max_outflow_limit_m3=max_outflow_limit_sub,
                        )

                        retention_inflow_m3[node_retention_id] += diverted_vol
                        retention_outflow_m3[node_retention_id] += outflow_vol
                        node_lateral_inflow_sub = (
                            river_volume_sub
                            - max(substep_discharge_m3_s[node], np.float32(0.0))
                            * substep_dt_s_f32
                        )

                    lateral_flux_m3[node] = node_lateral_inflow_sub

                    # --- Part 2: Momentum Solver & Stage Calculation ---
                    bed_elevation_node_m = bed_elevation[node]
                    river_width_m = river_width[node]
                    inv_reach_length_m = inv_reach_length_m_arr[node]
                    inv_cell_area_m2 = inv_cell_area_m2_arr[node]

                    flow_depth_m = max(
                        np.float32(river_storage_m3[node]) * inv_cell_area_m2,
                        np.float32(0.0),
                    )
                    water_stage_node_m = bed_elevation_node_m + flow_depth_m

                    ds = ds_node[node]

                    if ds != -1 and not is_pit[node]:
                        bed_elevation_ds_m = bed_elevation[ds]
                        if waterbody_id[ds] == -1 and not use_kinematic[ds]:
                            ds_area_m2 = river_width[ds] * river_length[ds]
                            flow_depth_ds_m = max(
                                np.float32(river_storage_m3[ds]) / ds_area_m2,
                                np.float32(0.0),
                            )
                            water_stage_ds_m = bed_elevation_ds_m + flow_depth_ds_m
                        else:
                            water_stage_ds_m = bed_elevation_ds_m
                    else:
                        bed_elevation_ds_m = bed_elevation_node_m
                        water_stage_ds_m = bed_elevation_node_m

                    max_stage = max(water_stage_node_m, water_stage_ds_m)
                    max_bed = max(bed_elevation_node_m, bed_elevation_ds_m)
                    effective_flow_depth_m = max(max_stage - max_bed, np.float32(0.0))

                    if effective_flow_depth_m < H_MIN_WET or river_storage_m3[
                        node
                    ] <= np.float64(0.0):
                        next_substep_discharge_m3_s[node] = np.float32(0.0)
                        continue

                    cross_sectional_area_m2 = effective_flow_depth_m * river_width_m
                    hydraulic_radius_m = max(effective_flow_depth_m, np.float32(1e-6))

                    slope_term: np.float32 = (
                        water_stage_ds_m - water_stage_node_m
                    ) * inv_reach_length_m

                    r_4_3: np.float32 = hydraulic_radius_m * np.cbrt(hydraulic_radius_m)
                    friction_denom_term: np.float32 = max(
                        r_4_3 * cross_sectional_area_m2,
                        np.float32(1e-6),
                    )

                    friction_denom: np.float32 = (
                        np.float32(1.0)
                        + (fric_coeff_f32[node] * np.abs(substep_discharge_m3_s[node]))
                        / friction_denom_term
                    )

                    calculated_discharge_m3_s: np.float32 = (
                        substep_discharge_m3_s[node]
                        - gravity_substep_dt_f32 * cross_sectional_area_m2 * slope_term
                    ) / friction_denom

                    boundary_mask: np.float32 = np.float32(
                        ds == -1
                        or is_pit[node]
                        or (ds != -1 and waterbody_id[ds] != -1)
                    )
                    calculated_discharge_m3_s = max(
                        calculated_discharge_m3_s, boundary_mask * np.float32(0.0)
                    )

                    discharge_critical: np.float32 = (
                        cross_sectional_area_m2
                        * SQRT_G
                        * np.sqrt(effective_flow_depth_m)
                    )
                    calculated_discharge_m3_s = min(
                        calculated_discharge_m3_s, discharge_critical
                    )
                    calculated_discharge_m3_s = max(
                        calculated_discharge_m3_s, -discharge_critical
                    )

                    if calculated_discharge_m3_s > np.float32(0.0):
                        max_q_avail = (
                            np.float32(river_storage_m3[node]) * inv_substep_dt_s_f32
                        )
                        calculated_discharge_m3_s = min(
                            calculated_discharge_m3_s, max_q_avail
                        )
                    elif calculated_discharge_m3_s < np.float32(0.0):
                        if ds != -1 and not is_pit[ds] and not use_kinematic[ds]:
                            max_q_avail = (
                                np.float32(river_storage_m3[ds]) * inv_substep_dt_s_f32
                            )
                            calculated_discharge_m3_s = -min(
                                abs(calculated_discharge_m3_s), max_q_avail
                            )
                        else:
                            calculated_discharge_m3_s = np.float32(0.0)

                    next_substep_discharge_m3_s[node] = calculated_discharge_m3_s

                # Pass 3: Mass Balance Storage Update
                for i in range(n_kinematic, inertial_end):
                    node = sorted_idxs[i]

                    river_storage_m3[node] += np.float64(
                        lateral_flux_m3[node]
                        + kinematic_inflow_rate[node] * substep_dt_s_f32
                    )

                    ds = ds_node[node]
                    vol_flux = next_substep_discharge_m3_s[node] * substep_dt_s_f32

                    if is_pit[node] or ds == -1:
                        river_storage_m3[node] -= np.float64(vol_flux)
                        pit_discharge_sum[node] += next_substep_discharge_m3_s[node]
                    else:
                        if waterbody_id[ds] != -1:
                            wb_ds_id = waterbody_id[ds]
                            wb_inflow = max(vol_flux, np.float32(0.0))
                            waterbody_storage_m3[wb_ds_id] += np.float64(wb_inflow)
                            waterbody_inflow_m3[wb_ds_id] += wb_inflow
                            river_storage_m3[node] -= np.float64(wb_inflow)
                        else:
                            river_storage_m3[node] -= np.float64(vol_flux)
                            if not use_kinematic[ds]:
                                river_storage_m3[ds] += np.float64(vol_flux)
                            else:
                                assumed_flux = (
                                    max(previous_discharge_m3_s[node], np.float32(0.0))
                                    * substep_dt_s_f32
                                )
                                river_storage_m3[ds] += np.float64(
                                    vol_flux - assumed_flux
                                )
                                if river_storage_m3[ds] < np.float64(0.0):
                                    over_abstraction_m3[ds] += np.float32(
                                        -river_storage_m3[ds]
                                    )
                                    river_storage_m3[ds] = np.float64(0.0)

                    if river_storage_m3[node] < np.float64(0.0):
                        over_abstraction_m3[node] += np.float32(-river_storage_m3[node])
                        river_storage_m3[node] = np.float64(0.0)

                    evap_sub = min(
                        evap_substep_m3[node],
                        np.float32(river_storage_m3[node]),
                    )
                    evap_sub = max(evap_sub, np.float32(0.0))
                    actual_evaporation_m3[node] += evap_sub
                    river_storage_m3[node] -= np.float64(evap_sub)

                    substep_discharge_m3_s[node] = next_substep_discharge_m3_s[node]

            for i in range(n_kinematic, inertial_end):
                node = sorted_idxs[i]
                if is_pit[node]:
                    updated_discharge_m3_s[node] = (
                        pit_discharge_sum[node] * inv_num_substeps_f32
                    )
                else:
                    updated_discharge_m3_s[node] = substep_discharge_m3_s[node]

        return (
            updated_discharge_m3_s,
            river_storage_m3,
            actual_evaporation_m3,
            over_abstraction_m3,
            waterbody_inflow_m3,
            retention_inflow_m3,
            retention_outflow_m3,
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
        river_width: ArrayFloat32,
        retention_activation_threshold_m3_s: ArrayFloat32,
        use_kinematic: ArrayBool,
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
        """Perform a single local inertial routing step.

        Args:
            Q_prev_m3_s: Previous discharge in m3/s for each cell.
            river_storage_m3: Current river storage in m3 for each cell.
            sideflow_m3: Lateral inflow in m3 for each cell.
            evaporation_m3: Evaporation volume in m3 for each cell.
            waterbody_storage_m3: Current waterbody storage in m3 for each waterbody.
            outflow_per_waterbody_m3: Outflow volume from each waterbody in m3.
            retention_storage_m3: Current retention basin storage in m3 for each retention basin.
            river_storage_alpha: Kinematic wave alpha parameter for each cell.
            river_storage_beta: Kinematic wave beta parameter for each cell.
            river_width: Width of the river reach in m for each cell.
            retention_activation_threshold_m3_s: Activation threshold for retention basins in m3/s.
            use_kinematic: Boolean array indicating whether to use kinematic wave routing for each cell.

        Returns:
            A tuple containing:
                - Q: Updated discharge in m3/s for each cell.
                - river_storage_m3: Updated river storage in m3 for each cell.
                - actual_evaporation_m3: Actual evaporation volume in m3 for each cell.
                - over_abstraction_m3: Over-abstraction volume in m3 for each cell.
                - waterbody_storage_m3: Updated waterbody storage in m3 for each waterbody.
                - waterbody_inflow_m3: Inflow volume to each waterbody in m3.
                - outflow_at_pits_m3: Total outflow volume at pit cells in m3.
                - retention_storage_m3: Updated retention basin storage in m3 for each retention basin.
                - retention_inflow_m3: Inflow volume to each retention basin in m3.
                - retention_outflow_m3: Outflow volume from each retention basin in m3.
        """
        (
            Q,
            river_storage_m3,
            actual_evaporation_m3,
            over_abstraction_m3,
            waterbody_inflow_m3,
            retention_inflow_m3,
            retention_outflow_m3,
        ) = self._step(
            routing_timestep_s=self.dt,
            previous_discharge_m3_s=Q_prev_m3_s,
            river_storage_m3=river_storage_m3,
            sideflow_m3=sideflow_m3,
            evaporation_m3=evaporation_m3,
            waterbody_storage_m3=waterbody_storage_m3,
            outflow_per_waterbody_m3=outflow_per_waterbody_m3,
            upstream_matrix_from_up_to_downstream=self.upstream_matrix_from_up_to_downstream,
            idxs_up_to_downstream=self.idxs_up_to_downstream,
            is_waterbody_outflow=self.is_waterbody_outflow,
            waterbody_id=self.waterbody_id,
            river_storage_alpha=river_storage_alpha,
            river_storage_beta=river_storage_beta,
            river_length=self.river_length,
            river_width=river_width,
            bed_elevation=self.bed_elevation,
            manning_n=self.manning_n,
            retention_storage_m3=retention_storage_m3,
            retention_max_storage_m3=self.retention_max_storage_m3,
            retention_node_id=self.retention_node_id,
            controlled_retention=self.controlled_retention,
            retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
            retention_basin_release_threshold_factor=self.retention_basin_release_threshold_factor,
            ds_node=self.ds_node,
            is_pit=self.is_pit,
            use_kinematic=use_kinematic,
        )

        outflow_at_pits_m3 = np.nansum(Q[self.is_pit] * self.dt)

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
