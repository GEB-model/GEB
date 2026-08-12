"""Local inertial routing algorithm for river networks."""

import numpy as np
import pyflwdir

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
    Node attributes and connectivity matrices are pre-ordered for contiguous memory layout
    internally while maintaining standard grid order for public APIs and attributes.
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
    ) -> None:
        """Initializes the LocalInertial class."""
        super().__init__(
            dt,
            river_network,
            waterbody_ids,
            is_waterbody_outflow,
            retention_basin_release_threshold_factor,
        )

        # Assert river_length is strictly positive
        assert np.all(river_length > 0), "River length must be strictly greater than 0."

        # Public attributes kept in standard original grid order
        self.river_length = np.maximum(river_length, np.float32(1.0))
        self.river_width = river_width.astype(np.float32)
        self.river_ids = river_ids
        self.bed_elevation = bankfull_river_elevation_m.astype(np.float32)
        self.manning_n = manning_n.astype(np.float32)
        self.use_kinematic = use_kinematic
        self.retention_node_id = retention_node_id
        self.retention_max_storage_m3 = retention_max_storage_m3
        self.controlled_retention = controlled_retention

        # -------------------------------------------------------------------------
        # 1. Un-permuted Downstream Connectivity
        # -------------------------------------------------------------------------
        mapper = np.full(river_network.size + 1, -1, dtype=np.int32)
        indices = np.arange(river_network.size, dtype=np.int32)[river_network.mask]
        mapper[indices] = np.arange(indices.size, dtype=np.int32)

        unmasked_ds = river_network.idxs_ds[indices]
        ds_node_orig = mapper[unmasked_ds]
        ds_node_orig[self.is_pit] = -1

        # Assert bed_elevation strictly decreases downstream
        has_ds = ds_node_orig != -1
        assert np.all(
            self.bed_elevation[has_ds] >= self.bed_elevation[ds_node_orig[has_ds]]
        ), "Bed elevation must strictly decrease downstream along the river network."

        # -------------------------------------------------------------------------
        # 2. Validate Kinematic/Inertial Topology
        # -------------------------------------------------------------------------
        for node_compressed, node in enumerate(indices):
            if (
                use_kinematic[node_compressed]
                or waterbody_ids[node_compressed] != -1
                or self.is_pit[node_compressed]
            ):
                continue

            assert not use_kinematic[node_compressed]

            ds_node = river_network.idxs_ds[node]
            ds_node_compressed = mapper[ds_node]
            assert ds_node_compressed != -1, (
                f"Downstream node for {node} is not in the river network."
            )

            if (
                waterbody_ids[ds_node_compressed] == -1
                and use_kinematic[ds_node_compressed]
                and not use_kinematic[node_compressed]
            ):
                raise ValueError(
                    "Invalid kinematic/inertial configuration: a kinematic river "
                    f"node ({ds_node} with river_id {river_ids[ds_node_compressed]}) is downstream of inertial node ({node} with river_id {river_ids[node_compressed]}). "
                    "Once a river becomes inertial, all downstream river nodes "
                    "must also be inertial."
                )

        # -------------------------------------------------------------------------
        # 3. Partition and Compute Permutation Map & Row Mapping
        # -------------------------------------------------------------------------
        n_up_nodes = self.upstream_matrix_from_up_to_downstream.shape[0]
        sorted_idxs = np.empty(n_up_nodes, dtype=np.int32)
        sorted_orig_i = np.empty(n_up_nodes, dtype=np.int32)

        n_kinematic = 0
        n_inertial = 0

        # Pass 1: Kinematic nodes [0 : n_kinematic]
        for i in range(n_up_nodes):
            node = self.idxs_up_to_downstream[i]
            if use_kinematic[node] and waterbody_ids[node] == -1:
                sorted_idxs[n_kinematic] = node
                sorted_orig_i[n_kinematic] = i
                n_kinematic += 1

        # Pass 2: Inertial nodes [n_kinematic : n_kinematic + n_inertial]
        for i in range(n_up_nodes):
            node = self.idxs_up_to_downstream[i]
            if not use_kinematic[node] and waterbody_ids[node] == -1:
                idx = n_kinematic + n_inertial
                sorted_idxs[idx] = node
                sorted_orig_i[idx] = i
                n_inertial += 1

        # Pass 3: Waterbody nodes [n_kinematic + n_inertial : n_up_nodes]
        n_other_start = n_kinematic + n_inertial
        n_other = 0
        for i in range(n_up_nodes):
            node = self.idxs_up_to_downstream[i]
            if waterbody_ids[node] != -1:
                idx = n_other_start + n_other
                sorted_idxs[idx] = node
                sorted_orig_i[idx] = i
                n_other += 1

        # Inverse index mapping (maps original cell ID -> permuted contiguous index)
        inv_idxs = np.full(n_up_nodes, -1, dtype=np.int32)
        inv_idxs[sorted_idxs] = np.arange(n_up_nodes, dtype=np.int32)

        self.sorted_idxs = sorted_idxs
        self.inv_idxs = inv_idxs
        self.n_kinematic = n_kinematic
        self.n_inertial = n_inertial

        # -------------------------------------------------------------------------
        # 4. Internal Permuted Arrays & Static Invariants (Used in _step)
        # -------------------------------------------------------------------------
        self._river_length_perm = self.river_length[sorted_idxs]
        self._river_width_perm = self.river_width[sorted_idxs]
        self._bed_elevation_perm = self.bed_elevation[sorted_idxs]
        self._manning_n_perm = self.manning_n[sorted_idxs]
        self._use_kinematic_perm = self.use_kinematic[sorted_idxs]
        self._is_pit_perm = self.is_pit[sorted_idxs]
        self._waterbody_ids_perm = self.waterbody_ids[sorted_idxs]
        self._river_ids_perm = self.river_ids[sorted_idxs]
        self._is_waterbody_outflow_perm = self.is_waterbody_outflow[sorted_idxs]
        self._retention_node_id_perm = self.retention_node_id[sorted_idxs]

        # Static performance invariants
        self._inv_reach_length_perm = np.float32(1.0) / self._river_length_perm
        self._cell_area_perm = self._river_width_perm * self._river_length_perm
        self._inv_cell_area_perm = np.float32(1.0) / self._cell_area_perm
        self._manning_n_sq_perm = self._manning_n_perm * self._manning_n_perm

        # Remap downstream node array into contiguous coordinates
        ds_perm = ds_node_orig[sorted_idxs]
        self._ds_node_perm = np.where(
            ds_perm != -1, inv_idxs[np.maximum(ds_perm, 0)], -1
        )

        # Remap upstream matrix using topological row indices
        up_perm = self.upstream_matrix_from_up_to_downstream[sorted_orig_i, :]
        self._upstream_matrix_perm = np.where(
            up_perm != -1, inv_idxs[np.maximum(up_perm, 0)], -1
        )

        # -------------------------------------------------------------------------
        # 5. Pre-allocated Reusable Workspaces (Avoids GC overhead per timestep)
        # -------------------------------------------------------------------------
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
        self._pit_discharge_sum_perm = np.empty(n_up_nodes, dtype=np.float32)

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
        """Calculate river storage from discharge using kinematic wave relation."""
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
        """Calculate discharge from river storage using kinematic wave relation."""
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
        """Get the available storage of the river network."""
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
        """Get total storage using standard grid-ordered attributes."""
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
    # @njit(cache=True)
    def _step(
        routing_timestep_s: float | int,
        previous_discharge_m3_s: ArrayFloat32,
        river_storage_m3: ArrayFloat64,
        sideflow_m3: ArrayFloat32,
        evaporation_m3: ArrayFloat32,
        waterbody_storage_m3: ArrayFloat64,
        outflow_per_waterbody_m3: ArrayFloat32,
        upstream_matrix_from_up_to_downstream: TwoDArrayInt32,
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
        pit_discharge_sum: ArrayFloat32,
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
        max_up_connections: int = upstream_matrix_from_up_to_downstream.shape[1]

        MAX_ALLOWED_DEPTH_M: np.float32 = np.float32(
            100.0
        )  # Extreme depth limit relative to cell area

        # Input Sanity Checks
        n_active = n_kinematic + n_inertial
        assert np.all(np.isfinite(river_storage_m3[:n_active])), (
            "Non-finite river storage detected on entry to _step."
        )
        assert np.all(river_storage_m3[:n_active] >= 0.0), (
            "Negative river storage detected on entry to _step."
        )
        assert np.all(np.isfinite(previous_discharge_m3_s[:n_active])), (
            "Non-finite previous discharge detected on entry to _step."
        )
        assert np.all(np.isfinite(sideflow_m3[:n_active])), (
            "Non-finite sideflow detected on entry to _step."
        )

        # In-place reset of pre-allocated buffers
        over_abstraction_m3.fill(np.float32(0.0))
        actual_evaporation_m3.fill(np.float32(0.0))
        waterbody_inflow_m3.fill(np.float32(0.0))
        retention_inflow_m3.fill(np.float32(0.0))
        retention_outflow_m3.fill(np.float32(0.0))
        updated_discharge_m3_s.fill(np.float32(0.0))
        lateral_flux_m3.fill(np.float32(0.0))
        next_substep_discharge_m3_s.fill(np.float32(0.0))
        kinematic_inflow_rate.fill(np.float32(0.0))
        pit_discharge_sum.fill(np.float32(0.0))

        # Immediate NaN assignment for waterbodies
        n_other_start = n_kinematic + n_inertial
        for i in range(n_other_start, n_cells):
            updated_discharge_m3_s[i] = np.float32(np.nan)

        H_MIN_WET: np.float32 = np.float32(1e-3)
        GRAVITY_M_PER_S2: np.float32 = np.float32(9.80665)
        SQRT_G: np.float32 = np.float32(3.1315574)
        CFL_SAFETY_FACTOR: np.float32 = np.float32(0.7)
        dt_f32: np.float32 = np.float32(routing_timestep_s)
        inv_dt_f32: np.float32 = np.float32(1.0) / dt_f32

        # -------------------------------------------------------------------------
        # Kinematic wave routing (Contiguous range 0 : n_kinematic)
        # -------------------------------------------------------------------------
        for i in range(n_kinematic):
            upstream_inflow_m3_s: np.float32 = np.float32(0.0)
            node_sideflow_m3: np.float32 = sideflow_m3[i]

            for j in range(max_up_connections):
                up_node = upstream_matrix_from_up_to_downstream[i, j]
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
                    upstream_inflow_m3_s += max(
                        updated_discharge_m3_s[up_node], np.float32(0.0)
                    )

            node_retention_id = retention_node_id[i]
            if node_retention_id != -1:
                discharge_before_diversion_m3_s = (
                    (upstream_inflow_m3_s + previous_discharge_m3_s[i])
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
                    discharge_before_diversion_m3_s > previous_discharge_m3_s[i]
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
                Qold=previous_discharge_m3_s[i],
                Qside=node_sideflow_m3 * inv_dt_f32,
                evaporation_m3_s=evaporation_m3[i] * inv_dt_f32,
                alpha=river_storage_alpha[i],
                beta=river_storage_beta[i],
                deltaT=dt_f32,
                deltaX=river_length[i],
            )

            kinematic_discharge_m3_s = max(kinematic_discharge_m3_s, np.float32(0.0))
            updated_discharge_m3_s[i] = kinematic_discharge_m3_s

            evap_vol = act_evap_rate * dt_f32
            actual_evaporation_m3[i] = evap_vol

            inflow_vol = upstream_inflow_m3_s * dt_f32 + node_sideflow_m3
            outflow_vol = kinematic_discharge_m3_s * dt_f32
            river_storage_m3[i] += np.float64(inflow_vol - outflow_vol - evap_vol)

            if river_storage_m3[i] < np.float64(0.0):
                over_abstraction_m3[i] += np.float32(-river_storage_m3[i])
                river_storage_m3[i] = np.float64(0.0)

            assert np.isfinite(updated_discharge_m3_s[i]), (
                f"Kinematic discharge non-finite at node {i}."
            )

        # -------------------------------------------------------------------------
        # Inertial wave routing (Contiguous range n_kinematic : n_kinematic + n_inertial)
        # -------------------------------------------------------------------------
        if n_inertial > 0:
            inertial_end: int = n_kinematic + n_inertial

            for i in range(n_kinematic, inertial_end):
                for j in range(max_up_connections):
                    up_node = upstream_matrix_from_up_to_downstream[i, j]
                    if up_node == -1:
                        break
                    if is_waterbody_outflow[up_node]:
                        continue
                    if use_kinematic[up_node] and waterbody_id[up_node] == -1:
                        kinematic_inflow_rate[i] += max(
                            updated_discharge_m3_s[up_node], np.float32(0.0)
                        )

            min_stable_dt_s: np.float32 = np.float32(1e9)

            for i in range(n_kinematic, inertial_end):
                water_depth_m: np.float32 = (
                    np.float32(river_storage_m3[i]) * inv_cell_area[i]
                )
                effective_depth_m: np.float32 = max(water_depth_m, H_MIN_WET)
                wave_celerity: np.float32 = np.sqrt(
                    GRAVITY_M_PER_S2 * effective_depth_m
                )

                depth_mask: np.float32 = np.float32(water_depth_m > H_MIN_WET)
                flow_vel: np.float32 = depth_mask * (
                    np.abs(previous_discharge_m3_s[i])
                    / (effective_depth_m * river_width[i])
                )

                max_dt: np.float32 = (CFL_SAFETY_FACTOR * river_length[i]) / (
                    wave_celerity + flow_vel + np.float32(1e-9)
                )
                if max_dt < min_stable_dt_s:
                    min_stable_dt_s = max_dt

            min_stable_dt_s = max(min_stable_dt_s, np.float32(1e-4))
            raw_substeps = int(np.ceil(dt_f32 / min_stable_dt_s))

            assert raw_substeps < 3600, (
                f"Inertial stability threshold exceeded: required substeps ({raw_substeps}) "
                f"exceed max allowance (3600). Minimum stable dt = {min_stable_dt_s}s."
            )

            num_inertial_substeps: int = max(1, min(raw_substeps, 3600))

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

            fric_coeff_f32: ArrayFloat32 = gravity_substep_dt_f32 * manning_n_sq
            substep_discharge_m3_s: ArrayFloat32 = previous_discharge_m3_s.copy()

            for substep in range(num_inertial_substeps):
                lateral_flux_m3.fill(np.float32(0.0))
                next_substep_discharge_m3_s.fill(np.float32(0.0))

                # Fused Pass 1 & 2: Lateral Fluxes + Momentum Solver
                for i in range(n_kinematic, inertial_end):
                    node_lateral_inflow_sub: np.float32 = sideflow_substep_m3[i]

                    for j in range(max_up_connections):
                        up_node = upstream_matrix_from_up_to_downstream[i, j]
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

                    node_retention_id = retention_node_id[i]
                    if node_retention_id != -1:
                        avail_flow_rate = (
                            node_lateral_inflow_sub * inv_substep_dt_s_f32
                        ) + max(substep_discharge_m3_s[i], np.float32(0.0))
                        inflow_limit_sub = retention_inflow_limit_sub[node_retention_id]
                        max_outflow_limit_sub = retention_max_outflow_limit_sub[
                            node_retention_id
                        ]
                        river_volume_sub = avail_flow_rate * substep_dt_s_f32
                        is_rising_limb = avail_flow_rate > previous_discharge_m3_s[i]

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
                            - max(substep_discharge_m3_s[i], np.float32(0.0))
                            * substep_dt_s_f32
                        )

                    lateral_flux_m3[i] = node_lateral_inflow_sub

                    # Momentum Solver & Stage Calculation
                    bed_elevation_node_m = bed_elevation[i]
                    river_width_m = river_width[i]
                    inv_reach_length_m = inv_reach_length[i]
                    inv_cell_area_m2 = inv_cell_area[i]

                    flow_depth_m = max(
                        np.float32(river_storage_m3[i]) * inv_cell_area_m2,
                        np.float32(0.0),
                    )

                    assert np.isfinite(flow_depth_m), (
                        f"Non-finite flow depth at node {i} in substep {substep}."
                    )

                    water_stage_node_m = bed_elevation_node_m + flow_depth_m

                    ds = ds_node[i]

                    if ds != -1 and not is_pit[i]:
                        bed_elevation_ds_m = bed_elevation[ds]
                        if waterbody_id[ds] == -1 and not use_kinematic[ds]:
                            flow_depth_ds_m = max(
                                np.float32(river_storage_m3[ds]) * inv_cell_area[ds],
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
                        i
                    ] <= np.float64(0.0):
                        next_substep_discharge_m3_s[i] = np.float32(0.0)
                        continue

                    cross_sectional_area_m2 = effective_flow_depth_m * river_width_m

                    assert np.isfinite(cross_sectional_area_m2), (
                        f"Non-finite cross sectional area at node {i}."
                    )
                    assert cross_sectional_area_m2 < 1e6, (
                        f"Cross-sectional area exploded ({cross_sectional_area_m2:.2f} m2) "
                        f"at node {i} (ds={ds}) during substep {substep}/{num_inertial_substeps}."
                    )

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
                        + (fric_coeff_f32[i] * np.abs(substep_discharge_m3_s[i]))
                        / friction_denom_term
                    )

                    calculated_discharge_m3_s: np.float32 = (
                        substep_discharge_m3_s[i]
                        - gravity_substep_dt_f32 * cross_sectional_area_m2 * slope_term
                    ) / friction_denom

                    boundary_mask: np.float32 = np.float32(
                        ds == -1 or is_pit[i] or (ds != -1 and waterbody_id[ds] != -1)
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
                            np.float32(river_storage_m3[i]) * inv_substep_dt_s_f32
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

                    assert np.isfinite(calculated_discharge_m3_s), (
                        f"Non-finite calculated discharge at node {i} in substep {substep}."
                    )

                    next_substep_discharge_m3_s[i] = calculated_discharge_m3_s

                # Pass 3: Mass Balance Storage Update & Stability Validation
                for i in range(n_kinematic, inertial_end):
                    sum_upstream_inertial_q = np.float32(0.0)
                    for j in range(max_up_connections):
                        up_node = upstream_matrix_from_up_to_downstream[i, j]
                        if up_node == -1:
                            break
                        if waterbody_id[up_node] == -1 and not use_kinematic[up_node]:
                            sum_upstream_inertial_q += substep_discharge_m3_s[up_node]

                    vol_in = (
                        sum_upstream_inertial_q + kinematic_inflow_rate[i]
                    ) * substep_dt_s_f32 + lateral_flux_m3[i]

                    river_storage_m3[i] += np.float64(vol_in)

                    vol_out = next_substep_discharge_m3_s[i] * substep_dt_s_f32
                    ds = ds_node[i]

                    if is_pit[i] or ds == -1:
                        river_storage_m3[i] -= np.float64(vol_out)
                        pit_discharge_sum[i] += next_substep_discharge_m3_s[i]
                    elif waterbody_id[ds] != -1:
                        wb_ds_id = waterbody_id[ds]
                        wb_inflow = max(vol_out, 0.0)
                        waterbody_storage_m3[wb_ds_id] += np.float64(wb_inflow)
                        waterbody_inflow_m3[wb_ds_id] += wb_inflow
                        river_storage_m3[i] -= np.float64(wb_inflow)
                    else:
                        river_storage_m3[i] -= np.float64(vol_out)
                        river_storage_m3[ds] += np.float64(vol_out)

                    if river_storage_m3[i] < np.float64(0.0):
                        over_abstraction_m3[i] += np.float32(-river_storage_m3[i])
                        river_storage_m3[i] = np.float64(0.0)

                    evap_sub = np.float32(min(evap_substep_m3[i], river_storage_m3[i]))
                    evap_sub = max(evap_sub, np.float32(0.0))
                    actual_evaporation_m3[i] += evap_sub
                    river_storage_m3[i] -= np.float64(evap_sub)

                    # --- EARLY INERTIAL EXTREME STORAGE ASSERTIONS ---
                    eq_depth_inertial_i = (
                        np.float32(river_storage_m3[i]) * inv_cell_area[i]
                    )
                    assert eq_depth_inertial_i < MAX_ALLOWED_DEPTH_M, (
                        f"Extreme storage in Inertial routing at node {i} during substep {substep}/{num_inertial_substeps}! "
                        f"Storage = {river_storage_m3[i]:.3e} m3, Equivalent Depth = {eq_depth_inertial_i:.2f} m "
                        f"(Width = {river_width[i]} m, Length = {river_length[i]} m)."
                    )

                    if (
                        ds != -1
                        and waterbody_id[ds] == -1
                        and not use_kinematic[ds]
                        and not is_pit[i]
                    ):
                        eq_depth_inertial_ds = (
                            np.float32(river_storage_m3[ds]) * inv_cell_area[ds]
                        )
                        assert eq_depth_inertial_ds < MAX_ALLOWED_DEPTH_M, (
                            f"Extreme storage in Inertial routing at downstream target node {ds} (transferred from node {i}) "
                            f"during substep {substep}/{num_inertial_substeps}! Storage = {river_storage_m3[ds]:.3e} m3, "
                            f"Equivalent Depth = {eq_depth_inertial_ds:.2f} m."
                        )

                    assert np.isfinite(river_storage_m3[i]), (
                        f"Non-finite storage at node {i} after mass balance in substep {substep}."
                    )

                    # --- NEW CFL CRITICAL STABILITY CHECK FOR UPDATED FLOW STATE ---
                    water_depth_m_new = eq_depth_inertial_i
                    eff_depth_new = max(water_depth_m_new, H_MIN_WET)
                    celerity_new = np.sqrt(GRAVITY_M_PER_S2 * eff_depth_new)
                    depth_mask_new = np.float32(water_depth_m_new > H_MIN_WET)
                    flow_vel_new = depth_mask_new * (
                        np.abs(next_substep_discharge_m3_s[i])
                        / (eff_depth_new * river_width[i])
                    )
                    max_dt_new = (CFL_SAFETY_FACTOR * river_length[i]) / (
                        celerity_new + flow_vel_new + np.float32(1e-9)
                    )

                    assert substep_dt_s_f32 <= max_dt_new, (
                        f"CFL stability limit violated at node {i} during substep {substep}/{num_inertial_substeps}! "
                        f"Substep dt ({substep_dt_s_f32:.4f} s) > Max stable dt ({max_dt_new:.4f} s). "
                        f"New Q = {next_substep_discharge_m3_s[i]:.2f} m3/s, Depth = {eff_depth_new:.2f} m, "
                        f"Celerity = {celerity_new:.2f} m/s, Vel = {flow_vel_new:.2f} m/s."
                    )

                    substep_discharge_m3_s[i] = next_substep_discharge_m3_s[i]
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
        """Perform a single local inertial routing step."""
        # Populate pre-allocated permutation workspace buffers in-place
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
        ) = self._step(
            routing_timestep_s=self.dt,
            previous_discharge_m3_s=self._Q_prev_perm,
            river_storage_m3=self._river_storage_perm,
            sideflow_m3=self._sideflow_perm,
            evaporation_m3=self._evaporation_perm,
            waterbody_storage_m3=waterbody_storage_m3,
            outflow_per_waterbody_m3=outflow_per_waterbody_m3,
            upstream_matrix_from_up_to_downstream=self._upstream_matrix_perm,
            is_waterbody_outflow=self._is_waterbody_outflow_perm,
            waterbody_id=self._waterbody_ids_perm,
            river_storage_alpha=self._alpha_perm,
            river_storage_beta=self._beta_perm,
            river_length=self._river_length_perm,
            inv_reach_length=self._inv_reach_length_perm,
            inv_cell_area=self._inv_cell_area_perm,
            river_width=self._river_width_perm,
            bed_elevation=self._bed_elevation_perm,
            manning_n_sq=self._manning_n_sq_perm,
            retention_storage_m3=retention_storage_m3,
            retention_max_storage_m3=self.retention_max_storage_m3,
            retention_node_id=self._retention_node_id_perm,
            controlled_retention=self.controlled_retention,
            retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
            retention_basin_release_threshold_factor=self.retention_basin_release_threshold_factor,
            ds_node=self._ds_node_perm,
            is_pit=self._is_pit_perm,
            use_kinematic=self._use_kinematic_perm,
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
            pit_discharge_sum=self._pit_discharge_sum_perm,
        )

        outflow_at_pits_m3 = np.nansum(Q_perm[self._is_pit_perm] * self.dt)

        # Restructure outputs back to original grid node layout
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
