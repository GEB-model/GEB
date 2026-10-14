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
    TwoDArrayInt32,
)

from .geometry import (
    InertialGeometry,
    build_inertial_geometry,
)
from .inertial_kernel import (
    INERTIAL_PARALLEL_THRESHOLD,
    RoutingStepDiagnostics,
    SubstepBuffers,
    _compute_inertial_substeps_cfl_parallel,
    _compute_inertial_substeps_cfl_serial,
    _compute_stage_from_storage_inertial,
    _gather_inputs,
    _run_inertial_substeps_parallel,
    _run_inertial_substeps_serial,
    _scatter_outputs,
)
from .kinematic import (
    KINEMATIC_PARALLEL_THRESHOLD,
    _run_kinematic_step_parallel,
    _run_kinematic_step_serial,
    _transfer_waterbody_outflows,
)
from .topology import (
    InertialTopology,
    KinematicTopology,
    RetentionCoupling,
    RetentionLinks,
    RoutingTopology,
    WaterbodyCoupling,
    WaterbodyLinks,
    build_routing_topology,
)

__all__: list[str] = [
    "LocalInertial",
]


@njit(fastmath=True, parallel=True)
def _scatter_water_stage(
    sorted_idxs: ArrayInt32,
    n_kinematic: int,
    n_inertial: int,
    n_wb: int,
    waterbody_ids: ArrayInt32,
    bed_elevation: ArrayFloat32,
    stage_buf: ArrayFloat32,
    stage_out: ArrayFloat32,
) -> None:
    """Scatters simulated reach and waterbody stages back to grid ordering without allocations.

    Args:
        sorted_idxs: Precomputed 1D permutation index array mapping topological to grid cells.
        n_kinematic: Number of kinematic wave cells.
        n_inertial: Number of local inertial momentum cells.
        n_wb: Number of waterbody objects.
        waterbody_ids: Array mapping grid cells to waterbody IDs (-1 if absent).
        bed_elevation: Channel bed elevation array in grid order (meters).
        stage_buf: Permuted stage buffer containing inertial and waterbody stages (meters).
        stage_out: Pre-allocated output stage array in grid order (meters).
    """
    n_cells: int = len(sorted_idxs)
    inertial_start: int = n_kinematic
    inertial_end: int = n_kinematic + n_inertial

    for i in prange(n_cells):
        node: int = sorted_idxs[i]
        if i < inertial_start:
            stage_out[node] = bed_elevation[node]
        elif i < inertial_end:
            stage_out[node] = stage_buf[i - inertial_start]
        else:
            wb_id: int = waterbody_ids[node]
            if 0 <= wb_id < n_wb:
                stage_out[node] = stage_buf[n_inertial + wb_id]
            else:
                stage_out[node] = bed_elevation[node]


@njit(fastmath=True)
def _sum_indexed_discharge(discharge: ArrayFloat32, indices: ArrayInt32) -> np.float32:
    """Sums discharge values at specific indices in-place without array slicing allocations.

    Args:
        discharge: 1D discharge array (m³/s).
        indices: 1D integer index array.

    Returns:
        Summed discharge at specified indices (m³/s).
    """
    total: np.float32 = np.float32(0.0)
    for i in range(indices.shape[0]):
        total += discharge[indices[i]]
    return total


class LocalInertial:
    """Local inertial river network router with CaMa-Flood style dead-storage floodplain geometry.

    Combines non-linear kinematic wave routing for overland / non-river cells with a
    1D local inertial dynamic routing formulation for river network reaches,
    supporting power-law (parabolic) channels and CaMa-Flood style dead-storage floodplains
    (Yamazaki et al. 2011; Bates et al. 2010).
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
            ValueError: If input configurations or array dimensions are invalid.
        """
        if dt <= 0:
            raise ValueError("dt must be greater than 0")
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

        if is_waterbody_outflow is None:
            raise ValueError(
                "is_waterbody_outflow must be provided if waterbody_id is provided"
            )
        if waterbody_ids.shape != self.idxs_up_to_downstream.shape:
            raise ValueError("waterbody_ids shape must match network size")
        self.waterbody_ids: ArrayInt32 = waterbody_ids

        if is_waterbody_outflow.shape != self.idxs_up_to_downstream.shape:
            raise ValueError("is_waterbody_outflow shape must match network size")
        if not (
            np.bincount(
                self.waterbody_ids[self.waterbody_ids != -1],
                weights=is_waterbody_outflow[self.waterbody_ids != -1],
            )
            == 1
        ).all():
            raise ValueError(
                "Each active waterbody must have exactly one outflow point"
            )

        self.river_length = river_length
        self.river_width = river_width
        self.bed_elevation = bankfull_river_elevation_m
        self.retention_max_storage_m3 = retention_max_storage_m3
        self.controlled_retention = controlled_retention
        self.bankfull_depth = bankfull_depth_m

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

        # Categorize all network nodes in a single topological pass:
        # kinematic headwaters, connected kinematic reaches, inertial reaches, and waterbodies.
        kin_headwaters: list[int] = []
        kin_connected: list[int] = []
        inertial_nodes_orig: list[int] = []
        waterbody_nodes: list[int] = []

        for i in range(n_up_nodes):
            node: int = int(self.idxs_up_to_downstream[i])
            if waterbody_ids[node] != -1:
                waterbody_nodes.append(node)
            elif use_kinematic[node]:
                if self.upstream_matrix_from_up_to_downstream[i, 0] == -1:
                    kin_headwaters.append(node)
                else:
                    kin_connected.append(node)
            else:
                inertial_nodes_orig.append(node)

        n_kin_headwater: int = len(kin_headwaters)
        n_kinematic: int = n_kin_headwater + len(kin_connected)
        sorted_idxs[:n_kin_headwater] = kin_headwaters
        sorted_idxs[n_kin_headwater:n_kinematic] = kin_connected

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

        n_inertial: int = len(chained_inertial_nodes)
        sorted_idxs[n_kinematic : n_kinematic + n_inertial] = chained_inertial_nodes
        sorted_idxs[n_kinematic + n_inertial :] = waterbody_nodes

        node_to_orig_i: ArrayInt32 = np.empty(n_up_nodes, dtype=np.int32)
        node_to_orig_i[self.idxs_up_to_downstream] = np.arange(
            n_up_nodes, dtype=np.int32
        )
        sorted_orig_i: ArrayInt32 = node_to_orig_i[sorted_idxs]

        inv_idxs: ArrayInt32 = np.full(n_up_nodes, -1, dtype=np.int32)
        inv_idxs[sorted_idxs] = np.arange(n_up_nodes, dtype=np.int32)

        self.sorted_idxs: ArrayInt32 = sorted_idxs
        self.inv_idxs: ArrayInt32 = inv_idxs
        self.n_kin_headwater: int = n_kin_headwater
        self.n_kinematic: int = n_kinematic
        self.n_inertial: int = n_inertial

        # Sliced topological parameters
        river_length_perm: ArrayFloat32 = self.river_length[sorted_idxs]
        self._river_width: ArrayFloat32 = self.river_width[sorted_idxs]
        self._bankfull_depth: ArrayFloat32 = self.bankfull_depth[sorted_idxs]
        bed_elevation_perm: ArrayFloat32 = self.bed_elevation[sorted_idxs]
        use_kinematic_perm: ArrayBool = use_kinematic[sorted_idxs]
        is_pit_perm: ArrayBool = is_pit[sorted_idxs]
        self._is_ocean_pit: ArrayBool = is_ocean_pit_orig[sorted_idxs]
        self._pit_slope: ArrayFloat32 = pit_slope_orig[sorted_idxs]
        waterbody_ids_perm: ArrayInt32 = self.waterbody_ids[sorted_idxs]
        self._river_ids: ArrayInt32 = river_ids[sorted_idxs]
        is_waterbody_outflow_perm: ArrayBool = is_waterbody_outflow[sorted_idxs]
        retention_node_id_perm: ArrayInt32 = retention_node_id[sorted_idxs]

        inertial_start: int = self.n_kinematic
        inertial_end: int = self.n_kinematic + self.n_inertial
        self._shape_exponent_inertial: ArrayFloat32 = shape_exponent[
            sorted_idxs[inertial_start:inertial_end]
        ].astype(np.float32)
        self._floodplain_width_inertial: ArrayFloat32 = floodplain_width_m[
            sorted_idxs[inertial_start:inertial_end]
        ].astype(np.float32)
        manning_n_inertial: ArrayFloat32 = manning_n[
            sorted_idxs[inertial_start:inertial_end]
        ].astype(np.float32)
        self._manning_n_sq_inertial: ArrayFloat32 = (
            manning_n_inertial * manning_n_inertial
        )

        ds_perm: ArrayInt32 = ds_node_orig[sorted_idxs]
        ds_node: ArrayInt32 = np.where(
            ds_perm != -1, inv_idxs[np.maximum(ds_perm, 0)], -1
        )

        up_perm: TwoDArrayInt32 = self.upstream_matrix_from_up_to_downstream[
            sorted_orig_i, :
        ]
        upstream_matrix: TwoDArrayInt32 = np.where(
            up_perm != -1, inv_idxs[np.maximum(up_perm, 0)], -1
        )

        self._pit_indices: ArrayInt32 = np.where(
            is_pit_perm[: self.n_kinematic + self.n_inertial]
        )[0].astype(np.int32)
        wb_outflow_indices: ArrayInt32 = np.where(is_waterbody_outflow_perm)[0].astype(
            np.int32
        )

        n_wb: int = int(is_waterbody_outflow.sum())
        self.n_wb: int = n_wb
        n_ret: int = retention_max_storage_m3.size
        assert controlled_retention.size == n_ret, (
            f"controlled_retention size ({controlled_retention.size}) must match "
            f"retention_max_storage_m3 size ({n_ret})."
        )

        self._river_length_inertial: ArrayFloat32 = river_length_perm[
            inertial_start:inertial_end
        ]
        self._bed_elevation_inertial: ArrayFloat32 = bed_elevation_perm[
            inertial_start:inertial_end
        ]
        routing_topo: RoutingTopology = build_routing_topology(
            n_kinematic=self.n_kinematic,
            n_inertial=self.n_inertial,
            n_kin_headwater=self.n_kin_headwater,
            upstream_matrix=upstream_matrix,
            ds_node=ds_node,
            bed_elevation_perm=bed_elevation_perm,
            river_length_perm=river_length_perm,
            is_pit_perm=is_pit_perm,
            is_ocean_pit_perm=self._is_ocean_pit,
            pit_slope_perm=self._pit_slope,
            waterbody_ids_perm=waterbody_ids_perm,
            use_kinematic_perm=use_kinematic_perm,
            is_waterbody_outflow_perm=is_waterbody_outflow_perm,
            retention_node_id_perm=retention_node_id_perm,
            wb_outflow_indices=wb_outflow_indices,
            river_storage_alpha_kin=river_storage_alpha[
                self.sorted_idxs[: self.n_kinematic]
            ].astype(np.float32),
            river_storage_beta_kin=river_storage_beta[
                self.sorted_idxs[: self.n_kinematic]
            ].astype(np.float32),
        )

        self._routing_topology: RoutingTopology = routing_topo
        self._topology: InertialTopology = routing_topo.topology
        self._kinematic_topology: KinematicTopology = routing_topo.kinematic_topology
        self._inertial_boundary_outflow_k: ArrayInt32 = (
            routing_topo.inertial_boundary_outflow_k
        )
        self._wb_links: WaterbodyLinks = routing_topo.waterbody_links
        self._ret_links: RetentionLinks = routing_topo.retention_links

        self._compute_static_geometry()

        self._river_storage_perm: ArrayFloat64 = np.empty(n_up_nodes, dtype=np.float64)
        self._discharge_prev_perm: ArrayFloat32 = np.empty(n_up_nodes, dtype=np.float32)
        self._sideflow_perm: ArrayFloat32 = np.empty(n_up_nodes, dtype=np.float32)
        self._evaporation_perm: ArrayFloat32 = np.empty(n_up_nodes, dtype=np.float32)
        self._over_abstraction_perm: ArrayFloat32 = np.empty(
            n_up_nodes, dtype=np.float32
        )
        self._actual_evaporation_perm: ArrayFloat32 = np.empty(
            n_up_nodes, dtype=np.float32
        )
        self._updated_discharge_perm: ArrayFloat32 = np.empty(
            n_up_nodes, dtype=np.float32
        )

        # Persistent inertial slice views avoiding per-timestep slicing overhead
        self._river_storage_inertial: ArrayFloat64 = self._river_storage_perm[
            inertial_start:inertial_end
        ]
        self._discharge_prev_inertial: ArrayFloat32 = self._discharge_prev_perm[
            inertial_start:inertial_end
        ]
        self._sideflow_inertial: ArrayFloat32 = self._sideflow_perm[
            inertial_start:inertial_end
        ]
        self._evaporation_inertial: ArrayFloat32 = self._evaporation_perm[
            inertial_start:inertial_end
        ]
        self._over_abs_inertial: ArrayFloat32 = self._over_abstraction_perm[
            inertial_start:inertial_end
        ]
        self._actual_evap_inertial: ArrayFloat32 = self._actual_evaporation_perm[
            inertial_start:inertial_end
        ]
        self._updated_discharge_inertial: ArrayFloat32 = self._updated_discharge_perm[
            inertial_start:inertial_end
        ]

        # Buffer holding both reach stages and lake stages
        self._stage_buf: ArrayFloat32 = np.empty(n_inertial + n_wb, dtype=np.float32)

        self._substep_buffers: SubstepBuffers = SubstepBuffers(
            stage_buf=self._stage_buf,
            min_dt_buf=np.empty(n_inertial, dtype=np.float32),
            rev_demand_buf=np.empty(n_inertial, dtype=np.float32),
            wb_release_volume_substep=np.empty(n_inertial, dtype=np.float32),
            total_flow_rate_buf=np.empty(n_inertial, dtype=np.float32),
            discharge_vol_sum_inertial=np.empty(n_inertial, dtype=np.float32),
            retention_inflow_limit_sub_buf=np.empty(n_ret, dtype=np.float64),
            retention_max_outflow_limit_sub_buf=np.empty(n_ret, dtype=np.float64),
            wb_outflow_avail_buf=np.empty(n_wb, dtype=np.float64),
            wb_extra_lateral_accum_m3=np.zeros(n_inertial, dtype=np.float64),
            terminal_wb_outflow_accum_m3=np.zeros(1, dtype=np.float32),
            substep_discharge_inertial=np.empty(n_inertial, dtype=np.float32),
            kinematic_inflow_rate_inertial=np.empty(n_inertial, dtype=np.float32),
        )

        self._waterbody_inflow_perm: ArrayFloat32 = np.empty(n_wb, dtype=np.float32)
        self._retention_inflow_perm: ArrayFloat32 = np.empty(n_ret, dtype=np.float32)
        self._retention_outflow_perm: ArrayFloat32 = np.empty(n_ret, dtype=np.float32)

        if waterbody_lake_area.size != n_wb:
            raise ValueError(
                f"waterbody_lake_area size ({waterbody_lake_area.size}) must match number of waterbodies ({n_wb})."
            )
        if waterbody_lake_factor.size != n_wb:
            raise ValueError(
                f"waterbody_lake_factor size ({waterbody_lake_factor.size}) must match number of waterbodies ({n_wb})."
            )
        if waterbody_outflow_height.size != n_wb:
            raise ValueError(
                f"waterbody_outflow_height size ({waterbody_outflow_height.size}) must match number of waterbodies ({n_wb})."
            )
        if waterbody_outflow_bed_elev.size != n_wb:
            raise ValueError(
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

        Raises:
            ValueError: If waterbody storage is missing when waterbodies exist.
        """
        inertial_start: int = self.n_kinematic

        if river_storage_m3 is not None:
            if already_permuted:
                self._river_storage_perm[:] = river_storage_m3
            else:
                self._river_storage_perm[:] = river_storage_m3[self.sorted_idxs]
        else:
            self._river_storage_perm[:] = 0.0

        inertial_end: int = inertial_start + self.n_inertial
        if water_stage_m is not None:
            if already_permuted:
                self._stage_buf[: self.n_inertial] = water_stage_m[
                    inertial_start:inertial_end
                ]
            else:
                self._stage_buf[: self.n_inertial] = water_stage_m[
                    self.sorted_idxs[inertial_start:inertial_end]
                ]
        else:
            # Compute stage directly from storage to ensure strict numerical consistency with geometry
            _compute_stage_from_storage_inertial(
                self.n_inertial,
                self._river_storage_perm[inertial_start:inertial_end],
                self._geometry.geom_inbank,
                self._geometry.geom_overbank,
                self._stage_buf,
            )

        if self.n_wb > 0:
            if waterbody_storage_m3 is None:
                raise ValueError("waterbody_storage_m3 must be provided when n_wb > 0")
            depth_from_bottom: ArrayFloat32 = (
                waterbody_storage_m3.astype(np.float32) / self._wb_lake_area
            )
            bottom_elevation: ArrayFloat32 = (
                self._wb_outflow_bed_elev - self._wb_outflow_height
            )
            self._stage_buf[self.n_inertial : self.n_inertial + self.n_wb] = (
                bottom_elevation + depth_from_bottom
            )

        check_spinup: bool = self.in_spinup if in_spinup is None else in_spinup
        if check_spinup and self.n_inertial > 0:
            internal_inertial_mask: ArrayBool = self._topology.ds_boundary_type == 0
            if np.any(internal_inertial_mask):
                k_up: ArrayInt32 = np.where(internal_inertial_mask)[0].astype(np.int32)
                k_ds: ArrayInt32 = self._topology.ds_inertial_k[k_up]

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
        stage_out: ArrayFloat32 = (
            np.empty(n_cells, dtype=np.float32) if out is None else out
        )

        _scatter_water_stage(
            sorted_idxs=self.sorted_idxs,
            n_kinematic=self.n_kinematic,
            n_inertial=self.n_inertial,
            n_wb=self.n_wb,
            waterbody_ids=self.waterbody_ids,
            bed_elevation=self.bed_elevation,
            stage_buf=self._stage_buf,
            stage_out=stage_out,
        )

        return stage_out

    def _compute_static_geometry(self) -> None:
        """Precomputes reach-static channel cross-sections, thresholds, and CFL parameters."""
        inertial_start: int = self.n_kinematic
        inertial_end: int = self.n_kinematic + self.n_inertial
        river_width_inertial: ArrayFloat32 = self._river_width[
            inertial_start:inertial_end
        ]
        bankfull_depth_inertial: ArrayFloat32 = self._bankfull_depth[
            inertial_start:inertial_end
        ]
        self._geometry: InertialGeometry = build_inertial_geometry(
            river_length=self._river_length_inertial,
            river_width=river_width_inertial,
            shape_exponent=self._shape_exponent_inertial,
            bankfull_depth=bankfull_depth_inertial,
            floodplain_width=self._floodplain_width_inertial,
            bed_elevation=self._bed_elevation_inertial,
            ds_bed_elevation=self._topology.ds_bed_elevation,
            interface_bed_elevation_max=self._routing_topology.interface_bed_elev_max,
            manning_n_sq=self._manning_n_sq_inertial,
            inv_dx_interface=self._routing_topology.inv_dx_interface,
            up_count=self._routing_topology.inertial_up_count,
            pit_slope=self._pit_slope[inertial_start:inertial_end],
        )

    def update_channel_geometry(
        self,
        river_width: ArrayFloat32,
        bankfull_depth_m: ArrayFloat32,
    ) -> None:
        """Updates channel bankfull top width and depth dynamically during simulation.

        Args:
            river_width: Updated channel bankfull top width (meters).
            bankfull_depth_m: Updated bankfull channel depth (meters).
        """
        self.river_width = river_width.astype(np.float32)
        self._river_width = self.river_width[self.sorted_idxs]
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

        self._substep_buffers.wb_extra_lateral_accum_m3.fill(0.0)
        self._substep_buffers.terminal_wb_outflow_accum_m3.fill(0.0)

        inertial_start: int = self.n_kinematic
        inertial_end: int = self.n_kinematic + self.n_inertial
        if __debug__:
            storage_inertial_before: float = float(
                np.sum(self._river_storage_perm[inertial_start:inertial_end])
            )
            sideflow_inertial: float = float(
                np.sum(self._sideflow_perm[inertial_start:inertial_end])
            )

        wb_coupling: WaterbodyCoupling = WaterbodyCoupling(
            storage_m3=waterbody_storage_m3,
            prescribed_outflow_m3=outflow_per_waterbody_m3,
            inflow_m3=self._waterbody_inflow_perm,
            lake_area=self._wb_lake_area,
            lake_factor=self._wb_lake_factor,
            outflow_height=self._wb_outflow_height,
            outflow_bed_elev=self._wb_outflow_bed_elev,
            inertial_wb_ds_k=self._wb_links.inertial_wb_ds_k,
            inertial_wb_ds_id=self._wb_links.inertial_wb_ds_id,
            lake_outflow_target_k=self._wb_links.lake_outflow_target_k,
            lake_outflow_wb_id=self._wb_links.lake_outflow_wb_id,
            wb_to_wb_src_wb=self._wb_links.wb_to_wb_src_wb,
            wb_to_wb_tgt_wb=self._wb_links.wb_to_wb_tgt_wb,
            wb_terminal_wb_ids=self._wb_links.wb_terminal_wb_ids,
        )
        ret_coupling: RetentionCoupling = RetentionCoupling(
            storage_m3=retention_storage_m3,
            inflow_m3=self._retention_inflow_perm,
            outflow_m3=self._retention_outflow_perm,
            max_storage_m3=self.retention_max_storage_m3,
            inertial_retention_k=self._ret_links.inertial_retention_k,
            inertial_retention_id=self._ret_links.inertial_retention_id,
            controlled_retention=self.controlled_retention,
            activation_threshold_m3_s=retention_activation_threshold_m3_s,
            release_threshold_factor=self.retention_basin_release_threshold_factor,
        )

        dt_f32: np.float32 = np.float32(self.dt)
        inv_dt_f32: np.float32 = np.float32(1.0 / dt_f32)

        self._over_abstraction_perm.fill(0.0)
        self._actual_evaporation_perm.fill(0.0)
        self._waterbody_inflow_perm.fill(0.0)
        self._retention_inflow_perm.fill(0.0)
        self._retention_outflow_perm.fill(0.0)
        self._updated_discharge_perm.fill(0.0)
        self._substep_buffers.kinematic_inflow_rate_inertial.fill(0.0)

        n_other_start: int = self.n_kinematic + self.n_inertial
        self._updated_discharge_perm[n_other_start:] = np.nan

        _transfer_waterbody_outflows(
            wb=wb_coupling,
            kin=self._kinematic_topology,
        )

        if self.n_kinematic > 0:
            run_kinematic = (
                _run_kinematic_step_parallel
                if self.n_kinematic >= KINEMATIC_PARALLEL_THRESHOLD
                else _run_kinematic_step_serial
            )
            run_kinematic(
                dt_f32=dt_f32,
                inv_dt_f32=inv_dt_f32,
                previous_discharge_m3_s=self._discharge_prev_perm,
                river_storage_m3=self._river_storage_perm,
                sideflow_m3=self._sideflow_perm,
                evaporation_m3=self._evaporation_perm,
                over_abstraction_m3=self._over_abstraction_perm,
                actual_evaporation_m3=self._actual_evaporation_perm,
                updated_discharge_m3_s=self._updated_discharge_perm,
                kin=self._kinematic_topology,
                wb=wb_coupling,
                ret=ret_coupling,
            )

        if self.n_inertial > 0:
            river_storage_m3_inertial: ArrayFloat64 = self._river_storage_inertial
            previous_discharge_m3_s_inertial: ArrayFloat32 = (
                self._discharge_prev_inertial
            )
            sideflow_m3_inertial: ArrayFloat32 = self._sideflow_inertial
            evaporation_m3_inertial: ArrayFloat32 = self._evaporation_inertial
            over_abstraction_m3_inertial: ArrayFloat32 = self._over_abs_inertial
            actual_evaporation_m3_inertial: ArrayFloat32 = self._actual_evap_inertial
            updated_discharge_m3_s_inertial: ArrayFloat32 = (
                self._updated_discharge_inertial
            )
            substep_discharge_inertial: ArrayFloat32 = (
                self._substep_buffers.substep_discharge_inertial
            )
            kinematic_inflow_rate_inertial: ArrayFloat32 = (
                self._substep_buffers.kinematic_inflow_rate_inertial
            )

            run_cfl = (
                _compute_inertial_substeps_cfl_parallel
                if self.n_inertial >= INERTIAL_PARALLEL_THRESHOLD
                else _compute_inertial_substeps_cfl_serial
            )
            diagnostics: RoutingStepDiagnostics = run_cfl(
                dt_f32=dt_f32,
                inv_dt_f32=inv_dt_f32,
                n_inertial=self.n_inertial,
                kin=self._kinematic_topology,
                previous_discharge_m3_s_inertial=previous_discharge_m3_s_inertial,
                updated_discharge_m3_s=self._updated_discharge_perm,
                river_storage_m3_inertial=river_storage_m3_inertial,
                sideflow_m3_inertial=sideflow_m3_inertial,
                kinematic_inflow_rate_inertial=kinematic_inflow_rate_inertial,
                geom=self._geometry,
                topo=self._topology,
                wb=wb_coupling,
                buffers=self._substep_buffers,
            )

            num_inertial_substeps: int = diagnostics.num_inertial_substeps
            substep_discharge_inertial[:] = previous_discharge_m3_s_inertial[:]

            run_inertial = (
                _run_inertial_substeps_parallel
                if self.n_inertial >= INERTIAL_PARALLEL_THRESHOLD
                else _run_inertial_substeps_serial
            )
            run_inertial(
                num_inertial_substeps=num_inertial_substeps,
                dt_f32=dt_f32,
                n_inertial=self.n_inertial,
                substep_discharge_m3_s=substep_discharge_inertial,
                river_storage_m3_inertial=river_storage_m3_inertial,
                sideflow_m3_inertial=sideflow_m3_inertial,
                evaporation_m3_inertial=evaporation_m3_inertial,
                over_abstraction_m3_inertial=over_abstraction_m3_inertial,
                actual_evaporation_m3_inertial=actual_evaporation_m3_inertial,
                updated_discharge_m3_s_inertial=updated_discharge_m3_s_inertial,
                kinematic_inflow_rate=kinematic_inflow_rate_inertial,
                geom=self._geometry,
                topo=self._topology,
                wb=wb_coupling,
                ret=ret_coupling,
                buffers=self._substep_buffers,
            )

        terminal_wb_outflow_m3: np.float32 = (
            self._substep_buffers.terminal_wb_outflow_accum_m3[0]
        )

        # if self.n_inertial > 0 and diagnostics.limiting_reach_idx >= 0:
        #     self._print_limiting_substep(diagnostics)

        if __debug__:
            storage_inertial_after: float = float(
                np.sum(self._river_storage_perm[inertial_start:inertial_end])
            )
            kinematic_inflow_total: float = float(
                np.sum(self._substep_buffers.kinematic_inflow_rate_inertial) * self.dt
            )
            evap_inertial: float = float(
                np.sum(self._actual_evaporation_perm[inertial_start:inertial_end])
            )
            over_abs_inertial: float = float(
                np.sum(self._over_abstraction_perm[inertial_start:inertial_end])
            )
            wb_extra_lateral_total: float = float(
                np.sum(self._substep_buffers.wb_extra_lateral_accum_m3)
            )

            net_retention_inertial: float = (
                float(
                    np.sum(
                        self._retention_inflow_perm[
                            self._ret_links.inertial_retention_id
                        ]
                        - self._retention_outflow_perm[
                            self._ret_links.inertial_retention_id
                        ]
                    )
                )
                if len(self._ret_links.inertial_retention_id) > 0
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

        # Add terminal waterbody releases outside the domain to pit outflows
        # so total domain boundary loss is conserved in the overall water balance.
        outflow_at_pits_m3: np.float32 = (
            (
                _sum_indexed_discharge(self._updated_discharge_perm, self._pit_indices)
                * np.float32(self.dt)
            )
            if len(self._pit_indices) > 0
            else np.float32(0.0)
        ) + terminal_wb_outflow_m3

        if not already_permuted:
            _scatter_outputs(
                self.sorted_idxs,
                self._updated_discharge_perm,
                self._actual_evaporation_perm,
                self._over_abstraction_perm,
                self._river_storage_perm,
                self._discharge_out,
                self._actual_evap_out,
                self._over_abs_out,
                river_storage_m3,
            )
            out_discharge = self._discharge_out
            out_storage = river_storage_m3
            out_actual_evap = self._actual_evap_out
            out_over_abs = self._over_abs_out
        else:
            out_discharge = self._updated_discharge_perm
            out_storage = self._river_storage_perm
            out_actual_evap = self._actual_evaporation_perm
            out_over_abs = self._over_abstraction_perm

        return (
            out_discharge,
            out_storage,
            out_actual_evap,
            out_over_abs,
            waterbody_storage_m3,
            self._waterbody_inflow_perm,
            outflow_at_pits_m3,
            retention_storage_m3,
            self._retention_inflow_perm,
            self._retention_outflow_perm,
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
            updated_discharge_inertial: ArrayFloat32 = self._updated_discharge_perm[
                self.n_kinematic : self.n_kinematic + self.n_inertial
            ]
            net_inertial_outflow_m3: float = float(
                np.sum(updated_discharge_inertial[self._inertial_boundary_outflow_k])
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
