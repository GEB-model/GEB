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
    INERTIAL_PARALLEL_THRESHOLD,
    InertialGeometry,
    InertialTopology,
    KinematicTopology,
    RetentionCoupling,
    RoutingStepDiagnostics,
    SubstepBuffers,
    WaterbodyCoupling,
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
            ret.storage_m3[ret_id],
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


class LocalInertial:
    """Local inertial river network router with CaMa-Flood style dead-storage floodplain geometry.

    Combines non-linear kinematic wave routing for overland / non-river cells with a
    1D local inertial dynamic routing formulation for river network reaches,
    supporting trapezoidal channels and CaMa-Flood style dead-storage floodplains.
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
        self._iteration_count: int = 0
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
        self._wb_outflow_indices: ArrayInt32 = np.where(is_waterbody_outflow_perm)[
            0
        ].astype(np.int32)

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
        self._setup_inertial_boundary_arrays(
            upstream_matrix=upstream_matrix,
            ds_node=ds_node,
            bed_elevation_perm=bed_elevation_perm,
            river_length_perm=river_length_perm,
            is_pit_perm=is_pit_perm,
            waterbody_ids_perm=waterbody_ids_perm,
            use_kinematic_perm=use_kinematic_perm,
            is_waterbody_outflow_perm=is_waterbody_outflow_perm,
            retention_node_id_perm=retention_node_id_perm,
        )
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

        self._geometry: InertialGeometry = InertialGeometry(
            geom_inbank=self._geom_inbank,
            geom_overbank=self._geom_overbank,
            geom_cfl=self._geom_cfl,
            pit_slope_inertial=self._pit_slope_inertial,
        )
        self._topology: InertialTopology = InertialTopology(
            inertial_up_offsets=self._inertial_up_offsets,
            inertial_up_indices=self._inertial_up_indices,
            inertial_up_reach_idx=self._inertial_up_reach_idx,
            ds_boundary_type=self._ds_boundary_type,
            ds_inertial_k=self._ds_inertial_k,
            ds_stage_idx=self._ds_stage_idx,
            ds_bed_elevation=self._ds_bed_elevation,
            inertial_topo_order=self._inertial_topo_order,
        )
        alpha_perm: ArrayFloat32 = river_storage_alpha[
            self.sorted_idxs[:n_kinematic]
        ].astype(np.float32)
        beta_perm: ArrayFloat32 = river_storage_beta[
            self.sorted_idxs[:n_kinematic]
        ].astype(np.float32)
        self._kinematic_topology: KinematicTopology = KinematicTopology(
            n_kinematic=self.n_kinematic,
            n_kin_headwater=self.n_kin_headwater,
            kin_ds_reach=self._kin_ds_reach,
            kin_inflow_m3_s=self._kin_inflow_m3_s,
            kin_ds_waterbody_id=self._kin_ds_waterbody_id,
            wb_to_kin_target_reach=self._wb_to_kin_target_reach,
            wb_to_kin_wb_id=self._wb_to_kin_wb_id,
            kin_wb_inflow_m3=self._kin_wb_inflow_m3,
            inertial_up_kin_offsets=self._inertial_up_kin_offsets,
            inertial_up_kin_indices=self._inertial_up_kin_indices,
            inertial_up_kin_reach_idx=self._inertial_up_kin_reach_idx,
            river_storage_alpha=alpha_perm,
            river_storage_beta=beta_perm,
            river_length=river_length_perm[:n_kinematic],
            retention_node_id=retention_node_id_perm[:n_kinematic],
        )

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
                self._river_storage_perm[
                    self.n_kinematic : self.n_kinematic + self.n_inertial
                ],
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

    def _setup_inertial_boundary_arrays(
        self,
        upstream_matrix: TwoDArrayInt32,
        ds_node: ArrayInt32,
        bed_elevation_perm: ArrayFloat32,
        river_length_perm: ArrayFloat32,
        is_pit_perm: ArrayBool,
        waterbody_ids_perm: ArrayInt32,
        use_kinematic_perm: ArrayBool,
        is_waterbody_outflow_perm: ArrayBool,
        retention_node_id_perm: ArrayInt32,
    ) -> None:
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

        for k in range(n_inertial):
            i: int = n_kinematic + k
            ds: int = int(ds_node[i])
            bed_node: float = float(bed_elevation_perm[i])
            r_len: float = max(float(river_length_perm[i]), 1.0)

            if is_pit_perm[i]:
                if self._is_ocean_pit[i]:
                    ds_boundary_type[k] = 4
                    ds_bed_elevation[k] = np.float32(bed_node)
                else:
                    ds_boundary_type[k] = 3
                    eff_slope: float = float(self._pit_slope[i])
                    ds_bed_elevation[k] = np.float32(bed_node - eff_slope * r_len)
            elif ds != -1:
                bed_ds: float = float(bed_elevation_perm[ds])
                ds_bed_elevation[k] = np.float32(bed_ds)
                wb_ds: int = int(waterbody_ids_perm[ds])
                if wb_ds != -1:
                    ds_boundary_type[k] = 2
                    ds_waterbody_id[k] = wb_ds
                else:
                    assert not use_kinematic_perm[ds], (
                        f"Inertial reach at sorted index {i} cannot discharge into a kinematic reach."
                    )
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

        # Upstream reach connectivity for inertial reaches
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
                if (
                    waterbody_ids_perm[up_node] == -1
                    and not use_kinematic_perm[up_node]
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
            ret_id: int = int(retention_node_id_perm[n_kinematic + k])
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

        # Waterbody outlet routing links (classified by downstream destination)
        lake_outflow_targets: list[int] = []
        lake_outflow_wbs: list[int] = []
        wb_to_wb_src: list[int] = []
        wb_to_wb_tgt: list[int] = []
        wb_to_kin_targets: list[int] = []
        wb_to_kin_wbs: list[int] = []
        wb_term_ids: list[int] = []

        for idx in range(len(self._wb_outflow_indices)):
            wb_node: int = int(self._wb_outflow_indices[idx])
            wb_id: int = int(waterbody_ids_perm[wb_node])
            if wb_id == -1:
                continue
            ds: int = int(ds_node[wb_node])

            if ds == -1 or is_pit_perm[wb_node]:
                wb_term_ids.append(wb_id)
            elif waterbody_ids_perm[ds] != -1:
                target_wb: int = int(waterbody_ids_perm[ds])
                if target_wb != wb_id:
                    wb_to_wb_src.append(wb_id)
                    wb_to_wb_tgt.append(target_wb)
            elif 0 <= ds < n_kinematic:
                wb_to_kin_targets.append(ds)
                wb_to_kin_wbs.append(wb_id)
            elif n_kinematic <= ds < n_kinematic + n_inertial:
                lake_outflow_targets.append(ds - n_kinematic)
                lake_outflow_wbs.append(wb_id)

        self._lake_outflow_target_k: ArrayInt32 = np.array(
            lake_outflow_targets, dtype=np.int32
        )
        self._lake_outflow_wb_id: ArrayInt32 = np.array(
            lake_outflow_wbs, dtype=np.int32
        )
        self._wb_to_wb_src_wb: ArrayInt32 = np.array(wb_to_wb_src, dtype=np.int32)
        self._wb_to_wb_tgt_wb: ArrayInt32 = np.array(wb_to_wb_tgt, dtype=np.int32)
        self._wb_to_kin_target_reach: ArrayInt32 = np.array(
            wb_to_kin_targets, dtype=np.int32
        )
        self._wb_to_kin_wb_id: ArrayInt32 = np.array(wb_to_kin_wbs, dtype=np.int32)
        self._kin_wb_inflow_m3: ArrayFloat32 = np.zeros(n_kinematic, dtype=np.float32)
        self._wb_terminal_wb_ids: ArrayInt32 = np.array(wb_term_ids, dtype=np.int32)

        # Downstream targets for kinematic reaches
        kin_ds_wb_id: ArrayInt32 = np.full(n_kinematic, -1, dtype=np.int32)
        kin_ds_reach: ArrayInt32 = np.full(n_kinematic, -1, dtype=np.int32)
        for i in range(n_kinematic):
            ds = int(ds_node[i])
            if ds != -1:
                if waterbody_ids_perm[ds] != -1:
                    kin_ds_wb_id[i] = waterbody_ids_perm[ds]
                elif 0 <= ds < n_kinematic:
                    kin_ds_reach[i] = ds
        self._kin_ds_waterbody_id: ArrayInt32 = kin_ds_wb_id
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
        self._river_width_inertial: ArrayFloat32 = self._river_width[
            inertial_start:inertial_end
        ]
        self._bankfull_depth_inertial: ArrayFloat32 = self._bankfull_depth[
            inertial_start:inertial_end
        ]
        bed_elevation_inertial: ArrayFloat32 = self._bed_elevation_inertial

        (
            inv_reach_length,
            inv_shape_exponent_plus_one,
            _bankfull_area,
            bankfull_volume,
            bankfull_perimeter,
            fp_side_slope,
            fp_depth_threshold,
            fp_area_threshold,
            _sqrt_fp_slope_sq,
            width_over_sqrt_bankfull_depth,
        ) = compute_static_geometry(
            river_length=self._river_length_inertial,
            river_width=self._river_width_inertial,
            shape_exponent=self._shape_exponent_inertial,
            bankfull_depth=self._bankfull_depth_inertial,
            floodplain_width=self._floodplain_width_inertial,
        )

        stage_vol_coeff: ArrayFloat32 = (
            self._bankfull_depth_inertial
            / (bankfull_volume**inv_shape_exponent_plus_one)
        ).astype(np.float32)
        self._pit_slope_inertial: ArrayFloat32 = self._pit_slope[
            inertial_start:inertial_end
        ]

        # Precompute in-bank channel geometry
        self._geom_inbank: TwoDArrayFloat32 = np.empty(
            (self.n_inertial, GEOM_IN_NUM_COLS), dtype=np.float32
        )
        self._geom_inbank[:, GEOM_IN_INVERSE_LENGTH] = inv_reach_length
        self._geom_inbank[:, GEOM_IN_BED_ELEVATION] = bed_elevation_inertial
        self._geom_inbank[:, GEOM_IN_BANKFULL_DEPTH] = self._bankfull_depth_inertial
        self._geom_inbank[:, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE] = (
            inv_shape_exponent_plus_one
        )
        self._geom_inbank[:, GEOM_IN_BANKFULL_VOLUME] = bankfull_volume
        self._geom_inbank[:, GEOM_IN_STAGE_VOLUME_COEFFICIENT] = stage_vol_coeff
        self._geom_inbank[:, GEOM_IN_INTERFACE_BED_ELEVATION_MAX] = (
            self._interface_bed_elev_max
        )
        self._geom_inbank[:, GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH] = (
            width_over_sqrt_bankfull_depth
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
        self._geom_overbank[:, GEOM_OV_BANKFULL_PERIMETER] = bankfull_perimeter
        self._geom_overbank[:, GEOM_OV_FLOODPLAIN_SIDE_SLOPE] = fp_side_slope
        self._geom_overbank[:, GEOM_OV_FLOODPLAIN_DEPTH_THRESHOLD] = fp_depth_threshold
        self._geom_overbank[:, GEOM_OV_FLOODPLAIN_AREA_THRESHOLD] = fp_area_threshold

        # Precompute CFL stability parameters
        self._geom_cfl: TwoDArrayFloat32 = np.empty(
            (self.n_inertial, GEOM_CFL_NUM_COLS), dtype=np.float32
        )
        bed_drop: ArrayFloat32 = np.maximum(
            bed_elevation_inertial - self._ds_bed_elevation,
            np.float32(0.0),
        )
        eff_slope: ArrayFloat32 = np.maximum(
            bed_drop * inv_reach_length, np.float32(1e-4)
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
        self._geometry = InertialGeometry(
            geom_inbank=self._geom_inbank,
            geom_overbank=self._geom_overbank,
            geom_cfl=self._geom_cfl,
            pit_slope_inertial=self._pit_slope_inertial,
        )

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
        kin: KinematicTopology,
        n_inertial: int,
        over_abstraction_m3: ArrayFloat32,
        actual_evaporation_m3: ArrayFloat32,
        updated_discharge_m3_s: ArrayFloat32,
        geom: InertialGeometry,
        topo: InertialTopology,
        wb: WaterbodyCoupling,
        ret: RetentionCoupling,
        buffers: SubstepBuffers,
    ) -> tuple[
        ArrayFloat32,
        ArrayFloat64,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        np.float32,
        RoutingStepDiagnostics,
    ]:
        """Routes a single macro-timestep across kinematic and inertial subdomains.

        Args:
            routing_timestep_s: Duration of macro timestep (seconds).
            previous_discharge_m3_s: Pre-allocated discharge array at start of timestep (m³/s).
            river_storage_m3: Full-network reach water storage array (m³).
            sideflow_m3: Full-network lateral sideflow volume array (m³).
            evaporation_m3: Full-network potential channel evaporation volume array (m³).
            kin: Kinematic wave network topology and reach parameter tables.
            n_inertial: Number of reaches routed with local inertial equations.
            over_abstraction_m3: Full-network over-abstraction deficit volume accumulator (m³).
            actual_evaporation_m3: Full-network actual evaporation volume accumulator (m³).
            updated_discharge_m3_s: Output full-network time-averaged discharge (m³/s).
            geom: Static channel and floodplain geometry lookup tables.
            topo: Network connectivity and topological ordering structure.
            wb: Waterbody coupling parameters, links, and storage state arrays.
            ret: Flood retention basin state arrays, capacities, and control parameters.
            buffers: Pre-allocated workspace scratch buffers.

        Returns:
            A tuple of (updated_discharge, river_storage, actual_evap, over_abs, wb_inflow, ret_inflow, ret_outflow, terminal_wb_outflow, diagnostics).
        """
        n_kinematic: int = kin.n_kinematic
        dt_f32: np.float32 = np.float32(routing_timestep_s)
        inv_dt_f32: np.float32 = np.float32(1.0 / dt_f32)
        inertial_start: int = kin.n_kinematic
        inertial_end: int = kin.n_kinematic + n_inertial
        river_storage_m3_inertial: ArrayFloat64 = river_storage_m3[
            inertial_start:inertial_end
        ]
        previous_discharge_m3_s_inertial: ArrayFloat32 = previous_discharge_m3_s[
            inertial_start:inertial_end
        ]
        sideflow_m3_inertial: ArrayFloat32 = sideflow_m3[inertial_start:inertial_end]
        evaporation_m3_inertial: ArrayFloat32 = evaporation_m3[
            inertial_start:inertial_end
        ]
        over_abstraction_m3_inertial: ArrayFloat32 = over_abstraction_m3[
            inertial_start:inertial_end
        ]
        actual_evaporation_m3_inertial: ArrayFloat32 = actual_evaporation_m3[
            inertial_start:inertial_end
        ]
        updated_discharge_m3_s_inertial: ArrayFloat32 = updated_discharge_m3_s[
            inertial_start:inertial_end
        ]
        substep_discharge_inertial: ArrayFloat32 = buffers.substep_discharge_inertial
        kinematic_inflow_rate_inertial: ArrayFloat32 = (
            buffers.kinematic_inflow_rate_inertial
        )

        n_cells: int = len(river_storage_m3)

        over_abstraction_m3.fill(0.0)
        actual_evaporation_m3.fill(0.0)
        wb.inflow_m3.fill(0.0)
        ret.inflow_m3.fill(0.0)
        ret.outflow_m3.fill(0.0)
        updated_discharge_m3_s.fill(0.0)
        kinematic_inflow_rate_inertial.fill(0.0)

        n_other_start: int = n_kinematic + n_inertial
        for i in range(n_other_start, n_cells):
            updated_discharge_m3_s[i] = np.float32(np.nan)

        _transfer_waterbody_outflows(
            wb=wb,
            kin=kin,
        )

        if n_kinematic > 0:
            if n_kinematic >= KINEMATIC_PARALLEL_THRESHOLD:
                _run_kinematic_step_parallel(
                    dt_f32=dt_f32,
                    inv_dt_f32=inv_dt_f32,
                    previous_discharge_m3_s=previous_discharge_m3_s,
                    river_storage_m3=river_storage_m3,
                    sideflow_m3=sideflow_m3,
                    evaporation_m3=evaporation_m3,
                    over_abstraction_m3=over_abstraction_m3,
                    actual_evaporation_m3=actual_evaporation_m3,
                    updated_discharge_m3_s=updated_discharge_m3_s,
                    kin=kin,
                    wb=wb,
                    ret=ret,
                )
            else:
                _run_kinematic_step_serial(
                    dt_f32=dt_f32,
                    inv_dt_f32=inv_dt_f32,
                    previous_discharge_m3_s=previous_discharge_m3_s,
                    river_storage_m3=river_storage_m3,
                    sideflow_m3=sideflow_m3,
                    evaporation_m3=evaporation_m3,
                    over_abstraction_m3=over_abstraction_m3,
                    actual_evaporation_m3=actual_evaporation_m3,
                    updated_discharge_m3_s=updated_discharge_m3_s,
                    kin=kin,
                    wb=wb,
                    ret=ret,
                )

        diagnostics: RoutingStepDiagnostics = RoutingStepDiagnostics(
            num_inertial_substeps=0,
            limiting_reach_idx=-1,
            min_stable_timestep_s=dt_f32,
            h_vol=np.float32(0.0),
            h_flow=np.float32(0.0),
            h_gov=np.float32(0.0),
            celerity=np.float32(0.0),
            limiting_flow=np.float32(0.0),
            limiting_storage=np.float32(0.0),
        )

        if n_inertial > 0:
            diagnostics = compute_inertial_substeps_cfl(
                dt_f32=dt_f32,
                inv_dt_f32=inv_dt_f32,
                n_inertial=n_inertial,
                kin=kin,
                previous_discharge_m3_s_inertial=previous_discharge_m3_s_inertial,
                updated_discharge_m3_s=updated_discharge_m3_s,
                river_storage_m3_inertial=river_storage_m3_inertial,
                sideflow_m3_inertial=sideflow_m3_inertial,
                kinematic_inflow_rate_inertial=kinematic_inflow_rate_inertial,
                geom=geom,
                topo=topo,
                wb=wb,
                buffers=buffers,
            )

            num_inertial_substeps: int = diagnostics.num_inertial_substeps
            substep_discharge_inertial[:] = previous_discharge_m3_s_inertial[:]

            if n_inertial >= INERTIAL_PARALLEL_THRESHOLD:
                actual_num_substeps: int = _run_inertial_substeps_parallel(
                    num_inertial_substeps=num_inertial_substeps,
                    dt_f32=dt_f32,
                    n_inertial=n_inertial,
                    substep_discharge_m3_s=substep_discharge_inertial,
                    river_storage_m3_inertial=river_storage_m3_inertial,
                    sideflow_m3_inertial=sideflow_m3_inertial,
                    evaporation_m3_inertial=evaporation_m3_inertial,
                    over_abstraction_m3_inertial=over_abstraction_m3_inertial,
                    actual_evaporation_m3_inertial=actual_evaporation_m3_inertial,
                    updated_discharge_m3_s_inertial=updated_discharge_m3_s_inertial,
                    kinematic_inflow_rate=kinematic_inflow_rate_inertial,
                    geom=geom,
                    topo=topo,
                    wb=wb,
                    ret=ret,
                    buffers=buffers,
                )
            else:
                actual_num_substeps: int = _run_inertial_substeps_serial(
                    num_inertial_substeps=num_inertial_substeps,
                    dt_f32=dt_f32,
                    n_inertial=n_inertial,
                    substep_discharge_m3_s=substep_discharge_inertial,
                    river_storage_m3_inertial=river_storage_m3_inertial,
                    sideflow_m3_inertial=sideflow_m3_inertial,
                    evaporation_m3_inertial=evaporation_m3_inertial,
                    over_abstraction_m3_inertial=over_abstraction_m3_inertial,
                    actual_evaporation_m3_inertial=actual_evaporation_m3_inertial,
                    updated_discharge_m3_s_inertial=updated_discharge_m3_s_inertial,
                    kinematic_inflow_rate=kinematic_inflow_rate_inertial,
                    geom=geom,
                    topo=topo,
                    wb=wb,
                    ret=ret,
                    buffers=buffers,
                )

            if actual_num_substeps != num_inertial_substeps:
                diagnostics = RoutingStepDiagnostics(
                    num_inertial_substeps=actual_num_substeps,
                    limiting_reach_idx=diagnostics.limiting_reach_idx,
                    min_stable_timestep_s=diagnostics.min_stable_timestep_s,
                    h_vol=diagnostics.h_vol,
                    h_flow=diagnostics.h_flow,
                    h_gov=diagnostics.h_gov,
                    celerity=diagnostics.celerity,
                    limiting_flow=diagnostics.limiting_flow,
                    limiting_storage=diagnostics.limiting_storage,
                )

        terminal_wb_outflow_m3: np.float32 = buffers.terminal_wb_outflow_accum_m3[0]

        return (
            updated_discharge_m3_s,
            river_storage_m3,
            actual_evaporation_m3,
            over_abstraction_m3,
            wb.inflow_m3,
            ret.inflow_m3,
            ret.outflow_m3,
            terminal_wb_outflow_m3,
            diagnostics,
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

    def _print_limiting_substep(
        self,
        diagnostics: RoutingStepDiagnostics,
    ) -> None:
        """Prints diagnostic information for the cell requiring the smallest substep.

        Args:
            diagnostics: Encapsulated routing step stability diagnostics.
        """
        limiting_reach_idx: int = diagnostics.limiting_reach_idx
        river_id: int = int(self._river_ids[self.n_kinematic + limiting_reach_idx])
        grid_idx: int = int(self.sorted_idxs[self.n_kinematic + limiting_reach_idx])
        dx: float = float(self._river_length_inertial[limiting_reach_idx])
        up_count: int = int(self._inertial_up_count[limiting_reach_idx])

        h_v: float = float(diagnostics.h_vol)
        h_f: float = float(diagnostics.h_flow)
        h_g: float = float(diagnostics.h_gov)
        c: float = float(diagnostics.celerity)
        q: float = float(diagnostics.limiting_flow)
        v: float = float(diagnostics.limiting_storage)
        min_dt: float = float(diagnostics.min_stable_timestep_s)
        num_substeps: int = diagnostics.num_inertial_substeps

        if h_v >= h_f:
            gov_str: str = (
                f"channel storage depth (h_vol={h_v:.2f} m >= h_flow={h_f:.2f} m, "
                f"storage={v:.1f} m³)"
            )
        else:
            gov_str: str = (
                f"flow normal depth / Manning (h_flow={h_f:.2f} m > h_vol={h_v:.2f} m, "
                f"flow={q:.2f} m³/s)"
            )

        confluence_info: str = (
            f", confluence of {up_count} tributaries" if up_count > 1 else ""
        )

        substep_str: str = (
            f"{num_substeps} substep"
            if num_substeps == 1
            else f"{num_substeps} substeps"
        )

        print(
            f"[Routing Iteration {self._iteration_count}] Cell reach_idx={limiting_reach_idx} "
            f"(river_id={river_id}, grid_idx={grid_idx}) "
            f"requires smallest substep dt={float(min_dt):.2f} s ({substep_str}).\n"
            f"  Why: reach length dx={dx:.1f} m, governing depth h={h_g:.2f} m via {gov_str}, "
            f"wave celerity c={c:.2f} m/s{confluence_info}."
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

        inertial_start: int = self.n_kinematic
        inertial_end: int = self.n_kinematic + self.n_inertial
        storage_inertial_before: float = float(
            np.sum(self._river_storage_perm[inertial_start:inertial_end])
        )
        sideflow_inertial: float = float(
            np.sum(self._sideflow_perm[inertial_start:inertial_end])
        )

        self._iteration_count += 1
        wb_coupling: WaterbodyCoupling = WaterbodyCoupling(
            storage_m3=waterbody_storage_m3,
            prescribed_outflow_m3=outflow_per_waterbody_m3,
            inflow_m3=self._waterbody_inflow_perm,
            lake_area=self._wb_lake_area,
            lake_factor=self._wb_lake_factor,
            outflow_height=self._wb_outflow_height,
            outflow_bed_elev=self._wb_outflow_bed_elev,
            inertial_wb_ds_k=self._inertial_wb_ds_k,
            inertial_wb_ds_id=self._inertial_wb_ds_id,
            lake_outflow_target_k=self._lake_outflow_target_k,
            lake_outflow_wb_id=self._lake_outflow_wb_id,
            wb_to_wb_src_wb=self._wb_to_wb_src_wb,
            wb_to_wb_tgt_wb=self._wb_to_wb_tgt_wb,
            wb_terminal_wb_ids=self._wb_terminal_wb_ids,
        )
        ret_coupling: RetentionCoupling = RetentionCoupling(
            storage_m3=retention_storage_m3,
            inflow_m3=self._retention_inflow_perm,
            outflow_m3=self._retention_outflow_perm,
            max_storage_m3=self.retention_max_storage_m3,
            inertial_retention_k=self._inertial_retention_k,
            inertial_retention_id=self._inertial_retention_id,
            controlled_retention=self.controlled_retention,
            activation_threshold_m3_s=retention_activation_threshold_m3_s,
            release_threshold_factor=self.retention_basin_release_threshold_factor,
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
            diagnostics,
        ) = self._step(
            routing_timestep_s=self.dt,
            previous_discharge_m3_s=self._discharge_prev_perm,
            river_storage_m3=self._river_storage_perm,
            sideflow_m3=self._sideflow_perm,
            evaporation_m3=self._evaporation_perm,
            kin=self._kinematic_topology,
            n_inertial=self.n_inertial,
            over_abstraction_m3=self._over_abstraction_perm,
            actual_evaporation_m3=self._actual_evaporation_perm,
            updated_discharge_m3_s=self._updated_discharge_perm,
            geom=self._geometry,
            topo=self._topology,
            wb=wb_coupling,
            ret=ret_coupling,
            buffers=self._substep_buffers,
        )

        # if self.n_inertial > 0 and diagnostics.limiting_reach_idx >= 0:
        #     self._print_limiting_substep(diagnostics)

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

        # Add terminal waterbody releases outside the domain to pit outflows
        # so total domain boundary loss is conserved in the overall water balance.
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
