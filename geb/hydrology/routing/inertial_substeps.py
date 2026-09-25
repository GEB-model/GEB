"""Substepping kernel for 1D local inertial dynamic river routing.

Solves the de Saint-Venant momentum and continuity equations using an adaptive
substepping scheme with analytical cross-sectional channel and floodplain geometry.
"""

import numpy as np
from numba import njit, prange

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    TwoDArrayFloat32,
)

from .geometry import (
    compute_cfl_area_and_top_width,
    compute_cross_section_from_depth,
    compute_overbank_depth,
)

__all__ = [
    "ALLOW_REVERSE_FLOW",
    "compute_retention_routing",
    "GEOM_CFL_CONSTANT",
    "GEOM_CFL_MANNING_COEFFICIENT",
    "GEOM_CFL_NUM_COLS",
    "GEOM_IN_BANKFULL_DEPTH",
    "GEOM_IN_BANKFULL_VOLUME",
    "GEOM_IN_BED_ELEVATION",
    "GEOM_IN_INTERFACE_BED_ELEVATION_MAX",
    "GEOM_IN_INVERSE_INTERFACE_LENGTH",
    "GEOM_IN_INVERSE_LENGTH",
    "GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE",
    "GEOM_IN_MANNING_N_SQUARED",
    "GEOM_IN_NUM_COLS",
    "GEOM_IN_STAGE_VOLUME_COEFFICIENT",
    "GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH",
    "GEOM_OV_BANKFULL_PERIMETER",
    "GEOM_OV_FLOODPLAIN_AREA_THRESHOLD",
    "GEOM_OV_FLOODPLAIN_DEPTH_THRESHOLD",
    "GEOM_OV_FLOODPLAIN_SIDE_SLOPE",
    "GEOM_OV_FLOODPLAIN_WIDTH",
    "GEOM_OV_NUM_COLS",
    "GEOM_OV_RIVER_WIDTH",
    "GEOM_OV_SQRT_ONE_PLUS_FLOODPLAIN_SLOPE_SQUARED",
    "INERTIAL_PARALLEL_THRESHOLD",
    "_gather_inputs",
    "_run_inertial_substeps",
    "_run_inertial_substeps_serial",
    "_run_inertial_substeps_parallel",
    "_compute_stage_from_storage_inertial",
    "_compute_reach_stage_from_vol",
    "_scatter_outputs",
    "_solve_inertial_momentum",
    "_evaluate_reach_cfl_dt",
    "compute_inertial_substeps_cfl",
]

# Global option to toggle reverse flow across river reaches
ALLOW_REVERSE_FLOW: bool = True

# Column indices for in-bank channel geometry (geom_inbank)
GEOM_IN_INVERSE_LENGTH: int = 0
GEOM_IN_BED_ELEVATION: int = 1
GEOM_IN_BANKFULL_DEPTH: int = 2
GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE: int = 3
GEOM_IN_BANKFULL_VOLUME: int = 4
GEOM_IN_STAGE_VOLUME_COEFFICIENT: int = 5
GEOM_IN_INTERFACE_BED_ELEVATION_MAX: int = 6
GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH: int = 7
GEOM_IN_MANNING_N_SQUARED: int = 8
GEOM_IN_INVERSE_INTERFACE_LENGTH: int = 9
GEOM_IN_NUM_COLS: int = 10

# Column indices for overbank floodplain geometry (geom_overbank)
GEOM_OV_FLOODPLAIN_WIDTH: int = 0
GEOM_OV_RIVER_WIDTH: int = 1
GEOM_OV_BANKFULL_PERIMETER: int = 2
GEOM_OV_FLOODPLAIN_SIDE_SLOPE: int = 3
GEOM_OV_FLOODPLAIN_DEPTH_THRESHOLD: int = 4
GEOM_OV_FLOODPLAIN_AREA_THRESHOLD: int = 5
GEOM_OV_SQRT_ONE_PLUS_FLOODPLAIN_SLOPE_SQUARED: int = 6
GEOM_OV_NUM_COLS: int = 7

# Column indices for CFL stability parameters (geom_cfl)
GEOM_CFL_MANNING_COEFFICIENT: int = 0
GEOM_CFL_CONSTANT: int = 1
GEOM_CFL_NUM_COLS: int = 2

# Threshold of reach count above which parallel multi-threading outperforms serial execution
INERTIAL_PARALLEL_THRESHOLD: int = 5000


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
    for i in prange(n):  # ty: ignore[not-iterable]
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
    for i in prange(n):  # ty: ignore[not-iterable]
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
        floodplain_side_slope=geom_overbank[reach_idx, GEOM_OV_FLOODPLAIN_SIDE_SLOPE],
        floodplain_depth_threshold_m=geom_overbank[
            reach_idx, GEOM_OV_FLOODPLAIN_DEPTH_THRESHOLD
        ],
        floodplain_area_threshold_m2=geom_overbank[
            reach_idx, GEOM_OV_FLOODPLAIN_AREA_THRESHOLD
        ],
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
    for reach_idx in range(n_inertial):
        reach_volume_m3: np.float32 = max(
            np.float32(river_storage_m3_inertial[reach_idx]), np.float32(0.0)
        )
        stage_buf[reach_idx] = _compute_reach_stage_from_vol(
            reach_idx, reach_volume_m3, geom_inbank, geom_overbank
        )


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

    Numerical stability is dictated by shallow-water wave celerity:
        dt_stable <= CFL * dx / wave_celerity
    where:
        - wave_celerity c = sqrt(g * hydraulic_depth)

    Notes:
        Wave celerity (c) is the propagation speed of a surface wave or disturbance
        relative to the moving water itself (distinct from the bulk flow velocity
        v = Q / A). In the local inertial approximation (Bates et al., 2010),
        the advective acceleration term (v * dv/dx) is omitted from the momentum equation,
        so information propagation and numerical stability are governed strictly by
        the gravity wave celerity c = sqrt(g * h).

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
    # A completely dry reach with no oncoming flow cannot propagate waves;
    # advancing by the full macro-timestep is safe.
    if current_volume_m3 <= 0.0 and total_flow_rate <= np.float32(0.0):
        return dt_f32

    # 1. Hydraulic depth from stored water volume:
    # We compute the wetted cross-sectional area (A) and surface top width (W)
    # corresponding to the water volume currently sitting in this reach (V = A * L).
    # The hydraulic depth is h = A / W.
    hydraulic_depth_from_volume: np.float32 = np.float32(0.0)
    wetted_cross_sectional_area_m2: np.float32 = np.float32(0.0)
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
                floodplain_side_slope=geom_overbank[
                    reach_idx, GEOM_OV_FLOODPLAIN_SIDE_SLOPE
                ],
                floodplain_depth_threshold_m=geom_overbank[
                    reach_idx, GEOM_OV_FLOODPLAIN_DEPTH_THRESHOLD
                ],
                floodplain_area_threshold_m2=geom_overbank[
                    reach_idx, GEOM_OV_FLOODPLAIN_AREA_THRESHOLD
                ],
            )
        )
        hydraulic_depth_from_volume = wetted_cross_sectional_area_m2 / max(
            water_top_width_m, np.float32(1e-3)
        )

    # 2. Anticipatory depth from oncoming flow (Manning normal depth):
    # If a nearly dry reach suddenly receives a fast-moving flood wave (Q > 0 while V ~ 0),
    # depth based on stored volume alone would be zero, which would cause instability.
    # Therefore, we estimate expected flow depth from Manning normal depth (h_normal ~ Q^0.6).
    hydraulic_depth_from_flow: np.float32 = np.float32(0.0)
    if total_flow_rate > np.float32(0.0):
        raw_depth_from_flow: np.float32 = geom_cfl[
            reach_idx, GEOM_CFL_MANNING_COEFFICIENT
        ] * (total_flow_rate ** np.float32(0.6))
        bankfull_depth_m: np.float32 = geom_inbank[reach_idx, GEOM_IN_BANKFULL_DEPTH]
        if raw_depth_from_flow > bankfull_depth_m and bankfull_depth_m > np.float32(
            0.0
        ):
            # When flow exceeds bankfull, water spills across the much wider floodplain.
            # Scale the excess depth by the width expansion ratio so low-slope reaches
            # do not predict absurdly deep narrow-slot depths.
            extra_depth: np.float32 = raw_depth_from_flow - bankfull_depth_m
            bf_width: np.float32 = geom_overbank[reach_idx, GEOM_OV_RIVER_WIDTH]
            fp_width: np.float32 = geom_overbank[reach_idx, GEOM_OV_FLOODPLAIN_WIDTH]
            width_ratio: np.float32 = bf_width / max(bf_width + fp_width, bf_width)
            hydraulic_depth_from_flow = bankfull_depth_m + (extra_depth * width_ratio)
        else:
            hydraulic_depth_from_flow = raw_depth_from_flow

    # 3. Take the governing (strictest/deepest) hydraulic depth:
    hydraulic_depth: np.float32 = max(
        hydraulic_depth_from_volume, hydraulic_depth_from_flow
    )

    if hydraulic_depth <= np.float32(0.0):
        # Dry channel: no waves can propagate, so full macro timestep is safe
        return dt_f32

    # 4. Compute shallow-water wave celerity and maximum stable timestep:
    # Wave celerity c = sqrt(g * h) is the speed at which a surface gravity wave disturbance
    # propagates relative to the water itself (in contrast to flow velocity v = Q / A).
    # In local inertial routing, this celerity governs how fast pressure/depth signals travel.
    wave_celerity: np.float32 = np.sqrt(gravity_acceleration * hydraulic_depth)

    # dt_max = CFL * dx / wave_celerity
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
    min_dt_buf: ArrayFloat32,
    kinematic_inflow_rate_inertial: ArrayFloat32,
    lake_outflow_target_k: ArrayInt32,
    lake_outflow_wb_id: ArrayInt32,
    outflow_per_waterbody_m3: ArrayFloat32,
    waterbody_storage_m3: ArrayFloat64,
    wb_lake_area: ArrayFloat32,
    wb_lake_factor: ArrayFloat32,
    wb_outflow_height: ArrayFloat32,
    wb_outflow_bed_elev: ArrayFloat32,
    inertial_topo_order: ArrayInt32,
    total_flow_rate_buf: ArrayFloat32,
) -> int:
    """Computes upstream kinematic inflow accumulations and adaptive CFL substepping stability constraint.

    Operates in parallel across available CPU threads without lock contention.

    Args:
        dt_f32: Macro routing timestep duration (seconds).
        inv_dt_f32: Reciprocal of macro routing timestep (1/seconds).
        n_inertial: Number of reaches routed with local inertial equations.
        inertial_up_kin_offsets: CSR offsets of upstream kinematic nodes per inertial reach.
        inertial_up_kin_indices: CSR indices of upstream kinematic nodes per inertial reach.
        inertial_up_kin_reach_idx: Precomputed direct upstream kinematic node index array.
        inertial_up_offsets: CSR offsets of upstream inertial reaches per reach.
        inertial_up_indices: CSR indices of upstream inertial reaches per reach.
        inertial_up_reach_idx: Precomputed direct upstream reach index array.
        previous_discharge_m3_s_inertial: Inertial reach discharge from previous timestep (m³/s).
        updated_discharge_m3_s: Discharge array updated from kinematic reaches (m³/s).
        river_storage_m3_inertial: Inertial reach water storage volume array (m³).
        sideflow_m3_inertial: Inertial reach lateral sideflow volume array (m³).
        geom_inbank: Compact 2D in-bank reach geometry array (meters, dimensionless).
        geom_overbank: Secondary 2D overbank floodplain geometry array (meters, dimensionless).
        geom_cfl: 2D array of CFL stability condition parameters.
        min_dt_buf: Pre-allocated workspace buffer storing per-reach stable timestep (seconds).
        kinematic_inflow_rate_inertial: Output array populated with upstream kinematic inflow rates (m³/s).
        lake_outflow_target_k: Target reach index receiving waterbody outflows.
        lake_outflow_wb_id: Source waterbody ID releasing outflow into target reaches.
        outflow_per_waterbody_m3: Prescribed outflow volume per waterbody (m³).
        waterbody_storage_m3: Waterbody storage volume array (m³).
        wb_lake_area: Surface area of each waterbody (m²).
        wb_lake_factor: Rating curve multiplier factor for weir outflow calculation.
        wb_outflow_height: Weir sill elevation above channel bed (meters).
        wb_outflow_bed_elev: Bed elevation at waterbody outlet (meters).
        inertial_topo_order: Topological traversal order of inertial reaches.
        total_flow_rate_buf: Pre-allocated workspace buffer storing peak flow rate per reach (m³/s).

    Returns:
        num_inertial_substeps: Total number of sub-timesteps required for stability.
    """
    kinematic_inflow_rate_inertial.fill(np.float32(0.0))
    total_flow_rate_buf.fill(np.float32(0.0))

    wb_inflow_rate: ArrayFloat32 = np.zeros(n_inertial, dtype=np.float32)
    n_lake_links: int = len(lake_outflow_target_k)
    if n_lake_links > 0:
        for link_idx in range(n_lake_links):
            wb_id: int = lake_outflow_wb_id[link_idx]
            target_reach: int = lake_outflow_target_k[link_idx]
            target_outflow: float = outflow_per_waterbody_m3[wb_id]

            # when waterbody outflow is determined outside the routing,
            # such as by reservoir operations
            if not np.isnan(target_outflow):
                wb_inflow_rate[target_reach] += np.float32(target_outflow) * inv_dt_f32
            else:
                lake_area: np.float32 = wb_lake_area[wb_id]
                depth_from_bottom: np.float32 = (
                    np.float32(waterbody_storage_m3[wb_id]) / lake_area
                )
                head_above_sill: np.float32 = max(
                    depth_from_bottom - wb_outflow_height[wb_id],
                    np.float32(0.0),
                )
                if head_above_sill > np.float32(1e-5):
                    lake_factor: np.float32 = wb_lake_factor[wb_id]
                    wb_inflow_rate[target_reach] += lake_factor * (
                        head_above_sill * head_above_sill
                    )

    gravity_acceleration: np.float32 = np.float32(9.80665)

    for reach_idx in range(n_inertial):
        # Accumulate inflow from upstream kinematic reaches.
        # Topology encoding:
        #   >= 0: Exactly one upstream tributary (fast-path direct index)
        #   == -1: No upstream kinematic tributary (headwater or inertial-only)
        #   < -1: Multi-river confluence (encoded CSR start index: -reach - 2)
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
        upstream_inertial_reach: int = inertial_up_reach_idx[reach_idx]
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
            end_inertial_idx: int = inertial_up_offsets[reach_idx + 1]
            for idx in range(start_inertial_idx, end_inertial_idx):
                upstream_junction_reach: int = inertial_up_indices[idx]
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
        total_flow_rate_buf[reach_idx] = total_flow_rate
        current_volume_m3: float = river_storage_m3_inertial[reach_idx]

        # Calculate the maximum CFL-stable timestep duration for this specific reach
        min_dt_buf[reach_idx] = _evaluate_reach_cfl_dt(
            reach_idx=reach_idx,
            total_flow_rate=total_flow_rate,
            current_volume_m3=current_volume_m3,
            geom_inbank=geom_inbank,
            geom_overbank=geom_overbank,
            geom_cfl=geom_cfl,
            gravity_acceleration=gravity_acceleration,
            dt_f32=dt_f32,
        )

    # Find the most restrictive reach in the entire domain and compute the required sub-steps
    limiting_reach_idx: int = int(np.argmin(min_dt_buf))
    min_stable_timestep_s: np.float32 = min_dt_buf[limiting_reach_idx]
    raw_substeps: int = int(np.ceil(dt_f32 / min_stable_timestep_s))
    num_inertial_substeps: int = max(1, raw_substeps)

    return num_inertial_substeps


@njit(inline="always")
def _solve_inertial_momentum(
    reach_idx: int,
    effective_depth: np.float32,
    water_slope: np.float32,
    curr_discharge: np.float32,
    min_wet_depth_m: np.float32,
    geom_inbank: TwoDArrayFloat32,
    geom_overbank: TwoDArrayFloat32,
    g_dt_substep: np.float32,
    sqrt_gravity: np.float32,
    can_reverse: bool,
    river_storage_m3_inertial: ArrayFloat64,
    inv_dt_substep: np.float32,
) -> np.float32:
    """Evaluates local inertial momentum update for a single reach using two-tier geometry.

    Fast-paths in-bank flow to read only the compact single-cache-line geom_inbank row,
    completely bypassing secondary floodplain parameters when unflooded.

    Args:
        reach_idx: Reach index within local inertial domain.
        effective_depth: Flow depth at the interface (meters).
        water_slope: Longitudinal water surface slope dz/dx (dimensionless).
        curr_discharge: Discharge at start of substep (m³/s).
        min_wet_depth_m: Minimum depth threshold below which reach is treated as dry (meters).
        geom_inbank: Compact 2D in-bank reach geometry array (meters, dimensionless).
        geom_overbank: Secondary 2D overbank floodplain geometry array (meters, dimensionless).
        g_dt_substep: Precalculated g * dt scalar.
        sqrt_gravity: Square root of gravitational acceleration (m^(1/2)/s).
        can_reverse: Boolean flag indicating whether reverse flow is permitted.
        river_storage_m3_inertial: Current reach storage volume array (m³).
        inv_dt_substep: Reciprocal of substep timestep (1/seconds).

    Returns:
        Computed instantaneous discharge at end of substep (m³/s).
    """
    if effective_depth < min_wet_depth_m:
        return np.float32(0.0)

    bankfull_depth_m: np.float32 = geom_inbank[reach_idx, GEOM_IN_BANKFULL_DEPTH]
    cross_sectional_area: np.float32 = np.float32(0.0)
    interface_top_width: np.float32 = np.float32(0.0)
    interface_p_wetted: np.float32 = np.float32(0.0)

    inv_shape_exponent_plus_one: np.float32 = geom_inbank[
        reach_idx, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE
    ]

    if effective_depth <= bankfull_depth_m or bankfull_depth_m <= np.float32(0.0):
        depth: np.float32 = effective_depth
        # Parabolic channel (shape exponent r = 0.5)
        if inv_shape_exponent_plus_one == np.float32(1.0 / 1.5):
            interface_top_width = max(
                np.sqrt(depth)
                * geom_inbank[reach_idx, GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH],
                np.float32(1e-3),
            )
        else:
            shape_exponent: np.float32 = (
                np.float32(1.0) / inv_shape_exponent_plus_one
            ) - np.float32(1.0)
            depth_ratio: np.float32 = depth / max(bankfull_depth_m, np.float32(1e-4))
            interface_top_width = max(
                geom_overbank[reach_idx, GEOM_OV_RIVER_WIDTH]
                * (depth_ratio**shape_exponent),
                np.float32(1e-3),
            )
        cross_sectional_area = interface_top_width * depth * inv_shape_exponent_plus_one
        interface_p_wetted = (
            interface_top_width
            + np.float32(8.0 / 3.0) * (depth * depth) / interface_top_width
        )
    else:
        shape_exponent = (
            np.float32(0.5)
            if inv_shape_exponent_plus_one == np.float32(1.0 / 1.5)
            else (np.float32(1.0) / inv_shape_exponent_plus_one) - np.float32(1.0)
        )
        bankfull_area: np.float32 = (
            geom_inbank[reach_idx, GEOM_IN_BANKFULL_VOLUME]
            * geom_inbank[reach_idx, GEOM_IN_INVERSE_LENGTH]
        )
        (
            cross_sectional_area,
            interface_top_width,
            interface_p_wetted,
        ) = compute_cross_section_from_depth(
            effective_depth_m=effective_depth,
            bankfull_width_m=geom_overbank[reach_idx, GEOM_OV_RIVER_WIDTH],
            shape_exponent=shape_exponent,
            bankfull_depth_m=bankfull_depth_m,
            floodplain_width_m=geom_overbank[reach_idx, GEOM_OV_FLOODPLAIN_WIDTH],
            inverse_shape_exponent_plus_one=inv_shape_exponent_plus_one,
            bankfull_area_m2=bankfull_area,
            bankfull_perimeter_m=geom_overbank[reach_idx, GEOM_OV_BANKFULL_PERIMETER],
            floodplain_side_slope=geom_overbank[
                reach_idx, GEOM_OV_FLOODPLAIN_SIDE_SLOPE
            ],
            floodplain_depth_threshold_m=geom_overbank[
                reach_idx, GEOM_OV_FLOODPLAIN_DEPTH_THRESHOLD
            ],
            floodplain_area_threshold_m2=geom_overbank[
                reach_idx, GEOM_OV_FLOODPLAIN_AREA_THRESHOLD
            ],
            sqrt_one_plus_floodplain_slope_squared=geom_overbank[
                reach_idx, GEOM_OV_SQRT_ONE_PLUS_FLOODPLAIN_SLOPE_SQUARED
            ],
            width_over_sqrt_bankfull_depth=geom_inbank[
                reach_idx, GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH
            ],
        )

    # Manning friction term
    hydraulic_radius_term: np.float32 = (
        cross_sectional_area / max(interface_p_wetted, np.float32(1e-6))
    ) ** np.float32(4.0 / 3.0)
    friction_denom_term: np.float32 = max(
        cross_sectional_area * hydraulic_radius_term, np.float32(1e-12)
    )

    incoming_discharge: np.float32 = curr_discharge
    gravity_dt_manning_n_sq: np.float32 = (
        g_dt_substep * geom_inbank[reach_idx, GEOM_IN_MANNING_N_SQUARED]
    )

    # Predictor step: intermediate frictionless gravity-accelerated flow.
    # Calculates how much gravity accelerates (or decelerates) flow along the water surface slope.
    intermediate_discharge: np.float32 = (
        incoming_discharge - g_dt_substep * cross_sectional_area * water_slope
    )

    # Apply implicit Manning bed friction damping.
    effective_friction_q: np.float32 = abs(incoming_discharge)
    friction_denom: np.float32 = np.float32(1.0) + (
        gravity_dt_manning_n_sq * effective_friction_q / friction_denom_term
    )
    computed_discharge: np.float32 = intermediate_discharge / friction_denom

    # Physical ceiling: critical flow limiter (Froude number Fr <= 1).
    # Prevents unphysical hyper-supercritical runaway over steep drops.
    hydraulic_depth: np.float32 = cross_sectional_area / max(
        interface_top_width, np.float32(1e-3)
    )
    critical_discharge: np.float32 = (
        sqrt_gravity * cross_sectional_area * np.sqrt(hydraulic_depth)
    )

    # Storage availability and directional discharge capping:
    if computed_discharge >= np.float32(0.0):
        available_storage_m3: float = river_storage_m3_inertial[reach_idx]
        if available_storage_m3 > 0.0:
            max_drain_rate: np.float32 = (
                np.float32(0.95) * np.float32(available_storage_m3) * inv_dt_substep
            )
            return min(computed_discharge, critical_discharge, max_drain_rate)
        return np.float32(0.0)

    # Reverse flow is only allowed for internal reaches, limited by critical discharge
    if not can_reverse:
        return np.float32(0.0)

    return -min(-computed_discharge, critical_discharge)


def _run_inertial_substeps(
    num_inertial_substeps: int,
    dt_f32: np.float32,
    n_inertial: int,
    substep_discharge_m3_s: ArrayFloat32,
    river_storage_m3_inertial: ArrayFloat64,
    sideflow_m3_inertial: ArrayFloat32,
    evaporation_m3_inertial: ArrayFloat32,
    waterbody_storage_m3: ArrayFloat64,
    outflow_per_waterbody_m3: ArrayFloat32,
    geom_inbank: TwoDArrayFloat32,
    geom_overbank: TwoDArrayFloat32,
    retention_storage_m3: ArrayFloat32,
    retention_max_storage_m3: ArrayFloat32,
    inertial_retention_k: ArrayInt32,
    inertial_retention_id: ArrayInt32,
    controlled_retention: ArrayBool,
    retention_activation_threshold_m3_s: ArrayFloat32,
    retention_basin_release_threshold_factor: np.float32,
    pit_slope_inertial: ArrayFloat32,
    over_abstraction_m3_inertial: ArrayFloat32,
    actual_evaporation_m3_inertial: ArrayFloat32,
    waterbody_inflow_m3: ArrayFloat32,
    retention_inflow_m3: ArrayFloat32,
    retention_outflow_m3: ArrayFloat32,
    updated_discharge_m3_s_inertial: ArrayFloat32,
    kinematic_inflow_rate: ArrayFloat32,
    discharge_vol_sum_m3_inertial: ArrayFloat32,
    wb_outflow_substep_m3_buf: ArrayFloat64,
    retention_inflow_limit_sub_buf: ArrayFloat64,
    retention_max_outflow_limit_sub_buf: ArrayFloat64,
    wb_outflow_avail_buf: ArrayFloat64,
    wb_extra_lateral_accum_m3: ArrayFloat64,
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
    inertial_up_offsets: ArrayInt32,
    inertial_up_indices: ArrayInt32,
    inertial_up_reach_idx: ArrayInt32,
    inertial_wb_ds_k: ArrayInt32,
    inertial_wb_ds_id: ArrayInt32,
    lake_outflow_target_k: ArrayInt32,
    lake_outflow_wb_id: ArrayInt32,
    geom_cfl: TwoDArrayFloat32,
    min_dt_buf: ArrayFloat32,
    rev_demand_buf: ArrayFloat32,
    wb_release_volume_substep: ArrayFloat32,
    inertial_topo_order: ArrayInt32,
    total_flow_rate_buf: ArrayFloat32,
) -> int:
    """Advances dynamic local inertial routing across adaptive substeps.

    Operates using a 3-stage architecture per substep:
    - Loop 1: Momentum Equation (evaluates instantaneous flux Q across all reaches in parallel)
    - Loop 2: Continuity & Stage Update (updates storage and water surface elevation in parallel)
    - Loop 3: Dynamic Stability (evaluates adaptive CFL stability in topological order with all boundary fluxes)

    Args:
        num_inertial_substeps: Initial number of sub-timesteps predicted by macro CFL.
        dt_f32: Macro routing timestep duration (seconds).
        n_inertial: Number of reaches routed with local inertial equations.
        substep_discharge_m3_s: Primary discharge array (m³/s).
        river_storage_m3_inertial: Primary reach water storage volume array (m³).
        sideflow_m3_inertial: Macro-timestep lateral sideflow volume (m³).
        evaporation_m3_inertial: Macro-timestep potential channel evaporation volume (m³).
        waterbody_storage_m3: Global waterbody storage volume array (m³).
        outflow_per_waterbody_m3: Fixed prescribed waterbody outflow volume array (m³).
        geom_inbank: Compact 2D in-bank reach geometry array (meters, dimensionless).
        geom_overbank: Secondary 2D overbank floodplain geometry array (meters, dimensionless).
        retention_storage_m3: Global retention basin storage array (m³).
        retention_max_storage_m3: Maximum storage capacity per retention basin (m³).
        inertial_retention_k: Sparse reach indices of retention basins.
        inertial_retention_id: Retention basin identifiers corresponding to sparse reaches.
        controlled_retention: Flag defining controlled vs uncontrolled retention dynamics.
        retention_activation_threshold_m3_s: Discharge threshold triggering retention basin diversion (m³/s).
        retention_basin_release_threshold_factor: Release threshold factor for retention basins.
        pit_slope_inertial: Effective slope for terminal inland pit reaches (dimensionless).
        over_abstraction_m3_inertial: Output array accumulating unmet extraction deficit (m³).
        actual_evaporation_m3_inertial: Output array accumulating realized channel evaporation (m³).
        waterbody_inflow_m3: Global array accumulating inflow into waterbodies (m³).
        retention_inflow_m3: Global array accumulating diversion volume into retention basins (m³).
        retention_outflow_m3: Global array accumulating release volume from retention basins (m³).
        updated_discharge_m3_s_inertial: Output array storing timestep-averaged reach discharge (m³/s).
        kinematic_inflow_rate: Upstream inflow rates from kinematic overland/headwater domain (m³/s).
        discharge_vol_sum_m3_inertial: Accumulator buffer summing volumetric outflow over substeps (m³).
        wb_outflow_substep_m3_buf: Allocated buffer for fixed waterbody outflow volume per substep (m³).
        retention_inflow_limit_sub_buf: Substep maximum intake diversion capacity (m³).
        retention_max_outflow_limit_sub_buf: Substep maximum release capacity (m³).
        wb_outflow_avail_buf: Temporary buffer holding available waterbody release volume (m³).
        wb_extra_lateral_accum_m3: Accumulator tracking total waterbody volume transferred to reaches (m³).
        wb_lake_area: Surface area of each waterbody (m²).
        wb_lake_factor: Rating curve multiplier factor for weir outflow calculation.
        wb_outflow_height: Weir sill elevation above channel bed (meters).
        wb_outflow_bed_elev: Bed elevation at waterbody outlet (meters).
        stage_buf: Workspace buffer storing reach and lake water surface elevations (meters).
        ds_boundary_type: Boundary condition classification for downstream connections.
        ds_inertial_k: Downstream reach index within inertial domain (-1 if boundary).
        ds_stage_idx: Unified downstream stage lookup index for internal and lake boundaries.
        ds_bed_elevation: Downstream bed elevation for external boundaries (meters).
        kin_ds_slope: Channel slope for reaches draining into kinematic domain (dimensionless).
        inertial_up_offsets: CSR offsets of upstream inertial reaches per reach.
        inertial_up_indices: CSR indices of upstream inertial reaches per reach.
        inertial_up_reach_idx: Precomputed direct upstream reach index array.
        inertial_wb_ds_k: Reach indices that terminate into downstream waterbodies.
        inertial_wb_ds_id: Waterbody identifier for each terminal reach.
        lake_outflow_target_k: Target reach index receiving waterbody outflows.
        lake_outflow_wb_id: Source waterbody ID releasing outflow into target reaches.
        geom_cfl: CFL hydraulic geometry lookup matrix.
        min_dt_buf: Buffer storing CFL minimum stable timestep per reach (seconds).
        rev_demand_buf: Workspace buffer accumulating reverse flow demands per cell (m³/s).
        wb_release_volume_substep: Workspace buffer buffering substep waterbody release volume into reaches (m³).
        inertial_topo_order: Topological traversal order of inertial reaches.
        total_flow_rate_buf: Workspace buffer propagating peak characteristic flow rate across reaches (m³/s).

    Returns:
        Total number of internal adaptive substeps executed.
    """
    gravity_acceleration: np.float32 = np.float32(9.80665)
    sqrt_gravity: np.float32 = np.sqrt(gravity_acceleration)
    min_wet_depth_m: np.float32 = np.float32(1e-4)

    discharge_vol_sum_m3_inertial.fill(0.0)
    wb_extra_lateral_accum_m3.fill(0.0)

    n_wb: int = outflow_per_waterbody_m3.size
    n_ret: int = retention_max_storage_m3.size
    n_lake_links: int = lake_outflow_target_k.size
    n_ret_inertial: int = len(inertial_retention_k)
    n_wb_ds: int = len(inertial_wb_ds_k)

    # Initialize reach water stages from current storage to ensure physical consistency
    _compute_stage_from_storage_inertial(
        n_inertial, river_storage_m3_inertial, geom_inbank, geom_overbank, stage_buf
    )

    min_dt_floor: np.float32 = min(np.float32(0.1), dt_f32)
    dt_substep: np.float32 = max(
        dt_f32 / np.float32(num_inertial_substeps), min_dt_floor
    )

    t_elapsed: np.float32 = np.float32(0.0)
    substep_count: int = 0

    while t_elapsed < dt_f32:
        time_remaining: np.float32 = dt_f32 - t_elapsed
        if time_remaining <= np.float32(1e-4):
            break

        if dt_substep >= time_remaining or (time_remaining - dt_substep < min_dt_floor):
            dt_substep = time_remaining

        inv_dt_substep: np.float32 = np.float32(1.0) / dt_substep
        g_dt_substep: np.float32 = gravity_acceleration * dt_substep
        dt_substep_f64: float = float(dt_substep)
        substep_fraction_f32: np.float32 = dt_substep / dt_f32
        substep_fraction_f64: float = float(substep_fraction_f32)

        for i in range(n_ret):
            ret_in_lim: np.float32 = np.float32(0.20) * retention_max_storage_m3[i]
            ret_out_lim: np.float32 = np.float32(0.05) * retention_max_storage_m3[i]
            retention_inflow_limit_sub_buf[i] = (
                np.float64(ret_in_lim) * substep_fraction_f64
            )
            retention_max_outflow_limit_sub_buf[i] = (
                np.float64(ret_out_lim) * substep_fraction_f64
            )

        wb_release_volume_substep.fill(0.0)

        # Update waterbody water levels and calculate continuous releases
        if n_wb > 0:
            for wb_id in range(n_wb):
                lake_area: np.float32 = wb_lake_area[wb_id]
                depth_from_bottom: np.float32 = (
                    np.float32(waterbody_storage_m3[wb_id]) / lake_area
                )
                bottom_elevation: np.float32 = (
                    wb_outflow_bed_elev[wb_id] - wb_outflow_height[wb_id]
                )
                wb_stage: np.float32 = bottom_elevation + depth_from_bottom
                stage_buf[n_inertial + wb_id] = wb_stage

                target_outflow: float = (
                    float(outflow_per_waterbody_m3[wb_id]) * substep_fraction_f64
                )
                # for waterbodies without pre-defined outflow
                if np.isnan(target_outflow):
                    head_above_sill: np.float32 = max(
                        depth_from_bottom - wb_outflow_height[wb_id],
                        np.float32(0.0),
                    )
                    if head_above_sill > np.float32(1e-5):
                        lake_factor: np.float32 = wb_lake_factor[wb_id]
                        outflow_rate_m3_s: np.float32 = lake_factor * (
                            head_above_sill * head_above_sill
                        )
                        lake_outflow_vol: float = min(
                            float(outflow_rate_m3_s * dt_substep),
                            float(head_above_sill * lake_area),
                        )
                        wb_outflow_avail_buf[wb_id] = lake_outflow_vol
                    else:
                        wb_outflow_avail_buf[wb_id] = 0.0
                else:  # for waterbodies with predefined outflow. Likely reservoirs.
                    wb_outflow_avail_buf[wb_id] = (
                        min(target_outflow, float(waterbody_storage_m3[wb_id]))
                        if waterbody_storage_m3[wb_id] > 0.0
                        else 0.0
                    )

            # Buffer waterbody releases into reaches as continuous boundary flux
            if n_lake_links > 0:
                for link_idx in range(n_lake_links):
                    wb_id_node: int = lake_outflow_wb_id[link_idx]
                    release: float = wb_outflow_avail_buf[wb_id_node]
                    if release > 0.0:
                        target_reach: int = lake_outflow_target_k[link_idx]
                        wb_release_volume_substep[target_reach] += np.float32(release)
                        wb_extra_lateral_accum_m3[target_reach] += release
                        wb_outflow_avail_buf[wb_id_node] = 0.0
                        waterbody_storage_m3[wb_id_node] -= release

        # Loop 1: Momentum Equation
        for reach_idx in prange(n_inertial):  # ty: ignore[not-iterable]
            boundary_type: int = ds_boundary_type[reach_idx]
            water_stage_node: np.float32 = stage_buf[reach_idx]
            inv_interface_len: np.float32 = geom_inbank[
                reach_idx, GEOM_IN_INVERSE_INTERFACE_LENGTH
            ]

            effective_depth: np.float32 = np.float32(0.0)
            water_slope: np.float32 = np.float32(0.0)

            reach_can_reverse: bool = False
            if boundary_type == 0 or boundary_type == 2:
                # Internal reach or lake boundary
                ds_idx: int = ds_stage_idx[reach_idx]
                water_stage_ds: np.float32 = stage_buf[ds_idx]
                max_bed: np.float32 = geom_inbank[
                    reach_idx, GEOM_IN_INTERFACE_BED_ELEVATION_MAX
                ]
                bed_elev_node: np.float32 = geom_inbank[
                    reach_idx, GEOM_IN_BED_ELEVATION
                ]
                if bed_elev_node >= max_bed and water_stage_ds <= bed_elev_node:
                    # Steep drop / free overfall where downstream water level is below upstream bed:
                    # By Bates' formulation, the flow depth is strictly upstream depth,
                    # water slope is -depth / dx, and reverse flow across the drop is impossible.
                    effective_depth = max(
                        water_stage_node - bed_elev_node, np.float32(0.0)
                    )
                    water_slope = -effective_depth * inv_interface_len
                    reach_can_reverse = False
                else:
                    # Submerged or backwater interface condition (Bates et al., 2010):
                    # Flow depth is the difference between the maximum free-surface elevation
                    # and the highest bed elevation of the two adjoining cells.
                    effective_stage_node: np.float32 = max(water_stage_node, max_bed)
                    effective_stage_ds: np.float32 = max(water_stage_ds, max_bed)
                    max_stage: np.float32 = max(
                        effective_stage_node, effective_stage_ds
                    )
                    effective_depth = max_stage - max_bed
                    water_slope = (
                        effective_stage_ds - effective_stage_node
                    ) * inv_interface_len
                    reach_can_reverse = ALLOW_REVERSE_FLOW and boundary_type == 0
            elif boundary_type == 1:
                bed_elev_node = geom_inbank[reach_idx, GEOM_IN_BED_ELEVATION]
                effective_depth = max(water_stage_node - bed_elev_node, np.float32(0.0))
                water_slope = -kin_ds_slope[reach_idx]
            elif boundary_type == 3:
                bed_elev_node = geom_inbank[reach_idx, GEOM_IN_BED_ELEVATION]
                bed_elev_ds: np.float32 = ds_bed_elevation[reach_idx]
                max_bed = max(bed_elev_node, bed_elev_ds)
                effective_stage_node = max(water_stage_node, max_bed)
                effective_depth = effective_stage_node - max_bed
                water_slope = (bed_elev_ds - bed_elev_node) * inv_interface_len
            else:  # boundary_type == 4 (ocean pit)
                bed_elev_node = geom_inbank[reach_idx, GEOM_IN_BED_ELEVATION]
                effective_depth = max(water_stage_node - bed_elev_node, np.float32(0.0))
                sea_stage: np.float32 = np.float32(0.0)
                if bed_elev_node < sea_stage:
                    water_slope = (sea_stage - water_stage_node) * inv_interface_len
                else:
                    water_slope = -pit_slope_inertial[reach_idx]

            discharge: np.float32 = _solve_inertial_momentum(
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
            )
            updated_discharge_m3_s_inertial[reach_idx] = discharge

        # Multi-outflow storage depletion limiter for reverse flows.
        # Unlike downstream forward flow where each reach has only one outlet, multiple upstream
        # tributaries at a confluence can simultaneously pull reverse flow out of the same downstream
        # cell during backwater conditions. Simultaneously, the downstream cell may discharge forward.
        # To prevent over-draining downstream cells and creating non-physical mass, we sum all reverse
        # demands per downstream cell, subtract any forward outflow volume, and scale reverse flows
        # proportionally if the combined demand exceeds available storage.
        has_reverse_flow: bool = False
        rev_demand_buf.fill(0.0)
        for reach_idx in range(n_inertial):
            q_cand: np.float32 = updated_discharge_m3_s_inertial[reach_idx]
            if q_cand < np.float32(0.0):
                ds_reach: int = ds_inertial_k[reach_idx]
                if ds_reach >= 0:
                    rev_demand_buf[ds_reach] += -q_cand
                    has_reverse_flow = True

        # Scale reverse flows proportionally if total demand exceeds available storage
        if has_reverse_flow:
            for reach_idx in range(n_inertial):
                q_cand = updated_discharge_m3_s_inertial[reach_idx]
                if q_cand < np.float32(0.0):
                    ds_reach = ds_inertial_k[reach_idx]
                    if ds_reach >= 0:
                        tot_rev_demand: np.float32 = rev_demand_buf[ds_reach]
                        ds_storage: float = river_storage_m3_inertial[ds_reach]
                        q_ds: np.float32 = updated_discharge_m3_s_inertial[ds_reach]
                        ds_fwd_vol: float = (
                            float(q_ds) * dt_substep_f64
                            if q_ds > np.float32(0.0)
                            else 0.0
                        )
                        ds_avail_vol: float = max(ds_storage - ds_fwd_vol, 0.0)
                        if ds_avail_vol > 0.0 and tot_rev_demand > np.float32(0.0):
                            max_tot_rev_rate: np.float32 = (
                                np.float32(0.95)
                                * np.float32(ds_avail_vol)
                                * inv_dt_substep
                            )
                            if tot_rev_demand > max_tot_rev_rate:
                                scale: np.float32 = max_tot_rev_rate / tot_rev_demand
                                updated_discharge_m3_s_inertial[reach_idx] = (
                                    q_cand * scale
                                )
                        else:
                            updated_discharge_m3_s_inertial[reach_idx] = np.float32(0.0)

        substep_discharge_m3_s[:] = updated_discharge_m3_s_inertial[:]

        # Route diversions and releases for retention basins
        if n_ret_inertial > 0:
            for idx in range(n_ret_inertial):
                reach_idx = inertial_retention_k[idx]
                retention_id: int = inertial_retention_id[idx]
                discharge_positive: np.float32 = max(
                    substep_discharge_m3_s[reach_idx], np.float32(0.0)
                )
                sideflow_sub: np.float32 = (
                    sideflow_m3_inertial[reach_idx] * substep_fraction_f32
                )
                avail_flow_rate: np.float32 = (
                    sideflow_sub * inv_dt_substep
                ) + discharge_positive
                inflow_limit: float = retention_inflow_limit_sub_buf[retention_id]
                max_outflow_limit: float = retention_max_outflow_limit_sub_buf[
                    retention_id
                ]
                river_volume_substep: np.float32 = avail_flow_rate * dt_substep
                is_rising_limb: bool = bool(avail_flow_rate > discharge_positive)

                (
                    diverted_vol,
                    released_vol,
                    retention_storage_m3[retention_id],
                    river_volume_substep,
                ) = compute_retention_routing(
                    dt=dt_substep,
                    river_volume_m3=river_volume_substep,
                    discharge_before_diversion_m3_s=avail_flow_rate,
                    is_rising_limb=is_rising_limb,
                    retention_storage_m3=retention_storage_m3[retention_id],
                    retention_max_storage_m3=retention_max_storage_m3[retention_id],
                    controlled_retention=controlled_retention[retention_id],
                    activation_threshold_m3_s=retention_activation_threshold_m3_s[
                        retention_id
                    ],
                    release_threshold_factor=retention_basin_release_threshold_factor,
                    inflow_limit_m3=np.float32(inflow_limit),
                    max_outflow_limit_m3=np.float32(max_outflow_limit),
                )

                retention_inflow_m3[retention_id] += diverted_vol
                retention_outflow_m3[retention_id] += released_vol
                net_retention_vol: float = float(diverted_vol - released_vol)
                river_storage_m3_inertial[reach_idx] = max(
                    river_storage_m3_inertial[reach_idx] - net_retention_vol, 0.0
                )

        # Loop 2: Continuity, Storage & Water Stage Update
        for reach_idx in prange(n_inertial):  # ty: ignore[not-iterable]
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

            net_flow_volume: np.float32 = (
                inflow_rate - outflow_rate
            ) * dt_substep + wb_release_volume_substep[reach_idx]
            sideflow_volume: np.float32 = (
                sideflow_m3_inertial[reach_idx] * substep_fraction_f32
            )
            current_storage: float = river_storage_m3_inertial[reach_idx]

            if sideflow_volume < np.float32(0.0):
                abstraction_demand: np.float32 = -sideflow_volume
                available_for_abstraction: np.float32 = max(
                    np.float32(current_storage) + net_flow_volume, np.float32(0.0)
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
            discharge_vol_sum_m3_inertial[reach_idx] += outflow_rate * dt_substep

            # Calculate water stage from updated storage volume
            reach_volume: np.float32 = np.float32(new_storage)
            stage_buf[reach_idx] = _compute_reach_stage_from_vol(
                reach_idx, reach_volume, geom_inbank, geom_overbank
            )

        # Loop 3: Dynamic Stability & CFL Evaluation (1-Hop Local Boundaries)
        for reach_idx in range(n_inertial):
            upstream_reach_idx = inertial_up_reach_idx[reach_idx]
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

        # Accumulate inflows into downstream waterbodies
        if n_wb_ds > 0:
            for idx in range(n_wb_ds):
                reach_idx = inertial_wb_ds_k[idx]
                waterbody_id = inertial_wb_ds_id[idx]
                q_wb: np.float32 = substep_discharge_m3_s[reach_idx]
                if q_wb > np.float32(0.0):
                    inflow_vol: float = float(q_wb * dt_substep)
                    waterbody_inflow_m3[waterbody_id] += np.float32(inflow_vol)
                    waterbody_storage_m3[waterbody_id] += inflow_vol

        t_elapsed += dt_substep
        substep_count += 1

        if t_elapsed < dt_f32:
            min_dt_next: np.float32 = dt_f32
            for r_idx in range(n_inertial):
                if min_dt_buf[r_idx] < min_dt_next:
                    min_dt_next = min_dt_buf[r_idx]
            dt_substep = max(min_dt_next, min_dt_floor)

    # Calculate average discharge over the time step
    inv_macro_dt: np.float32 = np.float32(1.0) / dt_f32
    updated_discharge_m3_s_inertial[:] = discharge_vol_sum_m3_inertial * inv_macro_dt
    actual_evaporation_m3_inertial[:] = np.minimum(
        actual_evaporation_m3_inertial, evaporation_m3_inertial
    )

    return substep_count


_run_inertial_substeps_serial = njit(parallel=False)(_run_inertial_substeps)
_run_inertial_substeps_parallel = njit(parallel=True)(_run_inertial_substeps)


@njit(cache=True)
def compute_retention_routing(
    dt: np.float32,
    river_volume_m3: np.float32,
    discharge_before_diversion_m3_s: np.float32,
    is_rising_limb: bool,
    retention_storage_m3: np.float32,
    retention_max_storage_m3: np.float32,
    controlled_retention: bool,
    activation_threshold_m3_s: np.float32,
    release_threshold_factor: np.float32,
    inflow_limit_m3: np.float32,
    max_outflow_limit_m3: np.float32,
) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    """Calculate the water diversion and release for a single retention basin node.

    This function computes how much water is diverted from the river into a
    retention basin (during high flows) or released from the basin back into
    the river (during low flows) based on predefined thresholds.

    Notes:
        Diversions only occur when the river's discharge rate exceeds the activation threshold.
        For controlled retention basins, diversion can only occur on a rising limb of the hydrograph.
        Releases only occur when the storage is non-empty and the initial discharge rate is below
        the release threshold. High flows and low flows are handled as mutually exclusive states.

    Args:
        dt: The length of the routing time step (seconds).
        river_volume_m3: The current available volume of water in the river cell (m3).
        discharge_before_diversion_m3_s: The initial discharge rate in the river cell before diversion (m3/s).
        is_rising_limb: Flag indicating if the discharge is currently rising compared to the previous time step.
        retention_storage_m3: The current amount of water stored in the retention basin (m3).
        retention_max_storage_m3: The maximum storage capacity of the retention basin (m3).
        controlled_retention: Flag indicating whether the retention basin is controlled.
        activation_threshold_m3_s: The discharge threshold above which water is diverted into the basin (m3/s).
        release_threshold_factor: The multiplier applied to the activation threshold to determine the release threshold.
        inflow_limit_m3: The physical limit of water volume that can enter the basin during this step (m3).
        max_outflow_limit_m3: The maximum volume of water that the basin can release during this step (m3).

    Returns:
        A tuple containing:
            diverted_volume_m3: The volume of water diverted into the basin (m3).
            outflow_volume_m3: The volume of water released from the basin (m3).
            updated_retention_storage_m3: The updated retention storage (m3).
            final_river_volume_m3: The updated available water volume in the river cell (m3).

    Raises:
        ValueError: If `dt` is <= 0 or if `retention_max_storage_m3` is negative.
    """
    if dt <= np.float32(0.0):
        raise ValueError("Time step dt must be positive.")
    if retention_max_storage_m3 < np.float32(0.0):
        raise ValueError("Maximum retention storage must be non-negative.")

    # Determine available storage in the retention basin. Can not be negative.
    available_storage_m3: np.float32 = max(
        np.float32(0.0),
        retention_max_storage_m3 - retention_storage_m3,
    )

    diverted_volume_m3: np.float32 = np.float32(0.0)
    outflow_volume_m3: np.float32 = np.float32(0.0)

    # Determine the release threshold
    release_threshold_m3_s: np.float32 = (
        activation_threshold_m3_s * release_threshold_factor
    )

    # During high flow, divert water into the retention basin
    if discharge_before_diversion_m3_s > activation_threshold_m3_s:
        if not controlled_retention or is_rising_limb:
            discharge_above_activation_threshold_m3_s: np.float32 = (
                discharge_before_diversion_m3_s - activation_threshold_m3_s
            )
            diverted_volume_m3 = min(
                river_volume_m3,
                available_storage_m3,
                inflow_limit_m3,
                discharge_above_activation_threshold_m3_s * dt,
            )

    # During low flow, release water from the retention basin to the river
    elif discharge_before_diversion_m3_s <= release_threshold_m3_s:
        if retention_storage_m3 > np.float32(0.0):
            # We ensure the release does not cause the river flow rate to exceed the release threshold.
            allowed_extra_outflow_m3: np.float32 = (
                release_threshold_m3_s - discharge_before_diversion_m3_s
            ) * dt
            outflow_volume_m3 = min(
                allowed_extra_outflow_m3,
                max_outflow_limit_m3,
            )
            outflow_volume_m3 = max(np.float32(0.0), outflow_volume_m3)
            outflow_volume_m3 = min(outflow_volume_m3, retention_storage_m3)

    # Update retention storage and remaining river volume.
    updated_retention_storage_m3: np.float32 = (
        retention_storage_m3 + diverted_volume_m3 - outflow_volume_m3
    )
    final_river_volume_m3: np.float32 = (
        river_volume_m3 - diverted_volume_m3 + outflow_volume_m3
    )

    return (
        diverted_volume_m3,
        outflow_volume_m3,
        updated_retention_storage_m3,
        final_river_volume_m3,
    )
