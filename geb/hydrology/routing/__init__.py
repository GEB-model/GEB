"""Routing algorithms for river networks."""

import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
from affine import Affine
from numba import njit, prange
from tqdm import tqdm

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    ArrayInt64 as ArrayInt64,
    ArrayUint8,
    TwoDArrayBool,
    TwoDArrayFloat32,
    TwoDArrayInt32,
    TwoDArrayUint8,
)
from geb.module import Module
from geb.store import Bucket
from geb.workflows import balance_check
from geb.workflows.extreme_value_analysis import ReturnPeriodModel
from geb.workflows.io import read_geom, read_table

from .geometry import (
    NUM_FLOODPLAIN_LEVELS,
    compute_hydraulic_geometry_width,
    compute_top_width_from_depth,
    estimate_outflow_height,
    get_lake_factor,
)
from .kinematic import update_node_kinematic as update_node_kinematic
from .local_inertial import LocalInertial as LocalInertial
from .solver import solve_steady_backwater_network

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


MINIMUM_KINEMATIC_SLOPE_M_PER_M: np.float32 = np.float32(1e-4)


def get_discharge_per_river(
    rivers: gpd.GeoDataFrame,
    all_rivers: pd.DataFrame,
    source: Literal["file", "memory"] = "file",
    folder: Path | None = None,
    variables_to_report: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Get the discharge for each river from the output files.

    Args:
        rivers: A GeoDataFrame containing the rivers in the model, with columns "is_downstream_outflow", "is_upstream_of_downstream_basin", and "hydrography_xy".
        all_rivers: A DataFrame containing all rivers in the model, with columns "represented_in_grid", "hydrography_xy", and "downstream_ID".
        source: The source of the discharge data. Can be "file" or "memory".
        folder: The folder where the discharge files are stored.
        variables_to_report: A dictionary containing the variables to report.

    Returns:
        A DataFrame with the discharge for each river, with columns "discharge_m3_per_s" and "hydrography_xy".

    Raises:
        ValueError: If source is "file" and folder is None.
        ValueError: If source is "memory" and variables_to_report is None.
    """
    outflow_table: pd.DataFrame
    if source == "file":
        if folder is None:
            raise ValueError("folder must be provided if source is 'file'")
        outflow_table = read_table(folder / "river_outflow_hourly_m3_per_s.parquet")
    elif source == "memory":
        if variables_to_report is None:
            raise ValueError(
                "variables_to_report must be provided if source is 'memory'"
            )
        prefix: str = "river_outflow_hourly_m3_per_s_"
        outflow_table = pd.DataFrame(
            {
                key.removeprefix(prefix): pd.Series(
                    val["_data_array"][: val["_var_index"]],
                    index=val["_time_array"][: val["_var_index"]].astype(
                        "datetime64[s]"
                    ),
                )
                for key, val in variables_to_report.items()
                if key.startswith(prefix)
            }
        )
    else:
        raise ValueError(f"Unsupported source '{source}': expected 'file' or 'memory'")

    outflow_table.columns = outflow_table.columns.astype(str)

    discharge_data: dict[int, pd.Series] = {}
    for river_id in rivers.index:
        assert isinstance(river_id, int)
        xys: list[tuple[int, int]] = get_river_representative_xys(river_id, all_rivers)
        if len(xys) == 1:
            discharge_data[river_id] = outflow_table[str(river_id)]
        else:
            reach_keys: list[str] = [f"{river_id}_{i}" for i in range(len(xys))]
            discharge_data[river_id] = outflow_table[reach_keys].sum(axis=1)

    if not discharge_data:
        return pd.DataFrame()

    return pd.DataFrame(discharge_data)


def get_river_representative_xys(
    river_id: int, all_rivers: pd.DataFrame
) -> list[tuple[int, int]]:
    """Recursively find the nearest represented upstream rivers.

    Args:
        river_id: The ID of the river to find the upstream represented rivers for.
        all_rivers: A DataFrame containing all rivers in the model, with columns "represented_in_grid", "hydrography_xy", and "downstream_ID".

    Returns:
        A list of tuples containing the grid pixel coordinates of the nearest represented upstream rivers.
    """
    river = all_rivers.loc[river_id]
    if river["represented_in_grid"]:
        return [river["hydrography_xy"][-1]]

    upstream_rivers = all_rivers[all_rivers["downstream_ID"] == river_id]
    xys = []
    for idx, _ in upstream_rivers.iterrows():
        xys.extend(get_river_representative_xys(idx, all_rivers))
    return xys


def get_river_width(
    alpha: ArrayFloat32,
    beta: ArrayFloat32,
    discharge_m3_s: ArrayFloat32,
) -> ArrayFloat32:
    """Calculate the river width based on the alpha and beta parameters and the discharge.

    Args:
        alpha: The alpha parameter for the river width calculation.
        beta: The beta parameter for the river width calculation.
        discharge_m3_s: The discharge in cubic meters per second.

    Returns:
        A 1D array with the calculated river width for each cell.
    """
    return alpha * np.abs(discharge_m3_s) ** beta


@njit(fastmath=True, parallel=True)
def calculate_wetted_channel_area(
    use_kinematic: ArrayBool,
    water_stage_m: ArrayFloat32,
    bed_elevation_m: ArrayFloat32,
    bankfull_width_m: ArrayFloat32,
    bankfull_depth_m: ArrayFloat32,
    shape_exponent: ArrayFloat32,
    floodplain_profile_widths_m: TwoDArrayFloat32,
    floodplain_side_slope_m_per_m: ArrayFloat32,
    river_width_alpha: ArrayFloat32,
    river_width_beta: ArrayFloat32,
    discharge_m3_s: ArrayFloat32,
    river_length_m: ArrayFloat32,
    cell_area_m2: ArrayFloat32,
    is_waterbody: ArrayBool,
    wetted_channel_area_m2: ArrayFloat32,
) -> None:
    """Calculates wetted channel area for all river reaches across the grid in parallel without allocations.

    Differentiates between local inertial and kinematic routing reaches:
    - Local Inertial: Top width is determined from simulated water surface elevation relative
      to channel bed and cross-sectional geometry (in-bank power-law profile and overbank multi-level
      floodplain profile widths via compute_top_width_from_depth).
    - Kinematic: Top width is determined from instantaneous discharge using the hydraulic
      geometry power law w = alpha * |Q|^beta via compute_hydraulic_geometry_width.

    Args:
        use_kinematic: Boolean mask identifying kinematic wave reaches (True) vs local inertial reaches (False).
        water_stage_m: River water surface elevation array (meters).
        bed_elevation_m: Channel bed elevation array (meters).
        bankfull_width_m: Main channel bankfull width array (meters).
        bankfull_depth_m: Bankfull channel depth before floodplain spilling occurs (meters).
        shape_exponent: Power-law cross-sectional shape exponent array (dimensionless).
        floodplain_profile_widths_m: 2D array of 10-point cross-sectional top widths at fixed depth levels (meters).
        floodplain_side_slope_m_per_m: Valley side-slope ratio z (m/m horizontal:vertical).
        river_width_alpha: Channel width coefficient array for hydraulic geometry power law.
        river_width_beta: Channel width exponent array for hydraulic geometry power law.
        discharge_m3_s: Instantaneous discharge array (m³/s).
        river_length_m: Channel length per grid cell (meters).
        cell_area_m2: Planform area per grid cell (m²).
        is_waterbody: Boolean mask identifying lake/waterbody cells.
        wetted_channel_area_m2: Pre-allocated output array receiving wetted channel area (m²).
    """
    n: int = use_kinematic.shape[0]
    for i in prange(n):
        if is_waterbody[i] or river_length_m[i] <= np.float32(0.0):
            wetted_channel_area_m2[i] = np.float32(0.0)
            continue

        if use_kinematic[i]:
            # Kinematic wave reaches derive channel width from instantaneous discharge
            top_width_m: np.float32 = compute_hydraulic_geometry_width(
                discharge_m3_s=discharge_m3_s[i],
                alpha=river_width_alpha[i],
                beta=river_width_beta[i],
            )
        else:
            # Local inertial reaches derive channel width from simulated depth and cross section geometry
            water_depth_m: np.float32 = water_stage_m[i] - bed_elevation_m[i]
            top_width_m = compute_top_width_from_depth(
                water_depth_m=water_depth_m,
                bankfull_depth_m=bankfull_depth_m[i],
                bankfull_width_m=bankfull_width_m[i],
                shape_exponent=shape_exponent[i],
                floodplain_widths=floodplain_profile_widths_m[i],
                side_slope_z=floodplain_side_slope_m_per_m[i],
            )

        wetted_channel_area_m2[i] = min(
            cell_area_m2[i], top_width_m * river_length_m[i]
        )


@njit(fastmath=True, parallel=True)
def calculate_potential_river_evaporation(
    reference_evapotranspiration_m: ArrayFloat32,
    wetted_channel_area_m2: ArrayFloat32,
    potential_evaporation_m3: ArrayFloat32,
) -> None:
    """Calculates potential river evaporation from wetted channel surface area in parallel without allocations.

    Args:
        reference_evapotranspiration_m: Reference evapotranspiration rate array for open water (meters).
        wetted_channel_area_m2: Planform wetted channel surface area array (m²).
        potential_evaporation_m3: Pre-allocated output array receiving potential evaporation volume (m³).
    """
    n: int = wetted_channel_area_m2.shape[0]
    for i in prange(n):
        potential_evaporation_m3[i] = (
            reference_evapotranspiration_m[i] * wetted_channel_area_m2[i]
        )


@njit(fastmath=True, parallel=True)
def calculate_potential_river_transmission_loss(
    wetted_channel_area_m2: ArrayFloat32,
    riverbed_hydraulic_conductivity_m_s: ArrayFloat32,
    bed_elevation_m: ArrayFloat32,
    water_stage_m: ArrayFloat32,
    groundwater_head_m: ArrayFloat32,
    timestep_s: np.float32,
    potential_transmission_loss_m3: ArrayFloat32,
) -> None:
    """Calculates potential transmission loss in river reaches.

    Calculates Darcy seepage through the riverbed colmation layer driven by channel water
    stage and unconfined groundwater table elevation.

    Approach:
        1. Wetted Channel Area: Planform wetted channel area A_channel is precomputed based on
           routing geometry (in-bank power-law and floodplain profile widths for local inertial;
           hydraulic geometry for kinematic reaches).
        2. Hydraulic Gradient:
           - Gaining reach (groundwater_head >= water_stage): The groundwater table is at or above
             the river water surface, so transmission loss ceases (hydraulic_gradient = 0.0).
             Baseflow entering the river is handled separately by the groundwater module.
           - Perched / disconnected reach (groundwater_head <= bed_elevation): The groundwater table
             drops below the riverbed, forming an unsaturated zone beneath the channel. Seepage reaches
             a gravity-driven maximum under unit hydraulic gradient (hydraulic_gradient = 1.0).
           - Connected losing reach (bed_elevation < groundwater_head < water_stage): The river and
             unconfined aquifer are hydraulically connected, with downward gradient scaled by the active
             water depth: hydraulic_gradient = (water_stage - groundwater_head) / max(water_stage - bed_elevation, 0.01).
        3. Seepage Volume: Potential volumetric seepage over timestep dt is
           V = k_bed * hydraulic_gradient * A_channel * dt. Realized loss is subsequently constrained
           by available stream storage within the routing solver.

    Args:
        wetted_channel_area_m2: Planform wetted channel surface area array (m²).
        riverbed_hydraulic_conductivity_m_s: Riverbed hydraulic conductivity array (m/s).
        bed_elevation_m: Channel bed elevation array (meters).
        water_stage_m: River water stage elevation array (meters).
        groundwater_head_m: Groundwater head array for topmost aquifer layer (meters).
        timestep_s: Routing step duration (seconds).
        potential_transmission_loss_m3: Pre-allocated output array receiving potential transmission loss volume (m³).
    """
    n: int = wetted_channel_area_m2.shape[0]
    for i in prange(n):
        wetted_area: np.float32 = wetted_channel_area_m2[i]
        if wetted_area <= np.float32(0.0):
            potential_transmission_loss_m3[i] = np.float32(0.0)
            continue

        hydraulic_conductivity: np.float32 = riverbed_hydraulic_conductivity_m_s[i]
        if hydraulic_conductivity <= np.float32(0.0):
            potential_transmission_loss_m3[i] = np.float32(0.0)
            continue

        water_stage: np.float32 = water_stage_m[i]
        bed_elevation: np.float32 = bed_elevation_m[i]
        gw_head: np.float32 = groundwater_head_m[i]

        if gw_head >= water_stage:
            hydraulic_gradient: np.float32 = np.float32(0.0)
        elif gw_head <= bed_elevation:
            hydraulic_gradient = np.float32(1.0)
        else:
            water_depth_m: np.float32 = max(
                water_stage - bed_elevation, np.float32(0.01)
            )
            hydraulic_gradient = (water_stage - gw_head) / water_depth_m

        potential_transmission_loss_m3[i] = (
            hydraulic_conductivity * hydraulic_gradient * wetted_area * timestep_s
        )


@njit(fastmath=True)
def _accumulate_waterbody_runoff_and_evaporation(
    active_wb_cells: ArrayInt32,
    wb_cell_indices: ArrayInt32,
    side_flow: ArrayFloat32,
    ref_et: ArrayFloat32,
    wb_cell_counts: ArrayInt32,
    lake_area: ArrayFloat32,
    wb_storage: ArrayFloat64,
    return_flow_m3_to_wb: ArrayFloat32,
    wb_potential_river_evaporation_buffer_m3: ArrayFloat32,
    wb_act_evap_buf: ArrayFloat32,
) -> None:
    """Accumulates lateral runoff and computes evaporation for waterbodies in-place without allocations.

    Args:
        active_wb_cells: Waterbody ID for each waterbody grid cell.
        wb_cell_indices: Linear grid cell indices corresponding to waterbodies.
        side_flow: Lateral sideflow array in m³ per grid cell (modified in-place).
        ref_et: Reference evapotranspiration rate array for water (meters).
        wb_cell_counts: Total number of active grid cells per waterbody.
        lake_area: Surface area per waterbody (m²).
        wb_storage: Waterbody storage array in m³ (modified in-place).
        return_flow_m3_to_wb: Prescribed return flow volume per waterbody (m³).
        wb_potential_river_evaporation_buffer_m3: Reusable buffer for intermediate potential evaporation (m³).
        wb_act_evap_buf: Reusable buffer for output actual evaporation (m³).
    """
    n_active: int = active_wb_cells.shape[0]
    n_wb: int = wb_storage.shape[0]

    for w in range(n_wb):
        if w < return_flow_m3_to_wb.shape[0]:
            wb_storage[w] += return_flow_m3_to_wb[w]
        wb_potential_river_evaporation_buffer_m3[w] = np.float32(0.0)

    for k in range(n_active):
        w_id: int = active_wb_cells[k]
        c_idx: int = wb_cell_indices[k]
        wb_storage[w_id] += side_flow[c_idx]
        side_flow[c_idx] = np.float32(0.0)
        wb_potential_river_evaporation_buffer_m3[w_id] += ref_et[c_idx]

    for w in range(n_wb):
        if wb_cell_counts[w] > 0:
            pot_evap: np.float64 = (
                np.float64(wb_potential_river_evaporation_buffer_m3[w])
                / np.float64(wb_cell_counts[w])
            ) * np.float64(lake_area[w])
            act_evap: np.float64 = min(pot_evap, wb_storage[w])
            wb_storage[w] -= act_evap
            wb_act_evap_buf[w] = np.float32(act_evap)
        else:
            wb_act_evap_buf[w] = np.float32(0.0)


@njit(fastmath=True)
def _compute_retention_basin_evaporation(
    retention_cell_indices: ArrayInt32,
    retention_cell_basin_ids: ArrayInt32,
    retention_basin_area: ArrayFloat32,
    ref_et: ArrayFloat32,
    retention_storage: ArrayFloat32,
    pot_evap_buf: ArrayFloat32,
    act_evap_buf: ArrayFloat32,
) -> None:
    """Computes retention basin evaporation in-place without array allocations.

    Args:
        retention_cell_indices: Linear grid cell indices for retention basin nodes.
        retention_cell_basin_ids: Retention basin ID for each retention node.
        retention_basin_area: Nominal water surface area per retention basin (m²).
        ref_et: Reference evapotranspiration rate array for water per cell (meters).
        retention_storage: Retention basin storage array in m³ (modified in-place).
        pot_evap_buf: Reusable buffer for intermediate potential evaporation (m³).
        act_evap_buf: Reusable buffer for output actual evaporation (m³).
    """
    n_basins: int = retention_storage.shape[0]
    for b in range(n_basins):
        pot_evap_buf[b] = np.float32(0.0)

    n_cells: int = retention_cell_indices.shape[0]
    for k in range(n_cells):
        b_id: int = retention_cell_basin_ids[k]
        c_idx: int = retention_cell_indices[k]
        pot_evap_buf[b_id] += ref_et[c_idx] * retention_basin_area[b_id]

    for b in range(n_basins):
        act_evap: np.float32 = min(pot_evap_buf[b], retention_storage[b])
        retention_storage[b] -= act_evap
        act_evap_buf[b] = act_evap


def create_river_network(
    ldd_uncompressed: TwoDArrayUint8, mask: TwoDArrayBool, transform: Affine
) -> pyflwdir.FlwdirRaster:
    """Create a river network from a local drain direction (LDD) array.

    Args:
        ldd_uncompressed: A 2D array with the local drain direction (LDD) values.
        mask: A 2D boolean array with the same shape as the LDD array, where True indicates
            that the cell is part of the river network.
        transform: The affine transformation for the river network.

    Returns:
        A FlwdirRaster object representing the river network.
    """
    return pyflwdir.from_array(
        ldd_uncompressed,
        ftype="ldd",
        latlon=True,
        mask=mask,
        transform=np.array(transform),
    )


@njit(cache=True)
def fill_discharge_in_waterbodies(
    discharge_m3_s: ArrayFloat32,
    upstream_matrix_from_up_to_downstream: TwoDArrayInt32,
    idxs_up_to_downstream: ArrayInt32,
) -> ArrayFloat32:
    """Fill the discharge in waterbodies based on the discharge in upstream cells.

    Args:
        discharge_m3_s: A 1D array with the discharge in m3/s for
            each cell in the river network. Discharge in waterbodies is NaN.
        upstream_matrix_from_up_to_downstream: Upstream matrix from the river network, which is
            a 2D array. For each cell (first dimension) in the river network, it contains the indices of the upstream cells (second dimension).
            -1 indicates no upstream cell.
        idxs_up_to_downstream: Indices of the cells in the river network, associated with the upstream_matrix_from_up_to_downstream.

    Returns:
        A 1D array with the discharge in m3/s for each cell in the river network, where the discharge in waterbodies is filled based on the discharge in upstream cells.
    """
    for i in range(upstream_matrix_from_up_to_downstream.shape[0]):
        node: np.int32 = idxs_up_to_downstream[i]
        if np.isnan(discharge_m3_s[node]):
            upstream_nodes: ArrayInt32 = upstream_matrix_from_up_to_downstream[i]

            discharge_m3_s_node: np.float32 = np.float32(0.0)

            for upstream_node in upstream_nodes:
                if upstream_node == -1:
                    break

                upstream_discharge_m3_s: np.float32 = discharge_m3_s[upstream_node]
                if not np.isnan(upstream_discharge_m3_s):
                    discharge_m3_s_node += discharge_m3_s[upstream_node]

            discharge_m3_s[node] = discharge_m3_s_node

    return discharge_m3_s


class RoutingVariables(Bucket):
    """Routing variables."""

    discharge_step_count: int
    sum_of_all_discharge_steps: ArrayFloat64
    rivers: gpd.GeoDataFrame
    river_ids: ArrayInt32
    river_ids_no_waterbodies_removed: ArrayInt32
    active_rivers: gpd.GeoDataFrame
    observed_average_river_width: ArrayFloat32
    river_width: ArrayFloat32
    river_storage_m3: ArrayFloat64
    water_stage_m: ArrayFloat32
    discharge_in_rivers_m3_s_substep: ArrayFloat32
    discharge_m3_s_per_substep: TwoDArrayFloat32
    retention_basin_storage_m3: ArrayFloat32
    retention_basin_storage_m3_per_substep: TwoDArrayFloat32
    discharge_m3_s: ArrayFloat32
    retention_inflow_m3_daily: ArrayFloat32
    retention_outflow_m3_daily: ArrayFloat32
    river_storage_alpha: ArrayFloat32
    river_storage_beta: ArrayFloat32
    river_width_alpha: ArrayFloat32
    river_width_beta: ArrayFloat32
    river_transmission_loss_m3: ArrayFloat32


def select_active_rivers(
    rivers: gpd.GeoDataFrame,
    include_rivers_not_represented_in_grid: bool = False,
) -> gpd.GeoDataFrame:
    """Select river segments simulated inside the model domain.

    Downstream outflow segments are excluded. A segment absent from the routing
    grid is retained only when it connects to an upstream segment in that grid,
    unless include_rivers_not_represented_in_grid is True.

    Args:
        rivers: Built river network indexed by river ID.
        include_rivers_not_represented_in_grid: Whether to include river segments
            that are not represented in the routing grid.

    Returns:
        Active river geometries and their original attributes.
    """
    active_rivers: gpd.GeoDataFrame = rivers[
        (~rivers["is_downstream_outflow"]) & (~rivers["is_further_downstream_outflow"])
    ]

    if include_rivers_not_represented_in_grid:
        return active_rivers.copy()

    to_remove: set[int] = set()
    river_id: int
    for river_id in active_rivers.index[~active_rivers["represented_in_grid"]]:
        to_search: set[int] = {river_id}
        upstream_rivers: set[int] = set()
        while to_search:
            current_id: int = to_search.pop()
            upstream_segments: pd.DataFrame = rivers[
                rivers["downstream_ID"] == current_id
            ]
            represented: pd.Series = upstream_segments["represented_in_grid"]
            upstream_rivers.update(upstream_segments.index[represented])
            to_search.update(upstream_segments.index[~represented])
        # If no connected upstream segments are represented on the grid, exclude this reach
        if not upstream_rivers:
            to_remove.add(river_id)

    active_rivers = active_rivers[~active_rivers.index.isin(to_remove)]

    return active_rivers.copy()


def prune_starved_river_cells(
    waterbody_id: TwoDArrayInt32,
    waterbody_outflows: TwoDArrayInt32,
    river_ids: TwoDArrayInt32,
    flow_raster: pyflwdir.FlwdirRaster,
) -> TwoDArrayInt32:
    """Prune river cells downstream of non-outflow waterbody cells that receive no river inflow.

    When waterbodies are resolved with a single designated outflow point, non-outflow
    lake cells do not release discharge into downstream channels. Any river channels
    originating from non-outflow lake cells that receive no other upstream inflows
    (from active rivers or designated outflows) are starved abandoned channels.
    This function modifies river_ids in-place to set starved channel cells to -1.

    Args:
        waterbody_id: 2D array of water body IDs (-1 for non-waterbody).
        waterbody_outflows: 2D array containing waterbody IDs at outflow points and -1 elsewhere.
        river_ids: 2D array of river reach IDs (-1 for non-river).
        flow_raster: FlwdirRaster flow direction object.

    Returns:
        Updated 2D array of river IDs with starved channels set to -1.
    """
    flat_river_ids: ArrayInt32 = river_ids.ravel()
    flat_wb_id: ArrayInt32 = waterbody_id.ravel()
    flat_outflows: ArrayInt32 = waterbody_outflows.ravel()
    idxs_ds: ArrayInt64 = flow_raster.idxs_ds

    non_outflow_mask: ArrayBool = (flat_wb_id != -1) & (flat_outflows == -1)
    non_outflow_idxs: ArrayInt64 = np.where(non_outflow_mask)[0]
    if non_outflow_idxs.size == 0:
        return river_ids

    downstream_idxs: ArrayInt64 = idxs_ds[non_outflow_idxs]
    valid_candidates: ArrayBool = (
        (downstream_idxs != -1)
        & (downstream_idxs != non_outflow_idxs)
        & (flat_river_ids[downstream_idxs] != -1)
        & (flat_wb_id[downstream_idxs] == -1)
    )
    if not np.any(valid_candidates):
        return river_ids

    # Upstream mapping tracking cells that drain into each node
    up_map: defaultdict[int, list[int]] = defaultdict(list)
    for u, d in enumerate(idxs_ds):
        if d != -1 and d != u:
            up_map[int(d)].append(int(u))

    def has_valid_river_inflow(cell: int) -> bool:
        for u in up_map.get(cell, ()):
            # Active lake outflow or active river outside waterbodies
            if flat_outflows[u] != -1 or (
                flat_wb_id[u] == -1 and flat_river_ids[u] != -1
            ):
                return True
        return False

    queue: list[int] = [int(idx) for idx in set(downstream_idxs[valid_candidates])]
    visited: set[int] = set()

    while queue:
        curr: int = queue.pop()
        if curr in visited or flat_wb_id[curr] != -1 or flat_river_ids[curr] == -1:
            continue
        visited.add(curr)

        if not has_valid_river_inflow(curr):
            flat_river_ids[curr] = -1
            ds: int = int(idxs_ds[curr])
            if (
                ds != -1
                and ds != curr
                and flat_wb_id[ds] == -1
                and flat_river_ids[ds] != -1
            ):
                queue.append(ds)

    return river_ids


class Routing(Module):
    """Routing module of the hydrological model.

    Args:
        model: The GEB model instance.
        hydrology: The hydrology submodel instance.
    """

    var: RoutingVariables
    inflow: dict[tuple[int, int], ArrayFloat32]
    floodplain_profile_widths_m: TwoDArrayFloat32
    floodplain_side_slope_m_per_m: ArrayFloat32
    use_kinematic: ArrayBool
    shape_exponent: ArrayFloat32

    def __init__(self, model: GEBModel, hydrology: Hydrology) -> None:
        """Initialize the Routing module.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology submodel instance.

        Raises:
            ValueError: If inflow is configured on a waterbody cell.
        """
        super().__init__(model)

        self.config = model.config["hydrology"]["routing"]
        self.fix_river_geometry_after_spinup: bool = bool(
            self.config["fix_river_geometry_after_spinup"]
        )

        self.default_missing_channel_width: float = 0.5  # 0.5 is based on assumption here, but could be improved: https://pubs.usgs.gov/publication/wri854004
        self.default_minimum_discharge_m3_per_s: float = 0.01

        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        self.ldd: ArrayUint8 = self.grid.load2d(
            self.model.files["grid"]["routing/ldd"],
        )

        mask: TwoDArrayBool = ~self.grid.mask

        ldd_uncompressed: TwoDArrayUint8 = np.full_like(mask, 255, dtype=self.ldd.dtype)
        ldd_uncompressed[mask] = self.ldd.ravel()

        self.river_network: pyflwdir.FlwdirRaster = create_river_network(
            ldd_uncompressed=ldd_uncompressed, mask=mask, transform=self.grid.transform
        )
        self.river_network.order_cells(method="walk")
        mapper: ArrayInt32 = np.full(self.river_network.size + 1, -1, dtype=np.int32)
        indices_init: ArrayInt64 = np.arange(self.river_network.size, dtype=np.int32)[
            self.river_network.mask
        ]
        mapper[indices_init] = np.arange(indices_init.size, dtype=np.int32)
        self._compressed_routing_seq: ArrayInt32 = mapper[self.river_network.idxs_seq]
        unmasked_ds: ArrayInt32 = self.river_network.idxs_ds[indices_init]
        self._compressed_ds: ArrayInt32 = mapper[unmasked_ds]

        self.basin_ids: TwoDArrayInt32 = self.hydrology.grid.load2d(
            self.model.files["grid"]["routing/basin_ids"], compress=False
        )

        self.retention_basin_ids: ArrayInt32 = self.grid.load2d(
            self.model.files["grid"]["routing/retention_basin_ids"],
        )
        self.retention_basin_data: pd.DataFrame = read_geom(
            self.model.files["geom"]["routing/retention_basins"]
        ).set_index("ID")
        # ensure retention basin data is continuous and starts from 0
        assert (
            self.retention_basin_data.empty  # allow empty retention basin data
            or (
                (self.retention_basin_data.index.min() == 0)
                and (
                    self.retention_basin_data.index.max()
                    == len(self.retention_basin_data) - 1
                )
            )
        ), "Retention basin data index must be continuous and start from 0"

        # initialize static retention arrays
        self.retention_max_storage_m3 = self.retention_basin_data[
            "retention_max_storage_m3"
        ].to_numpy(dtype=np.float32, copy=True)
        if np.isnan(self.retention_max_storage_m3).any():
            self.model.logger.warning(
                "Retention basin data contains NaN values in 'retention_max_storage_m3'. "
                "These will be treated as 0.0 m3. Please check your input data."
            )
            self.retention_max_storage_m3[np.isnan(self.retention_max_storage_m3)] = 0.0

        self.controlled_retention = self.retention_basin_data[
            "controlled_retention"
        ].to_numpy(dtype=bool, copy=True)

        self.retention_activation_threshold_m3_s = np.full(
            len(self.retention_basin_data), np.inf, dtype=np.float32
        )
        self.retention_basin_is_active = self.retention_basin_data["active"]

        if not self.retention_basin_data.empty:
            self._retention_basin_area: ArrayFloat32 = (
                self.retention_max_storage_m3 / np.float32(3.0)
            ).astype(np.float32)
            self._retention_cell_mask: ArrayBool = self.retention_basin_ids != -1
            self._retention_cell_indices: ArrayInt32 = np.where(
                self._retention_cell_mask
            )[0].astype(np.int32)
            self._retention_cell_basin_ids: ArrayInt32 = self.retention_basin_ids[
                self._retention_cell_indices
            ]
        else:
            self._retention_basin_area = np.zeros(0, dtype=np.float32)
            self._retention_cell_mask = np.zeros(0, dtype=bool)
            self._retention_cell_indices = np.zeros(0, dtype=np.int32)
            self._retention_cell_basin_ids = np.zeros(0, dtype=np.int32)

        # Initialize reusable buffers to avoid heap allocations in hourly step()
        self._potential_river_evaporation_buffer_m3: ArrayFloat32 = (
            self.grid.full_compressed(0, dtype=np.float32)
        )
        self._side_flow_buffer_m3: ArrayFloat32 = self.grid.full_compressed(
            0, dtype=np.float32
        )
        self._over_abstraction_daily: ArrayFloat32 = self.grid.full_compressed(
            0, dtype=np.float32
        )
        self._evaporation_in_rivers_m3: ArrayFloat32 = self.grid.full_compressed(
            0, dtype=np.float32
        )
        self._potential_transmission_loss_buffer_m3: ArrayFloat32 = (
            self.grid.full_compressed(0, dtype=np.float32)
        )
        self._groundwater_head_top_layer_m: ArrayFloat32 = self.grid.full_compressed(
            0, dtype=np.float32
        )
        self._wetted_channel_area_buffer_m2: ArrayFloat32 = self.grid.full_compressed(
            0, dtype=np.float32
        )
        self._riverbed_hydraulic_conductivity_m_per_s: ArrayFloat32 = (
            self.grid.full_compressed(0, dtype=np.float32)
        )
        self.var.river_transmission_loss_m3 = self.grid.full_compressed(
            0, dtype=np.float32
        )
        self._pot_evap_ret_buf: ArrayFloat32 = np.zeros(
            len(self.retention_basin_data), dtype=np.float32
        )
        self._retention_act_evap_buf: ArrayFloat32 = np.zeros(
            len(self.retention_basin_data), dtype=np.float32
        )
        self._wb_cell_mask: ArrayBool = np.zeros(0, dtype=bool)
        self._not_wb_cell_mask: ArrayBool = np.zeros(0, dtype=bool)
        self._wb_cell_indices: ArrayInt32 = np.zeros(0, dtype=np.int32)
        self._active_wb_cells: ArrayInt32 = np.zeros(0, dtype=np.int32)
        self._wb_cell_counts: ArrayInt32 = np.zeros(0, dtype=np.int32)
        self._wb_potential_river_evaporation_buffer_m3: ArrayFloat32 = np.zeros(
            0, dtype=np.float32
        )
        self._wb_act_evap_buf: ArrayFloat32 = np.zeros(0, dtype=np.float32)
        self._wb_outflow_rate_buf: ArrayFloat32 = np.zeros(0, dtype=np.float32)
        self._reservoir_indices: ArrayInt32 = np.zeros(0, dtype=np.int32)
        self._river_cell_indices: ArrayInt32 = np.zeros(0, dtype=np.int32)
        self.var.discharge_m3_s = self.grid.full_compressed(0, dtype=np.float32)

        self.inflow = {}
        self._inflow_list: list[tuple[int, ArrayFloat32]] = []
        self.inflow_idx: int = -1  # index for the current time step in the inflow data
        if "routing/inflow_m3_per_s" in self.model.files["table"]:
            inflow_per_location: pd.DataFrame = read_table(
                self.model.files["table"]["routing/inflow_m3_per_s"]
            )
            # select the right time steps from the inflow data
            expected_time_steps = pd.date_range(
                start=self.model.simulation_start,
                end=self.model.simulation_end + self.model.timestep_length,
                freq="H",
            )
            inflow_per_location = inflow_per_location.loc[expected_time_steps]

            inflow_locations: gpd.GeoDataFrame = read_geom(
                self.model.files["geom"]["routing/inflow_locations"]
            ).set_index("ID")  # ty:ignore[invalid-assignment]
            for inflow_id, inflow in inflow_per_location.items():
                location: pd.Series = inflow_locations.loc[inflow_id]
                y: int = location["y"]
                x: int = location["x"]
                cell_idx: int = self.grid.linear_mapping[y, x]
                if self.grid.var.waterbody_ids[cell_idx] != -1:
                    raise ValueError("Inflow cannot be added to waterbody cells.")
                inflow_arr = inflow.to_numpy(dtype=np.float32)
                self.inflow[(y, x)] = inflow_arr
                self._inflow_list.append((cell_idx, inflow_arr))

            assert self.model.current_time == inflow_per_location.index[0]

        self.var.retention_basin_storage_m3_per_substep = np.full(
            (24, mask.size),
            0,
            dtype=np.float32,
        )

        if self.model.in_spinup:
            self.spinup()

    def load_rivers(
        self,
        grid_linear_mapping: TwoDArrayInt32,
    ) -> tuple[gpd.GeoDataFrame, ArrayInt32, ArrayInt32]:
        """Load the river network geometries.

        Args:
            grid_linear_mapping: A 2D array mapping grid cells to linear indices.

        Returns:
            A GeoDataFrame containing the river network geometries, the updated river IDs and the original river IDs before removing waterbodies.
        """
        raw_wb_id = self.grid.load2d(
            self.model.files["grid"]["waterbodies/waterbody_id"], compress=False
        )
        wb_geom_file = self.model.files["geom"]["waterbodies/waterbody_data"]
        wb_data_df = read_geom(wb_geom_file).set_index("waterbody_id")
        active_ids = wb_data_df.index[wb_data_df["waterbody_type"] != 0].values
        is_waterbody: TwoDArrayBool = np.isin(raw_wb_id, active_ids)

        rivers: gpd.GeoDataFrame = read_geom(self.model.files["geom"]["routing/rivers"])
        rivers["return_period_2_years_daily_m3_per_s"] = np.nan
        rivers["width_is_observed"] = rivers["width"].notnull()

        # set river ID to -1 for waterbody cells
        river_ids: TwoDArrayInt32 = self.grid.load2d(
            self.model.files["grid"]["routing/river_ids"], compress=False
        )
        # keep a copy of the original river IDs before removing waterbodies,
        # which is needed for some output variables
        river_ids_no_waterbodies_removed: ArrayInt32 = self.grid.compress(river_ids)
        river_ids[is_waterbody] = -1  # set river ID to -1 for waterbody cells

        # Load waterbody outflow points to identify active outflows
        raw_wb_outflows: TwoDArrayInt32 = self.grid.load2d(
            self.model.files["grid"]["waterbodies/waterbody_outflow_points"],
            compress=False,
        )
        is_active_outflow: TwoDArrayBool = np.isin(raw_wb_outflows, active_ids)
        active_wb_outflows: TwoDArrayInt32 = np.where(
            is_active_outflow, raw_wb_outflows, -1
        ).astype(np.int32)
        active_wb_id: TwoDArrayInt32 = np.where(is_waterbody, raw_wb_id, -1).astype(
            np.int32
        )

        # Prune river cells downstream of non-outflow waterbody cells that receive no river inflow
        n_river_cells_before: int = int(np.sum(river_ids != -1))
        river_ids = prune_starved_river_cells(
            waterbody_id=active_wb_id,
            waterbody_outflows=active_wb_outflows,
            river_ids=river_ids,
            flow_raster=self.river_network,
        )
        self.model.logger.debug(
            f"Pruned {n_river_cells_before - int(np.sum(river_ids != -1))} starved river cells."
        )
        river_ids_compressed: ArrayInt32 = self.grid.compress(river_ids)

        # select only hydrography_xy that are still active river cells
        # and store the mask to filter other columns as well
        valid_river_cell_mask: TwoDArrayBool = river_ids != -1
        active_cell_mask: pd.Series = rivers["hydrography_xy"].apply(
            lambda xys: [valid_river_cell_mask[xy[1], xy[0]] for xy in xys]
        )
        rivers["hydrography_xy_no_waterbodies_removed"] = rivers["hydrography_xy"]

        rivers["hydrography_xy"] = [
            [(int(xy[0]), int(xy[1])) for xy, m in zip(xys, mask) if m]
            for xys, mask in zip(rivers["hydrography_xy"], active_cell_mask)
        ]

        rivers["hydrography_upstream_area_m2_no_waterbodies_removed"] = rivers[
            "hydrography_upstream_area_m2"
        ]
        rivers["hydrography_upstream_area_m2"] = [
            np.array([ua for ua, m in zip(uas, mask) if m])
            for uas, mask in zip(
                rivers["hydrography_upstream_area_m2"], active_cell_mask
            )
        ]

        # update represented_in_grid based on whether there are any hydrography_xy left after removing waterbodies and starved channels
        rivers["represented_in_grid"] = rivers["hydrography_xy"].apply(
            lambda xys: len(xys) > 0
        )
        rivers["hydrography_linear"] = rivers["hydrography_xy"].apply(
            lambda xys: np.array(
                [grid_linear_mapping[xy[1], xy[0]].item() for xy in xys]
            )
        )
        return rivers, river_ids_compressed, river_ids_no_waterbodies_removed

    def _update_rivers_geometry(
        self,
        width_grid: ArrayFloat32,
        depth_grid: ArrayFloat32,
        manning_grid: ArrayFloat32,
        floodplain_profile_widths_grid_m: TwoDArrayFloat32,
        floodplain_side_slope_grid_m_per_m: ArrayFloat32,
    ) -> None:
        """Update bankfull width, depth, and Manning's n on river reaches GeoDataFrame from routing grid arrays.

        Aggregates bankfull width (meters), depth (meters), Manning roughness, and 10-point floodplain profile
        widths across all grid cells belonging to each river reach, assigning reach-average values directly
        to self.var.rivers.

        Notes:
            During model execution, channel geometry is updated dynamically during the spinup period.
            When fix_river_geometry_after_spinup is True (the default), river geometry is kept
            fixed after spinup and this method is not called during the run period.

        Args:
            width_grid: Bankfull channel width per routing grid cell (meters).
            depth_grid: Bankfull channel depth per routing grid cell (meters).
            manning_grid: Channel Manning roughness per routing grid cell (s/m^(1/3)).
            floodplain_profile_widths_grid_m: 10-point cross-sectional top widths at fixed depth levels per routing grid cell (meters).
            floodplain_side_slope_grid_m_per_m: Valley side-slope ratio z per routing grid cell (m/m horizontal:vertical).
        """
        valid_river_cell_mask: ArrayBool = self.var.river_ids != -1
        valid_reach_ids: ArrayInt32 = self.var.river_ids[valid_river_cell_mask]
        valid_widths: ArrayFloat32 = width_grid[valid_river_cell_mask]
        valid_depths: ArrayFloat32 = depth_grid[valid_river_cell_mask]
        valid_mannings: ArrayFloat32 = manning_grid[valid_river_cell_mask]

        raw_slope: ArrayFloat32 = self.grid.var.river_slope_m_per_m
        assert raw_slope.shape == valid_river_cell_mask.shape, (
            f"river_slope_m_per_m shape {raw_slope.shape} does not match routing grid {valid_river_cell_mask.shape}"
        )
        grid_dict: dict[str, Any] = {
            "river_id": valid_reach_ids,
            "width": valid_widths,
            "depth": valid_depths,
            "manning": valid_mannings,
            "slope": raw_slope[valid_river_cell_mask],
            "river_width_alpha": self.var.river_width_alpha[valid_river_cell_mask],
            "river_width_beta": self.var.river_width_beta[valid_river_cell_mask],
        }
        grid_dict["floodplain_side_slope_m_per_m"] = floodplain_side_slope_grid_m_per_m[
            valid_river_cell_mask
        ]
        cols: list[str] = [
            "width",
            "depth",
            "manning",
            "slope",
            "river_width_alpha",
            "river_width_beta",
            "floodplain_side_slope_m_per_m",
        ]
        for k in range(NUM_FLOODPLAIN_LEVELS):
            col_k: str = f"floodplain_profile_width_{k}_m"
            grid_dict[col_k] = floodplain_profile_widths_grid_m[
                valid_river_cell_mask, k
            ]
            cols.append(col_k)

        df_grid: pd.DataFrame = pd.DataFrame(grid_dict)
        reach_means: pd.DataFrame = df_grid.groupby("river_id").mean()
        self.var.rivers.loc[reach_means.index, cols] = reach_means[cols].astype(
            np.float64
        )

        has_observed_cell: ArrayBool = ~np.isnan(
            self.var.observed_average_river_width[valid_river_cell_mask]
        )
        df_grid["width_is_observed"] = has_observed_cell
        reach_obs: pd.Series | pd.DataFrame = df_grid.groupby("river_id")[
            "width_is_observed"
        ].any()
        self.var.rivers.loc[reach_obs.index, "width_is_observed"] = reach_obs.astype(
            bool
        )

        self._estimate_unrepresented_rivers_geometry()

        represented_rivers: pd.DataFrame = self.var.rivers[
            self.var.rivers["represented_in_grid"]
        ]
        assert (
            represented_rivers["width"].notnull().all()
            and represented_rivers["depth"].notnull().all()
            and represented_rivers["manning"].notnull().all()
        ), (
            "All rivers represented in the grid must have width, depth, and manning defined."
        )

    def _estimate_unrepresented_rivers_geometry(self) -> None:
        """Estimate bankfull discharge and hydraulic geometry for rivers not represented in grid.

        Traverses unrepresented river reaches in topological order (using Shreve stream order)
        and calculates bankfull discharge, river width parameters (alpha, beta), and Manning
        roughness based on their upstream river network. Width is estimated using downstream
        hydraulic geometry and depth is derived via Manning normal depth inversion:
            h_bf = (r + 1) * (n * Q_bf / (W * sqrt(S)))^0.6
        Reaches without upstream rivers are not assigned geometry and retain NaN.

        Notes:
            Bankfull discharge is the sum of upstream bankfull discharges (m3/s).
            Parameters (alpha, beta, manning) are combined using discharge-weighted
            averaging across upstream segments, which ensures scaling consistency.
        """
        unrepresented_mask: pd.Series = ~self.var.rivers["represented_in_grid"]
        if not unrepresented_mask.any():
            return

        depth_params: dict[str, Any] = self.config["river_depth"]
        min_depth_m: float = float(depth_params["min_depth_m"])
        default_missing_width: float = float(self.default_missing_channel_width)
        shape_exponent: float = float(depth_params["shape_exponent"])
        shape_factor: float = shape_exponent + 1.0

        rivers: gpd.GeoDataFrame = self.var.rivers

        # Build upstream adjacency mapping: downstream reach -> list of upstream reaches
        downstream_ids: dict[int, Any] = rivers["downstream_ID"].to_dict()
        upstream_map: dict[int, list[int]] = {}
        for u_idx, ds_id in downstream_ids.items():
            if pd.notna(ds_id):
                upstream_map.setdefault(int(ds_id), []).append(int(u_idx))

        # Traverse unrepresented reaches from upstream to downstream using shreve_stream_order
        unrepresented_reaches: pd.Index = (
            rivers.loc[unrepresented_mask].sort_values("shreve_stream_order").index
        )

        for reach_id in unrepresented_reaches:
            upstream_reaches: list[int] = upstream_map.get(reach_id, [])
            if not upstream_reaches:
                continue

            up: pd.DataFrame = rivers.loc[upstream_reaches]
            valid_up: pd.DataFrame = up[up["width"].notnull() & up["depth"].notnull()]
            if valid_up.empty:
                continue

            assert (valid_up["width"] > 0).all(), (
                f"Upstream river widths for reach {reach_id} must be strictly positive."
            )
            assert (valid_up["river_width_alpha"] > 0).all(), (
                f"Upstream river_width_alpha for reach {reach_id} must be strictly positive."
            )
            assert (valid_up["river_width_beta"] > 0).all(), (
                f"Upstream river_width_beta for reach {reach_id} must be strictly positive."
            )

            # Upstream discharge: simulated Q2 or width inversion proxy
            q_inv: pd.Series = (valid_up["width"] / valid_up["river_width_alpha"]) ** (
                1.0 / valid_up["river_width_beta"]
            )
            q_sim: pd.Series = valid_up["return_period_2_years_daily_m3_per_s"]
            upstream_q: np.ndarray = np.where(
                q_sim.notnull() & (q_sim > 0), q_sim, q_inv
            )
            raw_summed_q: float = float(upstream_q.sum())
            assert raw_summed_q > 0.0, (
                f"Combined bankfull discharge for reach {reach_id} must be strictly positive."
            )

            weights: np.ndarray = upstream_q / raw_summed_q

            # Discharge-weighted average parameters
            combined_alpha: float = float(
                (weights * valid_up["river_width_alpha"]).sum()
            )
            combined_beta: float = float((weights * valid_up["river_width_beta"]).sum())
            combined_manning: float = max(
                float((weights * valid_up["manning"]).sum()), 0.01
            )

            # Calibrated bankfull discharge for deriving geometry
            bankfull_q: float = raw_summed_q * float(self.bankfull_discharge_multiplier)
            assert bankfull_q > 0.0, (
                f"Bankfull discharge for reach {reach_id} must be strictly positive."
            )

            # Width: propagate observed width or estimate from downstream hydraulic geometry
            has_observed_upstream: bool = bool(valid_up["width_is_observed"].any())
            reach_already_observed: bool = bool(
                rivers.loc[reach_id, "width_is_observed"]
            ) and bool(pd.notnull(rivers.loc[reach_id, "width"]))

            is_observed: bool = False
            estimated_width: float
            if reach_already_observed:
                estimated_width = float(rivers.loc[reach_id, "width"])
                is_observed = True
            elif has_observed_upstream:
                # Inherit and combine upstream channel widths using Leopold-Maddock power-law
                # summation W = (sum W_i^(1/beta))^beta (simplified to sqrt(sum W_i^2) for beta=0.5).
                sum_w_powers: float = float(
                    (valid_up["width"] ** (1.0 / valid_up["river_width_beta"])).sum()
                )
                estimated_width = max(
                    sum_w_powers**combined_beta,
                    default_missing_width,
                )
                is_observed = True
            else:
                estimated_width = max(
                    combined_alpha * (bankfull_q**combined_beta),
                    default_missing_width,
                )

            assert estimated_width > 0.0, (
                f"Estimated width for reach {reach_id} must be strictly positive."
            )

            reach_slope_raw: Any = (
                rivers.loc[reach_id, "slope"] if "slope" in rivers.columns else np.nan
            )
            if pd.notnull(reach_slope_raw) and float(reach_slope_raw) > 0.0:
                slope_val: float = float(reach_slope_raw)
            elif "slope" in valid_up and valid_up["slope"].notnull().any():
                slope_val = float(valid_up["slope"].mean())
            else:
                slope_val = float(MINIMUM_KINEMATIC_SLOPE_M_PER_M)

            eff_slope: float = max(slope_val, float(MINIMUM_KINEMATIC_SLOPE_M_PER_M))
            mean_channel_depth: float = (
                (combined_manning * bankfull_q)
                / (estimated_width * float(np.sqrt(eff_slope)))
            ) ** 0.6
            estimated_depth: float = max(
                shape_factor * mean_channel_depth,
                min_depth_m,
            )

            alpha_to_assign: float = (
                estimated_width / (bankfull_q**combined_beta)
                if is_observed and bankfull_q > 0.0
                else combined_alpha
            )

            rivers.loc[
                reach_id,
                [
                    "width",
                    "depth",
                    "manning",
                    "river_width_alpha",
                    "river_width_beta",
                    "return_period_2_years_daily_m3_per_s",
                    "width_is_observed",
                ],
            ] = [
                float(estimated_width),
                float(estimated_depth),
                float(combined_manning),
                float(alpha_to_assign),
                float(combined_beta),
                float(raw_summed_q),
                bool(is_observed),
            ]

    @property
    def bankfull_discharge_multiplier(self) -> float:
        """Calibration multiplier for bankfull discharge.

        Returns:
            The bankfull discharge multiplier (dimensionless).
        """
        return self.model.config["hydrology"]["routing"][
            "bankfull_discharge_multiplier"
        ]

    @property
    def has_simulated_bankfull_q(self) -> bool:
        """Whether simulated 2-year flood return periods are available for represented rivers."""
        return bool(
            self.var.rivers.loc[
                self.var.rivers["represented_in_grid"],
                "return_period_2_years_daily_m3_per_s",
            ]
            .notnull()
            .any()
        )

    def calculate_bankfull_width(
        self,
        use_simulated_bankfull_q: bool,
    ) -> ArrayFloat32:
        """Calculate bankfull top channel width across the routing grid.

        Uses observed widths where available. For unobserved channels, estimates width
        via downstream hydraulic geometry (W = alpha * Q2^beta) if simulated 2-year flood
        discharge is available, or defaults to default_missing_channel_width otherwise.
        Constrains bankfull channel width to the floodplain width at level 0
        (floodplain_profile_widths_m[:, 0]) with a floor of default_missing_channel_width
        so river width does not exceed the valley profile.

        Args:
            use_simulated_bankfull_q: Whether to estimate unobserved widths using
                simulated 2-year flood return periods from self.var.rivers.

        Returns:
            Bankfull top channel width per routing grid cell (meters).

        Raises:
            ValueError: If use_simulated_bankfull_q is True but
                'return_period_2_years_daily_m3_per_s' contains null values in self.var.rivers.
        """
        if use_simulated_bankfull_q:
            q2_col: pd.Series = self.var.rivers.loc[
                self.var.rivers["represented_in_grid"],
                "return_period_2_years_daily_m3_per_s",
            ]
            if q2_col.isnull().any():
                raise ValueError(
                    "use_simulated_bankfull_q is True, but 'return_period_2_years_daily_m3_per_s' contains null values in self.var.rivers."
                )
            bankfull_discharge_m3_s: ArrayFloat32 = q2_col.reindex(
                self.var.river_ids
            ).values.astype(np.float32) * np.float32(self.bankfull_discharge_multiplier)
            width_params: dict[str, Any] = self.config["river_width"]
            default_alpha: np.float32 = np.float32(width_params["default_alpha"])
            default_beta: np.float32 = np.float32(width_params["beta"])
            use_observed_width_as_bankfull: bool = bool(
                self.config["river_depth"]["use_observed_width_as_bankfull"]
            )
            alpha: ArrayFloat32 | np.float32 = (
                default_alpha
                if use_observed_width_as_bankfull
                else self.var.river_width_alpha
            )
            beta: ArrayFloat32 | np.float32 = (
                default_beta
                if use_observed_width_as_bankfull
                else self.var.river_width_beta
            )
            estimated_width: ArrayFloat32 = np.maximum(
                alpha * (np.maximum(bankfull_discharge_m3_s, np.float32(0.0)) ** beta),
                np.float32(self.default_missing_channel_width),
            )
            raw_width = np.where(
                np.isnan(self.var.observed_average_river_width),
                np.where(
                    self.var.river_ids == -1,
                    np.float32(self.default_missing_channel_width),
                    estimated_width,
                ),
                self.var.observed_average_river_width,
            ).astype(np.float32)
        else:
            upstream_area_km2: ArrayFloat32 = (
                self.grid.var.upstream_area / np.float32(1e6)
            ).astype(np.float32)
            estimated_width_from_area: ArrayFloat32 = np.maximum(
                np.float32(1.3)
                * (np.maximum(upstream_area_km2, np.float32(0.0)) ** np.float32(0.45)),
                np.float32(self.default_missing_channel_width),
            )
            raw_width = np.where(
                np.isnan(self.var.observed_average_river_width),
                np.where(
                    self.var.river_ids == -1,
                    np.float32(self.default_missing_channel_width),
                    estimated_width_from_area,
                ),
                self.var.observed_average_river_width,
            ).astype(np.float32)

        # Limit bankfull channel width to the floodplain width at level 0 (d0 = 0.1 m).
        # We enforce a floor of default_missing_channel_width so channels are not clamped below the minimum width.
        max_floodplain_width: ArrayFloat32 = np.maximum(
            self.floodplain_profile_widths_m[:, 0],
            np.float32(self.default_missing_channel_width),
        )
        return np.where(
            self.var.river_ids == -1,
            raw_width,
            np.minimum(raw_width, max_floodplain_width),
        ).astype(np.float32)

    def calculate_bankfull_depth(
        self,
        bankfull_top_width_m: ArrayFloat32,
        use_simulated_bankfull_q: bool,
    ) -> ArrayFloat32:
        """Calculate bankfull channel depth across the routing grid.

        Solves the 1D steady-state local inertial momentum equation (Standard Step Method)
        along the river network for local inertial reaches (river_ids != -1) from downstream
        boundary controls (ocean, lake, or terminal sinks) upstream to headwaters, capturing M1
        backwater deepening and relaxing to Manning normal depth on steep reaches:
            h_norm = (r + 1) * (n * Q / (W * sqrt(S)))^0.6
        For kinematic wave reaches (river_ids == -1), bankfull depth is directly calculated via
        Manning's normal depth inversion, which matches the kinematic wave conveyance equation.

        When use_simulated_bankfull_q is True, simulated 2-year flood discharge Q_2 (from
        self.var.rivers['return_period_2_years_daily_m3_per_s']) is mapped to grid cells.
        When False (e.g. during the first year of simulation or during spinup), empirical
        width inversion proxy is used.

        Args:
            bankfull_top_width_m: Bankfull top channel width per cell (meters).
            use_simulated_bankfull_q: Whether to use simulated 2-year flood return periods from
                self.var.rivers. Must be False during the first year of simulation or spinup.

        Returns:
            Bankfull centerline channel depth per cell (meters).

        Raises:
            ValueError: If use_simulated_bankfull_q is True but
                'return_period_2_years_daily_m3_per_s' contains null values in self.var.rivers,
                or if calculated bankfull depth exceeds 250 m.
        """
        width_params: dict[str, Any] = self.config["river_width"]
        default_alpha: np.float32 = np.float32(width_params["default_alpha"])
        default_beta: np.float32 = np.float32(width_params["beta"])

        depth_params: dict[str, Any] = self.config["river_depth"]
        min_depth_m: np.float32 = np.float32(depth_params["min_depth_m"])
        shape_exponent_val: np.float32 = np.float32(depth_params["shape_exponent"])
        use_observed_width_as_bankfull: bool = bool(
            depth_params["use_observed_width_as_bankfull"]
        )

        q_proxy: ArrayFloat32
        alpha: ArrayFloat32 | None = None
        beta: ArrayFloat32 | None = None
        q_mult: np.float32 = np.float32(self.bankfull_discharge_multiplier)
        if use_observed_width_as_bankfull:
            # Downstream hydraulic geometry width inversion proxy Q_proxy = (W / alpha)^(1 / beta):
            q_proxy = (bankfull_top_width_m / default_alpha) ** (
                np.float32(1.0) / default_beta
            )
        else:
            width_alpha: ArrayFloat32 = self.var.river_width_alpha
            width_beta: ArrayFloat32 = self.var.river_width_beta
            alpha = np.maximum(width_alpha, np.float32(1.0))
            beta = np.maximum(width_beta, np.float32(0.1))
            q_proxy = (bankfull_top_width_m / alpha) ** (np.float32(1.0) / beta)

        if use_simulated_bankfull_q:
            q2_col: pd.Series = self.var.rivers.loc[
                self.var.rivers["represented_in_grid"],
                "return_period_2_years_daily_m3_per_s",
            ]
            if q2_col.isnull().any():
                raise ValueError(
                    "use_simulated_bankfull_q is True, but 'return_period_2_years_daily_m3_per_s' contains null values in self.var.rivers."
                )
            q2_mapped: ArrayFloat32 = q2_col.reindex(self.var.river_ids).values.astype(
                np.float32
            )
            bankfull_discharge_m3_s: ArrayFloat32 = (
                np.where(
                    self.var.river_ids == -1,
                    q_proxy,
                    q2_mapped,
                ).astype(np.float32)
                * q_mult
            )
        else:
            bankfull_discharge_m3_s = q_proxy * q_mult

        # For a power-law channel W(y) = W_bf * (y / h_bf)^r, cross-sectional area is A_bf = (1 / (r + 1)) * W_bf * h_bf.
        # Mean bankfull depth is h_mean = A_bf / W_bf = h_bf / (r + 1).
        # Hence maximum centerline depth is h_bf = (r + 1) * h_mean (factor of 1.5 for a parabolic channel r = 0.5):
        shape_factor: np.float32 = shape_exponent_val + np.float32(1.0)

        bed_elevation: ArrayFloat32 = self.grid.load2d(
            self.model.files["grid"]["routing/bankfull_river_elevation_m"]
        )
        if self.hydrology.waterbodies.n > 0:
            bed_elevation = self.hydrology.waterbodies.flatten_waterbody_elevations(
                bed_elevation
            )

        is_kinematic: ArrayBool = self.var.river_ids == -1
        bankfull_depth_m: ArrayFloat32 = solve_steady_backwater_network(
            compressed_seq=self._compressed_routing_seq,
            compressed_ds=self._compressed_ds,
            discharge=bankfull_discharge_m3_s,
            width=bankfull_top_width_m,
            bed_elevation=bed_elevation,
            reach_length=self.grid.var.river_length,
            manning_n=self.grid.var.river_mannings,
            shape_exponent=shape_exponent_val,
            min_depth_m=min_depth_m,
            is_kinematic=is_kinematic,
            min_kinematic_slope_m_per_m=MINIMUM_KINEMATIC_SLOPE_M_PER_M,
        )

        # The Congo River is the deepest river in the world, with maximum recorded depths
        # reaching ~220 m (and up to ~250 m in canyon sections). Calculated bankfull channel
        # depths exceeding 250 m indicate something is wrong.
        max_realistic_depth_m: np.float32 = np.float32(250.0)
        if not bool(np.all(bankfull_depth_m <= max_realistic_depth_m)):
            max_idx: tuple[int, ...] = tuple(
                int(i)
                for i in np.unravel_index(
                    int(np.nanargmax(bankfull_depth_m)), bankfull_depth_m.shape
                )
            )
            river_id: int = int(self.var.river_ids[max_idx])
            observed_width_m: float = float(
                self.var.observed_average_river_width[max_idx]
            )
            alpha_val: float = (
                float(default_alpha)
                if use_observed_width_as_bankfull or alpha is None
                else float(alpha[max_idx])
            )
            beta_val: float = (
                float(default_beta)
                if use_observed_width_as_bankfull or beta is None
                else float(beta[max_idx])
            )
            diagnostics: list[str] = [
                f"Calculated bankfull river depth exceeds realistic bounds ({max_realistic_depth_m} m; "
                f"the Congo River is the deepest river in the world at ~220-250 m).",
                f"Max depth found: {float(bankfull_depth_m[max_idx]):.2f} m.",
                "Diagnostics for river segment with highest depth:",
                f"  - River ID: {river_id}",
                f"  - Grid index: {max_idx}",
                f"  - Bankfull depth: {float(bankfull_depth_m[max_idx]):.2f} m",
                f"  - Bankfull top width (W_actual): {float(bankfull_top_width_m[max_idx]):.2f} m",
                f"  - Observed average river width: {observed_width_m}",
                f"  - Bankfull discharge (Q): {float(bankfull_discharge_m3_s[max_idx]):.2f} m3/s",
                f"  - Discharge proxy (Q_proxy): {float(q_proxy[max_idx]):.2f} m3/s",
                f"  - Bed elevation: {float(bed_elevation[max_idx]):.2f} m",
                f"  - Reach length: {float(self.grid.var.river_length[max_idx]):.2f} m",
                f"  - Manning n: {float(self.grid.var.river_mannings[max_idx]):.4f}",
                f"  - Shape factor (r + 1): {float(shape_factor):.2f}",
                f"  - Width alpha: {alpha_val:.2f}",
                f"  - Width beta: {beta_val:.2f}",
                f"  - Bankfull discharge multiplier: {float(self.bankfull_discharge_multiplier):.2f}",
                f"  - Min depth: {float(min_depth_m):.2f} m",
                f"  - Shape exponent (r): {float(shape_exponent_val):.2f}",
                f"  - use_observed_width_as_bankfull: {use_observed_width_as_bankfull}",
                f"  - use_simulated_bankfull_q: {use_simulated_bankfull_q}",
            ]
            raise ValueError("\n".join(diagnostics))

        return bankfull_depth_m

    def set_router(self) -> None:
        """Initialize the local inertial routing algorithm with derived river geometry.

        Derives cross-sectional channel geometry (bankfull width, depth via hydraulic continuity
        or geomorphic power law, shape exponent, and floodplain width) and instantiates the
        LocalInertial solver with waterbody and retention basin boundary conditions. Then seeds
        or synchronizes the solver's internal stage and storage states.

        Notes:
            Called during model setup after checkpoint loading (self.store.load)
            and waterbody initialization, but before simulation stepping begins. This ensures
            that waterbody footprints are flattened and state variables (either synthesized during
            spinup or restored from checkpoint storage) are available before deriving
            setting up the local inertial routing solver.
        """
        is_waterbody_outflow: ArrayBool = self.grid.var.waterbody_outflow_points != -1
        retention_basin_release_threshold_factor: float = self.config[
            "retention_basins"
        ]["release_threshold_factor"]

        # All cells with a valid river ID (river_ids != -1) use local inertial routing, even if steep;
        # non-river overland cells (river_ids == -1) use kinematic wave routing:
        self.use_kinematic = self.var.river_ids == -1
        use_kinematic: ArrayBool = self.use_kinematic

        # Floodplain geometry beyond bankfull channel:
        raw_profile_widths: TwoDArrayFloat32 = np.nan_to_num(
            self.grid.load3d(
                self.model.files["grid"]["routing/floodplain_profile_widths_m"],
                compress=True,
            ),
            nan=0.0,
        ).astype(np.float32)
        self.floodplain_profile_widths_m = raw_profile_widths.T
        self.floodplain_side_slope_m_per_m = np.nan_to_num(
            self.grid.load2d(
                self.model.files["grid"]["routing/floodplain_side_slope_m_per_m"]
            ),
            nan=2.0,
        ).astype(np.float32)

        # Bankfull channel width and depth (m):
        # Use simulated 2-year return period discharge if it has already been estimated;
        # otherwise, fall back to empirical downstream hydraulic geometry width inversion:
        use_simulated_bankfull_q: bool = self.has_simulated_bankfull_q
        bankfull_top_width_m: ArrayFloat32 = self.calculate_bankfull_width(
            use_simulated_bankfull_q=use_simulated_bankfull_q
        )
        self.var.river_width = bankfull_top_width_m

        # Continuous power-law channel shape exponent r (dimensionless):
        # Local inertial routing assumes a parabolic cross-section (r = 0.5), where
        # W(y) = W_bf * sqrt(y / h_bf), yielding bankfull area A_bf = (2/3) * W_bf * h_bf
        # and centerline depth h_bf = 1.5 * h_mean (Dingman 2009, Moody & Troutman 2002).
        shape_exponent_val: np.float32 = np.float32(
            self.config["river_depth"]["shape_exponent"]
        )
        self.shape_exponent = np.full(
            self.var.river_ids.shape, shape_exponent_val, dtype=np.float32
        )
        shape_exponent: ArrayFloat32 = self.shape_exponent

        bankfull_depth_m: ArrayFloat32 = self.calculate_bankfull_depth(
            bankfull_top_width_m=bankfull_top_width_m,
            use_simulated_bankfull_q=use_simulated_bankfull_q,
        )

        # Update derived bankfull width, depth, and floodplain geometry to self.var.rivers
        self._update_rivers_geometry(
            width_grid=bankfull_top_width_m,
            depth_grid=bankfull_depth_m,
            manning_grid=self.grid.var.river_mannings,
            floodplain_profile_widths_grid_m=self.floodplain_profile_widths_m,
            floodplain_side_slope_grid_m_per_m=self.floodplain_side_slope_m_per_m,
        )

        bankfull_river_elev: ArrayFloat32 = self.grid.load2d(
            self.model.files["grid"]["routing/bankfull_river_elevation_m"]
        )
        if self.hydrology.waterbodies.n > 0:
            self._wb_cell_mask: ArrayBool = self.grid.var.waterbody_ids != -1
            self._wb_cell_indices: ArrayInt32 = np.where(self._wb_cell_mask)[0].astype(
                np.int32
            )
            self._active_wb_cells: ArrayInt32 = self.grid.var.waterbody_ids[
                self._wb_cell_indices
            ]
            self._wb_cell_counts: ArrayInt32 = np.bincount(
                self._active_wb_cells, minlength=self.hydrology.waterbodies.n
            ).astype(np.int32)
            bankfull_river_elev = (
                self.hydrology.waterbodies.flatten_waterbody_elevations(
                    bankfull_river_elev
                )
            )
            waterbody_outflow_linear: ArrayInt32 = (
                self.hydrology.waterbodies.var.waterbody_outflow_linear_mapping
            )
            waterbody_outflow_bed_elev: ArrayFloat32 = bankfull_river_elev[
                waterbody_outflow_linear
            ].astype(np.float32)

            lake_a_factor: np.float32 = np.float32(
                self.model.config["hydrology"]["routing"]["lake_outflow_multiplier"]
            )
            outflow_river_width: ArrayFloat32 = bankfull_top_width_m[
                waterbody_outflow_linear
            ]
            waterbody_lake_factor: ArrayFloat32 = get_lake_factor(
                river_width=outflow_river_width,
                lake_a_factor=lake_a_factor,
            )
            average_discharge: ArrayFloat32 = np.maximum(
                self.hydrology.waterbodies.var.average_discharge_m3_per_s,
                np.float32(0.1),
            )
            waterbody_outflow_height: ArrayFloat32 = estimate_outflow_height(
                lake_capacity=self.hydrology.waterbodies.var.capacity,
                lake_factor=waterbody_lake_factor,
                lake_area=self.hydrology.waterbodies.var.lake_area,
                avg_outflow=average_discharge,
            )
            self.hydrology.waterbodies.var.outflow_height = waterbody_outflow_height
            self.hydrology.waterbodies.lake_storage = waterbody_outflow_height[
                self.hydrology.waterbodies.is_lake
            ].astype(np.float64) * self.hydrology.waterbodies.var.lake_area[
                self.hydrology.waterbodies.is_lake
            ].astype(np.float64)
        else:
            self._wb_cell_mask = np.zeros_like(self.var.river_ids, dtype=bool)
            self._wb_cell_indices = np.zeros(0, dtype=np.int32)
            self._active_wb_cells = np.zeros(0, dtype=np.int32)
            self._wb_cell_counts = np.zeros(0, dtype=np.int32)
            waterbody_outflow_bed_elev = np.zeros(0, dtype=np.float32)
            waterbody_lake_factor = np.zeros(0, dtype=np.float32)
            waterbody_outflow_height = np.zeros(0, dtype=np.float32)

        self._not_wb_cell_mask: ArrayBool = ~self._wb_cell_mask
        self._wb_potential_river_evaporation_buffer_m3 = np.zeros(
            self.hydrology.waterbodies.n, dtype=np.float32
        )
        self._wb_act_evap_buf = np.zeros(self.hydrology.waterbodies.n, dtype=np.float32)
        self._wb_outflow_rate_buf = np.zeros(
            self.hydrology.waterbodies.n, dtype=np.float32
        )
        if self.hydrology.waterbodies.n > 0:
            self._reservoir_indices = np.where(self.hydrology.waterbodies.is_reservoir)[
                0
            ].astype(np.int32)
        else:
            self._reservoir_indices = np.zeros(0, dtype=np.int32)

        self.router = LocalInertial(
            dt=3600,
            river_network=self.river_network,
            river_length=self.grid.var.river_length,
            river_width=bankfull_top_width_m,
            waterbody_ids=self.grid.var.waterbody_ids,
            river_ids=self.var.river_ids,
            is_waterbody_outflow=is_waterbody_outflow,
            retention_max_storage_m3=self.retention_max_storage_m3,
            retention_node_id=self.retention_basin_ids,
            controlled_retention=self.controlled_retention,
            retention_basin_release_threshold_factor=retention_basin_release_threshold_factor,
            bankfull_river_elevation_m=bankfull_river_elev,
            manning_n=self.grid.var.river_mannings,
            use_kinematic=use_kinematic,
            rivers_gdf=self.var.rivers,
            shape_exponent=shape_exponent,
            bankfull_depth_m=bankfull_depth_m,
            floodplain_side_slope_m_per_m=self.floodplain_side_slope_m_per_m,
            waterbody_lake_area=self.hydrology.waterbodies.var.lake_area,
            waterbody_lake_factor=waterbody_lake_factor,
            waterbody_outflow_height=waterbody_outflow_height,
            waterbody_outflow_bed_elev=waterbody_outflow_bed_elev,
            river_storage_alpha=self.var.river_storage_alpha,
            river_storage_beta=self.var.river_storage_beta,
            in_spinup=self.model.in_spinup,
            floodplain_profile_widths_m=self.floodplain_profile_widths_m,
            max_threads=self.config["max_threads"],
        )

        if self.model.in_spinup:
            # Power-law bankfull storage V_bf = length * (W_bf * h_bf / (r + 1)) (m³)
            bankfull_volume: ArrayFloat64 = self.grid.var.river_length.astype(
                np.float64
            ) * (
                (
                    bankfull_top_width_m.astype(np.float64)
                    * bankfull_depth_m.astype(np.float64)
                )
                / (shape_exponent.astype(np.float64) + 1.0)
            )
            # Power-law storage for 1 cm (0.01 m) water depth: V = V_bf * (0.01 / h_bf) ** (r + 1)
            depth_ratio: ArrayFloat64 = np.minimum(
                np.float64(0.01) / bankfull_depth_m.astype(np.float64),
                np.float64(1.0),
            )
            initial_storage: ArrayFloat64 = bankfull_volume * (
                depth_ratio ** (shape_exponent.astype(np.float64) + 1.0)
            )
            initial_storage[self.grid.var.waterbody_ids != -1] = 0.0

            self.var.river_storage_m3 = initial_storage
            self.var.discharge_in_rivers_m3_s_substep = (
                self.router.calculate_discharge_from_river_storage(
                    river_storage=self.var.river_storage_m3,
                    river_storage_alpha=self.var.river_storage_alpha,
                    river_storage_beta=self.var.river_storage_beta,
                    river_length=self.grid.var.river_length,
                    waterbody_id=self.grid.var.waterbody_ids,
                )
            )
            # Initial water stage: 1 cm (0.01 m) above bed elevation in rivers, 0 in waterbodies
            self.var.water_stage_m = np.where(
                self.grid.var.waterbody_ids == -1,
                bankfull_river_elev + np.float32(0.01),
                bankfull_river_elev,
            ).astype(np.float32)

        self.router.initialize_stage(
            water_stage_m=self.var.water_stage_m,
            river_storage_m3=self.var.river_storage_m3,
            waterbody_storage_m3=self.hydrology.waterbodies.var.storage,
            in_spinup=self.model.in_spinup,
        )

        riverbed_conductivity_multiplier: np.float32 = np.float32(
            self.config["riverbed_hydraulic_conductivity_multiplier"]
        )
        groundwater_conductivity_m_per_s: np.ndarray = (
            self.grid.var.groundwater_hydraulic_conductivity_m_per_day[0]
            / np.float32(86400.0)
        )
        self._riverbed_hydraulic_conductivity_m_per_s[:] = (
            groundwater_conductivity_m_per_s * riverbed_conductivity_multiplier
        ).astype(np.float32)

    def spinup(self) -> None:
        """Initialize routing variables during model spinup.

        Steps:
        1. Load upstream area, Manning's n, river length, and river width from grid files.
        2. Set number of routing substeps per day and kinematic wave parameter.
        3. Calculate routing step length in seconds.
        4. Compute river alpha parameter for kinematic wave routing.
        5. Initialize discharge variables and counters.

        """
        (
            self.var.rivers,
            self.var.river_ids,
            self.var.river_ids_no_waterbodies_removed,
        ) = self.load_rivers(
            grid_linear_mapping=self.grid.linear_mapping,
        )
        self.var.rivers["return_period_2_years_daily_m3_per_s"] = np.nan
        self.var.active_rivers = self.get_active_rivers()
        self._river_cell_indices = np.flatnonzero(self.var.river_ids != -1).astype(
            np.int32
        )

        self.grid.var.upstream_area = self.grid.load2d(
            self.model.files["grid"]["routing/upstream_area_m2"]
        )
        self.grid.var.upstream_area_n_cells = self.grid.load2d(
            self.model.files["grid"]["routing/upstream_area_n_cells"]
        )

        # Channel length [meters]
        self.grid.var.river_length = self.grid.load2d(
            self.model.files["grid"]["routing/river_length_m"]
        )

        assert (self.grid.var.river_length > 0).all(), (
            "Channel length must be greater than 0 for all cells"
        )

        # Channel bottom width [meters]
        self.var.observed_average_river_width = self.grid.load2d(
            self.model.files["grid"]["routing/river_width_m"]
        )

        # for a river, the wetted perimeter can be approximated by the channel width
        river_wetted_perimeter = np.where(
            ~np.isnan(self.var.observed_average_river_width),
            self.var.observed_average_river_width,
            self.default_missing_channel_width,  # Default value for missing values
        )

        # Channel gradient (fraction, dy/dx) derived dynamically from DEM bed elevation drop
        bankfull_river_elev: ArrayFloat32 = self.grid.load2d(
            self.model.files["grid"]["routing/bankfull_river_elevation_m"]
        )

        mapper: ArrayInt32 = np.full(self.river_network.size + 1, -1, dtype=np.int32)
        indices_init: ArrayInt64 = np.arange(self.river_network.size, dtype=np.int32)[
            self.river_network.mask
        ]
        mapper[indices_init] = np.arange(indices_init.size, dtype=np.int32)
        unmasked_ds: ArrayInt32 = self.river_network.idxs_ds[indices_init]
        ds_node: ArrayInt32 = mapper[unmasked_ds]

        has_ds: ArrayBool = ds_node != -1
        bed_drop: ArrayFloat32 = np.zeros_like(bankfull_river_elev, dtype=np.float32)
        bed_drop[has_ds] = (
            bankfull_river_elev[has_ds] - bankfull_river_elev[ds_node[has_ds]]
        )

        assert np.all(bed_drop[has_ds] >= np.float32(0.0)), (
            "Channel bed elevations must strictly decrease or be flat downstream along the river network."
        )

        slope: ArrayFloat32 = np.where(
            has_ds,
            bed_drop / self.grid.var.river_length,
            MINIMUM_KINEMATIC_SLOPE_M_PER_M,
        )
        self.grid.var.river_slope_m_per_m = np.maximum(
            slope, MINIMUM_KINEMATIC_SLOPE_M_PER_M
        ).astype(np.float32)
        assert np.all(self.grid.var.river_slope_m_per_m > np.float32(0.0)), (
            "All river slopes must be strictly positive."
        )

        # Channel Manning's n derived from channel slope using Jarrett (1984),
        # assuming an effective reference hydraulic radius of 0.5 m for steep upland channels
        hydraulic_radius_m: np.float32 = np.float32(0.5)
        mannings_steep: ArrayFloat32 = (
            np.float32(0.323)
            * (self.grid.var.river_slope_m_per_m ** np.float32(0.38))
            * (hydraulic_radius_m ** np.float32(-0.16))
        )
        mannings_lowland: ArrayFloat32 = np.float32(0.020) + (
            np.float32(0.040 - 0.020)
            * (self.grid.var.river_slope_m_per_m / np.float32(0.002))
        )
        raw_mannings: ArrayFloat32 = np.where(
            self.grid.var.river_slope_m_per_m >= np.float32(0.002),
            mannings_steep,
            mannings_lowland,
        )
        self.grid.var.river_mannings = (
            np.clip(raw_mannings, np.float32(0.020), np.float32(0.075))
            * np.float32(
                self.model.config["hydrology"]["routing"]["mannings_n_multiplier"]
            )
        ).astype(np.float32)
        assert (self.grid.var.river_mannings > 0).all()

        # river_storage_alpha for kinematic wave storage calculation
        # source: https://gmd.copernicus.org/articles/13/3267/2020/ eq. 21
        # It's based on Manning's n, wetted perimeter, and slope.
        # wetted perimeter is approximated by width for rivers.
        # We use a constant beta of 0.6 for Broad Sheet Flow / Manning's equation.
        river_storage_beta_constant = np.float32(0.6)
        self.var.river_storage_beta = self.grid.full_compressed(
            river_storage_beta_constant, dtype=np.float32
        )
        self.var.river_storage_alpha = (
            self.grid.var.river_mannings
            * river_wetted_perimeter ** (2 / 3)
            / np.sqrt(self.grid.var.river_slope_m_per_m)
        ) ** self.var.river_storage_beta

        # For dynamic river width, we need the average discharge. Therefore,
        # we track the sum of all discharge steps and the number of discharge steps,
        # which can be used to calculate the average discharge at each time step.
        self.var.discharge_step_count = 0
        self.var.sum_of_all_discharge_steps = self.grid.full_compressed(
            0, dtype=np.float64
        )
        (
            self.var.river_width_alpha,
            self.var.river_width_beta,
        ) = self.get_river_width_alpha_and_beta(
            default_alpha=self.config["river_width"]["default_alpha"],
            beta=self.config["river_width"]["beta"],
        )

        # Initialize discharge with zero
        self.var.discharge_in_rivers_m3_s_substep = self.grid.full_compressed(
            1e-30, dtype=np.float32
        )
        n_cells: int = self.var.discharge_in_rivers_m3_s_substep.size
        self.var.discharge_m3_s_per_substep = np.full(
            (24, n_cells),
            0,
            dtype=np.float32,
        )
        self.var.retention_basin_storage_m3_per_substep = np.full(
            (24, n_cells),
            0,
            dtype=np.float32,
        )

        # initialize retention basin storage with zero
        self.var.retention_basin_storage_m3 = np.zeros(
            len(self.retention_basin_data), dtype=np.float32
        )

        # initialize daily total retention basin water fluxes for tracking and output
        self.var.retention_inflow_m3_daily = np.zeros(
            len(self.retention_basin_data), dtype=np.float32
        )
        self.var.retention_outflow_m3_daily = np.zeros(
            len(self.retention_basin_data), dtype=np.float32
        )

        self.var.river_transmission_loss_m3 = self.grid.full_compressed(
            0, dtype=np.float32
        )

        riverbed_conductivity_multiplier: np.float32 = np.float32(
            self.config["riverbed_hydraulic_conductivity_multiplier"]
        )
        groundwater_conductivity_m_per_s: np.ndarray = (
            self.grid.var.groundwater_hydraulic_conductivity_m_per_day[0]
            / np.float32(86400.0)
        )
        self._riverbed_hydraulic_conductivity_m_per_s[:] = (
            groundwater_conductivity_m_per_s * riverbed_conductivity_multiplier
        ).astype(np.float32)

    def get_river_width_alpha_and_beta(
        self,
        beta: float,
        default_alpha: float,
    ) -> tuple[ArrayFloat32, ArrayFloat32]:
        """Calculate the river alpha parameter for the kinematic wave routing.

        For river widths where we have an observed average river width, we use the default
        values for the first year of simulation, and then calculate the river width
        based on the average river width and the discharge using the a power law

            river_width = alpha * discharge^beta

        Re-arranging for alpha gives:

            alpha = river_width / discharge^beta

        for rivers where we don't have an observed average river width, we use the default
        for alpha throughout the simulation.

        Args:
            beta: The beta parameter for the kinematic wave routing.
            default_alpha: The default alpha value to use for rivers without an observed average river width.

        Returns:
            A tuple containing:
            - alpha: The alpha parameter for the kinematic wave routing, which is a 1D array with the same shape as the grid.
            - beta_array: The beta parameter for the kinematic wave routing, which is a 1D array with the same shape as the grid.
        """
        # for all rivers we use the default beta value.
        beta_array: ArrayFloat32 = np.full_like(
            self.var.observed_average_river_width, beta, dtype=np.float32
        )

        # for the first year of simulation, we use the default alpha value for all rivers
        if self.var.discharge_step_count < 365 * 24:
            alpha: ArrayFloat32 = np.full_like(
                self.var.observed_average_river_width,
                default_alpha,
                dtype=np.float32,
            )
        # after the first year, we calculate the alpha value based on the observed average river width and the discharge
        else:
            average_discharge: ArrayFloat64 = (
                self.var.sum_of_all_discharge_steps / (self.var.discharge_step_count)
            ).astype(np.float64)
            # re-arranged formula for alpha, where we use the observed average river width and the average discharge to calculate alpha
            alpha: ArrayFloat32 = np.full_like(
                self.var.observed_average_river_width,
                default_alpha,
                dtype=np.float32,
            )  # default alpha everywhere
            calculate_alpha = (
                (~np.isnan(self.var.observed_average_river_width))
                & (self.grid.var.waterbody_ids == -1)
                & (average_discharge > 0)
            )  # decide where alpha should be calculated
            alpha[calculate_alpha] = self.var.observed_average_river_width[
                calculate_alpha
            ] / (
                np.maximum(average_discharge[calculate_alpha], np.float32(1e-6))
                ** beta_array[calculate_alpha]
            )  # calculate alpha

        return alpha, beta_array

    def step(
        self,
        total_runoff_m: TwoDArrayFloat32,
        channel_abstraction_m3: ArrayFloat32,
        return_flow: ArrayFloat32,
        reference_evapotranspiration_water_m: TwoDArrayFloat32,
    ) -> tuple[
        np.float64,
        np.float64,
        np.float64,
    ]:
        """Perform a daily routing step with multiple substeps.

        Args:
            total_runoff_m: Total runoff in meters for each grid cell for each hour.
                Shape is (24, n_cells).
            channel_abstraction_m3: Channel abstraction in m3 for each grid cell over the whole day.
            return_flow: Return flow in meters for each grid cell over the whole day.
            reference_evapotranspiration_water_m: Reference evapotranspiration from water in meters for for each grid cell for each hour.

        Returns:
            A tuple containing:
            - Total routing loss, including outflow at pits, evaporation in rivers and water bodies,
            - Total over abstraction in m3. This should be zero if the abstraction is within the available storage.
                Otherwise, it indicates the amount of abstraction that could not be met and indicates an error
                in the model.

        Raises:
            ValueError: If inflow is added to waterbody cells.
        """
        if __debug__:
            pre_waterbody_storage: np.ndarray = (
                self.hydrology.waterbodies.var.storage.copy()
            )
            pre_river_storage_m3: ArrayFloat64 = self.var.river_storage_m3.copy()
            pre_retention_storage_m3: ArrayFloat32 = (
                self.var.retention_basin_storage_m3.copy()
            )

        channel_abstraction_m3_per_hour: ArrayFloat32 = (
            channel_abstraction_m3 / 24
        ).astype(np.float32)
        if __debug__:
            assert (channel_abstraction_m3_per_hour[self._wb_cell_mask] == 0.0).all(), (
                "Channel abstraction must be zero for water bodies, "
                "but found non-zero value."
            )

        return_flow_m3_per_hour: ArrayFloat32 = (
            return_flow * self.grid.var.cell_area / 24
        ).astype(np.float32)

        # add return flow to the water bodies
        if len(self._active_wb_cells) > 0:
            return_flow_m3_to_waterbodies_per_hour: np.ndarray = np.bincount(
                self._active_wb_cells,
                weights=return_flow_m3_per_hour[self._wb_cell_indices],
                minlength=self.hydrology.waterbodies.n,
            )
            return_flow_m3_per_hour[self._wb_cell_mask] = 0.0
        else:
            return_flow_m3_to_waterbodies_per_hour = np.zeros(0, dtype=np.float32)

        net_base_sideflow_per_hour: ArrayFloat32 = (
            return_flow_m3_per_hour - channel_abstraction_m3_per_hour
        )

        self.var.discharge_m3_s_per_substep.fill(np.nan)
        self.var.retention_basin_storage_m3_per_substep.fill(np.nan)

        if __debug__:
            # these are for balance checks, the sum of all routing steps
            evaporation_in_rivers_m3: ArrayFloat32 = self._evaporation_in_rivers_m3
            evaporation_in_rivers_m3.fill(0.0)
            waterbody_evaporation_m3: ArrayFloat32 = np.zeros(
                self.hydrology.waterbodies.n, dtype=np.float32
            )
            outflow_at_pits_m3 = np.float32(0)
            command_area_release_m3 = np.float32(0)
            total_inflow_m3: np.float64 = np.float64(0)

            # Initialize retention flows (as 0s); they will later be accumulated over 24h
            retention_inflow_m3: ArrayFloat32 = np.zeros_like(
                self.var.retention_basin_storage_m3, dtype=np.float32
            )
            retention_outflow_m3: ArrayFloat32 = np.zeros_like(
                self.var.retention_basin_storage_m3, dtype=np.float32
            )
            retention_evaporation_m3: ArrayFloat32 = np.zeros_like(
                self.var.retention_basin_storage_m3, dtype=np.float32
            )

        over_abstraction_m3: ArrayFloat32 = self._over_abstraction_daily
        over_abstraction_m3.fill(0.0)
        self.var.river_transmission_loss_m3.fill(0.0)

        self._groundwater_head_top_layer_m[:] = self.grid.var.heads[0].astype(
            np.float32
        )

        # update alpha and beta once per day during spinup or if river geometry is not fixed
        if self.model.in_spinup or not self.fix_river_geometry_after_spinup:
            (
                self.var.river_width_alpha,
                self.var.river_width_beta,
            ) = self.get_river_width_alpha_and_beta(
                default_alpha=self.config["river_width"]["default_alpha"],
                beta=self.config["river_width"]["beta"],
            )
            if self.has_simulated_bankfull_q:
                dynamic_width: ArrayFloat32 = self.calculate_bankfull_width(
                    use_simulated_bankfull_q=True
                )
                self.var.river_width = dynamic_width
                dynamic_depth_m: ArrayFloat32 = self.calculate_bankfull_depth(
                    bankfull_top_width_m=dynamic_width,
                    use_simulated_bankfull_q=True,
                )
                self.router.update_channel_geometry(dynamic_width, dynamic_depth_m)

        for hour in range(24):
            # increment inflow index for next hour
            self.inflow_idx += 1

            np.multiply(
                total_runoff_m[hour, :],
                self.grid.var.cell_area,
                out=self._side_flow_buffer_m3,
            )

            # then split the runoff into runoff directly to water bodies,
            # calculate waterbody evaporation, and route remaining runoff to channel network
            if len(self._active_wb_cells) > 0:
                _accumulate_waterbody_runoff_and_evaporation(
                    active_wb_cells=self._active_wb_cells,
                    wb_cell_indices=self._wb_cell_indices,
                    side_flow=self._side_flow_buffer_m3,
                    ref_et=reference_evapotranspiration_water_m[hour],
                    wb_cell_counts=self._wb_cell_counts,
                    lake_area=self.hydrology.waterbodies.var.lake_area,
                    wb_storage=self.hydrology.waterbodies.var.storage,
                    return_flow_m3_to_wb=return_flow_m3_to_waterbodies_per_hour,
                    wb_potential_river_evaporation_buffer_m3=self._wb_potential_river_evaporation_buffer_m3,
                    wb_act_evap_buf=self._wb_act_evap_buf,
                )

            # Calculate potential evaporation for retention basins
            if not self.retention_basin_data.empty:
                _compute_retention_basin_evaporation(
                    retention_cell_indices=self._retention_cell_indices,
                    retention_cell_basin_ids=self._retention_cell_basin_ids,
                    retention_basin_area=self._retention_basin_area,
                    ref_et=reference_evapotranspiration_water_m[hour],
                    retention_storage=self.var.retention_basin_storage_m3,
                    pot_evap_buf=self._pot_evap_ret_buf,
                    act_evap_buf=self._retention_act_evap_buf,
                )
                if __debug__:
                    retention_evaporation_m3 += self._retention_act_evap_buf

            outflow_per_waterbody_m3, command_area_release_m3_routing_step = (
                self.hydrology.waterbodies.substep(
                    current_substep=hour,
                    n_routing_substeps=24,
                    routing_step_length_seconds=3600,
                )
            )

            self.hydrology.waterbodies.var.storage -= (
                command_area_release_m3_routing_step
            )

            if __debug__:
                valid_prescribed_outflow = ~np.isnan(outflow_per_waterbody_m3)
                if valid_prescribed_outflow.any():
                    assert (
                        outflow_per_waterbody_m3[valid_prescribed_outflow]
                        <= self.hydrology.waterbodies.var.storage[
                            valid_prescribed_outflow
                        ].astype(np.float32)
                    ).all(), (
                        "Prescribed reservoir outflow cannot be greater than storage"
                    )

            self._side_flow_buffer_m3 += net_base_sideflow_per_hour
            if __debug__:
                assert (self._side_flow_buffer_m3[self._wb_cell_mask] == 0).all()

            for cell_index, inflow in self._inflow_list:
                inflow_m3 = inflow[self.inflow_idx] * np.float32(3600)
                self._side_flow_buffer_m3[cell_index] += inflow_m3
                if __debug__:
                    total_inflow_m3 += inflow_m3

            if __debug__:
                assert not np.isnan(
                    self.var.discharge_in_rivers_m3_s_substep[self._not_wb_cell_mask]
                ).all()

            calculate_wetted_channel_area(
                use_kinematic=self.use_kinematic,
                water_stage_m=self.var.water_stage_m,
                bed_elevation_m=self.router.bed_elevation,
                bankfull_width_m=self.router.river_width,
                bankfull_depth_m=self.router.bankfull_depth,
                shape_exponent=self.shape_exponent,
                floodplain_profile_widths_m=self.floodplain_profile_widths_m,
                floodplain_side_slope_m_per_m=self.floodplain_side_slope_m_per_m,
                river_width_alpha=self.var.river_width_alpha,
                river_width_beta=self.var.river_width_beta,
                discharge_m3_s=self.var.discharge_in_rivers_m3_s_substep,
                river_length_m=self.grid.var.river_length,
                cell_area_m2=self.grid.var.cell_area,
                is_waterbody=self._wb_cell_mask,
                wetted_channel_area_m2=self._wetted_channel_area_buffer_m2,
            )

            calculate_potential_river_evaporation(
                reference_evapotranspiration_m=reference_evapotranspiration_water_m[
                    hour
                ],
                wetted_channel_area_m2=self._wetted_channel_area_buffer_m2,
                potential_evaporation_m3=self._potential_river_evaporation_buffer_m3,
            )

            calculate_potential_river_transmission_loss(
                wetted_channel_area_m2=self._wetted_channel_area_buffer_m2,
                riverbed_hydraulic_conductivity_m_s=self._riverbed_hydraulic_conductivity_m_per_s,
                bed_elevation_m=self.router.bed_elevation,
                water_stage_m=self.var.water_stage_m,
                groundwater_head_m=self._groundwater_head_top_layer_m,
                timestep_s=np.float32(3600),
                potential_transmission_loss_m3=self._potential_transmission_loss_buffer_m3,
            )

            (
                self.var.discharge_in_rivers_m3_s_substep,
                self.var.river_storage_m3,
                actual_evaporation_in_rivers_m3_per_hour,
                actual_transmission_loss_m3_per_hour,
                over_abstraction_m3_routing_step,
                self.hydrology.waterbodies.var.storage,
                waterbody_inflow_m3,
                outflow_at_pits_m3_routing_step,
                self.var.retention_basin_storage_m3,
                retention_inflow_m3_hour,
                retention_outflow_m3_hour,
            ) = self.router.step(
                Q_prev_m3_s=self.var.discharge_in_rivers_m3_s_substep,
                river_storage_m3=self.var.river_storage_m3,
                sideflow_m3=self._side_flow_buffer_m3,
                evaporation_m3=self._potential_river_evaporation_buffer_m3,
                transmission_loss_m3=self._potential_transmission_loss_buffer_m3,
                waterbody_storage_m3=self.hydrology.waterbodies.var.storage,
                outflow_per_waterbody_m3=outflow_per_waterbody_m3,
                retention_storage_m3=self.var.retention_basin_storage_m3,
                retention_activation_threshold_m3_s=self.retention_activation_threshold_m3_s,
            )
            self.var.water_stage_m = self.router.get_water_stage(
                out=self.var.water_stage_m
            )

            self.var.river_transmission_loss_m3 += actual_transmission_loss_m3_per_hour

            if __debug__:
                if not (actual_evaporation_in_rivers_m3_per_hour >= 0.0).all():
                    raise ValueError(
                        f"Negative evaporation detected, min evaporation: {actual_evaporation_in_rivers_m3_per_hour.min()}, min discharge: {self.var.discharge_in_rivers_m3_s_substep.min()}. "
                    )
                assert (actual_evaporation_in_rivers_m3_per_hour >= 0.0).all()

            # the reservoir operators need to track the inflow to the reservoirs
            if len(self._reservoir_indices) > 0:
                self.model.agents.reservoir_operators.track_inflow(
                    waterbody_inflow_m3[self._reservoir_indices]
                )

            if __debug__:
                # ensure that discharge is nan for water bodies
                assert np.isnan(
                    self.var.discharge_in_rivers_m3_s_substep[self._wb_cell_mask]
                ).all()

                # ensure that discharge is not nan for river cells
                assert not np.isnan(
                    self.var.discharge_in_rivers_m3_s_substep[self._not_wb_cell_mask]
                ).any()

            discharge_row: ArrayFloat32 = self.var.discharge_m3_s_per_substep[hour, :]
            discharge_row[:] = self.var.discharge_in_rivers_m3_s_substep

            # check if discharge is not higher than the highest discharge ever recorded
            if np.nanmax(discharge_row) > 400_000:
                raise ValueError(
                    f"Discharge is higher than the highest discharge ever recorded. Max discharge: {np.nanmax(discharge_row)}"
                )

            # set waterbody outflow points to the outflow of the waterbody and fill waterbody reaches
            if self.hydrology.waterbodies.n > 0:
                np.divide(
                    outflow_per_waterbody_m3,
                    np.float32(3600),
                    out=self._wb_outflow_rate_buf,
                )
                self.hydrology.waterbodies.map_to_grid_outflow(
                    self._wb_outflow_rate_buf, out=discharge_row
                )
                fill_discharge_in_waterbodies(
                    discharge_m3_s=discharge_row,
                    upstream_matrix_from_up_to_downstream=self.router.upstream_matrix_from_up_to_downstream,
                    idxs_up_to_downstream=self.router.idxs_up_to_downstream,
                )

            if __debug__:
                # after filling the gaps, we should not have any nans in the river cells
                assert not np.isnan(discharge_row[self._river_cell_indices]).any()

            ret_row: ArrayFloat32 = self.var.retention_basin_storage_m3_per_substep[
                hour, :
            ]
            ret_row.fill(0.0)
            if not self.retention_basin_data.empty:
                ret_row[self._retention_cell_indices] = (
                    self.var.retention_basin_storage_m3[self._retention_cell_basin_ids]
                )

            if self.model.in_spinup:
                self.var.sum_of_all_discharge_steps += discharge_row
                self.var.discharge_step_count += 1

            if __debug__:
                assert (
                    self.router.get_available_storage(
                        self.var.discharge_in_rivers_m3_s_substep,
                        self.var.river_storage_alpha,
                        self.var.river_storage_beta,
                    )
                    >= 0.0
                ).all()
                # Discharge at outlets and lakes and reservoirs
                outflow_at_pits_m3 += outflow_at_pits_m3_routing_step
                waterbody_evaporation_m3 += self._wb_act_evap_buf
                evaporation_in_rivers_m3 += actual_evaporation_in_rivers_m3_per_hour
                over_abstraction_m3 += over_abstraction_m3_routing_step
                command_area_release_m3 += command_area_release_m3_routing_step
                # Accumulate retention flows across all hourly timesteps for the day
                retention_inflow_m3 += retention_inflow_m3_hour
                retention_outflow_m3 += retention_outflow_m3_hour

        np.mean(
            self.var.discharge_m3_s_per_substep,
            axis=0,
            out=self.var.discharge_m3_s,
        )

        # Update return periods on January 1st of each year, with the exception when the first
        # day of the year is the start of the spinup period. In that case, we don't have any
        # data to fit with yet.
        if self.model.current_day_of_year == 1 and (
            self.model.current_timestep > 0 or not self.model.in_spinup
        ):
            self.update_return_periods()

        if __debug__:
            balance_check(
                how="sum",
                influxes=[
                    total_runoff_m.sum(axis=0) * self.grid.var.cell_area,
                    return_flow * self.grid.var.cell_area,
                    over_abstraction_m3,
                    total_inflow_m3,
                ],
                outfluxes=[
                    channel_abstraction_m3,
                    outflow_at_pits_m3,
                    evaporation_in_rivers_m3,
                    waterbody_evaporation_m3,
                    retention_evaporation_m3,
                    command_area_release_m3,
                    self.var.river_transmission_loss_m3,
                ],
                prestorages=[
                    pre_waterbody_storage,
                    pre_river_storage_m3,
                    pre_retention_storage_m3,
                ],
                poststorages=[
                    self.hydrology.waterbodies.var.storage,
                    self.var.river_storage_m3,
                    self.var.retention_basin_storage_m3,
                ],
                name="routing_1",
                tolerance=self.var.river_storage_m3.size * 0.01,
            )
            total_evaporation_in_rivers_m3: np.float64 = (
                evaporation_in_rivers_m3.astype(np.float64).sum()
            )
            total_waterbody_evaporation_m3: np.float64 = (
                waterbody_evaporation_m3.astype(np.float64).sum()
            )
            total_outflow_at_pits_m3: np.float64 = outflow_at_pits_m3.astype(
                np.float64
            ).sum()
            total_retention_evaporation_m3: np.float64 = (
                retention_evaporation_m3.astype(np.float64).sum()
            )

            assert total_evaporation_in_rivers_m3 >= 0
            assert total_waterbody_evaporation_m3 >= 0
            assert total_outflow_at_pits_m3 >= 0
            assert total_retention_evaporation_m3 >= 0

            routing_loss: np.float64 = (
                total_evaporation_in_rivers_m3
                + total_waterbody_evaporation_m3
                + total_outflow_at_pits_m3
                + total_retention_evaporation_m3
            )

            assert routing_loss >= 0, "Routing loss cannot be negative"

        # outside debug, we return NaN for routing loss
        else:
            routing_loss: np.float64 = np.float64(np.nan)
            total_inflow_m3 = np.float64(np.nan)
            total_evaporation_in_rivers_m3: np.float64 = np.float64(np.nan)
            total_waterbody_evaporation_m3: np.float64 = np.float64(np.nan)
            total_outflow_at_pits_m3: np.float64 = np.float64(np.nan)
            total_retention_evaporation_m3: np.float64 = np.float64(np.nan)

        # store daily retention basin flows
        if __debug__:
            self.var.retention_inflow_m3_daily = retention_inflow_m3
            self.var.retention_outflow_m3_daily = retention_outflow_m3

        self.report(locals())

        total_over_abstraction_m3: np.float64 = over_abstraction_m3.astype(
            np.float64
        ).sum()
        if over_abstraction_m3.sum() > self.var.river_storage_m3.size * 0.01:
            self.model.logger.warning(
                f"Total over-abstraction in routing step is {total_over_abstraction_m3:.2f} m³"
            )

        return total_inflow_m3, routing_loss, total_over_abstraction_m3

    @property
    def name(self) -> str:
        """Name of the module."""
        return "hydrology.routing"

    @property
    def outflow_rivers(self) -> gpd.GeoDataFrame:
        """Get the outflow rivers.

        Returns:
            A GeoDataFrame containing the outflow rivers.
        """
        rivers: gpd.GeoDataFrame = self.var.rivers
        rivers = rivers[~rivers["is_downstream_outflow"]]
        rivers = rivers[~rivers["is_further_downstream_outflow"]]
        outflow_rivers: gpd.GeoDataFrame = rivers[
            ~rivers["downstream_ID"].isin(rivers.index)
        ]
        return outflow_rivers

    def get_active_rivers(
        self, include_rivers_not_represented_in_grid: bool = False
    ) -> gpd.GeoDataFrame:
        """Get the rivers that are simulated (i.e., not downstream of the model region).

        Args:
            include_rivers_not_represented_in_grid: Whether to include rivers that are
                not represented in the grid (e.g., for reporting river geometry).

        Returns:
            A GeoDataFrame containing the active rivers.
        """
        return select_active_rivers(
            self.var.rivers,
            include_rivers_not_represented_in_grid=include_rivers_not_represented_in_grid,
        )

    def get_active_and_downstream_outflow_rivers(self) -> gpd.GeoDataFrame:
        """Get the rivers that are simulated (i.e., not downstream of the model region) and the downstream outflow rivers.

        Returns:
            A GeoDataFrame containing the active rivers and the downstream outflow rivers.
        """
        rivers: gpd.GeoDataFrame = self.var.rivers
        active_and_downstream_outflow_rivers = rivers[
            ~rivers["is_further_downstream_outflow"]
        ]
        return active_and_downstream_outflow_rivers.copy()

    def update_return_periods(self) -> None:
        """Update flood return periods and hydraulic geometry for the routing module.

        Estimates the 2-year flood return period discharge (and retention basin
        activation threshold discharge) for each active river segment using extreme
        value analysis on daily simulated river outflow. If the model is in spinup or
        if fix_river_geometry_after_spinup is False, dynamically updates bankfull
        channel width, bankfull depth, and the local inertial router geometry.

        Raises:
            ValueError: If a retention basin has no associated grid cells.
        """
        activation_threshold_return_period_years: float = self.config[
            "retention_basins"
        ]["activation_threshold_return_period_years"]

        active_rivers: gpd.GeoDataFrame = self.get_active_rivers()
        discharge_by_river: pd.DataFrame
        if self.model.in_spinup:
            discharge_by_river = get_discharge_per_river(
                rivers=active_rivers,
                all_rivers=self.var.rivers,
                source="memory",
                variables_to_report=self.variables_to_report,
            )
        else:
            folder: Path = (
                self.model.report_folder.parent.parent
                / self.model.config["spinup_name"]
                / "report"
                / "hydrology.routing"
            )
            discharge_by_river = get_discharge_per_river(
                rivers=active_rivers,
                all_rivers=self.var.rivers,
                source="file",
                folder=folder,
            )

            if self.model.current_timestep > 0:
                sample_key: str = (
                    f"river_outflow_hourly_m3_per_s_{active_rivers.index[0]}"
                )
                if (
                    self.variables_to_report is not None
                    and sample_key in self.variables_to_report
                    and self.variables_to_report[sample_key]["_var_index"] > 0
                ):
                    discharge_by_river_run: pd.DataFrame = get_discharge_per_river(
                        rivers=active_rivers,
                        all_rivers=self.var.rivers,
                        source="memory",
                        variables_to_report=self.variables_to_report,
                    )

                    discharge_by_river = pd.concat(
                        [discharge_by_river, discharge_by_river_run], axis=0
                    )

        discharge_by_river_daily: pd.DataFrame = discharge_by_river.resample(
            "D", label="left"
        ).mean()

        discharge_by_river_daily.index.freq = pd.infer_freq(  # ty:ignore[unresolved-attribute]
            discharge_by_river_daily.index  # ty:ignore[invalid-argument-type]
        )
        return_periods_to_fit: list[int | float] = list(
            set(
                [2, activation_threshold_return_period_years]
                if (self.model.in_spinup or not self.fix_river_geometry_after_spinup)
                else [activation_threshold_return_period_years]
            )
        )
        for idx in tqdm(
            active_rivers.index,
            total=len(active_rivers),
            desc="Return period estimation",
        ):
            # Ideally we want to do "full estimation" of the extreme value distribution,
            # which includes selection of the ideal threshold. However, this requires
            # computationally expensive bootstrapping. For the bankful discharge, however,
            # this is not needed as the RP (currently 2 years) is low, and a good estimate
            # can be obtained by fixing the quantile and the shape.
            # For 95th quantile, see for example: https://doi.org/10.1002/wrcr.20381
            # For mathematical stability, only 1 exceedance is required because the shape
            # is fixed. However, in practice when a normal spinup is used (>= 10 years)
            # there should be more exceedances and a reliable fit for the low RPs used here.
            model: ReturnPeriodModel = ReturnPeriodModel(
                series=discharge_by_river_daily[idx],
                return_periods=return_periods_to_fit,
                fixed_quantile=0.95,  # use a fixed quantile
                min_exceed=1,
                fixed_shape=0.0,
                fixed_scale=None,
            )

            for return_period, return_water_level in model.rl_table.set_index(
                "T_years"
            )["GPD_POT_RL"].items():
                self.var.rivers.loc[
                    idx, f"return_period_{return_period}_years_daily_m3_per_s"
                ] = max(
                    float(return_water_level),
                    self.default_minimum_discharge_m3_per_s,
                )

        # Update river geometry only during spinup or if fix_river_geometry_after_spinup is False
        if self.model.in_spinup or not self.fix_river_geometry_after_spinup:
            # For channels without observed width, estimate bankfull width from the newly fitted 2-year return period discharge
            # using downstream hydraulic geometry W = alpha * Q2^beta:
            self.var.river_width = self.calculate_bankfull_width(
                use_simulated_bankfull_q=True
            )

            # Dynamically update the local inertial router's bankfull depth in place
            # using the newly fitted 2-year flood return periods:
            updated_bankfull_depth: ArrayFloat32 = self.calculate_bankfull_depth(
                bankfull_top_width_m=self.var.river_width,
                use_simulated_bankfull_q=True,
            )
            self.router.update_channel_geometry(
                river_width=self.var.river_width,
                bankfull_depth_m=updated_bankfull_depth,
            )
            self._update_rivers_geometry(
                width_grid=self.var.river_width,
                depth_grid=updated_bankfull_depth,
                manning_grid=self.grid.var.river_mannings,
                floodplain_profile_widths_grid_m=self.floodplain_profile_widths_m,
                floodplain_side_slope_grid_m_per_m=self.floodplain_side_slope_m_per_m,
            )

        for basin_id in range(len(self.retention_basin_data)):
            basin_cells = np.where(self.retention_basin_ids == basin_id)[0]
            if len(basin_cells) == 0:
                raise ValueError(
                    f"Retention basin {basin_id} has no associated grid cells."
                )
            river_id = self.var.river_ids[basin_cells[0]]
            if river_id == -1:
                warnings.warn(
                    f"Retention basin {basin_id} is not associated with any river."
                )
                self.retention_activation_threshold_m3_s[basin_id] = np.inf
            else:
                if not self.retention_basin_is_active[basin_id]:
                    self.retention_activation_threshold_m3_s[basin_id] = np.inf
                else:
                    return_period_col = f"return_period_{activation_threshold_return_period_years}_years_daily_m3_per_s"
                    self.retention_activation_threshold_m3_s[basin_id] = max(
                        float(self.var.rivers.loc[river_id, return_period_col]),
                        self.default_minimum_discharge_m3_per_s,
                    )

        return None
