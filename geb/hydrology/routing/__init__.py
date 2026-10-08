"""Routing algorithms for river networks."""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
from affine import Affine
from numba import njit
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

from .kinematic import update_node_kinematic as update_node_kinematic
from .local_inertial import LocalInertial as LocalInertial

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


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
    if source == "file" and folder is None:
        raise ValueError("folder must be provided if source is 'file'")
    elif source == "memory" and variables_to_report is None:
        raise ValueError("variables_to_report must be provided if source is 'memory'")

    def create_df_from_report_variable(
        river_id: int | str, variables_to_report: dict[str, Any]
    ) -> pd.Series:
        river_data = variables_to_report[f"river_outflow_hourly_m3_per_s_{river_id}"]
        return pd.Series(
            river_data["_data_array"][: river_data["_var_index"]],
            index=river_data["_time_array"][: river_data["_var_index"]].astype(
                "datetime64[s]"
            ),
        )

    consolidated_table: pd.DataFrame | None = None
    if source == "file":
        assert folder is not None
        consolidated_file: Path = folder / "river_outflow_hourly_m3_per_s.parquet"
        if consolidated_file.exists():
            consolidated_table = read_table(consolidated_file)

    def read_outflow_series_from_file(key: int | str) -> pd.Series:
        assert folder is not None
        key_str: str = str(key)
        if consolidated_table is not None and key_str in consolidated_table.columns:
            return consolidated_table[key_str]
        # Legacy fallback: individual parquet file
        return read_table(folder / f"river_outflow_hourly_m3_per_s_{key}.parquet")[
            f"river_outflow_hourly_m3_per_s_{key}"
        ]

    discharge_data = {}
    for river_id in rivers.index:
        assert isinstance(river_id, int)
        xys: list[tuple[int, int]] = get_river_representative_xys(river_id, all_rivers)
        if len(xys) == 1:
            if source == "file":
                discharge_data[river_id] = read_outflow_series_from_file(river_id)
            else:
                assert variables_to_report is not None
                discharge_data[river_id] = create_df_from_report_variable(
                    river_id, variables_to_report
                )
        else:
            total_discharge_part = None
            for i in range(len(xys)):
                if source == "file":
                    discharge_part = read_outflow_series_from_file(f"{river_id}_{i}")
                else:
                    assert variables_to_report is not None
                    discharge_part = create_df_from_report_variable(
                        f"{river_id}_{i}", variables_to_report
                    )
                if total_discharge_part is None:
                    total_discharge_part = discharge_part
                else:
                    total_discharge_part += discharge_part
            discharge_data[river_id] = total_discharge_part

    if not discharge_data:
        return pd.DataFrame()

    return pd.concat(discharge_data, axis=1)


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


def get_channel_ratio(
    river_width: ArrayFloat32,
    river_length: ArrayFloat32,
    cell_area: ArrayFloat32,
) -> ArrayFloat32:
    """Calculate the ratio of the river channel area to the cell area.

    Args:
        river_width: The width of the river in each cell, in meters.
        river_length: The length of the river in each cell, in meters.
        cell_area: The area of each cell, in square meters.

    Returns:
        A 1D array with the ratio of the river channel area to the cell area.
    """
    channel_ratio: ArrayFloat32 = np.minimum(
        1.0,
        river_width * river_length / cell_area,
    )

    assert not np.isnan(channel_ratio).any()
    return channel_ratio


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
    up_map: dict[int, list[int]] = {}
    for u, d in enumerate(idxs_ds):
        if d != -1 and d != u:
            d_int: int = int(d)
            if d_int not in up_map:
                up_map[d_int] = []
            up_map[d_int].append(int(u))

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

    def __init__(self, model: GEBModel, hydrology: Hydrology) -> None:
        """Initialize the Routing module.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology submodel instance.
        """  # noqa: DOC502
        super().__init__(model)

        self.config = model.config["hydrology"]["routing"]

        self.default_missing_channel_width: float = 0.5  # 0.5 is based on assumption here, but could be improved: https://pubs.usgs.gov/publication/wri854004
        self.default_minimum_discharge_m3_per_s: float = 0.01

        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        self.ldd: ArrayUint8 = self.grid.load2d(
            self.model.files["grid"]["routing/ldd"],
        )

        self.load_weirs()

        mask: TwoDArrayBool = ~self.grid.mask

        ldd_uncompressed: TwoDArrayUint8 = np.full_like(mask, 255, dtype=self.ldd.dtype)
        ldd_uncompressed[mask] = self.ldd.ravel()

        self.river_network: pyflwdir.FlwdirRaster = create_river_network(
            ldd_uncompressed=ldd_uncompressed, mask=mask, transform=self.grid.transform
        )

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
        ].to_numpy(dtype=np.float32)
        if np.isnan(self.retention_max_storage_m3).any():
            self.model.logger.warning(
                "Retention basin data contains NaN values in 'retention_max_storage_m3'. "
                "These will be treated as 0.0 m3. Please check your input data."
            )
        self.retention_max_storage_m3[np.isnan(self.retention_max_storage_m3)] = 0.0

        self.controlled_retention = self.retention_basin_data[
            "controlled_retention"
        ].to_numpy(dtype=bool)

        self.retention_activation_threshold_m3_s = np.full(
            len(self.retention_basin_data), np.inf, dtype=np.float32
        )
        self.retention_basin_is_active = self.retention_basin_data["active"]

        self.inflow = {}
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
                self.inflow[(y, x)] = inflow.to_numpy(dtype=np.float32)

            assert self.model.current_time == inflow_per_location.index[0]

        self.var.retention_basin_storage_m3_per_substep = np.full(
            (24, mask.size),
            0,
            dtype=np.float32,
        )

        if self.model.in_spinup:
            self.spinup()

    def load_weirs(self) -> None:
        """Load barrier heights and which AMBER dams can open.

        Notes:
            With routing.weirs disabled, input files are not read or modified.
            Lakes and reservoirs keep their own flow rules.

        Returns:
            None. Sets weir heights (m) and instream-dam flags in grid cell order.
        """  # noqa: DOC202
        self.instream_dam: ArrayBool = np.zeros_like(self.ldd, dtype=bool)
        if not self.config.get("weirs", True):
            # Zero heights remove the raised river sills from routing.
            self.weir_height_m: ArrayFloat32 = np.zeros_like(self.ldd, dtype=np.float32)
            return

        self.weir_height_m = self.grid.load2d(
            self.model.files["grid"]["routing/weir_height_m"]
        )
        dam_grid: Path | None = self.model.files["grid"].get("routing/instream_dam")
        if dam_grid is None:
            # Read the old grid name too, so existing builds still work.
            dam_grid = self.model.files["grid"].get("routing/gated_dam")
        if dam_grid is not None:
            self.instream_dam = self.grid.load2d(dam_grid)

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

        def remove_masked_river_cells(
            xys: list[tuple[int, int]], mask: list[bool]
        ) -> np.ndarray[tuple[int], np.dtype[Any]]:
            array: np.ndarray[tuple[int], np.dtype[Any]] = np.empty(
                sum(mask), dtype=object
            )
            i: int = 0
            for xy, m in zip(xys, mask):
                if m:
                    array[i] = np.array(xy, dtype=object)
                    i += 1
            return array

        rivers["hydrography_xy"] = [
            remove_masked_river_cells(xys, mask)
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

    def save_weirs(self) -> None:
        """Save barrier heights (m) and which dams can open to weir_heights.csv.

        Returns:
            None.

        Raises:
            OSError: If the output file cannot be written.
        """  # noqa: DOC202, DOC502
        heights: ArrayFloat32 = self.router._weir_height_inertial
        weir_mask: ArrayBool = heights > 0
        cells: ArrayInt32 = self.router._inertial_cells[weir_mask]
        input_heights: ArrayFloat32 = self.weir_height_m[cells]
        locations: TwoDArrayFloat32 = self.grid.lonlat[cells]
        weirs: pd.DataFrame = pd.DataFrame(
            {
                "grid_cell_index": cells,
                "longitude_deg": locations[:, 0],
                "latitude_deg": locations[:, 1],
                "river_id": self.var.river_ids[cells],
                "input_height_m": input_heights,
                "effective_height_m": heights[weir_mask],
                "instream_dam": self.router._instream_dam_inertial[weir_mask],
                "bankfull_depth_m": self.router.bankfull_depth[cells],
            }
        )
        weirs.to_csv(self.model.output_folder / "weir_heights.csv", index=False)

    def _update_rivers_geometry(
        self,
        width_grid: ArrayFloat32,
        depth_grid: ArrayFloat32,
        manning_grid: ArrayFloat32,
    ) -> None:
        """Update bankfull width, depth, and Manning's n on river reaches GeoDataFrame from routing grid arrays.

        Aggregates bankfull width (meters), depth (meters), and Manning roughness across all grid cells
        belonging to each river reach, assigning reach-average values directly
        to self.var.rivers['width'], self.var.rivers['depth'], and self.var.rivers['manning'].

        Args:
            width_grid: Bankfull channel width per routing grid cell (meters).
            depth_grid: Bankfull channel depth per routing grid cell (meters).
            manning_grid: Channel Manning roughness per routing grid cell (s/m^(1/3)).
        """
        valid_river_cell_mask: ArrayBool = self.var.river_ids != -1
        valid_reach_ids: ArrayInt32 = self.var.river_ids[valid_river_cell_mask]
        valid_widths: ArrayFloat32 = width_grid[valid_river_cell_mask]
        valid_depths: ArrayFloat32 = depth_grid[valid_river_cell_mask]
        valid_mannings: ArrayFloat32 = manning_grid[valid_river_cell_mask]

        df_grid: pd.DataFrame = pd.DataFrame(
            {
                "river_id": valid_reach_ids,
                "width": valid_widths,
                "depth": valid_depths,
                "manning": valid_mannings,
                "river_width_alpha": self.var.river_width_alpha[valid_river_cell_mask],
                "river_width_beta": self.var.river_width_beta[valid_river_cell_mask],
            }
        )

        reach_means: pd.DataFrame = df_grid.groupby("river_id").mean()

        cols: list[str] = [
            "width",
            "depth",
            "manning",
            "river_width_alpha",
            "river_width_beta",
        ]
        self.var.rivers.loc[reach_means.index, cols] = reach_means[cols].astype(
            np.float64
        )

        self.var.rivers.loc[reach_means.index, ["river_depth_c", "river_depth_d"]] = [
            float(self.config["river_depth"]["parameters"]["c"]),
            float(self.config["river_depth"]["parameters"]["d"]),
        ]

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
        and calculates bankfull discharge, river width parameters (alpha, beta), depth parameters
        (c, d), and Manning roughness based on their upstream river network. Width and depth are
        estimated using downstream hydraulic geometry and assigned directly to self.var.rivers.
        Reaches without upstream rivers are not assigned geometry and retain NaN.

        Notes:
            Bankfull discharge is the sum of upstream bankfull discharges (m3/s).
            Parameters (alpha, beta, c, d, manning) are combined using discharge-weighted
            averaging across upstream segments, which ensures scaling consistency.
        """
        unrepresented_mask: pd.Series = ~self.var.rivers["represented_in_grid"]
        if not unrepresented_mask.any():
            return

        depth_params: dict[str, Any] = self.config["river_depth"]["parameters"]
        width_params: dict[str, Any] = self.config["river_width"]["parameters"]
        default_alpha: float = float(width_params["default_alpha"])
        default_beta: float = float(width_params["beta"])
        min_depth_m: float = float(depth_params["min_depth_m"])
        default_missing_width: float = float(self.default_missing_channel_width)
        shape_exponent: float = float(depth_params["shape_exponent"])
        shape_factor: float = shape_exponent + 1.0
        use_observed_width_as_bankfull: bool = bool(
            depth_params["use_observed_width_as_bankfull"]
        )

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
            combined_c: float = float((weights * valid_up["river_depth_c"]).sum())
            combined_d: float = float((weights * valid_up["river_depth_d"]).sum())
            combined_manning: float = float((weights * valid_up["manning"]).sum())

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

            w_expected: float = (
                default_alpha * (bankfull_q**default_beta)
                if use_observed_width_as_bankfull
                else combined_alpha * (bankfull_q**combined_beta)
            )
            expected_mean_channel_depth: float = combined_c * (bankfull_q**combined_d)
            continuity_ratio: float = (
                w_expected / estimated_width
                if is_observed and w_expected > 0.0
                else 1.0
            )
            continuity_mean_channel_depth: float = (
                continuity_ratio * expected_mean_channel_depth
            )
            estimated_depth: float = max(
                shape_factor * continuity_mean_channel_depth,
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
                    "river_depth_c",
                    "river_depth_d",
                    "return_period_2_years_daily_m3_per_s",
                    "width_is_observed",
                ],
            ] = [
                float(estimated_width),
                float(estimated_depth),
                float(combined_manning),
                float(alpha_to_assign),
                float(combined_beta),
                float(combined_c),
                float(combined_d),
                float(raw_summed_q),
                bool(is_observed),
            ]

    @property
    def bankfull_discharge_multiplier(self) -> float:
        """Calibration multiplier for bankfull discharge.

        Returns:
            The bankfull discharge multiplier (dimensionless).
        """
        return self.model.config["parameters"]["bankfull_discharge_multiplier"]

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
            width_params: dict[str, Any] = self.config["river_width"]["parameters"]
            default_alpha: np.float32 = np.float32(width_params["default_alpha"])
            default_beta: np.float32 = np.float32(width_params["beta"])
            use_observed_width_as_bankfull: bool = bool(
                self.config["river_depth"]["parameters"][
                    "use_observed_width_as_bankfull"
                ]
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
            return np.where(
                np.isnan(self.var.observed_average_river_width),
                np.where(
                    self.var.river_ids == -1,
                    np.float32(self.default_missing_channel_width),
                    estimated_width,
                ),
                self.var.observed_average_river_width,
            ).astype(np.float32)
        else:
            return np.where(
                np.isnan(self.var.observed_average_river_width),
                np.float32(self.default_missing_channel_width),
                self.var.observed_average_river_width,
            ).astype(np.float32)

    def calculate_bankfull_depth(
        self,
        bankfull_top_width_m: ArrayFloat32,
        use_simulated_bankfull_q: bool,
    ) -> ArrayFloat32:
        """Calculate bankfull channel depth across the routing grid using hydraulic continuity.

        Derives bankfull depth (meters) using based on Andreadis et al. (2013).

        https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/wrcr.20440

        When use_simulated_bankfull_q is True, simulated 2-year flood discharge Q_2 (from
        self.var.rivers['return_period_2_years_daily_m3_per_s']) is mapped to grid cells.
        When False (e.g. during the first year of simulation or during spinup), empirical
        width inversion is used.

        Notes:
            Under continuity, expected channel dimensions follow downstream hydraulic geometry
            (Andreadis et al. 2013):
                W_expected = alpha * Q^beta
                h_mean_expected = c * Q^d
            When use_observed_width_as_bankfull is True (default), W_expected is computed using baseline
            default parameters (default_alpha, beta), and mean channel depth scales inversely with observed width:
                h_mean_continuity = (W_expected / W_obs) * h_mean_expected.
            When use_observed_width_as_bankfull is False, reverts to previous machinery where W_expected
            is computed using reach-specific river_width_alpha.
            For a parabolic channel cross-section (shape_exponent r = 0.5), maximum centerline
            bankfull depth is related to mean depth by:
                h_centerline = (r + 1) * h_mean = 1.5 * h_mean.
            This ensures that bankfull cross-sectional area A_bf = (1 / (r + 1)) * W_obs * h_centerline = W_obs * h_mean
            accurately reproduces the expected bankfull flow area from hydraulic geometry.

        Args:
            bankfull_top_width_m: Bankfull top channel width per cell (meters).
            use_simulated_bankfull_q: Whether to use simulated 2-year flood return periods from
                self.var.rivers. Must be False during the first year of simulation or spinup.

        Returns:
            Bankfull centerline channel depth per cell (meters).

        Raises:
            ValueError: If use_simulated_bankfull_q is True but
                'return_period_2_years_daily_m3_per_s' contains null values in self.var.rivers.
        """
        width_params: dict[str, Any] = self.config["river_width"]["parameters"]
        default_alpha: np.float32 = np.float32(width_params["default_alpha"])
        default_beta: np.float32 = np.float32(width_params["beta"])

        depth_params: dict[str, Any] = self.config["river_depth"]["parameters"]
        depth_c: np.float32 = np.float32(depth_params["c"])
        depth_d: np.float32 = np.float32(depth_params["d"])
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
            alpha: ArrayFloat32 = np.maximum(width_alpha, np.float32(1.0))
            beta: ArrayFloat32 = np.maximum(width_beta, np.float32(0.1))
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

        w_expected: ArrayFloat32 = (
            default_alpha * (bankfull_discharge_m3_s**default_beta)
            if use_observed_width_as_bankfull or alpha is None or beta is None
            else alpha * (bankfull_discharge_m3_s**beta)
        )

        expected_mean_channel_depth_m: ArrayFloat32 = depth_c * (
            bankfull_discharge_m3_s**depth_d
        )

        # Determine which cells have observed channel width:
        is_observed: ArrayBool = ~np.isnan(self.var.observed_average_river_width) & (
            self.var.river_ids != -1
        )

        # Continuity depth scaled by observed width: h_mean = (W_expected / W_obs) * expected_mean_channel_depth.
        # When an observed channel width is available, mean depth scales inversely with width.
        # For channels without observed width, width is estimated directly from hydraulic geometry (W = W_expected),
        # so the scaling ratio W_expected / W_obs is 1.0.
        continuity_ratio: ArrayFloat32 = np.where(
            is_observed,
            (w_expected / np.maximum(bankfull_top_width_m, np.float32(1e-3))).astype(
                np.float32
            ),
            np.float32(1.0),
        )

        continuity_mean_channel_depth_m: ArrayFloat32 = (
            continuity_ratio * expected_mean_channel_depth_m
        )
        bankfull_depth_m: ArrayFloat32 = np.maximum(
            shape_factor * continuity_mean_channel_depth_m, min_depth_m
        ).astype(np.float32)

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
                f"  - Bankfull top width (W_obs): {float(bankfull_top_width_m[max_idx]):.2f} m",
                f"  - Observed average river width: {observed_width_m}",
                f"  - Bankfull discharge (Q): {float(bankfull_discharge_m3_s[max_idx]):.2f} m3/s",
                f"  - Discharge proxy (Q_proxy): {float(q_proxy[max_idx]):.2f} m3/s",
                f"  - Expected top width (W_expected): {float(w_expected[max_idx]):.2f} m",
                f"  - Continuity ratio (W_expected / W_obs): {float(continuity_ratio[max_idx]):.4f}",
                f"  - Expected mean depth (c * Q^d): {float(expected_mean_channel_depth_m[max_idx]):.2f} m",
                f"  - Continuity mean depth: {float(continuity_mean_channel_depth_m[max_idx]):.2f} m",
                f"  - Shape factor (r + 1): {float(shape_factor):.2f}",
                f"  - Width alpha: {alpha_val:.2f}",
                f"  - Width beta: {beta_val:.2f}",
                f"  - Depth c: {float(depth_c):.4f}",
                f"  - Depth d: {float(depth_d):.4f}",
                f"  - Bankfull discharge multiplier: {float(self.bankfull_discharge_multiplier):.2f}",
                f"  - Min depth: {float(min_depth_m):.2f} m",
                f"  - Shape exponent (r): {float(shape_exponent_val):.2f}",
                f"  - use_observed_width_as_bankfull: {use_observed_width_as_bankfull}",
                f"  - use_simulated_bankfull_q: {use_simulated_bankfull_q}",
            ]
            raise ValueError("\n".join(diagnostics))

        return bankfull_depth_m

    def set_router(self, initialize_storage: bool = True) -> None:
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

        Args:
            initialize_storage: Initialize river storage during spinup. Set False
                when rebuilding connections after reservoir construction.

        Returns:
            None.

        Raises:
            ValueError: If channel geometry is invalid.
            OSError: If weir diagnostics cannot be written.
        """  # noqa: DOC202, DOC502
        is_waterbody_outflow: ArrayBool = self.grid.var.waterbody_outflow_points != -1
        retention_basin_release_threshold_factor: float = self.config[
            "retention_basins"
        ]["release_threshold_factor"]

        # All cells with a valid river ID (river_ids != -1) use local inertial routing, even if steep;
        # non-river overland cells (river_ids == -1) use kinematic wave routing:
        use_kinematic: ArrayBool = self.var.river_ids == -1

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
            self.config["river_depth"]["parameters"]["shape_exponent"]
        )
        shape_exponent: ArrayFloat32 = np.full(
            self.var.river_ids.shape, shape_exponent_val, dtype=np.float32
        )

        bankfull_depth_m: ArrayFloat32 = self.calculate_bankfull_depth(
            bankfull_top_width_m=bankfull_top_width_m,
            use_simulated_bankfull_q=use_simulated_bankfull_q,
        )

        # Update derived bankfull width and depth to self.var.rivers
        self._update_rivers_geometry(
            width_grid=bankfull_top_width_m,
            depth_grid=bankfull_depth_m,
            manning_grid=self.grid.var.river_mannings,
        )

        # 4. Floodplain width beyond bankfull channel (m):
        floodplain_width_m: ArrayFloat32 = np.nan_to_num(
            self.grid.load2d(self.model.files["grid"]["routing/floodplain_width_m"]),
            nan=0.0,
        ).astype(np.float32)

        bankfull_river_elev: ArrayFloat32 = self.grid.load2d(
            self.model.files["grid"]["routing/bankfull_river_elevation_m"]
        )
        if self.hydrology.waterbodies.n > 0:
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
            floodplain_width_m=floodplain_width_m,
            waterbody_lake_area=self.hydrology.waterbodies.var.lake_area,
            waterbody_lake_factor=self.hydrology.waterbodies.var.lake_factor,
            waterbody_outflow_height=self.hydrology.waterbodies.var.outflow_height,
            waterbody_outflow_bed_elev=waterbody_outflow_bed_elev,
            river_storage_alpha=self.var.river_storage_alpha,
            river_storage_beta=self.var.river_storage_beta,
            in_spinup=self.model.in_spinup,
            weir_height_m=self.weir_height_m,
            instream_dam=self.instream_dam,
        )

        self.save_weirs()

        if self.model.in_spinup and initialize_storage:
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
            in_spinup=self.model.in_spinup and initialize_storage,
        )

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

        # where there is a pit, the river length is set to distance to the center of the cell,
        # thus half of the sqrt of the cell area
        self.grid.var.river_length[self.ldd == 5] = (
            np.sqrt(self.grid.var.cell_area[self.ldd == 5]) / 2
        )
        assert (self.grid.var.river_length > 0).all(), (
            "Channel length must be greater than 0 for all cells"
        )
        # Enforce minimum channel reach length of 1000m to prevent severe CFL substepping constraints
        self.grid.var.river_length = np.maximum(
            self.grid.var.river_length, np.float32(1000.0)
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
        minimum_river_slope: np.float32 = np.float32(
            self.config["minimum_river_slope_m_per_m"]
        )
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
            minimum_river_slope,
        )
        self.grid.var.river_slope_m_per_m = np.maximum(
            slope, minimum_river_slope
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
            * np.float32(self.model.config["parameters"]["mannings_n_multiplier"])
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
            default_alpha=self.config["river_width"]["parameters"]["default_alpha"],
            beta=self.config["river_width"]["parameters"]["beta"],
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

        for alpha a global value of 7.2 is used, and beta is set to a constant value, usualy 0.50
        based on https://doi.org/10.1002/esp.403 (eq. 15).

        Re-arranging for alpha gives:

            alpha = river_width / discharge^beta

        for rivers where we don't have an observed average river width, we use the default
        for alpha throughout the simulation.

        Args:
            beta: The beta parameter for the kinematic wave routing.
            default_alpha: The default alpha value to use for rivers without an observed average river width,
                default is 7.2.

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

        channel_abstraction_m3_per_hour: np.ndarray = channel_abstraction_m3 / 24
        assert (
            channel_abstraction_m3_per_hour[self.grid.var.waterbody_ids != -1] == 0.0
        ).all(), (
            "Channel abstraction must be zero for water bodies, "
            "but found non-zero value."
        )

        return_flow_m3_per_hour: np.ndarray = return_flow * self.grid.var.cell_area / 24

        # add return flow to the water bodies
        return_flow_m3_to_waterbodies_per_hour: np.ndarray = np.bincount(
            self.grid.var.waterbody_ids[self.grid.var.waterbody_ids != -1],
            weights=return_flow_m3_per_hour[self.grid.var.waterbody_ids != -1],
            minlength=self.hydrology.waterbodies.n,
        )
        return_flow_m3_per_hour[self.grid.var.waterbody_ids != -1] = 0.0

        self.var.discharge_m3_s_per_substep = np.full_like(
            self.var.discharge_m3_s_per_substep,
            fill_value=np.nan,
        )
        self.var.retention_basin_storage_m3_per_substep = np.full_like(
            self.var.discharge_m3_s_per_substep,
            fill_value=np.nan,
        )

        if __debug__:
            # these are for balance checks, the sum of all routing steps
            evaporation_in_rivers_m3: ArrayFloat32 = self.grid.full_compressed(
                0, dtype=np.float32
            )
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

        over_abstraction_m3: ArrayFloat32 = self.grid.full_compressed(
            0, dtype=np.float32
        )

        # update alpha and beta once per day
        if self.model.in_spinup:
            (
                self.var.river_width_alpha,
                self.var.river_width_beta,
            ) = self.get_river_width_alpha_and_beta(
                default_alpha=self.config["river_width"]["parameters"]["default_alpha"],
                beta=self.config["river_width"]["parameters"]["beta"],
            )
            if self.var.discharge_step_count >= 24:
                use_simulated_bankfull_q: bool = self.has_simulated_bankfull_q
                dynamic_width: ArrayFloat32
                if use_simulated_bankfull_q:
                    dynamic_width = self.calculate_bankfull_width(
                        use_simulated_bankfull_q=True
                    )
                else:
                    avg_discharge: ArrayFloat32 = (
                        self.var.sum_of_all_discharge_steps
                        / max(self.var.discharge_step_count, 1)
                    ).astype(np.float32)
                    width_params_step: dict[str, Any] = self.config["river_width"][
                        "parameters"
                    ]
                    default_alpha_val: np.float32 = np.float32(
                        width_params_step["default_alpha"]
                    )
                    default_beta_val: np.float32 = np.float32(width_params_step["beta"])
                    use_obs_flag: bool = bool(
                        self.config["river_depth"]["parameters"][
                            "use_observed_width_as_bankfull"
                        ]
                    )
                    alpha_w: ArrayFloat32 | np.float32 = (
                        default_alpha_val
                        if use_obs_flag
                        else self.var.river_width_alpha
                    )
                    beta_w: ArrayFloat32 | np.float32 = (
                        default_beta_val if use_obs_flag else self.var.river_width_beta
                    )
                    dynamic_width = np.where(
                        np.isnan(self.var.observed_average_river_width),
                        np.maximum(
                            alpha_w
                            * (np.maximum(avg_discharge, np.float32(1e-4)) ** beta_w),
                            np.float32(self.default_missing_channel_width),
                        ),
                        self.var.observed_average_river_width,
                    ).astype(np.float32)
                self.var.river_width = dynamic_width
                dynamic_depth_m: ArrayFloat32 = self.calculate_bankfull_depth(
                    bankfull_top_width_m=dynamic_width,
                    use_simulated_bankfull_q=use_simulated_bankfull_q,
                )
                self.router.update_channel_geometry(dynamic_width, dynamic_depth_m)

        for hour in range(24):
            # increment inflow index for next hour
            self.inflow_idx += 1

            total_runoff_m3: np.ndarray = (
                total_runoff_m[hour, :] * self.grid.var.cell_area
            )

            # then split the runoff into runoff directly to water bodies
            # and runoff to the channel network
            self.hydrology.waterbodies.var.storage += np.bincount(
                self.grid.var.waterbody_ids[self.grid.var.waterbody_ids != -1],
                weights=total_runoff_m3[self.grid.var.waterbody_ids != -1],
                minlength=self.hydrology.waterbodies.n,
            )

            # after adding the runoff to the water bodies, we set the runoff to zero
            # in those grid cells
            total_runoff_m3[self.grid.var.waterbody_ids != -1] = 0.0

            self.hydrology.waterbodies.var.storage += (
                return_flow_m3_to_waterbodies_per_hour
            )

            evaporation_sum_m: np.ndarray = np.bincount(
                self.grid.var.waterbody_ids[self.grid.var.waterbody_ids != -1],
                weights=reference_evapotranspiration_water_m[
                    hour, self.grid.var.waterbody_ids != -1
                ],
                minlength=self.hydrology.waterbodies.n,
            )
            waterbody_cell_count: ArrayInt64 = np.bincount(
                self.grid.var.waterbody_ids[self.grid.var.waterbody_ids != -1],
                minlength=self.hydrology.waterbodies.n,
            )
            # Future reservoirs have no cells or evaporation.
            potential_evaporation_per_waterbody_m3 = (
                np.divide(
                    evaporation_sum_m,
                    waterbody_cell_count,
                    out=np.zeros(self.hydrology.waterbodies.n, dtype=np.float64),
                    where=waterbody_cell_count > 0,
                )
                * self.hydrology.waterbodies.var.lake_area
            )

            actual_evaporation_from_waterbodies_per_hour_m3 = np.minimum(
                potential_evaporation_per_waterbody_m3,
                self.hydrology.waterbodies.var.storage,
            )

            self.hydrology.waterbodies.var.storage -= (
                actual_evaporation_from_waterbodies_per_hour_m3
            )

            # Calculate potential evaporation for retention basins
            if not self.retention_basin_data.empty:
                retention_basin_area = (
                    self.retention_max_storage_m3
                    / np.float32(3.0)  # assumed depth of 3 meters
                ).astype(np.float32)
                retention_mask = self.retention_basin_ids != -1

                # aggregate potential ET for retention basins
                # Since each basin is exactly one cell, we can map the ET values directly
                potential_evaporation_per_retention_basin_m3 = np.zeros(
                    len(retention_basin_area), dtype=np.float32
                )
                basin_ids = self.retention_basin_ids[retention_mask]
                potential_evaporation_per_retention_basin_m3[basin_ids] = (
                    reference_evapotranspiration_water_m[hour, retention_mask]
                    * retention_basin_area[basin_ids]
                )

                assert not np.isnan(potential_evaporation_per_retention_basin_m3).any()

                actual_evaporation_from_retention_basins_m3 = np.minimum(
                    potential_evaporation_per_retention_basin_m3,
                    self.var.retention_basin_storage_m3,
                ).astype(np.float32)

                assert not np.isnan(potential_evaporation_per_retention_basin_m3).any()
                assert not np.isnan(actual_evaporation_from_retention_basins_m3).any()

                self.var.retention_basin_storage_m3 -= (
                    actual_evaporation_from_retention_basins_m3
                )
                if __debug__:
                    retention_evaporation_m3 += (
                        actual_evaporation_from_retention_basins_m3
                    )

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

            valid_prescribed_outflow = ~np.isnan(outflow_per_waterbody_m3)
            if valid_prescribed_outflow.any():
                assert (
                    outflow_per_waterbody_m3[valid_prescribed_outflow]
                    <= self.hydrology.waterbodies.var.storage[
                        valid_prescribed_outflow
                    ].astype(np.float32)
                ).all(), "Prescribed reservoir outflow cannot be greater than storage"

            side_flow_channel_m3_per_hour = (
                total_runoff_m3
                + return_flow_m3_per_hour
                - channel_abstraction_m3_per_hour
            )
            assert (
                side_flow_channel_m3_per_hour[self.grid.var.waterbody_ids != -1] == 0
            ).all()

            for (y, x), inflow in self.inflow.items():
                cell_index: int = self.grid.linear_mapping[y, x]
                if self.grid.var.waterbody_ids[cell_index] != -1:
                    raise ValueError("Inflow cannot be added to waterbody cells.")

                inflow_m3 = inflow[self.inflow_idx] * np.float32(3600)

                side_flow_channel_m3_per_hour[cell_index] += inflow_m3

                if __debug__:
                    total_inflow_m3 += inflow_m3

            assert not np.isnan(
                self.var.discharge_in_rivers_m3_s_substep[
                    self.grid.var.waterbody_ids == -1
                ]
            ).all()

            river_width: ArrayFloat32 = get_river_width(
                self.var.river_width_alpha,
                self.var.river_width_beta,
                self.var.discharge_in_rivers_m3_s_substep,
            )
            # the ratio of each grid cell that is currently covered by a river
            channel_ratio: ArrayFloat32 = get_channel_ratio(
                river_length=self.grid.var.river_length,
                river_width=np.where(self.grid.var.waterbody_ids == -1, river_width, 0),
                cell_area=self.grid.var.cell_area,
            )

            # calculate evaporation from rivers per timestep usting the current channel ratio
            potential_evaporation_in_rivers_m3_per_hour = (
                reference_evapotranspiration_water_m[hour]
                * channel_ratio
                * self.grid.var.cell_area
            )

            (
                self.var.discharge_in_rivers_m3_s_substep,
                self.var.river_storage_m3,
                actual_evaporation_in_rivers_m3_per_hour,
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
                sideflow_m3=side_flow_channel_m3_per_hour.astype(np.float32),
                evaporation_m3=potential_evaporation_in_rivers_m3_per_hour,
                waterbody_storage_m3=self.hydrology.waterbodies.var.storage,
                outflow_per_waterbody_m3=outflow_per_waterbody_m3,
                retention_storage_m3=self.var.retention_basin_storage_m3,
                retention_activation_threshold_m3_s=self.retention_activation_threshold_m3_s,
            )
            self.var.water_stage_m = self.router.get_water_stage(
                out=self.var.water_stage_m
            )
            if not (actual_evaporation_in_rivers_m3_per_hour >= 0.0).all():
                raise ValueError(
                    f"Negative evaporation detected, min evaporation: {actual_evaporation_in_rivers_m3_per_hour.min()}, min discharge: {self.var.discharge_in_rivers_m3_s_substep.min()}. "
                )

            assert (actual_evaporation_in_rivers_m3_per_hour >= 0.0).all()

            # Dam operators need past river flows when the reservoir opens.
            operator_inflow_m3: ArrayFloat32 = waterbody_inflow_m3.copy()
            future_reservoirs: ArrayBool = ~self.hydrology.waterbodies.is_active
            operator_inflow_m3[future_reservoirs] = (
                self.var.discharge_in_rivers_m3_s_substep[
                    self.hydrology.waterbodies.var.waterbody_outflow_linear_mapping[
                        future_reservoirs
                    ]
                ]
                * 3600
            )
            self.model.agents.reservoir_operators.track_inflow(
                operator_inflow_m3[self.model.hydrology.waterbodies.is_reservoir]
            )

            # ensure that discharge is nan for water bodies
            assert np.isnan(
                self.var.discharge_in_rivers_m3_s_substep[
                    self.grid.var.waterbody_ids != -1
                ]
            ).all()

            # ensure that discharge is not nan for river cells
            assert not np.isnan(
                self.var.discharge_in_rivers_m3_s_substep[
                    self.grid.var.waterbody_ids == -1
                ]
            ).any()

            discharge_m3_s_substep = self.var.discharge_in_rivers_m3_s_substep.copy()

            # check if discharge is not higher than the highest discharge ever recorded
            if np.nanmax(discharge_m3_s_substep) > 400_000:
                raise ValueError(
                    f"Discharge is higher than the highest discharge ever recorded. Max discharge: {np.nanmax(discharge_m3_s_substep)}"
                )

            # set waterbody outflow points to the outflow of the waterbody
            discharge_m3_s_substep = self.hydrology.waterbodies.map_to_grid_outflow(
                outflow_per_waterbody_m3 / 3600, out=discharge_m3_s_substep
            )
            discharge_m3_s_substep = fill_discharge_in_waterbodies(
                discharge_m3_s=discharge_m3_s_substep,
                upstream_matrix_from_up_to_downstream=self.router.upstream_matrix_from_up_to_downstream,
                idxs_up_to_downstream=self.router.idxs_up_to_downstream,
            )

            # after filling the gaps, we should not have any nans in the river cells
            assert not np.isnan(discharge_m3_s_substep[self.var.river_ids != -1]).any()

            self.var.discharge_m3_s_per_substep[hour, :] = discharge_m3_s_substep

            retention_basin_storage_m3_substep = self.grid.full_compressed(
                0, dtype=np.float32
            )
            retention_mask = self.retention_basin_ids != -1
            retention_basin_storage_m3_substep[retention_mask] = (
                self.var.retention_basin_storage_m3[
                    self.retention_basin_ids[retention_mask]
                ]
            )
            self.var.retention_basin_storage_m3_per_substep[hour, :] = (
                retention_basin_storage_m3_substep
            )

            self.var.sum_of_all_discharge_steps += discharge_m3_s_substep
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
                waterbody_evaporation_m3 += (
                    actual_evaporation_from_waterbodies_per_hour_m3
                )
                evaporation_in_rivers_m3 += actual_evaporation_in_rivers_m3_per_hour
                over_abstraction_m3 += over_abstraction_m3_routing_step
                command_area_release_m3 += command_area_release_m3_routing_step
                # Accumulate retention flows across all hourly timesteps for the day
                retention_inflow_m3 += retention_inflow_m3_hour
                retention_outflow_m3 += retention_outflow_m3_hour

        self.var.discharge_m3_s = self.var.discharge_m3_s_per_substep.mean(axis=0)

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
        value analysis on daily simulated river outflow. Dynamically updates bankfull
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
                / self.model.config["general"]["spinup_name"]
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
                return_periods=list(set([2, activation_threshold_return_period_years])),
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
