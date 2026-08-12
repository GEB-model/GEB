"""Routing algorithms for river networks."""

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

from .accuflux import Accuflux as Accuflux
from .kinematic_wave import KinematicWave as KinematicWave
from .local_inertial import LocalInertial

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

    discharge_data = {}
    for river_id in rivers.index:
        assert isinstance(river_id, int)
        xys: list[tuple[int, int]] = get_river_representative_xys(river_id, all_rivers)
        if len(xys) == 1:
            if source == "file":
                assert folder is not None
                discharge_data[river_id] = read_table(
                    folder / f"river_outflow_hourly_m3_per_s_{river_id}.parquet"
                )[f"river_outflow_hourly_m3_per_s_{river_id}"]
            else:
                assert variables_to_report is not None
                discharge_data[river_id] = create_df_from_report_variable(
                    river_id, variables_to_report
                )
        else:
            total_discharge_part = None
            for i in range(len(xys)):
                if source == "file":
                    assert folder is not None
                    discharge_part = read_table(
                        folder / f"river_outflow_hourly_m3_per_s_{river_id}_{i}.parquet"
                    )[f"river_outflow_hourly_m3_per_s_{river_id}_{i}"]
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
        transform=transform,
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

        """
        super().__init__(model)

        self.config = model.config["hydrology"]["routing"]

        self.default_missing_channel_width: float = (
            3.0  # Default width for missing values
        )

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

        # Can be removed in August 2026
        self.grid.var.retention_basin_storage_m3_per_substep = np.full(
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
        is_waterbody: TwoDArrayBool = (
            self.grid.load2d(
                self.model.files["grid"]["waterbodies/waterbody_id"], compress=False
            )
            != -1
        )

        rivers: gpd.GeoDataFrame = read_geom(self.model.files["geom"]["routing/rivers"])

        # set river ID to -1 for waterbody cells
        river_ids = self.grid.load2d(
            self.model.files["grid"]["routing/river_ids"], compress=False
        )
        # keep a copy of the original river IDs before removing waterbodies,
        # which is needed for some output variables
        river_ids_no_waterbodies_removed = self.grid.compress(river_ids)
        river_ids[is_waterbody] = -1  # set river ID to -1 for waterbody cells
        river_ids: ArrayInt32 = self.grid.compress(river_ids)

        # select only hydrography_xy that are not in waterbodies
        # and store the mask to filter other columns as well
        not_waterbody_mask = rivers["hydrography_xy"].apply(
            lambda xys: [not is_waterbody[xy[1], xy[0]] for xy in xys]
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
            for xys, mask in zip(rivers["hydrography_xy"], not_waterbody_mask)
        ]

        rivers["hydrography_upstream_area_m2_no_waterbodies_removed"] = rivers[
            "hydrography_upstream_area_m2"
        ]
        rivers["hydrography_upstream_area_m2"] = [
            np.array([ua for ua, m in zip(uas, mask) if m])
            for uas, mask in zip(
                rivers["hydrography_upstream_area_m2"], not_waterbody_mask
            )
        ]

        # update represented_in_grid based on whether there are any hydrography_xy left after removing waterbodies
        rivers["represented_in_grid"] = rivers["hydrography_xy"].apply(
            lambda xys: len(xys) > 0
        )
        rivers["hydrography_linear"] = rivers["hydrography_xy"].apply(
            lambda xys: np.array(
                [grid_linear_mapping[xy[1], xy[0]].item() for xy in xys]
            )
        )
        return rivers, river_ids, river_ids_no_waterbodies_removed

    def set_router(self) -> None:
        """Initialize the local inertial routing algorithm."""
        is_waterbody_outflow: ArrayBool = self.grid.var.waterbody_outflow_points != -1
        retention_basin_release_threshold_factor: float = self.config[
            "retention_basins"
        ]["release_threshold_factor"]

        use_kinematic: ArrayBool = np.isnan(self.var.observed_average_river_width)

        self.router = LocalInertial(
            dt=3600,
            river_network=self.river_network,
            river_length=self.grid.var.river_length,
            river_width=self.var.observed_average_river_width,
            waterbody_ids=self.grid.var.waterbody_ids,
            river_ids=self.var.river_ids,
            is_waterbody_outflow=is_waterbody_outflow,
            retention_max_storage_m3=self.retention_max_storage_m3,
            retention_node_id=self.retention_basin_ids,
            controlled_retention=self.controlled_retention,
            retention_basin_release_threshold_factor=retention_basin_release_threshold_factor,
            bankfull_river_elevation_m=self.grid.load2d(
                self.model.files["grid"]["routing/bankfull_river_elevation_m"]
            ),
            manning_n=self.grid.var.river_mannings,
            use_kinematic=use_kinematic,
        )

        # ---------------------------------------------------------------------
        # Initialize persistent river storage state (Width-based depth assumption)
        # ---------------------------------------------------------------------
        # Assume depth scales with river width (W / 20), clamped between 0.5 m and 3.0 m
        assumed_depth_m = np.clip(
            np.where(
                np.isnan(self.var.observed_average_river_width),
                0.25,
                self.var.observed_average_river_width / 20.0,
            ),
            0.5,
            3.0,
        )

        assumed_width_m = np.where(
            np.isnan(self.var.observed_average_river_width),
            1.0,
            self.var.observed_average_river_width,
        )

        # Storage = Length * Width * Depth
        initial_storage = self.grid.var.river_length * assumed_width_m * assumed_depth_m
        initial_storage[self.grid.var.waterbody_ids != -1] = 0.0

        self.grid.var.river_storage_m3 = initial_storage.astype(np.float64)

        # Calculate consistent initial discharge Q from storage
        self.grid.var.discharge_in_rivers_m3_s_substep = (
            self.router.calculate_discharge_from_river_storage(
                river_storage=self.grid.var.river_storage_m3,
                river_storage_alpha=self.grid.var.river_storage_alpha,
                river_storage_beta=self.grid.var.river_storage_beta,
                river_length=self.grid.var.river_length,
                waterbody_id=self.grid.var.waterbody_ids,
            )
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
        self.var.active_rivers = self.get_active_rivers()

        self.grid.var.upstream_area = self.grid.load2d(
            self.model.files["grid"]["routing/upstream_area_m2"]
        )
        if "routing/upstream_area_n_cells" in self.model.files["grid"]:
            self.grid.var.upstream_area_n_cells = self.grid.load2d(
                self.model.files["grid"]["routing/upstream_area_n_cells"]
            )
        else:
            # TODO: Remove this in feb 2026
            self.grid.var.upstream_area_n_cells = self.river_network.upstream_area(
                unit="cell"
            )[~self.grid.mask]

        # Channel Manning's n
        self.grid.var.river_mannings = (
            self.grid.load2d(self.model.files["grid"]["routing/mannings"])
            * self.model.config["parameters"]["mannings_n_multiplier"]
        )
        assert (self.grid.var.river_mannings > 0).all()

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

        # Channel gradient (fraction, dy/dx)
        minimum_river_slope = 0.00001
        self.grid.var.river_slope_m_per_m = np.maximum(
            self.grid.load2d(self.model.files["grid"]["routing/river_slope_m_per_m"]),
            minimum_river_slope,
        )

        # river_storage_alpha for kinematic wave storage calculation
        # source: https://gmd.copernicus.org/articles/13/3267/2020/ eq. 21
        # It's based on Manning's n, wetted perimeter, and slope.
        # wetted perimeter is approximated by width for rivers.
        # We use a constant beta of 0.6 for Broad Sheet Flow / Manning's equation.
        river_storage_beta_constant = np.float32(0.6)
        self.grid.var.river_storage_beta = self.grid.full_compressed(
            river_storage_beta_constant, dtype=np.float32
        )
        self.grid.var.river_storage_alpha = (
            self.grid.var.river_mannings
            * river_wetted_perimeter ** (2 / 3)
            / np.sqrt(self.grid.var.river_slope_m_per_m)
        ) ** self.grid.var.river_storage_beta

        # For dynamic river width, we need the average discharge. Therefore,
        # we track the sum of all discharge steps and the number of discharge steps,
        # which can be used to calculate the average discharge at each time step.
        self.var.discharge_step_count = 0
        self.var.sum_of_all_discharge_steps = self.grid.full_compressed(
            0, dtype=np.float64
        )
        (
            self.hydrology.grid.var.river_width_alpha,
            self.hydrology.grid.var.river_width_beta,
        ) = self.get_river_width_alpha_and_beta(
            default_alpha=self.config["river_width"]["parameters"]["default_alpha"],
            beta=self.config["river_width"]["parameters"]["beta"],
        )

        # Initialize discharge with zero
        self.grid.var.discharge_in_rivers_m3_s_substep = self.grid.full_compressed(
            1e-30, dtype=np.float32
        )
        self.grid.var.discharge_m3_s_substep = self.grid.full_compressed(
            1e-30, dtype=np.float32
        )
        self.grid.var.discharge_m3_s_per_substep = np.full(
            (24, self.grid.var.discharge_m3_s_substep.size),
            0,
            dtype=self.grid.var.discharge_m3_s_substep.dtype,
        )
        self.grid.var.retention_basin_storage_m3_per_substep = np.full(
            (24, self.grid.var.discharge_m3_s_substep.size),
            0,
            dtype=np.float32,
        )

        # initialize retention basin storage with zero
        self.grid.var.retention_basin_storage_m3 = np.zeros(
            len(self.retention_basin_data), dtype=np.float32
        )

        # initialize daily total retention basin water fluxes for tracking and output
        self.grid.var.retention_inflow_m3_daily = np.zeros(
            len(self.retention_basin_data), dtype=np.float32
        )
        self.grid.var.retention_outflow_m3_daily = np.zeros(
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
            average_discharge: ArrayFloat32 = (
                self.var.sum_of_all_discharge_steps / (self.var.discharge_step_count)
            ).astype(np.float64)
            # re-arranged formula for alpha, where we use the observed average river width and the average discharge to calculate alpha
            alpha: ArrayFloat32 = np.full_like(
                self.var.observed_average_river_width,
                default_alpha,
                dtype=np.float32,
            )  # default alpha everywhere
            calculate_alpha = (~np.isnan(self.var.observed_average_river_width)) & (
                self.grid.var.waterbody_ids == -1
            )  # decide where alpha should be calculated
            alpha[calculate_alpha] = self.var.observed_average_river_width[
                calculate_alpha
            ] / (
                average_discharge[calculate_alpha] ** beta_array[calculate_alpha]
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
            pre_river_storage_m3: ArrayFloat64 = self.grid.var.river_storage_m3.copy()
            pre_retention_storage_m3: ArrayFloat32 = (
                self.grid.var.retention_basin_storage_m3.copy()
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
        )
        return_flow_m3_per_hour[self.grid.var.waterbody_ids != -1] = 0.0

        self.grid.var.discharge_m3_s_per_substep = np.full_like(
            self.grid.var.discharge_m3_s_per_substep,
            fill_value=np.nan,
        )
        self.grid.var.retention_basin_storage_m3_per_substep = np.full_like(
            self.grid.var.discharge_m3_s_per_substep,
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
                self.grid.var.retention_basin_storage_m3, dtype=np.float32
            )
            retention_outflow_m3: ArrayFloat32 = np.zeros_like(
                self.grid.var.retention_basin_storage_m3, dtype=np.float32
            )
            retention_evaporation_m3: ArrayFloat32 = np.zeros_like(
                self.grid.var.retention_basin_storage_m3, dtype=np.float32
            )

        over_abstraction_m3: ArrayFloat32 = self.grid.full_compressed(
            0, dtype=np.float32
        )

        # update alpha and beta once per day
        if self.model.in_spinup:
            (
                self.hydrology.grid.var.river_width_alpha,
                self.hydrology.grid.var.river_width_beta,
            ) = self.get_river_width_alpha_and_beta(
                default_alpha=self.config["river_width"]["parameters"]["default_alpha"],
                beta=self.config["river_width"]["parameters"]["beta"],
            )

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
            )

            # after adding the runoff to the water bodies, we set the runoff to zero
            # in those grid cells
            total_runoff_m3[self.grid.var.waterbody_ids != -1] = 0.0

            self.hydrology.waterbodies.var.storage += (
                return_flow_m3_to_waterbodies_per_hour
            )

            # TODO: This calculation can be optimized by pre-calculating some parts
            potential_evaporation_per_waterbody_m3 = (
                np.bincount(
                    self.grid.var.waterbody_ids[self.grid.var.waterbody_ids != -1],
                    weights=reference_evapotranspiration_water_m[
                        hour, self.grid.var.waterbody_ids != -1
                    ],
                )
                / np.bincount(
                    self.grid.var.waterbody_ids[self.grid.var.waterbody_ids != -1]
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
                    self.grid.var.retention_basin_storage_m3,
                ).astype(np.float32)

                assert not np.isnan(potential_evaporation_per_retention_basin_m3).any()
                assert not np.isnan(actual_evaporation_from_retention_basins_m3).any()

                self.grid.var.retention_basin_storage_m3 -= (
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

            assert (
                outflow_per_waterbody_m3
                <= self.hydrology.waterbodies.var.storage.astype(np.float32)
            ).all(), "outflow cannot be greater than storage"

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
                self.grid.var.discharge_in_rivers_m3_s_substep[
                    self.grid.var.waterbody_ids == -1
                ]
            ).all()

            river_width: ArrayFloat32 = get_river_width(
                self.hydrology.grid.var.river_width_alpha,
                self.hydrology.grid.var.river_width_beta,
                self.grid.var.discharge_in_rivers_m3_s_substep,
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
                self.grid.var.discharge_in_rivers_m3_s_substep,
                self.grid.var.river_storage_m3,
                actual_evaporation_in_rivers_m3_per_hour,
                over_abstraction_m3_routing_step,
                self.hydrology.waterbodies.var.storage,
                waterbody_inflow_m3,
                outflow_at_pits_m3_routing_step,
                self.grid.var.retention_basin_storage_m3,
                retention_inflow_m3_hour,
                retention_outflow_m3_hour,
            ) = self.router.step(
                Q_prev_m3_s=self.grid.var.discharge_in_rivers_m3_s_substep,
                river_storage_m3=self.grid.var.river_storage_m3,
                sideflow_m3=side_flow_channel_m3_per_hour.astype(np.float32),
                evaporation_m3=potential_evaporation_in_rivers_m3_per_hour,
                waterbody_storage_m3=self.hydrology.waterbodies.var.storage,
                outflow_per_waterbody_m3=outflow_per_waterbody_m3,
                retention_storage_m3=self.grid.var.retention_basin_storage_m3,
                river_storage_alpha=self.grid.var.river_storage_alpha,
                river_storage_beta=self.grid.var.river_storage_beta,
                retention_activation_threshold_m3_s=self.retention_activation_threshold_m3_s,
            )

            if not (actual_evaporation_in_rivers_m3_per_hour >= 0.0).all():
                raise ValueError(
                    f"Negative evaporation detected, min evaporation: {actual_evaporation_in_rivers_m3_per_hour.min()}, min discharge: {self.grid.var.discharge_in_rivers_m3_s_substep.min()}. "
                )

            assert (actual_evaporation_in_rivers_m3_per_hour >= 0.0).all()

            # the reservoir operators need to track the inflow to the reservoirs
            self.model.agents.reservoir_operators.track_inflow(
                waterbody_inflow_m3[self.model.hydrology.waterbodies.is_reservoir]
            )

            # ensure that discharge is nan for water bodies
            assert np.isnan(
                self.grid.var.discharge_in_rivers_m3_s_substep[
                    self.grid.var.waterbody_ids != -1
                ]
            ).all()

            # ensure that discharge is not nan for river cells
            assert not np.isnan(
                self.grid.var.discharge_in_rivers_m3_s_substep[
                    self.grid.var.waterbody_ids == -1
                ]
            ).any()

            discharge_m3_s_substep = (
                self.grid.var.discharge_in_rivers_m3_s_substep.copy()
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

            self.grid.var.discharge_m3_s_per_substep[hour, :] = discharge_m3_s_substep

            retention_basin_storage_m3_substep = self.grid.full_compressed(
                0, dtype=np.float32
            )
            retention_mask = self.retention_basin_ids != -1
            retention_basin_storage_m3_substep[retention_mask] = (
                self.grid.var.retention_basin_storage_m3[
                    self.retention_basin_ids[retention_mask]
                ]
            )
            self.grid.var.retention_basin_storage_m3_per_substep[hour, :] = (
                retention_basin_storage_m3_substep
            )

            self.var.sum_of_all_discharge_steps += discharge_m3_s_substep
            self.var.discharge_step_count += 1

            if __debug__:
                assert (
                    self.router.get_available_storage(
                        self.grid.var.discharge_in_rivers_m3_s_substep,
                        self.grid.var.river_storage_alpha,
                        self.grid.var.river_storage_beta,
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

        self.grid.var.discharge_m3_s = self.grid.var.discharge_m3_s_per_substep.mean(
            axis=0
        )

        if not self.model.in_spinup and (
            self.model.current_day_of_year == 1 or self.model.current_timestep == 0
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
                    self.grid.var.river_storage_m3,
                    self.grid.var.retention_basin_storage_m3,
                ],
                name="routing_1",
                tolerance=100,
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
            self.grid.var.retention_inflow_m3_daily = retention_inflow_m3
            self.grid.var.retention_outflow_m3_daily = retention_outflow_m3

        self.report(locals())

        total_over_abstraction_m3: np.float64 = over_abstraction_m3.astype(
            np.float64
        ).sum()
        if over_abstraction_m3.sum() > 100:
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

        # TODO: Remove the if statement in March 2026. The part selection behind the statement
        # should always be done when it is removed.
        if "is_further_downstream_outflow" in rivers.columns:
            rivers = rivers[~rivers["is_further_downstream_outflow"]]
        outflow_rivers: gpd.GeoDataFrame = rivers[
            ~rivers["downstream_ID"].isin(rivers.index)
        ]
        return outflow_rivers

    def get_active_rivers(self) -> gpd.GeoDataFrame:
        """Get the rivers that are simulated (i.e., not downstream of the model region).

        Returns:
            A GeoDataFrame containing the active rivers.
        """
        rivers: gpd.GeoDataFrame = self.var.rivers
        active_rivers = rivers[
            (~rivers["is_downstream_outflow"])
            & (~rivers["is_further_downstream_outflow"])
        ]

        to_remove: set[int] = set()
        for river in active_rivers.itertuples():
            if not river.represented_in_grid:
                to_search: set[int] = {river.Index}
                upstream_rivers: set[int] = set()

                while to_search:
                    current_id: int = to_search.pop()
                    upstream_rivers_of_this_river: pd.DataFrame = rivers[
                        rivers["downstream_ID"] == current_id
                    ]
                    for upstream_river in upstream_rivers_of_this_river.itertuples():
                        if upstream_river.represented_in_grid:  # ty:ignore[unresolved-attribute]
                            upstream_rivers.add(upstream_river.Index)  # ty:ignore[unresolved-attribute]
                        else:
                            to_search.add(upstream_river.Index)  # ty:ignore[unresolved-attribute]

                if not upstream_rivers:
                    to_remove.add(river.Index)

        active_rivers = active_rivers[~active_rivers.index.isin(to_remove)]

        return active_rivers.copy()

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
        """Update the return periods for the routing module.

        Raises:
            ValueError: If the model is still in the spinup period, return periods cannot be updated.
        """
        if self.model.in_spinup:
            raise ValueError(
                "Return periods can only be updated after the spinup period is completed."
            )

        activation_threshold_return_period_years: float = self.config[
            "retention_basins"
        ]["activation_threshold_return_period_years"]

        active_rivers: gpd.GeoDataFrame = self.get_active_rivers()
        discharge_by_river: pd.DataFrame = get_discharge_per_river(
            rivers=active_rivers,
            all_rivers=self.var.rivers,
            source="file",
            folder=self.model.report_folder.parent.parent
            / self.model.config["general"]["spinup_name"]
            / "report"
            / "hydrology.routing",
        )

        if self.model.current_timestep > 0:
            discharge_by_river_run: pd.DataFrame = get_discharge_per_river(
                rivers=active_rivers,
                all_rivers=self.var.rivers,
                source="memory",
                variables_to_report=self.variables_to_report,
            )

            discharge_by_river: pd.DataFrame = pd.concat(
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
            model = ReturnPeriodModel(
                series=discharge_by_river_daily[idx],
                return_periods=list(set([2, activation_threshold_return_period_years])),
                min_exceed=30,
                nboot=2000,
                fixed_shape=0.0,
                fixed_scale=None,
                p_value_threshold=0.05,
                selection_strategy="first_significant",
            )

            for return_period, return_water_level in model.rl_table.set_index(
                "T_years"
            )["GPD_POT_RL"].items():
                self.var.rivers.loc[
                    idx, f"return_period_{return_period}_years_daily_m3_per_s"
                ] = return_water_level

        for basin_id in range(len(self.retention_basin_data)):
            basin_cells = np.where(self.retention_basin_ids == basin_id)[0]
            if len(basin_cells) == 0:
                raise ValueError(
                    f"Retention basin {basin_id} has no associated grid cells."
                )
            river_id = self.var.river_ids[basin_cells[0]]
            if river_id == -1:
                raise ValueError(
                    f"Retention basin {basin_id} is not associated with any river."
                )

            if not self.retention_basin_is_active[basin_id]:
                self.retention_activation_threshold_m3_s[basin_id] = np.inf
            else:
                return_period_col = f"return_period_{activation_threshold_return_period_years}_years_daily_m3_per_s"
                self.retention_activation_threshold_m3_s[basin_id] = (
                    self.var.rivers.loc[river_id, return_period_col]
                )

        return None
