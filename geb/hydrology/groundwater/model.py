"""The groundwater model using MODFLOW."""

# --------------------------------------------------------------------------------
# This file contains code that has been adapted from an original source available
# in a public repository under the GNU General Public License. The original code
# has been modified to fit the specific needs of this project.
#
# Original source repository: https://github.com/iiasa/CWatM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------------

import logging
import os
import platform
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Callable, overload

import flopy
import numpy as np
import numpy.typing as npt
from numba import njit
from pyproj import CRS, Transformer
from xmipy import XmiWrapper
from xmipy.errors import InputError

from geb.geb_types import (
    ArrayFloat,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt64,
    ArrayWithScalar,
    ThreeDArrayFloat64,
    ThreeDArrayWithScalar,
    TwoDArrayBool,
    TwoDArrayFloat,
    TwoDArrayFloat32,
    TwoDArrayFloat64,
    TwoDArrayWithScalar,
)
from geb.workflows.io import (
    WorkingDirectory,
    create_hash_from_parameters,
    read_hash,
    write_hash,
)
from geb.workflows.raster import decompress_with_mask

if TYPE_CHECKING:
    pass

MODFLOW_VERSION: str = "6.8.1"


@njit(cache=True)
def get_water_table_depth(
    layer_boundary_elevation: TwoDArrayFloat,
    head: TwoDArrayFloat,
    elevation: ArrayFloat,
    min_remaining_layer_storage_m: float,
) -> ArrayFloat64:
    """Calculate the water table depth.

    Iterates from the bottom layer to the top layer, checking if the head is below the top of the layer.
    If it is, the water table depth is calculated as the difference between the elevation and the head.

    For each layer, the min_remaining_layer_storage_m is subtracted from the head to ensure that a minimum amount of water
    is not considered as part of the water table.

    Args:
        layer_boundary_elevation: Elevation of the layer boundaries, in m.
        head: The heads of the model grid, in m.
        elevation: The topography or surface elevation of the model grid, in m.
        min_remaining_layer_storage_m: The minimum remaining layer storage in m.
            More storage cannot be abstracted with wells.

    Returns:
        The water table depth, in m.
    """
    water_table_depth = np.zeros(head.shape[1])
    for cell_ix in range(head.shape[1]):
        for layer_ix in range(head.shape[0] - 1, -1, -1):
            layer_head = head[layer_ix, cell_ix]

            # if the head is smaller than the top of the layer, the water table elevation is equal to the topogography minus head
            if (
                layer_head - min_remaining_layer_storage_m
                < layer_boundary_elevation[layer_ix, cell_ix]
            ):
                water_table_depth[cell_ix] = elevation[cell_ix] - max(
                    layer_boundary_elevation[layer_ix + 1, cell_ix],
                    min(layer_head, layer_boundary_elevation[layer_ix, cell_ix]),
                )
                break

            # else proceed to the next layer

        else:
            water_table_depth[cell_ix] = (
                elevation[cell_ix] - layer_boundary_elevation[0, cell_ix]
            )
    return water_table_depth


@njit(cache=True)
def get_groundwater_storage_m(
    layer_boundary_elevation: TwoDArrayFloat,
    head: TwoDArrayFloat,
    specific_yield: TwoDArrayFloat,
    specific_storage: TwoDArrayFloat,
    min_remaining_layer_storage_m: float = 0.0,
) -> ArrayFloat64:
    """Calculate the groundwater storage in meters.

    Args:
        layer_boundary_elevation: Elevation of the layer boundaries, in m.
        head: The heads of the model grid, in m.
        specific_yield: The specific yield of the model grid (-).
        specific_storage: The specific storage of the model grid (1/m).
        min_remaining_layer_storage_m: The minimum remaining layer storage in m.
            More storage cannot be abstracted with wells.

    Returns:
        The groundwater storage, in m.
    """
    storage = np.zeros(head.shape[1])
    for cell_ix in range(head.shape[1]):
        for layer_ix in range(head.shape[0]):
            layer_head = head[layer_ix, cell_ix]
            layer_top = layer_boundary_elevation[layer_ix, cell_ix]
            layer_bottom = layer_boundary_elevation[layer_ix + 1, cell_ix]
            layer_specific_yield = specific_yield[layer_ix, cell_ix]
            layer_thickness = layer_top - layer_bottom

            if layer_head >= layer_top:
                elastic_term = (
                    specific_storage[layer_ix, cell_ix]
                    * layer_thickness
                    * (layer_head - layer_top)
                )
                storage[cell_ix] += (
                    layer_thickness - min_remaining_layer_storage_m
                ) * layer_specific_yield + elastic_term
            elif layer_head - min_remaining_layer_storage_m > layer_bottom:
                storage[cell_ix] += (
                    layer_head - layer_bottom - min_remaining_layer_storage_m
                ) * layer_specific_yield
    return storage


@njit(cache=True)
def distribute_well_abstraction_m3_per_layer(
    well_rate: ArrayFloat,
    layer_boundary_elevation: TwoDArrayFloat,
    heads: TwoDArrayFloat,
    specific_yield: TwoDArrayFloat,
    area: ArrayFloat,
    min_remaining_layer_storage_m: float = 0.0,
) -> TwoDArrayFloat64:
    """Distribute the well abstraction rate over the layers.

    Abstraction is done from the top layer to the bottom layer.
    If the layer is dry, the next layer is considered until the well rate is fully distributed.

    Args:
        well_rate: The well rate, in m3/step. Negative values indicate abstraction.
        layer_boundary_elevation: Elevation of the layer boundaries, in m.
        heads: The heads of the model grid, in m.
        specific_yield: The specific yield of the model grid (-).
        area: The area of each cell, in m2.
        min_remaining_layer_storage_m: The minimum remaining layer storage in m.
            More storage cannot be abstracted with wells.

    Returns:
        The well rate per layer, in m3/step.
    """
    nlay, ncells = heads.shape
    well_rate_per_layer = np.zeros((nlay, ncells))
    for cell_ix in range(ncells):
        layer_area = area[cell_ix]
        remaining_well_rate = well_rate[cell_ix]
        for layer_ix in range(nlay):
            layer_head = heads[layer_ix, cell_ix]
            layer_top = layer_boundary_elevation[layer_ix, cell_ix]
            layer_bottom = layer_boundary_elevation[layer_ix + 1, cell_ix]

            groundwater_layer_top = min(layer_head, layer_top)
            if groundwater_layer_top - min_remaining_layer_storage_m > layer_bottom:
                layer_specific_yield = specific_yield[layer_ix, cell_ix]
                layer_storage = (
                    (
                        groundwater_layer_top
                        - layer_bottom
                        - min_remaining_layer_storage_m
                    )
                    * layer_specific_yield
                    * layer_area
                )

                well_rate_per_layer[layer_ix, cell_ix] = -min(
                    layer_storage, -remaining_well_rate
                )

                remaining_well_rate -= well_rate_per_layer[layer_ix, cell_ix]
                if remaining_well_rate == 0:
                    break

        assert remaining_well_rate > -1e-10, (
            "Well rate could not be distributed, layers are too dry"
        )  # leaving some tolerance for numerical errors

    assert np.allclose(well_rate_per_layer.sum(axis=0), well_rate)
    return well_rate_per_layer


def parse_boundary_heads(
    boundary_heads: ThreeDArrayFloat64,
    boundary_mask: TwoDArrayBool,
    ext_basin_mask: TwoDArrayBool,
    nlay: int,
    botm_ext: ThreeDArrayFloat64 | None = None,
) -> list[tuple[tuple[int, int], float]]:
    """Parse boundary heads into a list of tuples.

    Args:
        boundary_heads: The heads of the model grid, in m.
        boundary_mask: The boundary mask of the model grid.
        ext_basin_mask: The expanded basin mask of the model grid.
        nlay: The number of layers in the model grid.
        botm_ext: The bottom elevations of the model grid.

    Returns:
        A list of tuples of the form ((layer, cell), head).

    Raises:
        ValueError: If the shapes of the input arrays are incorrect
    """
    ext_nrow, ext_ncol = ext_basin_mask.shape

    if boundary_heads.shape != (nlay, ext_nrow, ext_ncol):
        raise ValueError(
            f"boundary_heads must have shape ({nlay}, {ext_nrow}, {ext_ncol}), "
            f"got {boundary_heads.shape}."
        )

    if boundary_mask.shape != (ext_nrow, ext_ncol):
        raise ValueError(
            f"boundary_mask must have shape ({ext_nrow}, {ext_ncol}), "
            f"got {boundary_mask.shape}."
        )

    chd_list: list[tuple[tuple[int, int], float]] = []

    for layer in range(nlay):
        for r in range(ext_nrow):
            for c in range(ext_ncol):
                if boundary_mask[r, c] and not ext_basin_mask[r, c]:
                    head_val: float = float(boundary_heads[layer, r, c])
                    if not np.isnan(head_val):
                        if botm_ext is not None:
                            min_head: float = float(botm_ext[layer, r, c]) + 0.05
                            if head_val < min_head:
                                head_val = min_head
                        cell_num: int = r * ext_ncol + c
                        chd_list.append(((layer, cell_num), head_val))

    if not chd_list:
        raise ValueError(
            "No active constant-head boundary cells were found with valid heads in boundary_heads."
        )

    return chd_list


class ModFlowSimulation:
    """Implements an instance of the MODFLOW model as well as methods to interact with it.

    Note:
        Communication of fluxes should only be done in m3. This is because the calculation
        of area in MODFLOW is slightly different from the area in GEB, which can lead to
        discrepancies in the fluxes if they are communicated in meters. This is also
        why all public methods of this class communicate in m3, and not in m.
    """

    area: ArrayFloat32
    heads: TwoDArrayFloat64
    _heads_ptr: ArrayFloat64
    _potential_well_rate_ptr: ArrayFloat64
    _actual_well_rate_ptr: ArrayFloat64
    _drainage_ptr: ArrayFloat64
    _recharge_ptr: ArrayFloat64
    _boundary_heads_ptr: ArrayFloat64
    _boundary_flow_ptr: ArrayFloat64
    _boundary_rate_in_ptr: ArrayFloat64
    _boundary_rate_out_ptr: ArrayFloat64
    _mxit_ptr: npt.NDArray[np.int32]
    chd_active_cell_indices: ArrayInt64

    def __init__(
        self,
        working_directory: Path,
        modflow_bin_folder: Path,
        topography: ArrayFloat32,
        gt: tuple[float, float, float, float, float, float],
        specific_storage: TwoDArrayFloat32,
        specific_yield: TwoDArrayFloat32,
        layer_boundary_elevation: TwoDArrayFloat32,
        basin_mask: TwoDArrayBool,
        hydraulic_conductivity: TwoDArrayFloat32,
        heads: TwoDArrayFloat64,
        heads_update_callback: Callable,
        logger: logging.Logger,
        boundary_heads: ThreeDArrayFloat64,
        boundary_mask: TwoDArrayBool,
        boundary_layer_boundary_elevation: ThreeDArrayFloat64,
        boundary_hydraulic_conductivity: ThreeDArrayFloat64,
        min_remaining_layer_storage_m: float = 0.1,
        verbose: bool = False,
        never_load_from_disk: bool = False,
    ) -> None:
        """Initialize the MODFLOW model.

        Args:
            working_directory: The working directory for the MODFLOW model.
            modflow_bin_folder: The folder containing the MODFLOW binaries.
            topography: The topography or surface elevation of the model grid (m).
            gt: The geotransform of the model grid (GDAL-style).
            specific_storage: The specific storage of the model grid (m-1).
            specific_yield: The specific yield of the model grid (-).
            layer_boundary_elevation: The elevation of the layer boundaries (m).
            basin_mask: A boolean mask indicating the active cells in the model grid.
            hydraulic_conductivity: The hydraulic conductivity of the model grid (m/day).
            heads: The initial heads of the model grid (m).
            heads_update_callback: A callback function to update the heads in the GEB model after each time step.
            logger: A logger instance to log information about the model.
            boundary_heads: A 3D array containing the boundary heads for the model grid.
            boundary_mask: A boolean mask indicating the boundary cells in the model grid.
            boundary_layer_boundary_elevation: The layer boundary elevations for the extended model grid (m).
            boundary_hydraulic_conductivity: The hydraulic conductivity for the extended model grid (m/day).
            min_remaining_layer_storage_m: The minimum remaining layer storage in m, defaults to 0.1. More storage cannot be abstracted with wells.
            verbose: Whether to print debug information, defaults to False.
            never_load_from_disk: Whether to never load the model from disk, defaults to False. If set to False, the model input
                will be loaded from disk if it exists and the input parameters have not changed.

        Raises:
            ValueError: If the shapes of the input arrays are incorrect.
        """
        self.logger = logger
        self.name = "MODEL"  # MODFLOW requires the name to be uppercase
        self.heads_update_callback = heads_update_callback
        self.basin_mask = basin_mask
        self.nlay = hydraulic_conductivity.shape[0]
        assert self.basin_mask.dtype == bool
        self.n_active_cells = self.basin_mask.size - self.basin_mask.sum()

        self.nrow, self.ncol = self.basin_mask.shape
        self.ext_nrow, self.ext_ncol = self.nrow + 2, self.ncol + 2

        self.ext_basin_mask = np.ones((self.ext_nrow, self.ext_ncol), dtype=bool)
        self.ext_basin_mask[1:-1, 1:-1] = self.basin_mask
        if boundary_mask.shape != (self.ext_nrow, self.ext_ncol):
            raise ValueError(
                f"boundary_mask must have shape ({self.ext_nrow}, {self.ext_ncol}), "
                f"got {boundary_mask.shape}."
            )
        internal_active = np.zeros((self.ext_nrow, self.ext_ncol), dtype=bool)
        internal_active[1:-1, 1:-1] = ~self.basin_mask
        if np.any(boundary_mask & internal_active):
            raise ValueError(
                "Boundary cells must be located strictly outside the active model domain."
            )
        self.ext_basin_mask[boundary_mask] = False
        self.n_mf_cells = self.ext_basin_mask.size - self.ext_basin_mask.sum()

        self.geb_to_mf = np.full(self.n_active_cells, -1, dtype=np.int64)
        self.mf_to_geb = np.full(self.n_mf_cells, -1, dtype=np.int64)
        geb_idx = 0
        mf_idx = 0
        for r in range(self.ext_nrow):
            for c in range(self.ext_ncol):
                if not self.ext_basin_mask[r, c]:
                    if (
                        1 <= r <= self.nrow
                        and 1 <= c <= self.ncol
                        and not self.basin_mask[r - 1, c - 1]
                    ):
                        self.geb_to_mf[geb_idx] = mf_idx
                        self.mf_to_geb[mf_idx] = geb_idx
                        geb_idx += 1
                    mf_idx += 1

        self.working_directory = working_directory
        os.makedirs(self.working_directory, exist_ok=True)
        self.verbose = verbose
        self.never_load_from_disk = never_load_from_disk
        self.min_remaining_layer_storage_m = min_remaining_layer_storage_m

        self.topography = topography
        self.layer_boundary_elevation = layer_boundary_elevation
        assert (self.topography >= self.layer_boundary_elevation[0]).all()
        self.specific_yield = specific_yield
        self.specific_storage = specific_storage
        self.hydraulic_conductivity_drainage = hydraulic_conductivity[0]

        self.ext_gt = (gt[0] - gt[1], gt[1], gt[2], gt[3] - gt[5], gt[4], gt[5])

        arguments = dict(locals())
        arguments.pop("working_directory")
        arguments.pop("modflow_bin_folder")
        arguments.pop("self")
        arguments.pop("logger")  # not hashable and not needed
        arguments.pop("heads_update_callback")  # not hashable and not needed
        arguments.pop(
            "heads"
        )  # heads is set after loading the model or writing to disk

        self.hash_file = Path(self.working_directory) / "input_hash"

        self.save_flows = False

        self.botm_ext = np.zeros(
            (self.nlay, self.ext_nrow, self.ext_ncol), dtype=np.float64
        )
        self.botm_ext[:, 1:-1, 1:-1] = self.decompress(
            self.layer_boundary_elevation[1:]
        )
        self.botm_ext[:, boundary_mask] = boundary_layer_boundary_elevation[1:][
            :, boundary_mask
        ]

        self.chd_data = parse_boundary_heads(
            boundary_heads=boundary_heads,
            boundary_mask=boundary_mask,
            ext_basin_mask=self.ext_basin_mask,
            nlay=self.nlay,
            botm_ext=self.botm_ext,
        )

        cell_to_mf_active = np.full(self.ext_nrow * self.ext_ncol, -1, dtype=np.int64)
        cell_to_mf_active[~self.ext_basin_mask.ravel()] = np.arange(
            self.n_mf_cells, dtype=np.int64
        )

        chd_cells = np.array([entry[0][1] for entry in self.chd_data], dtype=np.int64)
        self.chd_active_cell_indices = cell_to_mf_active[chd_cells]

        if not self.load_from_disk(arguments):
            try:
                if self.verbose:
                    self.logger.info("Creating MODFLOW model")
                sim = self.get_simulation(
                    self.ext_gt,
                    hydraulic_conductivity,
                    specific_storage,
                    specific_yield,
                    boundary_heads=boundary_heads,
                    boundary_mask=boundary_mask,
                    boundary_layer_boundary_elevation=boundary_layer_boundary_elevation,
                    boundary_hydraulic_conductivity=boundary_hydraulic_conductivity,
                )

                sim.write_simulation()
                write_hash(self.hash_file, self.hash)
            except:
                if self.hash_file.exists():
                    self.hash_file.unlink()
                raise
        elif self.verbose:
            self.logger.info("Loading MODFLOW model from disk")

        self.load_bmi(heads, modflow_bin_folder)

    def create_vertices(
        self,
        nrows: int,
        ncols: int,
        gt: tuple[float, float, float, float, float, float],
    ) -> tuple[TwoDArrayFloat64, TwoDArrayFloat64]:
        """Create the vertices of the model grid.

        Args:
            nrows: The number of rows in the model grid.
            ncols: The number of columns in the model grid.
            gt: The geotransform of the model grid (GDAL-style).

        Returns:
            A tuple of two 2D arrays containing the x and y coordinates of the vertices.
        """
        x_coordinates = np.linspace(gt[0], gt[0] + gt[1] * ncols, ncols + 1)
        y_coordinates = np.linspace(gt[3], gt[3] + gt[5] * nrows, nrows + 1)

        center_longitude = (x_coordinates[0] + x_coordinates[-1]) / 2
        center_latitude = (y_coordinates[0] + y_coordinates[-1]) / 2

        utm_crs: CRS = CRS.from_dict(
            {
                "proj": "utm",
                "ellps": "WGS84",
                "lat_0": center_latitude,
                "lon_0": center_longitude,
                "zone": int((center_longitude + 180) / 6) + 1,
            }
        )

        # Create a topography 2D map
        x_vertices, y_vertices = np.meshgrid(x_coordinates, y_coordinates)

        # convert to modflow coordinates
        transformer: Transformer = Transformer.from_crs(
            crs_from="EPSG:4326", crs_to=utm_crs, always_xy=True
        )

        # Transform the points
        x_transformed, y_transformed = transformer.transform(
            x_vertices.ravel(), y_vertices.ravel()
        )

        # Reshape back to the original grid shape
        x_transformed = x_transformed.reshape(x_vertices.shape)
        y_transformed = y_transformed.reshape(y_vertices.shape)

        return x_transformed, y_transformed

    def get_simulation(
        self,
        gt: tuple[float, float, float, float, float, float],
        hydraulic_conductivity: TwoDArrayFloat32,
        specific_storage: TwoDArrayFloat32,
        specific_yield: TwoDArrayFloat32,
        boundary_heads: ThreeDArrayFloat64,
        boundary_mask: TwoDArrayBool,
        boundary_layer_boundary_elevation: ThreeDArrayFloat64,
        boundary_hydraulic_conductivity: ThreeDArrayFloat64,
    ) -> flopy.mf6.MFSimulation:
        """Create a MODFLOW 6 simulation instance.

        Specific Storage is is the volume of water that a unit volume of
        a saturated aquifer material will release from or take into storage
        under a unit change in hydraulic head.

        Specific Yield is the volume of water that a unit volume of
        a saturated aquifer material will yield by gravity drainage. Also
        the 'drainable porosity'.

        Args:
            gt: The geotransform of the model grid (GDAL-style).
            hydraulic_conductivity: The hydraulic conductivity of the model grid (m/day).
            specific_storage: The specific storage of the model grid (m-1).
            specific_yield: The specific yield of the model grid (-).
            boundary_heads: The boundary heads of the model grid (m).
            boundary_mask: The boundary mask of the model grid.
            boundary_layer_boundary_elevation: The layer boundary elevations for the extended model grid (m).
            boundary_hydraulic_conductivity: The hydraulic conductivity for the extended model grid (m/day).

        Returns:
            The MODFLOW 6 simulation instance.
        """
        sim = flopy.mf6.MFSimulation(
            sim_name=self.name,
            version="mf6",
            sim_ws=os.path.realpath(self.working_directory),
            verbosity_level=0,
            write_headers=False,  # avoid writing flopy headers (not needed),
            print_input=False,  # avoid printing all input arrays/settings
        )
        number_of_periods: int = 1
        flopy.mf6.ModflowTdis(
            sim, nper=number_of_periods, perioddata=[(1.0, 1, 1)] * number_of_periods
        )

        # create iterative model solution
        flopy.mf6.ModflowIms(
            sim,
            print_option=None,
            complexity="SIMPLE",
            outer_maximum=100,
            inner_maximum=200,
            linear_acceleration="BICGSTAB",
        )

        # create groundwater flow model
        groundwater_flow = flopy.mf6.ModflowGwf(
            sim,
            modelname=self.name,
            newtonoptions="under_relaxation",
            print_input=self.save_flows,
            print_flows=self.save_flows,
        )

        # 1. Create vertices
        x_coordinates_vertices, y_coordinates_vertices = self.create_vertices(
            self.ext_nrow, self.ext_ncol, gt
        )
        vertices = [
            [i, x, y]
            for i, (x, y) in enumerate(
                zip(
                    x_coordinates_vertices.ravel(),
                    y_coordinates_vertices.ravel(),
                )
            )
        ]

        # 2. Create cell2d array
        cell2d = []
        xy_to_cell = np.full((self.ext_nrow, self.ext_ncol), -1, dtype=int)
        cell_areas = np.full((self.ext_nrow, self.ext_ncol), np.nan, dtype=np.float32)
        n_vert_x = self.ext_ncol + 1

        for row in range(self.ext_nrow):
            for column in range(self.ext_ncol):
                cell_number = row * self.ext_ncol + column
                xy_to_cell[row, column] = cell_number
                # not here that the vertices are 1 larger than the number of cells
                # therefore an additional offset of 1 is required for each row
                # thus adding 'row' to v1 to account for the offset
                v1 = row * n_vert_x + column  # top-left vertex
                v2 = v1 + 1  # top-right vertex
                v3 = (row + 1) * n_vert_x + column + 1  # bottom-right vertex
                v4 = (row + 1) * n_vert_x + column  # bottom-left vertex

                cell_center_x = (
                    x_coordinates_vertices[row, column]
                    + x_coordinates_vertices[row, column + 1]
                ) / 2
                cell_center_y = (
                    y_coordinates_vertices[row, column]
                    + y_coordinates_vertices[row + 1, column]
                ) / 2

                cell_area = (
                    y_coordinates_vertices[row + 1, column]
                    - y_coordinates_vertices[row, column]
                ) * (
                    x_coordinates_vertices[row, column]
                    - x_coordinates_vertices[row, column + 1]
                )
                assert cell_area > 0
                cell_areas[row, column] = cell_area

                cell = [
                    cell_number,
                    cell_center_x,
                    cell_center_y,
                    4,
                    v1,
                    v2,
                    v3,
                    v4,
                ]
                cell2d.append(cell)

        interior_verts = set()
        for r in range(self.ext_nrow):
            for c in range(self.ext_ncol):
                if not self.ext_basin_mask[r, c] and not boundary_mask[r, c]:
                    v1 = r * n_vert_x + c
                    v2 = v1 + 1
                    v3 = (r + 1) * n_vert_x + c + 1
                    v4 = (r + 1) * n_vert_x + c
                    interior_verts.update([v1, v2, v3, v4])

        for r in range(self.ext_nrow):
            for c in range(self.ext_ncol):
                if boundary_mask[r, c] and not self.ext_basin_mask[r, c]:
                    cell_idx = r * self.ext_ncol + c
                    old_verts = cell2d[cell_idx][4:8]
                    new_verts = []
                    for v in old_verts:
                        if v in interior_verts:
                            new_verts.append(v)
                        else:
                            new_v_id = len(vertices)
                            orig_x, orig_y = vertices[v][1], vertices[v][2]
                            vertices.append([new_v_id, orig_x, orig_y])
                            new_verts.append(new_v_id)
                    cell2d[cell_idx][4:8] = new_verts

        cell_areas = cell_areas[~self.ext_basin_mask]
        mf_active_cells = xy_to_cell[~self.ext_basin_mask].ravel()

        domain = np.stack([~self.ext_basin_mask] * self.nlay)

        top_ext = np.zeros((self.ext_nrow, self.ext_ncol), dtype=np.float64)
        top_ext[1:-1, 1:-1] = self.decompress(self.layer_boundary_elevation[0])
        top_ext[boundary_mask] = boundary_layer_boundary_elevation[0][boundary_mask]

        botm_ext = self.botm_ext

        k = np.zeros((self.nlay, self.ext_nrow, self.ext_ncol), dtype=np.float64)
        k[:, 1:-1, 1:-1] = self.decompress(hydraulic_conductivity)
        k[:, boundary_mask] = boundary_hydraulic_conductivity[:, boundary_mask]

        strt_data = np.zeros_like(k, dtype=np.float64)
        for layer in range(self.nlay):
            strt_data[layer] = top_ext - 1.0

        flopy.mf6.ModflowGwfdisv(
            groundwater_flow,
            nlay=self.nlay,
            ncpl=self.ext_nrow * self.ext_ncol,
            nvert=len(vertices),
            vertices=vertices,
            cell2d=cell2d,
            top={
                "filename": "top.bin",
                "factor": 1.0,
                "data": top_ext.tolist(),
                "iprn": 1,
                "binary": True,
            },
            botm={
                "filename": "botm.bin",
                "factor": 1.0,
                "data": botm_ext.tolist(),
                "iprn": 1,
                "binary": True,
            },
            idomain={
                "filename": "idomain.bin",
                "factor": 1.0,
                "data": domain.astype(np.int32).tolist(),
                "iprn": 1,
                "binary": True,
            },
        )

        flopy.mf6.ModflowGwfic(
            groundwater_flow,
            strt={
                "filename": "strt.bin",
                "factor": 1.0,
                "data": strt_data.tolist(),
                "iprn": 1,
                "binary": True,
            },
        )

        icelltype = np.ones_like(domain, dtype=np.int32)
        flopy.mf6.ModflowGwfnpf(
            groundwater_flow,
            save_flows=self.save_flows,
            print_flows=self.save_flows,
            icelltype={
                "filename": "icelltype.bin",
                "data": icelltype,
                "iprn": 11,
                "binary": True,
            },
            k={
                "filename": "k.bin",
                "factor": 1.0,
                "data": k.astype(np.float64),
                "iprn": 1,
                "binary": True,
            },
        )

        specific_storage_ext = np.zeros(
            (self.nlay, self.ext_nrow, self.ext_ncol), dtype=np.float64
        )
        specific_storage_ext[:, 1:-1, 1:-1] = self.decompress(specific_storage)

        specific_yield_ext = np.zeros(
            (self.nlay, self.ext_nrow, self.ext_ncol), dtype=np.float64
        )
        specific_yield_ext[:, 1:-1, 1:-1] = self.decompress(specific_yield)
        # Somehow modeltime is not available when loading_package is set to False (the default) and what it should be.
        # when loading_package is set to True, the model builds fine but the simulation doesn't work.
        # TODO: See if this is fixed in a future version of flopy/modflow6, and perhaps file an issue.
        # flopy.mf6.ModflowGwfsto(
        #     groundwater_flow,
        #     save_flows=self.save_flows,
        #     iconvert=1,
        #     ss={
        #         "filename": "ss.bin",
        #         "data": specific_storage.astype(np.float64),
        #         "binary": True,
        #     },
        #     sy={
        #         "filename": "sy.bin",
        #         "data": specific_yield.astype(np.float64),
        #         "binary": True,
        #     },
        #     steady_state=False,
        #     transient=True,
        #     loading_package=False,
        #     pname="sto",
        #     filename="model.sto",
        # )

        flopy.mf6.ModflowGwfsto(
            groundwater_flow,
            save_flows=self.save_flows,
            iconvert=1,
            ss=specific_storage_ext.astype(np.float64),
            sy=specific_yield_ext.astype(np.float64),
            steady_state=False,
            transient=True,
            pname="sto",
        )

        internal_cell_ids = mf_active_cells[self.geb_to_mf]

        recharge = []
        for cell in mf_active_cells:
            recharge.append(
                (0, cell, 0.0)
            )  # specifying the layer, cell number, and recharge rate

        recharge = flopy.mf6.ModflowGwfrch(
            groundwater_flow,
            fixed_cell=True,
            save_flows=self.save_flows,
            maxbound=len(recharge),
            stress_period_data={
                0: {
                    "filename": "recharge.bin",
                    "factor": 1.0,
                    "data": recharge,
                    "iprn": 1,
                    "binary": True,
                },
            },
        )

        # Wells
        wells = []
        for layer in range(self.nlay):
            for cell in internal_cell_ids:
                wells.append(
                    (layer, cell, 0.0)
                )  # specifying the layer, cell number, and well rate

        wells = flopy.mf6.ModflowGwfwel(
            groundwater_flow,
            maxbound=len(wells),
            stress_period_data={
                0: {
                    "filename": "wells.bin",
                    "factor": 1.0,
                    "data": wells,
                    "iprn": 1,
                    "binary": True,
                },
            },
            save_flows=self.save_flows,
        )

        # Drainage
        # Drainage rate is set as conductivity * area / drainage length
        # For conductivity we set the conductivity of the top layer
        # area the total size of the cell, and as we are are approximating
        # transmissivity, we can set the drainage length to 1
        drainage = []
        for geb_idx, mf_idx in enumerate(self.geb_to_mf):
            drn_cell_id = mf_active_cells[mf_idx]
            drn_rate = (
                self.hydraulic_conductivity_drainage[geb_idx] * cell_areas[mf_idx] / 1
            )
            drn_elev = self.layer_boundary_elevation[0, geb_idx]
            drainage.append((0, drn_cell_id, drn_elev, drn_rate))

        flopy.mf6.ModflowGwfdrn(
            groundwater_flow,
            maxbound=len(drainage),
            stress_period_data={
                0: {
                    "filename": "drainage.bin",
                    "factor": 1.0,
                    "data": drainage,
                    "iprn": 1,
                    "binary": True,
                },
            },
            print_flows=self.save_flows,
            save_flows=self.save_flows,
        )

        flopy.mf6.ModflowGwfchd(
            groundwater_flow,
            maxbound=len(self.chd_data),
            stress_period_data={
                0: {
                    "filename": "chd.bin",
                    "factor": 1.0,
                    "data": self.chd_data,
                    "iprn": 1,
                    "binary": True,
                },
            },
            save_flows=self.save_flows,
            pname="chd",
        )

        flopy.mf6.ModflowGwfoc(
            groundwater_flow,
            pname="oc",
            head_filerecord=f"{self.name}.hds",
            budget_filerecord=f"{self.name}.cbc",
            saverecord=[
                ("HEAD", "LAST"),  # Saves the HEAD file at last timestep
                ("BUDGET", "LAST"),  # Saves the BUDGET file at last timestep
            ],
            printrecord=[
                ("HEAD", "LAST"),  # Prints to LST file only at the last step
                (
                    "BUDGET",
                    "LAST",
                ),  # Prints budget summary to LST file only at the last step
            ],
        )

        sim.simulation_data.set_sci_note_upper_thres(
            1e99
        )  # effectively disable scientific notation
        sim.simulation_data.set_sci_note_lower_thres(
            1e-99
        )  # effectively disable scientific notation

        return sim

    def load_from_disk(self, arguments: dict[str, Any]) -> bool:
        """Check if the model input has changed and load from disk if not.

        If self.never_load_from_disk is True, the model will never be loaded from disk.

        Args:
            arguments: The input arguments to hash.

        Returns:
            True if the model input has not changed and the model can be loaded from disk, False otherwise.
        """
        hashable_dict = {}
        for key, value in arguments.items():
            if isinstance(value, np.ndarray):
                value = str(value.tobytes())
            hashable_dict[key] = value

        self.hash = create_hash_from_parameters(arguments, code_path=Path(__file__))
        if self.hash_file.exists():
            prev_hash = read_hash(self.hash_file)
        else:
            prev_hash = None

        if prev_hash == self.hash and not self.never_load_from_disk:
            return True
        else:
            return False

    def bmi_return(self) -> list[str]:
        """Parse the stdout file created by the modflow library.

        stdout is a file created by the modflow library that contains
        information about the model run.

        Returns:
            The contents of the stdout file as a list of strings.
        """
        with open("mfsim.stdout") as f:
            return f.readlines()

    def load_bmi(self, heads: TwoDArrayFloat64, modflow_bin_folder: Path) -> None:
        """Load the Basic Model Interface.

        Args:
            heads: The initial heads of the model grid, in m.
            modflow_bin_folder: The folder containing the MODFLOW binaries.

        Raises:
            FileNotFoundError: If the config file is not found on disk.
            ValueError: If the platform is not supported.
        """
        if platform.system() == "Windows":
            libary_name: str = "libmf6.dll"
        elif platform.system() == "Linux":
            libary_name: str = "libmf6.so"
        elif platform.system() == "Darwin":
            libary_name: str = "libmf6.dylib"
        else:
            raise ValueError(f"Platform {platform.system()} not supported.")

        with WorkingDirectory(self.working_directory):
            # XmiWrapper requires the real path (no symlinks etc.)
            # include the version in the folder name to allow updating the version
            # so that the user will automatically get the new version
            library_folder: Path = (modflow_bin_folder / MODFLOW_VERSION).resolve()
            library_path: Path = library_folder / libary_name

            if not library_path.exists():
                library_folder.mkdir(exist_ok=True, parents=True)

                flopy.utils.get_modflow(
                    bindir=str(library_folder),
                    repo="modflow6",
                    subset=[libary_name],
                    release_id=MODFLOW_VERSION,
                )

            assert os.path.exists(library_path)
            try:
                self.mf6 = XmiWrapper(library_path)
            except Exception as e:
                self.logger.error("Failed to load " + str(library_path))
                self.logger.error("with message: " + str(e))
                self.bmi_return()
                raise

            # modflow requires the real path (no symlinks etc.)
            config_file: str = os.path.realpath("mfsim.nam")
            if not os.path.exists(config_file):
                raise FileNotFoundError(
                    f"Config file {config_file} not found on disk. Did you create the model first (load_from_disk = False)?"
                )

            # initialize the model
            try:
                self.mf6.initialize(config_file)
            except:
                self.bmi_return()
                raise

            if self.verbose:
                self.logger.debug("MODFLOW model initialized")

        area_tag: str = self.mf6.get_var_address("AREA", self.name, "DIS")
        mf_area = self.mf6.get_value_ptr(area_tag).reshape(self.nlay, self.n_mf_cells)
        # ensure that the areas of all vertical cells are equal
        assert (np.diff(mf_area, axis=0) == 0).all()

        self.area = mf_area[0, self.geb_to_mf].astype(np.float32)

        # Cache frequently accessed BMI pointers. MODFLOW updates these arrays
        # in place, so repeated pointer retrieval is unnecessary.
        self._heads_ptr = self.mf6.get_value_ptr(
            self.mf6.get_var_address("X", self.name)
        )
        self._potential_well_rate_ptr = self.mf6.get_value_ptr(
            self.mf6.get_var_address("Q", self.name, "WEL_0")
        )
        self._actual_well_rate_ptr = self.mf6.get_value_ptr(
            self.mf6.get_var_address("SIMVALS", self.name, "WEL_0")
        )
        self._drainage_ptr = self.mf6.get_value_ptr(
            self.mf6.get_var_address("SIMVALS", self.name, "DRN_0")
        )
        self._recharge_ptr = self.mf6.get_value_ptr(
            self.mf6.get_var_address("RECHARGE", self.name, "RCH_0")
        )
        self._mxit_ptr = self.mf6.get_value_ptr(
            self.mf6.get_var_address("MXITER", "SLN_1")
        )

        self.prepare_time_step()

        chd_head_tag = self.mf6.get_var_address("HEAD", self.name, "CHD")
        self._boundary_heads_ptr = self.mf6.get_value_ptr(chd_head_tag)
        self._boundary_flow_ptr = self.mf6.get_value_ptr(
            self.mf6.get_var_address("SIMVALS", self.name, "CHD")
        )
        self._boundary_rate_in_ptr = self.mf6.get_value_ptr(
            self.mf6.get_var_address("RATECHDIN", self.name, "CHD")
        )
        self._boundary_rate_out_ptr = self.mf6.get_value_ptr(
            self.mf6.get_var_address("RATECHDOUT", self.name, "CHD")
        )

        self.heads = heads

        for chd_idx, (coord, b_head) in enumerate(self.chd_data):
            act_idx = self.chd_active_cell_indices[chd_idx]
            self._heads_ptr[coord[0] * self.n_mf_cells + act_idx] = b_head
            self._boundary_heads_ptr[chd_idx] = b_head
        assert not np.isnan(self.heads).any()

    @property
    def boundary_heads(self) -> ArrayFloat64:
        """Get the boundary heads.

        Returns:
            The boundary heads, in m.
        """
        return self._boundary_heads_ptr.copy()

    def set_boundary_heads(self, boundary_heads: ArrayFloat64) -> None:
        """Set the boundary heads.

        Args:
            boundary_heads: The boundary heads to set, in m.

        Raises:
            ValueError: If the boundary heads contain NaN values.
            ValueError: If the boundary heads shape is not equal to the expected shape.
        """
        if np.isnan(boundary_heads).any():
            raise ValueError("Boundary heads cannot contain NaN values.")
        if boundary_heads.shape != self._boundary_heads_ptr.shape:
            raise ValueError(
                f"Expected boundary heads shape {self._boundary_heads_ptr.shape}, got {boundary_heads.shape}."
            )
        clamped_heads = boundary_heads.copy()
        for chd_idx, (coord, _) in enumerate(self.chd_data):
            cell_num = coord[1]
            r = cell_num // self.ext_ncol
            c = cell_num % self.ext_ncol
            min_head = float(self.botm_ext[coord[0], r, c]) + 0.05
            if clamped_heads[chd_idx] < min_head:
                clamped_heads[chd_idx] = min_head

        self._boundary_heads_ptr[:] = clamped_heads

        for chd_idx, (coord, _) in enumerate(self.chd_data):
            act_idx = self.chd_active_cell_indices[chd_idx]
            self._heads_ptr[coord[0] * self.n_mf_cells + act_idx] = clamped_heads[
                chd_idx
            ]

    @property
    def boundary_flow_m3(self) -> ArrayFloat64:
        """Get net boundary flux across all boundary cells (m3/step).

        Returns:
            Net boundary flux for active basin cells (m3/step).
        """
        return self._boundary_flow_ptr.copy()

    @property
    def boundary_inflow_m3(self) -> ArrayFloat64:
        """Get total boundary inflow (m3/step).

        Returns:
            Boundary inflow for active basin cells (m3/step).
        """
        return np.maximum(0.0, self.boundary_flow_m3)

    @property
    def boundary_outflow_m3(self) -> ArrayFloat64:
        """Get total boundary outflow (m3/step).

        Returns:
            Boundary outflow for active basin cells (m3/step).
        """
        return np.maximum(0.0, -self.boundary_flow_m3)

    @property
    def heads(self) -> TwoDArrayFloat64:
        """Get the heads of the model grid for all layers.

        Returns:
            The heads of the model grid, in m.
        """
        mf_heads = self._heads_ptr.reshape(self.nlay, self.n_mf_cells)
        geb_heads = mf_heads[:, self.geb_to_mf]
        assert not np.isnan(geb_heads).any()
        return geb_heads

    @heads.setter
    def heads(self, value: TwoDArrayFloat64) -> None:
        """Set the heads of the model grid.

        Args:
            value: The heads to set, in m.
        """
        mf_heads = self._heads_ptr.reshape(self.nlay, self.n_mf_cells)
        for layer in range(self.nlay):
            mf_heads[layer, self.geb_to_mf] = value[layer]

    @property
    def groundwater_depth(self) -> ArrayFloat64:
        """Get the groundwater depth.

        Returns:
            The groundwater depth, in m.
        """
        groundwater_depth_m = get_water_table_depth(
            self.layer_boundary_elevation,
            self.heads,
            self.topography,
            min_remaining_layer_storage_m=self.min_remaining_layer_storage_m,
        )
        assert (groundwater_depth_m >= 0).all()
        return groundwater_depth_m

    @property
    def groundwater_content_m(self) -> ArrayFloat64:
        """Get the groundwater content in meters.

        Returns:
            The groundwater content, in m.
        """
        groundwater_content_m = get_groundwater_storage_m(
            self.layer_boundary_elevation,
            self.heads,
            self.specific_yield,
            self.specific_storage,
        )
        assert (groundwater_content_m >= 0).all()
        return groundwater_content_m

    @property
    def groundwater_content_m3(self) -> ArrayFloat64:
        """Get the groundwater content in cubic meters.

        Returns:
            The groundwater content, in m3.
        """
        return self.groundwater_content_m * self.area

    @property
    def available_groundwater_m(self) -> ArrayFloat64:
        """Get the available groundwater content in meters.

        Returns:
            The available groundwater content, in m.
        """
        groundwater_available_m = get_groundwater_storage_m(
            self.layer_boundary_elevation,
            self.heads,
            self.specific_yield,
            self.specific_storage,
            min_remaining_layer_storage_m=self.min_remaining_layer_storage_m,
        )
        assert (groundwater_available_m >= 0).all()
        return groundwater_available_m

    @property
    def available_groundwater_m3(self) -> ArrayFloat64:
        """Get the available groundwater content in cubic meters.

        Returns:
            The available groundwater content, in m3.
        """
        return self.available_groundwater_m * self.area

    @property
    def potential_well_rate(self) -> ArrayFloat64:
        """Get the potential well rate, value in m3/step.

        The potential well rate is the rate that is requested by the user. If more
        groundwater is requested than is available, the actual well rate will be lower.

        Returns:
            The potential well rate, value in m3/step.
        """
        return self._potential_well_rate_ptr

    @property
    def actual_well_rate(self) -> ArrayFloat64:
        """Get the actual simulated well rate, value in m3/step."""
        return self._actual_well_rate_ptr

    @potential_well_rate.setter
    def potential_well_rate(self, well_rate: ArrayFloat64) -> None:
        """Set the potential well rate, value in m3/step.

        Negative values indicate abstraction. Positive values result in injection.

        Args:
            well_rate: The potential well rate to set, value in m3/step.
        """
        well_rate_per_layer = distribute_well_abstraction_m3_per_layer(
            well_rate,
            self.layer_boundary_elevation,
            self.heads,
            self.specific_yield,
            self.area,
            min_remaining_layer_storage_m=np.float64(
                self.min_remaining_layer_storage_m
            ),
        ).ravel()
        self._potential_well_rate_ptr[:] = well_rate_per_layer

    @property
    def drainage_m3(self) -> npt.NDArray[np.float64]:
        """Get the drainage, value in m3/step.

        Returns:
            The drainage, value in m3/step.
        """
        drainage = -self._drainage_ptr
        assert not np.isnan(drainage).any()
        # TODO: This assert can become more strict when soil depth is considered
        assert (drainage / self.area < self.hydraulic_conductivity_drainage * 100).all()
        return drainage

    @property
    def _drainage_m(self) -> npt.NDArray[np.float64]:
        return self.drainage_m3 / self.area

    @property
    def _recharge_m(self) -> npt.NDArray[np.float64]:
        """Get the groundwater recharge for active GEB cells.

        Returns:
            Recharge rates for active basin cells (meters/step).
        """
        recharge = self._recharge_ptr[self.geb_to_mf].copy()
        assert not np.isnan(recharge).any()
        return recharge

    @_recharge_m.setter
    def recharge_m(self, value: ArrayFloat32 | npt.NDArray[np.float64]) -> None:
        """Set the groundwater recharge for active GEB cells.

        Maps the active cell recharge rates to the MODFLOW active grid
        via geb_to_mf. Boundary cells are located outside the domain and receive 0 recharge.

        Args:
            value: Recharge rate for active basin cells (meters/step).
        """
        assert not np.isnan(value).any()
        self._recharge_ptr[self.geb_to_mf] = value

    @property
    def recharge_m3(self) -> npt.NDArray[np.float64]:
        """Get the groundwater recharge volume for active GEB cells.

        Returns:
            Recharge volume for active basin cells (m3/step).
        """
        return self._recharge_m * self.area

    @property
    def max_iter(self) -> int:
        """Get the maximum number of iterations allowed for the solver.

        Returns:
            The maximum number of iterations.
        """
        return int(self._mxit_ptr[0])

    def prepare_time_step(self) -> None:
        """Prepare the model for the next time step."""
        dt: float = self.mf6.get_time_step()
        self.mf6.prepare_time_step(dt)

    def set_recharge_m3(self, recharge: ArrayFloat32) -> None:
        """Set recharge, value in m3/step.

        Args:
            recharge: Recharge volume for active basin cells (m3/step).
        """
        assert not np.isnan(recharge).any()
        assert (recharge >= 0).all()
        self.recharge_m = recharge / self.area

    def set_groundwater_abstraction_m3(
        self, groundwater_abstraction: ArrayFloat64
    ) -> None:
        """Set well rate, value in m3/step."""
        assert not np.isnan(groundwater_abstraction).any()

        assert (self.available_groundwater_m3 >= groundwater_abstraction).all(), (
            "Requested groundwater abstraction exceeds available groundwater storage. "
        )

        well_rate = -groundwater_abstraction
        assert (well_rate <= 0).all()
        self.potential_well_rate = well_rate

    def step(self) -> None:
        """Perform a single time step of the model.

        This method on purpose does not advance the time step of MODFLOW, but
        instead re-solves the current time step. This allows the input files
        to be as simple as possible, without the need to specify multiple time steps
        in advance. We instead use the BMI interface to set the data for the
        'current' time step, and then re-solve the time step.
        """
        t0 = time()
        # loop over subcomponents
        n_solutions = self.mf6.get_subcomponent_count()
        for solution_id in range(1, n_solutions + 1):
            # convergence loop
            kiter = 0
            self.mf6.prepare_solve(solution_id)
            while kiter < self.max_iter:
                has_converged = self.mf6.solve(solution_id)
                kiter += 1

                if has_converged:
                    break
            else:
                self.logger.error("MODFLOW did not converge")

            self.mf6.finalize_solve(solution_id)

        assert not np.isnan(self.heads).any()
        assert np.array_equal(self.actual_well_rate, self.potential_well_rate)
        assert not np.isnan(self.heads).any()
        assert not np.isnan(self.recharge_m).any()
        assert not np.isnan(self.potential_well_rate).any()
        assert not np.isnan(self.groundwater_content_m).any()
        assert not np.isnan(self.heads[-1] - self.layer_boundary_elevation[-1]).any()

        if self.verbose:
            self.logger.debug("MODFLOW")
            self.logger.debug(
                f"\ttimestep {int(self.mf6.get_current_time())} converged in {round(time() - t0, 2)} seconds"
            )
            self.logger.debug(
                "\tHead statictics: mean",
                self.heads.mean(),
                "min",
                self.heads.min(),
                "max",
                self.heads.max(),
            )
            self.logger.debug(
                "\tGroundwater depth: mean",
                self.groundwater_depth.mean(),
                "min",
                self.groundwater_depth.min(),
                "max",
                self.groundwater_depth.max(),
            )
            self.logger.debug(
                "\tGroundwater content: mean", self.groundwater_content_m3.mean()
            )
            self.logger.debug(
                "\tRecharge (mean)",
                (self.recharge_m * self.area).mean(),
                "m3",
                self.recharge_m.mean(),
                "m",
            )
            self.logger.debug(
                "\tAbstraction (mean)",
                self.actual_well_rate.mean(),
                "m3",
                (self.actual_well_rate.sum(axis=0) / self.area).mean(),
                "m",
            )
            self.logger.debug(
                "\tDrainage (mean)",
                self.drainage_m3.mean(),
                "m3",
                self._drainage_m.mean(),
                "m",
            )

        self.heads_update_callback(self.heads)

    def finalize(self) -> None:
        """Finalize the model.

        This method should be called at the end of the model run to ensure that all
        resources are properly released.

        If the model has already been finalized or was never
        initialised, this method will silently pass.
        """
        try:
            self.mf6.finalize()
        except InputError:
            pass
        self.logger.info("MODFLOW model finalized")

    def restore(self, heads: TwoDArrayFloat64) -> None:
        """Restore the model to a previous state by setting the heads.

        Args:
            heads: The heads to set, in m.
        """
        self.heads = heads

    @overload
    def decompress(
        self,
        array: TwoDArrayWithScalar,
    ) -> ThreeDArrayWithScalar: ...

    @overload
    def decompress(
        self,
        array: ArrayWithScalar,
    ) -> TwoDArrayWithScalar: ...

    def decompress(
        self,
        array: TwoDArrayWithScalar | ArrayWithScalar,
    ) -> ThreeDArrayWithScalar | TwoDArrayWithScalar:
        """Decompress a compressed array using the model's grid.

        Args:
            array: The compressed array to decompress.

        Returns:
            The decompressed array.
        """
        return decompress_with_mask(
            array,
            self.basin_mask,
        )
