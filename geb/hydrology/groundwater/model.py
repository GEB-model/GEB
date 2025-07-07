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

import hashlib
import json
import os
import platform
from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import Callable

import flopy
import numpy as np
import numpy.typing as npt
from numba import njit
from pyproj import CRS, Transformer
from xmipy import XmiWrapper

MODFLOW_VERSION: str = "6.6.2"


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


@njit(cache=True)
def get_water_table_depth(
    layer_boundary_elevation, head, elevation, min_remaining_layer_storage_m
):
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
    layer_boundary_elevation,
    head,
    specific_yield,
    min_remaining_layer_storage_m=0.0,
):
    storage = np.zeros(head.shape[1])
    for cell_ix in range(head.shape[1]):
        for layer_ix in range(head.shape[0]):
            layer_head = head[layer_ix, cell_ix]
            layer_top = layer_boundary_elevation[layer_ix, cell_ix]
            layer_bottom = layer_boundary_elevation[layer_ix + 1, cell_ix]
            layer_specific_yield = specific_yield[layer_ix, cell_ix]

            groundwater_layer_top = min(layer_head, layer_top)
            if groundwater_layer_top - min_remaining_layer_storage_m > layer_bottom:
                storage[cell_ix] += (
                    groundwater_layer_top - layer_bottom - min_remaining_layer_storage_m
                ) * layer_specific_yield
    return storage


@njit(cache=True)
def distribute_well_rate_per_layer(
    well_rate,
    layer_boundary_elevation,
    heads,
    specific_yield,
    area,
    min_remaining_layer_storage_m=0.0,
):
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
        )  # leaving some tollerance for numerical errors

    assert np.allclose(well_rate_per_layer.sum(axis=0), well_rate)
    return well_rate_per_layer


class ModFlowSimulation:
    """Implements an instance of the MODFLOW model as well as methods to interact with it.

    Args:
        model: The GEB model instance.
        topography: The topography or surface elevation of the model grid.
        gt: The geotransform of the model grid (GDAL-style).
        specific_storage: The specific storage of the model grid, in m-1.
        specific_yield: The specific yield of the model grid, in m-1.
        layer_boundary_elevation: The elevation of the layer boundaries, in m.
        basin_mask: A boolean mask indicating the active cells in the model grid.
        hydraulic_conductivity: The hydraulic conductivity of the model grid, in m/day.
        heads: The initial heads of the model grid, in m.
        heads_update_callback: A callback function to update the heads in the GEB model after each time step.
        min_remaining_layer_storage_m: The minimum remaining layer storage in m, defaults to 0.1. More storage cannot be abstracted with wells.
        verbose: Whether to print debug information, defaults to False.
        never_load_from_disk: Whether to never load the model from disk, defaults to False. If set to False, the model input
            will be loaded from disk if it exists and the input parameters have not changed.

    Note:
        Communication of fluxes should only be done in m3. This is because the calculation
        of area in MODFLOW is slightly different from the area in GEB, which can lead to
        discrepancies in the fluxes if they are communicated in meters. This is also
        why all public methods of this class communicate in m3, and not in m.
    """

    def __init__(
        self,
        model,
        topography: npt.NDArray[np.float32],
        gt: tuple[float, float, float, float, float, float],
        specific_storage: npt.NDArray[np.float32],
        specific_yield: npt.NDArray[np.float32],
        layer_boundary_elevation: npt.NDArray[np.float32],
        basin_mask: npt.NDArray[np.bool_],
        hydraulic_conductivity: npt.NDArray[np.float32],
        heads: npt.NDArray[np.float64],
        heads_update_callback: Callable,
        min_remaining_layer_storage_m: float = 0.1,
        verbose: bool = False,
        never_load_from_disk: bool = False,
    ):
        self.name = "MODEL"  # MODFLOW requires the name to be uppercase
        self.model = model
        self.heads_update_callback = heads_update_callback
        self.basin_mask = basin_mask
        self.nlay = hydraulic_conductivity.shape[0]
        assert self.basin_mask.dtype == bool
        self.n_active_cells = self.basin_mask.size - self.basin_mask.sum()
        self.working_directory = model.simulation_root_spinup / "modflow_model"
        os.makedirs(self.working_directory, exist_ok=True)
        self.verbose = verbose
        self.never_load_from_disk = never_load_from_disk

        self.min_remaining_layer_storage_m = min_remaining_layer_storage_m

        self.topography = topography
        self.layer_boundary_elevation = layer_boundary_elevation
        assert (self.topography >= self.layer_boundary_elevation[0]).all()
        self.specific_yield = specific_yield
        hydraulic_conductivity = hydraulic_conductivity
        self.hydraulic_conductivity_drainage = hydraulic_conductivity[0]

        arguments = dict(locals())
        arguments.pop("self")
        arguments.pop("model")
        arguments.pop("heads_update_callback")  # not hashable and not needed
        arguments.pop(
            "heads"
        )  # heads is set after loading the model or writing to disk

        self.hash_file = os.path.join(self.working_directory, "input_hash")

        self.save_flows = False

        if not self.load_from_disk(arguments):
            try:
                if self.verbose:
                    print("Creating MODFLOW model")

                sim = self.get_simulation(
                    gt,
                    hydraulic_conductivity,
                    specific_storage,
                    specific_yield,
                )

                sim.write_simulation()
                self.write_hash_to_disk()
            except:
                if os.path.exists(self.hash_file):
                    os.remove(self.hash_file)
                raise
            # sim.run_simulation()
        elif self.verbose:
            print("Loading MODFLOW model from disk")

        self.load_bmi(heads)

    def create_vertices(self, nrows, ncols, gt):
        x_coordinates = np.linspace(gt[0], gt[0] + gt[1] * ncols, ncols + 1)
        y_coordinates = np.linspace(gt[3], gt[3] + gt[5] * nrows, nrows + 1)

        center_longitude = (x_coordinates[0] + x_coordinates[-1]) / 2
        center_latitude = (y_coordinates[0] + y_coordinates[-1]) / 2

        utm_crs = CRS.from_dict(
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
        transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

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
        gt,
        hydraulic_conductivity,
        specific_storage,
        specific_yield,
    ):
        sim = flopy.mf6.MFSimulation(
            sim_name=self.name,
            version="mf6",
            sim_ws=os.path.realpath(self.working_directory),
        )
        number_of_periods = 100_000
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
        nrow, ncol = self.basin_mask.shape
        x_coordinates_vertices, y_coordinates_vertices = self.create_vertices(
            nrow, ncol, gt
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
        nrow, ncol = self.basin_mask.shape
        xy_to_cell = np.full((nrow, ncol), -1, dtype=int)
        cell_areas = np.full((nrow, ncol), np.nan, dtype=np.float32)
        for row in range(nrow):
            for column in range(ncol):
                cell_number = row * ncol + column
                xy_to_cell[row, column] = cell_number
                # not here that the vertices are 1 larger than the number of cells
                # therefore an additional offset of 1 is required for each row
                # thus adding 'row' to v1 to account for the offset
                v1 = row * ncol + column + row  # top-left vertex
                v2 = v1 + 1  # top-right vertex
                v3 = v1 + ncol + 2  # bottom-right vertex
                v4 = v1 + ncol + 1  # bottom-left vertex

                # import matplotlib.pyplot as plt

                # v1_point = vertices[v1][1:]
                # v2_point = vertices[v2][1:]
                # v3_point = vertices[v3][1:]
                # v4_point = vertices[v4][1:]
                # # plt.plot(*zip(*(v1_point, v2_point)), c="r")
                # plt.plot(
                #     *zip(*(v1_point, v2_point, v3_point, v4_point, v1_point)), c="r"
                # )
                # plt.savefig("plot.png")

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

        cell_areas = cell_areas[~self.basin_mask]

        # plt.savefig("cells.png")
        active_cells = xy_to_cell[~self.basin_mask].ravel()

        domain = np.stack([~self.basin_mask] * hydraulic_conductivity.shape[0])

        # Discretization for flexible grid
        flopy.mf6.ModflowGwfdisv(
            groundwater_flow,
            nlay=self.nlay,
            ncpl=nrow * ncol,
            nvert=len(vertices),
            vertices=vertices,
            cell2d=cell2d,
            top=self.model.hydrology.grid.decompress(
                self.layer_boundary_elevation[0]
            ).tolist(),
            botm=self.model.hydrology.grid.decompress(
                self.layer_boundary_elevation[1:]
            ).tolist(),
            idomain=domain.tolist(),
        )

        # Node property flow
        k = self.model.hydrology.grid.decompress(hydraulic_conductivity)

        # Initial conditions
        flopy.mf6.ModflowGwfic(groundwater_flow, strt=np.full_like(k, np.nan))

        # Create icelltype array (assuming convertible cells i.e., that can be converted between confined and unconfined)
        icelltype = np.ones_like(domain, dtype=np.int32)
        flopy.mf6.ModflowGwfnpf(
            groundwater_flow,
            save_flows=self.save_flows,
            icelltype=icelltype,
            k=k,
        )

        sy = self.model.hydrology.grid.decompress(specific_yield)
        ss = self.model.hydrology.grid.decompress(specific_storage)

        # Storage
        flopy.mf6.ModflowGwfsto(
            groundwater_flow,
            save_flows=self.save_flows,
            iconvert=1,
            ss=ss,
            sy=sy,
            steady_state=False,
            transient=True,
        )

        # Recharge
        recharge = []
        for cell in active_cells:
            recharge.append(
                (0, cell, 0.0)
            )  # specifying the layer, cell number, and recharge rate

        recharge = flopy.mf6.ModflowGwfrch(
            groundwater_flow,
            fixed_cell=True,
            save_flows=self.save_flows,
            maxbound=len(recharge),
            stress_period_data=recharge,
        )

        # Wells
        wells = []
        for layer in range(self.nlay):
            for cell in active_cells:
                wells.append(
                    (layer, cell, 0.0)
                )  # specifying the layer, cell number, and well rate

        wells = flopy.mf6.ModflowGwfwel(
            groundwater_flow,
            maxbound=len(wells),
            stress_period_data=wells,
            save_flows=self.save_flows,
        )

        # Drainage
        # Drainage rate is set as conductivity * area / drainage length
        # For conductivity we set the conductivity of the top layer
        # area the total size of the cell, and as we are are approximating
        # transmissivity, we can set the drainage length to 1
        drainage = []
        for idx, cell in enumerate(active_cells):
            drainage.append(
                (
                    0,  # top layer
                    cell,
                    self.layer_boundary_elevation[0, idx],  # at elevation of top layer
                    self.hydraulic_conductivity_drainage[idx]
                    * cell_areas[idx]
                    / 1,  # drainage rate
                )
            )

        flopy.mf6.ModflowGwfdrn(
            groundwater_flow,
            maxbound=len(drainage),
            stress_period_data=drainage,
            print_flows=self.save_flows,
            save_flows=self.save_flows,
        )

        sim.simulation_data.set_sci_note_upper_thres(
            1e99
        )  # effectively disable scientific notation
        sim.simulation_data.set_sci_note_lower_thres(
            1e-99
        )  # effectively disable scientific notation

        return sim

    def write_hash_to_disk(self):
        with open(self.hash_file, "wb") as f:
            f.write(self.hash)

    def load_from_disk(self, arguments):
        hashable_dict = {}
        for key, value in arguments.items():
            if isinstance(value, np.ndarray):
                value = str(value.tobytes())
            hashable_dict[key] = value

        self.hash = hashlib.md5(
            json.dumps(hashable_dict, sort_keys=True).encode()
        ).digest()
        if not os.path.exists(self.hash_file):
            prev_hash = None
        else:
            with open(self.hash_file, "rb") as f:
                prev_hash = f.read()
        if prev_hash == self.hash and not self.never_load_from_disk:
            return True
        else:
            return False

    def bmi_return(self) -> list[str]:
        """Parse libmf6.so and libmf6.dll stdout file."""
        with open("mfsim.stdout") as f:
            return f.readlines()

    def load_bmi(self, heads: npt.NDArray[np.float64]):
        """Load the Basic Model Interface."""
        # Current model version 6.5.0 from https://github.com/MODFLOW-USGS/modflow6/releases/tag/6.5.0
        if platform.system() == "Windows":
            libary_name: str = "libmf6.dll"
        elif platform.system() == "Linux":
            libary_name: str = "libmf6.so"
        elif platform.system() == "Darwin":
            libary_name: str = "libmf6.dylib"
        else:
            raise ValueError(f"Platform {platform.system()} not supported.")

        with cd(self.working_directory):
            # XmiWrapper requires the real path (no symlinks etc.)
            # include the version in the folder name to allow updating the version
            # so that the user will automatically get the new version
            library_folder: Path = (
                self.model.bin_folder / "modflow" / MODFLOW_VERSION
            ).resolve()
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
                print("Failed to load " + str(library_path))
                print("with message: " + str(e))
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
                print("MODFLOW model initialized")

        self.end_time: float = self.mf6.get_end_time()
        area_tag: str = self.mf6.get_var_address("AREA", self.name, "DIS")
        area: npt.NDArray[np.float64] = self.mf6.get_value_ptr(area_tag).reshape(
            self.nlay, self.n_active_cells
        )

        # ensure that the areas of all vertical cells are equal
        assert (np.diff(area, axis=0) == 0).all()

        # so we can use the area of the top layer
        self.area: npt.NDArray[np.float32] = area[0].astype(np.float32)

        self.prepare_time_step()

        # because modflow rounds heads when they are written to file, we set the modflow heads
        # to the actual model heads to ensure that the model is in the same state as the modflow model
        self.heads: npt.NDArray[np.float64] = heads
        assert not np.isnan(self.heads).any()

    @property
    def head_tag(self):
        return self.mf6.get_var_address("X", self.name)

    @property
    def heads(self):
        heads = self.mf6.get_value_ptr(self.head_tag).reshape(
            self.nlay, self.n_active_cells
        )
        assert not np.isnan(heads).any()
        return heads

    @heads.setter
    def heads(self, value):
        self.mf6.get_value_ptr(self.head_tag)[:] = value.ravel()

    @property
    def groundwater_depth(self):
        groundwater_depth = get_water_table_depth(
            self.layer_boundary_elevation,
            self.heads,
            self.topography,
            min_remaining_layer_storage_m=self.min_remaining_layer_storage_m,
        )
        assert (groundwater_depth >= 0).all()
        return groundwater_depth

    @property
    def groundwater_content_m(self):
        groundwater_content_m = get_groundwater_storage_m(
            self.layer_boundary_elevation, self.heads, self.specific_yield
        )
        assert (groundwater_content_m >= 0).all()
        return groundwater_content_m

    @property
    def groundwater_content_m3(self):
        return self.groundwater_content_m * self.area

    @property
    def available_groundwater_m(self):
        groundwater_available_m = get_groundwater_storage_m(
            self.layer_boundary_elevation,
            self.heads,
            self.specific_yield,
            min_remaining_layer_storage_m=self.min_remaining_layer_storage_m,
        )
        assert (groundwater_available_m >= 0).all()
        return groundwater_available_m

    @property
    def available_groundwater_m3(self):
        return self.available_groundwater_m * self.area

    @property
    def potential_well_rate_tag(self):
        return self.mf6.get_var_address("Q", self.name, "WEL_0")

    @property
    def actual_well_rate_tag(self):
        return self.mf6.get_var_address("SIMVALS", self.name, "WEL_0")

    @property
    def potential_well_rate(self):
        return self.mf6.get_value_ptr(self.potential_well_rate_tag)

    @property
    def actual_well_rate(self):
        return self.mf6.get_value_ptr(self.actual_well_rate_tag)

    @potential_well_rate.setter
    def potential_well_rate(self, well_rate):
        well_rate_per_layer = distribute_well_rate_per_layer(
            well_rate,
            self.layer_boundary_elevation,
            self.heads,
            self.specific_yield,
            self.area,
            min_remaining_layer_storage_m=self.min_remaining_layer_storage_m,
        ).ravel()
        self.mf6.get_value_ptr(self.potential_well_rate_tag)[:] = well_rate_per_layer

    @property
    def drainage_tag(self):
        return self.mf6.get_var_address("SIMVALS", self.name, "DRN_0")

    @property
    def drainage_m3(self):
        drainage = -self.mf6.get_value_ptr(self.drainage_tag)
        assert not np.isnan(drainage).any()
        # TODO: This assert can become more strict when soil depth is considered
        assert (drainage / self.area < self.hydraulic_conductivity_drainage * 100).all()
        return drainage

    @property
    def _drainage_m(self):
        return self.drainage_m3 / self.area

    @property
    def recharge_tag(self):
        return self.mf6.get_var_address("RECHARGE", self.name, "RCH_0")

    @property
    def _recharge_m(self):
        recharge = self.mf6.get_value_ptr(self.recharge_tag).copy()
        assert not np.isnan(recharge).any()
        return recharge

    @_recharge_m.setter
    def recharge_m(self, value):
        assert not np.isnan(value).any()
        self.mf6.get_value_ptr(self.recharge_tag)[:] = value

    @property
    def recharge_m3(self):
        return self._recharge_m * self.area

    @property
    def max_iter(self):
        mxit_tag = self.mf6.get_var_address("MXITER", "SLN_1")
        return self.mf6.get_value_ptr(mxit_tag)[0]

    def prepare_time_step(self):
        dt = self.mf6.get_time_step()
        self.mf6.prepare_time_step(dt)

    # def set_recharge_m(self, recharge):
    #     """Set recharge, value in m/day."""
    #     self.recharge_m = recharge

    def set_recharge_m3(self, recharge):
        self.recharge_m = recharge / self.area

    def set_groundwater_abstraction_m3(self, groundwater_abstraction):
        """Set well rate, value in m3/day."""
        assert not np.isnan(groundwater_abstraction).any()

        assert (self.available_groundwater_m3 >= groundwater_abstraction).all(), (
            "Requested groundwater abstraction exceeds available groundwater storage. "
        )

        well_rate = -groundwater_abstraction
        assert (well_rate <= 0).all()
        self.potential_well_rate = well_rate

    def step(self):
        if self.mf6.get_current_time() == self.end_time:
            raise StopIteration("MODFLOW used all iteration steps.")

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
                print("MODFLOW did not converge")
                # raise RuntimeError("MODFLOW did not converge")

            self.mf6.finalize_solve(solution_id)

        self.mf6.finalize_time_step()

        assert not np.isnan(self.heads).any()
        assert np.array_equal(self.actual_well_rate, self.potential_well_rate)
        assert not np.isnan(self.heads).any()
        assert not np.isnan(self.recharge_m).any()
        assert not np.isnan(self.potential_well_rate).any()
        assert not np.isnan(self.groundwater_content_m).any()
        assert not np.isnan(self.heads[-1] - self.layer_boundary_elevation[-1]).any()

        if self.verbose:
            print("MODFLOW")
            print(
                f"\ttimestep {int(self.mf6.get_current_time())} converged in {round(time() - t0, 2)} seconds"
            )
            print(
                "\tHead statictics: mean",
                self.heads.mean(),
                "min",
                self.heads.min(),
                "max",
                self.heads.max(),
            )
            print(
                "\tGroundwater depth: mean",
                self.groundwater_depth.mean(),
                "min",
                self.groundwater_depth.min(),
                "max",
                self.groundwater_depth.max(),
            )
            print("\tGroundwater content: mean", self.groundwater_content_m3.mean())
            print(
                "\tRecharge (mean)",
                (self.recharge_m * self.area).mean(),
                "m3",
                self.recharge_m.mean(),
                "m",
            )
            print(
                "\tAbstraction (mean)",
                self.actual_well_rate.mean(),
                "m3",
                (self.actual_well_rate.sum(axis=0) / self.area).mean(),
                "m",
            )
            print(
                "\tDrainage (mean)",
                self.drainage_m3.mean(),
                "m3",
                self._drainage_m.mean(),
                "m",
            )

        self.heads_update_callback(self.heads)

        # If next step exists, prepare timestep. Otherwise the data set through the bmi
        # will be overwritten when preparing the next timestep.
        if self.mf6.get_current_time() < self.end_time:
            self.prepare_time_step()

    def finalize(self):
        self.mf6.finalize()

    def restore(self, heads):
        self.heads = heads
