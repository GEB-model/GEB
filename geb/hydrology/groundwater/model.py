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

from time import time
from contextlib import contextmanager
import os
from pathlib import Path
import numpy as np
from numba import njit
from xmipy import XmiWrapper
import flopy
import json
import hashlib
import platform
from pyproj import CRS, Transformer

MODFLOW_VERSION = "6.5.0"


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

        assert (
            remaining_well_rate > -1e-10
        ), (
            "Well rate could not be distributed, layers are too dry"
        )  # leaving some tollerance for numerical errors

    assert np.allclose(well_rate_per_layer.sum(axis=0), well_rate)
    return well_rate_per_layer


class ModFlowSimulation:
    def __init__(
        self,
        model,
        topography,
        gt,
        ndays,
        specific_storage,
        specific_yield,
        layer_boundary_elevation,
        basin_mask,
        heads,
        hydraulic_conductivity,
        min_remaining_layer_storage_m=0.1,
        verbose=False,
        never_load_from_disk=False,
    ):
        self.name = "MODEL"  # MODFLOW requires the name to be uppercase
        self.model = model
        self.basin_mask = basin_mask
        self.nlay = hydraulic_conductivity.shape[0]
        assert self.basin_mask.dtype == bool
        self.n_active_cells = self.basin_mask.size - self.basin_mask.sum()
        self.working_directory = model.simulation_root / "modflow_model"
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
        self.hash_file = os.path.join(self.working_directory, "input_hash")

        save_flows = False

        if not self.load_from_disk(arguments):
            try:
                if self.verbose:
                    print("Creating MODFLOW model")

                sim = self.flexible_grid(
                    ndays,
                    gt,
                    save_flows,
                    heads,
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

        self.load_bmi()

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

    def flexible_grid(
        self,
        ndays,
        gt,
        save_flows,
        heads,
        hydraulic_conductivity,
        specific_storage,
        specific_yield,
    ):
        sim = flopy.mf6.MFSimulation(
            sim_name=self.name,
            version="mf6",
            sim_ws=os.path.realpath(self.working_directory),
        )
        flopy.mf6.ModflowTdis(sim, nper=ndays, perioddata=[(1.0, 1, 1)] * ndays)

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
            print_input=save_flows,
            print_flows=save_flows,
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
            top=self.model.data.grid.decompress(
                self.layer_boundary_elevation[0]
            ).tolist(),
            botm=self.model.data.grid.decompress(
                self.layer_boundary_elevation[1:]
            ).tolist(),
            idomain=domain.tolist(),
        )

        # Initial conditions
        heads = self.model.data.grid.decompress(heads)
        flopy.mf6.ModflowGwfic(groundwater_flow, strt=heads)

        # Node property flow
        k = self.model.data.grid.decompress(hydraulic_conductivity)
        # Create icelltype array (assuming convertible cells i.e., that can be converted between confined and unconfined)
        icelltype = np.ones_like(domain, dtype=np.int32)
        flopy.mf6.ModflowGwfnpf(
            groundwater_flow,
            save_flows=save_flows,
            icelltype=icelltype,
            k=k,
        )

        sy = self.model.data.grid.decompress(specific_yield)
        ss = self.model.data.grid.decompress(specific_storage)

        # Storage
        flopy.mf6.ModflowGwfsto(
            groundwater_flow,
            save_flows=save_flows,
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
            save_flows=save_flows,
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
            save_flows=save_flows,
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
            print_flows=save_flows,
            save_flows=save_flows,
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
                prev_hash = f.read().strip()
        if prev_hash == self.hash and not self.never_load_from_disk:
            return True
        else:
            return False

    def bmi_return(self, success):
        """
        parse libmf6.so and libmf6.dll stdout file
        """
        fpth = os.path.join("mfsim.stdout")
        with open(fpth) as f:
            lines = f.readlines()
        return success, lines

    def load_bmi(self):
        """Load the Basic Model Interface"""
        success = False

        # Current model version 6.5.0 from https://github.com/MODFLOW-USGS/modflow6/releases/tag/6.5.0
        if platform.system() == "Windows":
            libary_name = "libmf6.dll"
        elif platform.system() == "Linux":
            libary_name = "libmf6.so"
        elif platform.system() == "Darwin":
            libary_name = "libmf6.dylib"
        else:
            raise ValueError(f"Platform {platform.system()} not supported.")

        with cd(self.working_directory):
            # XmiWrapper requires the real path (no symlinks etc.)
            # include the version in the folder name to allow updating the version
            # so that the user will automatically get the new version
            library_folder = (Path(__file__).parent / "bin" / MODFLOW_VERSION).resolve()
            library_path = library_folder / libary_name

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
                print("Failed to load " + library_path)
                print("with message: " + str(e))
                self.bmi_return(success)
                raise

            # modflow requires the real path (no symlinks etc.)
            config_file = os.path.realpath("mfsim.nam")
            if not os.path.exists(config_file):
                raise FileNotFoundError(
                    f"Config file {config_file} not found on disk. Did you create the model first (load_from_disk = False)?"
                )

            # initialize the model
            try:
                self.mf6.initialize(config_file)
            except:
                self.bmi_return(success)
                raise

            if self.verbose:
                print("MODFLOW model initialized")

        self.end_time = self.mf6.get_end_time()
        area_tag = self.mf6.get_var_address("AREA", self.name, "DIS")
        area = self.mf6.get_value_ptr(area_tag).reshape(self.nlay, self.n_active_cells)

        # ensure that the areas of all vertical cells are equal
        assert (np.diff(area, axis=0) == 0).all()

        # so we can use the area of the top layer
        self.area = area[0].copy()
        assert not np.isnan(self.heads).any()

        self.prepare_time_step()

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
        self.mf6.get_value_ptr(self.head_tag)[:] = value

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
    def drainage_m(self):
        return self.drainage_m3 / self.area

    @property
    def recharge_tag(self):
        return self.mf6.get_var_address("RECHARGE", self.name, "RCH_0")

    @property
    def recharge_m(self):
        recharge = self.mf6.get_value_ptr(self.recharge_tag).copy()
        assert not np.isnan(recharge).any()
        return recharge

    @property
    def recharge_m3(self):
        return self.recharge_m * self.area

    @recharge_m.setter
    def recharge_m(self, value):
        assert not np.isnan(value).any()
        self.mf6.get_value_ptr(self.recharge_tag)[:] = value

    @property
    def max_iter(self):
        mxit_tag = self.mf6.get_var_address("MXITER", "SLN_1")
        return self.mf6.get_value_ptr(mxit_tag)[0]

    def prepare_time_step(self):
        dt = self.mf6.get_time_step()
        self.mf6.prepare_time_step(dt)

    def set_recharge_m(self, recharge):
        """Set recharge, value in m/day"""
        self.recharge_m = recharge

    def set_groundwater_abstraction_m3(self, groundwater_abstraction):
        """Set well rate, value in m3/day"""
        assert not np.isnan(groundwater_abstraction).any()

        assert (
            self.available_groundwater_m3 >= groundwater_abstraction
        ).all(), (
            "Requested groundwater abstraction exceeds available groundwater storage. "
        )

        well_rate = -groundwater_abstraction
        assert (well_rate <= 0).all()
        self.potential_well_rate = well_rate

    def step(self):
        if self.mf6.get_current_time() > self.end_time:
            raise StopIteration(
                "MODFLOW used all iteration steps. Consider increasing `ndays`"
            )

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
            print(
                f"MODFLOW timestep {int(self.mf6.get_current_time())} converged in {round(time() - t0, 2)} seconds"
            )

        # If next step exists, prepare timestep. Otherwise the data set through the bmi
        # will be overwritten when preparing the next timestep.
        if self.mf6.get_current_time() < self.end_time:
            self.prepare_time_step()

    def finalize(self):
        self.mf6.finalize()
