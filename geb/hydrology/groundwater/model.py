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
import numpy as np
from xmipy import XmiWrapper
import flopy
import json
import hashlib
import platform
from pyproj import CRS, Transformer


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


class ModFlowSimulation:
    def __init__(
        self,
        model,
        name,
        topography,
        gt,
        ndays,
        specific_storage,
        specific_yield,
        bottom_soil,
        bottom,
        basin_mask,
        head,
        hydraulic_conductivity,
        complexity="COMPLEX",
        verbose=False,
    ):
        self.name = name.upper()  # MODFLOW requires the name to be uppercase
        self.basin_mask = basin_mask
        assert self.basin_mask.dtype == bool
        self.n_active_cells = self.basin_mask.size - self.basin_mask.sum()
        self.working_directory = model.simulation_root / "modflow_model"
        os.makedirs(self.working_directory, exist_ok=True)
        self.verbose = verbose

        self.topography = topography
        self.bottom = bottom
        self.bottom_soil = bottom_soil
        self.specific_yield = specific_yield
        assert self.specific_yield.shape == self.bottom.shape

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
                    complexity,
                    save_flows,
                    head,
                    hydraulic_conductivity,
                    specific_storage,
                    specific_yield,
                    bottom_soil,
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
        complexity,
        save_flows,
        head,
        hydraulic_conductivity,
        specific_storage,
        specific_yield,
        bottom_soil,
    ):
        sim = flopy.mf6.MFSimulation(
            sim_name=self.name,
            version="mf6",
            exe_name=os.path.join("modflow", "mf6"),
            sim_ws=os.path.realpath(self.working_directory),
        )
        time_discretization = flopy.mf6.ModflowTdis(
            sim, nper=ndays, perioddata=[(1.0, 1, 1)] * ndays
        )

        # create iterative model solution
        iterative_model_solution = flopy.mf6.ModflowIms(
            sim,
            print_option=None,
            complexity=complexity,
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

        # plt.savefig("cells.png")
        active_cells = xy_to_cell[~self.basin_mask].ravel()

        # Create icelltype array (assuming convertible cells i.e., that can be converted between confined and unconfined)
        icelltype = np.ones(nrow * ncol, dtype=int)

        bottom = np.zeros_like(self.basin_mask, dtype=float)
        bottom[~self.basin_mask] = self.bottom[-1]

        top = np.zeros_like(self.basin_mask, dtype=float)
        top[~self.basin_mask] = self.topography

        # Discretization for flexible grid
        discretization = flopy.mf6.ModflowGwfdisv(
            groundwater_flow,
            nlay=hydraulic_conductivity.shape[0],
            ncpl=nrow * ncol,
            nvert=len(vertices),
            vertices=vertices,
            cell2d=cell2d,
            top=top.tolist(),
            botm=bottom.tolist(),
            idomain=~self.basin_mask,
        )

        head2d = np.zeros_like(self.basin_mask, dtype=float)
        head2d[~self.basin_mask] = head

        # Initial conditions
        initial_conditions = flopy.mf6.ModflowGwfic(groundwater_flow, strt=head2d)

        k = np.zeros(
            (hydraulic_conductivity.shape[0], *self.basin_mask.shape), dtype=float
        )
        k[:, ~self.basin_mask] = hydraulic_conductivity

        # Node property flow
        node_property_flow = flopy.mf6.ModflowGwfnpf(
            groundwater_flow,
            save_flows=save_flows,
            icelltype=icelltype,
            k=k,
        )

        sy = np.zeros((specific_yield.shape[0], *self.basin_mask.shape), dtype=float)
        sy[:, ~self.basin_mask] = specific_yield

        ss = np.zeros((specific_storage.shape[0], *self.basin_mask.shape), dtype=float)
        ss[:, ~self.basin_mask] = specific_storage

        # Storage
        storage = flopy.mf6.ModflowGwfsto(
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
            save_flows=True,
            maxbound=len(recharge),
            stress_period_data=recharge,
        )

        # Wells
        wells = []
        for cell in active_cells:
            wells.append(
                (0, cell, 0.0)
            )  # specifying the layer, cell number, and well rate

        wells = flopy.mf6.ModflowGwfwel(
            groundwater_flow,
            maxbound=len(wells),
            stress_period_data=wells,
        )

        # Drainage
        drainage = []
        for idx, cell in enumerate(active_cells):
            drainage.append((0, cell, bottom_soil[idx], hydraulic_conductivity[0, idx]))

        drainage = flopy.mf6.ModflowGwfdrn(
            groundwater_flow,
            maxbound=len(drainage),
            stress_period_data=drainage,
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
        if prev_hash == self.hash:
            if os.path.exists(os.path.join(self.working_directory, "mfsim.nam")):
                return True
            else:
                return False
        else:
            return False

    def bmi_return(self, success, model_ws):
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

        # Current model version 6.4.2 from https://github.com/MODFLOW-USGS/modflow6/releases/tag/6.4.2
        if platform.system() == "Windows":
            libary_name = "windows/libmf6.dll"
        elif platform.system() == "Linux":
            libary_name = "linux/libmf6.so"
        elif platform.system() == "Darwin":
            libary_name = "mac/libmf6.dylib"
        else:
            raise ValueError(f"Platform {platform.system()} not recognized.")

        with cd(self.working_directory):
            # modflow requires the real path (no symlinks etc.)
            library_path = os.path.realpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), libary_name)
            )
            assert os.path.exists(library_path)
            try:
                self.mf6 = XmiWrapper(library_path)
            except Exception as e:
                print("Failed to load " + library_path)
                print("with message: " + str(e))
                self.bmi_return(success, self.working_directory)
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
                self.bmi_return(success, self.working_directory)
                raise

            if self.verbose:
                print("MODFLOW model initialized")

        self.end_time = self.mf6.get_end_time()
        area_tag = self.mf6.get_var_address("AREA", self.name, "DIS")
        self.area = self.mf6.get_value_ptr(area_tag)

        self.prepare_time_step()

    @property
    def head(self):
        head_tag = self.mf6.get_var_address("X", self.name)
        return self.mf6.get_value_ptr(head_tag)

    @head.setter
    def head(self, value):
        head_tag = self.mf6.get_var_address("X", self.name)
        self.mf6.get_value_ptr(head_tag)[:] = value

    @property
    def groundwater_depth(self):
        return self.topography - self.head

    @property
    def groundwater_content_m(self):
        # use the bottom of the bottom layer
        assert self.specific_yield.shape[0] == 1, "Only 1 layer is supported"
        groundwater_content_m = (self.head - self.bottom[-1]) * self.specific_yield[0]
        # allow for small negative values as a result of numerical errors
        assert (groundwater_content_m > -1e-2).all()
        groundwater_content_m[groundwater_content_m < 0] = 0
        return groundwater_content_m

    @property
    def groundwater_content_m3(self):
        return self.groundwater_content_m * self.area

    @property
    def well_rate(self):
        well_tag = self.mf6.get_var_address("BOUND", self.name, "WEL_0")
        return self.mf6.get_value_ptr(well_tag)[:, 0]

    @well_rate.setter
    def well_rate(self, value):
        well_tag = self.mf6.get_var_address("BOUND", self.name, "WEL_0")
        self.mf6.get_value_ptr(well_tag)[:, 0][:] = value

    @property
    def drainage_m3(self):
        drainage_tag = self.mf6.get_var_address("SIMVALS", self.name, "DRN_0")
        drainage = self.mf6.get_value_ptr(drainage_tag)
        return -drainage

    @property
    def drainage_m(self):
        return self.drainage_m3 / self.area

    @property
    def recharge_m3(self):
        recharge_tag = self.mf6.get_var_address("BOUND", self.name, "RCH_0")
        recharge = self.mf6.get_value_ptr(recharge_tag)[:, 0]
        return recharge

    @property
    def recharge_m(self):
        return self.recharge_m3 / self.area

    @recharge_m3.setter
    def recharge_m3(self, value):
        recharge_tag = self.mf6.get_var_address("BOUND", self.name, "RCH_0")
        self.mf6.get_value_ptr(recharge_tag)[:, 0] = value

    @property
    def max_iter(self):
        mxit_tag = self.mf6.get_var_address("MXITER", "SLN_1")
        return self.mf6.get_value_ptr(mxit_tag)[0]

    def prepare_time_step(self):
        dt = self.mf6.get_time_step()
        self.mf6.prepare_time_step(dt)

    def set_recharge_m(self, recharge):
        """Set recharge, value in m/day"""
        self.recharge_m3 = recharge * self.area

    def set_groundwater_abstraction_m(self, groundwater_abstraction):
        """Set well rate, value in m/day"""
        well_rate = -groundwater_abstraction
        assert (well_rate <= 0).all()
        self.well_rate = well_rate * self.area

    def step(self, plot=False):
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
            # else:
            #     raise ValueError("MODFLOW did not converge")

            self.mf6.finalize_solve(solution_id)

        self.mf6.finalize_time_step()

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
