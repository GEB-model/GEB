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

import numpy as np
import numpy.typing as npt

from geb.module import Module
from geb.workflows import balance_check

from ..routing import get_channel_ratio
from .model import ModFlowSimulation


class GroundWater(Module):
    """Implements groundwater hydrology submodel, responsible for flow, abstraction, outflow, and percolation.

    This model communicates with the ModFlow simulation to manage groundwater flow and storage.

    Args:
        model: The GEB model instance.
        hydrology: The hydrology submodel instance.
    """

    def __init__(self, model, hydrology):
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self):
        return "hydrology.groundwater"

    def spinup(self):
        # load hydraulic conductivity (md-1)
        self.grid.var.hydraulic_conductivity = self.hydrology.grid.load(
            self.model.files["grid"]["groundwater/hydraulic_conductivity"],
            layer=None,
        )

        self.grid.var.specific_yield = self.hydrology.grid.load(
            self.model.files["grid"]["groundwater/specific_yield"],
            layer=None,
        )

        self.grid.var.layer_boundary_elevation = self.hydrology.grid.load(
            self.model.files["grid"]["groundwater/layer_boundary_elevation"],
            layer=None,
        )

        # recession_coefficient = self.hydrology.grid.load(
        #     self.model.files["grid"]["groundwater/recession_coefficient"],
        # )

        self.grid.var.elevation = self.hydrology.grid.load(
            self.model.files["grid"]["landsurface/elevation"]
        )

        assert (
            self.grid.var.hydraulic_conductivity.shape
            == self.grid.var.specific_yield.shape
        )

        self.grid.var.leakageriver_factor = 0.001  # in m/day
        self.grid.var.leakagelake_factor = 0.001  # in m/day

        self.initial_water_table_depth = 2

        def get_initial_head():
            heads = self.hydrology.grid.load(
                self.model.files["grid"]["groundwater/heads"], layer=None
            ).astype(np.float64)  # modflow is an exception, it needs double precision
            heads = np.where(
                ~np.isnan(heads),
                heads,
                self.grid.var.layer_boundary_elevation[1:] + 0.1,
            )
            heads = np.where(
                heads > self.grid.var.layer_boundary_elevation[1:],
                heads,
                self.grid.var.layer_boundary_elevation[1:] + 0.1,
            )
            return heads

        self.grid.var.heads = get_initial_head()

        self.grid.var.capillar = self.grid.full_compressed(0, dtype=np.float32)

    def heads_update_callback(self, heads):
        self.hydrology.grid.var.heads = heads

    def initalize_modflow_model(self):
        self.modflow = ModFlowSimulation(
            self.model,
            topography=self.grid.var.elevation,
            gt=self.model.hydrology.grid.gt,
            specific_storage=np.zeros_like(self.grid.var.specific_yield),
            specific_yield=self.grid.var.specific_yield,
            layer_boundary_elevation=self.grid.var.layer_boundary_elevation,
            basin_mask=self.model.hydrology.grid.mask,
            heads=self.grid.var.heads,
            hydraulic_conductivity=self.grid.var.hydraulic_conductivity,
            verbose=False,
            heads_update_callback=self.heads_update_callback,
        )

    def step(self, groundwater_recharge, groundwater_abstraction_m3):
        assert (groundwater_abstraction_m3 + 1e-7 >= 0).all()
        groundwater_abstraction_m3[groundwater_abstraction_m3 < 0] = 0
        assert (groundwater_recharge >= 0).all()

        if __debug__:
            groundwater_storage_pre = self.modflow.groundwater_content_m3

        self.modflow.set_recharge_m3(groundwater_recharge * self.grid.var.cell_area)
        self.modflow.set_groundwater_abstraction_m3(groundwater_abstraction_m3)
        self.modflow.step()

        if __debug__:
            balance_check(
                name="groundwater",
                how="sum",
                influxes=[
                    groundwater_recharge.astype(np.float64) * self.grid.var.cell_area
                ],
                outfluxes=[
                    groundwater_abstraction_m3.astype(np.float64),
                    self.modflow.drainage_m3.astype(np.float64),
                ],
                prestorages=[groundwater_storage_pre.astype(np.float64)],
                poststorages=[self.modflow.groundwater_content_m3.astype(np.float64)],
                tollerance=500,  # 500 m3
            )

        groundwater_drainage = self.modflow.drainage_m3 / self.grid.var.cell_area

        channel_ratio: npt.NDArray[np.float32] = get_channel_ratio(
            river_length=self.grid.var.river_length,
            river_width=np.where(
                ~np.isnan(self.grid.var.average_river_width),
                self.grid.var.average_river_width,
                0,
            ),
            cell_area=self.grid.var.cell_area,
        )

        # this is the capillary rise for the NEXT timestep
        self.grid.var.capillar = groundwater_drainage * (1 - channel_ratio)
        baseflow = groundwater_drainage * channel_ratio

        # capriseindex is 1 where capilary rise occurs
        self.hydrology.HRU.capriseindex = self.hydrology.to_HRU(
            data=np.float32(groundwater_drainage > 0)
        )

        self.report(self, locals())

        return baseflow

    @property
    def groundwater_content_m3(self):
        return self.modflow.groundwater_content_m3.astype(np.float32)

    @property
    def groundwater_depth(self):
        return self.modflow.groundwater_depth.astype(np.float32)

    def decompress(self, data):
        return self.hydrology.grid.decompress(data)
