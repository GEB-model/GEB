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
from .model import ModFlowSimulation
from geb.workflows import balance_check


class GroundWater:
    def __init__(self, model):
        self.var = model.data.grid
        self.model = model

        # load hydraulic conductivity (md-1)
        hydraulic_conductivity = self.model.data.grid.load(
            self.model.files["grid"]["groundwater/hydraulic_conductivity"],
            layer=None,
        )

        specific_yield = self.model.data.grid.load(
            self.model.files["grid"]["groundwater/specific_yield"],
            layer=None,
        )

        layer_boundary_elevation = self.model.data.grid.load(
            self.model.files["grid"]["groundwater/layer_boundary_elevation"],
            layer=None,
        )

        # recession_coefficient = self.model.data.grid.load(
        #     self.model.files["grid"]["groundwater/recession_coefficient"],
        # )

        elevation = self.model.data.grid.load(
            self.model.files["grid"]["landsurface/topo/elevation"]
        )

        assert hydraulic_conductivity.shape == specific_yield.shape

        self.var.channel_ratio = self.var.load(
            self.model.files["grid"]["routing/kinematic/channel_ratio"]
        )

        self.var.leakageriver_factor = 0.001  # in m/day
        self.var.leakagelake_factor = 0.001  # in m/day

        self.initial_water_table_depth = 2

        def get_initial_head():
            heads = self.model.data.grid.load(
                self.model.files["grid"]["groundwater/heads"], layer=None
            )
            heads = np.where(
                ~np.isnan(heads),
                heads,
                layer_boundary_elevation[1:] + 0.1,
            )
            heads = np.where(
                heads > layer_boundary_elevation[1:],
                heads,
                layer_boundary_elevation[1:] + 0.1,
            )
            return heads

        self.var.heads = self.model.data.grid.load_initial(
            "heads",
            default=get_initial_head(),
        )

        self.modflow = ModFlowSimulation(
            self.model,
            topography=elevation,
            gt=self.model.data.grid.gt,
            ndays=self.model.n_timesteps,
            specific_storage=np.zeros_like(specific_yield),
            specific_yield=specific_yield,
            layer_boundary_elevation=layer_boundary_elevation,
            basin_mask=self.model.data.grid.mask,
            heads=self.var.heads,
            hydraulic_conductivity=hydraulic_conductivity,
            verbose=False,
        )

        self.var.capillar = self.var.load_initial(
            "capillar", default=self.var.full_compressed(0, dtype=np.float32)
        )

    def step(self, groundwater_recharge, groundwater_abstraction_m3):
        assert (groundwater_abstraction_m3 + 1e-7 >= 0).all()
        groundwater_abstraction_m3[groundwater_abstraction_m3 < 0] = 0
        assert (groundwater_recharge >= 0).all()

        groundwater_storage_pre = self.modflow.groundwater_content_m3

        self.modflow.set_recharge_m(groundwater_recharge)
        self.modflow.set_groundwater_abstraction_m3(groundwater_abstraction_m3)
        self.modflow.step()

        drainage_m3 = self.modflow.drainage_m3
        recharge_m3 = groundwater_recharge * self.modflow.area
        groundwater_storage_post = self.modflow.groundwater_content_m3

        balance_check(
            name="groundwater",
            how="sum",
            influxes=[recharge_m3],
            outfluxes=[
                groundwater_abstraction_m3,
                drainage_m3,
            ],
            prestorages=[groundwater_storage_pre],
            poststorages=[groundwater_storage_post],
            tollerance=100,  # 100 m3
        )

        groundwater_drainage = self.modflow.drainage_m3 / self.var.cellArea

        self.var.capillar = groundwater_drainage * (1 - self.var.channel_ratio)
        self.var.baseflow = groundwater_drainage * self.var.channel_ratio
        self.var.heads = self.modflow.heads

        # capriseindex is 1 where capilary rise occurs
        self.model.data.HRU.capriseindex = self.model.data.to_HRU(
            data=np.float32(groundwater_drainage > 0)
        )

    @property
    def groundwater_content_m3(self):
        return self.modflow.groundwater_content_m3

    @property
    def groundwater_depth(self):
        return self.modflow.groundwater_depth

    def decompress(self, data):
        return self.model.data.grid.decompress(data)
