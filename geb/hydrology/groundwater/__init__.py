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

"""
Groundwater submodule for hydrology in GEB.

Provides groundwater simulation and ModFlow integration utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from geb.module import Module
from geb.types import ArrayFloat32, TwoDArrayFloat64
from geb.workflows import balance_check

from ..routing import get_channel_ratio
from .model import ModFlowSimulation

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


class GroundWater(Module):
    """Implements groundwater hydrology submodel, responsible for flow, abstraction, outflow, and percolation.

    This model communicates with the ModFlow simulation to manage groundwater flow and storage.
    """

    def __init__(self, model: GEBModel, hydrology: Hydrology) -> None:
        """Initialize the groundwater model.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology submodel instance.

        """
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        """Name of the module."""
        return "hydrology.groundwater"

    def spinup(self) -> None:
        """Initialize groundwater model parameters and state variables."""
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

        def get_initial_head() -> npt.NDArray[np.float64]:
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

    def heads_update_callback(self, heads: TwoDArrayFloat64) -> None:
        """Callback function to update groundwater heads after ModFlow simulation step.

        This is done to ensure the value is always in sync with ModFlow.

        Args:
            heads: Updated groundwater heads from ModFlow (m).
        """
        self.hydrology.grid.var.heads = heads

    def initalize_modflow_model(self) -> None:
        """Initialize the ModFlow groundwater simulation model."""
        self.modflow = ModFlowSimulation(
            working_directory=self.model.simulation_root_spinup / "modflow_model",
            modflow_bin_folder=self.model.bin_folder / "modflow",
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

    def step(
        self,
        groundwater_recharge_m: ArrayFloat32,
        groundwater_abstraction_m3: ArrayFloat32,
    ) -> ArrayFloat32:
        """Perform a groundwater model step.

        Args:
            groundwater_recharge_m: Recharge to the groundwater (m/step).
            groundwater_abstraction_m3: Groundwater abstraction (m3/step).

        Returns:
            Baseflow to rivers (m/step).
        """
        assert (groundwater_abstraction_m3 + 1e-7 >= 0).all()
        groundwater_abstraction_m3[groundwater_abstraction_m3 < 0] = 0
        assert (groundwater_recharge_m >= 0).all()

        if __debug__:
            groundwater_storage_pre = self.modflow.groundwater_content_m3

        self.modflow.set_recharge_m3(groundwater_recharge_m * self.grid.var.cell_area)
        self.modflow.set_groundwater_abstraction_m3(
            groundwater_abstraction_m3.astype(np.float64)
        )
        self.modflow.step()

        if __debug__:
            balance_check(
                name="groundwater",
                how="sum",
                influxes=[
                    groundwater_recharge_m.astype(np.float64) * self.grid.var.cell_area
                ],
                outfluxes=[
                    groundwater_abstraction_m3.astype(np.float64),
                    self.modflow.drainage_m3.astype(np.float64),
                ],
                prestorages=[groundwater_storage_pre.astype(np.float64)],
                poststorages=[self.modflow.groundwater_content_m3.astype(np.float64)],
                tolerance=500,  # 500 m3
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
        channel_ratio.fill(1)

        # this is the capillary rise for the NEXT timestep
        self.grid.var.capillar = groundwater_drainage * (1 - channel_ratio)
        baseflow = (groundwater_drainage * channel_ratio).astype(np.float32)

        self.report(locals())

        return baseflow

    @property
    def groundwater_content_m3(self) -> npt.NDArray[np.float32]:
        """Groundwater content in cubic meters.

        Returns:
            Groundwater content in cubic meters in active grid cells.
        """
        return self.modflow.groundwater_content_m3.astype(np.float32)

    @property
    def groundwater_depth(self) -> npt.NDArray[np.float32]:
        """Groundwater depth in meters.

        Returns:
            Groundwater depth in active grid cells.
        """
        return self.modflow.groundwater_depth.astype(np.float32)

    def decompress(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Decompress data from compressed grid format to full grid format.

        Args:
            data: Data in compressed grid format.

        Returns:
            Data in full grid format.
        """
        return self.hydrology.grid.decompress(data)
