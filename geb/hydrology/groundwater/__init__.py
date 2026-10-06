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

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from geb.geb_types import (
    ArrayFloat32,
    ArrayFloat64 as ArrayFloat64,
    ThreeDArrayFloat64,
    TwoDArrayBool,
    TwoDArrayFloat64,
)
from geb.module import Module
from geb.workflows import balance_check
from geb.workflows.io import read_zarr

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
        boundary_hydraulic_conductivity: ThreeDArrayFloat64 = read_zarr(
            self.model.files["other"]["groundwater/boundary_hydraulic_conductivity"]
        ).values.astype(np.float64) * np.float32(
            self.model.config["parameters"][
                "groundwater_hydraulic_conductivity_multiplier"
            ]
        )
        boundary_layer_boundary_elevation: ThreeDArrayFloat64 = read_zarr(
            self.model.files["other"]["groundwater/boundary_layer_boundary_elevation"]
        ).values.astype(np.float64)
        boundary_heads: ThreeDArrayFloat64 = read_zarr(
            self.model.files["other"]["groundwater/boundary_heads"]
        ).values.astype(np.float64)

        # load hydraulic conductivity (md-1)
        self.grid.var.groundwater_hydraulic_conductivity_m_per_day = self.grid.compress(
            boundary_hydraulic_conductivity[:, 1:-1, 1:-1].astype(np.float32)
        )

        self.grid.var.specific_yield = self.hydrology.grid.load3d(
            self.model.files["grid"]["groundwater/specific_yield"],
        )

        self.grid.var.layer_boundary_elevation = self.grid.compress(
            boundary_layer_boundary_elevation[:, 1:-1, 1:-1].astype(np.float32)
        )

        self.grid.var.elevation = self.hydrology.grid.load2d(
            self.model.files["grid"]["landsurface/elevation_m"]
        )

        assert (
            self.grid.var.groundwater_hydraulic_conductivity_m_per_day.shape
            == self.grid.var.specific_yield.shape
        )

        def get_initial_head() -> npt.NDArray[np.float64]:
            heads = self.grid.compress(boundary_heads[:, 1:-1, 1:-1])
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
        boundary_heads: ThreeDArrayFloat64 = read_zarr(
            self.model.files["other"]["groundwater/boundary_heads"]
        ).values.astype(np.float64)
        boundary_mask: TwoDArrayBool = read_zarr(
            self.model.files["other"]["groundwater/boundary_mask"]
        ).values.astype(bool)
        boundary_layer_boundary_elevation: ThreeDArrayFloat64 = read_zarr(
            self.model.files["other"]["groundwater/boundary_layer_boundary_elevation"]
        ).values.astype(np.float64)
        boundary_hydraulic_conductivity: ThreeDArrayFloat64 = read_zarr(
            self.model.files["other"]["groundwater/boundary_hydraulic_conductivity"]
        ).values.astype(np.float64)
        self.modflow = ModFlowSimulation(
            working_directory=self.model.simulation_root_spinup / "modflow_model",
            modflow_bin_folder=self.model.bin_folder / "modflow",
            topography=self.grid.var.elevation,
            gt=self.model.hydrology.grid.gt,
            specific_storage=np.zeros_like(self.grid.var.specific_yield),
            specific_yield=self.grid.var.specific_yield,
            layer_boundary_elevation=self.grid.var.layer_boundary_elevation,
            basin_mask=self.model.hydrology.grid.mask,
            hydraulic_conductivity=self.grid.var.groundwater_hydraulic_conductivity_m_per_day,
            heads=self.grid.var.heads,
            heads_update_callback=self.heads_update_callback,
            logger=self.model.logger,
            boundary_heads=boundary_heads,
            boundary_mask=boundary_mask,
            boundary_layer_boundary_elevation=boundary_layer_boundary_elevation,
            boundary_hydraulic_conductivity=boundary_hydraulic_conductivity,
            verbose=False,
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
            influxes: list[npt.NDArray[np.float64] | np.float64] = [
                groundwater_recharge_m.astype(np.float64) * self.grid.var.cell_area,
                self.boundary_inflow_m3,
            ]
            outfluxes: list[npt.NDArray[np.float64] | np.float64] = [
                groundwater_abstraction_m3.astype(np.float64),
                self.modflow.drainage_m3.astype(np.float64),
                self.boundary_outflow_m3,
            ]

            balance_check(
                name="groundwater",
                how="sum",
                influxes=influxes,
                outfluxes=outfluxes,
                prestorages=[groundwater_storage_pre.astype(np.float64)],
                poststorages=[self.modflow.groundwater_content_m3.astype(np.float64)],
                tolerance=groundwater_recharge_m.size,  # maximum of 1m3 per cell
            )

        groundwater_drainage = self.modflow.drainage_m3 / self.grid.var.cell_area

        # we assume that all the baseflow ends up in the river
        channel_ratio = np.float32(1.0)

        # this is the capillary rise for the NEXT timestep
        self.grid.var.capillar = (groundwater_drainage * (1 - channel_ratio)).astype(
            np.float32
        )
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
    def boundary_inflow_m3(self) -> np.float64:
        """Total inflow from groundwater boundary cells (m3/step).

        Returns:
            The total volume of groundwater entering the model domain from boundary cells (m3).
        """
        return np.float64(self.modflow.boundary_inflow_m3.sum())

    @property
    def boundary_outflow_m3(self) -> np.float64:
        """Total outflow to groundwater boundary cells (m3/step).

        Returns:
            The total volume of groundwater leaving the model domain to boundary cells (m3).
        """
        return np.float64(self.modflow.boundary_outflow_m3.sum())

    @property
    def boundary_flow_m3(self) -> np.float64:
        """Net groundwater flow across boundary cells (m3/step).

        Positive values represent net flow into the model domain,
        and negative values represent net flow out of the model domain.

        Returns:
            The net volume of groundwater flow across boundary cells (m3).
        """
        return np.float64(self.modflow.boundary_flow_m3.sum())

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
