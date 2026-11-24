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

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from scipy.ndimage import distance_transform_edt

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64 as ArrayFloat64,
    ThreeDArrayFloat64,
    TwoDArrayBool,
    TwoDArrayFloat32,
    TwoDArrayFloat64,
)
from geb.module import Module
from geb.numba_config import NUMBA_CACHE
from geb.workflows import TimingModule, balance_check
from geb.workflows.io import read_zarr

from .model import ModFlowSimulation

# Wetted perimeter of the streams that drain the aquifer (m), for the radial flow
# resistance near the stream. The resistance depends on it only logarithmically.
STREAM_WETTED_PERIMETER_M: float = 3.0

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology

DEPTH_OF_CONSTANT_CONDUCTIVITY_M: float = 1.5


def get_conductivity_depth_decay_factor(
    layer_boundary_elevation_m: npt.NDArray[np.floating],
    slope_m_per_m: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """Mean reduction of the hydraulic conductivity of each layer by its decay with depth.

    The hydraulic conductivity of the aquifer data is that of the rock near the surface.
    Below the soil it decreases exponentially with depth, faster in steep terrain where
    the weathered layer is thin (Fan et al., 2013, Science 339, 940-943):

        K(z) = K0 * exp(-(z - 1.5) / f),  f = 120 / (1 + 150 * slope) m,

    with z the depth below the surface (constant conductivity in the upper 1.5 m) and
    f = 5 m on slopes steeper than 0.16. The factor is the mean of K(z) / K0 over the
    layer, so that the conductivity times the factor times the layer thickness is the
    transmissivity of the layer.

    Args:
        layer_boundary_elevation_m: Elevation of the top and bottom of every layer,
            from the surface downwards, shape (number of layers + 1, ...) (m).
        slope_m_per_m: Terrain slope, shape (...) (m/m).

    Returns:
        Reduction factor per layer, shape (number of layers, ...) (-).
    """
    e_folding_depth_m = np.where(
        slope_m_per_m <= 0.16, 120.0 / (1.0 + 150.0 * slope_m_per_m), 5.0
    )

    def integral_to_depth(depth_m: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        return np.minimum(depth_m, DEPTH_OF_CONSTANT_CONDUCTIVITY_M) + (
            e_folding_depth_m
            * (
                1.0
                - np.exp(
                    -np.maximum(depth_m - DEPTH_OF_CONSTANT_CONDUCTIVITY_M, 0.0)
                    / e_folding_depth_m
                )
            )
        )

    depth_m = (layer_boundary_elevation_m[0] - layer_boundary_elevation_m).astype(
        np.float64
    )
    thickness_m = np.maximum(depth_m[1:] - depth_m[:-1], 1e-6)
    return (integral_to_depth(depth_m[1:]) - integral_to_depth(depth_m[:-1])) / (
        thickness_m
    )


def get_response_time_conductance(
    specific_yield: ArrayFloat32,
    cell_area_m2: ArrayFloat32,
    response_time_days: float,
) -> ArrayFloat32:
    """Drain conductance that empties the groundwater above the drain level in a set time.

    The groundwater above the drain level, specific yield * (head - drain level) * area,
    leaves as conductance * (head - drain level). With conductance = specific yield *
    area / response time the cell is a linear store: the head above the drain level
    falls to 1 / e of its value in the response time, in every cell alike.

    Args:
        specific_yield: Specific yield of the top layer per cell (-).
        cell_area_m2: Cell area (m2).
        response_time_days: Time in which the head above the drain level falls to 1 / e (days).

    Returns:
        Drain conductance per cell (m2/day).
    """
    return (specific_yield * cell_area_m2 / np.float32(response_time_days)).astype(
        np.float32
    )


def get_aquifer_drain_conductance(
    hydraulic_conductivity_m_per_day: ArrayFloat32,
    thickness_below_stream_m: ArrayFloat32,
    height_above_stream_m: ArrayFloat32,
    hillslope_length_m: ArrayFloat32,
    cell_area_m2: ArrayFloat32,
    stream_wetted_perimeter_m: float = STREAM_WETTED_PERIMETER_M,
) -> ArrayFloat32:
    """Drain conductance of an aquifer that drains to parallel streams.

    Horizontal flow and radial flow into the stream, as resistances in series:

        c_horizontal = 4 L^2 / (pi^2 K (D + 2 h / pi))
        c_radial     = 2 L / (pi K) ln(min(D, 2 L) / u),  >= 0
        C            = A / (c_horizontal + c_radial)

    with L the hillslope length, K the conductivity, D the aquifer thickness below the
    stream, h the mean height of the water table above the stream (the cell head), u
    the wetted perimeter of the stream and A the cell area. The drainage time is
    Sy * (c_horizontal + c_radial).

    Sources: Kraijenhoff van de Leur (1958, De Ingenieur 70(19), B87-B94, eq. 3) for
    c_horizontal; Van der Molen and Wesseling (1991, Agric. Water Manage. 19, 1-16) for
    the flow above the stream (Hooghoudt's term in their eq. 1: half the midway head
    m0, with m0 = 4 h / pi for the elliptical water table) and for c_radial (eqs. 2, 8
    and 16; within 10 % of their exact solution).

    Args:
        hydraulic_conductivity_m_per_day: Hydraulic conductivity of the aquifer (m/day).
        thickness_below_stream_m: Thickness of the aquifer below the stream level (m).
        height_above_stream_m: Height of the water table above the stream level (m), >= 0.
        hillslope_length_m: Distance from the divide to the stream (m), > 0.
        cell_area_m2: Cell area (m2).
        stream_wetted_perimeter_m: Wetted perimeter of the streams (m).

    Returns:
        Drain conductance per cell (m2/day).

    Raises:
        ValueError: If the conductivity, thickness or hillslope length is not positive,
            or the height above the stream is negative.
    """
    if not (hydraulic_conductivity_m_per_day > 0).all():
        raise ValueError("Hydraulic conductivity must be positive.")
    if not (thickness_below_stream_m > 0).all():
        raise ValueError("Aquifer thickness below the stream must be positive.")
    if not (height_above_stream_m >= 0).all():
        raise ValueError("Height of the water table above the stream must be >= 0.")
    if not (hillslope_length_m > 0).all():
        raise ValueError("Hillslope length must be positive.")

    horizontal_days = (
        4.0
        * hillslope_length_m**2
        / (
            np.pi**2
            * hydraulic_conductivity_m_per_day
            * (thickness_below_stream_m + (2.0 / np.pi) * height_above_stream_m)
        )
    )
    # flow converges from at most the stream spacing deep; no radial resistance where
    # the aquifer is thinner than the wetted perimeter
    convergence_depth_m = np.minimum(thickness_below_stream_m, 2.0 * hillslope_length_m)
    radial_days = (
        2.0
        * hillslope_length_m
        / (np.pi * hydraulic_conductivity_m_per_day)
        * np.maximum(np.log(convergence_depth_m / stream_wetted_perimeter_m), 0.0)
    )
    return (cell_area_m2 / (horizontal_days + radial_days)).astype(np.float32)


def get_karst_drain_conductance(
    matrix_conductance_m2_per_day: ArrayFloat32,
    karst_conductance_m2_per_day: ArrayFloat32,
    karst_fraction: ArrayFloat32,
) -> ArrayFloat32:
    """Drain conductance of cells that are partly karst.

    Karst conduits carry groundwater to the springs much faster than the rock matrix
    does. The karst part of a cell drains with the karst conductance, the rest with
    the matrix conductance.

    Args:
        matrix_conductance_m2_per_day: Drain conductance of rock without karst (m2/day).
        karst_conductance_m2_per_day: Drain conductance of karst (m2/day).
        karst_fraction: Karst fraction of each cell (-), clipped to 0-1.

    Returns:
        Drain conductance per cell (m2/day).
    """
    fraction = np.clip(karst_fraction, 0.0, 1.0)
    return (
        (1.0 - fraction) * matrix_conductance_m2_per_day
        + fraction * karst_conductance_m2_per_day
    ).astype(np.float32)


@njit(cache=NUMBA_CACHE, parallel=True)
def drain_groundwater_to_rivers(
    head_m: ArrayFloat64,
    drain_elevation_m: ArrayFloat64,
    conductance_m2_per_hour: ArrayFloat64,
    storage_coefficient_m2: ArrayFloat64,
    drainage_m3: ArrayFloat64,
    remaining_storage_m3: ArrayFloat64,
) -> None:
    """Drains groundwater above the drain level to the river network for one hour.

    The drain outflow is conductance * (head - drain level). With the head falling as the
    aquifer empties, dh/dt = -C (h - z) / (S_y A), the head above the drain level decays
    exponentially over the hour, (h - z) * exp(-C dt / (S_y A)). This exact solution is
    stable for any conductance (the cell drain empties a cell within hours) and never
    drains below the drain level. Cells with the head at or below the drain level do not
    drain. A cell cannot drain more than the groundwater it still holds (the drain
    level can lie below the lowest level to which the aquifer can be emptied). The
    head and the remaining storage are updated in place.

    Args:
        head_m: Water table of the top layer per cell (m), updated in place.
        drain_elevation_m: Drain level per cell (m).
        conductance_m2_per_hour: Drain conductance per cell (m2/hour).
        storage_coefficient_m2: Specific yield of the top layer times the cell area (m2).
        drainage_m3: Output: groundwater drained in this hour per cell (m3).
        remaining_storage_m3: Groundwater that can still be taken from the cell (m3),
            updated in place.
    """
    for i in prange(head_m.size):
        excess_m: float = head_m[i] - drain_elevation_m[i]
        if excess_m <= 0.0 or conductance_m2_per_hour[i] <= 0.0:
            drainage_m3[i] = 0.0
            continue
        drained_m: float = excess_m * (
            1.0 - np.exp(-conductance_m2_per_hour[i] / storage_coefficient_m2[i])
        )
        drained_m3: float = min(
            drained_m * storage_coefficient_m2[i], max(remaining_storage_m3[i], 0.0)
        )
        drainage_m3[i] = drained_m3
        remaining_storage_m3[i] -= drained_m3
        head_m[i] -= drained_m3 / storage_coefficient_m2[i]


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
        # The conductivity is stored without the multiplier; the multiplier is applied
        # at every model start (hydraulic_conductivity_m_per_day), so a run uses the
        # multiplier of its own configuration everywhere.
        boundary_hydraulic_conductivity: ThreeDArrayFloat64 = read_zarr(
            self.model.files["other"]["groundwater/boundary_hydraulic_conductivity"]
        ).values.astype(np.float64)
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

    @property
    def available_groundwater_m3(self) -> ArrayFloat64:
        """Groundwater that wells and capillary rise can still take (m3).

        The MODFLOW available storage minus the groundwater already drained to the rivers
        in routing (net of riverbed seepage), which MODFLOW removes in its next step.

        Returns:
            Available groundwater per cell (m3).
        """
        net_drainage_m3 = np.maximum(
            self.hydrology.grid.var.pending_groundwater_drainage_m3
            - self.hydrology.grid.var.pending_river_transmission_loss_m3,
            0.0,
        )
        return np.maximum(self.modflow.available_groundwater_m3 - net_drainage_m3, 0.0)

    def get_drain_elevation(self, water_stage_m: ArrayFloat32) -> ArrayFloat64:
        """Level to which groundwater drains in this routing hour.

        Groundwater drains to the river water level, so groundwater drainage and riverbed
        seepage use the same level: a cell drains where the head is above it and river
        water seeps into the aquifer where the head is below it. The level is kept inside
        the top layer: never above the aquifer top and 0.1 m above its bottom.

        Args:
            water_stage_m: River (or lake) water level per cell (m).

        Returns:
            Drain level per cell (m).
        """
        top_m = self.grid.var.layer_boundary_elevation[0].astype(np.float64)
        bottom_m = self.grid.var.layer_boundary_elevation[1].astype(np.float64)
        return np.clip(water_stage_m.astype(np.float64), bottom_m + 0.1, top_m)

    def set_drain_geometry(self) -> None:
        """Set the parts of the drain conductance that do not change in time.

        The streams of a cell are parallel drains at the stream level (the bankfull river
        elevation, kept inside the top layer), two hillslope lengths apart. The karst part
        of a cell drains through conduits in the karst drainage time.
        """
        # get_drain_elevation returns float64 for the hourly routing
        self._drain_stream_level_m: ArrayFloat32 = self.get_drain_elevation(
            self._stream_elevation_m
        ).astype(np.float32)
        self._drain_thickness_below_stream_m: ArrayFloat32 = (
            self._drain_stream_level_m - self.grid.var.layer_boundary_elevation[1]
        )
        # the build caps the hillslope length at 1000 m, also where a cell has no stream
        self._drain_hillslope_length_m: ArrayFloat32 = self.hydrology.grid.load2d(
            self.model.files["grid"]["drainage/hillslope_length_m"]
        )
        self._drain_karst_conductance_m2_per_day: ArrayFloat32 = (
            get_response_time_conductance(
                specific_yield=self.grid.var.specific_yield[0],
                cell_area_m2=self.grid.var.cell_area,
                response_time_days=self.model.config["hydrology"]["karst"][
                    "drainage_response_time_days"
                ],
            )
        )

    def get_drainage_conductance(self, head_m: ArrayFloat64) -> ArrayFloat32:
        """Get the conductance of the groundwater drain to the river network.

        The rock matrix drains to the streams as an aquifer between parallel streams,
        through horizontal flow and a radial resistance near the stream
        (``get_aquifer_drain_conductance``). The horizontal flow passes below and above
        the stream level, so the conductance grows with the height of the water table
        above the stream and is updated every day. The conductivity is that of the
        aquifer data, without the decay with depth that the lateral flow uses. Karst
        conduits carry groundwater to the springs much faster than the rock matrix, so
        the karst part of a cell drains in the karst drainage time.

        Args:
            head_m: Groundwater head of the top layer (m).

        Returns:
            Drain conductance per grid cell (m2/day).
        """
        # the saturated rock above the stream: none when the head is below the stream,
        # at most up to the top of the layer
        height_above_stream_m: ArrayFloat32 = (
            np.clip(
                head_m.astype(np.float32),
                self._drain_stream_level_m,
                self.grid.var.layer_boundary_elevation[0],
            )
            - self._drain_stream_level_m
        )
        matrix_conductance_m2_per_day: ArrayFloat32 = get_aquifer_drain_conductance(
            hydraulic_conductivity_m_per_day=self.hydraulic_conductivity_m_per_day[0],
            thickness_below_stream_m=self._drain_thickness_below_stream_m,
            height_above_stream_m=height_above_stream_m,
            hillslope_length_m=self._drain_hillslope_length_m,
            cell_area_m2=self.grid.var.cell_area,
        )
        return get_karst_drain_conductance(
            matrix_conductance_m2_per_day=matrix_conductance_m2_per_day,
            karst_conductance_m2_per_day=self._drain_karst_conductance_m2_per_day,
            karst_fraction=self.hydrology.karst.karst_fraction,
        ).astype(np.float32)

    def update_drain_conductance(self, head_m: ArrayFloat64) -> None:
        """Update the hourly drain conductance used in routing from the current head.

        Args:
            head_m: Groundwater head of the top layer (m).
        """
        self.drain_conductance_m2_per_hour: ArrayFloat64 = (
            self.get_drainage_conductance(head_m).astype(np.float64) / 24.0
        )

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
        # same multiplier as the interior conductivity, so flow across the model
        # boundary is consistent with the flow inside
        boundary_hydraulic_conductivity: ThreeDArrayFloat64 = read_zarr(
            self.model.files["other"]["groundwater/boundary_hydraulic_conductivity"]
        ).values.astype(np.float64) * np.float64(self.hydraulic_conductivity_multiplier)
        # Lateral flow through the aquifer: conductivity decreasing with depth. The
        # drains and the riverbed are at the surface and keep the full conductivity.
        slope_m_per_m: ArrayFloat32 = self.hydrology.grid.load2d(
            self.model.files["grid"]["landsurface/slope_m_per_m"]
        )
        lateral_hydraulic_conductivity_m_per_day = (
            self.hydraulic_conductivity_m_per_day
            * get_conductivity_depth_decay_factor(
                self.grid.var.layer_boundary_elevation, slope_m_per_m
            )
        ).astype(np.float32)
        # the cells around the model take the slope of the nearest cell of the model
        slope_2d = np.pad(
            self.hydrology.grid.decompress(slope_m_per_m, fillvalue=np.nan),
            1,
            constant_values=np.nan,
        )
        nearest_row, nearest_column = distance_transform_edt(
            np.isnan(slope_2d), return_distances=False, return_indices=True
        )
        boundary_slope_m_per_m = slope_2d[nearest_row, nearest_column]
        boundary_hydraulic_conductivity = (
            boundary_hydraulic_conductivity
            * get_conductivity_depth_decay_factor(
                boundary_layer_boundary_elevation, boundary_slope_m_per_m
            )
        )
        # mean height of the land of each elevation zone above its nearest stream, and the
        # level of the streams of the cell, for the depth to the water table per zone
        self._zone_height_above_stream_m: TwoDArrayFloat32 = self.hydrology.grid.load3d(
            self.model.files["grid"]["landsurface/elevation_zone_height_above_stream_m"]
        ).astype(np.float32)
        self._stream_elevation_m: ArrayFloat32 = self.hydrology.grid.load2d(
            self.model.files["grid"]["routing/bankfull_river_elevation_m"]
        ).astype(np.float32)
        self.set_drain_geometry()
        self.update_drain_conductance(self.grid.var.heads[0])
        self.modflow = ModFlowSimulation(
            # One MODFLOW directory per spinup/run, so parallel runs in one model folder do
            # not overwrite each other's MODFLOW input files.
            working_directory=Path(self.model.config["simulation_root"])
            / self.model.run_name
            / "modflow_model",
            modflow_bin_folder=self.model.bin_folder / "modflow",
            topography=self.grid.var.elevation,
            gt=self.model.hydrology.grid.gt,
            specific_storage=np.zeros_like(self.grid.var.specific_yield),
            specific_yield=self.grid.var.specific_yield,
            layer_boundary_elevation=self.grid.var.layer_boundary_elevation,
            basin_mask=self.model.hydrology.grid.mask,
            hydraulic_conductivity=lateral_hydraulic_conductivity_m_per_day,
            heads=self.grid.var.heads,
            heads_update_callback=self.heads_update_callback,
            logger=self.model.logger,
            boundary_heads=boundary_heads,
            boundary_mask=boundary_mask,
            boundary_layer_boundary_elevation=boundary_layer_boundary_elevation,
            boundary_hydraulic_conductivity=boundary_hydraulic_conductivity,
            # The drain to the river network is not in MODFLOW: it is computed hourly in
            # routing (drain_groundwater_to_rivers), in step with the river water level.
            verbose=False,
        )
        # MODFLOW's own cell area, so the hourly drainage matches its storage exactly
        self.drain_storage_coefficient_m2: ArrayFloat64 = np.maximum(
            self.grid.var.specific_yield[0].astype(np.float64)
            * self.modflow.area.astype(np.float64),
            1e-6,
        )

    def step(
        self,
        groundwater_recharge_m: ArrayFloat32,
        groundwater_abstraction_m3: ArrayFloat32,
        capillary_rise_m: ArrayFloat32,
        river_seepage_m3: ArrayFloat64,
        river_drainage_m3: ArrayFloat64,
    ) -> None:
        """Perform a groundwater model step.

        Args:
            groundwater_recharge_m: Recharge to the groundwater (m/step).
            groundwater_abstraction_m3: Groundwater abstraction (m3/step).
            capillary_rise_m: Capillary rise drawn from groundwater (m/step).
            river_seepage_m3: Riverbed seepage into the aquifer during the previous day's
                routing (m3/step).
            river_drainage_m3: Groundwater drained to the rivers during the previous day's
                routing (hourly, see drain_groundwater_to_rivers) (m3/step). Seepage and
                drainage are netted per cell: a surplus is recharge, a deficit is taken from
                the aquifer, never more than what the hourly head drained above the drain
                level.

        Raises:
            AssertionError: If total groundwater abstraction exceeds available groundwater storage by more than numerical tolerance.
        """
        timer: TimingModule = TimingModule("Groundwater")
        assert (groundwater_abstraction_m3 + 1e-7 >= 0).all()
        groundwater_abstraction_m3[groundwater_abstraction_m3 < 0] = 0
        assert (groundwater_recharge_m >= 0).all()
        assert (capillary_rise_m >= 0).all()

        # Net recharge after accounting for capillary rise
        net_recharge_m: ArrayFloat32 = np.maximum(
            np.float32(0.0), groundwater_recharge_m - capillary_rise_m
        )
        excess_capillary_rise_m: ArrayFloat32 = np.maximum(
            np.float32(0.0), capillary_rise_m - groundwater_recharge_m
        )
        excess_capillary_rise_m3: ArrayFloat64 = excess_capillary_rise_m.astype(
            np.float64
        ) * self.grid.var.cell_area.astype(np.float64)

        net_river_exchange_m3: ArrayFloat64 = river_seepage_m3 - river_drainage_m3
        net_recharge_m = net_recharge_m + (
            np.maximum(net_river_exchange_m3, 0.0) / self.grid.var.cell_area
        ).astype(np.float32)

        total_groundwater_abstraction_m3: ArrayFloat64 = (
            groundwater_abstraction_m3.astype(np.float64)
            + excess_capillary_rise_m3
            + np.maximum(-net_river_exchange_m3, 0.0)
        )

        available_groundwater_m3: ArrayFloat64 = self.modflow.available_groundwater_m3
        excess_storage_m3: ArrayFloat64 = (
            total_groundwater_abstraction_m3 - available_groundwater_m3
        )
        epsilon_m3: ArrayFloat64 = (
            np.float64(1e-4) + np.float64(1e-5) * available_groundwater_m3
        )
        if (excess_storage_m3 > epsilon_m3).any():
            worst_cell: int = int((excess_storage_m3 - epsilon_m3).argmax())
            raise AssertionError(
                "Total groundwater abstraction exceeds available groundwater storage by "
                f"more than numerical tolerance ({excess_storage_m3[worst_cell]} m3 > "
                f"tolerance) in cell {worst_cell}: storage "
                f"{available_groundwater_m3[worst_cell]} m3, wells "
                f"{groundwater_abstraction_m3[worst_cell]} m3, capillary rise above "
                f"recharge {excess_capillary_rise_m3[worst_cell]} m3, net river drainage "
                f"{max(-net_river_exchange_m3[worst_cell], 0.0)} m3."
            )

        # Only correct minor floating-point inaccuracies within epsilon tolerance
        epsilon_mask: ArrayBool = excess_storage_m3 > 0
        if epsilon_mask.any():
            total_groundwater_abstraction_m3[epsilon_mask] = available_groundwater_m3[
                epsilon_mask
            ]

        timer.finish_split("Recharge & abstraction")

        if __debug__:
            groundwater_storage_pre = self.modflow.groundwater_content_m3

        self.modflow.set_recharge_m3(
            (net_recharge_m * self.grid.var.cell_area).astype(np.float32)
        )
        self.modflow.set_groundwater_abstraction_m3(total_groundwater_abstraction_m3)
        timer.finish_split("Set MODFLOW forcing")

        self.modflow.step()
        timer.finish_split("MODFLOW solve")

        # the water table above the stream level conducts water to the stream as well, so
        # the drain conductance for the next day follows the new head
        self.update_drain_conductance(self.modflow.heads[0])

        if __debug__:
            influxes: list[npt.NDArray[np.float64] | np.float64] = [
                net_recharge_m.astype(np.float64) * self.grid.var.cell_area,
                self.boundary_inflow_m3,
            ]
            outfluxes: list[npt.NDArray[np.float64] | np.float64] = [
                total_groundwater_abstraction_m3,
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

        timer.finish_split("Balance check")

        # head of the top layer (m above sea level), e.g. for observation wells
        groundwater_head_top_layer_m: ArrayFloat64 = self.modflow.heads[0]

        self.report(locals())
        timer.finish_split("Reporting")

        if self.model.timing:
            self.model.logger.debug(timer)

    @property
    def hydraulic_conductivity_multiplier(self) -> float:
        """Multiplier on the aquifer hydraulic conductivity from the configuration (-)."""
        return float(
            self.model.config["hydrology"]["groundwater"][
                "hydraulic_conductivity_multiplier"
            ]
        )

    @property
    def hydraulic_conductivity_m_per_day(self) -> npt.NDArray[np.float32]:
        """Aquifer hydraulic conductivity per layer and cell, including the configured multiplier (m/day).

        The grid variable stores the conductivity without the multiplier.
        """
        return (
            self.grid.var.groundwater_hydraulic_conductivity_m_per_day
            * np.float32(self.hydraulic_conductivity_multiplier)
        ).astype(np.float32)

    @property
    def groundwater_content_m3(self) -> npt.NDArray[np.float64]:
        """Groundwater content in cubic meters.

        Kept in float64: storage is large relative to daily fluxes, so a float32 copy
        would lose the balance precision.

        Returns:
            Groundwater content in cubic meters in active grid cells.
        """
        return self.modflow.groundwater_content_m3

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

    def get_groundwater_depth_per_zone(
        self, HRU_to_grid: npt.NDArray[np.int32], elevation_zone: npt.NDArray[np.int32]
    ) -> ArrayFloat32:
        """Depth of the water table below the land surface of each HRU (m).

        A grid cell has one groundwater head, but its land lies at different heights
        above the streams: the valley bottom close to the water table, the slopes and
        plateaus far above it. The groundwater of the whole cell is connected, with the
        water table at the height of the head above the stream level of the cell. The
        depth to the water table of an HRU is therefore the height of its elevation zone
        above the nearest stream minus the height of the head above the stream level.

        Args:
            HRU_to_grid: Grid cell of each HRU.
            elevation_zone: Elevation zone of each HRU, counted from the stream upwards.

        Returns:
            Depth to the water table per HRU (m).
        """
        head_above_stream_m = (
            self.modflow.heads[0].astype(np.float32) - self._stream_elevation_m
        )
        zone_height_m = self._zone_height_above_stream_m[elevation_zone, HRU_to_grid]
        assert not np.isnan(zone_height_m).any()
        return np.maximum(
            zone_height_m - head_above_stream_m[HRU_to_grid], np.float32(0.0)
        ).astype(np.float32)

    def decompress(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Decompress data from compressed grid format to full grid format.

        Args:
            data: Data in compressed grid format.

        Returns:
            Data in full grid format.
        """
        return self.hydrology.grid.decompress(data)
