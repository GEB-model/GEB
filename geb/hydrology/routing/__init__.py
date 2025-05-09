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

import math

import numpy as np
import pyflwdir

from geb.hydrology.lakes_reservoirs import OFF
from geb.module import Module
from geb.workflows import balance_check

from .subroutines import (
    PIT,
    dirDownstream,
    dirUpstream,
    kinematic,
)


def get_channel_ratio(river_width, river_length, cell_area):
    return np.minimum(
        1.0,
        river_width * river_length / cell_area,
    )


def calculate_river_storage_from_discharge(
    discharge, river_alpha, river_length, river_beta
):
    # The momentum equation, see eq. 18 in https://gmd.copernicus.org/articles/13/3267/2020/
    cross_sectional_area_of_flow = river_alpha * discharge**river_beta
    return cross_sectional_area_of_flow * river_length


def calculate_discharge_from_storage(
    river_storage, river_alpha, river_length, river_beta
):
    # The momentum equation (solved for Q), see eq. 18 in https://gmd.copernicus.org/articles/13/3267/2020/
    return (river_storage / (river_length * river_alpha)) ** (1 / river_beta)


class Routing(Module):
    """
    ROUTING

    routing using the kinematic wave

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    EWRef                 potential evaporation rate from water surface                                     m
    waterBodyID           lakes/reservoirs map with a single ID for each lake/reservoir                     --
    dirUp                 river network in upstream direction                                               --
    routing_step_length_seconds             number of seconds per routing timestep                                            s
    discharge             discharge                                                                         m3/s
    """

    def __init__(self, model, hydrology):
        super().__init__(model)

        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    def spinup(self):
        ldd = self.grid.load(
            self.model.files["grid"]["routing/ldd"],
            compress=False,
        )

        self.river_network = pyflwdir.from_array(
            ldd,
            ftype="ldd",
            transform=self.grid.transform,
            latlon=True,
            mask=~self.grid.mask,
        )
        self.river_network.order_cells(method="walk")

        mapper = np.full(ldd.size, -1, dtype=np.int32)
        indices = np.arange(ldd.size)[~self.grid.mask.ravel()]
        mapper[indices] = np.arange(indices.size)
        self.grid.var.idx_ds = mapper[
            self.river_network.idxs_ds[~self.grid.mask.ravel()]
        ]

        self.grid.var.upstream_area = self.river_network.upstream_area(unit="m2")
        self.grid.var.upstream_area[self.grid.var.upstream_area < 0] = np.nan
        self.grid.var.upstream_area = self.grid.var.upstream_area[~self.grid.mask]
        self.grid.var.upstream_area_n_cells = self.river_network.upstream_area(
            unit="cell"
        )
        self.grid.var.upstream_area_n_cells[self.grid.var.upstream_area_n_cells < 0] = 0
        self.grid.var.upstream_area_n_cells = self.grid.var.upstream_area_n_cells[
            ~self.grid.mask
        ]

        # in previous versions of GEB we followed the CWatM specification, where masked data
        # was set at 0. We now use the official LDD specification where masked data is 255
        # (max value of uint8). To still support old versions we set these values of 255 to
        # 0 for now. When all models have been updated, this can be removed and the
        # subroutines can be updated accordingly.
        ldd[ldd == 255] = 0

        self.grid.var.lddCompress = self.grid.compress(ldd)

        # TODO: is this done in the right direction?
        self.grid.var.dirDown_ = mapper[self.river_network.idxs_seq]

        # create a compressed version of the ldd
        ldd_short = mapper[
            self.grid.compress(self.river_network.idxs_ds.reshape(ldd.shape))
        ]
        # set all pits (i.e., cells with an outflow to itself) to -1
        ldd_short[np.arange(ldd_short.size) == ldd_short] = -1

        dirUp, self.grid.var.dirupLen, self.grid.var.dirupID = dirUpstream(ldd_short)
        self.grid.var.dirDown, _ = dirDownstream(dirUp, self.grid.var.lddCompress, [])

        # number of substep per day
        self.var.n_routing_substeps = 24
        # kinematic wave parameter: 0.6 is for broad sheet flow

        self.var.river_beta = 0.6  # TODO: Make this a parameter

        # Channel Manning's n
        self.grid.var.river_mannings = (
            self.grid.load(self.model.files["grid"]["routing/mannings"])
            * self.model.config["parameters"]["manningsN"]
        )
        assert (self.grid.var.river_mannings > 0).all()

        # Channel length [meters]
        self.grid.var.river_length = self.grid.load(
            self.model.files["grid"]["routing/river_length"]
        )
        assert (
            self.grid.var.river_length[self.grid.var.lddCompress != PIT] > 0
        ).all(), "Channel length must be greater than 0 for all cells except for pits"

        # where there is a pit, the river length is set to distance to the center of the cell,
        # thus half of the sqrt of the cell area
        self.grid.var.river_length[self.grid.var.lddCompress == PIT] = (
            np.sqrt(self.grid.var.cell_area[self.grid.var.lddCompress == PIT]) / 2
        )

        # Channel bottom width [meters]
        self.grid.var.river_width = self.grid.load(
            self.model.files["grid"]["routing/river_width"]
        )

        # Corresponding sub-timestep (seconds)
        self.var.routing_step_length_seconds = (
            self.model.timestep_length.total_seconds() / self.var.n_routing_substeps
        )

        # for a river, the wetted perimeter can be approximated by the channel width
        river_wetted_perimeter = self.grid.var.river_width

        # Channel gradient (fraction, dy/dx)
        minimum_river_slope = 0.0001
        river_slope = np.maximum(
            self.grid.load(self.model.files["grid"]["routing/river_slope"]),
            minimum_river_slope,
        )

        # river_alpha for kinematic wave
        # source: https://gmd.copernicus.org/articles/13/3267/2020/ eq. 21
        self.grid.var.river_alpha = (
            self.grid.var.river_mannings
            * river_wetted_perimeter ** (2 / 3)
            / np.sqrt(river_slope)
        ) ** self.var.river_beta

        # Initialise water volume and discharge in rivers, just set at 0 [m3]
        self.grid.var.river_storage_m3 = np.ones_like(
            self.grid.var.river_width, dtype=np.float64
        )

        self.grid.var.discharge_m3_s = calculate_discharge_from_storage(
            self.grid.var.river_storage_m3,
            self.grid.var.river_alpha,
            self.grid.var.river_length,
            self.var.river_beta,
        )
        self.grid.var.discharge_m3_s_substep = np.full(
            (self.var.n_routing_substeps, self.grid.var.discharge_m3_s.size),
            0,
            dtype=self.grid.var.discharge_m3_s.dtype,
        )

    def step(self, total_runoff, channel_abstraction_m, return_flow):
        if __debug__:
            pre_river_storage_m3 = self.grid.var.river_storage_m3.copy()
            pre_storage = self.hydrology.lakes_reservoirs.var.storage.copy()

        return_flow_m3_per_routing_step = (
            return_flow * self.grid.var.cell_area / self.var.n_routing_substeps
        )
        channel_abstraction_m3_per_routing_step = (
            channel_abstraction_m
            * self.grid.var.cell_area
            / self.var.n_routing_substeps
        )

        return_flow_m3_to_water_bodies_per_routing_step = np.bincount(
            self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
            weights=return_flow_m3_per_routing_step[self.grid.var.waterBodyID != -1],
        )
        return_flow_m3_per_routing_step[self.grid.var.waterBodyID != -1] = 0.0

        runoff_m3_per_routing_step = (
            total_runoff * self.grid.var.cell_area / self.var.n_routing_substeps
        )

        runoff_m3_per_routing_step_pre = runoff_m3_per_routing_step.sum()

        runoff_m3_per_routing_step_water_bodies = np.bincount(
            self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
            weights=runoff_m3_per_routing_step[self.grid.var.waterBodyID != -1],
        )

        runoff_m3_per_routing_step[self.grid.var.waterBodyID != -1] = 0.0

        assert math.isclose(
            runoff_m3_per_routing_step_water_bodies.sum()
            + runoff_m3_per_routing_step.sum(),
            runoff_m3_per_routing_step_pre,
            rel_tol=1e-6,
            abs_tol=0,
        )

        self.grid.var.discharge_m3_s_substep = np.full(
            (self.var.n_routing_substeps, self.grid.var.discharge_m3_s.size),
            np.nan,
            dtype=self.grid.var.discharge_m3_s.dtype,
        )

        if self.model.current_timestep == 0 and self.model.in_spinup:
            self.var.discharge_out_of_water_bodies_into_other_water_bodies_m3 = (
                np.zeros(
                    self.hydrology.lakes_reservoirs.var.capacity.size, dtype=np.float32
                )
            )
            self.grid.var.river_storage_m3[self.grid.var.waterBodyID != -1] = 0

        total_discharge_out_of_water_bodies_into_other_water_bodies_m3_pre = (
            self.var.discharge_out_of_water_bodies_into_other_water_bodies_m3.sum()
        )

        potential_evaporation_per_water_body_m3_per_routing_step = (
            np.bincount(
                self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
                weights=self.grid.var.EWRef[self.grid.var.waterBodyID != -1],
            )
            / np.bincount(self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1])
            * self.hydrology.lakes_reservoirs.var.lake_area
        ) / self.var.n_routing_substeps
        potential_evaporation_per_water_body_m3_per_routing_step[
            self.hydrology.lakes_reservoirs.var.water_body_type == OFF
        ] = 0

        if __debug__:
            # these are for balance checks, the sum of all routing steps
            side_flow_channel_m3 = 0
            evaporation_in_rivers_m3 = 0
            waterbody_evaporation_m3 = 0
            outflow_at_outflows_m3 = 0

        for subrouting_step in range(self.var.n_routing_substeps):
            # ensure there is no river storage in the water bodies
            assert (
                self.grid.var.river_storage_m3[self.grid.var.waterBodyID != -1] == 0
            ).all()

            # the ratio of each grid cell that is currently covered by a river
            channel_ratio = get_channel_ratio(
                river_length=self.grid.var.river_length,
                river_width=self.grid.var.river_width,
                cell_area=self.grid.var.cell_area,
            )

            # calculate evaporation from rivers per timestep usting the current channel ratio
            evaporation_in_rivers_m3_per_substep = (
                self.grid.var.EWRef * channel_ratio * self.grid.var.cell_area
            ) / self.var.n_routing_substeps

            # limit evaporation to available water in river
            evaporation_in_rivers_m3_per_substep = np.minimum(
                evaporation_in_rivers_m3_per_substep, self.grid.var.river_storage_m3
            )

            # update river storage
            self.grid.var.river_storage_m3 -= evaporation_in_rivers_m3_per_substep
            assert (self.grid.var.river_storage_m3 >= 0).all()

            # when river storage is updated, discharge also needs to be updated
            self.grid.var.discharge_m3_s = calculate_discharge_from_storage(
                river_storage=self.grid.var.river_storage_m3,
                river_alpha=self.grid.var.river_alpha,
                river_length=self.grid.var.river_length,
                river_beta=self.var.river_beta,
            )

            outflow_per_waterbody_m3 = self.hydrology.lakes_reservoirs.substep(
                current_substep=subrouting_step,
                n_routing_substeps=self.var.n_routing_substeps,
                routing_step_length_seconds=self.var.routing_step_length_seconds,
            )

            actual_evaporation_from_water_bodies_per_routing_step_m3 = np.minimum(
                potential_evaporation_per_water_body_m3_per_routing_step,
                self.hydrology.lakes_reservoirs.var.storage,
            )

            self.hydrology.lakes_reservoirs.var.storage -= (
                actual_evaporation_from_water_bodies_per_routing_step_m3
            )

            side_flow_channel_m3_per_routing_step = (
                runoff_m3_per_routing_step
                + return_flow_m3_per_routing_step
                - channel_abstraction_m3_per_routing_step
            )
            assert (
                side_flow_channel_m3_per_routing_step[self.grid.var.waterBodyID != -1]
                == 0
            ).all()

            # m2 because this is per unit of channel length, see division
            side_flow_channel_m2_per_s = (
                side_flow_channel_m3_per_routing_step
                / self.grid.var.river_length
                / self.var.routing_step_length_seconds
            )

            assert (
                side_flow_channel_m2_per_s[self.grid.var.waterBodyID != -1] == 0.0
            ).all()

            self.grid.var.discharge_m3_s = kinematic(
                self.grid.var.discharge_m3_s,
                side_flow_channel_m2_per_s.astype(np.float64),
                self.grid.var.dirDown,
                self.grid.var.dirupLen,
                self.grid.var.dirupID,
                self.grid.var.river_alpha,
                self.var.river_beta,
                self.var.routing_step_length_seconds,
                self.grid.var.river_length,
                is_waterbody=self.grid.var.waterBodyID != -1,
                is_outflow=self.grid.var.waterbody_outflow_points != -1,
                waterbody_id=self.grid.var.waterBodyID,
                waterbody_storage=self.hydrology.lakes_reservoirs.var.storage,
                outflow_per_waterbody_m3=outflow_per_waterbody_m3,
            )

            # update river storage
            self.grid.var.river_storage_m3 = calculate_river_storage_from_discharge(
                self.grid.var.discharge_m3_s,
                self.grid.var.river_alpha,
                self.grid.var.river_length,
                self.var.river_beta,
            )

            outflow_at_outflows_m3_substep = (
                self.grid.var.discharge_m3_s[self.grid.var.lddCompress == PIT].sum()
                * self.var.routing_step_length_seconds
            )

            # get storage discharged into lakes and reservoirs
            discharge_into_water_bodies_m3 = (
                np.bincount(
                    self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
                    weights=self.grid.var.discharge_m3_s[
                        self.grid.var.waterBodyID != -1
                    ],
                )
                * self.var.routing_step_length_seconds
                + self.var.discharge_out_of_water_bodies_into_other_water_bodies_m3
                + runoff_m3_per_routing_step_water_bodies
                + return_flow_m3_to_water_bodies_per_routing_step
            )

            # remove storage and discharge
            # that is discharged into lakes and reservoirs from river storage
            # and set discharge to 0 in those locations
            self.grid.var.river_storage_m3[self.grid.var.waterBodyID != -1] = 0.0
            self.grid.var.discharge_m3_s[self.grid.var.waterBodyID != -1] = 0.0

            self.hydrology.lakes_reservoirs.var.storage += (
                discharge_into_water_bodies_m3
            )

            assert (
                self.grid.var.river_storage_m3[self.grid.var.waterBodyID != -1] == 0.0
            ).all()

            self.grid.var.discharge_m3_s_substep[subrouting_step, :] = (
                self.grid.var.discharge_m3_s.copy()
            )

            if __debug__:
                # Discharge at outlets and lakes and reservoirs
                outflow_at_outflows_m3 += outflow_at_outflows_m3_substep
                side_flow_channel_m3 += side_flow_channel_m3_per_routing_step
                waterbody_evaporation_m3 += (
                    actual_evaporation_from_water_bodies_per_routing_step_m3
                )
                evaporation_in_rivers_m3 += evaporation_in_rivers_m3_per_substep

        assert not np.isnan(self.grid.var.discharge_m3_s).any()

        if __debug__:
            # TODO: make dependent on routing step length
            balance_check(
                how="sum",
                influxes=[
                    total_runoff * self.grid.var.cell_area,
                    return_flow * self.grid.var.cell_area,
                ],
                outfluxes=[
                    channel_abstraction_m * self.grid.var.cell_area,
                    outflow_at_outflows_m3,
                    evaporation_in_rivers_m3,
                    waterbody_evaporation_m3,
                ],
                prestorages=[
                    pre_storage,
                    pre_river_storage_m3,
                    total_discharge_out_of_water_bodies_into_other_water_bodies_m3_pre,
                ],
                poststorages=[
                    self.hydrology.lakes_reservoirs.var.storage,
                    self.grid.var.river_storage_m3,
                    self.var.discharge_out_of_water_bodies_into_other_water_bodies_m3,
                ],
                name="routing_1",
                tollerance=100_000,
            )

            self.routing_loss = (
                evaporation_in_rivers_m3.sum()
                + waterbody_evaporation_m3.sum()
                + outflow_at_outflows_m3.sum()
            )

        assert (
            self.grid.var.river_storage_m3[self.grid.var.waterBodyID != -1] == 0.0
        ).all()

        self.report(self, locals())

    @property
    def name(self):
        return "hydrology.routing"
