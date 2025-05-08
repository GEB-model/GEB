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

from geb.module import Module
from geb.workflows import balance_check

from .subroutines import PIT, define_river_network, dirID, dirUpstream, kinematic


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

        (
            _,
            _,
            _,
            self.grid.var.dirupLen,
            self.grid.var.dirupID,
            self.grid.var.downstruct_no_water_bodies,
            _,
            self.grid.var.dirDown,
            _,
        ) = define_river_network(ldd, self.hydrology.grid)

        # TODO: is this done in the right direction?
        self.grid.var.dirDown_ = mapper[self.river_network.idxs_seq]
        # ldd_short = self.grid.compress(
        #     dirID(self.grid.decompress(np.arange(self.grid.var.idx_ds.size)), ldd)
        # )
        # _, self.grid.var.dirupLen, self.grid.var.dirupID = dirUpstream(ldd_short)

        # self.grid.var.upstream_area_n_cells = upstreamArea(
        #     self.grid.var.dirDown,
        #     dirshort,
        #     self.grid.full_compressed(1, dtype=np.int32),
        # )
        # self.grid.var.UpArea = upstreamArea(
        #     self.grid.var.dirDown,
        #     dirshort,
        #     self.grid.var.cell_area.astype(np.float64),
        # )

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
        self.grid.var.river_storage_m3 = np.zeros_like(
            self.grid.var.river_width, dtype=np.float32
        )
        self.grid.var.river_storage_m3.fill(1)
        self.grid.var.discharge = calculate_discharge_from_storage(
            self.grid.var.river_storage_m3,
            self.grid.var.river_alpha,
            self.grid.var.river_length,
            self.var.river_beta,
        )
        self.grid.var.discharge_substep = np.full(
            (self.var.n_routing_substeps, self.grid.var.discharge.size),
            0,
            dtype=self.grid.var.discharge.dtype,
        )

    def step(self, total_runoff, channel_abstraction_m, return_flow):
        """
        Dynamic part of the routing module

        * calculate evaporation from channels
        * calculate riverbed exchange between riverbed and groundwater
        * calculate sideflow -> inflow to river
        """

        print("REMOVE THIS")
        channel_abstraction_m.fill(0)
        return_flow.fill(0)
        # total_runoff.fill(0)

        if __debug__:
            pre_river_storage_m3 = self.grid.var.river_storage_m3.copy()
            pre_storage = self.hydrology.lakes_reservoirs.var.storage.copy()

        # riverbed infiltration (m3):
        # - current implementation based on Inge's principle (later, will be based on groundater head (MODFLOW) and can be negative)
        # - happening only if 0.0 < baseflow < nonFossilGroundwaterAbs
        # - infiltration rate will be based on aquifer saturated conductivity
        # - limited to fracWat
        # - limited to available river_storage_m3
        # - this infiltration will be handed to groundwater in the next time step

        """
        self.grid.var.riverbedExchange = np.maximum(0.0,  np.minimum(self.grid.var.river_storage_m3, np.where(self.grid.var.baseflow > 0.0, \
                                np.where(self.grid.var.nonFossilGroundwaterAbs > self.grid.var.baseflow, \
                                self.grid.var.kSatAquifer * self.grid.var.fracVegCover[5] * self.grid.var.cell_area, \
                                0.0), 0.0)))
        # to avoid flip flop
        self.grid.var.riverbedExchange = np.minimum(self.grid.var.riverbedExchange, 0.95 * self.grid.var.river_storage_m3)


                if self.grid.var.modflow:
            self.grid.var.interflow[No] = np.where(self.grid.var.capriseindex == 100, toGWorInterflow,
                                              self.grid.var.percolationImp * toGWorInterflow)
        else:
            self.grid.var.interflow[No] = self.grid.var.percolationImp * toGWorInterflow
        """

        # self.grid.var.riverbedExchange = np.where(self.grid.var.waterBodyID > 0, 0., self.grid.var.riverbedExchange)

        # riverbedExchangeDt = self.grid.var.riverbedExchangeM3 / self.var.n_routing_substeps

        net_channel_abstraction_m3_per_routing_step = (
            (channel_abstraction_m - return_flow)
            * self.grid.var.cell_area
            / self.var.n_routing_substeps
        )
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

        self.grid.var.discharge_substep = np.full(
            (self.var.n_routing_substeps, self.grid.var.discharge.size),
            np.nan,
            dtype=self.grid.var.discharge.dtype,
        )

        if self.model.current_timestep == 0 and self.model.in_spinup:
            self.var.discharge_volume_out_of_water_bodies_into_other_water_bodies = (
                np.zeros(
                    self.hydrology.lakes_reservoirs.var.capacity.size, dtype=np.float32
                )
            )
            self.grid.var.river_storage_m3[self.grid.var.waterBodyID != -1] = 0

        total_discharge_volume_out_of_water_bodies_into_other_water_bodies_pre = (
            self.var.discharge_volume_out_of_water_bodies_into_other_water_bodies.sum()
        )

        if __debug__:
            # these are for balance checks, the sum of all routing steps
            side_flow_channel_m3 = 0
            evaporation_in_rivers_m3 = 0
            waterbody_evaporation_m3 = 0
            outflow_at_outflows_m3 = 0

        for subrouting_step in range(self.var.n_routing_substeps):
            print(
                f"Routing step {subrouting_step + 1} of {self.var.n_routing_substeps}"
            )
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
            self.grid.var.discharge = calculate_discharge_from_storage(
                river_storage=self.grid.var.river_storage_m3,
                river_alpha=self.grid.var.river_alpha,
                river_length=self.grid.var.river_length,
                river_beta=self.var.river_beta,
            )

            # # this variable is named outflow_toriver_network in the lakes and reservoirs module
            # # because it is outflow from the waterbodies to the river network
            # inflow_to_river_network, waterbody_evaporation_m3_Dt = (
            #     self.hydrology.lakes_reservoirs.substep(
            #         current_substep=subrouting_step,
            #         n_routing_substeps=self.var.n_routing_substeps,
            #         routing_step_length_seconds=self.var.routing_step_length_seconds,
            #         discharge=self.grid.var.discharge,
            #         total_runoff=total_runoff,
            #     )
            # )

            waterbody_evaporation_m3_Dt = np.zeros_like(
                self.grid.var.discharge, dtype=np.float32
            )

            side_flow_channel_m3_per_routing_step = (
                runoff_m3_per_routing_step - net_channel_abstraction_m3_per_routing_step
            )
            assert (
                side_flow_channel_m3_per_routing_step[self.grid.var.waterBodyID != -1]
                == 0
            ).all()

            # m2 because this is per unit of channel length, see division
            side_flow_channel_m2_per_routing_step = (
                side_flow_channel_m3_per_routing_step
                / self.grid.var.river_length
                / self.var.routing_step_length_seconds
            )

            assert (
                side_flow_channel_m2_per_routing_step[self.grid.var.waterBodyID != -1]
                == 0.0
            ).all()

            total_river_storage_m3_pre_kinematic = self.grid.var.river_storage_m3.sum()

            # TODO: Stop water from flowing once it is in a water body
            self.grid.var.discharge = kinematic(
                self.grid.var.discharge,
                side_flow_channel_m2_per_routing_step.astype(np.float32),
                self.grid.var.dirDown,
                self.grid.var.dirupLen,
                self.grid.var.dirupID,
                self.grid.var.river_alpha,
                self.var.river_beta,
                self.var.routing_step_length_seconds,
                self.grid.var.river_length,
                is_waterbody=self.grid.var.waterBodyID != -1,
            )

            # discharge_in_water_bodies = self.grid.var.discharge.copy()
            # discharge_in_water_bodies = (discharge_in_water_bodies > 0).astype(np.int32)
            # discharge_in_water_bodies[self.grid.var.waterBodyID == -1] = -1
            # discharge_in_water_bodies[self.grid.var.waterbody_outflow_points != -1] = 2

            # import matplotlib.pyplot as plt

            # fig, (ax0, ax1) = plt.subplots(1, 2)

            # ax0.imshow(self.grid.decompress(discharge_in_water_bodies))
            # ax1.imshow(
            #     self.grid.decompress(self.grid.var.waterBodyID) % 7,
            # )

            # plt.savefig("discharge_in_water_bodies.png", dpi=300)

            # update river storage
            self.grid.var.river_storage_m3 = calculate_river_storage_from_discharge(
                self.grid.var.discharge,
                self.grid.var.river_alpha,
                self.grid.var.river_length,
                self.var.river_beta,
            )

            outflow_at_outflows_m3_substep = (
                self.grid.var.discharge[self.grid.var.lddCompress == PIT].sum() * 3600
            )

            assert math.isclose(
                total_river_storage_m3_pre_kinematic
                + side_flow_channel_m3_per_routing_step.sum(),
                self.grid.var.river_storage_m3.sum() + outflow_at_outflows_m3_substep,
                rel_tol=1e-2,
                abs_tol=100,
            )

            total_river_storage_m3_pre_step = self.grid.var.river_storage_m3.sum()
            total_discharge_volume_out_of_water_bodies_into_other_water_bodies_pre_step = self.var.discharge_volume_out_of_water_bodies_into_other_water_bodies.sum()

            # get storage discharged into lakes and reservoirs
            discharge_volume_into_water_bodies = (
                np.bincount(
                    self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
                    weights=self.grid.var.river_storage_m3[
                        self.grid.var.waterBodyID != -1
                    ],
                )
                + self.var.discharge_volume_out_of_water_bodies_into_other_water_bodies
                + runoff_m3_per_routing_step_water_bodies
            )
            discharge_volume_out_of_water_bodies = discharge_volume_into_water_bodies[
                self.grid.var.waterbody_outflow_points
            ]
            discharge_volume_out_of_water_bodies[
                self.grid.var.waterbody_outflow_points == -1
            ] = 0.0

            # move the water from the water bodies to the river network
            discharge_volume_out_of_water_bodies = discharge_volume_out_of_water_bodies[
                self.grid.var.dirDown_
            ]

            # remove storage that is discharged into lakes and reservoirs from river storage
            self.grid.var.river_storage_m3[self.grid.var.waterBodyID != -1] = 0.0

            self.var.discharge_volume_out_of_water_bodies_into_other_water_bodies = (
                np.bincount(
                    self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
                    weights=discharge_volume_out_of_water_bodies[
                        self.grid.var.waterBodyID != -1
                    ],
                )
            )
            discharge_volume_out_of_water_bodies[self.grid.var.waterBodyID != -1] = 0.0

            assert math.isclose(
                discharge_volume_into_water_bodies.sum(),
                discharge_volume_out_of_water_bodies.sum()
                + self.var.discharge_volume_out_of_water_bodies_into_other_water_bodies.sum(),
                rel_tol=1e-7,
            )

            # add the water from the water bodies to the river storage
            self.grid.var.river_storage_m3 += discharge_volume_out_of_water_bodies

            assert (
                self.grid.var.river_storage_m3[self.grid.var.waterBodyID != -1] == 0.0
            ).all()

            assert math.isclose(
                total_river_storage_m3_pre_step
                + total_discharge_volume_out_of_water_bodies_into_other_water_bodies_pre_step
                + runoff_m3_per_routing_step_water_bodies.sum(),
                self.grid.var.river_storage_m3.sum()
                + self.var.discharge_volume_out_of_water_bodies_into_other_water_bodies.sum(),
                abs_tol=0,
                rel_tol=1e-6,
            )

            self.grid.var.discharge_substep[subrouting_step, :] = (
                self.grid.var.discharge.copy()
            )

            if __debug__:
                # Discharge at outlets and lakes and reservoirs
                outflow_at_outflows_m3 += outflow_at_outflows_m3_substep
                side_flow_channel_m3 += side_flow_channel_m3_per_routing_step
                waterbody_evaporation_m3 += waterbody_evaporation_m3_Dt
                evaporation_in_rivers_m3 += evaporation_in_rivers_m3_per_substep

        assert not np.isnan(self.grid.var.discharge).any()

        if __debug__:
            # TODO: make dependent on routing step length
            assert balance_check(
                how="sum",
                influxes=[
                    total_runoff * self.grid.var.cell_area,
                ],
                outfluxes=[
                    # channel_abstraction_m * self.grid.var.cell_area,
                    # return_flow * self.grid.var.cell_area,
                    outflow_at_outflows_m3,
                    evaporation_in_rivers_m3,
                ],
                prestorages=[
                    pre_river_storage_m3,
                    total_discharge_volume_out_of_water_bodies_into_other_water_bodies_pre,
                ],
                poststorages=[
                    self.grid.var.river_storage_m3,
                    self.var.discharge_volume_out_of_water_bodies_into_other_water_bodies,
                ],
                name="routing_1",
                tollerance=1000,
            )

            self.routing_loss = (
                evaporation_in_rivers_m3.sum()
                + waterbody_evaporation_m3.sum()
                + outflow_at_outflows_m3.sum()
            )

        self.report(self, locals())

    @property
    def name(self):
        return "hydrology.routing"
