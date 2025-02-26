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

from .subroutines import (
    define_river_network,
    upstreamArea,
    kinematic,
)
import numpy as np
from geb.workflows import balance_check
from .subroutines import PIT


def get_channel_ratio(river_width, river_length, cell_area):
    return np.minimum(
        1.0,
        river_width * river_length / cell_area,
    )


class Routing(object):
    """
    ROUTING

    routing using the kinematic wave

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    EWRef                 potential evaporation rate from water surface                                     m
    waterBodyID           lakes/reservoirs map with a single ID for each lake/reservoir                     --
    dirUp                 river network in upstream direction                                               --
    lddCompress           compressed river network (without missing values)                                 --
    compress_LR           boolean map as mask map for compressing lake/reservoir                            --
    lakeEvaFactor         a factor which increases evaporation from lake because of wind                    --
    dtRouting             number of seconds per routing timestep                                            s
    evaporation_from_water_bodies_per_routing_step
    discharge             discharge                                                                         m3/s
    cellArea              Cell area [m²] of each simulated mesh
    openWaterEvap         Simulated evaporation from open areas                                             m
    """

    def __init__(self, model):
        """
        Initial part of the routing module

        * load and create a river network
        * calculate river network parameter e.g. river length, width, gradient etc.
        * calculate initial filling
        * calculate manning's roughness coefficient
        """
        self.grid = model.data.grid
        self.HRU = model.data.HRU
        self.model = model

        if self.model.in_spinup:
            self.spinup()

    def spinup(self):
        self.var = self.model.store.create_bucket("routing.var")

        ldd = self.grid.load(
            self.model.files["grid"]["routing/ldd"],
            compress=False,
        )
        # in previous versions of GEB we followed the CWatM specification, where masked data
        # was set at 0. We now use the official LDD specification where masked data is 255
        # (max value of uint8). To still support old versions we set these values of 255 to
        # 0 for now. When all models have been updated, this can be removed and the
        # subroutines can be updated accordingly.
        ldd[ldd == 255] = 0

        (
            self.grid.var.lddCompress,
            dirshort,
            self.grid.var.dirUp,
            self.grid.var.dirupLen,
            self.grid.var.dirupID,
            self.grid.var.downstruct_no_water_bodies,
            _,
            self.grid.var.dirDown,
            self.grid.var.lendirDown,
        ) = define_river_network(ldd, self.model.data.grid)

        self.grid.var.upstream_area_n_cells = upstreamArea(
            self.grid.var.dirDown,
            dirshort,
            self.grid.full_compressed(1, dtype=np.int32),
        )
        self.grid.var.UpArea = upstreamArea(
            self.grid.var.dirDown,
            dirshort,
            self.grid.var.cellArea.astype(np.float64),
        )

        # number of substep per day
        self.var.noRoutingSteps = 24
        # kinematic wave parameter: 0.6 is for broad sheet flow

        self.var.beta = 0.6  # TODO: Make this a parameter

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

        # Channel bottom width [meters]
        self.grid.var.river_width = self.grid.load(
            self.model.files["grid"]["routing/river_width"]
        )

        # Corresponding sub-timestep (seconds)
        self.var.dtRouting = self.model.seconds_per_timestep / self.var.noRoutingSteps

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
        ) ** self.var.beta

        # Initialise water volume and discharge in rivers, just set at 0 [m3]
        self.grid.var.river_storage_m3 = np.zeros_like(
            self.grid.var.river_width, dtype=np.float32
        )
        self.grid.var.discharge = np.zeros_like(
            self.grid.var.river_width, dtype=np.float32
        )
        self.grid.var.discharge_substep = np.full(
            (self.var.noRoutingSteps, self.grid.var.discharge.size),
            0,
            dtype=self.grid.var.discharge.dtype,
        )

        # factor for evaporation from lakes, reservoirs and open channels
        self.grid.var.lakeEvaFactor = self.grid.full_compressed(
            self.model.config["parameters"]["lakeEvaFactor"], dtype=np.float32
        )

    def step(self, openWaterEvap, channel_abstraction_m, return_flow):
        """
        Dynamic part of the routing module

        * calculate evaporation from channels
        * calculate riverbed exchange between riverbed and groundwater
        * if option **waterbodies** is true, calculate retention from water bodies
        * calculate sideflow -> inflow to river
        * calculate kinematic wave -> using C++ library for computational speed
        """

        if __debug__:
            pre_channel_storage_m3 = self.grid.var.river_storage_m3.copy()
            pre_storage = self.model.lakes_reservoirs.var.storage.copy()

        # Evaporation from open channel
        # from big lakes/res and small lakes/res is calculated separately
        channel_ratio = get_channel_ratio(
            river_length=self.grid.var.river_length,
            river_width=self.grid.var.river_width,
            cell_area=self.grid.var.cellArea,
        )

        evaporation_in_channels_m3 = (
            self.grid.var.EWRef * channel_ratio * self.grid.var.cellArea
        )

        # limit evaporation to available water
        evaporation_in_channels_m3[
            evaporation_in_channels_m3 > self.grid.var.river_storage_m3
        ] = self.grid.var.river_storage_m3[
            evaporation_in_channels_m3 > self.grid.var.river_storage_m3
        ]

        # ensure that there is no evaporation in water bodies
        # if self.grid.var.river_storage_m3.sum() > 0:
        #     assert True

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
                                self.grid.var.kSatAquifer * self.grid.var.fracVegCover[5] * self.grid.var.cellArea, \
                                0.0), 0.0)))
        # to avoid flip flop
        self.grid.var.riverbedExchange = np.minimum(self.grid.var.riverbedExchange, 0.95 * self.grid.var.river_storage_m3)


                if self.grid.var.modflow:
            self.grid.var.interflow[No] = np.where(self.grid.var.capriseindex == 100, toGWorInterflow,
                                              self.grid.var.percolationImp * toGWorInterflow)
        else:
            self.grid.var.interflow[No] = self.grid.var.percolationImp * toGWorInterflow
        """

        # add reservoirs depending on year

        # ------------------------------------------------------------
        # evaporation from water bodies (m3), will be limited by available water in lakes and reservoirs
        # calculate outflow from lakes and reservoirs

        # average evaporation overeach lake
        average_evaporation_per_water_body = np.bincount(
            self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
            weights=self.grid.var.EWRef[self.grid.var.waterBodyID != -1],
        ) / np.bincount(self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1])
        evaporation_from_water_bodies_per_routing_step = (
            average_evaporation_per_water_body
            * self.model.lakes_reservoirs.var.lake_area
            / self.var.noRoutingSteps
        )
        assert np.all(evaporation_from_water_bodies_per_routing_step >= 0.0), (
            "evaporation_from_water_bodies_per_routing_step < 0.0"
        )

        # self.grid.var.riverbedExchange = np.where(self.grid.var.waterBodyID > 0, 0., self.grid.var.riverbedExchange)

        evaporation_in_channels_m3_per_routing_step = (
            evaporation_in_channels_m3 / self.var.noRoutingSteps
        )
        # riverbedExchangeDt = self.grid.var.riverbedExchangeM3 / self.var.noRoutingSteps
        # del self.grid.var.riverbedExchangeM3

        WDAddM3Dt = 0
        # WDAddM3Dt = self.grid.var.act_SurfaceWaterAbstract.copy() #MS CWatM edit Shouldn't this only be from the river abstractions? Currently includes the larger reservoir...
        WDAddMDt = channel_abstraction_m
        # return flow from (m) non irrigation water demand
        # WDAddM3Dt = WDAddM3Dt - self.grid.var.nonIrrreturn_flowFraction * self.grid.var.act_nonIrrDemand
        WDAddMDt = (
            WDAddMDt - return_flow
        )  # Couldn't this be negative? If return flow is mainly coming from gw? Fine, then more water going in.
        WDAddM3Dt = WDAddMDt * self.grid.var.cellArea / self.var.noRoutingSteps

        # ------------------------------------------------------
        # ***** SIDEFLOW **************************************

        runoffM3 = (
            self.grid.var.runoff * self.grid.var.cellArea / self.var.noRoutingSteps
        )

        self.grid.var.discharge_substep = np.full(
            (self.var.noRoutingSteps, self.grid.var.discharge.size),
            np.nan,
            dtype=self.grid.var.discharge.dtype,
        )

        sumsideflow_m3 = 0
        sumwaterbody_evaporation = 0
        discharge_at_outlets = 0
        for subrouting_step in range(self.var.noRoutingSteps):
            # Runoff - Evaporation ( -riverbed exchange), this could be negative  with riverbed exhange also
            sideflowChanM3 = runoffM3.copy()
            # minus evaporation from channels
            sideflowChanM3 -= evaporation_in_channels_m3_per_routing_step
            # # minus riverbed exchange
            # sideflowChanM3 -= riverbedExchangeDt

            sideflowChanM3 -= WDAddM3Dt
            # minus waterdemand + return_flow

            outflow_to_river_network, waterbody_evaporation = (
                self.model.lakes_reservoirs.routing(
                    subrouting_step,
                    evaporation_from_water_bodies_per_routing_step,
                    self.grid.var.discharge,
                    self.grid.var.runoff,
                )
            )
            sideflowChanM3 += outflow_to_river_network
            sumwaterbody_evaporation += waterbody_evaporation

            sideflowChan = (
                sideflowChanM3 / self.grid.var.river_length / self.var.dtRouting
            )

            self.grid.var.discharge = kinematic(
                self.grid.var.discharge,
                sideflowChan.astype(np.float32),
                self.grid.var.dirDown,
                self.grid.var.dirupLen,
                self.grid.var.dirupID,
                self.grid.var.river_alpha,
                self.var.beta,
                self.var.dtRouting,
                self.grid.var.river_length,
            )

            self.grid.var.discharge_substep[subrouting_step, :] = (
                self.grid.var.discharge.copy()
            )

            if __debug__:
                # Discharge at outlets and lakes and reservoirs
                discharge_at_outlets += self.grid.var.discharge[
                    self.grid.var.lddCompress_LR == PIT
                ].sum()

                sumsideflow_m3 += sideflowChanM3

        assert not np.isnan(self.grid.var.discharge).any()

        # The momentum equation, see eq. 18 in https://gmd.copernicus.org/articles/13/3267/2020/
        cross_sectional_area_of_flow = (
            self.grid.var.river_alpha * self.grid.var.discharge**self.var.beta
        )
        self.grid.var.river_storage_m3 = (
            cross_sectional_area_of_flow * self.grid.var.river_length
        )

        if __debug__:
            # this check the last routing step, but that's okay
            balance_check(
                how="sum",
                influxes=[runoffM3, outflow_to_river_network],
                outfluxes=[
                    sideflowChanM3,
                    evaporation_in_channels_m3_per_routing_step,
                    WDAddM3Dt,
                ],
                name="routing_1",
                tollerance=1e-8,
            )
            balance_check(
                how="sum",
                influxes=[
                    self.grid.var.runoff / self.var.noRoutingSteps,
                    outflow_to_river_network / self.grid.var.cellArea,
                ],
                outfluxes=[
                    sideflowChanM3 / self.grid.var.cellArea,
                    evaporation_in_channels_m3_per_routing_step
                    / self.grid.var.cellArea,
                    WDAddM3Dt / self.grid.var.cellArea,
                ],
                name="routing_2",
                tollerance=1e-8,
            )
            balance_check(
                how="sum",
                influxes=[sumsideflow_m3],
                outfluxes=[discharge_at_outlets * self.model.seconds_per_timestep],
                prestorages=[pre_channel_storage_m3],
                poststorages=[self.grid.var.river_storage_m3],
                name="routing_3",
                tollerance=100,
            )
            balance_check(
                how="sum",
                influxes=[self.grid.var.runoff * self.grid.var.cellArea],
                outfluxes=[
                    discharge_at_outlets * self.model.seconds_per_timestep,
                    evaporation_in_channels_m3,
                    sumwaterbody_evaporation,
                ],
                prestorages=[pre_channel_storage_m3, pre_storage],
                poststorages=[
                    self.grid.var.river_storage_m3,
                    self.model.lakes_reservoirs.var.storage,
                ],
                name="routing_4",
                tollerance=100,
            )
