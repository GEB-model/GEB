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
from ..landcover import OPEN_WATER
from .subroutines import PIT


class Routing(object):
    """
    ROUTING

    routing using the kinematic wave


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    EWRef                 potential evaporation rate from water surface                                     m
    waterbalance_module
    seconds_per_timestep                 number of seconds per timestep (default = 86400)                                  s
    waterBodyID           lakes/reservoirs map with a single ID for each lake/reservoir                     --
    UpArea1               upstream area of a grid cell                                                      m2
    dirUp                 river network in upstream direction                                               --
    lddCompress           compressed river network (without missing values)                                 --
    compress_LR           boolean map as mask map for compressing lake/reservoir                            --
    lakeArea              area of each lake/reservoir                                                       m2
    lakeEvaFactor         a factor which increases evaporation from lake because of wind                    --
    lakeEvaFactorC        compressed map of a factor which increases evaporation from lake because of wind  --
    EvapWaterBodyM
    lakeResInflowM
    lakeResOutflowM
    dtRouting             number of seconds per routing timestep                                            s
    lakeResStorage
    evaporation_from_water_bodies_per_routing_step
    sumLakeEvapWaterBody
    noRoutingSteps
    discharge             discharge                                                                         m3/s
    runoff
    cellArea              Cell area [m²] of each simulated mesh
    downstruct
    pre_storage
    act_SurfaceWaterAbst
    fracVegCover          Fraction of area covered by the corresponding landcover type
    openWaterEvap         Simulated evaporation from open areas                                             m
    chanLength
    totalCrossSectionAre
    dirupLen
    dirupID
    catchment
    dirDown
    lendirDown
    UpArea
    beta
    channel_mannings
    chanGrad
    channel_width
    invchanLength
    invdtRouting
    totalCrossSectionAre
    chanWettedPerimeterA
    alpPower
    channelAlpha
    invchannelAlpha
    channelStorageM3
    riverbedExchange
    pre_channel_storage_m3
    EvapoChannel
    QDelta
    act_bigLakeResAbst
    act_smallLakeResAbst
    return_flow
    sumsideflow
    inflowDt
    ====================  ================================================================================  =========

    **Functions**
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
            self.model.files["grid"]["routing/kinematic/ldd"],
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
        self.grid.var.channel_mannings = (
            self.grid.load(self.model.files["grid"]["routing/kinematic/mannings"])
            * self.model.config["parameters"]["manningsN"]
        )
        assert (self.grid.var.channel_mannings > 0).all()

        # Channel gradient (fraction, dy/dx)
        minimum_channel_gradient = 0.0001
        self.grid.var.chanGrad = np.maximum(
            self.grid.load(self.model.files["grid"]["routing/kinematic/channel_slope"]),
            minimum_channel_gradient,
        )
        # Channel length [meters]
        self.grid.var.chanLength = self.grid.load(
            self.model.files["grid"]["routing/kinematic/channel_length"]
        )
        assert (self.grid.var.chanLength[self.grid.var.lddCompress != PIT] > 0).all(), (
            "Channel length must be greater than 0 for all cells except for pits"
        )
        # Channel bottom width [meters]
        self.grid.var.channel_width = self.grid.load(
            self.model.files["grid"]["routing/kinematic/channel_width"]
        )

        # Inverse of channel length [1/m]
        self.grid.var.invchanLength = 1 / self.grid.var.chanLength

        # Corresponding sub-timestep (seconds)
        self.var.dtRouting = self.model.seconds_per_timestep / self.var.noRoutingSteps
        self.var.invdtRouting = 1 / self.var.dtRouting

        # for a channel, the wetted perimeter can be approximated by the channel width
        channel_wetted_perimeter = self.grid.var.channel_width

        # ChannelAlpha for kinematic wave
        # source: https://gmd.copernicus.org/articles/13/3267/2020/ eq. 21
        self.grid.var.channelAlpha = (
            self.grid.var.channel_mannings
            * channel_wetted_perimeter ** (2 / 3)
            / np.sqrt(self.grid.var.chanGrad)
        ) ** self.var.beta

        # channel water volume [m3]
        # Initialise water volume in kinematic wave channels [m3]
        self.grid.var.channelStorageM3 = np.zeros_like(
            self.grid.var.channel_width, dtype=np.float32
        )
        self.grid.var.discharge = (
            self.grid.var.channelStorageM3
            * self.grid.var.invchanLength
            / self.grid.var.channelAlpha
        ) ** (1 / self.var.beta)
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
            pre_channel_storage_m3 = self.grid.var.channelStorageM3.copy()
            pre_storage = self.model.lakes_reservoirs.var.storage.copy()

        # Evaporation from open channel
        # from big lakes/res and small lakes/res is calculated separately
        channelFraction = np.minimum(
            1.0,
            self.grid.var.channel_width
            * self.grid.var.chanLength
            / self.grid.var.cellArea,
        )
        # put all the water area in which is not reflected in the lakes ,res
        # channelFraction = np.maximum(self.grid.var.fracVegCover[5], channelFraction)

        EWRefact = self.grid.var.lakeEvaFactor * self.model.data.to_grid(
            HRU_data=self.model.data.HRU.var.EWRef, fn="weightedmean"
        ) - self.model.data.to_grid(HRU_data=openWaterEvap, fn="weightedmean")
        # evaporation from channel minus the calculated evaporation from rainfall
        EvapoChannel = EWRefact * channelFraction * self.grid.var.cellArea
        # EvapoChannel = self.grid.var.EWRef * channelFraction * self.grid.var.cellArea

        # restrict to 95% of channel storage -> something should stay in the river
        EvapoChannel = np.where(
            (0.95 * self.grid.var.channelStorageM3 - EvapoChannel) > 0.0,
            EvapoChannel,
            0.95 * self.grid.var.channelStorageM3,
        )

        # riverbed infiltration (m3):
        # - current implementation based on Inge's principle (later, will be based on groundater head (MODFLOW) and can be negative)
        # - happening only if 0.0 < baseflow < nonFossilGroundwaterAbs
        # - infiltration rate will be based on aquifer saturated conductivity
        # - limited to fracWat
        # - limited to available channelStorageM3
        # - this infiltration will be handed to groundwater in the next time step

        """
        self.grid.var.riverbedExchange = np.maximum(0.0,  np.minimum(self.grid.var.channelStorageM3, np.where(self.grid.var.baseflow > 0.0, \
                                np.where(self.grid.var.nonFossilGroundwaterAbs > self.grid.var.baseflow, \
                                self.grid.var.kSatAquifer * self.grid.var.fracVegCover[5] * self.grid.var.cellArea, \
                                0.0), 0.0)))
        # to avoid flip flop
        self.grid.var.riverbedExchange = np.minimum(self.grid.var.riverbedExchange, 0.95 * self.grid.var.channelStorageM3)


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
        EWRefavg = np.bincount(
            self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
            weights=EWRefact[self.grid.var.waterBodyID != -1],
        ) / np.bincount(self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1])
        evaporation_from_water_bodies_per_routing_step = (
            EWRefavg
            * self.model.lakes_reservoirs.var.lake_area
            / self.var.noRoutingSteps
        )
        assert np.all(evaporation_from_water_bodies_per_routing_step >= 0.0), (
            "evaporation_from_water_bodies_per_routing_step < 0.0"
        )

        fraction_water = np.array(self.model.data.HRU.var.land_use_ratio)
        fraction_water[self.model.data.HRU.var.land_use_type != OPEN_WATER] = 0
        fraction_water = self.model.data.to_grid(HRU_data=fraction_water, fn="sum")

        EvapoChannel = np.where(
            self.grid.var.waterBodyID > 0,
            (1 - fraction_water) * EvapoChannel,
            EvapoChannel,
        )
        # self.grid.var.riverbedExchange = np.where(self.grid.var.waterBodyID > 0, 0., self.grid.var.riverbedExchange)

        EvapoChannelM3Dt = EvapoChannel / self.var.noRoutingSteps
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

        sumsideflow = 0
        sumwaterbody_evaporation = 0
        discharge_at_outlets = 0
        for subrouting_step in range(self.var.noRoutingSteps):
            # Runoff - Evaporation ( -riverbed exchange), this could be negative  with riverbed exhange also
            sideflowChanM3 = runoffM3.copy()
            # minus evaporation from channels
            sideflowChanM3 -= EvapoChannelM3Dt
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
                sideflowChanM3 * self.grid.var.invchanLength / self.var.dtRouting
            )

            self.grid.var.discharge = kinematic(
                self.grid.var.discharge,
                sideflowChan.astype(np.float32),
                self.grid.var.dirDown,
                self.grid.var.dirupLen,
                self.grid.var.dirupID,
                self.grid.var.channelAlpha,
                self.var.beta,
                self.var.dtRouting,
                self.grid.var.chanLength,
            )

            self.grid.var.discharge_substep[subrouting_step, :] = (
                self.grid.var.discharge.copy()
            )

            if __debug__:
                # Discharge at outlets and lakes and reservoirs
                discharge_at_outlets += self.grid.var.discharge[
                    self.grid.var.lddCompress_LR == PIT
                ].sum()

                sumsideflow += sideflowChanM3

        assert not np.isnan(self.grid.var.discharge).any()

        self.grid.var.channelStorageM3 = (
            self.grid.var.channelAlpha
            * self.grid.var.chanLength
            * self.grid.var.discharge**self.var.beta
        )

        if __debug__:
            # this check the last routing step, but that's okay
            balance_check(
                how="sum",
                influxes=[runoffM3, outflow_to_river_network],
                outfluxes=[sideflowChanM3, EvapoChannelM3Dt, WDAddM3Dt],
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
                    EvapoChannelM3Dt / self.grid.var.cellArea,
                    WDAddM3Dt / self.grid.var.cellArea,
                ],
                name="routing_2",
                tollerance=1e-8,
            )
            balance_check(
                how="sum",
                influxes=[sumsideflow],
                outfluxes=[discharge_at_outlets * self.model.seconds_per_timestep],
                prestorages=[pre_channel_storage_m3],
                poststorages=[self.grid.var.channelStorageM3],
                name="routing_3",
                tollerance=100,
            )
            balance_check(
                how="sum",
                influxes=[self.grid.var.runoff * self.grid.var.cellArea],
                outfluxes=[
                    discharge_at_outlets * self.model.seconds_per_timestep,
                    EvapoChannel,
                    sumwaterbody_evaporation,
                ],
                prestorages=[pre_channel_storage_m3, pre_storage],
                poststorages=[
                    self.grid.var.channelStorageM3,
                    self.model.lakes_reservoirs.var.storage,
                ],
                name="routing_4",
                tollerance=100,
            )
