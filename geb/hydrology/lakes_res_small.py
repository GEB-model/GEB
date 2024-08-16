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


class SmallLakesReservoirs(object):
    """
    Small LAKES AND RESERVOIRS

    Note:

        Calculate water retention in lakes and reservoirs

        Using the **Modified Puls approach** to calculate retention of a lake
        See also: LISFLOOD manual Annex 3 (Burek et al. 2013)


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    EWRef                 potential evaporation rate from water surface                                     m
    load_initial
    waterbalance_module
    seconds_per_timestep                 number of seconds per timestep (default = 86400)                                  s
    lakeEvaFactor         a factor which increases evaporation from lake because of wind                    --
    Invseconds_per_timestep
    runoff
    cellArea              Cell area [m²] of each simulated mesh
    smallpart
    smalllakeArea
    smalllakeDis0
    smalllakeA
    smalllakeFactor
    smalllakeFactorSqr
    smalllakeInflowOld
    smalllakeVolumeM3
    smalllakeOutflow
    smalllakeLevel
    smalllakeStorage
    minsmalllakeVolumeM3
    preSmalllakeStorage
    smallLakeIn
    smallevapWaterBody
    smallLakeout
    smallLakeDiff
    smallrunoffDiff
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        """
        Initialize small lakes and reservoirs
        Read parameters from maps e.g
        area, location, initial average discharge, type: reservoir or lake) etc.
        """
        self.var = model.data.grid
        self.model = model

        if self.model.useSmallLakes:

            if returnBool("useResAndLakes") and returnBool("dynamicLakesRes"):
                year = datetime.datetime(dateVar["currDate"].year, 1, 1)
            else:
                raise NotImplementedError

            # read which part of the cellarea is a lake/res catchment (sumed up for all lakes/res in a cell)
            self.var.smallpart = (
                readnetcdf2(
                    "smallLakesRes", year, useDaily="yearly", value="watershedarea"
                )
                * 1000
                * 1000
            )
            self.var.smallpart = self.var.smallpart / self.var.cellArea
            self.var.smallpart = np.minimum(1.0, self.var.smallpart)

            self.var.smalllakeArea = (
                readnetcdf2("smallLakesRes", year, useDaily="yearly", value="area")
                * 1000
                * 1000
            )

            # lake discharge at outlet to calculate alpha: parameter of channel width, gravity and weir coefficient
            # Lake parameter A (suggested  value equal to outflow width in [m])
            # multiplied with the calibration parameter LakeMultiplier
            testRunoff = "averageRunoff" in binding
            if testRunoff:
                self.var.smalllakeDis0 = (
                    loadmap("averageRunoff")
                    * self.var.smallpart
                    * self.var.cellArea
                    * self.model.Invseconds_per_timestep
                )
            else:
                self.var.smalllakeDis0 = loadmap("smallwaterBodyDis")
            self.var.smalllakeDis0 = np.maximum(self.var.smalllakeDis0, 0.01)
            chanwidth = 7.1 * np.power(self.var.smalllakeDis0, 0.539)
            self.var.smalllakeA = (
                loadmap("lakeAFactor") * 0.612 * 2 / 3 * chanwidth * (2 * 9.81) ** 0.5
            )
            self.var.smalllakeFactor = self.var.smalllakeArea / (
                self.model.seconds_per_timestep * np.sqrt(self.var.smalllakeA)
            )

            self.var.smalllakeFactorSqr = np.square(self.var.smalllakeFactor)
            # for faster calculation inside dynamic section

            self.var.smalllakeInflowOld = self.var.load_initial(
                "smalllakeInflow", self.var.smalllakeDis0
            )  # inflow in m3/s estimate

            old = self.var.smalllakeArea * np.sqrt(
                self.var.smalllakeInflowOld / self.var.smalllakeA
            )
            self.var.smalllakeVolumeM3 = self.var.load_initial("smalllakeStorage", old)

            smalllakeStorageIndicator = np.maximum(
                0.0,
                self.var.smalllakeVolumeM3 / self.model.seconds_per_timestep
                + self.var.smalllakeInflowOld / 2,
            )
            out = np.square(
                -self.var.smalllakeFactor
                + np.sqrt(self.var.smalllakeFactorSqr + 2 * smalllakeStorageIndicator)
            )
            # SI = S/dt + Q/2
            # solution of quadratic equation
            # 1. storage volume is increase proportional to elevation
            #  2. Q= a *H **2.0  (if you choose Q= a *H **1.5 you have to solve the formula of Cardano)
            self.var.smalllakeOutflow = self.var.load_initial("smalllakeOutflow", out)

            # lake storage ini
            self.var.smalllakeLevel = np.nan_to_num(
                self.var.smalllakeVolumeM3, self.var.smalllakeArea
            )

            self.var.smalllakeStorage = self.var.smalllakeVolumeM3.copy()

            testStorage = "minStorage" in binding
            if testStorage:
                self.var.minsmalllakeVolumeM3 = loadmap("minStorage")
            else:
                self.var.minsmalllakeVolumeM3 = 9.0e99

    def step(self):
        """
        Dynamic part to calculate outflow from small lakes and reservoirs

        * lakes with modified Puls approach
        * reservoirs with special filling levels

        **Flow out of lake:**

        :return: outflow in m3 to the network
        """

        def dynamic_smalllakes(inflow):
            """
            Lake routine to calculate lake outflow
            :param inflow: inflow to lakes and reservoirs
            :return: QLakeOutM3DtC - lake outflow in [m3] per subtime step
            """

            # ************************************************************
            # ***** LAKE
            # ************************************************************

            if __debug__:
                self.var.preSmalllakeStorage = self.var.smalllakeStorage.copy()

            # if (dateVar['curr'] == 998):
            #    ii = 1

            inflowM3S = inflow / self.model.seconds_per_timestep
            # Lake inflow in [m3/s]
            lakeIn = (inflowM3S + self.var.smalllakeInflowOld) * 0.5
            # for Modified Puls Method: (S2/dtime + Qout2/2) = (S1/dtime + Qout1/2) - Qout1 + (Qin1 + Qin2)/2
            # here: (Qin1 + Qin2)/2
            self.var.smallLakeIn = (
                lakeIn * self.model.seconds_per_timestep / self.var.cellArea
            )  # in [m]

            self.var.smallevapWaterBody = (
                self.var.lakeEvaFactor * self.var.EWRef * self.var.smalllakeArea
            )
            self.var.smallevapWaterBody = np.where(
                (self.var.smalllakeVolumeM3 - self.var.smallevapWaterBody) > 0.0,
                self.var.smallevapWaterBody,
                self.var.smalllakeVolumeM3,
            )
            self.var.smalllakeVolumeM3 = (
                self.var.smalllakeVolumeM3 - self.var.smallevapWaterBody
            )
            # lakestorage - evaporation from lakes

            self.var.smalllakeInflowOld = inflowM3S.copy()
            # Qin2 becomes Qin1 for the next time step

            lakeStorageIndicator = np.maximum(
                0.0,
                self.var.smalllakeVolumeM3 / self.model.seconds_per_timestep
                - 0.5 * self.var.smalllakeOutflow
                + lakeIn,
            )

            # here S1/dtime - Qout1/2 + lakeIn , so that is the right part
            # of the equation above

            self.var.smalllakeOutflow = np.square(
                -self.var.smalllakeFactor
                + np.sqrt(self.var.smalllakeFactorSqr + 2 * lakeStorageIndicator)
            )

            QsmallLakeOut = self.var.smalllakeOutflow * self.model.seconds_per_timestep

            self.var.smalllakeVolumeM3 = (
                lakeStorageIndicator - self.var.smalllakeOutflow * 0.5
            ) * self.model.seconds_per_timestep
            # Lake storage

            self.var.smalllakeStorage = (
                self.var.smalllakeStorage
                + lakeIn * self.model.seconds_per_timestep
                - QsmallLakeOut
                - self.var.smallevapWaterBody
            )
            # for mass balance, the lake storage is calculated every time step

            ### if dateVar['curr'] >= dateVar['intSpin']:
            ###   self.var.minsmalllakeStorageM3 = np.where(self.var.smalllakeStorageM3 < self.var.minsmalllakeStorageM3,self.var.smalllakeStorageM3,self.var.minsmalllakeStorageM3)

            self.var.smallevapWaterBody = (
                self.var.smallevapWaterBody / self.var.cellArea
            )  # back to [m]
            self.var.smalllakeLevel = np.nan_to_num(
                self.var.smalllakeVolumeM3, self.var.smalllakeArea
            )

            if __debug__:
                self.var.waterbalance_module.waterBalanceCheck(
                    [self.var.smallLakeIn],  # In
                    [
                        QsmallLakeOut / self.var.cellArea,
                        self.var.smallevapWaterBody,
                    ],  # Out
                    [self.var.preSmalllakeStorage / self.var.cellArea],  # prev storage
                    [self.var.smalllakeStorage / self.var.cellArea],
                    "smalllake",
                    False,
                )

            return QsmallLakeOut

        # ---------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------
        # Small lake and reservoirs

        if self.model.useSmallLakes:

            # check years
            if dateVar["newStart"] or dateVar["newYear"]:

                if returnBool("useResAndLakes") and returnBool("dynamicLakesRes"):
                    year = datetime.datetime(dateVar["currDate"].year, 1, 1)
                else:
                    year = datetime.datetime(int(binding["fixLakesResYear"]), 1, 1)
                self.var.smallpart = (
                    readnetcdf2(
                        "smallLakesRes", year, useDaily="yearly", value="watershedarea"
                    )
                    * 1000
                    * 1000
                )
                self.var.smallpart = self.var.smallpart / self.var.cellArea
                self.var.smallpart = np.minimum(1.0, self.var.smallpart)

                self.var.smalllakeArea = (
                    readnetcdf2("smallLakesRes", year, useDaily="yearly", value="area")
                    * 1000
                    * 1000
                )
                # mult with 1,000,000 to convert from km2 to m2

            # ----------
            # inflow lakes
            # 1.  dis = upstream1(self.var.downstruct_LR, self.var.discharge)   # from river upstream
            # 2.  runoff = npareatotal(self.var.waterBodyID, self.var.waterBodyID)  # from cell itself
            # 3.                  # outflow from upstream lakes

            # ----------

            # runoff to the lake as a part of the cell basin
            inflow = (
                self.var.smallpart * self.var.runoff * self.var.cellArea
            )  # inflow in m3
            self.var.smallLakeout = (
                dynamic_smalllakes(inflow) / self.var.cellArea
            )  # back to [m]

            self.var.smallLakeDiff = (
                self.var.smallpart * self.var.runoff - self.var.smallLakeIn
            )
            self.var.smallrunoffDiff = (
                self.var.smallpart * self.var.runoff - self.var.smallLakeout
            )

            self.var.runoff = (
                self.var.smallLakeout + (1 - self.var.smallpart) * self.var.runoff
            )  # back to [m]  # with and without in m3

            if __debug__:
                self.var.waterbalance_module.waterBalanceCheck(
                    [self.var.smallLakeIn],  # In
                    [self.var.smallLakeout, self.var.smallevapWaterBody],  # Out
                    [self.var.preSmalllakeStorage / self.var.cellArea],  # prev storage
                    [self.var.smalllakeStorage / self.var.cellArea],
                    "smalllake1",
                    False,
                )

            return
