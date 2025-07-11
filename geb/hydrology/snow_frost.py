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
import numpy.typing as npt

from geb.module import Module
from geb.workflows import balance_check


class SnowFrost(Module):
    """RAIN AND SNOW.

    Domain: snow calculations evaluated for center points of up to 7 sub-pixel
    snow zones 1 -7 which each occupy a part of the pixel surface

    Variables *snow* and *rain* at end of this module are the pixel-average snowfall and rain


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    Tavg                  average air Temperature (input for the model)                                     K
    Rain                  Precipitation less snow                                                           m
    SnowMelt              total snow melt from all layers                                                   m
    SnowCover             snow cover (sum over all layers)                                                  m
    elevation_std
    Precipitation         Precipitation (input for the model)                                               m
    DtDay                 seconds in a timestep (default=86400)                                             s
    numberSnowLayersFloa
    numberSnowLayers      Number of snow layers (up to 10)                                                  --
    glaciertransportZone  Number of layers which can be mimiced as glacier transport zone                   --
    deltaInvNorm          Quantile of the normal distribution (for different numbers of snow layers)        --
    DeltaTSnow            Temperature lapse rate x std. deviation of elevation                              C°
    SnowDayDegrees        day of the year to degrees: 360/365.25 = 0.9856                                   --
    summerSeasonStart     day when summer season starts = 165                                               --
    IceDayDegrees         days of summer (15th June-15th Sept.) to degree: 180/(259-165)                    --
    SnowSeason            seasonal melt factor                                                              m C°-1 da
    TempSnow              Average temperature at which snow melts                                           C°
    SnowFactor            Multiplier applied to precipitation that falls as snow                            --
    SnowMeltCoef          Snow melt coefficient - default: 0.004                                            --
    IceMeltCoef           Ice melt coefficnet - default  0.007                                              --
    TempMelt              Average temperature at which snow melts                                           C°
    SnowCoverS            snow cover for each layer                                                         m
    Kfrost                Snow depth reduction coefficient, (HH, p. 7.28)                                   m-1
    Afrost                Daily decay coefficient, (Handbook of Hydrology, p. 7.28)                         --
    frost_indexThreshold   Degree Days Frost Threshold (stops infiltration, percolation and capillary rise)  --
    SnowWaterEquivalent   Snow water equivalent, (based on snow density of 450 kg/m3) (e.g. Tarboton and L  --
    frost_index            frost_index - Molnau and Bissel (1983), A Continuous Frozen Ground Index for Floo  --
    extfrost_index         Flag for second frost_index                                                        --
    frost_indexThreshold2  frost_index2 - Molnau and Bissel (1983), A Continuous Frozen Ground Index for Flo
    frostInd1             forstindex 1
    frostInd2             frost_index 2
    frost_indexS           array for frost_index
    prevSnowCover         snow cover of previous day (only for water balance)                               m
    Snow                  Snow (equal to a part of Precipitation)                                           m
    ====================  ================================================================================  =========
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
        return "hydrology.snowfrost"

    def spinup(self):
        self.var.numberSnowLayers = 3  # default 3
        self.var.glaciertransportZone = (
            1.0  # default 1 -> highest zone is transported to middle zone
        )

        # Difference between (average) air temperature at average elevation of
        # pixel and centers of upper- and lower elevation zones [deg C]
        # 0.9674:    Quantile of the normal distribution: u(0,833)=0.9674 to split the pixel in 3 equal parts.
        # for different number of layers
        #  Number: 2 ,3, 4, 5, 6, 7, ,8, 9, 10
        dn = {}
        dn[1] = np.array([0])
        dn[2] = np.array([-0.67448975, 0.67448975])
        dn[3] = np.array([-0.96742157, 0.0, 0.96742157])
        dn[5] = np.array([-1.28155157, -0.52440051, 0.0, 0.52440051, 1.28155157])
        dn[7] = np.array(
            [
                -1.46523379,
                -0.79163861,
                -0.36610636,
                0.0,
                0.36610636,
                0.79163861,
                1.46523379,
            ]
        )
        dn[9] = np.array(
            [
                -1.59321882,
                -0.96742157,
                -0.5894558,
                -0.28221615,
                0.0,
                0.28221615,
                0.5894558,
                0.96742157,
                1.59321882,
            ]
        )
        dn[10] = np.array(
            [
                -1.64485363,
                -1.03643339,
                -0.67448975,
                -0.38532047,
                -0.12566135,
                0.12566135,
                0.38532047,
                0.67448975,
                1.03643339,
                1.64485363,
            ]
        )

        # divNo = 1./float(self.var.numberSnowLayers)
        # deltaNorm = np.linspace(divNo/2, 1-divNo/2, self.var.numberSnowLayers)
        self.var.deltaInvNorm = dn[self.var.numberSnowLayers]

        TemperatureLapseRate: float = 0.0065

        elevation_std = self.grid.load(
            self.model.files["grid"]["landsurface/elevation_standard_deviation"]
        )
        elevation_std = self.hydrology.to_HRU(data=elevation_std, fn=None)

        self.HRU.var.DeltaTSnow = elevation_std * TemperatureLapseRate

        self.var.SnowDayDegrees: float = 0.9856
        # day of the year to degrees: 360/365.25 = 0.9856
        self.var.summerSeasonStart: int = 165
        # self.var.IceDayDegrees = 1.915
        self.var.IceDayDegrees: float = 180.0 / (259 - self.var.summerSeasonStart)
        # days of summer (15th June-15th Sept.) to degree: 180/(259-165)
        SnowSeasonAdj: float = 0.001
        self.var.SnowSeason: float = SnowSeasonAdj * 0.5
        # default value of range  of seasonal melt factor is set to 0.001 m C-1 day-1
        # 0.5 x range of sinus function [-1,1]
        self.var.TempSnow: float = 1.0
        self.var.SnowFactor: float = 1.0
        self.var.SnowMeltCoef: float = self.model.config["parameters"]["SnowMeltCoef"]
        self.var.IceMeltCoef: float = 0.007

        self.var.TempMelt: float = 1.0

        # initialize snowcovers as many as snow layers -> read them as SnowCover1 , SnowCover2 ...
        # SnowCover1 is the highest zone
        self.HRU.var.SnowCoverS = np.tile(
            self.hydrology.to_HRU(
                data=self.grid.full_compressed(0, dtype=np.float32),
                fn=None,
            ),
            (self.var.numberSnowLayers, 1),
        )
        # Pixel-average initial snow cover: average of values in 3 elevation
        # zones

        # ---------------------------------------------------------------------------------
        # Initial part of frost index

        self.var.Afrost = 0.97
        self.var.frost_indexThreshold = 56.0
        self.var.SnowWaterEquivalent = 0.45

        self.HRU.var.frost_index = self.HRU.full_compressed(0, dtype=np.float32)
        self.var.extfrost_index = False

    def step(
        self,
    ) -> tuple[
        npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
    ]:
        """Dynamic part of the snow module.

        Distinguish between rain/snow and calculates snow melt and glacier melt
        The equation is a modification of:

        References:
            Speers, D.D., Versteeg, J.D. (1979) Runoff forecasting for reservoir operations - the pastand the future. In: Proceedings 52nd Western Snow Conference, 149-156

        Frost index in soil [degree days] based on:

        References:
            Molnau and Bissel (1983, A Continuous Frozen Ground Index for Flood Forecasting. In: Maidment, Handbook of Hydrology, p. 7.28, 7.55)

        Todo:
            calculate sinus shape function for the southern hemisspere
        """
        if __debug__:
            self.HRU.var.prevSnowCover = self.HRU.var.SnowCoverS.copy()

        day_of_year: int = self.model.current_day_of_year
        SeasSnowMeltCoef = (
            self.var.SnowSeason
            * np.sin(math.radians((day_of_year - 81) * self.var.SnowDayDegrees))
            + self.var.SnowMeltCoef
        )

        # sinus shaped function between the
        # annual minimum (December 21st) and annual maximum (June 21st)
        # TODO change this for the southern hemisspere

        if (day_of_year > self.var.summerSeasonStart) and (day_of_year < 260):
            SummerSeason = np.sin(
                math.radians(
                    (day_of_year - self.var.summerSeasonStart) * self.var.IceDayDegrees
                )
            )
        else:
            SummerSeason = 0.0

        snow: npt.NDArray[np.float32] = self.HRU.full_compressed(0, dtype=np.float32)
        rain: npt.NDArray[np.float32] = self.HRU.full_compressed(0, dtype=np.float32)
        snow_melt: npt.NDArray[np.float32] = self.HRU.full_compressed(
            0, dtype=np.float32
        )

        tas_C: npt.NDArray[np.float32] = self.HRU.tas - 273.15
        precipitation_m_day: npt.NDArray[np.float32] = (
            0.001 * 86400.0 * self.HRU.pr
        )  # kg/m2/s to m/day

        for i in range(self.var.numberSnowLayers):
            TavgS = tas_C + self.HRU.var.DeltaTSnow * self.var.deltaInvNorm[i]
            # Temperature at center of each zone (temperature at zone B equals Tavg)
            # i=0 -> highest zone
            # i=2 -> lower zone
            SnowS = np.where(
                TavgS < self.var.TempSnow,
                self.var.SnowFactor * precipitation_m_day,
                self.HRU.full_compressed(0, dtype=np.float32),
            )
            # Precipitation is assumed to be snow if daily average temperature is below TempSnow
            # Snow is multiplied by correction factor to account for undercatch of
            # snow precipitation (which is common)
            RainS = np.where(
                TavgS >= self.var.TempSnow,
                precipitation_m_day,
                self.HRU.full_compressed(0, dtype=np.float32),
            )
            # if it's snowing then no rain
            # snowmelt coeff in m/deg C/day
            SnowMeltS = (
                (TavgS - self.var.TempMelt) * SeasSnowMeltCoef * (1 + 0.01 * RainS)
            )
            SnowMeltS = np.maximum(
                SnowMeltS, self.HRU.full_compressed(0, dtype=np.float32)
            )

            # for which layer the ice melt is calcultated with the middle temp.
            # for the others it is calculated with the corrected temp
            # this is to mimic glacier transport to lower zones
            if i <= self.var.glaciertransportZone:
                IceMeltS = tas_C * self.var.IceMeltCoef * SummerSeason
                # if i = 0 and 1 -> higher and middle zone
                # Ice melt coeff in m/C/deg
            else:
                IceMeltS = TavgS * self.var.IceMeltCoef * SummerSeason

            IceMeltS = np.maximum(
                IceMeltS, self.HRU.full_compressed(0, dtype=np.float32)
            )
            SnowMeltS = np.maximum(
                np.minimum(SnowMeltS + IceMeltS, self.HRU.var.SnowCoverS[i]),
                self.HRU.full_compressed(0, dtype=np.float32),
            )
            # check if snow+ice not bigger than snowcover
            self.HRU.var.SnowCoverS[i] = self.HRU.var.SnowCoverS[i] + SnowS - SnowMeltS
            snow += SnowS
            rain += RainS
            snow_melt += SnowMeltS

            if self.var.extfrost_index:
                Kfrost = np.where(TavgS < 0, 0.08, 0.5)
                frost_indexChangeRate = -(
                    1 - self.var.Afrost
                ) * self.HRU.var.frost_indexS[i] - TavgS * np.exp(
                    -0.4
                    * 100
                    * Kfrost
                    * np.minimum(
                        1.0,
                        self.HRU.var.SnowCoverS[i] / self.var.SnowWaterEquivalent,
                    )
                )
                self.HRU.var.frost_indexS[i] = np.maximum(
                    self.HRU.var.frost_indexS[i] + frost_indexChangeRate, 0
                )

        snow = snow / self.var.numberSnowLayers
        rain /= self.var.numberSnowLayers
        snow_melt /= self.var.numberSnowLayers

        if __debug__:
            balance_check(
                name="snow_1",
                how="cellwise",
                influxes=[snow],
                outfluxes=[snow_melt],
                prestorages=[
                    np.sum(self.HRU.var.prevSnowCover, axis=0)
                    / self.var.numberSnowLayers
                ],
                poststorages=[
                    np.sum(self.HRU.var.SnowCoverS, axis=0) / self.var.numberSnowLayers
                ],
                tollerance=1e-7,
            )
            balance_check(
                name="snow_2",
                how="cellwise",
                influxes=[precipitation_m_day],
                outfluxes=[snow, rain],
                tollerance=1e-7,
            )

        # Calculation of the frost index
        # frost index in soil [degree days] based on Molnau and Bissel (1983, A Continuous Frozen Ground Index for Flood
        # Forecasting. In: Maidment, Handbook of Hydrology, p. 7.28, 7.55)
        # if Tavg is above zero, frost_index will stay 0
        # if Tavg is negative, frost_index will increase with 1 per degree C per day
        # Exponent of 0.04 (instead of 0.4 in HoH): conversion [cm] to [mm]!  -> from cm to m HERE -> 100 * 0.4
        # maximum snowlayer = 1.0 m
        # Division by SnowDensity because SnowDepth is expressed as equivalent water depth(always less than depth of snow pack)
        # SnowWaterEquivalent taken as 0.45
        # Afrost, (daily decay coefficient) is taken as 0.97 (Handbook of Hydrology, p. 7.28)
        # Kfrost, (snow depth reduction coefficient) is taken as 0.57 [1/cm], (HH, p. 7.28) -> from Molnau taken as 0.5 for t> 0 and 0.08 for T<0
        Kfrost = np.where(tas_C < 0, 0.08, 0.5).astype(tas_C.dtype)
        frost_indexChangeRate = -(
            1 - self.var.Afrost
        ) * self.HRU.var.frost_index - tas_C * np.exp(
            -0.4
            * 100
            * Kfrost
            * np.minimum(
                1.0,
                (np.sum(self.HRU.var.SnowCoverS, axis=0) / self.var.numberSnowLayers)
                / self.var.SnowWaterEquivalent,
            )
        )
        # Rate of change of frost index (expressed as rate, [degree days/day])
        self.HRU.var.frost_index = np.maximum(
            self.HRU.var.frost_index + frost_indexChangeRate, 0
        )

        self.report(self, locals())

        return snow, rain, snow_melt
