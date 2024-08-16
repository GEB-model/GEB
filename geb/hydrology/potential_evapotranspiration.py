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
from numba import njit

from geb.workflows import TimingModule


@njit(cache=True)
def get_saturation_vapour_pressure(tasmin_C, tasmax_C):
    # http://www.fao.org/docrep/X0490E/x0490e07.htm   equation 11/12
    ESatmin = np.float32(0.6108) * np.exp(
        (np.float32(17.27) * tasmin_C) / (tasmin_C + np.float32(237.3))
    )
    ESatmax = np.float32(0.6108) * np.exp(
        (np.float32(17.27) * tasmax_C) / (tasmax_C + np.float32(237.3))
    )
    saturated_vapour_pressure = (ESatmin + ESatmax) / np.float32(2.0)  # [KPa]
    return saturated_vapour_pressure


class PotentialEvapotranspiration(object):
    """
    POTENTIAL REFERENCE EVAPO(TRANSPI)RATION
    Calculate potential evapotranspiration from climate data mainly based on FAO 56 and LISVAP
    Based on Penman Monteith

    References:
        http://www.fao.org/docrep/X0490E/x0490e08.htm#penman%20monteith%20equation
        http://www.fao.org/docrep/X0490E/x0490e06.htm  http://www.fao.org/docrep/X0490E/x0490e06.htm
        https://ec.europa.eu/jrc/en/publication/eur-scientific-and-technical-research-reports/lisvap-evaporation-pre-processor-lisflood-water-balance-and-flood-simulation-model

    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    pet_modus             Flag: index which ETP approach is used e.g. 1 for Penman-Monteith                 --
    AlbedoCanopy          Albedo of vegetation canopy (FAO,1998) default = 0.23                              --
    AlbedoSoil            Albedo of bare soil surface (Supit et. al. 1994) default = 0.15                   --
    AlbedoWater           Albedo of water surface (Supit et. al. 1994) default = 0.05                       --
    co2
    TMin                  minimum air temperature                                                           K
    TMax                  maximum air temperature                                                           K
    Psurf                 Instantaneous surface pressure                                                    Pa
    Qair                  specific humidity                                                                 kg/kg
    Tavg                  average air Temperature (input for the model)                                     K
    Rsdl                  long wave downward surface radiation fluxes                                       W/m2
    albedoLand            albedo from land surface (from GlobAlbedo database)                               --
    albedoOpenWater       albedo from open water surface (from GlobAlbedo database)                         --
    Rsds                  short wave downward surface radiation fluxes                                      W/m2
    Wind                  wind speed                                                                        m/s
    ETRef                 potential evapotranspiration rate from reference crop                             m
    EWRef                 potential evaporation rate from water surface                                     m
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        """
        The constructor evaporationPot
        """
        self.var = model.data.HRU
        self.model = model

        self.var.AlbedoCanopy = 0.23
        self.var.AlbedoSoil = 0.15
        self.var.AlbedoWater = 0.05

    def step(self):
        """
        Dynamic part of the potential evaporation module
        Based on Penman Monteith - FAO 56

        """
        timer = TimingModule("Potential evaporation")

        timer.new_split("Start")

        tas_C = self.var.tas - 273.15
        tasmin_C = self.var.tasmin - 273.15
        tasmax_C = self.var.tasmax - 273.15
        rlds = self.var.rlds
        rsds = self.var.rsds
        hurs = self.var.hurs
        ps = self.var.ps
        sfcWind = self.var.sfcWind

        timer.new_split("Read data")

        saturated_vapour_pressure = get_saturation_vapour_pressure(tasmin_C, tasmax_C)
        actual_vapour_pressure = saturated_vapour_pressure * hurs / 100.0

        timer.new_split("Saturation vapour pressure")

        ps_kPa = ps * 0.001
        psychrometric_constant = 0.665e-3 * ps_kPa
        # psychrometric constant [kPa C-1]
        # http://www.fao.org/docrep/X0490E/x0490e07.htm  Equation 8
        # see http://www.fao.org/docrep/X0490E/x0490e08.htm#penman%20monteith%20equation

        timer.new_split("Psychrometric constant")

        # calculate vapor pressure
        # Fao 56 Page 36
        # calculate actual vapour pressure

        # longwave radiation balance

        # Up longwave radiation [MJ/m2/day]
        rlus_MJ_m2_day = (
            4.903e-9 * (((tasmin_C + 273.16) ** 4) + ((tasmax_C + 273.16) ** 4)) / 2
        )  # rlus = Surface Upwelling Longwave Radiation

        rlds_MJ_m2_day = rlds * 0.0864  # 86400 * 1E-6
        net_longwave_radation_MJ_m2_day = rlus_MJ_m2_day - rlds_MJ_m2_day

        # ************************************************************
        # ***** NET ABSORBED RADIATION *******************************
        # ************************************************************

        rsds_MJ_m2_day = rsds * 0.0864  # 86400 * 1E-6
        # net absorbed radiation of reference vegetation canopy [mm/d]
        RNA = np.maximum(
            (1 - self.var.AlbedoCanopy) * rsds_MJ_m2_day
            - net_longwave_radation_MJ_m2_day,
            0.0,
        )
        # net absorbed radiation of bare soil surface
        RNAWater = np.maximum(
            (1 - self.var.AlbedoWater) * rsds_MJ_m2_day
            - net_longwave_radation_MJ_m2_day,
            0.0,
        )
        # net absorbed radiation of water surface

        timer.new_split("Radiation")

        vapour_pressure_deficit = np.maximum(
            saturated_vapour_pressure - actual_vapour_pressure, 0.0
        )
        slope_of_saturated_vapour_pressure_curve = (
            4098.0 * saturated_vapour_pressure
        ) / ((tas_C + 237.3) ** 2)
        # slope of saturated vapour pressure curve [kPa/deg C]
        # Equation 13 Chapter 3

        timer.new_split("Vapour pressure")

        # Chapter 2 Equation 6
        # Adjust wind speed for measurement height: wind speed measured at
        # 10 m, but needed at 2 m height
        # Shuttleworth, W.J. (1993) in Maidment, D.R. (1993), p. 4.36
        wind_2m = sfcWind * 0.749

        # TODO: update this properly following PCR-GLOBWB (https://github.com/UU-Hydro/PCR-GLOBWB_model/blob/0511485ad3ac0a1367d9d4918d2f61ae0fa0e900/model/evaporation/ref_pot_et_penman_monteith.py#L227)

        denominator = (
            slope_of_saturated_vapour_pressure_curve
            + psychrometric_constant * (1 + 0.34 * wind_2m)
        )

        # TODO: check if this is correct. Specifically the replacement 0.408 constant with 1 / LatHeatVap. In the original code
        # it seems that the latent heat is only applied to the first nominator and not to the second one, see: https://en.wikipedia.org/wiki/Penman%E2%80%93Monteith_equation

        # latent heat of vaporization [MJ/kg]
        LatHeatVap = 2.501 - 0.002361 * tas_C
        # the 0.408 constant is replace by 1/LatHeatVap
        RNAN = RNA / LatHeatVap * slope_of_saturated_vapour_pressure_curve / denominator
        RNANWater = (
            RNAWater
            / LatHeatVap
            * slope_of_saturated_vapour_pressure_curve
            / denominator
        )

        EA = (
            psychrometric_constant
            * 900
            / (tas_C + 273.16)
            * wind_2m
            * vapour_pressure_deficit
            / denominator
        )

        timer.new_split("Penman-Monteith")

        # Potential evapo(transpi)ration is calculated for two reference surfaces:
        # 1. Reference vegetation canopy (ETRef)
        # 2. Open water surface (EWRef)

        self.var.ETRef = (RNAN + EA) * 0.001
        # potential reference evapotranspiration rate [m/day]  # from mm to m with 0.001
        # potential evaporation rate from a bare soil surface [m/day]

        self.var.EWRef = (RNANWater + EA) * 0.001
        # potential evaporation rate from water surface [m/day]

        # -> here we are at ET0 (see http://www.fao.org/docrep/X0490E/x0490e04.htm#TopOfPage figure 4:)

        timer.new_split("ETRef and EWRef")

        if self.model.timing:
            print(timer)

        self.model.agents.crop_farmers.save_water_deficit()
