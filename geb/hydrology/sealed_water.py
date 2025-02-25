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
from geb.workflows import balance_check
from .landcover import SEALED, OPEN_WATER


class SealedWater(object):
    """
    Sealed and open water runoff

    calculated runoff from impermeable surface (sealed) and into water bodies


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    EWRef                 potential evaporation rate from water surface                                     m
    capillar              Simulated flow from groundwater to the third CWATM soil layer                     m
    availWaterInfiltrati  quantity of water reaching the soil after interception, more snowmelt             m
    actual_evapotranspiration              simulated evapotranspiration from soil, flooded area and vegetation               m
    direct_runoff          Simulated surface runoff                                                          m
    open_water_evaporation         Simulated evaporation from open areas                                             m
    actual_bare_soil_evaporation       Simulated evaporation from the first soil layer                                   m
    ====================  ================================================================================  =========
    """

    def __init__(self, model):
        self.HRU = model.data.HRU
        self.model = model

        if self.model.in_spinup:
            self.spinup()

    def spinup(self):
        pass

    def step(self, capillar, open_water_evaporation, direct_runoff):
        """
        Dynamic part of the sealed_water module

        runoff calculation for open water and sealed areas

        :param coverType: Land cover type: forest, grassland  ...
        :param No: number of land cover type: forest = 0, grassland = 1 ...
        """
        sealed_water_area = np.where(
            (self.HRU.var.land_use_type == SEALED) | self.HRU.var.land_use_type
            == OPEN_WATER
        )
        sealed_area = self.HRU.var.land_use_type == SEALED

        assert (capillar[sealed_water_area] >= 0).all()

        # evaporation from precipitation fallen on sealed area (ponds) estimated as 0.2 x EWRef
        # evaporation from open water and channels is calculated in the routing module
        open_water_evaporation[sealed_area] = 0.2 * self.HRU.var.EWRef[sealed_area]

        # as there is no interception on sealed areas, the available water is the sum of the natural available water and the capillar rise
        direct_runoff[sealed_water_area] = (
            self.HRU.var.natural_available_water_infiltration[sealed_water_area]
            + capillar[sealed_water_area]
        )

        # limit the evaporation to the available water
        open_water_evaporation[sealed_area] = np.minimum(
            open_water_evaporation[sealed_area], direct_runoff[sealed_area]
        )

        # subtract the evaporation from the runoff water
        direct_runoff[sealed_water_area] -= open_water_evaporation[sealed_water_area]

        # make sure that the runoff is still positive
        assert (direct_runoff[sealed_water_area] >= 0).all()

        # open water evaporation is directly substracted from the river, lakes, reservoir
        self.HRU.var.actual_evapotranspiration[sealed_water_area] = (
            self.HRU.var.actual_evapotranspiration[sealed_water_area]
            + open_water_evaporation[sealed_water_area]
        )

        if __debug__:
            balance_check(
                name="sealed_water",
                how="cellwise",
                influxes=[
                    self.HRU.var.natural_available_water_infiltration[
                        sealed_water_area
                    ],
                    capillar[sealed_water_area],
                ],
                outfluxes=[
                    direct_runoff[sealed_water_area],
                    open_water_evaporation[sealed_water_area],
                ],
                tollerance=1e-6,
            )

        return direct_runoff
