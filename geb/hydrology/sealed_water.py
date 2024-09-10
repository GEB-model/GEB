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
    waterbalance_module
    availWaterInfiltrati  quantity of water reaching the soil after interception, more snowmelt             m
    actual_evapotranspiration              simulated evapotranspiration from soil, flooded area and vegetation               m
    directRunoff          Simulated surface runoff                                                          m
    openWaterEvap         Simulated evaporation from open areas                                             m
    actual_bare_soil_evaporation       Simulated evaporation from the first soil layer                                   m
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.data.HRU
        self.model = model

    def step(self, capillar, openWaterEvap, directRunoff):
        """
        Dynamic part of the sealed_water module

        runoff calculation for open water and sealed areas

        :param coverType: Land cover type: forest, grassland  ...
        :param No: number of land cover type: forest = 0, grassland = 1 ...
        """

        mult = self.var.full_compressed(0, dtype=np.float32)
        mult[self.var.land_use_type == OPEN_WATER] = 1
        mult[self.var.land_use_type == SEALED] = 0.2

        sealed_area = np.where(
            (self.var.land_use_type == SEALED) | self.var.land_use_type == OPEN_WATER
        )

        assert (capillar[sealed_area] >= 0).all()

        openWaterEvap[sealed_area] = mult[sealed_area] * self.var.EWRef[sealed_area]

        # as there is no interception on sealed areas, the available water is the sum of the natural available water and the capillar rise
        directRunoff[sealed_area] = (
            self.var.natural_available_water_infiltration[sealed_area]
            + capillar[sealed_area]
        )
        # limit the evaporation to the available water
        openWaterEvap[sealed_area] = np.minimum(
            openWaterEvap[sealed_area], directRunoff[sealed_area]
        )

        # subtract the evaporation from the runoff water
        directRunoff[sealed_area] -= openWaterEvap[sealed_area]

        # make sure that the runoff is still positive
        assert (directRunoff[sealed_area] >= 0).all()

        # open water evaporation is directly substracted from the river, lakes, reservoir
        self.var.actual_evapotranspiration[sealed_area] = (
            self.var.actual_evapotranspiration[sealed_area] + openWaterEvap[sealed_area]
        )

        if __debug__:
            balance_check(
                name="sealed_water",
                how="cellwise",
                influxes=[
                    self.var.natural_available_water_infiltration[sealed_area],
                    capillar[sealed_area],
                ],
                outfluxes=[
                    directRunoff[sealed_area],
                    openWaterEvap[sealed_area],
                ],
                tollerance=1e-6,
            )

        return directRunoff
