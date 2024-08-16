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
    actualET              simulated evapotranspiration from soil, flooded area and vegetation               m
    directRunoff          Simulated surface runoff                                                          m
    openWaterEvap         Simulated evaporation from open areas                                             m
    actTransTotal         Total actual transpiration from the three soil layers                             m
    actBareSoilEvap       Simulated evaporation from the first soil layer                                   m
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
        mult[self.var.land_use_type == 5] = 1
        mult[self.var.land_use_type == 4] = 0.2

        sealed_area = np.where(
            (self.var.land_use_type == 4) | self.var.land_use_type == 5
        )

        # GW capillary rise in sealed area is added to the runoff
        openWaterEvap[sealed_area] = np.minimum(
            mult[sealed_area] * self.var.EWRef[sealed_area],
            self.var.natural_available_water_infiltration[sealed_area]
            + capillar[sealed_area],
        )
        directRunoff[sealed_area] = (
            self.var.natural_available_water_infiltration[sealed_area]
            - openWaterEvap[sealed_area]
            + capillar[sealed_area]
        )

        # open water evaporation is directly substracted from the river, lakes, reservoir
        self.var.actualET[sealed_area] = (
            self.var.actualET[sealed_area] + openWaterEvap[sealed_area]
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
                    self.var.actTransTotal[sealed_area],
                    self.var.actBareSoilEvap[sealed_area],
                    openWaterEvap[sealed_area],
                ],
                tollerance=1e-6,
            )

        return directRunoff
