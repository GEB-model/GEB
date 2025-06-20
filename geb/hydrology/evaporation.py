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

from geb.module import Module


class Evaporation(Module):
    """
    Evaporation module
    Calculate potential evaporation and pot. transpiration
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
        return "hydrology.evaporation"

    def spinup(self):
        pass

    def step(self, ETRef, snow_melt):
        """
        Dynamic part of the soil module

        calculating potential Evaporation for each land cover class with kc factor
        get crop coefficient, use potential ET, calculate potential bare soil evaporation and transpiration

        :param coverType: Land cover type: forest, grassland  ...
        :param No: number of land cover type: forest = 0, grassland = 1 ...
        :return: potential evaporation from bare soil, potential transpiration
        """

        # get crop coefficient
        # to get ETc from ET0 x kc factor  ((see http://www.fao.org/docrep/X0490E/x0490e04.htm#TopOfPage figure 4:)
        # crop coefficient read for forest and grassland from file

        # calculate potential bare soil evaporation
        potential_bare_soil_evaporation = (
            self.hydrology.crop_factor_calibration_factor * 0.2 * ETRef
        )

        # calculate snow evaporation
        self.HRU.var.snowEvap = np.minimum(snow_melt, potential_bare_soil_evaporation)
        snow_melt -= self.HRU.var.snowEvap
        potential_bare_soil_evaporation = (
            potential_bare_soil_evaporation - self.HRU.var.snowEvap
        )

        # calculate potential ET
        ##  self.HRU.var.potential_evapotranspiration total potential evapotranspiration for a reference crop for a land cover class [m]
        potential_evapotranspiration = (
            self.hydrology.crop_factor_calibration_factor * self.HRU.var.cropKC * ETRef
        )

        ## potential_transpiration: Transpiration for each land cover class
        potential_transpiration = np.maximum(
            0.0,
            potential_evapotranspiration
            - potential_bare_soil_evaporation
            - self.HRU.var.snowEvap,
        )

        self.report(self, locals())

        return (
            potential_transpiration,
            potential_bare_soil_evaporation,
            potential_evapotranspiration,
            snow_melt,
        )
