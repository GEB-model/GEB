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
import xarray as xr
from geb.workflows import balance_check

from .landcover import (
    FOREST,
    GRASSLAND_LIKE,
    PADDY_IRRIGATED,
    NON_PADDY_IRRIGATED,
    SEALED,
    OPEN_WATER,
    ALL_LAND_COVER_TYPES,
)


class Interception(object):
    """
    INTERCEPTION


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    EWRef                 potential evaporation rate from water surface                                     m
    waterbalance_module
    interceptCap          interception capacity of vegetation                                               m
    minInterceptCap       Maximum interception read from file for forest and grassland land cover           m
    interceptStor         simulated vegetation interception storage                                         m
    Rain                  Precipitation less snow                                                           m
    availWaterInfiltrati  quantity of water reaching the soil after interception, more snowmelt             m
    SnowMelt              total snow melt from all layers                                                   m
    interception_evaporation         simulated evaporation from water intercepted by vegetation                        m
    potential_transpiration      Potential transpiration (after removing of evaporation)                           m
    actual_evapotranspiration              simulated evapotranspiration from soil, flooded area and vegetation               m
    snowEvap              total evaporation from snow for a snow layers                                     m
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.HRU = model.data.HRU
        self.grid = model.data.grid
        self.model = model
        if self.model.in_spinup:
            self.spinup()

    def spinup(self):
        self.HRU.var.minInterceptCap = self.HRU.full_compressed(
            np.nan, dtype=np.float32
        )
        self.HRU.var.interceptStor = self.model.data.HRU.full_compressed(
            0, dtype=np.float32
        )

        minimum_intercept_capacity = {
            FOREST: 0.001,
            GRASSLAND_LIKE: 0.001,
            PADDY_IRRIGATED: 0.001,
            NON_PADDY_IRRIGATED: 0.001,
            SEALED: 0.001,
            OPEN_WATER: 0.0,
        }

        for cover, minimum_intercept_capacity in minimum_intercept_capacity.items():
            coverType_indices = np.where(self.HRU.var.land_use_type == cover)
            self.HRU.var.minInterceptCap[coverType_indices] = minimum_intercept_capacity

        assert not np.isnan(self.HRU.var.interceptStor).any()
        assert not np.isnan(self.HRU.var.minInterceptCap).any()

        self.grid.var.interception = np.tile(
            self.grid.full_compressed(np.nan, dtype=np.float32),
            (len(ALL_LAND_COVER_TYPES), 37, 1),
        )
        for cover_name, cover in (("forest", FOREST), ("grassland", GRASSLAND_LIKE)):
            ds = xr.open_dataset(
                self.model.files["forcing"][
                    f"landcover/{cover_name}/interceptCap{cover_name.title()}_10days"
                ],
                engine="zarr",
            )

            self.grid.var.interception[cover] = self.grid.compress(
                ds[f"interceptCap{cover_name.title()}_10days"].values
            )

    def step(self, potential_transpiration):
        """
        Dynamic part of the interception module
        calculating interception for each land cover class

        :param coverType: Land cover type: forest, grassland  ...
        :param No: number of land cover type: forest = 0, grassland = 1 ...
        :return: interception evaporation, interception storage, reduced pot. transpiration

        """

        if __debug__:
            interceptStor_pre = self.HRU.var.interceptStor.copy()

        interceptCap = self.HRU.full_compressed(np.nan, dtype=np.float32)
        for cover in ALL_LAND_COVER_TYPES:
            coverType_indices = np.where(self.HRU.var.land_use_type == cover)
            if cover in (FOREST, GRASSLAND_LIKE):
                interception_cover = self.model.data.to_HRU(
                    data=self.grid.var.interception[cover][
                        (self.model.current_day_of_year - 1) // 10
                    ],
                    fn=None,
                )
                interceptCap[coverType_indices] = interception_cover[coverType_indices]
            else:
                interceptCap[coverType_indices] = self.HRU.var.minInterceptCap[
                    coverType_indices
                ]

        assert not np.isnan(interceptCap).any()

        # Rain instead Pr, because snow is substracted later
        # assuming that all interception storage is used the other time step
        throughfall = np.maximum(
            0.0, self.HRU.var.Rain + self.HRU.var.interceptStor - interceptCap
        )

        # update interception storage after throughfall
        self.HRU.var.interceptStor = (
            self.HRU.var.interceptStor + self.HRU.var.Rain - throughfall
        )

        # availWaterInfiltration Available water for infiltration: throughfall + snow melt
        self.HRU.var.natural_available_water_infiltration = np.maximum(
            0.0, throughfall + self.HRU.var.SnowMelt
        )

        sealed_area = np.where(self.HRU.var.land_use_type == SEALED)
        water_area = np.where(self.HRU.var.land_use_type == OPEN_WATER)
        bio_area = np.where(self.HRU.var.land_use_type < SEALED)

        self.HRU.var.interception_evaporation = self.HRU.full_compressed(
            np.nan, dtype=np.float32
        )
        # interception_evaporation evaporation from intercepted water (based on potential_transpiration)
        self.HRU.var.interception_evaporation[bio_area] = np.minimum(
            self.HRU.var.interceptStor[bio_area],
            potential_transpiration[bio_area]
            * np.nan_to_num(
                self.HRU.var.interceptStor[bio_area] / interceptCap[bio_area],
                nan=0.0,
            )
            ** (2.0 / 3.0),
        )

        self.HRU.var.interception_evaporation[sealed_area] = np.maximum(
            np.minimum(
                self.HRU.var.interceptStor[sealed_area],
                self.HRU.var.EWRef[sealed_area],
            ),
            self.HRU.full_compressed(0, dtype=np.float32)[sealed_area],
        )

        self.HRU.var.interception_evaporation[water_area] = (
            0  # never interception for water
        )

        # update interception storage and potential_transpiration
        self.HRU.var.interceptStor = (
            self.HRU.var.interceptStor - self.HRU.var.interception_evaporation
        )
        potential_transpiration = np.maximum(
            0, potential_transpiration - self.HRU.var.interception_evaporation
        )

        # update actual evaporation (after interception_evaporation)
        # interception_evaporation is the first flux in ET, soil evapo and transpiration are added later
        self.HRU.var.actual_evapotranspiration = (
            self.HRU.var.interception_evaporation + self.HRU.var.snowEvap
        )

        if __debug__:
            balance_check(
                name="interception",
                how="cellwise",
                influxes=[self.HRU.var.Rain, self.HRU.var.SnowMelt],  # In
                outfluxes=[
                    self.HRU.var.natural_available_water_infiltration,
                    self.HRU.var.interception_evaporation,
                ],  # Out
                prestorages=[interceptStor_pre],  # prev storage
                poststorages=[self.HRU.var.interceptStor],
                tollerance=1e-7,
            )

        assert not np.isnan(potential_transpiration[bio_area]).any()
        return potential_transpiration
