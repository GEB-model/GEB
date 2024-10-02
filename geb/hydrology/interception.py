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
    interceptEvap         simulated evaporation from water intercepted by vegetation                        m
    potential_transpiration      Potential transpiration (after removing of evaporation)                           m
    actual_evapotranspiration              simulated evapotranspiration from soil, flooded area and vegetation               m
    snowEvap              total evaporation from snow for a snow layers                                     m
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.data.HRU
        self.model = model

        self.var.minInterceptCap = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.interceptStor = self.var.full_compressed(np.nan, dtype=np.float32)

        self.var.interceptStor = self.model.data.HRU.load_initial(
            "interceptStor",
            default=self.model.data.HRU.full_compressed(0, dtype=np.float32),
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
            coverType_indices = np.where(self.var.land_use_type == cover)
            self.var.minInterceptCap[coverType_indices] = minimum_intercept_capacity

        assert not np.isnan(self.var.interceptStor).any()
        assert not np.isnan(self.var.minInterceptCap).any()

        self.interception = {}
        for cover_name, cover in (("forest", FOREST), ("grassland", GRASSLAND_LIKE)):
            ds = xr.open_dataset(
                self.model.files["forcing"][
                    f"landcover/{cover_name}/interceptCap{cover_name.title()}_10days"
                ],
                engine="zarr",
            )

            self.interception[cover] = ds[
                f"interceptCap{cover_name.title()}_10days"
            ].values

    def step(self, potential_transpiration):
        """
        Dynamic part of the interception module
        calculating interception for each land cover class

        :param coverType: Land cover type: forest, grassland  ...
        :param No: number of land cover type: forest = 0, grassland = 1 ...
        :return: interception evaporation, interception storage, reduced pot. transpiration

        """

        if __debug__:
            interceptStor_pre = self.var.interceptStor.copy()

        interceptCap = self.var.full_compressed(np.nan, dtype=np.float32)
        for cover in ALL_LAND_COVER_TYPES:
            coverType_indices = np.where(self.var.land_use_type == cover)
            if cover in (FOREST, GRASSLAND_LIKE):
                interception_cover = self.model.data.to_HRU(
                    data=self.model.data.grid.compress(
                        self.interception[cover][
                            (self.model.current_day_of_year - 1) // 10
                        ]
                    ),
                    fn=None,
                )
                interceptCap[coverType_indices] = interception_cover[coverType_indices]
            else:
                interceptCap[coverType_indices] = self.var.minInterceptCap[
                    coverType_indices
                ]

        assert not np.isnan(interceptCap).any()

        # Rain instead Pr, because snow is substracted later
        # assuming that all interception storage is used the other time step
        throughfall = np.maximum(
            0.0, self.var.Rain + self.var.interceptStor - interceptCap
        )

        # update interception storage after throughfall
        self.var.interceptStor = self.var.interceptStor + self.var.Rain - throughfall

        # availWaterInfiltration Available water for infiltration: throughfall + snow melt
        self.var.natural_available_water_infiltration = np.maximum(
            0.0, throughfall + self.var.SnowMelt
        )

        sealed_area = np.where(self.var.land_use_type == SEALED)
        water_area = np.where(self.var.land_use_type == OPEN_WATER)
        bio_area = np.where(self.var.land_use_type < SEALED)

        self.var.interceptEvap = self.var.full_compressed(np.nan, dtype=np.float32)
        # interceptEvap evaporation from intercepted water (based on potential_transpiration)
        self.var.interceptEvap[bio_area] = np.minimum(
            self.var.interceptStor[bio_area],
            potential_transpiration[bio_area]
            * np.nan_to_num(
                self.var.interceptStor[bio_area] / interceptCap[bio_area], nan=0.0
            )
            ** (2.0 / 3.0),
        )

        self.var.interceptEvap[sealed_area] = np.maximum(
            np.minimum(
                self.var.interceptStor[sealed_area], self.var.EWRef[sealed_area]
            ),
            self.var.full_compressed(0, dtype=np.float32)[sealed_area],
        )

        self.var.interceptEvap[water_area] = 0  # never interception for water

        # update interception storage and potential_transpiration
        self.var.interceptStor = self.var.interceptStor - self.var.interceptEvap
        potential_transpiration = np.maximum(
            0, potential_transpiration - self.var.interceptEvap
        )

        # update actual evaporation (after interceptEvap)
        # interceptEvap is the first flux in ET, soil evapo and transpiration are added later
        self.var.actual_evapotranspiration = self.var.interceptEvap + self.var.snowEvap

        if __debug__:
            balance_check(
                name="interception",
                how="cellwise",
                influxes=[self.var.Rain, self.var.SnowMelt],  # In
                outfluxes=[
                    self.var.natural_available_water_infiltration,
                    self.var.interceptEvap,
                ],  # Out
                prestorages=[interceptStor_pre],  # prev storage
                poststorages=[self.var.interceptStor],
                tollerance=1e-7,
            )

        assert not np.isnan(potential_transpiration[bio_area]).any()
        return potential_transpiration
