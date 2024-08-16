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
    potTranspiration      Potential transpiration (after removing of evaporation)                           m
    actualET              simulated evapotranspiration from soil, flooded area and vegetation               m
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
            "forest": 0.001,
            "grassland": 0.001,
            "irrPaddy": 0.001,
            "irrNonPaddy": 0.001,
            "sealed": 0.001,
            "water": 0.0,
        }

        for coverNum, coverType in enumerate(self.model.coverTypes):
            coverType_indices = np.where(self.var.land_use_type == coverNum)
            self.var.minInterceptCap[coverType_indices] = minimum_intercept_capacity[
                coverType
            ]

        assert not np.isnan(self.var.interceptStor).any()
        assert not np.isnan(self.var.minInterceptCap).any()

        self.interception = {}
        for coverType in ("forest", "grassland"):
            ds = xr.open_dataset(
                self.model.model_structure["forcing"][
                    f"landcover/{coverType}/interceptCap{coverType.title()}_10days"
                ]
            )

            self.interception[coverType] = ds[
                f"interceptCap{coverType.title()}_10days"
            ].values
            ds.close()

    def step(self, potTranspiration):
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
        for coverNum, coverType in enumerate(self.model.coverTypes):
            coverType_indices = np.where(self.var.land_use_type == coverNum)
            if coverType in ("forest", "grassland"):
                interception_cover = self.model.data.to_HRU(
                    data=self.model.data.grid.compress(
                        self.interception[coverType][
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

        sealed_area = np.where(self.var.land_use_type == 4)
        water_area = np.where(self.var.land_use_type == 5)
        bio_area = np.where(
            self.var.land_use_type < 4
        )  # 'forest', 'grassland', 'irrPaddy', 'irrNonPaddy'

        self.var.interceptEvap = self.var.full_compressed(np.nan, dtype=np.float32)
        # interceptEvap evaporation from intercepted water (based on potTranspiration)
        self.var.interceptEvap[bio_area] = np.minimum(
            self.var.interceptStor[bio_area],
            potTranspiration[bio_area]
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

        # update interception storage and potTranspiration
        self.var.interceptStor = self.var.interceptStor - self.var.interceptEvap
        potTranspiration = np.maximum(0, potTranspiration - self.var.interceptEvap)

        # update actual evaporation (after interceptEvap)
        # interceptEvap is the first flux in ET, soil evapo and transpiration are added later
        self.var.actualET = self.var.interceptEvap + self.var.snowEvap

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

        assert not np.isnan(potTranspiration[bio_area]).any()
        return potTranspiration
