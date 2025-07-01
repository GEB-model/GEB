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
import numpy.typing as npt
import zarr

from geb.module import Module
from geb.workflows import balance_check

from .landcover import (
    ALL_LAND_COVER_TYPES,
    FOREST,
    GRASSLAND_LIKE,
    NON_PADDY_IRRIGATED,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
)


class Interception(Module):
    """INTERCEPTION.

    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    EWRef                 potential evaporation rate from water surface                                     m
    interceptCap          interception capacity of vegetation                                               m
    minInterceptCap       Maximum interception read from file for forest and grassland land cover           m
    interception_storage         simulated vegetation interception storage                                         m
    Rain                  Precipitation less snow                                                           m
    availWaterInfiltrati  quantity of water reaching the soil after interception, more snowmelt             m
    SnowMelt              total snow melt from all layers                                                   m
    interception_evaporation         simulated evaporation from water intercepted by vegetation                        m
    potential_transpiration      Potential transpiration (after removing of evaporation)                           m
    snowEvap              total evaporation from snow for a snow layers                                     m
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
        return "hydrology.interception"

    def spinup(self):
        self.HRU.var.minInterceptCap = self.HRU.full_compressed(
            np.nan, dtype=np.float32
        )
        self.HRU.var.interception_storage = self.hydrology.HRU.full_compressed(
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

        assert not np.isnan(self.HRU.var.interception_storage).any()
        assert not np.isnan(self.HRU.var.minInterceptCap).any()

        self.grid.var.interception = np.tile(
            self.grid.full_compressed(np.nan, dtype=np.float32),
            (len(ALL_LAND_COVER_TYPES), 37, 1),
        )
        for cover_name, cover in (("forest", FOREST), ("grassland", GRASSLAND_LIKE)):
            store = zarr.storage.LocalStore(
                self.model.files["grid"][
                    f"landcover/{cover_name}/interception_capacity"
                ],
                read_only=True,
            )

            self.grid.var.interception[cover] = self.grid.compress(
                zarr.open_group(store, mode="r")["interception_capacity"][:]
            )

    def step(
        self,
        potential_transpiration: npt.NDArray[np.float32],
        rain: npt.NDArray[np.float32],
        snow_melt: npt.NDArray[np.float32],
    ):
        if __debug__:
            interception_storage_pre: npt.NDArray[np.float32] = (
                self.HRU.var.interception_storage.copy()
            )

        interceptCap: npt.NDArray[np.float32] = self.HRU.full_compressed(
            np.nan, dtype=np.float32
        )
        for cover in ALL_LAND_COVER_TYPES:
            coverType_indices = np.where(self.HRU.var.land_use_type == cover)
            if cover in (FOREST, GRASSLAND_LIKE):
                interception_cover = self.hydrology.to_HRU(
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
        throughfall: npt.NDArray[np.float32] = np.maximum(
            0.0, rain + self.HRU.var.interception_storage - interceptCap
        )

        # update interception storage after throughfall
        self.HRU.var.interception_storage: npt.NDArray[np.float32] = (
            self.HRU.var.interception_storage + rain - throughfall
        )

        # availWaterInfiltration Available water for infiltration: throughfall + snow melt
        self.HRU.var.natural_available_water_infiltration: npt.NDArray[np.float32] = (
            np.maximum(0.0, throughfall + snow_melt)
        )

        sealed_area: npt.NDArray[np.int64] = np.where(
            self.HRU.var.land_use_type == SEALED
        )[0]
        water_area: npt.NDArray[np.int64] = np.where(
            self.HRU.var.land_use_type == OPEN_WATER
        )[0]
        bio_area: npt.NDArray[np.int64] = np.where(self.HRU.var.land_use_type < SEALED)[
            0
        ]

        interception_evaporation = self.HRU.full_compressed(0, dtype=np.float32)
        # interception_evaporation evaporation from intercepted water (based on potential_transpiration)
        interception_evaporation[bio_area] = np.minimum(
            self.HRU.var.interception_storage[bio_area],
            potential_transpiration[bio_area]
            * np.nan_to_num(
                self.HRU.var.interception_storage[bio_area] / interceptCap[bio_area],
                nan=0.0,
            )
            ** (2.0 / 3.0),
        )

        interception_evaporation[sealed_area] = np.maximum(
            np.minimum(
                self.HRU.var.interception_storage[sealed_area],
                self.HRU.var.EWRef[sealed_area],
            ),
            self.HRU.full_compressed(0, dtype=np.float32)[sealed_area],
        )

        interception_evaporation[water_area] = 0  # never interception for water

        # update interception storage and potential_transpiration
        self.HRU.var.interception_storage = (
            self.HRU.var.interception_storage - interception_evaporation
        )
        potential_transpiration = np.maximum(
            0, potential_transpiration - interception_evaporation
        )

        if __debug__:
            balance_check(
                name="interception",
                how="cellwise",
                influxes=[rain, snow_melt],  # In
                outfluxes=[
                    self.HRU.var.natural_available_water_infiltration,
                    interception_evaporation,
                ],  # Out
                prestorages=[interception_storage_pre],  # prev storage
                poststorages=[self.HRU.var.interception_storage],
                tollerance=1e-7,
            )

        assert not np.isnan(potential_transpiration[bio_area]).any()

        self.report(self, locals())
        return potential_transpiration, interception_evaporation
