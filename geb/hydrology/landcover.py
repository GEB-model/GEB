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

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass
from numba import njit
from geb.workflows import TimingModule, balance_check

# All natural areas MUST be before the sealed and water areas
FOREST = 0
GRASSLAND_LIKE = 1
PADDY_IRRIGATED = 2
NON_PADDY_IRRIGATED = 3
SEALED = 4
OPEN_WATER = 5

ALL_LAND_COVER_TYPES = [
    FOREST,
    GRASSLAND_LIKE,
    PADDY_IRRIGATED,
    NON_PADDY_IRRIGATED,
    SEALED,
    OPEN_WATER,
]


@njit(cache=True)
def get_crop_kc_and_root_depths(
    crop_map,
    crop_age_days_map,
    crop_harvest_age_days,
    irrigated_fields,
    crop_stage_data,
    kc_crop_stage,
    rooth_depths,
    init_root_depth=0.2,
):
    kc = np.full_like(crop_map, np.nan, dtype=np.float32)
    root_depth = np.full_like(crop_map, np.nan, dtype=np.float32)
    irrigated_fields = irrigated_fields.astype(
        np.int8
    )  # change dtype to int, so that we can use the boolean array as index

    for i in range(crop_map.size):
        crop = crop_map[i]
        if crop != -1:
            age_days = crop_age_days_map[i]
            harvest_day = crop_harvest_age_days[i]
            assert harvest_day > 0
            crop_progress = age_days * 100 // harvest_day  # for to be integer
            assert crop_progress <= 100
            d1, d2, d3, d4 = crop_stage_data[crop]
            kc1, kc2, kc3 = kc_crop_stage[crop]
            assert d1 + d2 + d3 + d4 == 100
            if crop_progress <= d1:
                field_kc = kc1
            elif crop_progress <= d1 + d2:
                field_kc = kc1 + (crop_progress - d1) * (kc2 - kc1) / d2
            elif crop_progress <= d1 + d2 + d3:
                field_kc = kc2
            else:
                assert crop_progress <= d1 + d2 + d3 + d4
                field_kc = kc2 + (crop_progress - (d1 + d2 + d3)) * (kc3 - kc2) / d4
            assert not np.isnan(field_kc)
            kc[i] = field_kc

            root_depth[i] = (
                init_root_depth
                + age_days * rooth_depths[crop, irrigated_fields[i]] / harvest_day
            )

    return kc, root_depth


class LandCover(object):
    def __init__(self, model):
        """
        Initial part of the land cover type module
        Initialise the six land cover types

        * Forest No.0
        * Grasland/non irrigated land No.1
        * Paddy irrigation No.2
        * non-Paddy irrigation No.3
        * Sealed area No.4
        * Water covered area No.5
        """
        self.var = model.data.HRU
        self.model = model
        self.crop_farmers = model.agents.crop_farmers

        self.var.capriseindex = self.var.full_compressed(0, dtype=np.float32)

        self.forest_kc_per_10_days = xr.open_dataset(
            self.model.files["forcing"][
                "landcover/forest/cropCoefficientForest_10days"
            ],
            engine="zarr",
        )["cropCoefficientForest_10days"].values

    def water_body_exchange(self, groundwater_recharge):
        """computing leakage from rivers"""
        riverbedExchangeM3 = (
            self.model.data.grid.leakageriver_factor
            * self.var.cellArea
            * ((1 - self.var.capriseindex + 0.25) // 1)
        )
        riverbedExchangeM3[self.var.land_use_type != OPEN_WATER] = 0
        riverbedExchangeM3 = self.model.data.to_grid(
            HRU_data=riverbedExchangeM3, fn="sum"
        )
        riverbedExchangeM3 = np.minimum(
            riverbedExchangeM3, 0.80 * self.model.data.grid.channelStorageM3
        )
        # if there is a lake in this cell, there is no leakage
        riverbedExchangeM3[self.model.data.grid.waterBodyID > 0] = 0

        # adding leakage from river to the groundwater recharge
        waterbed_recharge = self.model.data.grid.M3toM(riverbedExchangeM3)

        # riverbed exchange means water is being removed from the river to recharge
        self.model.data.grid.riverbedExchangeM3 = (
            riverbedExchangeM3  # to be used in routing_kinematic
        )

        # first, lakes variable need to be extended to their area and not only to the discharge point
        lakeIDbyID = np.unique(self.model.data.grid.waterBodyID)

        lakestor_id = np.copy(self.model.data.grid.lakeStorage)
        resstor_id = np.copy(self.model.data.grid.resStorage)
        for id in range(len(lakeIDbyID)):  # for each lake or reservoir
            if lakeIDbyID[id] != 0:
                temp_map = np.where(
                    self.model.data.grid.waterBodyID == lakeIDbyID[id],
                    np.where(self.model.data.grid.lakeStorage > 0, 1, 0),
                    0,
                )  # Looking for the discharge point of the lake
                if np.sum(temp_map) == 0:  # try reservoir
                    temp_map = np.where(
                        self.model.data.grid.waterBodyID == lakeIDbyID[id],
                        np.where(self.model.data.grid.resStorage > 0, 1, 0),
                        0,
                    )  # Looking for the discharge point of the reservoir
                discharge_point = np.nanargmax(
                    temp_map
                )  # Index of the cell where the lake outlet is stored
                if self.model.data.grid.waterBodyTypTemp[discharge_point] != 0:
                    if (
                        self.model.data.grid.waterBodyTypTemp[discharge_point] == 1
                    ):  # this is a lake
                        # computing the lake area
                        area_stor = np.sum(
                            np.where(
                                self.model.data.grid.waterBodyID == lakeIDbyID[id],
                                self.model.data.grid.cellArea,
                                0,
                            )
                        )  # required to keep mass balance rigth
                        # computing the lake storage in meter and put this value in each cell including the lake
                        lakestor_id = np.where(
                            self.model.data.grid.waterBodyID == lakeIDbyID[id],
                            self.model.data.grid.lakeStorage[discharge_point]
                            / area_stor,
                            lakestor_id,
                        )  # in meter

                    else:  # this is a reservoir
                        # computing the reservoir area
                        area_stor = np.sum(
                            np.where(
                                self.model.data.grid.waterBodyID == lakeIDbyID[id],
                                self.model.data.grid.cellArea,
                                0,
                            )
                        )  # required to keep mass balance rigth
                        # computing the reservoir storage in meter and put this value in each cell including the reservoir
                        resstor_id = np.where(
                            self.model.data.grid.waterBodyID == lakeIDbyID[id],
                            self.model.data.grid.resStorage[discharge_point]
                            / area_stor,
                            resstor_id,
                        )  # in meter

        # Gathering lakes and reservoirs in the same array
        lakeResStorage = np.where(
            self.model.data.grid.waterBodyTypTemp == 0,
            0.0,
            np.where(
                self.model.data.grid.waterBodyTypTemp == 1, lakestor_id, resstor_id
            ),
        )  # in meter

        minlake = np.maximum(
            0.0, 0.98 * lakeResStorage
        )  # reasonable but arbitrary limit

        # leakage depends on water bodies storage, water bodies fraction and modflow saturated area
        lakebedExchangeM = self.model.data.grid.leakagelake_factor * (
            (1 - self.var.capriseindex + 0.25) // 1
        )
        lakebedExchangeM[self.var.land_use_type != OPEN_WATER] = 0
        lakebedExchangeM = self.model.data.to_grid(HRU_data=lakebedExchangeM, fn="sum")
        lakebedExchangeM = np.minimum(lakebedExchangeM, minlake)

        # Now, leakage is converted again from the lake/reservoir area to discharge point to be removed from the lake/reservoir store
        self.model.data.grid.lakebedExchangeM3 = np.zeros(
            self.model.data.grid.compressed_size, dtype=np.float32
        )
        for id in range(len(lakeIDbyID)):  # for each lake or reservoir
            if lakeIDbyID[id] != 0:
                temp_map = np.where(
                    self.model.data.grid.waterBodyID == lakeIDbyID[id],
                    np.where(self.model.data.grid.lakeStorage > 0, 1, 0),
                    0,
                )  # Looking for the discharge point of the lake
                if np.sum(temp_map) == 0:  # try reservoir
                    temp_map = np.where(
                        self.model.data.grid.waterBodyID == lakeIDbyID[id],
                        np.where(self.model.data.grid.resStorage > 0, 1, 0),
                        0,
                    )  # Looking for the discharge point of the reservoir
                discharge_point = np.nanargmax(
                    temp_map
                )  # Index of the cell where the lake outlet is stored
            # Converting the lake/reservoir leakage from meter to cubic meter and put this value in the cell corresponding to the outlet
            self.model.data.grid.lakebedExchangeM3[discharge_point] = np.sum(
                np.where(
                    self.model.data.grid.waterBodyID == lakeIDbyID[id],
                    lakebedExchangeM * self.model.data.grid.cellArea,
                    0,
                )
            )  # in m3
        self.model.data.grid.lakebedExchangeM = self.model.data.grid.M3toM(
            self.model.data.grid.lakebedExchangeM3
        )

        # compressed version for lakes and reservoirs
        lakeExchangeM3 = (
            np.compress(
                self.model.data.grid.compress_LR, self.model.data.grid.lakebedExchangeM
            )
            * self.model.data.grid.MtoM3C
        )

        # substract from both, because it is sorted by self.var.waterBodyTypCTemp
        self.model.data.grid.lakeStorageC = (
            self.model.data.grid.lakeStorageC - lakeExchangeM3
        )
        # assert (self.model.data.grid.lakeStorageC >= 0).all()
        self.model.data.grid.lakeVolumeM3C = (
            self.model.data.grid.lakeVolumeM3C - lakeExchangeM3
        )
        self.model.data.grid.reservoirStorageM3C = (
            self.model.data.grid.reservoirStorageM3C - lakeExchangeM3
        )

        # and from the combined one for waterbalance issues
        self.model.data.grid.lakeResStorageC = (
            self.model.data.grid.lakeResStorageC - lakeExchangeM3
        )
        # assert (self.model.data.grid.lakeResStorageC >= 0).all()
        self.model.data.grid.lakeResStorage = self.model.data.grid.full_compressed(
            0, dtype=np.float32
        )
        np.put(
            self.model.data.grid.lakeResStorage,
            self.model.data.grid.decompress_LR,
            self.model.data.grid.lakeResStorageC,
        )

        # adding leakage from lakes and reservoirs to the groundwater recharge
        waterbed_recharge += lakebedExchangeM

        groundwater_recharge += waterbed_recharge

    def step(self):
        timer = TimingModule("Landcover")

        if __debug__:
            interceptStor_pre = self.var.interceptStor.copy()
            w_pre = self.var.w.copy()
            topwater_pre = self.var.topwater.copy()

        crop_stage_lenghts = np.column_stack(
            [
                self.crop_farmers.crop_data["l_ini"],
                self.crop_farmers.crop_data["l_dev"],
                self.crop_farmers.crop_data["l_mid"],
                self.crop_farmers.crop_data["l_late"],
            ]
        )

        crop_factors = np.column_stack(
            [
                self.crop_farmers.crop_data["kc_initial"],
                self.crop_farmers.crop_data["kc_mid"],
                self.crop_farmers.crop_data["kc_end"],
            ]
        )

        root_depths = np.column_stack(
            [
                self.crop_farmers.crop_data["rd_rain"],
                self.crop_farmers.crop_data["rd_irr"],
            ]
        )

        self.var.cropKC, self.var.root_depth = get_crop_kc_and_root_depths(
            self.var.crop_map,
            self.var.crop_age_days_map,
            self.var.crop_harvest_age_days,
            irrigated_fields=self.model.agents.crop_farmers.irrigated_fields,
            crop_stage_data=crop_stage_lenghts,
            kc_crop_stage=crop_factors,
            rooth_depths=root_depths,
            init_root_depth=0.01,
        )

        self.var.root_depth[self.var.land_use_type == FOREST] = 2.0  # forest
        self.var.root_depth[
            (self.var.land_use_type == GRASSLAND_LIKE) & (self.var.land_owners == -1)
        ] = 0.1  # grassland
        self.var.root_depth[
            (self.var.land_use_type == GRASSLAND_LIKE) & (self.var.land_owners != -1)
        ] = 0.05  # fallow land. The rooting depth

        if self.model.use_gpu:
            self.var.cropKC = cp.array(self.var.cropKC)

        forest_cropCoefficientNC = self.model.data.to_HRU(
            data=self.model.data.grid.compress(
                self.forest_kc_per_10_days[(self.model.current_day_of_year - 1) // 10]
            ),
            fn=None,
        )

        self.var.cropKC[self.var.land_use_type == FOREST] = forest_cropCoefficientNC[
            self.var.land_use_type == FOREST
        ]  # forest
        assert (self.var.crop_map[self.var.land_use_type == GRASSLAND_LIKE] == -1).all()

        self.var.cropKC[self.var.land_use_type == GRASSLAND_LIKE] = 0.2

        (
            potential_transpiration,
            potential_bare_soil_evaporation,
            potential_evapotranspiration,
        ) = self.model.evaporation.step(self.var.ETRef)

        timer.new_split("PET")

        potential_transpiration_minus_interception_evaporation = (
            self.model.interception.step(potential_transpiration)
        )  # first thing that evaporates is the intercepted water.

        timer.new_split("Interception")

        (
            groundwater_abstraction_m3,
            channel_abstraction_m,
            returnFlow,
        ) = self.model.water_demand.step(potential_evapotranspiration)

        timer.new_split("Demand")

        openWaterEvap = self.var.full_compressed(0, dtype=np.float32)
        # Soil for forest, grassland, and irrigated land
        capillar = self.model.data.to_HRU(data=self.model.data.grid.capillar, fn=None)
        del self.model.data.grid.capillar

        (
            interflow,
            directRunoff,
            groundwater_recharge,
            openWaterEvap,
            actual_total_transpiration,
            actual_bare_soil_evaporation,
        ) = self.model.soil.step(
            capillar,
            openWaterEvap,
            potential_transpiration_minus_interception_evaporation,
            potential_bare_soil_evaporation,
            potential_evapotranspiration,
        )
        assert not (directRunoff < 0).any()
        timer.new_split("Soil")

        directRunoff = self.model.sealed_water.step(
            capillar, openWaterEvap, directRunoff
        )
        assert not (directRunoff < 0).any()

        timer.new_split("Sealed")

        self.var.actual_evapotranspiration_crop_life[self.var.crop_map != -1] += (
            np.minimum(
                self.var.actual_evapotranspiration[self.var.crop_map != -1],
                potential_evapotranspiration[self.var.crop_map != -1],
            )
        )
        self.var.potential_evapotranspiration_crop_life[self.var.crop_map != -1] += (
            potential_evapotranspiration[self.var.crop_map != -1]
        )

        assert not (directRunoff < 0).any()
        assert not np.isnan(interflow).any()
        assert not np.isnan(groundwater_recharge).any()
        assert not np.isnan(groundwater_abstraction_m3).any()
        assert not np.isnan(channel_abstraction_m).any()
        assert not np.isnan(openWaterEvap).any()

        if __debug__:
            balance_check(
                name="landcover_1",
                how="cellwise",
                influxes=[self.var.Rain, self.var.SnowMelt],
                outfluxes=[
                    self.var.natural_available_water_infiltration,
                    self.var.interceptEvap,
                ],
                prestorages=[interceptStor_pre],
                poststorages=[self.var.interceptStor],
                tollerance=1e-6,
            )

            balance_check(
                name="landcover_2",
                how="cellwise",
                influxes=[
                    self.var.natural_available_water_infiltration,
                    capillar,
                    self.var.actual_irrigation_consumption,
                ],
                outfluxes=[
                    directRunoff,
                    interflow,
                    groundwater_recharge,
                    actual_total_transpiration,
                    actual_bare_soil_evaporation,
                    openWaterEvap,
                ],
                prestorages=[w_pre, topwater_pre],
                poststorages=[
                    self.var.w,
                    self.var.topwater,
                ],
                tollerance=1e-6,
            )

            totalstorage = (
                np.sum(self.var.SnowCoverS, axis=0)
                / self.model.snowfrost.numberSnowLayers
                + self.var.interceptStor
                + self.var.w.sum(axis=0)
                + self.var.topwater
            )
            totalstorage_pre = (
                self.var.prevSnowCover
                + w_pre.sum(axis=0)
                + topwater_pre
                + interceptStor_pre
            )

            balance_check(
                name="landcover_3",
                how="cellwise",
                influxes=[
                    self.var.precipitation_m_day,
                    self.var.actual_irrigation_consumption,
                    capillar,
                ],
                outfluxes=[
                    directRunoff,
                    interflow,
                    groundwater_recharge,
                    actual_total_transpiration,
                    actual_bare_soil_evaporation,
                    openWaterEvap,
                    self.var.interceptEvap,
                    self.var.snowEvap,
                ],
                prestorages=[totalstorage_pre],
                poststorages=[totalstorage],
                tollerance=1e-6,
            )

        groundwater_recharge = self.model.data.to_grid(
            HRU_data=groundwater_recharge, fn="weightedmean"
        )
        # self.water_body_exchange(groundwater_recharge)

        timer.new_split("Waterbody exchange")

        if self.model.timing:
            print(timer)

        return (
            self.model.data.to_grid(HRU_data=interflow, fn="weightedmean"),
            self.model.data.to_grid(HRU_data=directRunoff, fn="weightedmean"),
            groundwater_recharge,
            groundwater_abstraction_m3,
            channel_abstraction_m,
            openWaterEvap,
            returnFlow,
        )
