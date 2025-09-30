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

from typing import Literal

import numpy as np
import numpy.typing as npt
import zarr
from numba import njit

from geb.module import Module
from geb.workflows import TimingModule, balance_check

# All natural areas MUST be before the sealed and water areas
FOREST: Literal[0] = 0
GRASSLAND_LIKE: Literal[1] = 1
PADDY_IRRIGATED: Literal[2] = 2
NON_PADDY_IRRIGATED: Literal[3] = 3
SEALED: Literal[4] = 4
OPEN_WATER: Literal[5] = 5


ALL_LAND_COVER_TYPES: list[int] = [
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
    crop_stage_lenghts,
    crop_sub_stage_lengths,
    kc_crop_stage,
    rooth_depths,
    init_root_depth=0.2,
    get_crop_sub_stage=False,
):
    kc = np.full_like(crop_map, np.nan, dtype=np.float32)
    root_depth = np.full_like(crop_map, np.nan, dtype=np.float32)
    crop_sub_stage = np.full_like(crop_map, -1, dtype=np.int8)
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
            l1, l2, l3, l4 = crop_stage_lenghts[crop]
            kc1, kc2, kc3 = kc_crop_stage[crop]
            assert l1 + l2 + l3 + l4 == 100
            if crop_progress <= l1:
                field_kc = kc1
            elif crop_progress <= l1 + l2:
                field_kc = kc1 + (crop_progress - l1) * (kc2 - kc1) / l2
            elif crop_progress <= l1 + l2 + l3:
                field_kc = kc2
            else:
                assert crop_progress <= l1 + l2 + l3 + l4
                field_kc = kc2 + (crop_progress - (l1 + l2 + l3)) * (kc3 - kc2) / l4
            assert not np.isnan(field_kc)
            kc[i] = field_kc

            root_depth[i] = (
                init_root_depth
                + age_days * rooth_depths[crop, irrigated_fields[i]] / harvest_day
            )

            if get_crop_sub_stage:
                d1, d2a, d2b, d3a, d3b, d4 = crop_sub_stage_lengths[crop]
                assert d1 + d2a + d2b + d3a + d3b + d4 == 100

                if crop_progress <= d1:
                    crop_sub_stage[i] = 0
                elif crop_progress <= d1 + d2a:
                    crop_sub_stage[i] = 1
                elif crop_progress <= d1 + d2a + d2b:
                    crop_sub_stage[i] = 2
                elif crop_progress <= d1 + d2a + d2b + d3a:
                    crop_sub_stage[i] = 3
                elif crop_progress <= d1 + d2a + d2b + d3a + d3b:
                    crop_sub_stage[i] = 4
                else:
                    assert crop_progress <= d1 + d2a + d2b + d3a + d3b + d4
                    crop_sub_stage[i] = 5

    return kc, root_depth, crop_sub_stage


class LandCover(Module):
    """Implements all land cover processes in the hydrological model.

    Args:
        model: The GEB model instance.
        hydrology: The hydrology submodel instance.
    """

    def __init__(self, model, hydrology) -> None:
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    def spinup(self) -> None:
        self.HRU.var.capriseindex = self.HRU.full_compressed(0, dtype=np.float32)

        store = zarr.storage.LocalStore(
            self.model.files["grid"]["landcover/forest/crop_coefficient"],
            read_only=True,
        )
        self.grid.var.forest_kc_per_10_days = zarr.open_group(store, mode="r")[
            "crop_coefficient"
        ][:]

    def step(
        self,
        snow: npt.NDArray[np.float32],
        rain: npt.NDArray[np.float32],
        snow_melt: npt.NDArray[np.float32],
    ):
        timer: TimingModule = TimingModule("Landcover")

        if __debug__:
            interception_storage_pre = self.HRU.var.interception_storage.copy()
            w_pre = self.HRU.var.w.copy()
            topwater_pre = self.HRU.var.topwater.copy()

        crop_stage_lenghts = np.column_stack(
            [
                self.model.agents.crop_farmers.var.crop_data["l_ini"],
                self.model.agents.crop_farmers.var.crop_data["l_dev"],
                self.model.agents.crop_farmers.var.crop_data["l_mid"],
                self.model.agents.crop_farmers.var.crop_data["l_late"],
            ]
        )

        get_crop_sub_stage = self.model.agents.crop_farmers.var.crop_data_type == "GAEZ"
        if get_crop_sub_stage:
            crop_sub_stage_lengths = np.column_stack(
                [
                    self.model.agents.crop_farmers.var.crop_data["d1"],
                    self.model.agents.crop_farmers.var.crop_data["d2a"],
                    self.model.agents.crop_farmers.var.crop_data["d2b"],
                    self.model.agents.crop_farmers.var.crop_data["d3a"],
                    self.model.agents.crop_farmers.var.crop_data["d3b"],
                    self.model.agents.crop_farmers.var.crop_data["d4"],
                ]
            )
        else:
            crop_sub_stage_lengths = np.full(
                (self.model.agents.crop_farmers.var.crop_data.shape[0], 6),
                np.nan,
                dtype=np.float32,
            )

        crop_factors = np.column_stack(
            [
                self.model.agents.crop_farmers.var.crop_data["kc_initial"],
                self.model.agents.crop_farmers.var.crop_data["kc_mid"],
                self.model.agents.crop_farmers.var.crop_data["kc_end"],
            ]
        )

        root_depths = np.column_stack(
            [
                self.model.agents.crop_farmers.var.crop_data["rd_rain"],
                self.model.agents.crop_farmers.var.crop_data["rd_irr"],
            ]
        )

        crop_factor, self.HRU.var.root_depth, crop_sub_stage = (
            get_crop_kc_and_root_depths(
                self.HRU.var.crop_map,
                self.HRU.var.crop_age_days_map,
                self.HRU.var.crop_harvest_age_days,
                irrigated_fields=self.model.agents.crop_farmers.irrigated_fields,
                crop_stage_lenghts=crop_stage_lenghts,
                crop_sub_stage_lengths=crop_sub_stage_lengths,
                kc_crop_stage=crop_factors,
                rooth_depths=root_depths,
                init_root_depth=0.01,
                get_crop_sub_stage=get_crop_sub_stage,
            )
        )

        self.HRU.var.root_depth[self.HRU.var.land_use_type == FOREST] = 2.0  # forest
        self.HRU.var.root_depth[
            (self.HRU.var.land_use_type == GRASSLAND_LIKE)
            & (self.HRU.var.land_owners == -1)
        ] = 0.1  # grassland
        self.HRU.var.root_depth[
            (self.HRU.var.land_use_type == GRASSLAND_LIKE)
            & (self.HRU.var.land_owners != -1)
        ] = 0.05  # fallow land. The rooting depth

        forest_cropCoefficientNC = self.hydrology.to_HRU(
            data=self.grid.compress(
                self.grid.var.forest_kc_per_10_days[
                    (self.model.current_day_of_year - 1) // 10
                ]
            ),
            fn=None,
        )

        crop_factor[self.HRU.var.land_use_type == FOREST] = forest_cropCoefficientNC[
            self.HRU.var.land_use_type == FOREST
        ]  # forest
        assert (
            self.HRU.var.crop_map[self.HRU.var.land_use_type == GRASSLAND_LIKE] == -1
        ).all()

        crop_factor[self.HRU.var.land_use_type == GRASSLAND_LIKE] = 1.0  # grassland

        (
            potential_transpiration,
            potential_bare_soil_evaporation,
            potential_evapotranspiration,
            snow_melt,
            snow_sublimation,
        ) = self.hydrology.evaporation.step(
            self.HRU.var.reference_evapotranspiration_grass, snow_melt, crop_factor
        )

        timer.finish_split("PET")

        (
            potential_transpiration_minus_interception_evaporation,
            interception_evaporation,
        ) = self.hydrology.interception.step(
            potential_transpiration=potential_transpiration,
            rain=rain,
            snow_melt=snow_melt,
        )  # first thing that evaporates is the intercepted water.

        del potential_transpiration

        timer.finish_split("Interception")

        (
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow,  # from all sources
            irrigation_loss_to_evaporation_m,
            total_water_demand_loss_m3,
        ) = self.hydrology.water_demand.step(potential_evapotranspiration)

        timer.finish_split("Demand")

        # Soil for forest, grassland, and irrigated land
        capillar = self.hydrology.to_HRU(data=self.grid.var.capillar, fn=None)

        (
            interflow,
            runoff,
            groundwater_recharge,
            open_water_evaporation,
            actual_transpiration,
            actual_bare_soil_evaporation,
        ) = self.hydrology.soil.step(
            capillar,
            potential_transpiration_minus_interception_evaporation,
            potential_bare_soil_evaporation,
            potential_evapotranspiration,
            natural_available_water_infiltration=self.HRU.var.natural_available_water_infiltration,
            actual_irrigation_consumption=self.HRU.var.actual_irrigation_consumption,
        )
        assert (runoff >= 0).all()
        timer.finish_split("Soil")

        self.HRU.var.actual_evapotranspiration = (
            actual_bare_soil_evaporation
            + actual_transpiration
            + open_water_evaporation
            + interception_evaporation
            + snow_sublimation  # ice should be included in the future
            + irrigation_loss_to_evaporation_m
        )

        growing_crop_mask = self.HRU.var.crop_map != -1

        self.HRU.var.actual_evapotranspiration_crop_life[growing_crop_mask] += (
            actual_transpiration[growing_crop_mask]
        )
        self.HRU.var.potential_evapotranspiration_crop_life[growing_crop_mask] += (
            potential_transpiration_minus_interception_evaporation[growing_crop_mask]
        )
        self.HRU.var.actual_evapotranspiration_crop_life_per_crop_stage[
            crop_sub_stage[growing_crop_mask], growing_crop_mask
        ] += actual_transpiration[growing_crop_mask]
        self.HRU.var.potential_evapotranspiration_crop_life_per_crop_stage[
            crop_sub_stage[growing_crop_mask], growing_crop_mask
        ] += potential_transpiration_minus_interception_evaporation[growing_crop_mask]

        assert not np.isnan(self.HRU.var.actual_evapotranspiration).any()
        assert not (runoff < 0).any()
        assert not np.isnan(interflow).any()
        assert not np.isnan(groundwater_recharge).any()
        assert not np.isnan(groundwater_abstraction_m3).any()
        assert not np.isnan(channel_abstraction_m3).any()
        assert not np.isnan(open_water_evaporation).any()

        if __debug__:
            balance_check(
                name="landcover_1",
                how="cellwise",
                influxes=[rain, snow_melt],
                outfluxes=[
                    self.HRU.var.natural_available_water_infiltration,
                    interception_evaporation,
                ],
                prestorages=[interception_storage_pre],
                poststorages=[self.HRU.var.interception_storage],
                tollerance=1e-6,
            )

            balance_check(
                name="landcover_2",
                how="cellwise",
                influxes=[
                    self.HRU.var.natural_available_water_infiltration,
                    capillar,
                    self.HRU.var.actual_irrigation_consumption,
                ],
                outfluxes=[
                    interflow,
                    runoff,
                    groundwater_recharge,
                    open_water_evaporation,
                    actual_transpiration,
                    actual_bare_soil_evaporation,
                ],
                prestorages=[np.nansum(w_pre, axis=0), topwater_pre],
                poststorages=[
                    np.nansum(self.HRU.var.w, axis=0),
                    self.HRU.var.topwater,
                ],
                tollerance=1e-6,
                error_identifiers={
                    "land_use_type": self.HRU.var.land_use_type,
                },
            )

            totalstorage_landcover = (
                np.sum(self.HRU.var.SnowCoverS, axis=0)
                / self.hydrology.snowfrost.var.numberSnowLayers
                + self.HRU.var.interception_storage
                + np.nansum(self.HRU.var.w, axis=0)
                + self.HRU.var.topwater
            )
            totalstorage_landcover_pre = (
                np.sum(self.HRU.var.prevSnowCover, axis=0)
                / self.hydrology.snowfrost.var.numberSnowLayers
                + np.nansum(w_pre, axis=0)
                + topwater_pre
                + interception_storage_pre
            )

            balance_check(
                name="landcover_3",
                how="cellwise",
                influxes=[
                    rain,
                    snow,
                    self.HRU.var.actual_irrigation_consumption,
                    capillar,
                ],
                outfluxes=[
                    runoff,
                    interflow,
                    groundwater_recharge,
                    actual_transpiration,
                    actual_bare_soil_evaporation,
                    open_water_evaporation,
                    interception_evaporation,
                    snow_sublimation,
                ],
                prestorages=[totalstorage_landcover_pre],
                poststorages=[totalstorage_landcover],
                tollerance=1e-6,
            )

            balance_check(
                name="landcover_4",
                how="cellwise",
                influxes=[
                    rain,
                    snow,
                    self.HRU.var.actual_irrigation_consumption,
                    capillar,
                    irrigation_loss_to_evaporation_m,  # irrigation loss is coming from external sources
                ],
                outfluxes=[
                    runoff,
                    interflow,
                    groundwater_recharge,
                    self.HRU.var.actual_evapotranspiration,
                ],
                prestorages=[totalstorage_landcover_pre],
                poststorages=[totalstorage_landcover],
                tollerance=1e-6,
            )

        groundwater_recharge = self.hydrology.to_grid(
            HRU_data=groundwater_recharge, fn="weightedmean"
        )
        timer.finish_split("Waterbody exchange")

        if self.model.timing:
            print(timer)

        self.report(locals())

        return (
            self.hydrology.to_grid(HRU_data=interflow, fn="weightedmean"),
            self.hydrology.to_grid(HRU_data=runoff, fn="weightedmean"),
            groundwater_recharge,
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow,
            capillar,
            total_water_demand_loss_m3,
        )

    @property
    def name(self) -> str:
        return "hydrology.landcover"
