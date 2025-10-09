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
from numba import njit

from .landcover import FOREST, GRASSLAND_LIKE, NON_PADDY_IRRIGATED, PADDY_IRRIGATED


@njit(cache=True, inline="always")
def get_potential_transpiration(
    potential_evapotranspiration_m: np.float32,
    potential_bare_soil_evaporation_m: np.float32,
) -> np.float32:
    """Calculate potential transpiration.

    Args:
        potential_evapotranspiration_m: Potential evapotranspiration [m]
        potential_bare_soil_evaporation_m: Potential bare soil evaporation [m]

    Returns:
        Potential transpiration [m]
    """
    return max(0.0, potential_evapotranspiration_m - potential_bare_soil_evaporation_m)


@njit(cache=True, inline="always")
def get_potential_bare_soil_evaporation(
    reference_evapotranspiration_grass_m_per_day: np.float32,
    sublimation_m: np.float32,
) -> np.float32:
    """Calculate potential bare soil evaporation.

    Removes sublimation from potential bare soil evaporation and ensures non-negative result.

    Args:
        reference_evapotranspiration_grass_m_per_day: Reference evapotranspiration [m]
        sublimation_m: Sublimation from snow [m]

    Returns:
        Potential bare soil evaporation [m]
    """
    return max(
        np.float32(0.2) * reference_evapotranspiration_grass_m_per_day - sublimation_m,
        np.float32(0.0),
    )


@njit(cache=True, inline="always")
def get_potential_evapotranspiration(
    reference_evapotranspiration_grass_m: np.float32,
    crop_factor: np.float32,
    CO2_induced_crop_factor_adustment: np.float32,
) -> np.float32:
    """Calculate potential evapotranspiration.

    Args:
        reference_evapotranspiration_grass_m: Reference evapotranspiration [m]
        crop_factor: Crop factor for each land use type [dimensionless]
        CO2_induced_crop_factor_adustment: Adjustment factor for CO2 effects [dimensionless]

    Returns:
        Potential evapotranspiration [m]
    """
    return (
        crop_factor * reference_evapotranspiration_grass_m
    ) * CO2_induced_crop_factor_adustment


@njit(cache=True)
def get_crop_factors_and_root_depths(
    land_use_map: npt.NDArray[np.int32],
    crop_factor_forest_map: npt.NDArray[np.float32],
    crop_map: npt.NDArray[np.int32],
    crop_age_days_map: npt.NDArray[np.int32],
    crop_harvest_age_days: npt.NDArray[np.int32],
    crop_stage_lengths: npt.NDArray[np.int32],
    crop_sub_stage_lengths: npt.NDArray[np.int32],
    crop_factor_per_crop_stage: npt.NDArray[np.float32],
    crop_root_depths: npt.NDArray[np.float32],
    crop_init_root_depth: np.float32 = np.float32(0.2),
    get_crop_sub_stage: bool = False,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int8],
]:
    """Calculate crop factors and root depths based on land use and crop information.

    Args:
        land_use_map: Map of land use types.
        crop_factor_forest_map: Map of crop factors for forest land use if forested.
        crop_map: Map of crop types, -1 for non-crop land use. Indices refer to crop arrays.
        crop_age_days_map: Map of crop ages in days, -1 for non-crop land use.
        crop_harvest_age_days: Array of harvest ages in days for each crop type.
        crop_stage_lengths: Array of lengths of growth stages for each crop type.
        crop_sub_stage_lengths: Array of lengths of sub-stages for each crop type.
        crop_factor_per_crop_stage: Array of crop factors for each growth stage for each crop type.
        crop_root_depths: Array of root depths for each crop type and irrigation status.
        crop_init_root_depth: Initial root depth for crops.
        get_crop_sub_stage: Whether to calculate and return crop sub-stages.

    Returns:
        crop_factor: Array of crop factors for each grid cell.
        root_depth: Array of root depths for each grid cell.
        crop_sub_stage: Array of crop sub-stages for each grid cell, -1 if not calculated.

    """
    crop_factor = np.full_like(crop_map, np.nan, dtype=np.float32)
    root_depth = np.full_like(crop_map, np.nan, dtype=np.float32)
    crop_sub_stage = np.full_like(crop_map, -1, dtype=np.int8)

    for i in range(crop_map.size):
        land_use = land_use_map[i]
        crop = crop_map[i]
        if crop != -1:
            age_days = crop_age_days_map[i]
            harvest_day = crop_harvest_age_days[i]
            assert harvest_day > 0
            crop_progress = age_days * 100 // harvest_day  # for to be integer
            assert crop_progress <= 100
            l1, l2, l3, l4 = crop_stage_lengths[crop]
            kc1, kc2, kc3 = crop_factor_per_crop_stage[crop]
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
            crop_factor[i] = field_kc

            is_irrigated: int = int(land_use in (PADDY_IRRIGATED, NON_PADDY_IRRIGATED))

            root_depth[i] = (
                crop_init_root_depth
                + age_days
                * max(
                    (crop_root_depths[crop, is_irrigated] - crop_init_root_depth), 0.0
                )
                / harvest_day
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

        elif land_use == FOREST:
            root_depth[i] = 2.0  # forest root depth is set to 2m
            # crop sub stage remains -1
            crop_factor[i] = crop_factor_forest_map[i]

        elif land_use == GRASSLAND_LIKE:
            root_depth[i] = 0.1  # grassland root depth is set to 0.1m
            # crop sub stage remains -1
            crop_factor[i] = 1.0

    return crop_factor, root_depth, crop_sub_stage


@njit(cache=True, inline="always")
def get_CO2_induced_crop_factor_adustment(
    CO2_concentration_ppm: float,
) -> float:
    """Calculate the CO2 induced crop factor adjustment.

    For reference see:
        Reference Manual, Chapter 3 – AquaCrop, Version 7.1
        Eq. 3.10e/2

    Args:
        CO2_concentration_ppm: The CO2 concentration in ppm.

    Returns:
        The CO2 induced crop factor adjustment [dimensionless]
    """
    base_co2_concentration_ppm: float = 369.41
    return 1.0 - 0.05 * (CO2_concentration_ppm - base_co2_concentration_ppm) / (
        550 - base_co2_concentration_ppm
    )
