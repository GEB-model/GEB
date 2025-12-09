"""Functions for calculating evapotranspiration."""

from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit

from geb.types import ArrayFloat32

from .landcovers import (
    FOREST,
    GRASSLAND_LIKE,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
)

N_SOIL_LAYERS: Literal[6] = 6


@njit(cache=True, inline="always")
def get_critical_soil_moisture_content(
    p: np.float32,
    wfc_m: ArrayFloat32,
    wwp_m: ArrayFloat32,
) -> ArrayFloat32:
    """Calculate the critical soil moisture content.

    The critical soil moisture content is defined as the quantity of stored soil moisture below
    which water uptake is impaired and the crop begins to close its stomata. It is not a fixed
    value. Restriction of water uptake due to water stress starts at a higher water content
    when the potential transpiration rate is higher" (Van Diepen et al., 1988: WOFOST 6.0, p.86).

    A higher p value means that the critical soil moisture content is higher, i.e. the plant can
    extract water from the soil at a lower soil moisture content. Thus when p is 1 the critical
    soil moisture content is equal to the wilting point, and when p is 0 the critical soil moisture
    content is equal to the field capacity.

    Args:
        p: The fraction of easily available soil water, between 0 and 1.
        wfc_m: The field capacity in m.
        wwp_m: The wilting point in m.

    Returns:
        The critical soil moisture content in m.

    """
    return (np.float32(1) - p) * (wfc_m - wwp_m) + wwp_m


@njit(cache=True, inline="always")
def get_root_mass_ratios(
    root_depth_m: np.float32,
    root_ratios: ArrayFloat32,
    soil_layer_height_m: ArrayFloat32,
) -> ArrayFloat32:
    """Calculate the root mass ratios for each soil layer assuming a triangular root distribution.

    Args:
        root_depth_m: The effective root depth in meters.
        root_ratios: The root ratios for each soil layer, where each ratio
        soil_layer_height_m: The height of each soil layer in meters.

    Returns:
        A numpy array of root mass ratios for each soil layer. The total root mass ratio is 1.
    """
    current_depth = np.float32(0)
    root_mass_ratio = np.zeros_like(soil_layer_height_m, dtype=np.float32)

    root_triangle_area = np.float32(0.5) * root_depth_m * root_depth_m

    for layer in range(N_SOIL_LAYERS):
        if root_ratios[layer] == np.float32(0):
            break  # No roots in this layer (so also not in remaining), skip further calculations

        triangle_block_height = soil_layer_height_m[layer] * root_ratios[layer]
        triangle_block_center_point_depth = triangle_block_height / 2 + current_depth
        triangle_block_width = root_depth_m - triangle_block_center_point_depth
        root_mass_ratio[layer] = (
            triangle_block_height * triangle_block_width / root_triangle_area
        )

        current_depth += soil_layer_height_m[layer]

    return root_mass_ratio


@njit(cache=True, inline="always")
def get_transpiration_factor(
    w_m: np.float32, wwp_m: np.float32, wcrit_m: np.float32
) -> np.float32:
    """Calculate the transpiration factor based on the available water in the soil.

    A factor of 1 means that there is no water stress and the plant can transpire at its potential rate.
    A factor of 0 means that the plant is at wilting point and cannot transpire.

    Args:
        w_m: The soil water content in m.
        wwp_m: The wilting point in m.
        wcrit_m: The critical soil moisture content in m.

    Returns:
        The transpiration factor between 0 and 1.
    """
    nominator = w_m - wwp_m
    denominator = wcrit_m - wwp_m
    if denominator == np.float32(0):
        if nominator > np.float32(0):
            return np.float32(1)
        else:
            return np.float32(0)
    factor = nominator / denominator
    if factor < np.float32(0):
        return np.float32(0)
    if factor > np.float32(1):
        return np.float32(1)
    return factor


@njit(cache=True, inline="always")
def get_root_ratios(
    root_depth_m: np.float32, soil_layer_height_m: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Calculate the root ratios for each soil layer based on the effective root depth.

    Args:
        root_depth_m: The effective root depth in meters.
        soil_layer_height_m: The height of each soil layer in meters.

    Returns:
        A numpy array of root ratios for each soil layer, where each ratio
        is the proportion of the soil layer that has roots in it.
    """
    remaining_root_depth = root_depth_m
    root_ratios = np.zeros_like(soil_layer_height_m)
    for layer in range(N_SOIL_LAYERS):
        root_ratios[layer] = min(
            remaining_root_depth / soil_layer_height_m[layer], np.float32(1)
        )
        remaining_root_depth -= soil_layer_height_m[layer]
        if remaining_root_depth < np.float32(0):
            return root_ratios
    return root_ratios


@njit(cache=True, inline="always")
def get_fraction_easily_available_soil_water(
    crop_group_number: np.float32,
    reference_evapotranspiration_grass_full_day_m: np.float32,
) -> np.float32:
    """Calculate the fraction of easily available soil water.

    Calculation is based on crop group number and potential evapotranspiration
    following Van Diepen et al., 1988: WOFOST 6.0, Theory and Algorithms p.87.

    Args:
        crop_group_number: The crop group number is a indicator of adaptation to dry climate,
            Van Diepen et al., 1988: WOFOST 6.0, Theory and Algorithms p.87
        reference_evapotranspiration_grass_full_day_m: Potential evapotranspiration in m for a full day.

    Returns:
        The fraction of easily available soil water, p is closer to 0 if evapo is bigger and cropgroup is smaller
    """
    potential_evapotranspiration_cm: np.float32 = (
        reference_evapotranspiration_grass_full_day_m * np.float32(100)
    )

    p: np.float32 = np.float32(1) / (
        np.float32(0.76) + np.float32(1.5) * potential_evapotranspiration_cm
    ) - np.float32(0.1) * (np.float32(5) - crop_group_number)

    # Additional correction for crop groups 1 and 2
    if crop_group_number <= np.float32(2.5):
        p: np.float32 = np.float32(p) + (
            potential_evapotranspiration_cm - np.float32(0.6)
        ) / (crop_group_number * (crop_group_number + np.float32(3.0)))

    if p < np.float32(0):
        p: np.float32 = np.float32(0)
    if p > np.float32(1):
        p: np.float32 = np.float32(1)
    return p


@njit(inline="always", cache=True)
def get_transpiration_factor_per_layer(
    soil_layer_height_m: npt.NDArray[np.float32],
    effective_root_depth_m: np.float32,
    w_m: npt.NDArray[np.float32],
    wfc_m: npt.NDArray[np.float32],
    wwp_m: npt.NDArray[np.float32],
    p: np.float32,
    correct_root_mass: bool = True,
) -> npt.NDArray[np.float32]:
    """Calculate the transpiration factor for each soil layer based on the available water and root ratios.

    Args:
        soil_layer_height_m: height of each soil layer in meters.
        effective_root_depth_m: root depth in meters, which is the maximum depth where roots can extract water.
        w_m: soil water content in each layer in m.
        wfc_m: field capacity in each layer in m.
        wwp_m: wilting point in each layer in m.
        p: ratio of easily available soil water, which is a measure of how easily plants can extract water from the soil.
        correct_root_mass: If True, the root mass ratio is corrected assuming a triangular root distribution.

    Returns:
        A numpy array of transpiration factors [0-1] for each soil layer, where each factor
            is the proportion of the available water in that layer that can be used
            for transpiration. If plenty of water available, sums to 1, else less than 1.
    """
    root_ratios = get_root_ratios(
        effective_root_depth_m,
        soil_layer_height_m,
    )

    if correct_root_mass:
        root_mass_ratio_per_layer = get_root_mass_ratios(
            effective_root_depth_m,
            root_ratios,
            soil_layer_height_m,
        )
    else:
        root_mass_ratio_per_layer = (
            soil_layer_height_m * root_ratios / effective_root_depth_m
        )

    w_available = (w_m * root_ratios).sum()
    wfc_available = (wfc_m * root_ratios).sum()
    wwp_available = (wwp_m * root_ratios).sum()

    # Compute total water stress factor based on the available water in the soil
    critical_soil_moisture_content = get_critical_soil_moisture_content(
        p, wfc_available, wwp_available
    )
    transpiration_factor_total = get_transpiration_factor(
        w_available, wwp_available, critical_soil_moisture_content
    )
    if transpiration_factor_total == np.float32(0):
        return np.zeros_like(w_m)

    transpiration_factor_per_layer = np.zeros_like(w_m)

    for layer in range(N_SOIL_LAYERS):
        critical_soil_moisture_content_layer = get_critical_soil_moisture_content(
            p, wfc_m[layer], wwp_m[layer]
        )

        transpiration_factor_per_layer[layer] = (
            get_transpiration_factor(
                w_m[layer], wwp_m[layer], critical_soil_moisture_content_layer
            )
            * root_mass_ratio_per_layer[layer]
        )

    transpiration_factor_per_layer = (
        transpiration_factor_per_layer
        / transpiration_factor_per_layer.sum()
        * transpiration_factor_total
    )

    return transpiration_factor_per_layer


@njit(cache=True, inline="always")
def calculate_transpiration(
    soil_is_frozen: bool,
    wwp_m: npt.NDArray[np.float32],  # [m]
    wfc_m: npt.NDArray[np.float32],  # [m]
    wres_m: npt.NDArray[np.float32],  # [m]
    soil_layer_height_m: npt.NDArray[np.float32],  # [m]
    land_use_type: np.int32,
    root_depth_m: np.float32,  # [m]
    crop_map: np.int32,
    natural_crop_groups: np.float32,
    potential_transpiration_m: np.float32,  # [m]
    reference_evapotranspiration_grass_m_hour: np.float32,  # [m]
    crop_group_number_per_group: npt.NDArray[np.float32],
    w_m: npt.NDArray[np.float32],  # [m]
    topwater_m: np.float32,  # [m]
    minimum_effective_root_depth_m: np.float32,  # [m]
) -> tuple[np.float32, np.float32]:
    """Calculate transpiration for a single soil cell.

    Args:
        soil_is_frozen: Boolean indicating whether the soil is frozen.
        wwp_m: Wilting point soil moisture content [m], shape (N_SOIL_LAYERS,).
        wfc_m: Field capacity soil moisture content [m], shape (N_SOIL_LAYERS,).
        wres_m: Residual soil moisture content [m], shape (N_SOIL_LAYERS,).
        soil_layer_height_m: Height of each soil layer [m], shape (N_SOIL_LAYERS,).
        land_use_type: Land use type of the hydrological response unit.
        root_depth_m: The root depth [m].
        crop_map: Crop map indicating the crop type for the hydrological response unit. -1 indicates no crop.
        natural_crop_groups: Crop group numbers for natural areas (see WOFOST 6.0).
        potential_transpiration_m: Potential transpiration [m].
        reference_evapotranspiration_grass_m_hour: Reference evapotranspiration for grass [m/hour]. This is an hourly value that will be converted to a daily value for the calculation.
        crop_group_number_per_group: Crop group numbers for each crop type.
        w_m: Soil water content [m], shape (N_SOIL_LAYERS,).
        topwater_m: Topwater [m], which is the water available for evaporation and transpiration for paddy irrigated fields.
        minimum_effective_root_depth_m: Minimum effective root depth [m], used to ensure that the effective root depth is not less than this value. Crops can extract water up to this depth.

    Returns:
        A tuple containing:
            - transpiration: The actual transpiration [m] for the cell.
            - topwater_m: Updated topwater [m] after transpiration.
    """
    transpiration: np.float32 = np.float32(0.0)

    remaining_potential_transpiration: np.float32 = potential_transpiration_m
    if land_use_type == PADDY_IRRIGATED:
        transpiration_from_topwater: np.float32 = min(
            topwater_m, remaining_potential_transpiration
        )
        remaining_potential_transpiration -= transpiration_from_topwater
        topwater_m -= transpiration_from_topwater
        transpiration += transpiration_from_topwater

    if not soil_is_frozen:
        # get group group numbers for natural areas
        if land_use_type == FOREST or land_use_type == GRASSLAND_LIKE:
            crop_group_number: np.float32 = natural_crop_groups
        else:  #
            crop_group_number: np.float32 = crop_group_number_per_group[crop_map]

        # vegetation-specific factor for easily available soil water
        fraction_easily_available_soil_water: np.float32 = get_fraction_easily_available_soil_water(
            crop_group_number=crop_group_number,
            reference_evapotranspiration_grass_full_day_m=reference_evapotranspiration_grass_m_hour
            * np.float32(24.0),
        )

        effective_root_depth: np.float32 = np.maximum(
            minimum_effective_root_depth_m, root_depth_m
        )
        fraction_easily_available_soil_water = np.float32(
            fraction_easily_available_soil_water
        )

        transpiration_factor_per_layer: npt.NDArray[np.float32] = (
            get_transpiration_factor_per_layer(
                soil_layer_height_m,
                effective_root_depth,
                w_m,
                wfc_m,
                wwp_m,
                fraction_easily_available_soil_water,
            )
        )

        for layer in range(N_SOIL_LAYERS):
            transpiration_layer: np.float32 = (
                remaining_potential_transpiration
                * transpiration_factor_per_layer[layer]
            )
            # limit the transpiration to the available water in the soil
            transpiration_layer = min(
                transpiration_layer,
                w_m[layer] - wres_m[layer],
            )

            w_m[layer] -= transpiration_layer
            w_m[layer] = max(
                w_m[layer], wres_m[layer]
            )  # soil moisture can never be lower than wres
            transpiration += transpiration_layer

    return transpiration, topwater_m


@njit(cache=True, inline="always")
def calculate_bare_soil_evaporation(
    soil_is_frozen: bool,
    land_use_type: np.int32,
    potential_bare_soil_evaporation_m: np.float32,  # [m]
    open_water_evaporation_m: np.float32,  # [m]
    w_m: npt.NDArray[np.float32],  # [m]
    wres_m: npt.NDArray[np.float32],  # [m]
) -> np.float32:
    """Calculate bare soil evaporation for a single soil cell.

    Args:
        soil_is_frozen: Boolean indicating whether the soil is frozen.
        land_use_type: Land use type of the hydrological response unit.
        potential_bare_soil_evaporation_m: Potential bare soil evaporation [m].
        open_water_evaporation_m: Open water evaporation [m], which is the water evaporated from open water areas.
        w_m: Soil water content [m], shape (N_SOIL_LAYERS,).
        wres_m: Residual soil moisture content [m], shape (N_SOIL_LAYERS,).

    Returns:
        The actual bare soil evaporation [m] for the cell.
    """
    # limit the bare soil evaporation to the available water in the soil
    if (
        not soil_is_frozen
        and land_use_type != PADDY_IRRIGATED
        and land_use_type != OPEN_WATER
        and land_use_type != SEALED
    ):
        # TODO: Minor bug, this should only occur when topwater is above 0
        # fix this after completing soil module speedup
        actual_bare_soil_evaporation = min(
            max(
                np.float32(0),
                potential_bare_soil_evaporation_m - open_water_evaporation_m,
            ),
            max(
                w_m[0] - wres_m[0], np.float32(0)
            ),  # soil moisture can never be lower than 0
        )
        # remove the bare soil evaporation from the top layer
        w_m[0] -= actual_bare_soil_evaporation
        w_m[0] = max(w_m[0], wres_m[0])  # soil moisture can never be lower than wres
    else:
        # if the soil is frozen, no evaporation occurs
        # if the field is flooded (paddy irrigation), no bare soil evaporation occurs
        actual_bare_soil_evaporation = np.float32(0)

    return actual_bare_soil_evaporation


@njit(cache=True, inline="always")
def evapotranspirate(
    soil_is_frozen: bool,
    wwp_m: npt.NDArray[np.float32],  # [m]
    wfc_m: npt.NDArray[np.float32],  # [m]
    wres_m: npt.NDArray[np.float32],  # [m]
    soil_layer_height_m: npt.NDArray[np.float32],  # [m]
    land_use_type: np.int32,
    root_depth_m: np.float32,  # [m]
    crop_map: np.int32,
    natural_crop_groups: np.float32,
    potential_transpiration_m: np.float32,  # [m]
    potential_bare_soil_evaporation_m: np.float32,  # [m]
    potential_evapotranspiration_m: np.float32,  # [m]
    frost_index: np.float32,
    crop_group_number_per_group: npt.NDArray[np.float32],
    w_m: npt.NDArray[np.float32],  # [m]
    topwater_m: np.float32,  # [m]
    open_water_evaporation_m: np.float32,  # [m]
    minimum_effective_root_depth_m: np.float32,  # [m]
) -> tuple[np.float32, np.float32, np.float32]:
    """Evapotranspiration calculation for a single soil cell.

    Args:
        soil_is_frozen: Boolean indicating whether the soil is frozen.
        wwp_m: Wilting point soil moisture content [m], shape (N_SOIL_LAYERS,).
        wfc_m: Field capacity soil moisture content [m], shape (N_SOIL_LAYERS,).
        wres_m: Residual soil moisture content [m], shape (N_SOIL_LAYERS,).
        soil_layer_height_m: Height of each soil layer [m], shape (N_SOIL_LAYERS,).
        land_use_type: Land use type of the hydrological response unit.
        root_depth_m: The root depth [m].
        crop_map: Crop map indicating the crop type for the hydrological response unit. -1 indicates no crop.
        natural_crop_groups: Crop group numbers for natural areas (see WOFOST 6.0).
        potential_transpiration_m: Potential transpiration [m].
        potential_bare_soil_evaporation_m: Potential bare soil evaporation [m].
        potential_evapotranspiration_m: Potential evapotranspiration [m].
        frost_index: Frost index indicating whether the soil is frozen.
        crop_group_number_per_group: Crop group numbers for each crop type.
        w_m: Soil water content [m], shape (N_SOIL_LAYERS,).
        topwater_m: Topwater [m], which is the water available for evaporation and transpiration for paddy irrigated fields.
        open_water_evaporation_m: Open water evaporation [m], which is the water evaporated from open water areas.
        minimum_effective_root_depth_m: Minimum effective root depth [m], used to ensure that the effective root depth is not less than this value. Crops can extract water up to this depth.

    Returns:
        A tuple containing:
            - The actual transpiration [m] for the cell.
            - The actual bare soil evaporation in meters for the cell.
            - Updated topwater [m] after transpiration.
    """
    transpiration_m, topwater_m = calculate_transpiration(
        soil_is_frozen,
        wwp_m,
        wfc_m,
        wres_m,
        soil_layer_height_m,
        land_use_type,
        root_depth_m,
        crop_map,
        natural_crop_groups,
        potential_transpiration_m,
        potential_evapotranspiration_m,
        crop_group_number_per_group,
        w_m,
        topwater_m,
        minimum_effective_root_depth_m,
    )

    actual_bare_soil_evaporation_m = calculate_bare_soil_evaporation(
        soil_is_frozen,
        land_use_type,
        potential_bare_soil_evaporation_m,
        open_water_evaporation_m,
        w_m,
        wres_m,
    )

    return (
        transpiration_m,
        actual_bare_soil_evaporation_m,
        topwater_m,
    )
