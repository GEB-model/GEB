"""Functions for calculating evapotranspiration."""

import numpy as np
import numpy.typing as npt
from numba import njit

from geb.geb_types import ArrayFloat32

from ..landcovers import (
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
)
from .constants import N_SOIL_LAYERS


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


@njit(cache=True, inline="always")
def calculate_transpiration(
    soil_is_frozen: bool,
    wwp_m: npt.NDArray[np.float32],  # [m]
    wfc_m: npt.NDArray[np.float32],  # [m]
    wres_m: npt.NDArray[np.float32],  # [m]
    soil_layer_height_m: npt.NDArray[np.float32],  # [m]
    land_use_type: np.int32,
    root_depth_m: np.float32,  # [m]
    crop_group_number: np.float32,
    potential_transpiration_m: np.float32,  # [m]
    reference_evapotranspiration_grass_m_hour: np.float32,  # [m]
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
        crop_group_number: Crop group number for this HRU, pre-resolved from land use type
            and crop map (see WOFOST 6.0).
        potential_transpiration_m: Potential transpiration [m].
        reference_evapotranspiration_grass_m_hour: Reference evapotranspiration for grass [m/hour]. This is an hourly value that will be converted to a daily value for the calculation.
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

        remaining_root_depth_m: np.float32 = effective_root_depth
        w_available_m: np.float32 = np.float32(0.0)
        wfc_available_m: np.float32 = np.float32(0.0)
        wwp_available_m: np.float32 = np.float32(0.0)

        for layer in range(N_SOIL_LAYERS):
            root_ratio: np.float32 = min(
                remaining_root_depth_m / soil_layer_height_m[layer], np.float32(1.0)
            )
            if root_ratio < np.float32(0.0):
                root_ratio = np.float32(0.0)

            w_available_m += w_m[layer] * root_ratio
            wfc_available_m += wfc_m[layer] * root_ratio
            wwp_available_m += wwp_m[layer] * root_ratio

            remaining_root_depth_m -= soil_layer_height_m[layer]
            if remaining_root_depth_m < np.float32(0.0):
                break

        critical_soil_moisture_content_m: np.float32 = (
            get_critical_soil_moisture_content(
                fraction_easily_available_soil_water,
                wfc_available_m,
                wwp_available_m,
            )
        )
        transpiration_factor_total: np.float32 = get_transpiration_factor(
            w_available_m,
            wwp_available_m,
            critical_soil_moisture_content_m,
        )
        if transpiration_factor_total == np.float32(0.0):
            return transpiration, topwater_m

        # A first pass is needed to normalize the layer weights without allocating
        # a temporary factor array for every HRU.
        normalization_factor: np.float32 = np.float32(0.0)
        current_depth_m: np.float32 = np.float32(0.0)
        remaining_root_depth_m = effective_root_depth
        root_triangle_area_m2: np.float32 = (
            np.float32(0.5) * effective_root_depth * effective_root_depth
        )

        for layer in range(N_SOIL_LAYERS):
            root_ratio = min(
                remaining_root_depth_m / soil_layer_height_m[layer], np.float32(1.0)
            )
            if root_ratio <= np.float32(0.0):
                break

            triangle_block_height_m: np.float32 = (
                soil_layer_height_m[layer] * root_ratio
            )
            triangle_block_center_depth_m: np.float32 = (
                triangle_block_height_m / np.float32(2.0) + current_depth_m
            )
            triangle_block_width_m: np.float32 = (
                effective_root_depth - triangle_block_center_depth_m
            )
            root_mass_ratio: np.float32 = (
                triangle_block_height_m * triangle_block_width_m / root_triangle_area_m2
            )

            critical_soil_moisture_content_layer_m: np.float32 = (
                get_critical_soil_moisture_content(
                    fraction_easily_available_soil_water,
                    wfc_m[layer],
                    wwp_m[layer],
                )
            )
            normalization_factor += (
                get_transpiration_factor(
                    w_m[layer],
                    wwp_m[layer],
                    critical_soil_moisture_content_layer_m,
                )
                * root_mass_ratio
            )

            current_depth_m += soil_layer_height_m[layer]
            remaining_root_depth_m -= soil_layer_height_m[layer]
            if remaining_root_depth_m < np.float32(0.0):
                break

        if normalization_factor == np.float32(0.0):
            return transpiration, topwater_m

        current_depth_m = np.float32(0.0)
        remaining_root_depth_m = effective_root_depth
        for layer in range(N_SOIL_LAYERS):
            root_ratio = min(
                remaining_root_depth_m / soil_layer_height_m[layer], np.float32(1.0)
            )
            if root_ratio <= np.float32(0.0):
                break

            triangle_block_height_m = soil_layer_height_m[layer] * root_ratio
            triangle_block_center_depth_m = (
                triangle_block_height_m / np.float32(2.0) + current_depth_m
            )
            triangle_block_width_m = (
                effective_root_depth - triangle_block_center_depth_m
            )
            root_mass_ratio = (
                triangle_block_height_m * triangle_block_width_m / root_triangle_area_m2
            )

            critical_soil_moisture_content_layer_m = get_critical_soil_moisture_content(
                fraction_easily_available_soil_water,
                wfc_m[layer],
                wwp_m[layer],
            )
            transpiration_factor_layer: np.float32 = (
                get_transpiration_factor(
                    w_m[layer],
                    wwp_m[layer],
                    critical_soil_moisture_content_layer_m,
                )
                * root_mass_ratio
                / normalization_factor
                * transpiration_factor_total
            )

            transpiration_layer: np.float32 = (
                remaining_potential_transpiration * transpiration_factor_layer
            )
            transpiration_layer = min(transpiration_layer, w_m[layer] - wres_m[layer])

            w_m[layer] -= transpiration_layer
            w_m[layer] = max(
                w_m[layer], wres_m[layer]
            )  # soil moisture can never be lower than wres
            transpiration += transpiration_layer

            current_depth_m += soil_layer_height_m[layer]
            remaining_root_depth_m -= soil_layer_height_m[layer]
            if remaining_root_depth_m < np.float32(0.0):
                break

    return transpiration, topwater_m


@njit(cache=True, inline="always")
def calculate_bare_soil_evaporation(
    soil_is_frozen: bool,
    land_use_type: np.int32,
    potential_direct_evaporation_m: np.float32,
    open_water_evaporation_m: np.float32,
    w_m: npt.NDArray[np.float32],
    wres_m: npt.NDArray[np.float32],
    wfc_m: npt.NDArray[np.float32],
    unsaturated_hydraulic_conductivity_m_per_s: np.float32,
) -> np.float32:
    """Calculate bare soil evaporation for a single soil cell.

    Args:
        soil_is_frozen: Indicate whether the soil is frozen.
        land_use_type: Land use type (-).
        potential_direct_evaporation_m: Potential direct evaporation (soil/water) (m).
        open_water_evaporation_m: Actual open water evaporation (m).
        w_m: Soil water content (m), shape (N_SOIL_LAYERS,).
        wres_m: Residual soil moisture content (m), shape (N_SOIL_LAYERS,).
        wfc_m: Field capacity soil moisture content (m), shape (N_SOIL_LAYERS,).
        unsaturated_hydraulic_conductivity_m_per_s: Unsaturated hydraulic conductivity (m/s).

    Returns:
        Actual bare soil evaporation (m).
    """
    # Bare soil evaporation is only applied for natural (non-watery, non-paddy, non-sealed) areas.
    if (
        not soil_is_frozen
        and land_use_type != PADDY_IRRIGATED
        and land_use_type != OPEN_WATER
        and land_use_type != SEALED
    ):
        # Apply an additional correction due to dust mulch
        # reduction based on relative moisture of the top layer.
        relative_moisture = max(
            np.float32(0.0),
            (w_m[0] - wres_m[0]) / (wfc_m[0] - wres_m[0]),
        )
        dust_mulch_reduction = min(np.float32(1.0), relative_moisture**2)

        potential_direct_evaporation_m = (
            potential_direct_evaporation_m * dust_mulch_reduction
        )

        # Limit potential evaporation by the unsaturated hydraulic conductivity
        # This accounts for the reduced ability of the soil to transport water to the surface
        potential_direct_evaporation_m = min(
            potential_direct_evaporation_m,
            unsaturated_hydraulic_conductivity_m_per_s * np.float32(3600),
        )

        # Subtract open water evaporation (though it's expected to be 0 for these land uses)
        actual_bare_soil_evaporation = min(
            max(
                np.float32(0),
                potential_direct_evaporation_m - open_water_evaporation_m,
            ),
            max(w_m[0] - wres_m[0], np.float32(0)),
        )
        # Update soil moisture
        w_m[0] -= actual_bare_soil_evaporation
        w_m[0] = max(w_m[0], wres_m[0])
    else:
        actual_bare_soil_evaporation = np.float32(0)

    return actual_bare_soil_evaporation
