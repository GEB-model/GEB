"""Functions for calculating evapotranspiration."""

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from geb.geb_types import ArrayFloat32, ArrayInt8, ArrayInt32, TwoDArrayFloat32
from geb.numba_config import NUMBA_CACHE

from ..landcovers import (
    FOREST,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
)
from .constants import MIN_ACTIVE_SNOW_SWE_M, N_SOIL_LAYERS


@njit(cache=NUMBA_CACHE, inline="always")
def get_critical_soil_moisture_content(
    p: float | np.floating,
    wfc_m: float | np.floating,
    wwp_m: float | np.floating,
) -> np.float32:
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


@njit(cache=NUMBA_CACHE, inline="always")
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


@njit(cache=NUMBA_CACHE, inline="always")
def get_transpiration_factor(
    w_m: float | np.floating, wwp_m: float | np.floating, wcrit_m: float | np.floating
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
    return np.float32(factor)


@njit(cache=NUMBA_CACHE, inline="always")
def get_root_ratios(
    root_depth_m: float | np.floating, soil_layer_height_m: npt.NDArray[np.float32]
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


@njit(cache=NUMBA_CACHE, inline="always")
def get_fraction_easily_available_soil_water(
    crop_group_number: float | np.floating,
    reference_evapotranspiration_grass_full_day_m: float | np.floating,
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


@njit(cache=NUMBA_CACHE, inline="always")
def calculate_transpiration(
    soil_is_frozen: bool | np.bool_,
    wwp_m: npt.NDArray[np.float32],  # [m]
    wfc_m: npt.NDArray[np.float32],  # [m]
    wres_m: npt.NDArray[np.float32],  # [m]
    soil_layer_height_m: npt.NDArray[np.float32],  # [m]
    land_use_type: int | np.integer,
    root_depth_m: float | np.floating,  # [m]
    crop_group_number: float | np.floating,
    potential_transpiration_m: float | np.floating,  # [m]
    daily_reference_evapotranspiration_grass_m: float | np.floating,  # [m]
    w_m: npt.NDArray[np.float32],  # [m]
    topwater_m: float | np.floating,  # [m]
    minimum_effective_root_depth_m: float | np.floating,  # [m]
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
        daily_reference_evapotranspiration_grass_m: Daily reference evapotranspiration for grass for a full day [m].
        w_m: Soil water content [m], shape (N_SOIL_LAYERS,).
        topwater_m: Topwater [m], which is the water available for evaporation and transpiration for paddy irrigated fields.
        minimum_effective_root_depth_m: Minimum effective root depth [m], used to ensure that the effective root depth is not less than this value. Crops can extract water up to this depth.

    Returns:
        A tuple containing:
            - transpiration: The actual transpiration [m] for the cell.
            - topwater_m: Updated topwater [m] after transpiration.
    """
    transpiration: np.float32 = np.float32(0.0)
    topwater_m: np.float32 = np.float32(topwater_m)
    if potential_transpiration_m <= np.float32(0.0):
        return transpiration, topwater_m

    remaining_potential_transpiration: np.float32 = np.float32(
        potential_transpiration_m
    )
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
            reference_evapotranspiration_grass_full_day_m=daily_reference_evapotranspiration_grass_m,
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


# Readily evaporable water (REW) as a fraction of the total evaporable water (TEW) of
# a 0.10 m surface layer: the median REW/TEW over the nine soil classes of FAO-56
# Table 19 (midpoints of the ranges; 0.39 for clay to 0.52 for loamy sand). Table 19
# gives TEW for Ze = 0.10 m. REW is a depth that does not grow with Ze (FAO-56: 5-12
# mm), so it is taken over READILY_EVAPORABLE_REFERENCE_DEPTH_M only.
READILY_EVAPORABLE_FRACTION: np.float32 = np.float32(0.44)
READILY_EVAPORABLE_REFERENCE_DEPTH_M: np.float32 = np.float32(0.10)
# Soil layers of the FAO-56 evaporating surface layer: 0.05 + 0.10 m = 0.15 m
# (FAO-56 Ze = 0.10-0.15 m).
N_EVAPORATION_LAYERS: int = 2


@njit(cache=NUMBA_CACHE, inline="always")
def get_soil_evaporation_reduction_coefficient(
    w_m: float | np.floating,
    wfc_m: float | np.floating,
    wwp_m: float | np.floating,
    layer_depth_m: float | np.floating,
) -> np.float32:
    """Evaporation reduction coefficient Kr of the surface layer (FAO-56, eq. 74).

    FAO-56 describes soil evaporation in two stages, using the depletion De of the
    surface layer (the cumulative evaporation since the layer was last at field
    capacity):

    - Stage 1 (energy limited), De <= REW: Kr = 1.
    - Stage 2 (falling rate), De > REW: Kr = (TEW - De) / (TEW - REW)  (eq. 74),
      reaching 0 when De = TEW.

    The symbols map to this function as follows:

    - TEW, total evaporable water (eq. 73): TEW = (theta_FC - 0.5 theta_WP) Ze.
      The water contents here are already depths over the layer (theta Ze), so
      TEW = wfc_m - 0.5 wwp_m.
    - REW, readily evaporable water (Table 19): READILY_EVAPORABLE_FRACTION of
      TEW, taken over the top READILY_EVAPORABLE_REFERENCE_DEPTH_M only, because
      REW is a depth (5-12 mm) that does not grow with the layer depth.
    - De, depletion: FAO-56 tracks De,i-1 with a daily water balance. Here it is
      read from the soil state at the start of the time step, as the water below
      field capacity: De = wfc_m - w_m (0 when the layer is at or above field
      capacity).

    Args:
        w_m: Water content of the surface layer (m).
        wfc_m: Water content at field capacity of the surface layer (m).
        wwp_m: Water content at wilting point of the surface layer (m).
        layer_depth_m: Depth of the surface layer (m).

    Returns:
        Kr between 0 and 1.
    """
    # TEW (FAO-56 eq. 73)
    TEW: np.float32 = np.float32(wfc_m - np.float32(0.5) * wwp_m)
    if TEW <= np.float32(0.0):
        return np.float32(0.0)

    # REW (FAO-56 Table 19), at most TEW
    REW: np.float32 = (
        READILY_EVAPORABLE_FRACTION
        * TEW
        * min(
            np.float32(READILY_EVAPORABLE_REFERENCE_DEPTH_M / layer_depth_m),
            np.float32(1.0),
        )
    )

    # De, depletion below field capacity
    De: np.float32 = np.float32(max(np.float32(wfc_m - w_m), np.float32(0.0)))

    # Stage 1: energy limited
    if De <= REW:
        return np.float32(1.0)

    # Stage 2: falling rate (FAO-56 eq. 74). Kr < 1 because De > REW; clipped at 0
    # because the soil can dry below 0.5 theta_WP (De > TEW).
    return np.float32(max((TEW - De) / (TEW - REW), np.float32(0.0)))


@njit(cache=NUMBA_CACHE, inline="always")
def calculate_bare_soil_evaporation(
    soil_is_frozen: bool | np.bool_,
    land_use_type: int | np.integer,
    potential_direct_evaporation_m: float | np.floating,
    energy_limited_soil_evaporation_m: float | np.floating,
    open_water_evaporation_m: float | np.floating,
    w_m: npt.NDArray[np.float32],
    wres_m: npt.NDArray[np.float32],
    wfc_m: npt.NDArray[np.float32],
    wwp_m: npt.NDArray[np.float32],
    soil_layer_height_m: npt.NDArray[np.float32],
) -> np.float32:
    """Calculate bare soil evaporation for a single soil cell.

    Soil evaporation draws on the surface layer of FAO-56 (Allen et al., 1998, Ch. 7):
    the top N_EVAPORATION_LAYERS soil layers (0-15 cm; FAO-56 Ze = 0.10-0.15 m).
    FAO-56 eq. 71: E = min(Kr (Kc_max - Kcb) ET0, few Kc_max ET0). The two-stage
    coefficient Kr of that layer (get_soil_evaporation_reduction_coefficient), from
    the soil moisture of this hour, reduces the energy-limited soil evaporation
    (Kc_max - Kcb) ET0. The potential direct evaporation is the wet-soil potential
    Ke_max ET0 = min((Kc_max - Kcb), few Kc_max) ET0 after interception has used
    part of the energy. The result is limited to the water above
    the residual content (and above half the wilting point). The water is taken from
    the top down, as the soil dries from the surface.

    Args:
        soil_is_frozen: Indicate whether the soil is frozen.
        land_use_type: Land use type (-).
        potential_direct_evaporation_m: Potential direct evaporation (soil/water) (m).
        energy_limited_soil_evaporation_m: Soil evaporation of a wet surface without
            the exposed-soil limit, (Kc_max - Kcb) ET0 (m).
        open_water_evaporation_m: Actual open water evaporation (m).
        w_m: Soil water content (m), shape (N_SOIL_LAYERS,).
        wres_m: Residual soil moisture content (m), shape (N_SOIL_LAYERS,).
        wfc_m: Field capacity soil moisture content (m), shape (N_SOIL_LAYERS,).
        wwp_m: Wilting point soil moisture content (m), shape (N_SOIL_LAYERS,).
        soil_layer_height_m: Soil layer thickness (m), shape (N_SOIL_LAYERS,).

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
        w_surface_m: np.float32 = np.float32(0.0)
        wfc_surface_m: np.float32 = np.float32(0.0)
        wwp_surface_m: np.float32 = np.float32(0.0)
        depth_surface_m: np.float32 = np.float32(0.0)
        available_m: np.float32 = np.float32(0.0)
        for layer in range(N_EVAPORATION_LAYERS):
            depth_surface_m += soil_layer_height_m[layer]
            w_surface_m += w_m[layer]
            wfc_surface_m += wfc_m[layer]
            wwp_surface_m += wwp_m[layer]
            available_m += max(
                w_m[layer] - max(wres_m[layer], np.float32(0.5) * wwp_m[layer]),
                np.float32(0.0),
            )
        reduction: np.float32 = get_soil_evaporation_reduction_coefficient(
            w_surface_m, wfc_surface_m, wwp_surface_m, depth_surface_m
        )
        # Subtract open water evaporation (though it's expected to be 0 for these land uses)
        actual_bare_soil_evaporation: np.float32 = np.float32(
            min(
                max(
                    np.float32(0),
                    min(
                        reduction * energy_limited_soil_evaporation_m,
                        potential_direct_evaporation_m,
                    )
                    - open_water_evaporation_m,
                ),
                available_m,
            )
        )
        remaining_m: np.float32 = actual_bare_soil_evaporation
        for layer in range(N_EVAPORATION_LAYERS):
            taken_m: np.float32 = min(
                remaining_m,
                max(
                    w_m[layer] - max(wres_m[layer], np.float32(0.5) * wwp_m[layer]),
                    np.float32(0.0),
                ),
            )
            w_m[layer] -= taken_m
            remaining_m -= taken_m
        actual_bare_soil_evaporation -= remaining_m
    else:
        actual_bare_soil_evaporation = np.float32(0)
    return actual_bare_soil_evaporation


@njit(cache=NUMBA_CACHE, inline="always")
def get_dew(
    wet_vegetation_evaporation_m: np.float32,
    snow_water_equivalent_m: np.float64,
) -> np.float32:
    """Dew from a negative Penman-Monteith evaporation of a wet surface.

    When the surface loses more energy than it gains (clear, calm nights) and the air
    is humid, the Penman-Monteith evaporation is negative: water vapour condenses on
    the vegetation and soil. Condensation forms a water film on the outside of the
    leaves and does not pass the stomata, so the rate is that of a wet surface
    (surface resistance zero), not that of the transpiring grass reference. On snow,
    deposition is part of the snow model.

    Args:
        wet_vegetation_evaporation_m: Evaporation of wet short vegetation in the time
            step, before clipping at zero (m).
        snow_water_equivalent_m: Snow water equivalent (m).

    Returns:
        Dew in the time step (m).
    """
    if snow_water_equivalent_m > np.float32(0.0):
        return np.float32(0.0)
    return max(-wet_vegetation_evaporation_m, np.float32(0.0))


@njit(cache=NUMBA_CACHE, inline="always")
def limit_evaporation_under_snow(
    land_use_type: np.int32,
    snow_water_equivalent_m: np.float32,
    potential_direct_evaporation_m: np.float32,
    potential_transpiration_m: np.float32,
) -> tuple[np.float32, np.float32]:
    """Limit soil evaporation and transpiration when snow covers the ground.

    Snow covers the soil, so no soil evaporation occurs. Low vegetation (grass, crops)
    is buried and does not transpire. Forest crowns stand above the snow pack, so
    forest keeps its potential transpiration (a model assumption). It is small in
    winter in any case: the frozen soil stops it (handled in the transpiration
    itself) and the seasonal activity of evergreen trees and the leaf area of
    deciduous trees are low.

    Args:
        land_use_type: Land use type of the hydrological response unit.
        snow_water_equivalent_m: Total snow water equivalent on the ground (m).
        potential_direct_evaporation_m: Potential soil/open-water evaporation (m).
        potential_transpiration_m: Potential transpiration (m).

    Returns:
        The potential direct evaporation and potential transpiration under snow (m).
    """
    if snow_water_equivalent_m <= MIN_ACTIVE_SNOW_SWE_M:
        return potential_direct_evaporation_m, potential_transpiration_m
    if land_use_type == FOREST:
        return np.float32(0.0), potential_transpiration_m
    return np.float32(0.0), np.float32(0.0)


@njit(cache=NUMBA_CACHE, parallel=True)
def resolve_crop_group_number(
    crop_map: ArrayInt32,
    land_use_type: ArrayInt32,
    crop_group_number_forest: ArrayFloat32,
    crop_group_number_grassland_like: ArrayFloat32,
    crop_group_number_per_group: ArrayFloat32,
    forest_land_use: int,
    grassland_like_land_use: int,
) -> ArrayFloat32:
    """Resolve crop group number per HRU without intermediate array allocations.

    Args:
        crop_map: Crop type ID per HRU (-).
        land_use_type: Land use type per HRU (-).
        crop_group_number_forest: Crop group number for forest HRUs (-).
        crop_group_number_grassland_like: Crop group number for grassland HRUs (-).
        crop_group_number_per_group: Crop group numbers indexed by crop ID (-).
        forest_land_use: Integer enum value for FOREST land use.
        grassland_like_land_use: Integer enum value for GRASSLAND_LIKE land use.

    Returns:
        Crop group number per HRU (-).
    """
    num_cells: int = crop_map.size
    max_crop_index: int = len(crop_group_number_per_group) - 1
    crop_group_numbers: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    for i in prange(num_cells):
        current_land_use_type: int = land_use_type[i]
        crop_id: int = crop_map[i]
        # a crop sets the crop group, also for rainfed crops on grassland-like land
        if crop_id >= 0:
            crop_group_numbers[i] = crop_group_number_per_group[
                min(crop_id, max_crop_index)
            ]
        elif current_land_use_type == forest_land_use:
            crop_group_numbers[i] = crop_group_number_forest[i]
        elif current_land_use_type == grassland_like_land_use:
            crop_group_numbers[i] = crop_group_number_grassland_like[i]
        else:
            crop_group_numbers[i] = crop_group_number_per_group[0]
    return crop_group_numbers


@njit(cache=NUMBA_CACHE, parallel=True)
def accumulate_crop_evapotranspiration(
    crop_map: ArrayInt32,
    crop_sub_stage: ArrayInt8 | ArrayInt32,
    actual_evapotranspiration_m: ArrayFloat32,
    potential_evapotranspiration_m: ArrayFloat32,
    actual_evapotranspiration_crop_life: ArrayFloat32,
    potential_evapotranspiration_crop_life: ArrayFloat32,
    actual_evapotranspiration_crop_life_per_crop_stage: TwoDArrayFloat32,
    potential_evapotranspiration_crop_life_per_crop_stage: TwoDArrayFloat32,
) -> None:
    """Accumulate daily actual and potential evapotranspiration into crop life totals in-place.

    Args:
        crop_map: Crop type ID per HRU (-1 for no active crop).
        crop_sub_stage: Current crop development sub-stage per HRU (-1 if not computed;
            then the per-stage accumulators are not updated).
        actual_evapotranspiration_m: Actual daily evapotranspiration (meters).
        potential_evapotranspiration_m: Potential daily evapotranspiration (meters).
        actual_evapotranspiration_crop_life: Cumulative actual ET over crop lifecycle (meters). Modified in place.
        potential_evapotranspiration_crop_life: Cumulative potential ET over crop lifecycle (meters). Modified in place.
        actual_evapotranspiration_crop_life_per_crop_stage: Cumulative actual ET per stage (meters). Modified in place.
        potential_evapotranspiration_crop_life_per_crop_stage: Cumulative potential ET per stage (meters). Modified in place.
    """
    num_cells: int = crop_map.size
    for i in prange(num_cells):
        if crop_map[i] != -1:
            actual_et_m: np.float32 = actual_evapotranspiration_m[i]
            potential_et_m: np.float32 = potential_evapotranspiration_m[i]
            actual_evapotranspiration_crop_life[i] += actual_et_m
            potential_evapotranspiration_crop_life[i] += potential_et_m
            crop_stage_index: int = crop_sub_stage[i]
            # sub-stages are only computed for GAEZ crop data (-1 otherwise)
            if crop_stage_index == -1:
                continue
            actual_evapotranspiration_crop_life_per_crop_stage[crop_stage_index, i] += (
                actual_et_m
            )
            potential_evapotranspiration_crop_life_per_crop_stage[
                crop_stage_index, i
            ] += potential_et_m
