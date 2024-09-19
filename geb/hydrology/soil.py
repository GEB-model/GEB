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
from pathlib import Path
from geb.workflows import TimingModule, balance_check
from numba import njit, prange


from .landcover import (
    FOREST,
    GRASSLAND_LIKE,
    PADDY_IRRIGATED,
    NON_PADDY_IRRIGATED,
    SEALED,
)


@njit(cache=True, inline="always", fastmath=True)
def get_soil_water_potential(
    theta,
    thetar,
    thetas,
    lambda_,
    bubbling_pressure_cm,
    minimum_effective_saturation=np.float32(0.01),
):
    """
    Calculates the soil water potential (capillary suction) using the van Genuchten model.

    Note that theta, thetar and thetas can also be given as the height of the water column
    in the soil layer as w, wres and ws since only there relative size is used. Of course
    consistency is key.

    Parameters
    ----------
    theta : np.ndarray
        The soil moisture content (m³/m³)
    thetar : np.ndarray
        The residual soil moisture content (m³/m³)
    thetas : np.ndarray
        The saturated soil moisture content (m³/m³)
    lambda_ : np.ndarray
        The van Genuchten parameter lambda (1/m)
    bubbling_pressure_cm : np.ndarray
        The bubbling pressure (cm)
    """
    # van Genuchten parameters
    alpha = np.float32(1) / bubbling_pressure_cm
    n = lambda_ + np.float32(1)
    m = np.float32(1) - np.float32(1) / n

    # Effective saturation
    effective_saturation = (theta - thetar) / (thetas - thetar)
    effective_saturation = max(minimum_effective_saturation, effective_saturation)
    effective_saturation = min(np.float32(1), effective_saturation)

    # Compute capillary pressure head (phi)
    phi = ((effective_saturation) ** (-np.float32(1) / m) - np.float32(1)) ** (
        np.float32(1) / n
    ) / alpha  # Positive value

    # Soil water potential (negative value for suction)
    capillary_suction = -phi

    return capillary_suction


@njit(cache=True, inline="always", fastmath=True)
def get_soil_moisture_at_pressure(
    capillary_suction, bubbling_pressure_cm, thetas, thetar, lambda_
):
    """
    Calculates the soil moisture content at a given soil water potential (capillary suction)
    using the van Genuchten model.

    Parameters
    ----------
    capillary_suction : np.ndarray
        The soil water potential (capillary suction) (m)
    bubbling_pressure_cm : np.ndarray
        The bubbling pressure (cm)
    thetas : np.ndarray
        The saturated soil moisture content (m³/m³)
    thetar : np.ndarray
        The residual soil moisture content (m³/m³)
    lambda_ : np.ndarray
        The van Genuchten parameter lambda (1/m)
    """
    alpha = np.float32(1) / bubbling_pressure_cm
    n = lambda_ + np.float32(1)
    m = np.float32(1) - np.float32(1) / n
    phi = -capillary_suction

    water_retention_curve = (np.float32(1) / (np.float32(1) + (alpha * phi) ** n)) ** m

    theta = water_retention_curve * (thetas - thetar) + thetar
    return theta


@njit(cache=True, inline="always", fastmath=True)
def get_aeration_stress_threshold(
    ws, soil_layer_height, crop_aeration_stress_threshold
):
    max_saturation_fraction = ws / soil_layer_height
    # Water storage in root zone at aeration stress threshold (m)
    return (
        max_saturation_fraction - (crop_aeration_stress_threshold / np.float32(100))
    ) * soil_layer_height


@njit(cache=True, inline="always", fastmath=True)
def get_aeration_stress_factor(
    aeration_days_counter, crop_lag_aeration_days, ws, w, aeration_stress_threshold
):
    if aeration_days_counter < crop_lag_aeration_days:
        stress = np.float32(1) - ((ws - w) / (ws - aeration_stress_threshold))
        aeration_stress_factor = np.float32(1) - ((aeration_days_counter / 3) * stress)
    else:
        aeration_stress_factor = (ws - w) / (ws - aeration_stress_threshold)
    return aeration_stress_factor


@njit(cache=True, inline="always", fastmath=True)
def get_critical_soil_moisture_content(p, wfc, wwp):
    """
    "The critical soil moisture content is defined as the quantity of stored soil moisture below
    which water uptake is impaired and the crop begins to close its stomata. It is not a fixed
    value. Restriction of water uptake due to water stress starts at a higher water content
    when the potential transpiration rate is higher" (Van Diepen et al., 1988: WOFOST 6.0, p.86)

    A higher p value means that the critical soil moisture content is higher, i.e. the plant can
    extract water from the soil at a lower soil moisture content. Thus when p is 1 the critical
    soil moisture content is equal to the wilting point, and when p is 0 the critical soil moisture
    content is equal to the field capacity.
    """
    return (np.float32(1) - p) * (wfc - wwp) + wwp


def get_available_water(w, wwp):
    return np.maximum(0.0, w - wwp)


def get_maximum_water_content(wfc, wwp):
    return wfc - wwp


@njit(cache=True, inline="always")
def get_fraction_easily_available_soil_water(
    crop_group_number, potential_evapotranspiration, fastmath=True
):
    """
    Calculate the fraction of easily available soil water, based on crop group number and potential evapotranspiration
    following Van Diepen et al., 1988: WOFOST 6.0, p.87

    Parameters
    ----------
    crop_group_number : np.ndarray
        The crop group number is a indicator of adaptation to dry climate,
        Van Diepen et al., 1988: WOFOST 6.0, p.87
    potential_evapotranspiration : np.ndarray
        Potential evapotranspiration in m

    Returns
    -------
    np.ndarray
        The fraction of easily available soil water, p is closer to 0 if evapo is bigger and cropgroup is smaller
    """

    p = np.zeros_like(crop_group_number)
    for i in range(crop_group_number.size):
        p[i] = get_fraction_easily_available_soil_water_single(
            crop_group_number[i], potential_evapotranspiration[i]
        )
    return p


@njit(cache=True, inline="always")
def get_fraction_easily_available_soil_water_single(
    crop_group_number, potential_evapotranspiration, fastmath=True
):
    """
    Calculate the fraction of easily available soil water, based on crop group number and potential evapotranspiration
    following Van Diepen et al., 1988: WOFOST 6.0, p.87

    Parameters
    ----------
    crop_group_number : np.ndarray
        The crop group number is a indicator of adaptation to dry climate,
        Van Diepen et al., 1988: WOFOST 6.0, p.87
    potential_evapotranspiration : np.ndarray
        Potential evapotranspiration in m

    Returns
    -------
    np.ndarray
        The fraction of easily available soil water, p is closer to 0 if evapo is bigger and cropgroup is smaller
    """
    potential_evapotranspiration_cm = potential_evapotranspiration * np.float32(100)

    p = np.float32(1) / (
        np.float32(0.76) + np.float32(1.5) * potential_evapotranspiration_cm
    ) - np.float32(0.1) * (np.float32(5) - crop_group_number)

    # Additional correction for crop groups 1 and 2
    if crop_group_number <= np.float32(2.5):
        p = p + (potential_evapotranspiration_cm - np.float32(0.6)) / (
            crop_group_number * (crop_group_number + np.float32(3.0))
        )

    if p < np.float32(0):
        p = np.float32(0)
    if p > np.float32(1):
        p = np.float32(1)
    return p


def get_critical_water_level(p, wfc, wwp):
    return np.maximum(
        get_critical_soil_moisture_content(p, wfc, wwp) - wwp, np.float32(0)
    )


def get_total_transpiration_factor(
    transpiration_factor_per_layer, root_ratios, soil_layer_height
):
    transpiration_factor_relative_contribution_per_layer = (
        soil_layer_height * root_ratios
    )
    transpiration_factor_total = np.sum(
        transpiration_factor_relative_contribution_per_layer
        / transpiration_factor_relative_contribution_per_layer.sum(axis=0)
        * transpiration_factor_per_layer,
        axis=0,
    )
    return transpiration_factor_total


@njit(cache=True, inline="always", fastmath=True)
def get_transpiration_factor_single(w, wwp, wcrit):
    nominator = w - wwp
    denominator = wcrit - wwp
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


@njit(cache=True, inline="always", fastmath=True)
def get_root_ratios(
    root_depth,
    soil_layer_height,
):
    root_ratios = np.zeros_like(soil_layer_height)
    for i in range(root_depth.size):
        set_root_ratios_single(
            root_depth[i], soil_layer_height[:, i], root_ratios[:, i]
        )
    return root_ratios


@njit(inline="always", fastmath=True)
def set_root_ratios_single(root_depth, soil_layer_height, root_ratios):
    remaining_root_depth = root_depth
    for layer in range(N_SOIL_LAYERS):
        root_ratios[layer] = min(
            remaining_root_depth / soil_layer_height[layer], np.float32(1)
        )
        remaining_root_depth -= soil_layer_height[layer]
        if remaining_root_depth < np.float32(0):
            return root_ratios
    return root_ratios


def get_crop_group_number(
    crop_map, crop_group_numbers, land_use_type, natural_crop_groups
):
    crop_group_map = np.take(crop_group_numbers, crop_map)
    crop_group_map[crop_map == -1] = np.nan

    natural_land = np.isin(land_use_type, (FOREST, GRASSLAND_LIKE))
    crop_group_map[natural_land] = natural_crop_groups[natural_land]
    return crop_group_map


@njit(cache=True, fastmath=True)
def get_unsaturated_hydraulic_conductivity(
    w,
    wres,
    ws,
    lambda_,
    saturated_hydraulic_conductivity,
    minimum_effective_saturation=np.float32(0.0),
):
    """This function calculates the unsaturated hydraulic conductivity based on the soil moisture content
    following van Genuchten (1980) and Mualem (1976)

    See https://archive.org/details/watershedmanagem0000unse_d4j9/page/295/mode/1up?view=theater (p. 295)
    """
    effective_saturation = (w - wres) / (ws - wres)
    if effective_saturation < np.float32(0):
        effective_saturation = np.float32(0)
    elif effective_saturation > np.float32(1):
        effective_saturation = np.float32(1)

    effective_saturation = max(minimum_effective_saturation, effective_saturation)

    n = lambda_ + np.float32(1)
    m = np.float32(1) - np.float32(1) / n

    return (
        saturated_hydraulic_conductivity
        * np.sqrt(effective_saturation)
        * (
            np.float32(1)
            - (np.float32(1) - effective_saturation ** (np.float32(1) / m)) ** m
        )
        ** 2
    )


PERCOLATION_SUBSTEPS = np.int32(3)


@njit(cache=True, parallel=True, fastmath=True)
def get_available_water_infiltration(
    natural_available_water_infiltration,
    actual_irrigation_consumption,
    land_use_type,
    crop_kc,
    EWRef,
    topwater,
    open_water_evaporation,
):
    """
    Update the soil water storage based on the water balance calculations.

    Parameters
    ----------
    wwp : np.ndarray

    Notes
    -----
    This function requires N_SOIL_LAYERS to be defined in the global scope. Which can help
    the compiler to optimize the code better.
    """

    available_water_infiltration = np.zeros_like(land_use_type, dtype=np.float32)
    for i in prange(land_use_type.size):
        available_water_infiltration[i] = (
            natural_available_water_infiltration[i] + actual_irrigation_consumption[i]
        )
        if available_water_infiltration[i] < np.float32(0):
            available_water_infiltration[i] = np.float32(0)
        # paddy irrigated land
        if land_use_type[i] == PADDY_IRRIGATED:
            if crop_kc[i] > np.float32(0.75):
                topwater[i] += available_water_infiltration[i]

            assert EWRef[i] >= np.float32(0)
            open_water_evaporation[i] = min(max(np.float32(0.0), topwater[i]), EWRef[i])
            topwater[i] -= open_water_evaporation[i]
            assert topwater[i] >= np.float32(0)
            if crop_kc[i] > np.float32(0.75):
                available_water_infiltration[i] = topwater[i]
            else:
                available_water_infiltration[i] += topwater[i]
    return available_water_infiltration


@njit(cache=True, parallel=True, fastmath=True)
def rise_from_groundwater(
    w,
    ws,
    capillary_rise_from_groundwater,
):
    bottom_soil_layer_index = N_SOIL_LAYERS - 1
    runoff_from_groundwater = np.zeros_like(
        capillary_rise_from_groundwater, dtype=np.float32
    )

    for i in prange(capillary_rise_from_groundwater.size):
        w[bottom_soil_layer_index, i] += capillary_rise_from_groundwater[
            i
        ]  # add capillar rise to the bottom soil layer

        # if the bottom soil layer is full, send water to the above layer, repeat until top layer
        for j in range(bottom_soil_layer_index, 0, -1):
            if w[j, i] > ws[j, i]:
                w[j - 1, i] += w[j, i] - ws[j, i]  # move excess water to layer above
                w[j, i] = ws[j, i]  # set the current layer to full

        # if the top layer is full, send water to the runoff
        # TODO: Send to topwater instead of runoff if paddy irrigated
        if w[0, i] > ws[0, i]:
            runoff_from_groundwater[i] = (
                w[0, i] - ws[0, i]
            )  # move excess water to runoff
            w[0, i] = ws[0, i]  # set the top layer to full
    return runoff_from_groundwater


@njit(cache=True, parallel=True, fastmath=True)
def evapotranspirate(
    wwp,
    wfc,
    ws,
    wres,
    aeration_days_counter,
    soil_layer_height,
    land_use_type,
    root_depth,
    crop_map,
    natural_crop_groups,
    crop_lag_aeration_days,
    potential_transpiration,
    potential_bare_soil_evaporation,
    potential_evapotranspiration,
    frost_index,
    crop_group_number_per_group,
    w,
    topwater,
    open_water_evaporation,
    available_water_infiltration,
):
    root_ratios_matrix = np.zeros_like(soil_layer_height)
    root_distribution_per_layer_rws_corrected_matrix = np.zeros_like(soil_layer_height)
    root_distribution_per_layer_aeration_stress_corrected_matrix = np.zeros_like(
        soil_layer_height
    )

    is_bioarea = land_use_type < SEALED
    soil_is_frozen = frost_index > FROST_INDEX_THRESHOLD

    actual_total_transpiration = np.zeros_like(land_use_type, dtype=np.float32)
    actual_bare_soil_evaporation = np.zeros_like(land_use_type, dtype=np.float32)

    for i in prange(land_use_type.size):
        remaining_potential_transpiration = potential_transpiration[i]
        if land_use_type[i] == PADDY_IRRIGATED:
            transpiration_from_topwater = min(
                topwater[i], remaining_potential_transpiration
            )
            remaining_potential_transpiration -= transpiration_from_topwater
            topwater[i] -= transpiration_from_topwater
            available_water_infiltration[i] -= transpiration_from_topwater
            actual_total_transpiration[i] += transpiration_from_topwater

        # get group group numbers for natural areas
        if land_use_type[i] == FOREST or land_use_type[i] == GRASSLAND_LIKE:
            crop_group_number = natural_crop_groups[i]
        else:  #
            crop_group_number = crop_group_number_per_group[crop_map[i]]

        p = get_fraction_easily_available_soil_water_single(
            crop_group_number, potential_evapotranspiration[i]
        )

        root_ratios = set_root_ratios_single(
            root_depth[i],
            soil_layer_height[:, i],
            root_ratios_matrix[:, i],
        )

        total_transpiration_factor_water_stress = np.float32(0.0)
        total_aeration_stress = np.float32(0.0)
        total_root_length_rws_corrected = np.float32(
            0.0
        )  # check if same as total_transpiration_factor * root_depth
        total_root_length_aeration_stress_corrected = np.float32(0.0)
        for layer in range(N_SOIL_LAYERS):
            root_length_within_layer = soil_layer_height[layer, i] * root_ratios[layer]

            # Water stress
            critical_soil_moisture_content = get_critical_soil_moisture_content(
                p, wfc[layer, i], wwp[layer, i]
            )
            transpiration_factor = get_transpiration_factor_single(
                w[layer, i], wwp[layer, i], critical_soil_moisture_content
            )

            total_transpiration_factor_water_stress += (
                transpiration_factor
            ) * root_length_within_layer

            root_length_within_layer_rws_corrected = (
                root_length_within_layer * transpiration_factor
            )
            total_root_length_rws_corrected += root_length_within_layer_rws_corrected
            root_distribution_per_layer_rws_corrected_matrix[layer, i] = (
                root_length_within_layer_rws_corrected
            )

            # Aeration stress
            aeration_stress_threshold = get_aeration_stress_threshold(
                ws[layer, i], soil_layer_height[layer, i], np.float32(5)
            )  # 15 is placeholder for crop_aeration_threshold
            if w[layer, i] > aeration_stress_threshold:
                aeration_days_counter[layer, i] += 1
                aeration_stress_factor = get_aeration_stress_factor(
                    aeration_days_counter[layer, i],
                    crop_lag_aeration_days[i],
                    ws[layer, i],
                    w[layer, i],
                    aeration_stress_threshold,
                )
            else:
                # Reset aeration days counter where w <= waer
                aeration_days_counter[layer, i] = 0
                aeration_stress_factor = np.float32(1)  # no stress

            total_aeration_stress += aeration_stress_factor * root_length_within_layer

            root_length_within_layer_aeration_stress_corrected = (
                root_length_within_layer * aeration_stress_factor
            )

            total_root_length_aeration_stress_corrected += (
                root_length_within_layer_aeration_stress_corrected
            )

            root_distribution_per_layer_aeration_stress_corrected_matrix[layer, i] = (
                root_length_within_layer_aeration_stress_corrected
            )

        total_transpiration_factor_water_stress /= root_depth[i]
        total_aeration_stress /= root_depth[i]

        # correct the transpiration reduction factor for water stress
        # if the soil is frozen, no transpiration occurs, so we can skip the loop
        # and thus transpiration is 0 this also avoids division by zero, and thus NaNs
        # likewise, if the total_transpiration_factor (full water stress) is 0
        # or full aeration stress, we can skip the loop
        if (
            not soil_is_frozen[i]
            and total_transpiration_factor_water_stress > np.float32(0)
            and total_aeration_stress > np.float32(0)
        ):
            maximum_transpiration = remaining_potential_transpiration * min(
                total_transpiration_factor_water_stress, total_aeration_stress
            )
            # distribute the transpiration over the layers, considering the root ratios
            # and the transpiration reduction factor per layer
            for layer in range(N_SOIL_LAYERS):
                transpiration_water_stress_corrected = (
                    maximum_transpiration
                    * root_distribution_per_layer_rws_corrected_matrix[layer, i]
                    / total_root_length_rws_corrected
                )
                transpiration_aeration_stress_corrected = (
                    maximum_transpiration
                    * root_distribution_per_layer_aeration_stress_corrected_matrix[
                        layer, i
                    ]
                    / total_root_length_aeration_stress_corrected
                )
                transpiration = min(
                    transpiration_water_stress_corrected,
                    transpiration_aeration_stress_corrected,
                )
                transpiration = transpiration_water_stress_corrected
                w[layer, i] -= transpiration
                w[layer, i] = max(
                    w[layer, i], wres[layer, i]
                )  # soil moisture can never be lower than wres
                if is_bioarea[i]:
                    actual_total_transpiration[i] += transpiration

        if is_bioarea[i]:
            # limit the bare soil evaporation to the available water in the soil
            if not soil_is_frozen[i] and topwater[i] == np.float32(0):
                # TODO: Minor bug, this should only occur when topwater is above 0
                # fix this after completing soil module speedup
                actual_bare_soil_evaporation[i] = min(
                    max(
                        np.float32(0),
                        potential_bare_soil_evaporation[i] - open_water_evaporation[i],
                    ),
                    max(
                        w[0, i] - wres[0, i], np.float32(0)
                    ),  # soil moisture can never be lower than 0
                )
                # remove the bare soil evaporation from the top layer
                w[0, i] -= actual_bare_soil_evaporation[i]
                w[0, i] = max(
                    w[0, i], wres[0, i]
                )  # soil moisture can never be lower than wres
            else:
                # if the soil is frozen, no evaporation occurs
                # if the field is flooded (paddy irrigation), no bare soil evaporation occurs
                actual_bare_soil_evaporation[i] = np.float32(0)

    return (
        actual_total_transpiration,
        actual_bare_soil_evaporation,
    )


@njit(cache=True, parallel=True, fastmath=True)
def vertical_water_transport(
    available_water_infiltration,
    ws,
    wres,
    saturated_hydraulic_conductivity,
    lambda_,
    bubbling_pressure_cm,
    land_use_type,
    frost_index,
    capillary_rise_from_groundwater,
    arno_beta,
    preferential_flow_constant,
    w,
    topwater,
    soil_layer_height,
):
    """
    Parameters
    ----------
    preferential_flow_constant : float
        The preferential flow constant. Because effective saturation is always below 1, a higher
        preferential flow constant will result in less preferential flow.

    Simulates vertical transport of water in the soil using Darcy's equation,
    combining infiltration, percolation, and capillary rise into a single process.
    Considers soil water potential and varying soil layer heights.

    Returns
    -------
    preferential_flow : np.ndarray
        The preferential flow of water through the soil
    direct_runoff : np.ndarray
        The direct runoff of water from the soil
    groundwater_recharge : np.ndarray
        The recharge of groundwater from the soil
    net_fluxes : np.ndarray
        The net fluxes of water between soil layers. Only used for tests.

    """
    # Initialize variables
    preferential_flow = np.zeros_like(land_use_type, dtype=np.float32)
    direct_runoff = np.zeros_like(land_use_type, dtype=np.float32)

    soil_is_frozen = frost_index > FROST_INDEX_THRESHOLD
    net_fluxes = np.zeros(
        (N_SOIL_LAYERS, w.shape[1]), dtype=np.float32
    )  # Fluxes between layers
    delta_z = (soil_layer_height[:-1, :] + soil_layer_height[1:, :]) / 2

    for i in prange(land_use_type.size):
        # Infiltration and preferential flow
        # Estimate the infiltration capacity
        # Use first 2 soil layers to estimate distribution between runoff and infiltration
        soil_water_storage = w[0, i] + w[1, i]
        soil_water_storage_max = ws[0, i] + ws[1, i]
        relative_saturation = soil_water_storage / soil_water_storage_max
        relative_saturation = min(relative_saturation, np.float32(1))

        # Fraction of pixel that is at saturation
        saturated_area_fraction = (
            np.float32(1) - (np.float32(1) - relative_saturation) ** arno_beta[i]
        )
        saturated_area_fraction = max(saturated_area_fraction, np.float32(0))
        saturated_area_fraction = min(saturated_area_fraction, np.float32(1))

        store = soil_water_storage_max / (arno_beta[i] + np.float32(1))
        pot_beta = (arno_beta[i] + np.float32(1)) / arno_beta[i]
        potential_infiltration = store - store * (
            np.float32(1) - (np.float32(1) - saturated_area_fraction) ** pot_beta
        )

        # Preferential flow calculation. Higher preferential flow constant results in less preferential flow
        # because relative saturation is always below 1
        if (
            not soil_is_frozen[i]
            and land_use_type[i] != PADDY_IRRIGATED
            and capillary_rise_from_groundwater[i]
            == np.float32(
                0
            )  # preferential flow only occurs when there is no capillary rise from groundwater
        ):
            preferential_flow[i] = (
                available_water_infiltration[i]
                * relative_saturation**preferential_flow_constant
            )

        # If the soil is frozen, no infiltration occurs
        if soil_is_frozen[i]:
            infiltration = np.float32(0)
        else:
            infiltration = min(
                potential_infiltration,
                available_water_infiltration[i] - preferential_flow[i],
            )

        # add infiltration to the soil
        w[0, i] += infiltration
        # if the top layer is full, send water to the second layer. Since we consider the
        # storage capacity of the first two layers for infiltration, we can assume that
        # the second layer is never full
        if w[0, i] > ws[0, i]:
            overcapacity = w[0, i] - ws[0, i]
            w[1, i] = min(
                w[1, i] + overcapacity, ws[1, i]
            )  # limit by storage capacity of second layer
            w[0, i] = ws[0, i]

        # Runoff and topwater update for paddy fields
        if land_use_type[i] == PADDY_IRRIGATED:
            topwater[i] = max(np.float32(0), topwater[i] - infiltration)
            direct_runoff[i] = max(0, topwater[i] - np.float32(0.05))
            topwater[i] = max(np.float32(0), topwater[i] - direct_runoff[i])
        else:
            direct_runoff[i] = max(
                (available_water_infiltration[i] - infiltration - preferential_flow[i]),
                np.float32(0),
            )

        # Add infiltration flux at the soil surface
        net_fluxes[0, i] = infiltration

    psi = np.zeros_like(net_fluxes)
    K_unsat = np.zeros_like(net_fluxes)

    for i in prange(land_use_type.size):
        # Compute unsaturated hydraulic conductivity and soil water potential
        for layer in range(N_SOIL_LAYERS):
            # Compute unsaturated hydraulic conductivity. Here it is important that some flow is always possible.
            # Therefore we use a minimum effective saturation to ensure that some flow is always possible.
            # This is something that could be better paremeterized, especially when looking at flood-drought
            K_unsat[layer, i] = get_unsaturated_hydraulic_conductivity(
                w=w[layer, i],
                wres=wres[layer, i],
                ws=ws[layer, i],
                lambda_=lambda_[layer, i],
                saturated_hydraulic_conductivity=saturated_hydraulic_conductivity[
                    layer, i
                ],
                minimum_effective_saturation=np.float32(
                    0.01
                ),  # this could be better defined when looking at flood-drought interactions
            )

            # Compute soil water potential
            psi[layer, i] = get_soil_water_potential(
                theta=w[layer, i],
                thetar=wres[layer, i],
                thetas=ws[layer, i],
                lambda_=lambda_[layer, i],
                bubbling_pressure_cm=bubbling_pressure_cm[layer, i],
                minimum_effective_saturation=np.float32(
                    0.01
                ),  # this could be better defined when looking at flood-drought interactions
            )

    groundwater_recharge = np.zeros_like(land_use_type, dtype=np.float32)
    is_bioarea = land_use_type < SEALED

    for i in prange(land_use_type.size):
        # Compute fluxes between layers using Darcy's law
        for layer in range(N_SOIL_LAYERS):  # From top (0) to bottom (N_SOIL_LAYERS)
            if layer == N_SOIL_LAYERS - 1:
                # If there is capillary rise from groundwater, there will be no
                # percolation to the groundwater. A potential capillary rise from
                # the groundwater is already accounted for in rise_from_groundwater
                if capillary_rise_from_groundwater[i] > np.float32(0):
                    flux = np.float32(0)
                else:
                    # Else we assume that the bottom layer is draining under gravity
                    # i.e., assuming homogeneous soil water potential below
                    # bottom layer all the way to groundwater
                    flux = K_unsat[layer, i]  # Assume draining under gravity
                    available_water_source = w[layer, i] - wres[layer, i]
                    flux = min(flux, available_water_source)
                    w[layer, i] -= flux
            else:
                # Taking the mean of the hydraulic conductivities
                # by using the geometric mean of the conductivities we put a bit more
                # weight on the lower layer with lower conductivity
                K_unsat_avg = np.sqrt(K_unsat[layer + 1, i] * K_unsat[layer, i])

                # Compute flux using Darcy's law
                flux = -K_unsat_avg * (
                    (psi[layer + 1, i] - psi[layer, i]) / delta_z[layer, i]
                    - np.float32(1)
                )

                if flux >= 0:  # Downward flux (percolation)
                    positive_flux = flux
                    source = layer
                    sink = layer + 1
                else:  # Upward flux (capillary rise)
                    positive_flux = -flux
                    source = layer + 1
                    sink = layer

                # Limit flux by available water in source and storage capacity of sink
                remaining_storage_capacity_sink = ws[sink, i] - w[sink, i]
                available_water_source = w[source, i] - wres[source, i]
                positive_flux = min(
                    positive_flux,
                    remaining_storage_capacity_sink,
                    available_water_source,
                )

                w[source, i] -= positive_flux
                w[sink, i] += positive_flux

            net_fluxes[layer, i] = flux

            # Due to numerical errors, the water content in the sink layer can exceed the storage capacity
            # and the source layer can fall below the residual water storage. Thus cap these to the
            # storage capacity and residual water storage, respectively
            w[sink, i] = min(w[sink, i], ws[sink, i])
            w[source, i] = max(w[source, i], wres[source, i])

        if is_bioarea[i]:
            groundwater_recharge[i] = net_fluxes[-1, i] + preferential_flow[i]

    return preferential_flow, direct_runoff, groundwater_recharge, net_fluxes


class Soil(object):
    def __init__(self, model, elevation_std):
        """

        Notes:
        - We don't consider that bedrock does not impede percolation in line with https://doi.org/10.1029/2018WR022920
        which states that connection to the stream is the exception rather than the rule
        A better implementation would be to consider travel distance. But this remains a topic for future work.
        """
        self.var = model.data.HRU
        self.model = model

        self.soil_layer_height = self.model.data.grid.load(
            self.model.files["grid"]["soil/soil_layer_height"],
            layer=None,
        )
        self.soil_layer_height = self.model.data.to_HRU(
            data=self.soil_layer_height, fn=None
        )

        # set number of soil layers as global variable for numba
        global N_SOIL_LAYERS
        N_SOIL_LAYERS = self.soil_layer_height.shape[0]

        # set the frost index threshold as global variable for numba
        global FROST_INDEX_THRESHOLD
        FROST_INDEX_THRESHOLD = np.float32(self.var.frost_indexThreshold)

        # θ saturation, field capacity, wilting point and residual moisture content
        thetas = self.model.data.grid.load(
            self.model.files["grid"]["soil/thetas"], layer=None
        )
        thetas = self.model.data.to_HRU(data=thetas, fn=None)
        thetar = self.model.data.grid.load(
            self.model.files["grid"]["soil/thetar"], layer=None
        )
        thetar = self.model.data.to_HRU(data=thetar, fn=None)

        bubbling_pressure_cm = self.model.data.grid.load(
            self.model.files["grid"]["soil/bubbling_pressure_cm"], layer=None
        )
        self.bubbling_pressure_cm = self.model.data.to_HRU(
            data=bubbling_pressure_cm, fn=None
        )

        lambda_pore_size_distribution = self.model.data.grid.load(
            self.model.files["grid"]["soil/lambda"], layer=None
        )
        lambda_pore_size_distribution = self.model.data.to_HRU(
            data=lambda_pore_size_distribution, fn=None
        )

        thetafc = get_soil_moisture_at_pressure(
            -100,  # assuming field capacity is at -100 cm (pF 2)
            self.bubbling_pressure_cm,
            thetas,
            thetar,
            lambda_pore_size_distribution,
        )

        thetawp = get_soil_moisture_at_pressure(
            -(10**4.2),  # assuming wilting point is at -10^4.2 cm (pF 4.2)
            self.bubbling_pressure_cm,
            thetas,
            thetar,
            lambda_pore_size_distribution,
        )

        self.ws = thetas * self.soil_layer_height
        self.wfc = thetafc * self.soil_layer_height
        self.wwp = thetawp * self.soil_layer_height
        self.wres = thetar * self.soil_layer_height

        # initial soil water storage between field capacity and wilting point
        self.var.w = self.model.data.HRU.load_initial(
            "w",
            default=(self.wfc + self.wwp) / 2,
        )
        # for paddy irrigation flooded paddy fields
        self.var.topwater = self.model.data.HRU.load_initial(
            "topwater", default=self.var.full_compressed(0, dtype=np.float32)
        )

        lambda_pore_size_distribution = self.model.data.grid.load(
            self.model.files["grid"]["soil/lambda"], layer=None
        )
        self.lambda_pore_size_distribution = self.model.data.to_HRU(
            data=lambda_pore_size_distribution, fn=None
        )
        ksat = self.model.data.grid.load(
            self.model.files["grid"]["soil/hydraulic_conductivity"], layer=None
        )
        self.ksat = self.model.data.to_HRU(data=ksat, fn=None)

        # soil water depletion fraction, Van Diepen et al., 1988: WOFOST 6.0, p.86, Doorenbos et. al 1978
        # crop groups for formular in van Diepen et al, 1988
        self.natural_crop_groups = self.model.data.grid.load(
            self.model.files["grid"]["soil/cropgrp"]
        )
        self.natural_crop_groups = self.model.data.to_HRU(data=self.natural_crop_groups)

        # ------------ Preferential Flow constant ------------------------------------------
        self.preferential_flow_constant = np.float32(
            self.model.config["parameters"]["preferentialFlowConstant"]
        )

        self.var.arnoBeta = self.var.full_compressed(np.nan, dtype=np.float32)

        # Improved Arno's scheme parameters: Hageman and Gates 2003
        # arnoBeta defines the shape of soil water capacity distribution curve as a function of  topographic variability
        # b = max( (oh - o0)/(oh + omax), 0.01)
        # oh: the standard deviation of orography, o0: minimum std dev, omax: max std dev
        arnoBetaOro = (elevation_std - 10.0) / (elevation_std + 1500.0)

        arnoBetaOro += self.model.config["parameters"][
            "arnoBeta_add"
        ]  # calibration parameter
        arnoBetaOro = np.clip(arnoBetaOro, 0.01, 1.2)

        arnobeta_cover_types = {
            FOREST: 0.2,
            GRASSLAND_LIKE: 0.0,
            PADDY_IRRIGATED: 0.2,
            NON_PADDY_IRRIGATED: 0.2,
        }

        for cover, arno_beta in arnobeta_cover_types.items():
            land_use_indices = np.where(self.var.land_use_type == cover)[0]

            self.var.arnoBeta[land_use_indices] = (arnoBetaOro + arno_beta)[
                land_use_indices
            ]
            self.var.arnoBeta[land_use_indices] = np.minimum(
                1.2, np.maximum(0.01, self.var.arnoBeta[land_use_indices])
            )

        self.var.aeration_days_counter = self.var.load_initial(
            "aeration_days_counter",
            default=np.full(
                (N_SOIL_LAYERS, self.var.compressed_size), 0, dtype=np.int32
            ),
        )
        self.crop_lag_aeration_days = np.full_like(
            self.var.land_use_type, 3, dtype=np.int32
        )

        def create_ini(yaml, idx, plantFATE_cluster, biodiversity_scenario):
            out_dir = self.model.simulation_root / "plantFATE" / f"cell_{idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            ini_file = out_dir / "p_daily.ini"

            yaml["> STRINGS"]["outDir"] = out_dir
            if self.model.spinup is True:
                original_state_file = (
                    Path("input")
                    / "plantFATE_initialization"
                    / biodiversity_scenario
                    / f"cluster_{plantFATE_cluster}"
                    / "pf_saved_state.txt"
                )
                assert original_state_file.exists()
                new_state_file = out_dir / "pf_saved_state_initialization.txt"
                with open(original_state_file, "r") as original_f:
                    state = original_f.read()
                    timetuple = self.model.current_time.timetuple()
                    year = timetuple.tm_year
                    day_of_year = timetuple.tm_yday
                    state = state.replace(
                        "6 2 0 2000 1 0 0",
                        f"6 2 0 {year + (day_of_year - 1) / 365} 1 0 0",
                    )
                    with open(new_state_file, "w") as new_f:
                        new_f.write(state)

                yaml["> STRINGS"]["continueFromState"] = new_state_file
                yaml["> STRINGS"]["continueFromConfig"] = ini_file
                yaml["> STRINGS"]["saveState"] = True
                yaml["> STRINGS"]["savedStateFile"] = "pf_saved_state_spinup.txt"
                yaml["> STRINGS"]["savedConfigFile"] = "pf_saved_config_spinup.txt"
            else:
                yaml["> STRINGS"]["continueFromState"] = (
                    out_dir
                    / self.model.config["plantFATE"]["> STRINGS"]["exptName"]
                    / "pf_saved_state_spinup.txt"
                )
                yaml["> STRINGS"]["continueFromConfig"] = ini_file
                yaml["> STRINGS"]["savedStateFile"] = None
                yaml["> STRINGS"]["saveState"] = False
                yaml["> STRINGS"]["savedConfigFile"] = None

            with open(ini_file, "w") as f:
                for section, section_dict in yaml.items():
                    f.write(section + "\n")
                    if section_dict is None:
                        continue
                    for key, value in section_dict.items():
                        if value is None:
                            value = "null"
                        elif value is False:
                            value = "no"
                        elif value is True:
                            value = "yes"
                        f.write(key + " " + str(value) + "\n")
            return ini_file

        if self.model.config["general"]["simulate_forest"]:
            plantFATE_cluster = 7
            biodiversity_scenario = "low"

            lon, lat = 73.5975501619, 19.1444726274

            from honeybees.library.raster import coord_to_pixel

            px, py = coord_to_pixel(np.array([lon, lat]), gt=self.model.data.grid.gt)

            cell_ids = np.arange(self.model.data.grid.compressed_size)
            cell_ids_map = self.model.data.grid.decompress(cell_ids, fillvalue=-1)
            cell_id = cell_ids_map[py, px]

            already_has_plantFATE_cell = False
            from . import plantFATE

            self.model.plantFATE = []
            self.plantFATE_forest_RUs = np.zeros_like(
                self.var.land_use_type, dtype=bool
            )
            for i, land_use_type_RU in enumerate(self.var.land_use_type):
                grid_cell = self.var.HRU_to_grid[i]
                # if land_use_type_RU == 0 and self.var.land_use_ratio[i] > 0.5:
                if land_use_type_RU == FOREST and grid_cell == cell_id:
                    if already_has_plantFATE_cell:
                        self.model.plantFATE.append(None)
                    else:
                        self.plantFATE_forest_RUs[i] = True

                        ini_path = create_ini(
                            self.model.config["plantFATE"],
                            i,
                            plantFATE_cluster,
                            biodiversity_scenario,
                        )
                        already_has_plantFATE_cell = True
                        self.model.plantFATE.append(plantFATE.Model(ini_path))
                else:
                    self.model.plantFATE.append(None)

    def calculate_soil_water_potential_MPa(
        self,
        soil_moisture,  # [m]
        soil_moisture_wilting_point,  # [m]
        soil_moisture_field_capacity,  # [m]
        soil_tickness,  # [m]
        wilting_point=-1500,  # kPa
        field_capacity=-33,  # kPa
    ):
        # https://doi.org/10.1016/B978-0-12-374460-9.00007-X (eq. 7.16)
        soil_moisture_fraction = soil_moisture / soil_tickness
        # assert (soil_moisture_fraction >= 0).all() and (soil_moisture_fraction <= 1).all()
        del soil_moisture
        soil_moisture_wilting_point_fraction = (
            soil_moisture_wilting_point / soil_tickness
        )
        # assert (soil_moisture_wilting_point_fraction).all() >= 0 and (
        #     soil_moisture_wilting_point_fraction
        # ).all() <= 1
        del soil_moisture_wilting_point
        soil_moisture_field_capacity_fraction = (
            soil_moisture_field_capacity / soil_tickness
        )
        # assert (soil_moisture_field_capacity_fraction >= 0).all() and (
        #     soil_moisture_field_capacity_fraction <= 1
        # ).all()
        del soil_moisture_field_capacity

        n_potential = -(
            np.log(wilting_point / field_capacity)
            / np.log(
                soil_moisture_wilting_point_fraction
                / soil_moisture_field_capacity_fraction
            )
        )
        # assert (n_potential >= 0).all()
        a_potential = 1.5 * 10**6 * soil_moisture_wilting_point_fraction**n_potential
        # assert (a_potential >= 0).all()
        soil_water_potential = -a_potential * soil_moisture_fraction ** (-n_potential)
        return soil_water_potential / 1_000_000  # Pa to MPa

    def calculate_vapour_pressure_deficit_kPa(self, temperature_K, relative_humidity):
        temperature_C = temperature_K - 273.15
        assert (
            temperature_C < 100
        ).all()  # temperature is in Celsius. So on earth should be well below 100.
        assert (
            temperature_C > -100
        ).all()  # temperature is in Celsius. So on earth should be well above -100.
        assert (
            relative_humidity >= 1
        ).all()  # below 1 is so rare that it shouldn't be there at the resolutions of current climate models, and this catches errors with relative_humidity as a ratio [0-1].
        assert (
            relative_humidity <= 100
        ).all()  # below 1 is so rare that it shouldn't be there at the resolutions of current climate models, and this catches errors with relative_humidity as a ratio [0-1].
        # https://soilwater.github.io/pynotes-agriscience/notebooks/vapor_pressure_deficit.html
        saturated_vapour_pressure = 0.611 * np.exp(
            (17.502 * temperature_C) / (temperature_C + 240.97)
        )  # kPa
        actual_vapour_pressure = (
            saturated_vapour_pressure * relative_humidity / 100
        )  # kPa
        vapour_pressure_deficit = saturated_vapour_pressure - actual_vapour_pressure
        return vapour_pressure_deficit

    def calculate_photosynthetic_photon_flux_density(self, shortwave_radiation, xi=0.5):
        # https://search.r-project.org/CRAN/refmans/bigleaf/html/Rg.to.PPFD.html
        photosynthetically_active_radiation = shortwave_radiation * xi
        photosynthetic_photon_flux_density = (
            photosynthetically_active_radiation * 4.6
        )  #  W/m2 -> umol/m2/s
        return photosynthetic_photon_flux_density

    def step(
        self,
        capillary_rise_from_groundwater,
        open_water_evaporation,
        potential_transpiration,
        potential_bare_soil_evaporation,
        potential_evapotranspiration,
    ):
        """
        Dynamic part of the soil module

        For each of the land cover classes the vertical water transport is simulated
        Distribution of water holding capiacity in 3 soil layers based on saturation excess overland flow, preferential flow
        Dependend on soil depth, soil hydraulic parameters
        """

        if __debug__:
            w_pre = self.var.w.copy()
            topwater_pre = self.var.topwater.copy()

        bioarea = np.where(self.var.land_use_type < SEALED)[0].astype(np.int32)

        interflow = self.var.full_compressed(0, dtype=np.float32)

        timer = TimingModule("Soil")

        available_water_infiltration = get_available_water_infiltration(
            natural_available_water_infiltration=self.var.natural_available_water_infiltration,
            actual_irrigation_consumption=self.var.actual_irrigation_consumption,
            land_use_type=self.var.land_use_type,
            crop_kc=self.var.cropKC,
            EWRef=self.var.EWRef,
            topwater=self.var.topwater,
            open_water_evaporation=open_water_evaporation,
        )

        timer.new_split("Available infiltration")

        assert (self.var.w[:, bioarea] <= self.ws[:, bioarea]).all()
        assert (self.var.w[:, bioarea] >= self.wres[:, bioarea]).all()

        runoff_from_groundwater = rise_from_groundwater(
            w=self.var.w,
            ws=self.ws,
            capillary_rise_from_groundwater=capillary_rise_from_groundwater.astype(
                np.float32
            ),
        )

        assert (self.var.w[:, bioarea] <= self.ws[:, bioarea]).all()
        assert (self.var.w[:, bioarea] >= self.wres[:, bioarea]).all()

        timer.new_split("Capillary rise from groundwater")

        (
            actual_total_transpiration,
            actual_bare_soil_evaporation,
        ) = evapotranspirate(
            wwp=self.wwp,
            wfc=self.wfc,
            ws=self.ws,
            wres=self.wres,
            aeration_days_counter=self.var.aeration_days_counter,
            soil_layer_height=self.soil_layer_height,
            land_use_type=self.var.land_use_type,
            root_depth=self.var.root_depth,
            crop_map=self.var.crop_map,
            natural_crop_groups=self.natural_crop_groups,
            crop_lag_aeration_days=self.crop_lag_aeration_days,
            potential_transpiration=potential_transpiration,
            potential_bare_soil_evaporation=potential_bare_soil_evaporation,
            potential_evapotranspiration=potential_evapotranspiration,
            frost_index=self.var.frost_index,
            crop_group_number_per_group=self.model.agents.crop_farmers.crop_data[
                "crop_group_number"
            ].values.astype(np.float32),
            w=self.var.w,
            topwater=self.var.topwater,
            open_water_evaporation=open_water_evaporation,
            available_water_infiltration=available_water_infiltration,
        )
        assert actual_total_transpiration.dtype == np.float32

        timer.new_split("Evapotranspiration")

        n_substeps = 3
        preferential_flow = np.zeros_like(self.var.land_use_type, dtype=np.float32)
        direct_runoff = np.zeros_like(self.var.land_use_type, dtype=np.float32)
        groundwater_recharge = np.zeros_like(self.var.land_use_type, dtype=np.float32)

        assert (self.var.w[:, bioarea] <= self.ws[:, bioarea]).all()
        assert (self.var.w[:, bioarea] >= self.wres[:, bioarea]).all()

        for _ in range(n_substeps):
            (
                preferential_flow_substep,
                direct_runoff_substep,
                groundwater_recharge_substep,
                _,
            ) = vertical_water_transport(
                available_water_infiltration / n_substeps,
                self.ws,
                self.wres,
                self.ksat / n_substeps,
                self.lambda_pore_size_distribution,
                self.bubbling_pressure_cm,
                self.var.land_use_type,
                self.var.frost_index,
                capillary_rise_from_groundwater,
                self.var.arnoBeta,
                self.preferential_flow_constant,
                self.var.w,
                self.var.topwater,
                self.soil_layer_height,
            )

            preferential_flow += preferential_flow_substep
            direct_runoff += direct_runoff_substep
            groundwater_recharge += groundwater_recharge_substep

        assert (self.var.w[:, bioarea] <= self.ws[:, bioarea]).all()
        assert (self.var.w[:, bioarea] >= self.wres[:, bioarea]).all()

        runoff = direct_runoff + runoff_from_groundwater

        timer.new_split("Vertical transport")

        assert preferential_flow.dtype == np.float32
        assert runoff.dtype == np.float32

        self.var.actual_evapotranspiration[bioarea] += (
            actual_bare_soil_evaporation[bioarea]
            + open_water_evaporation[bioarea]
            + actual_total_transpiration[bioarea]
        )

        if __debug__:
            assert (self.var.w[:, bioarea] <= self.ws[:, bioarea]).all()
            assert (self.var.w[:, bioarea] >= self.wres[:, bioarea]).all()
            assert (interflow == 0).all()  # interflow is not implemented (see above)
            balance_check(
                name="soil_1",
                how="cellwise",
                influxes=[
                    self.var.natural_available_water_infiltration[bioarea],
                    capillary_rise_from_groundwater[bioarea],
                    self.var.actual_irrigation_consumption[bioarea],
                ],
                outfluxes=[
                    runoff[bioarea],
                    interflow[bioarea],
                    groundwater_recharge[bioarea],
                    actual_total_transpiration[bioarea],
                    actual_bare_soil_evaporation[bioarea],
                    open_water_evaporation[bioarea],
                ],
                prestorages=[
                    w_pre[:, bioarea].sum(axis=0),
                    topwater_pre[bioarea],
                ],
                poststorages=[
                    self.var.w[:, bioarea].sum(axis=0),
                    self.var.topwater[bioarea],
                ],
                tollerance=1e-6,
            )

            balance_check(
                name="soil_2",
                how="cellwise",
                influxes=[
                    self.var.natural_available_water_infiltration[bioarea],
                    capillary_rise_from_groundwater[bioarea],
                    self.var.actual_irrigation_consumption[bioarea],
                    self.var.snowEvap[bioarea],
                    self.var.interceptEvap[bioarea],
                ],
                outfluxes=[
                    runoff[bioarea],
                    interflow[bioarea],
                    groundwater_recharge[bioarea],
                    self.var.actual_evapotranspiration[bioarea],
                ],
                prestorages=[
                    w_pre[:, bioarea].sum(axis=0),
                    topwater_pre[bioarea],
                ],
                poststorages=[
                    self.var.w[:, bioarea].sum(axis=0),
                    self.var.topwater[bioarea],
                ],
                tollerance=1e-6,
            )

            assert (
                actual_total_transpiration[bioarea]
                <= potential_transpiration[bioarea] + 1e-7
            ).all()
            assert (
                actual_bare_soil_evaporation[bioarea]
                <= potential_bare_soil_evaporation[bioarea] + 1e-7
            ).all()

        timer.new_split("Finalizing")
        if self.model.timing:
            print(timer)

        return (
            interflow,
            direct_runoff,
            groundwater_recharge,
            open_water_evaporation,
            actual_total_transpiration,
            actual_bare_soil_evaporation,
        )
