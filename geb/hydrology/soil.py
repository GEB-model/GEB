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
from numba import njit, prange, float32

from geb.HRUs import load_grid
from .landcover import (
    FOREST,
    GRASSLAND_LIKE,
    PADDY_IRRIGATED,
    NON_PADDY_IRRIGATED,
    SEALED,
)


@njit(
    cache=True,
    inline="always",
)
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


@njit(cache=True, inline="always")
def get_aeration_stress_threshold(
    ws, soil_layer_height, crop_aeration_stress_threshold
):
    max_saturation_fraction = ws / soil_layer_height
    # Water storage in root zone at aeration stress threshold (m)
    return (
        max_saturation_fraction - (crop_aeration_stress_threshold / np.float32(100))
    ) * soil_layer_height


@njit(cache=True, inline="always")
def get_aeration_stress_factor(
    aeration_days_counter, crop_lag_aeration_days, ws, w, aeration_stress_threshold
):
    if aeration_days_counter < crop_lag_aeration_days:
        stress = np.float32(1) - ((ws - w) / (ws - aeration_stress_threshold))
        aeration_stress_factor = np.float32(1) - ((aeration_days_counter / 3) * stress)
    else:
        aeration_stress_factor = (ws - w) / (ws - aeration_stress_threshold)
    return aeration_stress_factor


@njit(cache=True, inline="always")
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
    crop_group_number, potential_evapotranspiration
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
    crop_group_number, potential_evapotranspiration
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


@njit(cache=True, inline="always")
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


@njit(cache=True, inline="always")
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


@njit(cache=True, inline="always")
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


PERCOLATION_SUBSTEPS = np.int32(3)


@njit(cache=True, parallel=True)
def get_available_water_infiltration(
    natural_available_water_infiltration,
    actual_irrigation_consumption,
    land_use_type,
    crop_kc,
    EWRef,
    topwater,
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
    open_water_evaporation = np.zeros_like(land_use_type, dtype=np.float32)
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

            open_water_evaporation[i] = min(max(np.float32(0.0), topwater[i]), EWRef[i])
            topwater[i] -= open_water_evaporation[i]
            if crop_kc[i] > np.float32(0.75):
                available_water_infiltration[i] = topwater[i]
            else:
                available_water_infiltration[i] += topwater[i]
    return available_water_infiltration, open_water_evaporation


@njit(cache=True, parallel=True)
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


@njit(cache=True, parallel=True)
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


@njit(
    (
        float32[:],
        float32[:],
        float32[:],
        float32[:],
        float32[:],
        float32[:],
    ),
    parallel=True,
    fastmath=False,
    inline="always",
    cache=True,
)
def get_soil_water_flow_parameters(
    w,
    wres,
    ws,
    lambda_,
    saturated_hydraulic_conductivity,
    bubbling_pressure_cm,
):
    psi = np.empty_like(w)
    K_unsat = np.empty_like(w)

    for i in prange(w.shape[0]):
        # Compute unsaturated hydraulic conductivity and soil water potential. Here it is important that
        # some flow is always possible. Therefore we use a minimum effective saturation to ensure that
        # some flow is always possible. This is something that could be better paremeterized,
        # especially when looking at flood-drought

        # Compute effective saturation
        effective_saturation = (w[i] - wres[i]) / (ws[i] - wres[i])
        effective_saturation = np.maximum(effective_saturation, np.float32(0.01))
        effective_saturation = np.minimum(effective_saturation, np.float32(1))

        # Compute parameters n and m
        n = lambda_[i] + np.float32(1)
        m = np.float32(1) - np.float32(1) / n

        # Compute unsaturated hydraulic conductivity
        term1 = saturated_hydraulic_conductivity[i] * np.sqrt(effective_saturation)
        term2 = (
            np.float32(1)
            - np.power(
                (np.float32(1) - np.power(effective_saturation, (np.float32(1) / m))),
                m,
            )
        ) ** 2

        K_unsat[i] = term1 * term2

        alpha = np.float32(1) / bubbling_pressure_cm[i]

        # Compute capillary pressure head (phi)
        phi_power_term = np.power(effective_saturation, (-np.float32(1) / m))
        phi = (
            np.power(phi_power_term - np.float32(1), (np.float32(1) / n)) / alpha
        )  # Positive value

        # Soil water potential (negative value for suction)
        psi[i] = -phi

    return psi, K_unsat


# Do NOT use fastmath here. This leads to unexpected behaviour with NaNs
@njit(cache=True, parallel=True, fastmath=False)
def vertical_water_transport(
    available_water_infiltration,
    capillary_rise_from_groundwater,
    ws,
    wres,
    saturated_hydraulic_conductivity,
    lambda_,
    bubbling_pressure_cm,
    land_use_type,
    frost_index,
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

    """
    # Initialize variables
    preferential_flow = np.zeros_like(land_use_type, dtype=np.float32)
    direct_runoff = np.zeros_like(land_use_type, dtype=np.float32)

    soil_is_frozen = frost_index > FROST_INDEX_THRESHOLD
    delta_z = (soil_layer_height[:-1, :] + soil_layer_height[1:, :]) / 2

    potential_infiltration = np.zeros_like(land_use_type, dtype=np.float32)
    relative_saturation = np.zeros_like(land_use_type, dtype=np.float32)
    for i in prange(land_use_type.size):
        # Infiltration and preferential flow
        # Estimate the infiltration capacity
        # Use first 2 soil layers to estimate distribution between runoff and infiltration
        soil_water_storage = w[0, i] + w[1, i]
        soil_water_storage_max = ws[0, i] + ws[1, i]
        relative_saturation[i] = min(
            soil_water_storage / soil_water_storage_max, np.float32(1)
        )

        # Fraction of pixel that is at saturation
        saturated_area_fraction = (
            np.float32(1) - (np.float32(1) - relative_saturation[i]) ** arno_beta[i]
        )
        saturated_area_fraction = max(saturated_area_fraction, np.float32(0))
        saturated_area_fraction = min(saturated_area_fraction, np.float32(1))

        store = soil_water_storage_max / (arno_beta[i] + np.float32(1))
        pot_beta = (arno_beta[i] + np.float32(1)) / arno_beta[i]
        potential_infiltration[i] = store - store * (
            np.float32(1) - (np.float32(1) - saturated_area_fraction) ** pot_beta
        )

    for i in prange(land_use_type.size):
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
                * relative_saturation[i] ** preferential_flow_constant
            )

    for i in prange(land_use_type.size):
        # If the soil is frozen, no infiltration occurs
        infiltration = min(
            potential_infiltration[i] * ~soil_is_frozen[i],
            available_water_infiltration[i] - preferential_flow[i],
        )

        # add infiltration to the soil
        w[0, i] += infiltration
        # if the top layer is full, send water to the second layer. Since we consider the
        # storage capacity of the first two layers for infiltration, we can assume that
        # the second layer is never full
        overcapacity = w[0, i] - ws[0, i]
        if overcapacity > np.float32(0):
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

    psi, K_unsat = get_soil_water_flow_parameters(
        w.ravel(),
        wres.ravel(),
        ws.ravel(),
        lambda_.ravel(),
        saturated_hydraulic_conductivity.ravel(),
        bubbling_pressure_cm.ravel(),
    )
    psi = psi.reshape((N_SOIL_LAYERS, land_use_type.size))
    K_unsat = K_unsat.reshape((N_SOIL_LAYERS, land_use_type.size))

    groundwater_recharge = np.zeros_like(land_use_type, dtype=np.float32)

    for i in prange(land_use_type.size):
        # Compute fluxes between layers using Darcy's law
        for layer in range(
            N_SOIL_LAYERS - 1
        ):  # From top (0) to bottom (N_SOIL_LAYERS - 1)
            # Compute the geometric mean of the conductivities
            K_unsat_avg = np.sqrt(K_unsat[layer + 1, i] * K_unsat[layer, i])

            # Compute flux using Darcy's law. The -1 accounts for gravity.
            flux = -K_unsat_avg * (
                (psi[layer + 1, i] - psi[layer, i]) / delta_z[layer, i] - np.float32(1)
            )

            # Determine the positive flux and source/sink layers without if statements
            positive_flux = abs(flux)
            flux_direction = flux >= 0  # 1 if flux >= 0, 0 if flux < 0
            source = layer + (
                1 - flux_direction
            )  # layer if flux >= 0, layer + 1 if flux < 0
            sink = layer + flux_direction  # layer + 1 if flux >= 0, layer if flux < 0

            # Limit flux by available water in source and storage capacity of sink
            remaining_storage_capacity_sink = ws[sink, i] - w[sink, i]
            available_water_source = w[source, i] - wres[source, i]
            positive_flux = min(
                positive_flux, remaining_storage_capacity_sink, available_water_source
            )

            # Update water content in source and sink layers
            w[source, i] -= positive_flux
            w[sink, i] += positive_flux

            # Ensure water content stays within physical bounds
            w[sink, i] = min(w[sink, i], ws[sink, i])
            w[source, i] = max(w[source, i], wres[source, i])

        # for the last layer, we assume that the bottom layer is draining under gravity
        layer = N_SOIL_LAYERS - 1

        # We assume that the bottom layer is draining under gravity
        # i.e., assuming homogeneous soil water potential below
        # bottom layer all the way to groundwater
        # Assume draining under gravity. If there is capillary rise from groundwater, there will be no
        # percolation to the groundwater. A potential capillary rise from
        # the groundwater is already accounted for in rise_from_groundwater
        flux = K_unsat[layer, i] * (capillary_rise_from_groundwater[i] <= np.float32(0))
        available_water_source = w[layer, i] - wres[layer, i]
        flux = min(flux, available_water_source)
        w[layer, i] -= flux
        w[layer, i] = max(w[layer, i], wres[layer, i])

        groundwater_recharge[i] = flux + preferential_flow[i]

    return preferential_flow, direct_runoff, groundwater_recharge


def thetas_toth(soil_organic_carbon, bulk_density, is_top_soil, clay, silt):
    """
    Determine saturated water content [m3/m3].

    Based on:
    Tóth, B., Weynants, M., Nemes, A., Makó, A., Bilas, G., and Tóth, G.:
    New generation of hydraulic pedotransfer functions for Europe, Eur. J.
    Soil Sci., 66, 226-238. doi: 10.1111/ejss.121921211, 2015.

    Parameters
    ----------
    bdod : float
        bulk density [g /cm3].
    sand: float
        sand percentage [%].
    silt: float
        silt percentage [%].
    is_top_soil: bool
        top soil flag.

    Returns
    -------
    thetas : float
        saturated water content [cm3/cm3].

    """
    return (
        0.6819
        - 0.06480 * (1 / (soil_organic_carbon + 1))
        - 0.11900 * bulk_density**2
        - 0.02668 * is_top_soil
        + 0.001489 * clay
        + 0.0008031 * silt
        + 0.02321 * (1 / (soil_organic_carbon + 1)) * bulk_density**2
        + 0.01908 * bulk_density**2 * is_top_soil
        - 0.0011090 * clay * is_top_soil
        - 0.00002315 * silt * clay
        - 0.0001197 * silt * bulk_density**2
        - 0.0001068 * clay * bulk_density**2
    )


def thetar_brakensiek(sand, clay, thetas):
    """
    Determine residual water content [m3/m3].

    Thetas is equal to porosity (Φ) in this case.

    Equation found in https://archive.org/details/watershedmanagem0000unse_d4j9/page/294/mode/1up (p. 294)

    Based on:
        Brakensiek, D.L., Rawls, W.J.,and Stephenson, G.R.: Modifying scs hydrologic
        soil groups and curve numbers for range land soils, ASAE Paper no. PNR-84-203,
        St. Joseph, Michigan, USA, 1984.

    Parameters
    ----------
    sand: float
        sand percentage [%].
    clay: float
        clay percentage [%].
    thetas : float
        saturated water content [m3/m3].

    Returns
    -------
    thetar : float
        residual water content [m3/m3].

    """
    return (
        -0.0182482
        + 0.00087269 * sand
        + 0.00513488 * clay
        + 0.02939286 * thetas
        - 0.00015395 * clay**2
        - 0.0010827 * sand * thetas
        - 0.00018233 * clay**2 * thetas**2
        + 0.00030703 * clay**2 * thetas
        - 0.0023584 * thetas**2 * clay
    )


def get_bubbling_pressure(clay, sand, thetas):
    bubbling_pressure = np.exp(
        5.3396738
        + 0.1845038 * clay
        - 2.48394546 * thetas
        - 0.00213853 * clay**2
        - 0.0435649 * sand * thetas
        - 0.61745089 * clay * thetas
        - 0.00001282 * sand**2 * clay
        + 0.00895359 * clay**2 * thetas
        - 0.00072472 * sand**2 * thetas
        + 0.0000054 * sand * clay**2
        + 0.00143598 * sand**2 * thetas**2
        - 0.00855375 * clay**2 * thetas**2
        + 0.50028060 * thetas**2 * clay
    )
    return bubbling_pressure


def get_pore_size_index_brakensiek(sand, thetas, clay):
    """
    Determine Brooks-Corey pore size distribution index [-].

    Thetas is equal to porosity (Φ) in this case.

    Based on:
    Rawls,W. J., and Brakensiek, D. L.: Estimation of SoilWater Retention and
    Hydraulic Properties, In H. J. Morel-Seytoux (Ed.),
    Unsaturated flow in hydrologic modelling - Theory and practice, NATO ASI Series 9,
    275-300, Dordrecht, The Netherlands: Kluwer Academic Publishing, 1989.

    Parameters
    ----------
    sand: float
        sand percentage [%].
    thetas : float
        saturated water content [m3/m3].
    clay: float
        clay percentage [%].

    Returns
    -------
    poresizeindex : float
        pore size distribution index [-].

    """
    poresizeindex = np.exp(
        -0.7842831
        + 0.0177544 * sand
        - 1.062498 * thetas
        - (5.304 * 10**-5) * (sand**2)
        - 0.00273493 * (clay**2)
        + 1.11134946 * (thetas**2)
        - 0.03088295 * sand * thetas
        + (2.6587 * 10**-4) * (sand**2) * (thetas**2)
        - 0.00610522 * (clay**2) * (thetas**2)
        - (2.35 * 10**-6) * (sand**2) * clay
        + 0.00798746 * (clay**2) * thetas
        - 0.00674491 * (thetas**2) * clay
    )

    return poresizeindex


def kv_brakensiek(thetas, clay, sand):
    """
    Determine saturated hydraulic conductivity kv [m/day].

    Based on:
      Brakensiek, D.L., Rawls, W.J.,and Stephenson, G.R.: Modifying scs hydrologic
      soil groups and curve numbers for range land soils, ASAE Paper no. PNR-84-203,
      St. Joseph, Michigan, USA, 1984.

    Parameters
    ----------
    thetas: float
        saturated water content [m3/m3].
    clay : float
        clay percentage [%].
    sand: float
        sand percentage [%].

    Returns
    -------
    kv : float
        saturated hydraulic conductivity [m/day].

    """
    kv = (
        np.exp(
            19.52348 * thetas
            - 8.96847
            - 0.028212 * clay
            + (1.8107 * 10**-4) * sand**2
            - (9.4125 * 10**-3) * clay**2
            - 8.395215 * thetas**2
            + 0.077718 * sand * thetas
            - 0.00298 * sand**2 * thetas**2
            - 0.019492 * clay**2 * thetas**2
            + (1.73 * 10**-5) * sand**2 * clay
            + 0.02733 * clay**2 * thetas
            + 0.001434 * sand**2 * thetas
            - (3.5 * 10**-6) * clay**2 * sand
        )
        * (2.78 * 10**-6)
        * 1000
        * 3600
        * 24
    )

    return kv / 1000  # m/day


def estimate_soil_properties(
    self, soil_layer_height, soil_organic_carbon, bulk_density, sand, clay, silt
):
    is_top_soil = np.zeros_like(clay, dtype=bool)
    is_top_soil[0:3] = True

    thetas = thetas_toth(
        soil_organic_carbon=soil_organic_carbon,
        bulk_density=bulk_density,
        is_top_soil=is_top_soil,
        clay=clay,
        silt=silt,
    )
    thetar = thetar_brakensiek(sand=sand, clay=clay, thetas=thetas)
    self.HRU.var.bubbling_pressure_cm = get_bubbling_pressure(
        clay=clay, sand=sand, thetas=thetas
    )
    self.HRU.var.lambda_pore_size_distribution = get_pore_size_index_brakensiek(
        sand=sand, thetas=thetas, clay=clay
    )

    # θ saturation, field capacity, wilting point and residual moisture content
    thetafc = get_soil_moisture_at_pressure(
        np.float32(-100.0),  # assuming field capacity is at -100 cm (pF 2)
        self.HRU.var.bubbling_pressure_cm,
        thetas,
        thetar,
        self.HRU.var.lambda_pore_size_distribution,
    )

    thetawp = get_soil_moisture_at_pressure(
        np.float32(-(10**4.2)),  # assuming wilting point is at -10^4.2 cm (pF 4.2)
        self.HRU.var.bubbling_pressure_cm,
        thetas,
        thetar,
        self.HRU.var.lambda_pore_size_distribution,
    )

    self.HRU.var.ws = thetas * self.HRU.var.soil_layer_height
    self.HRU.var.wfc = thetafc * self.HRU.var.soil_layer_height
    self.HRU.var.wwp = thetawp * self.HRU.var.soil_layer_height
    self.HRU.var.wres = thetar * self.HRU.var.soil_layer_height

    # initial soil water storage between field capacity and wilting point
    # set soil moisture to nan where land use is not bioarea
    self.HRU.var.w = np.where(
        self.HRU.var.land_use_type[np.newaxis, :] < SEALED,
        (self.HRU.var.wfc + self.HRU.var.wwp) / 2,
        np.nan,
    )
    # for paddy irrigation flooded paddy fields
    self.HRU.var.topwater = self.HRU.full_compressed(0, dtype=np.float32)

    self.HRU.var.ksat = kv_brakensiek(thetas=thetas, clay=clay, sand=sand)

    # soil water depletion fraction, Van Diepen et al., 1988: WOFOST 6.0, p.86, Doorenbos et. al 1978
    # crop groups for formular in van Diepen et al, 1988
    natural_crop_groups = self.model.data.grid.load(
        self.model.files["grid"]["soil/cropgrp"]
    )
    self.HRU.var.natural_crop_groups = self.model.data.to_HRU(data=natural_crop_groups)

    # ------------ Preferential Flow constant ------------------------------------------
    self.HRU.var.preferential_flow_constant = float(
        self.model.config["parameters"]["preferentialFlowConstant"]
    )

    self.HRU.var.arnoBeta = self.HRU.full_compressed(np.nan, dtype=np.float32)

    # Improved Arno's scheme parameters: Hageman and Gates 2003
    # arnoBeta defines the shape of soil water capacity distribution curve as a function of  topographic variability
    # b = max( (oh - o0)/(oh + omax), 0.01)
    # oh: the standard deviation of orography, o0: minimum std dev, omax: max std dev
    elevation_std = self.grid.load(
        self.model.files["grid"]["landsurface/topo/elevation_STD"]
    )
    elevation_std = self.model.data.to_HRU(data=elevation_std, fn=None)
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
        land_use_indices = np.where(self.HRU.var.land_use_type == cover)[0]

        self.HRU.var.arnoBeta[land_use_indices] = (arnoBetaOro + arno_beta)[
            land_use_indices
        ]
        self.HRU.var.arnoBeta[land_use_indices] = np.minimum(
            1.2, np.maximum(0.01, self.HRU.var.arnoBeta[land_use_indices])
        )

    self.HRU.var.aeration_days_counter = np.full_like(
        self.HRU.var.ws, 0, dtype=np.int32
    )
    self.HRU.var.crop_lag_aeration_days = np.full_like(
        self.HRU.var.land_use_type, 3, dtype=np.int32
    )


class Soil(object):
    def __init__(self, model):
        """

        Notes:
        - We don't consider that bedrock does not impede percolation in line with https://doi.org/10.1029/2018WR022920
        which states that connection to the stream is the exception rather than the rule
        A better implementation would be to consider travel distance. But this remains a topic for future work.
        """
        self.HRU = model.data.HRU
        self.grid = model.data.grid
        self.model = model
        if self.model.spinup:
            self.spinup()

    def spinup(self):
        self.var = self.model.store.create_bucket("soil.var")

        # Load Soil properties from the original files
        self.HRU.var.soil_layer_height = self.HRU.compress(
            load_grid(
                self.model.files["subgrid"]["soil/soil_layer_height"],
                layer=None,
            ),
            method="mean",
        )

        self.HRU.var.soil_organic_carbon = self.HRU.compress(
            load_grid(
                self.model.files["subgrid"]["soil/soil_organic_carbon"],
                layer=None,
            ),
            method="mean",
        )
        self.HRU.var.bulk_density = self.HRU.compress(
            load_grid(
                self.model.files["subgrid"]["soil/bulk_density"],
                layer=None,
            ),
            method="mean",
        )
        self.HRU.var.sand = self.HRU.compress(
            load_grid(
                self.model.files["subgrid"]["soil/sand"],
                layer=None,
            ),
            method="mean",
        )
        self.HRU.var.silt = self.HRU.compress(
            load_grid(
                self.model.files["subgrid"]["soil/silt"],
                layer=None,
            ),
            method="mean",
        )
        self.HRU.var.clay = self.HRU.compress(
            load_grid(
                self.model.files["subgrid"]["soil/clay"],
                layer=None,
            ),
            method="mean",
        )

        # Estimate soil properties
        estimate_soil_properties(
            self,
            soil_layer_height=self.HRU.var.soil_layer_height,
            soil_organic_carbon=self.HRU.var.soil_organic_carbon,
            bulk_density=self.HRU.var.bulk_density,
            sand=self.HRU.var.sand,
            clay=self.HRU.var.clay,
            silt=self.HRU.var.silt,
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
                self.HRU.var.land_use_type, dtype=bool
            )
            for i, land_use_type_RU in enumerate(self.HRU.var.land_use_type):
                grid_cell = self.HRU.var.HRU_to_grid[i]
                # if land_use_type_RU == 0 and self.HRU.var.land_use_ratio[i] > 0.5:
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
        del soil_moisture
        soil_moisture_wilting_point_fraction = (
            soil_moisture_wilting_point / soil_tickness
        )
        del soil_moisture_wilting_point
        soil_moisture_field_capacity_fraction = (
            soil_moisture_field_capacity / soil_tickness
        )
        del soil_moisture_field_capacity

        n_potential = -(
            np.log(wilting_point / field_capacity)
            / np.log(
                soil_moisture_wilting_point_fraction
                / soil_moisture_field_capacity_fraction
            )
        )
        a_potential = 1.5 * 10**6 * soil_moisture_wilting_point_fraction**n_potential
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

    def set_global_variables(self):
        # set number of soil layers as global variable for numba
        global N_SOIL_LAYERS
        N_SOIL_LAYERS = self.HRU.var.soil_layer_height.shape[0]

        # set the frost index threshold as global variable for numba
        global FROST_INDEX_THRESHOLD
        FROST_INDEX_THRESHOLD = np.float32(self.HRU.var.frost_indexThreshold)

    def step(
        self,
        capillary_rise_from_groundwater,
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
        timer = TimingModule("Soil")

        if __debug__:
            w_pre = self.HRU.var.w.copy()
            topwater_pre = self.HRU.var.topwater.copy()

        bioarea = self.HRU.var.land_use_type < SEALED

        interflow = self.HRU.full_compressed(0, dtype=np.float32)

        available_water_infiltration, open_water_evaporation = (
            get_available_water_infiltration(
                natural_available_water_infiltration=self.HRU.var.natural_available_water_infiltration,
                actual_irrigation_consumption=self.HRU.var.actual_irrigation_consumption,
                land_use_type=self.HRU.var.land_use_type,
                crop_kc=self.HRU.var.cropKC,
                EWRef=self.HRU.var.EWRef,
                topwater=self.HRU.var.topwater,
            )
        )

        timer.new_split("Available infiltration")

        assert (self.HRU.var.w[:, bioarea] <= self.HRU.var.ws[:, bioarea]).all()
        assert (self.HRU.var.w[:, bioarea] >= self.HRU.var.wres[:, bioarea]).all()

        runoff_from_groundwater = rise_from_groundwater(
            w=self.HRU.var.w,
            ws=self.HRU.var.ws,
            capillary_rise_from_groundwater=capillary_rise_from_groundwater.astype(
                np.float32
            ),
        )

        assert (self.HRU.var.w[:, bioarea] <= self.HRU.var.ws[:, bioarea]).all()
        assert (self.HRU.var.w[:, bioarea] >= self.HRU.var.wres[:, bioarea]).all()

        timer.new_split("Capillary rise from groundwater")

        (
            actual_total_transpiration,
            actual_bare_soil_evaporation,
        ) = evapotranspirate(
            wwp=self.HRU.var.wwp,
            wfc=self.HRU.var.wfc,
            ws=self.HRU.var.ws,
            wres=self.HRU.var.wres,
            aeration_days_counter=self.HRU.var.aeration_days_counter,
            soil_layer_height=self.HRU.var.soil_layer_height,
            land_use_type=self.HRU.var.land_use_type,
            root_depth=self.HRU.var.root_depth,
            crop_map=self.HRU.var.crop_map,
            natural_crop_groups=self.HRU.var.natural_crop_groups,
            crop_lag_aeration_days=self.HRU.var.crop_lag_aeration_days,
            potential_transpiration=potential_transpiration,
            potential_bare_soil_evaporation=potential_bare_soil_evaporation,
            potential_evapotranspiration=potential_evapotranspiration,
            frost_index=self.HRU.var.frost_index,
            crop_group_number_per_group=self.model.agents.crop_farmers.var.crop_data[
                "crop_group_number"
            ].values.astype(np.float32),
            w=self.HRU.var.w,
            topwater=self.HRU.var.topwater,
            open_water_evaporation=open_water_evaporation,
            available_water_infiltration=available_water_infiltration,
        )
        assert actual_total_transpiration.dtype == np.float32
        assert (self.HRU.var.w[:, bioarea] <= self.HRU.var.ws[:, bioarea]).all()
        assert (self.HRU.var.w[:, bioarea] >= self.HRU.var.wres[:, bioarea]).all()

        timer.new_split("Evapotranspiration")

        n_substeps = 3
        preferential_flow = np.zeros_like(self.HRU.var.land_use_type, dtype=np.float32)
        direct_runoff = np.zeros_like(self.HRU.var.land_use_type, dtype=np.float32)
        groundwater_recharge = np.zeros_like(
            self.HRU.var.land_use_type, dtype=np.float32
        )

        for _ in range(n_substeps):
            (
                preferential_flow_substep,
                direct_runoff_substep,
                groundwater_recharge_substep,
            ) = vertical_water_transport(
                available_water_infiltration / n_substeps,
                capillary_rise_from_groundwater / n_substeps,
                self.HRU.var.ws,
                self.HRU.var.wres,
                self.HRU.var.ksat / n_substeps,
                self.HRU.var.lambda_pore_size_distribution,
                self.HRU.var.bubbling_pressure_cm,
                self.HRU.var.land_use_type,
                self.HRU.var.frost_index,
                self.HRU.var.arnoBeta,
                np.float32(self.HRU.var.preferential_flow_constant),
                self.HRU.var.w,
                self.HRU.var.topwater,
                self.HRU.var.soil_layer_height,
            )

            preferential_flow += preferential_flow_substep
            direct_runoff += direct_runoff_substep
            groundwater_recharge[bioarea] += groundwater_recharge_substep[bioarea]

        timer.new_split("Vertical transport")

        assert (self.HRU.var.w[:, bioarea] <= self.HRU.var.ws[:, bioarea]).all()
        assert (self.HRU.var.w[:, bioarea] >= self.HRU.var.wres[:, bioarea]).all()

        runoff = direct_runoff + runoff_from_groundwater

        assert preferential_flow.dtype == np.float32
        assert runoff.dtype == np.float32

        self.HRU.var.actual_evapotranspiration += actual_bare_soil_evaporation
        self.HRU.var.actual_evapotranspiration += actual_total_transpiration
        self.HRU.var.actual_evapotranspiration += open_water_evaporation

        if __debug__:
            assert (self.HRU.var.w[:, bioarea] <= self.HRU.var.ws[:, bioarea]).all()
            assert (self.HRU.var.w[:, bioarea] >= self.HRU.var.wres[:, bioarea]).all()
            assert (interflow == 0).all()  # interflow is not implemented (see above)
            balance_check(
                name="soil_1",
                how="cellwise",
                influxes=[
                    self.HRU.var.natural_available_water_infiltration[bioarea],
                    capillary_rise_from_groundwater[bioarea],
                    self.HRU.var.actual_irrigation_consumption[bioarea],
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
                    self.HRU.var.w[:, bioarea].sum(axis=0),
                    self.HRU.var.topwater[bioarea],
                ],
                tollerance=1e-6,
            )

            balance_check(
                name="soil_2",
                how="cellwise",
                influxes=[
                    self.HRU.var.natural_available_water_infiltration[bioarea],
                    capillary_rise_from_groundwater[bioarea],
                    self.HRU.var.actual_irrigation_consumption[bioarea],
                    self.HRU.var.snowEvap[bioarea],
                    self.HRU.var.interception_evaporation[bioarea],
                ],
                outfluxes=[
                    runoff[bioarea],
                    interflow[bioarea],
                    groundwater_recharge[bioarea],
                    self.HRU.var.actual_evapotranspiration[bioarea],
                ],
                prestorages=[
                    w_pre[:, bioarea].sum(axis=0),
                    topwater_pre[bioarea],
                ],
                poststorages=[
                    self.HRU.var.w[:, bioarea].sum(axis=0),
                    self.HRU.var.topwater[bioarea],
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
            runoff,
            groundwater_recharge,
            open_water_evaporation,
            actual_total_transpiration,
            actual_bare_soil_evaporation,
        )
