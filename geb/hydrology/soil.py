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


def get_soil_moisture_at_pressure(
    capillary_suction, bubbling_pressure_cm, thetas, thetar, lambda_
):

    alpha = bubbling_pressure_cm**-1
    n = lambda_ + 1
    m = 1 - 1 / n
    phi = -capillary_suction

    water_retention_curve = (1 / (1 + (alpha * phi) ** n)) ** m

    theta = water_retention_curve * (thetas - thetar) + thetar
    return theta


@njit(inline="always")
def get_aeration_stress_threshold(
    ws, soil_layer_height, crop_aeration_stress_threshold
):
    max_saturation_fraction = ws / soil_layer_height
    # Water storage in root zone at aeration stress threshold (m)
    return (
        max_saturation_fraction - (crop_aeration_stress_threshold / 100)
    ) * soil_layer_height


@njit(inline="always")
def get_aeration_stress_reduction_factor(
    aeration_days_counter, crop_lag_aeration_days, ws, w, aeration_stress_threshold
):
    if aeration_days_counter < crop_lag_aeration_days:
        stress = 1 - ((ws - w) / (ws - aeration_stress_threshold))
        aeration_stress_reduction_factor = 1 - ((aeration_days_counter / 3) * stress)
    else:
        aeration_stress_reduction_factor = (ws - w) / (ws - aeration_stress_threshold)
    return aeration_stress_reduction_factor


@njit(inline="always")
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
    return (1 - p) * (wfc - wwp) + wwp


def get_available_water(w, wwp):
    return np.maximum(0.0, w - wwp)


def get_maximum_water_content(wfc, wwp):
    return wfc - wwp


@njit
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


@njit(inline="always")
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
    potential_evapotranspiration_cm = potential_evapotranspiration * 100.0

    p = 1.0 / (0.76 + 1.5 * potential_evapotranspiration_cm) - 0.10 * (
        5.0 - crop_group_number
    )

    # Additional correction for crop groups 1 and 2
    if crop_group_number <= 2.5:
        p = p + (potential_evapotranspiration_cm - 0.6) / (
            crop_group_number * (crop_group_number + 3.0)
        )

    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    return p


def get_critical_water_level(p, wfc, wwp):
    return np.maximum(get_critical_soil_moisture_content(p, wfc, wwp) - wwp, 0)


def get_total_transpiration_reduction_factor(
    transpiration_reduction_factor_per_layer, root_ratios, soil_layer_height
):
    transpiration_reduction_factor_relative_contribution_per_layer = (
        soil_layer_height * root_ratios
    )
    transpiration_reduction_factor_total = np.sum(
        transpiration_reduction_factor_relative_contribution_per_layer
        / transpiration_reduction_factor_relative_contribution_per_layer.sum(axis=0)
        * transpiration_reduction_factor_per_layer,
        axis=0,
    )
    return transpiration_reduction_factor_total


@njit(inline="always")
def get_transpiration_reduction_factor_single(w, wwp, wcrit):
    nominator = w - wwp
    denominator = wcrit - wwp
    if denominator == 0:
        if nominator > 0:
            return 1
        else:
            return 0
    factor = nominator / denominator
    if factor < 0:
        return 0
    if factor > 1:
        return 1
    return factor


@njit
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


@njit(inline="always")
def set_root_ratios_single(root_depth, soil_layer_height, root_ratios):
    remaining_root_depth = root_depth
    for layer in range(N_SOIL_LAYERS):
        root_ratios[layer] = min(remaining_root_depth / soil_layer_height[layer], 1)
        remaining_root_depth -= soil_layer_height[layer]
        if remaining_root_depth < 0:
            return root_ratios
    return root_ratios


def get_crop_group_number(
    crop_map, crop_group_numbers, land_use_type, natural_crop_groups
):
    crop_group_map = np.take(crop_group_numbers, crop_map)
    crop_group_map[crop_map == -1] = np.nan

    natural_land = np.isin(land_use_type, (0, 1))
    crop_group_map[natural_land] = natural_crop_groups[natural_land]
    return crop_group_map


@njit
def get_unsaturated_hydraulic_conductivity(
    w, wres, ws, lambda_, saturated_hydraulic_conductivity
):
    """This function calculates the unsaturated hydraulic conductivity based on the soil moisture content
    following van Genuchten (1980) and Mualem (1976)

    See https://archive.org/details/watershedmanagem0000unse_d4j9/page/295/mode/1up?view=theater (p. 295)
    """
    effective_saturation = (w - wres) / (ws - wres)
    if effective_saturation < 0:
        effective_saturation = 0
    elif effective_saturation > 1:
        effective_saturation = 1

    n = lambda_ + 1
    m = 1 - 1 / n

    return (
        saturated_hydraulic_conductivity
        * effective_saturation**0.5
        * (1 - (1 - effective_saturation ** (1 / m)) ** m) ** 2
    )


PERCOLATION_SUBSTEPS = np.int32(3)


@njit(cache=True, parallel=True)
def update_soil_water_storage(
    wwp,
    wfc,
    ws,
    wres,
    aeration_days_counter,
    soil_layer_height,
    saturated_hydraulic_conductivity,
    lambda_,
    land_use_type,
    root_depth,
    actual_irrigation_consumption,
    natural_available_water_infiltration,
    crop_kc,
    crop_map,
    natural_crop_groups,
    crop_lag_aeration_days,
    EWRef,
    potential_transpiration,
    potential_bare_soil_evaporation,
    potential_evapotranspiration,
    frost_index,
    arno_beta,
    capillar,
    capillary_rise_index,
    crop_group_number_per_group,
    cPrefFlow,
    w,
    topwater_res,
    open_water_evaporation_res,
    actual_bare_soil_evaporation_res,
    actual_total_transpiration,
    groundwater_recharge,
    direct_runoff,
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

    bottom_soil_layer_index = N_SOIL_LAYERS - 1
    root_ratios_matrix = np.zeros_like(soil_layer_height)
    root_distribution_per_layer_rws_corrected_matrix = np.zeros_like(soil_layer_height)
    root_distribution_per_layer_aeration_stress_corrected_matrix = np.zeros_like(
        soil_layer_height
    )
    capillary_rise_soil_matrix = np.zeros_like(soil_layer_height)
    percolation_matrix = np.zeros_like(soil_layer_height)
    preferential_flow = np.zeros_like(land_use_type, dtype=np.float32)
    available_water_infiltration = np.zeros_like(land_use_type, dtype=np.float32)
    runoff_from_groundwater = np.zeros_like(land_use_type, dtype=np.float32)
    is_bioarea = land_use_type <= 3
    soil_is_frozen = frost_index > FROST_INDEX_THRESHOLD

    for i in prange(land_use_type.size):
        available_water_infiltration[i] = (
            natural_available_water_infiltration[i] + actual_irrigation_consumption[i]
        )
        if available_water_infiltration[i] < np.float32(0):
            available_water_infiltration[i] = np.float32(0)
        # paddy irrigated land
        if land_use_type[i] == 2:
            if crop_kc[i] > np.float32(0.75):
                topwater_res[i] += available_water_infiltration[i]

            assert EWRef[i] >= np.float32(0)
            open_water_evaporation_res[i] = min(
                max(np.float32(0.0), topwater_res[i]), EWRef[i]
            )
            topwater_res[i] -= open_water_evaporation_res[i]
            assert topwater_res[i] >= np.float32(0)
            if crop_kc[i] > np.float32(0.75):
                available_water_infiltration[i] = topwater_res[i]
            else:
                available_water_infiltration[i] += topwater_res[i]

            # TODO: Minor bug, this should only occur when topwater is above 0
            # fix this after completing soil module speedup
            potential_bare_soil_evaporation[i] = max(
                np.float32(0),
                potential_bare_soil_evaporation[i] - open_water_evaporation_res[i],
            )

    for i in prange(land_use_type.size):
        w[bottom_soil_layer_index, i] += capillar[
            i
        ]  # add capillar rise to the bottom soil layer

        # if the bottom soil layer is full, send water to the above layer, repeat until top layer
        for j in range(bottom_soil_layer_index, 0, -1):
            if w[j, i] > ws[j, i]:
                w[j - 1, i] += w[j, i] - ws[j, i]  # move excess water to layer above
                w[j, i] = ws[j, i]  # set the current layer to full

        # if the top layer is full, send water to the runoff
        if w[0, i] > ws[0, i]:
            runoff_from_groundwater[i] = (
                w[0, i] - ws[0, i]
            )  # move excess water to runoff
            w[0, i] = ws[0, i]  # set the top layer to full

        # get group group numbers for natural areas
        if land_use_type[i] == 0 or land_use_type[i] == 1:
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

        total_transpiration_reduction_factor_water_stress = np.float32(0.0)
        total_aeration_stress = np.float32(0.0)
        total_root_length_rws_corrected = np.float32(
            0.0
        )  # check if same as total_transpiration_reduction_factor * root_depth
        total_root_length_aeration_stress_corrected = np.float32(0.0)
        for layer in range(N_SOIL_LAYERS):
            root_length_within_layer = soil_layer_height[layer, i] * root_ratios[layer]

            # Water stress
            critical_soil_moisture_content = get_critical_soil_moisture_content(
                p, wfc[layer, i], wwp[layer, i]
            )
            transpiration_reduction_factor = get_transpiration_reduction_factor_single(
                w[layer, i], wwp[layer, i], critical_soil_moisture_content
            )

            total_transpiration_reduction_factor_water_stress += (
                transpiration_reduction_factor
            ) * root_length_within_layer

            root_length_within_layer_rws_corrected = (
                root_length_within_layer * transpiration_reduction_factor
            )
            total_root_length_rws_corrected += root_length_within_layer_rws_corrected
            root_distribution_per_layer_rws_corrected_matrix[layer, i] = (
                root_length_within_layer_rws_corrected
            )

            # Aeration stress
            aeration_stress_threshold = get_aeration_stress_threshold(
                ws[layer, i], soil_layer_height[layer, i], np.float32(15)
            )  # 15 is placeholder for crop_aeration_threshold
            if w[layer, i] > aeration_stress_threshold:
                aeration_days_counter[layer, i] += 1
                aeration_stress_reduction_factor = get_aeration_stress_reduction_factor(
                    aeration_days_counter[layer, i],
                    crop_lag_aeration_days[i],
                    ws[layer, i],
                    w[layer, i],
                    aeration_stress_threshold,
                )
            else:
                # Reset aeration days counter where w <= waer
                aeration_days_counter[layer, i] = 0
                aeration_stress_reduction_factor = np.float32(1)  # no stress

            total_aeration_stress += (
                aeration_stress_reduction_factor * root_length_within_layer
            )

            root_length_within_layer_aeration_stress_corrected = (
                root_length_within_layer * aeration_stress_reduction_factor
            )

            total_root_length_aeration_stress_corrected += (
                root_length_within_layer_aeration_stress_corrected
            )

            root_distribution_per_layer_aeration_stress_corrected_matrix[layer, i] = (
                root_length_within_layer_aeration_stress_corrected
            )

        total_transpiration_reduction_factor_water_stress /= root_depth[i]
        total_aeration_stress /= root_depth[i]

        actual_total_transpiration[i] = (
            0  # reset the total transpiration TODO: This may be confusing.
        )

        # correct the transpiration reduction factor for water stress
        # if the soil is frozen, no transpiration occurs, so we can skip the loop
        # and thus transpiration is 0 this also avoids division by zero, and thus NaNs
        # likewise, if the total_transpiration_reduction_factor (full water stress) is 0
        # or full aeration stress, we can skip the loop
        if (
            not soil_is_frozen[i]
            and total_transpiration_reduction_factor_water_stress > np.float32(0)
            and total_aeration_stress > np.float32(0)
        ):
            maximum_transpiration = potential_transpiration[i] * min(
                total_transpiration_reduction_factor_water_stress, total_aeration_stress
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
                w[layer, i] -= transpiration
                if is_bioarea[i]:
                    actual_total_transpiration[i] += transpiration

        if is_bioarea[i]:
            # limit the bare soil evaporation to the available water in the soil
            if not soil_is_frozen[i] and topwater_res[i] == np.float32(0):
                actual_bare_soil_evaporation_res[i] = min(
                    potential_bare_soil_evaporation[i],
                    max(
                        w[0, i] - wres[0, i], np.float32(0)
                    ),  # can never be lower than 0
                )
                # remove the bare soil evaporation from the top layer
                w[0, i] -= actual_bare_soil_evaporation_res[i]
            else:
                # if the soil is frozen, no evaporation occurs
                # if the field is flooded (paddy irrigation), no bare soil evaporation occurs
                actual_bare_soil_evaporation_res[i] = np.float32(0)

    for i in prange(land_use_type.size):
        # estimate the infiltration capacity
        # use first 2 soil layers to estimate distribution between runoff and infiltration
        soil_water_storage = w[0, i] + w[1, i]
        soil_water_storage_max = ws[0, i] + ws[1, i]
        relative_saturation = soil_water_storage / soil_water_storage_max
        if relative_saturation > np.float32(
            1
        ):  # cap the relative saturation at 1 - this should not happen
            relative_saturation = np.float32(1)

        # Fraction of pixel that is at saturation as a function of
        # the ratio Theta1/ThetaS1. Distribution function taken from
        # Zhao,1977, as cited in Todini, 1996 (JoH 175, 339-382) Eq. A.4.
        saturated_area_fraction = (
            np.float32(1) - (np.float32(1) - relative_saturation) ** arno_beta[i]
        )
        if saturated_area_fraction > np.float32(1):
            saturated_area_fraction = np.float32(1)
        elif saturated_area_fraction < np.float32(0):
            saturated_area_fraction = np.float32(0)

        store = soil_water_storage_max / (
            arno_beta[i] + np.float32(1)
        )  # it is unclear what "store" means exactly, refer to source material to improve variable name
        pot_beta = (arno_beta[i] + np.float32(1)) / arno_beta[i]  # idem
        potential_infiltration = store - store * (
            np.float32(1) - (np.float32(1) - saturated_area_fraction) ** pot_beta
        )

        # if soil is frozen, there is no preferential flow, also not on paddy fields
        if not soil_is_frozen[i] and land_use_type[i] != 2:
            preferential_flow[i] = (
                available_water_infiltration[i] * relative_saturation**cPrefFlow
            ) * (np.float32(1) - capillary_rise_index[i])

        # no infiltration if the soil is frozen
        if soil_is_frozen[i]:
            infiltration = np.float32(0)
        else:
            infiltration = min(
                potential_infiltration,
                available_water_infiltration[i] - preferential_flow[i],
            )

        if is_bioarea[i]:
            direct_runoff[i] = max(
                (available_water_infiltration[i] - infiltration - preferential_flow[i]),
                np.float32(0),
            )

        if land_use_type[i] == 2:
            topwater_res[i] = max(np.float32(0), topwater_res[i] - infiltration)
            if crop_kc[i] > np.float32(0.75):
                # if paddy fields flooded only runoff if topwater > 0.05m
                direct_runoff[i] = max(
                    0, topwater_res[i] - np.float32(0.05)
                )  # TODO: Potential minor bug, should this be added to runoff instead of replacing runoff?
            topwater_res[i] = max(np.float32(0), topwater_res[i] - direct_runoff[i])

        if is_bioarea[i]:
            direct_runoff[i] += runoff_from_groundwater[i]

        # add infiltration to the soil
        w[0, i] += infiltration
        if w[0, i] > ws[0, i]:
            w[1, i] += (
                w[0, i] - ws[0, i]
            )  # TODO: Solve edge case of the second layer being full, in principle this should not happen as infiltration should be capped by the infilation capacity
            w[0, i] = ws[0, i]

    for i in prange(land_use_type.size):
        # capillary rise between soil layers, iterate from top, but skip bottom (which is has capillary rise from groundwater)
        for layer in range(N_SOIL_LAYERS - 1):
            saturation_ratio = max(
                (w[layer, i] - wres[layer, i]) / (wfc[layer, i] - wres[layer, i]),
                np.float32(0),
            )
            unsaturated_hydraulic_conductivity_layer_below = (
                get_unsaturated_hydraulic_conductivity(
                    w[layer + 1, i],
                    wres[layer + 1, i],
                    ws[layer + 1, i],
                    lambda_[layer + 1, i],
                    saturated_hydraulic_conductivity[layer + 1, i],
                )
            )
            capillary_rise_soil = max(
                np.float32(0),
                (np.float32(1) - saturation_ratio)
                * unsaturated_hydraulic_conductivity_layer_below,
            )
            # penultimate layer, limit capillary rise to available water in bottom layer
            if layer == N_SOIL_LAYERS - 2:
                capillary_rise_soil = min(
                    capillary_rise_soil,
                    w[N_SOIL_LAYERS - 1, i] - wres[N_SOIL_LAYERS - 1, i],
                )

            capillary_rise_soil_matrix[layer, i] = capillary_rise_soil

        for layer in range(N_SOIL_LAYERS):
            # for all layers except the bottom layer, add capillary rise from below to the layer
            if layer != N_SOIL_LAYERS - 1:
                w[layer, i] += capillary_rise_soil_matrix[layer, i]
            # for all layers except the top layer, remove capillary rise from the layer above
            if layer != 0:
                w[layer, i] -= capillary_rise_soil_matrix[layer - 1, i]

    for i in prange(land_use_type.size):
        # percolcation (top to bottom soil layer)
        percolation_to_groundwater = np.float32(0.0)
        for _ in range(PERCOLATION_SUBSTEPS):
            for layer in range(N_SOIL_LAYERS):
                unsaturated_hydraulic_conductivity = (
                    get_unsaturated_hydraulic_conductivity(
                        w[layer, i],
                        wres[layer, i],
                        ws[layer, i],
                        lambda_[layer, i],
                        saturated_hydraulic_conductivity[layer, i],
                    )
                )
                percolation = unsaturated_hydraulic_conductivity / PERCOLATION_SUBSTEPS
                # no percolation if the soil is frozen in the top 2 layers.
                if not (soil_is_frozen[i] and (layer == 0 or layer == 1)):
                    # limit percolation by the available water in the layer
                    available_water = max(w[layer, i] - wres[layer, i], np.float32(0))
                    percolation = min(percolation, available_water)
                    if layer == N_SOIL_LAYERS - 1:  # last soil layer
                        # limit percolation by available water, and consider the capillary rise index
                        # TODO: Check how the capillary rise index works
                        percolation *= np.float32(1) - capillary_rise_index[i]
                    else:
                        # limit percolation the remaining water storage capacity of the layer below
                        percolation = min(
                            percolation, ws[layer + 1, i] - w[layer + 1, i]
                        )
                else:
                    percolation = (
                        0  # must be set for percolation to groundwater to be correct
                    )

                # save percolation in matrix
                percolation_matrix[layer, i] = percolation

            # for bottom soil layer, save percolation to groundwater
            percolation_to_groundwater += percolation

            for layer in range(N_SOIL_LAYERS):
                # for all layers, remove percolation from the layer
                w[layer, i] -= percolation_matrix[layer, i]
                # for all layers except the top layer, add percolation from the layer above (-1)
                if layer != 0:
                    w[layer, i] += percolation_matrix[layer - 1, i]

            if is_bioarea[i]:
                groundwater_recharge[i] = (
                    percolation_to_groundwater + preferential_flow[i]
                )


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
            self.model.model_structure["grid"]["soil/soil_layer_height"],
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
        FROST_INDEX_THRESHOLD = np.float32(self.var.FrostIndexThreshold)

        # θ saturation, field capacity, wilting point and residual moisture content
        thetas = self.model.data.grid.load(
            self.model.model_structure["grid"]["soil/thetas"], layer=None
        )
        thetas = self.model.data.to_HRU(data=thetas, fn=None)
        thetar = self.model.data.grid.load(
            self.model.model_structure["grid"]["soil/thetar"], layer=None
        )
        thetar = self.model.data.to_HRU(data=thetar, fn=None)

        bubbling_pressure_cm = self.model.data.grid.load(
            self.model.model_structure["grid"]["soil/bubbling_pressure_cm"], layer=None
        )
        bubbling_pressure_cm = self.model.data.to_HRU(
            data=bubbling_pressure_cm, fn=None
        )

        lambda_pore_size_distribution = self.model.data.grid.load(
            self.model.model_structure["grid"]["soil/lambda"], layer=None
        )
        lambda_pore_size_distribution = self.model.data.to_HRU(
            data=lambda_pore_size_distribution, fn=None
        )

        thetafc = get_soil_moisture_at_pressure(
            -100,  # assuming field capacity is at -100 cm (pF 2)
            bubbling_pressure_cm,
            thetas,
            thetar,
            lambda_pore_size_distribution,
        )

        thetawp = get_soil_moisture_at_pressure(
            -(10**4.2),  # assuming wilting point is at -10^4.2 cm (pF 4.2)
            bubbling_pressure_cm,
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
            self.model.model_structure["grid"]["soil/lambda"], layer=None
        )
        self.lambda_pore_size_distribution = self.model.data.to_HRU(
            data=lambda_pore_size_distribution, fn=None
        )
        ksat = self.model.data.grid.load(
            self.model.model_structure["grid"]["soil/ksat"], layer=None
        )
        self.ksat = self.model.data.to_HRU(data=ksat, fn=None)

        # soil water depletion fraction, Van Diepen et al., 1988: WOFOST 6.0, p.86, Doorenbos et. al 1978
        # crop groups for formular in van Diepen et al, 1988
        self.natural_crop_groups = self.model.data.grid.load(
            self.model.model_structure["grid"]["soil/cropgrp"]
        )
        self.natural_crop_groups = self.model.data.to_HRU(data=self.natural_crop_groups)

        # ------------ Preferential Flow constant ------------------------------------------
        self.var.cPrefFlow = np.float32(
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
            "forest": 0.2,
            "grassland": 0.0,
            "irrPaddy": 0.2,
            "irrNonPaddy": 0.2,
        }

        for coverNum, coverType in enumerate(self.model.coverTypes[:4]):
            land_use_indices = np.where(self.var.land_use_type == coverNum)[0]

            arnoBeta = arnobeta_cover_types[coverType]
            if not isinstance(arnoBeta, float):
                arnoBeta = arnoBeta[land_use_indices]
            self.var.arnoBeta[land_use_indices] = (arnoBetaOro + arnoBeta)[
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
            ini_file = out_dir / f"p_daily.ini"

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
                if land_use_type_RU == 0 and grid_cell == cell_id:
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
        capillar,
        open_water_evaporation,
        potTranspiration,
        potBareSoilEvap,
        totalPotET,
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

        groundwater_recharge = self.var.full_compressed(0, dtype=np.float32)
        direct_runoff = self.var.full_compressed(0, dtype=np.float32)
        interflow = self.var.full_compressed(0, dtype=np.float32)

        timer = TimingModule("Soil")

        update_soil_water_storage(
            self.wwp,
            self.wfc,
            self.ws,
            self.wres,
            self.var.aeration_days_counter,
            self.soil_layer_height,
            self.ksat,
            self.lambda_pore_size_distribution,
            self.var.land_use_type,
            self.var.root_depth,
            self.var.actual_irrigation_consumption,
            self.var.natural_available_water_infiltration,
            self.var.cropKC,
            self.var.crop_map,
            self.natural_crop_groups,
            self.crop_lag_aeration_days,
            self.var.EWRef,
            potTranspiration,
            potBareSoilEvap,
            totalPotET,
            self.var.FrostIndex,
            self.var.arnoBeta,
            capillar.astype(np.float32),
            self.var.capriseindex,
            self.model.agents.crop_farmers.crop_data["crop_group_number"].values.astype(
                np.float32
            ),
            self.var.cPrefFlow,
            self.var.w,
            self.var.topwater,
            open_water_evaporation,
            self.var.actBareSoilEvap,
            self.var.actTransTotal,
            groundwater_recharge,
            direct_runoff,
        )

        timer.new_split("Update soil water storage")

        bioarea = np.where(self.var.land_use_type < 4)[0].astype(np.int32)
        self.var.actualET[bioarea] = (
            self.var.actualET[bioarea]
            + self.var.actBareSoilEvap[bioarea]
            + open_water_evaporation[bioarea]
            + self.var.actTransTotal[bioarea]
        )

        if __debug__:
            balance_check(
                name="soil_1",
                how="cellwise",
                influxes=[
                    self.var.natural_available_water_infiltration[bioarea],
                    capillar[bioarea],
                    self.var.actual_irrigation_consumption[bioarea],
                ],
                outfluxes=[
                    direct_runoff[bioarea],
                    interflow[bioarea],
                    groundwater_recharge[bioarea],
                    self.var.actTransTotal[bioarea],
                    self.var.actBareSoilEvap[bioarea],
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
                    capillar[bioarea],
                    self.var.actual_irrigation_consumption[bioarea],
                    self.var.snowEvap[bioarea],
                    self.var.interceptEvap[bioarea],
                ],
                outfluxes=[
                    direct_runoff[bioarea],
                    interflow[bioarea],
                    groundwater_recharge[bioarea],
                    self.var.actualET[bioarea],
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

        timer.new_split("Finalizing")
        if self.model.timing:
            print(timer)

        return (
            interflow,
            direct_runoff,
            groundwater_recharge,
            open_water_evaporation,
        )
