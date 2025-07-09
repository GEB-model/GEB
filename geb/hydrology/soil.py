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


import platform
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numba import float32, njit, prange
from tqdm import tqdm

from geb.hydrology.HRUs import load_grid
from geb.module import Module
from geb.workflows import TimingModule, balance_check

from .landcover import (
    FOREST,
    GRASSLAND_LIKE,
    NON_PADDY_IRRIGATED,
    PADDY_IRRIGATED,
    SEALED,
)


def calculate_soil_water_potential_MPa(
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
    soil_moisture_wilting_point_fraction = soil_moisture_wilting_point / soil_tickness
    # assert (soil_moisture_wilting_point_fraction).all() >= 0 and (
    #     soil_moisture_wilting_point_fraction
    # ).all() <= 1
    del soil_moisture_wilting_point
    soil_moisture_field_capacity_fraction = soil_moisture_field_capacity / soil_tickness
    # assert (soil_moisture_field_capacity_fraction >= 0).all() and (
    #     soil_moisture_field_capacity_fraction <= 1
    # ).all()
    del soil_moisture_field_capacity

    n_potential = -(
        np.log(wilting_point / field_capacity)
        / np.log(
            soil_moisture_wilting_point_fraction / soil_moisture_field_capacity_fraction
        )
    )
    # assert (n_potential >= 0).all()
    a_potential = 1.5 * 10**6 * soil_moisture_wilting_point_fraction**n_potential
    # assert (a_potential >= 0).all()
    soil_water_potential = -a_potential * soil_moisture_fraction ** (-n_potential)
    return soil_water_potential / 1_000_000  # Pa to MPa


def calculate_vapour_pressure_deficit_kPa(temperature_K, relative_humidity):
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
    actual_vapour_pressure = saturated_vapour_pressure * relative_humidity / 100  # kPa
    vapour_pressure_deficit = saturated_vapour_pressure - actual_vapour_pressure
    return vapour_pressure_deficit


def calculate_photosynthetic_photon_flux_density(shortwave_radiation, xi=0.5):
    # https://search.r-project.org/CRAN/refmans/bigleaf/html/Rg.to.PPFD.html
    photosynthetically_active_radiation = shortwave_radiation * xi
    photosynthetic_photon_flux_density = (
        photosynthetically_active_radiation * 4.6
    )  #  W/m2 -> umol/m2/s
    return photosynthetic_photon_flux_density


@njit(cache=True, inline="always", fastmath=True)
def get_soil_water_potential(
    theta,
    thetar,
    thetas,
    lambda_,
    bubbling_pressure_cm,
    minimum_effective_saturation=np.float32(0.01),
):
    """Calculates the soil water potential (capillary suction) using the van Genuchten model.

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


@njit(
    cache=True,
    inline="always",
)
def get_soil_moisture_at_pressure(
    capillary_suction, bubbling_pressure_cm, thetas, thetar, lambda_
):
    """Calculates the soil moisture content at a given soil water potential (capillary suction) using the van Genuchten model.

    Args:
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
def get_critical_soil_moisture_content(p, wfc, wwp):
    """Calculate the critical soil moisture content.

    The critical soil moisture content is defined as the quantity of stored soil moisture below
    which water uptake is impaired and the crop begins to close its stomata. It is not a fixed
    value. Restriction of water uptake due to water stress starts at a higher water content
    when the potential transpiration rate is higher" (Van Diepen et al., 1988: WOFOST 6.0, p.86).

    A higher p value means that the critical soil moisture content is higher, i.e. the plant can
    extract water from the soil at a lower soil moisture content. Thus when p is 1 the critical
    soil moisture content is equal to the wilting point, and when p is 0 the critical soil moisture
    content is equal to the field capacity.
    """
    return (np.float32(1) - p) * (wfc - wwp) + wwp


@njit(cache=True, inline="always")
def get_fraction_easily_available_soil_water(
    crop_group_number: np.float32, potential_evapotranspiration: np.float32
) -> np.float32:
    """Calculate the fraction of easily available soil water.

    Calculation is based on crop group number and potential evapotranspiration
    following Van Diepen et al., 1988: WOFOST 6.0, p.87.

    Parameters
    ----------
    crop_group_number : np.ndarray
        The crop group number is a indicator of adaptation to dry climate,
        Van Diepen et al., 1988: WOFOST 6.0, p.87
    potential_evapotranspiration : np.ndarray
        Potential evapotranspiration in m

    Returns:
    -------
    np.ndarray
        The fraction of easily available soil water, p is closer to 0 if evapo is bigger and cropgroup is smaller
    """
    potential_evapotranspiration_cm: np.float32 = (
        potential_evapotranspiration * np.float32(100)
    )

    p: np.float32 = np.float32(1) / (
        np.float32(0.76) + np.float32(1.5) * potential_evapotranspiration_cm
    ) - np.float32(0.1) * (np.float32(5) - crop_group_number)

    # Additional correction for crop groups 1 and 2
    if crop_group_number <= np.float32(2.5):
        p: np.float32 = p + (potential_evapotranspiration_cm - np.float32(0.6)) / (
            crop_group_number * (crop_group_number + np.float32(3.0))
        )

    if p < np.float32(0):
        p: np.float32 = np.float32(0)
    if p > np.float32(1):
        p: np.float32 = np.float32(1)
    return p


@njit(cache=True, inline="always")
def get_transpiration_factor(w, wwp, wcrit):
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
def get_root_ratios(root_depth, soil_layer_height):
    remaining_root_depth = root_depth
    root_ratios = np.zeros_like(soil_layer_height)
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
    """Update the soil water storage based on the water balance calculations.

    Parameters
    ----------
    wwp : np.ndarray

    Notes:
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
    wres,
    soil_layer_height,
    land_use_type,
    root_depth,
    crop_map,
    natural_crop_groups,
    potential_transpiration,
    potential_bare_soil_evaporation,
    potential_evapotranspiration,
    frost_index,
    crop_group_number_per_group,
    w,
    topwater,
    open_water_evaporation,
    available_water_infiltration,
    minimum_effective_root_depth: float,
    mask,
    mask_transpiration,
):
    """Evapotranspiration calculation for the soil module.

    Parameters
    ----------
    mask : np.ndarray
        A mask indicating which pixels are valid for evapotranspiration calculation.
    mask_transpiration : np.ndarray
        A mask indicating which pixels are valid for transpiration calculation. This is
        useful when transpiration is calculated by an external module.
    """
    root_distribution_per_layer_rws_corrected_matrix = np.zeros_like(soil_layer_height)

    soil_is_frozen = frost_index > FROST_INDEX_THRESHOLD

    transpiration = np.zeros_like(land_use_type, dtype=np.float32)
    actual_bare_soil_evaporation = np.zeros_like(land_use_type, dtype=np.float32)

    for i in prange(land_use_type.size):
        remaining_potential_transpiration = potential_transpiration[i]
        if land_use_type[i] == PADDY_IRRIGATED and mask[i]:
            transpiration_from_topwater = min(
                topwater[i], remaining_potential_transpiration
            )
            remaining_potential_transpiration -= transpiration_from_topwater
            topwater[i] -= transpiration_from_topwater
            available_water_infiltration[i] -= transpiration_from_topwater
            transpiration[i] += transpiration_from_topwater

        # get group group numbers for natural areas
        if land_use_type[i] == FOREST or land_use_type[i] == GRASSLAND_LIKE:
            crop_group_number = natural_crop_groups[i]
        else:  #
            crop_group_number = crop_group_number_per_group[crop_map[i]]

        p = get_fraction_easily_available_soil_water(
            crop_group_number, potential_evapotranspiration[i]
        )

        effective_root_depth = np.maximum(
            np.float32(minimum_effective_root_depth), root_depth[i]
        )
        root_ratios = get_root_ratios(
            effective_root_depth,
            soil_layer_height[:, i],
        )

        total_transpiration_factor_water_stress = np.float32(0.0)
        total_root_length_rws_corrected = np.float32(
            0.0
        )  # check if same as total_transpiration_factor * root_depth
        for layer in range(N_SOIL_LAYERS):
            root_length_within_layer = soil_layer_height[layer, i] * root_ratios[layer]

            # Water stress
            critical_soil_moisture_content = get_critical_soil_moisture_content(
                p, wfc[layer, i], wwp[layer, i]
            )
            transpiration_factor = get_transpiration_factor(
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

        total_transpiration_factor_water_stress /= effective_root_depth

        # correct the transpiration reduction factor for water stress
        # if the soil is frozen, no transpiration occurs, so we can skip the loop
        # and thus transpiration is 0 this also avoids division by zero, and thus NaNs
        # likewise, if the total_transpiration_factor (full water stress) is 0,
        # we can skip the loop
        if not soil_is_frozen[
            i
        ] and total_transpiration_factor_water_stress > np.float32(0):
            maximum_transpiration = (
                remaining_potential_transpiration
                * total_transpiration_factor_water_stress
            )
            # distribute the transpiration over the layers, considering the root ratios
            # and the transpiration reduction factor per layer
            for layer in range(N_SOIL_LAYERS):
                transpiration_layer = (
                    maximum_transpiration
                    * root_distribution_per_layer_rws_corrected_matrix[layer, i]
                    / total_root_length_rws_corrected
                )

                if mask[i]:
                    w[layer, i] -= transpiration_layer
                    w[layer, i] = max(
                        w[layer, i], wres[layer, i]
                    )  # soil moisture can never be lower than wres
                    transpiration[i] += transpiration_layer

        if mask_transpiration[i]:
            # limit the bare soil evaporation to the available water in the soil
            if not soil_is_frozen[i] and land_use_type[i] != PADDY_IRRIGATED:
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
        transpiration,
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


@njit(cache=True, inline="always")
def get_infiltration_capacity(w, ws, arno_beta):
    # Estimate the infiltration capacity
    # Use first 2 soil layers to estimate distribution between runoff and infiltration
    soil_water_storage = w[0] + w[1]
    soil_water_storage_max = ws[0] + ws[1]

    relative_saturation = min(
        soil_water_storage / soil_water_storage_max, np.float32(1)
    )

    # Fraction of pixel that is at saturation
    saturated_area_fraction = (
        np.float32(1) - (np.float32(1) - relative_saturation) ** arno_beta
    )
    saturated_area_fraction = max(saturated_area_fraction, np.float32(0))
    saturated_area_fraction = min(saturated_area_fraction, np.float32(1))

    store = soil_water_storage_max / (arno_beta + np.float32(1))
    pot_beta = (arno_beta + np.float32(1)) / arno_beta
    return store - store * (
        np.float32(1) - (np.float32(1) - saturated_area_fraction) ** pot_beta
    )


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
    """Simulates vertical transport of water in the soil using Darcy's equation.

    Returns:
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
    for i in prange(land_use_type.size):
        potential_infiltration[i] = get_infiltration_capacity(
            w[:, i], ws[:, i], arno_beta[i]
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
            soil_water_storage = w[0, i] + w[1, i]
            soil_water_storage_max = ws[0, i] + ws[1, i]

            relative_saturation = min(
                soil_water_storage / soil_water_storage_max, np.float32(1)
            )
            preferential_flow[i] = (
                available_water_infiltration[i]
                * relative_saturation**preferential_flow_constant
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


def thetas_toth(
    soil_organic_carbon: npt.NDArray[np.float32],
    bulk_density: npt.NDArray[np.float32],
    is_top_soil: npt.NDArray[np.bool_],
    clay: npt.NDArray[np.float32],
    silt: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Determine saturated water content [m3/m3].

    Based on:
    Tóth, B., Weynants, M., Nemes, A., Makó, A., Bilas, G., and Tóth, G.:
    New generation of hydraulic pedotransfer functions for Europe, Eur. J.
    Soil Sci., 66, 226-238. doi: 10.1111/ejss.121921211, 2015.

    Parameters
    ----------
    bdod  bulk density [g /cm3].
    sand: sand percentage [%].
    silt: fsilt percentage [%].
    is_top_soil: top soil flag.

    Returns:
    -------
    thetas: saturated water content [cm3/cm3].

    """
    return (
        np.float32(0.6819)
        - np.float32(0.06480) * (1 / (soil_organic_carbon + 1))
        - np.float32(0.11900) * bulk_density**2
        - np.float32(0.02668) * is_top_soil
        + np.float32(0.001489) * clay
        + np.float32(0.0008031) * silt
        + np.float32(0.02321) * (1 / (soil_organic_carbon + 1)) * bulk_density**2
        + np.float32(0.01908) * bulk_density**2 * is_top_soil
        - np.float32(0.0011090) * clay * is_top_soil
        - np.float32(0.00002315) * silt * clay
        - np.float32(0.0001197) * silt * bulk_density**2
        - np.float32(0.0001068) * clay * bulk_density**2
    )


def thetar_brakensiek(
    sand: npt.NDArray[np.float32],
    clay: npt.NDArray[np.float32],
    thetas: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Determine residual water content [m3/m3].

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

    Returns:
    -------
    thetar : float
        residual water content [m3/m3].

    """
    return (
        np.float32(-0.0182482)
        + np.float32(0.00087269) * sand
        + np.float32(0.00513488) * clay
        + np.float32(0.02939286) * thetas
        - np.float32(0.00015395) * clay**2
        - np.float32(0.0010827) * sand * thetas
        - np.float32(0.00018233) * clay**2 * thetas**2
        + np.float32(0.00030703) * clay**2 * thetas
        - np.float32(0.0023584) * thetas**2 * clay
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
    """Determine Brooks-Corey pore size distribution index [-].

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

    Returns:
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
    """Determine saturated hydraulic conductivity kv [m/day].

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

    Returns:
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


class Soil(Module):
    """Soil module for the hydrological model.

    Args:
        model: The GEB model instance.
        hydrology: The hydrology submodel instance.
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
        return "hydrology.soil"

    def spinup(self):
        # use a minimum root depth of 25 cm, following AQUACROP recommendation
        # see: Reference manual for AquaCrop v7.1 – Chapter 3
        self.var.minimum_effective_root_depth = 0.25  # m

        # Soil properties
        self.HRU.var.soil_layer_height: npt.NDArray[np.float32] = self.HRU.compress(
            load_grid(
                self.model.files["subgrid"]["soil/soil_layer_height"],
                layer=None,
            ),
            method="mean",
        )

        soil_organic_carbon: npt.NDArray[np.float32] = self.HRU.compress(
            load_grid(
                self.model.files["subgrid"]["soil/soil_organic_carbon"],
                layer=None,
            ),
            method="mean",
        )
        bulk_density: npt.NDArray[np.float32] = self.HRU.compress(
            load_grid(
                self.model.files["subgrid"]["soil/bulk_density"],
                layer=None,
            ),
            method="mean",
        )
        self.HRU.var.silt: npt.NDArray[np.float32] = self.HRU.compress(
            load_grid(
                self.model.files["subgrid"]["soil/silt"],
                layer=None,
            ),
            method="mean",
        )
        self.HRU.var.clay: npt.NDArray[np.float32] = self.HRU.compress(
            load_grid(
                self.model.files["subgrid"]["soil/clay"],
                layer=None,
            ),
            method="mean",
        )

        # calculate sand content based on silt and clay content (together they should sum to 100%)
        self.HRU.var.sand: npt.NDArray[np.float32] = (
            100 - self.HRU.var.silt - self.HRU.var.clay
        )

        # the top 30 cm is considered as top soil (https://www.fao.org/uploads/media/Harm-World-Soil-DBv7cv_1.pdf)
        is_top_soil: npt.NDArray[np.bool_] = np.zeros_like(
            self.HRU.var.clay, dtype=bool
        )
        is_top_soil[0:3] = True

        thetas: npt.NDArray[np.float32] = thetas_toth(
            soil_organic_carbon=soil_organic_carbon,
            bulk_density=bulk_density,
            is_top_soil=is_top_soil,
            clay=self.HRU.var.clay,
            silt=self.HRU.var.silt,
        )
        thetar: npt.NDArray[np.float32] = thetar_brakensiek(
            sand=self.HRU.var.sand, clay=self.HRU.var.clay, thetas=thetas
        )
        self.HRU.var.bubbling_pressure_cm: npt.NDArray[np.float32] = (
            get_bubbling_pressure(
                clay=self.HRU.var.clay, sand=self.HRU.var.sand, thetas=thetas
            )
        )
        self.HRU.var.lambda_pore_size_distribution: npt.NDArray[np.float32] = (
            get_pore_size_index_brakensiek(
                sand=self.HRU.var.sand, thetas=thetas, clay=self.HRU.var.clay
            )
        )

        # θ saturation, field capacity, wilting point and residual moisture content
        thetafc: npt.NDArray[np.float32] = get_soil_moisture_at_pressure(
            np.float32(-100.0),  # assuming field capacity is at -100 cm (pF 2)
            self.HRU.var.bubbling_pressure_cm,
            thetas,
            thetar,
            self.HRU.var.lambda_pore_size_distribution,
        )

        thetawp: npt.NDArray[np.float32] = get_soil_moisture_at_pressure(
            np.float32(-(10**4.2)),  # assuming wilting point is at -10^4.2 cm (pF 4.2)
            self.HRU.var.bubbling_pressure_cm,
            thetas,
            thetar,
            self.HRU.var.lambda_pore_size_distribution,
        )

        self.HRU.var.ws: npt.NDArray[np.float32] = (
            thetas * self.HRU.var.soil_layer_height
        )
        self.HRU.var.wfc: npt.NDArray[np.float32] = (
            thetafc * self.HRU.var.soil_layer_height
        )
        self.HRU.var.wwp: npt.NDArray[np.float32] = (
            thetawp * self.HRU.var.soil_layer_height
        )
        self.HRU.var.wres: npt.NDArray[np.float32] = (
            thetar * self.HRU.var.soil_layer_height
        )

        # initial soil water storage between field capacity and wilting point
        # set soil moisture to nan where land use is not bioarea
        self.HRU.var.w: npt.NDArray[np.float32] = np.where(
            self.HRU.var.land_use_type[np.newaxis, :] < SEALED,
            (self.HRU.var.wfc - self.HRU.var.wwp) * 0.2 + self.HRU.var.wwp,
            np.nan,
        )
        # for paddy irrigation flooded paddy fields
        self.HRU.var.topwater: npt.NDArray[np.float32] = self.HRU.full_compressed(
            0, dtype=np.float32
        )

        self.HRU.var.ksat: npt.NDArray[np.float32] = kv_brakensiek(
            thetas=thetas, clay=self.HRU.var.clay, sand=self.HRU.var.sand
        )

        # soil water depletion fraction, Van Diepen et al., 1988: WOFOST 6.0, p.86, Doorenbos et. al 1978
        # crop groups for formular in van Diepen et al, 1988
        natural_crop_groups: npt.NDArray[np.float32] = self.hydrology.grid.load(
            self.model.files["grid"]["soil/crop_group"]
        )
        self.HRU.var.natural_crop_groups: npt.NDArray[np.float32] = (
            self.hydrology.to_HRU(data=natural_crop_groups)
        )

        # ------------ Preferential Flow constant ------------------------------------------
        self.var.preferential_flow_constant = float(
            self.model.config["parameters"]["preferentialFlowConstant"]
        )

        self.HRU.var.arnoBeta = self.HRU.full_compressed(np.nan, dtype=np.float32)

        # Improved Arno's scheme parameters: Hageman and Gates 2003
        # arnoBeta defines the shape of soil water capacity distribution curve as a function of  topographic variability
        # b = max( (oh - o0)/(oh + omax), 0.01)
        # oh: the standard deviation of orography, o0: minimum std dev, omax: max std dev
        elevation_std = self.grid.load(
            self.model.files["grid"]["landsurface/elevation_standard_deviation"]
        )
        elevation_std = self.hydrology.to_HRU(data=elevation_std, fn=None)
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

    def initiate_plantfate(self):
        # plantFATE only runs on Linux, so we check if the system is Linux
        assert platform.system() == "Linux", (
            "plantFATE only runs on Linux. Please run the model on a Linux system."
        )

        from . import plantFATE

        self.plantFATE_forest_RUs = np.zeros_like(
            self.HRU.var.land_use_type, dtype=bool
        )

        for i, land_use_type_RU in enumerate(self.HRU.var.land_use_type):
            if (
                (
                    self.model.config["plantFATE"]["n_cells"] == "all"
                    or self.plantFATE_forest_RUs.sum()
                    < self.model.config["plantFATE"]["n_cells"]
                )
                and land_use_type_RU == FOREST
                and self.HRU.var.land_use_ratio[i] > 0.5
            ):
                self.plantFATE_forest_RUs[i] = True

                if self.model.in_spinup:
                    PFconfig_ini = Path(
                        self.model.config["plantFATE"]["spinup_ini_file"]
                    )
                else:
                    PFconfig_ini = Path(self.model.config["plantFATE"]["run_ini_file"])

                if not PFconfig_ini.exists():
                    raise FileNotFoundError(
                        f"plantFATE spinup config file {PFconfig_ini} not found."
                    )

                PFconfig_ini.parent.mkdir(parents=True, exist_ok=True)

                pfModel = plantFATE.Model(PFconfig_ini, False, None)
                pfModel.plantFATE_model.config.parent_dir = (
                    self.model.simulation_root / "plantFATE"
                ).as_posix()
                pfModel.plantFATE_model.config.expt_dir = f"cell_{i}"

                out_dir = Path(self.model.simulation_root / "plantFATE" / f"cell_{i}")
                out_dir.mkdir(parents=True, exist_ok=True)

                pfModel.plantFATE_model.config.out_dir = out_dir.as_posix()
                pfModel.plantFATE_model.config.save_state = False

                if self.model.in_spinup:
                    pfModel.plantFATE_model.config.save_state = True
                else:
                    pfModel.plantFATE_model.config.continuePrevious = True
                    pfModel.plantFATE_model.config.continueFrom_stateFile = str(
                        self.model.simulation_root
                        / ".."
                        / "spinup"
                        / "plantFATE"
                        / f"cell_{i}"
                        / "pf_saved_state.txt"
                    )
                    pfModel.plantFATE_model.config.continueFrom_configFile = str(
                        self.model.simulation_root
                        / ".."
                        / "spinup"
                        / "plantFATE"
                        / f"cell_{i}"
                        / "pf_saved_config.ini"
                    )
                # pfModel.plantFATE_model.config.traits_file = "traits_file_for_cluster"
                self.model.plantFATE.append(pfModel)

            else:
                self.model.plantFATE.append(None)

        # print(len(self.model.plantFATE))
        # print(self.model.plantFATE[0])
        # print(self.model.plantFATE[0:10])
        # print(all(v is None for v in self.model.plantFATE))

    def plant_new_forest(self, indx):
        assert not self.model.in_spinup

        from . import plantFATE

        self.plantFATE_forest_RUs[indx] = True

        PFconfig_ini = self.model.config["plantFATE"]["new_forest_ini_file"]

        pfModel = plantFATE.Model(PFconfig_ini, False, None)
        pfModel.plantFATE_model.config.parent_dir = str(
            self.model.simulation_root / "plantFATE"
        )
        pfModel.plantFATE_model.config.expt_dir = f"cell_{indx}"
        pfModel.plantFATE_model.config.out_dir = str(
            self.model.simulation_root / "plantFATE" / f"cell_{indx}"
        )

        self.model.plantFATE[indx] = pfModel

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

    def calculate_topsoil_volumetric_content(
        self, topsoil_water_content, topsoil_wilting_point, topsoil_fieldcap
    ):
        topsoil_volumetric_content = (topsoil_water_content - topsoil_wilting_point) / (
            topsoil_fieldcap - topsoil_wilting_point
        )
        return topsoil_volumetric_content

    def calculate_net_radiation(
        self, shortwave_radiation_downwelling, longwave_radiation_net, albedo
    ):
        net_radiation = (
            shortwave_radiation_downwelling * (1 - albedo) + longwave_radiation_net
        )  # W/m2
        return net_radiation

    def evapotranspirate_plantFATE(
        self,
        indx,
        plantfate_transpiration,
        plantfate_bare_soil_evaporation,
        plantfate_transpiration_by_layer,
        plantfate_biomass,
        plantfate_co2,
        plantfate_num_ind,
    ):
        if self.plantFATE_forest_RUs[indx]:
            plantFATE_model = self.model.plantFATE[indx]
            if plantFATE_model is not None:
                plantFATE_data = {
                    "soil_water_potential": self.calculate_soil_water_potential_MPa(
                        soil_moisture=self.HRU.var.w[
                            0 : len(self.HRU.var.w), indx
                        ].sum(),
                        soil_moisture_wilting_point=self.HRU.var.wres[
                            0 : len(self.HRU.var.wres), indx
                        ].sum(),
                        soil_moisture_field_capacity=self.HRU.var.wfc[
                            0 : len(self.HRU.var.wfc), indx
                        ].sum(),
                        soil_tickness=self.HRU.var.soil_layer_height[
                            0 : len(self.HRU.var.soil_layer_height), indx
                        ].sum(),
                    ),
                    "vapour_pressure_deficit": self.calculate_vapour_pressure_deficit_kPa(
                        temperature_K=self.HRU.tas[indx],
                        relative_humidity=self.HRU.hurs[indx],
                    )
                    * 1000,  # kPa to Pa
                    "photosynthetic_photon_flux_density": self.calculate_photosynthetic_photon_flux_density(
                        shortwave_radiation=self.HRU.rsds[indx]
                    ),
                    "temperature": self.HRU.tas[indx] - 273.15,  # - 273.15,  # K to C
                    "topsoil_volumetric_water_content": self.calculate_topsoil_volumetric_content(
                        topsoil_water_content=self.HRU.var.w[0, indx],
                        topsoil_wilting_point=self.HRU.var.wres[0, indx],
                        topsoil_fieldcap=self.HRU.var.wfc[0, indx],
                    ),
                    "net_radiation": self.calculate_net_radiation(
                        shortwave_radiation_downwelling=self.HRU.rsds[indx],
                        longwave_radiation_net=self.HRU.rlds[indx],
                        albedo=0.13,  # temporary value for forest
                    ),
                    # "net_radiation": self.grid.net_absorbed_radiation_vegetation_MJ_m2_day[i]
                }

                # print(plantFATE_data)
                if self.model.current_timestep == 0:
                    plantFATE_model.first_step(
                        tstart=self.model.current_time, **plantFATE_data
                    )
                else:
                    # print(indx)
                    (
                        transpiration,
                        plantfate_bare_soil_evaporation[indx],
                        _,
                        _,
                        _,
                    ) = plantFATE_model.step(**plantFATE_data)

                    plantfate_biomass[indx] = plantFATE_model.biomass
                    plantfate_co2[indx] = plantFATE_model.npp
                    plantfate_num_ind[indx] = plantFATE_model.n_individuals

                    total_water_by_layer = (
                        self.HRU.var.w[0 : len(self.HRU.var.w), indx]
                        - self.HRU.var.wres[0 : len(self.HRU.var.w), indx]
                    )
                    # print("PF input data " + str(plantFATE_data))
                    # print("Total Available Water " + str(sum(total_water_by_layer)))
                    # print("Water we are taking " + str(transpiration))
                    if transpiration > sum(total_water_by_layer):
                        # print("Old transpiration: " + str(transpiration))
                        transpiration = min(transpiration, sum(total_water_by_layer))
                        # print("New transpiration: " + str(transpiration))
                        # print("water by layer " + str(total_water_by_layer))
                        # print("water by layer w " + str(self.HRU.var.w[0:len(self.HRU.var.w), indx]))
                        # print("water by layer wres " + str(self.HRU.var.wres[0:len(self.HRU.var.w), indx]))

                    plantfate_transpiration[indx] = transpiration
                    if total_water_by_layer.sum() > 0:
                        for layer in range(N_SOIL_LAYERS):
                            plantfate_transpiration_by_layer[layer, indx] = (
                                plantfate_transpiration[indx]
                                * (
                                    total_water_by_layer[layer]
                                    / total_water_by_layer.sum()
                                )
                            )

                    # print(
                    #     "PlantFATE transpiration: " + str(plantfate_transpiration[indx])
                    # )
                    # print(
                    #     "PlantFATE transpiration by layer "
                    #     + str(plantfate_transpiration_by_layer[:, indx])
                    # )

    def set_global_variables(self):
        # set number of soil layers as global variable for numba
        global N_SOIL_LAYERS
        N_SOIL_LAYERS = self.HRU.var.soil_layer_height.shape[0]

        # set the frost index threshold as global variable for numba
        global FROST_INDEX_THRESHOLD
        FROST_INDEX_THRESHOLD = np.float32(
            self.model.hydrology.snowfrost.var.frost_indexThreshold
        )

    def step(
        self,
        capillary_rise_from_groundwater,
        potential_transpiration,
        potential_bare_soil_evaporation,
        potential_evapotranspiration,
        natural_available_water_infiltration,
        actual_irrigation_consumption,
        crop_factor,
    ):
        """Dynamic part of the soil module.

        For each of the land cover classes the vertical water transport is simulated
        Distribution of water holding capiacity in 3 soil layers based on saturation excess overland flow, preferential flow
        Dependend on soil depth, soil hydraulic parameters
        """
        timer = TimingModule("Soil")

        bioarea = self.HRU.var.land_use_type < SEALED

        assert (self.HRU.var.w[:, bioarea] <= self.HRU.var.ws[:, bioarea]).all()
        assert (self.HRU.var.w[:, bioarea] >= self.HRU.var.wres[:, bioarea]).all()

        if (
            self.model.current_timestep == 0
            and self.model.config["general"]["simulate_forest"]
        ):
            self.initiate_plantfate()

        if __debug__:
            w_pre = self.HRU.var.w.copy()
            topwater_pre = self.HRU.var.topwater.copy()

        if (
            not self.model.in_spinup
            and self.model.config["general"]["simulate_forest"]
            and self.model.config["plantFATE"]["new_forest"]
            and self.model.current_timestep == 0
        ):
            import geopandas as gpd
            from rasterio.features import rasterize
            from shapely.geometry import shape

            forest = gpd.read_file(
                self.model.config["plantFATE"]["new_forest_filename"]
            )
            forest = rasterize(
                [(shape(geom), 1) for geom in forest.geometry],
                out_shape=self.HRU.shape,
                transform=self.HRU.transform,
                fill=False,
                dtype="uint8",  # bool is not supported, so we use uint8 and convert to bool
            ).astype(bool)
            # do not create forests outside the study area
            forest[self.HRU.mask] = False
            # only create forests in grassland or agricultural areas
            forest[
                ~np.isin(
                    self.HRU.decompress(self.HRU.var.land_use_type),
                    [GRASSLAND_LIKE, PADDY_IRRIGATED, NON_PADDY_IRRIGATED],
                )
            ] = False

            import matplotlib.pyplot as plt

            plt.imshow(forest)
            plt.savefig("forest.png")

            new_forest_HRUs = np.unique(self.HRU.var.unmerged_HRU_indices[forest])

            # set the land use type to forest
            self.HRU.var.land_use_type[new_forest_HRUs] = FOREST

            # get the farmers corresponding to the new forest HRUs
            farmers_with_land_converted_to_forest = np.unique(
                self.HRU.var.land_owners[new_forest_HRUs]
            )
            farmers_with_land_converted_to_forest = (
                farmers_with_land_converted_to_forest
            )[farmers_with_land_converted_to_forest != -1]

            HRUs_removed_farmers = self.model.agents.crop_farmers.remove_agents(
                farmers_with_land_converted_to_forest, new_land_use_type=FOREST
            )

            new_forest_HRUs = np.unique(
                np.concatenate([new_forest_HRUs, HRUs_removed_farmers])
            )

            ## NEW PLANTFATE CELL

            for i in new_forest_HRUs:
                self.plant_new_forest(i)
                # self.plantFATE_forest_RUs[new_forest_HRUs] = True

        interflow = self.HRU.full_compressed(0, dtype=np.float32)

        timer = TimingModule("Soil")

        self.grid.vapour_pressure_deficit_KPa = (
            self.calculate_vapour_pressure_deficit_kPa(
                temperature_K=self.grid.tas,
                relative_humidity=self.grid.hurs,
            )
        )
        self.grid.photosynthetic_photon_flux_density_umol_m2_s = (
            self.calculate_photosynthetic_photon_flux_density(
                shortwave_radiation=self.grid.rsds,
                xi=0.5,
            )
        )

        w_forest = self.HRU.var.w.sum(axis=0)
        w_forest[self.HRU.var.land_use_type != FOREST] = np.nan
        w_forest = self.hydrology.to_grid(HRU_data=w_forest, fn="nanmax")

        wwp_forest = self.HRU.var.wwp.sum(axis=0)
        wwp_forest[self.HRU.var.land_use_type != FOREST] = np.nan
        wwp_forest = self.hydrology.to_grid(HRU_data=wwp_forest, fn="nanmax")

        wfc_forest = self.HRU.var.wfc.sum(axis=0)
        wfc_forest[self.HRU.var.land_use_type != FOREST] = np.nan
        wfc_forest = self.hydrology.to_grid(HRU_data=wfc_forest, fn="nanmax")

        soil_height_forest = self.HRU.var.soil_layer_height.sum(axis=0)
        soil_height_forest[self.HRU.var.land_use_type != FOREST] = np.nan
        soil_height_forest = self.hydrology.to_grid(
            HRU_data=soil_height_forest, fn="nanmax"
        )

        self.grid.soil_water_potential_MPa = self.calculate_soil_water_potential_MPa(
            soil_moisture=w_forest,
            soil_moisture_wilting_point=wwp_forest,
            soil_moisture_field_capacity=wfc_forest,
            soil_tickness=soil_height_forest,
        )

        available_water_infiltration, open_water_evaporation = (
            get_available_water_infiltration(
                natural_available_water_infiltration=natural_available_water_infiltration,
                actual_irrigation_consumption=actual_irrigation_consumption,
                land_use_type=self.HRU.var.land_use_type,
                crop_kc=crop_factor,
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

        mask = self.HRU.var.land_use_type < SEALED
        mask_transpiration = self.HRU.var.land_use_type < SEALED
        if self.model.config["general"]["simulate_forest"]:
            mask[self.plantFATE_forest_RUs] = False

        (
            transpiration,
            actual_bare_soil_evaporation,
        ) = evapotranspirate(
            wwp=self.HRU.var.wwp,
            wfc=self.HRU.var.wfc,
            wres=self.HRU.var.wres,
            soil_layer_height=self.HRU.var.soil_layer_height,
            land_use_type=self.HRU.var.land_use_type,
            root_depth=self.HRU.var.root_depth,
            crop_map=self.HRU.var.crop_map,
            natural_crop_groups=self.HRU.var.natural_crop_groups,
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
            minimum_effective_root_depth=self.var.minimum_effective_root_depth,
            mask=mask,
            mask_transpiration=mask_transpiration,
        )
        assert transpiration.dtype == np.float32
        assert (self.HRU.var.w[:, bioarea] <= self.HRU.var.ws[:, bioarea]).all()
        assert (self.HRU.var.w[:, bioarea] >= self.HRU.var.wres[:, bioarea]).all()

        if self.model.config["general"]["simulate_forest"]:
            plantfate_transpiration = np.zeros(len(self.plantFATE_forest_RUs))
            plantfate_bare_soil_evaporation = np.zeros(len(self.plantFATE_forest_RUs))
            plantfate_transpiration_by_layer = np.zeros_like(self.HRU.var.w)
            plantfate_biomass = np.zeros(len(self.plantFATE_forest_RUs))
            plantfate_co2 = np.zeros(len(self.plantFATE_forest_RUs))
            plantfate_num_ind = np.zeros(len(self.plantFATE_forest_RUs))
            # import multiprocessing
            #
            # print(plantfate_transpiration[self.plantFATE_forest_RUs])
            #
            # # def evap_multi_plantfate(i):
            # #     self.evapotranspirate_plantFATE(i, plantfate_transpiration, plantfate_bare_soil_evaporation, plantfate_transpiration_by_layer)
            # #
            # # multiprocessing.set_start_method('fork')
            # # pool_obj = multiprocessing.Pool()
            # # pool_obj.map(evap_multi_plantfate, range(len(self.plantFATE_forest_RUs)))
            #
            # cpus = multiprocessing.cpu_count()
            # processes = [multiprocessing.Process(target = self.evapotranspirate_plantFATE, args = (i, plantfate_transpiration, plantfate_bare_soil_evaporation, plantfate_transpiration_by_layer)) for i in range(len(self.plantFATE_forest_RUs))]

            # print(np.where(new_forest_HRUs == 425))

            for i in tqdm(range(len(self.plantFATE_forest_RUs))):
                # print("New Forest " + str(i))
                self.evapotranspirate_plantFATE(
                    i,
                    plantfate_transpiration,
                    plantfate_bare_soil_evaporation,
                    plantfate_transpiration_by_layer,
                    plantfate_biomass,
                    plantfate_co2,
                    plantfate_num_ind,
                )

            # for p in processes:
            #     p.start()
            # for p in processes:
            #     p.join()
            #
            # print(plantfate_transpiration[self.plantFATE_forest_RUs])

            transpiration += plantfate_transpiration
            self.HRU.var.w -= plantfate_transpiration_by_layer

            assert np.allclose(
                plantfate_transpiration, plantfate_transpiration_by_layer.sum(axis=0)
            )

            self.grid.plantFATE_biomass = self.hydrology.to_grid(
                HRU_data=plantfate_biomass, fn="weightedmean"
            )
            self.grid.plantFATE_NPP = self.hydrology.to_grid(
                HRU_data=plantfate_co2, fn="weightedmean"
            )
            self.grid.plantFATE_num_ind = self.hydrology.to_grid(
                HRU_data=plantfate_num_ind, fn="weightedmean"
            )

        # actual_bare_soil_evaporation += plantfate_bare_soil_evaporation
        # print(plantfate_bare_soil_evaporation)

        timer.new_split("Evapotranspiration")

        n_substeps = 3
        preferential_flow = np.zeros_like(self.HRU.var.land_use_type, dtype=np.float32)
        direct_runoff = np.zeros_like(self.HRU.var.land_use_type, dtype=np.float32)
        groundwater_recharge = np.zeros_like(
            self.HRU.var.land_use_type, dtype=np.float32
        )
        #
        # print(self.HRU.var.w[:, self.plantFATE_forest_RUs])
        # print(self.HRU.var.ws[:, self.plantFATE_forest_RUs])
        # print(self.HRU.var.wres[:, self.plantFATE_forest_RUs])

        assert (self.HRU.var.w[:, bioarea] <= self.HRU.var.ws[:, bioarea]).all()
        assert (self.HRU.var.w[:, bioarea] >= self.HRU.var.wres[:, bioarea]).all()

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
                np.float32(self.var.preferential_flow_constant),
                self.HRU.var.w,
                self.HRU.var.topwater,
                self.HRU.var.soil_layer_height,
            )

            preferential_flow += preferential_flow_substep
            direct_runoff += direct_runoff_substep
            groundwater_recharge[bioarea] += groundwater_recharge_substep[bioarea]

        assert (self.HRU.var.w[:, bioarea] <= self.HRU.var.ws[:, bioarea]).all()
        assert (self.HRU.var.w[:, bioarea] >= self.HRU.var.wres[:, bioarea]).all()
        timer.new_split("Vertical transport")

        assert (self.HRU.var.w[:, bioarea] <= self.HRU.var.ws[:, bioarea]).all()
        assert (self.HRU.var.w[:, bioarea] >= self.HRU.var.wres[:, bioarea]).all()

        runoff = direct_runoff + runoff_from_groundwater

        assert preferential_flow.dtype == np.float32
        assert runoff.dtype == np.float32

        if __debug__:
            assert (self.HRU.var.w[:, bioarea] <= self.HRU.var.ws[:, bioarea]).all()
            assert (self.HRU.var.w[:, bioarea] >= self.HRU.var.wres[:, bioarea]).all()
            assert (interflow == 0).all()  # interflow is not implemented (see above)
            balance_check(
                name="soil_1",
                how="cellwise",
                influxes=[
                    natural_available_water_infiltration[bioarea],
                    capillary_rise_from_groundwater[bioarea],
                    actual_irrigation_consumption[bioarea],
                ],
                outfluxes=[
                    runoff[bioarea],
                    interflow[bioarea],
                    groundwater_recharge[bioarea],
                    transpiration[bioarea],
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
                    natural_available_water_infiltration[bioarea],
                    capillary_rise_from_groundwater[bioarea],
                    actual_irrigation_consumption[bioarea],
                ],
                outfluxes=[
                    runoff[bioarea],
                    interflow[bioarea],
                    groundwater_recharge[bioarea],
                    transpiration[bioarea],
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

            # print(transpiration[self.plantFATE_forest_RUs])
            # print(potential_transpiration[self.plantFATE_forest_RUs])
            # print(self.var.w[0:len(self.var.w), self.plantFATE_forest_RUs] - self.wres[0:len(self.var.w), self.plantFATE_forest_RUs])
            # print(self.var.w[0:len(self.var.w), self.plantFATE_forest_RUs])
            # print(self.wres[0:len(self.wres), self.plantFATE_forest_RUs])
            #
            # if self.model.config["general"]["simulate_forest"]:
            #     assert (
            #         transpiration[self.plantFATE_forest_RUs]
            #         <= potential_transpiration[self.plantFATE_forest_RUs] + 1e-7
            #     ).all()
            #
            # assert (
            #     transpiration[bioarea]
            #     <= potential_transpiration[bioarea] + 1e-7
            # ).all()
            # assert (
            #     actual_bare_soil_evaporation[bioarea]
            #     <= potential_bare_soil_evaporation[bioarea] + 1e-7
            # ).all()

        timer.new_split("Finalizing")
        if self.model.timing:
            print(timer)

        if self.model.config["general"]["simulate_forest"]:
            # Soil moisture
            soil_moisture = np.nan_to_num(self.HRU.var.w.sum(axis=0))

            soil_moisture_forest_HRU = soil_moisture.copy()
            soil_moisture_forest_HRU[self.HRU.var.land_use_type != FOREST] = np.nan
            soil_moisture_forest_grid = self.hydrology.to_grid(
                HRU_data=soil_moisture_forest_HRU, fn="weightednanmean"
            )
            soil_moisture_forest_plantFATE_HRU = soil_moisture.copy()
            soil_moisture_forest_plantFATE_HRU[~self.plantFATE_forest_RUs] = np.nan
            soil_moisture_forest_plantFATE_grid = self.hydrology.to_grid(
                HRU_data=soil_moisture_forest_plantFATE_HRU, fn="weightednanmean"
            )

            actual_bare_soil_evaporation_forest_HRU = (
                actual_bare_soil_evaporation.copy()
            )
            actual_bare_soil_evaporation_forest_HRU[
                self.HRU.var.land_use_type != FOREST
            ] = np.nan
            actual_bare_soil_evaporation_forest_grid = self.hydrology.to_grid(
                HRU_data=actual_bare_soil_evaporation_forest_HRU, fn="weightednanmean"
            )

            # Bare soil evaporation
            actual_bare_soil_evaporation_forest_plantFATE_HRU = (
                actual_bare_soil_evaporation.copy()
            )
            actual_bare_soil_evaporation_forest_plantFATE_HRU[
                ~self.plantFATE_forest_RUs
            ] = np.nan
            actual_bare_soil_evaporation_forest_plantFATE_grid = self.hydrology.to_grid(
                HRU_data=actual_bare_soil_evaporation_forest_plantFATE_HRU,
                fn="weightednanmean",
            )

            # Transpiration
            transpiration_forest_HRU = transpiration.copy()
            transpiration_forest_HRU[self.HRU.var.land_use_type != FOREST] = np.nan
            transpiration_forest_grid = self.hydrology.to_grid(
                HRU_data=transpiration_forest_HRU, fn="weightednanmean"
            )
            transpiration_forest_plantFATE_HRU = transpiration.copy()
            transpiration_forest_plantFATE_HRU[~self.plantFATE_forest_RUs] = np.nan
            transpiration_forest_plantFATE_grid = self.hydrology.to_grid(
                HRU_data=transpiration_forest_plantFATE_HRU, fn="weightednanmean"
            )

            # Groundwater recharge
            groundwater_recharge_forest_HRU = groundwater_recharge.copy()
            groundwater_recharge_forest_HRU[self.HRU.var.land_use_type != FOREST] = (
                np.nan
            )
            groundwater_recharge_forest_grid = self.hydrology.to_grid(
                HRU_data=groundwater_recharge_forest_HRU, fn="weightednanmean"
            )
            groundwater_recharge_forest_plantFATE_HRU = groundwater_recharge.copy()
            groundwater_recharge_forest_plantFATE_HRU[~self.plantFATE_forest_RUs] = (
                np.nan
            )
            groundwater_recharge_forest_plantFATE_grid = self.hydrology.to_grid(
                HRU_data=groundwater_recharge_forest_plantFATE_HRU, fn="weightednanmean"
            )

        self.report(self, locals())

        return (
            interflow,
            runoff,
            groundwater_recharge,
            open_water_evaporation,
            transpiration,
            actual_bare_soil_evaporation,
        )
