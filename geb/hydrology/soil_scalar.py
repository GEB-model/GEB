import numpy as np
import numpy.typing as npt
from numba import njit

from .landcovers import OPEN_WATER, PADDY_IRRIGATED, SEALED

# TODO: Load this dynamically as global var (see soil.py)
N_SOIL_LAYERS = 6


@njit(cache=True, inline="always")
def add_water_to_topwater_and_evaporate_open_water(
    natural_available_water_infiltration_m: np.float32,
    actual_irrigation_consumption_m: np.float32,
    land_use_type: np.int32,
    reference_evapotranspiration_water_m_per_day: np.float32,
    topwater_m: np.float32,
) -> tuple[np.float32, np.float32]:
    """Add available water from natural and innatural sources to the topwater and calculate open water evaporation.

    Args:
        natural_available_water_infiltration_m: The natural available water infiltration in m.
        actual_irrigation_consumption_m: The actual irrigation consumption in m.
        land_use_type: The land use type of the hydrological response unit.
        reference_evapotranspiration_water_m_per_day: The reference evapotranspiration from water in m.
        topwater_m: The topwater in m, which is the water available for evaporation and transpiration.

    Returns:
        A tuple containing:
            - The updated topwater in m
            - The open water evaporation in m, which is the water evaporated from open water areas.
    """
    # Add water to topwater
    topwater_m += (
        natural_available_water_infiltration_m + actual_irrigation_consumption_m
    )

    # Calculate open water evaporation based on land use type
    if land_use_type == PADDY_IRRIGATED:
        open_water_evaporation_m = min(
            max(np.float32(0.0), topwater_m),
            reference_evapotranspiration_water_m_per_day,
        )
    elif land_use_type == SEALED:
        # evaporation from precipitation fallen on sealed area (ponds)
        # estimated as 0.2 x reference_evapotranspiration_water_m_per_day
        open_water_evaporation_m = min(
            np.float32(0.2) * reference_evapotranspiration_water_m_per_day, topwater_m
        )
    else:
        # no open water evaporation for other land use types (thus using default of 0)
        # note that evaporation from open water and channels is calculated in the routing module
        open_water_evaporation_m = np.float32(0.0)

    # Subtract evaporation from topwater
    topwater_m -= open_water_evaporation_m

    return topwater_m, open_water_evaporation_m


@njit(cache=True, inline="always")
def rise_from_groundwater(
    w: npt.NDArray[np.float32],
    ws: npt.NDArray[np.float32],
    capillary_rise_from_groundwater: np.float32,
) -> np.float32:
    """Adds capillary rise from groundwater to the bottom soil layer and moves excess water upwards for a single cell.

    This function modifies the soil water content array in place for the given cell and returns the runoff from groundwater.

    Args:
        w: Soil water content in each layer for the cell in meters, shape (N_SOIL_LAYERS,).
        ws: Saturated soil water content in each layer for the cell in meters, shape (N_SOIL_LAYERS,).
        capillary_rise_from_groundwater: Capillary rise from groundwater for the cell in meters.

    Returns:
        The runoff from groundwater for the cell in meters, which is the excess water that cannot be stored in the soil layers.
    """
    bottom_soil_layer_index: int = N_SOIL_LAYERS - 1
    runoff_from_groundwater: np.float32 = np.float32(0.0)

    # Add capillary rise to the bottom soil layer
    w[bottom_soil_layer_index] += capillary_rise_from_groundwater

    # If the bottom soil layer is full, send water to the above layer, repeat until top layer
    for j in range(bottom_soil_layer_index, 0, -1):
        if w[j] > ws[j]:
            excess_water = w[j] - ws[j]
            w[j - 1] += excess_water  # Move excess water to layer above
            w[j] = ws[j]  # Set the current layer to full

    # If the top layer is full, send water to the runoff
    # TODO: Send to topwater instead of runoff if paddy irrigated
    if w[0] > ws[0]:
        runoff_from_groundwater = w[0] - ws[0]  # Move excess water to runoff
        w[0] = ws[0]  # Set the top layer to full

    return runoff_from_groundwater


@njit(cache=True, inline="always")
def get_infiltration_capacity(
    saturated_hydraulic_conductivity_cell: np.float32,
) -> np.float32:
    """Calculate the infiltration capacity for a single cell.

    TODO: This function is a placeholder for more complex logic should be added later.

    Args:
        saturated_hydraulic_conductivity_cell: Saturated hydraulic conductivity for the cell.

    Returns:
        The infiltration capacity for the cell.
    """
    return saturated_hydraulic_conductivity_cell[0]


@njit(cache=True, inline="always")
def infiltration(
    ws: npt.NDArray[np.float32],
    saturated_hydraulic_conductivity: np.float32,
    land_use_type: np.int32,
    frost_index: np.float32,
    w: npt.NDArray[np.float32],
    topwater_m: np.float32,
) -> tuple[np.float32, np.float32, np.float32]:
    """Simulates vertical transport of water in the soil for a single cell.

    Args:
        ws: Saturated soil water content in each layer for the cell in meters, shape (N_SOIL_LAYERS,).
        saturated_hydraulic_conductivity: Saturated hydraulic conductivity for the cell in m/day.
        land_use_type: Land use type for the cell.
        frost_index: Frost index for the cell.
        w: Soil water content in each layer for the cell in meters, shape (N_SOIL_LAYERS,), modified in place.
        topwater_m: Topwater for the cell in meters, modified in place.
        soil_layer_height: Soil layer heights for the cell in meters, shape (N_SOIL_LAYERS,).

    Returns:
        A tuple containing:
            - direct_runoff: Direct runoff from the cell in meters.
            - groundwater_recharge: Groundwater recharge from the cell in meters (currently set to 0.0).
            - infiltration: Infiltration into the soil for the cell in meters.
    """
    # Constants
    # TODO: Set this from a global var (see soil.py)
    FROST_INDEX_THRESHOLD = np.float32(85.0)

    # Calculate potential infiltration for the cell
    potential_infiltration = get_infiltration_capacity(saturated_hydraulic_conductivity)

    # Check if soil is frozen
    soil_is_frozen = frost_index > FROST_INDEX_THRESHOLD

    # Calculate infiltration for the cell
    infiltration_amount = min(
        potential_infiltration
        * ~soil_is_frozen
        * ~(land_use_type == SEALED)  # no infiltration on sealed areas
        * ~(land_use_type == OPEN_WATER),  # no infiltration on open water
        topwater_m,
    )

    remaining_infiltration = np.float32(infiltration_amount)
    for layer in range(N_SOIL_LAYERS):
        capacity = ws[layer] - w[layer]
        if remaining_infiltration > capacity:
            w[layer] = ws[layer]  # fill the layer to capacity
            remaining_infiltration -= capacity
        else:
            w[layer] += remaining_infiltration
            remaining_infiltration = np.float32(0)
            w[layer] = min(w[layer], ws[layer])
            break

    infiltration_amount -= remaining_infiltration
    topwater_m -= infiltration_amount

    # Calculate direct runoff
    direct_runoff = max(
        np.float32(0),
        topwater_m - np.float32(0.05) * (land_use_type == PADDY_IRRIGATED),
    )
    topwater_m -= direct_runoff
    return topwater_m, direct_runoff, np.float32(0.0), infiltration_amount


@njit(cache=True, inline="always")
def get_soil_water_flow_parameters(
    w: np.float32,
    wres: np.float32,
    ws: np.float32,
    lambda_pore_size_distribution: np.float32,
    saturated_hydraulic_conductivity: np.float32,
    bubbling_pressure_cm: np.float32,
) -> tuple[np.float32, np.float32]:
    """Calculate the soil water potential and unsaturated hydraulic conductivity for a single soil layer.

    Notes:
        - psi is cutoff at MAX_SUCTION_METERS because the van Genuchten model predicts infinite suction
        for very dry soils.

    Args:
        w: Soil water content in the layer in meters.
        wres: Residual soil water content in the layer in meters.
        ws: Saturated soil water content in the layer in meters.
        lambda_pore_size_distribution: Van Genuchten parameter lambda for the layer.
        saturated_hydraulic_conductivity: Saturated hydraulic conductivity for the layer in m/timestep
        bubbling_pressure_cm: Bubbling pressure for the layer in cm.

    Returns:
        A tuple containing:
            - psi: Soil water potential in the layer in meters (negative value for suction).
            - unsaturated_hydraulic_conductivity: Unsaturated hydraulic conductivity in the layer in m/timestep.

    """
    # oven-dried soil has a suction of 1 GPa, which is about 100000 m water column
    max_suction_meters = np.float32(1_000_000_000 / 1_000 / 9.81)

    # Compute effective saturation
    effective_saturation = (w - wres) / (ws - wres)
    effective_saturation = np.maximum(effective_saturation, np.float32(1e-9))
    effective_saturation = np.minimum(effective_saturation, np.float32(1))

    # Compute parameters n and m
    n = lambda_pore_size_distribution + np.float32(1)
    m = np.float32(1) - np.float32(1) / n

    # Compute unsaturated hydraulic conductivity
    term1 = saturated_hydraulic_conductivity * np.sqrt(effective_saturation)
    term2 = (
        np.float32(1)
        - np.power(
            (np.float32(1) - np.power(effective_saturation, (np.float32(1) / m))),
            m,
        )
    ) ** 2

    unsaturated_hydraulic_conductivity = term1 * term2

    alpha = np.float32(1) / (bubbling_pressure_cm / 100)  # convert cm to m

    # Compute capillary pressure head (phi)
    phi_power_term = np.power(effective_saturation, (-np.float32(1) / m))
    phi = (
        np.power(phi_power_term - np.float32(1), (np.float32(1) / n)) / alpha
    )  # Positive value
    phi = np.minimum(phi, max_suction_meters)  # Limit to maximum suction

    # Soil water potential (negative value for suction)
    psi = -phi

    return psi, unsaturated_hydraulic_conductivity


@njit(cache=True, inline="always")
def get_mean_unsaturated_hydraulic_conductivity(
    unsaturated_hydraulic_conductivity_1: np.float32,
    unsaturated_hydraulic_conductivity_2: np.float32,
) -> np.float32:
    """Calculate the mean unsaturated hydraulic conductivity between two soil layers using the geometric mean.

    Args:
        unsaturated_hydraulic_conductivity_1: Unsaturated hydraulic conductivity of the first soil layer.
        unsaturated_hydraulic_conductivity_2: Unsaturated hydraulic conductivity of the second soil layer.

    Returns:
        The mean unsaturated hydraulic conductivity between the two soil layers.
    """
    # harmonic mean
    # mean_unsaturated_hydraulic_conductivity = (
    #     2
    #     * unsaturated_hydraulic_conductivity_1
    #     * unsaturated_hydraulic_conductivity_2
    #     / (unsaturated_hydraulic_conductivity_1 + unsaturated_hydraulic_conductivity_2)
    # )
    # geometric mean
    mean_unsaturated_hydraulic_conductivity = np.sqrt(
        unsaturated_hydraulic_conductivity_1 * unsaturated_hydraulic_conductivity_2
    )
    # ensure that there is some minimum flow is possible
    mean_unsaturated_hydraulic_conductivity = max(
        mean_unsaturated_hydraulic_conductivity, np.float32(1e-9)
    )
    return mean_unsaturated_hydraulic_conductivity


@njit(cache=True, inline="always")
def get_flux(
    mean_unsaturated_hydraulic_conductivity: np.float32,
    psi_lower: np.float32,
    psi_upper: np.float32,
    delta_z: np.float32,
) -> np.float32:
    """Calculate the flux between two soil layers using Darcy's law.

    Args:
        mean_unsaturated_hydraulic_conductivity: Mean unsaturated hydraulic conductivity between the two soil layers.
        psi_lower: Soil water potential of the lower soil layer.
        psi_upper: Soil water potential of the upper soil layer.
        delta_z: Distance between the two soil layers.

    Returns:
        The flux between the two soil layers.
    """
    return -mean_unsaturated_hydraulic_conductivity * (
        (psi_lower - psi_upper) / delta_z - np.float32(1)
    )
