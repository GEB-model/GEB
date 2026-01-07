"""Soil hydrology functions."""

import numpy as np
from numba import njit

from geb.types import ArrayFloat32, Shape

from .landcovers import OPEN_WATER, PADDY_IRRIGATED, SEALED

# TODO: Load this dynamically as global var (see soil.py)
N_SOIL_LAYERS = 6


@njit(cache=True, inline="always")
def add_water_to_topwater_and_evaporate_open_water(
    natural_available_water_infiltration_m: np.float32,
    actual_irrigation_consumption_m: np.float32,
    land_use_type: np.int32,
    reference_evapotranspiration_water_m: np.float32,
    topwater_m: np.float32,
) -> tuple[np.float32, np.float32]:
    """Add available water from natural and innatural sources to the topwater and calculate open water evaporation.

    Args:
        natural_available_water_infiltration_m: The natural available water infiltration in m.
        actual_irrigation_consumption_m: The actual irrigation consumption in m.
        land_use_type: The land use type of the hydrological response unit.
        reference_evapotranspiration_water_m: The reference evapotranspiration from water in m.
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
            reference_evapotranspiration_water_m,
        )
    elif land_use_type == SEALED:
        # evaporation from precipitation fallen on sealed area (ponds)
        # estimated as 0.2 x reference evapotranspiration from water
        open_water_evaporation_m = min(
            np.float32(0.2) * reference_evapotranspiration_water_m, topwater_m
        )
    else:
        # no open water evaporation for other land use types (thus using default of 0)
        # note that evaporation from open water and channels is calculated in the routing module
        open_water_evaporation_m = np.float32(0.0)

    # Subtract evaporation from topwater
    topwater_m -= open_water_evaporation_m

    return topwater_m, open_water_evaporation_m


@njit(cache=True, inline="always")
def calculate_arno_runoff(
    current_soil_water_storage: np.float32,
    max_soil_water_capacity: np.float32,
    arno_shape_parameter: np.float32,
    topwater_m: np.float32,
    infiltration_capacity_m: np.float32,
) -> tuple[np.float32, np.float32]:
    """Calculate runoff and infiltration using the Arno/Xinanjiang scheme.

    This implementation uses the analytical solution for the Xinanjiang model,
    calculating the change in storage by moving along the capacity distribution curve.

    Derivation:
        The model assumes a spatial distribution of point infiltration capacities across the basin.
        This distribution is described by a probability density function, where the fraction of the basin
        with a capacity less than or equal to a value 'i' is given by a power law.

        Definitions:
        - i: Point infiltration capacity (or local storage capacity) at a specific fraction of the basin.
        - i_max: The maximum point infiltration capacity in the basin.
        - b: The shape parameter of the distribution.
        - ws: The maximum total water storage capacity of the basin (average of i over the basin).
        - w: The current total water storage of the basin.

        1. Distribution of Capacities:
        The fraction of the basin area (As) with infiltration capacity less than or equal to i is assumed to be:
            As = 1 - (1 - i / i_max)^b
        This implies that a small portion of the basin has very low capacity, while most has higher capacity, controlled by 'b'.

        2. Total Basin Capacity (ws):
        The maximum storage capacity of the basin (ws) is the integral of the capacity 'i' over the entire area (from As=0 to As=1).
        First, express i as a function of As by inverting the distribution equation:
            i(As) = i_max * [1 - (1 - As)^(1/b)]
        Then, integrate i(As) from 0 to 1:
            ws = ∫₀¹ { i_max * [1 - (1 - As)^(1/b)] } dAs
            ws = i_max * [ 1 - b / (b + 1) ]
            ws = i_max / (b + 1)
        Rearranging gives:
            i_max = ws * (b + 1)

        3. Current Basin Storage (w):
        The current storage 'w' is the integral of the capacity curve up to the current saturation point 'i'.
        It is derived by calculating the storage deficit (empty space) and subtracting it from the total capacity 'ws'.
        The deficit is the integral of remaining capacity (i(As) - i) over the unsaturated area (from As to 1).
        Here, i(As) is the local capacity at fraction As (as defined in step 2).
            Deficit = ∫_As^1 (i(As) - i) dAs
        Substituting i(As) = i_max * [1 - (1 - As)^(1/b)] and letting u = 1 - As:
            Deficit = ∫_0^(1-As) (i_max * [1 - u^(1/b)] - i) du
        Solving this integral and using the relation (1 - As) = (1 - i / i_max)^b yields:
            Deficit = ws * (1 - i / i_max)^(b + 1)
        Thus, the current storage is:
            w = ws - Deficit = ws * [1 - (1 - i / i_max)^(b + 1)]

        4. Inverting for Relative Saturation:
        To find the current position on the capacity curve based on the current storage, we invert the equation:
            1 - i / i_max = (1 - w / ws)^(1 / (b + 1))

        5. Adding Precipitation:
        The model assumes that precipitation 'p' is added uniformly to the soil water storage across the basin.
        Conceptually, 'i' represents the current "filled depth" or tension water level in the soil column.
        Since 'i' is measured in depth units (e.g., mm), adding precipitation 'p' directly increases this level.
        This assumes that infiltration happens uniformly until local capacity is reached.
        Therefore, the state of the basin moves along the capacity curve from 'i' to 'i + p':
            i_new = i_current + p

        6. Calculating New Storage:
        Substituting 'i_new' back into the storage equation gives the new total storage 'w_new'.
        We replace (1 - i_current / i_max) with the term derived in step 4.
            w_new = ws * [1 - ( (1 - w / ws)^(1 / (b + 1)) - p / i_max )^(b + 1) ]

    References:
    - Zhao Ren-Jun (1992). The Xinanjiang model applied in China. Journal of hydrology, 135(1-4), 371-381.
    - Todini, E. (1996). The ARNO rainfall—runoff model. Journal of hydrology, 175(1-4), 339-382.

    Args:
        current_soil_water_storage: Current soil water content (m).
        max_soil_water_capacity: Maximum soil water capacity (m).
        arno_shape_parameter: Arno shape parameter.
        topwater_m: Incoming water (precipitation) (m).
        infiltration_capacity_m: Maximum infiltration capacity (m).

    Returns:
        A tuple containing:
            - Runoff (m)
            - Infiltration (m)
    """
    # If precipitation is 0, no runoff
    if topwater_m <= np.float32(0.0):
        return np.float32(0.0), np.float32(0.0)

    # Relative saturation
    relative_saturation = current_soil_water_storage / max_soil_water_capacity
    if relative_saturation >= np.float32(1.0):
        # Already saturated, all P becomes runoff
        return topwater_m, np.float32(0.0)

    # Calculate the term (1 - w/ws)^(1/(b+1))
    # This corresponds to (1 - i / i_max)
    # where i_max = ws * (b + 1)
    # Step 4: Inverting for Relative Saturation (finding position on capacity curve)
    term_current = (np.float32(1.0) - relative_saturation) ** (
        np.float32(1.0) / (arno_shape_parameter + np.float32(1.0))
    )

    # Calculate the new term after adding precipitation 'p'
    # We are filling the capacity, so we move up the capacity curve (increasing i)
    # The amount of "capacity depth" filled is 'p'.
    # Step 2 & 3: Total Basin Capacity (i_max)
    max_infiltration_capacity = max_soil_water_capacity * (
        arno_shape_parameter + np.float32(1.0)
    )

    # The new term corresponds to (1 - i_new / i_max)
    # i_new = i + p
    # (1 - i_new/i_max) = (1 - i/i_max) - p/i_max
    # Step 5: Adding Precipitation (moving along the capacity curve)
    term_new = term_current - topwater_m / max_infiltration_capacity

    if term_new <= np.float32(0.0):
        # Saturated
        new_soil_water_storage = max_soil_water_capacity
    else:
        # Calculate new storage w_new
        # w_new = ws * (1 - term_new^(b+1))
        # Step 6: Calculating New Storage
        new_soil_water_storage = max_soil_water_capacity * (
            np.float32(1.0) - term_new ** (arno_shape_parameter + np.float32(1.0))
        )

    infiltration_amount_m: np.float32 = (
        new_soil_water_storage - current_soil_water_storage
    )
    infiltration_amount_m = max(infiltration_amount_m, np.float32(0.0))
    infiltration_amount_m = min(infiltration_amount_m, topwater_m)

    # Limit infiltration by infiltration capacity
    if infiltration_amount_m > infiltration_capacity_m:
        infiltration_amount_m = infiltration_capacity_m

    runoff_m: np.float32 = topwater_m - infiltration_amount_m

    return runoff_m, infiltration_amount_m


@njit(cache=True, inline="always")
def rise_from_groundwater(
    w: ArrayFloat32,
    ws: ArrayFloat32,
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
    saturated_hydraulic_conductivity: ArrayFloat32,
) -> np.float32:
    """Calculate the infiltration capacity for a single cell.

    TODO: This function is a placeholder for more complex logic should be added later.

    Args:
        saturated_hydraulic_conductivity: Saturated hydraulic conductivity for the cell.

    Returns:
        The infiltration capacity for the cell.
    """
    return saturated_hydraulic_conductivity[0]


@njit(cache=True, inline="always")
def infiltration(
    ws: ArrayFloat32,
    saturated_hydraulic_conductivity: ArrayFloat32,
    land_use_type: np.int32,
    soil_is_frozen: bool,
    w: ArrayFloat32,
    topwater_m: np.float32,
    arno_shape_parameter: np.float32,
) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    """Simulates vertical transport of water in the soil for a single cell.

    Args:
        ws: Saturated soil water content in each layer for the cell in meters, shape (N_SOIL_LAYERS,).
        saturated_hydraulic_conductivity: Saturated hydraulic conductivity for the cell in m in this timestep.
        land_use_type: Land use type for the cell.
        soil_is_frozen: Boolean indicating if the soil is frozen.
        w: Soil water content in each layer for the cell in meters, shape (N_SOIL_LAYERS,), modified in place.
        topwater_m: Topwater for the cell in meters, modified in place.
        arno_shape_parameter: Arno shape parameter for the cell.

    Returns:
        A tuple containing:
            - topwater_m: Updated topwater in meters.
            - direct_runoff: Direct runoff from the cell in meters.
            - groundwater_recharge: Groundwater recharge from the cell in meters (currently set to 0.0).
            - infiltration: Infiltration into the soil for the cell in meters.
    """
    if soil_is_frozen or land_use_type == SEALED or land_use_type == OPEN_WATER:
        # No infiltration allowed
        infiltration_amount: np.float32 = np.float32(0.0)
        direct_runoff: np.float32 = topwater_m
        topwater_m: np.float32 = np.float32(0.0)
        return topwater_m, direct_runoff, np.float32(0.0), infiltration_amount
    else:
        direct_runoff, infiltration_amount = calculate_arno_runoff(
            current_soil_water_storage=w[0] + w[1],
            max_soil_water_capacity=ws[0] + ws[1],
            arno_shape_parameter=arno_shape_parameter,
            topwater_m=topwater_m,
            infiltration_capacity_m=saturated_hydraulic_conductivity[0],
        )
        toplayer_infiltration = min(infiltration_amount, ws[0] - w[0])
        second_layer_infiltration = infiltration_amount - toplayer_infiltration

        w[0] += toplayer_infiltration
        # Ensure we don't exceed saturation due to float errors
        w[0] = min(w[0], ws[0])

        w[1] += second_layer_infiltration
        # Ensure we don't exceed saturation due to float errors
        w[1] = min(w[1], ws[1])

        # In Arno scheme, all topwater is processed into runoff or infiltration
        topwater_m: np.float32 = np.float32(0.0)

        if land_use_type == PADDY_IRRIGATED:
            ponding_allowance: np.float32 = np.float32(0.05)
            ponding = min(direct_runoff, ponding_allowance)
            topwater_m += ponding
            direct_runoff -= ponding

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


@njit(
    cache=True,
    inline="always",
)
def get_soil_moisture_at_pressure(
    capillary_suction: np.float32,
    bubbling_pressure_cm: np.ndarray[Shape, np.dtype[np.float32]],
    thetas: np.ndarray[Shape, np.dtype[np.float32]],
    thetar: np.ndarray[Shape, np.dtype[np.float32]],
    lambda_: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Calculates the soil moisture content at a given soil water potential (capillary suction) using the van Genuchten model.

    Args:
        capillary_suction: The soil water potential (capillary suction) (m)
        bubbling_pressure_cm: The bubbling pressure (cm)
        thetas: The saturated soil moisture content (m³/m³)
        thetar: The residual soil moisture content (m³/m³)
        lambda_: The van Genuchten parameter lambda (1/m)

    Returns:
        The soil moisture content at the given soil water potential (m³/m³)
    """
    alpha = np.float32(1) / bubbling_pressure_cm
    n = lambda_ + np.float32(1)
    m = np.float32(1) - np.float32(1) / n
    phi: np.float32 = -capillary_suction

    water_retention_curve = (np.float32(1) / (np.float32(1) + (alpha * phi) ** n)) ** m

    return water_retention_curve * (thetas - thetar) + thetar


def clip_brakensiek(
    clay: np.ndarray[Shape, np.dtype[np.float32]],
    sand: np.ndarray[Shape, np.dtype[np.float32]],
) -> tuple[
    np.ndarray[Shape, np.dtype[np.float32]], np.ndarray[Shape, np.dtype[np.float32]]
]:
    """Clip clay and sand percentages for Brakensiek pedotransfer functions.

    The Brakensiek functions expect clay in [5, 60] and sand in [5, 70].

    Args:
        clay: Clay percentage array [%].
        sand: Sand percentage array [%].

    Returns:
        Tuple of (clay_clipped, sand_clipped) with values clipped to the valid ranges.
    """
    clay_out = np.empty_like(clay)
    sand_out = np.empty_like(sand)

    np.clip(clay, np.float32(5), np.float32(60), out=clay_out)
    np.clip(sand, np.float32(5), np.float32(70), out=sand_out)

    return clay_out, sand_out


def thetas_toth(
    organic_carbon_percentage: np.ndarray[Shape, np.dtype[np.float32]],
    bulk_density: np.ndarray[Shape, np.dtype[np.float32]],
    is_top_soil: np.ndarray[Shape, np.dtype[np.bool_]],
    clay: np.ndarray[Shape, np.dtype[np.float32]],
    silt: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Determine saturated water content [m3/m3].

    Based on:
    Tóth, B., Weynants, M., Nemes, A., Makó, A., Bilas, G., and Tóth, G.:
    New generation of hydraulic pedotransfer functions for Europe, Eur. J.
    Soil Sci., 66, 226-238. doi: 10.1111/ejss.121921211, 2015.

    Args:
        organic_carbon_percentage: soil organic carbon content [%].
        bulk_density: bulk density [g /cm3].
        clay: clay percentage [%].
        silt: fsilt percentage [%].
        is_top_soil: top soil flag.

    Returns:
        thetas: saturated water content [cm3/cm3].

    """
    return (
        np.float32(0.6819)
        - np.float32(0.06480) * (1 / (organic_carbon_percentage + 1))
        - np.float32(0.11900) * bulk_density**2
        - np.float32(0.02668) * is_top_soil
        + np.float32(0.001489) * clay
        + np.float32(0.0008031) * silt
        + np.float32(0.02321) * (1 / (organic_carbon_percentage + 1)) * bulk_density**2
        + np.float32(0.01908) * bulk_density**2 * is_top_soil
        - np.float32(0.0011090) * clay * is_top_soil
        - np.float32(0.00002315) * silt * clay
        - np.float32(0.0001197) * silt * bulk_density**2
        - np.float32(0.0001068) * clay * bulk_density**2
    )


def thetas_wosten(
    clay: np.ndarray[Shape, np.dtype[np.float32]],
    bulk_density: np.ndarray[Shape, np.dtype[np.float32]],
    silt: np.ndarray[Shape, np.dtype[np.float32]],
    organic_carbon_percentage: np.ndarray[Shape, np.dtype[np.float32]],
    is_topsoil: np.ndarray[Shape, np.dtype[np.bool_]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Calculates the saturated water content (theta_S) based on the provided equation.

    From: https://doi.org/10.1016/S0016-7061(98)00132-3

    Args:
        clay: Clay percentage (C).
        bulk_density: Bulk density (D).
        silt: Silt percentage (S).
        organic_carbon_percentage: Organic matter percentage (OM).
        is_topsoil: 1 for topsoil, 0 for subsoil.

    Returns:
        float: The calculated saturated water content (theta_S).
    """
    theta_s = (
        0.7919
        + 0.00169 * clay
        - 0.29619 * bulk_density
        - 0.000001491 * silt**2
        + 0.0000821 * organic_carbon_percentage**2
        + 0.02427 * (1 / clay)
        + 0.01113 * (1 / silt)
        + 0.01472 * np.log(silt)
        - 0.0000733 * organic_carbon_percentage * clay
        - 0.000619 * bulk_density * clay
        - 0.001183 * bulk_density * organic_carbon_percentage
        - 0.0001664 * is_topsoil * silt
    )

    return theta_s


def thetar_brakensiek(
    sand: np.ndarray[Shape, np.dtype[np.float32]],
    clay: np.ndarray[Shape, np.dtype[np.float32]],
    thetas: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Determine residual water content [m3/m3].

    Thetas is equal to porosity (Φ) in this case.

    Equation found in https://archive.org/details/watershedmanagem0000unse_d4j9/page/294/mode/1up (p. 294)

    Based on:
        Brakensiek, D.L., Rawls, W.J.,and Stephenson, G.R.: Modifying scs hydrologic
        soil groups and curve numbers for range land soils, ASAE Paper no. PNR-84-203,
        St. Joseph, Michigan, USA, 1984.

    Args:
        sand: sand percentage [%].
        clay: clay percentage [%].
        thetas: saturated water content [m3/m3].

    Returns:
        residual water content [m3/m3].
    """
    # Clip clay and sand values to avoid unrealistic results and in accordance with original paper
    clay, sand = clip_brakensiek(clay, sand)
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


def get_bubbling_pressure(
    clay: np.ndarray[Shape, np.dtype[np.float32]],
    sand: np.ndarray[Shape, np.dtype[np.float32]],
    thetas: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Determine bubbling pressure [cm].

    Thetas is equal to porosity (Φ) in this case.

    Based on:
    Rawls,W. J., and Brakensiek, D. L.: Estimation of SoilWater Retention and
    Hydraulic Properties, In H. J. Morel-Seytoux (Ed.),
    Unsaturated flow in hydrologic modelling - Theory and practice, NATO ASI Series 9,
    275-300, Dordrecht, The Netherlands: Kluwer Academic Publishing, 1989.

    Args:
        clay: clay percentage [%].
        sand: sand percentage [%].
        thetas: saturated water content [m3/m3].

    Returns:
        bubbling_pressure: bubbling pressure [cm].
    """
    bubbling_pressure = np.exp(
        5.3396738
        + 0.1845038 * clay
        - 2.48394546 * thetas
        - 0.00213853 * clay**2
        - 0.04356349 * sand * thetas
        - 0.61745089 * clay * thetas
        - 0.00001282 * sand**2 * clay
        + 0.00895359 * clay**2 * thetas
        - 0.00072472 * sand**2 * thetas
        + 0.0000054 * clay**2 * sand
        + 0.00143598 * sand**2 * thetas**2
        - 0.00855375 * clay**2 * thetas**2
        + 0.50028060 * thetas**2 * clay
    )
    return bubbling_pressure


def get_pore_size_index_brakensiek(
    sand: np.ndarray[Shape, np.dtype[np.float32]],
    thetas: np.ndarray[Shape, np.dtype[np.float32]],
    clay: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Determine Brooks-Corey pore size distribution index [-].

    Thetas is equal to porosity (Φ) in this case.

    Based on:

    Rawls,W. J., and Brakensiek, D. L.: Estimation of SoilWater Retention and
    Hydraulic Properties, In H. J. Morel-Seytoux (Ed.),
    Unsaturated flow in hydrologic modelling - Theory and practice, NATO ASI Series 9,
    275-300, Dordrecht, The Netherlands: Kluwer Academic Publishing, 1989.

    Args:
        sand: sand percentage [%].
        thetas: saturated water content [m3/m3].
        clay: clay percentage [%].

    Returns:
        pore size distribution index [-].

    """
    # Clip clay and sand values to avoid unrealistic results and in accordance with original paper
    clay, sand = clip_brakensiek(clay, sand)
    poresizeindex = np.exp(
        -0.7842831
        + 0.0177544 * sand
        - 1.062498 * thetas
        - 0.00005304 * (sand**2)
        - 0.00273493 * (clay**2)
        + 1.11134946 * (thetas**2)
        - 0.03088295 * sand * thetas
        + 0.00026587 * (sand**2) * (thetas**2)
        - 0.00610522 * (clay**2) * (thetas**2)
        - 0.00000235 * (sand**2) * clay
        + 0.00798746 * (clay**2) * thetas
        - 0.00674491 * (thetas**2) * clay
    )

    return poresizeindex


def get_pore_size_index_wosten(
    clay: np.ndarray[Shape, np.dtype[np.float32]],
    silt: np.ndarray[Shape, np.dtype[np.float32]],
    organic_carbon_percentage: np.ndarray[Shape, np.dtype[np.float32]],
    bulk_density: np.ndarray[Shape, np.dtype[np.float32]],
    is_top_soil: np.ndarray[Shape, np.dtype[np.bool_]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Determine Brooks-Corey pore size distribution index [-].

    See: https://doi.org/10.1016/S0016-7061(98)00132-3

    Args:
        clay: clay percentage [%].
        silt: silt percentage [%].
        organic_carbon_percentage: soil organic carbon content [%].
        bulk_density: bulk density [g /cm3].
        is_top_soil: top soil flag.

    Returns:
        pore size distribution index [-].
    """
    return np.exp(
        -25.23
        - 0.02195 * clay
        + 0.0074 * silt
        - 0.1940 * organic_carbon_percentage
        + 45.5 * bulk_density
        - 7.24 * bulk_density**2
        + 0.0003658 * clay**2
        + 0.002855 * organic_carbon_percentage**2
        - 12.81 * bulk_density**-1
        - 0.1524 * silt**-1
        - 0.01958 * organic_carbon_percentage**-1
        - 0.2876 * np.log(silt)
        - 0.0709 * np.log(organic_carbon_percentage)
        - 44.6 * np.log(bulk_density)
        - 0.02264 * bulk_density * clay
        + 0.0896 * bulk_density * organic_carbon_percentage
        + 0.00718 * is_top_soil * clay
    )


def kv_brakensiek(
    thetas: np.ndarray[Shape, np.dtype[np.float32]],
    clay: np.ndarray[Shape, np.dtype[np.float32]],
    sand: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Determine saturated hydraulic conductivity kv [m/s].

    Based on:
      Brakensiek, D.L., Rawls, W.J.,and Stephenson, G.R.: Modifying scs hydrologic
      soil groups and curve numbers for range land soils, ASAE Paper no. PNR-84-203,
      St. Joseph, Michigan, USA, 1984.

    Args:
        thetas: saturated water content [m3/m3].
        clay: clay percentage [%].
        sand: sand percentage [%].

    Returns:
        saturated hydraulic conductivity [m/s].
    """
    # Clip clay and sand values to avoid unrealistic results and in accordance with original paper
    clay, sand = clip_brakensiek(clay, sand)
    kv = np.exp(
        19.52348 * thetas
        - 8.96847
        - 0.028212 * clay
        + 0.00018107 * sand**2
        - 0.0094125 * clay**2
        - 8.395215 * thetas**2
        + 0.077718 * sand * thetas
        - 0.00298 * sand**2 * thetas**2
        - 0.019492 * clay**2 * thetas**2
        + 0.0000173 * sand**2 * clay
        + 0.02733 * clay**2 * thetas
        + 0.001434 * sand**2 * thetas
        - 0.0000035 * clay**2 * sand
    )  # cm / hr
    kv = kv / 100 / 3600  # convert to m/s
    return kv


def kv_wosten(
    silt: np.ndarray[Shape, np.dtype[np.float32]],
    clay: np.ndarray[Shape, np.dtype[np.float32]],
    bulk_density: np.ndarray[Shape, np.dtype[np.float32]],
    organic_carbon_percentage: np.ndarray[Shape, np.dtype[np.float32]],
    is_topsoil: np.ndarray[Shape, np.dtype[np.bool_]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Calculates the saturated value based on the provided equation.

    From: https://doi.org/10.1016/S0016-7061(98)00132-3

    Args:
        silt: Silt percentage (S).
        is_topsoil: 1 for topsoil, 0 for subsoil.
        bulk_density: Bulk density (D).
        clay: Clay percentage (C).
        organic_carbon_percentage: Organic matter percentage (OM).

    Returns:
        float: The calculated Ks* value [m/s].
    """
    ks = np.exp(
        7.755
        + 0.0352 * silt
        + np.float32(0.93) * is_topsoil
        - 0.967 * bulk_density**2
        - 0.000484 * clay**2
        - 0.000322 * silt**2
        + 0.001 * (1 / silt)
        - 0.0748 * (1 / organic_carbon_percentage)
        - 0.643 * np.log(silt)
        - 0.01398 * bulk_density * clay
        - 0.1673 * bulk_density * organic_carbon_percentage
        + 0.02986 * np.float32(is_topsoil) * clay
        - 0.03305 * np.float32(is_topsoil) * silt
    ) / (100 * 86400)  # convert to m/s

    return ks


def kv_cosby(
    sand: np.ndarray[Shape, np.dtype[np.float32]],
    clay: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Determine saturated hydraulic conductivity kv [m/s].

    based on:
      Cosby, B.J., Hornberger, G.M., Clapp, R.B., Ginn, T.R., 1984.
      A statistical exploration of the relationship of soil moisture characteristics to
      the physical properties of soils. Water Resour. Res. 20(6) 682-690.
      https://doi.org/10.1029/WR020i006p00682

    Args:
        sand: sand percentage [%].
        clay: clay percentage [%].

    Returns:
        kv: saturated hydraulic conductivity [m/s].

    """
    kv = 60.96 * 10.0 ** (-0.6 + 0.0126 * sand - 0.0064 * clay) * 10.0  # mm / day
    kv = kv / (1000 * 86400)  # convert to m/s

    return kv
