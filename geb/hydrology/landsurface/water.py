"""Soil water flow functions."""

import numpy as np
from numba import njit

from geb.geb_types import ArrayFloat32, Shape, TwoDArrayFloat64
from geb.hydrology import TwoDArrayFloat32

from ..landcovers import OPEN_WATER, PADDY_IRRIGATED, SEALED
from .constants import N_SOIL_LAYERS
from .energy import (
    apply_rain_heat_advection,
    get_temperature_and_frozen_fraction_from_enthalpy_scalar,
)


@njit(cache=True, inline="always")
def add_water_to_topwater_and_evaporate_open_water(
    natural_available_water_infiltration_m: np.float32,
    actual_irrigation_consumption_m: np.float32,
    land_use_type: np.int32,
    potential_direct_evaporation_m: np.float32,
    topwater_m: np.float32,
) -> tuple[np.float32, np.float32]:
    """Add available water to topwater and calculate open water evaporation.

    Args:
        natural_available_water_infiltration_m: Natural available water (m).
        actual_irrigation_consumption_m: Actual irrigation consumption (m).
        land_use_type: Land use type (-).
        potential_direct_evaporation_m: Potential direct evaporation (soil/water) (m).
        topwater_m: Topwater storage before update (m).

    Returns:
        A tuple containing:
            - Updated topwater storage (m)
            - Actual open water evaporation (m)
    """
    # Add water to topwater
    topwater_m += (
        natural_available_water_infiltration_m + actual_irrigation_consumption_m
    )

    # Calculate open water evaporation for water-based or sealed land use types.
    # For natural areas, direct (bare soil) evaporation is handled later by
    # calculate_bare_soil_evaporation.
    if land_use_type in (OPEN_WATER, PADDY_IRRIGATED, SEALED):
        open_water_evaporation_m = min(
            max(np.float32(0.0), topwater_m),
            potential_direct_evaporation_m,
        )
    else:
        open_water_evaporation_m = np.float32(0.0)

    # Subtract evaporation from topwater
    topwater_m -= open_water_evaporation_m

    return topwater_m, open_water_evaporation_m


@njit(cache=True, inline="always", fastmath=True)
def calculate_spatial_infiltration_excess(
    infiltration_capacity_mean: np.float32,
    available_water: np.float32,
    shape_parameter_beta: np.float32,
) -> tuple[np.float32, np.float32]:
    """Calculate effective infiltration and runoff using spatial variability of infiltration capacity.

    This function implements a Hortonian runoff generation mechanism assuming that the
    Green-Ampt infiltration capacity within the cell follows a Reflected Power distribution.

    Args:
        infiltration_capacity_mean: The mean infiltration capacity (Green-Ampt capacity) in the cell (m).
        available_water: The amount of water available for infiltration (e.g. precipitation) (m).
        shape_parameter_beta: The shape parameter `b` of the Reflected Power distribution.

    Returns:
        A tuple containing:
            - infiltration: Effective infiltration amount (m).
            - runoff: Runoff amount due to infiltration excess (m).
    """
    if available_water <= np.float32(0.0):
        return np.float32(0.0), np.float32(0.0)

    if infiltration_capacity_mean <= np.float32(0.0):
        return np.float32(0.0), available_water

    # Higher beta = more early runoff, slower approach to mean
    # Limit as beta -> 0 is the Exponential Distribution: I = f_mean * (1 - exp(-P/f_mean))
    # For beta > 0, we use the Generalized Pareto / Lomax integral
    if shape_parameter_beta < np.float32(1e-4):
        infiltration: np.float32 = infiltration_capacity_mean * (
            np.float32(1.0) - np.exp(-available_water / infiltration_capacity_mean)
        )
    else:
        exponent = np.float32(-1.0) / shape_parameter_beta
        base = np.float32(1.0) + (
            shape_parameter_beta * available_water / infiltration_capacity_mean
        )
        infiltration: np.float32 = infiltration_capacity_mean * (
            np.float32(1.0) - base**exponent
        )

    # Clamp infiltration to available water to prevent precision errors
    infiltration: np.float32 = min(infiltration, available_water)

    runoff: np.float32 = available_water - infiltration
    # Prevent negative runoff due to float precision
    runoff: np.float32 = max(runoff, np.float32(0.0))

    return infiltration, runoff


@njit(cache=True, inline="always", fastmath=True)
def calculate_variable_source_area_fraction(
    effective_soil_saturation: np.float32,
    shape_parameter_beta: np.float32,
) -> np.float32:
    """Calculate the saturated variable source area fraction for Dunne runoff.

    Computes the fraction of the grid cell / catchment area (A_s) that is saturated
    to the surface based on the near-surface soil moisture state and topographic shape
    parameter, following the classical Variable Infiltration Capacity (VIC) and
    Probability Distributed Model (PDM) formulations (Zhao 1992; Liang et al., 1994;
    Moore 1985).

    Notes:
        Under the Pareto storage capacity distribution:
            A_s = 1 - (1 - S_soil)^b
        where S_soil is the effective saturation of the upper soil column and b is the
        calibrated topographic shape parameter.

    Args:
        effective_soil_saturation: Relative saturation of the upper soil column (-),
            bounded between 0.0 (dry) and 1.0 (fully saturated).
        shape_parameter_beta: Topographic roughness / infiltration shape parameter (-).

    Returns:
        Fraction of the cell area that is saturated and generating Dunne runoff (-).
    """
    if shape_parameter_beta <= np.float32(
        0.0
    ) or effective_soil_saturation <= np.float32(0.0):
        return np.float32(0.0)

    s_soil: np.float32 = min(effective_soil_saturation, np.float32(1.0))
    # Calibration scaling factor (0.3) applied to terrain shape parameter
    b: np.float32 = max(np.float32(0.01), shape_parameter_beta * np.float32(0.3))
    unsaturated_fraction: np.float32 = np.power(np.float32(1.0) - s_soil, b)
    saturated_area_fraction: np.float32 = np.float32(1.0) - unsaturated_fraction

    return min(max(saturated_area_fraction, np.float32(0.0)), np.float32(1.0))


@njit(cache=True, inline="always", fastmath=True)
def calculate_green_ampt_time_from_infiltration(
    cumulative_infiltration: np.float32,
    saturated_hydraulic_conductivity_m_per_s: np.float32,
    wetting_front_suction_head_m: np.float32,
    moisture_deficit: np.float32,
) -> np.float32:
    """Calculate the time required to infiltrate a given cumulative amount.

    This implements the exact analytical inversion of the Green-Ampt equation.

    The standard Green-Ampt rate equation is (Heber Green and Ampt, 1911):
        f = K (1 + (psi * dtheta) / F)

    Integrating this yields the cumulative infiltration equation (Chow et al., 1988; Eq 4.3.6):
        K * t = F - (psi * dtheta) * ln(1 + F / (psi * dtheta))

    Thus, given cumulative infiltration F, we can solve for time t:
        t = (F - (psi * dtheta) * ln(1 + F / (psi * dtheta))) / K

    References:
        Heber Green, W. and Ampt, G.A. (1911) ‘Studies on Soil Phyics.’, The Journal of Agricultural Science, 4(1), pp. 1–24. doi:10.1017/S0021859600001441.
        Chow, V. T., Maidment, D. R., & Mays, L. W. (1988). Applied Hydrology. McGraw-Hill.

    Args:
        cumulative_infiltration: Cumulative infiltration amount (m).
        saturated_hydraulic_conductivity_m_per_s: Saturated hydraulic conductivity (m/s).
        wetting_front_suction_head_m: Wetting front suction head (m).
        moisture_deficit: Moisture deficit [-].

    Returns:
        Time t corresponding to the infiltration amount.
    """
    if cumulative_infiltration <= np.float32(0.0):
        return np.float32(0.0)

    if saturated_hydraulic_conductivity_m_per_s <= np.float32(0.0):
        return np.float32(0.0)

    wetting_front_suction_potential: np.float32 = (
        wetting_front_suction_head_m * moisture_deficit
    )

    # Darcy limit: if there is no capillary suction effect (sf -> 0),
    # then cumulative infiltration is I = K_s t.
    if wetting_front_suction_potential <= np.float32(0.0):
        return cumulative_infiltration / saturated_hydraulic_conductivity_m_per_s

    # np.log1p(x) computes log(1 + x) accurately for small x
    term_log = np.log1p(cumulative_infiltration / wetting_front_suction_potential)
    t = (
        cumulative_infiltration - wetting_front_suction_potential * term_log
    ) / saturated_hydraulic_conductivity_m_per_s
    return max(t, np.float32(0.0))


@njit(cache=True, inline="always", fastmath=True)
def calculate_green_ampt_potential_cumulative_infiltration(
    seconds_since_start: np.float32,
    saturated_hydraulic_conductivity_m_per_s: np.float32,
    wetting_front_suction_head_m: np.float32,
    moisture_deficit: np.float32,
    adjust_for_coarse_soils: bool = False,
) -> np.float32:
    """Calculate cumulative infiltration using the Sadeghi et al. (2024) explicit Green-Ampt formula.

    The reported maximum error of this formula is 0.3 percent across a wide range of conditions.

    Based on:
        Sadeghi, S. H., Loescher, H. W., Jacoby, P. W., & Sullivan, P. L. (2024).
        A simple, accurate, and explicit form of the Green–Ampt model to estimate infiltration,
        sorptivity, and hydraulic conductivity. Vadose Zone Journal,  23, e20341

    Args:
        seconds_since_start: Seconds since the start of the infiltration event.
        saturated_hydraulic_conductivity_m_per_s: Saturated hydraulic conductivity (m/s).
        wetting_front_suction_head_m: Wetting front suction head ψ (m).
        moisture_deficit: Moisture deficit [-].
        adjust_for_coarse_soils: Whether to apply adjustment for coarse soils. For coarse soils,
            and very long times, the Sageghi et al. formula can be slightly more off than the
            0.3 percent error. Setting this to True applies an empirical adjustment to improve accuracy
            in those situations.

    Returns:
        Cumulative infiltration amount in meters.
    """
    if seconds_since_start == np.float32(0.0):
        return np.float32(0.0)

    # Darcy limit: if suction or the moisture deficit is zero, there is no
    # capillarity-driven enhancement and Green-Ampt reduces to I = K_s t.
    if wetting_front_suction_head_m <= np.float32(
        0.0
    ) or moisture_deficit <= np.float32(0.0):
        return saturated_hydraulic_conductivity_m_per_s * seconds_since_start

    # Sorptivity can be calculated as per Philip (1969):
    # S^2 = 2 K_s * psi * dtheta
    # Reference: Philip, J.R., 1969. Theory of infiltration. In Advances in hydroscience (Vol. 5, pp. 215-296). Elsevier.
    # Since the Sadeghi formula uses S^2 directly, we do not need to take the square root.
    sorptivity_squared: np.float32 = (
        np.float32(2.0)
        * saturated_hydraulic_conductivity_m_per_s
        * wetting_front_suction_head_m
        * moisture_deficit
    )

    # Apply Sadeghi et al. (2024) explicit formula
    hydraulic_conductivity_times_time: np.float32 = (
        saturated_hydraulic_conductivity_m_per_s * seconds_since_start
    )

    # sorptivity_time_ratio corresponds to S^2 / (Ks^2 * t)
    sorptivity_time_ratio: np.float32 = sorptivity_squared / (
        saturated_hydraulic_conductivity_m_per_s * hydraulic_conductivity_times_time
    )
    cumulative_infiltration: np.float32 = hydraulic_conductivity_times_time * (
        np.float32(0.70635)
        + np.float32(0.32415)
        * np.sqrt(np.float32(1.0) + np.float32(9.43456) * sorptivity_time_ratio)
    )

    # Apply adjustment for coarse soils if necessary
    if adjust_for_coarse_soils and (
        hydraulic_conductivity_times_time / cumulative_infiltration
    ) > np.float32(0.904):
        cumulative_infiltration: np.float32 = np.float32(
            0.9796
        ) * cumulative_infiltration + np.float32(0.335) * (
            sorptivity_squared / saturated_hydraulic_conductivity_m_per_s
        )

    return np.float32(cumulative_infiltration)


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


@njit(cache=True, inline="always", fastmath=True)
def get_soil_water_potential_van_genuchten(
    w: np.float32,
    wres: np.float32,
    ws: np.float32,
    lambda_pore_size_distribution: np.float32,
    bubbling_pressure_m_positive: np.float32,
) -> np.float32:
    """Calculate the soil water potential for a single soil layer using Van Genuchten.

    Args:
        w: Soil water content in the layer in meters.
        wres: Residual soil water content in the layer in meters.
        ws: Saturated soil water content in the layer in meters.
        lambda_pore_size_distribution: Van Genuchten parameter lambda for the layer.
        bubbling_pressure_m_positive: Bubbling pressure for the layer in m.

    Returns:
        psi: Soil water potential in the layer in meters (negative value for suction).
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

    alpha = np.float32(1) / bubbling_pressure_m_positive

    # Compute capillary pressure head (phi)
    phi_power_term = np.power(effective_saturation, (-np.float32(1) / m))
    phi = (
        np.power(phi_power_term - np.float32(1), (np.float32(1) / n)) / alpha
    )  # Positive value
    phi = np.minimum(phi, max_suction_meters)  # Limit to maximum suction

    # Soil water potential (negative value for suction)
    return -phi


@njit(cache=True, inline="always")
def calculate_effective_wetting_front_suction(
    bubbling_pressure_m_positive: np.float32,
    lambda_pore_size_distribution: np.float32,
    effective_saturation: np.float32 = np.float32(0.0),
) -> np.float32:
    """Calculate effective wetting front suction head for Green-Ampt infiltration.

    Computes the effective capillary drive across the wetting front (psi_f) based on
    Brooks-Corey parameters and initial moisture content ahead of the front, using the
    analytical solution of the hydraulic conductivity integral from Brakensiek (1977)
    and Rawls, Brakensiek & Miller (1983).

    Notes:
        The capillary drive across the wetting front is defined by:
            H_c = integral_0^{psi_i} K_r(psi) dpsi
        Under the assumption of rectangular piston flow, the effective wetting front
        suction head psi_f is H_c / 2 (Rawls et al., 1983; Chow et al., 1988):
            psi_f = (psi_b / 2) * (2 + 3*lambda - S_e^((1 + 3*lambda) / lambda)) / (1 + 3*lambda)
        where psi_b is the bubbling pressure, lambda is the pore size distribution index,
        and S_e is the initial effective saturation ahead of the wetting front.

    Args:
        bubbling_pressure_m_positive: Soil bubbling pressure / air entry potential (meters).
            Must be positive.
        lambda_pore_size_distribution: Brooks-Corey pore size distribution index (-).
        effective_saturation: Effective saturation ahead of the wetting front (-),
            bounded between 0.0 (dry) and 1.0 (saturated).

    Returns:
        Effective wetting front suction head (meters).
    """
    s_e: np.float32 = min(max(effective_saturation, np.float32(0.0)), np.float32(1.0))
    lambda_safe: np.float32 = max(lambda_pore_size_distribution, np.float32(1e-4))
    bubbling_safe: np.float32 = max(bubbling_pressure_m_positive, np.float32(1e-4))

    eta: np.float32 = np.float32(2.0) + np.float32(3.0) * lambda_safe
    denominator: np.float32 = np.float32(1.0) + np.float32(3.0) * lambda_safe
    exponent: np.float32 = denominator / lambda_safe

    if s_e > np.float32(0.0):
        sat_term: np.float32 = s_e**exponent
    else:
        sat_term = np.float32(0.0)

    psi_f: np.float32 = (bubbling_safe / np.float32(2.0)) * (
        (eta - sat_term) / denominator
    )

    return max(psi_f, np.float32(1e-4))


@njit(cache=True, inline="always")
def get_green_ampt_params(
    wetting_front_depth_m: np.float32,
    soil_layer_height_m: ArrayFloat32,
    w: ArrayFloat32,
    ws: ArrayFloat32,
    wres: ArrayFloat32,
    bubbling_pressure_m_positive: ArrayFloat32,
    lambda_pore_size_distribution: ArrayFloat32,
) -> tuple[int, np.float32, np.float32]:
    """Helper to determine active layer and Green-Ampt parameters at the wetting front depth.

    This function identifies which soil layer currently contains the wetting front and calculates
    the effective hydraulic parameters (suction head, moisture deficit) needed for the Green-Ampt
    infiltration equation. It accounts for soil layering by estimating the initial moisture content
    ahead of the front based on mass balance and "piston flow" assumptions.

    Args:
        wetting_front_depth_m: Current depth of the wetting front (meters).
        soil_layer_height_m: Thickness of each soil layer (meters).
        w: Current total water column in each soil layer (meters).
        ws: Saturated water column capacity of each soil layer (meters).
        wres: Residual water column of each soil layer (meters).
        bubbling_pressure_m_positive: Bubbling pressure parameter for each layer (meters).
        lambda_pore_size_distribution: Pore size distribution index (lambda) for each layer (-).

    Returns:
        A tuple containing:
            - idx: Index of the soil layer containing the wetting front.
            - psi: Effective wetting front suction head (meters).
            - delta_theta: Moisture deficit at the wetting front (-).

    Notes:
        - Assumes piston flow: Soil behind the wetting front is fully saturated.
        - Calculates moisture ahead of the front by subtracting the saturated water volume behind the front
          from the total layer water volume.
        - Uses effective wetting front suction head rather than matric tension to properly represent
          capillary drive under Green-Ampt theory (Brakensiek, 1977; Rawls et al., 1983).
    """
    current_depth: np.float32 = np.float32(0.0)
    n_layers: int = len(soil_layer_height_m)
    idx: int = 0
    depth_in_layer: np.float32 = np.float32(0.0)

    # Find which layer the wetting front is currently in
    for i in range(n_layers):
        h: np.float32 = soil_layer_height_m[i]
        # Use epsilon to handle boundaries
        if wetting_front_depth_m < current_depth + h - np.float32(1e-4):
            idx = i
            depth_in_layer = max(np.float32(0.0), wetting_front_depth_m - current_depth)
            break
        current_depth += h
    else:
        # Front is below the bottom layer
        idx = n_layers - 1
        depth_in_layer = soil_layer_height_m[idx]

    layer_h: np.float32 = soil_layer_height_m[idx]

    # Reconstruct initial moisture content ahead of the front.
    # w[idx] is the total water column in the layer (meters).
    # Since we assume piston flow, the part of the layer behind the front (depth_in_layer)
    # is saturated using the Green-Ampt assumption.
    theta_sat: np.float32 = ws[idx] / layer_h
    remaining_height: np.float32 = layer_h - depth_in_layer

    if remaining_height > np.float32(1e-4):
        # Mass balance: Total Water = Water_Behind + Water_Ahead
        water_behind: np.float32 = depth_in_layer * theta_sat
        water_ahead: np.float32 = max(np.float32(0.0), w[idx] - water_behind)
        theta_initial: np.float32 = water_ahead / remaining_height

        # Clamp to physical limits
        # use a small epsilon as floor
        theta_floor: np.float32 = np.float32(1e-9)

        theta_initial = max(theta_initial, theta_floor)
        theta_initial = min(theta_initial, theta_sat - np.float32(1e-9))
    else:
        # Layer fully invaded; assume near saturation (small deficit)
        theta_initial = theta_sat - np.float32(1e-3)

    delta_theta: np.float32 = max(theta_sat - theta_initial, np.float32(1e-4))

    # Calculate effective wetting front suction based on Brooks-Corey parameters and initial moisture
    theta_res: np.float32 = wres[idx] / layer_h
    delta_theta_max: np.float32 = max(theta_sat - theta_res, np.float32(1e-6))
    effective_saturation: np.float32 = min(
        max((theta_initial - theta_res) / delta_theta_max, np.float32(0.0)),
        np.float32(1.0),
    )

    psi: np.float32 = calculate_effective_wetting_front_suction(
        bubbling_pressure_m_positive=bubbling_pressure_m_positive[idx],
        lambda_pore_size_distribution=lambda_pore_size_distribution[idx],
        effective_saturation=effective_saturation,
    )

    return idx, psi, delta_theta


@njit(inline="always")
def splitmix64(x: np.uint64) -> np.uint64:
    """Deterministic stateless hash.

    Args:
        x: Input integer to hash.

    Returns:
        A hashed integer output.
    """
    x: np.uint64 = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    x: np.uint64 = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> 31)


# Precompute lognormal weights for rainfall distribution over substeps.
# Each row of LUT corresponds to a different random distribution of rainfall across
# the 6 substeps, simulating temporal variability in infiltration capacity.
# Generate continuous log-normal noise

# Table size MUST be a power of two
TABLE_SIZE: int = 1024
assert (TABLE_SIZE & (TABLE_SIZE - 1)) == 0, "TABLE_SIZE must be a power of two"
MASK: np.uint64 = np.uint64(TABLE_SIZE - 1)

# Default sigma presets by precipitation forcing source
FORCING_SOURCE_SIGMA_DEFAULTS: dict[str, float] = {
    "ERA5-Land": 1.2,
    "MSWEP": 0.8,
    "ECMWF": 1.2,
}


def generate_rainfall_lookup_table(
    sigma: float,
    table_size: int = TABLE_SIZE,
    seed: int = 42,
) -> TwoDArrayFloat32:
    """Generate precomputed lognormal weights for rainfall distribution across substeps.

    Args:
        sigma: Lognormal standard deviation shape parameter.
        table_size: Number of table rows (must be a power of two).
        seed: Random seed for reproducible table generation.

    Returns:
        2D array of shape (table_size, 6) with normalized row weights summing to 1.0.
    """
    raw: TwoDArrayFloat64 = np.random.default_rng(seed).lognormal(
        mean=0.0,
        sigma=sigma,
        size=(table_size, 6),
    )
    table: TwoDArrayFloat32 = (raw / raw.sum(axis=1)[:, np.newaxis]).astype(np.float32)

    return table


@njit(cache=True, inline="always", fastmath=True)
def calculate_depression_spillover_runoff(
    topwater_m: np.float32,
    depression_storage_capacity_m: np.float32 = np.float32(0.0025),
    runoff_shape_parameter: np.float32 = np.float32(1.0),
) -> tuple[np.float32, np.float32]:
    """Calculate surface runoff and retained topwater from depression storage spillover.

    Surface hollows and puddles retain ponded surface water up to the depression
    storage capacity. As topwater accumulates, water progressively connects and spills
    over into direct surface runoff.

    Args:
        topwater_m: Total surface water before runoff partitioning (meters).
        depression_storage_capacity_m: Maximum puddle/depression storage capacity (meters).
        runoff_shape_parameter: Exponent governing how quickly puddles overflow [-].
            Higher values retain more water before overflowing; lower values produce earlier runoff.

    Returns:
        A tuple containing:
            - retained_topwater_m: Water retained in surface depressions (meters).
            - direct_runoff_m: Direct surface runoff leaving the cell (meters).
    """
    if topwater_m <= np.float32(0.0):
        return np.float32(0.0), np.float32(0.0)

    if depression_storage_capacity_m <= np.float32(0.0):
        return np.float32(0.0), topwater_m

    inv_shape_plus_one: np.float32 = np.float32(1.0) / (
        runoff_shape_parameter + np.float32(1.0)
    )

    if topwater_m <= depression_storage_capacity_m:
        relative_storage: np.float32 = topwater_m / depression_storage_capacity_m
        direct_runoff_m: np.float32 = (
            topwater_m * inv_shape_plus_one * (relative_storage**runoff_shape_parameter)
        )
        retained_topwater_m: np.float32 = topwater_m - direct_runoff_m
    else:
        max_retention: np.float32 = depression_storage_capacity_m * (
            np.float32(1.0) - inv_shape_plus_one
        )
        retained_topwater_m = max_retention
        direct_runoff_m = topwater_m - retained_topwater_m

    return retained_topwater_m, direct_runoff_m


@njit(cache=True, inline="always")
def infiltration(
    seed: np.int64,
    ws: ArrayFloat32,
    wres: ArrayFloat32,
    saturated_hydraulic_conductivity_m_per_s: ArrayFloat32,
    groundwater_toplayer_conductivity_m_per_s: np.float32,
    land_use_type: np.int32,
    w: ArrayFloat32,
    topwater_m: np.float32,
    capillary_rise_from_groundwater_m: np.float32,
    wetting_front_depth_m: np.float32,
    wetting_front_suction_head_m: np.float32,
    wetting_front_moisture_deficit: np.float32,
    green_ampt_active_layer_idx: int | np.integer,
    variable_runoff_shape_beta: np.float32,
    bubbling_pressure_m_positive: ArrayFloat32,
    soil_layer_height_m: ArrayFloat32,
    lambda_pore_size_distribution: ArrayFloat32,
    soil_enthalpy_top_layer_J_per_m2: np.float32,
    solid_heat_capacity_top_layer_J_per_m2_K: np.float32,
    rain_temperature_C: np.float32,
    new_surface_water_input_m: np.float32,
    rainfall_lookup_table: TwoDArrayFloat32,
    distribute_rainfall_lognormally: bool = True,
    slope_m_per_m: np.float32 = np.float32(0.0),
) -> tuple[
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    int | np.integer,
    np.float32,
]:
    """Simulates vertical transport of water in the soil for a single cell.

    Uses an explicit Green-Ampt approximation (Salvucci, 1994) combined with the
    PDM variable infiltration capacity curve and depression storage spillover runoff.

    The function uses `wetting_front_suction_head_m` which is a state variable
    tracking the matric suction at the sharp wetting front. This should be
    initialized at the beginning of an infiltration event based on the soil
    moisture deficit.

    Args:
        seed: A seed for random number generation.
        ws: Saturated soil water content in each layer for the cell in meters, shape (N_SOIL_LAYERS,).
        wres: Residual soil water content in each layer for the cell in meters, shape (N_SOIL_LAYERS,).
        saturated_hydraulic_conductivity_m_per_s: Saturated hydraulic conductivity in each layer for the cell in m/s, shape (N_SOIL_LAYERS,).
        groundwater_toplayer_conductivity_m_per_s: Groundwater top layer conductivity limiting recharge (m/s).
        land_use_type: Land use type for the cell.
        w: Soil water content in each layer for the cell in meters, shape (N_SOIL_LAYERS,), modified in place.
        topwater_m: Topwater for the cell in meters, modified in place.
        capillary_rise_from_groundwater_m: Capillary rise from groundwater for the cell (m/timestep). If >0, percolation to groundwater is suppressed.
        wetting_front_depth_m: Depth of the wetting front in meters.
        wetting_front_suction_head_m: Suction head at the wetting front in meters.
        wetting_front_moisture_deficit: Moisture deficit at the wetting front [-].
        green_ampt_active_layer_idx: The index of the active soil layer for Green-Ampt.
        variable_runoff_shape_beta: Shape parameter `b` for the PDM distribution.
        bubbling_pressure_m_positive: Bubbling pressure for each soil layer [m], shape (N_SOIL_LAYERS,).
        soil_layer_height_m: Height of each soil layer [m], shape (N_SOIL_LAYERS,).
        lambda_pore_size_distribution: Van Genuchten parameter lambda for each soil layer, shape (N_SOIL_LAYERS,).
        soil_enthalpy_top_layer_J_per_m2: Top-layer soil enthalpy relative to 0°C liquid water (J/m2).
        solid_heat_capacity_top_layer_J_per_m2_K: Areal heat capacity of the top-layer solid soil fraction (J/m2/K).
        rain_temperature_C: Temperature of liquid water entering the soil surface (C).
        new_surface_water_input_m: Newly arriving liquid water reaching the surface
            during this timestep from rain, melt, or irrigation (m).
        rainfall_lookup_table: Precomputed lognormal weights lookup table of shape (table_size, 6).
        distribute_rainfall_lognormally: Whether to distribute rainfall across substeps using a log-normal distribution to simulate temporal variability. If False, rainfall is distributed evenly.
        slope_m_per_m: Hillslope slope for modulating depression storage capacity (m/m).

    Returns:
        A tuple containing:
            - topwater_m: Updated topwater in meters.
            - direct_runoff: Direct runoff from the cell in meters.
            - groundwater_recharge: Recharge to groundwater from the soil column (m/timestep).
            - infiltration: Infiltration into the soil for the cell in meters.
            - wetting_front_depth_m: Updated wetting front depth in meters.
            - wetting_front_suction_head_m: Updated wetting front suction head in meters.
            - wetting_front_moisture_deficit: Updated wetting front moisture deficit [-].
            - green_ampt_active_layer_idx: Updated active soil layer index.
            - soil_enthalpy_top_layer_J_per_m2: Updated top-layer soil enthalpy (J/m2).
    """
    no_infiltration_land_use: bool = land_use_type == OPEN_WATER
    no_topwater_available: bool = bool(topwater_m <= np.float32(1e-6))

    # Early return avoids expensive Green-Ampt substeps when infiltration is impossible:
    # Open water does not allow infiltration, and if there is no topwater, there is no water to infiltrate.
    if no_infiltration_land_use or no_topwater_available:
        direct_runoff: np.float32 = topwater_m
        return (
            np.float32(0.0),  # topwater_m
            direct_runoff,  # direct_runoff
            np.float32(0.0),  # groundwater_recharge
            np.float32(0.0),  # infiltration
            np.float32(0.0),  # wetting_front_depth_m
            np.float32(0.0),  # wetting_front_suction_head_m
            np.float32(0.0),  # wetting_front_moisture_deficit
            -1,
            soil_enthalpy_top_layer_J_per_m2,
        )

    # Initialize accumulators for the timestep
    total_infiltration_amount: np.float32 = np.float32(0.0)
    groundwater_recharge_m: np.float32 = np.float32(0.0)
    impervious_direct_runoff_m: np.float32 = np.float32(0.0)
    total_direct_runoff: np.float32 = np.float32(0.0)

    # Clamp newly arriving water between 0 and total surface water (e.g. after open-water evaporation).
    # Any excess of topwater_m represents pre-existing depression storage from previous timesteps.
    new_surface_water_input_m = min(
        topwater_m, max(np.float32(0.0), new_surface_water_input_m)
    )
    current_topwater_m: np.float32 = topwater_m - new_surface_water_input_m

    n_substeps: int = 6
    substep_time_s: np.float32 = np.float32(3600.0) / np.float32(n_substeps)

    # If starting with a new wetting front, calculate initial parameters
    if wetting_front_depth_m == np.float32(0.0):
        (
            green_ampt_active_layer_idx,
            wetting_front_suction_head_m,
            wetting_front_moisture_deficit,
        ) = get_green_ampt_params(
            wetting_front_depth_m,
            soil_layer_height_m,
            w,
            ws,
            wres,
            bubbling_pressure_m_positive,
            lambda_pore_size_distribution,
        )

    # Calculate depth limit for current layer
    current_layer_depth_limit = np.float32(0.0)

    # Ensure active_layer_idx is within bounds
    green_ampt_active_layer_idx = max(
        0, min(green_ampt_active_layer_idx, len(soil_layer_height_m) - 1)
    )

    for i in range(green_ampt_active_layer_idx + 1):
        current_layer_depth_limit += soil_layer_height_m[i]

    # Steeper slopes hold less puddle storage before spilling over.
    # Sealed surfaces have lower storage (0.4 - 1.5 mm) than natural ground (0.8 - 2.5 mm).
    if land_use_type == PADDY_IRRIGATED:
        ponding_allowance: np.float32 = np.float32(0.05)
        depression_storage_m: np.float32 = ponding_allowance
        runoff_shape_parameter: np.float32 = np.float32(1.0)
    elif land_use_type == SEALED:
        ponding_allowance = np.float32(0.0)
        depression_storage_m = max(
            np.float32(0.0004),
            np.float32(0.0015) / (np.float32(1.0) + np.float32(10.0) * slope_m_per_m),
        )
        runoff_shape_parameter = max(
            np.float32(0.4),
            np.float32(1.0) / (np.float32(1.0) + variable_runoff_shape_beta),
        )
    else:
        ponding_allowance = np.float32(0.0)
        depression_storage_m = max(
            np.float32(0.0008),
            np.float32(0.0025) / (np.float32(1.0) + np.float32(10.0) * slope_m_per_m),
        )
        runoff_shape_parameter = max(
            np.float32(0.4),
            np.float32(1.0) / (np.float32(1.0) + variable_runoff_shape_beta),
        )

    # Calculate effective saturation of the upper root-zone soil profile (top 3 layers, ~30 cm)
    # for variable source area (Dunne saturation-excess) runoff generation.
    upper_layers_to_consider: int = min(3, len(w))
    total_water_upper: np.float32 = np.float32(0.0)
    total_res_upper: np.float32 = np.float32(0.0)
    total_sat_upper: np.float32 = np.float32(0.0)
    for i in range(upper_layers_to_consider):
        total_water_upper += w[i]
        total_res_upper += wres[i]
        total_sat_upper += ws[i]

    delta_capacity_upper: np.float32 = max(
        total_sat_upper - total_res_upper, np.float32(1e-6)
    )
    s_soil_effective: np.float32 = min(
        max(
            (total_water_upper - total_res_upper) / delta_capacity_upper,
            np.float32(0.0),
        ),
        np.float32(1.0),
    )

    if land_use_type in (SEALED, PADDY_IRRIGATED, OPEN_WATER):
        saturated_area_fraction: np.float32 = np.float32(0.0)
    else:
        saturated_area_fraction = calculate_variable_source_area_fraction(
            effective_soil_saturation=s_soil_effective,
            shape_parameter_beta=variable_runoff_shape_beta,
        )
    unsaturated_area_fraction: np.float32 = np.float32(1.0) - saturated_area_fraction

    # rainfall is given per 1 hour, but in reality it is more likely to come in bursts. This
    # can lead to uneven infiltration and more runoff. To simulate this, we use a pre-generated lookup table of
    # log-normally distributed rainfall patterns across the 6 substeps. We use the seed to select a random row from the table,
    # which gives us a unique but deterministic rainfall distribution for this cell and timestep.
    # The total rainfall across the substeps is normalized to match the input rainfall for the timestep,
    # so we are not changing the total amount of water, just how it is distributed within the hour.
    idx: np.uint64 = splitmix64(np.uint64(seed))
    if distribute_rainfall_lognormally:
        rainfall_lookup_table_row: ArrayFloat32 = rainfall_lookup_table[idx & MASK]
    else:
        rainfall_lookup_table_row: ArrayFloat32 = np.full(
            n_substeps, 1.0 / n_substeps, dtype=np.float32
        )
    for substep_i in range(n_substeps):
        # Calculate the amount of water available for infiltration in this substep based on the rainfall distribution.
        rainfall_ratio_substep: np.float32 = rainfall_lookup_table_row[substep_i]
        water_input_substep_m: np.float32 = (
            new_surface_water_input_m * rainfall_ratio_substep
        )
        if land_use_type == SEALED:
            # Direct impervious fraction for sealed surfaces (e.g., roofs/gutters routed directly to drainage)
            sealed_impervious_step: np.float32 = water_input_substep_m * np.float32(
                0.35
            )
            impervious_direct_runoff_m += sealed_impervious_step
            rain_step: np.float32 = water_input_substep_m - sealed_impervious_step
        else:
            rain_step = water_input_substep_m

        # Partition incoming rain into Dunne saturation-excess on saturated riparian areas
        # and precipitation falling on the unsaturated hillslope area.
        step_sat_excess: np.float32 = rain_step * saturated_area_fraction
        p_unsat: np.float32 = rain_step * unsaturated_area_fraction

        # Add enthalpy from newly reaching rain/irrigation to the top-soil control volume.
        # This water enters at rain_temperature_C and equilibrates with the top layer.
        soil_enthalpy_top_layer_J_per_m2 = apply_rain_heat_advection(
            soil_enthalpy_top_layer_J_per_m2=soil_enthalpy_top_layer_J_per_m2,
            liquid_water_input_m=water_input_substep_m,
            rain_temperature_C=max(rain_temperature_C, np.float32(0.0)),
        )

        _, substep_frozen_fraction_top_layer = (
            get_temperature_and_frozen_fraction_from_enthalpy_scalar(
                enthalpy_J_per_m2=soil_enthalpy_top_layer_J_per_m2,
                solid_heat_capacity_J_per_m2_K=solid_heat_capacity_top_layer_J_per_m2_K,
                water_content_m=w[0],
                topwater_m=water_input_substep_m,
            )
        )
        liquid_fraction_top_layer = np.float32(1.0) - np.minimum(
            np.maximum(substep_frozen_fraction_top_layer, np.float32(0.0)),
            np.float32(1.0),
        )

        # Check if the wetting front has moved into a new layer
        # and if so, update Green-Ampt parameters. Otherwise, keep existing parameters.
        # note that this assumes that the suction remains stable
        # as long as the front is within the same layer.
        if green_ampt_active_layer_idx < len(
            soil_layer_height_m
        ) - 1 and wetting_front_depth_m >= current_layer_depth_limit - np.float32(1e-4):
            (
                green_ampt_active_layer_idx,
                wetting_front_suction_head_m,
                wetting_front_moisture_deficit,
            ) = get_green_ampt_params(
                wetting_front_depth_m,
                soil_layer_height_m,
                w,
                ws,
                wres,
                bubbling_pressure_m_positive,
                lambda_pore_size_distribution,
            )
            # Update limit for the new layer
            current_layer_depth_limit = np.float32(0.0)
            for i in range(green_ampt_active_layer_idx + 1):
                current_layer_depth_limit += soil_layer_height_m[i]

        # Calculate current cumulative infiltration implied by the wetting front depth
        current_cumulative_infiltration: np.float32 = (
            wetting_front_depth_m * wetting_front_moisture_deficit
        )

        saturated_conductivity_of_most_restrictive_layer = (
            saturated_hydraulic_conductivity_m_per_s[0]
        ) * np.float32(
            0.2
        )  # Apply a crust factor to the top layer to account for surface sealing.

        # Loop through other layers up to the active layer to find the most restrictive conductivity.
        for i in range(1, green_ampt_active_layer_idx + 1):
            saturated_conductivity_of_most_restrictive_layer = min(
                saturated_conductivity_of_most_restrictive_layer,
                saturated_hydraulic_conductivity_m_per_s[i],
            )

        # Calculate effective time since start of infiltration event
        # If wetting_front_depth is negligible, we start at t=0
        if wetting_front_depth_m == np.float32(0.0):
            effective_seconds_since_start_infiltration: np.float32 = np.float32(0.0)
        else:
            effective_seconds_since_start_infiltration: np.float32 = (
                calculate_green_ampt_time_from_infiltration(
                    current_cumulative_infiltration,
                    saturated_conductivity_of_most_restrictive_layer,
                    wetting_front_suction_head_m,
                    wetting_front_moisture_deficit,
                )
            )

        # Calculate potential cumulative infiltration at end of substep
        # We advance time by substep_time_s
        seconds_since_start_of_infiltration: np.float32 = (
            effective_seconds_since_start_infiltration + substep_time_s
        )

        potential_cumulative_infiltration = (
            calculate_green_ampt_potential_cumulative_infiltration(
                seconds_since_start_of_infiltration,
                saturated_conductivity_of_most_restrictive_layer,
                wetting_front_suction_head_m,
                wetting_front_moisture_deficit,
                adjust_for_coarse_soils=False,
            )
        )

        # Determine infiltration capacity for this substep
        infiltration_capacity_m_step: np.float32 = max(
            np.float32(0.0),
            potential_cumulative_infiltration - current_cumulative_infiltration,
        )
        infiltration_capacity_m_step *= liquid_fraction_top_layer

        # Available infiltration capacity on the unsaturated portion of the cell
        f_unsat_step: np.float32 = (
            infiltration_capacity_m_step * unsaturated_area_fraction
        )

        # Determine how deep we can infiltrate:
        # Scan for the first layer with available space starting from active layer.
        end_layer_idx = min(len(w), green_ampt_active_layer_idx + 1)
        for i in range(green_ampt_active_layer_idx, len(w)):
            if (ws[i] - w[i]) > np.float32(1e-4):
                end_layer_idx = i + 1
                break
            end_layer_idx = i + 1

        # Calculate available space up to the target layer
        space_available: np.float32 = np.float32(0.0)
        for i in range(end_layer_idx):
            space_available += max(np.float32(0.0), ws[i] - w[i])

        total_soil_depth: np.float32 = np.sum(soil_layer_height_m)
        wetting_front_at_bottom: bool = bool(
            wetting_front_depth_m >= total_soil_depth - np.float32(1e-4)
        )

        # Infiltrate rain falling on the unsaturated fraction
        if variable_runoff_shape_beta > np.float32(0.0):
            pot_infil_rain, step_horton_excess = calculate_spatial_infiltration_excess(
                infiltration_capacity_mean=f_unsat_step,
                available_water=p_unsat,
                shape_parameter_beta=variable_runoff_shape_beta,
            )
        else:
            pot_infil_rain = min(p_unsat, f_unsat_step)
            step_horton_excess = p_unsat - pot_infil_rain

        infil_from_rain: np.float32 = min(pot_infil_rain, space_available)
        step_horton_excess += pot_infil_rain - infil_from_rain

        # Pre-existing ponded surface water can infiltrate if unsaturated soil has spare infiltration capacity
        spare_capacity: np.float32 = max(
            np.float32(0.0), f_unsat_step - infil_from_rain
        )
        puddle_space: np.float32 = max(
            np.float32(0.0), space_available - infil_from_rain
        )
        puddle_infil: np.float32 = min(
            current_topwater_m, min(spare_capacity, puddle_space)
        )
        current_topwater_m -= puddle_infil
        step_infiltration: np.float32 = infil_from_rain + puddle_infil

        excess_surface_water: np.float32 = step_sat_excess + step_horton_excess
        if wetting_front_at_bottom:
            water_available_for_recharge: np.float32 = (
                excess_surface_water + current_topwater_m
            )
            recharge_capacity_m_step: np.float32 = (
                max(np.float32(0.0), groundwater_toplayer_conductivity_m_per_s)
                * substep_time_s
            )
            step_groundwater_recharge_m: np.float32 = min(
                water_available_for_recharge, recharge_capacity_m_step
            )
            step_groundwater_recharge_m *= (
                capillary_rise_from_groundwater_m <= np.float32(0.0)
            )
            groundwater_recharge_m += step_groundwater_recharge_m

            # Recharging water is drawn first from surface excess, then from current_topwater_m
            recharge_from_excess: np.float32 = min(
                excess_surface_water, step_groundwater_recharge_m
            )
            recharge_from_topwater: np.float32 = (
                step_groundwater_recharge_m - recharge_from_excess
            )
            current_topwater_m -= recharge_from_topwater
            step_excess_unabsorbed: np.float32 = (
                excess_surface_water - recharge_from_excess
            )
        else:
            step_excess_unabsorbed = excess_surface_water

        total_infiltration_amount += step_infiltration

        # Update wetting front depth
        # L_new = L_old + Infiltration / DeltaTheta
        if step_infiltration > np.float32(
            0.0
        ) and wetting_front_moisture_deficit > np.float32(1e-6):
            wetting_front_depth_m += step_infiltration / wetting_front_moisture_deficit
            wetting_front_depth_m = min(wetting_front_depth_m, total_soil_depth)

        # Update soil layers sequentially from top to bottom
        remaining_infiltration: np.float32 = step_infiltration
        for i in range(end_layer_idx):
            if remaining_infiltration <= np.float32(1e-9):
                break

            space_in_layer: np.float32 = max(np.float32(0.0), ws[i] - w[i])
            infiltration_to_layer: np.float32 = min(
                remaining_infiltration, space_in_layer
            )

            w[i] += infiltration_to_layer
            # Ensure we don't exceed saturation due to float errors
            w[i] = min(w[i], ws[i])

            remaining_infiltration -= infiltration_to_layer

        # Unabsorbed surface water generated in this substep (Dunne saturation excess,
        # Hortonian infiltration excess, and profile saturation excess) adds to the surface puddle pool
        current_topwater_m += step_excess_unabsorbed

        # Partition accumulated surface pool into retained puddle storage and spillover direct runoff
        if land_use_type == PADDY_IRRIGATED:
            retained_topwater_m: np.float32 = min(current_topwater_m, ponding_allowance)
            step_direct_runoff: np.float32 = current_topwater_m - retained_topwater_m
            current_topwater_m = retained_topwater_m
        elif land_use_type == OPEN_WATER:
            step_direct_runoff = current_topwater_m
            current_topwater_m = np.float32(0.0)
        else:
            current_topwater_m, step_direct_runoff = (
                calculate_depression_spillover_runoff(
                    topwater_m=current_topwater_m,
                    depression_storage_capacity_m=depression_storage_m,
                    runoff_shape_parameter=runoff_shape_parameter,
                )
            )

        total_direct_runoff += step_direct_runoff

    topwater_m = current_topwater_m
    total_direct_runoff += impervious_direct_runoff_m

    # Update enthalpy for water leaving the top-soil control volume as direct runoff.
    if total_direct_runoff > np.float32(0.0) and land_use_type != PADDY_IRRIGATED:
        # Water content of the top-soil control volume BEFORE direct runoff leaves
        # includes both the retained topwater and the departing direct runoff.
        (
            top_layer_temp_C,
            _,
        ) = get_temperature_and_frozen_fraction_from_enthalpy_scalar(
            enthalpy_J_per_m2=soil_enthalpy_top_layer_J_per_m2,
            solid_heat_capacity_J_per_m2_K=solid_heat_capacity_top_layer_J_per_m2_K,
            water_content_m=w[0],
            topwater_m=topwater_m + total_direct_runoff,
        )
        # Only advect sensible heat (T > 0).
        runoff_advection_temp_C: np.float32 = max(top_layer_temp_C, np.float32(0.0))
        advected_runoff_enthalpy_J_per_m2: np.float32 = (
            total_direct_runoff
            * np.float32(1000.0)  # RHO_WATER
            * np.float32(4186.0)  # C_WATER
            * runoff_advection_temp_C
        )
        soil_enthalpy_top_layer_J_per_m2 -= advected_runoff_enthalpy_J_per_m2

    return (
        topwater_m,
        total_direct_runoff,
        groundwater_recharge_m,
        total_infiltration_amount,
        wetting_front_depth_m,
        wetting_front_suction_head_m,
        wetting_front_moisture_deficit,
        green_ampt_active_layer_idx,
        soil_enthalpy_top_layer_J_per_m2,
    )


@njit(
    cache=True,
    inline="always",
)
def get_soil_moisture_at_pressure(
    pressure_head_m: float | np.floating | np.ndarray[Shape, np.dtype[np.float32]],
    bubbling_pressure_m_positive: np.ndarray[Shape, np.dtype[np.float32]],
    thetas: np.ndarray[Shape, np.dtype[np.float32]],
    thetar: np.ndarray[Shape, np.dtype[np.float32]],
    lambda_: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Calculates the soil moisture content at a given soil water potential (capillary suction) using the van Genuchten model.

    Args:
        pressure_head_m: The soil pressure_head. Must be negative. (m)
        bubbling_pressure_m_positive: The bubbling pressure (m)
        thetas: The saturated soil moisture content (m³/m³)
        thetar: The residual soil moisture content (m³/m³)
        lambda_: Lambda pore size distribution parameter (dimensionless)

    Returns:
        The soil moisture content at the given soil water potential (m³/m³)
    """
    alpha = np.float32(1) / bubbling_pressure_m_positive
    n = lambda_ + np.float32(1)
    m = np.float32(1) - np.float32(1) / n
    phi = -pressure_head_m

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
    bulk_density_kg_per_dm3: np.ndarray[Shape, np.dtype[np.float32]],
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
        bulk_density_kg_per_dm3: bulk density [kg/dm3].
        clay: clay percentage [%].
        silt: silt percentage [%].
        is_top_soil: top soil flag.

    Returns:
        thetas: saturated water content [m3/m3].

    """
    return (
        np.float32(0.6819)
        - np.float32(0.06480) * (1 / (organic_carbon_percentage + 1))
        - np.float32(0.11900) * bulk_density_kg_per_dm3**2
        - np.float32(0.02668) * is_top_soil
        + np.float32(0.001489) * clay
        + np.float32(0.0008031) * silt
        + np.float32(0.02321)
        * (1 / (organic_carbon_percentage + 1))
        * bulk_density_kg_per_dm3**2
        + np.float32(0.01908) * bulk_density_kg_per_dm3**2 * is_top_soil
        - np.float32(0.0011090) * clay * is_top_soil
        - np.float32(0.00002315) * silt * clay
        - np.float32(0.0001197) * silt * bulk_density_kg_per_dm3**2
        - np.float32(0.0001068) * clay * bulk_density_kg_per_dm3**2
    )


def thetas_wosten(
    clay: np.ndarray[Shape, np.dtype[np.float32]],
    bulk_density_kg_per_dm3: np.ndarray[Shape, np.dtype[np.float32]],
    silt: np.ndarray[Shape, np.dtype[np.float32]],
    organic_carbon_percentage: np.ndarray[Shape, np.dtype[np.float32]],
    is_topsoil: np.ndarray[Shape, np.dtype[np.bool_]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Calculates the saturated water content (theta_S) based on the provided equation.

    From: https://doi.org/10.1016/S0016-7061(98)00132-3

    Args:
        clay: Clay percentage (C).
        bulk_density_kg_per_dm3: Bulk density (D).
        silt: Silt percentage (S).
        organic_carbon_percentage: Organic matter percentage (OM).
        is_topsoil: 1 for topsoil, 0 for subsoil.

    Returns:
        float: The calculated saturated water content (theta_S).
    """
    theta_s = (
        0.7919
        + 0.00169 * clay
        - 0.29619 * bulk_density_kg_per_dm3
        - 0.000001491 * silt**2
        + 0.0000821 * organic_carbon_percentage**2
        + 0.02427 * (1 / clay)
        + 0.01113 * (1 / silt)
        + 0.01472 * np.log(silt)
        - 0.0000733 * organic_carbon_percentage * clay
        - 0.000619 * bulk_density_kg_per_dm3 * clay
        - 0.001183 * bulk_density_kg_per_dm3 * organic_carbon_percentage
        - 0.0001664 * is_topsoil * silt
    ).astype(np.float32)

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


def get_bubbling_pressure_m_positive(
    clay: np.ndarray[Shape, np.dtype[np.float32]],
    sand: np.ndarray[Shape, np.dtype[np.float32]],
    thetas: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Determine bubbling pressure [m].

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
        bubbling_pressure: bubbling pressure [m].
    """
    bubbling_pressure_cm: np.ndarray[Shape, np.dtype[np.float32]] = np.exp(
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
    ).astype(np.float32)
    return bubbling_pressure_cm / np.float32(
        100.0
    )  # convert from cm to m  # ty:ignore[invalid-return-type]


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
    ).astype(np.float32)

    return poresizeindex


def get_pore_size_index_wosten(
    clay: np.ndarray[Shape, np.dtype[np.float32]],
    silt: np.ndarray[Shape, np.dtype[np.float32]],
    organic_carbon_percentage: np.ndarray[Shape, np.dtype[np.float32]],
    bulk_density_kg_per_dm3: np.ndarray[Shape, np.dtype[np.float32]],
    is_top_soil: np.ndarray[Shape, np.dtype[np.bool_]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Determine Brooks-Corey pore size distribution index [-].

    See: https://doi.org/10.1016/S0016-7061(98)00132-3

    Args:
        clay: clay percentage [%].
        silt: silt percentage [%].
        organic_carbon_percentage: soil organic carbon content [%].
        bulk_density_kg_per_dm3: bulk density [kg/dm3].
        is_top_soil: top soil flag.

    Returns:
        pore size distribution index [-].
    """
    return np.exp(
        -25.23
        - 0.02195 * clay
        + 0.0074 * silt
        - 0.1940 * organic_carbon_percentage
        + 45.5 * bulk_density_kg_per_dm3
        - 7.24 * bulk_density_kg_per_dm3**2
        + 0.0003658 * clay**2
        + 0.002855 * organic_carbon_percentage**2
        - 12.81 * bulk_density_kg_per_dm3**-1
        - 0.1524 * silt**-1
        - 0.01958 * organic_carbon_percentage**-1
        - 0.2876 * np.log(silt)
        - 0.0709 * np.log(organic_carbon_percentage)
        - 44.6 * np.log(bulk_density_kg_per_dm3)
        - 0.02264 * bulk_density_kg_per_dm3 * clay
        + 0.0896 * bulk_density_kg_per_dm3 * organic_carbon_percentage
        + 0.00718 * is_top_soil * clay
    ).astype(np.float32)


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
    return kv.astype(np.float32)


def kv_wosten(
    silt: np.ndarray[Shape, np.dtype[np.float32]],
    clay: np.ndarray[Shape, np.dtype[np.float32]],
    bulk_density_kg_per_dm3: np.ndarray[Shape, np.dtype[np.float32]],
    organic_carbon_percentage: np.ndarray[Shape, np.dtype[np.float32]],
    is_topsoil: np.ndarray[Shape, np.dtype[np.bool_]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Calculates the saturated value based on the provided equation.

    From: https://doi.org/10.1016/S0016-7061(98)00132-3

    Args:
        silt: Silt percentage (S).
        is_topsoil: 1 for topsoil, 0 for subsoil.
        bulk_density_kg_per_dm3: Bulk density (D).
        clay: Clay percentage (C).
        organic_carbon_percentage: Organic matter percentage (OM).

    Returns:
        float: The calculated Ks* value [m/s].
    """
    ks: np.ndarray[Shape, np.dtype[np.float32]] = np.exp(
        7.755
        + 0.0352 * silt
        + np.float32(0.93) * is_topsoil
        - 0.967 * bulk_density_kg_per_dm3**2
        - 0.000484 * clay**2
        - 0.000322 * silt**2
        + 0.001 * (1 / silt)
        - 0.0748 * (1 / organic_carbon_percentage)
        - 0.643 * np.log(silt)
        - 0.01398 * bulk_density_kg_per_dm3 * clay
        - 0.1673 * bulk_density_kg_per_dm3 * organic_carbon_percentage
        + 0.02986 * np.float32(is_topsoil) * clay
        - 0.03305 * np.float32(is_topsoil) * silt
    ) / (100 * 86400)  # convert to m/s

    return ks.astype(np.float32)


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
    INCH_TO_M = 0.0254
    HOUR_TO_S = 3600.0

    ks_in_hr = 10 ** (-0.6 + 0.0126 * sand - 0.0064 * clay)
    kv = ks_in_hr * INCH_TO_M / HOUR_TO_S

    return kv.astype(np.float32)
