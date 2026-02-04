"""Soil hydrology functions."""

import numpy as np
from numba import njit

from geb.geb_types import ArrayFloat32, Shape

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
def calculate_spatial_infiltration_excess(
    infiltration_capacity_mean: np.float32,
    available_water: np.float32,
    shape_parameter_beta: np.float32,
) -> tuple[np.float32, np.float32]:
    """Calculate effective infiltration and runoff using spatial variability of infiltration capacity.

    This function implements a Hortonian runoff generation mechanism assuming that the
    Green-Ampt infiltration capacity within the cell follows a Reflected Power distribution.

    Args:
        infiltration_capacity_mean: The mean infiltration capacity (Green-Ampt capacity) in the cell [L/T].
        available_water: The amount of water available for infiltration (e.g. precipitation) [L/T].
        shape_parameter_beta: The shape parameter `b` of the Reflected Power distribution.

    Returns:
        A tuple containing:
            - infiltration: Effective infiltration amount [L/T].
            - runoff: Runoff amount due to infiltration excess [L/T].
    """
    if available_water <= np.float32(0.0):
        return np.float32(0.0), np.float32(0.0)

    if infiltration_capacity_mean <= np.float32(0.0):
        return np.float32(0.0), available_water

    # Calculate max capacity: f_max = f_GA * (b + 1)
    max_infiltration_capacity = infiltration_capacity_mean * (
        shape_parameter_beta + np.float32(1.0)
    )

    if available_water >= max_infiltration_capacity:
        # If precipitation exceeds max capacity, infiltration is limited by the mean capacity
        infiltration = infiltration_capacity_mean
    else:
        # Integration of infiltration capacity < available water portion
        # f_avg = f_GA * [1 - (1 - P/f_max)^(b+1)]
        ratio = available_water / max_infiltration_capacity
        power = shape_parameter_beta + np.float32(1.0)

        infiltration = infiltration_capacity_mean * (
            np.float32(1.0) - (np.float32(1.0) - ratio) ** power
        )

    # Clamp infiltration to available water to prevent precision errors
    if infiltration > available_water:
        infiltration = available_water

    runoff: np.float32 = available_water - infiltration
    # Prevent negative runoff due to float precision
    runoff: np.float32 = max(runoff, np.float32(0.0))

    return infiltration, runoff


@njit(cache=True)
def calculate_green_ampt_time_from_infiltration(
    cumulative_infiltration: np.float32,
    saturated_hydraulic_conductivity_m_per_time_unit: np.float32,
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
        cumulative_infiltration: Cumulative infiltration amount [L].
        saturated_hydraulic_conductivity_m_per_time_unit: Saturated hydraulic conductivity [L/T].
        wetting_front_suction_head_m: Wetting front suction head [L].
        moisture_deficit: Moisture deficit [-].

    Returns:
        Time t corresponding to the infiltration amount.
    """
    if cumulative_infiltration <= np.float32(0.0):
        return np.float32(0.0)

    if saturated_hydraulic_conductivity_m_per_time_unit <= np.float32(0.0):
        return np.float32(0.0)

    wetting_front_suction_potential: np.float32 = (
        wetting_front_suction_head_m * moisture_deficit
    )

    # Darcy limit: if there is no capillary suction effect (sf -> 0),
    # then cumulative infiltration is I = K_s t.
    if wetting_front_suction_potential <= np.float32(0.0):
        return (
            cumulative_infiltration / saturated_hydraulic_conductivity_m_per_time_unit
        )

    # np.log1p(x) computes log(1 + x) accurately for small x
    term_log = np.log1p(cumulative_infiltration / wetting_front_suction_potential)
    t = (
        cumulative_infiltration - wetting_front_suction_potential * term_log
    ) / saturated_hydraulic_conductivity_m_per_time_unit
    return max(t, np.float32(0.0))


@njit(cache=True)
def calculate_green_ampt_cumulative_infiltration(
    time: np.float32,
    saturated_hydraulic_conductivity_m_per_time_unit: np.float32,
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
        time: Time since start of infiltration [T].
        saturated_hydraulic_conductivity_m_per_time_unit: Saturated hydraulic conductivity (K_s) [L/T].
        wetting_front_suction_head_m: Wetting front suction head (Δθ) [L].
        moisture_deficit: Moisture deficit [-].
        adjust_for_coarse_soils: Whether to apply adjustment for coarse soils. For coarse soils,
            and very long times, the Sageghi et al. formula can be slightly more off than the
            0.3 percent error. Setting this to True applies an empirical adjustment to improve accuracy
            in those situations.

    Returns:
        Cumulative infiltration amount [L].
    """
    # Darcy limit: if suction or the moisture deficit is zero, there is no
    # capillarity-driven enhancement and Green-Ampt reduces to I = K_s t.
    if wetting_front_suction_head_m <= np.float32(
        0.0
    ) or moisture_deficit <= np.float32(0.0):
        return saturated_hydraulic_conductivity_m_per_time_unit * time

    # Sorptivity can be calculated as per Philip (1969):
    # S^2 = 2 K_s * psi * dtheta
    # Reference: Philip, J.R., 1969. Theory of infiltration. In Advances in hydroscience (Vol. 5, pp. 215-296). Elsevier.
    # Since the Sadeghi formula uses S^2 directly, we do not need to take the square root.
    sorptivity_squared: np.float32 = (
        np.float32(2.0)
        * saturated_hydraulic_conductivity_m_per_time_unit
        * wetting_front_suction_head_m
        * moisture_deficit
    )

    # Apply Sadeghi et al. (2024) explicit formula
    hydraulic_conductivity_times_time: np.float32 = (
        saturated_hydraulic_conductivity_m_per_time_unit * time
    )

    # sorptivity_time_ratio corresponds to S^2 / (Ks^2 * t)
    sorptivity_time_ratio: np.float32 = sorptivity_squared / (
        saturated_hydraulic_conductivity_m_per_time_unit
        * hydraulic_conductivity_times_time
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
            sorptivity_squared / saturated_hydraulic_conductivity_m_per_time_unit
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


@njit(cache=True, inline="always")
def get_green_ampt_params(
    wetting_front_depth_m: np.float32,
    soil_layer_height_m: ArrayFloat32,
    w: ArrayFloat32,
    ws: ArrayFloat32,
    wres: ArrayFloat32,
    saturated_hydraulic_conductivity_m_per_timestep: ArrayFloat32,
    bubbling_pressure_cm: ArrayFloat32,
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
        saturated_hydraulic_conductivity_m_per_timestep: Saturated hydraulic conductivity of each layer (m/timestep).
        bubbling_pressure_cm: Bubbling pressure parameter for each layer (cm).
        lambda_pore_size_distribution: Pore size distribution index (lambda) for each layer.

    Returns:
        A tuple containing:
            - idx (int): Index of the soil layer containing the wetting front.
            - psi (float): Wetting front suction head (meters).
            - delta_theta (float): Moisture deficit at the wetting front (dimensionless, m/m).

    Notes:
        - Assumes piston flow: Soil behind the wetting front is fully saturated.
        - Calculates moisture ahead of the front by subtracting the saturated water volume behind the front
          from the total layer water volume.
        - Includes heuristics (theta_floor) to prevent numerical instability when layers interact.
    """
    current_depth = np.float32(0.0)
    idx = 0
    depth_in_layer = np.float32(0.0)
    n_layers = len(soil_layer_height_m)

    # Find which layer the wetting front is currently in
    for i in range(n_layers):
        h = soil_layer_height_m[i]
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

    layer_h = soil_layer_height_m[idx]

    # Reconstruct initial moisture content ahead of the front.
    # w[idx] is the total water column in the layer (meters).
    # Since we assume piston flow, the part of the layer behind the front (depth_in_layer)
    # is saturated using the Green-Ampt assumption.
    theta_sat = ws[idx] / layer_h
    remaining_height = layer_h - depth_in_layer

    if remaining_height > np.float32(1e-4):
        # Mass balance: Total Water = Water_Behind + Water_Ahead
        water_behind = depth_in_layer * theta_sat
        water_ahead = max(np.float32(0.0), w[idx] - water_behind)
        theta_initial = water_ahead / remaining_height

        # Clamp to physical limits
        # use a small epsilon as floor
        theta_floor = np.float32(1e-9)

        theta_initial = max(theta_initial, theta_floor)
        theta_initial = min(theta_initial, theta_sat - np.float32(1e-9))
    else:
        # Layer fully invaded; assume near saturation (small deficit)
        theta_initial = theta_sat - np.float32(1e-3)

    delta_theta = max(theta_sat - theta_initial, np.float32(1e-4))

    # Calculate suction based on the initial moisture
    # get_soil_water_flow_parameters expects water content in meters (w), not theta
    w_initial_equiv = theta_initial * layer_h
    psi, _ = get_soil_water_flow_parameters(
        w=max(w_initial_equiv, wres[idx]),
        wres=wres[idx],
        ws=ws[idx],
        lambda_pore_size_distribution=lambda_pore_size_distribution[idx],
        saturated_hydraulic_conductivity_m_per_timestep=saturated_hydraulic_conductivity_m_per_timestep[
            idx
        ],
        bubbling_pressure_cm=bubbling_pressure_cm[idx],
    )

    return idx, abs(psi), delta_theta


@njit(cache=True, inline="always")
def infiltration(
    ws: ArrayFloat32,
    wres: ArrayFloat32,
    saturated_hydraulic_conductivity_m_per_timestep: ArrayFloat32,
    groundwater_toplayer_conductivity_m_per_timestep: np.float32,
    land_use_type: np.int32,
    soil_is_frozen: bool,
    w: ArrayFloat32,
    topwater_m: np.float32,
    capillary_rise_from_groundwater_m: np.float32,
    wetting_front_depth_m: np.float32,
    wetting_front_suction_head_m: np.float32,
    wetting_front_moisture_deficit: np.float32,
    green_ampt_active_layer_idx: int,
    variable_runoff_shape_beta: np.float32,
    bubbling_pressure_cm: ArrayFloat32,
    soil_layer_height_m: ArrayFloat32,
    lambda_pore_size_distribution: ArrayFloat32,
) -> tuple[
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    int,
]:
    """Simulates vertical transport of water in the soil for a single cell.

    Uses an explicit Green-Ampt approximation (Salvucci, 1994) combined with the
    PDM variable infiltration capacity curve.

    The function uses `wetting_front_suction_head_m` which is a state variable
    tracking the matric suction at the sharp wetting front. This should be
    initialized at the beginning of an infiltration event based on the soil
    moisture deficit.

    Args:
        ws: Saturated soil water content in each layer for the cell in meters, shape (N_SOIL_LAYERS,).
        wres: Residual soil water content in each layer for the cell in meters, shape (N_SOIL_LAYERS,).
        saturated_hydraulic_conductivity_m_per_timestep: Saturated hydraulic conductivity in each layer for the cell in m in this timestep, shape (N_SOIL_LAYERS,).
        groundwater_toplayer_conductivity_m_per_timestep: Groundwater top layer conductivity limiting recharge (m/timestep).
        land_use_type: Land use type for the cell.
        soil_is_frozen: Boolean indicating if the soil is frozen.
        w: Soil water content in each layer for the cell in meters, shape (N_SOIL_LAYERS,), modified in place.
        topwater_m: Topwater for the cell in meters, modified in place.
        capillary_rise_from_groundwater_m: Capillary rise from groundwater for the cell (m/timestep). If >0, percolation to groundwater is suppressed.
        wetting_front_depth_m: Depth of the wetting front in meters.
        wetting_front_suction_head_m: Suction head at the wetting front in meters.
        wetting_front_moisture_deficit: Moisture deficit at the wetting front [-].
        green_ampt_active_layer_idx: The index of the active soil layer for Green-Ampt.
        variable_runoff_shape_beta: Shape parameter `b` for the PDM distribution.
        bubbling_pressure_cm: Bubbling pressure for each soil layer [cm], shape (N_SOIL_LAYERS,).
        soil_layer_height_m: Height of each soil layer [m], shape (N_SOIL_LAYERS,).
        lambda_pore_size_distribution: Van Genuchten parameter lambda for each soil layer, shape (N_SOIL_LAYERS,).

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
    """
    if soil_is_frozen or land_use_type == SEALED or land_use_type == OPEN_WATER:
        # No infiltration allowed
        infiltration_amount: np.float32 = np.float32(0.0)
        direct_runoff: np.float32 = topwater_m
        topwater_m: np.float32 = np.float32(0.0)
        return (
            topwater_m,
            direct_runoff,
            np.float32(0.0),  # groundwater_recharge
            infiltration_amount,
            wetting_front_depth_m,
            wetting_front_suction_head_m,
            wetting_front_moisture_deficit,
            green_ampt_active_layer_idx,
        )

    # Redistribution: If there is no water available for infiltration,
    # we assume the wetting front dissipates/redistributes.
    if topwater_m == np.float32(0.0):
        wetting_front_depth_m: np.float32 = np.float32(0.0)
        green_ampt_active_layer_idx: int = 0

    # Initialize accumulators for the timestep
    total_infiltration_amount: np.float32 = np.float32(0.0)
    total_direct_runoff: np.float32 = np.float32(0.0)
    groundwater_recharge_m: np.float32 = np.float32(0.0)

    n_substeps: int = 6
    dt: np.float32 = np.float32(1.0) / np.float32(n_substeps)
    topwater_per_step: np.float32 = topwater_m / np.float32(n_substeps)

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
            saturated_hydraulic_conductivity_m_per_timestep,
            bubbling_pressure_cm,
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

    for _ in range(n_substeps):
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
                saturated_hydraulic_conductivity_m_per_timestep,
                bubbling_pressure_cm,
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

        # Calculate effective time since start of infiltration event
        # If wetting_front_depth is negligible, we start at t=0
        if wetting_front_depth_m == np.float32(0.0):
            effective_time_steps = np.float32(0.0)
        else:
            effective_time_steps = calculate_green_ampt_time_from_infiltration(
                current_cumulative_infiltration,
                saturated_hydraulic_conductivity_m_per_timestep[
                    green_ampt_active_layer_idx
                ],
                wetting_front_suction_head_m,
                wetting_front_moisture_deficit,
            )

        # Calculate potential cumulative infiltration at end of substep
        # We advance time by dt (fraction of timestep)
        new_time_steps = effective_time_steps + dt

        potential_cumulative_infiltration = (
            calculate_green_ampt_cumulative_infiltration(
                new_time_steps,
                saturated_hydraulic_conductivity_m_per_timestep[
                    green_ampt_active_layer_idx
                ],
                wetting_front_suction_head_m,
                wetting_front_moisture_deficit,
                adjust_for_coarse_soils=False,
            )
        )

        # Determine infiltration capacity for this substep
        infiltration_capacity_m_step = (
            potential_cumulative_infiltration - current_cumulative_infiltration
        )

        # Calculate potential infiltration considering spatial variability of infiltration capacity
        (
            potential_topwater_that_can_infiltrate,
            step_runoff,
        ) = calculate_spatial_infiltration_excess(
            infiltration_capacity_mean=max(
                np.float32(0.0), infiltration_capacity_m_step
            ),
            available_water=topwater_per_step,
            shape_parameter_beta=variable_runoff_shape_beta,
        )

        # Determine how deep we can infiltrate:
        # Scan for the first layer with available space starting from active layer.
        # This allows the wetting front to "jump" through saturated layers (where dL/dI -> inf)
        # but prevents pre-wetting deep unsaturated layers which would break Green-Ampt physics.
        end_layer_idx = min(len(w), green_ampt_active_layer_idx + 1)
        for i in range(green_ampt_active_layer_idx, len(w)):
            if (ws[i] - w[i]) > np.float32(1e-4):
                end_layer_idx = i + 1
                break
            # If layer is full, we continue to look deeper
            end_layer_idx = i + 1

        # Calculate available space up to the target layer
        space_available = np.float32(0.0)
        for i in range(end_layer_idx):
            space_available += max(np.float32(0.0), ws[i] - w[i])

        # If the wetting front has reached the bottom of the soil column, we treat
        # the profile as a pass-through system:
        # - fill remaining storage if any
        # - route remaining water to groundwater recharge, capped by the conductivity
        #   of the groundwater top layer (m/timestep)
        # - any remaining water becomes direct runoff
        total_soil_depth = np.sum(soil_layer_height_m)
        wetting_front_at_bottom = (
            wetting_front_depth_m >= total_soil_depth - np.float32(1e-4)
        )

        if wetting_front_at_bottom:
            step_infiltration = min(
                potential_topwater_that_can_infiltrate, space_available
            )
            potential_recharge_m = (
                potential_topwater_that_can_infiltrate - step_infiltration
            )

            recharge_capacity_m_step = (
                max(np.float32(0.0), groundwater_toplayer_conductivity_m_per_timestep)
                * dt
            )

            step_groundwater_recharge_m = min(
                potential_recharge_m, recharge_capacity_m_step
            )
            step_groundwater_recharge_m *= (
                capillary_rise_from_groundwater_m <= np.float32(0.0)
            )
            groundwater_recharge_m += step_groundwater_recharge_m

            step_runoff += potential_recharge_m - step_groundwater_recharge_m
        else:
            step_infiltration = min(
                potential_topwater_that_can_infiltrate,
                space_available,
            )
            step_runoff += potential_topwater_that_can_infiltrate - step_infiltration

        total_infiltration_amount += step_infiltration
        total_direct_runoff += step_runoff

        # Update wetting front depth
        # L_new = L_old + Infiltration / DeltaTheta
        if step_infiltration > np.float32(
            0.0
        ) and wetting_front_moisture_deficit > np.float32(0.0):
            wetting_front_depth_m += step_infiltration / wetting_front_moisture_deficit

        # Update soil layers sequentially from top to bottom
        remaining_infiltration = step_infiltration
        for i in range(end_layer_idx):
            if remaining_infiltration <= np.float32(1e-9):
                break

            space_in_layer = max(np.float32(0.0), ws[i] - w[i])
            infiltration_to_layer = min(remaining_infiltration, space_in_layer)

            w[i] += infiltration_to_layer
            # Ensure we don't exceed saturation due to float errors
            w[i] = min(w[i], ws[i])

            remaining_infiltration -= infiltration_to_layer

    topwater_m: np.float32 = np.float32(0.0)

    if land_use_type == PADDY_IRRIGATED:
        ponding_allowance: np.float32 = np.float32(0.05)
        ponding = min(total_direct_runoff, ponding_allowance)
        topwater_m += ponding
        total_direct_runoff -= ponding

    return (
        topwater_m,
        total_direct_runoff,
        groundwater_recharge_m,
        total_infiltration_amount,
        wetting_front_depth_m,
        wetting_front_suction_head_m,
        wetting_front_moisture_deficit,
        green_ampt_active_layer_idx,
    )


@njit(cache=True, inline="always")
def get_soil_water_flow_parameters(
    w: np.float32,
    wres: np.float32,
    ws: np.float32,
    lambda_pore_size_distribution: np.float32,
    saturated_hydraulic_conductivity_m_per_timestep: np.float32,
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
        saturated_hydraulic_conductivity_m_per_timestep: Saturated hydraulic conductivity for the layer in m/timestep
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
    term1 = saturated_hydraulic_conductivity_m_per_timestep * np.sqrt(
        effective_saturation
    )
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
        silt: fsilt percentage [%].
        is_top_soil: top soil flag.

    Returns:
        thetas: saturated water content [cm3/cm3].

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


def get_heat_capacity_solid_fraction(
    bulk_density_kg_per_dm3: np.ndarray[Shape, np.dtype[np.float32]],
    layer_thickness_m: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Calculate the heat capacity of the solid fraction of the soil layer [J/(m2·K)].

    This calculates the total heat capacity per unit area for the solid part of the soil layer.

    Args:
        bulk_density_kg_per_dm3: Soil bulk density [kg/dm3].
        layer_thickness_m: Thickness of the soil layer [m].

    Returns:
        The areal heat capacity of the solid fraction [J/(m2·K)].
    """
    # Constants for volumetric heat capacity [J/(m3·K)]
    C_MINERAL = np.float32(2.13e6)

    # Particle density of minerals [kg/m3]
    RHO_MINERAL = np.float32(2650.0)

    # Calculate total volume fraction of solids from bulk density
    # Convert bulk density from g/cm3 to kg/m3 (factor 1000)
    phi_s = (bulk_density_kg_per_dm3 * 1000.0) / RHO_MINERAL

    # Calculate volumetric heat capacity [J/(m3·K)]
    volumetric_heat_capacity_solid = phi_s * C_MINERAL

    # Calculate areal heat capacity [J/(m2·K)]
    areal_heat_capacity = volumetric_heat_capacity_solid * layer_thickness_m

    return areal_heat_capacity.astype(np.float32)


@njit(cache=True, inline="always")
def calculate_net_radiation_flux(
    shortwave_radiation_W_per_m2: np.float32,
    longwave_radiation_W_per_m2: np.float32,
    soil_temperature_C: np.float32,
) -> tuple[np.float32, np.float32]:
    """Calculate the net radiation energy flux and its derivative.

    Calculates absorbed incoming radiation - outgoing longwave radiation.
    Also returns the derivative of the outgoing longwave radiation with respect to temperature,
    which can be used for stability calculations in explicit schemes or damping in implicit schemes.

    Args:
        shortwave_radiation_W_per_m2: Incoming shortwave [W/m2].
        longwave_radiation_W_per_m2: Incoming longwave [W/m2].
        soil_temperature_C: Current soil temperature [C].

    Returns:
        Tuple of:
            - Net radiation flux [W/m2]. Positive = warming (incoming > outgoing).
            - Derivative of outgoing radiation flux [W/m2/K] (Conductance equivalent).
    """
    # Constants (matching other functions)
    STEFAN_BOLTZMANN_CONSTANT = np.float32(5.670374419e-8)
    SOIL_EMISSIVITY = np.float32(0.95)
    SOIL_ALBEDO = np.float32(0.23)

    # Calculate Fluxes
    temperature_K = soil_temperature_C + np.float32(273.15)

    absorbed_shortwave_W = (
        np.float32(1.0) - SOIL_ALBEDO
    ) * shortwave_radiation_W_per_m2
    absorbed_longwave_W = SOIL_EMISSIVITY * longwave_radiation_W_per_m2
    incoming_W = absorbed_shortwave_W + absorbed_longwave_W

    outgoing_W = SOIL_EMISSIVITY * STEFAN_BOLTZMANN_CONSTANT * (temperature_K**4)

    net_flux_W = incoming_W - outgoing_W

    # Calculate Derivative of Outgoing Radiation with respect to T:
    # d(sigma * eps * T^4)/dT = 4 * sigma * eps * T^3
    conductance_W_per_m2_K = (
        np.float32(4.0)
        * SOIL_EMISSIVITY
        * STEFAN_BOLTZMANN_CONSTANT
        * (temperature_K**3)
    )

    return net_flux_W, conductance_W_per_m2_K


@njit(cache=True, inline="always")
def calculate_sensible_heat_flux(
    soil_temperature_C: np.float32,
    air_temperature_K: np.float32,
    wind_speed_10m_m_per_s: np.float32,
    surface_pressure_pa: np.float32,
) -> tuple[np.float32, np.float32]:
    """Calculate the sensible heat flux and aerodynamic conductance.

    Args:
        soil_temperature_C: Soil temperature in Celsius [C].
        air_temperature_K: Air temperature at 2m height [K].
        wind_speed_10m_m_per_s: Wind speed at 10m height [m/s].
        surface_pressure_pa: Surface air pressure [Pa].

    Returns:
        Tuple of:
            - Sensible heat flux [W/m2]. Positive = warming (Heat flow from Air to Soil).
            - Aerodynamic conductance [W/m2/K].
    """
    # Physics Constants
    SPECIFIC_HEAT_AIR_J_KG_K: np.float32 = np.float32(1005.0)
    GAS_CONSTANT_AIR_J_KG_K: np.float32 = np.float32(287.058)
    VON_KARMAN_CONSTANT: np.float32 = np.float32(0.41)

    # Assumptions for Aerodynamic Resistance over bare soil
    WIND_MEASUREMENT_HEIGHT_M: np.float32 = np.float32(10.0)
    TEMP_MEASUREMENT_HEIGHT_M: np.float32 = np.float32(2.0)
    ROUGHNESS_LENGTH_M: np.float32 = np.float32(0.001)

    # Calculate Air Density [kg/m3]
    # Ideal Gas Law: rho = P / (R * T)
    # Using air temperature for density calculation
    air_density_kg_per_m3: np.float32 = surface_pressure_pa / (
        GAS_CONSTANT_AIR_J_KG_K * air_temperature_K
    )

    # Calculate Aerodynamic Resistance (ra) [s/m]
    # For neutral conditions: ra = (ln((zm-d)/z0m) * ln((zh-d)/z0h)) / (k^2 * u)
    # Assuming d=0, z0m=z0h=z0
    # where zm = wind measurement height, zh = temp measurement height, z0 = roughness length, u = wind speed

    # Ensure minimum wind speed to avoid division by zero
    wind_speed_safe = np.maximum(wind_speed_10m_m_per_s, np.float32(0.1))

    log_wind_height_over_roughness = np.log(
        WIND_MEASUREMENT_HEIGHT_M / ROUGHNESS_LENGTH_M
    )
    log_temp_height_over_roughness = np.log(
        TEMP_MEASUREMENT_HEIGHT_M / ROUGHNESS_LENGTH_M
    )

    aerodynamic_resistance_s_per_m = (
        log_wind_height_over_roughness * log_temp_height_over_roughness
    ) / (VON_KARMAN_CONSTANT**2 * wind_speed_safe)

    # Calculate Conductance [W/m2/K]
    # Conductance = rho * Cp / ra
    conductance_W_per_m2_K = (
        air_density_kg_per_m3 * SPECIFIC_HEAT_AIR_J_KG_K
    ) / aerodynamic_resistance_s_per_m

    # Calculate Explicit Sensible Heat Flux [W/m2]
    # H = Conductance * (Ta - Ts)
    air_temperature_C = air_temperature_K - np.float32(273.15)
    temperature_difference_C = air_temperature_C - soil_temperature_C

    sensible_heat_flux_W_per_m2 = conductance_W_per_m2_K * temperature_difference_C

    return sensible_heat_flux_W_per_m2, conductance_W_per_m2_K


@njit(cache=True, inline="always")
def solve_energy_balance_implicit_iterative(
    soil_temperature_C: np.float32,
    solid_heat_capacity_J_per_m2_K: np.float32,
    shortwave_radiation_W_per_m2: np.float32,
    longwave_radiation_W_per_m2: np.float32,
    air_temperature_K: np.float32,
    wind_speed_10m_m_per_s: np.float32,
    surface_pressure_pa: np.float32,
    timestep_seconds: np.float32,
) -> np.float32:
    """Update soil temperature solving energy balance with an iterative implicit scheme.

    Solves the non-linear energy balance equation using Newton-Raphson iteration.
    Equation: C * (T_new - T_old) / dt = Q_rad(T_new) + Q_sens(T_new)

    This combines the radiation and sensible heat balances into a single
    implicit solution, which is robust and stable for large time steps.

    Args:
        soil_temperature_C: Initial soil temperature [C].
        solid_heat_capacity_J_per_m2_K: Heat capacity of the soil layer [J/m2/K].
        shortwave_radiation_W_per_m2: Incoming shortwave radiation [W/m2].
        longwave_radiation_W_per_m2: Incoming longwave radiation [W/m2].
        air_temperature_K: Air temperature [K].
        wind_speed_10m_m_per_s: Wind speed [m/s].
        surface_pressure_pa: Surface pressure [Pa].
        timestep_seconds: Total time to simulate [s] (e.g. 3600.0).

    Returns:
        Updated soil temperature [C].
    """
    T_old = soil_temperature_C
    T_curr = soil_temperature_C

    # Newton-Raphson Configuration
    MAX_ITERATIONS = 10
    TOLERANCE_C = np.float32(0.01)

    for _ in range(MAX_ITERATIONS):
        # Calculate Fluxes and derivative/conductances at current estimate
        net_radiation_flux_W_per_m2, radiation_conductance_W_per_m2_K = (
            calculate_net_radiation_flux(
                shortwave_radiation_W_per_m2=shortwave_radiation_W_per_m2,
                longwave_radiation_W_per_m2=longwave_radiation_W_per_m2,
                soil_temperature_C=T_curr,
            )
        )

        sensible_heat_flux_W_per_m2, sensible_heat_conductance_W_per_m2_K = (
            calculate_sensible_heat_flux(
                soil_temperature_C=T_curr,
                air_temperature_K=air_temperature_K,
                wind_speed_10m_m_per_s=wind_speed_10m_m_per_s,
                surface_pressure_pa=surface_pressure_pa,
            )
        )

        # Function f(T_new) = C/dt * (T_new - T_old) - (Q_rad + Q_sens)
        # We want f(T_new) = 0

        storage_term_W_per_m2 = (solid_heat_capacity_J_per_m2_K / timestep_seconds) * (
            T_curr - T_old
        )
        total_flux_W_per_m2 = net_radiation_flux_W_per_m2 + sensible_heat_flux_W_per_m2

        f_val = storage_term_W_per_m2 - total_flux_W_per_m2

        # Derivative f'(T_new) = C/dt - (Q'_rad + Q'_sens)
        # Note: Q' terms are negative conductances (flux decreases as T increases)
        # Q'_rad = -radiation_conductance
        # Q'_sens = -sensible_heat_conductance
        # So f'(T_new) = C/dt + radiation_conductance + sensible_heat_conductance

        f_prime = (
            (solid_heat_capacity_J_per_m2_K / timestep_seconds)
            + radiation_conductance_W_per_m2_K
            + sensible_heat_conductance_W_per_m2_K
        )

        # Newton Step: T_next = T_curr - f(T_curr) / f'(T_curr)
        delta_T = f_val / f_prime

        T_curr -= delta_T

        if abs(delta_T) < TOLERANCE_C:
            break

    return T_curr


@njit(cache=True, inline="always")
def get_interflow(
    w: np.float32,
    wfc: np.float32,
    ws: np.float32,
    soil_layer_height_m: np.float32,
    saturated_hydraulic_conductivity_m_per_hour: np.float32,
    slope_m_per_m: np.float32,
    hillslope_length_m: np.float32,
    interflow_multiplier: np.float32,
) -> np.float32:
    """Calculate interflow from a soil layer.

    Args:
        w: Soil water content in the layer in meters.
        wfc: Field capacity soil water content in the layer in meters.
        ws: Saturated soil water content in the layer in meters.
        soil_layer_height_m: Height of the soil layer in meters.
        saturated_hydraulic_conductivity_m_per_hour: Saturated hydraulic conductivity for the layer in m/hour.
        slope_m_per_m: Slope of the terrain in m/m.
        hillslope_length_m: Length of the hillslope in meters.
        interflow_multiplier: Calibration factor for interflow calculation.

    Returns:
        Interflow from the layer in meters.
    """
    free_water_m: np.float32 = max(w - wfc, np.float32(0.0))
    drainable_porosity: np.float32 = (ws - wfc) / soil_layer_height_m

    # Convert vertical saturated hydraulic conductivity to lateral
    # Here we assume lateral conductivity is 10 times vertical
    # This factor can be adjusted based on soil anisotropy
    lateral_saturated_hydraulic_conductivity_m_per_hour = (
        saturated_hydraulic_conductivity_m_per_hour
    ) * 10

    # Implicitly assume that the step is identical to the time step of the
    # saturated hydraulic conductivity
    storage_coefficient: np.float32 = (
        lateral_saturated_hydraulic_conductivity_m_per_hour
        * slope_m_per_m
        / (drainable_porosity * hillslope_length_m)
    ) * interflow_multiplier

    interflow: np.float32 = free_water_m * storage_coefficient
    interflow: np.float32 = min(interflow, free_water_m)
    return interflow
