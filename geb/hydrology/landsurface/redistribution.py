"""Darcy flow redistribution for soil water."""

import numpy as np
from numba import njit

from geb.geb_types import ArrayFloat32
from geb.numba_config import NUMBA_CACHE
from geb.workflows.numba_stack_array import stack_empty

from .constants import (
    LATENT_HEAT_FUSION_J_PER_KG,
    N_SOIL_LAYERS,
    RHO_WATER_KG_PER_M3,
    SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K,
    SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K,
)
from .water import set_hillslope_store

_N_SOIL_LAYERS_PLUS_ONE: int = N_SOIL_LAYERS + 1


@njit(cache=NUMBA_CACHE, fastmath=True, inline="always")
def calculate_saturated_matric_flux_potential(
    saturated_hydraulic_conductivity_m_per_s: np.float32,
    bubbling_pressure_m_positive: np.float32,
    lambda_: np.float32,
    pore_size_index: np.float32,
) -> np.float32:
    """Calculate the matric flux potential at saturation (air-entry bubbling pressure).

    Based on the Campbell / Brooks-Corey Kirchhoff transform in Ross (2003):
    phi_e = (K_sat * -psi_b) / (1 - lambda * pore_size_index).

    Args:
        saturated_hydraulic_conductivity_m_per_s: Saturated hydraulic conductivity (m/s).
        bubbling_pressure_m_positive: Air-entry bubbling pressure head (m, positive).
        lambda_: van Genuchten / Brooks-Corey parameter (-).
        pore_size_index: Campbell pore size index (3 + 2/lambda) (-).

    Returns:
        Matric flux potential at saturation (m2/s).
    """
    return (
        saturated_hydraulic_conductivity_m_per_s * -bubbling_pressure_m_positive
    ) / (np.float32(1.0) - lambda_ * pore_size_index)


@njit(cache=NUMBA_CACHE, fastmath=True, inline="always")
def calculate_capillary_rise_and_percolation_parameters(
    groundwater_depth_m: np.float32,
    total_soil_depth_m: np.float32,
    bottom_layer_thickness_m: np.float32,
    bubbling_pressure_m_positive: np.float32,
    lambda_: np.float32,
    pore_size_index: np.float32,
    saturated_hydraulic_conductivity_m_per_s: np.float32,
    matric_flux_potential_bottom: np.float32,
    conductivity_bottom: np.float32,
    dmatric_flux_potential_dS_bottom: np.float32,
    dconductivity_dS_bottom: np.float32,
    liquid_fraction: np.float32,
    gw_ksat_m_per_s: np.float32,
    cos_slope: np.float32,
    max_capillary_flux_m_per_s: np.float32 = np.float32(np.inf),
) -> tuple[np.float32, np.float32]:
    """Calculate water flux between bottom soil and groundwater, and its sensitivity to soil moisture.

    Calculates bidirectional flow across the soil bottom: downward percolation
    (gravity drainage) or upward capillary rise (matric suction). Also returns
    how flux changes with bottom soil moisture for the implicit solver.

    Args:
        groundwater_depth_m: Depth of groundwater table below surface (meters).
        total_soil_depth_m: Total thickness of the soil profile (meters).
        bottom_layer_thickness_m: Thickness of the lowest soil layer (meters).
        bubbling_pressure_m_positive: Air-entry bubbling pressure head (meters, positive).
        lambda_: van Genuchten / Brooks-Corey pore size distribution index (-).
        pore_size_index: Campbell pore size index (3 + 2/lambda) (-).
        saturated_hydraulic_conductivity_m_per_s: Saturated hydraulic conductivity of bottom soil (m/s).
        matric_flux_potential_bottom: Matric flux potential in bottom soil layer (m2/s).
        conductivity_bottom: Unsaturated hydraulic conductivity in bottom soil layer (m/s).
        dmatric_flux_potential_dS_bottom: Derivative of matric flux potential with respect to saturation (m2/s).
        dconductivity_dS_bottom: Derivative of conductivity with respect to saturation (m/s).
        liquid_fraction: Unfrozen liquid water fraction of bottom soil layer (0 to 1).
        gw_ksat_m_per_s: Saturated hydraulic conductivity of groundwater toplayer (m/s).
        max_capillary_flux_m_per_s: Maximum allowable upward capillary flux based on available groundwater supply (m/s).
        cos_slope: Cosine of hillslope slope angle projecting normal gravity drainage (-).

    Returns:
        A tuple of (groundwater_flux_m_per_s, dgroundwater_flux_dS).
    """
    depth_below_soil_m: np.float32 = max(
        np.float32(0.0), groundwater_depth_m - total_soil_depth_m
    )
    effective_distance_m: np.float32 = (
        np.float32(0.5) * bottom_layer_thickness_m + depth_below_soil_m
    )

    matric_flux_potential_saturated_bottom: np.float32 = calculate_saturated_matric_flux_potential(
        saturated_hydraulic_conductivity_m_per_s=saturated_hydraulic_conductivity_m_per_s,
        bubbling_pressure_m_positive=bubbling_pressure_m_positive,
        lambda_=lambda_,
        pore_size_index=pore_size_index,
    )

    matric_flux_potential_gradient: np.float32 = (
        matric_flux_potential_bottom - matric_flux_potential_saturated_bottom
    ) / effective_distance_m

    # https://doi.org/10.1029/WR014i005p00722
    # Eagleson (1978) Eq. 57: Steady upward capillary capacity limit across unsaturated distance.
    if effective_distance_m <= bubbling_pressure_m_positive:
        eagleson_max_capillary_flux: np.float32 = (
            saturated_hydraulic_conductivity_m_per_s
        )
    else:
        # Eagleson (1978) Eq. 58: Analytical shape coefficient for steady capillary rise
        eagleson_b_coefficient: np.float32 = np.float32(1.0) + np.float32(1.5) / (
            pore_size_index - np.float32(1.0)
        )
        eagleson_max_capillary_flux = (
            saturated_hydraulic_conductivity_m_per_s
            * eagleson_b_coefficient
            * (bubbling_pressure_m_positive / effective_distance_m) ** pore_size_index
        )

    # Capillary suction flux is directed upward (negative flux in a downward-positive coordinate system).
    # We clamp the flux to [-max_upward_capillary_flux, 0.0] accounting for unsaturated capacity and aquifer supply:
    # - Lower bound (-max_upward_capillary_flux): steady upward unsaturated transmission and aquifer supply limit.
    # - Upper bound (0.0): matric suction cannot push water downward into groundwater (gravity handles downward drainage).
    max_upward_capillary_flux: np.float32 = min(
        eagleson_max_capillary_flux, max_capillary_flux_m_per_s
    )
    groundwater_capillary_flux_m_per_s: np.float32 = min(
        np.float32(0.0),
        max(-max_upward_capillary_flux, matric_flux_potential_gradient),
    )

    # Sensitivity d(flux)/dS is non-zero only within the active clamping bounds.
    if -max_upward_capillary_flux < matric_flux_potential_gradient < np.float32(0.0):
        dgroundwater_capillary_flux_dS: np.float32 = (
            dmatric_flux_potential_dS_bottom / effective_distance_m
        )
    else:
        dgroundwater_capillary_flux_dS = np.float32(0.0)

    # Total vertical flux in Richards equation is the sum of matric suction and gravity:
    # - groundwater_capillary_flux_m_per_s (<= 0): upward pull from matric suction
    # - conductivity_bottom * cos_slope (>= 0): downward drainage from slope-normal gravity
    # A soil whose base lies in the groundwater cannot drain into it.
    if groundwater_depth_m >= total_soil_depth_m:
        gravity_drainage_m_per_s: np.float32 = conductivity_bottom * cos_slope
        dgravity_drainage_dS: np.float32 = dconductivity_dS_bottom * cos_slope
    else:
        gravity_drainage_m_per_s = np.float32(0.0)
        dgravity_drainage_dS = np.float32(0.0)
    groundwater_flux_potential_m_per_s = (
        groundwater_capillary_flux_m_per_s + gravity_drainage_m_per_s
    )
    dgroundwater_flux_potential_dS = (
        dgroundwater_capillary_flux_dS + dgravity_drainage_dS
    )

    # Moisture content controls how much liquid water is available to move:
    # - Fluxes are scaled by the volumetric water fraction.
    # - Sensitivities are scaled by the same fraction.
    groundwater_flux_potential_m_per_s *= liquid_fraction
    dgroundwater_flux_potential_dS *= liquid_fraction

    # Water cannot cross the interface faster than the aquifer can conduct it (gw_ksat).
    # When pinned to this cap, the flux is fixed, so sensitivity to soil moisture is 0.
    if groundwater_flux_potential_m_per_s > gw_ksat_m_per_s:
        # Aquifer intake limit for downward percolation
        return gw_ksat_m_per_s, np.float32(0.0)

    if groundwater_flux_potential_m_per_s < -gw_ksat_m_per_s:
        # Aquifer supply limit for upward capillary rise
        return -gw_ksat_m_per_s, np.float32(0.0)

    return groundwater_flux_potential_m_per_s, dgroundwater_flux_potential_dS


@njit(cache=NUMBA_CACHE, fastmath=True, inline="always")
def fill_soil_below_water_table(
    water_content_m: np.ndarray,
    water_content_saturated_m: np.ndarray,
    soil_enthalpy_J_per_m2: np.ndarray,
    soil_layer_height: np.ndarray,
    groundwater_depth_m: np.float32,
    groundwater_temperature_C: np.float32,
    max_fill_m: np.float32,
) -> np.float32:
    """Fills the soil below the water table with groundwater.

    The part of a soil layer that lies below the water table is saturated: the layer
    holds at least that share of its saturated water content. Water that is missing
    is taken from the groundwater, with its heat. Water content and enthalpy are
    updated in place.

    Args:
        water_content_m: Water content per layer (m), updated in place.
        water_content_saturated_m: Saturated water content per layer (m).
        soil_enthalpy_J_per_m2: Enthalpy per layer (J/m2), updated in place.
        soil_layer_height: Thickness per layer (m).
        groundwater_depth_m: Depth of the water table below the surface (m).
        groundwater_temperature_C: Temperature of the groundwater (C).
        max_fill_m: Groundwater that can be taken (m).

    Returns:
        Groundwater taken up by the soil (m).
    """
    filled_m: np.float32 = np.float32(0.0)
    layer_bottom_m: np.float32 = np.float32(0.0)
    for i in range(N_SOIL_LAYERS):
        layer_bottom_m += soil_layer_height[i]
        submerged_fraction: np.float32 = min(
            max(
                (layer_bottom_m - groundwater_depth_m) / soil_layer_height[i],
                np.float32(0.0),
            ),
            np.float32(1.0),
        )
        fill_m: np.float32 = min(
            submerged_fraction * water_content_saturated_m[i] - water_content_m[i],
            max_fill_m - filled_m,
        )
        if fill_m > np.float32(0.0):
            water_content_m[i] += fill_m
            soil_enthalpy_J_per_m2[i] += (
                fill_m
                * RHO_WATER_KG_PER_M3
                * SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K
                * groundwater_temperature_C
            )
            filled_m += fill_m
    return filled_m


@njit(cache=NUMBA_CACHE, fastmath=True, inline="always")
def set_hillslope_interflow(
    water_content_m: ArrayFloat32,
    water_content_field_capacity_m: ArrayFloat32,
    water_content_saturated_m: ArrayFloat32,
    water_content_residual_m: ArrayFloat32,
    soil_layer_height_m: ArrayFloat32,
    horizontal_saturated_hydraulic_conductivity_m_per_s: ArrayFloat32,
    interflow_structural_conductivity_m_per_s: np.float32,
    interflow_structural_depth_m: np.float32,
    slope_m_per_m: np.float32,
    hillslope_length_m: np.float32,
    groundwater_depth_m: np.float32,
    liquid_fractions: ArrayFloat32,
    q_interflow: ArrayFloat32,
    dq_interflow_dS: ArrayFloat32,
) -> None:
    """Lateral outflow of the soil at the foot of the hillslope (kinematic storage model).

    Sloan, P. G., & Moore, I. D. (1984). Modeling subsurface stormflow on steeply
    sloping forested watersheds. Water Resources Research, 20(12), 1815-1822,
    doi:10.1029/WR020i012p01815. The soil above the rock is one store. Its saturated
    zone is a wedge that rests on the base and thickens linearly towards the foot of
    the hillslope, where it leaves the soil:

    - While the wedge stays inside the soil (their eq. 9) it is twice as thick at the
      foot as on average. With S the share of the drainable pores that hold perched
      water, the lower 2 S of the pore space is saturated at the foot.
    - Above half filling (S > 0.5) the wedge reaches the surface at the foot (their
      eq. 11-12): the whole soil is saturated there and the subsurface outflow cannot
      grow further. The additional water widens the part of the slope that is
      saturated to the surface (2 S - 1), where rain runs off (see
      calculate_hillslope_saturated_area_fraction).

    The outflow per unit width is the hydraulic gradient, sin(slope) (their eq. 6),
    times the transmissivity of the zone that is saturated at the foot. Sloan and
    Moore have one conductivity; here the lateral conductivity differs per layer (soil
    matrix plus the structure of the soil, which declines with depth), so the
    transmissivity is the sum of conductivity times saturated thickness over the
    layers, counted from the base up. When the wedge rises into the conductive upper
    layers at the foot of the slope the outflow increases steeply. For one
    conductivity K and soil thickness D this is their result, 2 K sin(slope) S D / L up
    to half filling and K sin(slope) D / L above it.

    The layer thicknesses are vertical and Sloan and Moore measure the thickness
    normal to the slope, a factor cos(slope); the hillslope length is horizontal, as
    the water depths are per unit of horizontal area. The slope term is therefore
    sin cos = s / (1 + s^2).

    The store (perched water and drainable pore space, see set_hillslope_store) is
    the one that also gives the saturated area for the infiltration. The outflow is
    taken from the layers in proportion to their perched water.

    Args:
        water_content_m: Water content per layer (m).
        water_content_field_capacity_m: Water content at field capacity per layer (m).
        water_content_saturated_m: Water content at saturation per layer (m).
        water_content_residual_m: Residual water content per layer (m).
        soil_layer_height_m: Thickness per layer (m).
        horizontal_saturated_hydraulic_conductivity_m_per_s: Lateral conductivity of
            the soil matrix per layer (m/s).
        interflow_structural_conductivity_m_per_s: Lateral conductivity of the soil
            structure at the surface (m/s).
        interflow_structural_depth_m: e-folding depth of the structural conductivity (m).
        slope_m_per_m: Slope of the hillslope (m/m).
        hillslope_length_m: Horizontal length of the hillslope (m).
        groundwater_depth_m: Depth of the water table below the surface (m).
        liquid_fractions: Unfrozen fraction of the water per layer (-).
        q_interflow: Output: lateral outflow per layer (m/s).
        dq_interflow_dS: Output: its derivative to the effective saturation of the
            layer (m/s), for the implicit solver.
    """
    perched_m = stack_empty(N_SOIL_LAYERS, np.float32)
    free_m = stack_empty(N_SOIL_LAYERS, np.float32)
    capacity_m = stack_empty(N_SOIL_LAYERS, np.float32)
    transmissivity_m2_per_s = stack_empty(N_SOIL_LAYERS, np.float32)

    total_perched_m, total_capacity_m = set_hillslope_store(
        water_content_m,
        water_content_field_capacity_m,
        water_content_saturated_m,
        soil_layer_height_m,
        groundwater_depth_m,
        perched_m,
        free_m,
        capacity_m,
    )
    layer_top_depth_m: np.float32 = np.float32(0.0)
    for i in range(N_SOIL_LAYERS):
        q_interflow[i] = np.float32(0.0)
        dq_interflow_dS[i] = np.float32(0.0)
        # transmissivity of the layer when it is saturated: lateral conductivity
        # (matrix plus structure at the middle of the layer) times its thickness
        # above the water table
        lateral_conductivity_m_per_s: np.float32 = (
            horizontal_saturated_hydraulic_conductivity_m_per_s[i]
            + interflow_structural_conductivity_m_per_s
            * np.exp(
                -(layer_top_depth_m + np.float32(0.5) * soil_layer_height_m[i])
                / interflow_structural_depth_m
            )
        )
        above_water_table: np.float32 = min(
            max(
                (groundwater_depth_m - layer_top_depth_m) / soil_layer_height_m[i],
                np.float32(0.0),
            ),
            np.float32(1.0),
        )
        transmissivity_m2_per_s[i] = (
            lateral_conductivity_m_per_s * above_water_table * soil_layer_height_m[i]
        )
        layer_top_depth_m += soil_layer_height_m[i]

    if total_perched_m <= np.float32(0.0) or total_capacity_m <= np.float32(0.0):
        return

    # drainable pore space that is saturated at the foot of the slope, from the base up
    saturated_at_foot_m: np.float32 = min(
        np.float32(2.0) * total_perched_m, total_capacity_m
    )
    transmissivity_at_foot_m2_per_s: np.float32 = np.float32(0.0)
    for i in range(N_SOIL_LAYERS - 1, -1, -1):
        if saturated_at_foot_m <= np.float32(0.0):
            break
        if capacity_m[i] <= np.float32(0.0):
            continue
        saturated_m: np.float32 = min(capacity_m[i], saturated_at_foot_m)
        transmissivity_at_foot_m2_per_s += (
            transmissivity_m2_per_s[i] * saturated_m / capacity_m[i]
        )
        saturated_at_foot_m -= saturated_m

    sine_times_cosine_of_slope: np.float32 = slope_m_per_m / (
        np.float32(1.0) + slope_m_per_m * slope_m_per_m
    )
    outflow_m_per_s: np.float32 = (
        sine_times_cosine_of_slope
        * transmissivity_at_foot_m2_per_s
        / hillslope_length_m
    )
    for i in range(N_SOIL_LAYERS):
        if perched_m[i] <= np.float32(0.0):
            continue
        q_interflow[i] = (
            outflow_m_per_s * perched_m[i] / total_perched_m * liquid_fractions[i]
        )
        # the outflow of a layer falls with its own free water
        dq_interflow_dS[i] = (
            q_interflow[i]
            / free_m[i]
            * (water_content_saturated_m[i] - water_content_residual_m[i])
        )


@njit(cache=NUMBA_CACHE, fastmath=True, inline="always")
def distribute_soil_water_ross(
    timestep_length_s: np.float32,
    water_content_m: np.ndarray,
    water_content_residual_m: np.ndarray,
    water_content_saturated_m: np.ndarray,
    water_content_field_capacity_m: np.ndarray,
    soil_enthalpy_J_per_m2: np.ndarray,
    solid_heat_capacity_J_per_m2_K: np.ndarray,
    vertical_saturated_hydraulic_conductivity_m_per_s: np.ndarray,
    horizontal_saturated_hydraulic_conductivity_m_per_s: np.ndarray,
    soil_layer_height: np.ndarray,
    bubbling_pressure_m_positive: np.ndarray,
    lambda_: np.ndarray,
    slope_m_per_m: np.float32,
    hillslope_length_m: np.float32,
    interflow_structural_conductivity_m_per_s: np.float32,
    green_ampt_active_layer_idx: np.int32,
    topwater_m: np.float32,
    gw_ksat_m_per_s: np.float32,
    groundwater_depth_m: np.float32,
    deep_soil_temperature_C: np.float32,
    interflow_structural_depth_m: np.float32,
    max_capillary_rise_m: np.float32 = np.float32(np.inf),
) -> tuple[
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
]:
    """One time-step update of the 6-layer soil moisture profile using the Ross (2003) scheme.

    Args:
        timestep_length_s: Timestep (seconds).
        water_content_m: Soil water content in each layer (m).
        water_content_residual_m: Residual soil water content in each layer (m).
        water_content_saturated_m: Saturated soil water content in each layer (m).
        water_content_field_capacity_m: Field capacity soil water content in each layer (m).
        soil_enthalpy_J_per_m2: Soil enthalpy in each layer (J/m2).
        solid_heat_capacity_J_per_m2_K: Heat capacity of the solid fraction in each layer (J/m2/K).
        vertical_saturated_hydraulic_conductivity_m_per_s: Saturated hydraulic conductivity for vertical redistribution (m/s).
        horizontal_saturated_hydraulic_conductivity_m_per_s: Saturated hydraulic conductivity for horizontal interflow (m/s).
        soil_layer_height: Height of each soil layer (m).
        bubbling_pressure_m_positive: Bubbling pressure in each layer (m). Must be positive.
        lambda_: van Genuchten n parameter minus one (lambda) (-).
        slope_m_per_m: Hillslope slope (m/m).
        hillslope_length_m: Hillslope length (m).
        interflow_structural_conductivity_m_per_s: Lateral conductivity of the soil
            structure at the surface (m/s), added to the matrix conductivity.
        interflow_structural_depth_m: e-folding depth (m) of the structural lateral
            conductivity.
        green_ampt_active_layer_idx: Index of wetting front layer.
        topwater_m: Topwater storage (m).
        gw_ksat_m_per_s: Saturated hydraulic conductivity of the groundwater toplayer (m/s).
        groundwater_depth_m: Depth of groundwater table below surface (m).
        deep_soil_temperature_C: Sub-surface temperature for advective heat of capillary rise (C).
        max_capillary_rise_m: Maximum available groundwater storage that can be drawn as capillary rise (meters).

    Returns:
        A tuple containing:
            - top_soil_percolation_to_layer_2_m (m).
            - top_soil_rise_from_layer_2_m (m).
            - percolation_to_groundwater_m (m).
            - capillary_rise_from_groundwater_m (m).
            - percolation_to_groundwater_enthalpy_loss_J_per_m2 (J/m2).
            - total_lateral_outflow_m (m).
            - total_interflow_enthalpy_loss_J_per_m2 (J/m2).
    """
    top_soil_percolation_to_layer_2_m = np.float32(0.0)
    top_soil_rise_from_layer_2_m = np.float32(0.0)

    groundwater_temperature_C: np.float32 = max(
        deep_soil_temperature_C, np.float32(0.0)
    )
    groundwater_fill_m: np.float32 = fill_soil_below_water_table(
        water_content_m=water_content_m,
        water_content_saturated_m=water_content_saturated_m,
        soil_enthalpy_J_per_m2=soil_enthalpy_J_per_m2,
        soil_layer_height=soil_layer_height,
        groundwater_depth_m=groundwater_depth_m,
        groundwater_temperature_C=groundwater_temperature_C,
        max_fill_m=max_capillary_rise_m,
    )
    max_capillary_rise_m = max_capillary_rise_m - groundwater_fill_m

    # Use stack-backed scratch arrays to avoid per-call heap/NRT allocation overhead.
    effective_saturation = stack_empty(N_SOIL_LAYERS, np.float32)
    liquid_fractions = stack_empty(N_SOIL_LAYERS, np.float32)
    K = stack_empty(N_SOIL_LAYERS, np.float32)
    phi = stack_empty(N_SOIL_LAYERS, np.float32)
    dK_dS = stack_empty(N_SOIL_LAYERS, np.float32)
    dphi_dS = stack_empty(N_SOIL_LAYERS, np.float32)

    q = stack_empty(_N_SOIL_LAYERS_PLUS_ONE, np.float32)
    q_interflow = stack_empty(N_SOIL_LAYERS, np.float32)
    dq_dS_i = stack_empty(_N_SOIL_LAYERS_PLUS_ONE, np.float32)
    dq_dS_iplus1 = stack_empty(_N_SOIL_LAYERS_PLUS_ONE, np.float32)
    dq_interflow_dS = stack_empty(N_SOIL_LAYERS, np.float32)

    b = stack_empty(N_SOIL_LAYERS, np.float32)
    d = stack_empty(N_SOIL_LAYERS, np.float32)
    inv_timestep_s: np.float32 = np.float32(1.0) / timestep_length_s
    cos_slope: np.float32 = np.float32(1.0) / np.sqrt(
        np.float32(1.0) + slope_m_per_m * slope_m_per_m
    )

    for i in range(N_SOIL_LAYERS):
        # Effective saturation and suction potential, limit to [0.001, 0.999] for numerical stability.
        available_capacity = water_content_saturated_m[i] - water_content_residual_m[i]
        s_eff = (water_content_m[i] - water_content_residual_m[i]) / available_capacity
        s_eff = max(np.float32(0.001), min(np.float32(0.999), s_eff))
        effective_saturation[i] = s_eff

        # Frozen fraction: only the latent-heat window matters; no temperature needed.
        topwater_frac_m = topwater_m if i == 0 else np.float32(0.0)
        water_depth_frac_m = water_content_m[i] + topwater_frac_m
        latent_heat_frac_J_per_m2 = (
            water_depth_frac_m * RHO_WATER_KG_PER_M3 * LATENT_HEAT_FUSION_J_PER_KG
        )
        h_frac = soil_enthalpy_J_per_m2[i]
        if h_frac >= np.float32(0.0):
            frozen_frac = np.float32(0.0)
        elif h_frac >= -latent_heat_frac_J_per_m2:
            frozen_frac = -h_frac / max(np.float32(1e-12), latent_heat_frac_J_per_m2)
        else:
            frozen_frac = np.float32(1.0)
        liq_frac = np.float32(1.0) - min(
            np.float32(1.0), max(np.float32(0.0), frozen_frac)
        )
        liquid_fractions[i] = liq_frac

        # When there is an active wetting front, the vertical flux is handled in the infiltration
        # module, so we set the vertical conductivity to near-zero here to effectively "turn off" the Richards
        # vertical redistribution for layers within or above the wetting front.
        sat_cond_s = vertical_saturated_hydraulic_conductivity_m_per_s[i]
        if green_ampt_active_layer_idx >= 0 and i < green_ampt_active_layer_idx:
            sat_cond_s = np.float32(1e-25)

        pore_size_index_i: np.float32 = np.float32(3.0) + (np.float32(2.0) / lambda_[i])

        # Campbell conductivity function
        # Campbell, G. S. (1974). A simple method for determining unsaturated conductivity
        # from moisture retention data. Soil science, 117(6), 311-314.
        s_eff_pow_pore_size = s_eff**pore_size_index_i
        K[i] = sat_cond_s * s_eff_pow_pore_size

        # Ross (2003) rewrites the Richards equation using the matric flux
        # potential, a Kirchhoff-style transform of hydraulic conductivity.
        # Ross, P.J. (2003), Modeling Soil Water and Solute Transport—Fast,
        # Simplified Numerical Solutions. Agron. J., 95: 1352-1361.
        # https://doi.org/10.2134/agronj2003.1352

        # The equation between 2 and 3 in Ross (2003) for matric flux potential phi is:
        # bubbling_pressure_m_positive is positive, in the paper it says:
        # h_e is pressure head at air entry (negative), so we add -1
        phi_e_val = calculate_saturated_matric_flux_potential(
            saturated_hydraulic_conductivity_m_per_s=sat_cond_s,
            bubbling_pressure_m_positive=bubbling_pressure_m_positive[i],
            lambda_=lambda_[i],
            pore_size_index=pore_size_index_i,
        )
        # Then equation 2 gives:
        phi_exponent_i = pore_size_index_i - (np.float32(1.0) / lambda_[i])
        phi[i] = phi_e_val * (s_eff**phi_exponent_i)

        # Derivatives of K and phi with respect to saturation, needed for the implicit solver.
        dK_dS[i] = pore_size_index_i * (K[i] / s_eff)
        dphi_dS[i] = phi_exponent_i * (phi[i] / s_eff)

    # Lateral outflow at the foot of the hillslope (interflow), from the soil as one store
    set_hillslope_interflow(
        water_content_m,
        water_content_field_capacity_m,
        water_content_saturated_m,
        water_content_residual_m,
        soil_layer_height,
        horizontal_saturated_hydraulic_conductivity_m_per_s,
        interflow_structural_conductivity_m_per_s,
        interflow_structural_depth_m,
        slope_m_per_m,
        hillslope_length_m,
        groundwater_depth_m,
        liquid_fractions,
        q_interflow,
        dq_interflow_dS,
    )

    # q holds vertical fluxes at layer interfaces:
    # q[0] is the top boundary (no imposed surface infiltration flux here),
    # q[1]..q[N-1] are internal interfaces, and q[N] is the groundwater boundary.
    # These interface fluxes and their saturation derivatives are assembled first,
    # then mapped into the tridiagonal matrix coefficients a, b, c, d below.
    q[0] = np.float32(
        0.0
    )  # No infiltration flux at the top boundary in this scheme. Surface fluxes are handled separately in the land surface model.

    for i in range(N_SOIL_LAYERS - 1):
        # Interface properties scaled by liquid fractions of BOTH adjacent layers.
        # This ensures that if layer i is frozen, it cannot pass water to i+1,
        # and if i+1 is frozen, it cannot pull water from i.
        liquid_fraction_of_interface = liquid_fractions[i] * liquid_fractions[i + 1]

        # Zero out the interface conductivity if either layer is essentially frozen,
        # or if the interface is within/above the active Green-Ampt wetting front
        # where vertical flux is governed explicitly by Green-Ampt infiltration.
        if (
            liquid_fractions[i] < 1e-4
            or liquid_fractions[i + 1] < 1e-4
            or (green_ampt_active_layer_idx >= 0 and i < green_ampt_active_layer_idx)
        ):
            liquid_fraction_of_interface = np.float32(0.0)

        # Compute the Ross (2003) interface blending weight from a dimensionless
        # capillary velocity ratio. This weight controls how strongly the gravity
        # conductivity term is taken from layer i versus layer i+1.
        # The model stores suction head as a negative pressure head, but this
        # formulation uses its magnitude as a capillary length scale.
        suction_head_magnitude_m = bubbling_pressure_m_positive[i] * (
            effective_saturation[i] ** (-(np.float32(1.0) / lambda_[i]))
        )
        # Dimensionless ratio (m/m): larger values push the Ross blend toward
        # a stronger upwind-like treatment of the conductivity term.
        # A 1e-3 m floor avoids singular behavior when suction head is tiny.
        # In equation 18 of Ross (2003), g is the cosine of the slope angle
        # projecting gravity normal to the hillslope surface.
        # On-the-fly interface distance between layer centers:
        # Computing (soil_layer_height[i] + soil_layer_height[i + 1]) * 0.5 in registers avoids allocating
        # a 25 MB 2D delta_z array in DRAM and eliminates 30 million array slice fetches per daily timestep.
        interface_dist: np.float32 = (
            soil_layer_height[i] + soil_layer_height[i + 1]
        ) * np.float32(0.5)

        v: np.float32 = min(
            (interface_dist * cos_slope)
            / max(np.float32(1e-3), suction_head_magnitude_m),
            np.float32(100.0),
        )

        pore_size_index_i = np.float32(3.0) + (np.float32(2.0) / lambda_[i])

        # Ross rational approximation for the conductivity blend weight.
        # See equation 18 in Ross (2003). This equation aims to avoid expensive
        # power functions.
        weight_numerator = (
            np.float32(60.0)
            + np.float32(10.0) * (np.float32(7.0) + lambda_[i] * pore_size_index_i) * v
            + (
                np.float32(16.0)
                + lambda_[i]
                * pore_size_index_i
                * (np.float32(5.0) + lambda_[i] * pore_size_index_i)
            )
            * (v**2)
        )
        weight_denominator = np.float32(2.0) * (
            np.float32(60.0)
            + np.float32(60.0) * v
            + (np.float32(11.0) + (lambda_[i] ** 2) * (pore_size_index_i**2)) * (v**2)
        )
        conductivity_blend_weight = max(
            np.float32(0.0), min(np.float32(1.0), weight_numerator / weight_denominator)
        )

        # Interface-local values: layer terms scaled by liquid_fraction_of_interface so
        # frozen interfaces naturally suppress both flux and Jacobian entries.
        conductivity_layer_i = K[i] * liquid_fraction_of_interface
        conductivity_layer_iplus1 = K[i + 1] * liquid_fraction_of_interface
        matric_flux_potential_layer_i = phi[i] * liquid_fraction_of_interface
        matric_flux_potential_layer_iplus1 = phi[i + 1] * liquid_fraction_of_interface

        dconductivity_dS_layer_i = dK_dS[i] * liquid_fraction_of_interface
        dconductivity_dS_layer_iplus1 = dK_dS[i + 1] * liquid_fraction_of_interface
        dmatric_flux_potential_dS_layer_i = dphi_dS[i] * liquid_fraction_of_interface
        dmatric_flux_potential_dS_layer_iplus1 = (
            dphi_dS[i + 1] * liquid_fraction_of_interface
        )

        # Interface flux q[i+1] has:
        # 1) a matric-flux-potential gradient term,
        # 2) a conductivity (gravity) term blended by Ross weight and projected
        #    normal to the hillslope by cos(slope).
        # The derivatives dq_dS_i and dq_dS_iplus1 are Jacobian terms that
        # populate the tridiagonal matrix entries for the implicit solve.
        gravity_flux = cos_slope * (
            conductivity_blend_weight * conductivity_layer_i
            + (np.float32(1.0) - conductivity_blend_weight) * conductivity_layer_iplus1
        )
        q[i + 1] = (
            matric_flux_potential_layer_i - matric_flux_potential_layer_iplus1
        ) / interface_dist + gravity_flux
        dq_dS_i[i + 1] = (dmatric_flux_potential_dS_layer_i / interface_dist) + (
            cos_slope * conductivity_blend_weight * dconductivity_dS_layer_i
        )
        dq_dS_iplus1[i + 1] = (
            -dmatric_flux_potential_dS_layer_iplus1 / interface_dist
        ) + (
            cos_slope
            * (np.float32(1.0) - conductivity_blend_weight)
            * dconductivity_dS_layer_iplus1
        )

    # Bottom boundary condition: Coupled groundwater interaction (percolation or capillary rise).
    total_soil_depth_m: np.float32 = np.float32(0.0)
    for k in range(N_SOIL_LAYERS):
        total_soil_depth_m += soil_layer_height[k]

    bottom_layer_index: int = N_SOIL_LAYERS - 1
    pore_size_index_bottom: np.float32 = np.float32(3.0) + (
        np.float32(2.0) / lambda_[bottom_layer_index]
    )

    # (outside the call: numba cannot compile inline conditionals in keyword arguments)
    max_capillary_flux_m_per_s: np.float32 = np.float32(np.inf)
    if np.isfinite(max_capillary_rise_m):
        max_capillary_flux_m_per_s = max_capillary_rise_m / timestep_length_s
    (
        q[N_SOIL_LAYERS],
        dq_dS_i[N_SOIL_LAYERS],
    ) = calculate_capillary_rise_and_percolation_parameters(
        groundwater_depth_m=groundwater_depth_m,
        total_soil_depth_m=total_soil_depth_m,
        bottom_layer_thickness_m=soil_layer_height[bottom_layer_index],
        bubbling_pressure_m_positive=bubbling_pressure_m_positive[bottom_layer_index],
        lambda_=lambda_[bottom_layer_index],
        pore_size_index=pore_size_index_bottom,
        saturated_hydraulic_conductivity_m_per_s=vertical_saturated_hydraulic_conductivity_m_per_s[
            bottom_layer_index
        ],
        matric_flux_potential_bottom=phi[bottom_layer_index],
        conductivity_bottom=K[bottom_layer_index],
        dmatric_flux_potential_dS_bottom=dphi_dS[bottom_layer_index],
        dconductivity_dS_bottom=dK_dS[bottom_layer_index],
        liquid_fraction=liquid_fractions[bottom_layer_index],
        gw_ksat_m_per_s=gw_ksat_m_per_s,
        max_capillary_flux_m_per_s=max_capillary_flux_m_per_s,
        cos_slope=cos_slope,
    )

    dq_dS_i[0] = np.float32(0.0)  # No flux at top, so no saturation dependence.
    dq_dS_iplus1[0] = np.float32(0.0)  # No flux at top, so no saturation dependence.

    # Build tridiagonal system A dS = d for implicit saturation increments.
    # a[i]: coupling to layer above dS[i-1] (sub-diagonal in matrix notation)
    # b[i]: local storage + local Jacobian terms (diagonal)
    # c[i]: coupling to layer below dS[i+1] (super-diagonal in matrix notation)
    # d[i]: local imbalance (net inflow - outflow) to be corrected by dS.
    for i in range(N_SOIL_LAYERS):
        # The storage term represents how a specific amount of water content change (dS)
        # translates to a change in water content (dW),
        # We can use the water content here, because dz is constant.
        # dW/dS = (W_sat - W_res), which is equivalent to dz * (theta_sat - theta_res).
        storage_term = (
            water_content_saturated_m[i] - water_content_residual_m[i]
        ) * inv_timestep_s

        # Matrix coefficients for the implicit Backward Euler step.
        # Include dq_interflow_dS in the diagonal element b[i]
        b[i] = -dq_dS_iplus1[i] + dq_dS_i[i + 1] + dq_interflow_dS[i] + storage_term

        # The imbalance term accounts for the difference between the linear approximation
        # of the S-transformation and the actual source terms.
        # Include q_interflow[i] in the imbalance (it's an outflow)
        d[i] = q[i] - q[i + 1] - q_interflow[i]

    # Thomas algorithm to solve the tridiagonal system Ax=d for saturation change dS:
    # Memory optimization:
    # - Subdiagonal a[i] is read directly from -dq_dS_i[i] (Jacobian of flux from layer above).
    # - Superdiagonal c[i] is read directly from dq_dS_iplus1[i + 1] (Jacobian of flux to layer below).
    # - RHS vector d is reused to store the forward substitution d' and is overwritten in-place with the solution dS.
    # This eliminates 3 scratch stack arrays (a, c, dS) with 100% bit-for-bit equivalence.
    # Forward pass:
    for i in range(1, N_SOIL_LAYERS):
        elimination_factor: np.float32 = -dq_dS_i[i] / b[i - 1]
        b[i] = b[i] - elimination_factor * dq_dS_iplus1[i]
        d[i] = d[i] - elimination_factor * d[i - 1]

    # Backward substitution writing solution dS directly into d:
    d[N_SOIL_LAYERS - 1] = d[N_SOIL_LAYERS - 1] / b[N_SOIL_LAYERS - 1]
    for i in range(N_SOIL_LAYERS - 2, -1, -1):
        d[i] = (d[i] - dq_dS_iplus1[i + 1] * d[i + 1]) / b[i]

    # Total interflow tracking
    total_lateral_outflow_m = np.float32(0.0)
    total_interflow_enthalpy_loss_J_per_m2 = np.float32(0.0)

    # Apply final interface fluxes to redistribute water and transport heat between layers.
    # For each internal interface, compute the actual water flux (capped by available water),
    # update layer water contents, and advect enthalpy with the moving water.
    for i in range(N_SOIL_LAYERS - 1):
        corrected_q_interface_m_per_s = q[i + 1] + (
            dq_dS_i[i + 1] * d[i] + dq_dS_iplus1[i + 1] * d[i + 1]
        )
        flux_m: np.float32 = corrected_q_interface_m_per_s * timestep_length_s

        # Interface flux: positive is downward (i -> i+1), negative is upward (i+1 -> i)
        # We cap the flux by the available water in the source layer.
        # This is a safety check for the linear approximation.
        available_layer_i: np.float32 = max(
            np.float32(0.0), water_content_m[i] - water_content_residual_m[i]
        )
        available_layer_iplus1: np.float32 = max(
            np.float32(0.0),
            water_content_m[i + 1] - water_content_residual_m[i + 1],
        )

        # Flux is limited by available capacity in the sink layer
        capacity_layer_i: np.float32 = max(
            np.float32(0.0),
            water_content_saturated_m[i] - water_content_m[i],
        )
        capacity_layer_iplus1: np.float32 = max(
            np.float32(0.0),
            water_content_saturated_m[i + 1] - water_content_m[i + 1],
        )

        # flux_m > 0: source is i, sink is i+1, cap is available_layer_i
        # flux_m < 0: source is i+1, sink is i, cap is available_layer_i+1
        actual_flux_m: np.float32 = max(
            -min(available_layer_iplus1, capacity_layer_i),
            min(flux_m, min(available_layer_i, capacity_layer_iplus1)),
        )

        # Track percolation/rise for the top interface (between layer 0 and 1)
        if i == 0:
            top_soil_percolation_to_layer_2_m = max(np.float32(0.0), actual_flux_m)
            top_soil_rise_from_layer_2_m = max(np.float32(0.0), -actual_flux_m)

        water_content_m[i] = water_content_m[i] - actual_flux_m
        water_content_m[i] = max(water_content_m[i], water_content_residual_m[i])
        water_content_m[i + 1] = water_content_m[i + 1] + actual_flux_m
        water_content_m[i + 1] = min(
            water_content_m[i + 1], water_content_saturated_m[i + 1]
        )

        # Advective heat transport
        # We need the temperature of the source layer.
        source_idx = i if actual_flux_m >= 0 else i + 1
        source_topwater_m: np.float32 = (
            topwater_m if source_idx == 0 else np.float32(0.0)
        )

        src_water_depth_m = water_content_m[source_idx] + source_topwater_m
        src_latent_heat_J_per_m2 = (
            src_water_depth_m * RHO_WATER_KG_PER_M3 * LATENT_HEAT_FUSION_J_PER_KG
        )
        src_heat_capacity_liquid_J_per_m2_K = (
            solid_heat_capacity_J_per_m2_K[source_idx]
            + src_water_depth_m
            * RHO_WATER_KG_PER_M3
            * SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K
        )
        src_heat_capacity_frozen_J_per_m2_K = (
            solid_heat_capacity_J_per_m2_K[source_idx]
            + src_water_depth_m
            * RHO_WATER_KG_PER_M3
            * SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K
        )
        src_enthalpy_J_per_m2 = soil_enthalpy_J_per_m2[source_idx]
        if src_enthalpy_J_per_m2 >= np.float32(0.0):
            source_temperature_C = src_enthalpy_J_per_m2 / max(
                np.float32(1e-12), src_heat_capacity_liquid_J_per_m2_K
            )
        elif src_enthalpy_J_per_m2 >= -src_latent_heat_J_per_m2:
            source_temperature_C = np.float32(0.0)
        else:
            source_temperature_C = (
                src_enthalpy_J_per_m2 + src_latent_heat_J_per_m2
            ) / max(np.float32(1e-12), src_heat_capacity_frozen_J_per_m2_K)

        advected_temp_C = max(source_temperature_C, np.float32(0.0))
        energy_transfer_J_per_m2 = (
            actual_flux_m
            * RHO_WATER_KG_PER_M3
            * SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K
            * advected_temp_C
        )

        soil_enthalpy_J_per_m2[i] = soil_enthalpy_J_per_m2[i] - energy_transfer_J_per_m2
        soil_enthalpy_J_per_m2[i + 1] = (
            soil_enthalpy_J_per_m2[i + 1] + energy_transfer_J_per_m2
        )

    # Percolation to groundwater or capillary rise
    bottom_layer: int = (
        N_SOIL_LAYERS - 1
    )  # lowest soil layer is the interface with groundwater
    linearized_gw_flux_m_per_s: np.float32 = q[N_SOIL_LAYERS] + (
        dq_dS_i[N_SOIL_LAYERS] * d[bottom_layer]
    )
    # Downward gravity drainage cannot turn into upward capillary rise through Taylor linearization
    if q[N_SOIL_LAYERS] >= np.float32(0.0):
        corrected_groundwater_flux_m_per_s: np.float32 = max(
            np.float32(0.0), linearized_gw_flux_m_per_s
        )
    else:
        corrected_groundwater_flux_m_per_s = linearized_gw_flux_m_per_s
    # A soil whose base lies in the groundwater cannot drain into it.
    if groundwater_depth_m < total_soil_depth_m:
        corrected_groundwater_flux_m_per_s = min(
            corrected_groundwater_flux_m_per_s, np.float32(0.0)
        )

    groundwater_boundary_flux_m: np.float32 = (
        corrected_groundwater_flux_m_per_s * timestep_length_s
    )

    if groundwater_boundary_flux_m >= np.float32(0.0):
        # Downward percolation to groundwater
        available_water_source: np.float32 = max(
            np.float32(0.0),
            (water_content_m[bottom_layer] - water_content_residual_m[bottom_layer]),
        )
        percolation_to_groundwater_m = min(
            groundwater_boundary_flux_m, available_water_source
        )
        capillary_rise_from_groundwater_m: np.float32 = np.float32(0.0)

        water_content_m[bottom_layer] -= percolation_to_groundwater_m
        water_content_m[bottom_layer] = max(
            water_content_m[bottom_layer], water_content_residual_m[bottom_layer]
        )
    else:
        # Upward capillary rise from groundwater
        remaining_capacity: np.float32 = max(
            np.float32(0.0),
            (water_content_saturated_m[bottom_layer] - water_content_m[bottom_layer]),
        )
        capillary_rise_from_groundwater_m = min(
            -groundwater_boundary_flux_m, remaining_capacity, max_capillary_rise_m
        )
        percolation_to_groundwater_m = np.float32(0.0)

        water_content_m[bottom_layer] += capillary_rise_from_groundwater_m
        water_content_m[bottom_layer] = min(
            water_content_m[bottom_layer], water_content_saturated_m[bottom_layer]
        )

    # Apply interflow changes to water content and enthalpy.
    # Since interflow is in the matrix, it's already accounted for in dS,
    # but we need to get it separately for to be able to return the total
    # interflow, and to apply the enthalpy loss from interflow advection.
    interflow_temp_C: np.float32 = np.float32(0.0)
    for i in range(N_SOIL_LAYERS):
        corrected_q_interflow_m_per_s = q_interflow[i] + (dq_interflow_dS[i] * d[i])
        available_for_interflow_m = max(
            np.float32(0.0),
            water_content_m[i] - water_content_field_capacity_m[i],
        )

        actual_interflow_m = max(
            np.float32(0.0),
            min(
                corrected_q_interflow_m_per_s * timestep_length_s,
                available_for_interflow_m,
            ),
        )

        # Update water content
        water_content_m[i] -= actual_interflow_m
        total_lateral_outflow_m += actual_interflow_m

        # Exact piecewise enthalpy-temperature relation with 0°C phase-change
        # plateau, written inline to avoid a separate helper call here.
        topwater_layer_m = topwater_m if i == 0 else np.float32(0.0)
        water_depth_m = water_content_m[i] + topwater_layer_m
        latent_heat_areal_J_per_m2 = (
            water_depth_m * RHO_WATER_KG_PER_M3 * LATENT_HEAT_FUSION_J_PER_KG
        )
        heat_capacity_liquid_J_per_m2_K = (
            solid_heat_capacity_J_per_m2_K[i]
            + water_depth_m
            * RHO_WATER_KG_PER_M3
            * SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K
        )
        heat_capacity_frozen_J_per_m2_K = (
            solid_heat_capacity_J_per_m2_K[i]
            + water_depth_m
            * RHO_WATER_KG_PER_M3
            * SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K
        )
        interflow_enthalpy_J_per_m2 = soil_enthalpy_J_per_m2[i]
        if interflow_enthalpy_J_per_m2 >= np.float32(0.0):
            interflow_temp_C = interflow_enthalpy_J_per_m2 / max(
                np.float32(1e-12), heat_capacity_liquid_J_per_m2_K
            )
        elif interflow_enthalpy_J_per_m2 >= -latent_heat_areal_J_per_m2:
            interflow_temp_C = np.float32(0.0)
        else:
            interflow_temp_C = (
                interflow_enthalpy_J_per_m2 + latent_heat_areal_J_per_m2
            ) / max(np.float32(1e-12), heat_capacity_frozen_J_per_m2_K)

        interflow_enthalpy_loss = (
            actual_interflow_m
            * RHO_WATER_KG_PER_M3
            * SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K
            * interflow_temp_C
        )

        soil_enthalpy_J_per_m2[i] -= interflow_enthalpy_loss
        total_interflow_enthalpy_loss_J_per_m2 += interflow_enthalpy_loss

    bottom_layer_temp_C = interflow_temp_C

    if percolation_to_groundwater_m > np.float32(0.0):
        percolation_to_groundwater_enthalpy_loss_J_per_m2 = (
            percolation_to_groundwater_m
            * RHO_WATER_KG_PER_M3
            * SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K
            * max(bottom_layer_temp_C, np.float32(0.0))
        )
        soil_enthalpy_J_per_m2[bottom_layer] -= (
            percolation_to_groundwater_enthalpy_loss_J_per_m2
        )
    elif capillary_rise_from_groundwater_m > np.float32(0.0):
        groundwater_source_temperature_C: np.float32 = max(
            deep_soil_temperature_C, np.float32(0.0)
        )
        capillary_rise_enthalpy_gain_J_per_m2: np.float32 = (
            capillary_rise_from_groundwater_m
            * RHO_WATER_KG_PER_M3
            * SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K
            * groundwater_source_temperature_C
        )
        soil_enthalpy_J_per_m2[bottom_layer] += capillary_rise_enthalpy_gain_J_per_m2
        # Net loss to groundwater is negative of the heat influx
        percolation_to_groundwater_enthalpy_loss_J_per_m2 = (
            -capillary_rise_enthalpy_gain_J_per_m2
        )
    else:
        percolation_to_groundwater_enthalpy_loss_J_per_m2 = np.float32(0.0)

    # the groundwater that filled the soil below the water table left the aquifer
    capillary_rise_from_groundwater_m += groundwater_fill_m
    percolation_to_groundwater_enthalpy_loss_J_per_m2 -= (
        groundwater_fill_m
        * RHO_WATER_KG_PER_M3
        * SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K
        * groundwater_temperature_C
    )

    return (
        top_soil_percolation_to_layer_2_m,
        top_soil_rise_from_layer_2_m,
        percolation_to_groundwater_m,
        capillary_rise_from_groundwater_m,
        percolation_to_groundwater_enthalpy_loss_J_per_m2,
        total_lateral_outflow_m,
        total_interflow_enthalpy_loss_J_per_m2,
    )
