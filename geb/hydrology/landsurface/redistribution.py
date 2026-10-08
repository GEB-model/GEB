"""Darcy flow redistribution for soil water."""

import numpy as np
from numba import njit

from geb.workflows.numba_stack_array import stack_empty

from .constants import (
    LATENT_HEAT_FUSION_J_PER_KG,
    N_SOIL_LAYERS,
    RHO_WATER_KG_PER_M3,
    SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K,
    SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K,
)

_N_SOIL_LAYERS_PLUS_ONE: int = N_SOIL_LAYERS + 1


@njit(cache=True, fastmath=True, inline="always")
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


@njit(cache=True, fastmath=True, inline="always")
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
    # - conductivity_bottom (>= 0): downward drainage from gravity
    groundwater_flux_potential_m_per_s = (
        groundwater_capillary_flux_m_per_s + conductivity_bottom
    )
    dgroundwater_flux_potential_dS = (
        dgroundwater_capillary_flux_dS + dconductivity_dS_bottom
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


@njit(cache=True, fastmath=True, inline="always")
def distribute_soil_water_ross(
    timestep_length_s: np.float32,
    water_content_m: np.ndarray,
    water_content_residual_m: np.ndarray,
    water_content_saturated_m: np.ndarray,
    water_content_field_capacity_m: np.ndarray,
    soil_enthalpy_J_per_m2: np.ndarray,
    solid_heat_capacity_J_per_m2_K: np.ndarray,
    saturated_hydraulic_conductivity_m_per_s: np.ndarray,
    interface_dist_m: np.ndarray,
    soil_layer_height: np.ndarray,
    bubbling_pressure_m_positive: np.ndarray,
    lambda_: np.ndarray,
    pore_size_index: np.ndarray,
    slope_m_per_m: np.float32,
    hillslope_length_m: np.float32,
    interflow_multiplier: np.float32,
    green_ampt_active_layer_idx: np.int32,
    topwater_m: np.float32,
    gw_ksat_m_per_s: np.float32,
    groundwater_depth_m: np.float32,
    deep_soil_temperature_C: np.float32,
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
        saturated_hydraulic_conductivity_m_per_s: Saturated hydraulic conductivity for each layer (m/s).
        interface_dist_m: Distance between centers of soil layers (m).
        soil_layer_height: Height of each soil layer (m).
        bubbling_pressure_m_positive: Bubbling pressure in each layer (m). Must be positive.
        lambda_: van Genuchten n parameter minus one (lambda) (-).
        pore_size_index: Pore size index (3 + 2/lambda) (-).
        slope_m_per_m: Hillslope slope (m/m).
        hillslope_length_m: Hillslope length (m).
        interflow_multiplier: Calibration multiplier for interflow (-).
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

    a = stack_empty(N_SOIL_LAYERS, np.float32)
    b = stack_empty(N_SOIL_LAYERS, np.float32)
    c = stack_empty(N_SOIL_LAYERS, np.float32)
    d = stack_empty(N_SOIL_LAYERS, np.float32)
    dS = stack_empty(N_SOIL_LAYERS, np.float32)
    inv_timestep_s: np.float32 = np.float32(1.0) / timestep_length_s

    for i in range(N_SOIL_LAYERS):
        # Effective saturation and suction potential, limit to [0.001, 0.999] for numerical stability.
        s_eff = (water_content_m[i] - water_content_residual_m[i]) / (
            water_content_saturated_m[i] - water_content_residual_m[i]
        )
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
        sat_cond_s = saturated_hydraulic_conductivity_m_per_s[i]
        if green_ampt_active_layer_idx >= 0 and i < green_ampt_active_layer_idx:
            sat_cond_s = np.float32(1e-25)

        # Campbell conductivity function
        # Campbell, G. S. (1974). A simple method for determining unsaturated conductivity
        # from moisture retention data. Soil science, 117(6), 311-314.
        K[i] = sat_cond_s * s_eff ** pore_size_index[i]

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
            pore_size_index=pore_size_index[i],
        )
        # Then equation 2 gives:
        phi_exponent_i = pore_size_index[i] - (np.float32(1.0) / lambda_[i])
        phi[i] = phi_e_val * (s_eff**phi_exponent_i)

        # Derivatives of K and phi with respect to saturation, needed for the implicit solver.
        dK_dS[i] = pore_size_index[i] * (K[i] / s_eff)
        dphi_dS[i] = phi_exponent_i * (phi[i] / s_eff)

        # Check whether this layer has an active wetting front.
        in_active_wetting_front: bool = bool(
            green_ampt_active_layer_idx >= 0 and i <= green_ampt_active_layer_idx
        )

        if in_active_wetting_front:
            # Active wetting front (storm quickflow):
            # Water in the perched transmission zone drains laterally via saturated lateral conductivity
            # Lateral flow operates on drainable water above field capacity.
            free_water_m = max(
                np.float32(0.0),
                water_content_m[i] - water_content_field_capacity_m[i],
            )
            drainable_porosity = max(
                np.float32(1e-6),
                (water_content_saturated_m[i] - water_content_field_capacity_m[i])
                / soil_layer_height[i],
            )
            lateral_K_m_per_s = (
                saturated_hydraulic_conductivity_m_per_s[i]
                * np.float32(10.0)
                * interflow_multiplier
            )
            interflow_drainage_rate_s = (lateral_K_m_per_s * slope_m_per_m) / (
                drainable_porosity * hillslope_length_m
            )
            q_interflow[i] = (
                interflow_drainage_rate_s * free_water_m * liquid_fractions[i]
            )

            # Saturated lateral conductivity does not depend on S; derivative is purely dW/dS:
            if free_water_m > np.float32(0.0):
                dq_interflow_dS[i] = (
                    interflow_drainage_rate_s
                    * (water_content_saturated_m[i] - water_content_residual_m[i])
                    * liquid_fractions[i]
                )
            else:
                dq_interflow_dS[i] = np.float32(0.0)
        else:
            # Inactive wetting front:
            # Drains more slowly via unsaturated matrix conductivity down to residual storage.
            free_water_m = max(
                np.float32(0.0),
                water_content_m[i] - water_content_residual_m[i],
            )
            drainable_porosity = max(
                np.float32(1e-6),
                (water_content_saturated_m[i] - water_content_residual_m[i])
                / soil_layer_height[i],
            )
            lateral_K_m_per_s = K[i] * np.float32(10.0) * interflow_multiplier
            interflow_drainage_rate_s = (lateral_K_m_per_s * slope_m_per_m) / (
                drainable_porosity * hillslope_length_m
            )
            q_interflow[i] = (
                interflow_drainage_rate_s * free_water_m * liquid_fractions[i]
            )

            d_lateral_K_dS = dK_dS[i] * np.float32(10.0)
            d_interflow_drainage_rate_dS = (
                (d_lateral_K_dS * slope_m_per_m)
                / (drainable_porosity * hillslope_length_m)
            ) * interflow_multiplier

            dq_interflow_dS[i] = (
                d_interflow_drainage_rate_dS * free_water_m
                + interflow_drainage_rate_s
                * (water_content_saturated_m[i] - water_content_residual_m[i])
            ) * liquid_fractions[i]

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
        # Note that in equation 18 in Ross (2003), g is also included. However,
        # g is NOT the gravity but is the cosine of the slope angle, which we assume
        # to be 1.0.
        v: np.float32 = min(
            interface_dist_m[i] / max(np.float32(1e-3), suction_head_magnitude_m),
            np.float32(100.0),
        )

        # Ross rational approximation for the conductivity blend weight.
        # See equation 18 in Ross (2003). This equation aims to avoid expensive
        # power functions.
        weight_numerator = (
            np.float32(60.0)
            + np.float32(10.0) * (np.float32(7.0) + lambda_[i] * pore_size_index[i]) * v
            + (
                np.float32(16.0)
                + lambda_[i]
                * pore_size_index[i]
                * (np.float32(5.0) + lambda_[i] * pore_size_index[i])
            )
            * (v**2)
        )
        weight_denominator = np.float32(2.0) * (
            np.float32(60.0)
            + np.float32(60.0) * v
            + (np.float32(11.0) + (lambda_[i] ** 2) * (pore_size_index[i] ** 2))
            * (v**2)
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
        # 2) a conductivity (gravity) term blended by Ross weight.
        # The derivatives dq_dS_i and dq_dS_iplus1 are Jacobian terms that
        # populate the tridiagonal matrix entries for the implicit solve.
        q[i + 1] = (
            matric_flux_potential_layer_i - matric_flux_potential_layer_iplus1
        ) / interface_dist_m[i] + (
            conductivity_blend_weight * conductivity_layer_i
            + (np.float32(1.0) - conductivity_blend_weight) * conductivity_layer_iplus1
        )
        dq_dS_i[i + 1] = (dmatric_flux_potential_dS_layer_i / interface_dist_m[i]) + (
            conductivity_blend_weight * dconductivity_dS_layer_i
        )
        dq_dS_iplus1[i + 1] = (
            -dmatric_flux_potential_dS_layer_iplus1 / interface_dist_m[i]
        ) + (
            (np.float32(1.0) - conductivity_blend_weight)
            * dconductivity_dS_layer_iplus1
        )

    # Bottom boundary condition: Coupled groundwater interaction (percolation or capillary rise).
    total_soil_depth_m: np.float32 = np.float32(0.0)
    for k in range(N_SOIL_LAYERS):
        total_soil_depth_m += soil_layer_height[k]

    bottom_layer_index: int = N_SOIL_LAYERS - 1

    (
        q[N_SOIL_LAYERS],
        dq_dS_i[N_SOIL_LAYERS],
    ) = calculate_capillary_rise_and_percolation_parameters(
        groundwater_depth_m=groundwater_depth_m,
        total_soil_depth_m=total_soil_depth_m,
        bottom_layer_thickness_m=soil_layer_height[bottom_layer_index],
        bubbling_pressure_m_positive=bubbling_pressure_m_positive[bottom_layer_index],
        lambda_=lambda_[bottom_layer_index],
        pore_size_index=pore_size_index[bottom_layer_index],
        saturated_hydraulic_conductivity_m_per_s=saturated_hydraulic_conductivity_m_per_s[
            bottom_layer_index
        ],
        matric_flux_potential_bottom=phi[bottom_layer_index],
        conductivity_bottom=K[bottom_layer_index],
        dmatric_flux_potential_dS_bottom=dphi_dS[bottom_layer_index],
        dconductivity_dS_bottom=dK_dS[bottom_layer_index],
        liquid_fraction=liquid_fractions[bottom_layer_index],
        gw_ksat_m_per_s=gw_ksat_m_per_s,
        max_capillary_flux_m_per_s=(
            max_capillary_rise_m / timestep_length_s
            if np.isfinite(max_capillary_rise_m)
            else np.float32(np.inf)
        ),
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
        a[i] = -dq_dS_i[i]  # Jacobian of flux from layer above (naturally 0 at top)
        c[i] = dq_dS_iplus1[i + 1]  # Jacobian of flux to layer below
        b[i] = -dq_dS_iplus1[i] + dq_dS_i[i + 1] + dq_interflow_dS[i] + storage_term

        # The imbalance term accounts for the difference between the linear approximation
        # of the S-transformation and the actual source terms.
        # Include q_interflow[i] in the imbalance (it's an outflow)
        d[i] = q[i] - q[i + 1] - q_interflow[i]

    # Thomas algorithm to solve the tridiagonal system Ax=d for dS

    # Forward pass:
    for i in range(1, N_SOIL_LAYERS):
        m = a[i] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    # Backward substitution:
    dS[N_SOIL_LAYERS - 1] = d[N_SOIL_LAYERS - 1] / b[N_SOIL_LAYERS - 1]
    for i in range(N_SOIL_LAYERS - 2, -1, -1):
        dS[i] = (d[i] - c[i] * dS[i + 1]) / b[i]

    # Total interflow tracking
    total_lateral_outflow_m = np.float32(0.0)
    total_interflow_enthalpy_loss_J_per_m2 = np.float32(0.0)

    # Apply final interface fluxes to redistribute water and transport heat between layers.
    # For each internal interface, compute the actual water flux (capped by available water),
    # update layer water contents, and advect enthalpy with the moving water.
    for i in range(N_SOIL_LAYERS - 1):
        corrected_q_interface_m_per_s = q[i + 1] + (
            dq_dS_i[i + 1] * dS[i] + dq_dS_iplus1[i + 1] * dS[i + 1]
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
    corrected_groundwater_flux_m_per_s: np.float32 = q[N_SOIL_LAYERS] + (
        dq_dS_i[N_SOIL_LAYERS] * dS[bottom_layer]
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
    for i in range(N_SOIL_LAYERS):
        corrected_q_interflow_m_per_s = q_interflow[i] + (dq_interflow_dS[i] * dS[i])
        in_active_wetting_front_apply: bool = bool(
            green_ampt_active_layer_idx >= 0 and i <= green_ampt_active_layer_idx
        )
        if in_active_wetting_front_apply:
            available_for_interflow_m = max(
                np.float32(0.0),
                water_content_m[i] - water_content_field_capacity_m[i],
            )
        else:
            available_for_interflow_m = max(
                np.float32(0.0),
                water_content_m[i] - water_content_residual_m[i],
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

    return (
        top_soil_percolation_to_layer_2_m,
        top_soil_rise_from_layer_2_m,
        percolation_to_groundwater_m,
        capillary_rise_from_groundwater_m,
        percolation_to_groundwater_enthalpy_loss_J_per_m2,
        total_lateral_outflow_m,
        total_interflow_enthalpy_loss_J_per_m2,
    )
