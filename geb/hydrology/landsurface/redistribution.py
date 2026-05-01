"""Darcy flow redistribution for soil water."""

import numpy as np
from numba import njit

from .constants import (
    RHO_WATER_KG_PER_M3,
    SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K,
)
from .energy import get_temperature_and_frozen_fraction_from_enthalpy_scalar


@njit(cache=True, fastmath=True)
def distribute_soil_water_ross(
    dt_s: np.float32,
    water_content_m: np.ndarray,
    water_content_residual_m: np.ndarray,
    water_content_saturated_m: np.ndarray,
    soil_enthalpy_J_per_m2: np.ndarray,
    solid_heat_capacity_J_per_m2_K: np.ndarray,
    saturated_hydraulic_conductivity_m_per_hour: np.ndarray,
    soil_layer_height_m: np.ndarray,
    alpha_vg: np.ndarray,
    n_vg: np.ndarray,
    green_ampt_active_layer_idx: np.int32,
    topwater_m: np.float32,
) -> tuple[np.ndarray, np.ndarray, np.float32, np.float32]:
    """One time-step update of the 6-layer soil moisture profile using the Ross (2003) scheme.

    Args:
        dt_s: Timestep (seconds).
        water_content_m: Soil water content in each layer (m).
        water_content_residual_m: Residual soil water content in each layer (m).
        water_content_saturated_m: Saturated soil water content in each layer (m).
        soil_enthalpy_J_per_m2: Soil enthalpy in each layer (J/m2).
        solid_heat_capacity_J_per_m2_K: Heat capacity of the solid fraction in each layer (J/m2/K).
        saturated_hydraulic_conductivity_m_per_hour: Saturated hydraulic conductivity for each layer (m/hour).
        soil_layer_height_m: Thickness of each soil layer (m).
        alpha_vg: van Genuchten alpha parameter (1/m).
        n_vg: van Genuchten n parameter (-).
        green_ampt_active_layer_idx: Index of wetting front layer.
        topwater_m: Topwater storage (m).

    Returns:
        A tuple containing:
            - Updated water_content_m (m).
            - Updated soil_enthalpy_J_per_m2 (J/m2).
            - top_soil_percolation_to_layer_2_m (m).
            - top_soil_rise_from_layer_2_m (m).
    """
    n_layers = len(water_content_m)
    top_soil_percolation_to_layer_2_m = np.float32(0.0)
    top_soil_rise_from_layer_2_m = np.float32(0.0)

    theta_s = water_content_saturated_m / soil_layer_height_m
    theta_r = water_content_residual_m / soil_layer_height_m
    saturated_hydraulic_conductivity_m_per_s = (
        saturated_hydraulic_conductivity_m_per_hour.copy() / np.float32(3600.0)
    )

    # Set hydraulic conductivity to ~0 for layers above the wetting front
    # to prevent tridiagonal solver redistribution during infiltration.
    for i in range(n_layers):
        if i < green_ampt_active_layer_idx:
            saturated_hydraulic_conductivity_m_per_s[i] = np.float32(1e-25)

    n_minus_one = n_vg - np.float32(1.0)
    pore_size_index = np.float32(3.0) + (np.float32(2.0) / n_minus_one)
    bubbling_pressure_m = -np.float32(1.0) / alpha_vg

    theta = water_content_m / soil_layer_height_m
    effective_saturation = np.zeros(n_layers, dtype=np.float32)
    for i in range(n_layers):
        s_val = (theta[i] - theta_r[i]) / (theta_s[i] - theta_r[i])
        effective_saturation[i] = max(np.float32(0.01), min(np.float32(1.0), s_val))

    phi_e = (saturated_hydraulic_conductivity_m_per_s * bubbling_pressure_m) / (
        np.float32(1.0) - n_minus_one * pore_size_index
    )

    # Scale hydraulic parameters by liquid fraction (frozen soil limit)
    # We must ensure that if BOTH layers involved in a flux are frozen,
    # the conductivity at the interface is zero.
    # We first calculate liquid_fractions for all layers.
    liquid_fractions = np.zeros(n_layers, dtype=np.float32)
    for i in range(n_layers):
        ti_c, frozen_frac = get_temperature_and_frozen_fraction_from_enthalpy_scalar(
            soil_enthalpy_J_per_m2[i],
            solid_heat_capacity_J_per_m2_K[i],
            water_content_m[i],
            topwater_m if i == 0 else np.float32(0.0),
        )
        liquid_fractions[i] = np.float32(1.0) - np.minimum(
            np.maximum(frozen_frac, np.float32(0.0)), np.float32(1.0)
        )
        # If the layer is above the wetting front, it acts as if frozen (no flux).
        if i < green_ampt_active_layer_idx:
            liquid_fractions[i] = np.float32(0.0)

    K = saturated_hydraulic_conductivity_m_per_s * (
        effective_saturation**pore_size_index
    )
    phi = phi_e * (
        effective_saturation ** (pore_size_index - np.float32(1.0) / n_minus_one)
    )

    dK_dS = pore_size_index * (K / effective_saturation)
    dphi_dS = (pore_size_index - np.float32(1.0) / n_minus_one) * (
        phi / effective_saturation
    )

    sigma = np.float32(0.5)

    q = np.zeros(n_layers + 1, dtype=np.float32)
    dq_dS_i = np.zeros(n_layers + 1, dtype=np.float32)
    dq_dS_iplus1 = np.zeros(n_layers + 1, dtype=np.float32)

    q[0] = np.float32(0.0)

    for i in range(n_layers - 1):
        interface_dist_m = (
            soil_layer_height_m[i] + soil_layer_height_m[i + 1]
        ) / np.float32(2.0)

        # Interface properties scaled by liquid fractions of BOTH adjacent layers.
        # This ensures that if layer i is frozen, it cannot pass water to i+1,
        # and if i+1 is frozen, it cannot pull water from i.
        interface_liq_frac = liquid_fractions[i] * liquid_fractions[i + 1]

        # Strictly zero out the interface conductivity if either layer is essentially frozen.
        if liquid_fractions[i] < 1e-4 or liquid_fractions[i + 1] < 1e-4:
            interface_liq_frac = np.float32(0.0)

        # Calculate the interface weighting factor w based on flow velocity (Ross 2003)
        v_raw = interface_dist_m / max(
            np.float32(1e-3),
            (
                bubbling_pressure_m[i + 1]
                * (
                    effective_saturation[i + 1]
                    ** (-np.float32(1.0) / n_minus_one[i + 1])
                )
            ),
        )
        v = min(np.float32(100.0), v_raw)

        w_top = (
            np.float32(60.0)
            + np.float32(10.0)
            * (np.float32(7.0) + n_minus_one[i] * pore_size_index[i])
            * v
            + (
                np.float32(16.0)
                + n_minus_one[i]
                * pore_size_index[i]
                * (np.float32(5.0) + n_minus_one[i] * pore_size_index[i])
            )
            * (v**2)
        )
        w_bot = np.float32(2.0) * (
            np.float32(60.0)
            + np.float32(60.0) * v
            + (np.float32(11.0) + (n_minus_one[i] ** 2) * (pore_size_index[i] ** 2))
            * (v**2)
        )
        weighting_factor = max(np.float32(0.0), min(np.float32(1.0), w_top / w_bot))

        # Build local values for this interface
        Ki = K[i] * interface_liq_frac
        Kiplus1 = K[i + 1] * interface_liq_frac
        phi_i = phi[i] * interface_liq_frac
        phi_iplus1 = phi[i + 1] * interface_liq_frac

        dK_dS_i_val = dK_dS[i] * interface_liq_frac
        dK_dS_iplus1_val = dK_dS[i + 1] * interface_liq_frac
        dphi_dS_i_val = dphi_dS[i] * interface_liq_frac
        dphi_dS_iplus1_val = dphi_dS[i + 1] * interface_liq_frac

        q[i + 1] = (phi_i - phi_iplus1) / interface_dist_m + (
            weighting_factor * Ki + (np.float32(1.0) - weighting_factor) * Kiplus1
        )
        dq_dS_i[i + 1] = (dphi_dS_i_val / interface_dist_m) + (
            weighting_factor * dK_dS_i_val
        )
        dq_dS_iplus1[i + 1] = (-dphi_dS_iplus1_val / interface_dist_m) + (
            (np.float32(1.0) - weighting_factor) * dK_dS_iplus1_val
        )

    q[n_layers] = np.float32(0.0)
    dq_dS_i[n_layers] = np.float32(0.0)

    a = np.zeros(n_layers, dtype=np.float32)
    b = np.zeros(n_layers, dtype=np.float32)
    c = np.zeros(n_layers, dtype=np.float32)
    d = np.zeros(n_layers, dtype=np.float32)

    for i in range(n_layers):
        storage_term = (soil_layer_height_m[i] * (theta_s[i] - theta_r[i])) / (
            sigma * dt_s
        )
        a[i] = -dq_dS_i[i]
        c[i] = dq_dS_iplus1[i + 1]
        b[i] = -dq_dS_iplus1[i] + dq_dS_i[i + 1] + storage_term

        # The imbalance term accounts for the difference between the linear approximation
        # of the S-transformation and the actual soil water storage.
        imbalance_term = (
            (
                (effective_saturation[i] * (theta_s[i] - theta_r[i]) + theta_r[i])
                - theta[i]
            )
            * soil_layer_height_m[i]
            / (sigma * dt_s)
        )
        d[i] = (q[i] - q[i + 1]) / sigma - imbalance_term

    # Thomas algorithm to solve the tridiagonal system Ax=d for dS
    cp_b = b.copy()
    cp_d = d.copy()
    for i in range(1, n_layers):
        m = a[i] / cp_b[i - 1]
        cp_b[i] = cp_b[i] - m * c[i - 1]
        cp_d[i] = cp_d[i] - m * cp_d[i - 1]

    dS = np.zeros(n_layers, dtype=np.float32)
    dS[n_layers - 1] = cp_d[n_layers - 1] / cp_b[n_layers - 1]
    for i in range(n_layers - 2, -1, -1):
        dS[i] = (cp_d[i] - c[i] * dS[i + 1]) / cp_b[i]

    # Calculate final updated fluxes including the implicit correction
    q_final = np.zeros(n_layers + 1, dtype=np.float32)
    q_final[0] = 0.0
    q_final[n_layers] = 0.0
    for i in range(n_layers - 1):
        q_final[i + 1] = q[i + 1] + sigma * (
            dq_dS_i[i + 1] * dS[i] + dq_dS_iplus1[i + 1] * dS[i + 1]
        )

    water_content_m_new = water_content_m.copy()

    for i in range(n_layers - 1):
        if i < green_ampt_active_layer_idx:
            continue

        flux_m = q_final[i + 1] * dt_s

        flux_direction = 1 if flux_m > 0 else 0
        source = i + (1 - flux_direction)
        sink = i + flux_direction
        abs_flux = abs(flux_m)

        # Cap the flux by available water in the source and remaining capacity in the sink.
        # This reflects physical constraints on saturation and residual water levels.
        remaining_storage_capacity_sink = max(
            np.float32(0.0),
            water_content_saturated_m[sink] - water_content_m_new[sink],
        )

        source_topwater_m = topwater_m if source == 0 else np.float32(0.0)
        source_temperature_C, frozen_fraction_source = (
            get_temperature_and_frozen_fraction_from_enthalpy_scalar(
                enthalpy_J_per_m2=soil_enthalpy_J_per_m2[source],
                solid_heat_capacity_J_per_m2_K=solid_heat_capacity_J_per_m2_K[source],
                water_content_m=water_content_m_new[source],
                topwater_m=source_topwater_m,
            )
        )
        liquid_fraction_source = np.float32(1.0) - np.minimum(
            np.maximum(frozen_fraction_source, np.float32(0.0)), np.float32(1.0)
        )

        # Available water for flux is water above residual content.
        available_water_source = max(
            np.float32(0.0),
            (water_content_m_new[source] - water_content_residual_m[source]),
        )

        # If the source layer is freezing or frozen, only liquid water is available for movement.
        if liquid_fraction_source < 0.999:
            available_water_source = max(
                np.float32(0.0), liquid_fraction_source * available_water_source
            )

        actual_flux_m = min(
            abs_flux, remaining_storage_capacity_sink, available_water_source
        )

        if i == 0:
            if flux_direction == 1:
                top_soil_percolation_to_layer_2_m += actual_flux_m
            else:
                top_soil_rise_from_layer_2_m += actual_flux_m

        water_content_m_new[source] -= actual_flux_m
        water_content_m_new[sink] += actual_flux_m

        # Update enthalpy based on the energy carried by the moving liquid water.
        advected_water_temperature_C = max(source_temperature_C, np.float32(0.0))
        energy_transfer_J_per_m2 = (
            actual_flux_m
            * RHO_WATER_KG_PER_M3
            * SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K
            * advected_water_temperature_C
        )

        soil_enthalpy_J_per_m2[source] -= energy_transfer_J_per_m2
        soil_enthalpy_J_per_m2[sink] += energy_transfer_J_per_m2

    return (
        water_content_m_new,
        soil_enthalpy_J_per_m2,
        top_soil_percolation_to_layer_2_m,
        top_soil_rise_from_layer_2_m,
    )
