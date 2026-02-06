"""Land surface module for GEB."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import zarr
import zarr.storage
from numba import njit, prange  # noqa: F401

from geb.geb_types import (
    ArrayFloat32,
    ArrayInt32,
    ThreeDArrayFloat32,
    TwoDArrayBool,
    TwoDArrayFloat32,
)
from geb.module import Module
from geb.store import Bucket
from geb.workflows import balance_check
from geb.workflows.io import read_grid

from .evapotranspiration import (
    calculate_bare_soil_evaporation,
    calculate_transpiration,
)
from .interception import get_interception_capacity, interception
from .landcovers import SEALED
from .potential_evapotranspiration import (
    get_CO2_induced_crop_factor_adustment,
    get_crop_factors_and_root_depths,
    get_potential_bare_soil_evaporation,
    get_potential_evapotranspiration,
    get_potential_transpiration,
    get_reference_evapotranspiration,
)
from .snow_glaciers import snow_model
from .soil import (
    add_water_to_topwater_and_evaporate_open_water,
    get_bubbling_pressure,
    get_heat_capacity_solid_fraction,
    get_interflow,
    get_pore_size_index_brakensiek,
    get_soil_moisture_at_pressure,
    get_soil_water_flow_parameters,
    infiltration,
    kv_wosten,
    rise_from_groundwater,
    solve_energy_balance_implicit_iterative,
    thetar_brakensiek,
    thetas_toth,
)

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


logger = logging.getLogger(__name__)


@njit(parallel=True, cache=True)
def land_surface_model(
    land_use_type: ArrayInt32,
    slope_m_per_m: ArrayFloat32,
    hillslope_length_m: ArrayFloat32,
    w: TwoDArrayFloat32,  # TODO: Check if fortran order speeds up
    wres: TwoDArrayFloat32,  # TODO: Check if fortran order speeds up
    wwp: TwoDArrayFloat32,  # TODO: Check if fortran order speeds up
    wfc: TwoDArrayFloat32,  # TODO: Check if fortran order speeds up
    ws: TwoDArrayFloat32,  # TODO: Check if fortran order speeds up
    soil_temperature_C: TwoDArrayFloat32,  # TODO: Check if fortran order speeds up
    solid_heat_capacity_J_per_m2_K: TwoDArrayFloat32,
    delta_z: TwoDArrayFloat32,  # TODO: Check if fortran order speeds up
    soil_layer_height: TwoDArrayFloat32,  # TODO: Check if fortran order speeds up
    root_depth_m: ArrayFloat32,
    topwater_m: ArrayFloat32,
    variable_runoff_shape_beta: ArrayFloat32,
    snow_water_equivalent_m: ArrayFloat32,
    liquid_water_in_snow_m: ArrayFloat32,
    snow_temperature_C: ArrayFloat32,
    interception_storage_m: ArrayFloat32,
    interception_capacity_m: ArrayFloat32,
    pr_kg_per_m2_per_s: TwoDArrayFloat32,
    tas_2m_K: TwoDArrayFloat32,
    dewpoint_tas_2m_K: TwoDArrayFloat32,
    ps_pascal: TwoDArrayFloat32,
    rlds_W_per_m2: TwoDArrayFloat32,
    rsds_W_per_m2: TwoDArrayFloat32,
    wind_u10m_m_per_s: TwoDArrayFloat32,
    wind_v10m_m_per_s: TwoDArrayFloat32,
    CO2_ppm: np.float32,
    crop_factor: ArrayFloat32,
    crop_map: ArrayInt32,
    actual_irrigation_consumption_m: ArrayFloat32,
    capillar_rise_m: ArrayFloat32,
    groundwater_toplayer_conductivity_m_per_day: ArrayFloat32,
    saturated_hydraulic_conductivity_m_per_s: ArrayFloat32,
    wetting_front_depth_m: ArrayFloat32,
    wetting_front_suction_head_m: ArrayFloat32,
    wetting_front_moisture_deficit: ArrayFloat32,
    green_ampt_active_layer_idx: ArrayInt32,
    lambda_pore_size_distribution: ArrayFloat32,
    bubbling_pressure_cm: ArrayFloat32,
    natural_crop_groups: ArrayFloat32,
    crop_group_number_per_group: ArrayFloat32,
    minimum_effective_root_depth_m: np.float32,
    interflow_multiplier: np.float32,
) -> tuple[
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    TwoDArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    TwoDArrayFloat32,
    ArrayFloat32,
    TwoDArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
]:
    """The main land surface model of GEB.

    This function coordinates the vertical water balance for each grid cell, including:
    - Snow accumulation and melt
    - Interception
    - Infiltration (Green-Ampt with variable infiltration capacity runoff)
    - Soil moisture redistribution (Richards equation / Darcy's Law)
    - Evapotranspiration

    Args:
        land_use_type: Land use type of the hydrological response unit.
        slope_m_per_m: Slope of the hydrological response unit in m/m.
        hillslope_length_m: Hillslope length of the hydrological response unit in m.
        w: Current soil moisture content [m3/m3].
        wres: Soil moisture content at residual [m3/m3].
        wwp: Wilting point soil moisture content [m3/m3].
        wfc: Field capacity soil moisture content [m3/m3].
        ws: Soil moisture content at saturation [m3/m3].
        soil_temperature_C: Soil temperature in Celsius.
        solid_heat_capacity_J_per_m2_K: Solid heat capacity of soil layers [J/m2/K].
        delta_z: Thickness of soil layers [m].
        soil_layer_height: Soil layer heights for the cell in meters, shape (N_SOIL_LAYERS,).
        root_depth_m: Root depth for the cell in meters.
        topwater_m: Topwater in meters, which is >=0 for paddy and 0 for non-paddy. Within
            this function topwater is used to add water from natural infiltration and
            irrigation and to calculate open water evaporation.
        variable_runoff_shape_beta: Variable infiltration capacity runoff model shape parameter for the cell.
        snow_water_equivalent_m: Snow water equivalent in meters.
        liquid_water_in_snow_m: Liquid water in snow in meters.
        snow_temperature_C: Snow temperature in Celsius.
        interception_storage_m: Interception storage in meters.
        interception_capacity_m: Interception capacity in meters.
        pr_kg_per_m2_per_s: Precipitation rate in kg/m^2/s.
        tas_2m_K: 2m air temperature in Kelvin.
        dewpoint_tas_2m_K: 2m dewpoint temperature in Kelvin.
        ps_pascal: Surface pressure in Pascals.
        rlds_W_per_m2: Downward longwave radiation in W/m^2.
        rsds_W_per_m2: Downward shortwave radiation in W/m^2.
        wind_u10m_m_per_s: U component of 10m wind speed in m/s.
        wind_v10m_m_per_s: V component of 10m wind speed in m/s.
        CO2_ppm: Atmospheric CO2 concentration in ppm.
        crop_factor: Crop factor for each HRU. Dimensionless.
        crop_map: Crop type map for each HRU.
        actual_irrigation_consumption_m: Actual irrigation consumption in meters.
        capillar_rise_m: Capillary rise in meters.
        saturated_hydraulic_conductivity_m_per_s: Saturated hydraulic conductivity in m/s.
        wetting_front_depth_m: Wetting front depth in meters.
        wetting_front_suction_head_m: Wetting front suction head [m].
        wetting_front_moisture_deficit: Moisture deficit at the wetting front [-].
        green_ampt_active_layer_idx: Index of the active soil layer for Green-Ampt infiltration.
        groundwater_toplayer_conductivity_m_per_day: Groundwater top layer conductivity in m/day.
        lambda_pore_size_distribution: Van Genuchten pore size distribution parameter.
        bubbling_pressure_cm: Bubbling pressure in cm.
        natural_crop_groups: Crop group numbers for natural areas (see WOFOST 6.0).
        crop_group_number_per_group: Crop group numbers for each crop type.
        minimum_effective_root_depth_m: Minimum effective root depth in meters.
        interflow_multiplier: Calibration factor for interflow calculation.

    Returns:
        Tuple of:
        - reference_evapotranspiration_grass_m: Reference evapotranspiration for
            grass in meters.
        - reference_evapotranspiration_water_m: Reference evapotranspiration for
            water in meters.
        - snow_water_equivalent_m: Updated snow water equivalent in meters.
        - liquid_water_in_snow_m: Updated liquid water in snow in meters.
        - sublimation_m: Sublimation in meters.
        - snow_temperature_C: Updated snow temperature in Celsius.
        - interception_storage_m: Updated interception storage in meters.
        - interception_evaporation_m: Evaporation from interception storage in meters.
        - open_water_evaporation_m: Evaporation from open water in meters.
        - bare_soil_evaporation: Evaporation from bare soil in meters.
        - transpiration_m: Transpiration in meters.
        - potential_transpiration_m: Potential transpiration in meters.
    """
    CO2_induced_crop_factor_adustment = get_CO2_induced_crop_factor_adustment(CO2_ppm)

    # convert values to substep (i.e., per hour)
    actual_irrigation_consumption_m = actual_irrigation_consumption_m / 24.0
    capillar_rise_m = capillar_rise_m / 24.0
    saturated_hydraulic_conductivity_m_per_hour = (
        saturated_hydraulic_conductivity_m_per_s
    ) * np.float32(3600.0)

    groundwater_toplayer_conductivity_m_per_hour = (
        groundwater_toplayer_conductivity_m_per_day
    ) / np.float32(24.0)

    runoff_m = np.zeros_like(pr_kg_per_m2_per_s)
    interflow_m = np.zeros_like(pr_kg_per_m2_per_s)
    reference_evapotranspiration_water_m = np.zeros_like(pr_kg_per_m2_per_s)

    # total per day variables for water balance
    reference_evapotranspiration_grass_m = np.zeros_like(snow_water_equivalent_m)
    rain_m = np.zeros_like(snow_water_equivalent_m)
    snow_m = np.zeros_like(snow_water_equivalent_m)
    sublimation_m = np.zeros_like(snow_water_equivalent_m)
    interception_evaporation_m = np.zeros_like(snow_water_equivalent_m)
    open_water_evaporation_m = np.zeros_like(snow_water_equivalent_m)
    bare_soil_evaporation = np.zeros_like(snow_water_equivalent_m)
    potential_transpiration_m = np.zeros_like(snow_water_equivalent_m)
    transpiration_m = np.zeros_like(snow_water_equivalent_m)
    groundwater_recharge_m = np.zeros_like(snow_water_equivalent_m)

    for i in prange(snow_water_equivalent_m.size):  # ty: ignore[not-iterable]
        pr_kg_per_m2_per_s_cell = pr_kg_per_m2_per_s[:, i]
        tas_2m_K_cell = tas_2m_K[:, i]
        dewpoint_tas_2m_K_cell = dewpoint_tas_2m_K[:, i]
        ps_pascal_cell = ps_pascal[:, i]
        rlds_W_per_m2_cell = rlds_W_per_m2[:, i]
        rsds_W_per_m2_cell = rsds_W_per_m2[:, i]
        wind_u10m_m_per_s_cell = wind_u10m_m_per_s[:, i]
        wind_v10m_m_per_s_cell = wind_v10m_m_per_s[:, i]

        snow_water_equivalent_m_cell = snow_water_equivalent_m[i]
        liquid_water_in_snow_m_cell = liquid_water_in_snow_m[i]
        snow_temperature_C_cell = snow_temperature_C[i]

        for hour in range(24):
            tas_C: np.float32 = tas_2m_K_cell[hour] - np.float32(273.15)
            dewpoint_tas_C: np.float32 = dewpoint_tas_2m_K_cell[hour] - np.float32(
                273.15
            )

            wind_10m_m_per_s: np.float32 = np.sqrt(
                wind_u10m_m_per_s_cell[hour] ** 2 + wind_v10m_m_per_s_cell[hour] ** 2
            )  # Wind speed at 10m height

            (
                reference_evapotranspiration_grass_m_hour_cell,
                reference_evapotranspiration_water_m_hour_cell,
                net_absorbed_radiation_vegetation_MJ_m2_dt,
                actual_vapour_pressure_Pa,
            ) = get_reference_evapotranspiration(
                tas_C=tas_C,
                dewpoint_tas_C=dewpoint_tas_C,
                ps_pa=ps_pascal_cell[hour],
                rlds_W_per_m2=rlds_W_per_m2_cell[hour],
                rsds_W_per_m2=rsds_W_per_m2_cell[hour],
                wind_10m_m_per_s=wind_10m_m_per_s,
            )

            # Ensure non-negative evapotranspiration values
            # TODO: Use this to implement dew rather than clipping
            reference_evapotranspiration_grass_m_hour_cell = max(
                reference_evapotranspiration_grass_m_hour_cell, np.float32(0.0)
            )
            reference_evapotranspiration_water_m_hour_cell = max(
                reference_evapotranspiration_water_m_hour_cell, np.float32(0.0)
            )

            reference_evapotranspiration_grass_m[i] += (
                reference_evapotranspiration_grass_m_hour_cell
            )
            reference_evapotranspiration_water_m[hour, i] += (
                reference_evapotranspiration_water_m_hour_cell
            )

            (
                rain_m_cell,
                snow_m_cell,
                snow_water_equivalent_m_cell,
                liquid_water_in_snow_m_cell,
                snow_temperature_C_cell,
                _,  # melt (before refreezing)
                runoff_from_melt_m,  # after refreezing
                rainfall_that_resulted_in_runoff_if_interception_was_not_considered_m_per_hour,
                sublimation_m_cell_hour,
                _,  # refreezing
                _,  # snow surface temperature
                _,  # net shortwave radiation
                _,  # net longwave radiation
                _,  # sensible heat flux
                _,  # latent heat flux
            ) = snow_model(
                pr_kg_per_m2_per_s=pr_kg_per_m2_per_s_cell[hour],
                air_temperature_C=tas_C,
                snow_water_equivalent_m=snow_water_equivalent_m_cell,
                liquid_water_in_snow_m=liquid_water_in_snow_m_cell,
                snow_temperature_C=snow_temperature_C_cell,
                shortwave_radiation_W_per_m2=rsds_W_per_m2_cell[hour],
                downward_longwave_radiation_W_per_m2=rlds_W_per_m2_cell[hour],
                vapor_pressure_air_Pa=actual_vapour_pressure_Pa,
                air_pressure_Pa=ps_pascal_cell[hour],
                wind_10m_m_per_s=wind_10m_m_per_s,
            )

            rain_m[i] += rain_m_cell
            snow_m[i] += snow_m_cell

            sublimation_m[i] += sublimation_m_cell_hour

            potential_bare_soil_evaporation_m: np.float32 = (
                get_potential_bare_soil_evaporation(
                    reference_evapotranspiration_grass_m_hour_cell,
                )
            )

            potential_evapotranspiration_m: np.float32 = get_potential_evapotranspiration(
                reference_evapotranspiration_grass_m=reference_evapotranspiration_grass_m_hour_cell,
                crop_factor=crop_factor[i],
                CO2_induced_crop_factor_adustment=CO2_induced_crop_factor_adustment,
            )

            potential_transpiration_m_cell_hour: np.float32 = (
                get_potential_transpiration(
                    potential_evapotranspiration_m=potential_evapotranspiration_m,
                    potential_bare_soil_evaporation_m=potential_bare_soil_evaporation_m,
                )
            )
            (
                interception_storage_m[i],
                throughfall_m,
                interception_evaporation_m_cell_hour,
                potential_transpiration_m_cell_hour,
            ) = interception(
                rainfall_m=rainfall_that_resulted_in_runoff_if_interception_was_not_considered_m_per_hour,
                storage_m=interception_storage_m[i],
                capacity_m=interception_capacity_m[i],
                potential_evaporation_m=reference_evapotranspiration_water_m_hour_cell,
                potential_transpiration_m=potential_transpiration_m_cell_hour,
            )

            interception_evaporation_m[i] += interception_evaporation_m_cell_hour

            natural_available_water_infiltration_m: np.float32 = (
                throughfall_m + runoff_from_melt_m
            )

            # TODO: Test if removing if-statements in the function speeds up the code
            topwater_m[i], open_water_evaporation_m_cell_hour = (
                add_water_to_topwater_and_evaporate_open_water(
                    natural_available_water_infiltration_m=natural_available_water_infiltration_m,
                    actual_irrigation_consumption_m=actual_irrigation_consumption_m[i],
                    land_use_type=land_use_type[i],
                    reference_evapotranspiration_water_m=reference_evapotranspiration_water_m_hour_cell,
                    topwater_m=topwater_m[i],
                )
            )
            open_water_evaporation_m[i] += open_water_evaporation_m_cell_hour

            runoff_m[hour, i] += rise_from_groundwater(
                w=w[:, i],
                ws=ws[:, i],
                capillary_rise_from_groundwater=capillar_rise_m[i],
            )

            soil_temperature_C[0, i] = solve_energy_balance_implicit_iterative(
                soil_temperature_C=soil_temperature_C[0, i],
                solid_heat_capacity_J_per_m2_K=solid_heat_capacity_J_per_m2_K[0, i],
                shortwave_radiation_W_per_m2=rsds_W_per_m2_cell[hour],
                longwave_radiation_W_per_m2=rlds_W_per_m2_cell[hour],
                air_temperature_K=tas_2m_K_cell[hour],
                wind_speed_10m_m_per_s=wind_10m_m_per_s,
                surface_pressure_pa=ps_pascal_cell[hour],
                timestep_seconds=np.float32(3600.0),
            )

            soil_is_frozen = soil_temperature_C[0, i] <= np.float32(0.0)

            (
                topwater_m[i],
                direct_runoff_m,
                groundwater_recharge_from_infiltraton_m,
                infiltration_amount,
                wetting_front_depth_m[i],
                wetting_front_suction_head_m[i],
                wetting_front_moisture_deficit[i],
                green_ampt_active_layer_idx[i],
            ) = infiltration(
                ws=ws[:, i],
                wres=wres[:, i],
                saturated_hydraulic_conductivity_m_per_timestep=saturated_hydraulic_conductivity_m_per_hour[
                    :, i
                ],
                groundwater_toplayer_conductivity_m_per_timestep=groundwater_toplayer_conductivity_m_per_hour[
                    i
                ],
                land_use_type=land_use_type[i],
                soil_is_frozen=soil_is_frozen,
                w=w[:, i],
                topwater_m=topwater_m[i],
                capillary_rise_from_groundwater_m=capillar_rise_m[i],
                wetting_front_depth_m=wetting_front_depth_m[i],
                wetting_front_suction_head_m=wetting_front_suction_head_m[i],
                wetting_front_moisture_deficit=wetting_front_moisture_deficit[i],
                green_ampt_active_layer_idx=green_ampt_active_layer_idx[i],
                variable_runoff_shape_beta=variable_runoff_shape_beta[i],
                bubbling_pressure_cm=bubbling_pressure_cm[:, i],
                soil_layer_height_m=soil_layer_height[:, i],
                lambda_pore_size_distribution=lambda_pore_size_distribution[:, i],
            )
            runoff_m[hour, i] += direct_runoff_m
            groundwater_recharge_m[i] += groundwater_recharge_from_infiltraton_m

            bottom_layer = N_SOIL_LAYERS - 1  # ty: ignore[unresolved-reference]

            psi: np.float32
            unsaturated_hydraulic_conductivity_m_per_hour: np.float32
            psi, unsaturated_hydraulic_conductivity_m_per_hour = (
                get_soil_water_flow_parameters(
                    w=w[bottom_layer, i],
                    wres=wres[bottom_layer, i],
                    ws=ws[bottom_layer, i],
                    lambda_pore_size_distribution=lambda_pore_size_distribution[
                        bottom_layer, i
                    ],
                    saturated_hydraulic_conductivity_m_per_timestep=saturated_hydraulic_conductivity_m_per_hour[
                        bottom_layer, i
                    ],
                    bubbling_pressure_cm=bubbling_pressure_cm[bottom_layer, i],
                )
            )

            # Percolation from bottom soil layer to groundwater.
            # This is the original bottom-layer drainage implementation. It is turned off
            # when the Green-Ampt infiltration routine already produced groundwater recharge
            # for this hour to avoid double counting.
            if groundwater_recharge_from_infiltraton_m <= np.float32(0.0):
                # We assume that the bottom layer is draining under gravity
                # i.e., assuming homogeneous soil water potential below
                # bottom layer all the way to groundwater.
                # If there is capillary rise from groundwater, we assume no simultaneous
                # percolation to groundwater.
                flux: np.float32 = unsaturated_hydraulic_conductivity_m_per_hour * (
                    capillar_rise_m[i] <= np.float32(0)
                )
                # Limit flux by the saturated hydraulic conductivity of groundwater top layer
                flux = min(flux, groundwater_toplayer_conductivity_m_per_hour[i])

                # Limit flux by available water in the bottom layer
                available_water_source: np.float32 = (
                    w[bottom_layer, i] - wres[bottom_layer, i]
                )
                flux = min(flux, available_water_source)
                w[bottom_layer, i] -= flux
                w[bottom_layer, i] = max(w[bottom_layer, i], wres[bottom_layer, i])
                groundwater_recharge_m[i] += flux

            # Calculate interflow from bottom layer
            interflow_cell_hour: np.float32 = get_interflow(
                w=w[bottom_layer, i],
                wfc=wfc[bottom_layer, i],
                ws=ws[bottom_layer, i],
                soil_layer_height_m=soil_layer_height[bottom_layer, i],
                saturated_hydraulic_conductivity_m_per_hour=saturated_hydraulic_conductivity_m_per_hour[
                    bottom_layer, i
                ],
                slope_m_per_m=slope_m_per_m[i],
                hillslope_length_m=hillslope_length_m[i],
                interflow_multiplier=interflow_multiplier,
            )

            interflow_m[hour, i] += interflow_cell_hour
            w[bottom_layer, i] -= interflow_cell_hour
            w[bottom_layer, i] = max(w[bottom_layer, i], wres[bottom_layer, i])

            psi_layer_below = psi
            unsaturated_hydraulic_conductivity_layer_below = (
                unsaturated_hydraulic_conductivity_m_per_hour
            )

            # iterate from bottom to top layer (ignoring the bottom layer which is treated above)
            for layer in range(N_SOIL_LAYERS - 2, -1, -1):  # ty: ignore[unresolved-reference]
                psi, unsaturated_hydraulic_conductivity_m_per_hour = (
                    get_soil_water_flow_parameters(
                        w=w[layer, i],
                        wres=wres[layer, i],
                        ws=ws[layer, i],
                        lambda_pore_size_distribution=lambda_pore_size_distribution[
                            layer, i
                        ],
                        saturated_hydraulic_conductivity_m_per_timestep=saturated_hydraulic_conductivity_m_per_hour[
                            layer, i
                        ],
                        bubbling_pressure_cm=bubbling_pressure_cm[layer, i],
                    )
                )

                # If the layer is above the wetting front, we skip Darcy flow calculations
                # because the Green-Ampt infiltration is handling the water movement in this
                # region. We effectively "freeze" the redistribution here to let the piston
                # flow dominate.

                # Important: in the current implementation, this means that no redistribution
                # occurs in the layer that is the wetting front layer. This is a simplification
                # that could be improved in future versions.
                if layer >= green_ampt_active_layer_idx[i]:
                    # Compute flux using Darcy's law. The -1 accounts for gravity.
                    # Positive flux is downwards; see minus sign in the equation, which negates
                    # the -1 of gravity and other terms.
                    # We use upstream weighting for the hydraulic conductivity,
                    # which means that we use the hydraulic conductivity of the layer
                    # that the flux is coming from.
                    flux_gradient_term: np.float32 = -(
                        (psi_layer_below - psi) / delta_z[layer, i] - np.float32(1.0)
                    )

                    # if the flux gradient term is positive, the flux is going downwards
                    # and we use the hydraulic conductivity of the current layer
                    if flux_gradient_term > 0:
                        flux: np.float32 = (
                            unsaturated_hydraulic_conductivity_m_per_hour
                            * flux_gradient_term
                        )
                        flux_direction = 1  # 1 if flux >= 0, 0 if flux < 0
                    # if the flux gradient term is negative, the flux is going upwards
                    # thus we use the hydraulic conductivity of the layer below
                    else:
                        flux: np.float32 = -(
                            unsaturated_hydraulic_conductivity_layer_below
                            * flux_gradient_term
                        )
                        flux_direction = 0  # 1 if flux >= 0, 0 if flux < 0

                    # Limit flux by the minimum saturated hydraulic conductivity of the two layers
                    min_saturated_hydraulic_conductivity_m_per_hour = min(
                        saturated_hydraulic_conductivity_m_per_hour[layer, i],
                        saturated_hydraulic_conductivity_m_per_hour[layer + 1, i],
                    )
                    flux = min(flux, min_saturated_hydraulic_conductivity_m_per_hour)

                    source: int = layer + (
                        1 - flux_direction
                    )  # layer if flux >= 0, layer + 1 if flux < 0
                    sink: int = (
                        layer + flux_direction
                    )  # layer + 1 if flux >= 0, layer if flux < 0

                    # Limit flux by available water in source and storage capacity of sink
                    remaining_storage_capacity_sink = ws[sink, i] - w[sink, i]
                    available_water_source = w[source, i] - wres[source, i]

                    flux = min(
                        flux,
                        remaining_storage_capacity_sink,
                        available_water_source,
                    )

                    # Update water content in source and sink layers
                    w[source, i] -= flux
                    w[sink, i] += flux

                    # Ensure water content stays within physical bounds
                    w[sink, i] = min(w[sink, i], ws[sink, i])
                    w[source, i] = max(w[source, i], wres[source, i])

                psi_layer_below = psi
                unsaturated_hydraulic_conductivity_layer_below = (
                    unsaturated_hydraulic_conductivity_m_per_hour
                )

                # Calculate interflow from this layer
                interflow_cell_hour: np.float32 = get_interflow(
                    w=w[layer, i],
                    wfc=wfc[layer, i],
                    ws=ws[layer, i],
                    soil_layer_height_m=soil_layer_height[layer, i],
                    saturated_hydraulic_conductivity_m_per_hour=saturated_hydraulic_conductivity_m_per_hour[
                        layer, i
                    ],
                    slope_m_per_m=slope_m_per_m[i],
                    hillslope_length_m=hillslope_length_m[i],
                    interflow_multiplier=interflow_multiplier,
                )

                interflow_m[hour, i] += interflow_cell_hour
                w[layer, i] -= interflow_cell_hour
                w[layer, i] = max(w[layer, i], wres[layer, i])

            # soil moisture is updated in place
            transpiration_m_cell_hour, topwater_m[i] = calculate_transpiration(
                soil_is_frozen=soil_is_frozen,
                wwp_m=wwp[:, i],
                wfc_m=wfc[:, i],
                wres_m=wres[:, i],
                soil_layer_height_m=soil_layer_height[:, i],
                land_use_type=land_use_type[i],
                root_depth_m=root_depth_m[i],
                crop_map=crop_map[i],
                natural_crop_groups=natural_crop_groups[i],
                potential_transpiration_m=potential_transpiration_m_cell_hour,
                reference_evapotranspiration_grass_m_hour=reference_evapotranspiration_grass_m_hour_cell,
                crop_group_number_per_group=crop_group_number_per_group,
                w_m=w[:, i],
                topwater_m=topwater_m[i],
                minimum_effective_root_depth_m=minimum_effective_root_depth_m,
            )

            potential_transpiration_m[i] += potential_transpiration_m_cell_hour
            transpiration_m[i] += transpiration_m_cell_hour

            # soil moisture is updated in place
            bare_soil_evaporation[i] += calculate_bare_soil_evaporation(
                soil_is_frozen=soil_is_frozen,
                land_use_type=land_use_type[i],
                potential_bare_soil_evaporation_m=potential_bare_soil_evaporation_m,
                open_water_evaporation_m=open_water_evaporation_m_cell_hour,
                w_m=w[:, i],
                wres_m=wres[:, i],
                unsaturated_hydraulic_conductivity_m_per_hour=unsaturated_hydraulic_conductivity_m_per_hour,
            )

        snow_water_equivalent_m[i] = snow_water_equivalent_m_cell
        liquid_water_in_snow_m[i] = liquid_water_in_snow_m_cell
        snow_temperature_C[i] = snow_temperature_C_cell

    # TODO: Also solve vertical soil water balance for non-bio land use types
    # some of the above calculations for non-bio land use types will lead to NaNs
    # but these can be safely converted to zeros
    groundwater_recharge_m = np.nan_to_num(groundwater_recharge_m)
    interflow_m = np.nan_to_num(interflow_m)
    bare_soil_evaporation = np.nan_to_num(bare_soil_evaporation)
    transpiration_m = np.nan_to_num(transpiration_m)
    potential_transpiration_m = np.nan_to_num(potential_transpiration_m)

    return (
        rain_m,
        snow_m,
        topwater_m,
        reference_evapotranspiration_grass_m,
        reference_evapotranspiration_water_m,
        snow_water_equivalent_m,
        liquid_water_in_snow_m,
        sublimation_m,
        snow_temperature_C,
        interception_storage_m,
        interception_evaporation_m,
        open_water_evaporation_m,
        runoff_m,
        groundwater_recharge_m,
        interflow_m,
        bare_soil_evaporation,
        transpiration_m,
        potential_transpiration_m,
    )


class LandSurfaceInputs(NamedTuple):
    """Container for `land_surface_model` inputs.

    This keeps the model call and debug dumps in sync by using the same
    ordered, named fields for both pathways.
    """

    land_use_type: ArrayInt32
    slope_m_per_m: ArrayFloat32
    hillslope_length_m: ArrayFloat32
    w: TwoDArrayFloat32
    wres: TwoDArrayFloat32
    wwp: TwoDArrayFloat32
    wfc: TwoDArrayFloat32
    ws: TwoDArrayFloat32
    soil_temperature_C: TwoDArrayFloat32
    solid_heat_capacity_J_per_m2_K: TwoDArrayFloat32
    delta_z: TwoDArrayFloat32
    soil_layer_height: TwoDArrayFloat32
    root_depth_m: ArrayFloat32
    topwater_m: ArrayFloat32
    variable_runoff_shape_beta: ArrayFloat32
    snow_water_equivalent_m: ArrayFloat32
    liquid_water_in_snow_m: ArrayFloat32
    snow_temperature_C: ArrayFloat32
    interception_storage_m: ArrayFloat32
    interception_capacity_m: ArrayFloat32
    pr_kg_per_m2_per_s: TwoDArrayFloat32
    tas_2m_K: TwoDArrayFloat32
    dewpoint_tas_2m_K: TwoDArrayFloat32
    ps_pascal: TwoDArrayFloat32
    rlds_W_per_m2: TwoDArrayFloat32
    rsds_W_per_m2: TwoDArrayFloat32
    wind_u10m_m_per_s: TwoDArrayFloat32
    wind_v10m_m_per_s: TwoDArrayFloat32
    CO2_ppm: np.float32
    crop_factor: ArrayFloat32
    crop_map: ArrayInt32
    actual_irrigation_consumption_m: ArrayFloat32
    capillar_rise_m: ArrayFloat32
    groundwater_toplayer_conductivity_m_per_day: ArrayFloat32
    saturated_hydraulic_conductivity_m_per_s: TwoDArrayFloat32
    wetting_front_depth_m: ArrayFloat32
    wetting_front_suction_head_m: ArrayFloat32
    wetting_front_moisture_deficit: ArrayFloat32
    green_ampt_active_layer_idx: ArrayInt32
    lambda_pore_size_distribution: TwoDArrayFloat32
    bubbling_pressure_cm: TwoDArrayFloat32
    natural_crop_groups: ArrayFloat32
    crop_group_number_per_group: ArrayFloat32
    minimum_effective_root_depth_m: np.float32
    interflow_multiplier: np.float32


class LandSurfaceVariables(Bucket):
    """Land surface variables for GEB."""

    topwater: ArrayFloat32
    clay_percentage: TwoDArrayFloat32
    sand_percentage: TwoDArrayFloat32
    silt_percentage: TwoDArrayFloat32
    soil_layer_height: TwoDArrayFloat32
    snow_water_equivalent_m: ArrayFloat32
    liquid_water_in_snow_m: ArrayFloat32
    snow_temperature_C: ArrayFloat32
    interception_storage_m: ArrayFloat32
    variable_runoff_shape_beta: TwoDArrayFloat32
    crop_map: ArrayInt32
    minimum_effective_root_depth_m: np.float32
    green_ampt_active_layer_idx: ArrayInt32


class LandSurface(Module):
    """Land surface module for GEB."""

    var: LandSurfaceVariables

    def __init__(self, model: GEBModel, hydrology: Hydrology) -> None:
        """Initialize the potential evaporation module.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology module.
        """
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        """Name of the module."""
        return "hydrology.landsurface"

    def set_global_variables(self) -> None:
        """Set global variables for the land surface module.

        This is nessecary because we want this variable to be available in the numba.
        Passing it as a global variable will allow numba to optimize the code better.
        At the same time we avoid using a constant variable, which does not allow
        to change the number of soil layers between different datasets.
        """
        # set number of soil layers as global variable for numba
        global N_SOIL_LAYERS  # ty: ignore[unresolved-global]
        N_SOIL_LAYERS = self.HRU.var.soil_layer_height_m.shape[0]

    def _build_land_surface_inputs(
        self,
        *,
        root_depth_m: ArrayFloat32,
        interception_capacity_m: ArrayFloat32,
        pr_kg_per_m2_per_s: TwoDArrayFloat32,
        crop_factor: ArrayFloat32,
        actual_irrigation_consumption_m: ArrayFloat32,
        capillar_rise_m: ArrayFloat32,
        delta_z: TwoDArrayFloat32,
    ) -> LandSurfaceInputs:
        """Build the input bundle for `land_surface_model`.

        Args:
            root_depth_m: Root depth for each HRU (m).
            interception_capacity_m: Interception capacity per HRU (m).
            pr_kg_per_m2_per_s: Precipitation rate per hour (kg/m2/s).
            crop_factor: Crop factor per HRU (-).
            actual_irrigation_consumption_m: Actual irrigation consumption (m).
            capillar_rise_m: Capillary rise (m).
            delta_z: Layer interface thicknesses (m).

        Returns:
            Bundle of inputs for `land_surface_model`.
        """
        pr_kg_per_m2_per_s_for_model: TwoDArrayFloat32 = np.asfortranarray(
            pr_kg_per_m2_per_s
        )
        tas_2m_K_for_model: TwoDArrayFloat32 = np.asfortranarray(self.HRU.tas_2m_K)
        dewpoint_tas_2m_K_for_model: TwoDArrayFloat32 = np.asfortranarray(
            self.HRU.dewpoint_tas_2m_K
        )
        ps_pascal_for_model: TwoDArrayFloat32 = np.asfortranarray(self.HRU.ps_pascal)
        rlds_W_per_m2_for_model: TwoDArrayFloat32 = np.asfortranarray(
            self.HRU.rlds_W_per_m2
        )
        rsds_W_per_m2_for_model: TwoDArrayFloat32 = np.asfortranarray(
            self.HRU.rsds_W_per_m2
        )
        wind_u10m_m_per_s_for_model: TwoDArrayFloat32 = np.asfortranarray(
            self.HRU.wind_u10m_m_per_s
        )
        wind_v10m_m_per_s_for_model: TwoDArrayFloat32 = np.asfortranarray(
            self.HRU.wind_v10m_m_per_s
        )
        CO2_ppm: np.float32 = np.float32(self.model.forcing.load("CO2_ppm"))
        groundwater_toplayer_conductivity_m_per_day: ArrayFloat32 = (
            self.hydrology.to_HRU(
                data=self.grid.var.groundwater_hydraulic_conductivity_m_per_day[0],
                fn=None,  # the top layer is the first groundwater layer
            )
        )
        crop_group_number_per_group: ArrayFloat32 = (
            self.model.agents.crop_farmers.var.crop_data[
                "crop_group_number"
            ].values.astype(np.float32)
        )

        return LandSurfaceInputs(
            land_use_type=self.HRU.var.land_use_type,
            slope_m_per_m=self.HRU.var.slope_m_per_m,
            hillslope_length_m=self.HRU.var.hillslope_length_m,
            w=self.HRU.var.w,
            wres=self.HRU.var.wres,
            wwp=self.HRU.var.wwp,
            wfc=self.HRU.var.wfc,
            ws=self.HRU.var.ws,
            soil_temperature_C=self.HRU.var.soil_temperature_C,
            solid_heat_capacity_J_per_m2_K=self.HRU.var.solid_heat_capacity_J_per_m2_K,
            delta_z=delta_z,
            soil_layer_height=self.HRU.var.soil_layer_height_m,
            root_depth_m=root_depth_m,
            topwater_m=self.HRU.var.topwater_m,
            variable_runoff_shape_beta=self.HRU.var.variable_runoff_shape_beta,
            snow_water_equivalent_m=self.HRU.var.snow_water_equivalent_m,
            liquid_water_in_snow_m=self.HRU.var.liquid_water_in_snow_m,
            snow_temperature_C=self.HRU.var.snow_temperature_C,
            interception_storage_m=self.HRU.var.interception_storage_m,
            interception_capacity_m=interception_capacity_m,
            pr_kg_per_m2_per_s=pr_kg_per_m2_per_s_for_model,
            tas_2m_K=tas_2m_K_for_model,
            dewpoint_tas_2m_K=dewpoint_tas_2m_K_for_model,
            ps_pascal=ps_pascal_for_model,
            rlds_W_per_m2=rlds_W_per_m2_for_model,
            rsds_W_per_m2=rsds_W_per_m2_for_model,
            wind_u10m_m_per_s=wind_u10m_m_per_s_for_model,
            wind_v10m_m_per_s=wind_v10m_m_per_s_for_model,
            CO2_ppm=CO2_ppm,
            crop_factor=crop_factor,
            crop_map=self.HRU.var.crop_map,
            actual_irrigation_consumption_m=actual_irrigation_consumption_m,
            capillar_rise_m=capillar_rise_m,
            groundwater_toplayer_conductivity_m_per_day=groundwater_toplayer_conductivity_m_per_day,
            saturated_hydraulic_conductivity_m_per_s=self.HRU.var.saturated_hydraulic_conductivity_m_per_s,
            wetting_front_depth_m=self.HRU.var.wetting_front_depth_m,
            wetting_front_suction_head_m=self.HRU.var.wetting_front_suction_head_m,
            wetting_front_moisture_deficit=self.HRU.var.wetting_front_moisture_deficit,
            green_ampt_active_layer_idx=self.HRU.var.green_ampt_active_layer_idx,
            lambda_pore_size_distribution=self.HRU.var.lambda_pore_size_distribution,
            bubbling_pressure_cm=self.HRU.var.bubbling_pressure_cm,
            natural_crop_groups=self.HRU.var.natural_crop_groups,
            crop_group_number_per_group=crop_group_number_per_group,
            minimum_effective_root_depth_m=self.var.minimum_effective_root_depth_m,
            interflow_multiplier=self.model.config["parameters"][
                "interflow_multiplier"
            ],
        )

    def _snapshot_land_surface_inputs_for_error(
        self,
        *,
        land_surface_inputs: LandSurfaceInputs,
        w_prev: TwoDArrayFloat32,
        topwater_m_prev: ArrayFloat32,
        snow_water_equivalent_prev: ArrayFloat32,
        liquid_water_in_snow_prev: ArrayFloat32,
        snow_temperature_C_prev: ArrayFloat32,
        interception_storage_prev: ArrayFloat32,
        soil_temperature_C_prev: TwoDArrayFloat32,
        wetting_front_depth_prev: ArrayFloat32,
        wetting_front_suction_head_prev: ArrayFloat32,
        wetting_front_moisture_deficit_prev: ArrayFloat32,
        green_ampt_active_layer_idx_prev: ArrayInt32,
    ) -> LandSurfaceInputs:
        """Build a snapshot of land surface inputs for error reproduction.

        Args:
            land_surface_inputs: Inputs used for the normal model call.
            w_prev: Pre-call soil water column (m).
            topwater_m_prev: Pre-call topwater (m).
            snow_water_equivalent_prev: Pre-call snow water equivalent (m).
            liquid_water_in_snow_prev: Pre-call liquid water in snow (m).
            snow_temperature_C_prev: Pre-call snow temperature (C).
            interception_storage_prev: Pre-call interception storage (m).
            soil_temperature_C_prev: Pre-call soil temperature (C).
            wetting_front_depth_prev: Pre-call wetting front depth (m).
            wetting_front_suction_head_prev: Pre-call wetting front suction head (m).
            wetting_front_moisture_deficit_prev: Pre-call wetting front moisture deficit (-).
            green_ampt_active_layer_idx_prev: Pre-call Green-Ampt active layer index (-).

        Returns:
            Snapshot of model inputs that reproduces the failure context.
        """
        error_inputs: LandSurfaceInputs = land_surface_inputs._replace(
            w=w_prev,
            topwater_m=topwater_m_prev,
            snow_water_equivalent_m=snow_water_equivalent_prev,
            liquid_water_in_snow_m=liquid_water_in_snow_prev,
            snow_temperature_C=snow_temperature_C_prev,
            interception_storage_m=interception_storage_prev,
            soil_temperature_C=soil_temperature_C_prev,
            wetting_front_depth_m=wetting_front_depth_prev,
            wetting_front_suction_head_m=wetting_front_suction_head_prev,
            wetting_front_moisture_deficit=wetting_front_moisture_deficit_prev,
            green_ampt_active_layer_idx=green_ampt_active_layer_idx_prev,
        )
        return error_inputs

    def spinup(self) -> None:
        """Spinup function for the land surface module."""
        self.HRU.var.topwater_m = self.HRU.full_compressed(0.0, dtype=np.float32)

        self.HRU.var.snow_water_equivalent_m = self.HRU.full_compressed(
            0.0, dtype=np.float32
        )
        self.HRU.var.liquid_water_in_snow_m = self.HRU.full_compressed(
            0.0, dtype=np.float32
        )
        self.HRU.var.snow_temperature_C = self.HRU.full_compressed(
            0.0, dtype=np.float32
        )
        self.HRU.var.interception_storage_m = self.HRU.full_compressed(
            0.0, dtype=np.float32
        )

        self.HRU.var.variable_runoff_shape_beta = self.HRU.full_compressed(
            0.0, dtype=np.float32
        )

        self.HRU.var.slope_m_per_m = self.hydrology.to_HRU(
            data=self.hydrology.grid.load(
                self.model.files["grid"]["landsurface/slope"]
            ),
            fn=None,
        )

        self.HRU.var.hillslope_length_m = self.hydrology.to_HRU(
            data=self.grid.var.cell_area**0.5, fn=None
        )

        store = zarr.storage.LocalStore(
            self.model.files["grid"]["landcover/forest/interception_capacity"],
            read_only=True,
        )

        interception_capacity_forest_group = zarr.open_group(store, mode="r")[
            "interception_capacity"
        ]
        assert isinstance(interception_capacity_forest_group, zarr.Array)
        interception_capacity_forest_array = interception_capacity_forest_group[:]
        assert isinstance(interception_capacity_forest_array, np.ndarray)
        # fmt: off
        interception_capacity_forest_array: ThreeDArrayFloat32 = (
            interception_capacity_forest_array
        )  # ty:ignore[invalid-assignment]
        # fmt: on

        self.grid.var.interception_capacity_forest: TwoDArrayFloat32 = (
            self.grid.compress(interception_capacity_forest_array)
        )

        store = zarr.storage.LocalStore(
            self.model.files["grid"]["landcover/grassland/interception_capacity"],
            read_only=True,
        )

        interception_capacity_grassland_group = zarr.open_group(store, mode="r")[
            "interception_capacity"
        ]
        assert isinstance(interception_capacity_grassland_group, zarr.Array)
        interception_capacity_grassland_array = interception_capacity_grassland_group[:]
        assert isinstance(interception_capacity_grassland_array, np.ndarray)

        # fmt: off
        interception_capacity_grassland_array: ThreeDArrayFloat32 = (
            interception_capacity_grassland_array
        )  # ty:ignore[invalid-assignment]
        # fmt: on

        self.grid.var.interception_capacity_grassland: TwoDArrayFloat32 = (
            self.grid.compress(interception_capacity_grassland_array)
        )

        store = zarr.storage.LocalStore(
            self.model.files["grid"]["landcover/forest/crop_coefficient"],
            read_only=True,
        )
        forest_crop_factor_per_10_days_group = zarr.open_group(store, mode="r")[
            "crop_coefficient"
        ]
        assert isinstance(forest_crop_factor_per_10_days_group, zarr.Array)
        forest_crop_factor_per_10_days_group_array = (
            forest_crop_factor_per_10_days_group[:]
        )
        # fmt: off
        forest_crop_factor_per_10_days_group_array: ThreeDArrayFloat32 = (
            forest_crop_factor_per_10_days_group_array
        )  # ty:ignore[invalid-assignment]
        # fmt: on

        self.grid.var.forest_crop_factor_per_10_days = (
            forest_crop_factor_per_10_days_group_array
        )

        # Default follows AQUACROP recommendation, see reference manual for AquaCrop v7.1 – Chapter 3
        self.var.minimum_effective_root_depth_m: np.float32 = np.float32(0.25)

        self.setup_soil_properties()

    def setup_soil_properties(self) -> None:
        """Setup soil properties for the land surface module."""
        # Soil properties
        self.HRU.var.soil_layer_height_m: TwoDArrayFloat32 = (
            self.HRU.convert_subgrid_to_HRU(
                read_grid(
                    self.model.files["subgrid"]["soil/soil_layer_height_m"],
                    layer=None,
                ),
                method="mean",
            )
        )

        # Load soil organic carbon and bulk density with logging
        soc_path = self.model.files["subgrid"]["soil/soil_organic_carbon_percentage"]
        bd_path = self.model.files["subgrid"]["soil/bulk_density_kg_per_dm3"]

        logger.info("\n" + "=" * 80)
        logger.info("LOADING SOIL DATA IN HYDROLOGY MODULE")
        logger.info("=" * 80)
        logger.info(f"Loading SOC from: {soc_path}")
        logger.info(f"Loading bulk density from: {bd_path}")

        if "forest_modified" in str(soc_path) or "forest_modified" in str(bd_path):
            logger.info("Using MODIFIED soil maps for forest planting scenario")
            print(
                "\n[HYDROLOGY] Loading MODIFIED soil maps (forest planting scenario)",
                flush=True,
            )
        else:
            logger.info("Using original soil maps")
            print(
                "\n[HYDROLOGY] Loading original soil maps, no forest planting scenario yet implemented",
                flush=True,
            )

        self.HRU.var.depth_to_bedrock_m: ArrayFloat32 = self.HRU.convert_subgrid_to_HRU(
            read_grid(
                self.model.files["subgrid"]["soil/depth_to_bedrock_m"],
                layer=None,
            ),
            method="mean",
        )

        organic_carbon_percentage: TwoDArrayFloat32 = self.HRU.convert_subgrid_to_HRU(
            read_grid(
                self.model.files["subgrid"]["soil/soil_organic_carbon_percentage"],
                layer=None,
            ),
            method="mean",
        )
        bulk_density_kg_per_dm3: TwoDArrayFloat32 = self.HRU.convert_subgrid_to_HRU(
            read_grid(
                self.model.files["subgrid"]["soil/bulk_density_kg_per_dm3"],
                layer=None,
            ),
            method="mean",
        )
        self.HRU.var.silt_percentage: TwoDArrayFloat32 = (
            self.HRU.convert_subgrid_to_HRU(
                read_grid(
                    self.model.files["subgrid"]["soil/silt_percentage"],
                    layer=None,
                ),
                method="mean",
            )
        )
        self.HRU.var.clay_percentage: TwoDArrayFloat32 = (
            self.HRU.convert_subgrid_to_HRU(
                read_grid(
                    self.model.files["subgrid"]["soil/clay_percentage"],
                    layer=None,
                ),
                method="mean",
            )
        )

        # calculate sand content based on silt and clay content (together they should sum to 100%)
        self.HRU.var.sand_percentage: TwoDArrayFloat32 = (
            100 - self.HRU.var.silt_percentage - self.HRU.var.clay_percentage
        )

        # the top 30 cm is considered as top soil (https://www.fao.org/uploads/media/Harm-World-Soil-DBv7cv_1.pdf)
        is_top_soil: TwoDArrayBool = np.zeros_like(
            self.HRU.var.clay_percentage, dtype=bool
        )
        is_top_soil[0:3] = True

        thetas: TwoDArrayFloat32 = thetas_toth(
            organic_carbon_percentage=organic_carbon_percentage,
            bulk_density_kg_per_dm3=bulk_density_kg_per_dm3,
            is_top_soil=is_top_soil,
            clay=self.HRU.var.clay_percentage,
            silt=self.HRU.var.silt_percentage,
        )

        thetar: TwoDArrayFloat32 = thetar_brakensiek(
            sand=self.HRU.var.sand_percentage,
            clay=self.HRU.var.clay_percentage,
            thetas=thetas,
        )
        self.HRU.var.bubbling_pressure_cm = get_bubbling_pressure(
            clay=self.HRU.var.clay_percentage,
            sand=self.HRU.var.sand_percentage,
            thetas=thetas,
        )
        self.HRU.var.lambda_pore_size_distribution = get_pore_size_index_brakensiek(
            sand=self.HRU.var.sand_percentage,
            thetas=thetas,
            clay=self.HRU.var.clay_percentage,
        )

        # θ saturation, field capacity, wilting point and residual moisture content
        thetafc: TwoDArrayFloat32 = get_soil_moisture_at_pressure(
            np.float32(-100.0),  # assuming field capacity is at -100 cm (pF 2)
            self.HRU.var.bubbling_pressure_cm,
            thetas,
            thetar,
            self.HRU.var.lambda_pore_size_distribution,
        )

        thetawp: TwoDArrayFloat32 = get_soil_moisture_at_pressure(
            np.float32(-(10**4.2)),  # assuming wilting point is at -10^4.2 cm (pF 4.2)
            self.HRU.var.bubbling_pressure_cm,
            thetas,
            thetar,
            self.HRU.var.lambda_pore_size_distribution,
        )

        self.HRU.var.ws: TwoDArrayFloat32 = thetas * self.HRU.var.soil_layer_height_m
        self.HRU.var.wfc: TwoDArrayFloat32 = thetafc * self.HRU.var.soil_layer_height_m
        self.HRU.var.wwp: TwoDArrayFloat32 = thetawp * self.HRU.var.soil_layer_height_m
        self.HRU.var.wres: TwoDArrayFloat32 = thetar * self.HRU.var.soil_layer_height_m

        # initial soil water storage between field capacity and wilting point
        # set soil moisture to nan where land use is not bioarea
        self.HRU.var.w: TwoDArrayFloat32 = np.where(
            self.HRU.var.land_use_type[np.newaxis, :] < SEALED,
            (self.HRU.var.wfc - self.HRU.var.wwp) * 0.2 + self.HRU.var.wwp,
            np.nan,
        )
        # for paddy irrigation flooded paddy fields
        self.HRU.var.topwater: ArrayFloat32 = self.HRU.full_compressed(
            0, dtype=np.float32
        )

        # self.HRU.var.saturated_hydraulic_conductivity_m_per_s: TwoDArrayFloat32 = (
        #     kv_brakensiek(thetas=thetas, clay=self.HRU.var.clay, sand=self.HRU.var.sand)
        # )

        # self.HRU.var.saturated_hydraulic_conductivity_m_per_s: TwoDArrayFloat32 = kv_cosby(
        #     sand=self.HRU.var.sand, clay=self.HRU.var.clay
        # )

        self.HRU.var.saturated_hydraulic_conductivity_m_per_s: TwoDArrayFloat32 = (
            kv_wosten(
                silt=self.HRU.var.silt_percentage,
                clay=self.HRU.var.clay_percentage,
                bulk_density_kg_per_dm3=bulk_density_kg_per_dm3,
                organic_carbon_percentage=organic_carbon_percentage,
                is_topsoil=is_top_soil,
            )
        )

        self.HRU.var.saturated_hydraulic_conductivity_m_per_s *= self.model.config[
            "parameters"
        ]["saturated_hydraulic_conductivity_multiplier"]  # calibration parameter

        self.HRU.var.wetting_front_depth_m = np.full_like(self.HRU.var.topwater, 0.0)
        self.HRU.var.wetting_front_moisture_deficit = np.full_like(
            self.HRU.var.topwater, 0.0
        )
        self.HRU.var.wetting_front_suction_head_m = np.full_like(
            self.HRU.var.topwater, 0.0
        )
        self.HRU.var.green_ampt_active_layer_idx = np.full_like(
            self.HRU.var.topwater, -1, dtype=np.int32
        )

        self.HRU.var.soil_temperature_C = np.full_like(
            self.HRU.var.soil_layer_height_m, 0.0, dtype=np.float32
        )

        self.HRU.var.solid_heat_capacity_J_per_m2_K = get_heat_capacity_solid_fraction(
            bulk_density_kg_per_dm3=bulk_density_kg_per_dm3,
            layer_thickness_m=self.HRU.var.soil_layer_height_m,
        )

        # soil water depletion fraction, Van Diepen et al., 1988: WOFOST 6.0, p.86, Doorenbos et. al 1978
        # crop groups for formular in van Diepen et al, 1988
        natural_crop_groups: ArrayFloat32 = self.hydrology.grid.load(
            self.model.files["grid"]["soil/crop_group"]
        )
        self.HRU.var.natural_crop_groups: ArrayFloat32 = self.hydrology.to_HRU(
            data=natural_crop_groups
        )

    def step(
        self,
    ) -> tuple[
        TwoDArrayFloat32,
        TwoDArrayFloat32,
        TwoDArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        float,
        ArrayFloat32,
        ArrayFloat32,
        np.float64,
    ]:
        """Step function for the land surface module.

        Currently, this function calculates the reference evapotranspiration
        for grass and water surfaces using meteorological data.

        Returns:
            A tuple containing:
            - snow_melt_m: Snow melt in meters.
            - rain_m: Rainfall in meters.
            - sublimation_m: Sublimation in meters.

        Raises:
            AssertionError: If any of the debug assertions fail.
        """
        if __debug__:
            snow_water_equivalent_prev: ArrayFloat32 = (
                self.HRU.var.snow_water_equivalent_m.copy()
            )
            liquid_water_in_snow_prev: ArrayFloat32 = (
                self.HRU.var.liquid_water_in_snow_m.copy()
            )
            interception_storage_prev: ArrayFloat32 = (
                self.HRU.var.interception_storage_m.copy()
            )
            topwater_m_prev: ArrayFloat32 = self.HRU.var.topwater_m.copy()
            snow_temperature_C_prev: ArrayFloat32 = (
                self.HRU.var.snow_temperature_C.copy()
            )
            soil_temperature_C_prev: TwoDArrayFloat32 = (
                self.HRU.var.soil_temperature_C.copy()
            )
            wetting_front_depth_prev: ArrayFloat32 = (
                self.HRU.var.wetting_front_depth_m.copy()
            )
            wetting_front_suction_head_prev: ArrayFloat32 = (
                self.HRU.var.wetting_front_suction_head_m.copy()
            )
            wetting_front_moisture_deficit_prev: ArrayFloat32 = (
                self.HRU.var.wetting_front_moisture_deficit.copy()
            )
            green_ampt_active_layer_idx_prev: ArrayInt32 = (
                self.HRU.var.green_ampt_active_layer_idx.copy()
            )
            w_prev: TwoDArrayFloat32 = self.HRU.var.w.copy()

        forest_crop_factor = self.hydrology.to_HRU(
            data=self.grid.compress(
                self.grid.var.forest_crop_factor_per_10_days[
                    (self.model.current_day_of_year - 1) // 10
                ]
            ),
            fn=None,
        )

        crop_stage_lenghts = np.column_stack(
            [
                self.model.agents.crop_farmers.var.crop_data["l_ini"],
                self.model.agents.crop_farmers.var.crop_data["l_dev"],
                self.model.agents.crop_farmers.var.crop_data["l_mid"],
                self.model.agents.crop_farmers.var.crop_data["l_late"],
            ]
        )

        get_crop_sub_stage = self.model.agents.crop_farmers.var.crop_data_type == "GAEZ"
        if get_crop_sub_stage:
            crop_sub_stage_lengths = np.column_stack(
                [
                    self.model.agents.crop_farmers.var.crop_data["d1"],
                    self.model.agents.crop_farmers.var.crop_data["d2a"],
                    self.model.agents.crop_farmers.var.crop_data["d2b"],
                    self.model.agents.crop_farmers.var.crop_data["d3a"],
                    self.model.agents.crop_farmers.var.crop_data["d3b"],
                    self.model.agents.crop_farmers.var.crop_data["d4"],
                ]
            )
        else:
            crop_sub_stage_lengths = np.full(
                (self.model.agents.crop_farmers.var.crop_data.shape[0], 6),
                np.nan,
                dtype=np.float32,
            )

        crop_factor_per_crop_stage = np.column_stack(
            [
                self.model.agents.crop_farmers.var.crop_data["kc_initial"],
                self.model.agents.crop_farmers.var.crop_data["kc_mid"],
                self.model.agents.crop_farmers.var.crop_data["kc_end"],
            ]
        )

        crop_root_depths = np.column_stack(
            [
                self.model.agents.crop_farmers.var.crop_data["rd_rain"],
                self.model.agents.crop_farmers.var.crop_data["rd_irr"],
            ]
        )

        crop_factor, root_depth_m, crop_sub_stage = get_crop_factors_and_root_depths(
            land_use_map=self.HRU.var.land_use_type,
            crop_factor_forest_map=forest_crop_factor,
            crop_map=self.HRU.var.crop_map,
            crop_age_days_map=self.HRU.var.crop_age_days_map,
            crop_harvest_age_days=self.HRU.var.crop_harvest_age_days,
            crop_stage_lengths=crop_stage_lenghts,
            crop_sub_stage_lengths=crop_sub_stage_lengths,
            crop_factor_per_crop_stage=crop_factor_per_crop_stage,
            crop_root_depths=crop_root_depths,
            get_crop_sub_stage=get_crop_sub_stage,
        )

        crop_factor *= self.model.config["parameters"][
            "crop_factor_multiplier"
        ]  # calibration parameter

        interception_capacity_m_forest_HRU = self.hydrology.to_HRU(
            data=self.grid.var.interception_capacity_forest[
                (self.model.current_day_of_year - 1) // 10
            ],
            fn=None,
        )
        interception_capacity_m_grassland_HRU = self.hydrology.to_HRU(
            data=self.grid.var.interception_capacity_grassland[
                (self.model.current_day_of_year - 1) // 10
            ],
            fn=None,
        )

        interception_capacity_m: ArrayFloat32 = get_interception_capacity(
            land_use_type=self.HRU.var.land_use_type,
            interception_capacity_m_forest_HRU=interception_capacity_m_forest_HRU,
            interception_capacity_m_grassland_HRU=interception_capacity_m_grassland_HRU,
        )

        pr_kg_per_m2_per_s = self.HRU.pr_kg_per_m2_per_s
        pr_total_m3 = (
            (
                pr_kg_per_m2_per_s.astype(np.float64).mean(axis=0)
                * self.HRU.var.cell_area
            ).sum()  # kg/s
            * 0.001  # to m3/s
            * (24 * 3600.0)  # to m3/day
        )

        (
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow_m,  # from all sources
            irrigation_loss_to_evaporation_m,
            total_water_demand_loss_m3,
            actual_irrigation_consumption_m,
        ) = self.hydrology.water_demand.step(root_depth_m)

        # Obtain capillary rise for the HRUs
        capillar_rise_m = self.hydrology.to_HRU(data=self.grid.var.capillar, fn=None)
        if capillar_rise_m.sum() > 0.0:
            raise NotImplementedError(
                "Capillary rise is not implemented in the land surface model yet."
            )

        # TODO: pre-compute this once only
        delta_z = (
            self.HRU.var.soil_layer_height_m[:-1, :]
            + self.HRU.var.soil_layer_height_m[1:, :]
        ) / 2

        land_surface_inputs: LandSurfaceInputs = self._build_land_surface_inputs(
            root_depth_m=root_depth_m,
            interception_capacity_m=interception_capacity_m,
            pr_kg_per_m2_per_s=pr_kg_per_m2_per_s,
            crop_factor=crop_factor,
            actual_irrigation_consumption_m=actual_irrigation_consumption_m,
            capillar_rise_m=capillar_rise_m,
            delta_z=delta_z,
        )

        (
            rain_m,
            snow_m,
            self.HRU.var.topwater_m,
            reference_evapotranspiration_grass_m,
            reference_evapotranspiration_water_m,
            self.HRU.var.snow_water_equivalent_m,
            self.HRU.var.liquid_water_in_snow_m,
            sublimation_or_deposition_m,
            self.HRU.var.snow_temperature_C,
            self.HRU.var.interception_storage_m,
            interception_evaporation_m,
            open_water_evaporation_m,
            runoff_m,
            groundwater_recharge_m,
            interflow_m,
            bare_soil_evaporation_m,
            transpiration_m,
            potential_transpiration_m,
        ) = land_surface_model(**land_surface_inputs._asdict())

        if not balance_check(
            name="land surface 1",
            how="cellwise",
            influxes=[
                pr_kg_per_m2_per_s.sum(axis=0)
                * 3.6,  # from kg/m2/s to m/hr and sum over hours to make it day (but then re-arranged for efficiency)
                actual_irrigation_consumption_m,
                capillar_rise_m,
            ],
            outfluxes=[
                -sublimation_or_deposition_m,
                interception_evaporation_m,
                open_water_evaporation_m,
                runoff_m.sum(axis=0),  # sum over hours to make it day
                interflow_m.sum(axis=0),  # sum over hours to make it day
                groundwater_recharge_m,
                bare_soil_evaporation_m,
                transpiration_m,
            ],
            prestorages=[
                snow_water_equivalent_prev,
                liquid_water_in_snow_prev,
                interception_storage_prev,
                topwater_m_prev,
                np.nansum(w_prev, axis=0),
            ],
            poststorages=[
                self.HRU.var.snow_water_equivalent_m,
                self.HRU.var.liquid_water_in_snow_m,
                self.HRU.var.interception_storage_m,
                self.HRU.var.topwater_m,
                np.nansum(self.HRU.var.w, axis=0),
            ],
            tolerance=1e-5,
            raise_on_error=False,
        ):
            error_inputs: LandSurfaceInputs = self._snapshot_land_surface_inputs_for_error(
                land_surface_inputs=land_surface_inputs,
                w_prev=w_prev,
                topwater_m_prev=topwater_m_prev,
                snow_water_equivalent_prev=snow_water_equivalent_prev,
                liquid_water_in_snow_prev=liquid_water_in_snow_prev,
                snow_temperature_C_prev=snow_temperature_C_prev,
                interception_storage_prev=interception_storage_prev,
                soil_temperature_C_prev=soil_temperature_C_prev,
                wetting_front_depth_prev=wetting_front_depth_prev,
                wetting_front_suction_head_prev=wetting_front_suction_head_prev,
                wetting_front_moisture_deficit_prev=wetting_front_moisture_deficit_prev,
                green_ampt_active_layer_idx_prev=green_ampt_active_layer_idx_prev,
            )
            np.savez(
                self.model.diagnostics_folder / "landsurface_model_error.npz",
                **error_inputs._asdict(),
            )
            raise AssertionError("Land surface water balance check failed.")

        actual_evapotranspiration_m = (
            interception_evaporation_m
            + open_water_evaporation_m
            + transpiration_m
            + bare_soil_evaporation_m
            + irrigation_loss_to_evaporation_m
        )

        runoff_m_daily = runoff_m.sum(axis=0)

        growing_crop_mask = self.HRU.var.crop_map != -1

        self.HRU.var.transpiration_crop_life[growing_crop_mask] += transpiration_m[
            growing_crop_mask
        ]
        self.HRU.var.potential_transpiration_crop_life[growing_crop_mask] += (
            potential_transpiration_m[growing_crop_mask]
        )
        self.HRU.var.transpiration_crop_life_per_crop_stage[
            crop_sub_stage[growing_crop_mask], growing_crop_mask
        ] += transpiration_m[growing_crop_mask]
        self.HRU.var.potential_transpiration_crop_life_per_crop_stage[
            crop_sub_stage[growing_crop_mask], growing_crop_mask
        ] += potential_transpiration_m[growing_crop_mask]

        reference_evapotranspiration_water_m = self.hydrology.to_grid(
            HRU_data=reference_evapotranspiration_water_m,
            fn="weightedmean",
        )
        assert (reference_evapotranspiration_water_m >= 0).all()

        self.model.agents.crop_farmers.save_water_deficit(
            reference_evapotranspiration_grass_m, pr_kg_per_m2_per_s
        )
        self.model.agents.crop_farmers.save_pr(pr_kg_per_m2_per_s)
        self.report(locals())

        return (
            reference_evapotranspiration_water_m,
            interflow_m,
            runoff_m,
            groundwater_recharge_m,
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow_m,
            total_water_demand_loss_m3,
            actual_evapotranspiration_m,
            sublimation_or_deposition_m,
            pr_total_m3,
        )
