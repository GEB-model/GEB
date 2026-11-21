"""Land surface module for GEB.

Computational approach
----------------------
The land surface is a set of independent vertical columns, one per hydrological
response unit (HRU). Within a time step no water or energy moves between columns;
lateral exchange is handled before or after, outside the column physics. The
problem is therefore embarrassingly parallel, and the full daily step is one
compiled kernel with a parallel loop over the columns:

- Each thread runs the whole day of one column, all processes and hours, so its
  state stays in the cache.
- Process functions are inlined into the kernel, so there is no call overhead and
  the compiler optimises one loop body.
- Single precision throughout; double precision only where small fluxes on large
  stores would be lost.
- No memory allocation inside the loop; small work arrays live on the stack.
- Memory is laid out per column, so the data of one column is contiguous.
- Stiff vertical processes are solved implicitly per column, which keeps them
  stable at the model time step without sub-stepping.
- Random elements use a stateless hash of time and column, so results do not
  depend on the number of threads.
- Water and energy balance checks run in parallel kernels in debug mode only.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from numba import njit, prange

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt8,
    ArrayInt32,
    ThreeDArrayFloat32,
    TwoDArrayBool,
    TwoDArrayFloat32,
    TwoDArrayFloat64,
)
from geb.module import Module
from geb.numba_config import NUMBA_CACHE
from geb.store import Bucket
from geb.workflows import TimingModule
from geb.workflows.dt import map_date_to_dekad
from geb.workflows.io import read_grid
from geb.workflows.numba_stack_array import stack_empty

from ..landcovers import (
    FOREST,
    GRASSLAND_LIKE,
    NON_PADDY_IRRIGATED,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
)
from .balance_checks import (
    ERR_NAN_EVAP_TRANS,
    ERR_NAN_GW_RECHARGE,
    ERR_NAN_INTERCEPTION,
    ERR_NAN_INTERFLOW,
    ERR_NAN_LIQUID_SNOW,
    ERR_NAN_RUNOFF,
    ERR_NAN_SWE,
    ERR_NAN_TOPWATER,
    ERR_NAN_WATER_CONTENT,
    ERR_NEG_BARE_SOIL,
    ERR_NEG_GW_RECHARGE,
    ERR_NEG_INTERCEPTION,
    ERR_NEG_INTERCEPTION_EVAP,
    ERR_NEG_INTERFLOW,
    ERR_NEG_LIQUID_SNOW,
    ERR_NEG_OPEN_WATER_EVAP,
    ERR_NEG_RUNOFF,
    ERR_NEG_SWE,
    ERR_NEG_TOPWATER,
    ERR_NEG_TRANSPIRATION,
    ERR_NEG_WATER_CONTENT,
    ERR_NONE,
    ERR_WATER_IMBALANCE,
    check_soil_enthalpy_balance_kernel,
    validate_and_check_water_balance_kernel,
)
from .constants import (
    CAPILLARY_REACH_M,
    DEEP_SOIL_ALPHA,
    DEEP_SOIL_ONE_MINUS_ALPHA,
    KELVIN_OFFSET,
    N_SNOW_LAYERS,
    N_SOIL_LAYERS,
    N_SOILGRIDS_LAYERS,
    N_STATIC_SOIL_PROPS,
    ORGANIC_MATTER_PER_ORGANIC_CARBON,
    REGOLITH_LAYER_INDEX,
    REGOLITH_LAYER_MIN_THICKNESS_M,
    REGOLITH_MIN_ROCK_FRAGMENTS_PERCENTAGE,
    RHO_MINERAL_KG_PER_M3,
    RHO_WATER_KG_PER_M3,
    SOIL_ALBEDO,
    SOIL_EMISSIVITY,
    SOIL_PROP_BUBBLING_PRESSURE_M_POSITIVE,
    SOIL_PROP_HORIZONTAL_SATURATED_HYDRAULIC_CONDUCTIVITY_M_PER_S,
    SOIL_PROP_LAMBDA_PORE_SIZE_DISTRIBUTION,
    SOIL_PROP_SAND_PERCENTAGE,
    SOIL_PROP_SOIL_LAYER_HEIGHT_M,
    SOIL_PROP_SOLID_HEAT_CAPACITY_J_PER_M2_K,
    SOIL_PROP_THERMAL_CONDUCTIVITY_DRY_SOIL_W_PER_M_K,
    SOIL_PROP_THERMAL_CONDUCTIVITY_SATURATED_FROZEN_W_PER_M_K,
    SOIL_PROP_THERMAL_CONDUCTIVITY_SATURATED_UNFROZEN_W_PER_M_K,
    SOIL_PROP_VERTICAL_SATURATED_HYDRAULIC_CONDUCTIVITY_M_PER_S,
    SOIL_PROP_WATER_CONTENT_FIELD_CAPACITY_M,
    SOIL_PROP_WATER_CONTENT_RESIDUAL_M,
    SOIL_PROP_WATER_CONTENT_SATURATED_M,
    SOIL_PROP_WATER_CONTENT_WILTING_POINT_M,
    SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K,
    THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN,
    THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN,
)
from .energy import (
    apply_evaporative_cooling,
    calculate_thermal_conductivity_dry_soil_johansen_watt_per_meter_kelvin,
    calculate_thermal_conductivity_saturated_soil_johansen_watt_per_meter_kelvin,
    calculate_thermal_conductivity_solid_fraction_watt_per_meter_kelvin,
    get_frozen_fraction_from_enthalpy,
    get_heat_capacity_solid_fraction,
    get_soil_heat_flux_net_of_evaporation,
    get_temperature_and_frozen_fraction_from_enthalpy_scalar,
    get_temperature_from_enthalpy,
    solve_soil_enthalpy_column,
)
from .evapotranspiration import (
    accumulate_crop_evapotranspiration,
    calculate_bare_soil_evaporation,
    calculate_transpiration,
    get_dew,
    limit_evaporation_under_snow,
    resolve_crop_group_number,
)
from .interception import (
    interception,
)
from .potential_evapotranspiration import (
    GROWING_SEASON_AVERAGING_DAYS,
    HourlyWeather,
    get_CO2_induced_crop_factor_adustment,
    get_crop_factors_and_root_depths_and_lai,
    get_daily_wind_and_minimum_relative_humidity,
    get_evergreen_leaf_area_index,
    get_forest_leaf_area_index_with_evergreen,
    get_hourly_weather,
    get_potential_evapotranspiration,
    get_reference_evapotranspiration,
    get_wet_forest_canopy_evaporation,
    update_canopy_temperature_acclimation,
    update_crop_heat_units,
)
from .redistribution import (
    distribute_soil_water_ross,
)
from .snow_glaciers import (
    FRESH_SNOW_DENSITY_KG_PER_M3,
    SNOW_ALBEDO_FRESH,
    apply_precipitation_compaction_and_top_layer_transfer,
    get_snow_holding_capacity_m,
    redistribute_snow_along_drainage_direction,
    update_snow_albedo,
    update_snow_mass_and_phase,
)
from .water import (
    FORCING_SOURCE_SIGMA_DEFAULTS,
    add_water_to_topwater_and_evaporate_open_water,
    generate_rainfall_lookup_table,
    get_bubbling_pressure_m_positive,
    get_bulk_density_with_organic_matter,
    get_coarse_fragment_factors,
    get_hillslope_store_filling,
    get_macropore_conductivity_jarvis,
    get_pore_size_index_brakensiek,
    get_regolith_layer_thickness_m,
    get_regolith_vertical_conductivity_factor,
    get_slope_soil_depth_factor,
    get_soil_cover_fraction,
    get_soil_moisture_at_pressure,
    get_surface_crust_factor,
    infiltration,
    kv_wosten,
    thetar_brakensiek,
    thetas_toth,
)

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology


@njit(parallel=True, cache=NUMBA_CACHE, fastmath=True)
def land_surface_model(
    unix_time_seconds: np.int64,
    land_use_type: ArrayInt32,
    slope_m_per_m: ArrayFloat32,
    hillslope_length_m: ArrayFloat32,
    water_content_m: TwoDArrayFloat32,
    packed_soil_properties: ThreeDArrayFloat32,
    soil_enthalpy_J_per_m2: TwoDArrayFloat32,
    root_depth_m: ArrayFloat32,
    topwater_m: ArrayFloat32,
    variable_runoff_shape_beta: ArrayFloat32,
    snow_water_equivalent_m: TwoDArrayFloat64,
    liquid_water_in_snow_m: TwoDArrayFloat64,
    snow_enthalpy_J_per_m2: TwoDArrayFloat32,
    snow_density_kg_per_m3: TwoDArrayFloat32,
    snow_albedo: ArrayFloat32,
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
    basal_crop_factor: ArrayFloat32,
    maximum_crop_factor: ArrayFloat32,
    actual_irrigation_consumption_m: ArrayFloat32,
    groundwater_depth_previous_day_m: ArrayFloat32,
    groundwater_depth_m: ArrayFloat32,
    groundwater_toplayer_conductivity_m_per_s: ArrayFloat32,
    wetting_front_depth_m: ArrayFloat32,
    wetting_front_suction_head_m: ArrayFloat32,
    wetting_front_moisture_deficit: ArrayFloat32,
    crust_rain_m: ArrayFloat32,
    green_ampt_active_layer_idx: ArrayInt32,
    crop_group_number: ArrayFloat32,
    minimum_effective_root_depth_m: np.float32,
    interflow_structural_conductivity_m_per_s: ArrayFloat32,
    deep_soil_temperature_C: ArrayFloat32,
    leaf_area_index: ArrayFloat32,
    rainfall_lookup_table: TwoDArrayFloat32,
    daily_reference_evapotranspiration_grass_m: ArrayFloat32,
    max_capillary_rise_m: ArrayFloat32,
    HRU_to_grid: ArrayInt32,
    interflow_structural_depth_m: np.float32,
    macropore_infiltration_capacity_m_per_s: ArrayFloat32,
    bare_soil_crust_factor: np.float32,
    crust_formation_rain_m: np.float32,
) -> tuple[
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    TwoDArrayFloat32,
    TwoDArrayFloat64,
    TwoDArrayFloat64,
    ArrayFloat32,
    TwoDArrayFloat32,
    TwoDArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    TwoDArrayFloat32,
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
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    ArrayFloat32,
    TwoDArrayFloat32,
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
        unix_time_seconds: Current model time in seconds since the Unix epoch (January 1, 1970). Can be negative for dates before the epoch.
        land_use_type: Land use type of the hydrological response unit.
        slope_m_per_m: Slope of the hydrological response unit in m/m.
        hillslope_length_m: Hillslope length of the hydrological response unit in m.
        water_content_m: Soil water storage per layer (m), i.e. equivalent water depth in the layer.
        packed_soil_properties: Packed static 2D soil properties array of shape
            (num_cells, N_STATIC_SOIL_PROPS, N_SOIL_LAYERS), containing residual water content (m),
            wilting point (m), field capacity (m), saturation capacity (m), solid heat capacity (J/m2/K),
            saturated unfrozen thermal conductivity (W/m/K), saturated frozen thermal conductivity (W/m/K),
            dry soil thermal conductivity (W/m/K), sand percentage (%), soil layer height (m),
            bubbling pressure (m), pore size distribution parameter lambda (-), and horizontal and vertical
            saturated hydraulic conductivities (m/s).
        soil_enthalpy_J_per_m2: Soil enthalpy per layer (J/m2), relative to 0°C liquid water.
        root_depth_m: Root depth for the cell in meters.
        topwater_m: Topwater in meters, which is >=0 for paddy and 0 for non-paddy. Within
            this function topwater is used to add water from natural infiltration and
            irrigation and to calculate open water evaporation.
        variable_runoff_shape_beta: Variable infiltration capacity runoff model shape parameter for the cell.
        snow_water_equivalent_m: Snow water equivalent per layer in meters (shape: [n_cells, N_SNOW_LAYERS]).
        liquid_water_in_snow_m: Liquid water in snow per layer in meters (shape: [n_cells, N_SNOW_LAYERS]).
        snow_enthalpy_J_per_m2: Snow enthalpy per layer in J/m2 (shape: [n_cells, N_SNOW_LAYERS]).
        snow_density_kg_per_m3: Snow bulk density per layer in kg/m3 (shape: [n_cells, N_SNOW_LAYERS]).
        snow_albedo: Albedo of the snowpack per cell (-), updated in place as snow ages.
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
        crop_factor: Crop factor (FAO-56 crop coefficient for a wet soil surface) for each HRU. Dimensionless.
        basal_crop_factor: Basal crop factor (transpiration part of the crop factor) for each HRU. Dimensionless.
        maximum_crop_factor: Upper limit of the crop factor of a wet surface (FAO-56 Kc_max) for each HRU. Dimensionless.
        actual_irrigation_consumption_m: Actual irrigation consumption in meters.
        groundwater_depth_previous_day_m: Depth of the groundwater table below the surface of
            each HRU at the start of the previous day (m).
        groundwater_depth_m: Depth of the groundwater table below the surface of each HRU at
            the start of the day (m). The soil follows the water table one day behind: over
            the hours of the day the depth moves from the previous day's depth to this one,
            so the water table, and the water it releases to or takes from the soil, changes
            smoothly instead of in a step at midnight.
        groundwater_toplayer_conductivity_m_per_s: Groundwater top layer conductivity in m/s (grid-level, indexed via HRU_to_grid).
        wetting_front_depth_m: Wetting front depth in meters.
        wetting_front_suction_head_m: Wetting front suction head [m].
        wetting_front_moisture_deficit: Moisture deficit at the wetting front [-].
        crust_rain_m: Rain that fell on the exposed soil of the HRU since its surface was
            last loosened by frost or tillage (m); updated in place.
        green_ampt_active_layer_idx: Index of the active soil layer for Green-Ampt infiltration.
        crop_group_number: Crop group number per HRU, pre-resolved from land use type and crop map (see WOFOST 6.0).
        minimum_effective_root_depth_m: Minimum effective root depth in meters.
        interflow_structural_conductivity_m_per_s: Lateral conductivity of the soil structure at the surface per HRU (m/s).
        interflow_structural_depth_m: e-folding depth (m) of the structural lateral conductivity.
        macropore_infiltration_capacity_m_per_s: Macropore infiltration capacity per HRU (m/s); None = no macropore infiltration.
        bare_soil_crust_factor: Factor on the surface conductivity of bare soil with a
            fully developed crust.
        crust_formation_rain_m: Rain on exposed soil over which the crust develops (m).
        deep_soil_temperature_C: Deep soil temperature in Celsius.
        leaf_area_index: Leaf area index for the cell.
        rainfall_lookup_table: Precomputed lognormal weights lookup table of shape (table_size, 6).
        daily_reference_evapotranspiration_grass_m: Daily reference evapotranspiration for grass (meters).
        max_capillary_rise_m: Maximum available groundwater storage for capillary rise per cell in meters (grid-level, indexed via HRU_to_grid).
        HRU_to_grid: Mapping from HRU cell index to compressed grid index. If empty, forcing arrays are indexed directly by HRU index.

    Returns:
        Tuple of:
        - rain_m: Precipitation reaching the surface as rain (m).
        - snow_m: Precipitation reaching the surface as snow (m).
        - topwater_m: Updated depth of surface water (m).
        - reference_evapotranspiration_grass_m: Reference evapotranspiration for grass (m).
        - potential_open_water_evaporation_m: Potential evaporation of open water per hour (m).
        - snow_water_equivalent_m: Snow water equivalent (m).
        - liquid_water_in_snow_m: Liquid water in snow (m).
        - sublimation_m: Sublimation (m).
        - snow_enthalpy_J_per_m2: Snow enthalpy (J/m2).
        - snow_density_kg_per_m3: Snow density (kg/m3).
        - interception_storage_m: Interception storage (m).
        - interception_evaporation_m: Evaporation from interception storage, net of
            dew (m).
        - open_water_evaporation_m: Evaporation from open water (m).
        - runoff_m: Surface runoff (m).
        - groundwater_recharge_m: Groundwater recharge (m).
        - interflow_m: Lateral subsurface flow (m).
        - bare_soil_evaporation: Evaporation from bare soil (m).
        - transpiration_m: Total transpiration (m).
        - potential_transpiration_m: Potential transpiration (m).
        - potential_evapotranspiration_m: Potential evapotranspiration (m).
        - soil_boundary_enthalpy_flux_J_per_m2: Net enthalpy flux entering the soil
            from the boundary (J/m2).
        - rain_advection_enthalpy_flux_J_per_m2: Enthalpy carried into the soil by
            rain (J/m2).
        - evaporative_cooling_enthalpy_loss_J_per_m2: Enthalpy loss due to
            evaporative cooling (J/m2).
        - interflow_enthalpy_loss_J_per_m2: Enthalpy loss carried by interflow (J/m2).
        - groundwater_recharge_enthalpy_loss_J_per_m2: Enthalpy loss carried by
            groundwater recharge (J/m2).
        - transpiration_enthalpy_loss_J_per_m2: Enthalpy loss carried by
            transpiration (J/m2).
        - top_soil_infiltration_m: Infiltration into the top soil layer (m).
        - top_soil_rise_from_layer_2_m: Upward redistribution from the second soil
            layer into the top soil layer (m).
        - top_soil_percolation_to_layer_2_m: Percolation from the top soil layer to
            layer 2 (m).
        - top_soil_transpiration_m: Transpiration extracted specifically from the
            top soil layer (m).
        - evapotranspiration_m: Actual hourly evapotranspiration (m/hour).
        - dew_m: Dew condensed on the vegetation and soil (m); included as negative
            evaporation in interception_evaporation_m.
    """
    CO2_induced_crop_factor_adustment: np.float32 = np.float32(
        get_CO2_induced_crop_factor_adustment(CO2_ppm)
    )

    # convert values to substep (i.e., per hour)
    actual_irrigation_consumption_m = actual_irrigation_consumption_m / 24.0

    num_cells: int = slope_m_per_m.size
    capillar_rise_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    runoff_m: TwoDArrayFloat32 = np.empty((num_cells, 24), dtype=np.float32)
    interflow_m: TwoDArrayFloat32 = np.empty((num_cells, 24), dtype=np.float32)
    potential_open_water_evaporation_m: TwoDArrayFloat32 = np.empty(
        (num_cells, 24), dtype=np.float32
    )
    evapotranspiration_m: TwoDArrayFloat32 = np.empty((num_cells, 24), dtype=np.float32)

    # total per day variables for water balance
    reference_evapotranspiration_grass_m: ArrayFloat32 = np.empty(
        num_cells, dtype=np.float32
    )
    rain_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    snow_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    sublimation_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    interception_evaporation_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    dew_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    open_water_evaporation_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    bare_soil_evaporation: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    potential_transpiration_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    potential_evapotranspiration_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    transpiration_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    groundwater_recharge_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    top_soil_infiltration_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    top_soil_rise_from_layer_2_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
    top_soil_percolation_to_layer_2_m: ArrayFloat32 = np.empty(
        num_cells, dtype=np.float32
    )
    top_soil_transpiration_m: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)

    # Daily integrated enthalpy flux diagnostics [J/m2].
    # Positive values indicate energy entering the soil enthalpy reservoir.
    soil_boundary_enthalpy_flux_J_per_m2: ArrayFloat32 = np.empty(
        num_cells, dtype=np.float32
    )
    rain_advection_enthalpy_flux_J_per_m2: ArrayFloat32 = np.empty(
        num_cells, dtype=np.float32
    )

    # Positive values indicate energy leaving the soil enthalpy reservoir.
    evaporative_cooling_enthalpy_loss_J_per_m2: ArrayFloat32 = np.empty(
        num_cells, dtype=np.float32
    )
    interflow_enthalpy_loss_J_per_m2: ArrayFloat32 = np.empty(
        num_cells, dtype=np.float32
    )
    groundwater_recharge_enthalpy_loss_J_per_m2: ArrayFloat32 = np.empty(
        num_cells, dtype=np.float32
    )
    transpiration_enthalpy_loss_J_per_m2: ArrayFloat32 = np.empty(
        num_cells, dtype=np.float32
    )

    for i in prange(num_cells):
        grid_i: int = HRU_to_grid[i]
        # Track remaining available groundwater storage across the 24 hourly sub-steps
        # to ensure capillary rise does not overdraw the aquifer during the daily step.
        remaining_capillary_rise_m: np.float32 = max_capillary_rise_m[grid_i]
        # Use stack-allocated arrays to avoid heap allocations in the parallel cell loop
        water_content_before_transpiration_m = stack_empty(N_SOIL_LAYERS, np.float32)
        snow_water_equivalent_layers_m = stack_empty(N_SNOW_LAYERS, np.float64)
        snow_enthalpy_layers_J_per_m2 = stack_empty(N_SNOW_LAYERS, np.float32)
        snow_density_layers_kg_per_m3 = stack_empty(N_SNOW_LAYERS, np.float32)

        # Packed static soil properties views for cell i:
        # Properties across all 14 variables are packed contiguously in memory (num_cells, 14, 6)
        # to ensure that an entire HRU's static properties (84 float32 values = 336 bytes) fit within
        # 6 consecutive cache lines, avoiding 14 scattered pointer dereferences across DRAM per cell.
        wres = packed_soil_properties[i, SOIL_PROP_WATER_CONTENT_RESIDUAL_M]
        wilting_point_m = packed_soil_properties[
            i, SOIL_PROP_WATER_CONTENT_WILTING_POINT_M
        ]
        field_capacity_m = packed_soil_properties[
            i, SOIL_PROP_WATER_CONTENT_FIELD_CAPACITY_M
        ]
        saturated_water_content_m = packed_soil_properties[
            i, SOIL_PROP_WATER_CONTENT_SATURATED_M
        ]
        solid_heat_capacity = packed_soil_properties[
            i, SOIL_PROP_SOLID_HEAT_CAPACITY_J_PER_M2_K
        ]
        thermal_cond_unfrozen = packed_soil_properties[
            i,
            SOIL_PROP_THERMAL_CONDUCTIVITY_SATURATED_UNFROZEN_W_PER_M_K,
        ]
        thermal_cond_frozen = packed_soil_properties[
            i, SOIL_PROP_THERMAL_CONDUCTIVITY_SATURATED_FROZEN_W_PER_M_K
        ]
        thermal_cond_dry = packed_soil_properties[
            i, SOIL_PROP_THERMAL_CONDUCTIVITY_DRY_SOIL_W_PER_M_K
        ]
        sand_percentage_cell = packed_soil_properties[i, SOIL_PROP_SAND_PERCENTAGE]
        soil_layer_height_cell = packed_soil_properties[
            i, SOIL_PROP_SOIL_LAYER_HEIGHT_M
        ]
        bubbling_pressure_cell = packed_soil_properties[
            i, SOIL_PROP_BUBBLING_PRESSURE_M_POSITIVE
        ]
        lambda_pore_size_cell = packed_soil_properties[
            i, SOIL_PROP_LAMBDA_PORE_SIZE_DISTRIBUTION
        ]
        horizontal_conductivity_cell = packed_soil_properties[
            i,
            SOIL_PROP_HORIZONTAL_SATURATED_HYDRAULIC_CONDUCTIVITY_M_PER_S,
        ]
        vertical_conductivity_cell = packed_soil_properties[
            i, SOIL_PROP_VERTICAL_SATURATED_HYDRAULIC_CONDUCTIVITY_M_PER_S
        ]

        snow_water_equivalent_top_m_cell = snow_water_equivalent_m[i, 0]
        snow_water_equivalent_bottom_m_cell = snow_water_equivalent_m[i, 1]
        liquid_water_top_m_cell = liquid_water_in_snow_m[i, 0]
        liquid_water_bottom_m_cell = liquid_water_in_snow_m[i, 1]
        snow_enthalpy_top_J_per_m2_cell = snow_enthalpy_J_per_m2[i, 0]
        snow_enthalpy_bottom_J_per_m2_cell = snow_enthalpy_J_per_m2[i, 1]
        snow_density_top_kg_per_m3_cell = snow_density_kg_per_m3[i, 0]
        snow_density_bottom_kg_per_m3_cell = snow_density_kg_per_m3[i, 1]

        topwater_m_cell: np.float32 = topwater_m[i]
        interception_storage_m_cell: np.float32 = interception_storage_m[i]

        # Initialize daily outputs inside prange for NUMA first-touch placement
        capillar_rise_m[i] = np.float32(0.0)
        reference_evapotranspiration_grass_m[i] = np.float32(0.0)
        rain_m[i] = np.float32(0.0)
        snow_m[i] = np.float32(0.0)
        sublimation_m[i] = np.float32(0.0)
        interception_evaporation_m[i] = np.float32(0.0)
        dew_m[i] = np.float32(0.0)
        open_water_evaporation_m[i] = np.float32(0.0)
        bare_soil_evaporation[i] = np.float32(0.0)
        potential_transpiration_m[i] = np.float32(0.0)
        potential_evapotranspiration_m[i] = np.float32(0.0)
        transpiration_m[i] = np.float32(0.0)
        groundwater_recharge_m[i] = np.float32(0.0)
        top_soil_infiltration_m[i] = np.float32(0.0)
        top_soil_rise_from_layer_2_m[i] = np.float32(0.0)
        top_soil_percolation_to_layer_2_m[i] = np.float32(0.0)
        top_soil_transpiration_m[i] = np.float32(0.0)
        soil_boundary_enthalpy_flux_J_per_m2[i] = np.float32(0.0)
        rain_advection_enthalpy_flux_J_per_m2[i] = np.float32(0.0)
        evaporative_cooling_enthalpy_loss_J_per_m2[i] = np.float32(0.0)
        interflow_enthalpy_loss_J_per_m2[i] = np.float32(0.0)
        groundwater_recharge_enthalpy_loss_J_per_m2[i] = np.float32(0.0)
        transpiration_enthalpy_loss_J_per_m2[i] = np.float32(0.0)

        # Update deep soil temperature boundary condition using exponential moving average of 2m air temperature
        temperature_sum_K: np.float32 = np.float32(0.0)
        for h in range(24):
            temperature_sum_K += tas_2m_K[grid_i, h]
        mean_daily_temperature_C: np.float32 = (
            temperature_sum_K / np.float32(24.0)
        ) - KELVIN_OFFSET
        deep_soil_temperature_C[i] = (
            deep_soil_temperature_C[i] * DEEP_SOIL_ONE_MINUS_ALPHA
            + mean_daily_temperature_C * DEEP_SOIL_ALPHA
        )

        # Soil and ponded-water evaporation of the previous hour, whose latent heat leaves
        # the soil surface (apply_evaporative_cooling). Zero at midnight: no evaporation
        # in the last hour of the night, so no state is carried between days.
        soil_surface_evaporation_previous_hour_m: np.float32 = np.float32(0.0)

        for hour in range(24):
            # Climate values for current hour (direct grid lookup [grid_i, hour])
            tas_2m_K_this_hour = tas_2m_K[grid_i, hour]
            ps_pascal_this_hour = ps_pascal[grid_i, hour]
            rlds_W_per_m2_this_hour = rlds_W_per_m2[grid_i, hour]
            rsds_W_per_m2_this_hour = rsds_W_per_m2[grid_i, hour]
            pr_kg_per_m2_per_s_this_hour = pr_kg_per_m2_per_s[grid_i, hour]

            tas_C: np.float32 = tas_2m_K_this_hour - KELVIN_OFFSET
            dewpoint_tas_C: np.float32 = dewpoint_tas_2m_K[grid_i, hour] - KELVIN_OFFSET

            wind_u = wind_u10m_m_per_s[grid_i, hour]
            wind_v = wind_v10m_m_per_s[grid_i, hour]
            wind_10m_m_per_s: np.float32 = np.sqrt(
                wind_u * wind_u + wind_v * wind_v
            )  # Wind speed at 10m height

            # Weather terms shared by snow and the evaporation of all surfaces
            weather: HourlyWeather = get_hourly_weather(
                tas_C=tas_C,
                dewpoint_tas_C=dewpoint_tas_C,
                ps_pa=ps_pascal_this_hour,
                rsds_W_per_m2=rsds_W_per_m2_this_hour,
                rlds_W_per_m2=rlds_W_per_m2_this_hour,
                wind_10m_m_per_s=wind_10m_m_per_s,
            )

            (
                rain_m_cell,
                snow_m_cell,
                snow_water_equivalent_top_m_cell,
                liquid_water_top_m_cell,
                snow_enthalpy_top_J_per_m2_cell,
                snow_density_top_kg_per_m3_cell,
                snow_water_equivalent_bottom_m_cell,
                liquid_water_bottom_m_cell,
                snow_enthalpy_bottom_J_per_m2_cell,
                snow_density_bottom_kg_per_m3_cell,
            ) = apply_precipitation_compaction_and_top_layer_transfer(
                pr_kg_per_m2_per_s=pr_kg_per_m2_per_s_this_hour,
                air_temperature_C=tas_C,
                relative_humidity_percent=weather.relative_humidity_percent,
                swe_top_m=snow_water_equivalent_top_m_cell,
                liquid_water_top_m=liquid_water_top_m_cell,
                enthalpy_top_J_per_m2=snow_enthalpy_top_J_per_m2_cell,
                density_top_kg_per_m3=snow_density_top_kg_per_m3_cell,
                swe_bottom_m=snow_water_equivalent_bottom_m_cell,
                liquid_water_bottom_m=liquid_water_bottom_m_cell,
                enthalpy_bottom_J_per_m2=snow_enthalpy_bottom_J_per_m2_cell,
                density_bottom_kg_per_m3=snow_density_bottom_kg_per_m3_cell,
                compaction_timestep_s=(
                    np.float32(86400.0) if hour == 0 else np.float32(0.0)
                ),
            )

            rain_m[i] += rain_m_cell
            snow_m[i] += snow_m_cell

            snow_water_equivalent_layers_m[0] = snow_water_equivalent_top_m_cell
            snow_water_equivalent_layers_m[1] = snow_water_equivalent_bottom_m_cell
            snow_enthalpy_layers_J_per_m2[0] = snow_enthalpy_top_J_per_m2_cell
            snow_enthalpy_layers_J_per_m2[1] = snow_enthalpy_bottom_J_per_m2_cell
            snow_density_layers_kg_per_m3[0] = snow_density_top_kg_per_m3_cell
            snow_density_layers_kg_per_m3[1] = snow_density_bottom_kg_per_m3_cell

            snow_albedo[i] = update_snow_albedo(
                snow_albedo=snow_albedo[i],
                snowfall_m=snow_m_cell,
                total_snow_water_equivalent_m=np.float32(
                    snow_water_equivalent_top_m_cell
                    + snow_water_equivalent_bottom_m_cell
                ),
                is_melting=bool(
                    liquid_water_top_m_cell > 0.0 or tas_C > np.float32(0.0)
                ),
                timestep_s=np.float32(3600.0),
            )
            snow_albedo_cell: np.float32 = snow_albedo[i]

            soil_enthalpy_before_solver_J_per_m2 = np.float32(0)
            for layer in range(N_SOIL_LAYERS):
                soil_enthalpy_before_solver_J_per_m2 += soil_enthalpy_J_per_m2[i, layer]

            (
                soil_heat_flux_W_per_m2_cell,
                top_layer_frozen_fraction,
            ) = solve_soil_enthalpy_column(
                soil_enthalpies_J_per_m2=soil_enthalpy_J_per_m2[i, :],
                layer_thicknesses_m=soil_layer_height_cell,
                thermal_conductivity_dry_soil_W_per_m_K=thermal_cond_dry,
                solid_heat_capacities_J_per_m2_K=solid_heat_capacity,
                thermal_conductivity_saturated_unfrozen_W_per_m_K=thermal_cond_unfrozen,
                thermal_conductivity_saturated_frozen_W_per_m_K=thermal_cond_frozen,
                water_content_saturated_m=saturated_water_content_m,
                sand_percentage=sand_percentage_cell,
                water_content_m=water_content_m[i, :],
                shortwave_radiation_W_per_m2=rsds_W_per_m2_this_hour,
                longwave_radiation_W_per_m2=rlds_W_per_m2_this_hour,
                air_temperature_K=tas_2m_K_this_hour,
                wind_speed_10m_m_per_s=wind_10m_m_per_s,
                surface_pressure_pa=ps_pascal_this_hour,
                timestep_seconds=np.float32(3600.0),
                deep_soil_temperature_C=deep_soil_temperature_C[i],
                soil_emissivity=SOIL_EMISSIVITY,
                soil_albedo=SOIL_ALBEDO,
                leaf_area_index=leaf_area_index[i],
                snow_water_equivalent_m=snow_water_equivalent_layers_m,
                snow_enthalpy_J_per_m2=snow_enthalpy_layers_J_per_m2,
                snow_density_kg_per_m3=snow_density_layers_kg_per_m3,
                topwater_m=topwater_m_cell,
                snow_albedo=snow_albedo_cell,
            )
            snow_enthalpy_top_J_per_m2_cell = snow_enthalpy_layers_J_per_m2[0]
            snow_enthalpy_bottom_J_per_m2_cell = snow_enthalpy_layers_J_per_m2[1]

            soil_enthalpy_J_per_m2_cell = np.float32(0.0)
            for layer in range(N_SOIL_LAYERS):
                soil_enthalpy_J_per_m2_cell += soil_enthalpy_J_per_m2[i, layer]

            soil_boundary_enthalpy_flux_J_per_m2[i] += (
                soil_enthalpy_J_per_m2_cell - soil_enthalpy_before_solver_J_per_m2
            )

            soil_heat_flux_W_per_m2_cell = get_soil_heat_flux_net_of_evaporation(
                surface_heat_flux_W_per_m2=soil_heat_flux_W_per_m2_cell,
                soil_surface_evaporation_m_per_hour=soil_surface_evaporation_previous_hour_m,
            )

            (
                reference_evapotranspiration_grass_m_hour_cell,
                wet_vegetation_evaporation_m_hour_cell,
                potential_open_water_evaporation_m_hour_cell,
            ) = get_reference_evapotranspiration(
                weather=weather,
                soil_heat_flux_W_per_m2=soil_heat_flux_W_per_m2_cell,
            )

            dew_m_cell_hour: np.float32 = get_dew(
                wet_vegetation_evaporation_m=wet_vegetation_evaporation_m_hour_cell,
                snow_water_equivalent_m=snow_water_equivalent_top_m_cell
                + snow_water_equivalent_bottom_m_cell,
            )
            reference_evapotranspiration_grass_m_hour_cell = max(
                reference_evapotranspiration_grass_m_hour_cell, np.float32(0.0)
            )
            potential_open_water_evaporation_m_hour_cell = max(
                potential_open_water_evaporation_m_hour_cell, np.float32(0.0)
            )

            reference_evapotranspiration_grass_m[i] += (
                reference_evapotranspiration_grass_m_hour_cell
            )
            potential_open_water_evaporation_m[i, hour] = (
                potential_open_water_evaporation_m_hour_cell
            )

            potential_evapotranspiration_m_cell: np.float32 = get_potential_evapotranspiration(
                reference_evapotranspiration_grass_m=reference_evapotranspiration_grass_m_hour_cell,
                crop_factor=crop_factor[i],
                CO2_induced_crop_factor_adustment=CO2_induced_crop_factor_adustment,
            )
            potential_evapotranspiration_m[i] += potential_evapotranspiration_m_cell

            # FAO-56 dual crop coefficient: transpiration from the basal crop factor,
            # the rest of the wet-surface potential is soil evaporation.
            potential_transpiration_m_cell_hour: np.float32 = min(
                get_potential_evapotranspiration(
                    reference_evapotranspiration_grass_m=reference_evapotranspiration_grass_m_hour_cell,
                    crop_factor=basal_crop_factor[i],
                    CO2_induced_crop_factor_adustment=CO2_induced_crop_factor_adustment,
                ),
                potential_evapotranspiration_m_cell,
            )
            potential_direct_evaporation_m: np.float32 = (
                potential_evapotranspiration_m_cell
                - potential_transpiration_m_cell_hour
            )
            # Energy-limited soil evaporation (Kc_max - Kcb) ET0, which the drying of
            # the soil surface reduces (FAO-56 eq. 71, calculate_bare_soil_evaporation).
            energy_limited_soil_evaporation_m: np.float32 = get_potential_evapotranspiration(
                reference_evapotranspiration_grass_m=reference_evapotranspiration_grass_m_hour_cell,
                crop_factor=max(
                    maximum_crop_factor[i] - basal_crop_factor[i], np.float32(0.0)
                ),
                CO2_induced_crop_factor_adustment=CO2_induced_crop_factor_adustment,
            )
            if land_use_type[i] == SEALED:
                # Water on the paved surface (wetted roofs and roads, puddles)
                # evaporates as from open water: a smooth wet surface without surface
                # resistance. The garden soil of the unpaved part only transpires.
                potential_direct_evaporation_m = (
                    potential_open_water_evaporation_m_hour_cell
                )

            # Water on leaves evaporates without stomatal resistance, at the rate of
            # a wet canopy, not limited by the crop-factor potential.
            if land_use_type[i] == FOREST:
                potential_interception_evaporation_m: np.float32 = (
                    get_wet_forest_canopy_evaporation(weather)
                )
            else:
                potential_interception_evaporation_m = max(
                    wet_vegetation_evaporation_m_hour_cell, np.float32(0.0)
                )
            (
                interception_storage_m_cell,
                throughfall_m,
                interception_evaporation_m_cell_hour,
                potential_transpiration_m_cell_hour,
                potential_direct_evaporation_m,
            ) = interception(
                rainfall_m=rain_m_cell,
                # dew condenses on the leaves (excess over the capacity, or all of it
                # without canopy, drips to the surface)
                storage_m=interception_storage_m_cell + dew_m_cell_hour,
                capacity_m=interception_capacity_m[i],
                potential_interception_evaporation_m=potential_interception_evaporation_m,
                potential_transpiration_m=potential_transpiration_m_cell_hour,
                potential_direct_evaporation_m=potential_direct_evaporation_m,
                leaf_area_index=leaf_area_index[i],
            )

            # dew is booked as negative evaporation: net latent heat flux, as measured
            # by flux towers
            interception_evaporation_m_cell_hour -= dew_m_cell_hour
            interception_evaporation_m[i] += interception_evaporation_m_cell_hour
            dew_m[i] += dew_m_cell_hour

            (
                snow_water_equivalent_top_m_cell,
                liquid_water_top_m_cell,
                snow_enthalpy_top_J_per_m2_cell,
                snow_density_top_kg_per_m3_cell,
                snow_water_equivalent_bottom_m_cell,
                liquid_water_bottom_m_cell,
                snow_enthalpy_bottom_J_per_m2_cell,
                snow_density_bottom_kg_per_m3_cell,
                _,  # snow_melt_m_per_hour
                runoff_from_melt_m,
                _,  # rainfall_that_resulted_in_runoff_if_interception_was_not_considered_m_per_hour
                sublimation_m_cell_hour,
                _,  # refreezing
            ) = update_snow_mass_and_phase(
                rainfall_m_per_hour=throughfall_m,
                swe_top_m=snow_water_equivalent_top_m_cell,
                liquid_water_top_m=liquid_water_top_m_cell,
                enthalpy_top_J_per_m2=snow_enthalpy_top_J_per_m2_cell,
                density_top_kg_per_m3=snow_density_top_kg_per_m3_cell,
                swe_bottom_m=snow_water_equivalent_bottom_m_cell,
                liquid_water_bottom_m=liquid_water_bottom_m_cell,
                enthalpy_bottom_J_per_m2=snow_enthalpy_bottom_J_per_m2_cell,
                density_bottom_kg_per_m3=snow_density_bottom_kg_per_m3_cell,
                air_temperature_C=tas_C,
                vapor_pressure_air_Pa=weather.actual_vapour_pressure_kPa
                * np.float32(1000.0),
                air_pressure_Pa=ps_pascal_this_hour,
                wind_10m_m_per_s=wind_10m_m_per_s,
                activate_layer_thickness_m=np.float32(0.2),
            )

            sublimation_m[i] += sublimation_m_cell_hour

            natural_available_water_infiltration_m: np.float32 = runoff_from_melt_m

            potential_direct_evaporation_m, potential_transpiration_m_cell_hour = (
                limit_evaporation_under_snow(
                    land_use_type=land_use_type[i],
                    snow_water_equivalent_m=np.float32(
                        snow_water_equivalent_top_m_cell
                        + snow_water_equivalent_bottom_m_cell
                    ),
                    potential_direct_evaporation_m=potential_direct_evaporation_m,
                    potential_transpiration_m=potential_transpiration_m_cell_hour,
                )
            )

            topwater_m_cell, open_water_evaporation_m_cell_hour = (
                add_water_to_topwater_and_evaporate_open_water(
                    natural_available_water_infiltration_m=natural_available_water_infiltration_m,
                    actual_irrigation_consumption_m=actual_irrigation_consumption_m[i],
                    land_use_type=land_use_type[i],
                    potential_direct_evaporation_m=potential_direct_evaporation_m,
                    topwater_m=topwater_m_cell,
                )
            )
            open_water_evaporation_m[i] += open_water_evaporation_m_cell_hour

            # Only newly added liquid surface water should advect rain heat here.
            # Pre-existing ponded water is already part of the top-soil control
            # volume and must not be reheated every infiltration call.
            new_surface_water_input_m: np.float32 = (
                natural_available_water_infiltration_m
                + actual_irrigation_consumption_m[i]
            )

            soil_is_frozen = top_layer_frozen_fraction > np.float32(0.0)
            # Over the whole profile: macropore infiltration carries rain heat to deeper layers.
            top_soil_enthalpy_before_rain_advection_J_per_m2: np.float32 = (
                soil_enthalpy_J_per_m2[i, :].sum()
            )

            groundwater_depth_hour_m: np.float32 = groundwater_depth_previous_day_m[
                i
            ] + (
                groundwater_depth_m[i] - groundwater_depth_previous_day_m[i]
            ) * np.float32(hour + 1) / np.float32(24.0)
            # Filling of the hillslope store with perched water, for the saturated
            # area: the same store that gives the interflow.
            dunne_effective_saturation: np.float32 = get_hillslope_store_filling(
                water_content_m=water_content_m[i, :],
                water_content_field_capacity_m=field_capacity_m,
                water_content_saturated_m=saturated_water_content_m,
                soil_layer_height_m=soil_layer_height_cell,
                groundwater_depth_m=groundwater_depth_hour_m,
            )
            # Frost loosens the soil surface and breaks the crust; rain on exposed,
            # snow-free soil builds it again.
            if soil_is_frozen:
                crust_rain_m[i] = np.float32(0.0)
            elif snow_water_equivalent_top_m_cell <= np.float32(0.0):
                crust_rain_m[i] += throughfall_m * (
                    np.float32(1.0)
                    - get_soil_cover_fraction(land_use_type[i], leaf_area_index[i])
                )
            surface_crust_factor_cell: np.float32 = get_surface_crust_factor(
                land_use_type[i],
                leaf_area_index[i],
                crust_rain_m[i],
                bare_soil_crust_factor,
                crust_formation_rain_m,
            )
            # A surface crust (raindrop impact on exposed soil) also seals the entrances of
            # macropores, so the macropore capacity is reduced by the same crust factor.
            # Computed outside the infiltration call: with this many keyword arguments
            # numba rejects a conditional inside the argument list (UnsupportedBytecodeError).
            macropore_capacity_cell_m_per_s: np.float32 = np.float32(0.0)
            if macropore_infiltration_capacity_m_per_s is not None:
                macropore_capacity_cell_m_per_s = (
                    macropore_infiltration_capacity_m_per_s[i]
                    * surface_crust_factor_cell
                )
            (
                topwater_m_cell,
                direct_runoff_m,
                groundwater_recharge_from_infiltraton_m,
                infiltration_amount,
                wetting_front_depth_m[i],
                wetting_front_suction_head_m[i],
                wetting_front_moisture_deficit[i],
                green_ampt_active_layer_idx[i],
                soil_enthalpy_J_per_m2[i, 0],
            ) = infiltration(
                seed=(unix_time_seconds + hour * np.int64(3600)) ^ np.int64(i),
                ws=saturated_water_content_m,
                wres=wres,
                saturated_hydraulic_conductivity_m_per_s=vertical_conductivity_cell,
                groundwater_toplayer_conductivity_m_per_s=groundwater_toplayer_conductivity_m_per_s[
                    grid_i
                ],
                land_use_type=land_use_type[i],
                w=water_content_m[i, :],
                topwater_m=topwater_m_cell,
                wetting_front_depth_m=wetting_front_depth_m[i],
                wetting_front_suction_head_m=wetting_front_suction_head_m[i],
                wetting_front_moisture_deficit=wetting_front_moisture_deficit[i],
                green_ampt_active_layer_idx=green_ampt_active_layer_idx[i],
                variable_runoff_shape_beta=variable_runoff_shape_beta[i],
                surface_crust_factor=surface_crust_factor_cell,
                bubbling_pressure_m_positive=bubbling_pressure_cell,
                soil_layer_height_m=soil_layer_height_cell,
                lambda_pore_size_distribution=lambda_pore_size_cell,
                soil_enthalpy_J_per_m2=soil_enthalpy_J_per_m2[i, :],
                solid_heat_capacity_J_per_m2_K=solid_heat_capacity,
                rain_temperature_C=tas_C,
                new_surface_water_input_m=new_surface_water_input_m,
                rainfall_lookup_table=rainfall_lookup_table,
                slope_m_per_m=slope_m_per_m[i],
                dunne_effective_saturation=dunne_effective_saturation,
                macropore_infiltration_capacity_m_per_s=macropore_capacity_cell_m_per_s,
            )

            runoff_m[i, hour] = direct_runoff_m
            groundwater_recharge_m[i] += groundwater_recharge_from_infiltraton_m
            top_soil_infiltration_m[i] += infiltration_amount

            rain_advection_enthalpy_flux_J_per_m2[i] += (
                soil_enthalpy_J_per_m2[i, :].sum()
                - top_soil_enthalpy_before_rain_advection_J_per_m2
            )

            # Redistribution between soil layers.
            # Use Ross (2003) as the default implicit vertical redistribution scheme.
            # This also handles percolation to groundwater if infiltration did not Already
            # produce recharge this hour.
            gw_ksat_m_per_s = np.float32(0.0)
            if groundwater_recharge_from_infiltraton_m <= np.float32(0.0):
                gw_ksat_m_per_s = groundwater_toplayer_conductivity_m_per_s[grid_i]

            (
                top_soil_percolation_ross,
                top_soil_rise_ross,
                percolation_to_groundwater_ross,
                capillary_rise_ross,
                percolation_to_groundwater_enthalpy_loss_ross,
                total_lateral_outflow_ross,
                interflow_enthalpy_loss_ross,
            ) = distribute_soil_water_ross(
                timestep_length_s=np.float32(3600.0),
                water_content_m=water_content_m[i, :],
                water_content_residual_m=wres,
                water_content_saturated_m=saturated_water_content_m,
                water_content_field_capacity_m=field_capacity_m,
                soil_enthalpy_J_per_m2=soil_enthalpy_J_per_m2[i, :],
                solid_heat_capacity_J_per_m2_K=solid_heat_capacity,
                vertical_saturated_hydraulic_conductivity_m_per_s=vertical_conductivity_cell,
                horizontal_saturated_hydraulic_conductivity_m_per_s=horizontal_conductivity_cell,
                soil_layer_height=soil_layer_height_cell,
                bubbling_pressure_m_positive=bubbling_pressure_cell,
                lambda_=lambda_pore_size_cell,
                slope_m_per_m=slope_m_per_m[i],
                hillslope_length_m=hillslope_length_m[i],
                interflow_structural_conductivity_m_per_s=interflow_structural_conductivity_m_per_s[
                    i
                ],
                green_ampt_active_layer_idx=green_ampt_active_layer_idx[i],
                topwater_m=topwater_m_cell,
                gw_ksat_m_per_s=gw_ksat_m_per_s,
                groundwater_depth_m=groundwater_depth_hour_m,
                deep_soil_temperature_C=deep_soil_temperature_C[i],
                max_capillary_rise_m=remaining_capillary_rise_m,
                interflow_structural_depth_m=interflow_structural_depth_m,
            )
            top_soil_percolation_to_layer_2_m[i] += top_soil_percolation_ross
            top_soil_rise_from_layer_2_m[i] += top_soil_rise_ross
            groundwater_recharge_m[i] += percolation_to_groundwater_ross
            capillar_rise_m[i] += capillary_rise_ross
            remaining_capillary_rise_m = max(
                np.float32(0.0), remaining_capillary_rise_m - capillary_rise_ross
            )
            interflow_m[i, hour] = total_lateral_outflow_ross
            groundwater_recharge_enthalpy_loss_J_per_m2[i] += (
                percolation_to_groundwater_enthalpy_loss_ross
            )
            interflow_enthalpy_loss_J_per_m2[i] += interflow_enthalpy_loss_ross

            # In-place copy into the pre-allocated buffer (avoids a heap allocation
            # per hourly substep that .copy() would trigger).
            water_content_before_transpiration_m[:] = water_content_m[i, :]
            topwater_before_transpiration_m: np.float32 = topwater_m_cell

            transpiration_m_cell_hour, topwater_m_cell = calculate_transpiration(
                soil_is_frozen=soil_is_frozen,
                wwp_m=wilting_point_m,
                wfc_m=field_capacity_m,
                wres_m=wres,
                soil_layer_height_m=soil_layer_height_cell,
                land_use_type=land_use_type[i],
                root_depth_m=root_depth_m[i],
                crop_group_number=crop_group_number[i],
                potential_transpiration_m=potential_transpiration_m_cell_hour,
                daily_reference_evapotranspiration_grass_m=daily_reference_evapotranspiration_grass_m[
                    i
                ],
                w_m=water_content_m[i, :],
                topwater_m=topwater_m_cell,
                minimum_effective_root_depth_m=minimum_effective_root_depth_m,
            )

            # Remove sensible enthalpy exported with transpired liquid water.
            for layer in range(N_SOIL_LAYERS):
                transpired_from_layer_m: np.float32 = max(
                    np.float32(0.0),
                    water_content_before_transpiration_m[layer]
                    - water_content_m[i, layer],
                )
                if transpired_from_layer_m > np.float32(0.0):
                    if layer == 0:
                        top_soil_transpiration_m[i] += transpired_from_layer_m
                    layer_temperature_C, _ = (
                        get_temperature_and_frozen_fraction_from_enthalpy_scalar(
                            enthalpy_J_per_m2=soil_enthalpy_J_per_m2[i, layer],
                            solid_heat_capacity_J_per_m2_K=solid_heat_capacity[layer],
                            water_content_m=water_content_before_transpiration_m[layer],
                            topwater_m=(
                                topwater_before_transpiration_m
                                if layer == 0
                                else np.float32(0.0)
                            ),
                        )
                    )
                    advected_temperature_C = max(layer_temperature_C, np.float32(0.0))
                    enthalpy_export_J_per_m2 = (
                        transpired_from_layer_m
                        * RHO_WATER_KG_PER_M3
                        * SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K
                        * advected_temperature_C
                    )
                    soil_enthalpy_J_per_m2[i, layer] -= enthalpy_export_J_per_m2
                    transpiration_enthalpy_loss_J_per_m2[i] += enthalpy_export_J_per_m2

            transpired_from_topwater_m: np.float32 = max(
                np.float32(0.0), topwater_before_transpiration_m - topwater_m_cell
            )
            if transpired_from_topwater_m > np.float32(0.0):
                top_layer_temperature_C, _ = (
                    get_temperature_and_frozen_fraction_from_enthalpy_scalar(
                        enthalpy_J_per_m2=soil_enthalpy_J_per_m2[i, 0],
                        solid_heat_capacity_J_per_m2_K=solid_heat_capacity[0],
                        water_content_m=water_content_before_transpiration_m[0],
                        topwater_m=topwater_before_transpiration_m,
                    )
                )
                topwater_advected_temperature_C = max(
                    top_layer_temperature_C, np.float32(0.0)
                )
                topwater_enthalpy_export_J_per_m2 = (
                    transpired_from_topwater_m
                    * RHO_WATER_KG_PER_M3
                    * SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K
                    * topwater_advected_temperature_C
                )
                soil_enthalpy_J_per_m2[i, 0] -= topwater_enthalpy_export_J_per_m2
                transpiration_enthalpy_loss_J_per_m2[i] += (
                    topwater_enthalpy_export_J_per_m2
                )

            potential_transpiration_m[i] += potential_transpiration_m_cell_hour
            transpiration_m[i] += transpiration_m_cell_hour

            # soil moisture is updated in place
            bare_soil_evaporation_m_cell_hour = calculate_bare_soil_evaporation(
                soil_is_frozen=soil_is_frozen,
                land_use_type=land_use_type[i],
                potential_direct_evaporation_m=potential_direct_evaporation_m,
                energy_limited_soil_evaporation_m=energy_limited_soil_evaporation_m,
                open_water_evaporation_m=open_water_evaporation_m_cell_hour,
                w_m=water_content_m[i, :],
                wres_m=wres,
                wfc_m=field_capacity_m,
                wwp_m=wilting_point_m,
                soil_layer_height_m=soil_layer_height_cell,
            )
            bare_soil_evaporation[i] += bare_soil_evaporation_m_cell_hour
            soil_surface_evaporation_previous_hour_m = (
                bare_soil_evaporation_m_cell_hour + open_water_evaporation_m_cell_hour
            )

            actual_evapotranspiration_m_cell_hour: np.float32 = (
                interception_evaporation_m_cell_hour
                + open_water_evaporation_m_cell_hour
                + bare_soil_evaporation_m_cell_hour
                + transpiration_m_cell_hour
            )
            evapotranspiration_m[i, hour] = actual_evapotranspiration_m_cell_hour

            _, frozen_fraction_top_layer = (
                get_temperature_and_frozen_fraction_from_enthalpy_scalar(
                    enthalpy_J_per_m2=soil_enthalpy_J_per_m2[i, 0],
                    solid_heat_capacity_J_per_m2_K=solid_heat_capacity[0],
                    water_content_m=water_content_m[i, 0],
                    topwater_m=topwater_m_cell,
                )
            )

            soil_enthalpy_before_evaporative_cooling_J_per_m2: np.float32 = (
                soil_enthalpy_J_per_m2[i, 0]
            )
            soil_enthalpy_J_per_m2[i, 0] = apply_evaporative_cooling(
                soil_enthalpy_top_layer_J_per_m2=soil_enthalpy_J_per_m2[i, 0],
                evaporation_m=bare_soil_evaporation_m_cell_hour
                + open_water_evaporation_m_cell_hour,
                frozen_fraction_top_layer=frozen_fraction_top_layer,
            )
            evaporative_cooling_enthalpy_loss_J_per_m2[i] += (
                soil_enthalpy_before_evaporative_cooling_J_per_m2
                - soil_enthalpy_J_per_m2[i, 0]
            )

        topwater_m[i] = topwater_m_cell
        interception_storage_m[i] = interception_storage_m_cell

        snow_water_equivalent_m[i, 0] = snow_water_equivalent_top_m_cell
        snow_water_equivalent_m[i, 1] = snow_water_equivalent_bottom_m_cell
        liquid_water_in_snow_m[i, 0] = liquid_water_top_m_cell
        liquid_water_in_snow_m[i, 1] = liquid_water_bottom_m_cell
        snow_enthalpy_J_per_m2[i, 0] = snow_enthalpy_top_J_per_m2_cell
        snow_enthalpy_J_per_m2[i, 1] = snow_enthalpy_bottom_J_per_m2_cell
        snow_density_kg_per_m3[i, 0] = snow_density_top_kg_per_m3_cell
        snow_density_kg_per_m3[i, 1] = snow_density_bottom_kg_per_m3_cell

    return (
        rain_m,
        snow_m,
        topwater_m,
        reference_evapotranspiration_grass_m,
        potential_open_water_evaporation_m,
        snow_water_equivalent_m,
        liquid_water_in_snow_m,
        sublimation_m,
        snow_enthalpy_J_per_m2,
        snow_density_kg_per_m3,
        interception_storage_m,
        interception_evaporation_m,
        open_water_evaporation_m,
        runoff_m,
        groundwater_recharge_m,
        capillar_rise_m,
        interflow_m,
        bare_soil_evaporation,
        transpiration_m,
        potential_transpiration_m,
        potential_evapotranspiration_m,
        soil_boundary_enthalpy_flux_J_per_m2,
        rain_advection_enthalpy_flux_J_per_m2,
        evaporative_cooling_enthalpy_loss_J_per_m2,
        interflow_enthalpy_loss_J_per_m2,
        groundwater_recharge_enthalpy_loss_J_per_m2,
        transpiration_enthalpy_loss_J_per_m2,
        top_soil_infiltration_m,
        top_soil_rise_from_layer_2_m,
        top_soil_percolation_to_layer_2_m,
        top_soil_transpiration_m,
        evapotranspiration_m,
        dew_m,
    )


class LandSurfaceInputs(NamedTuple):
    """Container for `land_surface_model` inputs.

    This keeps the model call and debug dumps in sync by using the same
    ordered, named fields for both pathways.
    """

    unix_time_seconds: np.int64
    land_use_type: ArrayInt32
    slope_m_per_m: ArrayFloat32
    hillslope_length_m: ArrayFloat32
    water_content_m: TwoDArrayFloat32
    packed_soil_properties: ThreeDArrayFloat32
    soil_enthalpy_J_per_m2: TwoDArrayFloat32
    root_depth_m: ArrayFloat32
    topwater_m: ArrayFloat32
    variable_runoff_shape_beta: ArrayFloat32
    snow_water_equivalent_m: TwoDArrayFloat64
    liquid_water_in_snow_m: TwoDArrayFloat64
    snow_enthalpy_J_per_m2: TwoDArrayFloat32
    snow_density_kg_per_m3: TwoDArrayFloat32
    snow_albedo: ArrayFloat32
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
    basal_crop_factor: ArrayFloat32
    maximum_crop_factor: ArrayFloat32
    actual_irrigation_consumption_m: ArrayFloat32
    groundwater_depth_previous_day_m: ArrayFloat32
    groundwater_depth_m: ArrayFloat32
    groundwater_toplayer_conductivity_m_per_s: ArrayFloat32
    wetting_front_depth_m: ArrayFloat32
    wetting_front_suction_head_m: ArrayFloat32
    wetting_front_moisture_deficit: ArrayFloat32
    crust_rain_m: ArrayFloat32
    green_ampt_active_layer_idx: ArrayInt32
    crop_group_number: ArrayFloat32
    minimum_effective_root_depth_m: np.float32
    interflow_structural_conductivity_m_per_s: ArrayFloat32
    interflow_structural_depth_m: np.float32
    macropore_infiltration_capacity_m_per_s: ArrayFloat32
    bare_soil_crust_factor: np.float32
    crust_formation_rain_m: np.float32
    deep_soil_temperature_C: ArrayFloat32
    leaf_area_index: ArrayFloat32
    rainfall_lookup_table: TwoDArrayFloat32
    daily_reference_evapotranspiration_grass_m: ArrayFloat32
    max_capillary_rise_m: ArrayFloat32
    HRU_to_grid: ArrayInt32


class LandSurfaceVariables(Bucket):
    """Land surface variables for GEB."""

    topwater: ArrayFloat32
    clay_percentage: TwoDArrayFloat32
    sand_percentage: TwoDArrayFloat32
    silt_percentage: TwoDArrayFloat32
    bulk_density_kg_per_dm3: TwoDArrayFloat32
    soil_layer_height: TwoDArrayFloat32
    snow_water_equivalent_m: TwoDArrayFloat64
    liquid_water_in_snow_m: TwoDArrayFloat64
    snow_enthalpy_J_per_m2: TwoDArrayFloat32
    snow_density_kg_per_m3: TwoDArrayFloat32
    snow_albedo: ArrayFloat32
    interception_storage_m: ArrayFloat32
    variable_runoff_shape_beta: TwoDArrayFloat32
    crop_map: ArrayInt32
    minimum_effective_root_depth_m: np.float32
    green_ampt_active_layer_idx: ArrayInt32
    deep_soil_temperature_C: ArrayFloat32
    thermal_conductivity_saturated_unfrozen_W_per_m_K: TwoDArrayFloat32
    thermal_conductivity_saturated_frozen_W_per_m_K: TwoDArrayFloat32
    thermal_conductivity_dry_soil_W_per_m_K: TwoDArrayFloat32


class LandSurface(Module):
    """Land surface module for GEB."""

    var: LandSurfaceVariables
    _crop_stage_lengths: np.ndarray
    _crop_sub_stage_lengths: np.ndarray
    _crop_factor_per_crop_stage: np.ndarray
    _crop_root_depths: np.ndarray
    _crop_base_temperature_C: ArrayFloat32
    _dekad_per_day_of_year: ArrayInt32
    _air_temperature_dekadal_mean_C: TwoDArrayFloat32
    _air_temperature_dekadal_standard_deviation_C: TwoDArrayFloat32

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

    def _build_land_surface_inputs(
        self,
        *,
        unix_time_seconds: np.int64,
        root_depth_m: ArrayFloat32,
        interception_capacity_m: ArrayFloat32,
        pr_kg_per_m2_per_s: TwoDArrayFloat32,
        tas_2m_K: TwoDArrayFloat32,
        crop_factor: ArrayFloat32,
        basal_crop_factor: ArrayFloat32,
        maximum_crop_factor: ArrayFloat32,
        actual_irrigation_consumption_m: ArrayFloat32,
        groundwater_depth_previous_day_m: ArrayFloat32,
        groundwater_depth_m: ArrayFloat32,
        deep_soil_temperature_C: ArrayFloat32,
        leaf_area_index: ArrayFloat32,
        max_capillary_rise_m: ArrayFloat32,
        HRU_to_grid: ArrayInt32,
    ) -> LandSurfaceInputs:
        """Build the input bundle for `land_surface_model`.

        Args:
            unix_time_seconds: Current model time in unix seconds.
            root_depth_m: Root depth for each HRU (m).
            interception_capacity_m: Interception capacity per HRU (m).
            pr_kg_per_m2_per_s: Precipitation rate per hour (kg/m2/s).
            tas_2m_K: 2m air temperature per hour (K).
            crop_factor: Crop factor per HRU (-).
            basal_crop_factor: Basal crop factor (transpiration) per HRU (-).
            maximum_crop_factor: Upper limit of the crop factor of a wet surface
                (FAO-56 Kc_max) per HRU (-).
            actual_irrigation_consumption_m: Actual irrigation consumption (m).
            groundwater_depth_previous_day_m: Depth of the groundwater table below the surface
                of each HRU at the start of the previous day (m).
            groundwater_depth_m: Depth of the groundwater table below the surface of each HRU
                at the start of the day (m).
            deep_soil_temperature_C: Sub-surface boundary temperature (C).
            leaf_area_index: Leaf area index per HRU (-).
            max_capillary_rise_m: Maximum available groundwater storage for capillary rise per HRU (meters).
            HRU_to_grid: Mapping from HRU cell index to compressed grid index (-).

        Returns:
            Bundle of inputs for `land_surface_model`.
        """
        CO2_ppm: np.float32 = np.float32(self.model.forcing.load("CO2_ppm"))
        groundwater_toplayer_conductivity_m_per_s: ArrayFloat32 = (
            self.hydrology.groundwater.hydraulic_conductivity_m_per_day[0]
            / np.float32(86400.0)
        ).astype(np.float32)
        crop_group_number: ArrayFloat32 = resolve_crop_group_number(
            crop_map=self.HRU.var.crop_map,
            land_use_type=self.HRU.var.land_use_type,
            crop_group_number_forest=self.HRU.var.crop_group_number_forest,
            crop_group_number_grassland_like=self.HRU.var.crop_group_number_grassland_like,
            crop_group_number_per_group=self.model.agents.crop_farmers.var.crop_data[
                "crop_group_number"
            ].values.astype(np.float32),
            forest_land_use=FOREST,
            grassland_like_land_use=GRASSLAND_LIKE,
        )

        return LandSurfaceInputs(
            unix_time_seconds=unix_time_seconds,
            land_use_type=self.HRU.var.land_use_type,
            slope_m_per_m=self.HRU.var.slope_m_per_m,
            hillslope_length_m=self.HRU.var.hillslope_length_m,
            water_content_m=self.HRU.var.water_content_m,
            packed_soil_properties=self.HRU.var.packed_soil_properties,
            soil_enthalpy_J_per_m2=self.HRU.var.soil_enthalpy_J_per_m2,
            root_depth_m=root_depth_m,
            topwater_m=self.HRU.var.topwater_m,
            variable_runoff_shape_beta=self.HRU.var.variable_runoff_shape_beta,
            snow_water_equivalent_m=self.HRU.var.snow_water_equivalent_m,
            liquid_water_in_snow_m=self.HRU.var.liquid_water_in_snow_m,
            snow_enthalpy_J_per_m2=self.HRU.var.snow_enthalpy_J_per_m2,
            snow_density_kg_per_m3=self.HRU.var.snow_density_kg_per_m3,
            snow_albedo=self.HRU.var.snow_albedo,
            interception_storage_m=self.HRU.var.interception_storage_m,
            interception_capacity_m=interception_capacity_m,
            pr_kg_per_m2_per_s=pr_kg_per_m2_per_s,
            tas_2m_K=tas_2m_K,
            dewpoint_tas_2m_K=self.grid.dewpoint_tas_2m_K,
            ps_pascal=self.grid.ps_pascal,
            rlds_W_per_m2=self.grid.rlds_W_per_m2,
            rsds_W_per_m2=self.grid.rsds_W_per_m2,
            wind_u10m_m_per_s=self.grid.wind_u10m_m_per_s,
            wind_v10m_m_per_s=self.grid.wind_v10m_m_per_s,
            CO2_ppm=CO2_ppm,
            crop_factor=crop_factor,
            basal_crop_factor=basal_crop_factor,
            maximum_crop_factor=maximum_crop_factor,
            actual_irrigation_consumption_m=actual_irrigation_consumption_m,
            groundwater_depth_previous_day_m=groundwater_depth_previous_day_m,
            groundwater_depth_m=groundwater_depth_m,
            groundwater_toplayer_conductivity_m_per_s=groundwater_toplayer_conductivity_m_per_s,
            wetting_front_depth_m=self.HRU.var.wetting_front_depth_m,
            wetting_front_suction_head_m=self.HRU.var.wetting_front_suction_head_m,
            wetting_front_moisture_deficit=self.HRU.var.wetting_front_moisture_deficit,
            crust_rain_m=self.HRU.var.crust_rain_m,
            green_ampt_active_layer_idx=self.HRU.var.green_ampt_active_layer_idx,
            crop_group_number=crop_group_number,
            minimum_effective_root_depth_m=self.var.minimum_effective_root_depth_m,
            interflow_structural_conductivity_m_per_s=self.get_interflow_structural_conductivity_m_per_s(),
            interflow_structural_depth_m=np.float32(
                self.model.config["hydrology"]["landsurface"][
                    "interflow_structural_depth_m"
                ]
            ),
            macropore_infiltration_capacity_m_per_s=self.get_macropore_infiltration_capacity_m_per_s(),
            bare_soil_crust_factor=np.float32(
                self.model.config["hydrology"]["landsurface"]["bare_soil_crust_factor"]
            ),
            crust_formation_rain_m=np.float32(
                self.model.config["hydrology"]["landsurface"]["crust_formation_rain_mm"]
                / 1000.0
            ),
            deep_soil_temperature_C=deep_soil_temperature_C,
            leaf_area_index=leaf_area_index,
            rainfall_lookup_table=self.rainfall_lookup_table,
            daily_reference_evapotranspiration_grass_m=self.HRU.var.daily_reference_evapotranspiration_grass_m,
            max_capillary_rise_m=max_capillary_rise_m,
            HRU_to_grid=HRU_to_grid,
        )

    def _snapshot_land_surface_inputs_for_error(
        self,
        *,
        land_surface_inputs: LandSurfaceInputs,
        water_content_m_prev: TwoDArrayFloat32,
        topwater_m_prev: ArrayFloat32,
        snow_water_equivalent_prev: TwoDArrayFloat64,
        liquid_water_in_snow_prev: TwoDArrayFloat64,
        snow_enthalpy_J_per_m2_prev: TwoDArrayFloat32,
        snow_density_kg_per_m3_prev: TwoDArrayFloat32,
        interception_storage_prev: ArrayFloat32,
        soil_enthalpy_J_per_m2_prev: TwoDArrayFloat32,
        deep_soil_temperature_C_prev: ArrayFloat32,
        wetting_front_depth_prev: ArrayFloat32,
        wetting_front_suction_head_prev: ArrayFloat32,
        wetting_front_moisture_deficit_prev: ArrayFloat32,
        green_ampt_active_layer_idx_prev: ArrayInt32,
        index: int,
    ) -> LandSurfaceInputs:
        """Build a snapshot of land surface inputs for error reproduction.

        Args:
            land_surface_inputs: Inputs used for the normal model call.
            water_content_m_prev: Pre-call soil water column (m).
            topwater_m_prev: Pre-call topwater (m).
            snow_water_equivalent_prev: Pre-call snow water equivalent (m).
            liquid_water_in_snow_prev: Pre-call liquid water in snow (m).
            snow_enthalpy_J_per_m2_prev: Pre-call snow enthalpy (J/m2).
            snow_density_kg_per_m3_prev: Pre-call snow density (kg/m3).
            interception_storage_prev: Pre-call interception storage (m).
            soil_enthalpy_J_per_m2_prev: Pre-call soil enthalpy (J/m2).
            deep_soil_temperature_C_prev: Pre-call deep soil temperature (C).
            wetting_front_depth_prev: Pre-call wetting front depth (m).
            wetting_front_suction_head_prev: Pre-call wetting front suction head (m).
            wetting_front_moisture_deficit_prev: Pre-call wetting front moisture deficit (-).
            green_ampt_active_layer_idx_prev: Pre-call Green-Ampt active layer index (-).
            index: Index to slice all inputs for (isolates a single cell).

        Returns:
            Snapshot of model inputs that reproduces the failure context.
        """
        # isolate the failing cell while keeping original ranks (dimensions)
        num_cells: int = land_surface_inputs.slope_m_per_m.shape[0]
        sliced_fields = {}
        for field in land_surface_inputs._fields:
            val = getattr(land_surface_inputs, field)
            if isinstance(val, np.ndarray) and val.shape[0] == num_cells:
                if val.ndim == 1:
                    sliced_fields[field] = val[index : index + 1]
                elif val.ndim == 2:
                    # Inputs are in cell-major layout [num_cells, N_LAYERS] or
                    # time-major layout [num_cells, 24]; cell axis is 0.
                    sliced_fields[field] = val[index : index + 1, :]
                else:
                    # 3-D packed properties layout [num_cells, N_PROPS, N_LAYERS]
                    sliced_fields[field] = val[index : index + 1, :, :]
            else:
                sliced_fields[field] = val

        land_surface_inputs = LandSurfaceInputs(**sliced_fields)

        error_inputs: LandSurfaceInputs = land_surface_inputs._replace(
            water_content_m=water_content_m_prev[index : index + 1, :],
            topwater_m=topwater_m_prev[index : index + 1],
            snow_water_equivalent_m=snow_water_equivalent_prev[index : index + 1, :],
            liquid_water_in_snow_m=liquid_water_in_snow_prev[index : index + 1, :],
            snow_enthalpy_J_per_m2=snow_enthalpy_J_per_m2_prev[index : index + 1, :],
            snow_density_kg_per_m3=snow_density_kg_per_m3_prev[index : index + 1, :],
            interception_storage_m=interception_storage_prev[index : index + 1],
            soil_enthalpy_J_per_m2=soil_enthalpy_J_per_m2_prev[index : index + 1, :],
            deep_soil_temperature_C=deep_soil_temperature_C_prev[index : index + 1],
            wetting_front_depth_m=wetting_front_depth_prev[index : index + 1],
            wetting_front_suction_head_m=wetting_front_suction_head_prev[
                index : index + 1
            ],
            wetting_front_moisture_deficit=wetting_front_moisture_deficit_prev[
                index : index + 1
            ],
            green_ampt_active_layer_idx=green_ampt_active_layer_idx_prev[
                index : index + 1
            ],
        )
        return error_inputs

    def _export_landsurface_water_error_snapshot(
        self,
        *,
        land_surface_inputs: LandSurfaceInputs,
        water_content_m_prev: TwoDArrayFloat32,
        topwater_m_prev: ArrayFloat32,
        snow_water_equivalent_prev: TwoDArrayFloat64,
        liquid_water_in_snow_prev: TwoDArrayFloat64,
        snow_enthalpy_J_per_m2_prev: TwoDArrayFloat32,
        snow_density_kg_per_m3_prev: TwoDArrayFloat32,
        interception_storage_prev: ArrayFloat32,
        soil_enthalpy_J_per_m2_prev: TwoDArrayFloat32,
        deep_soil_temperature_C_prev: ArrayFloat32,
        wetting_front_depth_prev: ArrayFloat32,
        wetting_front_suction_head_prev: ArrayFloat32,
        wetting_front_moisture_deficit_prev: ArrayFloat32,
        green_ampt_active_layer_idx_prev: ArrayInt32,
        index: int,
        error_message: str,
    ) -> None:
        """Build a snapshot of land surface inputs for error reproduction and raise.

        Notes:
            Saves the isolated cell inputs to the diagnostics folder as an NPZ file
            and re-runs the kernel on that isolated cell for reproduction before
            raising an AssertionError.

        Args:
            land_surface_inputs: Inputs used for the normal model call.
            water_content_m_prev: Pre-call soil water column (m).
            topwater_m_prev: Pre-call topwater (m).
            snow_water_equivalent_prev: Pre-call snow water equivalent (m).
            liquid_water_in_snow_prev: Pre-call liquid water in snow (m).
            snow_enthalpy_J_per_m2_prev: Pre-call snow enthalpy (J/m2).
            snow_density_kg_per_m3_prev: Pre-call snow density (kg/m3).
            interception_storage_prev: Pre-call interception storage (m).
            soil_enthalpy_J_per_m2_prev: Pre-call soil enthalpy (J/m2).
            deep_soil_temperature_C_prev: Pre-call deep soil temperature (C).
            wetting_front_depth_prev: Pre-call wetting front depth (m).
            wetting_front_suction_head_prev: Pre-call wetting front suction head (m).
            wetting_front_moisture_deficit_prev: Pre-call wetting front moisture deficit (-).
            green_ampt_active_layer_idx_prev: Pre-call Green-Ampt active layer index (-).
            index: HRU index of the failing cell.
            error_message: Description of the failure.

        Raises:
            AssertionError: Always raised with failure details and HRU land use type.
        """
        error_inputs: LandSurfaceInputs = self._snapshot_land_surface_inputs_for_error(
            land_surface_inputs=land_surface_inputs,
            water_content_m_prev=water_content_m_prev,
            topwater_m_prev=topwater_m_prev,
            snow_water_equivalent_prev=snow_water_equivalent_prev,
            liquid_water_in_snow_prev=liquid_water_in_snow_prev,
            snow_enthalpy_J_per_m2_prev=snow_enthalpy_J_per_m2_prev,
            snow_density_kg_per_m3_prev=snow_density_kg_per_m3_prev,
            interception_storage_prev=interception_storage_prev,
            soil_enthalpy_J_per_m2_prev=soil_enthalpy_J_per_m2_prev,
            deep_soil_temperature_C_prev=deep_soil_temperature_C_prev,
            wetting_front_depth_prev=wetting_front_depth_prev,
            wetting_front_suction_head_prev=wetting_front_suction_head_prev,
            wetting_front_moisture_deficit_prev=wetting_front_moisture_deficit_prev,
            green_ampt_active_layer_idx_prev=green_ampt_active_layer_idx_prev,
            index=index,
        )
        diag_path: Path = (
            self.model.diagnostics_folder
            / f"diagnostic_landsurface_water_error_cell_{index}.npz"
        )
        np.savez(diag_path, **error_inputs._asdict())
        self.model.logger.error(
            f"{error_message} at index {index}. Diagnostic data exported to {diag_path}"
        )
        # Re-run the model for the failing cell with the isolated inputs to confirm that the error can be reproduced
        land_surface_model(**error_inputs._asdict())

        raise AssertionError(
            f"{error_message} at HRU index {index}. Land use type: {self.HRU.var.land_use_type[index]}"
        )

    def _check_land_surface_water(
        self,
        *,
        land_surface_inputs: LandSurfaceInputs,
        water_content_m_prev: TwoDArrayFloat32,
        topwater_m_prev: ArrayFloat32,
        snow_water_equivalent_prev: TwoDArrayFloat64,
        liquid_water_in_snow_prev: TwoDArrayFloat64,
        snow_enthalpy_J_per_m2_prev: TwoDArrayFloat32,
        snow_density_kg_per_m3_prev: TwoDArrayFloat32,
        interception_storage_prev: ArrayFloat32,
        soil_enthalpy_J_per_m2_prev: TwoDArrayFloat32,
        deep_soil_temperature_C_prev: ArrayFloat32,
        wetting_front_depth_prev: ArrayFloat32,
        wetting_front_suction_head_prev: ArrayFloat32,
        wetting_front_moisture_deficit_prev: ArrayFloat32,
        green_ampt_active_layer_idx_prev: ArrayInt32,
        pr_kg_per_m2_per_s: TwoDArrayFloat32,
        actual_irrigation_consumption_m: ArrayFloat32,
        capillar_rise_m: ArrayFloat32,
        sublimation_or_deposition_m: ArrayFloat32,
        interception_evaporation_m: ArrayFloat32,
        dew_m: ArrayFloat32,
        open_water_evaporation_m: ArrayFloat32,
        runoff_m: TwoDArrayFloat32,
        interflow_m: TwoDArrayFloat32,
        groundwater_recharge_m: ArrayFloat32,
        bare_soil_evaporation_m: ArrayFloat32,
        transpiration_m: ArrayFloat32,
    ) -> None:
        """Validate land surface water state and flux variables and water balance.

        Args:
            land_surface_inputs: Inputs used for the normal model call.
            water_content_m_prev: Pre-call soil water column (m).
            topwater_m_prev: Pre-call topwater (m).
            snow_water_equivalent_prev: Pre-call snow water equivalent (m).
            liquid_water_in_snow_prev: Pre-call liquid water in snow (m).
            snow_enthalpy_J_per_m2_prev: Pre-call snow enthalpy (J/m2).
            snow_density_kg_per_m3_prev: Pre-call snow density (kg/m3).
            interception_storage_prev: Pre-call interception storage (m).
            soil_enthalpy_J_per_m2_prev: Pre-call soil enthalpy (J/m2).
            deep_soil_temperature_C_prev: Pre-call deep soil temperature (C).
            wetting_front_depth_prev: Pre-call wetting front depth (m).
            wetting_front_suction_head_prev: Pre-call wetting front suction head (m).
            wetting_front_moisture_deficit_prev: Pre-call wetting front moisture deficit (-).
            green_ampt_active_layer_idx_prev: Pre-call Green-Ampt active layer index (-).
            pr_kg_per_m2_per_s: Hourly precipitation rate (kg/m2/s).
            actual_irrigation_consumption_m: Irrigation water consumption (m).
            capillar_rise_m: Capillary rise from groundwater (m).
            sublimation_or_deposition_m: Net sublimation or deposition (m).
            interception_evaporation_m: Interception evaporation, net of dew (m).
            dew_m: Dew (m).
            open_water_evaporation_m: Open water evaporation (m).
            runoff_m: Surface runoff per hour (m/hour).
            interflow_m: Interflow per hour (m/hour).
            groundwater_recharge_m: Groundwater recharge (m).
            bare_soil_evaporation_m: Bare soil evaporation (m).
            transpiration_m: Plant transpiration (m).

        """

        def fail(index: int, message: str) -> None:
            self._export_landsurface_water_error_snapshot(
                land_surface_inputs=land_surface_inputs,
                water_content_m_prev=water_content_m_prev,
                topwater_m_prev=topwater_m_prev,
                snow_water_equivalent_prev=snow_water_equivalent_prev,
                liquid_water_in_snow_prev=liquid_water_in_snow_prev,
                snow_enthalpy_J_per_m2_prev=snow_enthalpy_J_per_m2_prev,
                snow_density_kg_per_m3_prev=snow_density_kg_per_m3_prev,
                interception_storage_prev=interception_storage_prev,
                soil_enthalpy_J_per_m2_prev=soil_enthalpy_J_per_m2_prev,
                deep_soil_temperature_C_prev=deep_soil_temperature_C_prev,
                wetting_front_depth_prev=wetting_front_depth_prev,
                wetting_front_suction_head_prev=wetting_front_suction_head_prev,
                wetting_front_moisture_deficit_prev=wetting_front_moisture_deficit_prev,
                green_ampt_active_layer_idx_prev=green_ampt_active_layer_idx_prev,
                index=index,
                error_message=message,
            )

        topwater: ArrayFloat32 = self.HRU.var.topwater_m
        water_content: TwoDArrayFloat32 = self.HRU.var.water_content_m
        interception: ArrayFloat32 = self.HRU.var.interception_storage_m
        snow_water_equivalent: TwoDArrayFloat64 = self.HRU.var.snow_water_equivalent_m
        liquid_water_in_snow: TwoDArrayFloat64 = self.HRU.var.liquid_water_in_snow_m

        num_cells: int = topwater.shape[0]
        err_flags: ArrayInt8 = np.zeros(num_cells, dtype=np.int8)

        validate_and_check_water_balance_kernel(
            topwater_m=topwater,
            runoff_m=runoff_m,
            water_content_m=water_content,
            interception_storage_m=interception,
            snow_water_equivalent_m=snow_water_equivalent,
            liquid_water_in_snow_m=liquid_water_in_snow,
            interflow_m=interflow_m,
            groundwater_recharge_m=groundwater_recharge_m,
            bare_soil_evaporation_m=bare_soil_evaporation_m,
            transpiration_m=transpiration_m,
            interception_evaporation_m=interception_evaporation_m,
            dew_m=dew_m,
            open_water_evaporation_m=open_water_evaporation_m,
            pr_kg_per_m2_per_s=pr_kg_per_m2_per_s,
            actual_irrigation_consumption_m=actual_irrigation_consumption_m,
            capillar_rise_m=capillar_rise_m,
            sublimation_or_deposition_m=sublimation_or_deposition_m,
            snow_water_equivalent_prev=snow_water_equivalent_prev,
            liquid_water_in_snow_prev=liquid_water_in_snow_prev,
            interception_storage_prev=interception_storage_prev,
            topwater_m_prev=topwater_m_prev,
            water_content_m_prev=water_content_m_prev,
            err_flags=err_flags,
            HRU_to_grid=land_surface_inputs.HRU_to_grid,
            tolerance=1e-5,
        )

        if np.any(err_flags != ERR_NONE):
            # Scan error priority matching validation order:
            # 1. Topwater
            if np.any(err_flags == ERR_NAN_TOPWATER):
                first_failing_hru_topwater: int = int(
                    np.where(err_flags == ERR_NAN_TOPWATER)[0][0]
                )
                fail(first_failing_hru_topwater, "NaN topwater detected")
            if np.any(err_flags == ERR_NEG_TOPWATER):
                worst_hru_topwater: int = int(np.argmin(topwater))
                value_topwater: float = float(topwater[worst_hru_topwater])
                fail(
                    worst_hru_topwater,
                    f"Negative topwater detected (value: {value_topwater:.6e} m)",
                )

            # 2. Runoff
            if np.any(err_flags == ERR_NAN_RUNOFF):
                first_failing_hru_runoff: int = int(
                    np.where(np.any(np.isnan(runoff_m), axis=1))[0][0]
                )
                fail(first_failing_hru_runoff, "NaN runoff detected")
            if np.any(err_flags == ERR_NEG_RUNOFF):
                min_runoff_per_hru: ArrayFloat32 = runoff_m.min(axis=1)
                worst_hru_runoff: int = int(np.argmin(min_runoff_per_hru))
                min_value_runoff: float = float(min_runoff_per_hru[worst_hru_runoff])
                fail(
                    worst_hru_runoff,
                    f"Negative runoff detected (min value: {min_value_runoff:.6e} m)",
                )

            # 3. Soil water content
            if np.any(err_flags == ERR_NAN_WATER_CONTENT):
                first_failing_hru_water_content: int = int(
                    np.where(np.any(np.isnan(water_content), axis=1))[0][0]
                )
                fail(first_failing_hru_water_content, "NaN soil water content detected")
            if np.any(err_flags == ERR_NEG_WATER_CONTENT):
                min_water_content_per_hru: ArrayFloat32 = water_content.min(axis=1)
                worst_hru_water_content: int = int(np.argmin(min_water_content_per_hru))
                min_value_water_content: float = float(
                    min_water_content_per_hru[worst_hru_water_content]
                )
                fail(
                    worst_hru_water_content,
                    f"Negative soil water content detected (min value: {min_value_water_content:.6e} m)",
                )

            # 4. Interception
            if np.any(err_flags == ERR_NAN_INTERCEPTION):
                first_failing_hru_interception: int = int(
                    np.where(err_flags == ERR_NAN_INTERCEPTION)[0][0]
                )
                fail(
                    first_failing_hru_interception, "NaN interception storage detected"
                )
            if np.any(err_flags == ERR_NEG_INTERCEPTION):
                worst_hru_interception: int = int(np.argmin(interception))
                value_interception: float = float(interception[worst_hru_interception])
                fail(
                    worst_hru_interception,
                    f"Negative interception storage detected (value: {value_interception:.6e} m)",
                )

            # 5. Snow
            if np.any(err_flags == ERR_NAN_SWE):
                first_failing_hru_snow: int = int(
                    np.where(np.any(np.isnan(snow_water_equivalent), axis=1))[0][0]
                )
                fail(first_failing_hru_snow, "NaN snow water equivalent detected")
            if np.any(err_flags == ERR_NEG_SWE):
                min_snow_per_hru: ArrayFloat64 = snow_water_equivalent.min(axis=1)
                worst_hru_snow: int = int(np.argmin(min_snow_per_hru))
                min_value_snow: float = float(min_snow_per_hru[worst_hru_snow])
                fail(
                    worst_hru_snow,
                    f"Negative snow water equivalent detected (min value: {min_value_snow:.6e} m)",
                )

            if np.any(err_flags == ERR_NAN_LIQUID_SNOW):
                first_failing_hru_liquid_snow: int = int(
                    np.where(np.any(np.isnan(liquid_water_in_snow), axis=1))[0][0]
                )
                fail(first_failing_hru_liquid_snow, "NaN liquid water in snow detected")
            if np.any(err_flags == ERR_NEG_LIQUID_SNOW):
                min_liquid_snow_per_hru: ArrayFloat64 = liquid_water_in_snow.min(axis=1)
                worst_hru_liquid_snow: int = int(np.argmin(min_liquid_snow_per_hru))
                min_value_liquid_snow: float = float(
                    min_liquid_snow_per_hru[worst_hru_liquid_snow]
                )
                fail(
                    worst_hru_liquid_snow,
                    f"Negative liquid water in snow detected (min value: {min_value_liquid_snow:.6e} m)",
                )

            # 6. Interflow and Groundwater recharge
            if np.any(err_flags == ERR_NAN_INTERFLOW):
                first_failing_hru_interflow: int = int(
                    np.where(np.any(np.isnan(interflow_m), axis=1))[0][0]
                )
                fail(first_failing_hru_interflow, "NaN interflow detected")
            if np.any(err_flags == ERR_NEG_INTERFLOW):
                min_interflow_per_hru: ArrayFloat32 = interflow_m.min(axis=1)
                worst_hru_interflow: int = int(np.argmin(min_interflow_per_hru))
                min_value_interflow: float = float(
                    min_interflow_per_hru[worst_hru_interflow]
                )
                fail(
                    worst_hru_interflow,
                    f"Negative interflow detected (min value: {min_value_interflow:.6e} m)",
                )

            if np.any(err_flags == ERR_NAN_GW_RECHARGE):
                first_failing_hru_groundwater_recharge: int = int(
                    np.where(np.isnan(groundwater_recharge_m))[0][0]
                )
                fail(
                    first_failing_hru_groundwater_recharge,
                    "NaN groundwater recharge detected",
                )
            if np.any(err_flags == ERR_NEG_GW_RECHARGE):
                worst_hru_groundwater_recharge: int = int(
                    np.argmin(groundwater_recharge_m)
                )
                value_groundwater_recharge: float = float(
                    groundwater_recharge_m[worst_hru_groundwater_recharge]
                )
                fail(
                    worst_hru_groundwater_recharge,
                    f"Negative groundwater recharge detected (value: {value_groundwater_recharge:.6e} m)",
                )

            # 7. Evaporation and transpiration
            if np.any(err_flags == ERR_NAN_EVAP_TRANS):
                nan_evap: ArrayBool = (
                    np.isnan(bare_soil_evaporation_m)
                    | np.isnan(transpiration_m)
                    | np.isnan(interception_evaporation_m)
                    | np.isnan(open_water_evaporation_m)
                )
                first_failing_hru_evaporation: int = int(np.where(nan_evap)[0][0])
                fail(
                    first_failing_hru_evaporation,
                    "NaN evaporation/transpiration flux detected",
                )

            if np.any(err_flags == ERR_NEG_BARE_SOIL):
                worst_hru_bare_soil_evaporation: int = int(
                    np.argmin(bare_soil_evaporation_m)
                )
                value_bare_soil_evaporation: float = float(
                    bare_soil_evaporation_m[worst_hru_bare_soil_evaporation]
                )
                fail(
                    worst_hru_bare_soil_evaporation,
                    f"Negative bare soil evaporation detected (value: {value_bare_soil_evaporation:.6e} m)",
                )
            if np.any(err_flags == ERR_NEG_TRANSPIRATION):
                worst_hru_transpiration: int = int(np.argmin(transpiration_m))
                value_transpiration: float = float(
                    transpiration_m[worst_hru_transpiration]
                )
                fail(
                    worst_hru_transpiration,
                    f"Negative transpiration detected (value: {value_transpiration:.6e} m)",
                )
            if np.any(err_flags == ERR_NEG_INTERCEPTION_EVAP):
                worst_hru_interception_evaporation: int = int(
                    np.argmin(interception_evaporation_m)
                )
                value_interception_evaporation: float = float(
                    interception_evaporation_m[worst_hru_interception_evaporation]
                )
                fail(
                    worst_hru_interception_evaporation,
                    f"Negative interception evaporation detected (value: {value_interception_evaporation:.6e} m)",
                )
            if np.any(err_flags == ERR_NEG_OPEN_WATER_EVAP):
                worst_hru_open_water_evaporation: int = int(
                    np.argmin(open_water_evaporation_m)
                )
                value_open_water_evaporation: float = float(
                    open_water_evaporation_m[worst_hru_open_water_evaporation]
                )
                fail(
                    worst_hru_open_water_evaporation,
                    f"Negative open water evaporation detected (value: {value_open_water_evaporation:.6e} m)",
                )

            # 8. Water balance imbalance
            if np.any(err_flags == ERR_WATER_IMBALANCE):
                water_imbalance_index: int = int(
                    np.where(err_flags == ERR_WATER_IMBALANCE)[0][0]
                )
                fail(
                    water_imbalance_index,
                    "Land surface water balance check failed",
                )

    def _check_soil_enthalpy_balance(
        self,
        *,
        soil_enthalpy_J_per_m2_prev: TwoDArrayFloat32,
        soil_boundary_enthalpy_flux_J_per_m2: ArrayFloat32,
        rain_advection_enthalpy_flux_J_per_m2: ArrayFloat32,
        evaporative_cooling_enthalpy_loss_J_per_m2: ArrayFloat32,
        interflow_enthalpy_loss_J_per_m2: ArrayFloat32,
        groundwater_recharge_enthalpy_loss_J_per_m2: ArrayFloat32,
        transpiration_enthalpy_loss_J_per_m2: ArrayFloat32,
    ) -> tuple[Literal[False], int] | tuple[Literal[True], None]:
        """Check closure of the daily soil enthalpy budget.

        Notes:
            Evaluates enthalpy balance closure in parallel without allocating temporary
            reduction arrays. Snowpack and canopy thermal stores are excluded because
            they are not represented as prognostic enthalpy state variables in this module.

        Args:
            soil_enthalpy_J_per_m2_prev: Soil enthalpy before the daily step (J/m2).
            soil_boundary_enthalpy_flux_J_per_m2: Net enthalpy flux from the
                implicit soil energy solver (J/m2, positive into soil).
            rain_advection_enthalpy_flux_J_per_m2: Enthalpy carried by newly added
                liquid water reaching the top-soil control volume during the day
                (J/m2, signed).
            evaporative_cooling_enthalpy_loss_J_per_m2: Latent cooling losses by
                evaporation/sublimation from the top layer (J/m2, positive out).
            interflow_enthalpy_loss_J_per_m2: Enthalpy exported by interflow
                (J/m2, positive out).
            groundwater_recharge_enthalpy_loss_J_per_m2: Enthalpy exported by
                groundwater recharge (J/m2, positive out).
            transpiration_enthalpy_loss_J_per_m2: Sensible enthalpy exported by
                transpiration withdrawals from soil/topwater (J/m2, positive out).

        Returns:
            A tuple with a boolean and the index of the max imbalance.
        """
        num_cells: int = soil_boundary_enthalpy_flux_J_per_m2.size
        soil_enthalpy_post: TwoDArrayFloat32 = self.HRU.var.soil_enthalpy_J_per_m2
        imbalances: ArrayFloat32 = np.empty(num_cells, dtype=np.float32)
        check_soil_enthalpy_balance_kernel(
            soil_enthalpy_prev=soil_enthalpy_J_per_m2_prev,
            soil_enthalpy_post=soil_enthalpy_post,
            soil_boundary_flux=soil_boundary_enthalpy_flux_J_per_m2,
            rain_advection_flux=rain_advection_enthalpy_flux_J_per_m2,
            evaporative_cooling_loss=evaporative_cooling_enthalpy_loss_J_per_m2,
            interflow_loss=interflow_enthalpy_loss_J_per_m2,
            groundwater_recharge_loss=groundwater_recharge_enthalpy_loss_J_per_m2,
            transpiration_loss=transpiration_enthalpy_loss_J_per_m2,
            imbalances=imbalances,
        )
        if np.isnan(imbalances).any():
            nan_index: int = int(np.where(np.isnan(imbalances))[0][0])
            self.model.logger.error(
                f"land surface enthalpy NaN values found. First NaN at index {nan_index}."
            )
            return False, nan_index

        max_absolute_imbalance_J_per_m2: float = float(np.abs(imbalances).max())
        if max_absolute_imbalance_J_per_m2 > 1e3:
            worst_hru: int = int(np.abs(imbalances).argmax())
            self.model.logger.error(
                f"land surface enthalpy {imbalances[worst_hru]} > tolerance 1000.0, max imbalance at index {worst_hru}."
            )
            return False, worst_hru

        return True, None

    def spinup(self) -> None:
        """Spinup function for the land surface module."""
        self.HRU.var.topwater_m = self.HRU.full_compressed(0.0, dtype=np.float32)
        self.HRU.var.daily_reference_evapotranspiration_grass_m = (
            self.HRU.full_compressed(0.003, dtype=np.float32)
        )

        n_cells: int = self.HRU.var.topwater_m.shape[0]
        self.HRU.var.snow_water_equivalent_m = np.zeros(
            (n_cells, N_SNOW_LAYERS), dtype=np.float64
        )
        self.HRU.var.liquid_water_in_snow_m = np.zeros(
            (n_cells, N_SNOW_LAYERS), dtype=np.float64
        )
        self.HRU.var.snow_enthalpy_J_per_m2 = np.zeros(
            (n_cells, N_SNOW_LAYERS), dtype=np.float32
        )
        # Initialise both layers with fresh-snow density; they compact over time.
        self.HRU.var.snow_density_kg_per_m3 = np.full(
            (n_cells, N_SNOW_LAYERS), FRESH_SNOW_DENSITY_KG_PER_M3, dtype=np.float32
        )
        self.HRU.var.snow_albedo = np.full(n_cells, SNOW_ALBEDO_FRESH, dtype=np.float32)
        self.HRU.var.interception_storage_m = self.HRU.full_compressed(
            0.0, dtype=np.float32
        )

        self.HRU.var.variable_runoff_shape_beta = self.HRU.full_compressed(
            np.nan,
            dtype=np.float32,
        )

        hillslope_length_m: ArrayFloat32 = self.hydrology.grid.load2d(
            self.model.files["grid"]["drainage/hillslope_length_m"]
        )

        self.HRU.var.hillslope_length_m = self.hydrology.to_HRU(data=hillslope_length_m)

        leaf_area_index_forest = self.hydrology.grid.load3d(
            self.model.files["other"]["vegetation/leaf_area_index_forest"]
        )
        forest_evergreen_fraction: ArrayFloat32 = self.hydrology.grid.load2d(
            self.model.files["grid"]["vegetation/forest_evergreen_fraction"]
        )
        leaf_area_index_forest = get_forest_leaf_area_index_with_evergreen(
            leaf_area_index_forest, forest_evergreen_fraction
        )
        self.HRU.var.leaf_area_index_forest = self.hydrology.to_HRU(
            data=leaf_area_index_forest
        )

        leaf_area_index_grassland_like = self.hydrology.grid.load3d(
            self.model.files["other"]["vegetation/leaf_area_index_grassland_like"],
        )
        self.HRU.var.leaf_area_index_grassland_like = self.hydrology.to_HRU(
            data=leaf_area_index_grassland_like
        )

        # Default follows AQUACROP recommendation, see reference manual for AquaCrop v7.1 – Chapter 3
        self.var.minimum_effective_root_depth_m = np.float32(0.25)

        self.setup_soil_properties()

    def setup_soil_properties(self) -> None:
        """Setup soil properties for the land surface module.

        Water contents, saturated conductivity and the solid fraction of each layer
        account for the volumetric coarse fragment (stone) content.

        """

        def load_soil_layers_to_HRU(filepath: Path) -> TwoDArrayFloat32:
            """Memory efficient loading of soil layers directly to HRU size.

            Args:
                filepath: Path to the .zarr file.

            Returns:
                Compressed array of shape (HRUs, layers).

            Raises:
                ValueError: If NaN values are found in the soil layer data.
            """
            output: TwoDArrayFloat32 = np.full(
                (self.HRU.compressed_size, N_SOIL_LAYERS), np.nan, dtype=np.float32
            )

            layers_data = read_grid(filepath, ndim=3, load=False)
            layer_hru: ArrayFloat32 = np.empty(
                self.HRU.compressed_size, dtype=np.float32
            )
            for i in range(N_SOILGRIDS_LAYERS):
                layer_data = layers_data[i]
                assert isinstance(layer_data, np.ndarray)
                if np.isnan(layer_data[~self.HRU.mask]).any():
                    raise ValueError(
                        f"Found NaN values in soil layer data. This likely means that something went wrong with the "
                        f"chunked interpolation of nan values in the build (setup_soil). Advice is to open the submask and "
                        f"soil data file: {filepath}."
                    )
                self.HRU.convert_subgrid_to_HRU_numba(
                    layer_data,
                    self.HRU.var.linear_mapping,
                    layer_hru,
                    nodatavalue=np.nan,
                    method="mean",
                    scaling=self.HRU.scaling,
                )
                output[:, i] = layer_hru
                del layer_data  # free memory immediately after use

            # The regolith layer below the SoilGrids profile has no data of its own;
            # it takes the properties of the deepest SoilGrids layer (100-200 cm).
            output[:, REGOLITH_LAYER_INDEX] = output[:, N_SOILGRIDS_LAYERS - 1]
            return output

        self.HRU.var.soil_layer_height_m = load_soil_layers_to_HRU(
            self.model.files["subgrid"]["soil/soil_layer_height_m"]
        )
        # Regolith layer: weathered rock between the bottom of the SoilGrids profile
        # (2 m) and the depth to bedrock, at most 2 m thick.
        depth_to_bedrock_m: ArrayFloat32 = self.HRU.convert_subgrid_to_HRU(
            read_grid(
                self.model.files["subgrid"]["soil/depth_to_bedrock_m"], ndim=2
            ).astype(np.float32),
            method="mean",
        )
        self.HRU.var.soil_layer_height_m[:, REGOLITH_LAYER_INDEX] = (
            get_regolith_layer_thickness_m(depth_to_bedrock_m)
        )
        # Thin the subsoil (layers 3 to 5) and the weathered rock below it on slopes,
        # before any quantities are derived from the layer heights: on steep slopes
        # erosion keeps both soil and weathered rock thin above the hard bedrock.
        self.HRU.var.soil_layer_height_m[:, 3:N_SOIL_LAYERS] *= (
            self.get_subsoil_depth_factor()[:, np.newaxis]
        )
        self.HRU.var.soil_layer_height_m[:, REGOLITH_LAYER_INDEX] = np.maximum(
            self.HRU.var.soil_layer_height_m[:, REGOLITH_LAYER_INDEX],
            REGOLITH_LAYER_MIN_THICKNESS_M,
        )
        organic_carbon_percentage = load_soil_layers_to_HRU(
            self.model.files["subgrid"]["soil/soil_organic_carbon_percentage"],
        )
        self.HRU.var.bulk_density_kg_per_dm3 = load_soil_layers_to_HRU(
            self.model.files["subgrid"]["soil/bulk_density_kg_per_dm3"],
        )
        self.HRU.var.silt_percentage = load_soil_layers_to_HRU(
            self.model.files["subgrid"]["soil/silt_percentage"],
        )
        self.HRU.var.clay_percentage = load_soil_layers_to_HRU(
            self.model.files["subgrid"]["soil/clay_percentage"],
        )

        # calculate sand content based on silt and clay content (together they should sum to 100%)
        self.HRU.var.sand_percentage = (
            100 - self.HRU.var.silt_percentage - self.HRU.var.clay_percentage
        )

        # Identify sealed land use
        is_sealed: ArrayBool = self.HRU.var.land_use_type == SEALED

        # The mapped bulk density of the forest topsoil is too high for the mapped
        # organic matter: lower it to what the organic matter implies (Adams, 1973).
        self.HRU.var.bulk_density_kg_per_dm3 = get_bulk_density_with_organic_matter(
            bulk_density_kg_per_dm3=self.HRU.var.bulk_density_kg_per_dm3,
            organic_matter_percentage=organic_carbon_percentage
            * ORGANIC_MATTER_PER_ORGANIC_CARBON,
            is_forest=self.HRU.var.land_use_type == FOREST,
        )

        # Prepare soil properties for pedotransfer functions, accounting for compaction in sealed areas
        # Top 30 cm is considered topsoil for unsealed areas (layers 0, 1, 2)
        is_top_soil: TwoDArrayBool = np.zeros_like(
            self.HRU.var.clay_percentage, dtype=bool
        )
        is_top_soil[:, 0:3] = True
        # For sealed surfaces, top layer lacks topsoil organic/root structure
        is_top_soil[is_sealed, :] = False

        # Compacted top layer (layer 0, 0-5 cm) for sealed surfaces: increase bulk density directly on HRU.var
        self.HRU.var.bulk_density_kg_per_dm3[is_sealed, 0] = np.minimum(
            np.float32(1.85),
            np.maximum(
                np.float32(1.75),
                self.HRU.var.bulk_density_kg_per_dm3[is_sealed, 0] * np.float32(1.3),
            ),
        )

        organic_carbon_pedotransfer: TwoDArrayFloat32 = np.copy(
            organic_carbon_percentage
        )
        # Construction/paving reduces top-layer organic matter
        organic_carbon_pedotransfer[is_sealed, 0] = np.float32(0.1)

        thetas: TwoDArrayFloat32 = thetas_toth(
            organic_carbon_percentage=organic_carbon_pedotransfer,
            bulk_density_kg_per_dm3=self.HRU.var.bulk_density_kg_per_dm3,
            is_top_soil=is_top_soil,
            clay=self.HRU.var.clay_percentage,
            silt=self.HRU.var.silt_percentage,
        )

        thetar: TwoDArrayFloat32 = thetar_brakensiek(
            sand=self.HRU.var.sand_percentage,
            clay=self.HRU.var.clay_percentage,
            thetas=thetas,
        )
        self.HRU.var.bubbling_pressure_m_positive = get_bubbling_pressure_m_positive(
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
            np.float32(-1.0),  # assuming field capacity is at -100 cm (pF 2) -> 1m
            self.HRU.var.bubbling_pressure_m_positive,
            thetas,
            thetar,
            self.HRU.var.lambda_pore_size_distribution,
        )

        thetawp: TwoDArrayFloat32 = get_soil_moisture_at_pressure(
            np.float32(
                -(10**2.2)
            ),  # assuming wilting point is at -10^4.2 cm (pF 4.2) -> -10^2.2 m
            self.HRU.var.bubbling_pressure_m_positive,
            thetas,
            thetar,
            self.HRU.var.lambda_pore_size_distribution,
        )

        # Stones hold no water: scale the water contents of the fine earth by the
        # fine-earth fraction. The retention curve shape (bubbling pressure, pore size
        # index) is a property of the fine earth and stays unchanged.
        coarse_fragments_percentage = load_soil_layers_to_HRU(
            self.model.files["subgrid"]["soil/coarse_fragments_percentage"],
        )
        # The regolith layer is weathered rock, not soil: mostly rock fragments.
        coarse_fragments_percentage[:, REGOLITH_LAYER_INDEX] = np.maximum(
            coarse_fragments_percentage[:, REGOLITH_LAYER_INDEX],
            REGOLITH_MIN_ROCK_FRAGMENTS_PERCENTAGE,
        )
        fine_earth_fraction, coarse_fragment_conductivity_factor = (
            get_coarse_fragment_factors(coarse_fragments_percentage)
        )
        thetas = thetas * fine_earth_fraction
        thetar = thetar * fine_earth_fraction
        thetafc = thetafc * fine_earth_fraction
        thetawp = thetawp * fine_earth_fraction

        self.HRU.var.water_content_saturated_m = (
            thetas * self.HRU.var.soil_layer_height_m
        )
        self.HRU.var.water_content_field_capacity_m = (
            thetafc * self.HRU.var.soil_layer_height_m
        )
        self.HRU.var.water_content_wilting_point_m = (
            thetawp * self.HRU.var.soil_layer_height_m
        )
        self.HRU.var.water_content_residual_m = (
            thetar * self.HRU.var.soil_layer_height_m
        )

        assert (self.HRU.var.water_content_residual_m > 0).all()
        assert (
            self.HRU.var.water_content_wilting_point_m
            > self.HRU.var.water_content_residual_m
        ).all()
        assert (
            self.HRU.var.water_content_field_capacity_m
            > self.HRU.var.water_content_wilting_point_m
        ).all()
        assert (
            self.HRU.var.water_content_saturated_m
            > self.HRU.var.water_content_field_capacity_m
        ).all()
        assert (
            self.HRU.var.water_content_saturated_m < self.HRU.var.soil_layer_height_m
        ).all()

        self.HRU.var.water_content_m = np.where(
            self.HRU.var.land_use_type[:, np.newaxis] < OPEN_WATER,
            (
                self.HRU.var.water_content_field_capacity_m
                - self.HRU.var.water_content_wilting_point_m
            )
            * np.float32(0.6)
            + self.HRU.var.water_content_wilting_point_m,
            self.HRU.var.water_content_residual_m,
        )
        # for paddy irrigation flooded paddy fields
        self.HRU.var.topwater_m = self.HRU.full_compressed(0, dtype=np.float32)
        self.HRU.var.daily_reference_evapotranspiration_grass_m = (
            self.HRU.full_compressed(0.003, dtype=np.float32)
        )

        self.HRU.var.horizontal_saturated_hydraulic_conductivity_m_per_s = kv_wosten(
            silt=self.HRU.var.silt_percentage,
            clay=self.HRU.var.clay_percentage,
            bulk_density_kg_per_dm3=self.HRU.var.bulk_density_kg_per_dm3,
            # the soil data give organic carbon, the equation takes organic matter
            organic_matter_percentage=organic_carbon_pedotransfer
            * ORGANIC_MATTER_PER_ORGANIC_CARBON,
            is_topsoil=is_top_soil,
        )
        self.HRU.var.horizontal_saturated_hydraulic_conductivity_m_per_s *= (
            coarse_fragment_conductivity_factor
        )

        self.HRU.var.horizontal_saturated_hydraulic_conductivity_m_per_s *= (
            self.model.config["hydrology"]["landsurface"][
                "saturated_hydraulic_conductivity_multiplier"
            ]
        )  # calibration parameter

        self.HRU.var.vertical_saturated_hydraulic_conductivity_m_per_s = (
            self.HRU.var.horizontal_saturated_hydraulic_conductivity_m_per_s
            * np.float32(
                self.model.config["hydrology"]["landsurface"][
                    "vertical_saturated_hydraulic_conductivity_multiplier"
                ]
            )
        )
        # Karst is a property of the bedrock, so it only changes the weathered-rock
        # layer below the soil: the epikarst drains faster, other weathered rock slower
        # (water perches on it and leaves as interflow). Karst and non-karst parts of a
        # cell drain side by side, so the factor is their area-weighted mean. Sealed
        # areas and open water have no karst drainage.
        karst_presence_hru: ArrayFloat32 = self.hydrology.to_HRU(
            data=self.hydrology.karst.karst_fraction
        ) * (
            (self.HRU.var.land_use_type != SEALED)
            & (self.HRU.var.land_use_type != OPEN_WATER)
        )
        self.HRU.var.vertical_saturated_hydraulic_conductivity_m_per_s[
            :, REGOLITH_LAYER_INDEX
        ] *= get_regolith_vertical_conductivity_factor(
            divisor=self.model.config["hydrology"]["landsurface"][
                "regolith_vertical_conductivity_divisor"
            ],
            karst_multiplier=self.model.config["hydrology"]["karst"][
                "vertical_conductivity_multiplier"
            ],
            karst_fraction=karst_presence_hru.astype(np.float32),
        )

        self.HRU.var.wetting_front_depth_m = np.full_like(self.HRU.var.topwater_m, 0.0)
        self.HRU.var.crust_rain_m = np.full_like(self.HRU.var.topwater_m, 0.0)
        self.HRU.var.wetting_front_moisture_deficit = np.full_like(
            self.HRU.var.topwater_m, 0.0
        )
        self.HRU.var.wetting_front_suction_head_m = np.full_like(
            self.HRU.var.topwater_m, 0.0
        )
        self.HRU.var.green_ampt_active_layer_idx = np.full_like(
            self.HRU.var.topwater_m, -1, dtype=np.int32
        )

        self.HRU.var.deep_soil_temperature_C = np.zeros(
            self.HRU.var.topwater_m.shape, dtype=np.float32
        )
        # Water table at the start of the previous day; set on the first day.
        self.HRU.var.groundwater_depth_previous_day_m = np.full(
            self.HRU.var.topwater_m.shape, np.nan, dtype=np.float32
        )
        # Daily minimum-temperature index of the last days per grid cell, for the
        # seasonal activity of trees; NaN before the start of the spinup.
        self.grid.var.minimum_temperature_index_record = np.tile(
            self.grid.full_compressed(np.nan, dtype=np.float32),
            (GROWING_SEASON_AVERAGING_DAYS, 1),
        )

        # Heat units of crops since sowing and those expected until harvest; set when
        # a field with a crop is first met (update_crop_heat_units).
        self.HRU.var.crop_heat_units_C_day = np.full(
            self.HRU.var.topwater_m.shape, np.nan, dtype=np.float32
        )
        self.HRU.var.crop_heat_units_at_harvest_C_day = np.full(
            self.HRU.var.topwater_m.shape, np.nan, dtype=np.float32
        )

        # Stones add mineral solids: bulk density of the whole layer (for heat capacity
        # and dry thermal conductivity) is the fine-earth bulk density over the
        # fine-earth volume plus mineral density over the stones.
        layer_bulk_density_kg_per_dm3: TwoDArrayFloat32 = (
            self.HRU.var.bulk_density_kg_per_dm3 * fine_earth_fraction
            + np.float32(RHO_MINERAL_KG_PER_M3 / 1000.0)
            * (np.float32(1.0) - fine_earth_fraction)
        ).astype(np.float32)
        self.HRU.var.solid_heat_capacity_J_per_m2_K = get_heat_capacity_solid_fraction(
            bulk_density_kg_per_dm3=layer_bulk_density_kg_per_dm3,
            layer_thickness_m=self.HRU.var.soil_layer_height_m,
        )

        # Enthalpy is relative to 0 °C, so by initializing it as 0
        # we are assuming that the initial soil temperature is 0 °C and that all water is liquid.
        self.HRU.var.soil_enthalpy_J_per_m2 = np.zeros_like(
            self.HRU.var.soil_layer_height_m, dtype=np.float32
        )

        self.HRU.var.solid_thermal_conductivity_W_per_m_K = (
            calculate_thermal_conductivity_solid_fraction_watt_per_meter_kelvin(
                sand_percentage=self.HRU.var.sand_percentage,
                silt_percentage=self.HRU.var.silt_percentage,
                clay_percentage=self.HRU.var.clay_percentage,
            )
        )

        # Pre-compute saturated and dry thermal conductivities.
        porosity = (
            self.HRU.var.water_content_saturated_m / self.HRU.var.soil_layer_height_m
        )
        self.HRU.var.thermal_conductivity_dry_soil_W_per_m_K = (
            calculate_thermal_conductivity_dry_soil_johansen_watt_per_meter_kelvin(
                layer_bulk_density_kg_per_dm3
            )
        )
        self.HRU.var.thermal_conductivity_saturated_unfrozen_W_per_m_K = calculate_thermal_conductivity_saturated_soil_johansen_watt_per_meter_kelvin(
            self.HRU.var.solid_thermal_conductivity_W_per_m_K,
            porosity,
            THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN,
        )
        self.HRU.var.thermal_conductivity_saturated_frozen_W_per_m_K = calculate_thermal_conductivity_saturated_soil_johansen_watt_per_meter_kelvin(
            self.HRU.var.solid_thermal_conductivity_W_per_m_K,
            porosity,
            THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN,
        )

        # soil water depletion fraction, Van Diepen et al., 1988: WOFOST 6.0, p.86, Doorenbos et. al 1978
        # crop groups for formular in van Diepen et al, 1988
        crop_group_forest: ArrayFloat32 = self.hydrology.grid.load2d(
            self.model.files["grid"]["vegetation/crop_group_number_forest"]
        )
        self.HRU.var.crop_group_number_forest = self.hydrology.to_HRU(
            data=crop_group_forest
        )

        crop_group_number_grassland_like: ArrayFloat32 = self.hydrology.grid.load2d(
            self.model.files["grid"]["vegetation/crop_group_number_grassland_like"]
        )
        self.HRU.var.crop_group_number_grassland_like = self.hydrology.to_HRU(
            data=crop_group_number_grassland_like
        )

        # Pack all 14 static 2D soil properties into a contiguous 3D array
        # to maximize hardware cache locality and eliminate redundant LLVM register descriptors.
        packed_soil_properties: ThreeDArrayFloat32 = np.empty(
            (self.HRU.compressed_size, N_STATIC_SOIL_PROPS, N_SOIL_LAYERS),
            dtype=np.float32,
        )
        packed_soil_properties[:, SOIL_PROP_WATER_CONTENT_RESIDUAL_M, :] = (
            self.HRU.var.water_content_residual_m
        )
        packed_soil_properties[:, SOIL_PROP_WATER_CONTENT_WILTING_POINT_M, :] = (
            self.HRU.var.water_content_wilting_point_m
        )
        packed_soil_properties[:, SOIL_PROP_WATER_CONTENT_FIELD_CAPACITY_M, :] = (
            self.HRU.var.water_content_field_capacity_m
        )
        packed_soil_properties[:, SOIL_PROP_WATER_CONTENT_SATURATED_M, :] = (
            self.HRU.var.water_content_saturated_m
        )
        packed_soil_properties[:, SOIL_PROP_SOLID_HEAT_CAPACITY_J_PER_M2_K, :] = (
            self.HRU.var.solid_heat_capacity_J_per_m2_K
        )
        packed_soil_properties[
            :, SOIL_PROP_THERMAL_CONDUCTIVITY_SATURATED_UNFROZEN_W_PER_M_K, :
        ] = self.HRU.var.thermal_conductivity_saturated_unfrozen_W_per_m_K
        packed_soil_properties[
            :, SOIL_PROP_THERMAL_CONDUCTIVITY_SATURATED_FROZEN_W_PER_M_K, :
        ] = self.HRU.var.thermal_conductivity_saturated_frozen_W_per_m_K
        packed_soil_properties[
            :, SOIL_PROP_THERMAL_CONDUCTIVITY_DRY_SOIL_W_PER_M_K, :
        ] = self.HRU.var.thermal_conductivity_dry_soil_W_per_m_K
        packed_soil_properties[:, SOIL_PROP_SAND_PERCENTAGE, :] = (
            self.HRU.var.sand_percentage
        )
        packed_soil_properties[:, SOIL_PROP_SOIL_LAYER_HEIGHT_M, :] = (
            self.HRU.var.soil_layer_height_m
        )
        packed_soil_properties[:, SOIL_PROP_BUBBLING_PRESSURE_M_POSITIVE, :] = (
            self.HRU.var.bubbling_pressure_m_positive
        )
        packed_soil_properties[:, SOIL_PROP_LAMBDA_PORE_SIZE_DISTRIBUTION, :] = (
            self.HRU.var.lambda_pore_size_distribution
        )
        packed_soil_properties[
            :,
            SOIL_PROP_HORIZONTAL_SATURATED_HYDRAULIC_CONDUCTIVITY_M_PER_S,
            :,
        ] = self.HRU.var.horizontal_saturated_hydraulic_conductivity_m_per_s
        packed_soil_properties[
            :,
            SOIL_PROP_VERTICAL_SATURATED_HYDRAULIC_CONDUCTIVITY_M_PER_S,
            :,
        ] = self.HRU.var.vertical_saturated_hydraulic_conductivity_m_per_s

        self.HRU.var.packed_soil_properties = packed_soil_properties

        # Point existing HRU variable attributes to slices of the packed array
        # to avoid holding duplicate arrays in memory while preserving downstream access.
        self.HRU.var.water_content_residual_m = self.HRU.var.packed_soil_properties[
            :, SOIL_PROP_WATER_CONTENT_RESIDUAL_M, :
        ]
        self.HRU.var.water_content_wilting_point_m = (
            self.HRU.var.packed_soil_properties[
                :, SOIL_PROP_WATER_CONTENT_WILTING_POINT_M, :
            ]
        )
        self.HRU.var.water_content_field_capacity_m = (
            self.HRU.var.packed_soil_properties[
                :, SOIL_PROP_WATER_CONTENT_FIELD_CAPACITY_M, :
            ]
        )
        self.HRU.var.water_content_saturated_m = self.HRU.var.packed_soil_properties[
            :, SOIL_PROP_WATER_CONTENT_SATURATED_M, :
        ]
        self.HRU.var.solid_heat_capacity_J_per_m2_K = (
            self.HRU.var.packed_soil_properties[
                :, SOIL_PROP_SOLID_HEAT_CAPACITY_J_PER_M2_K, :
            ]
        )
        self.HRU.var.thermal_conductivity_saturated_unfrozen_W_per_m_K = (
            self.HRU.var.packed_soil_properties[
                :, SOIL_PROP_THERMAL_CONDUCTIVITY_SATURATED_UNFROZEN_W_PER_M_K, :
            ]
        )
        self.HRU.var.thermal_conductivity_saturated_frozen_W_per_m_K = (
            self.HRU.var.packed_soil_properties[
                :, SOIL_PROP_THERMAL_CONDUCTIVITY_SATURATED_FROZEN_W_PER_M_K, :
            ]
        )
        self.HRU.var.thermal_conductivity_dry_soil_W_per_m_K = (
            self.HRU.var.packed_soil_properties[
                :, SOIL_PROP_THERMAL_CONDUCTIVITY_DRY_SOIL_W_PER_M_K, :
            ]
        )
        self.HRU.var.sand_percentage = self.HRU.var.packed_soil_properties[
            :, SOIL_PROP_SAND_PERCENTAGE, :
        ]
        self.HRU.var.soil_layer_height_m = self.HRU.var.packed_soil_properties[
            :, SOIL_PROP_SOIL_LAYER_HEIGHT_M, :
        ]
        self.HRU.var.bubbling_pressure_m_positive = self.HRU.var.packed_soil_properties[
            :, SOIL_PROP_BUBBLING_PRESSURE_M_POSITIVE, :
        ]
        self.HRU.var.lambda_pore_size_distribution = (
            self.HRU.var.packed_soil_properties[
                :, SOIL_PROP_LAMBDA_PORE_SIZE_DISTRIBUTION, :
            ]
        )
        self.HRU.var.horizontal_saturated_hydraulic_conductivity_m_per_s = (
            self.HRU.var.packed_soil_properties[
                :,
                SOIL_PROP_HORIZONTAL_SATURATED_HYDRAULIC_CONDUCTIVITY_M_PER_S,
                :,
            ]
        )
        self.HRU.var.vertical_saturated_hydraulic_conductivity_m_per_s = (
            self.HRU.var.packed_soil_properties[
                :,
                SOIL_PROP_VERTICAL_SATURATED_HYDRAULIC_CONDUCTIVITY_M_PER_S,
                :,
            ]
        )

    def get_subsoil_depth_factor(self) -> ArrayFloat32:
        """Get the slope-dependent scaling factor for the subsoil layer thickness per HRU.

        The factor is computed from the local slope of each subgrid cell (see
        get_slope_soil_depth_factor) and averaged over the subgrid cells of each HRU,
        because the slope averaged over a grid cell smooths out the valley sides where
        soil and weathered rock are thin.

        This is applied to the soil layer heights when the soil is set up in the
        spinup, so that all derived quantities (water capacities, initial water
        content, heat capacity, enthalpy) are consistent with the scaled profile.
        Runs inherit the scaled profile from the spinup checkpoint, so changing
        ``slope_dependent_soil_depth_factor`` requires a new spinup.

        Returns:
            Scaling factor per HRU (0.1 to 1.0). All ones if the factor is <= 0.

        """
        c_slope = float(
            self.model.config["hydrology"]["landsurface"][
                "slope_dependent_soil_depth_factor"
            ]
        )
        if c_slope <= 0.0:
            return np.ones(self.HRU.var.land_use_type.size, dtype=np.float32)
        subgrid_slope_m_per_m: TwoDArrayFloat32 = read_grid(
            self.model.files["subgrid"]["landsurface/slope_m_per_m"], ndim=2
        ).astype(np.float32)
        subgrid_factor: TwoDArrayFloat32 = get_slope_soil_depth_factor(
            subgrid_slope_m_per_m, c_slope
        )
        del subgrid_slope_m_per_m
        factor: ArrayFloat32 = self.HRU.convert_subgrid_to_HRU(
            subgrid_factor, method="mean"
        ).astype(np.float32)
        return factor

    def get_macropore_infiltration_capacity_m_per_s(self) -> ArrayFloat32:
        """Macropore infiltration capacity per HRU (m/s).

        The saturated conductivity of the macropores of the topsoil, from its clay
        content, bulk density and land use (Jarvis et al., 2013; see
        get_macropore_conductivity_jarvis), with the thickness-weighted mean clay
        content and bulk density of the top 30 cm. Tillage breaks the macropores of
        arable land, which is recognised by its owner (a farmer), also outside the
        growing season. Sealed land, paddy fields and open water have none.

        Returns:
            Macropore infiltration capacity (m/s), per HRU.
        """
        land_use_type = self.HRU.var.land_use_type
        is_cropland = self.HRU.var.land_owners != -1
        topsoil_thickness_m = self.HRU.var.soil_layer_height_m[:, :3]
        topsoil_clay_percentage = (
            self.HRU.var.clay_percentage[:, :3] * topsoil_thickness_m
        ).sum(axis=1) / topsoil_thickness_m.sum(axis=1)
        topsoil_bulk_density_kg_per_dm3 = (
            self.HRU.var.bulk_density_kg_per_dm3[:, :3] * topsoil_thickness_m
        ).sum(axis=1) / topsoil_thickness_m.sum(axis=1)
        capacity = get_macropore_conductivity_jarvis(
            clay_percentage=topsoil_clay_percentage,
            bulk_density_kg_per_dm3=topsoil_bulk_density_kg_per_dm3,
            is_arable=is_cropland,
        )
        has_macropores = (
            (land_use_type == FOREST)
            | (land_use_type == GRASSLAND_LIKE)
            | ((land_use_type == NON_PADDY_IRRIGATED) & is_cropland)
        )
        return np.where(has_macropores, capacity, np.float32(0.0)).astype(np.float32)

    def get_interflow_structural_conductivity_m_per_s(self) -> ArrayFloat32:
        """Lateral conductivity of the soil structure at the surface per HRU (m/s).

        Root and animal channels, pipes and loose aggregated topsoil carry most of the
        lateral flow on hillslopes; conductivities measured at the hillslope scale are
        one to three orders of magnitude above those of the soil matrix. Forest soils
        have the best developed structure, permanent grassland less, and tillage
        destroys it in cropland, which is recognised by its owner (a farmer), also
        outside the growing season. The unpaved part of sealed land is grass. Open
        water has none.

        Returns:
            Structural lateral conductivity at the surface (m/s), per HRU.
        """
        conductivity_m_per_day: dict[str, float] = self.model.config["hydrology"][
            "landsurface"
        ]["interflow_structural_conductivity_m_per_day"]
        land_use_type = self.HRU.var.land_use_type
        is_cropland = self.HRU.var.land_owners != -1
        conductivity = np.zeros(land_use_type.shape, dtype=np.float32)
        conductivity[land_use_type == FOREST] = conductivity_m_per_day["forest"]
        conductivity[
            ((land_use_type == GRASSLAND_LIKE) & ~is_cropland)
            | (land_use_type == SEALED)
        ] = conductivity_m_per_day["grassland"]
        conductivity[
            (
                (land_use_type == GRASSLAND_LIKE)
                | (land_use_type == NON_PADDY_IRRIGATED)
                | (land_use_type == PADDY_IRRIGATED)
            )
            & is_cropland
        ] = conductivity_m_per_day["cropland"]
        return conductivity / np.float32(86400.0)

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
        ArrayFloat32,
        float,
        ArrayFloat32,
        ArrayFloat32,
        np.float64,
    ]:
        """Step function for the land surface module.

        Executes the daily time step for the land surface, coordinating
        vegetation state, water demand, and the land surface model.

        Returns:
            A tuple containing:
            - potential_open_water_evaporation_m: Reference evapotranspiration for
                water per hour (m).
            - interflow_m: Lateral subsurface flow per hour (m/hour).
            - runoff_m: Surface runoff per hour (m/hour).
            - groundwater_recharge_m: Groundwater recharge per day (m).
            - capillar_rise_m: Capillary rise from groundwater per day (m).
            - groundwater_abstraction_m3: Groundwater abstraction for irrigation per day (m3).
            - channel_abstraction_m3: Surface water abstraction for irrigation per day (m3).
            - return_flow_m: Return flow to surface water per day (m).
            - total_water_demand_loss_m3: Total water demand loss per day (m3).
            - actual_evapotranspiration_m: Actual evapotranspiration per day (m).
            - sublimation_or_deposition_m: Sublimation or deposition per day (m).
            - pr_total_m3: Total precipitation volume per day (m3).

        Raises:
            AssertionError: If any of the balance checks fail.
        """
        if self.model.current_timestep == 0:
            # Evergreen part of the forest leaf area per grid cell, for the seasonal
            # activity of evergreen trees. Derived from the input data at every model
            # start, so it is not part of the saved state.
            self._leaf_area_index_forest_evergreen: ArrayFloat32 = (
                get_evergreen_leaf_area_index(
                    self.hydrology.grid.load3d(
                        self.model.files["other"]["vegetation/leaf_area_index_forest"]
                    ),
                    self.hydrology.grid.load2d(
                        self.model.files["grid"]["vegetation/forest_evergreen_fraction"]
                    ),
                )
            )
            self._snow_holding_capacity_m: ArrayFloat32 = get_snow_holding_capacity_m(
                self.HRU.var.slope_m_per_m,
                self.model.config["hydrology"]["landsurface"][
                    "max_snow_holding_capacity_m"
                ],
            )
            routing = self.hydrology.routing
            sequence: ArrayInt32 = routing._compressed_routing_seq
            self._snow_upstream_to_downstream: ArrayInt32 = np.ascontiguousarray(
                sequence[sequence >= 0][::-1]
            ).astype(np.int32)
            self._snow_downstream_cell: ArrayInt32 = np.ascontiguousarray(
                routing._compressed_ds
            ).astype(np.int32)
            sigma: float = FORCING_SOURCE_SIGMA_DEFAULTS[
                self.model.forcing.forcing_loaders["pr_kg_per_m2_per_s"].source
            ]
            self.rainfall_lookup_table: TwoDArrayFloat32 = (
                generate_rainfall_lookup_table(sigma)
            )
            # Scale topographic shape parameter beta with the user-defined calibration scale factor
            variable_runoff_shape_beta_scale: np.float32 = np.float32(
                self.model.config["hydrology"]["landsurface"][
                    "variable_runoff_shape_beta_multiplier"
                ]
            )
            surface_area_ratio = self.grid.load2d(
                self.model.files["grid"]["landsurface/surface_area_ratio"]
            )
            surface_area_ratio_hru: ArrayFloat32 = self.hydrology.to_HRU(
                surface_area_ratio
            )
            beta = (
                (surface_area_ratio_hru - np.float32(1.0)) + np.float32(0.2)
            ) * np.float32(5.0)
            self.HRU.var.variable_runoff_shape_beta[:] = (
                beta * variable_runoff_shape_beta_scale
            )
            assert not np.isnan(self.HRU.var.variable_runoff_shape_beta).any()
            assert (self.HRU.var.variable_runoff_shape_beta >= 0).all()

        # Snow above the holding capacity of a slope moves downslope along the
        # drainage direction (avalanches, glacier flow) before the land surface step.
        redistribute_snow_along_drainage_direction(
            snow_water_equivalent_m=self.HRU.var.snow_water_equivalent_m,
            snow_enthalpy_J_per_m2=self.HRU.var.snow_enthalpy_J_per_m2,
            snow_holding_capacity_m=self._snow_holding_capacity_m,
            hru_area_m2=self.HRU.var.cell_area,
            grid_to_HRU=self.HRU.var.grid_to_HRU,
            upstream_to_downstream=self._snow_upstream_to_downstream,
            downstream_cell=self._snow_downstream_cell,
        )

        timer = TimingModule("Land surface model")
        if __debug__:
            snow_water_equivalent_prev: TwoDArrayFloat64 = (
                self.HRU.var.snow_water_equivalent_m.copy()
            )
            liquid_water_in_snow_prev: TwoDArrayFloat64 = (
                self.HRU.var.liquid_water_in_snow_m.copy()
            )
            interception_storage_prev: ArrayFloat32 = (
                self.HRU.var.interception_storage_m.copy()
            )
            topwater_m_prev: ArrayFloat32 = self.HRU.var.topwater_m.copy()
            snow_enthalpy_J_per_m2_prev: TwoDArrayFloat32 = (
                self.HRU.var.snow_enthalpy_J_per_m2.copy()
            )
            snow_density_kg_per_m3_prev: TwoDArrayFloat32 = (
                self.HRU.var.snow_density_kg_per_m3.copy()
            )
            soil_enthalpy_J_per_m2_prev: TwoDArrayFloat32 = (
                self.HRU.var.soil_enthalpy_J_per_m2.copy()
            )
            deep_soil_temperature_C_prev: ArrayFloat32 = (
                self.HRU.var.deep_soil_temperature_C.copy()
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
            water_content_m_prev: TwoDArrayFloat32 = self.HRU.var.water_content_m.copy()

        timer.finish_split("Preprocessing")

        if self.model.current_timestep == 0:
            # crop tables and the temperature climatology do not change during a run
            self._crop_stage_lengths = np.column_stack(
                [
                    self.model.agents.crop_farmers.var.crop_data["l_ini"],
                    self.model.agents.crop_farmers.var.crop_data["l_dev"],
                    self.model.agents.crop_farmers.var.crop_data["l_mid"],
                    self.model.agents.crop_farmers.var.crop_data["l_late"],
                ]
            )
            get_crop_sub_stage = (
                self.model.agents.crop_farmers.var.crop_data_type == "GAEZ"
            )
            if get_crop_sub_stage:
                self._crop_sub_stage_lengths = np.column_stack(
                    [
                        self.model.agents.crop_farmers.var.crop_data["d1"],
                        self.model.agents.crop_farmers.var.crop_data["d2a"],
                        self.model.agents.crop_farmers.var.crop_data["d2b"],
                        self.model.agents.crop_farmers.var.crop_data["d3a"],
                        self.model.agents.crop_farmers.var.crop_data["d3b"],
                        self.model.agents.crop_farmers.var.crop_data["d4"],
                    ]
                ).astype(np.int32)
            else:
                self._crop_sub_stage_lengths = np.full(
                    (self.model.agents.crop_farmers.var.crop_data.shape[0], 6),
                    -1,
                    dtype=np.int32,
                )

            self._crop_factor_per_crop_stage = np.column_stack(
                [
                    self.model.agents.crop_farmers.var.crop_data["kc_initial"],
                    self.model.agents.crop_farmers.var.crop_data["kc_mid"],
                    self.model.agents.crop_farmers.var.crop_data["kc_end"],
                ]
            )

            self._crop_root_depths = np.column_stack(
                [
                    self.model.agents.crop_farmers.var.crop_data["rd_rain"],
                    self.model.agents.crop_farmers.var.crop_data["rd_irr"],
                ]
            )
            self._crop_base_temperature_C = (
                self.model.agents.crop_farmers.var.crop_data[
                    "base_temperature_C"
                ].to_numpy(dtype=np.float32)
            )
            # Temperature climatology for the heat units crops expect in their season
            (
                self._air_temperature_dekadal_mean_C,
                self._air_temperature_dekadal_standard_deviation_C,
            ) = self.model.forcing["tas_2m_K"].get_dekadal_climatology_of_daily_mean_C()
            # dekad of each day of a leap year
            self._dekad_per_day_of_year = np.array(
                [
                    map_date_to_dekad(datetime(2000, 1, 1) + timedelta(days=day))
                    for day in range(366)
                ],
                dtype=np.int32,
            )

        crop_stage_lengths: np.ndarray = self._crop_stage_lengths
        crop_sub_stage_lengths: np.ndarray = self._crop_sub_stage_lengths
        crop_factor_per_crop_stage: np.ndarray = self._crop_factor_per_crop_stage
        crop_root_depths: np.ndarray = self._crop_root_depths
        get_crop_sub_stage: bool = (
            self.model.agents.crop_farmers.var.crop_data_type == "GAEZ"
        )

        dekad: int = map_date_to_dekad(self.model.current_time)
        mean_daily_air_temperature_C: ArrayFloat32 = (
            self.grid.tas_2m_K.mean(axis=1) - np.float32(273.15)
        ).astype(np.float32)[self.HRU.var.HRU_to_grid]
        evergreen_activity: ArrayFloat32 = update_canopy_temperature_acclimation(
            self.grid.var.minimum_temperature_index_record,
            (self.grid.tas_2m_K.min(axis=1) - np.float32(273.15)).astype(np.float32),
        )

        wind_2m_m_per_s, minimum_relative_humidity_percent = (
            get_daily_wind_and_minimum_relative_humidity(
                self.grid.tas_2m_K,
                self.grid.dewpoint_tas_2m_K,
                self.grid.wind_u10m_m_per_s,
                self.grid.wind_v10m_m_per_s,
            )
        )

        crop_progress: ArrayFloat32 = update_crop_heat_units(
            crop_map=self.HRU.var.crop_map,
            crop_age_days_map=self.HRU.var.crop_age_days_map,
            crop_harvest_age_days=self.HRU.var.crop_harvest_age_days,
            crop_heat_units_C_day=self.HRU.var.crop_heat_units_C_day,
            crop_heat_units_at_harvest_C_day=self.HRU.var.crop_heat_units_at_harvest_C_day,
            crop_base_temperature_C=self._crop_base_temperature_C,
            mean_daily_air_temperature_C=mean_daily_air_temperature_C,
            HRU_to_grid=self.HRU.var.HRU_to_grid,
            air_temperature_dekadal_mean_C=self._air_temperature_dekadal_mean_C,
            air_temperature_dekadal_standard_deviation_C=self._air_temperature_dekadal_standard_deviation_C,
            dekad_per_day_of_year=self._dekad_per_day_of_year,
            day_of_year=self.model.current_time.timetuple().tm_yday - 1,
        )

        (
            crop_factor,
            basal_crop_factor,
            maximum_crop_factor,
            root_depth_m,
            crop_sub_stage,
            leaf_area_index,
            interception_capacity_m,
        ) = get_crop_factors_and_root_depths_and_lai(
            land_use_map=self.HRU.var.land_use_type,
            leaf_area_index_forest=self.HRU.var.leaf_area_index_forest[dekad],
            leaf_area_index_grassland_like=self.HRU.var.leaf_area_index_grassland_like[
                dekad
            ],
            leaf_area_index_forest_evergreen=self._leaf_area_index_forest_evergreen[
                self.HRU.var.HRU_to_grid
            ],
            evergreen_activity=evergreen_activity[self.HRU.var.HRU_to_grid],
            wind_2m_m_per_s=wind_2m_m_per_s[self.HRU.var.HRU_to_grid],
            minimum_relative_humidity_percent=minimum_relative_humidity_percent[
                self.HRU.var.HRU_to_grid
            ],
            crop_map=self.HRU.var.crop_map,
            crop_progress=crop_progress,
            land_owners=self.HRU.var.land_owners,
            crop_stage_lengths=crop_stage_lengths,
            crop_sub_stage_lengths=crop_sub_stage_lengths,
            crop_factor_per_crop_stage=crop_factor_per_crop_stage,
            crop_root_depths=crop_root_depths,
            get_crop_sub_stage=get_crop_sub_stage,
            grassland_root_depth_m=np.float32(
                self.model.config["hydrology"]["landsurface"]["grassland_root_depth_m"]
            ),
        )

        crop_factor *= self.model.config["hydrology"]["landsurface"][
            "crop_factor_multiplier"
        ]  # calibration parameter
        basal_crop_factor *= self.model.config["hydrology"]["landsurface"][
            "crop_factor_multiplier"
        ]
        maximum_crop_factor *= self.model.config["hydrology"]["landsurface"][
            "crop_factor_multiplier"
        ]

        timer.finish_split("Input preparation")

        (
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow_m,  # from all sources
            irrigation_loss_to_evaporation_m,
            total_water_demand_loss_m3,
            actual_irrigation_consumption_m,
        ) = self.hydrology.water_demand.step(root_depth_m)

        timer.finish_split("Water demand")

        # Depth of the water table below each HRU (m): the groundwater of a cell is
        # connected, the land of each elevation zone lies at its own height above it
        groundwater_depth_m: ArrayFloat32 = (
            self.hydrology.groundwater.get_groundwater_depth_per_zone(
                self.HRU.var.HRU_to_grid, self.HRU.var.elevation_zone
            )
        )
        # The soil follows the water table one day behind (see land_surface_model); on
        # the first day of the spinup there is no previous day yet.
        is_first_day: ArrayBool = np.isnan(
            self.HRU.var.groundwater_depth_previous_day_m
        )
        self.HRU.var.groundwater_depth_previous_day_m[is_first_day] = (
            groundwater_depth_m[is_first_day]
        )

        # Available groundwater remaining after human abstractions (m3)
        remaining_available_groundwater_m3: ArrayFloat64 = np.maximum(
            0.0,
            self.hydrology.groundwater.available_groundwater_m3
            - groundwater_abstraction_m3,
        )
        max_capillary_rise_m: ArrayFloat32 = (
            remaining_available_groundwater_m3 / self.grid.var.cell_area
        ).astype(np.float32)

        # Load temperature at 2m from grid; deep soil temperature boundary condition is
        # updated directly inside the land_surface_model physics kernel using an EMA.
        tas_2m_K: TwoDArrayFloat32 = self.grid.tas_2m_K

        pr_kg_per_m2_per_s: TwoDArrayFloat32 = self.grid.pr_kg_per_m2_per_s
        HRU_to_grid: ArrayInt32 = self.HRU.var.HRU_to_grid

        land_surface_inputs: LandSurfaceInputs = self._build_land_surface_inputs(
            unix_time_seconds=np.int64(self.model.current_time.timestamp()),
            root_depth_m=root_depth_m,
            interception_capacity_m=interception_capacity_m,
            pr_kg_per_m2_per_s=pr_kg_per_m2_per_s,
            tas_2m_K=tas_2m_K,
            crop_factor=crop_factor,
            basal_crop_factor=basal_crop_factor,
            maximum_crop_factor=maximum_crop_factor,
            actual_irrigation_consumption_m=actual_irrigation_consumption_m,
            groundwater_depth_previous_day_m=self.HRU.var.groundwater_depth_previous_day_m,
            groundwater_depth_m=groundwater_depth_m,
            deep_soil_temperature_C=self.HRU.var.deep_soil_temperature_C,
            leaf_area_index=leaf_area_index,
            max_capillary_rise_m=max_capillary_rise_m,
            HRU_to_grid=HRU_to_grid,
        )

        timer.finish_split("Prepare inputs for land surface model")

        (
            rain_m,
            snow_m,
            topwater_m,
            reference_evapotranspiration_grass_m,
            potential_open_water_evaporation_m,
            snow_water_equivalent_m,
            liquid_water_in_snow_m,
            sublimation_or_deposition_m,
            snow_enthalpy_J_per_m2,
            snow_density_kg_per_m3,
            interception_storage_m,
            interception_evaporation_m,
            open_water_evaporation_m,
            runoff_m,
            groundwater_recharge_m,
            capillar_rise_m,
            interflow_m,
            bare_soil_evaporation_m,
            transpiration_m,
            potential_transpiration_m,
            potential_evapotranspiration_m,
            soil_boundary_enthalpy_flux_J_per_m2,
            rain_advection_enthalpy_flux_J_per_m2,
            evaporative_cooling_enthalpy_loss_J_per_m2,
            interflow_enthalpy_loss_J_per_m2,
            groundwater_recharge_enthalpy_loss_J_per_m2,
            transpiration_enthalpy_loss_J_per_m2,
            top_soil_infiltration_m,
            top_soil_rise_from_layer_2_m,
            top_soil_percolation_to_layer_2_m,
            top_soil_transpiration_m,
            evapotranspiration_m,
            dew_m,
        ) = land_surface_model(**land_surface_inputs._asdict())
        self.HRU.var.groundwater_depth_previous_day_m = groundwater_depth_m.copy()

        self.HRU.var.topwater_m = topwater_m
        self.HRU.var.snow_water_equivalent_m = snow_water_equivalent_m
        self.HRU.var.liquid_water_in_snow_m = liquid_water_in_snow_m
        self.HRU.var.snow_enthalpy_J_per_m2 = snow_enthalpy_J_per_m2
        self.HRU.var.snow_density_kg_per_m3 = snow_density_kg_per_m3
        self.HRU.var.interception_storage_m = interception_storage_m
        self.HRU.var.daily_reference_evapotranspiration_grass_m = (
            reference_evapotranspiration_grass_m
        )
        local_variables_to_report: list[str] = self.local_variables_to_report
        if "evapotranspiration_m" in local_variables_to_report:
            evapotranspiration_m = (
                evapotranspiration_m
                + (irrigation_loss_to_evaporation_m / np.float32(24.0))[:, np.newaxis]
            ).transpose()

        timer.finish_split("Land surface model")

        # with open("output.ll", "w") as f:
        #     for signature, llvm_code in land_surface_model.inspect_llvm().items():
        #         f.write(f"; --- Signature: {signature} ---\n")
        #         f.write(llvm_code)

        # with open("output.asm", "w") as f:
        #     for signature, asm_code in land_surface_model.inspect_asm().items():
        #         f.write(f"; --- Signature: {signature} ---\n")
        #         f.write(asm_code)

        if __debug__:
            self._check_land_surface_water(
                land_surface_inputs=land_surface_inputs,
                water_content_m_prev=water_content_m_prev,
                topwater_m_prev=topwater_m_prev,
                snow_water_equivalent_prev=snow_water_equivalent_prev,
                liquid_water_in_snow_prev=liquid_water_in_snow_prev,
                snow_enthalpy_J_per_m2_prev=snow_enthalpy_J_per_m2_prev,
                snow_density_kg_per_m3_prev=snow_density_kg_per_m3_prev,
                interception_storage_prev=interception_storage_prev,
                soil_enthalpy_J_per_m2_prev=soil_enthalpy_J_per_m2_prev,
                deep_soil_temperature_C_prev=deep_soil_temperature_C_prev,
                wetting_front_depth_prev=wetting_front_depth_prev,
                wetting_front_suction_head_prev=wetting_front_suction_head_prev,
                wetting_front_moisture_deficit_prev=wetting_front_moisture_deficit_prev,
                green_ampt_active_layer_idx_prev=green_ampt_active_layer_idx_prev,
                pr_kg_per_m2_per_s=pr_kg_per_m2_per_s,
                actual_irrigation_consumption_m=actual_irrigation_consumption_m,
                capillar_rise_m=capillar_rise_m,
                sublimation_or_deposition_m=sublimation_or_deposition_m,
                interception_evaporation_m=interception_evaporation_m,
                dew_m=dew_m,
                open_water_evaporation_m=open_water_evaporation_m,
                runoff_m=runoff_m,
                interflow_m=interflow_m,
                groundwater_recharge_m=groundwater_recharge_m,
                bare_soil_evaporation_m=bare_soil_evaporation_m,
                transpiration_m=transpiration_m,
            )

            enthalpy_balance_result = self._check_soil_enthalpy_balance(
                soil_enthalpy_J_per_m2_prev=soil_enthalpy_J_per_m2_prev,
                soil_boundary_enthalpy_flux_J_per_m2=soil_boundary_enthalpy_flux_J_per_m2,
                rain_advection_enthalpy_flux_J_per_m2=rain_advection_enthalpy_flux_J_per_m2,
                evaporative_cooling_enthalpy_loss_J_per_m2=evaporative_cooling_enthalpy_loss_J_per_m2,
                interflow_enthalpy_loss_J_per_m2=interflow_enthalpy_loss_J_per_m2,
                groundwater_recharge_enthalpy_loss_J_per_m2=groundwater_recharge_enthalpy_loss_J_per_m2,
                transpiration_enthalpy_loss_J_per_m2=transpiration_enthalpy_loss_J_per_m2,
            )

            is_enthalpy_balanced, enthalpy_imbalance_index = enthalpy_balance_result

            if not is_enthalpy_balanced and enthalpy_imbalance_index is not None:
                error_inputs = self._snapshot_land_surface_inputs_for_error(
                    land_surface_inputs=land_surface_inputs,
                    water_content_m_prev=water_content_m_prev,
                    topwater_m_prev=topwater_m_prev,
                    snow_water_equivalent_prev=snow_water_equivalent_prev,
                    liquid_water_in_snow_prev=liquid_water_in_snow_prev,
                    snow_enthalpy_J_per_m2_prev=snow_enthalpy_J_per_m2_prev,
                    snow_density_kg_per_m3_prev=snow_density_kg_per_m3_prev,
                    interception_storage_prev=interception_storage_prev,
                    soil_enthalpy_J_per_m2_prev=soil_enthalpy_J_per_m2_prev,
                    deep_soil_temperature_C_prev=deep_soil_temperature_C_prev,
                    wetting_front_depth_prev=wetting_front_depth_prev,
                    wetting_front_suction_head_prev=wetting_front_suction_head_prev,
                    wetting_front_moisture_deficit_prev=wetting_front_moisture_deficit_prev,
                    green_ampt_active_layer_idx_prev=green_ampt_active_layer_idx_prev,
                    index=enthalpy_imbalance_index,
                )
                diag_path = (
                    self.model.diagnostics_folder
                    / f"diagnostic_enthalpy_error_cell_{enthalpy_imbalance_index}.npz"
                )
                np.savez(
                    diag_path,
                    **error_inputs._asdict(),
                )
                self.model.logger.error(
                    f"Enthalpy imbalance detected at index {enthalpy_imbalance_index}. Diagnostic data exported to {diag_path}"
                )
                raise AssertionError(
                    f"Land surface enthalpy balance check failed at HRU index {enthalpy_imbalance_index}. Land use type: {self.HRU.var.land_use_type[enthalpy_imbalance_index]}"
                )

        timer.finish_split("Water and enthalpy balance checks")

        # Transpiration from groundwater: where the water table is within reach of
        # the roots (valley floors, drained lowland), plants take the transpiration
        # demand that the soil did not meet from groundwater. Not under snow or on
        # frozen soil, as for transpiration from the soil.
        # Uptake declines linearly from full with the water table at the surface to
        # zero at the extinction depth (rooting depth plus the capillary reach), as
        # in the MODFLOW evapotranspiration package.
        groundwater_reach_fraction: ArrayFloat32 = np.clip(
            1.0 - groundwater_depth_m / (root_depth_m + CAPILLARY_REACH_M), 0.0, 1.0
        ).astype(np.float32)
        plants_are_active = (
            self.HRU.var.snow_water_equivalent_m.sum(axis=1) <= 0.001
        ) & (
            get_frozen_fraction_from_enthalpy(
                enthalpy_J_per_m2=self.HRU.var.soil_enthalpy_J_per_m2[:, 0],
                solid_heat_capacity_J_per_m2_K=self.HRU.var.solid_heat_capacity_J_per_m2_K[
                    :, 0
                ],
                water_content_m=self.HRU.var.water_content_m[:, 0],
                topwater_m=self.HRU.var.topwater_m,
            )
            <= 0.0
        )
        groundwater_transpiration_m: ArrayFloat32 = np.where(
            plants_are_active,
            np.minimum(
                groundwater_reach_fraction
                * np.maximum(potential_transpiration_m - transpiration_m, 0.0),
                # groundwater left in the cell after wells and capillary rise
                np.maximum(max_capillary_rise_m[HRU_to_grid] - capillar_rise_m, 0.0),
            ),
            0.0,
        ).astype(np.float32)
        transpiration_m = transpiration_m + groundwater_transpiration_m
        # taken from groundwater with the capillary rise
        capillar_rise_m = capillar_rise_m + groundwater_transpiration_m

        # evaporated irrigation losses (e.g. sprinkler drift) never reach the crop,
        # so they do not count towards the crop evapotranspiration used for yield
        actual_evapotranspiration_land_m: ArrayFloat32 = (
            interception_evaporation_m
            + open_water_evaporation_m
            + transpiration_m
            + bare_soil_evaporation_m
        )
        actual_evapotranspiration_m: ArrayFloat32 = (
            actual_evapotranspiration_land_m + irrigation_loss_to_evaporation_m
        )

        accumulate_crop_evapotranspiration(
            crop_map=self.HRU.var.crop_map,
            crop_sub_stage=crop_sub_stage,
            actual_evapotranspiration_m=actual_evapotranspiration_land_m,
            potential_evapotranspiration_m=potential_evapotranspiration_m,
            actual_evapotranspiration_crop_life=self.HRU.var.actual_evapotranspiration_crop_life,
            potential_evapotranspiration_crop_life=self.HRU.var.potential_evapotranspiration_crop_life,
            actual_evapotranspiration_crop_life_per_crop_stage=self.HRU.var.actual_evapotranspiration_crop_life_per_crop_stage,
            potential_evapotranspiration_crop_life_per_crop_stage=self.HRU.var.potential_evapotranspiration_crop_life_per_crop_stage,
        )

        potential_open_water_evaporation_m = self.hydrology.to_grid(
            HRU_data=potential_open_water_evaporation_m,
        )
        if __debug__:
            assert (potential_open_water_evaporation_m >= 0).all()

        self.model.agents.crop_farmers.save_pr_and_water_deficit(
            reference_evapotranspiration_grass_m, pr_kg_per_m2_per_s
        )

        timer.finish_split("Finalization")

        if "top_soil_frozen_fraction" in local_variables_to_report:
            top_soil_frozen_fraction: ArrayFloat32 = get_frozen_fraction_from_enthalpy(
                enthalpy_J_per_m2=self.HRU.var.soil_enthalpy_J_per_m2[:, 0],
                solid_heat_capacity_J_per_m2_K=self.HRU.var.solid_heat_capacity_J_per_m2_K[
                    :, 0
                ],
                water_content_m=self.HRU.var.water_content_m[:, 0],
                topwater_m=self.HRU.var.topwater_m,
            )

        if "top_soil_evaporation_m" in local_variables_to_report:
            top_soil_evaporation_m: ArrayFloat32 = (
                bare_soil_evaporation_m + open_water_evaporation_m
            )
        if "runoff_m_daily" in local_variables_to_report:
            runoff_m_daily: ArrayFloat32 = runoff_m.sum(axis=1)
        # Diagnostic output only: daily runoff, interflow and groundwater recharge of one
        # land use or elevation zone, zero for the other HRUs (e.g. runoff_sealed_m).
        land_use_diagnostics: dict[str, ArrayFloat32] = {}
        requested_diagnostics: list[tuple[str, str]] = [
            (flux_name, area_name)
            for flux_name in ("runoff", "interflow", "groundwater_recharge")
            for area_name in (
                "forest",
                "grassland",
                "farmland",
                "sealed",
                "open_water",
                "bottom_zone",
                "upper_zones",
            )
            if f"{flux_name}_{area_name}_m" in local_variables_to_report
        ]
        if requested_diagnostics:
            is_soil_land: ArrayBool = self.HRU.var.land_use_type < SEALED
            area_masks: dict[str, ArrayBool] = {
                "forest": self.HRU.var.land_use_type == FOREST,
                "grassland": (self.HRU.var.land_use_type == GRASSLAND_LIKE)
                & (self.HRU.var.land_owners == -1),
                "farmland": self.HRU.var.land_owners != -1,
                "sealed": self.HRU.var.land_use_type == SEALED,
                "open_water": self.HRU.var.land_use_type == OPEN_WATER,
                # soil-covered land in the lowest elevation zone and above it
                "bottom_zone": is_soil_land & (self.HRU.var.elevation_zone == 0),
                "upper_zones": is_soil_land & (self.HRU.var.elevation_zone > 0),
            }
            daily_fluxes_m: dict[str, ArrayFloat32] = {
                "runoff": runoff_m.sum(axis=1),
                "interflow": interflow_m.sum(axis=1),
                "groundwater_recharge": groundwater_recharge_m,
            }
            for flux_name, area_name in requested_diagnostics:
                land_use_diagnostics[f"{flux_name}_{area_name}_m"] = np.where(
                    area_masks[area_name], daily_fluxes_m[flux_name], np.float32(0.0)
                ).astype(np.float32)
        if "soil_temperature_C" in local_variables_to_report:
            # Temperature as a diagnostic output only; the kernel is enthalpy-only.
            soil_temperature_C: TwoDArrayFloat32 = np.full_like(
                self.HRU.var.soil_enthalpy_J_per_m2,
                np.float32(np.nan),
                dtype=np.float32,
            )
            soil_temperature_C[:, 0] = get_temperature_from_enthalpy(
                enthalpy_J_per_m2=self.HRU.var.soil_enthalpy_J_per_m2[:, 0],
                solid_heat_capacity_J_per_m2_K=self.HRU.var.solid_heat_capacity_J_per_m2_K[
                    :, 0
                ],
                water_content_m=self.HRU.var.water_content_m[:, 0],
                topwater_m=self.HRU.var.topwater_m,
            )
            soil_temperature_C[:, 1:] = get_temperature_from_enthalpy(
                enthalpy_J_per_m2=self.HRU.var.soil_enthalpy_J_per_m2[:, 1:],
                solid_heat_capacity_J_per_m2_K=self.HRU.var.solid_heat_capacity_J_per_m2_K[
                    :, 1:
                ],
                water_content_m=self.HRU.var.water_content_m[:, 1:],
            )

        runoff_m = runoff_m.transpose()
        interflow_m = interflow_m.transpose()

        self.report({**locals(), **land_use_diagnostics})

        timer.finish_split("Reporting")

        if self.model.timing:
            self.model.logger.debug(timer)

        if __debug__:
            pr_total_m3: np.float64 = (
                (
                    pr_kg_per_m2_per_s.astype(np.float64).mean(axis=1)
                    * self.grid.var.cell_area
                ).sum()  # kg/s
                * 0.001  # to m3/s
                * (24 * 3600.0)  # to m3/day
            )

        else:
            pr_total_m3 = np.float64(np.nan)

        return (
            potential_open_water_evaporation_m,
            interflow_m,
            runoff_m,
            groundwater_recharge_m,
            capillar_rise_m,
            groundwater_abstraction_m3,
            channel_abstraction_m3,
            return_flow_m,
            total_water_demand_loss_m3,
            actual_evapotranspiration_m,
            sublimation_or_deposition_m,
            pr_total_m3,
        )
