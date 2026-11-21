"""Fast parallel validation kernels for the land surface model.

The kernels check states and fluxes for NaN and negative values and the closure of
the daily water and soil enthalpy balances per HRU, without allocating temporary
arrays. The LandSurface module reports the first failing HRU.
"""

import numpy as np
from numba import njit, prange

from geb.geb_types import (
    ArrayFloat32,
    ArrayInt8,
    ArrayInt32,
    TwoDArrayFloat32,
    TwoDArrayFloat64,
)
from geb.numba_config import NUMBA_CACHE

from .constants import N_SOIL_LAYERS

# Error status codes for fast parallel water balance and state validation checks:
ERR_NONE: int = 0
ERR_NAN_TOPWATER: int = 1
ERR_NEG_TOPWATER: int = 2
ERR_NAN_RUNOFF: int = 3
ERR_NEG_RUNOFF: int = 4
ERR_NAN_WATER_CONTENT: int = 5
ERR_NEG_WATER_CONTENT: int = 6
ERR_NAN_INTERCEPTION: int = 7
ERR_NEG_INTERCEPTION: int = 8
ERR_NAN_SWE: int = 9
ERR_NEG_SWE: int = 10
ERR_NAN_LIQUID_SNOW: int = 11
ERR_NEG_LIQUID_SNOW: int = 12
ERR_NAN_INTERFLOW: int = 13
ERR_NEG_INTERFLOW: int = 14
ERR_NAN_GW_RECHARGE: int = 15
ERR_NEG_GW_RECHARGE: int = 16
ERR_NAN_EVAP_TRANS: int = 17
ERR_NEG_BARE_SOIL: int = 18
ERR_NEG_TRANSPIRATION: int = 19
ERR_NEG_INTERCEPTION_EVAP: int = 20
ERR_NEG_OPEN_WATER_EVAP: int = 21
ERR_WATER_IMBALANCE: int = 22


@njit(cache=NUMBA_CACHE, fastmath=False)
def validate_and_check_water_balance_kernel(
    topwater_m: ArrayFloat32,
    runoff_m: TwoDArrayFloat32,
    water_content_m: TwoDArrayFloat32,
    interception_storage_m: ArrayFloat32,
    snow_water_equivalent_m: TwoDArrayFloat64,
    liquid_water_in_snow_m: TwoDArrayFloat64,
    interflow_m: TwoDArrayFloat32,
    groundwater_recharge_m: ArrayFloat32,
    bare_soil_evaporation_m: ArrayFloat32,
    transpiration_m: ArrayFloat32,
    interception_evaporation_m: ArrayFloat32,
    dew_m: ArrayFloat32,
    open_water_evaporation_m: ArrayFloat32,
    pr_kg_per_m2_per_s: TwoDArrayFloat32,
    actual_irrigation_consumption_m: ArrayFloat32,
    capillar_rise_m: ArrayFloat32,
    sublimation_or_deposition_m: ArrayFloat32,
    snow_water_equivalent_prev: TwoDArrayFloat64,
    liquid_water_in_snow_prev: TwoDArrayFloat64,
    interception_storage_prev: ArrayFloat32,
    topwater_m_prev: ArrayFloat32,
    water_content_m_prev: TwoDArrayFloat32,
    err_flags: ArrayInt8,
    HRU_to_grid: ArrayInt32,
    tolerance: float = 1e-5,
) -> None:
    """Validate land surface water state/flux variables and cellwise water balance closure.

    Notes:
        Performs non-negativity and finite (non-NaN) checks across all hydrological state and flux
        variables and accumulates hourly and multi-layer water balances on-the-fly in registers.
        Avoids allocating intermediate boolean arrays and multi-megabyte reduction buffers in DRAM.

    Args:
        topwater_m: Surface ponding storage per HRU (m).
        runoff_m: Hourly surface runoff flux per HRU (m/hour).
        water_content_m: Updated soil layer water content (m).
        interception_storage_m: Canopy interception storage (m).
        snow_water_equivalent_m: Snowpack SWE per layer (m).
        liquid_water_in_snow_m: Liquid water in snowpack per layer (m).
        interflow_m: Hourly lateral interflow flux per HRU (m/hour).
        groundwater_recharge_m: Daily vertical groundwater recharge flux (m).
        bare_soil_evaporation_m: Daily bare soil evaporation flux (m).
        transpiration_m: Daily transpiration flux (m).
        interception_evaporation_m: Daily canopy interception evaporation flux, net of
            dew (m).
        dew_m: Daily dew (m).
        open_water_evaporation_m: Daily open water evaporation flux (m).
        pr_kg_per_m2_per_s: Hourly precipitation rate (kg/m2/s).
        actual_irrigation_consumption_m: Daily actual irrigation consumption (m).
        capillar_rise_m: Daily capillary rise from groundwater (m).
        sublimation_or_deposition_m: Daily net sublimation or deposition flux (m).
        snow_water_equivalent_prev: Pre-step snowpack SWE per layer (m).
        liquid_water_in_snow_prev: Pre-step liquid water in snow per layer (m).
        interception_storage_prev: Pre-step canopy interception storage (m).
        topwater_m_prev: Pre-step surface ponding storage (m).
        water_content_m_prev: Pre-step soil layer water content (m).
        err_flags: Output 1D int8 array of error status codes per HRU. Modified in place.
        HRU_to_grid: Mapping from HRU cell index to compressed grid index (-).
        tolerance: Absolute closure tolerance for water budget (meters).
    """
    num_cells: int = topwater_m.size

    for i in range(num_cells):
        error_code: int = ERR_NONE

        # 1. Topwater
        topwater_cell_m: np.float32 = topwater_m[i]
        if np.isnan(topwater_cell_m):
            error_code = ERR_NAN_TOPWATER
        elif topwater_cell_m < np.float32(0.0):
            error_code = ERR_NEG_TOPWATER

        # 2. Runoff
        runoff_sum_m: np.float64 = np.float64(0.0)
        min_runoff_m: np.float32 = np.float32(0.0)
        for hour in range(24):
            runoff_hour_m: np.float32 = runoff_m[i, hour]
            if np.isnan(runoff_hour_m) and error_code == ERR_NONE:
                error_code = ERR_NAN_RUNOFF
            elif runoff_hour_m < min_runoff_m:
                min_runoff_m = runoff_hour_m
            runoff_sum_m += np.float64(runoff_hour_m)
        if min_runoff_m < np.float32(0.0) and error_code == ERR_NONE:
            error_code = ERR_NEG_RUNOFF

        # 3. Water content
        water_content_post_sum_m: np.float64 = np.float64(0.0)
        water_content_pre_sum_m: np.float64 = np.float64(0.0)
        min_water_content_m: np.float32 = np.float32(0.0)
        for layer in range(N_SOIL_LAYERS):
            water_content_post_m: np.float32 = water_content_m[i, layer]
            water_content_pre_m: np.float32 = water_content_m_prev[i, layer]
            if np.isnan(water_content_post_m) and error_code == ERR_NONE:
                error_code = ERR_NAN_WATER_CONTENT
            elif water_content_post_m < min_water_content_m:
                min_water_content_m = water_content_post_m
            water_content_post_sum_m += np.float64(water_content_post_m)
            water_content_pre_sum_m += np.float64(water_content_pre_m)
        if min_water_content_m < np.float32(0.0) and error_code == ERR_NONE:
            error_code = ERR_NEG_WATER_CONTENT

        # 4. Interception storage
        interception_storage_cell_m: np.float32 = interception_storage_m[i]
        if np.isnan(interception_storage_cell_m) and error_code == ERR_NONE:
            error_code = ERR_NAN_INTERCEPTION
        elif interception_storage_cell_m < np.float32(0.0) and error_code == ERR_NONE:
            error_code = ERR_NEG_INTERCEPTION

        # 5. Snow water equivalent and liquid water in snow
        snow_post_top_m: np.float64 = snow_water_equivalent_m[i, 0]
        snow_post_bottom_m: np.float64 = snow_water_equivalent_m[i, 1]
        if (
            np.isnan(snow_post_top_m) or np.isnan(snow_post_bottom_m)
        ) and error_code == ERR_NONE:
            error_code = ERR_NAN_SWE
        elif (
            snow_post_top_m < 0.0 or snow_post_bottom_m < 0.0
        ) and error_code == ERR_NONE:
            error_code = ERR_NEG_SWE

        liquid_snow_post_top_m: np.float64 = liquid_water_in_snow_m[i, 0]
        liquid_snow_post_bottom_m: np.float64 = liquid_water_in_snow_m[i, 1]
        if (
            np.isnan(liquid_snow_post_top_m) or np.isnan(liquid_snow_post_bottom_m)
        ) and error_code == ERR_NONE:
            error_code = ERR_NAN_LIQUID_SNOW
        elif (
            liquid_snow_post_top_m < 0.0 or liquid_snow_post_bottom_m < 0.0
        ) and error_code == ERR_NONE:
            error_code = ERR_NEG_LIQUID_SNOW

        # 6. Interflow and groundwater recharge
        interflow_sum_m: np.float64 = np.float64(0.0)
        min_interflow_m: np.float32 = np.float32(0.0)
        for hour in range(24):
            interflow_hour_m: np.float32 = interflow_m[i, hour]
            if np.isnan(interflow_hour_m) and error_code == ERR_NONE:
                error_code = ERR_NAN_INTERFLOW
            elif interflow_hour_m < min_interflow_m:
                min_interflow_m = interflow_hour_m
            interflow_sum_m += np.float64(interflow_hour_m)
        if min_interflow_m < np.float32(0.0) and error_code == ERR_NONE:
            error_code = ERR_NEG_INTERFLOW

        groundwater_recharge_cell_m: np.float32 = groundwater_recharge_m[i]
        if np.isnan(groundwater_recharge_cell_m) and error_code == ERR_NONE:
            error_code = ERR_NAN_GW_RECHARGE
        elif groundwater_recharge_cell_m < np.float32(0.0) and error_code == ERR_NONE:
            error_code = ERR_NEG_GW_RECHARGE

        # 7. Evaporation and transpiration
        bare_soil_evaporation_cell_m: np.float32 = bare_soil_evaporation_m[i]
        transpiration_cell_m: np.float32 = transpiration_m[i]
        interception_evaporation_cell_m: np.float32 = interception_evaporation_m[i]
        open_water_evaporation_cell_m: np.float32 = open_water_evaporation_m[i]
        if (
            np.isnan(bare_soil_evaporation_cell_m)
            or np.isnan(transpiration_cell_m)
            or np.isnan(interception_evaporation_cell_m)
            or np.isnan(open_water_evaporation_cell_m)
        ) and error_code == ERR_NONE:
            error_code = ERR_NAN_EVAP_TRANS
        elif bare_soil_evaporation_cell_m < np.float32(0.0) and error_code == ERR_NONE:
            error_code = ERR_NEG_BARE_SOIL
        elif transpiration_cell_m < np.float32(0.0) and error_code == ERR_NONE:
            error_code = ERR_NEG_TRANSPIRATION
        elif (
            interception_evaporation_cell_m < -dew_m[i] - np.float32(1e-7)
            and error_code == ERR_NONE
        ):
            # net of dew, interception evaporation can be negative, but never below -dew
            error_code = ERR_NEG_INTERCEPTION_EVAP
        elif open_water_evaporation_cell_m < np.float32(0.0) and error_code == ERR_NONE:
            error_code = ERR_NEG_OPEN_WATER_EVAP

        # 8. Water balance closure
        precipitation_sum_kg_per_m2_per_s: np.float64 = np.float64(0.0)
        grid_i: int = HRU_to_grid[i]
        for hour in range(24):
            precipitation_sum_kg_per_m2_per_s += np.float64(
                pr_kg_per_m2_per_s[grid_i, hour]
            )
        precipitation_m: np.float64 = precipitation_sum_kg_per_m2_per_s * 3.6

        influx_m: np.float64 = (
            precipitation_m
            + np.float64(actual_irrigation_consumption_m[i])
            + np.float64(capillar_rise_m[i])
        )
        outflux_m: np.float64 = (
            np.float64(-sublimation_or_deposition_m[i])
            + np.float64(interception_evaporation_cell_m)
            + np.float64(open_water_evaporation_cell_m)
            + runoff_sum_m
            + interflow_sum_m
            + np.float64(groundwater_recharge_cell_m)
            + np.float64(bare_soil_evaporation_cell_m)
            + np.float64(transpiration_cell_m)
        )
        snow_pre_m: np.float64 = np.float64(
            snow_water_equivalent_prev[i, 0]
        ) + np.float64(snow_water_equivalent_prev[i, 1])
        liquid_snow_pre_m: np.float64 = np.float64(
            liquid_water_in_snow_prev[i, 0]
        ) + np.float64(liquid_water_in_snow_prev[i, 1])
        snow_post_m: np.float64 = snow_post_top_m + snow_post_bottom_m
        liquid_snow_post_m: np.float64 = (
            liquid_snow_post_top_m + liquid_snow_post_bottom_m
        )

        prestorage_m: np.float64 = (
            snow_pre_m
            + liquid_snow_pre_m
            + np.float64(interception_storage_prev[i])
            + np.float64(topwater_m_prev[i])
            + water_content_pre_sum_m
        )
        poststorage_m: np.float64 = (
            snow_post_m
            + liquid_snow_post_m
            + np.float64(interception_storage_cell_m)
            + np.float64(topwater_cell_m)
            + water_content_post_sum_m
        )

        balance_m: np.float64 = influx_m - outflux_m + prestorage_m - poststorage_m
        if abs(balance_m) > tolerance and error_code == ERR_NONE:
            error_code = ERR_WATER_IMBALANCE

        err_flags[i] = error_code


@njit(parallel=True, fastmath=False)
def check_soil_enthalpy_balance_kernel(
    soil_enthalpy_prev: TwoDArrayFloat32,
    soil_enthalpy_post: TwoDArrayFloat32,
    soil_boundary_flux: ArrayFloat32,
    rain_advection_flux: ArrayFloat32,
    evaporative_cooling_loss: ArrayFloat32,
    interflow_loss: ArrayFloat32,
    groundwater_recharge_loss: ArrayFloat32,
    transpiration_loss: ArrayFloat32,
    imbalances: ArrayFloat32,
) -> None:
    """Check closure of the daily soil enthalpy budget in parallel.

    Notes:
        Calculates column-integrated soil enthalpy pre/post storage sums and evaluates
        flux balance in parallel without allocating temporary reduction arrays.

    Args:
        soil_enthalpy_prev: Pre-step soil enthalpy per layer (J/m2).
        soil_enthalpy_post: Post-step soil enthalpy per layer (J/m2).
        soil_boundary_flux: Net boundary enthalpy flux (J/m2).
        rain_advection_flux: Rain advection enthalpy flux (J/m2).
        evaporative_cooling_loss: Evaporative cooling enthalpy loss (J/m2).
        interflow_loss: Lateral interflow enthalpy loss (J/m2).
        groundwater_recharge_loss: Groundwater recharge enthalpy loss (J/m2).
        transpiration_loss: Transpiration sensible enthalpy loss (J/m2).
        imbalances: Output array to store per-HRU enthalpy imbalances (J/m2). Modified in place.
    """
    num_cells: int = soil_boundary_flux.size
    for i in prange(num_cells):
        enthalpy_pre_J_per_m2: np.float32 = np.float32(0.0)
        enthalpy_post_J_per_m2: np.float32 = np.float32(0.0)
        for layer in range(N_SOIL_LAYERS):
            enthalpy_pre_J_per_m2 += soil_enthalpy_prev[i, layer]
            enthalpy_post_J_per_m2 += soil_enthalpy_post[i, layer]

        influx_J_per_m2: np.float32 = soil_boundary_flux[i] + rain_advection_flux[i]
        outflux_J_per_m2: np.float32 = (
            evaporative_cooling_loss[i]
            + interflow_loss[i]
            + groundwater_recharge_loss[i]
            + transpiration_loss[i]
        )
        balance_J_per_m2: np.float32 = (
            influx_J_per_m2
            - outflux_J_per_m2
            + enthalpy_pre_J_per_m2
            - enthalpy_post_J_per_m2
        )
        imbalances[i] = balance_J_per_m2
