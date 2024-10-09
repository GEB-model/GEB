# -*- coding: utf-8 -*-
import math
from datetime import datetime
import json
import copy
import calendar
from typing import Any, Dict, Tuple, Union

from scipy.stats import genextreme
from scipy.optimize import curve_fit

# from gplearn.genetic import SymbolicRegressor
from geb.workflows import TimingModule

import numpy as np
from numba import njit

from ..workflows import balance_check

from honeybees.library.raster import pixels_to_coords, sample_from_map
from honeybees.library.neighbors import find_neighbors

from ..data import (
    load_regional_crop_data_from_dict,
    load_crop_data,
    load_economic_data,
)
from .decision_module import DecisionModule
from .general import AgentArray, AgentBaseClass
from ..hydrology.landcover import GRASSLAND_LIKE, NON_PADDY_IRRIGATED, PADDY_IRRIGATED
from ..HRUs import load_grid


def cumulative_mean(mean, counter, update, mask=None):
    """Calculates the cumulative mean of a series of numbers. This function
    operates in place.

    Args:
        mean: The cumulative mean.
        counter: The number of elements that have been added to the mean.
        update: The new elements that needs to be added to the mean.

    """
    if mask is not None:
        mean[mask] = (mean[mask] * counter[mask] + update[mask]) / (counter[mask] + 1)
        counter[mask] += 1
    else:
        mean[:] = (mean * counter + update) / (counter + 1)
        counter += 1


def shift_and_update(array, update):
    """Shifts the array and updates the first element with the update value.

    Args:
        array: The array that needs to be shifted.
        update: The value that needs to be added to the first element of the array.
    """
    array[:, 1:] = array[:, :-1]
    array[:, 0] = update


def shift_and_reset_matrix(matrix: np.ndarray) -> None:
    """
    Shifts columns to the right in the matrix and sets the first column to zero.
    """
    matrix[:, 1:] = matrix[:, 0:-1]  # Shift columns to the right
    matrix[:, 0] = 0  # Reset the first column to 0


@njit(cache=True, inline="always")
def get_farmer_HRUs(
    field_indices: np.ndarray, field_indices_by_farmer: np.ndarray, farmer_index: int
) -> np.ndarray:
    """Gets indices of field for given farmer.

    Args:
        field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
        field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.

    Returns:
        field_indices_for_farmer: the indices of the fields for the given farmer.
    """
    return field_indices[
        field_indices_by_farmer[farmer_index, 0] : field_indices_by_farmer[
            farmer_index, 1
        ]
    ]


@njit(cache=True)
def farmer_command_area(
    n, field_indices, field_indices_by_farmer, reservoir_command_areas
):
    output = np.full(n, -1, dtype=np.int32)
    for farmer_i in range(n):
        farmer_fields = get_farmer_HRUs(
            field_indices, field_indices_by_farmer, farmer_i
        )
        for field in farmer_fields:
            command_area = reservoir_command_areas[field]
            if command_area != -1:
                output[farmer_i] = command_area
                break
    return output


@njit(cache=True)
def get_farmer_groundwater_depth(
    n, groundwater_depth, HRU_to_grid, field_indices, field_indices_by_farmer, cell_area
):
    groundwater_depth_by_farmer = np.full(n, np.nan, dtype=np.float32)
    for farmer_i in range(n):
        farmer_fields = get_farmer_HRUs(
            field_indices=field_indices,
            field_indices_by_farmer=field_indices_by_farmer,
            farmer_index=farmer_i,
        )
        total_cell_area = 0
        total_groundwater_depth_times_area = 0
        for field in farmer_fields:
            grid_cell = HRU_to_grid[field]
            total_cell_area += cell_area[field]
            total_groundwater_depth_times_area += (
                groundwater_depth[grid_cell] * cell_area[field]
            )
        groundwater_depth_by_farmer[farmer_i] = (
            total_groundwater_depth_times_area / total_cell_area
        )
    return groundwater_depth_by_farmer


@njit(cache=True)
def get_deficit_between_dates(cumulative_water_deficit_m3, farmer, start, end):
    return (
        cumulative_water_deficit_m3[farmer, end]
        - cumulative_water_deficit_m3[
            farmer, start
        ]  # current day of year is effectively starting "tommorrow" due to Python's 0-indexing
    )


@njit(cache=True)
def get_future_deficit(
    farmer: int,
    day_index: int,
    cumulative_water_deficit_m3: np.ndarray,
    crop_calendar: np.ndarray,
    crop_rotation_year_index: np.ndarray,
    potential_irrigation_consumption_farmer_m3: float,
):
    future_water_deficit = potential_irrigation_consumption_farmer_m3
    if day_index >= 365:
        return future_water_deficit
    for crop in crop_calendar[farmer]:
        crop_type = crop[0]
        crop_year_index = crop[3]
        if crop_type != -1 and crop_year_index == crop_rotation_year_index[farmer]:
            start_day = crop[1]
            growth_length = crop[2]
            end_day = start_day + growth_length

            if end_day > 365:
                future_water_deficit += get_deficit_between_dates(
                    cumulative_water_deficit_m3,
                    farmer,
                    max(start_day, day_index + 1),
                    365,
                )
                if growth_length < 366 and end_day - 366 > day_index:
                    future_water_deficit += get_deficit_between_dates(
                        cumulative_water_deficit_m3,
                        farmer,
                        day_index + 1,
                        end_day % 366,
                    )

            elif day_index < end_day:
                future_water_deficit += get_deficit_between_dates(
                    cumulative_water_deficit_m3,
                    farmer,
                    max(start_day, day_index + 1),
                    end_day,
                )

    assert future_water_deficit >= 0
    return future_water_deficit


@njit(cache=True)
def adjust_irrigation_to_limit(
    farmer: int,
    day_index: int,
    remaining_irrigation_limit_m3: np.ndarray,
    cumulative_water_deficit_m3: np.ndarray,
    crop_calendar: np.ndarray,
    crop_rotation_year_index: np.ndarray,
    irrigation_efficiency_farmer: float,
    potential_irrigation_consumption_m,
    potential_irrigation_consumption_farmer_m3,
):
    if remaining_irrigation_limit_m3[farmer] < 0:
        potential_irrigation_consumption_m.fill(0)
        return potential_irrigation_consumption_m
    # calculate future water deficit, but also include today's irrigation consumption
    future_water_deficit = get_future_deficit(
        farmer=farmer,
        day_index=day_index,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=crop_calendar,
        crop_rotation_year_index=crop_rotation_year_index,
        potential_irrigation_consumption_farmer_m3=potential_irrigation_consumption_farmer_m3,
    )

    assert future_water_deficit > 0

    # first find the total irrigation demand for the farmer in m3
    irrigation_water_withdrawal_farmer_m3 = (
        potential_irrigation_consumption_farmer_m3
        * remaining_irrigation_limit_m3[farmer]
        / future_water_deficit
    )
    assert not np.isnan(irrigation_water_withdrawal_farmer_m3)

    irrigation_water_consumption_farmer_m3 = (
        irrigation_water_withdrawal_farmer_m3 * irrigation_efficiency_farmer
    )
    irrigation_water_consumption_farmer_m3 = min(
        irrigation_water_consumption_farmer_m3,
        potential_irrigation_consumption_farmer_m3,
    )

    # if the irrigation demand is higher than the limit, reduce the irrigation demand by the calculated reduction factor
    reduction_factor = (
        irrigation_water_consumption_farmer_m3
        / potential_irrigation_consumption_farmer_m3
    )
    assert not np.isnan(reduction_factor)

    potential_irrigation_consumption_m = (
        potential_irrigation_consumption_m * reduction_factor
    )
    return potential_irrigation_consumption_m


@njit(cache=True)
def withdraw_channel(
    available_channel_storage_m3: np.ndarray,
    grid_cell: int,
    cell_area: np.ndarray,
    field: int,
    farmer: int,
    irrigation_water_demand_field: float,
    water_withdrawal_m: np.ndarray,
    remaining_irrigation_limit_m3: np.ndarray,
    channel_abstraction_m3_by_farmer: np.ndarray,
    minimum_channel_storage_m3: float = 100,
):
    # channel abstraction
    channel_abstraction_cell_m3 = min(
        max(available_channel_storage_m3[grid_cell] - minimum_channel_storage_m3, 0),
        irrigation_water_demand_field * cell_area[field],
    )
    assert channel_abstraction_cell_m3 >= 0
    channel_abstraction_cell_m = channel_abstraction_cell_m3 / cell_area[field]
    available_channel_storage_m3[grid_cell] -= channel_abstraction_cell_m3

    water_withdrawal_m[field] += channel_abstraction_cell_m

    if not np.isnan(remaining_irrigation_limit_m3[farmer]):
        remaining_irrigation_limit_m3[farmer] -= channel_abstraction_cell_m3

    channel_abstraction_m3_by_farmer[farmer] += channel_abstraction_cell_m3

    irrigation_water_demand_field -= channel_abstraction_cell_m

    return irrigation_water_demand_field


@njit(cache=True)
def withdraw_reservoir(
    command_area: int,
    field: int,
    farmer: int,
    available_reservoir_storage_m3: np.ndarray,
    irrigation_water_demand_field: float,
    water_withdrawal_m: np.ndarray,
    remaining_irrigation_limit_m3: np.ndarray,
    reservoir_abstraction_m3_by_farmer: np.ndarray,
    cell_area: np.ndarray,
):
    water_demand_cell_M3 = irrigation_water_demand_field * cell_area[field]
    reservoir_abstraction_m_cell_m3 = min(
        available_reservoir_storage_m3[command_area],
        water_demand_cell_M3,
    )
    available_reservoir_storage_m3[command_area] -= reservoir_abstraction_m_cell_m3

    reservoir_abstraction_m_cell = reservoir_abstraction_m_cell_m3 / cell_area[field]
    water_withdrawal_m[field] += reservoir_abstraction_m_cell

    if not np.isnan(remaining_irrigation_limit_m3[farmer]):
        remaining_irrigation_limit_m3[farmer] -= reservoir_abstraction_m_cell_m3

    reservoir_abstraction_m3_by_farmer[farmer] += reservoir_abstraction_m_cell_m3

    irrigation_water_demand_field -= reservoir_abstraction_m_cell
    return irrigation_water_demand_field


@njit(cache=True)
def withdraw_groundwater(
    farmer: int,
    grid_cell: int,
    field: int,
    available_groundwater_m3: np.ndarray,
    cell_area: np.ndarray,
    groundwater_depth: np.ndarray,
    well_depth: np.ndarray,
    irrigation_water_demand_field: float,
    water_withdrawal_m: np.ndarray,
    remaining_irrigation_limit_m3: np.ndarray,
    groundwater_abstraction_m3_by_farmer: np.ndarray,
):
    # groundwater irrigation
    if groundwater_depth[grid_cell] < well_depth[farmer]:
        groundwater_abstraction_cell_m3 = min(
            available_groundwater_m3[grid_cell],
            irrigation_water_demand_field * cell_area[field],
        )
        assert groundwater_abstraction_cell_m3 >= 0

        available_groundwater_m3[grid_cell] -= groundwater_abstraction_cell_m3

        groundwater_abstraction_cell_m = (
            groundwater_abstraction_cell_m3 / cell_area[field]
        )

        water_withdrawal_m[field] += groundwater_abstraction_cell_m

        if not np.isnan(remaining_irrigation_limit_m3[farmer]):
            remaining_irrigation_limit_m3[farmer] -= groundwater_abstraction_cell_m3

        groundwater_abstraction_m3_by_farmer[farmer] += groundwater_abstraction_cell_m3

        irrigation_water_demand_field -= groundwater_abstraction_cell_m
    return irrigation_water_demand_field


@njit(cache=True)
def abstract_water(
    day_index: int,
    n: int,
    activation_order: np.ndarray,
    field_indices_by_farmer: np.ndarray,
    field_indices: np.ndarray,
    irrigation_efficiency: np.ndarray,
    surface_irrigated: np.ndarray,
    well_irrigated: np.ndarray,
    cell_area: np.ndarray,
    HRU_to_grid: np.ndarray,
    crop_map: np.ndarray,
    field_is_paddy_irrigated: np.ndarray,
    paddy_level: np.ndarray,
    readily_available_water: np.ndarray,
    critical_water_level: np.ndarray,
    max_water_content: np.ndarray,
    potential_infiltration_capacity: np.ndarray,
    available_channel_storage_m3: np.ndarray,
    available_groundwater_m3: np.ndarray,
    groundwater_depth: np.ndarray,
    available_reservoir_storage_m3: np.ndarray,
    farmer_command_area: np.ndarray,
    return_fraction: float,
    well_depth: float,
    remaining_irrigation_limit_m3: np.ndarray,
    cumulative_water_deficit_m3: np.ndarray,
    crop_calendar: np.ndarray,
    current_crop_calendar_rotation_year_index: np.ndarray,
    max_paddy_water_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function is used to regulate the irrigation behavior of farmers. The farmers are "activated" by the given `activation_order` and each farmer can irrigate from the various water sources, given water is available and the farmers has the means to abstract water. The abstraction order is channel irrigation, reservoir irrigation, groundwater irrigation.

    Args:
        activation_order: Order in which the agents are activated. Agents that are activated first get a first go at extracting water, leaving less water for other farmers.
        field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
        field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.
        irrigation_efficiency: Boolean array that specifies whether the specific farmer is efficient with water use.
        irrigated: Array that specifies whether a farm is irrigated.
        well_irrigated: Array that specifies whether a farm is groundwater irrigated.
        cell_area: The area of each subcell in m2.
        HRU_to_grid: Array to map the index of each subcell to the corresponding cell.
        crop_map: Map of the currently growing crops.
        totalPotIrrConsumption: Potential irrigation consumption.
        available_channel_storage_m3: Water available for irrigation from channels.
        groundwater_head: Groundwater head.
        available_groundwater_m3: Water available for irrigation from groundwater.
        available_reservoir_storage_m3: Water available for irrigation from reservoirs.
        command_areas: Command areas associated with reservoirs (i.e., which areas can access water from which reservoir.)

    Returns:
        channel_abstraction_m3_by_farmer: Channel abstraction by farmer in m3.
        reservoir_abstraction_m3_by_farmer: Revervoir abstraction by farmer in m3.
        groundwater_abstraction_m3_by_farmer: Groundwater abstraction by farmer in m3.
        water_withdrawal_m: Water withdrawal in meters.
        water_consumption_m: Water consumption in meters.
        returnFlowIrr_m: Return flow in meters.
        addtoevapotrans_m: Evaporated irrigation water in meters.
    """
    assert n == activation_order.size

    n_hydrological_response_units = cell_area.size
    water_withdrawal_m = np.zeros(n_hydrological_response_units, dtype=np.float32)
    water_consumption_m = np.zeros(n_hydrological_response_units, dtype=np.float32)

    returnFlowIrr_m = np.zeros(n_hydrological_response_units, dtype=np.float32)
    addtoevapotrans_m = np.zeros(n_hydrological_response_units, dtype=np.float32)

    channel_abstraction_m3_by_farmer = np.zeros(activation_order.size, dtype=np.float32)
    reservoir_abstraction_m3_by_farmer = np.zeros(
        activation_order.size, dtype=np.float32
    )
    groundwater_abstraction_m3_by_farmer = np.zeros(
        activation_order.size, dtype=np.float32
    )

    for activated_farmer_index in range(activation_order.size):
        farmer = activation_order[activated_farmer_index]
        farmer_fields = get_farmer_HRUs(field_indices, field_indices_by_farmer, farmer)

        # if no crops are grown on the fields, skip the farmer
        farmer_crops = crop_map[farmer_fields]
        if farmer_crops[0] == -1:
            assert (
                farmer_crops == -1
            ).all()  # currently only support situation where all fields are growing crops, or all fields are fallow
            continue

        # Determine whether farmer would have access to irrigation water this timestep. Regardless of whether the water is actually used. This is used for making investment decisions.
        for field in farmer_fields:
            grid_cell = HRU_to_grid[field]

            if well_irrigated[farmer]:
                if groundwater_depth[grid_cell] < well_depth[farmer]:
                    farmer_has_access_to_irrigation_water = True
                    break
            elif surface_irrigated[farmer]:
                if available_channel_storage_m3[grid_cell] > 0:
                    farmer_has_access_to_irrigation_water = True
                    break
                command_area = farmer_command_area[farmer]
                # -1 means no command area
                if (
                    command_area != -1
                    and available_reservoir_storage_m3[command_area] > 0
                ):
                    farmer_has_access_to_irrigation_water = True
                    break

        # Actual irrigation from surface, reservoir and groundwater
        # If farmer doesn't have access to irrigation water, skip the irrigation abstraction
        if farmer_has_access_to_irrigation_water:
            irrigation_efficiency_farmer = irrigation_efficiency[farmer]

            # Calculate the potential irrigation consumption for the farmer
            if field_is_paddy_irrigated[farmer_fields][0]:
                # make sure all fields are paddy irrigateds
                assert (field_is_paddy_irrigated[farmer_fields]).all()
                # always irrigate to 0.05 m for paddy fields
                potential_irrigation_consumption_m = (
                    max_paddy_water_level - paddy_level[farmer_fields]
                )
                # make sure the potential irrigation consumption is not negative
                potential_irrigation_consumption_m[
                    potential_irrigation_consumption_m < 0
                ] = 0
            else:
                # assert none of the fiels are paddy irrigated
                assert (~field_is_paddy_irrigated[farmer_fields]).all()
                # if soil moisture content falls below critical level, irrigate to field capacity
                potential_irrigation_consumption_m = np.where(
                    readily_available_water[farmer_fields]
                    < critical_water_level[
                        farmer_fields
                    ],  # if there is not enough water
                    critical_water_level[farmer_fields]
                    - readily_available_water[
                        farmer_fields
                    ],  # irrigate to field capacity (if possible)
                    0.0,
                )
                # limit the irrigation consumption to the potential infiltration capacity
                potential_irrigation_consumption_m = np.minimum(
                    potential_irrigation_consumption_m,
                    potential_infiltration_capacity[farmer_fields],
                )

            assert (potential_irrigation_consumption_m >= 0).all()
            potential_irrigation_consumption_farmer_m3 = (
                potential_irrigation_consumption_m * cell_area[farmer_fields]
            ).sum()

            # If the potential irrigation consumption is larger than 0, the farmer needs to abstract water
            if potential_irrigation_consumption_farmer_m3 > 0.0:
                # if irrigation limit is active, reduce the irrigation demand
                if not np.isnan(remaining_irrigation_limit_m3[farmer]):
                    adjust_irrigation_to_limit(
                        farmer=farmer,
                        day_index=day_index,
                        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
                        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
                        crop_calendar=crop_calendar,
                        crop_rotation_year_index=current_crop_calendar_rotation_year_index,
                        irrigation_efficiency_farmer=irrigation_efficiency_farmer,
                        potential_irrigation_consumption_m=potential_irrigation_consumption_m,
                        potential_irrigation_consumption_farmer_m3=potential_irrigation_consumption_farmer_m3,
                    )

                # loop through all farmers fields and apply irrigation
                for field_idx, field in enumerate(farmer_fields):
                    grid_cell = HRU_to_grid[field]
                    if crop_map[field] != -1:
                        irrigation_water_demand_field = (
                            potential_irrigation_consumption_m[field_idx]
                            / irrigation_efficiency_farmer
                        )

                        if surface_irrigated[farmer]:
                            irrigation_water_demand_field = withdraw_channel(
                                available_channel_storage_m3=available_channel_storage_m3,
                                grid_cell=grid_cell,
                                cell_area=cell_area,
                                field=field,
                                farmer=farmer,
                                water_withdrawal_m=water_withdrawal_m,
                                irrigation_water_demand_field=irrigation_water_demand_field,
                                remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
                                channel_abstraction_m3_by_farmer=channel_abstraction_m3_by_farmer,
                                minimum_channel_storage_m3=100.0,
                            )
                            assert water_withdrawal_m[field] >= 0

                            # command areas
                            command_area = farmer_command_area[farmer]
                            if command_area != -1:  # -1 means no command area
                                irrigation_water_demand_field = withdraw_reservoir(
                                    command_area=command_area,
                                    field=field,
                                    farmer=farmer,
                                    available_reservoir_storage_m3=available_reservoir_storage_m3,
                                    irrigation_water_demand_field=irrigation_water_demand_field,
                                    water_withdrawal_m=water_withdrawal_m,
                                    remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
                                    reservoir_abstraction_m3_by_farmer=reservoir_abstraction_m3_by_farmer,
                                    cell_area=cell_area,
                                )
                                assert water_withdrawal_m[field] >= 0

                        if well_irrigated[farmer]:
                            irrigation_water_demand_field = withdraw_groundwater(
                                farmer=farmer,
                                field=field,
                                grid_cell=grid_cell,
                                available_groundwater_m3=available_groundwater_m3,
                                cell_area=cell_area,
                                groundwater_depth=groundwater_depth,
                                well_depth=well_depth,
                                irrigation_water_demand_field=irrigation_water_demand_field,
                                water_withdrawal_m=water_withdrawal_m,
                                remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
                                groundwater_abstraction_m3_by_farmer=groundwater_abstraction_m3_by_farmer,
                            )
                            assert water_withdrawal_m[field] >= 0

                        assert (
                            irrigation_water_demand_field >= -1e15
                        )  # Make sure irrigation water demand is zero, or positive. Allow very small error.

                    assert water_consumption_m[field] >= 0
                    assert water_withdrawal_m[field] >= 0
                    assert 1 >= irrigation_efficiency_farmer >= 0
                    assert 1 >= return_fraction >= 0

                    water_consumption_m[field] = (
                        water_withdrawal_m[field] * irrigation_efficiency_farmer
                    )
                    irrigation_loss_m = (
                        water_withdrawal_m[field] - water_consumption_m[field]
                    )
                    assert irrigation_loss_m >= 0
                    returnFlowIrr_m[field] = irrigation_loss_m * return_fraction
                    addtoevapotrans_m[field] = (
                        irrigation_loss_m - returnFlowIrr_m[field]
                    )

    return (
        channel_abstraction_m3_by_farmer,
        reservoir_abstraction_m3_by_farmer,
        groundwater_abstraction_m3_by_farmer,
        water_withdrawal_m,
        water_consumption_m,
        returnFlowIrr_m,
        addtoevapotrans_m,
    )


def advance_crop_rotation_year(
    current_crop_calendar_rotation_year_index: np.ndarray,
    crop_calendar_rotation_years: np.ndarray,
):
    """Update the crop rotation year for each farmer. This function is used to update the crop rotation year for each farmer at the end of the year.

    Args:
        current_crop_calendar_rotation_year_index: The current crop rotation year for each farmer.
        crop_calendar_rotation_years: The number of years in the crop rotation cycle for each farmer.
    """
    current_crop_calendar_rotation_year_index[:] = (
        current_crop_calendar_rotation_year_index + 1
    ) % crop_calendar_rotation_years


@njit(cache=True)
def plant(
    n: int,
    day_index: int,
    crop_calendar: np.ndarray,
    current_crop_calendar_rotation_year_index: np.ndarray,
    crop_map: np.ndarray,
    crop_harvest_age_days: np.ndarray,
    cultivation_cost: Union[np.ndarray, int, float],
    region_ids_per_farmer: np.ndarray,
    field_indices_by_farmer: np.ndarray,
    field_indices: np.ndarray,
    field_size_per_farmer: np.ndarray,
    all_loans_annual_cost: np.ndarray,
    loan_tracker: np.ndarray,
    interest_rate: np.ndarray,
    farmers_going_out_of_business: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Determines when and what crop should be planted, by comparing the current day to the next plant day. Also sets the haverst age of the plant.

    Args:
        n: Number of farmers.
        start_day_per_month: Starting day of each month of year.
        current_day: Current day.
        crop: Crops grown by each farmer.
        field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
        field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.
        field_size_per_farmer: Field size per farmer in m2

    Returns:
        plant: Subarray map of what crops are planted this day.
    """
    assert (
        farmers_going_out_of_business is False
    ), "Farmers going out of business not implemented."

    plant = np.full_like(crop_map, -1, dtype=np.int32)
    sell_land = np.zeros(n, dtype=np.bool_)

    planting_farmers_per_season = (crop_calendar[:, :, 1] == day_index) & (
        crop_calendar[:, :, 3]
        == current_crop_calendar_rotation_year_index[:, np.newaxis]
    )
    planting_farmers = planting_farmers_per_season.sum(axis=1)

    assert planting_farmers.max() <= 1, "Multiple crops planted on the same day"

    planting_farmers_idx = np.where(planting_farmers == 1)[0]
    if not planting_farmers_idx.size == 0:
        crop_rotation = np.argmax(
            planting_farmers_per_season[planting_farmers_idx], axis=1
        )

        assert planting_farmers_idx.size == crop_rotation.size

        for i in range(planting_farmers_idx.size):
            farmer_idx = planting_farmers_idx[i]
            farmer_crop_rotation = crop_rotation[i]

            farmer_fields = get_farmer_HRUs(
                field_indices, field_indices_by_farmer, farmer_idx
            )
            farmer_crop_data = crop_calendar[farmer_idx, farmer_crop_rotation]
            farmer_crop = farmer_crop_data[0]
            field_harvest_age = farmer_crop_data[2]

            assert farmer_crop != -1

            if isinstance(cultivation_cost, (int, float)):
                cultivation_cost_farmer = cultivation_cost
            else:
                farmer_region_id = region_ids_per_farmer[farmer_idx]
                cultivation_cost_farmer = (
                    cultivation_cost[farmer_region_id, farmer_crop]
                    * field_size_per_farmer[farmer_idx]
                )
            assert not np.isnan(cultivation_cost_farmer)

            interest_rate_farmer = interest_rate[farmer_idx]
            loan_duration = 2
            annual_cost_input_loan = cultivation_cost_farmer * (
                interest_rate_farmer
                * (1 + interest_rate_farmer) ** loan_duration
                / ((1 + interest_rate_farmer) ** loan_duration - 1)
            )
            for i in range(4):
                if all_loans_annual_cost[farmer_idx, 0, i] == 0:
                    all_loans_annual_cost[farmer_idx, 0, i] += (
                        annual_cost_input_loan  # Add the amount to the input specific loan
                    )
                    loan_tracker[farmer_idx, 0, i] = loan_duration
                    break  # Exit the loop after adding to the first zero value

            all_loans_annual_cost[farmer_idx, -1, 0] += (
                annual_cost_input_loan  # Add the amount to the total loan amount
            )

            for field in farmer_fields:
                # a crop is still growing here.
                if crop_harvest_age_days[field] != -1:
                    continue
                plant[field] = farmer_crop
                crop_harvest_age_days[field] = field_harvest_age

    farmers_selling_land = np.where(sell_land)[0]
    return plant, farmers_selling_land


class CropFarmers(AgentBaseClass):
    """The agent class for the farmers. Contains all data and behaviourial methods. The __init__ function only gets the model as arguments, the agent parent class and the redundancy. All other variables are loaded at later stages.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
        redundancy: a lot of data is saved in pre-allocated NumPy arrays. While this allows much faster operation, it does mean that the number of agents cannot grow beyond the size of the pre-allocated arrays. This parameter allows you to specify how much redundancy should be used. A lower redundancy means less memory is used, but the model crashes if the redundancy is insufficient.
    """

    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["farmers"]
            if "farmers" in self.model.config["agent_settings"]
            else {}
        )
        self.sample = [2000, 5500, 10000]
        self.var = model.data.HRU
        self.redundancy = reduncancy
        self.decision_module = DecisionModule(self)

        self.HRU_n = len(self.var.land_owners)
        self.crop_data_type, self.crop_data = load_crop_data(self.model.files)
        self.crop_ids = self.crop_data["name"].to_dict()
        # reverse dictionary
        self.crop_names = {
            crop_name: crop_id for crop_id, crop_name in self.crop_ids.items()
        }

        ## Set parameters required for drought event perception, risk perception and SEUT
        self.moving_average_threshold = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["drought_risk_calculations"]["event_perception"]["drought_threshold"]
        self.previous_month = 0

        # Assign risk aversion sigma, time discounting preferences, expenditure_cap
        self.expenditure_cap = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["decisions"]["expenditure_cap"]

        self.inflation_rate = load_economic_data(
            self.model.files["dict"]["economics/inflation_rates"]
        )
        self.lending_rate = load_economic_data(
            self.model.files["dict"]["economics/lending_rates"]
        )
        self.electricity_cost = load_economic_data(
            self.model.files["dict"]["economics/electricity_cost"]
        )

        self.max_paddy_water_level = 0.05

        # New global well variables
        self.pump_hours = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["pump_hours"]
        self.specific_weight_water = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["specific_weight_water"]
        self.max_initial_sat_thickness = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["max_initial_sat_thickness"]
        self.lifespan = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["lifespan"]
        self.pump_efficiency = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["pump_efficiency"]
        self.maintenance_factor = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["maintenance_factor"]

        self.why_10 = load_economic_data(self.model.files["dict"]["economics/why_10"])
        self.why_20 = load_economic_data(self.model.files["dict"]["economics/why_20"])
        self.why_30 = load_economic_data(self.model.files["dict"]["economics/why_30"])

        self.p_droughts = np.array([100, 50, 25, 10, 5, 2, 1])

        # Set water costs
        self.water_costs_m3_channel = 0.20
        self.water_costs_m3_reservoir = 0.20
        self.water_costs_m3_groundwater = 0.20

        self.elevation_subgrid = load_grid(
            self.model.files["MERIT_grid"]["landsurface/topo/subgrid_elevation"],
            return_transform_and_crs=True,
        )

        with open(
            self.model.files["dict"]["agents/farmers/irrigation_sources"], "r"
        ) as f:
            self.irrigation_source_key = json.load(f)

        # load map of all subdistricts
        self.subdistrict_map = load_grid(
            self.model.files["region_subgrid"]["areamaps/region_subgrid"]
        )

        self.crop_prices = load_regional_crop_data_from_dict(
            self.model, "crops/crop_prices"
        )

        self.cultivation_costs = load_regional_crop_data_from_dict(
            self.model, "crops/cultivation_costs"
        )

        # Set the cultivation costs
        cultivation_cost_fraction = self.model.config["agent_settings"]["farmers"][
            "cultivation_cost_fraction"
        ]  # Cultivation costs are set as a fraction of crop prices
        date_index, cultivation_costs_array = self.cultivation_costs
        adjusted_cultivation_costs_array = (
            cultivation_costs_array * cultivation_cost_fraction
        )
        self.cultivation_costs = (date_index, adjusted_cultivation_costs_array)

        # Test with a high variable for now
        self.total_spinup_time = max(
            self.model.config["general"]["start_time"].year
            - self.model.config["general"]["spinup_time"].year,
            30,
        )

        self.yield_ratio_multiplier_value = self.model.config["agent_settings"][
            "farmers"
        ]["expected_utility"]["adaptation_sprinkler"]["yield_multiplier"]

        self.var.actual_evapotranspiration_crop_life = self.var.load_initial(
            "actual_evapotranspiration_crop_life",
            default=self.var.full_compressed(0, dtype=np.float32, gpu=False),
            gpu=False,
        )
        self.var.potential_evapotranspiration_crop_life = self.var.load_initial(
            "potential_evapotranspiration_crop_life",
            default=self.var.full_compressed(0, dtype=np.float32, gpu=False),
            gpu=False,
        )
        self.var.crop_map = self.var.load_initial(
            "crop_map", default=np.full_like(self.var.land_owners, -1), gpu=False
        )
        self.var.crop_age_days_map = self.var.load_initial(
            "crop_age_days_map",
            default=np.full_like(self.var.land_owners, -1),
            gpu=False,
        )
        self.var.crop_harvest_age_days = self.var.load_initial(
            "crop_harvest_age_days",
            default=np.full_like(self.var.land_owners, -1),
            gpu=False,
        )

        super().__init__()

    def initiate(self) -> None:
        """Calls functions to initialize all agent attributes, including their locations. Then, crops are initially planted."""
        # If initial conditions based on spinup period need to be loaded, load them. Otherwise, generate them.

        farms = self.model.data.farms

        # Get number of farmers and maximum number of farmers that could be in the entire model run based on the redundancy.
        self.n = np.unique(farms[farms != -1]).size
        self.max_n = self.get_max_n(self.n)

        # The code below obtains the coordinates of the farmers' locations.
        # First the horizontal and vertical indices of the pixels that are not -1 are obtained. Then, for each farmer the
        # average of the horizontal and vertical indices is calculated. This is done by using the bincount function.
        # Finally, the coordinates are obtained by adding .5 to the pixels and converting them to coordinates using pixel_to_coord.
        vertical_index = (
            np.arange(farms.shape[0])
            .repeat(farms.shape[1])
            .reshape(farms.shape)[farms != -1]
        )
        horizontal_index = np.tile(np.arange(farms.shape[1]), farms.shape[0]).reshape(
            farms.shape
        )[farms != -1]
        pixels = np.zeros((self.n, 2), dtype=np.int32)
        pixels[:, 0] = np.round(
            np.bincount(farms[farms != -1], horizontal_index)
            / np.bincount(farms[farms != -1])
        ).astype(int)
        pixels[:, 1] = np.round(
            np.bincount(farms[farms != -1], vertical_index)
            / np.bincount(farms[farms != -1])
        ).astype(int)

        self.locations = AgentArray(
            pixels_to_coords(pixels + 0.5, self.var.gt), max_n=self.max_n
        )

        self.set_social_network()

        self.risk_aversion = AgentArray(
            n=self.n, max_n=self.max_n, dtype=np.float32, fill_value=np.nan
        )
        self.risk_aversion[:] = np.load(
            self.model.files["binary"]["agents/farmers/risk_aversion"]
        )["data"]

        self.interest_rate = AgentArray(
            n=self.n, max_n=self.max_n, dtype=np.float32, fill_value=np.nan
        )
        self.interest_rate[:] = np.load(
            self.model.files["binary"]["agents/farmers/interest_rate"]
        )["data"]

        self.discount_rate = AgentArray(
            n=self.n, max_n=self.max_n, dtype=np.float32, fill_value=np.nan
        )
        self.discount_rate[:] = np.load(
            self.model.files["binary"]["agents/farmers/discount_rate"]
        )["data"]

        self.intention_factor = AgentArray(
            n=self.n, max_n=self.max_n, dtype=np.float32, fill_value=np.nan
        )
        np.random.seed(42)
        self.intention_factor[:] = np.random.uniform(
            0, 0.6, size=self.intention_factor.shape
        )

        # Load the region_code of each farmer.
        self.region_id = AgentArray(
            input_array=np.load(self.model.files["binary"]["agents/farmers/region_id"])[
                "data"
            ],
            max_n=self.max_n,
        )

        # Find the elevation of each farmer on the map based on the coordinates of the farmer as calculated before.
        self.elevation = AgentArray(
            input_array=sample_from_map(
                self.elevation_subgrid[0],
                self.locations.data,
                self.elevation_subgrid[1].to_gdal(),
            ),
            max_n=self.max_n,
        )

        # Initiate adaptation status. 0 = not adapted, 1 adapted. Column 0 = no cost adaptation, 1 = well, 2 = sprinkler
        self.adapted = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(3,),
            extra_dims_names=("adaptation_type",),
            dtype=np.int32,
            fill_value=0,
        )
        # the time each agent has been paying off their dry flood proofing investment loan. Column 0 = no cost adaptation, 1 = well, 2 = sprinkler.  -1 if they do not have adaptations
        self.time_adapted = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(3,),
            extra_dims_names=("adaptation_type",),
            dtype=np.int32,
            fill_value=-1,
        )
        # Set SEUT of all agents to 0
        self.SEUT_no_adapt = AgentArray(
            n=self.n, max_n=self.max_n, dtype=np.float32, fill_value=0
        )
        # Set EUT of all agents to 0
        self.EUT_no_adapt = AgentArray(
            n=self.n, max_n=self.max_n, dtype=np.float32, fill_value=0
        )
        self.adaptation_mechanism = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(3,),
            extra_dims_names=("adaptation_type",),
            dtype=np.int32,
            fill_value=0,
        )

        self.crop_calendar = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(3, 4),
            extra_dims_names=("rotation", "calendar"),
            dtype=np.int32,
            fill_value=-1,
        )  # first dimension is the farmers, second is the rotation, third is the crop, planting and growing length
        self.crop_calendar[:] = np.load(
            self.model.files["binary"]["agents/farmers/crop_calendar"]
        )["data"]
        # assert self.crop_calendar[:, :, 0].max() < len(self.crop_ids)

        self.crop_calendar_rotation_years = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.int32,
            fill_value=0,
        )
        self.crop_calendar_rotation_years[:] = np.load(
            self.model.files["binary"]["agents/farmers/crop_calendar_rotation_years"]
        )["data"]

        self.current_crop_calendar_rotation_year_index = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.int32,
            fill_value=0,
        )
        # For each farmer set a random crop rotation year. The farmer starts in that year. First set a seed for reproducibility.
        np.random.seed(42)
        self.current_crop_calendar_rotation_year_index[:] = np.random.randint(
            0, self.crop_calendar_rotation_years
        )

        # Set irrigation source
        self.irrigation_source = AgentArray(
            np.load(self.model.files["binary"]["agents/farmers/irrigation_source"])[
                "data"
            ],
            max_n=self.max_n,
        )
        # set the adaptation of wells to 1 if farmers have well
        self.adapted[:, 1][
            np.isin(
                self.irrigation_source,
                np.array(
                    [
                        self.irrigation_source_key["well"],
                    ]
                ),
            )
        ] = 1
        # Set the initial well depth
        self.well_depth = AgentArray(
            n=self.n,
            max_n=self.max_n,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "expected_utility"
            ]["adaptation_well"]["max_initial_sat_thickness"],
            dtype=np.float32,
        )
        # Set how long the agents have adapted somewhere across the lifespan of farmers, would need to be a bit more realistic likely
        rng_wells = np.random.default_rng(17)
        self.time_adapted[self.adapted[:, 1] == 1, 1] = rng_wells.uniform(
            1,
            self.lifespan,
            np.sum(self.adapted[:, 1] == 1),
        )

        # Initiate a number of arrays with Nan, zero or -1 values for variables that will be used during the model run.
        self.channel_abstraction_m3_by_farmer = AgentArray(
            n=self.n, max_n=self.max_n, dtype=np.float32, fill_value=0
        )
        self.reservoir_abstraction_m3_by_farmer = AgentArray(
            n=self.n, max_n=self.max_n, dtype=np.float32, fill_value=0
        )
        self.groundwater_abstraction_m3_by_farmer = AgentArray(
            n=self.n, max_n=self.max_n, dtype=np.float32, fill_value=0
        )

        # 2D-array for storing yearly abstraction by farmer. 0: channel abstraction, 1: reservoir abstraction, 2: groundwater abstraction, 3: total abstraction
        self.yearly_abstraction_m3_by_farmer = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(4, self.total_spinup_time),
            extra_dims_names=("abstraction_type", "year"),
            dtype=np.float32,
            fill_value=0,
        )

        # Yield ratio and crop variables
        # 0 = kharif age, 1 = rabi age, 2 = summer age, 3 = total growth time
        self.total_crop_age = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(2,),
            extra_dims_names=("abstraction_type",),
            dtype=np.float32,
            fill_value=0,
        )

        self.cumulative_SPEI_during_growing_season = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.float32,
            fill_value=0,
        )
        self.cumulative_SPEI_count_during_growing_season = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.int32,
            fill_value=0,
        )

        # set no irrigation limit for farmers by default
        self.irrigation_limit_m3 = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.float32,
            fill_value=np.nan,  # m3
        )
        # set the remaining irrigation limit to the irrigation limit
        self.remaining_irrigation_limit_m3 = AgentArray(
            n=self.n, max_n=self.max_n, fill_value=np.nan, dtype=np.float32
        )

        self.yield_ratios_drought_event = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(self.p_droughts.size,),
            extra_dims_names=("drought_event",),
            dtype=np.float32,
            fill_value=0,
        )

        self.yield_ratio_multiplier = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.float32,
            fill_value=1,
        )

        self.actual_yield_per_farmer = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.float32,
            fill_value=np.nan,
        )

        self.harvested_crop = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.int32,
            fill_value=-1,
        )

        ## Risk perception variables
        self.risk_perception = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.float32,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "expected_utility"
            ]["drought_risk_calculations"]["risk_perception"]["min"],
        )
        self.drought_timer = AgentArray(
            n=self.n, max_n=self.max_n, dtype=np.float32, fill_value=99
        )

        agent_relation_attributes = [
            "yearly_yield_ratio",
            "yearly_SPEI_probability",
            "yearly_profits",
            "yearly_potential_profits",
            "farmer_yield_probability_relation",
        ]

        self.yearly_SPEI_probability = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(self.total_spinup_time,),
            extra_dims_names=("year",),
            dtype=np.float32,
            fill_value=0,
        )
        self.yearly_yield_ratio = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(self.total_spinup_time,),
            extra_dims_names=("year",),
            dtype=np.float32,
            fill_value=0,
        )
        self.yearly_profits = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(self.total_spinup_time,),
            extra_dims_names=("year",),
            dtype=np.float32,
            fill_value=0,
        )
        self.yearly_potential_profits = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(self.total_spinup_time,),
            extra_dims_names=("year",),
            dtype=np.float32,
            fill_value=0,
        )
        self.farmer_yield_probability_relation = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(2,),
            extra_dims_names=("log function parameters",),
            dtype=np.float32,
            fill_value=0,
        )
        for attribute in agent_relation_attributes:
            assert (
                getattr(self, attribute).shape[0] == self.n
            ), "attribute does not exist or is of wrong size"

        self.household_size = AgentArray(
            n=self.n, max_n=self.max_n, dtype=np.int32, fill_value=-1
        )
        self.household_size[:] = np.load(
            self.model.files["binary"]["agents/farmers/household_size"]
        )["data"]

        self.yield_ratios_drought_event = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(self.p_droughts.size,),
            extra_dims_names=("drought_event",),
            dtype=np.float32,
            fill_value=0,
        )

        # Create a random set of irrigating farmers --> chance that it does not line up with farmers that are expected to have this
        # Create a random generator object with a seed
        rng = np.random.default_rng(42)

        self.irrigation_efficiency = AgentArray(
            n=self.n, max_n=self.max_n, dtype=np.float32, fill_value=np.nan
        )
        self.irrigation_efficiency[:] = rng.random(self.n)
        # Set the people who already have more van 90% irrigation efficiency to already adapted for the drip irrgation adaptation
        self.adapted[:, 2][self.irrigation_efficiency >= 0.90] = 1
        self.adaptation_mechanism[self.adapted[:, 2] == 1, 2] = 1
        # set the yield_ratio_multiplier to x of people who have drip irrigation, set to 1 for all others
        self.yield_ratio_multiplier[:] = np.where(
            (self.irrigation_efficiency >= 0.90) & (self.irrigation_source_key != 0),
            self.yield_ratio_multiplier_value,
            1,
        )
        self.base_management_yield_ratio = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.float32,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "base_management_yield_ratio"
            ],
        )

        rng_drip = np.random.default_rng(70)
        self.time_adapted[self.adapted[:, 2] == 1, 2] = rng_drip.uniform(
            1,
            self.lifespan,
            np.sum(self.adapted[:, 2] == 1),
        )

        # Initiate array that tracks the overall yearly costs for all adaptations
        # 0 is input, 1 is microcredit, 2 is adaptation 1 (well), 3 is adaptation 2 (drip irrigation), 4 is water costs, last is total
        # Columns are the individual loans, i.e. if there are 2 loans for 2 wells, the first and second slot is used

        self.n_loans = 5

        self.all_loans_annual_cost = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(self.n_loans + 1, 5),
            extra_dims_names=("loan_type", "loans"),
            dtype=np.float32,
            fill_value=0,
        )

        self.adjusted_annual_loan_cost = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(self.n_loans + 1, 5),
            extra_dims_names=("loan_type", "loans"),
            dtype=np.float32,
            fill_value=np.nan,
        )

        # 0 is input, 1 is microcredit, 2 is adaptation 1 (well), 3 is adaptation 2 (drip irrigation)
        self.loan_tracker = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(self.n_loans, 5),
            extra_dims_names=("loan_type", "loans"),
            dtype=np.int32,
            fill_value=0,
        )

        # 0 is surface water / channel-dependent, 1 is reservoir-dependent, 2 is groundwater-dependent, 3 is rainwater-dependent
        self.farmer_class = AgentArray(
            n=self.n, max_n=self.max_n, dtype=np.int32, fill_value=-1
        )
        self.water_use = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(4,),
            extra_dims_names=("water_source",),
            dtype=np.int32,
            fill_value=0,
        )

        # Load the why class of agent's aquifer
        self.why_class = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.int32,
            fill_value=0,
        )

        why_map = load_grid(self.model.files["grid"]["groundwater/why_map"])

        self.why_class[:] = sample_from_map(
            why_map, self.locations.data, self.model.data.grid.gt
        )

        ## Load in the GEV_parameters, calculated from the extreme value distribution of the SPEI timeseries, and load in the original SPEI data
        self.GEV_parameters = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(3,),
            extra_dims_names=("gev_parameters",),
            dtype=np.float32,
            fill_value=np.nan,
        )

        if self.model.config["general"]["simulate_hydrology"]:
            for i, varname in enumerate(["gev_c", "gev_loc", "gev_scale"]):
                GEV_grid = getattr(self.model.data.grid, varname)
                self.GEV_parameters[:, i] = sample_from_map(
                    GEV_grid, self.locations.data, self.model.data.grid.gt
                )

        self.risk_perc_min = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.float32,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "expected_utility"
            ]["drought_risk_calculations"]["risk_perception"]["min"],
        )
        self.risk_perc_max = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.float32,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "expected_utility"
            ]["drought_risk_calculations"]["risk_perception"]["max"],
        )
        self.risk_decr = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.float32,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "expected_utility"
            ]["drought_risk_calculations"]["risk_perception"]["coef"],
        )
        self.decision_horizon = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.int32,
            fill_value=self.model.config["agent_settings"]["farmers"][
                "expected_utility"
            ]["decisions"]["decision_horizon"],
        )

        self.cumulative_water_deficit_m3 = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(366,),
            extra_dims_names=("day",),
            dtype=np.float32,
            fill_value=np.nan,
        )

        self.field_indices_by_farmer = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(2,),
            dtype=np.int32,
            fill_value=-1,
            extra_dims_names=("index",),
        )

        self.update_field_indices()

    @staticmethod
    @njit(cache=True)
    def update_field_indices_numba(
        land_owners: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Creates `field_indices_by_farmer` and `field_indices`. These indices are used to quickly find the fields for a specific farmer.

        Args:
            land_owners: Array of the land owners. Each unique ID is a different land owner. -1 means the land is not owned by anyone.

        Returns:
            field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.
        """
        agents = np.unique(land_owners)
        if agents[0] == -1:
            n_agents = agents.size - 1
        else:
            n_agents = agents.size
        field_indices_by_farmer = np.full((n_agents, 2), -1, dtype=np.int32)
        field_indices = np.full(land_owners.size, -1, dtype=np.int32)

        land_owners_sort_idx = np.argsort(land_owners)
        land_owners_sorted = land_owners[land_owners_sort_idx]

        last_not_owned = np.searchsorted(land_owners_sorted, -1, side="right")

        prev_land_owner = -1
        for i in range(last_not_owned, land_owners.size):
            land_owner = land_owners[land_owners_sort_idx[i]]
            if land_owner != -1:
                if land_owner != prev_land_owner:
                    field_indices_by_farmer[land_owner, 0] = i - last_not_owned
                field_indices_by_farmer[land_owner, 1] = i + 1 - last_not_owned
                field_indices[i - last_not_owned] = land_owners_sort_idx[i]
                prev_land_owner = land_owner
        field_indices = field_indices[:-last_not_owned]
        return field_indices_by_farmer, field_indices

    def update_field_indices(self) -> None:
        """Creates `field_indices_by_farmer` and `field_indices`. These indices are used to quickly find the fields for a specific farmer."""
        (
            self.field_indices_by_farmer[:],
            self.field_indices,
        ) = self.update_field_indices_numba(self.var.land_owners)

    def set_social_network(self) -> None:
        """
        Determines for each farmer a group of neighbors which constitutes their social network
        """
        nbits = 19
        radius = self.model.config["agent_settings"]["farmers"]["social_network"][
            "radius"
        ]
        n_neighbor = self.model.config["agent_settings"]["farmers"]["social_network"][
            "size"
        ]

        self.social_network = AgentArray(
            n=self.n,
            max_n=self.max_n,
            extra_dims=(n_neighbor,),
            extra_dims_names=("neighbors",),
            dtype=np.int32,
            fill_value=np.nan,
        )

        self.social_network[:] = find_neighbors(
            self.locations.data,
            radius=radius,
            n_neighbor=n_neighbor,
            bits=nbits,
            minx=self.model.bounds[0],
            maxx=self.model.bounds[1],
            miny=self.model.bounds[2],
            maxy=self.model.bounds[3],
        )

    @property
    def activation_order_by_elevation(self):
        """
        Activation order is determined by the agent elevation, starting from the highest.
        Agents with the same elevation are randomly shuffled.
        """
        # if activation order is fixed. Get the random state, and set a fixed seet.
        if self.model.config["agent_settings"]["fix_activation_order"]:
            if (
                hasattr(self, "activation_order_by_elevation_fixed")
                and self.activation_order_by_elevation_fixed[0] == self.n
            ):
                return self.activation_order_by_elevation_fixed[1]
            random_state = np.random.get_state()
            np.random.seed(42)
        elevation = self.elevation
        # Shuffle agent elevation and agent_ids in unision.
        p = np.random.permutation(elevation.size)
        # if activation order is fixed, set random state to previous state
        if self.model.config["agent_settings"]["fix_activation_order"]:
            np.random.set_state(random_state)
        elevation_shuffled = elevation[p]
        agent_ids_shuffled = np.arange(0, elevation.size, 1, dtype=np.int32)[p]
        # Use argsort to find the order or the shuffled elevation. Using a stable sorting
        # algorithm such that the random shuffling in the previous step is conserved
        # in groups with identical elevation.
        activation_order_shuffled = np.argsort(elevation_shuffled, kind="stable")[::-1]
        # unshuffle the agent_ids to get the activation order
        activation_order = agent_ids_shuffled[activation_order_shuffled]
        if self.model.config["agent_settings"]["fix_activation_order"]:
            self.activation_order_by_elevation_fixed = (self.n, activation_order)
        # Check if the activation order is correct, by checking if elevation is decreasing
        assert np.diff(elevation[activation_order]).max() <= 0
        return activation_order

    @property
    def farmer_command_area(self):
        return farmer_command_area(
            self.n,
            self.field_indices,
            self.field_indices_by_farmer.data,
            self.var.reservoir_command_areas,
        )

    @property
    def is_in_command_area(self):
        return self.farmer_command_area != -1

    def save_water_deficit(self, discount_factor=0.8):
        water_deficit_day_m3 = (
            self.model.data.HRU.ETRef - self.model.data.HRU.pr
        ) * self.model.data.HRU.cellArea
        water_deficit_day_m3[water_deficit_day_m3 < 0] = 0

        water_deficit_day_m3_per_farmer = np.bincount(
            self.var.land_owners[self.var.land_owners != -1],
            weights=water_deficit_day_m3[self.var.land_owners != -1],
        )

        def add_water_deficit(water_deficit_day_m3_per_farmer, index):
            self.cumulative_water_deficit_m3[:, index] = np.where(
                np.isnan(self.cumulative_water_deficit_m3[:, index]),
                water_deficit_day_m3_per_farmer,
                self.cumulative_water_deficit_m3[:, index],
            )

        if self.model.current_day_of_year == 1:
            add_water_deficit(
                water_deficit_day_m3_per_farmer, self.model.current_day_of_year - 1
            )
            self.cumulative_water_deficit_m3[:, self.model.current_day_of_year - 1] = (
                water_deficit_day_m3_per_farmer
            )
        else:
            self.cumulative_water_deficit_m3[:, self.model.current_day_of_year - 1] = (
                self.cumulative_water_deficit_m3[:, self.model.current_day_of_year - 2]
                + water_deficit_day_m3_per_farmer
            )
            # if this is the last day of the year, but not a leap year, the virtual
            # 366th day of the year is the same as the 365th day of the year
            # this avoids complications with the leap year
            if self.model.current_day_of_year == 365 and not calendar.isleap(
                self.model.current_time.year
            ):
                self.cumulative_water_deficit_m3[:, 365] = (
                    self.cumulative_water_deficit_m3[:, 364]
                )

    def abstract_water(
        self,
        cell_area: np.ndarray,
        paddy_level: np.ndarray,
        readily_available_water: np.ndarray,
        critical_water_level: np.ndarray,
        max_water_content: np.ndarray,
        potential_infiltration_capacity: np.ndarray,
        available_channel_storage_m3: np.ndarray,
        available_groundwater_m3: np.ndarray,
        groundwater_depth: np.ndarray,
        available_reservoir_storage_m3: np.ndarray,
        command_areas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function allows the abstraction of water by farmers for irrigation purposes. It's main purpose is to call the relevant numba function to do the actual abstraction. In addition, the function saves the abstraction from the various sources by farmer.

        Args:
            cell_area: the area of each subcell in m2.
            HRU_to_grid: array to map the index of each subcell to the corresponding cell.
            totalPotIrrConsumption: potential irrigation consumption.
            available_channel_storage_m3: water available for irrigation from channels.
            groundwater_head: groundwater head.
            available_groundwater_m3: water available for irrigation from groundwater.
            available_reservoir_storage_m3: water available for irrigation from reservoirs.
            command_areas: command areas associated with reservoirs (i.e., which areas can access water from which reservoir.)

        Returns:
            water_withdrawal_m: water withdrawal in meters
            water_consumption_m: water consumption in meters
            returnFlowIrr_m: return flow in meters
            addtoevapotrans_m: evaporated irrigation water in meters
        """
        assert (available_channel_storage_m3 >= 0).all()
        assert (available_groundwater_m3 >= 0).all()
        assert (available_reservoir_storage_m3 >= 0).all()

        self.activation_order_by_elevation_ = AgentArray(
            self.activation_order_by_elevation, max_n=self.max_n
        )

        if __debug__:
            irrigation_limit_pre = self.remaining_irrigation_limit_m3.copy()
            available_channel_storage_m3_pre = available_channel_storage_m3.copy()
            available_groundwater_m3_pre = available_groundwater_m3.copy()
            available_reservoir_storage_m3_pre = available_reservoir_storage_m3.copy()
        (
            self.channel_abstraction_m3_by_farmer[:],
            self.reservoir_abstraction_m3_by_farmer[:],
            self.groundwater_abstraction_m3_by_farmer[:],
            water_withdrawal_m,
            water_consumption_m,
            returnFlowIrr_m,
            addtoevapotrans_m,
        ) = abstract_water(
            self.model.current_day_of_year - 1,
            self.n,
            self.activation_order_by_elevation,
            self.field_indices_by_farmer.data,
            self.field_indices,
            self.irrigation_efficiency.data,
            surface_irrigated=np.isin(
                self.irrigation_source,
                np.array(
                    [
                        self.irrigation_source_key["canal"],
                    ]
                ),
            ),
            well_irrigated=np.isin(
                self.irrigation_source,
                np.array(
                    [
                        self.irrigation_source_key["well"],
                    ]
                ),
            ),
            cell_area=cell_area,
            HRU_to_grid=self.model.data.HRU.HRU_to_grid,
            crop_map=self.var.crop_map,
            field_is_paddy_irrigated=self.field_is_paddy_irrigated,
            paddy_level=paddy_level,
            readily_available_water=readily_available_water,
            critical_water_level=critical_water_level,
            max_water_content=max_water_content,
            potential_infiltration_capacity=potential_infiltration_capacity,
            available_channel_storage_m3=available_channel_storage_m3,
            available_groundwater_m3=available_groundwater_m3,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
            groundwater_depth=groundwater_depth,
            farmer_command_area=self.farmer_command_area,
            return_fraction=self.model.config["agent_settings"]["farmers"][
                "return_fraction"
            ],
            well_depth=self.well_depth.data,
            remaining_irrigation_limit_m3=self.remaining_irrigation_limit_m3.data,
            cumulative_water_deficit_m3=self.cumulative_water_deficit_m3.data,
            crop_calendar=self.crop_calendar.data,
            current_crop_calendar_rotation_year_index=self.current_crop_calendar_rotation_year_index.data,
            max_paddy_water_level=self.max_paddy_water_level,
        )

        if __debug__:
            # make sure the withdrawal per source is identical to the total withdrawal in m (corrected for cell area)
            balance_check(
                name="water withdrawal_1",
                how="sum",
                influxes=(
                    self.channel_abstraction_m3_by_farmer,
                    self.reservoir_abstraction_m3_by_farmer,
                    self.groundwater_abstraction_m3_by_farmer,
                ),
                outfluxes=[(water_withdrawal_m * cell_area)],
                tollerance=10,
            )

            # assert that the total amount of water withdrawn is equal to the total storage before and after abstraction
            balance_check(
                name="water withdrawal channel",
                how="sum",
                outfluxes=self.channel_abstraction_m3_by_farmer,
                prestorages=available_channel_storage_m3_pre,
                poststorages=available_channel_storage_m3,
                tollerance=10,
            )

            balance_check(
                name="water withdrawal reservoir",
                how="sum",
                outfluxes=self.reservoir_abstraction_m3_by_farmer,
                prestorages=available_reservoir_storage_m3_pre,
                poststorages=available_reservoir_storage_m3,
                tollerance=10,
            )

            balance_check(
                name="water withdrawal groundwater",
                how="sum",
                outfluxes=self.groundwater_abstraction_m3_by_farmer,
                prestorages=available_groundwater_m3_pre,
                poststorages=available_groundwater_m3,
                tollerance=10,
            )

            # assert that the total amount of water withdrawn is equal to the total storage before and after abstraction
            balance_check(
                name="water withdrawal_2",
                how="sum",
                influxes=(
                    self.channel_abstraction_m3_by_farmer[
                        ~np.isnan(self.remaining_irrigation_limit_m3)
                    ],
                    self.reservoir_abstraction_m3_by_farmer[
                        ~np.isnan(self.remaining_irrigation_limit_m3)
                    ],
                    self.groundwater_abstraction_m3_by_farmer[
                        ~np.isnan(self.remaining_irrigation_limit_m3)
                    ],
                ),
                prestorages=irrigation_limit_pre[
                    ~np.isnan(self.remaining_irrigation_limit_m3)
                ],
                poststorages=self.remaining_irrigation_limit_m3[
                    ~np.isnan(self.remaining_irrigation_limit_m3)
                ],
            )

            # make sure the total water consumption plus 'wasted' irrigation water (evaporation + return flow) is equal to the total water withdrawal
            balance_check(
                name="water consumption",
                how="sum",
                influxes=(
                    water_consumption_m,
                    returnFlowIrr_m,
                    addtoevapotrans_m,
                ),
                outfluxes=water_withdrawal_m,
                tollerance=0.0001,
            )

            assert water_withdrawal_m.dtype == np.float32
            assert water_consumption_m.dtype == np.float32
            assert returnFlowIrr_m.dtype == np.float32
            assert addtoevapotrans_m.dtype == np.float32

        return (
            water_withdrawal_m,
            water_consumption_m,
            returnFlowIrr_m,
            addtoevapotrans_m,
        )

    @staticmethod
    @njit(cache=True)
    def get_yield_ratio_numba_GAEZ(
        crop_map: np.ndarray, evap_ratios: np.ndarray, KyT
    ) -> float:
        """Calculate yield ratio based on https://doi.org/10.1016/j.jhydrol.2009.07.031

        Args:
            crop_map: array of currently harvested crops.
            evap_ratios: ratio of actual to potential evapotranspiration of harvested crops.
            KyT: Water stress reduction factor from GAEZ.

        Returns:
            yield_ratios: yield ratio (as ratio of maximum obtainable yield) per harvested crop.
        """
        yield_ratios = np.full(evap_ratios.size, -1, dtype=np.float32)

        assert crop_map.size == evap_ratios.size

        for i in range(evap_ratios.size):
            evap_ratio = evap_ratios[i]
            crop = crop_map[i]
            yield_ratios[i] = max(
                1 - KyT[crop] * (1 - evap_ratio), 0
            )  # Yield ratio is never lower than 0.

        return yield_ratios

    @staticmethod
    @njit(cache=True)
    def get_yield_ratio_numba_MIRCA2000(
        crop_map: np.ndarray,
        evap_ratios: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        P0: np.ndarray,
        P1: np.ndarray,
    ) -> float:
        """Calculate yield ratio based on https://doi.org/10.1016/j.jhydrol.2009.07.031

        Args:
            crop_map: array of currently harvested crops.
            evap_ratios: ratio of actual to potential evapotranspiration of harvested crops.
            alpha: alpha value per crop used in MIRCA2000.
            beta: beta value per crop used in MIRCA2000.
            P0: P0 value per crop used in MIRCA2000.
            P1: P1 value per crop used in MIRCA2000.

        Returns:
            yield_ratios: yield ratio (as ratio of maximum obtainable yield) per harvested crop.
        """
        yield_ratios = np.full(evap_ratios.size, -1, dtype=np.float32)

        assert crop_map.size == evap_ratios.size

        for i in range(evap_ratios.size):
            evap_ratio = evap_ratios[i]
            crop = crop_map[i]
            if alpha[crop] * evap_ratio + beta[crop] > 1:
                yield_ratio = 1
            elif P0[crop] < evap_ratio < P1[crop]:
                yield_ratio = (
                    alpha[crop] * P1[crop]
                    + beta[crop]
                    - (P1[crop] - evap_ratio)
                    * (alpha[crop] * P1[crop] + beta[crop])
                    / (P1[crop] - P0[crop])
                )
            elif evap_ratio < P0[crop]:
                yield_ratio = 0
            else:
                yield_ratio = alpha[crop] * evap_ratio + beta[crop]
            yield_ratios[i] = yield_ratio

        return yield_ratios

    def get_yield_ratio(
        self,
        harvest: np.ndarray,
        actual_transpiration: np.ndarray,
        potential_transpiration: np.ndarray,
        crop_map: np.ndarray,
    ) -> np.ndarray:
        """Gets yield ratio for each crop given the ratio between actual and potential evapostranspiration during growth.

        Args:
            harvest: Map of crops that are harvested.
            actual_transpiration: Actual evapotranspiration during crop growth period.
            potential_transpiration: Potential evapotranspiration during crop growth period.
            crop_map: Subarray of type of crop grown.

        Returns:
            yield_ratio: Map of yield ratio.

        TODO: Implement GAEZ crop stage function
        """
        if self.model.config["general"]["simulate_hydrology"]:
            if self.crop_data_type == "GAEZ":
                yield_ratio = self.get_yield_ratio_numba_GAEZ(
                    crop_map[harvest],
                    actual_transpiration[harvest] / potential_transpiration[harvest],
                    self.crop_data["KyT"].values,
                )
            elif self.crop_data_type == "MIRCA2000":
                yield_ratio = self.get_yield_ratio_numba_MIRCA2000(
                    crop_map[harvest],
                    actual_transpiration[harvest] / potential_transpiration[harvest],
                    self.crop_data["a"].values,
                    self.crop_data["b"].values,
                    self.crop_data["P0"].values,
                    self.crop_data["P1"].values,
                )
                if np.any(yield_ratio == 0):
                    pass
            else:
                raise ValueError(
                    f"Unknown crop data type: {self.crop_data_type}, must be 'GAEZ' or 'MIRCA2000'"
                )
            assert not np.isnan(yield_ratio).any()
        else:
            yield_ratio = np.full_like(crop_map[harvest], 1, dtype=np.float32)

        return yield_ratio

    def decompress(self, array):
        if np.issubsctype(array, np.floating):
            nofieldvalue = np.nan
        else:
            nofieldvalue = -1
        by_field = np.take(array, self.var.land_owners)
        by_field[self.var.land_owners == -1] = nofieldvalue
        return self.model.data.HRU.decompress(by_field)

    @property
    def mask(self):
        mask = self.model.data.HRU.mask.copy()
        mask[self.decompress(self.var.land_owners) == -1] = True
        return mask

    @staticmethod
    @njit(cache=True)
    def harvest_numba(
        n: np.ndarray,
        field_indices_by_farmer: np.ndarray,
        field_indices: np.ndarray,
        crop_map: np.ndarray,
        crop_age_days: np.ndarray,
        crop_harvest_age_days: np.ndarray,
    ) -> np.ndarray:
        """This function determines whether crops are ready to be harvested by comparing the crop harvest age to the current age of the crop. If the crop is harvested, the crops next multicrop index and next plant day are determined.

        Args:
            n: Number of farmers.
            start_day_per_month: Array containing the starting day of each month.
            field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.
            crop_map: Subarray map of crops.
            crop_age_days: Subarray map of current crop age in days.
            crop: Crops grown by each farmer.
            switch_crops: Whether to switch crops or not.

        Returns:
            harvest: Boolean subarray map of fields to be harvested.
        """
        harvest = np.zeros(crop_map.shape, dtype=np.bool_)
        for farmer_i in range(n):
            farmer_fields = get_farmer_HRUs(
                field_indices, field_indices_by_farmer, farmer_i
            )
            for field in farmer_fields:
                crop_age = crop_age_days[field]
                if crop_age >= 0:
                    crop = crop_map[field]
                    assert crop != -1
                    if crop_age == crop_harvest_age_days[field]:
                        harvest[field] = True
                        crop_harvest_age_days[field] = -1
                else:
                    assert crop_map[field] == -1
        return harvest

    def harvest(self):
        """
        Determine which crops need to be harvested based on their current age and their harvest age.
        Once harvested, compute various metrics related to the harvest including potential profit,
        actual profit, crop age, drought perception, and update corresponding attributes of the model.
        Save the corresponding SPEI over the last harvest.

        Attributes:
            harvest_numba: A helper function to obtain the harvest map.
            get_yield_ratio: A function to calculate yield ratio based on the ratio of actual to potential evapotranspiration.

        Note:
            The function also updates the drought risk perception and tracks disposable income.
        """

        # Using the helper function to determine which crops are ready to be harvested
        harvest = self.harvest_numba(
            n=self.n,
            field_indices_by_farmer=self.field_indices_by_farmer.data,
            field_indices=self.field_indices,
            crop_map=self.var.crop_map,
            crop_age_days=self.var.crop_age_days_map,
            crop_harvest_age_days=self.var.crop_harvest_age_days,
        )

        self.actual_yield_per_farmer.fill(np.nan)
        self.harvested_crop.fill(-1)
        # If there are fields to be harvested, compute yield ratio and various related metrics
        if np.count_nonzero(harvest):
            # Get yield ratio for the harvested crops
            yield_ratio_per_field = self.get_yield_ratio(
                harvest,
                self.var.actual_evapotranspiration_crop_life,
                self.var.potential_evapotranspiration_crop_life,
                self.var.crop_map,
            )
            assert (yield_ratio_per_field >= 0).all()

            harvesting_farmer_fields = self.var.land_owners[harvest]
            harvested_area = self.var.cellArea[harvest]
            if self.model.use_gpu:
                harvested_area = harvested_area.get()

            harvested_crops = self.var.crop_map[harvest]
            max_yield_per_crop = np.take(
                self.crop_data["reference_yield_kg_m2"].values, harvested_crops
            )
            harvesting_farmers = np.unique(harvesting_farmer_fields)

            # it's okay for some crop prices to be nan, as they will be filtered out in the next step
            crop_prices = self.agents.market.crop_prices
            region_id_per_field = self.region_id

            # Determine the region ids of harvesting farmers, as crop prices differ per region

            region_id_per_field = self.region_id[self.var.land_owners]
            region_id_per_field[self.var.land_owners == -1] = -1
            region_id_per_harvested_field = region_id_per_field[harvest]

            # Calculate the crop price per field
            crop_price_per_field = crop_prices[
                region_id_per_harvested_field, harvested_crops
            ]

            # but it's not okay for the crop price to be nan now
            assert not np.isnan(crop_price_per_field).any()

            # Correct yield ratio
            yield_ratio_per_field = (
                self.base_management_yield_ratio[harvesting_farmer_fields]
                * yield_ratio_per_field
            )

            # Calculate the potential yield per field
            potential_yield_per_field = max_yield_per_crop * harvested_area

            # Calculate the total yield per field
            actual_yield_per_field = yield_ratio_per_field * potential_yield_per_field

            # And sum the total yield per field to get the total yield per farmer
            self.actual_yield_per_farmer[:] = np.bincount(
                harvesting_farmer_fields,
                weights=actual_yield_per_field,
                minlength=self.n,
            )

            # get the harvested crop per farmer. This assumes each farmer only harvests one crop
            # on the same day
            self.harvested_crop[harvesting_farmers] = harvested_crops[
                np.unique(self.var.land_owners[harvest], return_index=True)[1]
            ]

            # Determine the actual and potential profits
            potential_profit_per_field = (
                potential_yield_per_field * crop_price_per_field
            )
            actual_profit_per_field = actual_yield_per_field * crop_price_per_field
            assert (potential_profit_per_field >= 0).all()
            assert (actual_profit_per_field >= 0).all()

            # Convert from the profit and potential profit per field to the profit per farmer
            potential_profit_farmer = np.bincount(
                harvesting_farmer_fields,
                weights=potential_profit_per_field,
                minlength=self.n,
            )
            self.profit_farmer = np.bincount(
                harvesting_farmer_fields,
                weights=actual_profit_per_field,
                minlength=self.n,
            )

            # Convert the yield_ratio per field to the average yield ratio per farmer
            # yield_ratio_per_farmer = self.profit_farmer / potential_profit_farmer

            # Get the current crop age
            crop_age = self.var.crop_age_days_map[harvest]
            total_crop_age = np.bincount(
                harvesting_farmer_fields, weights=crop_age, minlength=self.n
            ) / np.bincount(harvesting_farmer_fields, minlength=self.n)

            self.total_crop_age[harvesting_farmers, 0] = total_crop_age[
                harvesting_farmers
            ]

            harvesting_farmers_mask = np.zeros(self.n, dtype=bool)
            harvesting_farmers_mask[harvesting_farmers] = True

            self.save_yearly_profits(self.profit_farmer, potential_profit_farmer)
            self.save_harvest_spei(harvesting_farmers)
            self.drought_risk_perception(harvesting_farmers, total_crop_age)

            ## After updating the drought risk perception, set the previous month for the next timestep as the current for this timestep.
            # TODO: This seems a bit like a quirky solution, perhaps there is a better way to do this.
            self.previous_month = self.model.current_time.month

        else:
            self.profit_farmer = np.zeros(self.n, dtype=np.float32)

        # Reset transpiration values for harvested fields
        self.var.actual_evapotranspiration_crop_life[harvest] = 0
        self.var.potential_evapotranspiration_crop_life[harvest] = 0

        # Update crop and land use maps after harvest
        self.var.crop_map[harvest] = -1
        self.var.crop_age_days_map[harvest] = -1
        self.var.land_use_type[harvest] = GRASSLAND_LIKE

        # For unharvested growing crops, increase their age by 1
        self.var.crop_age_days_map[(~harvest) & (self.var.crop_map >= 0)] += 1

        assert (self.var.crop_age_days_map <= self.var.crop_harvest_age_days).all()

    def drought_risk_perception(
        self, harvesting_farmers: np.ndarray, total_crop_age: np.ndarray
    ) -> None:
        """Calculate and update the drought risk perception for harvesting farmers.

        Args:
            harvesting_farmers: Index array of farmers that are currently harvesting.

        This function computes the risk perception of farmers based on the difference
        between their latest profits and potential profits. The perception is influenced
        by the historical losses and time since the last drought event. Farmers who have
        experienced a drought event will have their drought timer reset.

        TODO: Perhaps move the constant to the model.yml
        """
        # constants
        HISTORICAL_PERIOD = min(5, self.yearly_potential_profits.shape[1])  # years

        # Convert the harvesting farmers index array to a boolean array of full length
        harvesting_farmers_long = np.zeros(self.n, dtype=bool)
        harvesting_farmers_long[harvesting_farmers] = True

        # Update the drought timer based on the months passed since the previous check
        months_passed = (self.model.current_time.month - self.previous_month) % 12
        self.drought_timer += months_passed / 12

        # Create an empty drought loss np.ndarray
        drought_loss_historical = np.zeros(
            (self.n, HISTORICAL_PERIOD), dtype=np.float32
        )

        # Compute the percentage loss between potential and actual profits for harvesting farmers
        potential_profits = self.yearly_potential_profits[
            harvesting_farmers_long, :HISTORICAL_PERIOD
        ]
        actual_profits = self.yearly_profits[
            harvesting_farmers_long, :HISTORICAL_PERIOD
        ]
        drought_loss_historical[harvesting_farmers_long] = (
            (potential_profits - actual_profits) / potential_profits
        ) * 100

        # Calculate the current and past average loss percentages
        drought_loss_latest = drought_loss_historical[:, 0]
        drought_loss_past = np.mean(drought_loss_historical[:, 1:], axis=1)

        # Identify farmers who experienced a drought event based on loss comparison with historical losses
        drought_loss_current = drought_loss_latest - drought_loss_past

        experienced_drought_event = (
            drought_loss_current >= self.moving_average_threshold
        )

        # Reset the drought timer for farmers who have harvested and experienced a drought event
        self.drought_timer[
            np.logical_and(harvesting_farmers_long, experienced_drought_event)
        ] = 0

        # Update the risk perception of all farmers
        self.risk_perception = (
            self.risk_perc_max * (1.6 ** (self.risk_decr * self.drought_timer))
            + self.risk_perc_min
        )

        # print("Risk perception mean = ", np.mean(self.risk_perception))

        # Determine which farmers need emergency microcredit to keep farming
        loaning_farmers = drought_loss_current >= self.moving_average_threshold

        # Determine their microcredit
        if (
            np.any(loaning_farmers)
            and "ruleset" in self.config
            and not self.config["ruleset"] == "no-adaptation"
        ):
            # print(np.count_nonzero(loaning_farmers), "farmers are getting microcredit")
            self.microcredit(loaning_farmers, drought_loss_current, total_crop_age)

    def microcredit(
        self,
        loaning_farmers: np.ndarray,
        drought_loss_current: np.ndarray,
        total_crop_age: np.ndarray,
    ) -> None:
        """
        Compute the microcredit for farmers based on their average profits, drought losses, and the age of their crops
        with respect to their total cropping time.

        Parameters:
        - loaning_farmers: Boolean mask of farmers looking to obtain a loan, based on drought loss of harvesting farmers.
        - drought_loss_current: Array of drought losses of the most recent harvest for each farmer.
        - total_crop_age: Array of total age for crops of each farmer.
        """

        # Compute the maximum loan amount based on the average profits of the last 10 years
        max_loan = np.median(self.yearly_profits[loaning_farmers, :5], axis=1)

        # Compute the crop age as a percentage of the average total time a farmer has had crops planted
        crop_age_fraction = total_crop_age[loaning_farmers] / np.mean(
            self.total_crop_age[loaning_farmers], axis=1
        )

        # Calculate the total loan amount based on drought loss, crop age percentage, and the maximum loan
        total_loan = (
            (drought_loss_current[loaning_farmers] / 100) * crop_age_fraction * max_loan
        )

        # Fetch loan configurations from the model settings
        loan_duration = self.model.config["agent_settings"]["farmers"]["microcredit"][
            "loan_duration"
        ]

        # Compute the annual cost of the loan using the interest rate and loan duration
        annual_cost_microcredit = total_loan * (
            self.interest_rate[loaning_farmers]
            * (1 + self.interest_rate[loaning_farmers]) ** loan_duration
            / ((1 + self.interest_rate[loaning_farmers]) ** loan_duration - 1)
        )

        # Add the amounts to the individual loan slots
        self.set_loans_numba(
            all_loans_annual_cost=self.all_loans_annual_cost.data,
            loan_tracker=self.loan_tracker.data,
            loaning_farmers=loaning_farmers,
            annual_cost_loan=annual_cost_microcredit,
            loan_duration=loan_duration,
            loan_type=1,
        )

        # Add it to the loan total
        self.all_loans_annual_cost[loaning_farmers, -1, 0] += annual_cost_microcredit

    @staticmethod
    @njit(cache=True)
    def set_loans_numba(
        all_loans_annual_cost: np.ndarray,
        loan_tracker: np.ndarray,
        loaning_farmers: np.ndarray,
        annual_cost_loan: np.ndarray,
        loan_duration: int,
        loan_type: int,
    ) -> None:
        farmers_getting_loan = np.where(loaning_farmers)[0]

        # Update the agent's loans and total annual costs with the computed annual cost
        # Make sure it is in an empty loan slot
        for farmer in farmers_getting_loan:
            for i in range(4):
                if all_loans_annual_cost[farmer, loan_type, i] == 0:
                    local_index = np.where(farmers_getting_loan == farmer)[0][0]
                    all_loans_annual_cost[farmer, loan_type, i] += annual_cost_loan[
                        local_index
                    ]
                    loan_tracker[farmer, loan_type, i] = loan_duration
                    break  # Exit the loop after adding to the first zero value

    def plant(self) -> None:
        """Determines when and what crop should be planted, mainly through calling the :meth:`agents.farmers.Farmers.plant_numba`. Then converts the array to cupy array if model is running with GPU."""

        if self.cultivation_costs[0] is None:
            cultivation_cost = self.cultivation_costs[1]
        else:
            index = self.cultivation_costs[0].get(self.model.current_time)
            cultivation_cost = self.cultivation_costs[1][index]
            assert cultivation_cost.shape[0] == len(self.model.regions)
            assert cultivation_cost.shape[1] == len(self.crop_ids)

        plant_map, farmers_selling_land = plant(
            n=self.n,
            day_index=self.model.current_time.timetuple().tm_yday - 1,  # 0-indexed
            crop_calendar=self.crop_calendar.data,
            current_crop_calendar_rotation_year_index=self.current_crop_calendar_rotation_year_index.data,
            crop_map=self.var.crop_map,
            crop_harvest_age_days=self.var.crop_harvest_age_days,
            cultivation_cost=cultivation_cost,
            region_ids_per_farmer=self.region_id.data,
            field_indices_by_farmer=self.field_indices_by_farmer.data,
            field_indices=self.field_indices,
            field_size_per_farmer=self.field_size_per_farmer.data,
            all_loans_annual_cost=self.all_loans_annual_cost.data,
            loan_tracker=self.loan_tracker.data,
            interest_rate=self.interest_rate.data,
            farmers_going_out_of_business=False,
        )
        if farmers_selling_land.size > 0:
            self.remove_agents(farmers_selling_land)

        self.var.crop_map = np.where(plant_map >= 0, plant_map, self.var.crop_map)
        self.var.crop_age_days_map[plant_map >= 0] = 1

        assert (self.var.crop_age_days_map[self.var.crop_map > 0] >= 0).all()

        self.var.land_use_type[
            (self.var.crop_map >= 0) & (self.field_is_paddy_irrigated)
        ] = PADDY_IRRIGATED
        self.var.land_use_type[
            (self.var.crop_map >= 0) & (~self.field_is_paddy_irrigated)
        ] = NON_PADDY_IRRIGATED

    @property
    def field_is_paddy_irrigated(self):
        return np.isin(
            self.var.crop_map, self.crop_data[self.crop_data["is_paddy"]].index
        )

    def water_abstraction_sum(self) -> None:
        """
        Aggregates yearly water abstraction from different sources (channel, reservoir, groundwater) for each farmer
        and also computes the total abstraction per farmer.

        Note:
            This function performs the following steps:
                1. Updates the yearly channel water abstraction for each farmer.
                2. Updates the yearly reservoir water abstraction for each farmer.
                3. Updates the yearly groundwater water abstraction for each farmer.
                4. Computes and updates the total water abstraction for each farmer.

        """

        # Update yearly channel water abstraction for each farmer
        self.yearly_abstraction_m3_by_farmer[:, 0, 0] += (
            self.channel_abstraction_m3_by_farmer
        )

        # Update yearly reservoir water abstraction for each farmer
        self.yearly_abstraction_m3_by_farmer[:, 1, 0] += (
            self.reservoir_abstraction_m3_by_farmer
        )

        # Update yearly groundwater water abstraction for each farmer
        self.yearly_abstraction_m3_by_farmer[:, 2, 0] += (
            self.groundwater_abstraction_m3_by_farmer
        )

        # Compute and update the total water abstraction for each farmer
        self.yearly_abstraction_m3_by_farmer[:, 3, 0] += (
            self.channel_abstraction_m3_by_farmer
            + self.reservoir_abstraction_m3_by_farmer
            + self.groundwater_abstraction_m3_by_farmer
        )

    def save_harvest_spei(self, harvesting_farmers) -> None:
        """
        Update the monthly Standardized Precipitation Evapotranspiration Index (SPEI) array by shifting past records and
        adding the SPEI for the current month.

        Note:
            This method updates the `monthly_SPEI` attribute in place.
        """
        current_SPEI_per_farmer = sample_from_map(
            array=self.model.data.grid.spei_uncompressed,
            coords=self.locations[harvesting_farmers].data,
            gt=self.model.data.grid.gt,
        )

        full_size_SPEI_per_farmer = np.zeros_like(
            self.cumulative_SPEI_during_growing_season
        )
        full_size_SPEI_per_farmer[harvesting_farmers] = current_SPEI_per_farmer

        cumulative_mean(
            mean=self.cumulative_SPEI_during_growing_season,
            counter=self.cumulative_SPEI_count_during_growing_season,
            update=full_size_SPEI_per_farmer,
            mask=harvesting_farmers,
        )
        print(
            "season SPEI",
            np.mean(full_size_SPEI_per_farmer[harvesting_farmers]),
        )

    def save_yearly_spei(self):
        assert self.model.current_time.month == 1

        # calculate the SPEI probability using GEV parameters
        SPEI_probability = genextreme.sf(
            self.cumulative_SPEI_during_growing_season,
            self.GEV_parameters[:, 0],
            self.GEV_parameters[:, 1],
            self.GEV_parameters[:, 2],
        )

        # SPEI_probability_norm = norm.cdf(self.cumulative_SPEI_during_growing_season)

        print("Yearly probability", np.mean(1 - SPEI_probability))

        shift_and_update(self.yearly_SPEI_probability, (1 - SPEI_probability))

        # Reset the cumulative SPEI array at the beginning of the year
        self.cumulative_SPEI_during_growing_season.fill(0)
        self.cumulative_SPEI_count_during_growing_season.fill(0)

    def save_yearly_profits(
        self,
        profit: np.ndarray,
        potential_profit: np.ndarray,
    ) -> None:
        """
        Saves the latest profit and potential profit values for harvesting farmers to determine yearly profits, considering inflation and field size.

        Args:
            harvesting_farmers: Array of farmers who are currently harvesting.
            profit: Array representing the profit value for each farmer per season.
            potential_profit: Array representing the potential profit value for each farmer per season.

        Note:
            This function performs the following operations:
                1. Asserts that all profit and potential profit values are non-negative.
                2. Updates the latest profits and potential profits matrices by shifting all columns one column further.
                The last column value is dropped.
                3. Adjusts the yearly profits by accounting for the latest profit, field size, and inflation.
        """

        # Ensure that all profit and potential profit values are non-negative
        assert (profit >= 0).all()
        assert (potential_profit >= 0).all()

        # Calculate the cumulative inflation from the start year to the current year for each farmer
        inflation_arrays = [
            self.get_value_per_farmer_from_region_id(
                self.inflation_rate, datetime(year, 1, 1)
            )
            for year in range(
                self.model.config["general"]["spinup_time"].year,
                self.model.current_time.year + 1,
            )
        ]

        cum_inflation = np.ones_like(inflation_arrays[0])
        for inflation in inflation_arrays:
            cum_inflation *= inflation

        # Adjust yearly profits by the latest profit, field size, and cumulative inflation for each harvesting farmer
        self.yearly_profits[:, 0] += profit / cum_inflation
        self.yearly_potential_profits[:, 0] += potential_profit / cum_inflation

    def calculate_yield_spei_relation_test_solo(self):
        import os
        import matplotlib

        matplotlib.use("Agg")  # Use the 'Agg' backend for non-interactive plotting
        import matplotlib.pyplot as plt

        # Number of agents
        n_agents = self.yearly_yield_ratio.shape[0]

        # Define regression models
        def linear_model(X, a, b):
            return a * X + b

        def exponential_model(X, a, b):
            return a * np.exp(b * X)

        def logarithmic_model(X, a, b):
            return a * np.log(X) + b

        def quadratic_model(X, a, b, c):
            return a * X**2 + b * X + c

        def power_model(X, a, b):
            return a * X**b

        # Initialize dictionaries for coefficients and R² values
        model_names = ["linear", "exponential", "logarithmic", "quadratic", "power"]
        r_squared_dict = {model: np.zeros(n_agents) for model in model_names}
        coefficients_dict = {model: [] for model in model_names}

        # Create a folder to save the plots
        output_folder = "plot/relation_test"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # For each agent, perform regression with different models
        for agent_idx in range(n_agents):
            # Get data for the agent
            y_data = self.yearly_yield_ratio[agent_idx, :]  # shape (n_years,)
            X_data = self.yearly_SPEI_probability[agent_idx, :]  # shape (n_years,)

            # Filter out invalid values
            valid_mask = (
                (~np.isnan(X_data)) & (~np.isnan(y_data)) & (X_data > 0) & (y_data != 0)
            )
            X_valid = X_data[valid_mask]
            y_valid = y_data[valid_mask]

            if len(X_valid) >= 2:
                # Prepare data
                X_log = np.log10(X_valid)

                # Model 1: Linear in log-transformed X
                try:
                    popt, _ = curve_fit(linear_model, X_log, y_valid, maxfev=10000)
                    a, b = popt
                    y_pred = linear_model(X_log, a, b)
                    ss_res = np.sum((y_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["linear"][agent_idx] = r_squared
                    coefficients_dict["linear"].append((a, b))
                except RuntimeError:
                    r_squared_dict["linear"][agent_idx] = np.nan
                    coefficients_dict["linear"].append((np.nan, np.nan))

                # Model 2: Exponential
                try:
                    popt, _ = curve_fit(
                        exponential_model, X_valid, y_valid, maxfev=10000
                    )
                    y_pred = exponential_model(X_valid, *popt)
                    ss_res = np.sum((y_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["exponential"][agent_idx] = r_squared
                    coefficients_dict["exponential"].append(popt)
                except RuntimeError:
                    r_squared_dict["exponential"][agent_idx] = np.nan
                    coefficients_dict["exponential"].append((np.nan, np.nan))

                # Model 3: Logarithmic (ensure X > 0)
                try:
                    popt, _ = curve_fit(
                        logarithmic_model, X_valid, y_valid, maxfev=10000
                    )
                    y_pred = logarithmic_model(X_valid, *popt)
                    ss_res = np.sum((y_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["logarithmic"][agent_idx] = r_squared
                    coefficients_dict["logarithmic"].append(popt)
                except RuntimeError:
                    r_squared_dict["logarithmic"][agent_idx] = np.nan
                    coefficients_dict["logarithmic"].append((np.nan, np.nan))

                # Model 4: Quadratic
                try:
                    popt, _ = curve_fit(quadratic_model, X_valid, y_valid, maxfev=10000)
                    y_pred = quadratic_model(X_valid, *popt)
                    ss_res = np.sum((y_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["quadratic"][agent_idx] = r_squared
                    coefficients_dict["quadratic"].append(popt)
                except RuntimeError:
                    r_squared_dict["quadratic"][agent_idx] = np.nan
                    coefficients_dict["quadratic"].append((np.nan, np.nan))

                # Model 5: Power
                try:
                    popt, _ = curve_fit(power_model, X_valid, y_valid, maxfev=10000)
                    y_pred = power_model(X_valid, *popt)
                    ss_res = np.sum((y_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["power"][agent_idx] = r_squared
                    coefficients_dict["power"].append(popt)
                except RuntimeError:
                    r_squared_dict["power"][agent_idx] = np.nan
                    coefficients_dict["power"].append((np.nan, np.nan))
            else:
                # Not enough data points
                for model in model_names:
                    r_squared_dict[model][agent_idx] = np.nan
                    coefficients_dict[model].append(None)

            # Plotting code for this agent

            # Create a new figure
            plt.figure(figsize=(10, 6))

            # Plot the data points
            plt.scatter(X_valid, y_valid, label="Data", color="black")

            # Generate x values for plotting fitted curves
            x_min = np.min(X_valid)
            x_max = np.max(X_valid)
            x_plot = np.linspace(x_min, x_max, 100)

            # Plot each fitted model with R² in the label
            for model in model_names:
                coeffs = coefficients_dict[model][agent_idx]
                r_squared = r_squared_dict[model][agent_idx]

                if (
                    coeffs is not None
                    and not any([np.isnan(c) for c in np.atleast_1d(coeffs)])
                    and not np.isnan(r_squared)
                ):
                    # Depending on the model, compute y values for plotting
                    if model == "linear":
                        a, b = coeffs
                        x_plot_log = np.log10(x_plot[x_plot > 0])
                        if len(x_plot_log) > 0:
                            y_plot = linear_model(x_plot_log, a, b)
                            plt.plot(
                                x_plot[x_plot > 0],
                                y_plot,
                                label=f"{model} (R²={r_squared:.3f})",
                                linewidth=2,
                            )
                    elif model == "exponential":
                        y_plot = exponential_model(x_plot, *coeffs)
                        plt.plot(
                            x_plot,
                            y_plot,
                            label=f"{model} (R²={r_squared:.3f})",
                            linewidth=2,
                        )
                    elif model == "logarithmic":
                        x_plot_positive = x_plot[x_plot > 0]
                        if len(x_plot_positive) > 0:
                            y_plot = logarithmic_model(x_plot_positive, *coeffs)
                            plt.plot(
                                x_plot_positive,
                                y_plot,
                                label=f"{model} (R²={r_squared:.3f})",
                                linewidth=2,
                            )
                    elif model == "quadratic":
                        y_plot = quadratic_model(x_plot, *coeffs)
                        plt.plot(
                            x_plot,
                            y_plot,
                            label=f"{model} (R²={r_squared:.3f})",
                            linewidth=2,
                        )
                    elif model == "power":
                        x_plot_positive = x_plot[x_plot > 0]
                        if len(x_plot_positive) > 0:
                            y_plot = power_model(x_plot_positive, *coeffs)
                            plt.plot(
                                x_plot_positive,
                                y_plot,
                                label=f"{model} (R²={r_squared:.3f})",
                                linewidth=2,
                            )
                else:
                    continue  # Skip models with invalid coefficients or R²

            # Add labels and legend
            plt.xlabel("SPEI Probability")
            plt.ylabel("Yield Ratio")
            plt.title(
                f"Agent {agent_idx}, irr class {self.farmer_class[agent_idx]}, crop {self.crop_calendar[agent_idx, 0, 0]} "
            )
            plt.legend()
            plt.grid(True)

            # Save the plot to a file
            filename = os.path.join(output_folder, f"agent_{agent_idx}.png")
            plt.savefig(filename)
            plt.close()

        # Compute median R² for each model
        for model in model_names:
            valid_r2 = r_squared_dict[model][~np.isnan(r_squared_dict[model])]
            median_r2 = np.median(valid_r2) if len(valid_r2) > 0 else np.nan
            print(f"Median R² for {model}: {median_r2}")

    def calculate_yield_spei_relation_test_group(self):
        import os
        import matplotlib

        matplotlib.use("Agg")  # Use the 'Agg' backend for non-interactive plotting
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
        import numpy as np

        # Create unique groups based on agent properties
        crop_elevation_group = self.create_unique_groups()
        unique_crop_combinations, group_indices = np.unique(
            crop_elevation_group, axis=0, return_inverse=True
        )

        # Mask out empty rows (agents) where data is zero or NaN
        mask_agents = np.any(self.yearly_yield_ratio != 0, axis=1) & np.any(
            self.yearly_SPEI_probability != 0, axis=1
        )

        # Apply the mask to data and group indices
        masked_yearly_yield_ratio = self.yearly_yield_ratio[mask_agents, :]
        masked_SPEI_probability = self.yearly_SPEI_probability[mask_agents, :]
        group_indices = group_indices[mask_agents]

        # Number of groups
        n_groups = unique_crop_combinations.shape[0]

        # Define regression models
        def linear_model(X, a, b):
            return a * X + b

        def exponential_model(X, a, b):
            return a * np.exp(b * X)

        def logarithmic_model(X, a, b):
            return a * np.log(X) + b

        def quadratic_model(X, a, b, c):
            return a * X**2 + b * X + c

        def power_model(X, a, b):
            return a * X**b

        # Initialize dictionaries for coefficients and R² values
        model_names = ["linear", "exponential", "logarithmic", "quadratic", "power"]
        r_squared_dict = {model: np.full(n_groups, np.nan) for model in model_names}
        coefficients_dict = {model: [None] * n_groups for model in model_names}

        # Create a folder to save the plots
        output_folder = "plots/relation_test"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # For each group, perform regression with different models
        for group_idx in range(n_groups):
            # Get indices of agents in this group
            agent_indices = np.where(group_indices == group_idx)[0]

            if len(agent_indices) == 0:
                # No data for this group
                continue

            # Get data for the group
            y_data = masked_yearly_yield_ratio[
                agent_indices, :
            ]  # shape (num_agents_in_group, num_years)
            X_data = masked_SPEI_probability[agent_indices, :]  # same shape

            # Remove values where SPEI probability is greater than 1
            invalid_mask = X_data >= 1
            y_data[invalid_mask] = np.nan
            X_data[invalid_mask] = np.nan

            # Compute mean over agents in the group (axis=0 corresponds to years)
            y_group = np.nanmean(y_data, axis=0)  # shape (num_years,)
            X_group = np.nanmean(X_data, axis=0)  # same shape

            # Remove any years with NaN values
            valid_indices = (~np.isnan(y_group)) & (~np.isnan(X_group)) & (X_group > 0)
            y_group_valid = y_group[valid_indices]
            X_group_valid = X_group[valid_indices]

            if len(X_group_valid) >= 2:
                # Prepare data
                X_group_log = np.log10(X_group_valid)

                # Model 1: Linear in log-transformed X
                try:
                    popt, _ = curve_fit(
                        linear_model, X_group_log, y_group_valid, maxfev=10000
                    )
                    a, b = popt
                    y_pred = linear_model(X_group_log, a, b)
                    ss_res = np.sum((y_group_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_group_valid - np.mean(y_group_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["linear"][group_idx] = r_squared
                    coefficients_dict["linear"][group_idx] = (a, b)
                except (RuntimeError, ValueError):
                    pass  # Keep NaN in R² and None in coefficients

                # Model 2: Exponential
                try:
                    popt, _ = curve_fit(
                        exponential_model, X_group_valid, y_group_valid, maxfev=10000
                    )
                    y_pred = exponential_model(X_group_valid, *popt)
                    ss_res = np.sum((y_group_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_group_valid - np.mean(y_group_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["exponential"][group_idx] = r_squared
                    coefficients_dict["exponential"][group_idx] = popt
                except (RuntimeError, ValueError):
                    pass

                # Model 3: Logarithmic
                try:
                    popt, _ = curve_fit(
                        logarithmic_model, X_group_valid, y_group_valid, maxfev=10000
                    )
                    y_pred = logarithmic_model(X_group_valid, *popt)
                    ss_res = np.sum((y_group_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_group_valid - np.mean(y_group_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["logarithmic"][group_idx] = r_squared
                    coefficients_dict["logarithmic"][group_idx] = popt
                except (RuntimeError, ValueError):
                    pass

                # Model 4: Quadratic
                try:
                    popt, _ = curve_fit(
                        quadratic_model, X_group_valid, y_group_valid, maxfev=10000
                    )
                    y_pred = quadratic_model(X_group_valid, *popt)
                    ss_res = np.sum((y_group_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_group_valid - np.mean(y_group_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["quadratic"][group_idx] = r_squared
                    coefficients_dict["quadratic"][group_idx] = popt
                except (RuntimeError, ValueError):
                    pass

                # Model 5: Power
                try:
                    popt, _ = curve_fit(
                        power_model, X_group_valid, y_group_valid, maxfev=10000
                    )
                    y_pred = power_model(X_group_valid, *popt)
                    ss_res = np.sum((y_group_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_group_valid - np.mean(y_group_valid)) ** 2)
                    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r_squared_dict["power"][group_idx] = r_squared
                    coefficients_dict["power"][group_idx] = popt
                except (RuntimeError, ValueError):
                    pass

                # Plotting code for this group
                plt.figure(figsize=(10, 6))
                plt.scatter(X_group_valid, y_group_valid, label="Data", color="black")

                # Generate x values for plotting fitted curves
                x_min = np.min(X_group_valid)
                x_max = np.max(X_group_valid)
                x_plot = np.linspace(x_min, x_max, 100)

                for model in model_names:
                    coeffs = coefficients_dict[model][group_idx]
                    r_squared = r_squared_dict[model][group_idx]

                    if (
                        coeffs is not None
                        and not any([np.isnan(c) for c in np.atleast_1d(coeffs)])
                        and not np.isnan(r_squared)
                    ):
                        if model == "linear":
                            a, b = coeffs
                            x_plot_positive = x_plot[x_plot > 0]
                            x_plot_log = np.log10(x_plot_positive)
                            if len(x_plot_log) > 0:
                                y_plot = linear_model(x_plot_log, a, b)
                                plt.plot(
                                    x_plot_positive,
                                    y_plot,
                                    label=f"{model} (R²={r_squared:.3f})",
                                    linewidth=2,
                                )
                        elif model == "exponential":
                            y_plot = exponential_model(x_plot, *coeffs)
                            plt.plot(
                                x_plot,
                                y_plot,
                                label=f"{model} (R²={r_squared:.3f})",
                                linewidth=2,
                            )
                        elif model == "logarithmic":
                            x_plot_positive = x_plot[x_plot > 0]
                            if len(x_plot_positive) > 0:
                                y_plot = logarithmic_model(x_plot_positive, *coeffs)
                                plt.plot(
                                    x_plot_positive,
                                    y_plot,
                                    label=f"{model} (R²={r_squared:.3f})",
                                    linewidth=2,
                                )
                        elif model == "quadratic":
                            y_plot = quadratic_model(x_plot, *coeffs)
                            plt.plot(
                                x_plot,
                                y_plot,
                                label=f"{model} (R²={r_squared:.3f})",
                                linewidth=2,
                            )
                        elif model == "power":
                            x_plot_positive = x_plot[x_plot > 0]
                            if len(x_plot_positive) > 0:
                                y_plot = power_model(x_plot_positive, *coeffs)
                                plt.plot(
                                    x_plot_positive,
                                    y_plot,
                                    label=f"{model} (R²={r_squared:.3f})",
                                    linewidth=2,
                                )
                # Add labels and legend
                plt.xlabel("SPEI Probability")
                plt.ylabel("Yield Ratio")
                plt.title(f"Group {group_idx}")
                plt.legend()
                plt.grid(True)

                # Save the plot to a file
                filename = os.path.join(output_folder, f"group_{group_idx}.png")
                plt.savefig(filename)
                plt.close()
            else:
                # Not enough data points for this group
                continue

        # Compute median R² for each model across all groups
        for model in model_names:
            valid_r2 = r_squared_dict[model][~np.isnan(r_squared_dict[model])]
            median_r2 = np.median(valid_r2) if len(valid_r2) > 0 else np.nan
            print(f"Median R² for {model}: {median_r2}")

        # Assign relations to agents based on their group
        # Here, we'll choose the model with the highest median R²
        # Alternatively, you can select the best model per group
        # For simplicity, we'll assign the linear model coefficients to agents

        # Example: Assign linear model coefficients to agents
        a_array = np.full(len(group_indices), np.nan)
        b_array = np.full(len(group_indices), np.nan)

        for group_idx in range(n_groups):
            if coefficients_dict["linear"][group_idx] is not None:
                a, b = coefficients_dict["linear"][group_idx]
                agent_mask = group_indices == group_idx
                a_array[agent_mask] = a
                b_array[agent_mask] = b

        # Assign to agents
        self.farmer_yield_probability_relation = np.column_stack((a_array, b_array))

        # Print overall best-fitting model based on median R²
        median_r2_values = {
            model: np.nanmedian(r_squared_dict[model]) for model in model_names
        }
        best_model_overall = max(median_r2_values, key=median_r2_values.get)
        print(f"Best-fitting model overall: {best_model_overall}")

    def calculate_yield_spei_relation(self):
        # Number of agents
        n_agents = self.yearly_yield_ratio.shape[0]

        # Initialize arrays for coefficients and R²
        a_array = np.zeros(n_agents)
        b_array = np.zeros(n_agents)
        r_squared_array = np.zeros(n_agents)

        # Loop over each agent
        for agent_idx in range(n_agents):
            # Get data for the agent
            y_data = self.yearly_yield_ratio[agent_idx, :]
            X_data = self.yearly_SPEI_probability[agent_idx, :]

            # Log-transform X_data, handling zeros
            with np.errstate(divide="ignore"):
                X_data_log = np.log10(X_data)

            # Mask out zeros and NaNs
            valid_mask = (
                (~np.isnan(y_data))
                & (~np.isnan(X_data_log))
                & (y_data != 0)
                & (X_data != 0)
            )
            y_valid = y_data[valid_mask]
            X_valid = X_data_log[valid_mask]

            if len(X_valid) >= 2:
                # Prepare matrices
                X_matrix = np.vstack([X_valid, np.ones(len(X_valid))]).T
                # Perform linear regression
                coefficients = np.linalg.lstsq(X_matrix, y_valid, rcond=None)[0]
                a, b = coefficients

                # Calculate R²
                y_pred = a * X_valid + b
                ss_res = np.sum((y_valid - y_pred) ** 2)
                ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            else:
                # Not enough data points
                a, b, r_squared = np.nan, np.nan, np.nan

            a_array[agent_idx] = a
            b_array[agent_idx] = b
            r_squared_array[agent_idx] = r_squared

        # Assign relations to agents
        self.farmer_yield_probability_relation = np.column_stack((a_array, b_array))

        # Print median R²
        valid_r2 = r_squared_array[~np.isnan(r_squared_array)]
        print("Median R²:", np.median(valid_r2) if len(valid_r2) > 0 else "N/A")

    def calculate_yield_spei_relation_group(self):
        # Create unique groups
        crop_elevation_group = self.create_unique_groups()
        unique_crop_combinations, group_indices = np.unique(
            crop_elevation_group, axis=0, return_inverse=True
        )

        # Mask out empty columns
        mask_columns = np.any(self.yearly_yield_ratio != 0, axis=0) & np.any(
            self.yearly_SPEI_probability != 0, axis=0
        )

        masked_yearly_yield_ratio = self.yearly_yield_ratio[:, mask_columns]
        masked_SPEI_probability = self.yearly_SPEI_probability[:, mask_columns]
        masked_SPEI_probability_log = np.log10(masked_SPEI_probability)

        # Number of groups
        n_groups = unique_crop_combinations.shape[0]

        # Number of groups
        n_groups = unique_crop_combinations.shape[0]

        # Initialize arrays for coefficients and R²
        a_array = np.zeros(n_groups)
        b_array = np.zeros(n_groups)
        r_squared_array = np.zeros(n_groups)

        for group_idx in range(n_groups):
            col_indices = np.where(group_indices == group_idx)[0]

            # Get data for the group
            y_data = masked_yearly_yield_ratio[col_indices, :]
            X_data_log = masked_SPEI_probability_log[col_indices, :]
            X_data_prob = masked_SPEI_probability[col_indices, :]

            # Remove values where the probability is > 1
            mask = X_data_prob >= 1
            y_data[mask] = np.nan
            X_data_log[mask] = np.nan

            # Compute mean over columns (axis=1)
            y_group = np.nanmean(y_data, axis=1)
            X_group = np.nanmean(X_data_log, axis=1)

            # Remove any years with NaN values
            valid_indices = ~np.isnan(y_group) & ~np.isnan(X_group)
            y_group = y_group[valid_indices]
            X_group = X_group[valid_indices]

            if len(X_group) >= 2:
                # Prepare matrices
                X_matrix = np.vstack([X_group, np.ones(len(X_group))]).T
                # Perform linear regression
                coefficients = np.linalg.lstsq(X_matrix, y_group, rcond=None)[0]
                a, b = coefficients

                # Calculate R²
                y_pred = a * X_group + b
                ss_res = np.sum((y_group - y_pred) ** 2)
                ss_tot = np.sum((y_group - np.mean(y_group)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            else:
                # Not enough data points
                a, b, r_squared = np.nan, np.nan, np.nan

            a_array[group_idx] = a
            b_array[group_idx] = b
            r_squared_array[group_idx] = r_squared

        # Assign relations to agents
        self.farmer_yield_probability_relation = np.column_stack(
            (a_array[group_indices], b_array[group_indices])
        )

        # Print median R²
        valid_r2 = r_squared_array[~np.isnan(r_squared_array)]
        print("Median R²:", np.median(valid_r2) if len(valid_r2) > 0 else "N/A")

    def adapt_crops(self) -> None:
        # Fetch loan configuration
        loan_duration = 2
        adaptation_type = 2

        # For now there are no differences in cost, so set the annual cost per farmer for crops as annual costs
        # Update to also be able to include the annual cost of the possible new crop
        # annual_cost = np.sum(self.all_loans_annual_cost[:, :, 0], axis=1)
        annual_cost = np.zeros(
            self.n
        )  # As both varieties cost the same, for now its 0 as the cost is already in the prior nnual costs
        crops_with_varieties = [0, 26, 27, 1, 28, 29, 2, 30, 31, 7, 32, 33, 9, 34, 35]

        extra_constraint = np.any(
            np.isin(self.crop_calendar[:, :, 0], crops_with_varieties), axis=1
        )

        (
            total_profits,
            profits_no_event,
            total_profits_adaptation_option_1,
            profits_no_event_adaptation_option_1,
            total_profits_adaptation_option_2,
            profits_no_event_adaptation_option_2,
            new_crop_nr,
            new_farmer_id,
        ) = self.profits_SEUT(adaptation_type)

        total_annual_costs_m2 = (
            self.all_loans_annual_cost[:, -1, 0] / self.field_size_per_farmer
        )

        # Solely the annual cost of the adaptation
        annual_cost_m2 = annual_cost

        # Construct a dictionary of parameters to pass to the decision module functions
        decision_params = {
            "loan_duration": loan_duration,
            "expenditure_cap": self.expenditure_cap,
            "n_agents": self.n,
            "sigma": self.risk_aversion.data,
            "p_droughts": 1 / self.p_droughts[:-1],
            "profits_no_event": profits_no_event,
            "total_profits": total_profits,
            "risk_perception": self.risk_perception.data,
            "total_annual_costs": total_annual_costs_m2,
            "adaptation_costs": annual_cost_m2,
            "adapted": None,
            "time_adapted": np.full(self.n, 1),
            "T": np.full(self.n, 2),
            "discount_rate": self.discount_rate.data,
            "extra_constraint": extra_constraint,
        }

        decision_params_1 = copy.deepcopy(decision_params)
        decision_params_1.update(
            {
                "total_profits": total_profits_adaptation_option_1,
                "profits_no_event": profits_no_event_adaptation_option_1,
            }
        )

        decision_params_2 = copy.deepcopy(decision_params)
        decision_params_2.update(
            {
                "total_profits": total_profits_adaptation_option_2,
                "profits_no_event": profits_no_event_adaptation_option_2,
            }
        )
        # Calculate the EU of not adapting and adapting respectively
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing(**decision_params)
        SEUT_adapt_option_1 = self.decision_module.calcEU_do_nothing(
            **decision_params_1
        )
        SEUT_adapt_option_2 = self.decision_module.calcEU_do_nothing(
            **decision_params_2
        )

        assert (
            (SEUT_do_nothing != -1).any
            or (SEUT_adapt_option_1 != -1).any()
            or (SEUT_adapt_option_2 != -1).any()
        )

        # Determine the best adaptation option
        best_option_SEUT = np.maximum(SEUT_adapt_option_1, SEUT_adapt_option_2)
        chosen_option = (SEUT_adapt_option_1 >= SEUT_adapt_option_2).astype(int)

        # Determine for which agents it is beneficial to switch crops
        SEUT_adaptation_decision = best_option_SEUT > SEUT_do_nothing

        # Determine whether the chosen option is the first or second option
        new_crop_nr_temp = np.where(
            chosen_option == 0,
            new_crop_nr[:, 0],
            new_crop_nr[:, 1],
        )

        # Adjust the intention threshold based on whether neighbors already have similar crop
        # Check for each farmer which crops their neighbors are cultivating
        social_network_crops = self.crop_calendar[self.social_network, 0, 0]

        # Check whether adapting agents have adaptation type in their network and create mask
        network_has_crop = np.any(
            social_network_crops == new_crop_nr_temp[:, None],
            axis=1,
        )

        # Increase intention factor if someone in network has crop
        intention_factor_adjusted = self.intention_factor.copy()
        intention_factor_adjusted[network_has_crop] += 0.3

        # Determine whether it passed the intention threshold
        random_values = np.random.rand(*intention_factor_adjusted.shape)
        intention_mask = random_values < intention_factor_adjusted

        # Set the adaptation mask
        SEUT_adaptation_decision = SEUT_adaptation_decision & intention_mask
        final_chosen_option = chosen_option[SEUT_adaptation_decision]

        print("Crop switching farmers", np.count_nonzero(SEUT_adaptation_decision))

        # Make final selection of which crops the agents will switch to
        new_crop_nr_final = np.where(
            final_chosen_option == 0,
            new_crop_nr[SEUT_adaptation_decision, 0],
            new_crop_nr[SEUT_adaptation_decision, 1],
        )

        # Select farmer id from which they will copy the yield/spei relation
        new_farmer_id = np.where(
            final_chosen_option == 0,
            new_farmer_id[SEUT_adaptation_decision, 0],
            new_farmer_id[SEUT_adaptation_decision, 1],
        )

        assert not np.any(new_crop_nr_final == -1)

        # # Assuming self.crop_calendar and extra_constraint are defined
        # unique_values_old, counts_old = np.unique(
        #     self.crop_calendar[extra_constraint, 0, 0], return_counts=True
        # )

        # # Create a formatted string for the old distribution
        # old_distribution = ", ".join(
        #     f"{val}: {cnt}" for val, cnt in zip(unique_values_old, counts_old)
        # )

        # # Print the old distribution
        # print("old distribution", "\n", old_distribution)

        # Switch their crops and update their yield-SPEI relation
        self.crop_calendar[SEUT_adaptation_decision, 0, 0] = new_crop_nr_final

        # unique_values, counts = np.unique(
        #     self.crop_calendar[extra_constraint, 0, 0], return_counts=True
        # )

        # # Calculate the difference in counts
        # difference_counts = counts - counts_old

        # # Create a formatted string for the difference
        # difference_distribution = ", ".join(
        #     f"{val}: {cnt}" for val, cnt in zip(unique_values_old, difference_counts)
        # )

        # # Print the difference
        # print("difference", "\n", difference_distribution)

        # Update yield-SPEI relation
        self.yearly_yield_ratio[SEUT_adaptation_decision, :] = self.yearly_yield_ratio[
            new_farmer_id, :
        ]
        self.yearly_SPEI_probability[SEUT_adaptation_decision, :] = (
            self.yearly_SPEI_probability[new_farmer_id, :]
        )

    def adapt_irrigation_well(
        self, average_extraction_speed, energy_cost, water_cost
    ) -> None:
        """
        Handle the adaptation of farmers to irrigation wells.

        This function checks which farmers will adopt irrigation wells based on their expected utility
        and the provided constraints. It calculates the costs and the benefits for each farmer and updates
        their statuses (e.g., irrigation source, adaptation costs) accordingly.

        Note:

        TODO:
            - Possibly externalize hard-coded values.
        """
        # Constants
        adaptation_type = 1

        groundwater_depth = self.groundwater_depth
        groundwater_depth[groundwater_depth < 0] = 0

        annual_cost, well_depth = self.calculate_well_costs_global(
            groundwater_depth, average_extraction_speed
        )

        # Compute the total annual per square meter costs if farmers adapt during this cycle
        # This cost is the cost if the farmer would adapt, plus its current costs of previous
        # adaptations

        total_annual_costs_m2 = (
            annual_cost + self.all_loans_annual_cost[:, -1, 0]
        ) / self.field_size_per_farmer

        # Solely the annual cost of the adaptation
        annual_cost_m2 = annual_cost / self.field_size_per_farmer

        loan_duration = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]["loan_duration"]

        # Reset farmers' status and irrigation type who exceeded the lifespan of their adaptation
        # and who's wells are much shallower than the groundwater depth
        expired_adaptations = (
            self.time_adapted[:, adaptation_type] == self.lifespan
        ) | (groundwater_depth > self.well_depth)
        self.adaptation_mechanism[expired_adaptations, adaptation_type] = 0
        self.adapted[expired_adaptations, adaptation_type] = 0
        self.time_adapted[expired_adaptations, adaptation_type] = -1
        self.irrigation_source[expired_adaptations] = self.irrigation_source_key["no"]

        # Define extra constraints (farmers' wells must reach groundwater)
        well_reaches_groundwater = self.well_depth > groundwater_depth
        extra_constraint = well_reaches_groundwater

        # To determine the benefit of irrigation, those who have a well are adapted
        adapted = np.where((self.adapted[:, 1] == 1), 1, 0)

        (
            energy_diff,
            water_diff,
        ) = self.adaptation_water_cost_difference(adapted, energy_cost, water_cost)

        (
            total_profits,
            profits_no_event,
            total_profits_adaptation,
            profits_no_event_adaptation,
        ) = self.profits_SEUT(adaptation_type, adapted)

        total_profits_adaptation = total_profits_adaptation + energy_diff + water_diff
        profits_no_event_adaptation = (
            profits_no_event_adaptation + energy_diff + water_diff
        )

        # Construct a dictionary of parameters to pass to the decision module functions
        decision_params = {
            "loan_duration": loan_duration,
            "expenditure_cap": self.expenditure_cap,
            "n_agents": self.n,
            "sigma": self.risk_aversion.data,
            "p_droughts": 1 / self.p_droughts[:-1],
            "total_profits_adaptation": total_profits_adaptation,
            "profits_no_event": profits_no_event,
            "profits_no_event_adaptation": profits_no_event_adaptation,
            "total_profits": total_profits,
            "risk_perception": self.risk_perception.data,
            "total_annual_costs": total_annual_costs_m2,
            "adaptation_costs": annual_cost_m2,
            "adapted": adapted,
            "time_adapted": self.time_adapted[:, adaptation_type],
            "T": np.full(
                self.n,
                self.model.config["agent_settings"]["farmers"]["expected_utility"][
                    "adaptation_well"
                ]["decision_horizon"],
            ),
            "discount_rate": self.discount_rate.data,
            "extra_constraint": extra_constraint,
        }

        # Calculate the EU of not adapting and adapting respectively
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing(**decision_params)

        SEUT_adapt = self.decision_module.calcEU_adapt(**decision_params)

        assert (SEUT_do_nothing != -1).any or (SEUT_adapt != -1).any()

        # Compare EU values for those who haven't adapted yet and get boolean results
        SEUT_adaptation_decision = SEUT_adapt > SEUT_do_nothing

        # Adjust the intention threshold based on whether neighbors already have similar crop
        # Check for each farmer which crops their neighbors are cultivating
        social_network_wells = adapted[self.social_network]

        # Check whether adapting agents have adaptation type in their network and create mask
        network_has_crop = np.any(social_network_wells == 1, axis=1)

        # Increase intention factor if someone in network has crop
        intention_factor_adjusted = self.intention_factor.copy()
        intention_factor_adjusted[network_has_crop] += 0.3

        # Determine whether it passed the intention threshold
        random_values = np.random.rand(*intention_factor_adjusted.shape)
        intention_mask = random_values < intention_factor_adjusted

        SEUT_adaptation_decision = SEUT_adaptation_decision & intention_mask

        # Update the adaptation status
        self.adapted[SEUT_adaptation_decision, adaptation_type] = 1

        # Reset the timer for newly adapting farmers and update timers for others
        self.time_adapted[SEUT_adaptation_decision, adaptation_type] = 0
        self.time_adapted[
            self.time_adapted[:, adaptation_type] != -1, adaptation_type
        ] += 1

        # Update irrigation source for farmers who adapted
        self.irrigation_source[SEUT_adaptation_decision] = self.irrigation_source_key[
            "well"
        ]

        # Set their well depth
        self.well_depth[SEUT_adaptation_decision] = well_depth[SEUT_adaptation_decision]

        # Update annual costs and disposable income for adapted farmers
        self.all_loans_annual_cost[
            SEUT_adaptation_decision, adaptation_type + 1, 0
        ] += annual_cost[SEUT_adaptation_decision]  # For wells specifically
        self.all_loans_annual_cost[SEUT_adaptation_decision, -1, 0] += annual_cost[
            SEUT_adaptation_decision
        ]  # Total loan amount

        # set loan tracker
        self.loan_tracker[SEUT_adaptation_decision, adaptation_type + 1, 0] += (
            loan_duration
        )

        # Print the percentage of adapted households
        percentage_adapted = round(
            np.sum(self.adapted[:, adaptation_type])
            / len(self.adapted[:, adaptation_type])
            * 100,
            2,
        )
        print("Irrigation well farms:", percentage_adapted, "(%)")

    def adapt_irrigation_efficiency(self, energy_cost, water_cost) -> None:
        """
        Handle the adaptation of farmers to irrigation wells.

        This function checks which farmers will adopt irrigation wells based on their expected utility
        and the provided constraints. It calculates the costs and the benefits for each farmer and updates
        their statuses (e.g., irrigation source, adaptation costs) accordingly.

        Note:

        TODO:
            - Possibly externalize hard-coded values.
        """
        # Constants
        adaptation_type = 2

        loan_duration = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_sprinkler"]["loan_duration"]

        # Placeholder
        costs_irrigation_system = 4 * self.field_size_per_farmer

        annual_cost = costs_irrigation_system * (
            self.interest_rate
            * (1 + self.interest_rate) ** loan_duration
            / ((1 + self.interest_rate) ** loan_duration - 1)
        )

        # Compute the total annual per square meter costs if farmers adapt during this cycle
        # This cost is the cost if the farmer would adapt, plus its current costs of previous
        # adaptations

        total_annual_costs_m2 = (
            annual_cost + self.all_loans_annual_cost[:, -1, 0]
        ) / self.field_size_per_farmer

        # Solely the annual cost of the adaptation
        annual_cost_m2 = annual_cost / self.field_size_per_farmer

        # Reset farmers' status and irrigation type who exceeded the lifespan of their adaptation
        # and who's wells are much shallower than the groundwater depth
        expired_adaptations = self.time_adapted[:, adaptation_type] == self.lifespan
        self.adaptation_mechanism[expired_adaptations, adaptation_type] = 0
        self.adapted[expired_adaptations, adaptation_type] = 0
        self.time_adapted[expired_adaptations, adaptation_type] = -1

        extra_constraint = np.full(self.n, 1, dtype=bool)

        # To determine the benefit of irrigation, those who have a well are adapted
        adapted = np.where((self.adapted[:, adaptation_type] == 1), 1, 0)

        (
            energy_diff,
            water_diff,
        ) = self.adaptation_water_cost_difference(adapted, energy_cost, water_cost)

        (
            total_profits,
            profits_no_event,
        ) = self.profits_SEUT(0, adapted)

        total_profits_adaptation = total_profits + energy_diff + water_diff
        profits_no_event_adaptation = profits_no_event + energy_diff + water_diff

        # Construct a dictionary of parameters to pass to the decision module functions
        decision_params = {
            "loan_duration": loan_duration,
            "expenditure_cap": self.expenditure_cap,
            "n_agents": self.n,
            "sigma": self.risk_aversion.data,
            "p_droughts": 1 / self.p_droughts[:-1],
            "total_profits_adaptation": total_profits_adaptation,
            "profits_no_event": profits_no_event,
            "profits_no_event_adaptation": profits_no_event_adaptation,
            "total_profits": total_profits,
            "risk_perception": self.risk_perception.data,
            "total_annual_costs": total_annual_costs_m2,
            "adaptation_costs": annual_cost_m2,
            "adapted": adapted,
            "time_adapted": self.time_adapted[:, adaptation_type],
            "T": np.full(
                self.n,
                self.model.config["agent_settings"]["farmers"]["expected_utility"][
                    "adaptation_sprinkler"
                ]["decision_horizon"],
            ),
            "discount_rate": self.discount_rate.data,
            "extra_constraint": extra_constraint,
        }

        # Calculate the EU of not adapting and adapting respectively
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing(**decision_params)
        SEUT_adapt = self.decision_module.calcEU_adapt(**decision_params)

        assert (SEUT_do_nothing != -1).any or (SEUT_adapt != -1).any()

        # Compare EU values for those who haven't adapted yet and get boolean results
        SEUT_adaptation_decision = SEUT_adapt > SEUT_do_nothing

        social_network_adaptation = adapted[self.social_network]

        # Check whether adapting agents have adaptation type in their network and create mask
        network_has_adaptation = np.any(social_network_adaptation == 1, axis=1)

        # Increase intention factor if someone in network has crop
        intention_factor_adjusted = self.intention_factor.copy()
        intention_factor_adjusted[network_has_adaptation] += 0.3

        # Determine whether it passed the intention threshold
        random_values = np.random.rand(*intention_factor_adjusted.shape)
        intention_mask = random_values < intention_factor_adjusted

        SEUT_adaptation_decision = SEUT_adaptation_decision & intention_mask

        # Update the adaptation status
        self.adapted[SEUT_adaptation_decision, adaptation_type] = 1

        # Reset the timer for newly adapting farmers and update timers for others
        self.time_adapted[SEUT_adaptation_decision, adaptation_type] = 0
        self.time_adapted[
            self.time_adapted[:, adaptation_type] != -1, adaptation_type
        ] += 1

        # Reset the timer for newly adapting farmers and update timers for others
        self.time_adapted[SEUT_adaptation_decision, adaptation_type] = 0
        self.time_adapted[
            self.time_adapted[:, adaptation_type] != -1, adaptation_type
        ] += 1

        # Update irrigation efficiency for farmers who adapted
        self.irrigation_efficiency[SEUT_adaptation_decision] = 0.9

        # Update annual costs and disposable income for adapted farmers
        self.all_loans_annual_cost[
            SEUT_adaptation_decision, adaptation_type + 1, 0
        ] += annual_cost[SEUT_adaptation_decision]  # For wells specifically
        self.all_loans_annual_cost[SEUT_adaptation_decision, -1, 0] += annual_cost[
            SEUT_adaptation_decision
        ]  # Total loan amount

        # set loan tracker
        self.loan_tracker[SEUT_adaptation_decision, adaptation_type + 1, 0] += (
            loan_duration
        )

        # Print the percentage of adapted households
        percentage_adapted = round(
            np.sum(self.adapted[:, adaptation_type])
            / len(self.adapted[:, adaptation_type])
            * 100,
            2,
        )
        print("Irrigation efficient farms:", percentage_adapted, "(%)")

    def calculate_water_costs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the water and energy costs per agent and the average extraction speed.

        This method computes the energy costs for agents using groundwater, the water costs for all agents
        depending on their water source, and the average extraction speed per agent. It also updates the
        loans and annual costs associated with water and energy use.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - energy_costs (np.ndarray): Energy costs per agent (LCU/year).
                - water_costs (np.ndarray): Water costs per agent (LCU/year).
                - average_extraction_speed (np.ndarray): Average water extraction speed per agent (m³/s).
        """
        # Get electricity costs per agent based on their region and current time
        electricity_costs = np.full(
            self.n,
            self.get_value_per_farmer_from_region_id(
                self.electricity_cost, self.model.current_time
            ),
            dtype=np.float32,
        )

        # Initialize energy and water costs arrays
        energy_costs = np.zeros(self.n, dtype=np.float32)
        water_costs = np.zeros(self.n, dtype=np.float32)

        # Compute total pump duration per agent (average over crops)
        total_pump_duration = np.mean(self.total_crop_age, axis=1)

        # Get groundwater depth per agent and ensure non-negative values
        groundwater_depth = self.groundwater_depth.copy()
        groundwater_depth[groundwater_depth < 0] = 0

        # Create unique groups based on crop combinations and elevation
        crop_elevation_group = self.create_unique_groups()
        unique_crop_groups = np.unique(crop_elevation_group, axis=0)

        # Initialize array to store average extraction per unique group
        average_extraction_array = np.full(len(unique_crop_groups), 0, dtype=np.float32)

        # Compute yearly water abstraction per m² per agent
        yearly_abstraction_m3_per_m2 = (
            self.yearly_abstraction_m3_by_farmer
            / self.field_size_per_farmer[..., None, None]
        )

        # Loop over each unique crop group to compute average extraction
        for idx, crop_combination in enumerate(unique_crop_groups):
            # Find agents belonging to the current unique crop group
            unique_farmer_group = np.where(
                (crop_elevation_group == crop_combination[None, ...]).all(axis=1)
            )[0]

            # Extract the abstraction values for the group, excluding zeros
            extraction_values = yearly_abstraction_m3_per_m2[unique_farmer_group, :3, :]
            non_zero_extractions = extraction_values[extraction_values != 0]

            # Compute average extraction for the group if there are non-zero values
            if non_zero_extractions.size > 0:
                average_extraction = np.mean(non_zero_extractions)  # m³ per m² per year
            else:
                average_extraction = 0.0

            # If the farmer is not rainfed (farmer_class != 3), store the average extraction
            if crop_combination[-1] != 3:
                average_extraction_array[idx] = average_extraction

        # Map each agent to their corresponding average extraction value
        positions_agent = np.array(
            [
                np.where((unique_crop_groups == group).all(axis=1))[0][0]
                for group in crop_elevation_group
            ]
        )
        average_extraction_m2 = average_extraction_array[positions_agent]

        # Compute average extraction per agent (m³/year)
        average_extraction = average_extraction_m2 * self.field_size_per_farmer

        # Compute average extraction speed per agent (m³/s)
        average_extraction_speed = (
            average_extraction / 365 / self.pump_hours / 3600
        )  # Convert from m³/year to m³/s

        # Create boolean masks for different types of water sources
        mask_channel = self.farmer_class == 0
        mask_reservoir = self.farmer_class == 1
        mask_groundwater = self.farmer_class == 2

        # Compute power required for groundwater extraction per agent (kW)
        power = (
            self.specific_weight_water
            * groundwater_depth[mask_groundwater]
            * average_extraction_speed[mask_groundwater]
            / self.pump_efficiency
        ) / 1000  # Convert from W to kW

        # Compute energy consumption per agent (kWh/year)
        energy = power * (total_pump_duration[mask_groundwater] * self.pump_hours)

        # Get energy cost rate per agent (LCU per kWh)
        energy_cost_rate = electricity_costs[mask_groundwater]

        # Compute energy costs per agent (LCU/year)
        energy_costs[mask_groundwater] = energy * energy_cost_rate

        # Compute water costs for agents using channel water (LCU/year)
        water_costs[mask_channel] = (
            average_extraction[mask_channel] * self.water_costs_m3_channel
        )

        # Compute water costs for agents using reservoir water (LCU/year)
        water_costs[mask_reservoir] = (
            average_extraction[mask_reservoir] * self.water_costs_m3_reservoir
        )

        # Compute water costs for agents using groundwater (LCU/year)
        water_costs[mask_groundwater] = (
            average_extraction[mask_groundwater] * self.water_costs_m3_groundwater
        )

        # Assume minimal interest rate as farmers pay directly
        interest_rate_farmer = 0.0001  # Annual interest rate
        loan_duration = 2  # Loan duration in years

        # Compute annual cost of water and energy using annuity formula
        # A = P * [r(1+r)^n] / [(1+r)^n -1], where P is principal, r is interest rate, n is loan duration
        annuity_factor = (
            interest_rate_farmer
            * (1 + interest_rate_farmer) ** loan_duration
            / ((1 + interest_rate_farmer) ** loan_duration - 1)
        )
        annual_cost_water_energy = (water_costs + energy_costs) * annuity_factor

        # Update loan records with the annual cost of water and energy
        for i in range(4):
            # Find the first available loan slot
            if np.all(self.all_loans_annual_cost.data[:, 4, i] == 0):
                self.all_loans_annual_cost.data[:, 4, i] = annual_cost_water_energy
                self.loan_tracker[annual_cost_water_energy > 0, 4, i] = loan_duration
                break

        # Add the annual cost to the total loan annual costs
        self.all_loans_annual_cost.data[:, -1, 0] += annual_cost_water_energy

        return energy_costs, water_costs, average_extraction_speed

    def calculate_well_costs_global(
        self, groundwater_depth: np.ndarray, average_extraction_speed: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the annual costs associated with well installation and operation globally.

        This function computes the annual costs for installing wells, maintaining them, and the energy costs
        associated with pumping groundwater for each agent (farmer). It takes into account regional variations
        in costs and agent-specific parameters such as groundwater depth and extraction speed.

        Parameters:
            groundwater_depth (np.ndarray): Array of groundwater depths per agent (in meters).
            average_extraction_speed (np.ndarray): Array of average water extraction speeds per agent (m³/s).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - annual_cost (np.ndarray): Annual cost per agent (local currency units per year).
                - potential_well_length (np.ndarray): Potential well length per agent (in meters).
        """

        # Retrieve aquifer-specific unit costs for well drilling per meter
        well_cost_class_1 = self.get_value_per_farmer_from_region_id(
            self.why_10, self.model.current_time
        )
        well_cost_class_2 = self.get_value_per_farmer_from_region_id(
            self.why_20, self.model.current_time
        )
        well_cost_class_3 = self.get_value_per_farmer_from_region_id(
            self.why_30, self.model.current_time
        )

        # Initialize the well unit cost array with zeros
        well_unit_cost = np.zeros_like(self.why_class, dtype=np.float32)

        # Assign unit costs to each agent based on their well class using boolean indexing
        well_unit_cost[self.why_class == 1] = well_cost_class_1[self.why_class == 1]
        well_unit_cost[self.why_class == 2] = well_cost_class_2[self.why_class == 2]
        well_unit_cost[self.why_class == 3] = well_cost_class_3[self.why_class == 3]

        # Get electricity costs per agent based on their region and current time
        electricity_costs = self.get_value_per_farmer_from_region_id(
            self.electricity_cost, self.model.current_time
        )

        # Calculate potential well length per agent
        # Potential well length is the sum of the maximum initial saturated thickness and the groundwater depth
        potential_well_length = self.max_initial_sat_thickness + groundwater_depth

        # Calculate the installation cost per agent (cost per meter * potential well length)
        install_cost = well_unit_cost * potential_well_length

        # Calculate maintenance cost per agent (as a fraction of the installation cost)
        maintenance_cost = self.maintenance_factor * install_cost

        # Calculate the total pump duration per agent (average over crops)
        total_pump_duration = np.mean(self.total_crop_age, axis=1)  # days

        # Calculate the power required per agent for pumping groundwater (in kilowatts)
        # specific_weight_water (N/m³), groundwater_depth (m), average_extraction_speed (m³/s), pump_efficiency (%)
        power = (
            self.specific_weight_water
            * groundwater_depth
            * average_extraction_speed
            / self.pump_efficiency
        ) / 1000  # Convert from watts to kilowatts

        # Calculate the energy consumption per agent (in kilowatt-hours)
        # power (kW), total_pump_duration (days), pump_hours (hours per day)
        energy = power * (total_pump_duration * self.pump_hours)

        # Get energy cost rate per agent (local currency units per kilowatt-hour)
        energy_cost_rate = electricity_costs

        # Calculate the energy cost per agent (local currency units per year)
        energy_cost = energy * energy_cost_rate

        # Fetch loan configuration for well installation
        loan_config = self.model.config["agent_settings"]["farmers"][
            "expected_utility"
        ]["adaptation_well"]
        loan_duration = loan_config["loan_duration"]  # Loan duration in years

        # Calculate annuity factor for loan repayment using the annuity formula
        # A = P * [r(1+r)^n] / [(1+r)^n -1], where:
        # A = annual payment, P = principal amount (install_cost), r = interest rate, n = loan duration
        interest_rate = self.interest_rate
        n = loan_duration
        annuity_factor = (interest_rate * (1 + interest_rate) ** n) / (
            (1 + interest_rate) ** n - 1
        )

        # Calculate the annual cost per agent (local currency units per year)
        annual_cost = (install_cost * annuity_factor) + energy_cost + maintenance_cost

        return annual_cost, potential_well_length

    def profits_SEUT(
        self, adaptation_type: int, adapted: np.ndarray = None
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ]:
        """
        Calculate total profits under different drought probability scenarios, with and without adaptation measures.

        This method computes the profits for agents under various drought probabilities, considering the potential
        impacts of different adaptation strategies (e.g., installing wells or changing crops). It returns the profits
        both with and without adaptation, depending on the adaptation type specified.

        Parameters:
            adaptation_type (int): The type of adaptation to consider.
                - 0: No adaptation.
                - 1: Adaptation Type 1 (e.g., installing wells).
                - Other: Other adaptation types (e.g., changing crops).
            adapted (np.ndarray, optional): An array indicating which agents have adapted (relevant for adaptation_type == 1).

        Returns:
            Depending on adaptation_type, returns a tuple containing total profits and profits under 'no drought' scenario,
            possibly including adaptation profits and additional data.
        """

        def compute_adaptation_gains(yield_ratios: np.ndarray) -> Dict[str, Any]:
            """
            Compute adaptation gains based on the adaptation type.

            Parameters:
                yield_ratios (np.ndarray): Original yield ratios without adaptation.

            Returns:
                Dict[str, Any]: Dictionary containing adaptation gains and additional data.
            """
            adaptation_data = {}
            if adaptation_type == 1:
                # Adaptation Type 1: e.g., installing wells
                gains_adaptation = self.adaptation_yield_ratio_difference(
                    adapted, yield_ratios
                )
                adaptation_data["gains_adaptation"] = gains_adaptation
            else:
                # Other Adaptation Types: e.g., changing crops
                (
                    gains_option_1,
                    gains_option_2,
                    new_crop_nr,
                    new_farmer_id,
                ) = self.crop_yield_ratio_difference(yield_ratios)
                adaptation_data.update(
                    {
                        "gains_option_1": gains_option_1,
                        "gains_option_2": gains_option_2,
                        "new_crop_nr": new_crop_nr,
                        "new_farmer_id": new_farmer_id,
                    }
                )
            return adaptation_data

        def adjust_yield_ratios_with_adaptation(
            yield_ratios: np.ndarray, adaptation_data: Dict[str, Any]
        ) -> Dict[str, np.ndarray]:
            """
            Adjust yield ratios by adding adaptation gains and clipping values between 0 and 1.

            Parameters:
                yield_ratios (np.ndarray): Original yield ratios.
                adaptation_data (Dict[str, Any]): Adaptation gains data.

            Returns:
                Dict[str, np.ndarray]: Dictionary containing adjusted yield ratios.
            """
            adjusted_yield_ratios = {}
            if adaptation_type == 1:
                yield_ratios_adaptation = np.clip(
                    yield_ratios + adaptation_data["gains_adaptation"], 0, 1
                )
                adjusted_yield_ratios["yield_ratios_adaptation"] = (
                    yield_ratios_adaptation
                )
            else:
                yield_ratios_adaptation_option_1 = np.clip(
                    yield_ratios + adaptation_data["gains_option_1"], 0, 1
                )
                yield_ratios_adaptation_option_2 = np.clip(
                    yield_ratios + adaptation_data["gains_option_2"], 0, 1
                )
                adjusted_yield_ratios.update(
                    {
                        "yield_ratios_adaptation_option_1": yield_ratios_adaptation_option_1,
                        "yield_ratios_adaptation_option_2": yield_ratios_adaptation_option_2,
                    }
                )
            return adjusted_yield_ratios

        def compute_total_profits(
            yield_ratios: np.ndarray, crops_mask: np.ndarray, nan_array: np.ndarray
        ) -> np.ndarray:
            """
            Compute total profits for all agents across different drought scenarios.

            Parameters:
                yield_ratios (np.ndarray): Yield ratios for agents under different drought scenarios.
                crops_mask (np.ndarray): Mask indicating valid crop entries.
                nan_array (np.ndarray): Array filled with NaNs for reference.

            Returns:
                np.ndarray: Total profits for agents under each drought scenario.
            """
            total_profits = np.zeros((self.n, yield_ratios.shape[1]))
            for col in range(yield_ratios.shape[1]):
                total_profits[:, col] = self.yield_ratio_to_profit(
                    yield_ratios[:, col], crops_mask, nan_array
                )
            return total_profits

        def format_results(total_profits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Transpose and slice the total profits matrix, and extract the 'no drought' scenario profits.

            """
            total_profits = total_profits.T
            profits_no_event = total_profits[-1, :]
            total_profits = total_profits[:-1, :]
            return total_profits, profits_no_event

        # Main function logic
        yield_ratios = self.convert_probability_to_yield_ratio()

        # Create a mask for valid crops (exclude non-crop values)
        crops_mask = (self.crop_calendar[:, :, 0] >= 0) & (
            self.crop_calendar[:, :, 0] < len(self.crop_data["reference_yield_kg_m2"])
        )

        # Create an array filled with NaNs for reference data
        nan_array = np.full_like(
            self.crop_calendar[:, :, 0], fill_value=np.nan, dtype=float
        )

        # Compute profits without adaptation
        total_profits = compute_total_profits(yield_ratios, crops_mask, nan_array)
        total_profits, profits_no_event = format_results(total_profits)

        # Initialize adaptation data
        if adaptation_type != 0:
            adaptation_data = compute_adaptation_gains(yield_ratios)
            adjusted_yield_ratios = adjust_yield_ratios_with_adaptation(
                yield_ratios, adaptation_data
            )

            if adaptation_type == 1:
                # Adaptation Type 1
                total_profits_adaptation = compute_total_profits(
                    adjusted_yield_ratios["yield_ratios_adaptation"],
                    crops_mask,
                    nan_array,
                )
                total_profits_adaptation, profits_no_event_adaptation = format_results(
                    total_profits_adaptation
                )
                return (
                    total_profits,
                    profits_no_event,
                    total_profits_adaptation,
                    profits_no_event_adaptation,
                )
            else:
                # Other Adaptation Types
                total_profits_adaptation_option_1 = compute_total_profits(
                    adjusted_yield_ratios["yield_ratios_adaptation_option_1"],
                    crops_mask,
                    nan_array,
                )
                (
                    total_profits_adaptation_option_1,
                    profits_no_event_adaptation_option_1,
                ) = format_results(total_profits_adaptation_option_1)

                total_profits_adaptation_option_2 = compute_total_profits(
                    adjusted_yield_ratios["yield_ratios_adaptation_option_2"],
                    crops_mask,
                    nan_array,
                )
                (
                    total_profits_adaptation_option_2,
                    profits_no_event_adaptation_option_2,
                ) = format_results(total_profits_adaptation_option_2)

                return (
                    total_profits,
                    profits_no_event,
                    total_profits_adaptation_option_1,
                    profits_no_event_adaptation_option_1,
                    total_profits_adaptation_option_2,
                    profits_no_event_adaptation_option_2,
                    adaptation_data["new_crop_nr"],
                    adaptation_data["new_farmer_id"],
                )
        else:
            return total_profits, profits_no_event

    def convert_probability_to_yield_ratio(self) -> np.ndarray:
        """
        Convert drought probabilities to yield ratios based on the given polynomial relationship.

        For each farmer's yield-probability relationship (represented as a polynomial),
        this function calculates the inverse of the relationship and then applies the
        inverted polynomial to a set of given probabilities to obtain yield ratios.
        The resulting yield ratios are then adjusted to lie between 0 and 1. The final
        results are stored in `self.yield_ratios_drought_event`.

        Note:
            - It assumes that the polynomial relationship is invertible.
            - Adjusts yield ratios to be non-negative and capped at 1.0.
        """

        def logarithmic_function(probability, params):
            a = params[:, 0]
            b = params[:, 1]
            x = probability[:, np.newaxis]

            return a * np.log10(x) + b

        yield_ratios = logarithmic_function(
            1 / self.p_droughts, self.farmer_yield_probability_relation
        ).T

        # Adjust the yield ratios to lie between 0 and 1
        yield_ratios[yield_ratios < 0] = 0  # Ensure non-negative yield ratios
        yield_ratios[yield_ratios > 1] = 1  # Cap the yield ratios at 1

        # Store the results in a global variable
        self.yield_ratios_drought_event = yield_ratios[:]

        return self.yield_ratios_drought_event

    def create_unique_groups(self, N=5):
        """
        Create unique groups based on elevation data and merge with crop calendar.

        Parameters:
        N (int): Number of groups to divide the elevation data into.

        Returns:
        numpy.ndarray: Merged array with crop calendar and elevation distribution groups.
        """
        # Calculating the thresholds for the N groups
        percentiles = [100 * i / N for i in range(1, N)]
        basin_elevation_thresholds = np.percentile(self.elevation.data, percentiles)

        # Use np.digitize to assign group labels
        distribution_array = np.digitize(
            self.elevation.data, bins=basin_elevation_thresholds, right=False
        )

        # Merging crop calendar and distribution array
        crop_elevation_group = np.hstack(
            (
                self.crop_calendar[:, :, 0],
                distribution_array.reshape(-1, 1),
                self.farmer_class.reshape(-1, 1),
            )
        )

        return crop_elevation_group.astype(np.int32)

    def crop_yield_ratio_difference(self, yield_ratios) -> np.ndarray:
        """
        Calculate the relative yield ratio improvement for farmers adopting a certain adaptation.

        This function determines how much better farmers that have adopted a particular adaptation
        are doing in terms of their yield ratio as compared to those who haven't.

        Args:
            adaptation_type: The type of adaptation being considered.

        Returns:
            An array representing the relative yield ratio improvement for each agent.

        TO DO: vectorize
        """

        # Make array based on elevation and crop calendar
        crop_elevation_group = self.create_unique_groups()

        # Get unique groups and group indices
        unique_crop_groups, group_indices = np.unique(
            crop_elevation_group, axis=0, return_inverse=True
        )
        # Initialize array to store relative yield ratio improvement for unique groups
        unique_yield_ratio_gain_option_1 = np.full(
            (len(unique_crop_groups), len(self.p_droughts)), 0, dtype=np.float32
        )
        unique_yield_ratio_gain_option_2 = np.full(
            (len(unique_crop_groups), len(self.p_droughts)), 0, dtype=np.float32
        )
        id_to_switch_to = np.full((len(unique_crop_groups), 2), -1)
        crop_to_switch_to = np.full((len(unique_crop_groups), 2), -1)

        def get_updated_combinations(unique_combination):
            """
            Update combinations based on the value of unique_combination[0].

            Parameters:
            unique_combination (list): List containing the unique combination values.

            Returns:
            tuple: Updated combinations (unique_combination_option_1, unique_combination_option_2).
            """
            # Define the mapping of initial values to their corresponding options
            condition_mapping = {
                0: (26, 27),
                26: (0, 27),
                27: (0, 26),
                1: (28, 29),
                28: (1, 29),
                29: (1, 28),
                2: (30, 31),
                30: (2, 31),
                31: (2, 30),
                7: (32, 33),
                32: (7, 33),
                33: (7, 32),
                9: (34, 35),
                34: (9, 35),
                35: (9, 34),
            }

            # Initialize the options with the original unique_combination values
            unique_combination_option_1 = unique_combination.copy()
            unique_combination_option_2 = unique_combination.copy()

            # Check if the initial value is in the mapping
            if unique_combination[0] in condition_mapping:
                option_1_value, option_2_value = condition_mapping[
                    unique_combination[0]
                ]
                unique_combination_option_1[0] = option_1_value
                unique_combination_option_2[0] = option_2_value

            return unique_combination_option_1, unique_combination_option_2

        crops_with_varieties = [0, 26, 27, 1, 28, 29, 2, 30, 31, 7, 32, 33, 9, 34, 35]

        # Loop over each unique group of farmers to determine their average yield ratio
        for idx, unique_combination in enumerate(unique_crop_groups):
            if unique_combination[0] in crops_with_varieties:
                unique_farmer_groups = (
                    crop_elevation_group == unique_combination[None, ...]
                ).all(axis=1)

                # Identify the adapted counterpart of the current group
                unique_combination_option_1 = unique_combination.copy()
                unique_combination_option_2 = unique_combination.copy()

                unique_combination_option_1, unique_combination_option_2 = (
                    get_updated_combinations(unique_combination)
                )

                unique_farmer_groups_option_1 = (
                    crop_elevation_group == unique_combination_option_1[None, ...]
                ).all(axis=1)

                unique_farmer_groups_option_2 = (
                    crop_elevation_group == unique_combination_option_2[None, ...]
                ).all(axis=1)

                def find_most_similar_index(target_series, yield_ratios, groups):
                    distances = np.linalg.norm(
                        yield_ratios[groups] - target_series, axis=1
                    )
                    most_similar_index_within_subset = np.argmin(distances)
                    global_indices = np.where(groups)[0]
                    return global_indices[most_similar_index_within_subset]

                def calculate_yield_ratio_gain(
                    yield_ratios, current_yield_ratio, groups
                ):
                    yield_ratio = np.mean(yield_ratios[groups, :], axis=0)
                    yield_ratio_gain = yield_ratio - current_yield_ratio
                    return yield_ratio_gain

                def process_option(
                    yield_ratios,
                    unique_combination_option,
                    groups,
                    current_yield_ratio,
                    crop_to_switch_to,
                    id_to_switch_to,
                    option_nr,
                    idx,
                ):
                    yield_ratio_gain = calculate_yield_ratio_gain(
                        yield_ratios, current_yield_ratio, groups
                    )
                    crop_to_switch_to[idx, option_nr] = unique_combination_option[0]

                    if np.isnan(yield_ratio_gain).all():
                        yield_ratio_gain = np.zeros_like(yield_ratio_gain)
                    else:
                        id_to_switch_to[idx, option_nr] = find_most_similar_index(
                            yield_ratio_gain, yield_ratios, groups
                        )
                    return yield_ratio_gain

                # Calculate mean yield ratio over past years for the unadapted group
                current_yield_ratio = np.mean(
                    yield_ratios[unique_farmer_groups, :], axis=0
                )

                if np.isnan(current_yield_ratio).all():
                    current_yield_ratio = np.zeros_like(current_yield_ratio)

                # Process option 1
                yield_ratio_gain_relative_option_1 = process_option(
                    yield_ratios,
                    unique_combination_option_1,
                    unique_farmer_groups_option_1,
                    current_yield_ratio,
                    crop_to_switch_to,
                    id_to_switch_to,
                    0,
                    idx,
                )

                # Process option 2
                yield_ratio_gain_relative_option_2 = process_option(
                    yield_ratios,
                    unique_combination_option_2,
                    unique_farmer_groups_option_2,
                    current_yield_ratio,
                    crop_to_switch_to,
                    id_to_switch_to,
                    1,
                    idx,
                )

                unique_yield_ratio_gain_option_1[idx, :] = (
                    yield_ratio_gain_relative_option_1
                )
                unique_yield_ratio_gain_option_2[idx, :] = (
                    yield_ratio_gain_relative_option_2
                )

                assert not (
                    np.isinf(yield_ratio_gain_relative_option_1).any()
                    or np.isinf(yield_ratio_gain_relative_option_2).any()
                ), "gains adaptation value is inf"

            else:
                pass

        # Convert group-based results into agent-specific results
        gains_adaptation_option_1 = unique_yield_ratio_gain_option_1[group_indices, :]
        gains_adaptation_option_2 = unique_yield_ratio_gain_option_2[group_indices, :]
        new_crop_nr = crop_to_switch_to[group_indices, :]
        new_farmer_id = id_to_switch_to[group_indices, :]

        return (
            gains_adaptation_option_1,
            gains_adaptation_option_2,
            new_crop_nr,
            new_farmer_id,
        )

    def adaptation_yield_ratio_difference(
        self, adapted: np.ndarray, yield_ratios
    ) -> np.ndarray:
        """
        Calculate the relative yield ratio improvement for farmers adopting a certain adaptation.

        This function determines how much better farmers that have adopted a particular adaptation
        are doing in terms of their yield ratio as compared to those who haven't.

        Args:
            adaptation_type: The type of adaptation being considered.

        Returns:
            An array representing the relative yield ratio improvement for each agent.

        TO DO: vectorize
        """

        crop_elevation_group = self.create_unique_groups()

        # Initialize array to store relative yield ratio improvement for unique groups
        unique_yield_ratio_gain_relative = np.zeros(
            (len(np.unique(crop_elevation_group, axis=0)), len(self.p_droughts)),
            dtype=np.float32,
        )

        # Get unique groups and group indices
        unique_groups, group_indices = np.unique(
            crop_elevation_group, axis=0, return_inverse=True
        )

        # Loop over each unique group of farmers to determine their average yield ratio
        for idx, unique_combination in enumerate(unique_groups):
            unique_farmer_groups = (
                crop_elevation_group == unique_combination[None, ...]
            ).all(axis=1)

            # Groups compare yield between those who have well (2) and those that dont
            unique_combination_adapted = unique_combination.copy()
            unique_combination_adapted[-1] = 2
            unique_farmer_groups_adapted = (
                crop_elevation_group == unique_combination_adapted[None, ...]
            ).all(axis=1)

            if (
                np.count_nonzero(unique_farmer_groups) != 0
                and np.count_nonzero(unique_farmer_groups_adapted) != 0
            ):
                # Calculate mean yield ratio over past years for both adapted and unadapted groups
                unadapted_yield_ratio = np.mean(
                    yield_ratios[unique_farmer_groups, :], axis=0
                )
                adapted_yield_ratio = np.mean(
                    yield_ratios[unique_farmer_groups_adapted, :], axis=0
                )

                yield_ratio_gain = adapted_yield_ratio - unadapted_yield_ratio

                unique_yield_ratio_gain_relative[idx, :] = yield_ratio_gain

        # Convert group-based results into agent-specific results
        gains_adaptation = unique_yield_ratio_gain_relative[group_indices, :]

        assert np.max(gains_adaptation) != np.inf, "gains adaptation value is inf"

        return gains_adaptation

    def adaptation_water_cost_difference(
        self, adapted: np.ndarray, energy_cost, water_cost
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the relative yield ratio improvement for farmers adopting a certain adaptation.

        This function determines how much better farmers that have adopted a particular adaptation
        are doing in terms of their yield ratio as compared to those who haven't.

        Args:
            adapted (np.ndarray): Array indicating adaptation status (0 or 1) for each agent.
            energy_cost (np.ndarray): Array of energy costs for each agent.
            water_cost (np.ndarray): Array of water costs for each agent.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays representing the relative energy cost and water cost improvements for each agent.
        """

        # Create unique groups based on elevation data
        crop_elevation_group = self.create_unique_groups()

        # Get unique groups and group indices
        unique_groups, group_indices = np.unique(
            crop_elevation_group, axis=0, return_inverse=True
        )

        n_groups = unique_groups.shape[0]

        # Initialize arrays to store gains per group
        unique_water_cost_gain = np.zeros(n_groups, dtype=np.float32)
        unique_energy_cost_gain = np.zeros(n_groups, dtype=np.float32)

        # For each group, compute gains
        for g in range(n_groups):
            # Agents in the current group
            group_members = group_indices == g

            # Split agents into adapted and unadapted within the group
            unadapted_agents = group_members & (adapted == 0)
            adapted_agents = group_members & (adapted == 1)

            # Check if both adapted and unadapted agents are present
            if np.any(unadapted_agents) and np.any(adapted_agents):
                # Calculate mean water and energy costs for unadapted agents
                unadapted_water_cost = np.mean(water_cost[unadapted_agents], axis=0)
                unadapted_energy_cost = np.mean(energy_cost[unadapted_agents], axis=0)

                # Calculate mean water and energy costs for adapted agents
                adapted_water_cost = np.mean(water_cost[adapted_agents], axis=0)
                adapted_energy_cost = np.mean(energy_cost[adapted_agents], axis=0)

                # Calculate gains
                water_cost_gain = adapted_water_cost - unadapted_water_cost
                energy_cost_gain = adapted_energy_cost - unadapted_energy_cost

                # Store gains for the group
                unique_water_cost_gain[g] = water_cost_gain
                unique_energy_cost_gain[g] = energy_cost_gain
            else:
                # If not enough data, set gains to zero or np.nan
                unique_water_cost_gain[g] = 0  # or np.nan
                unique_energy_cost_gain[g] = 0  # or np.nan

        # Map gains back to agents using group indices
        water_cost_adaptation_gain = unique_water_cost_gain[group_indices]
        energy_cost_adaptation_gain = unique_energy_cost_gain[group_indices]

        return energy_cost_adaptation_gain, water_cost_adaptation_gain

    def yield_ratio_to_profit(
        self, yield_ratios: np.ndarray, crops_mask: np.ndarray, nan_array: np.ndarray
    ) -> np.ndarray:
        """
        Convert yield ratios to monetary profit values.

        This function computes the profit values for each crop based on given yield ratios.
        The profit is calculated by multiplying the crop yield in kilograms per sqr. meter with
        the average crop price. The function leverages various data inputs, such as current crop
        prices and reference yields.

        Args:
            yield_ratios: The array of yield ratios for the crops.
            crops_mask: A mask that denotes valid crops, based on certain conditions.
            array_with_reference: An array initialized with NaNs, later used to store reference yields and crop prices.

        Returns:
            An array representing the profit values for each crop based on the given yield ratios.

        Note:
            - It assumes that the crop prices are non-NaN for the current model time.
            - The function asserts that crop yields in kg are always non-negative.

        TODO: Take the average crop prices over the last x years.
        """

        # Create blank arrays with only nans
        array_with_reference_yield = nan_array.copy()
        array_with_price = nan_array.copy()

        # Check if prices are monthly or yearly
        price_frequency = self.model.config["agent_settings"]["market"][
            "price_frequency"
        ]

        if price_frequency == "monthly":
            total_price = 0
            month_count = 0

            # Ending date and start date set to one year prior
            end_date = self.model.current_time
            start_date = datetime(end_date.year - 1, 1, 1)

            # Loop through each month from start_date to end_date to get the sum of crop costs over the past year
            current_date = start_date
            while current_date <= end_date:
                assert (
                    self.crop_prices[0] is not None
                ), "behavior needs crop prices to work"
                monthly_price = self.crop_prices[1][
                    self.crop_prices[0].get(current_date)
                ]
                total_price += monthly_price
                # Move to the next month
                if current_date.month == 12:
                    current_date = datetime(current_date.year + 1, 1, 1)
                else:
                    current_date = datetime(
                        current_date.year, current_date.month + 1, 1
                    )
                month_count += 1

            # Calculate the average price over the last year
            crop_prices = total_price / month_count

        else:
            crop_prices = self.agents.market.crop_prices[self.region_id]

        # Assign the reference yield and current crop price to the array based on valid crop mask
        array_with_price[crops_mask] = np.take(
            crop_prices, self.crop_calendar[:, :, 0][crops_mask].astype(int)
        )
        assert not np.isnan(
            array_with_price[crops_mask]
        ).any()  # Ensure there are no NaN values in crop prices

        array_with_reference_yield[crops_mask] = np.take(
            self.crop_data["reference_yield_kg_m2"].values,
            self.crop_calendar[:, :, 0][crops_mask].astype(int),
        )

        # Calculate the product of the average reference yield and average crop price ignoring NaN values
        reference_profit_m2 = np.nansum(
            array_with_reference_yield * array_with_price, axis=1
        )
        assert (
            reference_profit_m2 >= 0
        ).all()  # Ensure all crop yields are non-negative

        # Calculate profit by multiplying yield with price
        profit_m2 = yield_ratios * reference_profit_m2

        return profit_m2

    def update_loans(self) -> None:
        # Subtract 1 off each loan duration, except if that loan is at 0
        self.loan_tracker -= self.loan_tracker != 0
        # If the loan tracker is at 0, cancel the loan amount and subtract it of the total
        expired_loan_mask = self.loan_tracker == 0

        # Add a column to make it the same shape as the loan amount array
        new_column = np.full((self.n, 1, 5), False)
        expired_loan_mask = np.column_stack((expired_loan_mask, new_column))

        # Sum the expired loan amounts
        ending_loans = expired_loan_mask * self.all_loans_annual_cost
        total_loan_reduction = np.sum(ending_loans, axis=(1, 2))

        # Subtract it from the total loans and set expired loans to 0
        self.all_loans_annual_cost[:, -1, 0] -= total_loan_reduction
        self.all_loans_annual_cost[expired_loan_mask] = 0

        # Adjust for inflation in separate array for export
        # Calculate the cumulative inflation from the start year to the current year for each farmer
        inflation_arrays = [
            self.get_value_per_farmer_from_region_id(
                self.inflation_rate, datetime(year, 1, 1)
            )
            for year in range(
                self.model.config["general"]["spinup_time"].year,
                self.model.current_time.year + 1,
            )
        ]

        cum_inflation = np.ones_like(inflation_arrays[0])
        for inflation in inflation_arrays:
            cum_inflation *= inflation

        self.adjusted_annual_loan_cost = (
            self.all_loans_annual_cost / cum_inflation[..., None, None]
        )

    def get_value_per_farmer_from_region_id(self, data, time) -> np.ndarray:
        index = data[0].get(time)
        unique_region_ids, inv = np.unique(self.region_id, return_inverse=True)
        values = np.full_like(unique_region_ids, np.nan, dtype=np.float32)
        for i, region_id in enumerate(unique_region_ids):
            values[i] = data[1][region_id][index]
        return values[inv]

    @staticmethod
    @njit(cache=True)
    def field_size_per_farmer_numba(
        field_indices_by_farmer: np.ndarray,
        field_indices: np.ndarray,
        cell_area: np.ndarray,
    ) -> np.ndarray:
        """Gets the field size for each farmer.

        Args:
            field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
            field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.
            cell_area: Subarray of cell_area.

        Returns:
            field_size_per_farmer: Field size for each farmer in m2.
        """
        field_size_per_farmer = np.zeros(
            field_indices_by_farmer.shape[0], dtype=np.float32
        )
        for farmer in range(field_indices_by_farmer.shape[0]):
            for field in get_farmer_HRUs(
                field_indices, field_indices_by_farmer, farmer
            ):
                field_size_per_farmer[farmer] += cell_area[field]
        return field_size_per_farmer

    @property
    def field_size_per_farmer(self) -> np.ndarray:
        """Gets the field size for each farmer.

        Returns:
            field_size_per_farmer: Field size for each farmer in m2.
        """
        return self.field_size_per_farmer_numba(
            self.field_indices_by_farmer.data,
            self.field_indices,
            self.var.cellArea.get() if self.model.use_gpu else self.var.cellArea,
        )

    @property
    def irrigated_fields(self) -> np.ndarray:
        """Gets the indices of fields that are irrigated.

        Returns:
            irrigated_fields: Indices of fields that are irrigated.
        """
        is_irrigated = np.take(
            self.irrigation_source != self.irrigation_source_key["no"],
            self.var.land_owners,
        )
        is_irrigated[self.var.land_owners == -1] = False
        return is_irrigated

    @property
    def groundwater_depth(self):
        groundwater_depth = get_farmer_groundwater_depth(
            self.n,
            self.model.groundwater.groundwater_depth,
            self.model.data.HRU.HRU_to_grid,
            self.field_indices,
            self.field_indices_by_farmer.data,
            self.model.data.HRU.cellArea,
        )
        assert not np.isnan(groundwater_depth).any(), "groundwater depth is nan"
        return groundwater_depth

    def step(self) -> None:
        """
        This function is called at the beginning of each timestep.

        Then, farmers harvest and plant crops.
        """

        self.harvest()
        self.plant()
        self.water_abstraction_sum()

        ## yearly actions
        if self.model.current_time.month == 1 and self.model.current_time.day == 1:
            # Set yearly yield ratio based on the difference between saved actual and potential profit
            self.yearly_yield_ratio = (
                self.yearly_profits / self.yearly_potential_profits
            )

            self.save_yearly_spei()

            # reset the irrigation limit, but only if a full year has passed already. Otherwise
            # the cumulative water deficit is not year completed.
            if self.model.current_time.year - 1 > self.model.spinup_start.year:
                self.remaining_irrigation_limit_m3[:] = self.irrigation_limit_m3[:]

            # for now class is only dependent on being in a command area or not
            self.farmer_class = self.is_in_command_area.copy().astype(np.int32)

            # Set to 0 if channel abstraction is bigger than reservoir and groundwater, 1 for reservoir, 2 for groundwater and 3 no abstraction
            self.farmer_class[
                (
                    self.yearly_abstraction_m3_by_farmer[:, 0, 0]
                    > self.yearly_abstraction_m3_by_farmer[:, 1, 0]
                )
                & (
                    self.yearly_abstraction_m3_by_farmer[:, 0, 0]
                    > self.yearly_abstraction_m3_by_farmer[:, 2, 0]
                )
            ] = 0
            self.farmer_class[
                (
                    self.yearly_abstraction_m3_by_farmer[:, 1, 0]
                    > self.yearly_abstraction_m3_by_farmer[:, 0, 0]
                )
                & (
                    self.yearly_abstraction_m3_by_farmer[:, 1, 0]
                    > self.yearly_abstraction_m3_by_farmer[:, 2, 0]
                )
            ] = 1
            self.farmer_class[
                (
                    self.yearly_abstraction_m3_by_farmer[:, 2, 0]
                    > self.yearly_abstraction_m3_by_farmer[:, 0, 0]
                )
                & (
                    self.yearly_abstraction_m3_by_farmer[:, 2, 0]
                    > self.yearly_abstraction_m3_by_farmer[:, 1, 0]
                )
            ] = 2

            # Set to 3 for precipitation if there is no abstraction
            self.farmer_class[self.yearly_abstraction_m3_by_farmer[:, 3, 0] == 0] = 3

            print(
                "well",
                np.mean(self.yearly_yield_ratio[self.farmer_class == 2, 1]),
                "no well",
                np.mean(self.yearly_yield_ratio[self.farmer_class == 3, 1]),
                "total_mean",
                np.mean(self.yearly_yield_ratio[:, 1]),
            )

            energy_cost, water_cost, average_extraction_speed = (
                self.calculate_water_costs()
            )

            timer = TimingModule("crop_farmers")

            if (
                not self.model.spinup
                and "ruleset" in self.config
                and not self.config["ruleset"] == "no-adaptation"
            ):
                # Determine the relation between drought probability and yield
                self.calculate_yield_spei_relation()
                # self.calculate_yield_spei_relation_group()
                # self.calculate_yield_spei_relation_test_group()
                timer.new_split("yield-spei relation")

                # These adaptations can only be done if there is a yield-probability relation
                if not np.all(self.farmer_yield_probability_relation == 0):
                    if (
                        not self.config["expected_utility"]["adaptation_well"][
                            "ruleset"
                        ]
                        == "no-adaptation"
                    ):
                        self.adapt_irrigation_well(
                            average_extraction_speed, energy_cost, water_cost
                        )
                        timer.new_split("irr well")
                    if (
                        not self.config["expected_utility"]["adaptation_sprinkler"][
                            "ruleset"
                        ]
                        == "no-adaptation"
                    ):
                        self.adapt_irrigation_efficiency(energy_cost, water_cost)
                        timer.new_split("irr efficiency")
                    if (
                        not self.config["expected_utility"]["crop_switching"]["ruleset"]
                        == "no-adaptation"
                    ):
                        self.adapt_crops()
                        timer.new_split("adapt crops")
                    # self.switch_crops_neighbors()
                else:
                    raise AssertionError(
                        "Cannot adapt without yield - probability relation"
                    )

            print(timer)
            advance_crop_rotation_year(
                current_crop_calendar_rotation_year_index=self.current_crop_calendar_rotation_year_index,
                crop_calendar_rotation_years=self.crop_calendar_rotation_years,
            )

            # Update loans
            self.update_loans()

            # Reset total crop age
            shift_and_update(self.total_crop_age, self.total_crop_age[:, 0])

            for i in range(len(self.yearly_abstraction_m3_by_farmer[0, :, 0])):
                shift_and_reset_matrix(self.yearly_abstraction_m3_by_farmer[:, i, :])

            # Shift the potential and yearly profits forward
            shift_and_reset_matrix(self.yearly_profits)
            shift_and_reset_matrix(self.yearly_potential_profits)
        # if self.model.current_timestep == 100:
        #     self.add_agent(indices=(np.array([310, 309]), np.array([69, 69])))
        # if self.model.current_timestep == 105:
        #     self.remove_agent(farmer_idx=1000)

    def remove_agents(
        self, farmer_indices: list[int], land_use_type: int
    ) -> np.ndarray:
        farmer_indices = np.array(farmer_indices)
        if farmer_indices.size > 0:
            farmer_indices = np.sort(farmer_indices)[::-1]
            HRUs_with_removed_farmers = []
            for idx in farmer_indices:
                HRUs_with_removed_farmers.append(self.remove_agent(idx, land_use_type))
        return np.concatenate(HRUs_with_removed_farmers)

    def remove_agent(self, farmer_idx: int, land_use_type: int) -> np.ndarray:
        assert farmer_idx >= 0, "Farmer index must be positive."
        assert (
            farmer_idx < self.n
        ), "Farmer index must be less than the number of agents."
        last_farmer_HRUs = get_farmer_HRUs(
            self.field_indices, self.field_indices_by_farmer.data, -1
        )
        last_farmer_field_size = self.field_size_per_farmer[-1]  # for testing only

        # disown the farmer.
        HRUs_farmer_to_be_removed = get_farmer_HRUs(
            self.field_indices, self.field_indices_by_farmer.data, farmer_idx
        )
        self.var.land_owners[HRUs_farmer_to_be_removed] = -1
        self.var.crop_map[HRUs_farmer_to_be_removed] = -1
        self.var.crop_age_days_map[HRUs_farmer_to_be_removed] = -1
        self.var.crop_harvest_age_days[HRUs_farmer_to_be_removed] = -1
        self.var.land_use_type[HRUs_farmer_to_be_removed] = land_use_type

        # reduce number of agents
        self.n -= 1

        if not self.n == farmer_idx:
            # move data of last agent to the index of the agent that is to be removed, effectively removing that agent.
            for name, agent_array in self.agent_arrays.items():
                agent_array[farmer_idx] = agent_array[-1]
                # reduce the number of agents by 1
                assert agent_array.n == self.n + 1
                agent_array.n = self.n

            # update the field indices of the last agent
            self.var.land_owners[last_farmer_HRUs] = farmer_idx
        else:
            for agent_array in self.agent_arrays.values():
                agent_array.n = self.n

        # TODO: Speed up field index updating.
        self.update_field_indices()
        if self.n == farmer_idx:
            assert (
                get_farmer_HRUs(
                    self.field_indices, self.field_indices_by_farmer.data, farmer_idx
                ).size
                == 0
            )
        else:
            assert np.array_equal(
                np.sort(last_farmer_HRUs),
                np.sort(
                    get_farmer_HRUs(
                        self.field_indices,
                        self.field_indices_by_farmer.data,
                        farmer_idx,
                    )
                ),
            )
            assert math.isclose(
                last_farmer_field_size,
                self.field_size_per_farmer[farmer_idx],
                abs_tol=1,
            )

        assert (self.var.land_owners[HRUs_farmer_to_be_removed] == -1).all()
        return HRUs_farmer_to_be_removed

    def add_agent(
        self,
        indices,
        values={
            "risk_aversion": 1,
            "interest_rate": 1,
            "discount_rate": 1,
            "adapted": False,
            "time_adapted": False,
            "SEUT_no_adapt": 1,
            "EUT_no_adapt": 1,
            "crops": -1,
            "irrigation_source": -1,
            "well_depth": -1,
            "channel_abstraction_m3_by_farmer": 0,
            "reservoir_abstraction_m3_by_farmer": 0,
            "groundwater_abstraction_m3_by_farmer": 0,
            "yearly_abstraction_m3_by_farmer": 0,
            "total_crop_age": 0,
            "per_harvest_yield_ratio": 0,
            "per_harvest_SPEI": 0,
            "monthly_SPEI": 0,
            "disposable_income": 0,
            "household_size": 2,
            "yield_ratios_drought_event": 1,
            "risk_perception": 1,
            "drought_timer": 1,
            "yearly_SPEI_probability": 1,
            "yearly_yield_ratio": 1,
            "yearly_profits": 1,
            "yearly_potential_profits": 1,
            "farmer_yield_probability_relation": 1,
            "irrigation_efficiency": 0.9,
            "yield_ratio_multiplier": 1,
            "base_management_yield_ratio": 1,
            "yield_ratio_management": 1,
            "annual_costs_all_adaptations": 1,
            "farmer_class": 1,
            "water_use": 1,
            "GEV_parameters": 1,
            "risk_perc_min": 1,
            "risk_perc_max": 1,
            "risk_decr": 1,
            "decision_horizon": 1,
        },
    ):
        """This function can be used to add new farmers."""
        HRU = self.model.data.split(indices)
        assert self.var.land_owners[HRU] == -1, "There is already a farmer here."
        self.var.land_owners[HRU] = self.n

        pixels = np.column_stack(indices)[:, [1, 0]]
        agent_location = np.mean(
            pixels_to_coords(pixels + 0.5, self.var.gt), axis=0
        )  # +.5 to use center of pixels

        self.n += 1  # increment number of agents
        for name, agent_array in self.agent_arrays.items():
            agent_array.n += 1
            if name == "locations":
                agent_array[self.n - 1] = agent_location
            elif name == "elevation":
                agent_array[self.n - 1] = self.elevation_subgrid.sample_coords(
                    np.expand_dims(agent_location, axis=0)
                )
            elif name == "region_id":
                agent_array[self.n - 1] = self.subdistrict_map.sample_coords(
                    np.expand_dims(agent_location, axis=0)
                )
            elif name == "field_indices_by_farmer":
                # TODO: Speed up field index updating.
                self.update_field_indices()
            else:
                agent_array[self.n - 1] = values[name]

    def restore_state(self, folder):
        save_state_path = self.get_save_state_path(folder, mkdir=True)
        with open(save_state_path / "state.txt", "r") as f:
            for line in f:
                attribute = line.strip()
                fp = save_state_path / f"{attribute}.npz"
                values = np.load(fp)["data"]
                if not hasattr(self, "max_n"):
                    self.max_n = self.get_max_n(values.shape[0])
                values = AgentArray(values, max_n=self.max_n)

                setattr(self, attribute, values)

        self.n = self.locations.shape[0]
        self.update_field_indices()
