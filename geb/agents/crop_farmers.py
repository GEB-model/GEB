# -*- coding: utf-8 -*-
import math
from datetime import datetime
import json
import calendar
from typing import Tuple, Union

from scipy.stats import genextreme
from scipy.optimize import curve_fit

import numpy as np
from numba import njit

from ..workflows import balance_check

from honeybees.library.mapIO import MapReader
from honeybees.library.raster import pixels_to_coords, sample_from_map
from honeybees.library.neighbors import find_neighbors

from ..data import (
    load_regional_crop_data_from_dict,
    load_crop_data,
    load_economic_data,
)
from .decision_module import DecisionModule
from .general import AgentArray, AgentBaseClass


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


@njit(cache=True)
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
):
    # channel abstraction
    available_channel_storage_cell_m = (
        available_channel_storage_m3[grid_cell] / cell_area[field]
    )
    channel_abstraction_cell_m = min(
        available_channel_storage_cell_m,
        irrigation_water_demand_field,
    )
    channel_abstraction_cell_m3 = channel_abstraction_cell_m * cell_area[field]
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
        available_groundwater_cell_m = (
            available_groundwater_m3[grid_cell] / cell_area[field]
        )
        groundwater_abstraction_cell_m = min(
            available_groundwater_cell_m,
            irrigation_water_demand_field,
        )
        groundwater_abstraction_cell_m3 = (
            groundwater_abstraction_cell_m * cell_area[field]
        )
        available_groundwater_m3[grid_cell] -= groundwater_abstraction_cell_m3
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
    command_areas: np.ndarray,
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
    groundwater_depth_per_farmer = np.zeros(activation_order.size, dtype=np.float32)

    has_access_to_irrigation_water = np.zeros(activation_order.size, dtype=np.bool_)
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
        farmer_has_access_to_irrigation_water = False
        for field in farmer_fields:
            grid_cell = HRU_to_grid[field]

            # Convert the groundwater depth to groundwater depth per farmer
            groundwater_depth_per_farmer[farmer] = groundwater_depth[grid_cell]

            if well_irrigated[farmer]:
                if groundwater_depth[grid_cell] < well_depth[farmer]:
                    farmer_has_access_to_irrigation_water = True
                    break
            elif surface_irrigated[farmer]:
                if available_channel_storage_m3[grid_cell] > 0:
                    farmer_has_access_to_irrigation_water = True
                    break
                command_area = command_areas[field]
                # -1 means no command area
                if (
                    command_area != -1
                    and available_reservoir_storage_m3[command_area] > 0
                ):
                    farmer_has_access_to_irrigation_water = True
                    break

        has_access_to_irrigation_water[activated_farmer_index] = (
            farmer_has_access_to_irrigation_water
        )

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
                    max_water_content[farmer_fields]
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
                            )

                            # command areas
                            command_area = command_areas[field]
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

                        assert (
                            irrigation_water_demand_field >= -1e15
                        )  # Make sure irrigation water demand is zero, or positive. Allow very small error.

                    water_consumption_m[field] = (
                        water_withdrawal_m[field] * irrigation_efficiency_farmer
                    )
                    irrigation_loss_m = (
                        water_withdrawal_m[field] - water_consumption_m[field]
                    )
                    returnFlowIrr_m[field] = irrigation_loss_m * return_fraction
                    addtoevapotrans_m[field] = irrigation_loss_m * (1 - return_fraction)

    return (
        channel_abstraction_m3_by_farmer,
        reservoir_abstraction_m3_by_farmer,
        groundwater_abstraction_m3_by_farmer,
        water_withdrawal_m,
        water_consumption_m,
        returnFlowIrr_m,
        addtoevapotrans_m,
        has_access_to_irrigation_water,
        groundwater_depth_per_farmer,
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
        self.moving_average_threshold = self.model.config["agent_settings"][
            "expected_utility"
        ]["drought_risk_calculations"]["event_perception"]["drought_threshold"]
        self.previous_month = 0

        # Assign risk aversion sigma, time discounting preferences, expenditure_cap
        self.expenditure_cap = self.model.config["agent_settings"]["expected_utility"][
            "decisions"
        ]["expenditure_cap"]

        self.inflation_rate = load_economic_data(
            self.model.files["dict"]["economics/inflation_rates"]
        )
        self.lending_rate = load_economic_data(
            self.model.files["dict"]["economics/lending_rates"]
        )

        # India specific old well variables
        # Well cost variables
        self.borewell_cost_1 = load_economic_data(
            self.model.files["dict"]["economics/borewell_cost_1"]
        )
        self.borewell_cost_2 = load_economic_data(
            self.model.files["dict"]["economics/borewell_cost_2"]
        )
        self.pump_cost = load_economic_data(
            self.model.files["dict"]["economics/pump_cost"]
        )
        self.irrigation_maintenance = load_economic_data(
            self.model.files["dict"]["economics/irrigation_maintenance"]
        )
        self.electricity_cost = load_economic_data(
            self.model.files["dict"]["economics/electricity_cost"]
        )

        self.max_paddy_water_level = 0.05

        self.pump_hours = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well"
        ]["pump_hours"]
        self.probability_well_failure = self.model.config["agent_settings"][
            "expected_utility"
        ]["adaptation_well"]["probability_well_failure"]
        self.pump_horse_power = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well"
        ]["pump_horse_power"]
        self.proportion_irrigation_water_available = self.model.config[
            "agent_settings"
        ]["expected_utility"]["adaptation_well"][
            "proportion_irrigation_water_available"
        ]

        # New global well variables
        self.pump_hours = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well_global"
        ]["pump_hours"]
        self.depletion_limit = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well_global"
        ]["depletion_limit"]
        self.ponded_depth = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well_global"
        ]["ponded_depth"]
        self.specific_weight_water = self.model.config["agent_settings"][
            "expected_utility"
        ]["adaptation_well_global"]["specific_weight_water"]
        self.static_head = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well_global"
        ]["static_head"]
        self.max_initial_sat_thickness = self.model.config["agent_settings"][
            "expected_utility"
        ]["adaptation_well_global"]["max_initial_sat_thickness"]
        self.lifespan = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well_global"
        ]["lifespan"]
        self.well_diameter = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well_global"
        ]["well_diameter"]
        self.well_yield = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well_global"
        ]["well_yield"]
        self.pump_efficiency = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well_global"
        ]["pump_efficiency"]
        self.maintenance_factor = self.model.config["agent_settings"][
            "expected_utility"
        ]["adaptation_well_global"]["maintenance_factor"]
        self.WHY_10 = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well_global"
        ]["WHY_10"]
        self.WHY_20 = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well_global"
        ]["WHY_20"]
        self.WHY_30 = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well_global"
        ]["WHY_30"]

        # Placeholder values that will be changed by location specific values
        self.energy_cost_rate = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well_global"
        ]["energy_cost_rate"]
        self.aquifer_total_thickness = 1000  # Total acquifer thickness
        self.aquifer_porosity = 0.10
        self.aquifer_permeability = -13

        self.initial_groundwater_depth = 20

        Q_array_gpm = [
            10,
            20,
            30,
            40,
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1200,
            1300,
            1400,
            1500,
        ]

        # Convert candidate pumping rates to m^3/s
        self.Q_array = np.array(Q_array_gpm) / (60 * 264.17)

        self.p_droughts = np.array([50, 25, 10, 5, 2, 1])

        self.n_loans = 4

        self.elevation_subgrid = MapReader(
            fp=self.model.files["MERIT_grid"]["landsurface/topo/subgrid_elevation"],
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax,
        )
        self.elevation_grid = self.model.data.grid.compress(
            MapReader(
                fp=self.model.files["grid"]["landsurface/topo/elevation"],
                xmin=self.model.xmin,
                ymin=self.model.ymin,
                xmax=self.model.xmax,
                ymax=self.model.ymax,
            ).get_data_array()
        )

        with open(
            self.model.files["dict"]["agents/farmers/irrigation_sources"], "r"
        ) as f:
            self.irrigation_source_key = json.load(f)

        # load map of all subdistricts
        self.subdistrict_map = MapReader(
            fp=self.model.files["region_subgrid"]["areamaps/region_subgrid"],
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax,
        )

        self.crop_prices = load_regional_crop_data_from_dict(
            self.model, "crops/crop_prices"
        )
        self.cultivation_costs = load_regional_crop_data_from_dict(
            self.model, "crops/cultivation_costs"
        )

        self.total_spinup_time = (
            self.model.config["general"]["start_time"].year
            - self.model.config["general"]["spinup_time"].year
        )

        self.yield_ratio_multiplier_value = self.model.config["agent_settings"][
            "expected_utility"
        ]["adaptation_sprinkler"]["yield_multiplier"]

        self.var.actual_transpiration_crop = self.var.load_initial(
            "actual_transpiration_crop",
            default=self.var.full_compressed(0, dtype=np.float32, gpu=False),
            gpu=False,
        )
        self.var.potential_transpiration_crop = self.var.load_initial(
            "potential_transpiration_crop",
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

        # Load the region_code of each farmer.
        self.region_id = AgentArray(
            input_array=np.load(self.model.files["binary"]["agents/farmers/region_id"])[
                "data"
            ],
            max_n=self.max_n,
        )

        # Find the elevation of each farmer on the map based on the coordinates of the farmer as calculated before.
        self.elevation = AgentArray(
            input_array=self.elevation_subgrid.sample_coords(self.locations.data),
            max_n=self.max_n,
        )
        # Temporary well_unit_cost factor: change to values samples from map
        self.well_unit_cost = AgentArray(
            n=self.n,
            max_n=self.max_n,
            fill_value=self.WHY_10,
            dtype=np.float32,
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
        assert self.crop_calendar[:, :, 0].max() < len(self.crop_ids)

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
            fill_value=self.model.config["agent_settings"]["expected_utility"][
                "adaptation_well"
            ]["initial_depth"],
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
            extra_dims=(4,),
            extra_dims_names=("abstraction_type",),
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

        self.cumulative_yield_ratio = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.float32,
            fill_value=0,
        )
        self.cumulative_yield_ratio_count = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.int32,
            fill_value=0,
        )

        ## Base initial wealth on x days of daily expenses, sort of placeholder
        self.household_size = AgentArray(
            np.load(self.model.files["binary"]["agents/farmers/household_size"])[
                "data"
            ],
            max_n=self.max_n,
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

        self.yield_total_per_farmer = AgentArray(
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
            fill_value=self.model.config["agent_settings"]["expected_utility"][
                "drought_risk_calculations"
            ]["risk_perception"]["min"],
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
        self.irrigation_efficiency[:] = rng.uniform(0.50, 0.95, self.n)
        # Set the people who already have more van 90% irrigation efficiency to already adapted for the drip irrgation adaptation
        self.adapted[:, 2][self.irrigation_efficiency >= 0.90] = 1
        self.adaptation_mechanism[self.adapted[:, 2] == 1, 2] = 1
        # set the yield_ratio_multiplier to x of people who have drip irrigation, set to 1 for all others
        self.yield_ratio_multiplier[:] = np.where(
            (self.irrigation_efficiency >= 0.90) & (self.irrigation_source_key != 0),
            self.yield_ratio_multiplier_value,
            1,
        )
        self.base_management_yield_ratio = np.full(
            self.n,
            self.model.config["agent_settings"]["farmers"][
                "base_management_yield_ratio"
            ],
            dtype=np.float32,
        )

        # Increase yield ratio of those who use better management practices
        self.yield_ratio_management = (
            self.yield_ratio_multiplier * self.base_management_yield_ratio
        )

        rng_drip = np.random.default_rng(70)
        self.time_adapted[self.adapted[:, 2] == 1, 2] = rng_drip.uniform(
            1,
            self.lifespan,
            np.sum(self.adapted[:, 2] == 1),
        )

        # Initiate array that tracks the overall yearly costs for all adaptations
        # 0 is input, 1 is microcredit, 2 is adaptation 1 (well), 3 is adaptation 2 (drip irrigation), last is total
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
            fill_value=self.model.config["agent_settings"]["expected_utility"][
                "drought_risk_calculations"
            ]["risk_perception"]["min"],
        )
        self.risk_perc_max = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.float32,
            fill_value=self.model.config["agent_settings"]["expected_utility"][
                "drought_risk_calculations"
            ]["risk_perception"]["max"],
        )
        self.risk_decr = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.float32,
            fill_value=self.model.config["agent_settings"]["expected_utility"][
                "drought_risk_calculations"
            ]["risk_perception"]["coef"],
        )
        self.decision_horizon = AgentArray(
            n=self.n,
            max_n=self.max_n,
            dtype=np.int32,
            fill_value=self.model.config["agent_settings"]["expected_utility"][
                "decisions"
            ]["decision_horizon"],
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
        argsort_agend_ids = agent_ids_shuffled[activation_order_shuffled]
        # Return the agent ids ranks in the order of activation.
        ranks = np.empty_like(argsort_agend_ids)
        ranks[argsort_agend_ids] = np.arange(argsort_agend_ids.size)
        if self.model.config["agent_settings"]["fix_activation_order"]:
            self.activation_order_by_elevation_fixed = (self.n, ranks)
        return ranks

    @property
    def farmer_command_area(self):
        return farmer_command_area(
            self.n,
            self.field_indices,
            self.field_indices_by_farmer.data,
            self.var.reservoir_command_areas,
        )
        # output = np.full(self.n, -1, dtype=np.int32)
        # for farmer_i in range(self.n):
        #     farmer_fields = get_farmer_HRUs(
        #         self.field_indices, self.field_indices_by_farmer.data, farmer_i
        #     )
        #     for field in farmer_fields:
        #         command_area = self.var.reservoir_command_areas[field]
        #         if command_area != -1:
        #             output[farmer_i] = command_area
        #             break
        # return output

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
        HRU_to_grid: np.ndarray,
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
            self.has_access_to_irrigation_water,
            groundwater_depth_per_farmer,
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
            HRU_to_grid=HRU_to_grid,
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
            command_areas=command_areas,
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
                tollerance=1,
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

        self.groundwater_depth = AgentArray(
            groundwater_depth_per_farmer, max_n=self.max_n
        )
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
            else:
                raise ValueError(
                    f"Unknown crop data type: {self.crop_data_type}, must be 'GAEZ' or 'MIRCA2000'"
                )
            assert not np.isnan(yield_ratio).any()
        else:
            yield_ratio = np.full_like(crop_map[harvest], 1, dtype=np.float32)
        return yield_ratio

    def update_yield_ratio_management(self) -> None:
        # Increase yield ratio of those who use better management practices
        self.yield_ratio_management[:] = (
            self.yield_ratio_management * self.yield_ratio_multiplier
        )
        return None

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

        self.yield_total_per_farmer.fill(np.nan)
        self.harvested_crop.fill(-1)
        # If there are fields to be harvested, compute yield ratio and various related metrics
        if np.count_nonzero(harvest):
            # Get yield ratio for the harvested crops
            yield_ratio = self.get_yield_ratio(
                harvest,
                self.var.actual_transpiration_crop,
                self.var.potential_transpiration_crop,
                self.var.crop_map,
            )
            assert (yield_ratio >= 0).all()

            harvesting_farmer_fields = self.var.land_owners[harvest]
            harvested_area = self.var.cellArea[harvest]
            if self.model.use_gpu:
                harvested_area = harvested_area.get()

            harvested_crops = self.var.crop_map[harvest]
            max_yield_per_crop = np.take(
                self.crop_data["reference_yield_kg_m2"].values, harvested_crops
            )
            harvesting_farmers, index_farmer_to_field = np.unique(
                harvesting_farmer_fields, return_inverse=True
            )

            if self.crop_prices[0] is None:
                crop_prices = self.crop_prices[1]
                crop_price_per_field = np.full_like(
                    harvested_crops, crop_prices, dtype=np.float32
                )
            else:
                crop_prices = self.crop_prices[1][
                    self.crop_prices[0].get(self.model.current_time)
                ]
                assert not np.isnan(crop_prices).any()

                # Determine the region ids of harvesting farmers, as crop prices differ per region
                region_ids_harvesting_farmers = self.region_id[harvesting_farmers]

                # Calculate the crop price per field
                crop_prices_per_farmer = crop_prices[region_ids_harvesting_farmers]
                crop_prices_per_field = crop_prices_per_farmer[index_farmer_to_field]
                crop_price_per_field = np.take(crop_prices_per_field, harvested_crops)

            yield_ratio_total = (
                self.yield_ratio_management[harvesting_farmer_fields] * yield_ratio
            )

            yield_total_per_field = (
                yield_ratio_total * max_yield_per_crop * harvested_area
            )
            self.yield_total_per_farmer[:] = np.bincount(
                harvesting_farmer_fields,
                weights=yield_total_per_field,
                minlength=self.n,
            )

            # get the harvested crop per farmer. This assumes each farmer only harvests one crop
            # on the same day
            self.harvested_crop[harvesting_farmers] = harvested_crops[
                np.unique(self.var.land_owners[harvest], return_index=True)[1]
            ]

            # Determine the potential crop yield
            potential_profit_field = (
                harvested_area * max_yield_per_crop * crop_price_per_field
            )
            assert (potential_profit_field >= 0).all()

            # Determine the profit based on the crop yield in kilos and the price per kilo
            profit = potential_profit_field * yield_ratio_total
            assert (profit >= 0).all()

            # Convert from the profit and potential profit per field to the profit per farmer
            profit_farmer = np.bincount(
                harvesting_farmer_fields, weights=profit, minlength=self.n
            )
            potential_profit_farmer = np.bincount(
                harvesting_farmer_fields,
                weights=potential_profit_field,
                minlength=self.n,
            )

            ## Convert the yield_ratio per field to the average yield ratio per farmer
            yield_ratio_per_farmer = np.bincount(
                harvesting_farmer_fields, weights=yield_ratio_total, minlength=self.n
            ) / np.bincount(harvesting_farmer_fields, minlength=self.n)

            ## Get the current crop age
            crop_age = self.var.crop_age_days_map[harvest]
            total_crop_age = np.bincount(
                harvesting_farmer_fields, weights=crop_age, minlength=self.n
            ) / np.bincount(harvesting_farmer_fields, minlength=self.n)

            self.total_crop_age[harvesting_farmers, 0] = total_crop_age[
                harvesting_farmers
            ]

            harvesting_farmers_mask = np.zeros(self.n, dtype=bool)
            harvesting_farmers_mask[harvesting_farmers] = True

            # Update the cumulative yield ratio and count
            cumulative_mean(
                mean=self.cumulative_yield_ratio,
                counter=self.cumulative_yield_ratio_count,
                update=yield_ratio_per_farmer,
                mask=harvesting_farmers_mask,
            )

            self.save_yearly_profits(
                harvesting_farmers, profit_farmer, potential_profit_farmer
            )
            self.drought_risk_perception(harvesting_farmers, total_crop_age)

            ## After updating the drought risk perception, set the previous month for the next timestep as the current for this timestep.
            # TODO: This seems a bit like a quirky solution, perhaps there is a better way to do this.
            self.previous_month = self.model.current_time.month

        else:
            profit_farmer = np.zeros(self.n, dtype=np.float32)

        # Reset transpiration values for harvested fields
        self.var.actual_transpiration_crop[harvest] = 0
        self.var.potential_transpiration_crop[harvest] = 0

        # Update crop and land use maps after harvest
        self.var.crop_map[harvest] = -1
        self.var.crop_age_days_map[harvest] = -1
        self.var.land_use_type[harvest] = 1

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

        print("Risk perception mean = ", np.mean(self.risk_perception))

        # Determine which farmers need emergency microcredit to keep farming
        loaning_farmers = drought_loss_current >= self.moving_average_threshold

        # Determine their microcredit
        if (
            np.any(loaning_farmers)
            and "ruleset" in self.config
            and not self.config["ruleset"] == "no-adaptation"
        ):
            print(np.count_nonzero(loaning_farmers), "farmers are getting microcredit")
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
            annual_cost_microcredit=annual_cost_microcredit,
            loan_duration=loan_duration,
        )

        # Add it to the loan total
        self.all_loans_annual_cost[loaning_farmers, -1, 0] += annual_cost_microcredit

    @staticmethod
    @njit(cache=True)
    def set_loans_numba(
        all_loans_annual_cost: np.ndarray,
        loan_tracker: np.ndarray,
        loaning_farmers: np.ndarray,
        annual_cost_microcredit: np.ndarray,
        loan_duration: int,
    ) -> None:
        farmers_getting_loan = np.where(loaning_farmers)[0]

        # Update the agent's loans and total annual costs with the computed annual cost
        # Make sure it is in an empty loan slot
        for farmer in farmers_getting_loan:
            for i in range(4):
                if all_loans_annual_cost[farmer, 1, i] == 0:
                    local_index = np.where(farmers_getting_loan == farmer)[0][0]
                    all_loans_annual_cost[farmer, 1, i] += annual_cost_microcredit[
                        local_index
                    ]
                    loan_tracker[farmer, 1, i] = loan_duration
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
        self.var.crop_age_days_map[plant_map >= 0] = 0

        assert (self.var.crop_age_days_map[self.var.crop_map > 0] >= 0).all()

        self.var.land_use_type[
            (self.var.crop_map >= 0) & (self.field_is_paddy_irrigated)
        ] = 2
        self.var.land_use_type[
            (self.var.crop_map >= 0) & (~self.field_is_paddy_irrigated)
        ] = 3

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
        self.yearly_abstraction_m3_by_farmer[:, 0] += (
            self.channel_abstraction_m3_by_farmer
        )

        # Update yearly reservoir water abstraction for each farmer
        self.yearly_abstraction_m3_by_farmer[:, 1] += (
            self.reservoir_abstraction_m3_by_farmer
        )

        # Update yearly groundwater water abstraction for each farmer
        self.yearly_abstraction_m3_by_farmer[:, 2] += (
            self.groundwater_abstraction_m3_by_farmer
        )

        # Compute and update the total water abstraction for each farmer
        self.yearly_abstraction_m3_by_farmer[:, 3] += (
            self.channel_abstraction_m3_by_farmer
            + self.reservoir_abstraction_m3_by_farmer
            + self.groundwater_abstraction_m3_by_farmer
        )

    def cumulative_SPEI(self) -> None:
        """
        Update the monthly Standardized Precipitation Evapotranspiration Index (SPEI) array by shifting past records and
        adding the SPEI for the current month.

        Note:
            This method updates the `monthly_SPEI` attribute in place.
        """

        # Sample the SPEI value for the current month from `SPEI_map` based on the given locations.
        # The sampling is done for the first day of the current month.
        assert self.model.current_time.day == 1

        if self.model.current_time.month == 1:
            # calculate the SPEI probability using GEV parameters
            SPEI_probability = genextreme.sf(
                self.cumulative_SPEI_during_growing_season,
                self.GEV_parameters[:, 0],
                self.GEV_parameters[:, 1],
                self.GEV_parameters[:, 2],
            )

            shift_and_update(self.yearly_SPEI_probability, SPEI_probability)
            shift_and_update(self.yearly_yield_ratio, self.cumulative_yield_ratio)

            # Reset the cumulative SPEI array at the beginning of the year
            self.cumulative_SPEI_during_growing_season.fill(0)
            self.cumulative_SPEI_count_during_growing_season.fill(0)
            self.cumulative_yield_ratio.fill(0)
            self.cumulative_yield_ratio_count.fill(0)

        fields_with_growing_crops = self.var.crop_map[self.var.land_owners != -1] != -1
        farmers_with_growing_crops = (
            np.bincount(
                self.var.land_owners[self.var.land_owners != -1],
                weights=fields_with_growing_crops,
            )
            > 0
        )
        if farmers_with_growing_crops.sum() == 0:
            return

        current_SPEI_per_farmer = sample_from_map(
            array=self.model.data.grid.spei_uncompressed,
            coords=self.locations[farmers_with_growing_crops].data,
            gt=self.model.data.grid.gt,
        )

        full_size_SPEI_per_farmer = np.zeros_like(
            self.cumulative_SPEI_during_growing_season
        )
        full_size_SPEI_per_farmer[farmers_with_growing_crops] = current_SPEI_per_farmer

        cumulative_mean(
            mean=self.cumulative_SPEI_during_growing_season,
            counter=self.cumulative_SPEI_count_during_growing_season,
            update=full_size_SPEI_per_farmer,
            mask=farmers_with_growing_crops,
        )

    def save_yearly_profits(
        self,
        harvesting_farmers: np.ndarray,
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

        latest_profits = profit[harvesting_farmers]
        latest_potential_profits = potential_profit[harvesting_farmers]

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
        self.yearly_profits[harvesting_farmers, 0] += (
            latest_profits / cum_inflation[harvesting_farmers]
        )
        self.yearly_potential_profits[harvesting_farmers, 0] += (
            latest_potential_profits / cum_inflation[harvesting_farmers]
        )

    def calculate_yield_spei_relation(self) -> None:
        """
        Computes the yearly yield ratios and SPEI probabilities, then calculates the yearly mean for each unique farmer type.

        Note:
            This function performs the following operations:
                1. Group farmers based on crop combinations and compute averages for each group.
                2. Mask rows and columns with only zeros.
                3. Determine the relation between yield ratio and profit for all farmer types.
                4. Sample the individual agent relation from the agent groups and assign to agents.
        """
        # Step 1: Group farmers based on crop combinations and location in basin and compute averages
        unique_yearly_yield_ratio = np.empty(
            (0, self.total_spinup_time), dtype=np.float32
        )
        unique_SPEI_probability = np.empty(
            (0, self.total_spinup_time), dtype=np.float32
        )

        # Create unique groups
        # Calculating the thresholds for the top, middle, and lower thirds
        basin_elevation_thresholds = np.percentile(
            self.elevation.data, [100 / 3, 100 / 1.5]
        )
        # 0 for upper, 1 for mid, and 2 for lower
        distribution_array = np.zeros_like(self.elevation)
        distribution_array[self.elevation > basin_elevation_thresholds[1]] = 0  # Upper
        distribution_array[
            (self.elevation > basin_elevation_thresholds[0])
            & (self.elevation <= basin_elevation_thresholds[1])
        ] = 1  # Mid
        distribution_array[self.elevation <= basin_elevation_thresholds[0]] = 2  # Lower

        crop_elevation_group = np.hstack(
            (
                self.crop_calendar[:, :, 0],
                distribution_array.reshape(-1, 1),
                self.adapted[:, 1].reshape(-1, 1),
            )
        )

        for crop_combination in np.unique(crop_elevation_group, axis=0):
            unique_farmer_groups = np.where(
                (crop_elevation_group == crop_combination[None, ...]).all(axis=1)
            )[0]
            average_yield_ratio = np.mean(
                self.yearly_yield_ratio[unique_farmer_groups, :], axis=0
            )
            average_probability = np.mean(
                self.yearly_SPEI_probability[unique_farmer_groups, :], axis=0
            )
            unique_yearly_yield_ratio = np.vstack(
                (unique_yearly_yield_ratio, average_yield_ratio)
            )
            unique_SPEI_probability = np.vstack(
                (unique_SPEI_probability, average_probability)
            )

        # Step 2: Mask rows and columns with zeros
        mask_rows = np.any((unique_yearly_yield_ratio != 0), axis=1) & np.any(
            (unique_SPEI_probability != 0), axis=1
        )
        if np.any([~mask_rows]):
            # Sometimes very few farmer (groups) (1 in a million) get yield ratios of only 0
            # If so, give it the mean of all groups for both the spei and yield ratio
            unique_yearly_yield_ratio[~mask_rows] = np.mean(
                unique_yearly_yield_ratio, axis=0
            )
            unique_SPEI_probability[~mask_rows] = np.mean(
                unique_SPEI_probability, axis=0
            )

        mask_columns = np.any((unique_yearly_yield_ratio != 0), axis=0) & np.any(
            (unique_SPEI_probability != 0), axis=0
        )
        unique_yearly_yield_ratio_mask = unique_yearly_yield_ratio[:, mask_columns]
        unique_SPEI_probability_mask = unique_SPEI_probability[:, mask_columns]

        # Step 3: Determine the relation between yield ratio and profit
        group_yield_probability_relation_log = []
        yield_probability_R2_log = []
        yield_probability_p_scipy = []

        # Variables to store the last yield_ratio and spei_prob
        last_yield_ratio = None
        last_spei_prob = None

        def logarithmic_function(x, a, b):
            return a * np.log2(x) + b

        for idx, (yield_ratio, spei_prob) in enumerate(
            zip(unique_yearly_yield_ratio_mask, unique_SPEI_probability_mask)
        ):
            # Filter out zeros, some agents are nearly always at 0 yield ratio
            # This is a problem for the fitting (and likely an outlier)
            mask = (yield_ratio != 0) | (spei_prob != 0)
            yield_ratio = yield_ratio[mask]
            spei_prob = spei_prob[mask]

            # Set the a and b values of last year to prevent no values on this year
            if not ((self.farmer_yield_probability_relation == None).all()):
                a, b = np.median(
                    self.farmer_yield_probability_relation[
                        np.where(
                            (
                                crop_elevation_group
                                == np.unique(crop_elevation_group, axis=0)[idx]
                            ).all(axis=1)
                        )[0]
                    ],
                    axis=0,
                )
            else:
                a, b = 2, 3

            # Fit logarithmic function, except when there is an error
            try:
                # Attempt to fit the logarithmic_function function
                a, b = curve_fit(logarithmic_function, yield_ratio, spei_prob)[0]

            except RuntimeError:
                # RuntimeError is raised when curve_fit fails to converge
                # In this case, take the values of the previous (similar) group
                if last_yield_ratio is not None:
                    yield_ratio = last_yield_ratio
                    spei_prob = last_spei_prob
                    # Recalculate a, b with the previous values
                    a, b = curve_fit(
                        logarithmic_function, last_yield_ratio, last_spei_prob
                    )[0]

            group_yield_probability_relation_log.append(np.array([a, b]))

            residuals = spei_prob - logarithmic_function(yield_ratio, a, b)
            ss_tot = np.sum((spei_prob - np.mean(spei_prob)) ** 2)
            ss_res = np.sum(residuals**2)

            yield_probability_R2_log.append(1 - (ss_res / ss_tot))

            # Update last_yield_ratio and last_spei_prob
            last_yield_ratio = yield_ratio
            last_spei_prob = spei_prob

        # Assign relations to agents
        exact_positions = np.where(
            np.all(
                crop_elevation_group[:, np.newaxis, :]
                == np.unique(crop_elevation_group, axis=0),
                axis=-1,
            )
        )[1]
        if len(group_yield_probability_relation_log) > max(exact_positions):
            self.farmer_yield_probability_relation = np.array(
                group_yield_probability_relation_log
            )[exact_positions]
            assert isinstance(
                self.farmer_yield_probability_relation, np.ndarray
            ), "self.farmer_yield_probability_relation must be a np.ndarray"

        print(
            "r2_log:",
            np.median(yield_probability_R2_log),
            "p:",
            np.median(yield_probability_p_scipy),
        )

    def adapt_drip_irrigation(self) -> None:
        """
        Handle the adaptation of farmers to drip irrigation systems.

        This function checks which farmers will adopt drip irrigation based on their expected utility
        and the provided constraints. It calculates the costs and the benefits for each farmer and updates
        their statuses (e.g., irrigation efficiency, adaptation costs, yield ratio multiplier) accordingly.

        Note:

        TODO:
            - Possibly externalize hard-coded values.
        """
        # Constants
        ADAPTATION_TYPE = 2
        ACRE_TO_SQUARE_METER = 0.000247105

        # Compute total adaptation cost for each farmer
        per_farmer_cost = self.get_value_per_farmer_from_region_id(
            self.drip_irrigation_price, self.model.current_time
        )
        total_cost = (
            self.field_size_per_farmer
            * ACRE_TO_SQUARE_METER
            * np.full(self.n, per_farmer_cost, dtype=np.float32)
        )

        # Fetch loan configuration
        loan_duration = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_sprinkler"
        ]["loan_duration"]

        # Calculate annual cost based on the interest rate and loan duration
        annual_cost = total_cost * (
            self.interest_rate
            * (1 + self.interest_rate) ** loan_duration
            / ((1 + self.interest_rate) ** loan_duration - 1)
        )

        # Fetch lifespan of the adaptation
        lifespan_adaptation = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_sprinkler"
        ]["lifespan"]

        # Reset farmers' status who exceeded the lifespan of their adaptation
        expired_adaptations = (
            self.time_adapted[:, ADAPTATION_TYPE] == lifespan_adaptation
        )
        self.adaptation_mechanism[expired_adaptations, ADAPTATION_TYPE] = 0
        self.adapted[expired_adaptations, ADAPTATION_TYPE] = 0
        self.time_adapted[expired_adaptations, ADAPTATION_TYPE] = -1
        self.irrigation_efficiency[expired_adaptations] = 0.50
        self.yield_ratio_multiplier[expired_adaptations] = 1

        # Define extra constraints (farmers must have an irrigation source)
        extra_constraint = self.irrigation_source != 0

        # Get the mask of farmers who will adapt
        adaptation_mask = self.adapt_SEUT(
            ADAPTATION_TYPE, annual_cost, loan_duration, extra_constraint
        )

        # Update irrigation efficiency and yield multiplier for farmers who adapted
        self.irrigation_efficiency[adaptation_mask] = 0.90
        self.yield_ratio_multiplier[adaptation_mask] = self.yield_ratio_multiplier_value

        # Update annual costs
        self.all_loans_annual_cost[adaptation_mask, ADAPTATION_TYPE + 1, 0] += (
            annual_cost[adaptation_mask]
        )  # For drip irrigation specifically
        self.all_loans_annual_cost[adaptation_mask, -1, 0] += annual_cost[
            adaptation_mask
        ]  # Total loan costs

        # set loan tracker
        self.loan_tracker[adaptation_mask, ADAPTATION_TYPE + 1, 0] += loan_duration

        # Print the percentage of adapted households
        percentage_adapted = round(
            np.sum(self.adapted[:, ADAPTATION_TYPE])
            / len(self.adapted[:, ADAPTATION_TYPE])
            * 100,
            2,
        )
        print("Sprinkler irrigation farms:", percentage_adapted, "(%)")

    def adapt_irrigation_well(self) -> None:
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
        ADAPTATION_TYPE = 1

        # Compute total adaptation cost for each farmer
        fixed_investment_cost, yearly_costs, well_depth = (
            self.calculate_well_costs_india()
        )

        # Fetch loan configuration
        loan_duration = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well"
        ]["loan_duration"]

        # Calculate annual cost based on the interest rate and loan duration
        annual_cost = (
            fixed_investment_cost
            * (
                self.interest_rate
                * (1 + self.interest_rate) ** loan_duration
                / ((1 + self.interest_rate) ** loan_duration - 1)
            )
            + yearly_costs
        )

        # Reset farmers' status and irrigation type who exceeded the lifespan of their adaptation
        # and who's wells are much shallower than the groundwater depth
        expired_adaptations = (
            self.time_adapted[:, ADAPTATION_TYPE] == self.lifespan
        ) | (self.groundwater_depth > self.well_depth)
        self.adaptation_mechanism[expired_adaptations, ADAPTATION_TYPE] = 0
        self.adapted[expired_adaptations, ADAPTATION_TYPE] = 0
        self.time_adapted[expired_adaptations, ADAPTATION_TYPE] = -1
        self.irrigation_source[expired_adaptations] = self.irrigation_source_key["no"]

        # Define extra constraints (farmers' wells must reach groundwater)
        well_reaches_groundwater = self.well_depth > self.groundwater_depth
        extra_constraint = well_reaches_groundwater

        # To determine the benefit of irrigation, those who have a well are adapted
        adapted = np.where((self.farmer_class == 2), 1, 0)

        # Get the mask of farmers who will adapt
        adaptation_mask = self.adapt_SEUT(
            ADAPTATION_TYPE, annual_cost, loan_duration, extra_constraint, adapted
        )

        # Update irrigation source for farmers who adapted
        self.irrigation_source[adaptation_mask] = self.irrigation_source_key["well"]

        # Set their well depth
        self.well_depth[adaptation_mask] = well_depth[adaptation_mask]

        # Update annual costs and disposable income for adapted farmers
        self.all_loans_annual_cost[adaptation_mask, ADAPTATION_TYPE + 1, 0] += (
            annual_cost[adaptation_mask]
        )  # For wells specifically
        self.all_loans_annual_cost[adaptation_mask, -1, 0] += annual_cost[
            adaptation_mask
        ]  # Total loan amount

        # set loan tracker
        self.loan_tracker[adaptation_mask, ADAPTATION_TYPE + 1, 0] += loan_duration

        # Print the percentage of adapted households
        percentage_adapted = round(
            np.sum(self.adapted[:, ADAPTATION_TYPE])
            / len(self.adapted[:, ADAPTATION_TYPE])
            * 100,
            2,
        )
        print("Irrigation well farms:", percentage_adapted, "(%)")

    def calculate_well_costs_global(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Replace by actual data derived initial groundwater depth
        initial_sat_thickness = (
            self.aquifer_total_thickness - self.initial_groundwater_depth
        )
        current_sat_thickness = self.aquifer_total_thickness - self.groundwater_depth

        potential_well_length = np.full_like(
            self.groundwater_depth,
            self.max_initial_sat_thickness + self.groundwater_depth,
        )

        # Set the well length depending on the current acquifer thickness
        well_length_mask = current_sat_thickness > self.max_initial_sat_thickness
        potential_well_length[well_length_mask] = self.aquifer_total_thickness

        depleted_volume_fraction = current_sat_thickness / initial_sat_thickness

        # Set mask for all those who have not reached the depletion limit yet
        above_depletion_limit_mask = (
            1 - depleted_volume_fraction
        ) < self.depletion_limit

        ## Determine costs
        install_cost = self.well_unit_cost * potential_well_length

        maintenance_cost = self.maintenance_factor * install_cost

        # Determine the total number of hours that the pump is running
        total_pump_duration = np.mean(self.total_crop_age, axis=1)

        # TODO: Correct that this is not used? If correct, remove
        # # Calculate total hours per year that the pump is active
        # total_pumping_hours_yearly = self.pump_hours * total_pump_duration

        # TO DO: update well Q array: Determine through how much agents abstract per person, could be an agent array
        initial_Q = np.mean(self.Q_array)
        power = (
            self.specific_weight_water
            * self.groundwater_depth
            * initial_Q
            / self.pump_efficiency
        ) / 1000

        energy = power * (total_pump_duration * self.pump_hours)  # kWh
        energy_cost_rate = self.energy_cost_rate  # $ per kWh
        energy_cost = energy * energy_cost_rate  # $/year

        # Fetch loan configuration
        loan_duration = self.model.config["agent_settings"]["expected_utility"][
            "adaptation_well"
        ]["loan_duration"]

        # Calculate annual cost based on the interest rate and loan duration
        annual_cost = (
            install_cost
            * (
                self.interest_rate
                * (1 + self.interest_rate) ** loan_duration
                / ((1 + self.interest_rate) ** loan_duration - 1)
            )
            + energy_cost
            + maintenance_cost
        )

        return (
            annual_cost,
            potential_well_length,
            above_depletion_limit_mask,
        )

    def calculate_well_costs_india(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the construction and yearly costs of well and pump for irrigation, based on various parameters
        such as well depth, crop growth seasons, pump horsepower, and regional costs. The function considers
        different seasonal durations, well failure probabilities, and irrigation efficiency. Based on Robert, M.,
        Bergez, J. E., & Thomas, A. (2018). A stochastic dynamic programming approach to analyze adaptation to climate change â€“
        Application to groundwater irrigation in India. European Journal of Operational Research, 265(3), 1033â€“1045.
        https://doi.org/10.1016/j.ejor.2017.08.029

        Returns:
        - fixed_investment_cost (np.ndarray): An array representing the fixed construction costs of wells and pumps.
        - yearly_costs (np.ndarray): An array representing the yearly operational costs for irrigation maintenance
        and groundwater pumping.

        Notes:
        - `self.n` is assumed to be the number of farmers or irrigation units.
        - `self.get_value_per_farmer_from_region_id` is a method to retrieve cost data for the current time step.
        - All costs are calculated per farmer/irrigation unit.
        """

        # Initialize costs arrays with region-specific values
        borewell_cost_1 = np.full(
            self.n,
            self.get_value_per_farmer_from_region_id(
                self.borewell_cost_1, self.model.current_time
            ),
            dtype=np.float32,
        )
        borewell_cost_2 = np.full(
            self.n,
            self.get_value_per_farmer_from_region_id(
                self.borewell_cost_2, self.model.current_time
            ),
            dtype=np.float32,
        )
        pump_cost = np.full(
            self.n,
            self.get_value_per_farmer_from_region_id(
                self.pump_cost, self.model.current_time
            ),
            dtype=np.float32,
        )
        irrigation_maintenance = np.full(
            self.n,
            self.get_value_per_farmer_from_region_id(
                self.irrigation_maintenance, self.model.current_time
            ),
            dtype=np.float32,
        )
        electricity_costs = np.full(
            self.n,
            self.get_value_per_farmer_from_region_id(
                self.electricity_cost, self.model.current_time
            ),
            dtype=np.float32,
        )

        # Calculate new pump depth as a function of groundwater depth
        new_pump_depth = self.groundwater_depth + 20

        # Create a mask for valid crop indices
        crops_mask = (self.crop_calendar[:, :, 0] >= 0) & (
            self.crop_calendar[:, :, 0] < len(self.crop_data["season_#1_duration"])
        )
        nan_array = np.full_like(
            self.crop_calendar[:, :, 0], fill_value=np.nan, dtype=float
        )

        # Set the total crop grow time for each season
        season_selection = [
            "season_#1_duration",
            "season_#2_duration",
            "season_#3_duration",
        ]

        # Initialize an array to hold total growth length per agent
        seasons_total = nan_array.copy()

        for i, season_col in enumerate(season_selection):
            season_x_duration = nan_array.copy()
            season_x_duration[crops_mask] = np.take(
                self.crop_data[season_col].values,
                self.crop_calendar[:, :, 0][crops_mask].astype(int),
            )
            seasons_total[:, i] = season_x_duration[:, i]

        total_pump_duration = np.nansum(seasons_total, axis=1)

        # Calculate total hours per year that the pump is active
        total_pumping_hours_yearly = self.pump_hours * total_pump_duration

        # Calculate the electric power in kilowatt for irrigation
        electric_power_irrigation = 0.7457 * self.pump_horse_power

        # Calculate yearly groundwater pumping cost
        groundwater_pumping_cost_yearly = (
            total_pumping_hours_yearly
            * electricity_costs
            * electric_power_irrigation
            * self.proportion_irrigation_water_available
        )

        # Calculate the construction cost of the well and pump
        construction_cost_well = (1 + 100 * self.probability_well_failure) * (
            borewell_cost_1 * new_pump_depth - borewell_cost_2 * new_pump_depth**2
        )
        pump_cost = pump_cost * self.pump_horse_power

        # Calculate the irrigation maintenance costs
        flow_rate = 79.93 * self.groundwater_depth**-0.728
        expected_water_availability = flow_rate * total_pumping_hours_yearly
        irrigation_maintenance_costs = (
            irrigation_maintenance * expected_water_availability**0.16
        )

        # Determine fixed and yearly costs
        fixed_investment_cost = construction_cost_well + pump_cost
        yearly_costs = irrigation_maintenance_costs + groundwater_pumping_cost_yearly

        return fixed_investment_cost, yearly_costs, new_pump_depth

    def adapt_SEUT(
        self,
        adaptation_type,
        annual_cost,
        loan_duration,
        extra_constraint,
        adapted: np.ndarray = None,
    ):
        """
        Determine if farmers should adapt based on comparing the Expected Utility (EU) of adapting vs. doing nothing.

        The function considers both individual farmers' decisions and the potential influence
        of neighboring farmers' decisions. Adaptation decisions are based on Subjective Expected Utility Theory (SEUT).

        Parameters:
        - adaptation_type: Type of adaptation under consideration.
        - annual_cost: Annual cost for this adaptation type.
        - loan_duration: Duration of the loan.
        - extra_constraint: Additional constraint (may be specific to certain adaptation types).

        Returns:
        - adaptation_mask: Boolean array indicating which farmers decided to adapt.
        """
        if adapted is None:
            adapted = self.adapted[:, adaptation_type]

        # Calculate profits considering different scenarios (with/without adaptation and with/without drought events)
        (
            total_profits,
            total_profits_adaptation,
            profits_no_event,
            profits_no_event_adaptation,
        ) = self.profits_SEUT(adaptation_type, adapted)

        # Calculate the farm area per agent
        farmer_fields_ID = self.var.land_owners
        farm_area = np.bincount(
            farmer_fields_ID[farmer_fields_ID != -1],
            weights=self.var.cellArea[farmer_fields_ID != -1],
            minlength=self.n,
        )

        # Compute the total annual per square meter costs if farmers adapt during this cycle
        # This cost is the cost if the farmer would adapt, plus its current costs of previous
        # adaptations
        total_annual_costs_m2 = (
            annual_cost + self.all_loans_annual_cost[:, -1, 0]
        ) / farm_area

        # Solely the annual cost of the adaptation
        annual_cost_m2 = annual_cost / farm_area

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
            "risk_perception": self.risk_perception.data,
            "total_annual_costs": total_annual_costs_m2,
            "adaptation_costs": annual_cost_m2,
            "adapted": adapted,
            "time_adapted": self.time_adapted[:, adaptation_type],
            "T": np.full(
                self.n,
                self.model.config["agent_settings"]["expected_utility"][
                    "adaptation_well"
                ]["decision_horizon"],
            ).data,
            "discount_rate": self.discount_rate.data,
            "extra_constraint": extra_constraint.data,
        }

        # Calculate the EU of not adapting and adapting respectively
        SEUT_do_nothing = self.decision_module.calcEU_do_nothing(**decision_params)
        SEUT_adapt = self.decision_module.calcEU_adapt(**decision_params)

        # Ensure valid EU values
        assert (SEUT_do_nothing != -1).any or (SEUT_adapt != -1).any()

        # Compare EU values for those who haven't adapted yet and get boolean results
        SEUT_adaptation_decision = (
            SEUT_adapt[adapted == 0] > SEUT_do_nothing[adapted == 0]
        )

        # Initialize a mask with default value as False
        SEUT_adapt_mask = np.zeros_like(adapted, dtype=bool)

        # Update the mask based on EU decisions
        SEUT_adapt_mask[adapted == 0] = SEUT_adaptation_decision
        self.adaptation_mechanism[SEUT_adapt_mask == 1, adaptation_type] = 2

        # Get the final decision mask considering individual and neighbor influences
        adaptation_mask = SEUT_adapt_mask

        # Update the adaptation status
        self.adapted[adaptation_mask, adaptation_type] = 1

        # Reset the timer for newly adapting farmers and update timers for others
        self.time_adapted[adaptation_mask, adaptation_type] = 0
        self.time_adapted[
            self.time_adapted[:, adaptation_type] != -1, adaptation_type
        ] += 1

        return adaptation_mask

    def profits_SEUT(
        self, adaptation_type, adapted: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate and compare the profits under different drought scenarios and adaptation strategies.

        Args:
            adaptation_type (int): The type of adaptation being considered. 0 means no adaptation, then it is only used
            to calculate the no_adaptation profits

        Returns:
            total_profits (np.ndarray): Profits for each probability scenario without adaptation.
            total_profits_adaptation (np.ndarray): Profits for each probability scenario with the given adaptation.
            profits_no_event (np.ndarray): Profits for the scenario without a drought event (probability 1/1).
            profits_no_event_adaptation (np.ndarray): Profits for the scenario without a drought event with adaptation.

        This function evaluates the impact of an adaptation strategy on farmers' profits. It does this by simulating
        the profits for different drought probability scenarios, both with and without the adaptation strategy.
        """

        if adapted is None:
            adapted = self.adapted[:, adaptation_type]

        # Copy the yield ratios during drought events
        yield_ratios = self.convert_probability_to_yield_ratio()

        if adaptation_type != 0:
            # Compute the yield ratios when an adaptation strategy is applied
            gains_adaptation = self.adaptation_yield_ratio_difference(adapted)

            yield_ratios_adaptation = yield_ratios * gains_adaptation[:, None]
            # print(np.median(yield_ratios_adaptation), 'with adaptation', np.median(yield_ratios), 'without adaptation')

            # Ensure yield ratios do not exceed 1
            yield_ratios_adaptation[yield_ratios_adaptation > 1] = 1

            # Initialize profit matrices for adaptation
            total_profits_adaptation = np.zeros((self.n, len(self.p_droughts)))

        # Initialize profit matrices without adaptation
        total_profits = np.zeros((self.n, len(self.p_droughts)))

        # Mask out all non-crops in the crops array
        crops_mask = (self.crop_calendar[:, :, 0] >= 0) & (
            self.crop_calendar[:, :, 0] < len(self.crop_data["reference_yield_kg_m2"])
        )

        # Output array with NaNs for storing reference data
        nan_array = np.full_like(
            self.crop_calendar[:, :, 0], fill_value=np.nan, dtype=float
        )

        # Compute profits for each probability scenario, considering the adaptation impact
        for col in range(yield_ratios.shape[1]):
            total_profits[:, col] = self.yield_ratio_to_profit(
                yield_ratios[:, col], crops_mask, nan_array
            )

            if adaptation_type != 0:
                total_profits_adaptation[:, col] = self.yield_ratio_to_profit(
                    yield_ratios_adaptation[:, col], crops_mask, nan_array
                )

        # Transpose matrices to match the expected format
        total_profits = total_profits.T[:-1, :]

        # Extract profits for the "no drought" event scenario
        profits_no_event = total_profits[-1, :]

        # Do the same for with adaptation if required
        if adaptation_type != 0:
            total_profits_adaptation = total_profits_adaptation.T[:-1, :]
            profits_no_event_adaptation = total_profits_adaptation[-1, :]
            return (
                total_profits,
                total_profits_adaptation,
                profits_no_event,
                profits_no_event_adaptation,
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

        def inverse_logarithmic_function(probability, params):
            a = params[:, 0]
            b = params[:, 1]

            return np.power(2, (probability[:, np.newaxis] - b) / a)

        yield_ratios = inverse_logarithmic_function(
            1 / self.p_droughts, self.farmer_yield_probability_relation
        ).T

        # Adjust the yield ratios to lie between 0 and 1
        yield_ratios[yield_ratios < 0] = 0  # Ensure non-negative yield ratios
        yield_ratios[yield_ratios > 1] = 1  # Cap the yield ratios at 1

        # Store the results in a global variable
        self.yield_ratios_drought_event = yield_ratios[:]

        return self.yield_ratios_drought_event

    def adaptation_yield_ratio_difference(self, adapted: np.ndarray) -> np.ndarray:
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
        # Create unique groups
        # Calculating the thresholds for the top, middle, and lower thirds
        basin_elevation_thresholds = np.percentile(self.elevation.data, [33.33, 66.67])
        # 0 for upper, 1 for mid, and 2 for lower
        distribution_array = np.zeros_like(self.elevation)
        distribution_array[self.elevation > basin_elevation_thresholds[1]] = 0  # Upper
        distribution_array[
            (self.elevation > basin_elevation_thresholds[0])
            & (self.elevation <= basin_elevation_thresholds[1])
        ] = 1  # Mid
        distribution_array[self.elevation <= basin_elevation_thresholds[0]] = 2  # Lower

        crop_elevation_group = np.hstack(
            (
                self.crop_calendar[:, :, 0],
                distribution_array.reshape(-1, 1),
            )
        )
        # Add a column of zeros to represent farmers who have not adapted yet
        crop_groups_onlyzeros = np.hstack(
            (crop_elevation_group, np.zeros(self.n).reshape(-1, 1))
        )

        # Combine current crops with their respective adaptation status
        crop_groups = np.hstack((crop_elevation_group, adapted.reshape(-1, 1)))

        # Initialize array to store relative yield ratio improvement for unique groups
        unique_yield_ratio_gain_relative = np.full(
            len(np.unique(crop_groups_onlyzeros, axis=0)), 1, dtype=np.float32
        )

        # unique_yield_ratio_gain_relative_2 = np.full((len(np.unique(crop_groups_onlyzeros, axis=0)), 6), 1, dtype=np.float32)

        # Loop over each unique group of farmers to determine their average yield ratio
        for idx, unique_combination in enumerate(
            np.unique(crop_groups_onlyzeros, axis=0)
        ):
            unique_farmer_groups = (crop_groups == unique_combination[None, ...]).all(
                axis=1
            )

            # Identify the adapted counterpart of the current group
            unique_combination_adapted = unique_combination.copy()
            unique_combination_adapted[-1] = 1
            unique_farmer_groups_adapted = (
                crop_groups == unique_combination_adapted[None, ...]
            ).all(axis=1)

            if (
                np.count_nonzero(unique_farmer_groups) != 0
                and np.count_nonzero(unique_farmer_groups_adapted) != 0
            ):
                # Calculate mean yield ratio over past years for both adapted and unadapted groups
                unadapted_yield_ratio = np.mean(
                    self.yearly_yield_ratio[unique_farmer_groups, :5], axis=1
                )
                adapted_yield_ratio = np.mean(
                    self.yearly_yield_ratio[unique_farmer_groups_adapted, :5], axis=1
                )

                unadapted_median = np.median(unadapted_yield_ratio)
                adapted_median = np.median(adapted_yield_ratio)

                # add a small value to prevent division by 0
                adapted_value = adapted_median + 0.0001
                unadapted_value = unadapted_median + 0.0001

                yield_ratio_gain_relative = adapted_value / unadapted_value
                # yield_ratio_gain_relative_2 = np.median(yield_ratios[unique_farmer_groups_adapted, :], axis=0) / np.median(yield_ratios[unique_farmer_groups, :], axis=0)
                # unique_yield_ratio_gain_relative_2[idx] = yield_ratio_gain_relative_2
                # Determine the size of adapted group relative to unadapted group
                adapted_unadapted_ratio = min(
                    adapted_yield_ratio.size / unadapted_yield_ratio.size, 1.0
                )

                # Add to results depending on relative group sizes and random chance
                if np.random.rand() < (adapted_unadapted_ratio + 0.25):
                    unique_yield_ratio_gain_relative[idx] = yield_ratio_gain_relative

        # Identify each agent's position within the unique groups
        positions_agent = np.where(
            np.all(
                crop_groups_onlyzeros[:, np.newaxis, :]
                == np.unique(crop_groups_onlyzeros, axis=0),
                axis=-1,
            )
        )
        exact_position = positions_agent[1]

        # Convert group-based results into agent-specific results
        gains_adaptation = unique_yield_ratio_gain_relative[exact_position]
        assert np.max(gains_adaptation) != np.inf, "gains adaptation value is inf"

        return gains_adaptation

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

        total_price = 0
        month_count = 0

        # Ending date and start date set to one year prior
        end_date = self.model.current_time
        start_date = datetime(end_date.year - 1, 1, 1)

        # Loop through each month from start_date to end_date to get the sum of crop costs over the past year
        current_date = start_date
        while current_date <= end_date:
            assert self.crop_prices[0] is not None, "behavior needs crop prices to work"
            monthly_price = self.crop_prices[1][self.crop_prices[0].get(current_date)]
            total_price += monthly_price
            # Move to the next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
            month_count += 1

        # Calculate the average price over the last year
        average_monthly_price = total_price / month_count
        assert not np.isnan(
            average_monthly_price
        ).any()  # Ensure there are no NaN values in crop prices

        # Assign the reference yield and current crop price to the array based on valid crop mask
        array_with_price[crops_mask] = np.take(
            average_monthly_price, self.crop_calendar[:, :, 0][crops_mask].astype(int)
        )
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

    def compare_neighbor_EUT(
        self,
        EUT_do_nothing,
        SEUT_adapt,
        adapted,
        expenditure_cap,
        total_annual_costs,
        profits_no_event,
        extra_constraint,
    ):
        """
        Compares the Expected Utility Theory (EUT) of agents to that of their adapted neighbors and make a decision on adaptation.

        Parameters:
        - EUT_do_nothing: Expected utility of not adapting.
        - SEUT_adapt: Expected utility of adapting.
        - expenditure_cap: Expenditure capability.
        - total_annual_costs: Total annual costs for adaptation.
        - profits_no_event: Profits without an event.
        - extra_constraint: Additional constraints for adaptation.

        Returns:
        - numpy.ndarray: Boolean mask of farmers who decided to adapt.

        Note:

        """

        # Constants
        NBITS = 19
        RADIUS = 5_000
        N_NEIGHBOR = 10

        # Initialize investment decisions as a zero-filled boolean array
        invest_in_adaptation = np.zeros(self.n, dtype=np.bool_)

        # Iterate over unique crop options
        for crop_option in np.unique(self.crop_calendar[:, :, 0], axis=0):
            farmers_with_crop_option = np.where(
                (self.crop_calendar[:, :, 0] == crop_option[None, ...]).all(axis=1)
            )[0]

            # Create filters for adapted and non-adapted farmers
            farmers_not_adapted = np.where(
                (
                    profits_no_event[farmers_with_crop_option] * expenditure_cap
                    > total_annual_costs[farmers_with_crop_option]
                )
                & (adapted[farmers_with_crop_option] == 0)
                & extra_constraint[farmers_with_crop_option]
            )
            farmers_adapted = adapted[farmers_with_crop_option] == 1

            # Map local to global indices
            local_indices = np.arange(len(farmers_with_crop_option))
            global_indices_adapted = farmers_with_crop_option[
                local_indices[farmers_adapted]
            ]
            global_indices_not_adapted = farmers_with_crop_option[
                local_indices[farmers_not_adapted]
            ]
            # Check for neighbors with adaptations for non-adapted farmers
            if global_indices_not_adapted.size > 0 and global_indices_adapted.size > 0:
                neighbors_with_adaptation = find_neighbors(
                    self.locations.data,
                    radius=RADIUS,
                    n_neighbor=N_NEIGHBOR,
                    bits=NBITS,
                    minx=self.model.bounds[0],
                    maxx=self.model.bounds[1],
                    miny=self.model.bounds[2],
                    maxy=self.model.bounds[3],
                    search_ids=global_indices_not_adapted,
                    search_target_ids=global_indices_adapted,
                )

                # Calculate investment decisions for non-adapted farmers
                invest_decision = self.invest_numba(
                    neighbors_with_adaptation,
                    global_indices_not_adapted,
                    EUT_do_nothing,
                    SEUT_adapt,
                    self.yearly_yield_ratio.data,
                    self.yearly_SPEI_probability.data,
                    adapted,
                    self.n,
                    profits_no_event,
                    expenditure_cap,
                    total_annual_costs.data,
                    extra_constraint.data,
                )

                invest_in_adaptation[invest_decision] = True

        return invest_in_adaptation

    @staticmethod
    @njit(cache=True)
    def invest_numba(
        neighbors_with_adaptation: np.ndarray,
        farmers_without_adaptation: np.ndarray,
        EUT_do_nothing: np.ndarray,
        SEUT_adapt: np.ndarray,
        yield_ratio: np.ndarray,
        SPEI_prob: np.ndarray,
        adapted: np.ndarray,
        n: int,
        profits_no_event: np.ndarray,
        expenditure_cap: float,
        total_annual_costs: np.ndarray,
        extra_constraint: np.ndarray,
    ):
        """
        Calculate the investment decisions based on neighbors' adaptation and expected utilities.

        Parameters:
        - neighbors_with_adaptation: Array of neighbors with adaptation for each farmer.
        - farmers_without_adaptation: Array of indices of farmers without adaptation.
        - SEUT_do_nothing: Expected utility of not adapting.
        - SEUT_adapt: Expected utility of adapting.
        - adapted: Boolean array indicating if a farmer has adapted.
        - n: Total number of farmers.
        - profits_no_event: Profits without an event for each farmer.
        - expenditure_cap: Expenditure capability.
        - total_annual_costs: Total annual costs for adaptation.
        - extra_constraint: Additional constraints for adaptation.

        Returns:
        - numpy.ndarray: Boolean mask of farmers who decided to invest/adapt.
        """

        # Initialize investment decisions as a zero-filled boolean array
        invest_in_adaptation = np.zeros(n, dtype=np.bool_)

        # Placeholder value indicating no neighbor for a farmer
        neighbor_nan_value = np.iinfo(neighbors_with_adaptation.dtype).max

        for i, farmer_idx in enumerate(farmers_without_adaptation):
            # Confirm if the farmer fulfills the adaptation criteria
            can_adapt = (
                profits_no_event[farmer_idx] * expenditure_cap
                > total_annual_costs[farmer_idx]
                and adapted[farmer_idx] == 0
                and extra_constraint[farmer_idx]
            )
            if can_adapt:
                SEUT_farmer = SEUT_adapt[farmer_idx]
                assert SEUT_farmer != -np.inf, "Farmer is not able to adapt!"

                # Filter on only neighbors who have adapted
                adapted_neighbors = neighbors_with_adaptation[i]
                adapted_neighbors = adapted_neighbors[
                    adapted_neighbors != neighbor_nan_value
                ]

                # assert adapted_neighbors[adapted[adapted_neighbors] == 0], 'neighbor has not adapted'
                adapted_neighbors = adapted_neighbors[adapted[adapted_neighbors] == 1]

                if adapted_neighbors.size > 0:
                    mean_EUT_neighbors = np.mean(EUT_do_nothing[adapted_neighbors])
                    if mean_EUT_neighbors > SEUT_farmer:
                        # agent will adapt
                        invest_in_adaptation[farmer_idx] = True

                        # set SPEI and yr of adapting agent to that of agent copied
                        difference_to_mean = np.abs(
                            EUT_do_nothing[adapted_neighbors] - mean_EUT_neighbors
                        )
                        closest_neighbor_index = adapted_neighbors[
                            np.argmin(difference_to_mean)
                        ]

                        # Set spei and yield ratio to similar values
                        yield_ratio[farmer_idx] = yield_ratio[closest_neighbor_index, :]
                        SPEI_prob[farmer_idx] = yield_ratio[closest_neighbor_index, :]

        return invest_in_adaptation

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

    @staticmethod
    @njit(cache=True)
    def switch_crops_numba(
        ids, crops, neighbours, SEUT, EUT, yield_ratio, SPEI_prob
    ) -> None:
        """Switches crops for each farmer."""

        # Placeholder value indicating no neighbor for a farmer
        nodata_value_neighbors = np.iinfo(neighbours.dtype).max

        for i, farmer_idx in enumerate(ids):
            SEUT_self = SEUT[farmer_idx]
            neighbor_farmers = neighbours[i]
            neighbor_farmers = neighbor_farmers[
                neighbor_farmers != nodata_value_neighbors
            ]  # delete farmers without neighbors
            if neighbor_farmers.size == 0:  # no neighbors
                continue

            EUT_neighbor = EUT[neighbor_farmers]
            neighbor_with_max_EUT = np.argmax(EUT_neighbor)
            if EUT_neighbor[neighbor_with_max_EUT] > SEUT_self:
                # Let the agent copy the crop rotation of the neighbor
                crops[farmer_idx] = crops[neighbor_farmers[neighbor_with_max_EUT]]
                # Let the agent inherit the SPEI-yield relation of the neighbor
                yield_ratio[farmer_idx] = yield_ratio[
                    neighbor_farmers[neighbor_with_max_EUT], :
                ]
                SPEI_prob[farmer_idx] = SPEI_prob[
                    neighbor_farmers[neighbor_with_max_EUT], :
                ]

    def switch_crops(self):
        """Switches crops for each farmer."""
        for farmer_class in np.unique(self.farmer_class):
            ids = np.where(self.farmer_class == farmer_class)[0]
            neighbors = find_neighbors(
                self.locations.data,
                radius=1_000,
                n_neighbor=3,
                bits=19,
                minx=self.model.bounds[0],
                maxx=self.model.bounds[1],
                miny=self.model.bounds[2],
                maxy=self.model.bounds[3],
                search_ids=ids,
                search_target_ids=ids,
            )
            self.switch_crops_numba(
                ids,
                self.crop_calendar[:, :, 0].data,
                neighbors,
                self.SEUT_no_adapt,
                self.EUT_no_adapt,
                self.yearly_yield_ratio.data,
                self.yearly_SPEI_probability.data,
            )

    def step(self) -> None:
        """
        This function is called at the beginning of each timestep.

        Then, farmers harvest and plant crops.
        """
        # import random

        # for i in range(self.n):
        #     self.remove_agent(random.randint(0, self.n), land_use_type=1)

        self.harvest()
        self.plant()
        self.water_abstraction_sum()

        # if self.model.current_timestep > 1:

        #     print(
        #         "irr",
        #         self.remaining_irrigation_limit_m3[
        #             self.has_access_to_irrigation_water == 1
        #         ].mean(),
        #     )
        #     print(
        #         "command",
        #         self.remaining_irrigation_limit_m3[self.is_in_command_area].mean(),
        #     )
        #     print(
        #         "command&irr",
        #         self.remaining_irrigation_limit_m3[
        #             (
        #                 self.is_in_command_area
        #                 & (self.has_access_to_irrigation_water == 1)
        #             )
        #         ].mean(),
        #     )

        # monthly actions
        if (
            self.model.current_time.day == 1
            and self.model.config["general"]["simulate_hydrology"]
        ):
            self.cumulative_SPEI()

        ## yearly actions
        if self.model.current_time.month == 1 and self.model.current_time.day == 1:
            # Shift the potential and yearly profits forward
            shift_and_reset_matrix(self.yearly_profits)
            shift_and_reset_matrix(self.yearly_potential_profits)

            # reset the irrigation limit, but only if a full year has passed already. Otherwise
            # the cumulative water deficit is not year completed.
            if self.model.current_time.year - 1 > self.model.spinup_start.year:
                self.remaining_irrigation_limit_m3[:] = self.irrigation_limit_m3[:]

            advance_crop_rotation_year(
                current_crop_calendar_rotation_year_index=self.current_crop_calendar_rotation_year_index,
                crop_calendar_rotation_years=self.crop_calendar_rotation_years,
            )

            # for now class is only dependent on being in a command area or not
            self.farmer_class = self.is_in_command_area.copy().astype(np.int32)

            # Set to 0 if channel abstraction is bigger than reservoir and groundwater, 1 for reservoir, 2 for groundwater
            self.farmer_class[
                (
                    self.yearly_abstraction_m3_by_farmer[:, 0]
                    > self.yearly_abstraction_m3_by_farmer[:, 1]
                )
                & (
                    self.yearly_abstraction_m3_by_farmer[:, 0]
                    > self.yearly_abstraction_m3_by_farmer[:, 2]
                )
            ] = 0
            self.farmer_class[
                (
                    self.yearly_abstraction_m3_by_farmer[:, 1]
                    > self.yearly_abstraction_m3_by_farmer[:, 0]
                )
                & (
                    self.yearly_abstraction_m3_by_farmer[:, 1]
                    > self.yearly_abstraction_m3_by_farmer[:, 2]
                )
            ] = 1
            self.farmer_class[
                (
                    self.yearly_abstraction_m3_by_farmer[:, 2]
                    > self.yearly_abstraction_m3_by_farmer[:, 0]
                )
                & (
                    self.yearly_abstraction_m3_by_farmer[:, 2]
                    > self.yearly_abstraction_m3_by_farmer[:, 1]
                )
            ] = 2

            # Set to 3 for precipitation if there is no abstraction
            self.farmer_class[self.yearly_abstraction_m3_by_farmer[:, 3] == 0] = 3

            # Reset the yearly abstraction
            self.yearly_abstraction_m3_by_farmer[:, :] = 0

            if not self.model.spinup and not (
                "ruleset" in self.config and self.config["ruleset"] == "no-adaptation"
            ):
                # raise NotImplementedError(
                #     """Dynamic adaptation for farmers is not currently activated in the main branch, due
                #     to ongoing generalization efforts. Please use ruleset 'no-adaptation' for now. It is
                #     one of the main priorities to re-enable this functionality in the near future."""
                # )
                # Determine the relation between drought probability and yield
                self.calculate_yield_spei_relation()

                # Calculate the current SEUT and EUT of all agents. Used as base for all other adaptation calculations
                total_profits, profits_no_event = self.profits_SEUT(0)

                total_profits_adjusted = total_profits - (
                    np.sum(self.all_loans_annual_cost[:, :, 0], axis=1)
                    / self.field_size_per_farmer
                )
                profits_no_event_adjusted = profits_no_event - (
                    np.sum(self.all_loans_annual_cost[:, :, 0], axis=1)
                    / self.field_size_per_farmer
                )

                decision_params_SEUT = {
                    "n_agents": self.n,
                    "T": self.decision_horizon,
                    "discount_rate": self.discount_rate,
                    "sigma": self.risk_aversion,
                    "risk_perception": self.risk_perception,
                    "p_droughts": 1 / self.p_droughts[:-1],
                    "total_profits": total_profits_adjusted,
                    "profits_no_event": profits_no_event_adjusted,
                }

                decision_params_EUT = {
                    "n_agents": self.n,
                    "T": self.decision_horizon,
                    "discount_rate": np.full(
                        self.n, np.mean(self.discount_rate), dtype=np.int32
                    ),
                    "sigma": np.full(
                        self.n, np.mean(self.risk_aversion), dtype=np.int32
                    ),
                    "risk_perception": self.risk_perception,
                    "p_droughts": 1 / self.p_droughts[:-1],
                    "total_profits": total_profits_adjusted,
                    "profits_no_event": profits_no_event_adjusted,
                }

                self.SEUT_no_adapt = self.decision_module.calcEU_do_nothing(
                    **decision_params_SEUT
                )
                self.EUT_no_adapt = self.decision_module.calcEU_do_nothing(
                    **decision_params_EUT, subjective=False
                )

                # These adaptations can only be done if there is a yield-probability relation
                if not np.all(self.farmer_yield_probability_relation == 0):
                    self.adapt_irrigation_well()
                    # self.adapt_drip_irrigation()
                else:
                    raise AssertionError(
                        "Cannot adapt without yield - probability relation"
                    )

                self.switch_crops()

            # Update management yield ratio score
            self.update_yield_ratio_management()

            # Update loans
            self.update_loans()

            # Reset total crop age
            shift_and_update(self.total_crop_age, self.total_crop_age[:, 0])

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
