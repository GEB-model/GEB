from typing import Union

import numpy as np
from numba import njit


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


@njit(cache=True, inline="always")
def get_deficit_between_dates(
    cumulative_water_deficit_m3, farmer, start_index, end_index
):
    """
    Get the water deficit between two dates for a farmer.

    Parameters
    ----------
    cumulative_water_deficit_m3 : np.ndarray
        Cumulative water deficit in m3 for each day of the year for each farmer.
    farmer : int
        Farmer index.
    start_index : int
        Start day of cumulative water deficit calculation (index-based; Jan 1 == 0).
    end_index : int
        End day of cumulative water deficit calculation (index-based; Jan 1 == 0).

    Returns
    -------
    float
        Water deficit in m3 between the two dates.
    """
    if end_index == start_index:
        deficit = 0
    elif end_index > start_index:
        deficit = (
            cumulative_water_deficit_m3[farmer, end_index]
            - cumulative_water_deficit_m3[
                farmer, start_index
            ]  # current day of year is effectively starting "tommorrow" due to Python's 0-indexing
        )
    else:  # end < start
        deficit = cumulative_water_deficit_m3[farmer, -1] - (
            cumulative_water_deficit_m3[farmer, start_index]
            - cumulative_water_deficit_m3[farmer, end_index]
        )

    if deficit < 0:
        raise ValueError("Deficit must be positive or zero")
    return deficit


@njit(cache=True)
def get_future_deficit(
    farmer: int,
    day_index: int,
    cumulative_water_deficit_m3: np.ndarray,
    crop_calendar: np.ndarray,
    crop_rotation_year_index: np.ndarray,
    potential_irrigation_consumption_farmer_m3: float,
    reset_day_index=0,
):
    """
    Get the future water deficit for a farmer.

    Parameters
    ----------
    farmer : int
        Farmer index.
    day_index : int
        Current day index (0-indexed).
    cumulative_water_deficit_m3 : np.ndarray
        Cumulative water deficit in m3 for each day of the year for each farmer.
    crop_calendar : np.ndarray
        Crop calendar for each farmer. Each row is a farmer, and each column is a crop.
        Each crop is a list of [crop_type, planting_day, growing_days, crop_year_index].
        Planting day is 0-indexed (Jan 1 == 0).
        Growing days is the number of days the crop grows.
        Crop year index is the index of the year in the crop rotation.
    crop_rotation_year_index : np.ndarray
        Crop rotation year index for each farmer.
    potential_irrigation_consumption_farmer_m3 : float
        Potential irrigation consumption in m3 for each farmer on the current day.
    reset_day_index : int, optional
        Day index to reset the water year (0-indexed; Jan 1 == 0). Default is 0. Deficit
        is calculated up to this day. For example, when the reset day index is 364, the
        deficit is calculated up to Dec 31. When the reset day index is 0, the deficit is
        calculated up to Jan 1. Default is 0.

    Returns
    -------
    float
        Future water deficit in m3 for the farmer in the growing season.
    """
    if reset_day_index >= 365 or reset_day_index < 0:
        raise ValueError("Reset day index must be lower than 365 and greater than -1")
    future_water_deficit = potential_irrigation_consumption_farmer_m3
    for crop in crop_calendar[farmer]:
        crop_type = crop[0]
        crop_year_index = crop[3]
        if crop_type != -1 and crop_year_index == crop_rotation_year_index[farmer]:
            start_day_index = crop[1]
            if start_day_index < 0 or start_day_index >= 365:
                raise ValueError("Start day must be between 0 and 364")
            growth_length = crop[2]

            relative_start_day_index = (start_day_index - reset_day_index) % 365
            relative_end_day_index = relative_start_day_index + growth_length
            relative_day_index = (day_index - reset_day_index) % 365

            if relative_end_day_index > 365:
                relative_end_day_index = 365

            if relative_start_day_index < relative_day_index:
                relative_start_day_index = relative_day_index

            if relative_start_day_index == relative_day_index:
                relative_start_day_index = (relative_start_day_index + 1) % 365

            if relative_day_index >= relative_end_day_index:
                continue
            else:
                future_water_deficit += get_deficit_between_dates(
                    cumulative_water_deficit_m3,
                    farmer,
                    (relative_start_day_index + reset_day_index) % 365,
                    (relative_end_day_index + reset_day_index) % 365,
                )

    if future_water_deficit < 0:
        raise ValueError("Future water deficit must be positive or zero")
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
    potential_irrigation_consumption_m_farmer,
    potential_irrigation_consumption_farmer_m3,
):
    if remaining_irrigation_limit_m3[farmer] < np.float32(0):
        potential_irrigation_consumption_m_farmer[:] = np.float32(0)
        return potential_irrigation_consumption_m_farmer
    # calculate future water deficit, but also include today's irrigation consumption
    # Check whether a full year has passed
    if np.all(~np.isnan(cumulative_water_deficit_m3[farmer])):
        future_water_deficit = get_future_deficit(
            farmer=farmer,
            day_index=day_index,
            cumulative_water_deficit_m3=cumulative_water_deficit_m3,
            crop_calendar=crop_calendar,
            crop_rotation_year_index=crop_rotation_year_index,
            potential_irrigation_consumption_farmer_m3=potential_irrigation_consumption_farmer_m3,
        )
        limit_to_deficit_ratio = (
            remaining_irrigation_limit_m3[farmer] / future_water_deficit
        )
    else:
        limit_to_deficit_ratio = np.float32(1)

    assert future_water_deficit > np.float32(0)

    # first find the total irrigation demand for the farmer in m3
    irrigation_water_withdrawal_farmer_m3 = (
        potential_irrigation_consumption_farmer_m3 * limit_to_deficit_ratio
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

    potential_irrigation_consumption_m_farmer = (
        potential_irrigation_consumption_m_farmer * np.float32(reduction_factor)
    )
    return potential_irrigation_consumption_m_farmer


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
    irrigation_water_demand_field = max(irrigation_water_demand_field, 0)

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
    irrigation_water_demand_field = max(irrigation_water_demand_field, 0)
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
        irrigation_water_demand_field = max(irrigation_water_demand_field, 0)
    return irrigation_water_demand_field


@njit(cache=True, inline="always")
def get_potential_irrigation_consumption_m(
    fields,
    is_paddy_irrigated,
    paddy_level,
    readily_available_water,
    critical_water_level,
    max_water_content,
    potential_infiltration_capacity,
    fraction_irrigated_field,
    max_paddy_water_level,
) -> np.ndarray:
    """TODO: make this work for farmers with both paddy and non-paddy fields"""
    # Calculate the potential irrigation consumption for the farmer
    if is_paddy_irrigated[fields][0]:
        # make sure all fields are paddy irrigateds
        # always irrigate to 0.05 m for paddy fields
        potential_irrigation_consumption_m = max_paddy_water_level - paddy_level[fields]
        # make sure the potential irrigation consumption is not negative
        potential_irrigation_consumption_m[potential_irrigation_consumption_m < 0] = (
            np.float32(0)
        )
    else:
        # assert none of the fiels are paddy irrigated
        # if soil moisture content falls below critical level, irrigate to field capacity
        potential_irrigation_consumption_m = np.where(
            readily_available_water[fields]
            < critical_water_level[fields],  # if there is not enough water
            max_water_content[fields]
            - readily_available_water[
                fields
            ],  # irrigate to field capacity (if possible)
            np.float32(0),
        )
        # limit the irrigation consumption to the potential infiltration capacity
        potential_irrigation_consumption_m = (
            np.minimum(
                potential_irrigation_consumption_m,
                potential_infiltration_capacity[fields],
            )
            * fraction_irrigated_field
        )
    return potential_irrigation_consumption_m


# @njit(cache=True)
def get_gross_irrigation_demand_m3(
    day_index: int,
    n: int,
    activation_order: np.ndarray,
    field_indices_by_farmer: np.ndarray,
    field_indices: np.ndarray,
    irrigation_efficiency: np.ndarray,
    fraction_irrigated_field: np.ndarray,
    cell_area: np.ndarray,
    crop_map: np.ndarray,
    field_is_paddy_irrigated: np.ndarray,
    paddy_level: np.ndarray,
    readily_available_water: np.ndarray,
    critical_water_level: np.ndarray,
    max_water_content: np.ndarray,
    potential_infiltration_capacity: np.ndarray,
    remaining_irrigation_limit_m3: np.ndarray,
    cumulative_water_deficit_m3: np.ndarray,
    crop_calendar: np.ndarray,
    current_crop_calendar_rotation_year_index: np.ndarray,
    max_paddy_water_level: np.ndarray,
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
    gross_potential_irrigation_m3 = np.zeros(
        n_hydrological_response_units, dtype=np.float32
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

        potential_irrigation_consumption_m_farmer = (
            get_potential_irrigation_consumption_m(
                fields=farmer_fields,
                is_paddy_irrigated=field_is_paddy_irrigated,
                paddy_level=paddy_level,
                readily_available_water=readily_available_water,
                critical_water_level=critical_water_level,
                max_water_content=max_water_content,
                potential_infiltration_capacity=potential_infiltration_capacity,
                fraction_irrigated_field=fraction_irrigated_field[farmer],
                max_paddy_water_level=max_paddy_water_level[farmer],
            )
        )

        # If the potential irrigation consumption is larger than 0, the farmer needs to abstract water
        # if there is no irrigation limit, no need to adjust the irrigation
        if potential_irrigation_consumption_m_farmer.sum() > 0.0 and not np.isnan(
            remaining_irrigation_limit_m3[farmer]
        ):
            irrigation_efficiency_farmer = irrigation_efficiency[farmer]
            potential_irrigation_consumption_m_farmer = adjust_irrigation_to_limit(
                farmer=farmer,
                day_index=day_index,
                remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
                cumulative_water_deficit_m3=cumulative_water_deficit_m3,
                crop_calendar=crop_calendar,
                crop_rotation_year_index=current_crop_calendar_rotation_year_index,
                irrigation_efficiency_farmer=np.float32(irrigation_efficiency_farmer),
                potential_irrigation_consumption_m_farmer=potential_irrigation_consumption_m_farmer,
                potential_irrigation_consumption_farmer_m3=(
                    potential_irrigation_consumption_m_farmer * cell_area[farmer_fields]
                ).sum(),
            )
            gross_potential_irrigation_m3[farmer_fields] = (
                potential_irrigation_consumption_m_farmer * cell_area[farmer_fields]
            ).sum() / irrigation_efficiency_farmer

    return gross_potential_irrigation_m3


@njit(cache=True)
def abstract_water(
    activation_order: np.ndarray,
    field_indices_by_farmer: np.ndarray,
    field_indices: np.ndarray,
    irrigation_efficiency: np.ndarray,
    surface_irrigated: np.ndarray,
    well_irrigated: np.ndarray,
    cell_area: np.ndarray,
    HRU_to_grid: np.ndarray,
    nearest_river_grid_cell: np.ndarray,
    crop_map: np.ndarray,
    available_channel_storage_m3: np.ndarray,
    available_groundwater_m3: np.ndarray,
    groundwater_depth: np.ndarray,
    available_reservoir_storage_m3: np.ndarray,
    farmer_command_area: np.ndarray,
    return_fraction: float,
    well_depth: float,
    remaining_irrigation_limit_m3: np.ndarray,
    gross_potential_irrigation_m3: np.ndarray,
):
    for activated_farmer_index in range(activation_order.size):
        farmer = activation_order[activated_farmer_index]
        farmer_fields = get_farmer_HRUs(field_indices, field_indices_by_farmer, farmer)
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
        potential_irrigation_consumption_m3_farmer = gross_potential_irrigation_m3[
            farmer_fields
        ]
        if potential_irrigation_consumption_m3_farmer.sum() <= 0:
            continue

        # loop through all farmers fields and apply irrigation
        for field_idx, field in enumerate(farmer_fields):
            grid_cell = HRU_to_grid[field]
            grid_cell_nearest = nearest_river_grid_cell[field]
            if crop_map[field] != -1:
                irrigation_water_demand_field = gross_potential_irrigation_m3[field_idx]
                assert irrigation_water_demand_field >= 0

                if surface_irrigated[farmer]:
                    irrigation_water_demand_field = withdraw_channel(
                        available_channel_storage_m3=available_channel_storage_m3,
                        grid_cell=grid_cell_nearest,
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
                    assert irrigation_water_demand_field >= 0

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
                        assert irrigation_water_demand_field >= 0

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
                    assert irrigation_water_demand_field >= 0
                    assert water_withdrawal_m[field] >= 0

                assert (
                    irrigation_water_demand_field >= -1e15
                )  # Make sure irrigation water demand is zero, or positive. Allow very small error.

            assert water_consumption_m[field] >= 0
            assert water_withdrawal_m[field] >= 0
            assert 1 >= return_fraction >= 0

            water_consumption_m[field] = (
                water_withdrawal_m[field] * irrigation_efficiency[farmer]
            )
            irrigation_loss_m = water_withdrawal_m[field] - water_consumption_m[field]
            assert irrigation_loss_m >= 0
            returnFlowIrr_m[field] = irrigation_loss_m * return_fraction
            addtoevapotrans_m[field] = irrigation_loss_m - returnFlowIrr_m[field]

    return (
        channel_abstraction_m3_by_farmer,
        reservoir_abstraction_m3_by_farmer,
        groundwater_abstraction_m3_by_farmer,
        water_withdrawal_m,
        water_consumption_m,
        returnFlowIrr_m,
        addtoevapotrans_m,
    )


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
    assert farmers_going_out_of_business is False, (
        "Farmers going out of business not implemented."
    )

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


@njit(cache=True)
def arrays_equal(a, b):
    for i in range(a.size):
        if a.flat[i] != b.flat[i]:
            return True if a.flat[i] != b.flat[i] else False
    return True


@njit(cache=True)
def find_matching_rows(arr, target_row):
    n_rows = arr.shape[0]
    matches = np.empty(n_rows, dtype=np.bool_)
    for i in range(n_rows):
        matches[i] = True
        for j in range(arr.shape[1]):
            if arr[i, j] != target_row[j]:
                matches[i] = False
                break
    return matches


@njit(cache=True)
def find_most_similar_index(target_series, yield_ratios, groups):
    n = groups.size
    indices = []
    for i in range(n):
        if groups[i]:
            indices.append(i)
    n_indices = len(indices)
    distances = np.empty(n_indices, dtype=yield_ratios.dtype)
    for idx in range(n_indices):
        i = indices[idx]
        diff = yield_ratios[i] - target_series
        distances[idx] = np.linalg.norm(diff)
    min_idx = 0
    min_dist = distances[0]
    for idx in range(1, n_indices):
        if distances[idx] < min_dist:
            min_dist = distances[idx]
            min_idx = idx
    return indices[min_idx]


@njit(cache=True)
def crop_yield_ratio_difference_test_njit(
    yield_ratios,
    crop_elevation_group,
    unique_crop_groups,
    group_indices,
    crop_calendar,
    unique_crop_calendars,
    p_droughts,
):
    n_groups = len(unique_crop_groups)
    n_calendars = len(unique_crop_calendars)
    n_droughts = len(p_droughts)

    unique_yield_ratio_gain = np.full(
        (n_groups, n_calendars, n_droughts),
        0.0,
        dtype=np.float32,
    )

    id_to_switch_to = np.full(
        (n_groups, n_calendars),
        -1,
        dtype=np.int32,
    )

    crop_to_switch_to = np.full(
        (n_groups, n_calendars),
        -1,
        dtype=np.int32,
    )

    for group_id in range(n_groups):
        unique_group = unique_crop_groups[group_id]
        unique_farmer_groups = find_matching_rows(crop_elevation_group, unique_group)

        # Identify the adapted counterparts of the current group
        mask = np.empty(n_calendars, dtype=np.bool_)
        for i in range(n_calendars):
            match = True
            for j in range(unique_crop_calendars.shape[1]):
                if unique_crop_calendars[i, j] != unique_group[j]:
                    match = False
                    break
            mask[i] = not match

        # Collect candidate crop rotations
        candidate_crop_rotations = []
        for i in range(n_calendars):
            if mask[i]:
                candidate_crop_rotations.append(unique_crop_calendars[i])

        num_candidates = len(candidate_crop_rotations)

        # Loop over the counterparts
        for crop_id in range(num_candidates):
            unique_rotation = candidate_crop_rotations[crop_id]
            farmer_class = unique_group[-1]
            basin_location = unique_group[-2]
            unique_group_other_crop = np.empty(
                len(unique_rotation) + 2, dtype=unique_rotation.dtype
            )
            unique_group_other_crop[: len(unique_rotation)] = unique_rotation
            unique_group_other_crop[len(unique_rotation)] = basin_location
            unique_group_other_crop[len(unique_rotation) + 1] = farmer_class

            unique_farmer_groups_other_crop = find_matching_rows(
                crop_elevation_group, unique_group_other_crop
            )

            if np.any(unique_farmer_groups_other_crop):
                # Compute current yield ratio mean
                current_sum = np.zeros(n_droughts, dtype=yield_ratios.dtype)
                count_current = 0
                for i in range(unique_farmer_groups.size):
                    if unique_farmer_groups[i]:
                        current_sum += yield_ratios[i]
                        count_current += 1
                current_yield_ratio = (
                    current_sum / count_current if count_current > 0 else current_sum
                )

                # Compute candidate yield ratio mean
                candidate_sum = np.zeros(n_droughts, dtype=yield_ratios.dtype)
                count_candidate = 0
                for i in range(unique_farmer_groups_other_crop.size):
                    if unique_farmer_groups_other_crop[i]:
                        candidate_sum += yield_ratios[i]
                        count_candidate += 1
                candidate_yield_ratio = (
                    candidate_sum / count_candidate
                    if count_candidate > 0
                    else candidate_sum
                )

                yield_ratio_gain = candidate_yield_ratio - current_yield_ratio

                crop_to_switch_to[group_id, crop_id] = unique_rotation[0]

                if np.all(np.isnan(yield_ratio_gain)):
                    yield_ratio_gain = np.zeros_like(yield_ratio_gain)
                else:
                    id_to_switch_to[group_id, crop_id] = find_most_similar_index(
                        yield_ratio_gain,
                        yield_ratios,
                        unique_farmer_groups_other_crop,
                    )

                unique_yield_ratio_gain[group_id, crop_id, :] = yield_ratio_gain

    gains_adaptation = unique_yield_ratio_gain[group_indices, :, :]
    new_crop_nr = crop_to_switch_to[group_indices, :]
    new_farmer_id = id_to_switch_to[group_indices, :]

    return (
        gains_adaptation,
        new_crop_nr,
        new_farmer_id,
    )


@njit(cache=True)
def crop_profit_difference_njit(
    yield_ratios,
    crop_elevation_group,
    unique_crop_groups,
    group_indices,
    crop_calendar,
    unique_crop_calendars,
    p_droughts,
):
    n_groups = len(unique_crop_groups)
    n_calendars = len(unique_crop_calendars)
    n_droughts = len(p_droughts)

    unique_yield_ratio_gain = np.full(
        (n_groups, n_calendars, n_droughts),
        0.0,
        dtype=np.float32,
    )

    id_to_switch_to = np.full(
        (n_groups, n_calendars),
        -1,
        dtype=np.int32,
    )

    crop_to_switch_to = np.full(
        (n_groups, n_calendars),
        -1,
        dtype=np.int32,
    )

    for group_id in range(n_groups):
        unique_group = unique_crop_groups[group_id]
        unique_farmer_groups = find_matching_rows(crop_elevation_group, unique_group)

        # Identify the adapted counterparts of the current group
        mask = np.empty(n_calendars, dtype=np.bool_)
        for i in range(n_calendars):
            match = True
            for j in range(unique_crop_calendars.shape[1]):
                if unique_crop_calendars[i, j] != unique_group[j]:
                    match = False
                    break
            mask[i] = not match

        # Collect candidate crop rotations
        candidate_crop_rotations = []
        for i in range(n_calendars):
            if mask[i]:
                candidate_crop_rotations.append(unique_crop_calendars[i])

        num_candidates = len(candidate_crop_rotations)

        # Loop over the counterparts
        for crop_id in range(num_candidates):
            unique_rotation = candidate_crop_rotations[crop_id]
            farmer_class = unique_group[-1]
            basin_location = unique_group[-2]
            unique_group_other_crop = np.empty(
                len(unique_rotation) + 2, dtype=unique_rotation.dtype
            )
            unique_group_other_crop[: len(unique_rotation)] = unique_rotation
            unique_group_other_crop[len(unique_rotation)] = basin_location
            unique_group_other_crop[len(unique_rotation) + 1] = farmer_class

            unique_farmer_groups_other_crop = find_matching_rows(
                crop_elevation_group, unique_group_other_crop
            )

            if np.any(unique_farmer_groups_other_crop):
                # Compute current yield ratio mean
                current_sum = np.zeros(n_droughts, dtype=yield_ratios.dtype)
                count_current = 0
                for i in range(unique_farmer_groups.size):
                    if unique_farmer_groups[i]:
                        current_sum += yield_ratios[i]
                        count_current += 1
                current_yield_ratio = (
                    current_sum / count_current if count_current > 0 else current_sum
                )

                # Compute candidate yield ratio mean
                candidate_sum = np.zeros(n_droughts, dtype=yield_ratios.dtype)
                count_candidate = 0
                for i in range(unique_farmer_groups_other_crop.size):
                    if unique_farmer_groups_other_crop[i]:
                        candidate_sum += yield_ratios[i]
                        count_candidate += 1
                candidate_yield_ratio = (
                    candidate_sum / count_candidate
                    if count_candidate > 0
                    else candidate_sum
                )

                yield_ratio_gain = candidate_yield_ratio - current_yield_ratio

                crop_to_switch_to[group_id, crop_id] = unique_rotation[0]

                if np.all(np.isnan(yield_ratio_gain)):
                    yield_ratio_gain = np.zeros_like(yield_ratio_gain)
                else:
                    id_to_switch_to[group_id, crop_id] = find_most_similar_index(
                        yield_ratio_gain,
                        yield_ratios,
                        unique_farmer_groups_other_crop,
                    )

                unique_yield_ratio_gain[group_id, crop_id, :] = yield_ratio_gain

    gains_adaptation = unique_yield_ratio_gain[group_indices, :, :]
    new_crop_nr = crop_to_switch_to[group_indices, :]
    new_farmer_id = id_to_switch_to[group_indices, :]

    return (
        gains_adaptation,
        new_crop_nr,
        new_farmer_id,
    )
