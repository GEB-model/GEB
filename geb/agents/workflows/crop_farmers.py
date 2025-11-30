"""Contains several functions for crop farmer module."""

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from geb.hydrology.evapotranspiration import get_root_ratios
from geb.hydrology.soil import (
    get_infiltration_capacity,
)


@njit(cache=True, inline="always")
def get_farmer_HRUs(
    field_indices: npt.NDArray[np.int32],
    field_indices_by_farmer: npt.NDArray[np.int32],
    farmer_index: int,
) -> npt.NDArray[np.int32]:
    """Get HRU indices for a given farmer.

    Args:
        field_indices: Flat array of HRU indices for all fields (ordered by farmer).
        field_indices_by_farmer: Start/stop indices into ``field_indices`` per farmer.
        farmer_index: Farmer index to select.

    Returns:
        HRU indices that belong to ``farmer_index``.
    """
    return field_indices[
        field_indices_by_farmer[farmer_index, 0] : field_indices_by_farmer[
            farmer_index, 1
        ]
    ]


@njit(cache=True)
def farmer_command_area(
    n: int,
    field_indices: npt.NDArray[np.int32],
    field_indices_by_farmer: npt.NDArray[np.int32],
    reservoir_command_areas: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Map each farmer to the first available reservoir command area.

    Args:
        n: Number of farmers.
        field_indices: Flat array of HRU indices for all fields (ordered by farmer).
        field_indices_by_farmer: Start/stop indices into ``field_indices`` per farmer.
        reservoir_command_areas: Command-area id per HRU (``-1`` if none).

    Returns:
        Array of length ``n`` with command-area id per farmer (``-1`` if none).
    """
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
    n: int,
    groundwater_depth: npt.NDArray[np.float32],
    HRU_to_grid: npt.NDArray[np.int32],
    field_indices: npt.NDArray[np.int32],
    field_indices_by_farmer: npt.NDArray[np.int32],
    cell_area: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute area-weighted mean groundwater depth per farmer.

    Args:
        n: Number of farmers.
        groundwater_depth: Groundwater depth per grid cell.
        HRU_to_grid: Map from HRU index to grid-cell index.
        field_indices: Flat array of HRU indices for all fields (ordered by farmer).
        field_indices_by_farmer: Start/stop indices into ``field_indices`` per farmer.
        cell_area: HRU area.

    Returns:
        Area-weighted groundwater depth per farmer.
    """
    groundwater_depth_by_farmer = np.full(n, np.nan, dtype=np.float32)
    for farmer_i in range(n):
        farmer_fields = get_farmer_HRUs(
            field_indices=field_indices,
            field_indices_by_farmer=field_indices_by_farmer,
            farmer_index=farmer_i,
        )
        total_cell_area = 0.0
        total_groundwater_depth_times_area = 0.0
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
    cumulative_water_deficit_m3: np.ndarray,
    farmer: int,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    """Get the water deficit between two dates for a farmer.

    Args:
        cumulative_water_deficit_m3: Cumulative water deficit in m3 for each day of the year for each farmer.
        farmer: The index of the farmer in the cumulative water deficit.
        start_index: Start day of cumulative water deficit calculation (index-based; Jan 1 == 0).
        end_index: End day of cumulative water deficit calculation (index-based; Jan 1 == 0).

    Returns:
        Water deficit in m3 between the two dates.

    Raises:
        ValueError: If deficit is negative.
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
    cumulative_water_deficit_m3: npt.NDArray[np.float32],
    crop_calendar: npt.NDArray[np.int32],
    crop_rotation_year_index: npt.NDArray[np.int32],
    potential_irrigation_consumption_farmer_m3: float,
    reset_day_index: int,
) -> float:
    """Get the future water deficit for a farmer.

    Args:
        farmer: Farmer index.
        day_index: Current day index (0-indexed).
        cumulative_water_deficit_m3: Cumulative water deficit in m3 for each day of the year for each farmer.
        crop_calendar: Crop calendar for each farmer. Each row is a farmer, and each column is a crop.
            Each crop is a list of [crop_type, planting_day, growing_days, crop_year_index].
            Planting day is 0-indexed (Jan 1 == 0).
            Growing days is the number of days the crop grows.
            Crop year index is the index of the year in the crop rotation.
        crop_rotation_year_index: Crop rotation year index for each farmer.
        potential_irrigation_consumption_farmer_m3: Potential irrigation consumption in m3 for each farmer on the current day.
        reset_day_index: Day index to reset the water year (0-indexed; Jan 1 == 0). Default is 0. Deficit
            is calculated up to this day. For example, when the reset day index is 364, the
            deficit is calculated up to Dec 31. When the reset day index is 0, the deficit is
            calculated up to Jan 1.

    Returns:
        Future water deficit in m3 for the farmer in the growing season.

    Raises:
        ValueError: If reset_day_index is not between 0 and 364 inclusive.
        ValueError: If start_day_index is not between 0 and 364 inclusive.
        ValueError: If future_water_deficit is negative.
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
                raise ValueError("Start day must be lower than 365 and greater than -1")
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
    farmer_gross_irrigation_demand_m3: float,
    irrigation_efficiency_farmer: float,
    reset_day_index: np.int32,
) -> float:
    """Adjusts the irrigation to the remaining irrigation limit for a farmer.

    Args:
        farmer: Index of the farmer.
        day_index: Current day index (0-indexed).
        remaining_irrigation_limit_m3: Remaining irrigation limit in m3 for each farmer.
        cumulative_water_deficit_m3: Cumulative water deficit in m3 for each day of the year for each farmer.
        crop_calendar: Crop calendar for each farmer. Each row is a farmer, and each column is a crop.
        crop_rotation_year_index: Crop rotation year index for each farmer. Used to select the correct crop calendar.
        farmer_gross_irrigation_demand_m3: Gross irrigation demand in m3 for the farmer on the current day.
        irrigation_efficiency_farmer: Irrigation efficiency for the farmer.
        reset_day_index: Day to reset the irrigation year, effectively the end of the decision period.

    Returns:
        irrigation_correction_factor: The ratio of the remaining irrigation limit to the future water deficit.
        If the remaining irrigation limit is less than 0, returns 0.
    """
    if remaining_irrigation_limit_m3[farmer] < np.float32(0):
        return np.float32(0)
    # calculate future water deficit, but also include today's irrigation consumption
    # Check whether a full year has passed
    if np.all(~np.isnan(cumulative_water_deficit_m3[farmer])):
        potential_irrigation_consumption_farmer_m3 = (
            farmer_gross_irrigation_demand_m3 * irrigation_efficiency_farmer
        )
        future_water_deficit = get_future_deficit(
            farmer=farmer,
            day_index=day_index,
            cumulative_water_deficit_m3=cumulative_water_deficit_m3,
            crop_calendar=crop_calendar,
            crop_rotation_year_index=crop_rotation_year_index,
            potential_irrigation_consumption_farmer_m3=potential_irrigation_consumption_farmer_m3,
            reset_day_index=reset_day_index,
        )
        effective_remaining_irrigation_limit_m3 = (
            remaining_irrigation_limit_m3[farmer] * irrigation_efficiency_farmer
        )

        irrigation_correction_factor = (
            effective_remaining_irrigation_limit_m3 / future_water_deficit
        )
        # limit the ratio to 1 if the deficit is smaller than the limit
        irrigation_correction_factor = min(irrigation_correction_factor, 1)
    else:
        irrigation_correction_factor = np.float32(1)

    assert future_water_deficit > np.float32(0)
    return irrigation_correction_factor


@njit(cache=True)
def withdraw_channel(
    available_channel_storage_m3: npt.NDArray[np.float32],
    grid_cell: int,
    cell_area: npt.NDArray[np.float32],
    field: int,
    farmer: int,
    irrigation_water_demand_field_m: float,
    water_withdrawal_m: npt.NDArray[np.float32],
    remaining_irrigation_limit_m3: npt.NDArray[np.float32],
    channel_abstraction_m3_by_farmer: npt.NDArray[np.float32],
    maximum_abstraction_channel_m3_field: np.float32,
    minimum_channel_storage_m3: float = 100.0,
) -> float:
    """Withdraw irrigation water from channel for a field.

    Args:
        available_channel_storage_m3: Available channel storage by grid cell (m³).
        grid_cell: Grid-cell index of the field.
        cell_area: HRU area (m²).
        field: HRU index.
        farmer: Farmer index.
        irrigation_water_demand_field_m: Remaining demand at field (m).
        water_withdrawal_m: Per-field withdrawal accumulator (m).
        remaining_irrigation_limit_m3: Remaining seasonal limit per farmer (m³),
            or ``NaN`` when unlimited.
        channel_abstraction_m3_by_farmer: Per-farmer channel abstraction accumulator (m³).
        maximum_abstraction_channel_m3_field: Max channel abstraction allowed for this
            field in current step (m³).
        minimum_channel_storage_m3: Minimum storage to keep in channel (buffer).

    Returns:
        Updated remaining field demand (m).
    """
    water_demand_cell_m3 = irrigation_water_demand_field_m * cell_area[field]
    assert water_demand_cell_m3 >= 0

    remaining_channel_m3 = max(
        available_channel_storage_m3[grid_cell] - minimum_channel_storage_m3, 0
    )

    # channel abstraction
    channel_abstraction_cell_m3 = min(
        remaining_channel_m3,
        water_demand_cell_m3,
        maximum_abstraction_channel_m3_field,
    )
    # ensure channel abstraction is non-negative
    assert channel_abstraction_cell_m3 >= 0

    available_channel_storage_m3[grid_cell] -= channel_abstraction_cell_m3
    channel_abstraction_cell_m = channel_abstraction_cell_m3 / cell_area[field]
    water_withdrawal_m[field] += channel_abstraction_cell_m

    if not np.isnan(remaining_irrigation_limit_m3[farmer]):
        remaining_irrigation_limit_m3[farmer] -= channel_abstraction_cell_m3

    channel_abstraction_m3_by_farmer[farmer] += channel_abstraction_cell_m3

    irrigation_water_demand_field_m -= channel_abstraction_cell_m
    irrigation_water_demand_field_m = max(irrigation_water_demand_field_m, 0)

    return irrigation_water_demand_field_m


@njit(cache=True)
def withdraw_reservoir(
    command_area: int,
    field: int,
    farmer: int,
    reservoir_abstraction_m3: npt.NDArray[np.float32],
    available_reservoir_storage_m3: npt.NDArray[np.float32],
    irrigation_water_demand_field_m: np.float32,
    water_withdrawal_m: npt.NDArray[np.float32],
    remaining_irrigation_limit_m3: npt.NDArray[np.float32],
    reservoir_abstraction_m3_by_farmer: npt.NDArray[np.float32],
    maximum_abstraction_reservoir_m3_field: np.float32,
    cell_area: npt.NDArray[np.float32],
) -> np.float32:
    """Withdraw irrigation water from a reservoir for a field.

    Args:
        command_area: Reservoir command-area id for this farmer, or ``-1``.
        field: HRU index.
        farmer: Farmer index.
        reservoir_abstraction_m3: Per-reservoir abstraction accumulator (m³).
        available_reservoir_storage_m3: Available storage per reservoir (m³).
        irrigation_water_demand_field_m: Remaining demand at field (m).
        water_withdrawal_m: Per-field withdrawal accumulator (m).
        remaining_irrigation_limit_m3: Remaining seasonal limit per farmer (m³),
            or ``NaN`` when unlimited.
        reservoir_abstraction_m3_by_farmer: Per-farmer reservoir abstraction accumulator (m³).
        maximum_abstraction_reservoir_m3_field: Max reservoir abstraction allowed for
            this field in current step (m³).
        cell_area: HRU area (m²).

    Returns:
        Updated remaining field demand (m).
    """
    water_demand_cell_m3 = irrigation_water_demand_field_m * cell_area[field]
    assert water_demand_cell_m3 >= 0

    remaining_reservoir_m3 = max(
        available_reservoir_storage_m3[command_area]
        - reservoir_abstraction_m3[command_area],
        0,
    )

    reservoir_abstraction_cell_m3 = min(
        remaining_reservoir_m3,
        water_demand_cell_m3,
        maximum_abstraction_reservoir_m3_field,
    )
    # ensure reservoir abstraction is non-negative
    assert reservoir_abstraction_cell_m3 >= 0

    reservoir_abstraction_m3[command_area] += reservoir_abstraction_cell_m3
    reservoir_abstraction_cell_m = reservoir_abstraction_cell_m3 / cell_area[field]
    water_withdrawal_m[field] += reservoir_abstraction_cell_m

    if not np.isnan(remaining_irrigation_limit_m3[farmer]):
        remaining_irrigation_limit_m3[farmer] -= reservoir_abstraction_cell_m3

    reservoir_abstraction_m3_by_farmer[farmer] += reservoir_abstraction_cell_m3

    irrigation_water_demand_field_m -= reservoir_abstraction_cell_m
    irrigation_water_demand_field_m = np.maximum(
        irrigation_water_demand_field_m, np.float32(0)
    )

    return irrigation_water_demand_field_m


@njit(cache=True)
def withdraw_groundwater(
    farmer: int,
    grid_cell: int,
    field: int,
    groundwater_abstraction_m3: npt.NDArray[np.float32],
    available_groundwater_m3: npt.NDArray[np.float32],
    cell_area: npt.NDArray[np.float32],
    groundwater_depth: npt.NDArray[np.float32],
    well_depth: npt.NDArray[np.float32],
    irrigation_water_demand_field_m: float,
    water_withdrawal_m: npt.NDArray[np.float32],
    remaining_irrigation_limit_m3: npt.NDArray[np.float32],
    groundwater_abstraction_m3_by_farmer: npt.NDArray[np.float32],
    maximum_abstraction_groundwater_m3_field: np.float32,
) -> float:
    """Withdraw irrigation water from groundwater for a field.

    Args:
        farmer: Farmer index.
        grid_cell: Grid-cell index corresponding to field.
        field: HRU index.
        groundwater_abstraction_m3: Per-grid-cell groundwater abstraction accumulator (m³).
        available_groundwater_m3: Available groundwater per grid cell (m³).
        cell_area: HRU area (m²).
        groundwater_depth: Groundwater depth per grid cell (m).
        well_depth: Per-farmer well depth (m).
        irrigation_water_demand_field_m: Remaining demand at field (m).
        water_withdrawal_m: Per-field withdrawal accumulator (m).
        remaining_irrigation_limit_m3: Remaining seasonal limit per farmer (m³),
            or ``NaN`` when unlimited.
        groundwater_abstraction_m3_by_farmer: Per-farmer groundwater abstraction accumulator (m³).
        maximum_abstraction_groundwater_m3_field: Max allowed GW abstraction (m³) for this field.

    Returns:
        Updated remaining field demand (m).
    """
    # groundwater irrigation
    if groundwater_depth[grid_cell] < well_depth[farmer]:
        water_demand_cell_m3 = irrigation_water_demand_field_m * cell_area[field]
        assert water_demand_cell_m3 >= 0

        remaining_groundwater_m3 = max(
            available_groundwater_m3[grid_cell] - groundwater_abstraction_m3[grid_cell],
            0,
        )

        groundwater_abstraction_cell_m3 = min(
            remaining_groundwater_m3,
            water_demand_cell_m3,
            maximum_abstraction_groundwater_m3_field,
        )
        # ensure groundwater abstraction is non-negative
        assert groundwater_abstraction_cell_m3 >= 0

        groundwater_abstraction_m3[grid_cell] += groundwater_abstraction_cell_m3
        groundwater_abstraction_cell_m = (
            groundwater_abstraction_cell_m3 / cell_area[field]
        )
        water_withdrawal_m[field] += groundwater_abstraction_cell_m

        if not np.isnan(remaining_irrigation_limit_m3[farmer]):
            remaining_irrigation_limit_m3[farmer] -= groundwater_abstraction_cell_m3

        groundwater_abstraction_m3_by_farmer[farmer] += groundwater_abstraction_cell_m3

        irrigation_water_demand_field_m -= groundwater_abstraction_cell_m
        irrigation_water_demand_field_m = max(irrigation_water_demand_field_m, 0)

    return irrigation_water_demand_field_m


@njit(cache=True, inline="always")
def get_potential_irrigation_consumption_m(
    topwater: npt.NDArray[np.float32],
    root_depth_m: npt.NDArray[np.float32],
    soil_layer_height: npt.NDArray[np.float32],
    field_capacity: npt.NDArray[np.float32],
    wilting_point: npt.NDArray[np.float32],
    w: npt.NDArray[np.float32],
    ws: npt.NDArray[np.float32],
    saturated_hydraulic_conductivity_m_per_day: npt.NDArray[np.float32],
    fraction_irrigated_field: npt.NDArray[np.float32],
    max_paddy_water_level_farmer: npt.NDArray[np.float32],
    crop_group: npt.NDArray[np.float32],  # kept for future dynamic-p logic
    is_paddy: np.bool_,
    minimum_effective_root_depth_m: np.float32,
    depletion_factor: np.float32 = np.float32(0.5),
) -> np.float32:
    """Compute potential irrigation (m) for a field on the current day.

    For paddy: irrigate up to a target ponding level above the current topwater.
    For non-paddy: apply when soil depletion exceeds a fraction of readily
    available water in the root zone, capped by infiltration capacity.

    Args:
        topwater: Current surface water level at the field (m).
        root_depth_m: Current crop root depth (m).
        soil_layer_height: Soil layer heights (m), shape (n_layers,).
        field_capacity: Volumetric water at field capacity per layer, (n_layers,).
        wilting_point: Volumetric water at wilting point per layer, (n_layers,).
        w: Current soil water per layer (same units as FC/WP), (n_layers,).
        ws: Saturation per layer (unused here but kept for signature stability).
        saturated_hydraulic_conductivity_m_per_day: Layer/soil infiltration control (m/day).
        fraction_irrigated_field: Fraction of HRU actually irrigated [0,1].
        max_paddy_water_level_farmer: Target ponding level for paddy (m).
        crop_group: Crop stress group (placeholder for future dynamic p).
        is_paddy: Whether this field is paddy.
        minimum_effective_root_depth_m: Lower bound for effective root depth (m).
        depletion_factor: Multiplier on RAW threshold (default 0.5).

    Returns:
        Potential irrigation depth (m) to apply today (already scaled by
        ``fraction_irrigated_field``).
    """
    assert np.float32(0) <= fraction_irrigated_field <= np.float32(1)

    # Calculate the potential irrigation consumption for the farmer
    if is_paddy:
        # make sure all fields are paddy irrigateds
        # always irrigate to 0.05 m for paddy fields
        potential_irrigation_consumption_m = max(
            max_paddy_water_level_farmer - topwater,
            np.float32(0),
        )
    else:
        # use a minimum root depth of 25 cm, following AQUACROP recommendation
        # see: Reference manual for AquaCrop v7.1 – Chapter 3
        effective_root_depth: np.float32 = np.maximum(
            minimum_effective_root_depth_m, root_depth_m
        )
        root_ratios: npt.NDArray[np.float32] = get_root_ratios(
            effective_root_depth,
            soil_layer_height,
        )

        field_capacity_root_zone = (field_capacity * root_ratios).sum(axis=0)
        wilting_point_root_zone = (wilting_point * root_ratios).sum(axis=0)
        available_water_root_zone = (w * root_ratios).sum(axis=0)

        # calculate the total available soil water for the root zone
        # often refered to as TAW (see AquaCrop v7.1, Chapter 3)
        maximum_available_water = field_capacity_root_zone - wilting_point_root_zone

        # # calculate the depletion coefficient (P_{sto}), which is the fraction of
        # # total available water (TAW) at which stomata start to close
        # p: np.float32 = get_fraction_easily_available_soil_water(
        #     crop_group, potential_evapotranspiration
        # )
        # TODO: Re-enable some dynamic p but not based on PET since this gives
        # issues when used in irrigation
        p = np.float32(0.5)

        readily_available_water_root_zone = maximum_available_water * p

        soil_depletion = field_capacity_root_zone - available_water_root_zone

        if soil_depletion > readily_available_water_root_zone * depletion_factor:
            potential_irrigation_consumption_m = soil_depletion
        else:
            potential_irrigation_consumption_m = np.float32(0)

        infiltration_capacity = get_infiltration_capacity(
            saturated_hydraulic_conductivity=saturated_hydraulic_conductivity_m_per_day,
        )
        potential_irrigation_consumption_m = np.minimum(
            potential_irrigation_consumption_m, infiltration_capacity
        )

    return potential_irrigation_consumption_m * fraction_irrigated_field


@njit(cache=True)
def get_gross_irrigation_demand_m3(
    day_index: int,
    n: int,
    currently_irrigated_fields: npt.NDArray[np.bool_],
    field_indices_by_farmer: npt.NDArray[np.int32],
    field_indices: npt.NDArray[np.int32],
    irrigation_efficiency: npt.NDArray[np.float32],
    fraction_irrigated_field: npt.NDArray[np.float32],
    cell_area: npt.NDArray[np.float32],
    crop_map: npt.NDArray[np.int32],
    topwater: npt.NDArray[np.float32],
    root_depth_m: npt.NDArray[np.float32],
    soil_layer_height: npt.NDArray[np.float32],
    field_capacity: npt.NDArray[np.float64],
    wilting_point: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
    ws: npt.NDArray[np.float64],
    saturated_hydraulic_conductivity_m_per_day: npt.NDArray[np.float32],
    remaining_irrigation_limit_m3_reservoir: npt.NDArray[np.float32],
    remaining_irrigation_limit_m3_channel: npt.NDArray[np.float32],
    remaining_irrigation_limit_m3_groundwater: npt.NDArray[np.float32],
    irrigation_limit_reset_day_index: npt.NDArray[np.int32],
    cumulative_water_deficit_m3: npt.NDArray[np.float32],
    crop_calendar: npt.NDArray[np.int32],
    crop_group_numbers: npt.NDArray[np.float32],
    paddy_irrigated_crops: npt.NDArray[np.bool_],
    current_crop_calendar_rotation_year_index: npt.NDArray[np.int32],
    max_paddy_water_level: npt.NDArray[np.float32],
    minimum_effective_root_depth_m: np.float32,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Compute gross irrigation demand per field (m³) and limit-adjusted variants.

    Iterates per farmer and field to compute gross potential irrigation demand
    (m³) from soil/crop state, then applies irrigation-limit corrections for
    each source (in the order reservoir, channel, groundwater) when a limit is present.
    Used to regulate the irrigation behavior of farmers. The farmers are "activated"
    by the given `activation_order`

    Returns:
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]: A 4-tuple of arrays (all shape = n_fields, dtype float32):
            - gross_potential_irrigation_m3: Unconstrained gross demand (m³) per field.
            - gross_potential_irrigation_m3_limit_adjusted_reservoir: Demand limited by the reservoir cap.
            - gross_potential_irrigation_m3_limit_adjusted_channel: Demand limited by the channel cap.
            - gross_potential_irrigation_m3_limit_adjusted_groundwater: Demand limited by the groundwater cap.
    """
    n_hydrological_response_units: int = cell_area.size
    gross_potential_irrigation_m3: npt.NDArray[np.float32] = np.zeros(
        n_hydrological_response_units, dtype=np.float32
    )
    gross_potential_irrigation_m3_limit_adjusted_reservoir: npt.NDArray[np.float32] = (
        np.zeros_like(gross_potential_irrigation_m3)
    )
    gross_potential_irrigation_m3_limit_adjusted_channel: npt.NDArray[np.float32] = (
        np.zeros_like(gross_potential_irrigation_m3)
    )
    gross_potential_irrigation_m3_limit_adjusted_groundwater: npt.NDArray[
        np.float32
    ] = np.zeros_like(gross_potential_irrigation_m3)
    # TODO: First part of this function can be parallelized
    for farmer in range(n):
        farmer_fields: npt.NDArray[np.int32] = get_farmer_HRUs(
            field_indices, field_indices_by_farmer, farmer
        )
        irrigation_efficiency_farmer: np.float32 = irrigation_efficiency[farmer]
        for field in farmer_fields:
            if not currently_irrigated_fields[field]:
                continue

            crop: np.int32 = crop_map[field]
            assert crop != -1

            consumption_m: np.float32 = get_potential_irrigation_consumption_m(
                topwater=topwater[field],
                root_depth_m=root_depth_m[field],
                soil_layer_height=soil_layer_height[:, field],
                field_capacity=field_capacity[:, field],
                wilting_point=wilting_point[:, field],
                w=w[:, field],
                ws=ws[:, field],
                saturated_hydraulic_conductivity_m_per_day=saturated_hydraulic_conductivity_m_per_day[
                    :, field
                ],
                fraction_irrigated_field=fraction_irrigated_field[farmer],
                max_paddy_water_level_farmer=max_paddy_water_level[farmer],
                crop_group=crop_group_numbers[crop],
                is_paddy=paddy_irrigated_crops[crop],
                minimum_effective_root_depth_m=minimum_effective_root_depth_m,
            )

            assert consumption_m < 1

            gross_potential_irrigation_m3[field] = (
                consumption_m * cell_area[field]
            ) / irrigation_efficiency_farmer

        farmer_gross_irrigation_demand_m3: np.float32 = gross_potential_irrigation_m3[
            farmer_fields
        ].sum()
        # If the potential irrigation consumption is larger than 0, the farmer needs to abstract water
        # if there is no irrigation limit, no need to adjust the irrigation
        if not np.isnan(remaining_irrigation_limit_m3_reservoir[farmer]):
            if farmer_gross_irrigation_demand_m3 > 0.0:
                irrigation_correction_factor: float = adjust_irrigation_to_limit(
                    farmer=farmer,
                    day_index=day_index,
                    remaining_irrigation_limit_m3=remaining_irrigation_limit_m3_reservoir,
                    cumulative_water_deficit_m3=cumulative_water_deficit_m3,
                    crop_calendar=crop_calendar,
                    crop_rotation_year_index=current_crop_calendar_rotation_year_index,
                    farmer_gross_irrigation_demand_m3=farmer_gross_irrigation_demand_m3,
                    irrigation_efficiency_farmer=irrigation_efficiency_farmer,
                    reset_day_index=irrigation_limit_reset_day_index[farmer],
                )
                assert 0 <= irrigation_correction_factor <= 1

                gross_potential_irrigation_m3_limit_adjusted_reservoir[
                    farmer_fields
                ] = (
                    gross_potential_irrigation_m3[farmer_fields]
                    * irrigation_correction_factor
                )
        else:
            gross_potential_irrigation_m3_limit_adjusted_reservoir[farmer_fields] = (
                gross_potential_irrigation_m3[farmer_fields]
            )

        if not np.isnan(remaining_irrigation_limit_m3_channel[farmer]):
            if farmer_gross_irrigation_demand_m3 > 0.0:
                irrigation_correction_factor: float = adjust_irrigation_to_limit(
                    farmer=farmer,
                    day_index=day_index,
                    remaining_irrigation_limit_m3=remaining_irrigation_limit_m3_channel,
                    cumulative_water_deficit_m3=cumulative_water_deficit_m3,
                    crop_calendar=crop_calendar,
                    crop_rotation_year_index=current_crop_calendar_rotation_year_index,
                    farmer_gross_irrigation_demand_m3=farmer_gross_irrigation_demand_m3,
                    irrigation_efficiency_farmer=irrigation_efficiency_farmer,
                    reset_day_index=irrigation_limit_reset_day_index[farmer],
                )
                assert 0 <= irrigation_correction_factor <= 1

                gross_potential_irrigation_m3_limit_adjusted_channel[farmer_fields] = (
                    gross_potential_irrigation_m3[farmer_fields]
                    * irrigation_correction_factor
                )
        else:
            gross_potential_irrigation_m3_limit_adjusted_channel[farmer_fields] = (
                gross_potential_irrigation_m3[farmer_fields]
            )

        if not np.isnan(remaining_irrigation_limit_m3_groundwater[farmer]):
            if farmer_gross_irrigation_demand_m3 > 0.0:
                irrigation_correction_factor: float = adjust_irrigation_to_limit(
                    farmer=farmer,
                    day_index=day_index,
                    remaining_irrigation_limit_m3=remaining_irrigation_limit_m3_groundwater,
                    cumulative_water_deficit_m3=cumulative_water_deficit_m3,
                    crop_calendar=crop_calendar,
                    crop_rotation_year_index=current_crop_calendar_rotation_year_index,
                    farmer_gross_irrigation_demand_m3=farmer_gross_irrigation_demand_m3,
                    irrigation_efficiency_farmer=irrigation_efficiency_farmer,
                    reset_day_index=irrigation_limit_reset_day_index[farmer],
                )
                assert 0 <= irrigation_correction_factor <= 1

                gross_potential_irrigation_m3_limit_adjusted_groundwater[
                    farmer_fields
                ] = (
                    gross_potential_irrigation_m3[farmer_fields]
                    * irrigation_correction_factor
                )
        else:
            gross_potential_irrigation_m3_limit_adjusted_groundwater[farmer_fields] = (
                gross_potential_irrigation_m3[farmer_fields]
            )

        assert (
            gross_potential_irrigation_m3[farmer_fields] / cell_area[farmer_fields]
        ).max() <= 1

    return (
        gross_potential_irrigation_m3,
        gross_potential_irrigation_m3_limit_adjusted_reservoir,
        gross_potential_irrigation_m3_limit_adjusted_channel,
        gross_potential_irrigation_m3_limit_adjusted_groundwater,
    )


@njit(cache=True)
def abstract_water(
    activation_order: npt.NDArray[np.int32],
    field_indices_by_farmer: npt.NDArray[np.int32],
    field_indices: npt.NDArray[np.int32],
    irrigation_efficiency: npt.NDArray[np.float32],
    surface_irrigated: npt.NDArray[np.bool_],
    well_irrigated: npt.NDArray[np.bool_],
    cell_area: npt.NDArray[np.float32],
    HRU_to_grid: npt.NDArray[np.int32],
    nearest_river_grid_cell: npt.NDArray[np.int32],
    crop_map: npt.NDArray[np.int32],
    available_channel_storage_m3: npt.NDArray[np.float32],
    available_groundwater_m3: npt.NDArray[np.float32],
    groundwater_depth: npt.NDArray[np.float32],
    available_reservoir_storage_m3: npt.NDArray[np.float32],
    command_area_by_farmer: npt.NDArray[np.int32],
    return_fraction: npt.NDArray[np.float32],
    well_depth: npt.NDArray[np.float32],
    remaining_irrigation_limit_m3_reservoir: npt.NDArray[np.float32],
    remaining_irrigation_limit_m3_channel: npt.NDArray[np.float32],
    remaining_irrigation_limit_m3_groundwater: npt.NDArray[np.float32],
    maximum_abstraction_reservoir_m3_by_farmer: npt.NDArray[np.float32],
    maximum_abstraction_channel_m3_by_farmer: npt.NDArray[np.float32],
    maximum_abstraction_groundwater_m3_by_farmer: npt.NDArray[np.float32],
    gross_irrigation_demand_m3_per_field: npt.NDArray[np.float32],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Perform irrigation withdrawals from channel, reservoir, and groundwater.

    Args:
        activation_order: Farmer activation order (e.g., by elevation).
        field_indices_by_farmer: Start/stop indices into ``field_indices`` per farmer.
        field_indices: Flat array of HRU indices for all fields (ordered by farmer).
        irrigation_efficiency: Per-farmer irrigation efficiency.
        surface_irrigated: Whether farmer can access surface-water sources.
        well_irrigated: Whether farmer has a functioning well.
        cell_area: HRU areas.
        HRU_to_grid: Map from HRU to grid cell.
        nearest_river_grid_cell: Nearest river cell per HRU.
        crop_map: Crop id per HRU (``-1`` for none).
        available_channel_storage_m3: Available channel storage per river cell (m³).
        available_groundwater_m3: Available groundwater per grid cell (m³).
        groundwater_depth: Groundwater depth per grid cell (m).
        available_reservoir_storage_m3: Available reservoir storage (m³).
        command_area_by_farmer: Reservoir command area per farmer (``-1`` if none).
        return_fraction: Per-farmer fraction of applied losses that return as return-flow.
        well_depth: Per-farmer well depth (m).
        remaining_irrigation_limit_m3_reservoir: Remaining seasonal limit (m³).
        remaining_irrigation_limit_m3_channel: Remaining seasonal limit (m³).
        remaining_irrigation_limit_m3_groundwater: Remaining seasonal limit (m³).
        maximum_abstraction_reservoir_m3_by_farmer: Cap per farmer (m³) from reservoir.
        maximum_abstraction_channel_m3_by_farmer: Cap per farmer (m³) from channel.
        maximum_abstraction_groundwater_m3_by_farmer: Cap per farmer (m³) from GW.
        gross_irrigation_demand_m3_per_field: Unadjusted gross demand per HRU (m³).

    Returns:
        Tuple with per-source totals and field-level fluxes:
        (channel_by_farmer, reservoir_by_farmer, groundwater_by_farmer,
         water_withdrawal_m, water_consumption_m,
         irrigation_return_flow_m, irrigation_evaporation_m,
         reservoir_abstraction_m3, groundwater_abstraction_m3).
    """
    n_hydrological_response_units = cell_area.size
    water_withdrawal_m = np.zeros(n_hydrological_response_units, dtype=np.float32)
    water_consumption_m = np.zeros(n_hydrological_response_units, dtype=np.float32)

    irrigation_return_flow_m = np.zeros(n_hydrological_response_units, dtype=np.float32)
    irrigation_evaporation_m = np.zeros(n_hydrological_response_units, dtype=np.float32)

    channel_abstraction_m3_by_farmer = np.zeros(activation_order.size, dtype=np.float32)
    reservoir_abstraction_m3_by_farmer = np.zeros(
        activation_order.size, dtype=np.float32
    )
    groundwater_abstraction_m3_by_farmer = np.zeros(
        activation_order.size, dtype=np.float32
    )

    # Because the groundwater and reservoir source are much larger than the surface water
    # and taking out small amounts cannot be represented by
    # floating point numbers, we need to use a separate array that tracks
    # the groundwater abstraction for each field. This is used to
    # then remove the groundwater and reservoir abstraction from the available
    # in one go, reducing the risk of (larger) floating point errors.
    groundwater_abstraction_m3 = np.zeros_like(available_groundwater_m3)
    reservoir_abstraction_m3 = np.zeros_like(available_reservoir_storage_m3)

    for farmer in activation_order:
        farmer_fields = get_farmer_HRUs(field_indices, field_indices_by_farmer, farmer)
        potential_irrigation_consumption_m3_farmer = (
            gross_irrigation_demand_m3_per_field[farmer_fields]
        )

        if potential_irrigation_consumption_m3_farmer.sum() <= 0:
            continue

        command_area_farmer = command_area_by_farmer[farmer]
        return_fraction_farmer = return_fraction[farmer]

        if surface_irrigated[farmer]:
            maximum_abstraction_channel_m3_farmer = (
                maximum_abstraction_channel_m3_by_farmer[farmer]
            )
            maximum_abstraction_channel_m3_by_field = (
                maximum_abstraction_channel_m3_farmer
                * potential_irrigation_consumption_m3_farmer
                / potential_irrigation_consumption_m3_farmer.sum()
            )
            if command_area_farmer == -1:  # -1 means no command area
                pass
            else:
                maximum_abstraction_reservoir_m3_farmer = (
                    maximum_abstraction_reservoir_m3_by_farmer[farmer]
                )
                maximum_abstraction_reservoir_m3_by_field = (
                    maximum_abstraction_reservoir_m3_farmer
                    * potential_irrigation_consumption_m3_farmer
                    / potential_irrigation_consumption_m3_farmer.sum()
                )

        if well_irrigated[farmer]:
            maximum_abstraction_groundwater_m3_farmer = (
                maximum_abstraction_groundwater_m3_by_farmer[farmer]
            )
            maximum_abstraction_groundwater_m3_by_field = (
                maximum_abstraction_groundwater_m3_farmer
                * potential_irrigation_consumption_m3_farmer
                / potential_irrigation_consumption_m3_farmer.sum()
            )

        # loop through all farmers fields and apply irrigation
        for field_index, field in enumerate(farmer_fields):
            grid_cell = HRU_to_grid[field]
            grid_cell_nearest = nearest_river_grid_cell[field]
            if crop_map[field] != -1:
                irrigation_water_demand_field_m = (
                    gross_irrigation_demand_m3_per_field[field] / cell_area[field]
                )
                assert 1 >= irrigation_water_demand_field_m >= 0
                if surface_irrigated[farmer]:
                    # command areas
                    if command_area_farmer != -1:  # -1 means no command area
                        irrigation_water_demand_field_m = withdraw_reservoir(
                            command_area=command_area_farmer,
                            field=field,
                            farmer=farmer,
                            reservoir_abstraction_m3=reservoir_abstraction_m3,
                            available_reservoir_storage_m3=available_reservoir_storage_m3,
                            irrigation_water_demand_field_m=irrigation_water_demand_field_m,
                            water_withdrawal_m=water_withdrawal_m,
                            remaining_irrigation_limit_m3=remaining_irrigation_limit_m3_reservoir,
                            reservoir_abstraction_m3_by_farmer=reservoir_abstraction_m3_by_farmer,
                            maximum_abstraction_reservoir_m3_field=maximum_abstraction_reservoir_m3_by_field[
                                field_index
                            ],
                            cell_area=cell_area,
                        )
                        assert water_withdrawal_m[field] >= 0
                        assert irrigation_water_demand_field_m >= 0

                    irrigation_water_demand_field_m = withdraw_channel(
                        available_channel_storage_m3=available_channel_storage_m3,
                        grid_cell=grid_cell_nearest,
                        cell_area=cell_area,
                        field=field,
                        farmer=farmer,
                        water_withdrawal_m=water_withdrawal_m,
                        irrigation_water_demand_field_m=irrigation_water_demand_field_m,
                        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3_channel,
                        channel_abstraction_m3_by_farmer=channel_abstraction_m3_by_farmer,
                        maximum_abstraction_channel_m3_field=maximum_abstraction_channel_m3_by_field[
                            field_index
                        ],
                        minimum_channel_storage_m3=100.0,
                    )
                    assert water_withdrawal_m[field] >= 0
                    assert irrigation_water_demand_field_m >= 0

                if well_irrigated[farmer]:
                    irrigation_water_demand_field_m = withdraw_groundwater(
                        farmer=farmer,
                        field=field,
                        grid_cell=grid_cell,
                        groundwater_abstraction_m3=groundwater_abstraction_m3,
                        available_groundwater_m3=available_groundwater_m3,
                        cell_area=cell_area,
                        groundwater_depth=groundwater_depth,
                        well_depth=well_depth,
                        irrigation_water_demand_field_m=irrigation_water_demand_field_m,
                        water_withdrawal_m=water_withdrawal_m,
                        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3_groundwater,
                        groundwater_abstraction_m3_by_farmer=groundwater_abstraction_m3_by_farmer,
                        maximum_abstraction_groundwater_m3_field=maximum_abstraction_groundwater_m3_by_field[
                            field_index
                        ],
                    )
                    assert irrigation_water_demand_field_m >= 0
                    assert water_withdrawal_m[field] >= 0

                assert (
                    irrigation_water_demand_field_m >= -1e15
                )  # Make sure irrigation water demand is zero, or positive. Allow very small error.

                assert water_consumption_m[field] >= 0
                assert water_withdrawal_m[field] >= 0
                assert 1 >= return_fraction_farmer >= 0

                water_consumption_m[field] = (
                    water_withdrawal_m[field] * irrigation_efficiency[farmer]
                )
                irrigation_loss_m = (
                    water_withdrawal_m[field] - water_consumption_m[field]
                )
                assert irrigation_loss_m >= 0
                irrigation_return_flow_m[field] = (
                    irrigation_loss_m * return_fraction_farmer
                )
                irrigation_evaporation_m[field] = (
                    irrigation_loss_m - irrigation_return_flow_m[field]
                )

    return (
        channel_abstraction_m3_by_farmer,
        reservoir_abstraction_m3_by_farmer,
        groundwater_abstraction_m3_by_farmer,
        water_withdrawal_m,
        water_consumption_m,
        irrigation_return_flow_m,
        irrigation_evaporation_m,
        reservoir_abstraction_m3,
        groundwater_abstraction_m3,
    )


@njit(cache=True)
def plant(
    n: int,
    day_index: int,
    crop_calendar: np.ndarray,
    current_crop_calendar_rotation_year_index: np.ndarray,
    crop_map: np.ndarray,
    crop_harvest_age_days: np.ndarray,
    cultivation_cost: np.ndarray | int | float,
    region_ids_per_farmer: np.ndarray,
    field_indices_by_farmer: np.ndarray,
    field_indices: np.ndarray,
    field_size_per_farmer: np.ndarray,
    all_loans_annual_cost: np.ndarray,
    loan_tracker: np.ndarray,
    interest_rate: np.ndarray,
    farmers_going_out_of_business: bool,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]:
    """Determines when and what crop should be planted, by comparing the current day to the next plant day. Also sets the harvest age of the plant.

    Args:
        n: Number of farmers.
        day_index: Current day index (0-indexed; Jan 1 == 0).
        crop_calendar: Crop calendar for each farmer. Each row is a farmer, and each column is a crop.
            Each crop is a list of [crop_type, planting_day, growing_days, crop_year_index].
            Planting day is 0-indexed (Jan 1 == 0).
            Growing days is the number of days the crop grows.
            Crop year index is the index of the year in the crop rotation.
        current_crop_calendar_rotation_year_index: Current crop calendar rotation year index for each farmer.
        crop_map: Map of the currently growing crops.
        crop_harvest_age_days: Array that contains the harvest age of each field in days.
        cultivation_cost: Cultivation cost per farmer per crop in m2. If this is a single value, it is used for all farmers and crops.
        region_ids_per_farmer: Region IDs for each farmer.
            This is used to determine the cultivation cost for each farmer.
        field_indices_by_farmer: This array contains the indices where the fields of a farmer are stored in `field_indices`.
        field_indices: This array contains the indices of all fields, ordered by farmer. In other words, if a farmer owns multiple fields, the indices of the fields are indices.
        field_size_per_farmer: Field size per farmer in m2
        all_loans_annual_cost: Annual cost of all loans for each farmer and crop.
            This is a 3D array with shape (n, 1, 4), where n is the number of farmers.
            The first dimension is the farmer, the second dimension is the crop, and the third dimension is the loan type.
        loan_tracker: Loan tracker for each farmer and crop. This is a 3D array with shape (n, 1, 4).
        interest_rate: Interest rate for each farmer.
        farmers_going_out_of_business: If True, farmers are going out of business. Not implemented yet.

    Returns:
        plant: Subarray map of what crops are planted this day.
        farmers_selling_land: Indices of farmers selling land (currently always empty).
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
            loan_duration = 1
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
def arrays_equal(a: npt.NDArray, b: npt.NDArray) -> bool:
    """Check element-wise equality for two same-shaped arrays.

    Args:
        a: First array.
        b: Second array.

    Returns:
        True if every element is equal, otherwise False.
    """
    for i in range(a.size):
        if a.flat[i] != b.flat[i]:
            return True if a.flat[i] != b.flat[i] else False
    return True


@njit(cache=True)
def find_matching_rows(
    arr: npt.NDArray[np.int32], target_row: npt.NDArray[np.int32]
) -> npt.NDArray[np.bool_]:
    """Find rows in ``arr`` that match ``target_row`` exactly.

    Args:
        arr: 2D array of shape (n_rows, n_cols).
        target_row: 1D array of shape (n_cols,).

    Returns:
        Boolean mask of length ``n_rows`` where True indicates a match.
    """
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
def find_most_similar_index(
    target_series: npt.NDArray[np.float32],
    yield_ratios: npt.NDArray[np.float32],
    groups: npt.NDArray[np.bool_],
) -> int:
    """Find index (within ``groups``) whose series is closest to ``target_series``.

    Args:
        target_series: 1D reference series.
        yield_ratios: 2D matrix where each row is a candidate series.
        groups: Boolean mask selecting allowable rows.

    Returns:
        Index (into ``yield_ratios``) of the closest series among ``groups``.
    """
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


@njit(cache=True, parallel=True)
def crop_profit_difference_njit_parallel(
    yearly_profits: npt.NDArray[np.float32],
    crop_elevation_group: npt.NDArray[np.int32],
    unique_crop_groups: npt.NDArray[np.int32],
    group_indices: npt.NDArray[np.int32],
    crop_calendar: npt.NDArray[np.int32],
    unique_crop_calendars: npt.NDArray[np.int32],
    p_droughts: npt.NDArray[np.float32],
    past_window: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Compute profit gains for alternative crop calendars (parallel over groups).

    Args:
        yearly_profits: Per-farmer yearly profits (recent years in columns).
        crop_elevation_group: Per-farmer feature rows for grouping.
        unique_crop_groups: Unique group keys.
        group_indices: For each farmer, index into ``unique_crop_groups``.
        crop_calendar: Per-farmer crop calendar.
        unique_crop_calendars: Candidate alternative calendars (rows).
        p_droughts: Drought probabilities (not used directly in this body).
        past_window: Number of past years to use (with a decaying weight).

    Returns:
        Tuple of:
            - gains_adaptation: Profit gain per farmer for each candidate calendar.
            - new_farmer_id: Example “nearest” farmer id per candidate and farmer.
    """
    n_groups: int = len(unique_crop_groups)
    n_calendars: int = len(unique_crop_calendars)
    n_rotation: int = unique_crop_calendars.shape[1]

    unique_profit_gain = np.full((n_groups, n_calendars), 0.0, dtype=np.float32)
    id_to_switch_to = np.full((n_groups, n_calendars), -1, dtype=np.int32)

    # Precompute weights
    weights = np.empty(past_window, dtype=yearly_profits.dtype)
    weight_total = 0.0
    if past_window > 1:
        weight_step = (0.5 - 0.2) / (past_window - 1)
    else:
        weight_step = 0.0

    for j in range(past_window):
        w = 0.5 - j * weight_step
        weights[j] = w
        weight_total += w

    # Parallel only over groups
    for group_id in prange(n_groups):  # ty: ignore[not-iterable]
        unique_group = unique_crop_groups[group_id]
        unique_farmer_groups = find_matching_rows(crop_elevation_group, unique_group)

        # build mask & collect candidate calendar‐indices
        candidate_idxs = np.empty(n_calendars, dtype=np.int32)
        num_cand = 0
        for i in range(n_calendars):
            match = True
            for j in range(n_rotation):
                if unique_crop_calendars[i, j] != unique_group[j]:
                    match = False
                    break
            if not match:
                candidate_idxs[num_cand] = i
                num_cand += 1

        metadata = unique_group[n_rotation:]

        # sequentially over each candidate (same order as original)
        for c in range(num_cand):
            cal_idx = candidate_idxs[c]
            unique_rotation = unique_crop_calendars[cal_idx]

            # form the “other” group key
            other_key = np.empty(unique_group.shape[0], unique_group.dtype)
            for k in range(n_rotation):
                other_key[k] = unique_rotation[k]
            for k in range(n_rotation, other_key.shape[0]):
                other_key[k] = metadata[k - n_rotation]

            unique_farmer_other = find_matching_rows(crop_elevation_group, other_key)

            if not np.any(unique_farmer_other):
                continue

            # current group weighted average
            current_sum = 0.0
            count_cur = 0
            for i in range(unique_farmer_groups.size):
                if unique_farmer_groups[i]:
                    wp = 0.0
                    for j in range(past_window):
                        wp += yearly_profits[i, j] * weights[j]
                    current_sum += wp / weight_total
                    count_cur += 1
            current_avg = current_sum / count_cur if count_cur > 0 else 0.0

            # candidate group weighted average
            cand_sum = 0.0
            count_cand = 0
            for i in range(unique_farmer_other.size):
                if unique_farmer_other[i]:
                    wp = 0.0
                    for j in range(past_window):
                        wp += yearly_profits[i, j] * weights[j]
                    cand_sum += wp / weight_total
                    count_cand += 1
            cand_avg = cand_sum / count_cand if count_cand > 0 else 0.0

            gain = cand_avg - current_avg
            if gain != gain:  # NaN check
                gain = 0.0
            else:
                id_to_switch_to[group_id, c] = find_most_similar_index(
                    gain, yearly_profits, unique_farmer_other
                )

            unique_profit_gain[group_id, c] = gain

    # Re‐index per‐farmer
    gains_adaptation = unique_profit_gain[group_indices, :]
    new_farmer_id = id_to_switch_to[group_indices, :]

    return gains_adaptation, new_farmer_id


@njit
def gev_ppf_scalar(u: float, c: float, loc: float, scale: float) -> float:
    """Scalar GEV inverse CDF.

    CDF = Cumulative Distribution Function
    PPF = Percent Point Function = Inverse CDF
    GEV = Generalized Extreme Value distribution

    Args:
        u: A uniform random variable in (0,1).
        c: GEV shape parameter.
        loc: GEV location parameter.
        scale: GEV scale parameter.

    Returns:
        The inverse CDF value for a given uniform random variable u and GEV parameters c, loc, scale.

    """
    if c != 0.0:
        return loc + scale * ((-np.log(u)) ** (-c) - 1.0) / c
    else:
        return loc - scale * np.log(-np.log(u))


@njit(cache=True, parallel=True)
def compute_premiums_and_best_contracts_numba(
    gev_params: npt.NDArray[np.float64],
    values_history: npt.NDArray[np.float64],
    losses: npt.NDArray[np.float64],
    strike_vals: npt.NDArray[np.float64],
    exit_vals: npt.NDArray[np.float64],
    rate_vals: npt.NDArray[np.float64],
    n_sims: int,
    seed: int = 42,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Computes the best insurance contracts for each agent based on GEV parameters, historical SPEI data, and losses.

    For each agent, loop once over (strike, exit):
    1) Monte Carlo → expected_ratio
    2) Historical SPEI → sum_ratio_sq, sum_ratio_loss )
    3) Over all rates → compute both premium = rate * expected_ratio
                            and RMSE via quadratic form,
        tracking the (strike_idx, exit_idx, rate_idx) that minimizes RMSE.

    Returns:
        best_strike_idx : (n_agents,)  index into strike_vals
        best_exit_idx   : (n_agents,)  index into exit_vals
        best_rate_idx   : (n_agents,)  index into rate_vals
        best_rmse       : (n_agents,)  the minimal RMSE for each agent
        best_premium    : (n_agents,)  the premium corresponding to that minimal‐RMSE contract
    """
    np.random.seed(seed)
    n_agents = gev_params.shape[0]
    n_years = values_history.shape[1]
    n_strikes = strike_vals.shape[0]
    n_exits = exit_vals.shape[0]
    n_rates = rate_vals.shape[0]

    # Output arrays
    best_strike_idx = np.empty(n_agents, dtype=np.int64)
    best_exit_idx = np.empty(n_agents, dtype=np.int64)
    best_rate_idx = np.empty(n_agents, dtype=np.int64)
    best_rmse_arr = np.empty(n_agents, dtype=np.float64)
    best_prem_arr = np.empty(n_agents, dtype=np.float64)

    for agent_idx in prange(n_agents):  # ty: ignore[not-iterable]
        shape = -gev_params[agent_idx, 0]
        loc = gev_params[agent_idx, 1]
        scale = gev_params[agent_idx, 2]

        # Precompute sum of squared losses for this agent
        loss_sq_sum = 0.0
        for yr in range(n_years):
            loss_sq_sum += losses[agent_idx, yr] ** 2

        # Initialize "best so far" placeholders
        best_rmse_val = 1e20
        best_s_idx = 0
        best_e_idx = 0
        best_r_idx = 0
        best_premium = 0.0

        # Loop once over all (strike, exit) pairs
        for strike_idx in range(n_strikes):
            strike = strike_vals[strike_idx]

            for exit_idx in range(n_exits):
                exit_threshold = exit_vals[exit_idx]
                if exit_threshold >= strike:
                    continue
                denom = strike - exit_threshold

                # Monte Carlo → expected_ratio
                expected_ratio = 0.0
                for _ in range(n_sims):
                    u = np.random.random()  # Uniform(0,1)
                    spei_val = gev_ppf_scalar(u, shape, loc, scale)
                    shortfall = strike - spei_val
                    if shortfall <= 0.0:
                        continue
                    if shortfall > denom:
                        shortfall = denom
                    expected_ratio += shortfall / denom
                expected_ratio /= n_sims

                sum_ratio_sq = 0.0
                sum_ratio_loss = 0.0
                for yr in range(n_years):
                    shortfall = strike - values_history[agent_idx, yr]
                    if shortfall <= 0.0:
                        ratio = 0.0
                    elif shortfall >= denom:
                        ratio = 1.0
                    else:
                        ratio = shortfall / denom

                    sum_ratio_sq += ratio * ratio
                    sum_ratio_loss += ratio * losses[agent_idx, yr]

                # Over all candidate rates, compute premium & RMSE via quadratic form
                for rate_idx in range(n_rates):
                    rate = rate_vals[rate_idx]
                    # RMSE’s sum of squared errors:
                    sse = (
                        (rate * rate) * sum_ratio_sq
                        - 2.0 * rate * sum_ratio_loss
                        + loss_sq_sum
                    )
                    rmse_val = np.sqrt(sse / n_years)

                    if rmse_val < best_rmse_val:
                        best_rmse_val = rmse_val
                        best_s_idx = strike_idx
                        best_e_idx = exit_idx
                        best_r_idx = rate_idx
                        best_premium = rate * expected_ratio

        # Store best‐so‐far for this agent
        best_strike_idx[agent_idx] = best_s_idx
        best_exit_idx[agent_idx] = best_e_idx
        best_rate_idx[agent_idx] = best_r_idx
        best_rmse_arr[agent_idx] = best_rmse_val
        best_prem_arr[agent_idx] = best_premium

    return (
        best_strike_idx,
        best_exit_idx,
        best_rate_idx,
        best_rmse_arr,
        best_prem_arr,
    )
