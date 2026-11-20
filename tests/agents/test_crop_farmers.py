"""Tests for the crop farmers agent functions."""

import numpy as np

from geb.agents.crop_farmers import (
    advance_crop_rotation_year,
    cumulative_mean,
    shift_and_update,
)
from geb.agents.workflows.crop_farmers import (
    abstract_water,
    accumulate_farmer_pr_and_water_deficit,
    adjust_irrigation_to_limit,
    get_deficit_between_dates,
    get_future_deficit,
    withdraw_channel,
    withdraw_groundwater,
    withdraw_reservoir,
)
from geb.store import DynamicArray


def test_cumulative_mean() -> None:
    """Test the cumulative_mean function."""
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    mean = a.mean()
    assert mean == 4.5

    # test normal operation
    cumulative_mean_ = DynamicArray(np.array([0], dtype=np.float32))
    cumulative_count = DynamicArray(np.array([0], dtype=np.float32))
    for item in a:
        cumulative_mean(cumulative_mean_, cumulative_count, item)

    assert cumulative_mean_ == 4.5

    # test with masked array
    cumulative_mean_.fill(0)
    cumulative_count.fill(0)

    for item in a:
        cumulative_mean(cumulative_mean_[:], cumulative_count[:], item)

    assert cumulative_mean_ == 4.5


def test_shift_and_update() -> None:
    """Test the shift_and_update function."""
    a = DynamicArray(
        np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), extra_dims_names=["test"]
    )
    shift_and_update(a, DynamicArray(np.array([9, 10, 11])))
    assert np.array_equal(a, np.array([[9, 0, 1], [10, 3, 4], [11, 6, 7]]))


def test_get_deficit_between_dates() -> None:
    """Test the get_deficit_between_dates function.

    This is tested for various combinations of start and end dates, focussing
    on edge cases like wrapping around the end of the year.
    """
    cumulative_water_deficit_m3 = np.expand_dims(
        np.arange(0, 3660, 10, dtype=np.float32), axis=0
    )
    deficit = get_deficit_between_dates(cumulative_water_deficit_m3, 0, 1, 6)
    assert deficit == 50.0

    deficit = get_deficit_between_dates(cumulative_water_deficit_m3, 0, 1, 1)
    assert deficit == 0.0

    deficit = get_deficit_between_dates(cumulative_water_deficit_m3, 0, 1, 365)
    assert deficit == 3640.0

    deficit = get_deficit_between_dates(cumulative_water_deficit_m3, 0, 365, 1)
    assert deficit == 10.0

    deficit = get_deficit_between_dates(cumulative_water_deficit_m3, 0, 365, 5)
    assert deficit == 50.0

    deficit = get_deficit_between_dates(cumulative_water_deficit_m3, 0, 365, 365)
    assert deficit == 0.0

    deficit = get_deficit_between_dates(cumulative_water_deficit_m3, 0, 5, 1)
    assert deficit == 3610.0

    deficit = get_deficit_between_dates(cumulative_water_deficit_m3, 0, 5, 5)
    assert deficit == 0.0

    deficit = get_deficit_between_dates(cumulative_water_deficit_m3, 0, 5, 365)
    assert deficit == 3600.0


def test_get_future_deficit() -> None:
    """Test the get_future_deficit function."""
    cumulative_water_deficit_m3 = np.expand_dims(
        np.arange(0, 3660, 10, dtype=np.float32), axis=0
    )
    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=0,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 1, 5, 0],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=0,
    )
    assert future_water_deficit == 60.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=0,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 1, 5, 0],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=364,
    )
    assert future_water_deficit == 60.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=0,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 364, 5, 0],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=0,
    )
    assert future_water_deficit == 20.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=180,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 180, 5, 0],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=181,
    )
    assert future_water_deficit == 10.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=0,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 2, 5, 0],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=0,
    )
    assert future_water_deficit == 60.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=0,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 2, 5, 0],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=181,
    )
    assert future_water_deficit == 60.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=0,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 185, 5, 0],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=0,
        reset_day_index=181,
    )
    assert future_water_deficit == 0.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=0,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 0, 1, 0],
                    [0, 3, 1, 0],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=0,
    )
    assert future_water_deficit == 20.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=0,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 0, 1, 0],
                    [0, 3, 1, 0],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=181,
    )
    assert future_water_deficit == 20.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=360,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 360, 10, 0],  # crop grows beyond end of year
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=0,
    )
    assert future_water_deficit == 50.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=360,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 360, 10, 0],  # crop grows beyond end of year
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=364,
    )
    assert future_water_deficit == 40.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=360,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 360, 10, 0],  # crop grows beyond end of year
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=181,
    )
    assert future_water_deficit == 100.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=3,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 360, 15, 0],  # crop grows beyond end of year
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=0,
    )
    assert future_water_deficit == 60.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=3,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 360, 15, 0],  # crop grows beyond end of year
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=0,
        reset_day_index=181,
    )
    assert future_water_deficit == 60.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=200,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 100, 50, 0],  # crop grows beyond end of year
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=0,
        reset_day_index=0,
    )
    assert future_water_deficit == 0.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=200,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 100, 50, 0],  # crop grows beyond end of year
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=0,
        reset_day_index=181,
    )
    assert future_water_deficit == 500.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=200,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 100, 100, 0],  # crop grows beyond end of year
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=0,
        reset_day_index=0,
    )
    assert future_water_deficit == 0.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=200,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 100, 370, 0],  # crop grows beyond end of year
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=0,
        reset_day_index=0,
    )
    assert future_water_deficit == 1640.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=200,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 100, 370, 0],  # crop grows beyond end of year
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=0,
        reset_day_index=181,
    )
    assert future_water_deficit == 810.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=0,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 0, 1, 0],
                    [0, 3, 1, 1],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=0,
    )
    assert future_water_deficit == 10.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=182,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 182, 1, 0],
                    [0, 185, 1, 1],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=181,
    )
    assert future_water_deficit == 10.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=0,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 0, 1, 0],
                    [0, 3, 1, 1],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([1]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=0,
    )
    assert future_water_deficit == 20.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=182,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 182, 1, 0],
                    [0, 185, 1, 1],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([1]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=181,
    )
    assert future_water_deficit == 20.0

    future_water_deficit = get_future_deficit(
        farmer=0,
        day_index=365,
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=np.array(
            [
                [
                    [0, 360, 20, 0],
                    [0, 3, 1, 1],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=np.array([0]),
        potential_irrigation_consumption_farmer_m3=10,
        reset_day_index=0,
    )
    assert future_water_deficit == 60.0

    future_water_deficits = np.full(365, np.nan)
    for day_index in range(365):
        future_water_deficit = get_future_deficit(
            farmer=0,
            day_index=day_index,
            cumulative_water_deficit_m3=cumulative_water_deficit_m3,
            crop_calendar=np.array(
                [
                    [
                        [0, 1, 364, 0],
                    ]
                ]
            ),  # crop ID (irrelevant here), planting day, growing days
            crop_rotation_year_index=np.array([0]),
            potential_irrigation_consumption_farmer_m3=10,
            reset_day_index=0,
        )
        future_water_deficits[day_index] = future_water_deficit
    assert np.array_equal(future_water_deficits, np.arange(3650, 0, -10))

    future_water_deficits = np.full(365, np.nan)
    for day_index in range(365):
        future_water_deficit = get_future_deficit(
            farmer=0,
            day_index=day_index,
            cumulative_water_deficit_m3=cumulative_water_deficit_m3,
            crop_calendar=np.array(
                [
                    [
                        [0, 1, 364, 0],
                    ]
                ]
            ),  # crop ID (irrelevant here), planting day, growing days
            crop_rotation_year_index=np.array([0]),
            potential_irrigation_consumption_farmer_m3=10,
            reset_day_index=180,
        )
        future_water_deficits[day_index] = future_water_deficit
    assert future_water_deficits[180] == 1800.0
    assert np.array_equal(
        future_water_deficits,
        np.concatenate([np.arange(1800, 0, -10), np.full(185, 1800)]),
    )


def test_adjust_irrigation_to_limit() -> None:
    """Test the adjust_irrigation_to_limit function."""
    cumulative_water_deficit_m3 = np.array([[0.0, 15.0, 30.0]])
    # use full crop calendar with growing days 2 to match cumulative_water_deficit_m3
    crop_calendar = np.array([[[0, 0, 2, 0]]])
    farmer_gross_irrigation_demand_m3 = 20
    reduction_factor = adjust_irrigation_to_limit(
        farmer=0,
        day_index=0,
        remaining_irrigation_limit_m3=np.array([10]),
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=crop_calendar,
        crop_rotation_year_index=np.array([0]),
        farmer_gross_irrigation_demand_m3=farmer_gross_irrigation_demand_m3,
        irrigation_efficiency_farmer=0.5,
        reset_day_index=0,
    )
    adjusted_farmer_gross_irrigation_demand_m3 = (
        farmer_gross_irrigation_demand_m3 * reduction_factor
    )
    # In the future, still 15 + (20 * 0.5) irrigation consumption is needed
    # irrigation limit remaining is 10, and 10 requested today, so only 10/25*10 = 4. irrigation is possible
    np.testing.assert_almost_equal(
        adjusted_farmer_gross_irrigation_demand_m3, np.array([4.0])
    )

    # On the last day of the year, the farmer should be able to irrigate the remaining water deficit

    farmer_gross_irrigation_demand_m3 = 20
    reduction_factor = adjust_irrigation_to_limit(
        farmer=0,
        day_index=1,  # last day of fictitious year
        remaining_irrigation_limit_m3=np.array([5.0]),
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=crop_calendar,
        crop_rotation_year_index=np.array([0]),
        irrigation_efficiency_farmer=0.5,
        farmer_gross_irrigation_demand_m3=farmer_gross_irrigation_demand_m3,
        reset_day_index=0,
    )
    adjusted_farmer_gross_irrigation_demand_m3 = (
        farmer_gross_irrigation_demand_m3 * reduction_factor
    )

    # In the future, still 10 irrigation is needed
    # irrigation limit remaining is 5, and (20 * 0.5) requested today, so only 10/10*5 = 5. irrigation is possible
    # which is all irrigation for the last day of the year
    np.testing.assert_almost_equal(
        adjusted_farmer_gross_irrigation_demand_m3, np.array([5.0])
    )

    # If no allowed irrigation, the farmer should not irrigate
    farmer_gross_irrigation_demand_m3 = 20
    reduction_factor = adjust_irrigation_to_limit(
        farmer=0,
        day_index=1,  # last day of fictitious year
        remaining_irrigation_limit_m3=np.array([0]),
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=crop_calendar,
        crop_rotation_year_index=np.array([0]),
        irrigation_efficiency_farmer=0.5,
        farmer_gross_irrigation_demand_m3=farmer_gross_irrigation_demand_m3,
        reset_day_index=0,
    )
    adjusted_farmer_gross_irrigation_demand_m3 = (
        farmer_gross_irrigation_demand_m3 * reduction_factor
    )
    # As irrigation limit is 0, no irrigation is possible
    np.testing.assert_almost_equal(
        adjusted_farmer_gross_irrigation_demand_m3, np.array([0.0])
    )


def test_withdraw_groundwater() -> None:
    """Test the withdraw_groundwater function."""
    available_groundwater_m3 = np.array([2000.0])
    water_withdrawal_m = np.array([0.0])
    remaining_irrigation_limit_m3 = np.array([np.nan])
    groundwater_abstraction_m3_by_farmer = np.array([0.0])
    groundwater_abstraction_m3 = np.zeros_like(available_groundwater_m3)

    irrigation_water_demand_field = withdraw_groundwater(
        farmer=0,
        field=0,
        grid_cell=0,
        available_groundwater_m3=available_groundwater_m3,
        groundwater_abstraction_m3=groundwater_abstraction_m3,
        cell_area=np.array([100.0]),
        groundwater_depth=np.array([20.0]),
        well_depth=np.array([30.0]),
        irrigation_water_demand_field_m=10.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        groundwater_abstraction_m3_by_farmer=groundwater_abstraction_m3_by_farmer,
        maximum_abstraction_groundwater_m3_field=np.inf,
    )
    available_groundwater_m3 -= groundwater_abstraction_m3
    assert irrigation_water_demand_field == 0.0
    assert groundwater_abstraction_m3[0] == 1000.0
    assert water_withdrawal_m[0] == 10.0
    assert groundwater_abstraction_m3_by_farmer[0] == 1000.0

    groundwater_abstraction_m3.fill(0.0)

    irrigation_water_demand_field = withdraw_groundwater(
        farmer=0,
        field=0,
        grid_cell=0,
        groundwater_abstraction_m3=groundwater_abstraction_m3,
        available_groundwater_m3=available_groundwater_m3,
        cell_area=np.array([100.0]),
        groundwater_depth=np.array([20.0]),
        well_depth=np.array([30.0]),
        irrigation_water_demand_field_m=20.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        groundwater_abstraction_m3_by_farmer=groundwater_abstraction_m3_by_farmer,
        maximum_abstraction_groundwater_m3_field=np.inf,
    )
    assert irrigation_water_demand_field == 10.0
    assert groundwater_abstraction_m3[0] == 1000.0
    assert water_withdrawal_m[0] == 20.0
    assert groundwater_abstraction_m3_by_farmer[0] == 2000.0

    # if the well depth is less than the groundwater depth, no water can be withdrawn
    water_withdrawal_m = np.array([0.0])
    available_groundwater_m3 = np.array([2000.0])
    groundwater_abstraction_m3_by_farmer = np.array([0.0])
    groundwater_abstraction_m3.fill(0.0)
    irrigation_water_demand_field = withdraw_groundwater(
        farmer=0,
        field=0,
        grid_cell=0,
        groundwater_abstraction_m3=groundwater_abstraction_m3,
        available_groundwater_m3=available_groundwater_m3,
        cell_area=np.array([100.0]),
        groundwater_depth=np.array([20.0]),
        well_depth=np.array([10.0]),
        irrigation_water_demand_field_m=20.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        groundwater_abstraction_m3_by_farmer=groundwater_abstraction_m3_by_farmer,
        maximum_abstraction_groundwater_m3_field=np.inf,
    )
    assert irrigation_water_demand_field == 20.0
    assert groundwater_abstraction_m3[0] == 0.0
    assert water_withdrawal_m[0] == 0.0
    assert groundwater_abstraction_m3_by_farmer[0] == 0.0


def test_withdraw_channel() -> None:
    """Test the withdraw_channel function."""
    available_channel_storage_m3 = np.array([2000.0])
    water_withdrawal_m = np.array([0.0])
    remaining_irrigation_limit_m3 = np.array([np.nan])
    channel_abstraction_m3_by_farmer = np.array([0.0])

    irrigation_water_demand_field = withdraw_channel(
        available_channel_storage_m3=available_channel_storage_m3,
        grid_cell=0,
        cell_area=np.array([100.0]),
        field=0,
        farmer=0,
        irrigation_water_demand_field_m=10.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        channel_abstraction_m3_by_farmer=channel_abstraction_m3_by_farmer,
        maximum_abstraction_channel_m3_field=np.inf,
    )
    assert irrigation_water_demand_field == 0.0
    assert available_channel_storage_m3[0] == 1000.0
    assert water_withdrawal_m[0] == 10.0
    assert channel_abstraction_m3_by_farmer[0] == 1000.0

    irrigation_water_demand_field = withdraw_channel(
        field=0,
        grid_cell=0,
        farmer=0,
        cell_area=np.array([100.0]),
        available_channel_storage_m3=available_channel_storage_m3,
        irrigation_water_demand_field_m=20.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        channel_abstraction_m3_by_farmer=channel_abstraction_m3_by_farmer,
        maximum_abstraction_channel_m3_field=np.inf,
    )
    # the minimum channel storage is excluded from the available storage
    # beforehand, so all available storage can be abstracted
    assert irrigation_water_demand_field == 10.0
    assert available_channel_storage_m3[0] == 0.0
    assert water_withdrawal_m[0] == 20.0
    assert channel_abstraction_m3_by_farmer[0] == 2000.0

    # the channel share of the demand, adjusted to the channel limit, caps the withdrawal
    available_channel_storage_m3 = np.array([2000.0])
    remaining_irrigation_limit_m3 = np.array([300.0])
    irrigation_water_demand_field = withdraw_channel(
        field=0,
        grid_cell=0,
        farmer=0,
        cell_area=np.array([100.0]),
        available_channel_storage_m3=available_channel_storage_m3,
        irrigation_water_demand_field_m=10.0,
        water_withdrawal_m=np.array([0.0]),
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        channel_abstraction_m3_by_farmer=np.array([0.0]),
        maximum_abstraction_channel_m3_field=300.0,
    )
    assert irrigation_water_demand_field == 7.0
    assert available_channel_storage_m3[0] == 1700.0
    assert remaining_irrigation_limit_m3[0] == 0.0


def test_withdraw_reservoir() -> None:
    """Test the withdraw_reservoir function."""
    available_reservoir_storage_m3 = np.array([2000.0])
    water_withdrawal_m = np.array([0.0])
    remaining_irrigation_limit_m3 = np.array([np.nan])
    reservoir_abstraction_m3_by_farmer = np.array([0.0])
    reservoir_abstraction_m3 = np.zeros_like(available_reservoir_storage_m3)

    irrigation_water_demand_field = withdraw_reservoir(
        command_area=0,
        field=0,
        farmer=0,
        reservoir_abstraction_m3=reservoir_abstraction_m3,
        available_reservoir_storage_m3=available_reservoir_storage_m3,
        irrigation_water_demand_field_m=10.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        reservoir_abstraction_m3_by_farmer=reservoir_abstraction_m3_by_farmer,
        maximum_abstraction_reservoir_m3_field=np.inf,
        cell_area=np.array([100.0]),
    )
    available_reservoir_storage_m3 -= reservoir_abstraction_m3
    assert irrigation_water_demand_field == 0.0
    assert available_reservoir_storage_m3[0] == 1000.0
    assert water_withdrawal_m[0] == 10.0
    assert reservoir_abstraction_m3_by_farmer[0] == 1000.0

    reservoir_abstraction_m3 = np.zeros_like(available_reservoir_storage_m3)
    irrigation_water_demand_field = withdraw_reservoir(
        command_area=0,
        field=0,
        farmer=0,
        reservoir_abstraction_m3=reservoir_abstraction_m3,
        available_reservoir_storage_m3=available_reservoir_storage_m3,
        irrigation_water_demand_field_m=20.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        reservoir_abstraction_m3_by_farmer=reservoir_abstraction_m3_by_farmer,
        maximum_abstraction_reservoir_m3_field=np.inf,
        cell_area=np.array([100.0]),
    )
    available_reservoir_storage_m3 -= reservoir_abstraction_m3
    assert irrigation_water_demand_field == 10.0
    assert available_reservoir_storage_m3[0] == 0.0
    assert water_withdrawal_m[0] == 20.0
    assert reservoir_abstraction_m3_by_farmer[0] == 2000.0


def test_withdraw_reservoir_limit_demand() -> None:
    """Test the withdraw_reservoir function with limited demand."""
    # test with maximum abstraction reservoir
    available_reservoir_storage_m3 = np.array([2000.0])
    water_withdrawal_m = np.array([0.0])
    remaining_irrigation_limit_m3 = np.array([np.nan])
    reservoir_abstraction_m3_by_farmer = np.array([0.0])
    reservoir_abstraction_m3 = np.zeros_like(available_reservoir_storage_m3)
    maximum_abstraction_reservoir_m3_field = 1000.0
    irrigation_water_demand_field = withdraw_reservoir(
        command_area=0,
        field=0,
        farmer=0,
        reservoir_abstraction_m3=reservoir_abstraction_m3,
        available_reservoir_storage_m3=available_reservoir_storage_m3,
        irrigation_water_demand_field_m=20.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        reservoir_abstraction_m3_by_farmer=reservoir_abstraction_m3_by_farmer,
        cell_area=np.array([100.0]),
        maximum_abstraction_reservoir_m3_field=maximum_abstraction_reservoir_m3_field,
    )
    available_reservoir_storage_m3 -= reservoir_abstraction_m3
    assert irrigation_water_demand_field == 10.0  # only 10 should be withdrawn
    assert maximum_abstraction_reservoir_m3_field == reservoir_abstraction_m3
    assert (
        available_reservoir_storage_m3[0] == 1000.0
    )  # thus there is still 1000 m3 left
    assert water_withdrawal_m[0] == 10.0
    assert reservoir_abstraction_m3_by_farmer[0] == 1000.0


def test_withdraw_reservoir_maximum_abstraction() -> None:
    """Test the withdraw_reservoir function with maximum abstraction."""
    # test with maximum abstraction reservoir
    available_reservoir_storage_m3 = np.array([2000.0])
    water_withdrawal_m = np.array([0.0])
    remaining_irrigation_limit_m3 = np.array([np.nan])
    reservoir_abstraction_m3_by_farmer = np.array([0.0])
    reservoir_abstraction_m3 = np.zeros_like(available_reservoir_storage_m3)
    maximum_abstraction_reservoir_m3_field = 1000.0
    irrigation_water_demand_field = withdraw_reservoir(
        command_area=0,
        field=0,
        farmer=0,
        reservoir_abstraction_m3=reservoir_abstraction_m3,
        available_reservoir_storage_m3=available_reservoir_storage_m3,
        irrigation_water_demand_field_m=20.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        reservoir_abstraction_m3_by_farmer=reservoir_abstraction_m3_by_farmer,
        cell_area=np.array([100.0]),
        maximum_abstraction_reservoir_m3_field=maximum_abstraction_reservoir_m3_field,
    )
    available_reservoir_storage_m3 -= reservoir_abstraction_m3
    assert irrigation_water_demand_field == 10.0  # only 10 should be withdrawn
    assert maximum_abstraction_reservoir_m3_field == reservoir_abstraction_m3
    assert (
        available_reservoir_storage_m3[0] == 1000.0
    )  # thus there is still 1000 m3 left
    assert water_withdrawal_m[0] == 10.0
    assert reservoir_abstraction_m3_by_farmer[0] == 1000.0


def test_advance_crop_rotation_year() -> None:
    """Test the advance_crop_rotation_year function."""
    current_crop_calendar_rotation_year_index = DynamicArray(np.array([0, 1, 0, 6, 0]))
    crop_calendar_rotation_years = DynamicArray(np.array([1, 2, 2, 7, 10]))
    advance_crop_rotation_year(
        current_crop_calendar_rotation_year_index=current_crop_calendar_rotation_year_index,
        crop_calendar_rotation_years=crop_calendar_rotation_years,
    )
    assert np.array_equal(
        current_crop_calendar_rotation_year_index, np.array([0, 0, 1, 0, 1])
    )


def test_accumulate_farmer_pr_and_water_deficit() -> None:
    """Test parallel accumulation of daily precipitation and water deficit per farmer."""
    # 4 HRUs, 3 farmers: farmer 0 has HRU 0 and 1; farmer 1 has HRU 2; farmer 2 has 0 HRUs (edge case)
    pr_kg_per_m2_per_s = np.zeros((4, 24), dtype=np.float32)
    # HRU 0: 1 mm/hour = 1/3600 kg/m2/s for all 24h -> 24 mm/day = 0.024 m/day
    pr_kg_per_m2_per_s[0, :] = 1.0 / 3600.0
    # HRU 1: 2 mm/hour = 2/3600 kg/m2/s for all 24h -> 48 mm/day = 0.048 m/day
    pr_kg_per_m2_per_s[1, :] = 2.0 / 3600.0
    # HRU 2: 0.5 mm/hour for all 24h -> 12 mm/day = 0.012 m/day
    pr_kg_per_m2_per_s[2, :] = 0.5 / 3600.0
    # HRU 3: not owned by farmers

    ref_et_grass_m_per_day = np.array([0.030, 0.040, 0.005, 0.010], dtype=np.float32)
    cell_area = np.array([100.0, 200.0, 50.0, 100.0], dtype=np.float32)

    # farmer 0: fields in indices [0:2] (HRUs 0, 1)
    # farmer 1: fields in indices [2:3] (HRU 2)
    # farmer 2: fields in indices [3:3] (0 fields, edge case)
    field_indices_by_farmer = np.array([[0, 2], [2, 3], [3, 3]], dtype=np.int32)
    field_indices = np.array([0, 1, 2], dtype=np.int32)

    out_pr_mm = np.empty(3, dtype=np.float32)
    out_deficit_m3 = np.empty(3, dtype=np.float32)

    accumulate_farmer_pr_and_water_deficit(
        pr_kg_per_m2_per_s=pr_kg_per_m2_per_s,
        reference_evapotranspiration_grass_m_per_day=ref_et_grass_m_per_day,
        cell_area=cell_area,
        field_indices_by_farmer=field_indices_by_farmer,
        field_indices=field_indices,
        out_pr_mm_per_day=out_pr_mm,
        out_water_deficit_m3=out_deficit_m3,
        HRU_to_grid=np.empty(0, dtype=np.int32),
    )

    # Farmer 0: mean pr = (24 + 48) / 2 = 36 mm/day
    # Deficit: HRU 0: max(0, 0.030 - 0.024) * 100 = 0.006 * 100 = 0.6 m3
    #          HRU 1: max(0, 0.040 - 0.048) * 200 = 0 m3
    # Total deficit = 0.6 m3
    assert np.isclose(out_pr_mm[0], 36.0, atol=1e-4)
    assert np.isclose(out_deficit_m3[0], 0.6, atol=1e-4)

    # Farmer 1: mean pr = 12 mm/day
    # Deficit: HRU 2: max(0, 0.005 - 0.012) * 50 = 0 m3
    assert np.isclose(out_pr_mm[1], 12.0, atol=1e-4)
    assert np.isclose(out_deficit_m3[1], 0.0, atol=1e-4)

    # Farmer 2: 0 fields -> both 0.0
    assert out_pr_mm[2] == 0.0
    assert out_deficit_m3[2] == 0.0


def test_field_to_farmer() -> None:
    """Test CropFarmers.field_to_farmer aggregation and fast paths."""
    from unittest.mock import MagicMock

    from geb.agents.crop_farmers import CropFarmers

    farmers = CropFarmers.__new__(CropFarmers)
    farmers.var = MagicMock()
    farmers.var.n = 3
    farmers.HRU = MagicMock()
    farmers.HRU.var = MagicMock()
    # Fields 0 belongs to farmer 0, field 1 belongs to farmer 1, field 2 is unowned (-1), field 3 belongs to farmer 0
    farmers.HRU.var.land_owners = np.array([0, 1, -1, 0], dtype=np.int32)

    # Fast path: all zeros
    zero_vals = np.zeros(4, dtype=np.float32)
    res_zero = farmers.field_to_farmer(zero_vals)
    assert np.array_equal(res_zero, np.zeros(3, dtype=np.float32))

    # Fast path: empty array
    empty_vals = np.array([], dtype=np.float32)
    res_empty = farmers.field_to_farmer(empty_vals)
    assert np.array_equal(res_empty, np.zeros(3, dtype=np.float32))

    # Non-zero values: field 0 has 10.0, field 1 has 5.0, field 2 has 99.0 (unowned), field 3 has 2.5
    values = np.array([10.0, 5.0, 99.0, 2.5], dtype=np.float32)
    res = farmers.field_to_farmer(values)
    # Farmer 0: 10.0 + 2.5 = 12.5; Farmer 1: 5.0; Farmer 2: 0.0
    assert np.allclose(res, np.array([12.5, 5.0, 0.0], dtype=np.float32))


def test_update_field_indices_numba() -> None:
    """Field indices are correct both with and without unowned land."""
    from geb.agents.crop_farmers import CropFarmers

    # all land owned: previously field_indices[:-0] returned an empty array
    land_owners = np.array([1, 0, 1, 2], dtype=np.int32)
    by_farmer, field_indices = CropFarmers.update_field_indices_numba(land_owners)
    assert field_indices.size == 4
    for farmer in range(3):
        start, end = by_farmer[farmer]
        assert sorted(field_indices[start:end]) == sorted(
            np.where(land_owners == farmer)[0]
        )

    # with unowned land
    land_owners = np.array([-1, 1, 0, -1, 1], dtype=np.int32)
    by_farmer, field_indices = CropFarmers.update_field_indices_numba(land_owners)
    assert field_indices.size == 3
    assert (field_indices >= 0).all()
    for farmer in range(2):
        start, end = by_farmer[farmer]
        assert sorted(field_indices[start:end]) == sorted(
            np.where(land_owners == farmer)[0]
        )


def test_get_potential_irrigation_consumption_m_trigger() -> None:
    """Non-paddy irrigation starts when depletion exceeds RAW = 0.5 * TAW."""
    from geb.agents.workflows.crop_farmers import (
        get_potential_irrigation_consumption_m,
    )
    from geb.hydrology.landsurface.constants import N_SOIL_LAYERS

    layer_height = np.full(N_SOIL_LAYERS, 0.1, dtype=np.float32)
    # volumetric field capacity 0.3, wilting point 0.1 -> TAW over 0.1 m layers
    field_capacity = layer_height * np.float32(0.3)
    wilting_point = layer_height * np.float32(0.1)
    ksat = np.full(N_SOIL_LAYERS, 10.0, dtype=np.float32)

    def consumption(depleted_fraction_of_taw: float) -> float:
        w = field_capacity - (field_capacity - wilting_point) * np.float32(
            depleted_fraction_of_taw
        )
        return float(
            get_potential_irrigation_consumption_m(
                topwater=np.float32(0),
                root_depth_m=np.float32(0.3),
                soil_layer_height=layer_height,
                field_capacity=field_capacity,
                wilting_point=wilting_point,
                w=w.astype(np.float32),
                saturated_hydraulic_conductivity_m_per_day=ksat,
                fraction_irrigated_field=np.float32(1),
                max_paddy_water_level_farmer=np.float32(0.05),
                is_paddy=np.bool_(False),
                minimum_effective_root_depth_m=np.float32(0.25),
            )
        )

    # 40 % of TAW depleted: below RAW, no irrigation
    assert consumption(0.4) == 0
    # 60 % of TAW depleted: refill to field capacity (TAW = 0.3 m * 0.2 = 0.06 m)
    np.testing.assert_allclose(consumption(0.6), 0.6 * 0.06, rtol=1e-5)


def test_get_yield_ratio_numba_GAEZ_bounds() -> None:
    """GAEZ yield ratios are bounded to [0, 1]."""
    from geb.agents.crop_farmers import CropFarmers

    ky = np.array([1.2], dtype=np.float32)
    evaporation_ratio = np.array([1.3, 0.5, 0.0], dtype=np.float32)
    per_stage = np.full((6, 3), np.nan, dtype=np.float32)
    per_stage[2, 0] = 1.5  # actual ET above potential ET in one stage
    yield_ratio = CropFarmers.get_yield_ratio_numba_GAEZ(
        np.zeros(3, dtype=np.int32),
        evaporation_ratio=evaporation_ratio,
        evaporation_ratio_per_crop_stage=per_stage,
        KyT=ky,
        Ky1=ky,
        Ky2a=ky,
        Ky2b=ky,
        Ky3a=ky,
        Ky3b=ky,
        Ky4=ky,
    )
    np.testing.assert_allclose(yield_ratio, [1.0, 0.4, 0.0], rtol=1e-6)


def test_abstract_water_irrigation_limit_per_source() -> None:
    """Each water source has its own irrigation limit; a withdrawal reduces only its own."""
    n_fields = 1
    remaining_limit_reservoir_m3 = np.array([np.nan], dtype=np.float32)
    remaining_limit_channel_m3 = np.array([100.0], dtype=np.float32)
    remaining_limit_groundwater_m3 = np.array([30.0], dtype=np.float32)
    (
        channel_by_farmer,
        reservoir_by_farmer,
        groundwater_by_farmer,
        withdrawal_m,
        _,
        _,
        _,
        _,
        _,
    ) = abstract_water(
        activation_order=np.array([0], dtype=np.int32),
        field_indices_by_farmer=np.array([[0, 1]], dtype=np.int32),
        field_indices=np.array([0], dtype=np.int32),
        irrigation_efficiency=np.full(1, 1.0, dtype=np.float32),
        channel_irrigated=np.ones(1, dtype=np.bool_),
        well_irrigated=np.ones(1, dtype=np.bool_),
        cell_area=np.full(n_fields, 1000.0, dtype=np.float32),
        HRU_to_grid=np.zeros(n_fields, dtype=np.int32),
        nearest_river_grid_cell=np.zeros(n_fields, dtype=np.int32),
        crop_map=np.zeros(n_fields, dtype=np.int32),
        # plenty of water in the channel and the aquifer: the limits decide
        available_channel_storage_m3=np.full(1, 1e6, dtype=np.float32),
        available_groundwater_m3=np.full(1, 1e6, dtype=np.float64),
        groundwater_depth=np.full(1, 5.0, dtype=np.float64),
        available_reservoir_storage_m3=np.zeros(1, dtype=np.float32),
        command_area_by_farmer=np.full(1, -1, dtype=np.int32),
        return_fraction=np.full(1, 0.0, dtype=np.float32),
        well_depth=np.full(1, 50.0, dtype=np.float32),
        remaining_irrigation_limit_m3_reservoir=remaining_limit_reservoir_m3,
        remaining_irrigation_limit_m3_channel=remaining_limit_channel_m3,
        remaining_irrigation_limit_m3_groundwater=remaining_limit_groundwater_m3,
        maximum_abstraction_reservoir_m3_by_farmer=np.zeros(1, dtype=np.float32),
        gross_irrigation_demand_m3_per_field=np.full(1, 150.0, dtype=np.float32),
        # the demand adjusted to the remaining limit of each source
        demand_limited_channel_m3_per_field=np.full(1, 100.0, dtype=np.float32),
        demand_limited_groundwater_m3_per_field=np.full(1, 30.0, dtype=np.float32),
    )
    np.testing.assert_allclose(channel_by_farmer[0], 100.0, rtol=1e-6)
    np.testing.assert_allclose(groundwater_by_farmer[0], 30.0, rtol=1e-6)
    assert reservoir_by_farmer[0] == 0.0
    np.testing.assert_allclose(withdrawal_m[0] * 1000.0, 130.0, rtol=1e-6)
    np.testing.assert_allclose(remaining_limit_channel_m3[0], 0.0, atol=1e-4)
    np.testing.assert_allclose(remaining_limit_groundwater_m3[0], 0.0, atol=1e-4)
    assert np.isnan(remaining_limit_reservoir_m3[0])


def _gross_irrigation_demand_inputs(n_fields: int = 2) -> dict:
    """Inputs for get_gross_irrigation_demand_m3: one farmer owning dry fields.

    Args:
        n_fields: Number of fields of the farmer.

    Returns:
        Keyword arguments for get_gross_irrigation_demand_m3.
    """
    from geb.hydrology.landsurface.constants import N_SOIL_LAYERS

    layer_height = np.full((n_fields, N_SOIL_LAYERS), 0.1, dtype=np.float32)
    field_capacity = layer_height * np.float32(0.3)
    wilting_point = layer_height * np.float32(0.1)
    return dict(
        day_index=0,
        n=1,
        currently_irrigated_fields=np.ones(n_fields, dtype=np.bool_),
        field_indices_by_farmer=np.array([[0, n_fields]], dtype=np.int32),
        field_indices=np.arange(n_fields, dtype=np.int32),
        irrigation_efficiency=np.full(1, 0.5, dtype=np.float32),
        fraction_irrigated_field=np.ones(1, dtype=np.float32),
        cell_area=np.full(n_fields, 1000.0, dtype=np.float32),
        crop_map=np.zeros(n_fields, dtype=np.int32),
        topwater=np.zeros(n_fields, dtype=np.float32),
        root_depth_m=np.full(n_fields, 0.3, dtype=np.float32),
        soil_layer_height=layer_height,
        field_capacity=field_capacity,
        wilting_point=wilting_point,
        # completely dry: depletion is the full TAW of the root zone (0.06 m)
        w=wilting_point.copy(),
        saturated_hydraulic_conductivity_m_per_day=np.full(
            (n_fields, N_SOIL_LAYERS), 10.0, dtype=np.float32
        ),
        remaining_irrigation_limit_m3_reservoir=np.full(1, np.nan, dtype=np.float32),
        remaining_irrigation_limit_m3_channel=np.full(1, np.nan, dtype=np.float32),
        remaining_irrigation_limit_m3_groundwater=np.full(1, np.nan, dtype=np.float32),
        irrigation_limit_reset_day_index=np.zeros(1, dtype=np.int32),
        cumulative_water_deficit_m3=np.zeros((1, 366), dtype=np.float32),
        crop_calendar=np.array([[[0, 0, 100, 0]]], dtype=np.int32),
        paddy_irrigated_crops=np.zeros(1, dtype=np.bool_),
        current_crop_calendar_rotation_year_index=np.zeros(1, dtype=np.int32),
        max_paddy_water_level=np.full(1, 0.05, dtype=np.float32),
        minimum_effective_root_depth_m=np.float32(0.25),
    )


def test_get_gross_irrigation_demand_m3_limit_per_source() -> None:
    """The remaining limit of each water source caps the demand on that source."""
    from geb.agents.workflows.crop_farmers import get_gross_irrigation_demand_m3

    inputs = _gross_irrigation_demand_inputs()
    inputs["remaining_irrigation_limit_m3_channel"] = np.full(1, 60.0, dtype=np.float32)
    inputs["remaining_irrigation_limit_m3_groundwater"] = np.zeros(1, dtype=np.float32)
    total, reservoir, channel, groundwater = get_gross_irrigation_demand_m3(**inputs)
    # 0.06 m * 1000 m2 / 0.5 efficiency = 120 m3 per field
    np.testing.assert_allclose(total, [120.0, 120.0], rtol=1e-5)
    # no reservoir limit: the whole demand
    np.testing.assert_allclose(reservoir, total, rtol=1e-6)
    # no future deficit, so the whole remaining channel limit can be used today
    np.testing.assert_allclose(channel.sum(), 60.0, rtol=1e-5)
    # the groundwater limit is used up
    assert groundwater.sum() == 0.0


def test_reservoir_command_area_requires_surface_irrigation() -> None:
    """Well-only farmers in a command area cannot take reservoir water."""
    from unittest.mock import MagicMock, PropertyMock, patch

    from geb.agents.crop_farmers import (
        SURFACE_IRRIGATION_EQUIPMENT,
        WELL_ADAPTATION,
        CropFarmers,
    )

    farmers = CropFarmers.__new__(CropFarmers)
    farmers.var = MagicMock()
    adaptations = np.zeros((3, 2), dtype=bool)
    adaptations[0, SURFACE_IRRIGATION_EQUIPMENT] = True  # surface, in command area
    adaptations[1, WELL_ADAPTATION] = True  # well only, in command area
    adaptations[2, SURFACE_IRRIGATION_EQUIPMENT] = True  # surface, no command area
    farmers.var.adaptations = adaptations
    with patch.object(
        CropFarmers, "command_area", new_callable=PropertyMock
    ) as command_area:
        command_area.return_value = np.array([4, 4, -1], dtype=np.int32)
        np.testing.assert_array_equal(farmers.reservoir_command_area, [4, -1, -1])


def test_update_cropland_land_use_type() -> None:
    """Only fields of irrigating farmers get the irrigated land use types."""
    from unittest.mock import MagicMock

    import pandas as pd

    from geb.agents.crop_farmers import (
        SURFACE_IRRIGATION_EQUIPMENT,
        WELL_ADAPTATION,
        CropFarmers,
    )
    from geb.hydrology.landcovers import (
        FOREST,
        GRASSLAND_LIKE,
        NON_PADDY_IRRIGATED,
        PADDY_IRRIGATED,
    )

    farmers = CropFarmers.__new__(CropFarmers)
    farmers.var = MagicMock()
    # crop 0 is rice (paddy), crop 1 is wheat
    farmers.var.crop_data = pd.DataFrame({"is_paddy": [True, False]})
    adaptations = np.zeros((2, 2), dtype=bool)
    adaptations[0, SURFACE_IRRIGATION_EQUIPMENT] = True  # farmer 0 irrigates
    adaptations[1, WELL_ADAPTATION] = False  # farmer 1 is rainfed
    farmers.var.adaptations = adaptations
    farmers.HRU = MagicMock()
    farmers.HRU.var.land_owners = np.array([0, 0, 1, 1, 1, -1], dtype=np.int32)
    farmers.HRU.var.crop_map = np.array([0, 1, 0, 1, -1, -1], dtype=np.int32)
    farmers.HRU.var.land_use_type = np.array(
        [GRASSLAND_LIKE, GRASSLAND_LIKE, PADDY_IRRIGATED, NON_PADDY_IRRIGATED]
        + [GRASSLAND_LIKE, FOREST],
        dtype=np.int32,
    )

    farmers.update_cropland_land_use_type()

    np.testing.assert_array_equal(
        farmers.HRU.var.land_use_type,
        [
            PADDY_IRRIGATED,  # irrigated rice
            NON_PADDY_IRRIGATED,  # irrigated wheat
            GRASSLAND_LIKE,  # rainfed rice does not pond
            GRASSLAND_LIKE,  # rainfed wheat
            GRASSLAND_LIKE,  # fallow
            FOREST,  # not a field
        ],
    )
