"""Tests for the crop farmers agent functions."""

import numpy as np

from geb.agents.crop_farmers import (
    advance_crop_rotation_year,
    cumulative_mean,
    shift_and_update,
)
from geb.agents.workflows.crop_farmers import (
    adjust_irrigation_to_limit,
    get_deficit_between_dates,
    get_future_deficit,
    withdraw_channel,
    withdraw_groundwater,
    withdraw_reservoir,
)


def test_cumulative_mean() -> None:
    """Test the cumulative_mean function."""
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    mean = a.mean()
    assert mean == 4.5

    # test normal operation
    cumulative_mean_, cumulative_count = (
        np.array([0], dtype=np.float32),
        np.array([0], dtype=np.float32),
    )
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
    a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    shift_and_update(a, np.array([9, 10, 11]))
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
        minimum_channel_storage_m3=100.0,
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
        minimum_channel_storage_m3=100.0,
    )
    assert irrigation_water_demand_field == 11.0  # keep 100 m3 in the channel
    assert available_channel_storage_m3[0] == 100.0
    assert water_withdrawal_m[0] == 19.0
    assert channel_abstraction_m3_by_farmer[0] == 1900.0


def test_withdraw_reservoir() -> None:
    """Test the withdraw_reservoir function."""
    available_reservoir_storage_m3 = np.array([2000.0])
    water_withdrawal_m = np.array([0.0])
    remaining_irrigation_limit_m3 = np.array([np.nan])
    reservoir_abstraction_m3_by_farmer = np.array([0.0])
    reservoir_abstraction_m3 = np.zeros_like(available_reservoir_storage_m3)

    irrigation_water_demand_field, irrigation_water_demand_field_m_limit_adjusted = (
        withdraw_reservoir(
            command_area=0,
            reservoir_abstraction_m3=reservoir_abstraction_m3,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
            field=0,
            farmer=0,
            irrigation_water_demand_field_m=10.0,
            irrigation_water_demand_field_m_limit_adjusted=10.0,
            water_withdrawal_m=water_withdrawal_m,
            remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
            reservoir_abstraction_m3_by_farmer=reservoir_abstraction_m3_by_farmer,
            cell_area=np.array([100.0]),
            maximum_abstraction_reservoir_m3_field=np.inf,
        )
    )
    available_reservoir_storage_m3 -= reservoir_abstraction_m3
    assert irrigation_water_demand_field == 0.0
    assert irrigation_water_demand_field_m_limit_adjusted == 0.0
    assert available_reservoir_storage_m3[0] == 1000.0
    assert water_withdrawal_m[0] == 10.0
    assert reservoir_abstraction_m3_by_farmer[0] == 1000.0

    reservoir_abstraction_m3 = np.zeros_like(available_reservoir_storage_m3)
    irrigation_water_demand_field, irrigation_water_demand_field_m_limit_adjusted = (
        withdraw_reservoir(
            command_area=0,
            field=0,
            farmer=0,
            reservoir_abstraction_m3=reservoir_abstraction_m3,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
            irrigation_water_demand_field_m=20.0,
            irrigation_water_demand_field_m_limit_adjusted=20.0,
            water_withdrawal_m=water_withdrawal_m,
            remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
            reservoir_abstraction_m3_by_farmer=reservoir_abstraction_m3_by_farmer,
            cell_area=np.array([100.0]),
            maximum_abstraction_reservoir_m3_field=np.inf,
        )
    )
    available_reservoir_storage_m3 -= reservoir_abstraction_m3
    assert irrigation_water_demand_field == 10.0
    assert irrigation_water_demand_field_m_limit_adjusted == 10.0
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
    irrigation_water_demand_field, irrigation_water_demand_field_m_limit_adjusted = (
        withdraw_reservoir(
            command_area=0,
            field=0,
            farmer=0,
            reservoir_abstraction_m3=reservoir_abstraction_m3,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
            irrigation_water_demand_field_m=20.0,
            irrigation_water_demand_field_m_limit_adjusted=10.0,
            water_withdrawal_m=water_withdrawal_m,
            remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
            reservoir_abstraction_m3_by_farmer=reservoir_abstraction_m3_by_farmer,
            cell_area=np.array([100.0]),
            maximum_abstraction_reservoir_m3_field=np.inf,
        )
    )
    available_reservoir_storage_m3 -= reservoir_abstraction_m3
    assert irrigation_water_demand_field == 10.0  # only 10 should be withdrawn
    # as the limit is 10, and the reservoir abstraction is 20, only
    # 10 can be withdrawn, and the limit is adjusted to 10
    assert (
        irrigation_water_demand_field_m_limit_adjusted == 0.0
    )  # all of the limit is used
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
    irrigation_water_demand_field, irrigation_water_demand_field_m_limit_adjusted = (
        withdraw_reservoir(
            command_area=0,
            field=0,
            farmer=0,
            reservoir_abstraction_m3=reservoir_abstraction_m3,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
            irrigation_water_demand_field_m=20.0,
            irrigation_water_demand_field_m_limit_adjusted=20.0,
            water_withdrawal_m=water_withdrawal_m,
            remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
            reservoir_abstraction_m3_by_farmer=reservoir_abstraction_m3_by_farmer,
            cell_area=np.array([100.0]),
            maximum_abstraction_reservoir_m3_field=1000,
        )
    )
    available_reservoir_storage_m3 -= reservoir_abstraction_m3
    assert irrigation_water_demand_field == 10.0  # only 10 should be withdrawn
    # as the limit is 10, and the reservoir abstraction is 20, only
    # 10 can be withdrawn, and the limit is adjusted to 10
    assert (
        irrigation_water_demand_field_m_limit_adjusted == 10.0
    )  # all of the limit is used
    assert (
        available_reservoir_storage_m3[0] == 1000.0
    )  # thus there is still 1000 m3 left
    assert water_withdrawal_m[0] == 10.0
    assert reservoir_abstraction_m3_by_farmer[0] == 1000.0


def test_advance_crop_rotation_year() -> None:
    """Test the advance_crop_rotation_year function."""
    current_crop_calendar_rotation_year_index = np.array([0, 1, 0, 6, 0])
    crop_calendar_rotation_years = np.array([1, 2, 2, 7, 10])
    advance_crop_rotation_year(
        current_crop_calendar_rotation_year_index=current_crop_calendar_rotation_year_index,
        crop_calendar_rotation_years=crop_calendar_rotation_years,
    )
    assert np.array_equal(
        current_crop_calendar_rotation_year_index, np.array([0, 0, 1, 0, 1])
    )
