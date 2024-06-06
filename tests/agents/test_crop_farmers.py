from geb.agents.crop_farmers import (
    cumulative_mean,
    shift_and_update,
    get_future_deficit,
    adjust_irrigation_to_limit,
    withdraw_groundwater,
    withdraw_channel,
    withdraw_reservoir,
)

import numpy as np


def test_cumulative_mean():
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    mean = a.mean()
    assert mean == 4.5

    # test normal operation
    cumulative_mean_, cumulative_count = np.array([0], dtype=np.float32), np.array(
        [0], dtype=np.float32
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


def test_shift_and_update():
    a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    shift_and_update(a, np.array([9, 10, 11]))
    assert np.array_equal(a, np.array([[9, 0, 1], [10, 3, 4], [11, 6, 7]]))


def test_get_future_deficit():
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
                    [0, 0, 5, 0],
                ]
            ]
        ),  # crop ID (irrelevant here), planting day, growing days
        crop_rotation_year_index=0,
        potential_irrigation_consumption_farmer_m3=10,
    )
    assert future_water_deficit == 50.0

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
        crop_rotation_year_index=0,
        potential_irrigation_consumption_farmer_m3=10,
    )
    assert future_water_deficit == 60.0

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
        crop_rotation_year_index=0,
        potential_irrigation_consumption_farmer_m3=10,
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
        crop_rotation_year_index=0,
        potential_irrigation_consumption_farmer_m3=10,
    )
    assert future_water_deficit == 50.0

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
        crop_rotation_year_index=0,
        potential_irrigation_consumption_farmer_m3=10,
    )
    assert future_water_deficit == 110.0

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
        crop_rotation_year_index=0,
        potential_irrigation_consumption_farmer_m3=0,
    )
    assert future_water_deficit == 0.0

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
        crop_rotation_year_index=0,
        potential_irrigation_consumption_farmer_m3=0,
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
        crop_rotation_year_index=0,
        potential_irrigation_consumption_farmer_m3=0,
    )
    assert future_water_deficit == 1640.0

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
        crop_rotation_year_index=0,
        potential_irrigation_consumption_farmer_m3=10,
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
        crop_rotation_year_index=1,
        potential_irrigation_consumption_farmer_m3=10,
    )
    assert future_water_deficit == 20.0


def test_adjust_irrigation_to_limit():
    cumulative_water_deficit_m3 = np.array([[0.0, 15.0, 30.0]])
    # use full crop calendar with growing days 2 to match cumulative_water_deficit_m3
    crop_calendar = np.array([[[0, 0, 2]]])
    potential_irrigation_consumption_farmer_m3 = adjust_irrigation_to_limit(
        farmer=0,
        day_index=0,
        remaining_irrigation_limit_m3=np.array([10]),
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=crop_calendar,
        irrigation_efficiency_farmer=0.5,
        totalPotIrrConsumption=np.array([10.0]),
        potential_irrigation_consumption_farmer_m3=10,
        farmer_fields=np.array([0]),
    )
    # In the future, still 15 + 10 irrigation is needed
    # irrigation limit remaining is 10, and 10 requested today, so only 10/25*10 = 4. irrigation is possible
    # Then adjusted for the irrigation efficiency is 4*0.5 = 2.0
    assert np.array_equal(potential_irrigation_consumption_farmer_m3, np.array([2.0]))

    # On the last day of the year, the farmer should be able to irrigate the remaining water deficit
    potential_irrigation_consumption_farmer_m3 = adjust_irrigation_to_limit(
        farmer=0,
        day_index=1,  # last day of fictitious year
        remaining_irrigation_limit_m3=np.array([5.0]),
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=crop_calendar,
        irrigation_efficiency_farmer=0.5,
        totalPotIrrConsumption=np.array([10.0]),
        potential_irrigation_consumption_farmer_m3=10,
        farmer_fields=np.array([0]),
    )
    # In the future, still 10 irrigation is needed
    # irrigation limit remaining is 5, and 10 requested today, so only 10/10*5 = 5. irrigation is possible
    # which is all irrigation for the last day of the year
    # then adjusted for the irrigation efficiency is 5*0.5 = 2.5
    assert np.array_equal(potential_irrigation_consumption_farmer_m3, np.array([2.5]))

    # If no allowed irrigation, the farmer should not irrigate
    potential_irrigation_consumption_farmer_m3 = adjust_irrigation_to_limit(
        farmer=0,
        day_index=1,  # last day of fictitious year
        remaining_irrigation_limit_m3=np.array([0]),
        cumulative_water_deficit_m3=cumulative_water_deficit_m3,
        crop_calendar=crop_calendar,
        irrigation_efficiency_farmer=0.5,
        totalPotIrrConsumption=np.array([10.0]),
        potential_irrigation_consumption_farmer_m3=10,
        farmer_fields=np.array([0]),
    )
    # As irrigation limit is 0, no irrigation is possible
    assert np.array_equal(potential_irrigation_consumption_farmer_m3, np.array([0.0]))


def test_withdraw_groundwater():
    available_groundwater_m3 = np.array([2000.0])
    water_withdrawal_m = np.array([0.0])
    remaining_irrigation_limit_m3 = np.array([np.nan])
    groundwater_abstraction_m3_by_farmer = np.array([0.0])

    irrigation_water_demand_field = withdraw_groundwater(
        farmer=0,
        field=0,
        grid_cell=0,
        available_groundwater_m3=available_groundwater_m3,
        cell_area=np.array([100.0]),
        groundwater_depth=np.array([20.0]),
        well_depth=np.array([30.0]),
        irrigation_water_demand_field=10.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        groundwater_abstraction_m3_by_farmer=groundwater_abstraction_m3_by_farmer,
    )
    assert irrigation_water_demand_field == 0.0
    assert available_groundwater_m3[0] == 1000.0
    assert water_withdrawal_m[0] == 10.0
    assert groundwater_abstraction_m3_by_farmer[0] == 1000.0

    irrigation_water_demand_field = withdraw_groundwater(
        farmer=0,
        field=0,
        grid_cell=0,
        available_groundwater_m3=available_groundwater_m3,
        cell_area=np.array([100.0]),
        groundwater_depth=np.array([20.0]),
        well_depth=np.array([30.0]),
        irrigation_water_demand_field=20.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        groundwater_abstraction_m3_by_farmer=groundwater_abstraction_m3_by_farmer,
    )
    assert irrigation_water_demand_field == 10.0
    assert available_groundwater_m3[0] == 0.0
    assert water_withdrawal_m[0] == 20.0
    assert groundwater_abstraction_m3_by_farmer[0] == 2000.0

    # if the well depth is less than the groundwater depth, no water can be withdrawn
    water_withdrawal_m = np.array([0.0])
    available_groundwater_m3 = np.array([2000.0])
    groundwater_abstraction_m3_by_farmer = np.array([0.0])
    irrigation_water_demand_field = withdraw_groundwater(
        farmer=0,
        field=0,
        grid_cell=0,
        available_groundwater_m3=available_groundwater_m3,
        cell_area=np.array([100.0]),
        groundwater_depth=np.array([20.0]),
        well_depth=np.array([10.0]),
        irrigation_water_demand_field=20.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        groundwater_abstraction_m3_by_farmer=groundwater_abstraction_m3_by_farmer,
    )
    assert irrigation_water_demand_field == 20.0
    assert available_groundwater_m3[0] == 2000.0
    assert water_withdrawal_m[0] == 0.0
    assert groundwater_abstraction_m3_by_farmer[0] == 0.0


def test_withdraw_channel():
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
        irrigation_water_demand_field=10.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        channel_abstraction_m3_by_farmer=channel_abstraction_m3_by_farmer,
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
        irrigation_water_demand_field=20.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        channel_abstraction_m3_by_farmer=channel_abstraction_m3_by_farmer,
    )
    assert irrigation_water_demand_field == 10.0
    assert available_channel_storage_m3[0] == 0.0
    assert water_withdrawal_m[0] == 20.0
    assert channel_abstraction_m3_by_farmer[0] == 2000.0


def test_reservoir():
    available_reservoir_storage_m3 = np.array([2000.0])
    water_withdrawal_m = np.array([0.0])
    remaining_irrigation_limit_m3 = np.array([np.nan])
    reservoir_abstraction_m3_by_farmer = np.array([0.0])

    irrigation_water_demand_field = withdraw_reservoir(
        command_area=0,
        available_reservoir_storage_m3=available_reservoir_storage_m3,
        field=0,
        farmer=0,
        irrigation_water_demand_field=10.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        reservoir_abstraction_m3_by_farmer=reservoir_abstraction_m3_by_farmer,
        cell_area=np.array([100.0]),
    )
    assert irrigation_water_demand_field == 0.0
    assert available_reservoir_storage_m3[0] == 1000.0
    assert water_withdrawal_m[0] == 10.0
    assert reservoir_abstraction_m3_by_farmer[0] == 1000.0

    irrigation_water_demand_field = withdraw_reservoir(
        command_area=0,
        field=0,
        farmer=0,
        available_reservoir_storage_m3=available_reservoir_storage_m3,
        irrigation_water_demand_field=20.0,
        water_withdrawal_m=water_withdrawal_m,
        remaining_irrigation_limit_m3=remaining_irrigation_limit_m3,
        reservoir_abstraction_m3_by_farmer=reservoir_abstraction_m3_by_farmer,
        cell_area=np.array([100.0]),
    )
    assert irrigation_water_demand_field == 10.0
    assert available_reservoir_storage_m3[0] == 0.0
    assert water_withdrawal_m[0] == 20.0
    assert reservoir_abstraction_m3_by_farmer[0] == 2000.0
