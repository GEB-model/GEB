from geb.agents.crop_farmers import (
    cumulative_mean,
    shift_and_update,
    adjust_irrigation_to_irrigion_limit,
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


def test_adjust_irrigation_to_irrigion_limit():
    potential_irrigation_consumption_farmer_m3 = adjust_irrigation_to_irrigion_limit(
        farmer=0,
        current_day_of_year=1,
        remaining_irrigation_limit_m3=np.array([10]),
        cumulative_water_deficit_m3=np.array([[0.0, 15.0, 30.0]]),
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
    potential_irrigation_consumption_farmer_m3 = adjust_irrigation_to_irrigion_limit(
        farmer=0,
        current_day_of_year=2,  # last day of fictitious year
        remaining_irrigation_limit_m3=np.array([5.0]),
        cumulative_water_deficit_m3=np.array([[0.0, 50.0, 100.0]]),
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
    potential_irrigation_consumption_farmer_m3 = adjust_irrigation_to_irrigion_limit(
        farmer=0,
        current_day_of_year=2,  # last day of fictitious year
        remaining_irrigation_limit_m3=np.array([0]),
        cumulative_water_deficit_m3=np.array([[0.0, 50.0, 100.0]]),
        irrigation_efficiency_farmer=0.5,
        totalPotIrrConsumption=np.array([10.0]),
        potential_irrigation_consumption_farmer_m3=10,
        farmer_fields=np.array([0]),
    )
    # As irrigation limit is 0, no irrigation is possible
    assert np.array_equal(potential_irrigation_consumption_farmer_m3, np.array([0.0]))
