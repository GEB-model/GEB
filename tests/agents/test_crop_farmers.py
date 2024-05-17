from geb.agents.crop_farmers import cumulative_mean, shift_and_update

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
