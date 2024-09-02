import numpy as np
import pandas as pd
from random import random
from numba import njit
from honeybees.library.raster import pixels_to_coords


@njit(cache=True)
def create_farms_numba(cultivated_land, ids, farm_sizes):
    """
    Creates random farms considering the farm size distribution.

    Args:
        cultivated_land: map of cultivated land.
        gt: geotransformation of cultivated land map.
        farm_size_probabilities: map of the probabilities for the various farm sizes to exist in a specific cell.
        farm_size_choices: Lower and upper bound of the farm size correlating to the farm size probabilities. First dimension must be equal to number of layers of farm_size_probabilities. Size of the second dimension is 2, to represent the lower and upper bound.
        cell_area: map of cell areas for all cells.

    Returns:
        farms: map of farms. Each unique ID is land owned by a single farmer. Non-cultivated land is represented by -1.
        farmer_coords: 2 dimensional numpy array of farmer locations. First dimension corresponds to the IDs of `farms`, and the second dimension are longitude and latitude.
    """

    current_farm_counter = 0
    cur_farm_size = 0
    farm_done = False

    farm_id = ids[current_farm_counter]
    farm_size = farm_sizes[current_farm_counter]
    farms = np.where(cultivated_land, -1, -2).astype(np.int32)
    ysize, xsize = farms.shape
    for y in range(farms.shape[0]):
        for x in range(farms.shape[1]):
            f = farms[y, x]
            if f == -1:
                assert farm_size > 0

                xmin, xmax, ymin, ymax = 1e6, -1e6, 1e6, -1e6
                xlow, xhigh, ylow, yhigh = x, x + 1, y, y + 1

                xsearch, ysearch = 0, 0

                while True:
                    if not np.count_nonzero(
                        farms[ylow : yhigh + 1 + ysearch, xlow : xhigh + 1 + xsearch]
                        == -1
                    ):
                        break

                    for yf in range(ylow, yhigh + 1):
                        for xf in range(xlow, xhigh + 1):
                            if xf < xsize and yf < ysize and farms[yf, xf] == -1:
                                if xf > xmax:
                                    xmax = xf
                                if xf < xmin:
                                    xmin = xf
                                if yf > ymax:
                                    ymax = yf
                                if yf < ymin:
                                    ymin = yf
                                farms[yf, xf] = farm_id
                                cur_farm_size += 1
                                if cur_farm_size == farm_size:
                                    cur_farm_size = 0
                                    farm_done = True
                                    break

                        if farm_done is True:
                            break

                    if farm_done is True:
                        break

                    if random() < 0.5:
                        ylow -= 1
                        ysearch = 1
                    else:
                        yhigh += 1
                        ysearch = 0

                    if random() < 0.5:
                        xlow -= 1
                        xsearch = 1
                    else:
                        xhigh += 1
                        xsearch = 0

                if farm_done:
                    farm_done = False
                    current_farm_counter += 1
                    farm_id = ids[current_farm_counter]
                    farm_size = farm_sizes[current_farm_counter]

    assert np.count_nonzero(farms == -1) == 0
    farms = np.where(farms != -2, farms, -1)
    return farms


def create_farms(
    agents: pd.DataFrame,
    cultivated_land_tehsil: np.ndarray,
    farm_size_key="farm_size_n_cells",
) -> np.ndarray:
    assert cultivated_land_tehsil.sum().compute().item() == agents[farm_size_key].sum()

    agents = agents.sample(frac=1)
    farms = create_farms_numba(
        cultivated_land_tehsil.squeeze().values,  # using first (and should be only) layer
        ids=agents.index.to_numpy(),
        farm_sizes=agents[farm_size_key].to_numpy(),
    ).astype(np.int32)
    unique_farms = np.unique(farms)
    unique_farms = unique_farms[unique_farms != -1]
    assert np.array_equal(np.sort(agents.index.to_numpy()), unique_farms)
    assert unique_farms.size == len(agents)
    assert agents[farm_size_key].sum() == np.count_nonzero(farms != -1)
    assert ((farms >= 0) == (cultivated_land_tehsil == 1)).all()

    return farms


def fit_n_farms_to_sizes(n, estimate, farm_sizes, mean, offset):
    target_area = n * mean + offset
    n_farms = (estimate // 1).astype(int)
    estimated_area_int = (n_farms * farm_sizes).sum()

    missing = n - n_farms.sum()
    assert missing < n_farms.size

    # # because we can only have full farms (not half a farmer) we need to add some missing farms.
    # extra = np.ones_like(estimate, dtype=n_farms.dtype)
    # # because the endpoint is not included we can safely ceil and the index should never be higher than the size of the array
    # extra[np.ceil(np.linspace(0, n_farms.size, n_farms.size - missing, endpoint=False)).astype(int)] -= 1

    extra = np.zeros_like(estimate, dtype=n_farms.dtype)
    leftover_estimate = estimate % 1
    for i in range(len(leftover_estimate)):
        v = leftover_estimate[i]
        if v > 0.5:
            extra[i] += 1
            if i < len(leftover_estimate) - 1:
                leftover_estimate[i + 1] -= (1 - v) / farm_sizes[i + 1] * farm_sizes[i]
        else:
            if i < len(leftover_estimate) - 1:
                leftover_estimate[i + 1] += v / farm_sizes[i + 1] * farm_sizes[i]

    n_farms = n_farms + extra
    if n_farms.sum() != n:
        difference = n - n_farms.sum()
        n_farms[np.argmax(farm_sizes == mean)] += difference

    assert n_farms.sum() == n

    estimated_area_int = (n_farms * farm_sizes).sum()

    if estimated_area_int == target_area:
        assert n_farms.sum() == n
        return n_farms, farm_sizes

    elif abs(estimated_area_int - target_area) < farm_sizes.size:
        while True:
            difference = target_area - estimated_area_int
            if difference > 0:
                for i in range(len(n_farms)):
                    if n_farms[i] > 0:
                        n_farms[i] -= 1
                        if i == n_farms.size - 1:
                            farm_sizes = np.append(farm_sizes, farm_sizes[i] + 1)
                            n_farms = np.append(n_farms, 1)
                        else:
                            n_farms[min(i + difference, len(n_farms) - 1)] += 1
                        break
                assert n_farms.sum() == n
            else:
                assert n_farms.sum() == n
                for i in range(len(n_farms) - 1, -1, -1):
                    if n_farms[i] > 0:
                        n_farms[i] -= 1
                        n_farms[max(i + difference, 0)] += 1
                        break
                assert n_farms.sum() == n
            estimated_area_int = (n_farms * farm_sizes).sum()
            if estimated_area_int == target_area:
                break
            elif n_farms[0] > 0 and (n_farms[1:] == 0).all():
                n_farms[0] -= 1
                n_farms = np.insert(n_farms, 0, 1)
                assert n_farms.sum() == n
                farm_sizes = np.insert(
                    farm_sizes,
                    0,
                    max(farm_sizes[0] + target_area - estimated_area_int, 0),
                )
                break
        assert n_farms.sum() == n
        return n_farms, farm_sizes

    else:
        raise Exception(
            f"Could not fit {n} farmers with mean {mean} and offset {offset}."
        )


def get_farm_distribution(n, x0, x1, mean, offset, logger=None):
    assert (
        x0 * n <= n * mean + offset <= x1 * n
    ), f"There is no solution for this problem. The total farm size (incl. offset) is larger or smaller than possible with min (x0) and max (x1) farm size. n: {n}, x0: {x0}, x1: {x1}, mean: {mean}, offset: {offset}"  # make sure there is a solution to the problem.

    target_area = n * mean + offset

    # when the number of farms is very small, it is sometimes difficult to find a solution. This becomes easier when the number of possible farm sizes is reduced.
    smallest_possible_farm = x1 - (n * x1 - target_area)
    x0 = max(x0, smallest_possible_farm)

    largest_possible_farm = x0 + (target_area - n * x0)
    x1 = min(x1, largest_possible_farm)
    assert (
        x0 * n <= n * mean + offset <= x1 * n
    ), f"There is no solution for this problem. The total farm size (incl. offset) is larger or smaller than possible with min (x0) and max (x1) farm size. n: {n}, x0: {x0}, x1: {x1}, mean: {mean}, offset: {offset}"  # make sure there is a solution to the problem.

    farm_sizes = np.arange(x0, x1 + 1)
    n_farm_sizes = farm_sizes.size

    if n == 0:
        n_farms = np.zeros(n_farm_sizes, dtype=np.int32)
        assert target_area == (n_farms * farm_sizes).sum()

    elif n == 1:
        farm_sizes = np.array([mean + offset])
        n_farms = np.array([1])
        assert target_area == (n_farms * farm_sizes).sum()

    # elif mean == x0:
    #     n_farms = np.zeros(n_farm_sizes, dtype=np.int32)
    #     n_farms[0] = n
    #     if offset > 0:
    #         if offset < n_farms[0]:
    #             n_farms[0] -= offset
    #             n_farms[1] += offset
    #         else:
    #             raise NotImplementedError
    #     elif offset < 0:
    #         n_farms[0] -= 1
    #         n_farms = np.insert(n_farms, 0, 1)
    #         farm_sizes = np.insert(farm_sizes, 0, farm_sizes[0] + offset)
    #         assert (farm_sizes > 0).all()
    #     assert target_area == (n_farms * farm_sizes).sum()

    # elif mean == x1:
    #     n_farms = np.zeros(n_farm_sizes, dtype=np.int32)
    #     n_farms[-1] = n
    #     if offset < 0:
    #         if n_farms[-1] > -offset:
    #             n_farms[-1] += offset
    #             n_farms[-2] -= offset
    #         else:
    #             raise NotImplementedError
    #     elif offset > 0:
    #         n_farms[-1] -= 1
    #         n_farms = np.insert(n_farms, 0, 1)
    #         farm_sizes = np.insert(farm_sizes, 0, farm_sizes[-1] + offset)
    #         assert (farm_sizes > 0).all()
    #     assert target_area == (n_farms * farm_sizes).sum()

    else:
        growth_factor = 1

        start_from_bottom = True
        while True:
            if start_from_bottom:
                estimate = np.zeros(n_farm_sizes, dtype=np.float64)
                estimate[0] = 1
                for i in range(1, estimate.size):
                    estimate[i] = estimate[i - 1] * growth_factor
                estimate /= estimate.sum() / n

            # when there are only some farms at the top of the farm size distribution, the growth factor can become very large and the estimate can become very small.
            # is can lead to NaNs in the estimate. In this case we can start from the top of the farm size distribution.
            if np.isnan(estimate).any() or not start_from_bottom:
                if (
                    start_from_bottom
                ):  # reset growth factor, but only first time this code is run
                    start_from_bottom = False
                    growth_factor = 1
                if logger is not None:
                    logger.warning(
                        f"estimate contains NaNs; growth_factor: {growth_factor}, estimate size: {estimate.size}, estimate: {estimate}, start from the top"
                    )
                estimate = np.zeros(n_farm_sizes, dtype=np.float64)
                estimate[-1] = 1
                for i in range(estimate.size - 2, -1, -1):
                    estimate[i] = estimate[i + 1] * growth_factor
                estimate /= estimate.sum() / n

            assert (
                estimate >= 0
            ).all(), f"Some numbers are negative; growth_factor: {growth_factor}, estimate size: {estimate.size}, estimate: {estimate}"

            estimated_area = (estimate * farm_sizes).sum()

            absolute_difference = target_area - estimated_area
            if abs(absolute_difference) < 1e-3:
                break

            difference = (target_area / estimated_area) ** (1 / (n_farm_sizes - 1))
            if difference == 1:
                break
            growth_factor *= difference

        n_farms, farm_sizes = fit_n_farms_to_sizes(
            n, estimate, farm_sizes, mean, offset
        )
        assert n == n_farms.sum()
        estimated_area_int = (n_farms * farm_sizes).sum()
        assert estimated_area_int == target_area
        assert (n_farms >= 0).all()
        assert target_area == (n_farms * farm_sizes).sum()

    assert n == n_farms.sum()
    return n_farms, farm_sizes


def get_farm_locations(farms, method="centroid"):
    if method != "centroid":
        raise NotImplementedError
    gt = farms.raster.transform.to_gdal()

    farms = farms.values
    n_farmers = np.unique(farms[farms != -1]).size

    vertical_index = (
        np.arange(farms.shape[0])
        .repeat(farms.shape[1])
        .reshape(farms.shape)[farms != -1]
    )
    horizontal_index = np.tile(np.arange(farms.shape[1]), farms.shape[0]).reshape(
        farms.shape
    )[farms != -1]

    pixels = np.zeros((n_farmers, 2), dtype=np.int32)
    pixels[:, 0] = np.round(
        np.bincount(farms[farms != -1], horizontal_index)
        / np.bincount(farms[farms != -1])
    ).astype(int)
    pixels[:, 1] = np.round(
        np.bincount(farms[farms != -1], vertical_index)
        / np.bincount(farms[farms != -1])
    ).astype(int)

    locations = pixels_to_coords(
        pixels + 0.5,
        gt,
    )
    return locations
