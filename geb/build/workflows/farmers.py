"""Workflows for constructing farmer distributions and farm maps."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, prange
from scipy.spatial import cKDTree

from geb.geb_types import ArrayInt32, TwoDArrayBool, TwoDArrayInt32
from geb.workflows.raster import pixels_to_coords, rasterize_like


def create_farm_distributions(
    region_farm_sizes: pd.DataFrame,
    size_class_boundaries: dict[str, tuple[float, float]],
    cultivated_land_area_region_m2: float,
    average_subgrid_area_region: float,
    cultivated_land_region_total_cells: int,
    UID: int,
    ISO3: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Create farm size distributions for a region based on farm size data and cultivated land area.

    Args:
        region_farm_sizes: DataFrame containing farm size data for the region. Must have columns 'Holdings/ agricultural area', 'ISO3', and one column for each size class with the number of holdings or agricultural area for that size class.
        size_class_boundaries: Dictionary mapping size class names to their minimum and maximum size in m2.
        cultivated_land_area_region_m2: Total cultivated land area in the region in m2.
        average_subgrid_area_region: Average area of a subgrid cell in the region in m2.
        cultivated_land_region_total_cells: Total number of cultivated land cells in the region.
        UID: Unique ID of the region.
        ISO3: ISO3 code of the region.
        logger: Logger for logging warnings and errors.

    Returns:
        DataFrame containing the farm size distribution for the region, with columns 'average_farm_size_m2', 'n_holdings', and 'whole_cells' for each size class.

    Raises:
        ValueError: If the input data is inconsistent or if a valid farm size distribution cannot be created based on the input data.
    """
    # Extract holdings and agricultural area data
    # Note that this while the preprocessing is at the region level
    # within the study area, the source data can be for example on
    # country level, so we need to make sure to use the correct data
    # for the region we are processing
    n_holdings_database = (
        region_farm_sizes.loc[
            region_farm_sizes["Holdings/ agricultural area"] == "Holdings"
        ]
        .iloc[0]
        .drop(["Holdings/ agricultural area", "ISO3"])
        .replace("..", np.nan)
        .astype(np.float64)
    )
    agricultural_area_ha_database = (
        region_farm_sizes.loc[
            region_farm_sizes["Holdings/ agricultural area"] == "Agricultural area (Ha)"
        ]
        .iloc[0]
        .drop(["Holdings/ agricultural area", "ISO3"])
        .replace("..", np.nan)
        .astype(np.float64)
    )

    # Calculate average sizes for each bin
    farm_statistics: dict[str, tuple[float, int]] = {}
    for (
        size_class,
        all_holding_area_ha,
    ) in agricultural_area_ha_database.items():
        all_holding_area_m2 = all_holding_area_ha * 10000  # convert from ha to m2
        n_holdings = n_holdings_database[size_class]
        size_class = size_class.strip()

        min_size_m2, max_size_m2 = size_class_boundaries[size_class]

        if np.isnan(all_holding_area_ha) and (np.isnan(n_holdings) or n_holdings == 0):
            continue
        elif (
            np.isnan(all_holding_area_ha)
            and not np.isnan(n_holdings)
            and n_holdings > 0
        ):
            logger.warning(
                f"Total agricultural area for bin '{size_class}' in {ISO3} is missing, but number of holdings is {n_holdings}. "
                "Setting average farm size to the midpoint of the size class."
            )
            if np.isinf(max_size_m2):
                average_farm_size_m2 = (
                    min_size_m2 * 1.5
                )  # if max is infinite, set average to 1.5 times the min size
            else:
                average_farm_size_m2 = (min_size_m2 + max_size_m2) / 2
        else:  # both area and holdings are available, calculate average size as area / holdings
            average_farm_size_m2 = all_holding_area_m2 / n_holdings

            if average_farm_size_m2 < min_size_m2:
                logger.warning(
                    f"Average farm size for bin '{size_class}' in {ISO3} is {average_farm_size_m2:.2f} m², which is below the minimum expected size of {min_size_m2:.2f} m²."
                )
                average_farm_size_m2 = min_size_m2
            elif average_farm_size_m2 > max_size_m2:
                logger.warning(
                    f"Average farm size for bin '{size_class}' in {ISO3} is {average_farm_size_m2:.2f} m², which is above the maximum expected size of {max_size_m2:.2f} m²."
                )
                average_farm_size_m2 = max_size_m2

        assert not np.isnan(average_farm_size_m2)
        assert not np.isnan(n_holdings)
        assert n_holdings >= 0
        assert average_farm_size_m2 >= 0
        farm_statistics[size_class] = (average_farm_size_m2, n_holdings)

    farm_statistics: pd.DataFrame = pd.DataFrame.from_dict(
        farm_statistics,
        orient="index",
        columns=np.array(["average_farm_size_m2", "n_holdings"]),
    )
    total_farm_area_m2_database = (
        farm_statistics["average_farm_size_m2"] * farm_statistics["n_holdings"]
    ).sum()

    # correct number of holdings for the region size, based on the ratio of cultivated land area
    # in the region to the cultivated land area in the database
    farm_statistics["n_holdings"] = farm_statistics["n_holdings"] * (
        cultivated_land_area_region_m2 / total_farm_area_m2_database
    )
    farm_statistics["n_cells"] = (
        farm_statistics["n_holdings"]
        * farm_statistics["average_farm_size_m2"]
        / average_subgrid_area_region
    )

    # checking if all our corrections make sense by comparing the total cultivated land in the region to the total cultivated land implied by the number of holdings and their average size. We allow for a small difference of 1 cell, as there are some rounding errors in the corrections.
    assert math.isclose(
        cultivated_land_region_total_cells,
        farm_statistics["n_cells"].sum(),
        abs_tol=1,
    ), (
        f"{cultivated_land_region_total_cells}, {farm_statistics['n_cells'].sum().item()}"
    )

    farm_statistics["whole_cells"] = (farm_statistics["n_cells"] // 1).astype(int)
    farm_statistics["leftover_cells"] = farm_statistics["n_cells"] % 1
    whole_cells = farm_statistics["whole_cells"].sum()
    n_missing_cells = cultivated_land_region_total_cells - whole_cells

    original_index = farm_statistics.index.copy()
    farm_statistics = farm_statistics.sort_values(
        "leftover_cells", ascending=False
    ).copy()

    farm_statistics.loc[farm_statistics.index[:n_missing_cells], "whole_cells"] += 1

    assert farm_statistics["whole_cells"].sum() == cultivated_land_region_total_cells

    farm_statistics = farm_statistics.reindex(original_index).drop(
        ["leftover_cells", "n_cells"], axis=1
    )
    farm_statistics = farm_statistics[farm_statistics["whole_cells"] > 0]

    farm_statistics["n_holdings"] = farm_statistics["n_holdings"].round().astype(int)
    farm_statistics["n_holdings"] = farm_statistics["n_holdings"].clip(
        lower=1
    )  # at least 1 holding per size class, otherwise we cannot create agents for that size class

    region_farm_sizes: list[ArrayInt32] = []
    for size_class_data in farm_statistics.itertuples():
        size_class = size_class_data.Index
        min_size_m2, max_size_m2 = size_class_boundaries[size_class]

        # for the largest size class, we set the max size to 2 times the average size,
        # to avoid having some extremely large farms.
        if np.isinf(max_size_m2):
            max_size_m2 = size_class_data.average_farm_size_m2 * 2

        min_farm_size_cells: int = int(min_size_m2 / average_subgrid_area_region)
        min_farm_size_cells = max(
            min_farm_size_cells, 1
        )  # farm can never be smaller than one cell

        max_farm_size_cells: int = (
            int(max_size_m2 / average_subgrid_area_region) - 1
        )  # otherwise they overlap with next size class

        if not size_class_data.whole_cells >= size_class_data.n_holdings:
            raise ValueError(
                f"Number of holdings for size class '{size_class}' in {ISO3} is {size_class_data.n_holdings}, "
                f"which is greater than the number of whole cells {size_class_data.whole_cells}. "
                f"Consider adjusting the size class boundaries or the number of subgrid cells to ensure "
                f"that there are enough cells to accommodate the holdings."
            )

        mean_cells_per_agent: int = int(
            size_class_data.whole_cells / size_class_data.n_holdings
        )

        offset = (
            size_class_data.whole_cells
            - size_class_data.n_holdings * mean_cells_per_agent
        )

        if (
            size_class_data.n_holdings * mean_cells_per_agent + offset
            < min_farm_size_cells * size_class_data.n_holdings
        ):
            min_farm_size_cells = (
                size_class_data.n_holdings * mean_cells_per_agent + offset
            ) // size_class_data.n_holdings
        if (
            size_class_data.n_holdings * mean_cells_per_agent + offset
            > max_farm_size_cells * size_class_data.n_holdings
        ):
            max_farm_size_cells = (
                size_class_data.n_holdings * mean_cells_per_agent + offset
            ) // size_class_data.n_holdings + 1

        n_farms_size_class, farm_sizes_size_class = get_farm_distribution(
            size_class_data.n_holdings,
            min_farm_size_cells,
            max_farm_size_cells,
            mean_cells_per_agent,
            offset,
            logger,
        )

        assert n_farms_size_class.sum() == size_class_data.n_holdings
        assert (farm_sizes_size_class >= 1).all()
        assert (
            n_farms_size_class * farm_sizes_size_class
        ).sum() == size_class_data.whole_cells

        # expand farm sizes according to the number of farms in each size class
        farm_sizes = farm_sizes_size_class.repeat(n_farms_size_class)

        # shuffle farm sizes
        np.random.shuffle(farm_sizes)

        region_farm_sizes.append(farm_sizes)

        assert farm_sizes.sum() == size_class_data.whole_cells

    region_farm_sizes: ArrayInt32 = np.concatenate(region_farm_sizes)
    region_agents = pd.DataFrame(
        {
            "farm_size_cells": region_farm_sizes,
            "region_id": np.full_like(region_farm_sizes, UID, dtype=np.int32),
        }
    )
    return region_agents


@njit(cache=True, parallel=False)
def create_farms_numba(
    cultivated_land: TwoDArrayInt32, ids: ArrayInt32, farm_sizes: ArrayInt32
) -> TwoDArrayInt32:
    """Creates random farms considering the farm size distribution.

    Args:
        cultivated_land: map of cultivated land.
        ids: unique IDs of the farmers.
        farm_sizes: map of farm sizes. The size of the first dimension must be equal to the number of layers in `farm_size_probabilities`.

    Returns:
        farms: map of farms. Each unique ID is land owned by a single farmer. Non-cultivated land is represented by -1.
    """
    assert ids.size == farm_sizes.size

    farms: TwoDArrayInt32 = np.where(cultivated_land, -1, -2).astype(np.int32)
    if ids.size > 0:
        current_farm_counter: int = 0
        cur_farm_size: int = 0
        farm_done: bool = False
        farm_id: np.int32 = np.int32(ids[current_farm_counter])
        farm_size: np.int32 = np.int32(farm_sizes[current_farm_counter])
        ysize, xsize = farms.shape
        for y in range(ysize):
            for x in range(xsize):
                f: np.int32 = farms[y, x]
                if f == -1:
                    xmin, xmax, ymin, ymax = 1e6, -1e6, 1e6, -1e6
                    xlow, xhigh, ylow, yhigh = x, x + 1, y, y + 1

                    xsearch, ysearch = 0, 0

                    while True:
                        # Clamp slice bounds to grid to avoid negative-wrap/out-of-bounds
                        ys: int = max(0, ylow)
                        ye: int = min(ysize, yhigh + 1 + ysearch)
                        xs: int = max(0, xlow)
                        xe: int = min(xsize, xhigh + 1 + xsearch)
                        if (
                            ys >= ye
                            or xs >= xe
                            or not np.count_nonzero(farms[ys:ye, xs:xe] == -1)
                        ):
                            break

                        for yf in range(ylow, yhigh + 1):
                            for xf in range(xlow, xhigh + 1):
                                if (
                                    0 <= xf < xsize
                                    and 0 <= yf < ysize
                                    and farms[yf, xf] == -1
                                ):
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
                                        farm_done: bool = True
                                        break

                            if farm_done is True:
                                break

                        if farm_done is True:
                            break

                        if np.random.random() < 0.5:
                            ylow -= 1
                            ysearch: int = 1
                        else:
                            yhigh += 1
                            ysearch: int = 0

                        if np.random.random() < 0.5:
                            xlow -= 1
                            xsearch: int = 1
                        else:
                            xhigh += 1
                            xsearch: int = 0

                    if farm_done:
                        farm_done: bool = False
                        current_farm_counter += 1

                        # If no next farm, do not read next id and farm size (also because they don't exist).
                        # We expect no '-1' cells left at this point (sum of farm sizes equals cultivated area).
                        if current_farm_counter >= ids.size:
                            continue

                        farm_id: np.int32 = np.int32(ids[current_farm_counter])
                        farm_size: np.int32 = np.int32(farm_sizes[current_farm_counter])

    assert np.count_nonzero(farms == -1) == 0
    farms: TwoDArrayInt32 = np.where(farms != -2, farms, -1)
    return farms


def create_farms(
    agents: pd.DataFrame,
    cultivated_land_tehsil: TwoDArrayBool,
    farm_size_key: str = "farm_size_n_cells",
) -> TwoDArrayInt32:
    """Create a farm ownership map based on agent sizes and cultivated land.

    The function assigns unique agent IDs to cultivated land cells such that
    each agent owns a number of cells equal to their target size. Non-cultivated
    cells are set to -1.

    Args:
        agents: DataFrame indexed by agent IDs with a column containing farm sizes (cells).
        cultivated_land_tehsil: 2D binary array where 1 marks cultivated cells and 0 non-cultivated.
        farm_size_key: Column name in ``agents`` with per-agent farm size (cells).

    Returns:
        A 2D array of farm IDs where each cultivated cell is assigned to exactly one agent
        and non-cultivated cells are -1.
    """
    assert cultivated_land_tehsil.sum() == agents[farm_size_key].sum()

    agents: pd.DataFrame = agents.sample(
        frac=1
    )  # shuffle agents to randomize farm placement order
    assert cultivated_land_tehsil.ndim == 2

    farms: TwoDArrayInt32 = create_farms_numba(
        cultivated_land_tehsil,
        ids=agents.index.to_numpy(),
        farm_sizes=agents[farm_size_key].to_numpy(),
    )

    # some tests to ensure correctness
    unique_farms: ArrayInt32 = np.unique(farms)
    unique_farms: ArrayInt32 = unique_farms[unique_farms != -1]
    assert np.array_equal(np.sort(agents.index.to_numpy()), unique_farms)
    assert unique_farms.size == len(agents)
    assert agents[farm_size_key].sum() == np.count_nonzero(farms != -1)
    assert ((farms >= 0) == (cultivated_land_tehsil == 1)).all()

    return farms


def fit_n_farms_to_sizes(
    n: int,
    estimate: ArrayInt32,
    farm_sizes: ArrayInt32,
    mean: int,
    offset: int,
) -> tuple[ArrayInt32, ArrayInt32]:
    """Fit a distribution of farm counts to target total area.

    Converts a fractional estimated distribution of farm counts across sizes
    into integer counts that sum to ``n`` and whose total area equals the
    target area ``n * mean + offset``. The routine preserves the shape implied
    by ``estimate`` as much as possible, then iteratively adjusts counts to
    exactly match the target area.

    Notes:
        - All counts are dimensionless. Farm sizes and target area are in cells.
        - This function assumes ``farm_sizes`` are sorted ascending and are
          consecutive integers (x0..x1). It does not strictly require this, but
          the adjustment logic is designed with that typical setup in mind.
        - We bias rounding by distributing leftover fractional mass forward
          to neighboring sizes to reduce large deviations from the estimated shape.

    Args:
        n: Total number of farms (dimensionless).
        estimate: Fractional estimate of farm counts per size (dimensionless); same
            length as ``farm_sizes``. Only the fractional part is used for
            distributing rounding remainders.
        farm_sizes: Available farm sizes (cells), typically a consecutive range.
        mean: Target mean farm size (cells).
        offset: Additive offset to the total area (cells).

    Returns:
        A tuple ``(n_farms, farm_sizes)`` where:
        - n_farms: Integer number of farms per size (dimensionless), summing to ``n``.
        - farm_sizes: Possibly adjusted array of sizes (cells) if edge adjustments are needed.

    Raises:
        ValueError: If a valid configuration cannot be constructed to meet the target area.
    """
    # Target total area in cells
    target_area: int = int(n * mean + offset)

    # Start from the integer part of the estimate per size
    n_farms: ArrayInt32 = (estimate // 1).astype(np.int32)
    estimated_area_int: int = int((n_farms * farm_sizes).sum())

    # Sanity check: the number still to assign must be less than the number of bins
    missing: int = int(n - n_farms.sum())
    assert missing < n_farms.size

    # Distribute the leftover fractional mass to neighbors to mitigate rounding bias
    extra: ArrayInt32 = np.zeros_like(estimate, dtype=n_farms.dtype)
    leftover_estimate: np.ndarray = estimate % 1
    for i in range(len(leftover_estimate)):
        v: float = float(leftover_estimate[i])
        if v > 0.5:
            # Prefer rounding up here, then compensate by shifting the remainder forward
            extra[i] += 1
            if i < len(leftover_estimate) - 1:
                leftover_estimate[i + 1] -= (1 - v) / farm_sizes[i + 1] * farm_sizes[i]
        else:
            # Prefer rounding down here, move fractional mass forward to next size
            if i < len(leftover_estimate) - 1:
                leftover_estimate[i + 1] += v / farm_sizes[i + 1] * farm_sizes[i]

    n_farms = n_farms + extra
    if n_farms.sum() != n:
        # Small correction to guarantee the total count sums to n
        difference: int = int(n - n_farms.sum())
        n_farms[np.argmax(farm_sizes == mean)] += difference

    assert n_farms.sum() == n

    estimated_area_int = int((n_farms * farm_sizes).sum())

    if estimated_area_int == target_area:
        assert n_farms.sum() == n
        return n_farms, farm_sizes

    elif abs(estimated_area_int - target_area) < farm_sizes.size:
        # Iteratively shift one farm at a time towards meeting the target area.
        while True:
            difference: int = int(target_area - estimated_area_int)
            if difference > 0:
                # Need to increase total area: move a farm from a smaller size to a larger.
                for i in range(len(n_farms)):
                    if n_farms[i] > 0:
                        n_farms[i] -= 1
                        if i == n_farms.size - 1:
                            # If we're already at the largest bin, create a new largest size.
                            farm_sizes = np.append(farm_sizes, farm_sizes[i] + 1)
                            n_farms = np.append(n_farms, 1)
                        else:
                            n_farms[min(i + difference, len(n_farms) - 1)] += 1
                        break
                assert n_farms.sum() == n
            else:
                # Need to decrease total area: move a farm from a larger size to a smaller.
                assert n_farms.sum() == n
                for i in range(len(n_farms) - 1, -1, -1):
                    if n_farms[i] > 0:
                        n_farms[i] -= 1
                        n_farms[max(i + difference, 0)] += 1
                        break
                assert n_farms.sum() == n

            estimated_area_int = int((n_farms * farm_sizes).sum())
            if estimated_area_int == target_area:
                break
            elif n_farms[0] > 0 and (n_farms[1:] == 0).all():
                # If everything concentrates at the smallest bin, introduce a new smaller bin
                # sized to exactly close the remaining gap while keeping counts valid.
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
        raise ValueError(
            f"Could not fit {n} farmers with mean {mean} and offset {offset}."
        )


def get_farm_distribution(
    n: int,
    x0: int,
    x1: int,
    mean: int,
    offset: int,
    logger: logging.Logger | None = None,
) -> tuple[ArrayInt32, ArrayInt32]:
    """Generates a distribution of farm sizes and counts to match target area.

    This function computes the number of farms for each size in a given range
    to achieve a total area equal to n * mean + offset, using an iterative
    estimation process.

    The returned farm sizes range from x0 to x1 (inclusive), and the number of farms
    for each size is determined such that the total number of farms is n and the
    total area matches the target.

    The two arrays returned are of equal length.

    Args:
        n: Number of farms (dimensionless).
        x0: Minimum farm size (cells).
        x1: Maximum farm size (cells).
        mean: Mean farm size (cells).
        offset: Offset to total area (cells).
        logger: Optional logger for warnings; if None, no logging occurs.

    Returns:
        A tuple containing:
        - n_farms: Array of farm counts per size (dimensionless).
        - farm_sizes: Array of farm sizes (cells).
    """
    assert x0 * n <= n * mean + offset <= x1 * n, (
        f"There is no solution for this problem. The total farm size (incl. offset) is larger or smaller than possible with min (x0) and max (x1) farm size. n: {n}, x0: {x0}, x1: {x1}, mean: {mean}, offset: {offset}"
    )  # make sure there is a solution to the problem.

    target_area: int = n * mean + offset

    # when the number of farms is very small, it is sometimes difficult to find a solution. This becomes easier when the number of possible farm sizes is reduced.
    smallest_possible_farm: int = x1 - (n * x1 - target_area)
    x0 = max(x0, smallest_possible_farm)

    largest_possible_farm: int = x0 + (target_area - n * x0)
    x1 = min(x1, largest_possible_farm)
    assert x0 * n <= n * mean + offset <= x1 * n, (
        f"There is no solution for this problem. The total farm size (incl. offset) is larger or smaller than possible with min (x0) and max (x1) farm size. n: {n}, x0: {x0}, x1: {x1}, mean: {mean}, offset: {offset}"
    )  # make sure there is a solution to the problem.

    farm_sizes: ArrayInt32 = np.arange(x0, x1 + 1).astype(np.int32)
    n_farm_sizes: int = farm_sizes.size

    if n == 0:
        n_farms: ArrayInt32 = np.zeros(n_farm_sizes, dtype=np.int32)
        assert target_area == (n_farms * farm_sizes).sum()

    elif n == 1:
        farm_sizes = np.array([mean + offset], dtype=np.int32)
        n_farms = np.array([1], dtype=np.int32)
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
        growth_factor: float = 1

        dist_low = (n * mean + offset) - x0 * n
        dist_high = x1 * n - (n * mean + offset)

        if dist_low < dist_high:
            start_from_bottom: bool = True
        else:
            start_from_bottom: bool = False

        prev_growth_factor: float = -1.0
        prev_estimated_area: float = -1.0

        while True:
            if start_from_bottom:
                estimate = np.full(n_farm_sizes, growth_factor, dtype=np.float64)
                estimate[0] = 1
                estimate = np.cumprod(estimate)
            else:
                estimate = np.full(n_farm_sizes, 1.0 / growth_factor, dtype=np.float64)
                estimate[0] = 1
                estimate = np.cumprod(estimate)[::-1]

            estimate /= estimate.sum() / n

            assert (estimate >= 0).all(), (
                f"Some numbers are negative; growth_factor: {growth_factor}, estimate size: {estimate.size}, estimate: {estimate}"
            )

            estimated_area: float = (estimate * farm_sizes).sum()

            absolute_difference: float = target_area - estimated_area
            if abs(absolute_difference) < 1e-3:
                break

            # Calculate adaptive exponent based on secant method in log-log space
            exponent: float = 1.0 / (n_farm_sizes - 1)

            if prev_growth_factor > 0 and prev_estimated_area > 0:
                log_g_diff = np.log(growth_factor) - np.log(prev_growth_factor)
                log_A_diff = np.log(estimated_area) - np.log(prev_estimated_area)
                exponent = log_g_diff / log_A_diff

            # Update history
            prev_growth_factor = growth_factor
            prev_estimated_area = estimated_area

            difference: float = (target_area / estimated_area) ** exponent

            if difference == 1:
                break

            growth_factor *= difference

        n_farms, farm_sizes = fit_n_farms_to_sizes(
            n, estimate, farm_sizes, mean, offset
        )
        assert n == n_farms.sum()
        estimated_area_int: int = (n_farms * farm_sizes).sum()
        assert estimated_area_int == target_area
        assert (n_farms >= 0).all()
        assert target_area == (n_farms * farm_sizes).sum()

    assert n == n_farms.sum()
    return n_farms, farm_sizes


def get_farm_locations(farms: xr.DataArray, method: str = "centroid") -> TwoDArrayInt32:
    """Get farm locations from farm map.

    Args:
        farms: 2D array of farm IDs. Non-farm land is represented by -1.
        method: Method to determine farm location. Currently only 'centroid' is implemented.

    Returns:
        locations: 2D array of farm locations. First dimension corresponds to farm IDs, second dimension are longitude and latitude.
    """
    if method != "centroid":
        raise NotImplementedError
    gt = farms.rio.transform().to_gdal()

    farms: np.ndarray = farms.values
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


def decode_crop_type_with_secondary_crop(
    combined_crop_type: xr.DataArray | np.ndarray,
    *,
    invalid_crop_values: tuple[int, ...] = (0, 65535),
) -> tuple[xr.DataArray | np.ndarray, xr.DataArray | np.ndarray]:
    """Decode combined crop-secondary codes into main and secondary crop codes.

    Returns:
        One DataArray with the main-crop and one with the secondary-crop information.
    """
    main_crop = (combined_crop_type // 10) * 10
    secondary_crop = combined_crop_type % 10

    invalid_crop = np.isin(combined_crop_type, invalid_crop_values)

    main_crop = np.where(invalid_crop, combined_crop_type, main_crop)
    secondary_crop = np.where(invalid_crop, 0, secondary_crop)

    return main_crop, secondary_crop


@dataclass(frozen=True)
class TargetFarm:
    """Lowder-derived target farm used during synthetic farm construction.

    Each instance represents one synthetic farm that should be created from the
    available field-boundary data. The target area is derived from Lowder-style
    farm-size statistics after scaling the country-level farm-size distribution
    to the cultivated field area in the selected region.

    Attributes:
        target_area_m2: Target farm area in square metres.
        size_class: Original Lowder farm-size class from which the target farm
            was sampled.
    """

    target_area_m2: float
    size_class: str


def crop_sequence_similarity(
    sequence_i: np.ndarray,
    sequence_j: np.ndarray,
    *,
    missing_value: int = -1,
) -> float:
    """Calculate crop-sequence similarity between two fields.

    The similarity is calculated as the fraction of years in which both fields
    have a valid crop value and the crop values are equal. Years where either
    field has ``missing_value`` are ignored.

    This score is used during farm growing to prefer assigning fields with
    similar observed crop sequences to the same synthetic farm.

    Args:
        sequence_i: Crop sequence for the first field.
        sequence_j: Crop sequence for the second field.
        missing_value: Value used to indicate missing or invalid crop
            observations.

    Returns:
        Similarity score between 0 and 1. A value of 1 means both fields have
        identical valid crop values in all comparable years. A value of 0 means
        there are no matching valid crop values, or no comparable years.
    """
    valid = (sequence_i != missing_value) & (sequence_j != missing_value)

    if not np.any(valid):
        return 0.0

    return float(np.mean(sequence_i[valid] == sequence_j[valid]))


def switch_timing_similarity(
    sequence_i: np.ndarray,
    sequence_j: np.ndarray,
    *,
    missing_value: int = -1,
) -> float:
    """Calculate crop-switch timing similarity between two fields.

    The function compares whether two fields switch crops in the same year-to-year
    intervals. A switch is defined as a change in crop code between two
    consecutive valid years. Intervals with missing values in either sequence are
    ignored.

    The score is based on the overlap between switch events in both sequences.
    If neither field switches during any comparable interval, the function
    returns an intermediate similarity of 0.5 rather than treating the fields as
    either fully similar or fully dissimilar.

    Args:
        sequence_i: Crop sequence for the first field.
        sequence_j: Crop sequence for the second field.
        missing_value: Value used to indicate missing or invalid crop
            observations.

    Returns:
        Switch-timing similarity score. A value of 1 means all observed switch
        events occur in the same intervals, 0 means switch events do not overlap,
        and 0.5 means both fields have valid comparable intervals but neither
        field switches.
    """
    valid_i = (sequence_i[:-1] != missing_value) & (sequence_i[1:] != missing_value)
    valid_j = (sequence_j[:-1] != missing_value) & (sequence_j[1:] != missing_value)
    valid = valid_i & valid_j

    if not np.any(valid):
        return 0.0

    switches_i = sequence_i[1:] != sequence_i[:-1]
    switches_j = sequence_j[1:] != sequence_j[:-1]

    switches_i = switches_i[valid]
    switches_j = switches_j[valid]

    union = switches_i | switches_j
    if not np.any(union):
        return 0.5

    intersection = switches_i & switches_j
    return float(np.sum(intersection) / np.sum(union))


def candidate_score(
    farm_crop_sequence: np.ndarray,
    field_crop_sequence: np.ndarray,
    distance_m: float,
    *,
    max_distance_m: float,
    distance_weight: float,
    crop_sequence_weight: float,
    switch_timing_weight: float,
) -> float:
    """Score a candidate field for addition to a growing synthetic farm.

    The score combines spatial proximity, crop-sequence similarity, and
    crop-switch timing similarity. Candidate fields closer to the current farm
    geometry receive a higher distance score. Candidate fields with crop
    sequences and crop-switch timings similar to the current farm-level sequence
    receive higher crop-based scores.

    Args:
        farm_crop_sequence: Current representative crop sequence of the growing
            synthetic farm.
        field_crop_sequence: Crop sequence of the candidate field.
        distance_m: Distance in metres between the current farm geometry and the
            candidate field geometry.
        max_distance_m: Maximum distance considered for candidate selection.
            Distances at or above this value receive a distance score of 0.
        distance_weight: Weight applied to the spatial proximity score.
        crop_sequence_weight: Weight applied to the crop-sequence similarity
            score.
        switch_timing_weight: Weight applied to the crop-switch timing
            similarity score.

    Returns:
        Weighted candidate score. Higher values indicate a better candidate
        field for addition to the growing farm.
    """
    distance_score = max(0.0, 1.0 - distance_m / max_distance_m)

    sequence_score = crop_sequence_similarity(
        farm_crop_sequence,
        field_crop_sequence,
    )
    switch_score = switch_timing_similarity(
        farm_crop_sequence,
        field_crop_sequence,
    )

    return (
        distance_weight * distance_score
        + crop_sequence_weight * sequence_score
        + switch_timing_weight * switch_score
    )


def update_farm_crop_sequence(
    farm_crop_sequences: np.ndarray,
    *,
    missing_value: int = -1,
) -> np.ndarray:
    """Update the representative crop sequence of a growing synthetic farm.

    The representative sequence is selected from the full crop sequences already
    present in the farm. This avoids constructing artificial sequences by taking
    the modal crop independently for each year.

    Ties between equally frequent full sequences are resolved by choosing the
    sequence with the highest average similarity to all field sequences in the
    farm. This makes the selected sequence a simple medoid-like representative
    while still ensuring that it is an observed sequence.

    Args:
        farm_crop_sequences: Two-dimensional array with shape
            ``(n_farm_fields, n_years)`` containing the full crop sequence of
            each field currently assigned to the farm.
        missing_value: Value used to indicate missing crop observations.

    Returns:
        One complete observed crop sequence from the fields in the farm.

    Raises:
        ValueError: If ``farm_crop_sequences`` is not a two-dimensional array.
        ValueError: If ``farm_crop_sequences`` contains no field sequences.
    """
    if farm_crop_sequences.ndim != 2:
        raise ValueError("farm_crop_sequences must be a 2D array.")

    if farm_crop_sequences.shape[0] == 0:
        raise ValueError("Cannot update crop sequence for an empty farm.")

    unique_sequences, counts = np.unique(
        farm_crop_sequences,
        axis=0,
        return_counts=True,
    )

    most_common_count = counts.max()
    most_common_sequences = unique_sequences[counts == most_common_count]

    if len(most_common_sequences) == 1:
        return most_common_sequences[0].astype(np.int32)

    # If multiple full sequences are equally common, choose the one that is most
    # similar to the farm's full set of observed sequences.
    mean_similarities = np.empty(len(most_common_sequences), dtype=np.float64)

    for sequence_index, candidate_sequence in enumerate(most_common_sequences):
        similarities = np.array(
            [
                crop_sequence_similarity(
                    candidate_sequence,
                    farm_sequence,
                    missing_value=missing_value,
                )
                for farm_sequence in farm_crop_sequences
            ],
            dtype=np.float64,
        )
        mean_similarities[sequence_index] = similarities.mean()

    highest_similarity = mean_similarities.max()
    best_sequences = most_common_sequences[mean_similarities == highest_similarity]

    if len(best_sequences) == 1:
        return best_sequences[0].astype(np.int32)

    # If the medoid-like score is also tied, prefer the sequence with fewer
    # missing values.
    missing_counts = np.sum(best_sequences == missing_value, axis=1)
    fewest_missing = missing_counts.min()
    best_sequences = best_sequences[missing_counts == fewest_missing]

    if len(best_sequences) == 1:
        return best_sequences[0].astype(np.int32)

    # Final deterministic tie-breaker: choose the lexicographically smallest sequence.
    sort_order = np.lexsort(best_sequences.T[::-1])
    return best_sequences[sort_order[0]].astype(np.int32)


def _dataarray_to_int32_values(data: xr.DataArray) -> np.ndarray:
    """Convert an xarray DataArray to a contiguous int32 NumPy array.

    Floating rasters are converted to integers after replacing NaN values with
    the HRL outside-area value ``65535``.

    Args:
        data: Input raster data.

    Returns:
        Contiguous NumPy array with dtype ``np.int32``.
    """
    values = data.values

    if np.issubdtype(values.dtype, np.floating):
        values = np.nan_to_num(values, nan=65535)

    return np.ascontiguousarray(values.astype(np.int32, copy=False))


def assert_matching_raster_grid(
    crop_types: xr.DataArray,
    secondary_crop: xr.DataArray,
) -> None:
    """Check whether crop and secondary-crop rasters are exactly aligned.

    Args:
        crop_types: HRL crop-type raster.
        secondary_crop: HRL secondary-crop raster.

    Raises:
        ValueError: If the rasters do not have matching dimensions, shape, CRS,
            or coordinates.
    """
    if crop_types.ndim != 2 or secondary_crop.ndim != 2:
        raise ValueError("Crop and secondary-crop rasters must both be 2D.")

    if crop_types.rio.crs is None or secondary_crop.rio.crs is None:
        raise ValueError("Crop and secondary-crop rasters must both have a CRS.")

    if crop_types.rio.crs != secondary_crop.rio.crs:
        raise ValueError(
            "Crop and secondary-crop rasters must have the same CRS. "
            f"Got {crop_types.rio.crs} and {secondary_crop.rio.crs}."
        )

    if crop_types.shape != secondary_crop.shape:
        raise ValueError(
            "Crop and secondary-crop rasters must have the same shape. "
            f"Got {crop_types.shape} and {secondary_crop.shape}."
        )

    if crop_types.dims != secondary_crop.dims:
        raise ValueError(
            "Crop and secondary-crop rasters must have the same dimensions. "
            f"Got {crop_types.dims} and {secondary_crop.dims}."
        )

    for dim in crop_types.dims:
        if not np.array_equal(crop_types[dim].values, secondary_crop[dim].values):
            raise ValueError(f"Rasters are not aligned on dimension {dim!r}.")


def combine_crop_and_secondary_values(
    crop_values: np.ndarray,
    secondary_values: np.ndarray,
) -> np.ndarray:
    """Combine one year of HRL crop and secondary-crop values.

    The final digit of the returned crop code stores the secondary-crop class.
    Only secondary-crop values 1, 2, 3, and 4 are encoded. All other secondary
    values are treated as no valid secondary crop.

    Args:
        crop_values: Two-dimensional HRL crop-type values.
        secondary_values: Two-dimensional HRL secondary-crop values.

    Returns:
        Two-dimensional encoded crop raster with dtype ``np.int32``.

    Raises:
        ValueError: If both input arrays do not have the same shape.
    """
    if crop_values.shape != secondary_values.shape:
        raise ValueError(
            "crop_values and secondary_values must have the same shape. "
            f"Got {crop_values.shape} and {secondary_values.shape}."
        )

    crop_values = np.ascontiguousarray(crop_values.astype(np.int32, copy=False))
    secondary_values = np.ascontiguousarray(
        secondary_values.astype(np.int32, copy=False)
    )

    combined = crop_values.copy()

    valid_crop = (crop_values != 0) & (crop_values != 65535)
    valid_secondary = (secondary_values >= 1) & (secondary_values <= 4)

    # Only valid main-crop pixels receive the secondary-crop suffix.
    encode_mask = valid_crop & valid_secondary
    combined[encode_mask] = crop_values[encode_mask] + secondary_values[encode_mask]

    return combined


@njit(cache=True, parallel=True)
def _map_field_ids_to_indices_numba(
    field_ids: np.ndarray,
    unique_field_ids: np.ndarray,
    field_nodata: int,
    out: np.ndarray,
) -> None:
    """Map original field IDs to compact zero-based field indices.

    Original field IDs can be large and sparse. Compact indices keep later
    arrays small and contiguous.

    Args:
        field_ids: Two-dimensional raster of original field IDs.
        unique_field_ids: Sorted unique valid field IDs.
        field_nodata: Nodata value in ``field_ids``.
        out: Output array where compact field indices are written.
    """
    field_flat = field_ids.ravel()
    out_flat = out.ravel()

    for index in prange(field_flat.size):
        field_id = field_flat[index]

        if field_id == field_nodata:
            out_flat[index] = -1
            continue

        compact_index = np.searchsorted(unique_field_ids, field_id)

        if (
            compact_index < unique_field_ids.size
            and unique_field_ids[compact_index] == field_id
        ):
            out_flat[index] = compact_index
        else:
            out_flat[index] = -1


def create_field_index_grid(
    field_ids: xr.DataArray,
    *,
    field_nodata: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a compact field-index grid.

    Args:
        field_ids: Rasterized field-boundary IDs.
        field_nodata: Nodata value in ``field_ids``.

    Returns:
        Tuple containing the compact field-index grid and the original field IDs
        corresponding to each compact index.

    Raises:
        ValueError: If no valid field IDs are found.
    """
    field_values = _dataarray_to_int32_values(field_ids)

    unique_field_ids = np.unique(field_values[field_values != field_nodata]).astype(
        np.int32
    )

    if unique_field_ids.size == 0:
        raise ValueError("No valid field IDs found in field-ID raster.")

    field_index_grid = np.full(field_values.shape, -1, dtype=np.int32)

    _map_field_ids_to_indices_numba(
        field_values,
        unique_field_ids,
        field_nodata,
        field_index_grid,
    )

    return field_index_grid, unique_field_ids


@njit(cache=True)
def _pair_counts_chunk_numba(
    crop_chunk: np.ndarray,
    field_index_chunk: np.ndarray,
    pair_base: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count valid field/crop pairs in one raster chunk.

    Each valid pixel is encoded as ``field_index * pair_base + crop_code``.
    Sorting these encoded pairs makes it possible to count occurrences without
    constructing a dense ``field x crop`` matrix.

    Args:
        crop_chunk: Encoded crop-type raster chunk.
        field_index_chunk: Compact field-index raster chunk.
        pair_base: Multiplier used to encode field/crop pairs.

    Returns:
        Tuple containing encoded unique field/crop pair codes and pixel counts.
    """
    crop_flat = crop_chunk.ravel()
    field_flat = field_index_chunk.ravel()

    pair_codes = np.empty(crop_flat.size, dtype=np.int64)
    valid_count = 0

    for index in range(crop_flat.size):
        field_index = field_flat[index]

        if field_index < 0:
            continue

        crop = crop_flat[index]

        if crop == 0 or crop == 65535:
            continue

        if crop < 0 or crop >= pair_base:
            continue

        pair_codes[valid_count] = np.int64(field_index) * pair_base + crop
        valid_count += 1

    if valid_count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)

    sorted_pairs = np.sort(pair_codes[:valid_count])

    n_unique = 1
    for index in range(1, sorted_pairs.size):
        if sorted_pairs[index] != sorted_pairs[index - 1]:
            n_unique += 1

    unique_pairs = np.empty(n_unique, dtype=np.int64)
    counts = np.empty(n_unique, dtype=np.int32)

    unique_index = 0
    current_pair = sorted_pairs[0]
    current_count = 1

    for index in range(1, sorted_pairs.size):
        if sorted_pairs[index] == current_pair:
            current_count += 1
        else:
            unique_pairs[unique_index] = current_pair
            counts[unique_index] = current_count
            unique_index += 1

            current_pair = sorted_pairs[index]
            current_count = 1

    unique_pairs[unique_index] = current_pair
    counts[unique_index] = current_count

    return unique_pairs, counts


def dominant_crop_one_year_chunked(
    crop_types: xr.DataArray,
    secondary_crop: xr.DataArray,
    field_index_grid: np.ndarray,
    n_fields: int,
    *,
    chunk_rows: int = 2048,
    pair_base: int = 65536,
    nodata: int = -1,
) -> np.ndarray:
    """Compute the dominant encoded crop per field for one year.

    The raster is processed in row chunks to avoid holding large intermediate
    arrays in memory. Within each chunk, valid field/crop pixel pairs are counted
    and accumulated in a sparse dictionary.

    Args:
        crop_types: HRL crop-type raster for one year.
        secondary_crop: HRL secondary-crop raster for the same year.
        field_index_grid: Compact field-index grid aligned with the HRL raster.
        n_fields: Number of unique fields.
        chunk_rows: Number of raster rows processed at once.
        pair_base: Multiplier used to encode field/crop pairs. This must be
            larger than the maximum possible encoded crop code.
        nodata: Output value for fields without valid crop pixels.

    Returns:
        One-dimensional array with one dominant encoded crop code per field.

    """
    assert_matching_raster_grid(crop_types, secondary_crop)

    pair_totals: dict[int, int] = {}
    n_rows = crop_types.sizes["y"]

    for y_start in range(0, n_rows, chunk_rows):
        y_stop = min(y_start + chunk_rows, n_rows)

        crop_chunk = _dataarray_to_int32_values(
            crop_types.isel(y=slice(y_start, y_stop))
        )
        secondary_chunk = _dataarray_to_int32_values(
            secondary_crop.isel(y=slice(y_start, y_stop))
        )

        combined_chunk = combine_crop_and_secondary_values(
            crop_chunk,
            secondary_chunk,
        )

        field_index_chunk = np.ascontiguousarray(
            field_index_grid[y_start:y_stop],
            dtype=np.int32,
        )

        unique_pairs, counts = _pair_counts_chunk_numba(
            combined_chunk,
            field_index_chunk,
            pair_base,
        )

        # Accumulate sparse pair counts across row chunks.
        for pair_code, count in zip(unique_pairs, counts, strict=True):
            pair_code_int = int(pair_code)
            pair_totals[pair_code_int] = pair_totals.get(pair_code_int, 0) + int(count)

    dominant_crop = np.full(n_fields, nodata, dtype=np.int32)
    best_count = np.zeros(n_fields, dtype=np.int32)

    # Keep only the crop code with the largest pixel count per field.
    for pair_code, count in pair_totals.items():
        field_index = pair_code // pair_base
        crop_code = pair_code % pair_base

        if count > best_count[field_index]:
            best_count[field_index] = count
            dominant_crop[field_index] = crop_code

    return dominant_crop


def prepare_projected_field_arrays(
    fields: gpd.GeoDataFrame,
    crop_columns: list[str],
) -> tuple[gpd.GeoDataFrame, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare field geometry and arrays for farm growing.

    Field geometries are projected to a local metric CRS. The function then
    extracts compact arrays used by the farm-growing logic.

    Args:
        fields: Field-boundary GeoDataFrame containing geometry and crop columns.
        crop_columns: Names of crop-sequence columns to use during farm growing.

    Returns:
        Tuple containing the projected GeoDataFrame, original CRS, field areas,
        representative-point x coordinates, representative-point y coordinates,
        and field crop sequences.

    Raises:
        ValueError: If ``fields`` has no CRS.
        ValueError: If a projected CRS cannot be estimated.
    """
    if fields.crs is None:
        raise ValueError("Field boundaries must have a CRS.")

    original_crs = fields.crs
    projected_crs = fields.estimate_utm_crs()

    if projected_crs is None:
        raise ValueError("Could not estimate a projected CRS.")

    projected = fields.to_crs(projected_crs).copy()
    projected = projected.reset_index(drop=True)

    projected["field_index"] = np.arange(len(projected), dtype=np.int32)
    projected["field_area_m2"] = projected.geometry.area.astype(np.float64)

    for crop_column in crop_columns:
        projected[crop_column] = projected[crop_column].fillna(-1).astype(np.int32)

    representative_points = projected.geometry.representative_point()

    centroid_x = representative_points.x.to_numpy(dtype=np.float64)
    centroid_y = representative_points.y.to_numpy(dtype=np.float64)
    field_areas = projected["field_area_m2"].to_numpy(dtype=np.float64)
    field_sequences = projected[crop_columns].to_numpy(dtype=np.int32)

    return projected, original_crs, field_areas, centroid_x, centroid_y, field_sequences


def build_field_neighbor_graph(
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    *,
    max_distance_m: float,
    max_neighbors: int = 32,
    query_chunk_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a fixed-size nearest-neighbor graph for fields.

    The neighbor graph is stored in compressed sparse row format. This avoids
    repeated GeoPandas/Shapely distance queries during farm growing.

    Args:
        centroid_x: Field representative-point x coordinates.
        centroid_y: Field representative-point y coordinates.
        max_distance_m: Maximum search distance for neighboring fields.
        max_neighbors: Maximum number of neighbors stored per field.
        query_chunk_size: Number of fields queried at once against the KD-tree.

    Returns:
        Tuple containing the CSR index pointer array, neighbor field indices,
        and neighbor distances in metres.

    Raises:
        ValueError: If coordinate arrays do not have the same length.
        ValueError: If no field coordinates are provided.
    """
    if centroid_x.shape[0] != centroid_y.shape[0]:
        raise ValueError("centroid_x and centroid_y must have the same length.")

    if centroid_x.size == 0:
        raise ValueError("At least one field coordinate is required.")

    coordinates = np.column_stack([centroid_x, centroid_y])
    tree = cKDTree(coordinates)

    n_fields = coordinates.shape[0]
    counts = np.zeros(n_fields, dtype=np.int32)
    query_k = min(max_neighbors + 1, n_fields)

    # First pass: count valid neighbors so exact arrays can be allocated.
    for start in range(0, n_fields, query_chunk_size):
        stop = min(start + query_chunk_size, n_fields)

        distances, indices = tree.query(
            coordinates[start:stop],
            k=query_k,
            distance_upper_bound=max_distance_m,
        )

        if query_k == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        row_ids = np.arange(start, stop, dtype=np.int64)[:, None]
        valid = (indices < n_fields) & (indices != row_ids) & np.isfinite(distances)

        counts[start:stop] = valid.sum(axis=1).astype(np.int32)

    indptr = np.empty(n_fields + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    neighbor_indices = np.empty(int(indptr[-1]), dtype=np.int32)
    neighbor_distances = np.empty(int(indptr[-1]), dtype=np.float32)

    # Second pass: write neighbor indices and distances into CSR arrays.
    for start in range(0, n_fields, query_chunk_size):
        stop = min(start + query_chunk_size, n_fields)

        distances, indices = tree.query(
            coordinates[start:stop],
            k=query_k,
            distance_upper_bound=max_distance_m,
        )

        if query_k == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        for local_row, field_index in enumerate(range(start, stop)):
            valid = (
                (indices[local_row] < n_fields)
                & (indices[local_row] != field_index)
                & np.isfinite(distances[local_row])
            )

            n_valid = int(valid.sum())
            if n_valid == 0:
                continue

            write_start = indptr[field_index]
            write_stop = write_start + n_valid

            neighbor_indices[write_start:write_stop] = indices[local_row][valid].astype(
                np.int32
            )
            neighbor_distances[write_start:write_stop] = distances[local_row][
                valid
            ].astype(np.float32)

    return indptr, neighbor_indices, neighbor_distances


def _select_seed_field(
    target_area_m2: float,
    sorted_field_indices: np.ndarray,
    sorted_field_areas: np.ndarray,
    unassigned: np.ndarray,
) -> int:
    """Select an unassigned seed field close to the target area.

    Args:
        target_area_m2: Target farm area.
        sorted_field_indices: Field indices sorted by area.
        sorted_field_areas: Field areas sorted in ascending order.
        unassigned: Boolean array indicating which fields are still unassigned.

    Returns:
        Index of the selected seed field, or ``-1`` if no unassigned field is
        available.
    """
    insertion_position = np.searchsorted(sorted_field_areas, target_area_m2)

    left = insertion_position - 1
    right = insertion_position

    best_field = -1
    best_difference = np.inf

    # Search outward from the area closest to the target.
    while left >= 0 or right < sorted_field_indices.size:
        found_candidate = False

        if left >= 0:
            left_field = int(sorted_field_indices[left])
            if unassigned[left_field]:
                best_field = left_field
                best_difference = abs(sorted_field_areas[left] - target_area_m2)
                found_candidate = True

        if right < sorted_field_indices.size:
            right_field = int(sorted_field_indices[right])
            if unassigned[right_field]:
                right_difference = abs(sorted_field_areas[right] - target_area_m2)

                if (
                    not found_candidate
                    or right_difference < best_difference
                    or (
                        right_difference == best_difference and right_field < best_field
                    )
                ):
                    best_field = right_field
                    best_difference = right_difference
                    found_candidate = True

        if found_candidate:
            return best_field

        left -= 1
        right += 1

    return -1


def _add_neighbors_to_candidate_frontier(
    field_index: int,
    candidate_distances: dict[int, float],
    unassigned: np.ndarray,
    neighbor_indptr: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
) -> None:
    """Add unassigned neighbors of one field to the candidate frontier.

    Args:
        field_index: Field whose neighbors should be added.
        candidate_distances: Dictionary mapping candidate fields to their
            shortest known distance to the growing farm.
        unassigned: Boolean array indicating which fields are still unassigned.
        neighbor_indptr: CSR index pointer array for the neighbor graph.
        neighbor_indices: Neighbor field indices.
        neighbor_distances: Neighbor distances in metres.
    """
    edge_start = neighbor_indptr[field_index]
    edge_stop = neighbor_indptr[field_index + 1]

    for edge_index in range(edge_start, edge_stop):
        candidate_index = int(neighbor_indices[edge_index])

        if not unassigned[candidate_index]:
            continue

        distance_m = float(neighbor_distances[edge_index])

        # If a candidate is connected through multiple farm fields, keep the
        # shortest distance to the current farm.
        if (
            candidate_index not in candidate_distances
            or distance_m < candidate_distances[candidate_index]
        ):
            candidate_distances[candidate_index] = distance_m


def grow_farms_from_prepared_fields(
    projected_fields: gpd.GeoDataFrame,
    original_crs: Any,
    field_areas_m2: np.ndarray,
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    field_sequences: np.ndarray,
    target_farms: list[TargetFarm],
    *,
    max_distance_m: float = 500.0,
    max_neighbors: int = 32,
    distance_weight: float = 0.45,
    crop_sequence_weight: float = 0.35,
    switch_timing_weight: float = 0.20,
    target_overshoot_tolerance: float = 1.25,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Grow farms from prepared field arrays using a Python/NumPy loop.

    Uses a precomputed centroid-neighbor graph to grow farms to match the
    target farms based on Lowder statistics.

    Args:
        projected_fields: Field GeoDataFrame in projected coordinates.
        original_crs: CRS to convert the output fields back to.
        field_areas_m2: Field areas in square metres.
        centroid_x: Field representative-point x coordinates.
        centroid_y: Field representative-point y coordinates.
        field_sequences: Field crop-sequence array with shape
            ``(n_fields, n_years)``.
        target_farms: Lowder-derived target farms.
        max_distance_m: Maximum neighbor distance in metres.
        max_neighbors: Maximum number of neighbors stored per field.
        distance_weight: Weight for spatial proximity in candidate scoring.
        crop_sequence_weight: Weight for crop-sequence similarity.
        switch_timing_weight: Weight for crop-switch timing similarity.
        target_overshoot_tolerance: Maximum allowed target-area overshoot.

    Returns:
        Tuple containing the field GeoDataFrame with assigned ``farmer_id`` and a
        compact farmer table.

    Raises:
        ValueError: If no fields are available.
        ValueError: If no target farms are available.
        RuntimeError: If one or more fields remain unassigned.
    """
    if projected_fields.empty:
        raise ValueError("No fields available for farm growing.")

    if not target_farms:
        raise ValueError("No target farms available for farm growing.")

    target_farms = sorted(
        target_farms,
        key=lambda target_farm: target_farm.target_area_m2,
        reverse=True,
    )

    neighbor_indptr, neighbor_indices, neighbor_distances = build_field_neighbor_graph(
        centroid_x,
        centroid_y,
        max_distance_m=max_distance_m,
        max_neighbors=max_neighbors,
    )

    n_fields = len(projected_fields)

    sorted_field_indices = np.argsort(field_areas_m2).astype(np.int32)
    sorted_field_areas = field_areas_m2[sorted_field_indices]

    unassigned = np.ones(n_fields, dtype=bool)
    assigned_farmer_ids = np.full(n_fields, -1, dtype=np.int32)

    farmer_areas_m2: list[float] = []
    farmer_n_fields: list[int] = []
    farmer_size_class: list[str] = []
    farmer_target_area_m2: list[float] = []

    for target_farm in target_farms:
        if not unassigned.any():
            break

        farmer_id = len(farmer_areas_m2)

        seed_field = _select_seed_field(
            target_farm.target_area_m2,
            sorted_field_indices,
            sorted_field_areas,
            unassigned,
        )

        if seed_field < 0:
            break

        farm_field_indices = [seed_field]
        current_area_m2 = float(field_areas_m2[seed_field])

        unassigned[seed_field] = False
        assigned_farmer_ids[seed_field] = farmer_id

        farm_crop_sequences = field_sequences[[seed_field]].copy()
        farm_crop_sequence = field_sequences[seed_field].copy()

        candidate_distances: dict[int, float] = {}
        _add_neighbors_to_candidate_frontier(
            seed_field,
            candidate_distances,
            unassigned,
            neighbor_indptr,
            neighbor_indices,
            neighbor_distances,
        )

        while unassigned.any() and current_area_m2 < target_farm.target_area_m2:
            best_candidate = -1
            best_score = -np.inf
            best_distance = np.inf

            # Remove candidates that were assigned while growing another branch.
            for candidate_index in list(candidate_distances):
                if not unassigned[candidate_index]:
                    del candidate_distances[candidate_index]

            for candidate_index, distance_m in candidate_distances.items():
                new_area_m2 = current_area_m2 + float(field_areas_m2[candidate_index])

                if (
                    new_area_m2
                    > target_farm.target_area_m2 * target_overshoot_tolerance
                ):
                    continue

                score = candidate_score(
                    farm_crop_sequence,
                    field_sequences[candidate_index],
                    distance_m,
                    max_distance_m=max_distance_m,
                    distance_weight=distance_weight,
                    crop_sequence_weight=crop_sequence_weight,
                    switch_timing_weight=switch_timing_weight,
                )

                if (
                    score > best_score
                    or (score == best_score and distance_m < best_distance)
                    or (
                        score == best_score
                        and distance_m == best_distance
                        and candidate_index < best_candidate
                    )
                ):
                    best_candidate = candidate_index
                    best_score = score
                    best_distance = distance_m

            if best_candidate < 0:
                break

            farm_field_indices.append(best_candidate)
            current_area_m2 += float(field_areas_m2[best_candidate])

            unassigned[best_candidate] = False
            assigned_farmer_ids[best_candidate] = farmer_id
            candidate_distances.pop(best_candidate, None)

            farm_crop_sequences = np.vstack(
                [
                    farm_crop_sequences,
                    field_sequences[best_candidate],
                ]
            )
            farm_crop_sequence = update_farm_crop_sequence(farm_crop_sequences)

            _add_neighbors_to_candidate_frontier(
                best_candidate,
                candidate_distances,
                unassigned,
                neighbor_indptr,
                neighbor_indices,
                neighbor_distances,
            )

        farmer_areas_m2.append(current_area_m2)
        farmer_n_fields.append(len(farm_field_indices))
        farmer_size_class.append(target_farm.size_class)
        farmer_target_area_m2.append(float(target_farm.target_area_m2))

    # Attach leftover fields to the nearest assigned neighbor, or create a
    # singleton farm if no assigned neighbor exists.
    for field_index in np.flatnonzero(unassigned):
        best_farmer_id = -1
        best_distance = np.inf

        edge_start = neighbor_indptr[field_index]
        edge_stop = neighbor_indptr[field_index + 1]

        for edge_index in range(edge_start, edge_stop):
            neighbor_index = int(neighbor_indices[edge_index])
            neighbor_farmer_id = int(assigned_farmer_ids[neighbor_index])

            if neighbor_farmer_id < 0:
                continue

            distance_m = float(neighbor_distances[edge_index])

            if distance_m < best_distance:
                best_distance = distance_m
                best_farmer_id = neighbor_farmer_id

        if best_farmer_id < 0:
            best_farmer_id = len(farmer_areas_m2)
            farmer_areas_m2.append(0.0)
            farmer_n_fields.append(0)
            farmer_size_class.append("fallback")
            farmer_target_area_m2.append(float(field_areas_m2[field_index]))

        assigned_farmer_ids[field_index] = best_farmer_id
        unassigned[field_index] = False

        farmer_areas_m2[best_farmer_id] += float(field_areas_m2[field_index])
        farmer_n_fields[best_farmer_id] += 1

    projected_fields = projected_fields.copy()
    projected_fields["farmer_id"] = assigned_farmer_ids.astype(np.int32)

    if (projected_fields["farmer_id"] < 0).any():
        raise RuntimeError("Some fields were not assigned to a farmer.")

    farmers = pd.DataFrame(
        {
            "farmer_id": np.arange(len(farmer_areas_m2), dtype=np.int32),
            "target_area_m2": np.asarray(farmer_target_area_m2, dtype=np.float64),
            "area_m2": np.asarray(farmer_areas_m2, dtype=np.float64),
            "area_ha": np.asarray(farmer_areas_m2, dtype=np.float64) / 10_000,
            "size_class": farmer_size_class,
            "n_fields": np.asarray(farmer_n_fields, dtype=np.int32),
        }
    )

    fields_with_farms = projected_fields.to_crs(original_crs)

    return fields_with_farms, farmers


def assign_regions_to_fields(
    fields: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    *,
    region_id_column: str = "region_id",
    country_iso3_column: str = "ISO3",
    logger: logging.Logger | None = None,
) -> gpd.GeoDataFrame:
    """Assign model regions to field polygons using representative points.

    Args:
        fields: Field-boundary GeoDataFrame.
        regions: Region GeoDataFrame containing region and country columns.
        region_id_column: Name of the region ID column in ``regions``.
        country_iso3_column: Name of the country ISO3 column in ``regions``.
        logger: Optional logger used to report dropped fields.

    Returns:
        Field GeoDataFrame with region and ISO3 columns attached.

    Raises:
        ValueError: If required columns are missing.
        ValueError: If the input GeoDataFrames do not have a CRS.
        ValueError: If no fields can be assigned to a region.
    """
    required_region_columns = {region_id_column, country_iso3_column, "geometry"}
    missing_columns = required_region_columns - set(regions.columns)

    if missing_columns:
        raise ValueError(
            f"Region database is missing columns: {sorted(missing_columns)}"
        )

    if fields.crs is None:
        raise ValueError("Field boundaries must have a CRS.")

    if regions.crs is None:
        raise ValueError("Regions must have a CRS.")

    regions_for_join = regions[
        [region_id_column, country_iso3_column, "geometry"]
    ].to_crs(fields.crs)

    # Representative points are guaranteed to lie inside their field polygon.
    field_points = gpd.GeoDataFrame(
        {"__field_row": np.arange(len(fields), dtype=np.int64)},
        geometry=fields.geometry.representative_point(),
        crs=fields.crs,
    )

    joined = gpd.sjoin(
        field_points,
        regions_for_join,
        how="left",
        predicate="within",
    )

    # If boundaries overlap, keep the first match to avoid duplicating fields.
    joined = joined.drop_duplicates("__field_row", keep="first")
    joined = joined.set_index("__field_row").reindex(np.arange(len(fields)))

    fields_with_regions = fields.copy()
    fields_with_regions[region_id_column] = joined[region_id_column].to_numpy()
    fields_with_regions[country_iso3_column] = joined[country_iso3_column].to_numpy()

    missing_region = fields_with_regions[region_id_column].isna()
    if missing_region.any():
        n_missing = int(missing_region.sum())

        if logger is not None:
            logger.warning(
                "Dropping %s fields because they could not be assigned to a region.",
                n_missing,
            )

        fields_with_regions = fields_with_regions.loc[~missing_region].copy()

    if fields_with_regions.empty:
        raise ValueError("No fields could be assigned to model regions.")

    fields_with_regions[region_id_column] = fields_with_regions[
        region_id_column
    ].astype(np.int32)

    return fields_with_regions


def rasterize_and_compact_field_farms(
    fields_with_farms: gpd.GeoDataFrame,
    farmers: pd.DataFrame,
    template: xr.DataArray,
    *,
    farmer_id_column: str = "farmer_id",
    nodata: int = -1,
    all_touched: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[xr.DataArray, pd.DataFrame]:
    """Rasterize field-based farmer IDs and compact IDs to represented farms.

    Some very small fields may disappear when rasterized to a coarser model grid.
    This function therefore keeps only farmer IDs that are actually represented
    in the final raster and remaps them to contiguous IDs.

    Args:
        fields_with_farms: Field GeoDataFrame with a farmer ID column.
        farmers: Farmer table with one row per farmer.
        template: Target model grid, usually ``self.subgrid["region_ids"]``.
        farmer_id_column: Name of the farmer ID column.
        nodata: Nodata value for the farm raster.
        all_touched: Whether to burn all pixels touched by field polygons.
        logger: Optional logger used to report removed farmers.

    Returns:
        Tuple containing the compact farm raster and compact farmer table.

    Raises:
        ValueError: If required farmer ID columns are missing.
        ValueError: If no farmers are represented in the raster.
    """
    if farmer_id_column not in fields_with_farms.columns:
        raise ValueError(f"fields_with_farms must contain '{farmer_id_column}'.")

    if farmer_id_column not in farmers.columns:
        raise ValueError(f"farmers must contain '{farmer_id_column}'.")

    fields_for_raster = fields_with_farms

    if template.rio.crs is not None:
        fields_for_raster = fields_for_raster.to_crs(template.rio.crs)

    farms = rasterize_like(
        fields_for_raster,
        column=farmer_id_column,
        raster=template,
        dtype=np.int32,
        nodata=nodata,
        all_touched=all_touched,
    )

    farm_values = farms.values.astype(np.int32, copy=True)
    present_farmer_ids = np.unique(farm_values[farm_values != nodata]).astype(np.int32)

    if present_farmer_ids.size == 0:
        raise ValueError("No farmers are represented in the rasterized farm map.")

    if present_farmer_ids.size < len(farmers) and logger is not None:
        logger.warning(
            "Only %s of %s field-derived farmers are represented on the model grid. "
            "Missing farmers are dropped from the farmer table.",
            present_farmer_ids.size,
            len(farmers),
        )

    old_to_new = np.full(int(present_farmer_ids.max()) + 1, -1, dtype=np.int32)
    old_to_new[present_farmer_ids] = np.arange(present_farmer_ids.size, dtype=np.int32)

    valid_farm_pixels = farm_values != nodata
    farm_values[valid_farm_pixels] = old_to_new[farm_values[valid_farm_pixels]]

    compact_farms = xr.DataArray(
        farm_values,
        coords=farms.coords,
        dims=farms.dims,
        attrs=farms.attrs,
        name="agents/farmers/farms",
    )
    compact_farms.attrs["_FillValue"] = nodata

    compact_farmers = (
        farmers.set_index(farmer_id_column)
        .loc[present_farmer_ids]
        .reset_index(drop=True)
        .copy()
    )
    compact_farmers[farmer_id_column] = np.arange(
        len(compact_farmers),
        dtype=np.int32,
    )

    return compact_farms, compact_farmers


def _assign_size_class(
    area_m2: pd.Series,
    size_class_boundaries: dict[str, tuple[int | float, int | float]],
) -> pd.Series:
    """Assign farm areas to Lowder size classes.

    Args:
        area_m2: Farm areas in square metres.
        size_class_boundaries: Size-class boundaries in square metres.

    Returns:
        Size-class label for each farm.
    """
    size_classes = pd.Series(index=area_m2.index, dtype="object")

    for size_class, (lower_m2, upper_m2) in size_class_boundaries.items():
        if np.isinf(upper_m2):
            in_class = area_m2 >= lower_m2
        else:
            in_class = (area_m2 >= lower_m2) & (area_m2 < upper_m2)

        size_classes.loc[in_class] = size_class

    return size_classes


def _expected_lowder_farms_by_size_class(
    region_farm_sizes: pd.DataFrame,
    size_class_boundaries: dict[str, tuple[int | float, int | float]],
    cultivated_field_area_m2: float,
) -> pd.Series:
    """Scale Lowder farm counts to the generated cultivated area.

    Args:
        region_farm_sizes: Lowder data for one ISO3 code.
        size_class_boundaries: Size-class boundaries in square metres.
        cultivated_field_area_m2: Generated cultivated field area in the region.

    Returns:
        Expected number of farms per size class, scaled to the region area.

    Raises:
        ValueError: If the Lowder data do not contain one holdings row and one
            agricultural-area row.
        ValueError: If no usable Lowder size classes are available.
    """
    holdings = region_farm_sizes.loc[
        region_farm_sizes["Holdings/ agricultural area"] == "Holdings"
    ]
    agricultural_area = region_farm_sizes.loc[
        region_farm_sizes["Holdings/ agricultural area"] == "Agricultural area (Ha)"
    ]

    if len(holdings) != 1 or len(agricultural_area) != 1:
        raise ValueError("Expected one Holdings row and one Agricultural area row.")

    holdings = holdings.iloc[0].replace("..", np.nan)
    agricultural_area = agricultural_area.iloc[0].replace("..", np.nan)

    records: list[dict[str, float | str]] = []

    for size_class, (lower_m2, upper_m2) in size_class_boundaries.items():
        if (
            size_class not in holdings.index
            or size_class not in agricultural_area.index
        ):
            continue

        n_holdings = pd.to_numeric(holdings[size_class], errors="coerce")
        area_ha = pd.to_numeric(agricultural_area[size_class], errors="coerce")

        if pd.isna(n_holdings) or n_holdings <= 0:
            continue

        if pd.isna(area_ha):
            if np.isinf(upper_m2):
                average_farm_size_m2 = lower_m2 * 1.5
            else:
                average_farm_size_m2 = (lower_m2 + upper_m2) / 2

            area_m2 = n_holdings * average_farm_size_m2
        else:
            area_m2 = area_ha * 10_000

        records.append(
            {
                "size_class": size_class,
                "lowder_n_farms": float(n_holdings),
                "lowder_area_m2": float(area_m2),
            }
        )

    if not records:
        raise ValueError("No usable Lowder size classes are available.")

    lowder = pd.DataFrame(records).set_index("size_class")
    database_area_m2 = lowder["lowder_area_m2"].sum()

    if database_area_m2 <= 0:
        raise ValueError("Lowder agricultural area must be positive.")

    scale_factor = cultivated_field_area_m2 / database_area_m2

    expected_n_farms = lowder["lowder_n_farms"] * scale_factor
    return expected_n_farms.reindex(size_class_boundaries.keys()).fillna(0)


def farm_size_distribution_fit_by_size_class(
    farmers: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    farm_sizes_per_region: pd.DataFrame,
    size_class_boundaries: dict[str, tuple[int | float, int | float]],
    farm_size_donor_country: dict[str, str],
    *,
    region_id_column: str = "region_id",
    country_iso3_column: str = "ISO3",
    area_column: str = "area_m2",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Compare generated and Lowder-expected farm counts by size class.

    The Lowder counts are scaled per region using the generated cultivated field
    area in that region. The final table is aggregated over all regions.

    Args:
        farmers: Final compact farmer table.
        regions: Model region GeoDataFrame.
        farm_sizes_per_region: Lowder farm-size distribution table.
        size_class_boundaries: Size-class boundaries in square metres.
        farm_size_donor_country: Mapping from missing ISO3 codes to donor ISO3
            codes.
        region_id_column: Name of the region ID column.
        country_iso3_column: Name of the ISO3 column.
        area_column: Name of the generated farm-area column in square metres.
        logger: Optional logger.

    Returns:
        DataFrame with expected and actual farm counts per size class.

    Raises:
        ValueError: If required columns are missing.
    """
    required_farmer_columns = {region_id_column, area_column}
    missing_farmer_columns = required_farmer_columns - set(farmers.columns)

    if missing_farmer_columns:
        raise ValueError(
            f"Farmers table is missing columns: {sorted(missing_farmer_columns)}"
        )

    required_region_columns = {region_id_column, country_iso3_column}
    missing_region_columns = required_region_columns - set(regions.columns)

    if missing_region_columns:
        raise ValueError(
            f"Region database is missing columns: {sorted(missing_region_columns)}"
        )

    expected_counts = pd.Series(0.0, index=size_class_boundaries.keys())
    actual_counts = pd.Series(0, index=size_class_boundaries.keys(), dtype=np.int64)

    for _, region in regions.iterrows():
        region_id = int(region[region_id_column])
        original_iso3 = region[country_iso3_column]
        iso3 = farm_size_donor_country.get(original_iso3, original_iso3)

        farmers_region = farmers.loc[farmers[region_id_column] == region_id].copy()
        if farmers_region.empty:
            continue

        region_farm_sizes = farm_sizes_per_region.loc[
            farm_sizes_per_region["ISO3"] == iso3
        ]

        try:
            expected_region = _expected_lowder_farms_by_size_class(
                region_farm_sizes=region_farm_sizes,
                size_class_boundaries=size_class_boundaries,
                cultivated_field_area_m2=float(farmers_region[area_column].sum()),
            )
        except ValueError as error:
            if logger is not None:
                logger.warning(
                    "Could not calculate Lowder expected size-class counts for "
                    "region %s (%s, Lowder source %s): %s",
                    region_id,
                    original_iso3,
                    iso3,
                    error,
                )
            continue

        expected_counts = expected_counts.add(expected_region, fill_value=0)

        actual_size_classes = _assign_size_class(
            farmers_region[area_column],
            size_class_boundaries,
        )

        actual_region = actual_size_classes.value_counts().reindex(
            size_class_boundaries.keys(),
            fill_value=0,
        )

        actual_counts = actual_counts.add(actual_region, fill_value=0).astype(np.int64)

    result = pd.DataFrame(
        {
            "size_class": list(size_class_boundaries.keys()),
            "expected_n_farms_lowder": expected_counts.to_numpy(dtype=np.float64),
            "actual_n_farms": actual_counts.to_numpy(dtype=np.int64),
        }
    )

    result["difference"] = result["actual_n_farms"] - result["expected_n_farms_lowder"]

    result["actual_to_expected_ratio"] = np.where(
        result["expected_n_farms_lowder"] > 0,
        result["actual_n_farms"] / result["expected_n_farms_lowder"],
        np.nan,
    )

    result["expected_share"] = (
        result["expected_n_farms_lowder"] / result["expected_n_farms_lowder"].sum()
    )
    result["actual_share"] = result["actual_n_farms"] / result["actual_n_farms"].sum()

    return result
