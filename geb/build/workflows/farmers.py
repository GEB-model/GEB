"""Workflows for constructing farmer distributions and farm maps."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from numba import njit
from shapely.ops import unary_union

from geb.geb_types import ArrayInt32, TwoDArrayBool, TwoDArrayInt32
from geb.workflows.raster import pixels_to_coords


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


def combine_crop_types_with_secondary_crop(
    crop_types_over_time: xr.DataArray,
    secondary_crop_over_time: xr.DataArray,
    *,
    invalid_crop_values: tuple[int, ...] = (0, 65535),
    valid_secondary_crop_values: tuple[int, ...] = (1, 2, 3, 4),
) -> xr.DataArray:
    """Combine HRL crop types with HRL secondary crop type information.

    The combined code keeps the main crop code as the base value and adds the
    secondary crop type code when a valid secondary crop is present.

    Examples:
        1430 means rapeseed without a valid secondary crop.
        1433 means rapeseed with a short winter secondary crop.

    Args:
        crop_types_over_time: HRL Crop Types data with dimensions including
            ``year``, ``y`` and ``x``.
        secondary_crop_over_time: HRL Secondary Crops Type data with the same
            dimensions, coordinates and grid as ``crop_types_over_time``.
        invalid_crop_values: Crop type values that should not be encoded.
        valid_secondary_crop_values: Secondary crop type values to preserve.

    Returns:
        DataArray with encoded main-crop and secondary-crop information.

    Raises:
        ValueError: If both rasters are not aligned exactly.
    """
    if crop_types_over_time.dims != secondary_crop_over_time.dims:
        raise ValueError(
            "Crop types and secondary crop rasters must have identical dimensions. "
            f"Got {crop_types_over_time.dims} and {secondary_crop_over_time.dims}."
        )

    if crop_types_over_time.shape != secondary_crop_over_time.shape:
        raise ValueError(
            "Crop types and secondary crop rasters must have identical shapes. "
            f"Got {crop_types_over_time.shape} and {secondary_crop_over_time.shape}."
        )

    for dim in crop_types_over_time.dims:
        if not np.array_equal(
            crop_types_over_time[dim].values,
            secondary_crop_over_time[dim].values,
        ):
            raise ValueError(
                f"Crop types and secondary crop rasters are not aligned on "
                f"dimension {dim!r}."
            )

    if crop_types_over_time.rio.crs != secondary_crop_over_time.rio.crs:
        raise ValueError(
            "Crop types and secondary crop rasters must have the same CRS. "
            f"Got {crop_types_over_time.rio.crs} and "
            f"{secondary_crop_over_time.rio.crs}."
        )

    crop_types = crop_types_over_time.astype(np.int32)
    secondary_crop = secondary_crop_over_time.astype(np.int32)

    valid_crop = ~xr.apply_ufunc(
        np.isin,
        crop_types,
        np.array(invalid_crop_values, dtype=np.int32),
        kwargs={"invert": False},
        dask="allowed",
    )

    valid_secondary_crop = xr.apply_ufunc(
        np.isin,
        secondary_crop,
        np.array(valid_secondary_crop_values, dtype=np.int32),
        kwargs={"invert": False},
        dask="allowed",
    )

    secondary_crop_code = xr.where(
        valid_secondary_crop,
        secondary_crop,
        0,
        keep_attrs=True,
    ).astype(np.int32)

    combined_crop_types = xr.where(
        valid_crop,
        crop_types + secondary_crop_code,
        crop_types,
        keep_attrs=True,
    ).astype(np.int32)

    combined_crop_types.name = "crop_types_with_secondary_crop"
    combined_crop_types.attrs.update(crop_types_over_time.attrs)
    combined_crop_types.attrs["description"] = (
        "HRL crop type code plus secondary crop type code. "
        "The final digit indicates secondary crop type: 0 = none/invalid, "
        "1 = short summer, 2 = long summer, 3 = short winter, 4 = long winter."
    )

    return combined_crop_types


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


def count_crops_by_field_year(
    crop_types_over_time: xr.DataArray,
    field_ids: xr.DataArray,
    *,
    invalid_crop_values: Iterable[int] = (0, 65535),
    field_nodata: int = -1,
) -> xr.DataArray:
    """Count valid crop-type pixels for each field and year.

    The crop raster is categorical. Therefore, field boundaries should first be
    rasterized onto the exact crop-raster grid, rather than resampling the crop
    raster. Each valid crop pixel is then grouped by year, field ID, and crop
    type.

    Crop values listed in ``invalid_crop_values`` are ignored before counting.
    This is useful for HRL crop-type rasters where, for example, ``0`` indicates
    no crop and ``65535`` indicates nodata.

    Args:
        crop_types_over_time: Crop-type raster with dimensions
            ``("year", "y", "x")``. Values are categorical crop-type codes.
        field_ids: Field-ID raster with dimensions ``("y", "x")``. This raster
            must be on the same grid as ``crop_types_over_time``.
        invalid_crop_values: Crop raster values that should be ignored. These
            values do not enter the crop counts and cannot become dominant crops.
        field_nodata: Nodata value in ``field_ids``. These pixels are ignored
            because they are not assigned to a field.

    Returns:
        Pixel counts per year, field ID, and valid crop type. The returned array
        has dimensions ``("year", "field_id", "crop_type")`` and contains
        integer pixel counts.

    Raises:
        ValueError: If the crop raster and field-ID raster do not have matching
            spatial dimensions.
        ValueError: If no valid field pixels are found.
        ValueError: If no valid crop pixels are found after excluding
            ``invalid_crop_values``.
    """
    if crop_types_over_time.dims != ("year", "y", "x"):
        crop_types_over_time = crop_types_over_time.transpose("year", "y", "x")

    if field_ids.dims != ("y", "x"):
        field_ids = field_ids.transpose("y", "x")

    crop_y_size = crop_types_over_time.sizes["y"]
    field_y_size = field_ids.sizes["y"]
    if crop_y_size != field_y_size:
        message = (
            "crop_types_over_time and field_ids must have the same y size. "
            f"Got crop_types_over_time y size {crop_y_size} and "
            f"field_ids y size {field_y_size}."
        )
        raise ValueError(message)

    crop_x_size = crop_types_over_time.sizes["x"]
    field_x_size = field_ids.sizes["x"]
    if crop_x_size != field_x_size:
        message = (
            "crop_types_over_time and field_ids must have the same x size. "
            f"Got crop_types_over_time x size {crop_x_size} and "
            f"field_ids x size {field_x_size}."
        )
        raise ValueError(message)

    crops = crop_types_over_time.values
    fields = field_ids.values
    years = crop_types_over_time["year"].values

    invalid_crop_values_array = np.asarray(tuple(invalid_crop_values))

    fields_flat = fields.ravel()
    valid_field_mask = fields_flat != field_nodata

    if not valid_field_mask.any():
        message = (
            "No valid field pixels were found in field_ids after excluding "
            f"field_nodata={field_nodata}."
        )
        raise ValueError(message)

    valid_field_ids = fields_flat[valid_field_mask]
    unique_field_ids, field_inverse = np.unique(
        valid_field_ids,
        return_inverse=True,
    )

    valid_pixels_mask = fields != field_nodata
    valid_crops_all_years = crops[:, valid_pixels_mask]
    valid_crops_mask = ~np.isin(
        valid_crops_all_years,
        invalid_crop_values_array,
    )
    valid_crop_values = valid_crops_all_years[valid_crops_mask]

    if valid_crop_values.size == 0:
        message = (
            "No valid crop pixels were found after excluding invalid crop "
            f"values {tuple(invalid_crop_values_array)}."
        )
        raise ValueError(message)

    unique_crop_types = np.unique(valid_crop_values)

    n_years = years.size
    n_fields = unique_field_ids.size
    n_crops = unique_crop_types.size

    counts = np.zeros((n_years, n_fields, n_crops), dtype=np.int32)

    for year_idx in range(n_years):
        crops_year = crops[year_idx].ravel()[valid_field_mask]
        valid_crop_mask = ~np.isin(crops_year, invalid_crop_values_array)

        crop_index = np.searchsorted(
            unique_crop_types,
            crops_year[valid_crop_mask],
        )
        field_index = field_inverse[valid_crop_mask]

        combined_index = field_index * n_crops + crop_index

        counts[year_idx] = np.bincount(
            combined_index,
            minlength=n_fields * n_crops,
        ).reshape(n_fields, n_crops)

    return xr.DataArray(
        counts,
        dims=("year", "field_id", "crop_type"),
        coords={
            "year": years,
            "field_id": unique_field_ids,
            "crop_type": unique_crop_types,
        },
        name="crop_pixel_count",
    )


def dominant_crop_by_field_year(
    crop_counts: xr.DataArray,
    *,
    nodata: int = -1,
) -> xr.DataArray:
    """Determine the dominant crop per field and year.

    The dominant crop is defined as the crop type with the highest pixel count
    within a field-year combination. Field-year combinations without any valid
    crop pixels receive the provided nodata value.

    This function should usually be applied to the output of
    ``count_crops_by_field_year``. The full crop-count array can still be used
    separately to inspect mixed fields or calculate crop fractions.

    Args:
        crop_counts: Pixel-count array with dimensions
            ``("year", "field_id", "crop_type")``.
        nodata: Value assigned to field-year combinations without valid crop
            pixels.

    Returns:
        Dominant crop code per year and field ID. The returned array has
        dimensions ``("year", "field_id")``.

    Raises:
        ValueError: If ``crop_counts`` does not have the expected dimensions.
    """
    expected_dims = ("year", "field_id", "crop_type")
    if crop_counts.dims != expected_dims:
        message = (
            f"crop_counts must have dimensions {expected_dims}. Got {crop_counts.dims}."
        )
        raise ValueError(message)

    counts = crop_counts.values
    crop_types = crop_counts["crop_type"].values

    total_pixels = counts.sum(axis=2)
    dominant_crop_index = counts.argmax(axis=2)

    dominant_crops = crop_types[dominant_crop_index]
    dominant_crops = np.where(total_pixels > 0, dominant_crops, nodata)

    return xr.DataArray(
        dominant_crops.astype(np.int32),
        dims=("year", "field_id"),
        coords={
            "year": crop_counts["year"].values,
            "field_id": crop_counts["field_id"].values,
        },
        name="dominant_crop",
    )


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


def _largest_remainder_round(values: np.ndarray, target_sum: int) -> np.ndarray:
    """Round fractional values to integers while preserving a target sum.

    The function first floors all values and then distributes the remaining
    units to the values with the largest fractional remainders. If the floored
    values exceed the target sum, units are removed from values with the
    smallest fractional remainders, while avoiding negative counts.

    This is useful when expected farm counts per size class are fractional but
    need to be converted to integer counts while preserving the total number of
    target farms.

    Args:
        values: Array of fractional values to round.
        target_sum: Required sum of the returned integer array.

    Returns:
        Integer array with the same shape as ``values``. The values are rounded
        versions of the input values and sum to ``target_sum`` where possible.
    """
    floored = np.floor(values).astype(np.int64)
    missing = int(target_sum - floored.sum())

    if missing > 0:
        order = np.argsort(values - floored)[::-1]
        floored[order[:missing]] += 1
    elif missing < 0:
        order = np.argsort(values - floored)
        for index in order:
            if missing == 0:
                break
            if floored[index] > 0:
                floored[index] -= 1
                missing += 1

    return floored


def create_lowder_target_farm_areas(
    region_farm_sizes: pd.DataFrame,
    size_class_boundaries: dict[str, tuple[float, float]],
    cultivated_field_area_m2: float,
    iso3: str,
    logger: logging.Logger,
    *,
    random_seed: int = 42,
    minimum_fields_per_farm: float = 1.0,
    mean_field_area_m2: float | None = None,
) -> list[TargetFarm]:
    """Create target farm areas from Lowder-style farm-size statistics.

    This function converts country-level Lowder-style farm-size statistics into
    a list of target farm areas for the selected region. It first estimates the
    representative farm area in each size class, then scales the number of farms
    to match the cultivated field area available in the region.

    The resulting target farms are later used by
    ``grow_farms_from_lowder_targets`` to group individual field polygons into
    synthetic farms. A small deterministic lognormal perturbation is applied
    within each size class so that farms from the same class do not all have
    identical target areas.

    Args:
        region_farm_sizes: Lowder-style farm-size data for one ISO3 code. Must
            contain one row for ``"Holdings"`` and one row for
            ``"Agricultural area (Ha)"``.
        size_class_boundaries: Mapping from Lowder size-class labels to lower
            and upper area boundaries in square metres.
        cultivated_field_area_m2: Total cultivated field area in the selected
            region, calculated from the available field polygons.
        iso3: ISO3 country code used in warning and error messages.
        logger: Logger used to report missing, clipped, or adjusted farm-size
            statistics.
        random_seed: Seed used for deterministic variation in target farm areas.
        minimum_fields_per_farm: Minimum expected number of fields per synthetic
            farm. Used only when ``mean_field_area_m2`` is provided.
        mean_field_area_m2: Mean field area in the selected region. If provided,
            the target number of farms is reduced when the Lowder-scaled farm
            count would imply fewer fields than farms.

    Returns:
        List of ``TargetFarm`` objects. Each object contains a target farm area
        in square metres and the Lowder size class from which it was derived.

    Raises:
        ValueError: If no valid Lowder farm-size classes are available for the
            selected ISO3 code.
        ValueError: If the total Lowder-derived agricultural area is zero or
            negative after processing the valid size classes.
    """
    rng = np.random.default_rng(random_seed)

    holdings = (
        region_farm_sizes.loc[
            region_farm_sizes["Holdings/ agricultural area"] == "Holdings"
        ]
        .iloc[0]
        .drop(["Holdings/ agricultural area", "ISO3"])
        .replace("..", np.nan)
        .astype(np.float64)
    )

    agricultural_area_ha = (
        region_farm_sizes.loc[
            region_farm_sizes["Holdings/ agricultural area"] == "Agricultural area (Ha)"
        ]
        .iloc[0]
        .drop(["Holdings/ agricultural area", "ISO3"])
        .replace("..", np.nan)
        .astype(np.float64)
    )

    bin_records: list[dict[str, Any]] = []

    for raw_size_class, total_area_ha in agricultural_area_ha.items():
        size_class = str(raw_size_class).strip()

        if size_class not in size_class_boundaries:
            continue

        n_holdings = holdings[size_class]
        min_size_m2, max_size_m2 = size_class_boundaries[size_class]

        if np.isnan(total_area_ha) and (np.isnan(n_holdings) or n_holdings == 0):
            continue

        if np.isnan(n_holdings) or n_holdings <= 0:
            continue

        if np.isnan(total_area_ha):
            logger.warning(
                "Total agricultural area for bin '%s' in %s is missing; "
                "using class midpoint as average farm size.",
                size_class,
                iso3,
            )
            if np.isinf(max_size_m2):
                average_farm_size_m2 = min_size_m2 * 1.5
            else:
                average_farm_size_m2 = (min_size_m2 + max_size_m2) / 2
        else:
            average_farm_size_m2 = total_area_ha * 10_000 / n_holdings

        if average_farm_size_m2 < min_size_m2:
            logger.warning(
                "Average farm size for bin '%s' in %s is %.2f m², below the "
                "minimum %.2f m². Clipping to the minimum.",
                size_class,
                iso3,
                average_farm_size_m2,
                min_size_m2,
            )
            average_farm_size_m2 = min_size_m2

        if not np.isinf(max_size_m2) and average_farm_size_m2 > max_size_m2:
            logger.warning(
                "Average farm size for bin '%s' in %s is %.2f m², above the "
                "maximum %.2f m². Clipping to the maximum.",
                size_class,
                iso3,
                average_farm_size_m2,
                max_size_m2,
            )
            average_farm_size_m2 = max_size_m2

        bin_records.append(
            {
                "size_class": size_class,
                "n_holdings_database": float(n_holdings),
                "average_farm_size_m2": float(average_farm_size_m2),
                "database_area_m2": float(n_holdings * average_farm_size_m2),
                "min_size_m2": float(min_size_m2),
                "max_size_m2": float(max_size_m2),
            }
        )

    farm_statistics = pd.DataFrame(bin_records)

    if farm_statistics.empty:
        raise ValueError(f"No valid Lowder farm-size data found for {iso3}.")

    database_total_area_m2 = farm_statistics["database_area_m2"].sum()
    if database_total_area_m2 <= 0:
        raise ValueError(f"Invalid total Lowder farm area for {iso3}.")

    scale_factor = cultivated_field_area_m2 / database_total_area_m2

    farm_statistics["expected_n_farms"] = (
        farm_statistics["n_holdings_database"] * scale_factor
    )

    expected_total_n_farms = int(round(farm_statistics["expected_n_farms"].sum()))
    expected_total_n_farms = max(expected_total_n_farms, 1)

    if mean_field_area_m2 is not None:
        max_reasonable_n_farms = int(
            cultivated_field_area_m2 / (mean_field_area_m2 * minimum_fields_per_farm)
        )
        max_reasonable_n_farms = max(max_reasonable_n_farms, 1)

        if expected_total_n_farms > max_reasonable_n_farms:
            logger.warning(
                "Lowder implies %s farms, but the field-boundary data only support "
                "about %s farms under the current minimum_fields_per_farm setting. "
                "Reducing the target number of farms.",
                expected_total_n_farms,
                max_reasonable_n_farms,
            )
            expected_total_n_farms = max_reasonable_n_farms

    farm_statistics["target_n_farms"] = _largest_remainder_round(
        farm_statistics["expected_n_farms"].to_numpy(dtype=np.float64),
        expected_total_n_farms,
    )

    target_farms: list[TargetFarm] = []

    for row in farm_statistics.itertuples(index=False):
        if row.target_n_farms <= 0:
            continue

        target_bin_area_m2 = row.database_area_m2 * scale_factor
        mean_target_area_m2 = target_bin_area_m2 / row.target_n_farms

        # Add small deterministic variation around the class mean so all farms
        # in the same size class are not identical.
        variation = rng.lognormal(
            mean=0.0,
            sigma=0.15,
            size=int(row.target_n_farms),
        )
        farm_areas = variation / variation.sum() * target_bin_area_m2

        if np.isinf(row.max_size_m2):
            max_size_m2 = max(row.average_farm_size_m2 * 2, mean_target_area_m2)
        else:
            max_size_m2 = row.max_size_m2

        farm_areas = np.clip(
            farm_areas,
            row.min_size_m2,
            max_size_m2,
        )

        # Rescale after clipping to preserve the selected-region area as closely
        # as possible.
        if farm_areas.sum() > 0:
            farm_areas *= target_bin_area_m2 / farm_areas.sum()

        for farm_area_m2 in farm_areas:
            target_farms.append(
                TargetFarm(
                    target_area_m2=float(farm_area_m2),
                    size_class=str(row.size_class),
                )
            )

    rng.shuffle(target_farms)

    return target_farms


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


def prepare_projected_fields(
    fields: gpd.GeoDataFrame,
    crop_columns: list[str],
) -> tuple[gpd.GeoDataFrame, Any]:
    """Prepare field polygons for area and distance calculations.

    Field geometries are projected to an estimated local UTM CRS so that areas
    and distances can be calculated in metres. The function also creates a stable
    integer ``field_index`` column, calculates field area in square metres, and
    converts crop-sequence columns to integer values.

    Missing crop values in the selected crop columns are replaced by ``-1`` so
    that later crop-sequence comparisons can treat them consistently as missing
    values.

    Args:
        fields: Field-boundary GeoDataFrame containing geometry and crop-sequence
            columns.
        crop_columns: Names of the crop-sequence columns that should be prepared
            for farm-growing logic.

    Returns:
        Tuple containing the projected field GeoDataFrame and the original CRS.
        The projected GeoDataFrame contains additional ``field_index`` and
        ``field_area_m2`` columns.

    Raises:
        ValueError: If ``fields`` has no CRS.
        ValueError: If a projected CRS cannot be estimated from the field
            geometries.
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

    return projected, original_crs


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

    Args:
        farm_crop_sequences: Two-dimensional array with shape
            ``(n_farm_fields, n_years)`` containing the full crop sequence of
            each field currently assigned to the farm.
        missing_value: Value used to indicate missing crop observations.

    Returns:
        One complete observed crop sequence from the fields in the farm.
    Raises:
        ValueError: If farm_crop_sequences is not a 2D array.
        ValueError: If crop sequences is empty.
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

    # Deterministic tie-breaker: prefer the sequence with fewer missing values.
    missing_counts = np.sum(most_common_sequences == missing_value, axis=1)
    fewest_missing = missing_counts.min()
    best_sequences = most_common_sequences[missing_counts == fewest_missing]

    if len(best_sequences) == 1:
        return best_sequences[0].astype(np.int32)

    # Final deterministic tie-breaker: choose the lexicographically smallest sequence.
    sort_order = np.lexsort(best_sequences.T[::-1])
    return best_sequences[sort_order[0]].astype(np.int32)


def grow_farms_from_lowder_targets(
    fields: gpd.GeoDataFrame,
    target_farms: list[TargetFarm],
    crop_columns: list[str],
    *,
    random_seed: int = 42,
    max_distance_m: float = 500.0,
    distance_weight: float = 0.45,
    crop_sequence_weight: float = 0.35,
    switch_timing_weight: float = 0.20,
    target_overshoot_tolerance: float = 1.25,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """Grow synthetic farms from field boundaries and Lowder target areas.

    Each synthetic farm starts from one seed field and is expanded by adding
    nearby unassigned fields until it approaches its Lowder-derived target area.
    Candidate fields are selected using a weighted score based on distance,
    crop-sequence similarity, and crop-switch timing similarity.

    Fields are assigned exactly once. If the Lowder target farms are exhausted
    before all fields have been assigned, remaining fields are attached to the
    nearest existing synthetic farm. The function returns both the field-level
    assignment and the resulting farm geometries.

    Args:
        fields: Field-boundary GeoDataFrame with crop-sequence columns.
        target_farms: Target farm areas and size classes derived from
            Lowder-style farm-size statistics.
        crop_columns: Crop-sequence columns used to compare fields during farm
            growing, for example ``["crop_2017", ..., "crop_2023"]``.
        random_seed: Seed used for deterministic seed-field selection and
            tie-breaking noise.
        max_distance_m: Maximum distance in metres for considering candidate
            fields during farm expansion.
        distance_weight: Weight assigned to the spatial proximity component of
            the candidate score.
        crop_sequence_weight: Weight assigned to the crop-sequence similarity
            component of the candidate score.
        switch_timing_weight: Weight assigned to the crop-switch timing
            similarity component of the candidate score.
        target_overshoot_tolerance: Maximum allowed overshoot of the target farm
            area when adding a candidate field. For example, 1.25 allows the
            current farm area to exceed the target area by up to 25%.

    Returns:
        Tuple containing three objects:

        1. Field-level GeoDataFrame with a ``farmer_id`` column identifying the
           synthetic farm assigned to each field.
        2. Farm-level GeoDataFrame with one geometry per synthetic farm and
           summary attributes such as area, size class, and number of fields.
        3. Diagnostics dictionary with summary statistics about the generated
           farms and field assignments.

    Raises:
        ValueError: If no fields are available after preparing the projected
            field dataset.
        RuntimeError: If one or more fields remain unassigned after the farm
            growing and nearest-farm assignment steps.
    """
    rng = np.random.default_rng(random_seed)

    projected_fields, original_crs = prepare_projected_fields(
        fields,
        crop_columns,
    )

    if projected_fields.empty:
        raise ValueError("No fields available for farm growing.")

    target_farms = sorted(
        target_farms,
        key=lambda farm: farm.target_area_m2,
        reverse=True,
    )

    field_sequences = projected_fields[crop_columns].to_numpy(dtype=np.int32)
    field_areas = projected_fields["field_area_m2"].to_numpy(dtype=np.float64)

    unassigned_fields: set[int] = set(range(len(projected_fields)))
    assigned_farmer_ids = np.full(len(projected_fields), -1, dtype=np.int32)

    spatial_index = projected_fields.sindex

    farm_records: list[dict[str, Any]] = []

    for farmer_id, target_farm in enumerate(target_farms):
        if not unassigned_fields:
            break

        unassigned_array = np.array(sorted(unassigned_fields), dtype=np.int32)

        # Prefer a seed field with an area reasonably close to the target.
        area_difference = np.abs(
            field_areas[unassigned_array] - target_farm.target_area_m2
        )
        best_seed_candidates = unassigned_array[
            area_difference == area_difference.min()
        ]
        seed_field = int(rng.choice(best_seed_candidates))

        farm_field_indices = [seed_field]
        unassigned_fields.remove(seed_field)

        current_area_m2 = float(field_areas[seed_field])
        farm_crop_sequence = field_sequences[seed_field].copy()
        farm_geometry = projected_fields.loc[seed_field, "geometry"]

        while unassigned_fields and current_area_m2 < target_farm.target_area_m2:
            search_geometry = farm_geometry.buffer(max_distance_m)
            candidate_indices = list(spatial_index.intersection(search_geometry.bounds))

            valid_candidates: list[tuple[float, float, int]] = []

            for candidate_index in candidate_indices:
                candidate_index = int(candidate_index)

                if candidate_index not in unassigned_fields:
                    continue

                candidate_geometry = projected_fields.loc[candidate_index, "geometry"]
                distance_m = float(farm_geometry.distance(candidate_geometry))

                if distance_m > max_distance_m:
                    continue

                candidate_area_m2 = float(field_areas[candidate_index])
                new_area_m2 = current_area_m2 + candidate_area_m2

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

                # Small deterministic noise prevents unstable tie behaviour.
                score += float(rng.uniform(0, 1e-9))

                valid_candidates.append(
                    (
                        score,
                        -distance_m,
                        candidate_index,
                    )
                )

            if not valid_candidates:
                break

            valid_candidates.sort(reverse=True)
            selected_field = valid_candidates[0][2]

            farm_field_indices.append(selected_field)
            unassigned_fields.remove(selected_field)

            current_area_m2 += float(field_areas[selected_field])
            farm_geometry = unary_union(
                [
                    farm_geometry,
                    projected_fields.loc[selected_field, "geometry"],
                ]
            )

            farm_crop_sequence = update_farm_crop_sequence(
                field_sequences,
                farm_field_indices,
            )

        for field_index in farm_field_indices:
            assigned_farmer_ids[field_index] = farmer_id

        farm_records.append(
            {
                "farmer_id": farmer_id,
                "target_area_m2": float(target_farm.target_area_m2),
                "area_m2": float(current_area_m2),
                "area_ha": float(current_area_m2 / 10_000),
                "size_class": target_farm.size_class,
                "n_fields": int(len(farm_field_indices)),
                "geometry": unary_union(
                    projected_fields.loc[farm_field_indices, "geometry"].to_list()
                ),
            }
        )

    # If Lowder targets ran out before all fields were assigned, attach remaining
    # fields to the nearest existing farm.
    if unassigned_fields and farm_records:
        farm_geometries = {
            record["farmer_id"]: record["geometry"] for record in farm_records
        }

        for field_index in sorted(unassigned_fields):
            field_geometry = projected_fields.loc[field_index, "geometry"]

            nearest_farmer_id = min(
                farm_geometries,
                key=lambda farmer_id: field_geometry.distance(
                    farm_geometries[farmer_id]
                ),
            )

            assigned_farmer_ids[field_index] = nearest_farmer_id

            for record in farm_records:
                if record["farmer_id"] == nearest_farmer_id:
                    record["area_m2"] += float(field_areas[field_index])
                    record["area_ha"] = record["area_m2"] / 10_000
                    record["n_fields"] += 1
                    record["geometry"] = unary_union(
                        [
                            record["geometry"],
                            field_geometry,
                        ]
                    )
                    farm_geometries[nearest_farmer_id] = record["geometry"]
                    break

    projected_fields["farmer_id"] = assigned_farmer_ids

    if (projected_fields["farmer_id"] < 0).any():
        raise RuntimeError("Some fields were not assigned to a farmer.")

    farms = gpd.GeoDataFrame(
        farm_records,
        geometry="geometry",
        crs=projected_fields.crs,
    )

    fields_with_farms = projected_fields.to_crs(original_crs)
    farms = farms.to_crs(original_crs)

    diagnostics = {
        "random_seed": random_seed,
        "n_fields": int(len(fields_with_farms)),
        "n_farms": int(len(farms)),
        "n_target_farms": int(len(target_farms)),
        "total_field_area_ha": float(projected_fields["field_area_m2"].sum() / 10_000),
        "total_farm_area_ha": float(farms["area_m2"].sum() / 10_000),
        "mean_farm_area_ha": float(farms["area_ha"].mean()),
        "median_farm_area_ha": float(farms["area_ha"].median()),
        "mean_fields_per_farm": float(farms["n_fields"].mean()),
        "max_fields_per_farm": int(farms["n_fields"].max()),
    }

    return fields_with_farms, farms, diagnostics
