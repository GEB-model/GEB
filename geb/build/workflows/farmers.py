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
from numba import njit
from pyproj import CRS

from geb.geb_types import ArrayInt32, TwoDArrayBool, TwoDArrayInt32
from geb.workflows.raster import pixels_to_coords

_HRL_FALLOW_CROP_CODE = -1
_HRL_MISSING_CROP_CODE = -2
_HRL_NO_CROPLAND_CODE = 0
_HRL_OUTSIDE_AREA_CODE = 65535

# Quality flags written to the final farmer table by the multi-year sequence
# assignment. Larger values indicate stronger local observational support.
CROP_SEQUENCE_QUALITY_REGIONAL_FALLBACK = 0
CROP_SEQUENCE_QUALITY_LOCAL_ALTERNATIVE = 1
CROP_SEQUENCE_QUALITY_LOCAL_DOMINANT = 2


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
    """Create regional farm sizes from Lowder data and cultivated area.

    Args:
        region_farm_sizes: Regional Lowder rows containing the holding count and
            agricultural area for every available size class.
        size_class_boundaries: Minimum and maximum area in square metres for
            every size class.
        cultivated_land_area_region_m2: Total cultivated land area in the region in m2.
        average_subgrid_area_region: Average area of a subgrid cell in the region in m2.
        cultivated_land_region_total_cells: Number of cultivated model cells in
            the region.
        UID: Unique ID of the region.
        ISO3: ISO3 code of the region.
        logger: Logger for logging warnings and errors.

    Returns:
        Farmer table containing the target number of cells and region ID for
        every generated holding.

    Raises:
        ValueError: If the Lowder inputs cannot produce a valid positive farm-size
            distribution for the available model cells.
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
                f"Total agricultural area for bin {size_class!r} in {ISO3} is "
                f"missing, but the number of holdings is {n_holdings}. "
                "Setting average farm size to the midpoint of the size class."
            )
            if np.isinf(max_size_m2):
                average_farm_size_m2 = (
                    min_size_m2 * 1.5
                )  # if max is infinite, set average to 1.5 times the min size
            else:
                average_farm_size_m2 = (min_size_m2 + max_size_m2) / 2
        else:
            # Both area and holdings are available, so derive their mean size.
            average_farm_size_m2 = all_holding_area_m2 / n_holdings

            if average_farm_size_m2 < min_size_m2:
                logger.warning(
                    f"Average farm size for bin {size_class!r} in {ISO3} is "
                    f"{average_farm_size_m2:.2f} m², below the minimum "
                    f"{min_size_m2:.2f} m²."
                )
                average_farm_size_m2 = min_size_m2
            elif average_farm_size_m2 > max_size_m2:
                logger.warning(
                    f"Average farm size for bin {size_class!r} in {ISO3} is "
                    f"{average_farm_size_m2:.2f} m², above the maximum "
                    f"{max_size_m2:.2f} m²."
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

    # Scale national or donor-country holdings to the region's cultivated area.
    # in the region to the cultivated land area in the database
    farm_statistics["n_holdings"] = farm_statistics["n_holdings"] * (
        cultivated_land_area_region_m2 / total_farm_area_m2_database
    )
    farm_statistics["n_cells"] = (
        farm_statistics["n_holdings"]
        * farm_statistics["average_farm_size_m2"]
        / average_subgrid_area_region
    )

    # Floating-point scaling may differ by at most one model cell before the
    # largest-remainder correction below.
    assert math.isclose(
        cultivated_land_region_total_cells,
        farm_statistics["n_cells"].sum(),
        abs_tol=1,
    ), (
        f"{cultivated_land_region_total_cells}, "
        f"{farm_statistics['n_cells'].sum().item()}"
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
    # Retained size classes need at least one holding to create an agent.
    farm_statistics["n_holdings"] = farm_statistics["n_holdings"].clip(lower=1)

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
                f"Number of holdings for size class {size_class!r} in {ISO3} is "
                f"{size_class_data.n_holdings}, greater than the available "
                f"{size_class_data.whole_cells} whole cells. Consider adjusting "
                "the size-class boundaries or increasing subgrid resolution so "
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
        farm_sizes: Target number of cells per farmer. Its length must equal
            the number of farmer IDs.

    Returns:
        Farm ownership raster. Each non-negative ID represents one farmer and
        non-cultivated cells are ``-1``.
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
                        # Clamp bounds to avoid negative wrapping or overflow.
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

                        # Do not read beyond the final target. Exact target-area
                        # conservation means no unassigned cultivated cells remain.
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
        agents: Farmer table indexed by agent ID and containing target sizes in
            model cells.
        cultivated_land_tehsil: Two-dimensional mask in which 1 marks
            cultivated cells and 0 marks other cells.
        farm_size_key: Column name in ``agents`` with per-agent farm size (cells).

    Returns:
        Two-dimensional farm-ID array in which each cultivated cell belongs to
        exactly one agent and non-cultivated cells are ``-1``.
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
        - ``farm_sizes``: Potentially extended array of available cell counts.

    Raises:
        ValueError: If no integer count distribution can satisfy the target area.
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
                # Increase area by moving one farm to a larger size.
                for i in range(len(n_farms)):
                    if n_farms[i] > 0:
                        n_farms[i] -= 1
                        if i == n_farms.size - 1:
                            # Extend the upper edge when no larger bin exists.
                            farm_sizes = np.append(farm_sizes, farm_sizes[i] + 1)
                            n_farms = np.append(n_farms, 1)
                        else:
                            n_farms[min(i + difference, len(n_farms) - 1)] += 1
                        break
                assert n_farms.sum() == n
            else:
                # Decrease area by moving one farm to a smaller size.
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
                # Extend the lower edge when no smaller bin exists.
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
        "No farm-size solution exists within the requested bounds. "
        f"n={n}, x0={x0}, x1={x1}, mean={mean}, offset={offset}."
    )  # make sure there is a solution to the problem.

    target_area: int = n * mean + offset

    # Small samples can be infeasible across a wide size range. Narrowing the
    # candidate range reduces the number of integer combinations.
    smallest_possible_farm: int = x1 - (n * x1 - target_area)
    x0 = max(x0, smallest_possible_farm)

    largest_possible_farm: int = x0 + (target_area - n * x0)
    x1 = min(x1, largest_possible_farm)
    assert x0 * n <= n * mean + offset <= x1 * n, (
        "No farm-size solution exists within the requested bounds. "
        f"n={n}, x0={x0}, x1={x1}, mean={mean}, offset={offset}."
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
                "Negative farm counts were generated; "
                f"growth_factor={growth_factor}, estimate_size={estimate.size}, "
                f"estimate={estimate}."
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
    """Get representative farm locations from a compact farm-ID raster.

    The centroid is calculated in raster-index space using one flattened active-cell
    index array and three ``numpy.bincount`` operations. This avoids constructing full
    two-dimensional horizontal and vertical index grids, which is substantially more
    memory efficient for large European domains.

    Args:
        farms: Two-dimensional farm-ID raster. Non-farm land is represented by -1
            and farmer IDs must be compact from 0 to ``n_farmers - 1``.
        method: Location method. Currently only ``"centroid"`` is implemented.

    Returns:
        Two-dimensional coordinate array with one ``(x, y)`` location per compact
        farmer ID.

    Raises:
        NotImplementedError: If a method other than ``"centroid"`` is requested.
        ValueError: If the raster contains no farms or farmer IDs are not compact.
    """
    if method != "centroid":
        raise NotImplementedError

    transform = farms.rio.transform().to_gdal()
    farm_values = np.asarray(farms.values)

    if farm_values.ndim != 2:
        raise ValueError(f"farms must be two-dimensional. Got {farm_values.shape}.")

    flat_farms = farm_values.ravel()
    active_flat_indices = np.flatnonzero(flat_farms != -1)
    if active_flat_indices.size == 0:
        raise ValueError("Farm raster contains no represented farmers.")

    active_farmer_ids = flat_farms[active_flat_indices].astype(np.int64, copy=False)
    if (active_farmer_ids < 0).any():
        invalid_ids = np.unique(active_farmer_ids[active_farmer_ids < 0])
        raise ValueError(
            "Farm raster contains unsupported negative farm IDs. "
            f"Found {invalid_ids[:10].tolist()}."
        )

    n_farmers = int(active_farmer_ids.max()) + 1
    counts = np.bincount(active_farmer_ids, minlength=n_farmers)
    if (counts == 0).any():
        missing_farmer_ids = np.flatnonzero(counts == 0)
        raise ValueError(
            "Farm IDs must be compact from 0 to n_farmers - 1. Missing examples: "
            f"{missing_farmer_ids[:10].tolist()}."
        )

    n_columns = farm_values.shape[1]
    row_indices = active_flat_indices // n_columns
    column_indices = active_flat_indices % n_columns

    mean_columns = (
        np.bincount(
            active_farmer_ids,
            weights=column_indices,
            minlength=n_farmers,
        )
        / counts
    )
    mean_rows = (
        np.bincount(
            active_farmer_ids,
            weights=row_indices,
            minlength=n_farmers,
        )
        / counts
    )

    pixels = np.column_stack(
        (
            np.rint(mean_columns).astype(np.int32),
            np.rint(mean_rows).astype(np.int32),
        )
    )

    return pixels_to_coords(
        pixels + 0.5,
        transform,
    )


@dataclass(frozen=True)
class TargetFarm:
    """Lowder-derived target farm used during synthetic farm construction.

    Each instance represents one synthetic farm that should be created from the
    available raster cells. The target area is derived from Lowder-style
    farm-size statistics after scaling the country-level farm-size distribution
    to the cultivated raster area in the selected region.

    Attributes:
        target_area_m2: Target farm area in square metres.
        size_class: Original Lowder farm-size class from which the target farm
            was sampled.
    """

    target_area_m2: float
    size_class: str


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

    valid_crop = (crop_values > _HRL_NO_CROPLAND_CODE) & (
        crop_values != _HRL_OUTSIDE_AREA_CODE
    )
    valid_secondary = (secondary_values >= 1) & (secondary_values <= 4)

    # Only valid main-crop pixels receive the secondary-crop suffix.
    encode_mask = valid_crop & valid_secondary
    combined[encode_mask] = crop_values[encode_mask] + secondary_values[encode_mask]

    return combined


@njit(cache=True)
def _crop_sequence_similarity_numba(
    sequence_i: np.ndarray,
    sequence_j: np.ndarray,
    missing_value: int,
    fallow_value: int,
    min_valid_overlap: int,
    fallow_match_weight: float = 0.35,
) -> float:
    """Calculate crop-sequence similarity with distinct fallow and missing states.

    Native outside/missing years are excluded from comparison. Fallow remains a
    meaningful observed state, but a fallow-fallow match contributes less than an
    active-crop match so fallow-heavy sequences do not dominate farm grouping.
    The final score is coverage-penalized to avoid high similarity based on only a
    few comparable years.

    Args:
        sequence_i: Crop sequence for the first raster cell.
        sequence_j: Crop sequence for the candidate raster cell.
        missing_value: Native outside-area or missing-observation code.
        fallow_value: Genuine fallow/no-cropland code inside the field domain.
        min_valid_overlap: Minimum comparable years required for a positive score.
        fallow_match_weight: Contribution of a matching fallow year relative to a
            matching active-crop year.

    Returns:
        Coverage-penalized weighted match fraction in ``0..1``.
    """
    comparable_count = 0
    weighted_matches = 0.0

    for year_index in range(sequence_i.size):
        crop_i = sequence_i[year_index]
        crop_j = sequence_j[year_index]

        if crop_i == missing_value or crop_j == missing_value:
            continue

        comparable_count += 1
        if crop_i == crop_j:
            weighted_matches += fallow_match_weight if crop_i == fallow_value else 1.0

    if comparable_count < min_valid_overlap:
        return 0.0

    raw_similarity = weighted_matches / comparable_count
    coverage = comparable_count / sequence_i.size
    return raw_similarity * coverage


@njit(cache=True)
def _switch_timing_similarity_numba(
    sequence_i: np.ndarray,
    sequence_j: np.ndarray,
    missing_value: int,
) -> float:
    """Calculate crop-switch timing similarity in Numba-compatible form.

    Returns:
        Jaccard-style overlap between year-to-year crop-switch events. Returns
        0.0 if there are no comparable valid intervals, and 0.5 if both
        sequences have comparable intervals but neither sequence switches crop.
    """
    union_count = 0
    intersection_count = 0
    valid_interval_count = 0

    for year_index in range(sequence_i.size - 1):
        crop_i_previous = sequence_i[year_index]
        crop_i_next = sequence_i[year_index + 1]
        crop_j_previous = sequence_j[year_index]
        crop_j_next = sequence_j[year_index + 1]

        if (
            crop_i_previous == missing_value
            or crop_i_next == missing_value
            or crop_j_previous == missing_value
            or crop_j_next == missing_value
        ):
            continue

        valid_interval_count += 1
        switch_i = crop_i_next != crop_i_previous
        switch_j = crop_j_next != crop_j_previous

        if switch_i or switch_j:
            union_count += 1
            if switch_i and switch_j:
                intersection_count += 1

    if valid_interval_count == 0:
        return 0.0

    if union_count == 0:
        return 0.5

    return intersection_count / union_count


def raster_cell_area_m2(template: xr.DataArray) -> np.ndarray:
    """Calculate the area of every raster cell in square metres.

    Projected north-up rasters use the affine pixel area directly. Geographic
    north-up rasters use the CRS ellipsoid and calculate one geodesic cell area
    per row, which is then repeated across the columns.

    Args:
        template: Two-dimensional raster template with a valid CRS and affine
            transform.

    Returns:
        Two-dimensional float64 array with cell areas in square metres.

    Raises:
        ValueError: If the template is not two-dimensional, has no CRS, or uses
            a rotated/sheared affine transform.
    """
    if template.ndim != 2:
        raise ValueError("template must be a two-dimensional raster.")

    if template.rio.crs is None:
        raise ValueError("template must have a CRS to calculate cell areas.")

    transform = template.rio.transform(recalc=True)
    if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
        raise ValueError(
            "Rotated or sheared raster transforms are not supported for cell-area "
            "calculation."
        )

    crs = CRS.from_user_input(template.rio.crs)
    n_rows, n_cols = template.shape

    if crs.is_projected:
        cell_area = abs(transform.a * transform.e)
        return np.full((n_rows, n_cols), cell_area, dtype=np.float64)

    if not crs.is_geographic:
        raise ValueError(f"Unsupported raster CRS for cell-area calculation: {crs}.")

    geod = crs.get_geod()
    x_left = float(transform.c)
    x_right = float(transform.c + transform.a)
    row_areas = np.empty(n_rows, dtype=np.float64)

    for row in range(n_rows):
        y_top = float(transform.f + row * transform.e)
        y_bottom = float(y_top + transform.e)
        area_m2, _ = geod.polygon_area_perimeter(
            [x_left, x_right, x_right, x_left],
            [y_top, y_top, y_bottom, y_bottom],
        )
        row_areas[row] = abs(area_m2)

    return np.repeat(row_areas[:, None], n_cols, axis=1)


def select_cultivated_cells_by_area(
    selection_score: np.ndarray,
    eligible_mask: np.ndarray,
    cell_area_m2: np.ndarray,
    *,
    target_area_m2: float,
) -> np.ndarray:
    """Select the highest-ranked cells while meeting the static area target.

    Model cells are indivisible, so the selected full-cell area normally cannot
    equal the fractional HRL target exactly. Candidates are ranked by the supplied
    multi-year selection score and added until their cumulative area reaches or
    exceeds the target. This one-sided rule prevents the static farm domain from
    being too small to represent the largest annual cultivated area.

    Args:
        selection_score: Two-dimensional ranking score. Larger values receive
            precedence; non-finite values are treated as zero.
        eligible_mask: Boolean array limiting which cells may be selected.
        cell_area_m2: Two-dimensional model-cell areas in square metres.
        target_area_m2: Required minimum selected area in square metres.

    Returns:
        Boolean array identifying the selected static agricultural cells.

    Raises:
        ValueError: If array shapes differ, the target is not positive, or no
            eligible positive-score cell is available.
    """
    selection_score = np.asarray(selection_score, dtype=np.float64)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)
    cell_area_m2 = np.asarray(cell_area_m2, dtype=np.float64)

    if not (selection_score.shape == eligible_mask.shape == cell_area_m2.shape):
        raise ValueError(
            "selection_score, eligible_mask, and cell_area_m2 must have the same shape."
        )
    if target_area_m2 <= 0.0:
        raise ValueError("target_area_m2 must be positive.")

    scores = np.nan_to_num(selection_score, nan=0.0, posinf=0.0, neginf=0.0)
    candidates = eligible_mask & (scores > 0.0) & (cell_area_m2 > 0.0)
    candidate_indices = np.flatnonzero(candidates)
    if candidate_indices.size == 0:
        raise ValueError("No eligible HRL cultivated cells are available.")

    flat_scores = scores.ravel()[candidate_indices]
    flat_areas = cell_area_m2.ravel()[candidate_indices]
    target_area_m2 = float(np.clip(target_area_m2, flat_areas.min(), flat_areas.sum()))

    # Rank by descending multi-year support and use the flat index as a stable
    # tie-breaker so repeated builds are deterministic.
    order = np.lexsort((candidate_indices, -flat_scores))
    ordered_indices = candidate_indices[order]
    cumulative_area = np.cumsum(flat_areas[order])
    last_selected_index = int(
        np.searchsorted(cumulative_area, target_area_m2, side="left")
    )
    n_selected = min(last_selected_index + 1, cumulative_area.size)

    selected = np.zeros(scores.size, dtype=bool)
    selected[ordered_indices[:n_selected]] = True
    return selected.reshape(scores.shape)


def round_crop_states_to_area_targets(
    modal_crop_stack: np.ndarray,
    cultivated_fraction_stack: np.ndarray,
    coverage_fraction_stack: np.ndarray,
    cell_area_m2: np.ndarray,
    region_mask: np.ndarray,
    target_crop_areas_per_year: list[dict[int, float]],
    *,
    fallow_code: int = _HRL_FALLOW_CROP_CODE,
    missing_code: int = _HRL_MISSING_CROP_CODE,
    temporal_persistence_weight: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    """Round fractional HRL crops to binary model cells with area control.

    Every selected model cell can contain only one crop state per year. For each
    year and modal crop category, candidate cells are ranked by their annual HRL
    cultivated fraction, with a small preference for cells that are cultivated in
    many observed years. The highest-ranked cells are activated until their full-
    cell area is as close as possible to the native HRL target area for that crop.

    Cells activated in at least one year form the static agricultural union. A
    non-active year inside that union becomes valid fallow. Cells outside the union
    or without complete HRL coverage remain missing. Crop categories are never
    assigned to cells where that category is not the modal HRL crop, so the method
    improves absolute area conservation without spatially relocating crops.

    Args:
        modal_crop_stack: Integer array with shape ``(year, y, x)`` containing
            positive modal crop codes, native no-cropland ``0``, and a negative
            missing/outside sentinel.
        cultivated_fraction_stack: Fractional active-crop cover with the same
            shape as ``modal_crop_stack``.
        coverage_fraction_stack: HRL coverage fraction with the same shape.
        cell_area_m2: Model-cell area array with shape ``(y, x)``.
        region_mask: Boolean mask selecting the active region.
        target_crop_areas_per_year: Native HRL target area by positive crop code
            for every year.
        fallow_code: Final code assigned to non-active years inside the union.
        missing_code: Code assigned outside the union or where coverage is
            incomplete.
        temporal_persistence_weight: Weight in ``[0, 1]`` given to the fraction
            of observed years in which a cell is actively cultivated. The
            remaining weight is given to the current-year cultivated fraction.

    Returns:
        Tuple with the rounded crop-state stack, static agricultural-union mask,
        a per-year/per-category diagnostics table, and the number of potential
        crop cells excluded because HRL coverage is incomplete.

    Raises:
        ValueError: If array shapes, target years, codes, or weight are invalid.
        RuntimeError: If the resulting union contains an all-fallow sequence or
            an unresolved missing state.
    """
    modal_crop_stack = np.asarray(modal_crop_stack, dtype=np.int32)
    cultivated_fraction_stack = np.asarray(cultivated_fraction_stack, dtype=np.float64)
    coverage_fraction_stack = np.asarray(coverage_fraction_stack, dtype=np.float64)
    cell_area_m2 = np.asarray(cell_area_m2, dtype=np.float64)
    region_mask = np.asarray(region_mask, dtype=bool)

    if modal_crop_stack.ndim != 3:
        raise ValueError("modal_crop_stack must have shape (year, y, x).")
    if cultivated_fraction_stack.shape != modal_crop_stack.shape:
        raise ValueError("cultivated_fraction_stack must align with modal_crop_stack.")
    if coverage_fraction_stack.shape != modal_crop_stack.shape:
        raise ValueError("coverage_fraction_stack must align with modal_crop_stack.")
    if cell_area_m2.shape != modal_crop_stack.shape[1:]:
        raise ValueError("cell_area_m2 must align with the spatial crop grid.")
    if region_mask.shape != cell_area_m2.shape:
        raise ValueError("region_mask must align with cell_area_m2.")
    if len(target_crop_areas_per_year) != modal_crop_stack.shape[0]:
        raise ValueError("One target crop-area mapping is required per year.")
    if not 0.0 <= temporal_persistence_weight <= 1.0:
        raise ValueError("temporal_persistence_weight must be between 0 and 1.")
    if fallow_code >= 0 or missing_code >= 0 or fallow_code == missing_code:
        raise ValueError("fallow_code and missing_code must be distinct negatives.")
    if np.any(cell_area_m2[region_mask] <= 0.0):
        raise ValueError("All active model cells must have positive area.")

    n_years = modal_crop_stack.shape[0]
    complete_coverage = np.all(coverage_fraction_stack > 0.0, axis=0)
    has_modal_crop_any_year = np.any(modal_crop_stack > 0, axis=0)
    incomplete_candidate_mask = (
        region_mask & has_modal_crop_any_year & ~complete_coverage
    )
    incomplete_cell_count = int(np.count_nonzero(incomplete_candidate_mask))

    eligible = region_mask & complete_coverage
    active_frequency = np.mean(modal_crop_stack > 0, axis=0, dtype=np.float64)
    rounded = np.full(modal_crop_stack.shape, missing_code, dtype=np.int32)
    diagnostics: list[dict[str, float | int]] = []
    flat_indices = np.arange(cell_area_m2.size, dtype=np.int64).reshape(
        cell_area_m2.shape
    )

    for year_index in range(n_years):
        modal_year = modal_crop_stack[year_index]
        fraction_year = np.clip(cultivated_fraction_stack[year_index], 0.0, 1.0)
        targets = {
            int(code): max(float(area), 0.0)
            for code, area in target_crop_areas_per_year[year_index].items()
            if int(code) > 0
        }
        modal_codes = np.unique(modal_year[eligible & (modal_year > 0)])
        crop_codes = sorted(set(targets) | {int(code) for code in modal_codes})

        for crop_code in crop_codes:
            candidate_mask = eligible & (modal_year == crop_code)
            candidate_positions = np.flatnonzero(candidate_mask)
            target_area_m2 = targets.get(crop_code, 0.0)

            if candidate_positions.size == 0:
                diagnostics.append(
                    {
                        "year_index": year_index,
                        "crop_code": crop_code,
                        "target_area_m2": target_area_m2,
                        "candidate_area_m2": 0.0,
                        "assigned_area_m2": 0.0,
                        "difference_m2": -target_area_m2,
                        "candidate_cells": 0,
                        "selected_cells": 0,
                    }
                )
                continue

            candidate_area = cell_area_m2.ravel()[candidate_positions]
            candidate_fraction = fraction_year.ravel()[candidate_positions]
            candidate_persistence = active_frequency.ravel()[candidate_positions]
            candidate_score = (
                1.0 - temporal_persistence_weight
            ) * candidate_fraction + temporal_persistence_weight * candidate_persistence
            candidate_flat_index = flat_indices.ravel()[candidate_positions]
            order = np.lexsort((candidate_flat_index, -candidate_score))
            ordered_positions = candidate_positions[order]
            ordered_areas = candidate_area[order]
            cumulative_area = np.cumsum(ordered_areas, dtype=np.float64)
            possible_areas = np.concatenate(([0.0], cumulative_area))
            absolute_error = np.abs(possible_areas - target_area_m2)
            minimum_error = float(absolute_error.min())
            closest = np.flatnonzero(
                np.isclose(absolute_error, minimum_error, rtol=0.0, atol=1e-9)
            )
            # Prefer the smaller represented area for an exact tie so rounding
            # does not systematically inflate crop and water-use totals.
            selected_count = int(closest[0])
            selected_positions = ordered_positions[:selected_count]
            rounded[year_index].ravel()[selected_positions] = crop_code
            assigned_area_m2 = float(possible_areas[selected_count])

            diagnostics.append(
                {
                    "year_index": year_index,
                    "crop_code": crop_code,
                    "target_area_m2": target_area_m2,
                    "candidate_area_m2": float(candidate_area.sum()),
                    "assigned_area_m2": assigned_area_m2,
                    "difference_m2": assigned_area_m2 - target_area_m2,
                    "candidate_cells": int(candidate_positions.size),
                    "selected_cells": selected_count,
                }
            )

    agricultural_union = region_mask & np.any(rounded > 0, axis=0)
    rounded[:, agricultural_union] = np.where(
        rounded[:, agricultural_union] > 0,
        rounded[:, agricultural_union],
        fallow_code,
    )
    rounded[:, ~agricultural_union] = missing_code

    union_sequences = rounded[:, agricultural_union]
    if union_sequences.size == 0:
        raise RuntimeError("Area-constrained rounding produced no agricultural cells.")
    if np.any(np.all(union_sequences == fallow_code, axis=0)):
        raise RuntimeError("The agricultural union contains an all-fallow sequence.")
    if np.any(union_sequences == missing_code):
        raise RuntimeError("The agricultural union contains missing HRL states.")

    return (
        rounded,
        agricultural_union,
        pd.DataFrame(diagnostics),
        incomplete_cell_count,
    )


def _target_cell_counts_from_areas(
    target_farms: list[TargetFarm],
    n_cells: int,
) -> tuple[list[TargetFarm], np.ndarray]:
    """Convert continuous target areas to exact positive cell counts.

    Targets are ordered from largest to smallest, scaled to the available number of
    selected cells, and rounded while retaining at least one cell per farm. The
    rounding remainder is distributed deterministically, and the final counts are
    validated to exhaust the selected domain exactly.

    Args:
        target_farms: Continuous Lowder-derived target farms.
        n_cells: Number of selected model cells that must be assigned.

    Returns:
        The targets ordered by descending area and an aligned integer cell-count
        array whose values are positive and sum to ``n_cells``.

    Raises:
        RuntimeError: If the final rounded counts are not positive or do not sum to
            the selected cell count.
        ValueError: If no targets are supplied, areas are invalid, there are more
            farms than cells, or rounding cannot retain one cell per farm.
    """
    if n_cells <= 0:
        raise ValueError("n_cells must be positive.")
    if not target_farms:
        raise ValueError("target_farms must contain at least one target.")
    if len(target_farms) > n_cells:
        raise ValueError(
            f"Cannot create {len(target_farms)} farms from only {n_cells} cultivated "
            "model cells. Reduce the Lowder target count or use a finer subgrid."
        )

    ordered_targets = sorted(
        target_farms,
        key=lambda target: target.target_area_m2,
        reverse=True,
    )
    target_areas = np.asarray(
        [target.target_area_m2 for target in ordered_targets],
        dtype=np.float64,
    )

    if not np.isfinite(target_areas).all() or (target_areas <= 0).any():
        raise ValueError("All target farm areas must be finite and positive.")

    raw_counts = target_areas / target_areas.sum() * n_cells
    counts = np.floor(raw_counts).astype(np.int32)
    counts[counts < 1] = 1

    difference = int(n_cells - counts.sum())
    if difference > 0:
        fractional = raw_counts - np.floor(raw_counts)
        order = np.lexsort((np.arange(counts.size), -fractional))
        for index in range(difference):
            counts[order[index % order.size]] += 1
    elif difference < 0:
        for _ in range(-difference):
            removable = counts > 1
            if not removable.any():
                raise ValueError(
                    "Could not reduce target cell counts while retaining at "
                    "least one cell per farm."
                )
            excess = counts.astype(np.float64) - raw_counts
            excess[~removable] = -np.inf
            remove_index = int(np.argmax(excess))
            counts[remove_index] -= 1

    if (counts < 1).any() or int(counts.sum()) != n_cells:
        raise RuntimeError(
            "Target farm cell counts are not positive or do not sum exactly."
        )

    return ordered_targets, counts


@njit(cache=True)
def _add_raster_frontier_neighbors(
    flat_index: int,
    n_rows: int,
    n_cols: int,
    cultivated_flat: np.ndarray,
    assignments_flat: np.ndarray,
    frontier: np.ndarray,
    frontier_size: int,
    frontier_generation: np.ndarray,
    generation: int,
) -> int:
    """Add eligible four-neighbour cells to a farm frontier.

    A generation marker prevents the same cell from being inserted more than once
    while one farmer is growing.

    Args:
        flat_index: Flat index of the cell whose neighbours are inspected.
        n_rows: Number of raster rows.
        n_cols: Number of raster columns.
        cultivated_flat: Flat boolean cultivated-cell mask.
        assignments_flat: Flat current farmer assignments; negative values are free.
        frontier: Preallocated array storing frontier cell indices.
        frontier_size: Number of currently valid frontier entries.
        frontier_generation: Generation marker per raster cell.
        generation: Marker identifying the farm currently being grown.

    Returns:
        Updated number of valid entries in ``frontier``.
    """
    row = flat_index // n_cols
    col = flat_index - row * n_cols

    if row > 0:
        neighbor = flat_index - n_cols
        if (
            cultivated_flat[neighbor]
            and assignments_flat[neighbor] < 0
            and frontier_generation[neighbor] != generation
        ):
            frontier[frontier_size] = neighbor
            frontier_size += 1
            frontier_generation[neighbor] = generation

    if row + 1 < n_rows:
        neighbor = flat_index + n_cols
        if (
            cultivated_flat[neighbor]
            and assignments_flat[neighbor] < 0
            and frontier_generation[neighbor] != generation
        ):
            frontier[frontier_size] = neighbor
            frontier_size += 1
            frontier_generation[neighbor] = generation

    if col > 0:
        neighbor = flat_index - 1
        if (
            cultivated_flat[neighbor]
            and assignments_flat[neighbor] < 0
            and frontier_generation[neighbor] != generation
        ):
            frontier[frontier_size] = neighbor
            frontier_size += 1
            frontier_generation[neighbor] = generation

    if col + 1 < n_cols:
        neighbor = flat_index + 1
        if (
            cultivated_flat[neighbor]
            and assignments_flat[neighbor] < 0
            and frontier_generation[neighbor] != generation
        ):
            frontier[frontier_size] = neighbor
            frontier_size += 1
            frontier_generation[neighbor] = generation

    return frontier_size


@njit(cache=True)
def _raster_candidate_score(
    candidate: int,
    seed: int,
    crop_sequences_flat: np.ndarray,
    n_cols: int,
    target_size_cells: int,
    distance_weight: float,
    crop_sequence_weight: float,
    switch_timing_weight: float,
    min_valid_crop_sequence_overlap: int,
) -> float:
    """Score a candidate cell for the current Lowder-guided farm.

    The score combines compactness relative to the seed, complete-sequence
    similarity, and crop-switch timing similarity using normalized external weights.

    Args:
        candidate: Flat candidate-cell index.
        seed: Flat seed-cell index of the current farm.
        crop_sequences_flat: Crop states with shape ``(year, flat_cell)``.
        n_cols: Number of raster columns.
        target_size_cells: Target size of the current farm in cells.
        distance_weight: Weight for compactness.
        crop_sequence_weight: Weight for complete-sequence similarity.
        switch_timing_weight: Weight for switch-timing similarity.
        min_valid_crop_sequence_overlap: Minimum comparable years required for a
            positive sequence-similarity score.

    Returns:
        Weighted candidate score; larger values receive precedence.
    """
    candidate_row = candidate // n_cols
    candidate_col = candidate - candidate_row * n_cols
    seed_row = seed // n_cols
    seed_col = seed - seed_row * n_cols

    delta_row = candidate_row - seed_row
    delta_col = candidate_col - seed_col
    distance_cells = math.sqrt(delta_row * delta_row + delta_col * delta_col)
    target_radius_cells = max(math.sqrt(target_size_cells / math.pi), 1.0)
    compactness = 1.0 / (1.0 + distance_cells / target_radius_cells)

    crop_similarity = _crop_sequence_similarity_numba(
        crop_sequences_flat[:, seed],
        crop_sequences_flat[:, candidate],
        _HRL_MISSING_CROP_CODE,
        _HRL_FALLOW_CROP_CODE,
        min_valid_crop_sequence_overlap,
    )
    switch_similarity = _switch_timing_similarity_numba(
        crop_sequences_flat[:, seed],
        crop_sequences_flat[:, candidate],
        _HRL_MISSING_CROP_CODE,
    )

    return (
        distance_weight * compactness
        + crop_sequence_weight * crop_similarity
        + switch_timing_weight * switch_similarity
    )


@njit(cache=True)
def _grow_raster_farms_numba(
    cultivated_mask: np.ndarray,
    crop_sequences: np.ndarray,
    target_cell_counts: np.ndarray,
    seed_order: np.ndarray,
    distance_weight: float,
    crop_sequence_weight: float,
    switch_timing_weight: float,
    min_valid_crop_sequence_overlap: int,
    jump_candidate_sample: int,
    max_jump_distance_cells: float,
) -> np.ndarray:
    """Assign every selected cell to one Lowder-guided target farm.

    Farms first expand through connected frontier cells. When a connected component
    is exhausted, a nearby sequence-compatible unassigned cell starts an additional
    parcel. Each farm stops at its exact target cell count.

    Args:
        cultivated_mask: Boolean static agricultural mask.
        crop_sequences: Crop states with shape ``(year, y, x)``.
        target_cell_counts: Exact positive cell count per farm.
        seed_order: Deterministic shuffled order of selected cell indices.
        distance_weight: Normalized compactness weight.
        crop_sequence_weight: Normalized sequence-similarity weight.
        switch_timing_weight: Normalized switch-timing weight.
        min_valid_crop_sequence_overlap: Minimum comparable sequence years.
        jump_candidate_sample: Candidate cells inspected for a disconnected parcel.
        max_jump_distance_cells: Preferred maximum parcel jump in cell lengths.

    Returns:
        Two-dimensional compact local farmer-ID raster.

    Raises:
        RuntimeError: If no unassigned seed or fallback cell remains before all
            target farms reach their required sizes.
    """
    n_rows, n_cols = cultivated_mask.shape
    n_cells_total = n_rows * n_cols
    cultivated_flat = cultivated_mask.ravel()
    crop_sequences_flat = crop_sequences.reshape(crop_sequences.shape[0], n_cells_total)
    assignments_flat = np.full(n_cells_total, -1, dtype=np.int32)

    frontier = np.empty(n_cells_total, dtype=np.int32)
    frontier_generation = np.zeros(n_cells_total, dtype=np.int32)
    generation = 0
    seed_cursor = 0

    for farmer_id in range(target_cell_counts.size):
        target_size = int(target_cell_counts[farmer_id])

        while (
            seed_cursor < seed_order.size
            and assignments_flat[seed_order[seed_cursor]] >= 0
        ):
            seed_cursor += 1

        if seed_cursor >= seed_order.size:
            raise RuntimeError("No unassigned cultivated seed cell remains.")

        seed = int(seed_order[seed_cursor])
        assignments_flat[seed] = farmer_id
        assigned_count = 1
        seed_cursor += 1

        generation += 1
        frontier_size = 0
        frontier_size = _add_raster_frontier_neighbors(
            seed,
            n_rows,
            n_cols,
            cultivated_flat,
            assignments_flat,
            frontier,
            frontier_size,
            frontier_generation,
            generation,
        )

        while assigned_count < target_size:
            best_position = -1
            best_candidate = -1
            best_score = -1.0e30

            position = 0
            while position < frontier_size:
                candidate = int(frontier[position])
                if assignments_flat[candidate] >= 0:
                    frontier_size -= 1
                    frontier[position] = frontier[frontier_size]
                    continue

                score = _raster_candidate_score(
                    candidate,
                    seed,
                    crop_sequences_flat,
                    n_cols,
                    target_size,
                    distance_weight,
                    crop_sequence_weight,
                    switch_timing_weight,
                    min_valid_crop_sequence_overlap,
                )
                if score > best_score or (
                    score == best_score and candidate < best_candidate
                ):
                    best_score = score
                    best_candidate = candidate
                    best_position = position
                position += 1

            if best_candidate < 0:
                # The current connected cropland patch is exhausted. Select a
                # nearby crop-compatible unassigned cell as a new parcel seed.
                sampled = 0
                scan = seed_cursor
                best_jump = -1
                best_jump_score = -1.0e30
                seed_row = seed // n_cols
                seed_col = seed - seed_row * n_cols

                while scan < seed_order.size and sampled < jump_candidate_sample:
                    candidate = int(seed_order[scan])
                    scan += 1
                    if assignments_flat[candidate] >= 0:
                        continue
                    sampled += 1

                    candidate_row = candidate // n_cols
                    candidate_col = candidate - candidate_row * n_cols
                    delta_row = candidate_row - seed_row
                    delta_col = candidate_col - seed_col
                    distance_cells = math.sqrt(
                        delta_row * delta_row + delta_col * delta_col
                    )
                    if distance_cells > max_jump_distance_cells:
                        continue

                    score = _raster_candidate_score(
                        candidate,
                        seed,
                        crop_sequences_flat,
                        n_cols,
                        target_size,
                        distance_weight,
                        crop_sequence_weight,
                        switch_timing_weight,
                        min_valid_crop_sequence_overlap,
                    )
                    if score > best_jump_score or (
                        score == best_jump_score and candidate < best_jump
                    ):
                        best_jump_score = score
                        best_jump = candidate

                if best_jump < 0:
                    scan = seed_cursor
                    while scan < seed_order.size:
                        candidate = int(seed_order[scan])
                        if assignments_flat[candidate] < 0:
                            best_jump = candidate
                            break
                        scan += 1

                if best_jump < 0:
                    raise RuntimeError(
                        "No unassigned cultivated cell remains before all farm "
                        "targets are filled."
                    )

                best_candidate = best_jump
            else:
                frontier_size -= 1
                frontier[best_position] = frontier[frontier_size]

            assignments_flat[best_candidate] = farmer_id
            assigned_count += 1
            frontier_size = _add_raster_frontier_neighbors(
                best_candidate,
                n_rows,
                n_cols,
                cultivated_flat,
                assignments_flat,
                frontier,
                frontier_size,
                frontier_generation,
                generation,
            )

    return assignments_flat.reshape((n_rows, n_cols))


@njit(cache=True)
def _find_root(parent: np.ndarray, index: int) -> int:
    """Find a disjoint-set root and compress the traversed path.

    Args:
        parent: Parent index for every disjoint-set element.
        index: Element whose root is requested.

    Returns:
        Root index of ``index``.
    """
    root = index
    while parent[root] != root:
        root = parent[root]
    while parent[index] != index:
        next_index = parent[index]
        parent[index] = root
        index = next_index
    return root


@njit(cache=True)
def _union_roots(parent: np.ndarray, rank: np.ndarray, left: int, right: int) -> None:
    """Union two disjoint-set roots."""
    root_left = _find_root(parent, left)
    root_right = _find_root(parent, right)
    if root_left == root_right:
        return
    if rank[root_left] < rank[root_right]:
        parent[root_left] = root_right
    elif rank[root_left] > rank[root_right]:
        parent[root_right] = root_left
    else:
        parent[root_right] = root_left
        rank[root_left] += 1


@njit(cache=True)
def _count_farm_components_numba(
    farms: np.ndarray,
    n_farmers: int,
) -> np.ndarray:
    """Count four-connected parcels belonging to every farmer.

    Args:
        farms: Two-dimensional raster of compact farmer IDs with negative nodata.
        n_farmers: Number of compact farmer IDs.

    Returns:
        One-dimensional parcel count aligned with farmer ID.
    """
    n_rows, n_cols = farms.shape
    n_total = n_rows * n_cols
    flat = farms.ravel()
    parent = np.arange(n_total, dtype=np.int32)
    rank = np.zeros(n_total, dtype=np.int8)

    for row in range(n_rows):
        for col in range(n_cols):
            index = row * n_cols + col
            farmer_id = flat[index]
            if farmer_id < 0:
                continue
            if col > 0 and flat[index - 1] == farmer_id:
                _union_roots(parent, rank, index, index - 1)
            if row > 0 and flat[index - n_cols] == farmer_id:
                _union_roots(parent, rank, index, index - n_cols)

    counts = np.zeros(n_farmers, dtype=np.int32)
    for index in range(n_total):
        farmer_id = flat[index]
        if farmer_id >= 0 and _find_root(parent, index) == index:
            counts[farmer_id] += 1
    return counts


def _optimize_multiyear_sequence_assignments_numba(
    farmer_areas_m2: np.ndarray,
    sequence_category_indices: np.ndarray,
    candidate_sequence_ids: np.ndarray,
    candidate_priors: np.ndarray,
    target_areas_m2: np.ndarray,
    processing_order: np.ndarray,
    initial_assignments: np.ndarray,
    initial_priors: np.ndarray,
    alignment_weight: float,
    search_passes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Improve complete-sequence assignments without constructing new sequences.

    The optimizer starts from the supplied preliminary sequence for every farmer.
    A move always replaces the farmer's complete multi-year sequence at once. The
    area-fit gain is evaluated jointly over all years, while the candidate prior
    preserves locally supported, common, and low-fallow sequences.

    Args:
        farmer_areas_m2: Full-cell area of every farmer.
        sequence_category_indices: Crop-category index for every sequence and year.
        candidate_sequence_ids: Candidate original-sequence IDs per farmer.
        candidate_priors: Preference score aligned with ``candidate_sequence_ids``.
        target_areas_m2: Multi-year target matrix with shape ``(year, category)``.
        processing_order: Deterministic order in which farmers are reconsidered.
        initial_assignments: Preliminary original-sequence ID per farmer.
        initial_priors: Preference score of each preliminary assignment.
        alignment_weight: Weight placed on multi-year crop-area fit.
        search_passes: Maximum deterministic reassignment passes.

    Returns:
        Tuple containing final sequence IDs, their preference scores, and the
        resulting assigned-area matrix with shape ``(year, category)``.
    """
    assignments = initial_assignments.copy()
    assignment_priors = initial_priors.copy()
    assigned_areas = np.zeros_like(target_areas_m2)
    n_years = sequence_category_indices.shape[1]

    for farmer_id in range(farmer_areas_m2.size):
        sequence_id = int(assignments[farmer_id])
        farm_area = float(farmer_areas_m2[farmer_id])
        for year_index in range(n_years):
            category_index = int(sequence_category_indices[sequence_id, year_index])
            assigned_areas[year_index, category_index] += farm_area

    preference_weight = 1.0 - alignment_weight
    normalizer_years = max(float(n_years), 1.0)

    for _ in range(max(search_passes, 0)):
        moved = 0
        for order_index in range(processing_order.size):
            farmer_id = int(processing_order[order_index])
            farm_area = max(float(farmer_areas_m2[farmer_id]), 1e-12)
            current_sequence = int(assignments[farmer_id])
            current_prior = float(assignment_priors[farmer_id])
            best_sequence = current_sequence
            best_prior = current_prior
            best_gain = 0.0

            for candidate_index in range(candidate_sequence_ids.shape[1]):
                candidate_sequence = int(
                    candidate_sequence_ids[farmer_id, candidate_index]
                )
                if candidate_sequence < 0 or candidate_sequence == current_sequence:
                    continue

                duplicate = False
                for previous_index in range(candidate_index):
                    if (
                        candidate_sequence_ids[farmer_id, previous_index]
                        == candidate_sequence
                    ):
                        duplicate = True
                        break
                if duplicate:
                    continue

                before_error = 0.0
                after_error = 0.0
                for year_index in range(n_years):
                    current_category = int(
                        sequence_category_indices[current_sequence, year_index]
                    )
                    candidate_category = int(
                        sequence_category_indices[candidate_sequence, year_index]
                    )
                    if current_category == candidate_category:
                        continue

                    current_assigned = assigned_areas[year_index, current_category]
                    candidate_assigned = assigned_areas[year_index, candidate_category]
                    before_error += abs(
                        current_assigned - target_areas_m2[year_index, current_category]
                    )
                    before_error += abs(
                        candidate_assigned
                        - target_areas_m2[year_index, candidate_category]
                    )
                    after_error += abs(
                        current_assigned
                        - farm_area
                        - target_areas_m2[year_index, current_category]
                    )
                    after_error += abs(
                        candidate_assigned
                        + farm_area
                        - target_areas_m2[year_index, candidate_category]
                    )

                # Moving one farm can alter at most two categories per year. This
                # normalization keeps the alignment term approximately in [-1, 1].
                normalized_alignment_gain = (before_error - after_error) / (
                    2.0 * farm_area * normalizer_years
                )
                candidate_prior = float(candidate_priors[farmer_id, candidate_index])
                preference_gain = candidate_prior - current_prior
                gain = (
                    alignment_weight * normalized_alignment_gain
                    + preference_weight * preference_gain
                )

                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_sequence = candidate_sequence
                    best_prior = candidate_prior

            if best_sequence == current_sequence:
                continue

            for year_index in range(n_years):
                current_category = int(
                    sequence_category_indices[current_sequence, year_index]
                )
                candidate_category = int(
                    sequence_category_indices[best_sequence, year_index]
                )
                if current_category == candidate_category:
                    continue
                assigned_areas[year_index, current_category] -= farm_area
                assigned_areas[year_index, candidate_category] += farm_area

            assignments[farmer_id] = best_sequence
            assignment_priors[farmer_id] = best_prior
            moved += 1

        if moved == 0:
            break

    return assignments, assignment_priors, assigned_areas


def _multiyear_area_fit_score(
    assigned_areas_m2: np.ndarray,
    target_areas_m2: np.ndarray,
) -> float:
    """Return a bounded multi-year area-fit score in percent.

    Args:
        assigned_areas_m2: Assigned area matrix with shape ``(year, category)``.
        target_areas_m2: Target area matrix aligned with ``assigned_areas_m2``.

    Returns:
        ``100`` for an exact fit and progressively smaller values as total
        absolute area error increases.
    """
    denominator = float(np.abs(target_areas_m2).sum())
    if denominator <= 0.0:
        return 100.0
    absolute_error = float(np.abs(assigned_areas_m2 - target_areas_m2).sum())
    return max(0.0, 100.0 * (1.0 - absolute_error / denominator))


def _regional_sequence_fallback_candidates(
    sequences: np.ndarray,
    sequence_area_m2: np.ndarray,
    sequence_cell_counts: np.ndarray,
    preliminary_sequence_ids: np.ndarray,
    target_crop_areas_per_year: list[dict[int, float]],
    *,
    fallow_code: int,
    fallow_penalty: float,
    pool_size: int,
    candidates_per_sequence: int,
    similarity_weight: float = 0.75,
    commonness_weight: float = 0.20,
    low_fallow_weight: float = 0.05,
    chunk_size: int = 1_024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find similar and common regional fallback sequences in vectorized chunks.

    Only complete sequences observed on regional pixels are eligible. The candidate
    pool combines prevalent low-fallow sequences with representatives of rare target
    crops. Similarity is evaluated once per distinct preliminary sequence rather than
    once per farmer.

    Args:
        sequences: Unique complete original sequences with dimensions
            ``(sequence, year)``.
        sequence_area_m2: Regional full-cell area represented by every sequence.
        sequence_cell_counts: Number of source cells carrying every sequence.
        preliminary_sequence_ids: Preliminary sequence ID per farmer.
        target_crop_areas_per_year: Native crop-area targets for every year.
        fallow_code: Code representing genuine fallow inside the field domain.
        fallow_penalty: Penalty applied to fallow-heavy sequences.
        pool_size: Maximum size of the regional comparison pool.
        candidates_per_sequence: Number of fallback sequences retained per distinct
            preliminary sequence.
        similarity_weight: Weight assigned to full-sequence similarity.
        commonness_weight: Weight assigned to regional prevalence.
        low_fallow_weight: Weight assigned to avoiding fallow-heavy sequences.
        chunk_size: Distinct preliminary sequences processed per vectorized chunk.

    Returns:
        Distinct preliminary sequence IDs, farmer-to-distinct inverse indices,
        fallback sequence IDs, and aligned fallback preference scores.

    Raises:
        ValueError: If candidate settings are invalid or no usable regional sequence
            pool can be constructed.
    """
    n_sequences, n_years = sequences.shape
    if n_sequences == 0:
        raise ValueError("At least one complete regional sequence is required.")
    if candidates_per_sequence < 1:
        raise ValueError("candidates_per_sequence must be at least 1.")

    sequence_cell_counts = np.asarray(sequence_cell_counts, dtype=np.float64)
    if sequence_cell_counts.shape != sequence_area_m2.shape:
        raise ValueError("sequence_cell_counts must align with sequence_area_m2.")

    fallow_fraction = np.mean(sequences == fallow_code, axis=1, dtype=np.float64)
    log_area_all = np.log1p(sequence_area_m2)
    log_count_all = np.log1p(sequence_cell_counts)
    area_commonness_all = log_area_all / max(
        float(log_area_all.max(initial=0.0)),
        1e-12,
    )
    count_commonness_all = log_count_all / max(
        float(log_count_all.max(initial=0.0)),
        1e-12,
    )
    combined_commonness_all = 0.50 * area_commonness_all + 0.50 * count_commonness_all
    prevalence_score = combined_commonness_all * (
        1.0 - fallow_penalty * fallow_fraction
    )
    base_order = np.lexsort((np.arange(n_sequences), -prevalence_score))
    selected_pool = list(base_order[: min(max(pool_size, 1), n_sequences)])

    # Preserve candidate coverage for rare target crops. Without this addition a
    # pure top-frequency pool can omit the only original sequences containing a
    # small but regionally important crop category.
    representatives_per_target = max(2, candidates_per_sequence)
    for year_index, targets in enumerate(target_crop_areas_per_year):
        for crop_code, target_area in targets.items():
            if int(crop_code) <= 0 or float(target_area) <= 0.0:
                continue
            matching = np.flatnonzero(sequences[:, year_index] == int(crop_code))
            if matching.size == 0:
                continue
            order = np.lexsort((matching, -prevalence_score[matching]))
            selected_pool.extend(matching[order[:representatives_per_target]].tolist())

    selected_pool.extend(np.unique(preliminary_sequence_ids).tolist())
    pool_ids = np.unique(np.asarray(selected_pool, dtype=np.int32))
    pool_sequences = sequences[pool_ids]
    pool_fallow_fraction = fallow_fraction[pool_ids]
    commonness = combined_commonness_all[pool_ids]

    weight_sum = similarity_weight + commonness_weight + low_fallow_weight
    if weight_sum <= 0.0:
        raise ValueError("Regional fallback weights must sum to a positive value.")
    similarity_weight /= weight_sum
    commonness_weight /= weight_sum
    low_fallow_weight /= weight_sum

    distinct_preliminary, farmer_inverse = np.unique(
        preliminary_sequence_ids,
        return_inverse=True,
    )
    n_keep = min(candidates_per_sequence, max(pool_ids.size - 1, 1))
    fallback_ids = np.full(
        (distinct_preliminary.size, candidates_per_sequence),
        -1,
        dtype=np.int32,
    )
    fallback_scores = np.zeros_like(fallback_ids, dtype=np.float64)
    pool_switches = pool_sequences[:, 1:] != pool_sequences[:, :-1]

    for chunk_start in range(0, distinct_preliminary.size, max(chunk_size, 1)):
        chunk_end = min(chunk_start + max(chunk_size, 1), distinct_preliminary.size)
        base_ids = distinct_preliminary[chunk_start:chunk_end]
        base = sequences[base_ids]

        same_crop = base[:, None, :] == pool_sequences[None, :, :]
        positive_match = same_crop & (base[:, None, :] > 0)
        fallow_match = same_crop & (base[:, None, :] == fallow_code)
        identity_similarity = (
            positive_match.sum(axis=2, dtype=np.float64)
            + 0.25 * fallow_match.sum(axis=2, dtype=np.float64)
        ) / max(float(n_years), 1.0)

        base_switches = base[:, 1:] != base[:, :-1]
        switch_union = np.logical_or(
            base_switches[:, None, :],
            pool_switches[None, :, :],
        ).sum(axis=2)
        switch_intersection = np.logical_and(
            base_switches[:, None, :],
            pool_switches[None, :, :],
        ).sum(axis=2)
        switch_similarity = np.divide(
            switch_intersection,
            switch_union,
            out=np.full(identity_similarity.shape, 0.5, dtype=np.float64),
            where=switch_union > 0,
        )
        similarity = 0.80 * identity_similarity + 0.20 * switch_similarity
        scores = (
            similarity_weight * similarity
            + commonness_weight * commonness[None, :]
            + low_fallow_weight * (1.0 - pool_fallow_fraction)[None, :]
        )

        # The preliminary sequence is already available as a local/current option;
        # fallback slots should add genuinely different regional alternatives.
        scores[base_ids[:, None] == pool_ids[None, :]] = -np.inf
        if pool_ids.size == 1:
            continue

        top_positions = np.argpartition(-scores, n_keep - 1, axis=1)[:, :n_keep]
        top_scores = np.take_along_axis(scores, top_positions, axis=1)
        top_order = np.argsort(-top_scores, axis=1, kind="stable")
        top_positions = np.take_along_axis(top_positions, top_order, axis=1)
        top_scores = np.take_along_axis(top_scores, top_order, axis=1)

        fallback_ids[chunk_start:chunk_end, :n_keep] = pool_ids[top_positions]
        fallback_scores[chunk_start:chunk_end, :n_keep] = top_scores

    return (
        distinct_preliminary,
        farmer_inverse.astype(np.int32),
        fallback_ids,
        fallback_scores,
    )


def assign_farmer_sequences_to_area_targets(
    farm_values: np.ndarray,
    crop_sequences: np.ndarray,
    cell_area_m2: np.ndarray,
    farmer_areas_m2: np.ndarray,
    target_crop_areas_per_year: list[dict[int, float]],
    *,
    fallow_code: int = _HRL_FALLOW_CROP_CODE,
    missing_code: int = _HRL_MISSING_CROP_CODE,
    alignment_weight: float = 0.80,
    max_local_sequences: int = 4,
    max_regional_sequences: int = 4,
    regional_sequence_pool_size: int = 512,
    local_search_passes: int = 2,
    regional_search_passes: int = 2,
    local_fit_threshold_pct: float = 99.0,
    fallow_penalty: float = 0.35,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Assign complete original crop sequences in one multi-year optimization.

    The preliminary sequence of each farmer is the complete original pixel
    sequence occupying the largest full-cell area inside that farm. The first
    optimization stage may choose another complete sequence observed locally in
    the same farm. Only when the resulting regional multi-year fit remains below
    ``local_fit_threshold_pct`` is a second stage allowed to use similar, common
    complete sequences observed elsewhere in the same region.

    Fallow is retained as a meaningful state, while native outside/missing years
    invalidate a pixel sequence as a complete candidate. Fallow-heavy sequences
    remain available but receive a lower preference. At no point are annual crop
    labels combined independently, so every final farmer sequence is guaranteed
    to be an original observed full-year sequence.

    Args:
        farm_values: Compact two-dimensional farm-ID raster.
        crop_sequences: Combined HRL crop codes with shape ``(year, y, x)``.
        cell_area_m2: Full-cell model areas aligned with ``farm_values``.
        farmer_areas_m2: Actual full-cell area of every compact farmer.
        target_crop_areas_per_year: Native crop-area targets by combined crop code
            for every requested year.
        fallow_code: Genuine fallow/no-cropland code inside the field domain.
        missing_code: Native outside-area or missing-observation code.
        alignment_weight: Weight placed on joint multi-year crop-area fit.
        max_local_sequences: Maximum locally observed complete sequences retained
            per farmer, ordered by within-farm area support.
        max_regional_sequences: Maximum non-local regional fallback sequences per
            farmer.
        regional_sequence_pool_size: Number of common regional sequences compared
            when constructing fallback candidates. Rare target-crop representatives
            are added independently of this base pool.
        local_search_passes: Reassignment passes using local sequences only.
        regional_search_passes: Additional passes after regional fallback is enabled.
        local_fit_threshold_pct: Minimum fit score at which regional fallback is
            considered unnecessary.
        fallow_penalty: Preference penalty in ``0..1`` applied in proportion to a
            sequence's share of fallow years.

    Returns:
        Tuple containing:
        - final original crop sequences with shape ``(farmer, year)``;
        - farmer-level quality information, including ``crop_sequence_quality_flag``
          where 2 is the local dominant sequence, 1 is another local sequence, and
          0 is a regional fallback;
        - per-year, per-crop area diagnostics compatible with the Europe workflow.

    Raises:
        ValueError: If inputs are inconsistent or no complete original regional
            sequence exists.
        RuntimeError: If a final assignment is not an original complete sequence.
    """
    farm_values = np.asarray(farm_values, dtype=np.int32)
    crop_sequences = np.asarray(crop_sequences, dtype=np.int32)
    cell_area_m2 = np.asarray(cell_area_m2, dtype=np.float64)
    farmer_areas_m2 = np.asarray(farmer_areas_m2, dtype=np.float64)

    if farm_values.ndim != 2:
        raise ValueError("farm_values must be two-dimensional.")
    if crop_sequences.ndim != 3 or crop_sequences.shape[1:] != farm_values.shape:
        raise ValueError(
            "crop_sequences must have shape (year, y, x) aligned with farm_values."
        )
    if cell_area_m2.shape != farm_values.shape:
        raise ValueError("cell_area_m2 must align with farm_values.")
    if len(target_crop_areas_per_year) != crop_sequences.shape[0]:
        raise ValueError("One crop-area target mapping is required per year.")
    if not 0.0 <= alignment_weight <= 1.0:
        raise ValueError("alignment_weight must be between 0 and 1.")
    if max_local_sequences < 1 or max_regional_sequences < 1:
        raise ValueError("Sequence candidate counts must be at least 1.")
    if regional_sequence_pool_size < 1:
        raise ValueError("regional_sequence_pool_size must be at least 1.")
    if local_search_passes < 0 or regional_search_passes < 0:
        raise ValueError("Sequence search passes cannot be negative.")
    if not 0.0 <= local_fit_threshold_pct <= 100.0:
        raise ValueError("local_fit_threshold_pct must be between 0 and 100.")
    if not 0.0 <= fallow_penalty <= 1.0:
        raise ValueError("fallow_penalty must be between 0 and 1.")

    n_farmers = farmer_areas_m2.size
    represented = np.unique(farm_values[farm_values >= 0])
    if not np.array_equal(represented, np.arange(n_farmers, dtype=np.int32)):
        raise ValueError("farm_values must contain compact farmer IDs 0..n_farmers-1.")

    farms_flat = farm_values.ravel()
    active = farms_flat >= 0
    active_sequences = np.ascontiguousarray(
        crop_sequences.reshape(crop_sequences.shape[0], -1)[:, active].T,
        dtype=np.int32,
    )
    active_areas = cell_area_m2.ravel()[active]
    active_farmers = farms_flat[active]

    complete = ~np.any(active_sequences == missing_code, axis=1)
    complete &= np.any(active_sequences > 0, axis=1)
    if not complete.any():
        raise ValueError(
            "No complete original crop sequence is available in the selected region."
        )

    complete_sequences = active_sequences[complete]
    complete_areas = active_areas[complete]
    complete_farmers = active_farmers[complete]
    sequences, cell_sequence_ids, sequence_cell_counts = np.unique(
        complete_sequences,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    sequence_area_m2 = np.bincount(
        cell_sequence_ids,
        weights=complete_areas,
        minlength=sequences.shape[0],
    ).astype(np.float64)
    fallow_fraction = np.mean(sequences == fallow_code, axis=1, dtype=np.float64)

    # Tally complete sequences within each farm using a compact unique-pair table.
    # This avoids an n_farmers x n_sequences dense matrix, which would be
    # prohibitive for continental applications.
    encoded_pairs = complete_farmers.astype(np.int64) * np.int64(
        sequences.shape[0]
    ) + cell_sequence_ids.astype(np.int64)
    unique_pairs, pair_inverse = np.unique(encoded_pairs, return_inverse=True)
    pair_support_m2 = np.bincount(
        pair_inverse,
        weights=complete_areas,
    ).astype(np.float64)
    pair_farmers = (unique_pairs // sequences.shape[0]).astype(np.int32)
    pair_sequences = (unique_pairs % sequences.shape[0]).astype(np.int32)

    # Largest local sequence is the preliminary assignment. Fallow affects later
    # preference scores, but does not change the definition of local dominance.
    raw_order = np.lexsort((pair_sequences, -pair_support_m2, pair_farmers))
    sorted_farmers = pair_farmers[raw_order]
    group_start = np.flatnonzero(np.r_[True, sorted_farmers[1:] != sorted_farmers[:-1]])
    group_lengths = np.diff(np.r_[group_start, raw_order.size])
    within_group_rank = np.arange(raw_order.size) - np.repeat(
        group_start,
        group_lengths,
    )
    keep = within_group_rank < max_local_sequences
    kept = raw_order[keep]
    kept_ranks = within_group_rank[keep].astype(np.int32)

    local_candidate_ids = np.full(
        (n_farmers, max_local_sequences),
        -1,
        dtype=np.int32,
    )
    local_candidate_support = np.zeros(
        (n_farmers, max_local_sequences),
        dtype=np.float64,
    )
    local_candidate_ids[pair_farmers[kept], kept_ranks] = pair_sequences[kept]
    local_candidate_support[pair_farmers[kept], kept_ranks] = pair_support_m2[kept]

    regional_area_commonness = np.log1p(sequence_area_m2)
    regional_count_commonness = np.log1p(sequence_cell_counts.astype(np.float64))
    regional_preference = (
        0.50 * regional_area_commonness + 0.50 * regional_count_commonness
    ) * (1.0 - fallow_penalty * fallow_fraction)
    default_sequence_id = int(np.argmax(regional_preference))
    preliminary_ids = local_candidate_ids[:, 0].copy()
    no_local = preliminary_ids < 0
    preliminary_ids[no_local] = default_sequence_id

    support_ratio = np.divide(
        local_candidate_support,
        farmer_areas_m2[:, None],
        out=np.zeros_like(local_candidate_support),
        where=farmer_areas_m2[:, None] > 0.0,
    )
    local_priors = np.zeros_like(local_candidate_support)
    valid_local = local_candidate_ids >= 0
    if valid_local.any():
        local_fallow = np.zeros_like(local_candidate_support)
        local_fallow[valid_local] = fallow_fraction[local_candidate_ids[valid_local]]
        local_priors[valid_local] = (
            0.60 + 0.40 * np.clip(support_ratio[valid_local], 0.0, 1.0)
        ) * (1.0 - fallow_penalty * local_fallow[valid_local])
        dominant = valid_local[:, 0]
        local_priors[dominant, 0] = np.minimum(
            local_priors[dominant, 0] + 0.05,
            1.0,
        )

    initial_priors = np.where(
        no_local,
        0.50 * (1.0 - fallow_penalty * fallow_fraction[preliminary_ids]),
        local_priors[:, 0],
    ).astype(np.float64)

    # Request extra regional alternatives because the most similar sequences can
    # also be locally observed alternatives. Those local duplicates are filtered
    # per farmer below so the fallback slots add genuinely non-local choices.
    expanded_fallback_count = max_regional_sequences + max_local_sequences
    _, farmer_fallback_lookup, fallback_by_preliminary, fallback_scores = (
        _regional_sequence_fallback_candidates(
            sequences,
            sequence_area_m2,
            sequence_cell_counts,
            preliminary_ids,
            target_crop_areas_per_year,
            fallow_code=fallow_code,
            fallow_penalty=fallow_penalty,
            pool_size=regional_sequence_pool_size,
            candidates_per_sequence=expanded_fallback_count,
        )
    )
    expanded_ids = fallback_by_preliminary[farmer_fallback_lookup]
    expanded_priors = fallback_scores[farmer_fallback_lookup]
    regional_candidate_ids = np.full(
        (n_farmers, max_regional_sequences),
        -1,
        dtype=np.int32,
    )
    regional_candidate_priors = np.zeros(
        (n_farmers, max_regional_sequences),
        dtype=np.float64,
    )
    regional_counts = np.zeros(n_farmers, dtype=np.int32)
    farmer_rows = np.arange(n_farmers, dtype=np.int32)
    for expanded_index in range(expanded_fallback_count):
        candidate = expanded_ids[:, expanded_index]
        is_local_candidate = np.any(
            local_candidate_ids == candidate[:, None],
            axis=1,
        )
        take = (
            (candidate >= 0)
            & ~is_local_candidate
            & (regional_counts < max_regional_sequences)
        )
        if not take.any():
            continue
        rows = farmer_rows[take]
        slots = regional_counts[take]
        regional_candidate_ids[rows, slots] = candidate[take]
        regional_candidate_priors[rows, slots] = expanded_priors[take, expanded_index]
        regional_counts[take] += 1

    candidate_ids = np.concatenate(
        [local_candidate_ids, regional_candidate_ids],
        axis=1,
    )
    candidate_priors = np.concatenate(
        [local_priors, regional_candidate_priors],
        axis=1,
    )

    category_codes = np.unique(
        np.concatenate(
            [
                sequences.ravel(),
                np.asarray(
                    [
                        int(code)
                        for targets in target_crop_areas_per_year
                        for code in targets
                        if int(code) > 0
                    ],
                    dtype=np.int32,
                ),
                np.asarray([fallow_code], dtype=np.int32),
            ]
        )
    ).astype(np.int32)
    sequence_category_indices = np.searchsorted(category_codes, sequences).astype(
        np.int32
    )
    n_years = sequences.shape[1]
    source_targets = np.zeros((n_years, category_codes.size), dtype=np.float64)
    adjusted_targets = np.zeros_like(source_targets)
    positive_target_scales = np.ones(n_years, dtype=np.float64)
    total_farmer_area_m2 = float(farmer_areas_m2.sum())
    fallow_category = int(np.searchsorted(category_codes, fallow_code))

    for year_index, targets in enumerate(target_crop_areas_per_year):
        positive_total = 0.0
        for crop_code, target_area in targets.items():
            crop_code = int(crop_code)
            target_area = float(target_area)
            if crop_code <= 0 or not np.isfinite(target_area) or target_area <= 0.0:
                continue
            category_index = int(np.searchsorted(category_codes, crop_code))
            source_targets[year_index, category_index] = target_area
            positive_total += target_area

        scale = (
            min(1.0, total_farmer_area_m2 / positive_total)
            if positive_total > 0.0
            else 1.0
        )
        positive_target_scales[year_index] = scale
        adjusted_targets[year_index] = source_targets[year_index] * scale
        adjusted_targets[year_index, fallow_category] = max(
            total_farmer_area_m2 - float(adjusted_targets[year_index].sum()),
            0.0,
        )
        source_targets[year_index, fallow_category] = adjusted_targets[
            year_index, fallow_category
        ]

    dominant_support_fraction = np.clip(support_ratio[:, 0], 0.0, 1.0)
    processing_order = np.lexsort((farmer_areas_m2, dominant_support_fraction)).astype(
        np.int32
    )

    assignments, assignment_priors, assigned_areas = (
        _optimize_multiyear_sequence_assignments_numba(
            farmer_areas_m2,
            sequence_category_indices,
            local_candidate_ids,
            local_priors,
            adjusted_targets,
            processing_order,
            preliminary_ids,
            initial_priors,
            float(alignment_weight),
            int(local_search_passes),
        )
    )
    local_only_fit_score = _multiyear_area_fit_score(
        assigned_areas,
        adjusted_targets,
    )

    # Regional candidates are a second-stage fallback, not a co-equal first
    # choice. This preserves locally observed sequences whenever their aggregate
    # area fit is already adequate.
    used_regional_fallback_stage = local_only_fit_score < local_fit_threshold_pct
    if used_regional_fallback_stage:
        assignments, assignment_priors, assigned_areas = (
            _optimize_multiyear_sequence_assignments_numba(
                farmer_areas_m2,
                sequence_category_indices,
                candidate_ids,
                candidate_priors,
                adjusted_targets,
                processing_order,
                assignments,
                assignment_priors,
                float(alignment_weight),
                int(regional_search_passes),
            )
        )

    if (assignments < 0).any() or (assignments >= sequences.shape[0]).any():
        raise RuntimeError(
            "A final assignment does not reference the original catalog."
        )
    final_sequences = sequences[assignments]
    if np.any(final_sequences == missing_code):
        raise RuntimeError(
            "A final complete farmer sequence contains missing HRL years."
        )

    final_pair_codes = np.arange(n_farmers, dtype=np.int64) * np.int64(
        sequences.shape[0]
    ) + assignments.astype(np.int64)
    pair_positions = np.searchsorted(unique_pairs, final_pair_codes)
    local_mask = pair_positions < unique_pairs.size
    local_mask[local_mask] &= (
        unique_pairs[pair_positions[local_mask]] == final_pair_codes[local_mask]
    )
    local_support_fraction = np.zeros(n_farmers, dtype=np.float64)
    local_support_fraction[local_mask] = np.divide(
        pair_support_m2[pair_positions[local_mask]],
        farmer_areas_m2[local_mask],
        out=np.zeros(np.count_nonzero(local_mask), dtype=np.float64),
        where=farmer_areas_m2[local_mask] > 0.0,
    )
    local_dominant_mask = local_mask & (assignments == preliminary_ids)
    quality_flags = np.full(
        n_farmers,
        CROP_SEQUENCE_QUALITY_REGIONAL_FALLBACK,
        dtype=np.int8,
    )
    quality_flags[local_mask] = CROP_SEQUENCE_QUALITY_LOCAL_ALTERNATIVE
    quality_flags[local_dominant_mask] = CROP_SEQUENCE_QUALITY_LOCAL_DOMINANT

    quality = pd.DataFrame(
        {
            "crop_sequence_quality_flag": quality_flags,
            "crop_sequence_is_original": np.ones(n_farmers, dtype=bool),
            "crop_sequence_is_local": local_mask,
            "crop_sequence_is_local_dominant": local_dominant_mask,
            "crop_sequence_local_support_fraction": local_support_fraction.astype(
                np.float32
            ),
            "crop_sequence_fallow_fraction": fallow_fraction[assignments].astype(
                np.float32
            ),
        }
    )

    diagnostics_rows: list[dict[str, float | int | bool]] = []
    for year_index in range(n_years):
        source_positive_total = float(
            source_targets[year_index, category_codes > 0].sum()
        )
        assigned_positive_total = float(
            assigned_areas[year_index, category_codes > 0].sum()
        )
        for category_index, crop_code in enumerate(category_codes):
            source_area = float(source_targets[year_index, category_index])
            assigned_area = float(assigned_areas[year_index, category_index])
            adjusted_area = float(adjusted_targets[year_index, category_index])
            diagnostics_rows.append(
                {
                    "year_index": year_index,
                    "crop_code": int(crop_code),
                    "source_area_m2": source_area,
                    "adjusted_target_area_m2": adjusted_area,
                    "assigned_area_m2": assigned_area,
                    "difference_from_source_m2": assigned_area - source_area,
                    "difference_from_adjusted_target_m2": (
                        assigned_area - adjusted_area
                    ),
                    "source_share": (
                        source_area / source_positive_total
                        if crop_code > 0 and source_positive_total > 0.0
                        else 0.0
                    ),
                    "assigned_share": (
                        assigned_area / assigned_positive_total
                        if crop_code > 0 and assigned_positive_total > 0.0
                        else 0.0
                    ),
                    "positive_target_scale": positive_target_scales[year_index],
                    "local_only_fit_score_pct": local_only_fit_score,
                    "regional_fallback_stage_used": used_regional_fallback_stage,
                }
            )

    return final_sequences.astype(np.int32), quality, pd.DataFrame(diagnostics_rows)


def relax_lowder_targets_for_sequence_fit(
    target_farms: list[TargetFarm],
    *,
    extra_farm_fraction: float,
    n_available_cells: int,
    mean_cell_area_m2: float,
    minimum_cells_per_farm: float = 1.0,
) -> list[TargetFarm]:
    """Split a controlled fraction of the largest Lowder target farms.

    Multi-year crop-area matching is an indivisible assignment problem because a
    farmer receives one complete sequence and contributes its whole area to that
    sequence. A small number of additional, smaller farms improves the available
    assignment granularity. This helper therefore splits the largest eligible
    Lowder targets in half while preserving their combined area exactly.

    Args:
        target_farms: Original Lowder-derived target farms.
        extra_farm_fraction: Maximum additional farm count as a fraction of the
            original count. ``0`` leaves the Lowder count unchanged.
        n_available_cells: Number of selected agricultural cells; every final farm
            must receive at least one cell.
        mean_cell_area_m2: Mean selected model-cell area.
        minimum_cells_per_farm: Minimum expected cells represented by each split
            target.

    Returns:
        New target list with the same total target area and up to the requested
        additional number of farms.

    Raises:
        RuntimeError: If splitting changes the total target area.
        ValueError: If parameters are invalid.
    """
    if not target_farms:
        raise ValueError("target_farms must contain at least one target.")
    if extra_farm_fraction < 0.0:
        raise ValueError("extra_farm_fraction cannot be negative.")
    if n_available_cells < len(target_farms):
        raise ValueError("n_available_cells cannot be smaller than the farm count.")
    if mean_cell_area_m2 <= 0.0 or minimum_cells_per_farm <= 0.0:
        raise ValueError("Cell-area and minimum-cell parameters must be positive.")
    if extra_farm_fraction == 0.0 or len(target_farms) >= n_available_cells:
        return list(target_farms)

    requested_extra = int(round(len(target_farms) * extra_farm_fraction))
    requested_extra = min(requested_extra, n_available_cells - len(target_farms))
    if requested_extra <= 0:
        return list(target_farms)

    minimum_split_target_area = 2.0 * mean_cell_area_m2 * minimum_cells_per_farm
    target_areas = np.asarray(
        [target.target_area_m2 for target in target_farms],
        dtype=np.float64,
    )
    eligible = np.flatnonzero(target_areas >= minimum_split_target_area)
    if eligible.size == 0:
        return list(target_farms)

    n_split = min(requested_extra, int(eligible.size))
    if n_split < eligible.size:
        selected_local = np.argpartition(
            -target_areas[eligible],
            n_split - 1,
        )[:n_split]
        split_indices = set(eligible[selected_local].tolist())
    else:
        split_indices = set(eligible.tolist())

    relaxed: list[TargetFarm] = []
    for index, target in enumerate(target_farms):
        if index not in split_indices:
            relaxed.append(target)
            continue
        first_area = float(target.target_area_m2) / 2.0
        second_area = float(target.target_area_m2) - first_area
        relaxed.append(
            TargetFarm(target_area_m2=first_area, size_class=target.size_class)
        )
        relaxed.append(
            TargetFarm(target_area_m2=second_area, size_class=target.size_class)
        )

    if not math.isclose(
        sum(target.target_area_m2 for target in relaxed),
        sum(target.target_area_m2 for target in target_farms),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError("Relaxing Lowder targets changed their total area.")

    return relaxed


def grow_farms_from_raster_cells(
    cultivated_mask: np.ndarray,
    crop_sequences: np.ndarray,
    cell_area_m2: np.ndarray,
    target_farms: list[TargetFarm],
    *,
    random_seed: int = 42,
    distance_weight: float = 0.45,
    crop_sequence_weight: float = 0.40,
    switch_timing_weight: float = 0.15,
    min_valid_crop_sequence_overlap: int = 2,
    jump_candidate_sample: int = 256,
    max_jump_distance_m: float = 2_000.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Grow Lowder-guided farm geometry on the final model grid.

    Each selected model cell is indivisible. Farms grow through four-connected
    neighbours until their exact integer target cell count is reached. Candidate
    cells are scored against the seed cell using spatial compactness, complete
    crop-sequence similarity, and crop-switch timing. If a connected patch is
    exhausted, the same farmer may continue in a nearby disconnected parcel.

    This function creates geometry and farm-size metadata only. Complete farmer
    crop sequences are assigned afterward by the joint multi-year optimizer, so
    no redundant year-by-year crop labels are constructed here.

    Args:
        cultivated_mask: Boolean static agricultural mask on the final grid.
        crop_sequences: Combined HRL codes with shape ``(year, y, x)``; ``-1``
            denotes fallow and ``-2`` denotes outside or missing observations.
        cell_area_m2: Model-cell areas in square metres.
        target_farms: Lowder-derived, possibly relaxed target farms.
        random_seed: Seed controlling deterministic farm seed placement.
        distance_weight: Candidate-score weight for spatial compactness.
        crop_sequence_weight: Candidate-score weight for complete-sequence
            similarity.
        switch_timing_weight: Candidate-score weight for crop-switch timing.
        min_valid_crop_sequence_overlap: Minimum comparable years required for a
            positive sequence-similarity contribution.
        jump_candidate_sample: Unassigned cells sampled when another parcel is
            required.
        max_jump_distance_m: Preferred maximum distance to a new parcel.

    Returns:
        A compact local farm-ID raster and a farmer table containing target area,
        achieved area, Lowder size class, cell count, and connected parcel count.

    Raises:
        RuntimeError: If farm IDs are not compact or selected cells are not
            assigned exactly once.
        ValueError: If inputs are inconsistent, no selected cells exist, or the
            score weights are invalid.
    """
    cultivated_mask = np.asarray(cultivated_mask, dtype=bool)
    crop_sequences = np.asarray(crop_sequences, dtype=np.int32)
    cell_area_m2 = np.asarray(cell_area_m2, dtype=np.float64)

    if cultivated_mask.ndim != 2:
        raise ValueError("cultivated_mask must be two-dimensional.")
    if crop_sequences.ndim != 3 or crop_sequences.shape[1:] != cultivated_mask.shape:
        raise ValueError(
            "crop_sequences must have shape (year, y, x) aligned with cultivated_mask."
        )
    if cell_area_m2.shape != cultivated_mask.shape:
        raise ValueError("cell_area_m2 must align with cultivated_mask.")

    weight_sum = distance_weight + crop_sequence_weight + switch_timing_weight
    if weight_sum <= 0.0:
        raise ValueError("Farm-growing score weights must sum to a positive value.")
    distance_weight /= weight_sum
    crop_sequence_weight /= weight_sum
    switch_timing_weight /= weight_sum

    active_indices = np.flatnonzero(cultivated_mask).astype(np.int32)
    if active_indices.size == 0:
        raise ValueError(
            "No cultivated model cells are available for farm construction."
        )

    ordered_targets, target_cell_counts = _target_cell_counts_from_areas(
        target_farms,
        int(active_indices.size),
    )

    rng = np.random.default_rng(random_seed)
    seed_order = active_indices.copy()
    rng.shuffle(seed_order)

    mean_cell_area_m2 = float(cell_area_m2[cultivated_mask].mean())
    mean_cell_length_m = max(math.sqrt(mean_cell_area_m2), 1.0)
    max_jump_distance_cells = max_jump_distance_m / mean_cell_length_m

    farm_values = _grow_raster_farms_numba(
        cultivated_mask,
        crop_sequences,
        target_cell_counts,
        seed_order,
        float(distance_weight),
        float(crop_sequence_weight),
        float(switch_timing_weight),
        int(min_valid_crop_sequence_overlap),
        int(max(jump_candidate_sample, 1)),
        float(max_jump_distance_cells),
    )

    farm_values[~cultivated_mask] = -1
    represented = np.unique(farm_values[farm_values >= 0])
    expected = np.arange(len(ordered_targets), dtype=np.int32)
    if not np.array_equal(represented, expected):
        raise RuntimeError(
            "Raster farm IDs are not compact or a target farm disappeared."
        )
    if not ((farm_values >= 0) == cultivated_mask).all():
        raise RuntimeError(
            "Cultivated cells and assigned farm cells do not match exactly."
        )

    flat_farms = farm_values[cultivated_mask]
    actual_areas_m2 = np.bincount(
        flat_farms,
        weights=cell_area_m2[cultivated_mask],
        minlength=len(ordered_targets),
    )
    n_cells = np.bincount(flat_farms, minlength=len(ordered_targets)).astype(np.int32)
    n_parcels = _count_farm_components_numba(farm_values, len(ordered_targets))

    farmers = pd.DataFrame(
        {
            "farmer_id": np.arange(len(ordered_targets), dtype=np.int32),
            "target_area_m2": np.asarray(
                [target.target_area_m2 for target in ordered_targets],
                dtype=np.float64,
            ),
            "area_m2": actual_areas_m2.astype(np.float64),
            "area_ha": actual_areas_m2.astype(np.float64) / 10_000.0,
            "size_class": [target.size_class for target in ordered_targets],
            "n_cells": n_cells,
            "n_fields": n_parcels,
        }
    )
    return farm_values.astype(np.int32, copy=False), farmers


def _allocate_farm_counts_to_sequence_groups(
    group_cell_counts: np.ndarray,
    requested_n_farms: int,
) -> np.ndarray:
    """Allocate a regional farm count over exact-sequence groups.

    Every non-empty sequence group receives at least one farm. The Lowder-derived
    count is increased when required because one farm cannot contain multiple exact
    sequences. Counts remain proportional to group area where possible and cannot
    exceed one farm per cell.

    Args:
        group_cell_counts: Number of selected model cells in each sequence group.
        requested_n_farms: Lowder-derived number of farms for the region.

    Returns:
        Positive farm counts per sequence group. Their sum is the feasible maximum
        of the Lowder request and number of sequence groups.

    Raises:
        RuntimeError: If the requested total cannot be allocated over groups.
        ValueError: If group counts or the requested farm count are invalid.
    """
    group_cell_counts = np.asarray(group_cell_counts, dtype=np.int64)
    if group_cell_counts.ndim != 1 or group_cell_counts.size == 0:
        raise ValueError("group_cell_counts must be a non-empty 1D array.")
    if (group_cell_counts <= 0).any():
        raise ValueError("Every exact-sequence group must contain at least one cell.")

    n_groups = int(group_cell_counts.size)
    n_cells = int(group_cell_counts.sum())
    total_n_farms = min(max(int(requested_n_farms), n_groups), n_cells)

    raw = group_cell_counts.astype(np.float64) / n_cells * total_n_farms
    counts = np.floor(raw).astype(np.int64)
    counts = np.maximum(counts, 1)
    counts = np.minimum(counts, group_cell_counts)

    while int(counts.sum()) < total_n_farms:
        available = counts < group_cell_counts
        if not available.any():
            raise RuntimeError("No sequence group has capacity for another farm.")
        priority = raw - counts
        priority[~available] = -np.inf
        best = int(np.argmax(priority))
        if not np.isfinite(priority[best]):
            # All proportional remainders are exhausted. Split the group with
            # the currently largest mean number of cells per farm.
            mean_cells = np.divide(
                group_cell_counts,
                counts,
                out=np.zeros_like(raw),
                where=counts > 0,
            )
            mean_cells[~available] = -np.inf
            best = int(np.argmax(mean_cells))
        counts[best] += 1

    while int(counts.sum()) > total_n_farms:
        removable = counts > 1
        if not removable.any():
            raise RuntimeError("Cannot reduce sequence-group farm counts further.")
        priority = counts - raw
        priority[~removable] = -np.inf
        counts[int(np.argmax(priority))] -= 1

    return counts.astype(np.int32)


def _prepare_exact_sequence_targets(
    target_farms: list[TargetFarm],
    group_cell_counts: np.ndarray,
) -> tuple[list[TargetFarm], np.ndarray, np.ndarray]:
    """Allocate Lowder targets to exact crop-sequence groups.

    Lowder remains a soft size prior. Additional targets are created when the number
    of exact sequences exceeds the Lowder holding count. Within every sequence group,
    integer target counts exactly exhaust the available cells.

    Args:
        target_farms: Regional Lowder-derived target farms.
        group_cell_counts: Number of cells in every exact-sequence group.

    Returns:
        Ordered target farms, exact target cell counts, and the sequence-group ID of
        every target farm.

    Raises:
        ValueError: If targets are absent or sequence-group counts are inconsistent.
    """
    if not target_farms:
        raise ValueError("target_farms must contain at least one target.")

    group_cell_counts = np.asarray(group_cell_counts, dtype=np.int32)
    farms_per_group = _allocate_farm_counts_to_sequence_groups(
        group_cell_counts,
        len(target_farms),
    )
    required_n_farms = int(farms_per_group.sum())

    target_pool = sorted(
        list(target_farms),
        key=lambda target: target.target_area_m2,
        reverse=True,
    )
    if required_n_farms > len(target_pool):
        representative_target = target_pool[len(target_pool) // 2]
        target_pool.extend(
            TargetFarm(
                target_area_m2=float(representative_target.target_area_m2),
                size_class=str(representative_target.size_class),
            )
            for _ in range(required_n_farms - len(target_pool))
        )
    elif required_n_farms < len(target_pool):
        # This can only occur when Lowder requests more farms than there are
        # cultivated cells. Retain evenly spaced target quantiles rather than
        # keeping only one end of the size distribution.
        selected_indices = (
            np.linspace(
                0,
                len(target_pool) - 1,
                required_n_farms,
            )
            .round()
            .astype(np.int32)
        )
        target_pool = [target_pool[index] for index in selected_indices]

    # Pair larger Lowder targets with sequence groups that require larger farms
    # on average. Exact group totals are subsequently enforced in cell units.
    slot_groups = np.repeat(
        np.arange(group_cell_counts.size, dtype=np.int32),
        farms_per_group,
    )
    mean_cells_per_slot = group_cell_counts[slot_groups] / farms_per_group[slot_groups]
    slot_order = np.lexsort((slot_groups, -mean_cells_per_slot))
    slot_groups = slot_groups[slot_order]

    targets_by_group: list[list[TargetFarm]] = [
        [] for _ in range(group_cell_counts.size)
    ]
    for target, group_id in zip(target_pool, slot_groups, strict=True):
        targets_by_group[int(group_id)].append(target)

    ordered_targets: list[TargetFarm] = []
    target_cell_counts: list[np.ndarray] = []
    farm_sequence_groups: list[np.ndarray] = []
    for group_id, group_targets in enumerate(targets_by_group):
        group_ordered_targets, group_counts = _target_cell_counts_from_areas(
            group_targets,
            int(group_cell_counts[group_id]),
        )
        ordered_targets.extend(group_ordered_targets)
        target_cell_counts.append(group_counts)
        farm_sequence_groups.append(
            np.full(len(group_ordered_targets), group_id, dtype=np.int32)
        )

    return (
        ordered_targets,
        np.concatenate(target_cell_counts).astype(np.int32, copy=False),
        np.concatenate(farm_sequence_groups).astype(np.int32, copy=False),
    )


@njit(cache=True)
def _add_exact_sequence_frontier_neighbors(
    flat_index: int,
    required_sequence_group: int,
    n_rows: int,
    n_cols: int,
    cultivated_flat: np.ndarray,
    sequence_groups_flat: np.ndarray,
    assignments_flat: np.ndarray,
    frontier: np.ndarray,
    frontier_size: int,
    frontier_generation: np.ndarray,
    generation: int,
) -> int:
    """Add free four-neighbours carrying the required exact sequence.

    Args:
        flat_index: Flat index of the cell whose neighbours are inspected.
        required_sequence_group: Exact-sequence group required by the current farm.
        n_rows: Number of raster rows.
        n_cols: Number of raster columns.
        cultivated_flat: Flat boolean agricultural mask.
        sequence_groups_flat: Exact-sequence group per flat cell.
        assignments_flat: Current farmer assignment per flat cell.
        frontier: Preallocated frontier indices.
        frontier_size: Number of valid frontier entries.
        frontier_generation: Generation marker per cell.
        generation: Marker for the current farm.

    Returns:
        Updated number of valid frontier entries.
    """
    row = flat_index // n_cols
    col = flat_index - row * n_cols

    if row > 0:
        neighbor = flat_index - n_cols
        if (
            cultivated_flat[neighbor]
            and sequence_groups_flat[neighbor] == required_sequence_group
            and assignments_flat[neighbor] < 0
            and frontier_generation[neighbor] != generation
        ):
            frontier[frontier_size] = neighbor
            frontier_size += 1
            frontier_generation[neighbor] = generation

    if row + 1 < n_rows:
        neighbor = flat_index + n_cols
        if (
            cultivated_flat[neighbor]
            and sequence_groups_flat[neighbor] == required_sequence_group
            and assignments_flat[neighbor] < 0
            and frontier_generation[neighbor] != generation
        ):
            frontier[frontier_size] = neighbor
            frontier_size += 1
            frontier_generation[neighbor] = generation

    if col > 0:
        neighbor = flat_index - 1
        if (
            cultivated_flat[neighbor]
            and sequence_groups_flat[neighbor] == required_sequence_group
            and assignments_flat[neighbor] < 0
            and frontier_generation[neighbor] != generation
        ):
            frontier[frontier_size] = neighbor
            frontier_size += 1
            frontier_generation[neighbor] = generation

    if col + 1 < n_cols:
        neighbor = flat_index + 1
        if (
            cultivated_flat[neighbor]
            and sequence_groups_flat[neighbor] == required_sequence_group
            and assignments_flat[neighbor] < 0
            and frontier_generation[neighbor] != generation
        ):
            frontier[frontier_size] = neighbor
            frontier_size += 1
            frontier_generation[neighbor] = generation

    return frontier_size


@njit(cache=True)
def _exact_sequence_distance_score(
    candidate: int,
    seed: int,
    n_cols: int,
    target_size_cells: int,
) -> float:
    """Calculate compactness within an exact-sequence farm.

    Args:
        candidate: Flat candidate-cell index.
        seed: Flat seed-cell index.
        n_cols: Number of raster columns.
        target_size_cells: Target farm size in cells.

    Returns:
        Compactness score in the interval ``(0, 1]``.
    """
    candidate_row = candidate // n_cols
    candidate_col = candidate - candidate_row * n_cols
    seed_row = seed // n_cols
    seed_col = seed - seed_row * n_cols
    delta_row = candidate_row - seed_row
    delta_col = candidate_col - seed_col
    distance_cells = math.sqrt(delta_row * delta_row + delta_col * delta_col)
    target_radius_cells = max(math.sqrt(target_size_cells / math.pi), 1.0)
    return 1.0 / (1.0 + distance_cells / target_radius_cells)


@njit(cache=True)
def _grow_raster_farms_exact_sequences_numba(
    cultivated_mask: np.ndarray,
    sequence_groups: np.ndarray,
    target_cell_counts: np.ndarray,
    farm_sequence_groups: np.ndarray,
    grouped_seed_order: np.ndarray,
    group_offsets: np.ndarray,
    jump_candidate_sample: int,
    jump_distance_scale_cells: float,
) -> np.ndarray:
    """Assign cells without mixing complete crop-sequence groups.

    Each farm receives cells only from its required exact-sequence group. Connected
    cells are preferred; if the component is exhausted, another same-sequence cell
    starts a disconnected parcel.

    Args:
        cultivated_mask: Boolean exact-sequence agricultural union.
        sequence_groups: Exact-sequence group ID per selected cell.
        target_cell_counts: Exact positive target cell count per farmer.
        farm_sequence_groups: Required exact-sequence group per farmer.
        grouped_seed_order: Selected flat indices grouped by sequence.
        group_offsets: Start and end offsets of every sequence group.
        jump_candidate_sample: Same-sequence cells sampled for a new parcel.
        jump_distance_scale_cells: Distance scale for parcel-jump preference.

    Returns:
        Two-dimensional compact local farmer-ID raster.

    Raises:
        RuntimeError: If a sequence group is exhausted before all of its farm targets
            can be filled.
    """
    n_rows, n_cols = cultivated_mask.shape
    n_total = n_rows * n_cols
    cultivated_flat = cultivated_mask.ravel()
    sequence_groups_flat = sequence_groups.ravel()
    assignments_flat = np.full(n_total, -1, dtype=np.int32)

    frontier = np.empty(n_total, dtype=np.int32)
    frontier_generation = np.zeros(n_total, dtype=np.int32)
    generation = 0
    group_cursors = group_offsets[:-1].copy()
    distance_scale = max(jump_distance_scale_cells, 1.0)

    for farmer_id in range(target_cell_counts.size):
        required_group = int(farm_sequence_groups[farmer_id])
        cursor = int(group_cursors[required_group])
        group_end = int(group_offsets[required_group + 1])
        while cursor < group_end and assignments_flat[grouped_seed_order[cursor]] >= 0:
            cursor += 1
        if cursor >= group_end:
            raise RuntimeError("No exact-sequence seed remains for a target farm.")

        seed = int(grouped_seed_order[cursor])
        group_cursors[required_group] = cursor + 1
        assignments_flat[seed] = farmer_id
        assigned_count = 1
        target_size = int(target_cell_counts[farmer_id])

        generation += 1
        frontier_size = 0
        frontier_size = _add_exact_sequence_frontier_neighbors(
            seed,
            required_group,
            n_rows,
            n_cols,
            cultivated_flat,
            sequence_groups_flat,
            assignments_flat,
            frontier,
            frontier_size,
            frontier_generation,
            generation,
        )

        while assigned_count < target_size:
            best_position = -1
            best_candidate = -1
            best_score = -1.0e30
            position = 0
            while position < frontier_size:
                candidate = int(frontier[position])
                if assignments_flat[candidate] >= 0:
                    frontier_size -= 1
                    frontier[position] = frontier[frontier_size]
                    continue
                score = _exact_sequence_distance_score(
                    candidate,
                    seed,
                    n_cols,
                    target_size,
                )
                if score > best_score or (
                    score == best_score and candidate < best_candidate
                ):
                    best_score = score
                    best_candidate = candidate
                    best_position = position
                position += 1

            if best_candidate < 0:
                # Connected cells with this exact sequence are exhausted. Search
                # only within the same sequence group. Distance is deliberately a
                # soft preference rather than a hard exclusion.
                sampled = 0
                scan = int(group_cursors[required_group])
                best_jump = -1
                best_jump_score = -1.0e30
                seed_row = seed // n_cols
                seed_col = seed - seed_row * n_cols
                while scan < group_end and sampled < jump_candidate_sample:
                    candidate = int(grouped_seed_order[scan])
                    scan += 1
                    if assignments_flat[candidate] >= 0:
                        continue
                    sampled += 1
                    candidate_row = candidate // n_cols
                    candidate_col = candidate - candidate_row * n_cols
                    delta_row = candidate_row - seed_row
                    delta_col = candidate_col - seed_col
                    distance_cells = math.sqrt(
                        delta_row * delta_row + delta_col * delta_col
                    )
                    jump_score = 1.0 / (1.0 + distance_cells / distance_scale)
                    if jump_score > best_jump_score or (
                        jump_score == best_jump_score and candidate < best_jump
                    ):
                        best_jump_score = jump_score
                        best_jump = candidate

                if best_jump < 0:
                    scan = int(group_offsets[required_group])
                    while scan < group_end:
                        candidate = int(grouped_seed_order[scan])
                        if assignments_flat[candidate] < 0:
                            best_jump = candidate
                            break
                        scan += 1

                if best_jump < 0:
                    raise RuntimeError(
                        "The exact-sequence group was exhausted before its farm "
                        "targets were filled."
                    )
                best_candidate = best_jump
            else:
                frontier_size -= 1
                frontier[best_position] = frontier[frontier_size]

            assignments_flat[best_candidate] = farmer_id
            assigned_count += 1
            frontier_size = _add_exact_sequence_frontier_neighbors(
                best_candidate,
                required_group,
                n_rows,
                n_cols,
                cultivated_flat,
                sequence_groups_flat,
                assignments_flat,
                frontier,
                frontier_size,
                frontier_generation,
                generation,
            )

    return assignments_flat.reshape((n_rows, n_cols))


def grow_farms_from_exact_crop_sequences(
    cultivated_mask: np.ndarray,
    crop_sequences: np.ndarray,
    cell_area_m2: np.ndarray,
    target_farms: list[TargetFarm],
    *,
    crop_columns: list[str],
    random_seed: int = 42,
    jump_candidate_sample: int = 1_024,
    jump_distance_scale_m: float = 10_000.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create sequence-homogeneous farms with Lowder as a soft prior.

    The complete primary-plus-secondary sequence is a hard constraint. All cells of
    a farm carry the same observed sequence, including genuine fallow years. Missing
    values are excluded from the exact-sequence domain. Lowder controls the preferred
    number and relative sizes of farms but cannot override sequence homogeneity.
    Disconnected same-sequence parcels may belong to one farmer.

    Args:
        cultivated_mask: Static selected agricultural mask.
        crop_sequences: Combined HRL crop codes with shape ``(year, y, x)``.
        cell_area_m2: Model-cell areas in square metres.
        target_farms: Lowder-derived regional target farms.
        crop_columns: Farmer-table crop columns matching the year dimension.
        random_seed: Deterministic seed for within-sequence cell ordering.
        jump_candidate_sample: Same-sequence cells considered when a connected
            component is exhausted.
        jump_distance_scale_m: Distance scale used to prefer nearby disconnected
            parcels.

    Returns:
        Compact farm raster and farmer table. Each farmer contains one complete
        sequence observed on the selected model grid.

    Raises:
        RuntimeError: If growth violates compact IDs, exact cell coverage, sequence
            homogeneity, or total target counts.
        ValueError: If arrays, crop columns, targets, or jump settings are invalid.
    """
    cultivated_mask = np.asarray(cultivated_mask, dtype=bool)
    crop_sequences = np.asarray(crop_sequences, dtype=np.int32)
    cell_area_m2 = np.asarray(cell_area_m2, dtype=np.float64)

    if cultivated_mask.ndim != 2:
        raise ValueError("cultivated_mask must be two-dimensional.")
    if crop_sequences.ndim != 3 or crop_sequences.shape[1:] != cultivated_mask.shape:
        raise ValueError(
            "crop_sequences must have shape (year, y, x) aligned with cultivated_mask."
        )
    if cell_area_m2.shape != cultivated_mask.shape:
        raise ValueError("cell_area_m2 must align with cultivated_mask.")
    if len(crop_columns) != crop_sequences.shape[0]:
        raise ValueError("crop_columns must match the crop_sequences year dimension.")
    if jump_candidate_sample < 1:
        raise ValueError("jump_candidate_sample must be at least 1.")
    if jump_distance_scale_m <= 0:
        raise ValueError("jump_distance_scale_m must be positive.")

    active_indices = np.flatnonzero(cultivated_mask).astype(np.int32)
    if active_indices.size == 0:
        raise ValueError("No cultivated model cells are available.")

    active_sequences = np.ascontiguousarray(
        crop_sequences[:, cultivated_mask].T,
        dtype=np.int32,
    )
    if np.any(active_sequences == _HRL_MISSING_CROP_CODE):
        raise ValueError(
            "cultivated_mask contains native HRL outside/missing states (-2). "
            "These cells must be excluded rather than interpreted as fallow."
        )
    if np.any(np.all(active_sequences == _HRL_FALLOW_CROP_CODE, axis=1)):
        raise ValueError(
            "cultivated_mask contains cells that are fallow in every requested "
            "year; cells without any observed crop are not fields."
        )
    unique_sequences, inverse, group_cell_counts = np.unique(
        active_sequences,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    inverse = inverse.astype(np.int32, copy=False)
    group_cell_counts = group_cell_counts.astype(np.int32, copy=False)

    ordered_targets, target_cell_counts, farm_sequence_groups = (
        _prepare_exact_sequence_targets(
            target_farms,
            group_cell_counts,
        )
    )

    sequence_groups = np.full(cultivated_mask.shape, -1, dtype=np.int32)
    sequence_groups[cultivated_mask] = inverse

    rng = np.random.default_rng(random_seed)
    random_keys = rng.random(active_indices.size)
    grouped_order = np.lexsort((random_keys, inverse))
    grouped_seed_order = active_indices[grouped_order].astype(np.int32, copy=False)
    group_offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(group_cell_counts, dtype=np.int64),
        )
    )

    mean_cell_area_m2 = float(cell_area_m2[cultivated_mask].mean())
    mean_cell_length_m = max(math.sqrt(mean_cell_area_m2), 1.0)
    jump_distance_scale_cells = jump_distance_scale_m / mean_cell_length_m

    farm_values = _grow_raster_farms_exact_sequences_numba(
        cultivated_mask,
        sequence_groups,
        target_cell_counts,
        farm_sequence_groups,
        grouped_seed_order,
        group_offsets,
        int(jump_candidate_sample),
        float(jump_distance_scale_cells),
    )
    farm_values[~cultivated_mask] = -1

    represented = np.unique(farm_values[farm_values >= 0])
    expected = np.arange(len(ordered_targets), dtype=np.int32)
    if not np.array_equal(represented, expected):
        raise RuntimeError("Exact-sequence farm IDs are not compact.")
    if not ((farm_values >= 0) == cultivated_mask).all():
        raise RuntimeError("Every selected cultivated cell must have one farm ID.")

    flat_farms = farm_values[cultivated_mask]
    actual_areas_m2 = np.bincount(
        flat_farms,
        weights=cell_area_m2[cultivated_mask],
        minlength=len(ordered_targets),
    )
    n_cells = np.bincount(
        flat_farms,
        minlength=len(ordered_targets),
    ).astype(np.int32)
    n_parcels = _count_farm_components_numba(
        farm_values,
        len(ordered_targets),
    )

    farmer_sequences = unique_sequences[farm_sequence_groups]
    farmers = pd.DataFrame(
        {
            "farmer_id": np.arange(len(ordered_targets), dtype=np.int32),
            "target_area_m2": np.asarray(
                [target.target_area_m2 for target in ordered_targets],
                dtype=np.float64,
            ),
            "area_m2": actual_areas_m2.astype(np.float64),
            "area_ha": actual_areas_m2.astype(np.float64) / 10_000.0,
            "size_class": [target.size_class for target in ordered_targets],
            "n_cells": n_cells,
            "n_fields": n_parcels,
            "sequence_group_id": farm_sequence_groups.astype(np.int32),
        }
    )
    for year_index, crop_column in enumerate(crop_columns):
        farmers[crop_column] = farmer_sequences[:, year_index].astype(np.int32)

    # Efficient defensive validation of the hard constraint: all cells of one
    # farmer must map to the same exact-sequence group.
    active_farm_ids = farm_values[cultivated_mask].astype(np.int32, copy=False)
    active_group_ids = sequence_groups[cultivated_mask].astype(np.int32, copy=False)
    minimum_group = np.full(len(farmers), np.iinfo(np.int32).max, dtype=np.int32)
    maximum_group = np.full(len(farmers), -1, dtype=np.int32)
    np.minimum.at(minimum_group, active_farm_ids, active_group_ids)
    np.maximum.at(maximum_group, active_farm_ids, active_group_ids)
    if not np.array_equal(minimum_group, maximum_group):
        raise RuntimeError("At least one farmer contains multiple crop sequences.")
    if not np.array_equal(minimum_group, farm_sequence_groups):
        raise RuntimeError("Farmer sequence labels do not match assigned cells.")

    return farm_values.astype(np.int32, copy=False), farmers


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
    cultivated_area_m2: float,
) -> pd.Series:
    """Scale Lowder farm counts to the generated cultivated area.

    Args:
        region_farm_sizes: Lowder data for one ISO3 code.
        size_class_boundaries: Size-class boundaries in square metres.
        cultivated_area_m2: Generated cultivated raster area in the region.

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

    scale_factor = cultivated_area_m2 / database_area_m2

    expected_n_farms = lowder["lowder_n_farms"] * scale_factor
    return expected_n_farms.reindex(size_class_boundaries.keys()).fillna(0)


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
    cultivated_area_m2: float,
    iso3: str,
    logger: logging.Logger,
    *,
    random_seed: int = 42,
    minimum_cells_per_farm: float = 1.0,
    mean_cell_area_m2: float | None = None,
) -> list[TargetFarm]:
    """Create target farm areas from Lowder-style farm-size statistics.

    This function converts country-level Lowder-style farm-size statistics into
    a list of target farm areas for the selected region. It first estimates the
    representative farm area in each size class, then scales the number of farms
    to match the cultivated raster area available in the region.
    The resulting target farms are later used by
    the raster farm-growing workflows to group model cells into synthetic farms.
    A small deterministic lognormal perturbation is applied
    within each size class so that farms from the same class do not all have
    identical target areas.
    Args:
        region_farm_sizes: Lowder-style farm-size data for one ISO3 code. Must
            contain one row for ``"Holdings"`` and one row for
            ``"Agricultural area (Ha)"``.
        size_class_boundaries: Mapping from Lowder size-class labels to lower
            and upper area boundaries in square metres.
        cultivated_area_m2: Total cultivated raster area in the selected
            region, represented by the selected model cells.
        iso3: ISO3 country code used in warning and error messages.
        logger: Logger used to report missing, clipped, or adjusted farm-size
            statistics.
        random_seed: Seed used for deterministic variation in target farm areas.
        minimum_cells_per_farm: Minimum expected number of model cells per
            synthetic farm. Used only when ``mean_cell_area_m2`` is provided.
        mean_cell_area_m2: Mean model-cell area in the selected region. If provided,
            the target number of farms is reduced when the Lowder-scaled farm
            count would imply fewer represented cells than farms.
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

    scale_factor = cultivated_area_m2 / database_total_area_m2

    farm_statistics["expected_n_farms"] = (
        farm_statistics["n_holdings_database"] * scale_factor
    )

    expected_total_n_farms = int(round(farm_statistics["expected_n_farms"].sum()))
    expected_total_n_farms = max(expected_total_n_farms, 1)

    if mean_cell_area_m2 is not None:
        max_reasonable_n_farms = int(
            cultivated_area_m2 / (mean_cell_area_m2 * minimum_cells_per_farm)
        )
        max_reasonable_n_farms = max(max_reasonable_n_farms, 1)

        if expected_total_n_farms > max_reasonable_n_farms:
            logger.warning(
                "Lowder implies %s farms, but the selected raster cells only support "
                "about %s farms under the current minimum_cells_per_farm setting. "
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
                cultivated_area_m2=float(farmers_region[area_column].sum()),
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
