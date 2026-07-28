"""Workflows for constructing farmer distributions and farm maps."""

from __future__ import annotations

import logging
import math
import re
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import rasterio.enums
import rasterio.features
import rasterio.vrt
import rasterio.windows
from rasterio.merge import merge as rasterio_merge
from numba import njit
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry

from geb.geb_types import ArrayInt32, TwoDArrayBool, TwoDArrayInt32
from geb.workflows.raster import pixels_to_coords

_HRL_FALLOW_CROP_CODE = -1
_HRL_MISSING_CROP_CODE = -2
_HRL_NO_CROPLAND_CODE = 0
_HRL_OUTSIDE_AREA_CODE = 65535

_HRL_CTY_CONFIDENCE_NO_CROPLAND = 253
_HRL_CTY_CONFIDENCE_OUTSIDE_AREA = 255

HRL_ANNUAL_CTY_CLASS_CODES = (
    1110,
    1120,
    1130,
    1140,
    1150,
    1210,
    1220,
    1310,
    1320,
    1410,
    1420,
    1430,
    1440,
    3100,
)
HRL_PERMANENT_CTY_CLASS_CODES = (2100, 2200, 2310, 2320, 3200)
HRL_EXACT_PERMANENT_CTY_CLASS_CODES = (2100, 2200, 2310, 2320)

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


# =============================================================================
# AlphaEarth + HRL annual remote-sensing classification helpers
# =============================================================================

ALPHAEARTH_NODATA_VALUE = -128
ALPHAEARTH_EMBEDDING_BANDS = tuple(f"A{band:02d}" for band in range(64))
HRL_CTY_CLASS_CODES = (
    0,
    1110,
    1120,
    1130,
    1140,
    1150,
    1210,
    1220,
    1310,
    1320,
    1410,
    1420,
    1430,
    1440,
    2100,
    2200,
    2310,
    2320,
    3100,
    3200,
)
HRL_CTY_CLASS_NAMES = {
    0: "No cropland",
    1110: "Wheat",
    1120: "Barley",
    1130: "Maize",
    1140: "Rice",
    1150: "Other cereals",
    1210: "Fresh vegetables",
    1220: "Dry pulses",
    1310: "Potatoes",
    1320: "Sugar beet",
    1410: "Sunflower",
    1420: "Soybeans",
    1430: "Rapeseed",
    1440: "Flax, cotton and hemp",
    2100: "Grapes",
    2200: "Olives",
    2310: "Fruits",
    2320: "Nuts",
    3100: "Unclassified annual crop",
    3200: "Unclassified permanent crop",
}

# Crop-group aggregation used by the official HRL Croplands verification.
# The codes correspond to Table 8-3 of the Product User Manual.
HRL_CTY_CROP_GROUP_CLASS_CODES = (0, 11, 12, 13, 14, 20, 30)
HRL_CTY_CROP_GROUP_MAP = {
    0: 0,
    1110: 11,
    1120: 11,
    1130: 11,
    1140: 11,
    1150: 11,
    1210: 12,
    1220: 12,
    1310: 13,
    1320: 13,
    1410: 14,
    1420: 14,
    1430: 14,
    1440: 14,
    2100: 20,
    2200: 20,
    2310: 20,
    2320: 20,
    3100: 30,
    3200: 30,
}
HRL_CTY_CROP_GROUP_NAMES = {
    0: "No cropland",
    11: "Cereals",
    12: "Dry pulses & vegetables",
    13: "Root/tuber crops",
    14: "Non-permanent industrial crops",
    20: "Permanent crops",
    30: "Unclassified crop",
}
HRL_CTY_AGGREGATION_LEVEL_1_CLASS_CODES = (
    11,
    1130,
    1140,
    12,
    1320,
    1410,
    1420,
    1430,
    1440,
    20,
)
HRL_CTY_AGGREGATION_LEVEL_1_MAP = {
    1110: 11,
    1120: 11,
    1150: 11,
    1130: 1130,
    1140: 1140,
    1210: 12,
    1220: 12,
    1310: 12,
    1320: 1320,
    1410: 1410,
    1420: 1420,
    1430: 1430,
    1440: 1440,
    2100: 20,
    2200: 20,
    2310: 20,
    2320: 20,
}
HRL_CTY_AGGREGATION_LEVEL_1_NAMES = {
    11: "Other cereals",
    1130: "Maize",
    1140: "Rice",
    12: "Dry pulses, fresh vegetables & potatoes",
    1320: "Sugar beet",
    1410: "Sunflower",
    1420: "Soybeans",
    1430: "Rapeseed",
    1440: "Flax, cotton and hemp",
    20: "Permanent crops",
}

_HRL_TILE_NAME_PATTERN = re.compile(
    r"^CLMS_HRLVLCC_(?P<product>CTY|CTYCL)_S(?P<year>\d{4})_"
    r"R10m_(?P<tile>[EW]\d+[NS]\d+)_03035_(?P<version>V\d+_R\d+)$"
)


@dataclass
class AlphaEarthConstantClassifier:
    """Minimal probability classifier for a branch containing one CTY class."""

    class_code: int

    def __post_init__(self) -> None:
        self.classes_ = np.asarray([int(self.class_code)], dtype=np.int32)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return probability one for the only represented class."""
        features = np.asarray(features)
        return np.ones((features.shape[0], 1), dtype=np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the only represented class for every row."""
        features = np.asarray(features)
        return np.full(features.shape[0], int(self.class_code), dtype=np.int32)


@dataclass(slots=True)
class AlphaEarthCropModelBundle:
    """Trained annual CTY classifier and its complete inference configuration.

    The bundle supports flat or soft-hierarchical native-CTY classification with
    any estimator family exposed by :func:`fit_alphaearth_crop_models`.

    Attributes:
        cty_model: Flat native-CTY classifier, or ``None`` for a hierarchical model.
        feature_names: Ordered predictor names expected by the classifier.
        model_type: Estimator family used by all trainable branches.
        model_parameters: Configured estimator settings retained for reproducibility.
        cty_classes: Ordered complete CTY output schema.
        trained_cty_classes: Native classes represented directly by the estimator.
        normalize_embeddings: Whether each 64-dimensional AEF vector is L2-normalized.
        classifier_structure: ``"flat"`` or ``"hierarchical"``.
        residual_class_mode: ``"learned"`` or ``"uncertainty"``.
        unclassified_probability_threshold: Default confidence rejection threshold.
        class_probability_thresholds: Optional class-specific decision thresholds.
        group_model: Crop-group classifier for hierarchical inference.
        within_group_models: Conditional native classifiers keyed by crop-group code.
    """

    cty_model: Any | None
    feature_names: tuple[str, ...]
    model_type: str = "logistic_regression"
    model_parameters: dict[str, Any] = field(default_factory=dict)
    cty_classes: tuple[int, ...] = HRL_CTY_CLASS_CODES
    trained_cty_classes: tuple[int, ...] = HRL_CTY_CLASS_CODES
    normalize_embeddings: bool = True
    classifier_structure: str = "flat"
    residual_class_mode: str = "uncertainty"
    unclassified_probability_threshold: float = 0.25
    class_probability_thresholds: dict[int, float] = field(default_factory=dict)
    group_model: Any | None = None
    within_group_models: dict[int, Any] = field(default_factory=dict)


ALPHAEARTH_SUPPORTED_MODEL_TYPES = (
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
)


HRL_CTY_RESIDUAL_CLASS_CODES = (1150, 3100, 3200)
HRL_CTY_UNCERTAINTY_TRAINING_CLASS_CODES = tuple(
    code for code in HRL_CTY_CLASS_CODES if code not in HRL_CTY_RESIDUAL_CLASS_CODES
)
HRL_CTY_HIERARCHICAL_GROUP_CODES = (0, 11, 12, 13, 14, 20)
HRL_CTY_CEREAL_EXPLICIT_CLASS_CODES = (1110, 1120, 1130, 1140)


def alphaearth_crop_feature_names(
    *,
    include_coordinates: bool = False,
    include_topography: bool = False,
) -> tuple[str, ...]:
    """Return the annual AlphaEarth predictor schema.

    All 64 AlphaEarth embedding dimensions are always retained. Optional
    longitude/latitude and elevation/slope predictors are appended without
    replacing or aggregating any embedding axis.

    Args:
        include_coordinates: Include longitude and latitude predictors.
        include_topography: Include elevation and local terrain-gradient predictors.

    Returns:
        Ordered annual feature names.
    """
    names: list[str] = [f"alphaearth_{band}" for band in ALPHAEARTH_EMBEDDING_BANDS]
    if include_coordinates:
        names.extend(("longitude", "latitude"))
    if include_topography:
        names.extend(("elevation_m", "slope_gradient"))
    return tuple(names)


def dequantize_alphaearth_embeddings(values: np.ndarray) -> np.ndarray:
    """Convert raw AlphaEarth int8 embeddings to float32 values.

    Args:
        values: Raw AlphaEarth values. The input shape is preserved.

    Returns:
        Dequantized float32 values with raw ``-128`` converted to ``NaN``.
    """
    raw = np.asarray(values)
    nodata = raw == ALPHAEARTH_NODATA_VALUE
    raw_float = raw.astype(np.float32)
    result = ((raw_float / 127.5) ** 2) * np.sign(raw_float)
    result[nodata] = np.nan
    return result


def normalize_alphaearth_embeddings(values: np.ndarray) -> np.ndarray:
    """L2-normalize dequantized AlphaEarth vectors row by row.

    AlphaEarth embeddings are trained on a unit hypersphere. Quantization and
    dequantization introduce small norm deviations; this function restores the
    intended geometry without changing the information contained in each vector.

    Args:
        values: Dequantized embedding matrix with shape ``(n, 64)``.

    Returns:
        Float32 matrix with unit-length finite rows. Non-finite rows remain NaN.
    """
    embeddings = np.asarray(values, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != 64:
        raise ValueError("AlphaEarth embeddings must have shape (n, 64).")

    normalized = embeddings.copy()
    finite_rows = np.isfinite(normalized).all(axis=1)
    if finite_rows.any():
        norms = np.linalg.norm(normalized[finite_rows], axis=1)
        valid_norms = np.isfinite(norms) & (norms > 1.0e-12)
        finite_indices = np.flatnonzero(finite_rows)
        invalid_indices = finite_indices[~valid_norms]
        if invalid_indices.size:
            normalized[invalid_indices] = np.nan
        valid_indices = finite_indices[valid_norms]
        normalized[valid_indices] /= norms[valid_norms, None]
    return normalized.astype(np.float32, copy=False)


def alphaearth_embedding_diagnostics(
    samples: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarize whether stored AEF columns look dequantized and well formed.

    The check is intentionally performed before optional L2 normalization. Raw
    signed-int8 values or values outside the documented dequantized range cause
    an error, while ordinary quantization-related norm variation is reported.

    Args:
        samples: Sample table containing all 64 ``alphaearth_*`` columns.

    Returns:
        Scalar diagnostics suitable for a one-row table or log message.

    Raises:
        ValueError: If embedding columns are missing, non-finite, or inconsistent
            with dequantized AlphaEarth values.
    """
    embedding_names = tuple(f"alphaearth_{band}" for band in ALPHAEARTH_EMBEDDING_BANDS)
    missing = set(embedding_names) - set(samples.columns)
    if missing:
        raise ValueError(
            f"AlphaEarth sample table is missing embedding columns: {sorted(missing)}"
        )
    embeddings = samples.loc[:, embedding_names].to_numpy(dtype=np.float32)
    finite_rows = np.isfinite(embeddings).all(axis=1)
    if not finite_rows.any():
        raise ValueError("No fully finite AlphaEarth embedding rows are available.")

    finite = embeddings[finite_rows]
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    maximum_absolute = float(np.max(np.abs(finite)))
    if maximum_absolute > 1.0001:
        raise ValueError(
            "AlphaEarth features do not appear dequantized: expected values within "
            f"[-1, 1], found range [{minimum:.4f}, {maximum:.4f}]."
        )

    norms = np.linalg.norm(finite, axis=1)
    return {
        "sample_rows": int(len(samples)),
        "finite_rows": int(finite_rows.sum()),
        "nonfinite_rows": int((~finite_rows).sum()),
        "minimum_value": minimum,
        "maximum_value": maximum,
        "median_l2_norm_before_normalization": float(np.median(norms)),
        "p05_l2_norm_before_normalization": float(np.quantile(norms, 0.05)),
        "p95_l2_norm_before_normalization": float(np.quantile(norms, 0.95)),
    }


def _update_priority_reservoir(
    reservoir: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    class_code: int,
    rows: np.ndarray,
    cols: np.ndarray,
    priorities: np.ndarray,
    maximum_samples: int,
) -> None:
    """Retain a reproducible bounded sample reservoir for one class."""
    if rows.size == 0 or maximum_samples <= 0:
        return

    previous = reservoir.get(class_code)
    if previous is not None:
        rows = np.concatenate((previous[0], rows))
        cols = np.concatenate((previous[1], cols))
        priorities = np.concatenate((previous[2], priorities))

    if priorities.size > maximum_samples:
        keep = np.argpartition(priorities, maximum_samples - 1)[:maximum_samples]
        rows = rows[keep]
        cols = cols[keep]
        priorities = priorities[keep]

    reservoir[class_code] = (
        rows.astype(np.int64, copy=False),
        cols.astype(np.int64, copy=False),
        priorities.astype(np.float64, copy=False),
    )


def _stable_class_interior_mask(
    values: np.ndarray,
    edge_buffer_pixels: int,
) -> np.ndarray:
    """Return pixels whose square neighbourhood has one unchanged class label.

    Pixels within ``edge_buffer_pixels`` of a raster edge or any different class
    are excluded. The helper is deliberately NumPy-only so it does not introduce
    a SciPy dependency into the build workflow.
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional class raster.")
    if edge_buffer_pixels < 0:
        raise ValueError("edge_buffer_pixels cannot be negative.")
    if edge_buffer_pixels == 0:
        return np.ones(values.shape, dtype=bool)

    radius = int(edge_buffer_pixels)
    padded = np.pad(
        values,
        ((radius, radius), (radius, radius)),
        mode="constant",
        constant_values=np.iinfo(np.int32).min,
    )
    stable = np.ones(values.shape, dtype=bool)
    height, width = values.shape
    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            neighbour = padded[
                radius + row_offset : radius + row_offset + height,
                radius + col_offset : radius + col_offset + width,
            ]
            stable &= neighbour == values
    return stable


def _lattice_positions(
    *,
    stride: int,
    row_offset: int,
    col_offset: int,
    y_start: int,
    chunk_height: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local row and global column positions for one sampling lattice."""
    local_row_start = (row_offset - y_start) % stride
    return (
        np.arange(local_row_start, chunk_height, stride, dtype=np.int64),
        np.arange(col_offset, n_cols, stride, dtype=np.int64),
    )


def balanced_hrl_sample_locations(
    crop_types: xr.DataArray,
    clip_geometry: BaseGeometry,
    *,
    geometry_crs: str = "EPSG:4326",
    samples_per_cty_class: int = 500,
    sample_stride_pixels: int = 5,
    rare_class_sample_stride_pixels: int | None = 1,
    rare_class_threshold_candidates: int = 50_000,
    rare_class_sample_multiplier: float = 3.0,
    training_label_edge_buffer_pixels: int = 0,
    chunk_rows: int = 512,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Select edge-filtered, adaptive, class-balanced CTY sample locations.

    Common classes use the standard spatially thinned lattice. Classes with fewer
    than ``rare_class_threshold_candidates`` eligible observations on the denser
    rare-class lattice use that lattice instead and receive an enlarged bounded
    reservoir. Pixels close to CTY class boundaries can be excluded to reduce
    mixed-pixel and label-registration noise.
    """
    if crop_types.rio.crs is None:
        raise ValueError("The HRL CTY raster must have a CRS.")
    if samples_per_cty_class < 1:
        raise ValueError("samples_per_cty_class must be at least one.")
    if sample_stride_pixels < 1:
        raise ValueError("sample_stride_pixels must be at least one.")
    if (
        rare_class_sample_stride_pixels is not None
        and rare_class_sample_stride_pixels < 1
    ):
        raise ValueError("rare_class_sample_stride_pixels must be at least one.")
    if rare_class_threshold_candidates < 1:
        raise ValueError("rare_class_threshold_candidates must be at least one.")
    if rare_class_sample_multiplier < 1.0:
        raise ValueError("rare_class_sample_multiplier must be at least 1.0.")
    if training_label_edge_buffer_pixels < 0:
        raise ValueError("training_label_edge_buffer_pixels cannot be negative.")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be at least one.")

    rare_stride = (
        sample_stride_pixels
        if rare_class_sample_stride_pixels is None
        else int(rare_class_sample_stride_pixels)
    )
    geometry_in_raster_crs = (
        gpd.GeoSeries([clip_geometry], crs=geometry_crs)
        .to_crs(crop_types.rio.crs)
        .iloc[0]
    )
    if geometry_in_raster_crs.is_empty:
        raise ValueError("The HRL sampling geometry is empty after reprojection.")

    rng = np.random.default_rng(random_seed)
    standard_offsets = (
        int(rng.integers(0, sample_stride_pixels)),
        int(rng.integers(0, sample_stride_pixels)),
    )
    rare_offsets = (
        int(rng.integers(0, rare_stride)),
        int(rng.integers(0, rare_stride)),
    )

    standard_cty: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rare_cty: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    cty_candidate_counts = {int(code): 0 for code in HRL_CTY_CLASS_CODES}
    rare_cty_cap = max(
        samples_per_cty_class,
        int(np.ceil(samples_per_cty_class * rare_class_sample_multiplier)),
    )

    n_rows = crop_types.sizes["y"]
    n_cols = crop_types.sizes["x"]
    radius = int(training_label_edge_buffer_pixels)

    for y_start in range(0, n_rows, chunk_rows):
        y_stop = min(y_start + chunk_rows, n_rows)
        read_start = max(0, y_start - radius)
        read_stop = min(n_rows, y_stop + radius)
        crop_read_da = crop_types.isel(y=slice(read_start, read_stop))

        crop_read = np.asarray(crop_read_da.values)
        if np.issubdtype(crop_read.dtype, np.floating):
            crop_read = np.nan_to_num(crop_read, nan=_HRL_OUTSIDE_AREA_CODE)
        crop_read = crop_read.astype(np.int32, copy=False)

        core_start = y_start - read_start
        core_stop = core_start + (y_stop - y_start)
        crop_values = crop_read[core_start:core_stop]
        crop_interior = _stable_class_interior_mask(crop_read, radius)[
            core_start:core_stop
        ]

        crop_chunk_da = crop_types.isel(y=slice(y_start, y_stop))
        inside_region = ~rasterio.features.geometry_mask(
            [geometry_in_raster_crs.__geo_interface__],
            out_shape=crop_values.shape,
            transform=crop_chunk_da.rio.transform(recalc=True),
            invert=False,
            all_touched=False,
        )

        def update_lattice(
            *,
            stride: int,
            offsets: tuple[int, int],
            reservoir: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
            sample_cap: int,
            count_candidates: bool,
        ) -> None:
            row_positions, col_positions = _lattice_positions(
                stride=stride,
                row_offset=offsets[0],
                col_offset=offsets[1],
                y_start=y_start,
                chunk_height=crop_values.shape[0],
                n_cols=n_cols,
            )
            if row_positions.size == 0 or col_positions.size == 0:
                return

            crop_lattice = crop_values[np.ix_(row_positions, col_positions)]
            eligible = (
                inside_region[np.ix_(row_positions, col_positions)]
                & crop_interior[np.ix_(row_positions, col_positions)]
            )
            global_rows = row_positions + y_start

            for class_code in HRL_CTY_CLASS_CODES:
                candidate_rows, candidate_cols = np.where(
                    eligible & (crop_lattice == class_code)
                )
                if count_candidates:
                    cty_candidate_counts[int(class_code)] += int(candidate_rows.size)
                if candidate_rows.size:
                    rows = global_rows[candidate_rows]
                    cols = col_positions[candidate_cols]
                    _update_priority_reservoir(
                        reservoir,
                        class_code=int(class_code),
                        rows=rows,
                        cols=cols,
                        priorities=rng.random(rows.size),
                        maximum_samples=sample_cap,
                    )

        update_lattice(
            stride=sample_stride_pixels,
            offsets=standard_offsets,
            reservoir=standard_cty,
            sample_cap=samples_per_cty_class,
            count_candidates=(rare_stride == sample_stride_pixels),
        )
        if rare_stride != sample_stride_pixels:
            update_lattice(
                stride=rare_stride,
                offsets=rare_offsets,
                reservoir=rare_cty,
                sample_cap=rare_cty_cap,
                count_candidates=True,
            )
        else:
            rare_cty = standard_cty

    selected_locations: set[tuple[int, int]] = set()
    for class_code in HRL_CTY_CLASS_CODES:
        is_rare = (
            cty_candidate_counts[int(class_code)] < rare_class_threshold_candidates
        )
        reservoir = rare_cty if is_rare else standard_cty
        selected = reservoir.get(int(class_code))
        if selected is not None:
            selected_locations.update(
                (int(row), int(col))
                for row, col in zip(selected[0], selected[1], strict=True)
            )

    if not selected_locations:
        raise ValueError("No HRL CTY sample locations were selected.")

    ordered_locations = sorted(selected_locations)
    rows = np.asarray([location[0] for location in ordered_locations], dtype=np.int64)
    cols = np.asarray([location[1] for location in ordered_locations], dtype=np.int64)
    row_indexer = xr.DataArray(rows, dims=("sample",))
    col_indexer = xr.DataArray(cols, dims=("sample",))
    cty_labels = np.asarray(
        crop_types.isel(y=row_indexer, x=col_indexer).values,
        dtype=np.int32,
    )

    transform = crop_types.rio.transform(recalc=True)
    x_coordinates, y_coordinates = rasterio.transform.xy(
        transform,
        rows,
        cols,
        offset="center",
    )
    return pd.DataFrame(
        {
            "row": rows,
            "col": cols,
            "x": np.asarray(x_coordinates, dtype=np.float64),
            "y": np.asarray(y_coordinates, dtype=np.float64),
            "cty_label": cty_labels,
        }
    )


def select_alphaearth_cogs_for_geometry(
    selected_cogs: gpd.GeoDataFrame,
    *,
    year: int,
    clip_geometry: BaseGeometry,
) -> gpd.GeoDataFrame:
    """Select downloaded AlphaEarth COGs for one year and geometry.

    This helper allows one study-area-wide AlphaEarth download to be reused by
    many regional sampling passes or HRL prediction tiles without repeatedly
    downloading the same large COGs.

    Args:
        selected_cogs: Study-area AlphaEarth selection returned by the adapter.
        year: Required AlphaEarth observation year.
        clip_geometry: WGS84 region or output-tile geometry.

    Returns:
        GeoDataFrame containing only COGs from ``year`` that intersect
        ``clip_geometry``.

    Raises:
        ValueError: If required columns, CRS information, or geometry are invalid.
    """
    if "year" not in selected_cogs.columns:
        raise ValueError("selected_cogs must contain a 'year' column.")
    if selected_cogs.crs is None:
        raise ValueError("selected_cogs must have a CRS.")
    if clip_geometry is None or clip_geometry.is_empty:
        raise ValueError("clip_geometry must be a non-empty WGS84 geometry.")

    cogs_wgs84 = (
        selected_cogs
        if selected_cogs.crs.to_epsg() == 4326
        else selected_cogs.to_crs("EPSG:4326")
    )
    year_mask = cogs_wgs84["year"].astype(int).eq(int(year))
    geometry_mask = cogs_wgs84.geometry.intersects(clip_geometry)
    return cogs_wgs84.loc[year_mask & geometry_mask].copy()


def sample_alphaearth_embeddings(
    selected_cogs: gpd.GeoDataFrame,
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> np.ndarray:
    """Sample dequantized AlphaEarth embeddings from downloaded COGs.

    Args:
        selected_cogs: AlphaEarth index rows containing ``local_path`` and
            WGS84 geometries for one year.
        longitude: Sample longitudes in EPSG:4326.
        latitude: Sample latitudes in EPSG:4326.

    Returns:
        Float32 array with shape ``(n_samples, 64)``. Samples not covered by a
        valid COG remain ``NaN``.
    """
    if "local_path" not in selected_cogs.columns:
        raise ValueError("selected_cogs must contain a 'local_path' column.")
    longitude = np.asarray(longitude, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)
    if longitude.shape != latitude.shape:
        raise ValueError("longitude and latitude must align.")

    cogs = selected_cogs
    if cogs.crs is None:
        cogs = cogs.set_crs("EPSG:4326")
    elif cogs.crs.to_epsg() != 4326:
        cogs = cogs.to_crs("EPSG:4326")

    embeddings = np.full(
        (longitude.size, len(ALPHAEARTH_EMBEDDING_BANDS)),
        np.nan,
        dtype=np.float32,
    )
    unresolved = np.ones(longitude.size, dtype=bool)

    for cog in cogs.itertuples(index=False):
        if not unresolved.any():
            break
        local_path = Path(str(cog.local_path))
        if not local_path.exists():
            raise FileNotFoundError(f"Missing downloaded AlphaEarth COG: {local_path}")

        west, south, east, north = cog.geom.bounds
        candidate_mask = (
            unresolved
            & (longitude >= west)
            & (longitude <= east)
            & (latitude >= south)
            & (latitude <= north)
        )
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            continue

        with rasterio.open(local_path) as source:
            if source.count != len(ALPHAEARTH_EMBEDDING_BANDS):
                raise ValueError(
                    f"Expected 64 AlphaEarth bands in {local_path}, "
                    f"found {source.count}."
                )
            transformer = Transformer.from_crs(
                "EPSG:4326",
                source.crs,
                always_xy=True,
            )
            x_values, y_values = transformer.transform(
                longitude[candidate_indices],
                latitude[candidate_indices],
            )
            raw_samples = np.vstack(
                list(
                    source.sample(
                        zip(x_values, y_values, strict=True),
                        indexes=list(range(1, source.count + 1)),
                        masked=False,
                    )
                )
            )

        valid = np.all(raw_samples != ALPHAEARTH_NODATA_VALUE, axis=1)
        if valid.any():
            resolved_indices = candidate_indices[valid]
            embeddings[resolved_indices] = dequantize_alphaearth_embeddings(
                raw_samples[valid]
            )
            unresolved[resolved_indices] = False

    return embeddings


def sample_dataarray_at_coordinates(
    data: xr.DataArray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    *,
    coordinates_crs: str | CRS,
) -> np.ndarray:
    """Sample one regular two-dimensional raster using bilinear interpolation.

    The input coordinates may use any CRS. Values outside the raster or surrounded
    only by nodata are returned as ``NaN``.

    Args:
        data: Two-dimensional georeferenced DataArray.
        x_coordinates: X coordinates of sample locations.
        y_coordinates: Y coordinates of sample locations.
        coordinates_crs: CRS of the supplied coordinates.

    Returns:
        Float32 sampled values aligned with the input coordinates.
    """
    if data.ndim != 2:
        raise ValueError("Topographic predictor rasters must be two-dimensional.")
    if data.rio.crs is None:
        raise ValueError("Topographic predictor rasters must have a CRS.")

    x_coordinates = np.asarray(x_coordinates, dtype=np.float64)
    y_coordinates = np.asarray(y_coordinates, dtype=np.float64)
    if x_coordinates.shape != y_coordinates.shape:
        raise ValueError("Topographic sample coordinates must align.")

    coordinate_crs = CRS.from_user_input(coordinates_crs)
    data_crs = CRS.from_user_input(data.rio.crs)
    if coordinate_crs != data_crs:
        transformer = Transformer.from_crs(
            coordinate_crs,
            data_crs,
            always_xy=True,
        )
        x_coordinates, y_coordinates = transformer.transform(
            x_coordinates,
            y_coordinates,
        )
        x_coordinates = np.asarray(x_coordinates, dtype=np.float64)
        y_coordinates = np.asarray(y_coordinates, dtype=np.float64)

    values = np.asarray(data.values, dtype=np.float32)
    transform = data.rio.transform(recalc=True)
    inverse_transform = ~transform
    col_corner, row_corner = inverse_transform * (x_coordinates, y_coordinates)

    # Rasterio affine coordinates refer to pixel corners. Shift by half a pixel so
    # integer indices correspond to pixel centres before bilinear interpolation.
    col_position = np.asarray(col_corner, dtype=np.float64) - 0.5
    row_position = np.asarray(row_corner, dtype=np.float64) - 0.5

    height, width = values.shape
    inside = (
        (col_position >= -0.5)
        & (col_position <= width - 0.5)
        & (row_position >= -0.5)
        & (row_position <= height - 0.5)
    )
    result = np.full(x_coordinates.shape, np.nan, dtype=np.float32)
    if not inside.any():
        return result

    clipped_cols = np.clip(col_position[inside], 0.0, width - 1.0)
    clipped_rows = np.clip(row_position[inside], 0.0, height - 1.0)
    col0 = np.floor(clipped_cols).astype(np.int64)
    row0 = np.floor(clipped_rows).astype(np.int64)
    col1 = np.minimum(col0 + 1, width - 1)
    row1 = np.minimum(row0 + 1, height - 1)
    col_weight = clipped_cols - col0
    row_weight = clipped_rows - row0

    v00 = values[row0, col0]
    v01 = values[row0, col1]
    v10 = values[row1, col0]
    v11 = values[row1, col1]
    weights = np.column_stack(
        (
            (1.0 - row_weight) * (1.0 - col_weight),
            (1.0 - row_weight) * col_weight,
            row_weight * (1.0 - col_weight),
            row_weight * col_weight,
        )
    ).astype(np.float32)
    neighbours = np.column_stack((v00, v01, v10, v11)).astype(np.float32)
    finite = np.isfinite(neighbours)
    weighted_sum = np.sum(
        np.where(finite, neighbours * weights, 0.0),
        axis=1,
    )
    weight_sum = np.sum(np.where(finite, weights, 0.0), axis=1)
    sampled = np.divide(
        weighted_sum,
        weight_sum,
        out=np.full(weighted_sum.shape, np.nan, dtype=np.float32),
        where=weight_sum > 0.0,
    )
    result[np.flatnonzero(inside)] = sampled.astype(np.float32, copy=False)
    return result


def build_alphaearth_crop_feature_matrix(
    current_embeddings: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    include_coordinates: bool = False,
    include_topography: bool = False,
    normalize_embeddings: bool = False,
    elevation_m: np.ndarray | None = None,
    slope_gradient: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct target-year AlphaEarth predictors for training and inference.

    The 64 dequantized AEF dimensions are always retained. Optional coordinate and
    topographic predictors are appended only for controlled ablation experiments.
    No lagged year or external EO time-series feature is introduced.

    Args:
        current_embeddings: Same-year AlphaEarth embeddings with shape ``(n, 64)``.
        longitude: Longitude per observation.
        latitude: Latitude per observation.
        include_coordinates: Include longitude and latitude predictors.
        include_topography: Include elevation and local terrain gradient.
        normalize_embeddings: Restore unit-hypersphere geometry with row-wise L2
            normalization.
        elevation_m: Elevation per observation when topography is enabled.
        slope_gradient: Local terrain gradient per observation when enabled.

    Returns:
        Predictor matrix and its ordered feature names.
    """
    embeddings = np.asarray(current_embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != 64:
        raise ValueError("AlphaEarth embeddings must have shape (n, 64).")
    if normalize_embeddings:
        embeddings = normalize_alphaearth_embeddings(embeddings)

    n_samples = embeddings.shape[0]
    longitude = np.asarray(longitude, dtype=np.float32)
    latitude = np.asarray(latitude, dtype=np.float32)
    if longitude.shape != (n_samples,) or latitude.shape != (n_samples,):
        raise ValueError("Coordinates must contain one value per embedding sample.")

    feature_parts: list[np.ndarray] = [embeddings]
    if include_coordinates:
        feature_parts.append(np.column_stack((longitude, latitude)).astype(np.float32))

    if include_topography:
        if elevation_m is None or slope_gradient is None:
            raise ValueError(
                "elevation_m and slope_gradient are required when "
                "include_topography=True."
            )
        elevation_values = np.asarray(elevation_m, dtype=np.float32)
        slope_values = np.asarray(slope_gradient, dtype=np.float32)
        if elevation_values.shape != (n_samples,) or slope_values.shape != (n_samples,):
            raise ValueError(
                "Topographic predictors must contain one value per embedding sample."
            )
        feature_parts.append(
            np.column_stack((elevation_values, slope_values)).astype(np.float32)
        )

    feature_matrix = np.column_stack(feature_parts).astype(np.float32, copy=False)
    feature_names = alphaearth_crop_feature_names(
        include_coordinates=include_coordinates,
        include_topography=include_topography,
    )
    if feature_matrix.shape[1] != len(feature_names):
        raise AssertionError("AlphaEarth feature matrix and schema do not align.")
    return feature_matrix, feature_names


def alphaearth_features_from_samples(
    samples: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    normalize_embeddings: bool,
) -> np.ndarray:
    """Extract one model matrix from a reusable AEF sample table.

    Normalization is applied at fit/evaluation time so existing Parquet tables
    containing dequantized but non-normalized vectors remain reusable.
    """
    names = tuple(str(name) for name in feature_names)
    missing = set(names) - set(samples.columns)
    if missing:
        raise ValueError(f"AlphaEarth samples are missing features: {sorted(missing)}")
    # ``DataFrame.to_numpy`` may expose a read-only zero-copy view when the
    # reusable table was loaded from Parquet through an Arrow-backed block.
    # Normalization replaces only the embedding columns below, so explicitly
    # materialize an owned, C-contiguous and writable float32 matrix first.
    features = np.array(
        samples.loc[:, names].to_numpy(dtype=np.float32),
        dtype=np.float32,
        copy=True,
        order="C",
    )
    if not features.flags.writeable:
        # This is defensive; ``np.array(..., copy=True)`` should already own a
        # writable buffer on supported NumPy versions.
        features = features.copy(order="C")

    embedding_names = tuple(f"alphaearth_{band}" for band in ALPHAEARTH_EMBEDDING_BANDS)
    embedding_positions = [names.index(name) for name in embedding_names]
    if normalize_embeddings:
        features[:, embedding_positions] = normalize_alphaearth_embeddings(
            features[:, embedding_positions]
        )
    if not np.isfinite(features).all():
        raise ValueError("AlphaEarth model features contain non-finite values.")
    return features


def load_alphaearth_crop_training_samples(
    source: pd.DataFrame | str | Path,
    *,
    hrl_years: tuple[int, ...],
    include_coordinates: bool = False,
    include_topography: bool = False,
    active_region_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Load and validate a previously stored AlphaEarth-HRL CTY sample table.

    The loader allows repeated model-fitting and prediction experiments without
    rerunning the expensive HRL/AlphaEarth sampling stage. It selects only the
    requested observation years and validates that the stored feature schema is
    compatible with the current CTY classifier settings. Additional columns in an
    older reusable table are ignored by the CTY-only fitting workflow.

    Args:
        source: Existing sample DataFrame or Parquet, CSV, or pickle path.
        hrl_years: Observation years that must be present in the stored table.
        include_coordinates: Require stored longitude and latitude predictors.
        include_topography: Require stored elevation and slope predictors.
        active_region_ids: Optional region IDs belonging to the active model. Any
            stored sample from another region causes an error.

    Returns:
        Validated CTY sample table restricted to ``hrl_years``.

    Raises:
        FileNotFoundError: If a supplied sample-table path does not exist.
        TypeError: If the loaded object is not a DataFrame.
        ValueError: If required columns, years, labels, or regions are invalid.
    """
    requested_years = tuple(int(year) for year in hrl_years)
    if not requested_years:
        raise ValueError("hrl_years must contain at least one year.")

    if isinstance(source, pd.DataFrame):
        samples = source.copy()
    else:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(
                f"Stored AlphaEarth sample table does not exist: {source_path}"
            )
        suffix = source_path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            samples = pd.read_parquet(source_path)
        elif suffix == ".csv":
            samples = pd.read_csv(source_path)
        elif suffix in {".pkl", ".pickle"}:
            samples = pd.read_pickle(source_path)
        else:
            raise ValueError(
                "Stored AlphaEarth samples must use Parquet, CSV or pickle; "
                f"found {source_path}."
            )

    if not isinstance(samples, pd.DataFrame):
        raise TypeError(
            "Stored AlphaEarth samples must load as a pandas DataFrame; found "
            f"{type(samples).__name__}."
        )
    if samples.empty:
        raise ValueError("Stored AlphaEarth sample table is empty.")

    feature_names = alphaearth_crop_feature_names(
        include_coordinates=include_coordinates,
        include_topography=include_topography,
    )
    required_columns = set(feature_names) | {
        "cty_label",
        "year",
        "region_id",
        "country_iso3",
    }
    missing_columns = required_columns - set(samples.columns)
    if missing_columns:
        raise ValueError(
            "Stored AlphaEarth samples are incompatible with the current feature "
            f"settings; missing columns: {sorted(missing_columns)}."
        )

    numeric_years = pd.to_numeric(samples["year"], errors="coerce")
    if numeric_years.isna().any() or not np.allclose(
        numeric_years.to_numpy(dtype=np.float64),
        np.rint(numeric_years.to_numpy(dtype=np.float64)),
    ):
        raise ValueError("Stored AlphaEarth sample years must be finite integers.")
    samples["year"] = np.rint(numeric_years).astype(np.int32)

    available_years = set(int(year) for year in samples["year"].unique())
    missing_years = set(requested_years) - available_years
    if missing_years:
        raise ValueError(
            "Stored AlphaEarth samples do not contain every requested HRL year. "
            f"Missing years: {sorted(missing_years)}; available years: "
            f"{sorted(available_years)}."
        )
    samples = samples.loc[samples["year"].isin(requested_years)].copy()

    numeric_regions = pd.to_numeric(samples["region_id"], errors="coerce")
    if numeric_regions.isna().any() or not np.allclose(
        numeric_regions.to_numpy(dtype=np.float64),
        np.rint(numeric_regions.to_numpy(dtype=np.float64)),
    ):
        raise ValueError("Stored AlphaEarth sample region IDs must be finite integers.")
    samples["region_id"] = np.rint(numeric_regions).astype(np.int32)

    if active_region_ids is not None:
        expected_regions = np.unique(np.asarray(active_region_ids, dtype=np.int32))
        stored_regions = np.unique(samples["region_id"].to_numpy(dtype=np.int32))
        unknown_regions = np.setdiff1d(stored_regions, expected_regions)
        if unknown_regions.size:
            raise ValueError(
                "Stored AlphaEarth samples contain region IDs outside the active "
                f"model: {unknown_regions[:20].tolist()}."
            )

    cty_labels = samples["cty_label"].to_numpy(dtype=np.int32)
    invalid_cty = ~np.isin(cty_labels, HRL_CTY_CLASS_CODES)
    if invalid_cty.any():
        invalid_values = np.unique(cty_labels[invalid_cty])
        raise ValueError(
            "Stored AlphaEarth samples contain unsupported CTY labels: "
            f"{invalid_values[:20].tolist()}."
        )

    # A Parquet table written by an earlier run may contain a stale split column.
    # The caller always reconstructs temporal splits from the current settings.
    return samples.reset_index(drop=True)


def parse_europe_model_ids(
    values: Sequence[int | str] | int | str,
) -> tuple[int, ...]:
    """Parse Europe model IDs from integers, names, comma lists, and ranges.

    Args:
        values: One value or a sequence containing integers, ``Europe_###`` names,
            comma-separated values, or inclusive ranges such as ``"0-11"``.

    Returns:
        Unique model IDs in their configured order.

    Raises:
        ValueError: If no model IDs are provided or a token is invalid.
    """
    raw_values: Sequence[int | str]
    if isinstance(values, (int, str)):
        raw_values = (values,)
    else:
        raw_values = values

    tokens: list[str] = []
    for raw_value in raw_values:
        if isinstance(raw_value, (int, np.integer)):
            tokens.append(str(int(raw_value)))
        else:
            tokens.extend(str(raw_value).replace(",", " ").split())

    if not tokens:
        raise ValueError("At least one Europe model ID must be provided.")

    def parse_one(token: str) -> int:
        normalized = token.strip()
        if normalized.lower().startswith("europe_"):
            normalized = normalized[len("Europe_") :]
        if not normalized.isdigit():
            raise ValueError(
                "Europe model IDs must be integers or names such as Europe_003; "
                f"found {token!r}."
            )
        model_id = int(normalized)
        if not 0 <= model_id <= 999:
            raise ValueError(
                f"Europe model IDs must be between 0 and 999; found {model_id}."
            )
        return model_id

    model_ids: list[int] = []
    for token in tokens:
        if token.count("-") == 1:
            start_raw, end_raw = token.split("-", maxsplit=1)
            start_id = parse_one(start_raw)
            end_id = parse_one(end_raw)
            if end_id < start_id:
                raise ValueError(f"Europe model range {token!r} ends before it starts.")
            model_ids.extend(range(start_id, end_id + 1))
        else:
            model_ids.append(parse_one(token))

    return tuple(dict.fromkeys(model_ids))


def europe_model_build_context(
    base_directory: str | Path | None = None,
) -> tuple[int, Path, Path]:
    """Resolve the current Europe model ID, base directory, and common root.

    Args:
        base_directory: Europe model base directory. The current working directory
            is used when omitted.

    Returns:
        Tuple ``(model_id, base_directory, model_root)``.

    Raises:
        ValueError: If the directory is not ``.../Europe_###/base``.
    """
    resolved_base = (
        Path.cwd().resolve()
        if base_directory is None
        else Path(base_directory).expanduser().resolve()
    )
    model_directory = resolved_base.parent
    match = re.fullmatch(r"Europe_(\d{3})", model_directory.name)
    if resolved_base.name != "base" or match is None:
        raise ValueError(
            "The build working directory must have the structure "
            "'.../<model root>/Europe_###/base'. "
            f"Found {resolved_base}."
        )
    return int(match.group(1)), resolved_base, model_directory.parent


def europe_model_base_directory(model_root: str | Path, model_id: int) -> Path:
    """Return the base directory of one Europe model."""
    return Path(model_root) / f"Europe_{int(model_id):03d}" / "base"


def alphaearth_crop_training_samples_path(
    model_base_directory: str | Path,
    training_samples_table_name: str,
) -> Path:
    """Resolve a model-local ``self.table`` name to its Parquet path.

    Args:
        model_base_directory: Path to ``Europe_###/base``.
        training_samples_table_name: Slash-separated table name passed to
            ``self.set_table``.

    Returns:
        Path below ``base/input/table`` with a ``.parquet`` suffix.

    Raises:
        ValueError: If the table name is empty, absolute, or escapes its root.
    """
    table_name = str(training_samples_table_name).strip().replace("\\", "/")
    relative_path = Path(table_name)
    if not table_name or relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            "training_samples_table_name must be a non-empty relative table name."
        )
    if relative_path.suffix != ".parquet":
        relative_path = relative_path.with_suffix(".parquet")
    return Path(model_base_directory) / "input" / "table" / relative_path


def load_europe_alphaearth_crop_training_samples(
    model_root: str | Path,
    europe_model_ids: Sequence[int | str] | int | str,
    *,
    training_samples_table_name: str,
    hrl_years: tuple[int, ...],
    include_coordinates: bool = False,
    include_topography: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and pool model-local AlphaEarth CTY samples across Europe models.

    Local region IDs are only unique within one ``Europe_###`` model. The pooled
    table therefore retains them as ``local_region_id`` and replaces ``region_id``
    with deterministic Europe-wide IDs. This keeps region-year balancing and
    diagnostics separated across model boundaries.

    Args:
        model_root: Directory containing the ``Europe_###`` model folders.
        europe_model_ids: Selected model IDs or ranges.
        training_samples_table_name: Model-local sample table name.
        hrl_years: Observation years required from every selected model.
        include_coordinates: Require coordinate predictors.
        include_topography: Require elevation and slope predictors.

    Returns:
        Pooled sample table and a mapping from global to local region IDs.

    Raises:
        FileNotFoundError: If a selected model or sample table is missing.
        ValueError: If sample schemas or configured source-model metadata conflict.
    """
    selected_model_ids = parse_europe_model_ids(europe_model_ids)
    root = Path(model_root).expanduser().resolve()
    pooled_tables: list[pd.DataFrame] = []
    mapping_rows: list[dict[str, int | str]] = []
    next_global_region_id = 0

    for model_id in selected_model_ids:
        model_base = europe_model_base_directory(root, model_id)
        if not model_base.is_dir():
            raise FileNotFoundError(
                f"Europe model base directory does not exist: {model_base}"
            )
        sample_path = alphaearth_crop_training_samples_path(
            model_base,
            training_samples_table_name,
        )
        samples = load_alphaearth_crop_training_samples(
            sample_path,
            hrl_years=hrl_years,
            include_coordinates=include_coordinates,
            include_topography=include_topography,
        )

        if "europe_model_id" in samples.columns:
            stored_model_ids = pd.to_numeric(
                samples["europe_model_id"], errors="coerce"
            )
            if (
                stored_model_ids.isna().any()
                or not (stored_model_ids.astype(np.int64) == model_id).all()
            ):
                raise ValueError(
                    f"Stored source-model metadata in {sample_path} does not match "
                    f"Europe_{model_id:03d}."
                )

        samples["europe_model_id"] = np.int16(model_id)
        samples["europe_model_name"] = f"Europe_{model_id:03d}"
        samples["local_region_id"] = samples["region_id"].astype(np.int32)

        local_region_ids = np.sort(samples["local_region_id"].unique())
        local_to_global = {
            int(local_region_id): next_global_region_id + offset
            for offset, local_region_id in enumerate(local_region_ids)
        }
        samples["region_id"] = (
            samples["local_region_id"].map(local_to_global).astype(np.int32)
        )

        country_by_region = (
            samples[["local_region_id", "country_iso3"]]
            .drop_duplicates()
            .groupby("local_region_id", sort=True)["country_iso3"]
            .agg(lambda values: ",".join(sorted(set(map(str, values)))))
        )
        for local_region_id in local_region_ids:
            mapping_rows.append(
                {
                    "region_id": int(local_to_global[int(local_region_id)]),
                    "europe_model_id": int(model_id),
                    "europe_model_name": f"Europe_{model_id:03d}",
                    "local_region_id": int(local_region_id),
                    "country_iso3": str(country_by_region.loc[local_region_id]),
                }
            )

        next_global_region_id += len(local_region_ids)
        pooled_tables.append(samples)

    if not pooled_tables:
        raise ValueError("No AlphaEarth CTY sample tables were loaded.")

    pooled = pd.concat(pooled_tables, ignore_index=True)
    region_mapping = (
        pd.DataFrame(mapping_rows).sort_values("region_id").reset_index(drop=True)
    )
    return pooled, region_mapping


def create_alphaearth_crop_training_samples(
    crop_types: xr.DataArray,
    current_alphaearth_cogs: gpd.GeoDataFrame,
    clip_geometry: BaseGeometry,
    *,
    year: int,
    region_id: int,
    country_iso3: str,
    samples_per_cty_class: int = 500,
    sample_stride_pixels: int = 5,
    rare_class_sample_stride_pixels: int | None = 1,
    rare_class_threshold_candidates: int = 50_000,
    rare_class_sample_multiplier: float = 3.0,
    training_label_edge_buffer_pixels: int = 0,
    sample_chunk_rows: int = 512,
    include_coordinates: bool = False,
    include_topography: bool = False,
    elevation: xr.DataArray | None = None,
    slope: xr.DataArray | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Create one regional, annual AlphaEarth-HRL CTY training sample table.

    Each HRL observation year is independent: CTY labels from year ``t`` are paired
    only with AlphaEarth embeddings from year ``t``. Elevation and slope are
    optional static auxiliary predictors. No previous crop type is included.

    Args:
        crop_types: Same-year native HRL Crop Types raster.
        current_alphaearth_cogs: Downloaded same-year AlphaEarth COG index rows.
        clip_geometry: Active regional geometry in EPSG:4326.
        year: HRL/AlphaEarth observation year.
        region_id: Model region identifier.
        country_iso3: Region country code.
        samples_per_cty_class: Maximum CTY samples per class.
        sample_stride_pixels: Standard sampling-lattice spacing in native HRL pixels.
        rare_class_sample_stride_pixels: Denser lattice used for rare classes.
        rare_class_threshold_candidates: Candidate count below which a class is rare.
        rare_class_sample_multiplier: Multiplier for rare-class reservoir sizes.
        training_label_edge_buffer_pixels: CTY class-boundary erosion radius in
            native HRL pixels.
        sample_chunk_rows: Native HRL rows scanned in each sampling chunk.
        include_coordinates: Include coordinates as optional predictors.
        include_topography: Include elevation and local terrain gradient.
        elevation: Georeferenced model-subgrid elevation.
        slope: Georeferenced local terrain-gradient raster.
        random_seed: Reproducible regional-year seed.

    Returns:
        Sample table containing annual predictors, CTY labels, coordinates, year,
        and region metadata.
    """
    if include_topography and (elevation is None or slope is None):
        raise ValueError(
            "elevation and slope are required when include_topography=True."
        )

    locations = balanced_hrl_sample_locations(
        crop_types,
        clip_geometry,
        samples_per_cty_class=samples_per_cty_class,
        sample_stride_pixels=sample_stride_pixels,
        rare_class_sample_stride_pixels=rare_class_sample_stride_pixels,
        rare_class_threshold_candidates=rare_class_threshold_candidates,
        rare_class_sample_multiplier=rare_class_sample_multiplier,
        training_label_edge_buffer_pixels=training_label_edge_buffer_pixels,
        chunk_rows=sample_chunk_rows,
        random_seed=random_seed,
    )

    source_x = locations["x"].to_numpy(dtype=np.float64)
    source_y = locations["y"].to_numpy(dtype=np.float64)
    coordinate_transformer = Transformer.from_crs(
        crop_types.rio.crs,
        "EPSG:4326",
        always_xy=True,
    )
    longitude, latitude = coordinate_transformer.transform(source_x, source_y)
    longitude = np.asarray(longitude, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)

    embeddings = sample_alphaearth_embeddings(
        current_alphaearth_cogs,
        longitude,
        latitude,
    )

    elevation_values: np.ndarray | None = None
    slope_values: np.ndarray | None = None
    if include_topography:
        assert elevation is not None and slope is not None
        elevation_values = sample_dataarray_at_coordinates(
            elevation,
            source_x,
            source_y,
            coordinates_crs=crop_types.rio.crs,
        )
        slope_values = sample_dataarray_at_coordinates(
            slope,
            source_x,
            source_y,
            coordinates_crs=crop_types.rio.crs,
        )

    cty_label = locations["cty_label"].to_numpy(dtype=np.int32)
    valid = np.isfinite(embeddings).all(axis=1) & np.isin(
        cty_label,
        HRL_CTY_CLASS_CODES,
    )
    if include_topography:
        assert elevation_values is not None and slope_values is not None
        valid &= np.isfinite(elevation_values) & np.isfinite(slope_values)

    if not valid.any():
        raise ValueError(
            f"No valid same-year AlphaEarth-HRL CTY samples remain for region "
            f"{region_id}, year {year}."
        )

    feature_matrix, feature_names = build_alphaearth_crop_feature_matrix(
        embeddings[valid],
        longitude[valid],
        latitude[valid],
        include_coordinates=include_coordinates,
        include_topography=include_topography,
        elevation_m=(None if elevation_values is None else elevation_values[valid]),
        slope_gradient=(None if slope_values is None else slope_values[valid]),
    )
    samples = pd.DataFrame(feature_matrix, columns=feature_names)
    samples["cty_label"] = cty_label[valid]
    samples["year"] = int(year)
    samples["region_id"] = int(region_id)
    samples["country_iso3"] = str(country_iso3)
    samples["source_x"] = source_x[valid]
    samples["source_y"] = source_y[valid]
    return samples


def _create_alphaearth_crop_classifier(
    *,
    model_type: str,
    model_parameters: Mapping[str, Any] | None,
    random_seed: int,
    n_jobs: int,
) -> Any:
    """Construct one supported AlphaEarth CTY estimator.

    All estimators expose ``predict_proba`` and accept optional sample weights.
    Scaling is applied only to logistic regression; tree estimators operate on the
    normalized embedding coordinates directly.
    """
    parameters = dict(model_parameters or {})
    model_type = str(model_type).strip().lower()
    if model_type not in ALPHAEARTH_SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "model_type must be one of "
            f"{ALPHAEARTH_SUPPORTED_MODEL_TYPES}, found {model_type!r}."
        )

    if model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        C = float(parameters.get("C", 0.1))
        max_iter = int(parameters.get("max_iter", 1500))
        tolerance = float(parameters.get("tol", 1.0e-4))
        if C <= 0.0:
            raise ValueError("Logistic-regression C must be greater than zero.")
        if max_iter < 1:
            raise ValueError("Logistic-regression max_iter must be at least one.")
        if tolerance <= 0.0:
            raise ValueError("Logistic-regression tol must be greater than zero.")

        return Pipeline(
            (
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=C,
                        solver=str(parameters.get("solver", "lbfgs")),
                        max_iter=max_iter,
                        tol=tolerance,
                        random_state=int(random_seed),
                    ),
                ),
            )
        )

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        bootstrap = bool(parameters.get("bootstrap", True))
        max_samples = parameters.get("max_samples")
        if not bootstrap and max_samples is not None:
            raise ValueError("random_forest max_samples requires bootstrap=true.")
        return RandomForestClassifier(
            n_estimators=int(parameters.get("n_estimators", 800)),
            criterion=str(parameters.get("criterion", "gini")),
            max_depth=parameters.get("max_depth"),
            min_samples_split=int(parameters.get("min_samples_split", 2)),
            min_samples_leaf=int(parameters.get("min_samples_leaf", 1)),
            max_features=parameters.get("max_features", "sqrt"),
            max_samples=max_samples,
            bootstrap=bootstrap,
            n_jobs=int(n_jobs),
            random_state=int(random_seed),
            oob_score=False,
        )

    if model_type == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        bootstrap = bool(parameters.get("bootstrap", False))
        max_samples = parameters.get("max_samples")
        if not bootstrap and max_samples is not None:
            raise ValueError("extra_trees max_samples requires bootstrap=true.")
        return ExtraTreesClassifier(
            n_estimators=int(parameters.get("n_estimators", 800)),
            criterion=str(parameters.get("criterion", "gini")),
            max_depth=parameters.get("max_depth"),
            min_samples_split=int(parameters.get("min_samples_split", 2)),
            min_samples_leaf=int(parameters.get("min_samples_leaf", 1)),
            max_features=parameters.get("max_features", "sqrt"),
            max_samples=max_samples,
            bootstrap=bootstrap,
            n_jobs=int(n_jobs),
            random_state=int(random_seed),
        )

    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        learning_rate=float(parameters.get("learning_rate", 0.06)),
        max_iter=int(parameters.get("max_iter", 500)),
        max_leaf_nodes=parameters.get("max_leaf_nodes", 31),
        max_depth=parameters.get("max_depth"),
        min_samples_leaf=int(parameters.get("min_samples_leaf", 20)),
        l2_regularization=float(parameters.get("l2_regularization", 1.0)),
        max_features=float(parameters.get("max_features", 1.0)),
        max_bins=int(parameters.get("max_bins", 255)),
        early_stopping=parameters.get("early_stopping", "auto"),
        validation_fraction=float(parameters.get("validation_fraction", 0.10)),
        n_iter_no_change=int(parameters.get("n_iter_no_change", 20)),
        random_state=int(random_seed),
    )


def _fit_alphaearth_estimator(
    estimator: Any,
    features: np.ndarray,
    target: np.ndarray,
    *,
    sample_weight: np.ndarray | None,
) -> Any:
    """Fit one supported estimator with optional external sample weights."""
    target = np.asarray(target, dtype=np.int32)
    represented = np.unique(target)
    if represented.size == 1:
        return AlphaEarthConstantClassifier(int(represented[0]))

    if sample_weight is None:
        estimator.fit(features, target)
        return estimator

    weights = np.asarray(sample_weight, dtype=np.float64)
    if hasattr(estimator, "named_steps"):
        estimator.fit(features, target, classifier__sample_weight=weights)
    else:
        estimator.fit(features, target, sample_weight=weights)
    return estimator


def _class_region_year_balanced_weights(
    samples: pd.DataFrame,
    *,
    label_column: str,
) -> np.ndarray:
    """Balance classes and their represented region-year groups simultaneously.

    Every class receives equal total weight. Within a class, every represented
    ``(region_id, year)`` group also receives equal total weight, regardless of
    its raw number of sampled pixels. Returned weights are normalized to mean one.
    """
    required = {label_column, "region_id", "year"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(
            f"Cannot create class-region-year weights; missing {sorted(missing)}."
        )
    if samples.empty:
        raise ValueError("Cannot create sample weights for an empty table.")

    grouping = samples.groupby(
        [label_column, "region_id", "year"],
        sort=False,
        observed=True,
    )
    group_size = grouping[label_column].transform("size").to_numpy(dtype=np.float64)
    groups_per_class = (
        samples[[label_column, "region_id", "year"]]
        .drop_duplicates()
        .groupby(label_column, observed=True)
        .size()
    )
    class_group_count = (
        samples[label_column].map(groups_per_class).to_numpy(dtype=np.float64)
    )
    weights = 1.0 / (group_size * class_group_count)
    mean_weight = float(weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0.0:
        raise ValueError("Calculated invalid class-region-year sample weights.")
    return (weights / mean_weight).astype(np.float64, copy=False)


def _alphaearth_sample_weights(
    samples: pd.DataFrame,
    *,
    label_column: str,
    sample_weight_mode: str,
) -> np.ndarray | None:
    """Return the configured row weights for one estimator."""
    if sample_weight_mode == "none":
        return None
    if sample_weight_mode == "class_region_year_balanced":
        return _class_region_year_balanced_weights(
            samples,
            label_column=label_column,
        )
    if sample_weight_mode == "class_balanced":
        from sklearn.utils.class_weight import compute_sample_weight

        return np.asarray(
            compute_sample_weight(
                class_weight="balanced",
                y=samples[label_column].to_numpy(),
            ),
            dtype=np.float64,
        )
    raise ValueError(
        "sample_weight_mode must be 'none', 'class_balanced', or "
        "'class_region_year_balanced'."
    )


def _fit_alphaearth_model_core(
    samples: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    normalize_embeddings: bool,
    classifier_structure: str,
    residual_class_mode: str,
    model_type: str,
    model_parameters: Mapping[str, Any] | None,
    random_seed: int,
    sample_weight_mode: str,
    n_jobs: int,
    unclassified_probability_threshold: float,
    class_probability_thresholds: Mapping[int, float] | None,
) -> AlphaEarthCropModelBundle:
    """Fit one flat or soft-hierarchical classifier without calibration."""
    if classifier_structure not in {"flat", "hierarchical"}:
        raise ValueError("classifier_structure must be 'flat' or 'hierarchical'.")
    if residual_class_mode not in {"learned", "uncertainty"}:
        raise ValueError("residual_class_mode must be 'learned' or 'uncertainty'.")
    if classifier_structure == "hierarchical" and residual_class_mode != "uncertainty":
        raise ValueError(
            "Hierarchical classification requires residual_class_mode='uncertainty' "
            "because group 30 is generated from uncertainty rather than learned."
        )

    model_type = str(model_type).strip().lower()
    if model_type not in ALPHAEARTH_SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "model_type must be one of "
            f"{ALPHAEARTH_SUPPORTED_MODEL_TYPES}, found {model_type!r}."
        )
    resolved_parameters = dict(model_parameters or {})

    explicit_classes = (
        HRL_CTY_CLASS_CODES
        if residual_class_mode == "learned"
        else HRL_CTY_UNCERTAINTY_TRAINING_CLASS_CODES
    )
    normalized_thresholds = {
        int(code): float(value)
        for code, value in (class_probability_thresholds or {}).items()
    }
    for code, threshold in normalized_thresholds.items():
        if code not in HRL_CTY_CLASS_CODES:
            raise ValueError(f"Unknown CTY threshold class: {code}.")
        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                f"Class threshold for {code} must lie in (0, 1], found {threshold}."
            )

    def create_estimator(seed: int) -> Any:
        return _create_alphaearth_crop_classifier(
            model_type=model_type,
            model_parameters=resolved_parameters,
            random_seed=seed,
            n_jobs=n_jobs,
        )

    if classifier_structure == "flat":
        training = samples.loc[samples["cty_label"].isin(explicit_classes)].copy()
        if training.empty:
            raise ValueError("No explicit CTY samples remain for flat training.")

        features = alphaearth_features_from_samples(
            training,
            feature_names,
            normalize_embeddings=normalize_embeddings,
        )
        target = training["cty_label"].to_numpy(dtype=np.int32)
        weights = _alphaearth_sample_weights(
            training,
            label_column="cty_label",
            sample_weight_mode=sample_weight_mode,
        )
        estimator = _fit_alphaearth_estimator(
            create_estimator(random_seed),
            features,
            target,
            sample_weight=weights,
        )
        represented_classes = tuple(
            int(code) for code in np.asarray(estimator.classes_, dtype=np.int32)
        )
        return AlphaEarthCropModelBundle(
            cty_model=estimator,
            feature_names=feature_names,
            model_type=model_type,
            model_parameters=resolved_parameters,
            trained_cty_classes=represented_classes,
            normalize_embeddings=normalize_embeddings,
            classifier_structure=classifier_structure,
            residual_class_mode=residual_class_mode,
            unclassified_probability_threshold=unclassified_probability_threshold,
            class_probability_thresholds=normalized_thresholds,
        )

    hierarchical = samples.copy()
    hierarchical["group_label"] = _map_hrl_cty_to_crop_group(
        hierarchical["cty_label"].to_numpy(dtype=np.int32)
    )
    group_training = hierarchical.loc[
        hierarchical["group_label"].isin(HRL_CTY_HIERARCHICAL_GROUP_CODES)
    ].copy()
    if group_training.empty:
        raise ValueError("No supported crop-group samples remain for training.")

    group_features = alphaearth_features_from_samples(
        group_training,
        feature_names,
        normalize_embeddings=normalize_embeddings,
    )
    group_target = group_training["group_label"].to_numpy(dtype=np.int32)
    group_weights = _alphaearth_sample_weights(
        group_training,
        label_column="group_label",
        sample_weight_mode=sample_weight_mode,
    )
    group_estimator = _fit_alphaearth_estimator(
        create_estimator(random_seed),
        group_features,
        group_target,
        sample_weight=group_weights,
    )

    within_group_models: dict[int, Any] = {}
    represented_native: list[int] = []
    for group_code in HRL_CTY_HIERARCHICAL_GROUP_CODES:
        native_codes = tuple(
            code
            for code in HRL_CTY_UNCERTAINTY_TRAINING_CLASS_CODES
            if HRL_CTY_CROP_GROUP_MAP[code] == group_code
        )
        branch = hierarchical.loc[hierarchical["cty_label"].isin(native_codes)].copy()
        if branch.empty:
            continue

        branch_features = alphaearth_features_from_samples(
            branch,
            feature_names,
            normalize_embeddings=normalize_embeddings,
        )
        branch_target = branch["cty_label"].to_numpy(dtype=np.int32)
        represented = np.unique(branch_target)
        represented_native.extend(int(code) for code in represented)

        if represented.size == 1:
            within_group_models[int(group_code)] = AlphaEarthConstantClassifier(
                int(represented[0])
            )
            continue

        branch_weights = _alphaearth_sample_weights(
            branch,
            label_column="cty_label",
            sample_weight_mode=sample_weight_mode,
        )
        within_group_models[int(group_code)] = _fit_alphaearth_estimator(
            create_estimator(random_seed + int(group_code)),
            branch_features,
            branch_target,
            sample_weight=branch_weights,
        )

    return AlphaEarthCropModelBundle(
        cty_model=None,
        feature_names=feature_names,
        model_type=model_type,
        model_parameters=resolved_parameters,
        trained_cty_classes=tuple(sorted(set(represented_native))),
        normalize_embeddings=normalize_embeddings,
        classifier_structure=classifier_structure,
        residual_class_mode=residual_class_mode,
        unclassified_probability_threshold=unclassified_probability_threshold,
        class_probability_thresholds=normalized_thresholds,
        group_model=group_estimator,
        within_group_models=within_group_models,
    )


def predict_alphaearth_crop_probabilities(
    models: AlphaEarthCropModelBundle,
    features: np.ndarray,
) -> np.ndarray:
    """Return probabilities in the complete fixed native-CTY class order."""
    features = np.asarray(features, dtype=np.float32)
    if models.classifier_structure == "flat":
        if models.cty_model is None:
            raise ValueError("Flat AlphaEarth model bundle has no CTY estimator.")
        return _predict_full_class_probabilities(
            models.cty_model,
            features,
            models.cty_classes,
        )

    if models.group_model is None:
        raise ValueError("Hierarchical AlphaEarth model bundle has no group estimator.")
    group_probabilities = _predict_full_class_probabilities(
        models.group_model,
        features,
        HRL_CTY_HIERARCHICAL_GROUP_CODES,
    )
    native_probabilities = np.zeros(
        (features.shape[0], len(models.cty_classes)),
        dtype=np.float32,
    )
    native_positions = {
        int(code): index for index, code in enumerate(models.cty_classes)
    }
    group_positions = {
        int(code): index for index, code in enumerate(HRL_CTY_HIERARCHICAL_GROUP_CODES)
    }

    for group_code, branch_model in models.within_group_models.items():
        branch_classes = tuple(
            int(code) for code in np.asarray(branch_model.classes_, dtype=np.int32)
        )
        conditional = _predict_full_class_probabilities(
            branch_model,
            features,
            branch_classes,
        )
        group_probability = group_probabilities[:, group_positions[int(group_code)]]
        for branch_index, class_code in enumerate(branch_classes):
            native_probabilities[:, native_positions[class_code]] = (
                group_probability * conditional[:, branch_index]
            )

    row_sums = native_probabilities.sum(axis=1)
    valid = row_sums > 0.0
    native_probabilities[valid] /= row_sums[valid, None]
    return native_probabilities


def alphaearth_cty_from_probabilities(
    models: AlphaEarthCropModelBundle,
    probabilities: np.ndarray,
    *,
    unclassified_probability_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert native probabilities to CTY classes and confidence.

    Class-specific thresholds influence the multiclass decision through
    ``probability / threshold``. A lower calibrated threshold therefore permits
    conservative classes to compete more readily. Predictions below their winning
    class threshold are mapped to 1150, 3100, or 3200 in uncertainty mode.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(models.cty_classes):
        raise ValueError("CTY probability matrix does not match the model schema.")
    global_threshold = (
        models.unclassified_probability_threshold
        if unclassified_probability_threshold is None
        else float(unclassified_probability_threshold)
    )
    if not 0.0 <= global_threshold <= 1.0:
        raise ValueError("unclassified_probability_threshold must be in [0, 1].")

    classes = np.asarray(models.cty_classes, dtype=np.int32)
    thresholds = np.asarray(
        [
            float(models.class_probability_thresholds.get(int(code), global_threshold))
            for code in classes
        ],
        dtype=np.float32,
    )
    thresholds = np.clip(thresholds, 1.0e-6, 1.0)
    decision_scores = probabilities / thresholds[None, :]
    best_indices = np.argmax(decision_scores, axis=1)
    predicted = classes[best_indices].copy()
    confidence = probabilities[np.arange(probabilities.shape[0]), best_indices]
    winning_threshold = thresholds[best_indices]

    low_confidence = confidence < winning_threshold
    if models.residual_class_mode == "learned":
        low_confidence &= ~np.isin(predicted, HRL_CTY_RESIDUAL_CLASS_CODES)
    if low_confidence.any():
        cereal = low_confidence & np.isin(
            predicted,
            HRL_CTY_CEREAL_EXPLICIT_CLASS_CODES,
        )
        annual = low_confidence & np.isin(
            predicted,
            HRL_ANNUAL_CTY_CLASS_CODES,
        )
        permanent = low_confidence & np.isin(
            predicted,
            HRL_PERMANENT_CTY_CLASS_CODES,
        )
        predicted[cereal] = 1150
        predicted[annual & ~cereal] = 3100
        predicted[permanent] = 3200

    return predicted.astype(np.int32), confidence.astype(np.float32)


def predict_alphaearth_crop_models(
    models: AlphaEarthCropModelBundle,
    samples: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict CTY classes and confidence for a sample table."""
    features = alphaearth_features_from_samples(
        samples,
        models.feature_names,
        normalize_embeddings=models.normalize_embeddings,
    )
    probabilities = predict_alphaearth_crop_probabilities(models, features)
    return alphaearth_cty_from_probabilities(models, probabilities)


def calibrate_alphaearth_class_thresholds(
    models: AlphaEarthCropModelBundle,
    calibration_samples: pd.DataFrame,
    *,
    threshold_grid: Sequence[float],
    minimum_reference_samples: int = 25,
) -> dict[int, float]:
    """Calibrate per-class probability thresholds on a prior complete year.

    Each explicit class receives the grid threshold maximizing its one-vs-rest F1.
    During multiclass inference the resulting thresholds are used as relative
    decision scales and acceptance thresholds.
    """
    from sklearn.metrics import f1_score

    if calibration_samples.empty:
        return {}
    grid = np.asarray(tuple(float(value) for value in threshold_grid), dtype=np.float64)
    if grid.size == 0 or np.any((grid <= 0.0) | (grid > 1.0)):
        raise ValueError("threshold_grid values must lie in (0, 1].")

    features = alphaearth_features_from_samples(
        calibration_samples,
        models.feature_names,
        normalize_embeddings=models.normalize_embeddings,
    )
    probabilities = predict_alphaearth_crop_probabilities(models, features)
    observed = calibration_samples["cty_label"].to_numpy(dtype=np.int32)
    positions = {int(code): index for index, code in enumerate(models.cty_classes)}
    calibrated: dict[int, float] = {}

    for class_code in models.trained_cty_classes:
        reference = observed == int(class_code)
        if int(reference.sum()) < int(minimum_reference_samples):
            continue
        class_probability = probabilities[:, positions[int(class_code)]]
        scores = np.asarray(
            [
                f1_score(
                    reference,
                    class_probability >= threshold,
                    zero_division=0,
                )
                for threshold in grid
            ],
            dtype=np.float64,
        )
        best_score = float(scores.max())
        candidates = grid[np.isclose(scores, best_score)]
        calibrated[int(class_code)] = float(
            candidates[
                np.argmin(
                    np.abs(candidates - models.unclassified_probability_threshold)
                )
            ]
        )
    return calibrated


def fit_alphaearth_crop_models(
    samples: pd.DataFrame,
    *,
    include_coordinates: bool = False,
    include_topography: bool = False,
    normalize_embeddings: bool = True,
    classifier_structure: str = "flat",
    residual_class_mode: str = "uncertainty",
    model_type: str = "logistic_regression",
    model_parameters: Mapping[str, Any] | None = None,
    random_seed: int = 42,
    sample_weight_mode: str = "class_region_year_balanced",
    n_jobs: int = -1,
    unclassified_probability_threshold: float = 0.25,
    class_probability_thresholds: Mapping[int, float] | None = None,
    calibrate_class_thresholds: bool = False,
    class_threshold_grid: Sequence[float] = (
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
    ),
    calibration_min_reference_samples: int = 25,
) -> AlphaEarthCropModelBundle:
    """Fit one flat or soft-hierarchical target-year CTY classifier.

    Supported estimator families are multinomial logistic regression, Random
    Forest, Extra Trees, and histogram gradient boosting. All families use the
    same target-year predictor schema, temporal split, sample weighting, residual
    handling, and optional threshold calibration.

    When threshold calibration is enabled, the latest year in ``samples`` is used
    only to calibrate a temporary model fitted on earlier years. The final estimator
    is then refitted on every supplied row.
    """
    if sample_weight_mode not in {
        "none",
        "class_balanced",
        "class_region_year_balanced",
    }:
        raise ValueError(
            "sample_weight_mode must be 'none', 'class_balanced', or "
            "'class_region_year_balanced'."
        )
    if samples.empty:
        raise ValueError("Cannot train a CTY classifier from an empty sample table.")

    model_type = str(model_type).strip().lower()
    if model_type not in ALPHAEARTH_SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "model_type must be one of "
            f"{ALPHAEARTH_SUPPORTED_MODEL_TYPES}, found {model_type!r}."
        )
    if model_parameters is not None and not isinstance(model_parameters, Mapping):
        raise TypeError("model_parameters must be a mapping or None.")

    alphaearth_embedding_diagnostics(samples)
    feature_names = alphaearth_crop_feature_names(
        include_coordinates=include_coordinates,
        include_topography=include_topography,
    )
    required_columns = set(feature_names) | {"cty_label"}
    missing_columns = required_columns - set(samples.columns)
    if missing_columns:
        raise ValueError(
            f"AlphaEarth training samples are missing columns: {sorted(missing_columns)}"
        )

    thresholds = {
        int(code): float(value)
        for code, value in (class_probability_thresholds or {}).items()
    }
    if calibrate_class_thresholds:
        if "year" not in samples.columns:
            raise ValueError("Threshold calibration requires a year column.")
        calibration_year = int(samples["year"].max())
        calibration_samples = samples.loc[samples["year"] == calibration_year].copy()
        prior_samples = samples.loc[samples["year"] < calibration_year].copy()
        if not prior_samples.empty and not calibration_samples.empty:
            temporary = _fit_alphaearth_model_core(
                prior_samples,
                feature_names=feature_names,
                normalize_embeddings=normalize_embeddings,
                classifier_structure=classifier_structure,
                residual_class_mode=residual_class_mode,
                model_type=model_type,
                model_parameters=model_parameters,
                random_seed=random_seed,
                sample_weight_mode=sample_weight_mode,
                n_jobs=n_jobs,
                unclassified_probability_threshold=(unclassified_probability_threshold),
                class_probability_thresholds=None,
            )
            calibrated = calibrate_alphaearth_class_thresholds(
                temporary,
                calibration_samples,
                threshold_grid=class_threshold_grid,
                minimum_reference_samples=calibration_min_reference_samples,
            )
            calibrated.update(thresholds)
            thresholds = calibrated

    return _fit_alphaearth_model_core(
        samples,
        feature_names=feature_names,
        normalize_embeddings=normalize_embeddings,
        classifier_structure=classifier_structure,
        residual_class_mode=residual_class_mode,
        model_type=model_type,
        model_parameters=model_parameters,
        random_seed=random_seed,
        sample_weight_mode=sample_weight_mode,
        n_jobs=n_jobs,
        unclassified_probability_threshold=unclassified_probability_threshold,
        class_probability_thresholds=thresholds,
    )


def _map_hrl_cty_to_crop_group(values: np.ndarray) -> np.ndarray:
    """Map native CTY codes to the HRL crop-group accuracy classes."""
    values = np.asarray(values, dtype=np.int32)
    mapped = np.full(values.shape, -1, dtype=np.int32)
    for source_code, target_code in HRL_CTY_CROP_GROUP_MAP.items():
        mapped[values == source_code] = target_code
    return mapped


def _map_hrl_cty_to_aggregation_level_1(values: np.ndarray) -> np.ndarray:
    """Map native CTY codes to the HRL aggregation-level-1 classes."""
    values = np.asarray(values, dtype=np.int32)
    mapped = np.full(values.shape, -1, dtype=np.int32)
    for source_code, target_code in HRL_CTY_AGGREGATION_LEVEL_1_MAP.items():
        mapped[values == source_code] = target_code
    return mapped


def _append_remote_sensing_accuracy_assessment(
    *,
    split_name: str,
    target_name: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    labels: tuple[int, ...] | np.ndarray,
    metric_rows: list[dict[str, float | int | str]],
    confusion_rows: list[dict[str, int | str]],
    confusion_labels: tuple[int, ...] | np.ndarray | None = None,
) -> None:
    """Append classic thematic-map accuracy statistics for one target."""
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    observed = np.asarray(observed, dtype=np.int32)
    predicted = np.asarray(predicted, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    matrix_labels = (
        labels
        if confusion_labels is None
        else np.asarray(confusion_labels, dtype=np.int32)
    )
    if observed.size == 0:
        return

    user_accuracy, producer_accuracy, f_score, support = (
        precision_recall_fscore_support(
            observed,
            predicted,
            labels=labels,
            zero_division=0,
        )
    )
    product_support = np.asarray(
        [(predicted == class_code).sum() for class_code in labels],
        dtype=np.int64,
    )
    reference_present = support > 0
    if not reference_present.any():
        return
    macro_f_score_observed = float(np.mean(f_score[reference_present]))
    balanced_accuracy_observed = float(np.mean(producer_accuracy[reference_present]))
    observed_class_count = int(reference_present.sum())
    configured_class_count = int(labels.size)
    predicted_only_class_count = int(((support == 0) & (product_support > 0)).sum())

    for (
        class_code,
        class_ua,
        class_pa,
        class_f_score,
        reference_support,
        mapped_support,
    ) in zip(
        labels,
        user_accuracy,
        producer_accuracy,
        f_score,
        support,
        product_support,
        strict=True,
    ):
        metric_rows.append(
            {
                "split": split_name,
                "target": target_name,
                "metric_scope": "class",
                "class_code": int(class_code),
                "reference_support": int(reference_support),
                "product_support": int(mapped_support),
                "producer_accuracy": float(class_pa),
                "user_accuracy": float(class_ua),
                "omission_error": float(1.0 - class_pa),
                "commission_error": float(1.0 - class_ua),
                "f_score": float(class_f_score),
                # Backward-compatible aliases used by earlier tables.
                "precision": float(class_ua),
                "recall": float(class_pa),
                "f1": float(class_f_score),
                "support": int(reference_support),
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "kappa": np.nan,
            }
        )

    metric_rows.append(
        {
            "split": split_name,
            "target": target_name,
            "metric_scope": "summary",
            "class_code": -1,
            "reference_support": int(observed.size),
            "product_support": int(predicted.size),
            "producer_accuracy": np.nan,
            "user_accuracy": np.nan,
            "omission_error": np.nan,
            "commission_error": np.nan,
            # Retain the earlier all-configured-class macro F-score for
            # backwards compatibility in stored tables. The concise console
            # report uses macro_f_score_observed because region-specific
            # assessments often contain no reference samples for several CTY
            # classes.
            "f_score": float(np.mean(f_score)),
            "macro_f_score_observed": macro_f_score_observed,
            "precision": np.nan,
            "recall": np.nan,
            "f1": float(np.mean(f_score)),
            "support": int(observed.size),
            "accuracy": float(accuracy_score(observed, predicted)),
            "balanced_accuracy": balanced_accuracy_observed,
            "kappa": float(cohen_kappa_score(observed, predicted)),
            "observed_class_count": observed_class_count,
            "configured_class_count": configured_class_count,
            "predicted_only_class_count": predicted_only_class_count,
        }
    )

    # sklearn returns Reference rows x Product columns. The HRL PUM presents
    # Product rows x Reference columns, so transpose it for storage/reporting.
    product_by_reference = confusion_matrix(
        observed,
        predicted,
        labels=matrix_labels,
    ).T
    for product_index, product_class in enumerate(matrix_labels):
        for reference_index, reference_class in enumerate(matrix_labels):
            count = int(product_by_reference[product_index, reference_index])
            confusion_rows.append(
                {
                    "split": split_name,
                    "target": target_name,
                    "product_class": int(product_class),
                    "reference_class": int(reference_class),
                    # Backward-compatible aliases.
                    "predicted_class": int(product_class),
                    "observed_class": int(reference_class),
                    "count": count,
                }
            )


def evaluate_alphaearth_crop_predictions(
    samples: pd.DataFrame,
    predicted_cty: np.ndarray,
    *,
    split_name: str,
    cty_classes: Sequence[int] = HRL_CTY_CLASS_CODES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate aligned CTY predictions against stored HRL samples.

    This function is used for both ordinary classifier predictions and values
    sampled from fully post-processed raster products. Predictions equal to the
    HRL outside-area code, or otherwise outside the supported CTY schema, are
    excluded explicitly and counted in the summary rows.

    Args:
        samples: Evaluation sample table containing CTY references.
        predicted_cty: Final CTY prediction per sample.
        split_name: Display name such as ``"validation"`` or ``"test"``.
        cty_classes: Native CTY classes included in the assessment.

    Returns:
        Metric and Product-by-Reference confusion-matrix tables.
    """
    if samples.empty:
        raise ValueError(f"Cannot evaluate an empty {split_name!r} sample split.")

    predicted_cty = np.asarray(predicted_cty, dtype=np.int32).reshape(-1)
    if predicted_cty.shape != (len(samples),):
        raise ValueError(
            "predicted_cty must contain one value per sample; found "
            f"{predicted_cty.shape} for {len(samples)} samples."
        )

    observed_cty_all = samples["cty_label"].to_numpy(dtype=np.int32)
    valid_cty_prediction = np.isin(predicted_cty, np.asarray(cty_classes))
    if not valid_cty_prediction.any():
        raise ValueError(
            f"No valid CTY predictions remain for {split_name!r} evaluation."
        )

    observed_cty = observed_cty_all[valid_cty_prediction]
    evaluated_cty = predicted_cty[valid_cty_prediction]
    metric_rows: list[dict[str, float | int | str]] = []
    confusion_rows: list[dict[str, int | str]] = []

    _append_remote_sensing_accuracy_assessment(
        split_name=split_name,
        target_name="CTY_CROP_GROUP",
        observed=_map_hrl_cty_to_crop_group(observed_cty),
        predicted=_map_hrl_cty_to_crop_group(evaluated_cty),
        labels=HRL_CTY_CROP_GROUP_CLASS_CODES,
        metric_rows=metric_rows,
        confusion_rows=confusion_rows,
    )

    observed_level_1 = _map_hrl_cty_to_aggregation_level_1(observed_cty)
    predicted_level_1 = _map_hrl_cty_to_aggregation_level_1(evaluated_cty)
    level_1_reference_mask = observed_level_1 != -1
    _append_remote_sensing_accuracy_assessment(
        split_name=split_name,
        target_name="CTY_AGGREGATION_LEVEL_1",
        observed=observed_level_1[level_1_reference_mask],
        predicted=predicted_level_1[level_1_reference_mask],
        labels=HRL_CTY_AGGREGATION_LEVEL_1_CLASS_CODES,
        confusion_labels=(-1, *HRL_CTY_AGGREGATION_LEVEL_1_CLASS_CODES),
        metric_rows=metric_rows,
        confusion_rows=confusion_rows,
    )
    _append_remote_sensing_accuracy_assessment(
        split_name=split_name,
        target_name="CTY",
        observed=observed_cty,
        predicted=evaluated_cty,
        labels=np.asarray(cty_classes, dtype=np.int32),
        metric_rows=metric_rows,
        confusion_rows=confusion_rows,
    )

    metrics = pd.DataFrame(metric_rows)
    confusion = pd.DataFrame(confusion_rows)
    if not metrics.empty:
        excluded_by_target = {
            "CTY_CROP_GROUP": int((~valid_cty_prediction).sum()),
            "CTY_AGGREGATION_LEVEL_1": int((~valid_cty_prediction).sum()),
            "CTY": int((~valid_cty_prediction).sum()),
        }
        metrics["excluded_predictions"] = np.nan
        summary_mask = metrics["metric_scope"].eq("summary")
        metrics.loc[summary_mask, "excluded_predictions"] = (
            metrics.loc[summary_mask, "target"].map(excluded_by_target).astype(float)
        )
    return metrics, confusion


def evaluate_alphaearth_crop_models(
    models: AlphaEarthCropModelBundle,
    samples: pd.DataFrame,
    *,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one fitted CTY bundle with its configured decision rules.

    Native CTY, official crop-group, and aggregation-level-1 diagnostics are
    returned. Class-specific thresholds, residual-class generation, hierarchy,
    and embedding normalization are applied exactly as they are during raster
    inference.
    """
    if samples.empty:
        raise ValueError(f"Cannot evaluate an empty {split_name!r} sample split.")

    predicted_cty, _ = predict_alphaearth_crop_models(models, samples)
    return evaluate_alphaearth_crop_predictions(
        samples,
        predicted_cty,
        split_name=split_name,
        cty_classes=HRL_CTY_CLASS_CODES,
    )


def _accuracy_class_name(target: str, class_code: int) -> str:
    """Return a readable CTY class label for an accuracy report."""
    if target == "CTY_CROP_GROUP":
        return HRL_CTY_CROP_GROUP_NAMES.get(class_code, str(class_code))
    if target == "CTY_AGGREGATION_LEVEL_1":
        return HRL_CTY_AGGREGATION_LEVEL_1_NAMES.get(
            class_code,
            str(class_code),
        )
    return HRL_CTY_CLASS_NAMES.get(class_code, str(class_code))


def _alphaearth_summary_macro_f_score(
    target_metrics: pd.DataFrame,
) -> float:
    """Return macro F1 over classes represented in the reference data."""
    summary = target_metrics.loc[target_metrics["metric_scope"] == "summary"].iloc[0]
    stored = summary.get("macro_f_score_observed", np.nan)
    if pd.notna(stored):
        return float(stored)

    class_metrics = target_metrics.loc[
        (target_metrics["metric_scope"] == "class")
        & (target_metrics["reference_support"] > 0)
    ]
    if class_metrics.empty:
        return float("nan")
    return float(class_metrics["f_score"].mean())


def _alphaearth_summary_observed_class_count(
    target_metrics: pd.DataFrame,
) -> tuple[int, int]:
    """Return represented and configured class counts for one assessment."""
    summary = target_metrics.loc[target_metrics["metric_scope"] == "summary"].iloc[0]
    observed = summary.get("observed_class_count", np.nan)
    configured = summary.get("configured_class_count", np.nan)
    if pd.notna(observed) and pd.notna(configured):
        return int(observed), int(configured)

    class_metrics = target_metrics.loc[target_metrics["metric_scope"] == "class"]
    return (
        int((class_metrics["reference_support"] > 0).sum()),
        int(len(class_metrics)),
    )


def format_alphaearth_accuracy_report(
    metrics: pd.DataFrame,
    confusion: pd.DataFrame | None = None,
) -> str:
    """Format a compact raw CTY accuracy summary for console logging.

    The full per-class statistics and Product-by-Reference confusion matrices
    remain available in the stored parquet tables. Console output deliberately
    reports only the native CTY result and the coarser crop-group diagnostic.

    Args:
        metrics: Accuracy metric table returned by the CTY evaluation helpers.
        confusion: Retained for API compatibility. Confusion matrices are not
            printed to the console.

    Returns:
        Compact explanatory accuracy summary.
    """
    del confusion
    if metrics.empty:
        return "No AlphaEarth CTY-classification accuracy statistics available."

    split_order = [
        split for split in ("validation", "test") if split in set(metrics["split"])
    ]
    split_order.extend(sorted(set(metrics["split"]) - set(split_order)))
    target_order = ("CTY", "CTY_CROP_GROUP")
    target_titles = {
        "CTY": "Native CTY (primary)",
        "CTY_CROP_GROUP": "Crop group (coarse)",
    }

    rows: list[dict[str, str]] = []
    include_year = "evaluation_year" in metrics.columns
    for split in split_order:
        split_metrics = metrics.loc[metrics["split"] == split]
        for target in target_order:
            target_metrics = split_metrics.loc[split_metrics["target"] == target]
            if target_metrics.empty:
                continue
            summary = target_metrics.loc[
                target_metrics["metric_scope"] == "summary"
            ].iloc[0]
            observed_classes, configured_classes = (
                _alphaearth_summary_observed_class_count(target_metrics)
            )
            row = {
                "Split": split.capitalize(),
                "Level": target_titles[target],
                "Samples": f"{int(summary['reference_support']):,}",
                "OA": f"{float(summary['accuracy']):.2%}",
                "Macro F1*": (
                    f"{_alphaearth_summary_macro_f_score(target_metrics):.2%}"
                ),
                "Mean PA": f"{float(summary['balanced_accuracy']):.2%}",
                "Classes": f"{observed_classes}/{configured_classes}",
            }
            if include_year and pd.notna(summary.get("evaluation_year", np.nan)):
                row = {
                    "Year": str(int(summary["evaluation_year"])),
                    **row,
                }
            rows.append(row)

    if not rows:
        return "No AlphaEarth CTY summary rows are available."

    table = pd.DataFrame(rows).to_string(index=False)
    return "\n".join(
        [
            table,
            "",
            "Interpretation:",
            "  • Native CTY is the primary detailed crop-class result.",
            "  • Crop group is a coarser diagnostic and is normally higher.",
            "  • OA is the share of evaluated samples classified correctly.",
            "  • Macro F1* averages class F1 only over classes present in the reference.",
            "  • Mean PA is mean Producer's Accuracy over reference-present classes.",
            "  • Classes reports reference-present/configured classes.",
            "  • Raw per-class tables are omitted here to avoid repetition.",
            "  • Final crop-group and native-crop class tables are reported once in stage 2.",
            "  • Complete raw/final confusion matrices remain stored in parquet tables.",
        ]
    )


def _format_alphaearth_training_years(value: Any) -> str:
    """Format comma-separated training years as a compact range when possible."""
    years = [int(part) for part in str(value).replace(" ", "").split(",") if part]
    if not years:
        return "—"
    if years == list(range(years[0], years[-1] + 1)):
        return str(years[0]) if len(years) == 1 else f"{years[0]}–{years[-1]}"
    return ",".join(str(year) for year in years)


def _format_alphaearth_class_support(
    reference_support: int,
    product_support: int,
) -> str:
    """Format reference and product sample counts for one accuracy class."""
    return f"{reference_support:,}/{product_support:,}"


def _format_alphaearth_class_percentage(
    value: Any,
    *,
    weak: bool = False,
) -> str:
    """Format one class accuracy percentage and optionally flag a weak F-score."""
    if value is None or pd.isna(value):
        return "—"
    rendered = f"{float(value):.1%}"
    return f"{rendered}!" if weak else rendered


def _format_alphaearth_final_class_table(
    metrics: pd.DataFrame,
    *,
    target: str,
    title: str,
) -> str:
    """Format final per-class accuracy for all held-out years in one table.

    A class is included when it occurs in the reference data for at least one
    held-out year. Reference-absent cells are rendered as an em dash rather than
    as a misleading zero-percent accuracy.
    """
    class_metrics = metrics.loc[
        (metrics["assessment_stage"] == "final_postprocessed")
        & (metrics["target"] == target)
        & (metrics["metric_scope"] == "class")
    ].copy()
    if class_metrics.empty:
        return f"{title}\nNo final per-class statistics available."

    years = sorted(
        int(year) for year in class_metrics["evaluation_year"].dropna().unique()
    )
    represented = class_metrics.loc[class_metrics["reference_support"] > 0]
    class_codes = sorted(int(code) for code in represented["class_code"].unique())
    if not class_codes:
        return f"{title}\nNo reference-present classes available."

    rows: list[dict[str, str]] = []
    for class_code in class_codes:
        row: dict[str, str] = {
            "Code": str(class_code),
            "Class": _accuracy_class_name(target, class_code),
        }
        for year in years:
            selected = class_metrics.loc[
                (class_metrics["evaluation_year"] == year)
                & (class_metrics["class_code"] == class_code)
            ]
            if selected.empty or int(selected.iloc[0]["reference_support"]) == 0:
                row[f"{year} Ref/Pred"] = "—"
                row[f"{year} PA"] = "—"
                row[f"{year} UA"] = "—"
                row[f"{year} F1"] = "—"
                continue

            values = selected.iloc[0]
            reference_support = int(values["reference_support"])
            product_support = int(values["product_support"])
            weak = reference_support >= 100 and float(values["f_score"]) < 0.50
            row[f"{year} Ref/Pred"] = _format_alphaearth_class_support(
                reference_support,
                product_support,
            )
            row[f"{year} PA"] = _format_alphaearth_class_percentage(
                values["producer_accuracy"]
            )
            row[f"{year} UA"] = _format_alphaearth_class_percentage(
                values["user_accuracy"]
            )
            row[f"{year} F1"] = _format_alphaearth_class_percentage(
                values["f_score"],
                weak=weak,
            )
        rows.append(row)

    table = pd.DataFrame(rows).to_string(index=False)
    return f"{title}\n{table}"


def format_alphaearth_postprocessed_accuracy_report(
    metrics: pd.DataFrame,
) -> str:
    """Format leakage-safe map accuracy with non-redundant class tables.

    Console output contains:

    1. One raw-versus-final summary for native CTY and crop-group accuracy.
    2. One final crop-group class table with all held-out years side by side.
    3. One final native CTY class table with all held-out years side by side.

    Aggregation-level-1 diagnostics and all confusion matrices remain available
    in the parquet outputs but are not repeated in the console.
    """
    if metrics.empty:
        return "No post-processed AlphaEarth CTY accuracy statistics available."

    summary = metrics.loc[metrics["metric_scope"] == "summary"].copy()
    required_stages = {"raw_rolling_origin", "final_postprocessed"}
    if not required_stages.issubset(set(summary["assessment_stage"])):
        return "Raw and final post-processed summary rows are not both available."

    target_order = ("CTY", "CTY_CROP_GROUP")
    target_titles = {
        "CTY": "Native CTY (primary)",
        "CTY_CROP_GROUP": "Crop group (coarse)",
    }
    rows: list[dict[str, str]] = []

    group_columns = ["split", "evaluation_year", "target", "training_years"]
    for keys, group in summary.groupby(group_columns, sort=False, dropna=False):
        split, evaluation_year, target, training_years = keys
        if target not in target_order:
            continue
        raw_rows = group.loc[group["assessment_stage"] == "raw_rolling_origin"]
        final_rows = group.loc[group["assessment_stage"] == "final_postprocessed"]
        if raw_rows.empty or final_rows.empty:
            continue
        raw = raw_rows.iloc[0]
        final = final_rows.iloc[0]

        target_metrics = metrics.loc[
            (metrics["split"] == split)
            & (metrics["evaluation_year"] == evaluation_year)
            & (metrics["target"] == target)
            & (metrics["training_years"] == training_years)
        ]
        raw_target_metrics = target_metrics.loc[
            target_metrics["assessment_stage"] == "raw_rolling_origin"
        ]
        final_target_metrics = target_metrics.loc[
            target_metrics["assessment_stage"] == "final_postprocessed"
        ]
        raw_f1 = _alphaearth_summary_macro_f_score(raw_target_metrics)
        final_f1 = _alphaearth_summary_macro_f_score(final_target_metrics)
        observed_classes, configured_classes = _alphaearth_summary_observed_class_count(
            final_target_metrics
        )

        rows.append(
            {
                "Year": str(int(evaluation_year)),
                "Split": str(split).capitalize(),
                "Level": target_titles[target],
                "Train": _format_alphaearth_training_years(training_years),
                "Final N": f"{int(final['reference_support']):,}",
                "Excluded": f"{int(final.get('excluded_predictions', 0) or 0):,}",
                "Raw OA": f"{float(raw['accuracy']):.2%}",
                "Final OA": f"{float(final['accuracy']):.2%}",
                "ΔOA": f"{float(final['accuracy'] - raw['accuracy']):+.2%}",
                "Raw F1*": f"{raw_f1:.2%}",
                "Final F1*": f"{final_f1:.2%}",
                "ΔF1": f"{final_f1 - raw_f1:+.2%}",
                "Classes": f"{observed_classes}/{configured_classes}",
            }
        )

    if not rows:
        return "No comparable raw and final CTY summary rows are available."

    order = {target: index for index, target in enumerate(target_order)}
    rows.sort(
        key=lambda row: (
            int(row["Year"]),
            order["CTY" if row["Level"].startswith("Native") else "CTY_CROP_GROUP"],
        )
    )
    summary_table = pd.DataFrame(rows).to_string(index=False)

    crop_group_table = _format_alphaearth_final_class_table(
        metrics,
        target="CTY_CROP_GROUP",
        title="FINAL MAP CLASS ACCURACY — CROP GROUPS",
    )
    native_table = _format_alphaearth_final_class_table(
        metrics,
        target="CTY",
        title="FINAL MAP CLASS ACCURACY — NATIVE CTY CROPS",
    )

    return "\n".join(
        [
            "OVERALL RAW-VERSUS-FINAL ACCURACY",
            summary_table,
            "",
            "The two class tables below report final post-processed maps only.",
            "Years are shown side by side so each class is listed once.",
            "Ref/Pred = reference sample count / predicted sample count.",
            "PA = Producer's Accuracy (reference recall); UA = User's Accuracy "
            "(prediction precision).",
            "F1 balances PA and UA; ! marks F1 < 50% where reference n ≥ 100.",
            "An em dash means that the class was absent from that year's reference.",
            "",
            crop_group_table,
            "",
            native_table,
            "",
            "Aggregation-level-1 results and complete raw/final confusion matrices "
            "remain stored in parquet tables.",
        ]
    )


def _alphaearth_estimator_feature_importance(
    estimator: Any,
    feature_names: Sequence[str],
    *,
    target: str,
) -> pd.DataFrame:
    """Extract normalized absolute logistic-regression coefficient magnitudes."""
    fitted = (
        estimator.named_steps["classifier"]
        if hasattr(estimator, "named_steps")
        else estimator
    )
    importance = getattr(fitted, "feature_importances_", None)
    if importance is None:
        coefficients = getattr(fitted, "coef_", None)
        if coefficients is not None:
            importance = np.mean(np.abs(np.asarray(coefficients)), axis=0)
    if importance is None:
        return pd.DataFrame(columns=["target", "feature", "importance"])
    values = np.asarray(importance, dtype=np.float64).reshape(-1)
    if values.size != len(feature_names):
        return pd.DataFrame(columns=["target", "feature", "importance"])
    total = float(values.sum())
    if total > 0.0:
        values = values / total
    return pd.DataFrame(
        {
            "target": target,
            "feature": tuple(feature_names),
            "importance": values,
        }
    )


def alphaearth_crop_feature_importance(
    models: AlphaEarthCropModelBundle,
) -> pd.DataFrame:
    """Return available importance diagnostics for flat or hierarchical models."""
    tables: list[pd.DataFrame] = []
    if models.cty_model is not None:
        tables.append(
            _alphaearth_estimator_feature_importance(
                models.cty_model,
                models.feature_names,
                target="CTY",
            )
        )
    if models.group_model is not None:
        tables.append(
            _alphaearth_estimator_feature_importance(
                models.group_model,
                models.feature_names,
                target="CTY_GROUP",
            )
        )
    for group_code, estimator in sorted(models.within_group_models.items()):
        tables.append(
            _alphaearth_estimator_feature_importance(
                estimator,
                models.feature_names,
                target=f"CTY_WITHIN_GROUP_{group_code}",
            )
        )
    nonempty = [table for table in tables if not table.empty]
    if not nonempty:
        return pd.DataFrame(columns=["target", "feature", "importance"])
    return (
        pd.concat(nonempty, ignore_index=True)
        .sort_values(["target", "importance"], ascending=[True, False])
        .reset_index(drop=True)
    )


def save_alphaearth_crop_models(
    models: AlphaEarthCropModelBundle,
    path: str | Path,
) -> Path:
    """Serialize a trained AlphaEarth crop-model bundle with joblib."""
    import joblib

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, output_path)
    return output_path


def load_alphaearth_crop_models(path: str | Path) -> AlphaEarthCropModelBundle:
    """Load and validate a serialized AlphaEarth crop-model bundle.

    Args:
        path: Joblib file created by :func:`save_alphaearth_crop_models`.

    Returns:
        Loaded AlphaEarth CTY model bundle.

    Raises:
        FileNotFoundError: If the model file does not exist.
        TypeError: If the serialized object is not an AlphaEarth model bundle.
    """
    import joblib

    model_path = Path(path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"AlphaEarth CTY model does not exist: {model_path}")
    models = joblib.load(model_path)
    if not isinstance(models, AlphaEarthCropModelBundle):
        raise TypeError(
            "Serialized AlphaEarth model must contain an "
            f"AlphaEarthCropModelBundle; found {type(models).__name__}."
        )
    return models


def hrl_tile_code_from_name(tile_name: str | Path) -> str:
    """Extract an HRL 100-km tile code such as ``E40N30`` from a filename."""
    stem = Path(tile_name).stem
    match = _HRL_TILE_NAME_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"Unrecognized HRL Croplands tile name: {tile_name}")
    return match.group("tile")


def build_hrl_prediction_tile_name(
    template_tile_name: str,
    *,
    product_code: str,
    prediction_year: int,
) -> str:
    """Build an HRL-compatible CTY or CTY-confidence prediction filename.

    Args:
        template_tile_name: Existing HRL CTY filename.
        product_code: ``"CTY"`` or ``"CTYCL"``.
        prediction_year: Target annual product year.

    Returns:
        HRL-compatible GeoTIFF filename including the ``.tif`` suffix.
    """
    if product_code not in {"CTY", "CTYCL"}:
        raise ValueError("product_code must be 'CTY' or 'CTYCL'.")
    stem = Path(template_tile_name).stem
    match = _HRL_TILE_NAME_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"Unrecognized HRL Croplands tile name: {template_tile_name}")
    return (
        f"CLMS_HRLVLCC_{product_code}_S{int(prediction_year)}_R10m_"
        f"{match.group('tile')}_03035_{match.group('version')}.tif"
    )


def find_hrl_tile_path(
    adapter_root: str | Path,
    *,
    year: int,
    tile_id: str,
) -> Path:
    """Locate an extracted HRL GeoTIFF below an adapter year folder."""
    root = Path(adapter_root) / str(int(year))
    stem = Path(tile_id).stem
    direct = root / f"{stem}.tif"
    if direct.exists():
        return direct
    matches = sorted(root.rglob(f"{stem}.tif"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find extracted HRL tile {stem}.tif below {root}."
        )
    raise RuntimeError(
        f"Found multiple extracted HRL tiles for {stem} below {root}: {matches}"
    )


def _open_alphaearth_vrts_for_template(
    selected_cogs: gpd.GeoDataFrame,
    template: rasterio.io.DatasetReader,
    stack: ExitStack,
) -> list[rasterio.vrt.WarpedVRT]:
    """Open downloaded AlphaEarth COGs as VRTs on an exact HRL tile grid."""
    if "local_path" not in selected_cogs.columns:
        raise ValueError("selected_cogs must contain a 'local_path' column.")

    vrts: list[rasterio.vrt.WarpedVRT] = []
    for local_path_value in selected_cogs["local_path"].drop_duplicates():
        local_path = Path(str(local_path_value))
        if not local_path.exists():
            raise FileNotFoundError(f"Missing downloaded AlphaEarth COG: {local_path}")
        source = stack.enter_context(rasterio.open(local_path))
        if source.count != len(ALPHAEARTH_EMBEDDING_BANDS):
            raise ValueError(
                f"Expected 64 AlphaEarth bands in {local_path}, found {source.count}."
            )
        vrts.append(
            stack.enter_context(
                rasterio.vrt.WarpedVRT(
                    source,
                    crs=template.crs,
                    transform=template.transform,
                    width=template.width,
                    height=template.height,
                    # Nearest-neighbour preserves the quantized embedding vector;
                    # bilinear interpolation of the nonlinearly quantized int8 values
                    # would not represent interpolation in dequantized embedding space.
                    resampling=rasterio.enums.Resampling.nearest,
                    src_nodata=ALPHAEARTH_NODATA_VALUE,
                    nodata=ALPHAEARTH_NODATA_VALUE,
                )
            )
        )
    if not vrts:
        raise ValueError("No AlphaEarth COGs were supplied for prediction.")
    return vrts


def _read_merged_alphaearth_window(
    vrts: list[rasterio.vrt.WarpedVRT],
    window: rasterio.windows.Window,
) -> np.ndarray:
    """Read and merge one 64-band window from multiple AlphaEarth VRTs."""
    height = int(window.height)
    width = int(window.width)
    merged = np.full(
        (len(ALPHAEARTH_EMBEDDING_BANDS), height, width),
        ALPHAEARTH_NODATA_VALUE,
        dtype=np.int8,
    )
    unresolved = np.ones((height, width), dtype=bool)
    for vrt in vrts:
        raw = vrt.read(
            indexes=list(range(1, len(ALPHAEARTH_EMBEDDING_BANDS) + 1)),
            window=window,
            out_dtype="int8",
            masked=False,
        )
        valid = unresolved & np.all(raw != ALPHAEARTH_NODATA_VALUE, axis=0)
        if valid.any():
            merged[:, valid] = raw[:, valid]
            unresolved[valid] = False
        if not unresolved.any():
            break
    return dequantize_alphaearth_embeddings(merged)


def _assert_matching_rasterio_grid(
    first: rasterio.io.DatasetReader,
    second: rasterio.io.DatasetReader,
) -> None:
    """Validate that two raster products share the exact grid."""
    if (
        first.crs != second.crs
        or first.transform != second.transform
        or first.width != second.width
        or first.height != second.height
    ):
        raise ValueError(
            "Raster products do not share the exact grid: "
            f"{first.name} versus {second.name}."
        )


def _copy_raster_metadata(
    source: rasterio.io.DatasetReader,
    destination: rasterio.io.DatasetWriter,
) -> None:
    """Copy non-grid GeoTIFF metadata from an HRL template."""
    global_tags = source.tags()
    if global_tags:
        destination.update_tags(**global_tags)
    band_tags = source.tags(1)
    if band_tags:
        destination.update_tags(1, **band_tags)
    if source.descriptions and source.descriptions[0]:
        destination.set_band_description(1, source.descriptions[0])
    if source.units and source.units[0]:
        destination.set_band_unit(1, source.units[0])
    try:
        color_map = source.colormap(1)
    except ValueError:
        color_map = None
    if color_map:
        destination.write_colormap(1, color_map)


def _predict_full_class_probabilities(
    model: Any,
    features: np.ndarray,
    class_codes: Sequence[int],
) -> np.ndarray:
    """Return classifier probabilities in one fixed HRL class order.

    Empty feature batches can occur when a prediction chunk initially contains
    valid AlphaEarth/cropland pixels but all of them are subsequently rejected
    because elevation or slope is nodata. Returning an empty probability matrix
    lets the caller write the chunk without invoking scikit-learn with zero rows.
    """
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError("Crop classifiers must provide predict_proba and classes_.")

    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(
            "Prediction features must be a two-dimensional matrix; "
            f"found shape {features.shape}."
        )
    if features.shape[0] == 0:
        return np.zeros((0, len(class_codes)), dtype=np.float32)

    raw_probabilities = np.asarray(model.predict_proba(features), dtype=np.float32)
    if raw_probabilities.ndim != 2:
        raise ValueError(
            "Classifier predict_proba must return a two-dimensional matrix; "
            f"found shape {raw_probabilities.shape}."
        )
    raw_probabilities = np.nan_to_num(
        raw_probabilities,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    model_classes = np.asarray(model.classes_, dtype=np.int32)
    requested_classes = np.asarray(class_codes, dtype=np.int32)
    probabilities = np.zeros(
        (features.shape[0], requested_classes.size),
        dtype=np.float32,
    )
    class_positions = {int(code): index for index, code in enumerate(requested_classes)}
    for model_index, class_code in enumerate(model_classes):
        output_index = class_positions.get(int(class_code))
        if output_index is not None:
            probabilities[:, output_index] = raw_probabilities[:, model_index]
    row_sums = probabilities.sum(axis=1)
    valid = row_sums > 0.0
    probabilities[valid] /= row_sums[valid, None]
    if (~valid).any():
        represented_positions = [
            class_positions[int(code)]
            for code in model_classes
            if int(code) in class_positions
        ]
        if represented_positions:
            probabilities[np.ix_(~valid, represented_positions)] = 1.0 / len(
                represented_positions
            )
    return probabilities


def _normalized_gaussian_filter_3x3(
    probabilities: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Apply a nodata-aware 3×3 Gaussian filter to per-class probabilities."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape (class, y, x).")
    if probabilities.shape[1:] != valid_mask.shape:
        raise ValueError("Probability and validity grids must align.")

    kernel = np.asarray(
        (
            (1.0, 2.0, 1.0),
            (2.0, 4.0, 2.0),
            (1.0, 2.0, 1.0),
        ),
        dtype=np.float32,
    )
    padded_probabilities = np.pad(
        probabilities,
        ((0, 0), (1, 1), (1, 1)),
        mode="constant",
        constant_values=0.0,
    )
    padded_valid = np.pad(
        valid_mask.astype(np.float32),
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=0.0,
    )
    weighted_sum = np.zeros_like(probabilities, dtype=np.float32)
    weight_sum = np.zeros(valid_mask.shape, dtype=np.float32)
    height, width = valid_mask.shape
    for row_offset in range(3):
        for col_offset in range(3):
            weight = kernel[row_offset, col_offset]
            neighbour_valid = padded_valid[
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
            weighted_sum += (
                padded_probabilities[
                    :,
                    row_offset : row_offset + height,
                    col_offset : col_offset + width,
                ]
                * neighbour_valid[None, :, :]
                * weight
            )
            weight_sum += neighbour_valid * weight

    smoothed = np.divide(
        weighted_sum,
        weight_sum[None, :, :],
        out=np.zeros_like(weighted_sum),
        where=weight_sum[None, :, :] > 0.0,
    )
    smoothed[:, ~valid_mask] = 0.0
    class_sums = smoothed.sum(axis=0)
    normalizable = valid_mask & (class_sums > 0.0)
    smoothed[:, normalizable] /= class_sums[normalizable][None, :]
    return smoothed


def _binary_dilate_square(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a Boolean mask using a square neighbourhood."""
    mask = np.asarray(mask, dtype=bool)
    if radius < 0:
        raise ValueError("Dilation radius cannot be negative.")
    if radius == 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for row_offset in range(2 * radius + 1):
        for col_offset in range(2 * radius + 1):
            result |= padded[
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
    return result


def _open_dataarray_vrt_for_template(
    data: xr.DataArray,
    template: rasterio.io.DatasetReader,
    stack: ExitStack,
) -> rasterio.vrt.WarpedVRT:
    """Open a model-grid DataArray as a bilinear VRT on an HRL tile grid."""
    if data.ndim != 2:
        raise ValueError("Auxiliary predictor DataArrays must be two-dimensional.")
    if data.rio.crs is None:
        raise ValueError("Auxiliary predictor DataArrays must have a CRS.")

    nodata = np.float32(-9999.0)
    values = np.asarray(data.values, dtype=np.float32)
    values = np.where(np.isfinite(values), values, nodata).astype(np.float32)
    memory_file = stack.enter_context(rasterio.io.MemoryFile())
    with memory_file.open(
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs=data.rio.crs,
        transform=data.rio.transform(recalc=True),
        nodata=float(nodata),
    ) as writer:
        writer.write(values, 1)

    # WarpedVRT requires a source opened in read-only mode. Reopen the completed
    # in-memory GeoTIFF after closing its writer instead of passing the writer
    # dataset directly to the VRT.
    source = stack.enter_context(memory_file.open())
    return stack.enter_context(
        rasterio.vrt.WarpedVRT(
            source,
            crs=template.crs,
            transform=template.transform,
            width=template.width,
            height=template.height,
            resampling=rasterio.enums.Resampling.bilinear,
            src_nodata=float(nodata),
            nodata=float(nodata),
            dtype="float32",
        )
    )


def _read_historical_cropland_mask(
    historical_sources: Sequence[rasterio.io.DatasetReader],
    window: rasterio.windows.Window,
    *,
    dilation_pixels: int = 0,
) -> np.ndarray:
    """Return the union of observed HRL cropland extents for one window."""
    if not historical_sources:
        return np.ones((int(window.height), int(window.width)), dtype=bool)
    cropland = np.zeros((int(window.height), int(window.width)), dtype=bool)
    for source in historical_sources:
        values = source.read(1, window=window, masked=False)
        cropland |= (values > 0) & (values != _HRL_OUTSIDE_AREA_CODE)
    if dilation_pixels:
        cropland = _binary_dilate_square(cropland, dilation_pixels)
    return cropland


def _copy_raster_to_temporary(
    source_path: Path,
    temporary_path: Path,
) -> None:
    """Create an atomic-edit working copy of one raster."""
    temporary_path.unlink(missing_ok=True)
    with rasterio.open(source_path) as source:
        profile = source.profile.copy()
        with rasterio.open(temporary_path, "w", **profile) as destination:
            for _, window in source.block_windows(1):
                destination.write(source.read(1, window=window), 1, window=window)
            _copy_raster_metadata(source, destination)


def predict_alphaearth_crop_tile_to_hrl_geotiffs(
    models: AlphaEarthCropModelBundle,
    cty_template_path: str | Path,
    current_alphaearth_cogs: gpd.GeoDataFrame,
    clip_geometry: BaseGeometry,
    cty_output_path: str | Path,
    *,
    geometry_crs: str = "EPSG:4326",
    chunk_size: int = 512,
    overwrite: bool = True,
    elevation: xr.DataArray | None = None,
    slope: xr.DataArray | None = None,
    historical_cty_paths: Sequence[str | Path] = (),
    apply_historical_cropland_mask: bool = True,
    historical_cropland_mask_dilation_pixels: int = 0,
    smooth_cty_probabilities: bool = True,
    unclassified_probability_threshold: float = 0.25,
    cty_confidence_output_path: str | Path | None = None,
) -> Path:
    """Predict and spatially post-process one annual CTY HRL tile.

    The prediction uses all 64 AlphaEarth dimensions and any optional topographic
    features stored in the model schema. A maximum historical HRL cropland extent
    can be applied as a BVL-like mask. Per-class probabilities are optionally
    smoothed with a nodata-aware 3×3 Gaussian kernel before reclassification.
    Low-confidence crop predictions are mapped to HRL unclassified annual or
    permanent crop classes.

    CTY minimum-mapping-unit filtering and interannual permanent-crop consistency
    are intentionally applied later, once all tiles and prediction years exist.

    Args:
        models: Final trained annual CTY model bundle.
        cty_template_path: Existing HRL CTY tile defining the exact output grid.
        current_alphaearth_cogs: Downloaded target-year AlphaEarth COG rows.
        clip_geometry: Active study-area geometry intersecting this HRL tile.
        cty_output_path: HRL-compatible CTY output path.
        geometry_crs: CRS of ``clip_geometry``.
        chunk_size: Core prediction-window width and height in pixels.
        overwrite: Replace existing outputs when True.
        elevation: Model-subgrid elevation used when required by the model.
        slope: Model-subgrid terrain gradient used when required by the model.
        historical_cty_paths: Observed HRL CTY tiles used to form maximum cropland
            extent.
        apply_historical_cropland_mask: Restrict crop predictions to the union of
            historical cropland pixels.
        historical_cropland_mask_dilation_pixels: Optional square dilation of the
            historical cropland union.
        smooth_cty_probabilities: Apply 3×3 Gaussian smoothing to CTY probabilities.
        unclassified_probability_threshold: Maximum-probability threshold below
            which annual/permanent crops become 3100/3200.
        cty_confidence_output_path: Optional uint8 CTY confidence-layer output.

    Returns:
        Path of the written CTY GeoTIFF.
    """
    if chunk_size < 16:
        raise ValueError("chunk_size must be at least 16 pixels.")
    if historical_cropland_mask_dilation_pixels < 0:
        raise ValueError("historical_cropland_mask_dilation_pixels cannot be negative.")
    if not 0.0 <= unclassified_probability_threshold <= 1.0:
        raise ValueError(
            "unclassified_probability_threshold must lie between zero and one."
        )

    coordinates_required = {
        "longitude",
        "latitude",
    }.issubset(models.feature_names)
    topography_required = {
        "elevation_m",
        "slope_gradient",
    }.issubset(models.feature_names)
    if topography_required and (elevation is None or slope is None):
        raise ValueError(
            "The trained feature schema requires elevation and slope predictors."
        )

    cty_output = Path(cty_output_path)
    confidence_output = (
        None if cty_confidence_output_path is None else Path(cty_confidence_output_path)
    )
    cty_output.parent.mkdir(parents=True, exist_ok=True)
    if confidence_output is not None:
        confidence_output.parent.mkdir(parents=True, exist_ok=True)

    existing_outputs = [cty_output]
    if confidence_output is not None:
        existing_outputs.append(confidence_output)
    if not overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError("Prediction output already exists and overwrite=False.")

    cty_temporary = cty_output.with_name(f".{cty_output.name}.part")
    confidence_temporary = (
        None
        if confidence_output is None
        else confidence_output.with_name(f".{confidence_output.name}.part")
    )
    for temporary in (cty_temporary, confidence_temporary):
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    smoothing_halo = 1 if smooth_cty_probabilities else 0
    processing_halo = max(
        smoothing_halo,
        historical_cropland_mask_dilation_pixels,
    )

    try:
        with ExitStack() as stack:
            cty_template = stack.enter_context(rasterio.open(cty_template_path))
            if cty_template.count != 1:
                raise ValueError("The HRL CTY template must contain one band.")

            geometry_in_template_crs = (
                gpd.GeoSeries([clip_geometry], crs=geometry_crs)
                .to_crs(cty_template.crs)
                .iloc[0]
            )
            if geometry_in_template_crs.is_empty:
                raise ValueError("Prediction geometry is empty after reprojection.")

            historical_sources: list[rasterio.io.DatasetReader] = []
            if apply_historical_cropland_mask:
                for historical_path_value in historical_cty_paths:
                    historical_source = stack.enter_context(
                        rasterio.open(historical_path_value)
                    )
                    _assert_matching_rasterio_grid(cty_template, historical_source)
                    historical_sources.append(historical_source)
                if not historical_sources:
                    raise ValueError(
                        "At least one historical CTY tile is required when "
                        "apply_historical_cropland_mask=True."
                    )

            cty_profile = cty_template.profile.copy()
            cty_profile.update(
                driver="GTiff",
                count=1,
                dtype="uint16",
                nodata=_HRL_OUTSIDE_AREA_CODE,
                width=cty_template.width,
                height=cty_template.height,
                crs=cty_template.crs,
                transform=cty_template.transform,
            )

            vrts = _open_alphaearth_vrts_for_template(
                current_alphaearth_cogs,
                cty_template,
                stack,
            )
            elevation_vrt = None
            slope_vrt = None
            if topography_required:
                assert elevation is not None and slope is not None
                elevation_vrt = _open_dataarray_vrt_for_template(
                    elevation,
                    cty_template,
                    stack,
                )
                slope_vrt = _open_dataarray_vrt_for_template(
                    slope,
                    cty_template,
                    stack,
                )

            cty_writer = stack.enter_context(
                rasterio.open(cty_temporary, "w", **cty_profile)
            )
            _copy_raster_metadata(cty_template, cty_writer)

            confidence_writer = None
            if confidence_temporary is not None:
                confidence_profile = cty_template.profile.copy()
                confidence_profile.update(
                    driver="GTiff",
                    count=1,
                    dtype="uint8",
                    nodata=_HRL_CTY_CONFIDENCE_OUTSIDE_AREA,
                    width=cty_template.width,
                    height=cty_template.height,
                    crs=cty_template.crs,
                    transform=cty_template.transform,
                )
                confidence_writer = stack.enter_context(
                    rasterio.open(
                        confidence_temporary,
                        "w",
                        **confidence_profile,
                    )
                )
                confidence_writer.update_tags(
                    PRODUCT="AlphaEarth-derived HRL-compatible CTY confidence",
                    VALUE_RANGE="0-100",
                    NO_CROPLAND=str(_HRL_CTY_CONFIDENCE_NO_CROPLAND),
                    OUTSIDE_AREA=str(_HRL_CTY_CONFIDENCE_OUTSIDE_AREA),
                )
                confidence_writer.set_band_description(
                    1,
                    "Probability of selected CTY class (%)",
                )

            coordinate_transformer = None
            if coordinates_required:
                coordinate_transformer = Transformer.from_crs(
                    cty_template.crs,
                    "EPSG:4326",
                    always_xy=True,
                )

            height = cty_template.height
            width = cty_template.width
            transform = cty_template.transform
            for row_start in range(0, height, chunk_size):
                core_height = min(chunk_size, height - row_start)
                read_row_start = max(0, row_start - processing_halo)
                read_row_stop = min(
                    height,
                    row_start + core_height + processing_halo,
                )
                for col_start in range(0, width, chunk_size):
                    core_width = min(chunk_size, width - col_start)
                    read_col_start = max(0, col_start - processing_halo)
                    read_col_stop = min(
                        width,
                        col_start + core_width + processing_halo,
                    )
                    read_window = rasterio.windows.Window(
                        read_col_start,
                        read_row_start,
                        read_col_stop - read_col_start,
                        read_row_stop - read_row_start,
                    )
                    read_height = int(read_window.height)
                    read_width = int(read_window.width)
                    read_affine = rasterio.windows.transform(
                        read_window,
                        transform,
                    )
                    inside_region = ~rasterio.features.geometry_mask(
                        [geometry_in_template_crs.__geo_interface__],
                        out_shape=(read_height, read_width),
                        transform=read_affine,
                        invert=False,
                        all_touched=False,
                    )

                    embeddings = _read_merged_alphaearth_window(vrts, read_window)
                    embeddings_flat = np.moveaxis(embeddings, 0, -1).reshape(-1, 64)
                    alphaearth_valid = (
                        np.isfinite(embeddings_flat)
                        .all(axis=1)
                        .reshape(
                            read_height,
                            read_width,
                        )
                    )
                    valid_data = inside_region & alphaearth_valid

                    if apply_historical_cropland_mask:
                        historical_cropland = _read_historical_cropland_mask(
                            historical_sources,
                            read_window,
                            dilation_pixels=historical_cropland_mask_dilation_pixels,
                        )
                    else:
                        historical_cropland = np.ones_like(valid_data, dtype=bool)
                    classify_mask = valid_data & historical_cropland

                    cty_probabilities = np.zeros(
                        (len(HRL_CTY_CLASS_CODES), read_height, read_width),
                        dtype=np.float32,
                    )

                    classify_flat = classify_mask.reshape(-1)
                    if classify_flat.any():
                        local_rows, local_cols = np.indices(
                            (read_height, read_width),
                            dtype=np.int64,
                        )
                        global_rows = (
                            local_rows.reshape(-1)[classify_flat] + read_row_start
                        )
                        global_cols = (
                            local_cols.reshape(-1)[classify_flat] + read_col_start
                        )
                        x_coordinates, y_coordinates = rasterio.transform.xy(
                            transform,
                            global_rows,
                            global_cols,
                            offset="center",
                        )

                        if coordinates_required:
                            assert coordinate_transformer is not None
                            longitude, latitude = coordinate_transformer.transform(
                                x_coordinates,
                                y_coordinates,
                            )
                            longitude = np.asarray(longitude, dtype=np.float64)
                            latitude = np.asarray(latitude, dtype=np.float64)
                        else:
                            n_valid = int(classify_flat.sum())
                            longitude = np.zeros(n_valid, dtype=np.float64)
                            latitude = np.zeros(n_valid, dtype=np.float64)

                        elevation_values = None
                        slope_values = None
                        if topography_required:
                            assert elevation_vrt is not None and slope_vrt is not None
                            elevation_window = elevation_vrt.read(
                                1,
                                window=read_window,
                                out_dtype="float32",
                                masked=False,
                            ).reshape(-1)[classify_flat]
                            slope_window = slope_vrt.read(
                                1,
                                window=read_window,
                                out_dtype="float32",
                                masked=False,
                            ).reshape(-1)[classify_flat]
                            elevation_values = np.where(
                                elevation_window == elevation_vrt.nodata,
                                np.nan,
                                elevation_window,
                            )
                            slope_values = np.where(
                                slope_window == slope_vrt.nodata,
                                np.nan,
                                slope_window,
                            )
                            topography_valid = np.isfinite(
                                elevation_values
                            ) & np.isfinite(slope_values)
                            if not topography_valid.all():
                                valid_indices = np.flatnonzero(classify_flat)
                                invalid_indices = valid_indices[~topography_valid]
                                classify_flat[invalid_indices] = False
                                classify_mask = classify_flat.reshape(
                                    read_height,
                                    read_width,
                                )

                                # Missing topography means the complete trained
                                # feature schema is unavailable. Mark these pixels
                                # as invalid data rather than silently converting
                                # them to the no-cropland class downstream.
                                valid_data_flat = valid_data.reshape(-1)
                                valid_data_flat[invalid_indices] = False
                                valid_data = valid_data_flat.reshape(
                                    read_height,
                                    read_width,
                                )

                                longitude = longitude[topography_valid]
                                latitude = latitude[topography_valid]
                                elevation_values = elevation_values[topography_valid]
                                slope_values = slope_values[topography_valid]

                        features, feature_names = build_alphaearth_crop_feature_matrix(
                            embeddings_flat[classify_flat],
                            longitude,
                            latitude,
                            include_coordinates=coordinates_required,
                            include_topography=topography_required,
                            normalize_embeddings=models.normalize_embeddings,
                            elevation_m=elevation_values,
                            slope_gradient=slope_values,
                        )
                        if feature_names != models.feature_names:
                            raise ValueError(
                                "Prediction feature schema differs from the trained model."
                            )

                        cty_prediction_probabilities = (
                            predict_alphaearth_crop_probabilities(
                                models,
                                features,
                            )
                        )
                        cty_probabilities.reshape(
                            len(HRL_CTY_CLASS_CODES),
                            -1,
                        )[:, classify_flat] = cty_prediction_probabilities.T

                    if smooth_cty_probabilities:
                        cty_probabilities = _normalized_gaussian_filter_3x3(
                            cty_probabilities,
                            classify_mask,
                        )

                    row_offset = row_start - read_row_start
                    col_offset = col_start - read_col_start
                    core_rows = slice(row_offset, row_offset + core_height)
                    core_cols = slice(col_offset, col_offset + core_width)
                    core_valid_data = valid_data[core_rows, core_cols]
                    core_classify = classify_mask[core_rows, core_cols]

                    cty_values = np.full(
                        (core_height, core_width),
                        _HRL_OUTSIDE_AREA_CODE,
                        dtype=np.uint16,
                    )
                    confidence_values = np.full(
                        (core_height, core_width),
                        _HRL_CTY_CONFIDENCE_OUTSIDE_AREA,
                        dtype=np.uint8,
                    )

                    historical_background = core_valid_data & ~core_classify
                    cty_values[historical_background] = _HRL_NO_CROPLAND_CODE
                    confidence_values[historical_background] = (
                        _HRL_CTY_CONFIDENCE_NO_CROPLAND
                    )

                    if core_classify.any():
                        core_cty_probabilities = cty_probabilities[
                            :,
                            core_rows,
                            core_cols,
                        ]
                        probability_rows = core_cty_probabilities[
                            :,
                            core_classify,
                        ].T
                        predicted_rows, confidence_rows = (
                            alphaearth_cty_from_probabilities(
                                models,
                                probability_rows,
                                unclassified_probability_threshold=(
                                    unclassified_probability_threshold
                                ),
                            )
                        )
                        predicted_cty = np.full(
                            (core_height, core_width),
                            _HRL_OUTSIDE_AREA_CODE,
                            dtype=np.int32,
                        )
                        best_cty_probabilities = np.zeros(
                            (core_height, core_width),
                            dtype=np.float32,
                        )
                        predicted_cty[core_classify] = predicted_rows
                        best_cty_probabilities[core_classify] = confidence_rows
                        cty_values[core_classify] = predicted_rows.astype(np.uint16)

                        confidence_percent = np.clip(
                            np.rint(best_cty_probabilities * 100.0),
                            0,
                            100,
                        ).astype(np.uint8)
                        confidence_values[core_classify] = confidence_percent[
                            core_classify
                        ]
                        predicted_background = core_classify & (
                            predicted_cty == _HRL_NO_CROPLAND_CODE
                        )
                        confidence_values[predicted_background] = (
                            _HRL_CTY_CONFIDENCE_NO_CROPLAND
                        )

                    core_window = rasterio.windows.Window(
                        col_start,
                        row_start,
                        core_width,
                        core_height,
                    )
                    cty_writer.write(cty_values, 1, window=core_window)
                    if confidence_writer is not None:
                        confidence_writer.write(
                            confidence_values,
                            1,
                            window=core_window,
                        )

        cty_temporary.replace(cty_output)
        if confidence_temporary is not None and confidence_output is not None:
            confidence_temporary.replace(confidence_output)
        return cty_output
    except Exception:
        for temporary in (cty_temporary, confidence_temporary):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        raise


def sample_alphaearth_crop_prediction_tiles(
    samples: pd.DataFrame,
    cty_paths_by_tile: dict[str, str | Path],
    *,
    sample_coordinates_crs: str | CRS | None = None,
    batch_size: int = 100_000,
) -> np.ndarray:
    """Sample final CTY tile products at evaluation locations.

    ``source_x``/``source_y`` are preferred because they preserve the exact HRL
    sample positions. When those columns are unavailable, longitude/latitude are
    accepted. Samples outside all supplied tiles retain the HRL outside-area code.

    Args:
        samples: Evaluation sample table.
        cty_paths_by_tile: CTY prediction paths keyed by HRL tile code.
        sample_coordinates_crs: CRS of ``source_x`` and ``source_y``.
        batch_size: Maximum coordinates sampled from one raster at once.

    Returns:
        CTY array aligned with ``samples``.
    """
    if samples.empty:
        raise ValueError("Cannot sample predictions for an empty sample table.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one.")
    if not cty_paths_by_tile:
        raise ValueError("At least one CTY prediction tile is required.")

    if {"source_x", "source_y"}.issubset(samples.columns):
        if sample_coordinates_crs is None:
            raise ValueError(
                "sample_coordinates_crs is required for source_x/source_y samples."
            )
        original_x = samples["source_x"].to_numpy(dtype=np.float64)
        original_y = samples["source_y"].to_numpy(dtype=np.float64)
        original_crs = CRS.from_user_input(sample_coordinates_crs)
    elif {"longitude", "latitude"}.issubset(samples.columns):
        original_x = samples["longitude"].to_numpy(dtype=np.float64)
        original_y = samples["latitude"].to_numpy(dtype=np.float64)
        original_crs = CRS.from_epsg(4326)
    else:
        raise ValueError(
            "Post-processed evaluation requires source_x/source_y or "
            "longitude/latitude sample coordinates."
        )

    predicted_cty = np.full(
        len(samples),
        _HRL_OUTSIDE_AREA_CODE,
        dtype=np.int32,
    )
    assigned = np.zeros(len(samples), dtype=bool)
    transformed_coordinates: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for tile_code in sorted(cty_paths_by_tile):
        cty_path = Path(cty_paths_by_tile[tile_code])
        with rasterio.open(cty_path) as cty_source:
            destination_crs = CRS.from_user_input(cty_source.crs)
            cache_key = destination_crs.to_string()
            if cache_key not in transformed_coordinates:
                if original_crs == destination_crs:
                    transformed_coordinates[cache_key] = (original_x, original_y)
                else:
                    transformer = Transformer.from_crs(
                        original_crs,
                        destination_crs,
                        always_xy=True,
                    )
                    transformed_x, transformed_y = transformer.transform(
                        original_x,
                        original_y,
                    )
                    transformed_coordinates[cache_key] = (
                        np.asarray(transformed_x, dtype=np.float64),
                        np.asarray(transformed_y, dtype=np.float64),
                    )
            x_values, y_values = transformed_coordinates[cache_key]
            bounds = cty_source.bounds
            inside = (
                ~assigned
                & np.isfinite(x_values)
                & np.isfinite(y_values)
                & (x_values >= bounds.left)
                & (x_values <= bounds.right)
                & (y_values >= bounds.bottom)
                & (y_values <= bounds.top)
            )
            indices = np.flatnonzero(inside)
            if indices.size == 0:
                continue

            for start in range(0, indices.size, batch_size):
                batch_indices = indices[start : start + batch_size]
                coordinates = list(
                    zip(
                        x_values[batch_indices],
                        y_values[batch_indices],
                        strict=True,
                    )
                )
                cty_values = np.fromiter(
                    (
                        int(value[0])
                        for value in cty_source.sample(
                            coordinates,
                            indexes=1,
                            masked=False,
                        )
                    ),
                    dtype=np.int32,
                    count=len(coordinates),
                )
                predicted_cty[batch_indices] = cty_values
                assigned[batch_indices] = True

    return predicted_cty


def apply_alphaearth_permanent_crop_temporal_consistency(
    historical_cty_paths: Sequence[str | Path],
    predicted_cty_paths: dict[int, str | Path],
    *,
    predicted_confidence_paths: dict[int, str | Path] | None = None,
    chunk_size: int = 1024,
) -> dict[str, int]:
    """Apply conservative HRL-style consistency rules to permanent crops.

    Historical HRL labels act as categorical evidence because historical
    per-class probability rasters are not available in this workflow. Only
    generated prediction years are modified.

    Implemented rules are:

    * a complete permanent-crop sequence is assigned its dominant exact permanent
      class;
    * one non-permanent interior prediction surrounded by an otherwise permanent
      sequence is filled with the dominant permanent class;
    * a permanent prediction between two annual-crop years is changed to
      no-cropland.

    Args:
        historical_cty_paths: Ordered observed HRL CTY tiles.
        predicted_cty_paths: Mapping from prediction year to generated CTY tile.
        predicted_confidence_paths: Optional matching CTY confidence tiles.
        chunk_size: Processing window size.

    Returns:
        Counts of changed pixels by temporal rule.
    """
    if chunk_size < 16:
        raise ValueError("chunk_size must be at least 16.")
    if not historical_cty_paths or not predicted_cty_paths:
        return {
            "all_permanent_harmonized": 0,
            "interior_permanent_gap_filled": 0,
            "isolated_permanent_removed": 0,
        }

    ordered_prediction_years = sorted(int(year) for year in predicted_cty_paths)
    cty_paths = {
        year: Path(predicted_cty_paths[year]) for year in ordered_prediction_years
    }
    confidence_paths = {
        int(year): Path(path)
        for year, path in (predicted_confidence_paths or {}).items()
    }

    temporary_cty = {
        year: path.with_name(f".{path.name}.temporal.part")
        for year, path in cty_paths.items()
    }
    temporary_confidence = {
        year: path.with_name(f".{path.name}.temporal.part")
        for year, path in confidence_paths.items()
        if year in cty_paths
    }
    for year, path in cty_paths.items():
        _copy_raster_to_temporary(path, temporary_cty[year])
    for year, path in confidence_paths.items():
        if year in temporary_confidence:
            _copy_raster_to_temporary(path, temporary_confidence[year])

    stats = {
        "all_permanent_harmonized": 0,
        "interior_permanent_gap_filled": 0,
        "isolated_permanent_removed": 0,
    }
    try:
        with ExitStack() as stack:
            historical_sources = [
                stack.enter_context(rasterio.open(path))
                for path in historical_cty_paths
            ]
            predicted_sources = {
                year: stack.enter_context(rasterio.open(temporary_cty[year], "r+"))
                for year in ordered_prediction_years
            }
            confidence_sources = {
                year: stack.enter_context(
                    rasterio.open(temporary_confidence[year], "r+")
                )
                for year in temporary_confidence
            }
            reference = historical_sources[0]
            for source in historical_sources[1:]:
                _assert_matching_rasterio_grid(reference, source)
            for source in predicted_sources.values():
                _assert_matching_rasterio_grid(reference, source)
            for source in confidence_sources.values():
                _assert_matching_rasterio_grid(reference, source)

            exact_permanent_codes = np.asarray(
                HRL_EXACT_PERMANENT_CTY_CLASS_CODES,
                dtype=np.uint16,
            )
            permanent_codes = np.asarray(
                HRL_PERMANENT_CTY_CLASS_CODES,
                dtype=np.uint16,
            )
            annual_codes = np.asarray(HRL_ANNUAL_CTY_CLASS_CODES, dtype=np.uint16)
            height = reference.height
            width = reference.width

            for row_start in range(0, height, chunk_size):
                window_height = min(chunk_size, height - row_start)
                for col_start in range(0, width, chunk_size):
                    window_width = min(chunk_size, width - col_start)
                    window = rasterio.windows.Window(
                        col_start,
                        row_start,
                        window_width,
                        window_height,
                    )
                    historical = np.stack(
                        [
                            source.read(1, window=window, masked=False)
                            for source in historical_sources
                        ],
                        axis=0,
                    ).astype(np.uint16, copy=False)
                    predicted = np.stack(
                        [
                            predicted_sources[year].read(
                                1,
                                window=window,
                                masked=False,
                            )
                            for year in ordered_prediction_years
                        ],
                        axis=0,
                    ).astype(np.uint16, copy=False)
                    complete = np.concatenate((historical, predicted), axis=0)

                    exact_counts = np.stack(
                        [
                            np.count_nonzero(complete == code, axis=0)
                            for code in exact_permanent_codes
                        ],
                        axis=0,
                    )
                    exact_total = exact_counts.sum(axis=0)
                    dominant_indices = np.argmax(exact_counts, axis=0)
                    dominant_permanent = exact_permanent_codes[dominant_indices]
                    dominant_count = np.take_along_axis(
                        exact_counts,
                        dominant_indices[None, :, :],
                        axis=0,
                    )[0]
                    temporal_confidence = np.clip(
                        np.rint(
                            np.divide(
                                dominant_count,
                                exact_total,
                                out=np.zeros_like(
                                    dominant_count,
                                    dtype=np.float64,
                                ),
                                where=exact_total > 0,
                            )
                            * 100.0
                        ),
                        0,
                        100,
                    ).astype(np.uint8)

                    permanent_sequence = np.isin(
                        complete,
                        permanent_codes,
                    )
                    changed_by_year: dict[int, np.ndarray] = {
                        year: np.zeros(
                            (window_height, window_width),
                            dtype=bool,
                        )
                        for year in ordered_prediction_years
                    }

                    # First fill one interior non-permanent gap in an otherwise
                    # permanent sequence. This can turn the complete sequence into
                    # a valid all-permanent sequence for the harmonisation below.
                    for prediction_position, year in enumerate(
                        ordered_prediction_years
                    ):
                        complete_position = (
                            len(historical_sources) + prediction_position
                        )
                        if not 0 < complete_position < complete.shape[0] - 1:
                            continue
                        current = predicted[prediction_position]
                        permanent_except_current = np.delete(
                            permanent_sequence,
                            complete_position,
                            axis=0,
                        ).all(axis=0)
                        fill_gap = (
                            permanent_except_current
                            & ~np.isin(current, permanent_codes)
                            & (exact_total > 0)
                        )
                        if fill_gap.any():
                            current[fill_gap] = dominant_permanent[fill_gap]
                            predicted[prediction_position] = current
                            complete[complete_position] = current
                            permanent_sequence[complete_position] = np.isin(
                                current,
                                permanent_codes,
                            )
                            changed_by_year[year] |= fill_gap
                            stats["interior_permanent_gap_filled"] += int(
                                fill_gap.sum()
                            )

                    all_permanent = permanent_sequence.all(axis=0) & (exact_total > 0)
                    for prediction_position, year in enumerate(
                        ordered_prediction_years
                    ):
                        current = predicted[prediction_position]
                        harmonize = all_permanent & (current != dominant_permanent)
                        if harmonize.any():
                            current[harmonize] = dominant_permanent[harmonize]
                            predicted[prediction_position] = current
                            complete[len(historical_sources) + prediction_position] = (
                                current
                            )
                            changed_by_year[year] |= harmonize
                            stats["all_permanent_harmonized"] += int(harmonize.sum())

                    # Finally remove a one-year permanent prediction occurring
                    # between two annual-crop years.
                    for prediction_position, year in enumerate(
                        ordered_prediction_years
                    ):
                        complete_position = (
                            len(historical_sources) + prediction_position
                        )
                        if not 0 < complete_position < complete.shape[0] - 1:
                            continue
                        current = predicted[prediction_position]
                        previous_values = complete[complete_position - 1]
                        next_values = complete[complete_position + 1]
                        remove_isolated = (
                            np.isin(current, permanent_codes)
                            & np.isin(previous_values, annual_codes)
                            & np.isin(next_values, annual_codes)
                        )
                        if remove_isolated.any():
                            current[remove_isolated] = _HRL_NO_CROPLAND_CODE
                            predicted[prediction_position] = current
                            complete[complete_position] = current
                            changed_by_year[year] |= remove_isolated
                            stats["isolated_permanent_removed"] += int(
                                remove_isolated.sum()
                            )

                    for prediction_position, year in enumerate(
                        ordered_prediction_years
                    ):
                        predicted_sources[year].write(
                            predicted[prediction_position],
                            1,
                            window=window,
                        )
                        confidence_source = confidence_sources.get(year)
                        if confidence_source is None:
                            continue
                        confidence = confidence_source.read(
                            1,
                            window=window,
                            masked=False,
                        ).astype(np.uint8, copy=False)
                        changed = changed_by_year[year]
                        changed_to_background = changed & (
                            predicted[prediction_position] == _HRL_NO_CROPLAND_CODE
                        )
                        changed_to_permanent = changed & np.isin(
                            predicted[prediction_position],
                            exact_permanent_codes,
                        )
                        confidence[changed_to_background] = (
                            _HRL_CTY_CONFIDENCE_NO_CROPLAND
                        )
                        confidence[changed_to_permanent] = np.maximum(
                            confidence[changed_to_permanent],
                            temporal_confidence[changed_to_permanent],
                        )
                        confidence_source.write(confidence, 1, window=window)

        for year, temporary in temporary_cty.items():
            temporary.replace(cty_paths[year])
        for year, temporary in temporary_confidence.items():
            temporary.replace(confidence_paths[year])
        return stats
    except Exception:
        for temporary in (*temporary_cty.values(), *temporary_confidence.values()):
            temporary.unlink(missing_ok=True)
        raise


def _matching_class_neighbour_confidence(
    labels: np.ndarray,
    confidence: np.ndarray,
    target_class: int,
) -> np.ndarray:
    """Return mean 3×3 confidence of neighbours carrying ``target_class``."""
    valid = (labels == target_class) & (confidence <= 100)
    padded_values = np.pad(
        np.where(valid, confidence, 0).astype(np.float32),
        1,
        mode="constant",
        constant_values=0.0,
    )
    padded_valid = np.pad(
        valid.astype(np.float32),
        1,
        mode="constant",
        constant_values=0.0,
    )
    height, width = labels.shape
    total = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for row_offset in range(3):
        for col_offset in range(3):
            total += padded_values[
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
            count += padded_valid[
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
    return np.divide(
        total,
        count,
        out=np.zeros_like(total),
        where=count > 0.0,
    )


def apply_alphaearth_cty_mmu_sieve(
    cty_paths: Sequence[str | Path],
    *,
    confidence_paths: Sequence[str | Path] | None = None,
    minimum_mapping_unit_pixels: int = 25,
    connectivity: int = 4,
    padding_pixels: int = 25,
    maximum_iterations: int = 3,
) -> pd.DataFrame:
    """Apply a padded, multi-pass HRL-style MMU sieve to CTY tiles.

    Neighbouring generated tiles are mosaicked into a padded processing window so
    connected patches crossing 100-km tile boundaries are treated consistently.

    Args:
        cty_paths: Generated CTY tiles for one prediction year.
        confidence_paths: Optional CTY confidence tiles in matching order.
        minimum_mapping_unit_pixels: Sieve threshold; 25 pixels equals 0.25 ha at
            10 m.
        connectivity: Four- or eight-neighbour connectivity.
        padding_pixels: Tile padding used before sieving.
        maximum_iterations: Maximum sieve passes; multiple passes better enforce
            MMU in multi-class rasters.

    Returns:
        Per-tile table with changed-pixel counts and sieve iterations.
    """
    if minimum_mapping_unit_pixels < 1:
        raise ValueError("minimum_mapping_unit_pixels must be at least one.")
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8.")
    if padding_pixels < 0:
        raise ValueError("padding_pixels cannot be negative.")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be at least one.")

    tile_paths = [Path(path) for path in cty_paths]
    if not tile_paths:
        return pd.DataFrame()
    confidence_by_stem = {
        Path(path).stem.replace("CTYCL", "CTY"): Path(path)
        for path in (confidence_paths or ())
    }
    results: list[dict[str, int | str]] = []

    with ExitStack() as source_stack:
        all_sources = [
            source_stack.enter_context(rasterio.open(path)) for path in tile_paths
        ]
        reference_crs = all_sources[0].crs
        for source in all_sources:
            if source.crs != reference_crs:
                raise ValueError("All CTY tiles must share one CRS for MMU sieving.")

        confidence_sources = []
        confidence_paths_available = list(confidence_by_stem.values())
        for path in confidence_paths_available:
            confidence_sources.append(source_stack.enter_context(rasterio.open(path)))

        for tile_path in tile_paths:
            with rasterio.open(tile_path) as template:
                x_resolution = abs(float(template.transform.a))
                y_resolution = abs(float(template.transform.e))
                expanded_bounds = (
                    template.bounds.left - padding_pixels * x_resolution,
                    template.bounds.bottom - padding_pixels * y_resolution,
                    template.bounds.right + padding_pixels * x_resolution,
                    template.bounds.top + padding_pixels * y_resolution,
                )
                merged, merged_transform = rasterio_merge(
                    all_sources,
                    bounds=expanded_bounds,
                    res=(x_resolution, y_resolution),
                    nodata=_HRL_OUTSIDE_AREA_CODE,
                    dtype="uint16",
                    method="first",
                    target_aligned_pixels=True,
                )
                padded_labels = merged[0].astype(np.int32, copy=False)
                valid_mask = padded_labels != _HRL_OUTSIDE_AREA_CODE
                sieved = padded_labels
                iterations_used = 0
                for iteration in range(maximum_iterations):
                    next_sieved = rasterio.features.sieve(
                        sieved,
                        size=minimum_mapping_unit_pixels,
                        mask=valid_mask,
                        connectivity=connectivity,
                    )
                    iterations_used = iteration + 1
                    if np.array_equal(next_sieved, sieved):
                        sieved = next_sieved
                        break
                    sieved = next_sieved

                core_window = (
                    rasterio.windows.from_bounds(
                        *template.bounds,
                        transform=merged_transform,
                    )
                    .round_offsets()
                    .round_lengths()
                )
                row_start = int(core_window.row_off)
                col_start = int(core_window.col_off)
                row_stop = row_start + template.height
                col_stop = col_start + template.width
                final_core = sieved[
                    row_start:row_stop,
                    col_start:col_stop,
                ].astype(np.uint16)
                original_core = template.read(1, masked=False)
                changed = final_core != original_core

                temporary = tile_path.with_name(f".{tile_path.name}.sieve.part")
                temporary.unlink(missing_ok=True)
                profile = template.profile.copy()
                with rasterio.open(temporary, "w", **profile) as destination:
                    destination.write(final_core, 1)
                    _copy_raster_metadata(template, destination)

                confidence_path = confidence_by_stem.get(tile_path.stem)
                confidence_changed = 0
                confidence_temporary = None
                if confidence_path is not None and confidence_sources:
                    merged_confidence, _ = rasterio_merge(
                        confidence_sources,
                        bounds=expanded_bounds,
                        res=(x_resolution, y_resolution),
                        nodata=_HRL_CTY_CONFIDENCE_OUTSIDE_AREA,
                        dtype="uint8",
                        method="first",
                        target_aligned_pixels=True,
                    )
                    padded_confidence = merged_confidence[0].astype(
                        np.uint8,
                        copy=False,
                    )
                    with rasterio.open(confidence_path) as confidence_template:
                        confidence_core = confidence_template.read(
                            1,
                            masked=False,
                        ).astype(np.uint8, copy=False)
                        updated_confidence = confidence_core.copy()
                        changed_to_background = changed & (
                            final_core == _HRL_NO_CROPLAND_CODE
                        )
                        updated_confidence[changed_to_background] = (
                            _HRL_CTY_CONFIDENCE_NO_CROPLAND
                        )
                        for class_code in np.unique(final_core[changed]):
                            class_code = int(class_code)
                            if class_code in {
                                _HRL_NO_CROPLAND_CODE,
                                _HRL_OUTSIDE_AREA_CODE,
                            }:
                                continue
                            neighbour_confidence = _matching_class_neighbour_confidence(
                                sieved,
                                padded_confidence,
                                class_code,
                            )
                            neighbour_core = neighbour_confidence[
                                row_start:row_stop,
                                col_start:col_stop,
                            ]
                            class_changed = changed & (final_core == class_code)
                            usable = class_changed & (neighbour_core > 0.0)
                            updated_confidence[usable] = np.clip(
                                np.rint(neighbour_core[usable]),
                                0,
                                100,
                            ).astype(np.uint8)
                        confidence_changed = int(
                            np.count_nonzero(updated_confidence != confidence_core)
                        )
                        confidence_temporary = confidence_path.with_name(
                            f".{confidence_path.name}.sieve.part"
                        )
                        confidence_temporary.unlink(missing_ok=True)
                        confidence_profile = confidence_template.profile.copy()
                        with rasterio.open(
                            confidence_temporary,
                            "w",
                            **confidence_profile,
                        ) as confidence_destination:
                            confidence_destination.write(updated_confidence, 1)
                            _copy_raster_metadata(
                                confidence_template,
                                confidence_destination,
                            )

                temporary.replace(tile_path)
                if confidence_temporary is not None:
                    confidence_temporary.replace(confidence_path)
                results.append(
                    {
                        "tile": tile_path.stem,
                        "changed_pixels": int(changed.sum()),
                        "confidence_changed_pixels": confidence_changed,
                        "iterations": iterations_used,
                    }
                )

    return pd.DataFrame(results)


def remove_alphaearth_downloads(
    selected_cogs: gpd.GeoDataFrame,
    *,
    logger: logging.Logger | None = None,
) -> int:
    """Delete downloaded AlphaEarth COGs while preserving the cached index.

    Args:
        selected_cogs: Adapter selection containing ``local_path`` values.
        logger: Optional logger used for deletion diagnostics.

    Returns:
        Number of local files removed.
    """
    if "local_path" not in selected_cogs.columns:
        return 0
    removed = 0
    for local_path_value in selected_cogs["local_path"].drop_duplicates():
        local_path = Path(str(local_path_value))
        if local_path.exists():
            local_path.unlink()
            removed += 1
            if logger is not None:
                logger.debug("Removed cached AlphaEarth COG %s.", local_path)
        parent = local_path.parent
        while parent.name and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return removed
