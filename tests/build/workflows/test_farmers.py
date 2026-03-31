"""Tests for farmer workflows."""

import logging
import os

import numpy as np
import pytest
from tqdm import tqdm

from geb.build.data_catalog import DataCatalog
from geb.build.workflows.farmers import (
    create_farm_distributions,
    create_farms_numba,
)

LOWDER_SIZE_CLASS_BOUNDARIES_M2: dict[str, tuple[float, float]] = {
    "< 1 Ha": (0.0, 10_000.0),
    "1 - 2 Ha": (10_000.0, 20_000.0),
    "2 - 5 Ha": (20_000.0, 50_000.0),
    "5 - 10 Ha": (50_000.0, 100_000.0),
    "10 - 20 Ha": (100_000.0, 200_000.0),
    "20 - 50 Ha": (200_000.0, 500_000.0),
    "50 - 100 Ha": (500_000.0, 1_000_000.0),
    "100 - 200 Ha": (1_000_000.0, 2_000_000.0),
    "200 - 500 Ha": (2_000_000.0, 5_000_000.0),
    "500 - 1000 Ha": (5_000_000.0, 10_000_000.0),
    "> 1000 Ha": (10_000_000.0, np.inf),
}


def test_create_farms_numba_no_farms() -> None:
    """When there are no farms and no cultivated land, the result is all -1."""
    # No cultivated land (all zeros), and no farmers
    cultivated_land = np.zeros((3, 3), dtype=np.int32)
    ids = np.array([], dtype=np.int32)
    farm_sizes = np.array([], dtype=np.int32)

    np.random.seed(0)
    farms = create_farms_numba(cultivated_land, ids, farm_sizes)

    assert farms.shape == cultivated_land.shape
    # All cells should be -1 (non-farm land)
    assert np.all(farms == -1)


def test_create_farms_numba_some_farmers() -> None:
    """Allocate a small set of farms across a contiguous cultivated block."""
    # 4x5 grid with a 2x3 cultivated block (6 cells)
    cultivated_land = np.zeros((4, 5), dtype=np.int32)
    cultivated_land[0:2, 0:3] = 1

    # Two farmers with sizes summing to cultivated cells
    ids = np.array([1, 2], dtype=np.int32)
    farm_sizes = np.array([2, 4], dtype=np.int32)

    np.random.seed(42)
    farms = create_farms_numba(cultivated_land, ids, farm_sizes)

    assert farms.shape == cultivated_land.shape

    # Non-cultivated cells should be -1
    assert np.all(farms[cultivated_land == 0] == -1)

    # Cultivated cells must be assigned to one of the provided IDs
    assigned_ids = np.unique(farms[cultivated_land == 1])
    assigned_ids = assigned_ids[assigned_ids != -1]
    assert set(assigned_ids.tolist()) == {1, 2}

    # Check each farmer received exactly their target number of cells
    assert int(np.count_nonzero(farms == 1)) == 2
    assert int(np.count_nonzero(farms == 2)) == 4


def test_create_farms_numba_single_farmer_single_cell() -> None:
    """Single farmer owning exactly one cultivated cell."""
    cultivated_land = np.zeros((2, 2), dtype=np.int32)
    cultivated_land[1, 1] = 1

    ids = np.array([99], dtype=np.int32)
    farm_sizes = np.array([1], dtype=np.int32)

    np.random.seed(123)
    farms = create_farms_numba(cultivated_land, ids, farm_sizes)

    # Only the cultivated cell should be assigned to the farmer ID
    assert farms[1, 1] == 99
    # All other cells remain non-farm land (-1)
    mask_other = np.ones_like(cultivated_land, dtype=bool)
    mask_other[1, 1] = False
    assert np.all(farms[mask_other] == -1)


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Downloads the Lowder workbook and iterates over all ISO3 groups.",
)
def test_create_farm_distributions_for_all_lowder_regions() -> None:
    """Ensure each Lowder ISO3 group can produce a farm-size distribution."""
    logger = logging.getLogger("test_create_farm_distributions_for_all_lowder_regions")
    lowder_farm_sizes = (
        DataCatalog(logger=logger).fetch("lowder_farm_size_distribution").read()
    )
    cultivated_land_area_region_m2 = 10_000_000.0
    average_subgrid_area_region = 1.0
    cultivated_land_region_total_cells = int(
        cultivated_land_area_region_m2 / average_subgrid_area_region
    )

    processed_iso3: list[str] = []

    for iso3, region_farm_sizes in tqdm(
        lowder_farm_sizes.groupby("ISO3"), desc="Processing ISO3 groups"
    ):
        region_farm_sizes = region_farm_sizes.drop(
            ["Country", "Census Year", "Total"], axis=1
        )
        assert len(region_farm_sizes) == 2, (
            f"Expected two Lowder rows for {iso3}, found {len(region_farm_sizes)}."
        )

        region_agents = create_farm_distributions(
            region_farm_sizes=region_farm_sizes,
            size_class_boundaries=LOWDER_SIZE_CLASS_BOUNDARIES_M2,
            cultivated_land_area_region_m2=cultivated_land_area_region_m2,
            average_subgrid_area_region=average_subgrid_area_region,
            cultivated_land_region_total_cells=cultivated_land_region_total_cells,
            UID=0,
            ISO3=iso3,
            logger=logger,
        )

        assert not region_agents.empty, f"Expected at least one farm for {iso3}."
        assert region_agents["region_id"].eq(0).all()
        assert region_agents["farm_size_cells"].gt(0).all()

        processed_iso3.append(iso3)

    assert processed_iso3
