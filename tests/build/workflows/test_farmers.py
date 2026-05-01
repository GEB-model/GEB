"""Tests for farmer workflows."""

import logging
import os

import numpy as np
import pytest
import xarray as xr
from tqdm import tqdm

from geb.build.data_catalog import DataCatalog
from geb.build.workflows.farmers import (
    create_farm_distributions,
    create_farms_numba,
)
from geb.workflows.raster import (
    interpolate_na_2d,
    rasterize_like,
)

from ...testconfig import IN_GITHUB_ACTIONS

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

BOUNDS = (
    5.90,
    50.78,
    5.92,
    50.80,
)


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


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_fetch_field_boundaries() -> None:
    """Fetch field boundaries and rasterize."""
    logger = logging.getLogger("test_fetch_field_boundaries")

    field_boundaries = DataCatalog(logger=logger).fetch("field_boundaries").read(BOUNDS)

    assert not field_boundaries.empty
    assert {"id", "area", "geometry"}.issubset(field_boundaries.columns)
    assert field_boundaries.crs is not None
    assert field_boundaries.geometry.notna().all()
    assert (~field_boundaries.geometry.is_empty).all()

    field_boundaries["id"] = field_boundaries["id"].astype(np.int32)

    xmin, ymin, xmax, ymax = BOUNDS
    resolution = 1.5 / 3600  # 1.5 arcsec in degrees

    x = np.arange(xmin, xmax + resolution, resolution, dtype=np.float64)
    y = np.arange(ymax, ymin - resolution, -resolution, dtype=np.float64)

    test_raster = xr.DataArray(
        np.zeros((len(y), len(x)), dtype=np.float32),
        coords={"y": y, "x": x},
        dims=("y", "x"),
        name="test_raster",
        attrs={"_FillValue": np.nan},
    )

    test_raster = test_raster.rio.write_crs(field_boundaries.crs)

    assert test_raster.rio.crs == field_boundaries.crs
    assert test_raster.ndim == 2
    assert test_raster.dims == ("y", "x")

    field_boundaries_grid: xr.DataArray = rasterize_like(
        field_boundaries,
        column="id",
        raster=test_raster,
        dtype=np.int32,
        nodata=-1,
        all_touched=False,
    )

    assert isinstance(field_boundaries_grid, xr.DataArray)
    assert field_boundaries_grid.shape == test_raster.shape
    assert field_boundaries_grid.dims == test_raster.dims
    assert field_boundaries_grid.dtype == np.int32
    assert field_boundaries_grid.rio.crs == field_boundaries.crs
    assert field_boundaries_grid.attrs["_FillValue"] == -1

    unique_values_before = np.unique(field_boundaries_grid.values)
    assert np.any(unique_values_before != -1), "Rasterization produced only nodata."

    field_ids = set(field_boundaries["id"].unique().tolist())
    raster_ids_before = set(unique_values_before[unique_values_before != -1].tolist())
    assert raster_ids_before.issubset(field_ids)

    field_boundaries_grid = interpolate_na_2d(field_boundaries_grid)

    assert isinstance(field_boundaries_grid, xr.DataArray)
    assert field_boundaries_grid.shape == test_raster.shape
    assert field_boundaries_grid.dtype == np.int32

    unique_values_after = np.unique(field_boundaries_grid.values)
    assert -1 not in unique_values_after

    raster_ids_after = set(unique_values_after.tolist())
    assert raster_ids_after.issubset(field_ids)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_fetch_HRL_crop_types() -> None:
    """Fetch HRL crop types for different years."""
    logger = logging.getLogger("test_fetch_HRL_crop_types")

    years = [2017, 2023]
    crop_types_per_year: list[xr.DataArray] = []

    for year in years:
        crop_types = (
            DataCatalog(logger=logger)
            .fetch(
                f"hrl_crop_types_{year}",
                bounds=BOUNDS,
                year=year,
            )
            .read(
                bounds=BOUNDS,
                year=year,
            )
        )

        assert isinstance(crop_types, xr.DataArray)
        assert crop_types.ndim == 2
        assert crop_types.rio.crs is not None
        assert crop_types.shape[0] > 0
        assert crop_types.shape[1] > 0

        valid_values = crop_types.values
        valid_values = valid_values[~np.isnan(valid_values)]

        unique_values = np.unique(valid_values)

        assert unique_values.size > 1
        assert np.any(valid_values != 0)

        crop_types_per_year.append(crop_types.expand_dims(year=[year]))

    crop_types_over_time = xr.concat(
        crop_types_per_year,
        dim="year",
        join="exact",
    )

    assert isinstance(crop_types_over_time, xr.DataArray)
    assert crop_types_over_time.dims[0] == "year"
    assert list(crop_types_over_time["year"].values) == years
    assert crop_types_over_time.sizes["year"] == len(years)
    pass
