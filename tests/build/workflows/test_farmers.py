"""Tests for farmer workflows."""

import logging
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from tqdm import tqdm

from geb.build.data_catalog import DataCatalog
from geb.build.workflows.farmers import (
    assert_matching_raster_grid,
    create_farm_distributions,
    create_farms_numba,
    create_field_index_grid,
    create_lowder_target_farm_areas,
    dominant_crop_one_year_chunked,
    grow_farms_from_prepared_fields,
    prepare_projected_field_arrays,
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

    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
    crop_types_per_year: list[xr.DataArray] = []

    for year in years:
        adapter = DataCatalog(logger=logger).fetch(
            f"hrl_crop_types_{year}",
            bounds=BOUNDS,
            year=year,
        )

        crop_types = adapter.read(
            bounds=BOUNDS,
            year=year,
        )

        assert isinstance(crop_types, xr.DataArray)
        assert crop_types.ndim == 2
        assert crop_types.rio.crs is not None
        assert crop_types.shape[0] > 0
        assert crop_types.shape[1] > 0

        tile_ids = getattr(adapter, "tile_ids", None)
        assert tile_ids is not None
        assert len(tile_ids) > 0
        assert all("_CTY_" in tile_id for tile_id in tile_ids)

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


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_fetch_HRL_secondary_crop() -> None:
    """Fetch HRL secondary crop types for different years."""
    logger = logging.getLogger("test_fetch_HRL_secondary_crop")

    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
    secondary_crop_per_year: list[xr.DataArray] = []

    for year in years:
        adapter = DataCatalog(logger=logger).fetch(
            f"hrl_secondary_crop_{year}",
            bounds=BOUNDS,
            year=year,
        )

        secondary_crop = adapter.read(
            bounds=BOUNDS,
            year=year,
        )

        assert isinstance(secondary_crop, xr.DataArray)
        assert secondary_crop.ndim == 2
        assert secondary_crop.rio.crs is not None
        assert secondary_crop.shape[0] > 0
        assert secondary_crop.shape[1] > 0

        tile_ids = getattr(adapter, "tile_ids", None)
        assert tile_ids is not None
        assert len(tile_ids) > 0
        assert all("_CPSCT_" in tile_id for tile_id in tile_ids)

        valid_values = secondary_crop.values
        valid_values = valid_values[~np.isnan(valid_values)]

        unique_values = np.unique(valid_values)

        assert unique_values.size >= 1

        secondary_crop_per_year.append(secondary_crop.expand_dims(year=[year]))

    secondary_crop_over_time = xr.concat(
        secondary_crop_per_year,
        dim="year",
        join="exact",
    )

    assert isinstance(secondary_crop_over_time, xr.DataArray)
    assert secondary_crop_over_time.dims[0] == "year"
    assert list(secondary_crop_over_time["year"].values) == years
    assert secondary_crop_over_time.sizes["year"] == len(years)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_determine_farmers_from_boundaries_and_HRL_crops_optimized() -> None:
    """Use HRL crop data and field boundaries to create optimized synthetic farms."""
    logger = logging.getLogger(
        "test_determine_farmers_from_boundaries_and_HRL_crops_optimized"
    )

    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
    crop_columns = [f"crop_{year}" for year in years]

    field_boundaries = DataCatalog(logger=logger).fetch("field_boundaries").read(BOUNDS)
    field_boundaries["id"] = field_boundaries["id"].astype(np.int32)

    dominant_crop_per_year: list[np.ndarray] = []
    unique_field_ids: np.ndarray | None = None
    field_index_grid: np.ndarray | None = None

    for year_index, year in enumerate(years):
        crop_types_adapter = DataCatalog(logger=logger).fetch(
            f"hrl_crop_types_{year}",
            bounds=BOUNDS,
            year=year,
        )
        crop_types = crop_types_adapter.read(
            bounds=BOUNDS,
            year=year,
        )

        secondary_crop_adapter = DataCatalog(logger=logger).fetch(
            f"hrl_secondary_crop_{year}",
            bounds=BOUNDS,
            year=year,
        )
        secondary_crop = secondary_crop_adapter.read(
            bounds=BOUNDS,
            year=year,
        )

        assert_matching_raster_grid(crop_types, secondary_crop)

        crop_tile_ids = getattr(crop_types_adapter, "tile_ids", None)
        assert crop_tile_ids is not None
        assert len(crop_tile_ids) > 0
        assert all("_CTY_" in tile_id for tile_id in crop_tile_ids)

        secondary_tile_ids = getattr(secondary_crop_adapter, "tile_ids", None)
        assert secondary_tile_ids is not None
        assert len(secondary_tile_ids) > 0
        assert all("_CPSCT_" in tile_id for tile_id in secondary_tile_ids)

        if year_index == 0:
            # Rasterize fields only once because all HRL yearly rasters should
            # share the same grid.
            field_boundaries_grid: xr.DataArray = rasterize_like(
                field_boundaries,
                column="id",
                raster=crop_types,
                dtype=np.int32,
                nodata=-1,
                all_touched=False,
            )

            field_index_grid, unique_field_ids = create_field_index_grid(
                field_boundaries_grid,
                field_nodata=-1,
            )

        assert field_index_grid is not None
        assert unique_field_ids is not None

        dominant_crop_year = dominant_crop_one_year_chunked(
            crop_types=crop_types,
            secondary_crop=secondary_crop,
            field_index_grid=field_index_grid,
            n_fields=unique_field_ids.size,
            chunk_rows=2048,
            pair_base=65536,
            nodata=-1,
        )

        dominant_crop_per_year.append(dominant_crop_year)

    assert unique_field_ids is not None

    dominant_crop_table = pd.DataFrame(
        np.column_stack(dominant_crop_per_year).astype(np.int32),
        index=unique_field_ids,
        columns=crop_columns,
    )

    field_boundaries_with_crops = field_boundaries.merge(
        dominant_crop_table,
        left_on="id",
        right_index=True,
        how="left",
    )

    valid_crop_mask = (
        field_boundaries_with_crops[crop_columns].notna()
        & field_boundaries_with_crops[crop_columns].ne(-1)
    ).any(axis=1)

    field_boundaries_with_crops = field_boundaries_with_crops.loc[
        valid_crop_mask
    ].copy()

    assert not field_boundaries_with_crops.empty

    encoded_values = field_boundaries_with_crops[crop_columns].to_numpy(dtype=np.int32)
    valid_encoded_values = encoded_values[
        ~np.isin(encoded_values, np.array([-1, 0, 65535], dtype=np.int32))
    ]

    if valid_encoded_values.size > 0:
        assert set(np.unique(valid_encoded_values % 10)).issubset({0, 1, 2, 3, 4})

    iso3 = "NLD"

    farm_sizes_per_region = (
        DataCatalog(logger=logger).fetch("lowder_farm_size_distribution").read()
    )

    region_farm_sizes = farm_sizes_per_region.loc[
        farm_sizes_per_region["ISO3"] == iso3
    ].drop(["Country", "Census Year", "Total"], axis=1)

    assert len(region_farm_sizes) == 2, (
        f"Found {len(region_farm_sizes)} Lowder rows for {iso3}."
    )

    (
        projected_fields,
        original_crs,
        field_areas_m2,
        centroid_x,
        centroid_y,
        field_sequences,
    ) = prepare_projected_field_arrays(
        field_boundaries_with_crops,
        crop_columns,
    )

    target_farms = create_lowder_target_farm_areas(
        region_farm_sizes=region_farm_sizes,
        size_class_boundaries=LOWDER_SIZE_CLASS_BOUNDARIES_M2,
        cultivated_field_area_m2=float(field_areas_m2.sum()),
        iso3=iso3,
        logger=logger,
        random_seed=42,
        minimum_fields_per_farm=1.0,
        mean_field_area_m2=float(field_areas_m2.mean()),
    )

    fields_with_farms, farmers = grow_farms_from_prepared_fields(
        projected_fields=projected_fields,
        original_crs=original_crs,
        field_areas_m2=field_areas_m2,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        field_sequences=field_sequences,
        target_farms=target_farms,
        max_distance_m=500.0,
        max_neighbors=32,
        distance_weight=0.45,
        crop_sequence_weight=0.35,
        switch_timing_weight=0.20,
        target_overshoot_tolerance=1.25,
    )

    assert not fields_with_farms.empty
    assert not farmers.empty

    assert "farmer_id" in fields_with_farms.columns
    assert "farmer_id" in farmers.columns

    assert fields_with_farms["farmer_id"].notna().all()
    assert fields_with_farms["farmer_id"].min() == 0
    assert fields_with_farms["farmer_id"].max() == len(farmers) - 1

    assert farmers["farmer_id"].to_numpy().tolist() == list(range(len(farmers)))
    assert farmers["n_fields"].sum() == len(fields_with_farms)
    assert farmers["area_ha"].gt(0).all()

    assert np.isclose(
        farmers["area_m2"].sum(),
        projected_fields["field_area_m2"].sum(),
        rtol=0.01,
    )
