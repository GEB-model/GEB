"""Class to set GEB up for Europe."""

import calendar
import gc
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
import rasterio
import xarray as xr
from rasterio import features
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from geb.agents.crop_farmers import (
    FIELD_EXPANSION_ADAPTATION,
    INDEX_INSURANCE_ADAPTATION,
    IRRIGATION_EFFICIENCY_ADAPTATION_DRIP,
    IRRIGATION_EFFICIENCY_ADAPTATION_SPRINKLER,
    PR_INSURANCE_ADAPTATION,
    SURFACE_IRRIGATION_EQUIPMENT,
    TRADITIONAL_INSURANCE_ADAPTATION,
    WELL_ADAPTATION,
)
from geb.build.data_catalog.wekeo_copernicus import WEkEONoCoverageError
from geb.build.methods import build_method
from geb.build.workflows.crop_calendars import MIRCA_OS_CROP_CLASS_MAP
from geb.build.workflows.farmers import (
    HRL_CPSCT_CLASS_CODES,
    HRL_CTY_CLASS_CODES,
    alphaearth_crop_feature_importance,
    apply_alphaearth_cty_mmu_sieve,
    apply_alphaearth_permanent_crop_temporal_consistency,
    assert_matching_raster_grid,
    assign_farmer_sequences_to_area_targets,
    combine_crop_and_secondary_values,
    create_alphaearth_crop_training_samples,
    create_lowder_target_farm_areas,
    enforce_cpsct_annual_cropland_mask,
    evaluate_alphaearth_crop_models,
    evaluate_alphaearth_crop_predictions,
    format_alphaearth_accuracy_report,
    farm_size_distribution_fit_by_size_class,
    fit_alphaearth_crop_models,
    get_farm_locations,
    grow_farms_from_exact_crop_sequences,
    grow_farms_from_raster_cells,
    build_hrl_prediction_tile_name,
    find_hrl_tile_path,
    hrl_tile_code_from_name,
    load_alphaearth_crop_training_samples,
    predict_alphaearth_crop_tile_to_hrl_geotiffs,
    raster_cell_area_m2,
    relax_lowder_targets_for_sequence_fit,
    remove_alphaearth_downloads,
    round_crop_states_to_area_targets,
    save_alphaearth_crop_models,
    sample_alphaearth_crop_prediction_tiles,
    select_alphaearth_cogs_for_geometry,
    select_cultivated_cells_by_area,
)
from geb.geb_types import TwoDArrayInt32
from geb.workflows.io import get_window
from geb.workflows.raster import (
    fillna_2d,
    get_linear_indices,
    interpolate_na_2d,
    interpolate_na_along_dim,
    rasterize_like,
    sample_from_map,
)

from .. import GEBModel
from ..data_catalog import DataCatalog
from ..workflows.conversions import setup_donor_countries

_FARMERS_WITH_CROPS_TABLE = "agents/farmers/farmers_with_crops"
_DEFAULT_HRL_RASTER_CHUNKS = {"x": 4096, "y": 4096}
_HRL_FALLOW_CROP_CODE = -1
_HRL_MISSING_CROP_CODE = -2
_HRL_NO_CROPLAND_CODE = 0
_HRL_OUTSIDE_AREA_CODE = 65535


@dataclass(frozen=True, slots=True)
class _LowderSequenceSettings:
    """Settings for the current Lowder sequence-balanced workflow.

    Attributes:
        distance_weight: Farm-growth weight for spatial compactness.
        crop_sequence_weight: Farm-growth weight for complete-sequence similarity.
        switch_timing_weight: Farm-growth weight for crop-switch timing.
        min_valid_crop_sequence_overlap: Minimum comparable years for sequence
            similarity.
        jump_candidate_sample: Candidate cells sampled for a disconnected parcel.
        max_jump_distance_m: Preferred maximum parcel-jump distance.
        crop_area_alignment_weight: Weight assigned to regional crop-area fit.
        max_local_sequences: Maximum local sequence candidates per farmer.
        max_regional_sequences: Maximum regional fallback candidates per farmer.
        regional_sequence_pool_size: Common regional sequences considered for
            fallback construction.
        local_search_passes: Reassignment passes using local sequences only.
        regional_search_passes: Reassignment passes with regional fallbacks.
        local_fit_threshold_pct: Fit threshold for skipping regional fallbacks.
        fallow_penalty: Preference penalty for fallow-heavy sequences.
        extra_farm_fraction: Maximum Lowder target-count relaxation.
    """

    distance_weight: float
    crop_sequence_weight: float
    switch_timing_weight: float
    min_valid_crop_sequence_overlap: int
    jump_candidate_sample: int
    max_jump_distance_m: float
    crop_area_alignment_weight: float
    max_local_sequences: int
    max_regional_sequences: int
    regional_sequence_pool_size: int
    local_search_passes: int
    regional_search_passes: int
    local_fit_threshold_pct: float
    fallow_penalty: float
    extra_farm_fraction: float


@dataclass(frozen=True, slots=True)
class _ExactSequenceSettings:
    """Settings for hard exact-sequence grouping.

    Attributes:
        jump_candidate_sample: Same-sequence cells sampled for a new parcel.
        distance_scale_m: Distance scale used to prefer nearby parcels.
        temporal_persistence_weight: Persistence preference used during annual
            crop-area rounding.
    """

    jump_candidate_sample: int
    distance_scale_m: float
    temporal_persistence_weight: float


HRL_TO_MIRCA_OS_CROP_CLASS_MAP: dict[int, int | None] = {
    1110: 0,  # Wheat
    1120: 3,  # Barley
    1130: 1,  # Maize
    1140: 2,  # Rice
    1150: 25,  # Other cereals -> Others annual
    1210: 25,  # Fresh vegetables -> Others annual
    1220: 16,  # Dry pulses -> Pulses
    1310: 9,  # Potatoes
    1320: 12,  # Sugar beet
    1410: 8,  # Sunflower
    1420: 7,  # Soybeans
    1430: 14,  # Rapeseed
    1440: 25,  # Flax, cotton and hemp -> Others annual
    2100: 23,  # Grapes -> Others perennial
    2200: 23,  # Olives -> Others perennial
    2310: 23,  # Fruits -> Others perennial
    2320: 23,  # Nuts -> Others perennial
    3100: 25,  # Unclassified annual crop
    3200: 23,  # Unclassified permanent crop
    0: None,  # No cropland
    65535: None,  # Outside area
}

HRL_SECONDARY_CROP_NONE = 0
HRL_SECONDARY_CROP_SHORT_SUMMER = 1
HRL_SECONDARY_CROP_LONG_SUMMER = 2
HRL_SECONDARY_CROP_SHORT_WINTER = 3
HRL_SECONDARY_CROP_LONG_WINTER = 4

MIRCA2000_UNIT_GRID = "mirca2000_unit_grid"

_HRL_CROPLANDS_EEA38_ISO3 = frozenset(
    {
        # EU27
        "AUT",
        "BEL",
        "BGR",
        "HRV",
        "CYP",
        "CZE",
        "DNK",
        "EST",
        "FIN",
        "FRA",
        "DEU",
        "GRC",
        "HUN",
        "IRL",
        "ITA",
        "LVA",
        "LTU",
        "LUX",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "ESP",
        "SWE",
        # non-EU EEA/Eionet member countries
        "ISL",
        "LIE",
        "NOR",
        "CHE",
        "TUR",
        # cooperating Western Balkan countries
        "ALB",
        "BIH",
        "MNE",
        "MKD",
        "SRB",
        # Kosovo code
        "XKX",
    }
)


def _default_size_class_boundaries() -> dict[str, tuple[int | float, int | float]]:
    """Return default Lowder farm-size class boundaries.

    The boundaries are expressed in square metres and follow the Lowder-style
    farm-size classes used to sample synthetic target farm areas. The final class
    is open-ended and uses ``np.inf`` as the upper boundary.

    Returns:
        Mapping from farm-size class label to lower and upper area boundaries in
        square metres.
    """
    return {
        "< 1 Ha": (0, 10_000),
        "1 - 2 Ha": (10_000, 20_000),
        "2 - 5 Ha": (20_000, 50_000),
        "5 - 10 Ha": (50_000, 100_000),
        "10 - 20 Ha": (100_000, 200_000),
        "20 - 50 Ha": (200_000, 500_000),
        "50 - 100 Ha": (500_000, 1_000_000),
        "100 - 200 Ha": (1_000_000, 2_000_000),
        "200 - 500 Ha": (2_000_000, 5_000_000),
        "500 - 1000 Ha": (5_000_000, 10_000_000),
        "> 1000 Ha": (10_000_000, np.inf),
    }


def _active_subgrid_mask_geometry_for_hrl(
    template: xr.DataArray,
    active_mask: np.ndarray,
) -> BaseGeometry:
    """Convert the active model subgrid mask to a geometry for HRL clipping.

    The returned geometry is the exact active-mask geometry in EPSG:4326. Its
    bounds are used only as the WEkEO candidate-tile search envelope. The
    geometry itself is passed to the WEkEO adapter so tiles outside the active
    domain can be skipped before merging and intersecting tiles can be clipped
    before merging.

    Args:
        template: Subgrid template defining transform, shape, and CRS.
        active_mask: Boolean array where True indicates active model cells.

    Returns:
        Active-domain geometry in EPSG:4326.

    Raises:
        ValueError: If the active mask and template shapes differ, if the
            template CRS is missing, or if the active mask contains no active
            cells.
    """
    if active_mask.shape != template.shape:
        raise ValueError(
            "active_mask must have the same shape as template. "
            f"Got {active_mask.shape} and {template.shape}."
        )

    if template.rio.crs is None:
        raise ValueError(
            "Cannot derive an active-subgrid clip geometry because the template "
            "has no CRS."
        )

    mask_values = active_mask.astype(np.uint8)
    geometries = [
        shapely_shape(geometry)
        for geometry, value in features.shapes(
            mask_values,
            mask=active_mask,
            transform=template.rio.transform(),
        )
        if int(value) == 1
    ]

    if not geometries:
        raise ValueError("Cannot derive HRL clip geometry from an empty active mask.")

    active_geometry = unary_union(geometries)
    active_geometry = (
        gpd.GeoSeries([active_geometry], crs=template.rio.crs)
        .to_crs("EPSG:4326")
        .iloc[0]
    )

    if active_geometry.is_empty:
        raise ValueError("Derived HRL clip geometry is empty.")

    return active_geometry


def _align_hrl_rasters_to_common_grid(
    crop_types: xr.DataArray,
    secondary_crop: xr.DataArray,
    *,
    region_id: int,
    year: int,
    logger: Any,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Align HRL crop and secondary-crop rasters to their common grid.

    Crop-type and secondary-crop products can occasionally return slightly
    different clipped extents for the same request bounds. The downstream crop
    counting requires identical grids, so this function first accepts already
    matching rasters and otherwise trims both rasters to their shared x/y
    coordinates.

    Args:
        crop_types: HRL crop-type raster.
        secondary_crop: HRL secondary-crop raster.
        region_id: Region ID used for diagnostics.
        year: HRL crop year used for diagnostics.
        logger: Logger used for debug diagnostics.

    Returns:
        Crop-type and secondary-crop rasters on the same grid.

    Raises:
        ValueError: If the rasters have no common x/y overlap or cannot be aligned
            to a matching grid.
    """
    try:
        assert_matching_raster_grid(crop_types, secondary_crop)
        return crop_types, secondary_crop
    except AssertionError, ValueError:
        logger.debug(
            "Aligning HRL crop and secondary-crop rasters to common grid for "
            "region %s, year %s. Crop bounds=%s; secondary bounds=%s.",
            region_id,
            year,
            crop_types.rio.bounds(),
            secondary_crop.rio.bounds(),
        )

    crop_aligned, secondary_aligned = xr.align(
        crop_types,
        secondary_crop,
        join="inner",
    )

    if crop_aligned.sizes.get("x", 0) == 0 or crop_aligned.sizes.get("y", 0) == 0:
        raise ValueError(
            "HRL crop and secondary-crop rasters have no common overlap after "
            f"alignment for region {region_id}, year {year}. "
            f"Crop bounds={crop_types.rio.bounds()}; "
            f"secondary bounds={secondary_crop.rio.bounds()}."
        )

    if crop_types.rio.crs is not None:
        crop_aligned = crop_aligned.rio.write_crs(crop_types.rio.crs)
    if secondary_crop.rio.crs is not None:
        secondary_aligned = secondary_aligned.rio.write_crs(secondary_crop.rio.crs)

    crop_nodata = crop_types.rio.nodata
    secondary_nodata = secondary_crop.rio.nodata
    if crop_nodata is not None:
        crop_aligned = crop_aligned.rio.write_nodata(crop_nodata)
    if secondary_nodata is not None:
        secondary_aligned = secondary_aligned.rio.write_nodata(secondary_nodata)

    assert_matching_raster_grid(crop_aligned, secondary_aligned)
    return crop_aligned, secondary_aligned


def _assert_compact_farm_ids(
    farms: xr.DataArray,
    farmers: pd.DataFrame,
    *,
    farmer_id_column: str = "farmer_id",
    nodata: int = -1,
) -> None:
    """Validate that farm raster IDs and the farmer table are compact and aligned.

    Args:
        farms: Farm raster where non-farm cells are ``nodata``.
        farmers: Farmer table expected to align with the farm raster.
        farmer_id_column: Name of the farmer-ID column in ``farmers``.
        nodata: Nodata value in the farm raster.

    Raises:
        ValueError: If the farm raster has no represented farmers.
        ValueError: If raster IDs are not exactly ``0..len(farmers)-1``.
        ValueError: If the farmer table IDs are not exactly
            ``0..len(farmers)-1``.
    """
    farm_values = farms.values
    present_ids = np.unique(farm_values[farm_values != nodata]).astype(np.int32)

    if present_ids.size == 0:
        raise ValueError("Farm raster contains no represented farmers.")

    expected_ids = np.arange(len(farmers), dtype=np.int32)

    if not np.array_equal(present_ids, expected_ids):
        missing_ids = np.setdiff1d(expected_ids, present_ids)
        extra_ids = np.setdiff1d(present_ids, expected_ids)
        raise ValueError(
            "Farm raster IDs are not compact or not aligned with the farmer table. "
            f"Expected IDs 0..{len(farmers) - 1}. "
            f"Missing examples: {missing_ids[:10].tolist()}; "
            f"extra examples: {extra_ids[:10].tolist()}."
        )

    if farmer_id_column in farmers.columns:
        farmer_ids = farmers[farmer_id_column].to_numpy(dtype=np.int32)
        if not np.array_equal(farmer_ids, expected_ids):
            raise ValueError(
                f"Farmer table column {farmer_id_column!r} is not compact and "
                "aligned with row order."
            )


def decode_crop_type_with_secondary_crop(
    combined_crop_type: np.ndarray,
    *,
    invalid_crop_values: tuple[int, ...] = (-2, -1, 0, 65535),
) -> tuple[np.ndarray, np.ndarray]:
    """Decode combined HRL crop-secondary codes.

    Args:
        combined_crop_type: HRL crop code with optional secondary-crop suffix in
            the final digit.
        invalid_crop_values: Crop values treated as missing or outside the valid
            crop domain.

    Returns:
        Tuple with main HRL crop codes and secondary-crop timing codes.
    """
    combined_crop_type = np.asarray(combined_crop_type, dtype=np.int32)
    invalid_crop = np.isin(combined_crop_type, invalid_crop_values)

    main_crop = (combined_crop_type // 10) * 10
    secondary_crop = combined_crop_type % 10

    main_crop = np.where(invalid_crop, -1, main_crop).astype(np.int32)
    secondary_crop = np.where(invalid_crop, 0, secondary_crop).astype(np.int32)

    return main_crop, secondary_crop


def map_hrl_crop_to_mirca_crop(
    hrl_crop: np.ndarray,
    *,
    missing_value: int = -1,
) -> np.ndarray:
    """Map HRL crop classes to MIRCA crop classes.

    Args:
        hrl_crop: Array with HRL main crop codes.
        missing_value: Value assigned where the HRL crop cannot be mapped.

    Returns:
        Array with MIRCA crop-class IDs and ``missing_value`` for unmapped crops.
    """
    mapped = np.full(hrl_crop.shape, missing_value, dtype=np.int32)

    for hrl_code, mirca_code in HRL_TO_MIRCA_OS_CROP_CLASS_MAP.items():
        if mirca_code is None:
            continue
        mapped[hrl_crop == hrl_code] = int(mirca_code)

    return mapped


def _decode_hrl_crop_combinations_from_farmer_table(
    farmers_with_crops: pd.DataFrame,
    *,
    crop_column: str,
    n_farmers: int,
    farmer_region_ids: np.ndarray,
    logger: Any,
) -> pd.DataFrame:
    """Decode final farmer-level HRL crop codes to MIRCA crop combinations.

    The input table is the compact farmer table written by one of the
    HRL raster farm-construction workflows. Its ``farmer_id`` values
    already correspond to the final compact ``agents/farmers/farms`` raster and
    all other farmer arrays. Therefore, no field geometry or farm-raster sampling
    is needed here.

    The selected HRL crop column is decoded into a main HRL crop and secondary
    crop timing code. The main HRL crop is then mapped to the corresponding MIRCA
    crop class used by the crop-growth parameterization.

    Args:
        farmers_with_crops: Final compact farmer table with ``farmer_id``,
            ``area_m2``, and HRL crop-sequence columns.
        crop_column: HRL crop column to use, for example ``"crop_2023"``.
        n_farmers: Number of final compact farmers.
        farmer_region_ids: Region ID per final compact farmer.
        logger: Logger used for warnings.

    Returns:
        DataFrame with one row per final compact farmer and columns
        ``farmer_id``, ``mirca_crop``, ``secondary_crop_type``, and
        ``assigned_crop_area_m2``.

    Raises:
        ValueError: If required columns are missing.
        ValueError: If the farmer table contains duplicate farmer IDs.
        ValueError: If no farmer-level HRL crops can be mapped to MIRCA crops.
    """
    required_columns = {"farmer_id", "area_m2", crop_column}
    missing_columns = required_columns - set(farmers_with_crops.columns)
    if missing_columns:
        raise ValueError(
            "Final farmer crop table is missing required column(s): "
            f"{sorted(missing_columns)}"
        )

    farmers = farmers_with_crops[["farmer_id", "area_m2", crop_column]].copy()
    farmers["farmer_id"] = farmers["farmer_id"].astype(np.int32)

    if farmers["farmer_id"].duplicated().any():
        duplicated_ids = farmers.loc[
            farmers["farmer_id"].duplicated(), "farmer_id"
        ].tolist()
        raise ValueError(
            "Final farmer crop table contains duplicate farmer_id values. "
            f"Examples: {duplicated_ids[:10]}"
        )

    valid_id_mask = (farmers["farmer_id"] >= 0) & (farmers["farmer_id"] < n_farmers)
    if not valid_id_mask.all():
        invalid_ids = farmers.loc[~valid_id_mask, "farmer_id"].tolist()
        raise ValueError(
            "Final farmer crop table contains farmer_id values outside the final "
            f"compact range [0, {n_farmers - 1}]. Examples: {invalid_ids[:10]}"
        )

    main_hrl_crop, secondary_crop_type = decode_crop_type_with_secondary_crop(
        farmers[crop_column].fillna(-1).to_numpy(dtype=np.int32)
    )
    mirca_crop = map_hrl_crop_to_mirca_crop(main_hrl_crop)

    farmer_crops = pd.DataFrame(
        {
            "farmer_id": farmers["farmer_id"].to_numpy(dtype=np.int32),
            "mirca_crop": mirca_crop.astype(np.int32),
            "secondary_crop_type": secondary_crop_type.astype(np.int32),
            "assigned_crop_area_m2": farmers["area_m2"].to_numpy(dtype=np.float64),
        }
    )

    if farmer_crops.empty:
        raise ValueError(
            f"No farmer-level HRL crops in {crop_column!r} could be decoded."
        )

    missing_farmers = np.setdiff1d(
        np.arange(n_farmers, dtype=np.int32),
        farmer_crops["farmer_id"].to_numpy(dtype=np.int32),
    )
    if missing_farmers.size:
        logger.warning(
            "No valid HRL-to-MIRCA crop could be assigned to %s farmer(s); "
            "filling with regional modal crop combinations.",
            missing_farmers.size,
        )

        farmer_crops = _fill_missing_farmer_crops_with_region_mode(
            farmer_crops,
            missing_farmers=missing_farmers,
            farmer_region_ids=farmer_region_ids,
        )

    return farmer_crops.sort_values("farmer_id").reset_index(drop=True)


def _farmer_area_array_from_farmer_table(
    farmers_with_crops: pd.DataFrame,
    *,
    n_farmers: int,
) -> np.ndarray:
    """Create an area array aligned with final compact farmer IDs.

    Args:
        farmers_with_crops: Final compact farmer table containing ``farmer_id``
            and ``area_m2``.
        n_farmers: Number of final compact farmers.

    Returns:
        One-dimensional area array where index equals final compact farmer ID.

    Raises:
        ValueError: If required columns are missing.
        ValueError: If any final compact farmer is missing an area value.
    """
    required_columns = {"farmer_id", "area_m2"}
    missing_columns = required_columns - set(farmers_with_crops.columns)
    if missing_columns:
        raise ValueError(
            "Final farmer crop table is missing required column(s): "
            f"{sorted(missing_columns)}"
        )

    farmer_areas_m2 = np.full(n_farmers, np.nan, dtype=np.float64)

    farmer_ids = farmers_with_crops["farmer_id"].to_numpy(dtype=np.int32)
    valid_mask = (farmer_ids >= 0) & (farmer_ids < n_farmers)

    farmer_areas_m2[farmer_ids[valid_mask]] = farmers_with_crops.loc[
        valid_mask, "area_m2"
    ].to_numpy(dtype=np.float64)

    if np.isnan(farmer_areas_m2).any():
        missing_farmer_ids = np.flatnonzero(np.isnan(farmer_areas_m2))
        raise ValueError(
            "Final farmer crop table does not contain area_m2 for all compact "
            f"farmers. Missing examples: {missing_farmer_ids[:10].tolist()}"
        )

    return farmer_areas_m2


def _fill_missing_farmer_crops_with_region_mode(
    farmer_crops: pd.DataFrame,
    *,
    missing_farmers: np.ndarray,
    farmer_region_ids: np.ndarray,
) -> pd.DataFrame:
    """Fill missing farmer crop combinations with regional modal combinations.

    Args:
        farmer_crops: DataFrame with existing farmer crop assignments.
        missing_farmers: Farmer IDs without an HRL crop assignment.
        farmer_region_ids: Region ID per farmer.

    Returns:
        DataFrame with missing farmers appended.
    """
    crop_lookup = farmer_crops.copy()
    crop_lookup["region_id"] = farmer_region_ids[
        crop_lookup["farmer_id"].to_numpy(dtype=np.int32)
    ]

    fallback_rows: list[dict[str, float | int]] = []

    global_mode = (
        crop_lookup.groupby(["mirca_crop", "secondary_crop_type"], sort=False)
        .size()
        .idxmax()
    )

    for farmer_id in missing_farmers:
        region_id = int(farmer_region_ids[farmer_id])
        region_rows = crop_lookup.loc[crop_lookup["region_id"] == region_id]

        if region_rows.empty:
            mirca_crop, secondary_crop_type = global_mode
        else:
            mirca_crop, secondary_crop_type = (
                region_rows.groupby(["mirca_crop", "secondary_crop_type"], sort=False)
                .size()
                .idxmax()
            )

        fallback_rows.append(
            {
                "farmer_id": int(farmer_id),
                "mirca_crop": int(mirca_crop),
                "secondary_crop_type": int(secondary_crop_type),
                "assigned_crop_area_m2": 0.0,
            }
        )

    return pd.concat(
        [farmer_crops, pd.DataFrame(fallback_rows)],
        ignore_index=True,
    )


def _fix_365_in_crop_calendar(
    crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
) -> dict[int, list[tuple[float, TwoDArrayInt32]]]:
    """Replace growth lengths of 365 days with 364.

    Args:
        crop_calendar: Parsed MIRCA crop calendar dictionary.

    Returns:
        Crop calendar dictionary with 365-day growth lengths clamped to 364.

    Raises:
        ValueError: If a value of 365 is found outside the growth-length column.
    """
    crop_calendar_adjusted = crop_calendar.copy()

    for key, entries in crop_calendar_adjusted.items():
        for index, (area, arr) in enumerate(entries):
            rows, cols = np.where(arr == 365)

            if rows.size == 0:
                continue

            if not np.all(cols == 3):
                raise ValueError(
                    f"Found 365 outside column 3 for key={key}, index={index}: "
                    f"indices={list(zip(rows, cols))}"
                )

            arr[rows, 3] = 364
            entries[index] = (area, arr)

    return crop_calendar_adjusted


def check_crop_calendar(crop_calendar_per_farmer: np.ndarray) -> None:
    """Validate that active crop-calendar rows have unique timing combinations.

    The crop-calendar array is validated without iterating over farmers in Python.
    Two active rows for the same farmer are invalid when both their planting-day
    and growth-length values are equal.

    Args:
        crop_calendar_per_farmer: Crop-calendar array with shape
            ``(n_farmers, n_calendar_rows, n_variables)``. The active-row marker
            is expected in the final column, while planting day and growth length
            are expected in columns 1 and 3.

    Raises:
        ValueError: If the input does not have the expected three-dimensional
            structure or contains fewer than four calendar variables.
        AssertionError: If duplicate active timing combinations are found for one
            or more farmers.
    """
    crop_calendar_per_farmer = np.asarray(crop_calendar_per_farmer)

    if crop_calendar_per_farmer.ndim != 3:
        raise ValueError(
            "crop_calendar_per_farmer must be a three-dimensional array. "
            f"Got shape {crop_calendar_per_farmer.shape}."
        )

    if crop_calendar_per_farmer.shape[2] < 4:
        raise ValueError(
            "crop_calendar_per_farmer must contain at least four variables per "
            f"calendar row. Got shape {crop_calendar_per_farmer.shape}."
        )

    active = crop_calendar_per_farmer[:, :, -1] != -1
    planting_day = crop_calendar_per_farmer[:, :, 1]
    growth_length = crop_calendar_per_farmer[:, :, 3]

    invalid = np.zeros(crop_calendar_per_farmer.shape[0], dtype=bool)
    n_calendar_rows = crop_calendar_per_farmer.shape[1]

    # The number of calendar rows is very small (currently three), so looping over
    # row pairs is cheap while all farmer comparisons remain vectorized.
    for first_row in range(n_calendar_rows - 1):
        for second_row in range(first_row + 1, n_calendar_rows):
            invalid |= (
                active[:, first_row]
                & active[:, second_row]
                & (planting_day[:, first_row] == planting_day[:, second_row])
                & (growth_length[:, first_row] == growth_length[:, second_row])
            )

    if invalid.any():
        invalid_farmer_ids = np.flatnonzero(invalid)
        raise AssertionError(
            "Duplicate active crop-calendar timing combinations found for "
            f"{invalid_farmer_ids.size} farmer(s). Examples: "
            f"{invalid_farmer_ids[:10].tolist()}."
        )


# Manual replacement of certain crops
def replace_crop(
    crop_calendar_per_farmer: np.ndarray,
    crop_values: np.ndarray | list[int],
    replaced_crop_values: np.ndarray | list[int],
) -> np.ndarray:
    """Replace selected crops with the most common calendar of candidate crops.

    The function first determines which crop value from `crop_values` occurs most
    often in the first crop-calendar column. It then finds the most common full
    crop-calendar pattern among farmers growing that crop. Finally, every farmer
    growing one of the `replaced_crop_values` is assigned that replacement
    calendar.

    This is useful for removing unsupported or unwanted crop classes while
    preserving a realistic cropping calendar from crops that are actually present
    in the model domain.

    Args:
        crop_calendar_per_farmer: Crop-calendar array with shape
            ``(n_farmers, n_rotations_or_seasons, n_variables)``. The crop class
            is expected in ``crop_calendar_per_farmer[:, :, 0]``. Missing crop
            entries are expected to use ``-1``.
        crop_values: Candidate crop class values from which the replacement crop
            calendar may be selected.
        replaced_crop_values: Crop class values that should be replaced.

    Returns:
        The updated crop-calendar array. If none of the candidate crops are
        present, the input array is returned unchanged.
    """
    # Find the most common crop value among the given crop_values
    crop_instances = crop_calendar_per_farmer[:, :, 0][
        np.isin(crop_calendar_per_farmer[:, :, 0], crop_values)
    ]

    # if none of the crops are present, no need to replace anything
    if crop_instances.size == 0:
        return crop_calendar_per_farmer

    crops, crop_counts = np.unique(crop_instances, return_counts=True)
    most_common_crop = crops[np.argmax(crop_counts)]

    # If multiple calendars represent this crop, retain the most common one.
    new_crop_types = crop_calendar_per_farmer[
        (crop_calendar_per_farmer[:, :, 0] == most_common_crop).any(axis=1),
        :,
        :,
    ]
    unique_rows, counts = np.unique(new_crop_types, axis=0, return_counts=True)
    max_index = np.argmax(counts)
    crop_replacement = unique_rows[max_index]

    crop_replacement_only_crops = crop_replacement[crop_replacement[:, -1] != -1]
    if crop_replacement_only_crops.shape[0] > 1:
        assert (
            np.unique(crop_replacement_only_crops[:, [1, 3]], axis=0).shape[0]
            == crop_replacement_only_crops.shape[0]
        )

    for replaced_crop in replaced_crop_values:
        # Check where to be replaced crop is
        crop_mask = (crop_calendar_per_farmer[:, :, 0] == replaced_crop).any(axis=1)
        # Replace the crop
        crop_calendar_per_farmer[crop_mask] = crop_replacement

    return crop_calendar_per_farmer


def _fill_missing_mirca2000_crop_calendars(
    crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
    *,
    logger: Any,
) -> dict[int, list[tuple[float, TwoDArrayInt32]]]:
    """Fill empty MIRCA2000 unit calendars from the numerically closest unit.

    Args:
        crop_calendar: Parsed MIRCA2000 crop calendar dictionary.
        logger: Logger used for warnings and information messages.

    Returns:
        Crop calendar dictionary with empty unit entries filled where possible.

    Raises:
        ValueError: If no valid MIRCA unit calendar exists.
    """
    missing_mirca_units = [
        unit for unit, calendars in crop_calendar.items() if not calendars
    ]

    if not missing_mirca_units:
        logger.debug("All MIRCA2000 units have valid crop calendars.")
        return crop_calendar

    logger.warning(
        "Missing crop calendar for MIRCA2000 unit(s): %s.",
        missing_mirca_units,
    )

    valid_units = [unit for unit, calendars in crop_calendar.items() if calendars]
    if not valid_units:
        raise ValueError("No valid MIRCA2000 units found in crop calendar data.")

    for mirca_unit in missing_mirca_units:
        closest_mirca_unit = min(valid_units, key=lambda unit: abs(unit - mirca_unit))
        crop_calendar[mirca_unit] = crop_calendar[closest_mirca_unit]
        logger.info(
            "Filling missing crop calendar for MIRCA2000 unit %s with data from %s.",
            mirca_unit,
            closest_mirca_unit,
        )

    return crop_calendar


def _calendar_active_rows(calendar: np.ndarray) -> np.ndarray:
    """Return active crop rows from a crop calendar matrix.

    Args:
        calendar: Crop calendar matrix.

    Returns:
        Rows where the crop ID is not ``-1``.
    """
    return calendar[calendar[:, 0] != -1]


def _classify_season_from_start_and_length(
    start_day: int,
    growth_length: int,
    *,
    short_length_threshold_days: int = 150,
) -> int:
    """Classify a MIRCA season into an HRL secondary-crop timing class.

    Args:
        start_day: Zero-based planting day index.
        growth_length: Growing-season length in days.
        short_length_threshold_days: Maximum length treated as a short season.

    Returns:
        HRL-style secondary-crop timing class.
    """
    start_day = int(start_day)
    growth_length = int(growth_length)

    is_summer = 90 <= start_day < 273
    is_short = growth_length <= short_length_threshold_days

    if is_summer and is_short:
        return HRL_SECONDARY_CROP_SHORT_SUMMER
    if is_summer and not is_short:
        return HRL_SECONDARY_CROP_LONG_SUMMER
    if not is_summer and is_short:
        return HRL_SECONDARY_CROP_SHORT_WINTER
    return HRL_SECONDARY_CROP_LONG_WINTER


def _calendar_matches_secondary_type(
    calendar: np.ndarray,
    *,
    main_crop: int,
    secondary_crop_type: int,
) -> bool:
    """Check whether a MIRCA calendar matches an HRL secondary-crop type.

    Args:
        calendar: MIRCA crop calendar matrix.
        main_crop: HRL-derived MIRCA main crop class. A value of ``-1``
            indicates that no valid crop is assigned.
        secondary_crop_type: HRL secondary-crop timing class.

    Returns:
        True if the calendar is compatible with the HRL secondary-crop type.
        For ``main_crop == -1``, only an empty all-``-1`` calendar is considered
        compatible.
    """
    active_rows = _calendar_active_rows(calendar)

    # Missing HRL crop means the farmer should receive an empty calendar, not a
    # MIRCA fallback crop. This branch is mainly defensive because the caller
    # already handles main_crop == -1 before searching candidates.
    if main_crop == -1:
        return active_rows.shape[0] == 0

    if active_rows.shape[0] == 0:
        return False

    # No HRL secondary crop: prefer a single-crop MIRCA calendar for the same
    # main crop.
    if secondary_crop_type == HRL_SECONDARY_CROP_NONE:
        return active_rows.shape[0] == 1 and int(active_rows[0, 0]) == main_crop

    # HRL indicates a secondary crop. MIRCA2000 must therefore provide a
    # multi-crop calendar containing the HRL-derived main crop.
    if active_rows.shape[0] < 2:
        return False

    if main_crop not in active_rows[:, 0]:
        return False

    # HRL only gives the timing class of the secondary crop. The secondary crop
    # identity itself is taken from the matching MIRCA2000 calendar.
    for row in active_rows:
        season_type = _classify_season_from_start_and_length(
            int(row[2]),
            int(row[3]),
        )
        if season_type == secondary_crop_type:
            return True

    return False


def _select_mirca2000_calendar_for_farmer(
    crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
    *,
    mirca_unit: int,
    main_crop: int,
    secondary_crop_type: int,
    is_irrigated: bool,
    replace_crop_calendar_unit_code: dict[int, int],
) -> np.ndarray:
    """Select the most common MIRCA2000 calendar for a farmer crop combination.

    Args:
        crop_calendar: Parsed MIRCA2000 crop calendar dictionary.
        mirca_unit: MIRCA2000 unit assigned to the farmer.
        main_crop: HRL-derived MIRCA crop class. A value of ``-1`` indicates
            that no valid crop is assigned.
        secondary_crop_type: HRL secondary-crop timing class.
        is_irrigated: Whether the farmer has irrigation access.
        replace_crop_calendar_unit_code: Optional MIRCA unit replacement mapping.

    Returns:
        MIRCA2000 crop calendar matrix with shape ``(3, 5)``. If ``main_crop`` is
        ``-1``, returns an empty crop calendar where all values are ``-1``.

    Raises:
        ValueError: If no calendar exists for the MIRCA unit.
        ValueError: If no calendar can be found for the crop combination in any
            MIRCA2000 unit.
    """
    if main_crop == -1:
        return np.full((3, 5), -1, dtype=np.int32)

    lookup_unit = replace_crop_calendar_unit_code.get(mirca_unit, mirca_unit)

    if lookup_unit not in crop_calendar or not crop_calendar[lookup_unit]:
        raise ValueError(f"No crop calendar found for MIRCA2000 unit {lookup_unit}.")

    def _contains_main_crop(entry: tuple[float, np.ndarray]) -> bool:
        """Check whether a calendar entry contains the requested main crop.

        Returns:
            ``True`` when an active calendar row contains the requested crop.
        """
        _, calendar = entry
        active_rows = _calendar_active_rows(calendar)

        if active_rows.shape[0] == 0:
            return False

        return main_crop in active_rows[:, 0]

    def _has_irrigation_status(entry: tuple[float, np.ndarray]) -> bool:
        """Check whether a calendar entry has the requested irrigation status.

        Returns:
            ``True`` when the first active crop row matches ``is_irrigated``.
        """
        _, calendar = entry
        active_rows = _calendar_active_rows(calendar)

        if active_rows.shape[0] == 0:
            return False

        return bool(active_rows[0, 1]) == is_irrigated

    def _select_best_candidate(
        candidates: list[tuple[float, np.ndarray]],
    ) -> np.ndarray | None:
        """Select the largest-area calendar from a candidate list.

        Args:
            candidates: Calendar candidates paired with their represented areas.

        Returns:
            The selected calendar as an integer array, or ``None`` when no candidates
            are available.
        """
        if not candidates:
            return None

        return max(candidates, key=lambda entry: entry[0])[1].astype(np.int32)

    def _select_from_candidates(
        candidates: list[tuple[float, np.ndarray]],
    ) -> np.ndarray | None:
        """Select the best calendar while respecting secondary-crop timing.

        Args:
            candidates: Calendar candidates paired with their represented areas.

        Returns:
            The best compatible calendar, or ``None`` when no candidate can be used.
        """
        exact_candidates = [
            entry
            for entry in candidates
            if _calendar_matches_secondary_type(
                entry[1],
                main_crop=main_crop,
                secondary_crop_type=secondary_crop_type,
            )
        ]

        selected_calendar = _select_best_candidate(exact_candidates)
        if selected_calendar is not None:
            return selected_calendar

        if secondary_crop_type != HRL_SECONDARY_CROP_NONE:
            second_crop_candidates = [
                entry
                for entry in candidates
                if _calendar_active_rows(entry[1]).shape[0] >= 2
            ]

            selected_calendar = _select_best_candidate(second_crop_candidates)
            if selected_calendar is not None:
                return selected_calendar

        return _select_best_candidate(candidates)

    local_entries = crop_calendar[lookup_unit]

    # Preferred local search: same MIRCA unit, same crop, and matching
    # rainfed/irrigated calendar class.
    local_matching_irrigation_candidates = [
        entry
        for entry in local_entries
        if _contains_main_crop(entry) and _has_irrigation_status(entry)
    ]

    selected_calendar = _select_from_candidates(local_matching_irrigation_candidates)
    if selected_calendar is not None:
        return selected_calendar

    # Local fallback: keep the same MIRCA unit and crop, but ignore whether the
    # available MIRCA2000 calendar is rainfed or irrigated.
    local_any_irrigation_candidates = [
        entry for entry in local_entries if _contains_main_crop(entry)
    ]

    selected_calendar = _select_from_candidates(local_any_irrigation_candidates)
    if selected_calendar is not None:
        return selected_calendar

    # Cross-unit fallback 1: if this MIRCA unit does not contain this crop at all,
    # search other MIRCA units for the same crop and same rainfed/irrigated class.
    other_unit_matching_irrigation_candidates: list[tuple[float, np.ndarray]] = []
    for unit_code, entries in crop_calendar.items():
        if unit_code == lookup_unit:
            continue

        other_unit_matching_irrigation_candidates.extend(
            entry
            for entry in entries
            if _contains_main_crop(entry) and _has_irrigation_status(entry)
        )

    selected_calendar = _select_from_candidates(
        other_unit_matching_irrigation_candidates
    )
    if selected_calendar is not None:
        return selected_calendar

    # Cross-unit fallback 2: final fallback for this crop. Search all other units
    # for the same crop, ignoring the rainfed/irrigated calendar class.
    other_unit_any_irrigation_candidates: list[tuple[float, np.ndarray]] = []
    for unit_code, entries in crop_calendar.items():
        if unit_code == lookup_unit:
            continue

        other_unit_any_irrigation_candidates.extend(
            entry for entry in entries if _contains_main_crop(entry)
        )

    selected_calendar = _select_from_candidates(other_unit_any_irrigation_candidates)
    if selected_calendar is not None:
        return selected_calendar

    raise ValueError(
        f"No MIRCA2000 calendar found for unit={lookup_unit}, crop={main_crop}, "
        f"secondary_type={secondary_crop_type}, is_irrigated={is_irrigated}, "
        "including cross-unit fallbacks."
    )


def _select_mirca2000_calendars_for_farmers(
    crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
    *,
    farmer_mirca_units: np.ndarray,
    farmer_main_crops: np.ndarray,
    farmer_secondary_crop_types: np.ndarray,
    farmer_is_irrigated: np.ndarray,
    replace_crop_calendar_unit_code: dict[int, int],
    selection_cache: dict[tuple[int, int, int, bool], np.ndarray],
) -> tuple[np.ndarray, int, int]:
    """Select MIRCA2000 calendars once per unique farmer-state combination.

    Farmers sharing the same MIRCA unit, main crop, secondary-crop timing class,
    and irrigation state always receive the same MIRCA2000 calendar. This helper
    therefore resolves each unique combination once and broadcasts the selected
    calendar back to all farmers. The supplied cache is reused across HRL years.

    Args:
        crop_calendar: Parsed MIRCA2000 crop-calendar dictionary.
        farmer_mirca_units: MIRCA2000 unit per compact farmer ID.
        farmer_main_crops: HRL-derived MIRCA main crop per compact farmer ID.
        farmer_secondary_crop_types: HRL secondary-crop timing class per compact
            farmer ID.
        farmer_is_irrigated: Irrigation state used for calendar selection per
            compact farmer ID.
        replace_crop_calendar_unit_code: Optional MIRCA unit replacement mapping.
        selection_cache: Mutable cache shared across processed HRL years. Keys are
            ``(lookup_unit, main_crop, secondary_crop_type, is_irrigated)`` and
            values are selected calendars with shape ``(3, 4)``.

    Returns:
        Tuple containing the farmer-aligned calendar array, the number of unique
        combinations in the current year, and the number of new cache entries.

    Raises:
        ValueError: If the farmer arrays are not one-dimensional and equally sized,
            or if a selected calendar has an unexpected shape.
    """
    farmer_mirca_units = np.asarray(farmer_mirca_units, dtype=np.int32)
    farmer_main_crops = np.asarray(farmer_main_crops, dtype=np.int32)
    farmer_secondary_crop_types = np.asarray(
        farmer_secondary_crop_types,
        dtype=np.int32,
    )
    farmer_is_irrigated = np.asarray(farmer_is_irrigated, dtype=bool)

    arrays = (
        farmer_mirca_units,
        farmer_main_crops,
        farmer_secondary_crop_types,
        farmer_is_irrigated,
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError(
            "All farmer calendar-selection arrays must be one-dimensional."
        )

    n_farmers = farmer_mirca_units.size
    if any(array.size != n_farmers for array in arrays[1:]):
        raise ValueError(
            "All farmer calendar-selection arrays must have equal length. "
            f"Got {[array.size for array in arrays]}."
        )

    lookup_units = farmer_mirca_units.copy()
    if replace_crop_calendar_unit_code:
        original_units = farmer_mirca_units
        for source_unit, target_unit in replace_crop_calendar_unit_code.items():
            lookup_units[original_units == int(source_unit)] = int(target_unit)

    key_matrix = np.column_stack(
        (
            lookup_units,
            farmer_main_crops,
            farmer_secondary_crop_types,
            farmer_is_irrigated.astype(np.int32),
        )
    ).astype(np.int32, copy=False)

    # Missing crops always map to the same empty calendar, independent of location
    # or irrigation state. Canonicalizing this key avoids many redundant entries.
    missing_crop = farmer_main_crops == -1
    key_matrix[missing_crop, 0] = -1
    key_matrix[missing_crop, 2] = 0
    key_matrix[missing_crop, 3] = 0

    unique_keys, inverse = np.unique(key_matrix, axis=0, return_inverse=True)
    unique_calendars = np.full(
        (unique_keys.shape[0], 3, 4),
        -1,
        dtype=np.int32,
    )

    n_cache_misses = 0
    for key_index, key_values in enumerate(unique_keys):
        lookup_unit, main_crop, secondary_crop_type, is_irrigated_int = (
            int(value) for value in key_values
        )
        cache_key = (
            lookup_unit,
            main_crop,
            secondary_crop_type,
            bool(is_irrigated_int),
        )

        selected_calendar = selection_cache.get(cache_key)
        if selected_calendar is None:
            full_calendar = _select_mirca2000_calendar_for_farmer(
                crop_calendar,
                mirca_unit=lookup_unit,
                main_crop=main_crop,
                secondary_crop_type=secondary_crop_type,
                is_irrigated=bool(is_irrigated_int),
                # lookup_unit is already normalized above; passing an empty mapping
                # prevents a target unit from being remapped a second time.
                replace_crop_calendar_unit_code={},
            )
            selected_calendar = np.asarray(
                full_calendar[:, [0, 2, 3, 4]],
                dtype=np.int32,
            )
            if selected_calendar.shape != (3, 4):
                raise ValueError(
                    "Selected MIRCA2000 calendar must have shape (3, 4) after "
                    f"column selection. Got {selected_calendar.shape}."
                )
            selected_calendar = np.ascontiguousarray(selected_calendar)
            selection_cache[cache_key] = selected_calendar
            n_cache_misses += 1

        unique_calendars[key_index] = selected_calendar

    return (
        unique_calendars[inverse],
        int(unique_keys.shape[0]),
        n_cache_misses,
    )


def get_day_index(date: date) -> int:
    """Get the day index (0-364) for a given date.

    Args:
        date: The date for which to get the day index.

    Returns:
        The day index (0-364).
    """
    return date.timetuple().tm_yday - 1  # 0-indexed


def get_growing_season_length(start_day_index: int, end_day_index: int) -> int:
    """Calculate the length of the growing season in days.

    Essentially calculates (end_day_index - start_day_index) mod 365, thus
    wrapping around the year if necessary. If start and end are the same,
    we assume the growing season lasts the entire year (365 days) rather
    than 0 days.

    Args:
        start_day_index: The starting day index (0-364).
        end_day_index: The ending day index (0-364).

    Returns:
        The length of the growing season in days.
    """
    length = (end_day_index - start_day_index) % 365
    if length == 0:
        return 365
    else:
        return length


def _sample_grid_values_at_farmers(
    data: xr.DataArray,
    farmer_locations: np.ndarray,
) -> np.ndarray:
    """Sample grid values at farmer locations.

    Args:
        data: Raster data to sample.
        farmer_locations: Farmer centroid coordinates.

    Returns:
        One-dimensional sampled values.
    """
    return sample_from_map(
        data.values,
        farmer_locations,
        data.rio.transform(recalc=True).to_gdal(),
    )


def _assign_irrigation_by_area_targets(
    *,
    farmer_crops: pd.DataFrame,
    farmer_areas_m2: np.ndarray,
    farmer_mirca_os_cells: np.ndarray,
    farmer_hand_m: np.ndarray,
    farmer_groundwater_depth_m: np.ndarray,
    rainfed_fraction: xr.DataArray,
    irrigated_fraction: xr.DataArray,
    surface_water_fraction_by_cell: dict[int, float],
    n_farmers: int,
    logger: Any,
    fallback_to_cell_irrigated_fraction: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign irrigation status and source to farmers by MIRCA-OS area targets.

    Farmers are grouped by MIRCA-OS grid cell and HRL-derived MIRCA crop class.
    For each group, the target irrigated area is calculated from the crop-specific
    MIRCA-OS irrigated share in that grid cell. Surface-water irrigation is then
    assigned first to farms with lower HAND. Groundwater irrigation is assigned to
    the remaining farms with lower groundwater depth.

    Args:
        farmer_crops: DataFrame with ``farmer_id`` and ``mirca_crop`` columns.
        farmer_areas_m2: Farm area per farmer, indexed by compact farmer ID.
        farmer_mirca_os_cells: MIRCA-OS fraction-grid cell ID per farmer.
        farmer_hand_m: HAND value per farmer.
        farmer_groundwater_depth_m: Initial groundwater depth per farmer.
        rainfed_fraction: MIRCA-OS rainfed crop-area fraction stack with
            dimensions ``crop``, ``y``, and ``x``.
        irrigated_fraction: MIRCA-OS irrigated crop-area fraction stack with
            dimensions ``crop``, ``y``, and ``x``.
        surface_water_fraction_by_cell: Surface-water share of irrigated area by
            MIRCA-OS grid cell.
        n_farmers: Number of compact farmers.
        logger: Logger used for warnings.
        fallback_to_cell_irrigated_fraction: If True, use the total cell-level
            irrigated fraction when the requested crop has zero MIRCA-OS area in
            the sampled cell.

    Returns:
        Tuple containing a boolean irrigated-farmer array and an adaptations
        matrix with surface-water and groundwater source flags.

    Raises:
        ValueError: If rainfed and irrigated fraction stacks are not aligned.
        ValueError: If the fraction stacks do not contain a ``crop`` dimension.
    """
    if "crop" not in rainfed_fraction.dims or "crop" not in irrigated_fraction.dims:
        raise ValueError(
            "rainfed_fraction and irrigated_fraction must have a crop dimension."
        )

    if rainfed_fraction.shape != irrigated_fraction.shape:
        raise ValueError(
            "rainfed_fraction and irrigated_fraction must have the same shape. "
            f"Got {rainfed_fraction.shape} and {irrigated_fraction.shape}."
        )

    if not np.array_equal(rainfed_fraction.x.values, irrigated_fraction.x.values):
        raise ValueError(
            "rainfed_fraction and irrigated_fraction x coordinates differ."
        )

    if not np.array_equal(rainfed_fraction.y.values, irrigated_fraction.y.values):
        raise ValueError(
            "rainfed_fraction and irrigated_fraction y coordinates differ."
        )

    adaptations = np.full(
        (
            n_farmers,
            max(
                [
                    FIELD_EXPANSION_ADAPTATION,
                    INDEX_INSURANCE_ADAPTATION,
                    IRRIGATION_EFFICIENCY_ADAPTATION_DRIP,
                    IRRIGATION_EFFICIENCY_ADAPTATION_SPRINKLER,
                    TRADITIONAL_INSURANCE_ADAPTATION,
                    PR_INSURANCE_ADAPTATION,
                    SURFACE_IRRIGATION_EQUIPMENT,
                    WELL_ADAPTATION,
                ]
            )
            + 1,
        ),
        0,
        dtype=np.bool_,
    )

    is_irrigated = np.full(n_farmers, False, dtype=bool)

    n_crops = rainfed_fraction.sizes["crop"]
    rainfed_values = np.nan_to_num(
        rainfed_fraction.values.reshape(n_crops, -1).astype(np.float64),
        nan=0.0,
    )
    irrigated_values = np.nan_to_num(
        irrigated_fraction.values.reshape(n_crops, -1).astype(np.float64),
        nan=0.0,
    )

    n_cells = rainfed_values.shape[1]

    def get_crop_irrigated_fraction(cell_id: int, crop_id: int) -> float:
        """Return the MIRCA-OS irrigated fraction for one cell-crop pair.

        Args:
            cell_id: Flattened MIRCA-OS grid-cell index.
            crop_id: Zero-based MIRCA crop-class index.

        Returns:
            Irrigated fraction in the inclusive range 0 to 1. Invalid indices and
            cells without represented crop area return 0.
        """
        if cell_id < 0 or cell_id >= n_cells:
            return 0.0

        if crop_id < 0 or crop_id >= n_crops:
            return 0.0

        rainfed_crop_area = float(rainfed_values[crop_id, cell_id])
        irrigated_crop_area = float(irrigated_values[crop_id, cell_id])
        total_crop_area = rainfed_crop_area + irrigated_crop_area

        if total_crop_area > 0:
            return irrigated_crop_area / total_crop_area

        if not fallback_to_cell_irrigated_fraction:
            return 0.0

        total_rainfed_cell_area = float(rainfed_values[:, cell_id].sum())
        total_irrigated_cell_area = float(irrigated_values[:, cell_id].sum())
        total_cell_area = total_rainfed_cell_area + total_irrigated_cell_area

        if total_cell_area <= 0:
            return 0.0

        return total_irrigated_cell_area / total_cell_area

    farmer_ids = farmer_crops["farmer_id"].to_numpy(dtype=np.int32)
    farmer_main_crops = farmer_crops["mirca_crop"].to_numpy(dtype=np.int32)

    crop_cell_table = pd.DataFrame(
        {
            "farmer_id": farmer_ids,
            "mirca_crop": farmer_main_crops,
            "mirca_os_cell": farmer_mirca_os_cells[farmer_ids],
            "area_m2": farmer_areas_m2[farmer_ids],
            "hand_m": farmer_hand_m[farmer_ids],
            "groundwater_depth_m": farmer_groundwater_depth_m[farmer_ids],
        }
    )

    crop_cell_table["hand_m"] = crop_cell_table["hand_m"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    crop_cell_table["groundwater_depth_m"] = crop_cell_table[
        "groundwater_depth_m"
    ].replace([np.inf, -np.inf], np.nan)

    crop_cell_table["hand_m"] = crop_cell_table["hand_m"].fillna(np.inf)
    crop_cell_table["groundwater_depth_m"] = crop_cell_table[
        "groundwater_depth_m"
    ].fillna(np.inf)

    n_groups_without_crop_fraction = 0

    for (mirca_os_cell, mirca_crop), group in crop_cell_table.groupby(
        ["mirca_os_cell", "mirca_crop"],
        sort=False,
    ):
        mirca_os_cell = int(mirca_os_cell)
        mirca_crop = int(mirca_crop)

        rainfed_crop_area = 0.0
        irrigated_crop_area = 0.0
        if 0 <= mirca_os_cell < n_cells and 0 <= mirca_crop < n_crops:
            rainfed_crop_area = float(rainfed_values[mirca_crop, mirca_os_cell])
            irrigated_crop_area = float(irrigated_values[mirca_crop, mirca_os_cell])

        if rainfed_crop_area + irrigated_crop_area <= 0:
            n_groups_without_crop_fraction += 1

        crop_irrigated_fraction = get_crop_irrigated_fraction(
            mirca_os_cell,
            mirca_crop,
        )

        if crop_irrigated_fraction <= 0:
            continue

        total_group_area = float(group["area_m2"].sum())
        target_irrigated_area = total_group_area * crop_irrigated_fraction

        surface_water_fraction = surface_water_fraction_by_cell.get(mirca_os_cell, 0.0)
        target_surface_water_area = target_irrigated_area * surface_water_fraction
        target_groundwater_area = target_irrigated_area - target_surface_water_area

        surface_sorted = group.sort_values(
            ["hand_m", "farmer_id"],
            ascending=[True, True],
        )

        assigned_surface_area = 0.0
        surface_farmer_ids: list[int] = []

        for row in surface_sorted.itertuples(index=False):
            if assigned_surface_area >= target_surface_water_area:
                break

            farmer_id = int(row.farmer_id)
            surface_farmer_ids.append(farmer_id)
            assigned_surface_area += float(row.area_m2)

        if surface_farmer_ids:
            surface_farmer_ids_array = np.asarray(surface_farmer_ids, dtype=np.int32)
            adaptations[
                surface_farmer_ids_array,
                SURFACE_IRRIGATION_EQUIPMENT,
            ] = True
            is_irrigated[surface_farmer_ids_array] = True

        remaining = group.loc[~group["farmer_id"].isin(surface_farmer_ids)]
        groundwater_sorted = remaining.sort_values(
            ["groundwater_depth_m", "farmer_id"],
            ascending=[True, True],
        )

        assigned_groundwater_area = 0.0
        groundwater_farmer_ids: list[int] = []

        for row in groundwater_sorted.itertuples(index=False):
            if assigned_groundwater_area >= target_groundwater_area:
                break

            farmer_id = int(row.farmer_id)
            groundwater_farmer_ids.append(farmer_id)
            assigned_groundwater_area += float(row.area_m2)

        if groundwater_farmer_ids:
            groundwater_farmer_ids_array = np.asarray(
                groundwater_farmer_ids,
                dtype=np.int32,
            )
            adaptations[
                groundwater_farmer_ids_array,
                WELL_ADAPTATION,
            ] = True
            is_irrigated[groundwater_farmer_ids_array] = True

        if (
            target_irrigated_area > 0
            and assigned_surface_area + assigned_groundwater_area == 0
        ):
            logger.warning(
                "No irrigation assigned for MIRCA-OS cell %s and crop %s despite "
                "positive target area.",
                mirca_os_cell,
                mirca_crop,
            )

    if n_groups_without_crop_fraction > 0:
        logger.info(
            "%s MIRCA-OS cell-crop farmer group(s) had no crop-specific MIRCA-OS "
            "area fraction. Cell-level irrigation fallback was %s.",
            n_groups_without_crop_fraction,
            "enabled" if fallback_to_cell_irrigated_fraction else "disabled",
        )

    return is_irrigated, adaptations


def _build_surface_water_fraction_lookup(
    fraction_sw_irrigation_data: xr.DataArray,
    fraction_gw_irrigation_data: xr.DataArray,
    mirca_cell_grid: xr.DataArray,
) -> dict[int, float]:
    """Build surface-water fractions by MIRCA grid cell.

    Args:
        fraction_sw_irrigation_data: Surface-water irrigation fraction raster.
        fraction_gw_irrigation_data: Groundwater irrigation fraction raster.
        mirca_cell_grid: Linear MIRCA grid-cell ID raster.

    Returns:
        Mapping from MIRCA grid-cell ID to surface-water share of irrigated area.
    """
    sw_values = fraction_sw_irrigation_data.values.ravel()
    gw_values = fraction_gw_irrigation_data.values.ravel()
    cell_values = mirca_cell_grid.values.ravel().astype(np.int32)

    lookup: dict[int, float] = {}

    for cell_id in np.unique(cell_values[cell_values >= 0]):
        mask = cell_values == cell_id
        sw = float(np.nanmean(sw_values[mask]))
        gw = float(np.nanmean(gw_values[mask]))
        total = sw + gw

        if total <= 0 or np.isnan(total):
            lookup[int(cell_id)] = 0.0
        else:
            lookup[int(cell_id)] = sw / total

    return lookup


def parse_MIRCA_file(
    parsed_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
    crop_calendar_lines: list[str],
    MIRCA_units: list[int],
    is_irrigated: bool,
) -> dict[int, list[tuple[float, TwoDArrayInt32]]]:
    """Parse one MIRCA2000 crop-calendar file.

    The parser converts monthly planting and harvest information to day indices,
    retains at most two positive-area rotations per crop, and appends the resulting
    calendar matrices to the supplied dictionary.

    Args:
        parsed_calendar: Dictionary receiving parsed calendars by MIRCA unit.
        crop_calendar_lines: Raw lines from a MIRCA2000 calendar file.
        MIRCA_units: MIRCA unit codes that should be retained.
        is_irrigated: Whether the source file represents irrigated calendars.

    Returns:
        The input dictionary with parsed calendars appended.

    Raises:
        NotImplementedError: If a positive-area crop entry contains neither one nor
            two supported rotations after filtering.
    """
    lines: list[str] = [line.strip() for line in crop_calendar_lines if line.strip()]
    lines = lines[4:]

    for raw_line in lines:
        values: list[str] = raw_line.replace("  ", " ").split(" ")
        unit_code: int = int(values[0])
        if unit_code not in MIRCA_units:
            continue
        if unit_code not in parsed_calendar:
            parsed_calendar[unit_code] = []
        crop_class: int = int(values[1]) - 1  # minus one to make it zero based
        number_of_rotations: int = int(values[2])
        if number_of_rotations == 0:
            continue
        crops: list[str] = values[3:]
        crop_rotations: list[tuple[int, int, float]] = []
        for rotation in range(number_of_rotations):
            area: float = float(crops[rotation * 3])
            if area == 0:
                continue
            start_month: int = int(crops[rotation * 3 + 1])
            end_month: int = int(crops[rotation * 3 + 2])
            start_day_index: int = get_day_index(date(2000, start_month, 1))
            end_day_index: int = get_day_index(
                date(2000, end_month, calendar.monthrange(2000, end_month)[1])
            )
            growth_length: int = get_growing_season_length(
                start_day_index, end_day_index
            )
            crop_rotations.append((start_day_index, growth_length, area))

        # discard crop rotations with zero area
        crop_rotations = [
            crop_rotation for crop_rotation in crop_rotations if crop_rotation[2] > 0
        ]

        crop_rotations = sorted(crop_rotations, key=lambda x: x[2])  # sort by area
        if len(crop_rotations) > 2:
            crop_rotations = crop_rotations[-2:]
            import warnings

            warnings.warn(
                "More than two crop rotations found; discarding the rotation "
                "with the lowest represented area."
            )
        if len(crop_rotations) == 1:
            start_day_index, growth_length, area = crop_rotations[0]
            crop_rotation: tuple[float, TwoDArrayInt32] = (
                area,
                np.array(
                    (
                        (
                            crop_class,
                            is_irrigated,
                            start_day_index,
                            growth_length,
                            0,
                        ),
                        (-1, -1, -1, -1, -1),
                        (-1, -1, -1, -1, -1),
                    )
                ),
            )  # -1 means no crop
            parsed_calendar[unit_code].append(crop_rotation)
        elif len(crop_rotations) == 2:
            # if crop rotations start on the same day, they cannot be implemented
            # by the same farmer, so we split them
            # TODO: Verify that this branch is limited to non-overlapping rotations.
            if crop_rotations[0][0] == crop_rotations[1][0]:
                for crop_rotation in crop_rotations:
                    start_day_index, growth_length, area = crop_rotation
                    crop_rotation_entry: tuple[float, TwoDArrayInt32] = (
                        area,
                        np.array(
                            (
                                (
                                    crop_class,
                                    is_irrigated,
                                    start_day_index,
                                    growth_length,
                                    0,
                                ),
                                (-1, -1, -1, -1, -1),
                                (-1, -1, -1, -1, -1),
                            ),
                            dtype=np.int32,
                        ),
                    )
                    parsed_calendar[unit_code].append(crop_rotation_entry)
            # if the crop rotations are consecutive, we assume multi-cropping.
            else:
                crop_rotation_entry = (
                    crop_rotations[1][2] - crop_rotations[0][2],
                    np.array(
                        (
                            (
                                crop_class,
                                is_irrigated,
                                crop_rotations[1][0],
                                crop_rotations[1][1],
                                0,
                            ),
                            (-1, -1, -1, -1, -1),
                            (-1, -1, -1, -1, -1),
                        ),
                        dtype=np.int32,
                    ),  # -1 means no crop
                )
                parsed_calendar[unit_code].append(crop_rotation_entry)
                crop_rotation_entry = (
                    crop_rotations[0][2],
                    np.array(
                        (
                            (
                                crop_class,
                                is_irrigated,
                                crop_rotations[0][0],
                                crop_rotations[0][1],
                                0,
                            ),
                            (
                                crop_class,
                                is_irrigated,
                                crop_rotations[1][0],
                                crop_rotations[1][1],
                                0,
                            ),
                            (-1, -1, -1, -1, -1),
                        ),
                        dtype=np.int32,
                    ),
                )
            parsed_calendar[unit_code].append(crop_rotation_entry)
            assert crop_rotation_entry[1][0][2] != crop_rotation_entry[1][1][2]
        else:
            raise NotImplementedError
    return parsed_calendar


def parse_MIRCA2000_crop_calendar(
    data_catalog: DataCatalog, MIRCA_units: list[int]
) -> dict[int, list[tuple[float, TwoDArrayInt32]]]:
    """Parse MIRCA2000 crop calendars for given MIRCA units.

    Args:
        data_catalog: The data catalog containing the MIRCA2000 files.
        MIRCA_units: The list of MIRCA unit codes to parse.

    Returns:
        A dictionary containing the parsed crop calendars.

    Raises:
        TypeError: If the calendar data is not provided as a list of strings.
    """
    rainfed_source = data_catalog.fetch("mirca2000_cropping_calendar_rainfed").read()
    irrigated_source = data_catalog.fetch(
        "mirca2000_cropping_calendar_irrigated"
    ).read()

    if not isinstance(rainfed_source, list) or not isinstance(irrigated_source, list):
        raise TypeError("Expected MIRCA2000 calendar lines as a list of strings.")

    rainfed_lines: list[str] = rainfed_source
    irrigated_lines: list[str] = irrigated_source

    mirca2000_data: dict[int, list[tuple[float, TwoDArrayInt32]]] = {}

    mirca2000_data = parse_MIRCA_file(
        mirca2000_data,
        rainfed_lines,
        MIRCA_units,
        is_irrigated=False,
    )
    mirca2000_data = parse_MIRCA_file(
        mirca2000_data,
        irrigated_lines,
        MIRCA_units,
        is_irrigated=True,
    )

    return mirca2000_data


def _native_hrl_crop_category_areas_m2(
    crop_types: xr.DataArray,
    secondary_crop: xr.DataArray,
    clip_geometry: BaseGeometry,
    *,
    geometry_crs: str = "EPSG:4326",
    chunk_rows: int = 4096,
) -> dict[int, float]:
    """Calculate native HRL area by combined crop category inside a geometry.

    The source raster is scanned in row chunks before modal reprojection to the
    model grid. Minority crop categories inside a coarse model cell therefore
    remain represented without loading the complete regional HRL raster into one
    in-memory NumPy array.

    Args:
        crop_types: Native HRL Crop Types raster.
        secondary_crop: Native HRL Secondary Crops raster on the same grid.
        clip_geometry: Active regional geometry used to restrict source pixels.
        geometry_crs: CRS of ``clip_geometry``.
        chunk_rows: Number of native HRL rows processed at once.

    Returns:
        Mapping from combined primary-plus-secondary HRL crop code to cultivated
        area in square metres.

    Raises:
        ValueError: If source grids are inconsistent, lack a CRS, or are not 2D.
    """
    assert_matching_raster_grid(crop_types, secondary_crop)
    if crop_types.rio.crs is None:
        raise ValueError("Native HRL crop rasters must have a CRS.")
    if crop_types.ndim != 2 or secondary_crop.ndim != 2:
        raise ValueError("Native HRL crop rasters must be two-dimensional.")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be at least 1.")

    source_geometry = (
        gpd.GeoSeries([clip_geometry], crs=geometry_crs)
        .to_crs(crop_types.rio.crs)
        .iloc[0]
    )
    source_crs = crop_types.rio.crs
    projected_cell_area_m2: float | None = None
    if source_crs.is_projected:
        source_transform = crop_types.rio.transform(recalc=True)
        projected_cell_area_m2 = abs(
            float(source_transform.a) * float(source_transform.e)
        )

    totals: dict[int, float] = {}
    n_rows = crop_types.sizes["y"]
    for y_start in range(0, n_rows, chunk_rows):
        y_stop = min(y_start + chunk_rows, n_rows)
        crop_chunk_da = crop_types.isel(y=slice(y_start, y_stop))
        secondary_chunk_da = secondary_crop.isel(y=slice(y_start, y_stop))
        crop_values = crop_chunk_da.values
        secondary_values = secondary_chunk_da.values
        if np.issubdtype(crop_values.dtype, np.floating):
            crop_values = np.nan_to_num(crop_values, nan=_HRL_OUTSIDE_AREA_CODE)
        if np.issubdtype(secondary_values.dtype, np.floating):
            secondary_values = np.nan_to_num(
                secondary_values, nan=_HRL_OUTSIDE_AREA_CODE
            )

        crop_values = np.asarray(crop_values, dtype=np.int32)
        secondary_values = np.asarray(secondary_values, dtype=np.int32)
        valid_crop = (crop_values > _HRL_NO_CROPLAND_CODE) & (
            crop_values != _HRL_OUTSIDE_AREA_CODE
        )
        combined_values = combine_crop_and_secondary_values(
            crop_values,
            secondary_values,
        )
        inside_geometry = features.geometry_mask(
            [source_geometry.__geo_interface__],
            out_shape=crop_values.shape,
            transform=crop_chunk_da.rio.transform(recalc=True),
            invert=True,
            all_touched=False,
        )
        valid = valid_crop & inside_geometry
        if not valid.any():
            continue

        crop_codes, inverse = np.unique(
            combined_values[valid],
            return_inverse=True,
        )
        if projected_cell_area_m2 is not None:
            crop_areas_m2 = (
                np.bincount(inverse).astype(np.float64) * projected_cell_area_m2
            )
        else:
            chunk_cell_area_m2 = raster_cell_area_m2(crop_chunk_da)
            crop_areas_m2 = np.bincount(
                inverse,
                weights=chunk_cell_area_m2[valid],
            )

        for crop_code, crop_area_m2 in zip(
            crop_codes,
            crop_areas_m2,
            strict=True,
        ):
            crop_code_int = int(crop_code)
            if crop_code_int < 0 or crop_area_m2 <= 0.0:
                continue
            totals[crop_code_int] = totals.get(crop_code_int, 0.0) + float(crop_area_m2)

    return totals


def _crop_area_fit_scores(diagnostics: pd.DataFrame) -> dict[str, float]:
    """Calculate bounded crop-area alignment scores.

    The area score compares absolute category areas, the share score compares only
    composition, and the total-area score compares summed cultivated area. Negative
    crop codes are excluded from all three calculations.

    Args:
        diagnostics: Crop-area diagnostic table containing source and assigned area
            per crop code.

    Returns:
        Dictionary containing source and assigned totals, their percentage
        difference, and bounded total-area, crop-share, and crop-area fit scores.
    """
    if diagnostics.empty:
        return {
            "source_area_m2": 0.0,
            "assigned_area_m2": 0.0,
            "total_area_difference_pct": np.nan,
            "total_area_fit_score": np.nan,
            "crop_share_fit_score": np.nan,
            "crop_area_fit_score": np.nan,
        }

    positive = diagnostics["crop_code"].to_numpy(dtype=np.int32) > 0
    source = diagnostics.loc[positive, "source_area_m2"].to_numpy(
        dtype=np.float64, copy=True
    )
    assigned = diagnostics.loc[positive, "assigned_area_m2"].to_numpy(
        dtype=np.float64, copy=True
    )

    source_total = float(source.sum())
    assigned_total = float(assigned.sum())
    total_denominator = source_total + assigned_total
    maximum_total = max(source_total, assigned_total)

    total_area_difference_pct = (
        (assigned_total - source_total) / source_total * 100.0
        if source_total > 0.0
        else np.nan
    )
    total_area_fit_score = (
        min(source_total, assigned_total) / maximum_total * 100.0
        if maximum_total > 0.0
        else np.nan
    )

    absolute_category_error = float(np.abs(assigned - source).sum())
    crop_area_fit_score = (
        (1.0 - absolute_category_error / total_denominator) * 100.0
        if total_denominator > 0.0
        else np.nan
    )

    if source_total > 0.0 and assigned_total > 0.0:
        source_share = source / source_total
        assigned_share = assigned / assigned_total
        crop_share_fit_score = (
            1.0 - 0.5 * float(np.abs(source_share - assigned_share).sum())
        ) * 100.0
    else:
        crop_share_fit_score = np.nan

    return {
        "source_area_m2": source_total,
        "assigned_area_m2": assigned_total,
        "total_area_difference_pct": total_area_difference_pct,
        "total_area_fit_score": float(np.clip(total_area_fit_score, 0.0, 100.0))
        if np.isfinite(total_area_fit_score)
        else np.nan,
        "crop_share_fit_score": float(np.clip(crop_share_fit_score, 0.0, 100.0))
        if np.isfinite(crop_share_fit_score)
        else np.nan,
        "crop_area_fit_score": float(np.clip(crop_area_fit_score, 0.0, 100.0))
        if np.isfinite(crop_area_fit_score)
        else np.nan,
    }


def _multi_year_crop_area_comparison(
    diagnostics: pd.DataFrame,
    *,
    aggregate_to_main_crop: bool = False,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compare raw and final crop areas over every year-crop combination.

    The comparison is symmetric: each year-crop pair is evaluated against the
    larger of its raw HRL area and final farmer-assigned area. This means that
    both missing final area and spurious final area reduce the score. The
    area-weighted score gives larger year-crop pairs more influence, while the
    balanced score gives every represented year-crop pair equal influence.

    Args:
        diagnostics: Per-region crop-area diagnostics containing ``year``,
            ``crop_code``, ``source_area_m2``, and ``assigned_area_m2``.
        aggregate_to_main_crop: Whether combined primary-plus-secondary HRL crop
            codes should first be collapsed to their main crop classes.

    Returns:
        Tuple containing a per-year, per-crop comparison table and a dictionary
        with multi-year area-weighted and balanced summary scores.

    Raises:
        ValueError: If one or more required diagnostic columns are absent.
    """
    required_columns = {
        "year",
        "crop_code",
        "source_area_m2",
        "assigned_area_m2",
    }
    missing_columns = required_columns - set(diagnostics.columns)
    if missing_columns:
        raise ValueError(
            "Crop-area diagnostics are missing required column(s): "
            f"{sorted(missing_columns)}"
        )

    table = diagnostics.loc[
        diagnostics["crop_code"].to_numpy(dtype=np.int32) > 0,
        ["year", "crop_code", "source_area_m2", "assigned_area_m2"],
    ].copy()

    if aggregate_to_main_crop and not table.empty:
        table["crop_code"] = (
            table["crop_code"].to_numpy(dtype=np.int32) // 10 * 10
        ).astype(np.int32)

    table = (
        table.groupby(["year", "crop_code"], as_index=False)[
            ["source_area_m2", "assigned_area_m2"]
        ]
        .sum()
        .sort_values(["year", "crop_code"])
        .reset_index(drop=True)
    )
    table = table.loc[
        (table["source_area_m2"] > 0.0) | (table["assigned_area_m2"] > 0.0)
    ].reset_index(drop=True)

    if table.empty:
        return table, {
            "n_year_crop_pairs": 0,
            "raw_area_m2": 0.0,
            "final_area_m2": 0.0,
            "net_difference_pct": np.nan,
            "area_weighted_fit_pct": np.nan,
            "balanced_fit_pct": np.nan,
            "area_weighted_error_pct": np.nan,
            "balanced_error_pct": np.nan,
        }

    raw_area = table["source_area_m2"].to_numpy(dtype=np.float64)
    final_area = table["assigned_area_m2"].to_numpy(dtype=np.float64)
    comparison_area = np.maximum(raw_area, final_area)
    overlapping_area = np.minimum(raw_area, final_area)
    absolute_error = np.abs(final_area - raw_area)

    # A pair receives 100 when raw and final areas are equal, and 0 when area
    # occurs on only one side. Using max(raw, final) keeps the metric bounded and
    # also penalizes crop categories introduced only in the final assignment.
    pair_fit_pct = (
        np.divide(
            overlapping_area,
            comparison_area,
            out=np.zeros_like(overlapping_area),
            where=comparison_area > 0.0,
        )
        * 100.0
    )
    pair_error_pct = 100.0 - pair_fit_pct
    total_comparison_area = float(comparison_area.sum())

    table["main_crop"] = (
        table["crop_code"].to_numpy(dtype=np.int32) // 10 * 10
    ).astype(np.int32)
    table["secondary_crop"] = np.where(
        aggregate_to_main_crop,
        0,
        table["crop_code"].to_numpy(dtype=np.int32) % 10,
    ).astype(np.int32)
    table["difference_m2"] = final_area - raw_area
    table["difference_pct_raw"] = (
        np.divide(
            final_area - raw_area,
            raw_area,
            out=np.full(raw_area.size, np.nan, dtype=np.float64),
            where=raw_area > 0.0,
        )
        * 100.0
    )
    table["pair_fit_pct"] = pair_fit_pct
    table["area_weight_pct"] = (
        np.divide(
            comparison_area,
            total_comparison_area,
            out=np.zeros_like(comparison_area),
            where=total_comparison_area > 0.0,
        )
        * 100.0
    )

    raw_total = float(raw_area.sum())
    final_total = float(final_area.sum())
    area_weighted_fit_pct = (
        float(overlapping_area.sum()) / total_comparison_area * 100.0
        if total_comparison_area > 0.0
        else np.nan
    )
    area_weighted_error_pct = (
        float(absolute_error.sum()) / total_comparison_area * 100.0
        if total_comparison_area > 0.0
        else np.nan
    )

    summary: dict[str, float | int] = {
        "n_year_crop_pairs": int(len(table)),
        "raw_area_m2": raw_total,
        "final_area_m2": final_total,
        "net_difference_pct": (
            (final_total - raw_total) / raw_total * 100.0 if raw_total > 0.0 else np.nan
        ),
        "area_weighted_fit_pct": area_weighted_fit_pct,
        "balanced_fit_pct": float(pair_fit_pct.mean()),
        "area_weighted_error_pct": area_weighted_error_pct,
        "balanced_error_pct": float(pair_error_pct.mean()),
    }
    return table, summary


def _format_multi_year_crop_area_comparison(comparison: pd.DataFrame) -> str:
    """Format all raw-versus-final year-crop area comparisons for logging.

    Args:
        comparison: Table returned by ``_multi_year_crop_area_comparison``.

    Returns:
        Human-readable table with one row per represented year-crop pair.
    """
    if comparison.empty:
        return "no positive year-crop area pairs"

    table = comparison.copy()
    table["raw_km2"] = table["source_area_m2"] / 1_000_000.0
    table["final_km2"] = table["assigned_area_m2"] / 1_000_000.0
    table["difference_km2"] = table["difference_m2"] / 1_000_000.0
    return (
        table[
            [
                "year",
                "crop_code",
                "main_crop",
                "secondary_crop",
                "raw_km2",
                "final_km2",
                "difference_km2",
                "difference_pct_raw",
                "pair_fit_pct",
                "area_weight_pct",
            ]
        ]
        .round(
            {
                "raw_km2": 3,
                "final_km2": 3,
                "difference_km2": 3,
                "difference_pct_raw": 1,
                "pair_fit_pct": 1,
                "area_weight_pct": 2,
            }
        )
        .to_string(index=False)
    )


def _format_hrl_crop_area_alignment(
    diagnostics: pd.DataFrame,
    *,
    top_n: int,
) -> str:
    """Format the largest HRL-versus-agent crop-area differences for logging.

    Args:
        diagnostics: Crop-area diagnostic table for one region and year.
        top_n: Maximum number of crop categories to include.

    Returns:
        Human-readable table ordered by the largest represented category area, or a
        short message when no positive crop categories are available.
    """
    if diagnostics.empty:
        return "no crop categories"

    table = diagnostics.copy()
    table = table.loc[
        (table["source_area_m2"] > 0.0) | (table["assigned_area_m2"] > 0.0)
    ].copy()
    if table.empty:
        return "no positive crop area"

    table["main_crop"] = np.where(
        table["crop_code"] > 0,
        (table["crop_code"] // 10) * 10,
        -1,
    )
    table["secondary_crop"] = np.where(
        table["crop_code"] > 0,
        table["crop_code"] % 10,
        0,
    )
    table["source_km2"] = table["source_area_m2"] / 1_000_000.0
    table["agent_km2"] = table["assigned_area_m2"] / 1_000_000.0
    table["difference_km2"] = (
        table["assigned_area_m2"] - table["source_area_m2"]
    ) / 1_000_000.0
    source_values = table["source_area_m2"].to_numpy(dtype=np.float64)
    assigned_values = table["assigned_area_m2"].to_numpy(dtype=np.float64)
    table["difference_pct"] = (
        np.divide(
            assigned_values - source_values,
            source_values,
            out=np.full(len(table), np.nan, dtype=np.float64),
            where=source_values > 0.0,
        )
        * 100.0
    )
    table["share_difference_pp"] = (
        table["assigned_share"] - table["source_share"]
    ) * 100.0
    table["ranking_area"] = np.maximum(
        table["source_area_m2"],
        table["assigned_area_m2"],
    )
    table = table.sort_values(
        ["ranking_area", "crop_code"],
        ascending=[False, True],
    ).head(max(top_n, 1))
    return (
        table[
            [
                "crop_code",
                "main_crop",
                "secondary_crop",
                "source_km2",
                "agent_km2",
                "difference_km2",
                "difference_pct",
                "share_difference_pp",
            ]
        ]
        .round(
            {
                "source_km2": 3,
                "agent_km2": 3,
                "difference_km2": 3,
                "difference_pct": 1,
                "share_difference_pp": 2,
            }
        )
        .to_string(index=False)
    )


def _crop_area_targets_from_model_grid(
    crop_values: np.ndarray,
    cultivated_mask: np.ndarray,
    cell_area_m2: np.ndarray,
) -> dict[int, float]:
    """Aggregate positive crop areas on the selected model grid.

    Args:
        crop_values: Combined crop code per model cell.
        cultivated_mask: Boolean mask selecting cells in the static farm domain.
        cell_area_m2: Full area of each model cell in square metres.

    Returns:
        Mapping from positive crop code to represented full-cell area in square
        metres.

    Raises:
        ValueError: If the crop, mask, and cell-area arrays do not have identical
            shapes.
    """
    crop_values = np.asarray(crop_values, dtype=np.int32)
    cultivated_mask = np.asarray(cultivated_mask, dtype=bool)
    cell_area_m2 = np.asarray(cell_area_m2, dtype=np.float64)
    if not (crop_values.shape == cultivated_mask.shape == cell_area_m2.shape):
        raise ValueError("crop_values, cultivated_mask, and cell_area_m2 must align.")
    valid = cultivated_mask & (crop_values > 0)
    if not valid.any():
        return {}
    crop_codes, inverse = np.unique(crop_values[valid], return_inverse=True)
    crop_areas = np.bincount(inverse, weights=cell_area_m2[valid])
    return {
        int(code): float(area)
        for code, area in zip(crop_codes, crop_areas, strict=True)
    }


def _crop_area_diagnostics_from_assignments(
    assigned_crop_codes: np.ndarray,
    farmer_areas_m2: np.ndarray,
    source_crop_areas_m2: dict[int, float],
) -> pd.DataFrame:
    """Compare existing farmer crop assignments with source area targets.

    Args:
        assigned_crop_codes: One assigned combined crop code per farmer.
        farmer_areas_m2: Area of each farmer in square metres.
        source_crop_areas_m2: Source area target by combined crop code.

    Returns:
        DataFrame containing source area, assigned area, differences, and area shares
        for the union of source and assigned crop codes.

    Raises:
        ValueError: If crop assignments and farmer areas are not aligned.
    """
    assigned_crop_codes = np.asarray(assigned_crop_codes, dtype=np.int32)
    farmer_areas_m2 = np.asarray(farmer_areas_m2, dtype=np.float64)
    if assigned_crop_codes.shape != farmer_areas_m2.shape:
        raise ValueError("assigned_crop_codes and farmer_areas_m2 must align.")

    assigned_codes, inverse = np.unique(assigned_crop_codes, return_inverse=True)
    assigned_areas = np.bincount(inverse, weights=farmer_areas_m2)
    assigned_lookup = {
        int(code): float(area)
        for code, area in zip(assigned_codes, assigned_areas, strict=True)
    }
    crop_codes = sorted(set(source_crop_areas_m2) | set(assigned_lookup))
    source = np.asarray(
        [source_crop_areas_m2.get(code, 0.0) for code in crop_codes],
        dtype=np.float64,
    )
    assigned = np.asarray(
        [assigned_lookup.get(code, 0.0) for code in crop_codes],
        dtype=np.float64,
    )
    positive_codes = np.asarray(crop_codes, dtype=np.int32) > 0
    source_total = float(source[positive_codes].sum())
    assigned_total = float(assigned[positive_codes].sum())
    table = pd.DataFrame(
        {
            "crop_code": np.asarray(crop_codes, dtype=np.int32),
            "source_area_m2": source,
            "adjusted_target_area_m2": source,
            "assigned_area_m2": assigned,
        }
    )
    table["difference_from_source_m2"] = assigned - source
    table["difference_from_adjusted_target_m2"] = assigned - source
    table["source_share"] = np.divide(
        source,
        source_total,
        out=np.zeros_like(source),
        where=source_total > 0.0,
    )
    table["assigned_share"] = np.divide(
        assigned,
        assigned_total,
        out=np.zeros_like(assigned),
        where=assigned_total > 0.0,
    )
    table["positive_target_scale"] = 1.0
    return table


def _aggregate_crop_alignment_to_main_categories(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate combined HRL crop diagnostics to main crop classes.

    Args:
        diagnostics: Diagnostic table using combined primary-secondary crop codes.

    Returns:
        Diagnostic table with secondary-crop suffixes collapsed into their main HRL
        crop categories and all area and share fields recomputed.
    """
    if diagnostics.empty:
        return diagnostics.copy()
    table = diagnostics.copy()
    table["crop_code"] = np.where(
        table["crop_code"].to_numpy(dtype=np.int32) > 0,
        (table["crop_code"].to_numpy(dtype=np.int32) // 10) * 10,
        -1,
    ).astype(np.int32)
    table = table.groupby("crop_code", as_index=False)[
        [
            "source_area_m2",
            "adjusted_target_area_m2",
            "assigned_area_m2",
        ]
    ].sum()
    table["difference_from_source_m2"] = (
        table["assigned_area_m2"] - table["source_area_m2"]
    )
    table["difference_from_adjusted_target_m2"] = (
        table["assigned_area_m2"] - table["adjusted_target_area_m2"]
    )
    positive = table["crop_code"] > 0
    source_total = float(table.loc[positive, "source_area_m2"].sum())
    assigned_total = float(table.loc[positive, "assigned_area_m2"].sum())
    table["source_share"] = (
        table["source_area_m2"].to_numpy(dtype=np.float64) / source_total
        if source_total > 0.0
        else np.zeros(len(table), dtype=np.float64)
    )
    table["assigned_share"] = (
        table["assigned_area_m2"].to_numpy(dtype=np.float64) / assigned_total
        if assigned_total > 0.0
        else np.zeros(len(table), dtype=np.float64)
    )
    table["positive_target_scale"] = np.divide(
        table["adjusted_target_area_m2"].to_numpy(dtype=np.float64),
        table["source_area_m2"].to_numpy(dtype=np.float64),
        out=np.ones(len(table), dtype=np.float64),
        where=table["source_area_m2"].to_numpy(dtype=np.float64) > 0.0,
    )
    return table


def _reproject_HRL_year_to_subgrid(
    crop_types: xr.DataArray,
    secondary_crop: xr.DataArray,
    template: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate one native-resolution HRL year to the final model subgrid.

    Native Crop Types states remain distinct during reprojection:

    - positive HRL crop codes are active crops;
    - ``0`` is native no-cropland;
    - ``65535`` is outside the HRL product area.

    Active combined primary-plus-secondary codes are aggregated with modal
    resampling. Active-crop fraction and HRL coverage fraction are aggregated
    separately with average resampling. A destination cell without an active
    modal crop is returned as native no-cropland (``0``) when it has source
    coverage and as ``_HRL_MISSING_CROP_CODE`` when it has no HRL coverage.
    Conversion of no-cropland to model fallow (``-1``) is deliberately deferred
    until the complete multi-year agricultural union is known.

    Args:
        crop_types: Native HRL Crop Types raster.
        secondary_crop: Native HRL Secondary Crops raster on the same grid.
        template: Final model-grid template for the selected region window.

    Returns:
        Tuple containing modal combined crop state, active-crop fraction, and
        HRL coverage fraction on the model-grid template.

    Raises:
        ValueError: If the source rasters are not aligned or the template has no
            CRS.
    """
    assert_matching_raster_grid(crop_types, secondary_crop)
    if template.rio.crs is None:
        raise ValueError("The regional subgrid template must have a CRS.")

    crop_values = crop_types.values
    secondary_values = secondary_crop.values
    if np.issubdtype(crop_values.dtype, np.floating):
        crop_values = np.nan_to_num(
            crop_values,
            nan=_HRL_OUTSIDE_AREA_CODE,
        )
    if np.issubdtype(secondary_values.dtype, np.floating):
        secondary_values = np.nan_to_num(
            secondary_values,
            nan=_HRL_OUTSIDE_AREA_CODE,
        )

    crop_values = np.ascontiguousarray(crop_values.astype(np.int32, copy=False))
    secondary_values = np.ascontiguousarray(
        secondary_values.astype(np.int32, copy=False)
    )
    has_hrl_coverage = crop_values != _HRL_OUTSIDE_AREA_CODE
    active_crop = (crop_values > _HRL_NO_CROPLAND_CODE) & has_hrl_coverage

    combined_values = combine_crop_and_secondary_values(
        crop_values,
        secondary_values,
    )
    # Non-active native states are excluded from modal crop resampling. Their
    # distinction is reconstructed from the independently reprojected HRL
    # coverage fraction below.
    combined_values[~active_crop] = _HRL_MISSING_CROP_CODE

    combined = crop_types.copy(data=combined_values)
    combined.attrs = {}
    combined = combined.rio.write_crs(crop_types.rio.crs)
    combined = combined.rio.write_nodata(_HRL_MISSING_CROP_CODE)
    combined_subgrid = combined.rio.reproject_match(
        template,
        resampling=Resampling.mode,
        nodata=_HRL_MISSING_CROP_CODE,
    )
    combined_subgrid_values = combined_subgrid.values
    if np.issubdtype(combined_subgrid_values.dtype, np.floating):
        combined_subgrid_values = np.nan_to_num(
            combined_subgrid_values,
            nan=_HRL_MISSING_CROP_CODE,
        )
    combined_subgrid_values = combined_subgrid_values.astype(np.int32, copy=False)

    cultivated = crop_types.copy(data=active_crop.astype(np.float32))
    cultivated.attrs = {}
    cultivated = cultivated.rio.write_crs(crop_types.rio.crs)
    cultivated = cultivated.rio.write_nodata(None)
    cultivated_fraction = cultivated.rio.reproject_match(
        template,
        resampling=Resampling.average,
        nodata=0.0,
    )
    cultivated_fraction_values = np.nan_to_num(
        cultivated_fraction.values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)
    cultivated_fraction_values = np.clip(
        cultivated_fraction_values,
        0.0,
        1.0,
    )

    coverage = crop_types.copy(data=has_hrl_coverage.astype(np.float32))
    coverage.attrs = {}
    coverage = coverage.rio.write_crs(crop_types.rio.crs)
    coverage = coverage.rio.write_nodata(None)
    coverage_fraction = coverage.rio.reproject_match(
        template,
        resampling=Resampling.average,
        nodata=0.0,
    )
    coverage_fraction_values = np.nan_to_num(
        coverage_fraction.values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)
    coverage_fraction_values = np.clip(
        coverage_fraction_values,
        0.0,
        1.0,
    )

    # A destination cell with source coverage but no active modal crop retains
    # the native no-cropland state. Only cells with no source coverage remain
    # missing/outside.
    no_active_modal = combined_subgrid_values == _HRL_MISSING_CROP_CODE
    combined_subgrid_values[no_active_modal & (coverage_fraction_values > 0.0)] = (
        _HRL_NO_CROPLAND_CODE
    )

    return (
        combined_subgrid_values,
        cultivated_fraction_values,
        coverage_fraction_values,
    )


class Europe(GEBModel):
    """Build methods for agents in GEB, including Europe-specific logic."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Europe model setup class.

        All positional and keyword arguments are forwarded to the base
        ``GEBModel`` initializer.

        Args:
            *args: Positional arguments forwarded to ``GEBModel``.
            **kwargs: Keyword arguments forwarded to ``GEBModel``.
        """
        super().__init__(*args, **kwargs)

    @build_method(
        depends_on=["setup_regions_and_land_use"],
        required=False,
    )
    def setup_alphaearth_crop_classification(
        self,
        region_id_column: str = "region_id",
        country_iso3_column: str = "ISO3",
        hrl_years: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023),
        prediction_years: tuple[int, ...] = (2024, 2025),
        template_year: int | None = None,
        validation_year: int | None = 2022,
        test_year: int | None = 2023,
        samples_per_cty_class_per_region_year: int = 1000,
        samples_per_cpsct_class_per_region_year: int = 1000,
        sample_stride_pixels: int = 5,
        rare_class_sample_stride_pixels: int | None = 1,
        rare_class_threshold_candidates: int = 50_000,
        rare_class_sample_multiplier: float = 3.0,
        training_label_edge_buffer_pixels: int = 2,
        sample_chunk_rows: int = 512,
        prediction_chunk_size: int = 512,
        include_coordinates: bool = False,
        include_topography: bool = True,
        apply_historical_cropland_mask: bool = True,
        cropland_mask_years: tuple[int, ...] | None = None,
        historical_cropland_mask_dilation_pixels: int = 0,
        smooth_cty_probabilities: bool = True,
        smooth_cpsct_probabilities: bool = True,
        unclassified_probability_threshold: float = 0.25,
        write_cty_confidence: bool = True,
        cty_confidence_output_root: str | None = None,
        apply_permanent_crop_temporal_consistency: bool = True,
        apply_cty_mmu_sieve: bool = True,
        cty_mmu_minimum_pixels: int = 25,
        cty_mmu_connectivity: int = 4,
        cty_mmu_padding_pixels: int = 25,
        cty_mmu_maximum_iterations: int = 3,
        model_type: str = "random_forest",
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        max_features: float | int | str | None = 1.0,
        sample_weight_mode: str = "class_region_year_balanced",
        n_jobs: int = -1,
        random_seed: int = 42,
        hrl_raster_chunks: dict[str, int] | None = None,
        max_alphaearth_files_for_sampling: int | None = None,
        max_alphaearth_files_for_prediction: int | None = None,
        alphaearth_max_parallel_downloads: int | None = None,
        cleanup_alphaearth_downloads: bool = False,
        store_training_samples: bool = True,
        reuse_training_samples: bool = False,
        training_samples_table_name: str = (
            "machine_learning/crop_classification/samples"
        ),
        training_samples_path: str | None = None,
        evaluate_postprocessed_accuracy: bool = False,
        postprocessed_evaluation_output_root: str | None = None,
        cleanup_postprocessed_evaluation_outputs: bool = True,
        max_alphaearth_files_for_postprocessed_evaluation: int | None = None,
        model_output_path: str | None = None,
        overwrite_prediction_files: bool = True,
    ) -> None:
        """Train annual AlphaEarth crop classifiers for the active study area.

        This is a conventional annual remote-sensing classification workflow. For
        every observed year, HRL CTY and CPSCT labels from year ``t`` are paired
        only with AlphaEarth embeddings from year ``t``. No previous crop map,
        previous embedding year, crop rotation, or farmer state is used.

        AlphaEarth downloads are organized at study-area level rather than at
        region level. First, all COGs required for every configured HRL year and
        the full active model bounds are downloaded once. Regional sampling then
        reuses that shared local selection. After all regions and years have been
        sampled, the downloaded training COGs are optionally removed in one cleanup
        pass. The prediction year is handled similarly: it is downloaded once for
        the full active model bounds and reused by every HRL output tile.

        Samples are still generated region by region to bound memory use, but all
        regional samples are pooled into one CTY model and one CPSCT model for the
        complete active study area. Temporal validation and testing are performed
        by holding out complete HRL years. The final models are then retrained on
        all configured HRL years and applied independently to every year in
        ``prediction_years``.

        Validation and blind-test results are logged as Product-by-Reference
        confusion matrices with Overall Accuracy, Producer's Accuracy, User's
        Accuracy, omission and commission errors, F-score, balanced accuracy, and
        Cohen's kappa. CTY is reported at native class level, crop-group level,
        and HRL aggregation level 1.

        Prediction is performed on the exact native HRL 100-km tile grids supplied
        by ``template_year``. The resulting files use the original HRL filenames
        with the target year substituted and are written directly to the existing
        catalog locations::

            $GEB_DATA_ROOT/hrl_crop_types/v1/<prediction_year>/
            $GEB_DATA_ROOT/hrl_secondary_crop/v1/<prediction_year>/

        By default, both 2024 and 2025 are classified from their corresponding
        annual AlphaEarth embeddings. The same final model is applied separately
        to each year; no recursive or previous-year crop type is used as a model
        feature.

        The optional HRL-style post-classification chain uses only data already
        available in the setup: elevation and slope are appended to all 64
        AlphaEarth embedding dimensions; the union of observed HRL cropland years
        acts as a BVL-like crop-extent mask; per-class probabilities are smoothed
        with a 3x3 Gaussian kernel; low-confidence crop predictions become HRL
        classes 3100/3200; stable permanent crops receive conservative temporal
        corrections; and CTY is cleaned with a padded multi-pass 0.25 ha sieve.
        A CTY confidence layer is written alongside the generated crop maps.

        Args:
            region_id_column: Column containing compact model-region IDs.
            country_iso3_column: Column containing country ISO3 codes.
            hrl_years: Independent annual HRL/AlphaEarth observations used for
                sampling, temporal evaluation, and final model fitting.
            prediction_years: Ordered AlphaEarth years to classify. Defaults to
                2024 and 2025. Each year is predicted independently.
            template_year: Existing HRL year used only for tile names, profiles,
                transforms, and dimensions. Defaults to the final ``hrl_years`` year.
            validation_year: Optional complete observation year held out for model
                selection diagnostics.
            test_year: Optional complete observation year held out as a blind test.
            samples_per_cty_class_per_region_year: Maximum CTY samples retained per
                class, model region, and year.
            samples_per_cpsct_class_per_region_year: Maximum CPSCT samples retained
                per class, model region, and year.
            sample_stride_pixels: Standard sampling-lattice spacing in native
                10 m pixels.
            rare_class_sample_stride_pixels: Denser lattice spacing used for classes
                with fewer than ``rare_class_threshold_candidates`` eligible
                candidates. Set to None to disable adaptive rare-class sampling.
            rare_class_threshold_candidates: Candidate-count threshold below which a
                class uses the rare-class lattice and enlarged sample reservoir.
            rare_class_sample_multiplier: Multiplier applied to the per-class sample
                cap for rare classes.
            training_label_edge_buffer_pixels: Number of native pixels removed from
                every CTY/CPSCT class boundary before selecting samples. Prediction
                still covers every valid AlphaEarth pixel.
            sample_chunk_rows: Native HRL rows scanned in each sampling chunk.
            prediction_chunk_size: Width and height of each prediction window.
            include_coordinates: Include longitude/latitude predictors.
            include_topography: Include model-subgrid elevation and terrain gradient
                alongside all 64 AlphaEarth dimensions.
            apply_historical_cropland_mask: Restrict crop predictions to the union
                of observed HRL cropland extents.
            cropland_mask_years: Observed years used for the maximum cropland
                extent; defaults to all ``hrl_years``.
            historical_cropland_mask_dilation_pixels: Optional native-pixel
                dilation of the historical cropland union.
            smooth_cty_probabilities: Apply a nodata-aware 3x3 Gaussian filter to
                CTY class probabilities before reclassification.
            smooth_cpsct_probabilities: Apply the same spatial probability
                smoothing to CPSCT.
            unclassified_probability_threshold: CTY maximum-probability threshold
                below which annual/permanent crops become 3100/3200.
            write_cty_confidence: Write a uint8 0-100 CTY confidence product.
            cty_confidence_output_root: Optional confidence catalog root; when
                omitted, a sibling ``hrl_crop_types_confidence/v1`` root is used.
            apply_permanent_crop_temporal_consistency: Apply conservative
                historical permanent-crop consistency rules to generated years.
            apply_cty_mmu_sieve: Apply padded multi-pass CTY sieving.
            cty_mmu_minimum_pixels: MMU threshold; 25 pixels equals 0.25 ha.
            cty_mmu_connectivity: Four- or eight-neighbour sieve connectivity.
            cty_mmu_padding_pixels: Neighbour-tile padding used during sieving.
            cty_mmu_maximum_iterations: Maximum sieve passes.
            model_type: ``"random_forest"`` or ``"hist_gradient_boosting"``.
            n_estimators: Number of trees or boosting iterations.
            max_depth: Optional estimator tree-depth limit.
            min_samples_leaf: Minimum observations per terminal leaf.
            max_features: Number or fraction of predictors considered at every Random
                Forest split. ``1.0`` evaluates all 64 AlphaEarth embedding features
                simultaneously, plus coordinates when enabled.
            sample_weight_mode: ``"class_region_year_balanced"`` gives equal total
                influence to every represented class while balancing its region-year
                groups. ``"none"`` uses Random Forest balanced-subsample weights.
            n_jobs: Parallel jobs used by Random Forest.
            random_seed: Reproducible sampling and estimator seed.
            hrl_raster_chunks: Native HRL read chunks. Defaults to the Europe module
                chunk configuration.
            max_alphaearth_files_for_sampling: Optional safety limit for the total
                number of AlphaEarth COGs downloaded across all ``hrl_years`` and
                the full active model bounds.
            max_alphaearth_files_for_prediction: Optional safety limit for the total
                number of COGs downloaded across all ``prediction_years`` and the
                full active model bounds.
            alphaearth_max_parallel_downloads: Optional runtime override for the
                adapter's number of simultaneous COG downloads. When omitted, the
                value configured in the data catalog is retained.
            cleanup_alphaearth_downloads: Delete study-area AlphaEarth COGs after
                all sampling passes and, separately, after all prediction tiles.
                False keeps the cache for repeated testing and debugging.
            store_training_samples: Store newly generated pooled annual samples in
                GEB. Reused samples from the default table are not rewritten.
            reuse_training_samples: Load a previously written sample table and skip
                all HRL/AlphaEarth training-sample downloads and extraction.
            training_samples_table_name: Key in ``self.table`` used when reusing
                samples without an explicit path.
            training_samples_path: Optional explicit Parquet path. When supplied, it
                takes precedence over ``training_samples_table_name``.
            evaluate_postprocessed_accuracy: Generate leakage-safe full-tile
                validation/test predictions, apply the complete enabled HRL-style
                post-processing chain, sample the final maps at the held-out sample
                locations, and report final-map accuracy alongside raw rolling-origin
                accuracy.
            postprocessed_evaluation_output_root: Optional directory for temporary
                held-out-year CTY/CPSCT/confidence rasters. The observed HRL folders
                are never overwritten.
            cleanup_postprocessed_evaluation_outputs: Remove temporary held-out-year
                rasters after their accuracy statistics have been sampled.
            max_alphaearth_files_for_postprocessed_evaluation: Optional safety limit
                for AlphaEarth COGs selected across held-out evaluation years.
            model_output_path: Optional joblib path for the final fitted model bundle.
            overwrite_prediction_files: Replace existing generated HRL tiles.

        Raises:
            ValueError: If years, regions, samples, templates, or model settings are
                invalid.
        """
        hrl_years = tuple(int(year) for year in hrl_years)
        if len(hrl_years) < 3:
            raise ValueError("At least three HRL observation years are required.")
        if tuple(sorted(set(hrl_years))) != hrl_years:
            raise ValueError("hrl_years must be unique and strictly increasing.")
        template_year = hrl_years[-1] if template_year is None else int(template_year)
        if template_year not in hrl_years:
            raise ValueError("template_year must be included in hrl_years.")
        prediction_years = tuple(int(year) for year in prediction_years)
        if not prediction_years:
            raise ValueError("prediction_years must contain at least one year.")
        if tuple(sorted(set(prediction_years))) != prediction_years:
            raise ValueError("prediction_years must be unique and strictly increasing.")
        overlapping_prediction_years = set(prediction_years).intersection(hrl_years)
        if overlapping_prediction_years:
            raise ValueError(
                "prediction_years must not overlap the observed HRL years. "
                f"Overlap: {sorted(overlapping_prediction_years)}."
            )
        cropland_mask_years = (
            hrl_years
            if cropland_mask_years is None
            else tuple(int(year) for year in cropland_mask_years)
        )
        if not cropland_mask_years:
            raise ValueError("cropland_mask_years must contain at least one year.")
        if tuple(sorted(set(cropland_mask_years))) != cropland_mask_years:
            raise ValueError(
                "cropland_mask_years must be unique and strictly increasing."
            )
        unknown_cropland_mask_years = set(cropland_mask_years) - set(hrl_years)
        if unknown_cropland_mask_years:
            raise ValueError(
                "cropland_mask_years must be selected from hrl_years. "
                f"Unknown years: {sorted(unknown_cropland_mask_years)}."
            )
        for split_year, split_name in (
            (validation_year, "validation_year"),
            (test_year, "test_year"),
        ):
            if split_year is not None and int(split_year) not in hrl_years:
                raise ValueError(f"{split_name} must be one of {hrl_years}.")
        if validation_year is not None and validation_year == test_year:
            raise ValueError("validation_year and test_year must differ.")
        if sample_chunk_rows < 1:
            raise ValueError("sample_chunk_rows must be at least one.")
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
        if historical_cropland_mask_dilation_pixels < 0:
            raise ValueError(
                "historical_cropland_mask_dilation_pixels cannot be negative."
            )
        if not 0.0 <= unclassified_probability_threshold <= 1.0:
            raise ValueError(
                "unclassified_probability_threshold must lie between zero and one."
            )
        if cty_mmu_minimum_pixels < 1:
            raise ValueError("cty_mmu_minimum_pixels must be at least one.")
        if cty_mmu_connectivity not in {4, 8}:
            raise ValueError("cty_mmu_connectivity must be 4 or 8.")
        if cty_mmu_padding_pixels < 0:
            raise ValueError("cty_mmu_padding_pixels cannot be negative.")
        if cty_mmu_maximum_iterations < 1:
            raise ValueError("cty_mmu_maximum_iterations must be at least one.")
        if sample_weight_mode not in {"none", "class_region_year_balanced"}:
            raise ValueError(
                "sample_weight_mode must be 'none' or 'class_region_year_balanced'."
            )
        if prediction_chunk_size < 16:
            raise ValueError("prediction_chunk_size must be at least 16.")
        if (
            alphaearth_max_parallel_downloads is not None
            and alphaearth_max_parallel_downloads < 1
        ):
            raise ValueError("alphaearth_max_parallel_downloads must be at least 1.")
        if not str(training_samples_table_name).strip():
            raise ValueError("training_samples_table_name cannot be empty.")
        if (
            evaluate_postprocessed_accuracy
            and validation_year is None
            and test_year is None
        ):
            raise ValueError(
                "evaluate_postprocessed_accuracy=True requires validation_year and/or "
                "test_year."
            )

        regions_shapes: gpd.GeoDataFrame = self.geom["regions"]
        for required_column in (region_id_column, country_iso3_column):
            if required_column not in regions_shapes.columns:
                raise ValueError(f"Region database must contain {required_column!r}.")
        if regions_shapes.crs is None:
            raise ValueError("Region geometries must have a CRS.")

        region_ids: xr.DataArray = self.subgrid["region_ids"].compute()
        subgrid_mask: xr.DataArray = self.subgrid["mask"].compute()
        active_subgrid_mask = ~subgrid_mask.values
        region_id_values = region_ids.values.astype(np.int32, copy=False)
        active_geometry = _active_subgrid_mask_geometry_for_hrl(
            region_ids,
            active_subgrid_mask,
        )

        subgrid_elevation: xr.DataArray | None = None
        subgrid_slope: xr.DataArray | None = None
        if include_topography:
            subgrid_elevation = self.subgrid["landsurface/elevation"].compute()
            if subgrid_elevation.rio.crs is None:
                raise ValueError(
                    "Subgrid elevation must have a CRS when topography is enabled."
                )
            slope_values = pyflwdir.dem.slope(
                subgrid_elevation.values,
                nodata=np.nan,
                latlon=True,
                transform=subgrid_elevation.rio.transform(recalc=True),
            )
            subgrid_slope = subgrid_elevation.copy(
                data=np.asarray(slope_values, dtype=np.float32)
            )
            subgrid_slope.name = "slope_gradient"
            self.logger.info(
                "Including elevation and slope with all 64 AlphaEarth embedding "
                "dimensions in training and prediction."
            )

        study_bounds = tuple(float(value) for value in active_geometry.bounds)
        regions_wgs84 = regions_shapes[
            [region_id_column, country_iso3_column, "geometry"]
        ].to_crs("EPSG:4326")
        raster_chunks = (
            _DEFAULT_HRL_RASTER_CHUNKS
            if hrl_raster_chunks is None
            else hrl_raster_chunks
        )
        alphaearth_adapter = self.data_catalog.fetch("alphaearth")
        if alphaearth_max_parallel_downloads is not None:
            alphaearth_adapter.max_parallel_downloads = int(
                alphaearth_max_parallel_downloads
            )
        self.logger.info(
            "Using %s simultaneous AlphaEarth download(s).",
            alphaearth_adapter.max_parallel_downloads,
        )

        template_cty_tile_ids: set[str] = set()
        template_cpsct_tile_ids: set[str] = set()

        def read_hrl_year(
            *,
            year: int,
            region_bounds: tuple[float, float, float, float],
            region_id: int,
        ) -> tuple[xr.DataArray, xr.DataArray]:
            """Read and align one annual CTY/CPSCT observation."""
            crop_types_adapter = self.data_catalog.fetch(
                f"hrl_crop_types_{year}",
                bounds=region_bounds,
                year=year,
            )
            crop_types = crop_types_adapter.read(
                bounds=region_bounds,
                year=year,
                dst_crs=None,
                normalize_nodata=False,
                chunks=raster_chunks,
            )
            secondary_adapter = self.data_catalog.fetch(
                f"hrl_secondary_crop_{year}",
                bounds=region_bounds,
                year=year,
            )
            secondary_crop = secondary_adapter.read(
                bounds=region_bounds,
                year=year,
                dst_crs=None,
                normalize_nodata=False,
                chunks=raster_chunks,
            )

            if year == template_year:
                template_cty_tile_ids.update(
                    str(tile_id)
                    for tile_id in getattr(crop_types_adapter, "tile_ids", ())
                )
                template_cpsct_tile_ids.update(
                    str(tile_id)
                    for tile_id in getattr(secondary_adapter, "tile_ids", ())
                )

            return _align_hrl_rasters_to_common_grid(
                crop_types,
                secondary_crop,
                region_id=region_id,
                year=year,
                logger=self.logger,
            )

        reused_default_table = False
        if reuse_training_samples:
            if training_samples_path is not None:
                sample_source: pd.DataFrame | str | Path = Path(
                    training_samples_path
                ).expanduser()
                source_description = str(sample_source)
            else:
                if training_samples_table_name not in self.table:
                    raise ValueError(
                        "reuse_training_samples=True, but no stored sample table "
                        f"was found under self.table[{training_samples_table_name!r}]. "
                        "Run once with store_training_samples=True or provide "
                        "training_samples_path."
                    )
                sample_source = self.table[training_samples_table_name]
                source_description = f"self.table[{training_samples_table_name!r}]"
                reused_default_table = True

            samples = load_alphaearth_crop_training_samples(
                sample_source,
                hrl_years=hrl_years,
                include_coordinates=include_coordinates,
                include_topography=include_topography,
                active_region_ids=np.unique(region_id_values[active_subgrid_mask]),
            )
            self.logger.info(
                "Reusing %s stored annual AlphaEarth-HRL samples from %s. "
                "Skipping all training COG downloads and regional sampling.",
                len(samples),
                source_description,
            )

            # Sampling normally populates template tile IDs as each region reads
            # template_year. Prediction still needs those IDs, so collect them
            # directly from the template adapters when sampling is skipped.
            cty_template_fetch = self.data_catalog.fetch(
                f"hrl_crop_types_{template_year}",
                bounds=study_bounds,
                year=template_year,
            )
            cpsct_template_fetch = self.data_catalog.fetch(
                f"hrl_secondary_crop_{template_year}",
                bounds=study_bounds,
                year=template_year,
            )
            template_cty_tile_ids.update(
                str(tile_id) for tile_id in getattr(cty_template_fetch, "tile_ids", ())
            )
            template_cpsct_tile_ids.update(
                str(tile_id)
                for tile_id in getattr(cpsct_template_fetch, "tile_ids", ())
            )
        else:
            sample_tables: list[pd.DataFrame] = []
            selected_training_cogs: gpd.GeoDataFrame | None = None
            try:
                self.logger.info(
                    "Downloading all AlphaEarth training COGs for years %s and full "
                    "study-area bounds %s before regional sampling.",
                    hrl_years,
                    study_bounds,
                )
                selected_training_cogs = alphaearth_adapter.read(
                    years=hrl_years,
                    bounds=study_bounds,
                    dry_run=False,
                    max_files=max_alphaearth_files_for_sampling,
                )
                if selected_training_cogs.empty:
                    raise ValueError(
                        "No AlphaEarth training COGs were selected for the active "
                        "study-area bounds."
                    )

                self.logger.info(
                    "Creating independent annual AlphaEarth-HRL samples for %s "
                    "study-area regions and years %s using %s shared downloaded COGs.",
                    len(regions_wgs84),
                    hrl_years,
                    len(selected_training_cogs),
                )

                for region_index, (_, region) in enumerate(regions_wgs84.iterrows()):
                    region_id = int(region[region_id_column])
                    country_iso3 = str(region[country_iso3_column])
                    region_mask_full = active_subgrid_mask & (
                        region_id_values == region_id
                    )
                    if not region_mask_full.any():
                        continue
                    region_geometry = region.geometry.intersection(active_geometry)
                    if region_geometry.is_empty:
                        continue
                    region_bounds = tuple(
                        float(value) for value in region_geometry.bounds
                    )

                    for year in hrl_years:
                        try:
                            crop_types, secondary_crop = read_hrl_year(
                                year=year,
                                region_bounds=region_bounds,
                                region_id=region_id,
                            )
                            region_year_cogs = select_alphaearth_cogs_for_geometry(
                                selected_training_cogs,
                                year=year,
                                clip_geometry=region_geometry,
                            )
                            if region_year_cogs.empty:
                                raise ValueError(
                                    f"No downloaded AlphaEarth COGs intersect region "
                                    f"{region_id}, year {year}."
                                )

                            region_year_samples = (
                                create_alphaearth_crop_training_samples(
                                    crop_types,
                                    secondary_crop,
                                    region_year_cogs,
                                    region_geometry,
                                    year=year,
                                    region_id=region_id,
                                    country_iso3=country_iso3,
                                    samples_per_cty_class=(
                                        samples_per_cty_class_per_region_year
                                    ),
                                    samples_per_cpsct_class=(
                                        samples_per_cpsct_class_per_region_year
                                    ),
                                    sample_stride_pixels=sample_stride_pixels,
                                    rare_class_sample_stride_pixels=(
                                        rare_class_sample_stride_pixels
                                    ),
                                    rare_class_threshold_candidates=(
                                        rare_class_threshold_candidates
                                    ),
                                    rare_class_sample_multiplier=(
                                        rare_class_sample_multiplier
                                    ),
                                    training_label_edge_buffer_pixels=(
                                        training_label_edge_buffer_pixels
                                    ),
                                    sample_chunk_rows=sample_chunk_rows,
                                    include_coordinates=include_coordinates,
                                    include_topography=include_topography,
                                    elevation=subgrid_elevation,
                                    slope=subgrid_slope,
                                    random_seed=(
                                        random_seed + region_index * 10_000 + year
                                    ),
                                )
                            )
                            sample_tables.append(region_year_samples)
                            self.logger.info(
                                "Created %s annual AlphaEarth-HRL samples for "
                                "region %s (%s), year %s from %s cached COG(s).",
                                len(region_year_samples),
                                region_id,
                                country_iso3,
                                year,
                                len(region_year_cogs),
                            )
                        except WEkEONoCoverageError as error:
                            if country_iso3.upper() in _HRL_CROPLANDS_EEA38_ISO3:
                                raise
                            self.logger.warning(
                                "Skipping region %s (%s), year %s: no HRL coverage. %s",
                                region_id,
                                country_iso3,
                                year,
                                error,
                            )
                        finally:
                            gc.collect()
            finally:
                if cleanup_alphaearth_downloads and selected_training_cogs is not None:
                    removed = remove_alphaearth_downloads(
                        selected_training_cogs,
                        logger=self.logger,
                    )
                    self.logger.info(
                        "Removed %s AlphaEarth training COGs after all regional "
                        "sampling passes completed.",
                        removed,
                    )
                elif selected_training_cogs is not None:
                    self.logger.info(
                        "Retaining %s AlphaEarth training COGs in the local cache "
                        "for reuse because cleanup_alphaearth_downloads=False.",
                        len(selected_training_cogs),
                    )

            if not sample_tables:
                raise ValueError("No annual AlphaEarth-HRL samples were created.")
            samples = pd.concat(sample_tables, ignore_index=True)
        if evaluate_postprocessed_accuracy and not (
            {"source_x", "source_y"}.issubset(samples.columns)
            or {"longitude", "latitude"}.issubset(samples.columns)
        ):
            raise ValueError(
                "Post-processed accuracy evaluation requires stored sample "
                "coordinates. Recreate samples with the current workflow or reuse "
                "a table containing source_x/source_y or longitude/latitude."
            )

        samples["split"] = "train"
        if validation_year is not None:
            samples.loc[samples["year"] == validation_year, "split"] = "validation"
        if test_year is not None:
            samples.loc[samples["year"] == test_year, "split"] = "test"
        training_samples = samples.loc[samples["split"] == "train"].copy()
        if training_samples.empty:
            raise ValueError(
                "The configured temporal split leaves no training samples."
            )

        if store_training_samples and not reused_default_table:
            self.set_table(
                samples,
                name=training_samples_table_name,
            )
        elif store_training_samples and reused_default_table:
            self.logger.info(
                "Reused stored sample table %s without rewriting it.",
                training_samples_table_name,
            )

        evaluation_models = fit_alphaearth_crop_models(
            training_samples,
            include_coordinates=include_coordinates,
            include_topography=include_topography,
            model_type=model_type,
            random_seed=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            sample_weight_mode=sample_weight_mode,
            n_jobs=n_jobs,
        )
        metric_tables: list[pd.DataFrame] = []
        confusion_tables: list[pd.DataFrame] = []
        for split_name in ("validation", "test"):
            split_samples = samples.loc[samples["split"] == split_name]
            if split_samples.empty:
                continue
            metrics, confusion = evaluate_alphaearth_crop_models(
                evaluation_models,
                split_samples,
                split_name=split_name,
            )
            metric_tables.append(metrics)
            confusion_tables.append(confusion)

        if metric_tables:
            metrics = pd.concat(metric_tables, ignore_index=True)
            self.set_table(
                metrics,
                name="machine_learning/crop_classification/metrics",
            )
            accuracy_report = format_alphaearth_accuracy_report(
                metrics,
                pd.concat(confusion_tables, ignore_index=True),
            )
            self.logger.info(
                "Annual AlphaEarth crop-classification accuracy assessment:\n%s",
                accuracy_report,
            )
        if confusion_tables:
            self.set_table(
                pd.concat(confusion_tables, ignore_index=True),
                name="machine_learning/crop_classification/confusion_matrix",
            )

        final_models = fit_alphaearth_crop_models(
            samples,
            include_coordinates=include_coordinates,
            include_topography=include_topography,
            model_type=model_type,
            random_seed=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            sample_weight_mode=sample_weight_mode,
            n_jobs=n_jobs,
        )
        feature_importance = alphaearth_crop_feature_importance(final_models)
        if not feature_importance.empty:
            self.set_table(
                feature_importance,
                name="machine_learning/crop_classification/feature_importance",
            )
        if model_output_path is not None:
            saved_model_path = save_alphaearth_crop_models(
                final_models,
                model_output_path,
            )
            self.logger.info(
                "Saved final AlphaEarth crop models to %s.", saved_model_path
            )

        if not template_cty_tile_ids or not template_cpsct_tile_ids:
            raise ValueError(
                f"No {template_year} HRL CTY/CPSCT template tile IDs were collected."
            )

        cty_template_adapter = self.data_catalog.catalog[
            f"hrl_crop_types_{template_year}"
        ]["adapter"]
        cpsct_template_adapter = self.data_catalog.catalog[
            f"hrl_secondary_crop_{template_year}"
        ]["adapter"]

        cty_templates_by_code = {
            hrl_tile_code_from_name(tile_id): find_hrl_tile_path(
                cty_template_adapter.root,
                year=template_year,
                tile_id=tile_id,
            )
            for tile_id in template_cty_tile_ids
        }
        cpsct_templates_by_code = {
            hrl_tile_code_from_name(tile_id): find_hrl_tile_path(
                cpsct_template_adapter.root,
                year=template_year,
                tile_id=tile_id,
            )
            for tile_id in template_cpsct_tile_ids
        }
        missing_cpsct_templates = set(cty_templates_by_code) - set(
            cpsct_templates_by_code
        )
        if missing_cpsct_templates:
            raise ValueError(
                "Missing CPSCT templates for HRL tile codes: "
                f"{sorted(missing_cpsct_templates)}"
            )

        historical_cty_paths_by_code: dict[str, tuple[Path, ...]] = {}
        for tile_code, template_path in cty_templates_by_code.items():
            historical_paths: list[Path] = []
            for historical_year in hrl_years:
                historical_name = build_hrl_prediction_tile_name(
                    template_path.name,
                    product_code="CTY",
                    prediction_year=historical_year,
                )
                historical_paths.append(
                    find_hrl_tile_path(
                        cty_template_adapter.root,
                        year=historical_year,
                        tile_id=Path(historical_name).stem,
                    )
                )
            historical_cty_paths_by_code[tile_code] = tuple(historical_paths)

        if cty_confidence_output_root is None:
            cty_root = Path(cty_template_adapter.root)
            if cty_root.name.lower().startswith("v"):
                confidence_root = (
                    cty_root.parent.parent / "hrl_crop_types_confidence" / cty_root.name
                )
            else:
                confidence_root = cty_root.parent / "hrl_crop_types_confidence"
        else:
            confidence_root = Path(cty_confidence_output_root)

        if evaluate_postprocessed_accuracy:
            evaluation_year_to_split = {
                int(year): split_name
                for year, split_name in (
                    (validation_year, "validation"),
                    (test_year, "test"),
                )
                if year is not None
            }
            evaluation_years = tuple(sorted(evaluation_year_to_split))
            if postprocessed_evaluation_output_root is None:
                evaluation_root = (
                    Path(cty_template_adapter.root).parent.parent
                    / "alphaearth_postprocessed_evaluation"
                    / Path(cty_template_adapter.root).name
                )
            else:
                evaluation_root = Path(postprocessed_evaluation_output_root)
            evaluation_root.mkdir(parents=True, exist_ok=True)

            with rasterio.open(next(iter(cty_templates_by_code.values()))) as source:
                sample_coordinates_crs = source.crs

            postprocessed_metric_tables: list[pd.DataFrame] = []
            postprocessed_confusion_tables: list[pd.DataFrame] = []
            postprocessed_change_rows: list[dict[str, int | str]] = []
            selected_evaluation_cogs: gpd.GeoDataFrame | None = None
            try:
                self.logger.info(
                    "Downloading/reusing AlphaEarth COGs for leakage-safe "
                    "post-processed evaluation years %s.",
                    evaluation_years,
                )
                selected_evaluation_cogs = alphaearth_adapter.read(
                    years=list(evaluation_years),
                    bounds=study_bounds,
                    dry_run=False,
                    max_files=max_alphaearth_files_for_postprocessed_evaluation,
                )
                if selected_evaluation_cogs.empty:
                    raise ValueError(
                        "No AlphaEarth coverage selected for post-processed "
                        f"evaluation years {evaluation_years}."
                    )

                for evaluation_year in evaluation_years:
                    split_name = evaluation_year_to_split[evaluation_year]
                    prior_years = tuple(
                        year for year in hrl_years if year < evaluation_year
                    )
                    if not prior_years:
                        raise ValueError(
                            f"No observation years precede {evaluation_year}."
                        )
                    rolling_training_samples = samples.loc[
                        samples["year"].isin(prior_years)
                    ].copy()
                    held_out_samples = samples.loc[
                        samples["year"] == evaluation_year
                    ].copy()
                    if held_out_samples.empty:
                        raise ValueError(
                            f"No held-out samples are available for {evaluation_year}."
                        )

                    rolling_models = fit_alphaearth_crop_models(
                        rolling_training_samples,
                        include_coordinates=include_coordinates,
                        include_topography=include_topography,
                        model_type=model_type,
                        random_seed=random_seed,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        sample_weight_mode=sample_weight_mode,
                        n_jobs=n_jobs,
                    )
                    raw_features = held_out_samples.loc[
                        :, rolling_models.feature_names
                    ].to_numpy(dtype=np.float32)
                    raw_cty = np.asarray(
                        rolling_models.cty_model.predict(raw_features),
                        dtype=np.int32,
                    )
                    raw_cpsct = np.zeros(len(held_out_samples), dtype=np.int32)
                    observed_cropland = (
                        held_out_samples["cty_label"].to_numpy(dtype=np.int32) > 0
                    )
                    if observed_cropland.any():
                        raw_cpsct[observed_cropland] = np.asarray(
                            rolling_models.cpsct_model.predict(
                                raw_features[observed_cropland]
                            ),
                            dtype=np.int32,
                        )
                    raw_metrics, raw_confusion = evaluate_alphaearth_crop_predictions(
                        held_out_samples,
                        raw_cty,
                        raw_cpsct,
                        split_name=split_name,
                    )
                    raw_metrics.insert(0, "assessment_stage", "raw_rolling_origin")
                    raw_confusion.insert(
                        0,
                        "assessment_stage",
                        "raw_rolling_origin",
                    )
                    raw_metrics["evaluation_year"] = evaluation_year
                    raw_confusion["evaluation_year"] = evaluation_year
                    raw_metrics["training_years"] = ",".join(
                        str(year) for year in prior_years
                    )
                    raw_confusion["training_years"] = ",".join(
                        str(year) for year in prior_years
                    )
                    postprocessed_metric_tables.append(raw_metrics)
                    postprocessed_confusion_tables.append(raw_confusion)

                    split_root = evaluation_root / split_name / str(evaluation_year)
                    cty_directory = split_root / "cty"
                    cpsct_directory = split_root / "cpsct"
                    confidence_directory = split_root / "ctycl"
                    cty_directory.mkdir(parents=True, exist_ok=True)
                    cpsct_directory.mkdir(parents=True, exist_ok=True)
                    if write_cty_confidence:
                        confidence_directory.mkdir(parents=True, exist_ok=True)

                    evaluation_cty_paths: dict[str, Path] = {}
                    evaluation_cpsct_paths: dict[str, Path] = {}
                    evaluation_confidence_paths: dict[str, Path] = {}
                    for tile_code in sorted(cty_templates_by_code):
                        cty_template_path = cty_templates_by_code[tile_code]
                        cpsct_template_path = cpsct_templates_by_code[tile_code]
                        with rasterio.open(cty_template_path) as template_source:
                            tile_bounds_wgs84 = transform_bounds(
                                template_source.crs,
                                "EPSG:4326",
                                *template_source.bounds,
                                densify_pts=21,
                            )
                        prediction_geometry = box(*tile_bounds_wgs84).intersection(
                            active_geometry
                        )
                        if prediction_geometry.is_empty:
                            continue

                        tile_cogs = select_alphaearth_cogs_for_geometry(
                            selected_evaluation_cogs,
                            year=evaluation_year,
                            clip_geometry=prediction_geometry,
                        )
                        if tile_cogs.empty:
                            raise ValueError(
                                "No AlphaEarth COGs intersect evaluation tile "
                                f"{tile_code}, year {evaluation_year}."
                            )

                        cty_path = cty_directory / build_hrl_prediction_tile_name(
                            cty_template_path.name,
                            product_code="CTY",
                            prediction_year=evaluation_year,
                        )
                        cpsct_path = cpsct_directory / build_hrl_prediction_tile_name(
                            cpsct_template_path.name,
                            product_code="CPSCT",
                            prediction_year=evaluation_year,
                        )
                        confidence_path = (
                            confidence_directory
                            / build_hrl_prediction_tile_name(
                                cty_template_path.name,
                                product_code="CTYCL",
                                prediction_year=evaluation_year,
                            )
                            if write_cty_confidence
                            else None
                        )
                        prior_mask_years = tuple(
                            year
                            for year in cropland_mask_years
                            if year < evaluation_year
                        )
                        historical_mask_paths = tuple(
                            historical_cty_paths_by_code[tile_code][
                                hrl_years.index(year)
                            ]
                            for year in prior_mask_years
                        )
                        if apply_historical_cropland_mask and not historical_mask_paths:
                            raise ValueError(
                                "Historical cropland masking requires at least one "
                                f"year before {evaluation_year}."
                            )

                        predict_alphaearth_crop_tile_to_hrl_geotiffs(
                            rolling_models,
                            cty_template_path,
                            cpsct_template_path,
                            tile_cogs,
                            prediction_geometry,
                            cty_path,
                            cpsct_path,
                            chunk_size=prediction_chunk_size,
                            overwrite=True,
                            elevation=subgrid_elevation,
                            slope=subgrid_slope,
                            historical_cty_paths=historical_mask_paths,
                            apply_historical_cropland_mask=(
                                apply_historical_cropland_mask
                            ),
                            historical_cropland_mask_dilation_pixels=(
                                historical_cropland_mask_dilation_pixels
                            ),
                            smooth_cty_probabilities=smooth_cty_probabilities,
                            smooth_cpsct_probabilities=smooth_cpsct_probabilities,
                            unclassified_probability_threshold=(
                                unclassified_probability_threshold
                            ),
                            cty_confidence_output_path=confidence_path,
                        )
                        evaluation_cty_paths[tile_code] = cty_path
                        evaluation_cpsct_paths[tile_code] = cpsct_path
                        if confidence_path is not None:
                            evaluation_confidence_paths[tile_code] = confidence_path

                    if apply_permanent_crop_temporal_consistency:
                        for tile_code, cty_path in evaluation_cty_paths.items():
                            temporal_history = tuple(
                                historical_cty_paths_by_code[tile_code][
                                    hrl_years.index(year)
                                ]
                                for year in prior_years
                            )
                            temporal_stats = apply_alphaearth_permanent_crop_temporal_consistency(
                                temporal_history,
                                {evaluation_year: cty_path},
                                predicted_confidence_paths=(
                                    {
                                        evaluation_year: evaluation_confidence_paths[
                                            tile_code
                                        ]
                                    }
                                    if tile_code in evaluation_confidence_paths
                                    else None
                                ),
                                chunk_size=max(prediction_chunk_size, 512),
                            )
                            postprocessed_change_rows.append(
                                {
                                    "split": split_name,
                                    "evaluation_year": evaluation_year,
                                    "stage": "permanent_crop_temporal_consistency",
                                    "tile": tile_code,
                                    **temporal_stats,
                                }
                            )

                    if apply_cty_mmu_sieve:
                        ordered_tile_codes = sorted(evaluation_cty_paths)
                        sieve_results = apply_alphaearth_cty_mmu_sieve(
                            [
                                evaluation_cty_paths[tile_code]
                                for tile_code in ordered_tile_codes
                            ],
                            confidence_paths=(
                                [
                                    evaluation_confidence_paths[tile_code]
                                    for tile_code in ordered_tile_codes
                                ]
                                if write_cty_confidence
                                else None
                            ),
                            minimum_mapping_unit_pixels=cty_mmu_minimum_pixels,
                            connectivity=cty_mmu_connectivity,
                            padding_pixels=cty_mmu_padding_pixels,
                            maximum_iterations=cty_mmu_maximum_iterations,
                        )
                        if not sieve_results.empty:
                            for row in sieve_results.to_dict(orient="records"):
                                postprocessed_change_rows.append(
                                    {
                                        "split": split_name,
                                        "evaluation_year": evaluation_year,
                                        "stage": "cty_mmu_sieve",
                                        **row,
                                    }
                                )

                    for tile_code, cty_path in evaluation_cty_paths.items():
                        changed = enforce_cpsct_annual_cropland_mask(
                            cty_path,
                            evaluation_cpsct_paths[tile_code],
                        )
                        postprocessed_change_rows.append(
                            {
                                "split": split_name,
                                "evaluation_year": evaluation_year,
                                "stage": "cpsct_annual_cropland_mask",
                                "tile": tile_code,
                                "changed_pixels": changed,
                            }
                        )

                    final_cty, final_cpsct = sample_alphaearth_crop_prediction_tiles(
                        held_out_samples,
                        evaluation_cty_paths,
                        evaluation_cpsct_paths,
                        sample_coordinates_crs=sample_coordinates_crs,
                    )
                    final_metrics, final_confusion = (
                        evaluate_alphaearth_crop_predictions(
                            held_out_samples,
                            final_cty,
                            final_cpsct,
                            split_name=split_name,
                        )
                    )
                    final_metrics.insert(
                        0,
                        "assessment_stage",
                        "final_postprocessed",
                    )
                    final_confusion.insert(
                        0,
                        "assessment_stage",
                        "final_postprocessed",
                    )
                    final_metrics["evaluation_year"] = evaluation_year
                    final_confusion["evaluation_year"] = evaluation_year
                    final_metrics["training_years"] = ",".join(
                        str(year) for year in prior_years
                    )
                    final_confusion["training_years"] = ",".join(
                        str(year) for year in prior_years
                    )
                    postprocessed_metric_tables.append(final_metrics)
                    postprocessed_confusion_tables.append(final_confusion)

                    valid_final_cty = np.isin(final_cty, HRL_CTY_CLASS_CODES)
                    valid_final_cpsct = np.isin(
                        final_cpsct,
                        HRL_CPSCT_CLASS_CODES,
                    )
                    postprocessed_change_rows.extend(
                        [
                            {
                                "split": split_name,
                                "evaluation_year": evaluation_year,
                                "stage": "sample_level_comparison",
                                "target": "CTY",
                                "samples": len(held_out_samples),
                                "valid_final_predictions": int(valid_final_cty.sum()),
                                "changed_predictions": int(
                                    (valid_final_cty & (raw_cty != final_cty)).sum()
                                ),
                            },
                            {
                                "split": split_name,
                                "evaluation_year": evaluation_year,
                                "stage": "sample_level_comparison",
                                "target": "CPSCT",
                                "samples": int(observed_cropland.sum()),
                                "valid_final_predictions": int(
                                    (observed_cropland & valid_final_cpsct).sum()
                                ),
                                "changed_predictions": int(
                                    (
                                        observed_cropland
                                        & valid_final_cpsct
                                        & (raw_cpsct != final_cpsct)
                                    ).sum()
                                ),
                            },
                        ]
                    )
                    self.logger.info(
                        "Completed leakage-safe %s final-map evaluation for %s "
                        "using training years %s.",
                        split_name,
                        evaluation_year,
                        prior_years,
                    )
                    if cleanup_postprocessed_evaluation_outputs:
                        shutil.rmtree(split_root, ignore_errors=True)
                    gc.collect()
            finally:
                if (
                    cleanup_alphaearth_downloads
                    and selected_evaluation_cogs is not None
                ):
                    removed = remove_alphaearth_downloads(
                        selected_evaluation_cogs,
                        logger=self.logger,
                    )
                    self.logger.info(
                        "Removed %s AlphaEarth COGs after post-processed "
                        "validation/test evaluation.",
                        removed,
                    )

            if postprocessed_metric_tables:
                postprocessed_metrics = pd.concat(
                    postprocessed_metric_tables,
                    ignore_index=True,
                )
                postprocessed_confusion = pd.concat(
                    postprocessed_confusion_tables,
                    ignore_index=True,
                )
                self.set_table(
                    postprocessed_metrics,
                    name=(
                        "machine_learning/crop_classification/"
                        "postprocessed_accuracy_metrics"
                    ),
                )
                self.set_table(
                    postprocessed_confusion,
                    name=(
                        "machine_learning/crop_classification/"
                        "postprocessed_accuracy_confusion_matrix"
                    ),
                )
                summary_metrics = postprocessed_metrics.loc[
                    postprocessed_metrics["metric_scope"] == "summary"
                ].copy()
                comparison_keys = ["split", "evaluation_year", "target"]
                raw_summary = summary_metrics.loc[
                    summary_metrics["assessment_stage"] == "raw_rolling_origin"
                ].set_index(comparison_keys)
                final_summary = summary_metrics.loc[
                    summary_metrics["assessment_stage"] == "final_postprocessed"
                ].set_index(comparison_keys)
                common_index = raw_summary.index.intersection(final_summary.index)
                if len(common_index):
                    comparison = pd.DataFrame(index=common_index).reset_index()
                    for metric_name in (
                        "accuracy",
                        "balanced_accuracy",
                        "f_score",
                        "kappa",
                        "reference_support",
                        "excluded_predictions",
                    ):
                        comparison[f"raw_{metric_name}"] = raw_summary.loc[
                            common_index, metric_name
                        ].to_numpy()
                        comparison[f"final_{metric_name}"] = final_summary.loc[
                            common_index, metric_name
                        ].to_numpy()
                        if metric_name in {
                            "accuracy",
                            "balanced_accuracy",
                            "f_score",
                            "kappa",
                        }:
                            comparison[f"delta_{metric_name}"] = (
                                comparison[f"final_{metric_name}"]
                                - comparison[f"raw_{metric_name}"]
                            )
                    self.set_table(
                        comparison,
                        name=(
                            "machine_learning/crop_classification/"
                            "postprocessed_accuracy_comparison"
                        ),
                    )
                report_sections = []
                for stage, title in (
                    ("raw_rolling_origin", "RAW ROLLING-ORIGIN ACCURACY"),
                    ("final_postprocessed", "FINAL POST-PROCESSED MAP ACCURACY"),
                ):
                    stage_metrics = postprocessed_metrics.loc[
                        postprocessed_metrics["assessment_stage"] == stage
                    ]
                    stage_confusion = postprocessed_confusion.loc[
                        postprocessed_confusion["assessment_stage"] == stage
                    ]
                    if not stage_metrics.empty:
                        report_sections.extend(
                            [
                                title,
                                format_alphaearth_accuracy_report(
                                    stage_metrics,
                                    stage_confusion,
                                ),
                            ]
                        )
                self.logger.info(
                    "Leakage-safe AlphaEarth final-map accuracy assessment:\n%s",
                    "\n\n".join(report_sections),
                )
            if postprocessed_change_rows:
                self.set_table(
                    pd.DataFrame(postprocessed_change_rows),
                    name=(
                        "machine_learning/crop_classification/"
                        "postprocessed_accuracy_changes"
                    ),
                )

        generated_cty_tile_ids: dict[int, list[str]] = {
            year: [] for year in prediction_years
        }
        generated_cpsct_tile_ids: dict[int, list[str]] = {
            year: [] for year in prediction_years
        }
        generated_cty_paths: dict[int, dict[str, Path]] = {
            year: {} for year in prediction_years
        }
        generated_cpsct_paths: dict[int, dict[str, Path]] = {
            year: {} for year in prediction_years
        }
        generated_confidence_paths: dict[int, dict[str, Path]] = {
            year: {} for year in prediction_years
        }
        prediction_tasks: list[
            tuple[int, str, Path, Path, Path, Path, Path | None, BaseGeometry]
        ] = []

        for prediction_year in prediction_years:
            cty_output_directory = Path(cty_template_adapter.root) / str(
                prediction_year
            )
            cpsct_output_directory = Path(cpsct_template_adapter.root) / str(
                prediction_year
            )
            confidence_output_directory = confidence_root / str(prediction_year)
            cty_output_directory.mkdir(parents=True, exist_ok=True)
            cpsct_output_directory.mkdir(parents=True, exist_ok=True)
            if write_cty_confidence:
                confidence_output_directory.mkdir(parents=True, exist_ok=True)

            for tile_code in sorted(cty_templates_by_code):
                cty_template_path = cty_templates_by_code[tile_code]
                cpsct_template_path = cpsct_templates_by_code[tile_code]
                cty_output_name = build_hrl_prediction_tile_name(
                    cty_template_path.name,
                    product_code="CTY",
                    prediction_year=prediction_year,
                )
                cpsct_output_name = build_hrl_prediction_tile_name(
                    cpsct_template_path.name,
                    product_code="CPSCT",
                    prediction_year=prediction_year,
                )
                confidence_output_name = build_hrl_prediction_tile_name(
                    cty_template_path.name,
                    product_code="CTYCL",
                    prediction_year=prediction_year,
                )
                cty_output_path = cty_output_directory / cty_output_name
                cpsct_output_path = cpsct_output_directory / cpsct_output_name
                confidence_output_path = (
                    confidence_output_directory / confidence_output_name
                    if write_cty_confidence
                    else None
                )

                required_outputs_exist = (
                    cty_output_path.exists()
                    and cpsct_output_path.exists()
                    and (
                        confidence_output_path is None
                        or confidence_output_path.exists()
                    )
                )
                if not overwrite_prediction_files and required_outputs_exist:
                    self.logger.info(
                        "Using existing generated HRL tiles for %s, year %s.",
                        tile_code,
                        prediction_year,
                    )
                    generated_cty_tile_ids[prediction_year].append(cty_output_path.stem)
                    generated_cpsct_tile_ids[prediction_year].append(
                        cpsct_output_path.stem
                    )
                    generated_cty_paths[prediction_year][tile_code] = cty_output_path
                    generated_cpsct_paths[prediction_year][tile_code] = (
                        cpsct_output_path
                    )
                    if confidence_output_path is not None:
                        generated_confidence_paths[prediction_year][tile_code] = (
                            confidence_output_path
                        )
                    continue

                with rasterio.open(cty_template_path) as template_source:
                    tile_bounds_wgs84 = transform_bounds(
                        template_source.crs,
                        "EPSG:4326",
                        *template_source.bounds,
                        densify_pts=21,
                    )
                tile_geometry = box(*tile_bounds_wgs84)
                prediction_geometry = tile_geometry.intersection(active_geometry)
                if prediction_geometry.is_empty:
                    continue

                prediction_tasks.append(
                    (
                        prediction_year,
                        tile_code,
                        cty_template_path,
                        cpsct_template_path,
                        cty_output_path,
                        cpsct_output_path,
                        confidence_output_path,
                        prediction_geometry,
                    )
                )

        selected_prediction_cogs: gpd.GeoDataFrame | None = None
        if prediction_tasks:
            task_prediction_years = tuple(
                sorted({task[0] for task in prediction_tasks})
            )
            try:
                self.logger.info(
                    "Downloading all AlphaEarth prediction COGs for years %s and "
                    "full study-area bounds %s before classifying %s HRL "
                    "year-tile task(s).",
                    task_prediction_years,
                    study_bounds,
                    len(prediction_tasks),
                )
                selected_prediction_cogs = alphaearth_adapter.read(
                    years=list(task_prediction_years),
                    bounds=study_bounds,
                    dry_run=False,
                    max_files=max_alphaearth_files_for_prediction,
                )
                if selected_prediction_cogs.empty:
                    raise ValueError(
                        "No AlphaEarth coverage selected for prediction years "
                        f"{prediction_years}."
                    )

                cropland_year_positions = [
                    hrl_years.index(year) for year in cropland_mask_years
                ]
                for (
                    prediction_year,
                    tile_code,
                    cty_template_path,
                    cpsct_template_path,
                    cty_output_path,
                    cpsct_output_path,
                    confidence_output_path,
                    prediction_geometry,
                ) in prediction_tasks:
                    tile_alphaearth_cogs = select_alphaearth_cogs_for_geometry(
                        selected_prediction_cogs,
                        year=prediction_year,
                        clip_geometry=prediction_geometry,
                    )
                    if tile_alphaearth_cogs.empty:
                        raise ValueError(
                            f"No downloaded AlphaEarth COGs intersect HRL tile "
                            f"{tile_code}, year {prediction_year}."
                        )

                    historical_mask_paths = tuple(
                        historical_cty_paths_by_code[tile_code][position]
                        for position in cropland_year_positions
                    )
                    predict_alphaearth_crop_tile_to_hrl_geotiffs(
                        final_models,
                        cty_template_path,
                        cpsct_template_path,
                        tile_alphaearth_cogs,
                        prediction_geometry,
                        cty_output_path,
                        cpsct_output_path,
                        chunk_size=prediction_chunk_size,
                        overwrite=overwrite_prediction_files,
                        elevation=subgrid_elevation,
                        slope=subgrid_slope,
                        historical_cty_paths=historical_mask_paths,
                        apply_historical_cropland_mask=(apply_historical_cropland_mask),
                        historical_cropland_mask_dilation_pixels=(
                            historical_cropland_mask_dilation_pixels
                        ),
                        smooth_cty_probabilities=smooth_cty_probabilities,
                        smooth_cpsct_probabilities=smooth_cpsct_probabilities,
                        unclassified_probability_threshold=(
                            unclassified_probability_threshold
                        ),
                        cty_confidence_output_path=confidence_output_path,
                    )
                    generated_cty_tile_ids[prediction_year].append(cty_output_path.stem)
                    generated_cpsct_tile_ids[prediction_year].append(
                        cpsct_output_path.stem
                    )
                    generated_cty_paths[prediction_year][tile_code] = cty_output_path
                    generated_cpsct_paths[prediction_year][tile_code] = (
                        cpsct_output_path
                    )
                    if confidence_output_path is not None:
                        generated_confidence_paths[prediction_year][tile_code] = (
                            confidence_output_path
                        )
                    self.logger.info(
                        "Wrote smoothed, historically masked HRL-compatible "
                        "predictions for tile %s, year %s from %s cached COG(s).",
                        tile_code,
                        prediction_year,
                        len(tile_alphaearth_cogs),
                    )
                    gc.collect()
            finally:
                if (
                    cleanup_alphaearth_downloads
                    and selected_prediction_cogs is not None
                ):
                    removed = remove_alphaearth_downloads(
                        selected_prediction_cogs,
                        logger=self.logger,
                    )
                    self.logger.info(
                        "Removed %s AlphaEarth prediction COGs after all years "
                        "and HRL prediction tiles completed.",
                        removed,
                    )
                elif selected_prediction_cogs is not None:
                    self.logger.info(
                        "Retaining %s AlphaEarth prediction COGs in the local "
                        "cache for reuse because "
                        "cleanup_alphaearth_downloads=False.",
                        len(selected_prediction_cogs),
                    )

        postprocessing_rows: list[dict[str, int | str]] = []
        available_tile_codes = sorted(
            set.intersection(
                *[set(generated_cty_paths[year]) for year in prediction_years]
            )
            if prediction_years
            else set()
        )
        if apply_permanent_crop_temporal_consistency:
            for tile_code in available_tile_codes:
                temporal_stats = apply_alphaearth_permanent_crop_temporal_consistency(
                    historical_cty_paths_by_code[tile_code],
                    {
                        year: generated_cty_paths[year][tile_code]
                        for year in prediction_years
                    },
                    predicted_confidence_paths=(
                        {
                            year: generated_confidence_paths[year][tile_code]
                            for year in prediction_years
                            if tile_code in generated_confidence_paths[year]
                        }
                        if write_cty_confidence
                        else None
                    ),
                    chunk_size=max(prediction_chunk_size, 512),
                )
                postprocessing_rows.append(
                    {
                        "stage": "permanent_crop_temporal_consistency",
                        "year": -1,
                        "tile": tile_code,
                        **temporal_stats,
                    }
                )
                self.logger.info(
                    "Applied permanent-crop temporal consistency to %s: %s.",
                    tile_code,
                    temporal_stats,
                )

        if apply_cty_mmu_sieve:
            for prediction_year in prediction_years:
                year_tile_codes = sorted(generated_cty_paths[prediction_year])
                cty_year_paths = [
                    generated_cty_paths[prediction_year][tile_code]
                    for tile_code in year_tile_codes
                ]
                confidence_year_paths = (
                    [
                        generated_confidence_paths[prediction_year][tile_code]
                        for tile_code in year_tile_codes
                        if tile_code in generated_confidence_paths[prediction_year]
                    ]
                    if write_cty_confidence
                    else None
                )
                sieve_results = apply_alphaearth_cty_mmu_sieve(
                    cty_year_paths,
                    confidence_paths=confidence_year_paths,
                    minimum_mapping_unit_pixels=cty_mmu_minimum_pixels,
                    connectivity=cty_mmu_connectivity,
                    padding_pixels=cty_mmu_padding_pixels,
                    maximum_iterations=cty_mmu_maximum_iterations,
                )
                if not sieve_results.empty:
                    sieve_results.insert(0, "year", prediction_year)
                    sieve_results.insert(0, "stage", "cty_mmu_sieve")
                    postprocessing_rows.extend(sieve_results.to_dict(orient="records"))
                    self.logger.info(
                        "Applied %s-pixel CTY MMU sieve to %s tile(s) for %s; "
                        "%s pixels changed.",
                        cty_mmu_minimum_pixels,
                        len(sieve_results),
                        prediction_year,
                        int(sieve_results["changed_pixels"].sum()),
                    )

        for prediction_year in prediction_years:
            for tile_code, cty_path in generated_cty_paths[prediction_year].items():
                cpsct_path = generated_cpsct_paths[prediction_year][tile_code]
                changed = enforce_cpsct_annual_cropland_mask(
                    cty_path,
                    cpsct_path,
                )
                postprocessing_rows.append(
                    {
                        "stage": "cpsct_annual_cropland_mask",
                        "year": prediction_year,
                        "tile": tile_code,
                        "changed_pixels": changed,
                    }
                )

        if postprocessing_rows:
            self.set_table(
                pd.DataFrame(postprocessing_rows),
                name="machine_learning/crop_classification/postprocessing",
            )

        for prediction_year in prediction_years:
            year_cty_ids = generated_cty_tile_ids[prediction_year]
            year_cpsct_ids = generated_cpsct_tile_ids[prediction_year]
            if not year_cty_ids or not year_cpsct_ids:
                raise ValueError(
                    "No HRL-compatible CTY/CPSCT tiles were generated for "
                    f"{prediction_year}."
                )

            for catalog_name, tile_ids in (
                (f"hrl_crop_types_{prediction_year}", year_cty_ids),
                (f"hrl_secondary_crop_{prediction_year}", year_cpsct_ids),
            ):
                if catalog_name in self.data_catalog.catalog:
                    self.data_catalog.catalog[catalog_name][
                        "adapter"
                    ].tile_ids = tile_ids

            self.logger.info(
                "Finished annual AlphaEarth classification for %s. Generated %s "
                "CTY and %s CPSCT tiles in the standard HRL catalog folders.",
                prediction_year,
                len(year_cty_ids),
                len(year_cpsct_ids),
            )

    @build_method(
        depends_on=["setup_regions_and_land_use"],
        required=False,
    )
    def setup_create_farms_from_HRL_exact_sequences(
        self,
        region_id_column: str = "region_id",
        country_iso3_column: str = "ISO3",
        size_class_boundaries: dict[str, tuple[int | float, int | float]] | None = None,
        years: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023),
        random_seed: int = 42,
        hrl_raster_chunks: dict[str, int] | None = None,
        subgrid_chunk_size: int = 256,
        minimum_cells_per_farm: float = 1.0,
        exact_sequence_jump_candidate_sample: int = 1_024,
        exact_sequence_distance_scale_m: float = 10_000.0,
        rounding_temporal_persistence_weight: float = 0.15,
        crop_area_diagnostics_top_n: int = 0,
        crop_area_fit_warning_threshold_pct: float = 80.0,
    ) -> None:
        """Create farms whose cells share one exact observed crop sequence.

        Native fractional crop areas are first rounded to full model cells by
        year and crop category. The union of those rounded cells forms the static
        agricultural domain. A complete multi-year sequence is then a hard
        grouping constraint: a farmer can contain disconnected parcels, but all
        its cells must carry exactly the same observed sequence. Lowder remains a
        soft prior for the number and size of farms.

        Args:
            region_id_column: Column containing compact model-region IDs.
            country_iso3_column: Column containing country ISO3 codes.
            size_class_boundaries: Optional Lowder class boundaries in square
                metres. Default boundaries are used when omitted.
            years: Ordered HRL years included in each complete sequence.
            random_seed: Base seed for target sizes and raster farm growth.
            hrl_raster_chunks: Native HRL read chunks. Defaults to the module
                chunk configuration.
            subgrid_chunk_size: Destination-tile width and height in model cells.
            minimum_cells_per_farm: Minimum Lowder target size in model cells.
            exact_sequence_jump_candidate_sample: Same-sequence cells sampled
                when a connected parcel is exhausted.
            exact_sequence_distance_scale_m: Distance scale used to prefer nearby
                disconnected parcels carrying the same sequence.
            rounding_temporal_persistence_weight: Preference for retaining the
                same active cells between consecutive years during rounding.
            crop_area_diagnostics_top_n: Number of largest crop-area errors logged
                per region and year; zero disables the detailed listing.
            crop_area_fit_warning_threshold_pct: Combined-crop fit below which a
                completed region is logged as a warning.

        """
        self._setup_create_farms_from_HRL(
            region_id_column=region_id_column,
            country_iso3_column=country_iso3_column,
            size_class_boundaries=size_class_boundaries,
            years=years,
            random_seed=random_seed,
            hrl_raster_chunks=hrl_raster_chunks,
            subgrid_chunk_size=subgrid_chunk_size,
            minimum_cells_per_farm=minimum_cells_per_farm,
            workflow_settings=_ExactSequenceSettings(
                jump_candidate_sample=exact_sequence_jump_candidate_sample,
                distance_scale_m=exact_sequence_distance_scale_m,
                temporal_persistence_weight=rounding_temporal_persistence_weight,
            ),
            crop_area_diagnostics_top_n=crop_area_diagnostics_top_n,
            crop_area_fit_warning_threshold_pct=crop_area_fit_warning_threshold_pct,
        )

    @build_method(
        depends_on=["setup_regions_and_land_use"],
        required=False,
    )
    def setup_create_farms_from_HRL_lowder(
        self,
        region_id_column: str = "region_id",
        country_iso3_column: str = "ISO3",
        size_class_boundaries: dict[str, tuple[int | float, int | float]] | None = None,
        years: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023),
        random_seed: int = 42,
        hrl_raster_chunks: dict[str, int] | None = None,
        subgrid_chunk_size: int = 256,
        distance_weight: float = 0.45,
        crop_sequence_weight: float = 0.40,
        switch_timing_weight: float = 0.15,
        min_valid_crop_sequence_overlap: int = 2,
        minimum_cells_per_farm: float = 1.0,
        jump_candidate_sample: int = 256,
        max_jump_distance_m: float = 2_000.0,
        crop_area_alignment_weight: float = 0.80,
        max_crop_candidates_per_farmer: int = 4,
        max_regional_sequence_candidates_per_farmer: int = 4,
        regional_sequence_pool_size: int = 512,
        crop_area_local_search_passes: int = 2,
        crop_area_regional_search_passes: int = 2,
        local_sequence_fit_threshold_pct: float = 99.0,
        fallow_sequence_penalty: float = 0.35,
        lowder_extra_farm_fraction: float = 0.10,
        crop_area_diagnostics_top_n: int = 0,
        crop_area_fit_warning_threshold_pct: float = 80.0,
    ) -> None:
        """Create Lowder-guided farms and assign complete observed sequences.

        Farm geometry preserves the selected static agricultural area exactly.
        Lowder determines the initial number and relative sizes of farms, after
        which a limited fraction of large targets may be split to improve the
        granularity of the multi-year crop-area optimization.

        Each farmer starts with the complete local sequence occupying the largest
        area in that farm. The optimizer first considers other locally observed
        sequences and introduces similar, common regional sequences only when the
        local-only solution remains below the requested crop-area fit. Annual crop
        labels are never optimized independently, so the final sequence is always
        an original complete HRL sequence.

        Args:
            region_id_column: Column containing compact model-region IDs.
            country_iso3_column: Column containing country ISO3 codes.
            size_class_boundaries: Optional Lowder class boundaries in square
                metres. Default boundaries are used when omitted.
            years: Ordered HRL years included in each complete sequence.
            random_seed: Base seed for target sizes and raster farm growth.
            hrl_raster_chunks: Native HRL read chunks. Defaults to the module
                chunk configuration.
            subgrid_chunk_size: Destination-tile width and height in model cells.
            distance_weight: Farm-growth weight for spatial compactness.
            crop_sequence_weight: Farm-growth weight for full-sequence similarity.
            switch_timing_weight: Farm-growth weight for crop-switch timing.
            min_valid_crop_sequence_overlap: Minimum comparable years required for
                positive sequence similarity.
            minimum_cells_per_farm: Minimum Lowder target size in model cells.
            jump_candidate_sample: Candidate cells sampled when farm growth must
                start another disconnected parcel.
            max_jump_distance_m: Preferred maximum parcel-jump distance.
            crop_area_alignment_weight: Sequence-optimization weight assigned to
                matching regional year-by-crop target areas.
            max_crop_candidates_per_farmer: Maximum locally observed sequences
                retained for each farmer.
            max_regional_sequence_candidates_per_farmer: Maximum regional fallback
                sequences added for each farmer.
            regional_sequence_pool_size: Number of common regional sequences
                considered when constructing fallback candidates.
            crop_area_local_search_passes: Reassignment passes using local
                sequences only.
            crop_area_regional_search_passes: Reassignment passes after regional
                fallback sequences become available.
            local_sequence_fit_threshold_pct: Minimum local-only fit before the
                regional fallback stage is skipped.
            fallow_sequence_penalty: Preference penalty applied to fallow-heavy
                fallback sequences.
            lowder_extra_farm_fraction: Maximum proportional increase in target
                farm count obtained by splitting large Lowder targets.
            crop_area_diagnostics_top_n: Number of largest crop-area errors logged
                per region and year; zero disables the detailed listing.
            crop_area_fit_warning_threshold_pct: Combined-crop fit below which a
                completed region is logged as a warning.

        """
        self._setup_create_farms_from_HRL(
            region_id_column=region_id_column,
            country_iso3_column=country_iso3_column,
            size_class_boundaries=size_class_boundaries,
            years=years,
            random_seed=random_seed,
            hrl_raster_chunks=hrl_raster_chunks,
            subgrid_chunk_size=subgrid_chunk_size,
            minimum_cells_per_farm=minimum_cells_per_farm,
            workflow_settings=_LowderSequenceSettings(
                distance_weight=distance_weight,
                crop_sequence_weight=crop_sequence_weight,
                switch_timing_weight=switch_timing_weight,
                min_valid_crop_sequence_overlap=min_valid_crop_sequence_overlap,
                jump_candidate_sample=jump_candidate_sample,
                max_jump_distance_m=max_jump_distance_m,
                crop_area_alignment_weight=crop_area_alignment_weight,
                max_local_sequences=max_crop_candidates_per_farmer,
                max_regional_sequences=max_regional_sequence_candidates_per_farmer,
                regional_sequence_pool_size=regional_sequence_pool_size,
                local_search_passes=crop_area_local_search_passes,
                regional_search_passes=crop_area_regional_search_passes,
                local_fit_threshold_pct=local_sequence_fit_threshold_pct,
                fallow_penalty=fallow_sequence_penalty,
                extra_farm_fraction=lowder_extra_farm_fraction,
            ),
            crop_area_diagnostics_top_n=crop_area_diagnostics_top_n,
            crop_area_fit_warning_threshold_pct=crop_area_fit_warning_threshold_pct,
        )

    def _setup_create_farms_from_HRL(
        self,
        region_id_column: str,
        country_iso3_column: str,
        size_class_boundaries: dict[str, tuple[int | float, int | float]] | None,
        years: tuple[int, ...],
        random_seed: int,
        hrl_raster_chunks: dict[str, int] | None,
        subgrid_chunk_size: int,
        minimum_cells_per_farm: float,
        workflow_settings: _LowderSequenceSettings | _ExactSequenceSettings,
        crop_area_diagnostics_top_n: int,
        crop_area_fit_warning_threshold_pct: float,
    ) -> None:
        """Run the shared HRL loading and one of the two supported workflows.

        The shared stages load native HRL crop areas, reproject annual crops in
        destination-grid tiles, construct Lowder target sizes, register outputs,
        and produce diagnostics. ``workflow_settings`` selects either the current
        Lowder sequence-balanced method or hard exact-sequence grouping; no legacy
        annual crop-assignment or alternative mask/reprojection modes remain.

        Args:
            region_id_column: Column containing compact model-region IDs.
            country_iso3_column: Column containing country ISO3 codes.
            size_class_boundaries: Optional Lowder class boundaries in square
                metres.
            years: Ordered HRL years represented by the crop sequence.
            random_seed: Base random seed used throughout regional processing.
            hrl_raster_chunks: Native HRL read chunks.
            subgrid_chunk_size: Destination-tile width and height in model cells.
            minimum_cells_per_farm: Minimum target size in model cells.
            workflow_settings: Settings for the Lowder sequence-balanced or exact-
                sequence workflow.
            crop_area_diagnostics_top_n: Number of detailed crop errors to log.
            crop_area_fit_warning_threshold_pct: Regional fit warning threshold.

        Raises:
            RuntimeError: If a hard farm, crop-sequence, area, or land-use
                invariant is violated.
            ValueError: If inputs are invalid or no HRL farmer agents are created.
        """
        if not years:
            raise ValueError("years must contain at least one HRL year.")
        if len(set(years)) != len(years):
            raise ValueError("years must not contain duplicates.")
        if subgrid_chunk_size < 1:
            raise ValueError("subgrid_chunk_size must be at least 1.")
        if minimum_cells_per_farm <= 0.0:
            raise ValueError("minimum_cells_per_farm must be positive.")
        if crop_area_diagnostics_top_n < 0:
            raise ValueError("crop_area_diagnostics_top_n cannot be negative.")
        if not 0.0 <= crop_area_fit_warning_threshold_pct <= 100.0:
            raise ValueError(
                "crop_area_fit_warning_threshold_pct must be between 0 and 100."
            )

        exact_sequence_settings = (
            workflow_settings
            if isinstance(workflow_settings, _ExactSequenceSettings)
            else None
        )
        lowder_sequence_settings = (
            workflow_settings
            if isinstance(workflow_settings, _LowderSequenceSettings)
            else None
        )
        is_exact_sequence = exact_sequence_settings is not None

        if is_exact_sequence:
            assert exact_sequence_settings is not None
            if exact_sequence_settings.jump_candidate_sample < 1:
                raise ValueError(
                    "exact_sequence_jump_candidate_sample must be at least 1."
                )
            if exact_sequence_settings.distance_scale_m <= 0.0:
                raise ValueError("exact_sequence_distance_scale_m must be positive.")
            if not 0.0 <= exact_sequence_settings.temporal_persistence_weight <= 1.0:
                raise ValueError(
                    "rounding_temporal_persistence_weight must be between 0 and 1."
                )
            workflow_name = "exact_sequence"
        else:
            assert lowder_sequence_settings is not None
            score_weight_sum = (
                lowder_sequence_settings.distance_weight
                + lowder_sequence_settings.crop_sequence_weight
                + lowder_sequence_settings.switch_timing_weight
            )
            if score_weight_sum <= 0.0:
                raise ValueError(
                    "Farm-growth score weights must sum to a positive value."
                )
            if lowder_sequence_settings.min_valid_crop_sequence_overlap < 1:
                raise ValueError("min_valid_crop_sequence_overlap must be at least 1.")
            if lowder_sequence_settings.jump_candidate_sample < 1:
                raise ValueError("jump_candidate_sample must be at least 1.")
            if lowder_sequence_settings.max_jump_distance_m < 0.0:
                raise ValueError("max_jump_distance_m cannot be negative.")
            if not 0.0 <= lowder_sequence_settings.crop_area_alignment_weight <= 1.0:
                raise ValueError("crop_area_alignment_weight must be between 0 and 1.")
            if lowder_sequence_settings.max_local_sequences < 1:
                raise ValueError("max_crop_candidates_per_farmer must be at least 1.")
            if lowder_sequence_settings.max_regional_sequences < 1:
                raise ValueError(
                    "max_regional_sequence_candidates_per_farmer must be at least 1."
                )
            if lowder_sequence_settings.regional_sequence_pool_size < 1:
                raise ValueError("regional_sequence_pool_size must be at least 1.")
            if lowder_sequence_settings.local_search_passes < 0:
                raise ValueError("crop_area_local_search_passes cannot be negative.")
            if lowder_sequence_settings.regional_search_passes < 0:
                raise ValueError("crop_area_regional_search_passes cannot be negative.")
            if not 0.0 <= lowder_sequence_settings.local_fit_threshold_pct <= 100.0:
                raise ValueError(
                    "local_sequence_fit_threshold_pct must be between 0 and 100."
                )
            if not 0.0 <= lowder_sequence_settings.fallow_penalty <= 1.0:
                raise ValueError("fallow_sequence_penalty must be between 0 and 1.")
            if not 0.0 <= lowder_sequence_settings.extra_farm_fraction <= 1.0:
                raise ValueError("lowder_extra_farm_fraction must be between 0 and 1.")
            workflow_name = "lowder_sequence_balanced"

        # Bind workflow-specific values once so the regional loop can focus on
        # the shared data flow rather than repeatedly unpacking configuration.
        if is_exact_sequence:
            assert exact_sequence_settings is not None
            exact_sequence_jump_candidate_sample = (
                exact_sequence_settings.jump_candidate_sample
            )
            exact_sequence_distance_scale_m = exact_sequence_settings.distance_scale_m
            exact_sequence_rounding_temporal_persistence_weight = (
                exact_sequence_settings.temporal_persistence_weight
            )
        else:
            assert lowder_sequence_settings is not None
            distance_weight = lowder_sequence_settings.distance_weight
            crop_sequence_weight = lowder_sequence_settings.crop_sequence_weight
            switch_timing_weight = lowder_sequence_settings.switch_timing_weight
            min_valid_crop_sequence_overlap = (
                lowder_sequence_settings.min_valid_crop_sequence_overlap
            )
            jump_candidate_sample = lowder_sequence_settings.jump_candidate_sample
            max_jump_distance_m = lowder_sequence_settings.max_jump_distance_m
            crop_area_alignment_weight = (
                lowder_sequence_settings.crop_area_alignment_weight
            )
            max_crop_candidates_per_farmer = (
                lowder_sequence_settings.max_local_sequences
            )
            max_regional_sequence_candidates_per_farmer = (
                lowder_sequence_settings.max_regional_sequences
            )
            regional_sequence_pool_size = (
                lowder_sequence_settings.regional_sequence_pool_size
            )
            crop_area_local_search_passes = lowder_sequence_settings.local_search_passes
            crop_area_regional_search_passes = (
                lowder_sequence_settings.regional_search_passes
            )
            local_sequence_fit_threshold_pct = (
                lowder_sequence_settings.local_fit_threshold_pct
            )
            fallow_sequence_penalty = lowder_sequence_settings.fallow_penalty
            lowder_extra_farm_fraction = lowder_sequence_settings.extra_farm_fraction

        if size_class_boundaries is None:
            size_class_boundaries = _default_size_class_boundaries()

        regions_shapes: gpd.GeoDataFrame = self.geom["regions"]
        for required_column in (region_id_column, country_iso3_column):
            if required_column not in regions_shapes.columns:
                raise ValueError(f"Region database must contain {required_column!r}.")
        if regions_shapes.crs is None:
            raise ValueError("Region geometries must have a CRS.")

        crop_columns = [f"crop_{year}" for year in years]
        region_ids: xr.DataArray = self.subgrid["region_ids"].compute()
        subgrid_mask: xr.DataArray = self.subgrid["mask"].compute()
        active_subgrid_mask = ~subgrid_mask.values
        region_id_values = region_ids.values.astype(np.int32, copy=False)
        cell_area_m2 = raster_cell_area_m2(region_ids)

        hrl_active_geometry = _active_subgrid_mask_geometry_for_hrl(
            region_ids,
            active_subgrid_mask,
        )
        regions_shapes_hrl = regions_shapes[
            [region_id_column, country_iso3_column, "geometry"]
        ].to_crs("EPSG:4326")
        raster_chunks = (
            _DEFAULT_HRL_RASTER_CHUNKS
            if hrl_raster_chunks is None
            else hrl_raster_chunks
        )

        farm_sizes_per_region = self.data_catalog.fetch(
            "lowder_farm_size_distribution"
        ).read()
        farm_countries_list = list(farm_sizes_per_region["ISO3"].unique())
        farm_size_donor_country = setup_donor_countries(
            self.data_catalog,
            self.geom["global_countries"],
            farm_countries_list,
            alternative_countries=regions_shapes[country_iso3_column].unique().tolist(),
        )

        farms_values = np.full(region_ids.shape, -1, dtype=np.int32)
        all_farmers: list[pd.DataFrame] = []
        region_diagnostics: list[dict[str, float | int | str]] = []
        all_crop_area_diagnostics: list[pd.DataFrame] = []
        total_native_hrl_area_by_year_m2 = {year: 0.0 for year in years}
        total_subgrid_hrl_area_by_year_m2 = {year: 0.0 for year in years}
        total_selected_fractional_area_by_year_m2 = {year: 0.0 for year in years}
        total_selected_modal_area_by_year_m2 = {year: 0.0 for year in years}
        total_selected_fallow_area_by_year_m2 = {year: 0.0 for year in years}
        total_selected_missing_area_by_year_m2 = {year: 0.0 for year in years}
        active_farmer_crops_by_year = {year: 0 for year in years}
        fallow_farmers_by_year = {year: 0 for year in years}
        missing_farmers_by_year = {year: 0 for year in years}
        farmer_id_offset = 0

        static_selection_name = (
            "exact_sequence_union" if is_exact_sequence else "multiyear_coverage"
        )
        self.logger.info(
            "Starting HRL-only raster farm construction for %s model regions "
            "(reprojection=tiled; static selection=%s; workflow=%s).",
            len(regions_shapes_hrl),
            static_selection_name,
            workflow_name,
        )

        for region_index, (_, region) in enumerate(regions_shapes_hrl.iterrows()):
            region_id = int(region[region_id_column])
            original_iso3 = str(region[country_iso3_column])
            region_mask_full = active_subgrid_mask & (region_id_values == region_id)
            if not region_mask_full.any():
                continue

            row_indices, col_indices = np.where(region_mask_full)
            y_slice = slice(int(row_indices.min()), int(row_indices.max()) + 1)
            x_slice = slice(int(col_indices.min()), int(col_indices.max()) + 1)
            region_template = region_ids.isel(y=y_slice, x=x_slice)
            region_mask = region_mask_full[y_slice, x_slice]
            region_cell_area_m2 = cell_area_m2[y_slice, x_slice]

            region_active_geometry = region.geometry.intersection(hrl_active_geometry)
            if region_active_geometry.is_empty:
                continue
            region_bounds = tuple(
                float(value) for value in region_active_geometry.bounds
            )

            self.logger.info(
                "Reading HRL crop stack for region %s (%s), bounds=%s.",
                region_id,
                original_iso3,
                region_bounds,
            )

            combined_crop_per_year: list[np.ndarray] = []
            cultivated_fraction_per_year: list[np.ndarray] = []
            hrl_coverage_fraction_per_year: list[np.ndarray] = []
            native_crop_areas_per_year: list[dict[int, float]] = []
            region_has_hrl_coverage = True
            incomplete_field_cell_count = 0

            for year in years:
                crop_types = None
                secondary_crop = None
                crop_types_adapter = None
                secondary_crop_adapter = None
                try:
                    crop_types_adapter = self.data_catalog.fetch(
                        f"hrl_crop_types_{year}",
                        bounds=region_bounds,
                        year=year,
                    )
                    crop_types = crop_types_adapter.read(
                        bounds=region_bounds,
                        year=year,
                        dst_crs=None,
                        normalize_nodata=False,
                        chunks=raster_chunks,
                    )
                    secondary_crop_adapter = self.data_catalog.fetch(
                        f"hrl_secondary_crop_{year}",
                        bounds=region_bounds,
                        year=year,
                    )
                    secondary_crop = secondary_crop_adapter.read(
                        bounds=region_bounds,
                        year=year,
                        dst_crs=None,
                        normalize_nodata=False,
                        chunks=raster_chunks,
                    )
                except WEkEONoCoverageError as error:
                    if original_iso3.upper() in _HRL_CROPLANDS_EEA38_ISO3:
                        raise
                    self.logger.warning(
                        "Skipping region %s (%s): no HRL Croplands coverage for "
                        "year %s. Original error: %s",
                        region_id,
                        original_iso3,
                        year,
                        error,
                    )
                    region_has_hrl_coverage = False
                    break

                crop_types, secondary_crop = _align_hrl_rasters_to_common_grid(
                    crop_types,
                    secondary_crop,
                    region_id=region_id,
                    year=year,
                    logger=self.logger,
                )
                native_crop_areas_per_year.append(
                    _native_hrl_crop_category_areas_m2(
                        crop_types,
                        secondary_crop,
                        region_active_geometry,
                        chunk_rows=max(int(raster_chunks.get("y", 4096)), 1),
                    )
                )

                combined_year = np.full(
                    region_template.shape,
                    _HRL_MISSING_CROP_CODE,
                    dtype=np.int32,
                )
                cultivated_fraction_year = np.zeros(
                    region_template.shape,
                    dtype=np.float32,
                )
                coverage_fraction_year = np.zeros(
                    region_template.shape,
                    dtype=np.float32,
                )
                source_bounds = crop_types.rio.bounds()
                source_resolution = crop_types.rio.resolution()
                source_buffer_x = abs(float(source_resolution[0])) * 2.0
                source_buffer_y = abs(float(source_resolution[1])) * 2.0

                for tile_y_start in range(
                    0,
                    region_template.sizes["y"],
                    subgrid_chunk_size,
                ):
                    tile_y_stop = min(
                        tile_y_start + subgrid_chunk_size,
                        region_template.sizes["y"],
                    )
                    for tile_x_start in range(
                        0,
                        region_template.sizes["x"],
                        subgrid_chunk_size,
                    ):
                        tile_x_stop = min(
                            tile_x_start + subgrid_chunk_size,
                            region_template.sizes["x"],
                        )
                        tile_y_slice = slice(tile_y_start, tile_y_stop)
                        tile_x_slice = slice(tile_x_start, tile_x_stop)
                        tile_region_mask = region_mask[
                            tile_y_slice,
                            tile_x_slice,
                        ]
                        if not tile_region_mask.any():
                            continue

                        tile_template = region_template.isel(
                            y=tile_y_slice,
                            x=tile_x_slice,
                        )
                        tile_bounds = transform_bounds(
                            tile_template.rio.crs,
                            crop_types.rio.crs,
                            *tile_template.rio.bounds(),
                            densify_pts=21,
                        )
                        clip_min_x = max(
                            tile_bounds[0] - source_buffer_x,
                            source_bounds[0],
                        )
                        clip_min_y = max(
                            tile_bounds[1] - source_buffer_y,
                            source_bounds[1],
                        )
                        clip_max_x = min(
                            tile_bounds[2] + source_buffer_x,
                            source_bounds[2],
                        )
                        clip_max_y = min(
                            tile_bounds[3] + source_buffer_y,
                            source_bounds[3],
                        )
                        if clip_min_x >= clip_max_x or clip_min_y >= clip_max_y:
                            continue

                        crop_types_tile = crop_types.rio.clip_box(
                            minx=clip_min_x,
                            miny=clip_min_y,
                            maxx=clip_max_x,
                            maxy=clip_max_y,
                            allow_one_dimensional_raster=True,
                        )
                        secondary_crop_tile = secondary_crop.rio.clip_box(
                            minx=clip_min_x,
                            miny=clip_min_y,
                            maxx=clip_max_x,
                            maxy=clip_max_y,
                            allow_one_dimensional_raster=True,
                        )
                        crop_types_tile, secondary_crop_tile = (
                            _align_hrl_rasters_to_common_grid(
                                crop_types_tile,
                                secondary_crop_tile,
                                region_id=region_id,
                                year=year,
                                logger=self.logger,
                            )
                        )
                        (
                            tile_combined,
                            tile_fraction,
                            tile_coverage_fraction,
                        ) = _reproject_HRL_year_to_subgrid(
                            crop_types_tile,
                            secondary_crop_tile,
                            tile_template,
                        )
                        tile_combined[~tile_region_mask] = _HRL_MISSING_CROP_CODE
                        tile_fraction[~tile_region_mask] = 0.0
                        tile_coverage_fraction[~tile_region_mask] = 0.0
                        combined_year[
                            tile_y_slice,
                            tile_x_slice,
                        ] = tile_combined
                        cultivated_fraction_year[
                            tile_y_slice,
                            tile_x_slice,
                        ] = tile_fraction
                        coverage_fraction_year[
                            tile_y_slice,
                            tile_x_slice,
                        ] = tile_coverage_fraction

                        del (
                            crop_types_tile,
                            secondary_crop_tile,
                            tile_combined,
                            tile_fraction,
                            tile_coverage_fraction,
                        )

                combined_crop_per_year.append(combined_year)
                cultivated_fraction_per_year.append(cultivated_fraction_year)
                hrl_coverage_fraction_per_year.append(coverage_fraction_year)

                del (
                    crop_types,
                    secondary_crop,
                    crop_types_adapter,
                    secondary_crop_adapter,
                    combined_year,
                    cultivated_fraction_year,
                    coverage_fraction_year,
                )

            if not region_has_hrl_coverage:
                continue
            if (
                len(combined_crop_per_year) != len(years)
                or len(cultivated_fraction_per_year) != len(years)
                or len(hrl_coverage_fraction_per_year) != len(years)
                or len(native_crop_areas_per_year) != len(years)
            ):
                raise ValueError(f"Incomplete HRL crop stack for region {region_id}.")

            crop_stack = np.stack(combined_crop_per_year).astype(
                np.int32,
                copy=False,
            )
            fraction_stack = np.stack(cultivated_fraction_per_year).astype(
                np.float32,
                copy=False,
            )
            coverage_fraction_stack = np.stack(hrl_coverage_fraction_per_year).astype(
                np.float32,
                copy=False,
            )
            del (
                combined_crop_per_year,
                cultivated_fraction_per_year,
                hrl_coverage_fraction_per_year,
            )

            native_hrl_area_by_year_m2 = np.asarray(
                [
                    float(sum(category_areas.values()))
                    for category_areas in native_crop_areas_per_year
                ],
                dtype=np.float64,
            )
            subgrid_hrl_area_by_year_m2 = np.asarray(
                [
                    float(
                        np.sum(
                            fraction_stack[year_index][region_mask]
                            * region_cell_area_m2[region_mask]
                        )
                    )
                    for year_index in range(len(years))
                ],
                dtype=np.float64,
            )

            binary_rounding_diagnostics = pd.DataFrame()
            binary_rounding_wape_pct = np.nan

            if is_exact_sequence:
                # Exact-sequence farms require a complete crop state for every
                # selected cell and year. Category-aware rounding defines that
                # hard domain before any farm targets are grown.
                (
                    crop_stack,
                    eligible_mask,
                    binary_rounding_diagnostics,
                    incomplete_field_cell_count,
                ) = round_crop_states_to_area_targets(
                    modal_crop_stack=crop_stack,
                    cultivated_fraction_stack=fraction_stack,
                    coverage_fraction_stack=coverage_fraction_stack,
                    cell_area_m2=region_cell_area_m2,
                    region_mask=region_mask,
                    target_crop_areas_per_year=native_crop_areas_per_year,
                    fallow_code=_HRL_FALLOW_CROP_CODE,
                    missing_code=_HRL_MISSING_CROP_CODE,
                    temporal_persistence_weight=(
                        exact_sequence_rounding_temporal_persistence_weight
                    ),
                )
                if incomplete_field_cell_count > 0:
                    self.logger.warning(
                        "Region %s excludes %s potential crop cells because at "
                        "least one requested year lacks HRL coverage.",
                        region_id,
                        incomplete_field_cell_count,
                    )
                if not binary_rounding_diagnostics.empty:
                    rounding_target_total_m2 = float(
                        binary_rounding_diagnostics["target_area_m2"].sum()
                    )
                    binary_rounding_wape_pct = (
                        float(binary_rounding_diagnostics["difference_m2"].abs().sum())
                        / rounding_target_total_m2
                        * 100.0
                        if rounding_target_total_m2 > 0.0
                        else np.nan
                    )
                    capacity_shortfall = binary_rounding_diagnostics.loc[
                        binary_rounding_diagnostics["target_area_m2"]
                        > binary_rounding_diagnostics["candidate_area_m2"] + 1e-6
                    ]
                    if not capacity_shortfall.empty:
                        self.logger.warning(
                            "Region %s has %s year-category targets whose native "
                            "area exceeds the full-cell area where that category "
                            "is modal. Their combined unavoidable shortfall is "
                            "%.3f km².",
                            region_id,
                            len(capacity_shortfall),
                            float(
                                (
                                    capacity_shortfall["target_area_m2"]
                                    - capacity_shortfall["candidate_area_m2"]
                                ).sum()
                            )
                            / 1_000_000.0,
                        )

                selection_score = eligible_mask.astype(np.float64)
                base_static_target_area_m2 = float(
                    region_cell_area_m2[eligible_mask].sum()
                )
                selection_target_area_m2 = base_static_target_area_m2
                cultivated_mask = eligible_mask.copy()
            else:
                # The current Lowder workflow uses the most recent requested year
                # as its baseline, but never allows the static farm map to be
                # smaller than the largest native annual cultivated area.
                reference_index = years.index(max(years))
                reference_fraction = fraction_stack[reference_index].astype(
                    np.float64,
                    copy=False,
                )
                base_static_target_area_m2 = float(
                    np.sum(
                        reference_fraction[region_mask]
                        * region_cell_area_m2[region_mask]
                    )
                )
                selection_target_area_m2 = max(
                    base_static_target_area_m2,
                    float(native_hrl_area_by_year_m2.max(initial=0.0)),
                )

                # Only cells with at least one observed crop can support a valid
                # complete regional sequence. Repeated annual crop occurrence is
                # the primary ranking criterion; mean HRL fraction breaks ties.
                union_valid_crop = np.any(crop_stack > 0, axis=0)
                eligible_mask = region_mask & union_valid_crop
                valid_frequency = np.mean(crop_stack > 0, axis=0)
                mean_fraction = fraction_stack.mean(axis=0, dtype=np.float64)
                selection_score = 0.80 * valid_frequency + 0.20 * mean_fraction

                if not eligible_mask.any():
                    self.logger.warning(
                        "Skipping region %s because it has no model cells with an "
                        "observed HRL crop in any requested year.",
                        region_id,
                    )
                    continue

                available_static_capacity_m2 = float(
                    region_cell_area_m2[eligible_mask].sum()
                )
                if selection_target_area_m2 > available_static_capacity_m2:
                    self.logger.warning(
                        "Region %s has only %.3f km² of cells with at least one "
                        "valid modal crop, below the requested static capacity "
                        "%.3f km². Using the available modal-crop capacity.",
                        region_id,
                        available_static_capacity_m2 / 1_000_000.0,
                        selection_target_area_m2 / 1_000_000.0,
                    )
                    selection_target_area_m2 = available_static_capacity_m2

                cultivated_mask = select_cultivated_cells_by_area(
                    selection_score,
                    eligible_mask,
                    region_cell_area_m2,
                    target_area_m2=selection_target_area_m2,
                )

            if not eligible_mask.any():
                self.logger.warning(
                    "Skipping region %s because no valid HRL agricultural cells "
                    "remain after workflow-specific selection.",
                    region_id,
                )
                continue

            # Convert native HRL no-cropland (0) to model fallow (-1) only
            # inside the final multi-year agricultural domain. Native outside/
            # missing values remain -2 and can therefore never be confused with
            # fallow. Values outside the farm domain are also marked missing.
            selected_3d = cultivated_mask[None, :, :]
            crop_stack[selected_3d & (crop_stack == _HRL_NO_CROPLAND_CODE)] = (
                _HRL_FALLOW_CROP_CODE
            )
            crop_stack[:, ~cultivated_mask] = _HRL_MISSING_CROP_CODE

            selected_missing = selected_3d & (crop_stack == _HRL_MISSING_CROP_CODE)
            if is_exact_sequence and selected_missing.any():
                raise RuntimeError(
                    "Exact-sequence farm domain contains HRL outside/missing "
                    f"states in region {region_id}; these cells must be excluded "
                    "rather than interpreted as fallow."
                )

            selected_area_m2 = float(region_cell_area_m2[cultivated_mask].sum())
            selected_fractional_area_by_year_m2 = np.asarray(
                [
                    float(
                        np.sum(
                            fraction_stack[year_index][cultivated_mask]
                            * region_cell_area_m2[cultivated_mask]
                        )
                    )
                    for year_index in range(len(years))
                ],
                dtype=np.float64,
            )
            selected_modal_area_by_year_m2 = np.asarray(
                [
                    float(
                        region_cell_area_m2[
                            cultivated_mask & (crop_stack[year_index] > 0)
                        ].sum()
                    )
                    for year_index in range(len(years))
                ],
                dtype=np.float64,
            )
            selected_fallow_area_by_year_m2 = np.asarray(
                [
                    float(
                        region_cell_area_m2[
                            cultivated_mask
                            & (crop_stack[year_index] == _HRL_FALLOW_CROP_CODE)
                        ].sum()
                    )
                    for year_index in range(len(years))
                ],
                dtype=np.float64,
            )
            selected_missing_area_by_year_m2 = np.asarray(
                [
                    float(
                        region_cell_area_m2[
                            cultivated_mask
                            & (crop_stack[year_index] == _HRL_MISSING_CROP_CODE)
                        ].sum()
                    )
                    for year_index in range(len(years))
                ],
                dtype=np.float64,
            )
            represented_area_by_year_m2 = (
                selected_modal_area_by_year_m2
                + selected_fallow_area_by_year_m2
                + selected_missing_area_by_year_m2
            )
            if not np.allclose(
                represented_area_by_year_m2,
                selected_area_m2,
                rtol=0.0,
                atol=1e-4,
            ):
                raise RuntimeError(
                    "Active crop, fallow, and missing areas do not exhaust the "
                    f"static agricultural domain in region {region_id}."
                )

            if is_exact_sequence:
                active_sequences = crop_stack[:, cultivated_mask]
                if np.any(
                    np.all(
                        active_sequences == _HRL_FALLOW_CROP_CODE,
                        axis=0,
                    )
                ):
                    raise RuntimeError(
                        "Exact-sequence field union contains a cell that is "
                        "fallow in every requested year."
                    )
                if np.any(active_sequences == _HRL_MISSING_CROP_CODE):
                    raise RuntimeError(
                        "Exact-sequence field union contains an outside/missing "
                        "HRL state."
                    )

            iso3 = farm_size_donor_country.get(original_iso3, original_iso3)
            if iso3 != original_iso3:
                self.logger.info(
                    "Missing farm sizes for %s; using donor country %s.",
                    original_iso3,
                    iso3,
                )
            region_farm_sizes = farm_sizes_per_region.loc[
                farm_sizes_per_region["ISO3"] == iso3
            ].drop(["Country", "Census Year", "Total"], axis=1)
            if len(region_farm_sizes) != 2:
                raise ValueError(
                    f"No complete Lowder farm-size data are available for region "
                    f"{region_id} ({original_iso3}; source {iso3})."
                )

            target_farms = create_lowder_target_farm_areas(
                region_farm_sizes=region_farm_sizes,
                size_class_boundaries=size_class_boundaries,
                cultivated_area_m2=selected_area_m2,
                iso3=iso3,
                logger=self.logger,
                random_seed=random_seed + region_index,
                minimum_cells_per_farm=minimum_cells_per_farm,
                mean_cell_area_m2=float(region_cell_area_m2[cultivated_mask].mean()),
            )
            lowder_target_farm_count = len(target_farms)
            if not is_exact_sequence and lowder_extra_farm_fraction > 0.0:
                target_farms = relax_lowder_targets_for_sequence_fit(
                    target_farms,
                    extra_farm_fraction=lowder_extra_farm_fraction,
                    n_available_cells=int(np.count_nonzero(cultivated_mask)),
                    mean_cell_area_m2=float(
                        region_cell_area_m2[cultivated_mask].mean()
                    ),
                    minimum_cells_per_farm=minimum_cells_per_farm,
                )
            sequence_fit_target_farm_count = len(target_farms)

            if is_exact_sequence:
                local_farms, farmers_region = grow_farms_from_exact_crop_sequences(
                    cultivated_mask=cultivated_mask,
                    crop_sequences=crop_stack,
                    cell_area_m2=region_cell_area_m2,
                    target_farms=target_farms,
                    crop_columns=crop_columns,
                    random_seed=random_seed + 10_000 + region_index,
                    jump_candidate_sample=exact_sequence_jump_candidate_sample,
                    jump_distance_scale_m=exact_sequence_distance_scale_m,
                )
            else:
                local_farms, farmers_region = grow_farms_from_raster_cells(
                    cultivated_mask=cultivated_mask,
                    crop_sequences=crop_stack,
                    cell_area_m2=region_cell_area_m2,
                    target_farms=target_farms,
                    random_seed=random_seed + 10_000 + region_index,
                    distance_weight=distance_weight,
                    crop_sequence_weight=crop_sequence_weight,
                    switch_timing_weight=switch_timing_weight,
                    min_valid_crop_sequence_overlap=min_valid_crop_sequence_overlap,
                    jump_candidate_sample=jump_candidate_sample,
                    max_jump_distance_m=max_jump_distance_m,
                )

            farmer_areas_local_m2 = farmers_region["area_m2"].to_numpy(dtype=np.float64)
            sequence_alignment = pd.DataFrame()

            if is_exact_sequence:
                # Every exact-sequence farmer is necessarily assigned its locally
                # observed complete sequence. Use the same quality schema as the
                # Lowder-prioritized workflow for downstream sampling.
                farmers_region["crop_sequence_quality_flag"] = np.full(
                    len(farmers_region),
                    2,
                    dtype=np.int8,
                )
                farmers_region["crop_sequence_is_original"] = True
                farmers_region["crop_sequence_is_local"] = True
                farmers_region["crop_sequence_is_local_dominant"] = True
                farmers_region["crop_sequence_local_support_fraction"] = np.ones(
                    len(farmers_region),
                    dtype=np.float32,
                )
                farmers_region["crop_sequence_fallow_fraction"] = np.mean(
                    farmers_region[crop_columns].to_numpy(dtype=np.int32)
                    == _HRL_FALLOW_CROP_CODE,
                    axis=1,
                    dtype=np.float64,
                ).astype(np.float32)
            else:
                (
                    assigned_sequences,
                    sequence_quality,
                    sequence_alignment,
                ) = assign_farmer_sequences_to_area_targets(
                    farm_values=local_farms,
                    crop_sequences=crop_stack,
                    cell_area_m2=region_cell_area_m2,
                    farmer_areas_m2=farmer_areas_local_m2,
                    target_crop_areas_per_year=native_crop_areas_per_year,
                    fallow_code=_HRL_FALLOW_CROP_CODE,
                    missing_code=_HRL_MISSING_CROP_CODE,
                    alignment_weight=crop_area_alignment_weight,
                    max_local_sequences=max_crop_candidates_per_farmer,
                    max_regional_sequences=(
                        max_regional_sequence_candidates_per_farmer
                    ),
                    regional_sequence_pool_size=regional_sequence_pool_size,
                    local_search_passes=crop_area_local_search_passes,
                    regional_search_passes=crop_area_regional_search_passes,
                    local_fit_threshold_pct=local_sequence_fit_threshold_pct,
                    fallow_penalty=fallow_sequence_penalty,
                )
                farmers_region.loc[:, crop_columns] = assigned_sequences
                for quality_column in sequence_quality.columns:
                    farmers_region[quality_column] = sequence_quality[
                        quality_column
                    ].to_numpy()
            crop_alignment_summary_by_year: list[dict[str, float | int]] = []
            for year_index, (year, crop_column) in enumerate(
                zip(years, crop_columns, strict=True)
            ):
                selected_grid_fit_score = np.nan
                assigned_fallow_area_m2 = np.nan
                assigned_missing_area_m2 = np.nan
                if is_exact_sequence:
                    assigned_crop_codes = farmers_region[crop_column].to_numpy(
                        dtype=np.int32
                    )
                    assigned_fallow_area_m2 = float(
                        farmer_areas_local_m2[
                            assigned_crop_codes == _HRL_FALLOW_CROP_CODE
                        ].sum()
                    )
                    assigned_missing_area_m2 = float(
                        farmer_areas_local_m2[
                            assigned_crop_codes == _HRL_MISSING_CROP_CODE
                        ].sum()
                    )
                    if assigned_missing_area_m2 > 1e-3:
                        raise RuntimeError(
                            "Exact-sequence grouping assigned outside/missing "
                            f"HRL states in region {region_id}, year {year}."
                        )
                    if not np.isclose(
                        assigned_fallow_area_m2,
                        selected_fallow_area_by_year_m2[year_index],
                        rtol=1e-12,
                        atol=1e-3,
                    ):
                        raise RuntimeError(
                            "Exact-sequence grouping did not conserve fallow "
                            f"area for region {region_id}, year {year}."
                        )
                    crop_alignment = _crop_area_diagnostics_from_assignments(
                        assigned_crop_codes,
                        farmer_areas_local_m2,
                        native_crop_areas_per_year[year_index],
                    )
                    selected_grid_targets = _crop_area_targets_from_model_grid(
                        crop_stack[year_index],
                        cultivated_mask,
                        region_cell_area_m2,
                    )
                    selected_grid_alignment = _crop_area_diagnostics_from_assignments(
                        assigned_crop_codes,
                        farmer_areas_local_m2,
                        selected_grid_targets,
                    )
                    selected_grid_fit_score = _crop_area_fit_scores(
                        selected_grid_alignment
                    )["crop_area_fit_score"]
                    if selected_grid_fit_score < 100.0 - 1e-9:
                        raise RuntimeError(
                            "Exact-sequence grouping did not conserve selected-grid "
                            f"crop area for region {region_id}, year {year}."
                        )
                else:
                    assigned_crop_codes = farmers_region[crop_column].to_numpy(
                        dtype=np.int32
                    )
                    assigned_fallow_area_m2 = float(
                        farmer_areas_local_m2[
                            assigned_crop_codes == _HRL_FALLOW_CROP_CODE
                        ].sum()
                    )
                    assigned_missing_area_m2 = float(
                        farmer_areas_local_m2[
                            assigned_crop_codes == _HRL_MISSING_CROP_CODE
                        ].sum()
                    )
                    if assigned_missing_area_m2 > 1e-3:
                        raise RuntimeError(
                            "Sequence-balanced assignment produced a missing "
                            f"farmer crop in region {region_id}, year {year}."
                        )
                    crop_alignment = (
                        sequence_alignment.loc[
                            sequence_alignment["year_index"] == year_index
                        ]
                        .drop(columns="year_index")
                        .reset_index(drop=True)
                    )
                crop_alignment[region_id_column] = region_id
                crop_alignment[country_iso3_column] = original_iso3
                crop_alignment["year"] = int(year)
                all_crop_area_diagnostics.append(crop_alignment.copy())

                combined_fit = _crop_area_fit_scores(crop_alignment)
                main_crop_alignment = _aggregate_crop_alignment_to_main_categories(
                    crop_alignment
                )
                main_fit = _crop_area_fit_scores(main_crop_alignment)
                positive_target_scale = float(
                    crop_alignment["positive_target_scale"].iloc[0]
                )

                crop_alignment_summary_by_year.append(
                    {
                        "year": int(year),
                        "source_crop_area_m2": combined_fit["source_area_m2"],
                        "fractional_subgrid_area_m2": float(
                            subgrid_hrl_area_by_year_m2[year_index]
                        ),
                        "selected_fractional_area_m2": float(
                            selected_fractional_area_by_year_m2[year_index]
                        ),
                        "selected_modal_area_m2": float(
                            selected_modal_area_by_year_m2[year_index]
                        ),
                        "assigned_crop_area_m2": combined_fit["assigned_area_m2"],
                        "assigned_fallow_area_m2": assigned_fallow_area_m2,
                        "assigned_missing_area_m2": assigned_missing_area_m2,
                        "agricultural_union_area_m2": selected_area_m2,
                        "fallow_share_pct": (
                            assigned_fallow_area_m2 / selected_area_m2 * 100.0
                            if selected_area_m2 > 0.0
                            and np.isfinite(assigned_fallow_area_m2)
                            else np.nan
                        ),
                        "missing_share_pct": (
                            assigned_missing_area_m2 / selected_area_m2 * 100.0
                            if selected_area_m2 > 0.0
                            and np.isfinite(assigned_missing_area_m2)
                            else np.nan
                        ),
                        "total_area_difference_pct": combined_fit[
                            "total_area_difference_pct"
                        ],
                        "total_area_fit_score": combined_fit["total_area_fit_score"],
                        "combined_crop_area_fit_score": combined_fit[
                            "crop_area_fit_score"
                        ],
                        "combined_crop_share_fit_score": combined_fit[
                            "crop_share_fit_score"
                        ],
                        "main_crop_area_fit_score": main_fit["crop_area_fit_score"],
                        "main_crop_share_fit_score": main_fit["crop_share_fit_score"],
                        "positive_target_scale": positive_target_scale,
                        "selected_grid_crop_fit_score": selected_grid_fit_score,
                    }
                )

                # Per-year regional details are intentionally kept at DEBUG
                # level. INFO output is emitted as compact comparison tables after
                # all regions have been processed.
                if crop_area_diagnostics_top_n > 0:
                    self.logger.debug(
                        "Region %s detailed HRL crop-area categories for %s "
                        "(top %s):\n%s",
                        region_id,
                        year,
                        crop_area_diagnostics_top_n,
                        _format_hrl_crop_area_alignment(
                            crop_alignment,
                            top_n=crop_area_diagnostics_top_n,
                        ),
                    )

            local_valid = local_farms >= 0
            local_farms[local_valid] += farmer_id_offset
            farms_region_window = farms_values[y_slice, x_slice]
            farms_region_window[local_valid] = local_farms[local_valid]

            farmers_region["farmer_id"] = (
                farmers_region["farmer_id"].to_numpy(dtype=np.int32) + farmer_id_offset
            )
            farmers_region[region_id_column] = np.full(
                len(farmers_region),
                region_id,
                dtype=np.int32,
            )
            all_farmers.append(farmers_region)

            for year, native_area_m2, subgrid_area_m2 in zip(
                years,
                native_hrl_area_by_year_m2,
                subgrid_hrl_area_by_year_m2,
                strict=True,
            ):
                total_native_hrl_area_by_year_m2[year] += float(native_area_m2)
                total_subgrid_hrl_area_by_year_m2[year] += float(subgrid_area_m2)
            for year, selected_fractional_m2, selected_modal_m2 in zip(
                years,
                selected_fractional_area_by_year_m2,
                selected_modal_area_by_year_m2,
                strict=True,
            ):
                total_selected_fractional_area_by_year_m2[year] += float(
                    selected_fractional_m2
                )
                total_selected_modal_area_by_year_m2[year] += float(selected_modal_m2)
            for year, selected_fallow_m2 in zip(
                years,
                selected_fallow_area_by_year_m2,
                strict=True,
            ):
                total_selected_fallow_area_by_year_m2[year] += float(selected_fallow_m2)
            for year, selected_missing_m2 in zip(
                years,
                selected_missing_area_by_year_m2,
                strict=True,
            ):
                total_selected_missing_area_by_year_m2[year] += float(
                    selected_missing_m2
                )

            target_areas_m2 = farmers_region["target_area_m2"].to_numpy(
                dtype=np.float64
            )
            actual_areas_m2 = farmers_region["area_m2"].to_numpy(dtype=np.float64)
            area_errors_m2 = actual_areas_m2 - target_areas_m2
            relative_area_errors = np.divide(
                np.abs(area_errors_m2),
                target_areas_m2,
                out=np.zeros_like(area_errors_m2),
                where=target_areas_m2 > 0,
            )

            n_parcels = farmers_region["n_fields"].to_numpy(dtype=np.int32)
            n_multi_parcel_farms = int(np.count_nonzero(n_parcels > 1))
            multi_parcel_share = n_multi_parcel_farms / len(farmers_region)

            active_crop_counts: list[int] = []
            fallow_counts: list[int] = []
            missing_counts: list[int] = []
            unique_crop_counts: list[int] = []
            for year, crop_column in zip(years, crop_columns, strict=True):
                crop_values = farmers_region[crop_column].to_numpy(dtype=np.int32)
                active_crops = crop_values > 0
                fallow = crop_values == _HRL_FALLOW_CROP_CODE
                missing = crop_values == _HRL_MISSING_CROP_CODE
                active_count = int(np.count_nonzero(active_crops))
                fallow_count = int(np.count_nonzero(fallow))
                missing_count = int(np.count_nonzero(missing))
                active_crop_counts.append(active_count)
                fallow_counts.append(fallow_count)
                missing_counts.append(missing_count)
                unique_crop_counts.append(
                    int(np.unique(crop_values[active_crops]).size)
                    if active_count > 0
                    else 0
                )
                active_farmer_crops_by_year[year] += active_count
                fallow_farmers_by_year[year] += fallow_count
                missing_farmers_by_year[year] += missing_count

            area_difference_m2 = selected_area_m2 - selection_target_area_m2
            area_difference_pct = (
                area_difference_m2 / selection_target_area_m2 * 100.0
                if selection_target_area_m2 > 0
                else np.nan
            )
            eligible_cell_count = int(np.count_nonzero(eligible_mask))
            selected_cell_count = int(np.count_nonzero(cultivated_mask))
            mean_active_crop_coverage_pct = (
                float(np.mean(active_crop_counts)) / len(farmers_region) * 100.0
            )
            mean_fallow_share_pct = (
                float(np.mean(fallow_counts)) / len(farmers_region) * 100.0
            )
            mean_missing_share_pct = (
                float(np.mean(missing_counts)) / len(farmers_region) * 100.0
            )

            selected_sequences = crop_stack[:, cultivated_mask].T
            complete_original_mask = ~np.any(
                selected_sequences == _HRL_MISSING_CROP_CODE,
                axis=1,
            ) & np.any(selected_sequences > 0, axis=1)
            complete_original_sequences = np.unique(
                selected_sequences[complete_original_mask],
                axis=0,
            )
            n_exact_sequences = int(complete_original_sequences.shape[0])
            extra_sequence_farms = max(
                len(farmers_region) - lowder_target_farm_count,
                0,
            )

            quality_flags = farmers_region["crop_sequence_quality_flag"].to_numpy(
                dtype=np.int8
            )
            local_dominant_sequence_pct = float(np.mean(quality_flags == 2) * 100.0)
            local_sequence_pct = float(np.mean(quality_flags >= 1) * 100.0)
            regional_fallback_sequence_pct = float(np.mean(quality_flags == 0) * 100.0)

            if is_exact_sequence:
                sequence_homogeneity_pct = 100.0
                novel_sequence_count = 0
            else:
                # The assignment function returns only integer IDs into the
                # complete regional original-sequence catalog and validates those
                # IDs before returning. No additional large Python set is needed.
                novel_sequence_count = 0
                sequence_homogeneity_pct = np.nan
            region_diagnostics.append(
                {
                    region_id_column: region_id,
                    country_iso3_column: original_iso3,
                    "lowder_source_iso3": iso3,
                    "base_static_target_km2": (
                        base_static_target_area_m2 / 1_000_000.0
                    ),
                    "required_capacity_km2": (
                        native_hrl_area_by_year_m2.max(initial=0.0) / 1_000_000.0
                    ),
                    "selection_target_km2": (selection_target_area_m2 / 1_000_000.0),
                    "model_area_km2": selected_area_m2 / 1_000_000.0,
                    "area_difference_pct": area_difference_pct,
                    "eligible_cells": eligible_cell_count,
                    "cultivated_cells": selected_cell_count,
                    "n_farmers": int(len(farmers_region)),
                    "lowder_target_farms": int(lowder_target_farm_count),
                    "sequence_fit_target_farms": int(sequence_fit_target_farm_count),
                    "n_exact_sequences": n_exact_sequences,
                    "extra_sequence_farms": int(extra_sequence_farms),
                    "sequence_homogeneity_pct": sequence_homogeneity_pct,
                    "novel_sequence_count": novel_sequence_count,
                    "local_dominant_sequence_pct": (local_dominant_sequence_pct),
                    "local_sequence_pct": local_sequence_pct,
                    "regional_fallback_sequence_pct": (regional_fallback_sequence_pct),
                    "excluded_incomplete_hrl_cells": (incomplete_field_cell_count),
                    "binary_rounding_wape_pct": binary_rounding_wape_pct,
                    "median_farm_area_ha": float(np.median(actual_areas_m2) / 10_000.0),
                    "mean_farm_area_ha": float(actual_areas_m2.mean() / 10_000.0),
                    "mean_target_error_pct": float(relative_area_errors.mean() * 100.0),
                    "total_parcels": int(n_parcels.sum()),
                    "multi_parcel_farms_pct": multi_parcel_share * 100.0,
                    "mean_active_crop_coverage_pct": (mean_active_crop_coverage_pct),
                    "mean_fallow_farmers_pct": mean_fallow_share_pct,
                    "mean_missing_farmers_pct": mean_missing_share_pct,
                    "mean_fallow_area_pct": float(
                        np.mean(
                            [
                                summary["fallow_share_pct"]
                                for summary in crop_alignment_summary_by_year
                            ]
                        )
                    ),
                    "mean_missing_area_pct": float(
                        np.nanmean(
                            [
                                summary["missing_share_pct"]
                                for summary in crop_alignment_summary_by_year
                            ]
                        )
                    ),
                    "mean_selected_fraction_retention_pct": float(
                        np.mean(
                            np.divide(
                                selected_fractional_area_by_year_m2,
                                subgrid_hrl_area_by_year_m2,
                                out=np.full(len(years), np.nan),
                                where=subgrid_hrl_area_by_year_m2 > 0.0,
                            )
                        )
                        * 100.0
                    ),
                    "mean_agent_area_retention_pct": float(
                        np.mean(
                            [
                                summary["assigned_crop_area_m2"]
                                / summary["source_crop_area_m2"]
                                if summary["source_crop_area_m2"] > 0.0
                                else np.nan
                                for summary in crop_alignment_summary_by_year
                            ]
                        )
                        * 100.0
                    ),
                    "maximum_absolute_agent_area_difference_pct": float(
                        np.max(
                            [
                                abs(summary["total_area_difference_pct"])
                                for summary in crop_alignment_summary_by_year
                            ]
                        )
                    ),
                    "mean_combined_crop_fit_score": float(
                        np.mean(
                            [
                                summary["combined_crop_area_fit_score"]
                                for summary in crop_alignment_summary_by_year
                            ]
                        )
                    ),
                    "minimum_combined_crop_fit_score": float(
                        np.min(
                            [
                                summary["combined_crop_area_fit_score"]
                                for summary in crop_alignment_summary_by_year
                            ]
                        )
                    ),
                    "mean_main_crop_fit_score": float(
                        np.mean(
                            [
                                summary["main_crop_area_fit_score"]
                                for summary in crop_alignment_summary_by_year
                            ]
                        )
                    ),
                    "mean_crop_share_fit_score": float(
                        np.mean(
                            [
                                summary["combined_crop_share_fit_score"]
                                for summary in crop_alignment_summary_by_year
                            ]
                        )
                    ),
                }
            )

            mean_combined_fit = float(
                np.mean(
                    [
                        summary["combined_crop_area_fit_score"]
                        for summary in crop_alignment_summary_by_year
                    ]
                )
            )
            mean_agent_retention = float(
                np.mean(
                    [
                        summary["assigned_crop_area_m2"]
                        / summary["source_crop_area_m2"]
                        if summary["source_crop_area_m2"] > 0.0
                        else np.nan
                        for summary in crop_alignment_summary_by_year
                    ]
                )
                * 100.0
            )
            log_method = (
                self.logger.warning
                if mean_combined_fit < crop_area_fit_warning_threshold_pct
                else self.logger.info
            )
            if is_exact_sequence:
                log_method(
                    "Completed exact-sequence region %s (%s): %.2f km² static "
                    "area; %s agents from %s rounded sequences (%s Lowder "
                    "targets); mean annual active/native area %.1f%%; mean "
                    "combined crop fit %.1f/100.",
                    region_id,
                    original_iso3,
                    selected_area_m2 / 1_000_000.0,
                    len(farmers_region),
                    n_exact_sequences,
                    lowder_target_farm_count,
                    mean_agent_retention,
                    mean_combined_fit,
                )
            else:
                log_method(
                    "Completed Lowder-prioritized region %s (%s): %.2f km² "
                    "static area; %s farms from %s Lowder targets; mean annual "
                    "active/native area %.1f%%; mean combined crop fit %.1f/100.",
                    region_id,
                    original_iso3,
                    selected_area_m2 / 1_000_000.0,
                    len(farmers_region),
                    lowder_target_farm_count,
                    mean_agent_retention,
                    mean_combined_fit,
                )

            farmer_id_offset += len(farmers_region)
            del (
                crop_stack,
                fraction_stack,
                coverage_fraction_stack,
                native_crop_areas_per_year,
                native_hrl_area_by_year_m2,
                subgrid_hrl_area_by_year_m2,
                selected_fractional_area_by_year_m2,
                selected_modal_area_by_year_m2,
                selected_fallow_area_by_year_m2,
                selected_missing_area_by_year_m2,
                selection_score,
                eligible_mask,
                cultivated_mask,
                region_farm_sizes,
                target_farms,
                local_farms,
                farmers_region,
                crop_alignment_summary_by_year,
                binary_rounding_diagnostics,
            )
            gc.collect()

        if not all_farmers:
            raise ValueError("No HRL-only farmer agents could be created.")

        farmers = pd.concat(all_farmers, ignore_index=True)
        farmers = farmers.sort_values("farmer_id").reset_index(drop=True)
        farms_values[~active_subgrid_mask] = -1
        farms = xr.DataArray(
            farms_values,
            coords=region_ids.coords,
            dims=region_ids.dims,
            attrs=region_ids.attrs,
            name="agents/farmers/farms",
        )
        if region_ids.rio.crs is not None:
            farms = farms.rio.write_crs(region_ids.rio.crs)
        farms.attrs["_FillValue"] = -1

        _assert_compact_farm_ids(
            farms,
            farmers,
            farmer_id_column="farmer_id",
            nodata=-1,
        )

        self.set_table(farmers, name=_FARMERS_WITH_CROPS_TABLE)

        regional_diagnostics = pd.DataFrame(region_diagnostics)
        regional_summary_columns = [
            region_id_column,
            country_iso3_column,
            "lowder_source_iso3",
            "model_area_km2",
            "n_farmers",
            "lowder_target_farms",
            "sequence_fit_target_farms",
            "n_exact_sequences",
            "extra_sequence_farms",
            "local_dominant_sequence_pct",
            "local_sequence_pct",
            "regional_fallback_sequence_pct",
            "novel_sequence_count",
            "excluded_incomplete_hrl_cells",
            "binary_rounding_wape_pct",
            "median_farm_area_ha",
            "multi_parcel_farms_pct",
            "mean_fallow_area_pct",
            "mean_missing_area_pct",
            "mean_selected_fraction_retention_pct",
            "mean_agent_area_retention_pct",
            "maximum_absolute_agent_area_difference_pct",
            "mean_combined_crop_fit_score",
            "mean_main_crop_fit_score",
        ]
        regional_summary = regional_diagnostics[regional_summary_columns].rename(
            columns={
                "lowder_source_iso3": "Lowder",
                "model_area_km2": "static_km2",
                "n_farmers": "agents",
                "lowder_target_farms": "Lowder_n",
                "sequence_fit_target_farms": "sequence_target_n",
                "n_exact_sequences": "sequences",
                "extra_sequence_farms": "extra_n",
                "local_dominant_sequence_pct": "local_dominant_pct",
                "local_sequence_pct": "local_total_pct",
                "regional_fallback_sequence_pct": "regional_fallback_pct",
                "novel_sequence_count": "novel_sequences",
                "excluded_incomplete_hrl_cells": "incomplete_cells",
                "binary_rounding_wape_pct": "rounding_WAPE_pct",
                "median_farm_area_ha": "median_ha",
                "multi_parcel_farms_pct": "multi_parcel_pct",
                "mean_fallow_area_pct": "fallow_area_pct",
                "mean_missing_area_pct": "missing_area_pct",
                "mean_selected_fraction_retention_pct": "fraction_in_union_pct",
                "mean_agent_area_retention_pct": "agent_native_pct",
                "maximum_absolute_agent_area_difference_pct": "worst_area_diff_pct",
                "mean_combined_crop_fit_score": "combined_fit",
                "mean_main_crop_fit_score": "main_fit",
            }
        )
        self.logger.info(
            "HRL-only regional comparison summary:\n%s",
            regional_summary.round(
                {
                    "static_km2": 2,
                    "rounding_WAPE_pct": 2,
                    "median_ha": 2,
                    "multi_parcel_pct": 1,
                    "fallow_area_pct": 1,
                    "missing_area_pct": 2,
                    "fraction_in_union_pct": 1,
                    "agent_native_pct": 1,
                    "worst_area_diff_pct": 1,
                    "combined_fit": 1,
                    "main_fit": 1,
                }
            ).to_string(index=False),
        )

        total_selection_target_m2 = float(
            regional_diagnostics["selection_target_km2"].sum() * 1_000_000.0
        )
        total_selected_area_m2 = float(
            regional_diagnostics["model_area_km2"].sum() * 1_000_000.0
        )
        total_area_difference_pct = (
            (total_selected_area_m2 - total_selection_target_m2)
            / total_selection_target_m2
            * 100.0
            if total_selection_target_m2 > 0
            else np.nan
        )
        all_farm_areas_ha = farmers["area_m2"].to_numpy(dtype=np.float64) / 10_000.0
        all_parcel_counts = farmers["n_fields"].to_numpy(dtype=np.int32)
        self.logger.info(
            "HRL-only overall static-farm diagnostics: %s regions; selected "
            "target %.2f km²; selected model area %.2f km²; difference %+.2f%%; "
            "%s farmers; farm area [min/median/mean/p90/max]="
            "[%.2f/%.2f/%.2f/%.2f/%.2f] ha; %s parcels; %.1f%% "
            "multi-parcel farms.",
            len(regional_diagnostics),
            total_selection_target_m2 / 1_000_000.0,
            total_selected_area_m2 / 1_000_000.0,
            total_area_difference_pct,
            len(farmers),
            all_farm_areas_ha.min(),
            np.median(all_farm_areas_ha),
            all_farm_areas_ha.mean(),
            np.percentile(all_farm_areas_ha, 90),
            all_farm_areas_ha.max(),
            int(all_parcel_counts.sum()),
            np.count_nonzero(all_parcel_counts > 1) / len(farmers) * 100.0,
        )

        crop_area_diagnostics = pd.concat(
            all_crop_area_diagnostics,
            ignore_index=True,
        )
        (
            multi_year_crop_comparison,
            multi_year_crop_summary,
        ) = _multi_year_crop_area_comparison(crop_area_diagnostics)
        (
            _,
            multi_year_main_crop_summary,
        ) = _multi_year_crop_area_comparison(
            crop_area_diagnostics,
            aggregate_to_main_crop=True,
        )
        overall_annual_rows: list[dict[str, float | int]] = []
        low_fit_years: list[int] = []
        for year in years:
            overall_year = (
                crop_area_diagnostics.loc[crop_area_diagnostics["year"] == year]
                .groupby("crop_code", as_index=False)[
                    [
                        "source_area_m2",
                        "adjusted_target_area_m2",
                        "assigned_area_m2",
                    ]
                ]
                .sum()
            )
            overall_year["difference_from_source_m2"] = (
                overall_year["assigned_area_m2"] - overall_year["source_area_m2"]
            )
            overall_year["difference_from_adjusted_target_m2"] = (
                overall_year["assigned_area_m2"]
                - overall_year["adjusted_target_area_m2"]
            )
            positive = overall_year["crop_code"] > 0
            source_total_m2 = float(overall_year.loc[positive, "source_area_m2"].sum())
            assigned_total_m2 = float(
                overall_year.loc[positive, "assigned_area_m2"].sum()
            )
            overall_year["source_share"] = (
                overall_year["source_area_m2"].to_numpy(dtype=np.float64)
                / source_total_m2
                if source_total_m2 > 0.0
                else np.zeros(len(overall_year), dtype=np.float64)
            )
            overall_year["assigned_share"] = (
                overall_year["assigned_area_m2"].to_numpy(dtype=np.float64)
                / assigned_total_m2
                if assigned_total_m2 > 0.0
                else np.zeros(len(overall_year), dtype=np.float64)
            )
            overall_adjusted_targets = overall_year["adjusted_target_area_m2"].to_numpy(
                dtype=np.float64
            )
            overall_source_areas = overall_year["source_area_m2"].to_numpy(
                dtype=np.float64
            )
            overall_year["positive_target_scale"] = np.divide(
                overall_adjusted_targets,
                overall_source_areas,
                out=np.ones(len(overall_year), dtype=np.float64),
                where=overall_source_areas > 0.0,
            )

            combined_fit = _crop_area_fit_scores(overall_year)
            overall_main_year = _aggregate_crop_alignment_to_main_categories(
                overall_year
            )
            main_fit = _crop_area_fit_scores(overall_main_year)
            subgrid_total_m2 = total_subgrid_hrl_area_by_year_m2[year]
            native_to_subgrid_pct = (
                (subgrid_total_m2 - combined_fit["source_area_m2"])
                / combined_fit["source_area_m2"]
                * 100.0
                if combined_fit["source_area_m2"] > 0.0
                else np.nan
            )
            if (
                combined_fit["crop_area_fit_score"]
                < crop_area_fit_warning_threshold_pct
            ):
                low_fit_years.append(int(year))

            selected_fractional_total_m2 = total_selected_fractional_area_by_year_m2[
                year
            ]
            selected_modal_total_m2 = total_selected_modal_area_by_year_m2[year]
            selected_fallow_total_m2 = total_selected_fallow_area_by_year_m2[year]
            selected_missing_total_m2 = total_selected_missing_area_by_year_m2[year]
            agricultural_union_total_m2 = (
                selected_modal_total_m2
                + selected_fallow_total_m2
                + selected_missing_total_m2
            )
            selected_fraction_retention_pct = (
                selected_fractional_total_m2 / subgrid_total_m2 * 100.0
                if subgrid_total_m2 > 0.0
                else np.nan
            )
            modal_conversion_pct = (
                (selected_modal_total_m2 - selected_fractional_total_m2)
                / selected_fractional_total_m2
                * 100.0
                if selected_fractional_total_m2 > 0.0
                else np.nan
            )
            overall_annual_rows.append(
                {
                    "year": int(year),
                    "native_active_km2": (combined_fit["source_area_m2"] / 1_000_000.0),
                    "subgrid_active_km2": subgrid_total_m2 / 1_000_000.0,
                    "fraction_in_union_km2": (
                        selected_fractional_total_m2 / 1_000_000.0
                    ),
                    "agent_active_km2": (
                        combined_fit["assigned_area_m2"] / 1_000_000.0
                    ),
                    "fallow_km2": selected_fallow_total_m2 / 1_000_000.0,
                    "missing_km2": selected_missing_total_m2 / 1_000_000.0,
                    "agricultural_union_km2": (
                        agricultural_union_total_m2 / 1_000_000.0
                    ),
                    "native_subgrid_diff_pct": native_to_subgrid_pct,
                    "fraction_in_union_pct": selected_fraction_retention_pct,
                    "binary_vs_fraction_pct": modal_conversion_pct,
                    "active_native_diff_pct": combined_fit["total_area_difference_pct"],
                    "combined_fit": combined_fit["crop_area_fit_score"],
                    "main_fit": main_fit["crop_area_fit_score"],
                    "share_fit": combined_fit["crop_share_fit_score"],
                    "active_agents_pct": (
                        active_farmer_crops_by_year[year] / len(farmers) * 100.0
                    ),
                    "fallow_agents_pct": (
                        fallow_farmers_by_year[year] / len(farmers) * 100.0
                    ),
                    "missing_agents_pct": (
                        missing_farmers_by_year[year] / len(farmers) * 100.0
                    ),
                }
            )
            if crop_area_diagnostics_top_n > 0:
                self.logger.debug(
                    "Overall detailed HRL crop-area categories for %s (top %s):\n%s",
                    year,
                    crop_area_diagnostics_top_n,
                    _format_hrl_crop_area_alignment(
                        overall_year,
                        top_n=crop_area_diagnostics_top_n,
                    ),
                )

        overall_annual_summary = pd.DataFrame(overall_annual_rows)
        self.logger.info(
            "HRL-only annual active-crop, fallow, and crop-fit summary:\n%s",
            overall_annual_summary.round(
                {
                    "native_active_km2": 2,
                    "subgrid_active_km2": 2,
                    "fraction_in_union_km2": 2,
                    "agent_active_km2": 2,
                    "fallow_km2": 2,
                    "missing_km2": 2,
                    "agricultural_union_km2": 2,
                    "native_subgrid_diff_pct": 2,
                    "fraction_in_union_pct": 1,
                    "binary_vs_fraction_pct": 1,
                    "active_native_diff_pct": 1,
                    "combined_fit": 1,
                    "main_fit": 1,
                    "share_fit": 1,
                    "active_agents_pct": 1,
                    "fallow_agents_pct": 1,
                    "missing_agents_pct": 2,
                }
            ).to_string(index=False),
        )
        multi_year_fit_summary = pd.DataFrame(
            [
                {
                    "crop_level": "combined",
                    "year_crop_pairs": multi_year_crop_summary["n_year_crop_pairs"],
                    "raw_km2": multi_year_crop_summary["raw_area_m2"] / 1_000_000.0,
                    "final_km2": multi_year_crop_summary["final_area_m2"] / 1_000_000.0,
                    "net_diff_pct": multi_year_crop_summary["net_difference_pct"],
                    "area_weighted_fit": multi_year_crop_summary[
                        "area_weighted_fit_pct"
                    ],
                    "balanced_fit": multi_year_crop_summary["balanced_fit_pct"],
                    "area_weighted_error": multi_year_crop_summary[
                        "area_weighted_error_pct"
                    ],
                    "balanced_error": multi_year_crop_summary["balanced_error_pct"],
                },
                {
                    "crop_level": "main",
                    "year_crop_pairs": multi_year_main_crop_summary[
                        "n_year_crop_pairs"
                    ],
                    "raw_km2": multi_year_main_crop_summary["raw_area_m2"]
                    / 1_000_000.0,
                    "final_km2": multi_year_main_crop_summary["final_area_m2"]
                    / 1_000_000.0,
                    "net_diff_pct": multi_year_main_crop_summary["net_difference_pct"],
                    "area_weighted_fit": multi_year_main_crop_summary[
                        "area_weighted_fit_pct"
                    ],
                    "balanced_fit": multi_year_main_crop_summary["balanced_fit_pct"],
                    "area_weighted_error": multi_year_main_crop_summary[
                        "area_weighted_error_pct"
                    ],
                    "balanced_error": multi_year_main_crop_summary[
                        "balanced_error_pct"
                    ],
                },
            ]
        )
        self.logger.info(
            "HRL-only multi-year raw-versus-final crop-area fit across all "
            "year-crop pairs. Area-weighted metrics use max(raw, final) as "
            "the pair weight; balanced metrics give every pair equal weight:\n%s",
            multi_year_fit_summary.round(
                {
                    "raw_km2": 2,
                    "final_km2": 2,
                    "net_diff_pct": 2,
                    "area_weighted_fit": 1,
                    "balanced_fit": 1,
                    "area_weighted_error": 1,
                    "balanced_error": 1,
                }
            ).to_string(index=False),
        )
        self.logger.info(
            "HRL-only raw-versus-final crop-area comparison by year and "
            "combined crop. pair_fit_pct is min(raw, final) / max(raw, final); "
            "area_weight_pct is the pair's share of the total comparison area:\n%s",
            _format_multi_year_crop_area_comparison(multi_year_crop_comparison),
        )

        if low_fit_years:
            self.logger.warning(
                "Combined crop-area fit is below %.1f/100 for years %s.",
                crop_area_fit_warning_threshold_pct,
                low_fit_years,
            )

        farm_size_fit = farm_size_distribution_fit_by_size_class(
            farmers=farmers,
            regions=regions_shapes,
            farm_sizes_per_region=farm_sizes_per_region,
            size_class_boundaries=size_class_boundaries,
            farm_size_donor_country=farm_size_donor_country,
            region_id_column=region_id_column,
            country_iso3_column=country_iso3_column,
            area_column="area_m2",
            logger=self.logger,
        )
        self.logger.info(
            "HRL-only farm-size distribution fit by size class:\n%s",
            farm_size_fit.round(
                {
                    "expected_n_farms_lowder": 1,
                    "difference": 1,
                    "actual_to_expected_ratio": 2,
                    "expected_share": 3,
                    "actual_share": 3,
                }
            ).to_string(index=False),
        )
        self.logger.info(
            "Created %s HRL-only farmer agents on %s cultivated model cells.",
            len(farmers),
            int(np.count_nonzero(farms_values >= 0)),
        )

        self.set_subgrid(farms, name="agents/farmers/farms")
        self.set_array(
            farmers[region_id_column].to_numpy(dtype=np.int32),
            name="agents/farmers/region_id",
        )

        # Build the cultivated-land and land-use outputs positionally. Using
        # ``xr.where`` here can silently depend on coordinate-label alignment
        # between the newly created farm raster and the existing land-use
        # raster. HRU creation, however, combines these arrays strictly by
        # array position. Validate the grids and update the values directly so
        # every non-negative farm cell is guaranteed to have land-use class 1.
        farm_mask_values = farms_values >= 0
        if np.any(farm_mask_values & ~active_subgrid_mask):
            n_outside = int(np.count_nonzero(farm_mask_values & ~active_subgrid_mask))
            raise RuntimeError(
                f"The generated farm raster contains {n_outside} farm cells "
                "outside the active subgrid mask."
            )

        existing_land_use: xr.DataArray = self.subgrid[
            "landsurface/land_use_classes"
        ].compute()
        if existing_land_use.ndim != 2:
            raise ValueError(
                "landsurface/land_use_classes must be a two-dimensional raster."
            )
        if existing_land_use.shape != farms.shape:
            raise ValueError(
                "Farm and land-use rasters must have identical shapes before "
                "HRU creation. Got farms="
                f"{farms.shape} and land_use_classes={existing_land_use.shape}."
            )

        existing_land_use_values = np.asarray(existing_land_use.values)

        # Land-use class -1 is the nodata value outside the active hydrological
        # domain. ``create_HRUs_numba`` skips coarse-grid cells where the model
        # mask is true, so those subgrid values are not required to be one of
        # the active HRU classes. Validate only subgrid cells belonging to the
        # active domain. Inside that domain, every value must be one of the
        # classes accepted by the HRU constructor.
        valid_hru_land_use = np.isin(existing_land_use_values, [0, 1, 4, 5])
        invalid_active_nonfarm_land_use = (
            active_subgrid_mask & ~farm_mask_values & ~valid_hru_land_use
        )
        if np.any(invalid_active_nonfarm_land_use):
            invalid_values, invalid_counts = np.unique(
                existing_land_use_values[invalid_active_nonfarm_land_use],
                return_counts=True,
            )
            invalid_summary = ", ".join(
                f"{int(value)}={int(count)}"
                for value, count in zip(
                    invalid_values,
                    invalid_counts,
                    strict=True,
                )
            )
            invalid_rows, invalid_cols = np.where(invalid_active_nonfarm_land_use)
            sample_count = min(10, invalid_rows.size)
            samples = [
                (
                    int(invalid_rows[index]),
                    int(invalid_cols[index]),
                    int(
                        existing_land_use_values[
                            invalid_rows[index], invalid_cols[index]
                        ]
                    ),
                )
                for index in range(sample_count)
            ]
            raise ValueError(
                "The existing land-use raster contains values outside the HRU "
                "classes {0, 1, 4, 5} on non-farm cells inside the active "
                "subgrid domain. "
                f"Invalid values and counts: [{invalid_summary}]. Sample "
                f"(row, col, land_use): {samples}."
            )

        inactive_land_use_values = existing_land_use_values[~active_subgrid_mask]
        n_inactive_nodata = int(np.count_nonzero(inactive_land_use_values == -1))
        n_inactive_other = int(inactive_land_use_values.size - n_inactive_nodata)
        self.logger.info(
            "Existing land-use validation: all %s active non-farm subgrid "
            "cells use HRU classes {0, 1, 4, 5}; farm cells will be forced to "
            "class 1. Outside the active domain, %s cells use nodata -1 and "
            "%s cells retain another value.",
            int(np.count_nonzero(active_subgrid_mask & ~farm_mask_values)),
            n_inactive_nodata,
            n_inactive_other,
        )

        farm_land_use_before, farm_land_use_before_counts = np.unique(
            existing_land_use_values[farm_mask_values],
            return_counts=True,
        )
        before_summary = ", ".join(
            f"{int(value)}={int(count)}"
            for value, count in zip(
                farm_land_use_before,
                farm_land_use_before_counts,
                strict=True,
            )
        )

        land_use_values = existing_land_use_values.astype(np.int8, copy=True)
        land_use_values[farm_mask_values] = np.int8(1)
        cultivated_land_values = farm_mask_values.astype(bool, copy=True)

        farm_land_use_mismatch = farm_mask_values & (land_use_values != 1)
        nonfarm_marked_cultivated = (~farm_mask_values) & cultivated_land_values
        if np.any(farm_land_use_mismatch) or np.any(nonfarm_marked_cultivated):
            raise RuntimeError(
                "Internal farm/land-use consistency check failed before writing: "
                f"{int(np.count_nonzero(farm_land_use_mismatch))} farm cells "
                "do not have land-use class 1 and "
                f"{int(np.count_nonzero(nonfarm_marked_cultivated))} non-farm "
                "cells are marked as cultivated."
            )

        cultivated_land_subgrid = xr.DataArray(
            cultivated_land_values,
            coords=farms.coords,
            dims=farms.dims,
            name="landsurface/cultivated_land",
        )
        land_use_classes_subgrid = xr.DataArray(
            land_use_values,
            coords=farms.coords,
            dims=farms.dims,
            attrs=existing_land_use.attrs.copy(),
            name="landsurface/land_use_classes",
        )
        if farms.rio.crs is not None:
            cultivated_land_subgrid = cultivated_land_subgrid.rio.write_crs(
                farms.rio.crs
            )
            land_use_classes_subgrid = land_use_classes_subgrid.rio.write_crs(
                farms.rio.crs
            )
        cultivated_land_subgrid.attrs["_FillValue"] = None
        land_use_classes_subgrid.attrs["_FillValue"] = -1

        n_farm_cells = int(np.count_nonzero(farm_mask_values))
        n_changed_to_cropland = int(
            np.count_nonzero(farm_mask_values & (existing_land_use_values != 1))
        )
        self.logger.info(
            "Farm/land-use consistency before writing: %s farm cells; "
            "existing classes on those cells [%s]; %s cells changed to "
            "land-use class 1; final mismatch count 0.",
            n_farm_cells,
            before_summary or "none",
            n_changed_to_cropland,
        )

        self.set_subgrid(
            land_use_classes_subgrid,
            name="landsurface/land_use_classes",
        )
        self.set_subgrid(
            cultivated_land_subgrid,
            name="landsurface/cultivated_land",
        )

        # Validate the arrays registered in the builder as well. This catches a
        # stale or misaligned in-memory subgrid before the model is written and
        # later fails inside the Numba HRU constructor with an uninformative
        # assertion.
        registered_farms = np.asarray(self.subgrid["agents/farmers/farms"].values)
        registered_land_use = np.asarray(
            self.subgrid["landsurface/land_use_classes"].values
        )
        if registered_farms.shape != registered_land_use.shape:
            raise RuntimeError(
                "Registered farm and land-use rasters have different shapes: "
                f"{registered_farms.shape} versus {registered_land_use.shape}."
            )
        registered_farm_mask = registered_farms >= 0
        registered_mismatch = registered_farm_mask & (registered_land_use != 1)
        registered_invalid_active_nonfarm = (
            active_subgrid_mask
            & ~registered_farm_mask
            & ~np.isin(registered_land_use, [0, 1, 4, 5])
        )
        if np.any(registered_invalid_active_nonfarm):
            invalid_values, invalid_counts = np.unique(
                registered_land_use[registered_invalid_active_nonfarm],
                return_counts=True,
            )
            invalid_summary = ", ".join(
                f"{int(value)}={int(count)}"
                for value, count in zip(
                    invalid_values,
                    invalid_counts,
                    strict=True,
                )
            )
            invalid_rows, invalid_cols = np.where(registered_invalid_active_nonfarm)
            sample_count = min(10, invalid_rows.size)
            samples = [
                (
                    int(invalid_rows[index]),
                    int(invalid_cols[index]),
                    int(registered_land_use[invalid_rows[index], invalid_cols[index]]),
                )
                for index in range(sample_count)
            ]
            raise RuntimeError(
                "Registered land-use output contains unsupported values on "
                "active non-farm cells. Invalid values and counts: "
                f"[{invalid_summary}]. Sample (row, col, land_use): {samples}."
            )
        if np.any(registered_mismatch):
            mismatch_rows, mismatch_cols = np.where(registered_mismatch)
            sample_count = min(10, mismatch_rows.size)
            samples = [
                (
                    int(mismatch_rows[index]),
                    int(mismatch_cols[index]),
                    int(registered_farms[mismatch_rows[index], mismatch_cols[index]]),
                    int(
                        registered_land_use[mismatch_rows[index], mismatch_cols[index]]
                    ),
                )
                for index in range(sample_count)
            ]
            mismatch_classes, mismatch_counts = np.unique(
                registered_land_use[registered_mismatch],
                return_counts=True,
            )
            mismatch_summary = ", ".join(
                f"{int(value)}={int(count)}"
                for value, count in zip(
                    mismatch_classes,
                    mismatch_counts,
                    strict=True,
                )
            )
            raise RuntimeError(
                "Farm/land-use consistency failed after registering outputs: "
                f"{int(np.count_nonzero(registered_mismatch))} farm cells do "
                "not have land-use class 1. Mismatching classes: "
                f"[{mismatch_summary}]. Sample (row, col, farm_id, land_use): "
                f"{samples}."
            )
        self.logger.info(
            "Farm/land-use consistency after registering outputs: all %s farm "
            "cells have land-use class 1.",
            int(np.count_nonzero(registered_farm_mask)),
        )

    @build_method(depends_on=["setup_regions_and_land_use"], required=False)
    def setup_farmer_crop_calendar_from_HRL(
        self,
        hrl_year: int = 2017,
        mirca_year: int = 2015,
        minimum_area_ratio: float = 0.01,
        replace_crop_calendar_unit_code: dict[int, int] | None = None,
        multiple_years: bool = False,
        hrl_years: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023),
        reduce_crops: bool = False,
    ) -> None:
        """Build farmer crop calendars by combining HRL crops with MIRCA2000 calendars.

        The final compact farmer table from
        one of the two HRL raster farm-construction methods determines which
        HRL crop sequence assigned to each farmer. HRL crop classes are mapped
        to MIRCA crop classes,
        because crop-growth parametrization is available for MIRCA crops. MIRCA2000
        calendars then provide planting dates and growing-season lengths.

        MIRCA-OS crop-area fractions constrain which farmers receive irrigation
        access. Surface-water irrigation is assigned first to farmers with lower HAND;
        groundwater irrigation is then assigned to remaining farmers with lower
        groundwater depth.

        If ``multiple_years`` is False, only ``hrl_year`` is processed and the usual
        single-year arrays are written:

        - ``agents/farmers/crop_calendar`` with shape ``(farmer, 3, 4)``
        - ``agents/farmers/crop_calendar_rotation_years`` with shape ``(farmer,)``
        - ``agents/farmers/adaptations`` with shape ``(farmer, adaptation)``

        If ``multiple_years`` is True, all years in ``hrl_years`` are processed.
        Crop calendars are stacked by year, while irrigation adaptations are kept as
        one persistent final array:

        - ``agents/farmers/crop_calendar`` with shape ``(year, farmer, 3, 4)``
        - ``agents/farmers/crop_calendar_years`` with shape ``(year,)``
        - ``agents/farmers/crop_calendar_rotation_years`` with shape ``(farmer,)``
        - ``agents/farmers/adaptations`` with shape ``(farmer, adaptation)``

        In multi-year mode, the first processed year defines the baseline irrigation
        assignment. Later years can only add irrigation for farmers whose previous
        processed HRL years all had no valid crop. This avoids irrigation switching
        due to crop switching, while still recovering farmers that were missing crops
        in early HRL years.

        Args:
            hrl_year: HRL crop year used for farmer crop assignment when
                ``multiple_years`` is False.
            mirca_year: MIRCA reference year used for crop calendars and MIRCA-OS
                crop-area fractions.
            minimum_area_ratio: Minimum MIRCA-OS crop-area fraction used inside the
                MIRCA-OS fraction preprocessing.
            replace_crop_calendar_unit_code: Optional mapping to replace MIRCA2000
                unit codes when a unit has missing or unsuitable crop calendars.
            multiple_years: If True, build crop calendars for all years in
                ``hrl_years`` and accumulate irrigation adaptations only for farmers
                with missing crop histories in previous years.
            hrl_years: HRL years processed when ``multiple_years`` is True.
            reduce_crops: Replace rice by a different crop in region 4.

        Raises:
            ValueError: If required final farmer crop-table columns are missing.
            ValueError: If ``multiple_years`` is True and ``hrl_years`` is empty.
            ValueError: If farmers cannot be assigned to valid MIRCA2000 units.
            ValueError: If no MIRCA2000 calendar can be found for an assigned crop.
        """
        if replace_crop_calendar_unit_code is None:
            replace_crop_calendar_unit_code = {}

        if multiple_years and not hrl_years:
            raise ValueError("hrl_years must contain at least one year.")

        years_to_process = tuple(hrl_years) if multiple_years else (hrl_year,)

        n_farmers = self.array["agents/farmers/region_id"].size
        farmer_region_ids = self.array["agents/farmers/region_id"]
        farms = self.subgrid["agents/farmers/farms"]

        farmers_with_crops = self.table[_FARMERS_WITH_CROPS_TABLE]
        if not isinstance(farmers_with_crops, pd.DataFrame):
            farmers_with_crops = pd.read_parquet(farmers_with_crops)

        farmer_areas_m2 = _farmer_area_array_from_farmer_table(
            farmers_with_crops,
            n_farmers=n_farmers,
        )

        farmer_locations = get_farm_locations(farms, method="centroid")

        # MIRCA2000 is used for calendar timing, not for the rainfed/irrigated split.
        MIRCA_unit_grid = self.data_catalog.fetch(MIRCA2000_UNIT_GRID).read()
        assert isinstance(MIRCA_unit_grid, xr.DataArray)

        MIRCA_unit_grid = MIRCA_unit_grid.isel(
            {
                **get_window(
                    MIRCA_unit_grid.x,
                    MIRCA_unit_grid.y,
                    self.bounds,
                    buffer=2,
                ),
                **{"band": 0},
            }
        )

        MIRCA_units = np.unique(MIRCA_unit_grid.values)
        MIRCA_units = MIRCA_units[MIRCA_units > 0].astype(int).tolist()

        crop_calendar = parse_MIRCA2000_crop_calendar(
            self.data_catalog,
            MIRCA_units=MIRCA_units,
        )
        crop_calendar = _fix_365_in_crop_calendar(crop_calendar)
        crop_calendar = _fill_missing_mirca2000_crop_calendars(
            crop_calendar,
            logger=self.logger,
        )

        farmer_mirca_units = sample_from_map(
            MIRCA_unit_grid.values,
            farmer_locations,
            MIRCA_unit_grid.rio.transform(recalc=True).to_gdal(),
        ).astype(np.int32)

        if (farmer_mirca_units <= 0).any():
            raise ValueError(
                "All farmers should be assigned to a valid MIRCA2000 unit."
            )

        # MIRCA-OS is used for the crop-specific rainfed/irrigated area fractions.
        # These fractions are static here, so changes in yearly candidate irrigation
        # assignments come from changing HRL crop assignments, not changing MIRCA-OS.
        rainfed_fraction, irrigated_fraction = self.get_mirca_os_irrigation_fractions(
            year=mirca_year,
            minimum_area_ratio=minimum_area_ratio,
            replace_crop_calendar_unit_code=replace_crop_calendar_unit_code,
            farmer_locations=farmer_locations,
        )

        mirca_os_template = rainfed_fraction.isel(crop=0, drop=True)
        mirca_os_cell_grid = get_linear_indices(mirca_os_template)

        farmer_mirca_os_cells = sample_from_map(
            mirca_os_cell_grid.values,
            farmer_locations,
            mirca_os_cell_grid.rio.transform(recalc=True).to_gdal(),
        ).astype(np.int32)

        fraction_sw_irrigation_data = self.data_catalog.fetch(
            "global_irrigation_area_surface_water"
        ).read()
        fraction_sw_irrigation_data.attrs["_FillValue"] = np.nan
        fraction_sw_irrigation_data = fraction_sw_irrigation_data.isel(
            get_window(
                fraction_sw_irrigation_data.x,
                fraction_sw_irrigation_data.y,
                self.bounds,
                buffer=5,
            )
        )
        fraction_sw_irrigation_data = interpolate_na_2d(fraction_sw_irrigation_data)
        fraction_sw_irrigation_data = fraction_sw_irrigation_data.interp_like(
            mirca_os_template,
            method="nearest",
        )

        fraction_gw_irrigation_data = self.data_catalog.fetch(
            "global_irrigation_area_groundwater"
        ).read()
        fraction_gw_irrigation_data.attrs["_FillValue"] = np.nan
        fraction_gw_irrigation_data = fraction_gw_irrigation_data.isel(
            get_window(
                fraction_gw_irrigation_data.x,
                fraction_gw_irrigation_data.y,
                self.bounds,
                buffer=5,
            )
        )
        fraction_gw_irrigation_data = interpolate_na_2d(fraction_gw_irrigation_data)
        fraction_gw_irrigation_data = fraction_gw_irrigation_data.interp_like(
            mirca_os_template,
            method="nearest",
        )

        surface_water_fraction_by_cell = _build_surface_water_fraction_lookup(
            fraction_sw_irrigation_data,
            fraction_gw_irrigation_data,
            mirca_os_cell_grid,
        )

        hand = self.grid["routing/height_above_nearest_drainage_m"]
        hand = interpolate_na_2d(hand)
        farmer_hand_m = _sample_grid_values_at_farmers(hand, farmer_locations).astype(
            np.float64
        )

        farmer_groundwater_depth_m = self.load_initial_groundwater_depth_at_farmers(
            farmer_locations,
        )

        # Calendar selections depend only on MIRCA unit, crop combination, and
        # irrigation state. Reuse resolved combinations across all HRL years.
        calendar_selection_cache: dict[
            tuple[int, int, int, bool],
            np.ndarray,
        ] = {}
        expected_farmer_ids = np.arange(n_farmers, dtype=np.int32)

        if multiple_years:
            years_array = np.asarray(years_to_process, dtype=np.int32)

            crop_calendar_stack = np.full(
                (years_array.size, n_farmers, 3, 4),
                -1,
                dtype=np.int32,
            )

            # Persistent irrigation is stored only once. Later years may add farmers
            # only if their earlier HRL years did not contain a valid crop.
            persistent_adaptations: np.ndarray | None = None
            farmer_had_valid_crop_before = np.full(n_farmers, False, dtype=bool)

        for year_index, current_hrl_year in enumerate(years_to_process):
            self.logger.info(
                "Setting up HRL-based farmer crop calendars for HRL year %s.",
                current_hrl_year,
            )

            crop_column = f"crop_{current_hrl_year}"

            # Only this part is HRL-year specific. The spatial sampling, MIRCA2000
            # calendar parsing, and MIRCA-OS fraction loading can be reused.
            farmer_crops = _decode_hrl_crop_combinations_from_farmer_table(
                farmers_with_crops,
                crop_column=crop_column,
                n_farmers=n_farmers,
                farmer_region_ids=farmer_region_ids,
                logger=self.logger,
            )

            farmer_ids = farmer_crops["farmer_id"].to_numpy(dtype=np.int32)
            if not np.array_equal(farmer_ids, expected_farmer_ids):
                raise ValueError(
                    "Decoded farmer crops must contain exactly one row per compact "
                    "farmer ID in ascending order."
                )

            farmer_main_crops = farmer_crops["mirca_crop"].to_numpy(dtype=np.int32)
            farmer_secondary_crop_types = farmer_crops["secondary_crop_type"].to_numpy(
                dtype=np.int32
            )

            # Track whether the current HRL year provides a valid MIRCA crop. In
            # multi-year mode, this controls whether later years are allowed to add
            # irrigation for this farmer.
            current_valid_crop = farmer_main_crops != -1

            candidate_is_irrigated, candidate_adaptations = (
                _assign_irrigation_by_area_targets(
                    farmer_crops=farmer_crops,
                    farmer_areas_m2=farmer_areas_m2,
                    farmer_mirca_os_cells=farmer_mirca_os_cells,
                    farmer_hand_m=farmer_hand_m,
                    farmer_groundwater_depth_m=farmer_groundwater_depth_m,
                    rainfed_fraction=rainfed_fraction,
                    irrigated_fraction=irrigated_fraction,
                    surface_water_fraction_by_cell=surface_water_fraction_by_cell,
                    n_farmers=n_farmers,
                    logger=self.logger,
                )
            )

            if multiple_years:
                if persistent_adaptations is None:
                    persistent_adaptations = np.full_like(
                        candidate_adaptations,
                        False,
                        dtype=np.bool_,
                    )

                candidate_source_irrigated = (
                    candidate_adaptations[:, SURFACE_IRRIGATION_EQUIPMENT]
                    | candidate_adaptations[:, WELL_ADAPTATION]
                )
                persistent_source_irrigated = (
                    persistent_adaptations[:, SURFACE_IRRIGATION_EQUIPMENT]
                    | persistent_adaptations[:, WELL_ADAPTATION]
                )

                # Baseline year: all candidate irrigated farmers are accepted because
                # no earlier crop information exists.
                #
                # Later years: only farmers whose previous processed HRL years were
                # all missing can be added. This prevents crop switching from
                # inflating irrigation access, while still recovering farmers that
                # were unclassified in early years.
                eligible_for_later_irrigation = (
                    ~persistent_source_irrigated & ~farmer_had_valid_crop_before
                )
                newly_irrigated = (
                    candidate_source_irrigated & eligible_for_later_irrigation
                )

                if newly_irrigated.any():
                    persistent_adaptations[
                        newly_irrigated,
                        SURFACE_IRRIGATION_EQUIPMENT,
                    ] = candidate_adaptations[
                        newly_irrigated,
                        SURFACE_IRRIGATION_EQUIPMENT,
                    ]
                    persistent_adaptations[
                        newly_irrigated,
                        WELL_ADAPTATION,
                    ] = candidate_adaptations[
                        newly_irrigated,
                        WELL_ADAPTATION,
                    ]

                persistent_source_irrigated = (
                    persistent_adaptations[:, SURFACE_IRRIGATION_EQUIPMENT]
                    | persistent_adaptations[:, WELL_ADAPTATION]
                )

                candidate_count = int(candidate_source_irrigated.sum())
                eligible_count = int(eligible_for_later_irrigation.sum())
                newly_irrigated_count = int(newly_irrigated.sum())
                persistent_count = int(persistent_source_irrigated.sum())

                self.logger.info(
                    "HRL year %s irrigation candidates: %s farmers; eligible "
                    "missing-history farmers: %s; newly added: %s; persistent "
                    "multi-year irrigation after update: %s farmers.",
                    current_hrl_year,
                    candidate_count,
                    eligible_count,
                    newly_irrigated_count,
                    persistent_count,
                )

                # Calendar selection should use the persistent irrigation state. Once
                # a farmer has irrigation access, later calendars can use irrigated
                # variants where available.
                is_irrigated_for_calendar = persistent_source_irrigated
            else:
                is_irrigated_for_calendar = candidate_is_irrigated

            (
                crop_calendar_per_farmer,
                n_unique_calendar_keys,
                n_new_calendar_cache_entries,
            ) = _select_mirca2000_calendars_for_farmers(
                crop_calendar,
                farmer_mirca_units=farmer_mirca_units,
                farmer_main_crops=farmer_main_crops,
                farmer_secondary_crop_types=farmer_secondary_crop_types,
                farmer_is_irrigated=is_irrigated_for_calendar,
                replace_crop_calendar_unit_code=replace_crop_calendar_unit_code,
                selection_cache=calendar_selection_cache,
            )

            self.logger.info(
                "HRL year %s crop calendars resolved from %s unique farmer-state "
                "combination(s); %s new selection(s) added to the cross-year cache.",
                current_hrl_year,
                n_unique_calendar_keys,
                n_new_calendar_cache_entries,
            )

            check_crop_calendar(crop_calendar_per_farmer)

            # For region 4 there are a few instances of rice cultivation but no prices
            if reduce_crops:
                replaced_value = [MIRCA_OS_CROP_CLASS_MAP["Rice"]]

                most_common_check = [
                    crop_value
                    for crop_value in MIRCA_OS_CROP_CLASS_MAP.values()
                    if crop_value not in replaced_value
                ]

                crop_calendar_per_farmer = replace_crop(
                    crop_calendar_per_farmer,
                    most_common_check,
                    replaced_value,
                )

                # The replacement changes calendars, so validate once more only in
                # this branch. Without replacement, the first validation is sufficient.
                check_crop_calendar(crop_calendar_per_farmer)

            if multiple_years:
                crop_calendar_stack[year_index] = crop_calendar_per_farmer

                # Update after processing the year. This ensures the current year can
                # still fill farmers whose previous years were all missing, but it
                # prevents later years from repeatedly adding farmers after a valid
                # crop has appeared once.
                farmer_had_valid_crop_before |= current_valid_crop
            else:
                self.set_array(
                    crop_calendar_per_farmer,
                    name="agents/farmers/crop_calendar",
                )
                self.set_array(
                    np.full(n_farmers, 1, dtype=np.int32),
                    name="agents/farmers/crop_calendar_rotation_years",
                )
                self.set_array(
                    candidate_adaptations,
                    name="agents/farmers/adaptations",
                )

        if multiple_years:
            if persistent_adaptations is None:
                raise ValueError(
                    "No adaptations were created for the selected HRL years."
                )

            final_irrigated_count = int(
                (
                    persistent_adaptations[:, SURFACE_IRRIGATION_EQUIPMENT]
                    | persistent_adaptations[:, WELL_ADAPTATION]
                ).sum()
            )

            self.logger.info(
                "Final persistent multi-year irrigation count: %s farmers.",
                final_irrigated_count,
            )

            self.set_array(
                years_array,
                name="agents/farmers/crop_calendar_years",
            )
            self.set_array(
                crop_calendar_stack,
                name="agents/farmers/crop_calendar",
            )

            # Rotation length is still one year. The crop calendar itself varies by
            # year, but the model should not interpret this as a multi-year rotation
            # cycle unless that is implemented explicitly elsewhere.
            self.set_array(
                np.full(n_farmers, 1, dtype=np.int32),
                name="agents/farmers/crop_calendar_rotation_years",
            )
            self.set_array(
                persistent_adaptations,
                name="agents/farmers/adaptations",
            )

    def get_mirca_os_irrigation_fractions(
        self,
        *,
        year: int,
        minimum_area_ratio: float = 0.01,
        replace_crop_calendar_unit_code: dict[int, int] | None = None,
        farmer_locations: np.ndarray | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Derive MIRCA-OS rainfed and irrigated crop-area fractions.

        Args:
            year: MIRCA reference year.
            minimum_area_ratio: Minimum crop-area fraction retained during sampling.
            replace_crop_calendar_unit_code: Optional MIRCA-unit replacement mapping.
            farmer_locations: Optional farmer centroid coordinates with shape
                ``(n_farmers, 2)``. Locations are derived from the farm raster when
                omitted.

        Returns:
            Tuple containing rainfed and irrigated crop-area fraction arrays with
            dimensions ``crop``, ``y``, and ``x``.

        Raises:
            ValueError: If provided farmer locations are not aligned with the number of
                farmers.
        """
        if replace_crop_calendar_unit_code is None:
            replace_crop_calendar_unit_code = {}

        n_farmers = self.array["agents/farmers/region_id"].size

        # For alignment of various input data, we need a reference. So we just
        # load one. The crops itself are not used, but just the metadata.
        reference_crop_map = self.data_catalog.fetch(
            f"mirca_os_cropping_area_{year}_5-arcminute_Wheat_rf"
        ).read()
        reference_map_buffer: int = 100
        reference_crop_map = reference_crop_map.isel(
            get_window(
                reference_crop_map.x,
                reference_crop_map.y,
                self.bounds,
                buffer=reference_map_buffer,
                raise_on_buffer_out_of_bounds=False,
            )
            # A large buffer prevents interpolation artefacts near the model edge.
        )

        # Load MIRCA-OS data for the given year
        MIRCA_unit_geom = self.data_catalog.fetch(
            f"mirca_os_admin_boundaries_{year}"
        ).read()
        assert isinstance(MIRCA_unit_geom, gpd.GeoDataFrame)

        # Clip geometries to the reference crop map extent so they remain aligned.
        MIRCA_unit_geom = MIRCA_unit_geom.cx[
            reference_crop_map.x.values.min() : reference_crop_map.x.values.max(),
            reference_crop_map.y.values.min() : reference_crop_map.y.values.max(),
        ]

        if farmer_locations is None:
            farmer_locations = get_farm_locations(
                self.subgrid["agents/farmers/farms"],
                method="centroid",
            )
        else:
            farmer_locations = np.asarray(farmer_locations)
            if farmer_locations.shape != (n_farmers, 2):
                raise ValueError(
                    "farmer_locations must have shape (n_farmers, 2). "
                    f"Got {farmer_locations.shape} for {n_farmers} farmers."
                )

        MIRCA_unit_grid = rasterize_like(
            MIRCA_unit_geom,
            reference_crop_map,
            dtype=np.int32,
            nodata=-1,
            column="unit_code",
            name="MIRCA_unit",
        )
        MIRCA_unit_grid.values = fillna_2d(MIRCA_unit_grid.values, nodata=-1)
        farmer_mirca_units = sample_from_map(
            MIRCA_unit_grid.values,
            farmer_locations,
            MIRCA_unit_grid.rio.transform(recalc=True).to_gdal(),
        )

        assert not (farmer_mirca_units == -1).any(), (
            "All farmers should be assigned to a MIRCA unit."
        )

        rainfed_fraction, irrigated_fraction, MIRCA_unit_grid, farmer_mirca_units = (
            self.get_crop_area_fractions(
                year,
                MIRCA_unit_grid,
                MIRCA_unit_geom,
                farmer_mirca_units,
                reference_crop_map,
                reference_map_buffer,
            )
        )

        return rainfed_fraction, irrigated_fraction

    def load_initial_groundwater_depth_at_farmers(
        self,
        farmer_locations: np.ndarray,
    ) -> np.ndarray:
        """Load initial groundwater depth at farmer locations.

        Args:
            farmer_locations: Farmer centroid coordinates.

        Returns:
            One-dimensional array with initial groundwater depth at each farmer.
        """
        layer_boundary_elevation = self.grid["groundwater/layer_boundary_elevation"]
        layer_boundary_elevation = interpolate_na_along_dim(layer_boundary_elevation)

        heads = self.grid["groundwater/heads"]
        heads = interpolate_na_along_dim(heads)

        heads = np.where(
            ~np.isnan(heads),
            heads,
            layer_boundary_elevation[1:] + 0.1,
        )
        heads = np.where(
            heads > layer_boundary_elevation[1:],
            heads,
            layer_boundary_elevation[1:] + 0.1,
        )

        initial_head = heads[0]
        surface_elevation = layer_boundary_elevation[0]
        groundwater_depth = surface_elevation - initial_head

        return _sample_grid_values_at_farmers(groundwater_depth, farmer_locations)
