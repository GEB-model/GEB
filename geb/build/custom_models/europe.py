"""Class to set GEB up for Europe."""

import gc
from dataclasses import dataclass
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
from geb.build.workflows.crop_calendars import (
    MIRCA_OS_CROP_CLASS_MAP,
    parse_MIRCA_crop_calendar,
)
from geb.build.workflows.farmers import (
    alphaearth_crop_feature_importance,
    alphaearth_crop_training_samples_path,
    alphaearth_embedding_diagnostics,
    apply_alphaearth_cty_mmu_sieve,
    apply_alphaearth_permanent_crop_temporal_consistency,
    assign_farmer_sequences_to_area_targets,
    create_alphaearth_crop_training_samples,
    create_lowder_target_farm_areas,
    evaluate_alphaearth_crop_models,
    format_alphaearth_accuracy_report,
    farm_size_distribution_fit_by_size_class,
    fit_alphaearth_crop_models,
    get_farm_locations,
    grow_farms_from_raster_cells,
    build_hrl_prediction_tile_name,
    find_hrl_tile_path,
    hrl_tile_code_from_name,
    europe_model_build_context,
    load_alphaearth_crop_models,
    load_alphaearth_crop_training_samples,
    load_europe_alphaearth_crop_training_samples,
    parse_europe_model_ids,
    predict_alphaearth_crop_tile_to_hrl_geotiffs,
    raster_cell_area_m2,
    relax_lowder_targets_for_sequence_fit,
    remove_alphaearth_downloads,
    save_alphaearth_crop_models,
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


def _decode_hrl_crops_from_farmer_table(
    farmers_with_crops: pd.DataFrame,
    *,
    crop_column: str,
    n_farmers: int,
    farmer_region_ids: np.ndarray,
    logger: Any,
) -> pd.DataFrame:
    """Map final farmer-level HRL CTY codes to MIRCA crop classes.

    Args:
        farmers_with_crops: Final compact farmer table with ``farmer_id``,
            ``area_m2``, and HRL crop-sequence columns.
        crop_column: HRL CTY column to use, for example ``"crop_2023"``.
        n_farmers: Number of final compact farmers.
        farmer_region_ids: Region ID per final compact farmer.
        logger: Logger used for warnings.

    Returns:
        DataFrame with one row per final compact farmer and columns
        ``farmer_id``, ``mirca_crop``, and ``assigned_crop_area_m2``.

    Raises:
        ValueError: If required columns are missing, farmer IDs are invalid, or
            no farmer rows are available.
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

    hrl_crop = farmers[crop_column].fillna(-1).to_numpy(dtype=np.int32)
    invalid_crop = np.isin(
        hrl_crop,
        (
            _HRL_MISSING_CROP_CODE,
            _HRL_FALLOW_CROP_CODE,
            _HRL_NO_CROPLAND_CODE,
            _HRL_OUTSIDE_AREA_CODE,
        ),
    )
    hrl_crop = np.where(invalid_crop, -1, hrl_crop).astype(np.int32)
    mirca_crop = map_hrl_crop_to_mirca_crop(hrl_crop)

    farmer_crops = pd.DataFrame(
        {
            "farmer_id": farmers["farmer_id"].to_numpy(dtype=np.int32),
            "mirca_crop": mirca_crop.astype(np.int32),
            "assigned_crop_area_m2": farmers["area_m2"].to_numpy(dtype=np.float64),
        }
    )
    if farmer_crops.empty:
        raise ValueError(f"No farmer-level HRL crops are available in {crop_column!r}.")

    missing_farmers = np.setdiff1d(
        np.arange(n_farmers, dtype=np.int32),
        farmer_crops["farmer_id"].to_numpy(dtype=np.int32),
    )
    if missing_farmers.size:
        logger.warning(
            "No HRL CTY row is available for %s farmer(s); filling with the "
            "regional modal MIRCA crop.",
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
    """Fill missing farmer crop rows with regional modal MIRCA crops."""
    crop_lookup = farmer_crops.copy()
    crop_lookup["region_id"] = farmer_region_ids[
        crop_lookup["farmer_id"].to_numpy(dtype=np.int32)
    ]
    valid_lookup = crop_lookup.loc[crop_lookup["mirca_crop"] >= 0]
    if valid_lookup.empty:
        raise ValueError("No valid HRL-to-MIRCA crop exists for regional fallback.")

    global_mode = int(valid_lookup["mirca_crop"].mode().iloc[0])
    fallback_rows: list[dict[str, float | int]] = []
    for farmer_id in missing_farmers:
        region_id = int(farmer_region_ids[farmer_id])
        region_rows = valid_lookup.loc[valid_lookup["region_id"] == region_id]
        mirca_crop = (
            global_mode
            if region_rows.empty
            else int(region_rows["mirca_crop"].mode().iloc[0])
        )
        fallback_rows.append(
            {
                "farmer_id": int(farmer_id),
                "mirca_crop": mirca_crop,
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


def _calendar_active_rows(calendar: np.ndarray) -> np.ndarray:
    """Return active crop rows from a crop calendar matrix.

    Args:
        calendar: Crop calendar matrix.

    Returns:
        Rows where the crop ID is not ``-1``.
    """
    return calendar[calendar[:, 0] != -1]


def _candidate_mirca_calendars(
    crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
    *,
    mirca_unit: int,
    main_crop: int,
    is_irrigated: bool,
    replace_crop_calendar_unit_code: dict[int, int],
) -> tuple[int, list[tuple[float, TwoDArrayInt32]]]:
    """Return ordered MIRCA-OS calendar candidates for one farmer state.

    The search order is local unit with matching irrigation status, local unit
    regardless of irrigation status, other units with matching status, and finally
    other units regardless of status.
    """
    lookup_unit = int(replace_crop_calendar_unit_code.get(mirca_unit, mirca_unit))

    def contains_crop(entry: tuple[float, TwoDArrayInt32]) -> bool:
        active_rows = _calendar_active_rows(entry[1])
        return active_rows.size > 0 and main_crop in active_rows[:, 0]

    def matches_irrigation(entry: tuple[float, TwoDArrayInt32]) -> bool:
        active_rows = _calendar_active_rows(entry[1])
        return active_rows.size > 0 and bool(active_rows[0, 1]) == is_irrigated

    local_entries = crop_calendar.get(lookup_unit, [])
    search_groups = [
        [e for e in local_entries if contains_crop(e) and matches_irrigation(e)],
        [e for e in local_entries if contains_crop(e)],
        [
            e
            for unit_code, entries in crop_calendar.items()
            if unit_code != lookup_unit
            for e in entries
            if contains_crop(e) and matches_irrigation(e)
        ],
        [
            e
            for unit_code, entries in crop_calendar.items()
            if unit_code != lookup_unit
            for e in entries
            if contains_crop(e)
        ],
    ]
    for candidates in search_groups:
        if candidates:
            return lookup_unit, candidates

    raise ValueError(
        f"No MIRCA-OS calendar found for unit={lookup_unit}, crop={main_crop}, "
        f"is_irrigated={is_irrigated}, including cross-unit fallbacks."
    )


def _select_mirca_calendars_for_farmers(
    crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
    *,
    farmer_mirca_units: np.ndarray,
    farmer_main_crops: np.ndarray,
    farmer_is_irrigated: np.ndarray,
    replace_crop_calendar_unit_code: dict[int, int],
    selection_cache: dict[tuple[int, int, int, bool], np.ndarray],
    random_seed: int,
) -> tuple[np.ndarray, int, int]:
    """Assign area-weighted MIRCA-OS calendars using only the HRL main crop.

    Each farmer/crop/irrigation state is sampled deterministically from the
    represented MIRCA-OS calendar areas. The cache keeps repeated crop states stable
    across HRL years.
    """
    farmer_mirca_units = np.asarray(farmer_mirca_units, dtype=np.int32)
    farmer_main_crops = np.asarray(farmer_main_crops, dtype=np.int32)
    farmer_is_irrigated = np.asarray(farmer_is_irrigated, dtype=bool)
    arrays = (farmer_mirca_units, farmer_main_crops, farmer_is_irrigated)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError(
            "All farmer calendar-selection arrays must be one-dimensional."
        )
    n_farmers = farmer_mirca_units.size
    if any(array.size != n_farmers for array in arrays[1:]):
        raise ValueError("All farmer calendar-selection arrays must have equal length.")

    calendars = np.full((n_farmers, 3, 4), -1, dtype=np.int32)
    state_keys: set[tuple[int, int, bool]] = set()
    n_cache_misses = 0

    for farmer_id in range(n_farmers):
        main_crop = int(farmer_main_crops[farmer_id])
        if main_crop == -1:
            continue
        mirca_unit = int(farmer_mirca_units[farmer_id])
        is_irrigated = bool(farmer_is_irrigated[farmer_id])
        lookup_unit = int(replace_crop_calendar_unit_code.get(mirca_unit, mirca_unit))
        state_keys.add((lookup_unit, main_crop, is_irrigated))
        cache_key = (farmer_id, lookup_unit, main_crop, is_irrigated)
        selected = selection_cache.get(cache_key)
        if selected is None:
            _, candidates = _candidate_mirca_calendars(
                crop_calendar,
                mirca_unit=lookup_unit,
                main_crop=main_crop,
                is_irrigated=is_irrigated,
                replace_crop_calendar_unit_code={},
            )
            areas = np.asarray([max(float(area), 0.0) for area, _ in candidates])
            probabilities = (
                areas / areas.sum()
                if areas.sum() > 0.0
                else np.full(len(candidates), 1.0 / len(candidates))
            )
            seed = np.random.SeedSequence(
                [random_seed, farmer_id, lookup_unit, main_crop, int(is_irrigated)]
            )
            rng = np.random.default_rng(seed)
            candidate_index = int(rng.choice(len(candidates), p=probabilities))
            full_calendar = candidates[candidate_index][1]
            selected = np.asarray(full_calendar[:, [0, 2, 3, 4]], dtype=np.int32)
            if selected.shape != (3, 4):
                raise ValueError(
                    "Selected MIRCA-OS calendar must have shape (3, 4) after "
                    f"column selection. Got {selected.shape}."
                )
            selected = np.ascontiguousarray(selected)
            selection_cache[cache_key] = selected
            n_cache_misses += 1
        calendars[farmer_id] = selected

    return calendars, len(state_keys), n_cache_misses


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


def _native_hrl_crop_category_areas_m2(
    crop_types: xr.DataArray,
    clip_geometry: BaseGeometry,
    *,
    geometry_crs: str = "EPSG:4326",
    chunk_rows: int = 4096,
) -> dict[int, float]:
    """Calculate native HRL CTY area by main crop inside a geometry."""
    if crop_types.rio.crs is None:
        raise ValueError("The native HRL Crop Types raster must have a CRS.")
    if crop_types.ndim != 2:
        raise ValueError("The native HRL Crop Types raster must be two-dimensional.")
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
        crop_values = crop_chunk_da.values
        if np.issubdtype(crop_values.dtype, np.floating):
            crop_values = np.nan_to_num(crop_values, nan=_HRL_OUTSIDE_AREA_CODE)
        crop_values = np.asarray(crop_values, dtype=np.int32)
        valid_crop = (crop_values > _HRL_NO_CROPLAND_CODE) & (
            crop_values != _HRL_OUTSIDE_AREA_CODE
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

        crop_codes, inverse = np.unique(crop_values[valid], return_inverse=True)
        if projected_cell_area_m2 is not None:
            crop_areas_m2 = (
                np.bincount(inverse).astype(np.float64) * projected_cell_area_m2
            )
        else:
            chunk_cell_area_m2 = raster_cell_area_m2(crop_chunk_da)
            crop_areas_m2 = np.bincount(inverse, weights=chunk_cell_area_m2[valid])
        for crop_code, crop_area_m2 in zip(crop_codes, crop_areas_m2, strict=True):
            if int(crop_code) > 0 and crop_area_m2 > 0.0:
                totals[int(crop_code)] = totals.get(int(crop_code), 0.0) + float(
                    crop_area_m2
                )
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
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compare raw and final CTY areas over every year-crop combination."""
    required_columns = {"year", "crop_code", "source_area_m2", "assigned_area_m2"}
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
    pair_fit_pct = (
        np.divide(
            overlapping_area,
            comparison_area,
            out=np.zeros_like(overlapping_area),
            where=comparison_area > 0.0,
        )
        * 100.0
    )
    total_comparison_area = float(comparison_area.sum())
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
    return table, {
        "n_year_crop_pairs": int(len(table)),
        "raw_area_m2": raw_total,
        "final_area_m2": final_total,
        "net_difference_pct": (
            (final_total - raw_total) / raw_total * 100.0 if raw_total > 0.0 else np.nan
        ),
        "area_weighted_fit_pct": (
            float(overlapping_area.sum()) / total_comparison_area * 100.0
            if total_comparison_area > 0.0
            else np.nan
        ),
        "balanced_fit_pct": float(pair_fit_pct.mean()),
        "area_weighted_error_pct": (
            float(absolute_error.sum()) / total_comparison_area * 100.0
            if total_comparison_area > 0.0
            else np.nan
        ),
        "balanced_error_pct": float((100.0 - pair_fit_pct).mean()),
    }


def _format_multi_year_crop_area_comparison(comparison: pd.DataFrame) -> str:
    """Format raw-versus-final CTY area comparisons for logging."""
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
    """Format the largest CTY-versus-agent crop-area differences for logging."""
    if diagnostics.empty:
        return "no crop categories"
    table = diagnostics.copy()
    table = table.loc[
        (table["source_area_m2"] > 0.0) | (table["assigned_area_m2"] > 0.0)
    ].copy()
    if table.empty:
        return "no positive crop area"
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
        table["source_area_m2"], table["assigned_area_m2"]
    )
    table = table.sort_values(
        ["ranking_area", "crop_code"], ascending=[False, True]
    ).head(max(top_n, 1))
    return (
        table[
            [
                "crop_code",
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


def _crop_area_diagnostics_from_assignments(
    assigned_crop_codes: np.ndarray,
    farmer_areas_m2: np.ndarray,
    source_crop_areas_m2: dict[int, float],
) -> pd.DataFrame:
    """Compare existing farmer crop assignments with source area targets.

    Args:
        assigned_crop_codes: One assigned CTY crop code per farmer.
        farmer_areas_m2: Area of each farmer in square metres.
        source_crop_areas_m2: Source area target by CTY crop code.

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


def _reproject_HRL_year_to_subgrid(
    crop_types: xr.DataArray,
    template: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate one native-resolution HRL CTY year to the model subgrid."""
    if crop_types.rio.crs is None:
        raise ValueError("The HRL Crop Types raster must have a CRS.")
    if crop_types.ndim != 2:
        raise ValueError("The HRL Crop Types raster must be two-dimensional.")
    if template.rio.crs is None:
        raise ValueError("The regional subgrid template must have a CRS.")

    crop_values = crop_types.values
    if np.issubdtype(crop_values.dtype, np.floating):
        crop_values = np.nan_to_num(crop_values, nan=_HRL_OUTSIDE_AREA_CODE)
    crop_values = np.ascontiguousarray(crop_values.astype(np.int32, copy=False))
    has_hrl_coverage = crop_values != _HRL_OUTSIDE_AREA_CODE
    active_crop = (crop_values > _HRL_NO_CROPLAND_CODE) & has_hrl_coverage

    crop_states = crop_values.copy()
    crop_states[~active_crop] = _HRL_MISSING_CROP_CODE
    crop_state_da = crop_types.copy(data=crop_states)
    crop_state_da.attrs = {}
    crop_state_da = crop_state_da.rio.write_crs(crop_types.rio.crs)
    crop_state_da = crop_state_da.rio.write_nodata(_HRL_MISSING_CROP_CODE)
    crop_subgrid = crop_state_da.rio.reproject_match(
        template, resampling=Resampling.mode, nodata=_HRL_MISSING_CROP_CODE
    )
    crop_subgrid_values = crop_subgrid.values
    if np.issubdtype(crop_subgrid_values.dtype, np.floating):
        crop_subgrid_values = np.nan_to_num(
            crop_subgrid_values, nan=_HRL_MISSING_CROP_CODE
        )
    crop_subgrid_values = crop_subgrid_values.astype(np.int32, copy=False)

    cultivated = crop_types.copy(data=active_crop.astype(np.float32))
    cultivated.attrs = {}
    cultivated = cultivated.rio.write_crs(crop_types.rio.crs).rio.write_nodata(None)
    cultivated_fraction = cultivated.rio.reproject_match(
        template, resampling=Resampling.average, nodata=0.0
    )
    cultivated_fraction_values = np.clip(
        np.nan_to_num(cultivated_fraction.values, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)

    coverage = crop_types.copy(data=has_hrl_coverage.astype(np.float32))
    coverage.attrs = {}
    coverage = coverage.rio.write_crs(crop_types.rio.crs).rio.write_nodata(None)
    coverage_fraction = coverage.rio.reproject_match(
        template, resampling=Resampling.average, nodata=0.0
    )
    coverage_fraction_values = np.clip(
        np.nan_to_num(coverage_fraction.values, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)

    no_active_modal = crop_subgrid_values == _HRL_MISSING_CROP_CODE
    crop_subgrid_values[no_active_modal & (coverage_fraction_values > 0.0)] = (
        _HRL_NO_CROPLAND_CODE
    )
    return crop_subgrid_values, cultivated_fraction_values, coverage_fraction_values


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
    def setup_sample_alphaearth_crop_classification(
        self,
        europe_model_ids: tuple[int | str, ...] = tuple(range(12)),
        region_id_column: str = "region_id",
        country_iso3_column: str = "ISO3",
        hrl_years: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023),
        samples_per_cty_class_per_region_year: int = 1000,
        sample_stride_pixels: int = 5,
        rare_class_sample_stride_pixels: int | None = 1,
        rare_class_threshold_candidates: int = 50_000,
        rare_class_sample_multiplier: float = 3.0,
        training_label_edge_buffer_pixels: int = 2,
        sample_chunk_rows: int = 512,
        include_coordinates: bool = False,
        include_topography: bool = False,
        random_seed: int = 42,
        hrl_raster_chunks: dict[str, int] | None = None,
        max_alphaearth_files_for_sampling: int | None = None,
        alphaearth_max_parallel_downloads: int | None = None,
        cleanup_alphaearth_downloads: bool = True,
        training_samples_table_name: str = (
            "machine_learning/crop_classification/samples"
        ),
        overwrite_training_samples: bool = True,
    ) -> None:
        """Sample annual HRL CTY labels and same-year AlphaEarth embeddings.

        This is the first stage of the Europe-wide workflow. The same method can be
        configured in every ``Europe_###`` model. ``europe_model_ids`` determines
        which model-level builds participate; each participating build downloads
        AlphaEarth COGs once for its complete active model extent, samples every
        contained region and year, writes one model-local sample table, and removes
        the downloaded COGs by default.

        The resulting model-local tables are pooled by
        :meth:`setup_train_alphaearth_crop_classification`. Keeping sampling separate
        from fitting allows estimator and parameter experiments to reuse the same
        expensive Earth-observation samples.

        Args:
            europe_model_ids: Europe models included in the pooled workflow. Accepts
                integers, ``Europe_###`` names, comma-separated values, and ranges.
            region_id_column: Column containing compact model-region IDs.
            country_iso3_column: Column containing country ISO3 codes.
            hrl_years: Annual HRL and AlphaEarth years sampled together.
            samples_per_cty_class_per_region_year: Maximum retained samples per CTY
                class, region, and year.
            sample_stride_pixels: Standard native-HRL sampling-lattice spacing.
            rare_class_sample_stride_pixels: Denser lattice spacing for rare classes.
            rare_class_threshold_candidates: Candidate-count threshold defining rarity.
            rare_class_sample_multiplier: Sample-cap multiplier for rare classes.
            training_label_edge_buffer_pixels: Pixels removed from CTY boundaries.
            sample_chunk_rows: HRL rows scanned in each sampling chunk.
            include_coordinates: Store longitude and latitude predictors.
            include_topography: Store elevation and slope predictors.
            random_seed: Reproducible sampling seed.
            hrl_raster_chunks: Optional native HRL read chunks.
            max_alphaearth_files_for_sampling: Optional model-level download limit.
            alphaearth_max_parallel_downloads: Optional parallel-download override.
            cleanup_alphaearth_downloads: Remove downloaded AlphaEarth COGs after
                sampling. Defaults to ``True``.
            training_samples_table_name: Model-local ``self.table`` output name.
            overwrite_training_samples: Replace an existing model-local sample table.

        Raises:
            FileExistsError: If the sample table exists and overwrite is disabled.
            ValueError: If years, settings, geometry, or samples are invalid.
        """
        selected_model_ids = parse_europe_model_ids(europe_model_ids)
        current_model_id, current_base, _ = europe_model_build_context()
        current_model_name = f"Europe_{current_model_id:03d}"
        if current_model_id not in selected_model_ids:
            self.logger.info(
                "Skipping AlphaEarth CTY sampling for %s; selected models are %s.",
                current_model_name,
                [f"Europe_{model_id:03d}" for model_id in selected_model_ids],
            )
            return

        sample_output_path = alphaearth_crop_training_samples_path(
            current_base,
            training_samples_table_name,
        )
        if sample_output_path.exists() and not overwrite_training_samples:
            raise FileExistsError(
                "AlphaEarth CTY samples already exist and "
                f"overwrite_training_samples=False: {sample_output_path}"
            )

        hrl_years = tuple(int(year) for year in hrl_years)
        if not hrl_years:
            raise ValueError("hrl_years must contain at least one year.")
        if tuple(sorted(set(hrl_years))) != hrl_years:
            raise ValueError("hrl_years must be unique and strictly increasing.")
        if samples_per_cty_class_per_region_year < 1:
            raise ValueError(
                "samples_per_cty_class_per_region_year must be at least one."
            )
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
        if (
            alphaearth_max_parallel_downloads is not None
            and alphaearth_max_parallel_downloads < 1
        ):
            raise ValueError("alphaearth_max_parallel_downloads must be at least 1.")

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
            "Sampling %s with %s simultaneous AlphaEarth download(s).",
            current_model_name,
            alphaearth_adapter.max_parallel_downloads,
        )

        def read_cty_year(
            *,
            year: int,
            region_bounds: tuple[float, float, float, float],
        ) -> xr.DataArray:
            """Read one annual native-grid HRL CTY observation."""
            crop_types_adapter = self.data_catalog.fetch(
                f"hrl_crop_types_{year}",
                bounds=region_bounds,
                year=year,
            )
            return crop_types_adapter.read(
                bounds=region_bounds,
                year=year,
                dst_crs=None,
                normalize_nodata=False,
                chunks=raster_chunks,
            )

        sample_tables: list[pd.DataFrame] = []
        selected_training_cogs: gpd.GeoDataFrame | None = None
        try:
            self.logger.info(
                "Downloading AlphaEarth COGs for years %s and the complete %s "
                "model extent %s before regional sampling.",
                hrl_years,
                current_model_name,
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
                    "No AlphaEarth training COGs were selected for "
                    f"{current_model_name}."
                )

            for region_index, (_, region) in enumerate(regions_wgs84.iterrows()):
                region_id = int(region[region_id_column])
                country_iso3 = str(region[country_iso3_column])
                region_mask_full = active_subgrid_mask & (region_id_values == region_id)
                if not region_mask_full.any():
                    continue
                region_geometry = region.geometry.intersection(active_geometry)
                if region_geometry.is_empty:
                    continue
                region_bounds = tuple(float(value) for value in region_geometry.bounds)

                for year in hrl_years:
                    try:
                        crop_types = read_cty_year(
                            year=year,
                            region_bounds=region_bounds,
                        )
                        region_year_cogs = select_alphaearth_cogs_for_geometry(
                            selected_training_cogs,
                            year=year,
                            clip_geometry=region_geometry,
                        )
                        if region_year_cogs.empty:
                            raise ValueError(
                                "No downloaded AlphaEarth COGs intersect region "
                                f"{region_id}, year {year}, in {current_model_name}."
                            )

                        region_year_samples = create_alphaearth_crop_training_samples(
                            crop_types,
                            region_year_cogs,
                            region_geometry,
                            year=year,
                            region_id=region_id,
                            country_iso3=country_iso3,
                            samples_per_cty_class=(
                                samples_per_cty_class_per_region_year
                            ),
                            sample_stride_pixels=sample_stride_pixels,
                            rare_class_sample_stride_pixels=(
                                rare_class_sample_stride_pixels
                            ),
                            rare_class_threshold_candidates=(
                                rare_class_threshold_candidates
                            ),
                            rare_class_sample_multiplier=rare_class_sample_multiplier,
                            training_label_edge_buffer_pixels=(
                                training_label_edge_buffer_pixels
                            ),
                            sample_chunk_rows=sample_chunk_rows,
                            include_coordinates=include_coordinates,
                            include_topography=include_topography,
                            elevation=subgrid_elevation,
                            slope=subgrid_slope,
                            random_seed=(
                                random_seed
                                + current_model_id * 1_000_000
                                + region_index * 10_000
                                + year
                            ),
                        )
                        sample_tables.append(region_year_samples)
                    except WEkEONoCoverageError as error:
                        if country_iso3.upper() in _HRL_CROPLANDS_EEA38_ISO3:
                            raise
                        self.logger.warning(
                            "Skipping region %s (%s), year %s in %s: no HRL "
                            "coverage. %s",
                            region_id,
                            country_iso3,
                            year,
                            current_model_name,
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
                    "Removed %s AlphaEarth COGs after sampling %s.",
                    removed,
                    current_model_name,
                )
            elif selected_training_cogs is not None:
                self.logger.info(
                    "Retaining %s AlphaEarth COGs in the local cache for %s.",
                    len(selected_training_cogs),
                    current_model_name,
                )

        if not sample_tables:
            raise ValueError(
                "No annual AlphaEarth-HRL CTY samples were created for "
                f"{current_model_name}."
            )
        samples = pd.concat(sample_tables, ignore_index=True)
        samples["europe_model_id"] = np.int16(current_model_id)
        samples["europe_model_name"] = current_model_name

        embedding_diagnostics = alphaearth_embedding_diagnostics(samples)
        embedding_diagnostics.update(
            {
                "europe_model_id": current_model_id,
                "europe_model_name": current_model_name,
                "sample_count": len(samples),
            }
        )
        self.set_table(
            pd.DataFrame([embedding_diagnostics]),
            name="machine_learning/crop_classification/embedding_diagnostics",
        )
        self.set_table(samples, name=training_samples_table_name)
        self.logger.info(
            "Stored %s AlphaEarth-HRL CTY samples for %s at %s.",
            f"{len(samples):,}",
            current_model_name,
            sample_output_path,
        )

    @build_method(
        depends_on=["setup_regions_and_land_use"],
        required=False,
    )
    def setup_train_alphaearth_crop_classification(
        self,
        europe_model_ids: tuple[int | str, ...] = tuple(range(12)),
        training_model_id: int | str | None = None,
        hrl_years: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023),
        validation_year: int | None = 2022,
        test_year: int | None = 2023,
        include_coordinates: bool = False,
        include_topography: bool = False,
        normalize_embeddings: bool = True,
        classifier_structure: str = "flat",
        residual_class_mode: str = "uncertainty",
        calibrate_class_thresholds: bool = False,
        class_probability_thresholds: dict[int, float] | None = None,
        class_threshold_grid: tuple[float, ...] = (
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.50,
        ),
        class_threshold_min_reference_samples: int = 25,
        unclassified_probability_threshold: float = 0.25,
        model_type: str = "logistic_regression",
        model_parameters: dict[str, Any] | None = None,
        sample_weight_mode: str = "class_region_year_balanced",
        n_jobs: int = -1,
        random_seed: int = 42,
        training_samples_table_name: str = (
            "machine_learning/crop_classification/samples"
        ),
        training_samples_path: str | None = None,
        pooled_training_samples_path: str | None = None,
        reuse_pooled_training_samples: bool = False,
        model_output_path: str | None = None,
        overwrite_outputs: bool = True,
    ) -> None:
        """Pool Europe-model samples, evaluate one setting, and fit the final model.

        This is the second stage of the Europe-wide workflow. Only one selected model
        performs the pooled fit, controlled by ``training_model_id``; the other model
        builds return without duplicating the job. Samples from all selected models are
        loaded together, with local region IDs remapped to globally unique IDs before
        region-year balancing.

        The pooled sample table is written explicitly so the existing
        ``compare_alphaearth_cty_models.py``/``.sh`` workflow can compare estimator
        families and generic ``model_parameters`` mappings without rerunning sampling.
        The configured model is evaluated on complete held-out years and then refitted
        on every selected sample year for production inference.

        Args:
            europe_model_ids: Europe models whose model-local samples are pooled.
            training_model_id: Model build that performs training. Defaults to the first
                selected model ID.
            hrl_years: Observation years loaded from every selected model.
            validation_year: Optional complete year held out for validation.
            test_year: Optional complete year held out for blind testing.
            include_coordinates: Require and use coordinate predictors.
            include_topography: Require and use elevation and slope predictors.
            normalize_embeddings: L2-normalize dequantized embeddings.
            classifier_structure: ``"flat"`` or ``"hierarchical"``.
            residual_class_mode: ``"learned"`` or ``"uncertainty"``.
            calibrate_class_thresholds: Calibrate class-specific thresholds.
            class_probability_thresholds: Optional manual CTY thresholds.
            class_threshold_grid: Candidate thresholds for calibration.
            class_threshold_min_reference_samples: Minimum calibration support.
            unclassified_probability_threshold: Confidence rejection threshold.
            model_type: Estimator family supported by
                :func:`fit_alphaearth_crop_models`.
            model_parameters: Generic estimator-specific keyword mapping.
            sample_weight_mode: Sample weighting strategy.
            n_jobs: Parallel estimator workers where supported.
            random_seed: Reproducible estimator seed.
            training_samples_table_name: Model-local table name to load when pooling.
            training_samples_path: Optional existing Europe-wide or comparison-selected
                sample table. A relative path is resolved below the common model root.
            pooled_training_samples_path: Europe-wide Parquet output and optional
                reusable input. A relative path is resolved below the common model root.
            reuse_pooled_training_samples: Load the pooled table instead of reading and
                concatenating every model-local table. Ignored when
                ``training_samples_path`` is provided.
            model_output_path: Optional final Joblib path. A relative path is resolved
                below the common model root.
            overwrite_outputs: Replace pooled samples and model outputs.

        Raises:
            FileExistsError: If an output exists and overwrite is disabled.
            ValueError: If temporal splits or model settings are invalid.
        """
        selected_model_ids = parse_europe_model_ids(europe_model_ids)
        current_model_id, _, model_root = europe_model_build_context()
        if training_model_id is None:
            coordinator_id = selected_model_ids[0]
        else:
            coordinator_ids = parse_europe_model_ids(training_model_id)
            if len(coordinator_ids) != 1:
                raise ValueError(
                    "training_model_id must identify exactly one Europe model."
                )
            coordinator_id = coordinator_ids[0]
        if coordinator_id not in selected_model_ids:
            raise ValueError(
                "training_model_id must be included in europe_model_ids; found "
                f"Europe_{coordinator_id:03d}."
            )
        if current_model_id != coordinator_id:
            self.logger.info(
                "Skipping pooled AlphaEarth CTY training in Europe_%03d; "
                "Europe_%03d is the configured training model.",
                current_model_id,
                coordinator_id,
            )
            return

        hrl_years = tuple(int(year) for year in hrl_years)
        if len(hrl_years) < 3:
            raise ValueError("At least three HRL observation years are required.")
        if tuple(sorted(set(hrl_years))) != hrl_years:
            raise ValueError("hrl_years must be unique and strictly increasing.")
        for split_year, split_name in (
            (validation_year, "validation_year"),
            (test_year, "test_year"),
        ):
            if split_year is not None and int(split_year) not in hrl_years:
                raise ValueError(f"{split_name} must be one of {hrl_years}.")
        validation_year = None if validation_year is None else int(validation_year)
        test_year = None if test_year is None else int(test_year)
        if validation_year is not None and validation_year == test_year:
            raise ValueError("validation_year and test_year must differ.")
        if not 0.0 <= unclassified_probability_threshold <= 1.0:
            raise ValueError(
                "unclassified_probability_threshold must lie between zero and one."
            )
        if model_parameters is not None and not isinstance(model_parameters, dict):
            raise TypeError("model_parameters must be a dictionary or None.")
        if class_threshold_min_reference_samples < 1:
            raise ValueError(
                "class_threshold_min_reference_samples must be at least one."
            )

        shared_directory = (
            model_root / "machine_learning_models" / "alphaearth_crop_classification"
        )
        default_pooled_path = shared_directory / "samples.parquet"
        default_model_path = shared_directory / "alphaearth_cty_model.joblib"

        def resolve_shared_path(
            configured_path: str | None,
            default_path: Path,
        ) -> Path:
            path = default_path if configured_path is None else Path(configured_path)
            path = path.expanduser()
            return path if path.is_absolute() else model_root / path

        pooled_path = resolve_shared_path(
            pooled_training_samples_path,
            default_pooled_path,
        )
        final_model_path = resolve_shared_path(model_output_path, default_model_path)
        if final_model_path.exists() and not overwrite_outputs:
            raise FileExistsError(
                f"Model output exists and overwrite_outputs=False: {final_model_path}"
            )
        final_model_path.parent.mkdir(parents=True, exist_ok=True)

        reusable_source: Path | None = None
        if training_samples_path is not None:
            reusable_source = resolve_shared_path(
                training_samples_path,
                pooled_path,
            )
        elif reuse_pooled_training_samples:
            reusable_source = pooled_path

        if reusable_source is None:
            if pooled_path.exists() and not overwrite_outputs:
                raise FileExistsError(
                    "Pooled sample output exists and overwrite_outputs=False: "
                    f"{pooled_path}"
                )
            pooled_path.parent.mkdir(parents=True, exist_ok=True)
            samples, region_mapping = load_europe_alphaearth_crop_training_samples(
                model_root,
                selected_model_ids,
                training_samples_table_name=training_samples_table_name,
                hrl_years=hrl_years,
                include_coordinates=include_coordinates,
                include_topography=include_topography,
            )
            samples.drop(columns="split", errors="ignore").to_parquet(
                pooled_path,
                index=False,
            )
            sample_source_description = str(pooled_path)
        else:
            samples = load_alphaearth_crop_training_samples(
                reusable_source,
                hrl_years=hrl_years,
                include_coordinates=include_coordinates,
                include_topography=include_topography,
            )
            required_metadata = {
                "europe_model_id",
                "europe_model_name",
                "local_region_id",
            }
            missing_metadata = required_metadata - set(samples.columns)
            if missing_metadata:
                raise ValueError(
                    "Reusable Europe-wide samples are missing source metadata: "
                    f"{sorted(missing_metadata)}."
                )
            represented_model_ids = set(
                pd.to_numeric(samples["europe_model_id"], errors="raise")
                .astype(np.int64)
                .unique()
            )
            expected_model_ids = set(selected_model_ids)
            if represented_model_ids != expected_model_ids:
                raise ValueError(
                    "Reusable Europe-wide samples do not represent exactly the "
                    "selected models. Expected "
                    f"{sorted(expected_model_ids)}, found "
                    f"{sorted(represented_model_ids)}."
                )
            region_mapping = (
                samples[
                    [
                        "region_id",
                        "europe_model_id",
                        "europe_model_name",
                        "local_region_id",
                        "country_iso3",
                    ]
                ]
                .drop_duplicates()
                .sort_values("region_id")
                .reset_index(drop=True)
            )
            if (
                region_mapping["region_id"].duplicated().any()
                or region_mapping[["europe_model_id", "local_region_id"]]
                .duplicated()
                .any()
            ):
                raise ValueError(
                    "Reusable Europe-wide samples contain an ambiguous global-region "
                    "mapping."
                )
            sample_source_description = str(reusable_source)

        self.set_table(
            region_mapping,
            name="machine_learning/crop_classification/europe_region_mapping",
        )

        embedding_diagnostics = alphaearth_embedding_diagnostics(samples)
        embedding_diagnostics.update(
            {
                "sample_count": len(samples),
                "europe_model_count": len(selected_model_ids),
                "region_count": int(samples["region_id"].nunique()),
            }
        )
        self.set_table(
            pd.DataFrame([embedding_diagnostics]),
            name="machine_learning/crop_classification/embedding_diagnostics",
        )

        samples["split"] = "train"
        if validation_year is not None:
            samples.loc[samples["year"] == validation_year, "split"] = "validation"
        if test_year is not None:
            samples.loc[samples["year"] == test_year, "split"] = "test"
        training_samples = samples.loc[samples["split"] == "train"].copy()
        if training_samples.empty:
            raise ValueError(
                "The configured temporal split leaves no pooled training samples."
            )

        self.logger.info(
            "Loaded %s pooled CTY samples from %s Europe model(s), %s globally "
            "unique region(s), and years %s. Training source: %s.",
            f"{len(samples):,}",
            len(selected_model_ids),
            samples["region_id"].nunique(),
            tuple(sorted(int(year) for year in samples["year"].unique())),
            sample_source_description,
        )

        evaluation_models = fit_alphaearth_crop_models(
            training_samples,
            include_coordinates=include_coordinates,
            include_topography=include_topography,
            normalize_embeddings=normalize_embeddings,
            classifier_structure=classifier_structure,
            residual_class_mode=residual_class_mode,
            model_type=model_type,
            model_parameters=model_parameters,
            random_seed=random_seed,
            sample_weight_mode=sample_weight_mode,
            n_jobs=n_jobs,
            unclassified_probability_threshold=unclassified_probability_threshold,
            class_probability_thresholds=class_probability_thresholds,
            calibrate_class_thresholds=calibrate_class_thresholds,
            class_threshold_grid=class_threshold_grid,
            calibration_min_reference_samples=(class_threshold_min_reference_samples),
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
            confusion = pd.concat(confusion_tables, ignore_index=True)
            self.set_table(
                metrics,
                name="machine_learning/crop_classification/metrics",
            )
            self.set_table(
                confusion,
                name="machine_learning/crop_classification/confusion_matrix",
            )
            self.logger.info(
                "ACCURACY — POOLED EUROPE FIXED-HOLDOUT CLASSIFIER\n"
                "Training years=%s; validation=%s; test=%s.\n%s",
                tuple(sorted(int(year) for year in training_samples["year"].unique())),
                validation_year,
                test_year,
                format_alphaearth_accuracy_report(metrics, confusion),
            )

        final_models = fit_alphaearth_crop_models(
            samples,
            include_coordinates=include_coordinates,
            include_topography=include_topography,
            normalize_embeddings=normalize_embeddings,
            classifier_structure=classifier_structure,
            residual_class_mode=residual_class_mode,
            model_type=model_type,
            model_parameters=model_parameters,
            random_seed=random_seed,
            sample_weight_mode=sample_weight_mode,
            n_jobs=n_jobs,
            unclassified_probability_threshold=unclassified_probability_threshold,
            class_probability_thresholds=class_probability_thresholds,
            calibrate_class_thresholds=calibrate_class_thresholds,
            class_threshold_grid=class_threshold_grid,
            calibration_min_reference_samples=(class_threshold_min_reference_samples),
        )
        feature_importance = alphaearth_crop_feature_importance(final_models)
        if not feature_importance.empty:
            self.set_table(
                feature_importance,
                name="machine_learning/crop_classification/feature_importance",
            )
        if final_models.class_probability_thresholds:
            threshold_table = pd.DataFrame(
                {
                    "cty_class": sorted(final_models.class_probability_thresholds),
                    "probability_threshold": [
                        final_models.class_probability_thresholds[class_code]
                        for class_code in sorted(
                            final_models.class_probability_thresholds
                        )
                    ],
                }
            )
            self.set_table(
                threshold_table,
                name=(
                    "machine_learning/crop_classification/class_probability_thresholds"
                ),
            )

        saved_model_path = save_alphaearth_crop_models(
            final_models,
            final_model_path,
        )
        self.logger.info(
            "Saved pooled Europe AlphaEarth CTY model to %s. The estimator is %s "
            "with model_parameters=%s.",
            saved_model_path,
            final_models.model_type,
            final_models.model_parameters,
        )

    @build_method(
        depends_on=["setup_regions_and_land_use"],
        required=False,
    )
    def setup_classify_alphaearth_crop_types(
        self,
        europe_model_ids: tuple[int | str, ...] = tuple(range(12)),
        hrl_years: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023),
        prediction_years: tuple[int, ...] = (2024, 2025),
        template_year: int | None = None,
        model_path: str | None = None,
        prediction_chunk_size: int = 512,
        apply_historical_cropland_mask: bool = False,
        cropland_mask_years: tuple[int, ...] | None = None,
        historical_cropland_mask_dilation_pixels: int = 0,
        smooth_cty_probabilities: bool = True,
        unclassified_probability_threshold: float | None = None,
        write_cty_confidence: bool = True,
        cty_confidence_output_root: str | None = None,
        apply_permanent_crop_temporal_consistency: bool = False,
        apply_cty_mmu_sieve: bool = True,
        cty_mmu_minimum_pixels: int = 25,
        cty_mmu_connectivity: int = 4,
        cty_mmu_padding_pixels: int = 25,
        cty_mmu_maximum_iterations: int = 3,
        max_alphaearth_files_for_prediction: int | None = None,
        alphaearth_max_parallel_downloads: int | None = None,
        cleanup_alphaearth_downloads: bool = True,
        overwrite_prediction_files: bool = True,
    ) -> None:
        """Classify AlphaEarth years and write post-processed HRL-compatible CTY tiles.

        This is the third stage of the Europe-wide workflow. Each selected
        ``Europe_###`` model loads the same pooled model bundle, downloads AlphaEarth
        COGs once for its complete active extent, predicts its native HRL 100-km tile
        grids, applies the configured historical mask, probability smoothing,
        permanent-crop consistency and minimum-mapping-unit sieve, and writes the
        results into the standard HRL catalog directories.

        Args:
            europe_model_ids: Europe models that should classify their local extent.
            hrl_years: Historical HRL years used for templates and post-processing.
            prediction_years: Ordered AlphaEarth years to classify.
            template_year: HRL year defining output grids and filenames.
            model_path: Pooled Joblib model. A relative path is resolved below the
                common model root; the shared default from the training method is used
                when omitted.
            prediction_chunk_size: Prediction-window width and height.
            apply_historical_cropland_mask: Restrict predictions to historical cropland.
            cropland_mask_years: Historical years used to construct the mask.
            historical_cropland_mask_dilation_pixels: Optional mask dilation.
            smooth_cty_probabilities: Apply 3x3 probability smoothing.
            unclassified_probability_threshold: Optional override of the threshold
                stored in the model bundle.
            write_cty_confidence: Write uint8 CTY-confidence tiles.
            cty_confidence_output_root: Optional confidence-product root.
            apply_permanent_crop_temporal_consistency: Apply permanent-crop rules.
            apply_cty_mmu_sieve: Apply the CTY minimum-mapping-unit sieve.
            cty_mmu_minimum_pixels: Minimum retained CTY patch size.
            cty_mmu_connectivity: Four- or eight-neighbour connectivity.
            cty_mmu_padding_pixels: Neighbour-tile padding during sieving.
            cty_mmu_maximum_iterations: Maximum sieve passes.
            max_alphaearth_files_for_prediction: Optional model-level download limit.
            alphaearth_max_parallel_downloads: Optional parallel-download override.
            cleanup_alphaearth_downloads: Remove prediction COGs after use. Defaults
                to ``True``.
            overwrite_prediction_files: Replace existing generated CTY tiles.

        Raises:
            ValueError: If years, templates, settings, or model features are invalid.
        """
        selected_model_ids = parse_europe_model_ids(europe_model_ids)
        current_model_id, _, model_root = europe_model_build_context()
        current_model_name = f"Europe_{current_model_id:03d}"
        if current_model_id not in selected_model_ids:
            self.logger.info(
                "Skipping AlphaEarth CTY classification for %s; selected models "
                "are %s.",
                current_model_name,
                [f"Europe_{model_id:03d}" for model_id in selected_model_ids],
            )
            return

        hrl_years = tuple(int(year) for year in hrl_years)
        if not hrl_years:
            raise ValueError("hrl_years must contain at least one year.")
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
        overlap = set(prediction_years).intersection(hrl_years)
        if overlap:
            raise ValueError(
                "prediction_years must not overlap observed HRL years. "
                f"Overlap: {sorted(overlap)}."
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
        unknown_mask_years = set(cropland_mask_years) - set(hrl_years)
        if unknown_mask_years:
            raise ValueError(
                "cropland_mask_years must be selected from hrl_years. "
                f"Unknown years: {sorted(unknown_mask_years)}."
            )
        if historical_cropland_mask_dilation_pixels < 0:
            raise ValueError(
                "historical_cropland_mask_dilation_pixels cannot be negative."
            )
        if prediction_chunk_size < 16:
            raise ValueError("prediction_chunk_size must be at least 16.")
        if cty_mmu_minimum_pixels < 1:
            raise ValueError("cty_mmu_minimum_pixels must be at least one.")
        if cty_mmu_connectivity not in {4, 8}:
            raise ValueError("cty_mmu_connectivity must be 4 or 8.")
        if cty_mmu_padding_pixels < 0:
            raise ValueError("cty_mmu_padding_pixels cannot be negative.")
        if cty_mmu_maximum_iterations < 1:
            raise ValueError("cty_mmu_maximum_iterations must be at least one.")
        if (
            alphaearth_max_parallel_downloads is not None
            and alphaearth_max_parallel_downloads < 1
        ):
            raise ValueError("alphaearth_max_parallel_downloads must be at least 1.")

        default_model_path = (
            model_root
            / "machine_learning_models"
            / "alphaearth_crop_classification"
            / "alphaearth_cty_model.joblib"
        )
        resolved_model_path = (
            default_model_path if model_path is None else Path(model_path).expanduser()
        )
        if not resolved_model_path.is_absolute():
            resolved_model_path = model_root / resolved_model_path
        final_models = load_alphaearth_crop_models(resolved_model_path)

        threshold = (
            final_models.unclassified_probability_threshold
            if unclassified_probability_threshold is None
            else float(unclassified_probability_threshold)
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "unclassified_probability_threshold must lie between zero and one."
            )

        topographic_feature_names = {"elevation_m", "slope_gradient"}
        represented_topographic_features = topographic_feature_names.intersection(
            final_models.feature_names
        )
        if represented_topographic_features and represented_topographic_features != (
            topographic_feature_names
        ):
            raise ValueError(
                "The saved AlphaEarth model contains an incomplete topographic "
                f"feature schema: {sorted(represented_topographic_features)}."
            )
        include_topography = (
            represented_topographic_features == topographic_feature_names
        )

        region_ids: xr.DataArray = self.subgrid["region_ids"].compute()
        subgrid_mask: xr.DataArray = self.subgrid["mask"].compute()
        active_subgrid_mask = ~subgrid_mask.values
        active_geometry = _active_subgrid_mask_geometry_for_hrl(
            region_ids,
            active_subgrid_mask,
        )
        study_bounds = tuple(float(value) for value in active_geometry.bounds)

        subgrid_elevation: xr.DataArray | None = None
        subgrid_slope: xr.DataArray | None = None
        if include_topography:
            subgrid_elevation = self.subgrid["landsurface/elevation"].compute()
            if subgrid_elevation.rio.crs is None:
                raise ValueError(
                    "Subgrid elevation must have a CRS for the saved model schema."
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

        alphaearth_adapter = self.data_catalog.fetch("alphaearth")
        if alphaearth_max_parallel_downloads is not None:
            alphaearth_adapter.max_parallel_downloads = int(
                alphaearth_max_parallel_downloads
            )

        template_fetch = self.data_catalog.fetch(
            f"hrl_crop_types_{template_year}",
            bounds=study_bounds,
            year=template_year,
        )
        template_cty_tile_ids = {
            str(tile_id) for tile_id in getattr(template_fetch, "tile_ids", ())
        }
        if not template_cty_tile_ids:
            raise ValueError(
                f"No {template_year} HRL CTY template tiles intersect "
                f"{current_model_name}."
            )

        cty_template_adapter = self.data_catalog.catalog[
            f"hrl_crop_types_{template_year}"
        ]["adapter"]
        cty_templates_by_code = {
            hrl_tile_code_from_name(tile_id): find_hrl_tile_path(
                cty_template_adapter.root,
                year=template_year,
                tile_id=tile_id,
            )
            for tile_id in template_cty_tile_ids
        }

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

        generated_cty_tile_ids: dict[int, list[str]] = {
            year: [] for year in prediction_years
        }
        generated_cty_paths: dict[int, dict[str, Path]] = {
            year: {} for year in prediction_years
        }
        generated_confidence_paths: dict[int, dict[str, Path]] = {
            year: {} for year in prediction_years
        }
        prediction_tasks: list[
            tuple[int, str, Path, Path, Path | None, BaseGeometry]
        ] = []

        for prediction_year in prediction_years:
            cty_output_directory = Path(cty_template_adapter.root) / str(
                prediction_year
            )
            confidence_output_directory = confidence_root / str(prediction_year)
            cty_output_directory.mkdir(parents=True, exist_ok=True)
            if write_cty_confidence:
                confidence_output_directory.mkdir(parents=True, exist_ok=True)

            for tile_code in sorted(cty_templates_by_code):
                cty_template_path = cty_templates_by_code[tile_code]
                cty_output_path = cty_output_directory / build_hrl_prediction_tile_name(
                    cty_template_path.name,
                    product_code="CTY",
                    prediction_year=prediction_year,
                )
                confidence_output_path = (
                    confidence_output_directory
                    / build_hrl_prediction_tile_name(
                        cty_template_path.name,
                        product_code="CTYCL",
                        prediction_year=prediction_year,
                    )
                    if write_cty_confidence
                    else None
                )

                required_outputs_exist = cty_output_path.exists() and (
                    confidence_output_path is None or confidence_output_path.exists()
                )
                if not overwrite_prediction_files and required_outputs_exist:
                    generated_cty_tile_ids[prediction_year].append(cty_output_path.stem)
                    generated_cty_paths[prediction_year][tile_code] = cty_output_path
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
                prediction_geometry = box(*tile_bounds_wgs84).intersection(
                    active_geometry
                )
                if prediction_geometry.is_empty:
                    continue
                prediction_tasks.append(
                    (
                        prediction_year,
                        tile_code,
                        cty_template_path,
                        cty_output_path,
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
                    "Downloading AlphaEarth COGs for years %s and the complete %s "
                    "extent %s before %s tile predictions.",
                    task_prediction_years,
                    current_model_name,
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
                        f"{task_prediction_years} in {current_model_name}."
                    )

                cropland_year_positions = [
                    hrl_years.index(year) for year in cropland_mask_years
                ]
                for (
                    prediction_year,
                    tile_code,
                    cty_template_path,
                    cty_output_path,
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
                            "No downloaded AlphaEarth COGs intersect HRL tile "
                            f"{tile_code}, year {prediction_year}."
                        )

                    historical_mask_paths = tuple(
                        historical_cty_paths_by_code[tile_code][position]
                        for position in cropland_year_positions
                    )
                    predict_alphaearth_crop_tile_to_hrl_geotiffs(
                        final_models,
                        cty_template_path,
                        tile_alphaearth_cogs,
                        prediction_geometry,
                        cty_output_path,
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
                        unclassified_probability_threshold=threshold,
                        cty_confidence_output_path=confidence_output_path,
                    )
                    generated_cty_tile_ids[prediction_year].append(cty_output_path.stem)
                    generated_cty_paths[prediction_year][tile_code] = cty_output_path
                    if confidence_output_path is not None:
                        generated_confidence_paths[prediction_year][tile_code] = (
                            confidence_output_path
                        )
                    self.logger.info(
                        "Wrote HRL-compatible CTY prediction for %s, tile %s, year "
                        "%s from %s cached COG(s).",
                        current_model_name,
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
                        "Removed %s AlphaEarth prediction COGs for %s.",
                        removed,
                        current_model_name,
                    )
                elif selected_prediction_cogs is not None:
                    self.logger.info(
                        "Retaining %s AlphaEarth prediction COGs for %s.",
                        len(selected_prediction_cogs),
                        current_model_name,
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
                        "Applied %s-pixel CTY MMU sieve to %s tile(s) for %s in "
                        "%s; %s pixels changed.",
                        cty_mmu_minimum_pixels,
                        len(sieve_results),
                        prediction_year,
                        current_model_name,
                        int(sieve_results["changed_pixels"].sum()),
                    )

        if postprocessing_rows:
            self.set_table(
                pd.DataFrame(postprocessing_rows),
                name="machine_learning/crop_classification/postprocessing",
            )

        for prediction_year in prediction_years:
            year_cty_ids = generated_cty_tile_ids[prediction_year]
            if not year_cty_ids:
                raise ValueError(
                    f"No HRL-compatible CTY tiles were generated for "
                    f"{current_model_name}, {prediction_year}."
                )
            catalog_name = f"hrl_crop_types_{prediction_year}"
            if catalog_name in self.data_catalog.catalog:
                self.data_catalog.catalog[catalog_name][
                    "adapter"
                ].tile_ids = year_cty_ids
            self.logger.info(
                "Finished AlphaEarth CTY classification for %s, year %s. Generated "
                "%s standard HRL tiles using model %s.",
                current_model_name,
                prediction_year,
                len(year_cty_ids),
                resolved_model_path,
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
            crop_area_fit_warning_threshold_pct: CTY crop-area fit below which a
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
        workflow_settings: _LowderSequenceSettings,
        crop_area_diagnostics_top_n: int,
        crop_area_fit_warning_threshold_pct: float,
    ) -> None:
        """Create Lowder-guided farms from multi-year HRL CTY observations.

        The workflow loads native HRL main-crop areas, reprojects annual CTY states
        in destination-grid tiles, constructs Lowder target sizes, assigns complete
        observed main-crop sequences, registers outputs, and produces diagnostics.

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
            workflow_settings: Settings for the Lowder sequence-balanced workflow.
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

        lowder_sequence_settings = workflow_settings
        score_weight_sum = (
            lowder_sequence_settings.distance_weight
            + lowder_sequence_settings.crop_sequence_weight
            + lowder_sequence_settings.switch_timing_weight
        )
        if score_weight_sum <= 0.0:
            raise ValueError("Farm-growth score weights must sum to a positive value.")
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

        distance_weight = lowder_sequence_settings.distance_weight
        crop_sequence_weight = lowder_sequence_settings.crop_sequence_weight
        switch_timing_weight = lowder_sequence_settings.switch_timing_weight
        min_valid_crop_sequence_overlap = (
            lowder_sequence_settings.min_valid_crop_sequence_overlap
        )
        jump_candidate_sample = lowder_sequence_settings.jump_candidate_sample
        max_jump_distance_m = lowder_sequence_settings.max_jump_distance_m
        crop_area_alignment_weight = lowder_sequence_settings.crop_area_alignment_weight
        max_crop_candidates_per_farmer = lowder_sequence_settings.max_local_sequences
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

        static_selection_name = "multiyear_coverage"
        self.logger.info(
            "Starting HRL-only raster farm construction for %s model regions "
            "(reprojection=tiled; static selection=%s; workflow=%s).",
            len(regions_shapes_hrl),
            static_selection_name,
            "lowder_sequence_balanced",
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

            crop_per_year: list[np.ndarray] = []
            cultivated_fraction_per_year: list[np.ndarray] = []
            hrl_coverage_fraction_per_year: list[np.ndarray] = []
            native_crop_areas_per_year: list[dict[int, float]] = []
            region_has_hrl_coverage = True

            for year in years:
                crop_types = None
                crop_types_adapter = None
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
                except WEkEONoCoverageError as error:
                    if original_iso3.upper() in _HRL_CROPLANDS_EEA38_ISO3:
                        raise
                    self.logger.warning(
                        "Skipping region %s (%s): no HRL Crop Types coverage for "
                        "year %s. Original error: %s",
                        region_id,
                        original_iso3,
                        year,
                        error,
                    )
                    region_has_hrl_coverage = False
                    break

                native_crop_areas_per_year.append(
                    _native_hrl_crop_category_areas_m2(
                        crop_types,
                        region_active_geometry,
                        chunk_rows=max(int(raster_chunks.get("y", 4096)), 1),
                    )
                )

                crop_year = np.full(
                    region_template.shape,
                    _HRL_MISSING_CROP_CODE,
                    dtype=np.int32,
                )
                cultivated_fraction_year = np.zeros(
                    region_template.shape, dtype=np.float32
                )
                coverage_fraction_year = np.zeros(
                    region_template.shape, dtype=np.float32
                )
                source_bounds = crop_types.rio.bounds()
                source_resolution = crop_types.rio.resolution()
                source_buffer_x = abs(float(source_resolution[0])) * 2.0
                source_buffer_y = abs(float(source_resolution[1])) * 2.0

                for tile_y_start in range(
                    0, region_template.sizes["y"], subgrid_chunk_size
                ):
                    tile_y_stop = min(
                        tile_y_start + subgrid_chunk_size, region_template.sizes["y"]
                    )
                    for tile_x_start in range(
                        0, region_template.sizes["x"], subgrid_chunk_size
                    ):
                        tile_x_stop = min(
                            tile_x_start + subgrid_chunk_size,
                            region_template.sizes["x"],
                        )
                        tile_y_slice = slice(tile_y_start, tile_y_stop)
                        tile_x_slice = slice(tile_x_start, tile_x_stop)
                        tile_region_mask = region_mask[tile_y_slice, tile_x_slice]
                        if not tile_region_mask.any():
                            continue

                        tile_template = region_template.isel(
                            y=tile_y_slice, x=tile_x_slice
                        )
                        tile_bounds = transform_bounds(
                            tile_template.rio.crs,
                            crop_types.rio.crs,
                            *tile_template.rio.bounds(),
                            densify_pts=21,
                        )
                        clip_min_x = max(
                            tile_bounds[0] - source_buffer_x, source_bounds[0]
                        )
                        clip_min_y = max(
                            tile_bounds[1] - source_buffer_y, source_bounds[1]
                        )
                        clip_max_x = min(
                            tile_bounds[2] + source_buffer_x, source_bounds[2]
                        )
                        clip_max_y = min(
                            tile_bounds[3] + source_buffer_y, source_bounds[3]
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
                        tile_crop, tile_fraction, tile_coverage_fraction = (
                            _reproject_HRL_year_to_subgrid(
                                crop_types_tile, tile_template
                            )
                        )
                        tile_crop[~tile_region_mask] = _HRL_MISSING_CROP_CODE
                        tile_fraction[~tile_region_mask] = 0.0
                        tile_coverage_fraction[~tile_region_mask] = 0.0
                        crop_year[tile_y_slice, tile_x_slice] = tile_crop
                        cultivated_fraction_year[tile_y_slice, tile_x_slice] = (
                            tile_fraction
                        )
                        coverage_fraction_year[tile_y_slice, tile_x_slice] = (
                            tile_coverage_fraction
                        )
                        del (
                            crop_types_tile,
                            tile_crop,
                            tile_fraction,
                            tile_coverage_fraction,
                        )

                crop_per_year.append(crop_year)
                cultivated_fraction_per_year.append(cultivated_fraction_year)
                hrl_coverage_fraction_per_year.append(coverage_fraction_year)
                del (
                    crop_types,
                    crop_types_adapter,
                    crop_year,
                    cultivated_fraction_year,
                    coverage_fraction_year,
                )

            if not region_has_hrl_coverage:
                continue
            if (
                len(crop_per_year) != len(years)
                or len(cultivated_fraction_per_year) != len(years)
                or len(hrl_coverage_fraction_per_year) != len(years)
                or len(native_crop_areas_per_year) != len(years)
            ):
                raise ValueError(f"Incomplete HRL crop stack for region {region_id}.")

            crop_stack = np.stack(crop_per_year).astype(
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
                crop_per_year,
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

            reference_index = years.index(max(years))
            reference_fraction = fraction_stack[reference_index].astype(
                np.float64, copy=False
            )
            base_static_target_area_m2 = float(
                np.sum(
                    reference_fraction[region_mask] * region_cell_area_m2[region_mask]
                )
            )
            selection_target_area_m2 = max(
                base_static_target_area_m2,
                float(native_hrl_area_by_year_m2.max(initial=0.0)),
            )

            union_valid_crop = np.any(crop_stack > 0, axis=0)
            eligible_mask = region_mask & union_valid_crop
            valid_frequency = np.mean(crop_stack > 0, axis=0)
            mean_fraction = fraction_stack.mean(axis=0, dtype=np.float64)
            selection_score = 0.80 * valid_frequency + 0.20 * mean_fraction

            if not eligible_mask.any():
                self.logger.warning(
                    "Skipping region %s because it has no model cells with an "
                    "observed HRL CTY crop in any requested year.",
                    region_id,
                )
                continue

            available_static_capacity_m2 = float(
                region_cell_area_m2[eligible_mask].sum()
            )
            if selection_target_area_m2 > available_static_capacity_m2:
                self.logger.warning(
                    "Region %s has only %.3f km² of cells with at least one "
                    "valid modal CTY crop, below the requested static capacity "
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
            if lowder_extra_farm_fraction > 0.0:
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
                max_regional_sequences=max_regional_sequence_candidates_per_farmer,
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
                        "Sequence-balanced assignment produced a missing farmer "
                        f"crop in region {region_id}, year {year}."
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

                cty_fit = _crop_area_fit_scores(crop_alignment)
                positive_target_scale = float(
                    crop_alignment["positive_target_scale"].iloc[0]
                )

                crop_alignment_summary_by_year.append(
                    {
                        "year": int(year),
                        "source_crop_area_m2": cty_fit["source_area_m2"],
                        "fractional_subgrid_area_m2": float(
                            subgrid_hrl_area_by_year_m2[year_index]
                        ),
                        "selected_fractional_area_m2": float(
                            selected_fractional_area_by_year_m2[year_index]
                        ),
                        "selected_modal_area_m2": float(
                            selected_modal_area_by_year_m2[year_index]
                        ),
                        "assigned_crop_area_m2": cty_fit["assigned_area_m2"],
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
                        "total_area_difference_pct": cty_fit[
                            "total_area_difference_pct"
                        ],
                        "total_area_fit_score": cty_fit["total_area_fit_score"],
                        "cty_crop_area_fit_score": cty_fit["crop_area_fit_score"],
                        "cty_crop_share_fit_score": cty_fit["crop_share_fit_score"],
                        "positive_target_scale": positive_target_scale,
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
                selected_sequences == _HRL_MISSING_CROP_CODE, axis=1
            ) & np.any(selected_sequences > 0, axis=1)
            n_observed_sequences = int(
                np.unique(selected_sequences[complete_original_mask], axis=0).shape[0]
            )
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

            novel_sequence_count = 0

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
                    "n_observed_sequences": n_observed_sequences,
                    "extra_sequence_farms": int(extra_sequence_farms),
                    "novel_sequence_count": novel_sequence_count,
                    "local_dominant_sequence_pct": (local_dominant_sequence_pct),
                    "local_sequence_pct": local_sequence_pct,
                    "regional_fallback_sequence_pct": (regional_fallback_sequence_pct),
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
                    "mean_cty_crop_fit_score": float(
                        np.mean(
                            [
                                summary["cty_crop_area_fit_score"]
                                for summary in crop_alignment_summary_by_year
                            ]
                        )
                    ),
                    "minimum_cty_crop_fit_score": float(
                        np.min(
                            [
                                summary["cty_crop_area_fit_score"]
                                for summary in crop_alignment_summary_by_year
                            ]
                        )
                    ),
                    "mean_crop_share_fit_score": float(
                        np.mean(
                            [
                                summary["cty_crop_share_fit_score"]
                                for summary in crop_alignment_summary_by_year
                            ]
                        )
                    ),
                }
            )

            mean_cty_fit = float(
                np.mean(
                    [
                        summary["cty_crop_area_fit_score"]
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
                if mean_cty_fit < crop_area_fit_warning_threshold_pct
                else self.logger.info
            )
            log_method(
                "Completed Lowder-prioritized region %s (%s): %.2f km² static "
                "area; %s farms from %s Lowder targets; mean annual "
                "active/native area %.1f%%; mean CTY crop fit %.1f/100.",
                region_id,
                original_iso3,
                selected_area_m2 / 1_000_000.0,
                len(farmers_region),
                lowder_target_farm_count,
                mean_agent_retention,
                mean_cty_fit,
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
            "n_observed_sequences",
            "extra_sequence_farms",
            "local_dominant_sequence_pct",
            "local_sequence_pct",
            "regional_fallback_sequence_pct",
            "novel_sequence_count",
            "median_farm_area_ha",
            "multi_parcel_farms_pct",
            "mean_fallow_area_pct",
            "mean_missing_area_pct",
            "mean_selected_fraction_retention_pct",
            "mean_agent_area_retention_pct",
            "maximum_absolute_agent_area_difference_pct",
            "mean_cty_crop_fit_score",
        ]
        regional_summary = regional_diagnostics[regional_summary_columns].rename(
            columns={
                "lowder_source_iso3": "Lowder",
                "model_area_km2": "static_km2",
                "n_farmers": "agents",
                "lowder_target_farms": "Lowder_n",
                "sequence_fit_target_farms": "sequence_target_n",
                "n_observed_sequences": "sequences",
                "extra_sequence_farms": "extra_n",
                "local_dominant_sequence_pct": "local_dominant_pct",
                "local_sequence_pct": "local_total_pct",
                "regional_fallback_sequence_pct": "regional_fallback_pct",
                "novel_sequence_count": "novel_sequences",
                "median_farm_area_ha": "median_ha",
                "multi_parcel_farms_pct": "multi_parcel_pct",
                "mean_fallow_area_pct": "fallow_area_pct",
                "mean_missing_area_pct": "missing_area_pct",
                "mean_selected_fraction_retention_pct": "fraction_in_union_pct",
                "mean_agent_area_retention_pct": "agent_native_pct",
                "maximum_absolute_agent_area_difference_pct": "worst_area_diff_pct",
                "mean_cty_crop_fit_score": "cty_fit",
            }
        )
        self.logger.info(
            "HRL-only regional comparison summary:\n%s",
            regional_summary.round(
                {
                    "static_km2": 2,
                    "median_ha": 2,
                    "multi_parcel_pct": 1,
                    "fallow_area_pct": 1,
                    "missing_area_pct": 2,
                    "fraction_in_union_pct": 1,
                    "agent_native_pct": 1,
                    "worst_area_diff_pct": 1,
                    "cty_fit": 1,
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

            cty_fit = _crop_area_fit_scores(overall_year)
            subgrid_total_m2 = total_subgrid_hrl_area_by_year_m2[year]
            native_to_subgrid_pct = (
                (subgrid_total_m2 - cty_fit["source_area_m2"])
                / cty_fit["source_area_m2"]
                * 100.0
                if cty_fit["source_area_m2"] > 0.0
                else np.nan
            )
            if cty_fit["crop_area_fit_score"] < crop_area_fit_warning_threshold_pct:
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
                    "native_active_km2": (cty_fit["source_area_m2"] / 1_000_000.0),
                    "subgrid_active_km2": subgrid_total_m2 / 1_000_000.0,
                    "fraction_in_union_km2": (
                        selected_fractional_total_m2 / 1_000_000.0
                    ),
                    "agent_active_km2": (cty_fit["assigned_area_m2"] / 1_000_000.0),
                    "fallow_km2": selected_fallow_total_m2 / 1_000_000.0,
                    "missing_km2": selected_missing_total_m2 / 1_000_000.0,
                    "agricultural_union_km2": (
                        agricultural_union_total_m2 / 1_000_000.0
                    ),
                    "native_subgrid_diff_pct": native_to_subgrid_pct,
                    "fraction_in_union_pct": selected_fraction_retention_pct,
                    "binary_vs_fraction_pct": modal_conversion_pct,
                    "active_native_diff_pct": cty_fit["total_area_difference_pct"],
                    "cty_fit": cty_fit["crop_area_fit_score"],
                    "share_fit": cty_fit["crop_share_fit_score"],
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
                    "cty_fit": 1,
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
                    "crop_level": "CTY",
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
                }
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
            "CTY crop. pair_fit_pct is min(raw, final) / max(raw, final); "
            "area_weight_pct is the pair's share of the total comparison area:\n%s",
            _format_multi_year_crop_area_comparison(multi_year_crop_comparison),
        )

        if low_fit_years:
            self.logger.warning(
                "CTY crop-area fit is below %.1f/100 for years %s.",
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
        random_seed: int = 42,
    ) -> None:
        """Build farmer crop calendars from HRL CTY crops and MIRCA-OS calendars.

        The final compact farmer table from
        the HRL raster farm-construction method determines the main-crop sequence
        assigned to each farmer. HRL CTY classes are mapped to MIRCA crop classes
        because crop-growth parametrization is available for MIRCA crops. MIRCA-OS
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
            replace_crop_calendar_unit_code: Optional mapping to replace MIRCA-OS
                unit codes when a unit has missing or unsuitable crop calendars.
            multiple_years: If True, build crop calendars for all years in
                ``hrl_years`` and accumulate irrigation adaptations only for farmers
                with missing crop histories in previous years.
            hrl_years: HRL years processed when ``multiple_years`` is True.
            reduce_crops: Replace rice by a different crop in region 4.
            random_seed: Base seed for reproducible area-weighted MIRCA-OS calendar
                selection.

        Raises:
            ValueError: If required final farmer crop-table columns are missing.
            ValueError: If ``multiple_years`` is True and ``hrl_years`` is empty.
            ValueError: If farmers cannot be assigned to valid MIRCA-OS units.
            ValueError: If no MIRCA-OS calendar can be found for an assigned crop.
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

        # Use MIRCA-OS for both calendar timing and irrigation-area fractions,
        # matching the standard farmer crop-calendar setup workflow. The cropping-area
        # raster is loaded only as a spatial reference for the MIRCA unit grid.
        reference_crop_map = self.data_catalog.fetch(
            f"mirca_os_cropping_area_{mirca_year}_5-arcminute_Wheat_rf"
        ).read()
        reference_map_buffer = 100
        reference_crop_map = reference_crop_map.isel(
            get_window(
                reference_crop_map.x,
                reference_crop_map.y,
                self.bounds,
                buffer=reference_map_buffer,
                raise_on_buffer_out_of_bounds=False,
            )
        )

        mirca_unit_geom = self.data_catalog.fetch(
            f"mirca_os_admin_boundaries_{mirca_year}"
        ).read()
        if not isinstance(mirca_unit_geom, gpd.GeoDataFrame):
            raise TypeError(
                "MIRCA-OS administrative boundaries must be a GeoDataFrame."
            )

        mirca_unit_geom = mirca_unit_geom.cx[
            reference_crop_map.x.values.min() : reference_crop_map.x.values.max(),
            reference_crop_map.y.values.min() : reference_crop_map.y.values.max(),
        ]
        if mirca_unit_geom.empty:
            raise ValueError("No MIRCA-OS units overlap the model bounds.")

        rainfed_calendar_source = self.data_catalog.fetch(
            f"mirca_os_crop_calendar_{mirca_year}_rf"
        ).read()
        irrigated_calendar_source = self.data_catalog.fetch(
            f"mirca_os_crop_calendar_{mirca_year}_ir"
        ).read()

        mirca_units = mirca_unit_geom["unit_code"].astype(np.int64).tolist()
        rainfed_calendar_source = rainfed_calendar_source.loc[
            rainfed_calendar_source["unit_code"].isin(mirca_units)
        ]
        irrigated_calendar_source = irrigated_calendar_source.loc[
            irrigated_calendar_source["unit_code"].isin(mirca_units)
        ]

        crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]] = {}
        crop_calendar = parse_MIRCA_crop_calendar(
            crop_calendar,
            rainfed_calendar_source,
            mirca_units,
            is_irrigated=False,
        )
        crop_calendar = parse_MIRCA_crop_calendar(
            crop_calendar,
            irrigated_calendar_source,
            mirca_units,
            is_irrigated=True,
        )
        crop_calendar = _fix_365_in_crop_calendar(crop_calendar)
        if not crop_calendar:
            raise ValueError("No MIRCA-OS crop calendars overlap the model bounds.")

        mirca_unit_grid = rasterize_like(
            mirca_unit_geom,
            reference_crop_map,
            dtype=np.int32,
            nodata=-1,
            column="unit_code",
            name="MIRCA_unit",
        )
        mirca_unit_grid.values = fillna_2d(mirca_unit_grid.values, nodata=-1)
        farmer_mirca_units = sample_from_map(
            mirca_unit_grid.values,
            farmer_locations,
            mirca_unit_grid.rio.transform(recalc=True).to_gdal(),
        ).astype(np.int32)

        if (farmer_mirca_units == -1).any():
            raise ValueError("All farmers should be assigned to a valid MIRCA-OS unit.")

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

        # Cache each farmer's area-weighted MIRCA-OS calendar selection so that an
        # unchanged main crop and irrigation state remains stable across HRL years.
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

            # Only this part is HRL-year specific. MIRCA-OS calendar parsing and
            # spatial sampling are reused across all requested years.
            farmer_crops = _decode_hrl_crops_from_farmer_table(
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
            ) = _select_mirca_calendars_for_farmers(
                crop_calendar,
                farmer_mirca_units=farmer_mirca_units,
                farmer_main_crops=farmer_main_crops,
                farmer_is_irrigated=is_irrigated_for_calendar,
                replace_crop_calendar_unit_code=replace_crop_calendar_unit_code,
                selection_cache=calendar_selection_cache,
                random_seed=random_seed,
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
