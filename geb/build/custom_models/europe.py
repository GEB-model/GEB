"""Class to set GEB up for Europe."""

import gc
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio import features
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
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
from geb.build.data_catalog.copernicus_hrl import (
    CDSENoCoverageError,
)
from geb.build.data_catalog.wekeo_copernicus import (
    WEkEONoCoverageError,
)
from geb.build.methods import build_method
from geb.build.workflows.crop_calendars import (
    MIRCA_OS_CROP_CLASS_MAP,
    parse_MIRCA_crop_calendar,
)
from geb.build.workflows.farmers import (
    assign_farmer_sequences_to_area_targets,
    create_lowder_target_farm_areas,
    farm_size_distribution_fit_by_size_class,
    get_farm_locations,
    grow_farms_from_raster_cells,
    ensure_complete_sequence_in_selected_cells,
    raster_cell_area_m2,
    relax_lowder_targets_for_sequence_fit,
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
_HRL_GDAL_ALTERED_OUTSIDE_AREA_CODE = 65534
# Temporary destination sentinels used only inside GDAL warp calls. They are
# deliberately different from both the source missing code (-2) and every valid
# HRL category so GDAL >=3.11 never needs its nodata-collision replacement logic.
_HRL_WARP_DST_NODATA = -3
_HRL_FRACTION_WARP_DST_NODATA = -1.0

_HRL_NO_COVERAGE_ERRORS = (
    WEkEONoCoverageError,
    CDSENoCoverageError,
)


def _process_memory_mib() -> tuple[float | None, float | None]:
    """Return current RSS and Linux high-water RSS in MiB when available."""
    rss_mib: float | None = None
    hwm_mib: float | None = None
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    rss_mib = float(line.split()[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    hwm_mib = float(line.split()[1]) / 1024.0
                if rss_mib is not None and hwm_mib is not None:
                    break
    except OSError, ValueError, IndexError:
        return None, None
    return rss_mib, hwm_mib


def _record_phase_timing(
    timings: dict[str, float], phase: str, started: float
) -> float:
    """Accumulate elapsed wall time for one deterministic workflow phase."""
    elapsed = time.perf_counter() - started
    timings[phase] = timings.get(phase, 0.0) + elapsed
    return elapsed


def _validated_HRL_crop_values(values: np.ndarray) -> np.ndarray:
    """Return normalized int32 CTY values using the documented Croplands codes.

    The HRL Croplands PUM defines CTY value 0 as ``No cropland`` and 65535 as
    ``Outside area``. The model represents outside/no-coverage with -2, so 65535
    and non-finite values are normalized to that signed sentinel. Value 65534 is
    *not* a valid CTY code; it is a legitimate quality flag in some other HRL
    confidence products, so it must never be globally treated as nodata. If it
    reaches the CTY workflow, fail explicitly rather than silently relabelling it.
    """
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.floating):
        values = np.nan_to_num(
            values,
            nan=_HRL_MISSING_CROP_CODE,
            posinf=_HRL_MISSING_CROP_CODE,
            neginf=_HRL_MISSING_CROP_CODE,
        )
    values = np.ascontiguousarray(values.astype(np.int32, copy=False))
    if np.any(values == _HRL_GDAL_ALTERED_OUTSIDE_AREA_CODE):
        raise ValueError(
            "The HRL Crop Types (CTY) raster contains value 65534. According to "
            "the HRL Croplands Product User Manual, 65534 is not a valid CTY "
            "class/quality flag (CTY uses 0, the documented crop codes, and 65535 "
            "for outside area). A 65534 value in CTY therefore indicates an "
            "unexpected or altered source/cache value; do not normalize it silently."
        )
    if np.any(values == _HRL_OUTSIDE_AREA_CODE):
        values = values.copy()
        values[values == _HRL_OUTSIDE_AREA_CODE] = _HRL_MISSING_CROP_CODE
    return values


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

# Positive values that are semantically valid in the official HRL CTY product and
# have an explicit downstream MIRCA/GEB interpretation. Keep this contract separate
# from transport nodata handling: 65535 is normalized to -2 before this stage, while
# 0 remains the valid native ``No cropland`` state until the farm domain is selected.
_HRL_VALID_POSITIVE_CTY_CODES = frozenset(
    int(code)
    for code, mirca_code in HRL_TO_MIRCA_OS_CROP_CLASS_MAP.items()
    if int(code) > 0 and mirca_code is not None
)


def _validate_HRL_crop_area_codes(
    category_areas_m2: dict[int, float],
    *,
    context: str,
) -> None:
    """Reject positive CTY categories outside the official workflow contract.

    Validation uses the already aggregated category dictionary rather than rescanning
    the 10 m raster, so it adds negligible cost to native-area calculation.
    """
    unexpected = sorted(
        int(code)
        for code in category_areas_m2
        if int(code) > 0 and int(code) not in _HRL_VALID_POSITIVE_CTY_CODES
    )
    if unexpected:
        raise ValueError(
            f"{context} contains unexpected positive HRL CTY code(s) {unexpected[:10]}. "
            "Expected only documented CTY crop classes; 0 is no cropland, 65535 "
            "is outside area and is normalized to -2, and 65534 is invalid for CTY."
        )


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
    bounds are used only as the Copernicus HRL candidate-tile search envelope.
    The geometry itself is passed to the configured HRL adapter, which may use
    the local cache, Copernicus Data Space, or the legacy WEkEO fallback. Tiles
    outside the active domain can therefore be skipped before merging and
    intersecting tiles can be clipped before merging.

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
    present_ids = farm_values[farm_values != nodata]
    if present_ids.size == 0:
        raise ValueError("Farm raster contains no represented farmers.")

    n_farmers = len(farmers)
    minimum_present_id = int(present_ids.min())
    maximum_present_id = int(present_ids.max())
    if minimum_present_id < 0 or maximum_present_id >= n_farmers:
        raise ValueError(
            "Farm raster IDs are outside the compact farmer-table range. "
            f"Expected IDs 0..{n_farmers - 1}; observed range "
            f"{minimum_present_id}..{maximum_present_id}."
        )
    present_counts = np.bincount(present_ids, minlength=n_farmers)
    if present_counts.size != n_farmers or np.any(present_counts == 0):
        missing_ids = np.flatnonzero(present_counts == 0)
        raise ValueError(
            "Farm raster IDs are not compact or not aligned with the farmer table. "
            f"Expected IDs 0..{n_farmers - 1}. "
            f"Missing examples: {missing_ids[:10].tolist()}."
        )

    if farmer_id_column in farmers.columns:
        farmer_ids = farmers[farmer_id_column].to_numpy(dtype=np.int32)
        expected_ids = np.arange(n_farmers, dtype=np.int32)
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

    # Fast path for the normal output of setup_create_farms_from_HRL_lowder: the
    # table is already compact, ordered, and contains exactly one row per farmer.
    # Avoid copying/sorting a multi-million-row DataFrame once for every HRL year.
    compact_farmer_ids = farmers_with_crops["farmer_id"].to_numpy(dtype=np.int32)
    expected_farmer_ids = np.arange(n_farmers, dtype=np.int32)
    if compact_farmer_ids.size == n_farmers and np.array_equal(
        compact_farmer_ids, expected_farmer_ids
    ):
        hrl_crop = farmers_with_crops[crop_column].fillna(-1).to_numpy(dtype=np.int32)
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
        unexpected_crop = (hrl_crop > 0) & (mirca_crop == -1)
        if unexpected_crop.any():
            unexpected_codes = np.unique(hrl_crop[unexpected_crop])[:10].tolist()
            raise ValueError(
                f"Farmer crop column {crop_column!r} contains positive HRL CTY "
                f"code(s) without a configured MIRCA mapping: {unexpected_codes}. "
                "Refusing to treat an unknown positive category as fallow/missing."
            )
        farmer_crops = pd.DataFrame(
            {
                "farmer_id": compact_farmer_ids,
                "mirca_crop": mirca_crop.astype(np.int32),
                "assigned_crop_area_m2": farmers_with_crops["area_m2"].to_numpy(
                    dtype=np.float64
                ),
            }
        )
        if farmer_crops.empty:
            raise ValueError(
                f"No farmer-level HRL crops are available in {crop_column!r}."
            )
        return farmer_crops

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
    unexpected_crop = (hrl_crop > 0) & (mirca_crop == -1)
    if unexpected_crop.any():
        unexpected_codes = np.unique(hrl_crop[unexpected_crop])[:10].tolist()
        raise ValueError(
            f"Farmer crop column {crop_column!r} contains positive HRL CTY "
            f"code(s) without a configured MIRCA mapping: {unexpected_codes}. "
            "Refusing to treat an unknown positive category as fallow/missing."
        )

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


def _prepare_mirca_calendar_source_for_parsing(
    calendar_source: pd.DataFrame,
    *,
    mirca_units: list[int] | np.ndarray,
    source_name: str,
    logger: Any,
) -> pd.DataFrame:
    """Return in-scope MIRCA rows with crop names supported by the parser.

    Some MIRCA-OS calendar tables contain empty trailing or otherwise incomplete
    records. ``parse_MIRCA_crop_calendar`` creates ``crop_class`` by normalizing
    and mapping the raw ``Crop`` column, then casts the result to ``np.int64``.
    Missing or unsupported crop names therefore become NaN and previously raised an
    opaque ``IntCastingNaNError``. Remove only those unusable in-scope rows before
    parsing. Rows outside the current model domain are ignored before invalid-crop
    diagnostics are counted.

    Args:
        calendar_source: Raw rainfed or irrigated MIRCA-OS calendar table.
        mirca_units: MIRCA administrative unit codes overlapping the model.
        source_name: Short source label used in diagnostics.
        logger: Logger receiving invalid-row warnings.

    Returns:
        A copy containing only selected units and crop names represented in
        ``MIRCA_OS_CROP_CLASS_MAP``. The raw schema is otherwise unchanged.

    Raises:
        TypeError: If ``calendar_source`` is not a DataFrame.
        ValueError: If a required identifier column is absent.
    """
    if not isinstance(calendar_source, pd.DataFrame):
        raise TypeError(
            f"{source_name} MIRCA-OS calendar source must be a pandas DataFrame."
        )

    # ``crop_class`` is deliberately not required here: it is a derived column
    # created inside parse_MIRCA_crop_calendar from the raw ``Crop`` values.
    required_columns = {"unit_code", "Crop"}
    missing_columns = required_columns - set(calendar_source.columns)
    if missing_columns:
        raise ValueError(
            f"{source_name} MIRCA-OS calendar source is missing required "
            f"column(s): {sorted(missing_columns)}."
        )

    prepared = calendar_source.loc[
        calendar_source["unit_code"].isin(mirca_units)
    ].copy()
    if prepared.empty:
        return prepared

    normalized_crop_names = (
        prepared["Crop"]
        .astype(str)
        .str.strip()
        .str.replace(r"\d+$", "", regex=True)
        .str.strip()
    )
    valid_crop = normalized_crop_names.map(MIRCA_OS_CROP_CLASS_MAP).notna()
    if (~valid_crop).any():
        affected_units = prepared.loc[~valid_crop, "unit_code"].dropna().unique()
        invalid_names = normalized_crop_names.loc[~valid_crop]
        invalid_names = invalid_names.mask(
            prepared.loc[~valid_crop, "Crop"].isna(),
            "<missing>",
        )
        logger.warning(
            "Discarding %s %s MIRCA-OS calendar row(s) in selected unit(s) "
            "%s because Crop is missing or has no configured MIRCA-OS crop "
            "mapping. Crop examples: %s.",
            int(np.count_nonzero(~valid_crop)),
            source_name,
            affected_units[:10].tolist(),
            invalid_names.drop_duplicates().head(10).tolist(),
        )

    return prepared.loc[valid_crop].copy()


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


def _replacement_crop_calendar_candidates(
    crop_calendar_per_farmer: np.ndarray,
    crop_values: np.ndarray | list[int],
) -> np.ndarray:
    """Return replacement calendars in the established frequency order.

    The most frequent crop represented by ``crop_values`` is selected first.
    Its unique full calendars are then ordered by descending occurrence count,
    using the same stable tie handling as the original replacement workflow.

    Args:
        crop_calendar_per_farmer: Calendar array for one HRL year.
        crop_values: Crop classes eligible to provide replacement calendars.

    Returns:
        Ordered unique replacement calendars. The first axis has length zero
        when none of the requested crop classes is represented.
    """
    crop_instances = crop_calendar_per_farmer[:, :, 0][
        np.isin(crop_calendar_per_farmer[:, :, 0], crop_values)
    ]
    if crop_instances.size == 0:
        return np.empty(
            (0,) + crop_calendar_per_farmer.shape[1:],
            dtype=crop_calendar_per_farmer.dtype,
        )

    crops, crop_counts = np.unique(crop_instances, return_counts=True)
    most_common_crop = crops[np.argmax(crop_counts)]
    calendars_for_crop = crop_calendar_per_farmer[
        (crop_calendar_per_farmer[:, :, 0] == most_common_crop).any(axis=1)
    ]
    unique_calendars, counts = np.unique(
        calendars_for_crop,
        axis=0,
        return_counts=True,
    )
    candidate_order = np.argsort(-counts, kind="stable")
    return unique_calendars[candidate_order]


# Manual replacement of certain crops
def replace_crop(
    crop_calendar_per_farmer: np.ndarray,
    crop_values: np.ndarray | list[int],
    replaced_crop_values: np.ndarray | list[int],
    minimum_planting_days: np.ndarray | None = None,
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
        minimum_planting_days: Optional inclusive earliest planting-day bound
            for each farmer. When supplied, each replacement uses the most
            common candidate calendar that starts on or after its bound.

    Returns:
        The updated crop-calendar array. If none of the candidate crops are
        present, the input array is returned unchanged.
    """
    replacement_candidates = _replacement_crop_calendar_candidates(
        crop_calendar_per_farmer,
        crop_values,
    )
    if replacement_candidates.shape[0] == 0:
        return crop_calendar_per_farmer
    crop_replacement = replacement_candidates[0]

    crop_replacement_only_crops = crop_replacement[crop_replacement[:, -1] != -1]
    if crop_replacement_only_crops.shape[0] > 1:
        assert (
            np.unique(crop_replacement_only_crops[:, [1, 3]], axis=0).shape[0]
            == crop_replacement_only_crops.shape[0]
        )

    if minimum_planting_days is not None:
        minimum_planting_days = np.asarray(minimum_planting_days, dtype=np.int64)
        if (
            minimum_planting_days.ndim != 1
            or minimum_planting_days.size != crop_calendar_per_farmer.shape[0]
        ):
            raise ValueError(
                "minimum_planting_days must contain one value per farmer when "
                "replacing crop calendars."
            )
        _, replacement_start_days, _ = _calendar_timing_offsets(replacement_candidates)

    for replaced_crop in replaced_crop_values:
        # Check where to be replaced crop is
        crop_mask = (crop_calendar_per_farmer[:, :, 0] == replaced_crop).any(axis=1)
        if minimum_planting_days is None:
            crop_calendar_per_farmer[crop_mask] = crop_replacement
            continue

        for farmer_id in np.flatnonzero(crop_mask):
            required_day = max(int(minimum_planting_days[farmer_id]), 0)
            feasible = replacement_start_days >= required_day
            if not feasible.any():
                raise ValueError(
                    "No sequentially feasible replacement crop calendar for "
                    f"farmer={int(farmer_id)}, replaced crop={int(replaced_crop)}. "
                    f"Required planting day is {required_day}; candidate starts "
                    f"are {np.unique(replacement_start_days).astype(int).tolist()}."
                )
            crop_calendar_per_farmer[farmer_id] = replacement_candidates[
                np.flatnonzero(feasible)[0]
            ]

    return crop_calendar_per_farmer


def _calendar_active_rows(calendar: np.ndarray) -> np.ndarray:
    """Return active crop rows from a crop calendar matrix.

    Args:
        calendar: Crop calendar matrix.

    Returns:
        Rows where the crop ID is not ``-1``.
    """
    return calendar[calendar[:, 0] != -1]


_NO_HARVEST_DAY = np.iinfo(np.int64).min


def _calendar_timing_offsets(
    calendars: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return active flags plus earliest planting and latest harvest offsets.

    Compact crop-calendar rows use the layout ``[crop_id, planting_day,
    duration, rotation_year]``. Timing offsets are measured from 1 January of
    the calendar's HRL year. A rotation-year offset uses the model's 365-day
    crop-calendar convention. The HRL workflow currently writes rotation year
    zero, but including the offset keeps this check correct for other valid
    compact calendars too.

    Args:
        calendars: Compact calendars with shape ``(farmer, row, variable)``.

    Returns:
        A tuple containing whether each farmer has an active crop, the earliest
        active planting-day offset, and the latest active harvest-day offset.

    Raises:
        ValueError: If the array shape or active timing values are invalid.
    """
    calendars = np.asarray(calendars)
    if calendars.ndim != 3 or calendars.shape[2] < 4:
        raise ValueError(
            "Compact crop calendars must have shape (farmer, row, >=4). "
            f"Got {calendars.shape}."
        )

    active = calendars[:, :, 0] != -1
    planting_day = calendars[:, :, 1].astype(np.int64, copy=False)
    duration = calendars[:, :, 2].astype(np.int64, copy=False)
    rotation_year = calendars[:, :, 3].astype(np.int64, copy=False)

    invalid_timing = active & (
        (planting_day < 0) | (duration < 0) | (rotation_year < 0)
    )
    if invalid_timing.any():
        invalid_farmer, invalid_row = np.argwhere(invalid_timing)[0]
        raise ValueError(
            "Active compact crop-calendar rows require non-negative planting "
            "day, duration, and rotation year. First invalid row: farmer="
            f"{int(invalid_farmer)}, row={int(invalid_row)}, values="
            f"{calendars[invalid_farmer, invalid_row, :4].tolist()}."
        )

    start_offset = planting_day + rotation_year * 365
    harvest_offset = start_offset + duration
    has_active_crop = active.any(axis=1)

    earliest_planting = np.min(
        np.where(active, start_offset, np.iinfo(np.int64).max),
        axis=1,
    )
    latest_harvest = np.max(
        np.where(active, harvest_offset, _NO_HARVEST_DAY),
        axis=1,
    )
    earliest_planting[~has_active_crop] = _NO_HARVEST_DAY
    latest_harvest[~has_active_crop] = _NO_HARVEST_DAY
    return has_active_crop, earliest_planting, latest_harvest


def _year_start_day(year: int) -> int:
    """Return the NumPy day ordinal for 1 January of ``year``."""
    return int(np.datetime64(f"{int(year):04d}-01-01", "D").astype(np.int64))


def _minimum_planting_days_after_last_harvest(
    last_harvest_absolute_days: np.ndarray,
    *,
    current_hrl_year: int,
) -> np.ndarray:
    """Convert absolute last-harvest dates to current-year planting bounds.

    The returned bound is inclusive: planting on the harvest date is valid
    because the crop can be harvested before the new crop is planted that day.
    Farmers without an earlier active calendar receive a zero-day bound.
    """
    last_harvest_absolute_days = np.asarray(
        last_harvest_absolute_days,
        dtype=np.int64,
    )
    if last_harvest_absolute_days.ndim != 1:
        raise ValueError("last_harvest_absolute_days must be one-dimensional.")

    minimum_planting_days = np.zeros(
        last_harvest_absolute_days.size,
        dtype=np.int64,
    )
    has_previous_harvest = last_harvest_absolute_days != _NO_HARVEST_DAY
    minimum_planting_days[has_previous_harvest] = last_harvest_absolute_days[
        has_previous_harvest
    ] - _year_start_day(current_hrl_year)
    return minimum_planting_days


def _is_europe_004_model_root(model_root: str | os.PathLike[str]) -> bool:
    """Return whether a model input path belongs to exactly ``Europe_004``.

    The model input directory is normally nested below the model directory, so
    all complete path components are inspected. Similar names such as
    ``Europe_004_backup`` are deliberately excluded.

    Args:
        model_root: Model input directory or one of its parent directories.

    Returns:
        True only when an exact path component is named ``Europe_004``.
    """
    return any(part == "Europe_004" for part in Path(model_root).parts)


def _calendar_rows_as_keys(calendars: np.ndarray) -> np.ndarray:
    """Return exact byte keys for a two-dimensional collection of calendars.

    Args:
        calendars: Calendar array with shape ``(farmer, row, variable)``.

    Returns:
        One exact fixed-width key per farmer calendar.

    Raises:
        ValueError: If the input does not have three dimensions.
    """
    calendars = np.asarray(calendars)
    if calendars.ndim != 3:
        raise ValueError(
            f"calendars must have shape (farmer, row, variable). Got {calendars.shape}."
        )
    contiguous = np.ascontiguousarray(calendars)
    key_width = int(np.prod(contiguous.shape[1:])) * contiguous.dtype.itemsize
    return (
        contiguous.reshape(contiguous.shape[0], -1)
        .view(np.dtype((np.void, key_width)))
        .reshape(-1)
    )


def _most_common_calendar_sequence_donor(
    crop_calendar_stack: np.ndarray,
    candidate_farmer_ids: np.ndarray,
) -> tuple[int, int]:
    """Return a deterministic donor for the modal complete calendar sequence.

    Args:
        crop_calendar_stack: Calendar array with shape
            ``(year, farmer, row, variable)``.
        candidate_farmer_ids: Compact farmer IDs eligible to donate a sequence.

    Returns:
        Donor farmer ID and the number of eligible farmers sharing its sequence.

    Raises:
        ValueError: If no candidate farmer is provided.
    """
    candidate_farmer_ids = np.asarray(candidate_farmer_ids, dtype=np.int32)
    if candidate_farmer_ids.ndim != 1 or not candidate_farmer_ids.size:
        raise ValueError("At least one calendar-sequence donor is required.")

    n_years = crop_calendar_stack.shape[0]
    sequence_codes = np.empty(
        (candidate_farmer_ids.size, n_years),
        dtype=np.int32,
    )
    for year_index in range(n_years):
        annual_keys = _calendar_rows_as_keys(
            crop_calendar_stack[year_index, candidate_farmer_ids]
        )
        _, annual_inverse = np.unique(annual_keys, return_inverse=True)
        sequence_codes[:, year_index] = annual_inverse.astype(
            np.int32,
            copy=False,
        )

    sequence_keys = _calendar_rows_as_keys(sequence_codes[:, :, None])
    _, first_positions, sequence_counts = np.unique(
        sequence_keys,
        return_index=True,
        return_counts=True,
    )
    maximum_count = int(sequence_counts.max())
    modal_positions = np.flatnonzero(sequence_counts == maximum_count)
    # A frequency tie is resolved by the smallest farmer ID, which is stable
    # because candidate_farmer_ids is in ascending compact-ID order.
    modal_unique_position = int(
        modal_positions[np.argmin(first_positions[modal_positions])]
    )
    donor_position = int(first_positions[modal_unique_position])
    return int(candidate_farmer_ids[donor_position]), maximum_count


def _assert_europe_004_calendar_has_no_rice(
    crop_calendar_stack: np.ndarray,
    *,
    rice_crop_id: int,
) -> None:
    """Assert that the final Europe_004 calendar contains no rice crop rows.

    Args:
        crop_calendar_stack: Calendar array with shape
            ``(year, farmer, row, variable)``.
        rice_crop_id: MIRCA crop class used for rice.

    Raises:
        ValueError: If the calendar stack has an invalid shape.
        AssertionError: If rice remains in any year, farmer, or calendar row.
    """
    crop_calendar_stack = np.asarray(crop_calendar_stack)
    if crop_calendar_stack.ndim != 4 or crop_calendar_stack.shape[3] < 1:
        raise ValueError(
            "crop_calendar_stack must have shape (year, farmer, row, variable). "
            f"Got {crop_calendar_stack.shape}."
        )
    rice_locations = np.argwhere(crop_calendar_stack[:, :, :, 0] == int(rice_crop_id))
    if rice_locations.size:
        rice_farmer_ids = np.unique(rice_locations[:, 1])
        samples = [
            {
                "year_index": int(year_index),
                "farmer_id": int(farmer_id),
                "calendar_row": int(calendar_row),
            }
            for year_index, farmer_id, calendar_row in rice_locations[:10]
        ]
        raise AssertionError(
            "Europe_004 must not contain rice after its rare-rice calendar "
            f"replacement, but found {rice_locations.shape[0]} rice calendar "
            f"row(s) for {rice_farmer_ids.size} farmer(s). Examples: {samples}."
        )


def _replace_europe_004_rare_rice_sequence(
    crop_calendar_stack: np.ndarray,
    farmer_main_crops_by_year: np.ndarray,
    farmer_region_ids: np.ndarray,
    *,
    rice_crop_id: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replace every Europe_004 rice farmer with a local modal sequence.

    Rice is detected from every crop row of the final calendar stack, matching
    the data later inspected by crop-price validation. For each affected
    region, the complete calendar sequence is copied from the modal existing
    farmer sequence that contains no rice in any year or row. Copying a
    sequence that already passed the joint selector preserves all year-to-year
    harvest constraints.

    The final no-rice assertion makes the model-specific price-data assumption
    explicit and prevents a partially applied replacement from being written.

    Args:
        crop_calendar_stack: Calendar array with shape
            ``(year, farmer, row, variable)``. The array is updated in place.
        farmer_main_crops_by_year: MIRCA crop IDs with shape ``(year, farmer)``.
        farmer_region_ids: Region ID for each compact farmer.
        rice_crop_id: MIRCA crop class used for rice.

    Returns:
        Updated calendar stack and replacement diagnostics by region.

    Raises:
        ValueError: If arrays are misaligned, more than 5,000 rice farmers are
            found, or a rice farmer has no rice-free donor in the same region.
        AssertionError: If any rice calendar row remains after replacement.
    """
    crop_calendar_stack = np.asarray(crop_calendar_stack)
    main_crops = np.asarray(farmer_main_crops_by_year, dtype=np.int32)
    region_ids = np.asarray(farmer_region_ids, dtype=np.int32)
    if crop_calendar_stack.ndim != 4:
        raise ValueError(
            "crop_calendar_stack must have shape (year, farmer, row, variable). "
            f"Got {crop_calendar_stack.shape}."
        )
    if main_crops.shape != crop_calendar_stack.shape[:2]:
        raise ValueError(
            "farmer_main_crops_by_year must match the calendar year/farmer "
            f"axes. Got {main_crops.shape} and {crop_calendar_stack.shape[:2]}."
        )
    _, n_farmers = main_crops.shape
    if region_ids.ndim != 1 or region_ids.size != n_farmers:
        raise ValueError(
            "farmer_region_ids must contain one value per farmer. "
            f"Got {region_ids.shape} for {n_farmers} farmers."
        )

    calendar_rice_by_farmer = np.any(
        crop_calendar_stack[:, :, :, 0] == int(rice_crop_id),
        axis=(0, 2),
    )
    main_crop_rice_by_farmer = np.any(main_crops == int(rice_crop_id), axis=0)
    rice_by_farmer = calendar_rice_by_farmer | main_crop_rice_by_farmer
    rice_farmer_ids = np.flatnonzero(rice_by_farmer)
    if not rice_farmer_ids.size:
        _assert_europe_004_calendar_has_no_rice(
            crop_calendar_stack,
            rice_crop_id=rice_crop_id,
        )
        return crop_calendar_stack, {
            "rice_farmer_count": 0,
            "rice_calendar_farmer_count": 0,
            "region_count": 0,
            "regions": [],
        }
    if rice_farmer_ids.size > 5_000:
        raise ValueError(
            "The Europe_004 rice exception expects at most 5,000 farmers, "
            f"but found {rice_farmer_ids.size}. Refusing a broad replacement."
        )

    regional_replacements: list[dict[str, int]] = []
    for region_id in np.unique(region_ids[rice_farmer_ids]):
        region_id = int(region_id)
        regional_rice_farmer_ids = np.flatnonzero(
            rice_by_farmer & (region_ids == region_id)
        ).astype(np.int32, copy=False)
        donor_farmer_ids = np.flatnonzero(
            (region_ids == region_id) & ~rice_by_farmer
        ).astype(np.int32, copy=False)
        if not donor_farmer_ids.size:
            raise ValueError(
                "Europe_004 rice farmers have no rice-free calendar-sequence "
                f"donor in region {region_id}."
            )

        donor_farmer_id, modal_sequence_count = _most_common_calendar_sequence_donor(
            crop_calendar_stack,
            donor_farmer_ids,
        )
        crop_calendar_stack[:, regional_rice_farmer_ids] = crop_calendar_stack[
            :, donor_farmer_id, None, :, :
        ]
        regional_replacements.append(
            {
                "region_id": region_id,
                "rice_farmer_count": int(regional_rice_farmer_ids.size),
                "donor_farmer_id": donor_farmer_id,
                "modal_sequence_count": modal_sequence_count,
                "candidate_farmer_count": int(donor_farmer_ids.size),
            }
        )

    _assert_europe_004_calendar_has_no_rice(
        crop_calendar_stack,
        rice_crop_id=rice_crop_id,
    )

    return crop_calendar_stack, {
        "rice_farmer_count": int(rice_farmer_ids.size),
        "rice_calendar_farmer_count": int(np.count_nonzero(calendar_rice_by_farmer)),
        "region_count": len(regional_replacements),
        "regions": regional_replacements,
    }


def check_crop_calendar_sequence(
    crop_calendar_stack: np.ndarray,
    hrl_years: np.ndarray | tuple[int, ...],
) -> dict[str, int | None]:
    """Assert that each new calendar starts after the last assigned harvest.

    Fallow years do not erase a farmer's last harvest date. Therefore a crop
    assigned after one or more fallow years is still checked against the last
    earlier active calendar. Planting on exactly the harvest date is accepted.

    Args:
        crop_calendar_stack: Calendar array with shape
            ``(year, farmer, row, variable)``.
        hrl_years: Strictly increasing HRL years corresponding to axis zero.

    Returns:
        Counts useful for setup logging, including checked transitions,
        same-day harvest/plant transitions, and the minimum valid gap.

    Raises:
        ValueError: If stack and year structures do not line up.
        AssertionError: If any planting precedes the farmer's last harvest.
    """
    crop_calendar_stack = np.asarray(crop_calendar_stack)
    years = np.asarray(hrl_years, dtype=np.int64)
    if crop_calendar_stack.ndim != 4 or crop_calendar_stack.shape[3] < 4:
        raise ValueError(
            "crop_calendar_stack must have shape (year, farmer, row, >=4). "
            f"Got {crop_calendar_stack.shape}."
        )
    if years.ndim != 1 or years.size != crop_calendar_stack.shape[0]:
        raise ValueError(
            "hrl_years must be one-dimensional and match the calendar year "
            f"axis. Got {years.shape} for {crop_calendar_stack.shape[0]} years."
        )
    if years.size > 1 and np.any(np.diff(years) <= 0):
        raise ValueError(
            "hrl_years must be strictly increasing for sequential calendar "
            f"assignment. Got {years.tolist()}."
        )

    n_farmers = crop_calendar_stack.shape[1]
    last_harvest = np.full(n_farmers, _NO_HARVEST_DAY, dtype=np.int64)
    checked_transitions = 0
    same_day_transitions = 0
    minimum_gap_days: int | None = None

    for year_index, current_hrl_year in enumerate(years):
        calendars = crop_calendar_stack[year_index]
        has_crop, earliest_planting, latest_harvest = _calendar_timing_offsets(
            calendars
        )
        year_start = _year_start_day(int(current_hrl_year))
        earliest_absolute = year_start + earliest_planting
        latest_absolute = year_start + latest_harvest
        has_previous_harvest = last_harvest != _NO_HARVEST_DAY
        must_check = has_crop & has_previous_harvest
        gaps = np.zeros(n_farmers, dtype=np.int64)
        gaps[must_check] = earliest_absolute[must_check] - last_harvest[must_check]
        invalid = must_check & (gaps < 0)

        if invalid.any():
            invalid_farmer_ids = np.flatnonzero(invalid)
            sample = [
                {
                    "farmer_id": int(farmer_id),
                    "hrl_year": int(current_hrl_year),
                    "earliest_planting_day": int(earliest_planting[farmer_id]),
                    "required_minimum_day": int(last_harvest[farmer_id] - year_start),
                    "days_before_previous_harvest": int(-gaps[farmer_id]),
                }
                for farmer_id in invalid_farmer_ids[:10]
            ]
            raise AssertionError(
                "Crop-calendar sequence violation: the next assigned calendar "
                "starts before the farmer's last assigned calendar has been "
                f"harvested for {invalid_farmer_ids.size} farmer(s). Examples: "
                f"{sample}."
            )

        valid_gaps = gaps[must_check]
        checked_transitions += int(valid_gaps.size)
        same_day_transitions += int(np.count_nonzero(valid_gaps == 0))
        if valid_gaps.size:
            year_minimum_gap = int(valid_gaps.min())
            minimum_gap_days = (
                year_minimum_gap
                if minimum_gap_days is None
                else min(minimum_gap_days, year_minimum_gap)
            )

        # A fallow year intentionally leaves the previous harvest unchanged.
        last_harvest[has_crop] = latest_absolute[has_crop]

    return {
        "checked_transitions": checked_transitions,
        "same_day_transitions": same_day_transitions,
        "minimum_gap_days": minimum_gap_days,
    }


def _candidate_mirca_calendars(
    crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
    *,
    mirca_unit: int,
    main_crop: int,
    is_irrigated: bool,
    replace_crop_calendar_unit_code: dict[int, int],
) -> tuple[int, list[tuple[float, TwoDArrayInt32]]]:
    """Return ordered MIRCA-OS calendar candidates for one farmer state.

    The fallback order is unchanged from the original implementation, but each
    stage is evaluated lazily. This avoids scanning all MIRCA units when a local
    candidate already exists.
    """
    lookup_unit = int(replace_crop_calendar_unit_code.get(mirca_unit, mirca_unit))

    def contains_crop(entry: tuple[float, TwoDArrayInt32]) -> bool:
        active_rows = _calendar_active_rows(entry[1])
        return active_rows.size > 0 and main_crop in active_rows[:, 0]

    def matches_irrigation(entry: tuple[float, TwoDArrayInt32]) -> bool:
        active_rows = _calendar_active_rows(entry[1])
        return active_rows.size > 0 and bool(active_rows[0, 1]) == is_irrigated

    local_entries = crop_calendar.get(lookup_unit, [])

    candidates = [
        entry
        for entry in local_entries
        if contains_crop(entry) and matches_irrigation(entry)
    ]
    if candidates:
        return lookup_unit, candidates

    candidates = [entry for entry in local_entries if contains_crop(entry)]
    if candidates:
        return lookup_unit, candidates

    candidates = [
        entry
        for unit_code, entries in crop_calendar.items()
        if unit_code != lookup_unit
        for entry in entries
        if contains_crop(entry) and matches_irrigation(entry)
    ]
    if candidates:
        return lookup_unit, candidates

    candidates = [
        entry
        for unit_code, entries in crop_calendar.items()
        if unit_code != lookup_unit
        for entry in entries
        if contains_crop(entry)
    ]
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
    selection_cache: dict[tuple[int, int, int, bool, int], np.ndarray],
    random_seed: int,
    minimum_planting_days: np.ndarray | None = None,
    current_hrl_year: int | None = None,
    candidate_pool_cache: dict[
        tuple[int, int, bool],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]
    | None = None,
) -> tuple[np.ndarray, int, int, int, int]:
    """Assign area-weighted MIRCA-OS calendars using only the HRL main crop.

    Candidate calendars whose earliest planting precedes the farmer's last
    harvest are removed before the area-weighted selection. Farmer-specific
    random priorities retain the original deterministic RNG key and initial
    area-weighted draw, so existing unconstrained assignments remain unchanged.
    Tightening the constraint changes a selection only when the previously
    preferred candidate is no longer feasible.

    Candidate discovery, compact-calendar conversion, timing extraction, and
    probability construction are cached by MIRCA unit, crop, and irrigation
    state because those quantities do not depend on farmer ID.
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

    if minimum_planting_days is None:
        minimum_planting_days = np.zeros(n_farmers, dtype=np.int64)
    else:
        minimum_planting_days = np.asarray(
            minimum_planting_days,
            dtype=np.int64,
        )
        if minimum_planting_days.ndim != 1:
            raise ValueError("minimum_planting_days must be one-dimensional.")
        if minimum_planting_days.size != n_farmers:
            raise ValueError(
                "minimum_planting_days must contain one value per farmer. "
                f"Got {minimum_planting_days.size} values for {n_farmers} farmers."
            )

    if candidate_pool_cache is None:
        candidate_pool_cache = {}

    calendars = np.full((n_farmers, 3, 4), -1, dtype=np.int32)
    state_keys: set[tuple[int, int, bool]] = set()
    n_cache_misses = 0
    n_farmers_with_filtered_candidates = 0
    n_filtered_candidates = 0

    for farmer_id in range(n_farmers):
        main_crop = int(farmer_main_crops[farmer_id])
        if main_crop == -1:
            continue
        mirca_unit = int(farmer_mirca_units[farmer_id])
        is_irrigated = bool(farmer_is_irrigated[farmer_id])
        lookup_unit = int(replace_crop_calendar_unit_code.get(mirca_unit, mirca_unit))
        state_key = (lookup_unit, main_crop, is_irrigated)
        state_keys.add(state_key)

        candidate_pool = candidate_pool_cache.get(state_key)
        if candidate_pool is None:
            _, candidates = _candidate_mirca_calendars(
                crop_calendar,
                mirca_unit=lookup_unit,
                main_crop=main_crop,
                is_irrigated=is_irrigated,
                replace_crop_calendar_unit_code={},
            )
            areas = np.asarray(
                [max(float(area), 0.0) for area, _ in candidates],
                dtype=np.float64,
            )
            area_sum = areas.sum()
            probabilities = (
                areas / area_sum
                if area_sum > 0.0
                else np.full(len(candidates), 1.0 / len(candidates))
            )
            compact_calendars = np.stack(
                [
                    np.asarray(full_calendar[:, [0, 2, 3, 4]], dtype=np.int32)
                    for _, full_calendar in candidates
                ],
                axis=0,
            )
            if compact_calendars.ndim != 3 or compact_calendars.shape[1:] != (3, 4):
                raise ValueError(
                    "Every selected MIRCA-OS calendar must have shape (3, 4) "
                    "after compact column selection. Got candidate pool shape "
                    f"{compact_calendars.shape} for state {state_key}."
                )
            _, earliest_planting, _ = _calendar_timing_offsets(compact_calendars)
            candidate_pool = (
                np.ascontiguousarray(compact_calendars),
                probabilities,
                earliest_planting,
            )
            candidate_pool_cache[state_key] = candidate_pool

        compact_calendars, probabilities, earliest_planting = candidate_pool
        # Planting days cannot be negative, so all bounds before 1 January have
        # the same feasible pool and reuse the same cache entry.
        required_planting_day = max(int(minimum_planting_days[farmer_id]), 0)
        feasible = earliest_planting >= required_planting_day
        filtered_count = int(feasible.size - np.count_nonzero(feasible))
        if filtered_count:
            n_farmers_with_filtered_candidates += 1
            n_filtered_candidates += filtered_count
        if not feasible.any():
            available_days = np.unique(earliest_planting).astype(int).tolist()
            year_context = (
                f", HRL year={int(current_hrl_year)}"
                if current_hrl_year is not None
                else ""
            )
            raise ValueError(
                "No sequentially feasible MIRCA-OS calendar remains for "
                f"farmer={farmer_id}{year_context}, unit={lookup_unit}, "
                f"crop={main_crop}, irrigated={is_irrigated}. The next calendar "
                f"must start on or after day {required_planting_day}, but "
                f"candidate earliest planting days are {available_days}."
            )

        cache_key = (
            farmer_id,
            lookup_unit,
            main_crop,
            is_irrigated,
            required_planting_day,
        )
        selected = selection_cache.get(cache_key)
        if selected is None:
            seed = np.random.SeedSequence(
                [random_seed, farmer_id, lookup_unit, main_crop, int(is_irrigated)]
            )
            rng = np.random.default_rng(seed)
            # Draw a deterministic, area-weighted order without replacement and
            # take its first feasible item. The first draw is exactly the legacy
            # unconstrained ``rng.choice`` call. Consequently existing selections
            # are preserved unless that first choice violates the harvest bound.
            remaining_indices = np.arange(probabilities.size, dtype=np.int64)
            while remaining_indices.size:
                remaining_probabilities = probabilities[remaining_indices]
                probability_sum = float(remaining_probabilities.sum())
                draw_probabilities = (
                    remaining_probabilities / probability_sum
                    if probability_sum > 0.0
                    else np.full(
                        remaining_indices.size,
                        1.0 / remaining_indices.size,
                    )
                )
                draw_position = int(
                    rng.choice(remaining_indices.size, p=draw_probabilities)
                )
                candidate_index = int(remaining_indices[draw_position])
                if feasible[candidate_index]:
                    break
                remaining_indices = np.delete(remaining_indices, draw_position)
            else:  # pragma: no cover - guarded by feasible.any() above
                raise AssertionError("Feasible calendar candidate disappeared.")
            selected = np.ascontiguousarray(compact_calendars[candidate_index])
            selection_cache[cache_key] = selected
            n_cache_misses += 1
        calendars[farmer_id] = selected

    return (
        calendars,
        len(state_keys),
        n_cache_misses,
        n_farmers_with_filtered_candidates,
        n_filtered_candidates,
    )


_CALENDAR_FALLBACK_NAMES = (
    "local_matching_irrigation",
    "local_other_irrigation",
    "other_unit_matching_irrigation",
    "other_unit_other_irrigation",
)


@dataclass(frozen=True, slots=True)
class _MIRCACalendarCandidatePool:
    """Cached MIRCA calendars and timing metadata for one farmer state."""

    calendars: np.ndarray
    probabilities: np.ndarray
    fallback_tiers: np.ndarray
    source_units: np.ndarray
    earliest_planting: np.ndarray
    latest_harvest: np.ndarray


@dataclass(frozen=True, slots=True)
class _MIRCACalendarPoolTierMetadata:
    """Reusable fallback-tier indices for one cached candidate pool."""

    minimum_tier: int
    available_tiers: tuple[int, ...]
    tier_indices: tuple[np.ndarray, ...]
    eligible_indices: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class _MIRCACalendarIndex:
    """One-pass lookup index used by all farmer calendar states."""

    by_unit_crop_irrigation: dict[
        tuple[int, int, bool],
        list[tuple[int, float, TwoDArrayInt32]],
    ]
    by_crop_irrigation: dict[
        tuple[int, bool],
        list[tuple[int, float, TwoDArrayInt32]],
    ]


def _index_mirca_calendars(
    crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
) -> _MIRCACalendarIndex:
    """Index MIRCA entries once instead of scanning all units per state."""
    by_unit: dict[
        tuple[int, int, bool],
        list[tuple[int, float, TwoDArrayInt32]],
    ] = {}
    by_crop: dict[
        tuple[int, bool],
        list[tuple[int, float, TwoDArrayInt32]],
    ] = {}
    for source_unit, entries in crop_calendar.items():
        source_unit = int(source_unit)
        for area, full_calendar in entries:
            active_rows = _calendar_active_rows(full_calendar)
            if not active_rows.size:
                continue
            is_irrigated = bool(active_rows[0, 1])
            record = (source_unit, float(area), full_calendar)
            for crop_id in np.unique(active_rows[:, 0]).astype(np.int32):
                by_unit.setdefault(
                    (source_unit, int(crop_id), is_irrigated),
                    [],
                ).append(record)
                by_crop.setdefault((int(crop_id), is_irrigated), []).append(record)
    return _MIRCACalendarIndex(
        by_unit_crop_irrigation=by_unit,
        by_crop_irrigation=by_crop,
    )


def _build_mirca_calendar_candidate_pool(
    crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
    *,
    lookup_unit: int,
    main_crop: int,
    is_irrigated: bool,
    calendar_index: _MIRCACalendarIndex | None = None,
) -> _MIRCACalendarCandidatePool:
    """Build all four exclusive fallback tiers for joint sequence selection.

    Unlike :func:`_candidate_mirca_calendars`, this function does not stop at
    the first non-empty tier. Joint selection needs the later tiers when all
    local candidates lead to a temporal dead end elsewhere in the sequence.
    """

    if calendar_index is None:
        calendar_index = _index_mirca_calendars(crop_calendar)
    local_matching = calendar_index.by_unit_crop_irrigation.get(
        (lookup_unit, main_crop, is_irrigated),
        [],
    )
    local_other = calendar_index.by_unit_crop_irrigation.get(
        (lookup_unit, main_crop, not is_irrigated),
        [],
    )
    other_matching = [
        entry
        for entry in calendar_index.by_crop_irrigation.get(
            (main_crop, is_irrigated),
            [],
        )
        if entry[0] != lookup_unit
    ]
    other_irrigation = [
        entry
        for entry in calendar_index.by_crop_irrigation.get(
            (main_crop, not is_irrigated),
            [],
        )
        if entry[0] != lookup_unit
    ]
    tier_entries = [
        local_matching,
        local_other,
        other_matching,
        other_irrigation,
    ]

    compact_calendars: list[np.ndarray] = []
    probabilities: list[float] = []
    fallback_tiers: list[int] = []
    source_units: list[int] = []

    for fallback_tier, entries in enumerate(tier_entries):
        if not entries:
            continue
        # Multiple MIRCA units often contain exactly the same compact calendar.
        # Collapse those duplicates inside a tier and sum their areas. This keeps
        # the calendar-pattern probability unchanged while greatly reducing the
        # cross-unit dynamic-programming state space.
        unique_entries: dict[
            bytes,
            tuple[int, float, np.ndarray],
        ] = {}
        for source_unit, area, full_calendar in entries:
            compact_calendar = np.asarray(
                full_calendar[:, [0, 2, 3, 4]],
                dtype=np.int32,
            )
            if compact_calendar.shape != (3, 4):
                raise ValueError(
                    "MIRCA-OS calendars must have shape (3, 4) after compact "
                    f"column selection. Got {compact_calendar.shape} for "
                    f"unit={source_unit}, crop={main_crop}."
                )
            calendar_key = compact_calendar.tobytes()
            existing = unique_entries.get(calendar_key)
            if existing is None:
                unique_entries[calendar_key] = (
                    source_unit,
                    max(float(area), 0.0),
                    compact_calendar,
                )
            else:
                representative_unit, combined_area, existing_calendar = existing
                unique_entries[calendar_key] = (
                    representative_unit,
                    combined_area + max(float(area), 0.0),
                    existing_calendar,
                )
        entries_compact = list(unique_entries.values())
        areas = np.asarray(
            [area for _, area, _ in entries_compact],
            dtype=np.float64,
        )
        area_sum = float(areas.sum())
        tier_probabilities = (
            areas / area_sum
            if area_sum > 0.0
            else np.full(len(entries_compact), 1.0 / len(entries_compact))
        )
        for entry_index, (source_unit, _, compact_calendar) in enumerate(
            entries_compact
        ):
            compact_calendars.append(compact_calendar)
            probabilities.append(float(tier_probabilities[entry_index]))
            fallback_tiers.append(fallback_tier)
            source_units.append(source_unit)

    if not compact_calendars:
        return _MIRCACalendarCandidatePool(
            calendars=np.empty((0, 3, 4), dtype=np.int32),
            probabilities=np.empty(0, dtype=np.float64),
            fallback_tiers=np.empty(0, dtype=np.int8),
            source_units=np.empty(0, dtype=np.int32),
            earliest_planting=np.empty(0, dtype=np.int64),
            latest_harvest=np.empty(0, dtype=np.int64),
        )

    calendars = np.ascontiguousarray(np.stack(compact_calendars, axis=0))
    _, earliest_planting, latest_harvest = _calendar_timing_offsets(calendars)
    return _MIRCACalendarCandidatePool(
        calendars=calendars,
        probabilities=np.asarray(probabilities, dtype=np.float64),
        fallback_tiers=np.asarray(fallback_tiers, dtype=np.int8),
        source_units=np.asarray(source_units, dtype=np.int32),
        earliest_planting=earliest_planting,
        latest_harvest=latest_harvest,
    )


def _weighted_candidate_ranks(
    pool: _MIRCACalendarCandidatePool,
    *,
    random_seed: int,
    farmer_id: int,
    lookup_unit: int,
    main_crop: int,
    is_irrigated: bool,
    ranks: np.ndarray | None = None,
    maximum_fallback_tier: int | None = None,
    tier_indices_by_tier: tuple[np.ndarray, ...] | None = None,
) -> np.ndarray:
    """Fill deterministic area-weighted preference ranks within requested tiers.

    Each tier restarts the legacy farmer-state random stream. Therefore the
    first-ranked candidate in the first available tier matches the original
    single-year ``rng.choice`` result when no sequence constraint intervenes.

    Joint path selection normally succeeds using only the first available
    fallback tier. Ranking later, often much larger, cross-unit tiers eagerly
    was therefore wasted work for nearly every farmer-state. ``ranks`` permits
    those tiers to be filled lazily as the path solver broadens its search.
    Existing filled tiers are retained, and omitting both optional arguments
    preserves the former behavior of returning a complete preference order.

    Args:
        pool: Cached calendar candidates and area probabilities.
        random_seed: Base seed for reproducible farmer selection.
        farmer_id: Compact farmer identifier included in the RNG seed.
        lookup_unit: MIRCA-OS unit used for calendar lookup.
        main_crop: MIRCA crop class used for calendar lookup.
        is_irrigated: Whether irrigated calendars are preferred.
        ranks: Optional existing rank array to extend in place.
        maximum_fallback_tier: Highest fallback tier to rank. All tiers are
            ranked when omitted.
        tier_indices_by_tier: Optional precomputed candidate indices for each
            fallback tier.

    Returns:
        Integer rank per candidate. Unrequested tiers retain rank ``-1``.

    Raises:
        ValueError: If an optional rank or tier-index array is invalid.
        AssertionError: If a requested tier is partially ranked or remains
            unranked after processing.
    """
    if ranks is None:
        ranks = np.full(pool.probabilities.size, -1, dtype=np.int32)
    else:
        ranks = np.asarray(ranks)
        if ranks.shape != pool.probabilities.shape or ranks.dtype != np.int32:
            raise ValueError(
                "ranks must be an int32 array aligned with pool probabilities."
            )

    maximum_tier = (
        len(_CALENDAR_FALLBACK_NAMES) - 1
        if maximum_fallback_tier is None
        else int(maximum_fallback_tier)
    )
    if not 0 <= maximum_tier < len(_CALENDAR_FALLBACK_NAMES):
        raise ValueError(
            "maximum_fallback_tier must identify a configured fallback tier."
        )

    if tier_indices_by_tier is not None and len(tier_indices_by_tier) != len(
        _CALENDAR_FALLBACK_NAMES
    ):
        raise ValueError(
            "tier_indices_by_tier must contain one array per fallback tier."
        )

    for fallback_tier in range(maximum_tier + 1):
        tier_indices = (
            np.flatnonzero(pool.fallback_tiers == fallback_tier)
            if tier_indices_by_tier is None
            else tier_indices_by_tier[fallback_tier]
        )
        if not tier_indices.size:
            continue
        tier_ranks = ranks[tier_indices]
        if np.all(tier_ranks >= 0):
            continue
        if np.any(tier_ranks >= 0):
            raise AssertionError("A fallback tier has only partially filled ranks.")

        # A one-candidate tier has a deterministic order and needs no Generator.
        # For larger tiers, the final remaining candidate is also deterministic.
        # Skipping that last rng.choice call cannot affect any rank because every
        # tier deliberately starts a fresh legacy random stream.
        if tier_indices.size == 1:
            ranks[tier_indices[0]] = 0
            continue
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    random_seed,
                    farmer_id,
                    lookup_unit,
                    main_crop,
                    int(is_irrigated),
                ]
            )
        )
        remaining = tier_indices.copy()
        remaining_size = int(tier_indices.size)
        # A scalar ``Generator.choice(..., p=...)`` consumes one uniform draw
        # and selects with the normalized cumulative distribution. Drawing the
        # uniforms in one call preserves that exact PCG64 stream while avoiding
        # one Python-to-NumPy RNG call per rank.
        uniform_draws = rng.random(remaining_size - 1)
        for rank, uniform_draw in enumerate(uniform_draws):
            active_indices = remaining[:remaining_size]
            weights = pool.probabilities[active_indices]
            weight_sum = float(weights.sum())
            draw_probabilities = (
                weights / weight_sum
                if weight_sum > 0.0
                else np.full(remaining_size, 1.0 / remaining_size)
            )
            # This is the probability path used internally by NumPy's weighted
            # ``choice``. Keep both normalizations because their floating-point
            # rounding is part of the established seeded result.
            cumulative_probabilities = np.cumsum(draw_probabilities)
            cumulative_probabilities /= cumulative_probabilities[-1]
            draw_position = int(
                np.searchsorted(
                    cumulative_probabilities,
                    uniform_draw,
                    side="right",
                )
            )
            selected_index = int(remaining[draw_position])
            ranks[selected_index] = rank
            # Preserve the ascending remaining-index order used by np.delete,
            # but compact in place to avoid allocating a new array per draw.
            remaining[draw_position : remaining_size - 1] = remaining[
                draw_position + 1 : remaining_size
            ]
            remaining_size -= 1
        ranks[int(remaining[0])] = int(tier_indices.size - 1)

    requested = pool.fallback_tiers <= maximum_tier
    if np.any(ranks[requested] < 0):
        raise AssertionError("Not all requested MIRCA candidates received a rank.")
    return ranks


def _weighted_candidate_rank_zero(
    pool: _MIRCACalendarCandidatePool,
    *,
    random_seed: int,
    farmer_id: int,
    lookup_unit: int,
    main_crop: int,
    is_irrigated: bool,
    fallback_tier: int,
    tier_indices: np.ndarray | None = None,
) -> int:
    """Return only the first candidate in the legacy weighted tier order.

    The first draw is byte-for-byte the same RNG operation used by
    :func:`_weighted_candidate_ranks`. Most farmer sequences are feasible with
    this independently preferred candidate in every year, so constructing the
    rest of a potentially very large cross-unit order would not affect the
    selected calendars. The complete order is still built if that preferred
    path is temporally infeasible.

    Args:
        pool: Cached calendar candidates and area probabilities.
        random_seed: Base seed for reproducible farmer selection.
        farmer_id: Compact farmer identifier included in the RNG seed.
        lookup_unit: MIRCA-OS unit used for calendar lookup.
        main_crop: MIRCA crop class used for calendar lookup.
        is_irrigated: Whether irrigated calendars are preferred.
        fallback_tier: Tier from which to draw the first candidate.
        tier_indices: Optional precomputed indices belonging to that tier.

    Returns:
        Pool index of the first candidate in the weighted preference order.

    Raises:
        ValueError: If the requested tier contains no candidates.
    """
    if tier_indices is None:
        tier_indices = np.flatnonzero(pool.fallback_tiers == fallback_tier)
    else:
        tier_indices = np.asarray(tier_indices)
    if not tier_indices.size:
        raise ValueError("Cannot draw a preferred candidate from an empty tier.")
    if tier_indices.size == 1:
        return int(tier_indices[0])

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                random_seed,
                farmer_id,
                lookup_unit,
                main_crop,
                int(is_irrigated),
            ]
        )
    )
    weights = pool.probabilities[tier_indices]
    weight_sum = float(weights.sum())
    draw_probabilities = (
        weights / weight_sum
        if weight_sum > 0.0
        else np.full(tier_indices.size, 1.0 / tier_indices.size)
    )
    cumulative_probabilities = np.cumsum(draw_probabilities)
    cumulative_probabilities /= cumulative_probabilities[-1]
    draw_position = int(
        np.searchsorted(
            cumulative_probabilities,
            rng.random(),
            side="right",
        )
    )
    return int(tier_indices[draw_position])


def _mirca_calendar_pool_tier_metadata(
    pool: _MIRCACalendarCandidatePool,
) -> _MIRCACalendarPoolTierMetadata:
    """Build tier and cumulative eligible indices once for a shared pool.

    Args:
        pool: Shared MIRCA calendar candidate pool.

    Returns:
        Reusable tier indices, eligible indices, and tier bounds.

    Raises:
        ValueError: If the candidate pool is empty.
    """
    tier_indices = tuple(
        np.flatnonzero(pool.fallback_tiers == tier).astype(
            np.int32,
            copy=False,
        )
        for tier in range(len(_CALENDAR_FALLBACK_NAMES))
    )
    available_tiers = tuple(
        tier for tier, indices in enumerate(tier_indices) if indices.size
    )
    if not available_tiers:
        raise ValueError("Cannot build tier metadata for an empty candidate pool.")
    eligible_indices = tuple(
        np.flatnonzero(pool.fallback_tiers <= maximum_tier).astype(
            np.int32,
            copy=False,
        )
        for maximum_tier in range(len(_CALENDAR_FALLBACK_NAMES))
    )
    return _MIRCACalendarPoolTierMetadata(
        minimum_tier=available_tiers[0],
        available_tiers=available_tiers,
        tier_indices=tier_indices,
        eligible_indices=eligible_indices,
    )


def _best_lexicographic_index(
    maximum_fallback_tier: np.ndarray,
    fallback_tier_sum: np.ndarray,
    preference_rank_sum: np.ndarray,
) -> int:
    """Return the index minimizing the three sequence costs in priority order."""
    return int(
        np.lexsort(
            (
                preference_rank_sum,
                fallback_tier_sum,
                maximum_fallback_tier,
            )
        )[0]
    )


def _calendar_failure_diagnostic(
    *,
    farmer_id: int,
    failed_year: int,
    previous_active_year: int | None,
    lookup_unit: int,
    main_crop: int,
    is_irrigated: bool,
    reason: str,
    pool: _MIRCACalendarCandidatePool,
    reachable_previous_harvest_absolute: np.ndarray | None = None,
) -> dict[str, Any]:
    """Create one compact, log-friendly sequence failure record."""
    current_year_start = _year_start_day(failed_year)
    candidate_starts_by_tier = {
        _CALENDAR_FALLBACK_NAMES[tier]: np.unique(
            pool.earliest_planting[pool.fallback_tiers == tier]
        )
        .astype(int)
        .tolist()
        for tier in range(len(_CALENDAR_FALLBACK_NAMES))
        if np.any(pool.fallback_tiers == tier)
    }
    previous_days = (
        np.asarray(reachable_previous_harvest_absolute, dtype=np.int64)
        - current_year_start
        if reachable_previous_harvest_absolute is not None
        else np.empty(0, dtype=np.int64)
    )
    return {
        "farmer_id": int(farmer_id),
        "failed_year": int(failed_year),
        "previous_active_year": (
            None if previous_active_year is None else int(previous_active_year)
        ),
        "unit": int(lookup_unit),
        "crop": int(main_crop),
        "irrigated": bool(is_irrigated),
        "reason": reason,
        "candidate_count": int(pool.calendars.shape[0]),
        "candidate_starts_by_tier": candidate_starts_by_tier,
        "reachable_previous_harvest_day_min": (
            None if not previous_days.size else int(previous_days.min())
        ),
        "reachable_previous_harvest_day_max": (
            None if not previous_days.size else int(previous_days.max())
        ),
    }


def _lexicographic_prefix_best_predecessors(
    *,
    sorted_previous_positions: np.ndarray,
    maximum_tier: np.ndarray,
    tier_sum: np.ndarray,
    rank_sum: np.ndarray,
    candidate_tier: int,
) -> np.ndarray:
    """Return the cheapest predecessor in every harvest-sorted prefix.

    For a current candidate, temporal feasibility is a one-dimensional cutoff:
    every predecessor harvested no later than its planting day is feasible.
    Keeping the lexicographic prefix minimum lets all current candidates in one
    fallback tier obtain their best predecessor with one ``searchsorted`` call,
    rather than materializing a candidate-by-predecessor Boolean matrix and
    running ``lexsort`` separately for every current candidate.

    The predecessor position is the final tie-breaker. This exactly matches the
    former stable selection from ascending ``flatnonzero`` positions.
    """
    sorted_previous_positions = np.asarray(
        sorted_previous_positions,
        dtype=np.int32,
    )
    prefix_best = np.empty(sorted_previous_positions.size, dtype=np.int32)
    best_position = -1
    best_cost: tuple[int, int, int, int] | None = None

    for prefix_position, predecessor_position_value in enumerate(
        sorted_previous_positions
    ):
        predecessor_position = int(predecessor_position_value)
        cost = (
            max(int(maximum_tier[predecessor_position]), int(candidate_tier)),
            int(tier_sum[predecessor_position]),
            int(rank_sum[predecessor_position]),
            predecessor_position,
        )
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_position = predecessor_position
        prefix_best[prefix_position] = best_position

    return prefix_best


def _preferred_mirca_calendar_path_if_feasible(
    *,
    farmer_pools: list[_MIRCACalendarCandidatePool],
    farmer_ranks: list[np.ndarray],
    active_year_indices: np.ndarray,
    years: np.ndarray,
    preferred_indices: np.ndarray | None = None,
    year_start_days: np.ndarray | None = None,
) -> tuple[np.ndarray | None, int]:
    """Return the independently preferred path when it is temporally feasible.

    For every active year, the rank-zero candidate in the lowest available tier
    has the smallest possible contribution to all three lexicographic costs.
    If those independently optimal candidates already form a feasible temporal
    sequence, no dynamic-programming search can improve on that path. This is
    the common unconstrained case and needs only one transition check per pair
    of active years.

    Args:
        farmer_pools: Candidate pool for every active farmer year.
        farmer_ranks: Candidate ranks for every active year. These are used
            only when ``preferred_indices`` is omitted.
        active_year_indices: Positions of active crops in the HRL year array.
        years: Strictly increasing HRL years.
        preferred_indices: Optional precomputed preferred pool index for every
            active year.
        year_start_days: Optional NumPy day ordinal for 1 January of each HRL
            year.

    Returns:
        Preferred candidate indices when the path is feasible, otherwise
        ``None``, plus the number of temporal transition checks.

    Raises:
        ValueError: If optional precomputed arrays do not align with the years.
        AssertionError: If ranks do not identify exactly one preferred
            candidate in a lowest available tier.
    """
    if preferred_indices is None:
        selected_indices = np.empty(active_year_indices.size, dtype=np.int32)
        for active_position, pool in enumerate(farmer_pools):
            minimum_tier = int(pool.fallback_tiers.min())
            preferred = np.flatnonzero(
                (pool.fallback_tiers == minimum_tier)
                & (farmer_ranks[active_position] == 0)
            )
            if preferred.size != 1:
                raise AssertionError(
                    "Every lowest fallback tier must have exactly one rank-zero "
                    "calendar candidate."
                )
            selected_indices[active_position] = int(preferred[0])
    else:
        selected_indices = np.asarray(preferred_indices, dtype=np.int32)
        if selected_indices.shape != active_year_indices.shape:
            raise ValueError(
                "preferred_indices must contain one candidate per active year."
            )

    if year_start_days is None:
        year_start_days = np.asarray(
            [_year_start_day(int(year)) for year in years],
            dtype=np.int64,
        )
    else:
        year_start_days = np.asarray(year_start_days, dtype=np.int64)
        if year_start_days.shape != years.shape:
            raise ValueError("year_start_days must align with years.")

    transition_checks = max(int(active_year_indices.size) - 1, 0)
    for active_position in range(1, active_year_indices.size):
        previous_year_index = int(active_year_indices[active_position - 1])
        current_year_index = int(active_year_indices[active_position])
        previous_candidate = int(selected_indices[active_position - 1])
        current_candidate = int(selected_indices[active_position])
        previous_harvest_absolute = int(year_start_days[previous_year_index]) + int(
            farmer_pools[active_position - 1].latest_harvest[previous_candidate]
        )
        current_planting_absolute = int(year_start_days[current_year_index]) + int(
            farmer_pools[active_position].earliest_planting[current_candidate]
        )
        if current_planting_absolute < previous_harvest_absolute:
            return None, transition_checks

    return selected_indices, transition_checks


def _solve_mirca_calendar_path(
    *,
    farmer_pools: list[_MIRCACalendarCandidatePool],
    farmer_ranks: list[np.ndarray],
    active_year_indices: np.ndarray,
    years: np.ndarray,
    maximum_allowed_tier: int,
    eligible_indices: list[np.ndarray] | None = None,
    year_start_days: np.ndarray | None = None,
) -> tuple[np.ndarray | None, int | None, np.ndarray | None, int]:
    """Solve one farmer path using candidates up to one fallback tier.

    Trying increasingly broad maximum tiers keeps the common local-only case
    small and fast. The first successful call also minimizes the worst fallback
    tier before the dynamic-programming cost compares tier and preference sums.

    Args:
        farmer_pools: Candidate pool for every active farmer year.
        farmer_ranks: Complete candidate ranks through the allowed tier.
        active_year_indices: Positions of active crops in the HRL year array.
        years: Strictly increasing HRL years.
        maximum_allowed_tier: Highest fallback tier included in the path.
        eligible_indices: Optional precomputed eligible pool indices for every
            active year.
        year_start_days: Optional NumPy day ordinal for 1 January of each HRL
            year.

    Returns:
        Selected candidate indices when a path exists; otherwise the failed
        active position and reachable previous harvest days. The final integer
        is the number of candidate transition edges represented by the search.

    Raises:
        ValueError: If optional precomputed arrays are misaligned.
        AssertionError: If an eligible candidate has no preference rank.
    """
    if eligible_indices is None:
        eligible_indices = [
            np.flatnonzero(pool.fallback_tiers <= maximum_allowed_tier)
            for pool in farmer_pools
        ]
    elif len(eligible_indices) != len(farmer_pools):
        raise ValueError("eligible_indices must contain one array per active year.")
    if year_start_days is None:
        year_start_days = np.asarray(
            [_year_start_day(int(year)) for year in years],
            dtype=np.int64,
        )
    else:
        year_start_days = np.asarray(year_start_days, dtype=np.int64)
        if year_start_days.shape != years.shape:
            raise ValueError("year_start_days must align with years.")
    for active_position, indices in enumerate(eligible_indices):
        if not indices.size:
            return None, active_position, None, 0

    first_indices = eligible_indices[0]
    first_pool = farmer_pools[0]
    maximum_tier = first_pool.fallback_tiers[first_indices].astype(np.int16)
    tier_sum = first_pool.fallback_tiers[first_indices].astype(np.int32)
    rank_sum = farmer_ranks[0][first_indices].astype(np.int32)
    if np.any(rank_sum < 0):
        raise AssertionError("Eligible first-year candidates have no preference rank.")
    backpointers: list[np.ndarray] = [np.full(first_indices.size, -1, dtype=np.int32)]
    transitions_evaluated = 0

    for active_position in range(1, active_year_indices.size):
        previous_pool = farmer_pools[active_position - 1]
        current_pool = farmer_pools[active_position]
        previous_indices = eligible_indices[active_position - 1]
        current_indices = eligible_indices[active_position]
        previous_year_index = int(active_year_indices[active_position - 1])
        current_year_index = int(active_year_indices[active_position])
        previous_harvest_absolute = (
            int(year_start_days[previous_year_index])
            + previous_pool.latest_harvest[previous_indices]
        )
        current_planting_absolute = (
            int(year_start_days[current_year_index])
            + current_pool.earliest_planting[current_indices]
        )
        transitions_evaluated += int(current_indices.size * previous_indices.size)

        n_current_candidates = current_indices.size
        unreachable_tier = np.iinfo(np.int16).max
        next_maximum_tier = np.full(
            n_current_candidates,
            unreachable_tier,
            dtype=np.int16,
        )
        next_tier_sum = np.full(
            n_current_candidates,
            np.iinfo(np.int32).max,
            dtype=np.int32,
        )
        next_rank_sum = np.full(
            n_current_candidates,
            np.iinfo(np.int32).max,
            dtype=np.int32,
        )
        current_backpointer = np.full(
            n_current_candidates,
            -1,
            dtype=np.int32,
        )
        reachable_previous = maximum_tier != unreachable_tier
        reachable_positions = np.flatnonzero(reachable_previous).astype(
            np.int32,
            copy=False,
        )
        if reachable_positions.size:
            harvest_order = np.argsort(
                previous_harvest_absolute[reachable_positions],
                kind="stable",
            )
            sorted_previous_positions = reachable_positions[harvest_order]
            sorted_previous_harvest = previous_harvest_absolute[
                sorted_previous_positions
            ]
            current_tiers = current_pool.fallback_tiers[current_indices]
            current_ranks = farmer_ranks[active_position][current_indices]
            if np.any(current_ranks < 0):
                raise AssertionError(
                    "Eligible current-year candidates have no preference rank."
                )

            for candidate_tier_value in np.unique(current_tiers):
                candidate_tier = int(candidate_tier_value)
                current_positions = np.flatnonzero(
                    current_tiers == candidate_tier_value
                )
                feasible_prefix_end = (
                    np.searchsorted(
                        sorted_previous_harvest,
                        current_planting_absolute[current_positions],
                        side="right",
                    )
                    - 1
                )
                feasible_current = feasible_prefix_end >= 0
                if not np.any(feasible_current):
                    continue

                prefix_best = _lexicographic_prefix_best_predecessors(
                    sorted_previous_positions=sorted_previous_positions,
                    maximum_tier=maximum_tier,
                    tier_sum=tier_sum,
                    rank_sum=rank_sum,
                    candidate_tier=candidate_tier,
                )
                feasible_positions = current_positions[feasible_current]
                predecessor_positions = prefix_best[
                    feasible_prefix_end[feasible_current]
                ]
                current_backpointer[feasible_positions] = predecessor_positions
                next_maximum_tier[feasible_positions] = np.maximum(
                    maximum_tier[predecessor_positions],
                    candidate_tier,
                )
                next_tier_sum[feasible_positions] = (
                    tier_sum[predecessor_positions] + candidate_tier
                )
                next_rank_sum[feasible_positions] = (
                    rank_sum[predecessor_positions] + current_ranks[feasible_positions]
                )

        if not np.any(current_backpointer >= 0):
            return (
                None,
                active_position,
                previous_harvest_absolute[reachable_previous],
                transitions_evaluated,
            )

        backpointers.append(current_backpointer)
        maximum_tier = next_maximum_tier
        tier_sum = next_tier_sum
        rank_sum = next_rank_sum

    reachable_final = maximum_tier != np.iinfo(np.int16).max
    final_positions = np.flatnonzero(reachable_final)
    final_best = _best_lexicographic_index(
        maximum_tier[final_positions],
        tier_sum[final_positions],
        rank_sum[final_positions],
    )
    selected_positions = np.full(active_year_indices.size, -1, dtype=np.int32)
    selected_positions[-1] = int(final_positions[final_best])
    for active_position in range(active_year_indices.size - 1, 0, -1):
        selected_positions[active_position - 1] = backpointers[active_position][
            selected_positions[active_position]
        ]
    selected_indices = np.asarray(
        [
            eligible_indices[position][selected_positions[position]]
            for position in range(active_year_indices.size)
        ],
        dtype=np.int32,
    )
    return selected_indices, None, None, transitions_evaluated


def _select_mirca_calendar_sequences_for_farmers_serial(
    crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
    *,
    farmer_mirca_units: np.ndarray,
    farmer_main_crops_by_year: np.ndarray,
    farmer_is_irrigated_by_year: np.ndarray,
    hrl_years: np.ndarray | tuple[int, ...],
    replace_crop_calendar_unit_code: dict[int, int],
    random_seed: int,
    candidate_pool_cache: dict[
        tuple[int, int, bool],
        _MIRCACalendarCandidatePool,
    ]
    | None = None,
    logger: Any | None = None,
    progress_interval: int = 100_000,
    farmer_id_offset: int = 0,
    calendar_index: _MIRCACalendarIndex | None = None,
    candidate_pool_tier_cache: dict[
        tuple[int, int, bool],
        _MIRCACalendarPoolTierMetadata,
    ]
    | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Select complete farmer calendar sequences with dynamic programming.

    All fallback tiers are cached once per ``(unit, crop, irrigation)`` state.
    For each farmer, dynamic programming selects a globally feasible path over
    active HRL years. The objective first minimizes the worst fallback tier,
    then the sum of fallback tiers, and finally the deterministic area-weighted
    preference ranks. This allows an earlier calendar to be reconsidered when
    its harvest would otherwise make a later observed crop impossible.

    Fallow years are omitted from the path but retained in the output. Their
    omission deliberately preserves the last active harvest constraint across
    one or more fallow years.

    Args:
        crop_calendar: Parsed MIRCA-OS calendar entries by administrative unit.
        farmer_mirca_units: MIRCA-OS unit per compact farmer.
        farmer_main_crops_by_year: MIRCA crop class by HRL year and farmer.
        farmer_is_irrigated_by_year: Irrigation state by HRL year and farmer.
        hrl_years: Strictly increasing HRL years represented by axis zero.
        replace_crop_calendar_unit_code: Optional MIRCA unit replacements.
        random_seed: Base seed for reproducible weighted candidate ordering.
        candidate_pool_cache: Optional shared candidate pools keyed by farmer
            calendar state.
        logger: Optional logger for periodic throughput and memory diagnostics.
        progress_interval: Farmers processed between progress messages.
        farmer_id_offset: Global ID of the first farmer in this input slice.
        calendar_index: Optional prebuilt MIRCA calendar lookup index.
        candidate_pool_tier_cache: Optional prebuilt fallback-tier metadata.

    Returns:
        Calendar stack, aggregated farmer failure records, and selector
        statistics. The calendar stack has shape
        ``(year, farmer, calendar_row, calendar_variable)``.

    Raises:
        ValueError: If input arrays, years, or progress settings are invalid.
        AssertionError: If an internal path or rank invariant is violated.
    """
    farmer_mirca_units = np.asarray(farmer_mirca_units, dtype=np.int32)
    crops = np.asarray(farmer_main_crops_by_year, dtype=np.int32)
    irrigation = np.asarray(farmer_is_irrigated_by_year, dtype=bool)
    years = np.asarray(hrl_years, dtype=np.int32)
    if crops.ndim != 2:
        raise ValueError("farmer_main_crops_by_year must have shape (year, farmer).")
    if irrigation.shape != crops.shape:
        raise ValueError(
            "farmer_is_irrigated_by_year must match farmer_main_crops_by_year."
        )
    if years.ndim != 1 or years.size != crops.shape[0]:
        raise ValueError("hrl_years must match the first crop-calendar axis.")
    if farmer_mirca_units.ndim != 1 or farmer_mirca_units.size != crops.shape[1]:
        raise ValueError("farmer_mirca_units must contain one value per farmer.")
    if years.size > 1 and np.any(np.diff(years) <= 0):
        raise ValueError("hrl_years must be strictly increasing.")
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1.")
    if farmer_id_offset < 0:
        raise ValueError("farmer_id_offset must be non-negative.")
    if candidate_pool_cache is None:
        candidate_pool_cache = {}
    if calendar_index is None:
        calendar_index = _index_mirca_calendars(crop_calendar)
    year_start_days = np.asarray(
        [_year_start_day(int(year)) for year in years],
        dtype=np.int64,
    )

    n_years, n_farmers = crops.shape
    calendar_stack = np.full(
        (n_years, n_farmers, 3, 4),
        -1,
        dtype=np.int32,
    )
    failures: list[dict[str, Any]] = []
    selected_tier_counts = np.zeros(len(_CALENDAR_FALLBACK_NAMES), dtype=np.int64)
    maximum_tier_attempt_counts = np.zeros(
        len(_CALENDAR_FALLBACK_NAMES),
        dtype=np.int64,
    )
    farmers_using_fallback = 0
    farmers_reconsidering_earlier_choice = 0
    transitions_evaluated = 0
    preferred_path_transition_checks = 0
    preferred_path_fast_path_farmers = 0
    preference_candidates_ranked = 0
    preference_candidates_available = 0
    preference_order_count = 0
    full_preference_orders_built = 0
    peak_farmer_preference_cache_count = 0
    if candidate_pool_tier_cache is None:
        candidate_pool_tier_cache = {}
    starting_tier_farmer_counts = np.zeros(
        len(_CALENDAR_FALLBACK_NAMES),
        dtype=np.int64,
    )
    rank_order_seconds = 0.0
    dynamic_programming_seconds = 0.0
    selector_started = time.perf_counter()
    farmer_lookup_units = farmer_mirca_units.copy()
    for original_unit, replacement_unit in replace_crop_calendar_unit_code.items():
        farmer_lookup_units[farmer_mirca_units == int(original_unit)] = int(
            replacement_unit
        )

    def log_selector_progress(processed_farmers: int) -> None:
        if logger is None:
            return
        elapsed = time.perf_counter() - selector_started
        rate = processed_farmers / elapsed if elapsed > 0.0 else 0.0
        rss_mib, hwm_mib = _process_memory_mib()
        logger.info(
            "Joint HRL calendar selection progress: %s/%s farmers (%.1f%%) "
            "in %.1f s (%.0f farmers/s); preferred fast paths=%s; full "
            "preference orders=%s; ranked candidates=%s/%s; DP edges=%s; "
            "rank/DP time=%.1f/%.1f s; failures=%s; starting tiers=%s; "
            "RSS=%s MiB; HWM=%s MiB.",
            processed_farmers,
            n_farmers,
            processed_farmers / n_farmers * 100.0 if n_farmers else 100.0,
            elapsed,
            rate,
            preferred_path_fast_path_farmers,
            full_preference_orders_built,
            preference_candidates_ranked,
            preference_candidates_available,
            transitions_evaluated,
            rank_order_seconds,
            dynamic_programming_seconds,
            len(failures),
            {
                name: int(starting_tier_farmer_counts[index])
                for index, name in enumerate(_CALENDAR_FALLBACK_NAMES)
            },
            "n/a" if rss_mib is None else f"{rss_mib:.1f}",
            "n/a" if hwm_mib is None else f"{hwm_mib:.1f}",
        )

    next_progress = progress_interval
    for farmer_id in range(n_farmers):
        if farmer_id == next_progress:
            log_selector_progress(farmer_id)
            next_progress += progress_interval

        global_farmer_id = farmer_id_offset + farmer_id
        active_year_indices = np.flatnonzero(crops[:, farmer_id] != -1)
        if not active_year_indices.size:
            continue

        farmer_pools: list[_MIRCACalendarCandidatePool] = []
        farmer_ranks: list[np.ndarray] = []
        farmer_states: list[tuple[int, int, bool]] = []
        farmer_tier_metadata: list[_MIRCACalendarPoolTierMetadata] = []
        farmer_unique_states: list[tuple[int, int, bool]] = []
        farmer_pool_by_state: dict[
            tuple[int, int, bool],
            _MIRCACalendarCandidatePool,
        ] = {}
        farmer_metadata_by_state: dict[
            tuple[int, int, bool],
            _MIRCACalendarPoolTierMetadata,
        ] = {}
        farmer_preferred_index_by_state: dict[tuple[int, int, bool], int] = {}
        farmer_full_ranks_by_state: dict[tuple[int, int, bool], np.ndarray] = {}
        farmer_minimum_tiers: list[int] = []
        farmer_available_tiers: set[int] = set()
        missing_pool = False
        lookup_unit = int(farmer_lookup_units[farmer_id])
        for year_index in active_year_indices:
            main_crop = int(crops[year_index, farmer_id])
            is_irrigated = bool(irrigation[year_index, farmer_id])
            state_key = (lookup_unit, main_crop, is_irrigated)
            pool = candidate_pool_cache.get(state_key)
            if pool is None:
                pool = _build_mirca_calendar_candidate_pool(
                    crop_calendar,
                    lookup_unit=lookup_unit,
                    main_crop=main_crop,
                    is_irrigated=is_irrigated,
                    calendar_index=calendar_index,
                )
                candidate_pool_cache[state_key] = pool
            if not pool.calendars.shape[0]:
                failures.append(
                    _calendar_failure_diagnostic(
                        farmer_id=global_farmer_id,
                        failed_year=int(years[year_index]),
                        previous_active_year=None,
                        lookup_unit=lookup_unit,
                        main_crop=main_crop,
                        is_irrigated=is_irrigated,
                        reason="no_same_crop_candidate_in_any_fallback_tier",
                        pool=pool,
                    )
                )
                missing_pool = True
                break
            tier_metadata = candidate_pool_tier_cache.get(state_key)
            if tier_metadata is None:
                tier_metadata = _mirca_calendar_pool_tier_metadata(pool)
                candidate_pool_tier_cache[state_key] = tier_metadata
            farmer_minimum_tiers.append(tier_metadata.minimum_tier)
            farmer_available_tiers.update(tier_metadata.available_tiers)

            if state_key not in farmer_preferred_index_by_state:
                preferred_index = _weighted_candidate_rank_zero(
                    pool,
                    random_seed=random_seed,
                    farmer_id=global_farmer_id,
                    lookup_unit=lookup_unit,
                    main_crop=main_crop,
                    is_irrigated=is_irrigated,
                    fallback_tier=tier_metadata.minimum_tier,
                    tier_indices=tier_metadata.tier_indices[tier_metadata.minimum_tier],
                )
                farmer_unique_states.append(state_key)
                farmer_pool_by_state[state_key] = pool
                farmer_metadata_by_state[state_key] = tier_metadata
                farmer_preferred_index_by_state[state_key] = preferred_index
                preference_order_count += 1
                preference_candidates_available += int(pool.probabilities.size)
                preference_candidates_ranked += 1
            farmer_pools.append(pool)
            farmer_states.append(state_key)
            farmer_tier_metadata.append(tier_metadata)
        peak_farmer_preference_cache_count = max(
            peak_farmer_preference_cache_count,
            len(farmer_unique_states),
        )
        if missing_pool:
            continue

        starting_tier = max(farmer_minimum_tiers)
        starting_tier_farmer_counts[starting_tier] += 1
        tiers_to_try = sorted(
            tier for tier in farmer_available_tiers if tier >= starting_tier
        )
        selected_indices: np.ndarray | None = None
        failure_position: int | None = None
        reachable_previous_harvest: np.ndarray | None = None
        for maximum_allowed_tier in tiers_to_try:
            maximum_tier_attempt_counts[maximum_allowed_tier] += 1
            if maximum_allowed_tier == starting_tier:
                preferred_indices = np.asarray(
                    [farmer_preferred_index_by_state[state] for state in farmer_states],
                    dtype=np.int32,
                )
                (
                    preferred_indices,
                    preferred_checks,
                ) = _preferred_mirca_calendar_path_if_feasible(
                    farmer_pools=farmer_pools,
                    farmer_ranks=[],
                    active_year_indices=active_year_indices,
                    years=years,
                    preferred_indices=preferred_indices,
                    year_start_days=year_start_days,
                )
                preferred_path_transition_checks += preferred_checks
                if preferred_indices is not None:
                    selected_indices = preferred_indices
                    preferred_path_fast_path_farmers += 1
                    break

            # The independently preferred path conflicted. Build exactly the
            # same complete weighted orders as before, but retain them only for
            # this farmer. A wider fallback attempt extends these local orders.
            ranking_started = time.perf_counter()
            for state in farmer_unique_states:
                pool = farmer_pool_by_state[state]
                tier_metadata = farmer_metadata_by_state[state]
                ranks = farmer_full_ranks_by_state.get(state)
                first_full_order = ranks is None
                if ranks is None:
                    ranks = np.full(pool.probabilities.size, -1, dtype=np.int32)
                    farmer_full_ranks_by_state[state] = ranks
                    full_preference_orders_built += 1
                ranked_before = int(np.count_nonzero(ranks >= 0))
                state_lookup_unit, main_crop, is_irrigated = state
                _weighted_candidate_ranks(
                    pool,
                    random_seed=random_seed,
                    farmer_id=global_farmer_id,
                    lookup_unit=state_lookup_unit,
                    main_crop=main_crop,
                    is_irrigated=is_irrigated,
                    ranks=ranks,
                    maximum_fallback_tier=maximum_allowed_tier,
                    tier_indices_by_tier=tier_metadata.tier_indices,
                )
                newly_ranked = int(np.count_nonzero(ranks >= 0)) - ranked_before
                # The rank-zero candidate was already counted when the exact
                # fast-path draw was made. A full order recomputes that draw to
                # preserve the legacy RNG stream, but it is still one logical
                # candidate preference rather than an additional candidate.
                preference_candidates_ranked += max(
                    newly_ranked - int(first_full_order),
                    0,
                )
            rank_order_seconds += time.perf_counter() - ranking_started
            farmer_ranks = [
                farmer_full_ranks_by_state[state] for state in farmer_states
            ]
            dynamic_programming_started = time.perf_counter()
            (
                selected_indices,
                failure_position,
                reachable_previous_harvest,
                evaluated_edges,
            ) = _solve_mirca_calendar_path(
                farmer_pools=farmer_pools,
                farmer_ranks=farmer_ranks,
                active_year_indices=active_year_indices,
                years=years,
                maximum_allowed_tier=maximum_allowed_tier,
                eligible_indices=[
                    metadata.eligible_indices[maximum_allowed_tier]
                    for metadata in farmer_tier_metadata
                ],
                year_start_days=year_start_days,
            )
            dynamic_programming_seconds += (
                time.perf_counter() - dynamic_programming_started
            )
            transitions_evaluated += evaluated_edges
            if selected_indices is not None:
                break

        if selected_indices is None:
            if failure_position is None:
                raise AssertionError("Missing failed calendar sequence position.")
            current_year_index = int(active_year_indices[failure_position])
            previous_year_index = (
                None
                if failure_position == 0
                else int(active_year_indices[failure_position - 1])
            )
            lookup_unit, main_crop, is_irrigated = farmer_states[failure_position]
            failures.append(
                _calendar_failure_diagnostic(
                    farmer_id=global_farmer_id,
                    failed_year=int(years[current_year_index]),
                    previous_active_year=(
                        None
                        if previous_year_index is None
                        else int(years[previous_year_index])
                    ),
                    lookup_unit=lookup_unit,
                    main_crop=main_crop,
                    is_irrigated=is_irrigated,
                    reason="no_feasible_complete_sequence_path",
                    pool=farmer_pools[failure_position],
                    reachable_previous_harvest_absolute=(reachable_previous_harvest),
                )
            )
            continue

        used_fallback = False
        reconsidered_earlier = False
        for active_position, year_index in enumerate(active_year_indices):
            pool = farmer_pools[active_position]
            candidate_index = int(selected_indices[active_position])
            calendar_stack[year_index, farmer_id] = pool.calendars[candidate_index]
            selected_tier = int(pool.fallback_tiers[candidate_index])
            selected_tier_counts[selected_tier] += 1
            used_fallback |= selected_tier > 0
            if active_position < active_year_indices.size - 1:
                minimum_available_tier = int(pool.fallback_tiers.min())
                selected_rank = (
                    int(farmer_ranks[active_position][candidate_index])
                    if farmer_ranks
                    else 0
                )
                reconsidered_earlier |= (
                    selected_tier > minimum_available_tier or selected_rank > 0
                )
        farmers_using_fallback += int(used_fallback)
        farmers_reconsidering_earlier_choice += int(reconsidered_earlier)

    if n_farmers:
        log_selector_progress(n_farmers)

    statistics: dict[str, Any] = {
        "candidate_pool_count": len(candidate_pool_cache),
        "preference_cache_count": preference_order_count,
        "peak_farmer_preference_cache_count": peak_farmer_preference_cache_count,
        "full_preference_orders_built": full_preference_orders_built,
        "preference_candidates_ranked": preference_candidates_ranked,
        "preference_candidates_available": preference_candidates_available,
        "rank_order_seconds": rank_order_seconds,
        "dynamic_programming_seconds": dynamic_programming_seconds,
        "transitions_evaluated": transitions_evaluated,
        "preferred_path_transition_checks": preferred_path_transition_checks,
        "preferred_path_fast_path_farmers": preferred_path_fast_path_farmers,
        "farmers_using_fallback": farmers_using_fallback,
        "farmers_reconsidering_earlier_choice": (farmers_reconsidering_earlier_choice),
        "selected_tier_counts": {
            name: int(selected_tier_counts[index])
            for index, name in enumerate(_CALENDAR_FALLBACK_NAMES)
        },
        "maximum_tier_attempt_counts": {
            name: int(maximum_tier_attempt_counts[index])
            for index, name in enumerate(_CALENDAR_FALLBACK_NAMES)
        },
        "starting_tier_farmer_counts": {
            name: int(starting_tier_farmer_counts[index])
            for index, name in enumerate(_CALENDAR_FALLBACK_NAMES)
        },
    }
    return calendar_stack, failures, statistics


@dataclass(slots=True)
class _MIRCACalendarSequenceWorkerContext:
    """Read-only selector inputs inherited by forked farmer workers."""

    crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]]
    farmer_mirca_units: np.ndarray
    crops: np.ndarray
    irrigation: np.ndarray
    years: np.ndarray
    replace_crop_calendar_unit_code: dict[int, int]
    random_seed: int
    candidate_pool_cache: dict[
        tuple[int, int, bool],
        _MIRCACalendarCandidatePool,
    ]
    calendar_index: _MIRCACalendarIndex
    candidate_pool_tier_cache: dict[
        tuple[int, int, bool],
        _MIRCACalendarPoolTierMetadata,
    ]


_MIRCA_CALENDAR_SEQUENCE_WORKER_CONTEXT: _MIRCACalendarSequenceWorkerContext | None = (
    None
)


def _default_mirca_calendar_sequence_workers() -> int:
    """Return a conservative worker count from the current Slurm allocation.

    ``GEB_HRL_CALENDAR_WORKERS`` can explicitly override the automatically
    detected ``SLURM_CPUS_PER_TASK`` value. Outside Slurm the selector remains
    serial unless the override is set. At most eight workers are selected
    automatically to limit memory-bandwidth contention.

    Returns:
        Number of worker processes to use.

    Raises:
        ValueError: If an explicit worker override is not a positive integer.
    """
    explicit_workers = os.environ.get("GEB_HRL_CALENDAR_WORKERS")
    if explicit_workers is not None:
        try:
            worker_count = int(explicit_workers)
        except ValueError as error:
            raise ValueError(
                "GEB_HRL_CALENDAR_WORKERS must be a positive integer."
            ) from error
        if worker_count < 1:
            raise ValueError("GEB_HRL_CALENDAR_WORKERS must be a positive integer.")
        return worker_count

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is None:
        return 1
    try:
        allocated_cpus = int(slurm_cpus)
    except ValueError:
        return 1
    return min(max(allocated_cpus, 1), 8)


def _mirca_calendar_sequence_state_keys(
    *,
    farmer_lookup_units: np.ndarray,
    crops: np.ndarray,
    irrigation: np.ndarray,
    farmer_chunk_size: int = 500_000,
) -> set[tuple[int, int, bool]]:
    """Collect unique farmer calendar states with bounded temporary memory.

    Args:
        farmer_lookup_units: Final MIRCA-OS lookup unit per farmer.
        crops: MIRCA crop class by HRL year and farmer.
        irrigation: Irrigation state by HRL year and farmer.
        farmer_chunk_size: Farmers scanned per temporary integer-key chunk.

    Returns:
        Unique ``(unit, crop, irrigation)`` states used by active farmer years.

    Raises:
        ValueError: If arrays are misaligned or the chunk size is invalid.
    """
    farmer_lookup_units = np.asarray(farmer_lookup_units, dtype=np.int32)
    crops = np.asarray(crops, dtype=np.int32)
    irrigation = np.asarray(irrigation, dtype=bool)
    if crops.ndim != 2 or irrigation.shape != crops.shape:
        raise ValueError("Crop and irrigation arrays must have equal 2-D shapes.")
    if farmer_lookup_units.shape != (crops.shape[1],):
        raise ValueError("farmer_lookup_units must contain one value per farmer.")
    if farmer_chunk_size < 1:
        raise ValueError("farmer_chunk_size must be at least 1.")

    minimum_active_crop: int | None = None
    maximum_active_crop: int | None = None
    for year_index in range(crops.shape[0]):
        year_crops = crops[year_index]
        active = year_crops != -1
        if not np.any(active):
            continue
        active_year_crops = year_crops[active]
        year_minimum_crop = int(active_year_crops.min())
        year_maximum_crop = int(active_year_crops.max())
        minimum_active_crop = (
            year_minimum_crop
            if minimum_active_crop is None
            else min(minimum_active_crop, year_minimum_crop)
        )
        maximum_active_crop = (
            year_maximum_crop
            if maximum_active_crop is None
            else max(maximum_active_crop, year_maximum_crop)
        )
    crop_offset = 0 if minimum_active_crop is None else -minimum_active_crop
    crop_span = (
        1
        if maximum_active_crop is None or minimum_active_crop is None
        else maximum_active_crop - minimum_active_crop + 1
    )
    state_keys: set[tuple[int, int, bool]] = set()
    n_farmers = crops.shape[1]
    for farmer_start in range(0, n_farmers, farmer_chunk_size):
        farmer_stop = min(farmer_start + farmer_chunk_size, n_farmers)
        unit_chunk = farmer_lookup_units[farmer_start:farmer_stop].astype(
            np.int64,
            copy=False,
        )
        for year_index in range(crops.shape[0]):
            crop_chunk = crops[year_index, farmer_start:farmer_stop]
            active = crop_chunk != -1
            if not np.any(active):
                continue
            encoded_states = (
                unit_chunk[active] * crop_span
                + crop_chunk[active].astype(np.int64, copy=False)
                + crop_offset
            ) * 2 + irrigation[year_index, farmer_start:farmer_stop][active]
            for encoded_state_value in np.unique(encoded_states):
                encoded_state = int(encoded_state_value)
                irrigated = bool(encoded_state % 2)
                unit_crop_key = encoded_state // 2
                state_keys.add(
                    (
                        unit_crop_key // crop_span,
                        unit_crop_key % crop_span - crop_offset,
                        irrigated,
                    )
                )
    return state_keys


def _run_mirca_calendar_sequence_worker(
    farmer_bounds: tuple[int, int],
) -> tuple[
    int,
    int,
    np.ndarray,
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Select calendars for one disjoint farmer range in a forked worker.

    Args:
        farmer_bounds: Inclusive start and exclusive stop farmer IDs.

    Returns:
        Farmer bounds, calendar chunk, failure records, and additive statistics.

    Raises:
        RuntimeError: If the worker was started without inherited context.
    """
    context = _MIRCA_CALENDAR_SEQUENCE_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("MIRCA calendar worker context was not initialized.")
    farmer_start, farmer_stop = farmer_bounds
    calendar_stack, failures, statistics = (
        _select_mirca_calendar_sequences_for_farmers_serial(
            context.crop_calendar,
            farmer_mirca_units=context.farmer_mirca_units[farmer_start:farmer_stop],
            farmer_main_crops_by_year=context.crops[:, farmer_start:farmer_stop],
            farmer_is_irrigated_by_year=context.irrigation[:, farmer_start:farmer_stop],
            hrl_years=context.years,
            replace_crop_calendar_unit_code=(context.replace_crop_calendar_unit_code),
            random_seed=context.random_seed,
            candidate_pool_cache=context.candidate_pool_cache,
            logger=None,
            progress_interval=max(farmer_stop - farmer_start + 1, 1),
            farmer_id_offset=farmer_start,
            calendar_index=context.calendar_index,
            candidate_pool_tier_cache=context.candidate_pool_tier_cache,
        )
    )
    return farmer_start, farmer_stop, calendar_stack, failures, statistics


def _initialize_mirca_calendar_sequence_worker() -> None:
    """Disable cyclic GC in a dedicated fork worker to protect shared pages.

    Farmer-chunk temporaries are acyclic and are released by reference counting.
    Disabling cyclic collection prevents a child from scanning and dirtying
    inherited parent objects, which would otherwise increase copy-on-write RAM.
    """
    gc.disable()


def _empty_mirca_calendar_sequence_statistics(
    *,
    candidate_pool_count: int,
) -> dict[str, Any]:
    """Create the additive statistics structure used by parallel reduction.

    Args:
        candidate_pool_count: Number of shared candidate pools.

    Returns:
        Zero-initialized selector statistics.
    """
    tier_counts = {name: 0 for name in _CALENDAR_FALLBACK_NAMES}
    return {
        "candidate_pool_count": candidate_pool_count,
        "preference_cache_count": 0,
        "peak_farmer_preference_cache_count": 0,
        "full_preference_orders_built": 0,
        "preference_candidates_ranked": 0,
        "preference_candidates_available": 0,
        "rank_order_seconds": 0.0,
        "dynamic_programming_seconds": 0.0,
        "transitions_evaluated": 0,
        "preferred_path_transition_checks": 0,
        "preferred_path_fast_path_farmers": 0,
        "farmers_using_fallback": 0,
        "farmers_reconsidering_earlier_choice": 0,
        "selected_tier_counts": tier_counts.copy(),
        "maximum_tier_attempt_counts": tier_counts.copy(),
        "starting_tier_farmer_counts": tier_counts.copy(),
    }


def _merge_mirca_calendar_sequence_statistics(
    aggregate: dict[str, Any],
    chunk: dict[str, Any],
) -> None:
    """Add one worker's selector statistics to an aggregate in place.

    Args:
        aggregate: Mutable accumulated selector statistics.
        chunk: Statistics returned by one disjoint farmer chunk.
    """
    additive_keys = (
        "preference_cache_count",
        "full_preference_orders_built",
        "preference_candidates_ranked",
        "preference_candidates_available",
        "rank_order_seconds",
        "dynamic_programming_seconds",
        "transitions_evaluated",
        "preferred_path_transition_checks",
        "preferred_path_fast_path_farmers",
        "farmers_using_fallback",
        "farmers_reconsidering_earlier_choice",
    )
    for key in additive_keys:
        aggregate[key] += chunk[key]
    aggregate["peak_farmer_preference_cache_count"] = max(
        aggregate["peak_farmer_preference_cache_count"],
        chunk["peak_farmer_preference_cache_count"],
    )
    for tier_key in (
        "selected_tier_counts",
        "maximum_tier_attempt_counts",
        "starting_tier_farmer_counts",
    ):
        for tier_name in _CALENDAR_FALLBACK_NAMES:
            aggregate[tier_key][tier_name] += chunk[tier_key][tier_name]


def _select_mirca_calendar_sequences_for_farmers(
    crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
    *,
    farmer_mirca_units: np.ndarray,
    farmer_main_crops_by_year: np.ndarray,
    farmer_is_irrigated_by_year: np.ndarray,
    hrl_years: np.ndarray | tuple[int, ...],
    replace_crop_calendar_unit_code: dict[int, int],
    random_seed: int,
    candidate_pool_cache: dict[
        tuple[int, int, bool],
        _MIRCACalendarCandidatePool,
    ]
    | None = None,
    logger: Any | None = None,
    progress_interval: int = 100_000,
    n_workers: int | None = None,
    farmer_chunk_size: int = 25_000,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Select joint farmer calendars serially or in deterministic chunks.

    Each farmer is independent and every random stream is keyed by the global
    farmer ID. Forked workers can therefore process disjoint farmer slices
    without changing any calendar selection. Candidate pools and large input
    arrays are prepared before forking and inherited read-only, while calendar
    chunks are returned in bounded batches and written to their original IDs.

    Args:
        crop_calendar: Parsed MIRCA-OS calendar entries by administrative unit.
        farmer_mirca_units: MIRCA-OS unit per compact farmer.
        farmer_main_crops_by_year: MIRCA crop class by HRL year and farmer.
        farmer_is_irrigated_by_year: Irrigation state by HRL year and farmer.
        hrl_years: Strictly increasing HRL years represented by axis zero.
        replace_crop_calendar_unit_code: Optional MIRCA unit replacements.
        random_seed: Base seed for reproducible weighted candidate ordering.
        candidate_pool_cache: Optional shared candidate pools.
        logger: Optional progress logger.
        progress_interval: Farmers completed between progress messages.
        n_workers: Worker processes. When omitted, use the Slurm allocation,
            capped at eight, or one worker outside Slurm.
        farmer_chunk_size: Farmers returned by one worker task.

    Returns:
        Calendar stack, farmer failure records in ascending farmer-ID order,
        and aggregated selector statistics.

    Raises:
        ValueError: If arrays, worker settings, or years are invalid.
    """
    farmer_mirca_units = np.asarray(farmer_mirca_units, dtype=np.int32)
    crops = np.asarray(farmer_main_crops_by_year, dtype=np.int32)
    irrigation = np.asarray(farmer_is_irrigated_by_year, dtype=bool)
    years = np.asarray(hrl_years, dtype=np.int32)
    if crops.ndim != 2:
        raise ValueError("farmer_main_crops_by_year must have shape (year, farmer).")
    if irrigation.shape != crops.shape:
        raise ValueError(
            "farmer_is_irrigated_by_year must match farmer_main_crops_by_year."
        )
    if years.ndim != 1 or years.size != crops.shape[0]:
        raise ValueError("hrl_years must match the first crop-calendar axis.")
    if farmer_mirca_units.shape != (crops.shape[1],):
        raise ValueError("farmer_mirca_units must contain one value per farmer.")
    if years.size > 1 and np.any(np.diff(years) <= 0):
        raise ValueError("hrl_years must be strictly increasing.")
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1.")
    if farmer_chunk_size < 1:
        raise ValueError("farmer_chunk_size must be at least 1.")
    worker_count = (
        _default_mirca_calendar_sequence_workers()
        if n_workers is None
        else int(n_workers)
    )
    if worker_count < 1:
        raise ValueError("n_workers must be at least 1.")

    if worker_count == 1 or crops.shape[1] <= farmer_chunk_size:
        calendar_stack, failures, statistics = (
            _select_mirca_calendar_sequences_for_farmers_serial(
                crop_calendar,
                farmer_mirca_units=farmer_mirca_units,
                farmer_main_crops_by_year=crops,
                farmer_is_irrigated_by_year=irrigation,
                hrl_years=years,
                replace_crop_calendar_unit_code=(replace_crop_calendar_unit_code),
                random_seed=random_seed,
                candidate_pool_cache=candidate_pool_cache,
                logger=logger,
                progress_interval=progress_interval,
            )
        )
        statistics["worker_count"] = 1
        statistics["farmer_chunk_size"] = int(crops.shape[1])
        return calendar_stack, failures, statistics

    if "fork" not in mp.get_all_start_methods():
        if logger is not None:
            logger.warning(
                "Parallel joint HRL calendar selection requires the POSIX fork "
                "start method; falling back to one worker."
            )
        return _select_mirca_calendar_sequences_for_farmers(
            crop_calendar,
            farmer_mirca_units=farmer_mirca_units,
            farmer_main_crops_by_year=crops,
            farmer_is_irrigated_by_year=irrigation,
            hrl_years=years,
            replace_crop_calendar_unit_code=replace_crop_calendar_unit_code,
            random_seed=random_seed,
            candidate_pool_cache=candidate_pool_cache,
            logger=logger,
            progress_interval=progress_interval,
            n_workers=1,
            farmer_chunk_size=farmer_chunk_size,
        )

    if candidate_pool_cache is None:
        candidate_pool_cache = {}
    calendar_index = _index_mirca_calendars(crop_calendar)
    farmer_lookup_units = farmer_mirca_units.copy()
    for original_unit, replacement_unit in replace_crop_calendar_unit_code.items():
        farmer_lookup_units[farmer_mirca_units == int(original_unit)] = int(
            replacement_unit
        )
    state_keys = _mirca_calendar_sequence_state_keys(
        farmer_lookup_units=farmer_lookup_units,
        crops=crops,
        irrigation=irrigation,
    )
    candidate_pool_tier_cache: dict[
        tuple[int, int, bool],
        _MIRCACalendarPoolTierMetadata,
    ] = {}
    pool_preparation_started = time.perf_counter()
    for state_key in sorted(state_keys):
        lookup_unit, main_crop, is_irrigated = state_key
        candidate_pool = candidate_pool_cache.get(state_key)
        if candidate_pool is None:
            candidate_pool = _build_mirca_calendar_candidate_pool(
                crop_calendar,
                lookup_unit=lookup_unit,
                main_crop=main_crop,
                is_irrigated=is_irrigated,
                calendar_index=calendar_index,
            )
            candidate_pool_cache[state_key] = candidate_pool
        if candidate_pool.calendars.shape[0]:
            candidate_pool_tier_cache[state_key] = _mirca_calendar_pool_tier_metadata(
                candidate_pool
            )
    if logger is not None:
        logger.info(
            "Joint HRL calendar selection prepared %s candidate state(s) and "
            "%s pool(s) in %.2f s; starting %s forked worker(s) with chunks "
            "of at most %s farmers.",
            len(state_keys),
            len(candidate_pool_cache),
            time.perf_counter() - pool_preparation_started,
            min(
                worker_count,
                (crops.shape[1] + farmer_chunk_size - 1) // farmer_chunk_size,
            ),
            farmer_chunk_size,
        )

    farmer_ranges = [
        (farmer_start, min(farmer_start + farmer_chunk_size, crops.shape[1]))
        for farmer_start in range(0, crops.shape[1], farmer_chunk_size)
    ]
    worker_count = min(worker_count, len(farmer_ranges))
    worker_context = _MIRCACalendarSequenceWorkerContext(
        crop_calendar=crop_calendar,
        farmer_mirca_units=farmer_mirca_units,
        crops=crops,
        irrigation=irrigation,
        years=years,
        replace_crop_calendar_unit_code=replace_crop_calendar_unit_code,
        random_seed=random_seed,
        candidate_pool_cache=candidate_pool_cache,
        calendar_index=calendar_index,
        candidate_pool_tier_cache=candidate_pool_tier_cache,
    )
    global _MIRCA_CALENDAR_SEQUENCE_WORKER_CONTEXT
    _MIRCA_CALENDAR_SEQUENCE_WORKER_CONTEXT = worker_context

    n_years, n_farmers = crops.shape
    aggregated_statistics = _empty_mirca_calendar_sequence_statistics(
        candidate_pool_count=len(candidate_pool_cache),
    )
    failures: list[dict[str, Any]] = []
    processed_farmers = 0
    next_progress = progress_interval
    selector_started = time.perf_counter()
    fork_context = mp.get_context("fork")
    try:
        # Start workers before allocating the multi-gigabyte output so they do
        # not inherit its page table. Input arrays and candidate pools remain
        # shared copy-on-write.
        with fork_context.Pool(
            processes=worker_count,
            initializer=_initialize_mirca_calendar_sequence_worker,
        ) as process_pool:
            calendar_stack = np.full(
                (n_years, n_farmers, 3, 4),
                -1,
                dtype=np.int32,
            )
            results = process_pool.imap_unordered(
                _run_mirca_calendar_sequence_worker,
                farmer_ranges,
                chunksize=1,
            )
            for (
                farmer_start,
                farmer_stop,
                calendar_chunk,
                chunk_failures,
                chunk_statistics,
            ) in results:
                calendar_stack[:, farmer_start:farmer_stop] = calendar_chunk
                failures.extend(chunk_failures)
                _merge_mirca_calendar_sequence_statistics(
                    aggregated_statistics,
                    chunk_statistics,
                )
                processed_farmers += farmer_stop - farmer_start
                if logger is not None and (
                    processed_farmers >= next_progress or processed_farmers == n_farmers
                ):
                    elapsed = time.perf_counter() - selector_started
                    rss_mib, hwm_mib = _process_memory_mib()
                    logger.info(
                        "Joint HRL parallel calendar progress: %s/%s farmers "
                        "(%.1f%%) in %.1f s (%.0f farmers/s, %s workers); "
                        "preferred fast paths=%s; full orders=%s; ranked "
                        "candidates=%s/%s; DP edges=%s; cumulative worker "
                        "rank/DP time=%.1f/%.1f s; failures=%s; RSS=%s MiB; "
                        "HWM=%s MiB.",
                        processed_farmers,
                        n_farmers,
                        processed_farmers / n_farmers * 100.0,
                        elapsed,
                        processed_farmers / elapsed if elapsed > 0.0 else 0.0,
                        worker_count,
                        aggregated_statistics["preferred_path_fast_path_farmers"],
                        aggregated_statistics["full_preference_orders_built"],
                        aggregated_statistics["preference_candidates_ranked"],
                        aggregated_statistics["preference_candidates_available"],
                        aggregated_statistics["transitions_evaluated"],
                        aggregated_statistics["rank_order_seconds"],
                        aggregated_statistics["dynamic_programming_seconds"],
                        len(failures),
                        "n/a" if rss_mib is None else f"{rss_mib:.1f}",
                        "n/a" if hwm_mib is None else f"{hwm_mib:.1f}",
                    )
                    while next_progress <= processed_farmers:
                        next_progress += progress_interval
    finally:
        _MIRCA_CALENDAR_SEQUENCE_WORKER_CONTEXT = None

    failures.sort(key=lambda failure: int(failure["farmer_id"]))
    aggregated_statistics["candidate_pool_count"] = len(candidate_pool_cache)
    aggregated_statistics["worker_count"] = worker_count
    aggregated_statistics["farmer_chunk_size"] = farmer_chunk_size
    return calendar_stack, failures, aggregated_statistics


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


@dataclass(frozen=True, slots=True)
class _MIRCAIrrigationFractionLookup:
    """Reusable flattened MIRCA-OS crop-area arrays for irrigation assignment."""

    rainfed_values: np.ndarray
    irrigated_values: np.ndarray
    total_rainfed_by_cell: np.ndarray
    total_irrigated_by_cell: np.ndarray


def _prepare_mirca_irrigation_fraction_lookup(
    rainfed_fraction: xr.DataArray,
    irrigated_fraction: xr.DataArray,
) -> _MIRCAIrrigationFractionLookup:
    """Prepare static MIRCA-OS fraction arrays once for reuse across HRL years."""
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
    total_rainfed_by_cell = np.empty(n_cells, dtype=np.float64)
    total_irrigated_by_cell = np.empty(n_cells, dtype=np.float64)
    # Use the same per-cell 1-D sum operation as the original fallback path so the
    # floating-point result remains reproducible.
    for cell_id in range(n_cells):
        total_rainfed_by_cell[cell_id] = float(rainfed_values[:, cell_id].sum())
        total_irrigated_by_cell[cell_id] = float(irrigated_values[:, cell_id].sum())

    return _MIRCAIrrigationFractionLookup(
        rainfed_values=rainfed_values,
        irrigated_values=irrigated_values,
        total_rainfed_by_cell=total_rainfed_by_cell,
        total_irrigated_by_cell=total_irrigated_by_cell,
    )


@dataclass(frozen=True, slots=True)
class _MIRCASpatialContext:
    """Prepared MIRCA-OS spatial inputs that can be reused within one setup."""

    year: int
    reference_crop_map: xr.DataArray
    unit_geom: gpd.GeoDataFrame
    unit_grid: xr.DataArray
    farmer_units: np.ndarray
    reference_map_buffer: int


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
    fraction_lookup: _MIRCAIrrigationFractionLookup | None = None,
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
        fraction_lookup: Optional precomputed flattened MIRCA-OS fraction arrays.
            Reusing this across years avoids repeating identical conversions.

    Returns:
        Tuple containing a boolean irrigated-farmer array and an adaptations
        matrix with surface-water and groundwater source flags.

    Raises:
        ValueError: If rainfed and irrigated fraction stacks are not aligned.
        ValueError: If the fraction stacks do not contain a ``crop`` dimension.
    """
    if fraction_lookup is None:
        fraction_lookup = _prepare_mirca_irrigation_fraction_lookup(
            rainfed_fraction,
            irrigated_fraction,
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

    rainfed_values = fraction_lookup.rainfed_values
    irrigated_values = fraction_lookup.irrigated_values
    n_crops = rainfed_values.shape[0]
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

        total_rainfed_cell_area = float(fraction_lookup.total_rainfed_by_cell[cell_id])
        total_irrigated_cell_area = float(
            fraction_lookup.total_irrigated_by_cell[cell_id]
        )
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

        # Select the same sorted farm prefixes as the former DataFrame sorts and
        # Python row loops, but do the ordering and cumulative-area cutoff in
        # NumPy. With around one million farmers this runs for thousands of
        # groups in every HRL year, so avoiding two temporary DataFrames plus
        # ``itertuples`` per group is material.
        group_farmer_ids = group["farmer_id"].to_numpy(dtype=np.int32)
        group_areas_m2 = group["area_m2"].to_numpy(dtype=np.float64)
        group_hand_m = group["hand_m"].to_numpy(dtype=np.float64)
        group_groundwater_depth_m = group["groundwater_depth_m"].to_numpy(
            dtype=np.float64
        )

        surface_order = np.lexsort((group_farmer_ids, group_hand_m))
        if target_surface_water_area > 0.0 and surface_order.size:
            surface_cumulative_area = np.cumsum(
                group_areas_m2[surface_order],
                dtype=np.float64,
            )
            surface_count = min(
                int(
                    np.searchsorted(
                        surface_cumulative_area,
                        target_surface_water_area,
                        side="left",
                    )
                )
                + 1,
                surface_order.size,
            )
            surface_positions = surface_order[:surface_count]
            assigned_surface_area = float(surface_cumulative_area[surface_count - 1])
        else:
            surface_positions = np.empty(0, dtype=np.int64)
            assigned_surface_area = 0.0

        surface_farmer_ids = group_farmer_ids[surface_positions]
        if surface_farmer_ids.size:
            adaptations[
                surface_farmer_ids,
                SURFACE_IRRIGATION_EQUIPMENT,
            ] = True
            is_irrigated[surface_farmer_ids] = True

        remaining_mask = np.ones(group_farmer_ids.size, dtype=bool)
        remaining_mask[surface_positions] = False
        remaining_positions = np.flatnonzero(remaining_mask)
        groundwater_order_within_remaining = np.lexsort(
            (
                group_farmer_ids[remaining_positions],
                group_groundwater_depth_m[remaining_positions],
            )
        )
        groundwater_order = remaining_positions[groundwater_order_within_remaining]
        if target_groundwater_area > 0.0 and groundwater_order.size:
            groundwater_cumulative_area = np.cumsum(
                group_areas_m2[groundwater_order],
                dtype=np.float64,
            )
            groundwater_count = min(
                int(
                    np.searchsorted(
                        groundwater_cumulative_area,
                        target_groundwater_area,
                        side="left",
                    )
                )
                + 1,
                groundwater_order.size,
            )
            groundwater_positions = groundwater_order[:groundwater_count]
            assigned_groundwater_area = float(
                groundwater_cumulative_area[groundwater_count - 1]
            )
        else:
            groundwater_positions = np.empty(0, dtype=np.int64)
            assigned_groundwater_area = 0.0

        groundwater_farmer_ids = group_farmer_ids[groundwater_positions]
        if groundwater_farmer_ids.size:
            adaptations[
                groundwater_farmer_ids,
                WELL_ADAPTATION,
            ] = True
            is_irrigated[groundwater_farmer_ids] = True

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
        crop_values = _validated_HRL_crop_values(crop_chunk_da.values)
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

        valid_codes = crop_values[valid]
        if projected_cell_area_m2 is not None:
            counts = np.bincount(valid_codes)
            crop_codes = np.flatnonzero(counts)
            crop_areas_m2 = (
                counts[crop_codes].astype(np.float64) * projected_cell_area_m2
            )
        else:
            chunk_cell_area_m2 = raster_cell_area_m2(crop_chunk_da)
            weighted_areas = np.bincount(
                valid_codes,
                weights=chunk_cell_area_m2[valid],
            )
            crop_codes = np.flatnonzero(weighted_areas)
            crop_areas_m2 = weighted_areas[crop_codes]
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
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate one native-resolution HRL CTY year to the model subgrid.

    GDAL >= 3.11 contains nodata-collision avoidance logic that may replace a
    source value when it equals the destination nodata value. In affected GDAL
    releases this can also be triggered too aggressively. CTY uses UInt16 65535
    as the documented ``Outside area`` code, so allowing rioxarray/GDAL to choose
    its default UInt16 destination nodata (also 65535) can turn outside pixels
    into 65534.

    To make the result independent of that behavior, this routine materializes
    the already tile-clipped source once, converts CTY to signed int32, and calls
    rasterio.warp.reproject with *different* source and destination sentinels:
    - source missing/outside: -2
    - temporary categorical destination nodata: -3
    - temporary fraction destination nodata: -1.0

    The temporary destination values are normalized immediately after the warp.
    No UInt16/65535 nodata value is ever supplied to GDAL here.
    """
    if crop_types.rio.crs is None:
        raise ValueError("The HRL Crop Types raster must have a CRS.")
    if crop_types.ndim != 2:
        raise ValueError("The HRL Crop Types raster must be two-dimensional.")
    if template.rio.crs is None:
        raise ValueError("The regional subgrid template must have a CRS.")

    crop_values = _validated_HRL_crop_values(crop_types.values)
    has_hrl_coverage = crop_values != _HRL_MISSING_CROP_CODE
    active_crop = (crop_values > _HRL_NO_CROPLAND_CODE) & has_hrl_coverage

    source_transform = crop_types.rio.transform(recalc=True)
    destination_transform = template.rio.transform(recalc=True)
    destination_shape = (template.sizes["y"], template.sizes["x"])

    # Categorical crop state. Source -2 is excluded from mode resampling, while
    # destination -3 is guaranteed not to collide with any source value.
    crop_states = np.ascontiguousarray(crop_values, dtype=np.int32)
    crop_subgrid_values = np.full(
        destination_shape, _HRL_WARP_DST_NODATA, dtype=np.int32
    )
    reproject(
        source=crop_states,
        destination=crop_subgrid_values,
        src_transform=source_transform,
        src_crs=crop_types.rio.crs,
        src_nodata=_HRL_MISSING_CROP_CODE,
        dst_transform=destination_transform,
        dst_crs=template.rio.crs,
        dst_nodata=_HRL_WARP_DST_NODATA,
        resampling=Resampling.mode,
        init_dest_nodata=True,
    )
    crop_subgrid_values[crop_subgrid_values == _HRL_WARP_DST_NODATA] = (
        _HRL_MISSING_CROP_CODE
    )

    # Cultivated fraction. Both 0.0 and 1.0 are valid source values, so 0.0 must
    # never be used as destination nodata. Use -1.0 temporarily and convert
    # uncovered destination cells to a cultivated fraction of zero afterwards.
    cultivated_source = np.ascontiguousarray(active_crop.astype(np.float32))
    cultivated_fraction_values = np.full(
        destination_shape, _HRL_FRACTION_WARP_DST_NODATA, dtype=np.float32
    )
    reproject(
        source=cultivated_source,
        destination=cultivated_fraction_values,
        src_transform=source_transform,
        src_crs=crop_types.rio.crs,
        src_nodata=None,
        dst_transform=destination_transform,
        dst_crs=template.rio.crs,
        dst_nodata=_HRL_FRACTION_WARP_DST_NODATA,
        resampling=Resampling.average,
        init_dest_nodata=True,
    )
    cultivated_fraction_values = np.nan_to_num(
        cultivated_fraction_values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    cultivated_fraction_values[
        cultivated_fraction_values == _HRL_FRACTION_WARP_DST_NODATA
    ] = 0.0
    np.clip(cultivated_fraction_values, 0.0, 1.0, out=cultivated_fraction_values)

    return crop_subgrid_values, cultivated_fraction_values


def _chunk_slices_without_singletons(
    length: int,
    chunk_size: int,
) -> tuple[slice, ...]:
    """Split an axis while folding a one-cell tail into the previous chunk."""
    if length < 1:
        return ()
    slices = [
        slice(start, min(start + chunk_size, length))
        for start in range(0, length, chunk_size)
    ]
    if len(slices) > 1 and slices[-1].stop - slices[-1].start == 1:
        slices[-2] = slice(slices[-2].start, slices[-1].stop)
        slices.pop()
    return tuple(slices)


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
    def setup_create_farms_from_HRL_lowder(
        self,
        region_id_column: str = "region_id",
        country_iso3_column: str = "ISO3",
        size_class_boundaries: dict[str, tuple[int | float, int | float]] | None = None,
        years: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024),
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
        lowder_rows_by_iso3 = {
            str(iso3): group.drop(["Country", "Census Year", "Total"], axis=1)
            for iso3, group in farm_sizes_per_region.groupby("ISO3", sort=False)
        }
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
        workflow_started = time.perf_counter()
        phase_totals_seconds: dict[str, float] = {}
        farmer_chunks_in_compact_order = True

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

            if original_iso3.upper() not in _HRL_CROPLANDS_EEA38_ISO3:
                self.logger.warning(
                    "Skipping region %s (%s) because the country is outside the "
                    "published HRL Croplands coverage; no farms will be created.",
                    region_id,
                    original_iso3,
                )
                continue

            row_indices, col_indices = np.where(region_mask_full)
            y_slice = slice(int(row_indices.min()), int(row_indices.max()) + 1)
            x_slice = slice(int(col_indices.min()), int(col_indices.max()) + 1)
            region_template = region_ids.isel(y=y_slice, x=x_slice)
            region_mask = region_mask_full[y_slice, x_slice]
            region_cell_area_m2 = cell_area_m2[y_slice, x_slice]
            if region_template.sizes["x"] < 2 or region_template.sizes["y"] < 2:
                self.logger.warning(
                    "Skipping region %s (%s) because its model-cell extent is %sx%s; "
                    "raster reprojection requires at least two cells on each axis.",
                    region_id,
                    original_iso3,
                    region_template.sizes["x"],
                    region_template.sizes["y"],
                )
                continue

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
            region_started = time.perf_counter()
            region_phase_seconds: dict[str, float] = {}

            crop_stack = np.full(
                (len(years), *region_template.shape),
                _HRL_MISSING_CROP_CODE,
                dtype=np.int32,
            )
            fraction_stack = np.zeros(
                (len(years), *region_template.shape),
                dtype=np.float32,
            )
            native_crop_areas_per_year: list[dict[int, float]] = []
            region_has_hrl_coverage = True
            reprojection_source_crs: Any | None = None
            reprojection_tiles: (
                list[
                    tuple[
                        slice,
                        slice,
                        np.ndarray,
                        xr.DataArray,
                        tuple[float, float, float, float],
                    ]
                ]
                | None
            ) = None

            for year_index, year in enumerate(years):
                crop_types = None
                crop_types_adapter = None
                read_started = time.perf_counter()
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
                        normalize_nodata=True,
                        chunks=raster_chunks,
                    )
                    # ``dst_crs=None`` requests the native Copernicus HRL grid,
                    # which is EPSG:3035. Dask-backed xarray/rioxarray operations can
                    # occasionally drop only the grid-mapping metadata while retaining
                    # the correct x/y coordinates. Repair that metadata defensively at
                    # the workflow boundary so native-area and tiled reprojection never
                    # receive a CRS-less HRL raster.
                    if crop_types.rio.crs is None:
                        self.logger.warning(
                            "HRL adapter %s returned year %s without CRS metadata; "
                            "restoring native EPSG:3035 before processing.",
                            type(crop_types_adapter).__name__,
                            year,
                        )
                        crop_types = crop_types.rio.set_spatial_dims(
                            x_dim="x", y_dim="y", inplace=False
                        ).rio.write_crs("EPSG:3035", inplace=False)
                    read_seconds = _record_phase_timing(
                        region_phase_seconds, "hrl_read", read_started
                    )
                except _HRL_NO_COVERAGE_ERRORS as error:
                    if original_iso3.upper() in _HRL_CROPLANDS_EEA38_ISO3:
                        raise
                    self.logger.warning(
                        "Skipping region %s (%s): no HRL Crop Types coverage for "
                        "year %s from the configured Copernicus source. Error: %s",
                        region_id,
                        original_iso3,
                        year,
                        error,
                    )
                    region_has_hrl_coverage = False
                    break

                native_started = time.perf_counter()
                native_crop_areas = _native_hrl_crop_category_areas_m2(
                    crop_types,
                    region_active_geometry,
                    chunk_rows=max(int(raster_chunks.get("y", 4096)), 1),
                )
                _validate_HRL_crop_area_codes(
                    native_crop_areas,
                    context=f"Region {region_id}, HRL year {year}",
                )
                native_crop_areas_per_year.append(native_crop_areas)
                native_seconds = _record_phase_timing(
                    region_phase_seconds, "native_area", native_started
                )

                crop_year = crop_stack[year_index]
                cultivated_fraction_year = fraction_stack[year_index]
                source_bounds = crop_types.rio.bounds()
                source_resolution = crop_types.rio.resolution()
                source_buffer_x = abs(float(source_resolution[0])) * 2.0
                source_buffer_y = abs(float(source_resolution[1])) * 2.0

                if crop_types.rio.crs is None:
                    raise ValueError(
                        "The native HRL Crop Types raster must have a CRS."
                    )
                if reprojection_tiles is None:
                    reprojection_source_crs = crop_types.rio.crs
                    tile_y_slices = _chunk_slices_without_singletons(
                        region_template.sizes["y"], subgrid_chunk_size
                    )
                    tile_x_slices = _chunk_slices_without_singletons(
                        region_template.sizes["x"], subgrid_chunk_size
                    )
                    reprojection_tiles = []
                    for tile_y_slice in tile_y_slices:
                        for tile_x_slice in tile_x_slices:
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
                            reprojection_tiles.append(
                                (
                                    tile_y_slice,
                                    tile_x_slice,
                                    tile_region_mask,
                                    tile_template,
                                    tuple(float(value) for value in tile_bounds),
                                )
                            )
                elif crop_types.rio.crs != reprojection_source_crs:
                    raise ValueError(
                        "HRL Crop Types CRS changed between years within region "
                        f"{region_id}: {reprojection_source_crs} -> {crop_types.rio.crs}."
                    )

                reprojection_started = time.perf_counter()
                assert reprojection_tiles is not None
                for (
                    tile_y_slice,
                    tile_x_slice,
                    tile_region_mask,
                    tile_template,
                    tile_bounds,
                ) in reprojection_tiles:
                    clip_min_x = max(tile_bounds[0] - source_buffer_x, source_bounds[0])
                    clip_min_y = max(tile_bounds[1] - source_buffer_y, source_bounds[1])
                    clip_max_x = min(tile_bounds[2] + source_buffer_x, source_bounds[2])
                    clip_max_y = min(tile_bounds[3] + source_buffer_y, source_bounds[3])
                    if clip_min_x >= clip_max_x or clip_min_y >= clip_max_y:
                        continue

                    crop_types_tile = crop_types.rio.clip_box(
                        minx=clip_min_x,
                        miny=clip_min_y,
                        maxx=clip_max_x,
                        maxy=clip_max_y,
                        allow_one_dimensional_raster=True,
                    )
                    tile_crop, tile_fraction = _reproject_HRL_year_to_subgrid(
                        crop_types_tile, tile_template
                    )
                    tile_crop[~tile_region_mask] = _HRL_MISSING_CROP_CODE
                    tile_fraction[~tile_region_mask] = 0.0
                    crop_year[tile_y_slice, tile_x_slice] = tile_crop
                    cultivated_fraction_year[tile_y_slice, tile_x_slice] = tile_fraction
                    del crop_types_tile, tile_crop, tile_fraction

                reprojection_seconds = _record_phase_timing(
                    region_phase_seconds, "reprojection", reprojection_started
                )
                rss_mib, hwm_mib = _process_memory_mib()
                self.logger.debug(
                    "HRL timing region %s year %s [s]: read=%.3f native=%.3f "
                    "reprojection=%.3f; RSS=%s MiB; HWM=%s MiB.",
                    region_id,
                    year,
                    read_seconds,
                    native_seconds,
                    reprojection_seconds,
                    "n/a" if rss_mib is None else f"{rss_mib:.1f}",
                    "n/a" if hwm_mib is None else f"{hwm_mib:.1f}",
                )

                crop_types.close()
                del (
                    crop_types,
                    crop_types_adapter,
                    crop_year,
                    cultivated_fraction_year,
                )

            if not region_has_hrl_coverage:
                del crop_stack, fraction_stack
                continue
            if len(native_crop_areas_per_year) != len(years):
                raise ValueError(f"Incomplete HRL crop stack for region {region_id}.")

            selection_started = time.perf_counter()
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

            valid_count = np.count_nonzero(crop_stack > 0, axis=0)
            eligible_mask = region_mask & (valid_count > 0)
            valid_frequency = valid_count / float(len(years))
            mean_fraction = fraction_stack.mean(axis=0, dtype=np.float64)
            selection_score = 0.80 * valid_frequency + 0.20 * mean_fraction

            if not eligible_mask.any():
                self.logger.warning(
                    "Skipping region %s because it has no model cells with an "
                    "observed HRL CTY crop in any requested year.",
                    region_id,
                )
                continue

            # A complete source sequence may include native no-cropland (0), which is
            # converted to model fallow (-1) after static-cell selection. Native
            # outside/missing (-2), however, invalidates the complete sequence. Check
            # availability before selection so a lower-ranked valid sequence cannot be
            # accidentally discarded by the static-area ranking.
            regional_complete_original_sequence = (
                eligible_mask
                & ~np.any(crop_stack == _HRL_MISSING_CROP_CODE, axis=0)
                & np.any(crop_stack > 0, axis=0)
            )
            if not regional_complete_original_sequence.any():
                valid_cells_by_year = [
                    int(np.count_nonzero(crop_stack[year_index][region_mask] > 0))
                    for year_index in range(len(years))
                ]
                self.logger.warning(
                    "Skipping region %s (%s) because no eligible model cell has a "
                    "complete original HRL CTY sequence across requested years %s "
                    "(valid crop cells by year: %s); no farms will be created.",
                    region_id,
                    original_iso3,
                    years,
                    valid_cells_by_year,
                )
                del crop_stack, fraction_stack
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
            selected_area_before_sequence_repair_m2 = float(
                region_cell_area_m2[cultivated_mask].sum()
            )
            (
                cultivated_mask,
                sequence_selection_repaired,
                sequence_repair_used_extra_cell,
            ) = ensure_complete_sequence_in_selected_cells(
                cultivated_mask,
                regional_complete_original_sequence,
                selection_score,
                region_cell_area_m2,
                target_area_m2=selection_target_area_m2,
            )
            if sequence_selection_repaired:
                selected_area_after_sequence_repair_m2 = float(
                    region_cell_area_m2[cultivated_mask].sum()
                )
                self.logger.info(
                    "Region %s static HRL selection excluded every complete original "
                    "sequence; retained the highest-ranked complete sequence via %s "
                    "(selected area %.3f -> %.3f km²; %s complete source cells exist "
                    "regionally).",
                    region_id,
                    (
                        "one additional model cell"
                        if sequence_repair_used_extra_cell
                        else "an area-safe one-cell swap"
                    ),
                    selected_area_before_sequence_repair_m2 / 1_000_000.0,
                    selected_area_after_sequence_repair_m2 / 1_000_000.0,
                    int(np.count_nonzero(regional_complete_original_sequence)),
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

            complete_original_sequence = (
                cultivated_mask
                & ~np.any(crop_stack == _HRL_MISSING_CROP_CODE, axis=0)
                & np.any(crop_stack > 0, axis=0)
            )
            if not complete_original_sequence.any():
                raise RuntimeError(
                    "Static HRL sequence repair failed to retain a complete original "
                    f"crop sequence in region {region_id}."
                )
            del regional_complete_original_sequence

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

            _record_phase_timing(region_phase_seconds, "selection", selection_started)
            lowder_started = time.perf_counter()
            iso3 = farm_size_donor_country.get(original_iso3, original_iso3)
            if iso3 != original_iso3:
                self.logger.info(
                    "Missing farm sizes for %s; using donor country %s.",
                    original_iso3,
                    iso3,
                )
            region_farm_sizes = lowder_rows_by_iso3.get(str(iso3))
            if region_farm_sizes is None or len(region_farm_sizes) != 2:
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
            _record_phase_timing(region_phase_seconds, "lowder_targets", lowder_started)

            farm_growth_started = time.perf_counter()
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

            _record_phase_timing(
                region_phase_seconds, "farm_growth", farm_growth_started
            )
            if not np.any((local_farms >= 0) & complete_original_sequence):
                raise RuntimeError(
                    "Farm growth removed every complete original HRL crop-sequence "
                    f"source in region {region_id}."
                )
            del complete_original_sequence

            farmer_areas_local_m2 = farmers_region["area_m2"].to_numpy(dtype=np.float64)
            sequence_assignment_started = time.perf_counter()
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
            _record_phase_timing(
                region_phase_seconds, "sequence_assignment", sequence_assignment_started
            )
            n_observed_sequences = int(
                sequence_quality.attrs.get("n_observed_sequences", 0)
            )
            diagnostics_started = time.perf_counter()
            farmers_region.loc[:, crop_columns] = assigned_sequences
            for quality_column in sequence_quality.columns:
                farmers_region[quality_column] = sequence_quality[
                    quality_column
                ].to_numpy()

            crop_alignment_summary_by_year: list[dict[str, float | int]] = []
            for year_index, (year, crop_column) in enumerate(
                zip(years, crop_columns, strict=True)
            ):
                assigned_crop_codes = assigned_sequences[:, year_index]
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
            region_farmer_ids = farmers_region["farmer_id"]
            if len(region_farmer_ids) > 0:
                expected_last_id = farmer_id_offset + len(region_farmer_ids) - 1
                if not (
                    region_farmer_ids.is_monotonic_increasing
                    and region_farmer_ids.is_unique
                    and int(region_farmer_ids.iloc[0]) == farmer_id_offset
                    and int(region_farmer_ids.iloc[-1]) == expected_last_id
                ):
                    farmer_chunks_in_compact_order = False
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
            for year_index, year in enumerate(years):
                crop_values = assigned_sequences[:, year_index]
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

            _record_phase_timing(
                region_phase_seconds, "diagnostics", diagnostics_started
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

            region_total_seconds = time.perf_counter() - region_started
            for phase_name, phase_seconds in region_phase_seconds.items():
                phase_totals_seconds[phase_name] = (
                    phase_totals_seconds.get(phase_name, 0.0) + phase_seconds
                )
            phase_totals_seconds["region_total"] = (
                phase_totals_seconds.get("region_total", 0.0) + region_total_seconds
            )
            rss_mib, hwm_mib = _process_memory_mib()
            self.logger.info(
                "HRL phase timing region %s (%s) [s]: read=%.2f native=%.2f "
                "reproject=%.2f select=%.2f Lowder=%.2f grow=%.2f sequence=%.2f "
                "diagnostics=%.2f total=%.2f; RSS=%s MiB; HWM=%s MiB.",
                region_id,
                original_iso3,
                region_phase_seconds.get("hrl_read", 0.0),
                region_phase_seconds.get("native_area", 0.0),
                region_phase_seconds.get("reprojection", 0.0),
                region_phase_seconds.get("selection", 0.0),
                region_phase_seconds.get("lowder_targets", 0.0),
                region_phase_seconds.get("farm_growth", 0.0),
                region_phase_seconds.get("sequence_assignment", 0.0),
                region_phase_seconds.get("diagnostics", 0.0),
                region_total_seconds,
                "n/a" if rss_mib is None else f"{rss_mib:.1f}",
                "n/a" if hwm_mib is None else f"{hwm_mib:.1f}",
            )

            region_farmer_count = len(farmers_region)
            farmer_id_offset += region_farmer_count
            del (
                crop_stack,
                fraction_stack,
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
            if (
                region_farmer_count >= 100_000
                or selected_cell_count >= 1_000_000
                or (region_index + 1) % 8 == 0
            ):
                gc.collect()

        if not all_farmers:
            raise ValueError("No HRL-only farmer agents could be created.")

        farmers = pd.concat(all_farmers, ignore_index=True)
        farmer_ids_final = farmers["farmer_id"]
        if not (
            farmer_chunks_in_compact_order
            and farmer_ids_final.is_monotonic_increasing
            and farmer_ids_final.is_unique
        ):
            self.logger.warning(
                "Farmer chunks were not already in compact ID order; applying the "
                "legacy final farmer_id sort as a fallback."
            )
            farmers = farmers.sort_values("farmer_id").reset_index(drop=True)
        else:
            farmers = farmers.reset_index(drop=True)

        workflow_total_seconds = time.perf_counter() - workflow_started
        self.logger.info(
            "HRL completed-region phase totals [s]: read=%.1f native=%.1f "
            "reproject=%.1f select=%.1f Lowder=%.1f grow=%.1f sequence=%.1f "
            "diagnostics=%.1f region_total=%.1f workflow_total=%.1f.",
            phase_totals_seconds.get("hrl_read", 0.0),
            phase_totals_seconds.get("native_area", 0.0),
            phase_totals_seconds.get("reprojection", 0.0),
            phase_totals_seconds.get("selection", 0.0),
            phase_totals_seconds.get("lowder_targets", 0.0),
            phase_totals_seconds.get("farm_growth", 0.0),
            phase_totals_seconds.get("sequence_assignment", 0.0),
            phase_totals_seconds.get("diagnostics", 0.0),
            phase_totals_seconds.get("region_total", 0.0),
            workflow_total_seconds,
        )
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
        hrl_years: tuple[int, ...] = (
            2017,
            2018,
            2019,
            2020,
            2021,
            2022,
            2023,
            2024,
        ),
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

        In multi-year mode, calendars are selected jointly over each farmer's full
        active crop sequence. The selector searches four feasibility-aware fallback
        tiers, can reconsider an earlier calendar when its harvest blocks a later
        crop, and prefers local/irrigation-matched and area-weighted candidates.
        Fallow years retain the last active harvest constraint. Planting on the
        harvest date is allowed because harvest is processed before planting. All
        farmers are evaluated before an aggregated infeasibility error is raised.

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
            reduce_crops: Deprecated compatibility argument. Broad crop
                replacement is no longer performed; Europe_004's rice calendars
                are handled automatically and specifically.
            random_seed: Base seed for reproducible area-weighted MIRCA-OS calendar
                selection.

        Raises:
            ValueError: If required final farmer crop-table columns are missing.
            ValueError: If ``multiple_years`` is True and ``hrl_years`` is empty.
            ValueError: If farmers cannot be assigned to valid MIRCA-OS units.
            ValueError: If no MIRCA-OS calendar can be found for an assigned crop.
        """
        setup_timer = time.perf_counter()
        timing_rows: list[dict[str, float | int]] = []

        if replace_crop_calendar_unit_code is None:
            replace_crop_calendar_unit_code = {}

        if multiple_years and not hrl_years:
            raise ValueError("hrl_years must contain at least one year.")

        years_to_process = tuple(hrl_years) if multiple_years else (hrl_year,)
        if multiple_years and np.any(np.diff(np.asarray(years_to_process)) <= 0):
            raise ValueError(
                "hrl_years must be strictly increasing for sequential crop-calendar "
                f"assignment. Got {list(years_to_process)}."
            )

        is_europe_004_model = _is_europe_004_model_root(self.root)
        if reduce_crops:
            self.logger.warning(
                "setup_farmer_crop_calendar_from_HRL.reduce_crops is deprecated "
                "and no longer enables general crop replacement. The argument "
                "is accepted for configuration compatibility; only the automatic "
                "rice-free calendar exception for Europe_004 can change calendars."
            )

        n_farmers = self.array["agents/farmers/region_id"].size
        farmer_region_ids = self.array["agents/farmers/region_id"]
        farms = self.subgrid["agents/farmers/farms"]

        phase_timer = time.perf_counter()
        farmers_with_crops = self.table[_FARMERS_WITH_CROPS_TABLE]
        if not isinstance(farmers_with_crops, pd.DataFrame):
            farmers_with_crops = pd.read_parquet(farmers_with_crops)

        farmer_areas_m2 = _farmer_area_array_from_farmer_table(
            farmers_with_crops,
            n_farmers=n_farmers,
        )
        self.logger.info(
            "HRL calendar setup: loaded farmer table/areas for %s farmers in %.2f s.",
            n_farmers,
            time.perf_counter() - phase_timer,
        )

        phase_timer = time.perf_counter()
        farmer_locations = get_farm_locations(farms, method="centroid")
        self.logger.info(
            "HRL calendar setup: calculated %s farmer centroid locations in %.2f s.",
            n_farmers,
            time.perf_counter() - phase_timer,
        )

        # Use MIRCA-OS for both calendar timing and irrigation-area fractions,
        # matching the standard farmer crop-calendar setup workflow. The cropping-area
        # raster is loaded only as a spatial reference for the MIRCA unit grid.
        phase_timer = time.perf_counter()
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
        self.logger.info(
            "HRL calendar setup: loaded/clipped MIRCA-OS spatial reference in %.2f s.",
            time.perf_counter() - phase_timer,
        )

        phase_timer = time.perf_counter()
        rainfed_calendar_source = self.data_catalog.fetch(
            f"mirca_os_crop_calendar_{mirca_year}_rf"
        ).read()
        irrigated_calendar_source = self.data_catalog.fetch(
            f"mirca_os_crop_calendar_{mirca_year}_ir"
        ).read()
        self.logger.info(
            "HRL calendar setup: loaded MIRCA-OS crop-calendar tables in %.2f s.",
            time.perf_counter() - phase_timer,
        )

        phase_timer = time.perf_counter()
        mirca_units = mirca_unit_geom["unit_code"].astype(np.int64).tolist()
        rainfed_calendar_source = _prepare_mirca_calendar_source_for_parsing(
            rainfed_calendar_source,
            mirca_units=mirca_units,
            source_name="rainfed",
            logger=self.logger,
        )
        irrigated_calendar_source = _prepare_mirca_calendar_source_for_parsing(
            irrigated_calendar_source,
            mirca_units=mirca_units,
            source_name="irrigated",
            logger=self.logger,
        )

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
        self.logger.info(
            "HRL calendar setup: parsed MIRCA-OS calendars for %s unit(s) in %.2f s.",
            len(mirca_units),
            time.perf_counter() - phase_timer,
        )

        phase_timer = time.perf_counter()
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
        self.logger.info(
            "HRL calendar setup: rasterized MIRCA units and sampled farmers in %.2f s.",
            time.perf_counter() - phase_timer,
        )

        mirca_spatial_context = _MIRCASpatialContext(
            year=mirca_year,
            reference_crop_map=reference_crop_map,
            unit_geom=mirca_unit_geom,
            unit_grid=mirca_unit_grid,
            farmer_units=farmer_mirca_units,
            reference_map_buffer=reference_map_buffer,
        )

        # MIRCA-OS is used for the crop-specific rainfed/irrigated area fractions.
        # These fractions are static here, so changes in yearly candidate irrigation
        # assignments come from changing HRL crop assignments, not changing MIRCA-OS.
        phase_timer = time.perf_counter()
        rainfed_fraction, irrigated_fraction = self.get_mirca_os_irrigation_fractions(
            year=mirca_year,
            minimum_area_ratio=minimum_area_ratio,
            replace_crop_calendar_unit_code=replace_crop_calendar_unit_code,
            farmer_locations=farmer_locations,
            spatial_context=mirca_spatial_context,
        )
        irrigation_fraction_lookup = _prepare_mirca_irrigation_fraction_lookup(
            rainfed_fraction,
            irrigated_fraction,
        )
        self.logger.info(
            "HRL calendar setup: prepared MIRCA-OS irrigation fractions in %.2f s.",
            time.perf_counter() - phase_timer,
        )

        phase_timer = time.perf_counter()
        mirca_os_template = rainfed_fraction.isel(crop=0, drop=True)
        mirca_os_cell_grid = get_linear_indices(mirca_os_template)

        farmer_mirca_os_cells = sample_from_map(
            mirca_os_cell_grid.values,
            farmer_locations,
            mirca_os_cell_grid.rio.transform(recalc=True).to_gdal(),
        ).astype(np.int32)
        self.logger.info(
            "HRL calendar setup: sampled farmer MIRCA-OS fraction-grid cells in %.2f s.",
            time.perf_counter() - phase_timer,
        )

        phase_timer = time.perf_counter()
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
        self.logger.info(
            "HRL calendar setup: prepared surface/groundwater irrigation-source "
            "fractions in %.2f s.",
            time.perf_counter() - phase_timer,
        )

        phase_timer = time.perf_counter()
        hand = self.grid["routing/height_above_nearest_drainage_m"]
        hand = interpolate_na_2d(hand)
        farmer_hand_m = _sample_grid_values_at_farmers(hand, farmer_locations).astype(
            np.float64
        )
        self.logger.info(
            "HRL calendar setup: interpolated/sampled HAND in %.2f s.",
            time.perf_counter() - phase_timer,
        )

        phase_timer = time.perf_counter()
        farmer_groundwater_depth_m = self.load_initial_groundwater_depth_at_farmers(
            farmer_locations,
        )
        self.logger.info(
            "HRL calendar setup: prepared/sampled initial groundwater depth in %.2f s.",
            time.perf_counter() - phase_timer,
        )

        # Cache each farmer's area-weighted MIRCA-OS calendar selection so that an
        # unchanged main crop and irrigation state remains stable across HRL years.
        calendar_selection_cache: dict[
            tuple[int, int, int, bool, int],
            np.ndarray,
        ] = {}
        calendar_candidate_pool_cache: dict[
            tuple[int, int, bool],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        sequence_candidate_pool_cache: dict[
            tuple[int, int, bool],
            _MIRCACalendarCandidatePool,
        ] = {}
        expected_farmer_ids = np.arange(n_farmers, dtype=np.int32)

        if multiple_years:
            years_array = np.asarray(years_to_process, dtype=np.int32)

            # Persistent irrigation is stored only once. Later years may add farmers
            # only if their earlier HRL years did not contain a valid crop.
            persistent_adaptations: np.ndarray | None = None
            farmer_had_valid_crop_before = np.full(n_farmers, False, dtype=bool)
            farmer_main_crops_by_year = np.full(
                (years_array.size, n_farmers),
                -1,
                dtype=np.int32,
            )
            farmer_is_irrigated_by_year = np.full(
                (years_array.size, n_farmers),
                False,
                dtype=bool,
            )

        for year_index, current_hrl_year in enumerate(years_to_process):
            year_timer = time.perf_counter()
            self.logger.info(
                "Setting up HRL-based farmer crop calendars for HRL year %s.",
                current_hrl_year,
            )

            crop_column = f"crop_{current_hrl_year}"
            phase_timer = time.perf_counter()

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
            decode_seconds = time.perf_counter() - phase_timer

            # Track whether the current HRL year provides a valid MIRCA crop. In
            # multi-year mode, this controls whether later years are allowed to add
            # irrigation for this farmer.
            current_valid_crop = farmer_main_crops != -1

            phase_timer = time.perf_counter()
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
                    fraction_lookup=irrigation_fraction_lookup,
                )
            )
            irrigation_seconds = time.perf_counter() - phase_timer

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

            if multiple_years:
                # Calendar assignment is intentionally deferred until every HRL
                # crop and persistent irrigation state is known. This enables one
                # dynamic-programming pass over the complete farmer sequence and
                # avoids repeatedly revisiting partial paths.
                farmer_main_crops_by_year[year_index] = farmer_main_crops
                farmer_is_irrigated_by_year[year_index] = is_irrigated_for_calendar
                calendar_selection_seconds = 0.0
                validation_seconds = 0.0
            else:
                phase_timer = time.perf_counter()
                (
                    crop_calendar_per_farmer,
                    n_unique_calendar_keys,
                    n_new_calendar_cache_entries,
                    n_farmers_with_filtered_candidates,
                    n_filtered_calendar_candidates,
                ) = _select_mirca_calendars_for_farmers(
                    crop_calendar,
                    farmer_mirca_units=farmer_mirca_units,
                    farmer_main_crops=farmer_main_crops,
                    farmer_is_irrigated=is_irrigated_for_calendar,
                    replace_crop_calendar_unit_code=replace_crop_calendar_unit_code,
                    selection_cache=calendar_selection_cache,
                    random_seed=random_seed,
                    candidate_pool_cache=calendar_candidate_pool_cache,
                )
                calendar_selection_seconds = time.perf_counter() - phase_timer

                self.logger.info(
                    "HRL year %s crop calendars resolved from %s unique "
                    "farmer-state combination(s); %s new farmer selection(s) "
                    "added; %s shared candidate pool(s) cached; selection took "
                    "%.2f s.",
                    current_hrl_year,
                    n_unique_calendar_keys,
                    n_new_calendar_cache_entries,
                    len(calendar_candidate_pool_cache),
                    calendar_selection_seconds,
                )
                self.logger.info(
                    "HRL year %s feasibility removed %s candidate calendar(s) "
                    "across %s farmer(s).",
                    current_hrl_year,
                    n_filtered_calendar_candidates,
                    n_farmers_with_filtered_candidates,
                )

                phase_timer = time.perf_counter()
                check_crop_calendar(crop_calendar_per_farmer)
                validation_seconds = time.perf_counter() - phase_timer

                if is_europe_004_model:
                    single_year_stack, replacement_statistics = (
                        _replace_europe_004_rare_rice_sequence(
                            crop_calendar_per_farmer[None, :, :, :],
                            farmer_main_crops[None, :],
                            farmer_region_ids,
                            rice_crop_id=MIRCA_OS_CROP_CLASS_MAP["Rice"],
                        )
                    )
                    crop_calendar_per_farmer = single_year_stack[0]
                    for regional_replacement in replacement_statistics["regions"]:
                        self.logger.info(
                            "Europe_004 rare-rice exception replaced %s farmer(s) "
                            "in region %s with its most common rice-free calendar, "
                            "copied from farmer %s (%s of %s eligible farmers).",
                            regional_replacement["rice_farmer_count"],
                            regional_replacement["region_id"],
                            regional_replacement["donor_farmer_id"],
                            regional_replacement["modal_sequence_count"],
                            regional_replacement["candidate_farmer_count"],
                        )
                    check_crop_calendar(crop_calendar_per_farmer)
                    _assert_europe_004_calendar_has_no_rice(
                        single_year_stack,
                        rice_crop_id=MIRCA_OS_CROP_CLASS_MAP["Rice"],
                    )

            year_seconds = time.perf_counter() - year_timer
            timing_rows.append(
                {
                    "year": int(current_hrl_year),
                    "decode_s": decode_seconds,
                    "irrigation_s": irrigation_seconds,
                    "calendar_s": calendar_selection_seconds,
                    "validation_s": validation_seconds,
                    "total_s": year_seconds,
                }
            )
            self.logger.info(
                "HRL year %s timing: decode=%.2f s, irrigation=%.2f s, "
                "calendar=%.2f s, validation=%.2f s, total=%.2f s.",
                current_hrl_year,
                decode_seconds,
                irrigation_seconds,
                calendar_selection_seconds,
                validation_seconds,
                year_seconds,
            )

            if multiple_years:
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

            sequence_timer = time.perf_counter()
            (
                crop_calendar_stack,
                sequence_failures,
                sequence_statistics,
            ) = _select_mirca_calendar_sequences_for_farmers(
                crop_calendar,
                farmer_mirca_units=farmer_mirca_units,
                farmer_main_crops_by_year=farmer_main_crops_by_year,
                farmer_is_irrigated_by_year=farmer_is_irrigated_by_year,
                hrl_years=years_array,
                replace_crop_calendar_unit_code=replace_crop_calendar_unit_code,
                random_seed=random_seed,
                candidate_pool_cache=sequence_candidate_pool_cache,
                logger=self.logger,
            )
            sequence_selection_seconds = time.perf_counter() - sequence_timer
            self.logger.info(
                "Joint HRL calendar selection evaluated %s dynamic-programming "
                "candidate transition edges plus %s preferred-path checks for "
                "%s farmer(s) in %.2f s; preferred-path fast paths=%s; "
                "candidate pools=%s; farmer preference orders=%s; lazily "
                "ranked candidates=%s/%s; complete orders built=%s; peak "
                "per-farmer preference cache=%s; workers=%s; cumulative "
                "worker rank/DP time=%.1f/%.1f s.",
                sequence_statistics["transitions_evaluated"],
                sequence_statistics["preferred_path_transition_checks"],
                n_farmers,
                sequence_selection_seconds,
                sequence_statistics["preferred_path_fast_path_farmers"],
                sequence_statistics["candidate_pool_count"],
                sequence_statistics["preference_cache_count"],
                sequence_statistics["preference_candidates_ranked"],
                sequence_statistics["preference_candidates_available"],
                sequence_statistics["full_preference_orders_built"],
                sequence_statistics["peak_farmer_preference_cache_count"],
                sequence_statistics["worker_count"],
                sequence_statistics["rank_order_seconds"],
                sequence_statistics["dynamic_programming_seconds"],
            )
            self.logger.info(
                "Joint HRL calendar fallback use by selected farmer-year: %s; "
                "farmers using a fallback=%s; farmers whose earlier preferred "
                "calendar was reconsidered=%s.",
                sequence_statistics["selected_tier_counts"],
                sequence_statistics["farmers_using_fallback"],
                sequence_statistics["farmers_reconsidering_earlier_choice"],
            )
            self.logger.info(
                "Joint HRL calendar maximum-tier path attempts: %s.",
                sequence_statistics["maximum_tier_attempt_counts"],
            )

            if sequence_failures:
                failure_table = pd.DataFrame(sequence_failures)
                failure_summary = (
                    failure_table.groupby(
                        ["failed_year", "crop", "reason"],
                        dropna=False,
                    )
                    .size()
                    .rename("farmer_count")
                    .reset_index()
                    .sort_values(
                        ["farmer_count", "failed_year"],
                        ascending=[False, True],
                    )
                )
                unit_failure_summary = (
                    failure_table.groupby(
                        ["failed_year", "crop", "unit", "reason"],
                        dropna=False,
                    )
                    .size()
                    .rename("farmer_count")
                    .reset_index()
                    .sort_values(
                        ["farmer_count", "failed_year"],
                        ascending=[False, True],
                    )
                )
                self.logger.error(
                    "Joint crop-calendar selection found no complete feasible "
                    "path for %s farmer(s). All farmers were evaluated before "
                    "raising. Failures grouped by year/crop/reason:\n%s",
                    len(sequence_failures),
                    failure_summary.to_string(index=False),
                )
                self.logger.error(
                    "Top %s of %s failed year/crop/unit/reason groups:\n%s",
                    min(50, len(unit_failure_summary)),
                    len(unit_failure_summary),
                    unit_failure_summary.head(50).to_string(index=False),
                )
                self.logger.error(
                    "Examples of joint crop-calendar failures:\n%s",
                    failure_table.head(25).to_string(index=False),
                )
                example_records = failure_table.head(10).to_dict(orient="records")
                raise ValueError(
                    "No complete sequentially feasible MIRCA-OS calendar path "
                    f"exists for {len(sequence_failures)} of {n_farmers} "
                    "farmer(s). Every farmer was checked; grouped counts and up "
                    f"to 25 examples were logged. First examples: {example_records}."
                )

            for year_index in range(years_array.size):
                check_crop_calendar(crop_calendar_stack[year_index])

            if is_europe_004_model:
                crop_calendar_stack, replacement_statistics = (
                    _replace_europe_004_rare_rice_sequence(
                        crop_calendar_stack,
                        farmer_main_crops_by_year,
                        farmer_region_ids,
                        rice_crop_id=MIRCA_OS_CROP_CLASS_MAP["Rice"],
                    )
                )
                for regional_replacement in replacement_statistics["regions"]:
                    self.logger.info(
                        "Europe_004 rare-rice exception replaced %s farmer(s)' "
                        "complete calendar sequences in region %s with its most "
                        "common rice-free sequence, copied from farmer %s (%s "
                        "of %s eligible farmers).",
                        regional_replacement["rice_farmer_count"],
                        regional_replacement["region_id"],
                        regional_replacement["donor_farmer_id"],
                        regional_replacement["modal_sequence_count"],
                        regional_replacement["candidate_farmer_count"],
                    )
                if replacement_statistics["rice_farmer_count"]:
                    for year_index in range(years_array.size):
                        check_crop_calendar(crop_calendar_stack[year_index])

            final_sequence_summary = check_crop_calendar_sequence(
                crop_calendar_stack,
                years_array,
            )
            if is_europe_004_model:
                _assert_europe_004_calendar_has_no_rice(
                    crop_calendar_stack,
                    rice_crop_id=MIRCA_OS_CROP_CLASS_MAP["Rice"],
                )
                self.logger.info(
                    "Europe_004 final crop-calendar check passed: no rice "
                    "calendar rows remain."
                )
            self.logger.info(
                "Final HRL crop-calendar sequence validation passed: %s "
                "transitions checked; %s same-day harvest/plant transitions; "
                "minimum valid gap=%s day(s).",
                final_sequence_summary["checked_transitions"],
                final_sequence_summary["same_day_transitions"],
                final_sequence_summary["minimum_gap_days"],
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

        total_seconds = time.perf_counter() - setup_timer
        if timing_rows:
            timing_table = pd.DataFrame(timing_rows)
            self.logger.info(
                "HRL farmer crop-calendar timing by year:\n%s",
                timing_table.round(2).to_string(index=False),
            )
        final_preference_order_count = (
            int(sequence_statistics["preference_cache_count"])
            if multiple_years
            else len(calendar_selection_cache)
        )
        final_candidate_pool_cache_size = (
            len(sequence_candidate_pool_cache)
            if multiple_years
            else len(calendar_candidate_pool_cache)
        )
        self.logger.info(
            "HRL farmer crop-calendar setup finished for %s farmer(s) and %s year(s) "
            "in %.2f s; farmer-state preference orders=%s; candidate "
            "pool cache=%s shared states.",
            n_farmers,
            len(years_to_process),
            total_seconds,
            final_preference_order_count,
            final_candidate_pool_cache_size,
        )

    def get_mirca_os_irrigation_fractions(
        self,
        *,
        year: int,
        minimum_area_ratio: float = 0.01,
        replace_crop_calendar_unit_code: dict[int, int] | None = None,
        farmer_locations: np.ndarray | None = None,
        spatial_context: _MIRCASpatialContext | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Derive MIRCA-OS rainfed and irrigated crop-area fractions.

        Args:
            year: MIRCA reference year.
            minimum_area_ratio: Minimum crop-area fraction retained during sampling.
            replace_crop_calendar_unit_code: Optional MIRCA-unit replacement mapping.
            farmer_locations: Optional farmer centroid coordinates with shape
                ``(n_farmers, 2)``. Locations are derived from the farm raster when
                omitted.
            spatial_context: Optional precomputed MIRCA-OS reference map, unit
                geometry/grid, and farmer-unit sampling. Internal callers can reuse
                this to avoid repeating identical geospatial preprocessing.

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

        if spatial_context is None:
            # For alignment of various input data, load one cropping-area raster as
            # the spatial reference. This path preserves standalone/backwards-compatible
            # use of the method.
            reference_crop_map = self.data_catalog.fetch(
                f"mirca_os_cropping_area_{year}_5-arcminute_Wheat_rf"
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

            MIRCA_unit_geom = self.data_catalog.fetch(
                f"mirca_os_admin_boundaries_{year}"
            ).read()
            if not isinstance(MIRCA_unit_geom, gpd.GeoDataFrame):
                raise TypeError(
                    "MIRCA-OS administrative boundaries must be a GeoDataFrame."
                )
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
            if (farmer_mirca_units == -1).any():
                raise ValueError(
                    "All farmers should be assigned to a valid MIRCA-OS unit."
                )
        else:
            if int(spatial_context.year) != int(year):
                raise ValueError(
                    "MIRCA spatial-context year does not match requested year: "
                    f"{spatial_context.year} != {year}."
                )
            reference_crop_map = spatial_context.reference_crop_map
            reference_map_buffer = int(spatial_context.reference_map_buffer)
            MIRCA_unit_geom = spatial_context.unit_geom.copy()
            # get_crop_area_fractions may modify/fill unit assignments. Work on copies
            # so reusing this context cannot change the calendar-selection units.
            MIRCA_unit_grid = spatial_context.unit_grid.copy(deep=True)
            farmer_mirca_units = np.asarray(
                spatial_context.farmer_units,
                dtype=np.int32,
            ).copy()

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
