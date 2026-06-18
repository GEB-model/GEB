"""Class to set GEB up for Europe."""

import calendar
import gc
from datetime import date
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio import features
from shapely.geometry import box, shape as shapely_shape
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
from geb.build.methods import build_method
from geb.build.workflows.farmers import (
    assert_matching_raster_grid,
    assign_regions_to_fields,
    compact_farm_raster_values,
    create_field_index_grid,
    create_lowder_target_farm_areas,
    dominant_crop_one_year_chunked,
    farm_size_distribution_fit_by_size_class,
    get_farm_locations,
    grow_farms_from_prepared_fields,
    prepare_projected_field_arrays,
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

_FIELD_BOUNDARIES_WITH_CROPS_GEOM = "fields/field_boundaries_with_crops"
_LEGACY_FIELD_BOUNDARIES_WITH_CROP_GEOM = "fields/field_boundaries_with_crop"
_FARMERS_WITH_CROPS_TABLE = "agents/farmers/farmers_with_crops"
_DEFAULT_HRL_RASTER_CHUNKS = {"x": 4096, "y": 4096}

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


def _filter_fields_to_active_subgrid(
    fields: gpd.GeoDataFrame,
    *,
    template: xr.DataArray,
    active_mask: np.ndarray,
    field_id_column: str = "id",
    nodata: int = -1,
    all_touched: bool = False,
    logger: Any | None = None,
) -> gpd.GeoDataFrame:
    """Keep only fields represented inside the active model subgrid.

    ``set_subgrid`` masks values outside ``subgrid["mask"]`` before writing. If
    farms are compacted before that mask is applied, some farmer IDs can disappear
    during writing. This function prevents that by filtering out fields that do
    not rasterize to any active model cell.

    Args:
        fields: Field-boundary GeoDataFrame.
        template: Subgrid template used for the final farm raster.
        active_mask: Boolean array where True indicates active model cells.
        field_id_column: Unique field-ID column used for rasterization.
        nodata: Nodata value used during rasterization.
        all_touched: Whether all cells touched by a field polygon are considered.
        logger: Optional logger.

    Returns:
        Field-boundary GeoDataFrame restricted to fields represented in active
        model cells.

    Raises:
        ValueError: If ``field_id_column`` is missing.
        ValueError: If ``active_mask`` and ``template`` have different shapes.
        ValueError: If no fields remain after filtering.
    """
    if field_id_column not in fields.columns:
        raise ValueError(f"fields must contain column {field_id_column!r}.")

    if active_mask.shape != template.shape:
        raise ValueError(
            "active_mask must have the same shape as template. "
            f"Got {active_mask.shape} and {template.shape}."
        )

    fields_for_raster = fields
    if template.rio.crs is not None and fields.crs is not None:
        fields_for_raster = fields.to_crs(template.rio.crs)

    field_id_grid = rasterize_like(
        fields_for_raster,
        column=field_id_column,
        raster=template,
        dtype=np.int32,
        nodata=nodata,
        all_touched=all_touched,
    )

    represented_field_ids = np.unique(
        field_id_grid.values[(field_id_grid.values != nodata) & active_mask]
    ).astype(np.int32)

    filtered_fields = fields.loc[
        fields[field_id_column].isin(represented_field_ids)
    ].copy()

    if logger is not None and len(filtered_fields) < len(fields):
        logger.info(
            "Removed %s of %s fields because they are not represented inside "
            "the active model subgrid.",
            len(fields) - len(filtered_fields),
            len(fields),
        )

    if filtered_fields.empty:
        raise ValueError(
            "No fields remain after filtering to the active model subgrid."
        )

    return filtered_fields


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


def _filter_fields_to_hrl_clip_geometry(
    fields: gpd.GeoDataFrame,
    hrl_clip_geometry: BaseGeometry,
    *,
    geometry_crs: str = "EPSG:4326",
    logger: Any | None = None,
) -> gpd.GeoDataFrame:
    """Keep only fields whose geometry intersects the HRL clip geometry.

    This is a cheap vector prefilter before rasterization. It does not clip or
    modify field geometries; it only removes fields that are fully outside the
    exact active-mask geometry used to clip the HRL rasters.

    Args:
        fields: Field-boundary GeoDataFrame.
        hrl_clip_geometry: Active-mask geometry used for HRL raster clipping.
        geometry_crs: CRS of ``hrl_clip_geometry``.
        logger: Optional logger.

    Returns:
        GeoDataFrame restricted to fields intersecting ``hrl_clip_geometry``.

    Raises:
        ValueError: If no fields remain after filtering.
    """
    fields_for_filter = fields
    if fields.crs is not None:
        fields_for_filter = fields.to_crs(geometry_crs)

    keep = fields_for_filter.geometry.intersects(hrl_clip_geometry)
    filtered_fields = fields.loc[keep.to_numpy()].copy()

    if logger is not None and len(filtered_fields) < len(fields):
        logger.debug(
            "Removed %s of %s field boundaries before HRL rasterization because "
            "they do not intersect the active HRL clip geometry.",
            len(fields) - len(filtered_fields),
            len(fields),
        )

    if filtered_fields.empty:
        raise ValueError("No field boundaries intersect the active HRL clip geometry.")

    return filtered_fields


def _filter_fields_to_raster_footprint(
    fields: gpd.GeoDataFrame,
    raster: xr.DataArray,
    *,
    logger: Any | None = None,
) -> gpd.GeoDataFrame:
    """Keep only fields intersecting the actual raster footprint.

    This is a cheap vector filter used after reading a regional HRL raster. It
    prevents fields outside the returned raster extent from being passed to
    ``rasterize_like``. It does not clip or modify field geometries.

    Args:
        fields: Field-boundary GeoDataFrame.
        raster: Raster whose footprint should be used as the filter domain.
        logger: Optional logger.

    Returns:
        GeoDataFrame restricted to fields intersecting the raster footprint.

    Raises:
        ValueError: If the raster has no CRS.
    """
    if raster.rio.crs is None:
        raise ValueError("Cannot filter fields to raster footprint without raster CRS.")

    raster_footprint = box(*raster.rio.bounds())

    fields_for_filter = fields
    if fields.crs is not None:
        fields_for_filter = fields.to_crs(raster.rio.crs)

    keep = fields_for_filter.geometry.intersects(raster_footprint)
    filtered_fields = fields.loc[keep.to_numpy()].copy()

    if logger is not None and len(filtered_fields) < len(fields):
        logger.debug(
            "Removed %s of %s field(s) because they do not intersect the "
            "regional HRL raster footprint.",
            len(fields) - len(filtered_fields),
            len(fields),
        )

    return filtered_fields


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
    invalid_crop_values: tuple[int, ...] = (-1, 0, 65535),
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


def _field_area_m2(fields: gpd.GeoDataFrame) -> np.ndarray:
    """Calculate field areas in square metres using an equal-area projection.

    Args:
        fields: Field-boundary GeoDataFrame.

    Returns:
        One-dimensional array with field areas in square metres.

    Raises:
        ValueError: If the field GeoDataFrame has no CRS.
    """
    if fields.crs is None:
        raise ValueError("Field boundaries must have a CRS to calculate areas.")

    return fields.to_crs("EPSG:3035").geometry.area.to_numpy(dtype=np.float64)


def _sample_farm_ids_for_fields(
    fields: gpd.GeoDataFrame,
    farms: xr.DataArray,
) -> np.ndarray:
    """Sample compact farmer IDs at field representative points.

    Args:
        fields: Field-boundary GeoDataFrame.
        farms: Final compact farm raster where non-farm cells are ``-1``.

    Returns:
        One-dimensional array with sampled compact farmer IDs per field.
    """
    fields_for_sampling = fields
    if farms.rio.crs is not None and fields.crs is not None:
        fields_for_sampling = fields.to_crs(farms.rio.crs)

    representative_points = fields_for_sampling.geometry.representative_point()
    field_locations = np.column_stack(
        [
            representative_points.x.to_numpy(dtype=np.float64),
            representative_points.y.to_numpy(dtype=np.float64),
        ]
    )

    return sample_from_map(
        farms.values,
        field_locations,
        farms.rio.transform(recalc=True).to_gdal(),
    ).astype(np.int32)


def _decode_hrl_crop_combinations_from_farmer_table(
    farmers_with_crops: pd.DataFrame,
    *,
    crop_column: str,
    n_farmers: int,
    farmer_region_ids: np.ndarray,
    logger: Any,
) -> pd.DataFrame:
    """Decode final farmer-level HRL crop codes to MIRCA crop combinations.

    The input table is the compact farmer table written by
    ``setup_create_farms_from_HRL_field_boundaries``. Its ``farmer_id`` values
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
    """Validate that no overlapping crops exist per farmer calendar."""
    # this part asserts that the crop calendar is correctly set up
    # particularly that no two crops are planted at the same time
    for farmer_crop_calender in crop_calendar_per_farmer:
        farmer_crop_calender = farmer_crop_calender[farmer_crop_calender[:, -1] != -1]
        if farmer_crop_calender.shape[0] > 1:
            assert (
                np.unique(farmer_crop_calender[:, [1, 3]], axis=0).shape[0]
                == farmer_crop_calender.shape[0]
            )


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
        _, calendar = entry
        active_rows = _calendar_active_rows(calendar)

        if active_rows.shape[0] == 0:
            return False

        return main_crop in active_rows[:, 0]

    def _has_irrigation_status(entry: tuple[float, np.ndarray]) -> bool:
        _, calendar = entry
        active_rows = _calendar_active_rows(calendar)

        if active_rows.shape[0] == 0:
            return False

        return bool(active_rows[0, 1]) == is_irrigated

    def _select_best_candidate(
        candidates: list[tuple[float, np.ndarray]],
    ) -> np.ndarray | None:
        if not candidates:
            return None

        return max(candidates, key=lambda entry: entry[0])[1].astype(np.int32)

    def _select_from_candidates(
        candidates: list[tuple[float, np.ndarray]],
    ) -> np.ndarray | None:
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
        """Return the MIRCA-OS irrigated fraction for one cell-crop pair."""
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
    """Parse a MIRCA2000 crop calendar file.

    Args:
        parsed_calendar: The dictionary to store the parsed calendar in.
        crop_calendar_lines: Lines from the MIRCA2000 crop calendar file.
        MIRCA_units: The list of MIRCA unit codes to parse.
        is_irrigated: Whether the calendar is for irrigated crops.

    Returns:
        The updated parsed_calendar dictionary.
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
                "More than 2 crop rotations found, discarding the one with the lowest area. This should be fixed later."
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
            # TODO: Ensure that this only happens when the crop rotations cannot overlap.
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

    def _get_cached_field_boundaries_with_crops(
        self,
        crop_columns: list[str],
        *,
        region_id_column: str,
        country_iso3_column: str,
    ) -> gpd.GeoDataFrame | None:
        """Return a cached field crop-sequence geometry if a valid one exists.

        The current GeoParquet cache key is checked first, followed by the legacy
        singular cache key. A cached geometry is only reused if it contains the field
        ID, geometry, region ID, country ISO3 code, and all requested crop columns.

        Args:
            crop_columns: Crop-sequence columns required for the requested years.
            region_id_column: Name of the column containing model region IDs.
            country_iso3_column: Name of the column containing ISO3 country codes.

        Returns:
            Cached field-boundary GeoDataFrame with crop-sequence columns, or
            ``None`` if no valid cached geometry is available.
        """
        for geom_name in (
            _FIELD_BOUNDARIES_WITH_CROPS_GEOM,
            _LEGACY_FIELD_BOUNDARIES_WITH_CROP_GEOM,
        ):
            try:
                field_boundaries_with_crops = self.geom[geom_name]
            except AttributeError, KeyError:
                continue

            if not isinstance(field_boundaries_with_crops, gpd.GeoDataFrame):
                field_boundaries_with_crops = gpd.GeoDataFrame(
                    field_boundaries_with_crops,
                    geometry="geometry",
                )

            missing_columns = [
                column
                for column in [
                    "id",
                    "geometry",
                    region_id_column,
                    country_iso3_column,
                    *crop_columns,
                ]
                if column not in field_boundaries_with_crops.columns
            ]
            if missing_columns:
                self.logger.warning(
                    "Ignoring cached %s because it misses required column(s): %s.",
                    geom_name,
                    missing_columns,
                )
                continue

            self.logger.info(
                "Using cached field crop-sequence geometry %s with %s fields.",
                geom_name,
                len(field_boundaries_with_crops),
            )
            return field_boundaries_with_crops

        return None

    def _prepare_HRL_field_boundaries_with_crops(
        self,
        *,
        region_id_column: str,
        country_iso3_column: str,
        years: tuple[int, ...],
        chunk_rows: int,
        hrl_raster_chunks: dict[str, int] | None,
        force_recalculate: bool,
        active_subgrid_template: xr.DataArray,
        active_subgrid_mask: np.ndarray,
    ) -> gpd.GeoDataFrame:
        """Create or read field boundaries with dominant HRL crop sequences.

        The method first checks whether a reusable field crop-sequence geometry is
        already available. If not, it derives the exact active-domain geometry from
        the model subgrid mask and uses its bounds as the broad WEkEO candidate-tile
        search envelope. Field boundaries are read once within this active-domain
        envelope, filtered with a cheap vector intersection against the active
        geometry, and assigned to model regions.

        HRL crop extraction is then processed region by region. For each model
        region, the method reads and merges only the HRL crop-type and secondary-crop
        tiles needed for that region's active geometry bounds, rasterizes only that
        region's fields onto the regional HRL raster, derives dominant crop sequences
        for those fields, and immediately discards the regional rasters before moving
        to the next region. This avoids constructing one full active-domain HRL
        mosaic for every crop product and year.

        The final ``_filter_fields_to_active_subgrid`` check should still be kept
        later before farm construction to guarantee consistency with the written
        model subgrid.

        Args:
            region_id_column: Name of the column containing model region IDs.
            country_iso3_column: Name of the column containing ISO3 country codes.
            years: HRL crop years used to construct the field crop sequences.
            chunk_rows: Number of raster rows processed at once when deriving the
                dominant crop per field.
            hrl_raster_chunks: Optional xarray/rioxarray chunk sizes used when
                opening HRL rasters. If ``None``, default spatial chunks are used.
            force_recalculate: If True, ignore existing cached field crop sequences
                and recompute them from HRL rasters.
            active_subgrid_template: Model subgrid template used to derive the
                exact active-domain geometry.
            active_subgrid_mask: Boolean array where True indicates active model
                cells.

        Returns:
            Field-boundary GeoDataFrame containing field IDs, geometries, region
            IDs, ISO3 country codes, and one dominant crop column per requested
            year.

        Raises:
            ValueError: If the region table does not contain the required region
                or country column.
            ValueError: If no field boundaries are found within the active-domain
                search bounds.
            ValueError: If the field-index raster cannot be created for a region.
            ValueError: If no dominant crop sequences can be derived.
            ValueError: If no field contains a valid HRL crop observation.
        """
        crop_columns = [f"crop_{year}" for year in years]

        if not force_recalculate:
            cached = self._get_cached_field_boundaries_with_crops(
                crop_columns,
                region_id_column=region_id_column,
                country_iso3_column=country_iso3_column,
            )
            if cached is not None:
                return cached

        regions_shapes: gpd.GeoDataFrame = self.geom["regions"]
        if region_id_column not in regions_shapes.columns:
            raise ValueError(f"Region database must contain '{region_id_column}'.")

        if country_iso3_column not in regions_shapes.columns:
            raise ValueError(f"Region database must contain '{country_iso3_column}'.")

        if regions_shapes.crs is None:
            raise ValueError("Region geometries must have a CRS for HRL preprocessing.")

        hrl_active_geometry = _active_subgrid_mask_geometry_for_hrl(
            active_subgrid_template,
            active_subgrid_mask,
        )
        hrl_active_bounds = tuple(float(value) for value in hrl_active_geometry.bounds)

        self.logger.debug(
            "Using exact active-mask geometry for HRL preprocessing. WEkEO "
            "candidate search bounds=%s; exact active geometry bounds=%s.",
            hrl_active_bounds,
            hrl_active_geometry.bounds,
        )

        field_boundaries: gpd.GeoDataFrame = self.data_catalog.fetch(
            "field_boundaries"
        ).read(hrl_active_bounds)
        field_boundaries["id"] = field_boundaries["id"].astype(np.int32)

        if field_boundaries.empty:
            raise ValueError(
                "No field boundaries were found within the active-domain search "
                f"bounds {hrl_active_bounds}."
            )

        n_fields_before_geometry_filter = len(field_boundaries)
        field_boundaries = _filter_fields_to_hrl_clip_geometry(
            field_boundaries,
            hrl_active_geometry,
            logger=self.logger,
        )

        self.logger.debug(
            "Prepared %s field boundary/boundaries for region-wise HRL crop "
            "preprocessing from %s field boundary/boundaries inside active-domain "
            "search bounds %s.",
            len(field_boundaries),
            n_fields_before_geometry_filter,
            hrl_active_bounds,
        )

        field_boundaries = assign_regions_to_fields(
            field_boundaries,
            regions_shapes,
            region_id_column=region_id_column,
            country_iso3_column=country_iso3_column,
            logger=self.logger,
        )

        regions_shapes_hrl = regions_shapes[
            [region_id_column, country_iso3_column, "geometry"]
        ].copy()
        regions_shapes_hrl = regions_shapes_hrl.to_crs("EPSG:4326")

        raster_chunks = (
            _DEFAULT_HRL_RASTER_CHUNKS
            if hrl_raster_chunks is None
            else hrl_raster_chunks
        )

        field_boundaries_with_crops_per_region: list[gpd.GeoDataFrame] = []

        self.logger.debug(
            "Starting region-wise HRL crop sequence extraction for %s years and %s "
            "model region(s).",
            len(years),
            len(regions_shapes_hrl),
        )

        for region_index, (_, region) in enumerate(regions_shapes_hrl.iterrows()):
            region_id = int(region[region_id_column])
            country_iso3 = region[country_iso3_column]

            fields_region = field_boundaries.loc[
                field_boundaries[region_id_column] == region_id
            ].copy()

            if fields_region.empty:
                del fields_region
                continue

            region_active_geometry = region.geometry.intersection(hrl_active_geometry)
            if region_active_geometry.is_empty:
                self.logger.debug(
                    "Skipping region %s because it does not intersect the active "
                    "model geometry.",
                    region_id,
                )
                del fields_region
                continue

            # Use the active part of the model region as the WEkEO request
            # envelope. Do not expand this to field-derived bounds: outlier fields
            # can substantially increase the number of requested HRL tiles.
            region_bounds = tuple(
                float(value) for value in region_active_geometry.bounds
            )

            self.logger.info(
                "Processing HRL crop sequences for region %s (%s): %s field(s), "
                "WEkEO bounds=%s.",
                region_id,
                country_iso3,
                len(fields_region),
                region_bounds,
            )

            dominant_crop_per_year: list[np.ndarray] = []
            unique_field_ids: np.ndarray | None = None
            field_index_grid: np.ndarray | None = None

            for year_index, year in enumerate(years):
                self.logger.info("Processing HRL Crop Types for %s.", year)

                crop_types_adapter = self.data_catalog.fetch(
                    f"hrl_crop_types_{year}",
                    bounds=region_bounds,
                    year=year,
                )
                crop_types: xr.DataArray = crop_types_adapter.read(
                    bounds=region_bounds,
                    year=year,
                    dst_crs=None,
                    normalize_nodata=False,
                    chunks=raster_chunks,
                )

                self.logger.info("Processing HRL Secondary Crops Type for %s.", year)

                secondary_crop_adapter = self.data_catalog.fetch(
                    f"hrl_secondary_crop_{year}",
                    bounds=region_bounds,
                    year=year,
                )
                secondary_crop: xr.DataArray = secondary_crop_adapter.read(
                    bounds=region_bounds,
                    year=year,
                    dst_crs=None,
                    normalize_nodata=False,
                    chunks=raster_chunks,
                )

                crop_types, secondary_crop = _align_hrl_rasters_to_common_grid(
                    crop_types,
                    secondary_crop,
                    region_id=region_id,
                    year=year,
                    logger=self.logger,
                )

                if year_index == 0:
                    # HRL rasters stay in their native CRS to avoid a large
                    # reprojection. Reproject only the vector fields before
                    # rasterization. First remove fields that do not intersect the
                    # actual returned HRL raster footprint. This avoids expanding
                    # the WEkEO request bounds just to handle region-edge cases.
                    fields_region_for_hrl = _filter_fields_to_raster_footprint(
                        fields_region,
                        crop_types,
                        logger=self.logger,
                    )

                    if fields_region_for_hrl.empty:
                        self.logger.warning(
                            "Skipping region %s (%s) because none of its %s "
                            "field(s) intersect the returned HRL raster footprint "
                            "for year %s. WEkEO bounds=%s; HRL raster bounds=%s.",
                            region_id,
                            country_iso3,
                            len(fields_region),
                            year,
                            region_bounds,
                            crop_types.rio.bounds(),
                        )
                        del (
                            crop_types,
                            secondary_crop,
                            crop_types_adapter,
                            secondary_crop_adapter,
                            fields_region_for_hrl,
                        )
                        gc.collect()
                        field_index_grid = None
                        unique_field_ids = None
                        break

                    fields_region_for_grid = fields_region_for_hrl
                    if (
                        crop_types.rio.crs is not None
                        and fields_region_for_hrl.crs is not None
                    ):
                        fields_region_for_grid = fields_region_for_hrl.to_crs(
                            crop_types.rio.crs
                        )

                    field_boundaries_grid: xr.DataArray = rasterize_like(
                        fields_region_for_grid,
                        column="id",
                        raster=crop_types,
                        dtype=np.int32,
                        nodata=-1,
                        all_touched=False,
                    )

                    try:
                        field_index_grid, unique_field_ids = create_field_index_grid(
                            field_boundaries_grid,
                            field_nodata=-1,
                        )
                    except ValueError:
                        self.logger.debug(
                            "No fields rasterized to the HRL grid for region %s "
                            "with all_touched=False; retrying with all_touched=True.",
                            region_id,
                        )
                        del field_boundaries_grid
                        gc.collect()

                        field_boundaries_grid = rasterize_like(
                            fields_region_for_grid,
                            column="id",
                            raster=crop_types,
                            dtype=np.int32,
                            nodata=-1,
                            all_touched=True,
                        )
                        try:
                            field_index_grid, unique_field_ids = (
                                create_field_index_grid(
                                    field_boundaries_grid,
                                    field_nodata=-1,
                                )
                            )
                        except ValueError:
                            self.logger.warning(
                                "Skipping region %s (%s) because none of its %s "
                                "field(s) rasterize to the HRL crop grid for year "
                                "%s, even with all_touched=True. WEkEO bounds=%s; "
                                "HRL raster bounds=%s.",
                                region_id,
                                country_iso3,
                                len(fields_region_for_hrl),
                                year,
                                region_bounds,
                                crop_types.rio.bounds(),
                            )
                            del (
                                field_boundaries_grid,
                                fields_region_for_grid,
                                fields_region_for_hrl,
                                crop_types,
                                secondary_crop,
                                crop_types_adapter,
                                secondary_crop_adapter,
                            )
                            gc.collect()
                            field_index_grid = None
                            unique_field_ids = None
                            break

                    del (
                        field_boundaries_grid,
                        fields_region_for_grid,
                        fields_region_for_hrl,
                    )
                    gc.collect()

                if field_index_grid is None or unique_field_ids is None:
                    break

                dominant_crop_year = dominant_crop_one_year_chunked(
                    crop_types=crop_types,
                    secondary_crop=secondary_crop,
                    field_index_grid=field_index_grid,
                    n_fields=unique_field_ids.size,
                    chunk_rows=chunk_rows,
                    pair_base=65536,
                    nodata=-1,
                )
                dominant_crop_per_year.append(dominant_crop_year)

                del (
                    crop_types,
                    secondary_crop,
                    crop_types_adapter,
                    secondary_crop_adapter,
                )
                gc.collect()

            if unique_field_ids is None or len(dominant_crop_per_year) == 0:
                self.logger.warning(
                    "No dominant crop sequences could be derived for region %s; "
                    "skipping this region.",
                    region_id,
                )
                del fields_region, field_index_grid, unique_field_ids
                gc.collect()
                continue

            dominant_crop_table = pd.DataFrame(
                np.column_stack(dominant_crop_per_year).astype(np.int32),
                index=unique_field_ids,
                columns=crop_columns,
            )
            del dominant_crop_per_year, field_index_grid, unique_field_ids
            gc.collect()

            fields_region_with_crops = fields_region.merge(
                dominant_crop_table,
                left_on="id",
                right_index=True,
                how="left",
            )
            del fields_region, dominant_crop_table
            gc.collect()

            valid_crop_mask = (
                fields_region_with_crops[crop_columns].notna()
                & fields_region_with_crops[crop_columns].ne(-1)
            ).any(axis=1)
            fields_region_with_crops = fields_region_with_crops.loc[
                valid_crop_mask
            ].copy()
            del valid_crop_mask
            gc.collect()

            if fields_region_with_crops.empty:
                self.logger.warning(
                    "No field boundaries in region %s contain valid HRL crop "
                    "observations after processing.",
                    region_id,
                )
                del fields_region_with_crops
                gc.collect()
                continue

            field_boundaries_with_crops_per_region.append(fields_region_with_crops)

            self.logger.debug(
                "Finished HRL crop sequence extraction for region %s. Retained %s "
                "field(s) with valid crop observations.",
                region_id,
                len(field_boundaries_with_crops_per_region[-1]),
            )

            gc.collect()

        del field_boundaries
        gc.collect()

        if not field_boundaries_with_crops_per_region:
            raise ValueError("No dominant crop sequences could be derived.")

        field_boundaries_with_crops = gpd.GeoDataFrame(
            pd.concat(field_boundaries_with_crops_per_region, ignore_index=True),
            geometry="geometry",
            crs=field_boundaries_with_crops_per_region[0].crs,
        )
        del field_boundaries_with_crops_per_region
        gc.collect()

        if field_boundaries_with_crops.empty:
            raise ValueError("No field boundaries contain valid HRL crop observations.")

        self.set_geom(
            field_boundaries_with_crops,
            _FIELD_BOUNDARIES_WITH_CROPS_GEOM,
        )
        self.logger.info(
            "Stored field crop-sequence geometry %s with %s fields.",
            _FIELD_BOUNDARIES_WITH_CROPS_GEOM,
            len(field_boundaries_with_crops),
        )

        return field_boundaries_with_crops

    @build_method(
        depends_on=["setup_regions_and_land_use"],
        required=False,
    )
    def setup_prepare_HRL_field_boundaries_with_crops(
        self,
        region_id_column: str = "region_id",
        country_iso3_column: str = "ISO3",
        years: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023),
        chunk_rows: int = 512,
        hrl_raster_chunks: dict[str, int] | None = None,
        force_recalculate: bool = False,
    ) -> None:
        """Precompute and store field boundaries with dominant HRL crop sequences.

        This build method materializes the reusable field-level crop table used by
        farm reconstruction. It derives the exact active model geometry from the
        current subgrid mask, reads candidate field boundaries inside the active
        geometry bounds, and then extracts HRL crop sequences region by region.
        The region-wise processing avoids constructing one full active-domain HRL
        mosaic for every crop product and year.

        Args:
            region_id_column: Name of the column containing model region IDs.
            country_iso3_column: Name of the column containing ISO3 country codes.
            years: HRL crop years used to construct the field crop sequences.
            chunk_rows: Number of raster rows processed at once when deriving the
                dominant crop per field.
            hrl_raster_chunks: Optional xarray/rioxarray chunk sizes used when
                opening HRL rasters. If ``None``, default spatial chunks are used.
            force_recalculate: If True, ignore existing cached field crop sequences
                and recompute them from HRL rasters.
        """
        active_subgrid_template: xr.DataArray = self.subgrid["region_ids"].compute()
        subgrid_mask: xr.DataArray = self.subgrid["mask"].compute()
        active_subgrid_mask = ~subgrid_mask.values

        self._prepare_HRL_field_boundaries_with_crops(
            region_id_column=region_id_column,
            country_iso3_column=country_iso3_column,
            years=years,
            chunk_rows=chunk_rows,
            hrl_raster_chunks=hrl_raster_chunks,
            force_recalculate=force_recalculate,
            active_subgrid_template=active_subgrid_template,
            active_subgrid_mask=active_subgrid_mask,
        )

    @build_method(
        depends_on=["setup_regions_and_land_use"],
        required=False,
    )
    def setup_create_farms_from_HRL_field_boundaries(
        self,
        region_id_column: str = "region_id",
        country_iso3_column: str = "ISO3",
        data_source: Literal["lowder"] = "lowder",
        size_class_boundaries: dict[str, tuple[int | float, int | float]] | None = None,
        years: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023),
        random_seed: int = 42,
        chunk_rows: int = 512,
        hrl_raster_chunks: dict[str, int] | None = None,
        force_recalculate_field_crops: bool = False,
        max_distance_m: float = 500.0,
        max_neighbors: int = 32,
        distance_weight: float = 0.45,
        crop_sequence_weight: float = 0.35,
        switch_timing_weight: float = 0.20,
        target_overshoot_tolerance: float = 1.25,
        minimum_fields_per_farm: float = 1.0,
        all_touched_farm_raster: bool = False,
    ) -> None:
        """Set up farmer agents from HRL crop sequences and field boundaries.

        The method reconstructs synthetic farms from observed HRL field boundaries
        and cached field-level crop sequences. The HRL preprocessing stage is stored
        in ``fields/field_boundaries_with_crops`` and is reused unless
        ``force_recalculate_field_crops`` is True. For each model region, fields are
        projected to a local metric CRS, Lowder-derived farm-size targets are sampled
        and scaled to the available cultivated field area, and nearby fields with
        similar crop sequences are grouped into synthetic farms.

        To avoid repeated full-grid rasterization, farm geometries are collected
        while regions are processed and rasterized once after all regional farms have
        been created. Final farm IDs are compacted after the active model mask has
        been applied. The compacted farmer table is also written with representative
        HRL crop-sequence columns so later crop-calendar setup can use final farmer IDs
        directly.

        Args:
            region_id_column: Name of the column containing model region IDs.
            country_iso3_column: Name of the column containing ISO3 country codes.
            data_source: Farm-size data source. Currently only ``"lowder"`` is
                supported.
            size_class_boundaries: Optional farm-size class boundaries in square
                metres. If ``None``, default Lowder-style boundaries are used.
            years: HRL crop years used to construct or validate crop-sequence
                columns.
            random_seed: Base random seed used when sampling regional target farm
                areas.
            chunk_rows: Number of raster rows processed at once in HRL preprocessing
                and row-wise raster-copy operations.
            hrl_raster_chunks: Optional xarray/rioxarray chunk sizes used when
                opening HRL rasters during crop-sequence preprocessing.
            force_recalculate_field_crops: If True, recalculate the cached field-level
                crop table from HRL rasters before farm construction.
            max_distance_m: Maximum distance in metres for candidate neighboring
                fields during farm growing.
            max_neighbors: Maximum number of neighboring fields stored per field in
                the neighbor graph.
            distance_weight: Weight assigned to spatial proximity in the farm-growing
                candidate score.
            crop_sequence_weight: Weight assigned to crop-sequence similarity in the
                farm-growing candidate score.
            switch_timing_weight: Weight assigned to crop-switch timing similarity in
                the farm-growing candidate score.
            target_overshoot_tolerance: Maximum allowed target-area overshoot when
                adding a candidate field to a growing farm.
            minimum_fields_per_farm: Minimum expected number of fields per farm used
                when scaling Lowder farm counts to the available field area.
            all_touched_farm_raster: Whether to burn all model-grid cells touched by a
                field polygon when rasterizing final farmer IDs.

        Raises:
            ValueError: If ``data_source`` is not ``"lowder"``.
            ValueError: If the region table does not contain the required region or
                country column.
            ValueError: If field crop sequences cannot be prepared or no valid HRL
                crop observations are available.
            ValueError: If no field-based farmers can be created.
            ValueError: If the compact farm raster IDs are inconsistent with the
                compact farmer table.
        """
        if data_source != "lowder":
            raise ValueError(
                "Only the Lowder farm-size dataset is currently supported."
            )

        if size_class_boundaries is None:
            size_class_boundaries = _default_size_class_boundaries()

        regions_shapes: gpd.GeoDataFrame = self.geom["regions"]
        if region_id_column not in regions_shapes.columns:
            raise ValueError(f"Region database must contain '{region_id_column}'.")

        if country_iso3_column not in regions_shapes.columns:
            raise ValueError(f"Region database must contain '{country_iso3_column}'.")

        crop_columns = [f"crop_{year}" for year in years]
        region_ids: xr.DataArray = self.subgrid["region_ids"].compute()

        # GEB masks grid values outside the active catchment during set_subgrid().
        # Apply the same domain restriction before farm construction so farms are only
        # grown from fields that can actually survive on the written subgrid.
        subgrid_mask: xr.DataArray = self.subgrid["mask"].compute()
        active_subgrid_mask = ~subgrid_mask.values

        field_boundaries_with_crops = self._prepare_HRL_field_boundaries_with_crops(
            region_id_column=region_id_column,
            country_iso3_column=country_iso3_column,
            years=years,
            chunk_rows=chunk_rows,
            hrl_raster_chunks=hrl_raster_chunks,
            force_recalculate=force_recalculate_field_crops,
            active_subgrid_template=region_ids,
            active_subgrid_mask=active_subgrid_mask,
        )

        keep_columns = [
            "id",
            "geometry",
            region_id_column,
            country_iso3_column,
            *crop_columns,
        ]
        field_boundaries_with_crops = field_boundaries_with_crops[keep_columns].copy()

        field_boundaries_with_crops = _filter_fields_to_active_subgrid(
            field_boundaries_with_crops,
            template=region_ids,
            active_mask=active_subgrid_mask,
            field_id_column="id",
            nodata=-1,
            all_touched=all_touched_farm_raster,
            logger=self.logger,
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

        all_farmers: list[pd.DataFrame] = []
        all_assigned_fields: list[gpd.GeoDataFrame] = []
        farmer_id_offset = 0
        n_fields_with_farms = 0

        self.logger.info("Starting field-based farm construction for model regions.")

        for region_index, (_, region) in enumerate(regions_shapes.iterrows()):
            region_id = int(region[region_id_column])
            original_iso3 = region[country_iso3_column]
            iso3 = original_iso3

            self.logger.info("Setting up fields for region %s in %s.", region_id, iso3)

            fields_region = field_boundaries_with_crops.loc[
                field_boundaries_with_crops[region_id_column] == region_id
            ].copy()

            if fields_region.empty:
                del fields_region
                continue

            if iso3 in farm_size_donor_country:
                iso3 = farm_size_donor_country[iso3]
                self.logger.info(
                    "Missing farm sizes for %s; using donor country %s.",
                    original_iso3,
                    iso3,
                )

            region_farm_sizes = farm_sizes_per_region.loc[
                farm_sizes_per_region["ISO3"] == iso3
            ].drop(["Country", "Census Year", "Total"], axis=1)

            if len(region_farm_sizes) != 2:
                self.logger.warning(
                    "Skipping region %s because no complete Lowder farm-size data "
                    "are available for %s.",
                    region_id,
                    iso3,
                )
                del fields_region, region_farm_sizes
                continue

            (
                projected_fields,
                original_crs,
                field_areas_m2,
                centroid_x,
                centroid_y,
                field_sequences,
            ) = prepare_projected_field_arrays(
                fields_region,
                crop_columns,
            )

            target_farms = create_lowder_target_farm_areas(
                region_farm_sizes=region_farm_sizes,
                size_class_boundaries=size_class_boundaries,
                cultivated_field_area_m2=float(field_areas_m2.sum()),
                iso3=iso3,
                logger=self.logger,
                random_seed=random_seed + region_index,
                minimum_fields_per_farm=minimum_fields_per_farm,
                mean_field_area_m2=float(field_areas_m2.mean()),
            )

            fields_region_with_farms, farmers_region = grow_farms_from_prepared_fields(
                projected_fields=projected_fields,
                original_crs=original_crs,
                field_areas_m2=field_areas_m2,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                field_sequences=field_sequences,
                target_farms=target_farms,
                crop_columns=crop_columns,
                max_distance_m=max_distance_m,
                max_neighbors=max_neighbors,
                distance_weight=distance_weight,
                crop_sequence_weight=crop_sequence_weight,
                switch_timing_weight=switch_timing_weight,
                target_overshoot_tolerance=target_overshoot_tolerance,
            )

            fields_region_with_farms["farmer_id"] = (
                fields_region_with_farms["farmer_id"].astype(np.int32)
                + farmer_id_offset
            )

            farmers_region["farmer_id"] = (
                farmers_region["farmer_id"].astype(np.int32) + farmer_id_offset
            )
            farmers_region[region_id_column] = np.full(
                len(farmers_region),
                region_id,
                dtype=np.int32,
            )

            all_assigned_fields.append(
                fields_region_with_farms[["geometry", "farmer_id"]].copy()
            )
            all_farmers.append(farmers_region)
            farmer_id_offset += len(farmers_region)
            n_fields_with_farms += len(fields_region_with_farms)

            del (
                fields_region,
                projected_fields,
                field_areas_m2,
                centroid_x,
                centroid_y,
                field_sequences,
                target_farms,
                fields_region_with_farms,
                farmers_region,
                region_farm_sizes,
            )
            gc.collect()

        if not all_farmers or not all_assigned_fields:
            raise ValueError("No field-based farmers could be created.")

        farmers = pd.concat(all_farmers, ignore_index=True)
        farmers = farmers.sort_values("farmer_id").reset_index(drop=True)
        del all_farmers, field_boundaries_with_crops
        gc.collect()

        assigned_fields = gpd.GeoDataFrame(
            pd.concat(all_assigned_fields, ignore_index=True),
            geometry="geometry",
            crs=all_assigned_fields[0].crs,
        )
        del all_assigned_fields
        gc.collect()

        fields_for_raster = assigned_fields
        if region_ids.rio.crs is not None and assigned_fields.crs is not None:
            fields_for_raster = assigned_fields.to_crs(region_ids.rio.crs)

        self.logger.info(
            "Rasterizing %s farm-assigned field polygon(s) to the model subgrid once.",
            len(fields_for_raster),
        )

        farms = rasterize_like(
            fields_for_raster,
            column="farmer_id",
            raster=region_ids,
            dtype=np.int32,
            nodata=-1,
            all_touched=all_touched_farm_raster,
        )
        del assigned_fields, fields_for_raster
        gc.collect()

        # Mirror the mask that set_subgrid() will apply during writing. Compaction must
        # happen after this mask, otherwise farmer IDs that disappear outside the active
        # catchment create holes after saving.
        farms_values = farms.values
        farms_values[~active_subgrid_mask] = -1

        farms, farmers = compact_farm_raster_values(
            farm_values=farms_values,
            farmers=farmers,
            template=region_ids,
            farmer_id_column="farmer_id",
            nodata=-1,
            row_chunk_size=chunk_rows,
            logger=self.logger,
        )
        del farms_values
        gc.collect()

        _assert_compact_farm_ids(
            farms,
            farmers,
            farmer_id_column="farmer_id",
            nodata=-1,
        )

        self.set_table(
            farmers,
            name=_FARMERS_WITH_CROPS_TABLE,
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
            "Farm-size distribution fit by size class:\n%s",
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
            "Created %s field-based farmer agents from %s field polygons.",
            len(farmers),
            n_fields_with_farms,
        )

        self.set_subgrid(farms, name="agents/farmers/farms")
        self.set_array(
            farmers[region_id_column].to_numpy(dtype=np.int32),
            name="agents/farmers/region_id",
        )

        cultivated_land_subgrid = xr.where(
            farms != -1,
            True,
            False,
            keep_attrs=False,
        )

        if farms.rio.crs is not None:
            cultivated_land_subgrid = cultivated_land_subgrid.rio.write_crs(
                farms.rio.crs
            )

        cultivated_land_subgrid.attrs["_FillValue"] = None

        land_use_classes_subgrid: xr.DataArray = self.subgrid[
            "landsurface/land_use_classes"
        ]

        land_use_classes_subgrid = xr.where(
            cultivated_land_subgrid,
            np.int8(1),
            land_use_classes_subgrid,
            keep_attrs=True,
        ).astype(np.int8)

        if land_use_classes_subgrid.rio.crs is None and farms.rio.crs is not None:
            land_use_classes_subgrid = land_use_classes_subgrid.rio.write_crs(
                farms.rio.crs
            )

        land_use_classes_subgrid.attrs["_FillValue"] = -1

        self.set_subgrid(
            land_use_classes_subgrid,
            name="landsurface/land_use_classes",
        )

        self.set_subgrid(
            cultivated_land_subgrid,
            name="landsurface/cultivated_land",
        )

    @build_method(
        depends_on=["setup_create_farms_from_HRL_field_boundaries"], required=True
    )
    def setup_farmer_crop_calendar_from_HRL(
        self,
        hrl_year: int = 2017,
        mirca_year: int = 2015,
        minimum_area_ratio: float = 0.01,
        replace_crop_calendar_unit_code: dict[int, int] | None = None,
        multiple_years: bool = False,
        hrl_years: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023),
    ) -> None:
        """Build farmer crop calendars by combining HRL crops with MIRCA2000 calendars.

        The final compact farmer table from
        ``setup_create_farms_from_HRL_field_boundaries`` determines which HRL crop
        sequence each farmer has. HRL crop classes are mapped to MIRCA crop classes,
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

            farmer_crops_by_id = farmer_crops.set_index("farmer_id")

            # Track whether the current HRL year provides a valid MIRCA crop. In
            # multi-year mode, this controls whether later years are allowed to add
            # irrigation for this farmer.
            current_valid_crop = np.full(n_farmers, False, dtype=bool)
            valid_crop_farmer_ids = farmer_crops.loc[
                farmer_crops["mirca_crop"].to_numpy(dtype=np.int32) != -1,
                "farmer_id",
            ].to_numpy(dtype=np.int32)
            current_valid_crop[valid_crop_farmer_ids] = True

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

            crop_calendar_per_farmer = np.full((n_farmers, 3, 4), -1, dtype=np.int32)

            for farmer_id in range(n_farmers):
                row = farmer_crops_by_id.loc[farmer_id]

                selected_calendar = _select_mirca2000_calendar_for_farmer(
                    crop_calendar,
                    mirca_unit=int(farmer_mirca_units[farmer_id]),
                    main_crop=int(row["mirca_crop"]),
                    secondary_crop_type=int(row["secondary_crop_type"]),
                    is_irrigated=bool(is_irrigated_for_calendar[farmer_id]),
                    replace_crop_calendar_unit_code=replace_crop_calendar_unit_code,
                )

                crop_calendar_per_farmer[farmer_id] = selected_calendar[:, [0, 2, 3, 4]]

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
        replace_crop_calendar_unit_code: dict = {},
    ) -> xr.DataArray:
        """Derive MIRCA-OS crop-area fraction stack for rainfed or irrigated crops.

        Args:
            year: MIRCA reference year.
            minimum_area_ratio: Minimum fraction for a crop to be considered when sampling.
            replace_crop_calendar_unit_code: Optional remapping for MIRCA unit ids.

        Returns:
            DataArray with dimensions ``crop``, ``y``, and ``x``.
        """
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
            )  # use a very large buffer so that we use don't get edge effects in the interpolation
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

        MIRCA_units = (MIRCA_unit_geom.unit_code).tolist()

        farmer_locations = get_farm_locations(
            self.subgrid["agents/farmers/farms"], method="centroid"
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
