"""Class to set GEB up for Europe."""

import gc
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from geb.build.methods import build_method
from geb.build.workflows.crop_calendars import parse_MIRCA_crop_calendar
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
from geb.workflows.raster import fillna_2d, rasterize_like, sample_from_map

from .. import GEBModel
from ..workflows.conversions import setup_donor_countries

_FIELD_BOUNDARIES_WITH_CROPS_GEOM = "fields/field_boundaries_with_crops"
_LEGACY_FIELD_BOUNDARIES_WITH_CROP_GEOM = "fields/field_boundaries_with_crop"
_DEFAULT_HRL_RASTER_CHUNKS = {"x": 4096, "y": 4096}


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


def _copy_valid_farm_values_by_rows(
    target: np.ndarray,
    source: np.ndarray,
    *,
    nodata: int = -1,
    row_chunk_size: int = 512,
) -> None:
    """Copy valid farmer IDs from one raster array into another by row chunks.

    The function copies all cells from ``source`` that are not equal to
    ``nodata`` into ``target``. Rows are processed in chunks so that a temporary
    full-grid boolean mask is not created. This keeps peak memory lower when
    region-level farm rasters are burned into a shared model-scale farm raster.

    Args:
        target: Output raster array that receives valid farmer IDs.
        source: Input raster array containing farmer IDs and nodata cells.
        nodata: Value in ``source`` that should not be copied.
        row_chunk_size: Number of raster rows to process at once.

    Raises:
        ValueError: If ``target`` and ``source`` do not have the same shape.
    """
    if target.shape != source.shape:
        raise ValueError(
            "target and source farm rasters must have the same shape. "
            f"Got {target.shape} and {source.shape}."
        )

    for row_start in range(0, target.shape[0], row_chunk_size):
        row_stop = min(row_start + row_chunk_size, target.shape[0])
        source_chunk = source[row_start:row_stop]
        valid = source_chunk != nodata
        target[row_start:row_stop][valid] = source_chunk[valid]


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
    ) -> gpd.GeoDataFrame:
        """Create or read field boundaries with dominant HRL crop sequences.

        The method first checks whether a reusable field crop-sequence geometry is
        already available. If not, it reads the HRL crop-type and secondary-crop
        rasters year by year, keeps them in their native raster grid, and rasterizes
        the field boundaries once onto that grid. For each year, crop and secondary-crop
        values are combined in row chunks and reduced to the dominant encoded crop value
        per field. The resulting field-level crop geometry is filtered to fields with at
        least one valid crop observation, assigned to model regions, stored as
        ``geom/fields/field_boundaries_with_crops.geoparquet``, and returned.

        Args:
            region_id_column: Name of the column containing model region IDs.
            country_iso3_column: Name of the column containing ISO3 country codes.
            years: HRL crop years used to construct the field crop sequences.
            chunk_rows: Number of raster rows processed at once when deriving the
                dominant crop per field.
            hrl_raster_chunks: Optional xarray/rioxarray chunk sizes used when
                opening HRL rasters. If ``None``, default spatial chunks are used.
            force_recalculate: If True, ignore existing cached field crop
                sequences and recompute them from HRL rasters.

        Returns:
            Field-boundary GeoDataFrame containing field IDs, geometries, region
            IDs, ISO3 country codes, and one dominant crop column per requested
            year.

        Raises:
            ValueError: If the region table does not contain the required region
                or country column.
            ValueError: If no field boundaries are found within the model bounds.
            ValueError: If the field-index raster cannot be created.
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

        field_boundaries: gpd.GeoDataFrame = self.data_catalog.fetch(
            "field_boundaries"
        ).read(self.bounds)
        field_boundaries["id"] = field_boundaries["id"].astype(np.int32)

        if field_boundaries.empty:
            raise ValueError("No field boundaries were found within the model bounds.")

        dominant_crop_per_year: list[np.ndarray] = []
        unique_field_ids: np.ndarray | None = None
        field_index_grid: np.ndarray | None = None

        raster_chunks = (
            _DEFAULT_HRL_RASTER_CHUNKS
            if hrl_raster_chunks is None
            else hrl_raster_chunks
        )

        self.logger.info(
            "Starting HRL crop sequence extraction for %s years.", len(years)
        )

        for year_index, year in enumerate(years):
            self.logger.info(
                "Processing HRL crop and secondary-crop rasters for %s.", year
            )

            crop_types_adapter = self.data_catalog.fetch(
                f"hrl_crop_types_{year}",
                bounds=self.bounds,
                year=year,
            )
            crop_types: xr.DataArray = crop_types_adapter.read(
                bounds=self.bounds,
                year=year,
                dst_crs=None,
                normalize_nodata=False,
                chunks=raster_chunks,
            )

            secondary_crop_adapter = self.data_catalog.fetch(
                f"hrl_secondary_crop_{year}",
                bounds=self.bounds,
                year=year,
            )
            secondary_crop: xr.DataArray = secondary_crop_adapter.read(
                bounds=self.bounds,
                year=year,
                dst_crs=None,
                normalize_nodata=False,
                chunks=raster_chunks,
            )

            assert_matching_raster_grid(crop_types, secondary_crop)

            if year_index == 0:
                # HRL rasters stay in their native CRS to avoid a large reprojection.
                # Reproject only the vector fields before rasterization.
                field_boundaries_for_grid = field_boundaries
                if crop_types.rio.crs is not None and field_boundaries.crs is not None:
                    field_boundaries_for_grid = field_boundaries.to_crs(
                        crop_types.rio.crs
                    )

                field_boundaries_grid: xr.DataArray = rasterize_like(
                    field_boundaries_for_grid,
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
                del field_boundaries_grid, field_boundaries_for_grid
                gc.collect()

            if field_index_grid is None or unique_field_ids is None:
                raise ValueError("Field index grid could not be created.")

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

            del crop_types, secondary_crop, crop_types_adapter, secondary_crop_adapter
            gc.collect()

        if unique_field_ids is None or len(dominant_crop_per_year) == 0:
            raise ValueError("No dominant crop sequences could be derived.")

        dominant_crop_table = pd.DataFrame(
            np.column_stack(dominant_crop_per_year).astype(np.int32),
            index=unique_field_ids,
            columns=crop_columns,
        )
        del dominant_crop_per_year, field_index_grid, unique_field_ids
        gc.collect()

        field_boundaries_with_crops = field_boundaries.merge(
            dominant_crop_table,
            left_on="id",
            right_index=True,
            how="left",
        )
        del dominant_crop_table
        gc.collect()

        # Fields without any crop observation cannot inform farm reconstruction.
        valid_crop_mask = (
            field_boundaries_with_crops[crop_columns].notna()
            & field_boundaries_with_crops[crop_columns].ne(-1)
        ).any(axis=1)

        field_boundaries_with_crops = field_boundaries_with_crops.loc[
            valid_crop_mask
        ].copy()
        del valid_crop_mask, field_boundaries
        gc.collect()

        if field_boundaries_with_crops.empty:
            raise ValueError("No field boundaries contain valid HRL crop observations.")

        field_boundaries_with_crops = assign_regions_to_fields(
            field_boundaries_with_crops,
            regions_shapes,
            region_id_column=region_id_column,
            country_iso3_column=country_iso3_column,
            logger=self.logger,
        )

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
        farm reconstruction. It separates the expensive HRL raster-processing
        stage from the farm-growing stage, so repeated farm-reconstruction runs can
        reuse ``fields/field_boundaries_with_crops`` instead of recalculating
        dominant crops from the HRL rasters.

        Args:
            region_id_column: Name of the column containing model region IDs.
            country_iso3_column: Name of the column containing ISO3 country codes.
            years: HRL crop years used to construct the field crop sequences.
            chunk_rows: Number of raster rows processed at once when deriving the
                dominant crop per field.
            hrl_raster_chunks: Optional xarray/rioxarray chunk sizes used when
                opening HRL rasters. If ``None``, default spatial chunks are used.
            force_recalculate: If True, ignore existing cached field crop
                sequences and recompute them from HRL rasters.

        """
        self._prepare_HRL_field_boundaries_with_crops(
            region_id_column=region_id_column,
            country_iso3_column=country_iso3_column,
            years=years,
            chunk_rows=chunk_rows,
            hrl_raster_chunks=hrl_raster_chunks,
            force_recalculate=force_recalculate,
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
        and cached field-level crop sequences. The HRL preprocessing stage is
        stored in ``fields/field_boundaries_with_crops`` and is reused unless
        ``force_recalculate_field_crops`` is True. For each model region, fields
        are projected to a local metric CRS, Lowder-derived farm-size targets are
        sampled and scaled to the available cultivated field area, and nearby
        fields with similar crop sequences are grouped into synthetic farms.

        To reduce peak memory, each region is rasterized into the model-scale farm
        raster immediately after its farms are created. The method therefore keeps
        only the shared farm raster and farmer table fragments instead of storing
        all regional farm GeoDataFrames until the end. Final farm IDs are compacted
        after all regions have been processed.

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
            chunk_rows: Number of raster rows processed at once in HRL
                preprocessing and row-wise raster-copy operations.
            hrl_raster_chunks: Optional xarray/rioxarray chunk sizes used when
                opening HRL rasters during crop-sequence preprocessing.
            force_recalculate_field_crops: If True, recalculate the cached
                field-level crop table from HRL rasters before farm construction.
            max_distance_m: Maximum distance in metres for candidate neighboring
                fields during farm growing.
            max_neighbors: Maximum number of neighboring fields stored per field
                in the neighbor graph.
            distance_weight: Weight assigned to spatial proximity in the
                farm-growing candidate score.
            crop_sequence_weight: Weight assigned to crop-sequence similarity in
                the farm-growing candidate score.
            switch_timing_weight: Weight assigned to crop-switch timing similarity
                in the farm-growing candidate score.
            target_overshoot_tolerance: Maximum allowed target-area overshoot when
                adding a candidate field to a growing farm.
            minimum_fields_per_farm: Minimum expected number of fields per farm
                used when scaling Lowder farm counts to the available field area.
            all_touched_farm_raster: Whether to burn all model-grid cells touched
                by a field polygon when rasterizing final farmer IDs.

        Raises:
            ValueError: If ``data_source`` is not ``"lowder"``.
            ValueError: If the region table does not contain the required region
                or country column.
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

        field_boundaries_with_crops = self._prepare_HRL_field_boundaries_with_crops(
            region_id_column=region_id_column,
            country_iso3_column=country_iso3_column,
            years=years,
            chunk_rows=chunk_rows,
            hrl_raster_chunks=hrl_raster_chunks,
            force_recalculate=force_recalculate_field_crops,
        )

        keep_columns = [
            "id",
            "geometry",
            region_id_column,
            country_iso3_column,
            *crop_columns,
        ]
        field_boundaries_with_crops = field_boundaries_with_crops[keep_columns].copy()

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
        farm_values = np.full(region_ids.shape, -1, dtype=np.int32)
        farmer_id_offset = 0
        n_fields_with_farms = 0

        self.logger.info("Starting field-based farm construction for model regions.")

        for region_index, (_, region) in enumerate(regions_shapes.iterrows()):
            region_id = int(region[region_id_column])
            original_iso3 = region[country_iso3_column]
            iso3 = original_iso3
            self.logger.info(f"Setting up fields for region {region_id} in {iso3}..")

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

            fields_region_for_raster = fields_region_with_farms
            if region_ids.rio.crs is not None:
                fields_region_for_raster = fields_region_for_raster.to_crs(
                    region_ids.rio.crs
                )

            farms_region = rasterize_like(
                fields_region_for_raster,
                column="farmer_id",
                raster=region_ids,
                dtype=np.int32,
                nodata=-1,
                all_touched=all_touched_farm_raster,
            )
            _copy_valid_farm_values_by_rows(
                farm_values,
                farms_region.values,
                nodata=-1,
                row_chunk_size=chunk_rows,
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
                fields_region_for_raster,
                farms_region,
                farmers_region,
                region_farm_sizes,
            )
            gc.collect()

        if not all_farmers:
            raise ValueError("No field-based farmers could be created.")

        farmers = pd.concat(all_farmers, ignore_index=True)
        farmers = farmers.sort_values("farmer_id").reset_index(drop=True)
        del all_farmers, field_boundaries_with_crops
        gc.collect()

        farms, farmers = compact_farm_raster_values(
            farm_values=farm_values,
            farmers=farmers,
            template=region_ids,
            farmer_id_column="farmer_id",
            nodata=-1,
            row_chunk_size=chunk_rows,
            logger=self.logger,
        )
        del farm_values
        gc.collect()

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

        if farms.max().item() != len(farmers) - 1:
            raise ValueError(
                "Farm raster IDs are not consistent with the compact farmer table."
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
        self.set_subgrid(
            cultivated_land_subgrid,
            name="landsurface/cultivated_land",
        )

    @build_method(depends_on=["setup_create_farms"], required=True)
    def setup_farmer_crop_calendar(
        self,
        year: int = 2000,
        reduce_crops: bool = False,
        unify_variants: bool = False,
        replace_base: bool = False,
        minimum_area_ratio: float = 0.01,
        replace_crop_calendar_unit_code: dict = {},
    ) -> None:
        """Build per-farmer crop calendars for a single reference year.

        Args:
            year: Reference year (calendar year).
            reduce_crops: If True, reduce the number of crops per calendar based on area.
            unify_variants: If True, make different cropping patterns of the same crop into one.
            replace_base: If True, replace base crop definitions with alternatives.
            minimum_area_ratio: Threshold for considering a crop present in a unit.
            replace_crop_calendar_unit_code: Optional mapping to replace MIRCA unit codes.

        Raises:
            ValueError: If no rotations are found for a crop in a unit or no valid neighbor data is found.
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

        rainfed_source = self.data_catalog.fetch(
            f"mirca_os_crop_calendar_{year}_rf"
        ).read()
        rainfed_source = rainfed_source[
            rainfed_source["unit_code"].isin(MIRCA_unit_geom["unit_code"])
        ]
        irrigated_source = self.data_catalog.fetch(
            f"mirca_os_crop_calendar_{year}_ir"
        ).read()
        irrigated_source = irrigated_source[
            irrigated_source["unit_code"].isin(MIRCA_unit_geom["unit_code"])
        ]

        crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]] = {}

        MIRCA_units = (MIRCA_unit_geom.unit_code).tolist()
        crop_calendar = parse_MIRCA_crop_calendar(
            crop_calendar, rainfed_source, MIRCA_units, is_irrigated=False
        )
        crop_calendar = parse_MIRCA_crop_calendar(
            crop_calendar, irrigated_source, MIRCA_units, is_irrigated=True
        )

        def fix_365_in_crop_calendar(
            crop_calendar: dict[int, list[tuple[float, TwoDArrayInt32]]],
        ) -> dict[int, list[tuple[float, TwoDArrayInt32]]]:
            """Replace growth lengths of 365 days with 364 in the crop calendar.

            Returns:
                The crop calendar with any 365-day growth lengths clamped to 364.
            """
            for entries in crop_calendar.values():
                for _, arr in entries:
                    arr[arr[:, 3] == 365, 3] = 364

            return crop_calendar

        # Replace crop growth time of 365 with 364 as 365 leads to many issues
        crop_calendar = fix_365_in_crop_calendar(crop_calendar)

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

        farmer_crops, is_irrigated = self.assign_crops(
            reference_crop_map,
            reference_map_buffer,
            crop_calendar,
            farmer_locations,
            farmer_mirca_units,
            year,
            MIRCA_unit_grid,
            MIRCA_unit_geom,
            minimum_area_ratio=minimum_area_ratio,
            replace_crop_calendar_unit_code=replace_crop_calendar_unit_code,
        )

        self.setup_farmer_irrigation_source(is_irrigated, year)

        all_farmers_assigned = []

        crop_calendar_per_farmer = np.full((n_farmers, 3, 4), -1, dtype=np.int32)
        for mirca_unit in np.unique(farmer_mirca_units):
            farmers_in_unit = np.where(farmer_mirca_units == mirca_unit)[0]

            area_per_crop_rotation = []
            cropping_calenders_crop_rotation = []
            for crop_rotation in crop_calendar[
                replace_crop_calendar_unit_code.get(mirca_unit, mirca_unit)
            ]:
                area_per_crop_rotation.append(crop_rotation[0])
                crop_rotation_matrix = crop_rotation[1]
                starting_days = crop_rotation_matrix[:, 2]
                starting_days = starting_days[starting_days != -1]
                assert np.unique(starting_days).size == starting_days.size, (
                    "ensure all starting days are unique"
                )
                # TODO: Add check to ensure crop calendars are not overlapping.
                cropping_calenders_crop_rotation.append(crop_rotation_matrix)
            area_per_crop_rotation = np.array(area_per_crop_rotation)
            cropping_calenders_crop_rotation = np.stack(
                cropping_calenders_crop_rotation
            )

            crops_in_unit = np.unique(farmer_crops[farmers_in_unit])
            for crop_id in crops_in_unit:
                # Find rotations that include this crop
                rotations_with_crop_idx = []
                for idx, rotation in enumerate(cropping_calenders_crop_rotation):
                    # Get crop IDs in the rotation, excluding -1 entries
                    crop_ids_in_rotation = rotation[:, 0]
                    crop_ids_in_rotation = crop_ids_in_rotation[
                        crop_ids_in_rotation != -1
                    ]
                    if crop_id in crop_ids_in_rotation:
                        rotations_with_crop_idx.append(idx)

                if not rotations_with_crop_idx:
                    raise ValueError(
                        f"No rotations found for crop ID {crop_id} in mirca unit {mirca_unit}"
                    )

                # Get the area fractions and rotations for these indices
                areas_with_crop = area_per_crop_rotation[rotations_with_crop_idx]
                rotations_with_crop = cropping_calenders_crop_rotation[
                    rotations_with_crop_idx
                ]

                # Normalize the area fractions
                total_area_for_crop = areas_with_crop.sum()
                fractions = areas_with_crop / total_area_for_crop

                # Get farmers with this crop in the mirca_unit
                farmers_with_crop_in_unit = farmers_in_unit[
                    farmer_crops[farmers_in_unit] == crop_id
                ]

                # Assign crop rotations to these farmers
                assigned_rotation_indices = np.random.choice(
                    np.arange(len(rotations_with_crop)),
                    size=len(farmers_with_crop_in_unit),
                    replace=True,
                    p=fractions,
                )

                # Assign the crop calendars to the farmers
                for farmer_idx, rotation_idx in zip(
                    farmers_with_crop_in_unit, assigned_rotation_indices
                ):
                    assigned_rotation = rotations_with_crop[rotation_idx]
                    # Assign to farmer's crop calendar, taking columns [0, 2, 3, 4]
                    # Columns: [crop_id, planting_date, harvest_date, additional_attribute]
                    crop_calendar_per_farmer[farmer_idx] = assigned_rotation[
                        :, [0, 2, 3, 4]
                    ]
                    all_farmers_assigned.append(farmer_idx)

        def check_crop_calendar(crop_calendar_per_farmer: np.ndarray) -> None:
            """Validate that no overlapping crops exist per farmer calendar."""
            # this part asserts that the crop calendar is correctly set up
            # particularly that no two crops are planted at the same time
            for farmer_crop_calender in crop_calendar_per_farmer:
                farmer_crop_calender = farmer_crop_calender[
                    farmer_crop_calender[:, -1] != -1
                ]
                if farmer_crop_calender.shape[0] > 1:
                    assert (
                        np.unique(farmer_crop_calender[:, [1, 3]], axis=0).shape[0]
                        == farmer_crop_calender.shape[0]
                    )

        check_crop_calendar(crop_calendar_per_farmer)

        assert crop_calendar_per_farmer[:, :, 3].max() == 0

        self.set_array(crop_calendar_per_farmer, name="agents/farmers/crop_calendar")
        self.set_array(
            np.full_like(is_irrigated, 1, dtype=np.int32),
            name="agents/farmers/crop_calendar_rotation_years",
        )
