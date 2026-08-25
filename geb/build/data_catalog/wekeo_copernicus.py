"""Data adapter for obtaining Copernicus data from WEkEO."""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import xarray as xr
from hda import Client, Configuration
from rasterio.crs import CRS
from rioxarray import merge
from rioxarray.exceptions import NoDataInBounds, OneDimensionalRaster
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .base import Adapter

_TILE_CRS = "EPSG:3035"
_REQUEST_CRS = "EPSG:4326"
_EEA_TILE_SIZE_M = 100_000
_EEA_TILE_COORD_RE = re.compile(r"_E(?P<e>\d+)N(?P<n>\d+)_03035_")
_YEAR_COMPONENT_RE = re.compile(r"_S(?P<year>\d{4})_")

_CANONICAL_TILE_CRS = CRS.from_epsg(3035)


def _normalized_projection_parameters(crs: Any) -> dict[str, Any] | None:
    """Return a compact projection signature for robust HRL CRS comparison.

    Historical HRL GeoTIFFs and newly downloaded CDSE COGs can serialize the same
    ETRS89 / LAEA Europe CRS differently. In particular, Rasterio/GDAL 3.12 may
    report one file as ``EPSG:3035`` and another as a legacy WKT1 ``PROJCS`` whose
    spheroid inverse-flattening differs only in text precision. Direct CRS object
    equality can therefore be false even though the grids are semantically the same.

    The HRL tile grid is fixed, so compare the actual LAEA projection parameters
    instead of serialized WKT text.
    """
    if crs is None:
        return None
    try:
        candidate = CRS.from_user_input(crs)
        proj = candidate.to_dict()
    except Exception:
        return None

    result: dict[str, Any] = {}
    for key in ("proj", "ellps", "datum", "units"):
        if key in proj and proj[key] is not None:
            result[key] = str(proj[key]).lower()
    for key in ("lat_0", "lon_0", "x_0", "y_0", "k", "k_0"):
        if key in proj and proj[key] is not None:
            try:
                result[key] = float(proj[key])
            except TypeError, ValueError:
                result[key] = proj[key]
    return result


def _is_native_hrl_tile_crs(crs: Any) -> bool:
    """Return whether *crs* is equivalent to the native HRL EPSG:3035 grid."""
    if crs is None:
        return False
    try:
        candidate = CRS.from_user_input(crs)
    except Exception:
        return False
    if candidate == _CANONICAL_TILE_CRS:
        return True

    candidate_params = _normalized_projection_parameters(candidate)
    expected_params = _normalized_projection_parameters(_CANONICAL_TILE_CRS)
    if candidate_params is None or expected_params is None:
        return False

    # The core ETRS89/LAEA parameters define the fixed HRL tile grid. Datum text is
    # intentionally not required because legacy WKT1 and modern EPSG serialization
    # name ETRS89 differently while using the same GRS80 ellipsoid.
    for key in ("proj", "ellps", "units"):
        if candidate_params.get(key) != expected_params.get(key):
            return False
    for key in ("lat_0", "lon_0", "x_0", "y_0"):
        a = candidate_params.get(key)
        b = expected_params.get(key)
        if a is None or b is None or abs(float(a) - float(b)) > 1e-8:
            return False
    return True


_MANUAL_TILE_DOWNLOAD_URL = (
    "https://land.copernicus.eu/en/map-viewer?product=c6d1726c6e824ae4819bdf402b785956"
)


@dataclass(frozen=True)
class _TileDownloadStatus:
    """Small status object used to summarize tile download/cache handling."""

    tile_id: str
    status: str
    path: Path | None = None


@dataclass(frozen=True)
class _TileCacheStatus:
    """Summary of which requested tiles are already available locally."""

    tile_ids: tuple[str, ...]
    cached_tif_tile_ids: tuple[str, ...]
    cached_zip_tile_ids: tuple[str, ...]
    missing_tile_ids: tuple[str, ...]

    @property
    def total_tiles(self) -> int:
        """Total number of tiles required for the request."""
        return len(self.tile_ids)

    @property
    def cached_tiles(self) -> int:
        """Number of required tiles available as either TIFF or ZIP."""
        return len(self.cached_tif_tile_ids) + len(self.cached_zip_tile_ids)

    @property
    def missing_tiles(self) -> int:
        """Number of required tiles that still need to be downloaded."""
        return len(self.missing_tile_ids)

    @property
    def is_complete(self) -> bool:
        """Whether all required tiles are available locally."""
        return not self.missing_tile_ids


class WEkEODownloadError(RuntimeError):
    """Raised when WEkEO/HDA could not download one or more requested tiles."""


class WEkEONoCoverageError(FileNotFoundError):
    """Raised when WEkEO returns no tile coverage for a requested area."""


class WEkEOCopernicus(Adapter):
    """Downloader for WEkEO Copernicus tiles.

    Downloads and extracts the needed GeoTIFF tiles from WEkEO for a given
    bounding box or geometry mask.

    Notes:
        Tile filenames are assumed to follow the pattern
        ``CLMS_HRLVLCC_{product_code}_S{year}_R10m_{tile}_03035_V01_R00.zip``
        and the corresponding extracted TIFF is assumed to have the same basename
        with ``.tif`` extension.

        Tile identifiers can be discovered locally from the EEA 100 km grid code
        embedded in the filename (for example ``E73N22``). When a local catalogue is
        available, the adapter can therefore avoid a WEkEO search entirely. WEkEO is
        retained as an optional fallback for missing or previously unseen tiles.

        Some WEkEO tiles are known to be problematic because the API can return a
        file with the expected tile name but with incorrect underlying spatial data.
        These tiles are blocked from fresh WEkEO download unless a corrected local
        TIFF is already available.

    Attributes:
        cache_dir (Path): Directory to cache extracted TIFF tiles.
    """

    def __init__(
        self,
        *args: Any,
        dataset_id: str,
        default_query: dict[str, Any] | None = None,
        product_code: str | None = None,
        max_parallel_downloads: int | None = None,
        download_retries: int | None = None,
        download_backoff_seconds: float | None = None,
        show_download_progress: bool = False,
        prefer_local_tiles: bool = True,
        allow_wekeo_fallback: bool = True,
        normalize_nodata_values: tuple[int, ...] = (65535,),
        destination_nodata: int | None = -2,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter for WEkEO Copernicus data.

        Args:
            *args: Additional positional arguments passed to the base Adapter class.
            dataset_id: WEkEO HDA dataset identifier.
            default_query: Dataset-specific default query parameters.
            product_code: Optional HRL product code expected in the returned tile IDs,
                for example ``CTY`` for crop types or ``CPSCT`` for secondary crop
                types. If provided, WEkEO results with non-matching IDs are ignored.
            max_parallel_downloads: Maximum number of tile downloads to run in
                parallel. If None, the ``WEKEO_MAX_PARALLEL_DOWNLOADS`` environment
                variable is used, falling back to 1. Values larger than 4 can be
                unfriendly to WEkEO/HDA and to shared cluster file systems.
            download_retries: Number of retries after the first failed download
                attempt. If None, ``WEKEO_DOWNLOAD_RETRIES`` is used, falling back
                to 2.
            download_backoff_seconds: Base sleep time between retries. The actual
                delay scales linearly with the retry attempt. If None,
                ``WEKEO_DOWNLOAD_BACKOFF_SECONDS`` is used, falling back to 5 s.
            show_download_progress: Whether to allow HDA/tqdm progress bars. The
                default keeps build logs clean.
            prefer_local_tiles: If True, identify required HRL tiles from the local
                cache before querying WEkEO. Local tile discovery uses the EEA 100 km
                grid code embedded in filenames (for example ``E73N22``), so a fully
                populated local year can be used without any WEkEO catalogue request.
            allow_wekeo_fallback: If True, fall back to the WEkEO catalogue/download
                workflow when the local cache cannot identify or satisfy all required
                tiles. Set False for locally staged years that are not yet available
                through WEkEO, such as a manually populated 2024 directory.
            normalize_nodata_values: Product-defined source values to map to
                ``destination_nodata`` before any rasterio clipping, merge, or
                reprojection. The safe generic HRL default is only 65535, which the
                Croplands PUM defines as ``Outside area``. Value 65534 is deliberately
                not normalized globally because it is a legitimate quality flag for
                several Croplands confidence layers.
            destination_nodata: Nodata value used after normalization and
                reprojection. The HRL default is the signed value -2: category 0
                remains valid, while the configured outside code (65535 by default)
                is converted before any merge or warp can trigger GDAL's
                65535-to-65534 substitution. Value 65534 is deliberately preserved
                and is not a generic nodata code. Use None to leave nodata handling
                unchanged.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)
        self.dataset_id = dataset_id
        self.default_query = default_query or {}
        self.product_code = product_code.upper() if product_code is not None else None
        self.prefer_local_tiles = prefer_local_tiles
        self.allow_wekeo_fallback = allow_wekeo_fallback

        self.max_parallel_downloads = max(
            1,
            int(
                max_parallel_downloads
                if max_parallel_downloads is not None
                else os.getenv("WEKEO_MAX_PARALLEL_DOWNLOADS", "1")
            ),
        )
        self.download_retries = max(
            0,
            int(
                download_retries
                if download_retries is not None
                else os.getenv("WEKEO_DOWNLOAD_RETRIES", "2")
            ),
        )
        self.download_backoff_seconds = max(
            0.0,
            float(
                download_backoff_seconds
                if download_backoff_seconds is not None
                else os.getenv("WEKEO_DOWNLOAD_BACKOFF_SECONDS", "5")
            ),
        )
        self.normalize_nodata_values = normalize_nodata_values
        self.destination_nodata = destination_nodata

        # Local HRL catalogues are immutable during normal reads but were previously
        # rescanned for every region/year. Cache immutable tuples and invalidate them
        # explicitly whenever this adapter adds a ZIP/TIFF to the local cache.
        self._local_tile_id_cache: dict[str, tuple[str, ...]] = {}
        self._local_reference_years_cache: dict[str, tuple[str, ...]] = {}

        if not show_download_progress:
            # HDA uses tqdm internally. This avoids thousands of progress-bar lines
            # in non-interactive GEB build/update logs.
            os.environ.setdefault("TQDM_DISABLE", "1")
            logging.getLogger("hda").setLevel(logging.WARNING)

    def _invalidate_local_catalogue_cache(self, year: str | int | None = None) -> None:
        """Invalidate filename-catalogue caches after a local cache mutation."""
        if year is None:
            self._local_tile_id_cache.clear()
        else:
            self._local_tile_id_cache.pop(str(year), None)
        # Reference-year ordering depends on which numeric year directories exist.
        self._local_reference_years_cache.clear()

    def _get_client(self) -> Client:
        """Create an authenticated WEkEO HDA client.

        Returns:
            Authenticated WEkEO HDA client.

        Authentication:
            Basic auth is required and read from environment variables:
            WEKEO_USERNAME and WEKEO_PASSWORD. A new account can be made at
            https://wekeo.copernicus.eu/register .

        Raises:
            ValueError: If WEkEO credentials are not available.
        """
        username = os.getenv("WEKEO_USERNAME")
        password = os.getenv("WEKEO_PASSWORD")

        if username is None or password is None:
            raise ValueError(
                "WEKEO_USERNAME and WEKEO_PASSWORD must be set in the environment."
            )

        conf = Configuration(user=username, password=password)
        return Client(config=conf)

    def _year_dir(self, year: str | int) -> Path:
        """Return the directory containing cached files for a year.

        Args:
            year: Product year.

        Returns:
            Directory used to store the requested year's tiles.
        """
        return self.root / str(year)

    def _tile_zip_name(self, tile_id: str) -> str:
        """Return the ZIP filename for a tile result.

        Args:
            tile_id: WEkEO tile identifier.

        Returns:
            ZIP filename for the tile.
        """
        return f"{tile_id}.zip"

    def _tile_tif_name(self, tile_id: str) -> str:
        """Return the TIFF filename for a tile result.

        Args:
            tile_id: WEkEO tile identifier.

        Returns:
            TIFF filename for the tile.
        """
        return f"{tile_id}.tif"

    def _tile_zip_path(self, year: str | int, tile_id: str) -> Path:
        """Construct the local ZIP path for a tile.

        Args:
            year: Product year.
            tile_id: WEkEO tile identifier.

        Returns:
            Local ZIP file path.
        """
        return self._year_dir(year) / self._tile_zip_name(tile_id)

    def _tile_tif_path(self, year: str | int, tile_id: str) -> Path:
        """Construct the local TIFF path for a tile.

        Args:
            year: Product year.
            tile_id: WEkEO tile identifier.

        Returns:
            Local TIFF file path.
        """
        return self._year_dir(year) / self._tile_tif_name(tile_id)

    def _build_query(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
        query_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the WEkEO HDA query.

        Args:
            bounds: Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
            year: Product year.
            query_overrides: Additional dataset-specific query fields or overrides.

        Returns:
            Complete HDA query payload.
        """
        xmin, ymin, xmax, ymax = bounds

        query: dict[str, Any] = {"dataset_id": self.dataset_id}
        query.update(self.default_query)
        query["bbox"] = [xmin, ymin, xmax, ymax]
        query["year"] = str(year)

        if query_overrides is not None:
            query.update(query_overrides)

        return query

    def _matches_product_code(self, tile_id: str) -> bool:
        """Check whether a WEkEO tile ID matches the configured product code.

        Args:
            tile_id: WEkEO tile identifier.

        Returns:
            True if no product code is configured or if the tile ID contains the
            configured product code as a filename component.
        """
        if self.product_code is None:
            return True

        return f"_{self.product_code}_" in tile_id.upper()

    def _tile_grid_coordinates(self, tile_id: str) -> tuple[int, int] | None:
        """Return the EEA 100 km grid indices encoded in a HRL tile identifier.

        HRL filenames use components such as ``E73N22``. For the EEA 100 km grid,
        those indices correspond to a lower-left corner of 7,300,000 m East and
        2,200,000 m North in EPSG:3035.
        """
        match = _EEA_TILE_COORD_RE.search(tile_id)
        if match is None:
            return None
        return int(match.group("e")), int(match.group("n"))

    def _tile_intersects_projected_bounds(
        self,
        tile_id: str,
        projected_bounds: tuple[float, float, float, float],
    ) -> bool:
        """Check whether a filename-derived EEA 100 km tile intersects bounds."""
        coordinates = self._tile_grid_coordinates(tile_id)
        if coordinates is None:
            return False

        e_index, n_index = coordinates
        tile_min_x = e_index * _EEA_TILE_SIZE_M
        tile_min_y = n_index * _EEA_TILE_SIZE_M
        tile_max_x = tile_min_x + _EEA_TILE_SIZE_M
        tile_max_y = tile_min_y + _EEA_TILE_SIZE_M
        min_x, min_y, max_x, max_y = projected_bounds

        return not (
            tile_max_x <= min_x
            or tile_min_x >= max_x
            or tile_max_y <= min_y
            or tile_min_y >= max_y
        )

    def _project_bounds_to_tile_crs(
        self,
        bounds: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Project WGS84 request bounds to the native EEA tile CRS."""
        projected = (
            gpd.GeoSeries([box(*bounds)], crs=_REQUEST_CRS).to_crs(_TILE_CRS).iloc[0]
        )
        return tuple(float(value) for value in projected.bounds)

    def _scan_local_tile_ids(self, year: str | int) -> list[str]:
        """Return locally cached tile identifiers for a year from ZIP/TIFF names."""
        cache_key = str(year)
        cached = self._local_tile_id_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        year_dir = self._year_dir(year)
        if not year_dir.exists():
            self._local_tile_id_cache[cache_key] = ()
            return []

        year_component = f"_S{year}_"
        tile_ids: set[str] = set()
        for path in year_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in {
                ".zip",
                ".tif",
                ".tiff",
            }:
                continue
            tile_id = path.stem
            if year_component not in tile_id:
                continue
            if not self._matches_product_code(tile_id):
                continue
            if self._tile_grid_coordinates(tile_id) is None:
                continue
            tile_ids.add(tile_id)

        result = tuple(sorted(tile_ids))
        self._local_tile_id_cache[cache_key] = result
        return list(result)

    def _local_reference_years(self, year: str | int) -> list[str]:
        """Return local numeric year folders ordered by suitability as references."""
        cache_key = str(year)
        cached = self._local_reference_years_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        try:
            requested_year = int(year)
        except TypeError, ValueError:
            self._local_reference_years_cache[cache_key] = ()
            return []

        years: list[int] = []
        if self.root.exists():
            for path in self.root.iterdir():
                if path.is_dir() and path.name.isdigit():
                    years.append(int(path.name))

        # Prefer the closest earlier year, then the closest later year. For an
        # annual HRL product this normally selects 2023 as the reference for 2024.
        years = sorted(
            (candidate for candidate in years if candidate != requested_year),
            key=lambda candidate: (
                0 if candidate < requested_year else 1,
                abs(candidate - requested_year),
            ),
        )
        result = tuple(str(candidate) for candidate in years)
        self._local_reference_years_cache[cache_key] = result
        return list(result)

    def _replace_tile_year(self, tile_id: str, year: str | int) -> str:
        """Replace the ``SYYYY`` filename component while preserving all others."""
        return _YEAR_COMPONENT_RE.sub(f"_S{year}_", tile_id, count=1)

    def _discover_local_tiles_for_bounds(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
    ) -> list[str]:
        """Identify required tiles from local filenames without contacting WEkEO.

        The target-year directory is used when available. To detect accidentally
        missing target-year files, the closest locally available year is also used
        as a reference catalogue. Because HRL CTY/CPSCT use the same EEA 100 km grid
        every year, only the ``SYYYY`` filename component needs to be replaced.
        """
        projected_bounds = self._project_bounds_to_tile_crs(bounds)

        target_tile_ids = self._scan_local_tile_ids(year)
        target_by_coordinate = {
            coordinates: tile_id
            for tile_id in target_tile_ids
            if (coordinates := self._tile_grid_coordinates(tile_id)) is not None
        }

        reference_tile_ids: list[str] = []
        reference_year: str | None = None
        for candidate_year in self._local_reference_years(year):
            candidate_ids = self._scan_local_tile_ids(candidate_year)
            if candidate_ids:
                reference_tile_ids = candidate_ids
                reference_year = candidate_year
                break

        if reference_tile_ids:
            expected: list[str] = []
            for reference_tile_id in reference_tile_ids:
                if not self._tile_intersects_projected_bounds(
                    reference_tile_id, projected_bounds
                ):
                    continue
                coordinates = self._tile_grid_coordinates(reference_tile_id)
                assert coordinates is not None
                expected.append(
                    target_by_coordinate.get(
                        coordinates,
                        self._replace_tile_year(reference_tile_id, year),
                    )
                )

            if expected:
                self.logger.debug(
                    "Identified %s required %s tile(s) for year %s from local "
                    "filename catalogue year %s.",
                    len(expected),
                    self.product_code or "HRL",
                    year,
                    reference_year,
                )
                return sorted(set(expected))

        # If no other local year can act as a catalogue, use the target-year files
        # themselves. This is sufficient for a manually staged complete directory.
        local_matches = [
            tile_id
            for tile_id in target_tile_ids
            if self._tile_intersects_projected_bounds(tile_id, projected_bounds)
        ]
        if local_matches:
            self.logger.debug(
                "Identified %s required %s tile(s) for year %s directly from local "
                "filenames.",
                len(local_matches),
                self.product_code or "HRL",
                year,
            )
        return sorted(set(local_matches))

    def _local_tiles_or_none(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
    ) -> list[str] | None:
        """Return complete local tile IDs, or None when WEkEO fallback is allowed."""
        if not self.prefer_local_tiles:
            return None

        tile_ids = self._discover_local_tiles_for_bounds(bounds=bounds, year=year)
        if not tile_ids:
            if self.allow_wekeo_fallback:
                return None
            raise WEkEONoCoverageError(
                f"No local {self.product_code or 'HRL'} tile filenames were found for "
                f"year={year}, bounds={bounds}, and WEkEO fallback is disabled. "
                f"Expected local directory: {self._year_dir(year)}"
            )

        cache_status = self._inspect_tile_cache(tile_ids=tile_ids, year=year)
        if cache_status.is_complete:
            self.logger.info(
                "Using %s locally cached %s tile(s) for year %s; skipping WEkEO "
                "catalogue search.",
                cache_status.total_tiles,
                self.product_code or "HRL",
                year,
            )
            return tile_ids

        missing_names = [
            self._tile_zip_name(tile_id) for tile_id in cache_status.missing_tile_ids
        ]
        if not self.allow_wekeo_fallback:
            missing_preview = ", ".join(missing_names[:20])
            if len(missing_names) > 20:
                missing_preview += f", ... (+{len(missing_names) - 20} more)"
            raise FileNotFoundError(
                f"Local tile catalogue identified {cache_status.total_tiles} required "
                f"{self.product_code or 'HRL'} tile(s) for year {year}, but "
                f"{cache_status.missing_tiles} are missing and WEkEO fallback is "
                f"disabled. Missing files: {missing_preview}. Directory: "
                f"{self._year_dir(year)}"
            )

        self.logger.info(
            "Local catalogue for year %s is incomplete (%s/%s tiles cached); "
            "falling back to WEkEO for authoritative discovery/download.",
            year,
            cache_status.cached_tiles,
            cache_status.total_tiles,
        )
        return None

    def _search_tiles(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
        query_overrides: dict[str, Any] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        """Search WEkEO for tiles intersecting the requested bounds.

        Args:
            bounds: Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
            year: Product year.
            query_overrides: Additional dataset-specific query fields or overrides.

        Returns:
            A tuple containing:
                - a list of tile identifiers returned by WEkEO;
                - a lookup mapping tile identifier to downloadable WEkEO result object.

        Raises:
            WEkEONoCoverageError: If the query returns no results, or if no returned
                result IDs match the configured product code.
        """
        query = self._build_query(
            bounds=bounds,
            year=year,
            query_overrides=query_overrides,
        )

        client = self._get_client()

        matches = client.search(query)

        if len(matches.results) == 0:
            raise WEkEONoCoverageError(
                f"No WEkEO results found for year={year}, bounds={bounds}."
            )

        tile_ids: list[str] = []
        skipped_tile_ids: list[str] = []
        result_lookup: dict[str, Any] = {}

        for index, result in enumerate(matches.results):
            if isinstance(result, dict) and "id" in result:
                tile_id = str(result["id"])

                if not self._matches_product_code(tile_id):
                    skipped_tile_ids.append(tile_id)
                    continue

                tile_ids.append(tile_id)
                result_lookup[tile_id] = matches[index]

        if skipped_tile_ids:
            self.logger.debug(
                "Skipped %s WEkEO result(s) because they did not match product code %s: %s",
                len(skipped_tile_ids),
                self.product_code,
                skipped_tile_ids,
            )

        if not tile_ids:
            product_code_message = (
                ""
                if self.product_code is None
                else f" matching product_code={self.product_code!r}"
            )
            raise WEkEONoCoverageError(
                f"WEkEO returned results for query={query}, but no result IDs"
                f"{product_code_message} were found. Skipped result IDs: "
                f"{skipped_tile_ids}."
            )

        return sorted(set(tile_ids)), result_lookup

    def get_tiles_for_mask(
        self,
        mask: BaseGeometry,
        year: str | int,
        query_overrides: dict[str, Any] | None = None,
    ) -> list[str]:
        """Get tile IDs intersecting the mask, preferring the local filename catalogue.

        Args:
            mask: The geometry used to filter intersecting tiles. The input
                geometry is assumed to be in EPSG:4326.
            year: Product year.
            query_overrides: Additional dataset-specific query fields or overrides.

        Returns:
            A list of HRL tile identifiers that intersect with the mask.
        """
        local_tile_ids = self._local_tiles_or_none(bounds=mask.bounds, year=year)
        if local_tile_ids is not None:
            return local_tile_ids

        tile_ids, _ = self._search_tiles(
            bounds=mask.bounds,
            year=year,
            query_overrides=query_overrides,
        )
        return tile_ids

    def _find_downloaded_file(
        self,
        directory: Path,
        tile_id: str,
        suffixes: tuple[str, ...],
    ) -> Path | None:
        """Find a downloaded file in a temporary HDA download directory.

        Returns:
            A list of WEkEO tile identifiers that are present in the directory.
        """
        candidates = [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ]
        if not candidates:
            return None

        matching_candidates = [path for path in candidates if path.stem == tile_id]
        if matching_candidates:
            return sorted(matching_candidates)[0]
        return sorted(candidates)[0]

    def _format_download_failure(
        self,
        tile_id: str,
        year: str | int,
        target_dir: Path,
        reason: str,
    ) -> str:
        """Create a contextual error message for failed HDA tile downloads.

        Returns:
            A contextual error message for failed HDA tile downloads.
        """
        return (
            f"Failed to download WEkEO tile {tile_id!r} for year {year}. "
            f"Target directory: {target_dir}. Reason: {reason}\n"
            "If the preceding HDA log contains '401 Client Error: Unauthorized' "
            "for a 'termsaccepted' URL, the usual causes are that the WEkEO data "
            "policy/terms have not been accepted for this account, the credentials "
            "are wrong, or the HDA token expired. Log in to WEkEO once with the same "
            "account, accept the Copernicus Land Monitoring Service data policy, and "
            "check WEKEO_USERNAME/WEKEO_PASSWORD."
        )

    def _clear_raster_nodata(self, da: xr.DataArray) -> xr.DataArray:
        """Remove active raster nodata metadata without changing pixel values.

        HRL crop products are categorical rasters where values such as 65535
        should often remain available as raw class codes and be filtered
        downstream. Clearing both xarray attributes and encoding prevents stale
        nodata metadata from being propagated into rasterio operations.

        Args:
            da: DataArray whose active nodata metadata should be cleared.

        Returns:
            DataArray with the same pixel values as the input, but without active
            raster nodata metadata in the rioxarray accessor, attributes, or
            encoding.
        """
        da = da.rio.write_nodata(None, inplace=False)

        for key in ("_FillValue", "missing_value", "nodatavals"):
            da.attrs.pop(key, None)
            da.encoding.pop(key, None)

        return da

    def _prepare_categorical_nodata_for_rasterio(
        self,
        da: xr.DataArray,
    ) -> xr.DataArray:
        """Clear active nodata while preserving every raw categorical value.

        Raw mode is intended for inspection only. Clearing source metadata avoids
        GDAL rewriting the uint16 value 65535 to 65534 merely because 65535 is also
        registered as nodata. Normal model reads should use normalization, which
        converts those values to the signed destination nodata before merging.

        Args:
            da: Categorical DataArray to prepare for rasterio/rioxarray operations.

        Returns:
            DataArray with unchanged pixel values, stale nodata metadata removed,
            and no active raster nodata value.
        """
        return self._clear_raster_nodata(da)

    def _normalize_categorical_nodata_for_rasterio(
        self,
        da: xr.DataArray,
    ) -> xr.DataArray:
        """Map configured semantic outside codes to a signed model sentinel.

        This conversion is intentionally performed with xarray operations before any
        rasterio/GDAL clip, merge, or warp. For HRL Croplands, 65535 is a documented
        ``Outside area`` quality flag. Converting it to a signed sentinel early avoids
        the GDAL 65535/65534 nodata collision while preserving 65534 for products where
        that value is a legitimate quality flag.
        """
        da = self._clear_raster_nodata(da)
        if self.destination_nodata is None or not self.normalize_nodata_values:
            return da

        source_dtype = np.dtype(da.dtype)
        target_dtype = source_dtype
        if np.issubdtype(source_dtype, np.integer):
            limits = np.iinfo(source_dtype)
            if not limits.min <= self.destination_nodata <= limits.max:
                target_dtype = np.dtype(
                    np.int32 if source_dtype.itemsize <= 2 else np.int64
                )
        if target_dtype != source_dtype:
            da = da.astype(target_dtype)

        for nodata_value in self.normalize_nodata_values:
            da = da.where(da != nodata_value, self.destination_nodata)
        da = da.astype(target_dtype, copy=False)
        da = da.rio.write_nodata(self.destination_nodata, inplace=False)
        return da

    def _download_single_tile(
        self,
        tile_id: str,
        year: str | int,
        year_dir: Path,
        result: Any,
    ) -> _TileDownloadStatus:
        """Download one WEkEO tile and store it atomically in the year cache.

        The method first checks whether the requested tile is already available as an
        extracted TIFF or downloaded ZIP. If not, it downloads the tile into a
        tile-specific temporary directory, locates the returned ZIP or TIFF, and moves
        that file into the permanent year cache. The temporary directory is removed after
        each attempt, including failed attempts. Failed downloads are retried according
        to ``self.download_retries`` and ``self.download_backoff_seconds``.

        Args:
            tile_id: WEkEO tile identifier to download.
            year: Product year used to construct the cache paths.
            year_dir: Directory where cached files for the requested year are stored.
            result: WEkEO/HDA result object exposing a ``download`` method.

        Returns:
            Download status describing whether the tile was already cached or newly
            downloaded, and the local path of the cached ZIP or TIFF.

        Raises:
            WEkEODownloadError: If HDA does not create a ZIP or TIFF file, or if all
                download attempts fail.
        """
        zip_path = self._tile_zip_path(year, tile_id)
        tif_path = self._tile_tif_path(year, tile_id)

        if tif_path.exists():
            return _TileDownloadStatus(
                tile_id=tile_id, status="cached_tif", path=tif_path
            )
        if zip_path.exists():
            return _TileDownloadStatus(
                tile_id=tile_id, status="cached_zip", path=zip_path
            )

        last_error: BaseException | None = None
        attempts = self.download_retries + 1

        for attempt in range(1, attempts + 1):
            temp_dir = year_dir / f".download-{tile_id}-{uuid.uuid4().hex}"
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(parents=True, exist_ok=True)

            try:
                result.download(str(temp_dir))

                downloaded_zip = self._find_downloaded_file(
                    directory=temp_dir,
                    tile_id=tile_id,
                    suffixes=(".zip",),
                )
                downloaded_tif = self._find_downloaded_file(
                    directory=temp_dir,
                    tile_id=tile_id,
                    suffixes=(".tif", ".tiff"),
                )

                if downloaded_zip is None and downloaded_tif is None:
                    files = sorted(
                        str(path.relative_to(temp_dir)) for path in temp_dir.rglob("*")
                    )
                    raise WEkEODownloadError(
                        self._format_download_failure(
                            tile_id=tile_id,
                            year=year,
                            target_dir=year_dir,
                            reason=(
                                "HDA returned without raising an exception, but no ZIP "
                                f"or TIFF was created. Temporary files: {files or 'none'}."
                            ),
                        )
                    )

                if downloaded_zip is not None:
                    if zip_path.exists():
                        zip_path.unlink()
                    downloaded_zip.replace(zip_path)
                    return _TileDownloadStatus(
                        tile_id=tile_id,
                        status="downloaded_zip",
                        path=zip_path,
                    )

                assert downloaded_tif is not None

                if downloaded_tif.stem != tile_id:
                    raise WEkEODownloadError(
                        self._format_download_failure(
                            tile_id=tile_id,
                            year=year,
                            target_dir=year_dir,
                            reason=(
                                f"HDA returned a TIFF with unexpected name "
                                f"{downloaded_tif.name!r}. Expected stem {tile_id!r}."
                            ),
                        )
                    )

                if tif_path.exists():
                    tif_path.unlink()
                downloaded_tif.replace(tif_path)
                return _TileDownloadStatus(
                    tile_id=tile_id,
                    status="downloaded_tif",
                    path=tif_path,
                )

            except Exception as error:
                last_error = error
                if attempt < attempts:
                    self.logger.warning(
                        "Download failed for WEkEO tile %s, year %s, attempt %s/%s: %s. "
                        "Retrying after %.1f s.",
                        tile_id,
                        year,
                        attempt,
                        attempts,
                        error,
                        self.download_backoff_seconds * attempt,
                    )
                    time.sleep(self.download_backoff_seconds * attempt)
                    continue

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        assert last_error is not None
        raise WEkEODownloadError(
            self._format_download_failure(
                tile_id=tile_id,
                year=year,
                target_dir=year_dir,
                reason=str(last_error),
            )
        ) from last_error

    def _inspect_tile_cache(
        self,
        tile_ids: list[str],
        year: str | int,
    ) -> _TileCacheStatus:
        """Check all required tile paths before deciding whether to download.

        A tile is considered locally available if either the extracted TIFF exists
        or the original downloaded ZIP exists. ZIP files are treated as available
        because ``read()`` can unpack them later through ``_unpack_tiles``.

        Args:
            tile_ids: Complete list of tile identifiers required for the request.
            year: Product year.

        Returns:
            Cache summary containing cached TIFFs, cached ZIPs, and missing tiles.
        """
        cached_tif_tile_ids: list[str] = []
        cached_zip_tile_ids: list[str] = []
        missing_tile_ids: list[str] = []

        for tile_id in tile_ids:
            if self._tile_tif_path(year, tile_id).exists():
                cached_tif_tile_ids.append(tile_id)
            elif self._tile_zip_path(year, tile_id).exists():
                cached_zip_tile_ids.append(tile_id)
            else:
                missing_tile_ids.append(tile_id)

        return _TileCacheStatus(
            tile_ids=tuple(tile_ids),
            cached_tif_tile_ids=tuple(cached_tif_tile_ids),
            cached_zip_tile_ids=tuple(cached_zip_tile_ids),
            missing_tile_ids=tuple(missing_tile_ids),
        )

    def download_tiles(
        self,
        tile_ids: list[str],
        year: str | int,
        bounds: tuple[float, float, float, float],
        query_overrides: dict[str, Any] | None = None,
        result_lookup: dict[str, Any] | None = None,
    ) -> None:
        """Download WEkEO ZIP tiles for the specified tile identifiers.

        Downloads are parallelized with a thread pool because the work is primarily
        network-bound. Each tile is downloaded into its own temporary directory first,
        then moved into the permanent cache. This avoids the race condition in the old
        implementation, where several downloads could not safely compare the same
        directory's before/after file listing.

        Args:
            tile_ids: List of WEkEO tile identifiers to download.
            year: Product year.
            bounds: Requested bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
            query_overrides: Additional dataset-specific query fields or overrides.
            result_lookup: Optional lookup mapping tile identifiers to downloadable
                WEkEO result objects. If omitted, WEkEO is searched using ``bounds``,
                ``year``, and ``query_overrides``.

        Raises:
            FileNotFoundError: If a requested tile is not found in the WEkEO results.
            WEkEODownloadError: If one or more tile downloads fail.
        """
        year_dir = self._year_dir(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        if result_lookup is None:
            self.logger.debug(
                "No WEkEO result lookup was provided. Searching WEkEO before downloading "
                "tiles for year %s and bounds %s.",
                year,
                bounds,
            )
            _, result_lookup = self._search_tiles(
                bounds=bounds,
                year=year,
                query_overrides=query_overrides,
            )

        cache_status = self._inspect_tile_cache(tile_ids=tile_ids, year=year)
        tiles_to_download = list(cache_status.missing_tile_ids)

        for tile_id in tiles_to_download:
            if tile_id not in result_lookup:
                raise FileNotFoundError(
                    f"Tile {tile_id} was requested but not found in WEkEO search results."
                )

        worker_count = min(self.max_parallel_downloads, max(1, len(tiles_to_download)))
        self.logger.debug(
            "WEkEO cache preflight for year %s: %s required tile(s), "
            "%s cached TIFF(s), %s cached ZIP(s), %s missing tile(s).",
            year,
            cache_status.total_tiles,
            len(cache_status.cached_tif_tile_ids),
            len(cache_status.cached_zip_tile_ids),
            cache_status.missing_tiles,
        )

        if cache_status.is_complete:
            self.logger.debug(
                "All %s required WEkEO tile(s) for year %s are already available; "
                "skipping download stage.",
                cache_status.total_tiles,
                year,
            )
            return

        self.logger.debug(
            "Downloading %s missing WEkEO tile(s) for year %s using %s worker(s).",
            len(tiles_to_download),
            year,
            worker_count,
        )

        statuses: list[_TileDownloadStatus] = []
        failures: list[tuple[str, BaseException]] = []

        if worker_count == 1:
            for tile_id in tiles_to_download:
                try:
                    status = self._download_single_tile(
                        tile_id=tile_id,
                        year=year,
                        year_dir=year_dir,
                        result=result_lookup[tile_id],
                    )
                    statuses.append(status)
                    self.logger.debug(
                        "Downloaded WEkEO tile %s for year %s: %s",
                        status.tile_id,
                        year,
                        status.path,
                    )
                except Exception as error:
                    failures.append((tile_id, error))
                    break
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        self._download_single_tile,
                        tile_id,
                        year,
                        year_dir,
                        result_lookup[tile_id],
                    ): tile_id
                    for tile_id in tiles_to_download
                }
                for future in as_completed(futures):
                    tile_id = futures[future]
                    try:
                        status = future.result()
                    except Exception as error:
                        failures.append((tile_id, error))
                        self.logger.error(
                            "Failed to download WEkEO tile %s for year %s: %s",
                            tile_id,
                            year,
                            error,
                        )
                    else:
                        statuses.append(status)
                        self.logger.debug(
                            "Downloaded WEkEO tile %s for year %s: %s",
                            status.tile_id,
                            year,
                            status.path,
                        )

        if failures:
            details = "\n".join(f"- {tile_id}: {error}" for tile_id, error in failures)
            raise WEkEODownloadError(
                f"Failed to download {len(failures)} of {len(tiles_to_download)} "
                f"missing WEkEO tile(s) for year {year}.\n{details}"
            )

        if statuses:
            self._invalidate_local_catalogue_cache(year)

    def unpack_and_merge_tiles(
        self,
        tile_ids: list[str],
        year: str | int,
        *,
        chunks: dict[str, int] | None = None,
        clip_bounds: tuple[float, float, float, float] | None = None,
        normalize_nodata: bool = True,
    ) -> xr.DataArray:
        """Unpack and merge WEkEO tiles into a single dataarray.

        TIFFs are extracted into the permanent cache folder if needed and reused on
        subsequent reads. The tiles are opened with rioxarray and merged with
        ``rioxarray.merge.merge_arrays``, following the same general pattern as the
        MERIT Hydro adapter. If ``clip_bounds`` is provided, each tile is clipped
        before merging to reduce the amount of data passed to the merge step.

        Args:
            tile_ids: List of WEkEO tile identifiers to unpack and merge.
            year: Product year.
            chunks: Optional chunk sizes applied to the merged output. Chunks are
                intentionally not used while opening the individual source tiles,
                because dask-backed multi-tile merging can create very large graphs.
            clip_bounds: Optional bounding box in the native tile CRS, given as
                ``(min_x, min_y, max_x, max_y)``. Tiles are clipped to this box before
                merging.
            normalize_nodata: Whether to convert source nodata-like values to the
                configured destination nodata after merging. If False, active raster
                nodata metadata is removed while preserving raw pixel values.

        Returns:
            Merged dataarray of the specified tiles.
        """
        extracted_paths = self._unpack_tiles(tile_ids, year)
        da = self._merge_tiles(
            extracted_paths,
            chunks=chunks,
            clip_bounds=clip_bounds,
            normalize_nodata=normalize_nodata,
        )
        return da

    def _merge_tiles(
        self,
        tile_paths: list[Path],
        *,
        chunks: dict[str, int] | None = None,
        clip_bounds: tuple[float, float, float, float] | None = None,
        normalize_nodata: bool = True,
    ) -> xr.DataArray:
        """Merge HRL tiles using the mature pre-clip -> merge -> chunk workflow.

        This deliberately follows the older WEkEO implementation that was reliable
        for the Europe farm workflow. Source GeoTIFFs are *not* opened as Dask arrays.
        Each source is first reduced to the requested native-CRS window, then the
        clipped pieces are merged with :func:`rioxarray.merge.merge_arrays`, and Dask
        chunks are attached only to the merged result.

        Two conservative improvements are retained:

        * Source nodata metadata is cleared before clipping; for model reads, the
          documented HRL outside-area value 65535 is converted to signed ``-2`` only
          after the source has been spatially clipped. This avoids GDAL's
          65535 -> 65534 collision and avoids promoting an entire 100 km tile to int32.
        * Historical WKT1 and modern EPSG serializations of the native HRL grid are
          accepted as equivalent and rewritten to canonical EPSG:3035 before merge.
          Truly different projections are still rejected.
        """
        if not tile_paths:
            raise ValueError("No HRL tile paths were provided for merging.")

        das: list[xr.DataArray] = []
        diagnostics: list[str] = []
        skipped_paths: list[str] = []
        reference_resolution: tuple[float, float] | None = None

        for path in tile_paths:
            da = rxr.open_rasterio(
                path,
                masked=False,
                cache=False,
            ).sel(band=1, drop=True)

            source_crs = da.rio.crs
            if source_crs is None:
                da.close()
                raise ValueError(
                    f"HRL source tile {path} has no CRS in its GeoTIFF metadata."
                )
            if not _is_native_hrl_tile_crs(source_crs):
                da.close()
                raise ValueError(
                    f"HRL source tile {path.name} is not on the expected native "
                    f"ETRS89 / LAEA Europe grid ({_TILE_CRS}). Found CRS: {source_crs}."
                )

            source_resolution = tuple(float(value) for value in da.rio.resolution())
            if reference_resolution is None:
                reference_resolution = source_resolution
            elif not all(
                abs(a - b) <= 1e-8
                for a, b in zip(
                    source_resolution,
                    reference_resolution,
                    strict=True,
                )
            ):
                da.close()
                raise ValueError(
                    "Cannot merge HRL tiles with different native resolutions. "
                    f"Expected {reference_resolution}; {path.name} has "
                    f"{source_resolution}."
                )

            # First remove the UInt16 65535 nodata marker but keep pixel values raw.
            # clip_box is then only a coordinate/window operation and cannot confuse a
            # valid categorical value with an active GDAL nodata marker.
            da = self._clear_raster_nodata(da)
            da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
            da = da.rio.write_crs(_CANONICAL_TILE_CRS, inplace=False)

            if clip_bounds is not None:
                min_x, min_y, max_x, max_y = clip_bounds
                tile_min_x, tile_min_y, tile_max_x, tile_max_y = da.rio.bounds()
                intersects = not (
                    tile_max_x <= min_x
                    or tile_min_x >= max_x
                    or tile_max_y <= min_y
                    or tile_min_y >= max_y
                )
                if not intersects:
                    skipped_paths.append(path.name)
                    da.close()
                    continue

                try:
                    da = da.rio.clip_box(
                        minx=min_x,
                        miny=min_y,
                        maxx=max_x,
                        maxy=max_y,
                        allow_one_dimensional_raster=True,
                    )
                except NoDataInBounds, OneDimensionalRaster:
                    skipped_paths.append(path.name)
                    da.close()
                    continue

            # Promote/normalize only the clipped piece. With the standard CTY setup
            # this converts 65535 -> -2 and uint16 -> int32 while preserving 0 and all
            # official positive CTY class codes exactly.
            if normalize_nodata:
                da = self._normalize_categorical_nodata_for_rasterio(da)
            else:
                da = self._prepare_categorical_nodata_for_rasterio(da)

            da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
            da = da.rio.write_crs(_CANONICAL_TILE_CRS, inplace=False)
            if normalize_nodata and self.destination_nodata is not None:
                da = da.rio.write_nodata(self.destination_nodata, inplace=False)

            diagnostics.append(
                f"{path.name}: shape={da.shape}, bounds={da.rio.bounds()}, "
                f"resolution={da.rio.resolution()}, source_crs={source_crs}, "
                f"canonical_crs={da.rio.crs}, nodata={da.rio.nodata}, dtype={da.dtype}"
            )
            das.append(da)

        if not das:
            raise ValueError(
                "None of the HRL tiles intersect the requested clip bounds.\n"
                f"Clip bounds: {clip_bounds}\n"
                "Tile diagnostics:\n" + "\n".join(diagnostics)
            )

        first_dtype = das[0].dtype
        first_nodata = das[0].rio.nodata
        for da in das:
            if da.dtype != first_dtype:
                for source_da in das:
                    source_da.close()
                raise ValueError(
                    "Cannot merge HRL tiles because the clipped pieces have different "
                    "dtypes.\nTile diagnostics:\n" + "\n".join(diagnostics)
                )
            # Every accepted tile is explicitly rewritten to this canonical CRS, so a
            # serialization-only mismatch can no longer reach merge_arrays.
            if da.rio.crs != _CANONICAL_TILE_CRS:
                for source_da in das:
                    source_da.close()
                raise RuntimeError(
                    "Internal HRL CRS canonicalization failed before merge.\n"
                    + "\n".join(diagnostics)
                )

        if skipped_paths:
            self.logger.debug(
                "Skipped %s HRL tile(s) outside the requested clip bounds: %s",
                len(skipped_paths),
                skipped_paths,
            )

        merge_nodata = self.destination_nodata if normalize_nodata else None
        if normalize_nodata and merge_nodata is None:
            merge_nodata = first_nodata

        if len(das) == 1:
            # Avoid allocating a second regional array when one clipped source already
            # covers the request.
            merged = das[0]
        else:
            try:
                merged = merge.merge_arrays(das, nodata=merge_nodata)
            except Exception as error:
                for source_da in das:
                    source_da.close()
                raise ValueError(
                    "Failed to merge pre-clipped HRL tiles with "
                    "rioxarray.merge_arrays. Tile diagnostics:\n"
                    + "\n".join(diagnostics)
                ) from error
            for source_da in das:
                source_da.close()

        if "band" in merged.dims:
            merged = merged.sel(band=1, drop=True)

        merged = merged.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        merged = merged.rio.write_crs(_CANONICAL_TILE_CRS, inplace=False)
        if normalize_nodata and merge_nodata is not None:
            merged = merged.rio.write_nodata(merge_nodata, inplace=False)
        elif not normalize_nodata:
            merged = self._clear_raster_nodata(merged)

        # Match the old reliable adapter: Dask is an output-storage concern here, not
        # a source-mosaic mechanism. This keeps the merge graph small and predictable.
        if chunks is not None:
            chunk_spec = {
                dim: max(1, int(size))
                for dim, size in chunks.items()
                if dim in merged.dims
            }
            if chunk_spec:
                merged = merged.chunk(chunk_spec)

        return merged

    def _unpack_tiles(
        self,
        tile_ids: list[str],
        year: str | int,
    ) -> list[Path]:
        """Unpack the requested TIFF tiles from local ZIP files into the year folder.

        If a TIFF tile already exists locally, it is reused. If only a ZIP exists,
        the TIFF is extracted into the permanent cache folder and the ZIP is removed.

        Args:
            tile_ids: List of WEkEO tile identifiers to unpack.
            year: Product year.

        Returns:
            List of local TIFF tile paths.

        Raises:
            FileNotFoundError: If neither a TIFF nor a ZIP exists for a requested tile,
                or if a ZIP contains no TIFF.
            WEkEODownloadError: If the cache is corrupt for a certain file.
        """
        extracted_paths: list[Path] = []
        extracted_any = False

        for tile_id in tile_ids:
            tif_path = self._tile_tif_path(year, tile_id)
            zip_path = self._tile_zip_path(year, tile_id)

            if tif_path.exists():
                extracted_paths.append(tif_path)
                continue

            if not zip_path.exists():
                raise FileNotFoundError(
                    f"Neither TIFF nor ZIP found for tile {tile_id}, year {year}."
                )

            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    tif_members = [
                        name
                        for name in zip_ref.namelist()
                        if name.lower().endswith((".tif", ".tiff"))
                    ]
                    if not tif_members:
                        raise FileNotFoundError(
                            f"No TIFF found inside archive {zip_path} for tile {tile_id}, "
                            f"year {year}."
                        )

                    matching_member = next(
                        (name for name in tif_members if Path(name).stem == tile_id),
                        None,
                    )

                    if matching_member is None:
                        raise WEkEODownloadError(
                            f"ZIP for tile {tile_id}, year {year} does not contain the "
                            f"expected TIFF. Found TIFF members: {tif_members}. "
                            f"Archive: {zip_path}"
                        )

                    extracted_path = Path(
                        zip_ref.extract(matching_member, path=tif_path.parent)
                    )
            except zipfile.BadZipFile as error:
                raise WEkEODownloadError(
                    f"Cached ZIP file is corrupt for tile {tile_id}, year {year}: "
                    f"{zip_path}. "
                    "Delete this ZIP and rerun the build so it can be downloaded again."
                ) from error

            if extracted_path != tif_path:
                if tif_path.exists():
                    tif_path.unlink()
                extracted_path.replace(tif_path)

            zip_path.unlink()
            extracted_paths.append(tif_path)
            extracted_any = True

        if extracted_any:
            self._invalidate_local_catalogue_cache(year)
        return extracted_paths

    def fetch(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
        url: str | None = None,
        query_overrides: dict[str, Any] | None = None,
    ) -> WEkEOCopernicus:
        """Fetch HRL tiles for the specified bounds, preferring local files.

        Args:
            bounds: Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
            year: Product year.
            url: URL to download WEkEO data from. Defaults to None.
            query_overrides: Additional dataset-specific query fields or overrides.

        Returns:
            The adapter instance with the requested tiles available locally.
        """
        local_tile_ids = self._local_tiles_or_none(bounds=bounds, year=year)
        if local_tile_ids is not None:
            self.tile_ids = local_tile_ids
            self.year = year
            return self

        tile_ids, result_lookup = self._search_tiles(
            bounds=bounds,
            year=year,
            query_overrides=query_overrides,
        )

        self.tile_ids = tile_ids
        self.year = year

        cache_status = self._inspect_tile_cache(tile_ids=tile_ids, year=year)

        if not cache_status.is_complete:
            self.download_tiles(
                tile_ids=list(cache_status.missing_tile_ids),
                year=year,
                bounds=bounds,
                query_overrides=query_overrides,
                result_lookup=result_lookup,
            )

        return self

    def read(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int | None = None,
        query_overrides: dict[str, Any] | None = None,
        *,
        dst_crs: str | None = _REQUEST_CRS,
        normalize_nodata: bool = True,
        chunks: dict[str, int] | None = None,
    ) -> xr.DataArray:
        """Read and unpack the cached HRL data, clipping it to the requested bounds.

        Args:
            bounds: Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
            year: Product year. If None, uses the year from the most recent fetch.
            query_overrides: Additional dataset-specific query fields or overrides.
                Only used if local tile discovery cannot satisfy the request and
                WEkEO fallback is enabled.
            dst_crs: Output CRS. Use ``None`` to keep the native tile CRS and skip
                the expensive full-raster reprojection.
            normalize_nodata: Whether to map ``normalize_nodata_values`` to
                ``destination_nodata``. Set False for categorical HRL workflows that
                already treat the source nodata values as invalid downstream.
            chunks: Optional dask chunks passed to ``rioxarray.open_rasterio``.

        Returns:
            The downloaded and merged WEkEO data.

        Raises:
            ValueError: If no year is provided and no previous fetch set a year,
                or if clipping fails due to inconsistent tile spatial data.
        """
        read_year = year if year is not None else getattr(self, "year", None)
        if read_year is None:
            raise ValueError("Year must be provided to read WEkEO tiles.")

        tile_ids = getattr(self, "tile_ids", None)
        if tile_ids is None:
            mask = box(*bounds)
            tile_ids = self.get_tiles_for_mask(
                mask=mask,
                year=read_year,
                query_overrides=query_overrides,
            )

        mask_projected = (
            gpd.GeoSeries([box(*bounds)], crs=_REQUEST_CRS).to_crs(_TILE_CRS).iloc[0]
        )
        min_x, min_y, max_x, max_y = mask_projected.bounds
        clip_bounds = (min_x, min_y, max_x, max_y)

        close_callback = None
        try:
            da = self.unpack_and_merge_tiles(
                tile_ids,
                read_year,
                chunks=chunks,
                clip_bounds=clip_bounds,
                normalize_nodata=normalize_nodata,
            )
            # xarray/rioxarray indexing operations do not reliably propagate a
            # DataArray close callback. The CDSE lazy mosaic attaches one so its
            # Rasterio source managers can be released after computation. Preserve
            # it explicitly across the final safety clip and metadata operations.
            close_callback = getattr(da, "_close", None)

            # Safety clip. Most clipping has already happened tile-by-tile in
            # _merge_tiles, but this keeps the returned raster tightly aligned with
            # the requested bounds.
            da = da.rio.clip_box(
                minx=min_x,
                miny=min_y,
                maxx=max_x,
                maxy=max_y,
                allow_one_dimensional_raster=True,
            )
        except NoDataInBounds as error:
            if close_callback is not None:
                close_callback()
            data_bounds = da.rio.bounds() if "da" in locals() else None
            raise ValueError(
                "Failed to clip the merged HRL raster to the requested bounds because "
                "no raster data were found inside the requested bounding box. This can "
                "happen when WEkEO returns a tile with the expected file name but with "
                "incorrect underlying spatial data."
                f"Requested bounds in {_REQUEST_CRS}: {bounds}\n"
                f"Projected clip bounds in {_TILE_CRS}: {clip_bounds}\n"
                f"Merged raster bounds: {data_bounds}\n"
                f"Tile IDs used: {tile_ids}\n\n"
                "Retrieve the tile manually from "
                f"{_MANUAL_TILE_DOWNLOAD_URL}. "
                "Or contact: luca.battistella@eea.europa.eu"
            ) from error
        except ValueError as error:
            if close_callback is not None:
                close_callback()
            raise ValueError(
                "Failed to merge and clip HRL raster tiles.\n"
                f"Requested bounds in {_REQUEST_CRS}: {bounds}\n"
                f"Projected clip bounds in {_TILE_CRS}: {clip_bounds}\n"
                f"Tile IDs used: {tile_ids}"
            ) from error

        should_normalize_nodata = (
            normalize_nodata
            and self.destination_nodata is not None
            and bool(self.normalize_nodata_values)
        )
        if should_normalize_nodata:
            # Source nodata-like values were already mapped tile-by-tile before
            # merging to avoid rasterio/GDAL nodata collisions during the merge.
            da = da.rio.write_nodata(self.destination_nodata)

        if dst_crs is not None:
            da = da.rio.reproject(
                dst_crs,
                nodata=self.destination_nodata if normalize_nodata else None,
            )

        if not normalize_nodata:
            # Expose raw categorical values without stale nodata metadata.
            da = self._clear_raster_nodata(da)

        if close_callback is not None:
            da.set_close(close_callback)

        return da
