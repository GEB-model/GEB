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
from rioxarray import merge
from rioxarray.exceptions import NoDataInBounds, OneDimensionalRaster
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .base import Adapter

_TILE_CRS = "EPSG:3035"
_REQUEST_CRS = "EPSG:4326"

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

        Tile identifiers are taken directly from the WEkEO API results rather than
        inferred from a manually reconstructed projected tile grid. This avoids
        ambiguities in the EEA reference-grid naming convention.

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
        normalize_nodata_values: tuple[int, ...] = (65535,),
        destination_nodata: int | None = 0,
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
            normalize_nodata_values: Source values to map to ``destination_nodata``
                before reprojection. For HRL crop rasters, 65535 is a common invalid
                value and otherwise creates repeated rasterio warnings.
            destination_nodata: Nodata value used after normalization and
                reprojection. For HRL crop rasters, 0 is already treated as invalid
                downstream. Use None to leave nodata handling unchanged.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)
        self.dataset_id = dataset_id
        self.default_query = default_query or {}
        self.product_code = product_code.upper() if product_code is not None else None

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

        if not show_download_progress:
            # HDA uses tqdm internally. This avoids thousands of progress-bar lines
            # in non-interactive GEB build/update logs.
            os.environ.setdefault("TQDM_DISABLE", "1")
            logging.getLogger("hda").setLevel(logging.WARNING)

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
        """Get the WEkEO tile IDs that intersect with the mask.

        Args:
            mask: The geometry used to filter intersecting tiles. The input
                geometry is assumed to be in EPSG:4326.
            year: Product year.
            query_overrides: Additional dataset-specific query fields or overrides.

        Returns:
            A list of WEkEO tile identifiers that intersect with the mask.
        """
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
        """Use a safe active nodata value for categorical rasterio operations.

        This preserves the raw pixel values, including categorical invalid codes
        such as 65535. It only changes the active raster nodata marker used by
        rasterio/rioxarray during clip and merge operations. Using 0 avoids the
        GDAL warning where 65535 source values are changed to 65534 because 65535
        is also treated as nodata. In the HRL workflow, 0 is already handled as
        an invalid/no-crop code downstream.

        Args:
            da: Categorical DataArray to prepare for rasterio/rioxarray operations.

        Returns:
            DataArray with unchanged pixel values, stale nodata metadata removed,
            and ``self.destination_nodata`` set as the active raster nodata value
            when it is configured. If ``self.destination_nodata`` is ``None``, the
            returned DataArray has no active raster nodata value.
        """
        da = self._clear_raster_nodata(da)

        if self.destination_nodata is not None:
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
        """Merge extracted WEkEO tiles into a single xarray DataArray.

        This follows the MERIT Hydro adapter pattern: open each GeoTIFF with
        rioxarray, optionally clip individual tiles, and merge the resulting
        arrays with ``rioxarray.merge.merge_arrays``. This is generally faster
        than the lazy rectangular xarray mosaic when sufficient memory is
        available.

        Args:
            tile_paths: List of paths to extracted TIFF tiles.
            chunks: Optional chunk sizes applied to the merged output.
            clip_bounds: Optional bounding box in the native tile CRS, given as
                ``(min_x, min_y, max_x, max_y)``.
            normalize_nodata: Whether to preserve active raster nodata metadata.
                If False, active nodata metadata is removed before clipping and
                merging to preserve categorical values such as 65535.

        Returns:
            Merged dataarray of the specified tiles.

        Raises:
            ValueError: If no tile paths are provided, if no tile intersects the
                clip bounds, or if the tiles have inconsistent CRS or dtype.
        """
        if not tile_paths:
            raise ValueError("No WEkEO tile paths were provided for merging.")

        das: list[xr.DataArray] = []
        diagnostics: list[str] = []
        skipped_paths: list[str] = []

        for path in tile_paths:
            da = rxr.open_rasterio(
                path,
                masked=False,
                cache=False,
            ).sel(band=1, drop=True)

            if normalize_nodata:
                should_normalize_tile_nodata = (
                    self.destination_nodata is not None
                    and bool(self.normalize_nodata_values)
                )
                if should_normalize_tile_nodata:
                    source_dtype = da.dtype
                    for nodata_value in self.normalize_nodata_values:
                        da = da.where(da != nodata_value, self.destination_nodata)
                    da = da.astype(source_dtype)
                    da = da.rio.write_nodata(self.destination_nodata)
            else:
                da = self._prepare_categorical_nodata_for_rasterio(da)

            diagnostics.append(
                (
                    f"{path.name}: "
                    f"shape={da.shape}, "
                    f"bounds={da.rio.bounds()}, "
                    f"resolution={da.rio.resolution()}, "
                    f"crs={da.rio.crs}, "
                    f"nodata={da.rio.nodata}, "
                    f"dtype={da.dtype}"
                )
            )

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
                    continue

                try:
                    da = da.rio.clip_box(
                        minx=min_x,
                        miny=min_y,
                        maxx=max_x,
                        maxy=max_y,
                        allow_one_dimensional_raster=True,
                    )
                except NoDataInBounds:
                    skipped_paths.append(path.name)
                    continue
                except OneDimensionalRaster:
                    skipped_paths.append(path.name)
                    self.logger.debug(
                        "Skipped WEkEO tile %s because clipping to bounds %s "
                        "produced a one-dimensional raster.",
                        path.name,
                        clip_bounds,
                    )
                    continue

                if da.sizes.get("x", 0) == 1 or da.sizes.get("y", 0) == 1:
                    self.logger.debug(
                        "WEkEO tile %s clipped to a one-dimensional edge raster "
                        "with shape=%s for bounds %s.",
                        path.name,
                        da.shape,
                        clip_bounds,
                    )

            das.append(da)

        if not das:
            raise ValueError(
                "None of the WEkEO tiles intersect the requested clip bounds.\n"
                f"Clip bounds: {clip_bounds}\n"
                "Tile diagnostics:\n" + "\n".join(diagnostics)
            )

        first_dtype = das[0].dtype
        first_crs = das[0].rio.crs
        first_nodata = das[0].rio.nodata

        for da in das:
            if da.dtype != first_dtype:
                raise ValueError(
                    "Cannot merge WEkEO tiles because not all tiles have the same "
                    "dtype.\nTile diagnostics:\n" + "\n".join(diagnostics)
                )
            if da.rio.crs != first_crs:
                raise ValueError(
                    "Cannot merge WEkEO tiles because not all tiles have the same "
                    "CRS.\nTile diagnostics:\n" + "\n".join(diagnostics)
                )

        if skipped_paths:
            self.logger.debug(
                "Skipped %s WEkEO tile(s) outside the requested clip bounds: %s",
                len(skipped_paths),
                skipped_paths,
            )

        source_pixels = sum(int(da.sizes["y"]) * int(da.sizes["x"]) for da in das)
        source_gb = source_pixels * np.dtype(first_dtype).itemsize / 1024**3

        merge_nodata = self.destination_nodata
        if normalize_nodata and merge_nodata is None:
            merge_nodata = first_nodata

        source_tile_nodata_values = sorted({str(da.rio.nodata) for da in das})

        try:
            merged: xr.DataArray = merge.merge_arrays(das, nodata=merge_nodata)
        except Exception as error:
            raise ValueError(
                "Failed to merge WEkEO tiles with rioxarray.merge_arrays. "
                "Tile diagnostics:\n" + "\n".join(diagnostics)
            ) from error

        # rioxarray.merge can retain a singleton band dimension depending on input
        # metadata. Keep the adapter contract as a 2-D y/x DataArray.
        if "band" in merged.dims:
            merged = merged.sel(band=1, drop=True)

        if not normalize_nodata:
            # Keep a safe active nodata marker while read() performs its final
            # safety clip. read() clears it again before returning the array.
            merged = self._prepare_categorical_nodata_for_rasterio(merged)
        elif merge_nodata is not None:
            merged = merged.rio.write_nodata(merge_nodata)

        if chunks is not None:
            chunk_spec = {
                dim: size for dim, size in chunks.items() if dim in merged.dims
            }
            if chunk_spec:
                merged = merged.chunk(chunk_spec)

        merged_pixels = int(merged.sizes["y"]) * int(merged.sizes["x"])
        merged_gb = merged_pixels * np.dtype(merged.dtype).itemsize / 1024**3

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

        return extracted_paths

    def fetch(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
        url: str | None = None,
        query_overrides: dict[str, Any] | None = None,
    ) -> WEkEOCopernicus:
        """Fetch WEkEO tiles for the specified bounds.

        Args:
            bounds: Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
            year: Product year.
            url: URL to download WEkEO data from. Defaults to None.
            query_overrides: Additional dataset-specific query fields or overrides.

        Returns:
            The WEkEO Copernicus adapter instance with the downloaded data.
        """
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
        """Read and unpack the downloaded WEkEO data, clipping it to the requested bounds.

        Args:
            bounds: Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
            year: Product year. If None, uses the year from the most recent fetch.
            query_overrides: Additional dataset-specific query fields or overrides.
                Only used if tiles need to be re-identified from the API.
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

        try:
            da = self.unpack_and_merge_tiles(
                tile_ids,
                read_year,
                chunks=chunks,
                clip_bounds=clip_bounds,
                normalize_nodata=normalize_nodata,
            )

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
            data_bounds = da.rio.bounds() if "da" in locals() else None
            raise ValueError(
                "Failed to clip the merged WEkEO raster to the requested bounds because "
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
            raise ValueError(
                "Failed to merge and clip WEkEO raster tiles.\n"
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
        elif not normalize_nodata:
            da = self._clear_raster_nodata(da)

        if dst_crs is not None:
            reproject_nodata = self.destination_nodata if normalize_nodata else None
            da = da.rio.reproject(
                dst_crs,
                nodata=reproject_nodata,
            )

        return da


# -----------------------------------------------------------------------------
# Copernicus Data Space Ecosystem (CDSE) backend
# -----------------------------------------------------------------------------

_EEA_TILE_SIZE_M = 100_000
_EEA_TILE_COORD_RE = re.compile(r"_E(?P<e>\d+)N(?P<n>\d+)_03035_")
_YEAR_COMPONENT_RE = re.compile(r"_S(?P<year>\d{4})_")

_CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
_CDSE_S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
_CDSE_TILE_CODE_RE = re.compile(
    r"(?<![A-Z0-9])(?P<tile>E\d{2,3}N\d{2,3})(?![A-Z0-9])", re.IGNORECASE
)
_CDSE_COLLECTION_TITLE_BY_PRODUCT_CODE = {
    "CTY": "CLMS VLCC Crop Types (CTY) Europe 10m yearly V1 (COG)",
    "CPSCT": "CLMS VLCC Secondary Crops Type (CPSCT) Europe 10m yearly V1 (COG)",
}


@dataclass(frozen=True)
class _CDSEAsset:
    """Remote CDSE data asset corresponding to one canonical HRL tile."""

    tile_id: str
    href: str
    asset_key: str
    media_type: str | None = None
    item_id: str | None = None


class CDSEDownloadError(RuntimeError):
    """Raised when a tile cannot be obtained from Copernicus Data Space."""


class CDSENoCoverageError(FileNotFoundError):
    """Raised when CDSE STAC returns no matching HRL tiles."""


class CopernicusDataSpace(WEkEOCopernicus):
    """Local-first HRL adapter with CDSE STAC/S3 and optional WEkEO fallback.

    This class keeps the public ``fetch``/``read`` contract and the on-disk cache
    layout of :class:`WEkEOCopernicus`. Existing ZIP and TIFF files therefore remain
    directly usable. The acquisition order is:

    1. Identify expected 100 km EEA tiles from existing HRL filenames in the local
       catalogue and use the requested-year ZIP/TIFF files when they are complete.
    2. If files are missing (or no local reference catalogue exists), discover the
       authoritative tiles through the Copernicus Data Space STAC catalogue.
    3. Download missing COG assets from CDSE object storage and save them under the
       legacy HRL filename, as ``.tif`` files in ``root/<year>``.
    4. Optionally fall back to the existing WEkEO/HDA workflow if CDSE cannot supply
       a required tile.

    Existing adapter configuration remains valid. In particular, ``dataset_id`` and
    ``default_query`` are still accepted and are used only by the optional WEkEO
    fallback. For CTY and CPSCT, the CDSE STAC collection is resolved automatically
    from ``product_code``.
    """

    def __init__(
        self,
        *args: Any,
        dataset_id: str = "EO:EEA:DAT:HRL:CRL",
        default_query: dict[str, Any] | None = None,
        product_code: str | None = None,
        cdse_collection_id: str | None = None,
        cdse_collection_title: str | None = None,
        cdse_stac_url: str = _CDSE_STAC_URL,
        cdse_s3_endpoint: str = _CDSE_S3_ENDPOINT,
        cdse_s3_access_key: str | None = None,
        cdse_s3_secret_key: str | None = None,
        cdse_s3_session_token: str | None = None,
        cdse_access_token: str | None = None,
        cdse_request_timeout_seconds: float = 60.0,
        prefer_local_tiles: bool = True,
        allow_wekeo_fallback: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the local/CDSE/WEkEO HRL adapter.

        Args:
            *args: Additional positional arguments passed to the base adapter.
            dataset_id: Legacy WEkEO dataset ID. Kept for backwards compatibility
                and used only if ``allow_wekeo_fallback`` is True.
            default_query: Legacy WEkEO query parameters. Kept unchanged for the
                optional fallback.
            product_code: HRL product code, e.g. ``CTY`` or ``CPSCT``.
            cdse_collection_id: Optional explicit CDSE STAC collection ID. Usually
                unnecessary for CTY/CPSCT because the collection is resolved by title.
            cdse_collection_title: Optional exact STAC collection title override.
            cdse_stac_url: CDSE STAC API root.
            cdse_s3_endpoint: CDSE object-storage endpoint.
            cdse_s3_access_key: Optional S3 access key. If omitted, the adapter uses
                ``CDSE_S3_ACCESS_KEY`` and then the normal boto3/AWS credential chain.
            cdse_s3_secret_key: Optional S3 secret key. If omitted, the adapter uses
                ``CDSE_S3_SECRET_KEY`` and then the normal boto3/AWS credential chain.
            cdse_s3_session_token: Optional S3 session token.
            cdse_access_token: Optional CDSE bearer token for HTTPS assets. S3 is
                preferred for CLMS COG assets.
            cdse_request_timeout_seconds: Timeout for STAC/HTTPS requests.
            prefer_local_tiles: Use existing HRL filenames as the first source of tile
                discovery. This allows manually staged years to work without any API.
            allow_wekeo_fallback: Try the legacy WEkEO/HDA workflow if CDSE discovery
                or download cannot satisfy all required tiles.
            **kwargs: Existing ``WEkEOCopernicus`` options, including download retry,
                parallelism, and nodata settings.
        """
        super().__init__(
            *args,
            dataset_id=dataset_id,
            default_query=default_query,
            product_code=product_code,
            **kwargs,
        )
        self.prefer_local_tiles = prefer_local_tiles
        self.allow_wekeo_fallback = allow_wekeo_fallback
        self.cdse_collection_id = cdse_collection_id
        self.cdse_collection_title = cdse_collection_title
        self.cdse_stac_url = cdse_stac_url.rstrip("/")
        self.cdse_s3_endpoint = cdse_s3_endpoint.rstrip("/")
        self.cdse_s3_access_key = cdse_s3_access_key
        self.cdse_s3_secret_key = cdse_s3_secret_key
        self.cdse_s3_session_token = cdse_s3_session_token
        self.cdse_access_token = cdse_access_token
        self.cdse_request_timeout_seconds = max(
            1.0, float(cdse_request_timeout_seconds)
        )
        self._resolved_cdse_collection_id: str | None = cdse_collection_id
        self._cdse_s3_client: Any | None = None

    def _tile_grid_coordinates(self, tile_id: str) -> tuple[int, int] | None:
        """Return the EEA 100 km grid indices encoded in a HRL tile identifier."""
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
        """Return cached tile identifiers for a year from legacy ZIP/TIFF names."""
        year_dir = self._year_dir(year)
        if not year_dir.exists():
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

        return sorted(tile_ids)

    def _local_reference_years(self, year: str | int) -> list[str]:
        """Return local year folders ordered by suitability as tile references."""
        try:
            requested_year = int(year)
        except TypeError, ValueError:
            return []

        years: list[int] = []
        if self.root.exists():
            for path in self.root.iterdir():
                if path.is_dir() and path.name.isdigit():
                    years.append(int(path.name))

        years = sorted(
            (candidate for candidate in years if candidate != requested_year),
            key=lambda candidate: (
                0 if candidate < requested_year else 1,
                abs(candidate - requested_year),
            ),
        )
        return [str(candidate) for candidate in years]

    def _replace_tile_year(self, tile_id: str, year: str | int) -> str:
        """Replace only the SYYYY filename component of a legacy HRL tile ID."""
        return _YEAR_COMPONENT_RE.sub(f"_S{year}_", tile_id, count=1)

    def _discover_local_tiles_for_bounds(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
    ) -> list[str]:
        """Identify expected tiles from local legacy filenames without any API."""
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

        # If there is no other reference year, a complete manually staged target
        # year can still be used. This cannot detect an accidentally omitted file,
        # so a neighbouring reference year is preferable when available.
        local_matches = [
            tile_id
            for tile_id in target_tile_ids
            if self._tile_intersects_projected_bounds(tile_id, projected_bounds)
        ]
        return sorted(set(local_matches))

    def _requests(self) -> Any:
        """Import requests lazily so local-only use adds no hard dependency."""
        try:
            import requests
        except ImportError as error:
            raise ImportError(
                "CopernicusDataSpace requires the 'requests' package when CDSE "
                "catalogue access is needed. Existing fully cached years can still "
                "be read without it."
            ) from error
        return requests

    def _stac_request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue a retrying CDSE STAC request and return its JSON object."""
        requests = self._requests()
        attempts = self.download_retries + 1
        last_error: BaseException | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    timeout=self.cdse_request_timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise CDSEDownloadError(
                        f"CDSE STAC returned a non-object JSON response from {url}."
                    )
                return payload
            except Exception as error:
                last_error = error
                if attempt < attempts:
                    delay = self.download_backoff_seconds * attempt
                    self.logger.warning(
                        "CDSE STAC request failed (attempt %s/%s): %s. Retrying "
                        "after %.1f s.",
                        attempt,
                        attempts,
                        error,
                        delay,
                    )
                    time.sleep(delay)

        assert last_error is not None
        raise CDSEDownloadError(
            f"CDSE STAC request failed after {attempts} attempt(s): {url}: {last_error}"
        ) from last_error

    def _expected_cdse_collection_title(self) -> str | None:
        """Return the expected official STAC collection title when known."""
        if self.cdse_collection_title is not None:
            return self.cdse_collection_title
        if self.product_code is None:
            return None
        return _CDSE_COLLECTION_TITLE_BY_PRODUCT_CODE.get(self.product_code)

    def _resolve_cdse_collection_id(self) -> str:
        """Resolve the asset-level CDSE STAC collection for this HRL product."""
        if self._resolved_cdse_collection_id is not None:
            return self._resolved_cdse_collection_id

        if self.product_code is None and self.cdse_collection_title is None:
            raise ValueError(
                "A product_code, cdse_collection_title, or cdse_collection_id is "
                "required for CDSE STAC discovery."
            )

        expected_title = self._expected_cdse_collection_title()
        search_term = expected_title or self.product_code or self.cdse_collection_title
        assert search_term is not None

        # The CDSE /collections endpoint implements free-text collection search.
        payload = self._stac_request_json(
            "GET",
            f"{self.cdse_stac_url}/collections",
            params={"q": search_term},
        )
        collections = payload.get("collections", [])
        if not isinstance(collections, list):
            collections = []

        if expected_title is not None:
            exact = [
                collection
                for collection in collections
                if str(collection.get("title", "")).casefold()
                == expected_title.casefold()
            ]
            if len(exact) == 1:
                collection_id = str(exact[0]["id"])
                self._resolved_cdse_collection_id = collection_id
                return collection_id

        product_code_token = (
            f"({self.product_code})" if self.product_code is not None else None
        )
        candidates: list[dict[str, Any]] = []
        for collection in collections:
            title = str(collection.get("title", ""))
            if (
                product_code_token is not None
                and product_code_token not in title.upper()
            ):
                continue
            if "COG" not in title.upper():
                continue
            candidates.append(collection)

        if len(candidates) == 1:
            collection_id = str(candidates[0]["id"])
            self._resolved_cdse_collection_id = collection_id
            return collection_id

        candidate_titles = [
            str(item.get("title", item.get("id", "?"))) for item in collections
        ]
        raise CDSENoCoverageError(
            "Could not unambiguously resolve the CDSE STAC collection for "
            f"product_code={self.product_code!r}. Expected title={expected_title!r}. "
            f"Collections returned by free-text search: {candidate_titles}. "
            "Set cdse_collection_id explicitly if the CDSE collection naming changes."
        )

    def _canonical_tile_id(self, tile_code: str, year: str | int) -> str:
        """Build the legacy HRL tile identifier used by the existing cache."""
        if self.product_code is None:
            raise ValueError("product_code is required to construct HRL tile names.")
        return (
            f"CLMS_HRLVLCC_{self.product_code}_S{year}_R10m_"
            f"{tile_code.upper()}_03035_V01_R00"
        )

    def _feature_tile_code(self, feature: dict[str, Any]) -> str | None:
        """Extract an EEA grid code such as E73N22 from a STAC item."""
        strings: list[str] = [str(feature.get("id", ""))]
        properties = feature.get("properties", {})
        if isinstance(properties, dict):
            strings.extend(
                str(value) for value in properties.values() if isinstance(value, str)
            )

        assets = feature.get("assets", {})
        if isinstance(assets, dict):
            for key, asset in assets.items():
                strings.append(str(key))
                if not isinstance(asset, dict):
                    continue
                for field in ("href", "title"):
                    value = asset.get(field)
                    if isinstance(value, str):
                        strings.append(value)
                alternates = asset.get("alternate", {})
                if isinstance(alternates, dict):
                    for alternate in alternates.values():
                        if isinstance(alternate, dict) and isinstance(
                            alternate.get("href"), str
                        ):
                            strings.append(str(alternate["href"]))

        for text in strings:
            match = _CDSE_TILE_CODE_RE.search(text.upper())
            if match is not None:
                return match.group("tile").upper()

        # Projection metadata is a useful fallback if filenames change but each item
        # remains a 100 km EPSG:3035 tile.
        proj_bbox = (
            properties.get("proj:bbox") if isinstance(properties, dict) else None
        )
        proj_code = (
            properties.get("proj:code") if isinstance(properties, dict) else None
        )
        if (
            isinstance(proj_bbox, (list, tuple))
            and len(proj_bbox) >= 4
            and str(proj_code).upper() == _TILE_CRS
        ):
            try:
                min_x = float(proj_bbox[0])
                min_y = float(proj_bbox[1])
                e_index = round(min_x / _EEA_TILE_SIZE_M)
                n_index = round(min_y / _EEA_TILE_SIZE_M)
                return f"E{e_index:02d}N{n_index:02d}"
            except TypeError, ValueError:
                pass

        return None

    def _asset_href_candidates(self, asset: dict[str, Any]) -> list[str]:
        """Return candidate asset hrefs, preferring S3 over authenticated HTTPS."""
        hrefs: list[str] = []
        href = asset.get("href")
        if isinstance(href, str):
            hrefs.append(href)

        alternates = asset.get("alternate", {})
        if isinstance(alternates, dict):
            for alternate in alternates.values():
                if isinstance(alternate, dict):
                    alternate_href = alternate.get("href")
                    if isinstance(alternate_href, str):
                        hrefs.append(alternate_href)

        # Preserve order within each class but strongly prefer object-storage links.
        return sorted(
            dict.fromkeys(hrefs),
            key=lambda value: (
                0 if value.lower().startswith("s3://") else 1,
                0 if "eodata.dataspace.copernicus.eu" in value.lower() else 1,
            ),
        )

    def _select_feature_asset(
        self,
        feature: dict[str, Any],
        tile_id: str,
    ) -> _CDSEAsset | None:
        """Choose the product COG asset from a STAC item."""
        assets = feature.get("assets", {})
        if not isinstance(assets, dict):
            return None

        scored: list[tuple[int, str, dict[str, Any]]] = []
        product_code = (self.product_code or "").upper()
        for key, asset in assets.items():
            if not isinstance(asset, dict):
                continue
            key_upper = str(key).upper()
            title_upper = str(asset.get("title", "")).upper()
            media_type = str(asset.get("type", "")).lower()
            roles = [str(role).lower() for role in asset.get("roles", [])]
            hrefs = self._asset_href_candidates(asset)
            href_text = " ".join(hrefs).lower()

            score = 0
            if product_code and key_upper == product_code:
                score += 200
            if product_code and product_code in title_upper:
                score += 120
            if "image/tiff" in media_type or ".tif" in href_text:
                score += 100
            if "data" in roles:
                score += 30
            if key_upper == "PRODUCT":
                score -= 25
            if "thumbnail" in roles or "THUMBNAIL" in key_upper:
                score -= 500
            if not hrefs:
                score -= 1000

            scored.append((score, str(key), asset))

        if not scored:
            return None

        _, asset_key, asset = max(scored, key=lambda item: item[0])
        hrefs = self._asset_href_candidates(asset)
        if not hrefs:
            return None

        return _CDSEAsset(
            tile_id=tile_id,
            href=hrefs[0],
            asset_key=asset_key,
            media_type=str(asset.get("type"))
            if asset.get("type") is not None
            else None,
            item_id=str(feature.get("id")) if feature.get("id") is not None else None,
        )

    def _stac_features(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
    ) -> list[dict[str, Any]]:
        """Return all CDSE STAC items for the requested year and WGS84 bounds."""
        collection_id = self._resolve_cdse_collection_id()
        search_url = f"{self.cdse_stac_url}/search"
        params: dict[str, Any] | None = {
            "collections": collection_id,
            "bbox": ",".join(str(float(value)) for value in bounds),
            "datetime": (f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z"),
            "limit": 1000,
        }
        method = "GET"
        json_body: dict[str, Any] | None = None
        features: list[dict[str, Any]] = []
        seen_next_urls: set[str] = set()

        while True:
            payload = self._stac_request_json(
                method,
                search_url,
                params=params,
                json_body=json_body,
            )
            page_features = payload.get("features", [])
            if isinstance(page_features, list):
                features.extend(
                    feature for feature in page_features if isinstance(feature, dict)
                )

            next_link = next(
                (
                    link
                    for link in payload.get("links", [])
                    if isinstance(link, dict) and link.get("rel") == "next"
                ),
                None,
            )
            if next_link is None:
                break

            next_url = str(next_link.get("href", ""))
            if not next_url or next_url in seen_next_urls:
                break
            seen_next_urls.add(next_url)
            search_url = next_url
            method = str(next_link.get("method", "GET")).upper()
            params = None
            body = next_link.get("body")
            json_body = body if method == "POST" and isinstance(body, dict) else None

        return features

    def _search_cdse_assets(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
    ) -> tuple[list[str], dict[str, _CDSEAsset]]:
        """Discover canonical HRL tiles and their downloadable CDSE assets."""
        features = self._stac_features(bounds=bounds, year=year)
        asset_lookup: dict[str, _CDSEAsset] = {}
        skipped_without_grid_code = 0
        skipped_without_data_asset = 0

        for feature in features:
            tile_code = self._feature_tile_code(feature)
            if tile_code is None:
                skipped_without_grid_code += 1
                continue
            tile_id = self._canonical_tile_id(tile_code=tile_code, year=year)
            asset = self._select_feature_asset(feature=feature, tile_id=tile_id)
            if asset is None:
                skipped_without_data_asset += 1
                continue

            previous = asset_lookup.get(tile_id)
            if previous is None:
                asset_lookup[tile_id] = asset
                continue

            # Prefer an asset whose source name explicitly contains the legacy
            # V01_R00 revision used by the historical HRL files.
            previous_text = f"{previous.item_id or ''} {previous.href}".upper()
            new_text = f"{asset.item_id or ''} {asset.href}".upper()
            if "V01_R00" in new_text and "V01_R00" not in previous_text:
                asset_lookup[tile_id] = asset

        if skipped_without_grid_code:
            self.logger.debug(
                "Skipped %s CDSE STAC item(s) because no EEA tile code could be "
                "identified.",
                skipped_without_grid_code,
            )
        if skipped_without_data_asset:
            self.logger.debug(
                "Skipped %s CDSE STAC item(s) because no usable data asset was found.",
                skipped_without_data_asset,
            )

        tile_ids = sorted(asset_lookup)
        if not tile_ids:
            raise CDSENoCoverageError(
                f"CDSE STAC returned no usable {self.product_code or 'HRL'} COG tiles "
                f"for year={year}, bounds={bounds}."
            )

        return tile_ids, asset_lookup

    def _s3_location(self, href: str) -> tuple[str, str] | None:
        """Convert an S3 or CDSE object-storage URL to ``(bucket, key)``."""
        from urllib.parse import urlparse

        parsed = urlparse(href)
        if parsed.scheme.lower() == "s3":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            return (bucket, key) if bucket and key else None

        endpoint_host = urlparse(self.cdse_s3_endpoint).netloc.lower()
        if (
            parsed.scheme.lower() in {"http", "https"}
            and parsed.netloc.lower() == endpoint_host
        ):
            parts = parsed.path.lstrip("/").split("/", 1)
            if len(parts) == 2 and all(parts):
                return parts[0], parts[1]

        if href.startswith("/eodata/") or href.startswith("eodata/"):
            normalized = href.lstrip("/")
            bucket, key = normalized.split("/", 1)
            return bucket, key

        return None

    def _get_cdse_s3_client(self) -> Any:
        """Create a boto3 S3 client for the CDSE ``eodata`` object store."""
        if self._cdse_s3_client is not None:
            return self._cdse_s3_client

        try:
            import boto3
        except ImportError as error:
            raise ImportError(
                "Downloading missing CDSE HRL tiles through S3 requires 'boto3'. "
                "Install boto3 or provide the tiles in the local cache."
            ) from error

        access_key = self.cdse_s3_access_key or os.getenv("CDSE_S3_ACCESS_KEY")
        secret_key = self.cdse_s3_secret_key or os.getenv("CDSE_S3_SECRET_KEY")
        session_token = self.cdse_s3_session_token or os.getenv("CDSE_S3_SESSION_TOKEN")

        client_kwargs: dict[str, Any] = {
            "endpoint_url": self.cdse_s3_endpoint,
            "region_name": "default",
        }
        if access_key is not None:
            client_kwargs["aws_access_key_id"] = access_key
        if secret_key is not None:
            client_kwargs["aws_secret_access_key"] = secret_key
        if session_token is not None:
            client_kwargs["aws_session_token"] = session_token

        # If explicit CDSE_* variables are absent, boto3 can still use its standard
        # AWS environment/config credential chain (AWS_ACCESS_KEY_ID, profiles, etc.).
        self._cdse_s3_client = boto3.client("s3", **client_kwargs)
        return self._cdse_s3_client

    def _download_cdse_href(self, href: str, target: Path) -> None:
        """Download one STAC asset to a temporary local path."""
        s3_location = self._s3_location(href)
        if s3_location is not None:
            bucket, key = s3_location
            client = self._get_cdse_s3_client()
            client.download_file(bucket, key, str(target))
            return

        if href.lower().startswith(("http://", "https://")):
            requests = self._requests()
            headers: dict[str, str] = {}
            token = self.cdse_access_token or os.getenv("CDSE_ACCESS_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            with requests.get(
                href,
                headers=headers,
                stream=True,
                timeout=self.cdse_request_timeout_seconds,
            ) as response:
                response.raise_for_status()
                with target.open("wb") as file_obj:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            file_obj.write(chunk)
            return

        raise CDSEDownloadError(f"Unsupported CDSE asset URL: {href}")

    def _download_single_cdse_asset(
        self,
        asset: _CDSEAsset,
        year: str | int,
    ) -> _TileDownloadStatus:
        """Download one CDSE asset atomically under the legacy local filename."""
        tif_path = self._tile_tif_path(year, asset.tile_id)
        zip_path = self._tile_zip_path(year, asset.tile_id)
        if tif_path.exists():
            return _TileDownloadStatus(asset.tile_id, "cached_tif", tif_path)
        if zip_path.exists():
            return _TileDownloadStatus(asset.tile_id, "cached_zip", zip_path)

        href_path = asset.href.split("?", 1)[0].lower()
        is_zip = href_path.endswith(".zip") or (
            asset.media_type is not None and "zip" in asset.media_type.lower()
        )
        target_path = zip_path if is_zip else tif_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        attempts = self.download_retries + 1
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            temp_path = target_path.with_name(
                f".{target_path.name}.download-{uuid.uuid4().hex}"
            )
            try:
                self._download_cdse_href(asset.href, temp_path)
                if not temp_path.exists() or temp_path.stat().st_size == 0:
                    raise CDSEDownloadError(
                        f"CDSE download created an empty file for {asset.tile_id}: "
                        f"{asset.href}"
                    )
                temp_path.replace(target_path)
                return _TileDownloadStatus(
                    asset.tile_id,
                    "downloaded_zip" if is_zip else "downloaded_tif",
                    target_path,
                )
            except Exception as error:
                last_error = error
                temp_path.unlink(missing_ok=True)
                if attempt < attempts:
                    delay = self.download_backoff_seconds * attempt
                    self.logger.warning(
                        "CDSE download failed for %s (attempt %s/%s): %s. "
                        "Retrying after %.1f s.",
                        asset.tile_id,
                        attempt,
                        attempts,
                        error,
                        delay,
                    )
                    time.sleep(delay)
            finally:
                temp_path.unlink(missing_ok=True)

        assert last_error is not None
        raise CDSEDownloadError(
            f"Failed to download CDSE tile {asset.tile_id!r} for year {year} "
            f"from {asset.href}: {last_error}"
        ) from last_error

    def _download_cdse_tiles(
        self,
        tile_ids: list[str],
        year: str | int,
        asset_lookup: dict[str, _CDSEAsset],
    ) -> None:
        """Download a set of missing CDSE tiles using the existing worker setting."""
        if not tile_ids:
            return

        unavailable = [tile_id for tile_id in tile_ids if tile_id not in asset_lookup]
        if unavailable:
            raise CDSEDownloadError(
                "CDSE STAC did not provide assets for locally expected tile(s): "
                + ", ".join(unavailable)
            )

        worker_count = min(self.max_parallel_downloads, max(1, len(tile_ids)))
        failures: list[tuple[str, BaseException]] = []

        if worker_count == 1:
            for tile_id in tile_ids:
                try:
                    status = self._download_single_cdse_asset(
                        asset=asset_lookup[tile_id],
                        year=year,
                    )
                    self.logger.debug(
                        "CDSE tile %s ready for year %s: %s",
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
                        self._download_single_cdse_asset,
                        asset_lookup[tile_id],
                        year,
                    ): tile_id
                    for tile_id in tile_ids
                }
                for future in as_completed(futures):
                    tile_id = futures[future]
                    try:
                        status = future.result()
                    except Exception as error:
                        failures.append((tile_id, error))
                        self.logger.error(
                            "Failed to download CDSE tile %s for year %s: %s",
                            tile_id,
                            year,
                            error,
                        )
                    else:
                        self.logger.debug(
                            "CDSE tile %s ready for year %s: %s",
                            status.tile_id,
                            year,
                            status.path,
                        )

        if failures:
            details = "\n".join(f"- {tile_id}: {error}" for tile_id, error in failures)
            raise CDSEDownloadError(
                f"Failed to download {len(failures)} of {len(tile_ids)} CDSE tile(s) "
                f"for year {year}.\n{details}"
            )

    def _wekeo_fallback_for_missing(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
        required_tile_ids: list[str],
        query_overrides: dict[str, Any] | None,
    ) -> list[str]:
        """Use the parent WEkEO implementation to fill still-missing tiles."""
        wekeo_tile_ids, result_lookup = super()._search_tiles(
            bounds=bounds,
            year=year,
            query_overrides=query_overrides,
        )
        if not required_tile_ids:
            required_tile_ids = list(wekeo_tile_ids)
        else:
            required_tile_ids = sorted(set(required_tile_ids) | set(wekeo_tile_ids))

        cache_status = self._inspect_tile_cache(required_tile_ids, year)
        downloadable = [
            tile_id
            for tile_id in cache_status.missing_tile_ids
            if tile_id in result_lookup
        ]
        if downloadable:
            super().download_tiles(
                tile_ids=downloadable,
                year=year,
                bounds=bounds,
                query_overrides=query_overrides,
                result_lookup=result_lookup,
            )

        return required_tile_ids

    def fetch(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
        url: str | None = None,
        query_overrides: dict[str, Any] | None = None,
    ) -> CopernicusDataSpace:
        """Ensure the requested HRL tiles are cached, local first then CDSE.

        A complete local year returns without any network access. If the local
        filename catalogue identifies missing files, CDSE STAC is queried and the
        missing COGs are stored with the same legacy filename stem used by WEkEO.
        """
        required_tile_ids: list[str] = []
        cdse_assets: dict[str, _CDSEAsset] = {}
        cdse_error: BaseException | None = None

        if self.prefer_local_tiles:
            required_tile_ids = self._discover_local_tiles_for_bounds(
                bounds=bounds,
                year=year,
            )
            if required_tile_ids:
                cache_status = self._inspect_tile_cache(required_tile_ids, year)
                if cache_status.is_complete:
                    self.logger.info(
                        "Using %s locally cached %s tile(s) for year %s; no CDSE or "
                        "WEkEO catalogue request required.",
                        cache_status.total_tiles,
                        self.product_code or "HRL",
                        year,
                    )
                    self.tile_ids = required_tile_ids
                    self.year = year
                    return self

                self.logger.info(
                    "Local %s catalogue for year %s identified %s required tile(s); "
                    "%s are missing. Querying CDSE STAC only for authoritative "
                    "discovery/download.",
                    self.product_code or "HRL",
                    year,
                    cache_status.total_tiles,
                    cache_status.missing_tiles,
                )

        try:
            cdse_tile_ids, cdse_assets = self._search_cdse_assets(
                bounds=bounds,
                year=year,
            )
            required_tile_ids = sorted(set(required_tile_ids) | set(cdse_tile_ids))
            cache_status = self._inspect_tile_cache(required_tile_ids, year)
            if not cache_status.is_complete:
                self.logger.info(
                    "Downloading %s missing %s tile(s) for year %s from CDSE S3.",
                    cache_status.missing_tiles,
                    self.product_code or "HRL",
                    year,
                )
                self._download_cdse_tiles(
                    tile_ids=list(cache_status.missing_tile_ids),
                    year=year,
                    asset_lookup=cdse_assets,
                )
        except Exception as error:
            cdse_error = error
            self.logger.warning(
                "CDSE could not fully satisfy %s year %s: %s",
                self.product_code or "HRL",
                year,
                error,
            )

        if required_tile_ids:
            cache_status = self._inspect_tile_cache(required_tile_ids, year)
        else:
            cache_status = _TileCacheStatus((), (), (), ())

        if not required_tile_ids or not cache_status.is_complete:
            if self.allow_wekeo_fallback:
                try:
                    self.logger.info(
                        "Trying legacy WEkEO fallback for %s year %s.",
                        self.product_code or "HRL",
                        year,
                    )
                    required_tile_ids = self._wekeo_fallback_for_missing(
                        bounds=bounds,
                        year=year,
                        required_tile_ids=required_tile_ids,
                        query_overrides=query_overrides,
                    )
                except Exception as wekeo_error:
                    if cdse_error is not None:
                        raise CDSEDownloadError(
                            "Neither CDSE nor WEkEO could satisfy the HRL tile request. "
                            f"CDSE error: {cdse_error}. WEkEO error: {wekeo_error}."
                        ) from wekeo_error
                    raise
            elif cdse_error is not None:
                raise CDSEDownloadError(
                    "CDSE could not satisfy the HRL tile request and WEkEO fallback "
                    f"is disabled: {cdse_error}"
                ) from cdse_error

        if not required_tile_ids:
            raise CDSENoCoverageError(
                f"No {self.product_code or 'HRL'} tiles were identified for "
                f"year={year}, bounds={bounds}."
            )

        final_status = self._inspect_tile_cache(required_tile_ids, year)
        if not final_status.is_complete:
            missing = [
                self._tile_tif_name(tile_id)
                for tile_id in final_status.missing_tile_ids
            ]
            raise FileNotFoundError(
                f"{final_status.missing_tiles} required {self.product_code or 'HRL'} "
                f"tile(s) remain unavailable after local/CDSE/WEkEO acquisition: "
                f"{missing}. Cache directory: {self._year_dir(year)}"
            )

        self.tile_ids = required_tile_ids
        self.year = year
        return self

    def get_tiles_for_mask(
        self,
        mask: BaseGeometry,
        year: str | int,
        query_overrides: dict[str, Any] | None = None,
    ) -> list[str]:
        """Identify required tile IDs without forcing a remote catalogue query."""
        if self.prefer_local_tiles:
            local_tile_ids = self._discover_local_tiles_for_bounds(
                bounds=mask.bounds,
                year=year,
            )
            if local_tile_ids:
                cache_status = self._inspect_tile_cache(local_tile_ids, year)
                if cache_status.is_complete:
                    return local_tile_ids

        try:
            cdse_tile_ids, _ = self._search_cdse_assets(
                bounds=mask.bounds,
                year=year,
            )
            if self.prefer_local_tiles and local_tile_ids:
                return sorted(set(local_tile_ids) | set(cdse_tile_ids))
            return cdse_tile_ids
        except Exception:
            if self.allow_wekeo_fallback:
                return super().get_tiles_for_mask(
                    mask=mask,
                    year=year,
                    query_overrides=query_overrides,
                )
            raise
