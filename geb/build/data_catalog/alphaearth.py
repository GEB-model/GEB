"""Data adapter for downloading AlphaEarth annual satellite embeddings."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
import geopandas as gpd
import numpy as np
from aiohttp_retry import ExponentialRetry, RetryClient
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .base import Adapter


DEFAULT_BASE_URL = (
    "https://storage.googleapis.com/alphaearth_foundations/"
    "satellite_embedding/v1/annual"
)
INDEX_FILENAME = "aef_index.parquet"
AVAILABLE_YEARS = tuple(range(2017, 2026))
NODATA_VALUE = -128
DOWNLOAD_CHUNK_SIZE_BYTES = 8 * 1024 * 1024


class AlphaEarth(Adapter):
    """Select and download AlphaEarth embedding Cloud-Optimized GeoTIFFs.

    The adapter stores both the official spatial index and downloaded COGs
    below :attr:`Adapter.root`. Consequently, its storage location follows the
    normal GEB ``folder``/``local_version``/``cache`` mechanism and
    ``GEB_DATA_ROOT`` rather than a cluster-specific absolute path.

    Notes:
        Downloaded COGs contain 64 signed-int8 bands. Convert their raw values
        to analysis-ready floating-point embeddings with :meth:`dequantize`.
    """

    def __init__(
        self,
        *args: Any,
        max_parallel_downloads: int = 2,
        verbose_file_logging: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the AlphaEarth adapter.

        Args:
            *args: Positional arguments passed to :class:`Adapter`.
            max_parallel_downloads: Maximum number of simultaneous COG
                downloads. Individual COGs are large, so the default is low.
            verbose_file_logging: Print one line for every cached, downloading,
                and saved COG. The source-reference workflow disables this and
                emits aggregate cache/download summaries instead.
            **kwargs: Keyword arguments passed to :class:`Adapter`, including
                ``folder``, ``local_version``, ``filename``, and ``cache``.

        Raises:
            ValueError: If ``max_parallel_downloads`` is smaller than one.
        """
        super().__init__(*args, **kwargs)

        if max_parallel_downloads < 1:
            raise ValueError("max_parallel_downloads must be at least 1.")

        dataset_root = Path(self.root)
        self.index_dir = dataset_root / "index"
        self.download_root = dataset_root / "annual"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.download_root.mkdir(parents=True, exist_ok=True)

        self.max_parallel_downloads = max_parallel_downloads
        self.verbose_file_logging = bool(verbose_file_logging)
        self.url = DEFAULT_BASE_URL
        self.index_url = f"{self.url}/{INDEX_FILENAME}"
        self._path_column: str | None = None

    def fetch(self, url: str | None) -> AlphaEarth:
        """Set the AlphaEarth annual-data base URL.

        Args:
            url: Optional alternative base URL. ``None`` selects the official
                Google Cloud Storage HTTPS endpoint.

        Returns:
            The current adapter instance.
        """
        self.url = (url or DEFAULT_BASE_URL).rstrip("/")
        self.index_url = f"{self.url}/{INDEX_FILENAME}"
        return self

    @property
    def index_path(self) -> Path:
        """Return the catalog-managed path of the cached GeoParquet index."""
        return self.index_dir / INDEX_FILENAME

    @staticmethod
    def dequantize(values: np.ndarray) -> np.ndarray:
        """Convert raw AlphaEarth int8 values to float32 embeddings.

        Args:
            values: Raw AlphaEarth values. The input shape is preserved.

        Returns:
            Dequantized float32 values, with raw ``-128`` nodata values
            converted to ``NaN``.
        """
        raw = np.asarray(values)
        nodata = raw == NODATA_VALUE
        raw_float = raw.astype(np.float32)
        result = ((raw_float / 127.5) ** 2) * np.sign(raw_float)
        result[nodata] = np.nan
        return result

    async def _download_file(
        self,
        client: RetryClient,
        remote_url: str,
        destination: Path,
        overwrite: bool,
        semaphore: asyncio.Semaphore,
    ) -> Path:
        """Download one remote file atomically."""
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and destination.stat().st_size > 0 and not overwrite:
            if self.verbose_file_logging:
                print(f"Using cached AlphaEarth file: {destination}")
            return destination

        temporary_path = destination.with_suffix(destination.suffix + ".part")
        temporary_path.unlink(missing_ok=True)

        async with semaphore:
            if self.verbose_file_logging:
                print(f"Downloading {remote_url}")
            async with client.get(remote_url, raise_for_status=True) as response:
                with temporary_path.open("wb") as file:
                    async for chunk in response.content.iter_chunked(
                        DOWNLOAD_CHUNK_SIZE_BYTES
                    ):
                        file.write(chunk)

        temporary_path.replace(destination)
        if self.verbose_file_logging:
            print(f"Saved AlphaEarth file: {destination}")
        return destination

    async def _ensure_index(self, refresh: bool = False) -> Path:
        """Download the official AlphaEarth index when it is not cached."""
        if (
            self.index_path.exists()
            and self.index_path.stat().st_size > 0
            and not refresh
        ):
            print(f"Using cached AlphaEarth index: {self.index_path}")
            return self.index_path

        retry_options = ExponentialRetry(
            attempts=8,
            start_timeout=5,
            max_timeout=120,
            factor=2,
            retry_all_server_errors=True,
        )
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=60,
            sock_read=600,
        )

        async with RetryClient(
            retry_options=retry_options,
            timeout=timeout,
        ) as client:
            semaphore = asyncio.Semaphore(1)
            return await self._download_file(
                client=client,
                remote_url=self.index_url,
                destination=self.index_path,
                overwrite=refresh,
                semaphore=semaphore,
            )

    def _detect_path_column(self, index: gpd.GeoDataFrame) -> str:
        """Identify the index column containing each COG path."""
        if self._path_column is not None and self._path_column in index.columns:
            return self._path_column

        preferred_columns = (
            "path",
            "gcs_path",
            "gcs_url",
            "url",
            "uri",
            "href",
            "filename",
            "file",
        )

        for column in preferred_columns:
            if column in index.columns:
                self._path_column = column
                return column

        for column in index.columns:
            if column == index.geometry.name:
                continue

            values = index[column].dropna().head(50)
            if values.empty:
                continue

            if any(str(value).lower().endswith((".tif", ".tiff")) for value in values):
                self._path_column = column
                return column

        raise KeyError(
            "Could not identify the COG path column in the AlphaEarth index. "
            f"Available columns: {list(index.columns)}"
        )

    def _to_https_url(
        self,
        path_value: str,
        year: int,
        utm_zone: str,
    ) -> str:
        """Convert an index path value to a downloadable HTTPS URL."""
        value = path_value.strip()

        if value.startswith(("https://", "http://")):
            return value

        if value.startswith("gs://"):
            parsed = urlparse(value)
            return (
                f"https://storage.googleapis.com/{parsed.netloc}/"
                f"{parsed.path.lstrip('/')}"
            )

        relative = value.lstrip("/")

        if relative.startswith("alphaearth_foundations/"):
            return f"https://storage.googleapis.com/{relative}"

        if relative.startswith("satellite_embedding/"):
            return f"https://storage.googleapis.com/alphaearth_foundations/{relative}"

        if relative.startswith(f"{year}/") or "/" in relative:
            return f"{self.url}/{relative}"

        return f"{self.url}/{year}/{utm_zone}/{relative}"

    @staticmethod
    def _normalize_years(years: int | Sequence[int]) -> tuple[int, ...]:
        """Normalize and validate requested years."""
        requested = (years,) if isinstance(years, int) else tuple(years)

        if not requested:
            raise ValueError("At least one AlphaEarth year must be requested.")

        normalized = tuple(int(year) for year in requested)
        invalid = sorted(set(normalized).difference(AVAILABLE_YEARS))
        if invalid:
            raise ValueError(
                f"Unsupported AlphaEarth year(s): {invalid}. "
                f"Available years are {AVAILABLE_YEARS[0]}-{AVAILABLE_YEARS[-1]}."
            )

        return tuple(sorted(set(normalized)))

    @staticmethod
    def _validate_bounds(
        bounds: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Validate a WGS84 bounding box."""
        if len(bounds) != 4:
            raise ValueError("bounds must be (min_lon, min_lat, max_lon, max_lat).")

        min_lon, min_lat, max_lon, max_lat = map(float, bounds)

        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("bounds must have increasing longitude and latitude.")

        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("Longitude bounds must be between -180 and 180.")

        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError("Latitude bounds must be between -90 and 90.")

        return min_lon, min_lat, max_lon, max_lat

    def select_files(
        self,
        index: gpd.GeoDataFrame,
        years: int | Sequence[int],
        bounds: tuple[float, float, float, float],
        buffer_degrees: float = 0.0,
    ) -> gpd.GeoDataFrame:
        """Select COGs intersecting requested years and WGS84 bounds."""
        requested_years = self._normalize_years(years)
        min_lon, min_lat, max_lon, max_lat = self._validate_bounds(bounds)

        if buffer_degrees < 0:
            raise ValueError("buffer_degrees cannot be negative.")

        query_geometry = box(
            min_lon - buffer_degrees,
            min_lat - buffer_degrees,
            max_lon + buffer_degrees,
            max_lat + buffer_degrees,
        )

        if index.crs is None:
            index = index.set_crs(4326)
        elif index.crs.to_epsg() != 4326:
            index = index.to_crs(4326)

        if "year" not in index.columns:
            raise KeyError("AlphaEarth index does not contain a 'year' column.")

        selected = index.loc[index["year"].astype(int).isin(requested_years)].copy()

        bbox_columns = {
            "wgs84_west",
            "wgs84_south",
            "wgs84_east",
            "wgs84_north",
        }
        if bbox_columns.issubset(selected.columns):
            selected = selected.loc[
                (selected["wgs84_east"] >= query_geometry.bounds[0])
                & (selected["wgs84_west"] <= query_geometry.bounds[2])
                & (selected["wgs84_north"] >= query_geometry.bounds[1])
                & (selected["wgs84_south"] <= query_geometry.bounds[3])
            ].copy()

        selected = selected.loc[selected.geometry.intersects(query_geometry)].copy()

        if selected.empty:
            raise FileNotFoundError(
                "No AlphaEarth COGs intersect the requested years and bounds."
            )

        path_column = self._detect_path_column(selected)
        selected["remote_url"] = selected.apply(
            lambda row: self._to_https_url(
                path_value=str(row[path_column]),
                year=int(row["year"]),
                utm_zone=str(row.get("utm_zone", "unknown")),
            ),
            axis=1,
        )

        sort_columns = [
            column
            for column in ("year", "utm_zone", "remote_url")
            if column in selected.columns
        ]
        return selected.sort_values(sort_columns, ignore_index=True)

    @staticmethod
    def _normalize_query_geometry(
        geometry: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame,
    ) -> BaseGeometry:
        """Return one valid WGS84 geometry for exact tile selection.

        Bare Shapely geometries are interpreted as WGS84. GeoSeries and
        GeoDataFrames are reprojected to EPSG:4326 before their geometries are
        combined.
        """
        if isinstance(geometry, gpd.GeoDataFrame):
            if geometry.crs is None:
                raise ValueError("Query GeoDataFrame must define a CRS.")
            result = unary_union(geometry.to_crs(4326).geometry.dropna().tolist())
        elif isinstance(geometry, gpd.GeoSeries):
            if geometry.crs is None:
                raise ValueError("Query GeoSeries must define a CRS.")
            result = unary_union(geometry.to_crs(4326).dropna().tolist())
        elif isinstance(geometry, BaseGeometry):
            result = geometry
        else:
            raise TypeError(
                "geometry must be a Shapely geometry, GeoSeries or GeoDataFrame."
            )

        if result.is_empty:
            raise ValueError("AlphaEarth query geometry is empty.")
        if not result.is_valid:
            result = result.buffer(0)
        if result.is_empty:
            raise ValueError("AlphaEarth query geometry could not be repaired.")
        return result

    def select_files_for_geometry(
        self,
        index: gpd.GeoDataFrame,
        years: int | Sequence[int],
        geometry: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame,
        buffer_degrees: float = 0.0,
    ) -> gpd.GeoDataFrame:
        """Select unique COGs intersecting an exact WGS84 geometry.

        A bounds query is used only as a fast prefilter. Polygon queries then
        require positive-area overlap, preventing boundary-only tiles from being
        selected. Point and line queries retain every intersecting tile.
        """
        query_geometry = self._normalize_query_geometry(geometry)
        if buffer_degrees < 0:
            raise ValueError("buffer_degrees cannot be negative.")
        if buffer_degrees:
            query_geometry = query_geometry.buffer(float(buffer_degrees))

        selected = self.select_files(
            index=index,
            years=years,
            bounds=tuple(float(value) for value in query_geometry.bounds),
            buffer_degrees=0.0,
        )
        if selected.crs is None:
            selected = selected.set_crs(4326)
        elif selected.crs.to_epsg() != 4326:
            selected = selected.to_crs(4326)

        intersects = selected.geometry.intersects(query_geometry)
        if query_geometry.area > 0:
            intersects &= ~selected.geometry.touches(query_geometry)
        selected = selected.loc[intersects].copy()
        if selected.empty:
            raise FileNotFoundError(
                "No AlphaEarth COGs intersect the exact requested geometry."
            )

        path_column = self._detect_path_column(selected)
        selected["remote_url"] = selected.apply(
            lambda row: self._to_https_url(
                path_value=str(row[path_column]),
                year=int(row["year"]),
                utm_zone=str(row.get("utm_zone", "unknown")),
            ),
            axis=1,
        )
        sort_columns = [
            column
            for column in ("year", "utm_zone", "remote_url")
            if column in selected.columns
        ]
        return selected.drop_duplicates("remote_url").sort_values(
            sort_columns,
            ignore_index=True,
        )

    async def read_geometry_async(
        self,
        years: int | Sequence[int],
        geometry: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame,
        download_dir: str | Path | None = None,
        *,
        buffer_degrees: float = 0.0,
        dry_run: bool = False,
        overwrite: bool = False,
        refresh_index: bool = False,
        max_files: int | None = 50,
    ) -> gpd.GeoDataFrame:
        """Select and optionally download COGs for exact reference geometries."""
        index_path = await self._ensure_index(refresh=refresh_index)
        index = gpd.read_parquet(index_path)
        selected = self.select_files_for_geometry(
            index=index,
            years=years,
            geometry=geometry,
            buffer_degrees=buffer_degrees,
        )

        if not dry_run and max_files is not None and len(selected) > max_files:
            raise RuntimeError(
                f"Selection contains {len(selected)} COGs, exceeding "
                f"max_files={max_files}. Run with dry_run=True, narrow the "
                "geometry, or explicitly increase max_files."
            )

        root = self.download_root if download_dir is None else Path(download_dir)
        root.mkdir(parents=True, exist_ok=True)
        selected["local_path"] = selected.apply(
            lambda row: str(
                root
                / str(int(row["year"]))
                / str(row.get("utm_zone", "unknown"))
                / Path(urlparse(str(row["remote_url"])).path).name
            ),
            axis=1,
        )

        print(
            f"Selected {len(selected)} unique AlphaEarth COG(s) for exact "
            f"geometry and year(s) "
            f"{sorted(selected['year'].astype(int).unique().tolist())}."
        )
        if dry_run:
            return selected

        retry_options = ExponentialRetry(
            attempts=8,
            start_timeout=5,
            max_timeout=120,
            factor=2,
            retry_all_server_errors=True,
        )
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=60,
            sock_read=1800,
        )
        semaphore = asyncio.Semaphore(self.max_parallel_downloads)
        async with RetryClient(
            retry_options=retry_options,
            timeout=timeout,
        ) as client:
            await asyncio.gather(
                *(
                    self._download_file(
                        client=client,
                        remote_url=str(row.remote_url),
                        destination=Path(row.local_path),
                        overwrite=overwrite,
                        semaphore=semaphore,
                    )
                    for row in selected.itertuples(index=False)
                )
            )
        return selected

    def read_geometry(
        self,
        years: int | Sequence[int],
        geometry: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame,
        download_dir: str | Path | None = None,
        *,
        buffer_degrees: float = 0.0,
        dry_run: bool = False,
        overwrite: bool = False,
        refresh_index: bool = False,
        max_files: int | None = 50,
    ) -> gpd.GeoDataFrame:
        """Synchronously retrieve COGs intersecting exact geometries."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.read_geometry_async(
                    years=years,
                    geometry=geometry,
                    download_dir=download_dir,
                    buffer_degrees=buffer_degrees,
                    dry_run=dry_run,
                    overwrite=overwrite,
                    refresh_index=refresh_index,
                    max_files=max_files,
                )
            )
        raise RuntimeError(
            "AlphaEarth.read_geometry() cannot run inside an active asyncio "
            "event loop. Use 'await adapter.read_geometry_async(...)' instead."
        )

    async def read_async(
        self,
        years: int | Sequence[int],
        bounds: tuple[float, float, float, float],
        download_dir: str | Path | None = None,
        *,
        buffer_degrees: float = 0.0,
        dry_run: bool = False,
        overwrite: bool = False,
        refresh_index: bool = False,
        max_files: int | None = 50,
    ) -> gpd.GeoDataFrame:
        """Select and optionally download AlphaEarth COGs.

        Args:
            years: One year or multiple years in the 2017-2025 range.
            bounds: WGS84 bounds as ``(min_lon, min_lat, max_lon, max_lat)``.
            download_dir: Optional override for downloaded COGs. When omitted,
                files are stored below ``self.root / 'annual'``.
            buffer_degrees: Optional WGS84 buffer around the requested bounds.
            dry_run: Select files and return planned paths without downloading.
            overwrite: Redownload existing files when ``True``.
            refresh_index: Redownload the spatial index when ``True``.
            max_files: Safety limit for one call. Set to ``None`` to disable.

        Returns:
            Selected COG metadata with ``remote_url`` and ``local_path``.
        """
        index_path = await self._ensure_index(refresh=refresh_index)
        index = gpd.read_parquet(index_path)
        selected = self.select_files(
            index=index,
            years=years,
            bounds=bounds,
            buffer_degrees=buffer_degrees,
        )

        if not dry_run and max_files is not None and len(selected) > max_files:
            raise RuntimeError(
                f"Selection contains {len(selected)} COGs, exceeding "
                f"max_files={max_files}. Run with dry_run=True to inspect the "
                "selection, narrow the bounds, or explicitly increase max_files."
            )

        root = self.download_root if download_dir is None else Path(download_dir)
        root.mkdir(parents=True, exist_ok=True)

        selected["local_path"] = selected.apply(
            lambda row: str(
                root
                / str(int(row["year"]))
                / str(row.get("utm_zone", "unknown"))
                / Path(urlparse(str(row["remote_url"])).path).name
            ),
            axis=1,
        )

        print(
            f"Selected {len(selected)} AlphaEarth COG(s) for "
            f"year(s) {sorted(selected['year'].astype(int).unique().tolist())}."
        )

        if dry_run:
            return selected

        retry_options = ExponentialRetry(
            attempts=8,
            start_timeout=5,
            max_timeout=120,
            factor=2,
            retry_all_server_errors=True,
        )
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=60,
            sock_read=1800,
        )
        semaphore = asyncio.Semaphore(self.max_parallel_downloads)

        async with RetryClient(
            retry_options=retry_options,
            timeout=timeout,
        ) as client:
            tasks = [
                self._download_file(
                    client=client,
                    remote_url=str(row.remote_url),
                    destination=Path(row.local_path),
                    overwrite=overwrite,
                    semaphore=semaphore,
                )
                for row in selected.itertuples(index=False)
            ]
            await asyncio.gather(*tasks)

        return selected

    def read(
        self,
        years: int | Sequence[int],
        bounds: tuple[float, float, float, float],
        download_dir: str | Path | None = None,
        *,
        buffer_degrees: float = 0.0,
        dry_run: bool = False,
        overwrite: bool = False,
        refresh_index: bool = False,
        max_files: int | None = 50,
    ) -> gpd.GeoDataFrame:
        """Synchronously select and optionally download AlphaEarth COGs.

        Args:
            years: One year or multiple years in the 2017-2025 range.
            bounds: WGS84 bounds as ``(min_lon, min_lat, max_lon, max_lat)``.
            download_dir: Optional destination override. Defaults to the
                catalog-managed ``self.root / 'annual'`` directory.
            buffer_degrees: Optional WGS84 buffer around the requested bounds.
            dry_run: Select files and return planned paths without downloading.
            overwrite: Redownload existing files when ``True``.
            refresh_index: Redownload the spatial index when ``True``.
            max_files: Safety limit for one call. Set to ``None`` to disable.

        Returns:
            Selected COG metadata with local paths.

        Raises:
            RuntimeError: If called inside an already running event loop. Use
                ``await adapter.read_async(...)`` in that situation.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.read_async(
                    years=years,
                    bounds=bounds,
                    download_dir=download_dir,
                    buffer_degrees=buffer_degrees,
                    dry_run=dry_run,
                    overwrite=overwrite,
                    refresh_index=refresh_index,
                    max_files=max_files,
                )
            )

        raise RuntimeError(
            "AlphaEarth.read() cannot run inside an active asyncio event loop. "
            "Use 'await adapter.read_async(...)' instead."
        )
