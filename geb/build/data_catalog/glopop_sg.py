"""Data adapter for GLOPOP-SG population data."""

from __future__ import annotations

import gzip
import os
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from tqdm import tqdm

from geb.workflows.io import HTTP429Error, RemoteFile

from .base import Adapter


class GLOPOP_SG(Adapter):
    """Adapter for GLOPOP-SG population data.

    Region files are cached persistently when this adapter has a configured
    ``folder``, ``local_version`` and ``cache``. With the normal data-catalog
    configuration ``cache="global"``, the files are therefore shared across
    model builds through ``GEB_DATA_ROOT`` (or ``~/.geb_cache`` when that
    environment variable is not set).
    """

    # Central-directory metadata for each URL, shared across all instances so
    # the same remote ZIP is only interrogated once per process.
    _zip_info_cache: dict[str, dict[str, zipfile.ZipInfo]] = {}
    _RETRY_429_SLEEP_S: int = 10
    _MAX_RETRY_S: int = 6 * 3600  # give up after 6 hours of 429 responses
    _CACHE_SUBDIRECTORY = "region_files"
    _DOWNLOAD_CHUNK_SIZE = 1024 * 1024

    def fetch(self, url: str) -> GLOPOP_SG:
        """Fetch data for a specific region.

        Args:
            url: URL of the remote GLOPOP-SG ZIP archive.

        Returns:
            The GLOPOP_SG instance.
        """
        self.url = url
        return self

    def _get_zip_infolist(self, url: str) -> dict[str, zipfile.ZipInfo]:
        """Return the central-directory info for a remote ZIP, with caching.

        The central directory is fetched at most once per URL per process; all
        subsequent calls return the in-memory cache without any HTTP requests.
        Retries on HTTP 429 responses for up to ``_MAX_RETRY_S`` seconds.

        Args:
            url: URL of the remote ZIP archive.

        Returns:
            Mapping of filename to ZipInfo for every entry in the archive.

        Raises:
            TimeoutError: If HTTP 429 retries exceed the configured time limit.
        """
        if url not in self._zip_info_cache:
            retry_start: float = time.monotonic()
            while True:
                try:
                    with zipfile.ZipFile(RemoteFile(url), "r") as zf:
                        self._zip_info_cache[url] = {
                            info.filename: info for info in zf.infolist()
                        }
                    break
                except HTTP429Error:
                    if time.monotonic() - retry_start > self._MAX_RETRY_S:
                        raise TimeoutError(
                            "HTTP 429 retries exceeded the "
                            f"{self._MAX_RETRY_S / 3600:.0f}-hour limit for {url}."
                        )
                    time.sleep(self._RETRY_429_SLEEP_S)
        return self._zip_info_cache[url]

    def _cache_root(self) -> Path | None:
        """Return the persistent per-dataset cache directory when configured."""
        if self.folder is None or self.local_version is None or self.cache is None:
            return None
        cache_root = self.root / self._CACHE_SUBDIRECTORY
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root

    @staticmethod
    def _usable_cached_file(path: Path) -> bool:
        """Return True for an existing non-empty cached file."""
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def _find_cached_file(self, filename: str) -> Path | None:
        """Find a persistent cached member without contacting the remote server.

        The preferred location is ``<adapter root>/region_files``. A file placed
        directly in the adapter root is also recognized so manually staged or
        legacy files can be reused without being moved first.
        """
        cache_root = self._cache_root()
        if cache_root is None:
            return None

        preferred = cache_root / filename
        if self._usable_cached_file(preferred):
            return preferred

        legacy = self.root / filename
        if self._usable_cached_file(legacy):
            return legacy

        return None

    def _download_member_to_cache(self, filename: str, url: str) -> Path:
        """Download one ZIP member once and persist it atomically.

        The final file is only exposed after the complete member has been
        written. A unique temporary file also makes simultaneous model builds
        safe: they may redundantly download the same missing member, but a
        partial file can never be mistaken for a completed cache entry.
        """
        cached = self._find_cached_file(filename)
        if cached is not None:
            self.logger.info("Reusing cached GLOPOP-SG file %s", cached)
            return cached

        cache_root = self._cache_root()
        if cache_root is None:
            raise ValueError(
                "Persistent GLOPOP-SG caching requires a configured adapter cache."
            )
        output_path = cache_root / filename

        info_dict = self._get_zip_infolist(url)
        if filename not in info_dict:
            raise FileNotFoundError(f"{filename} not found in remote zip.")
        file_size = int(info_dict[filename].file_size)

        retry_start = time.monotonic()
        while True:
            temporary_path: Path | None = None
            try:
                # Another process may have finished while this process was
                # obtaining ZIP metadata or waiting after a 429 response.
                cached = self._find_cached_file(filename)
                if cached is not None:
                    self.logger.info("Reusing cached GLOPOP-SG file %s", cached)
                    return cached

                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{filename}.",
                    suffix=".part",
                    dir=cache_root,
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    with zipfile.ZipFile(RemoteFile(url), "r") as zf:
                        with (
                            zf.open(filename) as source,
                            tqdm(
                                total=file_size,
                                unit="B",
                                unit_scale=True,
                                desc=f"Downloading {filename}",
                            ) as pbar,
                        ):
                            while True:
                                chunk = source.read(self._DOWNLOAD_CHUNK_SIZE)
                                if not chunk:
                                    break
                                temporary_file.write(chunk)
                                pbar.update(len(chunk))
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())

                if temporary_path.stat().st_size != file_size:
                    raise IOError(
                        f"Incomplete GLOPOP-SG download for {filename}: "
                        f"expected {file_size} bytes, got "
                        f"{temporary_path.stat().st_size}."
                    )

                os.replace(temporary_path, output_path)
                self.logger.info("Cached GLOPOP-SG file at %s", output_path)
                return output_path
            except HTTP429Error:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                if time.monotonic() - retry_start > self._MAX_RETRY_S:
                    raise TimeoutError(
                        "HTTP 429 retries exceeded the "
                        f"{self._MAX_RETRY_S / 3600:.0f}-hour limit for {url}."
                    )
                time.sleep(self._RETRY_429_SLEEP_S)
            except Exception:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                raise

    def _extract_from_remote_to_temporary_file(
        self,
        filename: str,
        url: str,
    ) -> Path:
        """Download one member to a temporary file for unconfigured adapters."""
        info_dict = self._get_zip_infolist(url)
        if filename not in info_dict:
            raise FileNotFoundError(f"{filename} not found in remote zip.")
        file_size = int(info_dict[filename].file_size)

        retry_start = time.monotonic()
        while True:
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix="glopop_sg_",
                    suffix=Path(filename).suffix,
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    with zipfile.ZipFile(RemoteFile(url), "r") as zf:
                        with (
                            zf.open(filename) as source,
                            tqdm(
                                total=file_size,
                                unit="B",
                                unit_scale=True,
                                desc=f"Downloading {filename}",
                            ) as pbar,
                        ):
                            while True:
                                chunk = source.read(self._DOWNLOAD_CHUNK_SIZE)
                                if not chunk:
                                    break
                                temporary_file.write(chunk)
                                pbar.update(len(chunk))
                return temporary_path
            except HTTP429Error:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                if time.monotonic() - retry_start > self._MAX_RETRY_S:
                    raise TimeoutError(
                        "HTTP 429 retries exceeded the "
                        f"{self._MAX_RETRY_S / 3600:.0f}-hour limit for {url}."
                    )
                time.sleep(self._RETRY_429_SLEEP_S)
            except Exception:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                raise

    def _get_region_file(self, filename: str, url: str) -> tuple[Path, bool]:
        """Return a local region file and whether it should be deleted after use."""
        if self._cache_root() is not None:
            return self._download_member_to_cache(filename, url), False
        return self._extract_from_remote_to_temporary_file(filename, url), True

    def read(self, region: str) -> tuple[pd.DataFrame, xr.DataArray]:
        """Read GLOPOP-SG data for a region.

        When the adapter uses the normal data-catalog cache configuration, the
        two source members required for a region are downloaded only once and
        reused by subsequent builds. If no adapter cache is configured, the
        previous temporary-download behaviour is retained.

        Args:
            region: The GDL region code.

        Returns:
            Tuple of (DataFrame of population, DataArray of grid).
        """
        tif_name = f"{region}_grid_nr.tif"
        gz_name = f"synthpop_{region}_grid.dat.gz"

        tif_path, remove_tif = self._get_region_file(tif_name, self.url)
        gz_path, remove_gz = self._get_region_file(gz_name, self.url)

        try:
            GLOPOP_grid = rxr.open_rasterio(tif_path)
            assert isinstance(GLOPOP_grid, xr.DataArray)
            GLOPOP_grid = GLOPOP_grid.load()

            with gzip.open(gz_path, "rb") as f:
                GLOPOP_s = np.frombuffer(f.read(), dtype=np.int32)
        finally:
            if remove_tif:
                tif_path.unlink(missing_ok=True)
            if remove_gz:
                gz_path.unlink(missing_ok=True)

        GLOPOP_s_attribute_names: list[str] = [
            "HID",
            "RELATE_HEAD",
            "INCOME",
            "WEALTH",
            "RURAL",
            "AGE",
            "GENDER",
            "EDUC",
            "HHTYPE",
            "HHSIZE_CAT",
            "AGRI_OWNERSHIP",
            "FLOOR",
            "WALL",
            "ROOF",
            "SOURCE",
            "GRID_CELL",
        ]

        n_attr = len(GLOPOP_s_attribute_names)
        total = GLOPOP_s.size
        n_people = total // n_attr

        countries_with_17_columns = ["COD", "IRN", "KWT", "MKD", "THA"]
        # Some source regions contain 17 columns instead of the usual 16.
        if any(code in region for code in countries_with_17_columns):
            n_columns = 17
            n_people = GLOPOP_s.size // n_columns
            data_reshaped = np.reshape(GLOPOP_s, (n_columns, n_people)).transpose()
            data_reshaped = np.hstack((data_reshaped[:, :-2], data_reshaped[:, -1:]))
            print(region)
            print("17 columns")
        else:
            data_reshaped = np.reshape(GLOPOP_s, (n_attr, n_people)).transpose()

        df = pd.DataFrame(
            data_reshaped,
            columns=np.array(GLOPOP_s_attribute_names),
        )

        grid_vals = GLOPOP_grid.values
        if grid_vals.ndim == 3:
            grid_vals = grid_vals[0]

        mask = grid_vals > -1
        rows, cols = np.where(mask)
        grid_cells = grid_vals[mask]

        grid_coords = pd.DataFrame(
            {
                "GRID_Y": rows,
                "GRID_X": cols,
                "GRID_CELL": grid_cells,
            }
        )

        grid_coords["coord_Y"] = GLOPOP_grid.y.values[grid_coords["GRID_Y"].values]
        grid_coords["coord_X"] = GLOPOP_grid.x.values[grid_coords["GRID_X"].values]

        df = df.merge(grid_coords, on="GRID_CELL", how="left")

        return df, GLOPOP_grid
