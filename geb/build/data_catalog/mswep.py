"""The MSWEP Precipitation data adapter for downloading, processing, and storing precipitation data."""

import atexit
import calendar
import concurrent.futures
import io
import logging
import os
import re
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.features
import requests
import scipy.ndimage as ndimage
import xarray as xr
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from rasterio.transform import from_bounds
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry
from zarr.abc.codec import ArrayArrayCodec
from zarr.codecs import CastValue, ScaleOffset

from geb.build.data_catalog.base import Adapter
from geb.build.data_catalog.merit_basins import MeritBasinsCatchments
from geb.workflows.io import read_zarr, write_zarr

SCOPES: list[str] = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME_TYPE: str = "application/vnd.google-apps.folder"
KNOWN_MISSING_MSWEP_DATES: set[str] = {"1993-08-29", "1993-08-31"}


def _extract_folder_id(url_or_id: str) -> str:
    """Extract the Google Drive folder ID from a URL or raw ID string.

    Args:
        url_or_id: Google Drive folder URL or raw ID string.

    Returns:
        The extracted Google Drive folder ID.
    """
    cleaned: str = url_or_id.strip().strip("\"'")
    match: re.Match[str] | None = re.search(r"folders/([a-zA-Z0-9_-]+)", cleaned)
    if match:
        return match.group(1)
    match = re.search(r"file/d/([a-zA-Z0-9_-]+)", cleaned)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", cleaned)
    if match:
        return match.group(1)
    cleaned_id: str = cleaned.split("?")[0].split("/")[0]
    return cleaned_id


def _cleanup_loky() -> None:
    """Explicitly shutdown Loky process pools to prevent Python 3.14 semaphore leak warnings."""
    try:
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass


atexit.register(_cleanup_loky)


def _format_bytes(size_bytes: int) -> str:
    """Format bytes into human-readable string (B, KB, MB, GB, TB).

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., '14.25 MB').
    """
    if size_bytes == 0:
        return "0 B"
    units: list[str] = ["B", "KB", "MB", "GB", "TB"]
    unit_idx: int = 0
    size_float: float = float(size_bytes)
    while size_float >= 1024.0 and unit_idx < len(units) - 1:
        size_float /= 1024.0
        unit_idx += 1
    return f"{size_float:.2f} {units[unit_idx]}"


def _create_retry_session(
    retries: int = 5,
    backoff_factor: float = 1.0,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Create a requests Session configured with automatic connection retries and exponential backoff.

    Args:
        retries: Number of connection retries.
        backoff_factor: Multiplier for backoff delay.
        status_forcelist: HTTP status codes to automatically retry.

    Returns:
        A configured requests.Session instance.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy, pool_connections=32, pool_maxsize=32
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_google_credentials() -> Credentials:
    """Obtain valid Google OAuth2 credentials from config directory or environment variables.

    Returns:
        Valid Google Credentials instance with active access token.

    Raises:
        FileNotFoundError: If neither valid cached token nor client credentials in .env are found.
    """
    load_dotenv()

    token_dir: Path = Path.home() / ".config" / "geb"
    token_path: Path = token_dir / "google_drive_token.json"

    creds: Credentials | None = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if creds is not None and creds.valid:
        return creds

    if creds is not None and creds.expired and creds.refresh_token is not None:
        creds.refresh(Request())
        token_dir.mkdir(parents=True, exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())
        return creds

    client_id: str | None = os.getenv("GOOGLE_CLIENT_ID")
    client_secret: str | None = os.getenv("GOOGLE_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise FileNotFoundError(
            "Google Drive credentials not found. Please specify GOOGLE_CLIENT_ID and "
            "GOOGLE_CLIENT_SECRET in your .env file or authorize via ~/.config/geb/google_drive_token.json."
        )

    client_config: dict[str, Any] = {
        "installed": {
            "client_id": client_id.strip(),
            "client_secret": client_secret.strip(),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
    flow: InstalledAppFlow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    creds = flow.run_local_server(port=0)

    token_dir.mkdir(parents=True, exist_ok=True)
    with open(token_path, "w", encoding="utf-8") as token_file:
        token_file.write(creds.to_json())

    return creds


_token_lock = threading.Lock()


def _get_auth_header(
    creds: Credentials,
    force_refresh: bool = False,
) -> dict[str, str]:
    """Obtain authorization header for requests, refreshing the token if expired or forced.

    Args:
        creds: Authenticated Google Credentials instance.
        force_refresh: Whether to force a token refresh even if creds.valid is True.

    Returns:
        Dictionary containing the Authorization Bearer header.
    """
    with _token_lock:
        if force_refresh or not creds.valid or creds.expired:
            creds.refresh(Request())
            token_dir: Path = Path.home() / ".config" / "geb"
            token_path: Path = token_dir / "google_drive_token.json"
            if token_path.exists():
                token_dir.mkdir(parents=True, exist_ok=True)
                with open(token_path, "w", encoding="utf-8") as token_file:
                    token_file.write(creds.to_json())
        return {"Authorization": f"Bearer {creds.token}"}


def _scan_folder(
    creds: Credentials,
    folder_id: str,
    name_filter: str,
) -> tuple[list[dict[str, Any]], int]:
    """Scan a Google Drive folder to retrieve file metadata and total size matching a filter.

    Args:
        creds: Authenticated Google Credentials instance.
        folder_id: ID of the Google Drive folder.
        name_filter: Filter string to match file names (e.g. '2025').

    Returns:
        A tuple of (list of file metadata dicts, total size in bytes).

    Raises:
        FileNotFoundError: If the folder does not exist or contains no matching files.
    """
    _get_auth_header(creds)

    service = build("drive", "v3", credentials=creds)

    files: list[dict[str, Any]] = []
    total_bytes: int = 0
    page_token: str | None = None

    query: str = (
        f"'{folder_id}' in parents and trashed = false and "
        f"mimeType != '{FOLDER_MIME_TYPE}' and name contains '{name_filter}'"
    )

    while True:
        results = (
            service.files()
            .list(
                q=query,
                pageSize=1000,
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageToken=page_token,
            )
            .execute()
        )

        items: list[dict[str, Any]] = results.get("files", [])
        for item in items:
            size_int: int = int(item.get("size", 0))
            files.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "size_bytes": size_int,
                }
            )
            total_bytes += size_int

        page_token = results.get("nextPageToken")
        if not page_token:
            break

    if not files:
        raise FileNotFoundError(
            f"No NetCDF files matching '{name_filter}' found in Google Drive folder '{folder_id}'. "
            f"Please ensure that your MSWEP_URL environment variable points specifically to the "
            f"'MSWEP_V316_test/Past/Hourly' subfolder containing the .nc files (and not the root Google Drive folder), "
            f"and that your Google account has access to this folder."
        )

    return files, total_bytes


def _download_and_extract_in_memory(
    file_id: str,
    file_name: str,
    session: requests.Session,
    creds: Credentials,
    max_attempts: int = 5,
) -> tuple[pd.Timestamp, np.ndarray]:
    """Download a single NetCDF file directly into memory and parse its 2D slice without writing to disk.

    Args:
        file_id: Google Drive file ID.
        file_name: Google Drive file name (e.g. '1979001.00.nc').
        session: Reusable requests Session.
        creds: Authenticated Google Credentials instance.
        max_attempts: Maximum number of download attempts before raising an exception.

    Returns:
        A tuple of (exact hourly timestamp as pd.Timestamp, 2D precipitation numpy array as float32).

    Raises:
        requests.RequestException: If download fails after max_attempts.
        OSError: If network or memory stream error occurs.
        KeyError: If precipitation variable is missing.
        RuntimeError: If download fails after all retry attempts.
    """
    url: str = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"

    # Parse exact timestamp from MSWEP filename format YYYYJJJ.HH.nc to avoid float rounding in raw NetCDF time
    match: re.Match[str] | None = re.search(r"(\d{4})(\d{3})\.(\d{2})\.nc", file_name)
    if match:
        f_year: int = int(match.group(1))
        f_yday: int = int(match.group(2))
        f_hour: int = int(match.group(3))
        file_ts: pd.Timestamp = pd.Timestamp(  # ty:ignore[invalid-assignment]
            year=f_year, month=1, day=1
        ) + pd.Timedelta(days=f_yday - 1, hours=f_hour)
    else:
        file_ts = None

    for attempt in range(1, max_attempts + 1):
        try:
            auth_header: dict[str, str] = _get_auth_header(creds)
            response: requests.Response = session.get(
                url, headers=auth_header, timeout=(10, 60)
            )
            if response.status_code == 401:
                # Force refresh token in case token expired during download or concurrent requests
                _get_auth_header(creds, force_refresh=True)

            response.raise_for_status()

            bio = io.BytesIO(response.content)
            with xr.open_dataset(bio, engine="h5netcdf") as ds:
                if "precipitation" not in ds.data_vars:
                    raise KeyError(
                        f"'precipitation' variable missing. Available: {list(ds.data_vars.keys())}"
                    )
                if file_ts is not None:
                    t_val: pd.Timestamp = file_ts
                elif "time" in ds.coords and ds.time.size > 0:
                    raw_val = ds.time.values[0] if ds.time.ndim > 0 else ds.time.values
                    t_val = pd.Timestamp(raw_val).round("h")
                else:
                    t_val = pd.Timestamp("1970-01-01")

                arr_2d: np.ndarray = (
                    ds["precipitation"].values.squeeze().astype(np.float32)
                )

            return t_val, arr_2d  # ty:ignore[invalid-return-type]
        except (requests.RequestException, OSError) as exc:
            if attempt == max_attempts:
                raise exc
            sleep_duration: float = min(2.0**attempt, 30.0)
            time.sleep(sleep_duration)

    raise RuntimeError(f"Failed to download and parse file {file_id} in memory.")


def _extract_spatial_coords(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract 1D y (latitude) and x (longitude) coordinate arrays from a NetCDF dataset.

    Args:
        ds: The xarray Dataset from an MSWEP NetCDF file.

    Returns:
        A tuple of (y_coords, x_coords) as 1D numpy float arrays.

    Raises:
        KeyError: If neither 'lat'/'latitude' nor 'lon'/'longitude' are found in dataset coordinates.
    """
    if "lat" in ds.coords:
        y: np.ndarray = ds.coords["lat"].values.astype(np.float64)
    elif "latitude" in ds.coords:
        y = ds.coords["latitude"].values.astype(np.float64)
    elif "y" in ds.coords:
        y = ds.coords["y"].values.astype(np.float64)
    else:
        raise KeyError(
            f"Latitude coordinate not found in dataset. Available coordinates: {list(ds.coords.keys())}"
        )

    if "lon" in ds.coords:
        x: np.ndarray = ds.coords["lon"].values.astype(np.float64)
    elif "longitude" in ds.coords:
        x = ds.coords["longitude"].values.astype(np.float64)
    elif "x" in ds.coords:
        x = ds.coords["x"].values.astype(np.float64)
    else:
        raise KeyError(
            f"Longitude coordinate not found in dataset. Available coordinates: {list(ds.coords.keys())}"
        )

    return y, x


def _fetch_spatial_coords(
    file_id: str,
    session: requests.Session,
    creds: Credentials,
    max_attempts: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Download a single NetCDF file header to extract spatial coordinates directly from the source.

    Args:
        file_id: Google Drive file ID.
        session: Reusable requests Session.
        creds: Authenticated Google Credentials instance.
        max_attempts: Maximum number of download attempts.

    Returns:
        A tuple of (y_coords, x_coords) extracted from the NetCDF file.

    Raises:
        requests.RequestException: If download fails after max_attempts.
        OSError: If reading the NetCDF stream fails.
        RuntimeError: If download fails after all retry attempts.
    """
    url: str = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"

    for attempt in range(1, max_attempts + 1):
        try:
            auth_header: dict[str, str] = _get_auth_header(creds)
            response: requests.Response = session.get(
                url, headers=auth_header, timeout=(10, 60)
            )
            if response.status_code == 401:
                _get_auth_header(creds, force_refresh=True)

            response.raise_for_status()

            bio = io.BytesIO(response.content)
            with xr.open_dataset(bio, engine="h5netcdf") as ds:
                return _extract_spatial_coords(ds)
        except (requests.RequestException, OSError) as exc:
            if attempt == max_attempts:
                raise exc
            sleep_duration: float = min(2.0**attempt, 30.0)
            time.sleep(sleep_duration)

    raise RuntimeError(f"Failed to fetch spatial coordinates from file {file_id}.")


def create_merit_basins_mask(
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    cache_path: Path | None = None,
    buffer_km: float = 100.0,
    batch_size: int = 100_000,
) -> xr.DataArray:
    """Create and cache a 2D boolean raster mask from MERIT Basins catchments with a buffer.

    If cache_path exists, loads and returns the cached mask immediately. Otherwise rasterizes
    catchments in batches with a progress bar, applies morphological dilation, caches to disk, and returns.

    Args:
        y_coords: 1D array of latitude coordinates.
        x_coords: 1D array of longitude coordinates.
        cache_path: Optional path to save and load the cached Zarr mask store.
        buffer_km: Buffer distance in kilometers (default: 100.0).
        batch_size: Number of catchment geometries to rasterize per batch (default: 100_000).

    Returns:
        2D boolean xarray DataArray (shape: ny, nx) where True indicates valid basin/land cells.
    """
    if cache_path is not None and cache_path.exists():
        mask_da: xr.DataArray = read_zarr(cache_path)
        return mask_da

    ny: int = len(y_coords)
    nx: int = len(x_coords)

    # Load MERIT basins through the data catalog adapter
    merit_adapter = MeritBasinsCatchments(
        folder="merit_basins_catchments",
        filename="merit_basins_catchments.parquet",
        local_version=1,
        cache="global",
    )
    if not merit_adapter.is_ready:
        merit_adapter.fetch()

    logging.getLogger(__name__).debug("Loading MERIT basins catchments...")
    gdf: gpd.GeoDataFrame = gpd.read_parquet(merit_adapter.path, columns=["geometry"])

    west: float = float(x_coords.min()) - 0.05
    east: float = float(x_coords.max()) + 0.05
    south: float = float(y_coords.min()) - 0.05
    north: float = float(y_coords.max()) + 0.05

    transform = from_bounds(west, south, east, north, nx, ny)

    base_mask: np.ndarray = np.zeros((ny, nx), dtype=np.uint8)

    for i in tqdm(
        range(0, len(gdf), batch_size),
        desc="Rasterizing MERIT basins",
        dynamic_ncols=True,
    ):
        batch_geoms = gdf.geometry.iloc[i : i + batch_size]
        rasterio.features.rasterize(
            shapes=batch_geoms,
            out=base_mask,
            transform=transform,
            default_value=1,
        )

    if buffer_km > 0:
        cell_size_km: float = 11.132
        radius_pixels: int = max(1, int(np.round(buffer_km / cell_size_km)))
        y, x = np.ogrid[
            -radius_pixels : radius_pixels + 1,
            -radius_pixels : radius_pixels + 1,
        ]
        struct: np.ndarray = (x**2 + y**2) <= radius_pixels**2
        buffered_mask_arr: np.ndarray = ndimage.binary_dilation(
            base_mask > 0, structure=struct
        )
    else:
        buffered_mask_arr = base_mask > 0

    mask_da = xr.DataArray(
        buffered_mask_arr,
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x"),
        name="precipitation_mask",
        attrs={"_FillValue": None},
    )

    if cache_path is not None:
        logging.getLogger(__name__).debug(f"Caching mask to {cache_path}...")
        write_zarr(
            da=mask_da,
            path=cache_path,
            crs=4326,
            compression_level=22,
            progress=True,
        )

    return mask_da


def _export_mean_plot(da: xr.DataArray, output_path: Path, title: str) -> None:
    """Export a geographic plot of mean precipitation as a PNG file.

    Args:
        da: 3D or 2D xarray DataArray of precipitation in mm/hr.
        output_path: Target PNG file path.
        title: Plot title.
    """
    mean_da: xr.DataArray = (
        da.mean(dim="time", skipna=True) if "time" in da.dims else da
    )

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    mean_da.plot.imshow(
        ax=ax,
        cmap="viridis",
        robust=True,
        cbar_kwargs={"label": "Mean Precipitation (mm/h)"},
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


class MSWEPPrecipitation(Adapter):
    """The MSWEPPrecipitation adapter for downloading, masking, and storing precipitation data.

    Args:
        Adapter: The base Adapter class.
    """

    def __init__(
        self,
        folder: str = "mswep_precipitation",
        filename: str = "precipitation.zarr",
        local_version: int = 1,
        cache: str = "global",
        folder_id: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the MSWEPPrecipitation adapter.

        Args:
            folder: Subfolder within the cache directory.
            filename: Base filename for the processed Zarr dataset.
            local_version: Version number.
            cache: Cache type ('global' or 'local').
            folder_id: Optional Google Drive folder ID containing the precipitation NetCDF files.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.folder_id: str | None = folder_id
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        super().__init__(
            folder=folder,
            filename=filename,
            local_version=local_version,
            cache=cache,
            *args,
            **kwargs,
        )

    def _month_path(self, year: int, month: int) -> Path:
        """Get the file path for a specific month's Zarr store.

        Args:
            year: Year to retrieve path for.
            month: Month (1-12) to retrieve path for.

        Returns:
            Path to precipitation_{year}_{month:02d}.zarr.
        """
        return self.root / f"precipitation_{year}_{month:02d}.zarr"

    def _month_plot_path(self, year: int, month: int) -> Path:
        """Get the file path for a specific month's mean precipitation PNG plot.

        Args:
            year: Year to retrieve plot path for.
            month: Month (1-12) to retrieve plot path for.

        Returns:
            Path to precipitation_{year}_{month:02d}_mean.png.
        """
        return self.root / f"precipitation_{year}_{month:02d}_mean.png"

    @property
    def mask_path(self) -> Path:
        """Local path to the 2D spatial mask Zarr store.

        Returns:
            Path to precipitation_mask.zarr.
        """
        return self.root / "precipitation_mask.zarr"

    def fetch(
        self,
        url: str | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        threads: int = 16,
        compression_level: int = 22,
        *args: Any,
        **kwargs: Any,
    ) -> Adapter:
        """Download raw precipitation files from Google Drive month-by-month, mask, and save as 2D Zarr stores.

        Streams and parses files directly in memory month-by-month without writing temporary NetCDF files to disk.
        Chunks are configured to 200 x 200 cells (~20° x 20°) with 1 chunk in time and stored in a single shard.
        Also exports a summary PNG plot of the mean precipitation for each processed month.

        Args:
            url: Optional URL or Google Drive folder URL.
            start_date: Start date (inclusive, e.g. '2025-01-01' or Timestamp).
            end_date: End date (inclusive, e.g. '2025-12-31' or Timestamp).
            threads: Number of concurrent download worker threads (default: 16).
            compression_level: ZSTD compression level (default: 22).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The Adapter instance.

        Raises:
            ValueError: If start_date is after end_date.
            FileNotFoundError: If a required MSWEP precipitation file is missing from Google Drive.
        """
        if url is not None:
            self.folder_id = _extract_folder_id(url)

        if start_date is None or end_date is None:
            return self

        start_ts: pd.Timestamp = pd.Timestamp(start_date)  # ty:ignore[invalid-assignment]
        end_ts: pd.Timestamp = pd.Timestamp(end_date)  # ty:ignore[invalid-assignment]
        if start_ts.hour == 0 and start_ts.minute == 0 and start_ts.second == 0:
            start_ts = start_ts.replace(hour=1)

        if start_ts > end_ts:
            raise ValueError(
                f"start_date ({start_ts}) cannot be after end_date ({end_ts})"
            )

        if start_ts < pd.Timestamp("1979-01-01 01:00:00"):
            raise ValueError(
                f"MSWEP precipitation data is only available from 1979 onwards. "
                f"Requested start_date '{start_ts.strftime('%Y-%m-%d %H:%M:%S')}' is before 1979-01-01 01:00:00."
            )

        if self.folder_id is None:
            load_dotenv()
            env_url: str | None = os.getenv("MSWEP_URL")
            if env_url:
                self.folder_id = _extract_folder_id(env_url)

        if self.folder_id is None:
            raise ValueError(
                "MSWEP_URL environment variable is not set. "
                "Due to data distribution terms, the MSWEP download URL cannot be published publicly. "
                "Please visit https://www.gloh2o.org/mswep/ to request access to the dataset. "
                "Within the Google Drive, navigate to the 'MSWEP_V316_test/Past/Hourly' folder "
                "and set the MSWEP_URL environment variable in your .env file or environment "
                "(e.g. MSWEP_URL=https://drive.google.com/drive/folders/<folder_id>)."
            )

        creds: Credentials | None = None
        y_coords: np.ndarray | None = None
        x_coords: np.ndarray | None = None
        mask_arr: np.ndarray | None = None

        if self.mask_path.exists():
            mask_da: xr.DataArray = read_zarr(self.mask_path)
            y_coords = mask_da.y.values
            x_coords = mask_da.x.values
            mask_arr = mask_da.values.astype(bool)

        raw_end_ts: pd.Timestamp = end_ts - pd.Timedelta(hours=1)  # ty:ignore[invalid-assignment]

        for current_year in range(start_ts.year, raw_end_ts.year + 1):
            year_str: str = str(current_year)

            # Determine months to process for current_year
            min_m: int = start_ts.month if current_year == start_ts.year else 1
            max_m: int = raw_end_ts.month if current_year == raw_end_ts.year else 12

            months_to_process: list[int] = []
            for month in range(min_m, max_m + 1):
                month_zarr_path: Path = self._month_path(current_year, month)
                month_plot_path: Path = self._month_plot_path(current_year, month)
                _, last_day = calendar.monthrange(current_year, month)
                expected_hours: int = last_day * 24

                if month_zarr_path.exists() and self.mask_path.exists():
                    da_existing: xr.DataArray = read_zarr(month_zarr_path)
                    if da_existing.time.size == expected_hours:
                        if not month_plot_path.exists():
                            _export_mean_plot(
                                da_existing,
                                month_plot_path,
                                f"MSWEP Mean Precipitation ({current_year}-{month:02d})",
                            )
                        continue
                    else:
                        self.logger.warning(
                            f"Month {current_year}-{month:02d} store at {month_zarr_path} is incomplete "
                            f"({da_existing.time.size}/{expected_hours} hours). Re-fetching."
                        )
                months_to_process.append(month)

            if not months_to_process:
                continue

            if creds is None:
                self.logger.debug("Authenticating with Google Drive...")
                creds = _get_google_credentials()

            self.logger.debug(f"Scanning Google Drive for year {year_str}...")
            all_year_files, _ = _scan_folder(
                creds, self.folder_id, name_filter=year_str
            )

            if y_coords is None or x_coords is None:
                session: requests.Session = _create_retry_session()
                self.logger.debug(
                    "Extracting spatial coordinates directly from NetCDF source..."
                )
                y_coords, x_coords = _fetch_spatial_coords(
                    file_id=all_year_files[0]["id"],
                    session=session,
                    creds=creds,
                )

            if mask_arr is None:
                mask_da = create_merit_basins_mask(
                    y_coords=y_coords,
                    x_coords=x_coords,
                    cache_path=self.mask_path,
                    buffer_km=100.0,
                )
                mask_arr = mask_da.values.astype(bool)

            assert (
                y_coords is not None and x_coords is not None and mask_arr is not None
            )

            for month in months_to_process:
                month_zarr_path: Path = self._month_path(current_year, month)
                month_plot_path: Path = self._month_plot_path(current_year, month)

                # Calculate full start and end day of the month so stores are always complete
                _, last_day = calendar.monthrange(current_year, month)
                m_start: pd.Timestamp = pd.Timestamp(  # ty:ignore[invalid-assignment]
                    f"{current_year}-{month:02d}-01 00:00:00"
                )
                m_end: pd.Timestamp = pd.Timestamp(  # ty:ignore[invalid-assignment]
                    f"{current_year}-{month:02d}-{last_day:02d} 23:59:59"
                )

                start_yday: int = m_start.timetuple().tm_yday
                end_yday: int = m_end.timetuple().tm_yday

                files: list[dict[str, Any]] = []
                for f_info in all_year_files:
                    match: re.Match[str] | None = re.search(
                        r"(\d{4})(\d{3})\.\d{2}\.nc", f_info["name"]
                    )
                    if not match:
                        continue
                    file_year: int = int(match.group(1))
                    file_yday: int = int(match.group(2))
                    if (
                        file_year == current_year
                        and start_yday <= file_yday <= end_yday
                    ):
                        files.append(f_info)

                files.sort(key=lambda x: x["name"])

                if not files:
                    self.logger.debug(
                        f"No files for {current_year}-{month:02d} (days {start_yday}-{end_yday}). Skipping."
                    )
                    continue

                batch_bytes: int = sum(f["size_bytes"] for f in files)
                self.logger.debug(
                    f"Processing {current_year}-{month:02d} ({len(files)} files, {_format_bytes(batch_bytes)})"
                )

                expected_time_range: pd.DatetimeIndex = pd.date_range(
                    f"{current_year}-{month:02d}-01 00:00:00",
                    f"{current_year}-{month:02d}-{last_day:02d} 23:00:00",
                    freq="1h",
                )
                time_to_idx: dict[pd.Timestamp, int] = {
                    ts: idx for idx, ts in enumerate(expected_time_range)
                }
                n_timesteps: int = len(expected_time_range)

                # Allocate in-memory array for this full month (~9 GB RAM)
                precip_month: np.ndarray = np.full(
                    (n_timesteps, len(y_coords), len(x_coords)),
                    fill_value=65535,
                    dtype=np.uint16,
                )
                filled_timesteps: set[int] = set()

                def _worker_task(
                    f_info: dict[str, Any],
                ) -> tuple[pd.Timestamp, np.ndarray]:
                    session: requests.Session = _create_retry_session()
                    return _download_and_extract_in_memory(
                        file_id=f_info["id"],
                        file_name=f_info["name"],
                        session=session,
                        creds=creds,
                    )

                with tqdm(
                    total=len(files),
                    dynamic_ncols=True,
                ) as pbar:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=threads
                    ) as executor:
                        future_to_file = {
                            executor.submit(_worker_task, f_info): f_info
                            for f_info in files
                        }
                        for future in concurrent.futures.as_completed(future_to_file):
                            t_val, arr_2d = future.result()
                            idx = time_to_idx.get(t_val)
                            if idx is not None:
                                valid_mask = mask_arr & ~np.isnan(arr_2d)
                                encoded_arr = np.full_like(
                                    arr_2d, 65535, dtype=np.uint16
                                )
                                encoded_arr[valid_mask] = np.clip(
                                    np.round(arr_2d[valid_mask] * 100.0), 0, 50000
                                ).astype(np.uint16)

                                precip_month[idx] = encoded_arr
                                filled_timesteps.add(idx)
                            pbar.update(1)

                # Fill known missing dates with zeros
                zero_land_arr: np.ndarray = np.full(
                    (len(y_coords), len(x_coords)), 65535, dtype=np.uint16
                )
                zero_land_arr[mask_arr] = 0

                missing_indices: list[int] = [
                    i for i in range(n_timesteps) if i not in filled_timesteps
                ]
                for missing_idx in missing_indices:
                    missing_ts = expected_time_range[missing_idx]
                    missing_date_str = missing_ts.strftime("%Y-%m-%d")
                    if missing_date_str in KNOWN_MISSING_MSWEP_DATES:
                        precip_month[missing_idx] = zero_land_arr
                        self.logger.info(
                            f"Filled missing MSWEP date {missing_ts} with zeros."
                        )
                    else:
                        raise FileNotFoundError(
                            f"Missing MSWEP precipitation file for {missing_ts} during fetch of {current_year}-{month:02d}."
                        )

                da_precip: xr.DataArray = xr.DataArray(
                    np.where(
                        precip_month == 65535,
                        np.nan,
                        precip_month.astype(np.float32) / 100.0,
                    ),
                    coords={
                        "time": expected_time_range,
                        "y": y_coords,
                        "x": x_coords,
                    },
                    dims=("time", "y", "x"),
                    name="precipitation",
                    attrs={"/FillValue": np.nan},
                )
                del precip_month

                # Chunking: 200 x 200 spatial (~20° x 20°), all timesteps in 1 time chunk
                da_chunked: xr.DataArray = da_precip.chunk(
                    {"time": -1, "y": 200, "x": 200}
                )

                # Source precipitation in MSWEP has 0.01 mm/h precision over 0 - 500 mm/h.
                # ScaleOffset(scale=100.0) converts float mm/h to integer (0 - 50000).
                # CastValue quantizes to uint16, mapping NaN to 65535 for high compression.
                filters: list[ArrayArrayCodec] = [
                    ScaleOffset(offset=0.0, scale=100.0),
                    CastValue(
                        data_type="uint16",
                        rounding="nearest-even",
                        out_of_range=None,
                        scalar_map={
                            "encode": {"NaN": 65535},
                            "decode": {65535: np.nan},
                        },
                    ),
                ]

                write_zarr(
                    da_chunked,
                    month_zarr_path,
                    crs=4326,
                    filters=filters,
                    compression_level=compression_level,
                    shards={"time": 1, "y": 1, "x": 1},
                )

                _export_mean_plot(
                    da_precip,
                    month_plot_path,
                    f"MSWEP Mean Precipitation ({current_year}-{month:02d})",
                )

                self.logger.debug(
                    f"Month {current_year}-{month:02d} saved to {month_zarr_path}"
                )

        return self

    def read(
        self,
        start_date: datetime | date | str | pd.Timestamp,
        end_date: datetime | date | str | pd.Timestamp,
        bounds: tuple[float, float, float, float],
    ) -> xr.DataArray:
        """Read MSWEP precipitation data for a given time period and spatial bounding box.

        Converts precipitation from mm/hour to meters (depth of water per hour), matching ERA5-Land tp,
        and adds 1 hour to timestamps so start-of-interval MSWEP conventions align with end-of-interval ERA5 timestamps (starting at 01:00:00).
        Strictly validates that all required months within the date range are present.

        Args:
            start_date: Start date of the time period to read.
            end_date: End date of the time period to read.
            bounds: Bounding box in the format (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.

        Returns:
            The precipitation DataArray in meters (depth of water per hour).

        Raises:
            ValueError: If start_date is after end_date, start_date is before 1979-01-01 01:00:00, or bounds are invalid.
            FileNotFoundError: If any required month's precipitation store is missing.
        """
        start_ts: pd.Timestamp = pd.Timestamp(start_date)  # ty:ignore[invalid-assignment]
        end_ts: pd.Timestamp = pd.Timestamp(end_date)  # ty:ignore[invalid-assignment]
        if start_ts.hour == 0 and start_ts.minute == 0 and start_ts.second == 0:
            start_ts = start_ts.replace(hour=1)

        if start_ts > end_ts:
            raise ValueError(
                f"start_date ({start_ts}) cannot be after end_date ({end_ts})."
            )

        if start_ts < pd.Timestamp("1979-01-01 01:00:00"):
            raise ValueError(
                f"MSWEP precipitation data is only available from 1979 onwards. "
                f"Requested start_date '{start_ts.strftime('%Y-%m-%d %H:%M:%S')}' is before 1979-01-01 01:00:00."
            )

        if (
            len(bounds) != 4
            or bounds[0] >= bounds[2]
            or bounds[1] >= bounds[3]
            or bounds[0] < -180.0
            or bounds[2] > 180.0
            or bounds[1] < -90.0
            or bounds[3] > 90.0
        ):
            raise ValueError(
                f"Invalid bounding box: {bounds}. Expected (min_lon, min_lat, max_lon, max_lat) in EPSG:4326."
            )

        # Log a warning if any known missing dates (filled with zeros) are within the requested range
        requested_missing_dates: list[str] = [
            d
            for d in sorted(KNOWN_MISSING_MSWEP_DATES)
            if start_ts <= pd.Timestamp(f"{d} 23:59:59")
            and end_ts >= pd.Timestamp(f"{d} 00:00:00")
        ]
        if requested_missing_dates:
            missing_dates_str: str = ", ".join(requested_missing_dates)
            self.logger.warning(
                f"MSWEP precipitation data for date(s) {missing_dates_str} is missing from the upstream source "
                f"and has been filled with zeros."
            )

        # Raw MSWEP filenames YYYYDOY.HH represent accumulation over [HH:00, HH+1:00].
        # In ERA5, timestamps mark the end of the accumulation interval (e.g. 01:00 for [00:00, 01:00]).
        # To align MSWEP timestamps with ERA5, we add +1 hour to MSWEP coordinates upon reading.
        raw_end_ts: pd.Timestamp = end_ts - pd.Timedelta(hours=1)  # ty:ignore[invalid-assignment]

        # Enumerate all required (year, month) pairs in the date range
        required_months: list[tuple[int, int]] = []
        cur_year: int = start_ts.year
        cur_month: int = start_ts.month
        target_end_year: int = raw_end_ts.year
        target_end_month: int = raw_end_ts.month

        while (cur_year < target_end_year) or (
            cur_year == target_end_year and cur_month <= target_end_month
        ):
            required_months.append((cur_year, cur_month))
            if cur_month == 12:
                cur_year += 1
                cur_month = 1
            else:
                cur_month += 1

        # Check and collect stores for all required months
        stores_to_read: list[tuple[Path, int, int]] = []
        seen_paths: set[Path] = set()

        for yr, mo in required_months:
            month_path: Path = self._month_path(yr, mo)
            year_path: Path = self.root / f"precipitation_{yr}.zarr"

            if month_path.exists():
                chosen_path: Path = month_path
            elif year_path.exists():
                chosen_path = year_path
            else:
                raise FileNotFoundError(
                    f"Missing MSWEP precipitation data for {yr}-{mo:02d}. "
                    f"Expected store at '{month_path}'. Please fetch the dataset before reading."
                )

            if chosen_path not in seen_paths:
                seen_paths.add(chosen_path)
                stores_to_read.append((chosen_path, yr, mo))

        buffer_deg: float = 0.5
        min_x: float = max(-180.0, bounds[0] - buffer_deg)
        min_y: float = max(-90.0, bounds[1] - buffer_deg)
        max_x: float = min(180.0, bounds[2] + buffer_deg)
        max_y: float = min(90.0, bounds[3] + buffer_deg)

        das: list[xr.DataArray] = []
        for p, yr, mo in stores_to_read:
            da_part: xr.DataArray = read_zarr(p)
            da_part = da_part.assign_coords(
                time=pd.to_datetime(da_part.time.values).round("h").to_numpy()
            )

            # Validate that all expected hours for this month are present
            _, last_d = calendar.monthrange(yr, mo)
            expected_month_hours: int = last_d * 24
            if da_part.time.size != expected_month_hours:
                raise ValueError(
                    f"Incomplete MSWEP precipitation data in '{p.name}'. "
                    f"Found {da_part.time.size} hours, expected {expected_month_hours} hours for {yr}-{mo:02d}. "
                    f"Please re-fetch this month."
                )

            expected_month_times = pd.date_range(
                f"{yr}-{mo:02d}-01 00:00:00",
                f"{yr}-{mo:02d}-{last_d:02d} 23:00:00",
                freq="1h",
            )
            if not (da_part.time.values == expected_month_times.values).all():
                raise ValueError(
                    f"Time coordinates in '{p.name}' do not match expected continuous hourly timestamps for {yr}-{mo:02d}."
                )

            da_part = da_part.sel(
                x=slice(min_x, max_x),
                y=slice(max_y, min_y)
                if da_part.y[0] > da_part.y[-1]
                else slice(min_y, max_y),
            )
            if da_part.x.size == 0 or da_part.y.size == 0:
                raise ValueError(
                    f"Spatial slicing for bounds {bounds} produced empty dimensions in {p}."
                )
            # Convert from mm/hour to meters (depth of water per hour), matching ERA5-Land tp
            da_part = (da_part / 1000.0).astype(np.float32)

            # Add 1 hour to align MSWEP start-of-interval timestamps with ERA5 end-of-interval timestamps
            da_part = da_part.assign_coords(
                time=pd.to_datetime(da_part.time.values) + pd.Timedelta(hours=1)
            )
            das.append(da_part)

        combined_da: xr.DataArray = (
            xr.concat(das, dim="time") if len(das) > 1 else das[0]
        )

        combined_da = combined_da.sel(time=slice(start_ts, end_ts))

        expected_time_range = pd.date_range(start_ts, end_ts, freq="1h")
        if len(combined_da.time) != len(expected_time_range):
            raise ValueError(
                f"Missing timesteps in MSWEP data: expected {len(expected_time_range)} hourly timesteps "
                f"from {start_ts} to {end_ts}, but found {len(combined_da.time)}."
            )

        combined_da.attrs["units"] = "m"
        combined_da.attrs["long_name"] = "Total precipitation"
        combined_da.attrs["standard_name"] = "precipitation_amount"

        return combined_da
