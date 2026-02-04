"""Utilities to download MERIT Hydro tiles for a given bounding box.

This module provides a streaming downloader that extracts only the needed
GeoTIFF tiles from the remote 30x30-degree tar packages hosted by MERIT Hydro.
It avoids saving the full tar archives to disk by iterating over the HTTP
response stream and writing only the requested members.

Notes:
    - MERIT Hydro distributes 5x5-degree tiles grouped in 30x30-degree tar files.
      Tile filenames follow the pattern "n30w120_elv.tif" where the lat/lon refer
      to the lower-left corner of the tile in degrees. Package tar files follow
      the pattern "elv_n30w120.tar" where the coordinates indicate the lower-left
      corner of the 30x30 group. See the MERIT Hydro documentation for details.
    - Coverage spans from S60 to N90. Some packages or tiles are fully ocean and
      therefore do not exist. This module distinguishes between "missing because
      ocean/not provided" and actual download errors.

"""

from __future__ import annotations

import os
import tarfile
import time
from pathlib import Path
from typing import IO, Any, Iterable

import numpy as np
import requests
import rioxarray as rxr
import xarray as xr
from requests.auth import HTTPBasicAuth
from rioxarray import merge
from tqdm import tqdm

from geb.workflows.raster import convert_nodata

from .base import Adapter


class _ProgressReader:
    """Wrapper for file-like object to track read progress."""

    def __init__(self, fileobj: IO[bytes], pbar: tqdm) -> None:
        """Initialize the progress reader.

        Args:
            fileobj: File-like object to read from.
            pbar: tqdm progress bar to update.
        """
        self.fileobj: IO[bytes] = fileobj
        self.pbar: tqdm = pbar

    def read(self, size: int = -1) -> bytes:
        """Read data and update progress bar.

        Args:
            size: Read size in bytes. If -1, read all available. Defaults to -1.

        Returns:
            Data read from the file-like object.
        """
        data = self.fileobj.read(size)
        if data:
            self.pbar.update(len(data))
        return data

    def close(self) -> None:
        self.fileobj.close()


class MeritHydro(Adapter):
    """Dataset adapter for MERIT Hydro variables."""

    def __init__(self, variable: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter for a specific MERIT Hydro variable.

        Args:
            variable: MERIT Hydro variable to download ("elv" or "dir").

            *args: Additional positional arguments passed to the base Adapter class.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        self.variable = variable
        self._xmin: float | None = None
        self._xmax: float | None = None
        self._ymin: float | None = None
        self._ymax: float | None = None
        self._source_nodata: int | float | bool | None = None
        self._target_nodata: int | float | bool | None = None
        super().__init__(*args, **kwargs)

    @property
    def is_ready(self) -> bool:
        """Check if the data is already downloaded and processed.

        For MERIT Hydro, we always return False because readiness depends
        on the required tiles for a specific bounding box, which are checked
        during fetch().

        Returns:
            Always False to ensure fetch() is called.
        """
        return False

    def _get_latitude_hemisphere_and_degrees(self, lat_deg: int) -> tuple[str, int]:
        """Return hemisphere letter and absolute degrees for latitude.

        Args:
            lat_deg: Integer degrees latitude (lower-left corner of tile) (degrees).

        Returns:
            Tuple of hemisphere letter ("n" or "s") and two-digit absolute degrees.

        Raises:
            ValueError: If latitude is outside valid range [-90, 90].
        """
        if lat_deg < -90 or lat_deg > 90:
            raise ValueError("Latitude must be within [-90, 90] degrees.")
        if lat_deg >= 0:
            return "n", lat_deg
        return "s", -lat_deg

    def _get_longitude_hemisphere_and_degrees(self, lon_deg: int) -> tuple[str, int]:
        """Return hemisphere letter and absolute degrees for longitude.

        Args:
            lon_deg: Integer degrees longitude (lower-left corner of tile) (degrees).

        Returns:
            Tuple of hemisphere letter ("e" or "w") and three-digit absolute degrees.

        Raises:
            ValueError: If longitude is outside valid range [-180, 180].
        """
        if lon_deg < -180 or lon_deg > 180:
            raise ValueError("Longitude must be within [-180, 180] degrees.")
        if lon_deg >= 0:
            return "e", lon_deg
        return "w", -lon_deg

    def _compose_tile_filename(self, lat_ll: int, lon_ll: int) -> str:
        """Compose a 5x5-degree tile filename for a MERIT variable.

        Args:
            lat_ll: Lower-left latitude of tile (integer multiple of 5) (degrees).
            lon_ll: Lower-left longitude of tile (integer multiple of 5) (degrees).

        Returns:
            Tile filename like "n30w120_elv.tif".

        Raises:
            ValueError: If lat_ll or lon_ll are not multiples of 5.
        """
        if lat_ll % 5 != 0 or lon_ll % 5 != 0:
            raise ValueError(
                "lat_ll and lon_ll must be 5-degree aligned (multiples of 5)."
            )
        ns, alat = self._get_latitude_hemisphere_and_degrees(lat_ll)
        ew, alon = self._get_longitude_hemisphere_and_degrees(lon_ll)
        return f"{ns}{alat:02d}{ew}{alon:03d}_{self.variable}.tif"

    def _package_name(self, lat_ll: int, lon_ll: int) -> str:
        """Compose a 30x30-degree package tar filename for a MERIT variable.

        The package is defined by the lower-left corner of the 30-degree grid cell
        that contains the tile lower-left corner.

        Args:
            lat_ll: Lower-left latitude of tile (integer multiple of 5) (degrees).
            lon_ll: Lower-left longitude of tile (integer multiple of 5) (degrees).

        Returns:
            Package tar filename like "elv_n30w120.tar".
        """
        # Floor to 30-degree grid
        lat30 = (lat_ll // 30) * 30 if lat_ll >= 0 else -(((-lat_ll + 29) // 30) * 30)
        lon30 = (lon_ll // 30) * 30 if lon_ll >= 0 else -(((-lon_ll + 29) // 30) * 30)
        ns, alat = self._get_latitude_hemisphere_and_degrees(lat30)
        ew, alon = self._get_longitude_hemisphere_and_degrees(lon30)
        return f"{self.variable}_{ns}{alat:02d}{ew}{alon:03d}.tar"

    def _tiles_for_bbox(
        self, xmin: float, xmax: float, ymin: float, ymax: float
    ) -> list[tuple[int, int]]:
        """Compute all 5x5-degree lower-left tile coordinates intersecting a bbox.

        Args:
            xmin: Minimum longitude (degrees).
            xmax: Maximum longitude (degrees). May be less than xmin if crossing dateline is needed (not supported here).
            ymin: Minimum latitude (degrees).
            ymax: Maximum latitude (degrees).

        Returns:
            Sorted list of (lat_ll, lon_ll) integer tuples on 5-degree grid.

        Raises:
            ValueError: If bbox is invalid or crosses the antimeridian.
        """
        if xmax <= xmin:
            raise ValueError("Expected xmax > xmin and bbox not crossing antimeridian.")
        if ymax <= ymin:
            raise ValueError("Expected ymax > ymin.")

        # Clamp to plausible world bounds to avoid generating excessive tiles
        xmin_c = max(-180.0, xmin)
        xmax_c = min(180.0, xmax)
        ymin_c = max(-90.0, ymin)
        ymax_c = min(90.0, ymax)

        # Align to 5-degree grid (lower-left corners)
        def floor5(v: float) -> int:
            vi = int(v // 5 * 5)
            # Adjust for negatives that are exact multiples to ensure lower-left
            if v < 0 and v % 5 == 0:
                return vi
            return vi

        lat_start = floor5(ymin_c)
        lon_start = floor5(xmin_c)

        tiles: list[tuple[int, int]] = []
        lat = lat_start
        while lat < ymax_c:
            lon = lon_start
            while lon < xmax_c:
                # Tile bounds
                t_xmin, t_xmax = lon, lon + 5
                t_ymin, t_ymax = lat, lat + 5
                if not (
                    t_xmin >= xmax_c
                    or t_xmax <= xmin_c
                    or t_ymin >= ymax_c
                    or t_ymax <= ymin_c
                ):
                    tiles.append((lat, lon))
                lon += 5
            lat += 5
        # Unique and sorted for reproducibility
        tiles = sorted(set(tiles))
        return tiles

    def _group_tiles_by_package(
        self, tiles: Iterable[tuple[int, int]]
    ) -> dict[str, list[tuple[int, int]]]:
        """Group tile ll coords by the 30x30 package tar they belong to.

        Args:
            tiles: Iterable of (lat_ll, lon_ll) pairs (degrees).

        Returns:
            Mapping of package tar filename -> list of tile coords in that package.
        """
        groups: dict[str, list[tuple[int, int]]] = {}
        for lat_ll, lon_ll in tiles:
            pkg = self._package_name(lat_ll, lon_ll)
            groups.setdefault(pkg, []).append((lat_ll, lon_ll))
        return groups

    def _package_url(self, package_name: str, base_url: str) -> str:
        """Construct the full URL to a MERIT tar package for the given variable.

        Args:
            package_name: Tar filename produced by _package_name.
            base_url: Base URL for MERIT Hydro downloads.

        Returns:
            Full URL string to the tar file.
        """
        return f"{base_url}/{package_name}"

    def _merge_merit_tiles(self, tile_paths: list[Path]) -> xr.DataArray:
        """Load MERIT Hydro tiles into a single xarray DataArray.

        This function opens the provided GeoTIFF tile files using rioxarray,
        merges them into a single DataArray, and returns it in memory.

        Args:
            tile_paths: List of Paths to GeoTIFF files from download_merit.

        Returns:
            xarray DataArray with merged tiles, preserving CRS and coordinates.
        """
        das: list[xr.DataArray] = []
        for path in tile_paths:
            src = rxr.open_rasterio(path)
            assert isinstance(src, xr.DataArray)
            das.append(src.sel(band=1))

        da: xr.DataArray = merge.merge_arrays(das)
        return da

    def _missing_marker_path(self, tile_name: str) -> Path:
        """Construct path for a missing tile marker file.

        Args:
            tile_name: Name of the tile.

        Returns:
            Path to the marker file.
        """
        root = self.root
        assert root is not None
        return root / f"{tile_name}.missing.txt"

    def fetch(
        self,
        *,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        url: str,
        source_nodata: int | float | bool,
        target_nodata: int | float | bool,
        session: requests.Session | None = None,
        request_timeout_s: float = 60.0,
        attempts: int = 3,
    ) -> MeritHydro:
        """Ensure MERIT Hydro tiles intersecting a bbox are available locally.

        The function first checks for pre-staged 5x5-degree tiles in
        ``{cache_root}/{variable}/`` or ``{cache_root}/``. If tiles are
        missing, it downloads only the required GeoTIFFs for a single MERIT variable
        by streaming the remote 30x30-degree tar packages without saving the tars.
        If a package does not exist (HTTP 404), it is silently skipped. If a needed
        tile is not present inside an existing package (commonly ocean), it is also
        silently skipped. Any other error is retried up to ``attempts`` times and
        then raised.

        Authentication:
            Basic auth is required and read from environment variables:
            MERIT_USERNAME and MERIT_PASSWORD.

        Args:
            xmin: Minimum longitude of area of interest (degrees).
            xmax: Maximum longitude of area of interest (degrees).
            ymin: Minimum latitude of area of interest (degrees).
            ymax: Maximum latitude of area of interest (degrees).
            url: Base URL of the MERIT Hydro server.
            source_nodata: Nodata value in the source GeoTIFF files.
            target_nodata: Nodata value to use in the output DataArray.
            session: Optional requests.Session (used in tests or advanced usage).
            request_timeout_s: Timeout per HTTP request (seconds).
            attempts: Number of attempts for transient failures (errors are raised after this many tries).

        Returns:
            The MeritHydro instance.

        Raises:
            ValueError: If inputs are invalid or auth variables are missing.
            RuntimeError: If the HTTP client dependency is not available.
            requests.RequestException: If repeated HTTP errors occur.
            tarfile.ReadError: If tar parsing repeatedly fails.
        """
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._source_nodata = source_nodata
        self._target_nodata = target_nodata

        username = os.getenv("MERIT_USERNAME")
        password = os.getenv("MERIT_PASSWORD")
        if not username or not password:
            raise ValueError(
                "Authentication required: set MERIT_USERNAME and MERIT_PASSWORD in environment."
            )

        tiles: list[tuple[int, int]] = self._tiles_for_bbox(xmin, xmax, ymin, ymax)

        local_tile_dir: Path = self.root / self.variable
        missing_names: set[str] = set()
        for lat_ll, lon_ll in tiles:
            tile_name: str = self._compose_tile_filename(lat_ll, lon_ll)
            if (
                not (self.root / tile_name).exists()
                and not (local_tile_dir / tile_name).exists()
            ):
                if not self._missing_marker_path(tile_name).exists():
                    missing_names.add(tile_name)

        if not missing_names:
            return self

        # Prepare HTTP session
        if session is None:
            if requests is None:
                raise RuntimeError(
                    "requests is required for downloading but is not available."
                )
            session = requests.Session()
        auth = HTTPBasicAuth(username, password)

        missing_coords = []
        for lat_ll, lon_ll in tiles:
            if self._compose_tile_filename(lat_ll, lon_ll) in missing_names:
                missing_coords.append((lat_ll, lon_ll))

        groups = self._group_tiles_by_package(missing_coords)

        for package_name, coords in groups.items():
            package_url = self._package_url(package_name, base_url=url)
            needed_names = {
                self._compose_tile_filename(lat, lon) for lat, lon in coords
            }

            # HEAD to detect ocean/no-data packages (404) → skip silently
            try:
                head_resp = session.head(
                    package_url,
                    auth=auth,
                    allow_redirects=True,
                    timeout=request_timeout_s,
                )
            except Exception:
                if attempts <= 1:
                    raise
                retried = 1
                while retried < attempts:
                    time.sleep(min(2.0 * retried, 5.0))
                    try:
                        head_resp = session.head(
                            package_url,
                            auth=auth,
                            allow_redirects=True,
                            timeout=request_timeout_s,
                        )
                        break
                    except Exception:
                        retried += 1
                else:
                    raise

            if head_resp.status_code == 404:
                for tname in needed_names:
                    mm = self._missing_marker_path(tname)
                    mm.write_text(
                        "MERIT Hydro: tile not provided (ocean/no-data).\n",
                        encoding="utf-8",
                    )
                continue
            if head_resp.status_code in (401, 403):
                raise requests.RequestException(
                    f"Unauthorized: HTTP {head_resp.status_code} for {package_url}"
                )
            if head_resp.status_code >= 400:
                raise requests.RequestException(
                    f"HEAD error: HTTP {head_resp.status_code} for {package_url}"
                )

            content_length = int(head_resp.headers.get("Content-Length", 0))
            found_names: set[str] = set()
            for attempt in range(1, attempts + 1):
                try:
                    resp = session.get(
                        package_url, auth=auth, stream=True, timeout=request_timeout_s
                    )
                except Exception:
                    if attempt >= attempts:
                        raise
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue

                if resp.status_code >= 400:
                    resp.close()
                    if attempt >= attempts:
                        raise requests.RequestException(
                            f"GET error: HTTP {resp.status_code} for {package_url}"
                        )
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue

                try:
                    resp.raw.decode_content = True
                    print(f"Downloading and extracting from: {package_url}")
                    with tqdm(
                        total=content_length if content_length > 0 else None,
                        desc=f"Downloading {package_name}",
                        unit="B",
                        unit_scale=True,
                    ) as pbar:
                        progress_reader = _ProgressReader(resp.raw, pbar)
                        with tarfile.open(fileobj=progress_reader, mode="r|*") as tf:  # type: ignore
                            for member in tf:
                                if not member.isreg():
                                    continue
                                mname = Path(member.name).name
                                if mname in needed_names:
                                    ex = tf.extractfile(member)
                                    if ex is None:
                                        continue
                                    out_file = self.root / mname
                                    with out_file.open("wb") as fout:
                                        while True:
                                            chunk = ex.read(1024 * 1024)
                                            if not chunk:
                                                break
                                            fout.write(chunk)
                                    found_names.add(mname)
                                    if found_names == needed_names:
                                        break
                except tarfile.ReadError:
                    if attempt >= attempts:
                        raise
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue
                finally:
                    resp.close()
                break

            for tname in needed_names - found_names:
                mm = self._missing_marker_path(tname)
                mm.write_text(
                    "MERIT Hydro: tile not present in package (likely ocean).\n",
                    encoding="utf-8",
                )

        return self

    def read(self, **kwargs: Any) -> xr.DataArray:
        """Read and merge the MERIT Hydro tiles into a single DataArray.

        This method should be called after fetch(). It finds all available tiles
        for the stored bounding box, merges them, crops to the exact bbox,
        and applies nodata conversion.

        Args:
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Merged and cropped xarray DataArray.

        Raises:
            ValueError: If fetch() has not been called.
        """
        if self._xmin is None:
            raise ValueError(
                "fetch() must be called before read() to set the bounding box."
            )

        tiles: list[tuple[int, int]] = self._tiles_for_bbox(
            self._xmin, self._xmax, self._ymin, self._ymax
        )
        local_tile_dir: Path = self.root / self.variable
        results: list[Path] = []
        for lat_ll, lon_ll in tiles:
            tname = self._compose_tile_filename(lat_ll, lon_ll)
            tif_path = self.root / tname
            if tif_path.exists():
                results.append(tif_path)
                continue
            local_path = local_tile_dir / tname
            if local_path.exists():
                results.append(local_path)
                continue

        if not results:
            raise ValueError(
                f"No MERIT Hydro tiles found for bbox ({self._xmin}, {self._ymin}, {self._xmax}, {self._ymax})."
            )

        da: xr.DataArray = self._merge_merit_tiles(results)

        if "_FillValue" in da.attrs:
            assert da.attrs["_FillValue"] == self._source_nodata, (
                f"Expected source _FillValue {self._source_nodata}, got {da.attrs['_FillValue']}"
            )
        else:
            da.attrs["_FillValue"] = self._source_nodata

        da = da.sel(x=slice(self._xmin, self._xmax), y=slice(self._ymax, self._ymin))
        da = convert_nodata(da, self._target_nodata)
        return da


class MeritHydroDir(MeritHydro):
    """Dataset adapter for MERIT Hydro flow direction.

    Args:
        MeritHydro: Base class for MERIT Hydro datasets.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the adapter for flow direction data.

        Args:
            *args: Positional arguments passed to the base class.
            **kwargs: Keyword arguments passed to the base class.

        """
        super().__init__(variable="dir", **kwargs)

    def fetch(self, **kwargs: Any) -> MeritHydro:
        """Process and download flow direction data with specific fill value.

        Args:
            **kwargs: Keyword arguments passed to the base class fetcher.

        Returns:
            The MeritHydro instance.

        """
        return super().fetch(
            source_nodata=247,
            target_nodata=247,
            **kwargs,
        )


class MeritHydroElv(MeritHydro):
    """Dataset adapter for MERIT Hydro elevation.

    Args:
        MeritHydro: Base class for MERIT Hydro datasets.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the adapter for elevation data.

        Args:
            *args: Positional arguments passed to the base class.
            **kwargs: Keyword arguments passed to the base class.

        """
        super().__init__(variable="elv", **kwargs)

    def fetch(self, **kwargs: Any) -> MeritHydro:
        """Process and download elevation data with specific fill value.

        Args:
            **kwargs: Keyword arguments passed to the base class fetcher.

        Returns:
            The MeritHydro instance.

        """
        return super().fetch(
            source_nodata=-9999.0,
            target_nodata=np.nan,
            **kwargs,
        )
