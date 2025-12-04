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

from geb.workflows.io import write_zarr
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
        super().__init__(*args, **kwargs)

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
        das: list[xr.DataArray] = [rxr.open_rasterio(path) for path in tile_paths]
        das: list[xr.DataArray] = [da.sel(band=1) for da in das]
        da: xr.DataArray = merge.merge_arrays(das)
        return da

    def fetch(
        self,
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
    ) -> None:
        """Download MERIT Hydro tiles intersecting a bbox by streaming tars.

        The function downloads only the GeoTIFFs needed for a single MERIT variable
        by streaming the remote 30x30-degree tar packages, without saving the tars.
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

        Raises:
            ValueError: If inputs are invalid or auth variables are missing.
            RuntimeError: If the HTTP client dependency is not available.
            requests.RequestException: If repeated HTTP errors occur.
            tarfile.ReadError: If tar parsing repeatedly fails.
        """
        if self.is_ready:
            return self

        username = os.getenv("MERIT_USERNAME")
        password = os.getenv("MERIT_PASSWORD")
        if not username or not password:
            raise ValueError(
                "Authentication required: set MERIT_USERNAME and MERIT_PASSWORD in environment."
            )

        tiles = self._tiles_for_bbox(xmin, xmax, ymin, ymax)

        # Helper to get missing marker path for a tile name
        def missing_marker_path(tile_name: str) -> Path:
            return self.root / f"{tile_name}.missing.txt"

        # Prepare HTTP session
        if session is None:
            if requests is None:
                raise RuntimeError(
                    "requests is required for downloading but is not available."
                )
            session = requests.Session()
        auth = HTTPBasicAuth(username, password)

        groups = self._group_tiles_by_package(tiles)
        results: list[Path] = []
        missing_markers: list[Path] = []
        for package_name, coords in groups.items():
            package_url = self._package_url(package_name, base_url=url)

            # Determine which tiles in this package still need action
            needed_names: set[str] = set()
            for lat_ll, lon_ll in coords:
                tname = self._compose_tile_filename(lat_ll, lon_ll)
                tif_path = self.root / tname
                if tif_path.exists():
                    results.append(tif_path)
                    continue
                if missing_marker_path(tname).exists():
                    # Previously confirmed missing; skip re-checking
                    continue
                needed_names.add(tname)

            # If nothing to do for this package, skip any network I/O
            if not needed_names:
                continue

            # HEAD to detect ocean/no-data packages (404) → skip silently
            try:
                head_resp = session.head(
                    package_url,
                    auth=auth,
                    allow_redirects=True,
                    timeout=request_timeout_s,
                )
            except Exception as exc:
                # Retry transient head issues
                if attempts <= 1:
                    raise
                retried = 1
                while retried < attempts:
                    time.sleep(min(2.0 * retried, 5.0))
                    try:
                        head_resp = session.head(
                            url,
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
                # Mark all remaining needed tiles in this package as missing
                for tname in needed_names:
                    mm = missing_marker_path(tname)
                    mm.write_text(
                        "MERIT Hydro: tile not provided (ocean/no-data).\n",
                        encoding="utf-8",
                    )
                    missing_markers.append(mm)
                continue  # package does not exist (ocean)
            if head_resp.status_code in (401, 403):
                raise requests.RequestException(
                    f"Unauthorized: HTTP {head_resp.status_code} for {package_url}"
                )
            if head_resp.status_code >= 400:
                raise requests.RequestException(
                    f"HEAD error: HTTP {head_resp.status_code} for {package_url}"
                )

            content_length = int(head_resp.headers.get("Content-Length", 0))

            # Attempt to stream the tar and extract required tiles
            found_names: set[str] = set()
            for attempt in range(1, attempts + 1):
                try:
                    resp = session.get(
                        package_url, auth=auth, stream=True, timeout=request_timeout_s
                    )
                except Exception as exc:
                    if attempt >= attempts:
                        raise
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue

                if resp.status_code in (401, 403):
                    resp.close()
                    raise requests.RequestException(
                        f"Unauthorized: HTTP {resp.status_code} for {package_url}"
                    )
                if resp.status_code >= 400:
                    resp.close()
                    if attempt >= attempts:
                        raise requests.RequestException(
                            f"GET error: HTTP {resp.status_code} for {package_url}"
                        )
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue

                try:
                    # Ensure urllib3 decodes any transfer-encoding for a clean tar stream
                    try:
                        resp.raw.decode_content = True  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    # Stream the tar and extract only needed members sequentially
                    print(
                        "Downloading and extracting from:",
                        package_url,
                        "may take a while.. Don't worry, the files are cached for next time.",
                    )
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
                                    out_file.parent.mkdir(parents=True, exist_ok=True)
                                    with out_file.open("wb") as fout:
                                        while True:
                                            chunk = ex.read(1024 * 1024)
                                            if not chunk:
                                                break
                                            fout.write(chunk)
                                    found_names.add(mname)
                                    results.append(out_file)
                                    if found_names == needed_names:
                                        print(
                                            f"All {len(needed_names)} needed tiles extracted from {package_name}. Stopping download early."
                                        )
                                        break
                except tarfile.ReadError:
                    if attempt >= attempts:
                        raise
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue
                finally:
                    try:
                        resp.close()
                    except Exception:
                        pass

                # If we reach here without raising, either all found or some tiles were not present.
                # Both are acceptable; we do not retry just for missing tiles.
                break

            # Create missing markers for tiles that were not found in an existing package
            for tname in needed_names - found_names:
                mm = missing_marker_path(tname)
                mm.write_text(
                    "MERIT Hydro: tile not present in package (likely ocean).\n",
                    encoding="utf-8",
                )
                missing_markers.append(mm)

        if not results:
            raise ValueError(
                "No MERIT Hydro tiles were found or downloaded for the requested bbox and variable."
            )

        da: xr.DataArray = self._merge_merit_tiles(results)

        if "_FillValue" in da.attrs:
            assert da.attrs["_FillValue"] == source_nodata, (
                f"Expected source _FillValue {source_nodata}, got {da.attrs['_FillValue']}"
            )
        else:
            da.attrs["_FillValue"] = source_nodata

        da: xr.DataArray = da.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
        da: xr.DataArray = convert_nodata(da, target_nodata)

        write_zarr(da, self.path, crs=da.rio.crs)

        # Clean up individual tile files and missing markers
        for result in results:
            result.unlink()

        for mm in missing_markers:
            mm.unlink()

        return self


class MeritHydroDir(MeritHydro):
    """Dataset adapter for MERIT Hydro flow direction.

    Args:
        MeritHydro: Base class for MERIT Hydro datasets.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter for flow direction data.

        Args:
            *args: Positional arguments passed to the base class.
            **kwargs: Keyword arguments passed to the base class.

        """
        super().__init__(variable="dir", *args, **kwargs)

    def fetch(self, *args: Any, **kwargs: Any) -> None:
        """Process and download flow direction data with specific fill value.

        Args:
            *args: Positional arguments passed to the base class fetcher.
            **kwargs: Keyword arguments passed to the base class fetcher.

        """
        return super().fetch(*args, **kwargs, source_nodata=247, target_nodata=247)


class MeritHydroElv(MeritHydro):
    """Dataset adapter for MERIT Hydro elevation.

    Args:
        MeritHydro: Base class for MERIT Hydro datasets.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter for elevation data.

        Args:
            *args: Positional arguments passed to the base class.
            **kwargs: Keyword arguments passed to the base class.

        """
        super().__init__(variable="elv", *args, **kwargs)

    def fetch(self, *args: Any, **kwargs: Any) -> None:
        """Process and download elevation data with specific fill value.

        Args:
            *args: Positional arguments passed to the base class fetcher.
            **kwargs: Keyword arguments passed to the base class fetcher.

        """
        return super().fetch(
            *args, **kwargs, source_nodata=-9999.0, target_nodata=np.nan
        )
