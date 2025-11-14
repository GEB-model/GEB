"""Utilities to download FABDEM tiles for a given bounding box.

This module provides a downloader that downloads and extracts the needed
GeoTIFF tiles from the remote ZIP packages hosted by FABDEM.
Tiles are 10x10-degree and downloaded individually.

Notes:
    - FABDEM distributes 10x10-degree tiles as individual ZIP files.
      Tile filenames follow the pattern "N00E000-N10E010_FABDEM_V1-2.zip" where the coordinates
      indicate the bounding box of the tile. See the FABDEM documentation for details.
    - Coverage spans the global land areas.

"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import requests
import rioxarray as rxr
import xarray as xr
from rioxarray import merge
from tqdm import tqdm

from geb.workflows.io import fetch_and_save, open_zarr, to_zarr
from geb.workflows.raster import convert_nodata

from .base import Adapter


class Fabdem(Adapter):
    """Dataset adapter for FABDEM elevation data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter for FABDEM.

        Args:
            *args: Additional positional arguments passed to the base Adapter class.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)

    def _compose_tile_filename(self, lat_min: int, lon_min: int) -> str:
        """Compose a 10x10-degree tile ZIP filename.

        Args:
            lat_min: Lower latitude of tile (multiple of 10) (degrees).
            lon_min: Lower longitude of tile (multiple of 10) (degrees).

        Returns:
            Tile ZIP filename like "N00E000-N10E010_FABDEM_V1-2.zip".

        Raises:
            ValueError: If lat_min or lon_min are not multiples of 10.
        """
        if lat_min % 10 != 0 or lon_min % 10 != 0:
            raise ValueError("lat_min and lon_min must be multiples of 10.")

        lat_max = lat_min + 10
        lon_max = lon_min + 10

        ns_min = "S" if lat_min < 0 else "N"
        ew_min = "W" if lon_min < 0 else "E"
        ns_max = "S" if lat_max < 0 else "N"
        ew_max = "W" if lon_max < 0 else "E"

        abs_lat_min = abs(lat_min)
        abs_lon_min = abs(lon_min)
        abs_lat_max = abs(lat_max)
        abs_lon_max = abs(lon_max)

        return f"{ns_min}{abs_lat_min:02d}{ew_min}{abs_lon_min:03d}-{ns_max}{abs_lat_max:02d}{ew_max}{abs_lon_max:03d}_FABDEM_V1-2.zip"

    def _tiles_for_bbox(
        self, xmin: float, xmax: float, ymin: float, ymax: float
    ) -> list[tuple[int, int]]:
        """Compute all 10x10-degree lower-left tile coordinates intersecting a bbox.

        Args:
            xmin: Minimum longitude (degrees).
            xmax: Maximum longitude (degrees).
            ymin: Minimum latitude (degrees).
            ymax: Maximum latitude (degrees).

        Returns:
            Sorted list of (lat_min, lon_min) integer tuples on 10-degree grid.

        Raises:
            ValueError: If bbox is invalid.
        """
        if xmax <= xmin:
            raise ValueError("xmax must be greater than xmin.")
        if ymax <= ymin:
            raise ValueError("ymax must be greater than ymin.")

        # Clamp to plausible world bounds
        xmin_c = max(-180.0, xmin)
        xmax_c = min(180.0, xmax)
        ymin_c = max(-90.0, ymin)
        ymax_c = min(90.0, ymax)

        # Align to 10-degree grid (lower-left corners)
        def floor10(v: float) -> int:
            return int(v // 10) * 10

        lat_start = floor10(ymin_c)
        lon_start = floor10(xmin_c)

        tiles: list[tuple[int, int]] = []
        lat = lat_start
        while lat < ymax_c:
            lon = lon_start
            while lon < xmax_c:
                tiles.append((lat, lon))
                lon += 10
            lat += 10
        # Unique and sorted for reproducibility
        tiles = sorted(set(tiles))
        return tiles

    def _tif_intersects_bbox(
        self,
        tif_filename: str,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> bool:
        """Check if a GeoTIFF file in a ZIP intersects with the bounding box.

        Args:
            tif_filename: Name of the TIF file within the ZIP.
            xmin: Minimum longitude of bbox (degrees).
            xmax: Maximum longitude of bbox (degrees).
            ymin: Minimum latitude of bbox (degrees).
            ymax: Maximum latitude of bbox (degrees).

        Returns:
            True if the TIF intersects with the bbox, False otherwise.
        """
        # Parse bounds from filename
        # FABDEM TIF filenames follow pattern like "N00E000.tif" for 1x1 degree cells
        base_name = tif_filename[:-4]  # Remove .tif extension

        ns_lat = base_name[0]  # N or S
        lat_str = base_name[1:3]  # 00
        ew_lon = base_name[3]  # E or W
        lon_str = base_name[4:7]  # 000

        lat_val = int(lat_str)
        lon_val = int(lon_str)

        # Convert to actual coordinates
        if ns_lat == "S":
            lat_min = -lat_val - 1
            lat_max = -lat_val
        else:  # N
            lat_min = lat_val
            lat_max = lat_val + 1

        if ew_lon == "W":
            lon_min = -lon_val - 1
            lon_max = -lon_val
        else:  # E
            lon_min = lon_val
            lon_max = lon_val + 1

        # Check for intersection
        return not (
            lon_max <= xmin or lon_min >= xmax or lat_max <= ymin or lat_min >= ymax
        )

    def _download_and_extract_tile(
        self,
        tile_url: str,
        temp_dir: Path,
        tile_filename: str,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> list[Path]:
        """Download a tile ZIP and extract only GeoTIFF files that intersect with the bbox.

        Args:
            tile_url: URL of the tile ZIP file.
            temp_dir: Temporary directory to extract to.
            tile_filename: Filename of the tile ZIP.
            xmin: Minimum longitude of bbox (degrees).
            xmax: Maximum longitude of bbox (degrees).
            ymin: Minimum latitude of bbox (degrees).
            ymax: Maximum latitude of bbox (degrees).

        Returns:
            List of paths to the extracted GeoTIFF files that intersect with the bbox.

        Raises:
            RuntimeError: If download or extraction fails.
        """
        # first check that url exists. If it does not, return empty list
        # this is because not all tiles exist (e.g., over ocean)
        response = requests.head(tile_url)
        if response.status_code == 404:
            return []
        zip_path: Path = temp_dir / tile_filename
        success: bool = fetch_and_save(
            tile_url, zip_path, verbose=False, show_progress=True
        )
        if not success:
            raise RuntimeError(f"Failed to download {tile_url}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Get all .tif files in the ZIP
            tif_files: list[str] = [f for f in zip_ref.namelist() if f.endswith(".tif")]

            extracted_paths: list[Path] = []
            for tif_filename in tif_files:
                # Check if this TIF intersects with bbox before extracting
                if self._tif_intersects_bbox(tif_filename, xmin, xmax, ymin, ymax):
                    zip_ref.extract(tif_filename, temp_dir)
                    extracted_paths.append(temp_dir / tif_filename)

        return extracted_paths

    def _merge_fabdem_tiles(self, tile_paths: list[Path]) -> xr.DataArray:
        """Load FABDEM tiles into a single xarray DataArray.

        Args:
            tile_paths: List of Paths to GeoTIFF files.

        Returns:
            xarray DataArray with merged tiles.
        """
        das: list[xr.DataArray] = [rxr.open_rasterio(path) for path in tile_paths]
        das = [da.sel(band=1) for da in das]
        da: xr.DataArray = merge.merge_arrays(das)
        return da

    def get_filepath(self, prefix: str) -> Path:
        """Get the local file path for the FABDEM data.

        Args:
            prefix: Prefix for the local storage path.

        Returns:
            Path to the local FABDEM data file.
        """
        filepath: Path = self.root / f"{prefix}_{self.filename}"
        return filepath

    def fetch(
        self, url: str, xmin: float, xmax: float, ymin: float, ymax: float, prefix: str
    ) -> Fabdem:
        """Download FABDEM tiles intersecting a bbox.

        Args:
            xmin: Minimum longitude of area of interest (degrees).
            xmax: Maximum longitude of area of interest (degrees).
            ymin: Minimum latitude of area of interest (degrees).
            ymax: Maximum latitude of area of interest (degrees).
            url: Base URL of the FABDEM server.
            prefix: Prefix for the local storage path.

        Returns:
            The Fabdem instance.

        Raises:
            RuntimeError: If no tiles could be downloaded.
        """
        filepath: Path = self.get_filepath(prefix)
        if (filepath).exists():
            return self

        tiles: list[tuple[int, int]] = self._tiles_for_bbox(xmin, xmax, ymin, ymax)

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir: Path = Path(temp_dir_str)
            results: list[Path] = []

            for lat_min, lon_min in tqdm(tiles, desc="Downloading FABDEM tiles"):
                tile_filename = self._compose_tile_filename(lat_min, lon_min)
                tile_url: str = f"{url}/{tile_filename}"

                tif_paths: list[Path] = self._download_and_extract_tile(
                    tile_url, temp_dir, tile_filename, xmin, xmax, ymin, ymax
                )
                # Add all extracted TIF files (already filtered during extraction)
                results.extend(tif_paths)

            if not results:
                raise RuntimeError("No FABDEM tiles could be downloaded.")

            da: xr.DataArray = self._merge_fabdem_tiles(results)
            da = da.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
            da = convert_nodata(da, np.nan)

            to_zarr(da, filepath, crs=da.rio.crs)

        return self

    def read(self, prefix: str) -> xr.DataArray:
        """Read the FABDEM data as an xarray DataArray.

        Args:
            prefix: Prefix for the local storage path.

        Returns:
            xarray DataArray with FABDEM data.
        """
        filepath: Path = self.get_filepath(prefix)
        da: xr.DataArray = open_zarr(filepath)
        return da
