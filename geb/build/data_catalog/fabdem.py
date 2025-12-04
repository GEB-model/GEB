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
import rioxarray as rxr
import xarray as xr
from rioxarray import merge
from tqdm import tqdm

from geb.workflows.io import fetch_and_save, read_zarr, write_zarr
from geb.workflows.raster import convert_nodata

from .base import Adapter

# FABDEM is only available over land, so not all tiles exist. This is a set of
# all available 10x10-degree tile names which we use to check if the tile
# should exist before attempting download.
available_tiles: set[str] = {
    "N00E000-N10E010",
    "N00E010-N10E020",
    "N00E020-N10E030",
    "N00E030-N10E040",
    "N00E040-N10E050",
    "N00E050-N10E060",
    "N00E070-N10E080",
    "N00E080-N10E090",
    "N00E090-N10E100",
    "N00E100-N10E110",
    "N00E110-N10E120",
    "N00E120-N10E130",
    "N00E130-N10E140",
    "N00E140-N10E150",
    "N00E150-N10E160",
    "N00E160-N10E170",
    "N00E170-N10W180",
    "N00W010-N10E000",
    "N00W020-N10W010",
    "N00W050-N10W040",
    "N00W060-N10W050",
    "N00W070-N10W060",
    "N00W080-N10W070",
    "N00W090-N10W080",
    "N00W100-N10W090",
    "N00W160-N10W150",
    "N00W170-N10W160",
    "N00W180-N10W170",
    "N10E000-N20E010",
    "N10E010-N20E020",
    "N10E020-N20E030",
    "N10E030-N20E040",
    "N10E040-N20E050",
    "N10E050-N20E060",
    "N10E070-N20E080",
    "N10E080-N20E090",
    "N10E090-N20E100",
    "N10E100-N20E110",
    "N10E110-N20E120",
    "N10E120-N20E130",
    "N10E130-N20E140",
    "N10E140-N20E150",
    "N10E160-N20E170",
    "N10E170-N20W180",
    "N10W010-N20E000",
    "N10W020-N20W010",
    "N10W030-N20W020",
    "N10W060-N20W050",
    "N10W070-N20W060",
    "N10W080-N20W070",
    "N10W090-N20W080",
    "N10W100-N20W090",
    "N10W110-N20W100",
    "N10W120-N20W110",
    "N10W160-N20W150",
    "N10W170-N20W160",
    "N20E000-N30E010",
    "N20E010-N30E020",
    "N20E020-N30E030",
    "N20E030-N30E040",
    "N20E040-N30E050",
    "N20E050-N30E060",
    "N20E060-N30E070",
    "N20E070-N30E080",
    "N20E080-N30E090",
    "N20E090-N30E100",
    "N20E100-N30E110",
    "N20E110-N30E120",
    "N20E120-N30E130",
    "N20E130-N30E140",
    "N20E140-N30E150",
    "N20E150-N30E160",
    "N20W010-N30E000",
    "N20W020-N30W010",
    "N20W080-N30W070",
    "N20W090-N30W080",
    "N20W100-N30W090",
    "N20W110-N30W100",
    "N20W120-N30W110",
    "N20W160-N30W150",
    "N20W170-N30W160",
    "N20W180-N30W170",
    "N30E000-N40E010",
    "N30E010-N40E020",
    "N30E020-N40E030",
    "N30E030-N40E040",
    "N30E040-N40E050",
    "N30E050-N40E060",
    "N30E060-N40E070",
    "N30E070-N40E080",
    "N30E080-N40E090",
    "N30E090-N40E100",
    "N30E100-N40E110",
    "N30E110-N40E120",
    "N30E120-N40E130",
    "N30E130-N40E140",
    "N30E140-N40E150",
    "N30W010-N40E000",
    "N30W020-N40W010",
    "N30W030-N40W020",
    "N30W040-N40W030",
    "N30W070-N40W060",
    "N30W080-N40W070",
    "N30W090-N40W080",
    "N30W100-N40W090",
    "N30W110-N40W100",
    "N30W120-N40W110",
    "N30W130-N40W120",
    "N40E000-N50E010",
    "N40E010-N50E020",
    "N40E020-N50E030",
    "N40E030-N50E040",
    "N40E040-N50E050",
    "N40E050-N50E060",
    "N40E060-N50E070",
    "N40E070-N50E080",
    "N40E080-N50E090",
    "N40E090-N50E100",
    "N40E100-N50E110",
    "N40E110-N50E120",
    "N40E120-N50E130",
    "N40E130-N50E140",
    "N40E140-N50E150",
    "N40E150-N50E160",
    "N40W010-N50E000",
    "N40W060-N50W050",
    "N40W070-N50W060",
    "N40W080-N50W070",
    "N40W090-N50W080",
    "N40W100-N50W090",
    "N40W110-N50W100",
    "N40W120-N50W110",
    "N40W130-N50W120",
    "N50E000-N60E010",
    "N50E010-N60E020",
    "N50E020-N60E030",
    "N50E030-N60E040",
    "N50E040-N60E050",
    "N50E050-N60E060",
    "N50E060-N60E070",
    "N50E070-N60E080",
    "N50E080-N60E090",
    "N50E090-N60E100",
    "N50E100-N60E110",
    "N50E110-N60E120",
    "N50E120-N60E130",
    "N50E130-N60E140",
    "N50E140-N60E150",
    "N50E150-N60E160",
    "N50E160-N60E170",
    "N50E170-N60W180",
    "N50W010-N60E000",
    "N50W020-N60W010",
    "N50W050-N60W040",
    "N50W060-N60W050",
    "N50W070-N60W060",
    "N50W080-N60W070",
    "N50W090-N60W080",
    "N50W100-N60W090",
    "N50W110-N60W100",
    "N50W120-N60W110",
    "N50W130-N60W120",
    "N50W140-N60W130",
    "N50W150-N60W140",
    "N50W160-N60W150",
    "N50W170-N60W160",
    "N50W180-N60W170",
    "N60E000-N70E010",
    "N60E010-N70E020",
    "N60E020-N70E030",
    "N60E030-N70E040",
    "N60E040-N70E050",
    "N60E050-N70E060",
    "N60E060-N70E070",
    "N60E070-N70E080",
    "N60E080-N70E090",
    "N60E090-N70E100",
    "N60E100-N70E110",
    "N60E110-N70E120",
    "N60E120-N70E130",
    "N60E130-N70E140",
    "N60E140-N70E150",
    "N60E150-N70E160",
    "N60E160-N70E170",
    "N60E170-N70W180",
    "N60W010-N70E000",
    "N60W020-N70W010",
    "N60W030-N70W020",
    "N60W040-N70W030",
    "N60W050-N70W040",
    "N60W060-N70W050",
    "N60W070-N70W060",
    "N60W080-N70W070",
    "N60W090-N70W080",
    "N60W100-N70W090",
    "N60W110-N70W100",
    "N60W120-N70W110",
    "N60W130-N70W120",
    "N60W140-N70W130",
    "N60W150-N70W140",
    "N60W160-N70W150",
    "N60W170-N70W160",
    "N60W180-N70W170",
    "N70E010-N80E020",
    "N70E020-N80E030",
    "N70E030-N80E040",
    "N70E040-N80E050",
    "N70E050-N80E060",
    "N70E060-N80E070",
    "N70E070-N80E080",
    "N70E080-N80E090",
    "N70E090-N80E100",
    "N70E100-N80E110",
    "N70E110-N80E120",
    "N70E120-N80E130",
    "N70E130-N80E140",
    "N70E140-N80E150",
    "N70E150-N80E160",
    "N70E160-N80E170",
    "N70E170-N80W180",
    "N70W010-N80E000",
    "N70W020-N80W010",
    "N70W030-N80W020",
    "N70W040-N80W030",
    "N70W050-N80W040",
    "N70W060-N80W050",
    "N70W070-N80W060",
    "N70W080-N80W070",
    "N70W090-N80W080",
    "N70W100-N80W090",
    "N70W110-N80W100",
    "N70W120-N80W110",
    "N70W130-N80W120",
    "N70W140-N80W130",
    "N70W150-N80W140",
    "N70W160-N80W150",
    "N70W170-N80W160",
    "N70W180-N80W170",
    "S10E000-N00E010",
    "S10E010-N00E020",
    "S10E020-N00E030",
    "S10E030-N00E040",
    "S10E040-N00E050",
    "S10E050-N00E060",
    "S10E070-N00E080",
    "S10E090-N00E100",
    "S10E100-N00E110",
    "S10E110-N00E120",
    "S10E120-N00E130",
    "S10E130-N00E140",
    "S10E140-N00E150",
    "S10E150-N00E160",
    "S10E160-N00E170",
    "S10E170-N00W180",
    "S10W020-N00W010",
    "S10W040-N00W030",
    "S10W050-N00W040",
    "S10W060-N00W050",
    "S10W070-N00W060",
    "S10W080-N00W070",
    "S10W090-N00W080",
    "S10W100-N00W090",
    "S10W140-N00W130",
    "S10W150-N00W140",
    "S10W160-N00W150",
    "S10W170-N00W160",
    "S10W180-N00W170",
    "S20E010-S10E020",
    "S20E020-S10E030",
    "S20E030-S10E040",
    "S20E040-S10E050",
    "S20E050-S10E060",
    "S20E060-S10E070",
    "S20E090-S10E100",
    "S20E100-S10E110",
    "S20E110-S10E120",
    "S20E120-S10E130",
    "S20E130-S10E140",
    "S20E140-S10E150",
    "S20E150-S10E160",
    "S20E160-S10E170",
    "S20E170-S10W180",
    "S20W010-S10E000",
    "S20W040-S10W030",
    "S20W050-S10W040",
    "S20W060-S10W050",
    "S20W070-S10W060",
    "S20W080-S10W070",
    "S20W140-S10W130",
    "S20W150-S10W140",
    "S20W160-S10W150",
    "S20W170-S10W160",
    "S20W180-S10W170",
    "S30E010-S20E020",
    "S30E020-S20E030",
    "S30E030-S20E040",
    "S30E040-S20E050",
    "S30E050-S20E060",
    "S30E110-S20E120",
    "S30E120-S20E130",
    "S30E130-S20E140",
    "S30E140-S20E150",
    "S30E150-S20E160",
    "S30E160-S20E170",
    "S30E170-S20W180",
    "S30W030-S20W020",
    "S30W050-S20W040",
    "S30W060-S20W050",
    "S30W070-S20W060",
    "S30W080-S20W070",
    "S30W090-S20W080",
    "S30W110-S20W100",
    "S30W130-S20W120",
    "S30W140-S20W130",
    "S30W150-S20W140",
    "S30W160-S20W150",
    "S30W180-S20W170",
    "S40E010-S30E020",
    "S40E020-S30E030",
    "S40E030-S30E040",
    "S40E070-S30E080",
    "S40E110-S30E120",
    "S40E120-S30E130",
    "S40E130-S30E140",
    "S40E140-S30E150",
    "S40E150-S30E160",
    "S40E170-S30W180",
    "S40W020-S30W010",
    "S40W060-S30W050",
    "S40W070-S30W060",
    "S40W080-S30W070",
    "S40W090-S30W080",
    "S40W180-S30W170",
    "S50E030-S40E040",
    "S50E050-S40E060",
    "S50E060-S40E070",
    "S50E070-S40E080",
    "S50E140-S40E150",
    "S50E160-S40E170",
    "S50E170-S40W180",
    "S50W010-S40E000",
    "S50W020-S40W010",
    "S50W070-S40W060",
    "S50W080-S40W070",
    "S50W180-S40W170",
    "S60E000-S50E010",
    "S60E060-S50E070",
    "S60E070-S50E080",
    "S60E150-S50E160",
    "S60E160-S50E170",
    "S60W030-S50W020",
    "S60W040-S50W030",
    "S60W050-S50W040",
    "S60W060-S50W050",
    "S60W070-S50W060",
    "S60W080-S50W070",
}


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

        lat_max: int = lat_min + 10
        lon_max: int = lon_min + 10

        ns_min: str = "S" if lat_min < 0 else "N"
        ew_min: str = "W" if lon_min < 0 else "E"
        ns_max: str = "S" if lat_max < 0 else "N"
        ew_max: str = "W" if lon_max < 0 else "E"

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
        base_name: str = tif_filename[:-4]  # Remove .tif extension

        ns_lat: str = base_name[0]  # N or S
        lat_str: str = base_name[1:3]  # 00
        ew_lon: str = base_name[3]  # E or W
        lon_str: str = base_name[4:7]  # 000

        lat_val: int = int(lat_str)
        lon_val: int = int(lon_str)

        # Convert to actual coordinates
        if ns_lat == "S":
            lat_min: int = -lat_val - 1
            lat_max: int = -lat_val
        else:  # N
            lat_min: int = lat_val
            lat_max: int = lat_val + 1

        if ew_lon == "W":
            lon_min: int = -lon_val - 1
            lon_max: int = -lon_val
        else:  # E
            lon_min: int = lon_val
            lon_max: int = lon_val + 1

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
            RuntimeError: If download or extraction fails after all retries.
        """
        # Extract tile name from filename (remove "_FABDEM_V1-2.zip")
        tile_name = tile_filename.replace("_FABDEM_V1-2.zip", "")

        # Check if tile is in available tiles (only land tiles exist)
        if tile_name not in available_tiles:
            return []

        # Tile is available, so download with retry
        zip_path: Path = temp_dir / tile_filename
        success: bool = fetch_and_save(
            tile_url,
            zip_path,
            delay_seconds=1,
            verbose=False,
            show_progress=True,
            double_delay=True,
            max_retries=17,  # will be total of ~day
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
        das: list[xr.DataArray] = [rxr.open_rasterio(path) for path in tile_paths]  # ty: ignore[invalid-assignment]
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
        assert self.root is not None, (
            "Root directory must be set before calling get_filepath"
        )
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

            write_zarr(da, filepath, crs=da.rio.crs)

        return self

    def read(self, prefix: str) -> xr.DataArray:
        """Read the FABDEM data as an xarray DataArray.

        Args:
            prefix: Prefix for the local storage path.

        Returns:
            xarray DataArray with FABDEM data.
        """
        filepath: Path = self.get_filepath(prefix)
        da: xr.DataArray = read_zarr(filepath)
        return da
