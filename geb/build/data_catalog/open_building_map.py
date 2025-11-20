"""Utilities to download OpenBuildingMap tiles for a given bounding box.

This module provides a downloader that downloads and extracts the needed
gpkg tiles from the remote bz2 packages hosted by OpenBuildingMap.
Tiles are quadkey level 6 and downloaded individually.

Notes:
    - FABDEM distributes 10x10-degree tiles as individual ZIP files.
      Tile filenames follow the pattern " building.002232.gpkg.bz2" where the integers
      indicate the quadkey of the tile. See the OpenBuildingMap documentation for details.
    - Coverage spans the global land areas.

"""

import tempfile
import zipfile
from pathlib import Path
from typing import Any

import mercantile
import numpy as np
import rioxarray as rxr
import xarray as xr
from pyquadkey2 import quadkey
from rioxarray import merge
from tqdm import tqdm

from geb.workflows.io import fetch_and_save, open_zarr, to_zarr

from .base import Adapter


class OpenBuildingMap(Adapter):
    """Dataset adapter for OpenBuildingMap data."""

    def _quadkeys_for_box(self, xmin, xmax, ymin, ymax, zoom=6) -> None:
        """Gets the open building dataset. First it finds the quadkeys of tile within the model domain. Then it downloads the data and clips it to gdl region included in the domain."""
        quadkeys = []

        # iterate over tiles intersecting the bbox
        for tile in mercantile.tiles(xmin, xmax, ymin, ymax, zoom):
            qk = quadkey.from_tile((tile.x, tile.y), level=zoom)
            quadkeys.append(qk.key)

        pass

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
            gpkg_files: list[str] = [
                f for f in zip_ref.namelist() if f.endswith(".gpkg")
            ]

            extracted_paths: list[Path] = []
            for tif_filename in gpkg_files:
                zip_ref.extract(tif_filename, temp_dir)
                extracted_paths.append(temp_dir / tif_filename)

        return extracted_paths

    def fetch(
        self, url: str, xmin: float, xmax: float, ymin: float, ymax: float, prefix: str
    ):
        """Download OpenBuildingMap tiles intersecting a bbox.

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

        tiles: list = self._quadkeys_for_box(xmin, xmax, ymin, ymax)

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir: Path = Path(temp_dir_str)
            results: list[Path] = []

            for tile in tqdm(tiles, desc="Downloading OpenBuildingMap tiles"):
                tile_filename = f"OpenBuildingMap_tile_{tile}.gpkg"
                tile_url: str = f"{url}/{tile_filename}"

                tif_paths: list[Path] = self._download_and_extract_tile(
                    tile_url, temp_dir, tile_filename, xmin, xmax, ymin, ymax
                )
                # Add all extracted TIF files (already filtered during extraction)
                results.extend(tif_paths)

            if not results:
                raise RuntimeError("No OpenBuildingMap tiles could be downloaded.")

            results

        return self
