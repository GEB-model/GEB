"""Utilities to download DeltaDTM tiles for a given bounding box.

This module provides a downloader that downloads and extracts the needed
GeoTIFF tiles from the remote ZIP packages hosted by DeltaDTM.
Tiles are 10x10-degree and downloaded individually.

Notes:
    - DeltaDTM distributes 10x10-degree tiles as individual TIFF files zipped into ZIP archives per continent.
      Tile filenames follow the pattern "DeltaDTM_v1_1_N00W000.tif" where the coordinates
      indicate the bounding box of the tile. See the DeltaDTM documentation for details.
    - Coverage spans the global land areas.

"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import xarray as xr
from rioxarray import merge

from geb.workflows.io import fetch_and_save
from geb.workflows.raster import convert_nodata

from .base import Adapter

available_continents = {
    "Africa.zip": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/22ffa027-184b-4f67-9979-c182f3dfb1ab",
    "Antarctica.zip": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/ca957a40-34fa-41eb-b101-e45d1ccbd890",
    "Asia.zip": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/672eba4c-1334-44c6-8119-8879ded25912",
    "Europe.zip": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/cb0b8ee3-b018-4828-a74e-2fb05020b1b6",
    "North_America.zip": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/037664c6-1494-4889-9689-a56570728320",
    "Oceania.zip": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/de972de1-26bd-4303-afdf-21a90a232cff",
    "Seven_seas_(open_ocean).zip": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/fe986ba6-3db9-40e2-8a49-0fcdb341244a",
    "South_America.zip": "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/db980f00-63cd-4a07-a4df-55ab06510594",
}


class DeltaDTM(Adapter):
    """Downloader for DeltaDTM tiles.

    Downloads and extracts the needed GeoTIFF tiles from the remote ZIP packages
    hosted by DeltaDTM for a given bounding box.

    Attributes:
        cache_dir (Path): Directory to cache downloaded tiles.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter for DeltaDTM.

        Args:
            *args: Additional positional arguments passed to the base Adapter class.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)

    def get_tiles_in_model_bounds(
        self, xmin: float, xmax: float, ymin: float, ymax: float
    ) -> tuple[list[str], list[str]]:
        """Get the DeltaDTM tiles that intersect with the model bounds. This function uses the DeltaDTM tiles geopackage.

        Args:
            xmin: Minimum x-coordinate (longitude) of the model bounds.
            xmax: Maximum x-coordinate (longitude) of the model bounds.
            ymin: Minimum y-coordinate (latitude) of the model bounds.
            ymax: Maximum y-coordinate (latitude) of the model bounds.
        Returns:
            A tuple containing:
                - A list of tile filenames that intersect with the model bounds.
                - A list of continent ZIP filenames to download.
        Raises:
            RuntimeError: If the DeltaDTM tiles geopackage cannot be downloaded.
        """
        # download the DeltaDTM tiles geopackage
        url_delta_dtm_tiles = "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/60a69899-2e67-4f9f-8761-3b57094acd12"
        os.makedirs(self.root, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir: Path = Path(temp_dir_str)
            filepath = temp_dir / "delta_dtm_tiles.gpkg"
            success = fetch_and_save(
                url=url_delta_dtm_tiles,
                file_path=filepath,
            )
            if not success:
                raise RuntimeError("Failed to download DeltaDTM tiles geopackage.")
            # load the geopackage
            gdf_tiles = gpd.read_file(filepath)
        # get the tiles that intersect with the model bounds
        tiles_in_bounds = gdf_tiles.cx[xmin:xmax, ymin:ymax]
        tile_names = tiles_in_bounds["tile"].tolist()

        # continents(s) to download tiles for
        continents_to_download = tiles_in_bounds["zipfile"].unique().tolist()

        return tile_names, continents_to_download

    def download_deltadtm(self, continents_to_download: list[str]) -> None:
        """Download DeltaDTM tiles for the specified continents.

        Args:
            continents_to_download: List of continent ZIP filenames to download.
        Raises:
            RuntimeError: If downloading any of the continent ZIP files fails.
        """
        for continent in continents_to_download:
            url = available_continents[continent]
            zip_path = self._construct_filepath(continent[:-4])
            success = fetch_and_save(
                url=url,
                file_path=zip_path,
            )
            if not success:
                raise RuntimeError(
                    f"Failed to download DeltaDTM continent ZIP: {continent}"
                )

    def unpack_and_merge_tiles(
        self, continents_to_download: list[str], tile_names: list[str]
    ) -> xr.DataArray:
        """Unpack and merge DeltaDTM tiles into a single dataarray.

        Args:
            continents_to_download: List of continent ZIP filenames to extract from.
            tile_names: List of tile filenames to unpack and merge.

        Returns:
            Merged dataarray of the specified tiles.
        """
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir: Path = Path(temp_dir_str)
            extracted_paths = self._unpack_tiles(
                continents_to_download, tile_names, temp_dir
            )
            da = self._merge_tiles(extracted_paths)

        da = convert_nodata(da, np.nan)
        return da

    def _merge_tiles(self, tile_paths: list[Path]) -> xr.DataArray:
        """Merge extracted DeltaDTM tiles into a single xarray DataArray.

        Args:
            tile_paths: List of paths to the extracted tile files.

        Returns:
            Merged dataset of the specified tiles.
        """
        das: list[xr.DataArray] = [rxr.open_rasterio(path) for path in tile_paths]  # ty: ignore[invalid-assignment]
        das = [da.sel(band=1) for da in das]
        da: xr.DataArray = merge.merge_arrays(das)
        return da

    def _unpack_tiles(
        self, continents_to_download: list[str], tile_names: list[str], temp_dir: Path
    ) -> list[Path]:
        """Unpack and merge DeltaDTM tiles into a single xarray Dataset.

        Args:
            continents_to_download: List of continent ZIP filenames to extract from.
            tile_names: List of tile filenames to unpack and merge.
            temp_dir: Temporary directory to extract tiles into.
        Returns:
            Merged dataset of the specified tiles.
        """
        extracted_paths: list[Path] = []
        for continent in continents_to_download:
            zip_path = self._construct_filepath(continent[:-4])
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for tile_name in tile_names:
                    if tile_name in zip_ref.namelist():
                        zip_ref.extract(tile_name, path=temp_dir)
                        extracted_paths.append(temp_dir / tile_name)
        return extracted_paths

    def _construct_filepath(self, continent: str) -> Path:
        """Construct the file path for a given continent ZIP file.

        Args:
            continent: The continent ZIP filename.
        Returns:
            The constructed file path for the continent ZIP file.

        """
        return Path(str(self.path).format(continent))

    def fetch(
        self, xmin: float, xmax: float, ymin: float, ymax: float, url: str | None = None
    ) -> DeltaDTM:
        """Fetch DeltaDTM tiles for the specified bounding box.

        Args:
            xmin: Minimum x-coordinate (longitude) of the bounding box.
            xmax: Maximum x-coordinate (longitude) of the bounding box.
            ymin: Minimum y-coordinate (latitude) of the bounding box.
            ymax: Maximum y-coordinate (latitude) of the bounding box.
            url: URL to download DeltaDTM data from. Defaults to None.
        Returns:
            The DeltaDTM adapter instance with the downloaded data.
        """
        self.tile_names, self.continents_to_download = self.get_tiles_in_model_bounds(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
        )
        self.download_deltadtm(self.continents_to_download)
        return self

    def read(self) -> xr.DataArray:
        """Read and unpack the downloaded DeltaDTM data.

        Returns:
            The downloaded DeltaDTM data.
        """
        da = self.unpack_and_merge_tiles(self.continents_to_download, self.tile_names)
        return da
