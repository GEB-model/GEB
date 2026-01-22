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

import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rioxarray as rxr
import xarray as xr
from rioxarray import merge
import os
import geopandas as gpd
from tqdm import tqdm
from geb.workflows.io import fetch_and_save, read_zarr, write_zarr
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

    def get_tiles_in_model_bounds(self, xmin: float, xmax: float, ymin: float, ymax: float):
        # download the DeltaDTM tiles geopackage
        url_delta_dtm_tiles = "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/60a69899-2e67-4f9f-8761-3b57094acd12"
        os.makedirs(self.root, exist_ok=True)
        success = fetch_and_save(
            url = url_delta_dtm_tiles,
            file_path = self.root / "delta_dtm_tiles.gpkg",
        )
        if not success:
            raise RuntimeError("Failed to download DeltaDTM tiles geopackage.")
        # load the geopackage
        gdf_tiles = gpd.read_file(self.root / "delta_dtm_tiles.gpkg")
        # get the tiles that intersect with the model bounds
        tiles_in_bounds = gdf_tiles.cx[xmin:xmax, ymin:ymax]
        tile_names = tiles_in_bounds["tile"].tolist()

        # continents(s) to download tiles for
        continents_to_download = tiles_in_bounds["zipfile"].unique().tolist()
    
        return tile_names, continents_to_download


    def fetch(self, xmin: float, xmax: float, ymin: float, ymax: float, url: str = None):
        tile_names, continents_to_download = self.get_tiles_in_model_bounds(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        return self

    def read(self):
        return None

