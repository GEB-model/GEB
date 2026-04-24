"""Utilities to download DeltaDTM tiles for a given region.

This module provides a downloader that downloads and extracts the needed
GeoTIFF tiles from the remote ZIP packages hosted by DeltaDTM.
Tiles are 10x10-degree and downloaded individually.

Notes:
    - DeltaDTM distributes 10x10-degree tiles as individual TIFF files zipped into ZIP archives per continent.
      Tile filenames follow the pattern "DeltaDTM_v1_1_N00W000.tif" where the coordinates
      indicate the bounding box of the tile. See the DeltaDTM documentation for details.
    - Coverage spans the global land areas.

"""

import tempfile
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import xarray as xr
from shapely.geometry.base import BaseGeometry

from geb.workflows.io import fetch_and_save
from geb.workflows.raster import convert_nodata

from .base import Adapter

available_continents: dict[str, str] = {
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

    def get_tiles_for_mask(
        self, mask: BaseGeometry
    ) -> tuple[gpd.GeoDataFrame, list[str]]:
        """Get the DeltaDTM tiles that intersect with the mask, with a buffer. This function uses the DeltaDTM tiles geopackage.

        Args:
            mask: The geometry used to filter intersecting tiles.

        Returns:
            A tuple containing:
                - A GeoDataFrame of the tiles that intersect with the mask.
                - A list of continent ZIP filenames to download.

        Raises:
            RuntimeError: If the DeltaDTM tiles geopackage cannot be downloaded.
        """
        # download the DeltaDTM tiles geopackage
        url_delta_dtm_tiles = "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/60a69899-2e67-4f9f-8761-3b57094acd12"
        self.root.mkdir(parents=True, exist_ok=True)
        filepath: Path = self.root / "delta_dtm_tiles.gpkg"
        if not filepath.exists():
            success = fetch_and_save(
                url=url_delta_dtm_tiles,
                file_path=filepath,
            )
            if not success:
                raise RuntimeError("Failed to download DeltaDTM tiles geopackage.")

        # load the geopackage
        gdf_tiles: gpd.GeoDataFrame = gpd.read_file(filepath)

        # get the tiles that intersect with the model bounds, with a buffer (0.4 degrees)
        buffered_mask = mask.buffer(0.4)

        xmin, ymin, xmax, ymax = buffered_mask.bounds
        tiles_in_bounds = gdf_tiles.cx[xmin:xmax, ymin:ymax]

        # continents(s) to download tiles for
        continents_to_download: list[str] = tiles_in_bounds["zipfile"].unique().tolist()

        return tiles_in_bounds, continents_to_download

    def fetch(self, mask: BaseGeometry, url: str | None = None) -> DeltaDTM:
        """Fetch DeltaDTM tiles for the specified mask.

        Args:
            mask: The geometry used to filter intersecting tiles.
            url: URL to download DeltaDTM data from. Defaults to None.

        Returns:
            The DeltaDTM adapter instance with the downloaded data.

        Raises:
            RuntimeError: If the DeltaDTM tiles cannot be downloaded or extracted.
        """
        _, self.continents_to_download = self.get_tiles_for_mask(mask=mask)

        for zip_name in self.continents_to_download:
            # Check if we already unpacked this zip
            placeholder = self.root / f".{zip_name}.unpacked"
            if placeholder.exists():
                continue

            url: str = available_continents[zip_name]

            # Download and extract the zip file using a temporary directory for cleanup
            with tempfile.TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                temp_zip_path = tmpdir / zip_name

                success = fetch_and_save(url=url, file_path=temp_zip_path)
                if not success:
                    raise RuntimeError(f"Failed to download DeltaDTM zip: {zip_name}")

                # Unpack the zip file
                with zipfile.ZipFile(temp_zip_path, "r") as zf:
                    zf.extractall(path=self.root)

            # Create placeholder to remember it was unpacked
            placeholder.touch()

        return self

    def read(self, mask: BaseGeometry) -> xr.DataArray:
        """Read and merge the downloaded DeltaDTM tiles.

        Args:
            mask: The geometry used to filter intersecting tiles.

        Returns:
            The merged DeltaDTM data.
        """
        tiles_in_bounds, _ = self.get_tiles_for_mask(mask=mask)
        tile_names: list[str] = tiles_in_bounds["tile"].tolist()
        tile_paths: list[Path] = [self.root / tile_name for tile_name in tile_names]
        da: xr.DataArray = self._merge_tiles(tile_paths)
        da = convert_nodata(da, np.nan)
        return da

    def _merge_tiles(self, tile_paths: list[Path]) -> xr.DataArray:
        """Merge DeltaDTM tiles into a single xarray DataArray.

        Args:
            tile_paths: List of paths to the tile files.

        Returns:
            Merged dataset of the specified tiles.
        """
        das: list[xr.DataArray] = [
            rxr.open_rasterio(path, chunks={}) for path in tile_paths
        ]  # ty: ignore[invalid-assignment]
        das: list[xr.DataArray] = [da.sel(band=1) for da in das]
        da = xr.combine_by_coords(
            das,
            fill_value=das[0].rio.nodata,
            combine_attrs="drop_conflicts",
            join="outer",
            compat="broadcast_equals",
            data_vars="all",
        )
        assert isinstance(da, xr.DataArray)
        return da
