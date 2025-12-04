"""GEBCO data adapter for downloading and processing GEBCO data."""

import zipfile
from pathlib import Path
from typing import Any

import rioxarray as rxr

from geb.workflows.io import fetch_and_save, write_zarr

from .base import Adapter


class GEBCO(Adapter):
    """The GADM adapter for downloading and processing GAGEBCODM data.

    Args:
        Adapter: The base Adapter class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the GEBCO adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> Path:
        """Process GEBCO zip file to extract, combine tiles and convert to zarr.

        Args:
            url: The URL to download the GEBCO zip file from.

        Returns:
            The instance of the GEBCO Adapter.
        """
        if not self.is_ready:
            download_path: Path = self.root / "gebco.zip"
            fetch_and_save(url=url, file_path=download_path)

            print("Extracting tiles from zip file...")
            with zipfile.ZipFile(file=download_path, mode="r") as zip_ref:
                files = zip_ref.namelist()
                tile_paths = [f for f in files if f.endswith(".tif")]
                for tile in tile_paths:
                    zip_ref.extract(tile, self.root)

            print("Merging tiles into single DataArray. This may take a while...")
            das: list[xr.DataArray] = [
                rxr.open_rasterio(self.root / path) for path in tile_paths
            ]
            das: list[xr.DataArray] = [da.sel(band=1) for da in das]
            da: xr.DataArray = rxr.merge.merge_arrays(das)

            da.attrs = {
                k: v
                for k, v in da.attrs.items()
                if k
                in (
                    "_FillValue",
                    "scale_factor",
                    "add_offset",
                    "units",
                    "long_name",
                    "license",
                )
            }

            print("Saving merged DataArray to zarr...")
            write_zarr(da, self.path, crs=da.rio.crs)

            print("Cleaning up temporary files...")
            download_path.unlink()  # remove zip file
            for path in tile_paths:
                (self.root / path).unlink()  # remove extracted .tif files

        return self
