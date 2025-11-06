"""File adapter for FAO datasets."""

from __future__ import annotations

import zipfile
from typing import Any

import xarray as xr

from geb.workflows.io import fetch_and_save

from .base import Adapter


class GMIA(Adapter):
    """Adapter for Global Map of Irrigation Areas datasets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Global Map of Irrigation Areas adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> GMIA:
        """Fetch the dataset from the given URL if not already present.

        Args:
            url: The URL to fetch the dataset from.

        Returns:
            The GMIA adapter instance.
        """
        if not self.is_ready:
            download_path = self.path.with_suffix(".zip")
            fetch_and_save(url=url, file_path=download_path)

            # unzip the file
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                files = zip_ref.namelist()
                assert len(files) == 1, "Expected exactly one file in the zip."
                file = files[0]
                zip_ref.extract(file, self.root)

            extracted_file_path = self.root / file
            extracted_file_path.rename(self.path)  # rename to desired filename
            download_path.unlink()  # Remove the downloaded zip file
        return self

    def read(self, **kwargs: Any) -> xr.DataArray:
        """Read the dataset into an xarray DataArray.

        Also sets the appropriate CRS.

        Args:
            **kwargs: Additional keyword arguments to pass to the reader function.

        Returns:
            The dataset as an xarray DataArray with appropriate CRS.
        """
        da = super().read(**kwargs)
        da = da.sel(band=1)
        da = da.rio.write_crs("EPSG:4326")
        return da
