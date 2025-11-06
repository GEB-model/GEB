"""Adapters for GlobGM datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from geb.workflows.io import fetch_and_save

from .base import Adapter


class GlobGM(Adapter):
    """Adapter for GlobGM datasets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the GlobGM adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> GlobGM:
        """Fetch the dataset from the given URL if not already present.

        Args:
            url: The URL to fetch the dataset from.

        Returns:
            The GlobGM adapter instance.
        """
        if not self.is_ready:
            fetch_and_save(url=url, file_path=self.path)
        return self

    def read(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        """Read the dataset into an xarray DataArray.

        Args:
            *args: Additional positional arguments to pass to the reader function.
            **kwargs: Additional keyword arguments to pass to the reader function.

        Returns:
            The dataset as an xarray DataArray with appropriate CRS and attributes.

        Raises:
            ValueError: If the file format is unsupported.
        """
        da = super().read(*args, **kwargs)
        if self.path.suffix == ".tif":
            da = da.sel(band=1)
        elif self.path.suffix == ".nc":
            da = da.rename({"lon": "x", "lat": "y"})
        else:
            raise ValueError("Unsupported file format for GlobGM data.")
        da = da.rio.write_crs("EPSG:4326")
        da.attrs["_FillValue"] = np.nan
        return da


class GlobGMDEM(GlobGM):
    """Adapter for GlobGM DEM dataset.

    The GLOBGM DEM is part of a much larger dataset (~70GB). Therefore, we fetch the
    entire dataset and extract the DEM from it (4GB).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the GlobGMDEM adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> GlobGMDEM:
        """Fetch the DEM dataset from the given URL if not already present.

        Extracts the 'dem_average' variable from the downloaded dataset and
        saves it as a NetCDF file. Deletes the original downloaded file to save space.

        Args:
            url: The URL to fetch the dataset from.

        Returns:
            The GlobGMDEM adapter instance.
        """
        if not self.is_ready:
            download_path = self.root / url.split("/")[-1]
            fetch_and_save(url=url, file_path=download_path)
            ds = xr.open_dataset(download_path)
            da = ds["dem_average"]
            da.to_netcdf(self.path)
            download_path.unlink()  # Remove the downloaded file
        return self
