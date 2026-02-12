"""Adapter for LISFLOOD vegetation properties datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from geb.workflows.io import fetch_and_save

from .base import Adapter


class LISFLOOD(Adapter):
    """Adapter for LISFLOOD vegetation properties datasets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the vegetation properties adapter.

        Args:
            *args: Positional arguments passed to the base Adapter.
            **kwargs: Keyword arguments passed to the base Adapter.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> LISFLOOD:
        """Download a vegetation properties NetCDF file if missing.

        Args:
            url: URL of the vegetation properties NetCDF file.

        Returns:
            The vegetation properties adapter instance.
        """
        if not self.is_ready:
            fetch_and_save(url=url, file_path=self.path)
        return self

    def read(self, **kwargs: Any) -> xr.DataArray:
        """Read a vegetation properties NetCDF file as a DataArray.

        Args:
            **kwargs: Additional keyword arguments forwarded to xarray.

        Returns:
            The vegetation property as an xarray DataArray.
        """
        ds: xr.Dataset = xr.open_dataset(self.path, **kwargs)
        da: xr.DataArray = ds["Band1"]
        da = da.rio.write_grid_mapping("spatial_ref")
        da = da.rio.write_crs(4326)
        da = da.rename(
            {
                "lat": "y",
                "lon": "x",
            }
        )
        da.attrs["_FillValue"] = np.nan

        return da
