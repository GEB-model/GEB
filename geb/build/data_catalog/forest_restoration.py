"""Adapter for Forest Restoration Potential datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from geb.workflows.io import fetch_and_save

from .base import Adapter


class ForestRestorationPotential(Adapter):
    """Adapter for Forest Restoration Potential dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the ForestRestorationPotential adapter.

        Args:
            *args: Positional arguments passed to the base Adapter.
            **kwargs: Keyword arguments passed to the base Adapter.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> ForestRestorationPotential:
        """Download the Forest Restoration Potential dataset.

        Args:
            url: URL of the dataset file.

        Returns:
            The ForestRestorationPotential adapter instance.
        """
        if self.is_ready:
            return self

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fetch_and_save(url=url, file_path=self.path)

        return self

    def read(self) -> xr.DataArray:
        """Read the Forest Restoration Potential dataset.

        Returns:
            An xarray DataArray containing the dataset.
        """
        da = xr.open_dataarray(self.path).sel(band=1)
        da.attrs["_FillValue"] = np.nan
        assert da.rio.crs is not None, "CRS information is missing from the dataset."
        return da
