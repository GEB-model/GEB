"""File adapter for IPCC AR6 Sea Level Projections."""

from typing import Any

import pandas as pd

from geb.workflows.io import fetch_and_save

from .base import Adapter

import xarray as xr
import zipfile
import io


class AR6(Adapter):
    """Adapter for IPCC AR6 Sea Level Projections."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the AR6 data adapter."""
        super().__init__(*args, **kwargs)

    def fetch(
        self,
        url=None,
    ) -> AR6:
        """Fetch the dataset from the given URL if not already present.

        Args:
            url: The URL to fetch the dataset from.

        Returns:
            The AR6 adapter instance.
        """
        if not self.is_ready:
            fetch_and_save(url=url, file_path=self.path, logger=self.logger)
        return self

    def read(self, scenario, percentile) -> xr.DataArray:
        """Read the AR6 data."""
        file_in_zip = f"ar6-regional-confidence/regional/confidence_output_files/medium_confidence/ssp{scenario}/total_ssp{scenario}_medium_confidence_values.nc"
        with zipfile.ZipFile(self.path, "r") as z, z.open(file_in_zip) as f:
            data = f.read()

        ds = xr.open_dataset(io.BytesIO(data))
        ds = ds.sel(quantiles=percentile)
        return ds
