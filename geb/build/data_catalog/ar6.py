"""File adapter for IPCC AR6 Sea Level Projections."""

import io
import zipfile
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from geb.workflows.io import fetch_and_save

from .base import Adapter


class AR6(Adapter):
    """Adapter for IPCC AR6 Sea Level Projections."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the AR6 data adapter."""
        super().__init__(*args, **kwargs)

    def fetch(
        self,
        url: str | None = None,
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

    def read(self, scenario: str | int, percentile: float) -> xr.Dataset:
        """Read the AR6 data.

        Args:
            scenario: The SSP scenario to read (e.g., 119, 126, 245, 370, 585).
            percentile: The percentile to read (e.g., 0.5, 0.95, 1.0).
        Returns:
            The AR6 data as a Dataset.
        Raises:
            FileNotFoundError: If the specified file is not found in the zip archive.
            ValueError: If AR6 locations with invalid coordinates are found.
        """
        file_in_zip = f"ar6-regional-confidence/regional/confidence_output_files/medium_confidence/ssp{scenario}/total_ssp{scenario}_medium_confidence_values.nc"
        # check if the file exists in the zip
        with zipfile.ZipFile(self.path, "r") as z:
            if file_in_zip not in z.namelist():
                raise FileNotFoundError(
                    f"{file_in_zip} not found in {self.path}. Check if the scenario is correct."
                )
            with z.open(file_in_zip) as f:
                data = f.read()

        ds = xr.open_dataset(io.BytesIO(data))
        ds = ds.sel(quantiles=percentile)

        # Express relative to 2020 and convert from mm to m
        ds["sea_level_change"] = (
            ds["sea_level_change"] - ds["sea_level_change"].sel(years=2020)
        ) / 1000.0

        # assert all locations have valid coordinates
        lon_deg: npt.NDArray[np.floating] = ds["lon"].values
        lat_deg: npt.NDArray[np.floating] = ds["lat"].values

        valid_location_indices: npt.NDArray[np.intp] = np.flatnonzero(
            np.isfinite(lat_deg) & np.isfinite(lon_deg)
        )
        if valid_location_indices.size != lat_deg.size:
            raise ValueError("AR6 locations with invalid coordinates found.")

        return ds
