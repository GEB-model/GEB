"""Adapter for Global Runoff Data Centre."""

from __future__ import annotations

import tempfile
import zipfile
from typing import Any

import xarray as xr

from .base import Adapter


class GRDC(Adapter):
    """Adapter for GRDC discharge data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the GRDC adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> GRDC:
        """Fetch and process the Global Data Lab shapefiles.

        Because login is required to download the data, the user must manually
        download the zip file from the provided URL and place it in the expected
        location.

        Args:
            url: The URL to download the shapefiles from.

        Returns:
            GRDC: The adapter instance with the processed data.
        """
        while not self.is_ready:
            print(
                f"""\033[91mThis file requires manual download due to licensing restrictions. Download the Global Runoff Data Centre (GRDC) dataset via the GRDC station portal: {url}. You can use the map to draw a polygon or rectangle to obtain all (or a subset) of stations, then click download and fill out the required forms. Save the resulting "GRDC.zip" in {self.path}.\033[0m"""
            )
            input("\033[91mPress Enter after placing the file to continue...\033[0m")

        return self

    def read(self, **kwargs: Any) -> xr.Dataset:
        """Read the processed data from storage.

        Unpacks the zip file and reads the contained .nc file into an xarray Dataset.

        Args:
            **kwargs: Additional keyword arguments to pass to the reader function.

        Returns:
            The processed data as a Dataset
        """
        # Handle zip file containing single .nc file
        with zipfile.ZipFile(self.path, "r") as zip_file:
            netcdf_files = [f for f in zip_file.namelist() if f.endswith(".nc")]
            assert len(netcdf_files) == 1, "Expected exactly one .nc file in the zip."
            nc_filename = netcdf_files[0]

            # Extract to temporary file instead of reading into memory
            with tempfile.NamedTemporaryFile(suffix=".nc") as tmp_file:
                with zip_file.open(nc_filename) as zip_content:
                    tmp_file.write(zip_content.read())
                tmp_file_path = tmp_file.name

                # Open dataset from temporary file
                # Load the dataset into memory to avoid issues with closed files
                Q_obs: xr.Dataset = xr.open_dataset(tmp_file_path).load()

        # rename geo_x and geo_y to x and y
        Q_obs: xr.Dataset = Q_obs.rename({"geo_x": "x", "geo_y": "y"})

        return Q_obs
