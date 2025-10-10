"""Adapter for Global Runoff Data Centre."""

import io
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

    def fetch(self, url: str) -> "GRDC":
        """Fetch and process the Global Data Lab shapefiles.

        Because login is required to download the data, the user must manually
        download the zip file from the provided URL and place it in the expected
        location.

        Args:
            url: The URL to download the shapefiles from.

        Returns:
            GRDC: The adapter instance with the processed data.
        """
        if not self.is_ready:
            print(
                f"""\033[91mThis file requires manual download due to licensing restrictions. Download the Global Runoff Data Centre (GRDC) dataset via the GRDC station portal: {url}. You can use the map to draw a polygon or rectangle to obtain all (or a subset) of stations, then click download and fill out the required forms. Save the resulting "GRDC.zip" in {self.path}.\033[0m"""
            )
            exit(1)

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

            # Read file content into memory
            with zip_file.open(nc_filename) as file:
                file_content = file.read()

        # Open dataset from memory buffer
        Q_obs = xr.open_dataset(io.BytesIO(file_content), engine="h5netcdf")

        # rename geo_x and geo_y to x and y
        Q_obs = Q_obs.rename({"geo_x": "x", "geo_y": "y"})

        return Q_obs
