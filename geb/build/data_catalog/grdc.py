"""Adapter for Global Runoff Data Centre."""

from __future__ import annotations

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

        # Unpack the zip file to disk if it hasn't been done yet
        # The zip usually contains a single .nc file
        zarr_path = self.path.with_suffix(".zarr")

        if not zarr_path.exists():
            print(f"Processing the downloaded GRDC data from {self.path}...")
            netcdf_file = zarr_path.with_suffix(".nc")
            with zipfile.ZipFile(self.path, "r") as zip_file:
                with (
                    zip_file.open("GRDC-Daily.nc") as source,
                ):
                    netcdf_file.write_bytes(source.read())

                    with xr.open_dataset(
                        netcdf_file, engine="netcdf4", cache=False
                    ) as ds:
                        data = ds.load()

                    data["runoff_mean"] = data["runoff_mean"].chunk({"id": 100})

                    data.to_zarr(zarr_path, mode="w", consolidated=False)

                    netcdf_file.unlink()  # Remove the temporary .nc file

        return self

    def read(self, **kwargs: Any) -> xr.Dataset:
        """Read the processed data from storage.

        Reads the unpacked .zarr file directly.

        Args:
            **kwargs: Additional keyword arguments to pass to the reader function.

        Returns:
            The processed data as a Dataset
        """
        zarr_path = self.path.with_suffix(".zarr")
        if not zarr_path.exists():
            self.fetch("")

        discharge_observations: xr.Dataset = xr.open_zarr(zarr_path, consolidated=False)

        discharge_observations = discharge_observations.rename(
            {"geo_x": "x", "geo_y": "y"}
        )

        return discharge_observations
