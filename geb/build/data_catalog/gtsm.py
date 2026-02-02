"""Utilities for working with GTSM data in GEB."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import cdsapi
import xarray as xr

from .base import Adapter


class GTSM(Adapter):
    """Downloader for GTSM netcdf.

    Downloads and extracts the needed netcdf files from the remote GTSM for a given bounding box.
    Attributes:
        cache_dir (Path): Directory to cache downloaded tiles.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter for GTSM.

        Args:
            *args: Additional positional arguments passed to the base Adapter class.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)

    def construct_request(self) -> dict[str, Any]:
        """Construct the API call dictionary for GTSM data retrieval.

        Returns:
            A dictionary containing the parameters for the GTSM API call.
        """
        years = [str(year) for year in range(2015, 2051)]
        request = {
            "variable": ["mean_sea_level"],
            "experiment": "future",
            "temporal_aggregation": ["annual"],
            "year": years,
            "version": ["v1"],
        }

        return request

    def download_data(self, request: dict[str, Any], output_path: str) -> None:
        """Download GTSM data using the CDS API.

        Args:
            request: A dictionary containing the parameters for the GTSM API call.
            output_path: The file path where the downloaded data will be saved.
        """
        c = cdsapi.Client()
        c.retrieve(
            "sis-water-level-change-timeseries-cmip6",
            request,
            output_path,
        )

    def fetch(self, url: str | None = None) -> None:
        """Fetch GTSM data and save it to the cache directory.

        Args:
            url: Not used for GTSM, included for compatibility with base class.
        Returns:
            The current instance of the GTSM adapter.
        """
        self.output_path: Path = self.root / self.filename

        if self.output_path.exists():
            return self
        # make directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)
        request = self.construct_request()
        self.download_data(request, str(self.output_path))
        return self

    def read(self, bounds: tuple[float, float, float, float]) -> xr.DataArray:
        """Read GTSM data from the cached files.

        Args:
            bounds: A tuple of four floats representing the bounding box (min_x, min_y, max_x, max_y).
        Returns:
            An xarray DataArray containing the GTSM data clipped to the specified bounds.
        """
        zip_path = self.output_path
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_path = temp_dir

            # Extract the zip file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
                # read all netcdf files in the extracted folder
                gtsm_data = []
                for file in os.listdir(extract_path):
                    if file.endswith(".nc"):
                        file_path = os.path.join(extract_path, file)
                        # process the netcdf file as needed
                        gtsm_data.append(xr.open_dataset(file_path))
        # merge all datasets into a single xarray DataArray
        merged_data = xr.concat(gtsm_data, dim="time")

        # clip to model bounds
        mask = (
            (merged_data.station_x_coordinate >= bounds[0])
            & (merged_data.station_x_coordinate <= bounds[2])
            & (merged_data.station_y_coordinate >= bounds[1])
            & (merged_data.station_y_coordinate <= bounds[3])
        )

        # use boolean indexing along the 'stations' dimension
        merged_data = merged_data.isel(stations=mask)

        return merged_data
