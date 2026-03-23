"""Utilities for working with GTSM data in GEB."""

from __future__ import annotations

import os
import tempfile
import zipfile
from itertools import batched
from typing import Any

import cdsapi
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

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

    def fetch(self, url: str | None = None) -> GTSM:
        """Fetch GTSM data and save it to the cache directory.

        Args:
            url: Not used for GTSM, included for compatibility with base class.
        Returns:
            The current instance of the GTSM adapter.
        """
        if self.path.exists():
            return self
        # make directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)
        request = self.construct_request()
        self.download_data(request, str(self.path))
        return self

    def read(self, bounds: tuple[float, float, float, float]) -> xr.Dataset:
        """Read GTSM data from the cached files.

        Args:
            bounds: A tuple of four floats representing the bounding box (min_x, min_y, max_x, max_y).
        Returns:
            An xarray DataArray containing the GTSM data clipped to the specified bounds.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_path = temp_dir

            # Extract the zip file
            with zipfile.ZipFile(self.path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
                # read all netcdf files in the extracted folder
                gtsm_data = []
                for file in os.listdir(extract_path):
                    if file.endswith(".nc"):
                        file_path = os.path.join(extract_path, file)
                        # Open datasets lazily; they will be concatenated and loaded in-memory
                        gtsm_data.append(xr.open_dataset(file_path))
                # Merge all datasets into a single in-memory xarray object before temp dir cleanup
                merged_data = xr.concat(gtsm_data, dim="time").load()
                # Explicitly close underlying datasets to release file handles
                for ds in gtsm_data:
                    ds.close()

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


class GTSM_timeseries(Adapter):
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

    def fetch(self, variable: str, url: str | None = None) -> GTSM_timeseries:
        """Fetch GTSM data and save it to the cache directory.

        Args:
            variable: The variable to fetch from GTSM (e.g., "total_water_level" or "surge").
            url: Not used for GTSM, included for compatibility with base class.
        Returns:
            The current instance of the GTSM_timeseries adapter.
        """
        START_YEAR = 1979
        END_YEAR = 2018

        years = [str(year) for year in range(START_YEAR, END_YEAR + 1)]
        years_to_download = []
        if variable == "total_water_level":
            name_to_check = "waterlevel"
        else:
            name_to_check = variable
        # first do a filecheck to see if the files already exist
        for year in years:
            filepath = self.root / f"reanalysis_{name_to_check}_10min_{year}_01_v3.nc"
            if not filepath.exists():
                years_to_download.append(year)
        if not years_to_download:
            return self
        for year_batch in batched(years_to_download, 5):
            output_fp = self.root / f"{variable}_{year_batch[0]}_{year_batch[-1]}.zip"
            if output_fp.exists():
                print(f"Skipping {output_fp}")
                continue
            request = self.construct_request(year_batch, variable)
            self.download_data(request, str(output_fp))
            with zipfile.ZipFile(output_fp, "r") as zip_ref:
                zip_ref.extractall(self.root)
            output_fp.unlink()
        return self

    def construct_request(self, year_batch: list[str], variable: str) -> dict[str, Any]:
        """Construct the API call dictionary for GTSM data retrieval.

        Args:
             year_batch: A list of years to include in the request.
             variable: The variable to retrieve from GTSM.
        Returns:
             A dictionary containing the parameters for the GTSM API call.
        """
        request = {
            "format": "zip",
            "month": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ],
            "experiment": "reanalysis",
            "year": list(year_batch),
            "temporal_aggregation": "10_min",
            "variable": variable,
            "version": ["v3"],
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

    def read(
        self, bounds: tuple[float, float, float, float], variable: str
    ) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """Read GTSM data from the downloaded files.

        Args:
            bounds: A tuple of four floats representing the bounding box (min_x, min_y, max_x, max_y).
            variable: The variable to read from the GTSM data.

        Returns:
            A tuple containing a pandas DataFrame with the GTSM time series data clipped to the specified bounds and a GeoDataFrame with station information.
        """
        # get the model bounds and buffer by ~2km
        model_bounds = bounds
        model_bounds = (
            model_bounds[0] - 0.0166,  # min_lon
            model_bounds[1] - 0.0166,  # min_lat
            model_bounds[2] + 0.0166,  # max_lon
            model_bounds[3] + 0.0166,  # max_lat
        )
        min_lon, min_lat, max_lon, max_lat = model_bounds

        # First: get station indices from ONE representative file
        if variable == "total_water_level":
            name_to_check = "waterlevel"
        else:
            name_to_check = variable

        ref_file = self.root / f"reanalysis_{name_to_check}_10min_1979_01_v3.nc"
        # ty:ignore[possibly-missing-attribute]
        ref = xr.open_dataset(ref_file)

        x_coords = ref.station_x_coordinate.load()
        y_coords = ref.station_y_coordinate.load()

        mask = (
            (x_coords >= min_lon)
            & (x_coords <= max_lon)
            & (y_coords >= min_lat)
            & (y_coords <= max_lat)
        )
        station_idx = np.nonzero(mask.values)[0]

        station_df = pd.DataFrame(
            {
                "station_id": ref.stations.values[mask].astype(str),
                "longitude": x_coords[mask].values,
                "latitude": y_coords[mask].values,
            }
        )
        stations = gpd.GeoDataFrame(
            station_df,
            geometry=[
                Point(xy) for xy in zip(station_df.longitude, station_df.latitude)
            ],
            crs="EPSG:4326",
        )

        ref.close()

        # Then: loop through files in smaller batches
        gtsm_data_region = []
        for year in np.arange(1979, 2019):
            for month in range(1, 13):
                f = (
                    self.root
                    / f"reanalysis_{name_to_check}_10min_{year}_{month:02d}_v3.nc"
                )
                ds = xr.open_dataset(f, chunks={"time": -1})
                subset = ds.isel(stations=station_idx).drop_vars(
                    ["station_x_coordinate", "station_y_coordinate"]
                )
                gtsm_data_region.append(subset[name_to_check].to_pandas())
                print(f"Processed GTSM data for {year}-{month:02d}")
                ds.close()
        gtsm_data_region_pd = pd.concat(gtsm_data_region, axis=0)
        return gtsm_data_region_pd, stations
