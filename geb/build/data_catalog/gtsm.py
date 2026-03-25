"""Utilities for working with GTSM data in GEB."""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from itertools import batched
from pathlib import Path
from typing import Any

import cdsapi
import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from zarr.codecs.numcodecs import FixedScaleOffset

from geb.workflows.io import read_geom, write_geom, write_zarr

from .base import Adapter


def _retrieve(request: dict[str, Any], output_path: str) -> None:
    """Download CDS data to a temporary file and atomically move it into place.

    Args:
        request: Parameters for the CDS API request.
        output_path: Final destination for the downloaded file.
    """
    output_file: Path = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Keep the temporary file in the target directory so the final replace stays atomic.
    with tempfile.NamedTemporaryFile() as temporary_file:
        temporary_path = Path(temporary_file.name)

        client = cdsapi.Client()
        client.retrieve(
            "sis-water-level-change-timeseries-cmip6",
            request,
            str(temporary_path),
        )
        shutil.move(temporary_file.name, output_path)


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
        _retrieve(request, output_path)

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
        self.start_year = 1979
        self.end_year = 2018

    def _variable_dir(self, variable: str) -> Path:
        """Return the directory containing cached files for a variable.

        Args:
            variable: GTSM variable name.

        Returns:
            Directory used to store the requested variable.
        """
        return self.root / variable

    def _variable_station_locations_file(self, variable: str) -> Path:
        """Return the station metadata file path for a variable cache.

        Args:
            variable: GTSM variable name.

        Returns:
            Path to the station metadata parquet file stored with the variable data.
        """
        return self._variable_dir(variable) / "station_locations.parquet"

    def fetch(self, variable: str, url: str | None = None) -> GTSM_timeseries:
        """Fetch GTSM data and save it to the cache directory.

        Args:
            variable: The variable to fetch from GTSM (e.g., "total_water_level" or "surge").
            url: Not used for GTSM, included for compatibility with base class.
        Returns:
            The current instance of the GTSM_timeseries adapter.
        """
        variable_dir: Path = self._variable_dir(variable)
        variable_station_locations_file: Path = self._variable_station_locations_file(
            variable
        )
        variable_dir.mkdir(parents=True, exist_ok=True)

        if variable == "total_water_level":
            name_to_check = "waterlevel"
        elif variable == "storm_surge_residual":
            name_to_check = "surge"
        else:
            name_to_check = variable

        years = [str(year) for year in range(self.start_year, self.end_year + 1)]
        filters: list = [
            FixedScaleOffset(
                offset=0,
                scale=1000,  # gtsm has a precision of 0.001, so multiplying by 1000 allows us to store as int16 without losing any precision
                dtype="float32",  # float 32 is sufficient to store the data with the given scale and offset
                astype="int16",  # int16 has sufficient range here. The int16 range of -32768 to 32767, so can store values from -32.768 to 32.767 with a scale of 1000, which is sufficient for the GTSM data
            ),
        ]

        for year_batch in batched(years, 5):
            final_zarr_fp = (
                variable_dir
                / f"reanalysis_{variable}_10min_{year_batch[0]}_{year_batch[-1]}.zarr"
            )
            if final_zarr_fp.exists():
                continue

            zip_fp = (
                variable_dir
                / f"reanalysis_{variable}_10min_{year_batch[0]}_{year_batch[-1]}.zip"
            )
            if not zip_fp.exists():
                self.logger.info(
                    f"Downloading GTSM data for {variable} from {year_batch[0]} to {year_batch[-1]}"
                )
                request = self.construct_request(year_batch, variable)
                self.download_data(request, str(zip_fp))

            with zipfile.ZipFile(zip_fp, "r") as zip_ref:
                nc_files = [
                    Path(f"reanalysis_{name_to_check}_10min_{year}_{month:02d}_v3.nc")
                    for month in range(1, 13)
                    for year in year_batch
                ]
                # print files in zip
                zip_contents = zip_ref.namelist()
                print(f"Contents of zip file {zip_fp}: {zip_contents}")

                self.logger.info(
                    f"Processing GTSM data for {variable} from {year_batch[0]} to {year_batch[-1]}"
                )
                self.logger.debug(
                    f"Extracting and processing {len(nc_files)} netCDF files in batches. This will take a while but only has to be done once."
                )

                station_chunksize = 1000
                zarr_das: list[xr.DataArray] = []
                for nc_file in tqdm.tqdm(nc_files):
                    # decompress nc file to temporary file
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_dir = Path(temp_dir)
                        temp_nc_path = temp_dir / nc_file.name
                        zip_ref.extract(str(nc_file), path=temp_dir)

                        da: xr.DataArray = xr.open_dataarray(
                            temp_nc_path, engine="netcdf4"
                        ).load()
                        da = da.chunk({"stations": station_chunksize, "time": -1})
                        da = da.astype(np.float32)
                        da.attrs["_FillValue"] = np.nan

                        zarr_das.append(
                            write_zarr(
                                da,
                                path=None,
                                crs=4326,
                                compression_level=1,
                                shards={
                                    "stations": len(da.stations) // station_chunksize
                                    + 1
                                },
                                filters=filters,
                                progress=False,
                            )
                        )

                        if not variable_station_locations_file.exists():
                            station_locations: gpd.GeoDataFrame = gpd.GeoDataFrame(
                                {
                                    "station_id": da["stations"].load().values,
                                },
                                geometry=gpd.points_from_xy(
                                    da["station_x_coordinate"].values,
                                    da["station_y_coordinate"].values,
                                ),
                                crs="EPSG:4326",
                            ).set_index("station_id")  # ty:ignore[invalid-assignment]
                            # Keep a variable-local copy next to the cached data so each
                            # variable directory is self-contained.
                            write_geom(
                                station_locations,
                                variable_station_locations_file,
                                write_covering_bbox=True,
                            )

                da: xr.DataArray = xr.concat(
                    zarr_das, dim="time", combine_attrs="drop_conflicts"
                ).chunk({"time": -1})
                da.attrs["_FillValue"] = np.nan

                self.logger.info(
                    f"Writing merged GTSM data for {variable} from {year_batch[0]} to {year_batch[-1]} to Zarr format. This will take a while but only has to be done once."
                )
                write_zarr(
                    da,
                    final_zarr_fp,
                    crs=4326,
                    compression_level=12,
                    filters=filters,
                )

            zip_fp.unlink()

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
        _retrieve(request, output_path)

    def read(
        self, bounds: tuple[float, float, float, float], variable: str
    ) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """Read GTSM data from the downloaded files.

        Args:
            bounds: A tuple of four floats representing the bounding box (min_x, min_y, max_x, max_y).
            variable: The variable to read from the GTSM data.

        Returns:
            A tuple containing a pandas DataFrame with the GTSM time series data clipped to the specified bounds and a GeoDataFrame with station information.

        Raises:
            FileNotFoundError: If the station metadata for the requested variable
                has not been generated yet.
        """
        # get the model bounds and buffer by ~2km
        model_bounds = bounds
        model_bounds = (
            model_bounds[0] - 0.0166,  # min_lon
            model_bounds[1] - 0.0166,  # min_lat
            model_bounds[2] + 0.0166,  # max_lon
            model_bounds[3] + 0.0166,  # max_lat
        )
        variable_station_locations_file: Path = self._variable_station_locations_file(
            variable
        )
        if not variable_station_locations_file.exists():
            raise FileNotFoundError(
                f"Station metadata for variable '{variable}' was not found at "
                f"{variable_station_locations_file}. Fetch this variable before reading it."
            )

        station_locations = read_geom(
            variable_station_locations_file, bbox=model_bounds
        )

        if station_locations.empty:
            return pd.DataFrame(), station_locations

        variable_dir: Path = self._variable_dir(variable)
        zarr_paths: list[Path] = [
            variable_dir
            / f"reanalysis_{variable}_10min_{start_year}_{min(start_year + 4, self.end_year)}.zarr"
            for start_year in range(self.start_year, self.end_year + 1, 5)
        ]

        with xr.open_mfdataset(
            [str(zarr_path) for zarr_path in zarr_paths],
            engine="zarr",
            concat_dim="time",
            combine_attrs="drop_conflicts",
        ) as combined_dataset:
            selected_station_ids: list[int] = [
                int(station_id) for station_id in station_locations.index
            ]
            combined_data_array: xr.DataArray = combined_dataset[variable].sel(
                stations=selected_station_ids
            )
            gtsm_data_region_pd = combined_data_array.to_pandas()

        assert isinstance(gtsm_data_region_pd, pd.DataFrame)
        return gtsm_data_region_pd, station_locations
