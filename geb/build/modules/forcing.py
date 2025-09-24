import logging
import os
import shutil
import tempfile
import time
import zipfile
from ast import Raise
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, List
from urllib.parse import urlparse

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ecmwfapi
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rioxarray as rxr
import xarray as xr
import xclim.indices as xci
from cartopy.mpl.geoaxes import GeoAxes
from dateutil.relativedelta import relativedelta
from isimip_client.client import ISIMIPClient
from numcodecs.zarr3 import FixedScaleOffset
from tqdm import tqdm

from geb.build.methods import build_method
from geb.workflows.io import get_window

from ...workflows.io import calculate_scaling, open_zarr, to_zarr
from ..workflows.general import (
    interpolate_na_along_time_dim,
    resample_like,
)


def reproject_and_apply_lapse_rate_temperature(
    T, elevation_forcing, elevation_target, lapse_rate=-0.0065
) -> xr.DataArray:
    assert (T.x.values == elevation_forcing.x.values).all()
    assert (T.y.values == elevation_forcing.y.values).all()

    t_at_sea_level = T - elevation_forcing * lapse_rate
    t_at_sea_level_reprojected = resample_like(
        t_at_sea_level, elevation_target, method="conservative"
    )

    T_grid = t_at_sea_level_reprojected + lapse_rate * elevation_target
    T_grid.name = T.name
    T_grid.attrs = T.attrs
    return T_grid


def get_pressure_correction_factor(DEM, g, Mo, lapse_rate):
    return (288.15 / (288.15 + lapse_rate * DEM)) ** (g * Mo / (8.3144621 * lapse_rate))


def reproject_and_apply_lapse_rate_pressure(
    pressure: xr.DataArray,
    elevation_forcing: xr.DataArray,
    elevation_target: xr.DataArray,
    g: float = 9.80665,
    Mo: float = 0.0289644,
    lapse_rate: float = -0.0065,
) -> xr.DataArray:
    """Pressure correction based on elevation lapse_rate.

    Args:
        pressure: Pressure data to reproject and apply lapse rate to [Pa].
        elevation_forcing: Elevation data for the forcing grid [m].
        elevation_target: Elevation data for the target grid [m].
        g: gravitational constant [m s-2]
        Mo: molecular weight of gas [kg / mol]
        lapse_rate: lapse rate of temperature [C m-1]

    Returns:
        press_fact : xarray.DataArray
            pressure correction factor
    """
    assert (pressure.x.values == elevation_forcing.x.values).all()
    assert (pressure.y.values == elevation_forcing.y.values).all
    pressure_at_sea_level = pressure / get_pressure_correction_factor(
        elevation_forcing, g, Mo, lapse_rate
    )  # divide by pressure factor to get pressure at sea level
    pressure_at_sea_level_reprojected = resample_like(
        pressure_at_sea_level, elevation_target, method="conservative"
    )
    pressure_grid = (
        pressure_at_sea_level_reprojected
        * get_pressure_correction_factor(elevation_target, g, Mo, lapse_rate)
    )  # multiply by pressure factor to get pressure at DEM grid, corrected for elevation
    pressure_grid.name = pressure.name
    pressure_grid.attrs = pressure.attrs

    return pressure_grid


def download_ERA5(
    folder: Path,
    variable: str,
    start_date: datetime,
    end_date: datetime,
    bounds: tuple[float, float, float, float],
    logger: logging.Logger,
) -> xr.DataArray:
    """Download ERA5 data for a specific variable and time period and save it as a zarr file.

    If the data is already downloaded, it will be opened from the local zarr file.

    Args:
        folder: Folder to store the downloaded data
        variable: Short name of the variable to download (e.g., "t2m"). Codes can be found here: https://codes.ecmwf.int/grib/param-db/
        start_date: Start date of the time period to download
        end_date: End date of the time period to download
        bounds: Bounding box in the format (min_lon, min_lat, max_lon, max_lat)
        logger: Logger to use for logging

    Returns:
        da: Downloaded ERA5 data as an xarray DataArray.
    """

    output_fn = folder / f"{variable}.zarr"
    if output_fn.exists():
        da: xr.DataArray = open_zarr(output_fn)

        # check if entire time range is available. If available, return the cached data
        # otherwise, slice the data to the requested time range and return it
        if start_date >= pd.to_datetime(
            da.time[0].values
        ) and end_date <= pd.to_datetime(da.time[-1].values):
            logger.debug(
                f"Using cached ERA5 {variable} data from {output_fn} for time range {start_date} to {end_date}"
            )
            da: xr.DataArray = da.sel(
                time=slice(start_date, end_date),
            )
            return da
        else:
            # remove the existing zarr folder
            logger.debug(
                f"Removing existing zarr folder {output_fn} as it does not contain the requested time range"
            )
            shutil.rmtree(output_fn)

    folder.mkdir(parents=True, exist_ok=True)
    da: xr.DataArray = xr.open_dataset(
        "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr",
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
    )[variable].rename({"valid_time": "time", "latitude": "y", "longitude": "x"})

    da: xr.DataArray = da.drop_vars(["number", "surface", "depthBelowLandLayer"])

    buffer: float = 0.5

    # Check if region crosses the meridian (longitude=0)
    # use a slightly larger slice. The resolution is 0.1 degrees, so buffer degrees is a bit more than that (to be sure)
    if bounds[0] < 0 and bounds[2] > 0:
        # Need to handle the split across the meridian
        # Get western hemisphere part (longitude < 0)
        west_da: xr.DataArray = da.sel(
            time=slice(start_date, end_date),
            y=slice(bounds[3] + buffer, bounds[1] - buffer),
            x=slice(((bounds[0] - buffer) + 360) % 360, 360),
        )
        # Get eastern hemisphere part (longitude > 0)
        east_da: xr.DataArray = da.sel(
            time=slice(start_date, end_date),
            y=slice(bounds[3] + buffer, bounds[1] - buffer),
            x=slice(0, ((bounds[2] + buffer) + 360) % 360),
        )
        # Combine the two parts
        da: xr.DataArray = xr.concat([west_da, east_da], dim="x")
    else:
        # Regular case - doesn't cross meridian
        da: xr.DataArray = da.sel(
            time=slice(start_date, end_date),
            y=slice(bounds[3] + buffer, bounds[1] - buffer),
            x=slice(
                ((bounds[0] - buffer) + 360) % 360,
                ((bounds[2] + buffer) + 360) % 360,
            ),
        )

    # Reorder x to be between -180 and 180 degrees
    da: xr.DataArray = da.assign_coords(x=((da.x + 180) % 360 - 180))

    logger.info(f"Downloading ERA5 {variable} to {output_fn}")
    da.attrs["_FillValue"] = da.attrs["GRIB_missingValue"]
    da: xr.DataArray = da.raster.mask_nodata()
    da: xr.DataArray = to_zarr(
        da,
        output_fn,
        time_chunksize=get_chunk_size(da, target=1e7),
        crs=4326,
    )
    return da


def process_ERA5(
    variable: str,
    folder: Path,
    start_date: datetime,
    end_date: datetime,
    bounds: tuple[float, float, float, float],
    logger: logging.Logger,
) -> xr.DataArray:
    """Process ERA5 data for a given variable and time period.

    Downloads the data from the Climate Data Store (CDS) if not already available,
    processes it to ensure it is in the correct format, and applies de-accumulation
    for accumulated variables and interpolation of missing values.

    Args:
        variable: short name of the variable to process (e.g., "t2m"). Codes can be found here: https://codes.ecmwf.int/grib/param-db/
        folder: folder to store the downloaded data
        start_date: start date of the time period to process
        end_date: end date of the time period to process
        bounds:  bounding box in the format (min_lon, min_lat, max_lon, max_lat)
        logger:  logger to use for logging

    Raises:
        NotImplementedError: If the step type of the data is not "accum" or "instant".

    Returns:
        da: Processed ERA5 data as an xarray DataArray.
    """
    da: xr.DataArray = download_ERA5(
        folder, variable, start_date, end_date, bounds, logger
    )
    # assert that time is monotonically increasing with a constant step size
    assert (
        da.time.diff("time").astype(np.int64)
        == (da.time[1] - da.time[0]).astype(np.int64)
    ).all(), "time is not monotonically increasing with a constant step size"
    if da.attrs["GRIB_stepType"] == "accum":
        da: xr.DataArray = xr.where(
            da.isel(time=slice(1, None)).time.dt.hour == 1,
            da.isel(time=slice(1, None)),
            da.diff(dim="time", n=1),
        )

    elif da.attrs["GRIB_stepType"] == "instant":
        pass
    else:
        raise NotImplementedError

    da: xr.DataArray = da.rio.write_crs(4326)
    da.raster.set_crs(4326)
    da: xr.DataArray = interpolate_na_along_time_dim(da)

    return da


def generate_forecast_steps(forecast_date: datetime) -> str:
    """Generate ECMWF forecast step string based on the forecast date.

    ECMWF does not have a consistent 1h timestep for the entire operational archive. Asking hourly data to the server when it does not excist, will result in an error.
    Therefore, we need to adjust the requested steps based on the available data, which is different before and after 2016-11-23:
    - Before 2016-11-23: 3-hourly steps from 0-144h, 6-hourly steps from 144-360h
    - From 2016-11-23 onwards: hourly steps from 0-90h, 3-hourly steps from 90-144h, 6-hourly steps from 144-360h

    Args:
        forecast_date: The forecast initialization date and time.

    Returns:
        ECMWF MARS step string in the format "0/3/6/9/..." with actual step hours.

    Notes:
        Returns step hours as required by ECMWF MARS API.
    """
    cutoff_date = date(
        2016, 11, 23
    )  # cutoff date for the change in forecast step availability
    steps = []  # list to hold the forecast steps

    if (
        forecast_date.date() < cutoff_date
    ):  # Before 2016-11-23: 3-hourly from 0-144h, 6-hourly from 144-360h
        steps.extend(range(0, 145, 3))  # 0, 3, 6, 9, ..., 144 (3-hourly)
        steps.extend(range(150, 241, 6))  # 150, 156, 162, ..., 360 (6-hourly from 144h)
    else:  # From 2016-11-23: hourly from 0-90h, 3-hourly from 90-144h, 6-hourly from 144-360h
        steps.extend(range(0, 91))  # 0, 1, 2, 3, ..., 90 (hourly)
        steps.extend(range(93, 145, 3))  # 93, 96, 99, ..., 144 (3-hourly from 90h)
        steps.extend(range(150, 241, 6))  # 150, 156, 162, ..., 360 (6-hourly from 144h)

    return "/".join(str(step) for step in steps)  # return step string for MARS request


def download_forecasts_ECMWF(
    self,
    forecast_variables: list[float],
    preprocessing_folder: Path,
    bounds: tuple[float, float, float, float],
    forecast_start: date | datetime,
    forecast_end: date | datetime,
    forecast_model: str,
    forecast_resolution: float,
    forecast_horizon: int,
    forecast_timestep: int,
) -> None:
    """Download ECMWF forecasts using the ECMWF web API: https://github.com/ecmwf/ecmwf-api-client.

    This function downloads ECMWF forecast data for a specified variable and time period
    from the MARS archive using the ECMWF API. It handles the download and processing of both
    deterministic (cf) and ensemble (pf) forecasts returning them as an xarray DataArray.

    This function requires the ECMWF_API_KEY to be set in the environment variables. You can do this by adding it to your .env file.

    Your API key: https://api.ecmwf.int/v1/key/
    MARS data archive: https://apps.ecmwf.int/mars-catalogue/
    Extra Documentation: https://confluence.ecmwf.int/display/UDOC/MARS+content


    Args:
        bounds: The bounding box in the format (min_lon, min_lat, max_lon, max_lat).
        forecast_variable: List of ECMWF parameter codes to download (see ECMWF documentation).
        forecast_start: The forecast initialization time (date or datetime).
        forecast_end: The forecast end time (date or datetime).
        forecast_model: The ECMWF forecast model to use (e.g., "pf" or "cf").
        forecast_resolution: The spatial resolution of the forecast data (degrees).
        forecast_horizon: The forecast horizon in hours.
        forecast_timestep: The forecast timestep in hours.

    Raises:
        ImportError: If ECMWF_API_KEY is not found in environment variables.
        ValueError: If forecast dates are before 2010-01-01.
    """

    self.logger.info(
        f"Downloading forecast variables {forecast_variables}"
    )  # Log the forecast variables being downloaded

    preprocessing_folder.mkdir(
        parents=True, exist_ok=True
    )  # Create the directory structure if it doesn't exist

    if (
        "ECMWF_API_KEY" not in os.environ
    ):  # Check if ECMWF API key is available in environment
        raise ImportError(
            "ECMWF_API_KEY not found in environment variables. Please set it to your ECMWF API key in .env file. See https://github.com/ecmwf/ecmwf-api-client"
        )
    server = ecmwfapi.ECMWFService("mars")  # Initialize ECMWF MARS service connection

    fc_area_buffer: float = 1  # spatial buffer around the forecasts
    bounds = (  # Add buffer to bounding box coordinates
        bounds[0] - fc_area_buffer,
        bounds[1] - fc_area_buffer,
        bounds[2] + fc_area_buffer,
        bounds[3] + fc_area_buffer,
    )
    bounds_str: str = f"{bounds[3]}/{bounds[0]}/{bounds[1]}/{bounds[2]}"  # setup bounds -- > bounds should be in North/West/South/East format for MARS

    forecast_date_list = pd.date_range(
        forecast_start, forecast_end, freq="24H"
    )  # Generate list of forecast dates at 24-hour intervals

    earliest_allowed_date = date(2010, 1, 1)  # Set earliest allowed forecast date
    for forecast_date in forecast_date_list:  # Loop through all forecast dates
        if (
            forecast_date.date() < earliest_allowed_date
        ):  # Check if date is before allowed range
            raise ValueError(
                f"Forecast date {forecast_date.date()} is before 2010-01-01. "
                "For historical data before 2010, please use hindcast data instead."
            )

    for (
        forecast_date
    ) in forecast_date_list:  # Loop through each forecast date to download
        print(forecast_date)  # Print the current forecast date being processed

        forecast_datetime_str = forecast_date.strftime(
            "%Y%m%dT%H%M%S"
        )  # Format datetime as string for filename
        forecast_date_str = forecast_date.strftime(
            "%Y-%m-%d"
        )  # Format date as string for MARS request

        # Process MARS request parameters
        mars_class: str = "od"  # operational data class
        mars_expver: str = "1"  # operational version number
        mars_levtype: str = "sfc"  # surface level data type
        mars_param: str = "/".join(
            str(var) for var in forecast_variables
        )  # Join parameter codes with "/" separator
        if forecast_timestep == 1:  # Check if hourly timestep is requested
            mars_step: str = generate_forecast_steps(
                forecast_date
            )  # Generate forecast steps based on date using helper function
        elif forecast_timestep >= 6:  # Check if 6+ hourly timestep is requested
            mars_step: str = f"0/to/{forecast_horizon}/BY/{forecast_timestep}"  # Create step string for multi-hour intervals
        else:
            raise ValueError(
                f"Forecast timestep {forecast_timestep} is not supported. Please use 1 or >=6."
            )
        mars_stream: str = "enfo"  # Ensemble forecast stream
        mars_time: str = forecast_date.strftime(
            "%H"
        )  # Extract hour from forecast date for initialization time
        mars_type: str = (
            "pf" if forecast_model == "pf" else "cf"
        )  # Set forecast type: perturbed forecasts (pf) or control forecast (cf)
        mars_grid: str = str(
            forecast_resolution
        )  # Convert spatial resolution to string
        mars_area: str = (
            bounds_str  # Set bounding box area in North/West/South/East format
        )
        # download only necessary dates
        missing_dates = [  # Create list of dates that don't have downloaded files yet
            date
            for date in forecast_date_list
            if not (preprocessing_folder / f"{forecast_datetime_str}.grb").exists()
        ]
        if missing_dates:  # Check if any files are missing
            print(
                f"Missing forecast files for dates: {missing_dates}"
            )  # Log missing dates
        else:
            print(
                f"All forecast files for variable {forecast_variables} are already downloaded in {preprocessing_folder}, skipping download."
            )  # Log that all files already exist
            continue  # Skip to next forecast date

        # retrieve steps from mars
        mars_request: dict[
            str, Any
        ] = {  # Build MARS request dictionary with all parameters
            "class": mars_class,
            "date": forecast_date_str,
            "expver": mars_expver,
            "levtype": mars_levtype,
            "param": mars_param,
            "step": mars_step,
            "stream": mars_stream,
            "time": mars_time,
            "type": mars_type,
            "grid": mars_grid,
            "area": mars_area,
        }

        if forecast_model == "pf":  # check if ensemble forecasts are requested
            mars_request["number"] = "1/to/50"  # Add ensemble member numbers to request
            output_filename: Path = (
                preprocessing_folder / f"ENS_{forecast_datetime_str}.grb"
            )  # Create output file path with .grb extension
        else:
            output_filename: Path = (
                preprocessing_folder / f"CTRL_{forecast_datetime_str}.grb"
            )  # Create output file path with .grb extension

        print(
            f"Requesting data from ECMWF MARS server.. {mars_request}"
        )  # Log the MARS request parameters

        server.execute(  # Execute the MARS request to download data
            mars_request,
            output_filename,
        )  # start the download


def process_forecast_ECMWF(
    self,
    preprocessing_folder: Path,
    bounds: tuple[float, float, float, float],
    forecast_issue_date: date | datetime,
) -> None | xr.Dataset:
    """Process downloaded ECMWF forecast data.

    We process forecasts for each initialization time separately. The forecast file can contain all variables needed for GEB, or only rainfall (if only_rainfall is True in build.yml).

    Args:
        preprocessing_folder: Path to the folder containing the downloaded ECMWF forecast data.
        bounds: The bounding box in the format (min_lon, min_lat, max_lon,
                max_lat).
        forecast_issue_date: The forecast initialization time (date or datetime).

    Returns:
        da: processed ECMWF forecast data as an xarray Dataset.

    """

    self.logger.info(
        f"Processing ECMWF forecasts from {preprocessing_folder}"
    )  # Log the processing folder path

    # create variable folder path
    variable_folder = (
        preprocessing_folder  # Set the folder path containing forecast files
    )
    file = (
        variable_folder / f"{forecast_issue_date.strftime('%Y%m%dT%H%M%S')}.grb"
    )  # Create full file path using formatted date

    self.logger.info(
        f"Processing forecast file: {file.name}"
    )  # Log the filename being processed

    da: xr.Dataset = xr.open_dataset(  # Open GRIB file as xarray Dataset
        file,
        engine="cfgrib",  # Use cfgrib engine for GRIB files
    ).rename(
        {"latitude": "y", "longitude": "x", "number": "member"}
    )  # Rename dimensions to standard names

    # experimental! add control forecast if available
    ctrl_file = (
        variable_folder / f"CTRL_{forecast_issue_date.strftime('%Y%m%dT%H%M%S')}.grb"
    )
    if ctrl_file.exists():
        ctrl_da: xr.Dataset = xr.open_dataset(
            ctrl_file,
            engine="cfgrib",
        ).rename(
            {"latitude": "y", "longitude": "x", "number": "member"}
        )  # Rename dimensions to standard names

        da = xr.concat(
            [da, ctrl_da], dim="member"
        )  # add control forecast as an extra member

    # ensure all the timesteps are hourly
    if not (
        da.step.diff("step").astype(np.int64) == 3600 * 1e9
    ).all():  # Check if all time differences are exactly 1 hour (3600 seconds in nanoseconds)
        # print all the unique timesteps in the time dimension
        print(
            f"Timesteps in the forecast are not hourly, resampling to hourly. Found timesteps: {np.unique(da.step.diff('step').astype(np.int64) / 1e9 / 3600)} hours"
        )  # Log the current timesteps found in the data

        da = da.resample(step="1H").interpolate(
            "linear"
        )  # Resample to hourly timesteps using linear interpolation
        # convert back to float32
        da = da.astype(np.float32)  # Convert data type back to float32 to save memory
    else:
        print(
            "All timesteps are already hourly, no need to resample"
        )  # Log that resampling is not needed

    da["tp"] = da["tp"] * 1000  # Convert precipitation from meters to millimeters
    da["tp"] = da["tp"] / 3600  # Convert precipitation from mm/hr to mm/s
    da["tp"] = da["tp"].diff(
        dim="step", n=1, label="lower"
    )  # De-accumulate precipitation by taking differences between consecutive time steps
    if (
        len(list(da.data_vars)) > 1
    ):  # Check if there are multiple variables (more than just precipitation)
        da["ssrd"] = da["ssrd"].diff(
            dim="step", n=1, label="lower"
        )  # De-accumulate shortwave radiation
        da["ssrd"] = (
            da["ssrd"] / 3600
        )  # Convert shortwave radiation from J/m2 to W/m2 by dividing by 3600 seconds
        da["strd"] = da["strd"].diff(
            dim="step", n=1, label="lower"
        )  # De-accumulate longwave radiation
        da["strd"] = (
            da["strd"] / 3600
        )  # Convert from J/m2 to W/m2 by dividing by 3600 seconds

    da = da.assign_coords(
        valid_time=da.time + da.step
    )  # Create valid_time coordinate by adding forecast initialization time to forecast step
    da = da.swap_dims(
        {"step": "valid_time"}
    )  # Swap step dimension with valid_time to make valid_time the main time dimension
    da = da.drop_vars(
        ["time", "step", "surface"]
    )  # Remove unnecessary coordinate variables
    da = da.rename(
        {"valid_time": "time"}
    )  # Rename valid_time back to time for consistency

    buffer: float = 1  # Set spatial buffer in degrees

    # Check if region crosses the meridian (longitude=0)
    # use a slightly larger slice. The resolution is 0.1 degrees, so buffer degrees is a bit more than that (to be sure)
    if (
        bounds[0] < 0 and bounds[2] > 0
    ):  # Check if bounding box crosses the 0-degree meridian
        # Need to handle the split across the meridian
        # Get western hemisphere part (longitude < 0)
        west_da: xr.DataArray = da.sel(  # Select western hemisphere data
            y=slice(
                bounds[3] + buffer, bounds[1] - buffer
            ),  # Latitude slice (note: reversed for GRIB convention)
            x=slice(
                ((bounds[0] - buffer) + 360) % 360, 360
            ),  # Longitude slice for western part
        )
        # Get eastern hemisphere part (longitude > 0)
        east_da: xr.DataArray = da.sel(  # Select eastern hemisphere data
            y=slice(bounds[3] + buffer, bounds[1] - buffer),  # Same latitude slice
            x=slice(
                0, ((bounds[2] + buffer) + 360) % 360
            ),  # Longitude slice for eastern part
        )
        # Combine the two parts
        da: xr.DataArray = xr.concat(
            [west_da, east_da], dim="x"
        )  # Concatenate western and eastern parts along longitude dimension
    else:
        # Regular case - doesn't cross meridian
        if (
            da.x.min() >= 0 and da.x.max() <= 360
        ):  # Check if longitude coordinates are in 0-360 format (probably GRIB2 files)
            da: xr.DataArray = da.sel(  # Select data using 0-360 longitude format
                y=slice(bounds[3] + buffer, bounds[1] - buffer),  # Latitude slice
                x=slice(
                    ((bounds[0] - buffer) + 360)
                    % 360,  # Convert min longitude to 0-360 format
                    ((bounds[2] + buffer) + 360)
                    % 360,  # Convert max longitude to 0-360 format
                ),
            )
        else:  # Longitude coordinates are in -180 to 180 format (probably GRIB1 files)
            da: xr.DataArray = da.sel(  # Select data using -180 to 180 longitude format
                y=slice(bounds[3] + buffer, bounds[1] - buffer),  # Latitude slice
                x=slice(
                    bounds[0] - buffer, bounds[2] + buffer
                ),  # Longitude slice with buffer
            )

    # Reorder x to be between -180 and 180 degrees
    da: xr.DataArray = da.assign_coords(
        x=((da.x + 180) % 360 - 180)
    )  # Convert longitude coordinates to -180 to 180 format
    da.attrs["_FillValue"] = np.nan  # Set fill value attribute for missing data
    da: xr.DataArray = (
        da.raster.mask_nodata()
    )  # Mask no-data values using raster accessor

    # assert that time is monotonically increasing with a constant step size
    assert (
        da.time.diff("time").astype(np.int64)
        == (da.time[1] - da.time[0]).astype(np.int64)
    ).all(), (
        "time is not monotonically increasing with a constant step size"
    )  # Validate that time dimension is properly ordered with constant intervals

    da = da.rio.write_crs(4326)  # Set coordinate reference system to WGS84 (EPSG:4326)
    da.raster.set_crs(4326)  # Also set CRS using raster accessor

    # make sure forecasts are in same grid as original data
    forcing_da = xr.open_dataarray(  # Open existing forcing data to get target grid
        "input" + "/" + self.files["other"]["climate/pr_hourly"]
    )

    da = da.interp(  # Interpolate forecast data to match the target grid
        x=forcing_da.x,  # Target longitude coordinates
        y=forcing_da.y,  # Target latitude coordinates
        method="linear",  # Use linear interpolation
    )
    # convert back to float32
    da = da.astype(np.float32)  # Convert back to float32 to save memory

    # Handling of nan values and interpolation
    for variable_name in da.data_vars:  # Loop through all variables in the dataset
        variable_data: xr.DataArray = da[variable_name]  # Get data for current variable
        nan_percentage: float = float(
            variable_data.isnull().mean().compute().item()
            * 100  # Calculate percentage of NaN values
        )
        assert nan_percentage < 5, (  # Assert that less than 5% of data is missing
            f"More than 5% of the data is missing for variable '{variable_name}' "
            f"({nan_percentage:.2f}% missing) after regridding. Check the area and try to "
            "increase the buffer around the forecasts (fc_area_buffer), as probably not "
            "the whole area is downloaded"
        )
        # fill the nan values using interpolate_na_along_time_dim and interpolate_na in space
        if nan_percentage > 0:  # Check if there are any NaN values to fill
            self.logger.warning(
                f"Found {nan_percentage:.2f}% missing values for variable '{variable_name}' after regridding. Interpolating missing values."
            )  # Log warning about missing values
            da = da.interpolate_na(
                dim=["y", "x"], method="nearest"
            )  # Interpolate NaN values spatially using nearest neighbor
            da = da.interpolate_na(
                dim=["time"], method="nearest"
            )  # Interpolate NaN values temporally using nearest neighbor

            # fill nans in last timesteps (due to de-accumulation) with mean of recent known values
            recent_mean: xr.DataArray = (  # Calculate mean of recent time steps for gap filling
                da[variable_name]
                .isel(
                    time=slice(-25, -1)
                )  # Select last 25 time steps (excluding the very last one)
                .mean(
                    dim="time", skipna=True, keep_attrs=True
                )  # Calculate mean, skipping NaN values
            )
            # Fill any remaining NaNs with the recent mean
            da[variable_name] = da[variable_name].fillna(
                recent_mean
            )  # Fill remaining NaN values with calculated mean

            nan_percentage_after: float = float(
                da[variable_name].isnull().mean().compute().item()
                * 100  # Check percentage of NaN values after interpolation
            )
            assert (
                nan_percentage_after == 0
            ), (  # Assert that all NaN values have been filled
                f"Failed to interpolate all missing values for variable '{variable_name}'. "
                f"{nan_percentage_after:.2f}% missing values remain."
            )

    return da


def plot_forcing(self, da, name) -> None:
    """Plot forcing data with a temporal (timeline) plot and a spatial plot.
    Args:
        da: The xarray DataArray containing the forcing data. Must have dimensions 'time',
        name: The name of the variable being plotted, used for titles and filenames.
    """
    fig, axes = plt.subplots(
        4, 1, figsize=(20, 10), gridspec_kw={"hspace": 0.5}
    )  # Create 4 subplots stacked vertically

    mask = self.grid["mask"]  # get the GEB grid
    data = (
        (da * ~mask).sum(dim=("y", "x")) / (~mask).sum()
    ).compute()  # Area-weighted average
    assert not np.isnan(data.values).any(), (
        "data contains NaN values"
    )  # ensure no NaNs in data

    plot_timeline(da, data, name, axes[0])  # Plot the entire timeline on the first axis

    for i in range(0, 3):  # plot the first three years on separate axes
        year = data.time[0].dt.year + i  # get the year to plot
        year_data = data.sel(
            time=data.time.dt.year == year
        )  # select data for that year
        if year_data.size > 0:  # only plot if there is data for that year
            plot_timeline(
                da,  # original data
                data.sel(time=da.time.dt.year == year),  # data for that year
                f"{name} - {year.item()}",  # title
                axes[i + 1],  # axis to plot on
            )

    fp = self.report_dir / (
        name + "_timeline.png"
    )  # file path for saving the timeline plot
    fp.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    plt.savefig(fp)  # save the timeline plot
    plt.close(fig)  # close the figure to free memory

    spatial_data = da.mean(dim="time")  # mean over time for spatial plot

    spatial_data.plot()  # plot the spatial data

    plt.title(name)  # title
    plt.xlabel("Longitude")  # x-axis label
    plt.ylabel("Latitude")  # y-axis label

    spatial_fp: Path = self.report_dir / (
        name + "_spatial.png"
    )  # file path for saving the spatial plot
    plt.savefig(spatial_fp)  # save the spatial plot
    plt.close()  # close the plot to free memory


def plot_forecasts(self, da: xr.DataArray, name: str) -> None:
    """Plot forecast data with a temporal (timeline) plot and a spatial plot.

    Handles only ensemble forecasts for now. Makes a spatial plot for every single ensemble member.

    Args:
        da: The xarray DataArray containing the forecast data. Must have dimensions 'time', 'y', 'x', and 'member'.
        name: The name of the variable being plotted, used for titles and filenames.

    Returns: None

    """
    # pre-processing of plotting data
    mask = self.grid["mask"]  # get the GEB grid
    da_plot = da.copy()  # make a copy to avoid modifying the original data
    # Convert data to mm/hour if it's precipitation
    if "pr" in name.lower() and "kg m-2 s-1" in da_plot.attrs.get("units", ""):
        da_plot = da_plot * 3600  # convert to mm/hour
        ylabel = "mm/hour"  # set y-axis label
    else:
        da_plot = da_plot.copy()  # no conversion
        ylabel = da_plot.attrs.get("units", "")

    if "pr_hourly" in da_plot.name:
        da_plot = da_plot.interp(
            x=mask.x,
            y=mask.y,  # interp to GEB grid, if necessary
            method="linear",
        )
    n_members: int = da.sizes["member"]  # number of ensemble members

    # Timeline plot
    fig, ax_time = plt.subplots(1, 1, figsize=(12, 9))  # Create temporal plot

    colors = plt.cm.viridis(np.linspace(0, 1, n_members))  # Distinct colors for members

    spatial_average = (
        (da_plot * ~mask).sum(dim=("y", "x")) / (~mask).sum()
    ).compute()  # Area-weighted average

    ensemble_data = []  # Store ensemble member data
    for i, member in enumerate(spatial_average.member):  # Iterate over ensemble members
        member_avg = spatial_average.sel(member=member)  # Select member data
        ensemble_data.append(member_avg)  # Collect for ensemble mean

        ax_time.plot(  # plot member line
            member_avg.time,
            member_avg,
            color=colors[i],
            alpha=0.7,
            linewidth=1.2,
            label=f"Member {member.values}",
        )

    # Calculate ensemble mean and add to plot
    ensemble_mean = sum(ensemble_data) / len(ensemble_data)  # ensemble mean
    ax_time.plot(
        ensemble_mean.time,
        ensemble_mean,
        "k-",
        linewidth=3,
        label="Ensemble Mean",
    )  # plot ensemble mean
    ax_time.legend(
        bbox_to_anchor=(0.5, -0.2), loc="center", ncol=5, fontsize=8
    )  # legend
    ax_time.set_xlabel("Time")  # x-axis label
    ax_time.set_ylabel(ylabel)  # y-axis label
    ax_time.set_title(f"{name} - Ensemble Forecast Timeline")  # title
    ax_time.grid(True, alpha=0.3)  # light grid

    fp = self.report_dir / (name + "_ensemble_timeline.png")  # File path
    fp.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    plt.tight_layout()  # tight layout
    plt.savefig(fp, dpi=300, bbox_inches="tight")  # save figure
    plt.close(fig)  # close figure to free memory

    # Spatial plot (max over time))
    n_cols = min(6, n_members)  # Changed from 4 to 6 columns
    n_rows = (n_members + n_cols - 1) // n_cols  # Calculate rows needed

    # Create figure with cartopy projection
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )  # Create subplots with cartopy projection
    plt.subplots_adjust(
        hspace=0.2, wspace=0.2, top=0.95, bottom=0.05, left=0.05, right=0.85
    )  # Tighter spacing

    custom_cmap = plt.cm.Blues  # Use simple Blues colormap
    da_plot_max_over_time = da_plot.max(dim="time")  # max over time for color scale
    for i, member in enumerate(
        da_plot_max_over_time.member
    ):  # Iterate over ensemble members
        ax = axes.flatten()[i]  # Select subplot
        spatial_data = da_plot_max_over_time.sel(member=member)  # Select member data
        if "pr" in name.lower() and "kg m-2 s-1" in da_plot_max_over_time.attrs.get(
            "units", ""
        ):  # Convert spatial data to mm/hour if it's precipitation
            cbar_label = "mm/hour"  # units for colorbar
            vmin = 0  # minimum value for color scale
            vmax = 30  # maximum value for color scale, set to 30 mm/hour for better visualization
        else:
            cbar_label = da_plot_max_over_time.attrs.get(
                "units", ""
            )  # units for colorbar
            vmin = np.min(
                ensemble_data
            )  # min/max over all members for consistent color scale
            vmax = np.max(
                ensemble_data
            )  # min/max over all members for consistent color scale

        im = ax.pcolormesh(  # Plot spatial data
            spatial_data.x,
            spatial_data.y,
            spatial_data,
            cmap=custom_cmap,  # Use custom colormap
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_title(f"Member {member.values}")  # Title for each subplot
        ax.set_xlabel("Longitude")  # Longitude label
        ax.set_ylabel("Latitude")  # Latitude label
        ax.set_title(f"Member {member.values}")  # Title for each subplot
        ax.set_xlabel("Longitude")  # Longitude label
        ax.set_ylabel("Latitude")  # Latitude label
        # gridlines
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.2, color="gray", alpha=0.5
        )  # changed linewidth from 0.5 to 0.2
        gl.top_labels = False  # Remove top labels
        gl.right_labels = False  # Remove right labels
        gl.xlabel_style = {"size": 8}  # smaller font size
        gl.ylabel_style = {"size": 8}  # smaller font size
        # add coastlines and borders
        ax.add_feature(
            cfeature.COASTLINE, linewidth=0.5, color="black"
        )  # add coastlines
        ax.add_feature(
            cfeature.BORDERS, linewidth=0.5, color="gray"
        )  # add country borders

        # Add region shapefile boundary with thick line
        self.geom["mask"].boundary.plot(
            ax=ax, color="red", linewidth=3, transform=ccrs.PlateCarree()
        )

    fig.subplots_adjust(right=0.85)  # make space for colorbar
    cbar_ax = fig.add_axes((0.87, 0.15, 0.03, 0.7))  # (left, bottom, width, height)
    cbar = fig.colorbar(  # add colorbar
        im,
        cax=cbar_ax,
    )
    cbar.set_label(cbar_label)  # label for colorbar
    fig.suptitle(
        f"{name} - Ensemble Spatial Distribution (Max over Time)"
    )  # Overall title
    spatial_fp: Path = self.report_dir / (name + "_ensemble_spatial.png")  # File path
    plt.savefig(spatial_fp, dpi=300, bbox_inches="tight")  # Save figure
    plt.close(fig)  # Close figure to free memory


def _plot_data(self, da: xr.DataArray, name: str) -> None:
    """Plot data using appropriate method based on data type.

    Uses plot_forecasts if 'forecast' is in the name, otherwise uses plot_forcing.

    Args:
        da: Data to plot.
        name: Name for the plots and file outputs.
    """
    if "forecast" in name.lower():
        plot_forecasts(self, da, name)  # plot forecasts
    else:
        plot_forcing(self, da, name)  # plot historical forcing data


def plot_timeline(
    da: xr.DataArray, data: xr.DataArray, name: str, ax: plt.Axes
) -> None:
    """Plot a timeline of the data.

    Args:
        da: the original xarray DataArray containing the data to plot.
        data: the data to plot, should be a 1D xarray DataArray with a time dimension.
        name: the name of the data, used for the plot title.
        ax: the matplotlib axes to plot on.
    """
    ax.plot(data.time, data)
    ax.set_xlabel("Time")
    if "units" in da.attrs:
        ax.set_ylabel(da.attrs["units"])
    ax.set_xlim(data.time[0], data.time[-1])
    ax.set_ylim(data.min(), data.max() * 1.1)
    significant_digits: int = 6
    ax.set_title(
        f"{name} - mean: {data.mean().item():.{significant_digits}f} - min: {data.min().item():.{significant_digits}f} - max: {data.max().item():.{significant_digits}f}"
    )


def get_chunk_size(da, target: float | int = 1e8) -> int:
    """Calculate the optimal chunk size for the given xarray DataArray based on the target size.

    Args:
        da: The xarray DataArray for which to calculate the chunk size.
        target: The target size in bytes. Default is 1e8 (100 MB).

    Returns:
        The calculated chunk size in bytes.
    """
    return int(target / (da.dtype.itemsize * da.x.size * da.y.size))


class Forcing:
    """Contains methods to download and process climate forcing data for GEB."""

    def __init__(self) -> None:
        pass

    def download_isimip(
        self,
        product: str,
        variable: str,
        forcing: str | None = None,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
        simulation_round: str = "ISIMIP3a",
        climate_scenario: str = "obsclim",
        resolution: str | None = None,
        buffer: int = 0,
    ) -> xr.DataArray:
        """This method downloads ISIMIP climate data for GEB.

        It first retrieves the dataset
        metadata from the ISIMIP repository using the specified `product`, `variable`, `forcing`, and `resolution`
        parameters. It then downloads the data files that match the specified `start_date` and `end_date` parameters, and
        extracts them to the specified `download_path` directory.

        The resulting climate data is returned as an xarray dataset. The dataset is assigned the coordinate reference system
        EPSG:4326, and the spatial dimensions are set to 'lon' and 'lat'.

        Args:
            product: The name of the ISIMIP product to download.
            variable: The name of the climate variable to download.
            forcing: The name of the climate forcing to download.
            start_date: The start date of the data. Default is None.
            end_date: The end date of the data. Default is None.
            simulation_round: The ISIMIP simulation round to download data for. Default is "ISIMIP3a".
            climate_scenario: The climate scenario to download data for. Default is "obsclim".
            resolution: The resolution of the data to download. Default is None.
            buffer: The buffer size in degrees to add to the bounding box of the data to download. Default is 0.

        Returns:
            The downloaded climate data as an xarray dataset.

        Raises:
            ValueError: If no files are found for the specified variable in the ISIMIP dataset.
            ValueError: If the parse_files does not all end with either .nc or .txt.
            ValueError: If an unknown file type is encountered in the ISIMIP dataset.
            RuntimeError: If the ISIMIP server returns an unknown status during file download.
        """
        # if start_date is specified, end_date must be specified as well
        assert (start_date is None) == (end_date is None)

        if isinstance(start_date, datetime):
            start_date: date = start_date.date()
        if isinstance(end_date, datetime):
            end_date: date = end_date.date()

        client: ISIMIPClient = ISIMIPClient()
        download_path: Path = self.preprocessing_dir / "climate"
        if forcing is not None:
            download_path: Path = download_path / forcing
        download_path: Path = download_path / variable
        download_path.mkdir(parents=True, exist_ok=True)

        # Code to get data from disk rather than server.
        parse_files = []
        for file in os.listdir(download_path):
            if file.endswith(".nc"):
                fp = download_path / file
                parse_files.append(fp)

        # get the dataset metadata from the ISIMIP repository
        response = client.datasets(
            simulation_round=simulation_round,
            product=product,
            climate_forcing=forcing,
            climate_scenario=climate_scenario,
            climate_variable=variable,
            resolution=resolution,
        )
        assert len(response["results"]) == 1
        dataset = response["results"][0]
        files = dataset["files"]

        xmin, ymin, xmax, ymax = self.bounds
        xmin -= buffer
        ymin -= buffer
        xmax += buffer
        ymax += buffer

        if variable == "orog":
            assert len(files) == 1
            filename = files[
                0
            ][
                "name"
            ]  # global should be included due to error in ISIMIP API .replace('_global', '')
            parse_files = [filename]
            if not (download_path / filename).exists():
                download_files = [files[0]["path"]]
            else:
                download_files = []

        else:
            assert start_date is not None and end_date is not None
            download_files = []
            parse_files = []
            for file in files:
                name: str = file["name"]
                if name.endswith(".nc"):
                    splitted_filename = name.split("_")
                    dt = splitted_filename[-1].split(".")[0]
                    if "-" in dt:
                        file_start_date, file_end_date = date.split("-")
                        file_start_date = datetime.strptime(
                            file_start_date, "%Y%m%d"
                        ).date()
                        file_end_date = datetime.strptime(
                            file_end_date, "%Y%m%d"
                        ).date()
                    elif len(dt) == 6:
                        file_start_date = datetime.strptime(dt, "%Y%m").date()
                        file_end_date = (
                            file_start_date
                            + relativedelta(months=1)
                            - relativedelta(days=1)
                        )
                    elif len(dt) == 4:  # is year
                        assert splitted_filename[-2].isdigit()
                        file_start_date = datetime.strptime(
                            splitted_filename[-2], "%Y"
                        ).date()
                        file_end_date = date(int(dt), 12, 31)
                    else:
                        raise ValueError(f"could not parse date {dt} from file {name}")

                    if not (file_end_date < start_date or file_start_date > end_date):
                        parse_files.append(file["name"].replace("_global", ""))
                        if not (
                            download_path / file["name"].replace("_global", "")
                        ).exists():
                            download_files.append(file["path"])
                elif name.endswith(".txt"):
                    parse_files.append(name)
                    if not (download_path / name).exists():
                        download_files.append(file["path"])
                else:
                    raise ValueError(f"Unknown file type {name} in ISIMIP dataset")

        if not parse_files:
            raise ValueError(
                f"No files found for variable {variable} in ISIMIP dataset {dataset['id']}"
            )

        all_nc = all(str(f).endswith(".nc") for f in parse_files)
        all_txt = all(str(f).endswith(".txt") for f in parse_files)
        assert all_nc or all_txt, "All parse_files must end with either .nc or .txt"
        if all_txt:
            assert len(parse_files) == 1, "Only one .txt file is expected"

        if download_files:
            self.logger.info(f"Requesting download of {len(download_files)} files")
            if all_nc:
                while True:
                    try:
                        response = client.cutout(
                            download_files, [ymin, ymax, xmin, xmax]
                        )
                    except requests.exceptions.HTTPError:
                        self.logger.warning(
                            "HTTPError, could not download files, retrying in 60 seconds"
                        )
                    else:
                        if response["status"] == "finished":
                            break
                        elif response["status"] == "started":
                            self.logger.info(
                                f"{response['meta']['created_files']}/{response['meta']['total_files']} files prepared on ISIMIP server for {variable}, waiting 60 seconds before retrying"
                            )
                        elif response["status"] == "queued":
                            self.logger.info(
                                f"Data preparation queued for {variable} on ISIMIP server, waiting 60 seconds before retrying"
                            )
                        elif response["status"] == "failed":
                            self.logger.info(
                                "ISIMIP internal server error, waiting 60 seconds before retrying"
                            )
                        else:
                            raise RuntimeError(
                                f"Could not download files: {response['status']}"
                            )
                    time.sleep(60)

                self.logger.info(f"Starting download of files for {variable}")
                # download the file when it is ready
                client.download(
                    response["file_url"],
                    path=download_path,
                    validate=False,
                    extract=False,
                )

                self.logger.info(f"Download finished for {variable}")
                # remove zip file
                zip_file = download_path / Path(
                    urlparse(response["file_url"]).path.split("/")[-1]
                )
                # make sure the file exists
                assert zip_file.exists()
                # Open the zip file
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    # Get a list of all the files in the zip file
                    file_list = [f for f in zip_ref.namelist() if f.endswith(".nc")]
                    # Extract each file one by one
                    for i, file_name in enumerate(file_list):
                        # Rename the file
                        bounds_str = ""
                        if isinstance(ymin, float):
                            bounds_str += f"_lat{ymin}"
                        else:
                            bounds_str += f"_lat{ymin:.1f}"
                        if isinstance(ymax, float):
                            bounds_str += f"to{ymax}"
                        else:
                            bounds_str += f"to{ymax:.1f}"
                        if isinstance(xmin, float):
                            bounds_str += f"lon{xmin}"
                        else:
                            bounds_str += f"lon{xmin:.1f}"
                        if isinstance(xmax, float):
                            bounds_str += f"to{xmax}"
                        else:
                            bounds_str += f"to{xmax:.1f}"
                        assert bounds_str in file_name
                        new_file_name = file_name.replace(bounds_str, "")
                        zip_ref.getinfo(file_name).filename = new_file_name
                        # Extract the file
                        if os.name == "nt":
                            max_file_path_length = 260
                        else:
                            max_file_path_length = os.pathconf("/", "PC_PATH_MAX")
                        assert (
                            len(str(download_path / new_file_name))
                            <= max_file_path_length
                        ), (
                            f"File path too long: {download_path / zip_ref.getinfo(file_name).filename}"
                        )
                        zip_ref.extract(file_name, path=download_path)
                # remove zip file
                (
                    download_path
                    / Path(urlparse(response["file_url"]).path.split("/")[-1])
                ).unlink()

            else:
                client.download(
                    "https://files.isimip.org/" + download_files[0],
                    path=download_path,
                )

        if all_nc:
            datasets: list[xr.Dataset] = [
                xr.open_dataset(download_path / file, chunks={}) for file in parse_files
            ]
            for dataset in datasets:
                assert "lat" in dataset.coords and "lon" in dataset.coords

            # make sure y is decreasing rather than increasing
            datasets: list[xr.Dataset] = [
                (
                    dataset.reindex(lat=dataset.lat[::-1])
                    if dataset.lat[0] < dataset.lat[-1]
                    else dataset
                )
                for dataset in datasets
            ]

            reference = datasets[0]
            for dataset in datasets:
                # make sure all datasets have more or less the same coordinates
                assert np.isclose(
                    dataset.coords["lat"].values,
                    reference["lat"].values,
                    atol=abs(datasets[0].rio.resolution()[1] / 50),
                    rtol=0,
                ).all()
                assert np.isclose(
                    dataset.coords["lon"].values,
                    reference["lon"].values,
                    atol=abs(datasets[0].rio.resolution()[0] / 50),
                    rtol=0,
                ).all()

            datasets: list[xr.Dataset] = [
                ds.assign_coords(
                    lon=reference["lon"].values, lat=reference["lat"].values
                )
                for ds in datasets
            ]
            if len(datasets) > 1:
                ds: xr.Dataset = xr.concat(datasets, dim="time")
            else:
                ds: xr.Dataset = datasets[0]

            if start_date is not None:
                ds = ds.sel(time=slice(start_date, end_date))
                # assert that time is monotonically increasing with a constant step size
                assert (
                    ds.time.diff("time").astype(np.int64)
                    == (ds.time[1] - ds.time[0]).astype(np.int64)
                ).all()

            ds.raster.set_spatial_dims(x_dim="lon", y_dim="lat")
            assert not ds.lat.attrs, "lat already has attributes"
            assert not ds.lon.attrs, "lon already has attributes"
            ds.lat.attrs = {
                "long_name": "latitude of grid cell center",
                "units": "degrees_north",
            }
            ds.lon.attrs = {
                "long_name": "longitude of grid cell center",
                "units": "degrees_east",
            }
            ds.raster.set_crs(4326)

            ds: xr.Dataset = ds.rename({"lon": "x", "lat": "y"})

            # check whether data is for noon or midnight. If noon, subtract 12 hours from time coordinate to align with other datasets
            if hasattr(ds, "time") and pd.to_datetime(ds.time[0].values).hour == 12:
                # subtract 12 hours from time coordinate
                self.logger.warning(
                    "Subtracting 12 hours from time coordinate to align climate datasets"
                )
                ds: xr.Dataset = ds.assign_coords(
                    time=ds.time - np.timedelta64(12, "h")
                )
            return ds[variable]

        elif all_txt:
            assert len(parse_files) == 1, "Only one .txt file is expected"
            df = pd.read_csv(
                download_path / parse_files[0], sep=r"\s+", names=["year", variable]
            ).set_index("year")
            df = df[(df.index >= start_date.year) & (df.index <= end_date.year)]
            assert len(df.columns) == 1, "Only one column expected in .txt file"
            # convert df to xarray DataArray
            da = xr.DataArray(
                df[variable].values,
                coords={
                    "time": pd.date_range(
                        start=f"{df.index[0]}-01-01",
                        end=f"{df.index[-1]}-12-31",
                        freq="YS",
                    )
                },
                dims=["time"],
            )
            return da
        else:
            raise ValueError(
                "All parse_files must end with either .nc or .txt, but got: "
                f"{parse_files}"
            )

    def set_xy_attrs(self, da: xr.DataArray) -> None:
        """Set CF-compliant attributes for the x and y coordinates of a DataArray."""
        da.x.attrs = {"long_name": "longitude", "units": "degrees_east"}
        da.y.attrs = {"long_name": "latitude", "units": "degrees_north"}

    def set_pr_hourly(
        self,
        da: xr.DataArray,
        name: str = "climate/pr_hourly",
        *args: Any,
        **kwargs: Any,
    ) -> xr.DataArray:
        da.attrs = {
            "standard_name": "precipitation_flux",
            "long_name": "Precipitation",
            "units": "kg m-2 s-1",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0

        # maximum rainfall in one hour was 304.8 mm in 1956 in Holt, Missouri, USA
        # https://www.guinnessworldrecords.com/world-records/737965-greatest-rainfall-in-one-hour
        # we take a wide margin of 500 mm/h
        max_value = 500 / 3600  # convert to kg/m2/s
        precision = 0.01 / 3600  # 0.01 mm in kg/m2/s

        scaling_factor, out_dtype = calculate_scaling(
            0, max_value, offset=offset, precision=precision
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            time_chunksize=7 * 24,
            filters=filters,
        )

        # plot data (only forecasts)
        if "forecast" in name.lower():
            _plot_data(self, da, name)

        return da

    def set_pr(
        self, da: xr.DataArray, name: str = "climate/pr", *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        da.attrs = {
            "standard_name": "precipitation_flux",
            "long_name": "Precipitation",
            "units": "kg m-2 s-1",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        # maximum rainfall in one hour was 304.8 mm in 1956 in Holt, Missouri, USA
        # https://www.guinnessworldrecords.com/world-records/737965-greatest-rainfall-in-one-hour
        # we take a wide margin of 500 mm/h
        # this function is currently daily, so the hourly value should be save
        max_value: float = 500 / 3600  # convert to kg/m2/s
        precision: float = 0.01 / 3600  # 0.01 mm in kg/m2/s

        offset: int = 0
        scaling_factor, out_dtype = calculate_scaling(
            0, max_value, offset=offset, precision=precision
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self._mask_forcing(da, value=-offset)
        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        _plot_data(self, da, name)
        return da

    def set_rsds(
        self, da: xr.DataArray, name: str = "climate/rsds", *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        da.attrs = {
            "standard_name": "surface_downwelling_shortwave_flux_in_air",
            "long_name": "Surface Downwelling Shortwave Radiation",
            "units": "W m-2",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0
        scaling_factor, out_dtype = calculate_scaling(
            0, 1361, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        _plot_data(self, da, name)
        return da

    def set_rlds(
        self, da: xr.DataArray, name: str = "climate/rlds", *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        da.attrs = {
            "standard_name": "surface_downwelling_longwave_flux_in_air",
            "long_name": "Surface Downwelling Longwave Radiation",
            "units": "W m-2",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0
        scaling_factor, out_dtype = calculate_scaling(
            0, 1361, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        _plot_data(self, da, name)
        return da

    def set_tas(
        self, da: xr.DataArray, name: str = "climate/tas", *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        da.attrs = {
            "standard_name": "air_temperature",
            "long_name": "Near-Surface Air Temperature",
            "units": "K",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        K_to_C = 273.15
        offset = -15 - K_to_C  # average temperature on earth
        scaling_factor, out_dtype = calculate_scaling(
            -100 + K_to_C, 60 + K_to_C, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )

        _plot_data(self, da, name)
        return da

    def set_tasmax(
        self, da: xr.DataArray, name: str = "climate/tasmax", *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        da.attrs = {
            "standard_name": "air_temperature",
            "long_name": "Daily Maximum Near-Surface Air Temperature",
            "units": "K",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        K_to_C = 273.15
        offset = -15 - K_to_C  # average temperature on earth
        scaling_factor, out_dtype = calculate_scaling(
            -100 + K_to_C, 60 + K_to_C, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        _plot_data(self, da, name)
        return da

    def set_tasmin(
        self, da: xr.DataArray, name: str = "climate/tasmin", *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        da.attrs = {
            "standard_name": "air_temperature",
            "long_name": "Daily Minimum Near-Surface Air Temperature",
            "units": "K",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        K_to_C = 273.15
        offset = -15 - K_to_C  # average temperature on earth
        scaling_factor, out_dtype = calculate_scaling(
            -100 + K_to_C, 60 + K_to_C, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        _plot_data(self, da, name)
        return da

    def set_hurs(
        self, da: xr.DataArray, name: str = "climate/hurs", *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        da.attrs = {
            "standard_name": "relative_humidity",
            "long_name": "Near-Surface Relative Humidity",
            "units": "%",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = -50
        scaling_factor, out_dtype = calculate_scaling(
            0, 100, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        _plot_data(self, da, name)
        return da

    def _mask_forcing(self, da: xr.DataArray, value: int | float) -> xr.DataArray:
        """Mask the forcing data where the grid mask is False.

        Args:
            da: DataArray to mask.
            value: Value to use for masking where the grid mask is False.

        Returns:
            DataArray with replaced values with the same dimensions as the input DataArray.
        """
        da_: xr.DataArray = xr.where(~self.grid["mask"], da, value, keep_attrs=True)
        da_: xr.DataArray = da_.rio.write_crs(da.rio.crs)
        # restore the original dimensions order which can be changed by the mask operation
        da: xr.DataArray = da_.transpose(*da.dims)
        return da

    def set_ps(
        self, da: xr.DataArray, name: str = "climate/ps", *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        da.attrs = {
            "standard_name": "surface_air_pressure",
            "long_name": "Surface Air Pressure",
            "units": "Pa",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = -100_000
        scaling_factor, out_dtype = calculate_scaling(
            30_000, 120_000, offset=offset, precision=10
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        _plot_data(self, da, name)
        return da

    def set_sfcwind(
        self, da: xr.DataArray, name: str = "climate/sfcwind", *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        da.attrs = {
            "standard_name": "wind_speed",
            "long_name": "Near-Surface Wind Speed",
            "units": "m s-1",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0
        scaling_factor, out_dtype = calculate_scaling(
            0, 120, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        _plot_data(self, da, name)
        return da

    def set_SPEI(
        self, da: xr.DataArray, name: str = "climate/SPEI", *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        da.attrs = {
            "units": "-",
            "long_name": "Standard Precipitation Evapotranspiration Index",
            "name": "spei",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        # this range corresponds to probabilities of lower than 0.001 and higher than 0.999
        # which should be considered non-significant
        min_SPEI = -3.09
        max_SPEI = 3.09
        da = da.clip(min=min_SPEI, max=max_SPEI)

        offset = 0
        scaling_factor, out_dtype = calculate_scaling(
            min_SPEI, max_SPEI, offset=offset, precision=0.001
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        _plot_data(self, da, name)
        return da

    def setup_forcing_ERA5(self) -> None:
        target = self.grid["mask"]
        target.raster.set_crs(4326)

        download_args: dict[str, Any] = {
            "folder": self.preprocessing_dir / "climate" / "ERA5",
            "start_date": self.start_date
            - relativedelta(
                years=1
            ),  # include one year before the start date for SPEI calculation
            "end_date": self.end_date,
            "bounds": target.raster.bounds,
            "logger": self.logger,
        }

        pr_hourly = process_ERA5(
            "tp",  # total_precipitation
            **download_args,
        )

        elevation_forcing, elevation_target = self.get_elevation_forcing_and_grid(
            self.grid["mask"], pr_hourly, forcing_name="ERA5"
        )

        pr_hourly = pr_hourly * (1000 / 3600)  # convert from m/hr to kg/m2/s

        # ensure no negative values for precipitation, which may arise due to float precision
        pr_hourly = xr.where(pr_hourly > 0, pr_hourly, 0, keep_attrs=True)
        pr_hourly = self.set_pr_hourly(pr_hourly)  # weekly chunk size

        pr = pr_hourly.resample(time="D").mean()  # get daily mean
        pr = resample_like(pr, target, method="conservative")
        pr = self.set_pr(pr)

        hourly_tas = process_ERA5("t2m", **download_args)
        tas_avg = hourly_tas.resample(time="D").mean()
        tas_avg = reproject_and_apply_lapse_rate_temperature(
            tas_avg, elevation_forcing, elevation_target
        )

        self.set_tas(tas_avg)

        tasmax = hourly_tas.resample(time="D").max()
        tasmax = reproject_and_apply_lapse_rate_temperature(
            tasmax, elevation_forcing, elevation_target
        )
        self.set_tasmax(tasmax)

        tasmin = hourly_tas.resample(time="D").min()
        tasmin = reproject_and_apply_lapse_rate_temperature(
            tasmin, elevation_forcing, elevation_target
        )
        self.set_tasmin(tasmin)

        hourly_dew_point_tas = process_ERA5(
            "d2m",
            **download_args,
        )
        dew_point_tas = hourly_dew_point_tas.resample(time="D").mean()
        dew_point_tas = reproject_and_apply_lapse_rate_temperature(
            dew_point_tas, elevation_forcing, elevation_target
        )

        water_vapour_pressure = 0.6108 * np.exp(
            17.27 * (dew_point_tas - 273.15) / (237.3 + (dew_point_tas - 273.15))
        )  # calculate water vapour pressure (kPa)
        saturation_vapour_pressure = 0.6108 * np.exp(
            17.27 * (tas_avg - 273.15) / (237.3 + (tas_avg - 273.15))
        )

        assert water_vapour_pressure.shape == saturation_vapour_pressure.shape
        relative_humidity = (water_vapour_pressure / saturation_vapour_pressure) * 100

        original_crs = (
            relative_humidity.rio.crs if hasattr(relative_humidity, "rio") else None
        )

        # convert values between 100 and 101 to 100, leave others unchanged
        relative_humidity = xr.where(
            (relative_humidity > 100) & (relative_humidity <= 101),
            100,
            relative_humidity,
            keep_attrs=True,
        )
        if original_crs is not None and (
            not hasattr(relative_humidity, "rio") or relative_humidity.rio.crs is None
        ):
            relative_humidity = relative_humidity.rio.write_crs(original_crs)

        self.set_hurs(relative_humidity)

        hourly_rsds = process_ERA5(
            "ssrd",  # surface_solar_radiation_downwards
            **download_args,
        )
        rsds = hourly_rsds.resample(time="D").sum() / (
            24 * 3600
        )  # get daily sum and convert from J/m2 to W/m2

        rsds = resample_like(rsds, target, method="conservative")
        self.set_rsds(rsds)

        hourly_rlds = process_ERA5(
            "strd",  # surface_thermal_radiation_downwards
            **download_args,
        )
        rlds = hourly_rlds.resample(time="D").sum() / (24 * 3600)
        rlds = resample_like(rlds, target, method="conservative")
        self.set_rlds(rlds)

        pressure = process_ERA5("sp", **download_args)
        pressure = pressure.resample(time="D").mean()
        pressure = reproject_and_apply_lapse_rate_pressure(
            pressure, elevation_forcing, elevation_target
        )

        self.set_ps(pressure)

        u_wind = process_ERA5(
            "u10",
            **download_args,
        )
        u_wind = u_wind.resample(time="D").mean()

        v_wind = process_ERA5(
            "v10",
            **download_args,
        )
        v_wind = v_wind.resample(time="D").mean()
        wind_speed = np.sqrt(u_wind**2 + v_wind**2)
        wind_speed = resample_like(wind_speed, target, method="conservative")
        self.set_sfcwind(wind_speed)

    def setup_forcing_ISIMIP(self, resolution_arcsec: int, model: str) -> None:
        """Sets up the forcing data for GEB using ISIMIP data.

        Args:
            resolution_arcsec: The resolution of the data in arcseconds. Supported values are 30 and 1800.
            model: The forcing data to use. Supported values are 'chelsa-w5e5' for 30 arcsec resolution
                and ipsl-cm6a-lr, gfdl-esm4, mpi-esm1-2-hr, mri-esm2-0, and mri-esm2-0 for 1800 arcsec resolution.

        Raises:
            ValueError: If an unsupported resolution or model is specified.
        """
        if resolution_arcsec == 30:
            assert model == "chelsa-w5e5", (
                "Only chelsa-w5e5 is supported for 30 arcsec resolution"
            )
            # download source data from ISIMIP
            self.logger.info("setting up forcing data")
            high_res_variables: list = ["pr", "rsds", "tas", "tasmax", "tasmin"]
            self.setup_30arcsec_variables_isimip(high_res_variables)
            self.logger.info("setting up relative humidity...")
            self.setup_hurs_isimip_30arcsec()
            self.logger.info("setting up longwave radiation...")
            self.setup_longwave_isimip_30arcsec()
            self.logger.info("setting up pressure...")
            self.setup_pressure_isimip_30arcsec()
            self.logger.info("setting up wind...")
            self.setup_wind_isimip_30arcsec()
        elif resolution_arcsec == 1800:
            assert model in (
                "ipsl-cm6a-lr",
                "gfdl-esm4",
                "mpi-esm1-2-hr",
                "mri-esm2-0",
                "ukesm1-0-ll",
            ), (
                "Only ipsl-cm6a-lr, gfdl-esm4, mpi-esm1-2-hr, mri-esm2-0 and ukesm1-0-ll are supported for 1800 arcsec resolution"
            )
            variables = [
                "pr",
                "rsds",
                "tas",
                "tasmax",
                "tasmin",
                "hurs",
                "rlds",
                "ps",
                "sfcwind",
            ]
            self.setup_1800arcsec_variables_isimip(model, variables)
        else:
            raise ValueError(
                "Only 30 arcsec and 1800 arcsec resolution is supported for ISIMIP data"
            )

    @build_method(depends_on=["set_ssp", "set_time_range"])
    def setup_forcing(
        self,
        forcing: str = "ERA5",
        resolution_arcsec: int | None = None,
        model: str | None = None,
    ) -> None:
        """Sets up the forcing data for GEB.

        Args:
            forcing: The data source to use for the forcing data. Can be ERA5 or ISIMIP. Default is 'era5'.
            resolution_arcsec: The resolution of the data in arcseconds. Only used for ISIMIP. Supported values are 30 and 1800.
            model: The name of the forcing data to use within the dataset. Only required for ISIMIP data.
                For ISIMIP, this can be 'chelsa-w5e5' for 30 arcsec resolution
                or 'ipsl-cm6a-lr', 'gfdl-esm4', 'mpi-esm1-2-hr', 'mri-esm2-0', or 'mri-esm2-0' for 1800 arcsec resolution.

        Notes:
            This method sets up the forcing data for GEB. It first downloads the high-resolution variables
            (precipitation, surface solar radiation, air temperature, maximum air temperature, and minimum air temperature) from
            the ISIMIP dataset for the specified time period.

            The method then sets up the relative humidity, longwave radiation, pressure, and wind data for the model. The
            relative humidity data is downloaded from the ISIMIP dataset using the `setup_hurs_isimip_30arcsec` method. The longwave radiation
            data is calculated using the air temperature and relative humidity data and the `calculate_longwave` function. The
            pressure data is downloaded from the ISIMIP dataset using the `setup_pressure_isimip_30arcsec` method. The wind data is downloaded
            from the ISIMIP dataset using the `setup_wind_isimip_30arcsec` method. All these data are first downscaled to the model grid.

            The resulting forcing data is set as forcing data in the model with names of the form 'forcing/{variable_name}'.

        Raises:
            ValueError: If an unknown data source is specified.
        """
        if forcing == "ISIMIP":
            assert resolution_arcsec is not None, (
                "resolution_arcsec must be specified for ISIMIP forcing data"
            )
            assert model is not None, "model must be specified for ISIMIP forcing data"
            self.setup_forcing_ISIMIP(resolution_arcsec, model)
        elif forcing == "ERA5":
            assert resolution_arcsec is None, (
                "resolution_arcsec must be None for ERA5 forcing data"
            )
            assert model is None, "model must be None for ERA5 forcing data"
            self.setup_forcing_ERA5()
        elif forcing == "CMIP":
            raise NotImplementedError("CMIP forcing data is not yet supported")
        else:
            raise ValueError(
                f"Unknown data source: {forcing}, supported are 'ISIMIP' and 'ERA5'"
            )

    def construct_ISIMIP_variable(
        self, variable_name: str, forcing: str | None, ssp: str
    ) -> xr.DataArray:
        self.logger.info(f"Setting up {variable_name}...")
        first_year_future_climate: int = 2015
        da_list: list = []
        if ssp == "picontrol":
            da: xr.DataArray = self.download_isimip(
                product="InputData",
                simulation_round="ISIMIP3b",
                climate_scenario=ssp,
                variable=variable_name,
                start_date=self.start_date,
                end_date=self.end_date,
                forcing=forcing,
                resolution=None,
                buffer=1,
            )
            da_list.append(da)

        if (
            (
                self.end_date.year < first_year_future_climate
                or self.start_date.year < first_year_future_climate
            )
            and ssp != "picontrol"
        ):  # isimip cutoff date between historic and future climate
            da: xr.DataArray = self.download_isimip(
                product="InputData",
                simulation_round="ISIMIP3b",
                climate_scenario="historical",
                variable=variable_name,
                start_date=self.start_date,
                end_date=self.end_date,
                forcing=forcing,
                resolution=None,
                buffer=1,
            )
            da_list.append(da)
        if (
            self.start_date.year >= first_year_future_climate
            or self.end_date.year >= first_year_future_climate
        ) and ssp != "picontrol":
            assert ssp is not None, "ssp must be specified for future climate"
            assert ssp != "historical", "historical scenarios run until 2014"
            da: xr.DataArray = self.download_isimip(
                product="InputData",
                simulation_round="ISIMIP3b",
                climate_scenario=ssp,
                variable=variable_name,
                start_date=self.start_date,
                end_date=self.end_date,
                forcing=forcing,
                resolution=None,
                buffer=1,
            )
            da_list.append(da)

        da: xr.DataArray = xr.concat(
            da_list, dim="time", combine_attrs="drop_conflicts", compat="equals"
        )  # all values and dimensions must be the same

        step_size: np.int64 = (da.time[1] - da.time[0]).astype(np.int64)

        # assert that time is monotonically increasing with a constant step size
        # check if step size is yearly
        if step_size == 365 * 24 * 3600 * 1e9 or step_size == 366 * 24 * 3600 * 1e9:
            assert (da.time.dt.year.diff("time").astype(np.int64) == 1).all(), (
                "time is not monotonically increasing with a constant step size"
            )
        else:  # if not check if step size is constant
            assert (
                da.time.diff("time").astype(np.int64)
                == (da.time[1] - da.time[0]).astype(np.int64)
            ).all(), "time is not monotonically increasing with a constant step size"

        if variable_name in ("co2",):
            pass

        elif variable_name in ("tas", "tasmin", "tasmax", "ps"):
            elevation_forcing, elevation_grid = self.get_elevation_forcing_and_grid(
                self.grid["mask"], da, forcing_name="ISIMIP"
            )

            if variable_name in ("tas", "tasmin", "tasmax"):
                da: xr.DataArray = reproject_and_apply_lapse_rate_temperature(
                    da, elevation_forcing, elevation_grid
                )
            elif variable_name == "ps":
                da: xr.DataArray = reproject_and_apply_lapse_rate_pressure(
                    da, elevation_forcing, elevation_grid
                )
            else:
                raise ValueError
        else:
            target: xr.DataArray = self.grid["mask"]
            da: xr.DataArray = resample_like(
                da, target, method="conservative"
            )  # resample to the model grid
        da.attrs["_FillValue"] = np.nan
        self.logger.info(f"Completed {variable_name}")
        return da

    def setup_1800arcsec_variables_isimip(
        self,
        forcing: str,
        variables: List[str],
    ) -> None:
        """Sets up the high-resolution climate variables for GEB.

        Args:
            forcing: The forcing data to use. Supported values are 'ipsl-cm6a-lr', 'gfdl-esm4', 'mpi-esm1-2-hr', 'mri-esm2-0', or 'mri-esm2-0'.
            variables: The list of climate variables to set up.

        Notes:
            This method sets up the high-resolution climate variables for GEB. It downloads the specified
            climate variables from the ISIMIP dataset for the specified time period. The data is downloaded using the
            `download_isimip` method.

            The method renames the longitude and latitude dimensions of the downloaded data to 'x' and 'y', respectively. It
            then clips the data to the bounding box of the model grid using the `clip_bbox` method of the `raster` object.

            The resulting climate variables are set as forcing data in the model with names of the form 'climate/{variable_name}'.
        """
        for variable in variables:
            da: xr.DataArray = self.construct_ISIMIP_variable(
                variable, forcing, self.ISIMIP_ssp
            )
            getattr(self, f"set_{variable}")(da)

    def setup_30arcsec_variables_isimip(self, variables: List[str]) -> None:
        """Sets up the high-resolution climate variables for GEB.

        Args:
            variables: The list of climate variables to set up.

        Notes:
            This method sets up the high-resolution climate variables for GEB. It downloads the specified
            climate variables from the ISIMIP dataset for the specified time period. The data is downloaded using the
            `download_isimip` method.

            The method renames the longitude and latitude dimensions of the downloaded data to 'x' and 'y', respectively. It
            then clips the data to the bounding box of the model grid using the `clip_bbox` method of the `raster` object.

            The resulting climate variables are set as forcing data in the model with names of the form 'climate/{variable_name}'.
        """

        def download_variable(variable) -> None:
            self.logger.info(f"Setting up {variable}...")
            ds: xr.DataArray = self.download_isimip(
                product="InputData",
                variable=variable,
                start_date=self.start_date,
                end_date=self.end_date,
                forcing="chelsa-w5e5",
                resolution="30arcsec",
            )
            var: xr.DataArray = ds[variable].raster.clip_bbox(ds.raster.bounds)
            # TODO: Due to the offset of the MERIT grid, the snapping to the MERIT grid is not perfect
            # and thus the snapping needs to consider quite a large tollerance. We can consider interpolating
            # or this may be fixed when the we go to HydroBASINS v2
            var: xr.DataArray = self.snap_to_grid(
                var, self.grid, relative_tollerance=0.07
            )
            var.attrs["_FillValue"] = np.nan
            self.logger.info(f"Completed {variable}")
            getattr(self, f"set_{variable}")(var)

        for variable in variables:
            download_variable(variable)

    def setup_hurs_isimip_30arcsec(self) -> None:
        """Sets up the relative humidity data for GEB.

        Notes:
            This method sets up the relative humidity data for GEB. It first downloads the relative humidity
            data from the ISIMIP dataset for the specified time period using the `download_isimip` method. The data is downloaded
            at a 30 arcsec resolution.

            The method then downloads the monthly CHELSA-BIOCLIM+ relative humidity data at 30 arcsec resolution from the data
            catalog. The data is downloaded for each month in the specified time period and is clipped to the bounding box of
            the downloaded relative humidity data using the `clip_bbox` method of the `raster` object.

            The original ISIMIP data is then downscaled using the monthly CHELSA-BIOCLIM+ data. The downscaling method is adapted
            from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

            The resulting relative humidity data is set as forcing data in the model with names of the form 'climate/hurs'.
        """
        hurs_30_min: xr.DataArray = self.download_isimip(
            product="SecondaryInputData",
            variable="hurs",
            start_date=self.start_date,
            end_date=self.end_date,
            forcing="w5e5v2.0",
            buffer=1,
        )["hurs"]  # some buffer to avoid edge effects / errors in ISIMIP API

        # just taking the years to simplify things
        start_year: int = self.start_date.year
        end_year: int = self.end_date.year

        chelsa_folder: Path = (
            self.preprocessing_dir / "climate" / "chelsa-bioclim+" / "hurs"
        )
        chelsa_folder.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Downloading/reading monthly CHELSA-BIOCLIM+ hurs data at 30 arcsec resolution"
        )
        hurs_ds_30sec: list[xr.DataArray] = []
        hurs_time: list[str] = []
        for year in tqdm(range(start_year, end_year + 1)):
            for month in range(1, 13):
                fp: Path = chelsa_folder / f"hurs_{year}_{month:02d}.zarr"
                if not fp.exists():
                    url: str = self.data_catalog.get_source(
                        f"CHELSA-BIOCLIM+_monthly_hurs_{month:02d}_{year}"
                    ).path
                    hurs: xr.DataArray = rxr.open_rasterio(
                        url,
                    ).isel(band=0)
                    hurs: xr.DataArray = hurs.isel(
                        get_window(hurs.x, hurs.y, hurs_30_min.raster.bounds, buffer=1)
                    )

                    hurs: xr.DataArray = to_zarr(hurs, fp, crs=4326, progress=False)
                else:
                    hurs: xr.DataArray = open_zarr(fp)
                hurs_ds_30sec.append(hurs)
                hurs_time.append(f"{year}-{month:02d}")

        hurs_ds_30sec: xr.DataArray = xr.concat(hurs_ds_30sec, dim="time")
        hurs_ds_30sec["time"] = pd.date_range(hurs_time[0], hurs_time[-1], freq="MS")

        hurs_output: xr.DataArray = self.full_like(
            self.other["climate/tas"], fill_value=np.nan, nodata=np.nan
        )
        hurs_ds_30sec.raster.set_crs(4326)

        for year in tqdm(range(start_year, end_year + 1)):
            for month in range(1, 13):
                start_month = datetime(year, month, 1)
                end_month = datetime(year, month, monthrange(year, month)[1])

                w5e5_30min_sel = hurs_30_min.sel(time=slice(start_month, end_month))
                w5e5_regridded = resample_like(
                    w5e5_30min_sel, hurs_ds_30sec, method="conservative"
                )
                w5e5_regridded = w5e5_regridded * 0.01  # convert to fraction

                w5e5_regridded_mean = w5e5_regridded.mean(
                    dim="time"
                )  # get monthly mean
                w5e5_regridded_tr = np.log(
                    w5e5_regridded / (1 - w5e5_regridded)
                )  # assume beta distribuation => logit transform
                w5e5_regridded_mean_tr = np.log(
                    w5e5_regridded_mean / (1 - w5e5_regridded_mean)
                )  # logit transform

                chelsa = (
                    hurs_ds_30sec.sel(time=start_month) * 0.0001
                )  # convert to fraction

                chelsa_tr = np.log(
                    chelsa / (1 - chelsa)
                )  # assume beta distribuation => logit transform

                difference = chelsa_tr - w5e5_regridded_mean_tr

                # apply difference to w5e5
                w5e5_regridded_tr_corr = w5e5_regridded_tr + difference
                w5e5_regridded_corr = (
                    1 / (1 + np.exp(-w5e5_regridded_tr_corr))
                ) * 100  # back transform
                w5e5_regridded_corr.raster.set_crs(4326)
                w5e5_regridded_corr_clipped = w5e5_regridded_corr.raster.clip_bbox(
                    hurs_output.raster.bounds
                )

                hurs_output.loc[dict(time=slice(start_month, end_month))] = (
                    self.snap_to_grid(
                        w5e5_regridded_corr_clipped,
                        hurs_output,
                        relative_tollerance=0.07,
                    )
                )

        self.set_hurs(hurs_output)

    def setup_longwave_isimip_30arcsec(self) -> None:
        """Sets up the longwave radiation data for GEB.

        Notes:
            This method sets up the longwave radiation data for GEB. It first downloads the relative humidity,
            air temperature, and downward longwave radiation data from the ISIMIP dataset for the specified time period using the
            `download_isimip` method. The data is downloaded at a 30 arcsec resolution.

            The method then regrids the downloaded data to the target grid using the `xe.Regridder` method. It calculates the
            saturation vapor pressure, water vapor pressure, clear-sky emissivity, all-sky emissivity, and cloud-based component
            of emissivity for the coarse and fine grids. It then downscales the longwave radiation data for the fine grid using
            the calculated all-sky emissivity and Stefan-Boltzmann constant. The downscaling method is adapted
            from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

            The resulting longwave radiation data is set as forcing data in the model with names of the form 'climate/rlds'.
        """
        x1: float = 0.43
        x2: float = 5.7
        sbc: float = 5.67e-8  # stefan boltzman constant [Js−1 m−2 K−4]

        es0: float = 6.11  # reference saturation vapour pressure  [hPa]
        T0: float = 273.15
        lv: float = 2.5e6  # latent heat of vaporization of water
        Rv: float = 461.5  # gas constant for water vapour [J K kg-1]

        hurs_coarse: xr.DataArray = self.download_isimip(
            product="SecondaryInputData",
            variable="hurs",
            start_date=self.start_date,
            end_date=self.end_date,
            forcing="w5e5v2.0",
            buffer=1,
        ).hurs  # some buffer to avoid edge effects / errors in ISIMIP API
        tas_coarse: xr.DataArray = self.download_isimip(
            product="SecondaryInputData",
            variable="tas",
            start_date=self.start_date,
            end_date=self.end_date,
            forcing="w5e5v2.0",
            buffer=1,
        ).tas  # some buffer to avoid edge effects / errors in ISIMIP API
        rlds_coarse: xr.DataArray = self.download_isimip(
            product="SecondaryInputData",
            variable="rlds",
            start_date=self.start_date,
            end_date=self.end_date,
            forcing="w5e5v2.0",
            buffer=1,
        ).rlds  # some buffer to avoid edge effects / errors in ISIMIP API

        target: xr.DataArray = self.other["climate/hurs"]
        target.raster.set_crs(4326)

        hurs_coarse_regridded: xr.DataArray = resample_like(
            hurs_coarse, target, method="conservative"
        )

        tas_coarse_regridded: xr.DataArray = resample_like(
            tas_coarse, target, method="conservative"
        )
        rlds_coarse_regridded: xr.DataArray = resample_like(
            rlds_coarse, target, method="conservative"
        )

        hurs_fine: xr.DataArray = self.other["climate/hurs"]
        tas_fine: xr.DataArray = self.other["climate/tas"]

        # now ready for calculation:
        es_coarse = es0 * np.exp(
            (lv / Rv) * (1 / T0 - 1 / tas_coarse_regridded)
        )  # saturation vapor pressure
        pV_coarse = (
            hurs_coarse_regridded * es_coarse
        ) / 100  # water vapor pressure [hPa]

        es_fine = es0 * np.exp((lv / Rv) * (1 / T0 - 1 / tas_fine))
        pV_fine = (hurs_fine * es_fine) / 100  # water vapour pressure [hPa]

        e_cl_coarse = 0.23 + x1 * ((pV_coarse * 100) / tas_coarse_regridded) ** (1 / x2)
        # e_cl_coarse == clear-sky emissivity w5e5 (pV needs to be in Pa not hPa, hence *100)
        e_cl_fine = 0.23 + x1 * ((pV_fine * 100) / tas_fine) ** (1 / x2)
        # e_cl_fine == clear-sky emissivity target grid (pV needs to be in Pa not hPa, hence *100)

        e_as_coarse = rlds_coarse_regridded / (
            sbc * tas_coarse_regridded**4
        )  # all-sky emissivity w5e5
        e_as_coarse = xr.where(
            e_as_coarse < 1, e_as_coarse, 1
        )  # constrain all-sky emissivity to max 1
        delta_e = e_as_coarse - e_cl_coarse  # cloud-based component of emissivity w5e5

        e_as_fine = e_cl_fine + delta_e
        e_as_fine = xr.where(
            e_as_fine < 1, e_as_fine, 1
        )  # constrain all-sky emissivity to max 1
        lw_fine = (
            e_as_fine * sbc * tas_fine**4
        )  # downscaled lwr! assume cloud e is the same

        lw_fine: xr.DataArray = self.snap_to_grid(
            lw_fine, self.grid, relative_tollerance=0.07
        )
        self.set_rlds(lw_fine)

    def setup_pressure_isimip_30arcsec(self) -> None:
        """Sets up the surface pressure data for GEB.

        This method sets up the surface pressure data for GEB. It then downloads
        the orography data and surface pressure data from the ISIMIP dataset for the specified time period using the
        `download_isimip` method. The data is downloaded at a 30 arcsec resolution.

        The method then regrids the orography and surface pressure data to the target grid using the `xe.Regridder` method.
        It corrects the surface pressure data for orography using the gravitational acceleration, molar mass of
        dry air, universal gas constant, and sea level standard temperature. The downscaling method is adapted
        from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

        The resulting surface pressure data is set as forcing data in the model with names of the form 'climate/ps'.
        """
        g = 9.80665  # gravitational acceleration [m/s2]
        M = 0.02896968  # molar mass of dry air [kg/mol]
        r0 = 8.314462618  # universal gas constant [J/(mol·K)]
        T0 = 288.16  # Sea level standard temperature  [K]

        pressure_30_min = self.download_isimip(
            product="SecondaryInputData",
            variable="psl",  # pressure at sea level
            start_date=self.start_date,
            end_date=self.end_date,
            forcing="w5e5v2.0",
            buffer=1,
        ).psl  # some buffer to avoid edge effects / errors in ISIMIP API

        target = self.other["climate/hurs"]
        target.raster.set_crs(4326)

        orography = self.download_isimip(
            product="InputData", variable="orog", forcing="chelsa-w5e5", buffer=1
        ).orog  # some buffer to avoid edge effects / errors in ISIMIP API
        # TODO: This can perhaps be a clipped version of the orography data
        orography = resample_like(orography, target, method="bilinear")

        # pressure at sea level, so we can do bilinear interpolation before
        # applying the correction for orography
        pressure_30_min_regridded = resample_like(
            pressure_30_min, target, method="conservative"
        )

        pressure_30_min_regridded_corr = pressure_30_min_regridded * np.exp(
            -(g * orography * M) / (T0 * r0)
        )

        self.set_ps(pressure_30_min_regridded_corr)

    def setup_wind_isimip_30arcsec(self) -> None:
        """This method sets up the wind data for GEB.

        It first downloads the global wind atlas data and
        regrids it to the target grid using the `xe.Regridder` method. It then downloads the 30-minute average wind data
        from the ISIMIP dataset for the specified time period and regrids it to the target grid using the `xe.Regridder`
        method.

        The method then creates a diff layer by assuming that wind follows a Weibull distribution and taking the log
        transform of the wind data. It then subtracts the log-transformed 30-minute average wind data from the
        log-transformed global wind atlas data to create the diff layer.

        The method then downloads the wind data from the ISIMIP dataset for the specified time period and regrids it to the
        target grid using the `xe.Regridder` method. It applies the diff layer to the log-transformed wind data and then
        exponentiates the result to obtain the corrected wind data. The downscaling method is adapted
        from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

        The resulting wind data is set as forcing data in the model with names of the form 'climate/wind'.

        Currently the global wind atlas database is offline, so the correction is removed
        """
        global_wind_atlas = self.data_catalog.get_rasterdataset(
            "global_wind_atlas", bbox=self.grid.raster.bounds, buffer=10
        )
        target = self.grid["mask"]
        target.raster.set_crs(4326)

        global_wind_atlas_regridded = resample_like(
            global_wind_atlas, target, method="conservative"
        )

        wind_30_min_avg = self.download_isimip(
            product="SecondaryInputData",
            variable="sfcwind",
            start_date=date(2008, 1, 1),
            end_date=date(2017, 12, 31),
            forcing="w5e5v2.0",
            buffer=1,
        ).sfcWind.mean(
            dim="time"
        )  # some buffer to avoid edge effects / errors in ISIMIP API

        wind_30_min_avg_regridded = resample_like(
            wind_30_min_avg, target, method="conservative"
        )

        # create diff layer:
        # assume wind follows weibull distribution => do log transform
        wind_30_min_avg_regridded_log = np.log(wind_30_min_avg_regridded)

        global_wind_atlas_regridded_log = np.log(global_wind_atlas_regridded)

        diff_layer = (
            global_wind_atlas_regridded_log - wind_30_min_avg_regridded_log
        )  # to be added to log-transformed daily

        wind_30_min = self.download_isimip(
            product="SecondaryInputData",
            variable="sfcwind",
            start_date=self.start_date,
            end_date=self.end_date,
            forcing="w5e5v2.0",
            buffer=1,
        ).sfcWind  # some buffer to avoid edge effects / errors in ISIMIP API

        wind_30min_regridded = resample_like(wind_30_min, target, method="conservative")
        wind_30min_regridded_log = np.log(wind_30min_regridded)

        wind_30min_regridded_log_corr = wind_30min_regridded_log + diff_layer
        wind_30min_regridded_corr = np.exp(wind_30min_regridded_log_corr)

        wind_output_clipped = wind_30min_regridded_corr.raster.clip_bbox(
            self.grid.raster.bounds
        )

        wind_output_clipped = self.snap_to_grid(wind_output_clipped, self.grid)
        self.set_sfcwind(wind_output_clipped)

    @build_method(depends_on=["setup_forcing"])
    def setup_SPEI(
        self,
        calibration_period_start: date = date(1981, 1, 1),
        calibration_period_end: date = date(2010, 1, 1),
        window_months: int = 12,
    ) -> None:
        """Sets up the Standardized Precipitation Evapotranspiration Index (SPEI).

        Note that due to the sliding window, the SPEI data will be shorter than the original data. When
        a sliding window of 12 months is used, the SPEI data will be shorter by 11 months.

        Also sets up the Generalized Extreme Value (GEV) parameters for the SPEI data, being
        the c shape (ξ), loc location (μ), and scale (σ) parameters.

        The chunks for the climate data are optimized for reading the data in xy-direction. However,
        for the SPEI calculation, the data is needs to be read in time direction. Therefore, we
        create an intermediate temporary file of the water balance wher chunks are in an intermediate
        size between the xy and time chunks.

        Args:
            calibration_period_start: The start time of the reSPEI data in ISO 8601 format (YYYY-MM-DD).
            calibration_period_end: The end time of the SPEI data in ISO 8601 format (YYYY-MM-DD). Endtime is exclusive.
            window_months: The window size in months for the SPEI calculation. Default is 12 months.

        Raises:
            ValueError: If the input data do not have the same coordinates.
        """
        assert window_months <= 12, (
            "window_months must be less than or equal to 12 (otherwise we run out of climate data)"
        )
        assert window_months >= 1, (
            "window_months must be greater than or equal to 1 (otherwise we have no sliding window)"
        )

        # assert input data have the same coordinates
        assert np.array_equal(
            self.other["climate/pr"].x, self.other["climate/tasmin"].x
        )
        assert np.array_equal(
            self.other["climate/pr"].x, self.other["climate/tasmax"].x
        )
        assert np.array_equal(
            self.other["climate/pr"].y, self.other["climate/tasmin"].y
        )
        assert np.array_equal(
            self.other["climate/pr"].y, self.other["climate/tasmax"].y
        )
        if not self.other[
            "climate/pr"
        ].time.min().dt.date <= calibration_period_start and self.other[
            "climate/pr"
        ].time.max().dt.date >= calibration_period_end - timedelta(days=1):
            forcing_start_date = self.other["climate/pr"].time.min().dt.date.item()
            forcing_end_date = self.other["climate/pr"].time.max().dt.date.item()
            raise ValueError(
                f"water data does not cover the entire calibration period, forcing data covers from {forcing_start_date} to {forcing_end_date}, "
                f"while requested calibration period is from {calibration_period_start} to {calibration_period_end}"
            )

        pet = xci.potential_evapotranspiration(
            tasmin=self.other["climate/tasmin"],
            tasmax=self.other["climate/tasmax"],
            # hurs=self.other["climate/hurs"],
            # rsds=self.other["climate/rsds"],
            # rlds=self.other["climate/rlds"],
            # rsus=self.full_like(
            #     self.other["climate/rsds"],
            #     fill_value=0,
            #     nodata=np.nan,
            #     attrs=self.other["climate/rsds"].attrs,
            # ),
            # rlus=self.full_like(
            #     self.other["climate/rsds"],
            #     fill_value=0,
            #     nodata=np.nan,
            #     attrs=self.other["climate/rsds"].attrs,
            # ),
            # sfcWind=self.other["climate/sfcwind"],
            method="BR65",
        ).astype(np.float32)

        # Compute the potential evapotranspiration
        water_budget = xci.water_budget(pr=self.other["climate/pr"], evspsblpot=pet)

        water_budget = water_budget.resample(time="MS").mean(keep_attrs=True)
        water_budget.attrs["_FillValue"] = np.nan

        temp_xy_chunk_size = 50

        with tempfile.TemporaryDirectory() as tmp_water_budget_folder:
            tmp_water_budget_file = (
                Path(tmp_water_budget_folder) / "tmp_water_budget_file.zarr"
            )
            self.logger.info("Exporting temporary water budget to zarr")
            water_budget = to_zarr(
                water_budget,
                tmp_water_budget_file,
                crs=4326,
                x_chunksize=temp_xy_chunk_size,
                y_chunksize=temp_xy_chunk_size,
                time_chunksize=50,
                time_chunks_per_shard=None,
            ).chunk({"time": -1})  # for the SPEI calculation time must not be chunked

            # We set freq to None, so that the input frequency is used (no recalculating)
            # this means that we can calculate SPEI much more efficiently, as it is not
            # rechunked in the xclim package

            # The log-logistic distribution used in SPEI has three parameters: scale, shape, and location.
            # In practice, fixing the location (floc) to 0 simplifies the fitting process and often
            # provides satisfactory results.The fitting is then effectively done with two parameters,
            # also reducing the risk of overfitting, especially with limited data.
            # When empirical data suggest that the climatic water balance values are significantly shifted,
            # a non-zero floc may better fit the distribution. However, this is not typical in routine applications.
            SPEI = xci.standardized_precipitation_evapotranspiration_index(
                wb=water_budget,
                cal_start=calibration_period_start.strftime("%Y-%m-%d"),
                cal_end=calibration_period_end.strftime("%Y-%m-%d"),
                freq=None,
                window=window_months,
                dist="fisk",  # log-logistic distribution
                method="APP",  # approximative method
                fitkwargs={
                    "floc": water_budget.min().compute().item()
                },  # location parameter, assures that the distribution is always positive
            ).astype(np.float32)

            # remove all nan values as a result of the sliding window
            SPEI: xr.DataArray = SPEI.isel(time=slice(window_months - 1, None))

            with tempfile.TemporaryDirectory() as tmp_spei_folder:
                tmp_spei_file = Path(tmp_spei_folder) / "tmp_spei_file.zarr"
                self.logger.info("Calculating SPEI and exporting to temporary file...")
                SPEI.attrs = {
                    "_FillValue": np.nan,
                }
                SPEI: xr.DataArray = to_zarr(
                    SPEI,
                    tmp_spei_file,
                    x_chunksize=temp_xy_chunk_size,
                    y_chunksize=temp_xy_chunk_size,
                    time_chunksize=10,
                    time_chunks_per_shard=None,
                    crs=4326,
                )

                self.set_SPEI(SPEI)

                self.logger.info("calculating GEV parameters...")

                # Group the data by year and find the maximum monthly sum for each year
                SPEI_yearly_min = SPEI.groupby("time.year").min(dim="time", skipna=True)
                SPEI_yearly_min = (
                    SPEI_yearly_min.rename({"year": "time"})
                    .chunk({"time": -1})
                    .compute()
                )

                GEV = xci.stats.fit(SPEI_yearly_min, dist="genextreme").compute()

                self.set_grid(
                    GEV.sel(dparams="c").astype(np.float32), name="climate/gev_c"
                )
                self.set_grid(
                    GEV.sel(dparams="loc").astype(np.float32), name="climate/gev_loc"
                )
                self.set_grid(
                    GEV.sel(dparams="scale").astype(np.float32),
                    name="climate/gev_scale",
                )

    def get_elevation_forcing_and_grid(
        self, grid: xr.DataArray, forcing_grid: xr.DataArray, forcing_name: str
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Gets elevation maps for both the normal grid (target of resampling) and the forcing grid.

        Args:
            grid: the normal grid to which the forcing data is resampled
            forcing_grid: grid of the forcing data
            forcing_name: name of the forcing data, used to determine the file paths for caching

        Returns:
            elevation data for the forcing grid and the normal grid

        """
        # we also need to process the elevation data for the grid, because the
        # elevation data that is available in the model is masked to the grid
        elevation_forcing_fp: Path = (
            self.preprocessing_dir / "climate" / forcing_name / "DEM_forcing.zarr"
        )
        elevation_grid_fp = self.preprocessing_dir / "climate" / "DEM.zarr"

        if elevation_forcing_fp.exists() and elevation_grid_fp.exists():
            elevation_forcing = open_zarr(elevation_forcing_fp)
            elevation_grid = open_zarr(elevation_grid_fp)

            if (
                np.array_equal(grid.x.values, elevation_grid.x.values)
                and np.array_equal(grid.y.values, elevation_grid.y.values)
                and np.array_equal(forcing_grid.x.values, elevation_forcing.x.values)
                and np.array_equal(forcing_grid.y.values, elevation_forcing.y.values)
            ):
                self.logger.debug("Using cached elevation data")
                return elevation_forcing, elevation_grid
            else:
                self.logger.warning(
                    "Cached elevation data does not match the grid, recalculating elevation data"
                )
                shutil.rmtree(elevation_forcing_fp)
                shutil.rmtree(elevation_grid_fp)

        elevation = xr.open_dataarray(self.data_catalog.get_source("fabdem").path)
        elevation = elevation.isel(
            band=0,
            **get_window(
                elevation.x, elevation.y, forcing_grid.rio.bounds(), buffer=500
            ),
        )
        elevation = elevation.drop_vars("band")
        elevation = xr.where(elevation == -9999, 0, elevation)
        elevation.attrs["_FillValue"] = np.nan
        target = forcing_grid.isel(time=0).drop_vars("time")

        elevation_forcing = resample_like(elevation, target, method="bilinear")
        elevation_forcing = elevation_forcing.chunk({"x": -1, "y": -1})

        elevation_forcing = to_zarr(
            elevation_forcing,
            elevation_forcing_fp,
            crs=4326,
        )

        elevation_grid = resample_like(elevation, grid, method="bilinear")
        elevation_grid = elevation_grid.chunk({"x": -1, "y": -1})

        elevation_grid = to_zarr(
            elevation_grid,
            elevation_grid_fp,
            crs=4326,
        )

        return elevation_forcing, elevation_grid

    @build_method(depends_on=["set_ssp", "set_time_range"])
    def setup_CO2_concentration(self) -> None:
        """Aquires the CO2 concentration data for the specified SSP in ppm."""
        da: xr.DataArray = self.construct_ISIMIP_variable(
            variable_name="co2",
            forcing=None,
            ssp=self.ISIMIP_ssp,
        ).astype(np.float32)
        self.set_other(
            da,
            name="climate/CO2_ppm",
        )

    @build_method(depends_on=["set_ssp", "set_time_range"])
    def setup_forecasts(
        self,
        only_rainfall: bool,
        forecast_start: date | datetime,
        forecast_end: date | datetime,
        forecast_provider: str,
        forecast_model: str,
        forecast_resolution: float,
        forecast_horizon: int,
        forecast_timestep: int,
    ):
        """Sets up forecast data for the model based on configuration.

        Args:
            forecast_variable: List of ECMWF parameter codes to download (see ECMWF documentation).
            forecast_start: The forecast initialization time (date or datetime).
            forecast_end: The forecast end time (date or datetime).
            forecast_provider: The forecast data provider to use (default: "ECMWF").
            forecast_model: The ECMWF forecast model to use (e.g., "HRES", "pf").
            forecast_resolution: The spatial resolution of the forecast data (degrees).
            forecast_horizon: The forecast horizon in hours.
            forecast_timestep: The forecast timestep in hours.
        """

        if (
            forecast_provider == "ECMWF"
        ):  # Check if ECMWF is the selected forecast provider
            self.setup_forecasts_ECMWF(  # Call ECMWF-specific setup method
                only_rainfall,  # Pass rainfall-only flag
                forecast_start,  # Pass forecast start date
                forecast_end,  # Pass forecast end date
                forecast_model,  # Pass forecast model type
                forecast_resolution,  # Pass spatial resolution
                forecast_horizon,  # Pass forecast horizon in hours
                forecast_timestep,  # Pass timestep interval
            )

    def setup_forecasts_ECMWF(
        self,
        only_rainfall: bool,
        forecast_start: date | datetime,
        forecast_end: date | datetime,
        forecast_model: str,
        forecast_resolution: float,
        forecast_horizon: int,
        forecast_timestep: int,
    ) -> None:
        """Sets up the folder structure for ECMWF forecast data.

        Args:
            only_rainfall: If True, only download rainfall forecasts.
            forecast_start: The forecast initialization time (date or datetime).
            forecast_end: The forecast end time (date or datetime).
            forecast_model: The ECMWF forecast model to use (e.g., "HRES",
                "pf").
            forecast_resolution: The spatial resolution of the forecast data (degrees).
            forecast_horizon: The forecast horizon in hours.
            forecast_timestep: The forecast timestep in hours.
        """

        preprocessing_folder = (
            self.preprocessing_dir / "forecasts" / "ECMWF"
        )  # Set up forecast data path
        preprocessing_folder.mkdir(
            parents=True, exist_ok=True
        )  # Ensure directory exists

        target = self.grid["mask"]  # Get the model's spatial mask grid
        target.raster.set_crs(4326)  # Set coordinate reference system to WGS84
        bounds = target.raster.bounds  # Extract geographic bounding box

        if only_rainfall == True:  # If only precipitation is required
            MARS_codes: dict[
                str, float
            ] = {  # Dictionary mapping variable names to ECMWF parameter codes
                "total_precipitation": 228,  # Total precipitation parameter
            }  # https://codes.ecmwf.int/grib/param-db/ --> parameter IDs (the .128 is necessary for parameters with up to 3 digits)
        else:  # If full meteorological variables are needed
            MARS_codes: dict[str, float] = {  # Complete set of weather variables
                "tp": 228.128,  # total precipitation
                "t2m": 167.128,  # 2 metre temperature
                "d2m": 168.128,  # 2 metre dewpoint temperature
                "ssrd": 169.128,  # surface shortwave solar radiation downwards
                "strd": 175.128,  # surface longwave radiation downwards
                "sp": 134.128,  # surface pressure
                "u10": 165.128,  # 10 metre u-component of wind
                "v10": 166.128,  # 10 metre v-component of wind
            }

        # Configure arguments for the forecast download function
        download_args: dict[str, Any] = {  # Dictionary to store download parameters
            "preprocessing_folder": preprocessing_folder,  # Output directory path
            "bounds": bounds,  # Geographic bounding box for data extraction
            "forecast_start": forecast_start,  # Start date for forecast downloads
            "forecast_end": forecast_end,  # End date for forecast downloads
            "forecast_model": forecast_model,  # ECMWF model type (HRES, pf, etc.)
            "forecast_resolution": forecast_resolution,  # Spatial resolution in degrees
            "forecast_horizon": forecast_horizon,  # Forecast horizon in hours
            "forecast_timestep": forecast_timestep,  # Temporal resolution in hours
        }

        # Configure arguments for the forecast processing function
        process_args: dict[str, Any] = {  # Dictionary to store processing parameters
            "preprocessing_folder": preprocessing_folder,  # Input directory with downloaded files
            "bounds": bounds,  # Geographic bounds for spatial cropping
        }

        # Download the forecast data from ECMWF using MARS API
        self.logger.info("Downloading ECMWF forecasts...")  # Log download start
        download_forecasts_ECMWF(
            self, list(MARS_codes.values()), **download_args
        )  # Call download function with parameter codes

        self.logger.info(
            "Processing ECMWF precipitation forecasts..."
        )  # Process precipitation data (hourly resolution)

        forecast_issue_dates = pd.date_range(  # Create pandas date range
            start=forecast_start,  # Start from forecast start date
            end=forecast_end,  # End at forecast end date
            freq="24H",  # Daily frequency (24-hour intervals)
        )

        for (
            forecast_issue_date
        ) in forecast_issue_dates:  # # Process each forecast issue date separately
            forecast_issue_date_str = forecast_issue_date.strftime(
                "%Y%m%dT%H%M%S"
            )  # Format date for filenames

            self.logger.info(
                f"Processing forecast issued at {forecast_issue_date}..."
            )  # Log current forecast being processed
            process_args["forecast_issue_date"] = (
                forecast_issue_date  # Add current date to processing arguments
            )

            # Process the raw GRIB forecast data into xarray format
            ECMWF_forecast = process_forecast_ECMWF(
                self, **process_args
            )  # Call processing function

            # Extract and process hourly precipitation data
            pr_hourly = ECMWF_forecast["tp"]  # Get total precipitation variable

            pr_hourly = pr_hourly.where(
                pr_hourly >= 0, 0
            )  # Handle negative values (caused by floating-point precision issues) by setting them to zero
            pr_hourly = pr_hourly.rename(
                "precipitation"
            )  # Change from "tp" to "precipitation"
            # Store the processed hourly precipitation data
            self.set_pr_hourly(  # Save hourly precipitation
                pr_hourly,
                name=f"forecasts/ECMWF/pr_hourly_{forecast_issue_date_str}",  # Use date-specific filename
            )

            # Process additional meteorological variables if not rainfall-only mode
            if only_rainfall == False:  # If full meteorological processing is required
                # Load elevation grids for topographic corrections
                (
                    elevation_forcing,
                    elevation_target,
                ) = (  # Get elevation data for corrections
                    self.get_elevation_forcing_and_grid(  # Function to retrieve elevation grids
                        self.grid["mask"],
                        pr_hourly,
                        forcing_name="ECMWF",
                    )
                )

                pr = pr_hourly.resample(
                    time="D"
                ).mean()  # Calculate daily mean precipitation
                pr = resample_like(
                    pr, target, method="conservative"
                )  # Resample to target grid using conservative method
                pr = self.set_pr(
                    pr, f"forecasts/ECMWF/pr_{forecast_issue_date_str}"
                )  # Store daily precipitation

                hourly_tas = ECMWF_forecast["t2m"]  # Extract 2-meter temperature
                hourly_tas = hourly_tas.rename(
                    "tas"
                )  # Rename to standard climate variable name
                tas_avg = hourly_tas.resample(
                    time="D"
                ).mean()  # Calculate daily mean temperature
                tas_avg = reproject_and_apply_lapse_rate_temperature(  # Correct temperature for elevation
                    tas_avg,
                    elevation_forcing,
                    elevation_target,  # Use elevation grids for correction
                )
                self.set_tas(
                    tas_avg, f"forecasts/ECMWF/tas_{forecast_issue_date_str}"
                )  # Store average temperature
                tasmax = hourly_tas.resample(
                    time="D"
                ).max()  # Calculate daily maximum temperature
                tasmax = reproject_and_apply_lapse_rate_temperature(  # Correct for topography
                    tasmax, elevation_forcing, elevation_target
                )
                tasmax = tasmax.rename("tasmax")  # Ensure proper variable name
                self.set_tasmax(  # Store maximum temperature
                    tasmax, f"forecasts/ECMWF/tasmax_{forecast_issue_date_str}"
                )
                tasmin = hourly_tas.resample(
                    time="D"
                ).min()  # Calculate daily minimum temperature
                tasmin = (
                    reproject_and_apply_lapse_rate_temperature(  # Correct for elevation
                        tasmin, elevation_forcing, elevation_target
                    )
                )
                tasmin = tasmin.rename("tasmin")  # Ensure proper variable name
                self.set_tasmin(  # Store minimum temperature
                    tasmin, f"forecasts/ECMWF/tasmin_{forecast_issue_date_str}"
                )

                hourly_dew_point_tas = ECMWF_forecast[
                    "d2m"
                ]  # Extract dewpoint temperature
                hourly_dew_point_tas = hourly_dew_point_tas.rename(
                    "dew_point_tas"
                )  # Rename for clarity
                dew_point_tas = hourly_dew_point_tas.resample(
                    time="D"
                ).mean()  # Calculate daily mean dewpoint
                dew_point_tas = reproject_and_apply_lapse_rate_temperature(  # Correct dewpoint for elevation
                    dew_point_tas, elevation_forcing, elevation_target
                )
                water_vapour_pressure = (
                    0.6108
                    * np.exp(  # Magnus formula for vapor pressure
                        17.27  # Magnus coefficient
                        * (dew_point_tas - 273.15)  # Convert Kelvin to Celsius
                        / (237.3 + (dew_point_tas - 273.15))  # Magnus denominator
                    )
                )  # calculate water vapour pressure (kPa)
                saturation_vapour_pressure = (
                    0.6108
                    * np.exp(  # Magnus formula for saturation
                        17.27
                        * (tas_avg - 273.15)
                        / (237.3 + (tas_avg - 273.15))  # Using average temperature
                    )
                )
                assert (
                    water_vapour_pressure.shape == saturation_vapour_pressure.shape
                )  # Ensure compatible shapes
                relative_humidity = (
                    (  # RH = (e / e_sat) * 100
                        water_vapour_pressure
                        / saturation_vapour_pressure  # Ratio of actual to saturation vapor pressure
                    )
                    * 100
                )  # Convert to percentage
                original_crs = (  # Store original CRS if available
                    relative_humidity.rio.crs  # Get CRS from rioxarray
                    if hasattr(
                        relative_humidity, "rio"
                    )  # Check if rioxarray is available
                    else None  # Default to None if no CRS info
                )
                relative_humidity = xr.where(  # Handle relative humidity values slightly above 100% (due to numerical precision)
                    (relative_humidity > 100)
                    & (relative_humidity <= 101),  # Values between 100-101%
                    100,  # Set to exactly 100%
                    relative_humidity,  # Keep original values otherwise
                    keep_attrs=True,  # Preserve data attributes
                )

                if (
                    original_crs is not None
                    and (  # Restore CRS information if it was lost during calculations
                        not hasattr(relative_humidity, "rio")  # CRS was lost
                        or relative_humidity.rio.crs is None
                    )
                ):
                    relative_humidity = relative_humidity.rio.write_crs(
                        original_crs
                    )  # Restore CRS
                relative_humidity = relative_humidity.rename(
                    "hurs"
                )  # Rename to standard climate variable
                self.set_hurs(  # Store relative humidity
                    relative_humidity, f"forecasts/ECMWF/hurs_{forecast_issue_date_str}"
                )

                hourly_rsds = ECMWF_forecast["ssrd"]  # Extract shortwave radiation
                rsds = hourly_rsds.resample(  # Resample to daily resolution
                    time="D"
                ).mean()  # get average W/m2 over the day
                rsds = resample_like(
                    rsds, target, method="conservative"
                )  # Resample to target grid
                rsds = rsds.rename("rsds")  # Rename to standard variable name
                self.set_rsds(
                    rsds, f"forecasts/ECMWF/rsds_{forecast_issue_date_str}"
                )  # Store solar radiation

                # Process surface longwave (thermal) radiation downwards
                hourly_rlds = ECMWF_forecast["strd"]  # Extract longwave radiation
                rlds = hourly_rlds.resample(  # Resample to daily resolution
                    time="D"
                ).mean()  # get average W/m2 over the day
                rlds = resample_like(
                    rlds, target, method="conservative"
                )  # Resample to target grid
                rlds = rlds.rename("rlds")  # Rename to standard variable name
                self.set_rlds(
                    rlds, f"forecasts/ECMWF/rlds_{forecast_issue_date_str}"
                )  # Store longwave radiation

                pressure = ECMWF_forecast["sp"]  # Extract surface pressure
                pressure = pressure.resample(
                    time="D"
                ).mean()  # Calculate daily mean pressure
                pressure = reproject_and_apply_lapse_rate_pressure(  # Correct pressure for elevation
                    pressure,
                    elevation_forcing,
                    elevation_target,  # Use elevation grids
                )
                pressure = pressure.rename("ps")  # Rename to standard variable name

                self.set_ps(
                    pressure, f"forecasts/ECMWF/ps_{forecast_issue_date_str}"
                )  # Store surface pressure

                u_wind = ECMWF_forecast["u10"]  # Extract u-component of wind at 10m
                u_wind = u_wind.resample(time="D").mean()  # Calculate daily mean u-wind
                v_wind = ECMWF_forecast["v10"]  # Extract v-component of wind at 10m
                v_wind = v_wind.resample(time="D").mean()  # Calculate daily mean v-wind
                wind_speed = np.sqrt(
                    u_wind**2 + v_wind**2
                )  # Pythagorean theorem: |v| = sqrt(u² + v²) --> # Calculate wind speed magnitude from components (sfcWind: wind speed at 10 m height in m/s)
                wind_speed = resample_like(
                    wind_speed, target, method="conservative"
                )  # Resample to target grid
                wind_speed = wind_speed.rename(
                    "sfcWind"
                )  # Rename to standard variable name
                self.set_sfcwind(  # Store surface wind speed
                    wind_speed, f"forecasts/ECMWF/sfcwind_{forecast_issue_date_str}"
                )

    def setup_forcing_ISIMIP(self, resolution_arcsec: int, model: str) -> None:
        """Sets up the forcing data for GEB using ISIMIP data.

        Args:
            resolution_arcsec: The resolution of the data in arcseconds. Supported values are 30 and 1800.
            model: The forcing data to use. Supported values are 'chelsa-w5e5' for 30 arcsec resolution
                and ipsl-cm6a-lr, gfdl-esm4, mpi-esm1-2-hr, mri-esm2-0, and mri-esm2-0 for 1800 arcsec resolution.
        Raises:
            ValueError: If the resolution is not 30 or 1800 arcseconds, or
                if the model is not supported for the given resolution.

        """
        if resolution_arcsec == 30:
            assert model == "chelsa-w5e5", (
                "Only chelsa-w5e5 is supported for 30 arcsec resolution"
            )
            # download source data from ISIMIP
            self.logger.info("setting up forcing data")
            high_res_variables: list = ["pr", "rsds", "tas", "tasmax", "tasmin"]
            self.setup_30arcsec_variables_isimip(high_res_variables)
            self.logger.info("setting up relative humidity...")
            self.setup_hurs_isimip_30arcsec()
            self.logger.info("setting up longwave radiation...")
            self.setup_longwave_isimip_30arcsec()
            self.logger.info("setting up pressure...")
            self.setup_pressure_isimip_30arcsec()
            self.logger.info("setting up wind...")
            self.setup_wind_isimip_30arcsec()
        elif resolution_arcsec == 1800:
            assert model in (
                "ipsl-cm6a-lr",
                "gfdl-esm4",
                "mpi-esm1-2-hr",
                "mri-esm2-0",
                "ukesm1-0-ll",
            ), (
                "Only ipsl-cm6a-lr, gfdl-esm4, mpi-esm1-2-hr, mri-esm2-0 and ukesm1-0-ll are supported for 1800 arcsec resolution"
            )
            variables = [
                "pr",
                "rsds",
                "tas",
                "tasmax",
                "tasmin",
                "hurs",
                "rlds",
                "ps",
                "sfcwind",
            ]
            self.setup_1800arcsec_variables_isimip(model, variables)
        else:
            raise ValueError(
                "Only 30 arcsec and 1800 arcsec resolution is supported for ISIMIP data"
            )
