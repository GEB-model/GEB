import tempfile
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ecmwfapi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xclim.indices as xci
from dateutil.relativedelta import relativedelta
from zarr.codecs.numcodecs import FixedScaleOffset

from geb.build.data_catalog.base import Adapter
from geb.build.methods import build_method
from geb.workflows.io import get_window

from ...workflows.io import calculate_scaling, to_zarr
from ..workflows.general import (
    resample_like,
)


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
        self: The class instance.
        forecast_variables: List of ECMWF parameter codes to download (see ECMWF documentation).
        preprocessing_folder: Path to the folder where downloaded forecast files will be stored.
        bounds: The bounding box in the format (min_lon, min_lat, max_lon, max_lat).
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
        self: The class instance.
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
        self: The class instance.
        da: The xarray DataArray containing the forcing data. Must have dimensions 'time',
        name: The name of the variable being plotted, used for titles and filenames.
    """
    fig, axes = plt.subplots(
        4, 1, figsize=(20, 10), gridspec_kw={"hspace": 0.5}
    )  # Create 4 subplots stacked vertically

    data = (da.mean(dim=("y", "x"))).compute()  # Area-weighted average
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
        self: The class instance.
        da: The xarray DataArray containing the forecast data. Must have dimensions 'time', 'y', 'x', and 'member'.
        name: The name of the variable being plotted, used for titles and filenames.
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
    ).compute()  # Area-weighted average (only over the catchment area)

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
        hspace=0.2, wspace=0.2, bottom=0.05, left=0.05, right=0.85
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
        self: The class instance.
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

    def plot_forcing(self, da, name) -> None:
        fig, axes = plt.subplots(4, 1, figsize=(20, 10), gridspec_kw={"hspace": 0.5})

        mask = self.grid["mask"]
        data = ((da * ~mask).sum(dim=("y", "x")) / (~mask).sum()).compute()
        assert not np.isnan(data.values).any(), "data contains NaN values"

        # plot entire timeline on the first axis
        plot_timeline(da, data, name, axes[0])

        # plot the first three years on separate axes
        for i in range(0, 3):
            year = data.time[0].dt.year + i
            year_data = data.sel(time=data.time.dt.year == year)
            if year_data.size > 0:
                plot_timeline(
                    da,
                    data.sel(time=da.time.dt.year == year),
                    f"{name} - {year.item()}",
                    axes[i + 1],
                )

        fp = self.report_dir / (name + "_timeline.png")
        fp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fp)

        plt.close(fig)

        spatial_data = da.mean(dim="time").compute()

        spatial_data.plot()

        plt.title(name)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        spatial_fp: Path = self.report_dir / (name + "_spatial.png")
        plt.savefig(spatial_fp)

        plt.close()

    def set_xy_attrs(self, da: xr.DataArray) -> None:
        """Set CF-compliant attributes for the x and y coordinates of a DataArray."""
        da.x.attrs = {"long_name": "longitude", "units": "degrees_east"}
        da.y.attrs = {"long_name": "latitude", "units": "degrees_north"}

    def set_pr(self, da: xr.DataArray, *args: Any, **kwargs: Any) -> xr.DataArray:
        name: str = "climate/pr"
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
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, 0, max_value, offset=offset, precision=precision
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
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
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, 0, 1361, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
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
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, 0, 1361, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
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
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, -100 + K_to_C, 60 + K_to_C, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )

        _plot_data(self, da, name)
        return da

    def set_dewpoint_tas(
        self, da: xr.DataArray, *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        name: str = "climate/dewpoint_tas"
        da.attrs = {
            "standard_name": "air_temperature_dow_point",
            "long_name": "Hourly Near-Surface Dewpoint Temperature",
            "units": "K",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        K_to_C: float = 273.15
        offset: float = -15 - K_to_C  # average temperature on earth
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, -100 + K_to_C, 60 + K_to_C, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        _plot_data(self, da, name)
        return da

    def set_ps(self, da: xr.DataArray, *args: Any, **kwargs: Any) -> xr.DataArray:
        name: str = "climate/ps"
        da.attrs = {
            "standard_name": "surface_air_pressure",
            "long_name": "Surface Air Pressure",
            "units": "Pa",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset: int = -100_000
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, 30_000, 120_000, offset=offset, precision=10
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        _plot_data(self, da, name)
        return da

    def set_wind(
        self, da: xr.DataArray, direction: str, *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        name: str = f"climate/wind_{direction}10m"
        da.attrs = {
            "standard_name": "wind_speed",
            "long_name": "Near-Surface Wind Speed",
            "units": "m s-1",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0
        # wind can be both positive and negative
        # we assume a maximum wind speed of 120 m/s (432 km/h), which is a stronger
        # than the strongest wind gust ever recorded on earth (113 m/s)
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, -120, 120, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
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
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, min_SPEI, max_SPEI, offset=offset, precision=0.001
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
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
        era5_store: Adapter = self.new_data_catalog.fetch("era5")
        era5_loader: partial = partial(
            era5_store.read,
            start_date=self.start_date - relativedelta(years=1),
            end_date=self.end_date,
            bounds=self.grid["mask"].raster.bounds,
        )

        pr_hourly: xr.DataArray = era5_loader(variable="tp")
        pr_hourly: xr.DataArray = pr_hourly * (
            1000 / 3600
        )  # convert from m/hr to kg/m2/s

        # ensure no negative values for precipitation, which may arise due to float precision
        pr_hourly: xr.DataArray = xr.where(pr_hourly > 0, pr_hourly, 0, keep_attrs=True)
        pr_hourly: xr.DataArray = self.set_pr(pr_hourly)

        hourly_tas: xr.DataArray = era5_loader("t2m")
        self.set_tas(hourly_tas)

        hourly_dew_point_tas: xr.DataArray = era5_loader("d2m")
        self.set_dewpoint_tas(hourly_dew_point_tas)

        hourly_rsds: xr.DataArray = era5_loader("ssrd") / (
            24 * 3600  # convert from J/m2 to W/m2
        )  # surface_solar_radiation_downwards
        self.set_rsds(hourly_rsds)

        hourly_rlds: xr.DataArray = era5_loader(
            "strd"
        ) / (  # surface_thermal_radiation_downwards
            24 * 3600
        )  # convert from J/m2 to W/m2
        self.set_rlds(hourly_rlds)

        pressure: xr.DataArray = era5_loader("sp")
        self.set_ps(pressure)

        u_wind: xr.DataArray = era5_loader("u10")
        self.set_wind(u_wind, direction="u")

        v_wind: xr.DataArray = era5_loader("v10")
        self.set_wind(v_wind, direction="v")

        elevation_forcing: xr.DataArray = self.get_elevation_forcing(
            pr_hourly
        ).compute()
        self.set_other(
            elevation_forcing,
            name="climate/elevation_forcing",
        )

    @build_method(depends_on=["set_ssp", "set_time_range"])
    def setup_forcing(
        self,
        forcing: str = "ERA5",
    ) -> None:
        """Sets up the forcing data for GEB.

        Args:
            forcing: The data source to use for the forcing data. Currently only ERA5 is supported.

        Sets:
            The resulting forcing data is set as forcing data in the model with names of the form 'forcing/{variable_name}'.

        Raises:
            ValueError: If an unknown data source is specified.
        """
        if forcing == "ISIMIP":
            raise NotImplementedError(
                "ISIMIP forcing is not supported anymore. We switched fully to hourly forcing data."
            )
        elif forcing == "ERA5":
            assert resolution_arcsec is None, (
                "resolution_arcsec must be None for ERA5 forcing data"
            )
            assert model is None, "model must be None for ERA5 forcing data"
            self.setup_forcing_ERA5()
        elif forcing == "CMIP":
            raise NotImplementedError("CMIP forcing data is not yet supported")
        else:
            raise ValueError(f"Unknown data source: {forcing}, supported are 'ERA5'")

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
        tasmin = self.other["climate/tas"].resample(time="D").min()
        tasmax = self.other["climate/tas"].resample(time="D").max()
        pr = self.other["climate/pr"].resample(time="D").mean()

        assert np.array_equal(self.other["climate/pr"].x, tasmin.x)
        assert np.array_equal(self.other["climate/pr"].y, tasmin.y)

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
            tasmin=tasmin,
            tasmax=tasmax,
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
        water_budget = xci.water_budget(pr=pr, evspsblpot=pet)

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
            SPEI: xr.DataArray = SPEI.isel(
                time=slice(window_months - 1, None)
            ).compute()

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

                self.set_other(
                    GEV.sel(dparams="c").astype(np.float32), name="climate/gev_c"
                )
                self.set_other(
                    GEV.sel(dparams="loc").astype(np.float32), name="climate/gev_loc"
                )
                self.set_other(
                    GEV.sel(dparams="scale").astype(np.float32),
                    name="climate/gev_scale",
                )

    @build_method(depends_on=["setup_forcing"])
    def setup_pr_GEV(self) -> None:
        pr: xr.DataArray = self.other["climate/pr"] * 3600  # convert to mm/hour
        pr_monthly: xr.DataArray = pr.resample(time="M").sum(dim="time", skipna=True)

        pr_yearly_max = (
            pr_monthly.groupby("time.year")
            .max(dim="time", skipna=True)
            .rename({"year": "time"})
            .chunk({"time": -1})
            .compute()
        )

        gev_pr = xci.stats.fit(pr_yearly_max, dist="genextreme").compute()

        self.set_other(
            gev_pr.sel(dparams="c").astype(np.float32), name="climate/pr_gev_c"
        )
        self.set_other(
            gev_pr.sel(dparams="loc").astype(np.float32), name="climate/pr_gev_loc"
        )
        self.set_other(
            gev_pr.sel(dparams="scale").astype(np.float32),
            name="climate/pr_gev_scale",
        )

    def get_elevation_forcing(self, forcing_grid: xr.DataArray) -> xr.DataArray:
        """Gets elevation maps for both the normal grid (target of resampling) and the forcing grid.

        Args:
            forcing_grid: grid of the forcing data

        Returns:
            elevation data for the forcing grid and the normal grid
        """
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

        elevation_forcing = elevation_forcing.rio.write_crs(4326)

        return elevation_forcing

    @build_method(depends_on=["set_ssp", "set_time_range"])
    def setup_CO2_concentration(self) -> None:
        """Aquires the CO2 concentration data for the specified SSP in ppm."""
        df: pd.DataFrame = self.new_data_catalog.fetch("isimip_co2").read(
            scenario=self.ISIMIP_ssp
        )
        df: pd.DataFrame = df[
            (df.index >= self.start_date.year) & (df.index <= self.end_date.year)
        ]
        self.set_table(df, name="climate/CO2_ppm")

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
            only_rainfall: If True, only download rainfall forecasts.
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
