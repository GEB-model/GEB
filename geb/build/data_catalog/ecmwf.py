"""ECMWF data adapter module."""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import ecmwfapi
import numpy as np
import pandas as pd
import xarray as xr

from geb.workflows.raster import convert_nodata

from .base import Adapter


def format_path(path: Path, **kwargs: str | int) -> Path:
    """Format a Path object with given keyword arguments.

    Args:
        path: The Path object to format.
        **kwargs: Keyword arguments to format the path string.

    Returns:
        A new Path object with the formatted string.
    """
    string_path = str(path)
    string_path = string_path.format(**kwargs)
    path = Path(string_path)
    return path


def format_date(date_obj: datetime) -> str:
    """Format a date or datetime object to a string in 'YYYYMMDDTHHMMSS' format.

    Args:
        date_obj: The date or datetime object to format.

    Returns:
        A string representing the formatted date and time.

    Raises:
        ValueError: If the input is not a date or datetime object.
    """
    if isinstance(date_obj, datetime):
        return date_obj.strftime("%Y%m%dT%H%M%S")
    else:
        raise ValueError("Input must be a date or datetime object.")


def generate_forecast_steps(forecast_date: datetime) -> str:
    """Generate ECMWF forecast step string based on the forecast date.

    ECMWF does not have a consistent 1h timestep for the entire operational archive. Asking hourly data to the server when it does not exist, will result in an error.
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


class ECMWFForecasts(Adapter):
    """Data adapter for obtaining ECMWF forecast data from MARS Archive."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the ERA5 data adapter."""
        super().__init__(*args, **kwargs)

    def fetch(
        self,
        url: None,
        forecast_variables: list[float],
        bounds: tuple[float, float, float, float],
        forecast_start: date | datetime,
        forecast_end: date | datetime,
        forecast_model: str,
        forecast_resolution: str,
        forecast_horizon: int,
        forecast_timestep_hours: int,
        n_ensemble_members: int,
    ) -> ECMWFForecasts:
        """Download ECMWF forecasts using the ECMWF web API: https://github.com/ecmwf/ecmwf-api-client.

        This function downloads ECMWF forecast data for a specified variable and time period
        from the MARS archive using the ECMWF API. It handles the download and processing of deterministic (control forecast, cf)
        and ensemble (probabilistic forecast, pf) forecasts, returning them as an xarray DataArray.

        This function requires the ECMWF_API_KEY, ECMWF_API_URL and ECMWF_API_EMAIL to be set in the environment variables.
        You can do this by adding it to your .env file. For detailed instructions, see GEB documentation.

        Your API key: https://api.ecmwf.int/v1/key/
        MARS data archive: https://apps.ecmwf.int/mars-catalogue/
        Extra Documentation: https://confluence.ecmwf.int/display/UDOC/MARS+content

        Args:
            self: The class instance.
            url: Not used, present for compatibility with base class.
            forecast_variables: List of ECMWF parameter codes to download (see ECMWF documentation).
            bounds: The bounding box in the format (min_lon, min_lat, max_lon, max_lat).
            forecast_start: The forecast initialization time (date or datetime).
            forecast_end: The forecast end time (date or datetime).
            forecast_model: The ECMWF forecast model to use ("probabilistic_forecast", "control_forecast" or "both_control_and_probabilistic").
            forecast_resolution: The spatial resolution of the forecast data (degrees).
            forecast_horizon: The forecast horizon in hours.
            forecast_timestep_hours: The forecast timestep in hours.
            n_ensemble_members: The number of ensemble members to download.

        Returns:
            The ECMWFForecasts instance.

        Raises:
            ImportError: If ECMWF_API_KEY, ECMWF_API_URL or ECMWF_API_EMAIL is not found in environment variables.
            ValueError: If forecast dates are before 2010-01-01.
            ValueError: If the forecast model is not supported.
            APIException: If there is an error accessing the ECMWF MARS service.
        """
        assert url is None, "URL parameter is not used for ECMWF data adapter."

        print(
            f"Downloading forecast variables {forecast_variables}"
        )  # Log the forecast variables being downloaded

        # Check for ECMWF API key in environment variables
        for variable in ("ECMWF_API_KEY", "ECMWF_API_URL", "ECMWF_API_EMAIL"):
            if variable not in os.environ:
                raise ImportError(
                    f"{variable} not found in environment variables. "
                    f"Please set it as {variable}=XXXXX in your .env file. "
                    f"See https://github.com/ecmwf/ecmwf-api-client on how to obtain the keys."
                )
        server = ecmwfapi.ECMWFService(
            "mars"
        )  # Initialize ECMWF MARS service connection

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
        # Determine which model types to download based on YAML configuration
        if forecast_model == "both_control_and_probabilistic":
            model_types_to_download = ["control_forecast", "probabilistic_forecast"]
        elif forecast_model in ["control_forecast", "probabilistic_forecast"]:
            model_types_to_download = [forecast_model]
        else:
            raise ValueError(
                f"Unsupported forecast_model: '{forecast_model}'. "
                "Must be 'control_forecast', 'probabilistic_forecast', or 'both_control_and_probabilistic'."
            )

        for model_type in model_types_to_download:
            print(f"Processing {model_type} downloads...")
            for (
                forecast_date
            ) in forecast_date_list:  # Loop through each forecast date to download
                print(
                    f"Downloading {model_type} for {forecast_date}"
                )  # Print the current forecast date being processed

                # Process MARS request parameters
                mars_class: str = "od"  # operational data class
                mars_expver: str = "1"  # operational version number
                mars_levtype: str = "sfc"  # surface level data type
                mars_param: str = "/".join(
                    str(var) for var in forecast_variables
                )  # Join parameter codes with "/" separator

                if (
                    forecast_timestep_hours == 1
                ):  # Check if hourly timestep is requested
                    mars_step: str = generate_forecast_steps(
                        forecast_date
                    )  # Generate forecast steps based on date using helper function
                elif (
                    forecast_timestep_hours >= 6
                ):  # Check if 6+ hourly timestep is requested
                    mars_step: str = f"0/to/{forecast_horizon}/BY/{forecast_timestep_hours}"  # Create step string for multi-hour intervals
                else:
                    raise ValueError(
                        f"Forecast timestep {forecast_timestep_hours} is not supported. Please use 1 or >=6."
                    )

                mars_stream: str = "enfo"  # Ensemble forecast stream
                mars_time: str = forecast_date.strftime(
                    "%H"
                )  # Extract hour from forecast date for initialization time
                mars_type: str = (
                    "pf" if model_type == "probabilistic_forecast" else "cf"
                )  # Set forecast type: perturbed forecasts (pf) or control forecast (cf)
                mars_grid: str = str(
                    forecast_resolution
                )  # Convert spatial resolution to string
                mars_area: str = (
                    bounds_str  # Set bounding box area in North/West/South/East format
                )

                # retrieve steps from mars
                mars_request: dict[
                    str, Any
                ] = {  # Build MARS request dictionary with all parameters
                    "class": mars_class,
                    "date": forecast_date.strftime("%Y-%m-%d"),
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

                output_filename = format_path(
                    self.path,
                    forecast_date=format_date(forecast_date),
                    forecast_model=model_type,
                    forecast_resolution=forecast_resolution.replace("/", "-"),
                    forecast_horizon=forecast_horizon,
                    forecast_timestep_hours=forecast_timestep_hours,
                )

                if output_filename.exists():
                    print(
                        f"Forecast file {output_filename} already exists, skipping download."
                    )
                    continue  # Skip download if file already exists

                output_filename.parent.mkdir(parents=True, exist_ok=True)

                if (
                    model_type == "probabilistic_forecast"
                ):  # check if ensemble forecasts are requested
                    mars_request["number"] = (
                        f"1/to/{n_ensemble_members}"  # Add ensemble member numbers to request
                    )

                print(
                    f"Requesting data from ECMWF MARS server.. {mars_request}"
                )  # Log the MARS request parameters

                try:
                    server.execute(  # Execute the MARS request to download data
                        mars_request,
                        output_filename,
                    )  # start the download
                except ecmwfapi.api.APIException as e:
                    if "has no access to services/mars" in str(e):
                        raise ValueError(
                            "\033[91mAccess denied to ECMWF MARS service. To get access, please visit https://confluence.ecmwf.int/display/WEBAPI/Access+MARS, "
                            "register for an account if you don't have one, and request access to the MARS archive, usually through your country representative (see website). "
                            "Once approved, ensure your API key, URL, and email are set in your .env file as ECMWF_API_KEY, ECMWF_API_URL, and ECMWF_API_EMAIL.\033[0m"
                        ) from e
                    else:
                        raise  # Re-raise other API exceptions

        return self

    def read(
        self,
        bounds: tuple[float, float, float, float],
        forecast_issue_date: datetime,
        forecast_model: str,
        forecast_resolution: str,
        forecast_horizon: int,
        forecast_timestep_hours: int,
        reproject_like: xr.DataArray,
    ) -> None | xr.Dataset:
        """Process downloaded ECMWF forecast data.

        We process forecasts for each initialization time separately. The forecast file contains all variables needed for GEB.

        Args:
            preprocessing_folder: Path to the folder containing the downloaded ECMWF forecast data.
            bounds: The bounding box in the format (min_lon, min_lat, max_lon,
                    max_lat).
            forecast_issue_date: The forecast initialization time.
            forecast_model: The ECMWF forecast model from build.yml config ("probabilistic_forecast", "control_forecast" or "both_control_and_probabilistic").
            forecast_resolution: The spatial resolution of the forecast data (degrees).
            forecast_horizon: The forecast horizon in hours.
            forecast_timestep_hours: The forecast timestep in hours.
            reproject_like: An xarray DataArray to use as a template for reprojecting
                the forecast data.

        Returns:
            da: processed ECMWF forecast data as an xarray Dataset.

        Raises:
            ValueError: If forecast initialization dates or time dimensions don't match between control and ensemble.
        """

        def _load_forecast_file(model_type: str) -> xr.Dataset:
            """Load a single forecast dataset for the specified model type.

            Args:
                model_type: Either 'control_forecast' or 'probabilistic_forecast'.

            Returns:
                Loaded and renamed forecast dataset.

            Raises:
                FileNotFoundError: If the forecast file doesn't exist.
            """
            filename = format_path(
                self.path,
                forecast_date=format_date(forecast_issue_date),
                forecast_model=model_type,
                forecast_resolution=forecast_resolution.replace("/", "-"),
                forecast_horizon=forecast_horizon,
                forecast_timestep_hours=forecast_timestep_hours,
            )

            if not filename.exists():
                raise FileNotFoundError(f"Forecast file not found: {filename}")

            print(
                f"Processing forecast file: {filename.name}"
            )  # Log the filename being processed

            return xr.open_dataset(  # Open GRIB file as xarray Dataset
                filename,
                engine="cfgrib",  # Use cfgrib engine for GRIB files
            ).rename(
                {"latitude": "y", "longitude": "x", "number": "member"}
            )  # Rename dimensions to standard names

        def _validate_forecast_compatibility(
            control_ds: xr.Dataset, ensemble_ds: xr.Dataset
        ) -> None:
            """Validate that control and ensemble forecasts are compatible for merging.

            Args:
                control_ds: Control forecast dataset.
                ensemble_ds: Ensemble forecast dataset.

            Raises:
                ValueError: If forecasts have incompatible dimensions or initialization times.
            """
            # Check initialization times match
            if not np.array_equal(control_ds.time.values, ensemble_ds.time.values):
                raise ValueError(
                    "Control and ensemble forecasts have different initialization times. "
                    f"Control: {control_ds.time.values[0]}, Ensemble: {ensemble_ds.time.values[0]}"
                )

            # Check forecast steps match
            if not np.array_equal(control_ds.step.values, ensemble_ds.step.values):
                raise ValueError(
                    "Control and ensemble forecasts have different forecast steps. "
                    f"Control steps: {len(control_ds.step)}, Ensemble steps: {len(ensemble_ds.step)}"
                )

            # Check spatial dimensions match
            if not (
                np.allclose(control_ds.x.values, ensemble_ds.x.values)
                and np.allclose(control_ds.y.values, ensemble_ds.y.values)
            ):
                raise ValueError(
                    "Control and ensemble forecasts have different spatial coordinates"
                )

            # Check variables match
            control_vars = set(control_ds.data_vars)
            ensemble_vars = set(ensemble_ds.data_vars)
            if control_vars != ensemble_vars:
                raise ValueError(
                    f"Control and ensemble forecasts have different variables. "
                    f"Control: {control_vars}, Ensemble: {ensemble_vars}"
                )

        # Load forecast datasets based on YAML forecast_model parameter
        if forecast_model == "both_control_and_probabilistic":
            # Load both_control_and_probabilistic control and ensemble forecasts for combination
            control_ds = _load_forecast_file("control_forecast")
            ensemble_ds = _load_forecast_file("probabilistic_forecast")
            # Validate compatibility before merging
            _validate_forecast_compatibility(control_ds, ensemble_ds)

            # Assign member number 0 to control forecast (ECMWF convention)
            control_ds = control_ds.expand_dims(dim={"member": [0]})
            # Ensure ensemble members start from 1 (adjust if they start from 0)
            if ensemble_ds.member.min().item() == 0:
                ensemble_ds = ensemble_ds.assign_coords(member=ensemble_ds.member + 1)
            # Combine control and ensemble forecasts
            ds = xr.concat([control_ds, ensemble_ds], dim="member")
            print(
                f"Combined control and ensemble forecasts: {len(ds.member)} total members"
            )
        elif forecast_model in ["control_forecast", "probabilistic_forecast"]:
            # Load single forecast type without combining
            ds = _load_forecast_file(forecast_model)

            # Add member dimension to control forecast if not present for consistency
            if forecast_model == "control_forecast" and "member" not in ds.dims:
                ds = ds.expand_dims(dim={"member": [0]})

        else:
            raise ValueError(
                f"Unsupported forecast_model: '{forecast_model}'. "
                "Must be 'control_forecast', 'probabilistic_forecast', or 'both_control_and_probabilistic'."
            )

        # ensure all the timesteps are hourly
        if not (
            ds.step.diff("step").astype(np.int64) == 3600 * 1e9
        ).all():  # Check if all time differences are exactly 1 hour (3600 seconds in nanoseconds)
            # print all the unique timesteps in the time dimension
            print(
                f"Timesteps in the forecast are not hourly, resampling to hourly. Found timesteps: {np.unique(ds.step.diff('step').astype(np.int64) / 1e9 / 3600)} hours"
            )  # Log the current timesteps found in the data

            ds = ds.resample(step="1H").interpolate(
                "linear"
            )  # Resample to hourly timesteps using linear interpolation
            # convert back to float32
            ds = ds.astype(
                np.float32
            )  # Convert data type back to float32 to save memory
        else:
            print(
                "All timesteps are already hourly, no need to resample"
            )  # Log that resampling is not needed

        ds["tp"] = ds["tp"] * 1000  # Convert precipitation from meters to millimeters
        ds["tp"] = ds["tp"] / 3600  # Convert precipitation from mm/hr to mm/s
        ds["tp"] = ds["tp"].diff(
            dim="step", n=1, label="lower"
        )  # De-accumulate precipitation by taking differences between consecutive time steps
        if (
            len(list(ds.data_vars)) > 1
        ):  # Check if there are multiple variables (more than just precipitation)
            ds["ssrd"] = ds["ssrd"].diff(
                dim="step", n=1, label="lower"
            )  # De-accumulate shortwave radiation
            ds["ssrd"] = (
                ds["ssrd"] / 3600
            )  # Convert shortwave radiation from J/m2 to W/m2 by dividing by 3600 seconds
            ds["strd"] = ds["strd"].diff(
                dim="step", n=1, label="lower"
            )  # De-accumulate longwave radiation
            ds["strd"] = (
                ds["strd"] / 3600
            )  # Convert from J/m2 to W/m2 by dividing by 3600 seconds

        ds = ds.assign_coords(
            valid_time=ds.time + ds.step
        )  # Create valid_time coordinate by adding forecast initialization time to forecast step
        ds = ds.swap_dims(
            {"step": "valid_time"}
        )  # Swap step dimension with valid_time to make valid_time the main time dimension
        ds = ds.drop_vars(
            ["time", "step", "surface"]
        )  # Remove unnecessary coordinate variables
        ds = ds.rename(
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
            west_ds: xr.Dataset = ds.sel(  # Select western hemisphere data
                y=slice(
                    bounds[3] + buffer, bounds[1] - buffer
                ),  # Latitude slice (note: reversed for GRIB convention)
                x=slice(
                    ((bounds[0] - buffer) + 360) % 360, 360
                ),  # Longitude slice for western part
            )
            # Get eastern hemisphere part (longitude > 0)
            east_ds: xr.Dataset = ds.sel(  # Select eastern hemisphere data
                y=slice(bounds[3] + buffer, bounds[1] - buffer),  # Same latitude slice
                x=slice(
                    0, ((bounds[2] + buffer) + 360) % 360
                ),  # Longitude slice for eastern part
            )
            # Combine the two parts
            ds: xr.Dataset = xr.concat(
                [west_ds, east_ds], dim="x"
            )  # Concatenate western and eastern parts along longitude dimension
        else:
            # Regular case - doesn't cross meridian
            if (
                ds.x.min() >= 0 and ds.x.max() <= 360
            ):  # Check if longitude coordinates are in 0-360 format (probably GRIB2 files)
                ds: xr.Dataset = ds.sel(  # Select data using 0-360 longitude format
                    y=slice(bounds[3] + buffer, bounds[1] - buffer),  # Latitude slice
                    x=slice(
                        ((bounds[0] - buffer) + 360)
                        % 360,  # Convert min longitude to 0-360 format
                        ((bounds[2] + buffer) + 360)
                        % 360,  # Convert max longitude to 0-360 format
                    ),
                )
            else:  # Longitude coordinates are in -180 to 180 format (probably GRIB1 files)
                ds: xr.Dataset = (
                    ds.sel(  # Select data using -180 to 180 longitude format
                        y=slice(
                            bounds[3] + buffer, bounds[1] - buffer
                        ),  # Latitude slice
                        x=slice(
                            bounds[0] - buffer, bounds[2] + buffer
                        ),  # Longitude slice with buffer
                    )
                )

        # Reorder x to be between -180 and 180 degrees
        ds: xr.Dataset = ds.assign_coords(
            x=((ds.x + 180) % 360 - 180)
        )  # Convert longitude coordinates to -180 to 180 format
        ds.attrs["_FillValue"] = np.nan  # Set fill value attribute for missing data
        ds: xr.DataArray = convert_nodata(ds, np.nan)

        # assert that time is monotonically increasing with a constant step size
        assert (
            ds.time.diff("time").astype(np.int64)
            == (ds.time[1] - ds.time[0]).astype(np.int64)
        ).all(), (
            "time is not monotonically increasing with a constant step size"
        )  # Validate that time dimension is properly ordered with constant intervals

        ds = ds.rio.write_crs(
            4326
        )  # Set coordinate reference system to WGS84 (EPSG:4326)

        ds = ds.interp(  # Interpolate forecast data to match the target grid
            x=reproject_like.x,  # Target longitude coordinates
            y=reproject_like.y,  # Target latitude coordinates
            method="linear",  # Use linear interpolation
        )
        # convert back to float32
        ds = ds.astype(np.float32)  # Convert back to float32 to save memory

        # Handling of nan values and interpolation
        for variable_name in ds.data_vars:  # Loop through all variables in the dataset
            variable_data: xr.DataArray = ds[
                variable_name
            ]  # Get data for current variable
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
                print(
                    f"Found {nan_percentage:.2f}% missing values for variable '{variable_name}' after regridding. Interpolating missing values."
                )  # Log warning about missing values
                ds = ds.interpolate_na(
                    dim=["y", "x"], method="nearest"
                )  # Interpolate NaN values spatially using nearest neighbor
                ds = ds.interpolate_na(
                    dim=["time"], method="nearest"
                )  # Interpolate NaN values temporally using nearest neighbor

                # fill nans in last timesteps (due to de-accumulation) with mean of recent known values
                recent_mean: xr.DataArray = (  # Calculate mean of recent time steps for gap filling
                    ds[variable_name]
                    .isel(
                        time=slice(-25, -1)
                    )  # Select last 25 time steps (excluding the very last one)
                    .mean(
                        dim="time", skipna=True, keep_attrs=True
                    )  # Calculate mean, skipping NaN values
                )
                # Fill any remaining NaNs with the recent mean
                ds[variable_name] = ds[variable_name].fillna(
                    recent_mean
                )  # Fill remaining NaN values with calculated mean

                nan_percentage_after: float = float(
                    ds[variable_name].isnull().mean().compute().item()
                    * 100  # Check percentage of NaN values after interpolation
                )
                assert (
                    nan_percentage_after == 0
                ), (  # Assert that all NaN values have been filled
                    f"Failed to interpolate all missing values for variable '{variable_name}'. "
                    f"{nan_percentage_after:.2f}% missing values remain."
                )

        return ds
