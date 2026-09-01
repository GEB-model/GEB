"""ECMWF data adapter module."""

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

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


def generate_forecast_steps(forecast_date: datetime, forecast_horizon: int) -> str:
    """Generate ECMWF forecast step string based on the forecast date and horizon.

    ECMWF does not have a consistent 1h timestep for the entire operational archive. Asking hourly data to the server when it does not exist, will result in an error.
    Therefore, we need to adjust the requested steps based on the available data, which is different before and after 2016-11-23:
    - Before 2016-11-23: 3-hourly steps from 0-144h, 6-hourly steps from 144-360h
    - From 2016-11-23 onwards: hourly steps from 0-90h, 3-hourly steps from 90-144h, 6-hourly steps from 144-360h

    Args:
        forecast_date: The forecast initialization date and time.
        forecast_horizon: The forecast horizon in hours.

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
        if forecast_horizon <= 90:
            steps.extend(
                range(0, forecast_horizon + 1)
            )  # hourly steps up to forecast_horizon
        elif forecast_horizon <= 144:
            steps.extend(range(0, 91))  # hourly steps from 0-90h
            steps.extend(
                range(93, forecast_horizon + 1, 3)
            )  # 3-hourly steps from 90h to forecast_horizon
        else:
            steps.extend(range(0, 91))  # 0, 1, 2, 3, ..., 90 (hourly)
            steps.extend(range(93, 145, 3))  # 93, 96, 99, ..., 144 (3-hourly from 90h)
            steps.extend(
                range(150, 241, 6)
            )  # 150, 156, 162, ..., 360 (6-hourly from 144h)

    return "/".join(str(step) for step in steps)  # return step string for MARS request


def make_hindcast_dates_for_cycle_date(
    forecast_cycle_date: pd.Timestamp, n_hindcast_years: int
) -> str:
    """Generate a string of hindcast dates for a given forecast cycle date.

    Args:
        forecast_cycle_date: The forecast cycle date as a pandas Timestamp.
        n_hindcast_years: The number of hindcast years to request.

    Returns:
        A string of hindcast dates in the format "YYYY-MM-DD/YYYY-MM-DD/...".
    """
    forecast_cycle_date = pd.to_datetime(
        forecast_cycle_date
    )  # Ensure input is a Timestamp
    start_year = forecast_cycle_date.year - n_hindcast_years
    return "/".join(
        f"{year}-{forecast_cycle_date.month:02d}-{forecast_cycle_date.day:02d}"
        for year in range(start_year, forecast_cycle_date.year)
    )


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
        hindcast_cycle_start: date | datetime,
        hindcast_cycle_end: date | datetime,
        n_hindcast_years: int,
        forecast_model: str,
        forecast_resolution: str,
        forecast_horizon: int,
        forecast_timestep_hours: int,
        n_ensemble_members: int,
        forecast_product: Literal["forecast", "hindcast"],
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
            hindcast_cycle_start: The start date of the cycle you want to get the hindcasts for.
            hindcast_cycle_end: The end date of the cycle you want to get the hindcasts for.
            n_hindcast_years: The number of years of hindcast data to download before the forecast cycle date. Maximum is 20 years.
            forecast_model: The ECMWF forecast model to use ("probabilistic_forecast", "control_forecast" or "both_control_and_probabilistic").
            forecast_resolution: The spatial resolution of the forecast data (degrees).
            forecast_horizon: The forecast horizon in hours.
            forecast_timestep_hours: The forecast timestep in hours.
            n_ensemble_members: The number of ensemble members to download.
            forecast_product: The type of forecast product to download ("forecast" or "hindcast").

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
            forecast_start, forecast_end, freq="24h"
        )  # Generate list of forecast dates at 24-hour intervals
        earliest_allowed_date = date(2010, 1, 1)  # Set earliest allowed forecast date

        if forecast_product == "forecast":
            for forecast_date in forecast_date_list:  # Loop through all forecast dates
                if (
                    forecast_date.date() < earliest_allowed_date
                ):  # Check if date is before allowed range
                    raise ValueError(
                        f"Forecast date {forecast_date.date()} is before 2010-01-01. "
                        "For historical data before 2010, please use hindcast data instead."
                    )
        elif forecast_product == "hindcast":
            # If downloading hindcast data, check if the forecast start date is less then 20 years apart from the forecast cycle date, otherwise there will be no data available
            if n_hindcast_years > 20:
                raise ValueError(
                    "ECMWF hindcast data is only available for up to 20 years before the forecast cycle date. Please adjust the n_hindcast_years parameter in your build.yml file to be 20 or less."
                )

            HINDCAST_RUN_DAYS = [1, 5, 9, 13, 17, 21, 25, 29]
            forecast_date_list = [
                d
                for d in pd.date_range(
                    hindcast_cycle_start, hindcast_cycle_end, freq="D"
                )
                if d.day in HINDCAST_RUN_DAYS and not (d.month == 2 and d.day > 28)
            ]
        else:
            raise ValueError(
                f"Unsupported forecast_product: '{forecast_product}'. "
                "Must be 'forecast' or 'hindcast'."
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
            self.logger.info(f"Processing {model_type} downloads...")
            for (
                forecast_date
            ) in forecast_date_list:  # Loop through each forecast date to download
                self.logger.info(
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
                        forecast_date, forecast_horizon
                    )  # Generate forecast steps based on date using helper function
                elif (
                    forecast_timestep_hours >= 6
                ):  # Check if 6+ hourly timestep is requested
                    mars_step: str = f"0/to/{forecast_horizon}/BY/{forecast_timestep_hours}"  # Create step string for multi-hour intervals
                else:
                    raise ValueError(
                        f"Forecast timestep {forecast_timestep_hours} is not supported. Please use 1 or >=6."
                    )

                if forecast_product == "hindcast":
                    mars_stream = "enfh"  # Ensemble hindcast stream
                else:
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

                if forecast_product == "forecast":
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
                elif forecast_product == "hindcast":
                    # retrieve steps from mars
                    mars_request: dict[
                        str, Any
                    ] = {  # Build MARS request dictionary with all parameters
                        "class": mars_class,
                        "hdate": make_hindcast_dates_for_cycle_date(
                            forecast_cycle_date=forecast_date.strftime("%Y-%m-%d"),
                            n_hindcast_years=n_hindcast_years,
                        ),
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
                    forecast_product=forecast_product,
                    forecast_date=format_date(forecast_date),
                    forecast_model=forecast_model,
                    forecast_resolution=forecast_resolution.replace("/", "-"),
                    forecast_horizon=forecast_horizon,
                    forecast_timestep_hours=forecast_timestep_hours,
                )

                if output_filename.exists():
                    self.logger.info(
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

                self.logger.info(
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

    def load_and_merge_forecast_files(
        self,
        forecast_model: str,
        forecast_issue_date: pd.Timestamp,
        forecast_resolution: str,
        forecast_horizon: int,
        forecast_timestep_hours: int,
        forecast_product: Literal["forecast", "hindcast"] = "forecast",
    ) -> xr.Dataset:
        """Load and merge ECMWF forecast files based on the specified model type.

        Args:
            forecast_model: Either 'control_forecast', 'probabilistic_forecast', or 'both_control_and_probabilistic'.
            forecast_issue_date: The forecast initialization date and time.
            forecast_resolution: The spatial resolution of the forecast data (degrees).
            forecast_horizon: The forecast horizon in hours.
            forecast_timestep_hours: The temporal resolution of the forecast data in hours.
            forecast_product: The forecast product type (e.g., "hindcast" or "forecast").

        Returns:
            Merged forecast dataset.

        Raises:
            ValueError: If the forecast model is not supported.
        """

        def _load_forecast_files(forecast_model: str) -> xr.Dataset:
            """Load a single forecast dataset for the specified model type.

            Args:
                forecast_model: Either 'control_forecast' or 'probabilistic_forecast'.

            Returns:
                Loaded and renamed forecast dataset.

            Raises:
                FileNotFoundError: If the forecast file doesn't exist.
            """
            filename = format_path(
                self.path,
                forecast_product=forecast_product,
                forecast_date=format_date(forecast_issue_date),
                forecast_model=forecast_model,
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
            control_ds = _load_forecast_files("control_forecast")
            ensemble_ds = _load_forecast_files("probabilistic_forecast")
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
            ds = _load_forecast_files(forecast_model)

            # Add member dimension to control forecast if not present for consistency
            if forecast_model == "control_forecast" and "member" not in ds.dims:
                ds = ds.expand_dims(dim={"member": [0]})

        else:
            raise ValueError(
                f"Unsupported forecast_model: '{forecast_model}'. "
                "Must be 'control_forecast', 'probabilistic_forecast', or 'both_control_and_probabilistic'."
            )

        return ds

    def create_rainfall_statistics_table():
        return_periods = np.array(
            [0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]
        )
        duration = np.array([1 / 6, 1 / 2, 1, 2, 4, 8, 12, 24, 48, 96, 192])  # in hours
        rainfall_values = np.array(
            [
                # 10 min, 30 min, 1 h, 2 h, 4 h, 8 h, 12 h, 24 h, 2 d, 4 d, 8 d
                [
                    8.1,
                    10.4,
                    12.6,
                    15.3,
                    18.6,
                    22.2,
                    24.6,
                    30.4,
                    38.6,
                    50.4,
                    68.3,
                ],  # T = 0.5 years
                [
                    10.2,
                    13.5,
                    16.2,
                    19.5,
                    23.4,
                    27.7,
                    30.5,
                    36.8,
                    46.0,
                    59.3,
                    79.4,
                ],  # T = 1 year
                [
                    12.2,
                    16.6,
                    20.0,
                    24.0,
                    28.4,
                    33.4,
                    36.5,
                    43.8,
                    54.0,
                    68.6,
                    90.5,
                ],  # T = 2 years
                [
                    15.1,
                    21.2,
                    25.8,
                    30.7,
                    35.9,
                    41.7,
                    45.2,
                    54.2,
                    65.5,
                    81.4,
                    105.1,
                ],  # T = 5 years
                [
                    17.5,
                    25.3,
                    31.0,
                    36.8,
                    42.8,
                    49.1,
                    52.9,
                    63.0,
                    74.9,
                    91.6,
                    116.1,
                ],  # T = 10 years
                [
                    20.3,
                    30.2,
                    37.2,
                    44.2,
                    51.1,
                    58.0,
                    61.9,
                    72.6,
                    85.0,
                    102.1,
                    127.0,
                ],  # T = 20 years
                [
                    21.3,
                    32.0,
                    39.5,
                    46.9,
                    54.1,
                    61.2,
                    65.2,
                    75.9,
                    88.5,
                    105.6,
                    130.5,
                ],  # T = 25 years
                [
                    24.7,
                    38.2,
                    47.7,
                    56.5,
                    64.8,
                    72.5,
                    76.6,
                    86.9,
                    99.5,
                    116.6,
                    141.5,
                ],  # T = 50 years
                [
                    28.7,
                    45.8,
                    57.7,
                    68.4,
                    78.0,
                    86.2,
                    90.2,
                    98.9,
                    111.4,
                    128.1,
                    152.3,
                ],  # T = 100 years
                [
                    33.4,
                    55.0,
                    70.0,
                    81.3,
                    88.7,
                    95.0,
                    98.1,
                    112.1,
                    124.2,
                    140.0,
                    163.2,
                ],  # T = 200 years
                [
                    35.0,
                    58.4,
                    74.5,
                    86.5,
                    93.9,
                    100.0,
                    102.9,
                    116.7,
                    128.5,
                    143.9,
                    166.7,
                ],  # T = 250 years
                [
                    40.8,
                    70.4,
                    90.7,
                    105.0,
                    112.2,
                    117.5,
                    119.6,
                    131.7,
                    142.5,
                    156.4,
                    177.5,
                ],  # T = 500 years
                [
                    47.6,
                    84.9,
                    110.6,
                    127.6,
                    134.4,
                    138.3,
                    139.2,
                    148.2,
                    157.5,
                    169.4,
                    188.3,
                ],  # T = 1000 years
            ]
        )

        rainfall_statistics = xr.DataArray(
            data=rainfall_values,
            dims=["return_period", "duration"],
            coords={"return_period": return_periods, "duration": duration},
            name="rainfall_statistics",
        )

        rainfall_statistics.attrs["description"] = (
            "Rainfall statistics for different return periods and durations based on STOWA depth-duration curves"
        )
        rainfall_statistics.attrs["units"] = "mm"

        statistical_rainfall_6h = rainfall_statistics.interp(
            duration=6
        )  # Interpolate to get rainfall values for 6-hour duration across all return periods
        rainfall_statistics = xr.concat(
            [rainfall_statistics, statistical_rainfall_6h], dim="duration"
        )
        rainfall_statistics = rainfall_statistics.sortby("duration")

        rainfall_distribution_abs = np.zeros(
            (len(return_periods), 6), dtype=float
        )  # Initialize an array to hold rainfall distribution values

        rainfall_distribution_abs[:, 3] = rainfall_statistics.sel(
            duration=1
        ).values  # 1-hour rainfall
        rainfall_distribution_abs[:, 2] = (
            rainfall_statistics.sel(duration=2).values
            - rainfall_statistics.sel(duration=1).values
        )  # 2-hour rainfall minus 1-hour rainfall
        rainfall_distribution_abs[:, 1] = (
            rainfall_statistics.sel(duration=4).values
            - (rainfall_distribution_abs[:, 3] + rainfall_distribution_abs[:, 2])
        ) / 2  # 4-hour rainfall minus 2-hour rainfall
        rainfall_distribution_abs[:, 4] = rainfall_distribution_abs[:, 1]
        rainfall_distribution_abs[:, 0] = (
            rainfall_statistics.sel(duration=6).values
            - (
                rainfall_distribution_abs[:, 3]
                + rainfall_distribution_abs[:, 2]
                + rainfall_distribution_abs[:, 1]
                + rainfall_distribution_abs[:, 4]
            )
        ) / 2  # 6-hour rainfall minus sum of previous
        rainfall_distribution_abs[:, 5] = rainfall_distribution_abs[:, 0]

        rainfall_distribution_percentage = (
            rainfall_distribution_abs
            / rainfall_statistics.sel(duration=6).values[:, np.newaxis]
        )  # Convert absolute values to percentages based on 6-hour rainfall
        rainfall_distribution_percentage[0, :] = (
            1 / 6
        )  # For the 0.5-year return period, distribute the rainfall evenly across all timesteps
        rainfall_distribution_percentage = rainfall_distribution_percentage.round(3)

        rainfall_distribution_timestep = np.array([1, 2, 3, 4, 5, 6])  # in hours
        rainfall_distribution_percentage = xr.DataArray(
            data=rainfall_distribution_percentage,
            dims=["return_period", "distribution_timestep"],
            coords={
                "return_period": return_periods,
                "distribution_timestep": rainfall_distribution_timestep,
            },
            name="rainfall_distribution_percentage",
        )

        return rainfall_statistics, rainfall_distribution_percentage

    def process_hindcasts(
        self,
        ds: xr.Dataset,
        bounds: tuple[float, float, float, float],
        reproject_like: xr.DataArray,
    ) -> xr.Dataset:
        """Process ECMWF forecast dataset.

        Args:
            forecast_dataset: The xarray Dataset containing the forecast data.
            bounds: The bounding box in the format (min_lon, min_lat, max_lon, max_lat).
            reproject_like: An xarray DataArray to use as a template for reprojecting
                the forecast data.
        Returns:
            Processed forecast dataset.
        """
        ds["tp"] = ds["tp"] * 1000  # Convert precipitation from meters to millimeters
        ds["tp"] = ds["tp"].diff(
            dim="step", n=1, label="upper"
        )  # De-accumulate precipitation by taking differences between consecutive time steps

        rainfall_statistics, rainfall_distribution_percentage = (
            self.create_rainfall_statistics_table()
        )  # Create rainfall statistics and distribution tables

        rainfall_6h_statistics = rainfall_statistics.sel(duration=6)

        steps = ds.step.values[1:]

        hourly_blocks = []
        for step in steps:
            rainfall_6h_forecast = ds["tp"].sel(step=step)

            closest_return_period = abs(
                rainfall_6h_statistics - rainfall_6h_forecast
            ).idxmin(dim="return_period")

            # Pick the rainfall distribution (%) corresponding to the closest return period
            selected_distribution = rainfall_distribution_percentage.sel(
                return_period=closest_return_period
            )

            # Distribuir o total de chuva de 6 h entre as seis horas
            hourly_rainfall_forecast = rainfall_6h_forecast * selected_distribution

            # Create the new hourly step values by subtracting 6 hours from the original step and adding the hour offset
            new_steps = (
                step
                - np.timedelta64(6, "h")
                + hourly_rainfall_forecast["distribution_timestep"].values
                * np.timedelta64(1, "h")
            )

            # Change the "hour" dimension to a hourly step dimension called "step"
            hourly_rainfall_forecast = (
                hourly_rainfall_forecast.assign_coords(
                    step=("distribution_timestep", new_steps)
                )
                .swap_dims({"distribution_timestep": "step"})
                .drop_vars("distribution_timestep")
            )

            hourly_blocks.append(hourly_rainfall_forecast)

        hourly_rainfall_forecasts = xr.concat(
            hourly_blocks,
            dim="step",
        ).sortby("step")

        ds = ds.resample(
            step="1h"
        ).interpolate(
            "linear"
        )  # Resample to hourly timesteps using linear interpolation (will replace interpolated rainfall later with the hourly rainfall distribution calculated above)

        ds = ds.assign_coords(valid_time=(ds["time"] + ds["step"]))

        # forecast_dataset["tp"] = hourly_rainfall_forecasts  # Replace the interpolated rainfall with the hourly rainfall distribution calculated above
        ds["tp"].loc[{"step": hourly_rainfall_forecasts.step}] = (
            hourly_rainfall_forecasts
        )
        ds["tp"] = ds["tp"] / 3600  # Convert precipitation from mm/hr to mm/s

        if (
            len(list(ds.data_vars)) > 1
        ):  # Check if there are multiple variables (more than just precipitation)
            ds["ssrd"] = ds["ssrd"].diff(
                dim="step", n=1, label="upper"
            )  # De-accumulate shortwave radiation
            ds["ssrd"] = (
                ds["ssrd"] / 3600
            )  # Convert shortwave radiation from J/m2 to W/m2 by dividing by 3600 seconds
            ds["strd"] = ds["strd"].diff(
                dim="step", n=1, label="upper"
            )  # De-accumulate longwave radiation
            ds["strd"] = (
                ds["strd"] / 3600
            )  # Convert from J/m2 to W/m2 by dividing by 3600 seconds

        # # ensure all the timesteps are hourly
        # if not (
        #     ds.step.diff("step").astype(np.int64) == 3600 * 1e9
        # ).all():  # Check if all time differences are exactly 1 hour (3600 seconds in nanoseconds)
        #     # print all the unique timesteps in the time dimension
        #     print(
        #         f"Timesteps in the forecast are not hourly, resampling to hourly. Found timesteps: {np.unique(ds.step.diff('step').astype(np.int64) / 1e9 / 3600)} hours"
        #     )  # Log the current timesteps found in the data

        #     ds_resampled = ds.resample(step="1h").interpolate(
        #         "linear"
        #     )  # Resample to hourly timesteps using linear interpolation

        #     # convert back to float32
        #     ds_resampled = ds_resampled.astype(
        #         np.float32
        #     )  # Convert data type back to float32 to save memory
        # else:
        #     print(
        #         "All timesteps are already hourly, no need to resample"
        #     )  # Log that resampling is not needed

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
        ds: xr.Dataset = convert_nodata(ds, np.nan)

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
            # fill the nan values using interpolate_na_along_dim and interpolate_na in space
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

    def process_forecasts(
        self,
        ds: xr.Dataset,
        bounds: tuple[float, float, float, float],
        reproject_like: xr.DataArray,
    ) -> xr.Dataset:
        """Process ECMWF forecast dataset.

        Args:
            ds: The xarray Dataset containing the forecast data.
            bounds: The bounding box in the format (min_lon, min_lat, max_lon, max_lat).
            reproject_like: An xarray DataArray to use as a template for reprojecting
                the forecast data.
        Returns:
            Processed forecast dataset.
        """
        # TODO: Move interpolation to after deaccumulation, as interpolation of accumulated values can lead to incorrect results
        # ensure all the timesteps are hourly
        if not (
            ds.step.diff("step").astype(np.int64) == 3600 * 1e9
        ).all():  # Check if all time differences are exactly 1 hour (3600 seconds in nanoseconds)
            # print all the unique timesteps in the time dimension
            print(
                f"Timesteps in the forecast are not hourly, resampling to hourly. Found timesteps: {np.unique(ds.step.diff('step').astype(np.int64) / 1e9 / 3600)} hours"
            )  # Log the current timesteps found in the data

            ds = ds.resample(step="1h").interpolate(
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
        ds: xr.Dataset = convert_nodata(ds, np.nan)

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
            # fill the nan values using interpolate_na_along_dim and interpolate_na in space
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

    def read_and_process_forecasts(
        self,
        bounds: tuple[float, float, float, float],
        forecast_issue_date: datetime,
        forecast_model: str,
        forecast_resolution: str,
        forecast_horizon: int,
        forecast_timestep_hours: int,
        reproject_like: xr.DataArray,
    ) -> xr.Dataset:
        """Process downloaded ECMWF forecast data.

        We process forecasts for each initialization time separately. The forecast file contains all variables needed for GEB.

        Args:
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
        """
        ds = self.load_and_merge_forecast_files(
            forecast_model=forecast_model,
            forecast_issue_date=forecast_issue_date,
            forecast_resolution=forecast_resolution,
            forecast_horizon=forecast_horizon,
            forecast_timestep_hours=forecast_timestep_hours,
            forecast_product="forecast",
        )

        return self.process_forecasts(ds, bounds, reproject_like)

    def read_and_process_hindcasts(
        self,
        bounds: tuple[float, float, float, float],
        forecast_issue_date: datetime,
        forecast_model: str,
        forecast_resolution: str,
        forecast_horizon: int,
        forecast_timestep_hours: int,
        reproject_like: xr.DataArray,
    ) -> dict[str, xr.Dataset]:
        """Process downloaded ECMWF hindcast data.

        We process hindcasts for each initialization time separately. The hindcast file contains all variables needed for GEB.

        Args:
            bounds: The bounding box in the format (min_lon, min_lat, max_lon,
                    max_lat).
            forecast_issue_date: The forecast initialization time.
            forecast_model: The ECMWF forecast model from build.yml config ("probabilistic_forecast", "control_forecast" or "both_control_and_probabilistic").
            forecast_resolution: The spatial resolution of the hindcast data (degrees).
            forecast_horizon: The forecast horizon in hours.
            forecast_timestep_hours: The forecast timestep in hours.
            reproject_like: An xarray DataArray to use as a template for reprojecting
                the hindcast data.

        Returns:
            processed_hindcasts: A dictionary containing the processed hindcast data
                for each initialization time.
        """
        hindcasts = self.load_and_merge_forecast_files(
            forecast_model=forecast_model,
            forecast_issue_date=forecast_issue_date,
            forecast_resolution=forecast_resolution,
            forecast_horizon=forecast_horizon,
            forecast_timestep_hours=forecast_timestep_hours,
            forecast_product="hindcast",
        )

        processed_hindcasts = {}

        for hindcast_date in hindcasts.time.values:
            hindcast_date_str = pd.to_datetime(hindcast_date).strftime(
                "%Y%m%dT%H%M%S"
            )  # Format date for filenames

            hindcast = hindcasts.sel(
                time=hindcast_date
            )  # Select the hindcast for the specific initialization time

            processed_hindcasts[hindcast_date_str] = self.process_forecasts(
                ds=hindcast,
                bounds=bounds,
                reproject_like=reproject_like,
            )

        return processed_hindcasts
