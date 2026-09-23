"""Data adapter for obtaining data from Destination Earth."""

import base64
import os
import time
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import fsspec
import numpy as np
import xarray as xr
import zarr.storage
from aiohttp_retry import ExponentialRetry, RetryClient
from fsspec.asyn import AsyncFileSystem

from geb.workflows.raster import convert_nodata

from .base import Adapter

N_CONNECTION_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

DESTINATION_EARTH_DATASETS: dict[str, dict[str, Any]] = {
    "era5": {
        "pat_url": (
            "https://data.earthdatahub.destine.eu/"
            "era5/reanalysis-era5-land-no-antartica-v0.zarr"
        ),
        "api_url": (
            "https://api.earthdatahub.destine.eu/"
            "era5/reanalysis-era5-land-no-antartica-v0.zarr"
        ),
        "zarr_format": 2,
        "processing": "era5",
    },
    "climate_dt_ssp370": {
        # Climate DT Generation 2 is served through the authenticated API endpoint
        # for both supported Destination Earth credential types.
        "pat_url": (
            "https://api.earthdatahub.destine.eu/"
            "climate-dt-2/IFS-NEMO-SSP3-7.0-sfc-hourly-standard-v0.zarr"
        ),
        "api_url": (
            "https://api.earthdatahub.destine.eu/"
            "climate-dt-2/IFS-NEMO-SSP3-7.0-sfc-hourly-standard-v0.zarr"
        ),
        "zarr_format": 3,
        "processing": "direct",
    },
}


async def get_retry_client(**kwargs: Any) -> RetryClient:
    """Create a RetryClient with exponential backoff for transient errors.

    Args:
        **kwargs: Additional keyword arguments passed to RetryClient.

    Returns:
        RetryClient configured with exponential backoff.
    """
    retry_options = ExponentialRetry(
        attempts=100,
        start_timeout=10,
        max_timeout=3600,
        factor=2,
        retry_all_server_errors=True,
    )
    return RetryClient(retry_options=retry_options, **kwargs)


class DestinationEarth(Adapter):
    """Data adapter for obtaining data from Destination Earth."""

    def __init__(
        self,
        *args: Any,
        dataset: str = "era5",
        **kwargs: Any,
    ) -> None:
        """Initialize the Destination Earth data adapter.

        Args:
            *args: Positional arguments passed to the base Adapter.
            dataset: Destination Earth dataset identifier. Defaults to ``era5``
                for backwards compatibility.
            **kwargs: Keyword arguments passed to the base Adapter.

        Raises:
            ValueError: If the requested dataset is not supported.
        """
        super().__init__(*args, **kwargs)

        if dataset not in DESTINATION_EARTH_DATASETS:
            raise ValueError(
                f"Unknown Destination Earth dataset: {dataset}. Supported datasets are "
                f"{list(DESTINATION_EARTH_DATASETS)}."
            )

        self.dataset = dataset

    def get_authentication_header(self) -> dict[str, str]:
        """Generate the authentication header for Destination Earth access.

        Returns:
            A dictionary containing the Authorization header for HTTP requests.

        Raises:
            ValueError: If the DESTINATION_EARTH_KEY environment variable is not set.
        """
        destination_earth_key: str | None = os.getenv(key="DESTINATION_EARTH_KEY")
        if destination_earth_key is None:
            print("ERROR: DESTINATION_EARTH_KEY environment variable is not set.")
            print(
                "Please set your API KEY in your .env file or export it in your shell."
            )
            raise ValueError("DESTINATION_EARTH_KEY environment variable is not set.")

        auth_string: str = f"edh:{destination_earth_key}"
        # Base64 encode the "username:password" string (edh:<PAT>)
        encoded_auth: str = base64.b64encode(auth_string.encode("utf-8")).decode(
            "utf-8"
        )

        auth_headers: dict[str, str] = {"Authorization": f"Basic {encoded_auth}"}
        return auth_headers

    def fetch(self, url: None) -> DestinationEarth:
        """Set the URL for the selected Destination Earth data source.

        Args:
            url: Must be None because the dataset URL is determined automatically.

        Returns:
            The current DestinationEarth adapter.

        Raises:
            ValueError: If DESTINATION_EARTH_KEY is not set or has an invalid format.
        """
        assert url is None, (
            "URL must be None for Destination Earth, as it is determined automatically."
        )

        destination_earth_key: str | None = os.getenv(key="DESTINATION_EARTH_KEY")
        if destination_earth_key is None:
            print("ERROR: DESTINATION_EARTH_KEY environment variable is not set.")
            print(
                "Please set your API KEY in your .env file or export it in your shell."
            )
            raise ValueError("DESTINATION_EARTH_KEY environment variable is not set.")

        dataset_config = DESTINATION_EARTH_DATASETS[self.dataset]

        if destination_earth_key.startswith("edh_pat_"):
            self.url = dataset_config["pat_url"]
        elif destination_earth_key.startswith("edh_key_"):
            self.url = dataset_config["api_url"]
        else:
            raise ValueError(
                "Invalid DESTINATION_EARTH_KEY format. It should start with "
                "'edh_pat_' for Personal Access Tokens or 'edh_key_' for API keys."
            )

        return self

    def connect_API(
        self,
        variable: str,
        start_date: datetime,
        end_date: datetime,
        bounds: tuple[float, float, float, float],
    ) -> xr.DataArray:
        """Read one variable from the selected Destination Earth dataset.

        Args:
            variable: Short name of the variable to read.
            start_date: Start date of the requested period.
            end_date: End date of the requested period.
            bounds: Bounding box as (min_lon, min_lat, max_lon, max_lat).

        Returns:
            Requested data as an xarray DataArray.

        Raises:
            ConnectionError: If the remote dataset cannot be opened after retries.
            KeyError: If the requested variable is not present in the dataset.
            ValueError: If Climate DT does not contain the required dimensions.
        """
        dataset_config = DESTINATION_EARTH_DATASETS[self.dataset]

        for attempt in range(N_CONNECTION_ATTEMPTS):
            try:
                fs: AsyncFileSystem = fsspec.filesystem(
                    protocol="https",
                    headers=self.get_authentication_header(),
                    get_client=get_retry_client,
                    asynchronous=True,
                    client_kwargs={
                        "trust_env": True,
                        "raise_for_status": False,  # Let RetryClient and fsspec handle status codes
                    },
                    timeout=600,
                )
                store = zarr.storage.FsspecStore(path=self.url, fs=fs)

                open_kwargs: dict[str, Any] = {
                    "filename_or_obj": store,
                    "chunks": {},
                    "engine": "zarr",
                    "zarr_format": dataset_config["zarr_format"],
                }
                # Preserve the existing ERA5 opening behavior exactly. Climate DT
                # is Zarr v3 and its official access example does not require
                # consolidated metadata.
                if self.dataset == "era5":
                    open_kwargs["consolidated"] = True

                ds: xr.Dataset = xr.open_dataset(**open_kwargs)
                break

            except (aiohttp.ClientResponseError, aiohttp.ClientPayloadError) as e:
                print(
                    f"Error connecting to Destination Earth API: {e}. Retrying "
                    f"({attempt + 1}/{N_CONNECTION_ATTEMPTS})..."
                )
                time.sleep(RETRY_DELAY_SECONDS * (2**attempt))
        else:
            raise ConnectionError(
                "Failed to connect to Destination Earth API after "
                f"{N_CONNECTION_ATTEMPTS} attempts."
            )

        if variable not in ds:
            raise KeyError(
                f"Variable '{variable}' is not available in Destination Earth "
                f"dataset '{self.dataset}'. Available variables are: "
                f"{list(ds.data_vars)}"
            )

        da: xr.DataArray = ds[variable]

        if self.dataset == "era5":
            # Keep ERA5 coordinate handling identical to the previous adapter.
            da = da.rename(
                {"valid_time": "time", "latitude": "y", "longitude": "x"}
            ).drop_vars(["number", "surface", "depthBelowLandLayer"], errors="ignore")

            buffer: float = 0.5
            buffered_bounds: tuple[float, float, float, float] = (
                bounds[0] - buffer,
                bounds[1] - buffer,
                bounds[2] + buffer,
                bounds[3] + buffer,
            )

            # Check if region crosses the meridian (longitude=0)
            # use a slightly larger slice. The resolution is 0.1 degrees, so buffer degrees is a bit more than that (to be sure)
            if buffered_bounds[0] < 0 and buffered_bounds[2] > 0:
                # Need to handle the split across the meridian
                # Get western hemisphere part (longitude < 0)
                west_da: xr.DataArray = da.sel(
                    time=slice(start_date, end_date),
                    y=slice(buffered_bounds[3], buffered_bounds[1]),
                    x=slice(((buffered_bounds[0]) + 360) % 360, 360),
                )
                # Get eastern hemisphere part (longitude > 0)
                east_da: xr.DataArray = da.sel(
                    time=slice(start_date, end_date),
                    y=slice(buffered_bounds[3], buffered_bounds[1]),
                    x=slice(0, ((buffered_bounds[2]) + 360) % 360),
                )
                # Combine the two parts
                da = xr.concat([west_da, east_da], dim="x")
            else:
                # Regular case - doesn't cross meridian
                da = da.sel(
                    time=slice(start_date, end_date),
                    y=slice(buffered_bounds[3], buffered_bounds[1]),
                    x=slice(
                        ((buffered_bounds[0]) + 360) % 360,
                        ((buffered_bounds[2]) + 360) % 360,
                    ),
                )

            da = da.chunk({"y": -1, "x": -1})

            # Reorder x to be between -180 and 180 degrees
            da = da.assign_coords(x=((da.x + 180) % 360 - 180))

            assert da.x.size > 0 and da.y.size > 0, (
                "No data found for the specified bounds."
            )

            da.attrs["_FillValue"] = da.attrs["GRIB_missingValue"]
            da = convert_nodata(da, np.nan)
            return da

        # Climate DT Generation 2 uses Zarr v3 and can expose either already
        # normalized coordinate names or latitude/longitude coordinate names.
        rename: dict[str, str] = {}
        if "valid_time" in da.dims or "valid_time" in da.coords:
            rename["valid_time"] = "time"
        if "latitude" in da.dims or "latitude" in da.coords:
            rename["latitude"] = "y"
        if "longitude" in da.dims or "longitude" in da.coords:
            rename["longitude"] = "x"
        if rename:
            da = da.rename(rename)

        da = da.drop_vars(["number", "surface", "depthBelowLandLayer"], errors="ignore")

        if "time" not in da.dims:
            raise ValueError(
                f"Climate DT variable '{variable}' does not contain a time dimension."
            )
        if "x" not in da.dims or "y" not in da.dims:
            raise ValueError(
                f"Climate DT variable '{variable}' does not contain x/y dimensions."
            )

        # Normalize Climate DT longitudes to [-180, 180] and sort before
        # selecting the model domain. This also handles European domains that
        # cross longitude 0 without special-case concatenation.
        da = da.assign_coords(x=((da.x + 180) % 360 - 180)).sortby("x")

        buffer = 0.5
        min_lon = bounds[0] - buffer
        min_lat = bounds[1] - buffer
        max_lon = bounds[2] + buffer
        max_lat = bounds[3] + buffer

        # Climate DT latitude ordering may differ between products. Select in
        # the direction of the actual coordinate rather than assuming north-to-south.
        if da.y.values[0] > da.y.values[-1]:
            y_slice = slice(max_lat, min_lat)
        else:
            y_slice = slice(min_lat, max_lat)

        da = da.sel(
            time=slice(start_date, end_date),
            y=y_slice,
            x=slice(min_lon, max_lon),
        )

        if da.time.size == 0:
            raise ValueError(
                f"No Climate DT data found for '{variable}' between "
                f"{start_date} and {end_date}."
            )

        assert da.x.size > 0 and da.y.size > 0, (
            "No Climate DT data found for the specified bounds."
        )

        da = da.chunk({"y": -1, "x": -1})

        # Zarr decoding generally converts Climate DT fill values already. Do
        # not require ERA5-specific GRIB_missingValue metadata here.
        if "GRIB_missingValue" in da.attrs:
            da.attrs["_FillValue"] = da.attrs["GRIB_missingValue"]
            da = convert_nodata(da, np.nan)

        return da

    def read(
        self,
        variable: str,
        start_date: datetime,
        end_date: datetime,
        bounds: tuple[float, float, float, float],
    ) -> xr.DataArray:
        """Process data for a given variable and time period.

        ERA5 retains the previous de-accumulation behavior. Climate DT
        Generation 2 variables used by GEB are already instantaneous values or
        hourly mean rates/fluxes and are therefore read directly.

        Args:
            variable: Short name of the variable to process.
            start_date: Start date of the requested period.
            end_date: End date of the requested period.
            bounds: Bounding box as (min_lon, min_lat, max_lon, max_lat).

        Raises:
            NotImplementedError: If an ERA5 GRIB step type is unsupported.
            ValueError: If Climate DT is not continuous at hourly resolution.

        Returns:
            Processed data as an xarray DataArray.
        """
        dataset_config = DESTINATION_EARTH_DATASETS[self.dataset]

        if dataset_config["processing"] == "era5":
            # This is the original ERA5 read path. Keep it unchanged for
            # backwards compatibility with all existing callers.
            da: xr.DataArray = self.connect_API(
                variable, start_date - timedelta(hours=1), end_date, bounds
            )

            # assert that time is monotonically increasing with a constant step size
            assert (
                da.time.diff("time").astype(np.int64)
                == (da.time[1] - da.time[0]).astype(np.int64)
            ).all(), "time is not monotonically increasing with a constant step size"

            if da.attrs["GRIB_stepType"] == "accum":
                da = xr.where(
                    da.isel(time=slice(1, None)).time.dt.hour == 1,
                    da.isel(time=slice(1, None)),
                    da.diff(dim="time", n=1),
                )
            elif da.attrs["GRIB_stepType"] == "instant":
                da = da.isel(time=slice(1, None))
            else:
                raise NotImplementedError

            assert da.time.dt.hour.min().item() == 0, "time does not start at hour 0"

        elif dataset_config["processing"] == "direct":
            da = self.connect_API(variable, start_date, end_date, bounds)

            # Climate DT variables used by GEB are already hourly values or
            # hourly mean rates/fluxes, so no ERA5-style de-accumulation is applied.
            if da.time.size > 1:
                time_values = da.time.values.astype("datetime64[ns]")
                time_diff = np.diff(time_values)
                if not np.all(time_diff == np.timedelta64(1, "h")):
                    invalid = np.where(time_diff != np.timedelta64(1, "h"))[0]
                    first = int(invalid[0])
                    raise ValueError(
                        f"Destination Earth dataset '{self.dataset}' is not "
                        f"continuous at hourly resolution for variable '{variable}'. "
                        f"First discontinuity is between {time_values[first]} and "
                        f"{time_values[first + 1]}."
                    )
        else:
            raise ValueError(
                f"Unknown Destination Earth processing mode: "
                f"{dataset_config['processing']}"
            )

        da = da.rio.write_crs(4326)
        return da
