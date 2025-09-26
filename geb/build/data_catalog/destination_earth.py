"""Data adapter for obtaining ERA5 data from the Destination Earth."""

from datetime import datetime
from typing import Any

import numpy as np
import xarray as xr

from geb.build.workflows.general import interpolate_na_along_time_dim

from .base import Adapter


class DestinationEarth(Adapter):
    """Data adapter for obtaining ERA5 data from the Destination Earth."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the ERA5 data adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> "DestinationEarth":
        """Set the URL for the Destination Earth data source.

        Args:
            url: The URL of the Destination Earth data source.

        Returns:
            The current instance of the DestinationEarth adapter.
        """
        self.url = url
        return self

    def connect_API(
        self,
        variable: str,
        start_date: datetime,
        end_date: datetime,
        bounds: tuple[float, float, float, float],
    ) -> xr.DataArray:
        """Download ERA5 data for a specific variable and time period.

        Args:
            variable: Short name of the variable to download (e.g., "t2m"). Codes can be found here: https://codes.ecmwf.int/grib/param-db/
            start_date: start date of the time period to download
            end_date: end date of the time period to download
            bounds:  bounding box in the format (min_lon, min_lat, max_lon, max_lat)

        Returns:
            Downloaded ERA5 data as an xarray DataArray.
        """
        da: xr.DataArray = xr.open_dataset(
            self.url,
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

        da.attrs["_FillValue"] = da.attrs["GRIB_missingValue"]
        da: xr.DataArray = da.raster.mask_nodata()
        return da

    def read(
        self,
        variable: str,
        start_date: datetime,
        end_date: datetime,
        bounds: tuple[float, float, float, float],
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
            xr.DataArray: Processed ERA5 data as an xarray DataArray.
        """
        da: xr.DataArray = self.connect_API(variable, start_date, end_date, bounds)
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

        # rechunk to have all data for a time step in one chunk
        da = da.chunk({"x": -1, "y": -1, "time": 24})

        da: xr.DataArray = da.rio.write_crs(4326)
        da: xr.DataArray = interpolate_na_along_time_dim(da)

        return da
