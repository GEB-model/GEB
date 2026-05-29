"""ECMWF geopotential data adapter."""

from typing import Any

import rioxarray  # noqa: F401
import xarray as xr

from geb.workflows.io import fetch_and_save

from .base import Adapter


class ECMWFGeopotential(Adapter):
    """Adapter for the static ECMWF geopotential dataset used in forcing setup."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the ECMWF geopotential adapter.

        Args:
            *args: Positional arguments passed to the base adapter.
            **kwargs: Keyword arguments passed to the base adapter.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> ECMWFGeopotential:
        """Download the ECMWF geopotential dataset if it is not cached yet.

        Args:
            url: Direct download URL for the geopotential NetCDF file.

        Returns:
            The adapter instance.
        """
        if not self.is_ready:
            fetch_and_save(url=url, file_path=self.path)
        return self

    def read(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        """Read and normalize the ECMWF geopotential dataset.

        Notes:
            The upstream file stores a singleton time dimension and uses
            `latitude` and `longitude` coordinate names. The forcing workflow
            expects a 2D raster with `y` and `x` coordinates on EPSG:4326.

        Args:
            *args: Additional positional arguments passed to the base reader.
            **kwargs: Additional keyword arguments passed to the base reader.

        Returns:
            Normalized geopotential raster with `y` and `x` coordinates.
        """
        geopotential: xr.DataArray = super().read(*args, **kwargs)

        geopotential = geopotential.rename(
            {
                "latitude": "y",
                "longitude": "x",
            }
        )

        geopotential = geopotential.isel(time=0, drop=True)
        geopotential = geopotential.rio.write_crs(4326)

        # the geopotential has coordinates 0 -> 360,
        # but we want -180 -> 180, so we need to split and re-join the data array
        west_da: xr.DataArray = geopotential.sel(x=slice(180, 360))
        west_da = west_da.assign_coords(x=west_da.x.values - 360)
        east_da: xr.DataArray = geopotential.sel(x=slice(0, 180))

        # Combine the two parts
        geopotential: xr.DataArray = xr.concat([west_da, east_da], dim="x")

        return geopotential
