"""Utilities for working with Global Ocean Mean Dynamic Topography data in GEB."""

from __future__ import annotations

from typing import Any

import copernicusmarine
import numpy as np
import xarray as xr

from .base import Adapter


class GlobalOceanMeanDynamicTopography(Adapter):
    """Downloader for Global Ocean Mean Dynamic Topography netcdf."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter for Global Ocean Mean Dynamic Topography.

        Args:
            *args: Additional positional arguments passed to the base Adapter class.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url=None):
        """Fetch the Global Ocean Mean Dynamic Topography data."""
        if not self.is_ready:
            copernicusmarine.subset(
                dataset_id="cnes_obs-sl_glo_phy-mdt_my_0.125deg_P20Y",
                dataset_part="mdt",
                variables=["mdt"],
                minimum_longitude=-179.9375,
                maximum_longitude=179.9375,
                minimum_latitude=-89.9375,
                maximum_latitude=89.9375,
                start_datetime="2003-01-01T00:00:00",
                end_datetime="2003-01-01T00:00:00",
                output_filename=self.filename,
                output_directory=self.path.parent,
            )
        return self

    def read(self, model_bounds: tuple[float, float, float, float]) -> xr.DataArray:
        """Read the Global Ocean Mean Dynamic Topography data."""

        # read the global_ocean_mdt data
        global_ocean_mdt = xr.open_rasterio(self.path)

        # reproject global_ocean_mdt to 0.008333 grid (~1km)
        global_ocean_mdt = global_ocean_mdt.rio.write_crs("EPSG:4326")
        min_lon, min_lat, max_lon, max_lat = model_bounds

        # clip to model bounds
        global_ocean_mdt = global_ocean_mdt.rio.clip_box(
            minx=min_lon,
            miny=min_lat,
            maxx=max_lon,
            maxy=max_lat,
        )

        # write crs
        global_ocean_mdt = global_ocean_mdt.rio.write_crs("EPSG:4326")
        # drop unused columns
        global_ocean_mdt = global_ocean_mdt.squeeze(drop=True)
        # set datatype to float32 and set fillvalue to np.nan
        global_ocean_mdt = global_ocean_mdt.astype(np.float32)
        global_ocean_mdt.encoding["_FillValue"] = np.nan
        global_ocean_mdt.attrs["_FillValue"] = np.nan

        return global_ocean_mdt
