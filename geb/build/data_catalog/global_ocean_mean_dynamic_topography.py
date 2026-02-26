"""Utilities for working with Global Ocean Mean Dynamic Topography data in GEB."""

from __future__ import annotations

from typing import Any

import copernicusmarine
import numpy as np
import xarray as xr

from .base import Adapter


def _extrapolate_into_nodata(
    data: xr.DataArray,
    pad_cells: int,
) -> xr.DataArray:
    """Extrapolate values into adjacent no-data cells without changing extent.

    Notes:
        Each iteration fills NaN cells that have two valid neighbors in a straight
        line (left, right, up, down). Extrapolation uses a linear step based on
        the two adjacent values.

    Args:
        data: 2D raster data to fill.
        pad_cells: Number of cells to extrapolate outward (cells).

    Returns:
        Raster with NaN cells filled near valid data.

    Raises:
        ValueError: If `data` is not 2D or `pad_cells` is negative.
    """
    if data.ndim != 2:
        raise ValueError("data must be 2D to apply extrapolation.")
    if pad_cells < 0:
        raise ValueError("pad_cells must be non-negative.")

    if pad_cells == 0:
        return data

    values: np.ndarray = data.to_numpy().copy()

    for _ in range(pad_cells):
        nan_mask: np.ndarray = np.isnan(values)
        if not nan_mask.any():
            break

        sum_values: np.ndarray = np.zeros_like(values, dtype=np.float64)
        count_values: np.ndarray = np.zeros_like(values, dtype=np.int32)

        right1: np.ndarray = np.full_like(values, np.nan)
        right2: np.ndarray = np.full_like(values, np.nan)
        right1[:, :-1] = values[:, 1:]
        right2[:, :-2] = values[:, 2:]
        right_mask: np.ndarray = nan_mask & ~np.isnan(right1) & ~np.isnan(right2)
        right_extrap: np.ndarray = right1 + (right1 - right2)
        sum_values[right_mask] += right_extrap[right_mask]
        count_values[right_mask] += 1

        left1: np.ndarray = np.full_like(values, np.nan)
        left2: np.ndarray = np.full_like(values, np.nan)
        left1[:, 1:] = values[:, :-1]
        left2[:, 2:] = values[:, :-2]
        left_mask: np.ndarray = nan_mask & ~np.isnan(left1) & ~np.isnan(left2)
        left_extrap: np.ndarray = left1 + (left1 - left2)
        sum_values[left_mask] += left_extrap[left_mask]
        count_values[left_mask] += 1

        up1: np.ndarray = np.full_like(values, np.nan)
        up2: np.ndarray = np.full_like(values, np.nan)
        up1[1:, :] = values[:-1, :]
        up2[2:, :] = values[:-2, :]
        up_mask: np.ndarray = nan_mask & ~np.isnan(up1) & ~np.isnan(up2)
        up_extrap: np.ndarray = up1 + (up1 - up2)
        sum_values[up_mask] += up_extrap[up_mask]
        count_values[up_mask] += 1

        down1: np.ndarray = np.full_like(values, np.nan)
        down2: np.ndarray = np.full_like(values, np.nan)
        down1[:-1, :] = values[1:, :]
        down2[:-2, :] = values[2:, :]
        down_mask: np.ndarray = nan_mask & ~np.isnan(down1) & ~np.isnan(down2)
        down_extrap: np.ndarray = down1 + (down1 - down2)
        sum_values[down_mask] += down_extrap[down_mask]
        count_values[down_mask] += 1

        fill_mask: np.ndarray = nan_mask & (count_values > 0)
        if not fill_mask.any():
            break

        values[fill_mask] = sum_values[fill_mask] / count_values[fill_mask]

    filled: xr.DataArray = xr.DataArray(
        values,
        coords=data.coords,
        dims=data.dims,
        name=data.name,
        attrs=data.attrs,
    )

    if data.rio.crs is not None:
        filled = filled.rio.write_crs(data.rio.crs)

    return filled


class GlobalOceanMeanDynamicTopography(Adapter):
    """Downloader for Global Ocean Mean Dynamic Topography netcdf."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter for Global Ocean Mean Dynamic Topography.

        Args:
            *args: Additional positional arguments passed to the base Adapter class.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str | None = None) -> GlobalOceanMeanDynamicTopography:
        """Fetch the Global Ocean Mean Dynamic Topography data.

        Because login is required to download the data, the user enter
        their login credentials in the terminal to download the data.
        Args:
            url: The URL to download the data from. This parameter is not used
                 because the data is downloaded using the copernicusmarine library,
                 which handles the URL internally. It is included in the method signature for consistency with the Adapter interface.
        Returns:
            GlobalOceanMeanDynamicTopography: The adapter instance with the data fetched and saved to disk.
        """
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
        """Read the Global Ocean Mean Dynamic Topography data.

        This method reads the netCDF file containing the global ocean mean dynamic
        topography data, processes it by clipping to the specified model bounds and
        extrapolating values into no-data regions, and returns it as an xarray DataArray.
        Args:
            model_bounds: A tuple containing the minimum longitude, minimum latitude,
                          maximum longitude, and maximum latitude (min_lon, min_lat, max_lon, max_lat) that define the bounding box to which the data should be clipped.
        Returns:
            xr.DataArray: The processed global ocean mean dynamic topography data, clipped to the specified model bounds and with values extrapolated into no-data regions.
        Raises:
            ValueError: If the expected 'mdt' variable is not found in the dataset.
        """
        # read the global_ocean_mdt data
        global_ocean_mdt_dataset = xr.open_dataset(self.path)
        if "mdt" not in global_ocean_mdt_dataset:
            raise ValueError("Expected 'mdt' variable in global ocean MDT dataset.")
        global_ocean_mdt = global_ocean_mdt_dataset["mdt"]
        global_ocean_mdt = global_ocean_mdt.squeeze(drop=True)

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

        # extrapolate 3 cells into nodata regions to reach the coastline
        global_ocean_mdt = _extrapolate_into_nodata(global_ocean_mdt, pad_cells=5)

        # write crs
        global_ocean_mdt = global_ocean_mdt.rio.write_crs("EPSG:4326")
        # drop unused columns
        # set datatype to float32 and set fillvalue to np.nan
        global_ocean_mdt = global_ocean_mdt.astype(np.float32)
        global_ocean_mdt.encoding["_FillValue"] = np.nan
        global_ocean_mdt.attrs["_FillValue"] = np.nan

        return global_ocean_mdt
