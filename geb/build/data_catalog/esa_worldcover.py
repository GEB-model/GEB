"""Adapter for ESA WorldCover datasets using direct AWS S3 access."""

from __future__ import annotations

import math
from typing import Any

import rasterio
import rioxarray  # noqa: F401 – registers .rio accessor on xarray
import xarray as xr
from rioxarray.merge import merge_arrays

from .base import Adapter

_S3_BUCKET = "esa-worldcover"
_S3_REGION = "eu-central-1"  # AWS datacenter where the bucket is hosted; data covers the entire globe
_S3_TILE_PATH = "v200/2021/map"
_S3_TILE_TEMPLATE = "ESA_WorldCover_10m_2021_v200_{NS}{lat:02d}{EW}{lon:03d}_Map.tif"
_TILE_SIZE_DEG = 3


def _s3_tile_urls(xmin: float, ymin: float, xmax: float, ymax: float) -> list[str]:
    """Return S3 URLs for all 3°×3° tiles that overlap the bounding box.

    Args:
        xmin: Minimum x-coordinate (longitude).
        ymin: Minimum y-coordinate (latitude).
        xmax: Maximum x-coordinate (longitude).
        ymax: Maximum y-coordinate (latitude).

    Returns:
        List of S3 URLs for all tiles covering the bounding box.
    """
    lat_start = math.floor(ymin / _TILE_SIZE_DEG) * _TILE_SIZE_DEG
    lon_start = math.floor(xmin / _TILE_SIZE_DEG) * _TILE_SIZE_DEG
    urls: list[str] = []
    lat = lat_start
    while lat < ymax:
        lon = lon_start
        while lon < xmax:
            NS = "N" if lat >= 0 else "S"
            EW = "E" if lon >= 0 else "W"
            filename = _S3_TILE_TEMPLATE.format(
                NS=NS, lat=abs(lat), EW=EW, lon=abs(lon)
            )
            urls.append(f"s3://{_S3_BUCKET}/{_S3_TILE_PATH}/{filename}")
            lon += _TILE_SIZE_DEG
        lat += _TILE_SIZE_DEG
    return urls


class ESAWorldCover(Adapter):
    """Adapter for ESA WorldCover datasets using direct AWS S3 access."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the ESA WorldCover adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, *args: Any, **kwargs: Any) -> ESAWorldCover:
        """No pre-fetching required; tiles are read directly from AWS S3 in read().

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            The ESAWorldCover adapter instance.
        """
        return self

    def read(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> xr.DataArray:
        """Read ESA WorldCover data for the specified bounding box from AWS S3.

        Args:
            xmin: Minimum x-coordinate (longitude).
            ymin: Minimum y-coordinate (latitude).
            xmax: Maximum x-coordinate (longitude).
            ymax: Maximum y-coordinate (latitude).

        Returns:
            xarray.DataArray: The ESA WorldCover data for the specified bounding box.
        """
        tile_urls = _s3_tile_urls(xmin, ymin, xmax, ymax)

        # Scope GDAL config locally so it does not affect other S3 operations
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", AWS_DEFAULT_REGION=_S3_REGION):
            arrays: list[xr.DataArray] = []
            for url in tile_urls:
                # Convert s3:// URL to GDAL /vsis3/ path
                vsi_path = "/vsis3/" + url[len("s3://") :]
                da = rioxarray.open_rasterio(vsi_path, chunks={"x": 3000, "y": 3000})
                assert isinstance(da, xr.DataArray), (
                    f"Expected DataArray, got {type(da)}"
                )
                arrays.append(da.squeeze("band", drop=True))

            assert len(arrays) > 0, (
                f"No ESA WorldCover S3 tiles found for bbox ({xmin}, {ymin}, {xmax}, {ymax})."
            )

            merged = merge_arrays(arrays) if len(arrays) > 1 else arrays[0]
            nodata = merged.attrs.get("_FillValue")
            merged = merged.sel(
                x=slice(xmin, xmax),
                y=slice(ymax, ymin),
            )
            # Force all S3 reads within the scoped env before the context exits
            merged = merged.compute()

        merged.attrs["_FillValue"] = nodata
        return merged
