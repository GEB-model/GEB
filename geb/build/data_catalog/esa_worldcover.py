"""Adapter for ESA WorldCover datasets using STAC API, with direct S3 fallback."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import odc.stac
import pystac
import pystac_client
import rasterio
import rioxarray  # noqa: F401 – registers .rio accessor on xarray
import s3fs
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
    """Adapter for ESA WorldCover datasets using STAC API, with direct S3 fallback."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the ESA WorldCover adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)
        self._use_s3_fallback: bool = False

    def fetch(self, url: str, *args: Any, **kwargs: Any) -> ESAWorldCover:
        """Fetch the ESA WorldCover dataset from the specified STAC URL.

        Tries the Terrascope STAC endpoint first. If the connection fails (e.g.
        during a service outage), falls back to reading tiles directly from the
        public AWS S3 bucket ``esa-worldcover``.

        Args:
            url: The STAC API URL for the ESA WorldCover collection.

        Returns:
            The ESAWorldCover adapter instance.
        """
        try:
            self.collection = pystac.Collection.from_file(url)
            parent = self.collection.get_parent()
            assert parent is not None, "Collection has no parent catalog."
            root_catalog_url = parent.get_self_href()
            assert root_catalog_url, "Could not determine root catalog URL from collection."
            self.client = pystac_client.Client.open(root_catalog_url)
        except Exception as e:
            warnings.warn(
                f"Could not connect to Terrascope STAC ({e}). "
                "Falling back to direct S3 access for ESA WorldCover tiles.",
                stacklevel=2,
            )
            self._use_s3_fallback = True
        return self

    def read(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> xr.DataArray:
        """Read the ESA WorldCover data for the specified bounding box.

        Args:
            xmin: Minimum x-coordinate (longitude).
            ymin: Minimum y-coordinate (latitude).
            xmax: Maximum x-coordinate (longitude).
            ymax: Maximum y-coordinate (latitude).

        Returns:
            xarray.DataArray: The data array containing the ESA WorldCover data for the specified bounding box.
        """
        if self._use_s3_fallback:
            return self._read_from_s3(xmin, ymin, xmax, ymax)
        return self._read_from_stac(xmin, ymin, xmax, ymax)

    def _read_from_stac(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> xr.DataArray:
        """Read tiles via the Terrascope STAC API.

        Args:
            xmin: Minimum x-coordinate (longitude).
            ymin: Minimum y-coordinate (latitude).
            xmax: Maximum x-coordinate (longitude).
            ymax: Maximum y-coordinate (latitude).

        Returns:
            xarray.DataArray: The ESA WorldCover data for the specified bounding box.
        """
        search_result = self.client.search(
            collections=[self.collection.id],
            bbox=(xmin, ymin, xmax, ymax),
        )

        item_list = list(search_result.items())
        assert len(item_list) > 0, "No items found for the specified bounding box."

        assets = item_list[0].assets
        assert len(assets) == 1, "Expected exactly one asset per item."
        tif_url = list(assets.values())[0].href

        with s3fs.S3FileSystem(anon=True).open(tif_url) as f:
            src = rasterio.open(f)
            crs = src.profile["crs"]
            resolution = abs(src.profile["transform"].a)
            dtype = src.profile["dtype"]
            nodata = src.profile["nodata"]
            nodata = getattr(np, dtype)(nodata)

        ds = (
            odc.stac.load(
                item_list,
                crs=crs,
                chunks={"x": 3000, "y": 3000},
                resolution=resolution,
                dtype=dtype,
            )
            .isel(time=0)
            .rename({"latitude": "y", "longitude": "x"})
            .sel(
                x=slice(xmin, xmax),
                y=slice(ymax, ymin),
            )
        )
        da = ds["ESA_WORLDCOVER_10M_MAP"]
        da.attrs["_FillValue"] = nodata
        return da

    def _read_from_s3(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> xr.DataArray:
        """Read tiles directly from the public S3 bucket, bypassing STAC.

        Uses GDAL's /vsis3/ virtual filesystem so that rasterio manages the S3
        connections internally — avoiding the closed-file-handle issue that occurs
        when passing lazy dask arrays opened via s3fs context managers.

        Args:
            xmin: Minimum x-coordinate (longitude).
            ymin: Minimum y-coordinate (latitude).
            xmax: Maximum x-coordinate (longitude).
            ymax: Maximum y-coordinate (latitude).

        Returns:
            xarray.DataArray: The ESA WorldCover data for the specified bounding box.
        """
        import os

        # Tell GDAL to use anonymous access and the correct region
        os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
        os.environ.setdefault("AWS_DEFAULT_REGION", _S3_REGION)

        tile_urls = _s3_tile_urls(xmin, ymin, xmax, ymax)

        arrays: list[xr.DataArray] = []
        for url in tile_urls:
            # Convert s3:// URL to GDAL /vsis3/ path
            vsi_path = "/vsis3/" + url[len("s3://"):]
            da = rioxarray.open_rasterio(vsi_path, chunks={"x": 3000, "y": 3000})
            assert isinstance(da, xr.DataArray), f"Expected DataArray, got {type(da)}"
            arrays.append(da.squeeze("band", drop=True))

        assert len(arrays) > 0, (
            f"No ESA WorldCover S3 tiles found for bbox ({xmin}, {ymin}, {xmax}, {ymax})."
        )

        merged = merge_arrays(arrays) if len(arrays) > 1 else arrays[0]
        nodata = merged.attrs.get("_FillValue", merged.attrs.get("missing_value", 0))
        merged = merged.sel(
            x=slice(xmin, xmax),
            y=slice(ymax, ymin),
        )
        merged.attrs["_FillValue"] = nodata
        return merged
