"""Adapter for ESA WorldCover datasets using STAC API."""

from __future__ import annotations

from typing import Any

import numpy as np
import odc.stac
import pystac
import pystac_client
import rasterio
import s3fs
import xarray as xr

from .base import Adapter


class ESAWorldCover(Adapter):
    """Adapter for ESA WorldCover datasets using STAC API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Adapter for ESA WorldCover datasets using STAC API."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str, *args: Any, **kwargs: Any) -> ESAWorldCover:
        """Fetch the ESA WorldCover dataset from the specified STAC URL.

        Args:
            url: The STAC API URL for the ESA WorldCover collection.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            The ESAWorldCover adapter instance.
        """
        self.collection = pystac.Collection.from_file(url)
        root_catalog_url = self.collection.get_parent().get_self_href()
        self.client = pystac_client.Client.open(root_catalog_url)
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
        # Search for items within the specific collection and bounding box
        search_result = self.client.search(
            collections=[self.collection.id],
            bbox=(
                xmin,
                ymin,
                xmax,
                ymax,
            ),
        )

        # List all STAC items found in the search result
        item_list = list(search_result.items())
        assert len(item_list) > 0, "No items found for the specified bounding box."

        # The STAC items don't have resolution and crs information, so we need to open one asset to get it
        assets = item_list[0].assets
        assert len(assets) == 1, "Expected exactly one asset per item."
        tif_url = list(assets.values())[0].href

        with s3fs.S3FileSystem(anon=True).open(tif_url) as f:
            src = rasterio.open(f)
            crs = src.profile["crs"]
            resolution = abs(src.profile["transform"].a)
            dtype = src.profile["dtype"]
            nodata = src.profile["nodata"]
            nodata = getattr(np, dtype)(nodata)  # convert to correct type

        # Load the data using ODC STAC into an xarray Dataset
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
