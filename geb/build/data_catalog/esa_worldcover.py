"""Adapter for ESA WorldCover datasets using STAC API."""

from __future__ import annotations

from typing import Any

import odc.stac
import planetary_computer
import pystac_client
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
            url: The STAC API URL to fetch the dataset from.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            The ESAWorldCover adapter instance.
        """
        self.catalog = pystac_client.Client.open(
            url,
            modifier=planetary_computer.sign_inplace,
        )

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
        search_result = self.catalog.search(
            collections=["esa-worldcover"],
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

        # Load the data using ODC STAC into an xarray Dataset
        ds: xr.Dataset = odc.stac.load(
            item_list,
            chunks={"longitude": 3000, "latitude": 3000},
        )
        da: xr.DataArray = ds["map"]
        da = (
            da.isel(time=-1)
            .rename({"latitude": "y", "longitude": "x"})
            .sel(
                x=slice(xmin, xmax),
                y=slice(ymax, ymin),
            )
        )
        da.attrs["_FillValue"] = da.attrs["nodata"]
        del da.attrs["nodata"]
        return da
