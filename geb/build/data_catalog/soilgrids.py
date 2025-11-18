"""SoilGrids data catalog adapter and workflow."""

from __future__ import annotations

from typing import Any

import rioxarray as rxr
import xarray as xr

from .base import Adapter


class SoilGrids(Adapter):
    """SoilGrids 2020 data catalog adapter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the SoilGrids adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> SoilGrids:
        """Sets the SoilGrids dataset from the specified URL.

        Args:
            url: The URL template for the SoilGrids dataset, with placeholders for variable and depth.

        Returns:
            The SoilGrids adapter instance.
        """
        self.url = url
        return self

    def read(self, variable: str, depth: str, **kwargs: Any) -> xr.DataArray:
        """Read the SoilGrids data for the specified variable and depth.

        Args:
            variable: The soil variable to read (e.g., 'bdod', 'clay', 'silt', 'soc').
            depth: The soil depth layer to read (e.g., '0-5cm', '5-15cm', etc.).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            The (lazy) data array containing the SoilGrids data for the specified variable and depth.
        """
        return rxr.open_rasterio(self.url.format(variable=variable, depth=depth)).sel(
            band=1
        )
