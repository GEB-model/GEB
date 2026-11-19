"""Adapter for the ESA CCI Land Cover plant functional type (PFT) maps."""

from typing import Any

import xarray as xr

from geb.workflows.io import fetch_and_save, get_window

from .base import Adapter


class ESACCILandCoverPFT(Adapter):
    """Adapter for ESA CCI Land Cover PFT fractions (300 m, annual, percent cover).

    See Harper et al. (2023), https://doi.org/10.5194/essd-15-1465-2023.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> ESACCILandCoverPFT:
        """Download the global PFT map (about 1.8 GB) if it is not present yet.

        Args:
            url: The URL of the yearly PFT NetCDF file.

        Returns:
            The adapter instance.
        """
        if not self.is_ready:
            fetch_and_save(url=url, file_path=self.path, logger=self.logger)
        return self

    def read(
        self,
        variables: list[str],
        bounds: tuple[float, float, float, float],
        buffer: int = 10,
    ) -> xr.Dataset:
        """Read PFT cover fractions for a region.

        Args:
            variables: PFT variable names, e.g. ["TREES-NE", "TREES-BD"].
            bounds: Region bounds (xmin, ymin, xmax, ymax) in degrees.
            buffer: Number of extra cells around the bounds.

        Returns:
            Dataset with the requested variables as percent cover (0-100, float32)
            on dimensions y and x, with CRS EPSG:4326.
        """
        ds = xr.open_dataset(self.path, mask_and_scale=False)[variables]
        ds = ds.isel(time=0, drop=True).rename({"lon": "x", "lat": "y"})
        ds = ds.isel(get_window(ds.x, ds.y, bounds, buffer=buffer)).load()
        ds = ds.astype("float32")
        ds = ds.rio.write_crs("EPSG:4326")
        return ds
