"""Adapter for MIRCA-OS datasets."""

from __future__ import annotations

from typing import Any

import rioxarray  # noqa: F401

from .base import Adapter


class MIRCAOS(Adapter):
    """Adapter for MIRCA-OS datasets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the MIRCAOS adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        if kwargs["filename"].endswith("MIRCA-OS_Soybeans_2000_ir.tif"):
            kwargs["filename"] = kwargs["filename"].replace(
                "MIRCA-OS_Soybeans_2000_ir.tif", "MIRCA-OS_Soybeans2000_ir.tif"
            )
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> MIRCAOS:
        """Fetch the MIRCA-OS dataset.

        Because manual download is required, this method prompts the user to
        download and extract the RAR file into the expected location.

        Args:
            url: The URL to download the dataset from.

        Returns:
            MIRCAOS: The adapter instance.
        """
        if not self.is_ready:
            manual_folder = self.root / "Annual Harvested Area Grids"

            while not manual_folder.exists():
                print(
                    "\033[91mThis dataset requires manual download and extraction due to licensing restrictions. "
                    + f"Please download MIRCA-OS from: {url}\n"
                    + f"Then extract the contents and place the 'Annual Harvested Area Grids' folder at: {self.root}\033[0m"
                )
                input(
                    "\033[91mPress Enter after placing the folder to continue...\033[0m"
                )

        return self

    def read(self, **kwargs: Any) -> Any:
        """Read the processed data from storage.

        If the target file (e.g., .tif) does not exist, but a raw version (e.g., .asc)
        is available in the extracted folder, it is parsed on demand.

        Args:
            **kwargs: Additional keyword arguments to pass to the reader function.

        Returns:
            The processed data as an xarray DataArray.
        """
        da = super().read(**kwargs)
        da = da.sel(band=1)
        assert da.rio.crs is not None, "CRS information is missing in the dataset."

        return da
