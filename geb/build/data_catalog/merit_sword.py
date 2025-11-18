"""Data adapter for HydroLAKES data."""

import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import xarray as xr

from geb.workflows.io import fetch_and_save

from .base import Adapter


class MeritSword(Adapter):
    """The MERIT-SWORD adapter for downloading and processing HydroLAKES data.

    Args:
        Adapter: The base Adapter class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the MERIT-SWORD adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> Path:
        """Process MERIT-SWORD zip file to extract and convert to parquet.

        Args:
            url: The URL to download the HydroLAKES zip file from.

        Returns:
            Path to the processed parquet file.
        """
        if not self.is_ready:
            download_path: Path = self.root / url.split(sep="/")[-1]
            fetch_and_save(url=url, file_path=download_path)

            uncompressed_file: Path = download_path.with_suffix(suffix="")
            with zipfile.ZipFile(file=download_path, mode="r") as zip_ref:
                zip_ref.extractall(uncompressed_file)

            download_path.unlink()  # remove zip file

            template: str = "mb_to_sword_pfaf_{SWORD_Region}_translate.nc"

            SWORD_files = []
            for SWORD_region in (
                list(range(11, 19))
                + list(range(21, 30))
                + list(range(31, 37))
                + list(range(41, 50))
                + list(range(51, 58))
                + list(range(61, 68))
                + list(range(71, 79))
                + list(range(81, 87))
                + [91]
            ):
                SWORD_files.append(
                    uncompressed_file
                    / "ms_translate"
                    / "mb_to_sword"
                    / template.format(SWORD_Region=str(SWORD_region))
                )

            assert len(SWORD_files) == 61, "There are 61 SWORD regions"
            MERIT_Basins_to_SWORD: xr.Dataset = xr.open_mfdataset(
                paths=SWORD_files
            ).load()
            MERIT_Basins_to_SWORD.to_zarr(self.path)

            time.sleep(5)  # wait a bit to ensure all file handles are closed

            shutil.rmtree(
                path=uncompressed_file, ignore_errors=True
            )  # remove uncompressed folder

        return self

    def read(self) -> xr.Dataset:
        """Read the processed MERIT-SWORD data.

        Returns:
            The MERIT-SWORD data as an xarray Dataset.
        """
        return xr.open_zarr(store=self.path)
