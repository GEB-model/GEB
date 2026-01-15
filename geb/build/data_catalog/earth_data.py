"""Adapter for Earth Data datasets."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any

import rioxarray
import xarray as xr

from .base import Adapter


class GlobalSoilRegolithSediment(Adapter):
    """Adapter for Global Soil Regolith Sediment data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the GlobalSoilRegolithSediment adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> GlobalSoilRegolithSediment:
        """Fetch and process the Global Soil Regolith Sediment data.

        Because login is required to download the data, the user must manually
        download the zip file from the provided URL and place it in the expected
        location.

        Args:
            url: The URL to download the shapefiles from.

        Returns:
            GlobalSoilRegolithSediment: The adapter instance with the processed data.

        Raises:
            FileNotFoundError: If the expected files are not found after extraction.
        """
        if not self.is_ready:
            zip_filename = "global_soil_regolith_sediment_1304.zip"
            download_path = self.root / zip_filename

            while not download_path.exists():
                print(
                    "\033[91mThis file requires manual download due to licensing/login restrictions.\n"
                    f"Please download the data from: {url}\n"
                    f"Save the file as '{zip_filename}' and place it at: {download_path}\033[0m"
                )
                input(
                    "\033[91mPress Enter after placing the file to continue...\033[0m"
                )

            print(f"Extracting {zip_filename}...")
            with zipfile.ZipFile(file=download_path, mode="r") as zip_ref:
                zip_ref.extractall(self.root)

            target_files = {
                "average_soil_and_sedimentary-deposit_thickness.tif": "average_soil_and_sedimentary_deposit_thickness",
                "hill-slope_valley-bottom.tif": "hill_slope_valley_bottom",
                "land_cover_mask.tif": "land_cover_mask",
                "upland_hill-slope_regolith_thickness.tif": "upland_hill_slope_regolith_thickness",
                "upland_hill-slope_soil_thickness.tif": "upland_hill_slope_soil_thickness",
                "upland_valley-bottom_and_lowland_sedimentary_deposit_thickness.tif": "upland_valley_bottom_and_lowland_sedimentary_deposit_thickness",
            }

            # Locate extracted files
            found_files: dict[str, Path] = {}
            for path in self.root.rglob("*.tif"):
                if path.name in target_files:
                    found_files[path.name] = path

            # Check if all files were found
            missing = set(target_files.keys()) - set(found_files.keys())
            if missing:
                # Cleanup and raise error
                download_path.unlink()
                raise FileNotFoundError(
                    f"Could not find the following files in the zip archive: {missing}"
                )

            print("Processing TIF files...")
            data_arrays = []

            for fname, var_name in target_files.items():
                fpath = found_files[fname]
                da = rioxarray.open_rasterio(fpath)
                assert isinstance(da, xr.DataArray)
                da = da.sel(band=1, drop=True)
                da.name = var_name
                # Ensure correct type for storage if needed, but float32 is common for TIFs
                data_arrays.append(da)

            ds = xr.merge(data_arrays)

            print("Saving to Zarr...")
            # Using basic to_zarr as write_zarr only supports DataArray and specific use cases
            ds.to_zarr(self.path, mode="w", consolidated=False)

            print("Cleaning up...")
            download_path.unlink()  # remove zip file

            # Attempt to clean up extracted folder if it exists as expected
            extracted_folder = self.root / "Global_Soil_Regolith_Sediment_1304"
            if extracted_folder.exists() and extracted_folder.is_dir():
                shutil.rmtree(extracted_folder)
            else:
                # Fallback: delete the found files individually
                for fpath in found_files.values():
                    fpath.unlink()

        return self

    def read(self, **kwargs: Any) -> xr.Dataset:
        """Read the processed data from storage.

        Args:
            **kwargs: Additional keyword arguments to pass to the reader function.

        Returns:
            The processed data as an xarray Dataset.

        Raises:
            FileNotFoundError: If the data file does not exist.
        """
        # Override read because base implementation and read_zarr only support DataArray
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found at {self.path}")

        return xr.open_dataset(
            self.path, engine="zarr", chunks={}, consolidated=False, **kwargs
        )
