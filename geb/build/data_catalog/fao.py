"""File adapter for FAO datasets."""

from __future__ import annotations

import zipfile
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from geb.workflows.io import fetch_and_save

from ..workflows.conversions import (
    M49_to_ISO3,
)
from .base import Adapter


class GMIA(Adapter):
    """Adapter for Global Map of Irrigation Areas datasets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Global Map of Irrigation Areas adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> GMIA:
        """Fetch the dataset from the given URL if not already present.

        Args:
            url: The URL to fetch the dataset from.

        Returns:
            The GMIA adapter instance.
        """
        if not self.is_ready:
            download_path = self.path.with_suffix(".zip")
            fetch_and_save(url=url, file_path=download_path)

            # unzip the file
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                files = zip_ref.namelist()
                assert len(files) == 1, "Expected exactly one file in the zip."
                file = files[0]
                zip_ref.extract(file, self.root)

            extracted_file_path = self.root / file
            extracted_file_path.rename(self.path)  # rename to desired filename
            download_path.unlink()  # Remove the downloaded zip file
        return self

    def read(self, **kwargs: Any) -> xr.DataArray:
        """Read the dataset into an xarray DataArray.

        Also sets the appropriate CRS.

        Args:
            **kwargs: Additional keyword arguments to pass to the reader function.

        Returns:
            The dataset as an xarray DataArray with appropriate CRS.
        """
        da = super().read(**kwargs)
        da = da.sel(band=1)
        da = da.rio.write_crs("EPSG:4326")
        return da


class FAOSTAT(Adapter):
    """Adapter for FAOSTAT datasets."""

    def fetch(self, url: str) -> FAOSTAT:
        """Fetch the dataset from the given URL if not already present.

        Args:
            url: The URL to fetch the dataset from.

        Returns:
            The FAOSTAT adapter instance.

        Raises:
            ValueError: If the expected files are not found in the zip.
        """
        if not self.is_ready:
            download_path = self.path.with_suffix(".zip")
            fetch_and_save(url=url, file_path=download_path, show_progress=True)

            # unzip the file
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                file = "Prices_E_All_Data.csv"
                try:
                    zip_ref.extract(file, self.root)
                except KeyError:
                    # sometimes the file has a different name (e.g. including Normalized)
                    # try to find a csv file
                    files = [f for f in zip_ref.namelist() if f.endswith(".csv")]
                    if len(files) == 1:
                        file = files[0]
                        zip_ref.extract(file, self.root)
                    else:
                        raise ValueError(
                            f"Could not find {file} in zip, and ambiguous csvs found: {files}"
                        )

            extracted_file_path = self.root / file

            # Read and filter the CSV
            df = pd.read_csv(extracted_file_path, encoding="ISO-8859-1")

            # Filter for Producer Price (USD/tonne)
            df = df[df["Element"] == "Producer Price (USD/tonne)"]

            # Save as parquet
            df.to_parquet(self.path, index=False)

            # Cleanup
            extracted_file_path.unlink()
            download_path.unlink()

        return self

    def read(self) -> pd.DataFrame:
        """Read the dataset.

        Returns:
            The dataset as a pandas DataFrame.
        """
        df = pd.read_parquet(self.path)
        year_columns = [
            col for col in df.columns if col.startswith("Y") and col[1:].isdigit()
        ]
        df = df[["Area Code (M49)", "Item", *year_columns]]

        df = df.rename(columns={col: int(col[1:]) for col in year_columns})
        year_columns = [int(col[1:]) for col in year_columns]

        df["Area Code (M49)"] = df["Area Code (M49)"].apply(
            lambda x: int(x[1:]) if x.startswith("'") else int(x)
        )
        df = df[~df["Area Code (M49)"].isin([58, 200, 230, 891, 736])]
        df["ISO3"] = df["Area Code (M49)"].map(M49_to_ISO3)
        assert df["ISO3"].isnull().sum() == 0, (
            "Some M49 codes could not be mapped to ISO3."
        )
        df = df.drop(columns=["Area Code (M49)"])

        for year in year_columns:
            df[year] = df[year] / 1000.0  # Convert from USD/tonne to USD/kg
            df[year] = df[year].astype(np.float64)

        df = df.rename(
            columns={
                "Item": "crop",
            }
        )

        df["crop"] = df["crop"].str.lower()

        return df
