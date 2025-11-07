"""Adapter for Global Data Lab shapefiles."""

from __future__ import annotations

import shutil
import zipfile
from typing import Any

import geopandas as gpd

from .base import Adapter


class GlobalDataLabShapefile(Adapter):
    """Adapter for Global Data Lab shapefiles."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the GlobalDataLabShapefile adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> GlobalDataLabShapefile:
        """Fetch and process the Global Data Lab shapefiles.

        Because login is required to download the data, the user must manually
        download the zip file from the provided URL and place it in the expected
        location.

        Args:
            url: The URL to download the shapefiles from.

        Returns:
            GlobalDataLabShapefile: The adapter instance with the processed data.
        """
        if not self.is_ready:
            download_path = self.root / "GDL Shapefiles V4.zip"

            while not download_path.exists():
                print(
                    "\033[91mThis file requires manual download due to licensing restrictions. Please download GDL Shapefiles V4 from: "
                    + url
                    + f" and place it at: {download_path}\033[0m"
                )
                input(
                    "\033[91mPress Enter after placing the file to continue...\033[0m"
                )

            with zipfile.ZipFile(file=download_path, mode="r") as zip_ref:
                zip_ref.extractall(self.root)

            gdf: gpd.GeoDataFrame = gpd.read_file(
                self.root / "GDL Shapefiles V4" / "GDL Shapefiles V4.shp"
            )

            print("Saving processed data...")
            gdf.to_parquet(
                self.path,
                engine="pyarrow",
                compression="gzip",
                compression_level=9,
                write_covering_bbox=True,
            )

            download_path.unlink()  # remove zip file
            shutil.rmtree(
                path=self.root / "GDL Shapefiles V4"
            )  # remove uncompressed folder

        return self

    def read(self, **kwargs: Any) -> gpd.GeoDataFrame:
        """Read the processed data from storage.

        Args:
            **kwargs: Additional keyword arguments to pass to the reader function.
        Returns:
            The processed data as a GeoDataFrame.
        """
        GDL_regions = super().read(**kwargs)
        GDL_regions = GDL_regions[
            GDL_regions["GDLcode"] != "NA"
        ]  # remove regions without GDL code
        return GDL_regions
