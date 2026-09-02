"""Utilities to download and setup FLOPROS."""

import shutil
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd

from geb.workflows.io import fetch_and_save

from .base import Adapter


class FLOPROS(Adapter):
    """The FLOPROS adapter for downloading and processing FLOPROS data.

    Args:
        Adapter: The base Adapter class.
    """

    def __init__(self, column: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the FLOPROS adapter.

        Args:
            column: The column name to extract from the FLOPROS shapefile. Choose from:
                "DL_Min_Riv": minimum value of river flood protection standard in the Design layer;
                "DL_Max_Riv": maximum value of river flood protection standard in the Design layer;
                "DL_Min_Co": minimum value of coastal flood protection standard in the Design layer;
                "DL_Max_Co": maximum value of coastal flood protection standard in the Design layer;
                "PL_Min_Riv": minimum value of river flood protection standard in the Policy layer;
                "PL_Max_Riv": maximum value of river flood protection standard in the Policy layer;
                "PL_Min_Co": minimum value of coastal flood protection standard in the Policy layer;
                "PL_Max_Co": maximum value of coastal flood protection standard in the Policy layer;
                "ModL_Riv": value of river flood protection standard in the Model layer;
                "MerL_Riv": value of river flood protection standard in the Merged layer.
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)
        self.column = column

    def fetch(self, url: str) -> FLOPROS:
        """Process FLOPROS data to extract and convert to parquet.

        Args:
            url: The URL to download the FLOPROS zip file from.

        Returns:
            The instance of the Adapter after processing.
        """
        if not self.is_ready:
            download_path: Path = self.root / url.split(sep="/")[-1]
            fetch_and_save(url=url, file_path=download_path, logger=self.logger)

            uncompressed_file: Path = download_path.with_suffix(suffix="")
            with zipfile.ZipFile(file=download_path, mode="r") as zip_ref:
                zip_ref.extractall(uncompressed_file)
            download_path.unlink()  # remove zip file
            gdf: gpd.GeoDataFrame = gpd.read_file(
                filename=uncompressed_file
                / "Scussolini_etal_Suppl_info"
                / "FLOPROS_shp_V1"
                / "FLOPROS_shp_V1.shp",
            )[[self.column, "geometry"]]  # only keep relevant columns
            gdf = gdf.rename(columns={self.column: "flood_protection_standard"})  # ty: ignore[invalid-assignment]
            gdf.to_parquet(
                self.path,
                engine="pyarrow",
                compression="gzip",
                compression_level=9,
                write_covering_bbox=True,
            )
            shutil.rmtree(path=uncompressed_file)  # remove uncompressed folder
        return self
