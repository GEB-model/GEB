"""GADM data adapter for downloading and processing GADM data."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd

from geb.workflows.io import fetch_and_save

from .base import Adapter


class GADM(Adapter):
    """The GADM adapter for downloading and processing GADM data.

    Args:
        Adapter: The base Adapter class.
    """

    def __init__(self, level: int, *args: Any, **kwargs: Any) -> None:
        """Initialize the GADM adapter.

        Args:
            level: The administrative level to download (0 for countries, 1 for first-level subdivisions, etc.).
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        self.level = level
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> GADM:
        """Process GADM Level 1 zip file to extract and convert to parquet.

        Args:
            url: The URL to download the GADM zip file from.

        Returns:
            The instance of the Adapter after processing.
        """
        if not self.is_ready:
            download_path: Path = self.root / url.split(sep="/")[-1]
            fetch_and_save(url=url, file_path=download_path)

            uncompressed_file: Path = download_path.with_suffix(suffix="")
            with zipfile.ZipFile(file=download_path, mode="r") as zip_ref:
                zip_ref.extractall(uncompressed_file)
            download_path.unlink()  # remove zip file
            gdf: gpd.GeoDataFrame = gpd.read_file(
                filename=uncompressed_file / "gadm_410-levels.gpkg",
                layer=f"ADM_{self.level}",
            )
            gdf.to_parquet(
                self.path,
                engine="pyarrow",
                compression="gzip",
                compression_level=9,
                write_covering_bbox=True,
            )
            shutil.rmtree(path=uncompressed_file)  # remove uncompressed folder
        return self

    def read(self, **kwargs: Any) -> gpd.GeoDataFrame:
        """Read the GADM data as a GeoDataFrame.

        Args:
            **kwargs: Additional keyword arguments to pass to gpd.read_parquet.

        Returns:
            A GeoDataFrame with the GADM data.
        """
        gdf = Adapter.read(self, **kwargs)
        assert isinstance(gdf, gpd.GeoDataFrame)
        gdf["GID_0"] = gdf["GID_0"].replace(
            {"XKO": "XKX"}
        )  # XKO is a deprecated code for Kosovo, XKX is the new code
        gdf: gpd.GeoDataFrame = gdf[
            gdf["GID_0"] != "NA"
        ]  # Remove entries with invalid ISO3 code
        return gdf
