"""Data adapter for SWORD data."""

import shutil
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from geb.workflows.io import fetch_and_save

from .base import Adapter

FILES: list[str] = []


class Sword(Adapter):
    """The SWORD adapter for downloading and processing SWORD data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the SWORD adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> Adapter:
        """Process SWORD zip file to extract and convert to parquet.

        Args:
            url: The URL to download the SWORD zip file from.

        Returns:
            The adapter instance with the processed parquet file.
        """
        if not self.is_ready:
            download_path: Path = self.root / url.split(sep="/")[-1]
            fetch_and_save(url=url, file_path=download_path)

            uncompressed_file: Path = download_path.with_suffix(suffix="")
            with zipfile.ZipFile(file=download_path, mode="r") as zip_ref:
                zip_ref.extractall(uncompressed_file)

            download_path.unlink()  # remove zip file

            gdfs: list[gpd.GeoDataFrame] = []
            downloaded_files: list[Path] = []
            for continent in ("sa", "na", "oc", "eu", "af", "as"):
                reaches: Path = (
                    self.root
                    / "SWORD_v16_gpkg"
                    / "gpkg"
                    / f"{continent}_sword_reaches_v16.gpkg"
                )

                gdfs.append(gpd.read_file(reaches))

            SWORD = pd.concat(gdfs, ignore_index=True)
            SWORD.to_file(self.path)

            # Connect to the GPKG
            conn = sqlite3.connect(self.path)
            cursor = conn.cursor()

            # Create an index on COMID
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_reach_id
                ON sword (reach_id);
            """)

            conn.commit()

            shutil.rmtree(path=uncompressed_file)  # remove uncompressed folder
            conn.close()

        return self
