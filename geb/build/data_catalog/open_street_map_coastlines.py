"""Utilities to download OSM coastlines.

This module provides a downloader that downloads and extracts the needed
coastline data from the remote OSM coastlines dataset. The data is downloaded
from osmdata.openstreetmap.de and stored to a local file in the GEB data catalog.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd

from geb.workflows.io import fetch_and_save

from .base import Adapter


class OpenStreetMapCoastlines(Adapter):
    """OSM Coastlines data catalog adapter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the OSM Coastlines adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> OpenStreetMapCoastlines:
        """Sets the URL for the OSM coastlines data source.

        Args:
            url: The URL for the OSM coastlines dataset.

        Returns:
            The OpenStreetMapCoastlines adapter instance.

        Raises:
            RuntimeError: If the data download fails.
        """
        self.url = url
        succes = fetch_and_save(self.url, self.path)
        if not succes:
            raise RuntimeError("Failed to download OSM coastlines data.")
        return self

    def read(self) -> Path:
        """Reads the OSM coastlines data.

        Returns:
            The path to the downloaded OSM coastlines data file.

        Raises:
            RuntimeError: If reading the data fails.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(self.path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)
            extracted_files = list(Path(tmpdir).rglob("*.shp"))
            if not extracted_files:
                raise RuntimeError(
                    "No shapefile found in the extracted OSM coastlines data."
                )
            shp_path = extracted_files[0]
            gdf = gpd.read_file(shp_path)
        return gdf
