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
        success = fetch_and_save(self.url, self.path)
        if not success:
            raise RuntimeError("Failed to download OSM coastlines data.")
        return self

    def read(self) -> gpd.GeoDataFrame:
        """Reads the OSM coastlines data.

        Returns:
            A GeoDataFrame containing the OSM coastlines geometries.

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


class OpenStreetMapLandPolygons(Adapter):
    """OSM Land Polygons data catalog adapter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the OSM Land Polygons adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> OpenStreetMapLandPolygons:
        """Sets the URL for the OSM land polygons data source.

        Args:
            url: The URL for the OSM land polygons dataset.

        Returns:
            The OpenStreetMapLandPolygons adapter instance.

        Raises:
            RuntimeError: If the data download fails.
        """
        self.url = url
        success = fetch_and_save(self.url, self.path)
        if not success:
            raise RuntimeError("Failed to download OSM land polygons data.")
        return self

    def read(self) -> gpd.GeoDataFrame:
        """Reads the OSM land polygons data.

        Returns:
            A GeoDataFrame containing the OSM land polygons geometries.

        Raises:
            RuntimeError: If reading the data fails.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(self.path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)
            extracted_files = list(Path(tmpdir).rglob("*.shp"))
            if not extracted_files:
                raise RuntimeError(
                    "No shapefile found in the extracted OSM land polygons data."
                )
            shp_path = extracted_files[0]
            gdf = gpd.read_file(shp_path)
        return gdf
