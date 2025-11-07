"""WhyMap adapter."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd

from geb.workflows.io import fetch_and_save

from .base import Adapter


class WhyMap(Adapter):
    """Adapter for the WhyMap dataset.

    This datasets provides global information on aquifer properties.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the WhyMap adapter.

        Args:
            *args: Arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.

        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str, *args: Any, **kwargs: Any) -> WhyMap:
        """Fetch the WhyMap dataset.

        Args:
            url: URL to fetch the dataset from.
            *args: Additional arguments to pass to fetch_and_save.
            **kwargs: Additional keyword arguments to pass to fetch_and_save.

        Returns:
            The WhyMap adapter.
        """
        if not self.is_ready:
            download_path: Path = self.root / url.split("/")[-1]
            fetch_and_save(url, download_path)

            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(self.root)

            gdf: gpd.GeoDataFrame = gpd.read_file(
                filename=self.root
                / "WHYMAP_GWR"
                / "shp"
                / "whymap_GW_aquifers_v1_poly.shp"
            )
            gdf.to_parquet(self.path, index=False)

            shutil.rmtree(path=self.root / "WHYMAP_GWR")

            download_path.unlink()

        return self
