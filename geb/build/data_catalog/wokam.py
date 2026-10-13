"""Download the World Karst Aquifer Map (WOKAM)."""

from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd

from geb.workflows.io import fetch_and_save

from .base import Adapter


class WOKAM(Adapter):
    """Download and cache WOKAM polygons of rocks that can form karst."""

    def fetch(self, url: str) -> WOKAM:
        """Download WOKAM, fix invalid polygons and save them as GeoParquet.

        Args:
            url: URL of the official WOKAM ZIP archive.

        Returns:
            The adapter with the downloaded map ready to read.
        """
        if not self.is_ready:
            with TemporaryDirectory(dir=self.root) as download_folder:
                archive: Path = Path(download_folder) / "wokam.zip"
                fetch_and_save(url, archive, logger=self.logger)
                polygons: gpd.GeoDataFrame = gpd.read_file(
                    f"zip://{archive}!WHYMAP_WOKAM/shp/whymap_karst__v1_poly.shp"
                )
                # Fix invalid shapes so their area can be calculated during the build.
                polygons.geometry = polygons.geometry.make_valid()
                polygons.to_parquet(self.path, index=False)
        return self
