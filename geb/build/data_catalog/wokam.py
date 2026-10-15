"""Download the World Karst Aquifer Map (WOKAM)."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import geopandas as gpd

from geb.workflows.io import fetch_and_save, write_geom

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
                karst_polygons: gpd.GeoDataFrame = gpd.read_file(
                    f"zip://{archive}!WHYMAP_WOKAM/shp/whymap_karst__v1_poly.shp"
                )
                # Polygon intersections require valid geometries to compute overlap areas.
                karst_polygons.geometry = karst_polygons.geometry.make_valid()
                write_geom(karst_polygons, self.path)
        return self

    def read(self, **kwargs: Any) -> gpd.GeoDataFrame:
        """Read WOKAM with rock-class coverage fractions and source metadata.

        Args:
            **kwargs: Additional keyword arguments for the GeoParquet reader.

        Returns:
            Karst polygons with a dimensionless ``karst_fraction`` column (0–1)
            and source attribution in the frame attributes. Continuous rock
            classes use 0.825, discontinuous classes 0.4 and mixed rocks 0.65.

        Raises:
            ValueError: If WOKAM lacks a CRS or contains unknown rock classes.
        """
        karst_polygons: gpd.GeoDataFrame = super().read(**kwargs)
        rock_fractions: dict[int, float] = {1: 0.825, 2: 0.4, 3: 0.825, 4: 0.65, 5: 0.4}
        if karst_polygons.crs is None:
            raise ValueError("WOKAM polygons must have a CRS.")
        if not karst_polygons["rock_type"].isin(rock_fractions).all():
            raise ValueError("WOKAM contains unknown rock classes.")
        karst_polygons["karst_fraction"] = karst_polygons["rock_type"].map(
            rock_fractions
        )
        karst_polygons.attrs["source"] = "WHYMAP WOKAM, BGR, IAH, KIT, UNESCO, 2017"
        return karst_polygons
