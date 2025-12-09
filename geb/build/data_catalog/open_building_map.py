"""Utilities to download OpenBuildingMap tiles for a given bounding box.

This module provides a downloader that downloads and extracts the needed
gpkg tiles from the remote bz2 packages hosted by OpenBuildingMap.
Tiles are quadkey level 6 and downloaded individually.

Notes:
    - OpenBuildingMap distributes quadkey level 6 tiles as individual bz2 (zipped) files.
      Tile filenames follow the pattern " building.002232.gpkg.bz2" where the integers
      indicate the quadkey of the tile. See the OpenBuildingMap documentation for details.
    - Coverage spans the global land areas.

"""

import bz2
import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import mercantile
import pandas as pd
from pyquadkey2 import quadkey
from shapely import geometry
from tqdm import tqdm

from geb.workflows.io import fetch_and_save, write_geom

from .base import Adapter


class OpenBuildingMap(Adapter):
    """Dataset adapter for OpenBuildingMap data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter for OpenBuildingMap.

        Args:
            *args: Additional positional arguments passed to the base Adapter class.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)

    def _quadkeys_for_box(self, bounds: tuple, zoom: int = 6) -> None:
        """Gets the open building dataset. First it finds the quadkeys of tile within the model domain. Then it downloads the data and clips it to gdl region included in the domain.

        Args:
            bounds: Bounds of the geom for which to get quadkeys that intersect.
            zoom: Zoom level of the quadkeys. Zoomlevel 6 is used for open building map.
        """
        quadkeys = []

        # iterate over tiles intersecting the bbox
        for tile in mercantile.tiles(*bounds, zoom):
            qk = quadkey.from_tile((tile.x, tile.y), level=zoom)
            quadkeys.append(qk.key)

        return quadkeys

    def _extract_buildings_in_geom(
        self, gpkg_filename: Path, geom: geometry.polygon.Polygon
    ) -> gpd.GeoDataFrame:
        """This function reads the downloaded geopackage containing the buildings. It the extracts only the buildings that lie within the geom.

        Args:
            gpkg_filename: filename of the dowloaded geopackage.
            geom: geom representing the model region.
        Returns:
            A geopandas geodataframe containing all building within the geom.
        """
        # load buildings (should already be masked by region in this step)
        buildings = gpd.read_file(
            gpkg_filename,
            engine="pyogrio",
            mask=geom,
            columns=["id", "occupancy", "floorspace", "height", "geometry"],
        )
        # only keep buildings that intersect with the geom (to be sure, maybe can be removed)
        buildings = buildings[buildings.intersects(geom)]
        if len(buildings) == 0:
            print("No buildings found in region geom")
            return
        else:
            return buildings

    def _download_and_extract_tile(
        self,
        tile_url: str,
        temp_dir: Path,
        tile_filename: str,
    ) -> Path:
        """Download a tile .bz2 and extract only GeoTIFF files that intersect with the bbox.

        Args:
            tile_url: URL of the tile ZIP file.
            temp_dir: Temporary directory to extract to.
            tile_filename: Filename of the tile ZIP.

        Returns:
            Path to the dowloaded geopackage files.

        Raises:
            RuntimeError: If download or extraction fails after all retries.
        """
        # Tile is available, so download with retry
        zip_path: Path = temp_dir / tile_filename
        success: bool = fetch_and_save(
            tile_url,
            zip_path,
            delay_seconds=1,
            verbose=False,
            show_progress=True,
            double_delay=True,
            max_retries=17,  # will be total of ~day
        )
        if not success:
            raise RuntimeError(f"Failed to download {tile_url}")

        # Extract building.gpkg
        gpkg_filename = str(zip_path).replace(".bz2", "")
        gpkg_filename = gpkg_filename.replace("building.", "building_")
        with bz2.open(zip_path, "rb") as f_in, open(gpkg_filename, "wb") as f_out:
            f_out.write(f_in.read())
        return Path(gpkg_filename)

    def fetch(
        self, url: str, geom: geometry.polygon.Polygon, prefix: str
    ) -> "OpenBuildingMap":
        """Download OpenBuildingMap tiles intersecting a bbox.

        Args:
            url: Base URL of the OpenStreetMap server.
            geom: Polygon of the model region.
            prefix: Prefix for the local storage path.

        Returns:
            The OpenBuildingMap instance.

        Raises:
            RuntimeError: If no buildings can be downloaded for the model region.
        """
        if self.path.exists():
            return self

        # get bounds for geom
        bounds = geom.bounds
        tiles: list = self._quadkeys_for_box(bounds)
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir: Path = Path(temp_dir_str)
            list_of_buildings_in_geom: list[gpd.GeoDataFrame] = []

            for tile in tqdm(tiles, desc="Downloading OpenBuildingMap tiles"):
                tile_filename = f"building.{tile}.gpkg.bz2"
                tile_url: str = f"{url}/{tile_filename}"

                gpkg_filename: Path = self._download_and_extract_tile(
                    tile_url, temp_dir, tile_filename
                )
                # Add all extracted geopackage files (already filtered during extraction)
                buildings = self._extract_buildings_in_geom(gpkg_filename, geom)
                if buildings is not None:
                    list_of_buildings_in_geom.append(buildings)
        # concatenate all buildings
        buildings_in_geom: gpd.GeoDataFrame = pd.concat(
            list_of_buildings_in_geom, ignore_index=True
        )  # ty: ignore[invalid-assignment]

        # raise error if no buildings are found in model region
        if len(list_of_buildings_in_geom) == 0:
            raise RuntimeError("No OpenBuildingMap features were found in model domain")
        # write to file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_geom(buildings_in_geom, self.path)

        return self

    def read(self, **kwargs: Any) -> gpd.GeoDataFrame:
        """Read the OpenBuildingMap data as a GeoDataFrame.

        Args:
            **kwargs: Additional keyword arguments to pass to gpd.read_parquet.

        Returns:
            A GeoDataFrame with the GADM data.
        """
        gdf = Adapter.read(self, **kwargs)
        assert isinstance(gdf, gpd.GeoDataFrame)
        return gdf
