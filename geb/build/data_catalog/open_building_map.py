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

import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import mercantile
import pandas as pd
from pyquadkey2 import quadkey
from shapely import geometry
from tqdm import tqdm

from geb.workflows.io import fetch_and_save

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

    def _quadkeys_for_geom(
        self, geom: geometry.polygon.Polygon, zoom: int = 6
    ) -> list[str]:
        """Gets the quadkeys of tiles that intersect with the polygon geometry.

        Args:
            geom: Polygon geometry for which to get intersecting quadkeys.
            zoom: Zoom level of the quadkeys. Zoomlevel 6 is used for open building map.

        Returns:
            A list of quadkey strings intersecting the polygon.
        """
        quadkeys: list[str] = []

        west, south, east, north = geom.bounds

        # iterate over tiles intersecting the bbox
        for tile in mercantile.tiles(west, south, east, north, zooms=zoom):
            qk = quadkey.from_tile((tile.x, tile.y), level=zoom)

            # Create a polygon for the tile bounds
            tile_bounds = mercantile.bounds(tile)
            tile_polygon = geometry.box(
                tile_bounds.west, tile_bounds.south, tile_bounds.east, tile_bounds.north
            )

            # Only include tile if it intersects with the input geometry
            if tile_polygon.intersects(geom):
                quadkeys.append(qk.key)

        return quadkeys

    def _extract_buildings_in_geom(
        self, gpkg_filename: Path, geom: geometry.polygon.Polygon
    ) -> gpd.GeoDataFrame | None:
        """This function reads the downloaded geopackage containing the buildings. It then extracts only the buildings that lie within the geom.

        Args:
            gpkg_filename: filename of the dowloaded geopackage.
            geom: geom representing the model region.
        Returns:
            A geopandas geodataframe containing all building within the geom.
        """
        # load buildings (should already be masked by region in this step)
        from time import time

        t0 = time()
        buildings = gpd.read_file(
            gpkg_filename,
            engine="pyogrio",
            mask=geom,
            columns=["id", "occupancy", "floorspace", "height", "geometry"],
            layer="building",
            use_arrow=True,
        )
        # mask buildings to region geom
        t1 = time()
        buildings = buildings[buildings.intersects(geom)]
        t2 = time()
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
            Path to the downloaded geopackage files.

        Raises:
            RuntimeError: If download or extraction fails after all retries.
        """
        # Tile is available, so download with bz2 decompression to gpkg
        gpkg_filename = str(temp_dir / tile_filename).replace(".bz2", "")
        gpkg_filename = gpkg_filename.replace("building.", "building_")
        target_path = Path(gpkg_filename)

        success: bool = fetch_and_save(
            tile_url,
            target_path,
            decompress="bz2",
            delay_seconds=1,
            verbose=False,
            show_progress=True,
            double_delay=True,
            max_retries=17,  # will be total of ~day
        )
        if not success:
            raise RuntimeError(f"Failed to download and decompress {tile_url}")

        return target_path

    def fetch(
        self,
        url: str,
    ) -> OpenBuildingMap:
        """Fetch OpenBuildingMap data for a given region.

        Note:
            In this adapter, caching is disabled. fetch() simply returns self.
            The actual downloading and extraction happens in read().

        Args:
            url: Base URL of the OpenStreetMap server.

        Returns:
            The OpenBuildingMap instance.
        """
        self.url = url
        return self

    def read(
        self,
        geom: geometry.polygon.Polygon,
    ) -> gpd.GeoDataFrame:
        """Read the OpenBuildingMap data as a GeoDataFrame.

        Note:
            Since caching is disabled, this method downloads and processes
            the data into memory directly.

        Args:
            geom: Polygon representing the model region.

        Returns:
            A GeoDataFrame with the buildings data.

        Raises:
            RuntimeError: If no buildings can be downloaded for the model region.
        """
        # get bounds for geom
        tiles: list = self._quadkeys_for_geom(geom=geom)
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir: Path = Path(temp_dir_str)
            list_of_buildings_in_geom: list[gpd.GeoDataFrame] = []

            for tile in tqdm(tiles, desc="Downloading OpenBuildingMap tiles"):
                tile_filename = f"building.{tile}.gpkg.bz2"
                tile_url: str = f"{self.url}/{tile_filename}"

                gpkg_filename: Path = self._download_and_extract_tile(
                    tile_url, temp_dir, tile_filename
                )
                # Add all extracted geopackage files (already filtered during extraction)
                buildings = self._extract_buildings_in_geom(gpkg_filename, geom)
                if buildings is not None:
                    list_of_buildings_in_geom.append(buildings)

        # raise error if no buildings are found in model region
        if len(list_of_buildings_in_geom) == 0:
            raise RuntimeError("No OpenBuildingMap features were found in model domain")

        # concatenate all buildings
        gdf: gpd.GeoDataFrame = pd.concat(list_of_buildings_in_geom, ignore_index=True)  # ty:ignore[invalid-assignment]

        # add x and y columns for building centroids in EPSG:4326 (lon/lat)
        gdf["x"] = gdf.geometry.centroid.x
        gdf["y"] = gdf.geometry.centroid.y
        assert isinstance(gdf, gpd.GeoDataFrame)
        return gdf
