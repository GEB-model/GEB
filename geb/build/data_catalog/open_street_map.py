"""OpenStreetMap data catalog adapter and workflow."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry.polygon import Polygon
from tqdm import tqdm

from geb.workflows.io import fetch_and_save

from .base import Adapter


class OpenStreetMap(Adapter):
    """Open Street Map data catalog adapter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Open StreetMap adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> OpenStreetMap:
        """Sets the URL for the OSM data source.

        Args:
            url: The URL template for the OSM dataset.

        Returns:
            The OpenStreetMap adapter instance.
        """
        self.url = url
        return self

    def filter_regions(self, ID: str, parents: list[str]) -> bool:
        """Filter out regions that are children of other regions in the list.

        Args:
            ID: The ID of the region to check.
            parents: The list of parent region IDs.

        Returns:
            bool: True if the region is not a child of any other region in the list, False otherwise.
        """
        return ID not in parents

    def read(
        self, geom: Polygon, feature_types: list[str]
    ) -> dict[str, gpd.GeoDataFrame]:
        """Download and read OSM data for the specified geometry and feature types.

        First, the method identifies all regions in the OSM index that intersect with the provided geometry.
        It then filters out any regions that are children of other intersecting regions to avoid redundant data
        downloads. For each of the remaining regions, it downloads the corresponding OSM data file and
        extracts the specified feature types (e.g., buildings, roads, rails) using geopandas. The extracted features
        from all regions are concatenated into a single GeoDataFrame for each feature type.

        Args:
            geom: The geometry to read OSM data for.
            feature_types: The list of feature types to read (e.g., 'buildings', 'roads', 'rails').

        Returns:
            A dictionary with feature types as keys and GeoDataFrames as values.

        Raises:
            ValueError: If an unknown feature type is specified.
        """
        minx, miny, maxx, maxy = geom.bounds

        urls: list[str] = []
        for x in range(int(minx), int(maxx) + 1):
            # Movisda seems to switch the W and E for the x coordinate
            EW_code = f"E{-x:03d}" if x < 0 else f"W{x:03d}"
            for y in range(int(miny), int(maxy) + 1):
                NS_code = f"N{y:02d}" if y >= 0 else f"S{-y:02d}"
                # Create a polygon for the tile's bounding box (1x1 degree)
                tile_poly = Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
                # Only proceed if the tile intersects with the geometry
                if tile_poly.intersects(geom):
                    url = f"{self.url}/grid/{NS_code}{EW_code}-latest.osm.pbf"
                    # Some tiles do not exist because they are in the ocean. Therefore we check if they exist
                    # before adding the url
                    response = requests.head(url, allow_redirects=True)
                    if response.status_code != 404:
                        urls.append(url)

        all_features: dict[str, gpd.GeoDataFrame | list[gpd.GeoDataFrame]] = {}

        print(
            f"Downloading and processing OSM data for {len(urls)} tiles (note: some tiles are larger than others)..."
        )
        for url in tqdm(urls):
            with tempfile.NamedTemporaryFile(suffix=".pbf") as tmp:
                filepath: Path = Path(tmp.name)

                fetch_and_save(
                    url, filepath, overwrite=True, show_progress=False, verbose=False
                )

                for feature_type in feature_types:
                    if feature_type not in all_features:
                        all_features[feature_type] = []

                    if feature_type == "buildings":
                        features: gpd.GeoDataFrame = gpd.read_file(
                            filepath,
                            mask=geom,
                            layer="multipolygons",
                            use_arrow=True,
                        )
                        features = features[features["building"].notna()]
                    elif feature_type == "rails":
                        features: gpd.GeoDataFrame = gpd.read_file(
                            filepath,
                            mask=geom,
                            layer="lines",
                            use_arrow=True,
                        )
                        features = features[
                            features["railway"].isin(
                                ["rail", "tram", "subway", "light_rail", "narrow_gauge"]
                            )
                        ]
                    elif feature_type == "roads":
                        features: gpd.GeoDataFrame = gpd.read_file(
                            filepath,
                            mask=geom,
                            layer="lines",
                            use_arrow=True,
                        )
                        features: gpd.GeoDataFrame = features[
                            features["highway"].isin(
                                [
                                    "motorway",
                                    "trunk",
                                    "primary",
                                    "secondary",
                                    "tertiary",
                                    "unclassified",
                                    "residential",
                                    "motorway_link",
                                    "trunk_link",
                                    "primary_link",
                                    "secondary_link",
                                    "tertiary_link",
                                ]
                            )
                        ]
                    else:
                        raise ValueError(f"Unknown feature type {feature_type}")

                    all_features[feature_type].append(features)

        for feature_type in feature_types:
            all_features[feature_type] = pd.concat(
                all_features[feature_type], ignore_index=True
            )

        return all_features
