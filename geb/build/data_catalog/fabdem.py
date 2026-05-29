"""Data catalog to download FABDEM tiles for a given bounding box.

This module provides a downloader that queries the FABDEM STAC catalog hosted
on Hugging Face to discover 1x1-degree tiles intersecting a target area, then
downloads and merges the corresponding GeoTIFFs.
"""

import re
from typing import Any

import geopandas as gpd
import numpy as np
import pystac
import requests
import rioxarray as rxr
import xarray as xr
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from geb.workflows.raster import clip_with_geometry, convert_nodata

from .base import Adapter

_STAC_ITEM_NAME_RE = re.compile(r"([NS])(\d+)([EW])(\d+)")


class Fabdem(Adapter):
    """Dataset adapter for FABDEM elevation data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter for FABDEM.

        Args:
            *args: Additional positional arguments passed to the base Adapter class.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)

    def _get_item_names_from_catalog(self, catalog_url: str) -> list[str]:
        """Fetch all STAC item names from the catalog JSON.

        Parses only the catalog-level JSON (one HTTP request) to extract item
        directory names, without loading each individual item document.

        Args:
            catalog_url: URL of the STAC catalog root JSON
                (e.g. ``…/stac_catalog/catalog.json``).

        Returns:
            List of item names (directory names) found in the catalog.
        """
        response = requests.get(catalog_url, timeout=30)
        response.raise_for_status()
        catalog_data: dict = response.json()

        item_names: list[str] = []
        for link in catalog_data.get("links", []):
            if link.get("rel") == "item":
                href: str = link.get("href", "")
                # href is relative like "../N00E000_FABDEM_V1-2/N00E000_FABDEM_V1-2.json"
                # The second-to-last path segment is the item directory name.
                item_names.append(href.split("/")[-2])
        return item_names

    def _parse_item_bounds(self, item_name: str) -> dict[str, float] | None:
        """Parse the geographic bounding box of a FABDEM tile from its item name.

        FABDEM STAC item names follow the pattern ``N00W000_FABDEM_V1-2``, where
        the leading characters encode the lower-left corner of a 1×1-degree tile.

        Args:
            item_name: STAC item name such as ``N00W000_FABDEM_V1-2``.

        Returns:
            Dict with keys ``minx``, ``miny``, ``maxx``, ``maxy`` (degrees), or
            ``None`` when the name does not match the expected pattern.
        """
        match = _STAC_ITEM_NAME_RE.match(item_name)
        if not match:
            return None

        lat_dir, lat_str, lon_dir, lon_str = match.groups()
        lat: float = float(lat_str) if lat_dir == "N" else -float(lat_str)
        lon: float = float(lon_str) if lon_dir == "E" else -float(lon_str)

        # Each tile covers exactly 1×1 degrees
        return {"minx": lon, "miny": lat, "maxx": lon + 1, "maxy": lat + 1}

    def _filter_items_by_mask(
        self,
        item_names: list[str],
        mask: BaseGeometry,
    ) -> list[str]:
        """Return only those item names whose tiles intersect the mask geometry.

        Args:
            item_names: Full list of STAC item names from the catalog.
            mask: The geometry used to filter intersecting tiles.

        Returns:
            Subset of *item_names* whose 1×1-degree bounding boxes intersect *mask*.
        """
        intersecting: list[str] = []
        for item_name in item_names:
            bounds = self._parse_item_bounds(item_name)
            if bounds is None:
                continue
            tile_bbox = box(
                bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"]
            )
            if tile_bbox.intersects(mask):
                intersecting.append(item_name)
        return intersecting

    def _open_tile_from_stac_item(
        self,
        item_name: str,
        catalog_url: str,
    ) -> xr.DataArray:
        """Open a FABDEM tile as a lazy dask DataArray directly from its remote URL.

        Resolves the GeoTIFF asset URL via the STAC item JSON and opens it
        using GDAL's VSICURL virtual file system, which streams only the
        requested chunks rather than downloading the full file.

        Args:
            item_name: STAC item name such as ``N00W000_FABDEM_V1-2``.
            catalog_url: URL of the STAC catalog root JSON; used to derive the
                item JSON URL.

        Returns:
            Lazy xr.DataArray for the tile, with spatial coordinates in WGS-84
            degrees and elevation in metres.

        Raises:
            KeyError: If the asset URL cannot be resolved.
        """
        base_url: str = catalog_url.rsplit("/", 1)[0]
        item_url: str = f"{base_url}/{item_name}/{item_name}.json"
        item: pystac.Item = pystac.Item.from_file(href=item_url)

        asset = next(iter(item.assets.values()))
        asset_url: str | None = asset.get_absolute_href()
        if asset_url is None:
            raise KeyError(f"Could not resolve asset URL for STAC item '{item_name}'")

        # HuggingFace serves binary LFS files via /resolve/; /raw/ returns the
        # pointer file, which GDAL cannot parse as a GeoTIFF.
        asset_url = asset_url.replace(
            "huggingface.co/datasets/links-ads/fabdem-v12/raw/main/",
            "huggingface.co/datasets/links-ads/fabdem-v12/resolve/main/",
        )

        # GDAL VSICURL allows rioxarray to stream the remote GeoTIFF lazily;
        # dask chunks avoid loading the entire tile into memory at once.
        da: xr.DataArray = rxr.open_rasterio(
            f"/vsicurl/{asset_url}", chunks={"x": 1800, "y": 1800}
        )  # ty: ignore[invalid-assignment]
        return da.sel(band=1)

    def _merge_fabdem_tiles(self, tile_das: list[xr.DataArray]) -> xr.DataArray:
        """Merge FABDEM tile DataArrays into a single xarray DataArray.

        Args:
            tile_das: Individual 1×1-degree tile DataArrays (already band-selected).

        Returns:
            Merged xarray DataArray covering the union of all tile extents.
        """
        da = xr.combine_by_coords(
            tile_das,
            fill_value=tile_das[0].rio.nodata,
            combine_attrs="drop_conflicts",
            join="outer",
            compat="broadcast_equals",
            data_vars="all",
        )
        assert isinstance(da, xr.DataArray)
        return da

    def fetch(self, url: str) -> Fabdem:
        """Store the STAC catalog URL and return the adapter instance.

        Args:
            url: URL of the FABDEM STAC catalog root JSON on Hugging Face.

        Returns:
            The Fabdem instance (enables method chaining).
        """
        self.catalog_url = url
        return self

    def read(self, mask: BaseGeometry) -> xr.DataArray:
        """Read FABDEM elevation data for the area covered by *mask*.

        Queries the STAC catalog to discover which 1×1-degree tiles intersect
        *mask*, downloads those tiles, merges them, and clips the result to
        *mask*.

        Args:
            mask: The geometry used to select intersecting tiles and clip the
                result.

        Returns:
            xarray DataArray with FABDEM elevation values (metres) clipped to
            *mask*. Values outside the mask are set to NaN.

        Raises:
            RuntimeError: If no intersecting tiles can be downloaded.
        """
        item_names: list[str] = self._get_item_names_from_catalog(self.catalog_url)
        intersecting_items: list[str] = self._filter_items_by_mask(item_names, mask)

        tile_das: list[xr.DataArray] = [
            self._open_tile_from_stac_item(item_name, self.catalog_url)
            for item_name in intersecting_items
        ]

        if not tile_das:
            raise RuntimeError("No FABDEM tiles intersect the provided mask.")

        da: xr.DataArray = self._merge_fabdem_tiles(tile_das)
        da = clip_with_geometry(
            da,
            gdf=gpd.GeoDataFrame(geometry=[mask], crs=4326),
            all_touched=True,
            drop=False,
        )
        da = convert_nodata(da, np.nan)

        return da
