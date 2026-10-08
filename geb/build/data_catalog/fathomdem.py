"""Dataset adapter for FathomDEM (Fathom global DEM) from Zenodo."""

import json
import os
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import requests
import rioxarray as rxr
import xarray as xr
from dotenv import load_dotenv
from rioxarray import merge
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from geb.workflows.io import fetch_and_save
from geb.workflows.raster import clip_with_geometry, convert_nodata

from .base import Adapter

_ZIP_FILENAME_RE = re.compile(
    r"^([ns])(\d{2})([ew])(\d{3})-([ns])(\d{2})([ew])(\d{3})_FathomDEM_v1-0\.zip$"
)
_TILE_FILENAME_RE = re.compile(r"^([ns])(\d{2})([ew])(\d{3})\.tif$")


def parse_fathomdem_zip_bounds(filename: str) -> dict[str, float] | None:
    """Parse geographic bounding box for a 30x30 degree FathomDEM zip archive.

    Args:
        filename: Name of the zip archive, e.g. "n00e120-n30e150_FathomDEM_v1-0.zip".

    Returns:
        Dict with bounding box keys 'minx', 'miny', 'maxx', 'maxy' in degrees,
        or None if filename does not match pattern.
    """
    match = _ZIP_FILENAME_RE.match(filename)
    if not match:
        return None

    lat1_h, lat1_str, lon1_h, lon1_str, lat2_h, lat2_str, lon2_h, lon2_str = (
        match.groups()
    )

    lat1: float = -float(lat1_str) if lat1_h == "s" else float(lat1_str)
    lon1: float = -float(lon1_str) if lon1_h == "w" else float(lon1_str)
    lat2: float = -float(lat2_str) if lat2_h == "s" else float(lat2_str)
    lon2: float = -float(lon2_str) if lon2_h == "w" else float(lon2_str)

    minx: float = min(lon1, lon2)
    maxx: float = max(lon1, lon2)
    miny: float = min(lat1, lat2)
    maxy: float = max(lat1, lat2)

    return {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}


def parse_fathomdem_tile_bounds(filename: str) -> dict[str, float] | None:
    """Parse geographic bounding box for an individual 1x1 degree FathomDEM GeoTIFF tile.

    Args:
        filename: Name of the GeoTIFF file, e.g. "n00e120.tif".

    Returns:
        Dict with bounding box keys 'minx', 'miny', 'maxx', 'maxy' in degrees,
        or None if filename does not match pattern.
    """
    match = _TILE_FILENAME_RE.match(filename)
    if not match:
        return None

    lat_h, lat_str, lon_h, lon_str = match.groups()
    lat: float = -float(lat_str) if lat_h == "s" else float(lat_str)
    lon: float = -float(lon_str) if lon_h == "w" else float(lon_str)

    return {"minx": lon, "miny": lat, "maxx": lon + 1.0, "maxy": lat + 1.0}


class FathomDem(Adapter):
    """Dataset adapter for FathomDEM (Fathom global DEM) hosted on Zenodo."""

    DEFAULT_RECORD_IDS: tuple[str, ...] = ("14511570", "14523356")

    def __init__(
        self,
        record_ids: tuple[str, ...] = DEFAULT_RECORD_IDS,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter for FathomDEM.

        Args:
            record_ids: Zenodo record identifiers containing the FathomDEM files.
                Defaults to ('14511570', '14523356').
            *args: Additional positional arguments passed to the base Adapter class.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        self.record_ids: tuple[str, ...] = record_ids
        self._files_metadata_cache: list[dict[str, Any]] | None = None
        super().__init__(*args, **kwargs)

    @property
    def is_ready(self) -> bool:
        """Check if at least one extracted FathomDEM tile exists in cache.

        Returns:
            True if any extracted GeoTIFF file is found in cache root, False otherwise.
        """
        if not self.root.exists():
            return False
        has_tifs: bool = any(self.root.glob("*.tif"))
        return has_tifs

    def _get_access_token(self) -> str:
        """Retrieve the Zenodo access token from the environment.

        Loads environment variables from .env if present.

        Returns:
            The ZENODO_ACCESS_TOKEN string.

        Raises:
            ValueError: If ZENODO_ACCESS_TOKEN environment variable is not set.
        """
        load_dotenv()
        token: str | None = os.getenv("ZENODO_ACCESS_TOKEN")
        if token is None:
            raise ValueError(
                "Environment variable 'ZENODO_ACCESS_TOKEN' is required to access FathomDEM on Zenodo."
            )
        return token

    def _fetch_record_files_metadata(self, record_id: str) -> list[dict[str, Any]]:
        """Retrieve file metadata for a specific Zenodo record ID.

        Checks local disk cache first under self.root before querying the Zenodo REST API.

        Args:
            record_id: Zenodo record identifier.

        Returns:
            List of file metadata dictionaries.

        Raises:
            RuntimeError: If querying the Zenodo API fails.
        """
        metadata_cache_path: Path = self.root / f"record_{record_id}_files.json"
        if metadata_cache_path.exists():
            cached_data: list[dict[str, Any]] = json.loads(
                metadata_cache_path.read_text(encoding="utf-8")
            )
            return cached_data

        api_url: str = f"https://zenodo.org/api/records/{record_id}/files"
        token: str = self._get_access_token()
        headers: dict[str, str] = {"Authorization": f"Bearer {token}"}

        req = urllib.request.Request(api_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch file list from Zenodo API for record {record_id}: {e}"
            ) from e

        entries: list[dict[str, Any]] = payload["entries"]
        files: list[dict[str, Any]] = []
        for entry in entries:
            key: str = entry["key"]
            bounds = parse_fathomdem_zip_bounds(key)
            if bounds is None:
                continue

            download_url: str = entry["links"]["content"]
            files.append(
                {
                    "key": key,
                    "record_id": record_id,
                    "download_url": download_url,
                    "bounds": bounds,
                    "size": entry["size"],
                }
            )

        self.root.mkdir(parents=True, exist_ok=True)
        metadata_cache_path.write_text(json.dumps(files, indent=2), encoding="utf-8")
        return files

    def get_available_files_from_zenodo(self) -> list[dict[str, Any]]:
        """Retrieve the list of files and download links across all configured Zenodo records.

        Caches the combined response in memory to avoid redundant file system or network operations.

        Returns:
            List of file metadata dictionaries with keys 'key', 'record_id', 'download_url', 'bounds'.
        """
        if self._files_metadata_cache is not None:
            return self._files_metadata_cache

        combined_files: list[dict[str, Any]] = []
        for record_id in self.record_ids:
            record_files = self._fetch_record_files_metadata(record_id)
            combined_files.extend(record_files)

        self._files_metadata_cache = combined_files
        return combined_files

    def _get_intersecting_zips(self, mask: BaseGeometry) -> list[dict[str, Any]]:
        """Find zip files whose 30x30 degree bounding boxes intersect with mask.

        Args:
            mask: The geometry to test intersection against.

        Returns:
            List of file metadata dictionaries that intersect mask.
        """
        all_files: list[dict[str, Any]] = self.get_available_files_from_zenodo()
        intersecting: list[dict[str, Any]] = []
        for file_info in all_files:
            b = file_info["bounds"]
            zip_box = box(b["minx"], b["miny"], b["maxx"], b["maxy"])
            if zip_box.intersects(mask):
                intersecting.append(file_info)
        return intersecting

    def fetch(
        self,
        mask: BaseGeometry,
        url: str | None = None,
    ) -> FathomDem:
        """Fetch and extract required FathomDEM 30x30 zip files into the global cache.

        Downloads and unpacks all 30x30 zip archives intersecting the mask.

        Args:
            mask: The geometry used to filter intersecting zip packages.
            url: Optional source URL (unused, kept for Adapter interface compatibility).

        Returns:
            The FathomDem adapter instance with data unpacked in self.root.

        Raises:
            RuntimeError: If no archives intersect mask or if download/extraction fails.
        """
        del url
        self.root.mkdir(parents=True, exist_ok=True)

        zips_to_download: list[dict[str, Any]] = self._get_intersecting_zips(mask)
        if not zips_to_download:
            raise RuntimeError("No FathomDEM zip archives intersect the provided mask.")

        token: str = self._get_access_token()

        for file_info in zips_to_download:
            zip_name: str = file_info["key"]
            marker: Path = self.root / f".{zip_name}.unpacked"
            if marker.exists():
                continue

            download_url: str = file_info["download_url"]

            with tempfile.TemporaryDirectory() as tmpdir_str:
                temp_zip_path: Path = Path(tmpdir_str) / zip_name

                session = requests.Session()
                session.headers["Authorization"] = f"Bearer {token}"
                session.headers["Range"] = "bytes=0-"

                success: bool = fetch_and_save(
                    url=download_url,
                    file_path=temp_zip_path,
                    logger=self.logger,
                    session=session,
                )
                if not success:
                    raise RuntimeError(
                        f"Failed to download FathomDEM zip archive '{zip_name}'."
                    )

                with open(temp_zip_path, "rb") as f:
                    magic_bytes: bytes = f.read(4)
                if not magic_bytes.startswith(b"PK"):
                    with open(temp_zip_path, "rb") as f:
                        sample = f.read(250)
                    raise RuntimeError(
                        f"Downloaded file for '{zip_name}' is not a valid zip archive (starts with {sample!r})."
                    )

                with zipfile.ZipFile(temp_zip_path, "r") as zf:
                    zf.extractall(path=self.root)

            marker.touch()

        return self

    def _get_intersecting_tiles_for_mask(self, mask: BaseGeometry) -> list[Path]:
        """Find extracted GeoTIFF tile paths in cache that intersect with mask.

        Args:
            mask: The geometry to test intersection against.

        Returns:
            List of file paths to local 1x1 GeoTIFF tiles.
        """
        tile_paths: list[Path] = []
        for tif_file in self.root.glob("*.tif"):
            bounds = parse_fathomdem_tile_bounds(tif_file.name)
            if bounds is None:
                continue
            tile_box = box(
                bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"]
            )
            if tile_box.intersects(mask):
                tile_paths.append(tif_file)
        return sorted(tile_paths)

    def _merge_tiles(self, tile_paths: list[Path]) -> xr.DataArray:
        """Merge multiple FathomDEM GeoTIFF tiles into a single xarray DataArray.

        Args:
            tile_paths: List of file paths to local GeoTIFF tiles.

        Returns:
            Merged xarray DataArray.
        """
        das: list[xr.DataArray] = [
            rxr.open_rasterio(path, chunks={}) for path in tile_paths
        ]  # ty: ignore[invalid-assignment]
        das = [da.sel(band=1) for da in das]

        da_merged: xr.DataArray = merge.merge_arrays(das)
        assert isinstance(da_merged, xr.DataArray)
        return da_merged

    def read(self, mask: BaseGeometry) -> xr.DataArray:
        """Read FathomDEM elevation data for the area covered by mask.

        Finds all extracted 1x1 degree GeoTIFF tiles in cache that intersect
        with mask, merges them, clips to the mask geometry, and converts
        nodata values to np.nan.

        Args:
            mask: The geometry used to select and clip intersecting elevation tiles.

        Returns:
            xarray DataArray containing elevation values in meters, clipped to mask.

        Raises:
            RuntimeError: If no extracted tiles intersect the mask.
        """
        tile_paths: list[Path] = self._get_intersecting_tiles_for_mask(mask)
        if not tile_paths:
            raise RuntimeError(
                f"No extracted FathomDEM tiles found in {self.root} intersecting the provided mask. "
                "Ensure fetch(mask) has been called first."
            )

        da: xr.DataArray = self._merge_tiles(tile_paths)
        da = clip_with_geometry(
            da,
            gdf=gpd.GeoDataFrame(geometry=[mask], crs=4326),
            all_touched=True,
            drop=False,
        )
        da = convert_nodata(da, np.nan)
        chunk_x: int = min(1800, da.sizes.get("x", 1800))
        chunk_y: int = min(1800, da.sizes.get("y", 1800))
        da = da.chunk({"x": chunk_x, "y": chunk_y})
        da.attrs["_FillValue"] = np.nan
        return da.astype(np.float32)
