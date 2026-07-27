"""Local adapter for HRL-compatible generated Croplands tiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rioxarray.merge import merge_arrays
from shapely.geometry import box

from .base import Adapter


HRL_OUTSIDE_AREA_CODE = 65535


class LocalHRLCroplands(Adapter):
    """Read locally generated CTY or CPSCT tiles using the HRL layout.

    The adapter expects files below ``self.root / str(year)`` with names such as::

        CLMS_HRLVLCC_CTY_S2024_R10m_E40N30_03035_V01_R00.tif
        CLMS_HRLVLCC_CPSCT_S2024_R10m_E40N30_03035_V01_R00.tif

    It implements the same ``fetch(...).read(...)`` surface used by the existing
    Europe HRL workflow, but it never contacts WEkEO.
    """

    def __init__(
        self,
        *args: Any,
        product_code: str,
        **kwargs: Any,
    ) -> None:
        """Initialize a local HRL-compatible tile adapter.

        Args:
            *args: Positional arguments passed to :class:`Adapter`.
            product_code: ``"CTY"`` or ``"CPSCT"``.
            **kwargs: Standard adapter settings such as ``folder``,
                ``local_version``, ``filename``, and ``cache``.
        """
        super().__init__(*args, **kwargs)
        if product_code not in {"CTY", "CPSCT"}:
            raise ValueError("product_code must be 'CTY' or 'CPSCT'.")
        self.product_code = product_code
        self.tile_ids: list[str] = []
        self.year: int | None = None
        self.bounds: tuple[float, float, float, float] | None = None

    def _year_directory(self, year: int) -> Path:
        """Return the local directory containing one generated HRL year."""
        return Path(self.root) / str(int(year))

    def _candidate_paths(self, year: int) -> list[Path]:
        """Return all local tiles matching this product and year."""
        directory = self._year_directory(year)
        pattern = (
            f"CLMS_HRLVLCC_{self.product_code}_S{int(year)}_R10m_*_03035_V*_R*.tif"
        )
        return sorted(directory.rglob(pattern)) if directory.exists() else []

    @staticmethod
    def _intersects_wgs84_bounds(
        path: Path,
        bounds: tuple[float, float, float, float],
    ) -> bool:
        """Check whether one raster intersects a WGS84 query bounding box."""
        with rasterio.open(path) as source:
            raster_bounds = transform_bounds(
                source.crs,
                "EPSG:4326",
                *source.bounds,
                densify_pts=21,
            )
        return box(*raster_bounds).intersects(box(*bounds))

    def fetch(
        self,
        url: None,
        *,
        bounds: tuple[float, float, float, float],
        year: int,
        **_: Any,
    ) -> LocalHRLCroplands:
        """Discover local generated tiles intersecting a WGS84 bounding box.

        Args:
            url: Must be ``None`` because this adapter is local-only.
            bounds: Query bounds as ``(min_lon, min_lat, max_lon, max_lat)``.
            year: Generated HRL-compatible year.
            **_: Ignored compatibility arguments.

        Returns:
            The current adapter with ``tile_ids`` populated.

        Raises:
            FileNotFoundError: If no matching local generated tiles are found.
        """
        if url is not None:
            raise ValueError("LocalHRLCroplands requires url=None.")
        if len(bounds) != 4:
            raise ValueError("bounds must contain four WGS84 values.")
        min_lon, min_lat, max_lon, max_lat = (float(value) for value in bounds)
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError(f"Invalid WGS84 bounds: {bounds}")

        paths = [
            path
            for path in self._candidate_paths(int(year))
            if self._intersects_wgs84_bounds(
                path,
                (min_lon, min_lat, max_lon, max_lat),
            )
        ]
        if not paths:
            expected_directory = self._year_directory(int(year))
            raise FileNotFoundError(
                f"No local generated {self.product_code} tiles for year {year} "
                f"intersect bounds {bounds}. Expected files below "
                f"{expected_directory}."
            )

        self.year = int(year)
        self.bounds = (min_lon, min_lat, max_lon, max_lat)
        self.tile_ids = [path.stem for path in paths]
        return self

    def _tile_path(self, tile_id: str, year: int) -> Path:
        """Resolve one discovered tile ID to its GeoTIFF path."""
        directory = self._year_directory(year)
        direct = directory / f"{Path(tile_id).stem}.tif"
        if direct.exists():
            return direct
        matches = sorted(directory.rglob(f"{Path(tile_id).stem}.tif"))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(
                f"Missing generated HRL tile {tile_id}.tif below {directory}."
            )
        raise RuntimeError(
            f"Multiple local files match generated HRL tile {tile_id}: {matches}"
        )

    def read(
        self,
        *,
        bounds: tuple[float, float, float, float],
        year: int,
        dst_crs: str | int | None = None,
        normalize_nodata: bool = False,
        chunks: dict[str, int] | None = None,
        **_: Any,
    ) -> xr.DataArray:
        """Read, merge, and clip local generated HRL-compatible tiles.

        Args:
            bounds: WGS84 bounds as ``(min_lon, min_lat, max_lon, max_lat)``.
            year: Generated year to read.
            dst_crs: Optional output CRS.
            normalize_nodata: Convert the HRL outside-area code 65535 to 0.
            chunks: Optional rioxarray/dask chunk mapping.
            **_: Ignored compatibility arguments.

        Returns:
            Two-dimensional merged HRL-compatible raster.
        """
        if self.year != int(year) or self.bounds != tuple(float(v) for v in bounds):
            self.fetch(url=None, bounds=bounds, year=year)

        arrays: list[xr.DataArray] = []
        for tile_id in self.tile_ids:
            raster = rioxarray.open_rasterio(
                self._tile_path(tile_id, int(year)),
                masked=False,
                chunks=chunks,
            )
            if raster.sizes.get("band", 0) != 1:
                raise ValueError(f"Generated HRL tile {tile_id} must contain one band.")
            arrays.append(raster.squeeze("band", drop=True))

        merged = merge_arrays(arrays, nodata=HRL_OUTSIDE_AREA_CODE)
        projected_bounds = transform_bounds(
            "EPSG:4326",
            merged.rio.crs,
            *bounds,
            densify_pts=21,
        )
        merged = merged.rio.clip_box(
            minx=projected_bounds[0],
            miny=projected_bounds[1],
            maxx=projected_bounds[2],
            maxy=projected_bounds[3],
            allow_one_dimensional_raster=True,
        )

        if dst_crs is not None and str(merged.rio.crs) != str(dst_crs):
            merged = merged.rio.reproject(
                dst_crs,
                resampling=Resampling.nearest,
                nodata=HRL_OUTSIDE_AREA_CODE,
            )

        if normalize_nodata:
            merged = xr.where(merged == HRL_OUTSIDE_AREA_CODE, 0, merged)
            merged = merged.astype(np.uint16)
            merged = merged.rio.write_nodata(0)
        else:
            merged = merged.astype(np.uint16)
            merged = merged.rio.write_nodata(HRL_OUTSIDE_AREA_CODE)

        merged.name = "crop_type" if self.product_code == "CTY" else "secondary_crop"
        return merged
