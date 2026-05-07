"""Data adapter for obtaining Copernicus data from WEkEO."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from hda import Client, Configuration
from rioxarray.exceptions import NoDataInBounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .base import Adapter

_TILE_CRS = "EPSG:3035"
_REQUEST_CRS = "EPSG:4326"

_MANUAL_TILE_DOWNLOAD_URL = (
    "https://land.copernicus.eu/en/map-viewer?product=c6d1726c6e824ae4819bdf402b785956"
)


class WEkEOCopernicus(Adapter):
    """Downloader for WEkEO Copernicus tiles.

    Downloads and extracts the needed GeoTIFF tiles from WEkEO for a given
    bounding box or geometry mask.

    Notes:
        Tile filenames are assumed to follow the pattern
        ``CLMS_HRLVLCC_CTY_S{year}_R10m_{tile}_03035_V01_R00.zip`` and the
        corresponding extracted TIFF is assumed to have the same basename with
        ``.tif`` extension.

        Tile identifiers are taken directly from the WEkEO API results rather than
        inferred from a manually reconstructed projected tile grid. This avoids
        ambiguities in the EEA reference-grid naming convention.

        Some WEkEO tiles are known to be problematic because the API can return a
        file with the expected tile name but with incorrect underlying spatial data.
        These tiles are blocked from fresh WEkEO download unless a corrected local
        TIFF is already available.

    Attributes:
        cache_dir (Path): Directory to cache extracted TIFF tiles.
    """

    def __init__(
        self,
        *args: Any,
        dataset_id: str,
        default_query: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter for WEkEO Copernicus data.

        Args:
            *args: Additional positional arguments passed to the base Adapter class.
            dataset_id: WEkEO HDA dataset identifier.
            default_query: Dataset-specific default query parameters.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)
        self.dataset_id = dataset_id
        self.default_query = default_query or {}

    def _get_client(self) -> Client:
        """Create an authenticated WEkEO HDA client.

        Returns:
            Authenticated WEkEO HDA client.

        Authentication:
            Basic auth is required and read from environment variables:
            WEKEO_USERNAME and WEKEO_PASSWORD. A new account can be made at
            https://wekeo.copernicus.eu/register .


        Raises:
            ValueError: If WEkEO credentials are not available.
        """
        username = os.getenv("WEKEO_USERNAME")
        password = os.getenv("WEKEO_PASSWORD")

        if username is None or password is None:
            raise ValueError(
                "WEKEO_USERNAME and WEKEO_PASSWORD must be set in the environment."
            )

        conf = Configuration(user=username, password=password)
        return Client(config=conf)

    def _year_dir(self, year: str | int) -> Path:
        """Return the directory containing cached files for a year.

        Args:
            year: Product year.

        Returns:
            Directory used to store the requested year's tiles.
        """
        return self.root / str(year)

    def _tile_zip_name(self, tile_id: str) -> str:
        """Return the ZIP filename for a tile result.

        Args:
            tile_id: WEkEO tile identifier.

        Returns:
            ZIP filename for the tile.
        """
        return f"{tile_id}.zip"

    def _tile_tif_name(self, tile_id: str) -> str:
        """Return the TIFF filename for a tile result.

        Args:
            tile_id: WEkEO tile identifier.

        Returns:
            TIFF filename for the tile.
        """
        return f"{tile_id}.tif"

    def _tile_zip_path(self, year: str | int, tile_id: str) -> Path:
        """Construct the local ZIP path for a tile.

        Args:
            year: Product year.
            tile_id: WEkEO tile identifier.

        Returns:
            Local ZIP file path.
        """
        return self._year_dir(year) / self._tile_zip_name(tile_id)

    def _tile_tif_path(self, year: str | int, tile_id: str) -> Path:
        """Construct the local TIFF path for a tile.

        Args:
            year: Product year.
            tile_id: WEkEO tile identifier.

        Returns:
            Local TIFF file path.
        """
        return self._year_dir(year) / self._tile_tif_name(tile_id)

    def _problem_tile_name(self, tile_id: str) -> str | None:
        """Return the known problematic tile name contained in a tile ID.

        Args:
            tile_id: WEkEO tile identifier.

        Returns:
            Matching problematic tile name, or None if the tile is not known to be
            problematic.
        """
        for tile_name in _KNOWN_WEKEO_PROBLEM_TILES:
            if tile_name in tile_id:
                return tile_name
        return None

    def _build_query(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
        query_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the WEkEO HDA query.

        Args:
            bounds: Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
            year: Product year.
            query_overrides: Additional dataset-specific query fields or overrides.

        Returns:
            Complete HDA query payload.
        """
        xmin, ymin, xmax, ymax = bounds

        query: dict[str, Any] = {"dataset_id": self.dataset_id}
        query.update(self.default_query)
        query["bbox"] = [xmin, ymin, xmax, ymax]
        query["year"] = str(year)

        if query_overrides is not None:
            query.update(query_overrides)

        return query

    def _search_tiles(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
        query_overrides: dict[str, Any] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        """Search WEkEO for tiles intersecting the requested bounds.

        Args:
            bounds: Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
            year: Product year.
            query_overrides: Additional dataset-specific query fields or overrides.

        Returns:
            A tuple containing:
                - a list of tile identifiers returned by WEkEO;
                - a lookup mapping tile identifier to downloadable WEkEO result object.

        Raises:
            FileNotFoundError: If the query returns no results.
        """
        query = self._build_query(
            bounds=bounds,
            year=year,
            query_overrides=query_overrides,
        )

        client = self._get_client()

        self.logger.info("Searching WEkEO with query %s", query)
        matches = client.search(query)

        if len(matches.results) == 0:
            raise FileNotFoundError(
                f"No WEkEO results found for year={year}, bounds={bounds}."
            )

        tile_ids: list[str] = []
        result_lookup: dict[str, Any] = {}

        for index, result in enumerate(matches.results):
            if isinstance(result, dict) and "id" in result:
                tile_id = result["id"]
                tile_ids.append(tile_id)
                result_lookup[tile_id] = matches[index]

        if not tile_ids:
            raise FileNotFoundError(
                f"WEkEO returned results for query={query}, but no result IDs were found."
            )

        return sorted(set(tile_ids)), result_lookup

    def get_tiles_for_mask(
        self,
        mask: BaseGeometry,
        year: str | int,
        query_overrides: dict[str, Any] | None = None,
    ) -> list[str]:
        """Get the WEkEO tile IDs that intersect with the mask.

        Args:
            mask: The geometry used to filter intersecting tiles. The input
                geometry is assumed to be in EPSG:4326.
            year: Product year.
            query_overrides: Additional dataset-specific query fields or overrides.

        Returns:
            A list of WEkEO tile identifiers that intersect with the mask.
        """
        tile_ids, _ = self._search_tiles(
            bounds=mask.bounds,
            year=year,
            query_overrides=query_overrides,
        )
        return tile_ids

    def download_tiles(
        self,
        tile_ids: list[str],
        year: str | int,
        bounds: tuple[float, float, float, float],
        query_overrides: dict[str, Any] | None = None,
        result_lookup: dict[str, Any] | None = None,
    ) -> None:
        """Download WEkEO ZIP tiles for the specified tile identifiers.

        Args:
            tile_ids: List of WEkEO tile identifiers to download.
            year: Product year.
            bounds: Requested bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
            query_overrides: Additional dataset-specific query fields or overrides.
            result_lookup: Optional lookup mapping tile identifiers to downloadable
                WEkEO result objects. If omitted, WEkEO is searched using ``bounds``,
                ``year``, and ``query_overrides``.

        Raises:
            FileNotFoundError: If a requested tile is not found in the WEkEO results,
                if no ZIP file is downloaded for a tile, or if a known problematic
                tile would need to be downloaded from WEkEO.
        """
        year_dir = self._year_dir(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        if result_lookup is None:
            self.logger.info(
                "No WEkEO result lookup was provided. Searching WEkEO before downloading "
                "tiles for year %s and bounds %s.",
                year,
                bounds,
            )
            _, result_lookup = self._search_tiles(
                bounds=bounds,
                year=year,
                query_overrides=query_overrides,
            )

        self.logger.info(
            "Preparing to download %s WEkEO tile(s) for year %s into %s.",
            len(tile_ids),
            year,
            year_dir,
        )

        for tile_id in tile_ids:
            zip_path = self._tile_zip_path(year, tile_id)
            tif_path = self._tile_tif_path(year, tile_id)

            if tif_path.exists():
                self.logger.info(
                    "Skipping WEkEO tile %s for year %s because TIFF already exists: %s",
                    tile_id,
                    year,
                    tif_path,
                )
                continue

            if zip_path.exists():
                self.logger.info(
                    "Skipping download of WEkEO tile %s for year %s because ZIP already "
                    "exists and will be reused: %s",
                    tile_id,
                    year,
                    zip_path,
                )
                continue

            if tile_id not in result_lookup:
                raise FileNotFoundError(
                    f"Tile {tile_id} was requested but not found in WEkEO search results."
                )

            before_files = set(year_dir.iterdir()) if year_dir.exists() else set()

            self.logger.info(
                "Downloading WEkEO tile %s for year %s into %s.",
                tile_id,
                year,
                year_dir,
            )
            result_lookup[tile_id].download(str(year_dir))

            after_files = set(year_dir.iterdir())
            new_files = sorted(after_files - before_files)

            zip_files = [
                file_path
                for file_path in new_files
                if file_path.suffix.lower() == ".zip"
            ]
            if not zip_files:
                raise FileNotFoundError(
                    f"No ZIP file downloaded for tile {tile_id}, year {year}."
                )

            matching_zip = next(
                (file_path for file_path in zip_files if file_path.stem == tile_id),
                None,
            )
            downloaded_zip = matching_zip if matching_zip is not None else zip_files[0]

            if downloaded_zip != zip_path:
                self.logger.info(
                    "Renaming downloaded WEkEO ZIP from %s to expected path %s.",
                    downloaded_zip,
                    zip_path,
                )
                if zip_path.exists():
                    zip_path.unlink()
                downloaded_zip.replace(zip_path)

            self.logger.info(
                "Finished downloading WEkEO tile %s for year %s: %s",
                tile_id,
                year,
                zip_path,
            )

    def unpack_and_merge_tiles(
        self,
        tile_ids: list[str],
        year: str | int,
    ) -> xr.DataArray:
        """Unpack and merge WEkEO tiles into a single dataarray.

        TIFFs are extracted into the permanent cache folder if needed and reused on
        subsequent reads.

        Args:
            tile_ids: List of WEkEO tile identifiers to unpack and merge.
            year: Product year.

        Returns:
            Merged dataarray of the specified tiles.
        """
        extracted_paths = self._unpack_tiles(tile_ids, year)
        da = self._merge_tiles(extracted_paths)
        return da

    def _merge_tiles(self, tile_paths: list[Path]) -> xr.DataArray:
        """Merge extracted WEkEO tiles into a single xarray DataArray.

        Args:
            tile_paths: List of paths to the extracted tile files.

        Returns:
            Merged dataset of the specified tiles.
        """
        das: list[xr.DataArray] = [
            rxr.open_rasterio(path, chunks={}) for path in tile_paths
        ]
        das = [da.sel(band=1) for da in das]
        da = xr.combine_by_coords(
            das,
            fill_value=das[0].rio.nodata,
            combine_attrs="drop_conflicts",
            join="outer",
            compat="broadcast_equals",
            data_vars="all",
        )
        assert isinstance(da, xr.DataArray)
        return da

    def _unpack_tiles(
        self,
        tile_ids: list[str],
        year: str | int,
    ) -> list[Path]:
        """Unpack the requested TIFF tiles from local ZIP files into the year folder.

        If a TIFF tile already exists locally, it is reused. If only a ZIP exists,
        the TIFF is extracted into the permanent cache folder and the ZIP is removed.

        Args:
            tile_ids: List of WEkEO tile identifiers to unpack.
            year: Product year.

        Returns:
            List of local TIFF tile paths.

        Raises:
            FileNotFoundError: If neither a TIFF nor a ZIP exists for a requested tile,
                or if a ZIP contains no TIFF.
        """
        extracted_paths: list[Path] = []

        for tile_id in tile_ids:
            tif_path = self._tile_tif_path(year, tile_id)
            zip_path = self._tile_zip_path(year, tile_id)

            if tif_path.exists():
                extracted_paths.append(tif_path)
                continue

            if not zip_path.exists():
                raise FileNotFoundError(
                    f"Neither TIFF nor ZIP found for tile {tile_id}, year {year}."
                )

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                tif_members = [
                    name
                    for name in zip_ref.namelist()
                    if name.lower().endswith((".tif", ".tiff"))
                ]
                if not tif_members:
                    raise FileNotFoundError(f"No TIFF found inside archive {zip_path}.")

                matching_member = next(
                    (name for name in tif_members if Path(name).stem == tile_id),
                    None,
                )
                tif_member = (
                    matching_member if matching_member is not None else tif_members[0]
                )
                extracted_path = Path(zip_ref.extract(tif_member, path=tif_path.parent))

            if extracted_path != tif_path:
                if tif_path.exists():
                    tif_path.unlink()
                extracted_path.replace(tif_path)

            zip_path.unlink()
            extracted_paths.append(tif_path)

        return extracted_paths

    def fetch(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
        url: str | None = None,
        query_overrides: dict[str, Any] | None = None,
    ) -> WEkEOCopernicus:
        """Fetch WEkEO tiles for the specified bounds.

        Args:
            bounds: Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
            year: Product year.
            url: URL to download WEkEO data from. Defaults to None.
            query_overrides: Additional dataset-specific query fields or overrides.

        Returns:
            The WEkEO Copernicus adapter instance with the downloaded data.
        """
        self.logger.info(
            "Fetching WEkEO Copernicus data for year %s and bounds %s.",
            year,
            bounds,
        )

        tile_ids, result_lookup = self._search_tiles(
            bounds=bounds,
            year=year,
            query_overrides=query_overrides,
        )

        self.logger.info(
            "Found %s WEkEO tile(s) for year %s: %s",
            len(tile_ids),
            year,
            tile_ids,
        )

        self.tile_ids = tile_ids
        self.year = year

        self.download_tiles(
            tile_ids=self.tile_ids,
            year=year,
            bounds=bounds,
            query_overrides=query_overrides,
            result_lookup=result_lookup,
        )

        self.logger.info(
            "Finished fetching WEkEO Copernicus data for year %s and bounds %s.",
            year,
            bounds,
        )

        return self

    def read(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int | None = None,
        query_overrides: dict[str, Any] | None = None,
    ) -> xr.DataArray:
        """Read and unpack the downloaded WEkEO data, clipping it to the requested bounds.

        Args:
            bounds: Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
            year: Product year. If None, uses the year from the most recent fetch.
            query_overrides: Additional dataset-specific query fields or overrides.
                Only used if tiles need to be re-identified from the API.

        Returns:
            The downloaded and merged WEkEO data.

        Raises:
            ValueError: If no year is provided and no previous fetch set a year,
                or if clipping fails due to inconsistent tile spatial data.
        """
        read_year = year if year is not None else getattr(self, "year", None)
        if read_year is None:
            raise ValueError("Year must be provided to read WEkEO tiles.")

        tile_ids = getattr(self, "tile_ids", None)
        if tile_ids is None:
            mask = box(*bounds)
            tile_ids = self.get_tiles_for_mask(
                mask=mask,
                year=read_year,
                query_overrides=query_overrides,
            )

        da = self.unpack_and_merge_tiles(tile_ids, read_year)

        mask_projected = (
            gpd.GeoSeries([box(*bounds)], crs=_REQUEST_CRS).to_crs(_TILE_CRS).iloc[0]
        )
        min_x, min_y, max_x, max_y = mask_projected.bounds

        try:
            da = da.rio.clip_box(
                minx=min_x,
                miny=min_y,
                maxx=max_x,
                maxy=max_y,
            )
        except NoDataInBounds as error:
            data_bounds = da.rio.bounds()
            raise ValueError(
                "Failed to clip the merged WEkEO raster to the requested bounds because "
                "no raster data were found inside the requested bounding box. This can "
                "happen when WEkEO returns a tile with the expected file name but with "
                "incorrect underlying spatial data."
                f"Requested bounds in {_REQUEST_CRS}: {bounds}\n"
                f"Projected clip bounds in {_TILE_CRS}: "
                f"{(min_x, min_y, max_x, max_y)}\n"
                f"Merged raster bounds: {data_bounds}\n"
                f"Tile IDs used: {tile_ids}\n\n"
                "Retrieve the tile manually from "
                f"{_MANUAL_TILE_DOWNLOAD_URL}."
                "Or contact: luca.battistella@eea.europa.eu"
            ) from error

        da = da.rio.reproject(_REQUEST_CRS)

        return da
