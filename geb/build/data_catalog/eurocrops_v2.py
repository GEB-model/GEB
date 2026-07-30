"""Data adapter for the EuroCrops v2 parcel-level GSA reference dataset."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import io
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

import aiohttp
import geopandas as gpd
import pandas as pd
from aiohttp_retry import ExponentialRetry, RetryClient
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .base import Adapter


DEFAULT_BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/EuroCropsV2"
DEFAULT_MAPPING_URL = (
    "https://raw.githubusercontent.com/Martincccc/EuroCropsV2/main/"
    "data/cropcodemapping/eurocrops.csv"
)
MANIFEST_FILENAME = "all_gpqt_files.csv"
DOWNLOAD_CHUNK_SIZE_BYTES = 8 * 1024 * 1024
DEFAULT_DATA_SUBDIRECTORIES = ("gpqtv202", "gpqtv201", "gpqtv2", "gpqt")

# EuroCrops v2 uses subnational identifiers for several countries. This mapping
# provides the complete public coverage for country-level requests.
COUNTRY_TO_DATASET_REGIONS: dict[str, tuple[str, ...]] = {
    "AUT": ("at",),
    "BEL": ("be2", "be3"),
    "BGR": ("bg",),
    "CZE": ("cz",),
    "DEU": ("de4", "dea"),
    "DNK": ("dk",),
    "EST": ("ee",),
    "ESP": ("es",),
    "FIN": ("fi",),
    "FRA": ("fr",),
    "IRL": ("ie",),
    "ITA": ("iti1",),
    "NLD": ("nl",),
    "PRT": ("pt",),
    "SVN": ("si",),
    "SVK": ("sk",),
}


class EuroCropsV2(Adapter):
    """Download and harmonize selected EuroCrops v2 country-year parcels.

    EuroCrops v2 consists of annual GeoParquet files in EPSG:3035 and a separate
    HCAT v4 crop-code mapping. This adapter downloads only selected region-year
    files, joins the official mapping and optionally clips the parcels to an
    exact model geometry.
    """

    def __init__(
        self,
        *args: Any,
        max_parallel_downloads: int = 3,
        data_subdirectories: Sequence[str] = DEFAULT_DATA_SUBDIRECTORIES,
        **kwargs: Any,
    ) -> None:
        """Initialize the EuroCrops v2 adapter."""
        super().__init__(*args, **kwargs)

        if max_parallel_downloads < 1:
            raise ValueError("max_parallel_downloads must be at least 1.")
        normalized_subdirectories = tuple(
            dict.fromkeys(str(value).strip("/") for value in data_subdirectories)
        )
        if not normalized_subdirectories:
            raise ValueError("At least one data subdirectory must be configured.")

        # All cache paths are derived from Adapter.root, which in turn follows
        # GEB_DATA_ROOT and the catalog folder/local_version/cache settings.
        dataset_root = Path(self.root)
        self.metadata_root = dataset_root / "metadata"
        self.parcel_root = dataset_root / "parcels"
        self.mapping_root = dataset_root / "mapping"
        for directory in (
            self.metadata_root,
            self.parcel_root,
            self.mapping_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.max_parallel_downloads = int(max_parallel_downloads)
        self.data_subdirectories = normalized_subdirectories
        self.url = DEFAULT_BASE_URL
        self.mapping_url = DEFAULT_MAPPING_URL

    def fetch(self, url: str | None) -> EuroCropsV2:
        """Set the EuroCrops repository base URL and return this adapter."""
        self.url = (url or DEFAULT_BASE_URL).rstrip("/")
        return self

    @property
    def manifest_url(self) -> str:
        """Return the official file-manifest URL."""
        return f"{self.url}/{MANIFEST_FILENAME}"

    @property
    def manifest_path(self) -> Path:
        """Return the cached file manifest."""
        return self.metadata_root / MANIFEST_FILENAME

    @property
    def mapping_path(self) -> Path:
        """Return the cached EuroCrops crop-code mapping."""
        return self.mapping_root / "eurocrops.csv"

    @staticmethod
    def _retry_options() -> ExponentialRetry:
        """Return common retry settings."""
        return ExponentialRetry(
            attempts=8,
            start_timeout=5,
            max_timeout=120,
            factor=2,
            retry_all_server_errors=True,
        )

    @staticmethod
    def _timeout() -> aiohttp.ClientTimeout:
        """Return timeouts suitable for large GeoParquet files."""
        return aiohttp.ClientTimeout(
            total=None,
            sock_connect=60,
            sock_read=3600,
        )

    @staticmethod
    async def _download_file(
        client: RetryClient,
        remote_urls: Sequence[str],
        destination: Path,
        *,
        overwrite: bool,
        semaphore: asyncio.Semaphore,
    ) -> tuple[Path, str]:
        """Download one file atomically, trying versioned repository fallbacks."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0 and not overwrite:
            return destination, "cached"

        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        errors: list[str] = []

        async with semaphore:
            for remote_url in remote_urls:
                try:
                    async with client.get(
                        remote_url,
                        raise_for_status=True,
                    ) as response:
                        with temporary.open("wb") as file_handle:
                            async for chunk in response.content.iter_chunked(
                                DOWNLOAD_CHUNK_SIZE_BYTES
                            ):
                                file_handle.write(chunk)
                    if temporary.stat().st_size == 0:
                        raise IOError("downloaded file is empty")
                    temporary.replace(destination)
                    print(f"Saved EuroCrops file: {destination}")
                    return destination, remote_url
                except (aiohttp.ClientError, OSError) as error:
                    temporary.unlink(missing_ok=True)
                    errors.append(f"{remote_url}: {type(error).__name__}: {error}")

        raise FileNotFoundError(
            "Could not download EuroCrops file from any configured repository "
            f"location for {destination.name}:\n" + "\n".join(errors)
        )

    async def _ensure_small_file(
        self,
        *,
        remote_url: str,
        destination: Path,
        refresh: bool,
    ) -> Path:
        """Download one small metadata file when absent."""
        if destination.exists() and destination.stat().st_size > 0 and not refresh:
            return destination
        async with RetryClient(
            retry_options=self._retry_options(),
            timeout=self._timeout(),
        ) as client:
            await self._download_file(
                client,
                (remote_url,),
                destination,
                overwrite=refresh,
                semaphore=asyncio.Semaphore(1),
            )
        return destination

    @staticmethod
    def _parse_manifest_text(text: str) -> pd.DataFrame:
        """Parse the official manifest despite minor schema/version changes."""
        filenames: list[str] = []
        try:
            table = pd.read_csv(io.StringIO(text))
        except pd.errors.ParserError:
            table = pd.DataFrame()

        if not table.empty:
            for column in table.columns:
                values = table[column].dropna().astype(str)
                matches = values.loc[
                    values.str.lower().str.contains(r"\.parquet(?:$|\?)", regex=True)
                ]
                filenames.extend(matches.tolist())

        filenames.extend(
            re.findall(r"[A-Za-z0-9_./-]+\.parquet", text, flags=re.IGNORECASE)
        )
        normalized: list[str] = []
        for value in filenames:
            path = urlparse(str(value).strip()).path
            name = Path(path).name
            if name and name.lower().endswith(".parquet"):
                normalized.append(name)

        rows: list[dict[str, Any]] = []
        pattern = re.compile(
            r"^(?P<region>[a-z0-9]+)_(?P<year>\d{4}|stack)\.parquet$",
            re.IGNORECASE,
        )
        for filename in dict.fromkeys(normalized):
            match = pattern.match(filename)
            if match is None:
                continue
            raw_year = match.group("year").lower()
            rows.append(
                {
                    "filename": filename,
                    "dataset_region": match.group("region").lower(),
                    "year": None if raw_year == "stack" else int(raw_year),
                    "is_stack": raw_year == "stack",
                }
            )

        if not rows:
            raise ValueError(
                "The EuroCrops manifest did not contain recognizable "
                "<region>_<year>.parquet filenames."
            )
        return pd.DataFrame(rows).sort_values(
            ["dataset_region", "is_stack", "year"],
            na_position="last",
            ignore_index=True,
        )

    async def list_files_async(self, *, refresh: bool = False) -> pd.DataFrame:
        """Return the official annual and stack file inventory."""
        path = await self._ensure_small_file(
            remote_url=self.manifest_url,
            destination=self.manifest_path,
            refresh=refresh,
        )
        return self._parse_manifest_text(path.read_text(encoding="utf-8"))

    def list_files(self, *, refresh: bool = False) -> pd.DataFrame:
        """Synchronously return the EuroCrops file inventory."""
        return self._run_sync(
            self.list_files_async(refresh=refresh),
            "EuroCropsV2.list_files",
        )

    async def read_mapping_async(self, *, refresh: bool = False) -> pd.DataFrame:
        """Return the official HCAT v4 crop-code mapping table."""
        path = await self._ensure_small_file(
            remote_url=self.mapping_url,
            destination=self.mapping_path,
            refresh=refresh,
        )
        mapping = pd.read_csv(
            path,
            dtype={"nuts": "string", "original_code": "string"},
        )
        required = {"nuts", "original_code", "hcat4_code", "hcat4_name"}
        missing = required - set(mapping.columns)
        if missing:
            raise ValueError(
                f"EuroCrops mapping is missing required columns: {sorted(missing)}."
            )
        mapping["nuts"] = mapping["nuts"].str.lower().str.strip()
        mapping["original_code"] = mapping["original_code"].str.strip()
        mapping["hcat4_code"] = pd.to_numeric(
            mapping["hcat4_code"], errors="coerce"
        ).astype("Int64")
        return mapping.drop_duplicates(["nuts", "original_code"]).reset_index(drop=True)

    def read_mapping(self, *, refresh: bool = False) -> pd.DataFrame:
        """Synchronously return the HCAT v4 mapping table."""
        return self._run_sync(
            self.read_mapping_async(refresh=refresh),
            "EuroCropsV2.read_mapping",
        )

    @staticmethod
    def regions_for_countries(countries: str | Sequence[str]) -> tuple[str, ...]:
        """Translate ISO3 country codes to EuroCrops dataset regions."""
        values = (countries,) if isinstance(countries, str) else tuple(countries)
        regions: list[str] = []
        unknown: list[str] = []
        for country in values:
            normalized = str(country).upper().strip()
            if normalized not in COUNTRY_TO_DATASET_REGIONS:
                unknown.append(normalized)
                continue
            regions.extend(COUNTRY_TO_DATASET_REGIONS[normalized])
        if unknown:
            raise KeyError(
                "EuroCrops v2 has no configured public region mapping for "
                f"country code(s) {sorted(set(unknown))}."
            )
        return tuple(dict.fromkeys(regions))

    @staticmethod
    def _normalize_years(years: int | Sequence[int]) -> tuple[int, ...]:
        """Normalize requested observation years."""
        values = (years,) if isinstance(years, int) else tuple(years)
        normalized = tuple(sorted({int(value) for value in values}))
        if not normalized:
            raise ValueError("At least one EuroCrops year must be requested.")
        return normalized

    @staticmethod
    def _normalize_geometry(
        geometry: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | None,
        bounds: tuple[float, float, float, float] | None,
    ) -> BaseGeometry | None:
        """Return one EPSG:3035 clipping geometry."""
        if geometry is not None and bounds is not None:
            raise ValueError("Supply geometry or bounds, not both.")
        if bounds is not None:
            if len(bounds) != 4:
                raise ValueError("bounds must contain four WGS84 coordinates.")
            query = gpd.GeoSeries([box(*map(float, bounds))], crs=4326).to_crs(3035)
            return query.iloc[0]
        if geometry is None:
            return None
        if isinstance(geometry, gpd.GeoDataFrame):
            if geometry.crs is None:
                raise ValueError("Query GeoDataFrame must define a CRS.")
            result = unary_union(geometry.to_crs(3035).geometry.dropna().tolist())
        elif isinstance(geometry, gpd.GeoSeries):
            if geometry.crs is None:
                raise ValueError("Query GeoSeries must define a CRS.")
            result = unary_union(geometry.to_crs(3035).dropna().tolist())
        elif isinstance(geometry, BaseGeometry):
            # Bare Shapely geometries have no CRS; the adapter follows the GEB
            # convention and interprets them as WGS84.
            result = gpd.GeoSeries([geometry], crs=4326).to_crs(3035).iloc[0]
        else:
            raise TypeError(
                "geometry must be a Shapely geometry, GeoSeries, GeoDataFrame or None."
            )
        if result.is_empty:
            raise ValueError("EuroCrops query geometry is empty.")
        return result if result.is_valid else result.buffer(0)

    async def select_files_async(
        self,
        *,
        years: int | Sequence[int],
        regions: str | Sequence[str] | None = None,
        countries: str | Sequence[str] | None = None,
        refresh_manifest: bool = False,
    ) -> pd.DataFrame:
        """Select annual EuroCrops files by region/country and year."""
        if regions is None and countries is None:
            raise ValueError(
                "Select EuroCrops files using regions, countries, or both; "
                "unbounded Europe-wide downloads are intentionally disabled."
            )
        selected_regions: list[str] = []
        if regions is not None:
            values = (regions,) if isinstance(regions, str) else tuple(regions)
            selected_regions.extend(str(value).lower().strip() for value in values)
        if countries is not None:
            selected_regions.extend(self.regions_for_countries(countries))
        selected_regions = list(
            dict.fromkeys(value for value in selected_regions if value)
        )
        requested_years = self._normalize_years(years)

        inventory = await self.list_files_async(refresh=refresh_manifest)
        selection = inventory.loc[
            ~inventory["is_stack"]
            & inventory["dataset_region"].isin(selected_regions)
            & inventory["year"].isin(requested_years)
        ].copy()
        expected = {
            (region, year) for region in selected_regions for year in requested_years
        }
        represented = set(
            zip(selection["dataset_region"], selection["year"], strict=False)
        )
        missing = sorted(expected - represented)
        if missing:
            print(
                "EuroCrops v2 has no public annual file for region/year pairs: "
                f"{missing}"
            )
        if selection.empty:
            raise FileNotFoundError(
                "No EuroCrops v2 files match the selected regions and years."
            )
        return selection.reset_index(drop=True)

    def select_files(self, **kwargs: Any) -> pd.DataFrame:
        """Synchronously select EuroCrops files."""
        return self._run_sync(
            self.select_files_async(**kwargs),
            "EuroCropsV2.select_files",
        )

    def _candidate_urls(self, filename: str) -> tuple[str, ...]:
        """Return versioned repository candidates for one file."""
        return tuple(
            f"{self.url}/{subdirectory}/{filename}"
            for subdirectory in self.data_subdirectories
        )

    async def read_async(
        self,
        *,
        years: int | Sequence[int],
        regions: str | Sequence[str] | None = None,
        countries: str | Sequence[str] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        geometry: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
        include_mapping: bool = True,
        drop_unmapped: bool = False,
        overwrite: bool = False,
        refresh_manifest: bool = False,
        refresh_mapping: bool = False,
        max_files: int | None = 100,
    ) -> gpd.GeoDataFrame:
        """Download, harmonize and spatially filter EuroCrops parcels."""
        selection = await self.select_files_async(
            years=years,
            regions=regions,
            countries=countries,
            refresh_manifest=refresh_manifest,
        )
        if max_files is not None and len(selection) > max_files:
            raise RuntimeError(
                f"EuroCrops selection contains {len(selection)} files, exceeding "
                f"max_files={max_files}. Narrow the request or explicitly raise "
                "the safety limit."
            )

        semaphore = asyncio.Semaphore(self.max_parallel_downloads)
        async with RetryClient(
            retry_options=self._retry_options(),
            timeout=self._timeout(),
        ) as client:
            tasks = []
            for row in selection.itertuples(index=False):
                destination = (
                    self.parcel_root
                    / str(row.dataset_region)
                    / str(int(row.year))
                    / str(row.filename)
                )
                tasks.append(
                    self._download_file(
                        client,
                        self._candidate_urls(str(row.filename)),
                        destination,
                        overwrite=overwrite,
                        semaphore=semaphore,
                    )
                )
            downloaded = await asyncio.gather(*tasks)

        mapping = (
            await self.read_mapping_async(refresh=refresh_mapping)
            if include_mapping
            else None
        )
        query_geometry = self._normalize_geometry(geometry, bounds)
        tables: list[gpd.GeoDataFrame] = []

        for row, (path, resolved_url) in zip(
            selection.itertuples(index=False),
            downloaded,
            strict=True,
        ):
            parcels = gpd.read_parquet(path)
            if parcels.crs is None:
                parcels = parcels.set_crs(3035)
            elif parcels.crs.to_epsg() != 3035:
                parcels = parcels.to_crs(3035)
            if parcels.geometry.name != "geom":
                if "geom" in parcels.columns:
                    parcels = parcels.set_geometry("geom")
                else:
                    parcels = parcels.rename_geometry("geom")

            required = {"cropfield", "original_code", "area_ha", "geom"}
            missing = required - set(parcels.columns)
            if missing:
                raise ValueError(
                    f"EuroCrops file {path} is missing columns {sorted(missing)}."
                )

            # ``GeoSeries.notna()`` emits a compatibility warning when empty
            # geometries are present. ``isna()`` has the required missing-value
            # semantics without that warning; empty geometries are removed
            # explicitly and independently.
            valid_geometry = ~parcels.geometry.isna() & ~parcels.geometry.is_empty
            parcels = parcels.loc[valid_geometry].copy()
            if query_geometry is not None and not parcels.empty:
                indices = parcels.sindex.query(query_geometry, predicate="intersects")
                parcels = parcels.iloc[indices].copy()
                parcels = parcels.loc[
                    parcels.geometry.intersects(query_geometry)
                ].copy()
            if parcels.empty:
                continue

            parcels["nuts"] = str(row.dataset_region)
            parcels["observation_year"] = int(row.year)
            parcels["original_code"] = parcels["original_code"].astype(str).str.strip()
            parcels["source_feature_id"] = (
                parcels["nuts"].astype(str)
                + ":"
                + parcels["observation_year"].astype(str)
                + ":"
                + parcels["cropfield"].astype(str)
            )
            parcels["reference_source"] = "eurocrops_v2"
            parcels["source_file"] = str(path)
            parcels["source_url"] = str(resolved_url)

            if mapping is not None:
                parcels = parcels.merge(
                    mapping,
                    on=["nuts", "original_code"],
                    how="left",
                    validate="many_to_one",
                )
                if drop_unmapped:
                    parcels = parcels.loc[parcels["hcat4_code"].notna()].copy()
            tables.append(parcels)

        if not tables:
            return gpd.GeoDataFrame(
                columns=[
                    "cropfield",
                    "original_code",
                    "area_ha",
                    "nuts",
                    "observation_year",
                    "source_feature_id",
                    "geom",
                ],
                geometry="geom",
                crs="EPSG:3035",
            )

        combined = gpd.GeoDataFrame(
            pd.concat(tables, ignore_index=True),
            geometry="geom",
            crs="EPSG:3035",
        )
        return combined.drop_duplicates("source_feature_id").reset_index(drop=True)

    def read(self, **kwargs: Any) -> gpd.GeoDataFrame:
        """Synchronously retrieve selected EuroCrops v2 parcels."""
        return self._run_sync(
            self.read_async(**kwargs),
            "EuroCropsV2.read",
        )

    @staticmethod
    def _run_sync(coroutine: Any, method_name: str) -> Any:
        """Run a coroutine outside an active event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        raise RuntimeError(
            f"{method_name}() cannot run inside an active asyncio event loop. "
            "Use the corresponding async method instead."
        )
