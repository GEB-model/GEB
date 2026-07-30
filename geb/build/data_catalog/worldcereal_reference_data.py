"""Data adapter for public WorldCereal harmonized reference observations."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin

import aiohttp
import geopandas as gpd
import pandas as pd
from aiohttp_retry import ExponentialRetry, RetryClient
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .base import Adapter


DEFAULT_BASE_URL = "https://ewoc-rdm-api.iiasa.ac.at"
DEFAULT_LEGEND_URL = (
    "https://artifactory.vgt.vito.be/artifactory/auxdata-public/"
    "worldcereal/legend/WorldCereal_LC_CT_legend_latest.csv"
)
COLLECTIONS_ENDPOINT = "collections"
COLLECTION_SEARCH_ENDPOINT = "collections/search"
DEFAULT_PUBLIC_STORAGE_URL = "https://ewocstorage.blob.core.windows.net/collections"
DEFAULT_FEATURES_BASE_URL = "https://rdm.esa-worldcereal.org"
DOWNLOAD_CHUNK_SIZE_BYTES = 8 * 1024 * 1024
REQUIRED_REFERENCE_COLUMNS = ("sample_id", "ewoc_code", "valid_time")


class WorldCerealReferenceData(Adapter):
    """Discover and retrieve public WorldCereal Reference Data Module datasets.

    The RDM exposes harmonized public point and polygon observations through a
    REST API. This adapter caches collection metadata, query-specific GeoParquet
    extracts and the current WorldCereal hierarchical legend below
    :attr:`Adapter.root`.

    Notes:
        Dataset licenses differ between RDM collections. The adapter preserves
        collection metadata and adds ``source_collection_id`` to observations;
        callers remain responsible for applying the license and citation terms
        of every selected collection.
    """

    def __init__(
        self,
        *args: Any,
        max_parallel_downloads: int = 4,
        page_size: int = 10_000,
        **kwargs: Any,
    ) -> None:
        """Initialize the WorldCereal RDM adapter.

        Args:
            *args: Positional arguments forwarded to :class:`Adapter`.
            max_parallel_downloads: Maximum simultaneous collection requests.
            page_size: Requested API page size when pagination is supported.
            **kwargs: Keyword arguments forwarded to :class:`Adapter`.
        """
        super().__init__(*args, **kwargs)

        if max_parallel_downloads < 1:
            raise ValueError("max_parallel_downloads must be at least 1.")
        if page_size < 1:
            raise ValueError("page_size must be at least 1.")

        # All cache paths are derived from Adapter.root, which in turn follows
        # GEB_DATA_ROOT and the catalog folder/local_version/cache settings.
        dataset_root = Path(self.root)
        self.metadata_root = dataset_root / "metadata"
        self.collection_root = dataset_root / "collections"
        self.download_root = dataset_root / "downloads"
        self.legend_root = dataset_root / "legend"
        for directory in (
            self.metadata_root,
            self.collection_root,
            self.download_root,
            self.legend_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.max_parallel_downloads = int(max_parallel_downloads)
        self.page_size = int(page_size)
        self.url = DEFAULT_BASE_URL
        self.legend_url = DEFAULT_LEGEND_URL
        self.public_storage_url = DEFAULT_PUBLIC_STORAGE_URL
        self.features_base_url = DEFAULT_FEATURES_BASE_URL

    def fetch(self, url: str | None) -> WorldCerealReferenceData:
        """Set the RDM API base URL and return this adapter."""
        self.url = (url or DEFAULT_BASE_URL).rstrip("/")
        return self

    @property
    def collections_url(self) -> str:
        """Return the public collections endpoint."""
        return f"{self.url}/{COLLECTIONS_ENDPOINT}"

    @property
    def collection_search_url(self) -> str:
        """Return the public collection-search endpoint."""
        return f"{self.url}/{COLLECTION_SEARCH_ENDPOINT}"

    @property
    def feature_collections_url(self) -> str:
        """Return the public feature-service collections endpoint."""
        return f"{self.features_base_url.rstrip('/')}/collections"

    @property
    def collections_cache_path(self) -> Path:
        """Return the cached raw collection listing."""
        return self.metadata_root / "collections.json"

    @property
    def legend_cache_path(self) -> Path:
        """Return the cached WorldCereal legend CSV."""
        return self.legend_root / "WorldCereal_LC_CT_legend_latest.csv"

    @staticmethod
    def _retry_options() -> ExponentialRetry:
        """Return common retry settings for RDM requests."""
        return ExponentialRetry(
            attempts=8,
            start_timeout=5,
            max_timeout=120,
            factor=2,
            retry_all_server_errors=True,
        )

    @staticmethod
    def _timeout() -> aiohttp.ClientTimeout:
        """Return timeouts suitable for large vector responses."""
        return aiohttp.ClientTimeout(
            total=None,
            sock_connect=60,
            sock_read=1800,
        )

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        """Write UTF-8 text atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _atomic_write_geoparquet(data: gpd.GeoDataFrame, path: Path) -> None:
        """Write a GeoParquet file atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.unlink(missing_ok=True)
        data.to_parquet(temporary, index=False)
        temporary.replace(path)

    @staticmethod
    def _normalize_bounds(
        bounds: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float] | None:
        """Validate optional WGS84 bounds."""
        if bounds is None:
            return None
        if len(bounds) != 4:
            raise ValueError("bounds must be (min_lon, min_lat, max_lon, max_lat).")
        min_lon, min_lat, max_lon, max_lat = map(float, bounds)
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("bounds must have increasing coordinates.")
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("Longitude bounds must be between -180 and 180.")
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError("Latitude bounds must be between -90 and 90.")
        return min_lon, min_lat, max_lon, max_lat

    @staticmethod
    def _normalize_geometry(
        geometry: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | None,
    ) -> BaseGeometry | None:
        """Return one valid WGS84 query geometry."""
        if geometry is None:
            return None
        if isinstance(geometry, gpd.GeoDataFrame):
            if geometry.crs is None:
                raise ValueError("Query GeoDataFrame must define a CRS.")
            series = geometry.to_crs(4326).geometry
            result = unary_union(series.dropna().tolist())
        elif isinstance(geometry, gpd.GeoSeries):
            if geometry.crs is None:
                raise ValueError("Query GeoSeries must define a CRS.")
            result = unary_union(geometry.to_crs(4326).dropna().tolist())
        elif isinstance(geometry, BaseGeometry):
            result = geometry
        else:
            raise TypeError(
                "geometry must be a Shapely geometry, GeoSeries, GeoDataFrame or None."
            )

        if result.is_empty:
            raise ValueError("Query geometry is empty.")
        if not result.is_valid:
            result = result.buffer(0)
        if result.is_empty:
            raise ValueError("Query geometry could not be repaired.")
        return result

    @staticmethod
    def _collection_records(payload: Any) -> list[dict[str, Any]]:
        """Extract collection records from common API response shapes."""
        if isinstance(payload, list):
            return [record for record in payload if isinstance(record, dict)]
        if not isinstance(payload, dict):
            raise TypeError(
                "WorldCereal collection response must be a JSON object or list."
            )

        for key in ("collections", "items", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
            if isinstance(value, dict):
                try:
                    return WorldCerealReferenceData._collection_records(value)
                except TypeError, ValueError:
                    pass

        if payload.get("type") == "FeatureCollection" and isinstance(
            payload.get("features"), list
        ):
            return [
                record for record in payload["features"] if isinstance(record, dict)
            ]

        if any(key in payload for key in ("id", "collection_id", "name")):
            return [payload]
        raise ValueError(
            "Could not identify collection records in the WorldCereal response."
        )

    @staticmethod
    def _collection_id_column(table: pd.DataFrame) -> str:
        """Return the column containing collection identifiers."""
        # The public listing exposes both a database UUID in ``id`` and the
        # stable dataset identifier in ``collectionId``. Only ``collectionId``
        # is valid in public download, GeoParquet and feature-service URLs.
        preferred = (
            "collectionId",
            "properties.collectionId",
            "collection_id",
            "properties.collection_id",
            "name",
            "properties.name",
            "id",
            "properties.id",
        )
        for column in preferred:
            if column in table.columns:
                return column
        raise KeyError(
            "Could not identify a WorldCereal collection ID column. "
            f"Available columns: {list(table.columns)}"
        )

    async def list_collections_async(
        self,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return metadata for all public RDM collections."""
        if (
            self.collections_cache_path.exists()
            and self.collections_cache_path.stat().st_size > 0
            and not refresh
        ):
            payload = json.loads(
                self.collections_cache_path.read_text(encoding="utf-8")
            )
        else:
            async with RetryClient(
                retry_options=self._retry_options(),
                timeout=self._timeout(),
            ) as client:
                # The official WorldCereal client discovers collections through
                # /collections/search using four repeated Bbox parameters. Keep
                # /collections as a compatibility fallback for older deployments.
                search_params = [
                    ("Bbox", "-180"),
                    ("Bbox", "-90"),
                    ("Bbox", "180"),
                    ("Bbox", "90"),
                ]
                async with client.get(
                    self.collection_search_url,
                    params=search_params,
                    headers={"Accept": "application/json"},
                ) as response:
                    if response.status in {404, 405, 501}:
                        async with client.get(
                            self.collections_url,
                            raise_for_status=True,
                            headers={"Accept": "application/json"},
                        ) as fallback_response:
                            payload = await fallback_response.json(content_type=None)
                    else:
                        response.raise_for_status()
                        payload = await response.json(content_type=None)
            self._atomic_write_text(
                self.collections_cache_path,
                json.dumps(payload, indent=2, sort_keys=True),
            )

        records = self._collection_records(payload)
        table = pd.json_normalize(records, sep=".")
        if table.empty:
            return table

        id_column = self._collection_id_column(table)
        table["collection_id"] = table[id_column].astype(str)
        return table.drop_duplicates("collection_id").reset_index(drop=True)

    def list_collections(self, *, refresh: bool = False) -> pd.DataFrame:
        """Synchronously return metadata for all public RDM collections."""
        return self._run_sync(
            self.list_collections_async(refresh=refresh),
            "WorldCerealReferenceData.list_collections",
        )

    @staticmethod
    def _looks_like_database_uuid(value: str) -> bool:
        """Return whether a value resembles an RDM database UUID."""
        return bool(
            re.fullmatch(
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                value.strip(),
            )
        )

    async def _resolve_public_collection_id_async(self, value: str) -> str:
        """Resolve a legacy database UUID to the public ``collectionId``.

        Earlier adapter versions preferred the metadata field ``id`` and thus
        propagated internal database UUIDs into download URLs. The public RDM
        endpoints instead require the human-readable ``collectionId``. Canonical
        identifiers are returned unchanged; UUIDs are resolved from the cached
        public collection listing.
        """
        candidate = str(value).strip()
        if not self._looks_like_database_uuid(candidate):
            return candidate

        collections = await self.list_collections_async(refresh=False)
        uuid_columns = [
            column
            for column in ("id", "properties.id")
            if column in collections.columns
        ]
        for column in uuid_columns:
            match = collections.loc[collections[column].astype(str).eq(candidate)]
            if len(match) == 1:
                resolved = str(match.iloc[0]["collection_id"]).strip()
                if resolved and resolved != candidate:
                    print(
                        "Resolved WorldCereal database UUID "
                        f"{candidate!r} to public collectionId {resolved!r}."
                    )
                    return resolved
            if len(match) > 1:
                raise ValueError(
                    "WorldCereal collection UUID maps to multiple public "
                    f"collectionId values: {candidate!r}."
                )

        raise KeyError(
            "WorldCereal database UUID could not be resolved to a public "
            f"collectionId: {candidate!r}. Refresh the collection metadata."
        )

    def search_collections(
        self,
        *,
        keywords: str | Sequence[str] | None = None,
        years: int | Sequence[int] | None = None,
        require_crop_type: bool = True,
        geometry_types: str | Sequence[str] | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Search public collection metadata using permissive text matching.

        This method is intentionally metadata-schema agnostic because the RDM
        periodically evolves its STAC/collection metadata. Search is performed
        across every serialized metadata field and the standardized collection
        identifier.
        """
        table = self.list_collections(refresh=refresh)
        if table.empty:
            return table

        serialized = table.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        mask = pd.Series(True, index=table.index)

        if keywords is not None:
            values = (keywords,) if isinstance(keywords, str) else tuple(keywords)
            for keyword in values:
                mask &= serialized.str.contains(str(keyword).lower(), regex=False)

        if years is not None:
            values = (years,) if isinstance(years, int) else tuple(years)
            year_mask = pd.Series(False, index=table.index)
            for year in values:
                year_mask |= serialized.str.contains(str(int(year)), regex=False)
            mask &= year_mask

        if geometry_types is not None:
            values = (
                (geometry_types,)
                if isinstance(geometry_types, str)
                else tuple(geometry_types)
            )
            geometry_mask = pd.Series(False, index=table.index)
            for value in values:
                geometry_mask |= serialized.str.contains(
                    str(value).lower(), regex=False
                )
            mask &= geometry_mask

        if require_crop_type:
            crop_terms = ("crop", "_110", "_111", "110", "111")
            crop_mask = pd.Series(False, index=table.index)
            for term in crop_terms:
                crop_mask |= serialized.str.contains(term, regex=False)
            mask &= crop_mask

        return table.loc[mask].reset_index(drop=True)

    @staticmethod
    def _query_cache_key(
        *,
        collection_id: str,
        bounds: tuple[float, float, float, float] | None,
        years: tuple[int, ...] | None,
        use_extract_only: bool,
        min_quality_score_ct: int | None,
    ) -> str:
        """Return a stable cache key for one collection query."""
        payload = {
            "collection_id": collection_id,
            "bounds": bounds,
            "years": years,
            "use_extract_only": use_extract_only,
            "min_quality_score_ct": min_quality_score_ct,
        }
        digest = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return f"{collection_id}_{digest}"

    def _collection_download_cache_path(
        self,
        collection_id: str,
        *,
        use_extract_only: bool,
    ) -> Path:
        """Return the cached full or extraction-subset GeoParquet path."""
        subset_name = "extract" if use_extract_only else "full"
        safe_id = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in collection_id
        )
        return self.download_root / safe_id / f"{subset_name}.parquet"

    @staticmethod
    def _download_url_from_payload(payload: Any) -> str | None:
        """Find a signed GeoParquet URL in a nested API response."""
        preferred_keys = (
            "download_url",
            "downloadUrl",
            "url",
            "href",
            "link",
            "signed_url",
            "signedUrl",
        )
        if isinstance(payload, str):
            value = payload.strip()
            return value if value.startswith(("https://", "http://")) else None
        if isinstance(payload, list):
            for item in payload:
                result = WorldCerealReferenceData._download_url_from_payload(item)
                if result is not None:
                    return result
            return None
        if not isinstance(payload, dict):
            return None

        for key in preferred_keys:
            if key not in payload:
                continue
            result = WorldCerealReferenceData._download_url_from_payload(payload[key])
            if result is not None:
                return result
        for value in payload.values():
            result = WorldCerealReferenceData._download_url_from_payload(value)
            if result is not None:
                return result
        return None

    @staticmethod
    def _looks_like_parquet(
        payload: bytes,
        content_type: str,
    ) -> bool:
        """Return whether an HTTP response contains a Parquet file."""
        return payload[:4] == b"PAR1" or "parquet" in content_type.lower()

    @staticmethod
    def _atomic_write_bytes(path: Path, payload: bytes) -> None:
        """Write binary content atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.unlink(missing_ok=True)
        temporary.write_bytes(payload)
        temporary.replace(path)

    async def _download_public_storage_geoparquet(
        self,
        client: RetryClient,
        collection_id: str,
        cache_path: Path,
    ) -> gpd.GeoDataFrame:
        """Download a public collection directly from WorldCereal storage."""
        safe_collection_id = collection_id.strip().removesuffix(".parquet")
        download_url = (
            f"{self.public_storage_url.rstrip('/')}/{safe_collection_id}.parquet"
        )
        temporary = cache_path.with_suffix(cache_path.suffix + ".part")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.unlink(missing_ok=True)
        try:
            async with client.get(download_url, raise_for_status=True) as response:
                with temporary.open("wb") as file_handle:
                    async for chunk in response.content.iter_chunked(
                        DOWNLOAD_CHUNK_SIZE_BYTES
                    ):
                        file_handle.write(chunk)
            if temporary.stat().st_size == 0:
                raise RuntimeError(
                    f"Downloaded empty WorldCereal collection {collection_id!r}."
                )
            with temporary.open("rb") as file_handle:
                if file_handle.read(4) != b"PAR1":
                    raise RuntimeError(
                        "WorldCereal public-storage download did not return a "
                        f"Parquet file for {collection_id!r}."
                    )
            temporary.replace(cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        return gpd.read_parquet(cache_path)

    async def _download_collection_geoparquet(
        self,
        collection_id: str,
        *,
        use_extract_only: bool,
        refresh: bool,
    ) -> gpd.GeoDataFrame | None:
        """Download one public collection as GeoParquet.

        The RDM download endpoint may return Parquet bytes directly or a JSON
        descriptor containing a signed URL. When that endpoint is temporarily
        unavailable, public collections are retrieved from the documented Azure
        Blob collection path. HTTP 404/405/501 from both routes allows the caller
        to fall back to paginated feature retrieval.
        """
        cache_path = self._collection_download_cache_path(
            collection_id,
            use_extract_only=use_extract_only,
        )
        if cache_path.exists() and cache_path.stat().st_size > 0 and not refresh:
            return gpd.read_parquet(cache_path)

        endpoint = f"{self.collections_url}/{collection_id}/download"
        params = {"subset": str(bool(use_extract_only)).lower()}
        async with RetryClient(
            retry_options=self._retry_options(),
            timeout=self._timeout(),
        ) as client:
            try:
                async with client.get(
                    endpoint,
                    params=params,
                    headers={
                        "Accept": (
                            "application/vnd.apache.parquet, "
                            "application/octet-stream;q=0.9, "
                            "application/json;q=0.8"
                        )
                    },
                ) as response:
                    if response.status in {404, 405, 501}:
                        raise FileNotFoundError(endpoint)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "")
                    payload = await response.read()

                if self._looks_like_parquet(payload, content_type):
                    self._atomic_write_bytes(cache_path, payload)
                    return gpd.read_parquet(cache_path)

                response_payload = json.loads(payload.decode("utf-8"))
                download_url = self._download_url_from_payload(response_payload)
                if download_url is None:
                    raise RuntimeError(
                        "WorldCereal collection download response did not contain "
                        f"a download URL for {collection_id!r}."
                    )

                temporary = cache_path.with_suffix(cache_path.suffix + ".part")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary.unlink(missing_ok=True)
                try:
                    async with client.get(
                        download_url,
                        raise_for_status=True,
                    ) as download_response:
                        with temporary.open("wb") as file_handle:
                            async for chunk in download_response.content.iter_chunked(
                                DOWNLOAD_CHUNK_SIZE_BYTES
                            ):
                                file_handle.write(chunk)
                    if temporary.stat().st_size == 0:
                        raise RuntimeError(
                            f"Downloaded empty WorldCereal collection {collection_id!r}."
                        )
                    with temporary.open("rb") as file_handle:
                        if file_handle.read(4) != b"PAR1":
                            raise RuntimeError(
                                "WorldCereal signed download did not return a Parquet "
                                f"file for {collection_id!r}."
                            )
                    temporary.replace(cache_path)
                finally:
                    temporary.unlink(missing_ok=True)
                return gpd.read_parquet(cache_path)
            except (
                aiohttp.ClientError,
                FileNotFoundError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RuntimeError,
            ) as endpoint_error:
                print(
                    "WorldCereal API download route failed for "
                    f"{collection_id!r}; trying public collection storage. "
                    f"{type(endpoint_error).__name__}: {endpoint_error}"
                )
                try:
                    return await self._download_public_storage_geoparquet(
                        client,
                        collection_id,
                        cache_path,
                    )
                except aiohttp.ClientResponseError as storage_error:
                    if storage_error.status in {404, 405, 501}:
                        return None
                    raise

    @staticmethod
    def _features_from_payload(payload: Any) -> list[dict[str, Any]]:
        """Extract GeoJSON-like features from an API response."""
        if isinstance(payload, list):
            candidates = payload
        elif isinstance(payload, dict):
            for key in ("features", "items", "results", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    candidates = value
                    break
            else:
                if payload.get("type") == "Feature":
                    candidates = [payload]
                else:
                    candidates = []
        else:
            candidates = []

        features: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("type") == "Feature" and "geometry" in candidate:
                features.append(candidate)
                continue
            geometry = candidate.get("geometry") or candidate.get("geom")
            properties = candidate.get("properties")
            if geometry is not None:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": geometry,
                        "properties": (
                            properties
                            if isinstance(properties, dict)
                            else {
                                key: value
                                for key, value in candidate.items()
                                if key not in {"geometry", "geom", "properties"}
                            }
                        ),
                    }
                )
        return features

    @staticmethod
    def _next_link(payload: Any, current_url: str) -> str | None:
        """Return a pagination link from common API response formats."""
        if not isinstance(payload, dict):
            return None
        links = payload.get("links")
        if isinstance(links, list):
            for link in links:
                if not isinstance(link, dict):
                    continue
                if str(link.get("rel", "")).lower() == "next" and link.get("href"):
                    return urljoin(current_url, str(link["href"]))
        for key in ("next", "next_url", "nextUrl", "nextLink"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return urljoin(current_url, value)
        return None

    async def _request_collection_pages(
        self,
        collection_id: str,
        *,
        bounds: tuple[float, float, float, float] | None,
        max_features: int | None,
    ) -> list[dict[str, Any]]:
        """Retrieve one public collection with defensive pagination support."""
        url = f"{self.feature_collections_url}/{collection_id}"
        params: dict[str, str | int] = {
            "limit": self.page_size,
            "offset": 0,
        }
        if bounds is not None:
            params["bbox"] = ",".join(f"{value:.12g}" for value in bounds)

        features: list[dict[str, Any]] = []
        page_number = 0
        async with RetryClient(
            retry_options=self._retry_options(),
            timeout=self._timeout(),
        ) as client:
            while url is not None:
                page_number += 1
                if page_number > 100_000:
                    raise RuntimeError(
                        "WorldCereal API pagination exceeded 100,000 pages."
                    )

                async with client.get(
                    url,
                    params=params,
                    raise_for_status=True,
                    headers={
                        "Accept": (
                            "application/geo+json, application/json;q=0.9, "
                            "application/octet-stream;q=0.1"
                        )
                    },
                ) as response:
                    content_type = response.headers.get("Content-Type", "").lower()
                    payload_bytes = await response.read()

                if payload_bytes[:4] == b"PAR1" or "parquet" in content_type:
                    temporary = (
                        self.collection_root / f".{collection_id}.download.parquet"
                    )
                    temporary.write_bytes(payload_bytes)
                    try:
                        data = gpd.read_parquet(temporary)
                    finally:
                        temporary.unlink(missing_ok=True)
                    return json.loads(data.to_json())["features"]

                payload = json.loads(payload_bytes.decode("utf-8"))
                page_features = self._features_from_payload(payload)
                features.extend(page_features)

                if max_features is not None and len(features) > max_features:
                    raise RuntimeError(
                        f"Collection {collection_id!r} exceeded max_features="
                        f"{max_features:,}. Narrow the bounds, use the RDM extract "
                        "subset, or explicitly raise the safety limit."
                    )

                next_url = self._next_link(payload, url)
                if next_url is not None:
                    url = next_url
                    params = {}
                    continue

                matched = None
                if isinstance(payload, dict):
                    matched = payload.get("numberMatched") or payload.get("total")
                if (
                    matched is not None
                    and len(features) < int(matched)
                    and len(page_features) == self.page_size
                ):
                    params["offset"] = int(params.get("offset", 0)) + self.page_size
                    continue
                break

        return features

    @staticmethod
    def _normalize_reference_data(
        data: gpd.GeoDataFrame,
        *,
        collection_id: str,
    ) -> gpd.GeoDataFrame:
        """Normalize one harmonized RDM collection."""
        if data.empty:
            return gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326")
        if data.crs is None:
            data = data.set_crs(4326)
        elif data.crs.to_epsg() != 4326:
            data = data.to_crs(4326)

        geometry_name = data.geometry.name
        if geometry_name != "geometry":
            data = data.rename_geometry("geometry")

        data = data.loc[data.geometry.notna() & ~data.geometry.is_empty].copy()
        invalid = ~data.geometry.is_valid
        if invalid.any():
            data.loc[invalid, "geometry"] = data.loc[invalid, "geometry"].buffer(0)
            data = data.loc[data.geometry.notna() & ~data.geometry.is_empty].copy()

        for column in REQUIRED_REFERENCE_COLUMNS:
            if column not in data.columns:
                raise ValueError(
                    f"WorldCereal collection {collection_id!r} is missing required "
                    f"harmonized column {column!r}. Available columns: "
                    f"{list(data.columns)}"
                )

        data["sample_id"] = data["sample_id"].astype(str)
        data["ewoc_code"] = pd.to_numeric(data["ewoc_code"], errors="raise").astype(
            "int64"
        )
        data["valid_time"] = pd.to_datetime(data["valid_time"], errors="coerce")
        data["source_collection_id"] = str(collection_id)
        data["reference_source"] = "worldcereal_rdm"
        return data.reset_index(drop=True)

    @staticmethod
    def _filter_reference_data(
        data: gpd.GeoDataFrame,
        *,
        years: tuple[int, ...] | None,
        use_extract_only: bool,
        min_quality_score_ct: int | None,
        query_geometry: BaseGeometry | None,
        bounds: tuple[float, float, float, float] | None,
    ) -> gpd.GeoDataFrame:
        """Apply temporal, quality and exact spatial filters."""
        filtered = data
        if years is not None and not filtered.empty:
            filtered = filtered.loc[filtered["valid_time"].dt.year.isin(years)].copy()
        if use_extract_only and "extract" in filtered.columns:
            extract = pd.to_numeric(
                filtered["extract"],
                errors="coerce",
            ).fillna(0)
            filtered = filtered.loc[extract.gt(0)].copy()
        if min_quality_score_ct is not None and "quality_score_ct" in filtered.columns:
            quality = pd.to_numeric(
                filtered["quality_score_ct"],
                errors="coerce",
            ).fillna(-1)
            filtered = filtered.loc[quality.ge(min_quality_score_ct)].copy()

        clip_geometry = query_geometry
        if clip_geometry is None and bounds is not None:
            clip_geometry = box(*bounds)
        if clip_geometry is not None and not filtered.empty:
            positions = filtered.sindex.query(
                clip_geometry,
                predicate="intersects",
            )
            filtered = filtered.iloc[positions].copy()
            filtered = filtered.loc[filtered.geometry.intersects(clip_geometry)].copy()
        return filtered.reset_index(drop=True)

    async def read_collection_async(
        self,
        collection_id: str,
        *,
        bounds: tuple[float, float, float, float] | None = None,
        geometry: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
        years: int | Sequence[int] | None = None,
        use_extract_only: bool = True,
        min_quality_score_ct: int | None = None,
        refresh: bool = False,
        max_features: int | None = 2_000_000,
    ) -> gpd.GeoDataFrame:
        """Read and filter one public harmonized RDM collection."""
        collection_id = str(collection_id).strip()
        if not collection_id:
            raise ValueError("collection_id cannot be empty.")
        collection_id = await self._resolve_public_collection_id_async(collection_id)
        normalized_bounds = self._normalize_bounds(bounds)
        query_geometry = self._normalize_geometry(geometry)
        if normalized_bounds is None and query_geometry is not None:
            normalized_bounds = tuple(float(value) for value in query_geometry.bounds)

        normalized_years = None
        if years is not None:
            values = (years,) if isinstance(years, int) else tuple(years)
            normalized_years = tuple(sorted({int(value) for value in values}))
            if not normalized_years:
                raise ValueError("years cannot be empty.")

        if min_quality_score_ct is not None and not 0 <= min_quality_score_ct <= 100:
            raise ValueError("min_quality_score_ct must be between 0 and 100.")

        cache_key = self._query_cache_key(
            collection_id=collection_id,
            bounds=normalized_bounds,
            years=normalized_years,
            use_extract_only=use_extract_only,
            min_quality_score_ct=min_quality_score_ct,
        )
        cache_path = self.collection_root / collection_id / f"{cache_key}.parquet"

        if cache_path.exists() and cache_path.stat().st_size > 0 and not refresh:
            data = gpd.read_parquet(cache_path)
        else:
            data = await self._download_collection_geoparquet(
                collection_id,
                use_extract_only=use_extract_only,
                refresh=refresh,
            )
            if data is None:
                features = await self._request_collection_pages(
                    collection_id,
                    bounds=normalized_bounds,
                    max_features=max_features,
                )
                data = gpd.GeoDataFrame.from_features(
                    features,
                    crs="EPSG:4326",
                )
            data = self._normalize_reference_data(
                data,
                collection_id=collection_id,
            )
            data = self._filter_reference_data(
                data,
                years=normalized_years,
                use_extract_only=use_extract_only,
                min_quality_score_ct=min_quality_score_ct,
                query_geometry=query_geometry,
                bounds=normalized_bounds,
            )
            if max_features is not None and len(data) > max_features:
                raise RuntimeError(
                    f"Filtered collection {collection_id!r} contains "
                    f"{len(data):,} rows, exceeding max_features="
                    f"{max_features:,}. Narrow the query or raise the safety "
                    "limit explicitly."
                )
            self._atomic_write_geoparquet(data, cache_path)

        data = self._normalize_reference_data(data, collection_id=collection_id)
        data = self._filter_reference_data(
            data,
            years=normalized_years,
            use_extract_only=use_extract_only,
            min_quality_score_ct=min_quality_score_ct,
            query_geometry=query_geometry,
            bounds=normalized_bounds,
        )
        if max_features is not None and len(data) > max_features:
            raise RuntimeError(
                f"Filtered collection {collection_id!r} contains {len(data):,} "
                f"rows, exceeding max_features={max_features:,}."
            )
        return data

    def read_collection(self, collection_id: str, **kwargs: Any) -> gpd.GeoDataFrame:
        """Synchronously read one public RDM collection."""
        return self._run_sync(
            self.read_collection_async(collection_id, **kwargs),
            "WorldCerealReferenceData.read_collection",
        )

    async def read_async(
        self,
        collection_ids: str | Sequence[str],
        **kwargs: Any,
    ) -> gpd.GeoDataFrame:
        """Read and concatenate multiple public RDM collections."""
        ids = (
            (collection_ids,)
            if isinstance(collection_ids, str)
            else tuple(collection_ids)
        )
        ids = tuple(
            dict.fromkeys(str(value).strip() for value in ids if str(value).strip())
        )
        if not ids:
            raise ValueError("At least one collection ID must be supplied.")

        semaphore = asyncio.Semaphore(self.max_parallel_downloads)

        async def read_one(collection_id: str) -> gpd.GeoDataFrame:
            async with semaphore:
                return await self.read_collection_async(collection_id, **kwargs)

        tables = await asyncio.gather(
            *(read_one(collection_id) for collection_id in ids)
        )
        non_empty = [table for table in tables if not table.empty]
        if not non_empty:
            return gpd.GeoDataFrame(
                columns=[*REQUIRED_REFERENCE_COLUMNS, "geometry"],
                geometry="geometry",
                crs="EPSG:4326",
            )

        combined = gpd.GeoDataFrame(
            pd.concat(non_empty, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )
        return combined.drop_duplicates(
            ["source_collection_id", "sample_id"]
        ).reset_index(drop=True)

    def read(
        self,
        collection_ids: str | Sequence[str],
        **kwargs: Any,
    ) -> gpd.GeoDataFrame:
        """Synchronously read and concatenate public RDM collections."""
        return self._run_sync(
            self.read_async(collection_ids, **kwargs),
            "WorldCerealReferenceData.read",
        )

    @staticmethod
    def _read_legend_csv(path: Path) -> pd.DataFrame:
        """Read a cached WorldCereal legend using its published CSV dialect.

        The official land-cover/crop-type legend is semicolon-delimited even
        though its filename ends in ``.csv``. Definitions and labels can contain
        commas, so parsing it with pandas' comma default raises a ``ParserError``.
        A small fallback list is retained for compatibility with older or custom
        legend exports, but semicolon is always attempted first.
        """
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"WorldCereal legend is missing or empty: {path}")

        parse_errors: list[str] = []
        for separator in (";", ",", "\t"):
            try:
                legend = pd.read_csv(
                    path,
                    header=0,
                    sep=separator,
                    encoding="utf-8-sig",
                )
            except (pd.errors.ParserError, UnicodeDecodeError) as error:
                parse_errors.append(f"sep={separator!r}: {error}")
                continue

            legend.columns = [str(column).strip() for column in legend.columns]
            normalized_columns = {column.casefold() for column in legend.columns}
            code_columns = {
                "ewoc_code",
                "code",
                "ct_code",
                "crop_type_code",
            }
            if len(legend.columns) > 1 and normalized_columns.intersection(
                code_columns
            ):
                return legend

            parse_errors.append(
                f"sep={separator!r}: parsed columns {list(legend.columns)!r}, "
                "but no WorldCereal code column was found"
            )

        raise ValueError(
            "Could not parse the WorldCereal legend as a supported delimited "
            f"table: {path}. Attempts: {'; '.join(parse_errors)}"
        )

    async def read_legend_async(self, *, refresh: bool = False) -> pd.DataFrame:
        """Download and return the current WorldCereal hierarchical legend."""
        path = self.legend_cache_path
        if path.exists() and path.stat().st_size > 0 and not refresh:
            return self._read_legend_csv(path)

        temporary = path.with_suffix(path.suffix + ".part")
        temporary.unlink(missing_ok=True)
        async with RetryClient(
            retry_options=self._retry_options(),
            timeout=self._timeout(),
        ) as client:
            async with client.get(self.legend_url, raise_for_status=True) as response:
                with temporary.open("wb") as file_handle:
                    async for chunk in response.content.iter_chunked(
                        DOWNLOAD_CHUNK_SIZE_BYTES
                    ):
                        file_handle.write(chunk)
        temporary.replace(path)
        return self._read_legend_csv(path)

    def read_legend(self, *, refresh: bool = False) -> pd.DataFrame:
        """Synchronously return the WorldCereal hierarchical legend."""
        return self._run_sync(
            self.read_legend_async(refresh=refresh),
            "WorldCerealReferenceData.read_legend",
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
