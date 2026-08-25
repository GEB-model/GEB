"""Copernicus Data Space adapter for HRL raster products."""

from __future__ import annotations

import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from typing import Any

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import xarray as xr
from rioxarray.exceptions import NoDataInBounds, OneDimensionalRaster
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .wekeo_copernicus import (
    _CANONICAL_TILE_CRS,
    _REQUEST_CRS,
    _TILE_CRS,
    _is_native_hrl_tile_crs,
    WEkEOCopernicus,
    _TileCacheStatus,
    _TileDownloadStatus,
)

_EEA_TILE_SIZE_M = 100_000
_EEA_TILE_COORD_RE = re.compile(r"_E(?P<e>\d+)N(?P<n>\d+)_03035_")
_YEAR_COMPONENT_RE = re.compile(r"_S(?P<year>\d{4})_")

_CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
_CDSE_S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
_CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
_CDSE_TILE_CODE_RE = re.compile(
    r"(?<![A-Z0-9])(?P<tile>E\d{2,3}N\d{2,3})(?![A-Z0-9])", re.IGNORECASE
)
_CDSE_COLLECTION_TITLE_BY_PRODUCT_CODE = {
    "CTY": "CLMS VLCC Crop Types (CTY) Europe 10m yearly V1 (COG)",
    "CPSCT": "CLMS VLCC Secondary Crops Type (CPSCT) Europe 10m yearly V1 (COG)",
}
_CDSE_COLLECTION_ID_BY_PRODUCT_CODE = {
    "CTY": "clms_vlcc_crop-types_europe_10m_yearly_v1",
    "CPSCT": "clms_vlcc_secondary-crop-types_europe_10m_yearly_v1",
}


@dataclass(frozen=True)
class _CDSEAsset:
    """Remote CDSE data asset corresponding to one canonical HRL tile."""

    tile_id: str
    hrefs: tuple[str, ...]
    asset_key: str
    media_type: str | None = None
    item_id: str | None = None

    @property
    def href(self) -> str:
        """Return the first configured download candidate."""
        return self.hrefs[0]


class CDSEDownloadError(RuntimeError):
    """Raised when a tile cannot be obtained from Copernicus Data Space."""


class CDSENoCoverageError(FileNotFoundError):
    """Raised when CDSE STAC returns no matching HRL tiles."""


class CDSEAuthenticationError(CDSEDownloadError):
    """Raised when CDSE download credentials are absent or rejected."""


class CopernicusDataSpace(WEkEOCopernicus):
    """Local-first HRL adapter using CDSE STAC/S3 for remote acquisition.

    This class keeps the public ``fetch``/``read`` contract and the on-disk cache
    layout of :class:`WEkEOCopernicus`. Existing ZIP and TIFF files therefore remain
    directly usable. The acquisition order is:

    1. Identify expected 100 km EEA tiles from existing HRL filenames in the local
       catalogue and use the requested-year ZIP/TIFF files when they are complete.
    2. If files are missing (or no local reference catalogue exists), discover the
       authoritative tiles through the Copernicus Data Space STAC catalogue.
    3. Download missing COG assets from CDSE object storage and save them under the
       legacy HRL filename, as ``.tif`` files in ``root/<year>``.
    4. If CDSE cannot satisfy the request, fail with the authoritative CDSE
       diagnostic. Legacy WEkEO/HDA fallback is opt-in only.

    Existing adapter configuration remains valid. In particular, ``dataset_id`` and
    ``default_query`` are still accepted and are used only by the optional WEkEO
    fallback. For CTY and CPSCT, stable CDSE collection IDs are bundled and no
    fragile free-text collection search is needed.
    """

    def __init__(
        self,
        *args: Any,
        dataset_id: str = "EO:EEA:DAT:HRL:CRL",
        default_query: dict[str, Any] | None = None,
        product_code: str | None = None,
        cdse_collection_id: str | None = None,
        cdse_collection_title: str | None = None,
        cdse_stac_url: str = _CDSE_STAC_URL,
        cdse_s3_endpoint: str = _CDSE_S3_ENDPOINT,
        cdse_s3_access_key: str | None = None,
        cdse_s3_secret_key: str | None = None,
        cdse_s3_session_token: str | None = None,
        cdse_access_token: str | None = None,
        cdse_username: str | None = None,
        cdse_password: str | None = None,
        cdse_totp: str | None = None,
        cdse_token_url: str = _CDSE_TOKEN_URL,
        cdse_request_timeout_seconds: float = 60.0,
        prefer_local_tiles: bool = True,
        allow_wekeo_fallback: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the local/CDSE HRL adapter.

        Args:
            *args: Additional positional arguments passed to the base adapter.
            dataset_id: Legacy WEkEO dataset ID. Kept for backwards compatibility
                and used only if ``allow_wekeo_fallback`` is True.
            default_query: Legacy WEkEO query parameters. Kept unchanged for the
                optional fallback.
            product_code: HRL product code, e.g. ``CTY`` or ``CPSCT``.
            cdse_collection_id: Optional explicit CDSE STAC collection ID. Usually
                unnecessary for CTY/CPSCT because their stable IDs are bundled.
            cdse_collection_title: Optional exact STAC collection title override.
            cdse_stac_url: CDSE STAC API root.
            cdse_s3_endpoint: CDSE object-storage endpoint.
            cdse_s3_access_key: Optional S3 access key. If omitted, the adapter uses
                ``CDSE_S3_ACCESS_KEY`` and then the normal boto3/AWS credential chain.
            cdse_s3_secret_key: Optional S3 secret key. If omitted, the adapter uses
                ``CDSE_S3_SECRET_KEY`` and then the normal boto3/AWS credential chain.
            cdse_s3_session_token: Optional S3 session token.
            cdse_access_token: Optional CDSE bearer token for HTTPS assets. S3 is
                used only when explicit S3 keys are configured; otherwise the
                authenticated HTTPS alternate advertised by STAC is preferred.
            cdse_username: Optional CDSE account name used to obtain an OAuth token.
                Falls back to ``CDSE_USERNAME``.
            cdse_password: Optional CDSE password used to obtain an OAuth token.
                Falls back to ``CDSE_PASSWORD`` and is never logged.
            cdse_totp: Optional two-factor code used during token generation. Falls
                back to ``CDSE_TOTP``.
            cdse_token_url: CDSE OpenID Connect token endpoint.
            cdse_request_timeout_seconds: Timeout for STAC/HTTPS requests.
            prefer_local_tiles: Use existing HRL filenames as the first source of tile
                discovery. This allows manually staged years to work without any API.
            allow_wekeo_fallback: Use the separate legacy WEkEO/HDA workflow if
                CDSE cannot satisfy the request. Disabled by default because the
                historical HRL HDA dataset identifier is no longer a reliable
                standard acquisition route.
            **kwargs: Existing ``WEkEOCopernicus`` options, including download retry,
                parallelism, and nodata settings.
        """
        super().__init__(
            *args,
            dataset_id=dataset_id,
            default_query=default_query,
            product_code=product_code,
            prefer_local_tiles=prefer_local_tiles,
            allow_wekeo_fallback=allow_wekeo_fallback,
            **kwargs,
        )
        self.cdse_collection_id = cdse_collection_id
        self.cdse_collection_title = cdse_collection_title
        self.cdse_stac_url = cdse_stac_url.rstrip("/")
        self.cdse_s3_endpoint = cdse_s3_endpoint.rstrip("/")
        self.cdse_s3_access_key = cdse_s3_access_key
        self.cdse_s3_secret_key = cdse_s3_secret_key
        self.cdse_s3_session_token = cdse_s3_session_token
        self.cdse_access_token = cdse_access_token
        self.cdse_username = cdse_username
        self.cdse_password = cdse_password
        self.cdse_totp = cdse_totp
        self.cdse_token_url = cdse_token_url
        self.cdse_request_timeout_seconds = max(
            1.0, float(cdse_request_timeout_seconds)
        )
        self._resolved_cdse_collection_id: str | None = (
            cdse_collection_id
            or _CDSE_COLLECTION_ID_BY_PRODUCT_CODE.get(self.product_code or "")
        )
        self._cdse_s3_client: Any | None = None
        self._resolved_cdse_access_token: str | None = None
        self._cdse_token_lock = Lock()

    def _tile_grid_coordinates(self, tile_id: str) -> tuple[int, int] | None:
        """Return the EEA 100 km grid indices encoded in a HRL tile identifier."""
        match = _EEA_TILE_COORD_RE.search(tile_id)
        if match is None:
            return None
        return int(match.group("e")), int(match.group("n"))

    def _tile_intersects_projected_bounds(
        self,
        tile_id: str,
        projected_bounds: tuple[float, float, float, float],
    ) -> bool:
        """Check whether a filename-derived EEA 100 km tile intersects bounds."""
        coordinates = self._tile_grid_coordinates(tile_id)
        if coordinates is None:
            return False

        e_index, n_index = coordinates
        tile_min_x = e_index * _EEA_TILE_SIZE_M
        tile_min_y = n_index * _EEA_TILE_SIZE_M
        tile_max_x = tile_min_x + _EEA_TILE_SIZE_M
        tile_max_y = tile_min_y + _EEA_TILE_SIZE_M
        min_x, min_y, max_x, max_y = projected_bounds

        return not (
            tile_max_x <= min_x
            or tile_min_x >= max_x
            or tile_max_y <= min_y
            or tile_min_y >= max_y
        )

    def _project_bounds_to_tile_crs(
        self,
        bounds: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Project WGS84 request bounds to the native EEA tile CRS."""
        projected = (
            gpd.GeoSeries([box(*bounds)], crs=_REQUEST_CRS).to_crs(_TILE_CRS).iloc[0]
        )
        return tuple(float(value) for value in projected.bounds)

    def _scan_local_tile_ids(self, year: str | int) -> list[str]:
        """Return cached local tile identifiers using the shared adapter cache."""
        return super()._scan_local_tile_ids(year)

    def _local_reference_years(self, year: str | int) -> list[str]:
        """Return cached local reference years using the shared adapter cache."""
        return super()._local_reference_years(year)

    def _replace_tile_year(self, tile_id: str, year: str | int) -> str:
        """Replace only the SYYYY filename component of a legacy HRL tile ID."""
        return _YEAR_COMPONENT_RE.sub(f"_S{year}_", tile_id, count=1)

    def _discover_local_tiles_for_bounds(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
    ) -> list[str]:
        """Return only *existing target-year* local tiles intersecting the request.

        A different year's filename catalogue is deliberately not used to invent
        required tile IDs. Annual HRL coverage can differ at the EEA38 edge and a tile
        present in 2017 is not proof that an identically named 2018 asset exists in the
        authoritative CDSE collection. Missing coverage is resolved by a CDSE STAC
        query for the requested year instead.
        """
        projected_bounds = self._project_bounds_to_tile_crs(bounds)
        local_matches = [
            tile_id
            for tile_id in self._scan_local_tile_ids(year)
            if self._tile_intersects_projected_bounds(tile_id, projected_bounds)
        ]
        if local_matches:
            self.logger.debug(
                "Identified %s existing %s tile(s) for year %s directly from the "
                "requested-year local filename catalogue.",
                len(local_matches),
                self.product_code or "HRL",
                year,
            )
        return sorted(set(local_matches))

    def _required_grid_coordinates_for_bounds(
        self,
        bounds: tuple[float, float, float, float],
    ) -> set[tuple[int, int]]:
        """Return EEA 100 km grid cells geometrically touched by a WGS84 request."""
        min_x, min_y, max_x, max_y = self._project_bounds_to_tile_crs(bounds)
        # A request whose maximum lies exactly on a tile edge should not require the
        # neighbouring cell on the far side of that edge.
        max_x_inside = float(np.nextafter(max_x, -np.inf))
        max_y_inside = float(np.nextafter(max_y, -np.inf))
        e_min = int(np.floor(min_x / _EEA_TILE_SIZE_M))
        e_max = int(np.floor(max_x_inside / _EEA_TILE_SIZE_M))
        n_min = int(np.floor(min_y / _EEA_TILE_SIZE_M))
        n_max = int(np.floor(max_y_inside / _EEA_TILE_SIZE_M))
        return {
            (e_index, n_index)
            for e_index in range(e_min, e_max + 1)
            for n_index in range(n_min, n_max + 1)
        }

    def _local_tiles_cover_bounds(
        self,
        tile_ids: list[str],
        bounds: tuple[float, float, float, float],
    ) -> bool:
        """Return whether existing target-year tiles cover every touched grid cell."""
        required = self._required_grid_coordinates_for_bounds(bounds)
        available = {
            coordinates
            for tile_id in tile_ids
            if (coordinates := self._tile_grid_coordinates(tile_id)) is not None
        }
        return bool(required) and required.issubset(available)

    def _requests(self) -> Any:
        """Import requests lazily so local-only use adds no hard dependency."""
        try:
            import requests
        except ImportError as error:
            raise ImportError(
                "CopernicusDataSpace requires the 'requests' package when CDSE "
                "catalogue access is needed. Existing fully cached years can still "
                "be read without it."
            ) from error
        return requests

    @staticmethod
    def _retry_after_seconds(
        value: object,
        *,
        now_timestamp: float | None = None,
    ) -> float | None:
        """Parse an HTTP ``Retry-After`` value into a non-negative delay.

        ``Retry-After`` may be either a number of seconds or an HTTP-date. Invalid
        values are ignored so that the normal rate-limit backoff remains available.
        """
        if value is None:
            return None

        text = str(value).strip()
        if not text:
            return None

        try:
            return max(0.0, float(text))
        except TypeError, ValueError:
            pass

        try:
            retry_at = parsedate_to_datetime(text)
        except TypeError, ValueError, OverflowError:
            return None
        if retry_at is None:
            return None
        if retry_at.tzinfo is None:
            # RFC 7231 dates are GMT. Be permissive if a server omits the zone.
            from datetime import timezone

            retry_at = retry_at.replace(tzinfo=timezone.utc)

        now = time.time() if now_timestamp is None else float(now_timestamp)
        return max(0.0, retry_at.timestamp() - now)

    def _stac_retry_policy(
        self,
        error: BaseException,
        *,
        attempt: int,
    ) -> tuple[bool, float, int | None]:
        """Return whether a failed STAC request is retryable and its delay.

        Transport failures, HTTP 429, and HTTP 5xx are retryable. Other explicit
        HTTP 4xx responses are treated as deterministic request failures. For 429,
        a real exponential delay is enforced even when generic backoff is zero and
        a server-supplied ``Retry-After`` value takes precedence when longer.
        """
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if not isinstance(status_code, int):
            status_code = None

        retryable = (
            status_code is None or status_code == 429 or 500 <= status_code < 600
        )
        if not retryable:
            return False, 0.0, status_code

        delay = max(0.0, float(self.download_backoff_seconds)) * attempt
        if status_code == 429:
            headers = getattr(response, "headers", None)
            retry_after = None
            if headers is not None:
                retry_after = self._retry_after_seconds(headers.get("Retry-After"))
            rate_limit_floor = 5.0 * (2 ** (attempt - 1))
            delay = max(delay, rate_limit_floor, retry_after or 0.0)

        return True, delay, status_code

    def _stac_request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue a retrying CDSE STAC request and return its JSON object."""
        requests = self._requests()
        max_attempts = self.download_retries + 1
        last_error: BaseException | None = None
        attempts_used = 0

        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    timeout=self.cdse_request_timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise CDSEDownloadError(
                        f"CDSE STAC returned a non-object JSON response from {url}."
                    )
                return payload
            except Exception as error:
                last_error = error
                retryable, delay, status_code = self._stac_retry_policy(
                    error,
                    attempt=attempt,
                )
                if not retryable or attempt >= max_attempts:
                    break

                status_text = (
                    f"HTTP {status_code}"
                    if status_code is not None
                    else "transport error"
                )
                self.logger.warning(
                    "CDSE STAC request failed (%s; attempt %s/%s): %s. "
                    "Retrying after %.1f s.",
                    status_text,
                    attempt,
                    max_attempts,
                    error,
                    delay,
                )
                time.sleep(delay)

        assert last_error is not None
        raise CDSEDownloadError(
            f"CDSE STAC request failed after {attempts_used} attempt(s): "
            f"{url}: {last_error}"
        ) from last_error

    def _expected_cdse_collection_title(self) -> str | None:
        """Return the expected official STAC collection title when known."""
        if self.cdse_collection_title is not None:
            return self.cdse_collection_title
        if self.product_code is None:
            return None
        return _CDSE_COLLECTION_TITLE_BY_PRODUCT_CODE.get(self.product_code)

    def _resolve_cdse_collection_id(self) -> str:
        """Resolve the asset-level CDSE STAC collection for this HRL product."""
        if self._resolved_cdse_collection_id is not None:
            return self._resolved_cdse_collection_id

        if self.product_code is None and self.cdse_collection_title is None:
            raise ValueError(
                "A product_code, cdse_collection_title, or cdse_collection_id is "
                "required for CDSE STAC discovery."
            )

        expected_title = self._expected_cdse_collection_title()
        search_term = expected_title or self.product_code or self.cdse_collection_title
        assert search_term is not None

        # The CDSE /collections endpoint implements free-text collection search.
        payload = self._stac_request_json(
            "GET",
            f"{self.cdse_stac_url}/collections",
            params={"q": search_term},
        )
        collections = payload.get("collections", [])
        if not isinstance(collections, list):
            collections = []

        if expected_title is not None:
            exact = [
                collection
                for collection in collections
                if str(collection.get("title", "")).casefold()
                == expected_title.casefold()
            ]
            if len(exact) == 1:
                collection_id = str(exact[0]["id"])
                self._resolved_cdse_collection_id = collection_id
                return collection_id

        product_code_token = (
            f"({self.product_code})" if self.product_code is not None else None
        )
        candidates: list[dict[str, Any]] = []
        for collection in collections:
            title = str(collection.get("title", ""))
            if (
                product_code_token is not None
                and product_code_token not in title.upper()
            ):
                continue
            if "COG" not in title.upper():
                continue
            candidates.append(collection)

        if len(candidates) == 1:
            collection_id = str(candidates[0]["id"])
            self._resolved_cdse_collection_id = collection_id
            return collection_id

        candidate_titles = [
            str(item.get("title", item.get("id", "?"))) for item in collections
        ]
        raise CDSENoCoverageError(
            "Could not unambiguously resolve the CDSE STAC collection for "
            f"product_code={self.product_code!r}. Expected title={expected_title!r}. "
            f"Collections returned by free-text search: {candidate_titles}. "
            "Set cdse_collection_id explicitly if the CDSE collection naming changes."
        )

    def _canonical_tile_id(self, tile_code: str, year: str | int) -> str:
        """Build the legacy HRL tile identifier used by the existing cache."""
        if self.product_code is None:
            raise ValueError("product_code is required to construct HRL tile names.")
        return (
            f"CLMS_HRLVLCC_{self.product_code}_S{year}_R10m_"
            f"{tile_code.upper()}_03035_V01_R00"
        )

    def _feature_tile_code(self, feature: dict[str, Any]) -> str | None:
        """Extract an EEA grid code such as E73N22 from a STAC item."""
        strings: list[str] = [str(feature.get("id", ""))]
        properties = feature.get("properties", {})
        if isinstance(properties, dict):
            strings.extend(
                str(value) for value in properties.values() if isinstance(value, str)
            )

        assets = feature.get("assets", {})
        if isinstance(assets, dict):
            for key, asset in assets.items():
                strings.append(str(key))
                if not isinstance(asset, dict):
                    continue
                for field in ("href", "title"):
                    value = asset.get(field)
                    if isinstance(value, str):
                        strings.append(value)
                alternates = asset.get("alternate", {})
                if isinstance(alternates, dict):
                    for alternate in alternates.values():
                        if isinstance(alternate, dict) and isinstance(
                            alternate.get("href"), str
                        ):
                            strings.append(str(alternate["href"]))

        for text in strings:
            match = _CDSE_TILE_CODE_RE.search(text.upper())
            if match is not None:
                return match.group("tile").upper()

        # Projection metadata is a useful fallback if filenames change but each item
        # remains a 100 km EPSG:3035 tile.
        proj_bbox = (
            properties.get("proj:bbox") if isinstance(properties, dict) else None
        )
        proj_code = (
            properties.get("proj:code") if isinstance(properties, dict) else None
        )
        if (
            isinstance(proj_bbox, (list, tuple))
            and len(proj_bbox) >= 4
            and str(proj_code).upper() == _TILE_CRS
        ):
            try:
                min_x = float(proj_bbox[0])
                min_y = float(proj_bbox[1])
                e_index = round(min_x / _EEA_TILE_SIZE_M)
                n_index = round(min_y / _EEA_TILE_SIZE_M)
                return f"E{e_index:02d}N{n_index:02d}"
            except TypeError, ValueError:
                pass

        return None

    def _has_explicit_s3_credentials(self) -> bool:
        """Return whether a complete pair of CDSE/AWS S3 keys is configured."""
        access_key = (
            self.cdse_s3_access_key
            or os.getenv("CDSE_S3_ACCESS_KEY")
            or os.getenv("AWS_ACCESS_KEY_ID")
        )
        secret_key = (
            self.cdse_s3_secret_key
            or os.getenv("CDSE_S3_SECRET_KEY")
            or os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        return bool(access_key and secret_key)

    def _asset_href_candidates(self, asset: dict[str, Any]) -> list[str]:
        """Return all asset hrefs in the order suitable for configured auth."""
        hrefs: list[str] = []
        href = asset.get("href")
        if isinstance(href, str):
            hrefs.append(href)

        alternates = asset.get("alternate", {})
        if isinstance(alternates, dict):
            for alternate in alternates.values():
                if isinstance(alternate, dict):
                    alternate_href = alternate.get("href")
                    if isinstance(alternate_href, str):
                        hrefs.append(alternate_href)

        prefer_s3 = self._has_explicit_s3_credentials()

        # STAC advertises both an S3 URL and an OAuth-protected HTTPS alternate.
        # Use S3 only when keys are explicitly configured; otherwise HTTPS avoids
        # boto3's misleading "Unable to locate credentials" failure.
        return sorted(
            dict.fromkeys(hrefs),
            key=lambda value: (
                0 if value.lower().startswith("s3://") == prefer_s3 else 1,
                value,
            ),
        )

    def _select_feature_asset(
        self,
        feature: dict[str, Any],
        tile_id: str,
    ) -> _CDSEAsset | None:
        """Choose the product COG asset from a STAC item."""
        assets = feature.get("assets", {})
        if not isinstance(assets, dict):
            return None

        scored: list[tuple[int, str, dict[str, Any]]] = []
        product_code = (self.product_code or "").upper()
        for key, asset in assets.items():
            if not isinstance(asset, dict):
                continue
            key_upper = str(key).upper()
            title_upper = str(asset.get("title", "")).upper()
            media_type = str(asset.get("type", "")).lower()
            roles = [str(role).lower() for role in asset.get("roles", [])]
            hrefs = self._asset_href_candidates(asset)
            href_text = " ".join(hrefs).lower()

            score = 0
            if product_code and key_upper == product_code:
                score += 200
            if product_code and product_code in title_upper:
                score += 120
            if "image/tiff" in media_type or ".tif" in href_text:
                score += 100
            if "data" in roles:
                score += 30
            if key_upper == "PRODUCT":
                score -= 25
            if "thumbnail" in roles or "THUMBNAIL" in key_upper:
                score -= 500
            if not hrefs:
                score -= 1000

            scored.append((score, str(key), asset))

        if not scored:
            return None

        _, asset_key, asset = max(scored, key=lambda item: item[0])
        hrefs = self._asset_href_candidates(asset)
        if not hrefs:
            return None

        return _CDSEAsset(
            tile_id=tile_id,
            hrefs=tuple(hrefs),
            asset_key=asset_key,
            media_type=str(asset.get("type"))
            if asset.get("type") is not None
            else None,
            item_id=str(feature.get("id")) if feature.get("id") is not None else None,
        )

    def _stac_features(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
    ) -> list[dict[str, Any]]:
        """Return all CDSE STAC items for the requested year and WGS84 bounds."""
        collection_id = self._resolve_cdse_collection_id()
        search_url = f"{self.cdse_stac_url}/search"
        params: dict[str, Any] | None = {
            "collections": collection_id,
            "bbox": ",".join(str(float(value)) for value in bounds),
            "datetime": (f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z"),
            "limit": 1000,
        }
        method = "GET"
        json_body: dict[str, Any] | None = None
        features: list[dict[str, Any]] = []
        seen_next_urls: set[str] = set()

        while True:
            payload = self._stac_request_json(
                method,
                search_url,
                params=params,
                json_body=json_body,
            )
            page_features = payload.get("features", [])
            if isinstance(page_features, list):
                features.extend(
                    feature for feature in page_features if isinstance(feature, dict)
                )

            next_link = next(
                (
                    link
                    for link in payload.get("links", [])
                    if isinstance(link, dict) and link.get("rel") == "next"
                ),
                None,
            )
            if next_link is None:
                break

            next_url = str(next_link.get("href", ""))
            if not next_url or next_url in seen_next_urls:
                break
            seen_next_urls.add(next_url)
            search_url = next_url
            method = str(next_link.get("method", "GET")).upper()
            params = None
            body = next_link.get("body")
            json_body = body if method == "POST" and isinstance(body, dict) else None

        return features

    def _search_cdse_assets(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
    ) -> tuple[list[str], dict[str, _CDSEAsset]]:
        """Discover canonical HRL tiles and their downloadable CDSE assets."""
        features = self._stac_features(bounds=bounds, year=year)
        asset_lookup: dict[str, _CDSEAsset] = {}
        skipped_without_grid_code = 0
        skipped_without_data_asset = 0

        for feature in features:
            tile_code = self._feature_tile_code(feature)
            if tile_code is None:
                skipped_without_grid_code += 1
                continue
            tile_id = self._canonical_tile_id(tile_code=tile_code, year=year)
            asset = self._select_feature_asset(feature=feature, tile_id=tile_id)
            if asset is None:
                skipped_without_data_asset += 1
                continue

            previous = asset_lookup.get(tile_id)
            if previous is None:
                asset_lookup[tile_id] = asset
                continue

            # Prefer an asset whose source name explicitly contains the legacy
            # V01_R00 revision used by the historical HRL files.
            previous_text = f"{previous.item_id or ''} {previous.href}".upper()
            new_text = f"{asset.item_id or ''} {asset.href}".upper()
            if "V01_R00" in new_text and "V01_R00" not in previous_text:
                asset_lookup[tile_id] = asset

        if skipped_without_grid_code:
            self.logger.debug(
                "Skipped %s CDSE STAC item(s) because no EEA tile code could be "
                "identified.",
                skipped_without_grid_code,
            )
        if skipped_without_data_asset:
            self.logger.debug(
                "Skipped %s CDSE STAC item(s) because no usable data asset was found.",
                skipped_without_data_asset,
            )

        tile_ids = sorted(asset_lookup)
        if not tile_ids:
            raise CDSENoCoverageError(
                f"CDSE STAC returned no usable {self.product_code or 'HRL'} COG tiles "
                f"for year={year}, bounds={bounds}."
            )

        return tile_ids, asset_lookup

    def _s3_location(self, href: str) -> tuple[str, str] | None:
        """Convert an S3 or CDSE object-storage URL to ``(bucket, key)``."""
        from urllib.parse import urlparse

        parsed = urlparse(href)
        if parsed.scheme.lower() == "s3":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            return (bucket, key) if bucket and key else None

        endpoint_host = urlparse(self.cdse_s3_endpoint).netloc.lower()
        if (
            parsed.scheme.lower() in {"http", "https"}
            and parsed.netloc.lower() == endpoint_host
        ):
            parts = parsed.path.lstrip("/").split("/", 1)
            if len(parts) == 2 and all(parts):
                return parts[0], parts[1]

        if href.startswith("/eodata/") or href.startswith("eodata/"):
            normalized = href.lstrip("/")
            bucket, key = normalized.split("/", 1)
            return bucket, key

        return None

    def _get_cdse_s3_client(self) -> Any:
        """Create a boto3 S3 client for the CDSE ``eodata`` object store."""
        if self._cdse_s3_client is not None:
            return self._cdse_s3_client

        try:
            import boto3
        except ImportError as error:
            raise ImportError(
                "Downloading missing CDSE HRL tiles through S3 requires 'boto3'. "
                "Install boto3 or provide the tiles in the local cache."
            ) from error

        access_key = (
            self.cdse_s3_access_key
            or os.getenv("CDSE_S3_ACCESS_KEY")
            or os.getenv("AWS_ACCESS_KEY_ID")
        )
        secret_key = (
            self.cdse_s3_secret_key
            or os.getenv("CDSE_S3_SECRET_KEY")
            or os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        session_token = (
            self.cdse_s3_session_token
            or os.getenv("CDSE_S3_SESSION_TOKEN")
            or os.getenv("AWS_SESSION_TOKEN")
        )
        if not access_key or not secret_key:
            raise CDSEAuthenticationError(
                "The STAC asset uses CDSE S3, but no complete S3 key pair is "
                "configured. Set CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY, or "
                "configure CDSE_USERNAME/CDSE_PASSWORD so the advertised OAuth "
                "HTTPS alternate can be used."
            )

        client_kwargs: dict[str, Any] = {
            "endpoint_url": self.cdse_s3_endpoint,
            "region_name": "default",
        }
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key
        if session_token is not None:
            client_kwargs["aws_session_token"] = session_token

        self._cdse_s3_client = boto3.client("s3", **client_kwargs)
        return self._cdse_s3_client

    def _can_generate_cdse_access_token(self) -> bool:
        """Return whether username/password token generation is configured."""
        username = self.cdse_username or os.getenv("CDSE_USERNAME")
        password = self.cdse_password or os.getenv("CDSE_PASSWORD")
        return bool(username and password)

    def _get_cdse_access_token(self) -> str:
        """Return a supplied token or obtain one from the official OIDC endpoint."""
        with self._cdse_token_lock:
            if self._resolved_cdse_access_token:
                return self._resolved_cdse_access_token

            supplied_token = self.cdse_access_token or os.getenv("CDSE_ACCESS_TOKEN")
            if supplied_token:
                self._resolved_cdse_access_token = supplied_token
                return supplied_token

            username = self.cdse_username or os.getenv("CDSE_USERNAME")
            password = self.cdse_password or os.getenv("CDSE_PASSWORD")
            if not username or not password:
                raise CDSEAuthenticationError(
                    "CDSE STAC discovery succeeded, but downloading its HTTPS asset "
                    "requires authentication. Set CDSE_ACCESS_TOKEN, or set both "
                    "CDSE_USERNAME and CDSE_PASSWORD. Alternatively configure "
                    "CDSE_S3_ACCESS_KEY/CDSE_S3_SECRET_KEY; the separate WEkEO "
                    "adapter will be tried when fallback is enabled."
                )

            auth_data = {
                "client_id": "cdse-public",
                "grant_type": "password",
                "username": username,
                "password": password,
            }
            totp = self.cdse_totp or os.getenv("CDSE_TOTP")
            if totp:
                auth_data["totp"] = totp

            requests = self._requests()
            try:
                response = requests.post(
                    self.cdse_token_url,
                    data=auth_data,
                    timeout=self.cdse_request_timeout_seconds,
                )
                response.raise_for_status()
                token = response.json().get("access_token")
            except Exception as error:
                raise CDSEAuthenticationError(
                    "Failed to obtain a CDSE access token from the official OIDC "
                    "endpoint. Check CDSE_USERNAME/CDSE_PASSWORD and CDSE_TOTP."
                ) from error
            if not isinstance(token, str) or not token:
                raise CDSEAuthenticationError(
                    "The CDSE OIDC response did not contain an access token."
                )
            self._resolved_cdse_access_token = token
            return token

    def _download_cdse_https(self, href: str, target: Path) -> None:
        """Download an OAuth-protected HTTPS asset advertised by CDSE STAC."""
        requests = self._requests()
        for auth_attempt in range(2):
            token = self._get_cdse_access_token()
            with requests.get(
                href,
                headers={"Authorization": f"Bearer {token}"},
                stream=True,
                timeout=self.cdse_request_timeout_seconds,
            ) as response:
                if (
                    response.status_code == 401
                    and auth_attempt == 0
                    and self._can_generate_cdse_access_token()
                ):
                    with self._cdse_token_lock:
                        self._resolved_cdse_access_token = None
                    continue
                try:
                    response.raise_for_status()
                except Exception as error:
                    if response.status_code in {401, 403}:
                        raise CDSEAuthenticationError(
                            "CDSE rejected the OAuth credentials for the HTTPS asset."
                        ) from error
                    raise
                with target.open("wb") as file_obj:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            file_obj.write(chunk)
                return

        raise CDSEAuthenticationError(
            "CDSE rejected both the initial and refreshed OAuth access token."
        )

    def _download_cdse_href(self, href: str, target: Path) -> None:
        """Download one STAC asset to a temporary local path."""
        s3_location = self._s3_location(href)
        if s3_location is not None:
            bucket, key = s3_location
            client = self._get_cdse_s3_client()
            client.download_file(bucket, key, str(target))
            return

        if href.lower().startswith(("http://", "https://")):
            self._download_cdse_https(href, target)
            return

        raise CDSEDownloadError(f"Unsupported CDSE asset URL: {href}")

    def _download_cdse_asset(self, asset: _CDSEAsset, target: Path) -> str:
        """Try every STAC-advertised URL and return the successful href."""
        failures: list[tuple[str, BaseException]] = []
        for href in asset.hrefs:
            target.unlink(missing_ok=True)
            try:
                self._download_cdse_href(href, target)
                return href
            except Exception as error:
                failures.append((href, error))

        details = "; ".join(f"{href}: {error}" for href, error in failures)
        if failures and all(
            isinstance(error, CDSEAuthenticationError) for _, error in failures
        ):
            raise CDSEAuthenticationError(
                f"No authenticated CDSE download route was available for "
                f"{asset.tile_id}: {details}"
            )
        raise CDSEDownloadError(
            f"All CDSE download routes failed for {asset.tile_id}: {details}"
        )

    def _download_single_cdse_asset(
        self,
        asset: _CDSEAsset,
        year: str | int,
    ) -> _TileDownloadStatus:
        """Download one CDSE asset atomically under the legacy local filename."""
        tif_path = self._tile_tif_path(year, asset.tile_id)
        zip_path = self._tile_zip_path(year, asset.tile_id)
        if tif_path.exists():
            return _TileDownloadStatus(asset.tile_id, "cached_tif", tif_path)
        if zip_path.exists():
            return _TileDownloadStatus(asset.tile_id, "cached_zip", zip_path)

        href_path = asset.href.split("?", 1)[0].lower()
        is_zip = href_path.endswith(".zip") or (
            asset.media_type is not None and "zip" in asset.media_type.lower()
        )
        target_path = zip_path if is_zip else tif_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        attempts = self.download_retries + 1
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            temp_path = target_path.with_name(
                f".{target_path.name}.download-{uuid.uuid4().hex}"
            )
            try:
                successful_href = self._download_cdse_asset(asset, temp_path)
                if not temp_path.exists() or temp_path.stat().st_size == 0:
                    raise CDSEDownloadError(
                        f"CDSE download created an empty file for {asset.tile_id}: "
                        f"{successful_href}"
                    )
                temp_path.replace(target_path)
                return _TileDownloadStatus(
                    asset.tile_id,
                    "downloaded_zip" if is_zip else "downloaded_tif",
                    target_path,
                )
            except Exception as error:
                last_error = error
                temp_path.unlink(missing_ok=True)
                if isinstance(error, CDSEAuthenticationError):
                    break
                if attempt < attempts:
                    delay = self.download_backoff_seconds * attempt
                    self.logger.warning(
                        "CDSE download failed for %s (attempt %s/%s): %s. "
                        "Retrying after %.1f s.",
                        asset.tile_id,
                        attempt,
                        attempts,
                        error,
                        delay,
                    )
                    time.sleep(delay)
            finally:
                temp_path.unlink(missing_ok=True)

        assert last_error is not None
        raise CDSEDownloadError(
            f"Failed to download CDSE tile {asset.tile_id!r} for year {year} "
            f"from advertised routes {asset.hrefs}: {last_error}"
        ) from last_error

    def _download_cdse_tiles(
        self,
        tile_ids: list[str],
        year: str | int,
        asset_lookup: dict[str, _CDSEAsset],
    ) -> None:
        """Download a set of missing CDSE tiles using the existing worker setting."""
        if not tile_ids:
            return

        unavailable = [tile_id for tile_id in tile_ids if tile_id not in asset_lookup]
        if unavailable:
            raise CDSEDownloadError(
                "CDSE STAC did not provide assets for requested tile(s): "
                + ", ".join(unavailable)
            )

        worker_count = min(self.max_parallel_downloads, max(1, len(tile_ids)))
        failures: list[tuple[str, BaseException]] = []

        if worker_count == 1:
            for tile_id in tile_ids:
                try:
                    status = self._download_single_cdse_asset(
                        asset=asset_lookup[tile_id],
                        year=year,
                    )
                    self.logger.debug(
                        "CDSE tile %s ready for year %s: %s",
                        status.tile_id,
                        year,
                        status.path,
                    )
                except Exception as error:
                    failures.append((tile_id, error))
                    break
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        self._download_single_cdse_asset,
                        asset_lookup[tile_id],
                        year,
                    ): tile_id
                    for tile_id in tile_ids
                }
                for future in as_completed(futures):
                    tile_id = futures[future]
                    try:
                        status = future.result()
                    except Exception as error:
                        failures.append((tile_id, error))
                        self.logger.error(
                            "Failed to download CDSE tile %s for year %s: %s",
                            tile_id,
                            year,
                            error,
                        )
                    else:
                        self.logger.debug(
                            "CDSE tile %s ready for year %s: %s",
                            status.tile_id,
                            year,
                            status.path,
                        )

        if failures:
            details = "\n".join(f"- {tile_id}: {error}" for tile_id, error in failures)
            raise CDSEDownloadError(
                f"Failed to download {len(failures)} of {len(tile_ids)} CDSE tile(s) "
                f"for year {year}.\n{details}"
            )

        self._invalidate_local_catalogue_cache(year)

    def _wekeo_fallback_for_missing(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
        required_tile_ids: list[str],
        query_overrides: dict[str, Any] | None,
    ) -> list[str]:
        """Use the parent WEkEO implementation to fill still-missing tiles."""
        wekeo_tile_ids, result_lookup = super()._search_tiles(
            bounds=bounds,
            year=year,
            query_overrides=query_overrides,
        )
        if not required_tile_ids:
            required_tile_ids = list(wekeo_tile_ids)
        else:
            required_tile_ids = sorted(set(required_tile_ids) | set(wekeo_tile_ids))

        cache_status = self._inspect_tile_cache(required_tile_ids, year)
        downloadable = [
            tile_id
            for tile_id in cache_status.missing_tile_ids
            if tile_id in result_lookup
        ]
        if downloadable:
            super().download_tiles(
                tile_ids=downloadable,
                year=year,
                bounds=bounds,
                query_overrides=query_overrides,
                result_lookup=result_lookup,
            )

        return required_tile_ids

    def _merge_tiles(
        self,
        tile_paths: list[Path],
        *,
        chunks: dict[str, int] | None = None,
        clip_bounds: tuple[float, float, float, float] | None = None,
        normalize_nodata: bool = True,
    ) -> xr.DataArray:
        """Lazily mosaic clipped CDSE HRL tiles on the fixed EPSG:3035 grid.

        The lazy path is used only for normal chunked model reads. It deliberately
        keeps the reliable parts of the older WEkEO implementation: each tile is
        validated independently, clipped before mosaicking, and all source CRS
        serializations are canonicalized to the fixed HRL EPSG:3035 grid.

        Unlike the earlier experimental implementation, this method never compares
        raw Rasterio CRS objects for equality. Historical WKT1 and modern EPSG
        serializations of ETRS89 / LAEA Europe are accepted through
        ``_is_native_hrl_tile_crs`` and rewritten to ``_CANONICAL_TILE_CRS`` before
        xarray sees multiple tiles. This removes the observed false mismatch between
        ``EPSG:3035`` and the legacy PROJCS representation.

        Raw/unchunked reads fall back to the mature eager merge implementation.
        """
        if chunks is None or not normalize_nodata or self.destination_nodata is None:
            return super()._merge_tiles(
                tile_paths,
                chunks=chunks,
                clip_bounds=clip_bounds,
                normalize_nodata=normalize_nodata,
            )
        if not tile_paths:
            raise ValueError("No CDSE HRL tile paths were provided for merging.")

        chunk_spec = {
            dim: max(1, int(size)) for dim, size in chunks.items() if dim in {"x", "y"}
        }
        if not chunk_spec:
            return super()._merge_tiles(
                tile_paths,
                chunks=chunks,
                clip_bounds=clip_bounds,
                normalize_nodata=normalize_nodata,
            )

        prepared_tiles: list[xr.DataArray] = []
        opened_sources: list[xr.DataArray] = []
        diagnostics: list[str] = []
        skipped_paths: list[str] = []
        reference_resolution: tuple[float, float] | None = None
        merge_nodata = int(self.destination_nodata)

        def _close_sources() -> None:
            for source in opened_sources:
                try:
                    source.close()
                except Exception:
                    pass

        try:
            for path in tile_paths:
                source_da = rxr.open_rasterio(
                    path,
                    masked=False,
                    cache=False,
                    chunks=chunk_spec,
                )
                source_crs = source_da.rio.crs
                if source_crs is None:
                    source_da.close()
                    raise ValueError(
                        f"CDSE HRL source tile {path} has no CRS in its GeoTIFF metadata."
                    )
                if not _is_native_hrl_tile_crs(source_crs):
                    source_da.close()
                    raise ValueError(
                        f"CDSE HRL source tile {path.name} is not on the expected "
                        f"native ETRS89 / LAEA Europe grid ({_TILE_CRS}). "
                        f"Found CRS: {source_crs}."
                    )

                source_resolution = tuple(
                    float(value) for value in source_da.rio.resolution()
                )
                if reference_resolution is None:
                    reference_resolution = source_resolution
                elif not all(
                    abs(a - b) <= 1e-8
                    for a, b in zip(
                        source_resolution, reference_resolution, strict=True
                    )
                ):
                    source_da.close()
                    raise ValueError(
                        "Cannot lazily mosaic CDSE HRL tiles with different native "
                        f"resolutions. Expected {reference_resolution}; {path.name} "
                        f"has {source_resolution}."
                    )

                da = source_da.sel(band=1, drop=True)
                # First make clipping a pure coordinate/window operation. In
                # particular, do not expose UInt16 65535 as active GDAL nodata.
                da = self._clear_raster_nodata(da)
                da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
                da = da.rio.write_crs(_CANONICAL_TILE_CRS, inplace=False)

                if clip_bounds is not None:
                    min_x, min_y, max_x, max_y = clip_bounds
                    tile_min_x, tile_min_y, tile_max_x, tile_max_y = da.rio.bounds()
                    intersects = not (
                        tile_max_x <= min_x
                        or tile_min_x >= max_x
                        or tile_max_y <= min_y
                        or tile_min_y >= max_y
                    )
                    if not intersects:
                        skipped_paths.append(path.name)
                        source_da.close()
                        continue
                    try:
                        da = da.rio.clip_box(
                            minx=min_x,
                            miny=min_y,
                            maxx=max_x,
                            maxy=max_y,
                            allow_one_dimensional_raster=True,
                        )
                    except NoDataInBounds, OneDimensionalRaster:
                        skipped_paths.append(path.name)
                        source_da.close()
                        continue

                # Promote/normalize only after clipping. This keeps 65535 away from
                # GDAL and avoids building int32 Dask graphs for irrelevant tile area.
                da = self._normalize_categorical_nodata_for_rasterio(da)
                da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
                da = da.rio.write_crs(_CANONICAL_TILE_CRS, inplace=False)
                da = da.rio.write_nodata(merge_nodata, inplace=False)

                if da.rio.crs != _CANONICAL_TILE_CRS:
                    source_da.close()
                    raise RuntimeError(
                        f"Internal CRS canonicalization failed for {path.name}."
                    )
                diagnostics.append(
                    f"{path.name}: shape={da.shape}, bounds={da.rio.bounds()}, "
                    f"resolution={da.rio.resolution()}, source_crs={source_crs}, "
                    f"canonical_crs={da.rio.crs}, nodata={da.rio.nodata}, "
                    f"dtype={da.dtype}, chunks={getattr(da.data, 'chunks', None)}"
                )
                prepared_tiles.append(da)
                opened_sources.append(source_da)

            if not prepared_tiles:
                raise ValueError(
                    "None of the CDSE HRL tiles intersect the requested clip bounds.\n"
                    f"Clip bounds: {clip_bounds}\n"
                    "Tile diagnostics:\n" + "\n".join(diagnostics)
                )

            first_dtype = prepared_tiles[0].dtype
            if any(da.dtype != first_dtype for da in prepared_tiles[1:]):
                raise ValueError(
                    "Cannot lazily mosaic CDSE HRL tiles with different dtypes.\n"
                    + "\n".join(diagnostics)
                )

            if skipped_paths:
                self.logger.debug(
                    "Skipped %s CDSE HRL tile(s) outside the requested clip bounds: %s",
                    len(skipped_paths),
                    skipped_paths,
                )

            if len(prepared_tiles) == 1:
                merged = prepared_tiles[0]
            else:
                try:
                    merged = xr.combine_by_coords(
                        prepared_tiles,
                        join="outer",
                        fill_value=merge_nodata,
                        combine_attrs="override",
                    )
                except Exception as error:
                    raise ValueError(
                        "Failed to lazily mosaic CDSE HRL tiles by canonical "
                        "EPSG:3035 coordinates.\nTile diagnostics:\n"
                        + "\n".join(diagnostics)
                    ) from error

            if "band" in merged.dims:
                merged = merged.sel(band=1, drop=True)
            merged = merged.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
            merged = merged.rio.write_crs(_CANONICAL_TILE_CRS, inplace=False)
            merged = merged.rio.write_nodata(merge_nodata, inplace=False)
            merged = merged.chunk(
                {dim: size for dim, size in chunk_spec.items() if dim in merged.dims}
            )

            # Keep source file managers alive until the returned lazy array is
            # explicitly closed or released. The live tests force .load() after
            # return, which guards against accidentally closing these too early.
            merged.set_close(_close_sources)
            return merged
        except Exception:
            _close_sources()
            raise

    def read(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int | None = None,
        query_overrides: dict[str, Any] | None = None,
        *,
        dst_crs: str | None = _REQUEST_CRS,
        normalize_nodata: bool = True,
        chunks: dict[str, int] | None = None,
    ) -> xr.DataArray:
        """Read HRL through CDSE and guarantee spatial metadata at the public boundary.

        The inherited unpack/clip/reproject machinery remains shared with the legacy
        adapter, but a Dask-backed ``rio.clip_box`` can lose the grid-mapping variable.
        Downstream callers must therefore never depend on incidental metadata survival.
        """
        da = super().read(
            bounds=bounds,
            year=year,
            query_overrides=query_overrides,
            dst_crs=dst_crs,
            normalize_nodata=normalize_nodata,
            chunks=chunks,
        )
        close_callback = getattr(da, "_close", None)
        expected_crs = _TILE_CRS if dst_crs is None else dst_crs
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        da = da.rio.write_crs(expected_crs, inplace=False)
        if normalize_nodata and self.destination_nodata is not None:
            da = da.rio.write_nodata(self.destination_nodata, inplace=False)
            # A normalized HRL categorical raster must no longer be UInt16. If it
            # remained unsigned, a later rioxarray reprojection with no explicit
            # safe destination sentinel could select 65535 as UInt16 nodata and
            # trigger GDAL's 65535 -> 65534 collision-avoidance path.
            if np.issubdtype(np.dtype(da.dtype), np.unsignedinteger):
                if close_callback is not None:
                    close_callback()
                raise RuntimeError(
                    "Normalized Copernicus HRL data unexpectedly retained an "
                    f"unsigned dtype ({da.dtype}); expected a signed dtype with "
                    f"nodata {self.destination_nodata}."
                )
            if da.rio.nodata != self.destination_nodata:
                if close_callback is not None:
                    close_callback()
                raise RuntimeError(
                    "Normalized Copernicus HRL data returned with unexpected "
                    f"nodata {da.rio.nodata}; expected {self.destination_nodata}."
                )
        if da.rio.crs is None:
            if close_callback is not None:
                close_callback()
            raise ValueError(
                "CopernicusDataSpace.read() returned an HRL raster without CRS "
                "after explicit metadata restoration."
            )
        if close_callback is not None:
            da.set_close(close_callback)
        return da

    def fetch(
        self,
        bounds: tuple[float, float, float, float],
        year: str | int,
        url: str | None = None,
        query_overrides: dict[str, Any] | None = None,
    ) -> CopernicusDataSpace:
        """Ensure authoritative target-year HRL tiles are available locally.

        Fast path: if existing files from the requested year geometrically cover the
        full request, use them with no network call. Otherwise query CDSE STAC for the
        *requested year* and treat its returned assets as authoritative. Existing local
        target-year tiles are retained, but a different year's filename catalogue is
        never used to manufacture a required tile ID.
        """
        del url  # The canonical adapter is configured by CDSE collection, not URL.

        local_tile_ids: list[str] = []
        if self.prefer_local_tiles:
            local_tile_ids = self._discover_local_tiles_for_bounds(
                bounds=bounds,
                year=year,
            )
            if local_tile_ids and self._local_tiles_cover_bounds(
                local_tile_ids, bounds
            ):
                cache_status = self._inspect_tile_cache(local_tile_ids, year)
                if cache_status.is_complete:
                    self.logger.info(
                        "Using %s locally cached %s tile(s) for year %s; exact-year "
                        "files cover the requested EEA grid cells, so no CDSE query "
                        "is required.",
                        cache_status.total_tiles,
                        self.product_code or "HRL",
                        year,
                    )
                    self.tile_ids = local_tile_ids
                    self.year = year
                    return self

            if local_tile_ids:
                self.logger.info(
                    "Existing year-%s %s tiles cover only part of the requested EEA "
                    "grid. Querying CDSE STAC for authoritative target-year coverage.",
                    year,
                    self.product_code or "HRL",
                )

        cdse_error: BaseException | None = None
        cdse_tile_ids: list[str] = []
        cdse_assets: dict[str, _CDSEAsset] = {}
        try:
            cdse_tile_ids, cdse_assets = self._search_cdse_assets(
                bounds=bounds,
                year=year,
            )
            required_tile_ids = sorted(set(local_tile_ids) | set(cdse_tile_ids))

            touched_grid_cells = self._required_grid_coordinates_for_bounds(bounds)
            published_grid_cells = {
                coordinates
                for tile_id in required_tile_ids
                if (coordinates := self._tile_grid_coordinates(tile_id)) is not None
            }
            unpublished_touched_cells = sorted(
                touched_grid_cells - published_grid_cells
            )
            if unpublished_touched_cells:
                self.logger.debug(
                    "CDSE publishes no %s tile for %s geometrically touched EEA grid "
                    "cell(s) in year %s: %s. These cells are treated as outside/no "
                    "coverage; no synthetic filename is manufactured.",
                    self.product_code or "HRL",
                    len(unpublished_touched_cells),
                    year,
                    [f"E{e:02d}N{n:02d}" for e, n in unpublished_touched_cells],
                )

            # Only STAC-returned IDs are candidates for CDSE download. A local file
            # that happens not to be present in the current STAC response remains
            # usable, while a cross-year inferred filename can no longer create an
            # impossible download requirement.
            missing_cdse_ids = [
                tile_id
                for tile_id in cdse_tile_ids
                if not self._tile_tif_path(year, tile_id).exists()
                and not self._tile_zip_path(year, tile_id).exists()
            ]
            if missing_cdse_ids:
                self.logger.info(
                    "Downloading %s missing %s tile(s) for year %s from CDSE.",
                    len(missing_cdse_ids),
                    self.product_code or "HRL",
                    year,
                )
                self._download_cdse_tiles(
                    tile_ids=missing_cdse_ids,
                    year=year,
                    asset_lookup=cdse_assets,
                )

            final_status = self._inspect_tile_cache(required_tile_ids, year)
            if not final_status.is_complete:
                raise CDSEDownloadError(
                    "CDSE discovery succeeded but some authoritative target-year "
                    "tiles are still missing after download: "
                    + ", ".join(final_status.missing_tile_ids)
                )

            self.tile_ids = required_tile_ids
            self.year = year
            return self
        except Exception as error:
            cdse_error = error
            self.logger.warning(
                "CDSE could not satisfy %s year %s for bounds %s: %s",
                self.product_code or "HRL",
                year,
                bounds,
                error,
            )

        if self.allow_wekeo_fallback:
            # Kept strictly for explicit legacy opt-in. The historic
            # EO:EEA:DAT:HRL:CRL dataset currently returns HTTP 404 in the user's HDA
            # environment, so the standard CopernicusDataSpace path does not rely on it.
            try:
                required_tile_ids = self._wekeo_fallback_for_missing(
                    bounds=bounds,
                    year=year,
                    required_tile_ids=local_tile_ids,
                    query_overrides=query_overrides,
                )
                final_status = self._inspect_tile_cache(required_tile_ids, year)
                if final_status.is_complete and required_tile_ids:
                    self.tile_ids = required_tile_ids
                    self.year = year
                    return self
            except Exception as wekeo_error:
                raise CDSEDownloadError(
                    "CDSE failed and explicitly enabled legacy WEkEO fallback also "
                    f"failed. CDSE error: {cdse_error}. WEkEO error: {wekeo_error}."
                ) from wekeo_error

        if local_tile_ids:
            # Partial local coverage must not be silently presented as complete when
            # CDSE itself failed; that would turn a data-availability problem into
            # spatially biased crop areas.
            raise CDSEDownloadError(
                "Exact-year local HRL files provide only partial coverage and CDSE "
                f"could not provide authoritative completion: {cdse_error}"
            ) from cdse_error

        if isinstance(cdse_error, CDSENoCoverageError):
            raise cdse_error
        raise CDSEDownloadError(
            "CDSE could not satisfy the HRL tile request. Legacy WEkEO fallback is "
            f"disabled by default. Underlying error: {cdse_error}"
        ) from cdse_error

    def get_tiles_for_mask(
        self,
        mask: BaseGeometry,
        year: str | int,
        query_overrides: dict[str, Any] | None = None,
    ) -> list[str]:
        """Identify target-year tile IDs, using CDSE whenever local coverage is partial."""
        local_tile_ids: list[str] = []
        if self.prefer_local_tiles:
            local_tile_ids = self._discover_local_tiles_for_bounds(
                bounds=mask.bounds,
                year=year,
            )
            if local_tile_ids and self._local_tiles_cover_bounds(
                local_tile_ids,
                mask.bounds,
            ):
                return local_tile_ids

        try:
            cdse_tile_ids, _ = self._search_cdse_assets(
                bounds=mask.bounds,
                year=year,
            )
            return sorted(set(local_tile_ids) | set(cdse_tile_ids))
        except Exception:
            if self.allow_wekeo_fallback:
                return super().get_tiles_for_mask(
                    mask=mask,
                    year=year,
                    query_overrides=query_overrides,
                )
            raise
