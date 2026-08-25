"""Live integration tests for HRL Crop Types through Copernicus Data Space.

These tests intentionally cover the failure modes seen in the Europe HRL build:

* authoritative STAC discovery and direct asset download;
* a small real four-tile mosaic around an EEA 100 km grid corner;
* Dask-backed lazy CDSE mosaicking followed by actual computation;
* mixed legacy-WKT / EPSG:3035 CRS serialization using real downloaded CTY pixels;
* a partially populated temporary cache completed from CDSE;
* cross-year filename prediction checked against authoritative target-year STAC.

The expensive tests are skipped on GitHub Actions. Download tests additionally require
CDSE OAuth or S3 credentials.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from geb.build.data_catalog import DataCatalog
from geb.build.data_catalog.copernicus_hrl import (
    CDSEDownloadError,
    CDSENoCoverageError,
    CopernicusDataSpace,
)

from ...testconfig import IN_GITHUB_ACTIONS


HRL_BOUNDS = tuple(
    float(value)
    for value in os.environ.get(
        "GEB_LIVE_HRL_BOUNDS",
        "4.80,52.25,4.82,52.27",
    ).split(",")
)
HRL_YEARS = tuple(
    int(value)
    for value in os.environ.get("GEB_LIVE_HRL_YEARS", "2017,2018,2023,2024").split(",")
)
HRL_DIRECT_YEAR = int(os.environ.get("GEB_LIVE_HRL_DIRECT_YEAR", "2018"))
HRL_DIRECT_TILE_ID = os.environ.get(
    "GEB_LIVE_HRL_DIRECT_TILE_ID",
    "CLMS_HRLVLCC_CTY_S2018_R10m_E51N30_03035_V01_R00",
)

# A deliberately small box around the x=5,100,000 / y=3,000,000 EPSG:3035 tile
# corner. It touches E50N29, E50N30, E51N29 and E51N30 while keeping the actual
# returned raster only a few kilometres wide/high. These are the same four 2018
# tiles that appeared in the reported production merge failure.
HRL_MULTI_TILE_BOUNDS = tuple(
    float(value)
    for value in os.environ.get(
        "GEB_LIVE_HRL_MULTI_TILE_BOUNDS",
        "20.78682,49.58014,20.84966,49.61040",
    ).split(",")
)
HRL_MULTI_TILE_YEAR = int(os.environ.get("GEB_LIVE_HRL_MULTI_TILE_YEAR", "2018"))
HRL_MULTI_TILE_YEARS = tuple(
    int(value)
    for value in os.environ.get(
        "GEB_LIVE_HRL_MULTI_TILE_YEARS",
        "2018,2024",
    ).split(",")
)
HRL_REQUIRED_STAC_YEARS = tuple(
    int(value)
    for value in os.environ.get(
        "GEB_LIVE_HRL_REQUIRED_STAC_YEARS",
        "2017,2018,2019,2020,2021,2022,2023,2024",
    ).split(",")
)
HRL_OPTIONAL_FORWARD_STAC_YEARS = tuple(
    int(value)
    for value in os.environ.get(
        "GEB_LIVE_HRL_OPTIONAL_FORWARD_STAC_YEARS",
        "2025",
    ).split(",")
)
HRL_SECONDARY_STAC_YEARS = tuple(
    int(value)
    for value in os.environ.get(
        "GEB_LIVE_HRL_SECONDARY_STAC_YEARS",
        "2018,2024",
    ).split(",")
)
HRL_REPROJECT_YEARS = tuple(
    int(value)
    for value in os.environ.get(
        "GEB_LIVE_HRL_REPROJECT_YEARS",
        "2018,2024",
    ).split(",")
)
HRL_LAZY_CHUNKS = {"x": 512, "y": 512}

# Regression case from the failed build: a filename inferred from another year's
# EEA-grid catalogue was demanded for 2018 even though target-year CDSE STAC did not
# provide that asset. The live test verifies the current target-year catalogue before
# treating a predicted filename as required.
HRL_PREDICTION_REFERENCE_YEARS = tuple(
    int(value)
    for value in os.environ.get(
        "GEB_LIVE_HRL_PREDICTION_REFERENCE_YEARS",
        "2017,2019,2020,2021,2022,2023",
    ).split(",")
)
HRL_PREDICTION_TARGET_YEAR = int(
    os.environ.get("GEB_LIVE_HRL_PREDICTION_TARGET_YEAR", "2018")
)
HRL_PREDICTION_TILE_CODE = os.environ.get(
    "GEB_LIVE_HRL_PREDICTION_TILE_CODE",
    "E35N17",
)

# Exact legacy CRS serialization from the production traceback. Rasterio/GDAL 3.12.1
# does not consider this object equal to CRS.from_epsg(3035), despite the same LAEA
# grid parameters. The old lazy implementation therefore rejected a valid tile pair.
LEGACY_EPSG3035_WKT = (
    'PROJCS["ETRS89-extended / LAEA Europe",GEOGCS["ETRS89",'
    'DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",'
    '6378137,298.257222101004,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6258"]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4258"]],PROJECTION["Lambert_Azimuthal_Equal_Area"],'
    'PARAMETER["latitude_of_center",52],PARAMETER["longitude_of_center",10],'
    'PARAMETER["false_easting",4321000],PARAMETER["false_northing",3210000],'
    'UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH],'
    'AUTHORITY["EPSG","3035"]]'
)

# Exact WGS84 bounds from production failures. These are intentionally STAC-only
# checks: they exercise discovery at the problematic geographic edges without
# downloading/mosaicking very large regional windows.
HRL_REPORTED_2018_DISCOVERY_CASES = (
    (
        "central_europe_e50_e51",
        (19.46737670900012, 49.18758773900015, 21.39727020300012, 49.625625),
        ("E50N29", "E50N30", "E51N29", "E51N30"),
    ),
    (
        "turkey_e66_e67",
        (36.44259262000014, 36.80651855400015, 37.232708333333335, 37.367008209000176),
        ("E66N19", "E66N20", "E67N19", "E67N20"),
    ),
    (
        "greece_e55_e58",
        (23.513748000000135, 34.92124900000016, 26.318750000000136, 35.69597244200003),
        (
            "E55N14",
            "E55N15",
            "E56N14",
            "E56N15",
            "E57N14",
            "E57N15",
            "E58N14",
            "E58N15",
        ),
    ),
)

# Model-1 failure that previously routed 2024 through LocalHRLCroplands.
HRL_2024_NORTHERN_REGRESSION_BOUNDS = (
    21.9924793240001,
    68.55951296221221,
    24.88270833333334,
    69.07109069800003,
)


def _has_cdse_download_credentials() -> bool:
    """Return whether either supported CDSE download route is configured."""
    has_oauth = bool(os.environ.get("CDSE_ACCESS_TOKEN")) or bool(
        os.environ.get("CDSE_USERNAME") and os.environ.get("CDSE_PASSWORD")
    )
    has_s3 = bool(
        (os.environ.get("CDSE_S3_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID"))
        and (
            os.environ.get("CDSE_S3_SECRET_KEY")
            or os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
    )
    return has_oauth or has_s3


def _unfetched_catalog_adapter(name: str):
    """Return a catalog adapter without starting its fetch operation."""
    catalog = DataCatalog(logger=logging.getLogger(name))
    entry = catalog.catalog[name]
    return entry["adapter"], entry.get("url")


def _native_clip_bounds(bounds: tuple[float, float, float, float]) -> tuple[float, ...]:
    """Transform WGS84 test bounds to the fixed native HRL EPSG:3035 grid."""
    return tuple(
        float(value)
        for value in transform_bounds("EPSG:4326", "EPSG:3035", *bounds, densify_pts=21)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_real_tiles(
    adapter,
    *,
    bounds: tuple[float, float, float, float],
    year: int,
    target_dir: Path,
) -> tuple[list[str], list[Path]]:
    """Discover and download all real STAC assets intersecting a small test box."""
    tile_ids, assets = adapter._search_cdse_assets(bounds=bounds, year=year)
    assert len(tile_ids) >= 2, (
        "Multi-tile regression bounds unexpectedly returned fewer than two assets: "
        f"{tile_ids}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for tile_id in tile_ids:
        target = target_dir / f"{tile_id}.tif"
        adapter._download_cdse_asset(assets[tile_id], target)
        assert target.is_file() and target.stat().st_size > 100_000
        paths.append(target)
    return tile_ids, paths


def _write_vrt_with_srs(source: Path, target: Path, srs: str) -> Path:
    """Wrap a real downloaded tile in a VRT with a controlled CRS serialization."""
    with rasterio.open(source) as dataset:
        transform = ",".join(str(value) for value in dataset.transform.to_gdal())
        width = dataset.width
        height = dataset.height
        dtype = dataset.dtypes[0]
        nodata = dataset.nodata
        block_height, block_width = dataset.block_shapes[0]

    dtype_name = {
        "uint16": "UInt16",
        "int16": "Int16",
        "uint8": "Byte",
        "int32": "Int32",
    }.get(dtype, "UInt16")
    nodata_xml = "" if nodata is None else f"<NoDataValue>{nodata}</NoDataValue>"
    target.write_text(
        f'''<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n'''
        f"  <SRS>{escape(srs)}</SRS>\n"
        f"  <GeoTransform>{transform}</GeoTransform>\n"
        f'''  <VRTRasterBand dataType="{dtype_name}" band="1">\n'''
        f"    {nodata_xml}\n"
        "    <SimpleSource>\n"
        f"""      <SourceFilename relativeToVRT="0">{escape(str(source))}</SourceFilename>\n"""
        "      <SourceBand>1</SourceBand>\n"
        f'''      <SourceProperties RasterXSize="{width}" RasterYSize="{height}" '''
        f'''DataType="{dtype_name}" BlockXSize="{block_width}" BlockYSize="{block_height}"/>\n'''
        f'''      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'''
        f'''      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'''
        "    </SimpleSource>\n"
        "  </VRTRasterBand>\n"
        "</VRTDataset>\n"
    )
    return target


def _patch_adapter_cache_to_tmp(
    monkeypatch: pytest.MonkeyPatch, adapter, root: Path
) -> None:
    """Redirect one catalog adapter's cache helpers to an isolated temporary root."""

    def year_dir(year: str | int) -> Path:
        return root / str(year)

    def tif_path(year: str | int, tile_id: str) -> Path:
        return year_dir(year) / f"{tile_id}.tif"

    def zip_path(year: str | int, tile_id: str) -> Path:
        return year_dir(year) / f"{tile_id}.zip"

    def scan(year: str | int) -> list[str]:
        directory = year_dir(year)
        if not directory.exists():
            return []
        tile_ids = {path.stem for path in directory.glob("*.tif")}
        tile_ids.update(path.stem for path in directory.glob("*.zip"))
        return sorted(tile_ids)

    monkeypatch.setattr(adapter, "_year_dir", year_dir)
    monkeypatch.setattr(adapter, "_tile_tif_path", tif_path)
    monkeypatch.setattr(adapter, "_tile_zip_path", zip_path)
    monkeypatch.setattr(adapter, "_scan_local_tile_ids", scan)


def _live_collection_item_ids_for_year_range(
    adapter: CopernicusDataSpace,
    *,
    bounds: tuple[float, float, float, float],
    years: tuple[int, ...],
) -> list[str]:
    """Return live STAC item IDs for several years using one catalogue request.

    The extended regression suite used to issue one nearly identical STAC request per
    year. That made the test itself capable of triggering CDSE HTTP 429 responses.
    A single bounded multi-year query still verifies publication coverage while keeping
    the live suite friendly to the public catalogue.
    """
    if not years:
        raise ValueError("At least one STAC year is required.")
    collection_id = adapter._resolve_cdse_collection_id()
    payload = adapter._stac_request_json(
        "GET",
        f"{adapter.cdse_stac_url}/search",
        params={
            "collections": collection_id,
            "bbox": ",".join(str(float(value)) for value in bounds),
            "datetime": (f"{min(years)}-01-01T00:00:00Z/{max(years)}-12-31T23:59:59Z"),
            "limit": 1000,
        },
    )
    features = payload.get("features", [])
    assert isinstance(features, list)
    assert len(features) < 1000, (
        "The compact live regression bounds unexpectedly reached the STAC limit; "
        "the test must implement pagination before relying on this result."
    )
    return [
        str(feature.get("id", ""))
        for feature in features
        if isinstance(feature, dict) and feature.get("id")
    ]


def test_hrl_catalog_uses_only_cdse_and_disables_legacy_fallback() -> None:
    """All configured HRL years must route through CDSE, never the retired adapter."""
    assert (
        importlib.util.find_spec("geb.build.data_catalog.local_hrl_croplands") is None
    ), "The retired local_hrl_croplands module is still importable."

    for year in range(2017, 2031):
        for prefix in ("hrl_crop_types", "hrl_secondary_crop"):
            adapter, _ = _unfetched_catalog_adapter(f"{prefix}_{year}")
            assert isinstance(adapter, CopernicusDataSpace)
            assert adapter.allow_wekeo_fallback is False


def test_cdse_no_coverage_does_not_call_legacy_wekeo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CDSE no-coverage result must never fall through to retired WEkEO HDA."""
    adapter, _ = _unfetched_catalog_adapter("hrl_crop_types_2018")
    _patch_adapter_cache_to_tmp(monkeypatch, adapter, tmp_path / "empty_cache")

    def no_coverage(*, bounds, year):
        raise CDSENoCoverageError("deterministic no-coverage regression")

    def forbidden_wekeo(**kwargs):
        pytest.fail(
            "Legacy WEkEO fallback was called despite allow_wekeo_fallback=False."
        )

    monkeypatch.setattr(adapter, "_search_cdse_assets", no_coverage)
    monkeypatch.setattr(adapter, "_wekeo_fallback_for_missing", forbidden_wekeo)

    with pytest.raises(CDSENoCoverageError, match="deterministic no-coverage"):
        adapter.fetch(bounds=(-40.0, 10.0, -39.9, 10.1), year=2018)


def test_cdse_stac_request_retries_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recreate the earlier transient STAC 500/retry failure deterministically."""
    adapter, _ = _unfetched_catalog_adapter("hrl_crop_types_2018")
    adapter.download_retries = 2
    adapter.download_backoff_seconds = 0.0
    calls: list[int] = []

    class FakeResponse:
        def __init__(self, succeeds: bool) -> None:
            self.succeeds = succeeds

        def raise_for_status(self) -> None:
            if not self.succeeds:
                raise RuntimeError("synthetic CDSE HTTP 500")

        def json(self) -> dict[str, object]:
            return {"features": [], "synthetic": True}

    class FakeRequests:
        @staticmethod
        def request(**kwargs):
            del kwargs
            calls.append(1)
            return FakeResponse(succeeds=len(calls) >= 3)

    monkeypatch.setattr(adapter, "_requests", lambda: FakeRequests())
    payload = adapter._stac_request_json("GET", "https://example.invalid/stac")
    assert payload["synthetic"] is True
    assert len(calls) == 3


def test_cdse_stac_request_429_honors_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 429 must pause rather than immediately repeat the rejected request."""
    adapter, _ = _unfetched_catalog_adapter("hrl_crop_types_2018")
    adapter.download_retries = 1
    adapter.download_backoff_seconds = 0.0
    calls: list[int] = []
    sleeps: list[float] = []

    class FakeHTTPError(RuntimeError):
        def __init__(self, response) -> None:
            super().__init__("synthetic CDSE HTTP 429")
            self.response = response

    class FakeResponse:
        def __init__(self, succeeds: bool) -> None:
            self.succeeds = succeeds
            self.status_code = 200 if succeeds else 429
            self.headers = {} if succeeds else {"Retry-After": "7"}

        def raise_for_status(self) -> None:
            if not self.succeeds:
                raise FakeHTTPError(self)

        def json(self) -> dict[str, object]:
            return {"features": [], "synthetic": True}

    class FakeRequests:
        @staticmethod
        def request(**kwargs):
            del kwargs
            calls.append(1)
            return FakeResponse(succeeds=len(calls) >= 2)

    monkeypatch.setattr(adapter, "_requests", lambda: FakeRequests())
    monkeypatch.setattr(
        "geb.build.data_catalog.copernicus_hrl.time.sleep", sleeps.append
    )

    payload = adapter._stac_request_json("GET", "https://example.invalid/stac")
    assert payload["synthetic"] is True
    assert len(calls) == 2
    assert sleeps == [7.0]


def test_cdse_retry_after_http_date_is_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP-date Retry-After values must be converted to a real delay."""
    adapter, _ = _unfetched_catalog_adapter("hrl_crop_types_2018")
    monkeypatch.setattr(
        "geb.build.data_catalog.copernicus_hrl.time.time",
        lambda: 1_700_000_000.0,
    )
    # Exactly 11 seconds after the patched clock.
    retry_at = "Tue, 14 Nov 2023 22:13:31 GMT"
    assert adapter._retry_after_seconds(retry_at) == pytest.approx(11.0)


def test_cdse_stac_request_503_uses_generic_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server-side failures remain retryable using the configured backoff."""
    adapter, _ = _unfetched_catalog_adapter("hrl_crop_types_2018")
    adapter.download_retries = 1
    adapter.download_backoff_seconds = 2.5
    calls: list[int] = []
    sleeps: list[float] = []

    class FakeHTTPError(RuntimeError):
        def __init__(self, response) -> None:
            super().__init__("synthetic CDSE HTTP 503")
            self.response = response

    class FakeResponse:
        def __init__(self, succeeds: bool) -> None:
            self.succeeds = succeeds
            self.status_code = 200 if succeeds else 503
            self.headers = {}

        def raise_for_status(self) -> None:
            if not self.succeeds:
                raise FakeHTTPError(self)

        def json(self) -> dict[str, object]:
            return {"features": [], "synthetic": True}

    class FakeRequests:
        @staticmethod
        def request(**kwargs):
            del kwargs
            calls.append(1)
            return FakeResponse(succeeds=len(calls) >= 2)

    monkeypatch.setattr(adapter, "_requests", lambda: FakeRequests())
    monkeypatch.setattr(
        "geb.build.data_catalog.copernicus_hrl.time.sleep", sleeps.append
    )

    payload = adapter._stac_request_json("GET", "https://example.invalid/stac")
    assert payload["synthetic"] is True
    assert len(calls) == 2
    assert sleeps == [2.5]


def test_cdse_stac_request_404_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deterministic non-429 4xx responses must fail without sleeping/retrying."""
    adapter, _ = _unfetched_catalog_adapter("hrl_crop_types_2018")
    adapter.download_retries = 5
    adapter.download_backoff_seconds = 10.0
    calls: list[int] = []
    sleeps: list[float] = []

    class FakeHTTPError(RuntimeError):
        def __init__(self, response) -> None:
            super().__init__("synthetic CDSE HTTP 404")
            self.response = response

    class FakeResponse:
        status_code = 404
        headers: dict[str, str] = {}

        @staticmethod
        def raise_for_status() -> None:
            raise FakeHTTPError(FakeResponse())

    class FakeRequests:
        @staticmethod
        def request(**kwargs):
            del kwargs
            calls.append(1)
            return FakeResponse()

    monkeypatch.setattr(adapter, "_requests", lambda: FakeRequests())
    monkeypatch.setattr(
        "geb.build.data_catalog.copernicus_hrl.time.sleep", sleeps.append
    )

    with pytest.raises(CDSEDownloadError, match=r"after 1 attempt\(s\)"):
        adapter._stac_request_json("GET", "https://example.invalid/stac")
    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Queries live CTY STAC for all model years."
)
def test_live_cdse_required_cty_years_are_discoverable() -> None:
    """Every required CTY year must occur in one bounded live STAC query."""
    adapter, _ = _unfetched_catalog_adapter(
        f"hrl_crop_types_{HRL_REQUIRED_STAC_YEARS[0]}"
    )
    item_ids = _live_collection_item_ids_for_year_range(
        adapter, bounds=HRL_BOUNDS, years=HRL_REQUIRED_STAC_YEARS
    )
    assert item_ids
    for year in HRL_REQUIRED_STAC_YEARS:
        matching = [item_id for item_id in item_ids if f"_CTY_S{year}_" in item_id]
        assert matching, f"No live CTY item was returned for required year {year}."


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Queries live forward-year CTY STAC.")
@pytest.mark.parametrize("year", HRL_OPTIONAL_FORWARD_STAC_YEARS)
def test_live_cdse_forward_cty_year_if_published(year: int) -> None:
    """Probe newer CDSE years without making them a hard model requirement yet."""
    adapter, _ = _unfetched_catalog_adapter(f"hrl_crop_types_{year}")
    try:
        tile_ids, assets = adapter._search_cdse_assets(bounds=HRL_BOUNDS, year=year)
    except CDSENoCoverageError:
        pytest.skip(f"CTY {year} is not yet published for the live regression bounds.")
    assert tile_ids
    assert set(tile_ids) == set(assets)
    assert all(f"_S{year}_" in tile_id for tile_id in tile_ids)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Queries live CPSCT STAC.")
def test_live_cdse_secondary_crop_collection_is_discoverable() -> None:
    """The shared CDSE adapter must expose representative CPSCT years."""
    adapter, _ = _unfetched_catalog_adapter(
        f"hrl_secondary_crop_{HRL_SECONDARY_STAC_YEARS[0]}"
    )
    assert adapter._resolve_cdse_collection_id() == (
        "clms_vlcc_secondary-crop-types_europe_10m_yearly_v1"
    )
    item_ids = _live_collection_item_ids_for_year_range(
        adapter, bounds=HRL_BOUNDS, years=HRL_SECONDARY_STAC_YEARS
    )
    assert item_ids
    for year in HRL_SECONDARY_STAC_YEARS:
        matching = [item_id for item_id in item_ids if f"_CPSCT_S{year}_" in item_id]
        assert matching, f"No live CPSCT item was returned for required year {year}."


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Queries live STAC at prior failure regions."
)
@pytest.mark.parametrize(
    ("case_name", "bounds", "expected_tile_codes"),
    HRL_REPORTED_2018_DISCOVERY_CASES,
    ids=[case[0] for case in HRL_REPORTED_2018_DISCOVERY_CASES],
)
def test_live_cdse_reported_2018_failure_regions_discover_authoritative_tiles(
    case_name: str,
    bounds: tuple[float, float, float, float],
    expected_tile_codes: tuple[str, ...],
) -> None:
    """The exact production failure regions must resolve through target-year STAC."""
    del case_name
    adapter, _ = _unfetched_catalog_adapter("hrl_crop_types_2018")
    tile_ids, assets = adapter._search_cdse_assets(bounds=bounds, year=2018)
    assert tile_ids
    assert set(tile_ids) == set(assets)
    observed_codes = {
        code
        for tile_id in tile_ids
        for code in expected_tile_codes
        if f"_{code}_" in tile_id
    }
    # At least one of the exact tiles mentioned in each production traceback must be
    # discoverable; STAC may legitimately return extra edge-intersecting tiles.
    assert observed_codes


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Queries live 2024 CTY STAC in northern Europe."
)
def test_live_cdse_2024_northern_model1_bounds_use_official_cdse() -> None:
    """Recreate the model-1 2024 bounds that previously hit LocalHRLCroplands."""
    adapter, _ = _unfetched_catalog_adapter("hrl_crop_types_2024")
    assert isinstance(adapter, CopernicusDataSpace)
    tile_ids, assets = adapter._search_cdse_assets(
        bounds=HRL_2024_NORTHERN_REGRESSION_BOUNDS,
        year=2024,
    )
    assert tile_ids
    assert set(tile_ids) == set(assets)
    assert all("_CTY_S2024_" in tile_id for tile_id in tile_ids)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Queries the live CDSE STAC catalogue.")
def test_live_cdse_hrl_stac_discovery() -> None:
    """Resolve the official CTY collection and discover real data assets."""
    adapter, _ = _unfetched_catalog_adapter(f"hrl_crop_types_{HRL_DIRECT_YEAR}")
    tile_ids, assets = adapter._search_cdse_assets(
        bounds=HRL_BOUNDS,
        year=HRL_DIRECT_YEAR,
    )

    assert adapter._resolve_cdse_collection_id() == (
        "clms_vlcc_crop-types_europe_10m_yearly_v1"
    )
    assert tile_ids
    assert set(tile_ids) == set(assets)
    assert all("_CTY_" in tile_id for tile_id in tile_ids)
    assert all(asset.hrefs for asset in assets.values())
    assert any(
        href.startswith("https://download.dataspace.copernicus.eu/")
        for asset in assets.values()
        for href in asset.hrefs
    )


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS or not _has_cdse_download_credentials(),
    reason="Downloads a real CTY GeoTIFF and requires CDSE OAuth or S3 credentials.",
)
def test_live_cdse_hrl_direct_tile_download(tmp_path: Path) -> None:
    """Download one real STAC data asset and open it as a GeoTIFF."""
    adapter, _ = _unfetched_catalog_adapter(f"hrl_crop_types_{HRL_DIRECT_YEAR}")
    collection_id = adapter._resolve_cdse_collection_id()
    feature = adapter._stac_request_json(
        "GET",
        f"{adapter.cdse_stac_url}/collections/{collection_id}/items/"
        f"{HRL_DIRECT_TILE_ID}",
    )
    asset = adapter._select_feature_asset(feature, HRL_DIRECT_TILE_ID)
    assert asset is not None

    target = tmp_path / f"{HRL_DIRECT_TILE_ID}.tif"
    successful_href = adapter._download_cdse_asset(asset, target)

    assert target.is_file() and target.stat().st_size > 100_000
    assert successful_href in asset.hrefs
    with rasterio.open(target) as dataset:
        assert dataset.count == 1
        assert dataset.crs is not None
        assert dataset.width > 0 and dataset.height > 0
        assert dataset.dtypes[0] == "uint16"


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS or not _has_cdse_download_credentials(),
    reason="Downloads/reads a real four-tile CTY mosaic.",
)
@pytest.mark.parametrize("year", HRL_MULTI_TILE_YEARS)
def test_live_cdse_hrl_multi_tile_lazy_mosaic(
    year: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exercise the production lazy path on a small real multi-tile intersection."""
    logger = logging.getLogger(f"test_live_cdse_hrl_multi_tile_lazy_mosaic_{year}")
    caplog.set_level(logging.WARNING, logger="rasterio._err")
    adapter = DataCatalog(logger=logger).fetch(
        f"hrl_crop_types_{year}",
        bounds=HRL_MULTI_TILE_BOUNDS,
        year=year,
    )

    assert len(adapter.tile_ids) >= 2
    crop_types = adapter.read(
        bounds=HRL_MULTI_TILE_BOUNDS,
        year=year,
        dst_crs=None,
        normalize_nodata=True,
        chunks=HRL_LAZY_CHUNKS,
    )
    try:
        # This must still be lazy before the explicit load below.
        assert crop_types.chunks is not None
        assert getattr(crop_types, "_close", None) is not None
        assert crop_types.nbytes < 128 * 1024**2
        graph = crop_types.data.__dask_graph__()
        assert graph is not None and len(graph) < 10_000
        assert crop_types.rio.crs == CRS.from_epsg(3035)
        assert crop_types.rio.nodata == -2
        assert np.issubdtype(np.dtype(crop_types.dtype), np.signedinteger)

        crop_types.load()
        values = crop_types.values
        assert values.size > 0
        assert not np.any(values == 65534)
        assert not np.any(values == 65535)
        observed = values[values != -2]
        assert observed.size > 0
        assert np.any(observed != 0)
        assert "changed to 65534" not in caplog.text
        assert "Value 65535 in the source dataset" not in caplog.text
    finally:
        crop_types.close()


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS or not _has_cdse_download_credentials(),
    reason="Downloads real CTY pixels to reproduce the mixed-CRS lazy-mosaic failure.",
)
def test_live_cdse_lazy_mosaic_accepts_mixed_epsg3035_serializations(
    tmp_path: Path,
) -> None:
    """Reproduce the prior false CRS mismatch with real tiles and controlled VRT SRS."""
    adapter, _ = _unfetched_catalog_adapter(f"hrl_crop_types_{HRL_MULTI_TILE_YEAR}")
    _, paths = _download_real_tiles(
        adapter,
        bounds=HRL_MULTI_TILE_BOUNDS,
        year=HRL_MULTI_TILE_YEAR,
        target_dir=tmp_path / "raw",
    )

    canonical_vrt = _write_vrt_with_srs(
        paths[0], tmp_path / "canonical.vrt", "EPSG:3035"
    )
    legacy_vrt = _write_vrt_with_srs(
        paths[1], tmp_path / "legacy.vrt", LEGACY_EPSG3035_WKT
    )

    with rasterio.open(canonical_vrt) as canonical, rasterio.open(legacy_vrt) as legacy:
        # Precondition: this is the exact comparison that killed the old lazy path.
        assert canonical.crs == CRS.from_epsg(3035)
        assert legacy.crs != canonical.crs
        assert legacy.crs.to_epsg() is None

    merged = adapter._merge_tiles(
        [canonical_vrt, legacy_vrt, *paths[2:]],
        chunks=HRL_LAZY_CHUNKS,
        clip_bounds=_native_clip_bounds(HRL_MULTI_TILE_BOUNDS),
        normalize_nodata=True,
    )
    try:
        assert merged.chunks is not None
        assert getattr(merged, "_close", None) is not None
        assert merged.rio.crs == CRS.from_epsg(3035)
        assert merged.rio.nodata == -2
        merged.load()
        assert not np.any(merged.values == 65534)
        assert not np.any(merged.values == 65535)
    finally:
        merged.close()


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS or not _has_cdse_download_credentials(),
    reason="Uses an isolated partial cache and downloads its missing real CTY tiles.",
)
def test_live_cdse_partial_cache_plus_fresh_download_lazy_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recreate the production case where cached and freshly downloaded tiles mix."""
    adapter, _ = _unfetched_catalog_adapter(f"hrl_crop_types_{HRL_MULTI_TILE_YEAR}")
    tile_ids, assets = adapter._search_cdse_assets(
        bounds=HRL_MULTI_TILE_BOUNDS,
        year=HRL_MULTI_TILE_YEAR,
    )
    assert len(tile_ids) >= 2

    cache_root = tmp_path / "cache"
    _patch_adapter_cache_to_tmp(monkeypatch, adapter, cache_root)
    year_dir = cache_root / str(HRL_MULTI_TILE_YEAR)
    year_dir.mkdir(parents=True, exist_ok=True)

    # Seed exactly one real target-year asset, reproducing a partially populated cache.
    cached_tile = tile_ids[0]
    cached_path = year_dir / f"{cached_tile}.tif"
    adapter._download_cdse_asset(assets[cached_tile], cached_path)
    cached_hash = _sha256(cached_path)

    adapter.fetch(
        bounds=HRL_MULTI_TILE_BOUNDS,
        year=HRL_MULTI_TILE_YEAR,
    )
    assert set(adapter.tile_ids) == set(tile_ids)
    assert _sha256(cached_path) == cached_hash  # existing tile was reused, not replaced
    for tile_id in tile_ids:
        assert (year_dir / f"{tile_id}.tif").is_file()

    crop_types = adapter.read(
        bounds=HRL_MULTI_TILE_BOUNDS,
        year=HRL_MULTI_TILE_YEAR,
        dst_crs=None,
        normalize_nodata=True,
        chunks=HRL_LAZY_CHUNKS,
    )
    try:
        assert crop_types.chunks is not None
        assert getattr(crop_types, "_close", None) is not None
        assert crop_types.rio.crs == CRS.from_epsg(3035)
        crop_types.load()
        assert not np.any(crop_types.values == 65534)
        assert not np.any(crop_types.values == 65535)
    finally:
        crop_types.close()

    # Once all exact-year tiles are cached, the same request must be a true local
    # fast path. This catches accidental reintroduction of reference-year inference or
    # unnecessary STAC calls in fully populated production caches.
    def forbidden_stac(*, bounds, year):
        pytest.fail("A complete exact-year cache should not query CDSE STAC.")

    monkeypatch.setattr(adapter, "_search_cdse_assets", forbidden_stac)
    adapter.fetch(bounds=HRL_MULTI_TILE_BOUNDS, year=HRL_MULTI_TILE_YEAR)
    assert set(adapter.tile_ids) == set(tile_ids)


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS or not _has_cdse_download_credentials(),
    reason="Mixes a real legacy-style cached ZIP with freshly downloaded CDSE TIFFs.",
)
def test_live_cdse_legacy_zip_plus_fresh_tiff_lazy_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Legacy WEkEO-era ZIP caches must remain compatible with the lazy CDSE reader."""
    adapter, _ = _unfetched_catalog_adapter(f"hrl_crop_types_{HRL_MULTI_TILE_YEAR}")
    tile_ids, assets = adapter._search_cdse_assets(
        bounds=HRL_MULTI_TILE_BOUNDS,
        year=HRL_MULTI_TILE_YEAR,
    )
    assert len(tile_ids) >= 2

    cache_root = tmp_path / "zip_cache"
    _patch_adapter_cache_to_tmp(monkeypatch, adapter, cache_root)
    year_dir = cache_root / str(HRL_MULTI_TILE_YEAR)
    year_dir.mkdir(parents=True, exist_ok=True)

    # Create one genuine legacy-style ZIP containing the expected TIFF member.
    zip_tile = tile_ids[0]
    staging_tif = tmp_path / f"{zip_tile}.tif"
    adapter._download_cdse_asset(assets[zip_tile], staging_tif)
    zip_path = year_dir / f"{zip_tile}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(staging_tif, arcname=f"{zip_tile}.tif")
    staging_tif.unlink()
    assert zip_path.is_file()
    assert not (year_dir / f"{zip_tile}.tif").exists()

    # fetch() must recognize the ZIP as cached and download only the other target-year
    # assets. read() then extracts that ZIP and lazily mosaics it with the fresh TIFFs.
    adapter.fetch(bounds=HRL_MULTI_TILE_BOUNDS, year=HRL_MULTI_TILE_YEAR)
    assert set(adapter.tile_ids) == set(tile_ids)
    caplog.set_level(logging.WARNING, logger="rasterio._err")
    caplog.clear()
    crop_types = adapter.read(
        bounds=HRL_MULTI_TILE_BOUNDS,
        year=HRL_MULTI_TILE_YEAR,
        dst_crs=None,
        normalize_nodata=True,
        chunks=HRL_LAZY_CHUNKS,
    )
    try:
        assert crop_types.chunks is not None
        assert crop_types.rio.crs == CRS.from_epsg(3035)
        crop_types.load()
        assert not np.any(crop_types.values == 65534)
        assert not np.any(crop_types.values == 65535)
        assert "changed to 65534" not in caplog.text
        assert "Value 65535 in the source dataset" not in caplog.text
    finally:
        crop_types.close()

    assert not zip_path.exists(), "Legacy ZIP should be replaced by its extracted TIFF."
    assert (year_dir / f"{zip_tile}.tif").is_file()


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS or not _has_cdse_download_credentials(),
    reason="Runs the Europe HRL reprojection helper on a real lazy multi-tile CTY read.",
)
@pytest.mark.parametrize("year", HRL_REPROJECT_YEARS)
def test_live_cdse_lazy_output_reprojects_without_65535_to_65534_collision(
    year: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exercise the exact adapter -> Europe reprojection path behind the GDAL warning."""
    from geb.build.custom_models.europe import _reproject_HRL_year_to_subgrid

    adapter = DataCatalog(
        logger=logging.getLogger("test_live_cdse_lazy_output_reprojects")
    ).fetch(
        f"hrl_crop_types_{year}",
        bounds=HRL_MULTI_TILE_BOUNDS,
        year=year,
    )
    crop_types = adapter.read(
        bounds=HRL_MULTI_TILE_BOUNDS,
        year=year,
        dst_crs=None,
        normalize_nodata=True,
        chunks=HRL_LAZY_CHUNKS,
    )
    try:
        assert crop_types.chunks is not None
        assert crop_types.rio.crs == CRS.from_epsg(3035)
        min_x, min_y, max_x, max_y = crop_types.rio.bounds()
        resolution = 500.0
        width = max(1, int(np.ceil((max_x - min_x) / resolution)))
        height = max(1, int(np.ceil((max_y - min_y) / resolution)))
        x = min_x + (np.arange(width, dtype=np.float64) + 0.5) * resolution
        y = max_y - (np.arange(height, dtype=np.float64) + 0.5) * resolution
        template = xr.DataArray(
            np.zeros((height, width), dtype=np.uint8),
            dims=("y", "x"),
            coords={"y": y, "x": x},
        )
        template = template.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        template = template.rio.write_crs("EPSG:3035", inplace=False)

        caplog.set_level(logging.WARNING, logger="rasterio._err")
        caplog.clear()
        crop_states, cultivated_fraction = _reproject_HRL_year_to_subgrid(
            crop_types,
            template,
        )

        assert crop_states.shape == template.shape
        assert cultivated_fraction.shape == template.shape
        assert not np.any(crop_states == 65534)
        assert not np.any(crop_states == 65535)
        assert not np.any(crop_states == -3)
        negative = np.unique(crop_states[crop_states < 0])
        assert set(negative.tolist()).issubset({-2})
        assert np.isfinite(cultivated_fraction).all()
        assert float(cultivated_fraction.min()) >= 0.0
        assert float(cultivated_fraction.max()) <= 1.0
        assert "changed to 65534" not in caplog.text
        assert "Value 65535 in the source dataset" not in caplog.text
    finally:
        crop_types.close()


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS or not _has_cdse_download_credentials(),
    reason=(
        "Uses real reference/target-year CDSE assets in an isolated cache and "
        "requires CDSE download credentials."
    ),
)
def test_live_cdse_predicted_tile_name_must_be_confirmed_by_target_year_stac(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cross-year filename prediction must never expand the STAC download set.

    This regression originally occurred when a tile ID observed in another year was
    rewritten to the target year and then treated as required even though the
    target-year STAC response did not contain that ID.

    The test deliberately does *not* assume that a particular live CDSE asset remains
    absent forever. ``E35N17`` is currently published for 2018, so when necessary we
    recreate the historical response deterministically by removing only that one tile
    from an otherwise real live target-year STAC result. All files used by ``fetch``
    are genuine CDSE GeoTIFFs and all remaining asset metadata comes from live STAC.
    """
    target_adapter, _ = _unfetched_catalog_adapter(
        f"hrl_crop_types_{HRL_PREDICTION_TARGET_YEAR}"
    )
    probe_id = target_adapter._canonical_tile_id(
        HRL_PREDICTION_TILE_CODE, HRL_PREDICTION_TARGET_YEAR
    )
    e_index, n_index = target_adapter._tile_grid_coordinates(probe_id)
    projected = (
        e_index * 100_000 + 100.0,
        n_index * 100_000 + 100.0,
        (e_index + 1) * 100_000 - 100.0,
        (n_index + 1) * 100_000 - 100.0,
    )
    bounds = tuple(
        float(value)
        for value in transform_bounds(
            "EPSG:3035", "EPSG:4326", *projected, densify_pts=21
        )
    )

    # Prove that the prediction itself is based on a real tile from another year.
    reference_match: tuple[int, str, object] | None = None
    for reference_year in HRL_PREDICTION_REFERENCE_YEARS:
        reference_adapter, _ = _unfetched_catalog_adapter(
            f"hrl_crop_types_{reference_year}"
        )
        reference_id = reference_adapter._canonical_tile_id(
            HRL_PREDICTION_TILE_CODE, reference_year
        )
        try:
            reference_ids, reference_assets = reference_adapter._search_cdse_assets(
                bounds=bounds,
                year=reference_year,
            )
        except CDSENoCoverageError:
            continue
        if reference_id in reference_ids:
            reference_match = (
                reference_year,
                reference_id,
                reference_assets[reference_id],
            )
            break

    assert reference_match is not None, (
        "The configured predicted-tile regression anchor no longer exists in any "
        f"reference year: tile={HRL_PREDICTION_TILE_CODE}, "
        f"years={HRL_PREDICTION_REFERENCE_YEARS}."
    )
    reference_year, reference_id, reference_asset = reference_match
    assert (
        target_adapter._replace_tile_year(reference_id, HRL_PREDICTION_TARGET_YEAR)
        == probe_id
    )

    # Obtain the real target-year STAC response first. The regression assertion is
    # about the relationship between this authoritative lookup and the eventual
    # download request, not about whether one hard-coded tile happens to be published
    # on the day the test runs.
    live_target_ids, live_target_assets = target_adapter._search_cdse_assets(
        bounds=bounds,
        year=HRL_PREDICTION_TARGET_YEAR,
    )
    assert live_target_ids
    assert set(live_target_ids) == set(live_target_assets)

    if probe_id in live_target_ids:
        # CDSE currently publishes E35N17 for 2018. Recreate the historical failure
        # deterministically: authoritative target-year discovery returns the same real
        # neighboring assets, but not the cross-year-predicted tile. This is stronger
        # than relying on a catalogue omission that can change over time.
        authoritative_ids = [
            tile_id for tile_id in live_target_ids if tile_id != probe_id
        ]
        authoritative_assets = {
            tile_id: live_target_assets[tile_id] for tile_id in authoritative_ids
        }
        assert authoritative_ids, (
            "Prediction regression bounds now return only the probe tile; enlarge "
            "GEB_LIVE_HRL_PREDICTION search bounds or choose another edge tile."
        )
    else:
        # If CDSE again has a natural reference-only occurrence, use that live response
        # without modification.
        authoritative_ids = list(live_target_ids)
        authoritative_assets = dict(live_target_assets)

    assert probe_id not in authoritative_ids
    assert probe_id not in authoritative_assets

    # Isolate all cache state. Put a *real* reference-year tile in that cache, which is
    # exactly the information the old implementation used to manufacture the target
    # filename. It must have no influence on target-year acquisition now.
    cache_root = tmp_path / "prediction_cache"
    _patch_adapter_cache_to_tmp(monkeypatch, target_adapter, cache_root)
    reference_dir = cache_root / str(reference_year)
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_path = reference_dir / f"{reference_id}.tif"
    target_adapter._download_cdse_asset(reference_asset, reference_path)
    with rasterio.open(reference_path) as dataset:
        assert dataset.count == 1 and dataset.crs is not None

    # Seed all but one authoritative target-year asset. This guarantees fetch() has to
    # execute a genuine CDSE download request while still reproducing the partial-cache
    # production context in which the original bug surfaced.
    target_dir = cache_root / str(HRL_PREDICTION_TARGET_YEAR)
    target_dir.mkdir(parents=True, exist_ok=True)
    for tile_id in authoritative_ids[:-1]:
        target_adapter._download_cdse_asset(
            authoritative_assets[tile_id],
            target_dir / f"{tile_id}.tif",
        )

    # Patch only discovery, using the real live response prepared above. Capture the
    # actual download set and then delegate to the production downloader.
    monkeypatch.setattr(
        target_adapter,
        "_search_cdse_assets",
        lambda *, bounds, year: (authoritative_ids, authoritative_assets),
    )
    requested_downloads: list[str] = []
    production_download = target_adapter._download_cdse_tiles

    def checked_download(tile_ids, year, asset_lookup):
        assert probe_id not in tile_ids
        assert set(tile_ids).issubset(asset_lookup)
        assert set(tile_ids).issubset(authoritative_assets)
        requested_downloads.extend(tile_ids)
        return production_download(tile_ids, year, asset_lookup)

    monkeypatch.setattr(target_adapter, "_download_cdse_tiles", checked_download)

    target_adapter.fetch(
        bounds=bounds,
        year=HRL_PREDICTION_TARGET_YEAR,
    )

    assert requested_downloads, (
        "The isolated regression cache should require a download."
    )
    assert probe_id not in requested_downloads
    assert set(requested_downloads).issubset(authoritative_assets)
    assert probe_id not in target_adapter.tile_ids
    assert set(target_adapter.tile_ids) == set(authoritative_ids)
    assert reference_path.is_file()  # reference evidence was preserved, not consumed


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Downloads and reads real HRL Crop Types tiles."
)
def test_live_fetch_hrl_crop_types(caplog: pytest.LogCaptureFixture) -> None:
    """Read representative years and require an identical output grid across years."""
    if not _has_cdse_download_credentials():
        pytest.skip(
            "CDSE download credentials are required when the tile is not cached."
        )

    assert len(HRL_YEARS) >= 2, "Set GEB_LIVE_HRL_YEARS to at least two years."
    logger = logging.getLogger("test_live_fetch_hrl_crop_types")
    caplog.set_level(logging.WARNING, logger="rasterio._err")
    crop_types_per_year: list[xr.DataArray] = []

    for year in HRL_YEARS:
        adapter = DataCatalog(logger=logger).fetch(
            f"hrl_crop_types_{year}",
            bounds=HRL_BOUNDS,
            year=year,
        )
        caplog.clear()
        crop_types = adapter.read(
            bounds=HRL_BOUNDS,
            year=year,
            dst_crs=None,
            normalize_nodata=True,
            chunks=HRL_LAZY_CHUNKS,
        )
        try:
            assert isinstance(crop_types, xr.DataArray)
            assert crop_types.ndim == 2
            assert crop_types.chunks is not None
            assert getattr(crop_types, "_close", None) is not None
            assert crop_types.rio.crs is not None
            assert crop_types.shape[0] > 0 and crop_types.shape[1] > 0
            assert crop_types.rio.nodata == -2
            assert np.issubdtype(np.dtype(crop_types.dtype), np.signedinteger)

            tile_ids = getattr(adapter, "tile_ids", None)
            assert tile_ids
            assert all("_CTY_" in tile_id for tile_id in tile_ids)
            assert all(f"_S{year}_" in tile_id for tile_id in tile_ids)
            for tile_id in tile_ids:
                tile_path = adapter._tile_tif_path(year, tile_id)
                assert tile_path.is_file() and tile_path.stat().st_size > 0
                assert tile_path.resolve().is_relative_to(Path(adapter.root).resolve())

            crop_types.load()
            values = crop_types.values
            assert not np.any(values == 65534)
            assert not np.any(values == 65535)
            observed = values[values != -2]
            assert observed.size > 0
            assert np.unique(observed).size > 1
            assert np.any(observed != 0)
            assert "changed to 65534" not in caplog.text
            assert "Value 65535 in the source dataset" not in caplog.text

            # The data are now materialized, so closing the source managers must not
            # invalidate the array retained for the exact-grid multi-year concat below.
            crop_types_per_year.append(crop_types.expand_dims(year=[year]))
        finally:
            crop_types.close()

    # This assertion was previously ineffective because the default contained only
    # 2018. With 2017/2018/2023/2024 it now catches year-dependent clipping,
    # reprojection or coordinate-rounding changes.
    crop_types_over_time = xr.concat(
        crop_types_per_year,
        dim="year",
        join="exact",
    )
    assert list(crop_types_over_time["year"].values) == list(HRL_YEARS)
    assert crop_types_over_time.sizes["year"] == len(HRL_YEARS)
