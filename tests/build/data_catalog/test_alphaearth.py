"""Tests for the AlphaEarth data catalog adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from geb.build.data_catalog import alphaearth as alphaearth_module
from geb.build.data_catalog.alphaearth import (
    AVAILABLE_YEARS,
    DEFAULT_BASE_URL,
    INDEX_FILENAME,
    AlphaEarth,
)


def make_index() -> gpd.GeoDataFrame:
    """Create a small synthetic AlphaEarth index in WGS84."""
    return gpd.GeoDataFrame(
        {
            "year": [2023, 2024, 2024, 2025],
            "utm_zone": ["31N", "31N", "32N", "31N"],
            "path": [
                "2023/31N/aef_2023_31n.tif",
                "2024/31N/aef_2024_31n.tif",
                (
                    "gs://alphaearth_foundations/satellite_embedding/"
                    "v1/annual/2024/32N/aef_2024_32n.tif"
                ),
                "2025/31N/aef_2025_31n.tif",
            ],
            "wgs84_west": [3.0, 3.0, 6.0, 3.0],
            "wgs84_south": [50.0, 50.0, 50.0, 50.0],
            "wgs84_east": [6.0, 6.0, 9.0, 6.0],
            "wgs84_north": [54.0, 54.0, 54.0, 54.0],
        },
        geometry=[
            box(3.0, 50.0, 6.0, 54.0),
            box(3.0, 50.0, 6.0, 54.0),
            box(6.0, 50.0, 9.0, 54.0),
            box(3.0, 50.0, 6.0, 54.0),
        ],
        crs=4326,
    )


def patch_cached_index(
    monkeypatch: pytest.MonkeyPatch,
    adapter: AlphaEarth,
    tmp_path: Path,
    index: gpd.GeoDataFrame,
) -> Path:
    """Patch index access so adapter tests never contact the network."""
    index_path = tmp_path / INDEX_FILENAME
    index_path.write_bytes(b"synthetic-index")

    async def fake_ensure_index(refresh: bool = False) -> Path:
        del refresh
        return index_path

    monkeypatch.setattr(adapter, "_ensure_index", fake_ensure_index)
    monkeypatch.setattr(
        alphaearth_module.gpd,
        "read_parquet",
        lambda path: index.copy(),
    )
    return index_path


def test_alphaearth_init_rejects_invalid_parallel_download_count(
    tmp_path: Path,
) -> None:
    """Verify download concurrency must be at least one."""
    with pytest.raises(ValueError, match="at least 1"):
        AlphaEarth(
            cache_dir=tmp_path,
            max_parallel_downloads=0,
        )


def test_alphaearth_fetch_uses_official_default_url(tmp_path: Path) -> None:
    """Verify fetch(None) selects the official AlphaEarth endpoint."""
    adapter = AlphaEarth(cache_dir=tmp_path)

    result = adapter.fetch(None)

    assert result is adapter
    assert adapter.url == DEFAULT_BASE_URL
    assert adapter.index_url == f"{DEFAULT_BASE_URL}/{INDEX_FILENAME}"


def test_alphaearth_fetch_accepts_alternative_base_url(tmp_path: Path) -> None:
    """Verify an alternative mirror URL can be configured."""
    adapter = AlphaEarth(cache_dir=tmp_path)

    result = adapter.fetch("https://example.com/alphaearth/")

    assert result is adapter
    assert adapter.url == "https://example.com/alphaearth"
    assert adapter.index_url == ("https://example.com/alphaearth/aef_index.parquet")


def test_alphaearth_index_path_uses_configured_cache(tmp_path: Path) -> None:
    """Verify the spatial index is cached in the configured directory."""
    adapter = AlphaEarth(cache_dir=tmp_path / "cache")

    assert adapter.index_path == tmp_path / "cache" / INDEX_FILENAME
    assert adapter.index_path.parent.is_dir()


def test_alphaearth_dequantize_converts_values_and_nodata() -> None:
    """Verify int8 embeddings are converted to float32 and nodata to NaN."""
    raw = np.array(
        [-128, -127, -64, 0, 64, 127],
        dtype=np.int16,
    )

    result = AlphaEarth.dequantize(raw)

    expected = (raw.astype(np.float32) / 127.5) ** 2 * np.sign(raw.astype(np.float32))
    expected[0] = np.nan

    assert result.dtype == np.float32
    np.testing.assert_allclose(
        result[1:],
        expected[1:],
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.isnan(result[0])
    assert result[3] == pytest.approx(0.0)


def test_alphaearth_normalize_years_sorts_and_deduplicates() -> None:
    """Verify requested years are normalized deterministically."""
    result = AlphaEarth._normalize_years([2024, 2023, 2024])

    assert result == (2023, 2024)


def test_alphaearth_normalize_years_rejects_unsupported_year() -> None:
    """Verify years outside the published range raise a clear error."""
    unsupported_year = AVAILABLE_YEARS[-1] + 1

    with pytest.raises(ValueError, match=str(unsupported_year)):
        AlphaEarth._normalize_years([unsupported_year])


@pytest.mark.parametrize(
    "bounds",
    [
        (5.0, 50.0, 4.0, 51.0),
        (4.0, 52.0, 5.0, 51.0),
        (-181.0, 50.0, 5.0, 51.0),
        (4.0, -91.0, 5.0, 51.0),
    ],
)
def test_alphaearth_validate_bounds_rejects_invalid_values(
    bounds: tuple[float, float, float, float],
) -> None:
    """Verify malformed WGS84 bounding boxes are rejected."""
    with pytest.raises(ValueError):
        AlphaEarth._validate_bounds(bounds)


def test_alphaearth_select_files_filters_year_and_bounds(
    tmp_path: Path,
) -> None:
    """Verify selection uses both requested years and spatial intersection."""
    adapter = AlphaEarth(cache_dir=tmp_path)
    index = make_index()

    selected = adapter.select_files(
        index=index,
        years=[2024],
        bounds=(4.0, 51.0, 7.0, 53.0),
    )

    assert selected["year"].tolist() == [2024, 2024]
    assert selected["utm_zone"].tolist() == ["31N", "32N"]
    assert selected["remote_url"].tolist() == [
        f"{DEFAULT_BASE_URL}/2024/31N/aef_2024_31n.tif",
        (
            "https://storage.googleapis.com/alphaearth_foundations/"
            "satellite_embedding/v1/annual/2024/32N/"
            "aef_2024_32n.tif"
        ),
    ]


def test_alphaearth_select_files_reprojects_non_wgs84_index(
    tmp_path: Path,
) -> None:
    """Verify a projected spatial index is transformed before selection."""
    adapter = AlphaEarth(cache_dir=tmp_path)
    projected_index = make_index().to_crs(3857)

    selected = adapter.select_files(
        index=projected_index,
        years=2024,
        bounds=(4.0, 51.0, 5.0, 53.0),
    )

    assert len(selected) == 1
    assert selected.iloc[0]["utm_zone"] == "31N"
    assert selected.crs.to_epsg() == 4326


def test_alphaearth_select_files_raises_for_empty_selection(
    tmp_path: Path,
) -> None:
    """Verify an empty spatial selection raises a useful error."""
    adapter = AlphaEarth(cache_dir=tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="No AlphaEarth COGs",
    ):
        adapter.select_files(
            index=make_index(),
            years=2024,
            bounds=(-10.0, 35.0, -9.0, 36.0),
        )


def test_alphaearth_select_files_rejects_negative_buffer(
    tmp_path: Path,
) -> None:
    """Verify a negative selection buffer is not accepted."""
    adapter = AlphaEarth(cache_dir=tmp_path)

    with pytest.raises(ValueError, match="cannot be negative"):
        adapter.select_files(
            index=make_index(),
            years=2024,
            bounds=(4.0, 51.0, 5.0, 53.0),
            buffer_degrees=-0.1,
        )


def test_alphaearth_read_dry_run_uses_cached_index_and_plans_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify dry-run selection plans local files without downloading them."""
    adapter = AlphaEarth(cache_dir=tmp_path / "index-cache")
    patch_cached_index(
        monkeypatch=monkeypatch,
        adapter=adapter,
        tmp_path=tmp_path,
        index=make_index(),
    )

    download_dir = tmp_path / "downloads"
    selected = adapter.read(
        years=[2024],
        bounds=(4.0, 51.0, 5.0, 53.0),
        download_dir=download_dir,
        dry_run=True,
    )

    assert len(selected) == 1
    assert selected.iloc[0]["local_path"] == str(
        download_dir / "2024" / "31N" / "aef_2024_31n.tif"
    )
    assert not Path(selected.iloc[0]["local_path"]).exists()


def test_alphaearth_read_enforces_max_files_safety_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify broad non-dry-run selections require explicit approval."""
    adapter = AlphaEarth(cache_dir=tmp_path / "index-cache")
    patch_cached_index(
        monkeypatch=monkeypatch,
        adapter=adapter,
        tmp_path=tmp_path,
        index=make_index(),
    )

    with pytest.raises(RuntimeError, match="exceeding max_files=1"):
        adapter.read(
            years=[2024],
            bounds=(4.0, 51.0, 7.0, 53.0),
            download_dir=tmp_path / "downloads",
            dry_run=False,
            max_files=1,
        )


def test_alphaearth_download_file_reuses_existing_cached_file(
    tmp_path: Path,
) -> None:
    """Verify an existing non-empty COG is not downloaded again."""
    adapter = AlphaEarth(cache_dir=tmp_path / "index-cache")
    destination = tmp_path / "downloads" / "cached.tif"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"cached")

    class FailingClient:
        """Fail if the adapter unexpectedly attempts an HTTP request."""

        def get(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("HTTP download should not have been attempted")

    result = asyncio.run(
        adapter._download_file(
            client=FailingClient(),  # type: ignore[arg-type]
            remote_url="https://example.com/cached.tif",
            destination=destination,
            overwrite=False,
            semaphore=asyncio.Semaphore(1),
        )
    )

    assert result == destination
    assert destination.read_bytes() == b"cached"


def test_alphaearth_read_rejects_active_event_loop(tmp_path: Path) -> None:
    """Verify synchronous read directs async callers to read_async()."""
    adapter = AlphaEarth(cache_dir=tmp_path)

    async def call_read() -> None:
        with pytest.raises(RuntimeError, match="read_async"):
            adapter.read(
                years=2024,
                bounds=(4.0, 51.0, 5.0, 53.0),
                download_dir=tmp_path / "downloads",
                dry_run=True,
            )

    asyncio.run(call_read())
