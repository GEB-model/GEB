"""Tests for the FathomDEM data catalog adapter."""

import io
import json
import logging
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from geb.build.data_catalog.fathomdem import (
    FathomDem,
    parse_fathomdem_tile_bounds,
    parse_fathomdem_zip_bounds,
)


def test_parse_fathomdem_zip_bounds() -> None:
    """Test parsing 30x30 bounding box from FathomDEM zip filenames."""
    bounds = parse_fathomdem_zip_bounds("n00e120-n30e150_FathomDEM_v1-0.zip")
    assert bounds is not None
    assert bounds["minx"] == 120.0
    assert bounds["miny"] == 0.0
    assert bounds["maxx"] == 150.0
    assert bounds["maxy"] == 30.0

    bounds_sw = parse_fathomdem_zip_bounds("s30w030-n00e000_FathomDEM_v1-0.zip")
    assert bounds_sw is not None
    assert bounds_sw["minx"] == -30.0
    assert bounds_sw["miny"] == -30.0
    assert bounds_sw["maxx"] == 0.0
    assert bounds_sw["maxy"] == 0.0

    assert parse_fathomdem_zip_bounds("invalid_name.zip") is None


def test_parse_fathomdem_tile_bounds() -> None:
    """Test parsing 1x1 degree bounding box from FathomDEM tile filenames."""
    bounds = parse_fathomdem_tile_bounds("n00e120.tif")
    assert bounds is not None
    assert bounds["minx"] == 120.0
    assert bounds["miny"] == 0.0
    assert bounds["maxx"] == 121.0
    assert bounds["maxy"] == 1.0

    bounds_sw = parse_fathomdem_tile_bounds("s15w045.tif")
    assert bounds_sw is not None
    assert bounds_sw["minx"] == -45.0
    assert bounds_sw["miny"] == -15.0
    assert bounds_sw["maxx"] == -44.0
    assert bounds_sw["maxy"] == -14.0

    assert parse_fathomdem_tile_bounds("invalid_tile.tif") is None


def test_missing_zenodo_access_token_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify missing ZENODO_ACCESS_TOKEN raises ValueError without fallback."""
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("ZENODO_ACCESS_TOKEN", raising=False)
    adapter = FathomDem()
    with patch("geb.build.data_catalog.fathomdem.load_dotenv"):
        with pytest.raises(ValueError, match="ZENODO_ACCESS_TOKEN"):
            adapter._get_access_token()


def test_get_available_files_from_zenodo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test querying and caching available files from Zenodo API for multiple records."""
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("ZENODO_ACCESS_TOKEN", "test_token")
    adapter = FathomDem(
        record_ids=("14511570", "14523356"),
        folder="fathomdem",
        local_version=1,
        filename="placeholder.txt",
        cache="global",
    )

    mock_response_eurasia = {
        "entries": [
            {
                "key": "license.txt",
                "size": 100,
                "links": {
                    "content": "https://zenodo.org/api/records/14511570/files/license.txt/content"
                },
            },
            {
                "key": "n00e120-n30e150_FathomDEM_v1-0.zip",
                "size": 851057674,
                "links": {
                    "content": "https://zenodo.org/api/records/14511570/files/n00e120-n30e150_FathomDEM_v1-0.zip/content"
                },
            },
        ]
    }
    mock_response_americas = {
        "entries": [
            {
                "key": "license.txt",
                "size": 100,
                "links": {
                    "content": "https://zenodo.org/api/records/14523356/files/license.txt/content"
                },
            },
            {
                "key": "n00w060-n30w030_FathomDEM_v1-0.zip",
                "size": 1100000000,
                "links": {
                    "content": "https://zenodo.org/api/records/14523356/files/n00w060-n30w030_FathomDEM_v1-0.zip/content"
                },
            },
        ]
    }

    def mock_urlopen_dispatcher(req: object, timeout: int = 30) -> MagicMock:
        url_str = getattr(req, "full_url", "")
        resp_mock = MagicMock()
        if "14511570" in url_str:
            resp_mock.__enter__.return_value.read.return_value = json.dumps(
                mock_response_eurasia
            ).encode("utf-8")
        else:
            resp_mock.__enter__.return_value.read.return_value = json.dumps(
                mock_response_americas
            ).encode("utf-8")
        return resp_mock

    with patch("urllib.request.urlopen", side_effect=mock_urlopen_dispatcher):
        files = adapter.get_available_files_from_zenodo()

    assert len(files) == 2
    keys = [f["key"] for f in files]
    assert "n00e120-n30e150_FathomDEM_v1-0.zip" in keys
    assert "n00w060-n30w030_FathomDEM_v1-0.zip" in keys

    # Ensure disk cache was written for both records
    assert (adapter.root / "record_14511570_files.json").exists()
    assert (adapter.root / "record_14523356_files.json").exists()

    # Second call should use disk cache without calling urllib
    adapter._files_metadata_cache = None
    with patch(
        "urllib.request.urlopen", side_effect=AssertionError("Should not be called")
    ):
        cached_files = adapter.get_available_files_from_zenodo()
    assert len(cached_files) == 2


def test_fathomdem_fetch_and_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test fetch and read workflows with mock raster data."""
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("ZENODO_ACCESS_TOKEN", "test_token")
    adapter = FathomDem(
        record_ids=("14511570",),
        folder="fathomdem",
        local_version=1,
        filename="placeholder.txt",
        cache="global",
    )
    adapter.logger = logging.getLogger("test")

    # Create synthetic 1x1 degree GeoTIFF tile: n00e120
    x_coords = np.linspace(120.0, 121.0, 10)
    y_coords = np.linspace(1.0, 0.0, 10)
    data = np.ones((1, 10, 10), dtype=np.float32) * 42.0

    da = xr.DataArray(
        data,
        coords={"band": [1], "y": y_coords, "x": x_coords},
        dims=["band", "y", "x"],
    )
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.write_nodata(-9999.0, inplace=True)

    # Pack into zip
    zip_bytes_io = io.BytesIO()
    with zipfile.ZipFile(zip_bytes_io, "w") as zf:
        temp_tif = tmp_path / "n00e120.tif"
        da.rio.to_raster(temp_tif)
        zf.write(temp_tif, arcname="n00e120.tif")

    def mock_fetch_and_save(url: str, file_path: Path, **kwargs: object) -> bool:
        file_path.write_bytes(zip_bytes_io.getvalue())
        return True

    adapter._files_metadata_cache = [
        {
            "key": "n00e120-n30e150_FathomDEM_v1-0.zip",
            "download_url": "https://example.com/test.zip",
            "bounds": {"minx": 120.0, "miny": 0.0, "maxx": 150.0, "maxy": 30.0},
            "size": 1000,
        }
    ]

    target_mask = box(120.2, 0.2, 120.8, 0.8)

    with patch(
        "geb.build.data_catalog.fathomdem.fetch_and_save",
        side_effect=mock_fetch_and_save,
    ):
        res = adapter.fetch(mask=target_mask)

    assert res is adapter
    assert (adapter.root / ".n00e120-n30e150_FathomDEM_v1-0.zip.unpacked").exists()
    assert (adapter.root / "n00e120.tif").exists()
    assert adapter.is_ready

    # Read data for target mask
    read_da = adapter.read(mask=target_mask)
    assert isinstance(read_da, xr.DataArray)
    assert read_da.chunks is not None, (
        "FathomDem read() must return a chunked DataArray"
    )
    assert not np.isnan(read_da.values).all()
    # Check elevation value
    valid_vals = read_da.values[~np.isnan(read_da.values)]
    assert np.allclose(valid_vals, 0.42)

    # Test resampling with resample_chunked (as done in setup_elevation)
    from geb.workflows.raster import resample_chunked

    target_grid = xr.DataArray(
        np.zeros((5, 5), dtype=bool),
        coords={"y": np.linspace(0.8, 0.2, 5), "x": np.linspace(120.2, 120.8, 5)},
        dims=["y", "x"],
    )
    target_grid.rio.write_crs("EPSG:4326", inplace=True)
    target_grid = target_grid.chunk({"x": 5, "y": 5})

    resampled = resample_chunked(read_da, target_grid, method="nearest")
    assert isinstance(resampled, xr.DataArray)
    assert not np.isnan(resampled.values).all()

    # Edge case: unchunked source array raises TypeError with clear message
    unchunked_da = read_da.load()
    with pytest.raises(TypeError, match="Source DataArray must be chunked"):
        resample_chunked(unchunked_da, target_grid, method="nearest")

    # Edge case: read for an area with no tiles raises RuntimeError
    outside_mask = box(10.0, 10.0, 11.0, 11.0)
    with pytest.raises(RuntimeError, match="No extracted FathomDEM tiles found"):
        adapter.read(mask=outside_mask)

    # Edge case: fetch for an area with no intersecting zip packages raises RuntimeError
    with pytest.raises(RuntimeError, match="No FathomDEM zip archives intersect"):
        adapter.fetch(mask=outside_mask)


def test_fathomdem_read_multi_tile_lazy_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that multiple FathomDEM tiles are merged lazily with dask chunks."""
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("ZENODO_ACCESS_TOKEN", "test_token")
    adapter = FathomDem(
        record_ids=("14511570",),
        folder="fathomdem",
        local_version=1,
        filename="placeholder.txt",
        cache="global",
    )
    adapter.logger = logging.getLogger("test")

    # Create two adjacent 1x1 degree GeoTIFF tiles: n00e120 and n00e121
    tile1_path: Path = adapter.root / "n00e120.tif"
    tile2_path: Path = adapter.root / "n00e121.tif"
    adapter.root.mkdir(parents=True, exist_ok=True)

    y_coords = np.linspace(1.0, 0.0, 10)
    for path, x_start in [(tile1_path, 120.0), (tile2_path, 121.0)]:
        x_coords = np.linspace(x_start, x_start + 1.0, 10)
        data = np.ones((1, 10, 10), dtype=np.int32) * 5000
        da = xr.DataArray(
            data,
            coords={"band": [1], "y": y_coords, "x": x_coords},
            dims=["band", "y", "x"],
        )
        da.rio.write_crs("EPSG:4326", inplace=True)
        da.rio.write_nodata(-32768, inplace=True)
        da.rio.to_raster(path)

    # Read mask spanning both tiles
    spanning_mask = box(120.4, 0.2, 121.6, 0.8)
    merged_da = adapter.read(mask=spanning_mask)

    assert isinstance(merged_da, xr.DataArray)
    assert merged_da.chunks is not None, "Merged DataArray must retain dask chunks"
    # Ensure elevation is converted from cm to m (5000 cm -> 50.0 m)
    valid_values = merged_da.values[~np.isnan(merged_da.values)]
    assert np.allclose(valid_values, 50.0)
    assert merged_da.dtype == np.float32
