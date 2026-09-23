"""Unit tests for the MSWEP Precipitation data catalog adapter."""

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geb.build.data_catalog import DataCatalog
from geb.build.data_catalog.mswep import (
    MSWEPPrecipitation,
    _extract_folder_id,
    _extract_spatial_coords,
)
from geb.workflows.io import write_zarr


def test_extract_folder_id() -> None:
    """Test extracting Google Drive folder IDs from various URL formats and raw IDs."""
    raw_id = "dummy_folder_id_xyz123"
    url_standard = f"https://drive.google.com/drive/folders/{raw_id}"
    url_with_params = f"https://drive.google.com/drive/u/0/folders/{raw_id}?usp=sharing"
    url_with_id_param = f"https://drive.google.com/open?id={raw_id}"
    url_file_view = f"https://drive.google.com/file/d/{raw_id}/view?usp=sharing"
    url_trailing_slash = f"https://drive.google.com/drive/folders/{raw_id}/"
    raw_with_query = f"{raw_id}?usp=sharing"
    quoted = f'"{url_standard}"'

    assert _extract_folder_id(raw_id) == raw_id
    assert _extract_folder_id(url_standard) == raw_id
    assert _extract_folder_id(url_with_params) == raw_id
    assert _extract_folder_id(url_with_id_param) == raw_id
    assert _extract_folder_id(url_file_view) == raw_id
    assert _extract_folder_id(url_trailing_slash) == raw_id
    assert _extract_folder_id(raw_with_query) == raw_id
    assert _extract_folder_id(quoted) == raw_id


def test_mswep_missing_url_raises_helpful_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that an informative ValueError is raised when MSWEP_URL is not set."""
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("MSWEP_URL", raising=False)

    adapter = MSWEPPrecipitation(
        folder="mswep_test",
        filename="precipitation.zarr",
        local_version=1,
        cache="global",
        folder_id=None,
    )
    adapter.logger = logging.getLogger("test_mswep")

    with patch("geb.build.data_catalog.mswep.load_dotenv"):
        with pytest.raises(ValueError) as exc_info:
            adapter.fetch(start_date="2025-01-01", end_date="2025-01-31")

        err_msg: str = str(exc_info.value)
        assert "MSWEP_URL" in err_msg
        assert "https://www.gloh2o.org/mswep/" in err_msg
        assert "MSWEP_V316_test/Past/Hourly" in err_msg


def test_mswep_folder_id_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that providing a folder_id sets the adapter folder_id."""
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    folder_id = "dummy_folder_id_xyz123"
    adapter = MSWEPPrecipitation(
        folder="mswep_test",
        filename="precipitation.zarr",
        local_version=1,
        cache="global",
        folder_id=folder_id,
    )
    assert adapter.folder_id == folder_id


def test_data_catalog_mswep_entry() -> None:
    """Test that DataCatalog includes mswep_precipitation with adapter."""
    logger = logging.getLogger("test_catalog")
    catalog = DataCatalog(logger=logger)

    assert "mswep_precipitation" in catalog.catalog
    entry = catalog.catalog["mswep_precipitation"]
    assert isinstance(entry["adapter"], MSWEPPrecipitation)
    assert entry["source"]["name"] == "MSWEP Precipitation"


def test_extract_spatial_coords() -> None:
    """Test extracting spatial coordinates directly from NetCDF dataset and DataArray."""
    lats = np.linspace(-89.95, 89.95, 1800)
    lons = np.linspace(-179.95, 179.95, 3600)
    ds = xr.Dataset(
        data_vars={"precipitation": (("lat", "lon"), np.zeros((1800, 3600)))},
        coords={"lat": lats, "lon": lons},
    )

    y, x = _extract_spatial_coords(ds)
    np.testing.assert_array_equal(y, lats)
    np.testing.assert_array_equal(x, lons)

    da = xr.DataArray(
        np.zeros((1800, 3600)),
        dims=("lat", "lon"),
        coords={"lat": lats, "lon": lons},
    )
    y_da, x_da = _extract_spatial_coords(da)
    np.testing.assert_array_equal(y_da, lats)
    np.testing.assert_array_equal(x_da, lons)


def test_mswep_fetch_succeeds_without_url_when_data_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that fetch succeeds without MSWEP_URL when all required monthly stores exist."""
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("MSWEP_URL", raising=False)

    adapter = MSWEPPrecipitation(
        folder="mswep_test",
        filename="precipitation.zarr",
        local_version=1,
        cache="global",
        folder_id=None,
    )
    adapter.logger = logging.getLogger("test_mswep")

    # Create dummy complete zarr for 2025-01 (31 days * 24 hours = 744 hours)
    expected_hours: int = 31 * 24
    time_coords: pd.DatetimeIndex = pd.date_range(
        "2025-01-01 00:00:00", "2025-01-31 23:00:00", freq="1h"
    )
    dummy_da: xr.DataArray = xr.DataArray(
        np.zeros((expected_hours, 2, 2), dtype=np.uint16),
        dims=("time", "y", "x"),
        coords={
            "time": time_coords,
            "y": [10.0, 11.0],
            "x": [20.0, 21.0],
        },
        attrs={"_FillValue": 65535},
    )
    month_path: Path = adapter._month_path(2025, 1)
    month_path.parent.mkdir(parents=True, exist_ok=True)
    write_zarr(dummy_da, month_path, crs=4326)

    # precipitation_mask.zarr does not exist and MSWEP_URL is unset
    assert not adapter.mask_path.exists()

    with patch("geb.build.data_catalog.mswep.load_dotenv"):
        result = adapter.fetch(start_date="2025-01-01", end_date="2025-01-31")
        assert result is adapter


def test_mswep_read_missing_month_raises_filenotfounderror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that read raises FileNotFoundError when a monthly store is missing."""
    monkeypatch.setenv("GEB_DATA_ROOT", str(tmp_path))
    adapter = MSWEPPrecipitation(
        folder="mswep_test",
        filename="precipitation.zarr",
        local_version=1,
        cache="global",
        folder_id=None,
    )
    adapter.logger = logging.getLogger("test_mswep")

    with pytest.raises(FileNotFoundError) as exc_info:
        adapter.read(
            start_date="2025-01-01",
            end_date="2025-01-02",
            bounds=(19.0, 9.0, 22.0, 12.0),
        )
    assert "Missing MSWEP precipitation data for 2025-01" in str(exc_info.value)
