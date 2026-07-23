"""Live integration tests for the AlphaEarth data catalog adapter."""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.windows import Window

from geb.build.data_catalog import DataCatalog
from geb.build.data_catalog.alphaearth import NODATA_VALUE, AlphaEarth

from ...testconfig import IN_GITHUB_ACTIONS


# A very small area around Amsterdam, chosen to fall well within one
# AlphaEarth COG rather than near a COG or UTM-zone boundary.
BOUNDS = (
    4.865,
    52.333,
    4.870,
    52.338,
)


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Downloads and opens a real AlphaEarth Cloud-Optimized GeoTIFF.",
)
def test_fetch_alphaearth_real_cog() -> None:
    """Download one real AlphaEarth COG and verify that it is readable.

    This is a live integration test rather than an offline unit test. The first
    run downloads the AlphaEarth index and one complete 2024 COG into the
    catalog-managed cache below ``GEB_DATA_ROOT``. Later runs should reuse the
    cached files.
    """
    logger = logging.getLogger("test_fetch_alphaearth_real_cog")

    adapter = DataCatalog(logger=logger).fetch("alphaearth")

    assert isinstance(adapter, AlphaEarth)

    selected = adapter.read(
        years=[2024],
        bounds=BOUNDS,
        dry_run=False,
        max_files=1,
    )

    assert isinstance(selected, gpd.GeoDataFrame)
    assert len(selected) == 1
    assert selected.crs is not None
    assert {"year", "remote_url", "local_path"}.issubset(selected.columns)
    assert int(selected.iloc[0]["year"]) == 2024

    remote_url = str(selected.iloc[0]["remote_url"])
    assert remote_url.startswith(
        "https://storage.googleapis.com/alphaearth_foundations/"
    )

    local_path = Path(str(selected.iloc[0]["local_path"]))

    assert local_path.exists()
    assert local_path.is_file()
    assert local_path.stat().st_size > 0
    assert local_path.suffix.lower() in {".tif", ".tiff"}

    with rasterio.open(local_path) as source:
        assert source.driver == "GTiff"
        assert source.count == 64
        assert source.width > 0
        assert source.height > 0
        assert source.crs is not None
        assert source.transform is not None
        assert set(source.dtypes) == {"int8"}
        assert source.nodata == NODATA_VALUE
        assert source.is_tiled

        # Read only a small centre window. This verifies that the downloaded
        # GeoTIFF is not merely present but can actually be decoded by GDAL.
        window_width = min(64, source.width)
        window_height = min(64, source.height)
        column_offset = max(0, (source.width - window_width) // 2)
        row_offset = max(0, (source.height - window_height) // 2)

        raw = source.read(
            1,
            window=Window(
                column_offset,
                row_offset,
                window_width,
                window_height,
            ),
        )

    assert raw.shape == (window_height, window_width)
    assert raw.dtype == np.int8

    dequantized = adapter.dequantize(raw)

    assert dequantized.shape == raw.shape
    assert dequantized.dtype == np.float32

    valid = raw != NODATA_VALUE
    assert np.isnan(dequantized[~valid]).all()

    if valid.any():
        assert np.isfinite(dequantized[valid]).all()
        assert np.abs(dequantized[valid]).max() <= 1.0
