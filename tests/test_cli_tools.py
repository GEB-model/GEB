"""Tests for the rechunk tool in GEB's CLI."""

from pathlib import Path

import numpy as np
import xarray as xr
from click.testing import CliRunner

from geb.cli import rechunk


def test_rechunk_tool(tmp_path: Path) -> None:
    """Test the rechunk tool using the Click test runner."""
    # Create dummy zarr file
    input_path = tmp_path / "input.zarr"
    output_path = tmp_path / "output.zarr"

    # Create dataset with time, y, x dimensions
    da = xr.DataArray(
        np.random.rand(10, 20, 20).astype(np.float32),
        dims=("time", "y", "x"),
        coords={"time": np.arange(10), "y": np.arange(20), "x": np.arange(20)},
        attrs={"_FillValue": np.nan},
    )
    # Add dummy CRS for rio accessor
    da.rio.write_crs("EPSG:4326", inplace=True)

    # Save input
    da.to_zarr(input_path)

    runner = CliRunner()

    # Test time-optimized
    result = runner.invoke(
        rechunk, [str(input_path), str(output_path), "--how", "time-optimized"]
    )

    assert result.exit_code == 0
    assert output_path.exists()

    # Verify chunks
    ds_out = xr.open_dataarray(output_path, engine="zarr", chunks={})
    # For time-optimized, we expect full time chunk and small spatial chunks
    # Based on rechunk_zarr_file implementation:
    # x_chunk=10, y_chunk=10, time_chunk=da.sizes.get("time", 1) (which is 10)
    assert ds_out.chunks[0][0] == 10  # time  # ty:ignore[not-subscriptable]
    assert ds_out.chunks[1][0] == 10  # y  # ty:ignore[not-subscriptable]
    assert ds_out.chunks[2][0] == 10  # x  # ty:ignore[not-subscriptable]

    # Cleanup
    import shutil

    shutil.rmtree(output_path)

    # Test space-optimized
    result = runner.invoke(
        rechunk, [str(input_path), str(output_path), "--how", "space-optimized"]
    )

    assert result.exit_code == 0
    assert output_path.exists()

    ds_out = xr.open_dataarray(output_path, engine="zarr", chunks={})
    # Based on implementation: x=350, y=350, time=1
    # Since our dimensions are smaller than chunk size, it should take full dim size
    assert ds_out.chunks[0][0] == 1  # time  # ty:ignore[not-subscriptable]
    assert ds_out.chunks[1][0] == 20  # y (min(350, 20))  # ty:ignore[not-subscriptable]
    assert ds_out.chunks[2][0] == 20  # x (min(350, 20))  # ty:ignore[not-subscriptable]
