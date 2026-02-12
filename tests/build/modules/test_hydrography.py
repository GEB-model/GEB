"""Tests for hydrography module."""

import numpy as np
import xarray as xr
from affine import Affine

from geb.build.modules.hydrography import calculate_stream_length


def test_calculate_stream_length_high_res_diagonal() -> None:
    """Test calculate_stream_length_high_res with diagonal flow."""
    # Create simple grid: 1x1 degree cells
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    # 2x2 grid
    # Cell 0,0: Diagonal flow (2 is SE)
    # Cell 0,1: Below threshold
    # The dimensions in calculate_width_m rely on lat/lon
    # Using lat=0 for simplicity (equator)

    ups = xr.DataArray(
        np.array([[2_000_000, 500_000], [500_000, 500_000]], dtype=float),
        coords={"y": [0.5, -0.5], "x": [0.5, 1.5]},
        dims=("y", "x"),
    )
    ups.rio.write_transform(transform, inplace=True)

    # LDD: 2 is SE (diagonal), others irrelevant
    ldd = xr.DataArray(
        np.array([[2, 0], [0, 0]], dtype=np.uint8), coords=ups.coords, dims=ups.dims
    )

    length = calculate_stream_length(ldd, ups, threshold_m2=1_000_000)

    # Check shape
    assert length.shape == ups.shape

    # Calculate expected length for cell at equator
    # 1 degree approx 111320 m
    # Diagonal length = sqrt(111320**2 + 111320**2) approx 157424
    val = length.isel(y=0, x=0).item()
    assert 150000 < val < 160000

    # Check non-stream cell is 0
    assert length.isel(y=0, x=1).item() == 0


def test_calculate_stream_length_high_res_cardinal() -> None:
    """Test calculate_stream_length_high_res with cardinal flow."""
    # Create simple grid: 1x1 degree cells
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    ups = xr.DataArray(
        np.array([[2_000_000, 2_000_000], [0, 0]], dtype=float),
        coords={"y": [0.5, -0.5], "x": [0.5, 1.5]},
        dims=("y", "x"),
    )
    ups.rio.write_transform(transform, inplace=True)

    # LDD: 1 is East (horizontal), 4 is South (vertical)
    ldd = xr.DataArray(
        np.array([[1, 4], [0, 0]], dtype=np.uint8), coords=ups.coords, dims=ups.dims
    )

    length = calculate_stream_length(ldd, ups, threshold_m2=1_000_000)

    # Horizontal
    val_h = length.isel(y=0, x=0).item()
    # Vertical
    val_v = length.isel(y=0, x=1).item()

    # Approx 111km
    assert 110000 < val_h < 112000
    assert 110000 < val_v < 112000
