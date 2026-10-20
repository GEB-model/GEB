"""Tests for hydrography module."""

import numpy as np
import xarray as xr
from affine import Affine

from geb.build.modules.hydrography import (
    calculate_dem_floodplain_width,
    calculate_stream_length,
)


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


def test_calculate_dem_floodplain_width() -> None:
    """Test calculate_dem_floodplain_width extracts floodplain width per river reach using subbasins."""
    import pyflwdir

    # 4x4 high-res grid with 2x2 coarsening factor (ldd_scale_factor=2) -> 2x2 low-res grid
    # Low valley in columns 1-2, higher hills on borders
    elev_vals = np.array(
        [
            [12.0, 5.0, 5.0, 12.0],
            [11.0, 4.0, 4.0, 11.0],
            [10.0, 3.0, 3.0, 10.0],
            [9.0, 2.0, 2.0, 9.0],
        ],
        dtype=np.float32,
    )
    ldd_vals = np.array(
        [
            [2, 2, 2, 4],
            [2, 2, 2, 4],
            [2, 2, 2, 4],
            [5, 5, 5, 5],
        ],
        dtype=np.uint8,
    )

    transform = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 0.0)
    elev_high_res = xr.DataArray(
        elev_vals,
        coords={"y": [0.0, -0.01, -0.02, -0.03], "x": [0.0, 0.01, 0.02, 0.03]},
        dims=("y", "x"),
    )
    elev_high_res.rio.write_transform(transform, inplace=True)

    flw_high_res = pyflwdir.from_array(ldd_vals, ftype="ldd", transform=transform)
    upstream_area_vals = flw_high_res.upstream_area(unit="m2").astype(np.float32)
    upstream_area_high_res = xr.DataArray(
        upstream_area_vals, coords=elev_high_res.coords, dims=elev_high_res.dims
    )
    upstream_area_high_res.rio.write_transform(transform, inplace=True)

    # Low-res coordinates
    cell_area_low_res = xr.DataArray(
        np.full((2, 2), 4_000_000.0, dtype=np.float32),
        coords={"y": [-0.005, -0.025], "x": [0.005, 0.025]},
        dims=("y", "x"),
    )
    river_length_low_res = xr.DataArray(
        np.full((2, 2), 2000.0, dtype=np.float32),
        coords=cell_area_low_res.coords,
        dims=cell_area_low_res.dims,
    )
    idxs_outflow_low_res = xr.DataArray(
        np.array([[5, 7], [13, 15]], dtype=np.int64),
        coords=cell_area_low_res.coords,
        dims=cell_area_low_res.dims,
    )
    # Right column (col 1) has river segments; left column (col 0) is non-river
    river_raster_low_res = np.array([[-1, 101], [-1, 102]], dtype=np.int32)

    fp_width = calculate_dem_floodplain_width(
        flow_raster_high_res=flw_high_res,
        elevation_high_res=elev_high_res,
        upstream_area_high_res_m2=upstream_area_high_res,
        idxs_outflow_low_res=idxs_outflow_low_res,
        river_raster_low_res=river_raster_low_res,
        cell_area_low_res=cell_area_low_res,
        river_length_low_res=river_length_low_res,
        ldd_scale_factor=2,
    )

    assert fp_width.shape == (2, 2)
    # Non-river cells must be NaN
    assert np.isnan(fp_width.values[0, 0])
    assert np.isnan(fp_width.values[1, 0])
    # River cells must have valid floodplain widths
    assert not np.isnan(fp_width.values[0, 1])
    assert not np.isnan(fp_width.values[1, 1])
    assert (fp_width.values[:, 1] >= 0.0).all()
