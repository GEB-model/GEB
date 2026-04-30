"""Tests for raster workflow functions."""

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.transform import from_bounds
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Polygon

from geb.workflows.raster import (
    clip_with_geometry,
    clip_with_grid,
    compress,
    convert_nodata,
    coord_to_pixel,
    coords_to_pixels,
    full_like,
    interpolate_na_2d,
    interpolate_na_along_dim,
    pad_to_grid_alignment,
    pad_xy,
    pixel_to_coord,
    pixels_to_coords,
    rasterize_like,
    reclassify,
    repeat_grid,
    resample_chunked,
    sample_from_map,
)


def test_sample_from_map() -> None:
    """Test the sample_from_map function."""
    # Create a 2D array: y, x
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    # [[ 0.,  1.,  2.,  3.],
    #  [ 4.,  5.,  6.,  7.],
    #  [ 8.,  9., 10., 11.]]

    # Geotransformation: (x_min, x_step, 0, y_max, 0, y_step)
    # Using 1.0 unit steps for simplicity
    gt = (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)

    # Coordinates: (x, y)
    # (0.5, 2.5) -> y_idx = (2.5 - 3.0) / -1.0 = 0.5 -> int(0.5) = 0, x_idx = (0.5 - 0.0) / 1.0 = 0.5 -> int(0.5) = 0 -> Value 0
    # (1.5, 1.5) -> y_idx = (1.5 - 3.0) / -1.0 = 1.5 -> int(1.5) = 1, x_idx = (1.5 - 0.0) / 1.0 = 1.5 -> int(1.5) = 1 -> Value 5
    # (3.5, 0.5) -> y_idx = (0.5 - 3.0) / -1.0 = 2.5 -> int(2.5) = 2, x_idx = (3.5 - 0.0) / 1.0 = 3.5 -> int(3.5) = 3 -> Value 11
    coords = np.array(
        [[0.5, 2.5], [1.5, 1.5], [3.5, 0.5]],
        dtype=np.float64,
    )

    # 1) Test correct values selection
    sampled_values = sample_from_map(array, coords, gt)
    expected_values = np.array([0.0, 5.0, 11.0], dtype=np.float32)
    assert np.allclose(sampled_values, expected_values)

    # 2) Test out of bounds values
    # Coordinate (-1.0, 4.0) is out of bounds
    coords_oob = np.array(
        [[0.5, 2.5], [-1.0, 4.0]],
        dtype=np.float64,
    )

    # Test with out_of_bounds_value provided
    sampled_oob = sample_from_map(array, coords_oob, gt, out_of_bounds_value=-999.0)
    expected_oob = np.array([0.0, -999.0], dtype=np.float32)
    assert np.allclose(sampled_oob, expected_oob)

    # Test that IndexError is raised if out_of_bounds_value is None
    with pytest.raises(IndexError, match="is out of bounds"):
        sample_from_map(array, coords_oob, gt, out_of_bounds_value=None)

    # 3) Test multi-dimensional array (3D)
    # Shape (2, 3, 4)
    array_3d = np.stack([array, array + 100], axis=0)
    sampled_3d = sample_from_map(array_3d, coords, gt)
    # Expected shape (3, 2) -> (num_coords, num_extra_dims)
    expected_3d = np.array(
        [[0.0, 100.0], [5.0, 105.0], [11.0, 111.0]], dtype=np.float32
    )
    assert np.allclose(sampled_3d, expected_3d)

    # 4) Test negative steps (flipped x or y)
    # y_step is already negative (-1.0) in the original test.
    # Let's test positive y_step and negative x_step
    # Array: 3x4
    # gt: (x_max, x_step_neg, 0, y_min, 0, y_step_pos)
    gt_flipped = (4.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    # (3.5, 0.5) -> x_idx = (3.5 - 4.0) / -1.0 = 0.5 -> 0, y_idx = (0.5 - 0.0) / 1.0 = 0.5 -> 0
    # This should sample from array[0, 0]
    coords_flipped = np.array([[3.5, 0.5]], dtype=np.float64)
    sampled_flipped = sample_from_map(array, coords_flipped, gt_flipped)
    assert sampled_flipped[0] == array[0, 0]

    # Test both negative
    gt_both_neg = (4.0, -1.0, 0.0, 3.0, 0.0, -1.0)
    # (3.5, 2.5) -> x_idx = (3.5 - 4.0) / -1.0 = 0.5 -> 0, y_idx = (2.5 - 3.0) / -1.0 = 0.5 -> 0
    coords_both_neg = np.array([[3.5, 2.5]], dtype=np.float64)
    sampled_both_neg = sample_from_map(array, coords_both_neg, gt_both_neg)
    assert sampled_both_neg[0] == array[0, 0]

    # 5) Test strict out-of-bounds with flooring (negative coordinates)
    # gt: (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)
    # A coordinate slightly "left" of x=0 (e.g. -0.1)
    # (x_idx = -0.1 / 1.0 = -0.1).
    # Old logic: int(-0.1) = 0 (in-bounds!).
    # New logic: int(floor(-0.1)) = -1 (out-of-bounds).
    coords_edge = np.array([[-0.1, 2.5], [0.5, 3.1]], dtype=np.float64)
    sampled_edge = sample_from_map(array, coords_edge, gt, out_of_bounds_value=-888.0)
    assert np.all(sampled_edge == -888.0)

    # Test rotated
    gt_rotated = (0.0, 1.0, 0.5, 3.0, -0.5, -1.0)
    with pytest.raises(ValueError, match="Cannot sample from rotated maps"):
        sample_from_map(array, coords, gt_rotated)


def test_pixels_to_coords() -> None:
    """Test conversion from pixel coordinates to geographic coordinates."""
    gt = (-180, 1 / (12 * 360), 0, 90, 0, -1 / (12 * 360))
    n = 100

    xs = np.random.uniform(100, 1000, n)
    ys = np.random.uniform(90, 2300, n)
    pixels = np.squeeze(np.dstack([xs, ys]))
    coords = pixels_to_coords(pixels, gt)
    for i, (x, y) in enumerate(zip(xs, ys)):
        lon, lat = pixel_to_coord(x, y, gt)
        assert lon == coords[i, 0]
        assert lat == coords[i, 1]


def test_coord_to_pixel() -> None:
    """Test conversion from geographic coordinates to pixel coordinates."""
    gt = (
        73.517314369,
        0.00026949458512559436,
        0.0,
        18.438819522,
        0.0,
        -0.0002694945854623959,
    )
    px, py = coord_to_pixel(
        (
            73.517314369 + 10 * 0.00026949458512559436,
            18.438819522 + 56 * -0.0002694945854623959,
        ),
        gt,
    )
    assert px == 10 and py == 56
    px, py = coord_to_pixel(
        (
            73.517314369 + 10 * 0.00026949458512559436 - 0.000001,
            18.438819522 + 56 * -0.0002694945854623959 + 0.000001,
        ),
        gt,
    )
    assert px == 9 and py == 55


def test_coords_to_pixels() -> None:
    """Test conversion from multiple geographic coordinates to pixel coordinates."""
    gt = (-180, 1 / (12 * 360), 0, 90, 0, -1 / (12 * 360))
    n = 100

    lons = np.random.uniform(-180, 180, n)
    lats = np.random.uniform(-80, 80, n)
    coords = np.squeeze(np.dstack([lons, lats]))
    pxs, pys = coords_to_pixels(coords, gt)
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        px, py = coord_to_pixel((lon, lat), gt)
        assert px == pxs[i]
        assert py == pys[i]

    gt = (-10000, 1000, 0, 20000, 0, -1000)
    n = 100

    xs = np.random.uniform(-10000, 10000, n)
    ys = np.random.uniform(10000, 20000, n)
    coords = np.squeeze(np.dstack([xs, ys]))
    pxs, pys = coords_to_pixels(coords, gt)
    for i, (x, y) in enumerate(zip(xs, ys)):
        px, py = coord_to_pixel((x, y), gt)
        assert px == pxs[i]
        assert py == pys[i]


def test_compress() -> None:
    """Test the compress function."""
    array = np.array([[1, 2, 3], [4, 5, 6]])
    mask = np.array([[True, False, True], [False, True, False]])
    result = compress(array, mask)
    expected = np.array([2, 4, 6])
    assert np.array_equal(result, expected)


def test_full_like() -> None:
    """Test the full_like function."""
    data = xr.DataArray(
        np.random.rand(10, 10),
        dims=["y", "x"],
        coords={"y": np.arange(10), "x": np.arange(10)},
        attrs={"some_attr": "value"},
    )
    fill_value = 5.0
    nodata = -9999
    result = full_like(data, fill_value, nodata)
    assert result.shape == data.shape
    assert result.dims == data.dims
    assert np.allclose(result.values, fill_value)
    assert result.attrs["_FillValue"] == nodata
    assert result.coords.equals(data.coords)


def test_repeat_grid() -> None:
    """Test the repeat_grid function."""
    data = np.array([[1, 2], [3, 4]])
    factor = 3
    result = repeat_grid(data, factor)
    expected = np.array(
        [
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4],
            [3, 3, 3, 4, 4, 4],
            [3, 3, 3, 4, 4, 4],
        ]
    )
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "dtype", [bool, np.uint8, np.int32, np.int64, np.float32, np.float64]
)
def test_rasterize_like(dtype: type) -> None:
    """Test the rasterize_like function.

    Args:
        dtype: The data type to test.
    """
    raster: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)),
        dims=["y", "x"],
        coords={"y": np.arange(0.5, 10), "x": np.arange(0.5, 10)},
    )
    raster = raster.rio.write_crs("EPSG:28992")
    raster = raster.rio.set_spatial_dims("x", "y")
    raster = raster.rio.write_transform(from_bounds(0, 10, 10, 0, 10, 10))

    # Create GeoDataFrame with polygons and a value column
    poly1 = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    poly2 = Polygon([(5, 5), (5, 8), (8, 8), (8, 5)])

    if dtype is bool:
        values = [True, True]
    else:
        values = [1, 2]
    gdf = gpd.GeoDataFrame({"value": values}, geometry=[poly1, poly2], crs="EPSG:28992")

    nodata = False if dtype is bool else 255
    result: xr.DataArray = rasterize_like(
        gdf, column="value", raster=raster, dtype=dtype, nodata=nodata, all_touched=True
    )

    assert isinstance(result, xr.DataArray)
    assert result.shape == raster.shape
    assert result.dims == raster.dims
    assert result.dtype == dtype
    assert result.coords.equals(raster.coords)

    if dtype is bool:
        assert np.all(result.values[0:4, 0:4])
        assert np.all(result.values[5:8, 5:8])
        assert result.attrs["_FillValue"] == None
    else:
        assert result.attrs["_FillValue"] == nodata
        assert np.all(result.values[0:4, 0:4] == 1)
        assert np.all(result.values[5:8, 5:8] == 2)
        # Check that areas outside polygons are nodata
        # For example, bottom-left corner
        assert result.values[0, -1] == nodata


def test_rasterize_like_geographic() -> None:
    """Test the rasterize_like function."""
    # test with geographic coordinates
    raster_geo: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)),
        dims=["y", "x"],
        coords={"y": np.linspace(59.5, 49.5, 10), "x": np.linspace(10.5, 20.5, 10)},
    )
    raster_geo = raster_geo.rio.write_crs("EPSG:4326")
    raster_geo = raster_geo.rio.set_spatial_dims("x", "y")
    raster_geo = raster_geo.rio.write_transform(from_bounds(10, 50, 20, 60, 10, 10))

    gdf = gpd.GeoDataFrame(
        {"value": [1, 2]},
        geometry=[
            Polygon([(10, 50), (14, 50), (14, 54), (10, 54)]),
            Polygon([(15, 55), (15, 58), (18, 58), (18, 55)]),
        ],
        crs="EPSG:4326",
    )
    dtype = np.int32
    nodata = -1

    result_geo = rasterize_like(
        gdf,
        column="value",
        raster=raster_geo,
        dtype=dtype,
        nodata=nodata,
        all_touched=False,
    )

    assert isinstance(result_geo, xr.DataArray)
    assert result_geo.shape == raster_geo.shape
    assert result_geo.dims == raster_geo.dims
    assert result_geo.dtype == dtype
    assert result_geo.attrs["_FillValue"] == nodata
    assert result_geo.coords.equals(raster_geo.coords)
    assert np.all(result_geo.values[-1:-4, 0:4] == 1)
    assert np.all(result_geo.values[-5:-8, 5:8] == 2)
    # Check that areas outside polygons are nodata
    # For example, top-left corner
    assert result_geo.values[0, -1] == nodata


def test_reclassify() -> None:
    """Test the reclassify function."""
    classification_dict = {
        1: 10,
        2: 20,
        3: 30,
        4: 40,
        5: 50,
    }

    array = np.array([[1, 2, 3], [4, 5, 1]])
    expected_output = array * 10

    output_dict = reclassify(array, classification_dict, method="dict")
    assert np.array_equal(output_dict, expected_output)

    output_lookup = reclassify(array, classification_dict, method="lookup")
    assert np.array_equal(output_lookup, expected_output)

    xr_array = xr.DataArray(array, dims=["x", "y"])
    output_xr_dict = reclassify(xr_array, classification_dict, method="dict")
    assert np.array_equal(output_xr_dict.values, expected_output)

    output_xr_lookup = reclassify(xr_array, classification_dict, method="lookup")
    assert np.array_equal(output_xr_lookup.values, expected_output)

    classification_dict_with_negative = {
        -1: 10,
        -2: 20,
        -3: 30,
        -4: 40,
        -5: 50,
    }
    array_with_negative = np.array([[-1, -2, -3], [-4, -5, -1]])
    expected_output_with_negative = array_with_negative * 10 * -1

    output_dict_with_negative = reclassify(
        array_with_negative, classification_dict_with_negative, method="dict"
    )
    assert np.array_equal(output_dict_with_negative, expected_output_with_negative)

    # Test if ValueError is raised when using lookup method with negative keys
    with pytest.raises(
        ValueError,
        match="Keys must be positive integers for lookup method, recommend using dict method",
    ):
        reclassify(
            array_with_negative, classification_dict_with_negative, method="lookup"
        )


def test_convert_nodata() -> None:
    """Test the convert_nodata function."""
    data = xr.DataArray(
        np.array([[1, 2, -9999], [4, -9999, 6]]),
        dims=["y", "x"],
        coords={"y": np.arange(2), "x": np.arange(3)},
        attrs={"_FillValue": -9999},
    )
    new_nodata = -1
    result = convert_nodata(data, new_nodata)
    expected = np.array([[1, 2, -1], [4, -1, 6]])
    assert np.array_equal(result.values, expected)
    assert result.attrs["_FillValue"] == new_nodata
    assert result.coords.equals(data.coords)


def test_interpolate_na_2d() -> None:
    """Test the interpolate_na_2d function."""
    # Create a 2D array with NaNs
    data = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": [0, 1, 2], "x": [0, 1, 2]},
        attrs={"_FillValue": np.nan},
    )

    result = interpolate_na_2d(da)

    # Check that NaNs are filled
    assert not np.isnan(result.values).any()
    # The NaN at (0,2) should be interpolated from neighbors, e.g., 2.0 or 6.0
    # The NaN at (1,1) from 2.0, 4.0, 6.0, 8.0
    # Exact values depend on griddata, but ensure no NaNs
    assert result.dims == da.dims
    assert result.coords.equals(da.coords)


def test_interpolate_na_2d_chunked_with_buffer() -> None:
    """Test chunked interpolation with overlap across chunk boundaries."""
    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, np.nan, np.nan],
            [13.0, 14.0, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": [0, 1, 2, 3], "x": [0, 1, 2, 3]},
        attrs={"_FillValue": np.nan},
    )

    expected = interpolate_na_2d(da)
    result = interpolate_na_2d(da.chunk({"y": 2, "x": 2}), buffer=(1, 1)).compute()

    assert np.allclose(result.values, expected.values, equal_nan=True)
    assert result.dims == da.dims
    assert result.coords.equals(da.coords)


def test_interpolate_na_along_dim() -> None:
    """Test the interpolate_na_along_dim function."""
    # Create a 3D array (time, y, x) with NaNs in spatial dims
    data = np.array(
        [
            [[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]],  # time 0
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],  # time 1
        ]
    )
    da = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": [0, 1], "y": [0, 1], "x": [0, 1, 2]},
        attrs={"_FillValue": np.nan},
    )

    result = interpolate_na_along_dim(da)

    # Check that NaNs are filled
    assert not np.isnan(result.values).any()
    assert result.dims == da.dims
    assert result.coords.equals(da.coords)


def test_interpolate_na_alignment() -> None:
    """Test that interpolate_na_along_dim aligns with interpolate_na_2d applied per slice."""
    # Create a 3D array (time, y, x) with NaNs
    data = np.array(
        [
            [[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]],  # time 0
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],  # time 1
        ]
    )
    da = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": [0, 1], "y": [0, 1], "x": [0, 1, 2]},
        attrs={"_FillValue": np.nan},
    )

    # Apply along time
    result_along_time = interpolate_na_along_dim(da)

    # Apply 2d to each slice manually
    slices = []
    for t in range(da.sizes["time"]):
        slice_da = da.isel(time=t)
        filled_slice = interpolate_na_2d(slice_da)
        slices.append(filled_slice.values)

    manual_result = xr.DataArray(
        np.stack(slices, axis=0), dims=da.dims, coords=da.coords
    )

    # Check they are equal
    assert np.allclose(result_along_time.values, manual_result.values, equal_nan=True)
    assert result_along_time.dims == manual_result.dims
    assert result_along_time.coords.equals(manual_result.coords)


def test_interpolate_na_2d_missing_fillvalue() -> None:
    """Test that interpolate_na_2d raises ValueError when _FillValue is missing."""
    data = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    da = xr.DataArray(data, dims=["y", "x"], coords={"y": [0, 1, 2], "x": [0, 1, 2]})
    with pytest.raises(ValueError, match="DataArray must have '_FillValue' attribute"):
        interpolate_na_2d(da)


def test_interpolate_na_2d_integer_fillvalue() -> None:
    """Test the interpolate_na_2d function with integer _FillValue."""
    # Create a 2D array with -9999 as missing value
    data = np.array([[1.0, 2.0, -9999.0], [4.0, -9999.0, 6.0], [7.0, 8.0, 9.0]])
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": [0, 1, 2], "x": [0, 1, 2]},
        attrs={"_FillValue": -9999},
    )
    result = interpolate_na_2d(da)
    # Check that -9999 values are filled
    assert not np.any(result.values == -9999)
    assert result.dims == da.dims
    assert result.coords.equals(da.coords)


def test_interpolate_na_2d_invalid_buffer() -> None:
    """Test that interpolate_na_2d rejects invalid overlap settings."""
    data = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": [0, 1], "x": [0, 1]},
        attrs={"_FillValue": np.nan},
    )

    with pytest.raises(ValueError, match="buffer"):
        interpolate_na_2d(da, buffer=-1)


def test_interpolate_na_2d_buffer_larger_than_array() -> None:
    """Test that oversized overlap buffers are capped for small chunked arrays."""
    data = np.array(
        [[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": [0, 1, 2], "x": [0, 1, 2]},
        attrs={"_FillValue": np.nan},
    )

    expected = interpolate_na_2d(da)
    result = interpolate_na_2d(da.chunk({"y": 2, "x": 2}), buffer=1000).compute()

    assert np.allclose(result.values, expected.values, equal_nan=True)


def test_interpolate_na_along_dim_missing_fillvalue() -> None:
    """Test that interpolate_na_along_dim raises ValueError when _FillValue is missing."""
    data = np.array(
        [
            [[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]],  # time 0
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],  # time 1
        ]
    )
    da = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": [0, 1], "y": [0, 1], "x": [0, 1, 2]},
    )
    with pytest.raises(ValueError, match="DataArray must have '_FillValue' attribute"):
        interpolate_na_along_dim(da)


def test_interpolate_na_along_dim_integer_fillvalue() -> None:
    """Test the interpolate_na_along_dim function with integer _FillValue."""
    data = np.array(
        [
            [[1.0, 2.0, -9999.0], [4.0, -9999.0, 6.0]],  # time 0
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],  # time 1
        ]
    )
    da = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": [0, 1], "y": [0, 1], "x": [0, 1, 2]},
        attrs={"_FillValue": -9999},
    )
    result = interpolate_na_along_dim(da)
    # Check that -9999 values are filled
    assert not np.any(result.values == -9999)
    assert result.dims == da.dims


def test_interpolate_na_2d_with_mask() -> None:
    """Test the interpolate_na_2d function with an optional mask."""
    # Create a 5x5 array with NaN in the middle and some other cells
    data = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, np.nan, np.nan, np.nan, 1],
            [1, np.nan, 2, np.nan, 1],
            [1, np.nan, np.nan, np.nan, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    da = xr.DataArray(data, dims=("y", "x"), attrs={"_FillValue": np.nan})

    # Mask that only includes the inner 3x3
    mask_data = np.zeros((5, 5), dtype=bool)
    mask_data[1:4, 1:4] = True
    mask = xr.DataArray(mask_data, dims=("y", "x"))

    # Interpolate with mask
    # The cell at (2,2) is 2 and is within the mask.
    # The cells around it (1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,3) are NaN and within mask.
    # They should be filled with 2.
    # Outer cells should NOT be used for interpolation since they are outside the mask.

    result = interpolate_na_2d(da, mask=mask)

    # Inner 3x3 (except center which was already 2) should be 2
    expected_inner = np.full((3, 3), 2.0)
    np.testing.assert_array_equal(result.values[1:4, 1:4], expected_inner)

    # Outer cells should remain 1 (they weren't NaN, but even if they were,
    # they shouldn't be filled if they are outside the mask)
    assert result.values[0, 0] == 1
    assert result.values[4, 4] == 1


def test_interpolate_na_along_dim_with_mask() -> None:
    """Test the interpolate_na_along_dim function with an optional mask."""
    # Create a 2x3x3 array (time, y, x)
    data = np.array(
        [
            [[1, 1, 1], [1, np.nan, 1], [1, 1, 1]],
            [[3, 3, 3], [3, 4, 3], [3, 3, 3]],
        ],
        dtype=np.float32,
    )
    da = xr.DataArray(data, dims=("time", "y", "x"), attrs={"_FillValue": np.nan})

    # Mask that excludes the center cell (1, 1)
    mask_data = np.ones((3, 3), dtype=bool)
    mask_data[1, 1] = False  # Do NOT interpolate center cell
    mask = xr.DataArray(mask_data, dims=("y", "x"))

    result = interpolate_na_along_dim(da, mask=mask)

    # Center cell at time 0 was NaN and mask was False there.
    # It should still be NaN.
    assert np.isnan(result.values[0, 1, 1])

    # If we set mask to True at (1,1)
    mask_data[1, 1] = True
    mask = xr.DataArray(mask_data, dims=("y", "x"))
    result = interpolate_na_along_dim(da, mask=mask)

    # Now it should be filled. At time 0, center cell neighbors are 1.
    assert result.values[0, 1, 1] == 1.0

    assert result.coords.equals(da.coords)


@pytest.mark.parametrize(
    "pad_bounds",
    [
        (5, 5, 15, 15),  # No padding
        (0, 0, 20, 20),  # Padding on all sides
        (5, 5, 20, 15),  # Padding on the right
        (0, 5, 15, 15),  # Padding on the left
        (5, 0, 15, 15),  # Padding on the bottom
        (5, 5, 15, 20),  # Padding on the top
        (0, 0, 15, 15),  # Padding left and bottom
    ],
)
def test_pad_xy(pad_bounds: tuple[int, int, int, int]) -> None:
    """Test the pad_xy function under various conditions.

    Args:
        pad_bounds: The bounds to pad to (minx, miny, maxx, maxy).
    """
    original_da = xr.DataArray(
        np.ones((10, 10)),
        dims=["y", "x"],
        coords={"y": np.arange(14.5, 4.5, -1), "x": np.arange(5.5, 15.5, 1)},
        attrs={"_FillValue": -9999},
    )
    original_da.rio.write_crs("EPSG:28992", inplace=True)
    original_da.rio.write_transform(from_bounds(5, 5, 15, 15, 10, 10), inplace=True)

    minx, miny, maxx, maxy = pad_bounds
    constant_values = -1.0

    # Get expected from rioxarray
    expected_padded = original_da.rio.pad_box(
        minx, miny, maxx, maxy, constant_values=constant_values
    )

    padded_da, returned_slice = pad_xy(
        original_da,
        minx,
        miny,
        maxx,
        maxy,
        constant_values=constant_values,
        return_slice=True,
    )

    # Check shape
    assert padded_da.shape == expected_padded.shape

    # Check bounds
    assert np.allclose(padded_da.rio.bounds(), expected_padded.rio.bounds())

    # Check x and y coordinates are allclose
    assert np.allclose(padded_da.x.values, expected_padded.x.values)
    assert np.allclose(padded_da.y.values, expected_padded.y.values)

    # Check coordinate differences (resolution)
    # np.diff should be constant and equal to original resolution
    x_diff = np.diff(padded_da.x.values)
    y_diff = np.diff(padded_da.y.values)
    assert np.allclose(x_diff, original_da.rio.resolution()[0])
    assert np.allclose(y_diff, original_da.rio.resolution()[1])

    # Check that the original data is preserved
    original_data_in_padded = padded_da.isel(returned_slice)
    assert np.allclose(original_data_in_padded.values, original_da.values)
    assert (original_data_in_padded.x.values == original_da.x.values).all()
    assert (original_data_in_padded.y.values == original_da.y.values).all()

    # Check that padded areas have the constant value
    # Create a mask of the original data area and invert it to get the padded area
    mask = np.zeros(padded_da.shape, dtype=bool)
    mask[returned_slice["y"], returned_slice["x"]] = True
    assert np.allclose(padded_da.values[~mask], constant_values)


@pytest.mark.parametrize(
    "pad_bounds",
    [
        (5, 40, 15, 50),  # No padding
        (0, 0, 20, 20),  # Padding on all sides
        (5, 40, 20, 50),  # Padding on the right
        (0, 40, 15, 50),  # Padding on the left
        (5, 0, 15, 50),  # Padding on the bottom
        (5, 40, 15, 55),  # Padding on the top
        (0, 0, 15, 50),  # Padding left and bottom
    ],
)
def test_pad_xy_geographical(pad_bounds: tuple[int, int, int, int]) -> None:
    """Test the pad_xy function with geographical coordinates (y descending).

    Args:
        pad_bounds: The bounds to pad to (minx, miny, maxx, maxy).
    """
    # Geographical coordinates: latitude descending, longitude ascending
    original_da = xr.DataArray(
        np.ones((10, 10)),
        dims=["y", "x"],
        coords={"y": np.arange(50.5, 40.5, -1), "x": np.arange(5.5, 15.5, 1)},
        attrs={"_FillValue": -9999},
    )
    original_da.rio.write_crs("EPSG:4326", inplace=True)
    original_da.rio.write_transform(from_bounds(5, 40, 15, 50, 10, 10), inplace=True)

    minx, miny, maxx, maxy = pad_bounds
    constant_values = -1.0

    # Get expected from rioxarray
    expected_padded = original_da.rio.pad_box(
        minx, miny, maxx, maxy, constant_values=constant_values
    )

    padded_da, returned_slice = pad_xy(
        original_da,
        minx,
        miny,
        maxx,
        maxy,
        constant_values=constant_values,
        return_slice=True,
    )

    # Check shape
    assert padded_da.shape == expected_padded.shape

    # Check bounds
    assert np.allclose(padded_da.rio.bounds(), expected_padded.rio.bounds())

    # Check x and y coordinates are allclose
    assert np.allclose(padded_da.x.values, expected_padded.x.values)
    assert np.allclose(padded_da.y.values, expected_padded.y.values)

    # Check coordinate differences (resolution)
    # np.diff should be constant and equal to original resolution
    x_diff = np.diff(padded_da.x.values)
    y_diff = np.diff(padded_da.y.values)
    assert np.allclose(x_diff, original_da.rio.resolution()[0])
    assert np.allclose(y_diff, original_da.rio.resolution()[1])

    # Check that the original data is preserved
    original_data_in_padded = padded_da.isel(returned_slice)
    assert np.allclose(original_data_in_padded.values, original_da.values)

    # Check that padded areas have the constant value
    # Create a mask of the original data area and invert it to get the padded area
    mask = np.zeros(padded_da.shape, dtype=bool)
    mask[returned_slice["y"], returned_slice["x"]] = True
    assert np.allclose(padded_da.values[~mask], constant_values)


def _edge_from_centers(values: np.ndarray) -> float:
    step = float(values[1] - values[0]) if values.size > 1 else 0.0
    return float(values[0] - step / 2)


def _bottom_edge_from_centers(values: np.ndarray) -> float:
    step = float(values[1] - values[0]) if values.size > 1 else 0.0
    return float(values[-1] + step / 2)


def test_pad_to_grid_alignment_projected() -> None:
    """Test grid-aligned padding for a projected raster."""
    da = xr.DataArray(
        np.ones((4, 4)),
        dims=["y", "x"],
        coords={
            "y": np.arange(1.875, 0.875, -0.25),
            "x": np.arange(0.375, 1.375, 0.25),
        },
    )
    da.rio.write_crs("EPSG:28992", inplace=True)
    da.rio.write_transform(from_bounds(0.25, 0.75, 1.25, 1.75, 4, 4), inplace=True)

    padded = pad_to_grid_alignment(da, grid_size_multiplier=5, constant_values=0)

    coarse_step = 0.25 * 5
    left_edge = _edge_from_centers(padded.x.values)
    bottom_edge = _bottom_edge_from_centers(padded.y.values)

    assert np.isclose(left_edge % coarse_step, 0.0)
    assert np.isclose(bottom_edge % coarse_step, 0.0)
    assert padded.sizes["x"] % 5 == 0
    assert padded.sizes["y"] % 5 == 0


def test_pad_to_grid_alignment_geographic() -> None:
    """Test grid-aligned padding for geographic rasters with descending y."""
    da = xr.DataArray(
        np.ones((6, 7)),
        dims=["y", "x"],
        coords={
            "y": np.arange(50.875, 49.375, -0.25),
            "x": np.arange(0.125, 1.875, 0.25),
        },
    )
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.write_transform(from_bounds(0.0, 49.25, 1.75, 51.0, 7, 6), inplace=True)

    padded = pad_to_grid_alignment(da, grid_size_multiplier=4, constant_values=-1)

    coarse_step = 0.25 * 4
    left_edge = _edge_from_centers(padded.x.values)
    bottom_edge = _bottom_edge_from_centers(padded.y.values)

    assert np.isclose(left_edge % coarse_step, 0.0)
    assert np.isclose(bottom_edge % coarse_step, 0.0)
    assert padded.sizes["x"] % 4 == 0
    assert padded.sizes["y"] % 4 == 0


def test_resample_chunked() -> None:
    """Test resample_chunked against scipy RegularGridInterpolator."""
    # Create source grid
    src_x = np.linspace(0, 10, 11)
    src_y = np.linspace(0, 10, 11)
    X, Y = np.meshgrid(src_x, src_y)
    # Create a simple function z = x + y
    data = (X + Y).astype(np.float32)

    source = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": src_y, "x": src_x},
        name="test_data",
        attrs={"_FillValue": np.nan},
    )
    # Add CRS (required by resample_chunked)
    source.rio.write_crs("EPSG:4326", inplace=True)

    # Create target grid (finer resolution)
    tgt_x = np.linspace(0, 10, 21)
    tgt_y = np.linspace(0, 10, 21)

    target = xr.DataArray(
        np.zeros((21, 21)),
        dims=("y", "x"),
        coords={"y": tgt_y, "x": tgt_x},
    )
    # Add CRS (required by resample_chunked)
    target.rio.write_crs("EPSG:4326", inplace=True)

    # Chunk the target (required by resample_chunked)
    target = target.chunk({"y": 10, "x": 10})
    # Chunk the source (required by resample_chunked)
    source = source.chunk({"y": 10, "x": 10})

    # Run resample_chunked
    resampled = resample_chunked(source, target, method="bilinear")

    # Run scipy RegularGridInterpolator for comparison
    # Note: RegularGridInterpolator expects (y, x) order for points if grid is defined as (y, x)
    interp = RegularGridInterpolator(
        (src_y, src_x), data, method="linear", bounds_error=False, fill_value=np.nan
    )

    tgt_X, tgt_Y = np.meshgrid(tgt_x, tgt_y)
    # Create points array of shape (N, 2) where columns are (y, x)
    points = np.column_stack((tgt_Y.ravel(), tgt_X.ravel()))

    expected_data = interp(points).reshape(21, 21)

    # Compare results
    # Allow small differences due to float precision and implementation details
    # Also mask NaNs because pyresample might handle boundaries differently than scipy
    mask = ~np.isnan(resampled.values) & ~np.isnan(expected_data)
    np.testing.assert_allclose(
        resampled.values[mask], expected_data[mask], rtol=1e-5, atol=1e-5
    )

    # Verify metadata
    assert resampled.rio.crs == source.rio.crs
    assert resampled.dims == target.dims
    assert resampled.shape == target.shape


@pytest.mark.parametrize("y_step", [-1, 1])
@pytest.mark.parametrize("grid_size", [10, 100])
def test_clip_with_grid(y_step: int, grid_size: int) -> None:
    """Test clip_with_grid with positive and negative y steps and different grid sizes."""
    # Create grid
    ny, nx = grid_size, grid_size

    # Coordinates
    if y_step < 0:
        ys = np.arange(ny)[::-1]  # decreasing coordinates
    else:
        ys = np.arange(ny)  # increasing coordinates

    xs = np.arange(nx)

    data = np.random.rand(ny, nx)
    da = xr.DataArray(data, coords={"y": ys, "x": xs}, dims=("y", "x"))

    # Create a mask that selects a subset
    # For size 10: 3:7 (size 4)
    # For size 100: 30:70 (size 40)
    start_idx = int(0.3 * grid_size)
    end_idx = int(0.7 * grid_size)

    mask_data = np.zeros((ny, nx), dtype=bool)
    mask_data[start_idx:end_idx, start_idx:end_idx] = True
    mask = xr.DataArray(mask_data, coords={"y": ys, "x": xs}, dims=("y", "x"))

    clipped_ds, bounds = clip_with_grid(da, mask)

    # Expected bounds indices
    assert bounds["y"] == slice(start_idx, end_idx)
    assert bounds["x"] == slice(start_idx, end_idx)

    assert clipped_ds.shape == (end_idx - start_idx, end_idx - start_idx)
    np.testing.assert_array_equal(
        clipped_ds.values, data[start_idx:end_idx, start_idx:end_idx]
    )

    # Check coordinates of clipped result
    np.testing.assert_array_equal(clipped_ds.y.values, ys[start_idx:end_idx])
    np.testing.assert_array_equal(clipped_ds.x.values, xs[start_idx:end_idx])

    # Check that sum is preserved (since mask is rectangular and matches bounds)
    assert np.isclose(clipped_ds.sum(), da.where(mask).sum())


@pytest.mark.parametrize("dask", [True, False])
def test_clip_with_geometry(dask: bool) -> None:
    """Test the clip_with_geometry function."""
    # Create a dummy DataArray
    data = np.random.rand(10, 10).astype(np.float32)
    # y coordinates should be decreasing for standard geo alignment (top to bottom)
    y = np.linspace(60, 50, 10)
    x = np.linspace(10, 20, 10)
    da = xr.DataArray(
        data,
        coords={"y": y, "x": x},
        dims=("y", "x"),
        attrs={"_FillValue": np.nan},
    )
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.set_spatial_dims("x", "y", inplace=True)
    # Important: rio.clip needs a transform to properly identify pixels
    # We'll use the one from_bounds (left, bottom, right, top, width, height)
    da.rio.write_transform(from_bounds(10, 50, 20, 60, 10, 10), inplace=True)

    if dask:
        da_test = da.chunk({"y": 6, "x": 6})
    else:
        da_test = da

    # 1) Test clipping with a polygon that contains everything
    full_poly = Polygon([(5, 45), (25, 45), (25, 65), (5, 65)])
    gdf_full = gpd.GeoDataFrame(geometry=[full_poly], crs="EPSG:4326")
    clipped_full = clip_with_geometry(da_test, gdf_full).compute()
    expected_full = clip_with_geometry(da, gdf_full)
    assert clipped_full.shape == da.shape
    assert np.allclose(clipped_full.values, expected_full.values, equal_nan=True)

    # 2) Test clipping with a polygon that contains only part of the data
    part_poly = Polygon([(12, 52), (18, 52), (18, 58), (12, 58)])
    gdf_part = gpd.GeoDataFrame(geometry=[part_poly], crs="EPSG:4326")
    clipped_part = clip_with_geometry(da_test, gdf_part, drop=True).compute()
    expected_part = clip_with_geometry(da, gdf_part, drop=True)
    assert clipped_part.shape[0] < da.shape[0]

    assert (clipped_part.x == expected_part.x).all()
    assert (clipped_part.y == expected_part.y).all()
    assert np.allclose(clipped_part.values, expected_part.values, equal_nan=True)

    # 3) Test all_touched=True vs False
    small_poly = Polygon([(10.2, 59.2), (10.6, 59.2), (10.6, 59.6), (10.2, 59.6)])
    gdf_small = gpd.GeoDataFrame(geometry=[small_poly], crs="EPSG:4326")

    clipped_untouched = clip_with_geometry(
        da_test, gdf_small, all_touched=False
    ).compute()
    expected_untouched = clip_with_geometry(da, gdf_small, all_touched=False)
    assert np.allclose(
        clipped_untouched.values, expected_untouched.values, equal_nan=True
    )

    clipped_touched = clip_with_geometry(da_test, gdf_small, all_touched=True).compute()
    expected_touched = clip_with_geometry(da, gdf_small, all_touched=True)
    assert np.allclose(clipped_touched.values, expected_touched.values, equal_nan=True)

    # 4) Test MultiPolygon (Non-contiguous area)
    poly1 = Polygon([(11, 51), (13, 51), (13, 53), (11, 53)])
    poly2 = Polygon([(17, 57), (19, 57), (19, 59), (17, 59)])
    gdf_multi = gpd.GeoDataFrame(geometry=[poly1, poly2], crs="EPSG:4326")

    clipped_multi = clip_with_geometry(da_test, gdf_multi).compute()
    expected_multi = clip_with_geometry(da, gdf_multi)
    assert np.allclose(clipped_multi.values, expected_multi.values, equal_nan=True)

    # 5) Test error handling: missing coords
    da_no_coords = xr.DataArray(data)
    with pytest.raises(ValueError, match="DataArray must have x and y coordinates"):
        clip_with_geometry(da_no_coords, gdf_full)

    # 6) Test error handling: missing CRS
    da_no_crs = xr.DataArray(data, coords={"y": y, "x": x}, dims=("y", "x"))
    with pytest.raises(ValueError, match="DataArray must have a CRS defined"):
        clip_with_geometry(da_no_crs, gdf_full)

    # 7) Test error handling: CRS mismatch
    gdf_wrong_crs = gpd.GeoDataFrame(geometry=[full_poly], crs="EPSG:3857")
    with pytest.raises(ValueError, match="Geometry must be in the same CRS"):
        clip_with_geometry(da_test, gdf_wrong_crs)
