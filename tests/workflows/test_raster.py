"""Tests for raster workflow functions."""

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

from geb.workflows.raster import (
    compress,
    convert_nodata,
    coord_to_pixel,
    coords_to_pixels,
    full_like,
    interpolate_na_2d,
    interpolate_na_along_time_dim,
    pad_xy,
    pixel_to_coord,
    pixels_to_coords,
    rasterize_like,
    reclassify,
    repeat_grid,
)


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
    "dtype", [np.uint8, np.int32, np.int64, np.float32, np.float64]
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
    gdf = gpd.GeoDataFrame({"value": [1, 2]}, geometry=[poly1, poly2], crs="EPSG:28992")

    nodata = 255
    result = rasterize_like(
        gdf, column="value", raster=raster, dtype=dtype, nodata=nodata, all_touched=True
    )

    assert isinstance(result, xr.DataArray)
    assert result.shape == raster.shape
    assert result.dims == raster.dims
    assert result.dtype == dtype
    assert result.attrs["_FillValue"] == nodata
    assert result.coords.equals(raster.coords)

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


def test_interpolate_na_along_time_dim() -> None:
    """Test the interpolate_na_along_time_dim function."""
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

    result = interpolate_na_along_time_dim(da)

    # Check that NaNs are filled
    assert not np.isnan(result.values).any()
    assert result.dims == da.dims
    assert result.coords.equals(da.coords)


def test_interpolate_na_alignment() -> None:
    """Test that interpolate_na_along_time_dim aligns with interpolate_na_2d applied per slice."""
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
    result_along_time = interpolate_na_along_time_dim(da)

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


def test_interpolate_na_along_time_dim_missing_fillvalue() -> None:
    """Test that interpolate_na_along_time_dim raises ValueError when _FillValue is missing."""
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
        interpolate_na_along_time_dim(da)


def test_interpolate_na_along_time_dim_integer_fillvalue() -> None:
    """Test the interpolate_na_along_time_dim function with integer _FillValue."""
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
    result = interpolate_na_along_time_dim(da)
    # Check that -9999 values are filled
    assert not np.any(result.values == -9999)
    assert result.dims == da.dims
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
