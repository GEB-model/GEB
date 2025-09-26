"""Tests for raster workflow functions."""

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

from geb.workflows.raster import (
    compress,
    convert_nodata,
    full_like,
    rasterize_like,
    reclassify,
    repeat_grid,
)


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
    result = rasterize_like(gdf, "value", raster, dtype, nodata, all_touched=True)

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
        gdf, "value", raster_geo, dtype, nodata, all_touched=False
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
