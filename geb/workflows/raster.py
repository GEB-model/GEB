"""Some raster utility functions that are not included in major raster processing libraries but used in multiple places in GEB."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from typing import Any, Literal, cast, overload

import dask
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import rioxarray
import xarray
import xarray as xr
import xarray_regrid
from affine import Affine
from numba import njit, prange
from pyresample.geometry import AreaDefinition
from pyresample.gradient import (
    block_bilinear_interpolator,
    block_nn_interpolator,
    gradient_resampler_indices_block,
)
from pyresample.resampler import resample_blocks
from rasterio.features import rasterize
from scipy.interpolate import griddata
from shapely.geometry import Polygon

from geb.types import (
    ArrayWithScalar,
    ThreeDArrayWithScalar,
    TwoDArrayBool,
    TwoDArrayFloat32,
    TwoDArrayFloat64,
    TwoDArrayWithScalar,
)


@overload
def decompress_with_mask(
    array: TwoDArrayWithScalar,
    mask: TwoDArrayBool,
    fillvalue: int | float | None = None,
) -> ThreeDArrayWithScalar: ...


@overload
def decompress_with_mask(
    array: ArrayWithScalar, mask: TwoDArrayBool, fillvalue: int | float | None = None
) -> TwoDArrayWithScalar: ...


def decompress_with_mask(
    array: TwoDArrayWithScalar | ArrayWithScalar,
    mask: TwoDArrayBool,
    fillvalue: int | float | None = None,
) -> ThreeDArrayWithScalar | TwoDArrayWithScalar:
    """Decompress array.

    Args:
        array: Compressed array.
        mask: Mask used for compression. True values are masked out.
        fillvalue: Value to use for masked values. If None, uses NaN for float arrays and 0 for int arrays.

    Returns:
        array: Decompressed array.
    """
    if fillvalue is None:
        if array.dtype in (np.float32, np.float64):
            fillvalue = np.nan
        else:
            fillvalue = 0
    outmap = np.full(mask.size, fillvalue, dtype=array.dtype)
    output_shape = mask.shape
    if array.ndim == 2:
        array = cast(TwoDArrayWithScalar[Any], array)
        assert array.shape[1] == mask.size - mask.sum()
        outmap = np.broadcast_to(outmap, (array.shape[0], outmap.size)).copy()
        output_shape = (array.shape[0], *output_shape)
    outmap[..., ~mask.ravel()] = array
    return outmap.reshape(output_shape)


@njit(cache=True)
def pixel_to_coord(px: int, py: int, gt: tuple) -> tuple[float, float]:
    """Converts pixel (x, y) to coordinate (lon, lat) for given geotransformation.

    Uses the upper left corner of the pixel. To use the center, add 0.5 to input pixel.

    Args:
        px: The pixel x coordinate.
        py: The pixel y coordinate.
        gt: The geotransformation. Must be unrotated.

    Returns:
        array: the coordinate (lon, lat)

    Raises:
        ValueError: If the geotransformation indicates a rotated map.
    """
    if gt[2] + gt[4] == 0:
        lon = px * gt[1] + gt[0]
        lat = py * gt[5] + gt[3]
        return lon, lat
    else:
        raise ValueError("Cannot convert rotated maps")


@njit(cache=True, parallel=True)
def pixels_to_coords(
    pixels: np.ndarray, gt: tuple[float, float, float, float, float, float]
) -> np.ndarray:
    """Converts pixels (x, y) to coordinates (lon, lat) for given geotransformation.

    Uses the upper left corner of the pixels. To use the centers, add 0.5 to input pixels.

    Args:
        pixels: The pixels (x, y) that need to be transformed to coordinates (shape: n, 2).
        gt: The geotransformation. Must be unrotated.

    Returns:
        The coordinates (lon, lat) with shape (n, 2).

    Raises:
        ValueError: If the geotransformation indicates a rotated map.
    """
    assert pixels.shape[1] == 2
    if gt[2] + gt[4] == 0:
        coords = np.empty(pixels.shape, dtype=np.float64)
        for i in prange(coords.shape[0]):  # ty: ignore[not-iterable]
            coords[i, 0] = pixels[i, 0] * gt[1] + gt[0]
            coords[i, 1] = pixels[i, 1] * gt[5] + gt[3]
        return coords
    else:
        raise ValueError("Cannot convert rotated maps")


@njit(parallel=False)
def sample_from_map(
    array: np.ndarray,
    coords: np.ndarray,
    gt: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Sample coordinates from a map. Can handle multiple dimensions.

    Args:
        array: The map to sample from (2+n dimensions).
        coords: The coordinates used to sample (shape: m, 2).
        gt: The geotransformation. Must be unrotated.

    Returns:
        The values at each coordinate.
    """
    assert gt[2] + gt[4] == 0
    size = coords.shape[0]
    x_offset = gt[0]
    y_offset = gt[3]
    x_step = gt[1]
    y_step = gt[5]
    values = np.empty((size,) + array.shape[:-2], dtype=array.dtype)
    for i in prange(size):  # ty: ignore[not-iterable]
        values[i] = array[
            ...,
            int((coords[i, 1] - y_offset) / y_step),
            int((coords[i, 0] - x_offset) / x_step),
        ]
    return values


@njit(
    parallel=False,
    cache=True,
)  # Writing to an array cannot be parallelized as race conditions would occur.
def write_to_array(
    array: np.ndarray,
    values: np.ndarray,
    coords: np.ndarray,
    gt: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Write values using coordinates to a map.

    If multiple coordinates map to a single cell,
    the values are added. The operation is inplace.

    Args:
        array: The 2-dimensional array to write to.
        values: The values to write (shape: n).
        coords: The coordinates of the values (shape: n, 2).
        gt: The geotransformation. Must be unrotated.

    Returns:
        The array with the values added (operation is inplace).
    """
    assert values.size == coords.shape[0]
    assert gt[2] + gt[4] == 0
    size = values.size
    x_offset = gt[0]
    y_offset = gt[3]
    x_step = gt[1]
    y_step = gt[5]
    for i in range(size):
        array[
            int((coords[i, 1] - y_offset) / y_step),
            int((coords[i, 0] - x_offset) / x_step),
        ] += values[i]
    return array


@njit(cache=True)
def coord_to_pixel(
    coord: tuple[float, float], gt: tuple[float, float, float, float, float, float]
) -> tuple[int, int]:
    """Converts coordinate to pixel (x, y) for given geotransformation.

    Args:
        coord: The coordinate (lon, lat) that need to be transformed to pixel.
        gt: The geotransformation. Must be unrotated.

    Returns:
        A tuple of pixel coordinates (x, y).

    Raises:
        ValueError: If the geotransformation indicates a rotated map.
    """
    if gt[2] + gt[4] == 0:
        px = (coord[0] - gt[0]) / gt[1]
        py = (coord[1] - gt[3]) / gt[5]
        return int(px), int(py)
    else:
        raise ValueError("Cannot convert rotated maps")


@njit(parallel=True)
def coords_to_pixels(
    coords: np.ndarray,
    gt: tuple[float, float, float, float, float, float],
    dtype: type[np.uint32] = np.uint32,
) -> tuple[np.ndarray, np.ndarray]:
    """Converts array of coordinates to array of pixels for given geotransformation.

    Args:
        coords: The coordinates (lon, lat) that need to be transformed to pixels (shape: n, 2).
        gt: The geotransformation. Must be unrotated.
        dtype: The data type of the output pixel arrays.

    Returns:
        A tuple of two arrays: pixel x coordinates and pixel y coordinates, each of shape (n,).

    Raises:
        ValueError: If the geotransformation indicates a rotated map.
    """
    if gt[2] + gt[4] == 0:
        size = coords.shape[0]
        x_offset = gt[0]
        y_offset = gt[3]
        x_step = gt[1]
        y_step = gt[5]
        pxs = np.empty(size, dtype=dtype)
        pys = np.empty(size, dtype=dtype)
        for i in prange(size):  # ty: ignore[not-iterable]
            pxs[i] = int((coords[i, 0] - x_offset) / x_step)
            pys[i] = int((coords[i, 1] - y_offset) / y_step)
        return pxs, pys
    else:
        raise ValueError("Cannot convert rotated maps")


@overload
def compress(
    array: ThreeDArrayWithScalar, mask: TwoDArrayBool
) -> TwoDArrayWithScalar: ...


@overload
def compress(array: TwoDArrayWithScalar, mask: TwoDArrayBool) -> ArrayWithScalar: ...


def compress(
    array: ThreeDArrayWithScalar | TwoDArrayWithScalar, mask: TwoDArrayBool
) -> TwoDArrayWithScalar | ArrayWithScalar:
    """Compress an array by applying a mask.

    Args:
        array: The array to be compressed.
        mask: The mask to apply. True values are masked out.

    Returns:
        The compressed array.
    """
    return array[..., ~mask]


def repeat_grid(data: npt.NDArray[Any], factor: int) -> npt.NDArray[Any]:
    """Repeat a 2D grid by a given factor in both dimensions.

    Args:
        data: The input 2D array to be repeated.
        factor: The number of times to repeat the array in both dimensions.

    Returns:
        The repeated 2D array.
    """
    return data.repeat(factor, axis=-2).repeat(factor, axis=-1)


@overload
def reclassify(
    data_array: xr.DataArray, remap_dict: dict, method: str = "dict"
) -> xr.DataArray: ...


@overload
def reclassify(
    data_array: np.ndarray, remap_dict: dict, method: str = "dict"
) -> np.ndarray: ...


def reclassify(
    data_array: xr.DataArray | np.ndarray, remap_dict: dict, method: str = "dict"
) -> xr.DataArray | np.ndarray:
    """Reclassify values in an xarray DataArray using a dictionary.

    Args:
        data_array: The input data array to be reclassified
        remap_dict: Dictionary mapping old values to new values
        method: The method to use for reclassification. Dict or lookup.
            If lookup, it will use a lookup array for faster performance, but
            keys must be positive integers only and be limited in size.

    Returns:
        Reclassified array with the same dimensions and coordinates

    Raises:
        ValueError: If the method is not 'dict' or 'lookup', or in the case of lookup method if keys are not positive integers
        TypeError: If the input is not an xarray.DataArray or numpy.ndarray
    """
    # create numpy array from dictionary values to get the dtype of the output
    values = np.array(list(remap_dict.values()))
    # assert that dtype is not object
    assert values.dtype != np.dtype("O")

    if method == "dict":
        # Create a vectorized function using the dictionary approach
        remap_func_dict = np.vectorize(remap_dict.get)
    elif method == "lookup":
        # Get all keys from remap_dict to create a lookup array
        keys = np.array(list(remap_dict.keys()))

        if keys.min() < 0:
            raise ValueError(
                "Keys must be positive integers for lookup method, recommend using dict method"
            )

        lookup_array = np.full(keys.max() + 1, -1, dtype=values.dtype)
        lookup_array[keys] = values
    else:
        raise ValueError("Method must be either 'dict' or 'lookup'")

    if isinstance(data_array, xr.DataArray):
        # Apply the function and keep attributes
        if method == "dict":
            # Use the dictionary method
            result = xr.apply_ufunc(
                remap_func_dict,
                data_array,
                keep_attrs=True,
                dask="parallelized",
                output_dtypes=[values.dtype],
            )
        elif method == "lookup":
            # Use the lookup method
            result = xr.apply_ufunc(
                lambda x: lookup_array[x],
                data_array,
                keep_attrs=True,
                dask="parallelized",
                output_dtypes=[values.dtype],
            )
        else:
            raise ValueError("Method must be either 'dict' or 'lookup'")

    elif isinstance(data_array, np.ndarray):
        # Apply the function and keep attributes
        if method == "dict":
            # Use the dictionary method
            # result = remap_func_dict(data_array)
            result = remap_func_dict(data_array)
        elif method == "lookup":
            result = lookup_array[data_array]
        else:
            raise ValueError("Method must be either 'dict' or 'lookup'")
    else:
        raise TypeError("Input must be either xarray.DataArray or numpy.ndarray")

    return result


def full_like(
    data: xr.DataArray,
    fill_value: int | float | bool,
    nodata: int | float | bool | None,
    attrs: None | dict = None,
    name: str | None = None,
    *args: Any,
    **kwargs: Any,
) -> xr.DataArray:
    """Create a new xarray DataArray with the same shape and coordinates as the input data.

    The new DataArray is filled with the specified fill_value and has the specified nodata value.

    Args:
        data: The input DataArray to use as a template.
        fill_value: The value to fill the new DataArray with.
        nodata: The nodata value to set in the attributes of the new DataArray.
        attrs: Optional dictionary of attributes to set in the new DataArray.
        name: Optional name for the new DataArray.
        *args: Additional positional arguments to pass to xr.full_like.
        **kwargs: Additional keyword arguments to pass to xr.full_like.

    Returns:
        A new DataArray with the same shape and coordinates as the input data,
        filled with the specified fill_value and with the specified nodata value in the attributes.

    Raises:
        ValueError: If nodata is None and fill_value is not a boolean.
    """
    assert isinstance(data, xr.DataArray)
    if nodata is None and not isinstance(fill_value, bool):
        raise ValueError("Nodata value must be set unless fill_value is a boolean. ")
    da: xr.DataArray = xr.full_like(data, fill_value, *args, **kwargs)
    if name is not None:
        da.name = name
    da.attrs = attrs or {}
    da.attrs["_FillValue"] = nodata
    return da


def rasterize_like(
    gdf: gpd.GeoDataFrame,
    raster: xr.DataArray,
    dtype: type,
    nodata: float | int | bool,
    all_touched: bool = False,
    column: str | None = None,
    burn_value: float | int | bool | None = None,
    name: str | None = "rasterized",
    **kwargs: Any,
) -> xr.DataArray:
    """Rasterize a geometry to match the spatial properties of a given raster.

    Args:
        gdf: The geodataframe containing geometries to rasterize
        raster: The reference raster DataArray to match spatial properties
        dtype: The data type of the output rasterized array
        nodata: The nodata value to use in the output array
        all_touched: If True, all pixels touched by geometries will be burned in
        column: If provided, values from this column will be used for rasterization.
            Cannot be used together with `burn_value`.
        burn_value: If provided, this value will be used for all geometries instead of values from a column.
            Cannot be used together with `column`.
        name: If provided, the name of the output DataArray. If 'column' is provided and name is not set,
            the name will be set to the column name.
        **kwargs: Additional keyword arguments to pass to rasterio.features.rasterize

    Returns:
        A new DataArray with the rasterized geometry, matching the spatial properties of the input raster.
            The name of the DataArray will be set to the provided column name.

    Raises:
        ValueError: If both `column` and `burn_value` are provided or if neither is provided.

    """
    if (column is None and burn_value is None) or (
        column is not None and burn_value is not None
    ):
        raise ValueError("Either column or burn_value must be provided, but not both.")

    if name == "rasterized" and column is not None:
        name: str = column

    da: xr.DataArray = full_like(
        data=raster,
        fill_value=nodata,
        nodata=nodata,
        attrs=raster.attrs,
        dtype=dtype,
        name=name,
    )
    geoms: gpd.Geoseries = gdf.geometry

    assert da.rio.crs == gdf.crs, "CRS of raster and GeoDataFrame must match"

    if burn_value is not None:
        values = [burn_value] * len(gdf)
    else:
        values: list[Any] = gdf[column].tolist()
    shapes: list[tuple[Polygon, int | float]] = list(zip(geoms, values, strict=True))
    out = rasterize(
        shapes,
        out_shape=raster.rio.shape,
        fill=nodata,
        transform=raster.rio.transform(recalc=True),
        out=da.values,
        all_touched=all_touched,
        **kwargs,
    )
    da.values = out
    return da


def convert_nodata(
    da: xr.DataArray,
    new_nodata: float | int | bool,
) -> xr.DataArray:
    """Convert the nodata value of a DataArray to a new nodata value.

    Args:
        da: The input DataArray.
        new_nodata: The new nodata value to set in the DataArray.

    Returns:
        A new DataArray with the same data, but with the nodata values converted to the new nodata value.

    Raises:
        ValueError: If the input DataArray does not have a '_FillValue' attribute.
    """
    if "_FillValue" not in da.attrs:
        raise ValueError("Input DataArray must have a '_FillValue' attribute")
    da_new = da.where(da != da.attrs["_FillValue"], new_nodata)
    da_new.attrs = da.attrs.copy()
    da_new.attrs["_FillValue"] = new_nodata
    return da_new


def snap_to_grid(
    ds: xr.DataArray | xr.Dataset,
    reference: xr.DataArray | xr.Dataset,
    relative_tolerance: float = 0.02,
    ydim: str = "y",
    xdim: str = "x",
) -> xr.Dataset | xr.DataArray:
    """Snaps the coordinates of a dataset to a reference dataset.

    Some datasets have a slightly different grid than the model grid, usually
    because of different rounding errors when creating the grid, and floating
    point precision issues. This method checks if the coordinates are more or
    less the same, and if so, snaps the coordinates of the dataset to the
    reference dataset.

    Args:
        ds: The dataset to snap.
        reference: The reference dataset.
        relative_tolerance: The relative tolerance for snapping.
        ydim: The name of the y dimension.
        xdim: The name of the x dimension.

    Returns:
        The snapped dataset.
    """
    # make sure all datasets have more or less the same coordinates
    assert np.isclose(
        ds.coords[ydim].values,
        reference[ydim].values,
        atol=abs(ds.rio.resolution()[1] * relative_tolerance),
        rtol=0,
    ).all()
    assert np.isclose(
        ds.coords[xdim].values,
        reference[xdim].values,
        atol=abs(ds.rio.resolution()[0] * relative_tolerance),
        rtol=0,
    ).all()
    return ds.assign_coords({ydim: reference[ydim], xdim: reference[xdim]})


def clip_with_grid(
    ds: xr.Dataset | xr.DataArray, mask: xr.DataArray
) -> tuple[xr.Dataset | xr.DataArray, dict[str, slice]]:
    """Clip a dataset to the extent of a mask.

    True values in the mask indicate valid data.

    Args:
        ds: The dataset to clip.
        mask: The mask to clip with. Must have the same x and y coordinates as the dataset.

    Returns:
        A tuple containing the clipped dataset and a dictionary with slices for x and y dimensions.
    """
    assert ds.shape == mask.shape
    cells_along_y = mask.sum(dim="x").values.ravel()
    miny = (cells_along_y > 0).argmax().item()
    maxy = cells_along_y.size - (cells_along_y[::-1] > 0).argmax().item()

    cells_along_x = mask.sum(dim="y").values.ravel()
    minx = (cells_along_x > 0).argmax().item()
    maxx = cells_along_x.size - (cells_along_x[::-1] > 0).argmax().item()

    bounds = {"y": slice(miny, maxy), "x": slice(minx, maxx)}

    return ds.isel(bounds), bounds


def bounds_are_within(
    small_bounds: tuple[float, float, float, float],
    large_bounds: tuple[float, float, float, float],
    tolerance: int | float = 0,
) -> bool:
    """Check if one set of bounds is within another set of bounds, with an optional tolerance.

    Args:
        small_bounds: The bounds to check (minx, miny, maxx, maxy).
        large_bounds: The bounds to check against (minx, miny, maxx, maxy).
        tolerance: The tolerance to apply when checking the bounds.

    Returns:
        True if the small bounds are within the large bounds, False otherwise.
    """
    return (
        small_bounds[0] + tolerance >= large_bounds[0]
        and small_bounds[1] + tolerance >= large_bounds[1]
        and small_bounds[2] <= large_bounds[2] + tolerance
        and small_bounds[3] <= large_bounds[3] + tolerance
    )


@overload
def pad_xy(
    da: xr.DataArray,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    constant_values: float
    | tuple[int, int]
    | Mapping[Any, tuple[int, int]]
    | None = None,
    return_slice: Literal[True] = True,
) -> tuple[xr.DataArray, dict[str, slice]]: ...


@overload
def pad_xy(
    da: xr.DataArray,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    constant_values: float
    | tuple[int, int]
    | Mapping[Any, tuple[int, int]]
    | None = None,
    return_slice: Literal[False] = False,
) -> xr.DataArray: ...


def pad_xy(
    da: xr.DataArray,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    constant_values: float
    | tuple[int, int]
    | Mapping[Any, tuple[int, int]]
    | None = None,
    return_slice: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, dict[str, slice]]:
    """Pad the array to x,y bounds, while preserving old coordinates exactly.

    Rather than re-calculating the x and y coordinates, this function
    uses the original coordinates and extends them as needed. This ensures
    that the original coordinates are preserved without introducing floating point
    imprecision.

    Args:
        da: the DataArray to pad.
        minx: Minimum bound for x coordinate.
        miny: Minimum bound for y coordinate.
        maxx: Maximum bound for x coordinate.
        maxy: Maximum bound for y coordinate.
        constant_values: scalar, tuple or mapping of hashable to tuple
            The value used for padding. If None, nodata will be used if it is
            set, and np.nan otherwise.
        return_slice: If True, returns a dictionary with slices for x and y.

    Returns:
        Padded DataArray with new x and y coordinates.

    If `return_slice` is True, also returns a dictionary with slices for x and y
    dimensions that can be used to index the padded DataArray to get the original
    data.

    """
    array_rio: rioxarray.rioxarray.RioXarrayAccessor = da.rio

    left, bottom, right, top = array_rio._internal_bounds()
    resolution_x, resolution_y = array_rio.resolution()
    y_coord: xarray.DataArray | np.ndarray = da[array_rio.y_dim].values
    x_coord: xarray.DataArray | np.ndarray = da[array_rio.x_dim].values

    y_before = y_after = 0
    x_before = x_after = 0

    # Create new coordinates by extending existing ones
    if top - resolution_y < maxy:
        new_top_coords = np.arange(top - resolution_y, maxy, -resolution_y)[::-1]
        new_y_coord = np.concatenate([new_top_coords, y_coord])
        y_before = len(new_y_coord) - len(y_coord)
        y_coord = new_y_coord
        top = y_coord[0]
    if bottom + resolution_y > miny:
        new_bottom_coords = np.arange(bottom + resolution_y, miny, resolution_y)
        new_y_coord = np.concatenate([y_coord, new_bottom_coords])
        y_after = len(new_y_coord) - len(y_coord)
        y_coord = new_y_coord
        bottom = y_coord[-1]

    if left - resolution_x > minx:
        new_left_coords: np.ndarray = np.arange(
            left - resolution_x, minx, -resolution_x
        )[::-1]
        new_x_coord = np.concatenate([new_left_coords, x_coord])
        x_before = len(new_x_coord) - len(x_coord)
        x_coord = new_x_coord
        left = x_coord[0]
    if right + resolution_x < maxx:
        new_right_coords = np.arange(x_coord[-1] + resolution_x, maxx, resolution_x)
        new_x_coord = np.concatenate([x_coord, new_right_coords])
        x_after = len(new_x_coord) - len(x_coord)
        x_coord = new_x_coord
        right = x_coord[-1]

    if constant_values is None:
        constant_values = np.nan if array_rio.nodata is None else array_rio.nodata

    superset = da.pad(
        pad_width={
            array_rio.x_dim: (x_before, x_after),
            array_rio.y_dim: (y_before, y_after),
        },
        constant_values=constant_values,  # type: ignore
    ).rio.set_spatial_dims(x_dim=array_rio.x_dim, y_dim=array_rio.y_dim, inplace=True)
    superset[array_rio.x_dim] = x_coord
    superset[array_rio.y_dim] = y_coord
    superset.rio.write_transform(inplace=True)
    if return_slice:
        return superset, {
            "x": slice(x_before, superset["x"].size - x_after),
            "y": slice(y_before, superset["y"].size - y_after),
        }
    else:
        return superset


def interpolate_na_along_time_dim(da: xr.DataArray) -> xr.DataArray:
    """Interpolate NaN values along the time dimension of a DataArray.

    Uses nearest neighbor interpolation in the spatial dimensions for each time slice.

    Args:
        da: The input DataArray with a time dimension.

    Returns:
        A new DataArray with NaN values interpolated along the time dimension.

    Raises:
        ValueError: If '_FillValue' attribute is missing.
    """
    if "_FillValue" not in da.attrs:
        raise ValueError("DataArray must have '_FillValue' attribute")

    nodata = da.attrs["_FillValue"]

    def fillna_nearest_2d(
        arr: TwoDArrayFloat32 | TwoDArrayFloat64, dims: tuple[str, ...], nodata: float
    ) -> TwoDArrayFloat32 | TwoDArrayFloat64:
        """Fill NaN values in a 2D array using nearest neighbor interpolation.

        Args:
            arr: The input 2D array with NaN values.
            dims: The dimensions of the array.
            nodata: The nodata value to treat as missing.

        Returns:
            The array with NaN values filled.
        """
        mask = np.isnan(arr) if np.isnan(nodata) else arr == nodata
        if not mask.any():
            return arr

        assert dims == ("time", "y", "x")

        for time_idx in range(arr.shape[0]):
            mask_slice = mask[time_idx]
            time_slice = arr[time_idx]

            y, x = np.indices(time_slice.shape)
            known_x, known_y = x[~mask_slice], y[~mask_slice]
            known_v = time_slice[~mask_slice]
            missing_x, missing_y = x[mask_slice], y[mask_slice]

            filled_values = griddata(
                (known_x, known_y), known_v, (missing_x, missing_y), method="nearest"
            )
            arr[time_idx][mask_slice] = filled_values
        return arr

    da = xr.apply_ufunc(
        fillna_nearest_2d,
        da,
        dask="parallelized",  # Enable parallelized computation
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        kwargs={
            "dims": da.dims,
            "nodata": nodata,
        },  # Additional arguments for the function
        output_dtypes=[da.dtype],
        keep_attrs=True,
    )
    return da


def interpolate_na_2d(da: xr.DataArray) -> xr.DataArray:
    """Interpolate NaN values in a 2D DataArray using nearest neighbor interpolation.

    Args:
        da: The input DataArray with dimensions ('y', 'x').

    Returns:
        A new DataArray with NaN values interpolated.

    Raises:
        ValueError: If '_FillValue' attribute is missing.
    """
    if "_FillValue" not in da.attrs:
        raise ValueError("DataArray must have '_FillValue' attribute")

    nodata = da.attrs["_FillValue"]

    mask = np.isnan(da.values) if np.isnan(nodata) else da.values == nodata

    if not mask.any():
        return da

    y, x = np.indices(da.values.shape)
    known_x, known_y = x[~mask], y[~mask]
    known_v = da.values[~mask]
    missing_x, missing_y = x[mask], y[mask]

    filled_values = griddata(
        (known_x, known_y), known_v, (missing_x, missing_y), method="nearest"
    )
    da_filled = da.copy()
    da_filled.values[mask] = filled_values
    return da_filled


def resample_like(
    source: xr.DataArray, target: xr.DataArray, method: str = "bilinear"
) -> xr.DataArray:
    """Resample the source DataArray to match the target DataArray's grid.

    Args:
        source: the source DataArray to be resampled.
        target: the target DataArray to match.
        method: the resampling method, can be 'bilinear', 'nearest', or 'conservative'.

    Returns:
        A new DataArray that has been resampled to match the target's grid.

    Raises:
        ValueError: if the method is not 'bilinear', 'nearest', or 'conservative'.

    """
    source_spatial_ref: Any = source.spatial_ref

    # xarray-regrid does not handle integer types well
    assert not np.issubdtype(source.dtype, np.integer), (
        "Source data must not be an integer type for resampling"
    )

    source: xr.DataArray = source.drop_vars("spatial_ref")
    target: xr.DataArray = target.drop_vars("spatial_ref")  # TODO: Perhaps not needed

    regridder = xarray_regrid.regrid.Regridder(source)

    if method == "bilinear":
        dst = regridder.linear(target)  # ty: ignore[invalid-argument-type]
    elif method == "conservative":
        # conservative regridding uses the chunks of the source it not explicitly set
        # here we use the chunks of the source as base, and overwrite it with the
        # chunks of the target where they both exist
        dst = regridder.conservative(
            target,  # ty: ignore[invalid-argument-type]
            latitude_coord="y",
            output_chunks={**source.chunksizes, **target.chunksizes},
        )
    elif method == "nearest":
        dst = regridder.nearest(target)  # ty: ignore[invalid-argument-type]
    else:
        raise ValueError(
            f"Unknown method: {method}, must be 'bilinear', 'nearest', or 'conservative'"
        )

    assert isinstance(dst, xr.DataArray)

    if source.dtype == np.float32:
        dst: xr.DataArray = dst.astype(np.float32)

    # Set the spatial reference back to the original
    dst: xr.DataArray = dst.assign_coords({"spatial_ref": source_spatial_ref})
    return dst


def get_area_definition(da: xr.DataArray) -> AreaDefinition:
    """Get the pyresample area definition from an xarray DataArray.

    This is a requirement for the resampling functions in pyresample.

    Args:
        da: The xarray DataArray with spatial dimensions.

    Returns:
        A pyresample AreaDefinition object.
    """
    return AreaDefinition(
        area_id="",
        description="",
        proj_id="",
        projection=da.rio.crs.to_proj4(),
        width=da.x.size,
        height=da.y.size,
        area_extent=da.rio.bounds(),
    )


def _fill_in_coords(
    target_coords: xr.core.coordinates.DataArrayCoordinates,
    source_coords: xr.core.coordinates.DataArrayCoordinates,
    data_dims: tuple[Hashable, ...],
) -> list[xr.core.coordinates.DataArrayCoordinates]:
    """Fill in missing coordinates that are also dimensions from source except for 'x' and 'y' which are taken from target.

    For example useful to fill the time coordinate

    Args:
        target_coords: All coordinates from the target DataArray.
        source_coords: All coordinates from the source DataArray.
        data_dims: Dimensions to transfer coordinates for. 'x' and 'y' are skipped.

    Returns:
        A list of coordinates in the order of data_dims.
    """
    x_coord, y_coord = target_coords["x"], target_coords["y"]
    coords = []
    for key in data_dims:
        if key == "x":
            coords.append(x_coord)
        elif key == "y":
            coords.append(y_coord)
        else:
            coords.append(source_coords[key])
    return coords


def resample_chunked(
    source: xr.DataArray, target: xr.DataArray, method: str = "bilinear"
) -> xr.DataArray:
    """Resample a source DataArray to match the grid of a target DataArray using block-based resampling.

    This function uses the pyresample library to perform block-based resampling,
    which is suitable for large datasets that do not fit into memory.

    Args:
        source: DataArray to be resampled, must have spatial dimensions "y" and "x".
        target: DataArray that defines the target grid, must have spatial dimensions "y" and "x".
        method: Resampling method, must be 'bilinear' or 'nearest'. Defaults to "bilinear".

    Raises:
        ValueError: If the method is not 'bilinear' or 'nearest'.
        ValueError: If the target DataArray is not chunked.

    Returns:
        A new DataArray that has been resampled to match the target's grid.
    """
    if method == "nearest":
        interpolator = block_nn_interpolator
    elif method == "bilinear":
        interpolator = block_bilinear_interpolator
    else:
        raise ValueError(f"Unknown method: {method}, must be 'bilinear' or 'nearest'")

    assert target.dims == ("y", "x")

    source_geo: AreaDefinition = get_area_definition(source)
    target_geo: AreaDefinition = get_area_definition(target)

    if target.chunks is None:
        raise ValueError("Target DataArray must be chunked for resample_chunked")

    indices: dask.Array = resample_blocks(
        gradient_resampler_indices_block,
        source_geo,
        [],
        target_geo,
        chunk_size=(2, *target.chunks),
        dtype=np.float64,
    )

    resampled_data: dask.Array = resample_blocks(
        interpolator,
        source_geo,
        [source.data],
        target_geo,
        dst_arrays=[indices],
        chunk_size=(*source.data.shape[:-2], *target.chunks),
        dtype=source.dtype,
        fill_value=source.attrs["_FillValue"],
    )

    # Convert result back to xarray DataArray
    da: xr.DataArray = xr.DataArray(
        resampled_data,
        dims=source.dims,
        coords=_fill_in_coords(target.coords, source.coords, source.dims),
        name=source.name,
        attrs=source.attrs.copy(),
    )
    da.rio.set_crs(source.rio.crs)
    return da


def calculate_cell_area(
    affine_transform: Affine, shape: tuple[int, int]
) -> TwoDArrayFloat32:
    """Calculate the area of each cell in a grid given its affine transform.

    Must be in a geographic coordinate system (degrees).

    Args:
        affine_transform: The affine transformation of the grid.
        shape: The shape of the grid as (height, width).

    Returns:
        A 2D array of cell areas in square meters.
    """
    RADIUS_EARTH_EQUATOR: Literal[40075017] = 40075017  # m
    distance_1_degree_latitude: float = RADIUS_EARTH_EQUATOR / 360

    height, width = shape

    lat_idx = np.arange(0, height).repeat(width).reshape((height, width))
    lat = (lat_idx + 0.5) * affine_transform.e + affine_transform.f
    width_m = (
        distance_1_degree_latitude * np.cos(np.radians(lat)) * abs(affine_transform.a)
    )
    height_m = distance_1_degree_latitude * abs(affine_transform.e)
    return (width_m * height_m).astype(np.float32)


def clip_region(
    mask: xr.DataArray, *data_arrays: xr.DataArray, align: float | int
) -> tuple[xr.DataArray, ...]:
    """Use the given mask to clip the mask itself and the given data arrays.

    The clipping is done to the bounding box of the True values in the mask. The bounding box
    is aligned to the given align value. The align value is in the same units as the coordinates
    of the mask and data arrays.

    Args:
        mask: The mask to use for clipping. Must be a 2D boolean DataArray with x and y coordinates.
            True values indicate the area to keep.
        *data_arrays: The data arrays to clip. Must have the same x and y coordinates as the mask.
        align: Align the bounding box to a specific grid spacing. For example, when this is set to 1
            the bounding box will be aligned to whole numbers. If set to 0.5, the bounding box will
            be aligned to 0.5 intervals.

    Returns:
        A tuple containing the clipped mask and the clipped data arrays.

    Raises:
        ValueError: If the data arrays do not have the same shape or coordinates as the mask.
    """
    rows, cols = np.where(mask)
    mincol = cols.min()
    maxcol = cols.max()
    minrow = rows.min()
    maxrow = rows.max()

    minx = mask.x[mincol].item()
    maxx = mask.x[maxcol].item()
    miny = mask.y[minrow].item()
    maxy = mask.y[maxrow].item()

    xres, yres = mask.rio.resolution()

    mincol_aligned = mincol + round(((minx // align * align) - minx) / xres)
    maxcol_aligned = maxcol + round(((maxx // align * align) + align - maxx) / xres)
    minrow_aligned = minrow + round(((miny // align * align) + align - miny) / yres)
    maxrow_aligned = maxrow + round((((maxy // align) * align) - maxy) / yres)

    assert math.isclose(mask.x[mincol_aligned] // align % 1, 0)
    assert math.isclose(mask.x[maxcol_aligned] // align % 1, 0)
    assert math.isclose(mask.y[minrow_aligned] // align % 1, 0)
    assert math.isclose(mask.y[maxrow_aligned] // align % 1, 0)

    assert mincol_aligned <= mincol
    assert maxcol_aligned >= maxcol
    assert minrow_aligned <= minrow
    assert maxrow_aligned >= maxrow

    clipped_mask = mask.isel(
        y=slice(minrow_aligned, maxrow_aligned),
        x=slice(mincol_aligned, maxcol_aligned),
    )
    clipped_arrays = []
    for da in data_arrays:
        if da.shape != mask.shape:
            raise ValueError("All data arrays must have the same shape as the mask.")
        if not np.array_equal(da.x, mask.x) or not np.array_equal(da.y, mask.y):
            raise ValueError(
                "All data arrays must have the same coordinates as the mask."
            )
        clipped_arrays.append(
            da.isel(
                y=slice(minrow_aligned, maxrow_aligned),
                x=slice(mincol_aligned, maxcol_aligned),
            )
        )
    return clipped_mask, *clipped_arrays
