"""Some raster utility functions that are not included in major raster processing libraries but used in multiple places in GEB."""

from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import xarray as xr
from rasterio.features import rasterize
from shapely.geometry import Polygon


def compress(array: npt.NDArray[Any], mask: npt.NDArray[np.bool_]) -> npt.NDArray[Any]:
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
    nodata: int | float | bool,
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
    """
    assert isinstance(data, xr.DataArray)
    da: xr.DataArray = xr.full_like(data, fill_value, *args, **kwargs)
    if name is not None:
        da.name = name
    da.attrs = attrs or {}
    da.attrs["_FillValue"] = nodata
    return da


def rasterize_like(
    gpd: gpd.GeoDataFrame,
    column: str,
    raster: xr.DataArray,
    dtype: type,
    nodata: float | int | bool,
    all_touched: bool = False,
    **kwargs: Any,
) -> xr.DataArray:
    """Rasterize a geometry to match the spatial properties of a given raster.

    Args:
        gpd: The geodataframe containing geometries to rasterize
        column: The column in the GeoDataFrame to use for rasterization values
        raster: The reference raster DataArray to match spatial properties
        dtype: The data type of the output rasterized array
        nodata: The nodata value to use in the output array
        all_touched: If True, all pixels touched by geometries will be burned in
        **kwargs: Additional keyword arguments to pass to rasterio.features.rasterize

    Returns:
        A new DataArray with the rasterized geometry, matching the spatial properties of the input raster.
            The name of the DataArray will be set to the provided column name.

    """
    da: xr.DataArray = full_like(
        data=raster,
        fill_value=nodata,
        nodata=nodata,
        attrs=raster.attrs,
        dtype=dtype,
        name=column,
    )
    geoms: gpd.Geoseries = gpd.geometry

    assert da.rio.crs == gpd.crs, "CRS of raster and GeoDataFrame must match"

    values: list[Any] = gpd[column].tolist()
    shapes: list[tuple[Polygon, int | float]] = list(zip(geoms, values, strict=True))
    out = rasterize(
        shapes,
        out_shape=raster.rio.shape,
        fill=nodata,
        transform=raster.rio.transform(),
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
