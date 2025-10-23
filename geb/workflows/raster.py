"""Some raster utility functions that are not included in major raster processing libraries but used in multiple places in GEB."""

from collections.abc import Mapping
from typing import Any, Literal, Union, overload

import dask
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import rioxarray
import xarray
import xarray as xr
import xarray_regrid
from affine import Affine
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

from geb.typing import (
    TwoDFloatArrayFloat32,
    TwoDFloatArrayFloat64,
)


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


def pad_xy(
    array_rio: rioxarray.raster_array.RasterArray,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    constant_values: Union[
        float, tuple[int, int], Mapping[Any, tuple[int, int]], None
    ] = None,
    return_slice: bool = False,
) -> xr.DataArray:
    """Pad the array to x,y bounds.

    Args:
        array_rio: rio assecor of xarray DataArray
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
    left, bottom, right, top = array_rio._internal_bounds()
    resolution_x, resolution_y = array_rio.resolution()
    y_before = y_after = 0
    x_before = x_after = 0
    y_coord: Union[xarray.DataArray, np.ndarray] = array_rio._obj[array_rio.y_dim]
    x_coord: Union[xarray.DataArray, np.ndarray] = array_rio._obj[array_rio.x_dim]

    if top - resolution_y < maxy:
        new_y_coord: np.ndarray = np.arange(bottom, maxy, -resolution_y)[::-1]
        y_before = len(new_y_coord) - len(y_coord)
        y_coord = new_y_coord
        top = y_coord[0]
    if bottom + resolution_y > miny:
        new_y_coord = np.arange(top, miny, resolution_y)
        y_after = len(new_y_coord) - len(y_coord)
        y_coord = new_y_coord
        bottom = y_coord[-1]

    if left - resolution_x > minx:
        new_x_coord: np.ndarray = np.arange(right, minx, -resolution_x)[::-1]
        x_before = len(new_x_coord) - len(x_coord)
        x_coord = new_x_coord
        left = x_coord[0]
    if right + resolution_x < maxx:
        new_x_coord = np.arange(left, maxx, resolution_x)
        x_after = len(new_x_coord) - len(x_coord)
        x_coord = new_x_coord
        right = x_coord[-1]

    if constant_values is None:
        constant_values = np.nan if array_rio.nodata is None else array_rio.nodata

    superset = array_rio._obj.pad(
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
    """

    def fillna_nearest_2d(
        arr: TwoDFloatArrayFloat32 | TwoDFloatArrayFloat64, dims: tuple[str, ...]
    ) -> TwoDFloatArrayFloat32 | TwoDFloatArrayFloat64:
        """Fill NaN values in a 2D array using nearest neighbor interpolation.

        Args:
            arr: The input 2D array with NaN values.
            dims: The dimensions of the array.

        Returns:
            The array with NaN values filled.
        """
        mask = np.isnan(arr)
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
        kwargs={"dims": da.dims},  # Additional arguments for the function
        output_dtypes=[da.dtype],
        keep_attrs=True,
    )
    return da


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
        dst: xr.DataArray = regridder.linear(target)
    elif method == "conservative":
        # conservative regridding uses the chunks of the source it not explicitly set
        # here we use the chunks of the source as base, and overwrite it with the
        # chunks of the target where they both exist
        dst: xr.DataArray = regridder.conservative(
            target,
            latitude_coord="y",
            output_chunks={**source.chunksizes, **target.chunksizes},
        )
    elif method == "nearest":
        dst: xr.DataArray = regridder.nearest(target)
    else:
        raise ValueError(
            f"Unknown method: {method}, must be 'bilinear', 'nearest', or 'conservative'"
        )

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
    data_dims: tuple[str, ...],
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
) -> TwoDFloatArrayFloat32:
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
