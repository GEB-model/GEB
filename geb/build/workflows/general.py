from collections.abc import Mapping
from datetime import date
from typing import Any, Literal, Union

import dask
import numpy as np
import pandas as pd
import xarray
import xarray as xr
import xarray_regrid
from affine import Affine
from pyresample import geometry
from pyresample.gradient import (
    block_bilinear_interpolator,
    block_nn_interpolator,
    gradient_resampler_indices_block,
)
from pyresample.resampler import resample_blocks
from scipy.interpolate import griddata
from tqdm import tqdm


def repeat_grid(data, factor):
    return data.repeat(factor, axis=-2).repeat(factor, axis=-1)


def calculate_cell_area(affine_transform: Affine, shape: tuple[int, int]) -> np.ndarray:
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


def clip_with_grid(ds, mask):
    assert ds.shape == mask.shape
    cells_along_y = mask.sum(dim="x").values.ravel()
    miny = (cells_along_y > 0).argmax().item()
    maxy = cells_along_y.size - (cells_along_y[::-1] > 0).argmax().item()

    cells_along_x = mask.sum(dim="y").values.ravel()
    minx = (cells_along_x > 0).argmax().item()
    maxx = cells_along_x.size - (cells_along_x[::-1] > 0).argmax().item()

    bounds = {"y": slice(miny, maxy), "x": slice(minx, maxx)}

    return ds.isel(bounds), bounds


def bounds_are_within(small_bounds, large_bounds, tollerance=0):
    assert small_bounds[0] + tollerance >= large_bounds[0], "Region bounds do not match"
    assert small_bounds[1] + tollerance >= large_bounds[1], "Region bounds do not match"
    assert small_bounds[2] <= large_bounds[2] + tollerance, "Region bounds do not match"
    assert small_bounds[3] <= large_bounds[3] + tollerance, "Region bounds do not match"
    return True


def pad_xy(
    array_rio,
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


def project_to_future(df, project_future_until_year, inflation_rates):
    # expand table until year
    assert isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex)
    future_index = pd.date_range(
        df.index[-1],
        date(project_future_until_year, 12, 31),
        freq=pd.infer_freq(df.index),
        inclusive="right",
    )
    df = df.reindex(df.index.union(future_index))
    for future_date in tqdm(future_index):
        source_date = future_date - pd.DateOffset(years=1)  # source is year ago
        inflation_index = inflation_rates["time"].index(str(future_date.year))
        for region_id, _ in df.columns:
            region_inflation_rate = inflation_rates["data"][str(region_id)][
                inflation_index
            ]
            df.loc[future_date, region_id] = (
                df.loc[source_date, region_id] * region_inflation_rate
            ).values
    return df


def interpolate_na_along_time_dim(da):
    def fillna_nearest_2d(arr, dims):
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


def get_area_definition(da: xr.DataArray) -> geometry.AreaDefinition:
    """Get the pyresample area definition from an xarray DataArray.

    This is a requirement for the resampling functions in pyresample.

    Args:
        da: The xarray DataArray with spatial dimensions.

    Returns:
        A pyresample AreaDefinition object.
    """
    return geometry.AreaDefinition(
        area_id="",
        description="",
        proj_id="",
        projection=da.rio.crs.to_proj4(),
        width=da.x.size,
        height=da.y.size,
        area_extent=da.rio.bounds(),
    )


def _fill_in_coords(target_coords, source_coords, data_dims):
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

    source_geo: geometry.AreaDefinition = get_area_definition(source)
    target_geo: geometry.AreaDefinition = get_area_definition(target)

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


def validate_farm_size_data(
    agricultural_area_db_ha,
    region_n_holdings,
    size_class_boundaries,
    ISO3,
    tolerance=0.3,
):
    """Validate that agricultural area is consistent with farm size class boundaries.

    Parameters
    ----------
    agricultural_area_db_ha : pd.Series
        Agricultural area in hectares per size class
    region_n_holdings : pd.Series
        Number of holdings per size class
    size_class_boundaries : dict
        Dictionary mapping size class names to (min, max) boundaries in m²
    region : pd.Series
        Region information containing NAME_1 and ISO3
    tolerance : float, optional
        Tolerance for validation (default: 0.3 = 30%)

    Raises:
    ------
    ValueError
        If agricultural area falls outside expected range for any size class
    """
    for size_class in agricultural_area_db_ha.index:
        actual_area = agricultural_area_db_ha[size_class]

        # Get the size class boundaries for validation
        min_size_ha, max_size_ha = size_class_boundaries[size_class]
        # Convert from m² to ha
        min_size_ha = min_size_ha / 10000
        max_size_ha = max_size_ha / 10000 if max_size_ha != np.inf else np.inf

        # Calculate expected area range based on class boundaries
        min_expected_area = region_n_holdings[size_class] * min_size_ha
        max_expected_area = region_n_holdings[size_class] * max_size_ha

        # Check if actual area falls within reasonable bounds
        if not (
            min_expected_area * (1 - tolerance)
            <= actual_area
            <= max_expected_area * (1 + tolerance)
        ):
            raise ValueError(
                f"Incorrect farm size data for region {ISO3}. "
                f"Size class {size_class}: agricultural area ({actual_area:.1f} ha) outside expected range "
                f"[{min_expected_area * (1 - tolerance):.1f}, {max_expected_area * (1 + tolerance):.1f}] ha "
                f" Check the farm size data and correct the data, Tim can help."
            )
