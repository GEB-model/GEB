"""I/O related functions and classes for the GEB project."""

import asyncio
import bz2
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
import warnings
from collections.abc import Hashable
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Literal, cast, overload

import dask.array
import dask.tokenize
import geopandas as gpd
import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import rasterio
import requests
import rioxarray  # noqa: F401
import s3fs
import xarray as xr
import yaml
import zarr
import zarr.storage
from pyproj import CRS
from rasterio.transform import Affine
from tqdm import tqdm
from zarr.codecs import BloscCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.errors import ZarrUserWarning

from geb.geb_types import (
    ArrayDatetime64,
    ThreeDArray,
    TwoDArray,
    TwoDArrayFloat32,
)

zarr.config.set({"codec_pipeline.fill_missing_chunks": False})


def read_table(fp: Path, **kwargs: Any) -> pd.DataFrame:
    """Load a parquet file as a pandas DataFrame.

    Args:
        fp: The path to the parquet file.
        kwargs: Additional keyword arguments to pass to `pd.read_parquet`.

    Returns:
        The pandas DataFrame.
    """
    return pd.read_parquet(fp, engine="pyarrow", **kwargs)


def write_table(df: pd.DataFrame, fp: Path) -> None:
    """Save a pandas DataFrame to a parquet file.

    Args:
        df: The pandas DataFrame to save.
        fp: The path to the output parquet file.
    """
    df.to_parquet(
        fp,
        engine="pyarrow",
        compression="zstd",
        compression_level=22,
        row_group_size=max(min(10_000, len(df)), 1),
    )


@overload
def read_array(fp: Path, return_attributes: bool = False) -> np.ndarray: ...


@overload
def read_array(
    fp: Path, return_attributes: bool = True
) -> tuple[np.ndarray, dict[str, Any]]: ...


def read_array(
    fp: Path, return_attributes: bool = False
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Load a numpy array from a .zarr file.

    Args:
        fp: The path to the .zarr file.
        return_attributes: Whether to return the attributes along with the array.

    Returns:
        The numpy array, or a tuple of the array and its attributes if return_attributes is True.
    """
    zarr_object = zarr.open_array(fp, mode="r")
    array = zarr_object[:]
    assert isinstance(array, np.ndarray)
    if return_attributes:
        return array, dict(zarr_object.attrs)
    return array


def write_array(
    arr: np.ndarray, fp: Path, attributes: dict[str, Any] | None = None
) -> None:
    """Save a numpy array to a .zarr file.

    Args:
        arr: The numpy array to save.
        fp: The path to the output .zarr file.
        attributes: Optional dictionary of attributes to store with the array.
    """
    zarr.save_array(fp, arr, overwrite=True, attributes=attributes)  # ty:ignore[invalid-argument-type]


@overload
def read_grid(
    filepath: Path,
    ndim: Literal[2] = 2,
    load: Literal[True] = True,
    return_transform_and_crs: Literal[False] = False,
) -> TwoDArray: ...


@overload
def read_grid(
    filepath: Path,
    ndim: Literal[2] = 2,
    load: Literal[True] = True,
    *,
    return_transform_and_crs: Literal[True],
) -> tuple[TwoDArray, Affine, str]: ...


@overload
def read_grid(
    filepath: Path,
    ndim: Literal[3],
    load: Literal[True] = True,
    return_transform_and_crs: Literal[False] = False,
) -> ThreeDArray: ...


@overload
def read_grid(
    filepath: Path,
    ndim: Literal[3],
    load: Literal[True] = True,
    *,
    return_transform_and_crs: Literal[True],
) -> tuple[ThreeDArray, Affine, str]: ...


@overload
def read_grid(
    filepath: Path,
    ndim: Literal[2] = 2,
    load: Literal[False] = False,
    return_transform_and_crs: Literal[False] = False,
) -> zarr.Array: ...


@overload
def read_grid(
    filepath: Path,
    ndim: Literal[2] = 2,
    load: Literal[False] = False,
    *,
    return_transform_and_crs: Literal[True],
) -> tuple[zarr.Array, Affine, str]: ...


@overload
def read_grid(
    filepath: Path,
    ndim: Literal[3],
    load: Literal[False] = False,
    return_transform_and_crs: Literal[False] = False,
) -> zarr.Array: ...


@overload
def read_grid(
    filepath: Path,
    ndim: Literal[3],
    load: Literal[False] = False,
    *,
    return_transform_and_crs: Literal[True],
) -> tuple[zarr.Array, Affine, str]: ...


def read_grid(
    filepath: Path,
    ndim: Literal[2, 3],
    load: bool = True,
    return_transform_and_crs: bool = False,
) -> (
    TwoDArray
    | ThreeDArray
    | zarr.Array
    | tuple[TwoDArray | ThreeDArray | zarr.Array, Affine, str]
):
    """Load a raster grid from .zarr file.

    Args:
        filepath: The path of the .zarr file.
        ndim: The expected number of dimensions of the raster data (2 or 3).
        load: Whether to load the data into memory. If False, a zarr array will be returned instead of a numpy array. Default is True.
        return_transform_and_crs: Whether to return the affine transform and CRS along with the data. Default is False.

    Returns:
        The raster data as a numpy array, or a tuple of the raster data, affine transform, and CRS string if return_transform_and_crs is True.

    Raises:
        ValueError: If the loaded data does not have the expected number of dimensions.
    """
    store: zarr.storage.LocalStore = zarr.storage.LocalStore(filepath, read_only=True)
    group: zarr.Group = zarr.open_group(store, mode="r")
    data_array: zarr.Array | zarr.Group = group[filepath.stem]

    assert isinstance(data_array, zarr.Array)

    if load:
        data = data_array[:]
        assert isinstance(data, np.ndarray)
        data: TwoDArray | ThreeDArray = data  # ty:ignore[invalid-assignment]
    else:
        data = data_array
        assert isinstance(data, zarr.Array)

    if data.ndim != ndim:
        raise ValueError(f"Expected data with {ndim} dimensions, but got {data.ndim}")

    if return_transform_and_crs:
        x_array: zarr.Array | zarr.Group = group["x"]
        assert isinstance(x_array, zarr.Array)
        x = x_array[:]
        assert isinstance(x, np.ndarray)
        y_array: zarr.Array | zarr.Group = group["y"]
        assert isinstance(y_array, zarr.Array)
        y = y_array[:]
        assert isinstance(y, np.ndarray)
        x_diff: float = np.diff(x[:]).mean().item()  # ty:ignore[invalid-argument-type]
        y_diff: float = np.diff(y[:]).mean().item()  # ty:ignore[invalid-argument-type]
        transform: Affine = Affine(
            a=x_diff,
            b=0,
            c=x[0] - x_diff / 2,  # ty:ignore[invalid-argument-type]
            d=0,
            e=y_diff,
            f=y[0] - y_diff / 2,  # ty:ignore[invalid-argument-type]
        )
        crs = data_array.attrs["_CRS"]
        assert isinstance(crs, dict)
        wkt: str = crs["wkt"]  # ty:ignore[invalid-argument-type]
        store.close()
        return data, transform, wkt
    else:
        store.close()
        return data


def read_geom(filepath: str | Path, **kwargs: Any) -> gpd.GeoDataFrame:
    """Load a geometry for the GEB model from disk.

    Args:
        filepath: Path to the geometry file.
        **kwargs: Additional keyword arguments to pass to `gpd.read_parquet`.

    Returns:
        A GeoDataFrame containing the geometries.
    """
    return gpd.read_parquet(filepath, **kwargs)


def write_geom(
    gdf: gpd.GeoDataFrame, filepath: Path, write_covering_bbox: bool = False
) -> None:
    """Save a GeoDataFrame to a parquet file.

    Args:
        gdf: The GeoDataFrame to save.
        filepath: Path to the output parquet file.
        write_covering_bbox: Whether to write the covering bounding box to the file.
    """
    gdf.to_parquet(
        filepath,
        engine="pyarrow",
        compression="zstd",
        compression_level=9,
        row_group_size=max(min(10_000, len(gdf)), 1),
        schema_version="1.1.0",
        write_covering_bbox=write_covering_bbox,
    )


def read_params(filepath: Path) -> Any:
    """Load a dictionary from a JSON or YAML file.

    Args:
        filepath: Path to the JSON or YAML file.

    Returns:
        A dictionary containing the data.

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix: str = filepath.suffix
    if suffix == ".json":
        return json.loads(filepath.read_text())
    elif suffix in (".yml", ".yaml"):
        return yaml.safe_load(filepath.read_text())
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported formats are .json, .yml, .yaml"
        )


def _convert_paths_to_strings(obj: Any) -> Any:
    """Recursively convert Path objects to strings in nested data structures.

    Args:
        obj: The object to convert.

    Returns:
        The object with Path objects converted to strings.
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: _convert_paths_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_convert_paths_to_strings(item) for item in obj)
    else:
        return obj


def write_params(d: dict, filepath: Path) -> None:
    """Save a dictionary to a YAML file.

    Args:
        d: The dictionary to save.
        filepath: Path to the output YAML file.
    """
    # Convert Path objects to strings before saving
    d_converted = _convert_paths_to_strings(d)
    with open(filepath, "w") as f:
        yaml.dump(d_converted, f, default_flow_style=False, sort_keys=False)


def calculate_scaling(
    da: xr.DataArray | np.ndarray,
    min_value: float,
    max_value: float,
    precision: float,
    offset: float | int = 0.0,
) -> tuple[float, str, str]:
    """This function calculates the scaling factor and output dtype for a fixed scale and offset codec.

    The expected minimum and maximum values along with the precision are used to determine the number
    of bits required to represent the data. The scaling factor is then
    calculated to scale the original data to the required integer
    range. The output dtype is determined based on the number of bits
    required.

    Note that for very high precision in relation to the min and max values,
    there may be some issues due to rounding and the given factors may
    become slighly imprecise.

    Args:
        da: The input xarray DataArray to be encoded.
        min_value: The minimum expected value of the original data. Outside this range
            the data may start to behave unexpectedly.
        max_value: The maximum expected value of the original data. Outside this range
            the data may start to behave unexpectedly.
        precision: The precision of the data, i.e. the maximum difference between the
            original and decoded data.
        offset: The offset to apply to the original data before scaling.

    Returns:
        scaling_factor: The scaling factor to apply to the original data.
        out_dtype: The output dtype to use for the fixed scale and offset codec.

    Raises:
        ValueError: If more than 64 bits are required for the given precision and range
            and thus the data cannot be represented with a fixed scale and offset codec.
    """
    assert min_value < max_value, "min_value must be less than max_value"
    assert precision > 0, "precision must be greater than 0"

    min_with_offset: float = min_value + offset
    max_with_offset: float = max_value + offset

    max_abs_value: float = max(abs(min_with_offset), abs(max_with_offset))

    steps_required: int = int(max_abs_value / precision / 2) + 1

    bits_required: int = steps_required.bit_length()

    steps_available: int = 2**bits_required

    if min_with_offset < 0:
        bits_required += 1  # need to account for the sign bit
        out_dtype_prefix: str = ""
    else:
        out_dtype_prefix: str = "u"

    scaling_factor: float = steps_available / max_abs_value

    if bits_required <= 8:
        out_dtype: str = out_dtype_prefix + "int8"
    elif bits_required <= 16:
        out_dtype: str = out_dtype_prefix + "int16"
    elif bits_required <= 32:
        out_dtype: str = out_dtype_prefix + "int32"
    elif bits_required <= 64:
        out_dtype: str = out_dtype_prefix + "int64"
    else:
        raise ValueError("Too many bits required for precision and range")

    in_dtype: str = da.dtype.name

    return scaling_factor, in_dtype, out_dtype


def parse_and_set_zarr_CRS(da: xr.DataArray) -> xr.DataArray:
    """Parse the _CRS attribute of an xarray DataArray and set it as the CRS using rioxarray.

    The _CRS attribute is expected to be a dictionary with a "wkt" key containing the WKT string.

    Args:
        da: The xarray DataArray to parse and set the CRS for.

    Returns:
        The xarray DataArray with the CRS set.
    """
    if "_CRS" in da.attrs:
        crs_attr = da.attrs["_CRS"]
        if isinstance(crs_attr, dict) and "wkt" in crs_attr:
            wkt: str = crs_attr["wkt"]
            da.rio.write_crs(pyproj.CRS(wkt), inplace=True)
            del da.attrs["_CRS"]
    return da


def read_zarr(zarr_folder: Path | str) -> xr.DataArray:
    """Open a zarr file as an xarray DataArray.

    If the data is a boolean type and does not have a _FillValue attribute,
    a _FillValue attribute with value None will be added.

    The _CRS attribute will be converted to a pyproj CRS object following
    the conventions used by rioxarray. The original _CRS attribute will be removed.

    Args:
        zarr_folder: The path to the zarr folder.

    Raises:
        ValueError: If the zarr file contains multiple data variables.
        FileNotFoundError: If the zarr folder does not exist.

    Returns:
        The xarray DataArray.
    """
    # it is rather odd, but in some cases using mask_and_scale=False is necessary
    # or dtypes start changing, seemingly randomly
    # consolidated metadata is off-spec for zarr, therefore we set it to False
    path: Path = Path(zarr_folder)
    if not path.exists():
        raise FileNotFoundError(f"Zarr folder {zarr_folder} does not exist")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ZarrUserWarning)
        ds: xr.Dataset = xr.open_dataset(
            zarr_folder,
            engine="zarr",
            chunks={},
            consolidated=False,
            mask_and_scale=False,
        )
    if "spatial_ref" in ds.data_vars:
        spatial_ref_data = ds["spatial_ref"]
        ds = ds.drop_vars("spatial_ref")
        ds = ds.assign_coords(spatial_ref=spatial_ref_data)
    if len(ds.data_vars) > 1:
        raise ValueError(
            f"Only one data variable is supported, found multiple: {list(ds.data_vars)}"
        )

    da: xr.DataArray = ds[list(ds.data_vars)[0]]

    if da.dtype == bool and "_FillValue" not in da.attrs:
        da.attrs["_FillValue"] = None

    da = parse_and_set_zarr_CRS(da)

    return da


def to_wkt(crs_obj: int | pyproj.CRS | rasterio.crs.CRS) -> str:
    """Convert a CRS object (pyproj CRS, rasterio CRS or EPSG code) to a WKT string.

    Args:
        crs_obj: The CRS object to convert.

    Raises:
        TypeError: If the CRS object is not a pyproj CRS, rasterio CRS or EPSG code.

    Returns:
        The WKT string representation of the CRS.
    """
    if isinstance(crs_obj, int):  # EPSG code
        return CRS.from_epsg(crs_obj).to_wkt()
    elif isinstance(crs_obj, CRS):  # Pyproj CRS
        return crs_obj.to_wkt()
    elif isinstance(crs_obj, rasterio.crs.CRS):
        return CRS(crs_obj.to_wkt()).to_wkt()
    else:
        raise TypeError("Unsupported CRS type")


def check_attrs(da1: xr.DataArray, da2: xr.DataArray) -> bool:
    """Check if the attributes of two xarray DataArrays are equal.

    The _CRS and grid_mapping attributes are ignored in the comparison.

    Args:
        da1: The first xarray DataArray.
        da2: The second xarray DataArray.

    Returns:
        True if the attributes are equal, False otherwise.
    """
    if "_CRS" in da1.attrs:
        del da1.attrs["_CRS"]
    if "_CRS" in da2.attrs:
        del da2.attrs["_CRS"]
    if "grid_mapping" in da1.attrs:
        del da1.attrs["grid_mapping"]
    if "grid_mapping" in da2.attrs:
        del da2.attrs["grid_mapping"]

    assert len(da1.attrs) == len(da2.attrs), "number of attributes is not equal"

    for key, value in da1.attrs.items():
        # perform a special check for nan values, which are not equal to each other
        if (
            key == "_FillValue"
            and isinstance(value, (float, np.float32, np.float64))
            and np.isnan(value)
        ):
            assert np.isnan(da2.attrs["_FillValue"]), f"attribute {key} is not equal"
        else:
            assert da1.attrs[key] == da2.attrs[key], f"attribute {key} is not equal"

    return True


def _chunk_index_to_region(
    chunk_structure: tuple[tuple[int, ...], ...],
    block_index: tuple[int, ...],
) -> tuple[slice, ...]:
    """Convert a Dask block index into absolute array slices.

    Args:
        chunk_structure: Chunk sizes for each array dimension.
        block_index: Block index for each array dimension.

    Returns:
        The selection tuple describing the block position in the full array.
    """
    selection: list[slice] = []
    for dimension_chunks, dimension_index in zip(
        chunk_structure, block_index, strict=True
    ):
        start: int = sum(dimension_chunks[:dimension_index])
        stop: int = start + dimension_chunks[dimension_index]
        selection.append(slice(start, stop))
    return tuple(selection)


def _store_dask_array_blocks(
    da: dask.array.core.Array,
    store_target: Any,
    progress: bool,
    target_write_size_mb: int = 3000,
) -> None:
    """Store a Dask array into a Zarr target block by block.

    Args:
        da: Source Dask array whose existing block structure defines
            the write granularity.
        store_target: Writable Zarr target array.
        progress: Whether to wrap the block iterator in a progress bar.
        target_write_size_mb: Target size in megabytes for each write operation.
    """
    block_indices: Any = np.ndindex(*da.numblocks)

    array_blocks = da.blocks

    # the last chunk may be smaller than the specified chunk size,
    # which can cause a RuntimeWarning when using FixedScaleOffset with astype.
    # This can be safely ignored in this context.
    with np.errstate(invalid="ignore"):
        stores: list = []
        for block_index in block_indices:
            region = _chunk_index_to_region(da.chunks, block_index)
            block = array_blocks[block_index]
            stores.append(
                dask.array.store(
                    block,
                    store_target,
                    regions=region,
                    lock=False,
                    compute=False,
                    return_stored=False,
                )
            )

        # target chunk size of around 3000MB, but at least 1 block
        chunk_size: int = (
            np.prod([da.chunks[i][0] for i in range(len(da.chunks))])
            * da.dtype.itemsize
        )
        batch_size: int = max(1, target_write_size_mb * 1_000_000 // chunk_size)

        batches: Iterable[int] = list(range(0, len(stores), batch_size))
        if progress:
            batches = tqdm(
                batches,
                total=int(np.ceil(len(stores) / batch_size)),
                desc="Writing progress",
                leave=False,
            )

        for i in batches:
            batch = stores[i : i + batch_size]
            dask.compute(*batch)


def _normalize_shards(
    chunk_spec: dict[str, int],
    requested_shards: dict[str, int] | None,
    dimension_sizes: dict[str, int],
) -> dict[str, int] | None:
    """Normalize shard sizes from chunk multiples to exact shard sizes.

    Args:
        chunk_spec: Chunk sizes for each array dimension.
        requested_shards: Requested shard sizes expressed as the number of chunks
            per shard for a subset of dimensions.
        dimension_sizes: Full array size for each dimension.

    Returns:
        A complete shard specification using chunk-sized shards for unspecified
        dimensions, or None when no sharding is requested. Each shard size is an
        exact multiple of the chunk size for its dimension.

    Raises:
        ValueError: If a requested shard dimension does not exist.
        ValueError: If a requested shard size is smaller than 1.
    """
    if requested_shards is None:
        return None

    invalid_dimensions = sorted(set(requested_shards) - set(chunk_spec))
    if invalid_dimensions:
        raise ValueError(
            f"Shard dimensions {invalid_dimensions} are not present in the array"
        )

    shard_spec: dict[str, int] = chunk_spec.copy()
    for dim_name, requested_shard_size in requested_shards.items():
        if requested_shard_size < 1:
            raise ValueError(
                f"Shard size for dimension '{dim_name}' must be at least 1"
            )

        chunk_size: int = chunk_spec[dim_name]
        dimension_size: int = dimension_sizes[dim_name]

        requested_shard_size_in_elements: int = requested_shard_size * chunk_size
        minimum_full_shard_size: int = (
            (dimension_size + chunk_size - 1) // chunk_size
        ) * chunk_size
        shard_spec[dim_name] = min(
            requested_shard_size_in_elements, minimum_full_shard_size
        )

    return shard_spec


def write_zarr(
    da: xr.DataArray,
    path: str | Path | None,
    crs: int | pyproj.CRS,
    shards: dict[str, int] | None = None,
    filters: list | None = None,
    compression_level: int = 18,
    progress: bool = True,
) -> xr.DataArray:
    """Save an xarray DataArray to a zarr file.

    Args:
        da: The xarray DataArray to save.
        path: The path to the zarr file. If None, the file will be saved to a temporary location.
        crs: The coordinate reference system to use.
        shards: A dictionary with the number of chunks per shard for each
            dimension. If provided, these will be used instead of chunk-sized
            shards. Default is None.
        filters: A list of filters to apply. Default is [].
        compression_level: The level of compression for the ZSTD compressor (1-22). Default is 18.
        progress: Whether to show a progress bar. Default is True.

    Returns:
        The xarray DataArray saved to disk.

    Raises:
        ValueError: If the DataArray has invalid dimensions or attributes.
        ValueError: If the DataArray dtype is float64.
        ValueError: If shards contains an unknown dimension or non-positive size.

    """
    assert isinstance(da, xr.DataArray), "da must be an xarray DataArray"
    assert "longitudes" not in da.dims, "longitudes should be x"
    assert "latitudes" not in da.dims, "latitudes should be y"

    if filters is None:
        filters: list = []

    if "y" in da.dims and "x" in da.dims:
        assert da.dims[-2] == "y", "y should be the second last dimension"
        assert da.dims[-1] == "x", "x should be the last dimension"

    if da.dtype == np.float64:
        raise ValueError("DataArray dtype should be float32, not float64")

    assert "_FillValue" in da.attrs, "Fill value must be set"
    if da.dtype == bool:
        assert da.attrs["_FillValue"] is None, (
            f"Fill value must be None, not {da.attrs['_FillValue']}"
        )
    # for integer types, fill value must not be nan
    elif np.issubdtype(da.dtype, np.integer):
        assert ~np.isnan(da.attrs["_FillValue"]), (
            f"Fill value must not be nan, not {da.attrs['_FillValue']}"
        )
    # for float types, fill value must be nan
    else:
        assert np.isnan(da.attrs["_FillValue"]), (
            f"Fill value must be nan, not {da.attrs['_FillValue']}"
        )

    if path is None:
        is_temporary_file = True
        path: Path = Path(f"temp_{dask.tokenize.tokenize(da)}.zarr")
    else:
        path: Path = Path(path)
        is_temporary_file = False

    with tempfile.TemporaryDirectory(delete=not is_temporary_file) as tmp_dir:
        tmp_zarr = Path(tmp_dir) / path.name

        da.name = path.stem.split("/")[-1]

        da: xr.DataArray = da.drop_vars([v for v in da.coords if v not in da.dims])
        if "y" in da.dims and "x" in da.dims:
            da.attrs["_CRS"] = {"wkt": to_wkt(crs)}

        chunk_spec: dict[str, int] = {}
        for dim in da.dims:
            dim_name = str(dim)
            requested_chunk_size = (
                da.chunksizes[dim_name][0]
                if dim_name in da.chunksizes
                else da.sizes[dim_name]
            )
            chunk_spec[dim_name] = min(requested_chunk_size, da.sizes[dim_name])

        compressor: ZstdCodec = ZstdCodec(
            level=compression_level,
        )

        dimension_sizes: dict[str, int] = {
            str(dim): da.sizes[str(dim)] for dim in da.dims
        }
        shard_spec = _normalize_shards(chunk_spec, shards, dimension_sizes)

        # when sharding is enabled, the write block size is determined by the shard size, otherwise by the chunk size
        write_block_spec: dict[str, int] = (
            shard_spec if shard_spec is not None else chunk_spec
        )

        da = da.chunk(write_block_spec)

        # to display maps in QGIS, the "other" dimensions must have a chunk size of 1
        storage_chunks = tuple(chunk_spec[str(dim)] for dim in da.dims)

        array_encoding: dict[str, Any] = {
            "compressors": (compressor,),
            "chunks": storage_chunks,
            "filters": filters,
        }

        if shard_spec is not None:
            storage_shards = tuple(shard_spec[str(dim)] for dim in da.dims)
            array_encoding["shards"] = storage_shards

        assert isinstance(da.name, str)
        encoding: dict[Hashable, dict[str, Any]] = {da.name: array_encoding}
        for coord in da.coords:
            encoding[coord] = {
                "compressors": (compressor,),
                "chunks": (da.coords[coord].size,),
            }

        da.to_zarr(
            store=tmp_zarr,
            mode="w",
            encoding=encoding,
            zarr_format=3,
            consolidated=False,  # consolidated metadata is off-spec for zarr, therefore we set it to False
            write_empty_chunks=True,
            compute=False,
        )

        # Open the target array in the Zarr store for writing
        target = zarr.open_array(
            store=tmp_zarr,
            zarr_format=3,
            path=da.name,
            mode="r+",
        )
        store_target = cast(Any, target)

        # When sharding is enabled we rechunk the source data to the shard layout
        # so each write call covers one shard instead of one storage chunk.
        da = da.chunk(shard_spec) if shard_spec is not None else da
        _store_dask_array_blocks(da.data, store_target, progress)

        if not is_temporary_file:
            folder: Path = path.parent
            folder.mkdir(parents=True, exist_ok=True)

            if path.exists():
                shutil.rmtree(path)
            shutil.move(tmp_zarr, folder)
        if is_temporary_file:
            path = tmp_zarr

    da_disk: xr.DataArray = read_zarr(path)

    # perform some asserts to check if the data was written and read correctly
    assert da.dtype == da_disk.dtype, "dtype mismatch"
    assert check_attrs(da, da_disk), "attributes mismatch"
    assert da.dims == da_disk.dims, "dimensions mismatch"
    assert da.shape == da_disk.shape, "shape mismatch"

    return da_disk


def get_window(
    x: xr.DataArray,
    y: xr.DataArray,
    bounds: tuple[float, float, float, float],
    buffer: int = 0,
    raise_on_out_of_bounds: bool = True,
    raise_on_buffer_out_of_bounds: bool = True,
) -> dict[str, slice]:
    """Get a window for the given x and y coordinates based on the provided bounds and buffer.

    Args:
        x: The x coordinates as an xarray DataArray.
        y: The y coordinates as an xarray DataArray.
        bounds: A tuple of four values representing the bounds in the form (min_x, min_y, max_x, max_y).
        buffer: The buffer size to apply to the bounds. Default is 0.
        raise_on_out_of_bounds: Whether to raise an error if the bounds are out of the x or y coordinate range. Default is True.
        raise_on_buffer_out_of_bounds: Whether to raise an error if the buffer goes out of the x or y coordinate range. Default is True.

    Returns:
        A dictionary with slices for the x and y coordinates, e.g. {"x": slice(start, stop), "y": slice(start, stop)}.

    Raises:
        ValueError: If the bounds are invalid or out of range,
            or if the buffer is invalid,
            or if x or y are empty,
            or the resulting slices are invalid.
    """
    assert x.ndim == 1, "x must be 1-dimensional"
    assert y.ndim == 1, "y must be 1-dimensional"

    if not isinstance(buffer, int):
        raise ValueError("buffer must be an integer")
    if buffer < 0:
        raise ValueError("buffer must be greater than or equal to 0")
    if len(bounds) != 4:
        raise ValueError("bounds must be a tuple of 4 values")
    if bounds[0] >= bounds[2]:
        raise ValueError("bounds must be in the form (min_x, min_y, max_x, max_y)")
    if bounds[1] >= bounds[3]:
        raise ValueError("bounds must be in the form (min_x, min_y, max_x, max_y)")
    if x.size <= 0:
        raise ValueError("x must not be empty")
    if y.size <= 0:
        raise ValueError("y must not be empty")

    # So that we can do item assignment
    bounds: list = list(bounds)

    if bounds[0] < x[0]:
        if raise_on_out_of_bounds:
            raise ValueError("xmin must be greater than x[0]")
        else:
            bounds[0] = x[0].item()
    if bounds[2] > x[-1]:
        if raise_on_out_of_bounds:
            raise ValueError("xmax must be less than x[-1]")
        else:
            bounds[2] = x[-1].item()
    if bounds[1] < y[-1]:
        if raise_on_out_of_bounds:
            raise ValueError("ymin must be greater than y[-1]")
        else:
            bounds[1] = y[-1].item()
    if bounds[3] > y[0]:
        if raise_on_out_of_bounds:
            raise ValueError("ymax must be less than y[0]")
        else:
            bounds[3] = y[0].item()

    # reverse the y array
    y_reversed = y[::-1]

    assert np.all(np.diff(x) >= 0)
    assert np.all(np.diff(y_reversed) >= 0)

    xmin = np.searchsorted(x, bounds[0], side="right")
    xmax = np.searchsorted(x, bounds[2], side="left")

    if bounds[0] - x[xmin - 1] < x[xmin] - bounds[0]:
        xmin -= 1

    if x[xmax - 1] - bounds[2] < bounds[2] - x[xmax]:
        xmax += 1

    if raise_on_buffer_out_of_bounds:
        xmin = xmin - buffer
        xmax = xmax + buffer
    else:
        xmin = max(0, xmin - buffer)
        xmax = min(x.size, xmax + buffer)

    xslice = slice(xmin, xmax)

    ymin = np.searchsorted(y_reversed, bounds[1], side="right")
    ymax = np.searchsorted(y_reversed, bounds[3], side="left")

    if bounds[1] - y_reversed[ymin - 1] < y_reversed[ymin] - bounds[1]:
        ymin -= 1
    if y_reversed[ymax - 1] - bounds[3] < bounds[3] - y_reversed[ymax]:
        ymax += 1

    if raise_on_buffer_out_of_bounds:
        ymin = ymin - buffer
        ymax = ymax + buffer
    else:
        ymin = max(0, ymin - buffer)
        ymax = min(y.size, ymax + buffer)

    ymin = y.size - ymin
    ymax = y.size - ymax

    yslice = slice(ymax, ymin)

    if xslice.start < 0:
        raise ValueError("x slice start is negative")
    if yslice.start < 0:
        raise ValueError("y slice start is negative")
    if xslice.stop > x.size:
        raise ValueError("x slice stop is greater than x size")
    if yslice.stop > y.size:
        raise ValueError("y slice stop is greater than y size")
    if xslice.stop <= xslice.start:
        raise ValueError("x slice is empty")
    if yslice.start >= yslice.stop:
        raise ValueError("y slice is empty")
    return {"x": xslice, "y": yslice}


class AsyncGriddedForcingReader:
    """Thread-safe asynchronous Zarr forcing reader with preload caching.

    This reader uses the Zarr async API for efficient reads, with a workaround
    for occasional Zarr async loading issues.
    """

    array: zarr.Array

    def __init__(
        self,
        filepath: Path,
        variable_name: str,
        asynchronous: bool = True,
    ) -> None:
        """Initialize the async gridded forcing reader.

        Args:
            filepath: Path to the Zarr file containing the forcing data.
            variable_name: Name of the variable to read from the Zarr file.
            asynchronous: Whether to use asynchronous reading. Default is True.

        Raises:
            ValueError: If the variable does not use NaN as fill value.
        """
        self.filepath = filepath
        self.variable_name = variable_name
        self.asynchronous = asynchronous

        # Synchronous store and dataset (metadata and coordinates only)
        self.store = zarr.storage.LocalStore(filepath, read_only=True)
        self.ds = zarr.open_group(self.store, mode="r")

        # Metadata and time index
        time_arr = self.ds["time"]
        assert isinstance(time_arr, zarr.Array)
        time = time_arr[:]
        assert isinstance(time, np.ndarray)

        assert self.ds["time"].attrs.get("calendar") == "proleptic_gregorian"

        time_unit = self.ds["time"].attrs.get("units")
        assert isinstance(time_unit, str)
        time_unit, origin = time_unit.split(" since ")
        pandas_time_unit: str = {
            "seconds": "s",
            "minutes": "m",
            "hours": "h",
            "days": "D",
        }[time_unit]

        self.datetime_index: ArrayDatetime64 = pd.to_datetime(
            time, unit=pandas_time_unit, origin=origin
        ).to_numpy()
        self.time_size = self.datetime_index.size

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ZarrUserWarning)
            # Check if the variable uses NaN as fill value for the retry workaround
            array = self.ds[self.variable_name]

        assert isinstance(array, zarr.Array)
        self.array: zarr.Array = array

        for compressor in self.array.compressors:
            # Blosc is not supported due to known issues with async reading
            if isinstance(compressor, BloscCodec):
                raise ValueError(
                    f"Variable {self.variable_name} uses Blosc compression, which is not supported by AsyncGriddedForcingReader. Please recompress the data using a different codec (e.g., Zstd)."
                )

        fill_value = self.array.fill_value
        # The fill value is NaN if it's a float type and is NaN, or explicitly None for some types
        has_nan_fill = isinstance(fill_value, (float, np.floating)) and np.isnan(
            fill_value
        )

        if not has_nan_fill:
            raise ValueError(
                f"Variable {self.variable_name} does not use NaN as fill value, AsyncGriddedForcingReader requires NaN fill value for retry workaround."
            )

        # The on-disk chunk size along the time dimension - we always load full chunks
        # from disk (e.g. 7 * 24 = 168 for weekly hourly data).
        self.time_chunk_size: int = int(self.array.chunks[1])

        # Chunk-aligned cache: holds the start index and data for the currently loaded chunk.
        self.current_chunk_start_index: int = -1
        self.current_chunk_data: TwoDArrayFloat32 | None = None
        # The time-index at which a background preload has been (or is being) fetched.
        self.preloaded_chunk_start_index: int = -1
        self.preloaded_data_future: asyncio.Task | None = None

        # Async event loop setup
        if self.asynchronous:
            self.loop: asyncio.AbstractEventLoop | None = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
            self.thread.start()

            # Initialize lock in the shared loop
            async def _init_lock() -> tuple[asyncio.Lock, asyncio.Lock]:
                return asyncio.Lock(), asyncio.Lock()

            self.async_lock, self.io_lock = asyncio.run_coroutine_threadsafe(
                _init_lock(), self.loop
            ).result()
        else:
            self.loop = None
            self.async_lock = None
            self.io_lock = None

    def load(self, start_index: int, end_index: int) -> TwoDArrayFloat32:
        """Safe synchronous load (only used if asynchronous=False).

        Returns:
            The requested data slice.
        """
        assert isinstance(self.array, zarr.Array)
        data = self.array[:, start_index:end_index]
        assert (
            isinstance(data, np.ndarray) and data.dtype == np.float32 and data.ndim == 2
        )
        return data  # ty:ignore[invalid-return-type]

    async def load_await(self, start_index: int, end_index: int) -> TwoDArrayFloat32:
        """Load data asynchronously via reusable async group.

        Returns:
            The requested data slice (not a copy - caller must copy if needed).

        Raises:
            IOError: If the async load returns only NaN values after multiple attempts.
        """
        assert self.io_lock is not None
        async with self.io_lock:
            # Select the variable array from the pre-opened async group.
            arr: zarr.AsyncArray[Any] = self.array.async_array

            attempts: int = 100

            # Try up to 100 times
            for _ in range(attempts):
                data = await arr.getitem((slice(None), slice(start_index, end_index)))

                if not np.any(np.isnan(data)):
                    assert (
                        isinstance(data, np.ndarray)
                        and data.dtype == np.float32
                        and data.ndim == 2
                    )
                    return data  # ty:ignore[invalid-return-type]
                print(
                    f"Async load returned NaN values for indices {start_index}:{end_index}, retrying..."
                )

            else:
                raise IOError(
                    f"Async load failed after {attempts} attempts for indices {start_index}:{end_index}"
                )

    async def preload_chunk(self, chunk_start: int) -> TwoDArrayFloat32 | None:
        """Preload the chunk starting at chunk_start asynchronously.

        Args:
            chunk_start: The time index at which the chunk to preload begins.

        Returns:
            The preloaded chunk data, or None if beyond available data.
        """
        if chunk_start >= self.time_size:
            return None
        chunk_end: int = min(chunk_start + self.time_chunk_size, self.time_size)
        return await self.load_await(chunk_start, chunk_end)

    async def read_timestep_async(
        self, start_index: int, end_index: int
    ) -> tuple[TwoDArrayFloat32, np.datetime64]:
        """Core async read with chunk-aligned caching and background preloading.

        Loads the full on-disk chunk that contains the requested timesteps,
        caches it, and schedules the next chunk for background preloading so
        that subsequent requests within the same or the next chunk are cheap.

        Args:
            start_index: The starting index of the time slice to read.
            end_index: The exclusive ending index of the time slice to read.

        Returns:
            A tuple of (requested data slice as a NumPy array, start datetime of the slice).

        Raises:
            ValueError: If the requested index is out of bounds or spans more
                than one on-disk chunk.
            IOError: If the async read returns incomplete data.
        """
        if start_index < 0 or end_index > self.time_size:
            raise ValueError(f"Index out of bounds ({start_index}:{end_index})")

        n: int = end_index - start_index
        start_date: np.datetime64 = self.datetime_index[start_index]

        # Determine which on-disk chunk this request falls in.
        chunk_start: int = (start_index // self.time_chunk_size) * self.time_chunk_size
        chunk_end: int = min(chunk_start + self.time_chunk_size, self.time_size)
        offset: int = start_index - chunk_start

        # Requests must be aligned with chunk boundaries and never cross it.
        # This simplifies the reader and ensures that the source data is saved
        # efficiently for the intended access pattern.
        if n > self.time_chunk_size:
            raise ValueError(
                f"Requested {n} timesteps exceeds on-disk chunk size {self.time_chunk_size}."
            )
        if (start_index % n != 0) or (offset + n > (chunk_end - chunk_start)):
            raise ValueError(
                f"Requested slice {start_index}:{end_index} is not aligned with "
                f"the chunk size {self.time_chunk_size} or spacing {n}."
            )

        assert self.async_lock is not None
        async with self.async_lock:
            chunk_data: TwoDArrayFloat32 | None = None

            # Cache hit: the required chunk is already in memory.
            if (
                self.current_chunk_data is not None
                and self.current_chunk_start_index == chunk_start
            ):
                chunk_data = self.current_chunk_data

            # Preload hit: the required chunk was being preloaded in the background.
            elif (
                self.preloaded_data_future is not None
                and self.preloaded_chunk_start_index == chunk_start
            ):
                try:
                    chunk_data = await self.preloaded_data_future
                except Exception:
                    chunk_data = None

            # Cache miss: cancel any pending (wrong) preload and load from disk.
            if chunk_data is None:
                if self.preloaded_data_future and not self.preloaded_data_future.done():
                    self.preloaded_data_future.cancel()
                    try:
                        await self.preloaded_data_future
                    except asyncio.CancelledError:
                        pass
                chunk_data = await self.load_await(chunk_start, chunk_end)

            expected_chunk_len: int = chunk_end - chunk_start
            if chunk_data.shape[1] != expected_chunk_len:
                raise IOError(
                    "Async read returned incomplete data; possible disk contention"
                )

            # Update the chunk cache.
            self.current_chunk_start_index = chunk_start
            self.current_chunk_data = chunk_data

            # Schedule preload of the next chunk unless it is already in flight.
            next_chunk_start: int = chunk_start + self.time_chunk_size
            assert self.loop is not None
            if self.preloaded_chunk_start_index != next_chunk_start:
                if self.preloaded_data_future and not self.preloaded_data_future.done():
                    self.preloaded_data_future.cancel()
                    try:
                        await self.preloaded_data_future
                    except asyncio.CancelledError:
                        pass
                self.preloaded_chunk_start_index = next_chunk_start
                self.preloaded_data_future = self.loop.create_task(
                    self.preload_chunk(next_chunk_start)
                )

            # Slice out only the requested timesteps from the cached chunk.
            return chunk_data[:, offset : offset + n], start_date

    def get_index(self, date: datetime.datetime) -> int:
        """Get the time index for a given datetime.

        Uses binary search for correctness and falls back to an O(1) check
        against the current chunk boundaries for the common sequential-access case.

        Args:
            date: The datetime to find the index for.

        Returns:
            The integer index for the given date.

        Raises:
            ValueError: If the date is not found in the time index.
        """
        numpy_date = np.datetime64(date, "ns")

        # Very fast (lol): check whether the date falls within the currently loaded chunk.
        if self.current_chunk_start_index >= 0:
            chunk_end: int = min(
                self.current_chunk_start_index + self.time_chunk_size, self.time_size
            )
            chunk_slice = self.datetime_index[
                self.current_chunk_start_index : chunk_end
            ]
            local_idx: npt.NDArray[np.intp] = np.where(chunk_slice == numpy_date)[0]
            if local_idx.size > 0:
                return int(self.current_chunk_start_index + local_idx[0])

        # Still fast: check whether the date falls within the next chunk.
        # This the most logical for climate data. That the model requests the next
        # chunk.
        next_chunk_start: int = self.current_chunk_start_index + self.time_chunk_size
        if self.current_chunk_start_index >= 0 and next_chunk_start < self.time_size:
            next_chunk_end: int = min(
                next_chunk_start + self.time_chunk_size, self.time_size
            )
            next_chunk_slice = self.datetime_index[next_chunk_start:next_chunk_end]
            local_idx_next: npt.NDArray[np.intp] = np.where(
                next_chunk_slice == numpy_date
            )[0]
            if local_idx_next.size > 0:
                return int(next_chunk_start + local_idx_next[0])

        # Full search via binary search (handles non-sequential access)
        # This should happen on the very first access and when the model were
        # to jump around in time, which it typically shouldn't do.
        idx: int = int(np.searchsorted(self.datetime_index, numpy_date))
        if idx >= self.time_size or self.datetime_index[idx] != numpy_date:
            raise ValueError(f"Date {date} not found in {self.filepath}")
        return idx

    def read_timestep(
        self, date: datetime.datetime, n: int = 1
    ) -> tuple[npt.NDArray[Any], np.datetime64]:
        """Return n timesteps starting at date, loading from the on-disk chunk as needed.

        On the first call (or whenever the required chunk is not cached) the full
        on-disk chunk is fetched from disk and the next chunk is queued for
        background pre-loading. Subsequent calls for timesteps within the same
        chunk are served entirely from memory.

        Args:
            date: Start datetime of the slice to return.
            n: Number of consecutive timesteps to return. Must not exceed the
               on-disk chunk size along the time dimension.

        Returns:
            A tuple of (Array of shape (n, y, x) with dtype float32, start datetime of the slice as np.datetime64).

        Raises:
            ValueError: If the requested range exceeds available data or the
                on-disk chunk size.
        """
        start_index: int = self.get_index(date)
        end_index: int = start_index + n
        if end_index > self.time_size:
            raise ValueError(
                f"Requested {n} timesteps from {date} exceeds available range"
            )

        start_date: np.datetime64 = self.datetime_index[start_index]

        if self.asynchronous:
            coro = self.read_timestep_async(start_index, end_index)
            assert isinstance(self.loop, asyncio.AbstractEventLoop)
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            return future.result()
        else:
            # Synchronous path: respect chunk alignment for consistency.
            chunk_start: int = (
                start_index // self.time_chunk_size
            ) * self.time_chunk_size
            chunk_end: int = min(chunk_start + self.time_chunk_size, self.time_size)
            offset: int = start_index - chunk_start

            # Requests must be aligned with chunk boundaries and never cross it.
            if n > self.time_chunk_size:
                raise ValueError(
                    f"Requested {n} timesteps exceeds on-disk chunk size {self.time_chunk_size}."
                )
            if (start_index % n != 0) or (offset + n > (chunk_end - chunk_start)):
                raise ValueError(
                    f"Requested slice {start_index}:{end_index} is not aligned with "
                    f"the chunk size {self.time_chunk_size} or spacing {n}."
                )

            if (
                self.current_chunk_data is None
                or self.current_chunk_start_index != chunk_start
            ):
                self.current_chunk_data = self.load(chunk_start, chunk_end)
                self.current_chunk_start_index = chunk_start

            return self.current_chunk_data[:, offset : offset + n], start_date

    def close(self) -> None:
        """Clean up this instance's async resources."""
        if not self.asynchronous:
            return

        async def cleanup() -> None:
            """Cancel this instance's pending tasks and close async group."""
            if self.preloaded_data_future and not self.preloaded_data_future.done():
                self.preloaded_data_future.cancel()
            # Stop the loop
            asyncio.get_event_loop().stop()

        if self.loop and self.loop.is_running():
            try:
                # Because we are not writing, we can just cancel the pending preload
                # task and stop the loop without waiting for it to finish.
                # This allows for a much faster shutdown, especially if the loop is currently waiting on a slow disk read.
                asyncio.run_coroutine_threadsafe(cleanup(), self.loop)
            except Exception:
                pass
            # Don't join the thread - it's a daemon and will die when the process dies


class WorkingDirectory:
    """A context manager for temporarily changing the current working directory.

    Usage:
        with WorkingDirectory('/path/to/new/directory'):
            # Code executed here will have the new directory as the CWD
    """

    def __init__(self, new_path: Path) -> None:
        """Initializes the context manager with the path to change to.

        Args:
            new_path: The path to the directory to change into.
        """
        self._new_path = new_path

    def __enter__(self) -> WorkingDirectory:
        """Enters the context, changing the current working directory.

        Returns:
            The context manager instance.
        """
        # Store the current working directory
        self._original_path = os.getcwd()

        # Change to the new directory
        os.chdir(self._new_path)

        # Return self (optional, but common)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exits the context, reverting to the original working directory.

        Args:
            exc_type: The type of exception raised (if any).
            exc_val: The exception instance raised (if any).
            exc_tb: The traceback of the exception raised (if any).
        """
        # Change back to the original directory
        os.chdir(self._original_path)


class RemoteFile:
    """A file-like object that reads from a remote URL using HTTP Range headers.

    This class mimics a file object (seek, read) but fetches data on-demand
    using HTTP Range requests. It includes retry logic for robust downloads.
    """

    def __init__(self, url: str, max_retries: int = 5, base_delay: float = 1.0) -> None:
        """Initialize the RemoteFile.

        Args:
            url: The URL of the remote file.
            max_retries: Maximum number of retries for HTTP requests.
            base_delay: Base delay in seconds for exponential backoff.

        Raises:
            OSError: If the URL cannot be accessed.
        """
        self.url_original = url
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Resolve redirects and get size
        resp = self._request_with_retry("HEAD", url, allow_redirects=True)

        # Confirm range support
        range_supported = (
            resp.status_code == 200 and resp.headers.get("Accept-Ranges") == "bytes"
        )

        if not range_supported:
            # Close the HEAD response before overwriting
            resp.close()
            # Fallback to GET with Range if HEAD fails or doesn't confirm support
            resp = self._request_with_retry(
                "GET",
                url,
                headers={"Range": "bytes=0-0"},
                stream=True,
                allow_redirects=True,
            )
            if resp.status_code != 206:
                resp.close()
                raise OSError(
                    f"Server does not support HTTP Range requests (Expected 206, got {resp.status_code})"
                )

        try:
            if resp.url is None:
                raise OSError("Failed to resolve URL: response URL is None")
            self.url: str = resp.url

            if "Content-Range" in resp.headers:
                self.size = int(resp.headers["Content-Range"].split("/")[-1])
            else:
                self.size = int(resp.headers.get("Content-Length", 0))
        finally:
            # Ensure connection is closed
            resp.close()

        self.pos = 0

    def _request_with_retry(
        self, method: str, url: str, **kwargs: Any
    ) -> requests.Response:
        """Execute HTTP request with exponential backoff retry.

        Args:
            method: HTTP method (e.g. "GET").
            url: The URL to request.
            **kwargs: Additional arguments for requests.request.

        Returns:
            The response object.

        Raises:
            OSError: If the request fails after maximum retries.
        """
        last_error: str | None = None
        resp: requests.Response | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.request(method, url, timeout=30, **kwargs)
                if resp.status_code in [200, 206]:
                    return resp
                if resp.status_code in [429, 500, 502, 503, 504]:
                    last_error = f"HTTP {resp.status_code}"
                    # Transient error, continue to retry
                else:
                    # Likely a permanent error (404, 403, etc)
                    # Return response so caller can handle the status code
                    return resp
            except requests.RequestException as e:
                last_error = str(e)

            if attempt < self.max_retries:
                sleep_time = self.base_delay * (2**attempt)
                time.sleep(sleep_time)

        if resp is not None:
            # Return the last failed response
            return resp

        raise OSError(
            f"Failed to request {url} after {self.max_retries} retries: {last_error}"
        )

    def seek(self, offset: int, whence: int = 0) -> int:
        """Move to a new file position.

        Args:
            offset: The offset.
            whence: The reference point (0=start, 1=current, 2=end).

        Returns:
            The new file position.
        """
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        """Return the current file position."""
        return self.pos

    def read(self, size: int = -1) -> bytes:
        """Read bytes from the remote file.

        Args:
            size: Number of bytes to read. -1 means read until end.

        Returns:
            The read bytes.

        Raises:
            OSError: If reading from the URL fails.
        """
        if size == 0:
            return b""

        if size == -1:
            range_header = f"bytes={self.pos}-"
        else:
            end = self.pos + size - 1
            range_header = f"bytes={self.pos}-{end}"

        headers = {"Range": range_header}
        try:
            resp = self._request_with_retry("GET", self.url, headers=headers)
        except OSError as e:
            raise OSError(f"Failed to read from {self.url}: {e}")

        # If we asked for a range but got 200, it means the server sent the whole file.
        # This is only okay if we requested from the beginning (pos=0) and wanted everything (size=-1)
        # or the whole file matches the requested size.
        # However, for simplicity and safety in random access:
        # If we expect a partial response (which we always do here with Range header),
        # we should require 206 Partial Content unless we happen to request the exact full content
        # effectively making 200 OK valid. But typically 206 is strict for Range.
        is_partial_request = self.pos > 0 or (size != -1 and size < self.size)

        if is_partial_request and resp.status_code == 200:
            # Server ignored Range header and sent full file - this is bad for seek/partial read
            raise OSError(
                f"Server returned 200 OK but 206 Partial Content was expected for range {range_header}"
            )

        if resp.status_code not in [200, 206]:
            raise OSError(f"Failed to read from {self.url}: {resp.status_code}")

        data = resp.content
        self.pos += len(data)
        return data

    def seekable(self) -> bool:
        """Return True if the file is seekable."""
        return True


def fetch_and_save(
    url: str,
    file_path: Path,
    overwrite: bool = False,
    max_retries: int = 3,
    delay_seconds: float | int = 5,
    double_delay: bool = False,
    chunk_size: int = 16384,
    session: requests.Session | None = None,
    params: None | dict[str, Any] = None,
    timeout: float | int = 30,
    show_progress: bool = True,
    verbose: bool = True,
    decompress: str | None = None,
) -> bool:
    """Fetches data from a URL and saves it to a file, with a retry mechanism.

    This function supports both S3 and HTTP(S) URLs. It downloads the file to a
    temporary location and then moves it to the final destination to ensure
    atomicity. It includes retry logic for transient network errors.

    Args:
        url: The URL to fetch data from (S3 or HTTP/HTTPS).
        file_path: The local path to save the file to.
        overwrite: If True, overwrite the file if it already exists.
        max_retries: The maximum number of times to retry a failed download.
        delay_seconds: The delay in seconds between retries.
        double_delay: If True, double the delay between retries on each attempt.
        chunk_size: The chunk size for streaming downloads.
        session: An optional requests.Session object to use for HTTP requests.
        params: Optional dictionary of query parameters for HTTP requests.
        timeout: The timeout in seconds for HTTP requests.
        show_progress: Whether to show a progress bar during download.
        verbose: Whether to print download status messages. Default is True.
        decompress: The decompression type to use. Supported values are 'bz2'. Default is None.

    Returns:
        True if the file was successfully downloaded, False otherwise.

    Raises:
        RuntimeError: If the download fails after all retries.
        ValueError: If an unsupported decompression type is specified.
    """
    if file_path.exists() and not overwrite:
        return True

    if decompress is not None and decompress != "bz2":
        raise ValueError(f"Unsupported decompression type: {decompress}")

    if session is None:
        session = requests.Session()

    if url.startswith("s3://"):
        # Fetch from S3 without authentication
        fs = s3fs.S3FileSystem(anon=True)
        attempts = 0
        temp_file = None
        current_delay_seconds: int | float = delay_seconds

        while attempts < max_retries:
            try:
                if verbose:
                    print(f"Downloading {url} to {file_path}")
                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.close()

                # Download from S3
                fs.get(url, temp_file.name)

                # Move the temporary file to the destination
                shutil.move(temp_file.name, file_path)
                return True

            except Exception as e:
                # Log the error
                if verbose:
                    print(
                        f"S3 download failed: {e}. Attempt {attempts + 1} of {max_retries}"
                    )

                # Remove the temporary file if it exists
                if temp_file is not None and os.path.exists(temp_file.name):
                    os.remove(temp_file.name)

                # Increment the attempt counter and wait before retrying
                attempts += 1
                if attempts < max_retries:
                    time.sleep(current_delay_seconds)
                    if double_delay:
                        current_delay_seconds *= 2

        # If all attempts fail, raise an exception
        raise RuntimeError(
            f"Failed to download '{url}' from S3 to '{file_path}' after {max_retries} attempts."
        )

    elif url.startswith("http://") or url.startswith("https://"):
        attempts = 0
        temp_file = None
        current_delay_seconds: int | float = delay_seconds

        while attempts < max_retries:
            try:
                if verbose:
                    print(f"Downloading {url} to {file_path}")
                # Attempt to make the request
                response = session.get(url, stream=True, params=params, timeout=timeout)
                response.raise_for_status()  # Raises HTTPError for bad status codes

                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False)

                # Write to the temporary file
                if show_progress:
                    total_size = int(response.headers.get("content-length", 0))
                    progress_bar = tqdm(
                        total=total_size, unit="B", unit_scale=True, leave=False
                    )
                    if decompress == "bz2":
                        decompressor = bz2.BZ2Decompressor()
                        for data in response.iter_content(chunk_size=chunk_size):
                            temp_file.write(decompressor.decompress(data))
                            progress_bar.update(len(data))
                    else:
                        for data in response.iter_content(chunk_size=chunk_size):
                            temp_file.write(data)
                            progress_bar.update(len(data))
                    progress_bar.close()
                else:
                    if decompress == "bz2":
                        decompressor = bz2.BZ2Decompressor()
                        for data in response.iter_content(chunk_size=chunk_size):
                            temp_file.write(decompressor.decompress(data))
                    else:
                        for data in response.iter_content(chunk_size=chunk_size):
                            temp_file.write(data)

                # Close the temporary file
                temp_file.close()

                # Move the temporary file to the destination
                shutil.move(temp_file.name, file_path)

                return True  # Exit the function after successful write

            except requests.RequestException as e:
                # Log the error
                if verbose:
                    print(
                        f"Request failed: {e}. Attempt {attempts + 1} of {max_retries}"
                    )

                # Remove the temporary file if it exists
                if temp_file is not None and os.path.exists(temp_file.name):
                    os.remove(temp_file.name)

                # Increment the attempt counter and wait before retrying
                attempts += 1
                if attempts < max_retries:
                    time.sleep(current_delay_seconds)
                    if double_delay:
                        current_delay_seconds *= 2

        # If all attempts fail, raise an exception
        raise RuntimeError(
            f"Failed to download '{url}' to '{file_path}' after {max_retries} attempts. "
            "Please check the URL, network connectivity, and destination permissions."
        )
    return False


def fast_rmtree(path: Path) -> None:
    """Deletes a directory recursively using only the fastest native OS command.

    - Windows: RD /S /Q
    - Linux/macOS (POSIX): rm -rf
    - Raises NotImplementedError for other systems.

    Args:
        path: The path to the directory to be deleted.

    Raises:
        FileNotFoundError: If the path does not exist.
        NotImplementedError: If the operating system is not explicitly supported.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Handle files/links separately, as native directory commands expect directories
    if not path.is_dir():
        path.unlink()
        return

    system: str = platform.system()

    if system == "Windows":
        # Windows command: RD /S /Q
        # cmd /C is used to execute the built-in RD command and terminate.
        command = f'cmd /C RD /S /Q "{path}"'
        # Setting shell=True is required to execute cmd /C
        subprocess.run(command, shell=True, check=False)

    elif system in ("Linux", "Darwin"):  # 'Darwin' is macOS
        # POSIX command: rm -rf
        subprocess.run(["rm", "-rf", str(path)], check=False)

    else:
        # Raise an error for unsupported systems instead of falling back
        raise NotImplementedError(
            f"Optimized fast deletion is not implemented for system: {system}"
        )


def create_hash_from_parameters(
    parameters: dict[str, Any], code_path: Path | None = None
) -> str:
    """Create a hash from a dictionary of parameters.

    Args:
        parameters: A dictionary of parameters.
        code_path: Optional path to a file or directory containing code to include in the hash.

    Returns:
        A hexadecimal string representing the hash of the parameters.

    Raises:
        ValueError: If the parameters dictionary contains a key named '_code_content'.
    """

    def make_hashable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            value = str(value.tobytes())
        elif isinstance(value, (xr.DataArray, xr.Dataset)):
            value = dask.tokenize.tokenize(value, ensure_deterministic=True)
        elif isinstance(value, dict):
            value = {k: make_hashable(v) for k, v in value.items()}
        elif isinstance(value, list):
            value = [make_hashable(v) for v in value]
        elif isinstance(value, (pd.DataFrame, gpd.GeoDataFrame)):
            value = joblib.hash(value, hash_name="md5", coerce_mmap=True)
        elif isinstance(value, np.generic):
            value = value.item()
        try:
            json.dumps(value)
        except TypeError, ValueError:
            raise ValueError(f"Value {value} is not JSON serializable")
        return value

    hashable_dict = make_hashable(parameters)

    if code_path is not None:
        if "_code_content" in hashable_dict:
            raise ValueError(
                "The parameters dictionary cannot contain a key named '_code_content'"
            )
        code_content: dict[str, str] = {}
        if code_path.is_file():
            code_content[code_path.name] = code_path.read_text()
        elif code_path.is_dir():
            for root, _, files in sorted(os.walk(code_path)):
                for file in sorted(files):
                    file_path = Path(root) / file
                    try:
                        rel_path = str(file_path.relative_to(code_path))
                        code_content[rel_path] = file_path.read_text()
                    except UnicodeDecodeError:
                        continue
        hashable_dict["_code_content"] = code_content

    hash_: str = hashlib.md5(
        json.dumps(hashable_dict, sort_keys=True).encode()
    ).hexdigest()
    return hash_


def write_hash(path: Path, hash: str) -> None:
    """Write a hash to a file in hexadecimal format.

    Args:
        path: The path to the file where the hash will be written.
        hash: The hash as a str object.
    """
    path.write_text(hash)


def read_hash(path: Path) -> str:
    """Read a hash from a file in hexadecimal format.

    Args:
        path: The path to the file containing the hash.

    Returns:
        The hash as a str object.
    """
    return path.read_text().strip()
