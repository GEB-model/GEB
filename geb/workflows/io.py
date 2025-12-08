"""I/O related functions and classes for the GEB project."""

import asyncio
import datetime
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
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any, overload

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import rasterio
import requests
import s3fs
import xarray as xr
import yaml
import zarr
import zarr.storage
from dask.diagnostics import ProgressBar
from pyproj import CRS
from rasterio.transform import Affine
from tqdm import tqdm
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec
from zarr.codecs.zstd import ZstdCodec

from geb.types import (
    ArrayDatetime64,
    ThreeDArray,
    ThreeDArrayFloat32,
    TwoDArray,
    TwoDArrayFloat32,
)


def read_table(fp: Path) -> pd.DataFrame:
    """Load a parquet file as a pandas DataFrame.

    Args:
        fp: The path to the parquet file.

    Returns:
        The pandas DataFrame.
    """
    return pd.read_parquet(fp, engine="pyarrow")


def write_table(df: pd.DataFrame, fp: Path) -> None:
    """Save a pandas DataFrame to a parquet file.

    brotli is a bit slower but gives better compression,
    gzip is faster to read. Higher compression levels
    generally don't make it slower to read, therefore
    we use the highest compression level for gzip

    Args:
        df: The pandas DataFrame to save.
        fp: The path to the output parquet file.
    """
    df.to_parquet(fp, engine="pyarrow", compression="gzip", compression_level=9)


def read_array(fp: Path) -> np.ndarray:
    """Load a numpy array from a .npz or .zarr file.

    Args:
        fp: The path to the .npz or .zarr file.

    Returns:
        The numpy array.

    Raises:
        ValueError: If the file format is not supported.
    """
    if fp.suffix == ".npz":
        return np.load(fp)["data"]
    elif fp.suffix == ".zarr":
        zarr_object = zarr.load(fp)
        assert isinstance(zarr_object, np.ndarray)
        return zarr_object
    else:
        raise ValueError(f"Unsupported file format: {fp.suffix}")


@overload
def read_grid(
    filepath: Path, layer: int = 1, return_transform_and_crs: bool = False
) -> TwoDArray: ...


@overload
def read_grid(
    filepath: Path, layer: int = 1, return_transform_and_crs: bool = True
) -> tuple[TwoDArray, Affine, str]: ...


@overload
def read_grid(
    filepath: Path, layer: None = None, return_transform_and_crs: bool = False
) -> ThreeDArray: ...


@overload
def read_grid(
    filepath: Path, layer: None = None, return_transform_and_crs: bool = True
) -> tuple[ThreeDArray, Affine, str]: ...


def read_grid(
    filepath: Path, layer: int | None = 1, return_transform_and_crs: bool = False
) -> TwoDArray | ThreeDArray | tuple[TwoDArray | ThreeDArray, Affine, str]:
    """Load a raster grid from a .tif or .zarr file.

    Args:
        filepath: The path to the .tif or .zarr file.
        layer: The layer to load from the .tif file. If None, all layers are loaded. Default is 1.
        return_transform_and_crs: Whether to return the affine transform and CRS along with the data. Default is False.

    Returns:
        The raster data as a numpy array, or a tuple of the raster data, affine transform, and CRS string if return_transform_and_crs is True.

    Raises:
        ValueError: If the file format is not supported.
    """
    if filepath.suffix == ".tif":
        warnings.warn("tif files are now deprecated. Consider rebuilding the model.")
        with rasterio.open(filepath) as src:
            data: TwoDArray | ThreeDArray = src.read(layer)
            data: TwoDArray | ThreeDArray = (
                data.astype(np.float32) if data.dtype == np.float64 else data
            )
            if return_transform_and_crs:
                return data, src.transform, src.crs
            else:
                return data

    elif filepath.suffix == ".zarr":
        store: zarr.storage.LocalStore = zarr.storage.LocalStore(
            filepath, read_only=True
        )
        group: zarr.Group = zarr.open_group(store, mode="r")
        data_array: zarr.Array | zarr.Group = group[filepath.stem]
        assert isinstance(data_array, zarr.Array)
        data = data_array[:]
        if data.dtype == np.float64:
            data: TwoDArrayFloat32 | ThreeDArrayFloat32 = data.asfloat(np.float32)
        if return_transform_and_crs:
            x_array: zarr.Array | zarr.Group = group["x"]
            assert isinstance(x_array, zarr.Array)
            x = x_array[:]
            assert isinstance(x, np.ndarray)
            y_array: zarr.Array | zarr.Group = group["y"]
            assert isinstance(y_array, zarr.Array)
            y = y_array[:]
            assert isinstance(y, np.ndarray)
            x_diff: float = np.diff(x[:]).mean().item()
            y_diff: float = np.diff(y[:]).mean().item()
            transform: Affine = Affine(
                a=x_diff,
                b=0,
                c=x[0] - x_diff / 2,
                d=0,
                e=y_diff,
                f=y[0] - y_diff / 2,
            )
            crs = data_array.attrs["_CRS"]
            assert isinstance(crs, dict)
            wkt: str = crs["wkt"]
            return data, transform, wkt
        else:
            return data
    else:
        raise ValueError("File format not supported.")


def read_geom(filepath: str | Path) -> gpd.GeoDataFrame:
    """Load a geometry for the GEB model from disk.

    Args:
        filepath: Path to the geometry file.

    Returns:
        A GeoDataFrame containing the geometries.

    """
    return gpd.read_parquet(filepath)


def write_geom(gdf: gpd.GeoDataFrame, filepath: Path) -> None:
    """Save a GeoDataFrame to a parquet file.

    Args:
        gdf: The GeoDataFrame to save.
        filepath: Path to the output parquet file.
    """
    gdf.to_parquet(
        filepath,
        engine="pyarrow",
        compression="zstd",
        compression_level=9,
        row_group_size=10_000,
        schema_version="1.1.0",
    )


def read_dict(filepath: Path) -> Any:
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


def write_dict(d: dict, filepath: Path) -> None:
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
    ds: xr.Dataset = xr.open_dataset(
        zarr_folder, engine="zarr", chunks={}, consolidated=False, mask_and_scale=False
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

    if "_CRS" in da.attrs:
        da.rio.write_crs(pyproj.CRS(da.attrs["_CRS"]["wkt"]), inplace=True)
        del da.attrs["_CRS"]

    return da


def to_wkt(crs_obj: int | pyproj.CRS | rasterio.crs.CRS) -> str:  # ty: ignore[unresolved-attribute]
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
    elif isinstance(crs_obj, rasterio.crs.CRS):  # ty: ignore[unresolved-attribute]
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


def check_buffer_size(
    da: xr.DataArray,
    chunks_or_shards: dict[str, int],
    max_buffer_size: int = 2147483647,
) -> None:
    """Check if the buffer size for the given chunks or shards is within the maximum allowed size.

    Args:
        da: The xarray DataArray to check.
        chunks_or_shards: A dictionary with the chunk or shard sizes for each dimension.
        max_buffer_size: The maximum allowed buffer size in bytes. Default is 2GB (2147483647 bytes).

    Raises:
        ValueError: If the buffer size exceeds the maximum allowed size.
    """
    buffer_size = (
        np.prod([size for size in chunks_or_shards.values()]) * da.dtype.itemsize
    )
    if buffer_size >= max_buffer_size:
        raise ValueError(
            f"Buffer size exceeds maximum size, current shards or chunks are {chunks_or_shards}"
        )


def write_zarr(
    da: xr.DataArray,
    path: str | Path,
    crs: int | pyproj.CRS,
    x_chunksize: int = 350,
    y_chunksize: int = 350,
    time_chunksize: int = 1,
    time_chunks_per_shard: int | None = 30,
    filters: list = [],
    compressor: None | BytesBytesCodec = None,
    progress: bool = True,
) -> xr.DataArray:
    """Save an xarray DataArray to a zarr file.

    Args:
        da: The xarray DataArray to save.
        path: The path to the zarr file.
        crs: The coordinate reference system to use.
        x_chunksize: The chunk size for the x dimension. Default is 350.
        y_chunksize: The chunk size for the y dimension. Default is 350.
        time_chunksize: The chunk size for the time dimension. Default is 1.
        time_chunks_per_shard: The number of time chunks per shard. Default is 30. Set to None
            to disable sharding.
        filters: A list of filters to apply. Default is [].
        compressor: The compressor to use. Default is None, using the default Blosc compressor.
        progress: Whether to show a progress bar. Default is True.

    Returns:
        The xarray DataArray saved to disk.

    """
    assert isinstance(da, xr.DataArray), "da must be an xarray DataArray"
    assert "longitudes" not in da.dims, "longitudes should be x"
    assert "latitudes" not in da.dims, "latitudes should be y"

    if "y" in da.dims and "x" in da.dims:
        assert da.dims[-2] == "y", "y should be the second last dimension"
        assert da.dims[-1] == "x", "x should be the last dimension"

    assert da.dtype != np.float64, "should be float32"

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

    path: Path = Path(path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_zarr = Path(tmp_dir) / path.name

        da.name = path.stem.split("/")[-1]

        da: xr.DataArray = da.drop_vars([v for v in da.coords if v not in da.dims])

        chunks, shards = {}, None
        if "y" in da.dims and "x" in da.dims:
            chunks.update(
                {
                    "y": min(y_chunksize, da.sizes["y"]),
                    "x": min(x_chunksize, da.sizes["x"]),
                }
            )
            da.attrs["_CRS"] = {"wkt": to_wkt(crs)}

        if "member" in da.dims:
            member_chunksize = 1  # Use full size as default da.sizes["member"]
            chunks.update({"member": member_chunksize})

        if "time" in da.dims:
            chunks.update({"time": min(time_chunksize, da.sizes["time"])})
            if time_chunks_per_shard is not None:
                shards = chunks.copy()
                shards["time"] = time_chunks_per_shard * chunks["time"]

        if compressor is None:
            compressor: ZstdCodec = ZstdCodec(
                level=22,
            )

        check_buffer_size(da, chunks_or_shards=shards if shards else chunks)

        da: xr.DataArray = da.chunk(shards if shards is not None else chunks)

        # to display maps in QGIS, the "other" dimensions must have a chunk size of 1
        chunks = tuple((chunks[dim] if dim in chunks else 1) for dim in da.dims)

        array_encoding: dict[str, Any] = {
            "compressors": (compressor,),
            "chunks": chunks,
            "filters": filters,
        }

        if shards is not None:
            shards = tuple(
                (shards[dim] if dim in shards else getattr(da, str(dim)).size)
                for dim in da.dims
            )
            array_encoding["shards"] = shards

        assert isinstance(da.name, str)
        encoding: dict[Hashable, dict[str, Any]] = {da.name: array_encoding}
        for coord in da.coords:
            encoding[coord] = {"compressors": (compressor,)}

        to_zarr_partial = partial(
            da.to_zarr,
            store=tmp_zarr,
            mode="w",
            encoding=encoding,
            zarr_format=3,
            consolidated=False,  # consolidated metadata is off-spec for zarr, therefore we set it to False
            write_empty_chunks=True,
        )

        if progress:
            # start writing after 10 seconds, and update every 0.1 seconds
            with ProgressBar(
                minimum=0.1,
                dt=float(os.environ.get("GEB_OVERRIDE_PROGRESSBAR_DT", 0.1)),
            ):
                store = to_zarr_partial()
        else:
            store = to_zarr_partial()

        store.close()

        folder: Path = path.parent
        folder.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.rmtree(path)
        shutil.move(tmp_zarr, folder)

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
    bounds: tuple[int | float, int | float, int | float, int | float],
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

    All instances of this class share a single event loop running in a background thread.
    """

    # Class-level shared event loop and thread
    _shared_loop: asyncio.AbstractEventLoop | None = None
    _shared_thread: threading.Thread | None = None
    _init_lock = threading.Lock()

    @classmethod
    def _get_loop(cls) -> asyncio.AbstractEventLoop:
        """Get or create the shared event loop.

        Returns:
            The shared event loop.
        """
        if cls._shared_loop is None:
            with cls._init_lock:
                if cls._shared_loop is None:
                    cls._shared_loop = asyncio.new_event_loop()
                    cls._shared_thread = threading.Thread(
                        target=cls._run_shared_loop,
                        daemon=True,
                        name="AsyncZarrLoop",
                    )
                    cls._shared_thread.start()
        return cls._shared_loop

    @classmethod
    def _run_shared_loop(cls) -> None:
        """Run the shared event loop in a background thread."""
        assert cls._shared_loop is not None
        asyncio.set_event_loop(cls._shared_loop)
        cls._shared_loop.run_forever()

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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Numcodecs codecs are not in the Zarr version 3 specification",
            )
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

        # Check if the variable uses NaN as fill value for the retry workaround
        self.array = self.ds[self.variable_name]

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

        # Cache tracking
        self.current_start_index = -1
        self.current_end_index = -1
        self.current_data: npt.NDArray[Any] | None = None
        self.preloaded_data_future: asyncio.Task | None = None

        # Async event loop setup
        if self.asynchronous:
            self.loop: asyncio.AbstractEventLoop | None = self._get_loop()
            assert self.loop is not None

            # Initialize lock in the shared loop
            async def _init_lock() -> asyncio.Lock:
                return asyncio.Lock()

            self.async_lock = asyncio.run_coroutine_threadsafe(
                _init_lock(), self.loop
            ).result()
        else:
            self.loop = None
            self.async_lock = None

    def load(self, start_index: int, end_index: int) -> np.ndarray:
        """Safe synchronous load (only used if asynchronous=False).

        Returns:
            The requested data slice.
        """
        array = self.ds[self.variable_name]
        assert isinstance(array, zarr.Array)
        data = array[start_index:end_index]
        assert isinstance(data, np.ndarray)
        return data

    async def load_await(self, start_index: int, end_index: int) -> npt.NDArray[Any]:
        """Load data asynchronously via reusable async group.

        Uses a workaround for Zarr async loading issues: if the first read returns
        all NaN values (and the array uses NaN as fill value), retry the read.

        Returns:
            The requested data slice (not a copy - caller must copy if needed).
        """
        # Select the variable array from the pre-opened async group.
        arr: zarr.AsyncArray[Any] = self.array.async_array
        data = await arr.getitem(
            (slice(start_index, end_index), slice(None), slice(None))
        )
        assert isinstance(data, np.ndarray)
        return data

    async def preload_next(
        self, start_index: int, end_index: int, n: int
    ) -> npt.NDArray[Any] | None:
        """Preload next timestep asynchronously.

        Returns:
            The preloaded data, or None if out of bounds.
        """
        if end_index + n <= self.time_size:
            return await self.load_await(start_index + n, end_index + n)
        return None

    async def read_timestep_async(
        self, start_index: int, end_index: int, n: int
    ) -> npt.NDArray[Any]:
        """Core async read with safe preload caching.

        This method ensures that only one read operation (including preloading)
        occurs at a time. It fetches the requested data, updates the cache,
        and then triggers a background task to preload the next timestep.

        Args:
            start_index: The starting index of the time slice to read.
            end_index: The ending index of the time slice to read.
            n: The number of timesteps in a single read operation, used for preloading.

        Returns:
            The requested data slice as a NumPy array.

        Raises:
            ValueError: If the requested index is out of bounds.
            IOError: If the async read returns incomplete data.
        """
        if start_index < 0 or end_index > self.time_size:
            raise ValueError(f"Index out of bounds ({start_index}:{end_index})")

        assert self.async_lock is not None
        async with self.async_lock:
            data: npt.NDArray[Any] | None = None

            # Cache hit
            if (
                self.current_data is not None
                and start_index == self.current_start_index
                and end_index == self.current_end_index
            ):
                data = self.current_data

            # Check Preload
            elif (
                self.preloaded_data_future is not None
                and self.current_start_index + n == start_index
            ):
                try:
                    data = await self.preloaded_data_future
                except Exception:
                    pass

            # Load if needed
            if data is None:
                if self.preloaded_data_future and not self.preloaded_data_future.done():
                    self.preloaded_data_future.cancel()
                data = await self.load_await(start_index, end_index)

            # Consistency check
            if data.shape[0] != (end_index - start_index):
                raise IOError(
                    "Async read returned incomplete data; possible disk contention"
                )

            # Update Cache
            self.current_start_index = start_index
            self.current_end_index = end_index
            self.current_data = data

            # Schedule next preload
            assert self.loop is not None
            self.preloaded_data_future = self.loop.create_task(
                self.preload_next(start_index, end_index, n)
            )

            return data

    def get_index(self, date: datetime.datetime, n: int) -> int:
        """Get the time index for a given datetime.

        Optimized for sequential access by checking the current and next indices
        before performing a full search.

        Args:
            date: The datetime to find the index for.
            n: The step size for the 'next' index check.

        Returns:
            The integer index for the given date.

        Raises:
            ValueError: If the date is not found in the time index.
        """
        numpy_date = np.datetime64(date, "ns")
        if (
            self.current_start_index != -1
            and self.datetime_index[self.current_start_index] == numpy_date
        ):
            return self.current_start_index
        elif (
            self.current_start_index != -1
            and self.current_start_index + n < self.time_size
            and self.datetime_index[self.current_start_index + n] == numpy_date
        ):
            return self.current_start_index + n
        else:
            indices = np.where(self.datetime_index == numpy_date)[0]
            if indices.size == 0:
                raise ValueError(f"Date {date} not found in {self.filepath}")
            return indices[0]

    def read_timestep(self, date: datetime.datetime, n: int = 1) -> npt.NDArray[Any]:
        """Public synchronous entrypoint; blocks until async result ready.

        Returns:
            The requested data slice.

        Raises:
            ValueError: If the requested time range exceeds available data.
        """
        start_index = self.get_index(date, n)
        end_index = start_index + n
        if end_index > self.time_size:
            raise ValueError(
                f"Requested {n} timesteps from {date} exceeds available range"
            )

        if self.asynchronous:
            coro = self.read_timestep_async(start_index, end_index, n)
            assert isinstance(self.loop, asyncio.AbstractEventLoop)
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            return future.result()
        else:
            return self.load(start_index, end_index)

    def close(self) -> None:
        """Clean up this instance's async resources."""
        if not self.asynchronous:
            return

        async def cleanup() -> None:
            """Cancel this instance's pending tasks and close async group."""
            if self.preloaded_data_future and not self.preloaded_data_future.done():
                self.preloaded_data_future.cancel()
                try:
                    await self.preloaded_data_future
                except asyncio.CancelledError:
                    pass

        if self.loop and self.loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(cleanup(), self.loop).result(timeout=5)
            except Exception:
                pass

    @property
    def x(self) -> np.ndarray:
        """The x-coordinates of the grid."""
        x_array = self.ds["x"]
        assert isinstance(x_array, zarr.Array)
        x = x_array[:]
        assert isinstance(x, np.ndarray)
        return x

    @property
    def y(self) -> npt.NDArray[Any]:
        """The y-coordinates of the grid."""
        y_array = self.ds["y"]
        assert isinstance(y_array, zarr.Array)
        y = y_array[:]
        assert isinstance(y, np.ndarray)
        return y


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

    def __enter__(self) -> "WorkingDirectory":
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

    Returns:
        True if the file was successfully downloaded, False otherwise.

    Raises:
        RuntimeError: If the download fails after all retries.
    """
    if file_path.exists() and not overwrite:
        return True

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
                    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
                    for data in response.iter_content(chunk_size=chunk_size):
                        temp_file.write(data)
                        progress_bar.update(len(data))
                    progress_bar.close()
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
