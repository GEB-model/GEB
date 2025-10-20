"""I/O related functions and classes for the GEB project."""

import asyncio
import datetime
import os
import shutil
import tempfile
import threading
import time
import warnings
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any, overload

import cftime
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import rasterio
import requests
import s3fs
import xarray as xr
import zarr
import zarr.storage
from dask.diagnostics import ProgressBar
from pyproj import CRS
from rasterio.transform import Affine
from tqdm import tqdm
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec
from zarr.codecs.blosc import BloscShuffle


def load_table(fp: Path | str) -> pd.DataFrame:
    """Load a parquet file as a pandas DataFrame.

    Args:
        fp: The path to the parquet file.

    Returns:
        The pandas DataFrame.
    """
    return pd.read_parquet(fp, engine="pyarrow")


def load_array(fp: Path) -> np.ndarray:
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
def load_grid(
    filepath: Path, layer: int | None = 1, return_transform_and_crs: bool = False
) -> np.ndarray: ...


@overload
def load_grid(
    filepath: Path, layer: int | None = 1, return_transform_and_crs: bool = True
) -> tuple[np.ndarray, Affine, str]: ...


def load_grid(
    filepath: Path, layer: int | None = 1, return_transform_and_crs: bool = False
) -> np.ndarray | tuple[np.ndarray, Affine, str]:
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
            data: np.ndarray = src.read(layer)
            data: np.ndarray = (
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
        array: zarr.Array | zarr.Group = group[filepath.stem]
        assert isinstance(array, zarr.Array)
        data: np.ndarray = array[:]
        data: np.ndarray = data.astype(np.float32) if data.dtype == np.float64 else data
        if return_transform_and_crs:
            x: zarr.Array | zarr.Group = group["x"]
            assert isinstance(x, zarr.Array)
            x: np.ndarray = x[:]
            y: zarr.Array | zarr.Group = group["y"]
            assert isinstance(y, zarr.Array)
            y: np.ndarray = y[:]
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
            wkt: str = group[filepath.stem].attrs["_CRS"]["wkt"]
            return data, transform, wkt
        else:
            return data
    else:
        raise ValueError("File format not supported.")


def load_geom(filepath: str | Path) -> gpd.GeoDataFrame:
    """Load a geometry for the GEB model from disk.

    Args:
        filepath: Path to the geometry file.

    Returns:
        A GeoDataFrame containing the geometries.

    """
    return gpd.read_parquet(filepath)


def calculate_scaling(
    da: xr.DataArray,
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


def open_zarr(zarr_folder: Path | str) -> xr.DataArray:
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
    if len(ds.data_vars) > 1:
        raise ValueError("Only one data variable is supported")

    da: xr.DataArray = ds[list(ds.data_vars)[0]]

    if da.dtype == bool and "_FillValue" not in da.attrs:
        da.attrs["_FillValue"] = None

    if "_CRS" in da.attrs:
        da.rio.write_crs(pyproj.CRS(da.attrs["_CRS"]["wkt"]), inplace=True)
        del da.attrs["_CRS"]

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
    elif isinstance(crs_obj, rasterio.crs.CRS):  # Rasterio CRS
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


def to_zarr(
    da: xr.DataArray,
    path: str | Path | zarr.storage.LocalStore,
    crs: int | pyproj.CRS,
    x_chunksize: int = 350,
    y_chunksize: int = 350,
    time_chunksize: int = 1,
    time_chunks_per_shard: int | None = 30,
    byteshuffle: bool = True,
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
        byteshuffle: Whether to use byteshuffle compression. Default is True.
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
            compressor: BloscCodec = BloscCodec(
                cname="zstd",
                clevel=9,
                shuffle=BloscShuffle.shuffle if byteshuffle else BloscShuffle.noshuffle,
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
        encoding: dict[str, dict[str, Any]] = {da.name: array_encoding}
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

    da_disk: xr.DataArray = open_zarr(path)

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

    This reader uses a single reusable AsyncGroup for all async reads, with a
    workaround for occasional Zarr async loading issues that return NaN on first read.

    All instances of this class share a single event loop running in a background thread,
    which is more efficient than creating separate loops for each reader.

    TODO: Perhaps this is a bug in zarr, or in our implementation, but in any case this
    class retries reads that return all NaN values (assuming the variable uses NaN
    as fill value) up to a maximum number of retries. Should be investigated in the future
    but for now this workaround allows reliable async reading.
    """

    # Class-level shared event loop and thread
    _shared_loop: asyncio.AbstractEventLoop | None = None
    _shared_thread: threading.Thread | None = None
    _shared_loop_lock = threading.Lock()
    _loop_ready = threading.Event()
    _loop_refcount = 0

    @classmethod
    def _ensure_shared_loop(cls) -> None:
        """Ensure the shared event loop is running and increment reference count."""
        with cls._shared_loop_lock:
            if cls._shared_loop is None or not cls._shared_loop.is_running():
                cls._loop_ready.clear()
                cls._shared_loop = asyncio.new_event_loop()
                cls._shared_thread = threading.Thread(
                    target=cls._run_shared_loop, daemon=True, name="AsyncZarrLoop"
                )
                cls._shared_thread.start()
                cls._loop_ready.wait()
            cls._loop_refcount += 1

    @classmethod
    def _release_shared_loop(cls) -> None:
        """Decrement reference count and stop loop if no longer needed."""
        with cls._shared_loop_lock:
            cls._loop_refcount -= 1
            if cls._loop_refcount <= 0 and cls._shared_loop is not None:
                cls._shared_loop.call_soon_threadsafe(cls._shared_loop.stop)
                if cls._shared_thread is not None:
                    cls._shared_thread.join(timeout=5)
                cls._shared_loop = None
                cls._shared_thread = None
                cls._loop_refcount = 0

    @classmethod
    def _run_shared_loop(cls) -> None:
        """Run the shared event loop in a background thread."""
        asyncio.set_event_loop(cls._shared_loop)
        cls._loop_ready.set()
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
            time_arr = self.ds["time"][:]

        self.datetime_index = cftime.num2date(
            time_arr,
            units=self.ds["time"].attrs.get("units"),
            calendar=self.ds["time"].attrs.get("calendar"),
        )
        self.datetime_index = pd.to_datetime(
            [obj.isoformat() for obj in self.datetime_index]
        ).to_numpy()
        self.time_size = self.datetime_index.size

        # Check if the variable uses NaN as fill value for the retry workaround
        variable_array = self.ds[self.variable_name]
        fill_value = variable_array.fill_value
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
            # Ensure the shared event loop is running
            self._ensure_shared_loop()
            self.loop = self._shared_loop

            # Initialize this instance's async components
            self.async_ready = threading.Event()
            future = asyncio.run_coroutine_threadsafe(
                self._initialize_async_components(), self.loop
            )
            future.result()  # Wait for initialization to complete
            self.async_ready.set()
        else:
            self.loop = None
            self.async_lock = None
            self.async_group = None

    async def _initialize_async_components(self) -> None:
        """Initialize async lock and reusable async Zarr group for this instance."""
        self.async_lock = asyncio.Lock()
        # Open the main group store once and reuse it for all reads.
        self.async_group = await zarr.AsyncGroup.open(self.store)

    def load(self, start_index: int, end_index: int) -> npt.NDArray[Any]:
        """Safe synchronous load (only used if asynchronous=False).

        Returns:
            The requested data slice.
        """
        array = self.ds[self.variable_name]
        data = array[start_index:end_index]
        return data

    async def load_await(self, start_index: int, end_index: int) -> npt.NDArray[Any]:
        """Load data asynchronously via reusable async group.

        Uses a workaround for Zarr async loading issues: if the first read returns
        all NaN values (and the array uses NaN as fill value), retry the read.

        Returns:
            The requested data slice (not a copy - caller must copy if needed).

        Raises:
            RuntimeError: If data loading fails after maximum retries.
        """
        # Select the variable array from the pre-opened async group.
        arr = await self.async_group.getitem(self.variable_name)
        max_retries = 100
        retries = 0
        while retries < max_retries:
            data = await arr.get_orthogonal_selection(
                (slice(start_index, end_index), slice(None), slice(None))
            )

            # Only apply the NaN workaround if the array actually uses NaN as fill value
            if np.all(np.isnan(data)):
                retries += 1
                print(
                    f"Warning: Async read returned all NaN values for {self.variable_name}, retrying {retries}/{max_retries}..."
                )
                await asyncio.sleep(delay=0.1)  # brief pause before retrying
            else:
                return data

        # If still all NaN after retries, raise an error
        raise RuntimeError(
            f"Failed to load data for {self.variable_name} after {max_retries} retries due to all NaN values."
        )

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

        async with self.async_lock:
            # --- Step 1: Load current data ---
            data: npt.NDArray[Any]

            # Cache hit
            if (
                self.current_data is not None
                and start_index == self.current_start_index
                and end_index == self.current_end_index
            ):
                data = self.current_data

            # Preload hit
            elif (
                self.preloaded_data_future is not None
                and self.current_start_index + n == start_index
            ):
                try:
                    preloaded = await self.preloaded_data_future
                    data = (
                        preloaded
                        if preloaded is not None
                        else await self.load_await(start_index, end_index)
                    )
                except asyncio.CancelledError:
                    data = await self.load_await(start_index, end_index)
                except Exception:
                    data = await self.load_await(start_index, end_index)

            # Cache miss
            else:
                if self.preloaded_data_future and not self.preloaded_data_future.done():
                    self.preloaded_data_future.cancel()
                data = await self.load_await(start_index, end_index)

            # --- Step 2: Consistency check ---
            if data.shape[0] != (end_index - start_index):
                raise IOError(
                    "Async read returned incomplete data; possible disk contention"
                )

            # --- Step 3: Update cache and return data ---
            # Copy the data to protect the cache from external mutations.
            self.current_start_index = start_index
            self.current_end_index = end_index
            self.current_data = data.copy()

            # --- Step 4: Start preloading next timestep in the background ---
            # This task will run after the current data is returned and will
            # acquire the lock for its own read operation.
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
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            return future.result()
        else:
            return self.load(start_index, end_index)

    def close(self) -> None:
        """Clean up this instance's async resources and release the shared loop reference."""
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

            # Close the async group if it exists
            if hasattr(self, "async_group") and self.async_group is not None:
                await self.async_group.aclose()

        if self.loop and self.loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(cleanup(), self.loop).result(timeout=5)
            except Exception:
                pass

        # Release the shared event loop reference
        self._release_shared_loop()

    @property
    def x(self) -> npt.NDArray[Any]:
        """The x-coordinates of the grid."""
        return self.ds["x"][:]

    @property
    def y(self) -> npt.NDArray[Any]:
        """The y-coordinates of the grid."""
        return self.ds["y"][:]


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
        self._original_path = None  # To store the original path

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
    delay: float | int = 5,
    chunk_size: int = 16384,
    session: requests.Session | None = None,
    params: None | dict[str, Any] = None,
    timeout: float | int = 30,
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
        delay: The delay in seconds between retries.
        chunk_size: The chunk size for streaming downloads.
        session: An optional requests.Session object to use for HTTP requests.
        params: Optional dictionary of query parameters for HTTP requests.
        timeout: The timeout in seconds for HTTP requests.

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

        while attempts < max_retries:
            try:
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
                print(
                    f"S3 download failed: {e}. Attempt {attempts + 1} of {max_retries}"
                )

                # Remove the temporary file if it exists
                if temp_file is not None and os.path.exists(temp_file.name):
                    os.remove(temp_file.name)

                # Increment the attempt counter and wait before retrying
                attempts += 1
                if attempts < max_retries:
                    time.sleep(delay)

        # If all attempts fail, raise an exception
        raise RuntimeError(
            f"Failed to download '{url}' from S3 to '{file_path}' after {max_retries} attempts."
        )

    elif url.startswith("http://") or url.startswith("https://"):
        attempts = 0
        temp_file = None

        while attempts < max_retries:
            try:
                print(f"Downloading {url} to {file_path}")
                # Attempt to make the request
                response = session.get(url, stream=True, params=params, timeout=timeout)
                response.raise_for_status()  # Raises HTTPError for bad status codes

                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False)

                # Write to the temporary file
                total_size = int(response.headers.get("content-length", 0))
                progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
                for data in response.iter_content(chunk_size=chunk_size):
                    temp_file.write(data)
                    progress_bar.update(len(data))
                progress_bar.close()

                # Close the temporary file
                temp_file.close()

                # Move the temporary file to the destination
                shutil.move(temp_file.name, file_path)

                return True  # Exit the function after successful write

            except requests.RequestException as e:
                # Log the error
                print(f"Request failed: {e}. Attempt {attempts + 1} of {max_retries}")

                # Remove the temporary file if it exists
                if temp_file is not None and os.path.exists(temp_file.name):
                    os.remove(temp_file.name)

                # Increment the attempt counter and wait before retrying
                attempts += 1
                time.sleep(delay)

        # If all attempts fail, raise an exception
        raise RuntimeError(
            f"Failed to download '{url}' to '{file_path}' after {max_retries} attempts. "
            "Please check the URL, network connectivity, and destination permissions."
        )
    return False
