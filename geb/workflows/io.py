"""I/O related functions and classes for the GEB project."""

import asyncio
import datetime
import os
import shutil
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import Any

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
from dask.diagnostics import ProgressBar
from pyproj import CRS
from rasterio.transform import Affine
from tqdm import tqdm
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec
from zarr.codecs.blosc import BloscShuffle

all_async_readers: list = []


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
        return zarr.load(fp)
    else:
        raise ValueError(f"Unsupported file format: {fp.suffix}")


def load_grid(
    filepath: str | Path, layer: int = 1, return_transform_and_crs: bool = False
) -> np.ndarray | tuple[np.ndarray, Affine, str]:
    """Load a raster grid from a .tif or .zarr file.

    Args:
        filepath: The path to the .tif or .zarr file.
        layer: The layer to load from the .tif file. Default is 1.
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
        store: zarr.storage._local.LocalStore = zarr.storage.LocalStore(
            filepath, read_only=True
        )
        group: zarr.core.group.Group = zarr.open_group(store, mode="r")
        data: np.ndarray = group[filepath.stem][:]
        data: np.ndarray = data.astype(np.float32) if data.dtype == np.float64 else data
        if return_transform_and_crs:
            x: np.ndarray = group["x"][:]
            y: np.ndarray = group["y"][:]
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
    path: str | Path,
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

        array_encoding = {
            "compressors": (compressor,),
            "chunks": chunks,
            "filters": filters,
        }

        if shards is not None:
            shards = tuple(
                (shards[dim] if dim in shards else getattr(da, dim).size)
                for dim in da.dims
            )
            array_encoding["shards"] = shards

        encoding: dict[str, dict[str, Any]] = {da.name: array_encoding}
        for coord in da.coords:
            encoding[coord] = {"compressors": (compressor,)}

        arguments: dict[str, Any] = {
            "store": tmp_zarr,
            "mode": "w",
            "encoding": encoding,
            "zarr_format": 3,
            "consolidated": False,  # consolidated metadata is off-spec for zarr, therefore we set it to False
        }

        if progress:
            # start writing after 10 seconds, and update every 0.1 seconds
            with ProgressBar(
                minimum=0.1,
                dt=float(os.environ.get("GEB_OVERRIDE_PROGRESSBAR_DT", 0.1)),
            ):
                store = da.to_zarr(**arguments)
        else:
            store = da.to_zarr(**arguments)

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
            bounds[0] = x[0]
    if bounds[2] > x[-1]:
        if raise_on_out_of_bounds:
            raise ValueError("xmax must be less than x[-1]")
        else:
            bounds[2] = x[-1]
    if bounds[1] < y[-1]:
        if raise_on_out_of_bounds:
            raise ValueError("ymin must be greater than y[-1]")
        else:
            bounds[1] = y[-1]
    if bounds[3] > y[0]:
        if raise_on_out_of_bounds:
            raise ValueError("ymax must be less than y[0]")
        else:
            bounds[3] = y[0]

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
    """Asynchronous reader for a forcing variable stored in a zarr file.

    This class allows for asynchronous reading of a forcing variable from a zarr file.
    It supports preloading the next timestep to improve performance when reading
    multiple timesteps sequentially.
    """

    def __init__(self, filepath: Path, variable_name: str) -> None:
        """Initializes the AsyncGriddedForcingReader.

        Args:
            filepath: The path to the zarr file.
            variable_name: The name of the variable to read from the zarr file.
        """
        self.variable_name = variable_name
        self.filepath = filepath

        store = zarr.storage.LocalStore(self.filepath, read_only=True)
        self.ds = zarr.open_group(store, mode="r")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Numcodecs codecs are not in the Zarr version 3 specification and may not be supported by other zarr implementations.",
            )
            self.var = self.ds[variable_name]

        self.datetime_index = cftime.num2date(
            self.ds["time"][:],
            units=self.ds["time"].attrs.get("units"),
            calendar=self.ds["time"].attrs.get("calendar"),
        )
        self.datetime_index = pd.DatetimeIndex(
            pd.to_datetime([obj.isoformat() for obj in self.datetime_index])
        ).to_numpy()
        self.time_size = self.datetime_index.size

        all_async_readers.append(self)
        self.preloaded_data_future = None
        self.current_index = -1  # Initialize to -1 to indicate no data loaded yet

        self.loop = asyncio.new_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def load(self, start_index: int, end_index: int) -> npt.NDArray[Any]:
        """Load the data for the given index from the zarr file.

        Args:
            start_index: The start index of the timestep to load in the zarr file, along the time dimension.
            end_index: The final index of the timestep to load in the zarr file, along the time dimension.

        Returns:
            The data for the given index from the zarr file.
        """
        data: npt.NDArray[Any] = self.var[start_index:end_index, :]
        return data

    async def load_await(self, index: int) -> npt.NDArray[Any]:
        """Load the data for the given index from the zarr file asynchronously.

        Args:
            index: The index of the timestep to read in the zarr file, along the time dimension.

        Returns:
            An awaitable that resolves to the data for the given index from the zarr file.
        """
        return await self.loop.run_in_executor(self.executor, lambda: self.load(index))

    async def preload_next(self, index: int) -> None | npt.NDArray[Any]:
        """Preload the next timestep asynchronously.

        Args:
            index: The index of the timestep to read in the zarr file, along the time dimension.

        Returns:
            An awaitable that resolves to the data for the next index, or None if there is no next index.
        """
        # Preload the next timestep asynchronously
        if index + 1 < self.time_size:
            return await self.load_await(index + 1)
        return None

    async def read_timestep_async(self, index: int) -> npt.NDArray[Any]:
        """Read the data for the given index from the zarr file asynchronously.

        Also preloads the next timestep asynchronously so that it is ready when needed.

        Args:
            index: The index of the timestep to read in the zarr file, along the time dimension.

        Returns:
            The data for the given index from the zarr file.

        """
        assert index < self.time_size, "Index out of bounds."
        assert index >= 0, "Index out of bounds."
        # Check if the requested data is already preloaded, if so, just return that data
        if index == self.current_index:
            return self.current_data
        # Check if the data for the next timestep is preloaded, if so, await for it to complete
        if self.preloaded_data_future is not None and self.current_index + 1 == index:
            data = await self.preloaded_data_future
        # Load the requested data if not preloaded
        else:
            data = await self.load_await(index)

        # Initiate preloading the next timestep, do not await here, this returns a future
        self.preloaded_data_future = asyncio.create_task(self.preload_next(index))
        self.current_index = index
        self.current_data = data
        return data

    def get_index(self, date: datetime.datetime) -> int:
        """Convert datetime object to dtype of time coordinate.

        There is a very high probability that the dataset is the same as
        the previous one or the next one in line, so we can just check the current
        index and the next one. Only if those do not match we have to search for the correct index.

        Args:
            date: The date to convert.

        Returns:
            The index of the given date in the zarr file.

        """
        numpy_date: np.datetime64 = np.datetime64(date, "ns")
        if self.datetime_index[self.current_index] == numpy_date:
            return self.current_index
        elif self.datetime_index[self.current_index + 1] == numpy_date:
            return self.current_index + 1
        else:
            indices = self.datetime_index == numpy_date
            assert np.count_nonzero(indices) == 1, (
                f"Date not found in the dataset. The first date available in {self.variable_name} ({self.filepath}) is {self.datetime_index[0]} and the last date is {self.datetime_index[-1]}, while requested date is {date}"
            )
            return indices.argmax()

    def read_timestep(
        self, date: datetime.datetime, n: int = 1, asynchronous: bool = False
    ) -> npt.NDArray[Any]:
        """Read the data for the given date from the zarr file.

        Args:
            date: The date of the timestep to read.
            n: The number of timesteps to read. Defaults to 1.
            asynchronous: If True, the data is read asynchronously. Defaults to False.

        Note:
            The asynchronous mode is currently broken.

        Returns:
            The data for the given date from the zarr file.
        """
        if asynchronous:
            index = self.get_index(date)
            fn = self.read_timestep_async(index)
            data = self.loop.run_until_complete(fn)
            return data
        else:
            start_index = self.get_index(date)
            end_index = start_index + n
            data = self.load(start_index, end_index)
            return data

    def close(self) -> None:
        """Close the reader and clean up resources."""
        # cancel the preloading of the next timestep
        if self.preloaded_data_future is not None:
            self.preloaded_data_future.cancel()
        # close the executor
        self.executor.shutdown(wait=False)

        self.loop.call_soon_threadsafe(self.loop.stop)

    @property
    def x(self) -> npt.NDArray[Any]:
        """Get the x coordinates of the variable.

        Returns:
            The x coordinates of the variable.
        """
        return self.ds["x"][:]

    @property
    def y(self) -> npt.NDArray[Any]:
        """Get the y coordinates of the variable.

        Returns:
            The y coordinates of the variable.
        """
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
    """Fetches data from a URL and saves it to a temporary file, with a retry mechanism.

    Moves the file to the destination if the download is complete.
    Removes the temporary file if the download is interrupted.

    Args:
        url: path or URL to the file to download.
        file_path: path to the file to save the downloaded data.
        overwrite: whether to overwrite the file if it already exists. Default is False.
        max_retries: maximum number of retries in case of failure. Default is 3.
        delay: delay between retries in seconds. Default is 5 seconds.
        chunk_size: size of the chunks to read from the response.
        session: Optional requests.Session object to use for the download. If None, a new session will be created.
        params: Optional dictionary of query parameters to include in the request.
        timeout: Timeout for the request in seconds. Default is 30 seconds.

    Returns:
        Returns True if the file was downloaded successfully and saved to the specified path.
        Raises an exception if all attempts to download the file fail.

    Raises:
        RuntimeError: If all attempts to download the file fail.

    """
    if not session:
        session = requests.Session()

    if not overwrite and file_path.exists():
        return True

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
