import asyncio
import os
import shutil
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import pyproj
import rasterio.crs
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from pyproj import CRS

all_async_readers: list = []


def load_table(fp):
    return pd.read_parquet(fp, engine="pyarrow")


def load_array(fp: Path) -> np.ndarray:
    if fp.suffix == ".npz":
        return np.load(fp)["data"]
    elif fp.suffix == ".zarr":
        return zarr.load(fp)
    else:
        raise ValueError(f"Unsupported file format: {fp.suffix}")


def calculate_scaling(min_value, max_value, precision, offset=0):
    """
    This function calculates the scaling factor and output dtype for
    a fixed scale and offset codec. The expected minimum and maximum
    values along with the precision are used to determine the number
    of bits required to represent the data. The scaling factor is then
    calculated to scale the original data to the required integer
    range. The output dtype is determined based on the number of bits
    required.

    Note that for very high precision in relation to the min and max values,
    there may be some issues due to rounding and the given factors may
    become slighly imprecise.

    Parameters
    ----------
    min_value : float
        The minimum expected value of the original data. Outside this range
        the data may start to behave unexpectedly.
    max_value : float
        The maximum expected value of the original data. Outside this range
        the data may start to behave unexpectedly.
    precision : float
        The precision of the data, i.e. the maximum difference between the
        original and decoded data.
    offset : float, optional
        The offset to apply to the original data before scaling.

    Returns
    -------
    scaling_factor : float
        The scaling factor to apply to the original data.
    out_dtype : str
        The output dtype to use for the fixed scale and offset codec.
    """

    assert min_value < max_value, "min_value must be less than max_value"
    assert precision > 0, "precision must be greater than 0"

    min_with_offset = min_value + offset
    max_with_offset = max_value + offset

    max_abs_value = max(abs(min_with_offset), abs(max_with_offset))

    steps_required = int(max_abs_value / precision / 2) + 1

    bits_required = steps_required.bit_length()

    steps_available = 2**bits_required

    if min_with_offset < 0:
        bits_required += 1  # need to account for the sign bit
        out_dtype_prefix = ""
    else:
        out_dtype_prefix = "u"

    scaling_factor = steps_available / max_abs_value

    if bits_required <= 8:
        out_dtype = out_dtype_prefix + "int8"
    elif bits_required <= 16:
        out_dtype = out_dtype_prefix + "int16"
    elif bits_required <= 32:
        out_dtype = out_dtype_prefix + "int32"
    elif bits_required <= 64:
        out_dtype = out_dtype_prefix + "int64"
    else:
        raise ValueError("Too many bits required for precision and range")

    return scaling_factor, out_dtype


def open_zarr(zarr_folder):
    # it is rather odd, but in some cases using mask_and_scale=False is necessary
    # or dtypes start changing, seemingly randomly
    # consolidated metadata is off-spec for zarr, therefore we set it to False
    da = xr.open_dataset(
        zarr_folder, engine="zarr", chunks={}, consolidated=False, mask_and_scale=False
    )
    assert len(da.data_vars) == 1, "Only one data variable is supported"
    da = da[list(da.data_vars)[0]]

    if da.dtype == bool and "_FillValue" not in da.attrs:
        da.attrs["_FillValue"] = None

    if "_CRS" in da.attrs:
        da.rio.write_crs(pyproj.CRS(da.attrs["_CRS"]["wkt"]), inplace=True)
        del da.attrs["_CRS"]

    return da


def to_wkt(crs_obj):
    if isinstance(crs_obj, int):  # EPSG code
        return CRS.from_epsg(crs_obj).to_wkt()
    elif isinstance(crs_obj, CRS):  # Pyproj CRS
        return crs_obj.to_wkt()
    elif isinstance(crs_obj, rasterio.crs.CRS):  # Rasterio CRS
        return CRS(crs_obj.to_wkt()).to_wkt()
    else:
        raise TypeError("Unsupported CRS type")


def check_attrs(da1, da2):
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


def check_buffer_size(da, chunks_or_shards, max_buffer_size=2147483647):
    buffer_size = (
        np.prod([size for size in chunks_or_shards.values()]) * da.dtype.itemsize
    )
    assert buffer_size <= max_buffer_size, (
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
    compressor=None,
    progress: bool = True,
):
    """
    Save an xarray DataArray to a zarr file.

    Parameters
    ----------
    da : xarray.DataArray
        The xarray DataArray to save.
    path : str
        The path to the zarr file.
    crs : int or pyproj.CRS
        The coordinate reference system to use.
    x_chunksize : int, optional
        The chunk size for the x dimension. Default is 350.
    y_chunksize : int, optional
        The chunk size for the y dimension. Default is 350.
    time_chunksize : int, optional
        The chunk size for the time dimension. Default is 1.
    time_chunks_per_shard : int, optional
        The number of time chunks per shard. Default is 30. Set to None
        to disable sharding.
    byteshuffle : bool, optional
        Whether to use byteshuffle compression. Default is True.
    filters : list, optional
        A list of filters to apply. Default is [].
    compressor : numcodecs, optional
        The compressor to use. Default is None, using the default Blosc compressor.
    progress : bool, optional
        Whether to show a progress bar. Default is True.

    Returns
    -------
    da_disk : xarray.DataArray
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

        da = da.drop_vars([v for v in da.coords if v not in da.dims])

        chunks, shards = {}, None
        if "y" in da.dims and "x" in da.dims:
            chunks.update(
                {
                    "y": min(y_chunksize, da.sizes["y"]),
                    "x": min(x_chunksize, da.sizes["x"]),
                }
            )

        if "time" in da.dims:
            chunks.update({"time": min(time_chunksize, da.sizes["time"])})
            if time_chunks_per_shard is not None:
                shards = chunks.copy()
                shards["time"] = time_chunks_per_shard * time_chunksize

        # currently we are in a conondrum, where gdal does not yet support zarr version 3.
        # support seems recently merged, and we need to wait for the 3.11 release, and
        # subsequent QGIS support for GDAL 3.11. See https://github.com/OSGeo/gdal/pull/11787
        # For anything with a shard, we opt for zarr version 3, for anything without, we use version 2.
        if shards:
            zarr_format = 3
            from zarr.codecs import BloscCodec
            from zarr.codecs.blosc import BloscShuffle

            if compressor is None:
                compressor = BloscCodec(
                    cname="zstd",
                    clevel=9,
                    shuffle=BloscShuffle.shuffle
                    if byteshuffle
                    else BloscShuffle.noshuffle,
                )

            check_buffer_size(da, chunks_or_shards=shards)
        else:
            assert not filters, "Filters are only supported for zarr version 3"
            zarr_format = 2
            from numcodecs import Blosc

            if compressor is None:
                compressor = Blosc(
                    cname="zstd", clevel=9, shuffle=1 if byteshuffle else 0
                )

            check_buffer_size(da, chunks_or_shards=chunks)

        da = da.chunk(shards if shards is not None else chunks)

        da.attrs["_CRS"] = {"wkt": to_wkt(crs)}

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

        encoding = {da.name: array_encoding}
        for coord in da.coords:
            encoding[coord] = {"compressors": (compressor,)}

        arguments = {
            "store": tmp_zarr,
            "mode": "w",
            "encoding": encoding,
            "zarr_format": zarr_format,
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

        folder = path.parent
        folder.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.rmtree(path)
        shutil.move(tmp_zarr, folder)

    da_disk = open_zarr(path)

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
    """
    Get a window for the given x and y coordinates based on the provided bounds and buffer.

    Parameters
    ----------
    x : xr.DataArray
        The x coordinates as an xarray DataArray.
    y : xr.DataArray
        The y coordinates as an xarray DataArray.
    bounds : tuple
        A tuple of four values representing the bounds in the form (min_x, min_y, max_x, max_y).
    buffer : int, optional
        The buffer size to apply to the bounds. Default is 0.
    raise_on_out_of_bounds : bool, optional
        Whether to raise an error if the bounds are out of the x or y coordinate range. Default is True.
    raise_on_buffer_out_of_bounds : bool, optional
        Whether to raise an error if the buffer goes out of the x or y coordinate range. Default is True.

    Returns
    -------
    dict
        A dictionary with slices for the x and y coordinates, e.g. {"x": slice(start, stop), "y": slice(start, stop)}.
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


class AsyncForcingReader:
    def __init__(self, filepath, variable_name):
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

    def load(self, index):
        data = self.var[index, :]
        return data

    async def load_await(self, index):
        return await self.loop.run_in_executor(self.executor, lambda: self.load(index))

    async def preload_next(self, index):
        # Preload the next timestep asynchronously
        if index + 1 < self.time_size:
            return await self.load_await(index + 1)
        return None

    async def read_timestep_async(self, index):
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

    def get_index(self, date):
        # convert datetime object to dtype of time coordinate. There is a very high probability
        # that the dataset is the same as the previous one or the next one in line,
        # so we can just check the current index and the next one. Only if those do not match
        # we have to search for the correct index.
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

    def read_timestep(self, date, asynchronous=False):
        if asynchronous:
            index = self.get_index(date)
            fn = self.read_timestep_async(index)
            data = self.loop.run_until_complete(fn)
            return data
        else:
            index = self.get_index(date)
            return self.load(index)

    def close(self):
        # cancel the preloading of the next timestep
        if self.preloaded_data_future is not None:
            self.preloaded_data_future.cancel()
        # close the executor
        self.executor.shutdown(wait=False)

        self.loop.call_soon_threadsafe(self.loop.stop)


class WorkingDirectory:
    """
    A context manager for temporarily changing the current working directory.

    Usage:
        with WorkingDirectory('/path/to/new/directory'):
            # Code executed here will have the new directory as the CWD
    """

    def __init__(self, new_path):
        """
        Initializes the context manager with the path to change to.

        Args:
            new_path (str): The path to the directory to change into.
        """
        self._new_path = new_path
        self._original_path = None  # To store the original path

    def __enter__(self):
        # Store the current working directory
        self._original_path = os.getcwd()

        # Change to the new directory
        os.chdir(self._new_path)

        # Return self (optional, but common)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Change back to the original directory
        os.chdir(self._original_path)
