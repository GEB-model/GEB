import asyncio
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cftime
import numcodecs
import numpy as np
import pandas as pd
import pyproj
import rasterio.crs
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from pyproj import CRS
from zarr.codecs import BloscCodec, BloscShuffle

all_async_readers = []


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
    da,
    path,
    crs,
    x_chunksize=350,
    y_chunksize=350,
    time_chunksize=1,
    time_chunks_per_shard=30,
    byteshuffle=True,
    filters=[],
    progress=True,
):
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

    path = Path(path)

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
            zarr_version = 3
            compressor = BloscCodec(
                cname="zstd",
                clevel=9,
                shuffle=BloscShuffle.shuffle if byteshuffle else BloscShuffle.noshuffle,
            )

            check_buffer_size(da, chunks_or_shards=shards)
        else:
            assert not filters, "Filters are only supported for zarr version 3"
            zarr_version = 2
            compressor = numcodecs.Blosc(cname="zstd", clevel=9, shuffle=byteshuffle)

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
            encoding[coord] = {"compressors": None}

        arguments = {
            "store": tmp_zarr,
            "mode": "w",
            "encoding": encoding,
            "zarr_version": zarr_version,
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


class AsyncForcingReader:
    def __init__(self, filepath, variable_name):
        self.variable_name = variable_name
        self.filepath = filepath

        store = zarr.storage.LocalStore(self.filepath, read_only=True)
        self.ds = zarr.open_group(store, mode="r")
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
        numpy_date = np.datetime64(date, "ns")
        if self.datetime_index[self.current_index] == numpy_date:
            return self.current_index
        elif self.datetime_index[self.current_index + 1] == numpy_date:
            return self.current_index + 1
        else:
            indices = self.datetime_index == numpy_date
            assert np.count_nonzero(indices) == 1, "Date not found in the dataset."
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
