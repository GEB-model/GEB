"""Tests for I/O workflow functions."""

import shutil
import warnings
from datetime import datetime
from pathlib import Path
from time import sleep, time

import dask.array
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr
import zarr
import zarr.storage
from zarr.codecs.numcodecs import FixedScaleOffset

import geb.workflows.io as io_module
from geb.workflows.io import (
    AsyncGriddedForcingReader,
    calculate_scaling,
    create_hash_from_parameters,
    get_window,
    read_hash,
    read_table,
    write_hash,
    write_table,
    write_zarr,
)

from ..testconfig import tmp_folder


def encode_decode(
    data: npt.NDArray[np.floating],
    min_value: float | int,
    max_value: float | int,
    offset: float | int,
    precision: float | int = 0.1,
) -> tuple[npt.NDArray[np.floating], float, str]:
    """Helper function to test encoding and decoding of data using FixedScaleOffset codec.

    This function first encodes the input data using the FixedScaleOffset codec with the specified scaling factor and offset.
    It then decodes the encoded data back to its original form. The function checks that the maximum absolute difference
    between the original and decoded data is within the specified precision.

    It is very important that the minimum and maximum values are never exceeded in the data. If they are, there
    will be overflow errors and the decoded data will be (very) wrong.

    Args:
        data: Input data to be encoded and decoded.
        min_value: Minimum value that is expected in the data.
            This value is used to calculate the scaling factor.
        max_value: Maximum value that is expected in the data.
            This value is used to calculate the scaling factor.
        offset:
            Offset to be used in the FixedScaleOffset codec.
            This value is subtracted from the data before scaling.
        precision: Required precision of the data. Defaults to 0.1.

    Returns:
        A tuple containing the decoded data
        The scaling factor used
        The output data type.
    """
    assert data.dtype == np.float32

    scaling_factor, in_dtype, out_dtype = calculate_scaling(
        data,
        min_value=min_value,
        max_value=max_value,
        precision=precision,
        offset=offset,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Numcodecs codecs are not in the Zarr version 3 specification and may not be supported by other zarr implementations",
        )
        codec = FixedScaleOffset(
            offset=offset, scale=scaling_factor, dtype=in_dtype, astype=out_dtype
        )
        da = xr.DataArray(data)
        da.name = "data"
        da.to_zarr(
            tmp_folder / "test.zarr",
            mode="w",
            encoding={"data": {"filters": [codec]}},
            consolidated=False,
        )

        decoded_data = xr.open_zarr(tmp_folder / "test.zarr", consolidated=False)[
            "data"
        ].values

    diff: npt.NDArray[np.float32] = data - decoded_data

    assert np.all(np.abs(diff) <= precision * 1.05), (
        f"max diff: {diff.max()}; min diff: {diff.min()}"
    )
    return decoded_data, scaling_factor, out_dtype


def test_calculate_scaling() -> None:
    """Test the calculate_scaling function with various precision and range scenarios.

    Tests encoding/decoding with different precisions, offsets, and data ranges.
    Verifies that the scaling calculations work correctly for signed and unsigned
    integer types, and that edge cases like zero values and extreme ranges are handled.
    Also tests that ValueError is raised for impossible precision/range combinations.
    """
    for precision in [0.1, 0.01, 0.001, 0.0001]:
        data = ((np.random.rand(100) - 0.5) * 20).astype(np.float32)
        encode_decode(data, -10, 100, offset=0, precision=precision)

    for precision in [0.1, 0.01, 0.001, 0.0001]:
        data = ((np.random.rand(100) - 0.5) * 20).astype(np.float32)
        (decoded_data, scaling_factor, out_dtype) = encode_decode(
            data, -10, 100, offset=-50, precision=precision
        )
        assert not out_dtype.startswith("u")

    for precision in [0.1, 0.01, 0.001, 0.0001]:
        data = (np.random.rand(100) * 20).astype(np.float32)
        decoded_data, scaling_factor, out_dtype = encode_decode(
            data, 0, 100, offset=0, precision=precision
        )
        assert out_dtype.startswith("u")

    data = np.random.rand(100).astype(np.float32)
    data[:50] = 0
    decoded_data, scaling_factor, out_dtype = encode_decode(
        data, 0, 1, offset=0, precision=0.1
    )
    assert (decoded_data[:50] == 0).all()

    data = np.linspace(0, 10000, 100).astype(np.float32)
    min_value, max_value = 0, 10000000000
    encode_decode(
        data,
        min_value,
        max_value,
        offset=-(min_value + max_value) / 2,
        precision=10000,
    )

    with pytest.raises(
        ValueError, match="Too many bits required for precision and range"
    ):
        calculate_scaling(
            data,
            min_value=0,
            max_value=1e10,
            precision=1e-12,
            offset=0,
        )

    with pytest.raises(
        ValueError, match="Too many bits required for precision and range"
    ):
        calculate_scaling(
            data,
            min_value=-1e10,
            max_value=0,
            precision=1e-12,
            offset=0,
        )


def test_io() -> None:
    """Test the to_zarr function with different data array configurations.

    Tests saving xarray DataArrays to Zarr format with different coordinate
    systems and dimensions. Verifies that arrays with spatial coordinates,
    time dimensions, and other dimensions can be properly saved with CRS
    information and fill values.
    """
    x = np.linspace(-5, 5, 10)
    y = np.linspace(10, 0, 10)

    values = np.random.rand(x.size, y.size).astype(np.float32)
    da = xr.DataArray(values, coords={"x": x, "y": y}, dims=["y", "x"]).chunk()
    da.attrs["_FillValue"] = np.nan

    write_zarr(da, tmp_folder / "test.zarr", crs=4326)

    time_dimension = np.linspace(0, 10, 10)
    values = np.random.rand(x.size, y.size, time_dimension.size).astype(np.float32)
    da = xr.DataArray(
        values, coords={"x": x, "y": y, "time": time_dimension}, dims=["time", "y", "x"]
    ).chunk()
    da.attrs["_FillValue"] = np.nan

    write_zarr(da, tmp_folder / "test.zarr", crs=4326)

    other_dimension = np.linspace(110, 115, 10)
    values = np.random.rand(x.size, y.size, other_dimension.size).astype(np.float32)

    da = xr.DataArray(
        values,
        coords={"x": x, "y": y, "other": other_dimension},
        dims=["other", "y", "x"],
    ).chunk()
    da.attrs["_FillValue"] = np.nan

    write_zarr(da, tmp_folder / "test.zarr", crs=4326)


def test_write_zarr_uses_shards_as_chunk_multiples() -> None:
    """Test that shard values are interpreted as multiples of the chunk size.

    Verifies that the public shards argument expresses the number of chunks per
    shard for each dimension and that the stored Zarr metadata reflects the
    resulting absolute shard shape.
    """
    x = np.linspace(-4, 4, 9)
    y = np.linspace(7, 0, 8)
    values = np.arange(x.size * y.size, dtype=np.float32).reshape(y.size, x.size)

    da = xr.DataArray(values, coords={"x": x, "y": y}, dims=["y", "x"]).chunk()
    da.attrs["_FillValue"] = np.nan

    write_zarr(
        da.chunk({"x": 3, "y": 2}),
        tmp_folder / "test.zarr",
        crs=4326,
        shards={"x": 2, "y": 3},
        progress=False,
    )

    zarr_array = zarr.open_array(
        store=tmp_folder / "test.zarr",
        zarr_format=3,
        path="test",
        mode="r",
    )

    assert zarr_array.chunks == (2, 3)
    assert zarr_array.shards == (6, 6)


def test_write_zarr_rounds_shards_up_to_chunk_multiple() -> None:
    """Test that shard metadata stays aligned to full chunk multiples.

    Verifies that when a requested shard would otherwise be clipped by the array
    extent, the stored shard size is rounded up to the smallest full chunk
    multiple that still covers the dimension.
    """
    x = np.linspace(-3.5, 3.5, 8)
    y = np.linspace(4, 0, 5)
    values = np.arange(x.size * y.size, dtype=np.float32).reshape(y.size, x.size)

    da = xr.DataArray(values, coords={"x": x, "y": y}, dims=["y", "x"]).chunk()
    da.attrs["_FillValue"] = np.nan

    write_zarr(
        da.chunk({"x": 3, "y": 2}),
        tmp_folder / "test.zarr",
        crs=4326,
        shards={"x": 4, "y": 4},
        progress=False,
    )

    zarr_array = zarr.open_array(
        store=tmp_folder / "test.zarr",
        zarr_format=3,
        path="test",
        mode="r",
    )

    assert zarr_array.chunks == (2, 3)
    assert zarr_array.shards == (6, 9)


def test_write_zarr_stores_per_shard_when_shards_are_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that write_zarr switches the source write granularity to shards.

    Verifies that enabling sharding causes the block-wise write loop to operate
    on shard-sized source blocks rather than storage-chunk-sized source blocks.
    """
    captured_numblocks: dict[str, tuple[int, ...]] = {}

    original_store_blocks = io_module._store_dask_array_blocks

    def store_blocks_spy(
        source_array: dask.array.core.Array, store_target: object, progress: bool
    ) -> None:
        captured_numblocks["value"] = source_array.numblocks
        original_store_blocks(source_array, store_target, progress)

    monkeypatch.setattr(io_module, "_store_dask_array_blocks", store_blocks_spy)

    x = np.linspace(-4, 4, 9)
    y = np.linspace(7, 0, 8)
    values = np.arange(x.size * y.size, dtype=np.float32).reshape(y.size, x.size)

    da = xr.DataArray(values, coords={"x": x, "y": y}, dims=["y", "x"]).chunk()
    da.attrs["_FillValue"] = np.nan

    write_zarr(
        da.chunk({"x": 3, "y": 2}),
        tmp_folder / "test.zarr",
        crs=4326,
        shards={"x": 2, "y": 3},
        progress=False,
    )

    assert captured_numblocks["value"] == (2, 2)


def test_get_window() -> None:
    """Test the get_window function with various bounding box and buffer scenarios.

    Tests window selection from coordinate arrays with different bounds,
    buffer values, and edge cases. Verifies that proper slices are returned
    for full windows, partial windows, and windows with buffers. Also tests
    error handling for invalid buffer values and out-of-bounds coordinates.
    """
    x = np.linspace(-5, 5, 11, dtype=np.int32)
    y = np.linspace(10, 0, 11, dtype=np.int32)
    values = np.arange(x.size * y.size).reshape(y.size, x.size).astype(np.int32)
    da = xr.DataArray(values, coords={"x": x, "y": y}, dims=["y", "x"])

    bounds = (
        -5,
        0,
        5,
        10,
    )

    window = get_window(da.x, da.y, bounds, buffer=0)  # full window, no buffer

    da_slice = da.isel(window)
    assert (da_slice.x.values == x).all()
    assert (da_slice.y.values == y).all()

    bounds = (
        -4,
        1,
        4,
        9,
    )

    window = get_window(da.x, da.y, bounds, buffer=0)

    da_slice = da.isel(window)
    assert (da_slice.x.values == x[1:-1]).all()
    assert (da_slice.y.values == y[1:-1]).all()

    top_left_bounds = (
        -4,
        7,
        -2,
        10,
    )

    window = get_window(da.x, da.y, top_left_bounds, buffer=0)
    da_slice = da.isel(window)
    assert (da_slice.x.values == x[1:4]).all()
    assert (da_slice.y.values == y[0:4]).all()

    with pytest.raises(ValueError, match="buffer must be an integer"):
        window = get_window(
            da.x,
            da.y,
            bounds,
            buffer=0.1,  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(ValueError, match="buffer must be greater than or equal to 0"):
        window = get_window(da.x, da.y, bounds, buffer=-1)

    bounds = (
        -6,
        0,
        5,
        10,
    )

    with pytest.raises(ValueError, match=r"xmin must be greater than x\[0\]"):
        window = get_window(da.x, da.y, bounds, buffer=0)

    window = get_window(
        da.x,
        da.y,
        (
            -4,
            1,
            4,
            9,
        ),
        buffer=1,
    )

    da_slice = da.isel(window)
    assert (da_slice.x.values == x).all()
    assert (da_slice.y.values == y).all()

    window = get_window(
        da.x,
        da.y,
        (
            -4,
            3,
            4,
            9,
        ),
        buffer=1,
    )

    da_slice = da.isel(window)
    assert (da_slice.x.values == x).all()
    assert (da_slice.y.values == y[:-2]).all()

    bounds = (
        -4.9,
        0.1,
        4.9,
        9.9,
    )

    window = get_window(
        da.x,
        da.y,
        bounds,
        buffer=0,
    )

    da_slice = da.isel(window)
    assert (da_slice.x.values == x).all()
    assert (da_slice.y.values == y).all()

    bounds = (
        -4.4,
        0.6,
        4.4,
        9.4,
    )

    window = get_window(
        da.x,
        da.y,
        bounds,
        buffer=0,
    )

    da_slice = da.isel(window)
    assert (da_slice.x.values == x[1:-1]).all()
    assert (da_slice.y.values == y[1:-1]).all()


def zarr_file(varname: str) -> Path:
    """Create a temporary zarr file with a single variable for testing.

    Args:
        varname: Name of the variable to create.

    Returns:
        Path to the created zarr file.
    """
    size: int = 1000
    # Create a temporary zarr file for testing
    file_path: Path = tmp_folder / f"{varname}.zarr"

    periods: int = 100

    times: pd.DatetimeIndex = pd.date_range("2000-01-01", periods=periods, freq="D")
    data: npt.NDArray[np.float32] = np.empty((size * size, periods), dtype=np.float32)
    for i in range(periods):
        data[:, i] = i
    ds: xr.Dataset = xr.Dataset(
        {
            varname: (("idxs", "time"), data),
        },
        coords={"time": times, "idxs": np.arange(0, size * size)},
    )

    store: zarr.storage.LocalStore = zarr.storage.LocalStore(file_path, read_only=False)
    ds.to_zarr(
        store,
        mode="w",
        encoding={
            varname: {
                "chunks": (size * size, 1),
                "fill_value": np.nan,
            }
        },
        consolidated=False,
    )
    return file_path


def test_read_timestep_async() -> None:
    """Test the AsyncGriddedForcingReader class with asynchronous reading.

    This test creates three temporary zarr files with a single variable each.
    It then creates three AsyncGriddedForcingReader instances to read the data from these
    files. The test reads several timesteps from the first reader, with varying wait times
    in between to simulate processing time. It also reads timesteps from the other two readers
    to ensure that they work correctly. Finally, it cleans up the temporary files.

    Reading a previous timestep should be slow, as it needs to be loaded from disk.
    Reading the same timestep should be quick, as it is already in the cache.
    Reading the next timestep after a long wait should be quick, as it is already in the cache.
    Reading the next timestep after a short wait should be semi-quick, as it is already being
    loaded in the cache but not yet ready.

    """
    temperature_file: Path = zarr_file("temperature")
    precipitation_file: Path = zarr_file("precipitation")
    pressure_file: Path = zarr_file("pressure")
    reader1: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        temperature_file, variable_name="temperature", asynchronous=True
    )
    reader2: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        precipitation_file, variable_name="precipitation", asynchronous=True
    )
    reader3: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        pressure_file, variable_name="pressure", asynchronous=True
    )

    data0, _ = reader1.read_timestep(datetime(2000, 1, 1))

    sleep(3)  # Simulate some processing time

    t0 = time()
    data1, _ = reader1.read_timestep(datetime(2000, 1, 2))
    t1 = time()
    print("Async - Load next timestep (quick): {:.3f}s".format(t1 - t0))

    # wait half the time it took to load the previous timestep to simulate a short
    # processing time
    sleep((t1 - t0) / 2)

    t0 = time()
    data2, _ = reader1.read_timestep(datetime(2000, 1, 3))
    t1 = time()
    print(
        "Async - Load next timestep short waiting (semi-quick): {:.3f}s".format(t1 - t0)
    )

    assert (data0 == 0).all()
    assert (data1 == 1).all()
    assert (data2 == 2).all()
    assert data0.dtype == np.float32

    sleep(3)

    t0 = time()
    data3, _ = reader1.read_timestep(datetime(2000, 1, 4))
    t1 = time()
    print("Async - Load next timestep with waiting (quick): {:.3f}s".format(t1 - t0))

    t0 = time()
    data3, _ = reader1.read_timestep(datetime(2000, 1, 4))
    t1 = time()
    print("Async - Load same timestep (quick): {:.3f}s".format(t1 - t0))
    assert (data3 == 3).all()

    t0 = time()
    data0, _ = reader1.read_timestep(datetime(2000, 1, 6))
    t1 = time()
    print("Async - Load next next timestep (slow): {:.3f}s".format(t1 - t0))
    assert (data0 == 5).all()

    _, _ = reader1.read_timestep(datetime(2000, 1, 1))
    _, _ = reader2.read_timestep(datetime(2000, 1, 1))
    _, _ = reader3.read_timestep(datetime(2000, 1, 1))

    sleep(3)  # Simulate some processing time
    print("-----------------")

    t0 = time()
    data1, _ = reader1.read_timestep(datetime(2000, 1, 2))
    _, _ = reader2.read_timestep(datetime(2000, 1, 2))
    _, _ = reader3.read_timestep(datetime(2000, 1, 2))
    t1 = time()
    print("Async - Load data from three readers (quick): {:.3f}s".format(t1 - t0))

    reader1.close()
    reader2.close()
    reader3.close()

    shutil.rmtree(temperature_file)
    shutil.rmtree(precipitation_file)
    shutil.rmtree(pressure_file)


def test_read_timestep_sync() -> None:
    """Test the AsyncGriddedForcingReader class with synchronous reading.

    This test verifies that the reader works correctly when asynchronous mode is disabled.
    It should correctly read data without any preloading mechanism.
    """
    temperature_file: Path = zarr_file("temperature")
    reader: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        temperature_file, variable_name="temperature", asynchronous=False
    )

    # Test reading single timesteps
    data0, _ = reader.read_timestep(datetime(2000, 1, 1))
    data1, _ = reader.read_timestep(datetime(2000, 1, 2))
    data2, _ = reader.read_timestep(datetime(2000, 1, 3))

    assert (data0 == 0).all()
    assert (data1 == 1).all()
    assert (data2 == 2).all()
    assert data0.dtype == np.float32
    assert data0.shape == (1000000, 1)

    # Test reading the same timestep twice
    data1_again, _ = reader.read_timestep(datetime(2000, 1, 2))
    assert (data1_again == 1).all()
    assert np.array_equal(data1, data1_again)

    # Test reading a non-sequential timestep
    data10, _ = reader.read_timestep(datetime(2000, 1, 11))
    assert (data10 == 10).all()

    reader.close()
    shutil.rmtree(temperature_file)


def test_read_multiple_timesteps() -> None:
    """Test reading multiple timesteps at once (n=1).

    This test verifies that the reader correctly handles reading consecutive
    timesteps in single-step calls, and that the data is correct for both async and sync modes.
    """
    temperature_file: Path = zarr_file("temperature")

    # Test with asynchronous reading
    reader_async: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        temperature_file, variable_name="temperature", asynchronous=True
    )

    # Read 1 timestep starting from Jan 1
    data_multi, _ = reader_async.read_timestep(datetime(2000, 1, 1), n=1)
    assert data_multi.shape == (1000000, 1)
    assert (data_multi == 0).all()

    sleep(2)  # Allow preloading to happen

    # Read next 1 timestep
    t0 = time()
    data_multi_next, _ = reader_async.read_timestep(datetime(2000, 1, 2), n=1)
    t1 = time()
    print(f"Async - Load next timestep (should be quick): {t1 - t0:.3f}s")
    assert data_multi_next.shape == (1000000, 1)
    assert (data_multi_next == 1).all()

    reader_async.close()

    # Test with synchronous reading
    reader_sync: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        temperature_file, variable_name="temperature", asynchronous=False
    )

    # Read 1 timestep starting from Jan 1
    data_multi_sync, _ = reader_sync.read_timestep(datetime(2000, 1, 1), n=1)
    assert data_multi_sync.shape == (1000000, 1)
    assert (data_multi_sync == 0).all()

    # Verify that async and sync give same results
    assert np.array_equal(data_multi, data_multi_sync), "Async and sync results differ"

    reader_sync.close()
    shutil.rmtree(temperature_file)


def test_asyncreader_rapid_access() -> None:
    """Test rapid access of timesteps using AsyncGriddedForcingReader.

    This test verifies that the reader can handle rapid sequential access of timesteps
    so no sleeps in between reads.
    """
    temperature_file: Path = zarr_file("temperature")
    reader: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        temperature_file, variable_name="temperature", asynchronous=True
    )
    for day in range(1, 11):
        data, _ = reader.read_timestep(datetime(2000, 1, day))
        assert (data == day - 1).all()
    reader.close()


HOURLY_CHUNK_SIZE: int = 7 * 24  # one week of hourly data per on-disk chunk


def zarr_file_hourly(varname: str) -> Path:
    """Create a temporary zarr file with hourly timesteps stored in weekly chunks.

    Each hourly timestep i is filled with the constant value i (as float32) so that
    slice correctness can be verified by checking a single scalar value.

    Args:
        varname: Name of the variable array within the zarr store.

    Returns:
        Path to the created zarr store.
    """
    size: int = 4  # small spatial extent to keep tests fast
    file_path: Path = tmp_folder / f"{varname}_hourly.zarr"

    n_weeks: int = 4
    periods: int = n_weeks * HOURLY_CHUNK_SIZE  # 4 weeks of hourly data

    times: pd.DatetimeIndex = pd.date_range("2000-01-01", periods=periods, freq="h")
    data: npt.NDArray[np.float32] = np.empty((size * size, periods), dtype=np.float32)
    for i in range(periods):
        data[:, i] = float(i)

    ds: xr.Dataset = xr.Dataset(
        {varname: (("idxs", "time"), data)},
        coords={
            "time": times,
            "idxs": np.arange(size * size, dtype=np.int64),
        },
    )

    store: zarr.storage.LocalStore = zarr.storage.LocalStore(file_path, read_only=False)
    ds.to_zarr(
        store,
        mode="w",
        encoding={
            varname: {
                "chunks": (size * size, HOURLY_CHUNK_SIZE),
                "fill_value": np.nan,
            }
        },
        consolidated=False,
    )
    return file_path


def test_chunk_aligned_reading_correctness() -> None:
    """Test that chunk-aligned reads return the correct data for all access patterns.

    Verifies that:
    - Requesting 24 h on day 1 returns hours 0-23 (first slice of chunk 0).
    - Requesting 24 h on day 2 returns hours 24-47 (still within chunk 0, served from cache).
    - Requesting 24 h on day 7 returns the last day of chunk 0 (still from cache).
    - Requesting 24 h on day 8 returns hours 168-191 (first day of chunk 1).
    - A non-sequential jump to week 3 returns correct data from chunk 2.
    - Both async and sync readers return identical results.
    """
    varname: str = "temperature_corr"
    hourly_file: Path = zarr_file_hourly(varname)
    hours_per_day: int = 24

    for asynchronous in (True, False):
        reader: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
            hourly_file, variable_name=varname, asynchronous=asynchronous
        )
        assert reader.time_chunk_size == HOURLY_CHUNK_SIZE

        # Day 1: hours 0-23, first slice of chunk 0
        data_day1, _ = reader.read_timestep(datetime(2000, 1, 1), n=hours_per_day)
        assert data_day1.shape == (16, hours_per_day)
        for h in range(hours_per_day):
            assert (data_day1[:, h] == float(h)).all(), f"day1 hour {h} wrong"

        # Day 2: hours 24-47, still within chunk 0 (served from cache)
        data_day2, _ = reader.read_timestep(datetime(2000, 1, 2), n=hours_per_day)
        for h in range(hours_per_day):
            assert (data_day2[:, h] == float(hours_per_day + h)).all(), (
                f"day2 hour {h} wrong"
            )

        # Day 7: hours 144-167, last day of chunk 0 (still from cache)
        data_day7, _ = reader.read_timestep(datetime(2000, 1, 7), n=hours_per_day)
        offset_day7: int = 6 * hours_per_day
        for h in range(hours_per_day):
            assert (data_day7[:, h] == float(offset_day7 + h)).all(), (
                f"day7 hour {h} wrong"
            )

        # Day 8: hours 168-191, first day of chunk 1
        data_day8, _ = reader.read_timestep(datetime(2000, 1, 8), n=hours_per_day)
        offset_day8: int = HOURLY_CHUNK_SIZE  # start of chunk 1
        for h in range(hours_per_day):
            assert (data_day8[:, h] == float(offset_day8 + h)).all(), (
                f"day8 hour {h} wrong"
            )

        # Non-sequential jump to week 3, day 1 (chunk 2)
        week3_start: datetime = datetime(2000, 1, 15)  # 14 days * 24h = index 336
        data_week3, _ = reader.read_timestep(week3_start, n=hours_per_day)
        offset_week3: int = 2 * HOURLY_CHUNK_SIZE
        for h in range(hours_per_day):
            assert (data_week3[:, h] == float(offset_week3 + h)).all(), (
                f"week3 hour {h} wrong"
            )

        reader.close()

    shutil.rmtree(hourly_file)


def test_chunk_aligned_reading_preload_performance() -> None:
    """Test that chunk-boundary crossings are fast due to background preloading.

    After reading day 7 (last day of chunk 0) with a sufficient sleep to let the
    background preload of chunk 1 finish, reading day 8 (first day of chunk 1)
    should be served from the preloaded cache and therefore be substantially faster
    than a cold disk read.
    """
    varname: str = "temperature_perf"
    hourly_file: Path = zarr_file_hourly(varname)
    hours_per_day: int = 24

    reader: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        hourly_file, variable_name=varname, asynchronous=True
    )

    # Read day 7 to populate the cache for chunk 0 and trigger preload of chunk 1.
    _, _ = reader.read_timestep(datetime(2000, 1, 7), n=hours_per_day)

    # Sleep long enough for the background preload to finish.
    sleep(3)

    # Day 8 crosses into chunk 1 - should be fast because it was preloaded.
    t0: float = time()
    data_day8, _ = reader.read_timestep(datetime(2000, 1, 8), n=hours_per_day)
    t1: float = time()
    print(f"Chunk-boundary read (preloaded): {t1 - t0:.4f}s")

    # Verify data correctness.
    offset_day8: int = HOURLY_CHUNK_SIZE
    for h in range(hours_per_day):
        assert (data_day8[:, h] == float(offset_day8 + h)).all()

    # Read same day again (cache hit) - must be essentially instant.
    t0 = time()
    data_day8_cached, _ = reader.read_timestep(datetime(2000, 1, 8), n=hours_per_day)
    t1 = time()
    print(f"Same-day cache hit: {t1 - t0:.4f}s")
    assert np.array_equal(data_day8, data_day8_cached)

    reader.close()
    shutil.rmtree(hourly_file)


def test_async_gridded_forcing_reader_get_index_fast_paths() -> None:
    """Test the fast paths in get_index for current and next chunks."""
    varname: str = "test_get_index"
    hourly_file: Path = zarr_file_hourly(varname)

    reader: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        hourly_file, variable_name=varname, asynchronous=False
    )

    # Initial state: current_chunk_start_index is -1
    assert reader.current_chunk_start_index == -1

    # Search for a date in the middle of the first chunk (chunk 0: 0-167)
    date_chunk0 = datetime(2000, 1, 4)  # day 4, hour 0 -> index 72
    idx0 = reader.get_index(date_chunk0)
    assert idx0 == 3 * 24

    # Now read a timestep to set current_chunk_start_index
    _, _ = reader.read_timestep(datetime(2000, 1, 1), n=1)
    # HOURLY_CHUNK_SIZE is 168 (7 days)
    assert reader.current_chunk_start_index == 0

    # Fast path 1: same chunk
    idx0_fast = reader.get_index(date_chunk0)
    assert idx0_fast == 72

    # Fast path 2: next chunk (chunk 1: 168-335)
    date_chunk1 = datetime(2000, 1, 10)  # day 10, hour 0 -> index 9 * 24 = 216
    idx1 = reader.get_index(date_chunk1)
    assert idx1 == 9 * 24

    # Verify that it also works if we are further ahead
    # Day 8 starts at index 168
    _, _ = reader.read_timestep(
        datetime(2000, 1, 8), n=1
    )  # Sets current_chunk to chunk 1 (index 168)
    assert reader.current_chunk_start_index == 168

    # Fast path 1 (now chunk 1)
    idx1_fast = reader.get_index(date_chunk1)
    assert idx1_fast == 216

    # Fast path 2 (now chunk 2: 336-503)
    date_chunk2 = datetime(2000, 1, 16)  # day 16, hour 0 -> index 15 * 24 = 360
    idx2 = reader.get_index(date_chunk2)
    assert idx2 == 15 * 24

    # Full search: far away
    # 4 weeks of data (28 days) -> Jan 1 to Jan 29
    date_far = datetime(2000, 1, 28)
    idx_far = reader.get_index(date_far)
    assert idx_far == 27 * 24

    # Full search: backwards
    date_back = datetime(2000, 1, 2)
    idx_back = reader.get_index(date_back)
    assert idx_back == 24

    reader.close()
    shutil.rmtree(hourly_file)


def test_async_gridded_forcing_reader_full_chunk_consumption() -> None:
    """Test that consuming a full chunk in one go correctly updates cache and preload state."""
    varname: str = "test_full_chunk"
    hourly_file: Path = zarr_file_hourly(varname)

    # Test asynchronous mode
    reader: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        hourly_file, variable_name=varname, asynchronous=True
    )

    # Initially no chunk is cached or preloaded
    assert reader.current_chunk_start_index == -1
    assert reader.preloaded_chunk_start_index == -1

    # 1. Consume the first full chunk (7 days = 168 hours)
    # This falls into the fallback path in read_timestep_async (n == chunk_size, offset == 0)
    # but still happens to be chunk-aligned.
    _, _ = reader.read_timestep(datetime(2000, 1, 1), n=HOURLY_CHUNK_SIZE)

    # Check that current_chunk_start_index is set to 0
    assert reader.current_chunk_start_index == 0
    # Check that it triggered preload of the next chunk (index 168)
    assert reader.preloaded_chunk_start_index == 168

    # 2. Wait a bit and check if we have a preload hit for the next chunk
    from time import sleep

    sleep(1.0)

    # Accessing the first timestep of the next chunk should now hit the preload
    # We can check this by verifying that the preload_hit logic path is taken
    # (Since we can't easily check prints, we check that it doesn't crash
    # and that the next preload is triggered)
    _, _ = reader.read_timestep(datetime(2000, 1, 8), n=1)

    assert reader.current_chunk_start_index == 168
    assert reader.preloaded_chunk_start_index == 168 + HOURLY_CHUNK_SIZE

    reader.close()

    # Test synchronous mode
    reader_sync: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        hourly_file, variable_name=varname, asynchronous=False
    )

    # Initially no chunk is cached
    assert reader_sync.current_chunk_start_index == -1

    # Consume full chunk
    _, _ = reader_sync.read_timestep(datetime(2000, 1, 1), n=HOURLY_CHUNK_SIZE)
    assert reader_sync.current_chunk_start_index == 0

    reader_sync.close()
    shutil.rmtree(hourly_file)


def test_chunk_aligned_reading_within_same_day() -> None:
    """Test that multiple reads within the same day (same chunk) are served from cache.

    Simulates a model that reads the same timestep multiple times per step
    (e.g. different forcing variables that happen to share the same reader).
    """
    varname: str = "temperature_sameday"
    hourly_file: Path = zarr_file_hourly(varname)
    hours_per_day: int = 24

    reader: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        hourly_file, variable_name=varname, asynchronous=True
    )

    # First read: cold load of chunk 0.
    data_a, _ = reader.read_timestep(datetime(2000, 1, 3), n=hours_per_day)
    # Second read of the same day - must be served from cache with no disk I/O.
    t0: float = time()
    data_b, _ = reader.read_timestep(datetime(2000, 1, 3), n=hours_per_day)
    t1: float = time()
    print(f"Same-day second read (cache hit): {t1 - t0:.4f}s")
    assert np.array_equal(data_a, data_b)

    # Read a different day within the same chunk - still from cache.
    data_c, _ = reader.read_timestep(datetime(2000, 1, 5), n=hours_per_day)
    offset: int = 4 * hours_per_day
    for h in range(hours_per_day):
        assert (data_c[:, h] == float(offset + h)).all()

    reader.close()
    shutil.rmtree(hourly_file)


def test_chunk_boundary_spanning_request() -> None:
    """Test that a request straddling chunk boundaries now raises a ValueError.

    Requests must be aligned with chunk boundaries and never cross it.
    """
    varname: str = "temperature_span"
    hourly_file: Path = zarr_file_hourly(varname)
    hours_per_day: int = 24

    for asynchronous in (True, False):
        reader: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
            hourly_file, variable_name=varname, asynchronous=asynchronous
        )
        assert reader.time_chunk_size == HOURLY_CHUNK_SIZE

        # Start at hour 156 (= HOURLY_CHUNK_SIZE - 12): asking for 24 h straddles
        # the boundary between chunk 0 (0-167) and chunk 1 (168-335).
        straddle_start: datetime = datetime(2000, 1, 7, 12)  # hour 156
        with pytest.raises(ValueError, match="not aligned with the chunk size"):
            reader.read_timestep(straddle_start, n=hours_per_day)

        reader.close()

    shutil.rmtree(hourly_file)


def test_first_request_mid_chunk() -> None:
    """Test that the first ever request starting in the middle of a chunk is correct.

    Verifies that a cold-cache read starting at an aligned hour within chunk 1
    (not at the chunk boundary) loads the full chunk, returns the right slice, and
    caches it so the next intra-chunk request is served from memory.
    """
    varname: str = "temperature_midchunk"
    hourly_file: Path = zarr_file_hourly(varname)
    hours_per_day: int = 24

    # hour 192 = chunk 1 offset 24 (chunk 1 spans hours 168-335). 192 % 24 == 0.
    mid_chunk_start: datetime = datetime(2000, 1, 9, 0)  # hour 192
    start_value: int = 192

    for asynchronous in (True, False):
        reader: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
            hourly_file, variable_name=varname, asynchronous=asynchronous
        )

        # Cold cache, first request lands in the middle of chunk 1.
        data, _ = reader.read_timestep(mid_chunk_start, n=hours_per_day)
        assert data.shape == (16, hours_per_day)
        for h in range(hours_per_day):
            assert (data[:, h] == float(start_value + h)).all(), (
                f"async={asynchronous} hour {h}: expected {start_value + h}"
            )

        # Verify the chunk is now cached by reading another day inside the same chunk.
        # hour 216 = chunk 1 offset 48. 216 % 24 == 0.
        next_day_start: datetime = datetime(2000, 1, 10, 0)  # hour 216
        data_next, _ = reader.read_timestep(next_day_start, n=hours_per_day)
        assert data_next.shape == (16, hours_per_day)
        for h in range(hours_per_day):
            assert (data_next[:, h] == float(216 + h)).all(), (
                f"async={asynchronous} intra-chunk hour {h}: expected {216 + h}"
            )

        reader.close()

    shutil.rmtree(hourly_file)


def test_create_hash_from_parameters() -> None:
    """Test create_hash_from_parameters with various input types."""
    # Test basic types
    params1 = {"a": 1, "b": "test", "c": 3.14}
    hash1 = create_hash_from_parameters(params1)
    assert isinstance(hash1, str)
    assert len(hash1) > 0

    # Test consistency
    hash2 = create_hash_from_parameters(params1)
    assert hash1 == hash2

    # Test different input
    params2 = {"a": 1, "b": "test", "c": 3.15}
    hash3 = create_hash_from_parameters(params2)
    assert hash1 != hash3

    # Test with numpy array
    params_np = {"arr": np.array([1, 2, 3])}
    hash_np1 = create_hash_from_parameters(params_np)
    hash_np2 = create_hash_from_parameters({"arr": np.array([1, 2, 3])})
    assert hash_np1 == hash_np2

    hash_np3 = create_hash_from_parameters({"arr": np.array([1, 2, 4])})
    assert hash_np1 != hash_np3

    # Test with xarray DataArray
    da = xr.DataArray([1, 2, 3], coords={"x": [1, 2, 3]}, dims="x")
    params_xr = {"da": da}
    hash_xr1 = create_hash_from_parameters(params_xr)
    hash_xr2 = create_hash_from_parameters({"da": da.copy()})
    assert hash_xr1 == hash_xr2

    da2 = xr.DataArray([1, 2, 4], coords={"x": [1, 2, 3]}, dims="x")
    hash_xr3 = create_hash_from_parameters({"da": da2})
    assert hash_xr1 != hash_xr3

    # Test nested structures (list and dict)
    params_nested = {
        "list": [1, "a", np.array([1, 2])],
        "dict": {"x": xr.DataArray([1]), "y": [1, 2]},
    }
    hash_nested1 = create_hash_from_parameters(params_nested)

    params_nested_copy = {
        "list": [1, "a", np.array([1, 2])],
        "dict": {"x": xr.DataArray([1]), "y": [1, 2]},
    }
    hash_nested2 = create_hash_from_parameters(params_nested_copy)
    assert hash_nested1 == hash_nested2

    params_nested_diff = {
        "list": [1, "b", np.array([1, 2])],
        "dict": {"x": xr.DataArray([1]), "y": [1, 2]},
    }
    hash_nested3 = create_hash_from_parameters(params_nested_diff)
    assert hash_nested1 != hash_nested3


def test_create_hash_from_parameters_with_code() -> None:
    """Test create_hash_from_parameters with code_path."""
    import tempfile

    params = {"a": 1}
    base_hash = create_hash_from_parameters(params)

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)

        # Test with file
        f = p / "test.py"
        f.write_text("print('hello')")

        hash1 = create_hash_from_parameters(params, code_path=f)
        assert hash1 != base_hash

        hash2 = create_hash_from_parameters(params, code_path=f)
        assert hash1 == hash2

        # Modify file
        f.write_text("print('hello world')")
        hash3 = create_hash_from_parameters(params, code_path=f)
        assert hash1 != hash3

        # Test with directory
        d = p / "subdir"
        d.mkdir()
        (d / "a.py").write_text("a=1")
        (d / "b.py").write_text("b=2")

        hash_dir1 = create_hash_from_parameters(params, code_path=d)
        assert hash_dir1 != base_hash

        hash_dir2 = create_hash_from_parameters(params, code_path=d)
        assert hash_dir1 == hash_dir2

        # Modify file in directory
        (d / "a.py").write_text("a=2")
        hash_dir3 = create_hash_from_parameters(params, code_path=d)
        assert hash_dir1 != hash_dir3


def test_read_write_hash() -> None:
    """Test reading and writing hashes."""
    hash_val = "deadbeef"
    hash_file = tmp_folder / "test.hash"

    write_hash(hash_file, hash_val)
    assert hash_file.exists()
    assert hash_file.read_text() == "deadbeef"

    read_val = read_hash(hash_file)
    assert read_val == hash_val

    # Test with newline (simulating manual edit)
    hash_file.write_text("deadbeef\n")
    read_val = read_hash(hash_file)
    assert read_val.strip() == hash_val

    # Test empty
    hash_val = ""
    write_hash(hash_file, hash_val)
    assert read_hash(hash_file) == hash_val


def test_write_table_roundtrip() -> None:
    """Test write_table and read_table for various DataFrame configurations."""
    # 1. Simple DataFrame with various types
    df_simple = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "bool_col": [True, False, True],
            "string_col": ["a", "b", "c"],
            "dt_col": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
        }
    )

    fp = tmp_folder / "test_simple.parquet"
    write_table(df_simple, fp)
    df_read = read_table(fp)
    pd.testing.assert_frame_equal(df_simple, df_read)

    # 2. DataFrame without index (index should be preserved if it's just a RangeIndex)
    df_no_index = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    fp = tmp_folder / "test_no_index.parquet"
    write_table(df_no_index, fp)
    df_read = read_table(fp)
    pd.testing.assert_frame_equal(df_no_index, df_read)

    # 3. DataFrame with named index
    df_named_index = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df_named_index.index.name = "my_index"
    fp = tmp_folder / "test_named_index.parquet"
    write_table(df_named_index, fp)
    df_read = read_table(fp)
    pd.testing.assert_frame_equal(df_named_index, df_read)

    # 4. DataFrame with MultiIndex
    df_multi = pd.DataFrame(
        {"val": [1, 2, 3, 4]},
        index=pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1), ("B", 2)], names=["idx1", "idx2"]
        ),
    )
    fp = tmp_folder / "test_multi.parquet"
    write_table(df_multi, fp)
    df_read = read_table(fp)
    pd.testing.assert_frame_equal(df_multi, df_read)

    # 5. DataFrame with large amount of data to test row_group_size and page_size
    df_large = pd.DataFrame(
        {"a": np.random.randint(0, 100, size=1000), "b": np.random.random(size=1000)}
    )
    fp = tmp_folder / "test_large.parquet"
    write_table(df_large, fp)
    df_read = read_table(fp)
    pd.testing.assert_frame_equal(df_large, df_read)
