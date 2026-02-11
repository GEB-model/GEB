"""Tests for I/O workflow functions."""

import shutil
import warnings
from datetime import datetime
from pathlib import Path
from time import sleep, time

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr
import zarr.storage
from zarr.codecs.numcodecs import FixedScaleOffset

from geb.workflows.io import (
    AsyncGriddedForcingReader,
    calculate_scaling,
    create_hash_from_parameters,
    get_window,
    read_hash,
    write_hash,
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
    data: npt.NDArray[np.float32] = np.empty((periods, size, size), dtype=np.float32)
    for i in range(periods):
        data[i][:] = i
    ds: xr.Dataset = xr.Dataset(
        {
            varname: (("time", "x", "y"), data),
        },
        coords={"time": times, "x": np.arange(0, size), "y": np.arange(0, size)[::-1]},
    )

    store: zarr.storage.LocalStore = zarr.storage.LocalStore(file_path, read_only=False)
    ds.to_zarr(
        store,
        mode="w",
        encoding={
            varname: {
                "chunks": (1, size, size),
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

    data0 = reader1.read_timestep(datetime(2000, 1, 1))

    sleep(3)  # Simulate some processing time

    t0 = time()
    data1 = reader1.read_timestep(datetime(2000, 1, 2))
    t1 = time()
    print("Async - Load next timestep (quick): {:.3f}s".format(t1 - t0))

    # wait half the time it took to load the previous timestep to simulate a short
    # processing time
    sleep((t1 - t0) / 2)

    t0 = time()
    data2 = reader1.read_timestep(datetime(2000, 1, 3))
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
    data3 = reader1.read_timestep(datetime(2000, 1, 4))
    t1 = time()
    print("Async - Load next timestep with waiting (quick): {:.3f}s".format(t1 - t0))

    t0 = time()
    data3 = reader1.read_timestep(datetime(2000, 1, 4))
    t1 = time()
    print("Async - Load same timestep (quick): {:.3f}s".format(t1 - t0))
    assert (data3 == 3).all()

    t0 = time()
    data0 = reader1.read_timestep(datetime(2000, 1, 6))
    t1 = time()
    print("Async - Load next next timestep (slow): {:.3f}s".format(t1 - t0))
    assert (data0 == 5).all()

    reader1.read_timestep(datetime(2000, 1, 1))
    reader2.read_timestep(datetime(2000, 1, 1))
    reader3.read_timestep(datetime(2000, 1, 1))

    sleep(3)  # Simulate some processing time
    print("-----------------")

    t0 = time()
    data1 = reader1.read_timestep(datetime(2000, 1, 2))
    reader2.read_timestep(datetime(2000, 1, 2))
    reader3.read_timestep(datetime(2000, 1, 2))
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
    data0 = reader.read_timestep(datetime(2000, 1, 1))
    data1 = reader.read_timestep(datetime(2000, 1, 2))
    data2 = reader.read_timestep(datetime(2000, 1, 3))

    assert (data0 == 0).all()
    assert (data1 == 1).all()
    assert (data2 == 2).all()
    assert data0.dtype == np.float32
    assert data0.shape == (1, 1000, 1000)

    # Test reading the same timestep twice
    data1_again = reader.read_timestep(datetime(2000, 1, 2))
    assert (data1_again == 1).all()
    assert np.array_equal(data1, data1_again)

    # Test reading a non-sequential timestep
    data10 = reader.read_timestep(datetime(2000, 1, 11))
    assert (data10 == 10).all()

    reader.close()
    shutil.rmtree(temperature_file)


def test_read_multiple_timesteps() -> None:
    """Test reading multiple timesteps at once (n>1).

    This test verifies that the reader correctly handles reading multiple consecutive
    timesteps in a single call, and that the data is correct for both async and sync modes.
    """
    temperature_file: Path = zarr_file("temperature")

    # Test with asynchronous reading
    reader_async: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        temperature_file, variable_name="temperature", asynchronous=True
    )

    # Read 5 timesteps starting from Jan 1
    data_multi = reader_async.read_timestep(datetime(2000, 1, 1), n=5)
    assert data_multi.shape == (5, 1000, 1000)
    for i in range(5):
        assert (data_multi[i] == i).all(), f"Timestep {i} has incorrect data"

    sleep(2)  # Allow preloading to happen

    # Read next 5 timesteps
    t0 = time()
    data_multi_next = reader_async.read_timestep(datetime(2000, 1, 6), n=5)
    t1 = time()
    print(f"Async - Load next 5 timesteps (should be quick): {t1 - t0:.3f}s")
    assert data_multi_next.shape == (5, 1000, 1000)
    for i in range(5):
        assert (data_multi_next[i] == i + 5).all(), (
            f"Timestep {i + 5} has incorrect data"
        )

    # Read 3 timesteps from a different position
    data_multi_jump = reader_async.read_timestep(datetime(2000, 1, 20), n=3)
    assert data_multi_jump.shape == (3, 1000, 1000)
    for i in range(3):
        assert (data_multi_jump[i] == i + 19).all(), (
            f"Timestep {i + 19} has incorrect data"
        )

    reader_async.close()

    # Test with synchronous reading
    reader_sync: AsyncGriddedForcingReader = AsyncGriddedForcingReader(
        temperature_file, variable_name="temperature", asynchronous=False
    )

    # Read 5 timesteps starting from Jan 1
    data_multi_sync = reader_sync.read_timestep(datetime(2000, 1, 1), n=5)
    assert data_multi_sync.shape == (5, 1000, 1000)
    for i in range(5):
        assert (data_multi_sync[i] == i).all(), f"Sync: Timestep {i} has incorrect data"

    # Verify that async and sync give same results
    assert np.array_equal(data_multi, data_multi_sync), "Async and sync results differ"

    # Read next 5 timesteps
    data_multi_next_sync = reader_sync.read_timestep(datetime(2000, 1, 6), n=5)
    assert data_multi_next_sync.shape == (5, 1000, 1000)
    for i in range(5):
        assert (data_multi_next_sync[i] == i + 5).all(), (
            f"Sync: Timestep {i + 5} has incorrect data"
        )

    # Verify that async and sync give same results
    assert np.array_equal(data_multi_next, data_multi_next_sync), (
        "Async and sync results differ for next batch"
    )

    # Test edge case: reading exactly 1 timestep with n=1
    data_single = reader_sync.read_timestep(datetime(2000, 1, 15), n=1)
    assert data_single.shape == (1, 1000, 1000)
    assert (data_single[0] == 14).all()

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
        data = reader.read_timestep(datetime(2000, 1, day))
        assert (data == day - 1).all()
    reader.close()


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
