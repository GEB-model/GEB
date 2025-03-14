import numpy as np
import zarr
import pandas as pd
from datetime import date
import xarray as xr
from time import time, sleep
from geb.workflows import (
    AsyncForcingReader,
)
from .testconfig import tmp_folder


def zarr_file(varname):
    size = 1000
    # Create a temporary zarr file for testing
    file_path = tmp_folder / f"{varname}.zarr.zip"
    if file_path.exists():
        file_path.unlink()

    periods = 100

    times = pd.date_range("2000-01-01", periods=periods, freq="D")
    data = np.empty((periods, size, size), dtype=np.int32)
    for i in range(periods):
        data[i][:] = i
    ds = xr.Dataset(
        {
            varname: (("time", "x", "y"), data),
        },
        coords={"time": times, "x": np.arange(0, size), "y": np.arange(0, size)},
    )

    store = zarr.storage.ZipStore(file_path, mode="w")
    ds.to_zarr(
        store,
        mode="w",
        encoding={
            varname: {
                "chunks": (1, size, size),
            }
        },
    )
    return file_path


def test_read_timestep():
    temperature_file = zarr_file("temperature")
    precipitation_file = zarr_file("precipitation")
    pressure_file = zarr_file("pressure")
    reader1 = AsyncForcingReader(temperature_file, variable_name="temperature")
    reader2 = AsyncForcingReader(precipitation_file, variable_name="precipitation")
    reader3 = AsyncForcingReader(pressure_file, variable_name="pressure")

    data0 = reader1.read_timestep(date(2000, 1, 1))

    sleep(3)  # Simulate some processing time

    t0 = time()
    data1 = reader1.read_timestep(date(2000, 1, 2))
    t1 = time()
    print("Load next timestep (quick): {:.3f}s".format(t1 - t0))

    # wait half the time it took to load the previous timestep to simulate a short
    # processing time
    sleep((t1 - t0) / 2)

    t0 = time()
    data2 = reader1.read_timestep(date(2000, 1, 3))
    t1 = time()
    print("Load next timestep short waiting (semi-quick): {:.3f}s".format(t1 - t0))

    assert (data0 == 0).all()
    assert (data1 == 1).all()
    assert (data2 == 2).all()
    assert data0.dtype == np.int32

    sleep(3)

    t0 = time()
    data3 = reader1.read_timestep(date(2000, 1, 4))
    t1 = time()
    print("Load next timestep with waiting (quick): {:.3f}s".format(t1 - t0))

    t0 = time()
    data3 = reader1.read_timestep(date(2000, 1, 4))
    t1 = time()
    print("Load same timestep (quick): {:.3f}s".format(t1 - t0))
    assert (data3 == 3).all()

    t0 = time()
    data0 = reader1.read_timestep(date(2000, 1, 1))
    t1 = time()
    print("Load previous timestep (slow): {:.3f}s".format(t1 - t0))
    assert (data0 == 0).all()

    reader2.read_timestep(date(2000, 1, 1))
    reader3.read_timestep(date(2000, 1, 1))

    sleep(3)  # Simulate some processing time
    print("-----------------")

    t0 = time()
    data1 = reader1.read_timestep(date(2000, 1, 2))
    reader2.read_timestep(date(2000, 1, 2))
    reader3.read_timestep(date(2000, 1, 2))
    t1 = time()
    print("Load data from three readers (quick): {:.3f}s".format(t1 - t0))

    reader1.close()
    reader2.close()
    reader3.close()

    temperature_file.unlink()
    precipitation_file.unlink()
    pressure_file.unlink()
