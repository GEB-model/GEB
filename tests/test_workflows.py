import pytest
import numpy as np
import pandas as pd
from datetime import date
import xarray as xr
from time import time, sleep
import asyncio
import threading
from geb.workflows import (
    AsyncXarrayReader,
)


@pytest.fixture
def netcdf_file(tmp_path):
    size = 20_000
    # Create a temporary NetCDF file for testing
    filename = tmp_path / "test_data.nc"

    periods = 5

    times = pd.date_range("2000-01-01", periods=periods, freq="D")
    data = np.empty((periods, size, size), dtype=np.int32)
    for i in range(periods):
        data[i][:] = i
    ds = xr.Dataset(
        {"temperature": (("time", "x", "y"), data)},
        coords={"time": times, "x": np.arange(0, size), "y": np.arange(0, size)},
    )

    ds.to_netcdf(filename)
    return filename


def test_read_timestep(netcdf_file):
    loop = asyncio.new_event_loop()

    if not loop.is_running():
        loop_thread = threading.Thread(target=loop.run_forever)
        loop_thread.start()

    reader = AsyncXarrayReader(netcdf_file, variable_name="temperature", loop=loop)

    data0 = reader.read_timestep(date(2000, 1, 1))

    sleep(3)  # Simulate some processing time

    t0 = time()
    data1 = reader.read_timestep(date(2000, 1, 2))
    t1 = time()
    print("Load next timestep (quick): {:.2f}s".format(t1 - t0))

    # wait half the time it took to load the previous timestep to simulate a short
    # processing time
    sleep((t1 - t0) / 2)

    t0 = time()
    data2 = reader.read_timestep(date(2000, 1, 3))
    t1 = time()
    print("Load next timestep short waiting (semi-quick): {:.2f}s".format(t1 - t0))

    assert (data0 == 0).all()
    assert (data1 == 1).all()
    assert (data2 == 2).all()
    assert data0.dtype == np.int32

    sleep(3)

    t0 = time()
    data3 = reader.read_timestep(date(2000, 1, 4))
    t1 = time()
    print("Load next timestep with waiting (quick): {:.2f}s".format(t1 - t0))

    t0 = time()
    data3 = reader.read_timestep(date(2000, 1, 4))
    t1 = time()
    print("Load same timestep (quick): {:.2f}s".format(t1 - t0))
    assert (data3 == 3).all()

    t0 = time()
    data0 = reader.read_timestep(date(2000, 1, 1))
    t1 = time()
    print("Load previous timestep (slow): {:.2f}s".format(t1 - t0))
    assert (data0 == 0).all()

    reader.close()
    loop.call_soon_threadsafe(loop.stop)
    # Wait for the loop thread to finish
    loop_thread.join()
    loop.close()
    netcdf_file.unlink()
