import pytest
import numpy as np
import pandas as pd
from datetime import date
import xarray as xr
from time import time, sleep
from geb.workflows import (
    AsyncXarrayReader,
)


@pytest.fixture
def netcdf_file(tmp_path):
    size = 10_000
    # Create a temporary NetCDF file for testing
    filename = tmp_path / "test_data.nc"

    periods = 5

    times = pd.date_range("2000-01-01", periods=periods, freq="D")
    data = np.empty((periods, size, size), dtype=np.int32)
    for i in range(periods):
        data[i][:] = i
    ds = xr.Dataset(
        {
            "temperature": (("time", "x", "y"), data),
            "precipitation": (("time", "x", "y"), data),
            "pressure": (("time", "x", "y"), data),
        },
        coords={"time": times, "x": np.arange(0, size), "y": np.arange(0, size)},
    )

    ds.to_netcdf(filename)
    return filename


def test_read_timestep(netcdf_file):
    reader1 = AsyncXarrayReader(netcdf_file, variable_name="temperature")
    reader2 = AsyncXarrayReader(netcdf_file, variable_name="precipitation")
    reader3 = AsyncXarrayReader(netcdf_file, variable_name="pressure")

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

    data0_2 = reader2.read_timestep(date(2000, 1, 1))
    data0_3 = reader3.read_timestep(date(2000, 1, 1))

    sleep(3)  # Simulate some processing time
    print("-----------------")

    t0 = time()
    data1 = reader1.read_timestep(date(2000, 1, 2))
    data1_2 = reader2.read_timestep(date(2000, 1, 2))
    data1_3 = reader3.read_timestep(date(2000, 1, 2))
    t1 = time()
    print("Load data from three readers (quick): {:.3f}s".format(t1 - t0))

    reader1.close()
    reader2.close()

    netcdf_file.unlink()
