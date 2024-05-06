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

    periods = 4

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
    reader = AsyncXarrayReader(netcdf_file, variable_name="temperature")

    data0 = reader.read_timestep(date(2000, 1, 1))

    sleep(3)  # Simulate some processing time

    t0 = time()
    data1 = reader.read_timestep(date(2000, 1, 2))
    t1 = time()
    print("Load next timestep (quick): {:.2f}s".format(t1 - t0))

    assert (data0 == 0).all()
    assert data0.dtype == np.int32
    assert (data1 == 1).all()

    t0 = time()
    data2 = reader.read_timestep(date(2000, 1, 2))
    t1 = time()
    print("Load same timestep (quick): {:.2f}s".format(t1 - t0))
    assert (data2 == 1).all()

    t0 = time()
    data3 = reader.read_timestep(date(2000, 1, 1))
    t1 = time()
    print("Load previous timestep (slow): {:.2f}s".format(t1 - t0))
    assert data3.dtype == np.int32
    assert (data3 == 0).all()

    reader.close()
    netcdf_file.unlink()
