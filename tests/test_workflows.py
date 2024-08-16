import numpy as np
import pandas as pd
from datetime import date
import xarray as xr
from time import time, sleep
from geb.workflows import (
    AsyncXarrayReader,
)

from .setup import tmp_folder


def netcdf_file(varname):
    size = 100
    # Create a temporary NetCDF file for testing
    filename = tmp_folder / f"{varname}.nc"

    periods = 5

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

    ds.to_netcdf(filename)
    return filename


def test_read_timestep():
    temperature_file = netcdf_file("temperature")
    precipitation_file = netcdf_file("precipitation")
    pressure_file = netcdf_file("pressure")
    reader1 = AsyncXarrayReader(temperature_file, variable_name="temperature")
    reader2 = AsyncXarrayReader(precipitation_file, variable_name="precipitation")
    reader3 = AsyncXarrayReader(pressure_file, variable_name="pressure")

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
    reader3.close()

    temperature_file.unlink()
    precipitation_file.unlink()
    pressure_file.unlink()
