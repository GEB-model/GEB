import numpy as np
import pytest
import xarray as xr
from numcodecs import PackBits

from geb.workflows.io import to_zarr

from ..testconfig import tmp_folder


def test_io():
    x = np.linspace(-5, 5, 10)
    y = np.linspace(0, 10, 10)

    values = np.random.rand(x.size, y.size).astype(np.float32)
    da = xr.DataArray(values, coords={"x": x, "y": y}, dims=["y", "x"]).chunk()
    da.attrs["_FillValue"] = np.nan

    to_zarr(da, tmp_folder / "test.zarr", crs=4326)

    time_dimension = np.linspace(0, 10, 10)
    values = np.random.rand(x.size, y.size, time_dimension.size).astype(np.float32)
    da = xr.DataArray(
        values, coords={"x": x, "y": y, "time": time_dimension}, dims=["time", "y", "x"]
    ).chunk()
    da.attrs["_FillValue"] = np.nan

    to_zarr(da, tmp_folder / "test.zarr", crs=4326)

    other_dimension = np.linspace(110, 115, 10)
    values = np.random.rand(x.size, y.size, other_dimension.size).astype(np.float32)

    da = xr.DataArray(
        values,
        coords={"x": x, "y": y, "other": other_dimension},
        dims=["other", "y", "x"],
    ).chunk()
    da.attrs["_FillValue"] = np.nan

    to_zarr(da, tmp_folder / "test.zarr", crs=4326)


@pytest.mark.skip(reason="Skip test until fixes are included in xarray")
def test_filters():
    x = np.linspace(-5, 5, 10)
    y = np.linspace(0, 10, 10)
    time_dimension = np.linspace(0, 10, 10)

    values = np.random.rand(x.size, y.size, time_dimension.size).astype(np.float32)
    da = xr.DataArray(
        values, coords={"x": x, "y": y, "time": time_dimension}, dims=["time", "y", "x"]
    ).chunk()
    da.attrs["_FillValue"] = np.nan

    to_zarr(da, tmp_folder / "test.zarr", crs=4326, filters=[PackBits()])
