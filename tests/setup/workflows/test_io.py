import xarray as xr
import numpy as np

# from numcodecs import PackBits
# from zarr.codecs import PackBits
from geb.setup.workflows.io import to_zarr
from ...testconfig import tmp_folder


def test_io():
    x = np.linspace(-5, 5, 10)
    y = np.linspace(0, 10, 10)

    values = np.random.rand(x.size, y.size)
    da = xr.DataArray(values, coords={"x": x, "y": y}, dims=["y", "x"])

    to_zarr(da, tmp_folder / "test.zarr", crs=4326)

    time_dimension = np.linspace(0, 10, 10)
    values = np.random.rand(x.size, y.size, time_dimension.size)
    da = xr.DataArray(
        values, coords={"x": x, "y": y, "time": time_dimension}, dims=["y", "x", "time"]
    )

    to_zarr(da, tmp_folder / "test.zarr", crs=4326)

    other_dimension = np.linspace(110, 115, 10)
    values = np.random.rand(x.size, y.size, other_dimension.size)

    da = xr.DataArray(
        values,
        coords={"x": x, "y": y, "other": other_dimension},
        dims=["other", "y", "x"],
    )

    to_zarr(da, tmp_folder / "test.zarr", crs=4326)


# def test_filters():
#     x = np.linspace(-5, 5, 10)
#     y = np.linspace(0, 10, 10)
#     time = np.linspace(0, 10, 10)

#     values = np.random.choice([True, False], size=(x.size, y.size, time.size))
#     da = xr.DataArray(
#         values, coords={"x": x, "y": y, "time": time}, dims=["y", "x", "time"]
#     )

#     to_zarr(da, tmp_folder / "test.zarr", crs=4326, filters=[PackBits()])
