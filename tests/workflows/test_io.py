import numpy as np
import pytest
import xarray as xr
from numcodecs import FixedScaleOffset

from geb.workflows.io import calculate_scaling, get_window, to_zarr

from ..testconfig import tmp_folder


def test_calculate_scaling() -> None:
    def encode_decode(data, min_value, max_value, offset, precision=0.1):
        assert data.dtype == np.float32

        scaling_factor, out_dtype = calculate_scaling(
            min_value=min_value,
            max_value=max_value,
            precision=precision,
            offset=offset,
        )
        codec = FixedScaleOffset(
            offset=offset, scale=scaling_factor, dtype=np.float32, astype=out_dtype
        )

        encoded_data = codec.encode(data)

        decoded_data = codec.decode(encoded_data)

        diff = data - decoded_data

        assert np.all(np.abs(diff) <= precision * 1.05), (
            f"max diff: {diff.max()}; min diff: {diff.min()}"
        )
        return decoded_data, scaling_factor, out_dtype

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
            min_value=0,
            max_value=1e10,
            precision=1e-12,
            offset=0,
        )

    with pytest.raises(
        ValueError, match="Too many bits required for precision and range"
    ):
        calculate_scaling(
            min_value=-1e10,
            max_value=0,
            precision=1e-12,
            offset=0,
        )


def test_io() -> None:
    x = np.linspace(-5, 5, 10)
    y = np.linspace(10, 0, 10)

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


def test_get_window() -> None:
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
        window = get_window(da.x, da.y, bounds, buffer=0.1)
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
