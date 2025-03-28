import numpy as np
import pytest
import xarray as xr
from numcodecs import FixedScaleOffset

from geb.workflows.io import calculate_scaling, to_zarr

from ..testconfig import tmp_folder


def test_calculate_scaling():
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
