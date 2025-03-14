import zarr
import tempfile
from pathlib import Path
import xarray as xr
import shutil
import numpy as np
from numcodecs import Blosc


def open_zarr(file_path):
    store = zarr.storage.ZipStore(file_path, mode="r")
    da = xr.open_dataset(store, engine="zarr", chunks={})
    if len(da.data_vars) == 1:
        da = da[list(da.data_vars)[0]]
    return da


def to_zarr(
    da, file_path, x_chunksize=350, y_chunksize=350, time_chunksize=1, byteshuffle=True
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "data.zarr.zip"
        store = zarr.storage.ZipStore(tmp_file, mode="w")
        if "latitudes" in da.dims:
            da = da.rename({"latitudes": "y"})
        if "longitudes" in da.dims:
            da = da.rename({"longitudes": "x"})
        if da.dtype == np.float64:
            da = da.astype(np.float32)

        if "time" in da.dims:
            if "y" in da.dims and "x" in da.dims:
                assert da.dims[1] == "y" and da.dims[2] == "x", (
                    "y and x dimensions must be second and third, otherwise xarray will not chunk correctly"
                )
                chunksizes = {
                    "time": time_chunksize,
                    "y": y_chunksize,
                    "x": x_chunksize,
                }
            else:
                chunksizes = {"time": time_chunksize}

            encoding = {
                da.name: {
                    "compressor": Blosc(
                        cname="zstd",
                        clevel=9,
                        shuffle=Blosc.SHUFFLE if byteshuffle else Blosc.NOSHUFFLE,
                    ),
                    "chunks": tuple(
                        (
                            chunksizes[dim]
                            if dim in chunksizes
                            else max(getattr(da, dim).size, 1)
                        )
                        for dim in da.dims
                    ),
                }
            }

            da.chunk(chunksizes).to_zarr(
                store,
                mode="w",
                encoding=encoding,
                zarr_version=2,
            )
        else:
            raise NotImplementedError("Only time dimension is supported")

        store.close()
        # move file to final location
        shutil.copy(tmp_file, file_path)

    return open_zarr(file_path)
