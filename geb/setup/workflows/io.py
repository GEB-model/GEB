from pathlib import Path
import xarray as xr
import numpy as np
from numcodecs import Blosc


def open_zarr(zarr_folder):
    da = xr.open_dataset(zarr_folder, engine="zarr", chunks={})
    assert len(da.data_vars) == 1, "Only one data variable is supported"
    da = da[list(da.data_vars)[0]]
    return da


def to_zarr(
    da,
    zarr_folder,
    x_chunksize=350,
    y_chunksize=350,
    time_chunksize=1,
    byteshuffle=True,
):
    zarr_folder = Path(zarr_folder)
    zarr_folder.parent.mkdir(parents=True, exist_ok=True)

    da.name = da.name.split("/")[-1]

    if "latitudes" in da.dims:
        da = da.rename({"latitudes": "y"})
    if "longitudes" in da.dims:
        da = da.rename({"longitudes": "x"})
    if da.dtype == np.float64:
        da = da.astype(np.float32)

    da = da.drop_vars([v for v in da.coords if v not in da.dims])

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

    else:
        if "y" in da.dims and "x" in da.dims:
            assert da.dims[0] == "y" and da.dims[1] == "x", (
                "y and x dimensions must be first and second, otherwise xarray will not chunk correctly"
            )
            chunksizes = {
                "y": y_chunksize,
                "x": x_chunksize,
            }
        else:
            chunksizes = {}

    da = da.chunk(chunksizes)

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

    store = da.to_zarr(
        zarr_folder,
        mode="w",
        encoding=encoding,
        compute=True,
        zarr_version=2,
        consolidated=False,
    )

    store.close()
    # with tempfile.TemporaryDirectory() as tmp_dir:
    #     tmp_file = Path(tmp_dir) / "data.zarr.zip"
    #     store = zarr.storage.ZipStore(tmp_file, mode="w")
    #     if "latitudes" in da.dims:
    #         da = da.rename({"latitudes": "y"})
    #     if "longitudes" in da.dims:
    #         da = da.rename({"longitudes": "x"})
    #     if da.dtype == np.float64:
    #         da = da.astype(np.float32)

    #     if "spatial_ref" in da.coords:
    #         da = da.drop_vars("spatial_ref")

    #     if "time" in da.dims:
    #         if "y" in da.dims and "x" in da.dims:
    #             assert da.dims[1] == "y" and da.dims[2] == "x", (
    #                 "y and x dimensions must be second and third, otherwise xarray will not chunk correctly"
    #             )
    #             chunksizes = {
    #                 "time": time_chunksize,
    #                 "y": y_chunksize,
    #                 "x": x_chunksize,
    #             }
    #         else:
    #             chunksizes = {"time": time_chunksize}

    #     else:
    #         if "y" in da.dims and "x" in da.dims:
    #             assert da.dims[0] == "y" and da.dims[1] == "x", (
    #                 "y and x dimensions must be first and second, otherwise xarray will not chunk correctly"
    #             )
    #             chunksizes = {
    #                 "y": y_chunksize,
    #                 "x": x_chunksize,
    #             }
    #         else:
    #             chunksizes = {}

    #     encoding = {
    #         da.name: {
    #             "compressor": Blosc(
    #                 cname="zstd",
    #                 clevel=9,
    #                 shuffle=Blosc.SHUFFLE if byteshuffle else Blosc.NOSHUFFLE,
    #             ),
    #             "chunks": tuple(
    #                 (
    #                     chunksizes[dim]
    #                     if dim in chunksizes
    #                     else max(getattr(da, dim).size, 1)
    #                 )
    #                 for dim in da.dims
    #             ),
    #         }
    #     }

    #     da.to_zarr(
    #         store,
    #         mode="w",
    #         encoding=encoding,
    #         zarr_version=2,
    #     )

    #     store.close()
    #     # move file to final location
    #     shutil.copy(tmp_file, zarr_folder)

    return open_zarr(zarr_folder)
