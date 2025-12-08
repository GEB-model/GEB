"""GEB simulates the environment, the individual behaviour of people, households and organizations - including their interactions - at small and large scale."""

import faulthandler
import os
import platform
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from dotenv import load_dotenv
from llvmlite import binding
from numba import config, njit, prange, threading_layer

from geb.workflows.io import fetch_and_save

__version__: str = version("GEB")

# set environment variable for GEB package directory
GEB_PACKAGE_DIR = cast(Path, files("geb"))
os.environ["GEB_PACKAGE_DIR"] = str(files("geb"))

# Load environment variables from .env file
load_dotenv()


# Auto-detect whether we are on the Ada HPC cluster of the Vrije Universiteit Amsterdam. If so, set some environment variables accordingly.
if Path("/research/BETA-IVM-HPC/GEB").exists():
    os.environ["GEB_DATA_ROOT"] = "/research/BETA-IVM-HPC/GEB/data_catalog/"
    os.environ["SFINCS_CONTAINER"] = os.getenv(
        "SFINCS_CONTAINER",
        "/ada-software/containers/sfincs-cpu-v2.2.0-col-dEze-Release.sif",
    )
    os.environ["SFINCS_CONTAINER_GPU"] = os.getenv(
        "SFINCS_CONTAINER_GPU",
        "/ada-software/containers/sfincs-gpu.coldeze_combo_ccall.sif",
    )
else:
    os.environ["SFINCS_SIF_CONTAINER"] = os.getenv(
        "SFINCS_SIF_CONTAINER", "deltares/sfincs-cpu:sfincs-v2.2.0-col-dEze-Release"
    )
    os.environ["SFINCS_SIF_CONTAINER_GPU"] = os.getenv(
        "SFINCS_SIF_CONTAINER_GPU", "mvanormondt/sfincs-gpu:coldeze_combo_ccall"
    )


def load_numba_threading_layer(version: str = "2022.1.0") -> None:
    """Load TBB shared library, a very efficient threading layer for parallelizing CPU-bound tasks in Numba-compiled functions.

    Args:
        version: version of TBB to use, default is "2022.1.0".

    Raises:
        RuntimeError: If the platform is not supported.

    """
    version = "2022.1.0"
    bin_path: Path = GEB_PACKAGE_DIR / "bin" / "tbb"
    tbb_uncompressed_folder: Path = Path("oneapi-tbb-" + version)

    if platform.system() == "Linux":
        tbb_platform: str = "lin"
        tbb_file: Path = Path("lib") / "intel64" / "gcc4.8" / "libtbb.so.12"
        tbb_compressed_file: str = f"{tbb_platform}.tgz"
    elif platform.system() == "Windows":
        tbb_platform: str = "win"
        tbb_file: Path = Path("redist") / "intel64" / "vc14" / "tbb12.dll"
        tbb_compressed_file: str = f"{tbb_platform}.zip"
    elif platform.system() == "Darwin":
        tbb_platform: str = "mac"
        tbb_file: Path = Path("lib") / "libtbb.12.dylib"
        tbb_compressed_file: str = f"{tbb_platform}.tgz"
    else:
        raise RuntimeError(f"Unsupported platform: {platform.system()}")

    tbb_path: Path = bin_path / tbb_platform
    tbb_library: Path = tbb_path / tbb_uncompressed_folder / tbb_file

    if not tbb_library.exists():
        tbb_url: str = f"https://github.com/uxlfoundation/oneTBB/releases/download/v{version}/oneapi-tbb-{version}-{tbb_compressed_file}"

        tbb_path.mkdir(parents=True, exist_ok=True)

        # download the TBB library
        print(
            f"Downloading TBB library from {tbb_url} to {tbb_path / tbb_compressed_file}"
        )
        fetch_and_save(
            url=tbb_url,
            file_path=tbb_path / tbb_compressed_file,
        )

        # uncompress the TBB library
        if tbb_compressed_file.endswith(".tgz"):
            import tarfile

            with tarfile.open(tbb_path / tbb_compressed_file, "r:gz") as tar:
                tar.extractall(path=tbb_path)
        elif tbb_compressed_file.endswith(".zip"):
            import zipfile

            with zipfile.ZipFile(tbb_path / tbb_compressed_file, "r") as zip_ref:
                zip_ref.extractall(path=tbb_path)

        (
            tbb_path / tbb_compressed_file
        ).unlink()  # remove compressed file after extraction

        assert tbb_path.exists(), f"tbb shared library not found at {tbb_path}"

    binding.load_library_permanently(str(tbb_library))

    # set threading layer
    config.THREADING_LAYER = "tbb"

    # test import
    from numba.np.ufunc import tbbpool  # noqa: F401 # ty: ignore[unresolved-import]
    from numba.np.ufunc.parallel import _check_tbb_version_compatible

    _check_tbb_version_compatible()

    @njit(parallel=True)
    def test_threading_layer() -> npt.NDArray[np.int32]:
        array = np.zeros(10, dtype=np.int32)
        """Test function to check if TBB is loaded correctly."""
        for i in prange(10):  # ty: ignore[not-iterable]
            array[i] = i
        return array

    test_threading_layer()

    assert threading_layer() == "tbb", (
        f"Expected threading layer to be 'tbb', but got {threading_layer()}"
    )


if __debug__:
    import numba

    # By default, instead of causing an IndexError, accessing an out-of-bound index
    # of an array in a Numba-compiled function will return invalid values or lead
    # to an access violation error (it’s reading from invalid memory locations).
    # Setting BOUNDSCHECK to 1 will enable bounds checking for all array accesses
    numba.config.BOUNDSCHECK = 1

os.environ["NUMBA_ENABLE_AVX"] = "0"  # Enable AVX instructions
# os.environ["NUMBA_PARALLEL_DIAGNOSTICS"] = "4"

if platform.system() == "Darwin":
    print(
        "On Mac OS X, we disable the multi-threading layer by default due to compatibility issues."
    )
    os.environ["NUMBA_NUM_THREADS"] = "1"
else:
    load_numba_threading_layer()

# xarray uses bottleneck for some operations to speed up computations
# however, some implementations are numerically unstable, so we disable it
xr.set_options(use_bottleneck=False, keep_attrs=True)

faulthandler.enable()
