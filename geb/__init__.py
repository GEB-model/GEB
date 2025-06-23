"""GEB simulates the environment, the individual behaviour of people, households and organizations - including their interactions - at small and large scale."""

__version__ = "1.0.0b5"

import faulthandler
import os
import platform
import sys
from pathlib import Path

import xarray as xr
from dotenv import load_dotenv
from llvmlite import binding
from numba import config

from geb.workflows.io import fetch_and_save

# Load environment variables from .env file
load_dotenv()

# set environment variable for GEB package directory
os.environ["GEB_PACKAGE_DIR"] = str(Path(__file__).parent)


def load_numba_threading_layer(version: str = "2022.1.0") -> None:
    """Load TBB shared library, a very efficient threading layer for parallelizing CPU-bound tasks in Numba-compiled functions.

    Args:
        version: version of TBB to use, default is "2022.1.0".

    """
    bin_path: Path = Path(os.environ.get("GEB_PACKAGE_DIR")) / "bin" / "tbb"
    tbb_uncompressed_folder: Path = Path("oneapi-tbb-" + version)

    if platform.system() == "Linux":
        tbb_platform: str = "lin"
        tbb_file: Path = Path("intel64") / "gcc4.8" / "libtbb.so.12"
        tbb_compressed_file: str = f"{tbb_platform}.tgz"
    elif platform.system() == "Windows":
        tbb_platform: str = "win"
        tbb_file: Path = Path("intel64") / "vc14" / "tbb12.dll"
        tbb_compressed_file: str = f"{tbb_platform}.zip"
    elif platform.system() == "Darwin":
        tbb_platform: str = "mac"
        tbb_file: Path = Path("lib") / "libtbb.12.dylib"
        tbb_compressed_file: str = f"{tbb_platform}.tgz"
    else:
        raise RuntimeError(f"Unsupported platform: {platform.system()}")

    tbb_path: Path = bin_path / tbb_platform
    tbb_library: Path = tbb_path / tbb_uncompressed_folder / "lib" / tbb_file

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

    # test import
    from numba.np.ufunc import tbbpool  # noqa: F401

    # set threading layer
    config.THREADING_LAYER = "tbb"


if __debug__:
    import numba

    # By default, instead of causing an IndexError, accessing an out-of-bound index
    # of an array in a Numba-compiled function will return invalid values or lead
    # to an access violation error (it’s reading from invalid memory locations).
    # Setting BOUNDSCHECK to 1 will enable bounds checking for all array accesses
    numba.config.BOUNDSCHECK = 1

os.environ["NUMBA_ENABLE_AVX"] = "0"  # Enable AVX instructions
# os.environ["NUMBA_PARALLEL_DIAGNOSTICS"] = "4"

load_numba_threading_layer()

# xarray uses bottleneck for some operations to speed up computations
# however, some implementations are numerically unstable, so we disable it
xr.set_options(use_bottleneck=False, keep_attrs=True)

faulthandler.enable()
