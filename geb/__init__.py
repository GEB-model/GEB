"""GEB simulates the environment, the individual behaviour of people, households and organizations - including their interactions - at small and large scale."""

__version__ = "1.0.0b4"

import os
import platform
import sys
from pathlib import Path
from llvmlite import binding
from numba import config
import faulthandler
import xarray as xr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __debug__:
    import numba

    # By default, instead of causing an IndexError, accessing an out-of-bound index
    # of an array in a Numba-compiled function will return invalid values or lead
    # to an access violation error (it’s reading from invalid memory locations).
    # Setting BOUNDSCHECK to 1 will enable bounds checking for all array accesses
    numba.config.BOUNDSCHECK = 1

os.environ["NUMBA_ENABLE_AVX"] = "0"  # Enable AVX instructions
# os.environ["NUMBA_PARALLEL_DIAGNOSTICS"] = "4"


if platform.system() != "Windows":
    # Modify LD_LIBRARY_PATH on Unix-like systems (Linux, macOS)
    tbb_path = Path(sys.prefix) / "lib" / "libtbb.so.12"
    assert tbb_path.exists(), f"tbb shared library not found at {tbb_path}"

    binding.load_library_permanently(str(tbb_path))

    from numba.np.ufunc import tbbpool  # noqa: F401
else:
    # test if import works
    from numba.np.ufunc import tbbpool  # noqa: F401

# set threading layer to tbb, this is much faster than other threading layers
config.THREADING_LAYER = "tbb"

# xarray uses bottleneck for some operations to speed up computations
# however, some implementations are numerically unstable, so we disable it
xr.set_options(use_bottleneck=False)

# set environment variable for GEB package directory
os.environ["GEB_PACKAGE_DIR"] = str(Path(__file__).parent)

faulthandler.enable()
