"""GEB simulates the environment, the individual behaviour of people, households and organizations - including their interactions - at small and large scale."""

__version__ = "1.0.0b4"

import os
import platform
import sys
from pathlib import Path
from numba import config
import faulthandler


if platform.system() != "Windows":
    # Modify LD_LIBRARY_PATH on Unix-like systems (Linux, macOS)
    import tbb  # noqa: F401

    tbb_path = Path(sys.prefix) / "lib" / "libtbb.so"
    assert tbb_path.exists(), f"tbb shared library not found at {tbb_path}"
    os.environ["LD_LIBRARY_PATH"] = str(tbb_path)

# set threading layer to tbb, this is much faster than other threading layers
config.THREADING_LAYER = "tbb"

# set environment variable for GEB package directory
os.environ["GEB_PACKAGE_DIR"] = str(Path(__file__).parent)

faulthandler.enable()
