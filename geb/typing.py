"""Typing definitions for GEB."""

import numpy as np

ArrayFloat32 = np.ndarray[tuple[int], np.dtype[np.float32]]
ArrayFloat64 = np.ndarray[tuple[int], np.dtype[np.float64]]
TwoDFloatArrayFloat32 = np.ndarray[tuple[int, int], np.dtype[np.float32]]
TwoDFloatArrayFloat64 = np.ndarray[tuple[int, int], np.dtype[np.float64]]
