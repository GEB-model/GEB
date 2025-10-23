"""Typing definitions for GEB."""

import numpy as np

ArrayFloat32 = np.ndarray[tuple[int], np.dtype[np.float32]]
ArrayFloat64 = np.ndarray[tuple[int], np.dtype[np.float64]]
ArrayInt32 = np.ndarray[tuple[int], np.dtype[np.int32]]
ArrayInt64 = np.ndarray[tuple[int], np.dtype[np.int64]]
ArrayBool = np.ndarray[tuple[int], np.dtype[np.bool_]]

TwoDFloatArrayFloat32 = np.ndarray[tuple[int, int], np.dtype[np.float32]]
TwoDFloatArrayFloat64 = np.ndarray[tuple[int, int], np.dtype[np.float64]]
TwoDIntArrayInt32 = np.ndarray[tuple[int, int], np.dtype[np.int32]]
TwoDIntArrayInt64 = np.ndarray[tuple[int, int], np.dtype[np.int64]]
TwoDBoolArray = np.ndarray[tuple[int, int], np.dtype[np.bool_]]

ThreeDIntArrayInt32 = np.ndarray[tuple[int, int, int], np.dtype[np.int32]]
ThreeDIntArrayInt64 = np.ndarray[tuple[int, int, int], np.dtype[np.int64]]
ThreeDFloatArrayFloat32 = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
ThreeDFloatArrayFloat64 = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
ThreeDBoolArray = np.ndarray[tuple[int, int, int], np.dtype[np.bool_]]
