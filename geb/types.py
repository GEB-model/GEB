"""Typing definitions for GEB."""

from typing import Any, TypeVar

import numpy as np
from numpy.typing import DTypeLike

Shape = TypeVar("Shape", bound=tuple[int, ...])


ArrayFloat16 = np.ndarray[tuple[int], np.dtype[np.float16]]
ArrayFloat32 = np.ndarray[tuple[int], np.dtype[np.float32]]
ArrayFloat64 = np.ndarray[tuple[int], np.dtype[np.float64]]
ArrayFloat = ArrayFloat16 | ArrayFloat32 | ArrayFloat64

ArrayInt8 = np.ndarray[tuple[int], np.dtype[np.int8]]
ArrayUint8 = np.ndarray[tuple[int], np.dtype[np.uint8]]
ArrayInt16 = np.ndarray[tuple[int], np.dtype[np.int16]]
ArrayUint16 = np.ndarray[tuple[int], np.dtype[np.uint16]]
ArrayInt32 = np.ndarray[tuple[int], np.dtype[np.int32]]
ArrayUint32 = np.ndarray[tuple[int], np.dtype[np.uint32]]
ArrayInt64 = np.ndarray[tuple[int], np.dtype[np.int64]]
ArrayUint64 = np.ndarray[tuple[int], np.dtype[np.uint64]]

ArrayInt = (
    ArrayInt8
    | ArrayUint8
    | ArrayInt16
    | ArrayUint16
    | ArrayInt32
    | ArrayUint32
    | ArrayInt64
    | ArrayUint64
)

ArrayBool = np.ndarray[tuple[int], np.dtype[np.bool_]]
ArrayDatetime64 = np.ndarray[tuple[int], np.dtype[np.datetime64]]

Array = np.ndarray[tuple[int], Any]  # General array type

TwoDArrayFloat16 = np.ndarray[tuple[int, int], np.dtype[np.float16]]
TwoDArrayFloat32 = np.ndarray[tuple[int, int], np.dtype[np.float32]]
TwoDArrayFloat64 = np.ndarray[tuple[int, int], np.dtype[np.float64]]
TwoDArrayFloat = TwoDArrayFloat16 | TwoDArrayFloat32 | TwoDArrayFloat64

TwoDArrayInt8 = np.ndarray[tuple[int, int], np.dtype[np.int8]]
TwoDArrayInt16 = np.ndarray[tuple[int, int], np.dtype[np.int16]]
TwoDArrayInt32 = np.ndarray[tuple[int, int], np.dtype[np.int32]]
TwoDArrayInt64 = np.ndarray[tuple[int, int], np.dtype[np.int64]]
TwoDArrayUint8 = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
TwoDArrayUint16 = np.ndarray[tuple[int, int], np.dtype[np.uint16]]
TwoDArrayUint32 = np.ndarray[tuple[int, int], np.dtype[np.uint32]]
TwoDArrayUint64 = np.ndarray[tuple[int, int], np.dtype[np.uint64]]
TwoDArrayInt = (
    TwoDArrayInt8
    | TwoDArrayInt16
    | TwoDArrayInt32
    | TwoDArrayInt64
    | TwoDArrayUint8
    | TwoDArrayUint16
    | TwoDArrayUint32
    | TwoDArrayUint64
)

TwoDArrayBool = np.ndarray[tuple[int, int], np.dtype[np.bool_]]

TwoDArray = np.ndarray[tuple[int, int], Any]  # General 2D array type

ThreeDArrayFloat16 = np.ndarray[tuple[int, int, int], np.dtype[np.float16]]
ThreeDArrayFloat32 = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
ThreeDArrayFloat64 = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
ThreeDArrayFloat = ThreeDArrayFloat16 | ThreeDArrayFloat32 | ThreeDArrayFloat64

ThreeArrayInt8 = np.ndarray[tuple[int, int, int], np.dtype[np.int8]]
ThreeArrayInt16 = np.ndarray[tuple[int, int, int], np.dtype[np.int16]]
ThreeArrayInt32 = np.ndarray[tuple[int, int, int], np.dtype[np.int32]]
ThreeArrayInt64 = np.ndarray[tuple[int, int, int], np.dtype[np.int64]]
ThreeDArrayUint8 = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
ThreeDArrayUint16 = np.ndarray[tuple[int, int, int], np.dtype[np.uint16]]
ThreeDArrayUint32 = np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]
ThreeDArrayUint64 = np.ndarray[tuple[int, int, int], np.dtype[np.uint64]]
ThreeDArrayInt = (
    ThreeArrayInt8
    | ThreeArrayInt16
    | ThreeArrayInt32
    | ThreeArrayInt64
    | ThreeDArrayUint8
    | ThreeDArrayUint16
    | ThreeDArrayUint32
    | ThreeDArrayUint64
)

ThreeDArrayBool = np.ndarray[tuple[int, int, int], np.dtype[np.bool_]]

ThreeDArray = np.ndarray[tuple[int, int, int], Any]  # General 3D array type


T_Array = TypeVar("T_Array", bound=Array)
T_TwoDArray = TypeVar("T_TwoDArray", bound=TwoDArray)
T_ThreeDArray = TypeVar("T_ThreeDArray", bound=ThreeDArray)

DType = TypeVar("DType", bound=DTypeLike)
T_OneorTwoDArray = TypeVar("T_OneorTwoDArray", bound=Array | TwoDArray)
T_Scalar = TypeVar("T_Scalar", bound=np.generic)
T_ArrayNumber = TypeVar("T_ArrayNumber", bound=ArrayFloat | ArrayInt)

# Aliases for compress overloads using T_Scalar
ArrayWithScalar = np.ndarray[tuple[int], np.dtype[T_Scalar]]
TwoDArrayWithScalar = np.ndarray[tuple[int, int], np.dtype[T_Scalar]]
ThreeDArrayWithScalar = np.ndarray[tuple[int, int, int], np.dtype[T_Scalar]]
AnyDArrayWithScalar = np.ndarray[Shape, np.dtype[T_Scalar]]
