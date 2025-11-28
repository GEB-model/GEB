"""Typing definitions for GEB."""

import numpy as np

type ArrayFloat16 = np.ndarray[tuple[int], np.dtype[np.float16]]
type ArrayFloat32 = np.ndarray[tuple[int], np.dtype[np.float32]]
type ArrayFloat64 = np.ndarray[tuple[int], np.dtype[np.float64]]
type ArrayFloat = ArrayFloat16 | ArrayFloat32 | ArrayFloat64

type ArrayInt8 = np.ndarray[tuple[int], np.dtype[np.int8]]
type ArrayUint8 = np.ndarray[tuple[int], np.dtype[np.uint8]]
type ArrayInt16 = np.ndarray[tuple[int], np.dtype[np.int16]]
type ArrayUint16 = np.ndarray[tuple[int], np.dtype[np.uint16]]
type ArrayInt32 = np.ndarray[tuple[int], np.dtype[np.int32]]
type ArrayUint32 = np.ndarray[tuple[int], np.dtype[np.uint32]]
type ArrayInt64 = np.ndarray[tuple[int], np.dtype[np.int64]]
type ArrayUint64 = np.ndarray[tuple[int], np.dtype[np.uint64]]

type ArrayInt = (
    ArrayInt8
    | ArrayUint8
    | ArrayInt16
    | ArrayUint16
    | ArrayInt32
    | ArrayUint32
    | ArrayInt64
    | ArrayUint64
)

type ArrayBool = np.ndarray[tuple[int], np.dtype[np.bool_]]
type ArrayDatetime64 = np.ndarray[tuple[int], np.dtype[np.datetime64]]

type Array = ArrayFloat | ArrayInt | ArrayBool | ArrayDatetime64

type TwoDArrayFloat16 = np.ndarray[tuple[int, int], np.dtype[np.float16]]
type TwoDArrayFloat32 = np.ndarray[tuple[int, int], np.dtype[np.float32]]
type TwoDArrayFloat64 = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type TwoDArrayFloat = TwoDArrayFloat16 | TwoDArrayFloat32 | TwoDArrayFloat64

type TwoDArrayInt8 = np.ndarray[tuple[int, int], np.dtype[np.int8]]
type TwoDArrayInt16 = np.ndarray[tuple[int, int], np.dtype[np.int16]]
type TwoDArrayInt32 = np.ndarray[tuple[int, int], np.dtype[np.int32]]
type TwoDArrayInt64 = np.ndarray[tuple[int, int], np.dtype[np.int64]]
type TwoDArrayUint8 = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
type TwoDArrayUint16 = np.ndarray[tuple[int, int], np.dtype[np.uint16]]
type TwoDArrayUint32 = np.ndarray[tuple[int, int], np.dtype[np.uint32]]
type TwoDArrayUint64 = np.ndarray[tuple[int, int], np.dtype[np.uint64]]
type TwoDArrayInt = (
    TwoDArrayInt8
    | TwoDArrayInt16
    | TwoDArrayInt32
    | TwoDArrayInt64
    | TwoDArrayUint8
    | TwoDArrayUint16
    | TwoDArrayUint32
    | TwoDArrayUint64
)

type TwoDArrayBool = np.ndarray[tuple[int, int], np.dtype[np.bool_]]

type TwoDArray = TwoDArrayFloat | TwoDArrayInt | TwoDArrayBool

type ThreeDArrayFloat16 = np.ndarray[tuple[int, int, int], np.dtype[np.float16]]
type ThreeDArrayFloat32 = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
type ThreeDArrayFloat64 = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
type ThreeDArrayFloat = ThreeDArrayFloat16 | ThreeDArrayFloat32 | ThreeDArrayFloat64

type ThreeArrayInt8 = np.ndarray[tuple[int, int, int], np.dtype[np.int8]]
type ThreeArrayInt16 = np.ndarray[tuple[int, int, int], np.dtype[np.int16]]
type ThreeArrayInt32 = np.ndarray[tuple[int, int, int], np.dtype[np.int32]]
type ThreeArrayInt64 = np.ndarray[tuple[int, int, int], np.dtype[np.int64]]
type ThreeDArrayUint8 = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
type ThreeDArrayUint16 = np.ndarray[tuple[int, int, int], np.dtype[np.uint16]]
type ThreeDArrayUint32 = np.ndarray[tuple[int, int, int], np.dtype[np.uint32]]
type ThreeDArrayUint64 = np.ndarray[tuple[int, int, int], np.dtype[np.uint64]]
type ThreeDArrayInt = (
    ThreeArrayInt8
    | ThreeArrayInt16
    | ThreeArrayInt32
    | ThreeArrayInt64
    | ThreeDArrayUint8
    | ThreeDArrayUint16
    | ThreeDArrayUint32
    | ThreeDArrayUint64
)

type ThreeDArrayBool = np.ndarray[tuple[int, int, int], np.dtype[np.bool_]]

type ThreeDArray = ThreeDArrayFloat | ThreeDArrayInt | ThreeDArrayBool
