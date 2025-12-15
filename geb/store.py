"""Storage classes for model data."""

from __future__ import annotations

import json
import pickle
import shutil
from collections import deque
from datetime import datetime
from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Literal, overload

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from numpy.typing import NDArray

from geb.workflows.io import read_geom

if TYPE_CHECKING:
    from geb.model import GEBModel


class DynamicArray:
    """A resizable array container that behaves like a NumPy array but can grow and shrink.

    This class wraps a contiguous underlying buffer and exposes most array-like
    behaviour while tracking a logical length that may be less than the physical
    capacity. It is designed for use cases where elements are appended, removed,
    or otherwise managed without reallocating on every size change.

    The DynamicArray:
    - Stores data in a fixed-capacity backing buffer and tracks a current length.
    - Exposes a `data` view representing the active portion of the buffer.
    - Implements NumPy protocols (array conversion, array function and ufunc)
        so it interoperates with NumPy operations.
    - Supports slicing, arithmetic, in-place operations and comparisons, delegating
        to the underlying array where appropriate.
    - Can be saved to and loaded from disk using the provided save and load helpers.
    """

    __slots__: list = ["_data", "_n", "_extra_dims_names"]

    def __init__(
        self,
        input_array: npt.ArrayLike | None = None,
        n: int | None = None,
        max_n: int | None = None,
        extra_dims: tuple[int, ...] | None = None,
        extra_dims_names: Iterable[str] = [],
        dtype: npt.DTypeLike | None = None,
        fill_value: Any | None = None,
    ) -> None:
        """Initialize a DynamicArray.

        Args:
            input_array: An existing array-like object to initialize from. If provided,
                    the DynamicArray will use its values as the initial contents.
            n: The initial logical length of the DynamicArray. Required when creating
                    from scratch (i.e. when input_array is not provided).
            max_n: The maximum capacity (physical size) of the backing buffer. If not
                    provided when initializing from an existing array, the array itself is
                    used as the backing buffer and no extra capacity is allocated.
            extra_dims: Optional shape for additional trailing dimensions stored for
                    each element (for example, per-element vectors or matrices). When
                    provided, the backing buffer will have shape (max_n, *extra_dims).
            extra_dims_names: Optional list of names for the extra dimensions. These
                    are preserved and adjusted when slicing across extra dimensions.
            dtype: Data type to allocate when creating an empty or preallocated array.
                    Mutually exclusive with input_array.
            fill_value: Optional fill value used to initialize the backing buffer when
                    creating from dtype and max_n.

        Raises:
                ValueError: If neither input_array nor dtype is provided, or if both are
                provided, or if requested logical length exceeds capacity.

        Attributes:
                data: A view of the active portion of the buffer (up to the logical length).
                n: The logical length of the array (number of active elements).
                max_n: The capacity of the backing buffer (maximum number of elements).
                extra_dims_names: Names associated with any extra trailing dimensions.
                (Internal) _data: The backing buffer storing the raw values.

        Examples:
                - Create from existing array:
                        dyn = DynamicArray(input_array=some_array)

                - Create an empty buffer with capacity and logical length:
                        dyn = DynamicArray(n=5, max_n=100, dtype=some_dtype)

                - Perform NumPy-like operations:
                        dyn2 = dyn + 3
                        mask = dyn > 0

                - Slice preserving DynamicArray wrapper across the first axis:
                        dyn_slice = dyn[:]  # returns a DynamicArray copy
        """
        self.extra_dims_names = np.array(extra_dims_names, dtype=str)

        if input_array is None and dtype is None:
            raise ValueError("Either input_array or dtype must be given")
        elif input_array is not None and dtype is not None:
            raise ValueError("Only one of input_array or dtype can be given")

        if input_array is not None:
            assert extra_dims is None, (
                "extra_dims cannot be given if input_array is given"
            )
            assert n is None, "n cannot be given if input_array is given"
            # assert dtype is not object
            assert input_array.dtype != object, "dtype cannot be object"
            input_array: np.ndarray = np.asarray(input_array)
            n = input_array.shape[0]
            if max_n:
                if input_array.ndim == 1:
                    shape = max_n
                else:
                    shape = (max_n, *input_array.shape[1:])
                self._data = np.empty_like(input_array, shape=shape)
                n = input_array.shape[0]
                self._n = n
                self._data[:n] = input_array
            else:
                self._data = input_array
                self._n = n
        else:
            assert dtype is not None
            assert dtype is not object
            assert n is not None
            assert max_n is not None
            if extra_dims is None:
                shape = max_n
            else:
                shape = (max_n,) + extra_dims
                assert self.extra_dims_names is not None
                assert len(extra_dims) == len(self.extra_dims_names)

            if fill_value is not None:
                self._data = np.full(shape, fill_value, dtype=dtype)
            else:
                self._data = np.empty(shape, dtype=dtype)
            self._n = n

    @property
    def data(self) -> npt.NDArray[Any]:
        """
        View of the active portion of the DynamicArray.

        Returns:
            A NumPy view of the active elements (shape depends on extra dimensions).
        """
        return self._data[: self.n]

    @data.setter
    def data(self, value: npt.NDArray[Any]) -> None:
        """
        Replace the active portion of the DynamicArray with `value`.

        Args:
            value: Array-like object with shape compatible with the active slice.
        """
        self._data[: self.n] = value

    @property
    def max_n(self) -> int:
        """
        The maximum capacity of the DynamicArray.

        Returns:
            The maximum number of elements the DynamicArray buffer can hold (elements).
        """
        return self._data.shape[0]

    @property
    def n(self) -> int:
        """
        The logical length (number of active elements).

        Returns:
            Current logical length (elements).
        """
        return self._n

    @property
    def extra_dims_names(self) -> npt.NDArray[np.str_] | None:
        """
        Names associated with any extra trailing dimensions.

        Returns:
            Array of names for extra trailing dimensions, or None.
        """
        return self._extra_dims_names

    @extra_dims_names.setter
    def extra_dims_names(self, value: npt.NDArray[np.str_]) -> None:
        """
        Set names for extra trailing dimensions.

        Args:
            value: Iterable of strings representing names for each extra trailing dimension.
        """
        self._extra_dims_names = value

    @n.setter
    def n(self, value: int) -> None:
        """
        Set the logical length.

        Args:
            value: New logical length (elements).

        Raises:
            ValueError: If `value` exceeds the capacity `max_n`.
        """
        if value > self.max_n:
            raise ValueError("n cannot exceed max_n")
        self._n = value

    def __array_finalize__(self, obj: object) -> None:
        """
        Array finalize hook for NumPy interoperability.

        Notes:
            Kept minimal; no additional finalization is required currently.

        Args:
            obj: The object being finalized (handled by NumPy protocols).
        """
        if obj is None:
            return

    def __array__(self, dtype: type | None = None) -> np.ndarray:
        """
        Return a NumPy array representation of the active data.

        Args:
            dtype: Optional dtype to cast the returned array.

        Returns:
            A NumPy array containing the active elements.
        """
        return np.asarray(self._data[: self.n], dtype=dtype)

    @property
    def __array_interface__(self) -> dict[str, Any]:
        """
        Expose the array interface of the entire underlying data (up to max_n).

        Returns:
            The __array_interface__ mapping from the underlying NumPy array.
        """
        return self._data.__array_interface__

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        *inputs: tuple[Any],
        **kwargs: dict[str, Any],
    ) -> Any:
        """
        Handle NumPy ufuncs applied to DynamicArray instances.

        The active slices of DynamicArray operands are used for the operation. Results
        are wrapped back into DynamicArray when appropriate.

        Args:
            ufunc: The NumPy ufunc being applied.
            method: Ufunc method name (e.g., "__call__", "reduce").
            *inputs: Operands for the ufunc.
            **kwargs: Keyword arguments forwarded to the ufunc.

        Returns:
            Result of the ufunc, possibly wrapped as a DynamicArray.
        """
        modified_inputs = tuple(
            input_.data if isinstance(input_, DynamicArray) else input_
            for input_ in inputs
        )
        result = self._data.__array_ufunc__(ufunc, method, *modified_inputs, **kwargs)
        if method == "reduce":
            return result
        elif not isinstance(inputs[0], DynamicArray):
            return result
        else:
            return self.__class__(result, max_n=self._data.shape[0])

    def __array_function__(
        self,
        func: Callable,
        types: tuple[Any],
        args: tuple[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Delegate NumPy __array_function__ calls to the underlying NumPy array.

        Args:
            func: The NumPy function being invoked.
            types: Sequence of types participating in the operation.
            args: Positional arguments passed to the function.
            kwargs: Keyword arguments passed to the function.

        Returns:
            Result of calling the function on the underlying NumPy array(s).
        """

        def recursive_convert(arg: Any) -> Any:
            if isinstance(arg, DynamicArray):
                return arg.data
            if isinstance(arg, (list, tuple)):
                return type(arg)(recursive_convert(x) for x in arg)
            return arg

        # Explicitly call __array_function__ of the underlying NumPy array
        modified_args = tuple(recursive_convert(arg) for arg in args)

        return self._data.__array_function__(func, (np.ndarray,), modified_args, kwargs)

    def __setitem__(
        self,
        key: int | slice | ... | NDArray[np.integer] | NDArray[np.bool_],
        value: Any,
    ) -> None:
        """
        Set item(s) in the active portion of the array.

        Args:
            key: Index or slice.
            value: Value to assign.
        """
        self.data.__setitem__(key, value)

    def __getitem__(
        self,
        key: int | slice | ... | NDArray[np.integer] | NDArray[np.bool_],
    ) -> DynamicArray | np.ndarray:
        """
        Retrieve item(s) or a sliced DynamicArray.

        Notes:
            - If the slice selects the entire first axis (elements), returns a
              DynamicArray preserving extra-dimension names when possible.
            - If a full slice (:) is requested, returns a copy.
            - Otherwise returns a NumPy array view.

        Args:
            key: Index, slice, or tuple of indices/slices.

        Returns:
            A DynamicArray for full-element selections or a NumPy array for other slices.
        """
        # if the first key selects the entire array, we can return
        # a new DynamicArray, but with only the extra dimensions
        # sliced
        if (
            isinstance(key, tuple)
            and isinstance(key[0], slice)
            and key[0] == slice(None, None, None)
        ):
            data = self.data.__getitem__(key)

            new_extra_dims_names: list = []
            assert self.extra_dims_names is not None
            for i, slicer in enumerate(key[1:]):
                if isinstance(slicer, (slice, list)):
                    new_extra_dims_names.append(self.extra_dims_names[i])

            assert len(data.shape[1:]) == len(new_extra_dims_names), (
                "Mismatch in number of extra dimensions"
            )

            return DynamicArray(
                data,
                max_n=self.max_n,
                extra_dims_names=new_extra_dims_names,
            )
        elif isinstance(key, slice) and key == slice(None, None, None):
            return self.copy()

        # otherwise, we return a numpy array with the sliced data
        else:
            return self.data.__getitem__(key)

    def copy(self) -> DynamicArray:
        """
        Create a deep copy of this DynamicArray.

        Returns:
            A new DynamicArray instance that is a deep copy of the current instance.
        """
        new_array = DynamicArray.__new__(DynamicArray)
        new_array._data = self._data.copy()
        new_array._n = self._n
        new_array._extra_dims_names = (
            self._extra_dims_names.copy()
            if self._extra_dims_names is not None
            else None
        )
        return new_array

    def __repr__(self) -> str:
        """
        Formal string representation.

        Returns:
            A string that represents this DynamicArray.
        """
        return "DynamicArray(" + self.data.__str__() + ")"

    def __str__(self) -> str:
        """
        Informal string representation.

        Returns:
            A human-readable string for the active data.
        """
        return self.data.__str__()

    def __len__(self) -> int:
        """
        Number of active elements.

        Returns:
            The logical length (elements).
        """
        return self._n

    def __getattr__(self, name: str) -> Any:
        """
        Get attributes either from the wrapper internals or the active data.

        If the attribute is one of the internal attributes, defer to the normal
        attribute lookup. Otherwise, forward the attribute access to the active
        NumPy data view.

        Args:
            name: Attribute name.

        Returns:
            The requested attribute value.
        """
        if name in (
            "_data",
            "data",
            "_n",
            "n",
            "_extra_dims_names",
            "extra_dims_names",
        ):
            return super().__getattr__(name)
        else:
            return getattr(self.data, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set attributes either on the wrapper internals or on the active data.

        Notes:
            Internal attributes are set on the object itself. Other attributes
            are forwarded to the active NumPy data view.

        Args:
            name: Attribute name.
            value: Value to set.
        """
        if name in (
            "_data",
            "data",
            "_n",
            "n",
            "_extra_dims_names",
            "extra_dims_names",
        ):
            super().__setattr__(name, value)
        else:
            setattr(self.data, name, value)

    def __sizeof__(self) -> int:
        """
        Return the memory size of the active data.

        Returns:
            Size in bytes as reported by the underlying array.
        """
        return self.data.__sizeof__()

    def _perform_operation(
        self,
        other: Any,
        operation: str,
        inplace: bool = False,
    ) -> DynamicArray:
        """
        Helper to perform binary/unary array operations delegating to NumPy.

        This normalizes DynamicArray operands to NumPy arrays and performs the
        requested operation by name. Optionally performs the operation in-place
        on the active slice.

        Args:
            other: The other operand (DynamicArray or array-like).
            operation: Name of the operation method to call on the active data.
            inplace: Whether to perform the operation in-place.

        Returns:
            A new DynamicArray wrapping the result, or self if inplace=True.
        """
        if isinstance(other, DynamicArray):
            other = other._data[: other._n]
        fn = getattr(self.data, operation)
        if other is None:
            args: tuple[()] = ()
        else:
            args = (other,)
        result = fn(*args)
        if inplace:
            self.data = result
            return self
        else:
            return self.__class__(result, max_n=self._data.shape[0])

    def __add__(self, other: Any) -> DynamicArray:
        """Addition operator.

        Args:
            other: The value to add (scalar or array-like).

        Returns:
            A DynamicArray with each element resulting from the addition operation.
        """
        return self._perform_operation(other, "__add__")

    def __radd__(self, other: Any) -> DynamicArray:
        """Right-hand addition operator.

        Args:
            other: The value to add (scalar or array-like).

        Returns:
            A DynamicArray with each element resulting from the addition operation.
        """
        return self._perform_operation(other, "__radd__")

    def __iadd__(self, other: Any) -> DynamicArray:
        """In-place addition operator.

        Args:
            other: The value to add (scalar or array-like).

        Returns:
            The modified DynamicArray instance.
        """
        return self._perform_operation(other, "__add__", inplace=True)

    def __sub__(self, other: Any) -> DynamicArray:
        """Subtraction operator.

        Args:
            other: The value to subtract by (scalar or array-like).

        Returns:
            A DynamicArray with each element resulting from the subtraction operation.
        """
        return self._perform_operation(other, "__sub__")

    def __rsub__(self, other: Any) -> DynamicArray:
        """Right-hand subtraction operator.

        Args:
            other: The value to subtract by (scalar or array-like).

        Returns:
            A DynamicArray with each element resulting from the subtraction operation.

        """
        return self._perform_operation(other, "__rsub__")

    def __isub__(self, other: Any) -> DynamicArray:
        """In-place subtraction operator.

        Args:
            other: The value to subtract by (scalar or array-like).

        Returns:
            The modified DynamicArray instance.

        """
        return self._perform_operation(other, "__sub__", inplace=True)

    def __mul__(self, other: Any) -> DynamicArray:
        """Multiplication operator.

        Args:
            other: The value to multiply by (scalar or array-like).

        Returns:
            A DynamicArray with each element resulting from the multiplication operation.

        """
        return self._perform_operation(other, "__mul__")

    def __rmul__(self, other: Any) -> DynamicArray:
        """Right-hand multiplication operator.

        Args:
            other: The value to multiply by (scalar or array-like).

        Returns:
            A DynamicArray with each element resulting from the multiplication operation.

        """
        return self._perform_operation(other, "__rmul__")

    def __imul__(self, other: Any) -> DynamicArray:
        """In-place multiplication operator.

        Args:
            other: The value to multiply by (scalar or array-like).

            Returns:
            The modified DynamicArray instance.

        """
        return self._perform_operation(other, "__mul__", inplace=True)

    def __truediv__(self, other: Any) -> DynamicArray:
        """True division operator.

        Args:
            other: The value to divide by (scalar or array-like).

        Returns:
            A DynamicArray with each element resulting from the true division operation.

        """
        return self._perform_operation(other, "__truediv__")

    def __rtruediv__(self, other: Any) -> DynamicArray:
        """Right-hand true division operator.

        Args:
            other: The value to divide by (scalar or array-like).

        Returns:
            A DynamicArray with each element resulting from the true division operation.

        """
        return self._perform_operation(other, "__rtruediv__")

    def __itruediv__(self, other: Any) -> DynamicArray:
        """In-place true division operator.

        Args:
            other: The value to divide by (scalar or array-like).

        Returns:
            The modified DynamicArray instance.

        """
        return self._perform_operation(other, "__truediv__", inplace=True)

    def __floordiv__(self, other: Any) -> DynamicArray:
        """Floor division operator.

        Args:
            other: The value to floor divide by (scalar or array-like).

        Returns:
            A DynamicArray with each element resulting from the floor division operation.

        """
        return self._perform_operation(other, "__floordiv__")

    def __rfloordiv__(self, other: Any) -> DynamicArray:
        """Right-hand floor division operator.

        Args:
            other: The value to floor divide by (scalar or array-like).

        Returns:
            A DynamicArray with each element resulting from the floor division operation.

        """
        return self._perform_operation(other, "__rfloordiv__")

    def __ifloordiv__(self, other: Any) -> DynamicArray:
        """In-place floor division operator.

        Args:
            other: The value to floor divide by (scalar or array-like).

        Returns:
            The modified DynamicArray instance.

        """
        return self._perform_operation(other, "__floordiv__", inplace=True)

    def __mod__(self, other: Any) -> DynamicArray:
        """Modulo operator.

        Args:
            other: The value to modulo by (scalar or array-like).

        Returns:
            A DynamicArray with each element resulting from the modulo operation.

        """
        return self._perform_operation(other, "__mod__")

    def __rmod__(self, other: Any) -> DynamicArray:
        """Right-hand modulo operator.

        Args:
            other: The value to modulo by (scalar or array-like).

        Returns:
            A DynamicArray with each element resulting from the modulo operation.

        """
        return self._perform_operation(other, "__rmod__")

    def __imod__(self, other: Any) -> DynamicArray:
        """In-place modulo operator.

        Args:
            other: The value to modulo by (scalar or array-like).

        Returns:
            The modified DynamicArray instance.
        """
        return self._perform_operation(other, "__mod__", inplace=True)

    def __pow__(self, other: Any) -> DynamicArray:
        """Power operator.

        Args:
            other: The exponent value (scalar or array-like).

        Returns:
            A DynamicArray with each element raised to the power of `other`.
        """
        return self._perform_operation(other, "__pow__")

    def __rpow__(self, other: Any) -> DynamicArray:
        """Right-hand power operator.

        Args:
            other: The exponent value (scalar or array-like).

        Returns:
            A DynamicArray with each element raised to the power of `other`.
        """
        return self._perform_operation(other, "__rpow__")

    def __ipow__(self, other: Any) -> DynamicArray:
        """In-place power operator.

        Returns:
            The modified DynamicArray instance.
        """
        return self._perform_operation(other, "__pow__", inplace=True)

    def _compare(self, value: object, operation: str) -> Any:
        """
        Helper for comparison operations.

        Args:
            value: Value to compare against (DynamicArray or scalar).
            operation: Comparison method name.

        Returns:
            Result of the comparison.
        """
        if isinstance(value, DynamicArray):
            res = getattr(self.data, operation)(value.data)
            if res is NotImplemented:
                return NotImplemented
            return self.__class__(res, max_n=self._data.shape[0])
        res = getattr(self.data, operation)(value)
        if res is NotImplemented:
            return NotImplemented
        return self.__class__(res)

    @overload
    def __eq__(self, value: DynamicArray) -> DynamicArray: ...

    @overload
    def __eq__(self, value: object) -> Any: ...

    def __eq__(self, value: object) -> Any:
        """Equality comparison.

        Args:
            value: Value to compare against (DynamicArray or scalar).

        Returns:
            Result of the comparison.
        """
        return self._compare(value, "__eq__")

    @overload
    def __ne__(self, value: DynamicArray) -> DynamicArray: ...

    @overload
    def __ne__(self, value: object) -> Any: ...

    def __ne__(self, value: object) -> Any:
        """Inequality comparison.

        Args:
            value: Value to compare against (DynamicArray or scalar).

        Returns:
            Result of the comparison.
        """
        return self._compare(value, "__ne__")

    def __gt__(self, value: Any) -> DynamicArray:
        """Greater-than comparison.

        Args:
            value: Value to compare against (DynamicArray or scalar).

        Returns:
            Result of the comparison.
        """
        return self._compare(value, "__gt__")

    def __ge__(self, value: Any) -> DynamicArray:
        """Greater-than-or-equal comparison.

        Args:
            value: Value to compare against (DynamicArray or scalar).

        Returns:
            Result of the comparison.
        """
        return self._compare(value, "__ge__")

    def __lt__(self, value: Any) -> DynamicArray:
        """Less-than comparison.

        Args:
            value: Value to compare against (DynamicArray or scalar).

        Returns:
            Result of the comparison, possibly wrapped as a DynamicArray.
        """
        return self._compare(value, "__lt__")

    def __le__(self, value: Any) -> DynamicArray:
        """Less-than-or-equal comparison.

        Args:
            value: Value to compare against (DynamicArray or scalar).

        Returns:
            Result of the comparison.
        """
        return self._compare(value, "__le__")

    def __and__(self, other: npt.NDArray[Any]) -> DynamicArray:
        """Bitwise and / logical and operator.

        Returns:
            A DynamicArray with each element resulting from the bitwise and operation.
        """
        return self._perform_operation(other, "__and__")

    def __or__(self, other: npt.NDArray[Any]) -> DynamicArray:
        """Bitwise or / logical or operator.

        Returns:
            A DynamicArray with each element resulting from the bitwise or operation.

        """
        return self._perform_operation(other, "__or__")

    def __neg__(self) -> DynamicArray:
        """Unary negation.

        Returns:
            A DynamicArray with each element negated.

        """
        return self._perform_operation(None, "__neg__")

    def __pos__(self) -> DynamicArray:
        """Unary plus (no-op).

        Returns:
            A DynamicArray identical to self.

        """
        return self._perform_operation(None, "__pos__")

    def __invert__(self) -> DynamicArray:
        """Bitwise invert / logical not.

        Returns:
            A DynamicArray with each element inverted.
        """
        return self._perform_operation(None, "__invert__")

    def save(self, path: Path) -> None:
        """
        Save the DynamicArray to disk in a compressed NumPy archive.

        Args:
            path: Path-like object (without suffix) where the .storearray.npz file will be written.
        """
        np.savez_compressed(
            path.with_suffix(".storearray.npz"),
            **{slot: getattr(self, slot) for slot in self.__slots__},
        )

    @classmethod
    def load(cls, path: Path) -> DynamicArray:
        """
        Load a DynamicArray previously saved with `save`.

        Args:
            path: Path to a .storearray.npz file.

        Returns:
            A reconstructed DynamicArray instance.
        """
        assert path.suffixes == [".storearray", ".npz"]
        with np.load(path) as data:
            obj = cls.__new__(cls)
            for slot in cls.__slots__:
                setattr(obj, slot, data[slot])
            return obj


class Bucket:
    """A class to manage the storage of model data in a bucket.

    Each bucket is associated with a specific part of the model, usually a Module.

    Args:
        validator: A function to validate values before setting them.
            If provided, it should return True for valid values and False for invalid ones.
            Defaults to None, meaning no validation is performed.

    """

    def __init__(self, validator: Callable | None = None) -> None:
        """Initialize the Bucket with an optional validator.

        Args:
            validator: Validation function to use on any data that is set
                in the bucket. Must return True for correct data, False
                otherwise. Defaults to None.
        """
        self._validator = validator

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        """Iterate over the items in the bucket.

        Yields:
            Tuples of (name, value) for each item in the bucket, excluding the validator.
        """
        for name, value in self.__dict__.items():
            if name == "_validator":
                continue
            yield name, value

    def __setattr__(
        self,
        name: str,
        value: DynamicArray
        | int
        | float
        | np.ndarray
        | list
        | gpd.GeoDataFrame
        | pd.DataFrame
        | str
        | dict
        | datetime
        | Callable,
    ) -> None:
        """Set an value in the bucket with optional validation, except if the name is '_validator'.

        If the name is '_validator', it sets the validator function directly.
        If the name is not '_validator' and a validator function is provided, it validates the value before setting it.
        If no validator is provided, it sets the value directly.

        Args:
            name: The name of the attribute to set.
            value: The value to set.

        Raises:
            ValueError: If the value does not pass validation.
        """
        # If the name is 'validator', we allow setting it directly.
        # i.e., we do not validate the validator itself.
        if name == "_validator":
            super().__setattr__(name, value)
            return

        if self._validator is not None:
            if not self._validator(value):
                raise ValueError(f"Value for {name} does not pass validation: {value}")

        assert isinstance(
            value,
            (
                DynamicArray,
                int,
                float,
                np.ndarray,
                list,
                gpd.GeoDataFrame,
                pd.DataFrame,
                str,
                dict,
                datetime,
                np.generic,
                deque,
            ),
        )
        super().__setattr__(name, value)

    def save(self, path: Path) -> None:
        """Save the bucket data to disk.

        Can then be loaded back with `load`.

        Args:
            path: The location where the data should be saved. Must be a directory.

        Raises:
            ValueError: If a value type is not supported for saving.
        """
        path.mkdir(parents=True, exist_ok=True)
        for name, value in self.__dict__.items():
            # do not save the validator itself
            if name == "_validator":
                continue
            if isinstance(value, DynamicArray):
                value.save(path / name)
            elif isinstance(value, np.ndarray):
                np.savez_compressed(
                    (path / name).with_suffix(".array.npz"), value=value
                )
            elif isinstance(value, gpd.GeoDataFrame):
                value.to_parquet(
                    (path / name).with_suffix(".geoparquet"),
                    engine="pyarrow",
                    compression="gzip",
                    compression_level=9,
                )
            elif isinstance(value, pd.DataFrame):
                value.to_parquet(
                    (path / name).with_suffix(".parquet"),
                    engine="pyarrow",
                    compression="gzip",
                    compression_level=9,
                )
            elif isinstance(value, (list, dict, float, int, str, datetime)):
                with open((path / name).with_suffix(".yml"), "w") as f:
                    yaml.safe_dump(value, f, default_flow_style=False)
            elif isinstance(value, np.ndarray):
                if value.ndim == 0:
                    raise ValueError(
                        "0-dim arrays should be saved as scalars. Otherwise we get undefined and unexpected behavior when loading the array back. Here, 0-dim array are converted to scalars."
                    )
                np.save((path / name).with_suffix(".npy"), value)
            elif isinstance(value, np.generic):
                np.save((path / name).with_suffix(".npy"), value)
            elif isinstance(value, deque):
                # TODO: Remove this option when we use the BMI of SFINCS and deques
                # are no longer needed.
                with open((path / name).with_suffix(".pkl"), "wb") as f:
                    pickle.dump(value, f)
            else:
                raise ValueError(f"Cannot save value of type {type(value)} for {name}")

    def load(self, path: Path) -> Bucket:
        """Load the bucket data from disk to the Bucket instance.

        Args:
            path: The location of the data to be restored.

        Returns:
            The Bucket instance itself with the loaded data.

        Raises:
            ValueError: If a value type is not supported for loading.
        """
        for filename in path.iterdir():
            if filename.suffixes == [".storearray", ".npz"]:
                setattr(
                    self,
                    filename.name.removesuffix("".join(filename.suffixes)),
                    DynamicArray.load(filename),
                )
            elif filename.suffixes == [".array", ".npz"] or filename.suffix == ".npy":
                value = np.load(filename)
                # unpack the value if it was saved as a .array.npz
                if filename.suffixes == [".array", ".npz"]:
                    value = value["value"]
                if value.ndim == 0:
                    value = value[()]  # convert to scalar but keep dtype
                setattr(
                    self,
                    filename.name.removesuffix("".join(filename.suffixes)),
                    value,
                )
            elif filename.suffix == ".geoparquet":
                setattr(
                    self,
                    filename.stem,
                    read_geom(filename),
                )
            elif filename.suffix == ".parquet":
                setattr(
                    self,
                    filename.stem,
                    pd.read_parquet(filename),
                )
            elif filename.suffix == ".yml":
                with open(filename, "r") as f:
                    setattr(self, filename.stem, yaml.safe_load(f))
            # TODO: Can be removed in 2026
            elif filename.suffix == ".txt":
                with open(filename, "r") as f:
                    setattr(self, filename.stem, f.read())
            # TODO: Can be removed in 2026
            elif filename.suffix == ".datetime":
                with open(filename, "r") as f:
                    setattr(
                        self,
                        filename.stem,
                        datetime.fromisoformat(f.read()),
                    )
            # TODO: Can be removed in 2026
            elif filename.suffix == ".json":
                with open(filename, "r") as f:
                    setattr(self, filename.stem, json.load(f))
            elif filename.suffix == ".pkl":
                # TODO: Remove this option when we use the BMI of SFINCS and deques
                # are no longer needed.
                with open(filename, "rb") as f:
                    setattr(
                        self,
                        filename.stem,
                        pickle.load(f),
                    )
            else:
                raise ValueError(f"Cannot load value {filename}")

        return self


class Store:
    """A class to manage the storage of model data in buckets.

    This class is use to store and restore the model's state in a structured way.
    """

    def __init__(self, model: GEBModel) -> None:
        """Initialize the Store with a reference to the model.

        Args:
            model: The GEBModel instance that this store is associated with.
        """
        self.model = model
        self.buckets = {}

    def create_bucket(self, name: str, validator: Callable | None = None) -> Bucket:
        """Create a new bucket in the store.

        The bucket is used to store data for a specific part of the model, usually a Module,
        which can then be restored later. This is useful for saving the state of the model
        at a specific point in time, and for restoring it later.

        Args:
            name: The name of the bucket to create.
            validator: A function to validate values before setting them in the bucket.
                If provided, it should return True for valid values and False for invalid ones.
                Defaults to None, meaning no validation is performed.

        Returns:
            The created Bucket instance.
        """
        assert name not in self.buckets
        bucket: Bucket = Bucket(validator=validator)
        self.buckets[name] = bucket
        return bucket

    def save(self, path: None | Path = None) -> None:
        """Save the store data from the model to disk.

        Removes any existing data in the target directory before saving.

        Args:
            path: A Path object representing the directory to load the model data from. Defaults to None.
                In this case, a default path is used. In most cases this should not be changed, but can
                be useful for special cases such as forecasting and testing.
        """
        if path is None:
            path: Path = self.path

        shutil.rmtree(path, ignore_errors=True)
        for name, bucket in self.buckets.items():
            self.model.logger.debug(f"Saving {name}")
            bucket.save(path / name)

    def load(self, path: None | Path = None, omit: None | str = None) -> None:
        """Load the store data from disk into the model.

        If no path is provided, it defaults to the store path of the model.

        Args:
            path: A Path object representing the directory to load the model data from. Defaults to None.
                In this case, a default path is used. In most cases this should not be changed, but can
                be useful for special cases such as forecasting and testing.
            omit: An optional string. If provided, any bucket whose name contains this string will be skipped during loading.
        """
        if path is None:
            path = self.path

        for bucket_folder in path.iterdir():
            # Mac OS X creates a .DS_Store file in directories, which we ignore
            if bucket_folder.name == ".DS_Store":
                continue
            elif omit is not None and omit in bucket_folder.name:
                self.model.logger.info(f"Skipping loading of bucket {bucket_folder}")
                continue
            bucket = Bucket().load(bucket_folder)

            self.buckets[bucket_folder.name] = bucket

            split_name = bucket_folder.name.split(".")

            # Skip loading hydrology-related buckets if hydrology simulation is disabled
            if (
                not self.model.simulate_hydrology
                and (split_name[0] == "hydrology")
                and not split_name[1] == "grid"
            ):
                continue

            # Skip loading flood-related buckets if flood simulation is disabled
            if self.model.config["hazards"]["floods"]["simulate"] is False and (
                split_name[0] == "floods"
            ):
                continue

            if len(split_name) == 1:
                bucket_parent_class = self.model
            else:
                bucket_parent_class = attrgetter(".".join(split_name[:-1]))(self.model)
            setattr(bucket_parent_class, split_name[-1], bucket)

    @property
    def path(self) -> Path:
        """The path where the store data is saved.

        Returns:
            A Path object representing the directory for storing model data.
        """
        return self.model.simulation_root_spinup / "store"
