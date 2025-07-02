import json
import shutil
from datetime import datetime
from operator import attrgetter
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd

from .hydrology.HRUs import load_geom


class DynamicArray:
    """A dynamic array almost identical to a Numpy array, but that can grow and shrink in size."""

    __slots__: list = ["_data", "_n", "_extra_dims_names"]

    def __init__(
        self,
        input_array=None,
        n=None,
        max_n=None,
        extra_dims=None,
        extra_dims_names=[],
        dtype=None,
        fill_value=None,
    ):
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
    def data(self):
        return self._data[: self.n]

    @data.setter
    def data(self, value):
        self._data[: self.n] = value

    @property
    def max_n(self):
        return self._data.shape[0]

    @property
    def n(self):
        return self._n

    @property
    def extra_dims_names(self):
        return self._extra_dims_names

    @extra_dims_names.setter
    def extra_dims_names(self, value):
        self._extra_dims_names = value

    @n.setter
    def n(self, value):
        if value > self.max_n:
            raise ValueError("n cannot exceed max_n")
        self._n = value

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __array__(self, dtype=None):
        return np.asarray(self._data[: self.n], dtype=dtype)

    def __array_interface__(self):
        return self._data.__array_interface__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
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

    def __array_function__(self, func, types, args, kwargs):
        # Explicitly call __array_function__ of the underlying NumPy array
        modified_args: tuple = tuple(
            arg.data if isinstance(arg, DynamicArray) else arg for arg in args
        )
        modified_types: tuple = tuple(
            type(arg.data) if isinstance(arg, DynamicArray) else type(arg)
            for arg in args
        )
        return self._data.__array_function__(
            func, modified_types, modified_args, kwargs
        )

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __getitem__(self, key):
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

    def copy(self):
        """Create a deep copy of this DynamicArray."""
        new_array = DynamicArray.__new__(DynamicArray)
        new_array._data = self._data.copy()
        new_array._n = self._n
        new_array._extra_dims_names = (
            self._extra_dims_names.copy()
            if self._extra_dims_names is not None
            else None
        )
        return new_array

    def __repr__(self):
        return "DynamicArray(" + self.data.__str__() + ")"

    def __str__(self):
        return self.data.__str__()

    def __len__(self):
        return self._n

    def __getattr__(self, name):
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

    def __setattr__(self, name, value):
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

    def __getstate__(self):
        return self.data.__getstate__()

    def __setstate__(self, state):
        self.data.__setstate__(state)

    def __sizeof__(self):
        return self.data.__sizeof__()

    def _perform_operation(self, other, operation: str, inplace: bool = False):
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

    def __add__(self, other):
        return self._perform_operation(other, "__add__")

    def __radd__(self, other):
        return self._perform_operation(other, "__radd__")

    def __iadd__(self, other):
        return self._perform_operation(other, "__add__", inplace=True)

    def __sub__(self, other):
        return self._perform_operation(other, "__sub__")

    def __rsub__(self, other):
        return self._perform_operation(other, "__rsub__")

    def __isub__(self, other):
        return self._perform_operation(other, "__sub__", inplace=True)

    def __mul__(self, other):
        return self._perform_operation(other, "__mul__")

    def __rmul__(self, other):
        return self._perform_operation(other, "__rmul__")

    def __imul__(self, other):
        return self._perform_operation(other, "__mul__", inplace=True)

    def __truediv__(self, other):
        return self._perform_operation(other, "__truediv__")

    def __rtruediv__(self, other):
        return self._perform_operation(other, "__rtruediv__")

    def __itruediv__(self, other):
        return self._perform_operation(other, "__truediv__", inplace=True)

    def __floordiv__(self, other):
        return self._perform_operation(other, "__floordiv__")

    def __rfloordiv__(self, other):
        return self._perform_operation(other, "__rfloordiv__")

    def __ifloordiv__(self, other):
        return self._perform_operation(other, "__floordiv__", inplace=True)

    def __mod__(self, other):
        return self._perform_operation(other, "__mod__")

    def __rmod__(self, other):
        return self._perform_operation(other, "__rmod__")

    def __imod__(self, other):
        return self._perform_operation(other, "__mod__", inplace=True)

    def __pow__(self, other):
        return self._perform_operation(other, "__pow__")

    def __rpow__(self, other):
        return self._perform_operation(other, "__rpow__")

    def __ipow__(self, other):
        return self._perform_operation(other, "__pow__", inplace=True)

    def _compare(self, value: object, operation: str) -> bool:
        if isinstance(value, DynamicArray):
            return self.__class__(
                getattr(self.data, operation)(value.data), max_n=self._data.shape[0]
            )
        return getattr(self.data, operation)(value)

    def __eq__(self, value: object) -> bool:
        return self._compare(value, "__eq__")

    def __ne__(self, value: object) -> bool:
        return self._compare(value, "__ne__")

    def __gt__(self, value: object) -> bool:
        return self._compare(value, "__gt__")

    def __ge__(self, value: object) -> bool:
        return self._compare(value, "__ge__")

    def __lt__(self, value: object) -> bool:
        return self._compare(value, "__lt__")

    def __le__(self, value: object) -> bool:
        return self._compare(value, "__le__")

    def __and__(self, other):
        return self._perform_operation(other, "__and__")

    def __or__(self, other):
        return self._perform_operation(other, "__or__")

    def __neg__(self):
        return self._perform_operation(None, "__neg__")

    def __pos__(self):
        return self._perform_operation(None, "__pos__")

    def __invert__(self):
        return self._perform_operation(None, "__invert__")

    def save(self, path):
        np.savez_compressed(
            path.with_suffix(".storearray.npz"),
            **{slot: getattr(self, slot) for slot in self.__slots__},
        )

    @classmethod
    def load(cls, path):
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

    def __init__(self, validator: Callable | None = None):
        self.validator = validator

    def __setattr__(self, name, value):
        # If the name is 'validator', we allow setting it directly.
        # i.e., we do not validate the validator itself.
        if name == "validator":
            super().__setattr__(name, value)
            return

        if self.validator is not None:
            if not self.validator(value):
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
            ),
        )
        super().__setattr__(name, value)

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        for name, value in self.__dict__.items():
            # do not save the validator itself
            if name == "validator":
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
            elif isinstance(value, (list, dict)):
                with open((path / name).with_suffix(".json"), "w") as f:
                    json.dump(value, f)
            elif isinstance(value, str):
                with open((path / name).with_suffix(".txt"), "w") as f:
                    f.write(value)
            elif isinstance(value, datetime):
                with open((path / name).with_suffix(".datetime"), "w") as f:
                    f.write(value.isoformat())
            else:
                np.save((path / name).with_suffix(".npy"), value)

    def load(self, path):
        for filename in path.iterdir():
            if filename.suffixes == [".storearray", ".npz"]:
                setattr(
                    self,
                    filename.name.removesuffix("".join(filename.suffixes)),
                    DynamicArray.load(filename),
                )
            elif filename.suffixes == [".array", ".npz"]:
                setattr(
                    self,
                    filename.name.removesuffix("".join(filename.suffixes)),
                    np.load(filename)["value"],
                )
            elif filename.suffix == ".geoparquet":
                setattr(
                    self,
                    filename.stem,
                    load_geom(filename),
                )
            elif filename.suffix == ".parquet":
                setattr(
                    self,
                    filename.stem,
                    pd.read_parquet(filename),
                )
            elif filename.suffix == ".txt":
                with open(filename, "r") as f:
                    setattr(self, filename.stem, f.read())
            elif filename.suffix == ".datetime":
                with open(filename, "r") as f:
                    setattr(
                        self,
                        filename.stem,
                        datetime.fromisoformat(f.read()),
                    )
            elif filename.suffix == ".json":
                with open(filename, "r") as f:
                    setattr(self, filename.stem, json.load(f))
            else:
                setattr(self, filename.stem, np.load(filename).item())

        return self


class Store:
    """A class to manage the storage of model data in buckets.

    This class is use to store and restore the model's state in a structured way.
    """

    def __init__(self, model):
        self.model = model
        self.buckets = {}

    def create_bucket(self, name, validator=None):
        assert name not in self.buckets
        bucket = Bucket(validator=validator)
        self.buckets[name] = bucket
        return bucket

    def get_bucket(self, cls):
        name = self.get_name(cls)
        return self.buckets[name]

    def save(self, path=None):
        if path is None:
            path = self.path

        shutil.rmtree(path, ignore_errors=True)
        for name, bucket in self.buckets.items():
            self.model.logger.debug(f"Saving {name}")
            bucket.save(path / name)

    def load(self, path=None):
        if path is None:
            path = self.path

        for bucket_folder in path.iterdir():
            bucket = Bucket().load(bucket_folder)

            self.buckets[bucket_folder.name] = bucket

            split_name = bucket_folder.name.split(".")

            if not self.model.simulate_hydrology and split_name[0] == "hydrology":
                continue

            if len(split_name) == 1:
                bucket_parent_class = self.model
            else:
                bucket_parent_class = attrgetter(".".join(split_name[:-1]))(self.model)
            setattr(bucket_parent_class, split_name[-1], bucket)

    @property
    def path(self):
        return self.model.simulation_root_spinup / "store"
