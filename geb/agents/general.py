from numba import njit
from typing import Tuple
import math
import numpy as np
from pathlib import Path

from honeybees.agents import AgentBaseClass as HoneybeesAgentBaseClass


class AgentArray:
    def __init__(
        self,
        input_array=None,
        n=None,
        max_n=None,
        extra_dims=None,
        extra_dims_names=None,
        dtype=None,
        fill_value=None,
    ):
        if input_array is None and dtype is None:
            raise ValueError("Either input_array or dtype must be given")
        elif input_array is not None and dtype is not None:
            raise ValueError("Only one of input_array or dtype can be given")

        if input_array is not None:
            assert (
                extra_dims is None
            ), "extra_dims cannot be given if input_array is given"
            assert n is None, "n cannot be given if input_array is given"
            # assert dtype is not object
            assert input_array.dtype != object, "dtype cannot be object"
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
                assert extra_dims_names is not None
                assert len(extra_dims) == len(extra_dims_names)
                self.extra_dims_names = extra_dims_names
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
        return np.asarray(self._data, dtype=dtype)

    def __array_interface__(self):
        return self._data.__array_interface__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        modified_inputs = tuple(
            input_.data if isinstance(input_, AgentArray) else input_
            for input_ in inputs
        )
        result = self._data.__array_ufunc__(ufunc, method, *modified_inputs, **kwargs)
        if method == "reduce":
            return result
        elif not isinstance(inputs[0], AgentArray):
            return result
        else:
            return self.__class__(result, max_n=self._data.shape[0])

    def __array_function__(self, func, types, args, kwargs):
        # Explicitly call __array_function__ of the underlying NumPy array
        modified_args = tuple(
            arg.data if isinstance(arg, AgentArray) else arg for arg in args
        )
        modified_types = tuple(
            type(arg.data) if isinstance(arg, AgentArray) else type(arg) for arg in args
        )
        return self._data.__array_function__(
            func, modified_types, modified_args, kwargs
        )

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __repr__(self):
        return "AgentArray(" + self.data.__str__() + ")"

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
        if isinstance(other, AgentArray):
            other = other._data[: other._n]
        fn = getattr(self.data, operation)
        if other is None:
            args = ()
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
        if isinstance(value, AgentArray):
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


@njit(cache=True)
def downscale_volume(
    data_gt: Tuple[float, float, float, float, float, float],
    model_gt: Tuple[float, float, float, float, float, float],
    data: np.ndarray,
    mask: np.ndarray,
    grid_to_HRU_uncompressed: np.ndarray,
    downscale_mask: np.ndarray,
    HRU_land_size: np.ndarray,
) -> np.ndarray:
    xoffset = (model_gt[0] - data_gt[0]) / model_gt[1]
    assert 0.0001 > xoffset - round(xoffset) > -0.0001
    xoffset = round(xoffset)
    assert xoffset >= 0

    yoffset = (model_gt[3] - data_gt[3]) / model_gt[5]
    assert 0.0001 > yoffset - round(yoffset) > -0.0001
    yoffset = round(yoffset)
    assert yoffset >= 0

    xratio = data_gt[1] / model_gt[1]
    assert 0.0001 > xratio - round(xratio) > -0.0001
    assert xratio > 0
    xratio = round(xratio)

    yratio = data_gt[5] / model_gt[5]
    assert 0.0001 > yratio - round(yratio) > -0.0001
    assert yratio > 0
    yratio = round(yratio)

    downscale_invmask = ~downscale_mask
    assert xratio > 0
    assert yratio > 0
    assert xoffset > 0
    assert yoffset > 0
    ysize, xsize = data.shape
    yvarsize, xvarsize = mask.shape
    downscaled_array = np.zeros(HRU_land_size.size, dtype=np.float32)
    i = 0
    for y in range(ysize):
        y_left = y * yratio - yoffset
        y_right = min(y_left + yratio, yvarsize)
        y_left = max(y_left, 0)
        for x in range(xsize):
            x_left = x * xratio - xoffset
            x_right = min(x_left + xratio, xvarsize)
            x_left = max(x_left, 0)

            land_area_cell = 0
            for yvar in range(y_left, y_right):
                for xvar in range(x_left, x_right):
                    if not mask[yvar, xvar]:
                        k = yvar * xvarsize + xvar
                        HRU_right = grid_to_HRU_uncompressed[k]
                        # assert HRU_right != -1
                        if k > 0:
                            HRU_left = grid_to_HRU_uncompressed[k - 1]
                            # assert HRU_left != -1
                        else:
                            HRU_left = 0
                        land_area_cell += (
                            downscale_invmask[HRU_left:HRU_right]
                            * HRU_land_size[HRU_left:HRU_right]
                        ).sum()
                        i += 1

            if land_area_cell:
                for yvar in range(y_left, y_right):
                    for xvar in range(x_left, x_right):
                        if not mask[yvar, xvar]:
                            k = yvar * xvarsize + xvar
                            HRU_right = grid_to_HRU_uncompressed[k]
                            # assert HRU_right != -1
                            if k > 0:
                                HRU_left = grid_to_HRU_uncompressed[k - 1]
                                # assert HRU_left != -1
                            else:
                                HRU_left = 0
                            downscaled_array[HRU_left:HRU_right] = (
                                downscale_invmask[HRU_left:HRU_right]
                                * HRU_land_size[HRU_left:HRU_right]
                                / land_area_cell
                                * data[y, x]
                            )

    assert i == mask.size - mask.sum()
    return downscaled_array


class AgentBaseClass(HoneybeesAgentBaseClass):
    def __init__(self):
        if not hasattr(self, "redundancy"):
            self.redundancy = None  # default redundancy is None
        super().__init__()

    def get_max_n(self, n):
        if self.redundancy is None:
            return n
        else:
            max_n = math.ceil(n * (1 + self.redundancy))
            assert (
                max_n < 4294967295
            )  # max value of uint32, consider replacing with uint64
            return max_n

    def get_save_state_path(self, folder, mkdir=False):
        folder = Path(self.model.initial_conditions_folder, folder)
        if mkdir:
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_state(self, folder: str):
        save_state_path = self.get_save_state_path(folder, mkdir=True)
        with open(save_state_path / "state.txt", "w") as f:
            for attribute, value in self.agent_arrays.items():
                f.write(f"{attribute}\n")
                fp = save_state_path / f"{attribute}.npz"
                np.savez_compressed(fp, data=value.data)

    def restore_state(self, folder: str):
        save_state_path = self.get_save_state_path(folder)
        with open(save_state_path / "state.txt", "r") as f:
            for line in f:
                attribute = line.strip()
                fp = save_state_path / f"{attribute}.npz"
                values = np.load(fp)["data"]
                if not hasattr(self, "max_n"):
                    self.max_n = self.get_max_n(values.shape[0])
                values = AgentArray(values, max_n=self.max_n)

                setattr(self, attribute, values)

    @property
    def agent_arrays(self):
        agent_arrays = {
            name: value
            for name, value in vars(self).items()
            if isinstance(value, AgentArray)
        }
        ids = [id(v) for v in agent_arrays.values()]
        if len(set(ids)) != len(ids):
            duplicate_arrays = [
                name for name, value in agent_arrays.items() if ids.count(id(value)) > 1
            ]
            raise AssertionError(
                f"Duplicate agent array names: {', '.join(duplicate_arrays)}."
            )
        return agent_arrays
