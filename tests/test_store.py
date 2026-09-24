"""Tests for storage objects in GEB."""

import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from numba import njit

from geb.store import DynamicArray

from .testconfig import tmp_folder


def test_1D_dynamic_array_slice() -> None:
    """Test slicing operations on a 1D DynamicArray.

    Tests various slicing operations including full slice, single element access,
    range slicing, boolean indexing, and array indexing. Verifies that slicing
    returns appropriate types (DynamicArray vs ndarray) and preserves metadata.
    """
    a = DynamicArray(np.array([1, 2, 3]), max_n=10)

    sliced = a[:]
    assert (sliced == a).all()
    assert isinstance(sliced, DynamicArray)
    assert sliced.max_n == 10
    assert sliced.n == 3
    assert sliced.extra_dims_names.size == 0

    sliced = a[0]
    assert (sliced == 1).all()

    sliced = a[0:2]
    assert (sliced == np.array([1, 2])).all()
    assert isinstance(sliced, np.ndarray)

    sliced = a[[True, False, True]]
    assert (sliced == np.array([1, 3])).all()
    assert isinstance(sliced, np.ndarray)

    sliced = a[np.array([True, False, True])]
    assert (sliced == np.array([1, 3])).all()
    assert isinstance(sliced, np.ndarray)


def test_2D_dynamic_array_slice() -> None:
    """Test slicing operations on a 2D DynamicArray.

    Tests various 2D slicing operations including full slices, column/row selection,
    boolean indexing, and array indexing. Verifies that slicing returns appropriate
    types and correctly handles extra dimensions metadata.
    """
    a = DynamicArray(
        np.array([[1, 2], [3, 4], [5, 6]]), max_n=10, extra_dims_names=["extra"]
    )

    sliced = a[:, :]
    assert (sliced == a).all()
    assert isinstance(sliced, DynamicArray)
    assert sliced.max_n == 10
    assert sliced.n == 3
    assert sliced.extra_dims_names == ["extra"]

    sliced = a[:, 0]
    assert (sliced == np.array([1, 3, 5])).all()
    assert isinstance(sliced, DynamicArray)
    assert sliced.max_n == 10
    assert sliced.n == 3
    assert sliced.extra_dims_names.size == 0

    sliced = a[0, :]
    assert (sliced == np.array([1, 2])).all()
    assert isinstance(sliced, np.ndarray)

    sliced = a[[True, False, True], :]
    assert (sliced == np.array([[1, 2], [5, 6]])).all()
    assert isinstance(sliced, np.ndarray)

    sliced = a[np.array([True, False, True]), :]
    assert (sliced == np.array([[1, 2], [5, 6]])).all()
    assert isinstance(sliced, np.ndarray)

    sliced = a[:, [True, False]]
    assert (sliced == np.array([[1], [3], [5]])).all()
    assert isinstance(sliced, DynamicArray)
    assert sliced.max_n == 10
    assert sliced.n == 3
    assert sliced.extra_dims_names == ["extra"]

    sliced = a[:]
    assert (sliced == a).all()
    assert isinstance(sliced, DynamicArray)
    assert sliced.max_n == 10
    assert sliced.n == 3
    assert sliced.extra_dims_names == ["extra"]


def test_dynamic_array_copy() -> None:
    """Test the copy functionality of DynamicArray.

    Verifies that copying creates an independent instance with the same data
    and metadata. Tests that modifications to the original do not affect the copy.
    """
    a = DynamicArray(
        np.array([[1, 2], [3, 4], [5, 6]]), max_n=10, extra_dims_names=["extra"]
    )

    copied = a.copy()
    assert (copied == a).all()
    assert isinstance(copied, DynamicArray)
    assert copied.max_n == 10
    assert copied.n == 3
    assert copied.extra_dims_names == ["extra"]

    # Test that modifying the original does not affect the copy
    a[0, 0] = 99
    assert copied[0, 0] == 1


def test_dynamic_array_operations() -> None:
    """Test comprehensive DynamicArray operations and functionality.

    Tests initialization, arithmetic operations (addition, subtraction, multiplication,
    division, power), comparison operations, array methods, reshaping, size management,
    and various numpy ufunc operations. Verifies that operations preserve metadata
    and behave correctly with scalars and other arrays.
    """
    # Test initialization with max_n
    a = DynamicArray(np.array([1, 2, 3]), max_n=10)
    a_ = DynamicArray(dtype=np.int64, n=3, max_n=10)
    a_[:3] = np.array([1, 2, 3])
    assert np.array_equal(a, a_)

    assert np.isin(a, np.array([1, 3])).sum() == 2

    assert (a == 2).sum() == 1
    assert (a != 2).sum() == 2

    assert (a > 2).sum() == 1
    assert (a >= 2).sum() == 2
    assert (a < 2).sum() == 1
    assert (a <= 2).sum() == 2

    assert np.array(a) is not a
    assert isinstance(np.array(a), np.ndarray)
    assert np.array(a).size == 3

    # test reshaping (to not Agent array)
    assert a.reshape(-1, 1).shape == (3, 1)

    assert a.max_n == 10 == (a * 10).max_n

    assert np.array_equal(a, np.array([1, 2, 3]))
    assert a.max_n == 10
    assert a.n == 3

    # Test addition with scalar
    result = a + 1
    assert np.array_equal(result, np.array([2, 3, 4]))

    # Test addition with array
    b = np.array([1, 2, 3])
    result = a + b
    assert np.array_equal(result, np.array([2, 4, 6]))

    # Test multiplication with scalar
    result = a * 2
    assert np.array_equal(result, np.array([2, 4, 6]))

    # Test multiplication with array
    result = a * b
    assert np.array_equal(result, np.array([1, 4, 9]))

    # Test subtraction with scalar
    result = a - 1
    assert np.array_equal(result, np.array([0, 1, 2]))

    # Test subtraction with array
    result = a - b
    assert np.array_equal(result, np.array([0, 0, 0]))

    # Test division with scalar
    result = a / 2
    assert np.array_equal(result, np.array([0.5, 1, 1.5]))

    # Test division with array
    result = a / b
    assert np.array_equal(result, np.array([1, 1, 1]))

    # Test power with scalar
    result = a**2
    assert np.array_equal(result, np.array([1, 4, 9]))

    # Test power with array
    result = a**b
    assert np.array_equal(result, np.array([1, 4, 27]))

    # Test floor division with scalar
    result = a // 3
    assert np.array_equal(result, np.array([0, 0, 1]))

    # Test floor division with array
    result = a // b
    assert np.array_equal(result, np.array([1, 1, 1]))

    # Test modulo with scalar
    result = a % 2
    assert np.array_equal(result, np.array([1, 0, 1]))

    # Test modulo with array
    result = a % b
    assert np.array_equal(result, np.array([0, 0, 0]))

    # Test addition with scalar
    a += 1
    assert np.array_equal(a, np.array([2, 3, 4]))

    # Test addition with array
    b = np.array([1, 2, 3])
    a += b
    assert np.array_equal(a, np.array([3, 5, 7]))

    # Test multiplication with scalar
    a *= 2
    assert np.array_equal(a, np.array([6, 10, 14]))

    # Test multiplication with array
    a *= b
    assert np.array_equal(a, np.array([6, 20, 42]))

    # Test subtraction with scalar
    a -= 1
    assert np.array_equal(a, np.array([5, 19, 41]))

    # Test subtraction with array
    a -= b
    assert np.array_equal(a, np.array([4, 17, 38]))

    # Test division with scalar
    a /= 2
    assert np.array_equal(a, np.array([2, 8, 19]))

    # Test division with array
    a /= b
    assert np.array_equal(a, np.array([2, 4, 6]))

    # Test power with scalar
    a **= 2
    assert np.array_equal(a, np.array([4, 16, 36]))

    # Test power with array
    a **= b
    assert np.array_equal(a, np.array([4, 256, 46656]))

    # Test floor division with scalar
    a //= 3
    assert np.array_equal(a, np.array([1, 85, 15552]))

    # Test floor division with array
    a //= b
    assert np.array_equal(a, np.array([1, 42, 5184]))

    # Test modulo with scalar
    a %= 25
    assert np.array_equal(a, np.array([1, 17, 9]))

    # Test modulo with array
    a %= b
    assert np.array_equal(a, np.array([0, 1, 0]))

    # Test item assignment
    a[0] = 4
    a[1] = 5
    a[2] = 7
    assert np.array_equal(a, np.array([4, 5, 7]))

    # Test slicing
    assert np.array_equal(a[:2], np.array([4, 5]))

    # Test array methods
    assert np.array_equal(
        np.unique(a, return_counts=True), (np.array([4, 5, 7]), np.array([1, 1, 1]))
    )
    assert a.sum() == 16
    assert np.array_equal(a.mean(), 16 / 3)
    assert a.std() == np.std(a)
    assert a.min() == np.min(a)
    assert a.max() == np.max(a)

    # Test setting n
    a.n = 2
    assert np.array_equal(a, np.array([4, 5]))
    assert a.n == 2

    # Test setting n and adding new items
    a.n = 3
    a[2] = 6
    assert np.array_equal(a, np.array([4, 5, 6]))
    assert a.n == 3

    # Test setting n and adding new items beyond max_n
    a.n = 4
    a[3] = 7
    assert np.array_equal(a, np.array([4, 5, 6, 7]))
    assert a.n == 4
    assert a.max_n == 10

    # Test setting n to exceed max_n
    try:
        a.n = 11
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

    # Test size property
    assert a.size == 4

    a.fill(42)
    assert np.array_equal(a, np.array([42, 42, 42, 42]))

    # test that numba edits data in-place
    @njit
    def numba_function(data: npt.NDArray[np.integer]) -> None:
        data[:] = -99

    numba_function(a.data)
    assert (a == -99).all()

    # test ~ unary operator
    a = np.zeros(10, dtype=bool)
    assert ~a.all()

    # test + and - unary operators
    a = np.ones(10, dtype=np.int32)
    assert (+a == 1).all()
    assert (-a == -1).all()

    assert a[0] == 1


@pytest.fixture
def array() -> DynamicArray:
    """Fixture that provides a DynamicArray for testing.

    Returns:
        A DynamicArray instance with sample data.
    """
    return DynamicArray(np.array([1, 2, 3, 4, 5]), max_n=10)


def test_add_ufunc(array: DynamicArray) -> None:
    """Test add ufunc on DynamicArray.

    Args:
        array: The DynamicArray to be tested.
    """
    result = np.add(array, 1)
    np.testing.assert_array_equal(result.data, np.array([2, 3, 4, 5, 6]))


def test_subtract_ufunc(array: DynamicArray) -> None:
    """Test subtract ufunc on DynamicArray.

    Args:
        array: The DynamicArray to be tested.
    """
    result = np.subtract(array, 1)
    np.testing.assert_array_equal(result.data, np.array([0, 1, 2, 3, 4]))


def test_multiply_ufunc(array: DynamicArray) -> None:
    """Test multiply ufunc on DynamicArray.

    Args:
        array: The DynamicArray to be tested.
    """
    result = np.multiply(array, 2)
    np.testing.assert_array_equal(result.data, np.array([2, 4, 6, 8, 10]))


def test_divide_ufunc(array: DynamicArray) -> None:
    """Test divide ufunc on DynamicArray.

    Args:
        array: The DynamicArray to be tested.
    """
    result = np.divide(array, 2)
    np.testing.assert_array_equal(result.data, np.array([0.5, 1.0, 1.5, 2.0, 2.5]))


def test_power_ufunc(array: DynamicArray) -> None:
    """Test power ufunc on DynamicArray.

    Args:
        array: The DynamicArray to be tested.
    """
    result = np.power(array, 2)
    np.testing.assert_array_equal(result.data, np.array([1, 4, 9, 16, 25]))


def test_reduce_ufunc(array: DynamicArray) -> None:
    """Test reduction ufuncs on DynamicArray.

    Args:
        array: The DynamicArray to be tested.
    """
    result = np.add.reduce(array)
    assert result == 15


def test_stack_dynamic_arrays() -> None:
    """Test stacking multiple DynamicArrays using np.stack.

    Verifies that np.stack works correctly with a list of DynamicArrays,
    returning a numpy array with the expected shape and values.
    """
    d1 = DynamicArray(np.array([1, 2, 3]), max_n=10)
    d2 = DynamicArray(np.array([4, 5, 6]), max_n=10)

    stacked = np.stack([d1, d2])

    assert isinstance(stacked, np.ndarray)
    assert stacked.shape == (2, 3)
    np.testing.assert_array_equal(stacked, np.array([[1, 2, 3], [4, 5, 6]]))


def test_concatenate_dynamic_arrays() -> None:
    """Test concatenating multiple DynamicArrays using np.concatenate.

    Verifies that np.concatenate works correctly with a list of DynamicArrays.
    """
    d1 = DynamicArray(np.array([1, 2, 3]), max_n=10)
    d2 = DynamicArray(np.array([4, 5, 6]), max_n=10)

    concatenated = np.concatenate([d1, d2])

    assert isinstance(concatenated, np.ndarray)
    assert concatenated.shape == (6,)
    np.testing.assert_array_equal(concatenated, np.array([1, 2, 3, 4, 5, 6]))


def test_save_and_restore(array: DynamicArray) -> None:
    """Test saving to disk and restoring a DynamicArray.

    Makes a round trip to disk and checks for equality.

    Args:
        array: The DynamicArray to be saved and restored.
    """
    array.save(tmp_folder / "test")
    array2 = DynamicArray.load(tmp_folder / "test.dynamicarray.zarr")
    assert np.array_equal(array, array2)
    # test for equality of class attributes
    assert array.max_n == array2.max_n
    assert array.n == array2.n
    assert (array.extra_dims_names == array2.extra_dims_names).all()
    shutil.rmtree(tmp_folder / "test.dynamicarray.zarr")


def test_dynamic_array_where() -> None:
    """Test np.where usage with DynamicArray.

    Verifies that np.where(condition, x, y) returns a DynamicArray when inputs are
    DynamicArrays, and that the result preserves properties like max_n.
    """
    da1 = DynamicArray(np.array([1, 2, 3]), max_n=10)
    da2 = DynamicArray(np.array([10, 20, 30]), max_n=10)
    cond = DynamicArray(np.array([True, False, True]), max_n=10)

    # Test where with DynamicArray condition
    res = np.where(cond, da1, da2)
    assert isinstance(res, DynamicArray)
    assert np.array_equal(res.data, np.array([1, 20, 3]))
    assert res.max_n == 10
    assert res.n == 3

    # Test with numpy array condition
    res_np_cond = np.where(cond.data, da1, da2)
    assert isinstance(res_np_cond, DynamicArray)
    assert np.array_equal(res_np_cond.data, np.array([1, 20, 3]))
    assert res_np_cond.max_n == 10

    # Test with mixed scalar/array
    res_scalar = np.where(cond, da1, 0)
    assert isinstance(res_scalar, DynamicArray)
    assert np.array_equal(res_scalar.data, np.array([1, 0, 3]))
    assert res_scalar.max_n == 10


def test_dynamic_array_min_max() -> None:
    """Test np.minimum and np.maximum with DynamicArray.

    Verifies that np.minimum and np.maximum work correctly with DynamicArray,
    supporting both DynamicArray and scalar/ndarray inputs as first or second
    arguments, and handling the 'out' parameter.
    """
    da1 = DynamicArray(np.array([10, 2, 30]), max_n=10)
    da2 = DynamicArray(np.array([5, 15, 20]), max_n=10)

    # Test between two DynamicArrays
    res_min = np.minimum(da1, da2)
    assert isinstance(res_min, DynamicArray)
    np.testing.assert_array_equal(res_min.data, np.array([5, 2, 20]))

    # Test with scalar as first argument
    res_scalar = np.minimum(5, da1)
    assert isinstance(res_scalar, DynamicArray)
    np.testing.assert_array_equal(res_scalar.data, np.array([5, 2, 5]))

    # Test with ndarray as first argument
    arr = np.array([5, 15, 20])
    res_arr = np.minimum(arr, da1)
    assert isinstance(res_arr, DynamicArray)
    np.testing.assert_array_equal(res_arr.data, np.array([5, 2, 20]))

    # Test np.maximum
    res_max = np.maximum(da1, da2)
    assert isinstance(res_max, DynamicArray)
    np.testing.assert_array_equal(res_max.data, np.array([10, 15, 30]))


def test_store_checkpoint_metadata() -> None:
    """Test that Store saves checkpoint.json metadata and loads correctly."""
    import datetime
    import json

    from geb.store import Store

    class DummyModel:
        def __init__(self) -> None:
            self.current_time = datetime.datetime(1985, 6, 15, 0, 0, 0)
            self.run_name = "test_run"
            self.in_spinup = False
            self.simulate_hydrology = True
            self.config = {"hazards": {"floods": {"simulate": False}}}
            self.logger = type(
                "DummyLogger",
                (),
                {"debug": lambda *args: None, "info": lambda *args: None},
            )()

        def get_checkpoint_path(self, dt: datetime.datetime | None = None) -> Path:
            if dt is None:
                dt = self.current_time
            return tmp_folder / "checkpoints" / dt.strftime("%Y-%m-%d")

    model = DummyModel()
    from typing import Any, cast

    from geb.model import GEBModel

    store = Store(cast(GEBModel, model))
    bucket = store.create_bucket("var")
    bucket.my_val = np.array([10, 20, 30])

    checkpoint_dir = model.get_checkpoint_path()
    store.save(checkpoint_dir)

    metadata_path = checkpoint_dir / "checkpoint.json"
    assert metadata_path.exists()
    with open(metadata_path) as f:
        meta = json.load(f)
    assert meta["timestamp"] == "1985-06-15T00:00:00"
    assert meta["run_name"] == "test_run"
    assert meta["in_spinup"] is False

    # Load store from checkpoint_dir
    new_model = DummyModel()
    new_store = Store(cast(GEBModel, new_model))
    new_store.load(checkpoint_dir)

    assert hasattr(new_model, "var")
    var_bucket: Any = getattr(new_model, "var")
    assert np.array_equal(var_bucket.my_val, np.array([10, 20, 30]))

    shutil.rmtree(checkpoint_dir)


def test_model_checkpoint_helpers(tmp_path: Path) -> None:
    """Test GEBModel checkpoint path generation, listing, and resolution."""
    import datetime
    import json

    from geb.model import GEBModel

    # Mock GEBModel minimal configuration
    config = {
        "general": {
            "name": "default",
            "spinup_name": "spinup",
            "simulation_root": str(tmp_path / "simulation_root"),
            "spinup_time": "1970-01-01",
            "start_time": "1980-01-01",
            "end_time": "1990-01-01",
            "hazards": {"floods": {"simulate": False}},
        }
    }

    class MockModel:
        def __init__(self) -> None:
            self.config = config
            self.run_name = "default"
            self.in_spinup = False
            self.current_time = datetime.datetime(1980, 1, 1, 0, 0, 0)
            self.timestep_length = datetime.timedelta(days=1)
            self.logger = type(
                "DummyLogger",
                (),
                {"debug": lambda *args: None, "info": lambda *args: None},
            )()

        checkpoints_folder = GEBModel.checkpoints_folder
        spinup_checkpoints_folder = GEBModel.spinup_checkpoints_folder
        get_checkpoints_folder = GEBModel.get_checkpoints_folder
        format_timestamp = GEBModel.format_timestamp
        parse_checkpoint_timestamp = GEBModel.parse_checkpoint_timestamp
        get_checkpoint_path = GEBModel.get_checkpoint_path
        list_checkpoints = GEBModel.list_checkpoints
        get_latest_checkpoint = GEBModel.get_latest_checkpoint
        _resolve_checkpoint_to_load = GEBModel._resolve_checkpoint_to_load

    m = MockModel()

    # Test format_timestamp
    assert m.format_timestamp(datetime.datetime(1980, 1, 1)) == "1980-01-01"
    assert m.format_timestamp(datetime.datetime(1985, 6, 15)) == "1985-06-15"

    # Test get_checkpoint_path
    dt1 = datetime.datetime(1980, 1, 1)
    p1 = m.get_checkpoint_path(dt1)
    assert (
        p1
        == Path(config["general"]["simulation_root"])
        / "default"
        / "checkpoints"
        / "1980-01-01"
    )

    # Test spinup checkpoint path
    p_spinup = m.get_checkpoint_path(dt1, run_name="spinup")
    assert (
        p_spinup
        == Path(config["general"]["simulation_root"])
        / "spinup"
        / "checkpoints"
        / "1980-01-01"
    )

    # Create dummy checkpoint folders
    p1.mkdir(parents=True, exist_ok=True)
    with open(p1 / "checkpoint.json", "w") as f:
        json.dump({"timestamp": "1980-01-01T00:00:00", "run_name": "default"}, f)

    dt2 = datetime.datetime(1985, 6, 1)
    p2 = m.get_checkpoint_path(dt2)
    p2.mkdir(parents=True, exist_ok=True)
    with open(p2 / "checkpoint.json", "w") as f:
        json.dump({"timestamp": "1985-06-01T00:00:00", "run_name": "default"}, f)

    # Test list_checkpoints
    ckpts = m.list_checkpoints()
    assert len(ckpts) == 2
    assert ckpts[0][0] == dt1
    assert ckpts[1][0] == dt2

    # Test get_latest_checkpoint
    latest = m.get_latest_checkpoint()
    assert latest is not None
    assert latest[0] == dt2
    assert latest[1] == p2

    # Test _resolve_checkpoint_to_load
    res_latest_dt, res_latest_path = m._resolve_checkpoint_to_load("latest")
    assert res_latest_dt == dt2
    assert res_latest_path == p2

    res_date_dt, res_date_path = m._resolve_checkpoint_to_load("1980-01-01")
    assert res_date_dt == dt1
    assert res_date_path == p1

    res_path_dt, res_path_path = m._resolve_checkpoint_to_load(p2)
    assert res_path_dt == dt2
    assert res_path_path == p2
