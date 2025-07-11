import os

import numpy as np
import pytest

from geb.store import DynamicArray

from .testconfig import tmp_folder


def test_1D_dynamic_array_slice():
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


def test_2D_dynamic_array_slice():
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


def test_dynamic_array_copy():
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


def test_dynamic_array_operations():
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
    def numba_function(data):
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
def array():
    return DynamicArray(np.array([1, 2, 3, 4, 5]), max_n=10)


def test_add_ufunc(array):
    result = np.add(array, 1)
    np.testing.assert_array_equal(result.data, np.array([2, 3, 4, 5, 6]))


def test_subtract_ufunc(array):
    result = np.subtract(array, 1)
    np.testing.assert_array_equal(result.data, np.array([0, 1, 2, 3, 4]))


def test_multiply_ufunc(array):
    result = np.multiply(array, 2)
    np.testing.assert_array_equal(result.data, np.array([2, 4, 6, 8, 10]))


def test_divide_ufunc(array):
    result = np.divide(array, 2)
    np.testing.assert_array_equal(result.data, np.array([0.5, 1.0, 1.5, 2.0, 2.5]))


def test_power_ufunc(array):
    result = np.power(array, 2)
    np.testing.assert_array_equal(result.data, np.array([1, 4, 9, 16, 25]))


def test_reduce_ufunc(array):
    result = np.add.reduce(array)
    assert result == 15


def test_save_and_restore(array):
    array.save(tmp_folder / "test")
    array2 = DynamicArray.load(tmp_folder / "test.storearray.npz")
    assert np.array_equal(array, array2)
    # test for equality of class attributes
    assert array.max_n == array2.max_n
    assert array.n == array2.n
    assert (array.extra_dims_names == array2.extra_dims_names).all()
    os.remove(tmp_folder / "test.storearray.npz")


if __name__ == "__main__":
    pass
