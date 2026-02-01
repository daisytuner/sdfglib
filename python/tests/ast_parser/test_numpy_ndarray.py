from docc.python import native
import pytest
import numpy as np


def test_numpy_ndarray_basic():
    """Test basic np.ndarray allocation."""

    @native
    def alloc_ndarray(n) -> float:
        a = np.ndarray((n,), dtype=np.float64)
        a[0] = 1.0
        return a[0]

    assert alloc_ndarray(10) == 1.0


def test_numpy_ndarray_2d():
    """Test 2D np.ndarray allocation."""

    @native
    def alloc_ndarray_2d(n, m) -> float:
        a = np.ndarray((n, m), dtype=np.float64)
        a[0, 0] = 42.0
        return a[0, 0]

    assert alloc_ndarray_2d(10, 20) == 42.0


def test_numpy_ndarray_3d():
    """Test 3D np.ndarray allocation."""

    @native
    def alloc_ndarray_3d(arr) -> float:
        a = np.ndarray((arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.float64)
        a[0, 0, 0] = 3.14
        return a[0, 0, 0]

    arr = np.zeros((10, 20, 30), dtype=np.float64)
    assert alloc_ndarray_3d(arr) == 3.14


def test_numpy_ndarray_dtype_from_array():
    """Test np.ndarray with dtype from another array."""

    @native
    def alloc_ndarray_dtype(arr) -> float:
        a = np.ndarray((arr.shape[0], arr.shape[1]), dtype=arr.dtype)
        a[0, 0] = 1.5
        return a[0, 0]

    arr = np.zeros((10, 20), dtype=np.float64)
    assert alloc_ndarray_dtype(arr) == 1.5


def test_numpy_ndarray_int_dtype():
    """Test np.ndarray with integer dtype."""

    @native
    def alloc_ndarray_int(n) -> int:
        a = np.ndarray((n,), dtype=np.int64)
        a[0] = 42
        return a[0]

    assert alloc_ndarray_int(10) == 42


def test_numpy_ndarray_shape_from_variables():
    """Test np.ndarray with shape from unpacked variables - uses direct shape access."""

    @native
    def alloc_with_shape(arr) -> float:
        temp = np.ndarray((arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.float64)
        temp[0, 0, 0] = 2.5
        return temp[0, 0, 0]

    arr = np.zeros((10, 20, 30), dtype=np.float64)
    assert alloc_with_shape(arr) == 2.5
