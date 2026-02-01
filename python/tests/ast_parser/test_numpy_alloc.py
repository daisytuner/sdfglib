from docc.compiler import native
import pytest
import numpy as np


def test_numpy_empty():
    @native
    def alloc_empty(n) -> float:
        a = np.empty(n, dtype=float)
        a[0] = 1.0
        return a[0]

    assert alloc_empty(10) == 1.0


def test_numpy_zeros():
    @native
    def alloc_zeros(n) -> float:
        a = np.zeros(n, dtype=float)
        return a[0]

    assert alloc_zeros(10) == 0.0


def test_numpy_zeros_2d():
    @native
    def alloc_zeros_2d(n) -> float:
        a = np.zeros((n, n), dtype=float)
        return a[0, 0]

    assert alloc_zeros_2d(10) == 0.0


def test_numpy_eye_simple():
    @native
    def alloc_eye(n) -> float:
        a = np.eye(n, dtype=float)
        return a[0, 0]

    assert alloc_eye(10) == 1.0


def test_numpy_eye_off_diagonal():
    @native
    def alloc_eye_k(n) -> float:
        a = np.eye(n, k=1, dtype=float)
        return a[0, 1]

    assert alloc_eye_k(10) == 1.0


def test_numpy_eye_rectangular():
    @native
    def alloc_eye_rect(n) -> float:
        a = np.eye(n, M=n + 2, dtype=float)
        return a[0, 0]

    assert alloc_eye_rect(10) == 1.0


def test_numpy_eye_none_m():
    @native
    def alloc_eye_none(n) -> float:
        a = np.eye(n, M=None, dtype=float)
        return a[0, 0]

    assert alloc_eye_none(10) == 1.0


def test_numpy_ones():
    @native
    def alloc_ones(n) -> float:
        a = np.ones(n, dtype=float)
        return a[0]

    assert alloc_ones(10) == 1.0


def test_numpy_ones_2d():
    @native
    def alloc_ones_2d(n) -> float:
        a = np.ones((n, n), dtype=float)
        return a[0, 0]

    assert alloc_ones_2d(10) == 1.0


def test_numpy_ones_int():
    @native
    def alloc_ones_int(n) -> int:
        a = np.ones(n, dtype=int)
        return a[0]

    assert alloc_ones_int(10) == 1


def test_dtype_from_array():
    @native
    def dtype_from_array(A, n):
        # Should infer dtype from A
        B = np.empty(n, dtype=A.dtype)
        # Verify B has same type by writing A's value
        B[0] = A[0]
        return B[0]

    A = np.array([3.14], dtype=np.float64)
    res = dtype_from_array(A, 10)
    assert res == 3.14

    A_int = np.array([42], dtype=np.int64)
    res_int = dtype_from_array(A_int, 10)
    assert res_int == 42
