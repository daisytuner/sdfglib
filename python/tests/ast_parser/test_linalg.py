import numpy as np
import pytest
from docc import *
import os


class TypeFactory:
    def __init__(self, dtype):
        self.dtype = dtype

    def __getitem__(self, shape):
        if not isinstance(shape, tuple):
            shape = (shape,)
        return Annotated[np.ndarray, shape, self.dtype]


float64 = TypeFactory(np.float64)


def test_matmul_operator():
    @program
    def matmul_op(a: float64[10, 10], b: float64[10, 10]) -> float64[10, 10]:
        return a @ b

    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)
    res = matmul_op(a, b)
    assert np.allclose(res, a @ b)


def test_numpy_matmul():
    @program
    def np_matmul(a: float64[10, 10], b: float64[10, 10]) -> float64[10, 10]:
        return np.matmul(a, b)

    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)
    res = np_matmul(a, b)
    assert np.allclose(res, np.matmul(a, b))


def test_numpy_dot_matvec():
    @program
    def np_dot_mv(a: float64[10, 10], b: float64[10]) -> float64[10]:
        return np.dot(a, b)

    a = np.random.rand(10, 10)
    b = np.random.rand(10)
    res = np_dot_mv(a, b)
    assert np.allclose(res, np.dot(a, b))


def test_matmul_slicing():
    @program
    def matmul_slice(a: float64[20, 20], b: float64[20, 20]) -> float64[10, 10]:
        return a[:10, :10] @ b[:10, :10]

    a = np.random.rand(20, 20)
    b = np.random.rand(20, 20)
    res = matmul_slice(a, b)
    assert np.allclose(res, a[:10, :10] @ b[:10, :10])


def test_matmul_broadcasting():
    # (2, 10, 10) @ (2, 10, 10) -> (2, 10, 10)
    @program
    def matmul_broadcast(
        a: float64[2, 10, 10], b: float64[2, 10, 10]
    ) -> float64[2, 10, 10]:
        return np.matmul(a, b)

    a = np.random.rand(2, 10, 10)
    b = np.random.rand(2, 10, 10)
    res = matmul_broadcast(a, b)
    assert np.allclose(res, np.matmul(a, b))


def test_matmul_matvec():
    @program
    def matmul_mv(a: float64[10, 10], b: float64[10]) -> float64[10]:
        return np.matmul(a, b)

    a = np.random.rand(10, 10)
    b = np.random.rand(10)
    res = matmul_mv(a, b)
    assert np.allclose(res, np.matmul(a, b))


def test_dot_product_operator():
    @program
    def dot_op(a: float64[10], b: float64[10]) -> float:
        return a @ b

    a = np.random.rand(10)
    b = np.random.rand(10)
    res = dot_op(a, b)
    assert np.allclose(res, a @ b)


def test_dot_product_slicing_scalar():
    @program
    def dot_slice(a: float64[10], b: float64[10]) -> float:
        return a[:5] @ b[:5]

    a = np.random.rand(10)
    b = np.random.rand(10)
    res = dot_slice(a, b)
    assert np.allclose(res, a[:5] @ b[:5])


def test_numpy_outer():
    @program
    def np_outer(a: float64[10], b: float64[10]) -> float64[10, 10]:
        return np.outer(a, b)

    a = np.random.rand(10)
    b = np.random.rand(10)
    res = np_outer(a, b)
    assert np.allclose(res, np.outer(a, b))


def test_outer_slicing():
    @program
    def outer_slice(a: float64[20], b: float64[20]) -> float64[10, 10]:
        return np.outer(a[:10], b[10:])

    a = np.random.rand(20)
    b = np.random.rand(20)
    res = outer_slice(a, b)
    assert np.allclose(res, np.outer(a[:10], b[10:]))


def test_outer_accumulate():
    @program
    def outer_acc(
        a: float64[10], b: float64[10], C: float64[10, 10]
    ) -> float64[10, 10]:
        C[:] += np.outer(a, b)
        return C

    a = np.random.rand(10)
    b = np.random.rand(10)
    C = np.zeros((10, 10))
    expected = C.copy() + np.outer(a, b)

    res = outer_acc(a, b, C)
    assert np.allclose(res, expected)


def test_outer_double_accumulate():
    @program
    def outer_double_acc(
        a: float64[10],
        b: float64[10],
        c: float64[10],
        d: float64[10],
        C: float64[10, 10],
    ) -> float64[10, 10]:
        C[:] += np.outer(a, b) + np.outer(c, d)
        return C

    a = np.random.rand(10)
    b = np.random.rand(10)
    c_arr = np.random.rand(10)
    d = np.random.rand(10)
    C = np.zeros((10, 10))
    expected = C.copy() + np.outer(a, b) + np.outer(c_arr, d)

    res = outer_double_acc(a, b, c_arr, d, C)
    assert np.allclose(res, expected)


def test_2d_addition():
    @program
    def add_2d(n: int) -> float64[10, 10]:
        a = np.zeros((10, 10), dtype=float)
        b = np.zeros((10, 10), dtype=float)
        a[0, 0] = 1.0
        b[0, 0] = 2.0
        c = a + b
        return c

    res = add_2d(10)
    assert res[0, 0] == 3.0
