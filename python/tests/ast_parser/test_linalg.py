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
