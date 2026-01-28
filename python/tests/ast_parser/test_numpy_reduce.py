import pytest
import numpy as np
from docc import *
from typing import Annotated

import scipy.special


class TypeFactory:
    def __init__(self, dtype):
        self.dtype = dtype

    def __getitem__(self, shape):
        if not isinstance(shape, tuple):
            shape = (shape,)
        return Annotated[np.ndarray, shape, self.dtype]


float64 = TypeFactory(np.float64)


def test_sum_simple():
    @program
    def sum_simple(
        A: Annotated[np.ndarray, (10, 10), np.float64],
    ) -> Annotated[np.ndarray, (10,), np.float64]:
        return np.sum(A, axis=0)

    A = np.ones((10, 10))
    res = sum_simple(A)
    assert np.allclose(res, np.sum(A, axis=0))


def test_sum_keepdims():
    @program
    def sum_keepdims(
        A: Annotated[np.ndarray, (10, 10), np.float64],
    ) -> Annotated[np.ndarray, (10, 1), np.float64]:
        return np.sum(A, axis=1, keepdims=True)

    A = np.ones((10, 10))
    res = sum_keepdims(A)
    assert np.allclose(res, np.sum(A, axis=1, keepdims=True))


def test_sum_all():
    @program
    def sum_all(A: Annotated[np.ndarray, (10, 10), np.float64]) -> float:
        return np.sum(A)

    A = np.ones((10, 10))
    res = sum_all(A)
    assert np.allclose(res, np.sum(A))


def test_mean_simple():
    @program
    def mean_simple(
        A: Annotated[np.ndarray, (10, 10), np.float64],
    ) -> Annotated[np.ndarray, (10,), np.float64]:
        return np.mean(A, axis=0)

    A = np.random.rand(10, 10)
    res = mean_simple(A)
    assert np.allclose(res, np.mean(A, axis=0))


def test_mean_all():
    @program
    def mean_all(A: Annotated[np.ndarray, (10, 10), np.float64]) -> float:
        return np.mean(A)

    A = np.random.rand(10, 10)
    res = mean_all(A)
    assert np.allclose(res, np.mean(A))


def test_std_simple():
    @program
    def std_simple(
        A: Annotated[np.ndarray, (10, 10), np.float64],
    ) -> Annotated[np.ndarray, (10,), np.float64]:
        return np.std(A, axis=0)

    A = np.random.rand(10, 10)
    res = std_simple(A)
    assert np.allclose(res, np.std(A, axis=0))


def test_std_all():
    @program
    def std_all(A: Annotated[np.ndarray, (10, 10), np.float64]) -> float:
        return np.std(A)

    A = np.random.rand(10, 10)
    res = std_all(A)
    assert np.allclose(res, np.std(A))


def test_max_simple():
    @program
    def max_simple(
        A: Annotated[np.ndarray, (10, 10), np.float64],
    ) -> Annotated[np.ndarray, (10,), np.float64]:
        return np.max(A, axis=0)

    A = np.random.rand(10, 10)
    res = max_simple(A)
    assert np.allclose(res, np.max(A, axis=0))


def test_max_all():
    @program
    def max_all(A: Annotated[np.ndarray, (10, 10), np.float64]) -> float:
        return np.max(A)

    A = np.random.rand(10, 10)
    res = max_all(A)
    assert np.allclose(res, np.max(A))


def test_min_simple():
    @program
    def min_simple(
        A: Annotated[np.ndarray, (10, 10), np.float64],
    ) -> Annotated[np.ndarray, (10,), np.float64]:
        return np.min(A, axis=0)

    A = np.random.rand(10, 10)
    res = min_simple(A)
    assert np.allclose(res, np.min(A, axis=0))


def test_min_all():
    @program
    def min_all(A: Annotated[np.ndarray, (10, 10), np.float64]) -> float:
        return np.min(A)

    A = np.random.rand(10, 10)
    res = min_all(A)
    assert np.allclose(res, np.min(A))


def test_softmax_simple():
    def numpy_softmax(x, axis=None):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @program
    def softmax_simple(
        A: Annotated[np.ndarray, (10, 10), np.float64],
    ) -> Annotated[np.ndarray, (10, 10), np.float64]:
        return scipy.special.softmax(A, axis=0)

    A = np.random.rand(10, 10).astype(np.float64)
    A_ = A.copy()
    res = softmax_simple(A)
    assert np.allclose(res, numpy_softmax(A_, axis=0))


def test_softmax_all():
    def numpy_softmax(x, axis=None):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @program
    def softmax_all(
        A: Annotated[np.ndarray, (10, 10), np.float64],
    ) -> Annotated[np.ndarray, (10, 10), np.float64]:
        return scipy.special.softmax(A)

    A = np.random.rand(10, 10).astype(np.float64)
    A_ = A.copy()
    res = softmax_all(A)
    assert np.allclose(res, numpy_softmax(A_))
