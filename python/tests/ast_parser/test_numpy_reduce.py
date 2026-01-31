import pytest
import numpy as np
from docc.compiler import native

import scipy.special


def test_sum_simple():
    @native
    def sum_simple(
        A,
    ):
        return np.sum(A, axis=0)

    A = np.ones((10, 10))
    res = sum_simple(A)
    assert np.allclose(res, np.sum(A, axis=0))


def test_sum_keepdims():
    @native
    def sum_keepdims(
        A,
    ):
        return np.sum(A, axis=1, keepdims=True)

    A = np.ones((10, 10))
    res = sum_keepdims(A)
    assert np.allclose(res, np.sum(A, axis=1, keepdims=True))


def test_sum_all():
    @native
    def sum_all(A) -> float:
        return np.sum(A)

    A = np.ones((10, 10))
    res = sum_all(A)
    assert np.allclose(res, np.sum(A))


def test_mean_simple():
    @native
    def mean_simple(
        A,
    ):
        return np.mean(A, axis=0)

    A = np.random.rand(10, 10)
    res = mean_simple(A)
    assert np.allclose(res, np.mean(A, axis=0))


def test_mean_all():
    @native
    def mean_all(A) -> float:
        return np.mean(A)

    A = np.random.rand(10, 10)
    res = mean_all(A)
    assert np.allclose(res, np.mean(A))


def test_std_simple():
    @native
    def std_simple(
        A,
    ):
        return np.std(A, axis=0)

    A = np.random.rand(10, 10)
    res = std_simple(A)
    assert np.allclose(res, np.std(A, axis=0))


def test_std_all():
    @native
    def std_all(A) -> float:
        return np.std(A)

    A = np.random.rand(10, 10)
    res = std_all(A)
    assert np.allclose(res, np.std(A))


def test_max_simple():
    @native
    def max_simple(
        A,
    ):
        return np.max(A, axis=0)

    A = np.random.rand(10, 10)
    res = max_simple(A)
    assert np.allclose(res, np.max(A, axis=0))


def test_max_all():
    @native
    def max_all(A) -> float:
        return np.max(A)

    A = np.random.rand(10, 10)
    res = max_all(A)
    assert np.allclose(res, np.max(A))


def test_min_simple():
    @native
    def min_simple(
        A,
    ):
        return np.min(A, axis=0)

    A = np.random.rand(10, 10)
    res = min_simple(A)
    assert np.allclose(res, np.min(A, axis=0))


def test_min_all():
    @native
    def min_all(A) -> float:
        return np.min(A)

    A = np.random.rand(10, 10)
    res = min_all(A)
    assert np.allclose(res, np.min(A))


def test_softmax_simple():
    def numpy_softmax(x, axis=None):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @native
    def softmax_simple(
        A,
    ):
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

    @native
    def softmax_all(
        A,
    ):
        return scipy.special.softmax(A)

    A = np.random.rand(10, 10).astype(np.float64)
    A_ = A.copy()
    res = softmax_all(A)
    assert np.allclose(res, numpy_softmax(A_))
