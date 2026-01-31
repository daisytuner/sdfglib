import numpy as np
import pytest
from docc.compiler import native


def test_zeros_like():
    @native
    def zeros_like_test(a):
        return np.zeros_like(a)

    a = np.random.rand(10, 10)
    res = zeros_like_test(a)
    assert np.allclose(res, np.zeros_like(a))
    assert res.shape == (10, 10)
    assert np.all(res == 0)


def test_zeros_like_dtype():
    @native
    def zeros_like_dtype(a):
        return np.zeros_like(a, dtype=np.float64)

    a = np.random.rand(5, 5)
    res = zeros_like_dtype(a)
    assert np.allclose(res, np.zeros_like(a, dtype=np.float64))
    assert res.dtype == np.float64
