import numpy as np
import pytest
from docc import *


class TypeFactory:
    def __init__(self, dtype):
        self.dtype = dtype

    def __getitem__(self, shape):
        if not isinstance(shape, tuple):
            shape = (shape,)
        return Annotated[np.ndarray, shape, self.dtype]


float64 = TypeFactory(np.float64)


def test_zeros_like():
    @program
    def zeros_like_test(a: float64[10, 10]) -> float64[10, 10]:
        return np.zeros_like(a)

    a = np.random.rand(10, 10)
    res = zeros_like_test(a)
    assert np.allclose(res, np.zeros_like(a))
    assert res.shape == (10, 10)
    assert np.all(res == 0)


def test_zeros_like_dtype():
    @program
    def zeros_like_dtype(a: float64[5, 5]) -> float64[5, 5]:
        return np.zeros_like(a, dtype=np.float64)

    a = np.random.rand(5, 5)
    res = zeros_like_dtype(a)
    assert np.allclose(res, np.zeros_like(a, dtype=np.float64))
    assert res.dtype == np.float64
