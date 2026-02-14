from docc.python import native
import pytest
import numpy as np


def test_numpy_empty_like_simple():
    @native
    def alloc_empty_like(n: int) -> float:
        a = np.empty(n, dtype=float)
        a[0] = 5.0
        b = np.empty_like(a)
        b[0] = a[0]
        return b[0]

    assert alloc_empty_like(10) == 5.0


def test_numpy_empty_like_with_dtype():
    @native
    def alloc_empty_like_dtype(n: int) -> int:
        a = np.empty(n, dtype=float)
        b = np.empty_like(a, dtype=int)
        b[0] = 3
        return b[0]

    assert alloc_empty_like_dtype(10) == 3


def test_numpy_empty_like_multi_dim():
    @native
    def alloc_empty_like_2d(n: int) -> float:
        a = np.empty((n, n), dtype=float)
        a[0, 0] = 2.0
        b = np.empty_like(a)
        b[0, 0] = a[0, 0] * 2.0
        return b[0, 0]

    assert alloc_empty_like_2d(10) == 4.0
