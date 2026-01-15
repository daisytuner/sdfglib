import docc
import numpy as np
import pytest


def test_transpose_T():
    @docc.program
    def transpose_T(n: int, m: int) -> float:
        a = np.zeros((n, m), dtype=float)
        # Initialize
        for i in range(n):
            for j in range(m):
                a[i, j] = i * m + j

        b = a.T
        return b[0, 1]  # Should be a[1, 0] = 1*m + 0 = m

    n = 4
    m = 5
    expected = (np.arange(n * m).reshape(n, m).T)[0, 1]
    assert transpose_T(n, m) == expected


def test_transpose_func():
    @docc.program
    def transpose_func(n: int, m: int) -> float:
        a = np.zeros((n, m), dtype=float)
        for i in range(n):
            for j in range(m):
                a[i, j] = i + j

        b = np.transpose(a)
        return b[1, 0]  # Should be a[0, 1] = 1

    n = 3
    m = 3
    expected = 1.0
    assert transpose_func(n, m) == expected


def test_transpose_axes():
    @docc.program
    def transpose_axes(n: int, m: int) -> float:
        a = np.zeros((n, m), dtype=float)
        for i in range(n):
            for j in range(m):
                a[i, j] = i * 10 + j

        # Identity transpose for 2D with axes=(0,1)
        # Transpose with axes=(1,0) is T
        b = np.transpose(a, axes=(1, 0))
        return b[m - 1, n - 1]

    n = 4
    m = 5
    val = transpose_axes(n, m)
    # b[m-1, n-1] corresponds to a[n-1, m-1]
    # = (n-1)*10 + (m-1) = 30 + 4 = 34
    assert val == 34.0
