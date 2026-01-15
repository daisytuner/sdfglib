import docc
import os
import shutil
import pytest
import numpy as np


def test_compile_if():
    @docc.program
    def if_test(a) -> int:
        if a > 10:
            return 1
        return 0

    assert if_test(20) == 1
    assert if_test(5) == 0


def test_compile_if_else():
    @docc.program
    def if_else_test(a) -> int:
        if a > 10:
            return 1
        else:
            return 0

    assert if_else_test(20) == 1
    assert if_else_test(5) == 0


def test_compile_while():
    @docc.program
    def while_test(a) -> int:
        while a > 10:
            return 1
        return 0

    assert while_test(20) == 1
    assert while_test(5) == 0


def test_compile_for():
    @docc.program
    def for_test(n) -> int:
        s = 0
        for i in range(n):
            s = s + i
        return s

    assert for_test(10) == 45


def test_array_loop():
    @docc.program
    def array_loop(A, n):
        for i in range(n):
            A[i] = A[i] + 1

    arr = np.zeros(10, dtype=np.int32)
    array_loop(arr, 10)

    expected = np.ones(10, dtype=np.int32)
    np.testing.assert_array_equal(arr, expected)
