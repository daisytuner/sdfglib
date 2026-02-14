from docc.python import native
import os
import shutil
import pytest
import numpy as np


def test_compile_if():
    @native
    def if_test(a) -> int:
        if a > 10:
            return 1
        return 0

    assert if_test(20) == 1
    assert if_test(5) == 0


def test_compile_if_else():
    @native
    def if_else_test(a) -> int:
        if a > 10:
            return 1
        else:
            return 0

    assert if_else_test(20) == 1
    assert if_else_test(5) == 0


def test_compile_while():
    @native
    def while_test(a) -> int:
        while a > 10:
            return 1
        return 0

    assert while_test(20) == 1
    assert while_test(5) == 0


def test_compile_for():
    @native
    def for_test(n) -> int:
        s = 0
        for i in range(n):
            s = s + i
        return s

    assert for_test(10) == 45


def test_array_loop():
    @native
    def array_loop(A, n):
        for i in range(n):
            A[i] = A[i] + 1

    arr = np.zeros(10, dtype=np.int32)
    array_loop(arr, 10)

    expected = np.ones(10, dtype=np.int32)
    np.testing.assert_array_equal(arr, expected)


def test_negative_step_loop():
    @native
    def negative_step_loop(n):
        res = 0
        for i in range(n - 1, -1, -1):
            res = res + i
        return res

    # sum(0..9) = 45
    assert negative_step_loop(10) == 45


def test_negative_step_loop_array():
    @native
    def negative_step_loop_array(n):
        A = np.zeros((n,), dtype=np.int64)
        for i in range(n - 1, -1, -1):
            A[i] = i
        return A

    res = negative_step_loop_array(5)
    np.testing.assert_array_equal(res, [0, 1, 2, 3, 4])


def test_positive_step_loop():
    @native
    def positive_step_loop(n):
        res = 0
        for i in range(0, n, 1):
            res = res + i
        return res

    res = positive_step_loop(10)
    assert res == 45


def test_step_greater_than_one():
    @native
    def step_greater_than_one(n):
        res = 0
        for i in range(0, n, 2):
            res = res + i
        return res

    res = step_greater_than_one(10)
    # 0 + 2 + 4 + 6 + 8 = 20
    assert res == 20


def test_slice_assignment_loop_var():
    @native
    def slice_assignment(n):
        A = np.zeros((n, n), dtype=np.float64)
        B = np.ones((n, n), dtype=np.float64)

        # Pattern from ADI: A[row, slice] = B[row, slice]
        for i in range(n):
            A[i, 1 : n - 1] = B[i, 1 : n - 1]

        return A

    res = slice_assignment(10)
    expected = np.zeros((10, 10))
    expected[:, 1:9] = 1.0
    np.testing.assert_array_equal(res, expected)


def test_reverse_loop_dependency():
    @native
    def reverse_dep(n):
        A = np.zeros(n, dtype=np.float64)
        # Init A with 1.0 at end
        A[n - 1] = 1.0

        # Propagate backwards: A[i] = A[i+1]
        for i in range(n - 2, -1, -1):
            A[i] = A[i + 1]

        return A

    res = reverse_dep(10)
    expected = np.ones(10)
    np.testing.assert_array_equal(res, expected)
