import docc
import pytest
import numpy as np
import math


def test_numpy_add():
    @docc.program
    def numpy_add_float(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[0] = 1.0
        b[0] = 2.0
        c = a + b
        return c[0]

    assert numpy_add_float(10) == 3.0

    @docc.program
    def numpy_add_int(n) -> int:
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        a[0] = 1
        b[0] = 2
        c = a + b
        return c[0]

    assert numpy_add_int(10) == 3


def test_numpy_sub():
    @docc.program
    def numpy_sub_float(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[0] = 3.0
        b[0] = 1.0
        c = np.subtract(a, b)
        return c[0]

    assert numpy_sub_float(10) == 2.0

    @docc.program
    def numpy_sub_int(n) -> int:
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        a[0] = 3
        b[0] = 1
        c = np.subtract(a, b)
        return c[0]

    assert numpy_sub_int(10) == 2


def test_numpy_mul():
    @docc.program
    def numpy_mul_float(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[0] = 2.0
        b[0] = 3.0
        c = np.multiply(a, b)
        return c[0]

    assert numpy_mul_float(10) == 6.0

    @docc.program
    def numpy_mul_int(n) -> int:
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        a[0] = 2
        b[0] = 3
        c = np.multiply(a, b)
        return c[0]

    assert numpy_mul_int(10) == 6


def test_numpy_mul_scalar():
    @docc.program
    def numpy_mul_scalar(n) -> float:
        a = np.zeros(n, dtype=float)
        a[0] = 2.0
        c = a * 3.0
        return c[0]

    assert numpy_mul_scalar(10) == 6.0


def test_numpy_div():
    @docc.program
    def numpy_div_float(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[:] = 1.0
        b[:] = 1.0

        a[0] = 6.0
        b[0] = 2.0
        c = np.divide(a, b)
        return c[0]

    assert numpy_div_float(10) == 3.0

    @docc.program
    def numpy_div_int(n) -> int:
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        a[:] = 1
        b[:] = 1

        a[0] = 6
        b[0] = 4
        c = np.divide(a, b)
        return c[0]

    assert numpy_div_int(10) == 1  # Integer division


def test_numpy_pow():
    @docc.program
    def numpy_pow_float(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[0] = 2.0
        b[0] = 3.0
        c = np.power(a, b)
        return c[0]

    assert numpy_pow_float(10) == 8.0

    # @docc.program
    # def numpy_pow_int(n) -> int:
    #     a = np.zeros(n, dtype=int)
    #     b = np.zeros(n, dtype=int)
    #     a[0] = 2
    #     b[0] = 3
    #     c = np.power(a, b)
    #     return c[0]

    # assert numpy_pow_int(10) == 8


def test_numpy_abs():
    @docc.program
    def numpy_abs_float(n) -> float:
        a = np.zeros(n, dtype=float)
        a[0] = -5.0
        c = np.abs(a)
        return c[0]

    assert numpy_abs_float(10) == 5.0

    @docc.program
    def numpy_abs_int(n) -> int:
        a = np.zeros(n, dtype=int)
        a[0] = -5
        c = np.abs(a)
        return c[0]

    assert numpy_abs_int(10) == 5


def test_numpy_sqrt():
    @docc.program
    def numpy_sqrt(n) -> float:
        a = np.zeros(n, dtype=float)
        a[0] = 9.0
        c = np.sqrt(a)
        return c[0]

    assert numpy_sqrt(10) == 3.0


def test_numpy_tanh():
    @docc.program
    def numpy_tanh(n) -> float:
        a = np.zeros(n, dtype=float)
        a[0] = 0.0
        c = np.tanh(a)
        return c[0]

    assert numpy_tanh(10) == 0.0


def test_numpy_op_operator():
    @docc.program
    def numpy_op_operator(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[0] = 2.0
        b[0] = 3.0
        c = a * b
        d = c / a
        e = d**2.0
        return e[0]

    assert numpy_op_operator(10) == 9.0


def test_numpy_exp():
    @docc.program
    def numpy_exp(n) -> float:
        a = np.zeros(n, dtype=float)
        a[0] = 1.0
        c = np.exp(a)
        return c[0]

    assert abs(numpy_exp(10) - math.e) < 1e-6


def test_numpy_minimum():
    @docc.program
    def numpy_minimum_float(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[0] = 1.0
        b[0] = 2.0
        c = np.minimum(a, b)
        return c[0]

    assert numpy_minimum_float(10) == 1.0

    @docc.program
    def numpy_minimum_int(n) -> int:
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        a[0] = 1
        b[0] = 2
        c = np.minimum(a, b)
        return c[0]

    assert numpy_minimum_int(10) == 1


def test_numpy_maximum():
    @docc.program
    def numpy_maximum_float(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[0] = 1.0
        b[0] = 2.0
        c = np.maximum(a, b)
        return c[0]

    assert numpy_maximum_float(10) == 2.0

    @docc.program
    def numpy_maximum_int(n) -> int:
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        a[0] = 1
        b[0] = 2
        c = np.maximum(a, b)
        return c[0]

    assert numpy_maximum_int(10) == 2
