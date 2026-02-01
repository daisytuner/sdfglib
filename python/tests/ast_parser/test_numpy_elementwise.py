from docc.python import native
import pytest
import numpy as np
import math


def test_numpy_add():
    @native
    def numpy_add_float(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[0] = 1.0
        b[0] = 2.0
        c = a + b
        return c[0]

    assert numpy_add_float(10) == 3.0

    @native
    def numpy_add_int(n) -> int:
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        a[0] = 1
        b[0] = 2
        c = a + b
        return c[0]

    assert numpy_add_int(10) == 3


def test_numpy_sub():
    @native
    def numpy_sub_float(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[0] = 3.0
        b[0] = 1.0
        c = np.subtract(a, b)
        return c[0]

    assert numpy_sub_float(10) == 2.0

    @native
    def numpy_sub_int(n) -> int:
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        a[0] = 3
        b[0] = 1
        c = np.subtract(a, b)
        return c[0]

    assert numpy_sub_int(10) == 2


def test_numpy_mul():
    @native
    def numpy_mul_float(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[0] = 2.0
        b[0] = 3.0
        c = np.multiply(a, b)
        return c[0]

    assert numpy_mul_float(10) == 6.0

    @native
    def numpy_mul_int(n) -> int:
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        a[0] = 2
        b[0] = 3
        c = np.multiply(a, b)
        return c[0]

    assert numpy_mul_int(10) == 6


def test_numpy_mul_scalar():
    @native
    def numpy_mul_scalar(n) -> float:
        a = np.zeros(n, dtype=float)
        a[0] = 2.0
        c = a * 3.0
        return c[0]

    assert numpy_mul_scalar(10) == 6.0


def test_numpy_div():
    @native
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

    @native
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
    @native
    def numpy_pow_float(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[0] = 2.0
        b[0] = 3.0
        c = np.power(a, b)
        return c[0]

    assert numpy_pow_float(10) == 8.0

    # @native
    # def numpy_pow_int(n) -> int:
    #     a = np.zeros(n, dtype=int)
    #     b = np.zeros(n, dtype=int)
    #     a[0] = 2
    #     b[0] = 3
    #     c = np.power(a, b)
    #     return c[0]

    # assert numpy_pow_int(10) == 8


def test_numpy_abs():
    @native
    def numpy_abs_float(n) -> float:
        a = np.zeros(n, dtype=float)
        a[0] = -5.0
        c = np.abs(a)
        return c[0]

    assert numpy_abs_float(10) == 5.0

    @native
    def numpy_abs_int(n) -> int:
        a = np.zeros(n, dtype=int)
        a[0] = -5
        c = np.abs(a)
        return c[0]

    assert numpy_abs_int(10) == 5


def test_numpy_sqrt():
    @native
    def numpy_sqrt(n) -> float:
        a = np.zeros(n, dtype=float)
        a[0] = 9.0
        c = np.sqrt(a)
        return c[0]

    assert numpy_sqrt(10) == 3.0


def test_numpy_tanh():
    @native
    def numpy_tanh(n) -> float:
        a = np.zeros(n, dtype=float)
        a[0] = 0.0
        c = np.tanh(a)
        return c[0]

    assert numpy_tanh(10) == 0.0


def test_numpy_op_operator():
    @native
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
    @native
    def numpy_exp(n) -> float:
        a = np.zeros(n, dtype=float)
        a[0] = 1.0
        c = np.exp(a)
        return c[0]

    assert abs(numpy_exp(10) - math.e) < 1e-6


def test_numpy_minimum():
    @native
    def numpy_minimum_float(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[0] = 1.0
        b[0] = 2.0
        c = np.minimum(a, b)
        return c[0]

    assert numpy_minimum_float(10) == 1.0

    @native
    def numpy_minimum_int(n) -> int:
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        a[0] = 1
        b[0] = 2
        c = np.minimum(a, b)
        return c[0]

    assert numpy_minimum_int(10) == 1


def test_numpy_maximum():
    @native
    def numpy_maximum_float(n) -> float:
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        a[0] = 1.0
        b[0] = 2.0
        c = np.maximum(a, b)
        return c[0]

    assert numpy_maximum_float(10) == 2.0

    @native
    def numpy_maximum_int(n) -> int:
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        a[0] = 1
        b[0] = 2
        c = np.maximum(a, b)
        return c[0]

    assert numpy_maximum_int(10) == 2


def test_scalar_array_broadcasting_mul():
    """Test scalar * array broadcasting (scalar on left side)."""

    @native
    def scalar_times_array(x: float, n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 1.0
        arr[1] = 2.0
        arr[2] = 3.0
        result = np.sqrt(x) * arr
        return result[1]

    # sqrt(4.0) * 2.0 = 2.0 * 2.0 = 4.0
    assert abs(scalar_times_array(4.0, 5) - 4.0) < 1e-10


def test_scalar_array_broadcasting_add():
    """Test scalar + array broadcasting (scalar on left side)."""

    @native
    def scalar_plus_array(x: float, n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 1.0
        arr[1] = 2.0
        arr[2] = 3.0
        result = np.sqrt(x) + arr
        return result[1]

    # sqrt(4.0) + 2.0 = 2.0 + 2.0 = 4.0
    assert abs(scalar_plus_array(4.0, 5) - 4.0) < 1e-10


def test_array_scalar_broadcasting_mul():
    """Test array * scalar broadcasting (scalar on right side)."""

    @native
    def array_times_scalar(x: float, n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 1.0
        arr[1] = 2.0
        arr[2] = 3.0
        result = arr * np.sqrt(x)
        return result[1]

    # 2.0 * sqrt(4.0) = 2.0 * 2.0 = 4.0
    assert abs(array_times_scalar(4.0, 5) - 4.0) < 1e-10


def test_scalar_array_broadcasting_sub():
    """Test scalar - array broadcasting."""

    @native
    def scalar_minus_array(x: float, n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 1.0
        arr[1] = 2.0
        arr[2] = 3.0
        result = np.sqrt(x) - arr
        return result[0]

    # sqrt(9.0) - 1.0 = 3.0 - 1.0 = 2.0
    assert abs(scalar_minus_array(9.0, 5) - 2.0) < 1e-10


def test_scalar_array_broadcasting_div():
    """Test scalar / array broadcasting."""

    @native
    def scalar_div_array(x: float, n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[:] = 1.0
        arr[0] = 2.0
        arr[1] = 4.0
        result = np.sqrt(x) / arr
        return result[1]

    # sqrt(16.0) / 4.0 = 4.0 / 4.0 = 1.0
    assert abs(scalar_div_array(16.0, 5) - 1.0) < 1e-10


def test_2d_1d_broadcasting_inplace_sub():
    """Test 2D -= 1D broadcasting (row-wise subtraction)."""

    @native
    def array_2d_minus_1d(n, m) -> float:
        data = np.zeros((n, m), dtype=float)
        mean = np.zeros(m, dtype=float)
        # Set up data: row i has values i, i, i, ...
        for i in range(n):
            for j in range(m):
                data[i, j] = float(i)
        # mean = [1.0, 1.0, ...]
        for j in range(m):
            mean[j] = 1.0
        # After data -= mean, row i should have values i-1, i-1, ...
        data -= mean
        return data[2, 0]  # Should be 2.0 - 1.0 = 1.0

    result = array_2d_minus_1d(5, 4)
    assert abs(result - 1.0) < 1e-10


def test_int_scalar_times_float_array():
    """Test integer scalar * float array type promotion (int -> float)."""

    @native
    def int_scalar_times_float_array(n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 2.5
        arr[1] = 3.5
        c = 2  # integer scalar
        result = c * arr
        return result[0]

    # 2 * 2.5 = 5.0
    assert abs(int_scalar_times_float_array(5) - 5.0) < 1e-10


def test_float_array_times_int_scalar():
    """Test float array * integer scalar type promotion (int -> float)."""

    @native
    def float_array_times_int_scalar(n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 2.5
        arr[1] = 3.5
        c = 3  # integer scalar
        result = arr * c
        return result[1]

    # 3.5 * 3 = 10.5
    assert abs(float_array_times_int_scalar(5) - 10.5) < 1e-10


def test_int_scalar_plus_float_array():
    """Test integer scalar + float array type promotion."""

    @native
    def int_scalar_plus_float_array(n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 1.5
        arr[1] = 2.5
        c = 10  # integer scalar
        result = c + arr
        return result[1]

    # 10 + 2.5 = 12.5
    assert abs(int_scalar_plus_float_array(5) - 12.5) < 1e-10


def test_float_array_minus_int_scalar():
    """Test float array - integer scalar type promotion."""

    @native
    def float_array_minus_int_scalar(n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 5.5
        arr[1] = 10.5
        c = 3  # integer scalar
        result = arr - c
        return result[0]

    # 5.5 - 3 = 2.5
    assert abs(float_array_minus_int_scalar(5) - 2.5) < 1e-10


def test_int_scalar_div_float_array():
    """Test integer scalar / float array type promotion."""

    @native
    def int_scalar_div_float_array(n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[:] = 1.0
        arr[0] = 2.0
        arr[1] = 4.0
        c = 8  # integer scalar
        result = c / arr
        return result[1]

    # 8 / 4.0 = 2.0
    assert abs(int_scalar_div_float_array(5) - 2.0) < 1e-10


def test_int_var_times_float_array_sum():
    """Test integer variable * (float array + float array) - mimics deriche pattern."""

    @native
    def int_var_times_array_sum(n) -> float:
        y1 = np.zeros(n, dtype=float)
        y2 = np.zeros(n, dtype=float)
        y1[0] = 1.5
        y1[1] = 2.5
        y2[0] = 0.5
        y2[1] = 1.5
        c1 = 1  # integer, like in deriche: c1 = c2 = 1
        result = c1 * (y1 + y2)
        return result[1]

    # 1 * (2.5 + 1.5) = 4.0
    assert abs(int_var_times_array_sum(5) - 4.0) < 1e-10


def test_chained_int_float_operations():
    """Test chained operations with mixed int/float types."""

    @native
    def chained_int_float_ops(n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 2.0
        arr[1] = 3.0
        a = 2  # int
        b = 3  # int
        # (2 * arr + 3) should promote correctly
        result = a * arr + b
        return result[1]

    # 2 * 3.0 + 3 = 9.0
    assert abs(chained_int_float_ops(5) - 9.0) < 1e-10
