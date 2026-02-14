from docc.python import native
import numpy as np


def test_tuple_unpacking_simple():
    """Test simple tuple unpacking with constants."""

    @native
    def unpack_simple() -> int:
        a, b = 1, 2
        return a + b

    assert unpack_simple() == 3


def test_tuple_unpacking_three_values():
    """Test unpacking three values."""

    @native
    def unpack_three() -> int:
        a, b, c = 1, 2, 3
        return a + b + c

    assert unpack_three() == 6


def test_tuple_unpacking_from_shape():
    """Test unpacking array shape attributes."""

    @native
    def unpack_shape(arr):
        I, J, K = arr.shape[0], arr.shape[1], arr.shape[2]
        return I + J + K

    arr = np.zeros((10, 20, 30))
    assert unpack_shape(arr) == 60


def test_tuple_unpacking_mixed():
    """Test unpacking with mixed expressions."""

    @native
    def unpack_mixed(arr, x) -> int:
        a, b = arr.shape[0], x + 1
        return a + b

    arr = np.zeros((10, 20))
    assert unpack_mixed(arr, 5) == 16


def test_tuple_unpacking_with_computation():
    """Test that unpacked values can be used in computation."""

    @native
    def use_unpacked(arr) -> int:
        I, J = arr.shape[0], arr.shape[1]
        return I * J

    arr = np.zeros((10, 20), dtype=np.float64)
    assert use_unpacked(arr) == 200


def test_multiple_assignment_scalar():
    @native
    def multi_assign(in_val: int) -> int:
        a = b = in_val
        return a + b

    assert multi_assign(5) == 10


def test_multiple_assignment_expression():
    @native
    def multi_assign_expr(in_val: int) -> int:
        a = b = in_val + 2
        return a * b

    assert multi_assign_expr(3) == 25


def test_multiple_assignment_array():
    @native
    def multi_assign_array(in_arr):
        a = b = in_arr
        return a + b

    arr = np.array([1, 2, 3], dtype=np.int32)
    res = multi_assign_array(arr)
    expected = arr * 2
    assert np.allclose(res, expected)


def test_chained_assignment_multiple():
    @native
    def chained_assign(val: int) -> int:
        a = b = c = val
        return a + b + c

    assert chained_assign(10) == 30
