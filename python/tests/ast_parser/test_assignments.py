from docc.python import native
import numpy as np


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
