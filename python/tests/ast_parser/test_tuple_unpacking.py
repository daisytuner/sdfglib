import docc
import pytest
import numpy as np


def test_tuple_unpacking_simple():
    """Test simple tuple unpacking with constants."""

    @docc.program
    def unpack_simple() -> int:
        a, b = 1, 2
        return a + b

    assert unpack_simple() == 3


def test_tuple_unpacking_three_values():
    """Test unpacking three values."""

    @docc.program
    def unpack_three() -> int:
        a, b, c = 1, 2, 3
        return a + b + c

    assert unpack_three() == 6


def test_tuple_unpacking_from_shape():
    """Test unpacking array shape attributes."""

    @docc.program
    def unpack_shape(arr):
        I, J, K = arr.shape[0], arr.shape[1], arr.shape[2]
        return I + J + K

    arr = np.zeros((10, 20, 30))
    assert unpack_shape(arr) == 60


def test_tuple_unpacking_mixed():
    """Test unpacking with mixed expressions."""

    @docc.program
    def unpack_mixed(arr, x) -> int:
        a, b = arr.shape[0], x + 1
        return a + b

    arr = np.zeros((10, 20))
    assert unpack_mixed(arr, 5) == 16


def test_tuple_unpacking_with_computation():
    """Test that unpacked values can be used in computation."""

    @docc.program
    def use_unpacked(arr) -> int:
        I, J = arr.shape[0], arr.shape[1]
        return I * J

    arr = np.zeros((10, 20), dtype=np.float64)
    assert use_unpacked(arr) == 200
