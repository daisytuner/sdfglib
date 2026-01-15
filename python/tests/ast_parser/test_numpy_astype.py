import docc
import numpy as np


def test_astype_float64_to_int64():
    """Test casting from float64 to int64"""

    @docc.program
    def cast_to_int(A, B):
        B_casted = A.astype(np.int64)
        for i in range(A.shape[0]):
            B[i] = B_casted[i]

    N = 10
    A = np.array([1.1, 2.9, 3.5, 4.2, 5.8, 6.1, 7.7, 8.3, 9.9, 10.5], dtype=np.float64)
    B = np.zeros(N, dtype=np.int64)

    cast_to_int(A, B)
    expected = A.astype(np.int64)
    assert np.array_equal(B, expected), f"Expected {expected}, got {B}"


def test_astype_int64_to_float64():
    """Test casting from int64 to float64"""

    @docc.program
    def cast_to_float(A, B):
        B_casted = A.astype(np.float64)
        for i in range(A.shape[0]):
            B[i] = B_casted[i]

    N = 10
    A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
    B = np.zeros(N, dtype=np.float64)

    cast_to_float(A, B)
    expected = A.astype(np.float64)
    assert np.allclose(B, expected), f"Expected {expected}, got {B}"


def test_astype_float64_to_float32():
    """Test casting from float64 to float32"""

    @docc.program
    def cast_to_float32(A, B):
        B_casted = A.astype(np.float32)
        for i in range(A.shape[0]):
            B[i] = B_casted[i]

    N = 10
    A = np.random.rand(N).astype(np.float64)
    B = np.zeros(N, dtype=np.float32)

    cast_to_float32(A, B)
    expected = A.astype(np.float32)
    assert np.allclose(B, expected, rtol=1e-5), f"Expected {expected}, got {B}"


def test_astype_2d_array():
    """Test casting 2D arrays"""

    @docc.program
    def cast_2d(A, B):
        B_casted = A.astype(np.int32)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                B[i, j] = B_casted[i, j]

    N, M = 5, 5
    A = np.random.rand(N, M).astype(np.float64) * 10.0
    B = np.zeros((N, M), dtype=np.int32)

    cast_2d(A, B)
    expected = A.astype(np.int32)
    assert np.array_equal(B, expected), f"Expected {expected}, got {B}"


def test_astype_int32_to_int64():
    """Test casting from int32 to int64"""

    @docc.program
    def cast_int32_to_int64(A, B):
        B_casted = A.astype(np.int64)
        for i in range(A.shape[0]):
            B[i] = B_casted[i]

    N = 10
    A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    B = np.zeros(N, dtype=np.int64)

    cast_int32_to_int64(A, B)
    expected = A.astype(np.int64)
    assert np.array_equal(B, expected), f"Expected {expected}, got {B}"
