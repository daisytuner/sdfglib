from docc.python import native
import ctypes
import numpy as np


def test_ndarray_shape_1d():
    @native
    def ndarray_shape_1d(A, B, C):
        for i in range(A.shape[0]):
            C[i] = A[i] + B[i]

    N = 1024
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(N, dtype=np.float64)

    ndarray_shape_1d(A, B, C)
    assert np.allclose(C, A + B)

    # Check shape arguments
    compiled = ndarray_shape_1d.compile(A, B, C)
    int64_count = sum(1 for t in compiled.arg_types if t == ctypes.c_int64)
    assert int64_count == 1
    assert len(compiled.arg_types) == 4


def test_ndarray_shape_2d_uniform():
    @native
    def ndarray_shape_2d_uniform(A, B, C):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                C[i, j] = A[i, j] + B[i, j]

    N, M = 32, 32
    A = np.random.rand(N, M).astype(np.float64)
    B = np.random.rand(N, M).astype(np.float64)
    C = np.zeros((N, M), dtype=np.float64)

    ndarray_shape_2d_uniform(A, B, C)
    assert np.allclose(C, A + B)

    # Check shape arguments
    compiled = ndarray_shape_2d_uniform.compile(A, B, C)
    int64_count = sum(1 for t in compiled.arg_types if t == ctypes.c_int64)
    assert int64_count == 1
    assert len(compiled.arg_types) == 4


def test_ndarray_shape_mixed():
    @native
    def ndarray_shape_mixed(A, B):
        for i in range(A.shape[0]):
            B[i] = A[i]

    N = 512
    M = 1024
    A = np.random.rand(N).astype(np.float64)
    B = np.zeros(M, dtype=np.float64)

    ndarray_shape_mixed(A, B)
    assert np.allclose(B[:N], A)
    assert np.allclose(B[N:], 0)

    # Check shape arguments
    compiled = ndarray_shape_mixed.compile(A, B)
    int64_count = sum(1 for t in compiled.arg_types if t == ctypes.c_int64)
    assert int64_count == 2
    assert len(compiled.arg_types) == 4
