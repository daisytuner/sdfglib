from docc.compiler import native
import ctypes
import numpy as np


def test_implicit_shape_1d():
    @native
    def vec_add_implicit(A, B, C):
        for i in range(A.shape[0]):
            C[i] = A[i] + B[i]

    N = 1024
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(N, dtype=np.float64)

    vec_add_implicit(A, B, C)
    assert np.allclose(C, A + B)


def test_implicit_shape_2d():
    @native
    def mat_add_implicit(A, B, C):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                C[i, j] = A[i, j] + B[i, j]

    N, M = 32, 32
    A = np.random.rand(N, M).astype(np.float64)
    B = np.random.rand(N, M).astype(np.float64)
    C = np.zeros((N, M), dtype=np.float64)

    mat_add_implicit(A, B, C)
    assert np.allclose(C, A + B)


def test_implicit_shape_mixed():
    @native
    def mixed_args(A, scalar, B):
        for i in range(A.shape[0]):
            B[i] = A[i] + scalar

    N = 100
    A = np.random.rand(N).astype(np.float64)
    B = np.zeros(N, dtype=np.float64)
    scalar = 5.0

    mixed_args(A, scalar, B)
    assert np.allclose(B, A + scalar)


def test_shape_unification_1d():
    @native
    def vec_add(A, B, C):
        for i in range(A.shape[0]):
            C[i] = A[i] + B[i]

    N = 1024
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(N, dtype=np.float64)

    # Compile
    compiled = vec_add.compile(A, B, C)

    # Check arguments
    # Expected: A, B, C (pointers) + 1 shape (int64)
    # Total 4 arguments

    print(f"Arg types: {compiled.arg_types}")

    # Count int64 arguments (shapes are int64)
    int64_count = sum(1 for t in compiled.arg_types if t == ctypes.c_int64)

    # We expect exactly 1 int64 argument (the shared size N)
    assert int64_count == 1
    assert len(compiled.arg_types) == 4


def test_shape_unification_2d():
    @native
    def mat_add(A, B, C):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                C[i, j] = A[i, j] + B[i, j]

    N, M = 32, 64
    A = np.random.rand(N, M).astype(np.float64)
    B = np.random.rand(N, M).astype(np.float64)
    C = np.zeros((N, M), dtype=np.float64)

    compiled = mat_add.compile(A, B, C)

    # Expected: A, B, C + 2 shapes (N, M)
    # Total 5 arguments

    int64_count = sum(1 for t in compiled.arg_types if t == ctypes.c_int64)
    assert int64_count == 2
    assert len(compiled.arg_types) == 5


def test_shape_unification_mismatch():
    @native
    def vec_add_mismatch(A, B):
        for i in range(A.shape[0]):
            B[i] = A[i]

    N = 1024
    M = 512
    A = np.random.rand(N).astype(np.float64)
    B = np.zeros(M, dtype=np.float64)  # Different size

    compiled = vec_add_mismatch.compile(A, B)

    # Expected: A, B + 2 shapes (N, M)
    # Total 4 arguments

    int64_count = sum(1 for t in compiled.arg_types if t == ctypes.c_int64)
    assert int64_count == 2
    assert len(compiled.arg_types) == 4
