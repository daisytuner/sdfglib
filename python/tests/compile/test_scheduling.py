import docc
import numpy as np


def test_scheduling_default():
    @docc.program
    def vec_add_default(A, B, C, N):
        for i in range(N):
            C[i] = A[i] + B[i]

    N = 1024
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(N, dtype=np.float64)

    vec_add_default(A, B, C, N)
    assert np.allclose(C, A + B)


def test_scheduling_sequential():
    @docc.program(target="sequential", category="desktop")
    def vec_add(A, B, C, N):
        for i in range(N):
            C[i] = A[i] + B[i]

    N = 1024
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(N, dtype=np.float64)

    vec_add(A, B, C, N)
    assert np.allclose(C, A + B)


def test_scheduling_openmp():
    # Assuming OpenMP is available and supported
    @docc.program(target="openmp", category="desktop")
    def vec_add_omp(A, B, C, N):
        for i in range(N):
            C[i] = A[i] + B[i]

    N = 1024
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(N, dtype=np.float64)

    vec_add_omp(A, B, C, N)
    assert np.allclose(C, A + B)
