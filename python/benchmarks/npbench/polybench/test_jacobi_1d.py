import pytest
import numpy as np
from benchmarks.npbench.harness import run_benchmark, run_pytest

PARAMETERS = {
    "S": {"TSTEPS": 800, "N": 3200},
    "M": {"TSTEPS": 3000, "N": 12000},
    "L": {"TSTEPS": 8500, "N": 34000},
    "paper": {"TSTEPS": 4000, "N": 32000},
}


def initialize(TSTEPS, N, datatype=np.float64):
    A = np.fromfunction(lambda i: (i + 2) / N, (N,), dtype=datatype)
    B = np.fromfunction(lambda i: (i + 3) / N, (N,), dtype=datatype)

    return TSTEPS, A, B


def kernel(TSTEPS, A, B):

    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


@pytest.mark.skip()
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_jacobi_1d(target):
    run_pytest(initialize, kernel, PARAMETERS, target)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "jacobi_1d")
