import pytest
import numpy as np
from benchmarks.npbench.harness import run_benchmark, run_pytest

PARAMETERS = {
    "S": {"N": 1000},
    "M": {"N": 2200},
    "L": {"N": 8000},
    "paper": {"N": 2000},
}


def initialize(N, datatype=np.float64):
    A = np.empty((N, N), dtype=datatype)
    for i in range(N):
        A[i, : i + 1] = np.fromfunction(
            lambda j: (-j % N) / N + 1, (i + 1,), dtype=datatype
        )
        A[i, i + 1 :] = 0.0
        A[i, i] = 1.0
    A[:] = A @ np.transpose(A)

    return (A,)


def kernel(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)


@pytest.mark.skip()
@pytest.mark.parametrize("target", ["none", "sequential", "openmp"])
def test_cholesky2(target):
    run_pytest(initialize, kernel, PARAMETERS, target)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "cholesky2")
