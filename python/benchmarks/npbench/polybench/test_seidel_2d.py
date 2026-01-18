import pytest
import numpy as np
from benchmarks.npbench.harness import run_benchmark, run_pytest

PARAMETERS = {
    "S": {"TSTEPS": 8, "N": 50},
    "M": {"TSTEPS": 15, "N": 100},
    "L": {"TSTEPS": 40, "N": 200},
    "paper": {"TSTEPS": 100, "N": 400},
}


def initialize(TSTEPS, N, datatype=np.float64):
    A = np.fromfunction(lambda i, j: (i * (j + 2) + 2) / N, (N, N), dtype=datatype)

    return TSTEPS, N, A


def kernel(TSTEPS, N, A):

    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (
                A[i - 1, :-2]
                + A[i - 1, 1:-1]
                + A[i - 1, 2:]
                + A[i, 2:]
                + A[i + 1, :-2]
                + A[i + 1, 1:-1]
                + A[i + 1, 2:]
            )
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0


@pytest.mark.skip()
@pytest.mark.parametrize("target", ["none", "sequential", "openmp"])
def test_seidel_2d(target):
    run_pytest(initialize, kernel, PARAMETERS, target)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "seidel_2d")
