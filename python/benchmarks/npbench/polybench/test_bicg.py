import pytest
import numpy as np
from benchmarks.npbench.harness import run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 4000, "N": 5000},
    "M": {"M": 10000, "N": 12500},
    "L": {"M": 20000, "N": 25000},
    "paper": {"M": 18000, "N": 22000},
}


def initialize(M, N, datatype=np.float64):
    A = np.fromfunction(lambda i, j: (i * (j + 1) % N) / N, (N, M), dtype=datatype)
    p = np.fromfunction(lambda i: (i % M) / M, (M,), dtype=datatype)
    r = np.fromfunction(lambda i: (i % N) / N, (N,), dtype=datatype)

    return A, p, r


def kernel(A, p, r):

    return r @ A, A @ p


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_bicg(target):
    run_pytest(initialize, kernel, PARAMETERS, target)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "bicg")
