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
    fn = datatype(N)
    x = np.fromfunction(lambda i: 1 + (i / fn), (N,), dtype=datatype)
    A = np.fromfunction(lambda i, j: ((i + j) % N) / (5 * M), (M, N), dtype=datatype)

    return A, x


def kernel(A, x):

    return (A @ x) @ A


@pytest.mark.parametrize("target", ["none", "sequential", "openmp"])
def test_atax(target):
    run_pytest(initialize, kernel, PARAMETERS, target=target)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "atax")
