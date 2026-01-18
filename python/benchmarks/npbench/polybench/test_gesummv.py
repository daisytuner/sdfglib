import pytest
import numpy as np
from benchmarks.npbench.harness import run_benchmark, run_pytest

PARAMETERS = {
    "S": {"N": 2000},
    "M": {"N": 4000},
    "L": {"N": 14000},
    "paper": {"N": 11200},
}


def initialize(N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, (N, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * j + 2) % N) / N, (N, N), dtype=datatype)
    x = np.fromfunction(lambda i: (i % N) / N, (N,), dtype=datatype)

    return alpha, beta, A, B, x


def kernel(alpha, beta, A, B, x):
    return alpha * A @ x + beta * B @ x


@pytest.mark.parametrize("target", ["none", "sequential", "openmp"])
def test_gesummv(target):
    run_pytest(initialize, kernel, PARAMETERS, target)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "gesummv")
