import pytest
import numpy as np
from npbench.harness import run_benchmark, run_pytest

PARAMETERS = {
    "S": {"N": 2000},
    "M": {"N": 5000},
    "L": {"N": 14000},
    "paper": {"N": 16000},
}


def initialize(N, datatype=np.float64):
    L = np.fromfunction(lambda i, j: (i + N - j + 1) * 2 / N, (N, N), dtype=datatype)
    x = np.full((N,), -999, dtype=datatype)
    b = np.fromfunction(lambda i: i, (N,), dtype=datatype)

    return L, x, b


def kernel(L, x, b):

    for i in range(x.shape[0]):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]


@pytest.mark.skip()
@pytest.mark.parametrize("target", ["none", "sequential", "openmp"])
def test_trisolv(target):
    run_pytest(initialize, kernel, PARAMETERS, target)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "trisolv")
