import pytest
import numpy as np
from benchmarks.npbench.harness import run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 500, "N": 600},
    "M": {"M": 1400, "N": 1800},
    "L": {"M": 3200, "N": 4000},
    "paper": {"M": 1200, "N": 1400},
}


def initialize(M, N, datatype=np.float64):
    float_n = datatype(N)
    data = np.fromfunction(lambda i, j: (i * j) / M, (N, M), dtype=datatype)

    return M, float_n, data


def kernel(M, float_n, data):
    return np.cov(np.transpose(data))


@pytest.mark.skip()
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_covariance2(target):
    run_pytest(initialize, kernel, PARAMETERS, target)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "covariance2")
