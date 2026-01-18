import pytest
import numpy as np
from benchmarks.npbench.harness import run_benchmark, run_pytest

PARAMETERS = {
    "S": {"N": 1000},
    "M": {"N": 6000},
    "L": {"N": 20000},
    "paper": {"N": 16000},
}


def initialize(N, datatype=np.float64):
    r = np.fromfunction(lambda i: N + 1 - i, (N,), dtype=datatype)
    return r


def kernel(r):
    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta
        y[:k] += alpha * np.flip(y[:k])
        y[k] = alpha

    return y


@pytest.mark.skip()
@pytest.mark.parametrize("target", ["none", "sequential", "openmp"])
def test_durbin(target):
    run_pytest(initialize, kernel, PARAMETERS, target)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "durbin")
