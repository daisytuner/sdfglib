import pytest
import numpy as np
from npbench.harness import run_benchmark, run_pytest

PARAMETERS = {"S": {"N": 200}, "M": {"N": 400}, "L": {"N": 850}, "paper": {"N": 2800}}


def initialize(N, datatype=np.int32):
    path = np.fromfunction(lambda i, j: i * j % 7 + 1, (N, N), dtype=datatype)
    for i in range(N):
        for j in range(N):
            if (i + j) % 13 == 0 or (i + j) % 7 == 0 or (i + j) % 11 == 0:
                path[i, j] = 999

    return path


def kernel(path):
    for k in range(path.shape[0]):
        path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))


@pytest.mark.skip()
@pytest.mark.parametrize("target", ["none", "sequential", "openmp"])
def test_floyd_warshall(target):
    run_pytest(initialize, kernel, PARAMETERS, target)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "floyd_warshall")
