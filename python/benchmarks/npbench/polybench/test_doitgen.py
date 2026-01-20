import pytest
import numpy as np
from benchmarks.npbench.harness import run_benchmark, run_pytest

PARAMETERS = {
    "S": {"NR": 60, "NQ": 60, "NP": 128},
    "M": {"NR": 110, "NQ": 125, "NP": 256},
    "L": {"NR": 220, "NQ": 250, "NP": 512},
    "paper": {"NR": 220, "NQ": 250, "NP": 270},
}


def initialize(NR, NQ, NP, datatype=np.float64):
    A = np.fromfunction(
        lambda i, j, k: ((i * j + k) % NP) / NP, (NR, NQ, NP), dtype=datatype
    )
    C4 = np.fromfunction(lambda i, j: (i * j % NP) / NP, (NP, NP), dtype=datatype)

    return NR, NQ, NP, A, C4


def kernel(NR, NQ, NP, A, C4):
    A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))


@pytest.mark.skip()
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_doitgen(target):
    run_pytest(initialize, kernel, PARAMETERS, target)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "doitgen")
