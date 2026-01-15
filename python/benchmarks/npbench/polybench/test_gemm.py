import pytest
import docc
import numpy as np
from benchmarks.npbench.harness import run_benchmark, run_pytest

PARAMETERS = {
    "S": { "NI": 1000, "NJ": 1100, "NK": 1200 },
    "M": { "NI": 2500, "NJ": 2750, "NK": 3000 },
    "L": { "NI": 7000, "NJ": 7500, "NK": 8000 },
    "paper": { "NI": 2000, "NJ": 2300, "NK": 2600 }
}

def initialize(NI, NJ, NK, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI, (NI, NJ),
                        dtype=datatype)
    A = np.fromfunction(lambda i, k: (i * (k + 1) % NK) / NK, (NI, NK),
                        dtype=datatype)
    B = np.fromfunction(lambda k, j: (k * (j + 2) % NJ) / NJ, (NK, NJ),
                        dtype=datatype)

    return alpha, beta, C, A, B


def kernel(alpha, beta, C, A, B):
    C[:] = alpha * A @ B + beta * C

@pytest.mark.parametrize("target", ["none", "sequential", "openmp"])
def test_gemm(target):
    run_pytest(initialize, kernel, PARAMETERS, target)

if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "gemm")
