import pytest
import docc
import numpy as np
from benchmarks.npbench.harness import run_benchmark, run_pytest

PARAMETERS = {
    "S": { "NI": 800, "NJ": 850, "NK": 900, "NL": 950 },
    "M": { "NI": 2000, "NJ": 2250, "NK": 2500, "NL": 2750 },
    "L": { "NI": 6000, "NJ": 6500, "NK": 7000, "NL": 7500 },
    "paper": { "NI": 3200, "NJ": 3600, "NK": 4400, "NL": 4800 }
}

def initialize(NI, NJ, NK, NL, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI, (NI, NK),
                        dtype=datatype)
    B = np.fromfunction(lambda i, j: (i * (j + 1) % NJ) / NJ, (NK, NJ),
                        dtype=datatype)
    C = np.fromfunction(lambda i, j: ((i * (j + 3) + 1) % NL) / NL, (NJ, NL),
                        dtype=datatype)
    D = np.fromfunction(lambda i, j: (i * (j + 2) % NK) / NK, (NI, NL),
                        dtype=datatype)

    return alpha, beta, A, B, C, D


def kernel(alpha, beta, A, B, C, D):
    D[:] = alpha * A @ B @ C + beta * D

@pytest.mark.parametrize("target", ["none", "sequential", "openmp"])
def test_k2mm(target):
    run_pytest(initialize, kernel, PARAMETERS, target)

if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "k2mm")
