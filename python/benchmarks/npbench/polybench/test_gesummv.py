import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

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


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_gesummv(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "FOR": 6,
                "MAP": 6,
                "SEQUENTIAL": 6,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 2,
                "DOT": 0,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "FOR": 6,
                "MAP": 6,
                "SEQUENTIAL": 2,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 4,
                "GEMM": 2,
                "DOT": 0,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "FOR": 6,
                "MAP": 6,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 4,
                "HIGHWAY": 2,
                "GEMM": 2,
                "DOT": 0,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "FOR": 6,
                "MAP": 6,
                "SEQUENTIAL": 0,
                "CUDA": 6,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 2,
                "DOT": 0,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "gesummv")
