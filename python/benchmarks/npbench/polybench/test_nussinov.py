import pytest
import numpy as np
from benchmarks.npbench.harness import run_benchmark, run_pytest

PARAMETERS = {"S": {"N": 40}, "M": {"N": 90}, "L": {"N": 200}, "paper": {"N": 500}}


def initialize(N, datatype=np.int32):
    seq = np.fromfunction(lambda i: (i + 1) % 4, (N,), dtype=datatype)

    return N, seq


def match(b1, b2):
    if b1 + b2 == 3:
        return 1
    else:
        return 0


def kernel(N, seq):

    table = np.zeros((N, N), np.int32)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                table[i, j] = max(table[i, j], table[i, j - 1])
            if i + 1 < N:
                table[i, j] = max(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    table[i, j] = max(
                        table[i, j], table[i + 1, j - 1] + match(seq[i], seq[j])
                    )
                else:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j):
                table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])

    return table


@pytest.mark.parametrize("target", ["none", "sequential", "openmp"])
def test_nussinov(target):
    run_pytest(initialize, kernel, PARAMETERS, target)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "nussinov")
