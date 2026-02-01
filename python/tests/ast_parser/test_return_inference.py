from docc.python import native
import numpy as np
import pytest


def test_no_annotation_return_scalar():
    print("Testing scalar return without annotation...")

    @native
    def func(a: int, b: int):
        return a + b

    res = func(1, 2)
    print(f"Result: {res}")
    assert res == 3


def test_no_annotation_return_array():
    print("Testing array return without annotation...")

    @native
    def func(A):
        return A

    A = np.ones((10,))
    res = func(A)
    print(f"Result type: {type(res)}")
    assert np.allclose(res, A)


def test_matmul_return():
    print("Testing matmul return...")

    @native
    def func(A, x):
        return A @ x

    M, N = 10, 10
    A = np.ones((M, N))
    x = np.ones((N,))
    res = func(A, x)
    print(f"Result type: {type(res)}")

    assert np.allclose(res, A @ x)


if __name__ == "__main__":
    try:
        test_no_annotation_return_scalar()
        print("Scalar test passed")
    except Exception as e:
        print(f"Scalar test failed: {e}")

    try:
        test_no_annotation_return_array()
        print("Array test passed")
    except Exception as e:
        print(f"Array test failed: {e}")

    try:
        test_matmul_return()
        print("Matmul test passed")
    except Exception as e:
        print(f"Matmul test failed: {e}")
