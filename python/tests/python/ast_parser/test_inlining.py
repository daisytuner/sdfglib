from docc.python import native
import numpy as np
import pytest


def helper_add(a, b):
    return a + b


def test_simple_inlining():
    @native
    def simple_inlining(a, b):
        return helper_add(a, b)

    assert simple_inlining(10, 20) == 30


def helper_nested(x):
    return x * 2


def helper_outer(x):
    return helper_nested(x) + 1


def test_nested_inlining():
    @native
    def nested_inlining(x):
        return helper_outer(x)

    assert nested_inlining(10) == 21


def helper_with_local_var(x):
    y = 10
    return x + y


def test_inlining_local_vars():
    @native
    def inlining_local_vars(x):
        y = 5
        return helper_with_local_var(x) + y

    assert inlining_local_vars(20) == 35  # (20 + 10) + 5


def relu_float32(x):
    return np.maximum(x, 0)


def test_inlining_preserves_float32_array():
    @native
    def apply_relu(x):
        return relu_float32(x)

    x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    result = apply_relu(x)
    expected = np.maximum(x, 0)
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    assert np.allclose(result, expected)


def helper_matmul(a, b):
    return a @ b


def test_inlining_matmul_float32():
    @native
    def inlined_matmul(a, b):
        return helper_matmul(a, b)

    a = np.random.rand(10, 10).astype(np.float32)
    b = np.random.rand(10, 10).astype(np.float32)
    result = inlined_matmul(a, b)
    expected = a @ b
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    assert np.allclose(result, expected, rtol=1e-5)
