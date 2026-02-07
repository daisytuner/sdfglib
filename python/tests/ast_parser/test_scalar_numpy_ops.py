"""
Tests for scalar (0-D array) numpy operations.

These tests cover patterns that were previously failing in the Python frontend,
particularly:
1. np.sqrt on scalar array elements (e.g., np.sqrt(A[0, 0]))
2. 0-D array temporaries created by numpy operations
3. Proper handling of pointer dereference for 0-D arrays
"""

import sys
from docc.python import native
import pytest
import numpy as np
import math


class TestScalarSqrt:
    """Test np.sqrt on scalar values and array elements."""

    def test_numpy_sqrt_scalar_element(self):
        """Test np.sqrt on a single array element."""

        @native
        def sqrt_element(A):
            A[0, 0] = np.sqrt(A[0, 0])

        A = np.array([[4.0, 1.0], [1.0, 1.0]])
        sqrt_element(A)
        assert abs(A[0, 0] - 2.0) < 1e-10

    def test_numpy_sqrt_scalar_element_assign_to_different(self):
        """Test np.sqrt and assign to different location."""

        @native
        def sqrt_assign(A):
            A[1, 1] = np.sqrt(A[0, 0])

        A = np.array([[9.0, 1.0], [1.0, 0.0]])
        sqrt_assign(A)
        assert abs(A[1, 1] - 3.0) < 1e-10

    def test_numpy_sqrt_chain(self):
        """Test chained sqrt operations."""

        @native
        def sqrt_chain(A):
            A[0, 0] = np.sqrt(A[0, 0])
            A[1, 1] = np.sqrt(A[1, 1])

        A = np.array([[16.0, 1.0], [1.0, 81.0]])
        sqrt_chain(A)
        assert abs(A[0, 0] - 4.0) < 1e-10
        assert abs(A[1, 1] - 9.0) < 1e-10


class TestScalarUnaryOps:
    """Test various scalar unary operations."""

    def test_numpy_exp_scalar(self):
        """Test np.exp on a single array element."""

        @native
        def exp_element(A):
            A[0, 0] = np.exp(A[0, 0])

        A = np.array([[1.0, 0.0], [0.0, 0.0]])
        exp_element(A)
        assert abs(A[0, 0] - math.e) < 1e-10

    def test_numpy_abs_scalar(self):
        """Test np.abs on a single array element."""

        @native
        def abs_element(A):
            A[0, 0] = np.abs(A[0, 0])

        A = np.array([[-5.0, 0.0], [0.0, 0.0]])
        abs_element(A)
        assert abs(A[0, 0] - 5.0) < 1e-10

    def test_numpy_tanh_scalar(self):
        """Test np.tanh on a single array element."""

        @native
        def tanh_element(A):
            A[0, 0] = np.tanh(A[0, 0])

        A = np.array([[0.0, 1.0], [1.0, 1.0]])
        tanh_element(A)
        assert abs(A[0, 0] - 0.0) < 1e-10


class TestScalarOpsInLoop:
    """Test scalar operations inside loops."""

    def test_sqrt_in_loop(self):
        """Test np.sqrt inside a loop."""

        @native
        def sqrt_loop(A):
            for i in range(A.shape[0]):
                A[i, i] = np.sqrt(A[i, i])

        A = np.diag([4.0, 9.0, 16.0, 25.0]).astype(np.float64)
        sqrt_loop(A)
        expected = np.diag([2.0, 3.0, 4.0, 5.0])
        assert np.allclose(np.diag(A), np.diag(expected))

    def test_sqrt_with_arithmetic(self):
        """Test np.sqrt combined with arithmetic operations."""

        @native
        def sqrt_arithmetic(A):
            for i in range(1, A.shape[0]):
                A[i, i] = np.sqrt(A[i, i] - A[i - 1, i - 1])

        A = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 13.0]])
        A_ref = A.copy()

        # Reference calculation (note: A[1,1] changes, then A[2,2] uses the new value)
        for i in range(1, 3):
            A_ref[i, i] = np.sqrt(A_ref[i, i] - A_ref[i - 1, i - 1])

        sqrt_arithmetic(A)
        assert np.allclose(A, A_ref)


class TestScalarWithDotProduct:
    """Test scalar operations combined with dot products (cholesky pattern)."""

    def test_dot_then_sqrt(self):
        """Test np.dot followed by np.sqrt (partial cholesky pattern)."""

        @native
        def dot_sqrt(A):
            A[1, 1] -= np.dot(A[1, :1], A[1, :1])
            A[1, 1] = np.sqrt(A[1, 1])

        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        A_ref = A.copy()

        # Reference calculation
        A_ref[1, 1] -= np.dot(A_ref[1, :1], A_ref[1, :1])
        A_ref[1, 1] = np.sqrt(A_ref[1, 1])

        dot_sqrt(A)
        assert np.allclose(A, A_ref)

    @pytest.mark.skipif(
        sys.platform == "darwin", reason="Instrumentation not supported on macOS"
    )
    def test_full_cholesky_pattern(self):
        """Test the full Cholesky-like pattern."""

        @native
        def cholesky_like(A):
            A[0, 0] = np.sqrt(A[0, 0])
            for i in range(1, A.shape[0]):
                for j in range(i):
                    A[i, j] -= np.dot(A[i, :j], A[j, :j])
                    A[i, j] /= A[j, j]
                A[i, i] -= np.dot(A[i, :i], A[i, :i])
                A[i, i] = np.sqrt(A[i, i])

        # Create a small positive definite matrix
        N = 4
        A = np.eye(N, dtype=np.float64)
        for i in range(N):
            for j in range(i):
                A[i, j] = 0.5
                A[j, i] = 0.5
        A = A @ A.T  # Make positive definite

        A_ref = A.copy()

        # Reference Cholesky
        A_ref[0, 0] = np.sqrt(A_ref[0, 0])
        for i in range(1, N):
            for j in range(i):
                A_ref[i, j] -= np.dot(A_ref[i, :j], A_ref[j, :j])
                A_ref[i, j] /= A_ref[j, j]
            A_ref[i, i] -= np.dot(A_ref[i, :i], A_ref[i, :i])
            A_ref[i, i] = np.sqrt(A_ref[i, i])

        cholesky_like(A)
        assert np.allclose(A, A_ref)


class TestZeroDimensionalArrays:
    """Test that 0-D arrays are handled correctly."""

    def test_0d_sqrt_result(self):
        """Test that sqrt of a scalar produces correct result."""

        @native
        def zero_d_sqrt(A) -> float:
            tmp = np.sqrt(A[0, 0])
            return tmp

        A = np.array([[9.0, 0.0], [0.0, 0.0]])
        result = zero_d_sqrt(A)
        assert abs(result - 3.0) < 1e-10

    def test_0d_intermediate(self):
        """Test 0-D intermediate values."""

        @native
        def intermediate_sqrt(A):
            tmp = np.sqrt(A[0, 0])
            A[1, 1] = tmp * 2.0

        A = np.array([[4.0, 0.0], [0.0, 0.0]])
        intermediate_sqrt(A)
        assert abs(A[1, 1] - 4.0) < 1e-10
