"""Tests for NumPy broadcasting in elementwise operations."""

import numpy as np
import pytest
from docc.python import native


class TestBroadcast1DTo2D:
    """Test broadcasting 1D arrays to 2D (common bias addition pattern)."""

    def test_add_1d_bias_to_2d(self):
        """Test adding 1D bias to 2D matrix (row-wise broadcast)."""

        @native
        def add_bias(x, b):
            return x + b

        x = np.random.rand(8, 5)
        b = np.random.rand(5)
        res = add_bias(x, b)
        expected = x + b
        assert np.allclose(res, expected)

    def test_add_1d_bias_to_2d_float32(self):
        """Test adding 1D bias to 2D matrix preserves float32."""

        @native
        def add_bias_f32(x, b):
            return x + b

        x = np.random.rand(8, 5).astype(np.float32)
        b = np.random.rand(5).astype(np.float32)
        res = add_bias_f32(x, b)
        expected = x + b
        assert res.dtype == np.float32, f"Expected float32, got {res.dtype}"
        assert np.allclose(res, expected, rtol=1e-5)

    def test_sub_1d_from_2d(self):
        """Test subtracting 1D array from 2D matrix."""

        @native
        def sub_bias(x, b):
            return x - b

        x = np.random.rand(8, 5)
        b = np.random.rand(5)
        res = sub_bias(x, b)
        expected = x - b
        assert np.allclose(res, expected)

    def test_mul_1d_with_2d(self):
        """Test multiplying 2D matrix by 1D array (scaling)."""

        @native
        def scale(x, s):
            return x * s

        x = np.random.rand(8, 5)
        s = np.random.rand(5)
        res = scale(x, s)
        expected = x * s
        assert np.allclose(res, expected)

    def test_div_2d_by_1d(self):
        """Test dividing 2D matrix by 1D array."""

        @native
        def normalize(x, d):
            return x / d

        x = np.random.rand(8, 5)
        d = np.random.rand(5) + 0.1  # Avoid division by zero
        res = normalize(x, d)
        expected = x / d
        assert np.allclose(res, expected)


class TestBroadcastScalarTo2D:
    """Test broadcasting scalars to 2D arrays."""

    def test_add_scalar_to_2d(self):
        """Test adding scalar to 2D matrix."""

        @native
        def add_scalar(x):
            return x + 1.0

        x = np.random.rand(8, 5)
        res = add_scalar(x)
        expected = x + 1.0
        assert np.allclose(res, expected)

    def test_mul_2d_by_scalar(self):
        """Test multiplying 2D matrix by scalar."""

        @native
        def scale_scalar(x):
            return x * 2.0

        x = np.random.rand(8, 5)
        res = scale_scalar(x)
        expected = x * 2.0
        assert np.allclose(res, expected)


class TestBroadcastColumnVector:
    """Test broadcasting column vectors (N, 1) to 2D.

    Note: Column vector broadcasting requires matching dimensions on left side,
    which is more complex than row broadcasting. These tests are marked as
    expected failures until full NumPy-style broadcasting is implemented.
    """

    @pytest.mark.xfail(reason="Column vector broadcast not yet fully supported")
    def test_add_column_to_2d(self):
        """Test adding column vector to 2D matrix."""

        @native
        def add_column(x, c):
            return x + c

        x = np.random.rand(8, 5)
        c = np.random.rand(8, 1)
        res = add_column(x, c)
        expected = x + c
        assert np.allclose(res, expected)

    @pytest.mark.xfail(reason="Column vector broadcast not yet fully supported")
    def test_sub_column_from_2d(self):
        """Test subtracting column vector from 2D matrix."""

        @native
        def sub_column(x, c):
            return x - c

        x = np.random.rand(8, 5)
        c = np.random.rand(8, 1)
        res = sub_column(x, c)
        expected = x - c
        assert np.allclose(res, expected)


class TestGEMMWithBroadcastBias:
    """Test GEMM + broadcast bias pattern (MLP layers)."""

    def test_matmul_plus_1d_bias(self):
        """Test matmul followed by 1D bias addition."""

        @native
        def linear_layer(x, w, b):
            return x @ w + b

        x = np.random.rand(8, 10)
        w = np.random.rand(10, 5)
        b = np.random.rand(5)
        res = linear_layer(x, w, b)
        expected = x @ w + b
        assert np.allclose(res, expected, rtol=1e-5)

    def test_matmul_plus_1d_bias_float32(self):
        """Test matmul + 1D bias preserves float32."""

        @native
        def linear_layer_f32(x, w, b):
            return x @ w + b

        x = np.random.rand(8, 10).astype(np.float32)
        w = np.random.rand(10, 5).astype(np.float32)
        b = np.random.rand(5).astype(np.float32)
        res = linear_layer_f32(x, w, b)
        expected = x @ w + b
        assert res.dtype == np.float32, f"Expected float32, got {res.dtype}"
        assert np.allclose(res, expected, rtol=1e-4)

    def test_mlp_layer_with_relu(self):
        """Test full MLP layer: matmul + bias + relu."""

        @native
        def mlp_layer(x, w, b):
            y = x @ w + b
            return np.maximum(y, 0)

        x = np.random.rand(8, 10)
        w = np.random.rand(10, 5)
        b = np.random.rand(5)
        res = mlp_layer(x, w, b)
        expected = np.maximum(x @ w + b, 0)
        assert np.allclose(res, expected, rtol=1e-5)

    def test_mlp_layer_with_relu_float32(self):
        """Test MLP layer preserves float32."""

        @native
        def mlp_layer_f32(x, w, b):
            y = x @ w + b
            return np.maximum(y, 0)

        x = np.random.rand(8, 10).astype(np.float32)
        w = np.random.rand(10, 5).astype(np.float32)
        b = np.random.rand(5).astype(np.float32)
        res = mlp_layer_f32(x, w, b)
        expected = np.maximum(x @ w + b, 0)
        assert res.dtype == np.float32, f"Expected float32, got {res.dtype}"
        assert np.allclose(res, expected, rtol=1e-4)


class TestTwoLayerMLP:
    """Test two-layer MLP patterns with broadcasting."""

    def test_two_layer_mlp(self):
        """Test two-layer MLP with broadcast biases."""

        @native
        def two_layer_mlp(x, w1, b1, w2, b2):
            h = np.maximum(x @ w1 + b1, 0)
            return h @ w2 + b2

        x = np.random.rand(8, 10)
        w1 = np.random.rand(10, 20)
        b1 = np.random.rand(20)
        w2 = np.random.rand(20, 5)
        b2 = np.random.rand(5)

        res = two_layer_mlp(x, w1, b1, w2, b2)
        h = np.maximum(x @ w1 + b1, 0)
        expected = h @ w2 + b2
        assert np.allclose(res, expected, rtol=1e-4)

    def test_two_layer_mlp_float32(self):
        """Test two-layer MLP preserves float32."""

        @native
        def two_layer_mlp_f32(x, w1, b1, w2, b2):
            h = np.maximum(x @ w1 + b1, 0)
            return h @ w2 + b2

        x = np.random.rand(8, 10).astype(np.float32)
        w1 = np.random.rand(10, 20).astype(np.float32)
        b1 = np.random.rand(20).astype(np.float32)
        w2 = np.random.rand(20, 5).astype(np.float32)
        b2 = np.random.rand(5).astype(np.float32)

        res = two_layer_mlp_f32(x, w1, b1, w2, b2)
        h = np.maximum(x @ w1 + b1, 0)
        expected = h @ w2 + b2
        assert res.dtype == np.float32, f"Expected float32, got {res.dtype}"
        assert np.allclose(res, expected, rtol=1e-4)


class TestBroadcastMaxMin:
    """Test broadcasting with max/min operations."""

    def test_maximum_with_1d(self):
        """Test np.maximum with 1D threshold array."""

        @native
        def max_threshold(x, t):
            return np.maximum(x, t)

        x = np.random.rand(8, 5)
        t = np.random.rand(5)
        res = max_threshold(x, t)
        expected = np.maximum(x, t)
        assert np.allclose(res, expected)

    def test_minimum_with_1d(self):
        """Test np.minimum with 1D threshold array."""

        @native
        def min_threshold(x, t):
            return np.minimum(x, t)

        x = np.random.rand(8, 5)
        t = np.random.rand(5)
        res = min_threshold(x, t)
        expected = np.minimum(x, t)
        assert np.allclose(res, expected)

    def test_clip_pattern(self):
        """Test clipping pattern with broadcasting."""

        @native
        def clip_values(x, low, high):
            return np.minimum(np.maximum(x, low), high)

        x = np.random.rand(8, 5)
        low = np.random.rand(5) * 0.3
        high = np.random.rand(5) * 0.3 + 0.7
        res = clip_values(x, low, high)
        expected = np.minimum(np.maximum(x, low), high)
        assert np.allclose(res, expected)
