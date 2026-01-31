"""Tests for unary negation of arrays."""

import docc
import numpy as np
import pytest


def test_negate_1d_array():
    """Test negating a 1D array."""

    @docc.program
    def negate_1d(arr) -> float:
        neg_arr = -arr
        return neg_arr[0]

    arr = np.array([3.0, -2.0, 5.0], dtype=np.float64)
    result = negate_1d(arr)
    assert result == -3.0


def test_negate_2d_intermediate():
    """Test negating a 2D intermediate array from slicing."""

    @docc.program
    def negate_2d_slice(arr, k) -> float:
        # Create 2D slice and negate it
        slice_2d = arr[:, :, k]
        neg_slice = -slice_2d
        return neg_slice[0, 0]

    arr = np.ones((3, 4, 5), dtype=np.float64) * 2.0
    result = negate_2d_slice(arr, 2)
    assert result == -2.0


def test_negate_in_multiplication():
    """Test negation used in multiplication (-arr * other)."""

    @docc.program
    def negate_mul(arr1, arr2) -> float:
        result = -arr1 * arr2
        return result[0]

    arr1 = np.array([2.0, 3.0], dtype=np.float64)
    arr2 = np.array([4.0, 5.0], dtype=np.float64)
    result = negate_mul(arr1, arr2)
    assert result == -8.0  # -2.0 * 4.0


def test_negate_intermediate_slice_expression():
    """Test negating result of slice arithmetic (vadv pattern)."""

    @docc.program
    def negate_slice_expr(wcon, k) -> float:
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * 0.5
        neg_cs = -cs
        return neg_cs[0, 0]

    wcon = np.ones((4, 3, 5), dtype=np.float64)
    result = negate_slice_expr(wcon, 1)
    # 0.25 * (1 + 1) * 0.5 = 0.25, negated = -0.25
    assert result == -0.25


def test_negate_with_subtraction():
    """Test negation combined with subtraction (-arr * (a - b))."""

    @docc.program
    def negate_with_sub(cs, u_stage, k) -> float:
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        return correction_term[0, 0]

    ni, nj, nk = 3, 4, 5
    cs = np.ones((ni, nj), dtype=np.float64) * 0.5
    u_stage = np.zeros((ni, nj, nk), dtype=np.float64)
    u_stage[:, :, 2] = 1.0  # k+1
    u_stage[:, :, 1] = 0.5  # k

    result = negate_with_sub(cs, u_stage, 1)
    # -0.5 * (1.0 - 0.5) = -0.5 * 0.5 = -0.25
    assert result == -0.25


def test_negate_zeros():
    """Test negating an array of zeros."""

    @docc.program
    def negate_zeros(arr) -> float:
        neg_arr = -arr
        return neg_arr[0]

    arr = np.zeros(3, dtype=np.float64)
    result = negate_zeros(arr)
    assert result == 0.0


def test_negate_mixed_signs():
    """Test negating an array with mixed positive/negative values."""

    @docc.program
    def negate_mixed(arr) -> float:
        neg_arr = -arr
        # Sum first two elements
        return neg_arr[0] + neg_arr[1]

    arr = np.array([3.0, -2.0, 5.0], dtype=np.float64)
    result = negate_mixed(arr)
    # -3.0 + 2.0 = -1.0
    assert result == -1.0


def test_double_negate():
    """Test double negation returns original value."""

    @docc.program
    def double_negate(arr) -> float:
        neg1 = -arr
        neg2 = -neg1
        return neg2[0]

    arr = np.array([5.0], dtype=np.float64)
    result = double_negate(arr)
    assert result == 5.0


def test_negate_in_loop():
    """Test array negation inside a loop."""

    @docc.program
    def negate_in_loop(wcon) -> float:
        ni = wcon.shape[0] - 1
        nj = wcon.shape[1]
        nk = wcon.shape[2]
        result = np.zeros((ni, nj), dtype=wcon.dtype)

        for k in range(1, 3):
            gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
            result[:, :] = result[:, :] + gav

        return result[0, 0]

    wcon = np.ones((4, 3, 5), dtype=np.float64)
    result = negate_in_loop(wcon)
    # Each iteration: -0.25 * (1 + 1) = -0.5
    # 2 iterations: -0.5 + -0.5 = -1.0
    assert result == -1.0
