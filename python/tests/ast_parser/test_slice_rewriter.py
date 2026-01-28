import pytest
import numpy as np
import docc
from docc._sdfg import Pointer, Scalar, PrimitiveType


class TestSliceRewriterSubscript:
    """Tests for SliceRewriter handling of subscript expressions."""

    def test_array_element_to_1d_slice(self):
        """Test assigning a scalar array element to a 1D slice: arr[:] = src[0]"""

        @docc.program
        def array_element_to_1d_slice(src: np.ndarray, dst: np.ndarray):
            dst[:] = src[0]

        src = np.array([42.0, 1.0, 2.0], dtype=np.float64)
        dst = np.zeros(5, dtype=np.float64)
        array_element_to_1d_slice(src, dst)
        # All elements should be 42.0
        np.testing.assert_array_equal(dst, np.full(5, 42.0))

    def test_array_element_to_2d_row_slice(self):
        """Test assigning 1D array element to a row of 2D array: ey[0, :] = _fict_[0]"""

        @docc.program
        def array_element_to_2d_row_slice(_fict_: np.ndarray, ey: np.ndarray):
            ey[0, :] = _fict_[0]

        _fict_ = np.array([99.0, 1.0, 2.0], dtype=np.float64)
        ey = np.zeros((3, 4), dtype=np.float64)
        array_element_to_2d_row_slice(_fict_, ey)
        # First row should be all 99.0
        np.testing.assert_array_equal(ey[0, :], np.full(4, 99.0))
        # Other rows unchanged
        np.testing.assert_array_equal(ey[1:, :], np.zeros((2, 4)))

    def test_array_element_to_2d_col_slice(self):
        """Test assigning array element to a column: arr[:, 0] = src[1]"""

        @docc.program
        def array_element_to_2d_col_slice(src: np.ndarray, dst: np.ndarray):
            dst[:, 0] = src[1]

        src = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        dst = np.zeros((4, 3), dtype=np.float64)
        array_element_to_2d_col_slice(src, dst)
        # First column should be all 20.0
        np.testing.assert_array_equal(dst[:, 0], np.full(4, 20.0))
        # Other columns unchanged
        np.testing.assert_array_equal(dst[:, 1:], np.zeros((4, 2)))

    def test_scalar_to_slice(self):
        """Test assigning a scalar variable to a slice."""

        @docc.program
        def scalar_to_slice(val: float, dst: np.ndarray):
            dst[:] = val

        dst = np.zeros(5, dtype=np.float64)
        scalar_to_slice(3.14, dst)
        np.testing.assert_array_almost_equal(dst, np.full(5, 3.14))

    def test_constant_to_slice(self):
        """Test assigning a constant to a slice."""

        @docc.program
        def constant_to_slice(dst: np.ndarray):
            dst[:] = 2.5

        dst = np.zeros(4, dtype=np.float64)
        constant_to_slice(dst)
        np.testing.assert_array_almost_equal(dst, np.full(4, 2.5))

    def test_indexed_expr_to_slice(self):
        """Test assigning indexed expression to slice: dst[:] = src[i] where i is variable."""

        @docc.program
        def indexed_expr_to_slice(src: np.ndarray, idx: int, dst: np.ndarray):
            dst[:] = src[idx]

        src = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        dst = np.zeros(5, dtype=np.float64)
        indexed_expr_to_slice(src, 2, dst)
        np.testing.assert_array_equal(dst, np.full(5, 3.0))

    def test_mixed_fixed_and_slice_indices(self):
        """Test 3D array with mixed fixed and slice indices."""

        @docc.program
        def mixed_fixed_and_slice_indices(src: np.ndarray, dst: np.ndarray):
            # dst[0, :, 1] = scalar from src[0]
            dst[0, :, 1] = src[0]

        src = np.array([7.0, 8.0], dtype=np.float64)
        dst = np.zeros((2, 3, 2), dtype=np.float64)
        mixed_fixed_and_slice_indices(src, dst)
        # dst[0, :, 1] should all be 7.0
        np.testing.assert_array_equal(dst[0, :, 1], np.full(3, 7.0))

    def test_fdtd_2d_pattern(self):
        """Test the actual fdtd_2d pattern: ey[0, :] = _fict_[t] inside loop."""

        @docc.program
        def fdtd_2d_pattern(TMAX: int, _fict_: np.ndarray, ey: np.ndarray):
            for t in range(TMAX):
                ey[0, :] = _fict_[t]

        TMAX = 3
        _fict_ = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ey = np.zeros((2, 4), dtype=np.float64)
        fdtd_2d_pattern(TMAX, _fict_, ey)
        # After loop, ey[0, :] should be _fict_[2] = 3.0
        np.testing.assert_array_equal(ey[0, :], np.full(4, 3.0))

    def test_slice_to_slice_same_size(self):
        """Test slice-to-slice assignment of same size (no broadcasting)."""

        @docc.program
        def slice_to_slice_same_size(src: np.ndarray, dst: np.ndarray):
            dst[:] = src[:]

        src = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        dst = np.zeros(3, dtype=np.float64)
        slice_to_slice_same_size(src, dst)
        np.testing.assert_array_equal(dst, src)


class TestSliceRewriterNoTransform:
    """Tests verifying that fully-indexed arrays are NOT transformed."""

    def test_point_indexed_array_unchanged(self):
        """Verify that arr[i] (point index) is not transformed to arr[loop_var][i]."""

        @docc.program
        def point_indexed_array_unchanged(src: np.ndarray, dst: np.ndarray):
            # src[0] should remain src[0], not become src[loop_var][0]
            dst[:] = src[0] + 1.0

        src = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        dst = np.zeros(4, dtype=np.float64)
        point_indexed_array_unchanged(src, dst)
        np.testing.assert_array_equal(dst, np.full(4, 11.0))

    def test_2d_point_indexed_unchanged(self):
        """Verify that arr[i, j] (2D point indices) is not transformed."""

        @docc.program
        def point_indexed_2d_unchanged(src: np.ndarray, dst: np.ndarray):
            dst[:] = src[1, 2]

        src = np.arange(12.0, dtype=np.float64).reshape(3, 4)
        dst = np.zeros(5, dtype=np.float64)
        point_indexed_2d_unchanged(src, dst)
        np.testing.assert_array_equal(dst, np.full(5, src[1, 2]))
