"""Unit tests for np.ndarray.copy() support in the Python frontend."""

import numpy as np
import pytest
from docc.compiler import native


class TestNumpyCopyBasic:
    """Tests for basic array.copy() functionality."""

    def test_copy_1d(self):
        """Test copying a 1D array."""

        @native
        def copy_1d(a):
            b = a.copy()
            return b

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = copy_1d(a)
        np.testing.assert_allclose(result, a)

    def test_copy_2d(self):
        """Test copying a 2D array."""

        @native
        def copy_2d(a):
            b = a.copy()
            return b

        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = copy_2d(a)
        np.testing.assert_allclose(result, a)

    def test_copy_preserves_values(self):
        """Test that copy preserves all values exactly."""

        @native
        def copy_preserve(a):
            b = a.copy()
            return b

        a = np.random.rand(10, 10)
        result = copy_preserve(a)
        np.testing.assert_allclose(result, a)

    def test_copy_int_array(self):
        """Test copying an integer array."""

        @native
        def copy_int(a):
            b = a.copy()
            return b

        a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        result = copy_int(a)
        np.testing.assert_array_equal(result, a)


class TestNumpyCopyIndependence:
    """Tests that verify the copy is independent of the original."""

    def test_copy_independence_modify_original(self):
        """Test that modifying original doesn't affect copy."""

        @native
        def copy_then_modify_original(a):
            b = a.copy()
            a[0] = 999.0
            return b

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = copy_then_modify_original(a)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(result, expected)

    def test_copy_independence_modify_copy(self):
        """Test that modifying copy doesn't affect original."""

        @native
        def copy_then_modify_copy(a):
            b = a.copy()
            b[0] = 999.0
            return a

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = copy_then_modify_copy(a)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(result, expected)

    def test_copy_2d_independence(self):
        """Test independence with 2D arrays."""

        @native
        def copy_2d_modify(a):
            b = a.copy()
            a[0, 0] = 999.0
            return b

        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = copy_2d_modify(a)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(result, expected)


class TestNumpyCopyInLoop:
    """Tests for copy inside loops."""

    def test_copy_in_for_loop(self):
        """Test copy inside a for loop."""

        @native
        def copy_in_loop(a, n):
            for i in range(n):
                b = a.copy()
                a[0] = a[0] + 1.0
            return b

        a = np.array([1.0, 2.0, 3.0])
        result = copy_in_loop(a, 3)
        # After 3 iterations: a[0] goes 1->2->3->4
        # Last copy captures a when a[0]=3
        expected = np.array([3.0, 2.0, 3.0])
        np.testing.assert_allclose(result, expected)

    def test_copy_multiple_in_loop(self):
        """Test multiple copies inside a loop."""

        @native
        def multiple_copies_in_loop(a, b, n):
            for i in range(n):
                a_copy = a.copy()
                b_copy = b.copy()
                a[0] = a[0] + b_copy[0]
                b[0] = b[0] + 1.0
            return a_copy

        a = np.array([1.0, 2.0])
        b = np.array([10.0, 20.0])
        result = multiple_copies_in_loop(a, b, 2)
        # Iteration 1: a_copy=[1,2], b_copy=[10,20], a[0]=1+10=11, b[0]=11
        # Iteration 2: a_copy=[11,2], b_copy=[11,20], a[0]=11+11=22, b[0]=12
        expected = np.array([11.0, 2.0])
        np.testing.assert_allclose(result, expected)


class TestNumpyCopyWithOperations:
    """Tests for copy combined with other operations."""

    def test_copy_then_slice_assign(self):
        """Test copying then assigning to a slice."""

        @native
        def copy_slice_assign(a):
            b = a.copy()
            b[1:-1] = 0.0
            return b

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = copy_slice_assign(a)
        expected = np.array([1.0, 0.0, 0.0, 0.0, 5.0])
        np.testing.assert_allclose(result, expected)

    def test_copy_use_in_computation(self):
        """Test using copy in a computation."""

        @native
        def copy_compute(a):
            b = a.copy()
            c = b + a
            return c

        a = np.array([1.0, 2.0, 3.0])
        result = copy_compute(a)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, expected)

    def test_copy_overwrite(self):
        """Test pattern: empty_like then copy (like cavity_flow)."""

        @native
        def copy_overwrite(a):
            b = np.empty_like(a)
            b = a.copy()
            b[0] = b[0] + 10.0
            return b

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = copy_overwrite(a)
        expected = np.array([11.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(result, expected)


class TestNumpyCopy2DSlicing:
    """Tests for copy with 2D array slicing patterns (like cavity_flow)."""

    def test_copy_2d_slice_read(self):
        """Test copying 2D array and reading slices."""

        @native
        def copy_2d_slice_read(p):
            pn = p.copy()
            result = pn[1:-1, 1:-1]
            return result

        p = np.arange(25.0).reshape(5, 5)
        result = copy_2d_slice_read(p)
        expected = p[1:-1, 1:-1].copy()
        np.testing.assert_allclose(result, expected)

    def test_copy_2d_stencil_pattern(self):
        """Test the stencil pattern from cavity_flow."""

        @native
        def stencil_pattern(p):
            pn = p.copy()
            # Simple stencil: average of neighbors
            p[1:-1, 1:-1] = (
                pn[1:-1, 2:] + pn[1:-1, 0:-2] + pn[2:, 1:-1] + pn[0:-2, 1:-1]
            ) / 4.0
            return p

        p = np.ones((5, 5), dtype=np.float64)
        p[2, 2] = 5.0  # Center point different
        result = stencil_pattern(p.copy())

        # Reference computation
        p_ref = np.ones((5, 5), dtype=np.float64)
        p_ref[2, 2] = 5.0
        pn_ref = p_ref.copy()
        p_ref[1:-1, 1:-1] = (
            pn_ref[1:-1, 2:]
            + pn_ref[1:-1, 0:-2]
            + pn_ref[2:, 1:-1]
            + pn_ref[0:-2, 1:-1]
        ) / 4.0

        np.testing.assert_allclose(result, p_ref)

    def test_copy_in_iterative_stencil(self):
        """Test copy in iterative stencil computation."""

        @native
        def iterative_stencil(p, nit):
            for q in range(nit):
                pn = p.copy()
                p[1:-1, 1:-1] = (
                    pn[1:-1, 2:] + pn[1:-1, 0:-2] + pn[2:, 1:-1] + pn[0:-2, 1:-1]
                ) / 4.0
            return p

        p = np.zeros((5, 5), dtype=np.float64)
        p[2, 2] = 1.0  # Single point source
        result = iterative_stencil(p.copy(), 3)

        # Reference computation
        p_ref = np.zeros((5, 5), dtype=np.float64)
        p_ref[2, 2] = 1.0
        for q in range(3):
            pn_ref = p_ref.copy()
            p_ref[1:-1, 1:-1] = (
                pn_ref[1:-1, 2:]
                + pn_ref[1:-1, 0:-2]
                + pn_ref[2:, 1:-1]
                + pn_ref[0:-2, 1:-1]
            ) / 4.0

        np.testing.assert_allclose(result, p_ref, rtol=1e-10)


class TestNumpyCopyEdgeCases:
    """Edge case tests for copy."""

    def test_copy_single_element(self):
        """Test copying a single element array."""

        @native
        def copy_single(a):
            b = a.copy()
            return b

        a = np.array([42.0])
        result = copy_single(a)
        np.testing.assert_allclose(result, a)

    def test_copy_large_array(self):
        """Test copying a large array."""

        @native
        def copy_large(a):
            b = a.copy()
            return b

        a = np.random.rand(1000)
        result = copy_large(a)
        np.testing.assert_allclose(result, a)

    def test_multiple_sequential_copies(self):
        """Test multiple sequential copies."""

        @native
        def multi_copy(a):
            b = a.copy()
            c = b.copy()
            d = c.copy()
            return d

        a = np.array([1.0, 2.0, 3.0])
        result = multi_copy(a)
        np.testing.assert_allclose(result, a)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
