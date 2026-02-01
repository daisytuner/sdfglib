import numpy as np
from docc.compiler import native


def test_loop_var_index():
    """Test kernel pattern: ey[0, :] = _fict_[t] where t is loop variable."""

    @native
    def kernel(TMAX: int, _fict_: np.ndarray, ey: np.ndarray):
        for t in range(TMAX):
            ey[0, :] = _fict_[t]

    TMAX = 3
    _fict_ = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    ey = np.zeros((2, 4), dtype=np.float64)
    kernel(TMAX, _fict_, ey)
    # After loop, ey[0, :] should be _fict_[2] = 3.0
    np.testing.assert_array_equal(ey[0, :], np.full(4, 3.0))


def test_inplace_slice_sub_with_sliced_diff():
    """Test pattern: ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])"""

    @native
    def kernel(ey: np.ndarray, hz: np.ndarray):
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])

    ey = np.ones((3, 4), dtype=np.float64)
    hz = np.arange(12.0, dtype=np.float64).reshape(3, 4)

    # Expected result: ey[1:, :] = ey[1:, :] - 0.5 * (hz[1:, :] - hz[:-1, :])
    # hz[1:, :] - hz[:-1, :] = [[ 4,  5,  6,  7]] - [[0, 1, 2, 3]] = [[4, 4, 4, 4]]
    #                        and [[8,  9, 10, 11]] - [[4, 5, 6, 7]] = [[4, 4, 4, 4]]
    # So 0.5 * diff = [[2, 2, 2, 2], [2, 2, 2, 2]]
    # ey[1:, :] = [[1, 1, 1, 1], [1, 1, 1, 1]] - [[2, 2, 2, 2], [2, 2, 2, 2]] = [[-1, -1, -1, -1], [-1, -1, -1, -1]]
    expected = np.ones((3, 4), dtype=np.float64)
    expected[1:, :] = -1.0

    kernel(ey, hz)
    np.testing.assert_array_almost_equal(ey, expected)
