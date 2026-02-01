from docc.python import native
import numpy as np


def test_adi_simplified():
    # Simplified ADI-like pattern to check correctness of logic
    @native
    def adi_simple(n, steps):
        # 1D diffusion implicit-like
        u = np.zeros(n, dtype=np.float64)
        u[0] = 1.0
        u[n - 1] = 1.0

        tmp = np.empty(n, dtype=u.dtype)

        for t in range(steps):
            # Forward sweep
            tmp[0] = u[0]
            for i in range(1, n - 1):
                tmp[i] = 0.5 * (u[i - 1] + u[i + 1])
            tmp[n - 1] = u[n - 1]

            # Copy back
            for i in range(n):
                u[i] = tmp[i]

        return u

    n = 10
    steps = 5
    res = adi_simple(n, steps)

    # Check correctness against python
    u_ref = np.zeros(n, dtype=np.float64)
    u_ref[0] = 1.0
    u_ref[n - 1] = 1.0
    tmp_ref = np.empty(n, dtype=np.float64)
    for t in range(steps):
        tmp_ref[0] = u_ref[0]
        for i in range(1, n - 1):
            tmp_ref[i] = 0.5 * (u_ref[i - 1] + u_ref[i + 1])
        tmp_ref[n - 1] = u_ref[n - 1]
        u_ref[:] = tmp_ref[:]

    np.testing.assert_allclose(res, u_ref)


def test_column_assignment_indexing():
    @native
    def column_assign_kernel(N):
        A = np.zeros((N, N), dtype=np.float64)
        # Assign to column 0, rows 1 to N-1
        # Indices: (1, 0), (2, 0), ..., (N-2, 0)
        A[1 : N - 1, 0] = 1.0
        return A

    N = 10
    res = column_assign_kernel(N)

    # Check Column 0
    assert res[0, 0] == 0.0  # Row 0 is excluded
    assert np.all(res[1 : N - 1, 0] == 1.0)
    assert res[N - 1, 0] == 0.0  # Row N-1 is excluded

    # Check Column 1 (should be empty)
    assert np.all(res[:, 1:] == 0.0)


def test_row_assignment_complex_index():
    @native
    def row_assign_complex(N):
        A = np.zeros((N, N), dtype=np.float64)
        # Testing logic where index expression is complex
        # A[1+1 : N-1, 0]
        A[2 : N - 1, 0] = 1.0
        return A

    N = 10
    res = row_assign_complex(N)

    assert res[0, 0] == 0.0
    assert res[1, 0] == 0.0
    assert np.all(res[2 : N - 1, 0] == 1.0)
    assert np.all(res[:, 1:] == 0.0)


def test_adi_simplified():
    # Simplified ADI-like pattern to check correctness of logic
    @native
    def adi_simple(n, steps):
        # 1D diffusion implicit-like
        u = np.zeros(n, dtype=np.float64)
        u[0] = 1.0
        u[n - 1] = 1.0

        tmp = np.empty(n, dtype=u.dtype)

        for t in range(steps):
            # Forward sweep
            tmp[0] = u[0]
            for i in range(1, n - 1):
                tmp[i] = 0.5 * (u[i - 1] + u[i + 1])
            tmp[n - 1] = u[n - 1]

            # Copy back
            for i in range(n):
                u[i] = tmp[i]

        return u

    n = 10
    steps = 5
    res = adi_simple(n, steps)

    # Check correctness against python
    u_ref = np.zeros(n, dtype=np.float64)
    u_ref[0] = 1.0
    u_ref[n - 1] = 1.0
    tmp_ref = np.empty(n, dtype=np.float64)
    for t in range(steps):
        tmp_ref[0] = u_ref[0]
        for i in range(1, n - 1):
            tmp_ref[i] = 0.5 * (u_ref[i - 1] + u_ref[i + 1])
        tmp_ref[n - 1] = u_ref[n - 1]
        u_ref[:] = tmp_ref[:]

    np.testing.assert_allclose(res, u_ref)
