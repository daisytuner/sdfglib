import docc
import numpy as np

# Set the global context for the docc module
# docc.sdfg_rpc_context = docc.SimpleRpcContext("localhost:8080/docc", "transfertune", {"AddHeader":"Content"})
docc.sdfg_rpc_context = docc.DaisytunerTransfertuningRpcContext.from_docc_config()


def test_scheduling_sequential():
    @docc.program(target="sequential", category="server")
    def vec_add(A, B, C, N):
        for i in range(N):
            C[i] = A[i] + B[i]

    N = 1024
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(N, dtype=np.float64)

    vec_add(A, B, C, N)
    assert np.allclose(C, A + B)
