#include <cuda_runtime.h>
#include <daisy_rtl/daisy_rtl.h>
#include <stdio.h>

// CUDA kernels -----------------------------------------------------------
__global__ void initKernel(double *A, double *B, double *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        A[idx] = idx;
        B[idx] = N - idx;
        C[idx] = 0.0;
    }
}

__global__ void addKernel(const double *A, const double *B, double *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv) {
    const int N = 1000;
    const size_t bytes = N * sizeof(double);

    double *hC = (double *) malloc(bytes);

    double *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    dim3 blockSize = {256, 1, 1};
    dim3 gridSize = {(unsigned int) ((N + blockSize.x - 1) / blockSize.x), 1, 1};

    __daisy_metadata_t metadata = {
        .file_name = "instrumentation_cuda_test.cu",
        .function_name = "main",
        .line_begin = 18,
        .line_end = 31,
        .column_begin = 4,
        .column_end = 5,
        .region_name = "instrumentation_cuda_test_main",
    };
    unsigned long long region_id = __daisy_instrumentation_init(&metadata, __DAISY_EVENT_SET_CUDA);

    for (size_t rep = 0; rep < 10; rep++) {
        __daisy_instrumentation_enter(region_id);

        initKernel<<<gridSize, blockSize>>>(dA, dB, dC, N);
        addKernel<<<gridSize, blockSize>>>(dA, dB, dC, N);
        cudaDeviceSynchronize();

        __daisy_instrumentation_exit(region_id);

        // Copy result back to host for verification/printing
        cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++) {
            printf("%f\n", hC[i]);
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hC);

    __daisy_instrumentation_finalize(region_id);
}
