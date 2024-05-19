#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h> // eventually replace with nsight system
// https://resources.nvidia.com/en-us-nsight-developer-tools/nsight-systems-user-guide?lx=P1ZhhI


// matrix dims
const int ROWS = 1024;
const int COLS = 1024;

using namespace std;

// utility fn for CUDA errors
#define CUDA_CHECK(err) \ 
    if (err != cudaSuccess) { \ 
        cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \ 
        exit(-1); \ 
    }

// create blocks to map all of C
dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1);
// 32 * 32 = 1024 thread per block
// laaunch asyncrhonous execution execution of the kernel on the device
// functioncall returns immediately on the host
sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float *B, float beta, float *C) {

// compute position in C that this thread is responsible for
const uint x = blockIdx.x * blockDim.x + threadIdx.x;
const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // if condition is necessary for when M or N aren't multiples of 32
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = ⍺*(A@B) + β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int main() {

    // allocate host matrices
    float* h_matrixA = new float[ROWS * COLS];
    float* h_matrixB = new float[ROWS * COLS];
    float* h_result = new float[ROWS * COLS];

    // init host matrices
    for (int i = 0; i < ROWS * COLS; i++) {
        h_matrixA[i] = static_cast<float>(i);
        h_matrixB[i] = 1.0f;
    }

    // allocate device matrices
    float* d_matrixA;
    float* d_matrixB;
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_matrixA, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_matrixB, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, ROWS * COLS * sizeof(float)));

    // copy host matrices to device
    CUDA_CHECK(cudaMemcpy(d_matrixA, h_matrixA, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_matrixB, h_matrixB, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));

    // start CUDA profiling
    CUDA_CHECK(cudaProfilerStart());

    // Benchmark kernel
    auto start = chrono::high_resolution_clock::now();
    // kernel here
    sgemm_naive(ROWS, COLS, 1.0, 0.0, d_matrixA, d_matrixB, d_result);
    auto end = chrono::high_resolution_clock::now();
    auto duration chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Kernel duration: " << duration.count() << " microseconds" << endl;

    CUDA_CHECK(cudaProfilerStop());

    CUDA_CHECK(cudaMemcpy(h_result, d_result, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));

    // free device memory
    CUDA_CHECK(cudaFree(d_matrixA));
    CUDA_CHECK(cudaFree(d_matrixB));
    CUDA_CHECK(cudaFree(d_result));

    // Free host memory
    delete[] h_matrixA;
    delete[] h_matrixB;
    delete[] h_result;

    return 0;
}

