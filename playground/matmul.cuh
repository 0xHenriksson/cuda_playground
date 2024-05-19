#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

/* Lower bound fastest possible runtime
For matmul of two 4092^2 matrices followed by an addition of a 4092^2 matrix to make the GEMM
1. Total FLOPS: 2*4092^3 + 4092 = 137 GLOPS
2. Total data to read (minimum!): 3 * 4092 * 4B = 201MB
3. Total data to store: 4092^2 * 4B = 67mb
*/

// Kernel 2: Global Memory Coalescing
// Call like so:
// dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
// dim3 blockDim(32 * 32)
// sgemm_coalescing<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float *B, float beta, float *C) {

// change how we assign positions of the result matrix C to threads
// compute position in C that this thread is responsible for
const uint x = blockIdx.x BLOCKSIZE * (threadIdx.x / BLOCKSIZE);
const uint y = blockIdx.y * BLOCKSIZE * (threadIdx.y % BLOCKSIZE);

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

// Kernel 3: Shared Memory Cache-Blocking




