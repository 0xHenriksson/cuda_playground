#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void sgemm_v01(int M, int N, int K, float alpha, const float *A,
                           int lda, const float *B, int ldb, float beta, float *C,
                           int const float *B, float beta, float *C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

