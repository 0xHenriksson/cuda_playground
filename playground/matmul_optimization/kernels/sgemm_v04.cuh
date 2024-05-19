#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const in BN, const int BK, const int TM>
__global__ void sgemm_v04(int M, int N, int K, float alpha
                        const float *A, const float *B, float beta,
                        float *C) {

    const uint cRow = blockIdx.y;
    const uint CCol = blockIdx.x;

    // each warp calculates 32*TM elements, with 32 as columnar dim
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;

    // allocate thread-local cache for results in register file
    float threadResults[TM] = {0.0};

    // outer loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate smem caches
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * N;

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // outside loop is the dotproduct loop
            // facilitates resuse of the Bs entry, which
            // is then cached in tmp var
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    As[(threadRows * TM +resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    // write out results
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] = 
            alpha * threadResults[resIdx] + beta *
            C[(threadRow * TM + resIdx) * N + threadCol];
    }
}