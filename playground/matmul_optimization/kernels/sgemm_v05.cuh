#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


// 2D Blocktiling increases arithmetic intensity
template <const int BM, const int BN, const int BK, const int TM, const int TN>
// __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)
// when launch bounds are specified the compiler first derives from them the upper limit L
// on the number of registers the kernel should use to ensure that 'minBlocksPerMultiProcessor'
// blocks of 'maxThreadPerBlock' threads can resid eon the multiprocessor
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm_v05(int M, int N, int K, float alpha, const float *A,
            const float *B, float beta, float *C) {
    
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint totalResultsBlockTile = BM * BN;
    // thread is responsible for claculating TM*TN elements in the blocktile
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
    assert(numThreadsBlocktile == blockDim.x);

    // BM/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx. / (BN / TN);

    // allocate space for current blocktile in smem
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * Bn];

    // move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // calculating the indices that htis thread will load into SMEM
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    // calculate number of Rows of As that loaded ina single step by a single block
    const uint strideA = numThreadsBlocktile / BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    // for As and Bs we want each load to span full column-width
    // for better GMEM coalescing
    const uint strideB = numThreadsBlocktile / BN;

    // allocate thread-local cache for results 
    float threadResults[TM * TN] = {0.0};
    // register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate smem caches
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] = 
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] = 
                B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        // advance blocktile
        A += BK; // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // block into registers
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    // write out results
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            c[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] + 
                beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }

}