#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h> // eventually replace with nsight system
// https://resources.nvidia.com/en-us-nsight-developer-tools/nsight-systems-user-guide?lx=P1ZhhI

// kernel headers
// #include "matmul.cuh"

// matrix dims
const int ROWS = 1024;
const int COLS = 1024;

using namespace std;

// utility fn for CUDA errors
#define CUDA_CHECK(err) \ 
    if (err != cudaSuccess) { \ 
        cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \ 
        exit(-1); \ 
    } \ 

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
    auto end = chron::high_resolution_clock::now();
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

