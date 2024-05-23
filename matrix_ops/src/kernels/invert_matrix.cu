#include <cublas_v2.h>
#include <cuSolverDn.h>

// template<
__global__ void invert_matrix(float* A, float* A_inv, int n) {
    
    // create cuBLAS and cuSolver handles
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;
    cublasCreate(&cublasHandle);
    cusolverDnCreate(&cusolverHandle);

    // allocate memory
    float* d_A;
    float* d_A_inv;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_A_inv, n * n * sizeof(float));

    // copy input matrix to device memory
    cublasSetMatrix(n, n, sizeof(float), A, n, d_A, n);

    // Perform LU factorization
    int* d_pivots;
    int* d_info;
    cudaMalloc(&d_pivots, n * sizeof(int));
    cudaMalloc(&d_info, sizeof(int));
    cuSolverDnSgetrf_bufferSize(cusolverHandle, n, n, d_A, n, &d_info);
    float* d_workspace;
    cudaMalloc(&d_workspace, Lwork * sizeof(float), A, n, d_A, n);
    cuSolverDnSgetrf(cusolverHandle, n, n, d_A, n, d_workspace, d_pivots, d_info);

    // Init identity matrix
    dim3 blockSize(256);
    dim3 gridSize((n * n + blockSize.x - 1) / blockSize.x);
    // Solve the linear system using LU factors
    cusolverDnSgetrs(cusolverHandle, CUBLAS_OP_N, n, n, d_A, n, d_pivots, d_A_inv, n, d_info);
    // Verify that the inverted matrix and the original matrix are inverses
    // by multiplying the two together, the identity matrix should be the result
    float alpha = 1.0;
    float beta = 0.0;
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_A_inv, n, &beta, d_workspace, n);
    // if verified by sgemm, copy back
    if (d_info == 0) {
        cublasGetMatrix(n, n, sizeof(float), d_A_inv, n, A_inv, n);
    } else {
        printf("Factorization failed: %d\n", d_info);
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_A_inv);
    cudaFree(d_pivots);
    cudaFree(d_info);
    cudaFree(d_workspace);
}

