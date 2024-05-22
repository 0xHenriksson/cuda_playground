# CUDA PLAYGROUND
A collection of my expolorations with CUDA on my NVIDIA Jetson Orin Nano 8gb Dev Kit

TO-DO:
- cuda-gdb exploration
- computer sanitizer 
- set up nsight systems and compute from remote
- nvidia nvtx
- Use cuBLAS/CUTLASS to implement various numerical linear algebra operations/algorithms
    - SVD
    - QR Factorization Algorithm
    - Graham-Schmidt Ortogonalization
    - Least-Squares Algorithms
    - Arnoldi Iteration
    - GMRES

In Progress:
- SGEMM optimization based on https://siboehm.com/articles/22/CUDA-MMM
    - experiencing shared memory errors
        - consider number of threads used
        - possible shared memory error (can't dynamically allocate)


Completed:
