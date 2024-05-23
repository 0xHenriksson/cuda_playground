# Matrix Inversion

Now that I've got my matrix multiplication implementation and I've got a better understanding of how to get closer to the cuBLAS implementation, it's time to start using cuBLAS to build some other numerical linear algebra implementations. 

## Math review (basic)

A nonsingular or invertible matrix is a square matrix that has a multiplicative inverse. In other words, if A is an invertible matrix, then there exists another matrix B such that A × B = B × A = I, where I is the identity matrix. The inverse of A is typically denoted as A^(-1). For a matrix to be invertible, it must be square (i.e., have the same number of rows and columns) and have a non-zero determinant. The determinant is a scalar value that can be computed from the elements of a square matrix and provides information about the matrix's properties. If the determinant of a matrix is zero, the matrix is singular or non-invertible.
