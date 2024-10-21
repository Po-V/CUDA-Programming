# COO sparse matrix vector multiplication

This project implements multiplication between a sparse matrix ordered in coordinate list (COO) with a vector.

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `spmv_coo.cu`

- **spmv coo kernel**: 
    - The `spmv_coo_kernel` performs matrix multiplication between non-zero elements in a sparse matrix and a vector using atomic operation. 

- **Main Function**:
    - `Malloc` allocates memory for non-zero elements in sparse matrix, a vector of ones and output vector.
    - The `spmv_coo_gpu` is then called to allocate memory on GPU using `cudaMalloc` and call the `spmv_coo_kernel`.
    - After kernel execution and verifying the results, the device and host memory is freed.

### `spmv_csr.cu`

- **spmv csr kernel**: 
    - The `spmv_csr_kernel` performs matrix multiplication between non-zero elements in a sparse matrix and a vector. 
    - The sparse matrix is stored in Compressed Sparse Row (CSR) format where the values array contains the non-zero elements of the matrix in row-major order. The column indices array holds the column indices corresponding to each non-zero element in Values array. The row pointer helps to identify where each row starts and ends within values and column indices array. For ex, for row 2, the row pointer gives `[1,3]` that means the non-zero values are between indices 1 and 3 of the values array: `[3,4]`.

- **Main Function**:
    - `Malloc` allocates memory for non-zero elements in sparse matrix, a vector of ones and output vector.
    - The `spmv_csr_gpu` is then called to allocate memory on GPU using `cudaMalloc` and call the `spmv_csr_kernel`.
    - After kernel execution and verifying the results, the device and host memory is freed.

### `spmv_ellpack.cu`

- **Ellpack kernel**:
    -  The `spmv_ell_kernel` implement Sparse Matrix-Vector Multiplication on GPU using ELLPACK format where each row is padded to a uniform length for parallel processing. It transfers the parse matrix and input vector to the GPU, launches a CUDA kernel to compute the result and copies the output back to the host. The kernel computes each row's dot product of non-zero matrix elements with corresponding elements in the input vector.

- **Main Function**:
    - `Malloc` allocates memory for non-zero elements in sparse matrix, a vector of ones and output vector.
    - The `spmv_ell_gpu` is then called to allocate memory on GPU using `cudaMalloc` and call the `spmv_ell_kernel`.
    - After kernel execution and verifying the results, the device and host memory is freed.

## Compilation and Execution

To compile and run the coo example, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd sparseMatrixComputation

2. **Compile the CUDA program**:
   ```bash
   nvcc spmv-coo.cu -o coo

3. **Run the compiled program**:
   ```bash
   ./coo

