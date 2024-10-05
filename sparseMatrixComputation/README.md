# COO sparse matrix vector multiplication

This project implements multiplication between a sparse matrix ordered in coordinate list (COO) with a vector.

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `spmv_coo_kernel.cu`

- **improved scan kernel**: 
    - The `spmv_coo_kernel` performs matrix multiplication between non-zero elements in a sparse matrix and a vector using atomic operation. 

- **Main Function**:
    - `Malloc` allocates memory for non-zero elements in sparse matrix, a vector of ones and output vector.
    - The `spmv_coo_gpu` is then called to allocate memory on GPU using `cudaMalloc` and call the `spmv_coo_kernel`.
    - After kernel execution and verifying the results, the device and host memory is freed.

## Compilation and Execution

To compile and run the program, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd sparseMatrixComputation

2. **Compile the CUDA program**:
   ```bash
   nvcc spmv-coo.cu -o coo

3. **Run the compiled program**:
   ```bash
   ./coo

