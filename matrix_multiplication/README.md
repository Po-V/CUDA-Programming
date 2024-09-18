# CUDA Matrix Multiplication

This project demonstrates how perform multiplication of 2 matrices using CUDA on the GPU. The code performs matrix multiplication by tiling the matrices into 32x32 size and storing each tile in shared memory.

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `matrixMul.cu`

- **matrix multiplication kernel**: 
    - The kernel `mm_kernel` performs matrix multiplication by first tiling the 2 matrices into 32x32 tiles. 
    - The tiles are loaded into shared memory with size of 32x32. 
    - The dot product is calculated for each tile where each thread is responsible for calculating dot product of one row from matrix A and one column from matrix B. The dot product is sum of element-wise products between corresponding row in A and column in B. 
    - The result in terms of sum is stored by thread into matrix C.

- **Initialization**:
    - The `init_array` function fills an array with value 2.

- **Main Function**:
    - Memory for the input and output matrices are allocated using CUDA's `cudaMalloc`, which allows the memory to be accessible by GPU.
    - The mm_kernel is then launched with a grid of blocks and threads.
    - After kernel execution, the host memory is freed.

## Compilation and Execution

To compile and run the program, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd matrix_multiplication

2. **Compile the CUDA program**:
   ```bash
   nvcc matrixMul.cu -o matrixMul

3. **Run the compiled program**:
   ```bash
   ./matrixMul
