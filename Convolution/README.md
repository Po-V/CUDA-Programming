# CUDA Convolution

This project demonstrates how perform convolution using CUDA on the GPU. The code performs convolution without tiling and with tiling by tiling the matrices into 36x36 size and storing each tile in shared memory.

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `convolution.cu`

- **convolution kernel**: 
    - The kernel `convolution_kernel` computes convolution of 2D input matrix with 2D mask using constant memory for mask and global memory for input and output. 
    - Each thread calculates the convolution result for a single pixel in the output matrix, by applying the mask centered on that pixel and summing the results. 
    - Boundary checks are done to ensure threads working on edge pixels don't access out-of-bounds memory. 

- **Main Function**:
    - Memory for the input and output matrices are allocated using `Malloc`, which allows the memory on host device for input and output.
    - The input and mask matrices are initialized with values between 0 and 1.
    - `cudaMalloc` allocates memory for input, output and mask on GPU device.
    - The convolution_kernel is then launched with a grid of blocks and threads.
    - After kernel execution, the host memory is freed.

### `convWithTiling.cu`

- **convolution kernel**: 
    - The kernel `convolution_kernel_tiled` uses tiled approach to optimize performance by reducing the number of redundant global memory accesses. Shared memory, which is much faster than global memory, is utilized to store data needed by multiple threads within a block. 
    - The use of `__syncthreads()` ensures all threads in a block have loaded their data into shared memory before any computation is done. 
    - Halo cells (if out-of-bounds) are set to zero (zero-padding).
    - Each thread computes the convolution for one output element by applying the convolution mask to the tile in shared memory.
    - The result is stored in output matrix.  

- **Main Function**:
    - Memory for the input and output matrices are allocated using `Malloc`, which allows the memory on host device for input and output.
    - The input and mask matrices are initialized with values between 0 and 1.
    - `cudaMalloc` allocates memory for input, output and mask on GPU device.
    - The convolution_kernel is then launched with a grid of blocks and threads.
    - After kernel execution, the host memory is freed.

## Compilation and Execution

To compile and run the program, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd Convolution

2. **Compile the CUDA program**:
   ```bash
   nvcc convolution.cu -o convolution

3. **Run the compiled program**:
   ```bash
   ./convolution
