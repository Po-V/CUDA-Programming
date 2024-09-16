# CUDA Vector addition

This project demonstrates how to compute the sum of two arrays using CUDA on the GPU. The code performs the vector addition in parallel by utilizing GPU threads.

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `vecAdd.cu`

- **vecAdd Kernel**: 
    - The kernel `vecAdd` performs element-wise addition of two arrays (`a` and `b`) and stores the result in array `c`. 
    - The kernel uses the thread ID to compute which element each thread is responsible for.

- **Initialization**:
    - The `init_array` function fills an array with random values between 0 and 100.

- **Main Function**:
    - Memory for the input and output arrays is allocated using CUDA's `cudaMallocManaged`, which allows the memory to be accessible by both the CPU and GPU.
    - The vector addition kernel is then launched with a grid of blocks and threads.
    - After kernel execution, `cudaDeviceSynchronize` ensures that the GPU computations finish before the program terminates.

## Compilation and Execution

To compile and run the program, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd vector_addition

2. **Compile the CUDA program**:
   ```bash
   nvcc vecAdd.cu -o vecAdd

3. **Run the compiled program**:
   ```bash
   ./vecAdd