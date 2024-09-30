# Histogram computation

This project demonstrates how perform histogram computation using CUDA on the GPU. The code performs computation using atomic operations.

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `histogram.cu`

- **histogram kernel**: 
    - The kernel `histogram_kernel` histogram of image using shared memory within each block to create local histogram. 
    - Threads update this shared histogram based on pixel values and then synchronize to safely combine results into global histogram using atomic operations.
    - Atomic operation ensures safe updates to both shared memory `arr` and global memory `bins` histogram.  

- **Main Function**:
    - Memory for the input images are allocated using `Malloc`, which allows the memory on host device for input and output. The bins array is initialized to 0.
    - `cudaMalloc` allocates memory for image and bins on GPU device.
    - The histogram_kernel is then launched with a grid of blocks and threads.
    - After kernel execution, the device and host memory is freed.

## Compilation and Execution

To compile and run the program, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd histogram

2. **Compile the CUDA program**:
   ```bash
   nvcc histogram.cu -o histogram

3. **Run the compiled program**:
   ```bash
   ./histogram

