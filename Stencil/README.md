# Stencil computation

This project demonstrates how perform stencil computation using CUDA on the GPU. The code performs computation without thread coarsening and with thread coarsening using shared memory and registers.

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `stencil.cu`

- **stencil kernel**: 
    - The kernel `stencil_kernel` performs a 3D stencil operation on a grid where is grid point is updated based on its neighboring points using shared memory for efficient data access.
    - The threads loads a tile of input data into shared memory, ensuring boundary conditions are respected and then compute a weighted sum involving the central point and its 6 neighbors. 
    - The result is written to an output array.

- **Main Function**:
    - Memory for the input and output matrices are allocated using `Malloc`, which allows the memory on host device for input and output.
    - `cudaMalloc` allocates memory for input, output and mask on GPU device.
    - The stencil_kernel is then launched with a grid of blocks and threads.
    - After kernel execution, the host memory is freed.

### `stencilWithThreadCoarsening.cu`

- **stencil kernel**: 
    - The kernel `stencil_kernel` performs 3D stencil operation by iterating through planes in z-dimension, using shared memory for the current plane and registers for the previous and next planes. 
    - Compared to previous kernel which loaded all neighbors into shared memory, this version optimizes memory usage by handling only one plane at a time in shared memory and shifting neighboring planes through registers. 

- **Main Function**:
    - Memory for the input and output matrices are allocated using `Malloc`, which allows the memory on host device for input and output.
    - `cudaMalloc` allocates memory for input, output and mask on GPU device.
    - The convolution_kernel is then launched with a grid of blocks and threads.
    - After kernel execution, the host memory is freed.

## Compilation and Execution

To compile and run the program, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd Stencil

2. **Compile the CUDA program**:
   ```bash
   nvcc stencil.cu -o stencil

3. **Run the compiled program**:
   ```bash
   ./stencil

