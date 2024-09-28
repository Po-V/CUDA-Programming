# Brent Kung parallel scan algorithm

This project implements parallel scan (or prefit sum) operation using CUDA, optimizing it with block-level and thread-level operations.

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `scan-BK.cu`

- **improved scan kernel**: 
    - The kernel `scan_kernel` performs optimized parallel scan for large datasets.
    - Each thread loads 2 elements from global memory into shared memory (buffer_s), one from the first half of segment and one from the second half. This maximizes parallel data access.
    - The kernel performs parallel reduction to compute partial sums within each block. 
    - In post-reduction step, the kernel applies prefix sum on block results.
    - The last element from block-level scan is stored in partialSum array and the result of scan is written from shared memory (`buffer_s`) back to global memory (`output`).

- **Main Function**:
    - `cudaMalloc` allocates memory for input, output and partial sums array on GPU device.
    - The `scan_kernel` is then launched with a grid of blocks and threads.
    - After kernel execution and verifying the results, the host memory is freed.

## Compilation and Execution

To compile and run the program, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd scan-BrentKung

2. **Compile the CUDA program**:
   ```bash
   nvcc scan-BK.cu -o bk

3. **Run the compiled program**:
   ```bash
   ./bk


