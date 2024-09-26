# Kogge-Stone algorithm

This project implements parallel scan (or prefit sum) operation using CUDA, optimizing it with block-level and thread-level operations.

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `scan-KS.cu`

- **improved scan kernel**: 
    - The kernel `scan_kernel_improved` performs optimized parallel scan for large datasets, where each thread processes 8 elements.
    - Each thread loads 8 elements from global memory into shared memory in a coalesced manner to optimize memory access and performs a scan on the 8 elements it loaded. The results are written back to buffer_s.
    - A parallel scan is also performed on block-level partial sums. This uses shared memory buffers `buffer1_s` and `buffer2_s` to propagate results across threads. 
    - The last element from block-level scan is stored in partislSum array and the result of scan is written from shared memory (`buffer_s`) back to global memory (`output`).

- **Main Function**:
    - `cudaMalloc` allocates memory for input, output and partial sums array on GPU device.
    - The `scan_kernel_improved` is then launched with a grid of blocks and threads.
    - After kernel execution and verifying the results, the host memory is freed.

## Compilation and Execution

To compile and run the program, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd scan-KoggeStone

2. **Compile the CUDA program**:
   ```bash
   nvcc scan-KS.cu -o ks

3. **Run the compiled program**:
   ```bash
   ./ks

