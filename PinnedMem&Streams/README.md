# Vector addition using Pinned Memory and Streams

This project implements vector addition on GPU using multiple CUDA streams to divide workload into segments. 

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `vecAdd_kernel.cu`

- **vecAdd kernel**: 
    - This kernel performs addition between 2 vectors using threads.

- **vecAdd_gpu**:
    - 10 cuda streams are created and initialized.
    - The vectors are divided into chunks of 10 segments and these chunks of data are copied from host to device asynchronously. This transfer is assigned to corresponding `stream[s]` so it happens concurrently with other operations in other streams. 
    - `vecAdd_kernel` launches the kernel on GPU using current stream `stream[s]` ensuring it runs concurrently with other streams. 
    - After kernel finishes, the result is copied back from device to host asynchronously using `cudaMemcpyAsync` in corresponding stream. 

- **Main Function**:
    - `cudaMallocHost` allocates pinned memory on host. Pinned memory is a region of host memory that cannot be swapped out by operating system, ensuring data remains in physical RAM. When using pinned memory, data transfers between host and device are faster compared to transfers using regular pageable memory. 
    - 2 vectors are initialized with random values between 1 and 10 and the `vecAdd_gpu` function is then called.
    - After kernel execution, the host memory is freed.

## Compilation and Execution

To compile and run the program, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd "PinnedMem&Streams"

2. **Compile the CUDA program**:
   ```bash
   nvcc vector-addition.cu -o vecAdd

3. **Run the compiled program**:
   ```bash
   ./vecAdd


