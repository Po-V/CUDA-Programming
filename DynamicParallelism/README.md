# Dynamic Parallelism
This project demonstrates how to implement Breadth-First Search (BFS) algorithm on a
graph represented in Compressed Sparse Row (CSR) format. It uses dynamic parallelism, which enables a kernel to launch other kernels directly on GPU without needing to return control to the CPU

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `bfs-graph.cu`

- **bfs kernel**: 
    - This kernel is responsible for processing each node in `prevFrontier`, which contains the nodes discovered in previous BFS level. 
    - For each node in `prevFrontier`, it launches a child kernel `bfs_child_kernel` to handle its neighbors concurrently, distributing them among threads within a block.
    - The `bfs_child_kernel` checks each neighbor and if it has not been visited, updates its level and adds it to the currFrontier using atomic operations. 

- **Main Function**:
    - `Malloc` allocates memory for source and destination arrays. The `src` array contains the source vertices, and `dst` array contains the destination vertives of edges.
    - The `bfs_gpu` is then called to allocate memory on GPU using `cudaMalloc` and call the `bfs_kernel`.
    - The kernel loop in `bfs_gpu` is responsible for managing multiple BFS levels, calling CUDA kernel to process each level and stop the traversal of graph when no new vertices were discovered in current level of traversal. 
    - The number of concurrent kernel launched is limited to avoid exceeding GPU resources, ` cudaLimitDevRuntimePendingLaunchCount` sets a limit on maximum number of pending launches of child kernels for each thread block. 
    - After kernel execution and verifying the results, the device and host memory is freed.

## Compilation and Execution

To compile and run the program, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd DynamicParallelism

2. **Compile the CUDA program**:
   ```bash
   nvcc bfs-graph.cu -o bfs

3. **Run the compiled program**:
   ```bash
   ./bfs

