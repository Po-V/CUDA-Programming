# Breadth-First Search (BFS) on graph using CUDA

This project implements a breath-first search on a graph using CUDA where graph is represented in Coordinate List (COO) format.

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `bfs_coo.cu`

- **bfs kernel**: 
    - In `bfs_kernel`, each thread processes one edge of the graph. The kernel check whether an edge connects a vertext at current level to an unvisited vertex. If such an edge is found, the destination vertex is marked as visited at the current level and the `newVertexVisited` flag is set to 1. 

- **Main Function**:
    - `Malloc` allocates memory for source and destination arrays. The `src` array contains the source vertices, and `dst` array contains the destination vertives of edges.
    - The `bfs_gpu` is then called to allocate memory on GPU using `cudaMalloc` and call the `bfs_kernel`.
    - The kernel loop in `bfs_gpu` is responsible for managing multiple BFS levels, calling CUDA kernel to process each level and stop the traversal of graph when no new vertices were discovered in current level of traversal. 
    - After kernel execution and verifying the results, the device and host memory is freed.


## Compilation and Execution

To compile and run the coo example, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd Graphs

2. **Compile the CUDA program**:
   ```bash
   nvcc bfs-coo.cu -o coo

3. **Run the compiled program**:
   ```bash
   ./coo

