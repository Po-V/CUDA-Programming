# Breadth-First Search (BFS) on graph using CUDA

This project implements a breath-first search on a graph using CUDA where the graph is represented in different formats.

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `bfs_csr.cu`

- **bfs kernel**: 
    - In `bfs_kernel`, each thread processes one vertex of the graph. If the vertex belongs to previous BFS level, the thread looks at all the neighbors of that vertex by iterating over its edges in the CSR structure. For each unvisited neighbor, it marks the neighbor as visited by setting its level to the current level and the `newVertexVisited` flag is set to 1 to indicate at least one new vertex is visited. 

- **Main Function**:
    - `Malloc` allocates memory for source and destination arrays. The `src` array contains the source vertices, and `dst` array contains the destination vertives of edges.
    - The `bfs_gpu` is then called to allocate memory on GPU using `cudaMalloc` and call the `bfs_kernel`.
    - The kernel loop in `bfs_gpu` is responsible for managing multiple BFS levels, calling CUDA kernel to process each level and stop the traversal of graph when no new vertices were discovered in current level of traversal. 
    - After kernel execution and verifying the results, the device and host memory is freed.

### `bfs_csr_v2.cu`

- **bfs kernel**:
    - In the first version of bfs kernel above, each thread is responsible for checking its assigned vertex if it has not been visited `(level[vertex] == UINT_MAX)`. The neighbors are only checked if this condition is met. This approach lets threads independently look for new vertices to add to current BFS level by starting with unvisited vertices. 
    - In this kernel, each thread only processes vertices that are already in previous BFS level `(level[vertex] == currLevel -1)`. It checks each neighbor and marks any unvisited neighbor as part of current BFS level. Here threads are responsible for expanding the search only from vertices found at previous level. 
    - Kernel in `bfs_csr_v2.cu` is more efficient for sparse graphs as only vertices with an established connection to current frontier (previous level) attempt to visit new neighbors while in the original implementation, each univisited vertex is checked, regardless of whether it has any connection to previous level.

### `bfs_coo.cu`

- **bfs kernel**: 
    - In `bfs_kernel`, each thread processes one edge of the graph. The kernel checks whether an edge connects a vertex at current level to an unvisited vertex. If such an edge is found, the destination vertex is marked as visited at the current level and the `newVertexVisited` flag is set to 1. 

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

