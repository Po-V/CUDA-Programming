# Merge
This project demonstrates how perform parallel merge sort on GPU using CUDA, where 
2 sorted arrays are merged into third array.

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `merge_kernel_with_tiling.cu`

- **merge kernel with tiling**: 
    - This is optimzied version of `merge_kernel` that uses tiling and shared memory. The blocks of threads cooperatively laods segments of arrays `A` and `B` into shared memory. 
    - Compute the block-level segment (`kBlock`) and find the co-rank boundaries for this block. 
    - Load the subarrays for this block from global memory into shared memory.
    - Use `mergeSequential` to merge subarrays into shared memory. 
    - Write merged results back to global memory. 

- **Main Function**:
    - Memory for the input images are allocated using `Malloc`, which allows the memory on host device for input and output. Array A and B are initialized to 1.

## Compilation and Execution

To compile and run the program, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd Merge

2. **Compile the CUDA program**:
   ```bash
   nvcc merge.cu -o merge

3. **Run the compiled program**:
   ```bash
   ./merge

