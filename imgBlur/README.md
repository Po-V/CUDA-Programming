# CUDA image blur

This project demonstrates how to blur an image using CUDA on the GPU. The code performs image blurring using 3x3 kernel in parallel by utilizing GPU threads.

## Requirements

To run this project, you'll need:

- **CUDA Toolkit** (version 10 or later recommended)
- **NVIDIA GPU** with CUDA support

Ensure that the CUDA environment is correctly set up and your GPU drivers are installed.

### `imgBlur.cu`

- **blur Kernel**: 
    - The kernel `blurKernel` performs blurring for each pixel value by calculating the the average of neighboring pixel values using a 3x3 kernel. 
    - The kernel ensures that selected pixel is within boundary of the image height and width. 
    - The kernel uses the thread ID to compute which element each thread is responsible for in x and y dimension.

- **Initialization**:
    - The `init_array` function fills an array with random values between 0 to 255.

- **Main Function**:
    - Memory for the input and output images are allocated using CUDA's `cudaMalloc`, which allows the memory to be accessible by GPU.
    - The blurkernel is then launched with a grid of blocks and threads.
    - After kernel execution, the blurred image is copied back to host from GPU device.

## Compilation and Execution

To compile and run the program, use the following commands:

1. **Navigate to the subfolder**:
   ```bash
   cd imgBlur

2. **Compile the CUDA program**:
   ```bash
   nvcc imgBlur.cu -o imgBlur

3. **Run the compiled program**:
   ```bash
   ./imgBlur