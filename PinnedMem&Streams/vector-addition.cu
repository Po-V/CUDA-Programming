// This program computes the sum of 2 arrays on the GPU using CUDA with pinned memory and streams

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>


// CUDA kernel function for vector addition
__global__ void vecAdd_kernel(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

// GPU function to set up and launch the kernel
void vecAdd_gpu(float *a, float *b, float *c, int n)
{
    float *a_d, *b_d, *c_d;
    int size = n * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&a_d, size);
    cudaMalloc((void **)&b_d, size);
    cudaMalloc((void **)&c_d, size);
    cudaDeviceSynchronize();

    unsigned int numStreams = 10; // divide input into 32 segments
    cudaStream_t stream[numStreams];
    for(unsigned int s = 0; s < numStreams; ++s){
        // create a stream
        cudaStreamCreate(&stream[s]);
    }


    // stream the segments
    unsigned int numSegments = numStreams;
    unsigned int segmentSize = (n + numSegments -1) / numSegments;

    // determine input for each segment
    for(unsigned int s = 0; s < numSegments; ++s){

        // finding the segment bounds
        unsigned int start = s*segmentSize;
        // finding end of segment and check if end execeed boundary
        unsigned int end = (start + segmentSize < n)?(start + segmentSize):n;
        // number of elements in a segment
        unsigned int Nsegment = end - start;

        // Copy input data from host to device. copy from offset of start
        cudaMemcpyAsync(&a_d[start], &a[start], Nsegment*sizeof(float), cudaMemcpyHostToDevice, stream[s]);
        cudaMemcpyAsync(&b_d[start], &b[start], Nsegment*sizeof(float), cudaMemcpyHostToDevice, stream[s]);

        // Launch the kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        // kernel calls by definition is asynchronous. Kernel would not start before cudaMemcpyAsync completed
        vecAdd_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream[s]>>>(&a_d[start], &b_d[start], &c_d[start], Nsegment);

        
        // Copy result back to host
        cudaMemcpyAsync(&c[start], &c_d[start], Nsegment*sizeof(float), cudaMemcpyDeviceToHost, stream[s]);
    }
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaDeviceSynchronize();

}

// Function to initialize array with random float values
void initializeRandomArray(float *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = (float)rand() / RAND_MAX * 10.0f;  // Random float between 0 and 10
    }
}

// Initialize an array of size "N" with numbers between 0 and 100
void init_array(float *a, int N){
    for(int i = 0; i < N; i++){
        a[i] = (float) rand() / RAND_MAX * 10.0f;
    }
}


// Main function
int main()
{
    srand(time(NULL));  // Seed for random number generation
    int N = 10;

    // Allocate memory for vectors
    float *a ; cudaMallocHost((void**) &a, N * sizeof(float));
    float *b ; cudaMallocHost((void**) &b, N * sizeof(float));
    float *c ; cudaMallocHost((void**) &c, N * sizeof(float));

    // Initialize input vectors with random values
    init_array(a, N);
    init_array(b, N);

    // call kernel
    vecAdd_gpu(a, b, c, N);

    // Free host memory
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);

    return 0;
}