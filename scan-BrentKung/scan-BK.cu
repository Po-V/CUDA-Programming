#include <iostream>

#define BLOCK_DIM 1024


__global__ void scan_kernel(float* input, float* output, float* partialSum, unsigned int N){
    //there is twice as many input elements as there are threads
    unsigned int segment = blockIdx.x*blockDim.x*2;
    __shared__ float buffer_s[2*BLOCK_DIM];
    // each thread load one element from the beginning
    buffer_s[threadIdx.x] = input[segment + threadIdx.x];
    // top half of segment
    buffer_s[threadIdx.x + BLOCK_DIM] = input[segment + threadIdx.x + BLOCK_DIM];
    __syncthreads();

    // reduction step
    // largest stride would be 2x size of array which is block_dim
    for(unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2){
        unsigned int i = (threadIdx.x +1)*2*stride -1;
        // each thread adds element its responsible for to element stride to the left
        if(i < 2*BLOCK_DIM){
            buffer_s[i] += buffer_s[i-stride];
        }
        __syncthreads();
    }

    // post reduction step
    for(unsigned int stride = BLOCK_DIM/2; stride >= 1; stride /= 2){
        unsigned int i = (threadIdx.x +1)*2*stride -1;
        // increment element stride to the right by element the thread is responsible for
        if(i+stride < 2*BLOCK_DIM){
            buffer_s[i+stride] += buffer_s[i];
        }
        __syncthreads();
    }

    if(threadIdx.x ==0){
        partialSum[blockIdx.x] = buffer_s[2*BLOCK_DIM -1];
    }
    // each thread responsible for loading 2 output values
    output[segment + threadIdx.x] = buffer_s[threadIdx.x];
    // store second half of result array
    output[segment + threadIdx.x + BLOCK_DIM] = buffer_s[threadIdx.x + BLOCK_DIM];
}


// Main function
int main() {
    unsigned int N = 1 << 10; // 
    unsigned int numBlocks = (N + (BLOCK_DIM * 2) - 1) / (BLOCK_DIM * 2); // Number of blocks

    // Allocate memory on host
    float *h_input = new float[N];
    float *h_output = new float[N];
    float *h_partialSum = new float[numBlocks];

    // Initialize input data
    for (unsigned int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;  // Example input, all ones
    }

    // Allocate memory on device
    float *d_input, *d_output, *d_partialSum;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    cudaMalloc((void**)&d_partialSum, numBlocks * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the scan kernel
    scan_kernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, d_partialSum, N);

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_partialSum, d_partialSum, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Print some results for verification (optional)
    std::cout << "First 10 output values:" << std::endl;
    for (unsigned int i = 0; i < 10; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    delete[] h_input;
    delete[] h_output;
    delete[] h_partialSum;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partialSum);

    return 0;
}






