#include <vector>
#include <iostream>
#include <cuda.h>

#define BLOCK_DIM 1024
#define COARSE_FACTOR 8 // every thread takes 8 elements


__global__ void scan_kernel_improved(float* input, float* output, float* partialSum, unsigned int N){


    // there are 2 segments, block segment and thread segment

    // to determine where the segment starts
    unsigned int bSegment = BLOCK_DIM*COARSE_FACTOR*blockIdx.x;

    __shared__ float buffer_s[BLOCK_DIM*COARSE_FACTOR];
    // load the entire block segment to shared memory
    for(unsigned int C=0; C < COARSE_FACTOR; ++C){
        // thread 0 will load element 0 at each segment
        // we could have had each thread load all elements in each segment but it won't have memory coalescing
        buffer_s[C*BLOCK_DIM+threadIdx.x] = input[bSegment + C*BLOCK_DIM+threadIdx.x];
    }
    __syncthreads();

    // Thread scan

    //To find beginning of thread segment. Every thread has 8 elements
    unsigned int tSegment = COARSE_FACTOR*threadIdx.x;
    // find each thread segment under buffer_s
    for(unsigned int C = 1; C< COARSE_FACTOR; ++C){
        buffer_s[tSegment + C] += buffer_s[tSegment + C -1];
    }

    // declare shared memory buffers
    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];
    float* inBuffer_s = buffer1_s;
    float* outBuffer_s = buffer2_s;

    //move all partislsum of thread block into shared memory and work in shared memory
    // this will be in the last element of thread segment
    inBuffer_s[threadIdx.x] = buffer_s[tSegment + COARSE_FACTOR -1];
    // ensure a thread has finished moving element to output array before the element can be accessed by next thread
    __syncthreads();

    // at every level stride goes from 1, 2, to 4 for a block_dim of 8
    for(unsigned int stride = 1; stride <= BLOCK_DIM/2; stride *= 2){
        // not all threads compute in every iteration. Only threads 4-1023 compute when stride is 4
        if(threadIdx.x >= stride){
            // for each current element, we add element stride to the left
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        }else{
            // transfer element from input to output buffer
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        // syncthreads require all threads in a block to reach it
        __syncthreads(); 

        // swap buffers
        float* tmp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = tmp;
    }

    // the first thread is not doing any addition
    if(threadIdx.x > 0){
        for(unsigned int C =0; C < COARSE_FACTOR; ++C){
            buffer_s[tSegment + C] += inBuffer_s[threadIdx.x -1];
        }
    }

    if(threadIdx.x == BLOCK_DIM -1){
        // the final iteration is in input buffer
        partialSum[blockIdx.x] = inBuffer_s[threadIdx.x];
    }

    __syncthreads();
    // write back to final output array
    for(unsigned int C = 0; C < COARSE_FACTOR; ++C){
        output[bSegment + C * BLOCK_DIM + threadIdx.x] = buffer_s[C * BLOCK_DIM + threadIdx.x];
    }

}

__global__ void scan_kernel(float* input, float* output, float* partialSum, unsigned int N){

    //every element will be assigned a thread
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    // declare shared memory buffer
    __shared__ float buffer_s[BLOCK_DIM];

    //move all input values into shared memory and work in shared memory
    buffer_s[threadIdx.x] = input[i];
    // ensure a thread has finished moving element to output array before the element can be accessed by next thread
    __syncthreads();

    // at every level stride goes from 1, 2, to 4 for a block_dim of 8
    for(unsigned int stride = 1; stride <= BLOCK_DIM/2; stride *= 2){
        float v;
        // not all threads compute in every iteration. Only threads 4-1023 compute when stride is 4
        if(threadIdx.x >= stride){
            // for each element, we add element stride to the left
            v = buffer_s[threadIdx.x-stride];
        }
        // syncthreads require all threads in a block to reach it
        __syncthreads(); // need to do seperate syncthreads as not all thradIdx.x has val > stride
        if(threadIdx.x >= stride){
            buffer_s[threadIdx.x] += v;
        }
        __syncthreads();
    }

    if(threadIdx.x == BLOCK_DIM -1){
        partialSum[blockIdx.x] = buffer_s[threadIdx.x];
    }
    // write back to final output array
    output[i] = buffer_s[threadIdx.x];

}

__global__ void add_kernel(float* output, float* partialSums, unsigned int N){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int bSegment = BLOCK_DIM*COARSE_FACTOR*blockIdx.x;
    if(blockIdx.x > 0){
        for(unsigned int C =0; C< COARSE_FACTOR; ++C){
            output[bSegment + C*BLOCK_DIM +threadIdx.x] += partialSums[blockIdx.x -1];
        }
    }
}


int main() {
    // Input size (for example, 1 million elements)
    unsigned int N = 1 << 10;
    size_t size = N * sizeof(float);

    // Host input and output arrays
    std::vector<float> h_input(N, 1.0f); // Initialize with 1 for simplicity
    std::vector<float> h_output(N, 0.0f);

    // Device input, output, and partial sum arrays
    float *d_input, *d_output, *d_partialSum;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_partialSum, (N / (BLOCK_DIM * COARSE_FACTOR)) * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

    // Launch scan kernel
    unsigned int numBlocks = (N + BLOCK_DIM * COARSE_FACTOR - 1) / (BLOCK_DIM * COARSE_FACTOR);
    scan_kernel_improved<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, d_partialSum, N);

    // Launch add kernel
    add_kernel<<<numBlocks, BLOCK_DIM>>>(d_output, d_partialSum, N);

    // Copy results back to host
    cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);

    // Verify the result (print the first few elements)
    for (int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partialSum);

    return 0;
}



int main() {
    // Input size (for example, 1 million elements)
    unsigned int N = 1 << 10;
    size_t size = N * sizeof(float);

    // Host input and output arrays
    std::vector<float> h_input(N, 1.0f); // Initialize with 1 for simplicity
    std::vector<float> h_output(N, 0.0f);

    // Device input, output, and partial sum arrays
    float *d_input, *d_output, *d_partialSum;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_partialSum, (N / (BLOCK_DIM * COARSE_FACTOR)) * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

    // Launch scan kernel
    unsigned int numBlocks = (N + BLOCK_DIM * COARSE_FACTOR - 1) / (BLOCK_DIM * COARSE_FACTOR);
    scan_kernel_improved<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, d_partialSum, N);

    // Launch add kernel
    add_kernel<<<numBlocks, BLOCK_DIM>>>(d_output, d_partialSum, N);

    // Copy results back to host
    cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);

    // Verify the result (print the first few elements)
    for (int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partialSum);

    return 0;
}





