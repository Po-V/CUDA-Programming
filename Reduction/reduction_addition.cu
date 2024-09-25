#include <stdio.h>
#include <cuda.h>

#define BLOCK_DIM 1024
#define COARSE_FACTOR 4

__global__ void reduce_kernel_with_coarsening(float* input, float* partialSums, unsigned int N){

    // blockIdx.x * blockDim.x gives us the starting thread index for the current block.
    // multiplication by 2 as each thread is responsible to sum up 2 elements
    // the coarse factor further multiplies this as each thread will handle multiple elements due to coarsening
    unsigned int segment = blockIdx.x*blockDim.x*2*COARSE_FACTOR;

    // segment is the starting point for current block in input array
    // threadIdx.x is the local thread index within a block
    // calculates global index of thread. Thread 0 starts at index segment, thread 1 at index segment +1 and so on.
    unsigned int i = segment + threadIdx.x;

    __shared__ float input_s[BLOCK_DIM];
    float sum =0.0f;
    for(unsigned int tile = 0; tile < COARSE_FACTOR*2; ++tile){
        // i is the starting index for thread. calculated as segment + threadIdx.x
        // block_dim is number of threads per block
        // each thread is not processing contiguous elements, instead its processing elements block_dim apart
        // save sum in register
        sum += input[i+tile*BLOCK_DIM];
    }
    // move sum to shared memory
    input_s[threadIdx.x] = sum;
    __syncthreads();

    // reduction loop
    // starts with stride half of block dim and at each iteration stride is halved
    // the loop continues until stride becomes 0.
    for(unsigned int stride = BLOCK_DIM/2; stride > 0; stride /= 2){
        // ensures that only first stride threads in block performs computation in each iteration
        if(threadIdx.x < stride){
            // each active thread adds value from stride positions ahead of its own position
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        // the first thread in each block writes the final sum to partialSums array
        partialSums[blockIdx.x] = input_s[threadIdx.x];
    }

}


__global__ void reduce_kernel(float* input, float * partial_sum, unsigned int N){

    unsigned int segment = blockIdx.x*blockDim.x*2;
    // index of element thread is responsible for in global array
    // thread 4 for ex will be responsible for element 4, hence threadIdx.x *2
    unsigned int i = segment + threadIdx.x * 2;

    for(unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2){

        // guard for determining threads that are active for calculation
        if(threadIdx.x % stride == 0){
            // at every iteration, every thread will add to element that is stride to the right
            input[i] += input[i+stride];
        }
        __syncthreads();
    
    }

    if(threadIdx.x == 0){
        // block 0 stores partial sum of segment 0 and so on. 
        partial_sum[blockIdx.x] = input[i];
    }

}

float reduce_gpu(float* input, unsigned int N){

    //Allocate memory
    float *input_d;
    cudaMalloc((void**) &input_d, N*sizeof(float));
    cudaDeviceSynchronize();

    //Copy data to GPU
    cudaMemcpy(input_d, input, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    //Allocate partial sums
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    // the number of elements threads are responsible for are 2x the number of threads
    const unsigned int numElementsPerBlock = 2*numThreadsPerBlock*COARSE_FACTOR;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/ numElementsPerBlock;
    float* partial_sums = (float*) malloc(numBlocks*sizeof(float));
    float *partial_sums_d;
    cudaMalloc((void**) &partial_sums_d, numBlocks*sizeof(float));
    cudaDeviceSynchronize();

    // call kernel
    reduce_kernel_with_coarsening<<<numBlocks, numThreadsPerBlock>>>(input_d, partial_sums_d, N);
    cudaDeviceSynchronize();

    // copy data from GPU
    cudaMemcpy(partial_sums, partial_sums_d, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // reduce partial sums on CPU
    float sum = 0.0f;
    for(unsigned int i = 0; i < numBlocks; ++i){
        sum += partial_sums[i];
    }

    // free memory
    cudaFree(input_d);
    free(partial_sums);
    cudaFree(partial_sums_d);
    cudaDeviceSynchronize();

    return sum;

}

int main(){

    unsigned int N = 1024*1024;
    // Allocate memory on host for vector
    float *input = (float*)malloc(N*sizeof(float));


    // initialize input array with 1.0
    for(unsigned int i = 0; i<N; i++){
        input[i] = 1.0f;
    }

    float sum = reduce_gpu(input, N);

    free(input);

    return 0;
}



