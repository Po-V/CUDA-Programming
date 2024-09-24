#include <stdio.h>
#include <cuda.h>

#define BLOCK_DIM 8
#define C0 1
#define C1 2
#define IN_TITLE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TITLE_DIM -2)

__global__ void stencil_kernel(float* in, float* out, int N){

    int iStart = blockIdx.z*OUT_TILE_DIM;
    int j = blockIdx.y*OUT_TILE_DIM+threadIdx.y -1;
    int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x-1;

    // store the previous value in register
    float inPrev;
    __shared__ float inCurr_s[IN_TITLE_DIM][IN_TITLE_DIM];
    // store the next value in register
    float inNext;

    if(iStart -1 >= 0 && iStart-1< N && j >= 0 && j< N && k>= 0 && k <N){
        // store previous plane in register
        inPrev = in[(iStart - 1)*N*N + j*N + k];
    }

    if(iStart >= 0 && iStart< N && j >= 0 && j< N && k>= 0 && k <N){
        // store current plane in shared memory
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N + j*N+k];
    }

    __syncthreads();

    for(int i = iStart; i < iStart + OUT_TILE_DIM; ++i){
       
        if(i+1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N){
            // load the next plane into register
            inNext = in[(i+1)*N*N + j*N + k];
        }
        __syncthreads();

        if(i >= 1 && i < N - 1 && j >= 1 && j < N -1 && k >= 1 && k < N -1){
            if(threadIdx.y >= 1 && threadIdx.y < IN_TITLE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TITLE_DIM - 1){
                //
                out[i*N*N + j*N + k] = C0*inCurr_s[threadIdx.y][threadIdx.x] + 
                                                        C1*(inCurr_s[threadIdx.y][threadIdx.x - 1] +
                                                            inCurr_s[threadIdx.y][threadIdx.x + 1] +
                                                            inCurr_s[threadIdx.y + 1][threadIdx.x] +
                                                            inCurr_s[threadIdx.y - 1][threadIdx.x] +
                                                            inPrev + inNext );
            }
        }

        __syncthreads();
        // move current plane from shared memory to register
        inPrev = inCurr_s[threadIdx.y][threadIdx.x];
        // move next plane to current plane in shared memory
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;

    }
}
    



void stencil_gpu(float* input, float* output, int N){

    // declare device pointers
    float *in_d, *out_d;

    // allocate memory on device
    cudaMalloc((void**) &in_d, N*N*N*sizeof(float));
    cudaMalloc((void**) &out_d, N*N*N*sizeof(float));

    // copy input matrices from host to device
    cudaMemcpy(in_d, input, N*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, output, N*N*N, cudaMemcpyHostToDevice);

    // call the kernel
    dim3 numThreadsPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM); // input tile dimension is same as block_dim
    dim3 numBlocks((N + OUT_TILE_DIM - 1)/ OUT_TILE_DIM, (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (N+ OUT_TILE_DIM -1) / OUT_TILE_DIM);
    stencil_kernel<<<numBlocks, numThreadsPerBlock>>>(in_d, out_d, N);

    // copy data from device to host
    cudaMemcpy(output, out_d, N*N*N, cudaMemcpyDeviceToHost);

    // deallocate memory on device
    cudaFree(in_d);
    cudaFree(out_d);
}

int main(){

    int N = 8;

    // Allocate memory on host for matrices
    float *input = (float*)malloc(N*N*N*sizeof(float));
    float *output = (float*)malloc(N*N*N*sizeof(float));

    stencil_gpu(input, output, N);

    free(input);
    free(output);

    return 0;

}