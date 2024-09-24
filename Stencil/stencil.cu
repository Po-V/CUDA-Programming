#include <stdio.h>
#include <cuda.h>

#define BLOCK_DIM 8
#define C0 1
#define C1 2
#define IN_TITLE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TITLE_DIM -2)

__global__ void stencil_kernel(float* in, float* out, int N){

    // -1 as thread 0,0 is not involved in computation but its involved in loading
    // thread 0,0 is 1 column to the left and 1 row above
    int i = blockIdx.z*OUT_TILE_DIM+threadIdx.z -1;
    int j = blockIdx.y*OUT_TILE_DIM+threadIdx.y -1;
    int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x-1;

    __shared__ float in_s[IN_TITLE_DIM][IN_TITLE_DIM][IN_TITLE_DIM];

    if(i >= 0 && i< N && j >= 0 && j< N && k>= 0 && k <N){
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N+k];
    }

    __syncthreads();

    // elements in the boundary (n=0 & n-1) would not be processed
    if(i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1){
        // boundary check to ensure only relevant threads are active
        // blockDim.x -1 is when thread is out of bounds from thread0,0
        if(threadIdx.x >= 1 &&threadIdx.x <blockDim.x -1 && threadIdx.y >= 1 && threadIdx.y < blockDim.y -1 && threadIdx.z >= 1 && threadIdx.z < blockDim.z -1){
            // every plane is NXN and to go to ith plane, it would be ixNxN
            // the size of row will be JXN, to go to element of the row k will be added
            // out[i*N*N + j*N + k] = C0*in[i*N*N + j*N +k] + C1*(in[i*N*N + j*N+(k-1)] +
            //                                                 in[i*N*N + j*N+(k+1)] +
            //                                                 in[i*N*N + (j-1)*N+k] +
            //                                                 in[i*N*N + (j+1)*N+k] +
            //                                                 in[(i-1)*N*N + j*N+k] +
            //                                                 in[(i+1)*N*N + j*N+k] );

            out[i*N*N + j*N + k] = C0*in_s[threadIdx.z][threadIdx.y][threadIdx.x] + 
                                                        C1*(in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
                                                            in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
                                                            in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
                                                            in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
                                                            in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x] +
                                                            in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] );
        }
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