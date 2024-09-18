#include <stdio.h>
#include <cuda.h>

#define TILE_SIZE 32

__global__ void mm_kernel(float* A, float* B, float* C, int m, int n, int k){

    // declare shared memory for the tile for matrix A & B
    __shared__ float A_s[TILE_SIZE][TILE_SIZE]; // tile size = block size
    __shared__ float B_s[TILE_SIZE][TILE_SIZE];

    // calculate row and column indices for current thread
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    // initialize accumulator for result
    float sum = 0.0f;

    // iterate over shared dimension of a, b which is n. 
    // moves across columns of A and rows of B, tile by tile.
    // add tile_size -1 before dividing to round up the result
    for(int t = 0; t < (TILE_SIZE + n -1) / TILE_SIZE; ++t){ 
        // load data into shared memory
        if(row < m && t*TILE_SIZE + threadIdx.x < n){
            A_s[threadIdx.y][threadIdx.x] = A[row *n +t *TILE_SIZE + threadIdx.x];
        }else{
            A_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(t*TILE_SIZE+threadIdx.y <n && col<k){
            B_s[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE+threadIdx.y) *k + col];
        }else{
            B_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // synchronize threads to ensure all data is loaded
        __syncthreads();

        // perform dot product for current tile
        for (int i = 0; i < TILE_SIZE; i++){
            sum += A_s[threadIdx.y][i]* B_s[i][threadIdx.x];
        }

        __syncthreads();

    }

    if(row < m && col <k){
        C[row*k+col] = sum;
    }


}

void init_array(float* matrix, int img_size){

    for (int i = 0; i < img_size; i++) matrix[i] = 2.0f;

}

void mm_gpu(float* A, float*B, float* C, int m, int n, int k){

    // declare device pointers
    float *A_d, *B_d, *C_d;
    
    // allocate memory on device
    cudaMalloc((void**) &A_d, m*n*sizeof(float));
    cudaMalloc((void**) &B_d, n*k*sizeof(float));
    cudaMalloc((void**) &C_d, m*k*sizeof(float));

    // copy input matrices from host to device
    cudaMemcpy(A_d, A, m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, n*k, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, m*k, cudaMemcpyHostToDevice);

    // set up dim and block dimension
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    // m is the height while k is the width of matrix C
    dim3 gridDim((k+ TILE_SIZE - 1)/ TILE_SIZE, (m+TILE_SIZE-1) / TILE_SIZE);

    // Launch the kernel
    mm_kernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, m, n, k);


    // copy matrix C from device to host
    cudaMemcpy(C, C_d, m*k, cudaMemcpyDeviceToHost);

    // deallocate memory on device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){

    // matrix dimension definition
    int m = 1024;
    int n = 512;
    int k = 2048;

    //Allocate memory on host for matrices
    float *A = (float*)malloc(m*n*sizeof(float));
    float *B = (float*)malloc(n*k*sizeof(float));
    float *C = (float*)malloc(m*k*sizeof(float));

    //Allocate device memory
    init_array(A, m*n);
    init_array(B, n*k);

    mm_gpu(A, B, C, m, n, k);

    // free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}