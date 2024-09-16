// This program computes the sum of 2 arrays on the GPU using CUDA

#include <cstdlib>

// computes the sum of 2 arrays
__global__ void vecAdd(int *a, int *b, int *c, int N){
    // void because it does not return anything from the GPU
    // calculate the global thread ID

    int tid = (blockIdx.x * blockDim.x )+ threadIdx.x;

    if(tid < N)
        c[tid] = a[tid] + b[tid];
}

// Initialize an array of size "N" with numbers between 0 and 100
void init_array(int *a, int N){
    for(int i = 0; i < N; i++){
        a[i] = rand() % 100;
    }
}

int main(){
    // set our problem size (Default = 2^20)
    int N = 1 << 10;
    size_t bytes = N * sizeof(int);

    // Allocate memory for input/outputs

    int *a, *b, *c;
    cudaMallocManaged(&a, bytes); // this runs on both host and device so need to be synchronized later
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // initialize data
    init_array(a, N);
    init_array(b, N);

    // initialize CTA and Grid dimension
    int THREADS = 256;
    int BLOCKS = (N + THREADS - 1)/ THREADS;

    // Call the kernel
    vecAdd<<<BLOCKS, THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize(); // kernels calls are asynchronous and we are using unified memory

    return 0;

}