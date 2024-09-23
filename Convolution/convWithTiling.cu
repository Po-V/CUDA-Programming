#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define OUT_TILE_DIM 32
#define MASK_DIM 5
#define MASK_RADIUS 2
#define WIDTH 1024
#define HEIGHT 1024
#define TILE_DIM (OUT_TILE_DIM + 2 * MASK_RADIUS)

__constant__ float mask_c[MASK_DIM][MASK_DIM];


__global__ void convolution_kernel_tiled(float* input, float* output, unsigned int width, unsigned int height){

    __shared__ float tile[TILE_DIM][TILE_DIM];
    int outRow = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    int outCol = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
    int inRow = outRow - MASK_RADIUS;
    int inCol = outCol - MASK_RADIUS;

    // with tiling, considering tile dimensions: input = T, output = T-M +1
    //  operations per block: (T-M+1)^2 x 2M^2 (mask elements) operations
    //  global memory loads per thread: T^2 x 4 bytes
    // ratio: ((T-M +1)^2 X 2M^2 operations) / (T62 X 4 B)) = 0.5M^2(1-(M-1)/T)^2
    // For M=5 and T=32: 9.5 operations/B (about 19x improvement)


    // load tile data into shared memory
    if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width){
        tile[threadIdx.y][threadIdx.x] = input[inRow * width + inCol];
    }else{
        // define halo cells if out of bound
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if(threadIdx.y < OUT_TILE_DIM && threadIdx.x < OUT_TILE_DIM && outRow < height && outCol < width){

        float sum = 0.0f;
        for(int maskRow = 0; maskRow < MASK_DIM; ++maskRow){
            for(int maskCol = 0; maskCol < MASK_DIM; ++maskCol){
                sum += mask_c[maskRow][maskCol]*tile[threadIdx.y + maskRow][threadIdx.x + maskCol];
            }
        }

        output[outRow*width+outCol] = sum;
    }

}


void convolution_gpu(float mask[][MASK_DIM], float* input, float* output, unsigned int width, unsigned int height){
    
    float *input_d, *output_d;

    // Allocate GPU memory
    cudaMalloc((void**) &input_d, width*height*sizeof(float));
    cudaMalloc((void**) &output_d, width*height*sizeof(float));
    cudaDeviceSynchronize();

    // Copy data to GPU
    cudaMemcpy(input_d, input, width*height*sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy mask array to GPU constant memory using source pointer to destination pointer
    cudaMemcpyToSymbol(mask_c, mask, MASK_DIM*MASK_DIM*sizeof(float));
    cudaDeviceSynchronize();

    // call Kernel
    // TILE_DIM to ensure there is enough threads to load input tile 
    dim3 numThreadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM -1)/ OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    convolution_kernel_tiled<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, width, height);
    cudaDeviceSynchronize();

    // Copy data from GPU
    cudaMemcpy(output, output_d, width*height*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Free GPU memory
    cudaFree(input_d);
    cudaFree(output_d);
    cudaDeviceSynchronize();

}


int main(){

    float *input, *output;
    float mask[MASK_DIM][MASK_DIM];

    // Allocate host memory
    input = (float*)malloc(WIDTH*HEIGHT*sizeof(float));
    output = (float*)malloc(WIDTH*HEIGHT*sizeof(float));

    // seed random number generator
    srand(time(NULL));

    // initialize input matrix with random values between 0 and 1
    for(int i=0; i< WIDTH*HEIGHT; i++){
        input[i] = (float)rand() /RAND_MAX; 
    }

    // initialize mask with input matrix between 0 and 1
    for(int i = 0; i< MASK_DIM; i++){
        for(int j=0; j<MASK_DIM; j++){
            mask[i][j] = (float)rand() / RAND_MAX;
        }
    }

    convolution_gpu(mask, input, output, WIDTH, HEIGHT);

    free(input);
    free(output);

    return 0;
}