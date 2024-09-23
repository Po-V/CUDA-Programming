#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define OUT_TILE_DIM 32
#define MASK_DIM 5
#define MASK_RADIUS 2
#define WIDTH 1024
#define HEIGHT 1024

__constant__ float mask_c[MASK_DIM][MASK_DIM];


__global__ void convolution_kernel(float* input, float* output, unsigned int width, unsigned int height){

    // get indices for outrow and outcol that thread needs to compute
    int outRow = blockIdx.y*blockDim.y+threadIdx.y;
    int outCol = blockIdx.x*blockDim.x+threadIdx.x;

    if(outRow< height && outCol < width){

        float sum = 0.0f;

        // iterate over the mask
        for(int maskRow =0; maskRow < MASK_DIM; ++maskRow){
            for(int maskCol =0; maskCol < MASK_DIM; ++maskCol){

                int inRow = outRow - MASK_RADIUS + maskRow;
                int inCol = outCol - MASK_RADIUS + maskCol;

                // boundary check over input
                // there exist control divergence 
                if(inRow < height && inRow >= 0 && inCol < width && inCol >= 0){
                    // constant memory with global memory load, M2 times 4 bytes; with 2 operations per byte (*, +)
                    // operations per thread: M^2 adds + M^2 muls = 2M^2 operations
                    // Global loads per thread: M^2 x 4B = 4M^2 B.
                    // Ratio: (2M^2 operations) / (4M^2 B) = 0.5 operations / byte
                    sum += mask_c[maskRow][maskCol]*input[inRow*width + inCol];
                }

            }
        }

        output[outRow*width + outCol] = sum;
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
    dim3 numThreadsPerBlock(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM -1)/ OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    convolution_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, width, height);
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