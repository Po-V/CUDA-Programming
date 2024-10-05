
#include "common.h"

__global__ void spmv_coo_kernel(COOMatrix cooMatrix, float* inVector, float* outVector){
    
    // index of non-zero elements
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

    // boundary check to ensure thread passes a valid non-zero
    if(i < cooMatrix.numNonZeros){
        // i is the indexes of the non-zero values
        unsigned int row = cooMatrix.rowIdxs[i];
        unsigned int col = cooMatrix.colIdxs[i];
        float value = cooMatrix.values[i];
        atomicAdd(&outVector[row] ,inVector[col]*value);
    }
}

void spmv_coo_gpu(COOMatrix cooMatrix, float* inVector, float* outVector){

    // Allocate GPU memory
    COOMatrix cooMatrix_d;
    cooMatrix_d.numRows = cooMatrix.numRows;
    cooMatrix_d.numCols = cooMatrix.numCols;
    cooMatrix_d.numNonZeros = cooMatrix.numNonZeros;
    cudaMalloc((void**) &cooMatrix_d.rowIdxs, cooMatrix_d.numNonZeros*sizeof(unsigned int));
    cudaMalloc((void**) &cooMatrix_d.colIdxs, cooMatrix_d.numNonZeros*sizeof(unsigned int));
    cudaMalloc((void**) &cooMatrix_d.values, cooMatrix_d.numNonZeros*sizeof(float));
    float* inVector_d;
    cudaMalloc((void**) &inVector_d, cooMatrix_d.numCols*sizeof(float));
    float* outVector_d;
    cudaMalloc((void**) &outVector_d, cooMatrix_d.numRows*sizeof(float));
    cudaDeviceSynchronize();

    // copy data to GPU
    cudaMemcpy(cooMatrix_d.rowIdxs, cooMatrix.rowIdxs, cooMatrix_d.numNonZeros*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cooMatrix_d.colIdxs, cooMatrix.colIdxs, cooMatrix_d.numNonZeros*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cooMatrix_d.values, cooMatrix.values, cooMatrix_d.numNonZeros*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inVector_d, inVector, cooMatrix_d.numCols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(outVector_d, 0, cooMatrix_d.numRows*sizeof(float));
    cudaDeviceSynchronize();

    // Call kernel
    unsigned int numTHreadsPerBlock = 1024;
    unsigned int numBlocks = (cooMatrix_d.numNonZeros + numTHreadsPerBlock - 1) / numTHreadsPerBlock;
    spmv_coo_kernel<<<numBlocks, numTHreadsPerBlock>>>(cooMatrix_d, inVector_d, outVector_d);
    cudaDeviceSynchronize();

    // copy data from GPU
    cudaMemcpy(outVector, outVector_d, cooMatrix_d.numRows*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Free GPU memory
    cudaFree(cooMatrix_d.rowIdxs);
    cudaFree(cooMatrix_d.colIdxs);
    cudaFree(cooMatrix_d.values);
    cudaFree(inVector_d);
    cudaFree(outVector_d);
    cudaDeviceSynchronize();
}


int main(){

    COOMatrix cooMatrix;
    cooMatrix.numRows = 4;
    cooMatrix.numCols = 4;
    cooMatrix.numNonZeros = 8;

    // Allocate memory for COO matrix
    cooMatrix.rowIdxs = (unsigned int*)malloc(cooMatrix.numNonZeros*sizeof(unsigned int));
    cooMatrix.colIdxs = (unsigned int*)malloc(cooMatrix.numNonZeros*sizeof(unsigned int));
    cooMatrix.values = (float*)malloc(cooMatrix.numNonZeros*sizeof(float));

    unsigned int rowIdxs[] = {0,0,1,1,1,2,2,3};
    unsigned int colIdxs[] = {0,1,0,2,3,1,2,3};
    float values[] = {1.0,7.0,5.0,3.0,9.0,2.0,8.0,6.0};

    for(int i = 0; i < cooMatrix.numNonZeros; i++){
        cooMatrix.rowIdxs[i] = rowIdxs[i];
        cooMatrix.colIdxs[i] = colIdxs[i];
        cooMatrix.values[i] = values[i];
    }

    float* inVector = (float*)malloc(cooMatrix.numCols*sizeof(float));
    for (int i = 0; i < cooMatrix.numCols; i++) {
        inVector[i] = 1.0f;  // Initialize with 1.0 for simplicity
    }

    // Output vector
    float* outVector = (float*)malloc(cooMatrix.numRows * sizeof(float));

    // Perform SpMV on GPU
    spmv_coo_gpu(cooMatrix, inVector, outVector);

    // Free memory
    free(cooMatrix.rowIdxs);
    free(cooMatrix.colIdxs);
    free(cooMatrix.values);
    free(inVector);
    free(outVector);

    return 0;
}
