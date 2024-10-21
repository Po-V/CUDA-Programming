#include "common.h"
#include <limits.h>

__global__ void spmv_ell_kernel(ELLMatrix ellMatrix, float* inVector, float* outVector){
    // index of starting element in row
    unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < ellMatrix.numRows){

        // accumulator to prevent accessing global memory for sum
        float sum = 0.0;
        // loop over non-zeros elements in current row
        for(unsigned int iter = 0; iter < ellMatrix.nnzPerRow[row]; ++iter){
            // get index of non-zero element
            unsigned int i = iter*ellMatrix.numRows + row;
            // get column index of current non-zero element in ELL matrix
            unsigned int col = ellMatrix.colIdxs[i];
            // get the value of current non-zero element
            float value = ellMatrix.values[i];
            sum += inVector[col]*value;
        }
        outVector[row] = sum;
    }
}

void spmv_ell_gpu(ELLMatrix ellMatrix, float* inVector, float* outVector){

    // Allocate GPU memory
    ELLMatrix ellMatrix_d;
    ellMatrix_d.numRows = ellMatrix.numRows;
    ellMatrix_d.numCols = ellMatrix.numCols;
    ellMatrix_d.maxNNZPerRow = ellMatrix.maxNNZPerRow;
    // allocates space for 1 unsigned int per row (memory needed to store number of non-zero elements for each row)
    cudaMalloc((void**) &ellMatrix_d.nnzPerRow, ellMatrix_d.numRows*sizeof(unsigned int));
    // numRows is multiplied with maxNNZPerRow instead of nnzPerRow to ensure each row in GPU memory has same length
    // here we are allocating more memory than necessary for non-zero elements but this would ensure we fit ELL format's rectangular structure.
    cudaMalloc((void**) &ellMatrix_d.colIdxs, ellMatrix_d.numRows*ellMatrix_d.maxNNZPerRow*sizeof(unsigned int));
    cudaMalloc((void**) &ellMatrix_d.values, ellMatrix_d.numRows*ellMatrix_d.maxNNZPerRow*sizeof(float));
    float* inVector_d;
    cudaMalloc((void**) &inVector_d, ellMatrix_d.numCols*sizeof(float));
    float* outVector_d;
    cudaMalloc((void**) &outVector_d, ellMatrix_d.numRows*sizeof(float));
    cudaDeviceSynchronize();

    // copy data to GPU
    cudaMemcpy(ellMatrix_d.nnzPerRow, ellMatrix.nnzPerRow, ellMatrix_d.numRows*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(ellMatrix_d.colIdxs, ellMatrix.colIdxs, ellMatrix_d.numRows*ellMatrix_d.maxNNZPerRow*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(ellMatrix_d.values, ellMatrix.values, ellMatrix_d.numRows*ellMatrix_d.maxNNZPerRow*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inVector_d, inVector, ellMatrix_d.numCols * sizeof(float), cudaMemcpyHostToDevice);

    // initialize output vector to zero
    cudaMemset(outVector_d, 0, ellMatrix_d.numRows*sizeof(float));

    // define grid and block dimensions
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (ellMatrix_d.numRows + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // launch kernel
    spmv_ell_kernel<<<numBlocks, numThreadsPerBlock>>>(ellMatrix_d, inVector_d, outVector_d);

    cudaMemcpy(outVector, outVector_d, ellMatrix_d.numRows * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(ellMatrix_d.nnzPerRow);
    cudaFree(ellMatrix_d.colIdxs);
    cudaFree(ellMatrix_d.values);
    cudaFree(inVector_d);
    cudaFree(outVector_d);
}

int main(){

    ELLMatrix ellMatrix;
    ellMatrix.numRows = 4;
    ellMatrix.numCols = 4;
    ellMatrix.maxNNZPerRow =3;
    
    // Allocate memory for ELL matrix
    ellMatrix.nnzPerRow = (unsigned int*)malloc(ellMatrix.numRows * sizeof(unsigned int));
    ellMatrix.colIdxs = (unsigned int*)malloc(ellMatrix.numRows * ellMatrix.maxNNZPerRow * sizeof(unsigned int));
    ellMatrix.values = (float*)malloc(ellMatrix.numRows * ellMatrix.maxNNZPerRow * sizeof(float));

    unsigned int nnzPerRow[] = {2,3,2,1};
    unsigned int colIdxs[] = {0,0,1,3,1,2,2,UINT_MAX,UINT_MAX,3,UINT_MAX,UINT_MAX};
    float values[] = {1.0, 5.0, 2.0, 6.0, 7.0, 3.0, 8.0, 0.0, 0.0, 9.0, 0.0, 0.0};

    memcpy(ellMatrix.nnzPerRow, nnzPerRow, ellMatrix.numRows * sizeof(unsigned int));
    memcpy(ellMatrix.colIdxs, colIdxs, ellMatrix.numRows * ellMatrix.maxNNZPerRow * sizeof(unsigned int));
    memcpy(ellMatrix.values, values, ellMatrix.numRows * ellMatrix.maxNNZPerRow * sizeof(float));

    float* inVector = (float*)malloc(ellMatrix.numCols*sizeof(float));

    for(int i = 0; i < ellMatrix.numCols; i++) {
        inVector[i] = 1.0f;  // Initialize with 1.0 for simplicity
    }

    // Output vector
    float* outVector = (float*)malloc(ellMatrix.numRows * sizeof(float));

    // Perform SpMV on GPU
    spmv_ell_gpu(ellMatrix, inVector, outVector);

    // Free memory
    free(ellMatrix.nnzPerRow);
    free(ellMatrix.colIdxs);
    free(ellMatrix.values);
    free(inVector);
    free(outVector);

    return 0;
}














