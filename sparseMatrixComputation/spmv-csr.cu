#include "common.h"

__global__ void spmv_csr_kernel(CSRMatrix csrMatrix, float* inVector, float* outVector){
    
    // index of non-zero elements
    unsigned int row = blockIdx.x*blockDim.x+threadIdx.x;

    // boundary check to ensure rows are valid
    if(row < csrMatrix.numRows){       
        float sum = 0.0f;
        for(unsigned int i = csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row + 1]; ++i){
            unsigned int col = csrMatrix.colIdxs[i];
            float value = csrMatrix.values[i];
            sum += value*inVector[col];
        }
        outVector[row] = sum;
    }
}

void spmv_csr_gpu(CSRMatrix csrMatrix, float* inVector, float* outVector){

    // Allocate GPU memory
    CSRMatrix csrMatrix_d;
    csrMatrix_d.numRows = csrMatrix.numRows;
    csrMatrix_d.numCols = csrMatrix.numCols;
    csrMatrix_d.numNonZeros = csrMatrix.numNonZeros;
    cudaMalloc((void**) &csrMatrix_d.rowPtrs, (csrMatrix_d.numRows+1)*sizeof(unsigned int));
    cudaMalloc((void**) &csrMatrix_d.colIdxs, csrMatrix_d.numNonZeros * sizeof(unsigned int));
    cudaMalloc((void**) &csrMatrix_d.values, csrMatrix_d.numNonZeros * sizeof(float));
    float* inVector_d;
    cudaMalloc((void**) &inVector_d, csrMatrix_d.numCols*sizeof(float));
    float* outVector_d;
    cudaMalloc((void**) &outVector_d, csrMatrix_d.numRows*sizeof(float));
    cudaDeviceSynchronize();

    // copy data to GPU
    cudaMemcpy(csrMatrix_d.rowPtrs, csrMatrix.rowPtrs, (csrMatrix_d.numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);  
    cudaMemcpy(csrMatrix_d.colIdxs, csrMatrix.colIdxs, csrMatrix_d.numNonZeros * sizeof(unsigned int), cudaMemcpyHostToDevice);  
    cudaMemcpy(csrMatrix_d.values, csrMatrix.values, csrMatrix_d.numNonZeros*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inVector_d, inVector, csrMatrix_d.numCols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(outVector_d, 0, csrMatrix_d.numRows*sizeof(float));
    cudaDeviceSynchronize();

    // Call kernel
    unsigned int numTHreadsPerBlock = 1024;
    unsigned int numBlocks = (csrMatrix_d.numNonZeros + numTHreadsPerBlock - 1) / numTHreadsPerBlock;
    spmv_csr_kernel<<<numBlocks, numTHreadsPerBlock>>>(csrMatrix_d, inVector_d, outVector_d);
    cudaDeviceSynchronize();

    // copy data from GPU
    cudaMemcpy(outVector, outVector_d, csrMatrix_d.numRows*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Free GPU memory
    cudaFree(csrMatrix_d.rowPtrs);
    cudaFree(csrMatrix_d.colIdxs);
    cudaFree(csrMatrix_d.values);
    cudaFree(inVector_d);
    cudaFree(outVector_d);
    cudaDeviceSynchronize();
}

//void spmv_coo_gpu(COOMatrix cooMatrix, float*inVector, float* outVector);

int main(){

    CSRMatrix csrMatrix;
    csrMatrix.numRows = 4;
    csrMatrix.numCols = 4;
    csrMatrix.numNonZeros = 8;

    // Allocate memory for COO matrix
    csrMatrix.rowPtrs = (unsigned int*)malloc((csrMatrix.numRows + 1) * sizeof(unsigned int));
    csrMatrix.colIdxs = (unsigned int*)malloc(csrMatrix.numNonZeros*sizeof(unsigned int));
    csrMatrix.values = (float*)malloc(csrMatrix.numNonZeros*sizeof(float));

    unsigned int rowPtrs[] = {0,2,5,7,8};
    unsigned int colIdxs[] = {0,1,0,2,3,1,2,3};
    float values[] = {1.0, 7.0, 5.0, 3.0, 9.0, 2.0, 8.0, 6.0};

    for(int i = 0; i <= csrMatrix.numRows; i++){
        csrMatrix.rowPtrs[i] = rowPtrs[i];
    }
    
    for(int i = 0; i < csrMatrix.numNonZeros; i++){
        csrMatrix.colIdxs[i] = colIdxs[i];
        csrMatrix.values[i] = values[i];
    }

    float* inVector = (float*)malloc(csrMatrix.numCols*sizeof(float));
    for (int i = 0; i < csrMatrix.numCols; i++) {
        inVector[i] = 1.0f;  // Initialize with 1.0 for simplicity
    }

    // Output vector
    float* outVector = (float*)malloc(csrMatrix.numRows * sizeof(float));

    // Perform SpMV on GPU
    spmv_csr_gpu(csrMatrix, inVector, outVector);

    // Free memory
    free(csrMatrix.rowPtrs);
    free(csrMatrix.colIdxs);
    free(csrMatrix.values);
    free(inVector);
    free(outVector);

    return 0;
}
