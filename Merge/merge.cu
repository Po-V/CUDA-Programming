#define ELEM_PER_THREAD 6
#define THREADS_PER_BLOCK 128
#define ELEM_PER_BLOCK (ELEM_PER_THREAD*THREADS_PER_BLOCK)

__device__ void mergeSequential(float* A, float* B, float* C, unsigned int m, unsigned int n){

    unsigned int i = 0; //A
    unsigned int j =0; //B
    unsigned int k =0; //C

    while(i < m && j < n){
        if(A[i] < B[j]){
            C[k++] = A[i++];
        }else{
            C[k++] = B[j++];
        }
    }
    // if A is larger than B
    while(i<m){
        C[k++] = A[i++];
    }
    // if B is larger than A
    while(j <n){
        C[k++] = B[j++];
    }
}

__device__ unsigned int coRank(float* A, float* B, unsigned int m, unsigned int n, unsigned int k){
    
    // we put k>n for safety rather than k-n > 0. If k is unsigned int and n is unsigned int and n > k, 
    // k is going to wrap around and be a very large number
    unsigned int iLow = (k > n)?(k-n):0; // lower bound on i
    unsigned int iHigh = (m < k)?m:k; // upper bound on i

    // binary search
    while(true){
        unsigned int i = (iLow + iHigh)/2;
        unsigned int j = k-i;

        // check of guess is too high
        // if i is too low, j could be too high and out of bound
        if (i > 0 && j < n && A[i-1] > B[j]){
            iHigh = i; // lower the upper bound
            // check if guess is too low
        }else if(i > 0 && i <m && B[j-1] > A[i]){
            iLow =i; // found a new lower bound
        }else{
            // guess is correct
            return i;
        }
    }
}

// this is a memory bound kernel
__global__ void merge_kernel(float* A, float* B, float* C, unsigned int m, unsigned int n){

    unsigned int k = (blockIdx.x*blockDim.x+threadIdx.x )*ELEM_PER_THREAD;

    if(k<m + n){
        unsigned int i = coRank(A, B, m, n, k);
        unsigned int j = k-i;
        unsigned int KNext = (k + ELEM_PER_THREAD < m+ n)?(k + ELEM_PER_THREAD): (m+n);
        unsigned int iNext = coRank(A, B, m, n, KNext);
        unsigned int jNext = KNext - iNext;
        mergeSequential(&A[i], &B[j], &C[k], iNext - i, jNext - j);
    }
}

__global__ void merge_kernel_with_tiling(float* A, float* B, float* C, unsigned int m, unsigned int n){

    // find the block's segments

    // find k_block
    unsigned int kBlock = blockIdx.x*ELEM_PER_BLOCK;
    // find k_block + 1
    unsigned int kNextBlock = (blockIdx.x < gridDim.x -1 )?(kBlock + ELEM_PER_BLOCK):(m+n);
    __shared__ unsigned int iBlock;
    __shared__ unsigned int iNextBlock;
    if(threadIdx.x == 0){
        // find the coranks
        iBlock = coRank(A, B, m, n, kBlock);
        iNextBlock = coRank(A, B, m, n, kNextBlock);
    }
    __syncthreads();
    unsigned int jBlock = kBlock - iBlock;
    unsigned int jNextBlock = kNextBlock - iNextBlock;

    // load block's segments to shared memory
    __shared__ float A_s[ELEM_PER_BLOCK];
    unsigned int mBlock = iNextBlock - iBlock;
    for(unsigned int i =  threadIdx.x; i < mBlock; i += blockDim.x){
        A_s[i] = A[iBlock + i];
    }

    // B_s begins whenever A_s ends
    float* B_s = A_s + mBlock;
    unsigned int nBlock = jNextBlock - jBlock;
    for(unsigned int j = threadIdx.x; j < nBlock; j += blockDim.x){
        // load the segment of B in coalesced manner and put in shared memory
        B_s[j] = B[jBlock + j];
    }
    __syncthreads();

    // merge in shared memory
    __shared__ float C_s[ELEM_PER_BLOCK];
    unsigned int k = threadIdx.x*ELEM_PER_THREAD;
    if(k < mBlock + nBlock){
        unsigned int i = coRank(A_s, B_s, mBlock, nBlock, k);
        unsigned int j = k - i;
        unsigned int kNext = (k + ELEM_PER_THREAD < mBlock + nBlock)?(k+ELEM_PER_THREAD):(mBlock+nBlock);
        unsigned int iNext = coRank(A_s, B_s, mBlock, nBlock, kNext);
        unsigned int jNext = kNext - iNext;
        mergeSequential(&A_s[i], &B_s[j], &C_s[k], iNext - i, jNext - j);
    }
    __syncthreads();

    // write block's segment to global memory
    for(unsigned int k = threadIdx.x; k < mBlock + nBlock; k+= blockDim.x){
        C[kBlock + k] = C_s[k];
    }
}


void merge_gpu(float* A, float* B, float* C, unsigned int m, unsigned int n){

    // Allocate GPU memory
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, m*sizeof(float));
    cudaMalloc((void**) &B_d, n*sizeof(float));
    cudaMalloc((void**) &C_d, (m+n)*sizeof(float));
    cudaDeviceSynchronize();

    // copy data to GPU
    cudaMemcpy(A_d, A, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // vector addition on GPU
    unsigned int numBlocks = (m+n+ ELEM_PER_BLOCK - 1) / ELEM_PER_BLOCK;
    merge_kernel_with_tiling<<<numBlocks, THREADS_PER_BLOCK>>>(A_d, B_d, C_d, m, n);
    cudaDeviceSynchronize();

    // copy data from GPU
    cudaMemcpy(C, C_d, (m+n)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

int main(){

    unsigned int m = 1024;
    unsigned int n = 1024;
    // Allocate memory on host for vector
    float *A = (float*)malloc(m*sizeof(float));
    float *B = (float*)malloc(n*sizeof(float));
    float *C = (float*)malloc((m+n)*sizeof(float));

    // initialize input array with 1.0
    for(unsigned int i = 0; i<m; i++){
        A[i] = 1.0f;
    }

    for(unsigned int i = 0; i<n; i++){
        B[i] = 1.0f;
    }

    merge_gpu(A, B, C, m, n);

    free(A);
    free(B);
    free(C);

    return 0;
}


