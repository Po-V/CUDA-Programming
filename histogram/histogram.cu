#define NUM_BINS 256 // 256 possible values for 8 bit integers (unsigned char)

__global__ void histogram_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height){

    // implement privatisation. Shared memory array is declared for each block
    // this acts as a private histogram for entire block.
    __shared__ unsigned int arr[NUM_BINS];

    if(threadIdx.x < NUM_BINS){
        arr[threadIdx.x] = 0;
    }
    __syncthreads();

    //find element thread is responsible for
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i < width*height){
        unsigned char b = image[i];
        // update address of b by 1
        atomicAdd(&arr[b], 1);
    }
    __syncthreads();

    // combine shared memory results into global memory
    if(threadIdx.x < NUM_BINS){
        atomicAdd(&bins[threadIdx.x], arr[threadIdx.x]);
    }
}

void histogram_gpu(unsigned char * image, unsigned int* bins, unsigned int width, unsigned int height){

    // allocate GPU memory
    unsigned char *image_d;
    unsigned int *bins_d;
    cudaMalloc((void**) &image_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &bins_d, NUM_BINS*sizeof(unsigned int));
    cudaDeviceSynchronize();

    // copy data to GPU
    cudaMemcpy(image_d, image, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(bins_d, 0, NUM_BINS*sizeof(unsigned int));
    cudaDeviceSynchronize();

    // call kernel
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (width*height + numThreadsPerBlock - 1) / numThreadsPerBlock;
    histogram_kernel<<< numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);
    cudaDeviceSynchronize();

    // copy data from GPU
    cudaMemcpy(bins, bins_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free GPU memory
    cudaFree(image_d);
    cudaFree(bins_d);
    cudaDeviceSynchronize();
}


int main(){

    unsigned int width = 1024;
    unsigned int height = 1024;
    unsigned char img_size = width*height;
    
    // Allocate memory on host for image
    unsigned char* img = (unsigned char*)malloc(img_size*sizeof(unsigned char));
    unsigned int* bins = new unsigned int[NUM_BINS];

    // initialize image with random values
    for(unsigned int i =0; i < img_size; ++ i){
        img[i] = rand() % 256;
    }

    memset(bins, 0, NUM_BINS*sizeof(unsigned int));

    histogram_gpu(img, bins, width, height);

    free(img);
    free(bins);

    return 0;
}