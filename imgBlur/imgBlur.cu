#include <stdio.h>
#include <cuda.h>

#define WIDTH 1024 // image width
#define HEIGHT 1024 // image heigth


// kernel function to perform blurring operation
__global__ void blurKernel(unsigned char *input, unsigned char *output, int width, int height){

    // calculate global row and column index
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    // ensure thread is processing a valid pixel within image boundaries
    if(Col < width && Row < height){
        int pixVal = 0; 
        int pixels = 0;
        int BLUR_SIZE = 3; // size of blur kernel (3x3)

        // iterate over blur kernel
        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow){
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE +1; ++blurCol){

                int curRow = Row + blurRow;
                int curCol = Col + blurCol;

                // check if neighbouring poxel is within image boundaries
                if (curRow > -1 && curRow < HEIGHT && curCol > -1 && curCol < WIDTH){
                    pixVal += input[curRow * WIDTH + curCol];
                    pixels++;
                }
            }
        }

        output[Row*WIDTH +Col] = (unsigned char) (pixVal / pixels); // calculate average of pixel value
    }



}

// function to initialize array with random values
void init_array(unsigned char *img, int imgSize){

    for(int i = 0; i < imgSize; i++){
        img[i] = rand() % 256;
    }
}

int main(){

    int imgSize = WIDTH*HEIGHT*3; // calculate total size of image data
    unsigned char *h_input_Img = (unsigned char *)malloc(imgSize); // allocate memory for input image on host
    unsigned char *h_output_Img = (unsigned char *)malloc(imgSize); // allocate memory for output image on host

    init_array(h_input_Img, imgSize);
    init_array(h_output_Img, imgSize);

    unsigned char *d_input_Img, *d_output_Img; // pointers declarations for device memory

    cudaMalloc((void **)&d_input_Img, imgSize); // allocate memory on device for input image
    cudaMalloc((void **)&d_output_Img, imgSize); // allocate memory on device for output image

    cudaMemcpy(d_input_Img, h_input_Img, imgSize, cudaMemcpyHostToDevice); // copy image from host to device

    dim3 blockDim(16, 16); // create instance of dim3 that defines dimensions of block as 16x16
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y); // create instance of dim3 that defines dimensions of grid
    blurKernel<<<gridDim, blockDim>>>(d_input_Img, d_output_Img, WIDTH, HEIGHT); // launch kernel with specified grid and block dimension

    cudaMemcpy(h_output_Img, d_output_Img, imgSize, cudaMemcpyDeviceToHost); // copy output image from device to host

    cudaFree(d_input_Img); // free device memory for input image
    cudaFree(d_output_Img); // free device memory for output image
    free(h_input_Img); // free host memory for input image
    free(h_output_Img); // free host memory for output image


    return 0;
}