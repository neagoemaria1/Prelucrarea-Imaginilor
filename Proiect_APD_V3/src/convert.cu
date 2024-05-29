#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <iostream>

__global__ void convertToBlackAndWhiteKernel(const uchar3* input, uchar* output, int width, int height) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x < width && y < height) {
      uchar3 pixel = input[y * width + x];
      uchar intensity = 0.299 * pixel.z + 0.587 * pixel.y + 0.114 * pixel.x;
      output[y * width + x] = intensity;
   }
}

extern "C" void convertToBlackAndWhite(const cv::Mat & input, cv::Mat & output) {
   uchar3* d_input;
   uchar* d_output;

   size_t inputSize = input.rows * input.cols * sizeof(uchar3);
   size_t outputSize = input.rows * input.cols * sizeof(uchar);

   cudaMalloc(&d_input, inputSize);
   cudaMalloc(&d_output, outputSize);

   cudaMemcpy(d_input, input.data, inputSize, cudaMemcpyHostToDevice);

   dim3 blockSize(16, 16);
   dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x, (input.rows + blockSize.y - 1) / blockSize.y);

   convertToBlackAndWhiteKernel << <gridSize, blockSize >> > (d_input, d_output, input.cols, input.rows);

   cudaMemcpy(output.data, d_output, outputSize, cudaMemcpyDeviceToHost);

   cudaFree(d_input);
   cudaFree(d_output);
}