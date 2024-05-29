#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <iostream>

__global__ void resizeImageKernel(const uchar3* input, uchar3* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x < outputWidth && y < outputHeight) {
      int srcX = static_cast<int>(x * static_cast<double>(inputWidth) / outputWidth);
      int srcY = static_cast<int>(y * static_cast<double>(inputHeight) / outputHeight);

      output[y * outputWidth + x] = input[srcY * inputWidth + srcX];
   }
}

extern "C" void resizeImage(const cv::Mat & input, cv::Mat & output, int newWidth, int newHeight) {
   uchar3* d_input;
   uchar3* d_output;

   size_t inputSize = input.rows * input.cols * sizeof(uchar3);
   size_t outputSize = newHeight * newWidth * sizeof(uchar3);

   cudaMalloc(&d_input, inputSize);
   cudaMalloc(&d_output, outputSize);

   cudaMemcpy(d_input, input.data, inputSize, cudaMemcpyHostToDevice);

   dim3 blockSize(16, 16);
   dim3 gridSize((newWidth + blockSize.x - 1) / blockSize.x, (newHeight + blockSize.y - 1) / blockSize.y);

   resizeImageKernel << <gridSize, blockSize >> > (d_input, d_output, input.cols, input.rows, newWidth, newHeight);

   cudaMemcpy(output.data, d_output, outputSize, cudaMemcpyDeviceToHost);

   cudaFree(d_input);
   cudaFree(d_output);
}

