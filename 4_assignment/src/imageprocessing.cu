#include "imageprocessing.cuh"
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <sstream>
#include <cuda_runtime.h>

// TODO: read about the CUDA programming model: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
// If everything is setup correctly, this file is compiled by the CUDA/C++ compiler (that is different from the C++ compiler).
// The CUDA/C++ compiler understands certain things that your C++ compiler doesn't understand - like '__global__', 'threadIdx', and function calls with triple-angle brackets, e.g., testArray<<<...>>>();
#define RIDX(X, Y, H, W) (X*W+Y)
#define GIDX(X, Y, H, W) ((H+X)*W+Y)
#define BIDX(X, Y, H, W) ((H*2+X)*W+Y)

#define KERNEL_SIZE 9
__constant__ float KERNEL[KERNEL_SIZE*KERNEL_SIZE];     // define a constant memory

__global__
void _brighten(unsigned char *out_image, unsigned char *in_image, float brightnessfactor, int height, int width) {
    // name the output before input as CUDA style
    int pos_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pos_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (pos_x >= width || pos_y >= height) return;

    // images are stacked horizontally
    int pos_r = pos_x * width + pos_y;  // get the position of r channel
    int pos_g = (height + pos_x) * width + pos_y;  // get the position of r channel
    int pos_b = (height * 2 + pos_x) * width + pos_y;  // get the position of r channel

    float val_r, val_g, val_b;
    // change brightness
    val_r = in_image[pos_r] * brightnessfactor;
    val_g = in_image[pos_g] * brightnessfactor;
    val_b = in_image[pos_b] * brightnessfactor;


    // check legality, assume it is 8 bit image
    if (val_r > 255)
        val_r = 255;
    if (val_g > 255)
        val_g = 255;
    if (val_b > 255)
        val_b = 255;

    out_image[pos_r] = val_r;
    out_image[pos_g] = val_g;
    out_image[pos_b] = val_b;
}

__global__
void _contrast(unsigned char *out_image, unsigned char *in_image, float contrastfactor, int height, int width) {
    int pos_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pos_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (pos_x >= width || pos_y >= height) return;

    // images are stacked horizontally
    int pos_r = pos_x * width + pos_y;  // get the position of r channel
    int pos_g = (height + pos_x) * width + pos_y;  // get the position of r channel
    int pos_b = (height * 2 + pos_x) * width + pos_y;  // get the position of r channel

    float val_r, val_g, val_b;
    val_r = in_image[pos_r] * contrastfactor + (1 - contrastfactor) * 255;
    val_g = in_image[pos_g] * contrastfactor + (1 - contrastfactor) * 255;
    val_b = in_image[pos_b] * contrastfactor + (1 - contrastfactor) * 255;


    // check legality, assume it is 8 bit image
    if (val_r > 255)
        val_r = 255;
    if (val_g > 255)
        val_g = 255;
    if (val_b > 255)
        val_b = 255;

    out_image[pos_r] = val_r;
    out_image[pos_g] = val_g;
    out_image[pos_b] = val_b;
}

__global__ void
_saturation(unsigned char *out_image, unsigned char *in_image, float saturationfactor, int height, int width) {
    int pos_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pos_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (pos_x >= width || pos_y >= height) return;

    // images are stacked horizontally
    int pos_r = pos_x * width + pos_y;  // get the position of r channel
    int pos_g = (height + pos_x) * width + pos_y;  // get the position of r channel
    int pos_b = (height * 2 + pos_x) * width + pos_y;  // get the position of r channel

    // intensity
    // lumCoeff: 0.2125, 0.7154, 0.0721
    float intensity = (0.2125 * in_image[pos_r] + 0.7154 * in_image[pos_g] + 0.0721 * in_image[pos_b]);

    float val_r, val_g, val_b;
    val_r = in_image[pos_r] * saturationfactor + (1 - saturationfactor) * intensity;
    val_g = in_image[pos_g] * saturationfactor + (1 - saturationfactor) * intensity;
    val_b = in_image[pos_b] * saturationfactor + (1 - saturationfactor) * intensity;

    // check legality, assume it is 8 bit image
    if (val_r > 255)
        val_r = 255;
    if (val_g > 255)
        val_g = 255;
    if (val_b > 255)
        val_b = 255;

    out_image[pos_r] = val_r;
    out_image[pos_g] = val_g;
    out_image[pos_b] = val_b;
}

__global__ void
_sharpen(unsigned char *out_image, unsigned char *in_image, float sharpenfactor, float *conv_kernel, int length,
         int height, int width) {
    int pos_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pos_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (pos_x >= width || pos_y >= height) return;
    int l = 2 * length + 1;

    float tmpr, tmpg, tmpb, originr, origing, originb;
    originr = ((float) in_image[RIDX(pos_x, pos_y, height, width)]) / 255.0f;
    origing = ((float) in_image[GIDX(pos_x, pos_y, height, width)]) / 255.0f;
    originb = ((float) in_image[BIDX(pos_x, pos_y, height, width)]) / 255.0f;
    tmpr = tmpg = tmpb = 0.0f;
    for (int i = (-length); i <= length; i++) {
        for (int j = (-length); j <= length; j++) {
            int convidx = (i + length) * l + j + length;
            if (pos_x + i < 0 || pos_x + i >= height || pos_y + j < 0 || pos_y + j >= width) {
                tmpr += conv_kernel[convidx] * originr;
                tmpg += conv_kernel[convidx] * origing;
                tmpb += conv_kernel[convidx] * originb;
            } else {
                tmpr += conv_kernel[convidx] * ((float) in_image[RIDX((pos_x + i), (pos_y + j), height, width)]) /
                        255.0f;
                tmpg += conv_kernel[convidx] * ((float) in_image[GIDX((pos_x + i), (pos_y + j), height, width)]) /
                        255.0f;
                tmpb += conv_kernel[convidx] * ((float) in_image[BIDX((pos_x + i), (pos_y + j), height, width)]) /
                        255.0f;
            }
        }
    }
    tmpr = (originr + sharpenfactor * tmpr)*255;
    tmpg = (origing + sharpenfactor * tmpg)*255;
    tmpb = (originb + sharpenfactor * tmpb)*255;

    // check legality, assume it is 8 bit image
    if (tmpr > 255)
        tmpr = 255;
    if (tmpg > 255)
        tmpg = 255;
    if (tmpb > 255)
        tmpb = 255;

    out_image[RIDX(pos_x, pos_y, height, width)] = tmpr;
    out_image[GIDX(pos_x, pos_y, height, width)] = tmpg;
    out_image[BIDX(pos_x, pos_y, height, width)] = tmpb;
}


__global__ void
_conv2d(unsigned char *out_image, unsigned char *in_image, float *conv_kernel, int length, int height, int width) {
    int pos_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pos_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (pos_x >= width || pos_y >= height) return;
    int l = 2 * length + 1;

    float tmpr, tmpg, tmpb, originr, origing, originb;
    originr = ((float) in_image[RIDX(pos_x, pos_y, height, width)]) / 255.0f;
    origing = ((float) in_image[GIDX(pos_x, pos_y, height, width)]) / 255.0f;
    originb = ((float) in_image[BIDX(pos_x, pos_y, height, width)]) / 255.0f;
    tmpr = tmpg = tmpb = 0.0f;
    for (int i = (-length); i <= length; i++) {
        for (int j = (-length); j <= length; j++) {
            int convidx = (i + length) * l + j + length;
            if (pos_x + i < 0 || pos_x + i >= height || pos_y + j < 0 || pos_y + j >= width) {
                tmpr += conv_kernel[convidx] * originr;
                tmpg += conv_kernel[convidx] * origing;
                tmpb += conv_kernel[convidx] * originb;
            } else {
                // min
                tmpr += conv_kernel[convidx] * ((float) in_image[RIDX((pos_x + i), (pos_y + j), height, width)]) /
                        255.0f;
                tmpg += conv_kernel[convidx] * ((float) in_image[GIDX((pos_x + i), (pos_y + j), height, width)]) /
                        255.0f;
                tmpb += conv_kernel[convidx] * ((float) in_image[BIDX((pos_x + i), (pos_y + j), height, width)]) /
                        255.0f;
            }
        }
    }
    tmpr = tmpr*255;
    tmpg = tmpg*255;
    tmpb = tmpb*255;
    // check legality, assume it is 8 bit image
    if (tmpr > 255)
        tmpr = 255;
    if (tmpg > 255)
        tmpg = 255;
    if (tmpb > 255)
        tmpb = 255;

    out_image[RIDX(pos_x, pos_y, height, width)] = tmpr;
    out_image[GIDX(pos_x, pos_y, height, width)] = tmpg;
    out_image[BIDX(pos_x, pos_y, height, width)] = tmpb;
}


__global__ void
_conv2d_constant_memory(unsigned char *out_image, unsigned char *in_image, int height, int width) {
    // since we are using constant memory, we remote the conv_kernel from the kernel function signature.
    int pos_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pos_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (pos_x >= width || pos_y >= height) return;

    int length = KERNEL_SIZE/2;
    int l = KERNEL_SIZE;

    float tmpr, tmpg, tmpb, originr, origing, originb;
    originr = ((float) in_image[RIDX(pos_x, pos_y, height, width)]) / 255.0f;
    origing = ((float) in_image[GIDX(pos_x, pos_y, height, width)]) / 255.0f;
    originb = ((float) in_image[BIDX(pos_x, pos_y, height, width)]) / 255.0f;
    tmpr = tmpg = tmpb = 0.0f;
    for (int i = (-length); i <= length; i++) {
        for (int j = (-length); j <= length; j++) {
            int convidx = (i + length) * l + j + length;
            if (pos_x + i < 0 || pos_x + i >= height || pos_y + j < 0 || pos_y + j >= width) {
                tmpr += KERNEL[convidx] * originr;
                tmpg += KERNEL[convidx] * origing;
                tmpb += KERNEL[convidx] * originb;
            } else {
                tmpr += KERNEL[convidx] * ((float) in_image[RIDX((pos_x + i), (pos_y + j), height, width)]) /
                        255.0f;
                tmpg += KERNEL[convidx] * ((float) in_image[GIDX((pos_x + i), (pos_y + j), height, width)]) /
                        255.0f;
                tmpb += KERNEL[convidx] * ((float) in_image[BIDX((pos_x + i), (pos_y + j), height, width)]) /
                        255.0f;
            }
        }
    }
    tmpr = tmpr*255;
    tmpg = tmpg*255;
    tmpb = tmpb*255;
    // check legality, assume it is 8 bit image
    if (tmpr > 255)
        tmpr = 255;
    if (tmpg > 255)
        tmpg = 255;
    if (tmpb > 255)
        tmpb = 255;

    out_image[RIDX(pos_x, pos_y, height, width)] = tmpr;
    out_image[GIDX(pos_x, pos_y, height, width)] = tmpg;
    out_image[BIDX(pos_x, pos_y, height, width)] = tmpb;

}

__global__ void
_conv2d_shared_mem(unsigned char *out_image, unsigned char *in_image, float *conv_kernel,
                   int length, int height, int width) {
    int pos_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pos_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (pos_x >= width || pos_y >= height) return;

    const int share_size=16;
    __shared__ float sdata[share_size * share_size * 3];
    __shared__ float sconv_kernel[share_size * share_size * 3];

    if (threadIdx.x * share_size + threadIdx.y < length * length * 3)
        sconv_kernel[threadIdx.x * share_size + threadIdx.y] = conv_kernel[threadIdx.x * share_size + threadIdx.y];

    sdata[threadIdx.x * share_size + threadIdx.y] = ((float) in_image[RIDX(pos_x, pos_y, height, width)]) / 255.0f;
    sdata[threadIdx.x * share_size + threadIdx.y + share_size * share_size] = ((float) in_image[GIDX(pos_x, pos_y, height, width)]) / 255.0f;
    sdata[threadIdx.x * share_size + threadIdx.y + share_size * share_size * 2] =
            ((float) in_image[BIDX(pos_x, pos_y, height, width)]) / 255.0f;
    __syncthreads(); // then other process will be free using the shared memory based on cache mechanism.

    int l = 2 * length + 1;
    float tmpr, tmpg, tmpb, originr, origing, originb;

    originr = sdata[threadIdx.x * share_size + threadIdx.y];
    origing = sdata[threadIdx.x * share_size + threadIdx.y + share_size * share_size];
    originb = sdata[threadIdx.x * share_size + threadIdx.y + share_size * share_size * 2];
    tmpr = tmpg = tmpb = 0.0f;


    for (int i = (-length); i <= length; i++) {
        for (int j = (-length); j <= length; j++) {
            int convidx = (i + length) * l + j + length;
            if (pos_x + i < 0 || pos_x + i >= height || pos_y + j < 0 || pos_y + j >= width) {
                tmpr += sconv_kernel[convidx] * originr;
                tmpg += sconv_kernel[convidx] * origing;
                tmpb += sconv_kernel[convidx] * originb;
            } else {
                int idx = (int) threadIdx.x;
                int idy = (int) threadIdx.y;
                if (idx + i < 0 || idx + i >= 16 || idy + j < 0 || idy + j >= 16) {
                    tmpr += sconv_kernel[convidx] * ((float) in_image[RIDX((pos_x + i), (pos_y + j), height, width)]) /
                            255.0f;
                    tmpg += sconv_kernel[convidx] * ((float) in_image[GIDX((pos_x + i), (pos_y + j), height, width)]) /
                            255.0f;
                    tmpb += sconv_kernel[convidx] * ((float) in_image[BIDX((pos_x + i), (pos_y + j), height, width)]) /
                            255.0f;
                } else {
                    tmpr += sconv_kernel[convidx] * sdata[(threadIdx.x + i) * 16 + threadIdx.y];
                    tmpg += sconv_kernel[convidx] * sdata[(threadIdx.x + i) * 16 + threadIdx.y + 16 * 16];
                    tmpb += sconv_kernel[convidx] * sdata[(threadIdx.x + i) * 16 + threadIdx.y + 16 * 16 * 2];
                }
            }
        }
    }
    if (tmpr >= 1.0f) tmpr = 1.0f;
    if (tmpg >= 1.0f) tmpg = 1.0f;
    if (tmpb >= 1.0f) tmpb = 1.0f;
    out_image[RIDX(pos_x, pos_y, height, width)] = (unsigned char) (tmpr * 255.0f);
    out_image[GIDX(pos_x, pos_y, height, width)] = (unsigned char) (tmpg * 255.0f);
    out_image[BIDX(pos_x, pos_y, height, width)] = (unsigned char) (tmpb * 255.0f);
}


void brighten_func(unsigned char *out_image, unsigned char *in_image,
                    float brightness_factor, int height, int width) {
    dim3 grid(1, 512);
    dim3 block(512, 1);
    int imgproduct = height * width;

    unsigned char *cuda_out, *cuda_in;
    cudaMalloc((void **) &cuda_in, imgproduct * 3 * sizeof(unsigned char));
    cudaMalloc((void **) &cuda_out, imgproduct * 3 * sizeof(unsigned char));

    // copy the original_image into cuda
    cudaMemcpy(cuda_in, in_image, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // cudaEvent
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    _brighten <<< grid, block >>> (cuda_out, cuda_in, brightness_factor, height, width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // copy the output into the image array
    cudaMemcpy(out_image, cuda_out, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(cuda_out);
    cudaFree(cuda_in);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("contrast takes time: %f ms \n", milliseconds);
}

void contrast_func(unsigned char *out_image, unsigned char *in_image,
                  float contrastfactor, int height, int width) {
    dim3 grid(1, 512);
    dim3 block(512, 1);
    int imgproduct = height * width;

    unsigned char *cuda_out, *cuda_in;
    cudaMalloc((void **) &cuda_in, imgproduct * 3 * sizeof(unsigned char));
    cudaMalloc((void **) &cuda_out, imgproduct * 3 * sizeof(unsigned char));

    // copy the original_image into cuda
    cudaMemcpy(cuda_in, in_image, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);


    // cudaEvent
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    _contrast <<< grid, block >>> (cuda_out, cuda_in, contrastfactor, height, width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // copy the output into the image array
    cudaMemcpy(out_image, cuda_out, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(cuda_out);
    cudaFree(cuda_in);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("contrast takes time: %f ms \n", milliseconds);
}

void saturation_func(unsigned char *out_image, unsigned char *in_image,
                    float saturationfactor, int height, int width) {
    dim3 grid(1, 512);
    dim3 block(512, 1);
    int imgproduct = height * width;

    unsigned char *cuda_out, *cuda_in;
    cudaMalloc((void **) &cuda_in, imgproduct * 3 * sizeof(unsigned char));
    cudaMalloc((void **) &cuda_out, imgproduct * 3 * sizeof(unsigned char));

    // copy the original_image into cuda
    cudaMemcpy(cuda_in, in_image, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // cudaEvent
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    _saturation <<< grid, block >>> (cuda_out, cuda_in, saturationfactor, height, width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // copy the output into the image array
    cudaMemcpy(out_image, cuda_out, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(cuda_out);
    cudaFree(cuda_in);


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("saturation takes time: %f ms \n", milliseconds);
}

void sharpen_func(unsigned char *out_image, unsigned char *in_image, float sharpenfactor,
                  float *conv_kernel, int length, int height, int width) {
    assert(length % 2 == 1);
    int halflen = length / 2;

    dim3 grid(1, 512);
    dim3 block(512, 1);
    int imgproduct = height * width;

    unsigned char *cuda_out, *cuda_in;
    float *cuda_conv_kernel;
    cudaMalloc((void **) &cuda_in, imgproduct * 3 * sizeof(unsigned char));
    cudaMalloc((void **) &cuda_out, imgproduct * 3 * sizeof(unsigned char));
    cudaMalloc((void **) &cuda_conv_kernel, length * length * sizeof(float));

    // copy the original_image into cuda
    cudaMemcpy(cuda_in, in_image, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_conv_kernel, conv_kernel, length * length * sizeof(float), cudaMemcpyHostToDevice);

    // cudaEvent
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    _sharpen <<< grid, block >>> (cuda_out, cuda_in, sharpenfactor, cuda_conv_kernel, halflen, height, width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // copy the output into the image array
    cudaMemcpy(out_image, cuda_out, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(cuda_in);
    cudaFree(cuda_out);
    cudaFree(cuda_conv_kernel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("sharpening takes time: %f ms \n", milliseconds);
}

void conv2d_func(unsigned char *out_image, unsigned char *in_image, float *conv_kernel, int length,
                int height, int width) {

    assert(length % 2 == 1);
    int halflen = length / 2;

    dim3 grid(512, 512);
    dim3 block(1, 1);
    int imgproduct = height * width;

    unsigned char *cuda_out, *cuda_in;
    float *cuda_conv_kernel;
    cudaMalloc((void **) &cuda_in, imgproduct * 3 * sizeof(unsigned char));
    cudaMalloc((void **) &cuda_out, imgproduct * 3 * sizeof(unsigned char));
    cudaMalloc((void **) &cuda_conv_kernel, length * length * sizeof(float));

    // copy the original_image into cuda
    cudaMemcpy(cuda_in, in_image, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_conv_kernel, conv_kernel, length * length * sizeof(float), cudaMemcpyHostToDevice);

    // cudaEvent
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    _conv2d <<< grid, block >>> (cuda_out, cuda_in, cuda_conv_kernel, halflen, height, width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // copy the output into the image array
    cudaMemcpy(out_image, cuda_out, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(cuda_in);
    cudaFree(cuda_out);
    cudaFree(cuda_conv_kernel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("conv takes time: %f ms \n", milliseconds);
}


void conv2d_const_mem_func(unsigned char *out_image, unsigned char *in_image, float *conv_kernel,
                 int height, int width) {

    assert(KERNEL_SIZE % 2 == 1);

    dim3 grid(1, 512);
    dim3 block(512, 1);
    int imgproduct = height * width;

    unsigned char *cuda_out, *cuda_in;
    cudaMalloc((void **) &cuda_in, imgproduct * 3 * sizeof(unsigned char));
    cudaMalloc((void **) &cuda_out, imgproduct * 3 * sizeof(unsigned char));
//    cudaMalloc((void **) KERNEL, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    // copy the original_image into cuda
    cudaMemcpy(cuda_in, in_image, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // show the err returned from here
    cudaError_t mem_err = cudaMemcpyToSymbol(KERNEL, conv_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (mem_err != cudaSuccess)
        printf("Mem copy Error: %s\n", cudaGetErrorString(mem_err));

    // cudaEvent
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // could do something similar here.
    _conv2d_constant_memory <<< grid, block >>> (cuda_out, cuda_in, height, width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // copy the output into the image array
    cudaMemcpy(out_image, cuda_out, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(cuda_in);
    cudaFree(cuda_out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("conv takes time: %f ms \n", milliseconds);
}


void conv2d_shared_mem_func(unsigned char *out_image, unsigned char *in_image, float *conv_kernel, int share_size,
        int length, int height, int width) {

    assert(length % 2 == 1);
    int halflen = length / 2;

    dim3 grid((int) (height*width/16), (int) (height*width/16));
    dim3 block(16, 16);
    int imgproduct = height * width;

    unsigned char *cuda_out, *cuda_in;
    float *cuda_conv_kernel;
    cudaMalloc((void **) &cuda_in, imgproduct * 3 * sizeof(unsigned char));
    cudaMalloc((void **) &cuda_out, imgproduct * 3 * sizeof(unsigned char));
    cudaMalloc((void **) &cuda_conv_kernel, length * length * sizeof(float));

    // copy the original_image into cuda
    cudaMemcpy(cuda_in, in_image, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_conv_kernel, conv_kernel, length * length * sizeof(float), cudaMemcpyHostToDevice);

    // cudaEvent
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    _conv2d_shared_mem <<< grid, block >>> (cuda_out, cuda_in, cuda_conv_kernel, halflen, height, width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // copy the output into the image array
    cudaMemcpy(out_image, cuda_out, imgproduct * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(cuda_in);
    cudaFree(cuda_out);
    cudaFree(cuda_conv_kernel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("conv takes time: %f ms \n", milliseconds);
}

// do not use this method for anything else than verifying cuda compiled, linked and executed
__global__ void testArray(float *dst, float value) {
    unsigned int index = threadIdx.x;
    dst[index] = value;
}

void testCudaCall() {
    // quick and dirty test of CUDA setup
    const unsigned int N = 1024;
    float *device_array;
    cudaMalloc(&device_array, N * sizeof(float));
    testArray <<< 1, N >>> (device_array, -0.5f);
    float x[N];
    cudaMemcpy(x, device_array, N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "quick and dirty test of CUDA setup: " << x[0] << " " << x[1] << " " << x[1023] << std::endl;
    cudaFree(device_array);

    // initilize global
//    cudaMalloc((void **) &KERNEL, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

}
