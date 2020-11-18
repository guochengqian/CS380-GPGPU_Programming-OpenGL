#pragma once

void brighten_func(unsigned char* out_image, unsigned char* in_image, float brightfactor, int height, int width);
void contrast_func(unsigned char* out_image, unsigned char* in_image, float contrastfactor, int height, int width);
void saturation_func(unsigned char* out_image, unsigned char* in_image, float saturationfactor, int height, int width);
void sharpen_func(unsigned char* out_image, unsigned char* in_image, float sharpenfactor, float *conv_kernel, int length, int height, int width);
void conv2d_func(unsigned char* out_image,  unsigned char* in_image, float *conv_kernel, int length,int height, int width);
void conv2d_const_mem_func(unsigned char* out_image,  unsigned char* in_image, float *conv_kernel, int height, int width);
void conv2d_shared_mem_func(unsigned char* out_image,  unsigned char* in_image, float *conv_kernel, int share_size, int length, int height, int width);

void testCudaCall();
