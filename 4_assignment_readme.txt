=====================================================================
CS380 GPU and GPGPU Programming, KAUST
Programming Assignment #4
Image Processing with CUDA

Contacts: 
peter.rautek@kaust.edu.sa
=====================================================================

Tasks:

1. Image Processing with CUDA
- implement the same image processing operations as in Assignment #3: brightness, contrast, saturation, smoothing, edge detection, sharpening), but this time using CUDA
- implement mean (box) filtering and Gaussian smoothing.
- implement larger convolution kernels (5x5, 7x7, 9x9, ...) for the smoothing operation. Find out what is the largest possible kernel size your code can successfully run. 

Use a c-style array as input to the CUDA kernels. 
cudaMalloc is used to allocate memory and cudaMemcpy is used to initialize device memory. 
You can (i) either re-use your framework that you have developed so far, (ii) start completely from scratch, or (iii) use the provided framework.

2. Export Images
Either use a library to export the images to a common file format on disk or visualize them on screen using the provided CImg library [3] or OpenGL.

3. Profiling
Measure the time it takes to apply the image processing operations (See: 'Timing using CUDA Events' at [2]). 
Measure the time of the CUDA kernel only (without memory initialization, transfer, ...)!
3.a) Run the Gaussian smoothing filter with increasing kernel size on a fixed image (e.g., 1024x1024) to analyze the scaling behavior 
3.b) Run a with fixed size smoothing filter (e.g., 25x25) on randomly initialized images of increasing size (128x128, ..., 1024x1024, ...) to analyze the scaling behavior.

4. Implement the convolution for small filter kernels (e.g., 9x9) using shared memory on the GPU. 
Think about how to most effectively share computations between threads. 
In Chapter 7.6 'Tiled 2D Convolution With Halo Cells' of 'Massively Parallel Processors, 3rd Edition' [4], a simple method to benefit from faster access times of shared memory is described.

5. Benchmark different methods (global vs. local size, constant memory, shared memory, loop unrolling, ...) 
5.a) Benchmark for different global/local sizes
5.b) Benchmark with and without the use of constant memory (see Chapter 7 [4])
5.c) Benchmark with and without the use of shared memory
5.d) Document the attempts to make it faster (why did it (not) become faster?).

6. Submit your program and a report including result images and profiling results for the different image processing operations.

BONUS 1: Implement a single-threaded and/or multi-threaded CPU version of the convolution. Benchmark it and compare performance with the GPU implementation.

BONUS 2: Use your convolution as it is used for neural networks (for instance for image classification) in CUDA. 
Implement convolution for (only) one layer of a neural network. 
a) Load the weights of the first layer of a pre-trained VGG16 [1] deep neural network. 
The weights are in the file 'data/vgg16_layer_1.raw'. 
The data is in binary float32 format. It contains the weights of 64 3x3 kernels for the 3 channels RGB (total number of floats=64x3x3x3). 
The array is sorted 'RGB channel last'.
b) Implement convolution of one 3x3 kernel with an RGB image (of your choice). 
Hints: 	(i)  Kernel values can also be negative. 
		(ii) Convert the datatype of your image to float for convolution
		(iii) Apply a suitable color map to scale the value range of the resulting images to the unsigned char range (0-255)
c) Iterate over all 64 kernels and produce one response image for each kernel and store them on disk. 


References and Acknowledgments:
[1] Karen Simonyan, Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition.
url: https://arxiv.org/abs/1409.1556
full dataset: https://www.kaggle.com/keras/vgg16

[2] NVIDIA - CUDA Performance Profiling: https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/

[3] CImg library: http://cimg.eu/

[4] Chapter 7.6 'Tiled 2D Convolution With Halo Cells' in 'Massively Parallel Processors, 3rd Edition':
https://learning.oreilly.com/library/view/Programming+Massively+Parallel+Processors,+3rd+Edition/9780128119877/xhtml/chp007.xhtml#s0020

The provided textures are from http://www.grsites.com/
