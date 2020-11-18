// CS 380 - GPGPU Programming
// Programming Assignment #4

// system includes
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#define _USE_MATH_DEFINES
#include <sstream>
#include <algorithm>
#include <array>
# include <list>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "CImg.h"
#include "imageprocessing.cuh"

#define IMGPATH "../../data/images/baboon_full.png"
//#define IMGPATH "../../data/images/lichtenstein_full.png"
#define VGGPATH "../../data/vgg16_layer_1.raw"

bool queryGPUCapabilitiesCUDA()
{
    // Device Count
    int devCount;

    // Get the Device Count
    cudaGetDeviceCount(&devCount);

    // Print Device Count
    printf("Device(s): %i\n", devCount);

    // query and print CUDA functionality:
    // - CUDA device properties for every found GPU using cudaGetDeviceProperties():
    //   - device name
    //   - compute capability (The Compute Capability describes the features supported by a CUDA hardware.)
    //   - multi-processor count
    //   - clock rate (clock frequency in kilohertz; Also known as engine clock, GPU clock speed indicates how fast the cores of a graphics processing unit (GPU) are. The function of these cores is to render graphics; therefore, the higher the GPU clock speed, the faster the processing.)
    //   - total global memory
    //   - shared memory per block (A thread block is a programming abstraction that represents a group of threads that can be executed serially or in parallel. On the hardware side, a thread block is composed of 'warps'. )
    //   - num registers per block
    //   - warp size (in threads)  A warp is a set of 32 threads within a thread block such that all the threads in a warp execute the same instruction. Warp size is the number of threads in a warp, which is a sub-division used in the hardware implementation to coalesce memory access and instruction dispatch
    //   - max threads per block
    // =============================================================================
    for (int i = 0; i < devCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("device NO.: %d\n", i);
        printf("device name: %s\n", prop.name);
        printf("compute capability: major: %d, minor: %d\n",prop.major, prop.minor);
        printf("multi-processor count: %d\n",prop.multiProcessorCount);
        printf("clock rate (KHz): %d\n", prop.clockRate);
        printf("total global memory (Gb): %.2f\n", float(prop.totalGlobalMem / std::pow(1024, 3)));
        printf("shared memory per block: %lu\n", prop.sharedMemPerBlock);
        printf("num registers per block: %d\n", prop.regsPerBlock);
        printf("warp size (in threads): %d\n", prop.warpSize);
        printf("max threads per block: %d\n", prop.maxThreadsPerBlock);
    }
    printf("\n=============================================================\n");
    return true;
}

void show_help(){
    fprintf(stdout, "================================ usage ================================\n");
    fprintf(stdout, "H: Help.\n" );
    fprintf(stdout, "Q/A button: brightness +/-\n");
    fprintf(stdout, "W/S button: contrast_func +/-\n");
    fprintf(stdout, "E/D button: saturation +/-\n");
    fprintf(stdout, "R button: box_filtering on/off \n");
    fprintf(stdout, "F/v button: increase/decrease box_filtering kernel size \n");
    fprintf(stdout, "T button: sharpening on/off \n");
    fprintf(stdout, "G/B button: sharpening +/ \n");
    fprintf(stdout, "Y button: Edge Dectection on/off \n");
    fprintf(stdout, "U button: Gaussian smoothing on/off \n");
    fprintf(stdout, "J/M button: increase/decrease Gaussian smoothing kernel size \n");
    fprintf(stdout, "I/K button: Profiling 1 or 2 \n");
    fprintf(stdout, "O button: constant memory \n");
    fprintf(stdout, "L button: shared memory \n");
    fprintf(stdout, "C button: convolution \n");

    fprintf(stdout, "P: Saving Image.\n" );
    fprintf(stdout, "=======================================================================\n");
}

void get_conv(float **conv_kernel, int k) {
    // l=3 means conv kernel size is 3*3
    float *tmp = (float *) malloc(k * k * sizeof(float));
    for (int i = 0; i < k * k; i++) {
        tmp[i] = 1.0 / float(k*k);
    }
    *(conv_kernel) = tmp;
}

void get_gaussian(float **gaussian_kernel, int l, float sigma) {
    // l=3 means conv kernel size is 3*3
    float *tmp = (float *) malloc(l * l * sizeof(float));
    int halflen = l / 2;
    for (int i = -halflen; i < halflen + 1; i++)
        for (int j = -halflen; j < halflen + 1; j++) {
            tmp[(i + halflen) * l + j + halflen] =
                    1 /  (2 * M_PI * sigma *sigma) * exp(-1 * ((i * i) + (j * j)) / (2 * sigma * sigma));
//            printf("gaussian weight: %f \n",  1 /  (2 * M_PI * sigma *sigma) * exp(-1 * ((i * i) + (j * j)) / (2 * sigma * sigma)));
        }
    *(gaussian_kernel) = tmp;
}


// entry point
int main(int argc, char **argv) {

    // query CUDA capabilities
    if (!queryGPUCapabilitiesCUDA()) {
        // quit in case capabilities are insufficient
        exit(EXIT_FAILURE);
    }

    show_help();

    testCudaCall();


    // simple example taken and modified from http://cimg.eu/
    // load image
    cimg_library::CImg<unsigned char> image(IMGPATH);
    int img_h = image.height();
    int img_w = image.width();
    // create image for simple visualization
    cimg_library::CImg<unsigned char> visualization(512, 300, 1, 3, 0);
    const unsigned char red[] = {255, 0, 0};
    const unsigned char green[] = {0, 255, 0};
    const unsigned char blue[] = {0, 0, 255};

    // create displays
    cimg_library::CImgDisplay inputImageDisplay(image, "click to select row of pixels");
    inputImageDisplay.move(40, 40);
    cimg_library::CImgDisplay visualizationDisplay(visualization, "intensity profile of selected row");
    visualizationDisplay.move(600, 40);


    cimg_library::CImg<unsigned char> original_image = image;

    // image processing part paramter
    bool box_filtering, edgedetection, sharpen, gaussian_filtering;
    box_filtering = edgedetection = sharpen = gaussian_filtering = false;
    float brightnessfactor = 1.0f;
    float contrast_factor = 1.0f;
    float saturationfactor = 1.0f;

    int box_filtering_k = 3;
    float *box_filtering_kernel;

    // edge conv kernel
    int edge_k = 3;
    float *edgeconv;
    get_conv(&edgeconv, edge_k);
    edgeconv[0] = edgeconv[2] = edgeconv[6] = edgeconv[8] = 0.0;
    edgeconv[1] = edgeconv[3] = edgeconv[5] = edgeconv[7] = -1.0;
    edgeconv[4] = 4.0;

    // sharpen factor
    float sharp_factor = 0.5;

    // gaussian kernel
    float *gaussian_kernel;
    int gaussian_k = 5;
    float sigma = 1.0;

//	printf("image origin %d %d %d %d %d %d %d %d %d", image[0], image[1], image[2], image[512], image[513], image[514], image[1024], image[1025], image[1026]);
    while (!inputImageDisplay.is_closed() && !visualizationDisplay.is_closed()) {
        inputImageDisplay.wait();
        if (inputImageDisplay.button() && inputImageDisplay.mouse_y() >= 0) {
            // on click redraw visualization
            const int y = inputImageDisplay.mouse_y();
            visualization.fill(0).draw_graph(image.get_crop(0, y, 0, 0, image.width() - 1, y, 0, 0), red, 1, 1, 0, 255,
                                             0);
            visualization.draw_graph(image.get_crop(0, y, 0, 1, image.width() - 1, y, 0, 1), green, 1, 1, 0, 255, 0);
            visualization.draw_graph(image.get_crop(0, y, 0, 2, image.width() - 1, y, 0, 2), blue, 1, 1, 0, 255,
                                     0).display(visualizationDisplay);
        }
        else if (inputImageDisplay.is_keyH()) { // brightness up
            show_help();
        }

        else if (inputImageDisplay.is_keyQ()) { // brightness up
            brightnessfactor -= 0.1;
            printf("brightness %f \n", brightnessfactor);
            brighten_func(image.data(), original_image.data(), brightnessfactor, img_h, img_w);

        }
        else if (inputImageDisplay.is_keyA()) { // brightness down
            brightnessfactor += 0.1;
            printf("brightness %f \n", brightnessfactor);
            brighten_func(image.data(), original_image.data(), brightnessfactor, img_h, img_w);
        }

        // contrast_func
        else if (inputImageDisplay.is_keyW()) { // contrast_func up
            contrast_factor += 0.1;
            printf("contrast_factor %f \n", contrast_factor);
            contrast_func(image.data(), original_image.data(), contrast_factor, img_h, img_w);

        } else if (inputImageDisplay.is_keyS()) { // contrast_func down
            contrast_factor -= 0.1;
            printf("contrast_factor %f \n", contrast_factor);
            contrast_func(image.data(), original_image.data(), contrast_factor, img_h, img_w);
        }

        // saturation
        else if (inputImageDisplay.is_keyE()) { // saturation up
            saturationfactor += 0.1;
            printf("saturationfactor %f \n", saturationfactor);
            saturation_func(image.data(), original_image.data(), saturationfactor, img_h, img_w);
        } else if (inputImageDisplay.is_keyD()) { // saturation down
            saturationfactor -= 0.1;
            printf("saturationfactor %f \n", saturationfactor);
            saturation_func(image.data(), original_image.data(), saturationfactor, img_h, img_w);
        }

        else if (inputImageDisplay.is_keyR()) { // box_filtering on/off
            if (!box_filtering) {
                printf("turn off box_filtering\n");
                image = original_image;
            } else {
                printf("turn on box_filtering\n");
                printf("box_filtering kernel is %d \n", box_filtering_k);
                get_conv(&box_filtering_kernel, box_filtering_k);
                conv2d_func(image.data(), original_image.data(), box_filtering_kernel, box_filtering_k, img_h, img_w);
            }
            box_filtering = !box_filtering;
        }
        else if (inputImageDisplay.is_keyF()) { // box_filtering increase
            box_filtering_k += 2;
            printf("box_filtering kernel is increased to %d \n", box_filtering_k);
            get_conv(&box_filtering_kernel, box_filtering_k);
            conv2d_func(image.data(), original_image.data(), box_filtering_kernel, box_filtering_k, img_h, img_w);
        }
        else if (inputImageDisplay.is_keyV()) { // box_filtering decrease
            box_filtering_k -= 2;
            box_filtering_k = box_filtering_k < 3 ? 3 : box_filtering_k;
            printf("box_filtering kernel is decreased to %d \n", box_filtering_k);
            get_conv(&box_filtering_kernel, box_filtering_k);
            conv2d_func(image.data(), original_image.data(), box_filtering_kernel, box_filtering_k, img_h, img_w);
        }

        else if (inputImageDisplay.is_keyY()) { // edge detection on/off
            if (!edgedetection) {
                printf("turn off edge detection\n");
                image = original_image;
            } else {
                printf("turn on edge detection\n");
                conv2d_func(image.data(), original_image.data(), edgeconv, edge_k, img_h, img_w);
            }
            edgedetection = !edgedetection;
        }
        else if (inputImageDisplay.is_keyT()) { // sharpening on/off
            if (!sharpen) {
                printf("turn off sharpen\n");
                image = original_image;
            } else {
                printf("turn on sharpen\n");
                sharpen_func(image.data(), original_image.data(),sharp_factor, edgeconv, edge_k, img_h, img_w);
            }
            sharpen = !sharpen;

        }
        else if (inputImageDisplay.is_keyG()) { // sharpening up
            sharp_factor += 0.1;
            printf("sharp_factor %f \n", sharp_factor);
            sharpen_func(image.data(), original_image.data(), sharp_factor, edgeconv, edge_k, img_h, img_w);
        } else if (inputImageDisplay.is_keyB()) { // sharpening down
            sharp_factor -= 0.1;
            printf("sharp_factor %f \n", sharp_factor);
            sharpen_func(image.data(), original_image.data(), sharp_factor, edgeconv, edge_k, img_h, img_w);
        }

        else if (inputImageDisplay.is_keyU()) { // Gaussian on/off
            if (!gaussian_filtering) {
                printf("turn off Gaussian Smoothing\n");
                image = original_image;
            } else {
                printf("turn on Gaussian Smoothing\n");
                printf("Gaussian smooth kernel is %d \n", gaussian_k);
                get_gaussian(&gaussian_kernel, gaussian_k, sigma);
                conv2d_func(image.data(), original_image.data(), gaussian_kernel, gaussian_k, img_h, img_w);
            }
            gaussian_filtering = !gaussian_filtering;
        }
        else if (inputImageDisplay.is_keyJ()) { // Gaussian increase
            gaussian_k += 2;
            sigma = (gaussian_k-1)/4;
            printf("Gaussian smooth kernel is increased to %d \n", gaussian_k);
            get_gaussian(&gaussian_kernel, gaussian_k, sigma);
            conv2d_func(image.data(), original_image.data(), gaussian_kernel, gaussian_k, img_h, img_w);
        }
        else if (inputImageDisplay.is_keyM()) { // Gaussian decrease
            gaussian_k -= 2;
            gaussian_k = gaussian_k < 5 ? 5 : gaussian_k;
            sigma = (gaussian_k-1)/4;
            printf("Gaussian smooth kernel is decreased to %d \n", gaussian_k);
            get_gaussian(&gaussian_kernel, gaussian_k, sigma);
            conv2d_func(image.data(), original_image.data(), gaussian_kernel, gaussian_k, img_h, img_w);
        }
        else if (inputImageDisplay.is_keyI()) { // Gaussian decrease
            // profiling 1:
            printf("============== Profiling 1 ==============\n");
            printf("Profiling the Gaussian smoothing filter with increasing kernel size "
                   "on a fixed image (e.g., 1024x1024) to analyze the scaling behavior. \n");
            gaussian_k = 5;
            for (int i = 1; i <= 5; i++) {
                gaussian_k += 2;
                sigma = (gaussian_k-1)/4;
                printf("profiling with gaussian kernel %d \n", gaussian_k);
                get_gaussian(&gaussian_kernel, gaussian_k, sigma);
                conv2d_func(image.data(), original_image.data(), gaussian_kernel, gaussian_k, img_h, img_w);
            }
            printf("============== End of Profiling 1 ==============\n");
        }
        else if (inputImageDisplay.is_keyK()) { // Gaussian decrease
            // profiling 2:
            printf("============== Profiling 2 ==============\n");
            printf("Profiling with fixed size smoothing filter (e.g., 25x25) on randomly initialized images "
                   "of increasing size (128x128, ..., 1024x1024, ...) to analyze the scaling behavior. \n");
            sigma = 6;
            gaussian_k = 25;
            get_gaussian(&gaussian_kernel, gaussian_k, sigma);
            for (int i = 6; i <= 9; i++) {

                int imagesize = 2 << i, imgproduct = imagesize * imagesize;
                printf("profiling an random initialized image with image size %d \n", imagesize);
                cimg_library::CImg<unsigned char> square(imagesize, imagesize, 1, 3, 0);
                conv2d_func(square.data(), square.data(), gaussian_kernel, gaussian_k, imagesize, imagesize);
            }
            printf("============== End of Profiling 2 ==============\n");
        }
        else if (inputImageDisplay.is_keyO()) { // constant memory
            if (!gaussian_filtering) {
                printf("turn off Gaussian Smoothing using constant memory\n");
                image = original_image;
            } else {
                printf("turn on Gaussian Smoothing using constant memory\n");
                get_gaussian(&gaussian_kernel, 9, 2);
                conv2d_const_mem_func(image.data(), original_image.data(), gaussian_kernel, img_h, img_w);
            }
            gaussian_filtering = !gaussian_filtering;
        }
        else if (inputImageDisplay.is_keyL()) { // shared memory on
            if (!gaussian_filtering) {
                printf("turn off Gaussian Smoothing using shared memory\n");
                image = original_image;
            } else {
                printf("turn on Gaussian Smoothing using shared memory\n");
                get_gaussian(&gaussian_kernel, 5, 1);
                conv2d_shared_mem_func(image.data(), original_image.data(), gaussian_kernel, 16, 5, img_h, img_w);
            }
            gaussian_filtering = !gaussian_filtering;
        }
        else if (inputImageDisplay.is_keyC()) { // convolution
            printf("VGG convolution, saving results to ./");
            int length = 64 * 3 * 3 * 3;
            std::string filename = "../../data/vgg16_layer_1.raw";
            FILE *fhandle = fopen(filename.c_str(), "rb");
            float *vec = (float *) malloc(length * sizeof(float));
            fread((void *) vec, sizeof(float), length, fhandle);
            std::string save_name;
            for (int i = 0; i < 64; i++) {
                conv2d_func(image.data(), original_image.data(), vec + 3 * 3 * 3 * i, 3, img_h, img_w);
                save_name = std::to_string(i) + ".png";
                image.save(save_name.c_str());
            }
        }

        else if (inputImageDisplay.is_keyESC()) { // quit
            break;
        }
        else if (inputImageDisplay.is_keyP()) { // quit
            printf("saving image to ./");
            visualization.save("./test_output.png");
            image.save("./test_image.png");
        }

        inputImageDisplay.display(image);
    }

    // save test output image
    visualization.save("./test_output.png");
    image.save("./test_image.png");

    return EXIT_SUCCESS;
}


