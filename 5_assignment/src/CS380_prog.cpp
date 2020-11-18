// CS 380 - GPGPU Programming
// Programming Assignment #5


// system includes
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <sstream>
#include <algorithm> 
#include <array>
#include <assert.h> 

#include "timer.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "CImg.h"
#include "matrixkernels.cuh"


#include <fstream>


enum MAJORITY {
	ROW_MAJOR = 0,
	COLUMN_MAJOR = 1
};


// query GPU functionality we need for CUDA, return false when not available
bool queryGPUCapabilitiesCUDA()
{
	// Device Count
	int devCount;

	// Get the Device Count
	cudaGetDeviceCount(&devCount);

	// Print Device Count
	printf("Device(s): %i\n", devCount);

	// query anything else you will need
	return devCount > 0;
}



// compute and print 
// sum, average and maximum 
// of error: (abs(A*x-b))
void computeResultError(float *h_A, float *h_b, float *h_x, unsigned int dim)
{
	float* h_e0 = new float[dim];
	multiplyMatrixVector(h_A, h_x, h_e0, dim);

	float sum_abs_diff_0 = 0;
	float max_abs_diff_0 = 0;

	for (unsigned int i = 0; i < dim; i++) {

		float abs_diff_0 = h_e0[i] - h_b[i];
		abs_diff_0 = abs_diff_0 > 0.0f ? abs_diff_0 : -abs_diff_0;

		if (abs_diff_0 > max_abs_diff_0)
		{
			max_abs_diff_0 = abs_diff_0;
		}
		sum_abs_diff_0 += abs_diff_0;
	}

	
	std::cout << "errors:\n";
	

	std::cout << "sum  | " << sum_abs_diff_0 << "\navgs | " << sum_abs_diff_0 / dim << "\nmax  | " << max_abs_diff_0 << std::endl << std::endl;

	delete[] h_e0;
}

// compute blurred image and return in Image format;
void computeBlurredImage(cimg_library::CImg<unsigned char> &im, cimg_library::CImg<unsigned char> &imBlurred, int kernelDim, float* kernel) {
	int channels = im.spectrum();
	if (channels > 3)
	{
		channels = 3;
	}

	int kernelDimHalf = kernelDim / 2;
	unsigned int indexOffset = 0;

	// go through each pixel of the output blurred image to compute pixel value
	for (int y = 0; y < im.height(); ++y) {
		for (int x = 0; x < im.width(); ++x) {

			float blurredPixel[4] = { 0.0f, 0.0f, 0.0f, 0.0f };		// start with empty color (intensity value)
			float sumKernel = 0.0f;			// sum of the used kernel weights (for normalization)

			// compute blurred pixel by going through the blur kernel
			for (int offsetX = -kernelDimHalf; offsetX <= kernelDimHalf; ++offsetX) {

				int curPixelX = offsetX + x;
				if (curPixelX < 0 || curPixelX >= im.width()) continue;
				for (int offsetY = -kernelDimHalf; offsetY <= kernelDimHalf; ++offsetY) {

					int kernelIndex = (offsetX + kernelDimHalf) * kernelDim + (offsetY + kernelDimHalf);

					int curPixelY = offsetY + y;
					if (curPixelY < 0 || curPixelY >= im.height()) continue;

					// add the contribution of pixel( curPixelX, curPixelY ) to the output
					for (int c = 0; c < channels; c++) {
						blurredPixel[c] += kernel[kernelIndex] * im(curPixelX, curPixelY, 0, c);
					}

					// add the kernel weight to the sum
					sumKernel += kernel[kernelIndex];
				}
			}

			// store the blurred pixel value to the output Image and vector
			for (int c = 0; c < channels; c++) {
				imBlurred(x, y, 0, c) = blurredPixel[c] / sumKernel;
			}
		}
	}

}

// compute the blur matrix resulting from the given blur kernel and store it in filterKernelMatrix
// compute the sum of the kernel and obtain the normalized kernel
void initFilterKernelMatrix(cimg_library::CImg<unsigned char> &im, int kernelDim, float* kernel, float*& filterKernelMatrix)
{

	// set up cpu memory for blur matrix and blurred image
	unsigned int numPixels = im.width() * im.height();
	filterKernelMatrix = new float[numPixels * numPixels];
	memset(filterKernelMatrix, 0, numPixels * numPixels * sizeof(float));

	int kernelDimHalf = kernelDim / 2;

	unsigned int indexOffset = 0;

	// go through each pixel of the blur matrix
	for (int y = 0; y < im.height(); ++y) {
		for (int x = 0; x < im.width(); ++x) {
			float sumKernel = 0.0f;			// sum of the used kernel weights (for normalization)

			// compute the sum of the kernel weights first
			for (int offsetX = -kernelDimHalf; offsetX <= kernelDimHalf; ++offsetX) {
				int curPixelX = offsetX + x;
				if (curPixelX < 0 || curPixelX >= im.width()) continue;
				for (int offsetY = -kernelDimHalf; offsetY <= kernelDimHalf; ++offsetY) {
					int curPixelY = offsetY + y;
					if (curPixelY < 0 || curPixelY >= im.height()) continue;
					int kernelIndex = (offsetX + kernelDimHalf) * kernelDim + (offsetY + kernelDimHalf);
					sumKernel += kernel[kernelIndex];
				}
			}

			// write blur matrix (row-major order) using the normalized weights;
			// note that we compute the blur matrix in a separate loop because we need the sumKernel value
			for (int offsetX = -kernelDimHalf; offsetX <= kernelDimHalf; ++offsetX) {

				int curPixelX = offsetX + x;
				if (curPixelX < 0 || curPixelX >= im.width()) continue;
				for (int offsetY = -kernelDimHalf; offsetY <= kernelDimHalf; ++offsetY) {

					int kernelIndex = (offsetX + kernelDimHalf) * kernelDim + (offsetY + kernelDimHalf);

					int curPixelY = offsetY + y;
					if (curPixelY < 0 || curPixelY >= im.height()) continue;

					// write the normalized kernel value in the blur matrix
					unsigned int curIndex = indexOffset + (im.width() * curPixelY + curPixelX);
					filterKernelMatrix[curIndex] = kernel[kernelIndex] / sumKernel;
				}
			}

			indexOffset += numPixels;

		}
	}


}

// linearize channel c of image im into vector vec
void initImageVector(cimg_library::CImg<unsigned char> &im, int c, float*& vec) {

	unsigned int numPixels = im.width() * im.height();

	vec = new float[numPixels];
	memset(vec, 0, numPixels * sizeof(float));

	// go through each pixel of the image
	for (int y = 0; y < im.height(); ++y) {
		for (int x = 0; x < im.width(); ++x) {
			vec[y * im.width() + x] = im(x, y, 0, c);
		}
	}
}

// loading matrix files 
bool readMatrix(char* filename, float* &matrix, unsigned int *dim = NULL, int majority = ROW_MAJOR)
{
	unsigned int w, h, x, y, num_entries;

	float val;

	std::ifstream file(filename);

	if (file)
	{
		file >> h >> w >> num_entries;
		std::cout << w << " " << h << " " << num_entries << "\n";

		assert(w == h || w == 1 || h == 1);

		if (dim != NULL) {
			*dim = w;
			if (h > w) { *dim = h; }
		}

		matrix = new float[w * h];
		memset(matrix, 0, num_entries * sizeof(float));

		unsigned int i;
		for (i = 0; i < num_entries; i++) {

			if (file.eof()) break;

			file >> y >> x >> val;

			if (majority == ROW_MAJOR) {

				matrix[w * y + x] = val;

			}
			else if (majority == COLUMN_MAJOR) {

				matrix[h * x + y] = val;
			}
		}
		file.close();

		if (i == num_entries)
			std::cout << "\nFile read successfully\n";
		else
			std::cout << "\nFile read successfully but seems defective:\n num entries read = " << i << ", entries epected = " << num_entries << "\n";
	}
	else {
		std::cout << "Unable to open file\n";
		return false;
	}

	return true;
}





// entry point
int main(int argc, char** argv)
{
    // matrix is assigned to an array with length=dim.
	unsigned int dim;
	float *h_A, *h_b, *h_x;
	h_A = h_b = h_x = NULL;
	float *h_bR, *h_bG, *h_bB;
	h_bR = h_bG = h_bB = NULL;



	/**************************************************************
	get the conjugate gradient method first running with the
	test matrices. afterwards change the parameter matrixSet to 0
	and try with different input images!
	***************************************************************/
	
	// PARAMETERS: 
	int matrixSet;
	matrixSet = 0;			// set this to 0 for the image deblurring;
//	matrixSet = 16;		// set this to 16, 64, 200 for the other matrices
	//matrixSet = 64;										
	//matrixSet = 200;								

	// unblurred input image
	std::string inputImageFilename;			// set this to a valid png filename
//	inputImageFilename = "bigben_small";
	//inputImageFilename = "bigben_med";
	//inputImageFilename = "peppers_small";
	//inputImageFilename = "peppers_med";
	//inputImageFilename = "lichtenstein_small";	
	//inputImageFilename = "lichtenstein_med";	
	//inputImageFilename = "house_small";
	inputImageFilename = "baboon_small";




	// define the blurring filter 
	int kernelDim = 5;
	float* kernel = new float[kernelDim * kernelDim];		// define symmetric positive blur kernel
	kernel[0] = 0.0f; kernel[5] = 1.0f; kernel[10] = 2.0f; kernel[15] = 1.0f; kernel[20] = 0.0f;
	kernel[1] = 1.0f; kernel[6] = 2.0f; kernel[11] = 3.0f; kernel[16] = 2.0f; kernel[21] = 1.0f;
	kernel[2] = 2.0f; kernel[7] = 3.0f; kernel[12] = 4.0f; kernel[17] = 3.0f; kernel[22] = 2.0f;
	kernel[3] = 1.0f; kernel[8] = 2.0f; kernel[13] = 3.0f; kernel[18] = 2.0f; kernel[23] = 1.0f;
	kernel[4] = 0.0f; kernel[9] = 1.0f; kernel[14] = 2.0f; kernel[19] = 1.0f; kernel[24] = 0.0f;



	unsigned int numPixels;
	bool success = true;

	cimg_library::CImg<unsigned char> im;
	cimg_library::CImg<unsigned char> imBlurred;

	// query CUDA capabilities
	if (!queryGPUCapabilitiesCUDA())
	{
		// quit in case capabilities are insufficient
		exit(EXIT_FAILURE);
	}



	// set up matrices
	switch (matrixSet) {
	case(0):
		if (inputImageFilename.length() == 0)
		{
			std::cout << "please set a input image filename" << std::endl;
			return -1;
		}
		// load input image (not blurred)
		im.load(std::string("../../data/images/" + inputImageFilename + ".png").c_str());

		imBlurred = im;

		numPixels = im.width() * im.height();
		dim = im.width() * im.height();

		// compute blurred image 

		computeBlurredImage(im, imBlurred, kernelDim, kernel);

		imBlurred.save(std::string("../../data/images/" + inputImageFilename + "_blurred.png").c_str());

		// init kernel matrix
		initFilterKernelMatrix(im, kernelDim, kernel, h_A);

		// init image vectors. turn the image from matrix to vector.
		// red channel
		initImageVector(imBlurred, 0, h_bR);
		if (im.spectrum() > 1) {
			// green channel
			initImageVector(imBlurred, 1, h_bG);
		}
		if (im.spectrum() > 2) {
			// blue channel
			initImageVector(imBlurred, 2, h_bB);
		}
		break;

	case(16):
		// load input matrix A and vector b
		success = readMatrix((char *)"../../data/matrices/A16x16.txt", h_A, &dim, ROW_MAJOR);
		success = success && readMatrix((char *)"../../data/matrices/b16x1.txt", h_b);
		break;
	case(64):
		// load input matrix A and vector b
		success = readMatrix((char *)"../../data/matrices/A64x64.txt", h_A, &dim, ROW_MAJOR);
		success = success && readMatrix((char *)"../../data/matrices/b64x1.txt", h_b);
		break;
	case(200):
		// load input matrix A and vector b
		success = readMatrix((char *)"../../data/matrices/A200x200.txt", h_A, &dim, ROW_MAJOR);
		success = success && readMatrix((char *)"../../data/matrices/b200x1.txt", h_b);
		break;

	default:
		// init default input matrix A and vector b
		h_A = new float[16]; h_b = new float[4]; dim = 4;
		h_A[0] = 6.25f; h_A[4] = 3.5f; h_A[8] = 4.0f; h_A[12] = 5.5f;
		h_A[1] = 3.5f; h_A[5] = 5.25f; h_A[9] = 0.5f; h_A[13] = 4.5f;
		h_A[2] = 4.0f; h_A[6] = 0.5f; h_A[10] = 10.0f; h_A[14] = 2.0f;
		h_A[3] = 5.5f; h_A[7] = 4.5f; h_A[11] = 2.0f; h_A[15] = 7.25f;
		h_b[0] = 7.0f; h_b[1] = 5.5f; h_b[2] = 11.0f; h_b[3] = 6.75f;
		break;
	}

	if (!success) {
		std::cout << "File input error";
		return 0;
	}

	// init CUDA
	cudaSetDevice(0);

	if (matrixSet != 0) {
		// matrix application
		// init the solution to a vector of 0's
		h_x = new float[dim];
		memset(h_x, 0, dim * sizeof(float));

		StartTimer();
		// find h_x where h_A * h_x = h_b
		float errorTolerance = 0.0000001f * dim;

		computeConjugateGradient(h_A, h_b, h_x, dim, errorTolerance);

		double t = GetTimer();
		std::cout << "elapsed time: " << t << "ms" << std::endl;
		// compute the error
		computeResultError(h_A, h_b, h_x, dim);

	}
	else
	{
		// image deblurring application
		// init the solution to a vector of 0's
		h_x = new float[dim];


		cimg_library::CImg<unsigned char> deblurredImage = imBlurred;
	
		// we are fine with an error of 1 per pixel on average
		float errorTolerance = 1.f * dim;
	
		// for each channel (RGB) do the deblurring
		for (int c = 0; c < im.spectrum(); c++)
		{
			memset(h_x, 0, dim * sizeof(float));
			if (c == 0) {
				std::cout << "computing red channel" << std::endl;
				// find h_x where h_A * h_x = h_bR

				// h_x is the deblurred image, h_bR is the bulrred R channel (Known)

				computeConjugateGradient(h_A, h_bR, h_x, dim, errorTolerance);
				// compute the error
				computeResultError(h_A, h_bR, h_x, dim);
			}
			else if (c == 1) {
				std::cout << "computing green channel" << std::endl;
				// find h_x where h_A * h_x = h_bG
				computeConjugateGradient(h_A, h_bG, h_x, dim, errorTolerance);
				// compute the error
				computeResultError(h_A, h_bG, h_x, dim);
			}
			else if (c == 2) {
				std::cout << "computing blue channel" << std::endl;
				computeConjugateGradient(h_A, h_bB, h_x, dim, errorTolerance);
				// compute the error
				computeResultError(h_A, h_bB, h_x, dim);
			}

//            printf("size of h_x is: %.2f \n", float(sizeof(h_x))/ sizeof(h_x[0]));
			// for the image deblurring application, write the deblurred image (h_x) to disk
			unsigned int pixelIndex = 0;
			for (int y = 0; y < im.height(); ++y) {
				for (int x = 0; x < im.width(); ++x) {
					if (c == 3) {
						deblurredImage(x, y, 0, 3) = 1.f;
					}
					else {
						deblurredImage(x, y, 0, c) = h_x[pixelIndex];
					}
					pixelIndex++;
				}
			}
		}

		deblurredImage.save(std::string("../../data/images/" + inputImageFilename + "_deblurred.png").c_str());

		// create displays 
		cimg_library::CImgDisplay inputImageDisplay(im, "input");
		inputImageDisplay.move(40, 40);
		cimg_library::CImgDisplay blurredImageDisplay(imBlurred, "blurred");
		blurredImageDisplay.move(40+im.width(), 40);
		cimg_library::CImgDisplay deblurredImageDisplay(deblurredImage, "deblurred");
		deblurredImageDisplay.move(40 + im.width()*2, 40);



		while ( (!deblurredImageDisplay.is_closed()) && (!blurredImageDisplay.is_closed()) && (!inputImageDisplay.is_closed())){
		}

		}

	delete[] h_A;
	delete[] h_x;

	if (h_b != NULL) {
		delete[] h_b;
	}
	if (h_bR != NULL) {
		delete[] h_bR;
	}
	if (h_bG != NULL) {
		delete[] h_bG;
	}
	if (h_bB != NULL) {
		delete[] h_bB;
	}

	
	return EXIT_SUCCESS;
}

