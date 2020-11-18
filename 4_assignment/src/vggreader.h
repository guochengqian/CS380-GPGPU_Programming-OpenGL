#ifndef VGGREADER_H_
#define VGGREADER_H_

#include <array>
#include <stdio.h>
#include <iomanip>
#include <string>


class VggReader {
public:
	
	//loads a raw float file that contains a vgg16 layer
	static void readfloat32(std::string path, float* v, size_t sizeOfFloatArray, bool flipEndianess);
	static float flipEndianessFloat(const float inFloat);
	static void printKernels(float* values, int size);
	
};

#endif
