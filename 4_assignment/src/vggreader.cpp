#define _CRT_SECURE_NO_WARNINGS 1

#include "vggreader.h"
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>


void VggReader::readfloat32(std::string path, float* v, size_t sizeOfFloatArray, bool flipEndianess) {
	FILE * pFile;
	pFile = fopen(path.c_str(), "rb");
	if (pFile) {

		fread((void*)v, sizeof(float), sizeOfFloatArray, pFile);
		if (flipEndianess) {
			for (int i = 0; i < sizeOfFloatArray; i++) {
				float f = v[i];
				v[i] = flipEndianessFloat(f);
			}
		}
	}
}

float VggReader::flipEndianessFloat(const float inFloat)
{
	float retVal;
	char *floatToConvert = (char*)& inFloat;
	char *returnFloat = (char*)& retVal;

	// swap the bytes into a temporary buffer
	returnFloat[0] = floatToConvert[3];
	returnFloat[1] = floatToConvert[2];
	returnFloat[2] = floatToConvert[1];
	returnFloat[3] = floatToConvert[0];

	return retVal;
}

void VggReader::printKernels(float* values, int size) {
	int count = 0;
	for (int i = 0; i < size; i++) {
		float f = values[i];
		if ((count % 27) == 0) std::cout << "[\n";
		if ((count % 9) == 0) std::cout << "[";
		if ((count % 3) == 0) std::cout << "[";
		std::cout << std::fixed << std::setprecision(4) << f;
		count++;
		if ((count % 3) != 0) std::cout << ", ";
		if ((count % 3) == 0) std::cout << "]";
		if ((count % 9) == 0) std::cout << "]\n";
		if ((count % 27) == 0) std::cout << "]\n";
	}
	std::cout << std::endl; 
}