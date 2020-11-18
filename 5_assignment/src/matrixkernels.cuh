#pragma once


enum VEC_OP{

	NONE = -1,
	CL_ADD = 0,
	CL_SUB = 1,
	CL_MULT = 2,
	CL_MAX = 3,
	CL_MIN = 4,
	CL_ABS = 5
};


// TODO remove!!!
__device__ float operation(int op, float x1, float x2); 
// vector operation: x = fac0*a op fac1*b
__global__ void _cl_vector_op_( int op, float fac0, float fac1, float *a, float *b, float *x, int dim );

// matrix vector multiplication: x = A*b op c
__global__ void _cl_matrix_vector_( int op, float *A, float *b, float *c, float *x, int dim );
	
// d_x = SUM[d_a * d_b]
float gpuReduceSUM( float* d_a, float *d_b, float* d_x, int dim, int nBlocks, int nThreads );


// x = A*a
extern "C"
void multiplyMatrixVector( float *h_A, float *h_a, float *h_x, int dim );

extern "C"
void computeConjugateGradient( float *h_A, float *h_b, float *h_x, int dim, float errorTolerance);
