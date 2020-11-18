#include <iostream>
#include "matrixkernels.cuh"
#include "helper_cuda.h"

extern "C"
int iDivUp( int a, int b ){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


extern "C"
unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


// vector operation: x = fac0*a op fac1*b
__global__ void
_cl_vector_op_( int op, float fac0, float fac1, float *a, float *b, float *x, int dim )
{
	/* TASK 1.1: implement the elementwise vector operations
	
	 	x = fac0 * a (op) fac1 * b
	
		with op = {+,-,*, NONE}.
		NONE means x = fac0 * a   (b might be NULL)

		
		HINT: remember to safeguard the index (the thread id might be larger than the array size)! 
		-> if the thread index is >= dim return!
		
	*/
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // remember to safeguard the index
    if (idx >= dim) return;

    // op = {+,-,*, NONE}.
    switch(op){
        case(-1): // NONE
            x[idx] = a[idx] * fac0;
            break;
        case(0):  // ADD
            x[idx] = a[idx] * fac0 + b[idx] * fac1;
            break;
        case(1):  // SUB
            x[idx] = a[idx] * fac0 - b[idx] * fac1;
            break;
        case(2):  // MULT
            x[idx] = a[idx] * fac0 * b[idx] * fac1;
            break;
    }

}




// matrix vector multiplication: x = A*b op c
__global__ void
_cl_matrix_vector_( int op, float *A, float *b, float *c, float *x, int dim )
{
	/* TASK 1.2: implement the matrix vector multiplication
	
		x = A * b (op) c
	
		with op = {+,-,*,NONE}.
		NONE means x = A * b     (c might be NULL)

        x is the output vector

		HINT: remember to safeguard the index (the thread id might be larger than the array size)!
		-> if the thread index is >= dim return!
	*/

    int idx = blockIdx.x * blockDim.x + threadIdx.x;    // sync computing rows of A.

    // load data sync
    float out = 0.0f;

    // dynamic shared memory
    // may have problem in memory size.
    extern __shared__ float b_s[]; // vector
    for(int m=0; m<dim/blockDim.x+1; m++){
        int tmpidx = m*blockDim.x+threadIdx.x;
        if( tmpidx < dim) b_s[tmpidx] = b[tmpidx];

    }
    __syncthreads();

    if (idx >= dim) return; // safeguard the index
    for(int i=0; i<dim; i++){
        out += A[idx * dim + i] * b_s[i];
    }

    // x is the output vector
    // x[idx] equals to the dot product of a row in A with b_s
    switch(op){
        case(-1):
            x[idx] = out;
            break;
        case(0):
            x[idx] = out + c[idx];
            break;
        case(1):
            x[idx] = out - c[idx];
            break;
        case(2):
            x[idx] = out * c[idx];
            break;
    }
}


// reduceSum: CUDA https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__global__
void _cl_vector_reduce_(int op, float* d_a, int dim){

    int idx = blockDim.x * blockIdx.x + threadIdx.x; // parallel along the vector

    for(unsigned int s=1; s<dim; s*=2){  // step. log2(dim)
        if (idx % (2*s) == 0 && idx + s < dim){
            switch(op){
                case(0):
                    d_a[idx] += d_a[idx + s];
                    break;
                case(2):
                    d_a[idx] *= d_a[idx + s];
            }
        }
    }
}



// d_x = SUM[d_a * d_b]
float gpuReduceSUM( float* d_a, float *d_b, float* d_x, int dim, int nBlocks, int nThreads ){

	/* TASK 1.3: implement the vector multiplication and sum reduction

		d_x = SUM[d_a * d_b]
		
		implement reduction as discussed in the lecture using shared memory.
		
	*/
	
	float sum = 0;

    _cl_vector_op_<<<nBlocks, nThreads>>>(CL_MULT, 1.0f, 1.0f, d_a, d_b, d_x, dim);
    checkCudaErrors( cudaMemcpy( &sum, d_x, 1 * sizeof( float ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaDeviceSynchronize() );

    _cl_vector_reduce_<<<nBlocks, nThreads>>>(CL_ADD, d_x, dim);
    checkCudaErrors( cudaDeviceSynchronize() );

    checkCudaErrors( cudaMemcpy( &sum, d_x, 1 * sizeof( float ), cudaMemcpyDeviceToHost ) );
//    printf("sum is %f \n", sum);
	return sum;
}

// x = A*a
extern "C" 
void multiplyMatrixVector( float *h_A, float *h_a, float *h_x, int dim )
{
	float *d_A, *d_a, *d_x;

    // square matrix. dim * dim.
	checkCudaErrors( cudaMalloc( (void**) &d_A, dim * dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_a, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_x, dim * sizeof( float ) ) );

	checkCudaErrors( cudaMemcpy( d_A, h_A, dim * dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_a, h_a, dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_x, h_x, dim * sizeof( float ), cudaMemcpyHostToDevice ) );

	checkCudaErrors( cudaDeviceSynchronize() );

	// x = A*a
	int nThreads = 128;
	int nBlocks = iDivUp( dim, nThreads );
	_cl_matrix_vector_<<< nBlocks, nThreads, dim*sizeof(float) >>>( NONE, d_A, d_a, NULL, d_x, dim );
	checkCudaErrors( cudaDeviceSynchronize() );

	// copy solution from device to host
	checkCudaErrors( cudaMemcpy( h_x, d_x, dim * sizeof( float ), cudaMemcpyDeviceToHost ) );

	// release device memory
	checkCudaErrors( cudaFree( d_A ) );
	checkCudaErrors( cudaFree( d_a ) );
	checkCudaErrors( cudaFree( d_x ) );
	
	
}


/*
 * example code of conjgrad in MATLAB styple
 function x = conjgrad(A, b, x)
    r = b - A * x;
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
              break
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end
 */
extern "C" 
void computeConjugateGradient( float *h_A, float *h_b, float *h_x, int dim, float errorTolerance )
{
    // dim is the weight*hight of image

	int nThreads = 128;							// set the number of threads per block to use by default
	int nBlocks = iDivUp( dim, nThreads );
	
	float *d_A, *d_b, *d_x, *d_r, *d_p, *d_q, *d_tmp;
	float alpha, beta, rho = 0;

	//allocate device memory
	checkCudaErrors( cudaMalloc( (void**) &d_A, dim * dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_b, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_x, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_r, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_p, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_q, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_tmp, dim * sizeof( float ) ) );
	
	// copy host to device
	checkCudaErrors( cudaMemcpy( d_A, h_A, dim * dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_b, h_b, dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_x, h_x, dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaDeviceSynchronize() );

	// init CG
	// ALGORITHM: r_0 = b-Ax_0
	// r_0 = A*x_0 - b
	// r_0 is the initial residual
	_cl_matrix_vector_<<< nBlocks, nThreads, dim*sizeof(float) >>>( CL_SUB, d_A, d_x, d_b, d_r, dim );
	checkCudaErrors( cudaDeviceSynchronize() );
	

	// r_0 = -r_0
	_cl_vector_op_<<< nBlocks, nThreads >>>( NONE, -1.0f, 0.0f, d_r, NULL, d_r, dim );
	checkCudaErrors( cudaDeviceSynchronize() );
	
	// p_0 = r_0
	_cl_vector_op_<<< nBlocks, nThreads >>>( NONE,  1.0f, 0.0f, d_r, NULL, d_p, dim );
	checkCudaErrors( cudaDeviceSynchronize() );

	// CG needs max dim iterations
	int i = 0;
	float minRho = 1000000000;
	for( i = 0; i < dim; i++ ){
		
		// rho_k = sum(r_k * r_k)
		rho = gpuReduceSUM(d_r, d_r, d_tmp, dim, nBlocks, nThreads );
		checkCudaErrors( cudaDeviceSynchronize() );
		
		if (minRho > rho) {
			minRho = rho;
		}
		
		//printf("iteration #%d, with rho = %f", i, rho);
		std::cout << "iteration #" << i << ", with rho = " << rho << "          " << '\r' << std::flush;
		// check here for criterion
		if( rho < errorTolerance) {
			break;
		}
		
		// q_k = A*p_k
		_cl_matrix_vector_<<< nBlocks, nThreads, dim*sizeof(float) >>>( NONE, d_A, d_p, NULL, d_q, dim );
		checkCudaErrors( cudaDeviceSynchronize() );
		
		// alpha_k = rho_k / sum(p_k * q_k)
		alpha = rho / gpuReduceSUM(d_p, d_q, d_tmp, dim, nBlocks, nThreads );
		checkCudaErrors( cudaDeviceSynchronize() );
		
		 // x_(k+1) = x_k + alpha_k * p_k
		_cl_vector_op_<<< nBlocks, nThreads >>>( CL_ADD, 1.0f, alpha, d_x, d_p, d_x, dim );
		checkCudaErrors( cudaDeviceSynchronize() );
		
		// r_(k+1) = r_k + (-alpha_k * q_k)
		_cl_vector_op_<<< nBlocks, nThreads >>>( CL_ADD, 1.0f, -alpha, d_r, d_q, d_r, dim );
		checkCudaErrors( cudaDeviceSynchronize() );

		// beta_k = sum(r_(k+1) * r_(k+1)) / rho_k
		beta = gpuReduceSUM(d_r, d_r, d_tmp, dim, nBlocks, nThreads ) / rho;
		checkCudaErrors( cudaDeviceSynchronize() );
		
		// p_(k+1) = r_(k+1) + beta_k * p_k
		_cl_vector_op_<<< nBlocks, nThreads >>>( CL_ADD, 1.0f, beta, d_r, d_p, d_p, dim );
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	rho = gpuReduceSUM(d_r, d_r, d_tmp, dim, nBlocks, nThreads );

	printf("\nSolution found at iteration #%d, with rho = %f\n", i, rho);
	printf("\nminrho was %f\n", minRho);
	
	// copy solution from device to host
	checkCudaErrors( cudaMemcpy( h_x, d_x, dim * sizeof( float ), cudaMemcpyDeviceToHost ) );

	// release device memory
	checkCudaErrors( cudaFree( d_A ) );
	checkCudaErrors( cudaFree( d_b ) );
	checkCudaErrors( cudaFree( d_x ) );
	checkCudaErrors( cudaFree( d_r ) );
	checkCudaErrors( cudaFree( d_p ) );
	checkCudaErrors( cudaFree( d_q ) );
	checkCudaErrors( cudaFree( d_tmp ) );
}
