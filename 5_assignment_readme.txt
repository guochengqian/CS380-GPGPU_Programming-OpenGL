=====================================================================
CS380 GPU and GPGPU Programming, KAUST
Programming Assignment #5
Matrix Vector Multiplication, Reduction and Image De-blurring

Contacts: 
peter.rautek@kaust.edu.sa
=====================================================================

Task: 

1. Conjugate Gradient Linear Systems Solver
The conjugate gradient method [1] iteratively solves the linear system
Ax = b

The skeleton of the algorithm is already provided. 
You have to program the matrix and vector operations (matrix-vector multiplication, reduction, ...) that are executed on the GPU.
Find the TODOs in the code and implement them. 
For testing a small hard-coded 4x4 matrix, and .txt files containing sparse matrices are provided. 

Measure the performance of the method.
Print out the residual of the method after each iteration to see how it converges. 

You must use shared memory for the vector in the matrix-vector-multiplication!
For the vector reduction you must use shared memory as well, as discussed in the lecture!

2. Image De-blurring
Once the Conjugate Gradient method is working for the matrices it can be used to solve practical applications.
Implement a simple image de-blurring application that works for a known blurring filter.
An image is loaded and blurred with a filter kernel.
We can de-blur the image again by solving the equation 
Ax = b
were x is the unknown input image, that was filtered with filter operation A such that the result is the known blurred image b.

In order to formulate the blurring operation (convolution) as a matrix multiplication, 
b and x are represented as vectors and each row of matrix A is one filter kernel for the whole image x.
This naive method is very memory inefficient (O(N^2) where N is the number of pixels). 
Hint: Because of numerical inaccuracies for the very large matrices that occur you will have to increase the error tolerance for the CG-method to terminate. 


BONUS: 
For bonus points implement a method that is more efficient in terms of memory and/or performance.
For instance you can modify the code to divide the image into small patches and solve for them separately.
Alternatively, you can implement a sparse matrix format.

References:
[1] http://en.wikipedia.org/wiki/Conjugate_gradient_method