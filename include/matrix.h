#ifndef MATRIX_H
#define MATRIX_H

#include <cuda_runtime.h>

template <typename T>
__global__ void matrixMul(T *a, T *b, T *c, int N);

template <typename T>
__global__ void relu(T *input, T *output, int  N);

template <typename T>
void init_matrix(T *m, int N);

template <typename T>
void verify_multiply_result(T *a, T *b, T *c, int N);

template <typename T>
void verify_relu_result(T *a, T *b, T *c, int N);

template <typename T>
void launch_kernel_and_profile(T *a, T *b, T *c, int N, dim3 THREADS, dim3 BLOCKS);

#endif // MATRIX_H
