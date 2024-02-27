#ifndef MATRIX_H
#define MATRIX_H

#include <cuda_runtime.h>

template <typename T>
__global__ void matrixMul(T *a, T *b, T *c, int N);

template <typename T>
void init_matrix(T *m, int N);

template <typename T>
void verify_result(T *a, T *b, T *c, int N);

template <typename T>
void launch_kernel_and_profile(T *a, T *b, T *c, int N, dim3 THREADS, dim3 BLOCKS);

#endif // MATRIX_H
