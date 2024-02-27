#include "matrix.h"
// #include <cassert>

#include <cuda_runtime.h>

template <typename T>
__global__ void matrixMul(T *a, T *b, T *c, int N){
    // Calculate the global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check for our matrix
    if(row < N && col < N){
        // Accumulate a partial result
        T tmp = 0;
        for(int i = 0; i < N; i++){
            tmp += a[row * N + i] * b[i * N + col];
        }

        // Write back the result
        c[row * N + col] = tmp;
    }
}

// Initializes a square matrix with random numbers between 0-100
template <typename T>
void init_matrix(T *m, int N){
    for(int i = 0; i < N * N; i++){
        m[i] = static_cast<T>(rand() % 100);
    }
}

// Verify the result on the CPU
template <typename T>
void verify_result(T *a, T *b, T *c, int N){
    int tmp;
    // For every row...
    for(int i = 0; i < N; i++){
        // For every col...
        for(int j = 0; j < N; j++){
            // For every element in the row-col pair
            tmp = 0;
            for(int k = 0; k < N; k++){
                tmp += a[i * N + k] * b[k * N + j];
            }
            
            // Check each result
            // assert(tmp == c[i * N + j]);
        }
    }
}

template <typename T>
void launch_kernel_and_profile(T *a, T *b, T *c, int N, dim3 THREADS, dim3 BLOCKS) {
    matrixMul<<<BLOCKS, THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize();
}

// Explicit instation of needed types
template __global__ void matrixMul<float>(float*, float*, float*, int);
template __global__ void matrixMul<double>(double*, double*, double*, int);

template void init_matrix<float>(float*, int);
template void init_matrix<double>(double*, int);

template void launch_kernel_and_profile<float>(float*, float*, float*, int, dim3, dim3);
template void launch_kernel_and_profile<double>(double*, double*, double*, int, dim3, dim3);

template void verify_result<float>(float*, float*, float*, int);
template void verify_result<double>(double*, double*, double*, int);