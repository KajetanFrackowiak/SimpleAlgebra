#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "matrix.h"


#ifdef __CUDACC__
// CUDA kernel for matrix multiplication
template <typename T>
__global__ void matrixMulKernel(const T* A, const T* B, T* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        T value = 0.0; // Use T for the data type
        for (int k = 0; k < colsA; ++k) {
            value += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = value;
    }
}


// CUDA kernel for ReLU activation
__global__ void reluKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

#endif  // __CUDACC__

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(rows * cols) {}

template <typename T>
Matrix<T>::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {}

template <typename T>
Matrix<T>::~Matrix() {}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
    Matrix<T> result(rows, other.cols);
    T *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void**)&d_A, rows * cols * sizeof(T));
    cudaMalloc((void**)&d_B, other.rows * other.cols * sizeof(T));
    cudaMalloc((void**)&d_C, rows * other.cols * sizeof(T));

    // Check for allocation errors
    if (d_A == nullptr || d_B == nullptr || d_C == nullptr) {
        // Handle memory allocation failure
        throw std::runtime_error("CUDA memory allocation failed");
    }

    // Copy data to device
    cudaMemcpy(d_A, data.data(), rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, other.data.data(), other.rows * other.cols * sizeof(T), cudaMemcpyHostToDevice);

    // CUDA kernel launch
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((result.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (result.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Check for kernel launch errors
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols, other.cols);
    cudaDeviceSynchronize();  // Wait for the kernel to finish

    // Check for kernel execution errors
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        // Handle kernel execution failure
        throw std::runtime_error("CUDA kernel execution failed");
    }

    // Copy result back to host
    cudaMemcpy(result.data.data(), d_C, rows * other.cols * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const std::vector<std::vector<T>>& values) {
    if (values.size() == 0 || values[0].size() == 0) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }

    rows = values.size();
    cols = values[0].size();
    data = values;

    return *this;
}


template <typename T>
void Matrix<T>::relu() {
    float* d_data;
    cudaMalloc((void**)&d_data, data.size() * sizeof(float));
    cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((data.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);

    reluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, data.size());

    cudaMemcpy(data.data(), d_data, data.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
}


template class Matrix<float>;
template class Matrix<double>;

// int main() {
//     // Create two matrices
//     Matrix<float> matrix1(2, 3);
//     Matrix<float> matrix2(3, 4);

//     matrix1 = {{1.0f, 2.0f, 3.0f},
//                {4.0f, 5.0f, 6.0f}};

//     matrix2 = {{7.0f, 8.0f, 9.0f, 10.0f},
//                {11.0f, 12.0f, 13.0f, 14.0f},
//                {15.0f, 16.0f, 17.0f, 18.0f}};

//     // Multiply the matrices
//     Matrix<float> result = matrix1 * matrix2;

//     // Print the result
//     // std::cout << "Matrix 1:\n" << matrix1 << "\n";
//     // std::cout << "Matrix 2:\n" << matrix2 << "\n";
//     // std::cout << "Result:\n" << result << "\n";

//     return 0;
// }
