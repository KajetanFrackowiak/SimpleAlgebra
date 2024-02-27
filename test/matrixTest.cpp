#include <gtest/gtest.h>
#include "matrix.h"

TEST(MatrixMultiplicationTest, FloatMatrixMultiplication) {
    const int N = 10;

    float *a, *b, *c;
    cudaMallocManaged(&a, N * N * sizeof(float));
    cudaMallocManaged(&b, N * N * sizeof(float));
    cudaMallocManaged(&c, N * N * sizeof(float));

    init_matrix(a, N);
    init_matrix(b, N);

    dim3 THREADS(16, 16);
    dim3 BLOCKS((N + THREADS.x - 1) / THREADS.x, (N + THREADS.y - 1) / THREADS.y);
    launch_kernel_and_profile(a, b, c, N, THREADS, BLOCKS);
    cudaDeviceSynchronize();

    verify_multiply_result(a, b, c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

TEST(MatrixMultiplicationTest, DoubleMatrixMultiplication) {
    const int N = 10;

    double *a, *b, *c;
    cudaMallocManaged(&a, N * N * sizeof(double));
    cudaMallocManaged(&b, N * N * sizeof(double));
    cudaMallocManaged(&c, N * N * sizeof(double));

    init_matrix(a, N);
    init_matrix(b, N);

    dim3 THREADS(16, 16);
    dim3 BLOCKS((N + THREADS.x - 1) / THREADS.x, (N + THREADS.y - 1) / THREADS.y);

    launch_kernel_and_profile(a, b, c, N, THREADS, BLOCKS);
    cudaDeviceSynchronize();

    verify_multiply_result(a, b, c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

TEST(MatrixOperationsTest, FloatReLUOperation) {
    const int N = 10;

    float *a, *b;
    cudaMallocManaged(&a, N * N * sizeof(float));
    cudaMallocManaged(&b, N * N * sizeof(float));

    init_matrix(a, N);

    dim3 THREADS(16, 16);
    dim3 BLOCKS((N + THREADS.x - 1) / THREADS.x, (N + THREADS.y - 1) / THREADS.y);

    relu(a, b, N * N);
    cudaDeviceSynchronize();

    verify_relu_result(a, a, b, N);

    cudaFree(a);
    cudaFree(b);
}

TEST(MatrixOperationsTest, DoubleReLUOperation) {
    const int N = 10;

    double *a, *b;
    cudaMallocManaged(&a, N * N * sizeof(double));
    cudaMallocManaged(&b, N * N * sizeof(double));

    init_matrix(a, N);

    dim3 THREADS(16, 16);
    dim3 BLOCKS((N + THREADS.x - 1) / THREADS.x, (N + THREADS.y - 1) / THREADS.y);

    relu(a, b, N * N);
    cudaDeviceSynchronize();

    verify_relu_result(a, a, b, N);

    cudaFree(a);
    cudaFree(b);
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}