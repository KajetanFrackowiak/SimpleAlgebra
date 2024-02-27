#include <gtest/gtest.h>
#include "matrix.h"

TEST(MatrixMultiplicationTest, FloatMatrixMultiplication) {
    const int N = 10;  // Adjust the size as needed

    // Allocate memory for matrices
    float *a, *b, *c;
    cudaMallocManaged(&a, N * N * sizeof(float));
    cudaMallocManaged(&b, N * N * sizeof(float));
    cudaMallocManaged(&c, N * N * sizeof(float));

    // Initialize matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // Set up kernel launch parameters
    dim3 THREADS(16, 16);
    dim3 BLOCKS((N + THREADS.x - 1) / THREADS.x, (N + THREADS.y - 1) / THREADS.y);

    // Launch kernel and synchronize
    launch_kernel_and_profile(a, b, c, N, THREADS, BLOCKS);
    cudaDeviceSynchronize();

    // Verify the result
    verify_result(a, b, c, N);

    // Free allocated memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

TEST(MatrixMultiplicationTest, DoubleMatrixMultiplication) {
    const int N = 10;  // Adjust the size as needed

    // Allocate memory for matrices
    double *a, *b, *c;
    cudaMallocManaged(&a, N * N * sizeof(double));
    cudaMallocManaged(&b, N * N * sizeof(double));
    cudaMallocManaged(&c, N * N * sizeof(double));

    // Initialize matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // Set up kernel launch parameters
    dim3 THREADS(16, 16);
    dim3 BLOCKS((N + THREADS.x - 1) / THREADS.x, (N + THREADS.y - 1) / THREADS.y);

    // Launch kernel and synchronize
    launch_kernel_and_profile(a, b, c, N, THREADS, BLOCKS);
    cudaDeviceSynchronize();

    // Verify the result
    verify_result(a, b, c, N);

    // Free allocated memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}