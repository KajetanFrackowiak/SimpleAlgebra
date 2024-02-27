// #include "matrix.h"
// #include <cstdlib>
// #include <iostream>


// int main(){
//     // Set our square matrix dimension (2^10 x 2^10 default) 
//     int N = 1 << 10;
//     size_t bytes = N * N * sizeof(float);

//     // Allocate memory for our matrices
//     int *a, *b, *c;
//     cudaMallocManaged(&a, bytes);
//     cudaMallocManaged(&b, bytes);
//     cudaMallocManaged(&c, bytes);

//     // Initialize our matrices
//     init_matrix(a, N);
//     init_matrix(b, N);

//     // Set our CTA and Grid dimensions
//     int threads = 16;
//     int blocks = (N + threads - 1) / threads;

//     // Setup our kernel launch parameters
//     dim3 THREADS(threads, threads);
//     dim3 BLOCKS(blocks, blocks);

//     launch_kernel_and_profile(a, b, c, N, THREADS, BLOCKS);
//     cudaDeviceSynchronize();

//     // Verify the result
//     verify_result(a, b, c, N);

//     std::cout << "PROGRAM COMPLETED SUCCESSFULLY!" << std::endl;
    
//     // Free allocated memory
//     cudaFree(a);
//     cudaFree(b);
//     cudaFree(c);

//     return 0;
// }