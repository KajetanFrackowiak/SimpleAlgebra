cmake_minimum_required(VERSION 3.22)
project(matrixTest LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_COMPILER "g++")

# Include directories
include_directories(googletest/googletest/include)
include_directories(include)

# Enable debugging information for C++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Enable debugging information for CUDA
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")


# Add executables
add_executable(${PROJECT_NAME} src/matrix.cu test/matrixTest.cpp)
# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/googletest/build/lib/libgtest.a
    ${CMAKE_SOURCE_DIR}/googletest/build/lib/libgtest_main.a)

# Enable CUDA separable compilation
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)