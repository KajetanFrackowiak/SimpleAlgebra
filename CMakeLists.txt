cmake_minimum_required(VERSION 3.22)
project(matrixTest LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_COMPILER "g++")

# Include directories
include_directories(googletest/googletest/include)
include_directories(include)

# Add main executable
add_executable(${PROJECT_NAME} src/matrix.cu test/matrixTest.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/googletest/build/lib/libgtest.a
    ${CMAKE_SOURCE_DIR}/googletest/build/lib/libgtest_main.a)

