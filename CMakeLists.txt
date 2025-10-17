cmake_minimum_required(VERSION 3.28)
project(matrix-multiplication-on-crack LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)

find_package(CUDAToolkit REQUIRED)

add_executable(matmul_benchmark
    src/main.cpp
)

target_include_directories(matmul_benchmark PRIVATE include)
target_link_libraries(matmul_benchmark CUDA::cudart)
