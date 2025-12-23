cmake_minimum_required(VERSION 3.28)
project(matrix-multiplication-on-crack LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)

find_package(CUDAToolkit REQUIRED)

add_executable(matmul_benchmark
    src/main.cpp
    src/kernels/CoalescedKernel.cu
    src/kernels/UncoalescedKernel.cu
    src/kernels/SharedMemCachingKernel.cu
    src/kernels/Tiling1DKernel.cu
)

target_compile_options(matmul_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall>
)
target_include_directories(matmul_benchmark PRIVATE include)
target_link_libraries(matmul_benchmark CUDA::cudart)

# Test executable
add_executable(matmul_test
    src/test.cpp
    src/kernels/CoalescedKernel.cu
    src/kernels/UncoalescedKernel.cu
    src/kernels/SharedMemCachingKernel.cu
    src/kernels/Tiling1DKernel.cu
)

target_compile_options(matmul_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall>
)
target_include_directories(matmul_test PRIVATE include)
target_link_libraries(matmul_test CUDA::cudart)
