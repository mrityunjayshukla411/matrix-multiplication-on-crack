/**
 * @file CudaUtils.h
 * @brief CUDA error-checking macro and device information helper.
 *
 * Include this header in every translation unit that makes CUDA API calls.
 * The CUDA_CHECK macro wraps any cudaXxx() call and terminates the process
 * with a descriptive message on failure, making bugs easy to diagnose.
 */
#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

/**
 * @brief Wraps a CUDA API call and aborts on error.
 *
 * Prints the CUDA error string, source file, and line number to stderr
 * before calling exit(EXIT_FAILURE). Should be used around every CUDA
 * runtime call outside of performance-critical inner loops.
 *
 * Example:
 * @code
 *   CUDA_CHECK(cudaMalloc(&ptr, size));
 *   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
 * @endcode
 */
#define CUDA_CHECK(call)                                                   \
    do                                                                     \
    {                                                                      \
        cudaError_t error = call;                                          \
        if (error != cudaSuccess)                                          \
        {                                                                  \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error)       \
                      << "at" << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/**
 * @brief Queries and prints properties of all available CUDA devices.
 *
 * Iterates over all devices visible to the process and prints:
 *   - Device name and compute capability
 *   - Total global memory (MB)
 *   - Shared memory per block (KB)
 *   - Maximum threads per block and maximum blocks per SM
 *
 * Called at the start of both matmul_benchmark and matmul_test to record
 * the hardware context alongside benchmark results.
 */
inline void printDeviceInfo()
{
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties_v2(&prop, i));

        std::cout << "Device" << i << ": " << prop.name << "\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << "MB\n";
        std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / (1024) << "\n";
        std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Max Blocks per SM: " << prop.maxBlocksPerMultiProcessor << "\n\n";
    }
}
