#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Error checking macro for CUDA calls
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

// Device properties query helper

inline void printDeviceInfo()
{
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties_v2(&prop, i));

        std::cout << "Device" << i << ":" << prop.name << "\n";
        std::cout << "Compute Capability:" << prop.major << "." << prop.minor << "\n";
        std::cout << "Global Memory:" << prop.totalGlobalMem / (1024 * 1024) << "MB\n";
        std::cout << "Shared Memory per Block" << prop.sharedMemPerBlock / (1024) << "\n";
        std::cout << "Max Threads per Block" << prop.maxThreadsPerBlock << "\n\n";
    }
}   