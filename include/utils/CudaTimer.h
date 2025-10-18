#pragma once
#include<cuda_runtime.h>
#include <CudaUtils.h>

class CudaTimer {
    private:
    cudaEvent_t m_start_event, m_stop_event;

    public:
    CudaTimer()
    {
        CUDA_CHECK(cudaEventCreate(&m_start_event));
        CUDA_CHECK(cudaEventCreate(&m_stop_event));
    }
    ~CudaTimer()
    {
        CUDA_CHECK(cudaEventDestroy(m_start_event));
        CUDA_CHECK(cudaEventDestroy(m_stop_event));
    }

    void start()
    {
        CUDA_CHECK(cudaEventRecord(m_start_event));
    }

    float stop()
    {
        CUDA_CHECK(cudaEventRecord(m_stop_event));
        CUDA_CHECK(cudaEventSynchronize(m_stop_event));
        float elapsedMs;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs,m_start_event,m_stop_event));
        return elapsedMs;
    }
};