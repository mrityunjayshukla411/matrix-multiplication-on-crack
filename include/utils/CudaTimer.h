/**
 * @file CudaTimer.h
 * @brief RAII wrapper around CUDA events for accurate GPU kernel timing.
 *
 * CudaTimer uses cudaEvent_t to measure elapsed time on the GPU command stream.
 * Unlike wall-clock timers, CUDA events capture timestamps directly in the GPU
 * timeline, giving accurate measurements even when the CPU and GPU run asynchronously.
 *
 * Typical usage:
 * @code
 *   CudaTimer timer;
 *   timer.start();
 *   myKernel<<<grid, block>>>(...);
 *   float ms = timer.stop();  // blocks CPU until kernel completes
 * @endcode
 */
#pragma once
#include <cuda_runtime.h>
#include "utils/CudaUtils.h"

/**
 * @brief RAII CUDA event timer. Creates events on construction, destroys on destruction.
 */
class CudaTimer {
    private:
    cudaEvent_t m_start_event; ///< CUDA event marking the start of the timed region.
    cudaEvent_t m_stop_event;  ///< CUDA event marking the end of the timed region.

    public:
    /**
     * @brief Creates start and stop CUDA events.
     */
    CudaTimer()
    {
        CUDA_CHECK(cudaEventCreate(&m_start_event));
        CUDA_CHECK(cudaEventCreate(&m_stop_event));
    }

    /**
     * @brief Destroys CUDA events, releasing driver resources.
     */
    ~CudaTimer()
    {
        CUDA_CHECK(cudaEventDestroy(m_start_event));
        CUDA_CHECK(cudaEventDestroy(m_stop_event));
    }

    /**
     * @brief Records the start event into the default CUDA stream.
     *
     * Must be called before any kernel whose execution should be timed.
     */
    void start()
    {
        CUDA_CHECK(cudaEventRecord(m_start_event));
    }

    /**
     * @brief Records the stop event, synchronizes, and returns elapsed time.
     *
     * Blocks the calling CPU thread until the stop event has been processed
     * by the GPU (i.e., all preceding GPU work is complete).
     *
     * @return Elapsed time in milliseconds between start() and stop().
     */
    float stop()
    {
        CUDA_CHECK(cudaEventRecord(m_stop_event));
        CUDA_CHECK(cudaEventSynchronize(m_stop_event));
        float elapsedMs;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, m_start_event, m_stop_event));
        return elapsedMs;
    }
};
