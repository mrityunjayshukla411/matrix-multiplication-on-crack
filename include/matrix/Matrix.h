/**
 * @file Matrix.h
 * @brief Dual-buffer matrix container with pinned host memory and device memory.
 *
 * The Matrix<T> class manages a pair of memory regions:
 *   - m_h_data: page-locked (pinned) host memory via cudaMallocHost, enabling
 *     DMA transfers at full PCIe bandwidth.
 *   - m_d_data: device global memory via cudaMalloc.
 *
 * Matrices are stored in row-major order. Element (i, j) lives at offset
 * i*m_cols + j. Copy semantics are deleted to prevent accidental deep copies;
 * use move semantics to transfer ownership.
 */
#pragma once
#include <cstddef>
#include <memory>
#include "utils/CudaUtils.h"

/**
 * @brief Row-major matrix with paired host (pinned) and device buffers.
 *
 * @tparam T Element type (float, double, int).
 */
template <typename T>
class Matrix
{
public:
    size_t m_rows, m_cols; ///< Matrix dimensions.
    T *m_h_data;           ///< Pinned host buffer (page-locked for fast PCIe DMA).
    T *m_d_data;           ///< Device global memory buffer.

    /**
     * @brief Allocates pinned host and device memory for a rows×cols matrix.
     *
     * Both buffers are left uninitialized. Call MatrixInitializer::initialize()
     * to populate host data, then cudaMemcpy to transfer it to the device before
     * launching any GPU kernel.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_h_data(nullptr), m_d_data(nullptr)
    {
        CUDA_CHECK(cudaMallocHost(&m_h_data, bytes()));
        CUDA_CHECK(cudaMalloc(&m_d_data, bytes()));
    }

    /**
     * @brief Frees both pinned host and device memory, ignoring nullptr safely.
     */
    ~Matrix()
    {
        if (m_h_data)
            cudaFreeHost(m_h_data);
        if (m_d_data)
            cudaFree(m_d_data);
    }

    // Delete copy operations (prevent accidental copies)
    Matrix(const Matrix &) = delete;
    Matrix &operator=(const Matrix &) = delete;

    // Move operations (allow transfer of ownership)
    Matrix(Matrix &&other) noexcept
        : m_rows(other.m_rows), m_cols(other.m_cols),
          m_h_data(other.m_h_data), m_d_data(other.m_d_data)
    {
        // Null out the moved-from object so its destructor is a no-op.
        other.m_h_data = nullptr;
        other.m_d_data = nullptr;
    }

    /**
     * @brief Total size of the matrix in bytes.
     * @return m_rows * m_cols * sizeof(T)
     */
    size_t bytes() const
    {
        return m_rows * m_cols * sizeof(T);
    }

    /**
     * @brief Total number of elements in the matrix.
     * @return m_rows * m_cols
     */
    size_t size() const
    {
        return m_rows * m_cols;
    }
};
