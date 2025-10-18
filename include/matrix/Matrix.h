#pragma once
#include <cstddef>
#include <memory>
#include "utils/CudaUtils.h"

template <typename T>
class Matrix
{
public:
    size_t m_rows, m_cols;
    T *m_h_data; 
    T *m_d_data;

    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_h_data(nullptr), m_d_data(nullptr)
    {
        CUDA_CHECK(cudaMallocHost(&m_h_data, bytes()));
        CUDA_CHECK(cudaMalloc(&m_d_data, bytes()));
    }

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
        other.m_h_data = nullptr;
        other.m_d_data = nullptr;
    }
    size_t bytes() const
    {
        return m_rows * m_cols * sizeof(T);
    }

    size_t size() const
    {
        return m_rows * m_cols;
    }
};