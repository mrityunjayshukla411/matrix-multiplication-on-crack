#pragma once
#include "Matrix.h"
#include <random>
#include <algorithm>

template <typename T>
class MatrixInitializer
{
public:
    enum class Pattern
    {
        ZEROS,
        ONES,
        RANDOM_UNIFORM,
        SEQUENTIAL
    };
    static void initialize(Matrix<T> &matrix, Pattern pattern, T param1 = T(0), T param2 = T(1))
    {
        size_t total_elements = matrix.size();
        T *host_data = matrix.m_h_data;

        switch (pattern)
        {
        case Pattern::ZEROS:
            std::fill(host_data, host_data + total_elements, T(0));
            break;
        case Pattern::ONES:
            std::fill(host_data, host_data + total_elements, T(1));
            break;
        case Pattern::RANDOM_UNIFORM:
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dis(param1, param2);
            for (size_t i = 0; i < total_elements; ++i)
            {
                // Sanity check typecasting
                host_data[i] = static_cast<T>(dis(gen));
            }
            break;
        }
        case Pattern::SEQUENTIAL:
            for (size_t i = 0; i < total_elements; ++i)
            {
                host_data[i] = static_cast<T>(i);
            }
            break;
        default:
            throw std::invalid_argument("Unknown initialization pattern");
        }
    }
};