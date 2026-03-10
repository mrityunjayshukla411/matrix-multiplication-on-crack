/**
 * @file MatrixInitializer.h
 * @brief Host-side matrix initialization patterns.
 *
 * Fills the host buffer (m_h_data) of a Matrix<T> according to a chosen
 * pattern. After initialization, the caller must cudaMemcpy the data to
 * the device before launching any GPU kernel.
 */
#pragma once
#include "Matrix.h"
#include <random>
#include <algorithm>

/**
 * @brief Static utility class for filling Matrix host buffers.
 *
 * @tparam T Element type (float, double, int).
 */
template <typename T>
class MatrixInitializer
{
public:
    /**
     * @brief Initialization patterns available.
     *
     * - ZEROS:          Every element set to 0.
     * - ONES:           Every element set to 1.
     * - RANDOM_UNIFORM: Uniform random in [param1, param2] (default [0, 1]).
     * - SEQUENTIAL:     Element at linear index i = static_cast<T>(i).
     */
    enum class Pattern
    {
        ZEROS,
        ONES,
        RANDOM_UNIFORM,
        SEQUENTIAL
    };

    /**
     * @brief Initialize the host buffer of @p matrix with a given pattern.
     *
     * Only m_h_data is written; device memory is not touched.
     *
     * @param matrix  Matrix whose host buffer will be filled.
     * @param pattern Initialization pattern to apply.
     * @param param1  Lower bound for RANDOM_UNIFORM (ignored otherwise). Default 0.
     * @param param2  Upper bound for RANDOM_UNIFORM (ignored otherwise). Default 1.
     *
     * @throws std::invalid_argument If an unknown pattern is provided.
     */
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
            // Use a non-deterministic seed so each run produces different data.
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
